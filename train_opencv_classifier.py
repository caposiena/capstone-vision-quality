from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import cv2
import joblib
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

from opencv_features import (
    FEATURE_VERSION,
    build_product_mask,
    crop_object,
    extract_features,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ML_CLASSES = ("buono", "cotto", "poco", "rotto")


def load_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    if isinstance(names, list):
        return [str(x) for x in names]
    raise ValueError("data.yaml non contiene 'names' valido")


def find_image_for_label(label_path: Path) -> Path | None:
    candidates = []

    if "labels" in label_path.parts:
        parts = list(label_path.parts)
        idx = parts.index("labels")
        parts[idx] = "images"
        base = Path(*parts).with_suffix("")
        candidates.extend(base.with_suffix(ext) for ext in IMAGE_EXTENSIONS)

    stem = label_path.stem
    dataset_root = label_path
    while (
        dataset_root.parent != dataset_root
        and dataset_root.name not in ("labels", "images")
    ):
        dataset_root = dataset_root.parent

    root = (
        dataset_root.parent
        if dataset_root.name in ("labels", "images")
        else label_path.parent
    )

    for ext in IMAGE_EXTENSIONS:
        candidates.extend(root.rglob(stem + ext))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def yolo_bbox(
    image: np.ndarray,
    values: list[float],
) -> tuple[int, int, int, int]:
    _class_id, xc, yc, bw, bh = values
    h, w = image.shape[:2]

    x1 = int((xc - bw / 2.0) * w)
    y1 = int((yc - bh / 2.0) * h)
    x2 = int((xc + bw / 2.0) * w)
    y2 = int((yc + bh / 2.0) * h)

    return (
        max(0, x1),
        max(0, y1),
        min(w, x2),
        min(h, y2),
    )


def iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter

    return float(inter) / float(max(1, union))


def runtime_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = build_product_mask(image)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    boxes = []
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 120:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + w, y + h))

    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "taralli_yolo_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "models"
        / "opencv_extratrees.joblib",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    data_yaml = dataset / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError("Manca data.yaml: %s" % data_yaml)

    source_names = load_names(data_yaml)
    class_to_model = {name: idx for idx, name in enumerate(ML_CLASSES)}
    labels = sorted(dataset.rglob("*.txt"))

    samples = []
    targets = []
    source_images = []

    unmatched = 0
    skipped_blob = 0

    for label_path in labels:
        image_path = find_image_for_label(label_path)
        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        boxes_runtime = runtime_boxes(image)

        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            values = [float(x) for x in parts[:5]]
            source_class_id = int(values[0])
            if source_class_id < 0 or source_class_id >= len(source_names):
                continue

            class_name = source_names[source_class_id]
            if class_name == "blob":
                skipped_blob += 1
                continue
            if class_name not in class_to_model:
                continue

            gt_box = yolo_bbox(image, values)
            if not boxes_runtime:
                unmatched += 1
                continue

            best_box = max(boxes_runtime, key=lambda box: iou(gt_box, box))
            best_iou = iou(gt_box, best_box)
            if best_iou < 0.50:
                unmatched += 1
                continue

            crop = crop_object(image, best_box)
            if crop.size == 0:
                unmatched += 1
                continue

            samples.append(extract_features(crop)[0])
            targets.append(class_to_model[class_name])
            source_images.append(image_path.name)

    if not samples:
        raise RuntimeError("Nessun campione coerente estratto dal dataset")

    x = np.asarray(samples, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int32)

    train_idx = []
    test_idx = []

    for i, image_name in enumerate(source_images):
        bucket = zlib.crc32(image_name.encode("utf-8")) % 5
        if bucket == 0:
            test_idx.append(i)
        else:
            train_idx.append(i)

    # Se una classe manca dal test, sposta un'INTERA immagine contenente
    # quella classe, mantenendo l'isolamento per immagine.
    for class_id in sorted(set(y.tolist())):
        if any(int(y[i]) == class_id for i in test_idx):
            continue

        candidate_image = next(
            (
                source_images[i]
                for i in train_idx
                if int(y[i]) == class_id
            ),
            None,
        )
        if candidate_image is None:
            continue

        moved = [i for i in train_idx if source_images[i] == candidate_image]
        train_idx = [i for i in train_idx if source_images[i] != candidate_image]
        test_idx.extend(moved)

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    pred = model.predict(x_test).astype(np.int32)
    accuracy = float(np.mean(pred == y_test)) if len(y_test) else 0.0

    confusion = confusion_matrix(
        y_test,
        pred,
        labels=list(range(len(ML_CLASSES))),
    ).astype(np.int32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)

    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "feature_version": FEATURE_VERSION,
                "feature_count": int(x.shape[1]),
                "class_names": list(ML_CLASSES),
                "samples": int(len(y)),
                "train_samples": int(len(y_train)),
                "test_samples": int(len(y_test)),
                "validation_accuracy": accuracy,
                "confusion_matrix": confusion.tolist(),
                "runtime_unmatched": unmatched,
                "blob_labels_excluded": skipped_blob,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Campioni ML:", len(y))
    print("Classi ML:", list(ML_CLASSES))
    print("Feature:", x.shape[1])
    print("YOLO non abbinati alla segmentazione runtime:", unmatched)
    print("Blob esclusi dal ML:", skipped_blob)
    print("Accuracy validation: %.2f%%" % (accuracy * 100.0))
    print("Confusion matrix:")
    print(confusion)
    print("Modello:", args.output)
    print("Metadata:", meta_path)


if __name__ == "__main__":
    main()
