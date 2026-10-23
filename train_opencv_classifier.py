from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import yaml

from opencv_features import FEATURE_VERSION, extract_features


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    if isinstance(names, list):
        return [str(x) for x in names]
    raise ValueError("data.yaml non contiene 'names' valido")


def find_image_for_label(label_path: Path) -> Path | None:
    text = str(label_path)
    candidates = []

    if "labels" in label_path.parts:
        parts = list(label_path.parts)
        idx = parts.index("labels")
        parts[idx] = "images"
        base = Path(*parts).with_suffix("")
        candidates.extend(base.with_suffix(ext) for ext in IMAGE_EXTENSIONS)

    # Fallback nello stesso dataset.
    stem = label_path.stem
    dataset_root = label_path
    while dataset_root.parent != dataset_root and dataset_root.name not in ("labels", "images"):
        dataset_root = dataset_root.parent
    root = dataset_root.parent if dataset_root.name in ("labels", "images") else label_path.parent

    for ext in IMAGE_EXTENSIONS:
        candidates.extend(root.rglob(stem + ext))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def crop_from_yolo(image: np.ndarray, values: list[float]) -> np.ndarray:
    _class_id, xc, yc, bw, bh = values
    h, w = image.shape[:2]

    x1 = int((xc - bw / 2.0) * w)
    y1 = int((yc - bh / 2.0) * h)
    x2 = int((xc + bw / 2.0) * w)
    y2 = int((yc + bh / 2.0) * h)

    pad_x = max(2, int((x2 - x1) * 0.06))
    pad_y = max(2, int((y2 - y1) * 0.06))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return image[y1:y2, x1:x2]


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
        default=Path(__file__).resolve().parent / "models" / "opencv_rtrees.yml",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    data_yaml = dataset / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError("Manca data.yaml: %s" % data_yaml)

    names = load_names(data_yaml)
    labels = sorted(dataset.rglob("*.txt"))

    samples = []
    targets = []
    source_keys = []

    for label_path in labels:
        image_path = find_image_for_label(label_path)
        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        for row_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            values = [float(x) for x in parts[:5]]
            class_id = int(values[0])
            if class_id < 0 or class_id >= len(names):
                continue

            crop = crop_from_yolo(image, values)
            if crop.size == 0:
                continue

            samples.append(extract_features(crop)[0])
            targets.append(class_id)
            source_keys.append("%s:%d" % (image_path.name, row_index))

    if not samples:
        raise RuntimeError("Nessun campione estratto dal dataset")

    x = np.asarray(samples, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int32)

    # Holdout per IMMAGINE: tutti gli oggetti della stessa foto finiscono
    # nello stesso split. Evita una validazione artificialmente ottimistica.
    train_idx = []
    test_idx = []

    for i, key in enumerate(source_keys):
        image_key = key.split(":", 1)[0]
        bucket = zlib.crc32(image_key.encode("utf-8")) % 5
        if bucket == 0:
            test_idx.append(i)
        else:
            train_idx.append(i)

    # Garantisce che tutte le classi siano presenti nel test; se una manca,
    # sposta un campione di quella classe dal train.
    for class_id in sorted(set(y.tolist())):
        if not any(int(y[i]) == class_id for i in test_idx):
            candidate = next(
                (i for i in train_idx if int(y[i]) == class_id),
                None,
            )
            if candidate is not None:
                train_idx.remove(candidate)
                test_idx.append(candidate)

    x_train_raw = x[train_idx]
    y_train_raw = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    # Bilanciamento semplice: replica le classi minoritarie fino alla numerosita
    # della classe piu rappresentata. Utile soprattutto per blob/rotto.
    class_counts = {
        class_id: int(np.sum(y_train_raw == class_id))
        for class_id in sorted(set(y_train_raw.tolist()))
    }
    target_count = max(class_counts.values())

    balanced_x = []
    balanced_y = []

    for class_id, count in class_counts.items():
        class_x = x_train_raw[y_train_raw == class_id]
        repeats = target_count // count
        remainder = target_count % count

        if repeats > 0:
            balanced_x.append(np.tile(class_x, (repeats, 1)))
            balanced_y.append(
                np.full((count * repeats,), class_id, dtype=np.int32)
            )

        if remainder > 0:
            balanced_x.append(class_x[:remainder])
            balanced_y.append(
                np.full((remainder,), class_id, dtype=np.int32)
            )

    x_train = np.vstack(balanced_x).astype(np.float32)
    y_train = np.concatenate(balanced_y).astype(np.int32)

    if hasattr(cv2.ml, "RTrees_create"):
        model = cv2.ml.RTrees_create()
    elif hasattr(cv2.ml, "RTrees") and hasattr(cv2.ml.RTrees, "create"):
        model = cv2.ml.RTrees.create()
    else:
        raise RuntimeError(
            "Questa build OpenCV non espone RTrees. Disponibili in cv2.ml: %s"
            % ", ".join(sorted(name for name in dir(cv2.ml) if "Tree" in name or "Forest" in name))
        )
    model.setMaxDepth(14)
    model.setMinSampleCount(2)
    model.setRegressionAccuracy(0.0)
    model.setUseSurrogates(False)
    model.setMaxCategories(len(names))
    model.setCalculateVarImportance(True)
    model.setActiveVarCount(0)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 250, 0))

    if not model.train(x_train, cv2.ml.ROW_SAMPLE, y_train.astype(np.float32)):
        raise RuntimeError("Training RTrees fallito")

    _, predicted = model.predict(x_test)
    pred = predicted.reshape(-1).astype(np.int32)

    accuracy = float(np.mean(pred == y_test)) if len(y_test) else 0.0
    confusion = np.zeros((len(names), len(names)), dtype=np.int32)
    for truth, guess in zip(y_test, pred):
        confusion[int(truth), int(guess)] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))

    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "feature_version": FEATURE_VERSION,
                "feature_count": int(x.shape[1]),
                "class_names": names,
                "samples": int(len(y)),
                "train_samples_raw": int(len(y_train_raw)),
                "train_samples_balanced": int(len(y_train)),
                "test_samples": int(len(y_test)),
                "validation_accuracy": accuracy,
                "confusion_matrix": confusion.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Campioni:", len(y))
    print("Classi:", names)
    print("Feature:", x.shape[1])
    print("Accuracy validation: %.2f%%" % (accuracy * 100.0))
    print("Confusion matrix:")
    print(confusion)
    print("Modello:", args.output)
    print("Metadata:", meta_path)


if __name__ == "__main__":
    main()
