from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from opencv_features import (
    DEFAULT_SEGMENTATION,
    SEGMENTATION_CONFIG_PATH,
    build_product_mask,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    if isinstance(names, list):
        return [str(x) for x in names]
    raise ValueError("data.yaml non contiene names valido")


def find_image_for_label(label_path: Path) -> Path | None:
    if "labels" not in label_path.parts:
        return None

    parts = list(label_path.parts)
    idx = parts.index("labels")
    parts[idx] = "images"
    base = Path(*parts).with_suffix("")

    for ext in IMAGE_EXTENSIONS:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def yolo_bbox(
    image: np.ndarray,
    values: list[float],
) -> tuple[int, int, int, int]:
    _class_id, xc, yc, bw, bh = values
    h, w = image.shape[:2]

    return (
        max(0, int((xc - bw / 2.0) * w)),
        max(0, int((yc - bh / 2.0) * h)),
        min(w, int((xc + bw / 2.0) * w)),
        min(h, int((yc + bh / 2.0) * h)),
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

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / float(max(1, area_a + area_b - inter))


def runtime_boxes(
    image: np.ndarray,
    params: dict[str, int],
) -> list[tuple[int, int, int, int]]:
    mask = build_product_mask(image, params)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    boxes = []
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < int(params["min_component_area"]):
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + w, y + h))

    return boxes


def match_counts(
    gt_boxes: list[tuple[int, int, int, int]],
    pred_boxes: list[tuple[int, int, int, int]],
    threshold: float = 0.50,
) -> tuple[int, int, int, float]:
    pairs = []

    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            score = iou(gt, pred)
            if score >= threshold:
                pairs.append((score, gi, pi))

    pairs.sort(reverse=True)

    used_gt = set()
    used_pred = set()
    iou_sum = 0.0

    for score, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        iou_sum += score

    tp = len(used_gt)
    fn = len(gt_boxes) - tp
    fp = len(pred_boxes) - len(used_pred)

    return tp, fp, fn, iou_sum


def evaluate(
    samples: list[tuple[np.ndarray, list[tuple[int, int, int, int]]]],
    params: dict[str, int],
) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    iou_sum = 0.0

    for image, gt_boxes in samples:
        pred_boxes = runtime_boxes(image, params)
        c_tp, c_fp, c_fn, c_iou = match_counts(gt_boxes, pred_boxes)
        tp += c_tp
        fp += c_fp
        fn += c_fn
        iou_sum += c_iou

    recall = tp / float(max(1, tp + fn))
    precision = tp / float(max(1, tp + fp))
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    mean_iou = iou_sum / float(max(1, tp))

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "mean_iou": mean_iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "taralli_yolo_v2",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    names = load_names(dataset / "data.yaml")
    labels = sorted(dataset.rglob("*.txt"))

    samples = []

    for label_path in labels:
        image_path = find_image_for_label(label_path)
        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        gt_boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            values = [float(x) for x in parts[:5]]
            class_id = int(values[0])
            if class_id < 0 or class_id >= len(names):
                continue

            # Blob e un gruppo, quindi non lo usiamo per ottimizzare
            # la segmentazione del singolo tarallo.
            if names[class_id] == "blob":
                continue

            gt_boxes.append(yolo_bbox(image, values))

        if gt_boxes:
            samples.append((image, gt_boxes))

    if not samples:
        raise RuntimeError("Nessuna immagine valida per la calibrazione")

    base = dict(DEFAULT_SEGMENTATION)

    best_params = dict(base)
    best_metrics = evaluate(samples, best_params)

    # Ricerca coarse sulle soglie cromatiche.
    for r_margin, g_margin, sat_min, val_min in itertools.product(
        (4, 7, 10),
        (0, 2, 4),
        (8, 14, 18),
        (30, 40, 45),
    ):
        params = {
            "r_margin": r_margin,
            "g_margin": g_margin,
            "sat_min": sat_min,
            "val_min": val_min,
            "min_component_area": 80,
        }

        metrics = evaluate(samples, params)

        key = (
            metrics["f1"],
            metrics["recall"],
            metrics["mean_iou"],
        )
        best_key = (
            best_metrics["f1"],
            best_metrics["recall"],
            best_metrics["mean_iou"],
        )

        if key > best_key:
            best_params = params
            best_metrics = metrics

    # Rifinitura della sola area minima sulla migliore soglia colore.
    color_best = dict(best_params)

    for min_area in (50, 70, 90, 110, 130):
        params = dict(color_best)
        params["min_component_area"] = min_area
        metrics = evaluate(samples, params)

        key = (
            metrics["f1"],
            metrics["recall"],
            metrics["mean_iou"],
        )
        best_key = (
            best_metrics["f1"],
            best_metrics["recall"],
            best_metrics["mean_iou"],
        )

        if key > best_key:
            best_params = params
            best_metrics = metrics

    output = {
        **best_params,
        "calibration": {
            key: round(float(value), 6)
            for key, value in best_metrics.items()
        },
        "images": len(samples),
    }

    SEGMENTATION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_CONFIG_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("CALIBRAZIONE SEGMENTAZIONE")
    print("Immagini:", len(samples))
    print("Parametri:", best_params)
    print(
        "Recall: %.2f%% | Precision: %.2f%% | F1: %.2f%% | mean IoU: %.3f"
        % (
            best_metrics["recall"] * 100.0,
            best_metrics["precision"] * 100.0,
            best_metrics["f1"] * 100.0,
            best_metrics["mean_iou"],
        )
    )
    print(
        "TP=%d FP=%d FN=%d"
        % (
            int(best_metrics["tp"]),
            int(best_metrics["fp"]),
            int(best_metrics["fn"]),
        )
    )
    print("Configurazione:", SEGMENTATION_CONFIG_PATH)


if __name__ == "__main__":
    main()
