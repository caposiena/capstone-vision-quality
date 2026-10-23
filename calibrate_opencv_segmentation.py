from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from opencv_features import (
    DEFAULT_SEGMENTATION,
    SEGMENTATION_CONFIG_PATH,
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
    image_shape: tuple[int, int],
    values: list[float],
) -> tuple[int, int, int, int]:
    h, w = image_shape
    _class_id, xc, yc, bw, bh = values

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


def precompute_image(image: np.ndarray) -> dict[str, np.ndarray]:
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(blurred)

    return {
        "hue": hsv[:, :, 0],
        "sat": hsv[:, :, 1],
        "b16": b.astype(np.int16),
        "g16": g.astype(np.int16),
        "r16": r.astype(np.int16),
    }


def build_mask_precomputed(
    pre: dict[str, np.ndarray],
    params: dict[str, int],
) -> np.ndarray:
    hue = pre["hue"]
    sat = pre["sat"]
    b16 = pre["b16"]
    g16 = pre["g16"]
    r16 = pre["r16"]

    blue_background = (
        (hue >= int(params["hue_low"]))
        & (hue <= int(params["hue_high"]))
        & (sat >= int(params["sat_min"]))
        & (b16 >= g16 + int(params["blue_margin"]))
        & (b16 >= r16 + int(params["blue_margin"]))
    )

    raw = (~blue_background).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    if count <= 1:
        return np.zeros_like(opened)

    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = (
        stats[1:, cv2.CC_STAT_AREA]
        >= int(params["min_component_area"])
    ).astype(np.uint8) * 255

    return keep[labels]


def runtime_boxes_precomputed(
    pre: dict[str, np.ndarray],
    params: dict[str, int],
) -> list[tuple[int, int, int, int]]:
    mask = build_mask_precomputed(pre, params)
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
    samples: list[
        tuple[
            dict[str, np.ndarray],
            list[tuple[int, int, int, int]],
        ]
    ],
    params: dict[str, int],
) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    iou_sum = 0.0

    for pre, gt_boxes in samples:
        pred_boxes = runtime_boxes_precomputed(pre, params)
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


def metric_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        metrics["f1"],
        metrics["recall"],
        metrics["mean_iou"],
    )


def save_best(
    params: dict[str, int],
    metrics: dict[str, float],
    images: int,
) -> None:
    output = {
        **params,
        "calibration": {
            key: round(float(value), 6)
            for key, value in metrics.items()
        },
        "images": images,
    }

    SEGMENTATION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_CONFIG_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )


def print_progress(
    phase: str,
    index: int,
    total: int,
    started: float,
    best_metrics: dict[str, float],
) -> None:
    elapsed = time.perf_counter() - started
    pct = (index / total * 100.0) if total else 100.0

    print(
        "%s %d/%d (%.1f%%) | elapsed %.1fs | best F1 %.2f%% recall %.2f%%"
        % (
            phase,
            index,
            total,
            pct,
            elapsed,
            best_metrics["f1"] * 100.0,
            best_metrics["recall"] * 100.0,
        ),
        flush=True,
    )


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

    print("Precomputazione immagini...", flush=True)

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

            if names[class_id] == "blob":
                continue

            gt_boxes.append(
                yolo_bbox(
                    image.shape[:2],
                    values,
                )
            )

        if gt_boxes:
            samples.append(
                (
                    precompute_image(image),
                    gt_boxes,
                )
            )

    if not samples:
        raise RuntimeError("Nessuna immagine valida per la calibrazione")

    print("Immagini precomputate:", len(samples), flush=True)

    best_params = dict(DEFAULT_SEGMENTATION)
    best_metrics = evaluate(samples, best_params)
    save_best(best_params, best_metrics, len(samples))

    started = time.perf_counter()

    coarse_grid = list(
        itertools.product(
            (4, 10, 16),
            (82, 88, 94),
            (130, 140),
            (35, 55),
        )
    )

    total_coarse = len(coarse_grid)

    for index, (blue_margin, hue_low, hue_high, sat_min) in enumerate(
        coarse_grid,
        start=1,
    ):
        params = {
            "blue_margin": blue_margin,
            "hue_low": hue_low,
            "hue_high": hue_high,
            "sat_min": sat_min,
            "min_component_area": 80,
        }

        metrics = evaluate(samples, params)

        if metric_key(metrics) > metric_key(best_metrics):
            best_params = params
            best_metrics = metrics
            save_best(best_params, best_metrics, len(samples))

        if index == 1 or index % 6 == 0 or index == total_coarse:
            print_progress(
                "COARSE",
                index,
                total_coarse,
                started,
                best_metrics,
            )

    bm = int(best_params["blue_margin"])
    hl = int(best_params["hue_low"])
    hh = int(best_params["hue_high"])
    sm = int(best_params["sat_min"])

    refine_grid = []

    for blue_margin, hue_low, hue_high, sat_min, min_area in itertools.product(
        sorted(set((max(0, bm - 3), bm, bm + 3))),
        sorted(set((max(0, hl - 4), hl, hl + 4))),
        sorted(set((max(1, hh - 5), hh, min(179, hh + 5)))),
        sorted(set((max(0, sm - 10), sm, min(255, sm + 10)))),
        (50, 80, 110),
    ):
        if hue_low >= hue_high:
            continue

        refine_grid.append(
            (
                blue_margin,
                hue_low,
                hue_high,
                sat_min,
                min_area,
            )
        )

    total_refine = len(refine_grid)
    refine_started = time.perf_counter()

    for index, (
        blue_margin,
        hue_low,
        hue_high,
        sat_min,
        min_area,
    ) in enumerate(refine_grid, start=1):
        params = {
            "blue_margin": blue_margin,
            "hue_low": hue_low,
            "hue_high": hue_high,
            "sat_min": sat_min,
            "min_component_area": min_area,
        }

        metrics = evaluate(samples, params)

        if metric_key(metrics) > metric_key(best_metrics):
            best_params = params
            best_metrics = metrics
            save_best(best_params, best_metrics, len(samples))

        if index == 1 or index % 15 == 0 or index == total_refine:
            print_progress(
                "REFINE",
                index,
                total_refine,
                refine_started,
                best_metrics,
            )

    save_best(best_params, best_metrics, len(samples))

    total_elapsed = time.perf_counter() - started

    print("")
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
    print("Tempo totale: %.1fs" % total_elapsed)
    print("Configurazione:", SEGMENTATION_CONFIG_PATH)


if __name__ == "__main__":
    main()
