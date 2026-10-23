from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

FEATURE_VERSION = 3

PROJECT_ROOT = Path(__file__).resolve().parent
SEGMENTATION_CONFIG_PATH = PROJECT_ROOT / "models" / "opencv_segmentation.json"

DEFAULT_SEGMENTATION = {
    "blue_margin": 10,
    "hue_low": 85,
    "hue_high": 135,
    "sat_min": 55,
    "min_component_area": 90,
}


def load_segmentation_config() -> dict[str, int]:
    config = dict(DEFAULT_SEGMENTATION)

    if SEGMENTATION_CONFIG_PATH.exists():
        saved = json.loads(
            SEGMENTATION_CONFIG_PATH.read_text(encoding="utf-8")
        )
        for key in config:
            if key in saved:
                config[key] = int(saved[key])

    return config


SEGMENTATION = load_segmentation_config()


def build_product_mask(
    image: np.ndarray,
    params: dict[str, int] | None = None,
) -> np.ndarray:
    """Segmenta il prodotto sottraendo il fondo blu del nastro.

    Il prodotto puo essere chiaro, scuro o molto cotto; il criterio primario
    e quindi il colore stabile del fondo, non il colore assoluto del tarallo.
    """
    cfg = SEGMENTATION if params is None else params

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(blurred)

    b16 = b.astype(np.int16)
    g16 = g.astype(np.int16)
    r16 = r.astype(np.int16)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]

    blue_background = (
        (hue >= int(cfg["hue_low"]))
        & (hue <= int(cfg["hue_high"]))
        & (sat >= int(cfg["sat_min"]))
        & (b16 >= g16 + int(cfg["blue_margin"]))
        & (b16 >= r16 + int(cfg["blue_margin"]))
    )

    # Il prodotto e tutto cio che non appartiene con buona confidenza al fondo blu.
    raw = (~blue_background).astype(np.uint8) * 255

    # Rimuove bordi/rumore sottile senza unire prodotti vicini.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    if count <= 1:
        return np.zeros_like(opened)

    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = (
        stats[1:, cv2.CC_STAT_AREA]
        >= int(cfg["min_component_area"])
    ).astype(np.uint8) * 255

    return keep[labels]


def crop_object(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad_frac: float = 0.06,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    pad_x = max(2, int((x2 - x1) * pad_frac))
    pad_y = max(2, int((y2 - y1) * pad_frac))

    return image[
        max(0, y1 - pad_y):min(h, y2 + pad_y),
        max(0, x1 - pad_x):min(w, x2 + pad_x),
    ]


def count_holes(local_mask: np.ndarray, min_hole_px: int = 10) -> int:
    contours, hierarchy = cv2.findContours(
        local_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return 0

    holes = 0
    for contour, item in zip(contours, hierarchy[0]):
        if int(item[3]) >= 0 and cv2.contourArea(contour) >= min_hole_px:
            holes += 1
    return holes


def _hist(values: np.ndarray, bins: int, max_value: float) -> list[float]:
    if values.size == 0:
        return [0.0] * bins
    hist, _ = np.histogram(values, bins=bins, range=(0.0, max_value))
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist.tolist()


def extract_features(crop: np.ndarray) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.zeros((1, 59), dtype=np.float32)

    mask = build_product_mask(crop)
    valid = mask > 0

    if np.count_nonzero(valid) < 20:
        valid = np.ones(mask.shape, dtype=bool)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    h = hsv[:, :, 0][valid].astype(np.float32)
    s = hsv[:, :, 1][valid].astype(np.float32)
    v = hsv[:, :, 2][valid].astype(np.float32)
    l = lab[:, :, 0][valid].astype(np.float32)
    a = lab[:, :, 1][valid].astype(np.float32)
    b = lab[:, :, 2][valid].astype(np.float32)

    def stats(values: np.ndarray) -> list[float]:
        if values.size == 0:
            return [0.0] * 7
        return [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.percentile(values, 10)),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 50)),
            float(np.percentile(values, 75)),
            float(np.percentile(values, 90)),
        ]

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contour_areas = [float(cv2.contourArea(c)) for c in contours]
    min_geom_area = max(20.0, crop.shape[0] * crop.shape[1] * 0.01)

    external = []
    hole_count = 0.0

    if hierarchy is not None:
        for idx, contour in enumerate(contours):
            area = contour_areas[idx]
            parent = int(hierarchy[0][idx][3])

            if parent < 0 and area >= min_geom_area:
                external.append((contour, area))
            elif parent >= 0 and area >= 10.0:
                hole_count += 1.0

    component_count = float(len(external))
    circularity = 0.0
    solidity = 0.0
    extent = 0.0
    aspect = 0.0
    largest_ratio = 0.0

    if external:
        largest_contour, largest_area = max(external, key=lambda item: item[1])

        perimeter = float(cv2.arcLength(largest_contour, True))
        circularity = (
            4.0 * np.pi * largest_area / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )

        hull = cv2.convexHull(largest_contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = largest_area / hull_area if hull_area > 0 else 0.0

        _x, _y, w, h_box = cv2.boundingRect(largest_contour)
        extent = largest_area / float(max(1, w * h_box))
        aspect = float(w) / float(max(1, h_box))
        largest_ratio = largest_area / float(max(1, crop.shape[0] * crop.shape[1]))

    occupancy = float(np.count_nonzero(mask)) / float(max(1, mask.size))

    vector = []
    vector += stats(s)
    vector += stats(v)
    vector += stats(l)
    vector += [
        float(np.mean(h)) if h.size else 0.0,
        float(np.std(h)) if h.size else 0.0,
        float(np.mean(a)) if a.size else 0.0,
        float(np.std(a)) if a.size else 0.0,
        float(np.mean(b)) if b.size else 0.0,
        float(np.std(b)) if b.size else 0.0,
    ]
    vector += _hist(h, 8, 180.0)
    vector += _hist(v, 8, 256.0)
    vector += _hist(l, 8, 256.0)
    vector += [
        occupancy,
        component_count,
        hole_count,
        circularity,
        solidity,
        extent,
        aspect,
        largest_ratio,
    ]

    return np.asarray(vector, dtype=np.float32).reshape(1, -1)
