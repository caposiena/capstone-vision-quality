from __future__ import annotations

import cv2
import numpy as np

FEATURE_VERSION = 1


def warm_mask(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    b, g, r = cv2.split(blurred)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    b16 = b.astype(np.int16)
    g16 = g.astype(np.int16)
    r16 = r.astype(np.int16)

    mask = (
        (r16 > b16 + 8)
        & (g16 > b16 + 2)
        & (hsv[:, :, 1] >= 12)
        & (hsv[:, :, 2] >= 35)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


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

    mask = warm_mask(crop)
    valid = mask > 0

    # Fallback prudente: se la segmentazione fallisce, usa l'intero crop.
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

    # Geometria della silhouette.
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

        x, y, w, h_box = cv2.boundingRect(largest_contour)
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

    arr = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    return arr
