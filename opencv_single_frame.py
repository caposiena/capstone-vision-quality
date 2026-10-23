from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from opencv_features import extract_features


WINDOW_VIDEO = "visionTar - OpenCV"
WINDOW_MASK = "visionTar - Maschera"
WINDOW_SETTINGS = "visionTar - Settaggi"

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "opencv_rtrees.yml"

# Interfaccia operativa: solo parametri che servono davvero durante la prova.
DEFAULT_ROI_LEFT = 30
DEFAULT_ROI_RIGHT = 12
DEFAULT_ROI_TOP = 7
DEFAULT_ROI_BOTTOM = 51
DEFAULT_DEADLINE = 35
DEFAULT_SPEED = 100

# Parametri tecnici interni. Non sono esposti all'operatore.
BLUE_DOMINANCE = 12
MIN_COMPONENT_AREA = 120
TRACK_MAX_DISTANCE = 85.0
TRACK_MAX_MISSED = 10

# Firma geometrica/cromatica iniziale.
# Le soglie cromatiche riprendono la caratterizzazione del progetto main.
COTTO_MAX_V = 190.0
COTTO_MAX_LAB_L = 168.0
COTTO_MIN_S = 125.0
POCO_MIN_V = 212.0
POCO_MIN_LAB_L = 198.0
POCO_MAX_S = 105.0

# Colorimetria relativa al frame: evita che la sola illuminazione faccia
# diventare "poco" un tarallo normalmente cotto.
POCO_MIN_V_MARGIN = 7.0
POCO_MIN_LAB_L_MARGIN = 6.0
COTTO_MIN_V_MARGIN = 12.0
COTTO_MIN_LAB_L_MARGIN = 10.0

# Normalizzazione locale rispetto al fondo blu vicino al singolo pezzo.
POCO_MIN_V_BG_RATIO = 0.92
POCO_MIN_L_BG_DELTA = -5.0
COTTO_MAX_V_BG_RATIO = 0.72
COTTO_MAX_L_BG_DELTA = -45.0

ROTTO_MAX_CIRCULARITY = 0.62
ROTTO_MAX_SOLIDITY = 0.88
ROTTO_MIN_ASPECT = 0.72
ROTTO_MAX_ASPECT = 1.38
ROTTO_MAX_EXTENT = 0.55
ROTTO_MIN_SIGNALS = 2

# Firma esplicita del BUONO: serve a evitare che "buono" sia solo il residuo.
GOOD_MIN_CIRCULARITY = 0.58
GOOD_MIN_SOLIDITY = 0.90
GOOD_MIN_ASPECT = 0.76
GOOD_MAX_ASPECT = 1.32
GOOD_MIN_EXTENT = 0.50

# Blob: area sensibilmente superiore al riferimento del frame.
BLOB_AREA_FACTOR = 1.55

CLASS_COLORS = {
    "buono": (0, 255, 0),
    "cotto": (0, 0, 255),
    "poco": (255, 0, 0),
    "blob": (0, 255, 255),
    "rotto": (255, 255, 255),
}


def noop(_value: int) -> None:
    pass


def create_settings_window() -> None:
    cv2.namedWindow(WINDOW_SETTINGS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_SETTINGS, 560, 300)

    cv2.createTrackbar("ROI SX %", WINDOW_SETTINGS, DEFAULT_ROI_LEFT, 45, noop)
    cv2.createTrackbar("ROI DX %", WINDOW_SETTINGS, DEFAULT_ROI_RIGHT, 45, noop)
    cv2.createTrackbar("ROI ALTO %", WINDOW_SETTINGS, DEFAULT_ROI_TOP, 45, noop)
    cv2.createTrackbar("ROI BASSO %", WINDOW_SETTINGS, DEFAULT_ROI_BOTTOM, 80, noop)
    cv2.createTrackbar("DEADLINE %", WINDOW_SETTINGS, DEFAULT_DEADLINE, 100, noop)
    cv2.createTrackbar("VELOCITA %", WINDOW_SETTINGS, DEFAULT_SPEED, 300, noop)
    cv2.createTrackbar("MASCHERA", WINDOW_SETTINGS, 0, 1, noop)


def get_settings() -> dict[str, float | bool]:
    return {
        "roi_left": cv2.getTrackbarPos("ROI SX %", WINDOW_SETTINGS),
        "roi_right": cv2.getTrackbarPos("ROI DX %", WINDOW_SETTINGS),
        "roi_top": cv2.getTrackbarPos("ROI ALTO %", WINDOW_SETTINGS),
        "roi_bottom": cv2.getTrackbarPos("ROI BASSO %", WINDOW_SETTINGS),
        "deadline": cv2.getTrackbarPos("DEADLINE %", WINDOW_SETTINGS),
        "speed_pct": max(10, cv2.getTrackbarPos("VELOCITA %", WINDOW_SETTINGS)),
        "show_mask": cv2.getTrackbarPos("MASCHERA", WINDOW_SETTINGS) == 1,
    }


def build_mask(roi: np.ndarray) -> np.ndarray:
    """Silhouette dei soli pixel caldi del prodotto, evitando ponti tra vicini."""
    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    b, g, r = cv2.split(blurred)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    b16 = b.astype(np.int16)
    g16 = g.astype(np.int16)
    r16 = r.astype(np.int16)

    warm_product = (
        (r16 > b16 + 10)
        & (g16 > b16 + 4)
        & (hsv[:, :, 1] >= 18)
        & (hsv[:, :, 2] >= 45)
    )

    mask = (warm_product.astype(np.uint8) * 255)

    # Solo apertura: la chiusura tendeva a creare ponti fra taralli vicini.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)

    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == label_id] = 255

    return cleaned



def split_touching_objects(mask: np.ndarray) -> np.ndarray:
    """Separa i gruppi lavorando componente per componente.

    Il precedente watershed usava una soglia globale e poteva lasciare uniti
    due/tre taralli vicini. Qui ogni componente grande ha la propria distance
    transform e la propria soglia locale.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask

    component_areas = [
        int(stats[i, cv2.CC_STAT_AREA])
        for i in range(1, count)
        if int(stats[i, cv2.CC_STAT_AREA]) >= MIN_COMPONENT_AREA
    ]
    if not component_areas:
        return mask

    median_area = float(np.median(component_areas))
    output = np.zeros_like(mask)

    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA:
            continue

        component = np.uint8(labels == label_id) * 255

        # Un componente di dimensione normale viene mantenuto com'e.
        if area < median_area * 1.35:
            output = cv2.bitwise_or(output, component)
            continue

        dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        local_max = float(dist.max())
        if local_max <= 0.0:
            output = cv2.bitwise_or(output, component)
            continue

        # Semi interni piu selettivi per distinguere anelli adiacenti.
        sure_fg = np.uint8(dist >= (0.48 * local_max)) * 255
        sure_fg = cv2.morphologyEx(
            sure_fg,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        seeds, markers = cv2.connectedComponents(sure_fg)

        # Se non emergono almeno due nuclei reali, non forziamo lo split.
        if seeds <= 2:
            output = cv2.bitwise_or(output, component)
            continue

        unknown = cv2.subtract(component, sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        canvas = cv2.cvtColor(component, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(canvas, markers.astype(np.int32))

        pieces = 0
        for marker_id in range(2, int(markers.max()) + 1):
            piece = np.uint8(markers == marker_id) * 255
            piece_area = cv2.countNonZero(piece)
            if piece_area >= max(MIN_COMPONENT_AREA, int(median_area * 0.35)):
                output = cv2.bitwise_or(output, piece)
                pieces += 1

        if pieces < 2:
            output = cv2.bitwise_or(output, component)

    return output



def contour_features(
    contour: np.ndarray,
    frame: np.ndarray,
    binary_mask: np.ndarray,
    offset_x: int,
    offset_y: int,
) -> dict[str, object]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))

    circularity = (
        4.0 * math.pi * area / (perimeter * perimeter)
        if perimeter > 0
        else 0.0
    )
    solidity = area / hull_area if hull_area > 0 else 0.0
    extent = area / float(max(1, w * h))
    aspect = float(w) / float(max(1, h))

    x1 = offset_x + x
    y1 = offset_y + y
    x2 = x1 + w
    y2 = y1 + h
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    crop = frame[y1:y2, x1:x2]
    local_mask = binary_mask[y:y + h, x:x + w].copy()

    # Numero di fori chiusi nella silhouette.
    # 1 foro -> tarallo normale; >=2 -> probabile gruppo/blob; 0 -> possibile rotto.
    _hole_contours, hierarchy = cv2.findContours(
        local_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    hole_count = 0
    if hierarchy is not None:
        hierarchy_row = hierarchy[0]
        hole_count = sum(1 for item in hierarchy_row if int(item[3]) >= 0)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    valid = local_mask > 0
    values_v = hsv[:, :, 2][valid]
    values_s = hsv[:, :, 1][valid]
    values_l = lab[:, :, 0][valid]

    median_v = float(np.median(values_v)) if values_v.size else 0.0
    median_s = float(np.median(values_s)) if values_s.size else 0.0
    median_lab_l = float(np.median(values_l)) if values_l.size else 0.0

    # Fondo locale: anello rettangolare attorno alla bbox, escluso il prodotto.
    frame_h, frame_w = frame.shape[:2]
    pad_x = max(6, int(w * 0.35))
    pad_y = max(6, int(h * 0.35))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(frame_w, x2 + pad_x)
    ry2 = min(frame_h, y2 + pad_y)

    context = frame[ry1:ry2, rx1:rx2]
    bg_v = 0.0
    bg_l = 0.0

    if context.size:
        context_hsv = cv2.cvtColor(context, cv2.COLOR_BGR2HSV)
        context_lab = cv2.cvtColor(context, cv2.COLOR_BGR2LAB)

        cb, cg, cr = cv2.split(context)
        blue_bg = (
            (cb.astype(np.int16) > cg.astype(np.int16) + 8)
            & (cb.astype(np.int16) > cr.astype(np.int16) + 8)
        )

        if np.count_nonzero(blue_bg) >= 20:
            bg_v = float(np.median(context_hsv[:, :, 2][blue_bg]))
            bg_l = float(np.median(context_lab[:, :, 0][blue_bg]))

    v_bg_ratio = median_v / bg_v if bg_v > 1.0 else 0.0
    l_bg_delta = median_lab_l - bg_l if bg_l > 1.0 else 0.0

    return {
        "bbox": (x1, y1, x2, y2),
        "center": (cx, cy),
        "area": area,
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
        "aspect": aspect,
        "v": median_v,
        "s": median_s,
        "lab_l": median_lab_l,
        "holes": hole_count,
        "bg_v": bg_v,
        "bg_lab_l": bg_l,
        "v_bg_ratio": v_bg_ratio,
        "l_bg_delta": l_bg_delta,
    }


def classify(
    features: dict[str, object],
    reference_area: float | None,
    color_reference: dict[str, float] | None,
) -> str:
    area = float(features["area"])
    circularity = float(features["circularity"])
    solidity = float(features["solidity"])
    aspect = float(features["aspect"])
    extent = float(features["extent"])
    median_v = float(features["v"])
    median_s = float(features["s"])
    median_lab_l = float(features["lab_l"])
    hole_count = int(features.get("holes", 0))
    v_bg_ratio = float(features.get("v_bg_ratio", 0.0))
    l_bg_delta = float(features.get("l_bg_delta", 0.0))

    # 1) BLOB: non basta piu la sola area.
    # Due o piu fori sono evidenza diretta di gruppo.
    if hole_count >= 2:
        return "blob"

    oversized = (
        reference_area is not None
        and area > reference_area * 1.80
    )
    group_geometry = (
        aspect < 0.62
        or aspect > 1.55
        or solidity < 0.78
        or extent < 0.42
    )

    if oversized and group_geometry:
        return "blob"

    # 2) ROTTO: richiede segnali geometrici concordanti.
    shape_signals = 0
    if circularity < ROTTO_MAX_CIRCULARITY:
        shape_signals += 1
    if solidity < ROTTO_MAX_SOLIDITY:
        shape_signals += 1
    if aspect < ROTTO_MIN_ASPECT or aspect > ROTTO_MAX_ASPECT:
        shape_signals += 1
    if extent < ROTTO_MAX_EXTENT:
        shape_signals += 1
    if hole_count == 0:
        shape_signals += 1

    if shape_signals >= ROTTO_MIN_SIGNALS:
        return "rotto"

    # 3) Colore: decisione assoluta + normalizzazione rispetto al fondo locale.
    # Evita che esposizione/ombre globali trasformino un buono in poco/cotto.
    poco_absolute = (
        median_v >= POCO_MIN_V
        and median_lab_l >= POCO_MIN_LAB_L
        and median_s <= POCO_MAX_S
    )
    poco_local = (
        v_bg_ratio >= POCO_MIN_V_BG_RATIO
        and l_bg_delta >= POCO_MIN_L_BG_DELTA
    )

    cotto_absolute = (
        median_v <= COTTO_MAX_V
        and median_lab_l <= COTTO_MAX_LAB_L
        and median_s >= COTTO_MIN_S
    )
    cotto_local = (
        v_bg_ratio > 0.0
        and v_bg_ratio <= COTTO_MAX_V_BG_RATIO
        and l_bg_delta <= COTTO_MAX_L_BG_DELTA
    )

    # Il riferimento del frame resta un controllo secondario, non piu la base.
    frame_poco_support = False
    frame_cotto_support = False
    if color_reference is not None:
        ref_v = float(color_reference["v"])
        ref_l = float(color_reference["lab_l"])
        frame_poco_support = (
            median_v >= ref_v + POCO_MIN_V_MARGIN
            or median_lab_l >= ref_l + POCO_MIN_LAB_L_MARGIN
        )
        frame_cotto_support = (
            median_v <= ref_v - COTTO_MIN_V_MARGIN
            or median_lab_l <= ref_l - COTTO_MIN_LAB_L_MARGIN
        )

    if poco_absolute and (poco_local or frame_poco_support):
        return "poco"

    if cotto_absolute and (cotto_local or frame_cotto_support):
        return "cotto"

    # 4) BUONO: firma geometrica esplicita.
    good_geometry = (
        circularity >= GOOD_MIN_CIRCULARITY
        and solidity >= GOOD_MIN_SOLIDITY
        and GOOD_MIN_ASPECT <= aspect <= GOOD_MAX_ASPECT
        and extent >= GOOD_MIN_EXTENT
        and hole_count == 1
    )

    if good_geometry:
        return "buono"

    return "buono"



def detect_taralli(
    frame: np.ndarray,
    settings: dict[str, float | bool],
) -> tuple[list[dict[str, object]], np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]

    x1 = int(w * float(settings["roi_left"]) / 100.0)
    x2 = int(w * (100.0 - float(settings["roi_right"])) / 100.0)
    y1 = int(h * float(settings["roi_top"]) / 100.0)
    y2 = int(h * (100.0 - float(settings["roi_bottom"])) / 100.0)

    x1 = max(0, min(w - 2, x1))
    x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 2, y1))
    y2 = max(y1 + 1, min(h, y2))

    roi = frame[y1:y2, x1:x2]
    mask = build_mask(roi)
    mask = split_touching_objects(mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    roi_area = max(1, (x2 - x1) * (y2 - y1))
    min_area = max(120.0, roi_area * 0.0010)
    max_area = roi_area * 0.20

    features: list[dict[str, object]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        features.append(contour_features(contour, frame, mask, x1, y1))

    # Nessuna classificazione qui: prima della deadline OpenCV deve solo
    # segmentare e tracciare. La classe viene assegnata una sola volta
    # dal modello RTrees sul frame di crossing.

    return features, mask, (x1, y1, x2, y2)


class LearnedClassifier:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.meta_path = model_path.with_suffix(".json")

        if not model_path.exists():
            raise FileNotFoundError(
                "Manca il classificatore OpenCV: %s\n"
                "Esegui prima: python train_opencv_classifier.py"
                % model_path
            )

        if not self.meta_path.exists():
            raise FileNotFoundError(
                "Mancano i metadata del classificatore: %s"
                % self.meta_path
            )

        import json

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.class_names = [str(x) for x in meta["class_names"]]
        self.feature_count = int(meta["feature_count"])
        self.validation_accuracy = float(meta.get("validation_accuracy", 0.0))
        self.model = cv2.ml.RTrees_load(str(model_path))

    def predict(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = max(2, int(box_w * 0.08))
        pad_y = max(2, int(box_h * 0.08))

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]
        features = extract_features(crop)

        if features.shape[1] != self.feature_count:
            raise RuntimeError(
                "Feature incompatibili: runtime=%d modello=%d"
                % (features.shape[1], self.feature_count)
            )

        _value, prediction = self.model.predict(features)
        class_id = int(round(float(prediction[0, 0])))

        if class_id < 0 or class_id >= len(self.class_names):
            return "buono"

        return self.class_names[class_id]


class CentroidTracker:
    def __init__(self) -> None:
        self.next_id = 1
        self.tracks: dict[int, dict[str, object]] = {}

    def _new_track(self, detection: dict[str, object], frame_index: int) -> int:
        track_id = self.next_id
        self.next_id += 1

        cx, cy = detection["center"]
        self.tracks[track_id] = {
            "center": (int(cx), int(cy)),
            "prev_center": None,
            "bbox": detection["bbox"],
            "final_label": None,
            "last_seen": frame_index,
            "missed": 0,
        }
        return track_id

    def update(
        self,
        detections: list[dict[str, object]],
        frame_index: int,
        deadline_x: int,
    ) -> list[tuple[int, dict[str, object], bool]]:
        assigned_tracks: set[int] = set()
        results: list[tuple[int, dict[str, object], bool]] = []

        # Associazione semplice e veloce per distanza del baricentro.
        for detection in detections:
            dcx, dcy = detection["center"]
            best_id = None
            best_distance = TRACK_MAX_DISTANCE

            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks:
                    continue

                tcx, tcy = track["center"]
                distance = math.hypot(float(dcx) - float(tcx), float(dcy) - float(tcy))
                if distance < best_distance:
                    best_distance = distance
                    best_id = track_id

            if best_id is None:
                best_id = self._new_track(detection, frame_index)
                track = self.tracks[best_id]
            else:
                track = self.tracks[best_id]
                track["prev_center"] = track["center"]
                track["center"] = (int(dcx), int(dcy))
                track["bbox"] = detection["bbox"]
                track["last_seen"] = frame_index
                track["missed"] = 0

            assigned_tracks.add(best_id)

            crossed = False
            prev_center = track.get("prev_center")
            if prev_center is not None and track["final_label"] is None:
                prev_x = int(prev_center[0])
                curr_x = int(track["center"][0])
                crossed = (
                    (prev_x - deadline_x) * (curr_x - deadline_x) <= 0
                    and abs(curr_x - prev_x) >= 2
                )
                if crossed:
                    pass

            display_detection = dict(detection)
            display_detection["display_label"] = track["final_label"]
            display_detection["is_final"] = track["final_label"] is not None
            results.append((best_id, display_detection, crossed))

        # Invecchiamento track non osservati.
        expired = []
        for track_id, track in self.tracks.items():
            if track_id not in assigned_tracks:
                track["missed"] = int(track["missed"]) + 1
                if int(track["missed"]) > TRACK_MAX_MISSED:
                    expired.append(track_id)

        for track_id in expired:
            del self.tracks[track_id]

        return results


def draw_overlay(
    frame: np.ndarray,
    tracked: list[tuple[int, dict[str, object], bool]],
    roi: tuple[int, int, int, int],
    deadline_x: int,
    processing_fps: float,
    source_fps: float,
    speed_pct: float,
) -> np.ndarray:
    output = frame.copy()
    x1, y1, x2, y2 = roi

    cv2.rectangle(output, (x1, y1), (x2, y2), (180, 180, 180), 1)
    cv2.line(output, (deadline_x, y1), (deadline_x, y2), (0, 165, 255), 2)
    cv2.putText(
        output,
        "DEADLINE",
        (max(5, deadline_x - 42), max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )

    counts = {name: 0 for name in CLASS_COLORS}

    for track_id, detection, _crossed in tracked:
        final_label = detection.get("display_label")

        bx1, by1, bx2, by2 = detection["bbox"]
        cx, cy = detection["center"]

        if final_label:
            label = str(final_label)
            counts[label] = counts.get(label, 0) + 1
            color = CLASS_COLORS.get(label, (200, 200, 200))
            text = "#%d %s FINAL" % (track_id, label.upper())
        else:
            color = (180, 180, 180)
            text = "#%d TRACK" % track_id

        cv2.rectangle(output, (bx1, by1), (bx2, by2), color, 2)
        cv2.circle(output, (int(cx), int(cy)), 3, color, -1)

        text_y = max(22, by1 - 8)

        if final_label:
            (tw, th), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                2,
            )
            bg_y1 = max(0, text_y - th - 5)
            bg_y2 = min(output.shape[0] - 1, text_y + baseline + 3)
            bg_x2 = min(output.shape[1] - 1, bx1 + tw + 8)
            cv2.rectangle(
                output,
                (bx1, bg_y1),
                (bg_x2, bg_y2),
                color,
                -1,
            )
            text_color = (0, 0, 0) if label in ("blob", "rotto") else (255, 255, 255)
        else:
            text_color = color

        cv2.putText(
            output,
            text,
            (bx1 + 4 if final_label else bx1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            text_color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        "OpenCV | %.1f FPS proc | %.1f FPS video | velocita %.0f%%"
        % (processing_fps, source_fps, speed_pct),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        "B:%d  C:%d  P:%d  BL:%d  R:%d"
        % (
            counts.get("buono", 0),
            counts.get("cotto", 0),
            counts.get("poco", 0),
            counts.get("blob", 0),
            counts.get("rotto", 0),
        ),
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def write_final_report(
    report_rows: list[dict[str, object]],
    video_path: Path,
    source_fps: float,
) -> tuple[Path, Path]:
    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = video_path.stem.replace(" ", "_")
    csv_path = reports_dir / ("%s_%s_final.csv" % (stem, stamp))
    summary_path = reports_dir / ("%s_%s_summary.txt" % (stem, stamp))

    fieldnames = [
        "id",
        "classe",
        "frame",
        "tempo_video_s",
        "x",
        "y",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "area",
        "circularity",
        "solidity",
        "extent",
        "aspect",
        "v_median",
        "s_median",
        "lab_l_median",
        "holes",
        "bg_v",
        "bg_lab_l",
        "v_bg_ratio",
        "l_bg_delta",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    counts = {name: 0 for name in CLASS_COLORS}
    for row in report_rows:
        label = str(row["classe"])
        counts[label] = counts.get(label, 0) + 1

    total = len(report_rows)
    defects = sum(
        counts.get(name, 0)
        for name in ("cotto", "poco", "blob", "rotto")
    )

    lines = [
        "visionTar - Report finale OpenCV",
        "Video: %s" % video_path,
        "FPS sorgente: %.3f" % source_fps,
        "Decisioni FINAL: %d" % total,
        "Buono: %d" % counts.get("buono", 0),
        "Cotto: %d" % counts.get("cotto", 0),
        "Poco: %d" % counts.get("poco", 0),
        "Blob: %d" % counts.get("blob", 0),
        "Rotto: %d" % counts.get("rotto", 0),
        "Difettosi totali: %d" % defects,
        "Percentuale difettosi: %.2f%%"
        % ((defects / total * 100.0) if total else 0.0),
        "",
        "CSV dettaglio: %s" % csv_path,
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="visionTar - pipeline OpenCV pura realtime"
    )
    parser.add_argument("video", type=Path, help="Percorso del file MP4")
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Impossibile aprire il video")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    source_period = 1.0 / source_fps

    cv2.namedWindow(WINDOW_VIDEO, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_VIDEO, 1280, 720)
    create_settings_window()

    global tracker
    tracker = CentroidTracker()
    report_rows: list[dict[str, object]] = []

    frame_index = 0
    processing_fps = 0.0
    paused = False
    mask_visible = False
    last_output = None

    print("Video:", video_path)
    print("FPS sorgente:", source_fps)
    print("Pipeline: OpenCV puro | tracking continuo | RTrees SINGLE-FRAME alla deadline")
    print("Tasti: SPAZIO pausa | M maschera | Q/ESC uscita")

    try:
        while True:
            loop_start = time.perf_counter()
            settings = get_settings()

            if not paused:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_index += 1
                analysis_start = time.perf_counter()

                detections, mask, roi = detect_taralli(frame, settings)

                deadline_x = int(
                    frame.shape[1] * float(settings["deadline"]) / 100.0
                )

                tracked = tracker.update(
                    detections,
                    frame_index,
                    deadline_x,
                )

                analysis_elapsed = max(
                    time.perf_counter() - analysis_start,
                    1e-6,
                )
                instant_fps = 1.0 / analysis_elapsed
                processing_fps = (
                    instant_fps
                    if processing_fps == 0.0
                    else processing_fps * 0.90 + instant_fps * 0.10
                )

                for track_id, detection, crossed in tracked:
                    if crossed:
                        label = classifier.predict(
                            frame,
                            detection["bbox"],
                        )
                        tracker.tracks[track_id]["final_label"] = label
                        detection["display_label"] = label
                        cx, cy = detection["center"]
                        bx1, by1, bx2, by2 = detection["bbox"]

                        report_rows.append(
                            {
                                "id": track_id,
                                "classe": label,
                                "frame": frame_index,
                                "tempo_video_s": round(frame_index / source_fps, 3),
                                "x": int(cx),
                                "y": int(cy),
                                "bbox_x1": int(bx1),
                                "bbox_y1": int(by1),
                                "bbox_x2": int(bx2),
                                "bbox_y2": int(by2),
                                "area": round(float(detection["area"]), 2),
                                "circularity": round(float(detection["circularity"]), 4),
                                "solidity": round(float(detection["solidity"]), 4),
                                "extent": round(float(detection["extent"]), 4),
                                "aspect": round(float(detection["aspect"]), 4),
                                "v_median": round(float(detection["v"]), 2),
                                "s_median": round(float(detection["s"]), 2),
                                "lab_l_median": round(float(detection["lab_l"]), 2),
                                "holes": int(detection.get("holes", 0)),
                                "bg_v": round(float(detection.get("bg_v", 0.0)), 2),
                                "bg_lab_l": round(float(detection.get("bg_lab_l", 0.0)), 2),
                                "v_bg_ratio": round(float(detection.get("v_bg_ratio", 0.0)), 4),
                                "l_bg_delta": round(float(detection.get("l_bg_delta", 0.0)), 2),
                            }
                        )

                        print(
                            "FINAL ID %d | %s | x=%d y=%d"
                            % (
                                track_id,
                                label,
                                int(cx),
                                int(cy),
                            )
                        )

                last_output = draw_overlay(
                    frame,
                    tracked,
                    roi,
                    deadline_x,
                    processing_fps,
                    source_fps,
                    float(settings["speed_pct"]),
                )
                cv2.imshow(WINDOW_VIDEO, last_output)

                if bool(settings["show_mask"]):
                    if not mask_visible:
                        cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
                        mask_visible = True
                    cv2.imshow(WINDOW_MASK, mask)
                elif mask_visible:
                    cv2.destroyWindow(WINDOW_MASK)
                    mask_visible = False

            elif last_output is not None:
                cv2.imshow(WINDOW_VIDEO, last_output)

            elapsed = time.perf_counter() - loop_start
            speed_factor = float(settings["speed_pct"]) / 100.0
            target_period = source_period / speed_factor
            delay_ms = max(1, int((target_period - elapsed) * 1000.0))
            key = cv2.waitKey(delay_ms) & 0xFF

            if key in (27, ord("q")):
                break
            if key == 32:
                paused = not paused
            if key == ord("m"):
                current = 1 if bool(settings["show_mask"]) else 0
                cv2.setTrackbarPos("MASCHERA", WINDOW_SETTINGS, 1 - current)

    finally:
        capture.release()
        cv2.destroyAllWindows()

        csv_path, summary_path = write_final_report(
            report_rows,
            video_path,
            source_fps,
        )
        print("")
        print("REPORT FINALE")
        print("CSV:", csv_path)
        print("Riepilogo:", summary_path)


if __name__ == "__main__":
    main()
