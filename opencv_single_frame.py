from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


WINDOW_VIDEO = "visionTar - OpenCV single frame"
WINDOW_MASK = "visionTar - Maschera"
WINDOW_SETTINGS = "visionTar - Settaggi"

# Default richiesti per la prova PC.
DEFAULT_ROI_LEFT = 30
DEFAULT_ROI_RIGHT = 12
DEFAULT_ROI_TOP = 7
DEFAULT_ROI_BOTTOM = 51

# Soglie iniziali, volutamente regolabili live.
DEFAULT_MIN_AREA = 500
DEFAULT_MAX_AREA = 14000
DEFAULT_MIN_SAT = 25
DEFAULT_BLUE_MARGIN = 10
DEFAULT_DARK_V = 150
DEFAULT_LIGHT_V = 220
DEFAULT_BLOB_AREA = 8000
DEFAULT_ROTTO_SOLIDITY = 82
DEFAULT_ROTTO_CIRCULARITY = 55

CLASS_COLORS = {
    "buono": (0, 255, 0),
    "cotto": (0, 0, 255),
    "poco": (255, 0, 0),
    "blob": (0, 255, 255),
    "rotto": (255, 255, 255),
}


def noop(_value: int) -> None:
    pass


def create_trackbars() -> None:
    cv2.namedWindow(WINDOW_SETTINGS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_SETTINGS, 560, 430)

    cv2.createTrackbar("ROI SX %", WINDOW_SETTINGS, DEFAULT_ROI_LEFT, 45, noop)
    cv2.createTrackbar("ROI DX %", WINDOW_SETTINGS, DEFAULT_ROI_RIGHT, 45, noop)
    cv2.createTrackbar("ROI ALTO %", WINDOW_SETTINGS, DEFAULT_ROI_TOP, 45, noop)
    cv2.createTrackbar("ROI BASSO %", WINDOW_SETTINGS, DEFAULT_ROI_BOTTOM, 80, noop)

    cv2.createTrackbar("MIN AREA", WINDOW_SETTINGS, DEFAULT_MIN_AREA, 5000, noop)
    cv2.createTrackbar("MAX AREA /100", WINDOW_SETTINGS, DEFAULT_MAX_AREA // 100, 400, noop)
    cv2.createTrackbar("SAT MIN", WINDOW_SETTINGS, DEFAULT_MIN_SAT, 255, noop)
    cv2.createTrackbar("BLUE MARGIN", WINDOW_SETTINGS, DEFAULT_BLUE_MARGIN, 80, noop)

    cv2.createTrackbar("V MIN SCURO", WINDOW_SETTINGS, DEFAULT_DARK_V, 255, noop)
    cv2.createTrackbar("V MAX CHIARO", WINDOW_SETTINGS, DEFAULT_LIGHT_V, 255, noop)
    cv2.createTrackbar("ROTTO SOL %", WINDOW_SETTINGS, DEFAULT_ROTTO_SOLIDITY, 100, noop)
    cv2.createTrackbar("ROTTO CIRC %", WINDOW_SETTINGS, DEFAULT_ROTTO_CIRCULARITY, 100, noop)
    cv2.createTrackbar("AREA GROSSO /100", WINDOW_SETTINGS, DEFAULT_BLOB_AREA // 100, 400, noop)
    cv2.createTrackbar("MOSTRA MASCHERA", WINDOW_SETTINGS, 1, 1, noop)
    cv2.createTrackbar("MOSTRA DEBUG", WINDOW_SETTINGS, 1, 1, noop)
    cv2.createTrackbar("VELOCITA %", WINDOW_SETTINGS, 100, 300, noop)


def get_settings() -> dict[str, float]:
    return {
        "roi_left": cv2.getTrackbarPos("ROI SX %", WINDOW_SETTINGS),
        "roi_right": cv2.getTrackbarPos("ROI DX %", WINDOW_SETTINGS),
        "roi_top": cv2.getTrackbarPos("ROI ALTO %", WINDOW_SETTINGS),
        "roi_bottom": cv2.getTrackbarPos("ROI BASSO %", WINDOW_SETTINGS),
        "min_area": max(20, cv2.getTrackbarPos("MIN AREA", WINDOW_SETTINGS)),
        "max_area": max(
            100,
            cv2.getTrackbarPos("MAX AREA /100", WINDOW_SETTINGS) * 100,
        ),
        "min_sat": cv2.getTrackbarPos("SAT MIN", WINDOW_SETTINGS),
        "blue_margin": cv2.getTrackbarPos("BLUE MARGIN", WINDOW_SETTINGS),
        "dark_v": cv2.getTrackbarPos("V MIN SCURO", WINDOW_SETTINGS),
        "light_v": cv2.getTrackbarPos("V MAX CHIARO", WINDOW_SETTINGS),
        "rotto_solidity": cv2.getTrackbarPos("ROTTO SOL %", WINDOW_SETTINGS) / 100.0,
        "rotto_circularity": cv2.getTrackbarPos("ROTTO CIRC %", WINDOW_SETTINGS) / 100.0,
        "blob_area": max(
            100.0,
            float(cv2.getTrackbarPos("AREA GROSSO /100", WINDOW_SETTINGS) * 100),
        ),
        "show_mask": cv2.getTrackbarPos("MOSTRA MASCHERA", WINDOW_SETTINGS) == 1,
        "show_debug": cv2.getTrackbarPos("MOSTRA DEBUG", WINDOW_SETTINGS) == 1,
        "speed_pct": max(
            10,
            cv2.getTrackbarPos("VELOCITA %", WINDOW_SETTINGS),
        ),
    }


def build_mask(roi: np.ndarray, min_sat: int, blue_margin: int) -> np.ndarray:
    # Il fondo e blu: selezioniamo i pixel che NON hanno dominante blu.
    b, g, r = cv2.split(roi)
    non_blue = (
        (b.astype(np.int16) + blue_margin < g.astype(np.int16))
        | (b.astype(np.int16) + blue_margin < r.astype(np.int16))
    )

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat_ok = hsv[:, :, 1] >= min_sat

    mask = ((non_blue & sat_ok).astype(np.uint8) * 255)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)
    return mask


def contour_features(
    contour: np.ndarray,
    frame: np.ndarray,
    offset_x: int,
    offset_y: int,
) -> dict[str, float | tuple[int, int, int, int]]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))

    circularity = (
        4.0 * np.pi * area / (perimeter * perimeter)
        if perimeter > 0
        else 0.0
    )
    solidity = area / hull_area if hull_area > 0 else 0.0
    extent = area / float(w * h) if w > 0 and h > 0 else 0.0
    aspect = float(w) / float(h) if h > 0 else 0.0

    fx1, fy1 = offset_x + x, offset_y + y
    fx2, fy2 = fx1 + w, fy1 + h

    crop = frame[fy1:fy2, fx1:fx2]
    local_mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(local_mask, [shifted], -1, 255, -1)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    valid_l = lab[:, :, 0][local_mask > 0]
    lab_l = float(np.median(valid_l)) if valid_l.size else 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    valid_v = hsv[:, :, 2][local_mask > 0]
    hsv_v = float(np.median(valid_v)) if valid_v.size else 0.0

    return {
        "bbox": (fx1, fy1, fx2, fy2),
        "area": area,
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
        "aspect": aspect,
        "lab_l": lab_l,
        "hsv_v": hsv_v,
    }


def classify(
    features: dict[str, float | tuple[int, int, int, int]],
    settings: dict[str, float],
    reference_area: float | None,
) -> str:
    area = float(features["area"])
    circularity = float(features["circularity"])
    solidity = float(features["solidity"])
    lab_l = float(features["lab_l"])
    hsv_v = float(features["hsv_v"])

    # Geometria prima: rotto/blob hanno priorita su colore.
    if (
        solidity < settings["rotto_solidity"]
        or circularity < settings["rotto_circularity"]
    ):
        return "rotto"

    if area > settings["blob_area"]:
        return "blob"

    if hsv_v < settings["dark_v"]:
        return "cotto"

    if hsv_v > settings["light_v"]:
        return "poco"

    return "buono"


def process_frame(
    frame: np.ndarray,
    settings: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    h, w = frame.shape[:2]
    x1 = int(w * settings["roi_left"] / 100.0)
    x2 = int(w * (100.0 - settings["roi_right"]) / 100.0)
    y1 = int(h * settings["roi_top"] / 100.0)
    y2 = int(h * (100.0 - settings["roi_bottom"]) / 100.0)

    x1 = max(0, min(w - 2, x1))
    x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 2, y1))
    y2 = max(y1 + 1, min(h, y2))

    roi = frame[y1:y2, x1:x2]
    mask = build_mask(
        roi,
        int(settings["min_sat"]),
        int(settings["blue_margin"]),
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    raw_features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < settings["min_area"] or area > settings["max_area"]:
            continue
        raw_features.append(contour_features(contour, frame, x1, y1))

    areas = sorted(float(item["area"]) for item in raw_features)
    reference_area = float(np.median(areas)) if areas else None

    output = frame.copy()
    counts = {name: 0 for name in CLASS_COLORS}

    cv2.rectangle(output, (x1, y1), (x2, y2), (180, 180, 180), 1)

    for item in raw_features:
        label = classify(item, settings, reference_area)
        counts[label] += 1

        bx1, by1, bx2, by2 = item["bbox"]
        color = CLASS_COLORS[label]

        cv2.rectangle(output, (bx1, by1), (bx2, by2), color, 2)
        if settings["show_debug"]:
            cv2.putText(
                output,
                "%s | A=%.0f | V=%.0f | circ=%.2f | sol=%.2f"
                % (
                    label,
                    float(item["area"]),
                    float(item["hsv_v"]),
                    float(item["circularity"]),
                    float(item["solidity"]),
                ),
                (bx1, max(18, by1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                color,
                1,
                cv2.LINE_AA,
            )

    total = sum(counts.values())
    cv2.putText(
        output,
        "Totale=%d  buono=%d  cotto=%d  poco=%d  blob=%d  rotto=%d"
        % (
            total,
            counts["buono"],
            counts["cotto"],
            counts["poco"],
            counts["blob"],
            counts["rotto"],
        ),
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output, mask, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisionTar - prova PC solo OpenCV, classificazione per singolo frame"
    )
    parser.add_argument("video", type=Path, help="Percorso del file MP4")
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Impossibile aprire il video")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_period = 1.0 / fps

    cv2.namedWindow(WINDOW_VIDEO, cv2.WINDOW_NORMAL)
    create_trackbars()

    print("Video:", video_path)
    print("FPS sorgente:", fps)
    print("Pipeline: OpenCV puro | trigger su singolo frame | nessun YOLO | nessun tracking")
    print("LIVE alla velocita sorgente; analisi solo al trigger")

    paused = False
    frozen = False
    frozen_frame = None
    frozen_output = None
    frozen_mask = None
    fps_ema = 0.0
    mask_visible = False

    print("Comandi:")
    print("  T = TRIGGER: congela il frame corrente e lo analizza una volta")
    print("  R = ritorna al video LIVE")
    print("  SPAZIO = pausa/riprendi")
    print("  Q/ESC = esci")

    try:
        while True:
            loop_started = time.perf_counter()
            settings = get_settings()

            if not frozen and not paused:
                ok, frame = capture.read()
                if not ok:
                    break

                # In LIVE non facciamo analisi: massima velocita possibile.
                live = frame.copy()
                cv2.putText(
                    live,
                    "LIVE %.0f%% - premi T per trigger" % settings["speed_pct"],
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                frozen_frame = frame.copy()
                cv2.imshow(WINDOW_VIDEO, live)

            elif frozen and frozen_output is not None:
                cv2.imshow(WINDOW_VIDEO, frozen_output)

                if settings["show_mask"] and frozen_mask is not None:
                    if not mask_visible:
                        cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
                        mask_visible = True
                    cv2.imshow(WINDOW_MASK, frozen_mask)
                elif mask_visible:
                    cv2.destroyWindow(WINDOW_MASK)
                    mask_visible = False

            elapsed = time.perf_counter() - loop_started
            speed_factor = settings["speed_pct"] / 100.0
            target_period = frame_period / speed_factor
            delay_ms = max(1, int((target_period - elapsed) * 1000))
            key = cv2.waitKey(delay_ms) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord("t") and frozen_frame is not None:
                trigger_started = time.perf_counter()
                frozen_output, frozen_mask, counts = process_frame(
                    frozen_frame.copy(),
                    settings,
                )
                trigger_ms = (time.perf_counter() - trigger_started) * 1000.0

                cv2.putText(
                    frozen_output,
                    "FROZEN - trigger %.1f ms | premi R per LIVE" % trigger_ms,
                    (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                print(
                    "TRIGGER: %.1f ms | totale=%d | %s"
                    % (trigger_ms, sum(counts.values()), counts)
                )
                frozen = True

            elif key == ord("r"):
                frozen = False
                frozen_output = None
                frozen_mask = None
                if mask_visible:
                    cv2.destroyWindow(WINDOW_MASK)
                    mask_visible = False
                print("LIVE")

            elif key == 32:
                paused = not paused

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
