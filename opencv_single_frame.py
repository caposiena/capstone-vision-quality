from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np

from opencv_features import (
    FEATURE_VERSION,
    build_product_mask,
    count_holes,
    crop_object,
    extract_features,
)

WINDOW_VIDEO = "visionTar - OpenCV"
WINDOW_MASK = "visionTar - Maschera"
WINDOW_SETTINGS = "visionTar - Settaggi"

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "opencv_extratrees.joblib"

DEFAULT_ROI_LEFT = 30
DEFAULT_ROI_RIGHT = 12
DEFAULT_ROI_TOP = 7
DEFAULT_ROI_BOTTOM = 51
DEFAULT_DEADLINE = 35
DEFAULT_SPEED = 100

TRACK_MAX_DISTANCE = 85.0
TRACK_MAX_MISSED = 10
BLOB_AREA_RATIO = 1.60

CLASS_COLORS = {
    "buono": (0, 255, 0),
    "cotto": (0, 0, 255),
    "poco": (255, 0, 0),
    "blob": (0, 255, 255),
    "rotto": (255, 255, 255),
    "non_classificato": (160, 160, 160),
}

PROFILE_STAGES = (
    "read",
    "detect",
    "track",
    "classify",
    "draw",
    "imshow",
)


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


def _local_holes(mask: np.ndarray, bbox_roi: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox_roi
    local = mask[y1:y2, x1:x2]
    return count_holes(local)


def detect_taralli(
    frame: np.ndarray,
    settings: dict[str, float | bool],
) -> tuple[
    list[dict[str, object]],
    np.ndarray,
    tuple[int, int, int, int],
    dict[str, int],
    float | None,
]:
    h, w = frame.shape[:2]

    roi_x1 = int(w * float(settings["roi_left"]) / 100.0)
    roi_x2 = int(w * (100.0 - float(settings["roi_right"])) / 100.0)
    roi_y1 = int(h * float(settings["roi_top"]) / 100.0)
    roi_y2 = int(h * (100.0 - float(settings["roi_bottom"])) / 100.0)

    roi_x1 = max(0, min(w - 2, roi_x1))
    roi_x2 = max(roi_x1 + 1, min(w, roi_x2))
    roi_y1 = max(0, min(h - 2, roi_y1))
    roi_y2 = max(roi_y1 + 1, min(h, roi_y2))

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    mask = build_product_mask(roi)

    count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    roi_h, roi_w = mask.shape[:2]

    detections: list[dict[str, object]] = []
    excluded = {
        "too_small": 0,
        "touching_border": 0,
    }

    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 120:
            excluded["too_small"] += 1
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        x2 = x + bw
        y2 = y + bh

        touches_border = (
            x <= 0
            or y <= 0
            or x2 >= roi_w
            or y2 >= roi_h
        )
        if touches_border:
            excluded["touching_border"] += 1

        holes = _local_holes(mask, (x, y, x2, y2))

        fx1 = roi_x1 + x
        fy1 = roi_y1 + y
        fx2 = roi_x1 + x2
        fy2 = roi_y1 + y2

        detections.append(
            {
                "bbox": (fx1, fy1, fx2, fy2),
                "bbox_roi": (x, y, x2, y2),
                "center": ((fx1 + fx2) // 2, (fy1 + fy2) // 2),
                "area": float(area),
                "holes": int(holes),
                "touches_border": bool(touches_border),
            }
        )

    single_candidates = [
        float(d["area"])
        for d in detections
        if int(d["holes"]) == 1 and not bool(d["touches_border"])
    ]

    reference_area = (
        float(np.median(single_candidates))
        if single_candidates
        else None
    )

    for detection in detections:
        area = float(detection["area"])
        holes = int(detection["holes"])

        area_ratio = (
            area / reference_area
            if reference_area and reference_area > 0
            else 0.0
        )

        detection["area_ratio"] = area_ratio
        detection["structural_blob"] = bool(
            holes >= 2
            or (
                reference_area is not None
                and area_ratio >= BLOB_AREA_RATIO
            )
        )

    return (
        detections,
        mask,
        (roi_x1, roi_y1, roi_x2, roi_y2),
        excluded,
        reference_area,
    )


class LearnedClassifier:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.meta_path = model_path.with_suffix(".json")

        if not model_path.exists():
            raise FileNotFoundError(
                "Manca il classificatore: %s\n"
                "Esegui prima: python train_opencv_classifier.py"
                % model_path
            )

        if not self.meta_path.exists():
            raise FileNotFoundError(
                "Mancano i metadata: %s" % self.meta_path
            )

        import json

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))

        if int(meta.get("feature_version", -1)) != FEATURE_VERSION:
            raise RuntimeError(
                "Versione feature non coerente: modello=%s runtime=%s"
                % (meta.get("feature_version"), FEATURE_VERSION)
            )

        self.class_names = [str(x) for x in meta["class_names"]]
        self.feature_count = int(meta["feature_count"])
        self.validation_accuracy = float(meta.get("validation_accuracy", 0.0))
        self.model = joblib.load(model_path)

    def predict_single(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[str, float]:
        crop = crop_object(frame, bbox)
        features = extract_features(crop)

        if features.shape[1] != self.feature_count:
            raise RuntimeError(
                "Feature incompatibili: runtime=%d modello=%d"
                % (features.shape[1], self.feature_count)
            )

        prediction = self.model.predict(features)
        class_id = int(prediction[0])

        if class_id < 0 or class_id >= len(self.class_names):
            return "non_classificato", 0.0

        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))

        return self.class_names[class_id], confidence


class CentroidTracker:
    def __init__(self) -> None:
        self.next_id = 1
        self.tracks: dict[int, dict[str, object]] = {}
        self.expired_without_verdict = 0

    def _new_track(
        self,
        detection: dict[str, object],
        frame_index: int,
    ) -> int:
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

        for detection in detections:
            dcx, dcy = detection["center"]
            best_id = None
            best_distance = TRACK_MAX_DISTANCE

            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks:
                    continue

                tcx, tcy = track["center"]
                distance = math.hypot(
                    float(dcx) - float(tcx),
                    float(dcy) - float(tcy),
                )

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

            display_detection = dict(detection)
            display_detection["display_label"] = track["final_label"]
            results.append((best_id, display_detection, crossed))

        expired = []

        for track_id, track in self.tracks.items():
            if track_id in assigned_tracks:
                continue

            track["missed"] = int(track["missed"]) + 1
            if int(track["missed"]) > TRACK_MAX_MISSED:
                expired.append(track_id)

        for track_id in expired:
            if self.tracks[track_id].get("final_label") is None:
                self.expired_without_verdict += 1
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
            text_color = (
                (0, 0, 0)
                if label in ("blob", "rotto")
                else (255, 255, 255)
            )
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
        "area_ratio",
        "holes",
        "touches_border",
        "decision_rule",
        "ml_confidence",
        "classify_ms",
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
        "Non classificato: %d" % counts.get("non_classificato", 0),
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
        description="visionTar - pipeline OpenCV single-frame"
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

    tracker = CentroidTracker()
    classifier = LearnedClassifier(DEFAULT_CLASSIFIER_PATH)

    print(
        "Classificatore 4 classi caricato | validation accuracy: %.2f%%"
        % (classifier.validation_accuracy * 100.0)
    )

    report_rows: list[dict[str, object]] = []
    profile_samples: dict[str, list[float]] = defaultdict(list)

    excluded_totals = {
        "too_small": 0,
        "touching_border": 0,
    }

    frame_index = 0
    processing_fps = 0.0
    paused = False
    mask_visible = False
    last_output = None

    print("Video:", video_path)
    print("FPS sorgente:", source_fps)
    print(
        "Pipeline: una maschera | blob strutturale | "
        "ExtraTrees 4 classi al crossing"
    )
    print("Tasti: SPAZIO pausa | M maschera | Q/ESC uscita")

    try:
        while True:
            loop_start = time.perf_counter()
            settings = get_settings()

            if not paused:
                read_start = time.perf_counter()
                ok, frame = capture.read()
                profile_samples["read"].append(
                    (time.perf_counter() - read_start) * 1000.0
                )

                if not ok:
                    break

                frame_index += 1

                detect_start = time.perf_counter()
                (
                    detections,
                    mask,
                    roi,
                    excluded,
                    reference_area,
                ) = detect_taralli(frame, settings)

                profile_samples["detect"].append(
                    (time.perf_counter() - detect_start) * 1000.0
                )

                excluded_totals["too_small"] += excluded["too_small"]
                excluded_totals["touching_border"] += excluded["touching_border"]

                deadline_x = int(
                    frame.shape[1]
                    * float(settings["deadline"])
                    / 100.0
                )

                track_start = time.perf_counter()
                tracked = tracker.update(
                    detections,
                    frame_index,
                    deadline_x,
                )
                profile_samples["track"].append(
                    (time.perf_counter() - track_start) * 1000.0
                )

                for track_id, detection, crossed in tracked:
                    if not crossed:
                        continue

                    classify_start = time.perf_counter()

                    if bool(detection["touches_border"]):
                        label = "non_classificato"
                        decision_rule = "roi_border"
                        ml_confidence = 0.0
                    elif bool(detection["structural_blob"]):
                        label = "blob"
                        decision_rule = "structural_blob"
                        ml_confidence = 1.0
                    else:
                        label, ml_confidence = classifier.predict_single(
                            frame,
                            detection["bbox"],
                        )
                        decision_rule = "ml_single"

                    classify_ms = (
                        time.perf_counter() - classify_start
                    ) * 1000.0
                    profile_samples["classify"].append(classify_ms)

                    tracker.tracks[track_id]["final_label"] = label
                    detection["display_label"] = label

                    cx, cy = detection["center"]
                    bx1, by1, bx2, by2 = detection["bbox"]

                    report_rows.append(
                        {
                            "id": track_id,
                            "classe": label,
                            "frame": frame_index,
                            "tempo_video_s": round(
                                frame_index / source_fps,
                                3,
                            ),
                            "x": int(cx),
                            "y": int(cy),
                            "bbox_x1": int(bx1),
                            "bbox_y1": int(by1),
                            "bbox_x2": int(bx2),
                            "bbox_y2": int(by2),
                            "area": round(float(detection["area"]), 2),
                            "area_ratio": round(
                                float(detection["area_ratio"]),
                                3,
                            ),
                            "holes": int(detection["holes"]),
                            "touches_border": bool(
                                detection["touches_border"]
                            ),
                            "decision_rule": decision_rule,
                            "ml_confidence": round(
                                float(ml_confidence),
                                4,
                            ),
                            "classify_ms": round(
                                float(classify_ms),
                                3,
                            ),
                        }
                    )

                    print(
                        "FINAL ID %d | %s | rule=%s | "
                        "area_ratio=%.2f | holes=%d | ml_conf=%.3f | %.2f ms"
                        % (
                            track_id,
                            label,
                            decision_rule,
                            float(detection["area_ratio"]),
                            int(detection["holes"]),
                            float(ml_confidence),
                            classify_ms,
                        )
                    )

                analysis_elapsed = (
                    sum(
                        profile_samples[name][-1]
                        for name in ("detect", "track")
                        if profile_samples[name]
                    )
                    / 1000.0
                )

                instant_fps = (
                    1.0 / analysis_elapsed
                    if analysis_elapsed > 0
                    else source_fps
                )
                processing_fps = (
                    instant_fps
                    if processing_fps == 0.0
                    else processing_fps * 0.90 + instant_fps * 0.10
                )

                draw_start = time.perf_counter()
                last_output = draw_overlay(
                    frame,
                    tracked,
                    roi,
                    deadline_x,
                    processing_fps,
                    source_fps,
                    float(settings["speed_pct"]),
                )
                profile_samples["draw"].append(
                    (time.perf_counter() - draw_start) * 1000.0
                )

                imshow_start = time.perf_counter()
                cv2.imshow(WINDOW_VIDEO, last_output)
                profile_samples["imshow"].append(
                    (time.perf_counter() - imshow_start) * 1000.0
                )

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
                cv2.setTrackbarPos(
                    "MASCHERA",
                    WINDOW_SETTINGS,
                    1 - current,
                )

    finally:
        capture.release()
        cv2.destroyAllWindows()

        print("")
        print("PROFILO PRESTAZIONI (ms)")

        for stage in PROFILE_STAGES:
            values = profile_samples.get(stage, [])
            if not values:
                continue

            arr = np.asarray(values, dtype=np.float64)
            print(
                "%-9s p50=%6.2f  p95=%6.2f  p99=%6.2f  n=%d"
                % (
                    stage,
                    float(np.percentile(arr, 50)),
                    float(np.percentile(arr, 95)),
                    float(np.percentile(arr, 99)),
                    len(values),
                )
            )

        print("")
        print("COPERTURA / ESCLUSIONI")
        print(
            "Track terminati senza verdetto:",
            tracker.expired_without_verdict,
        )
        print(
            "Contorni troppo piccoli:",
            excluded_totals["too_small"],
        )
        print(
            "Contorni a contatto col bordo ROI:",
            excluded_totals["touching_border"],
        )

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
