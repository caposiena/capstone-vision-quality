"""
visionTar.py

Prototipo Capstone per il riconoscimento di taralli da file video MP4.

Versione 1.3
- GUI unica per selezione video e modello;
- modalita TEST per salvare tutte le detection del modello generico;
- modalita NORMALE per salvare solo le classi considerate difettose;
- acquisizione frame tramite OpenCV;
- object detection tramite Ultralytics YOLO;
- visualizzazione di bounding box e baricentro;
- registrazione CSV delle coordinate;
- chiusura ordinata di video, finestra OpenCV e file CSV.

Dipendenze:
    pip install opencv-python ultralytics

Nota:
Le classi considerate difettose sono definite in DEFECT_CLASSES.
Per il dataset V2 le classi difettose sono blob, cotto, poco e rotto.
La sola classe non difettosa e buono.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from ultralytics import YOLO


DEFECT_CLASSES = {
    "blob",
    "cotto",
    "poco",
    "rotto",
}

CONFIDENCE_THRESHOLD = 0.35

# Area centrale di ispezione. Le detection con baricentro fuori da questa
# regione vengono ignorate per ridurre falsi positivi ai bordi del frame.
ROI_LEFT_RATIO = 0.08
ROI_RIGHT_RATIO = 0.08
ROI_TOP_RATIO = 0.20
ROI_BOTTOM_RATIO = 0.20
DEADLINE_X_RATIO = 0.50
TRACK_MAX_MISSED_FRAMES = 8
TRACK_OBSERVATION_WINDOW = 12
TRACK_MIN_FINAL_OBSERVATIONS = 4
TRACK_MIN_VOTE_SHARE = 0.55
DEFECT_BINARY_THRESHOLD = 0.40
DEFECT_BINARY_THRESHOLD_SENSITIVE = 0.30
DEFECT_CLASS_MIN_SHARE = 0.38
SENSITIVE_DEFECT_CLASSES = {"cotto", "rotto"}
SENSITIVE_MIN_OBSERVATIONS = 2
SENSITIVE_MIN_AVG_CONFIDENCE = 0.55
TRACK_MIN_MATCH_SCORE = 0.18
TRACK_MAX_JUMP_RATIO = 0.06
TRACK_MIN_CROSSING_DELTA = 4


def choose_inputs() -> tuple[Path, Path, bool, float, float, float, float, float] | None:
    """Mostra una sola GUI per selezionare video, modello e modalita di prova."""

    root = tk.Tk()
    root.title("visionTar - Selezione input")
    root.geometry("820x620")
    root.resizable(False, False)

    video_var = tk.StringVar()
    model_var = tk.StringVar()
    test_mode_var = tk.BooleanVar(value=True)
    roi_left_var = tk.DoubleVar(value=ROI_LEFT_RATIO * 100)
    roi_right_var = tk.DoubleVar(value=ROI_RIGHT_RATIO * 100)
    roi_top_var = tk.DoubleVar(value=ROI_TOP_RATIO * 100)
    roi_bottom_var = tk.DoubleVar(value=ROI_BOTTOM_RATIO * 100)
    deadline_var = tk.DoubleVar(value=DEADLINE_X_RATIO * 100)
    roi_reference_var = tk.StringVar()
    video_size = {"width": None, "height": None}

    def update_roi_reference(*_args) -> None:
        left = roi_left_var.get()
        right = roi_right_var.get()
        top = roi_top_var.get()
        bottom = roi_bottom_var.get()
        deadline = deadline_var.get()

        x1_pct = left
        x2_pct = 100.0 - right
        y1_pct = top
        y2_pct = 100.0 - bottom

        width = video_size["width"]
        height = video_size["height"]

        if width and height:
            x1_px = int(width * x1_pct / 100.0)
            x2_px = int(width * x2_pct / 100.0)
            y1_px = int(height * y1_pct / 100.0)
            y2_px = int(height * y2_pct / 100.0)
            deadline_px = int(width * deadline / 100.0)
            roi_reference_var.set(
                f"BOX ROI: X {x1_pct:.0f}% -> {x2_pct:.0f}%  "
                f"({x1_px} -> {x2_px} px) | "
                f"Y {y1_pct:.0f}% -> {y2_pct:.0f}%  "
                f"({y1_px} -> {y2_px} px) | "
                f"DEADLINE X {deadline:.0f}% ({deadline_px}px) | "
                f"Video {width}x{height}"
            )
        else:
            roi_reference_var.set(
                f"BOX ROI: X {x1_pct:.0f}% -> {x2_pct:.0f}% | "
                f"Y {y1_pct:.0f}% -> {y2_pct:.0f}% | "
                f"DEADLINE X {deadline:.0f}% "
                "(seleziona il video per vedere anche i pixel)"
            )

    default_model = Path.cwd() / "yolo11n.pt"
    if default_model.exists():
        model_var.set(str(default_model))

    result: dict[str, object] = {}

    def select_video() -> None:
        filename = filedialog.askopenfilename(
            parent=root,
            title="Seleziona il video MP4",
            filetypes=[("Video MP4", "*.mp4"), ("Tutti i file", "*.*")],
        )
        if filename:
            video_var.set(filename)

            capture = cv2.VideoCapture(filename)
            if capture.isOpened():
                video_size["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_size["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                video_size["width"] = None
                video_size["height"] = None
            capture.release()

            update_roi_reference()
            root.lift()
            root.focus_force()

    def select_model() -> None:
        filename = filedialog.askopenfilename(
            parent=root,
            title="Seleziona il modello YOLO",
            filetypes=[("Modello YOLO", "*.pt"), ("Tutti i file", "*.*")],
        )
        if filename:
            model_var.set(filename)
            root.lift()
            root.focus_force()

    def start_analysis() -> None:
        video_text = video_var.get().strip()
        model_text = model_var.get().strip()

        if not video_text:
            messagebox.showwarning(
                "visionTar",
                "Seleziona prima un file video MP4.",
                parent=root,
            )
            return

        if not model_text:
            messagebox.showwarning(
                "visionTar",
                "Seleziona prima un modello YOLO (.pt).",
                parent=root,
            )
            return

        video_path = Path(video_text)
        model_path = Path(model_text)

        if not video_path.exists():
            messagebox.showerror(
                "visionTar",
                f"Il video non esiste:\n{video_path}",
                parent=root,
            )
            return

        if video_path.suffix.lower() != ".mp4":
            messagebox.showwarning(
                "visionTar",
                "Per questa versione seleziona un file .mp4.",
                parent=root,
            )
            return

        if not model_path.exists():
            messagebox.showerror(
                "visionTar",
                f"Il modello non esiste:\n{model_path}",
                parent=root,
            )
            return

        left = roi_left_var.get() / 100.0
        right = roi_right_var.get() / 100.0
        top = roi_top_var.get() / 100.0
        bottom = roi_bottom_var.get() / 100.0
        deadline = deadline_var.get() / 100.0

        if left + right >= 0.90 or top + bottom >= 0.90:
            messagebox.showwarning(
                "visionTar",
                "Area di ispezione troppo stretta: riduci i margini ROI.",
                parent=root,
            )
            return

        result["video"] = video_path
        result["model"] = model_path
        result["test_mode"] = bool(test_mode_var.get())
        result["roi_left"] = left
        result["roi_right"] = right
        result["roi_top"] = top
        result["roi_bottom"] = bottom
        result["deadline"] = deadline
        root.destroy()

    def cancel() -> None:
        root.destroy()

    container = tk.Frame(root, padx=20, pady=20)
    container.pack(fill="both", expand=True)

    tk.Label(
        container,
        text="visionTar - Analisi video con OpenCV + YOLO",
        font=("Segoe UI", 15, "bold"),
    ).pack(pady=(0, 20))

    video_frame = tk.Frame(container)
    video_frame.pack(fill="x", pady=5)

    tk.Button(
        video_frame,
        text="Seleziona video MP4",
        width=22,
        command=select_video,
    ).pack(side="left")

    tk.Entry(
        video_frame,
        textvariable=video_var,
        width=76,
        state="readonly",
    ).pack(side="left", padx=(10, 0), fill="x", expand=True)

    model_frame = tk.Frame(container)
    model_frame.pack(fill="x", pady=5)

    tk.Button(
        model_frame,
        text="Seleziona modello YOLO",
        width=22,
        command=select_model,
    ).pack(side="left")

    tk.Entry(
        model_frame,
        textvariable=model_var,
        width=76,
        state="readonly",
    ).pack(side="left", padx=(10, 0), fill="x", expand=True)

    test_frame = tk.Frame(container)
    test_frame.pack(fill="x", pady=(15, 5))

    tk.Checkbutton(
        test_frame,
        text=(
            "Modalita TEST: salva nel CSV tutte le detection del modello "
            "(utile con yolo11n.pt)"
        ),
        variable=test_mode_var,
    ).pack(side="left")

    roi_frame = tk.LabelFrame(
        container,
        text="Area di ispezione - margini ROI (%)",
        padx=10,
        pady=8,
    )
    roi_frame.pack(fill="x", pady=(12, 8))

    roi_items = [
        ("Sinistra", roi_left_var),
        ("Destra", roi_right_var),
        ("Alto", roi_top_var),
        ("Basso", roi_bottom_var),
        ("Deadline X", deadline_var),
    ]

    for row_index, (label_text, variable) in enumerate(roi_items):
        tk.Label(roi_frame, text=label_text, width=10, anchor="w").grid(
            row=row_index, column=0, sticky="w"
        )
        tk.Scale(
            roi_frame,
            from_=0,
            to=100 if label_text == "Deadline X" else 45,
            orient="horizontal",
            resolution=1,
            variable=variable,
            length=560,
            showvalue=True,
            command=update_roi_reference,
        ).grid(row=row_index, column=1, sticky="ew", padx=(8, 0))

    roi_frame.grid_columnconfigure(1, weight=1)

    tk.Label(
        roi_frame,
        textvariable=roi_reference_var,
        anchor="w",
        justify="left",
        font=("Segoe UI", 9, "bold"),
        wraplength=740,
    ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))

    update_roi_reference()

    tk.Label(
        container,
        text=(
            "I quattro slider restringono l'area analizzata dai bordi del video. "
            "In modalita normale il CSV contiene solo le classi difettose. "
            "Premi Q oppure ESC per interrompere."
        ),
        anchor="w",
        justify="left",
        wraplength=760,
    ).pack(fill="x", pady=(8, 15))

    buttons = tk.Frame(container)
    buttons.pack(fill="x", pady=(10, 0))

    tk.Button(
        buttons,
        text="Avvia analisi",
        width=18,
        command=start_analysis,
    ).pack(side="left")

    tk.Button(
        buttons,
        text="Annulla",
        width=14,
        command=cancel,
    ).pack(side="right")

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.lift()
    root.focus_force()
    root.mainloop()

    if "video" not in result or "model" not in result:
        return None

    return (
        result["video"],
        result["model"],
        bool(result["test_mode"]),
        float(result["roi_left"]),
        float(result["roi_right"]),
        float(result["roi_top"]),
        float(result["roi_bottom"]),
        float(result["deadline"]),
    )


CLASS_COLORS = {
    "buono": (0, 255, 0),      # verde
    "cotto": (0, 0, 255),      # rosso
    "poco": (255, 0, 0),       # blu
    "blob": (0, 255, 255),     # giallo
    "rotto": (255, 255, 255),  # bianco
}


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Calcola Intersection over Union tra due bounding box."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return (inter_area / union) if union > 0 else 0.0


def draw_detection(
    frame,
    label: str,
    confidence: float,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    cx: int,
    cy: int,
    is_defect: bool,
    display_text: str | None = None,
) -> None:
    """Disegna bounding box, baricentro e informazioni della detection."""

    color = CLASS_COLORS.get(label, (0, 0, 255) if is_defect else (0, 255, 0))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 5, color, -1)

    text = display_text or f"{label} | conf={confidence:.2f} | baricentro=({cx},{cy})"

    cv2.putText(
        frame,
        text,
        (max(5, x1), max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        1,
        cv2.LINE_AA,
    )


def process_video(
    video_path: Path,
    model_path: Path,
    test_mode: bool = False,
    roi_left: float = ROI_LEFT_RATIO,
    roi_right: float = ROI_RIGHT_RATIO,
    roi_top: float = ROI_TOP_RATIO,
    roi_bottom: float = ROI_BOTTOM_RATIO,
    deadline_ratio: float = DEADLINE_X_RATIO,
) -> Path:
    """Analizza il video e salva su CSV le coordinate delle detection."""

    print(f"Caricamento modello YOLO: {model_path}")
    print(f"Modalita: {'TEST - tutte le detection' if test_mode else 'NORMALE - soli difetti'}")

    model = YOLO(str(model_path))

    print(f"Apertura video: {video_path}")
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    suffix = "_detections_test.csv" if test_mode else "_difetti.csv"
    output_path = video_path.with_name(f"{video_path.stem}{suffix}")

    frame_number = 0
    saved_rows = 0

    window_name = "visionTar - Q o ESC per terminare"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Slider LIVE: modificano realmente la ROI durante la riproduzione.
    cv2.createTrackbar("Sinistra %", window_name, int(roi_left * 100), 45, lambda _v: None)
    cv2.createTrackbar("Destra %", window_name, int(roi_right * 100), 45, lambda _v: None)
    cv2.createTrackbar("Alto %", window_name, int(roi_top * 100), 45, lambda _v: None)
    cv2.createTrackbar("Basso %", window_name, int(roi_bottom * 100), 45, lambda _v: None)
    cv2.createTrackbar("Deadline X %", window_name, int(deadline_ratio * 100), 100, lambda _v: None)

    tracks = {}
    next_track_id = 1
    recent_final = None
    recent_final_until = 0

    try:
        with output_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "frame",
                    "classe",
                    "confidence",
                    "x_baricentro",
                    "y_baricentro",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "difetto",
                    "track_id",
                    "stato",
                    "responso_finale",
                    "confidence_finale",
                    "quota_difetto",
                    "quota_classe_finale",
                    "soglia_difetto_usata",
                ]
            )

            while True:
                ok, frame = capture.read()

                if not ok:
                    print("Fine del video raggiunta.")
                    break

                frame_number += 1

                frame_height, frame_width = frame.shape[:2]

                # Legge gli slider ad ogni frame: la ROI cambia in tempo reale.
                live_left = cv2.getTrackbarPos("Sinistra %", window_name) / 100.0
                live_right = cv2.getTrackbarPos("Destra %", window_name) / 100.0
                live_top = cv2.getTrackbarPos("Alto %", window_name) / 100.0
                live_bottom = cv2.getTrackbarPos("Basso %", window_name) / 100.0
                live_deadline = cv2.getTrackbarPos("Deadline X %", window_name) / 100.0

                # Evita ROI invalide anche durante la regolazione live.
                if live_left + live_right >= 0.90:
                    live_right = max(0.0, 0.89 - live_left)
                    cv2.setTrackbarPos("Destra %", window_name, int(live_right * 100))
                if live_top + live_bottom >= 0.90:
                    live_bottom = max(0.0, 0.89 - live_top)
                    cv2.setTrackbarPos("Basso %", window_name, int(live_bottom * 100))

                roi_x1 = int(frame_width * live_left)
                roi_x2 = int(frame_width * (1.0 - live_right))
                roi_y1 = int(frame_height * live_top)
                roi_y2 = int(frame_height * (1.0 - live_bottom))
                deadline_x = int(frame_width * live_deadline)
                deadline_x = max(roi_x1 + 5, min(roi_x2 - 5, deadline_x))

                # NMS agnostica rispetto alla classe: se due box di classi
                # diverse descrivono lo stesso oggetto, resta quella piu forte.
                result = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    agnostic_nms=True,
                    verbose=False,
                )[0]

                cv2.rectangle(
                    frame,
                    (roi_x1, roi_y1),
                    (roi_x2, roi_y2),
                    (180, 180, 180),
                    1,
                )
                cv2.putText(
                    frame,
                    (
                        f"AREA DI ISPEZIONE | "
                        f"L {int(live_left * 100)}%  "
                        f"R {int(live_right * 100)}%  "
                        f"A {int(live_top * 100)}%  "
                        f"B {int(live_bottom * 100)}%"
                    ),
                    (roi_x1 + 5, max(20, roi_y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                        class_id = int(box.cls[0].cpu().item())
                        confidence = float(box.conf[0].cpu().item())
                        label = str(model.names[class_id])
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
                            continue

                        detections.append(
                            {
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "cx": cx, "cy": cy,
                                "label": label, "confidence": confidence,
                            }
                        )

                # Tracking leggero per associare lo stesso tarallo tra frame consecutivi.
                unmatched_track_ids = set(tracks.keys())
                max_distance = max(45.0, frame_width * 0.10)

                for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
                    best_id = None
                    best_distance = max_distance

                    best_match_score = -1.0
                    det_bbox = (det["x1"], det["y1"], det["x2"], det["y2"])
                    det_area = max(
                        1,
                        (det["x2"] - det["x1"]) * (det["y2"] - det["y1"]),
                    )
                    hard_max_jump = max(30.0, frame_width * TRACK_MAX_JUMP_RATIO)

                    for track_id in list(unmatched_track_ids):
                        track = tracks[track_id]
                        missed = frame_number - track["last_frame"]
                        if missed > TRACK_MAX_MISSED_FRAMES:
                            continue

                        predicted_cx = track["cx"] + (track["vx"] * missed)
                        predicted_cy = track["cy"] + (track["vy"] * missed)

                        predicted_distance = math.hypot(
                            det["cx"] - predicted_cx,
                            det["cy"] - predicted_cy,
                        )
                        raw_distance = math.hypot(
                            det["cx"] - track["cx"],
                            det["cy"] - track["cy"],
                        )
                        iou = box_iou(det_bbox, track["bbox"])

                        # Un salto incompatibile col moto del nastro spezza il track,
                        # evitando che l'ID venga trasferito a un tarallo vicino.
                        if raw_distance > hard_max_jump and predicted_distance > hard_max_jump:
                            continue

                        size_ratio = min(det_area, track["area"]) / max(det_area, track["area"])
                        if size_ratio < 0.45 and iou < 0.10:
                            continue

                        distance_score = max(
                            0.0,
                            1.0 - (predicted_distance / max_distance),
                        )
                        match_score = (
                            (0.50 * iou)
                            + (0.35 * distance_score)
                            + (0.15 * size_ratio)
                        )

                        if (
                            match_score >= TRACK_MIN_MATCH_SCORE
                            and match_score > best_match_score
                        ):
                            best_match_score = match_score
                            best_id = track_id

                    if best_id is None:
                        best_id = next_track_id
                        next_track_id += 1
                        tracks[best_id] = {
                            "cx": det["cx"],
                            "cy": det["cy"],
                            "prev_cx": None,
                            "bbox": (det["x1"], det["y1"], det["x2"], det["y2"]),
                            "vx": 0.0,
                            "vy": 0.0,
                            "area": max(1, (det["x2"] - det["x1"]) * (det["y2"] - det["y1"])),
                            "last_frame": frame_number,
                            "observations": deque(maxlen=TRACK_OBSERVATION_WINDOW),
                            "final_label": None,
                            "final_confidence": None,
                            "final_frame": None,
                            "final_state": None,
                            "defect_share": None,
                            "final_class_share": None,
                            "defect_threshold_used": None,
                        }
                    else:
                        unmatched_track_ids.discard(best_id)

                    track = tracks[best_id]
                    previous_cx = track["cx"]
                    previous_cy = track["cy"]
                    previous_frame = track["last_frame"]
                    frame_gap = max(1, frame_number - previous_frame)

                    measured_vx = (det["cx"] - previous_cx) / frame_gap
                    measured_vy = (det["cy"] - previous_cy) / frame_gap

                    track["prev_cx"] = previous_cx
                    track["vx"] = (0.65 * track["vx"]) + (0.35 * measured_vx)
                    track["vy"] = (0.65 * track["vy"]) + (0.35 * measured_vy)
                    track["cx"] = det["cx"]
                    track["cy"] = det["cy"]
                    track["bbox"] = det_bbox
                    track["area"] = det_area
                    track["last_frame"] = frame_number

                    # Conserva solo le osservazioni piu recenti prima della deadline,
                    # cosi la decisione non viene contaminata da classificazioni lontane.
                    if track["final_label"] is None:
                        track["observations"].append(
                            (det["label"], det["confidence"], frame_number)
                        )

                    crossed = (
                        track["prev_cx"] is not None
                        and (track["prev_cx"] - deadline_x) * (det["cx"] - deadline_x) <= 0
                        and abs(det["cx"] - track["prev_cx"]) >= TRACK_MIN_CROSSING_DELTA
                    )

                    if track["final_label"] is None and crossed:
                        observations = list(track["observations"])

                        if len(observations) < TRACK_MIN_FINAL_OBSERVATIONS:
                            final_label = "incerto"
                            final_conf = 0.0
                            final_state = "INCERTO"
                        else:
                            weighted_scores = defaultdict(float)
                            confidences = defaultdict(list)

                            # Le osservazioni piu vicine alla deadline pesano di piu.
                            for index, (obs_label, obs_conf, _obs_frame) in enumerate(observations):
                                recency_weight = 0.70 + (0.30 * (index + 1) / len(observations))
                                weighted_scores[obs_label] += obs_conf * recency_weight
                                confidences[obs_label].append(obs_conf)

                            # STADIO 1: BUONO vs DIFETTOSO.
                            good_score = weighted_scores.get("buono", 0.0)
                            defect_score = sum(
                                weighted_scores.get(label, 0.0)
                                for label in DEFECT_CLASSES
                            )
                            binary_total = good_score + defect_score
                            defect_share = (
                                defect_score / binary_total
                                if binary_total > 0
                                else 0.0
                            )

                            # STADIO 2: scegli il difetto prevalente e applica
                            # una soglia piu sensibile solo a cotto/rotto.
                            defect_ranked = sorted(
                                (
                                    (label, weighted_scores.get(label, 0.0))
                                    for label in DEFECT_CLASSES
                                ),
                                key=lambda item: item[1],
                                reverse=True,
                            )
                            candidate_label, top_defect_score = defect_ranked[0]
                            defect_class_share = (
                                top_defect_score / defect_score
                                if defect_score > 0
                                else 0.0
                            )

                            candidate_confidences = confidences[candidate_label]
                            candidate_observations = len(candidate_confidences)
                            candidate_avg_confidence = (
                                sum(candidate_confidences) / candidate_observations
                                if candidate_observations > 0
                                else 0.0
                            )

                            defect_threshold = DEFECT_BINARY_THRESHOLD
                            if (
                                candidate_label in SENSITIVE_DEFECT_CLASSES
                                and candidate_observations >= SENSITIVE_MIN_OBSERVATIONS
                                and candidate_avg_confidence >= SENSITIVE_MIN_AVG_CONFIDENCE
                            ):
                                defect_threshold = DEFECT_BINARY_THRESHOLD_SENSITIVE

                            if defect_share >= defect_threshold and defect_score > 0:
                                final_label = candidate_label
                                final_conf = candidate_avg_confidence

                                if defect_class_share < DEFECT_CLASS_MIN_SHARE:
                                    final_label = "incerto"
                                    final_state = "INCERTO"
                                else:
                                    final_state = "FINALE"
                            else:
                                final_label = "buono"
                                defect_class_share = 0.0
                                final_conf = (
                                    sum(confidences["buono"]) / len(confidences["buono"])
                                    if confidences["buono"]
                                    else 0.0
                                )
                                final_state = "FINALE"

                        track["final_label"] = final_label
                        track["final_confidence"] = final_conf
                        track["final_frame"] = frame_number
                        track["final_state"] = final_state
                        track["defect_share"] = (
                            defect_share if len(observations) >= TRACK_MIN_FINAL_OBSERVATIONS else 0.0
                        )
                        track["final_class_share"] = (
                            defect_class_share
                            if len(observations) >= TRACK_MIN_FINAL_OBSERVATIONS
                            else 0.0
                        )
                        track["defect_threshold_used"] = (
                            defect_threshold
                            if len(observations) >= TRACK_MIN_FINAL_OBSERVATIONS
                            else 0.0
                        )
                        recent_final = (best_id, final_label, final_conf)
                        recent_final_until = frame_number + 20

                    display_label = track["final_label"] or det["label"]
                    display_conf = (
                        track["final_confidence"]
                        if track["final_label"] is not None
                        else det["confidence"]
                    )
                    is_final = track["final_label"] is not None
                    is_uncertain = display_label == "incerto"
                    is_defect = display_label in DEFECT_CLASSES
                    display_text = (
                        f"ID {best_id} | INCERTO"
                        if is_uncertain
                        else (
                            f"ID {best_id} | FINALE {display_label} | conf={display_conf:.2f}"
                            if is_final
                            else f"ID {best_id} | LIVE {display_label} | conf={display_conf:.2f}"
                        )
                    )

                    draw_detection(
                        frame=frame,
                        label=display_label,
                        confidence=display_conf,
                        x1=det["x1"],
                        y1=det["y1"],
                        x2=det["x2"],
                        y2=det["y2"],
                        cx=det["cx"],
                        cy=det["cy"],
                        is_defect=is_defect,
                        display_text=display_text,
                    )

                    if test_mode or is_defect:
                        writer.writerow(
                            [
                                frame_number,
                                det["label"],
                                f"{det['confidence']:.4f}",
                                det["cx"],
                                det["cy"],
                                det["x1"],
                                det["y1"],
                                det["x2"],
                                det["y2"],
                                int(det["label"] in DEFECT_CLASSES),
                                best_id,
                                (
                                    track["final_state"]
                                    if is_final
                                    else "LIVE"
                                ),
                                track["final_label"] or "",
                                (
                                    f"{track['final_confidence']:.4f}"
                                    if track["final_confidence"] is not None
                                    else ""
                                ),
                                (
                                    f"{track['defect_share']:.4f}"
                                    if track["defect_share"] is not None
                                    else ""
                                ),
                                (
                                    f"{track['final_class_share']:.4f}"
                                    if track["final_class_share"] is not None
                                    else ""
                                ),
                                (
                                    f"{track['defect_threshold_used']:.4f}"
                                    if track["defect_threshold_used"] is not None
                                    else ""
                                ),
                            ]
                        )
                        saved_rows += 1

                # Pulisce tracce vecchie.
                stale_ids = [
                    track_id
                    for track_id, track in tracks.items()
                    if frame_number - track["last_frame"] > TRACK_MAX_MISSED_FRAMES
                ]
                for track_id in stale_ids:
                    del tracks[track_id]

                cv2.line(
                    frame,
                    (deadline_x, roi_y1),
                    (deadline_x, roi_y2),
                    (255, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "DEADLINE / RESPONSO",
                    (max(5, deadline_x - 85), min(frame_height - 10, roi_y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                if recent_final is not None and frame_number <= recent_final_until:
                    final_id, final_label, final_conf = recent_final
                    final_color = CLASS_COLORS.get(final_label, (128, 128, 128))
                    cv2.putText(
                        frame,
                        (
                            f"RESPONSO -> ID {final_id}: INCERTO"
                            if final_label == "incerto"
                            else f"RESPONSO FINALE -> ID {final_id}: {final_label.upper()} ({final_conf:.2f})"
                        ),
                        (15, max(55, roi_y1 + 28)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        final_color,
                        2,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    frame,
                    (
                        f"Frame: {frame_number} | "
                        f"CSV: {saved_rows} | "
                        f"{'TEST' if test_mode else 'NORMALE'}"
                    ),
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    print("Analisi interrotta dall'utente.")
                    break

    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Righe salvate nel CSV: {saved_rows}")
    print(f"File CSV chiuso correttamente: {output_path}")

    return output_path


def show_final_message(title: str, text: str, error: bool = False) -> None:
    """Mostra il messaggio finale in una finestra separata."""
    root = tk.Tk()
    root.withdraw()

    if error:
        messagebox.showerror(title, text, parent=root)
    else:
        messagebox.showinfo(title, text, parent=root)

    root.destroy()


def main() -> None:
    selected = choose_inputs()

    if selected is None:
        print("Operazione annullata.")
        return

    (
        video_path,
        model_path,
        test_mode,
        roi_left,
        roi_right,
        roi_top,
        roi_bottom,
        deadline_ratio,
    ) = selected

    try:
        output_path = process_video(
            video_path=video_path,
            model_path=model_path,
            test_mode=test_mode,
            roi_left=roi_left,
            roi_right=roi_right,
            roi_top=roi_top,
            roi_bottom=roi_bottom,
            deadline_ratio=deadline_ratio,
        )
    except Exception as exc:
        print(f"Errore: {exc}")
        show_final_message(
            "visionTar - Errore",
            str(exc),
            error=True,
        )
        return

    print(f"Analisi completata. File risultati: {output_path}")
    show_final_message(
        "visionTar",
        f"Analisi completata.\n\nRisultati salvati in:\n{output_path}",
    )


if __name__ == "__main__":
    main()
