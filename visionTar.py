"""
visionTar.py

Prototipo Capstone per il riconoscimento di taralli da file video MP4.

Versione 2.3
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
from typing import Callable
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from ultralytics import YOLO

from acquisition_config import get_profile


DEFECT_CLASSES = {
    "blob",
    "cotto",
    "poco",
    "rotto",
}

CONFIDENCE_THRESHOLD = 0.35

# Geometria di acquisizione centralizzata. Il profilo standard puo essere
# sostituito dalla dashboard senza modificare le soglie del classificatore.
BASE_ACQUISITION_PROFILE = get_profile()
ROI_LEFT_RATIO = BASE_ACQUISITION_PROFILE.roi_left
ROI_RIGHT_RATIO = BASE_ACQUISITION_PROFILE.roi_right
ROI_TOP_RATIO = BASE_ACQUISITION_PROFILE.roi_top
ROI_BOTTOM_RATIO = BASE_ACQUISITION_PROFILE.roi_bottom
DEADLINE_X_RATIO = BASE_ACQUISITION_PROFILE.deadline_x

DEFAULT_MODEL_PATH = Path(
    "C:/Users/Lello/Documents/GitHub/capstone-vision-quality/"
    "runs/detect/runs/train/taralli_v2_yolo11n/weights/best.pt"
)
DEFAULT_DATASET_PATH = Path(
    "C:/Users/Lello/Documents/GitHub/capstone-vision-quality/taralli_yolo_v2"
)
TRACK_MAX_MISSED_FRAMES = 14
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
TRACK_DEADLINE_RESCUE_RATIO = 0.08
COLOR_INNER_CROP_RATIO = 0.15

# Fusion colore prudenziale: il colore non sostituisce YOLO, ma puo
# rafforzare una candidatura "cotto" o "poco" quando i descrittori del track
# sono coerenti con i profili osservati nel dominio video controllato.
COTTO_MAX_V = 190.0
COTTO_MAX_LAB_L = 168.0
COTTO_MIN_S = 125.0
POCO_MIN_V = 212.0
POCO_MIN_LAB_L = 198.0
POCO_MAX_S = 105.0
COTTO_COLOR_SUPPORTED_THRESHOLD = 0.24
COTTO_COLOR_OVERRIDE_MIN_OBSERVATIONS = 1
COTTO_COLOR_OVERRIDE_MIN_AVG_CONFIDENCE = 0.40
POCO_COLOR_SUPPORTED_THRESHOLD = 0.32
POCO_COLOR_MIN_OBSERVATIONS = 2

# Conferma finale di luminosita per "poco".
# Calibrazione sperimentale derivata dal confronto visivo dei 9 track "poco"
# del video 2: i 5 casi realmente poco cotti mostrano Lab L >= 208.
POCO_FINAL_MIN_LAB_L = 208.0

# Supporto geometrico prudenziale per la classe rotto.
# Le soglie sono iniziali e verranno validate sul benchmark manuale del video 2.
ROTTO_MAX_CIRCULARITY = 0.62
ROTTO_MAX_SOLIDITY = 0.88
ROTTO_MIN_ASPECT = 0.72
ROTTO_MAX_ASPECT = 1.38
ROTTO_MAX_EXTENT = 0.55
ROTTO_MIN_ANOMALY_SIGNALS = 2
ROTTO_SHAPE_SUPPORTED_THRESHOLD = 0.24

# Override geometrico forte: puo recuperare un "rotto" anche quando il voto
# complessivo resta dominato da "buono", ma solo con evidenza molto netta.
ROTTO_STRONG_MIN_ANOMALIES = 3
ROTTO_STRONG_MAX_CIRCULARITY = 0.55
ROTTO_STRONG_MAX_SOLIDITY = 0.82
ROTTO_STRONG_MIN_OBSERVATIONS = 1
ROTTO_STRONG_MIN_AVG_CONFIDENCE = 0.45
ROTTO_STRONG_MIN_SCORE_RATIO = 0.55
ROTTO_STRONG_THRESHOLD = 0.12


def choose_inputs() -> tuple[Path, Path, bool, float, float, float, float, float] | None:
    """Mostra una sola GUI per selezionare video, modello e modalita di prova."""

    root = tk.Tk()
    root.title("visionTar - Selezione input")
    root.geometry("820x670")
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

    if DEFAULT_MODEL_PATH.exists():
        model_var.set(str(DEFAULT_MODEL_PATH))
    else:
        fallback_model = Path.cwd() / "yolo11n.pt"
        if fallback_model.exists():
            model_var.set(str(fallback_model))

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

    dataset_frame = tk.Frame(container)
    dataset_frame.pack(fill="x", pady=(8, 2))

    tk.Label(
        dataset_frame,
        text="Dataset V2:",
        width=22,
        anchor="w",
        font=("Segoe UI", 9, "bold"),
    ).pack(side="left")

    tk.Entry(
        dataset_frame,
        width=76,
        state="normal",
    ).pack(side="left", padx=(10, 0), fill="x", expand=True)

    dataset_entry = dataset_frame.winfo_children()[-1]
    dataset_entry.insert(0, str(DEFAULT_DATASET_PATH))
    dataset_entry.configure(state="readonly")

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


def extract_color_features(frame, x1: int, y1: int, x2: int, y2: int) -> dict[str, float]:
    """Estrae descrittori colore robusti dalla parte interna della box.

    I pixel chiaramente blu vengono esclusi quando possibile per ridurre
    l'influenza del nastro trasportatore. Per ora questi valori sono diagnostici:
    vengono registrati nel CSV ma non modificano ancora il responso finale.
    """
    height, width = frame.shape[:2]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    ix1 = max(0, min(width - 1, int(x1 + box_w * COLOR_INNER_CROP_RATIO)))
    ix2 = max(ix1 + 1, min(width, int(x2 - box_w * COLOR_INNER_CROP_RATIO)))
    iy1 = max(0, min(height - 1, int(y1 + box_h * COLOR_INNER_CROP_RATIO)))
    iy2 = max(iy1 + 1, min(height, int(y2 - box_h * COLOR_INNER_CROP_RATIO)))

    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return {"h": 0.0, "s": 0.0, "v": 0.0, "lab_l": 0.0, "lab_a": 0.0, "lab_b": 0.0}

    b, g, r = cv2.split(crop)
    non_blue_mask = ~((b.astype("float32") > g.astype("float32") * 1.15) & (b.astype("float32") > r.astype("float32") * 1.15))

    pixels = crop[non_blue_mask]
    if pixels.size < 30:
        pixels = crop.reshape(-1, 3)

    pixels_img = pixels.reshape(-1, 1, 3)
    hsv = cv2.cvtColor(pixels_img, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(pixels_img, cv2.COLOR_BGR2LAB).reshape(-1, 3)

    return {
        "h": float(sorted(hsv[:, 0])[len(hsv) // 2]),
        "s": float(sorted(hsv[:, 1])[len(hsv) // 2]),
        "v": float(sorted(hsv[:, 2])[len(hsv) // 2]),
        "lab_l": float(sorted(lab[:, 0])[len(lab) // 2]),
        "lab_a": float(sorted(lab[:, 1])[len(lab) // 2]),
        "lab_b": float(sorted(lab[:, 2])[len(lab) // 2]),
    }


def extract_shape_features(frame, x1: int, y1: int, x2: int, y2: int) -> dict[str, float]:
    """Estrae indicatori geometrici semplici dalla sagoma sul nastro blu."""
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"circularity": 0.0, "solidity": 0.0, "aspect": 0.0, "extent": 0.0}

    b, g, r = cv2.split(crop)
    non_blue = (
        (b.astype("float32") <= g.astype("float32") * 1.15)
        | (b.astype("float32") <= r.astype("float32") * 1.15)
    )
    mask = (non_blue.astype("uint8") * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return {"circularity": 0.0, "solidity": 0.0, "aspect": 0.0, "extent": 0.0}

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    bx, by, bw, bh = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))

    circularity = (
        4.0 * math.pi * area / (perimeter * perimeter)
        if perimeter > 0
        else 0.0
    )
    solidity = area / hull_area if hull_area > 0 else 0.0
    aspect = float(bw) / float(bh) if bh > 0 else 0.0
    extent = area / float(bw * bh) if bw > 0 and bh > 0 else 0.0

    return {
        "circularity": circularity,
        "solidity": solidity,
        "aspect": aspect,
        "extent": extent,
    }


def calculate_final_verdict(observations) -> dict[str, float | str]:
    """Calcola il responso finale a due stadi con supporto colore prudenziale."""
    if len(observations) < TRACK_MIN_FINAL_OBSERVATIONS:
        return {
            "label": "incerto",
            "confidence": 0.0,
            "state": "INCERTO",
            "defect_share": 0.0,
            "class_share": 0.0,
            "threshold": 0.0,
            "color_support": "nessuno",
            "track_v": 0.0,
            "track_s": 0.0,
            "track_lab_l": 0.0,
            "shape_support": "nessuno",
            "shape_anomalies": 0,
            "track_circularity": 0.0,
            "track_solidity": 0.0,
            "track_aspect": 0.0,
            "track_extent": 0.0,
            "strong_shape_override": False,
            "rotto_observations": 0,
            "rotto_avg_confidence": 0.0,
            "rotto_score_ratio": 0.0,
        }

    weighted_scores = defaultdict(float)
    confidences = defaultdict(list)

    for index, obs in enumerate(observations):
        obs_label = obs["label"]
        obs_conf = obs["confidence"]
        recency_weight = 0.70 + (0.30 * (index + 1) / len(observations))
        weighted_scores[obs_label] += obs_conf * recency_weight
        confidences[obs_label].append(obs_conf)

    good_score = weighted_scores.get("buono", 0.0)
    defect_score = sum(weighted_scores.get(label, 0.0) for label in DEFECT_CLASSES)
    binary_total = good_score + defect_score
    defect_share = defect_score / binary_total if binary_total > 0 else 0.0

    defect_ranked = sorted(
        ((label, weighted_scores.get(label, 0.0)) for label in DEFECT_CLASSES),
        key=lambda item: item[1],
        reverse=True,
    )
    candidate_label, top_defect_score = defect_ranked[0]
    class_share = top_defect_score / defect_score if defect_score > 0 else 0.0

    candidate_confidences = confidences[candidate_label]
    candidate_observations = len(candidate_confidences)
    candidate_avg_confidence = (
        sum(candidate_confidences) / candidate_observations
        if candidate_observations > 0
        else 0.0
    )

    # Evidenza YOLO specifica per "rotto", calcolata indipendentemente
    # dalla classe difettosa candidata come prevalente.
    rotto_confidences = confidences["rotto"]
    rotto_observations = len(rotto_confidences)
    rotto_avg_confidence = (
        sum(rotto_confidences) / rotto_observations
        if rotto_observations > 0
        else 0.0
    )
    rotto_score = weighted_scores.get("rotto", 0.0)
    rotto_score_ratio = (
        rotto_score / top_defect_score
        if top_defect_score > 0
        else 0.0
    )

    track_s_values = sorted(
        obs["color"]["s"] for obs in observations if "color" in obs
    )
    track_v_values = sorted(
        obs["color"]["v"] for obs in observations if "color" in obs
    )
    track_l_values = sorted(
        obs["color"]["lab_l"] for obs in observations if "color" in obs
    )
    mid = len(observations) // 2
    track_s = float(track_s_values[mid]) if track_s_values else 0.0
    track_v = float(track_v_values[mid]) if track_v_values else 0.0
    track_lab_l = float(track_l_values[mid]) if track_l_values else 0.0

    shape_circularity_values = sorted(
        obs["shape"]["circularity"] for obs in observations if "shape" in obs
    )
    shape_solidity_values = sorted(
        obs["shape"]["solidity"] for obs in observations if "shape" in obs
    )
    shape_aspect_values = sorted(
        obs["shape"]["aspect"] for obs in observations if "shape" in obs
    )
    shape_extent_values = sorted(
        obs["shape"]["extent"] for obs in observations if "shape" in obs
    )
    shape_mid = len(shape_circularity_values) // 2 if shape_circularity_values else 0
    track_circularity = (
        float(shape_circularity_values[shape_mid]) if shape_circularity_values else 0.0
    )
    track_solidity = (
        float(shape_solidity_values[shape_mid]) if shape_solidity_values else 0.0
    )
    track_aspect = (
        float(shape_aspect_values[shape_mid]) if shape_aspect_values else 0.0
    )
    track_extent = (
        float(shape_extent_values[shape_mid]) if shape_extent_values else 0.0
    )

    color_support = "nessuno"
    color_supports_candidate = False

    if (
        candidate_label == "cotto"
        and track_v <= COTTO_MAX_V
        and track_lab_l <= COTTO_MAX_LAB_L
        and track_s >= COTTO_MIN_S
    ):
        color_support = "cotto"
        color_supports_candidate = True
    elif (
        candidate_label == "poco"
        and candidate_observations >= POCO_COLOR_MIN_OBSERVATIONS
        and track_v >= POCO_MIN_V
        and track_lab_l >= POCO_MIN_LAB_L
        and track_s <= POCO_MAX_S
    ):
        color_support = "poco"
        color_supports_candidate = True

    threshold = DEFECT_BINARY_THRESHOLD
    if (
        candidate_label in SENSITIVE_DEFECT_CLASSES
        and candidate_observations >= SENSITIVE_MIN_OBSERVATIONS
        and candidate_avg_confidence >= SENSITIVE_MIN_AVG_CONFIDENCE
    ):
        threshold = DEFECT_BINARY_THRESHOLD_SENSITIVE

    if color_supports_candidate:
        if candidate_label == "cotto":
            threshold = min(threshold, COTTO_COLOR_SUPPORTED_THRESHOLD)
        elif candidate_label == "poco":
            threshold = min(threshold, POCO_COLOR_SUPPORTED_THRESHOLD)

    shape_support = "nessuno"
    shape_anomalies = 0
    strong_shape_override = False

    # La geometria viene valutata su ogni track, non solo quando "rotto"
    # e gia la prima candidatura YOLO tra i difetti.
    shape_anomalies += int(track_circularity <= ROTTO_MAX_CIRCULARITY)
    shape_anomalies += int(track_solidity <= ROTTO_MAX_SOLIDITY)
    shape_anomalies += int(
        track_aspect <= ROTTO_MIN_ASPECT or track_aspect >= ROTTO_MAX_ASPECT
    )
    shape_anomalies += int(track_extent <= ROTTO_MAX_EXTENT)

    if candidate_label == "rotto" and shape_anomalies >= ROTTO_MIN_ANOMALY_SIGNALS:
        shape_support = "rotto"
        threshold = min(threshold, ROTTO_SHAPE_SUPPORTED_THRESHOLD)

    strong_shape_override = (
        shape_anomalies >= ROTTO_STRONG_MIN_ANOMALIES
        and (
            track_circularity <= ROTTO_STRONG_MAX_CIRCULARITY
            or track_solidity <= ROTTO_STRONG_MAX_SOLIDITY
        )
        and rotto_observations >= ROTTO_STRONG_MIN_OBSERVATIONS
        and rotto_avg_confidence >= ROTTO_STRONG_MIN_AVG_CONFIDENCE
        and rotto_score_ratio >= ROTTO_STRONG_MIN_SCORE_RATIO
        and color_support == "nessuno"
    )

    if strong_shape_override:
        # Secondo canale geometrico: puo ribaltare anche un voto complessivo
        # dominato da "buono", pur richiedendo evidenza YOLO reale di rotto.
        shape_support = "rotto_forte"
        return {
            "label": "rotto",
            "confidence": rotto_avg_confidence,
            "state": "FINALE",
            "defect_share": defect_share,
            "class_share": (
                rotto_score / defect_score if defect_score > 0 else 0.0
            ),
            "threshold": ROTTO_STRONG_THRESHOLD,
            "color_support": color_support,
            "track_v": track_v,
            "track_s": track_s,
            "track_lab_l": track_lab_l,
            "shape_support": shape_support,
            "shape_anomalies": shape_anomalies,
            "track_circularity": track_circularity,
            "track_solidity": track_solidity,
            "track_aspect": track_aspect,
            "track_extent": track_extent,
            "strong_shape_override": True,
            "rotto_observations": rotto_observations,
            "rotto_avg_confidence": rotto_avg_confidence,
            "rotto_score_ratio": rotto_score_ratio,
        }

    # Per "poco" il voto YOLO deve essere confermato anche dalla luminosita.
    # Se il tarallo non e abbastanza chiaro, il responso viene ricondotto a
    # "buono" invece di scartarlo sulla sola evidenza del detector.
    if (
        candidate_label == "poco"
        and defect_share >= threshold
        and track_lab_l < POCO_FINAL_MIN_LAB_L
    ):
        good_conf = (
            sum(confidences["buono"]) / len(confidences["buono"])
            if confidences["buono"]
            else candidate_avg_confidence
        )
        return {
            "label": "buono",
            "confidence": good_conf,
            "state": "FINALE",
            "defect_share": defect_share,
            "class_share": class_share,
            "threshold": threshold,
            "color_support": "poco_non_confermato",
            "track_v": track_v,
            "track_s": track_s,
            "track_lab_l": track_lab_l,
            "shape_support": shape_support,
            "shape_anomalies": shape_anomalies,
            "track_circularity": track_circularity,
            "track_solidity": track_solidity,
            "track_aspect": track_aspect,
            "track_extent": track_extent,
            "strong_shape_override": strong_shape_override,
            "rotto_observations": rotto_observations,
            "rotto_avg_confidence": rotto_avg_confidence,
            "rotto_score_ratio": rotto_score_ratio,
        }

    # Conferma colore forte per "cotto": nel video 2 i tre falsi negativi
    # residui hanno profilo colore coerente con i tre cotto gia riconosciuti.
    # L'override resta vincolato a una vera osservazione YOLO di cotto.
    if (
        candidate_label == "cotto"
        and color_support == "cotto"
        and candidate_observations >= COTTO_COLOR_OVERRIDE_MIN_OBSERVATIONS
        and candidate_avg_confidence >= COTTO_COLOR_OVERRIDE_MIN_AVG_CONFIDENCE
    ):
        return {
            "label": "cotto",
            "confidence": candidate_avg_confidence,
            "state": "FINALE",
            "defect_share": defect_share,
            "class_share": class_share,
            "threshold": COTTO_COLOR_SUPPORTED_THRESHOLD,
            "color_support": "cotto_forte",
            "track_v": track_v,
            "track_s": track_s,
            "track_lab_l": track_lab_l,
            "shape_support": shape_support,
            "shape_anomalies": shape_anomalies,
            "track_circularity": track_circularity,
            "track_solidity": track_solidity,
            "track_aspect": track_aspect,
            "track_extent": track_extent,
            "strong_shape_override": strong_shape_override,
            "rotto_observations": rotto_observations,
            "rotto_avg_confidence": rotto_avg_confidence,
            "rotto_score_ratio": rotto_score_ratio,
        }

    if defect_share >= threshold and defect_score > 0:
        if class_share < DEFECT_CLASS_MIN_SHARE:
            return {
                "label": "incerto",
                "confidence": candidate_avg_confidence,
                "state": "INCERTO",
                "defect_share": defect_share,
                "class_share": class_share,
                "threshold": threshold,
                "color_support": color_support,
                "track_v": track_v,
                "track_s": track_s,
                "track_lab_l": track_lab_l,
                "shape_support": shape_support,
                "shape_anomalies": shape_anomalies,
                "track_circularity": track_circularity,
                "track_solidity": track_solidity,
                "track_aspect": track_aspect,
                "track_extent": track_extent,
                "strong_shape_override": strong_shape_override,
                "rotto_observations": rotto_observations,
                "rotto_avg_confidence": rotto_avg_confidence,
                "rotto_score_ratio": rotto_score_ratio,
            }
        return {
            "label": candidate_label,
            "confidence": candidate_avg_confidence,
            "state": "FINALE",
            "defect_share": defect_share,
            "class_share": class_share,
            "threshold": threshold,
            "color_support": color_support,
            "track_v": track_v,
            "track_s": track_s,
            "track_lab_l": track_lab_l,
            "shape_support": shape_support,
            "shape_anomalies": shape_anomalies,
            "track_circularity": track_circularity,
            "track_solidity": track_solidity,
            "track_aspect": track_aspect,
            "track_extent": track_extent,
            "strong_shape_override": strong_shape_override,
            "rotto_observations": rotto_observations,
            "rotto_avg_confidence": rotto_avg_confidence,
            "rotto_score_ratio": rotto_score_ratio,
        }

    good_conf = (
        sum(confidences["buono"]) / len(confidences["buono"])
        if confidences["buono"]
        else 0.0
    )
    return {
        "label": "buono",
        "confidence": good_conf,
        "state": "FINALE",
        "defect_share": defect_share,
        "class_share": 0.0,
        "threshold": threshold,
        "color_support": color_support,
        "track_v": track_v,
        "track_s": track_s,
        "track_lab_l": track_lab_l,
        "shape_support": shape_support,
        "shape_anomalies": shape_anomalies,
        "track_circularity": track_circularity,
        "track_solidity": track_solidity,
        "track_aspect": track_aspect,
        "track_extent": track_extent,
        "strong_shape_override": strong_shape_override,
        "rotto_observations": rotto_observations,
        "rotto_avg_confidence": rotto_avg_confidence,
        "rotto_score_ratio": rotto_score_ratio,
    }


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
    show_window: bool = True,
    annotated_output_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
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
    final_suffix = "_finali_test.csv" if test_mode else "_finali.csv"
    final_output_path = video_path.with_name(f"{video_path.stem}{final_suffix}")
    defect_suffix = "_scarti_test.csv" if test_mode else "_scarti.csv"
    defect_output_path = video_path.with_name(f"{video_path.stem}{defect_suffix}")
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_number = 0
    saved_rows = 0

    window_name = "visionTar - Q o ESC per terminare"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Slider LIVE: modificano realmente la ROI durante la riproduzione.
        cv2.createTrackbar("Sinistra %", window_name, int(roi_left * 100), 45, lambda _v: None)
        cv2.createTrackbar("Destra %", window_name, int(roi_right * 100), 45, lambda _v: None)
        cv2.createTrackbar("Alto %", window_name, int(roi_top * 100), 45, lambda _v: None)
        cv2.createTrackbar("Basso %", window_name, int(roi_bottom * 100), 45, lambda _v: None)
        cv2.createTrackbar("Deadline X %", window_name, int(deadline_ratio * 100), 100, lambda _v: None)

    annotated_writer = None
    if annotated_output_path is not None:
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_for_writer = video_fps if video_fps > 0 else 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        annotated_writer = cv2.VideoWriter(
            str(annotated_output_path),
            fourcc,
            fps_for_writer,
            (frame_width, frame_height),
        )
        if not annotated_writer.isOpened():
            raise RuntimeError(f"Impossibile creare il video elaborato: {annotated_output_path}")

    tracks = {}
    next_track_id = 1
    recent_final = None
    recent_final_until = 0
    finalized_track_ids = set()

    try:
        with (
            output_path.open(
                mode="w",
                newline="",
                encoding="utf-8",
            ) as csv_file,
            final_output_path.open(
                mode="w",
                newline="",
                encoding="utf-8",
            ) as final_csv_file,
            defect_output_path.open(
                mode="w",
                newline="",
                encoding="utf-8",
            ) as defect_csv_file,
        ):
            writer = csv.writer(csv_file)
            final_writer = csv.writer(final_csv_file)
            defect_writer = csv.writer(defect_csv_file)

            defect_writer.writerow(
                [
                    "track_id",
                    "classe",
                    "confidence",
                    "x_px",
                    "y_px",
                    "x_norm",
                    "y_norm",
                    "frame_finale",
                    "timestamp_s",
                ]
            )

            final_writer.writerow(
                [
                    "track_id",
                    "frame_finale",
                    "stato",
                    "responso_finale",
                    "confidence_finale",
                    "quota_difetto",
                    "quota_classe_finale",
                    "soglia_difetto_usata",
                    "supporto_colore",
                    "track_v_median",
                    "track_s_median",
                    "track_lab_l_median",
                    "supporto_forma",
                    "anomalie_forma",
                    "track_circularity_median",
                    "track_solidity_median",
                    "track_aspect_median",
                    "track_extent_median",
                    "override_forma_forte",
                    "rotto_osservazioni",
                    "rotto_conf_media",
                    "rotto_score_ratio",
                    "x_ultimo",
                    "y_ultimo",
                ]
            )

            def persist_final_track(track_id, track) -> None:
                """Scrive una sola riga per ogni tarallo finalizzato."""
                if track_id in finalized_track_ids or track["final_label"] is None:
                    return

                final_writer.writerow(
                    [
                        track_id,
                        track["final_frame"],
                        track["final_state"],
                        track["final_label"],
                        f"{track['final_confidence']:.4f}",
                        f"{track['defect_share']:.4f}",
                        f"{track['final_class_share']:.4f}",
                        f"{track['defect_threshold_used']:.4f}",
                        track["color_support"] or "",
                        f"{track['track_v']:.2f}",
                        f"{track['track_s']:.2f}",
                        f"{track['track_lab_l']:.2f}",
                        track["shape_support"] or "",
                        track["shape_anomalies"],
                        f"{track['track_circularity']:.4f}",
                        f"{track['track_solidity']:.4f}",
                        f"{track['track_aspect']:.4f}",
                        f"{track['track_extent']:.4f}",
                        int(bool(track["strong_shape_override"])),
                        track["rotto_observations"],
                        f"{track['rotto_avg_confidence']:.4f}",
                        f"{track['rotto_score_ratio']:.4f}",
                        track["cx"],
                        track["cy"],
                    ]
                )
                final_csv_file.flush()

                if track["final_label"] in DEFECT_CLASSES:
                    timestamp_s = (
                        float(track["final_frame"]) / video_fps
                        if video_fps > 0
                        else 0.0
                    )
                    defect_writer.writerow(
                        [
                            track_id,
                            track["final_label"],
                            f"{track['final_confidence']:.4f}",
                            track["cx"],
                            track["cy"],
                            f"{track['cx'] / max(1, frame_width):.6f}",
                            f"{track['cy'] / max(1, frame_height):.6f}",
                            track["final_frame"],
                            f"{timestamp_s:.3f}",
                        ]
                    )
                    defect_csv_file.flush()

                finalized_track_ids.add(track_id)

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
                    "h_median",
                    "s_median",
                    "v_median",
                    "lab_l_median",
                    "lab_a_median",
                    "lab_b_median",
                    "supporto_colore",
                    "track_v_median",
                    "track_s_median",
                    "track_lab_l_median",
                    "circularity",
                    "solidity",
                    "aspect_ratio",
                    "extent",
                    "supporto_forma",
                    "anomalie_forma",
                    "track_circularity_median",
                    "track_solidity_median",
                    "track_aspect_median",
                    "track_extent_median",
                    "override_forma_forte",
                    "rotto_osservazioni",
                    "rotto_conf_media",
                    "rotto_score_ratio",
                ]
            )

            while True:
                ok, frame = capture.read()

                if not ok:
                    print("Fine del video raggiunta.")
                    break

                frame_number += 1

                if progress_callback is not None and (
                    frame_number == 1
                    or frame_number % 5 == 0
                    or (total_frames > 0 and frame_number >= total_frames)
                ):
                    progress_callback(frame_number, total_frames)

                frame_height, frame_width = frame.shape[:2]

                if show_window:
                    # Legge gli slider ad ogni frame: la ROI cambia in tempo reale.
                    live_left = cv2.getTrackbarPos("Sinistra %", window_name) / 100.0
                    live_right = cv2.getTrackbarPos("Destra %", window_name) / 100.0
                    live_top = cv2.getTrackbarPos("Alto %", window_name) / 100.0
                    live_bottom = cv2.getTrackbarPos("Basso %", window_name) / 100.0
                    live_deadline = cv2.getTrackbarPos("Deadline X %", window_name) / 100.0
                else:
                    live_left = roi_left
                    live_right = roi_right
                    live_top = roi_top
                    live_bottom = roi_bottom
                    live_deadline = deadline_ratio

                # Evita ROI invalide anche durante la regolazione live.
                if live_left + live_right >= 0.90:
                    live_right = max(0.0, 0.89 - live_left)
                    if show_window:
                        cv2.setTrackbarPos("Destra %", window_name, int(live_right * 100))
                if live_top + live_bottom >= 0.90:
                    live_bottom = max(0.0, 0.89 - live_top)
                    if show_window:
                        cv2.setTrackbarPos("Basso %", window_name, int(live_bottom * 100))

                roi_x1 = int(frame_width * live_left)
                roi_x2 = int(frame_width * (1.0 - live_right))
                roi_y1 = int(frame_height * live_top)
                roi_y2 = int(frame_height * (1.0 - live_bottom))
                deadline_x = int(frame_width * live_deadline)

                # La deadline deve restare ben dentro la ROI: servono alcuni
                # frame dopo il passaggio della linea per osservare davvero
                # l'attraversamento e non dipendere quasi solo dal rescue.
                # Buffer ridotto ma mantenuto: la deadline resta prima della
                # zona con riflessi/luminosita potenzialmente contaminanti.
                post_deadline_buffer = max(20, int(frame_width * 0.05))
                min_deadline_x = roi_x1 + post_deadline_buffer
                deadline_x = max(min_deadline_x, min(roi_x2 - 5, deadline_x))

                effective_deadline_pct = int(round(deadline_x * 100 / frame_width))
                if show_window and cv2.getTrackbarPos("Deadline X %", window_name) != effective_deadline_pct:
                    cv2.setTrackbarPos(
                        "Deadline X %",
                        window_name,
                        effective_deadline_pct,
                    )

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

                        color_features = extract_color_features(frame, x1, y1, x2, y2)
                        detections.append(
                            {
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "cx": cx, "cy": cy,
                                "label": label, "confidence": confidence,
                                "color": color_features,
                                "shape": extract_shape_features(frame, x1, y1, x2, y2),
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
                            "color_support": None,
                            "track_v": None,
                            "track_s": None,
                            "track_lab_l": None,
                            "shape_support": None,
                            "shape_anomalies": None,
                            "track_circularity": None,
                            "track_solidity": None,
                            "track_aspect": None,
                            "track_extent": None,
                            "strong_shape_override": None,
                            "rotto_observations": None,
                            "rotto_avg_confidence": None,
                            "rotto_score_ratio": None,
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
                            {
                                "label": det["label"],
                                "confidence": det["confidence"],
                                "frame": frame_number,
                                "color": det["color"],
                                "shape": det["shape"],
                            }
                        )

                    crossed = (
                        track["prev_cx"] is not None
                        and (track["prev_cx"] - deadline_x) * (det["cx"] - deadline_x) <= 0
                        and abs(det["cx"] - track["prev_cx"]) >= TRACK_MIN_CROSSING_DELTA
                    )

                    if track["final_label"] is None and crossed:
                        verdict = calculate_final_verdict(list(track["observations"]))
                        final_label = str(verdict["label"])
                        final_conf = float(verdict["confidence"])
                        final_state = str(verdict["state"])

                        track["final_label"] = final_label
                        track["final_confidence"] = final_conf
                        track["final_frame"] = frame_number
                        track["final_state"] = final_state
                        track["defect_share"] = float(verdict["defect_share"])
                        track["final_class_share"] = float(verdict["class_share"])
                        track["defect_threshold_used"] = float(verdict["threshold"])
                        track["color_support"] = str(verdict["color_support"])
                        track["track_v"] = float(verdict["track_v"])
                        track["track_s"] = float(verdict["track_s"])
                        track["track_lab_l"] = float(verdict["track_lab_l"])
                        track["shape_support"] = str(verdict["shape_support"])
                        track["shape_anomalies"] = int(verdict["shape_anomalies"])
                        track["track_circularity"] = float(verdict["track_circularity"])
                        track["track_solidity"] = float(verdict["track_solidity"])
                        track["track_aspect"] = float(verdict["track_aspect"])
                        track["track_extent"] = float(verdict["track_extent"])
                        track["strong_shape_override"] = bool(verdict["strong_shape_override"])
                        track["rotto_observations"] = int(verdict["rotto_observations"])
                        track["rotto_avg_confidence"] = float(verdict["rotto_avg_confidence"])
                        track["rotto_score_ratio"] = float(verdict["rotto_score_ratio"])
                        persist_final_track(best_id, track)
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
                                f"{det['color']['h']:.2f}",
                                f"{det['color']['s']:.2f}",
                                f"{det['color']['v']:.2f}",
                                f"{det['color']['lab_l']:.2f}",
                                f"{det['color']['lab_a']:.2f}",
                                f"{det['color']['lab_b']:.2f}",
                                track["color_support"] or "",
                                (
                                    f"{track['track_v']:.2f}"
                                    if track["track_v"] is not None
                                    else ""
                                ),
                                (
                                    f"{track['track_s']:.2f}"
                                    if track["track_s"] is not None
                                    else ""
                                ),
                                (
                                    f"{track['track_lab_l']:.2f}"
                                    if track["track_lab_l"] is not None
                                    else ""
                                ),
                                f"{det['shape']['circularity']:.4f}",
                                f"{det['shape']['solidity']:.4f}",
                                f"{det['shape']['aspect']:.4f}",
                                f"{det['shape']['extent']:.4f}",
                                track["shape_support"] or "",
                                (
                                    track["shape_anomalies"]
                                    if track["shape_anomalies"] is not None
                                    else ""
                                ),
                                (
                                    f"{track['track_circularity']:.4f}"
                                    if track["track_circularity"] is not None
                                    else ""
                                ),
                                (
                                    f"{track['track_solidity']:.4f}"
                                    if track["track_solidity"] is not None
                                    else ""
                                ),
                                (
                                    f"{track['track_aspect']:.4f}"
                                    if track["track_aspect"] is not None
                                    else ""
                                ),
                                (
                                    f"{track['track_extent']:.4f}"
                                    if track["track_extent"] is not None
                                    else ""
                                ),
                                (
                                    int(track["strong_shape_override"])
                                    if track["strong_shape_override"] is not None
                                    else ""
                                ),
                                (
                                    track["rotto_observations"]
                                    if track["rotto_observations"] is not None
                                    else ""
                                ),
                                (
                                    f"{track['rotto_avg_confidence']:.4f}"
                                    if track["rotto_avg_confidence"] is not None
                                    else ""
                                ),
                                (
                                    f"{track['rotto_score_ratio']:.4f}"
                                    if track["rotto_score_ratio"] is not None
                                    else ""
                                ),
                            ]
                        )
                        saved_rows += 1

                # Pulisce tracce vecchie. Se un track si perde poco prima
                # della deadline ma il moto previsto lo porta oltre la linea,
                # recupera il responso invece di perderlo.
                stale_ids = [
                    track_id
                    for track_id, track in tracks.items()
                    if frame_number - track["last_frame"] > TRACK_MAX_MISSED_FRAMES
                ]
                for track_id in stale_ids:
                    track = tracks[track_id]

                    if track["final_label"] is None:
                        missed = frame_number - track["last_frame"]
                        predicted_cx = track["cx"] + (track["vx"] * missed)
                        rescue_zone = max(20.0, frame_width * TRACK_DEADLINE_RESCUE_RATIO)
                        moving_toward_deadline = (
                            (deadline_x - track["cx"]) * track["vx"] > 0
                        )
                        predicted_cross = (
                            (track["cx"] - deadline_x)
                            * (predicted_cx - deadline_x)
                            <= 0
                        )

                        if (
                            abs(track["cx"] - deadline_x) <= rescue_zone
                            and moving_toward_deadline
                            and predicted_cross
                        ):
                            verdict = calculate_final_verdict(list(track["observations"]))
                            track["final_label"] = str(verdict["label"])
                            track["final_confidence"] = float(verdict["confidence"])
                            track["final_frame"] = frame_number
                            track["final_state"] = "RECUPERO_" + str(verdict["state"])
                            track["defect_share"] = float(verdict["defect_share"])
                            track["final_class_share"] = float(verdict["class_share"])
                            track["defect_threshold_used"] = float(verdict["threshold"])
                            track["color_support"] = str(verdict["color_support"])
                            track["track_v"] = float(verdict["track_v"])
                            track["track_s"] = float(verdict["track_s"])
                            track["track_lab_l"] = float(verdict["track_lab_l"])
                            track["shape_support"] = str(verdict["shape_support"])
                            track["shape_anomalies"] = int(verdict["shape_anomalies"])
                            track["track_circularity"] = float(verdict["track_circularity"])
                            track["track_solidity"] = float(verdict["track_solidity"])
                            track["track_aspect"] = float(verdict["track_aspect"])
                            track["track_extent"] = float(verdict["track_extent"])
                            track["strong_shape_override"] = bool(verdict["strong_shape_override"])
                            track["rotto_observations"] = int(verdict["rotto_observations"])
                            track["rotto_avg_confidence"] = float(verdict["rotto_avg_confidence"])
                            track["rotto_score_ratio"] = float(verdict["rotto_score_ratio"])
                            persist_final_track(track_id, track)
                            recent_final = (
                                track_id,
                                track["final_label"],
                                track["final_confidence"],
                            )
                            recent_final_until = frame_number + 20

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

                if annotated_writer is not None:
                    annotated_writer.write(frame)

                if show_window:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (27, ord("q")):
                        print("Analisi interrotta dall'utente.")
                        break

    finally:
        capture.release()
        if annotated_writer is not None:
            annotated_writer.release()
        if show_window:
            cv2.destroyAllWindows()

    print(f"Righe salvate nel CSV detection: {saved_rows}")
    print(f"File detection: {output_path}")
    print(f"Responsi finali unici: {len(finalized_track_ids)}")
    print(f"File finali: {final_output_path}")
    print(f"Eventi scarto: {defect_output_path}")

    if progress_callback is not None:
        progress_callback(frame_number, total_frames)

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
