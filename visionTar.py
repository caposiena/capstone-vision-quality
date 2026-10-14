"""
visionTar.py

Prototipo Capstone per il riconoscimento di taralli da file video MP4.

Versione 0.6
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


def choose_inputs() -> tuple[Path, Path, bool, float, float, float, float] | None:
    """Mostra una sola GUI per selezionare video, modello e modalita di prova."""

    root = tk.Tk()
    root.title("visionTar - Selezione input")
    root.geometry("820x560")
    root.resizable(False, False)

    video_var = tk.StringVar()
    model_var = tk.StringVar()
    test_mode_var = tk.BooleanVar(value=True)
    roi_left_var = tk.DoubleVar(value=ROI_LEFT_RATIO * 100)
    roi_right_var = tk.DoubleVar(value=ROI_RIGHT_RATIO * 100)
    roi_top_var = tk.DoubleVar(value=ROI_TOP_RATIO * 100)
    roi_bottom_var = tk.DoubleVar(value=ROI_BOTTOM_RATIO * 100)

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
    ]

    for row_index, (label_text, variable) in enumerate(roi_items):
        tk.Label(roi_frame, text=label_text, width=10, anchor="w").grid(
            row=row_index, column=0, sticky="w"
        )
        tk.Scale(
            roi_frame,
            from_=0,
            to=45,
            orient="horizontal",
            resolution=1,
            variable=variable,
            length=560,
            showvalue=True,
        ).grid(row=row_index, column=1, sticky="ew", padx=(8, 0))

    roi_frame.grid_columnconfigure(1, weight=1)

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
    )


CLASS_COLORS = {
    "buono": (0, 255, 0),      # verde
    "cotto": (0, 0, 255),      # rosso
    "poco": (255, 0, 0),       # blu
    "blob": (0, 255, 255),     # giallo
    "rotto": (255, 255, 255),  # bianco
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
) -> None:
    """Disegna bounding box, baricentro e informazioni della detection."""

    color = CLASS_COLORS.get(label, (0, 0, 255) if is_defect else (0, 255, 0))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 5, color, -1)

    text = f"{label} | conf={confidence:.2f} | baricentro=({cx},{cy})"

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
                ]
            )

            while True:
                ok, frame = capture.read()

                if not ok:
                    print("Fine del video raggiunta.")
                    break

                frame_number += 1

                frame_height, frame_width = frame.shape[:2]
                roi_x1 = int(frame_width * roi_left)
                roi_x2 = int(frame_width * (1.0 - roi_right))
                roi_y1 = int(frame_height * roi_top)
                roi_y2 = int(frame_height * (1.0 - roi_bottom))

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
                    "AREA DI ISPEZIONE",
                    (roi_x1 + 5, max(20, roi_y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].cpu().tolist(),
                        )

                        class_id = int(box.cls[0].cpu().item())
                        confidence = float(box.conf[0].cpu().item())
                        label = str(model.names[class_id])

                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # Considera solo oggetti il cui baricentro e dentro
                        # l'area centrale di ispezione.
                        if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
                            continue

                        is_defect = label in DEFECT_CLASSES

                        draw_detection(
                            frame=frame,
                            label=label,
                            confidence=confidence,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            cx=cx,
                            cy=cy,
                            is_defect=is_defect,
                        )

                        if test_mode or is_defect:
                            writer.writerow(
                                [
                                    frame_number,
                                    label,
                                    f"{confidence:.4f}",
                                    cx,
                                    cy,
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    int(is_defect),
                                ]
                            )
                            saved_rows += 1

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

                cv2.imshow(
                    "visionTar - Q o ESC per terminare",
                    frame,
                )

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
