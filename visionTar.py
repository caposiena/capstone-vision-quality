"""
visionTar.py

Prototipo Capstone per il riconoscimento di taralli da file video MP4.

Versione 0.2
- una sola GUI iniziale;
- selezione del file video MP4;
- selezione del modello YOLO .pt;
- pulsante esplicito "Avvia analisi";
- acquisizione frame tramite OpenCV;
- object detection tramite Ultralytics YOLO;
- visualizzazione di bounding box e baricentro;
- registrazione CSV delle coordinate dei soli taralli classificati come difettosi;
- chiusura ordinata di video, finestra OpenCV e file CSV.

Dipendenze:
    pip install opencv-python ultralytics

Nota:
Le classi considerate difettose sono definite in DEFECT_CLASSES.
Dopo l'addestramento definitivo del modello YOLO, aggiornare questo insieme
in modo che corrisponda esattamente ai nomi delle classi del dataset.
"""

from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from ultralytics import YOLO


DEFECT_CLASSES = {
    "poco_cotto",
    "troppo_cotto",
    "forma_anomala",
    "sovrapposto",
    "difettoso",
    "scarto",
}

CONFIDENCE_THRESHOLD = 0.35


def choose_inputs() -> tuple[Path, Path] | None:
    """
    Mostra una sola finestra GUI per selezionare video e modello.

    Restituisce:
        (video_path, model_path) se l'utente avvia l'analisi;
        None se la finestra viene chiusa o l'operazione viene annullata.
    """

    root = tk.Tk()
    root.title("visionTar - Selezione input")
    root.geometry("720x300")
    root.resizable(False, False)

    video_var = tk.StringVar()
    model_var = tk.StringVar()

    # Se il modello di prova esiste nella cartella corrente, lo proponiamo.
    default_model = Path.cwd() / "yolo11n.pt"
    if default_model.exists():
        model_var.set(str(default_model))

    result: dict[str, Path] = {}

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

        result["video"] = video_path
        result["model"] = model_path
        root.destroy()

    def cancel() -> None:
        root.destroy()

    container = tk.Frame(root, padx=20, pady=20)
    container.pack(fill="both", expand=True)

    title = tk.Label(
        container,
        text="visionTar - Analisi video con OpenCV + YOLO",
        font=("Segoe UI", 15, "bold"),
    )
    title.pack(pady=(0, 20))

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
        width=70,
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
        width=70,
        state="readonly",
    ).pack(side="left", padx=(10, 0), fill="x", expand=True)

    info = tk.Label(
        container,
        text=(
            "Dopo Avvia analisi si aprira la finestra OpenCV. "
            "Premi Q oppure ESC per interrompere."
        ),
        anchor="w",
    )
    info.pack(fill="x", pady=(18, 10))

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

    return result["video"], result["model"]


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

    color = (0, 0, 255) if is_defect else (0, 255, 0)

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


def process_video(video_path: Path, model_path: Path) -> Path:
    """
    Analizza il video e salva su CSV le coordinate dei taralli difettosi.

    Il CSV viene creato accanto al video con suffisso "_difetti.csv".
    """

    print(f"Caricamento modello YOLO: {model_path}")
    model = YOLO(str(model_path))

    print(f"Apertura video: {video_path}")
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    output_path = video_path.with_name(f"{video_path.stem}_difetti.csv")
    frame_number = 0

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
                ]
            )

            while True:
                ok, frame = capture.read()

                if not ok:
                    print("Fine del video raggiunta.")
                    break

                frame_number += 1

                result = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                )[0]

                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].cpu().tolist(),
                        )

                        class_id = int(box.cls[0].cpu().item())
                        confidence = float(box.conf[0].cpu().item())
                        label = str(model.names[class_id])

                        # In questa prima versione usiamo il centro geometrico
                        # della bounding box come baricentro operativo.
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

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

                        if is_defect:
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
                                ]
                            )

                cv2.putText(
                    frame,
                    f"Frame: {frame_number}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
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

    print(f"File CSV chiuso correttamente: {output_path}")
    return output_path


def show_final_message(title: str, text: str, error: bool = False) -> None:
    """Mostra il messaggio finale in una finestra separata e controllata."""
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

    video_path, model_path = selected

    try:
        output_path = process_video(
            video_path=video_path,
            model_path=model_path,
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
