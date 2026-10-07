"""
visionTar.py

Prototipo Capstone per il riconoscimento di taralli da file video MP4.

Funzioni principali:
- selezione GUI del file .mp4;
- selezione GUI del modello YOLO addestrato (.pt);
- acquisizione frame tramite OpenCV;
- object detection tramite Ultralytics YOLO;
- visualizzazione delle bounding box e del baricentro;
- registrazione su CSV delle coordinate (x, y) dei soli taralli
  classificati come difettosi;
- chiusura ordinata del file CSV al termine del video.

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


# Classi considerate scarto.
# Aggiornare questi nomi dopo la definizione definitiva del dataset YOLO.
DEFECT_CLASSES = {
    "poco_cotto",
    "troppo_cotto",
    "forma_anomala",
    "sovrapposto",
    "difettoso",
    "scarto",
}

CONFIDENCE_THRESHOLD = 0.35


def select_video_file() -> Path | None:
    """Apre una GUI per selezionare il file video MP4."""
    root = tk.Tk()
    root.withdraw()
    root.update()

    filename = filedialog.askopenfilename(
        title="Seleziona il video MP4",
        filetypes=[("Video MP4", "*.mp4"), ("Tutti i file", "*.*")],
    )

    root.destroy()

    if not filename:
        return None

    return Path(filename)


def select_model_file() -> Path | None:
    """Apre una GUI per selezionare il modello YOLO addestrato."""
    root = tk.Tk()
    root.withdraw()
    root.update()

    filename = filedialog.askopenfilename(
        title="Seleziona il modello YOLO",
        filetypes=[("Modello YOLO", "*.pt"), ("Tutti i file", "*.*")],
    )

    root.destroy()

    if not filename:
        return None

    return Path(filename)


def show_message(title: str, text: str) -> None:
    """Visualizza un semplice messaggio GUI."""
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, text)
    root.destroy()


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

    # Rosso per difetto, verde per prodotto non classificato come difettoso.
    color = (0, 0, 255) if is_defect else (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 5, color, -1)

    text = (
        f"{label} | conf={confidence:.2f} | "
        f"baricentro=({cx},{cy})"
    )

    text_y = max(20, y1 - 8)
    cv2.putText(
        frame,
        text,
        (max(5, x1), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        1,
        cv2.LINE_AA,
    )


def process_video(video_path: Path, model_path: Path) -> Path:
    """
    Analizza il video e salva su CSV le coordinate dei taralli difettosi.

    Il file CSV viene creato accanto al video con suffisso "_difetti.csv".
    """

    model = YOLO(str(model_path))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    output_path = video_path.with_name(
        f"{video_path.stem}_difetti.csv"
    )

    frame_number = 0

    try:
        # Il context manager garantisce la chiusura del file in scrittura
        # sia al termine naturale del video sia in caso di interruzione.
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

                        # Il baricentro viene approssimato con il centro
                        # geometrico della bounding box restituita da YOLO.
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

                        # Salviamo soltanto le posizioni dei prodotti
                        # classificati come difettosi.
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
                    "visionTar - ESC o Q per terminare",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    finally:
        capture.release()
        cv2.destroyAllWindows()

    return output_path


def main() -> None:
    video_path = select_video_file()
    if video_path is None:
        print("Nessun video selezionato.")
        return

    model_path = select_model_file()
    if model_path is None:
        print("Nessun modello YOLO selezionato.")
        return

    try:
        output_path = process_video(
            video_path=video_path,
            model_path=model_path,
        )
    except Exception as exc:
        print(f"Errore: {exc}")
        show_message(
            "visionTar - Errore",
            str(exc),
        )
        return

    print(f"Analisi completata. File risultati: {output_path}")
    show_message(
        "visionTar",
        f"Analisi completata.\n\nRisultati salvati in:\n{output_path}",
    )


if __name__ == "__main__":
    main()
