"""
testTar.py

Valuta il modello YOLO custom sul test set e salva immagini annotate.

Uso tipico:
    python testTar.py

Lo script:
1. trova automaticamente il best.pt piu recente sotto runs/;
2. esegue la validazione sullo split test;
3. salva le immagini annotate in runs/test/taralli_test;
4. crea detections.csv con le predizioni.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test del modello YOLO custom sui taralli."
    )
    parser.add_argument(
        "--data",
        default="dataset/data.yaml",
        help="Percorso al file data.yaml.",
    )
    parser.add_argument(
        "--images",
        default="dataset/images/test",
        help="Cartella delle immagini di test.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Percorso esplicito a best.pt. Se omesso viene trovato automaticamente.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Soglia confidence per le immagini annotate.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Dimensione immagine per inferenza.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Dispositivo di calcolo. Default: "cpu".',
    )
    return parser.parse_args()


def find_latest_best_model() -> Path:
    candidates = list(Path("runs").rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "Nessun best.pt trovato sotto la cartella runs/. "
            "Esegui prima trainTar.py oppure usa --model."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    images_dir = Path(args.images)

    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml non trovato: {data_path.resolve()}")
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Cartella immagini test non trovata: {images_dir.resolve()}"
        )

    model_path = Path(args.model) if args.model else find_latest_best_model()
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path.resolve()}")

    output_dir = Path("runs/test/taralli_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "detections.csv"

    print("=== visionTar - Gate 4 test set ===")
    print(f"Modello : {model_path.resolve()}")
    print(f"Dataset : {data_path.resolve()}")
    print(f"Immagini: {images_dir.resolve()}")
    print(f"Output  : {output_dir.resolve()}")

    model = YOLO(str(model_path))

    print("\n=== METRICHE SUL TEST SET ===")
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        verbose=True,
        plots=False,
    )

    try:
        print(f"mAP50    : {metrics.box.map50:.4f}")
        print(f"mAP50-95 : {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall   : {metrics.box.mr:.4f}")
    except AttributeError:
        print("Metriche aggregate completate. Vedi output Ultralytics sopra.")

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError("Nessuna immagine trovata nella cartella test.")

    total_detections = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "immagine",
                "classe",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "x_centro",
                "y_centro",
            ]
        )

        for image_path in image_paths:
            results = model.predict(
                source=str(image_path),
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            result = results[0]
            annotated = result.plot()
            output_image = output_dir / image_path.name
            cv2.imwrite(str(output_image), annotated)

            image_count = 0

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    label = result.names[cls_id]
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    writer.writerow(
                        [
                            image_path.name,
                            label,
                            f"{confidence:.4f}",
                            f"{x1:.1f}",
                            f"{y1:.1f}",
                            f"{x2:.1f}",
                            f"{y2:.1f}",
                            f"{cx:.1f}",
                            f"{cy:.1f}",
                        ]
                    )
                    image_count += 1
                    total_detections += 1

            print(f"{image_path.name}: {image_count} rilevamenti")

    print("\n=== TEST COMPLETATO ===")
    print(f"Immagini analizzate: {len(image_paths)}")
    print(f"Rilevamenti totali : {total_detections}")
    print(f"Immagini annotate  : {output_dir.resolve()}")
    print(f"CSV predizioni     : {csv_path.resolve()}")


if __name__ == "__main__":
    main()
