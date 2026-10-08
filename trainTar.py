"""
trainTar.py

Primo training pilota YOLO per il progetto visionTar.

Uso tipico:
    python trainTar.py --data dataset/data.yaml

Il dataset deve essere estratto localmente nella cartella:
    dataset/

Il training produce i risultati nella cartella:
    runs/train/taralli_pilot/

Il modello migliore sara disponibile in:
    runs/train/taralli_pilot/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training YOLO custom per il dataset taralli."
    )
    parser.add_argument(
        "--data",
        default="dataset/data.yaml",
        help="Percorso al file data.yaml del dataset.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Modello YOLO di partenza.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Numero di epoche di training.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Dimensione immagine per il training.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Dispositivo di calcolo. Default: "cpu". In futuro: "0" per GPU CUDA.',
    )
    parser.add_argument(
        "--name",
        default="taralli_pilot",
        help="Nome della run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {data_path.resolve()}\n"
            "Estrai prima taralli_yolo_pilot.zip nella cartella dataset/."
        )

    print("=== visionTar - primo training YOLO ===")
    print(f"Dataset : {data_path}")
    print(f"Modello : {args.model}")
    print(f"Epoche  : {args.epochs}")
    print(f"Img size: {args.imgsz}")
    print(f"Batch   : {args.batch}")
    print(f"Device  : {args.device}")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/train",
        name=args.name,
        exist_ok=False,
        plots=True,
        verbose=True,
    )

    run_dir = Path("runs/train") / args.name
    best_model = run_dir / "weights" / "best.pt"

    print("\n=== TRAINING COMPLETATO ===")
    print(f"Risultati: {run_dir.resolve()}")
    print(f"Best model atteso: {best_model.resolve()}")


if __name__ == "__main__":
    main()
