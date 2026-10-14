"""
buildDatasetV2.py

Ricostruisce localmente il dataset YOLO V2 direttamente dall'export
Label Studio YOLOv8 completo, evitando download intermedi da ChatGPT.

Uso:
    python buildDatasetV2.py --zip "C:\\Users\\Lello\\Downloads\\project-1-at-2026-10-14-15-40-01a2ffa6.zip"

Output:
    taralli_yolo_v2/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Costruisce il dataset YOLO detection V2 dall'export Label Studio."
    )
    parser.add_argument("--zip", required=True, help="Percorso allo ZIP Label Studio completo.")
    parser.add_argument(
        "--output",
        default="taralli_yolo_v2",
        help="Cartella di output. Default: taralli_yolo_v2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed deterministico per lo split validation.",
    )
    return parser.parse_args()


def parse_label(path: Path):
    objects = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        values = line.split()
        if len(values) != 9:
            raise ValueError(
                f"{path.name}:{line_number} - attesi 9 valori, trovati {len(values)}"
            )
        cls = int(float(values[0]))
        coords = list(map(float, values[1:]))
        xs = coords[0::2]
        ys = coords[1::2]
        objects.append((cls, xs, ys))
    return objects


def to_detection_lines(label_path: Path) -> list[str]:
    output = []
    for cls, xs, ys in parse_label(label_path):
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xc = (xmin + xmax) / 2.0
        yc = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        values = [xc, yc, width, height]
        if width <= 0 or height <= 0:
            raise ValueError(f"Bounding box degenerata in {label_path.name}: {values}")
        if not all(-1e-6 <= value <= 1.0 + 1e-6 for value in values):
            raise ValueError(f"Bounding box fuori intervallo in {label_path.name}: {values}")

        values = [min(1.0, max(0.0, value)) for value in values]
        output.append(f"{cls} " + " ".join(f"{value:.6f}" for value in values))
    return output


def class_count(stems: list[str], label_by_stem: dict[str, Path]) -> Counter:
    counts = Counter()
    for stem in stems:
        for cls, _, _ in parse_label(label_by_stem[stem]):
            counts[cls] += 1
    return counts


def choose_validation(
    new_stems: list[str],
    label_by_stem: dict[str, Path],
    class_total: int,
    val_size: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    overall = class_count(new_stems, label_by_stem)
    target = {
        cls: overall[cls] * val_size / len(new_stems)
        for cls in range(class_total)
    }

    best_score = None
    best_sample = None

    iterations = min(20000, max(2000, len(new_stems) * 200))
    for _ in range(iterations):
        sample = rng.sample(new_stems, val_size)
        counts = class_count(sample, label_by_stem)
        score = 0.0

        for cls in range(class_total):
            denominator = max(target[cls], 1.0)
            score += ((counts[cls] - target[cls]) / denominator) ** 2
            if counts[cls] == 0:
                score += 100.0

        if best_score is None or score < best_score:
            best_score = score
            best_sample = sample

    assert best_sample is not None
    return sorted(best_sample)


def main() -> None:
    args = parse_args()
    source_zip = Path(args.zip).expanduser().resolve()
    output_dir = Path(args.output).resolve()

    if not source_zip.exists():
        raise FileNotFoundError(f"ZIP non trovato: {source_zip}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    with tempfile.TemporaryDirectory(prefix="visiontar_v2_") as tmp:
        extracted = Path(tmp) / "export"
        extracted.mkdir(parents=True)

        with zipfile.ZipFile(source_zip, "r") as archive:
            archive.extractall(extracted)

        classes_file = extracted / "classes.txt"
        images_dir = extracted / "images"
        labels_dir = extracted / "labels"

        if not classes_file.exists() or not images_dir.exists() or not labels_dir.exists():
            raise RuntimeError(
                "Export non riconosciuto: servono classes.txt, images/ e labels/."
            )

        classes = [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        images = [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        labels = list(labels_dir.glob("*.txt"))

        image_by_stem = {path.stem: path for path in images}
        label_by_stem = {path.stem: path for path in labels}

        missing_images = sorted(set(label_by_stem) - set(image_by_stem))
        missing_labels = sorted(set(image_by_stem) - set(label_by_stem))
        if missing_images or missing_labels:
            raise RuntimeError(
                f"Corrispondenza immagini/label non completa. "
                f"Label senza immagine: {len(missing_images)}; "
                f"immagini senza label: {len(missing_labels)}"
            )

        all_stems = sorted(image_by_stem)

        # La sessione del primo pilot (08/10) viene tenuta completamente fuori
        # da train/val e usata come test indipendente.
        test_stems = sorted(
            stem for stem in all_stems if "WhatsApp_Image_2026-10-08" in stem
        )
        new_stems = sorted(set(all_stems) - set(test_stems))

        if not test_stems:
            raise RuntimeError(
                "Non trovo le immagini del pilot 08/10 da usare come test indipendente."
            )
        if len(new_stems) < 10:
            raise RuntimeError("Troppo poche immagini nuove per train/validation.")

        val_size = max(1, round(len(new_stems) * 0.20))
        val_stems = choose_validation(
            new_stems,
            label_by_stem,
            len(classes),
            val_size,
            args.seed,
        )
        train_stems = sorted(set(new_stems) - set(val_stems))

        split_map = {
            "train": train_stems,
            "val": val_stems,
            "test": test_stems,
        }

        for split in split_map:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        manifest = []

        for split, stems in split_map.items():
            for stem in stems:
                image = image_by_stem[stem]
                label = label_by_stem[stem]

                shutil.copy2(image, output_dir / "images" / split / image.name)

                detection_lines = to_detection_lines(label)
                output_label = output_dir / "labels" / split / f"{stem}.txt"
                output_label.write_text(
                    "\n".join(detection_lines) + ("\n" if detection_lines else ""),
                    encoding="utf-8",
                )

                manifest.append(
                    {
                        "split": split,
                        "image": image.name,
                        "label": f"{stem}.txt",
                    }
                )

        # Percorso assoluto per evitare ambiguita di risoluzione in Ultralytics.
        dataset_path = output_dir.as_posix()

        yaml_lines = [
            f"path: {dataset_path}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
        ]
        yaml_lines.extend(
            f"  {index}: {name}" for index, name in enumerate(classes)
        )
        (output_dir / "data.yaml").write_text(
            "\n".join(yaml_lines) + "\n",
            encoding="utf-8",
        )

        with (output_dir / "split_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=["split", "image", "label"]
            )
            writer.writeheader()
            writer.writerows(manifest)

        total_counts = class_count(all_stems, label_by_stem)
        report = {
            "version": "V2",
            "source_export": source_zip.name,
            "images_total": len(all_stems),
            "objects_total": sum(total_counts.values()),
            "classes": classes,
            "class_counts_total": {
                classes[index]: total_counts[index]
                for index in range(len(classes))
            },
            "split_strategy": (
                "train/val dalle nuove immagini; sessione 2026-10-08 "
                "interamente riservata al test indipendente"
            ),
            "splits": {},
        }

        for split, stems in split_map.items():
            counts = class_count(stems, label_by_stem)
            report["splits"][split] = {
                "images": len(stems),
                "objects": sum(counts.values()),
                "class_counts": {
                    classes[index]: counts[index]
                    for index in range(len(classes))
                },
            }

        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n=== DATASET V2 CREATO ===")
        print(f"Output: {output_dir}")
        print(f"Immagini totali: {len(all_stems)}")
        print(f"Oggetti totali: {sum(total_counts.values())}")
        print("\nDistribuzione classi:")
        for index, name in enumerate(classes):
            print(f"  {name}: {total_counts[index]}")

        print("\nSplit:")
        for split in ("train", "val", "test"):
            info = report["splits"][split]
            print(
                f"  {split}: {info['images']} immagini, "
                f"{info['objects']} oggetti"
            )

        print(f"\ndata.yaml: {output_dir / 'data.yaml'}")
        print("Dataset pronto per il training.")


if __name__ == "__main__":
    main()
