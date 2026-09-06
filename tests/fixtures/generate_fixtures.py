"""
Synthetic YOLO Dataset Fixture Generator.
Creates minimal synthetic datasets with PIL images and YOLO labels
for unit testing and validating dataset/tools/merge_datasets.py.
"""

from pathlib import Path
from PIL import Image
import yaml

FIXTURES_DIR = Path(__file__).resolve().parent / "synthetic_datasets"


def create_solid_image(path: Path, color: tuple[int, int, int] = (128, 128, 128), size: tuple[int, int] = (64, 64)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)


def write_label(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")


def generate_all_fixtures():
    # -------------------------------------------------------------
    # 1. dataset_alpha: Pre-split into train / val
    # Classes: 0: pedestrian, 1: car, 2: tricycle
    # -------------------------------------------------------------
    alpha_dir = FIXTURES_DIR / "dataset_alpha"
    alpha_dir.mkdir(parents=True, exist_ok=True)

    alpha_yaml = {
        "nc": 3,
        "names": {
            0: "pedestrian",
            1: "car",
            2: "tricycle",
        },
        "train": "images/train",
        "val": "images/val",
    }
    with open(alpha_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(alpha_yaml, f, sort_keys=False)

    # Train items
    create_solid_image(alpha_dir / "images" / "train" / "alpha_001.jpg", (255, 0, 0))
    write_label(
        alpha_dir / "labels" / "train" / "alpha_001.txt",
        [
            "0 0.500000 0.500000 0.200000 0.400000",  # pedestrian -> person
            "1 0.300000 0.300000 0.400000 0.300000",  # car -> car
            "2 0.700000 0.700000 0.200000 0.200000",  # tricycle -> DROP
        ],
    )

    create_solid_image(alpha_dir / "images" / "train" / "alpha_002.jpg", (0, 255, 0))
    write_label(
        alpha_dir / "labels" / "train" / "alpha_002.txt",
        [
            "0 0.400000 0.400000 0.300000 0.500000",  # pedestrian -> person
        ],
    )

    create_solid_image(alpha_dir / "images" / "train" / "alpha_003_neg.jpg", (0, 0, 255))
    write_label(alpha_dir / "labels" / "train" / "alpha_003_neg.txt", [])  # Empty label (negative image)

    # Val items
    create_solid_image(alpha_dir / "images" / "val" / "alpha_val_001.jpg", (255, 255, 0))
    write_label(
        alpha_dir / "labels" / "val" / "alpha_val_001.txt",
        [
            "1 0.200000 0.200000 0.300000 0.300000",  # car -> car
        ],
    )

    create_solid_image(alpha_dir / "images" / "val" / "alpha_val_002_drop.jpg", (0, 255, 255))
    write_label(
        alpha_dir / "labels" / "val" / "alpha_val_002_drop.txt",
        [
            "2 0.500000 0.500000 0.200000 0.200000",  # tricycle -> DROP (all boxes in file dropped!)
        ],
    )

    # -------------------------------------------------------------
    # 2. dataset_beta: Flat layout (no splits), tests random split
    # Classes: 0: drone, 1: unknown_noise
    # -------------------------------------------------------------
    beta_dir = FIXTURES_DIR / "dataset_beta"
    beta_dir.mkdir(parents=True, exist_ok=True)

    beta_yaml = {
        "nc": 2,
        "names": {
            0: "drone",
            1: "unknown_noise",
        },
    }
    with open(beta_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(beta_yaml, f, sort_keys=False)

    beta_samples = [
        ("beta_001.jpg", (255, 128, 0), ["0 0.50 0.50 0.10 0.10"]),
        ("beta_002.jpg", (128, 0, 255), ["0 0.20 0.20 0.10 0.10", "1 0.80 0.80 0.10 0.10"]),
        ("beta_003.jpg", (100, 100, 100), ["1 0.50 0.50 0.20 0.20"]),  # all dropped
        ("beta_004.jpg", (255, 192, 203), []),  # empty label
        ("beta_005.jpg", (0, 128, 128), ["0 0.60 0.60 0.20 0.20"]),
        ("beta_006.jpg", (139, 69, 19), ["0 0.40 0.40 0.15 0.15"]),
        ("beta_007.jpg", (128, 0, 0), ["0 0.30 0.30 0.20 0.20"]),
        ("beta_008.jpg", (0, 0, 128), ["0 0.70 0.70 0.10 0.10"]),
        ("beta_009.jpg", (50, 205, 50), ["0 0.10 0.10 0.05 0.05"]),
        ("beta_010.jpg", (128, 128, 0), ["0 0.90 0.90 0.08 0.08"]),
    ]

    for filename, color, label_lines in beta_samples:
        create_solid_image(beta_dir / "images" / filename, color)
        stem = Path(filename).stem
        write_label(beta_dir / "labels" / f"{stem}.txt", label_lines)

    print(f"Fixtures generated under: {FIXTURES_DIR}")


if __name__ == "__main__":
    generate_all_fixtures()
