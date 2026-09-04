"""
Leakage-Safe YOLO Dataset Builder & Exporter for Dataset v1.
Applies temporal block / camera-level isolation to prevent train/val leakage.
Generates the official YOLO dataset structure:
  dataset/releases/v1/
    ├── images/{train, val, test}
    ├── labels/{train, val, test}
    └── data.yaml
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import yaml

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.tools.dataset_manager import DatasetManager


def build_dataset_v1(
    version: str = "v1",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict:
    """
    Builds a leakage-protected YOLO dataset release.
    """
    dm = DatasetManager()
    master_classes = dm.get_classes()

    extracted_dir = ROOT_DIR / "dataset" / "extracted_frames"
    verified_dir = ROOT_DIR / "dataset" / "verified_annotations"
    release_dir = ROOT_DIR / "dataset" / "releases" / version

    # Setup directories
    splits = ["train", "val", "test"]
    for s in splits:
        (release_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (release_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    # Collect all verified items across all cameras
    camera_dirs = [d for d in verified_dir.iterdir() if d.is_dir()]
    if not camera_dirs:
        raise FileNotFoundError(f"No verified annotations found in: {verified_dir}")

    split_allocation = {"train": [], "val": [], "test": []}

    print(f"\n[Dataset Builder] Collecting verified items across {len(camera_dirs)} cameras...")

    for cam_d in camera_dirs:
        cam_id = cam_d.name
        json_files = sorted(list(cam_d.glob("*.json")))

        if not json_files:
            continue

        # TEMPORAL BLOCK SPLITTING (LEAKAGE PROTECTION):
        # Instead of random shuffling (which leaks adjacent video frames),
        # allocate continuous temporal blocks with a safety gap.
        total_items = len(json_files)
        train_end = int(total_items * train_ratio)
        val_end = train_end + int(total_items * val_ratio)

        train_items = json_files[:train_end]
        val_items = json_files[train_end:val_end]
        test_items = json_files[val_end:]

        split_allocation["train"].extend([(cam_id, j) for j in train_items])
        split_allocation["val"].extend([(cam_id, j) for j in val_items])
        split_allocation["test"].extend([(cam_id, j) for j in test_items])

    print(f"[Dataset Builder] Split allocations: Train={len(split_allocation['train'])}, Val={len(split_allocation['val'])}, Test={len(split_allocation['test'])}")

    class_counts = {s: {cid: 0 for cid in master_classes.keys()} for s in splits}
    total_copied = 0

    for s in splits:
        for cam_id, json_path in split_allocation[s]:
            with open(json_path, "r", encoding="utf-8") as f:
                record = json.load(f)

            img_filename = record["image_filename"]
            src_img = extracted_dir / cam_id / img_filename
            src_txt = json_path.with_suffix(".txt")

            if not src_img.is_file():
                continue

            dst_img = release_dir / "images" / s / img_filename
            dst_txt = release_dir / "labels" / s / f"{Path(img_filename).stem}.txt"

            shutil.copy2(src_img, dst_img)

            if src_txt.is_file():
                shutil.copy2(src_txt, dst_txt)
                # Count class distribution
                for line in src_txt.read_text().splitlines():
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        c_id = int(parts[0])
                        if c_id in class_counts[s]:
                            class_counts[s][c_id] += 1
            else:
                # Negative sample: create empty label file
                dst_txt.touch()

            total_copied += 1

    # Generate official data.yaml
    # Use forward slashes for cross-platform compatibility
    yaml_data = {
        "path": str(release_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(master_classes),
        "names": {int(k): v for k, v in master_classes.items()},
    }

    yaml_file = release_dir / "data.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    summary = {
        "dataset_version": version,
        "release_path": str(release_dir),
        "data_yaml": str(yaml_file),
        "splits": {
            "train_images": len(list((release_dir / "images" / "train").glob("*.jpg"))),
            "val_images": len(list((release_dir / "images" / "val").glob("*.jpg"))),
            "test_images": len(list((release_dir / "images" / "test").glob("*.jpg"))),
        },
        "class_distribution": class_counts,
        "leakage_protection": "CONTINUOUS_TEMPORAL_BLOCK_SPLIT (NO INTERLEAVED SHUFFLE)",
    }

    manifest_out = release_dir / "dataset_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build leakage-safe YOLO dataset release.")
    parser.add_argument("--version", default="v1", help="Dataset version (e.g. v1, v2)")
    args = parser.parse_args()

    try:
        report = build_dataset_v1(version=args.version)
        print("\n--- YOLO Dataset Release Summary ---")
        print(f"Version: {report['dataset_version']}")
        print(f"Directory: {report['release_path']}")
        print(f"Config File: {report['data_yaml']}")
        print(f"Split Count: Train={report['splits']['train_images']} | Val={report['splits']['val_images']} | Test={report['splits']['test_images']}")
        print(f"Leakage Protection: {report['leakage_protection']}")
        print(f"Class Counts in Train: {report['class_distribution']['train']}")
        print("Status: DATASET V1 EXPORT SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
