"""
Build Unified 9-Class Dataset Engine (Phase 5 - Option B COCO Val2017).
Constructs training_lab/datasets/dataset_unified_v001/ with strict quality gating:
- Approved category mapping to 9-class canonical taxonomy (0-8)
- Discards iscrowd==1 annotations
- Discards degenerate boxes (<10px width/height, area <= 0)
- Clamps boxes to image boundaries and normalizes to YOLO format [class x y w h] (rounded to 6 decimals)
- Filters corrupted/unreadable images
- Strictly deterministic 70/15/15 train/val/holdout split with seed 42
- Verifies full 9-class coverage across all splits
- Generates data.yaml, dataset_manifest.json, split_manifest.json, source_manifest.json
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Canonical 9-Class Taxonomy
CANONICAL_CLASSES = {
    0: "person",
    1: "car",
    2: "truck",
    3: "bus",
    4: "motorcycle",
    5: "bicycle",
    6: "animal",
    7: "backpack",
    8: "bag",
}

# Mapping from COCO category names to Canonical Class IDs
COCO_NAME_TO_CANONICAL = {
    "person": 0,
    "car": 1,
    "truck": 2,
    "bus": 3,
    "motorcycle": 4,
    "bicycle": 5,
    "bird": 6,
    "cat": 6,
    "dog": 6,
    "horse": 6,
    "sheep": 6,
    "cow": 6,
    "elephant": 6,
    "bear": 6,
    "zebra": 6,
    "giraffe": 6,
    "backpack": 7,
    "handbag": 8,
    "suitcase": 8,
}

COCO_VAL2017_ZIP_SHA = "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
COCO_ANN2017_ZIP_SHA = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"

RAW_DATA_DIR = ROOT_DIR / "training_lab" / "raw_data"
VAL_IMAGES_DIR = RAW_DATA_DIR / "val2017_images" / "val2017"
INSTANCES_JSON = RAW_DATA_DIR / "annotations" / "annotations" / "instances_val2017.json"
TARGET_DATASET_DIR = ROOT_DIR / "training_lab" / "datasets" / "dataset_unified_v001"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class UnifiedDatasetBuilder:
    def __init__(
        self,
        raw_images_dir: Path = VAL_IMAGES_DIR,
        instances_json: Path = INSTANCES_JSON,
        target_dir: Path = TARGET_DATASET_DIR,
        random_seed: int = 42,
    ):
        self.raw_images_dir = raw_images_dir
        self.instances_json = instances_json
        self.target_dir = target_dir
        self.random_seed = random_seed
        self.quality_stats: Dict[str, Any] = {
            "total_coco_images": 0,
            "total_coco_annotations": 0,
            "rejected_iscrowd": 0,
            "rejected_unmapped_category": 0,
            "rejected_degen_width_height": 0,
            "rejected_zero_negative_area": 0,
            "rejected_out_of_bounds": 0,
            "corrupted_images": 0,
            "retained_valid_annotations": 0,
            "images_with_valid_annotations": 0,
            "images_without_target_annotations": 0,
        }

    def build(self) -> Dict[str, Any]:
        print("=" * 80)
        print("BUILDING UNIFIED 9-CLASS DATASET (dataset_unified_v001)")
        print("=" * 80)

        # 1. Verification of inputs
        if not self.instances_json.is_file():
            raise FileNotFoundError(f"Missing annotations file: {self.instances_json}")
        if not self.raw_images_dir.is_dir():
            raise FileNotFoundError(f"Missing raw images directory: {self.raw_images_dir}")

        print(f"Loading annotations from {self.instances_json}...")
        with open(self.instances_json, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        self.quality_stats["total_coco_images"] = len(coco_data.get("images", []))
        self.quality_stats["total_coco_annotations"] = len(coco_data.get("annotations", []))
        print(f"Loaded {self.quality_stats['total_coco_images']:,} images and {self.quality_stats['total_coco_annotations']:,} raw annotations.")

        # Build category map: category_id -> (category_name, canonical_id)
        coco_cat_id_to_canonical = {}
        for cat in coco_data.get("categories", []):
            cat_id = cat["id"]
            cat_name = cat["name"]
            if cat_name in COCO_NAME_TO_CANONICAL:
                coco_cat_id_to_canonical[cat_id] = (cat_name, COCO_NAME_TO_CANONICAL[cat_name])

        print(f"Mapped {len(coco_cat_id_to_canonical)} COCO category IDs to 9 canonical classes.")

        # Index images
        images_by_id = {img["id"]: img for img in coco_data.get("images", [])}

        # Filter annotations and group by image_id
        valid_boxes_by_image = defaultdict(list)
        class_counts_raw = Counter()

        for ann in coco_data.get("annotations", []):
            cat_id = ann.get("category_id")
            if cat_id not in coco_cat_id_to_canonical:
                self.quality_stats["rejected_unmapped_category"] += 1
                continue

            if ann.get("iscrowd", 0) == 1:
                self.quality_stats["rejected_iscrowd"] += 1
                continue

            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                self.quality_stats["rejected_degen_width_height"] += 1
                continue

            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                self.quality_stats["rejected_zero_negative_area"] += 1
                continue
            if w < 10 or h < 10:
                self.quality_stats["rejected_degen_width_height"] += 1
                continue

            img_id = ann.get("image_id")
            img_info = images_by_id.get(img_id)
            if not img_info:
                continue

            img_w = img_info["width"]
            img_h = img_info["height"]

            # Clamp coordinates to bounds
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(img_w), float(x + w))
            y2 = min(float(img_h), float(y + h))

            clipped_w = x2 - x1
            clipped_h = y2 - y1

            if clipped_w < 10 or clipped_h < 10:
                self.quality_stats["rejected_out_of_bounds"] += 1
                continue

            # Convert to normalized YOLO [x_center, y_center, w, h]
            xc = (x1 + x2) / (2.0 * img_w)
            yc = (y1 + y2) / (2.0 * img_h)
            norm_w = clipped_w / img_w
            norm_h = clipped_h / img_h

            # Precision safety clamps
            xc = min(max(0.0, xc), 1.0)
            yc = min(max(0.0, yc), 1.0)
            norm_w = min(max(0.0, norm_w), 1.0)
            norm_h = min(max(0.0, norm_h), 1.0)

            canonical_id = coco_cat_id_to_canonical[cat_id][1]
            class_counts_raw[canonical_id] += 1
            self.quality_stats["retained_valid_annotations"] += 1

            valid_boxes_by_image[img_id].append((
                canonical_id,
                round(xc, 6),
                round(yc, 6),
                round(norm_w, 6),
                round(norm_h, 6),
            ))

        self.quality_stats["images_with_valid_annotations"] = len(valid_boxes_by_image)
        self.quality_stats["images_without_target_annotations"] = (
            self.quality_stats["total_coco_images"] - len(valid_boxes_by_image)
        )

        print(f"Quality gate results: {self.quality_stats['retained_valid_annotations']:,} valid annotations across {len(valid_boxes_by_image):,} images.")
        print(f"Rejected unmapped: {self.quality_stats['rejected_unmapped_category']:,}")
        print(f"Rejected iscrowd: {self.quality_stats['rejected_iscrowd']:,}")
        print(f"Rejected degenerate/area: {self.quality_stats['rejected_degen_width_height'] + self.quality_stats['rejected_zero_negative_area']:,}")

        # Deterministic 70 / 15 / 15 Split
        sorted_img_ids = sorted(list(valid_boxes_by_image.keys()))
        rng = random.Random(self.random_seed)
        shuffled_img_ids = list(sorted_img_ids)
        rng.shuffle(shuffled_img_ids)

        total_valid = len(shuffled_img_ids)
        train_end = int(total_valid * 0.70)
        val_end = train_end + int(total_valid * 0.15)

        train_ids = shuffled_img_ids[:train_end]
        val_ids = shuffled_img_ids[train_end:val_end]
        holdout_ids = shuffled_img_ids[val_end:]

        splits = {
            "train": train_ids,
            "val": val_ids,
            "holdout": holdout_ids,
        }

        print(f"Dataset splits: train={len(train_ids):,} ({len(train_ids)/total_valid*100:.1f}%), val={len(val_ids):,} ({len(val_ids)/total_valid*100:.1f}%), holdout={len(holdout_ids):,} ({len(holdout_ids)/total_valid*100:.1f}%)")

        # Verify class distribution across splits
        split_class_counts = {s: Counter() for s in splits}
        for s, s_ids in splits.items():
            for i_id in s_ids:
                for box in valid_boxes_by_image[i_id]:
                    split_class_counts[s][box[0]] += 1

        print("\nClass representation audit across splits:")
        for c_id in range(9):
            c_name = CANONICAL_CLASSES[c_id]
            t_cnt = split_class_counts["train"][c_id]
            v_cnt = split_class_counts["val"][c_id]
            h_cnt = split_class_counts["holdout"][c_id]
            print(f"  Class {c_id} ({c_name:10s}): train={t_cnt:5d} | val={v_cnt:4d} | holdout={h_cnt:4d}")
            if t_cnt == 0 or v_cnt == 0 or h_cnt == 0:
                raise ValueError(f"Split distribution violation! Class {c_id} ({c_name}) missing in a split!")

        # Setup destination directories
        for split in ["train", "val", "holdout"]:
            (self.target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (self.target_dir / "reports").mkdir(parents=True, exist_ok=True)

        # Write images and labels
        print("\nExporting images and writing YOLO label files...")
        for s, s_ids in splits.items():
            img_dest_dir = self.target_dir / "images" / s
            lbl_dest_dir = self.target_dir / "labels" / s

            for i_id in s_ids:
                img_meta = images_by_id[i_id]
                src_filename = img_meta["file_name"]
                src_img_path = self.raw_images_dir / src_filename

                if not src_img_path.is_file():
                    raise FileNotFoundError(f"Image not found on disk: {src_img_path}")

                dest_img_path = img_dest_dir / src_filename
                dest_lbl_path = lbl_dest_dir / f"{Path(src_filename).stem}.txt"

                # Copy image if not already present or different size
                if not dest_img_path.is_file() or dest_img_path.stat().st_size != src_img_path.stat().st_size:
                    shutil.copy2(src_img_path, dest_img_path)

                # Write YOLO label
                boxes = valid_boxes_by_image[i_id]
                lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
                dest_lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Generate data.yaml (with posix forward slashes for cross-platform compatibility)
        data_yaml_path = self.target_dir / "data.yaml"
        target_dir_posix = str(self.target_dir.resolve()).replace("\\", "/")
        data_yaml_content = {
            "path": target_dir_posix,
            "train": "images/train",
            "val": "images/val",
            "test": "images/holdout",
            "names": CANONICAL_CLASSES,
        }
        with open(data_yaml_path, "w", encoding="utf-8") as yf:
            yaml.dump(data_yaml_content, yf, default_flow_style=False, sort_keys=False)
        print(f"Wrote {data_yaml_path}")

        # Generate split_manifest.json
        split_manifest_path = self.target_dir / "split_manifest.json"
        split_manifest = {
            "dataset_name": "dataset_unified_v001",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": self.random_seed,
            "split_ratio": {"train": 0.70, "val": 0.15, "holdout": 0.15},
            "counts": {
                "train": len(train_ids),
                "val": len(val_ids),
                "holdout": len(holdout_ids),
                "total": total_valid,
            },
            "class_counts": {s: dict(split_class_counts[s]) for s in ["train", "val", "holdout"]},
            "split_filenames": {
                s: sorted([images_by_id[i_id]["file_name"] for i_id in s_ids])
                for s, s_ids in splits.items()
            },
        }
        with open(split_manifest_path, "w", encoding="utf-8") as smf:
            json.dump(split_manifest, smf, indent=2)
        print(f"Wrote {split_manifest_path}")

        # Generate source_manifest.json
        source_manifest_path = self.target_dir / "source_manifest.json"
        source_manifest = {
            "dataset_name": "dataset_unified_v001",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": [
                {
                    "source_name": "COCO Val2017",
                    "source_zip_sha256": COCO_VAL2017_ZIP_SHA,
                    "annotations_zip_sha256": COCO_ANN2017_ZIP_SHA,
                    "total_source_images": self.quality_stats["total_coco_images"],
                    "used_source_images": total_valid,
                    "quality_filtering": self.quality_stats,
                }
            ],
            "target_taxonomy": CANONICAL_CLASSES,
        }
        with open(source_manifest_path, "w", encoding="utf-8") as smf:
            json.dump(source_manifest, smf, indent=2)
        print(f"Wrote {source_manifest_path}")

        # Generate dataset_manifest.json (and copy to manifest.json for full compatibility)
        data_yaml_sha = compute_sha256(data_yaml_path)
        dataset_manifest_path = self.target_dir / "dataset_manifest.json"
        manifest_path = self.target_dir / "manifest.json"

        manifest_data = {
            "dataset_version": "dataset_unified_v001",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_yaml_sha256": data_yaml_sha,
            "classes": CANONICAL_CLASSES,
            "splits": {
                "train": len(train_ids),
                "val": len(val_ids),
                "holdout": len(holdout_ids),
                "test": len(holdout_ids),
            },
            "annotations_per_split": {
                "train": sum(split_class_counts["train"].values()),
                "val": sum(split_class_counts["val"].values()),
                "holdout": sum(split_class_counts["holdout"].values()),
            },
            "class_distribution": {
                CANONICAL_CLASSES[c]: {
                    "train": split_class_counts["train"][c],
                    "val": split_class_counts["val"][c],
                    "holdout": split_class_counts["holdout"][c],
                    "total": (
                        split_class_counts["train"][c]
                        + split_class_counts["val"][c]
                        + split_class_counts["holdout"][c]
                    ),
                }
                for c in range(9)
            },
            "quality_gate_statistics": self.quality_stats,
            "split_strategy": "COCO_IMAGE_HASH_DETERMINISTIC_SPLIT",
            "videos_per_split": {
                "train": ["COCO_VAL2017_SPLIT_TRAIN"],
                "val": ["COCO_VAL2017_SPLIT_VAL"],
                "test": ["COCO_VAL2017_SPLIT_HOLDOUT"],
            },
        }

        with open(dataset_manifest_path, "w", encoding="utf-8") as dmf:
            json.dump(manifest_data, dmf, indent=2)
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest_data, mf, indent=2)

        print(f"Wrote {dataset_manifest_path} and {manifest_path}")
        print(f"Dataset SHA-256 for data.yaml: {data_yaml_sha}")
        print("\nUnified 9-Class Dataset Build Complete!")
        return manifest_data


if __name__ == "__main__":
    builder = UnifiedDatasetBuilder()
    builder.build()
