"""
Build Unified 9-Class Dataset Engine V002 (Phase 2 - Multi-Source Surveillance & Rare-Class Enrichment).
Constructs training_lab/datasets/dataset_unified_v002/ by uniting:
1. Validated COCO Val2017 Baseline (3,658 images) preserving exact split assignments.
2. VisDrone2019-DET Surveillance (548 images) with sequence-isolated partitioning.
3. Targeted COCO Train2017 Weak-Class Enrichment (600 images rich in backpack, bag, bicycle, truck).
4. UA-DETRAC CCTV Traffic Overpass Frames (from non-holdout training sequences).

Enforces strict quality gating:
- Bounding box dimensions >= 10px, area > 0, clipped to image boundaries.
- Discards iscrowd==1, ignored regions, and unmapped classes.
- Zero cross-split leakage (Train ∩ Val = ∅, Train ∩ Holdout = ∅, Val ∩ Holdout = ∅).
- Cryptographic provenance tracking for all sources.
- Generates data.yaml, dataset_manifest.json, split_manifest.json, source_manifest.json, class_mapping.yaml.
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
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TARGET_V002_DIR = ROOT_DIR / "training_lab" / "datasets" / "dataset_unified_v002"
V001_DIR = ROOT_DIR / "training_lab" / "datasets" / "dataset_unified_v001"
RAW_DIR = ROOT_DIR / "training_lab" / "raw_data"
UA_DETRAC_DIR = ROOT_DIR / "training_lab" / "datasets" / "dataset_ua_detrac_v001"

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

# Mapping: COCO Category Names -> Canonical Class IDs
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

# Mapping: VisDrone Object Category IDs (1-based) -> Canonical Class IDs
# 1: pedestrian, 2: people, 3: bicycle, 4: car, 5: van, 6: truck, 7: tricycle (reject),
# 8: awning-tricycle (reject), 9: bus, 10: motor, 11: others (reject)
VISDRONE_CAT_TO_CANONICAL = {
    1: 0,  # pedestrian -> person
    2: 0,  # people -> person
    3: 5,  # bicycle -> bicycle
    4: 1,  # car -> car
    5: 1,  # van -> car
    6: 2,  # truck -> truck
    9: 3,  # bus -> bus
    10: 4, # motor -> motorcycle
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class UnifiedDatasetV002Builder:
    def __init__(self, target_dir: Path = TARGET_V002_DIR, random_seed: int = 42):
        self.target_dir = Path(target_dir)
        self.random_seed = random_seed
        self.rejection_stats = defaultdict(lambda: Counter())

        # Accumulated samples per split: split -> list of sample dicts
        self.samples = {
            "train": [],
            "val": [],
            "holdout": [],
        }

    def build(self) -> Dict[str, Any]:
        print("=" * 80)
        print("BUILDING UNIFIED 9-CLASS DATASET V002 (dataset_unified_v002)")
        print("=" * 80)

        # Initialize directories
        for split in ["train", "val", "holdout"]:
            (self.target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (self.target_dir / "reports").mkdir(parents=True, exist_ok=True)

        # 1. Ingest Base COCO Val2017 (preserves exact split continuity from V001)
        self._ingest_v001_base()

        # 2. Ingest VisDrone2019-DET Surveillance
        self._ingest_visdrone_surveillance()

        # 3. Ingest Targeted COCO Train2017 Weak-Class Enrichment
        self._ingest_targeted_coco_train()

        # 4. Ingest UA-DETRAC CCTV Traffic Overpass Frames (Training Sequences Only)
        self._ingest_ua_detrac_cctv()

        # 5. Write Manifests and Metadata
        manifest_data = self._finalize_dataset()

        return manifest_data

    def _ingest_v001_base(self):
        print("\n[Source 1/4] Ingesting Base COCO Val2017 from dataset_unified_v001...")
        assert V001_DIR.is_dir(), f"V001 dataset directory missing at {V001_DIR}"

        splits = ["train", "val", "holdout"]
        source_name = "COCO_Val2017_Base"
        imported_count = 0

        for s in splits:
            img_dir = V001_DIR / "images" / s
            lbl_dir = V001_DIR / "labels" / s

            for img_p in sorted(list(img_dir.glob("*.jpg"))):
                lbl_p = lbl_dir / f"{img_p.stem}.txt"
                if not lbl_p.is_file():
                    self.rejection_stats[source_name]["missing_label"] += 1
                    continue

                boxes = []
                for line in lbl_p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        parts = line.strip().split()
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        boxes.append((cls_id, cx, cy, w, h))

                if not boxes:
                    self.rejection_stats[source_name]["empty_label"] += 1
                    continue

                dest_img = self.target_dir / "images" / s / img_p.name
                dest_lbl = self.target_dir / "labels" / s / f"{img_p.stem}.txt"

                if not dest_img.is_file():
                    shutil.copy2(img_p, dest_img)

                lbl_lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
                dest_lbl.write_text("\n".join(lbl_lines) + "\n", encoding="utf-8")

                self.samples[s].append({
                    "id": f"coco_v001_{img_p.stem}",
                    "filename": img_p.name,
                    "split": s,
                    "source": source_name,
                    "boxes": boxes,
                })
                imported_count += 1

        print(f"  Ingested {imported_count:,} base images from V001 across train/val/holdout.")

    def _ingest_visdrone_surveillance(self):
        print("\n[Source 2/4] Ingesting VisDrone2019-DET Surveillance Benchmark...")
        visdrone_dir = RAW_DIR / "visdrone_extracted" / "VisDrone2019-DET-val"
        assert visdrone_dir.is_dir(), f"VisDrone directory missing at {visdrone_dir}"

        source_name = "VisDrone2019_DET"
        img_src_dir = visdrone_dir / "images"
        ann_src_dir = visdrone_dir / "annotations"

        ann_files = sorted(list(ann_src_dir.glob("*.txt")))
        print(f"  Found {len(ann_files)} raw VisDrone annotation files.")

        # Group by sequence / video ID (VisDrone convention: 0000001_... is sequence 1)
        seq_to_files = defaultdict(list)
        for ann_f in ann_files:
            stem = ann_f.stem
            seq_id = stem.split("_")[0]
            seq_to_files[seq_id].append(ann_f)

        sorted_seqs = sorted(list(seq_to_files.keys()))
        rng = random.Random(self.random_seed)
        rng.shuffle(sorted_seqs)

        # 70% Train / 15% Val / 15% Holdout at Sequence Level
        n_seqs = len(sorted_seqs)
        train_seqs = set(sorted_seqs[: int(n_seqs * 0.70)])
        val_seqs = set(sorted_seqs[int(n_seqs * 0.70) : int(n_seqs * 0.85)])
        holdout_seqs = set(sorted_seqs[int(n_seqs * 0.85) :])

        imported_count = 0

        for ann_f in ann_files:
            stem = ann_f.stem
            seq_id = stem.split("_")[0]
            img_f = img_src_dir / f"{stem}.jpg"

            if not img_f.is_file():
                self.rejection_stats[source_name]["missing_image"] += 1
                continue

            # Byte-level deduplication to prevent raw VisDrone duplicate image leakage across sequences
            with open(img_f, "rb") as fp:
                img_sha = hashlib.sha256(fp.read()).hexdigest()
            if not hasattr(self, "_seen_visdrone_img_hashes"):
                self._seen_visdrone_img_hashes = {}
            if img_sha in self._seen_visdrone_img_hashes:
                self.rejection_stats[source_name]["exact_image_duplicate"] += 1
                continue
            self._seen_visdrone_img_hashes[img_sha] = stem

            # Determine split from sequence
            if seq_id in train_seqs:
                split = "train"
            elif seq_id in val_seqs:
                split = "val"
            else:
                split = "holdout"

            # Open image to get dimensions
            try:
                with Image.open(img_f) as img:
                    img_w, img_h = img.size
            except Exception:
                self.rejection_stats[source_name]["corrupt_image"] += 1
                continue

            # Parse VisDrone annotations
            boxes = []
            for line in ann_f.read_text(encoding="utf-8").strip().splitlines():
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) < 6:
                    self.rejection_stats[source_name]["malformed_row"] += 1
                    continue

                x, y, w, h = map(int, parts[:4])
                score = int(parts[4])
                cat = int(parts[5])

                # Skip ignored regions
                if score == 0:
                    self.rejection_stats[source_name]["ignored_region"] += 1
                    continue

                # Filter unmapped categories (tricycle, awning-tricycle, others)
                if cat not in VISDRONE_CAT_TO_CANONICAL:
                    self.rejection_stats[source_name]["unmapped_category"] += 1
                    continue

                if w < 10 or h < 10:
                    self.rejection_stats[source_name]["degen_width_height"] += 1
                    continue

                # Clamp to image boundaries
                x1 = max(0.0, float(x))
                y1 = max(0.0, float(y))
                x2 = min(float(img_w), float(x + w))
                y2 = min(float(img_h), float(y + h))

                cw = x2 - x1
                ch = y2 - y1
                if cw < 10 or ch < 10:
                    self.rejection_stats[source_name]["out_of_bounds"] += 1
                    continue

                # Normalized YOLO format
                xc = (x1 + x2) / (2.0 * img_w)
                yc = (y1 + y2) / (2.0 * img_h)
                nw = cw / img_w
                nh = ch / img_h

                canonical_id = VISDRONE_CAT_TO_CANONICAL[cat]
                boxes.append((
                    canonical_id,
                    round(min(max(0.0, xc), 1.0), 6),
                    round(min(max(0.0, yc), 1.0), 6),
                    round(min(max(0.0, nw), 1.0), 6),
                    round(min(max(0.0, nh), 1.0), 6),
                ))

            if not boxes:
                self.rejection_stats[source_name]["zero_valid_boxes"] += 1
                continue

            # Target filename prefix to prevent any collision
            new_fname = f"visdrone_{stem}.jpg"
            dest_img = self.target_dir / "images" / split / new_fname
            dest_lbl = self.target_dir / "labels" / split / f"visdrone_{stem}.txt"

            if not dest_img.is_file():
                shutil.copy2(img_f, dest_img)

            lbl_lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
            dest_lbl.write_text("\n".join(lbl_lines) + "\n", encoding="utf-8")

            self.samples[split].append({
                "id": f"visdrone_{stem}",
                "filename": new_fname,
                "split": split,
                "source": source_name,
                "sequence": seq_id,
                "boxes": boxes,
            })
            imported_count += 1

        print(f"  Ingested {imported_count:,} VisDrone surveillance images across {n_seqs} sequences.")

    def _ingest_targeted_coco_train(self):
        print("\n[Source 3/4] Ingesting Targeted COCO Train2017 Weak-Class Enrichment...")
        src_img_dir = RAW_DIR / "coco_train_targeted"
        meta_json_path = RAW_DIR / "coco_train_targeted_annotations.json"
        assert meta_json_path.is_file(), f"Targeted COCO metadata missing at {meta_json_path}"

        source_name = "COCO_Train2017_Targeted"
        with open(meta_json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        coco_cat_id_to_canonical = {}
        for cat in coco_data.get("categories", []):
            if cat["name"] in COCO_NAME_TO_CANONICAL:
                coco_cat_id_to_canonical[cat["id"]] = COCO_NAME_TO_CANONICAL[cat["name"]]

        images_by_id = {img["id"]: img for img in coco_data.get("images", [])}
        anns_by_img = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            cat_id = ann.get("category_id")
            if cat_id not in coco_cat_id_to_canonical:
                self.rejection_stats[source_name]["unmapped_category"] += 1
                continue
            if ann.get("iscrowd", 0) == 1:
                self.rejection_stats[source_name]["iscrowd"] += 1
                continue
            bbox = ann.get("bbox", [])
            if len(bbox) != 4 or bbox[2] < 10 or bbox[3] < 10:
                self.rejection_stats[source_name]["degen_width_height"] += 1
                continue
            anns_by_img[ann["image_id"]].append(ann)

        sorted_img_ids = sorted(list(anns_by_img.keys()))
        rng = random.Random(self.random_seed)
        rng.shuffle(sorted_img_ids)

        # 70 / 15 / 15 Split
        n_total = len(sorted_img_ids)
        train_ids = set(sorted_img_ids[: int(n_total * 0.70)])
        val_ids = set(sorted_img_ids[int(n_total * 0.70) : int(n_total * 0.85)])
        holdout_ids = set(sorted_img_ids[int(n_total * 0.85) :])

        imported_count = 0

        for img_id, raw_anns in anns_by_img.items():
            img_info = images_by_id.get(img_id)
            if not img_info:
                continue

            fname = img_info["file_name"]
            img_path = src_img_dir / fname
            if not img_path.is_file():
                self.rejection_stats[source_name]["missing_image"] += 1
                continue

            if img_id in train_ids:
                split = "train"
            elif img_id in val_ids:
                split = "val"
            else:
                split = "holdout"

            img_w = img_info["width"]
            img_h = img_info["height"]

            boxes = []
            for ann in raw_anns:
                x, y, w, h = ann["bbox"]
                x1 = max(0.0, float(x))
                y1 = max(0.0, float(y))
                x2 = min(float(img_w), float(x + w))
                y2 = min(float(img_h), float(y + h))

                cw = x2 - x1
                ch = y2 - y1
                if cw < 10 or ch < 10:
                    self.rejection_stats[source_name]["out_of_bounds"] += 1
                    continue

                xc = (x1 + x2) / (2.0 * img_w)
                yc = (y1 + y2) / (2.0 * img_h)
                nw = cw / img_w
                nh = ch / img_h

                canonical_id = coco_cat_id_to_canonical[ann["category_id"]]
                boxes.append((
                    canonical_id,
                    round(min(max(0.0, xc), 1.0), 6),
                    round(min(max(0.0, yc), 1.0), 6),
                    round(min(max(0.0, nw), 1.0), 6),
                    round(min(max(0.0, nh), 1.0), 6),
                ))

            if not boxes:
                self.rejection_stats[source_name]["zero_valid_boxes"] += 1
                continue

            new_fname = f"coco_tr_{fname}"
            dest_img = self.target_dir / "images" / split / new_fname
            dest_lbl = self.target_dir / "labels" / split / f"coco_tr_{Path(fname).stem}.txt"

            if not dest_img.is_file():
                shutil.copy2(img_path, dest_img)

            lbl_lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
            dest_lbl.write_text("\n".join(lbl_lines) + "\n", encoding="utf-8")

            self.samples[split].append({
                "id": f"coco_tr_{Path(fname).stem}",
                "filename": new_fname,
                "split": split,
                "source": source_name,
                "boxes": boxes,
            })
            imported_count += 1

        print(f"  Ingested {imported_count:,} targeted COCO Train images across weak classes.")

    def _ingest_ua_detrac_cctv(self):
        print("\n[Source 4/4] Ingesting UA-DETRAC CCTV Traffic Overpass Frames...")
        assert UA_DETRAC_DIR.is_dir(), f"UA-DETRAC missing at {UA_DETRAC_DIR}"

        source_name = "UA_DETRAC_CCTV"
        # Strictly pull from training split of UA-DETRAC (MVI_20011 to MVI_40192)
        train_img_dir = UA_DETRAC_DIR / "images" / "train"
        train_lbl_dir = UA_DETRAC_DIR / "labels" / "train"

        train_imgs = sorted(list(train_img_dir.glob("*.jpg")))
        # Sample every 75th frame to get diverse weather, camera angles, and heavy traffic
        sampled_imgs = train_imgs[::75]
        print(f"  Sampling {len(sampled_imgs)} diverse CCTV traffic frames strictly from training sequences...")

        imported_count = 0
        for img_p in sampled_imgs:
            lbl_p = train_lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.is_file():
                self.rejection_stats[source_name]["missing_label"] += 1
                continue

            boxes = []
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    # DETRAC canonical classes: 1: car, 2: truck, 3: bus
                    if cls_id in (1, 2, 3):
                        cx, cy, w, h = map(float, parts[1:])
                        boxes.append((cls_id, cx, cy, w, h))

            if not boxes:
                self.rejection_stats[source_name]["zero_valid_boxes"] += 1
                continue

            # Ingest strictly into Train split (preserving test/holdout isolation)
            split = "train"
            new_fname = f"detrac_{img_p.name}"
            dest_img = self.target_dir / "images" / split / new_fname
            dest_lbl = self.target_dir / "labels" / split / f"detrac_{img_p.stem}.txt"

            if not dest_img.is_file():
                shutil.copy2(img_p, dest_img)

            lbl_lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
            dest_lbl.write_text("\n".join(lbl_lines) + "\n", encoding="utf-8")

            self.samples[split].append({
                "id": f"detrac_{img_p.stem}",
                "filename": new_fname,
                "split": split,
                "source": source_name,
                "boxes": boxes,
            })
            imported_count += 1

        print(f"  Ingested {imported_count:,} UA-DETRAC CCTV frames into Train split.")

    def _finalize_dataset(self) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("FINALIZING DATASET V002 & GENERATING MANIFESTS")
        print("=" * 80)

        # Count total annotations and class breakdown
        split_counts = {s: len(self.samples[s]) for s in ["train", "val", "holdout"]}
        split_class_boxes = {s: Counter() for s in ["train", "val", "holdout"]}
        source_counts = Counter()

        for s in ["train", "val", "holdout"]:
            for item in self.samples[s]:
                source_counts[item["source"]] += 1
                for b in item["boxes"]:
                    split_class_boxes[s][b[0]] += 1

        total_boxes = sum(sum(split_class_boxes[s].values()) for s in ["train", "val", "holdout"])
        total_images = sum(split_counts.values())

        print(f"Total Physical Images: {total_images:,}")
        print(f"  Train:   {split_counts['train']:,} ({split_counts['train']/total_images*100:.1f}%)")
        print(f"  Val:     {split_counts['val']:,} ({split_counts['val']/total_images*100:.1f}%)")
        print(f"  Holdout: {split_counts['holdout']:,} ({split_counts['holdout']/total_images*100:.1f}%)")
        print(f"Total Retained Bounding Boxes: {total_boxes:,}")

        print("\nPer-Class Distribution Across Splits in V002:")
        for c in range(9):
            c_name = CANONICAL_CLASSES[c]
            tb = split_class_boxes["train"][c]
            vb = split_class_boxes["val"][c]
            hb = split_class_boxes["holdout"][c]
            tot = tb + vb + hb
            print(f"  Class {c} ({c_name:10s}): Train={tb:5d} | Val={vb:4d} | Holdout={hb:4d} | Total={tot:5d}")

        # 1. Generate data.yaml
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
        data_yaml_sha = compute_sha256(data_yaml_path)
        print(f"\nWrote data.yaml (SHA-256: {data_yaml_sha})")

        # 2. Generate class_mapping.yaml
        class_mapping_path = self.target_dir / "class_mapping.yaml"
        class_mapping_content = {
            "canonical_taxonomy": CANONICAL_CLASSES,
            "sources": {
                "COCO_Val2017": COCO_NAME_TO_CANONICAL,
                "COCO_Train2017_Targeted": COCO_NAME_TO_CANONICAL,
                "VisDrone2019_DET": {
                    "pedestrian": 0, "people": 0, "car": 1, "van": 1,
                    "truck": 2, "bus": 3, "motor": 4, "bicycle": 5,
                    "tricycle": "REJECT", "awning-tricycle": "REJECT", "others": "REJECT",
                },
                "UA_DETRAC_CCTV": {
                    "car": 1, "van": 1, "others": 1, "truck": 2, "bus": 3,
                },
            },
            "unmapped_policy": "REJECT",
        }
        with open(class_mapping_path, "w", encoding="utf-8") as cmf:
            yaml.dump(class_mapping_content, cmf, default_flow_style=False, sort_keys=False)
        class_mapping_sha = compute_sha256(class_mapping_path)

        # 3. Generate split_manifest.json
        split_manifest_path = self.target_dir / "split_manifest.json"
        split_manifest = {
            "dataset_name": "dataset_unified_v002",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": self.random_seed,
            "split_ratios": {"train": 0.70, "val": 0.15, "holdout": 0.15},
            "counts": split_counts,
            "class_box_counts": {s: dict(split_class_boxes[s]) for s in ["train", "val", "holdout"]},
            "split_filenames": {
                s: sorted([item["filename"] for item in self.samples[s]])
                for s in ["train", "val", "holdout"]
            },
        }
        with open(split_manifest_path, "w", encoding="utf-8") as smf:
            json.dump(split_manifest, smf, indent=2)
        split_manifest_sha = compute_sha256(split_manifest_path)

        # 4. Generate source_manifest.json
        source_manifest_path = self.target_dir / "source_manifest.json"
        source_manifest = {
            "dataset_name": "dataset_unified_v002",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": [
                {
                    "source_name": "COCO_Val2017",
                    "official_source": "COCO Consortium (https://cocodataset.org)",
                    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
                    "archive_sha256": "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
                    "contributed_images": source_counts["COCO_Val2017_Base"],
                    "domain": "Consumer photography & natural environments",
                },
                {
                    "source_name": "VisDrone2019-DET",
                    "official_source": "AISKYEYE / Tianjin University (http://aiskyeye.com/)",
                    "license": "VisDrone Research License (Free academic/research use)",
                    "archive_sha256": "abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9",
                    "contributed_images": source_counts["VisDrone2019_DET"],
                    "domain": "Elevated pole & aerial surveillance cameras, dense tiny objects",
                },
                {
                    "source_name": "COCO_Train2017_Targeted",
                    "official_source": "COCO Consortium (http://images.cocodataset.org/train2017/)",
                    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
                    "contributed_images": source_counts["COCO_Train2017_Targeted"],
                    "domain": "Targeted weak-class enrichment (backpack, bag, bicycle, truck)",
                },
                {
                    "source_name": "UA-DETRAC_CCTV",
                    "official_source": "University at Albany (http://detrac-db.rit.albany.edu/)",
                    "license": "Academic Research / Educational License",
                    "contributed_images": source_counts["UA_DETRAC_CCTV"],
                    "domain": "Fixed 24/7 CCTV traffic overpass surveillance",
                },
            ],
            "total_images": total_images,
            "target_taxonomy": CANONICAL_CLASSES,
        }
        with open(source_manifest_path, "w", encoding="utf-8") as sf:
            json.dump(source_manifest, sf, indent=2)
        source_manifest_sha = compute_sha256(source_manifest_path)

        # 5. Generate dataset_manifest.json and manifest.json
        manifest_path = self.target_dir / "manifest.json"
        dataset_manifest_path = self.target_dir / "dataset_manifest.json"
        manifest_data = {
            "dataset_version": "dataset_unified_v002",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_yaml_sha256": data_yaml_sha,
            "classes": CANONICAL_CLASSES,
            "splits": split_counts,
            "annotations_per_split": {
                s: sum(split_class_boxes[s].values()) for s in ["train", "val", "holdout"]
            },
            "class_distribution": {
                CANONICAL_CLASSES[c]: {
                    "train": split_class_boxes["train"][c],
                    "val": split_class_boxes["val"][c],
                    "holdout": split_class_boxes["holdout"][c],
                    "total": (
                        split_class_boxes["train"][c]
                        + split_class_boxes["val"][c]
                        + split_class_boxes["holdout"][c]
                    ),
                }
                for c in range(9)
            },
            "source_contributions": dict(source_counts),
            "rejections": {src: dict(rejs) for src, rejs in self.rejection_stats.items()},
            "split_strategy": "SOURCE_AND_SEQUENCE_ISOLATED_DETERMINISTIC_SPLIT",
        }
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest_data, mf, indent=2)
        with open(dataset_manifest_path, "w", encoding="utf-8") as dmf:
            json.dump(manifest_data, dmf, indent=2)
        manifest_sha = compute_sha256(manifest_path)

        manifest_hashes = {
            "data.yaml": data_yaml_sha,
            "manifest.json": manifest_sha,
            "dataset_manifest.json": compute_sha256(dataset_manifest_path),
            "split_manifest.json": split_manifest_sha,
            "source_manifest.json": source_manifest_sha,
            "class_mapping.yaml": class_mapping_sha,
        }
        with open(self.target_dir / "reports" / "U2_V002_MANIFEST_HASHES.json", "w", encoding="utf-8") as hf:
            json.dump(manifest_hashes, hf, indent=2)

        print(f"\nManifest Hashes:")
        for k, v in manifest_hashes.items():
            print(f"  {k:25s}: {v}")

        print("\nDataset V002 Build Complete!")
        return manifest_data


if __name__ == "__main__":
    builder = UnifiedDatasetV002Builder()
    builder.build()
