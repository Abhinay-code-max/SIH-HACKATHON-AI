"""
Builder for Iteration 3F Controlled 9-Class Dataset View.
Materializes: training_lab/runs/controlled_9class_dataset_view_3f
- Train: 4,523 images (1,911 UA-DETRAC stride=20, 2,560 unified, 52 v2)
- Val: 1,674 images (identical to 3E)
"""

import os
import shutil
from pathlib import Path
from collections import Counter
import yaml

ROOT = Path(__file__).resolve().parent.parent
src_view = ROOT / "training_lab/runs/controlled_9class_dataset_view"
out_view = ROOT / "training_lab/runs/controlled_9class_dataset_view_3f"

out_view.mkdir(parents=True, exist_ok=True)
(out_view / "images/train").mkdir(parents=True, exist_ok=True)
(out_view / "labels/train").mkdir(parents=True, exist_ok=True)
(out_view / "images/val").mkdir(parents=True, exist_ok=True)
(out_view / "labels/val").mkdir(parents=True, exist_ok=True)

def link_file(src, dst):
    if not dst.exists():
        try:
            os.link(str(src), str(dst))
        except Exception:
            shutil.copy2(str(src), str(dst))

print("1. Materializing 3F Train Split (4,523 images)...")
train_img_src = src_view / "images/train"
train_lbl_src = src_view / "labels/train"
train_img_dst = out_view / "images/train"
train_lbl_dst = out_view / "labels/train"

train_count = 0
for entry in train_img_src.iterdir():
    name = entry.name
    include = False
    if name.startswith("dataset_ua_detrac_v001_"):
        fnum = int(name.split("_f")[-1].split(".")[0])
        if fnum % 20 == 1:
            include = True
    elif name.startswith(("uni_", "v2_")):
        include = True

    if include:
        train_count += 1
        link_file(entry, train_img_dst / name)
        lbl_file = train_lbl_src / f"{entry.stem}.txt"
        dst_lbl_file = train_lbl_dst / f"{entry.stem}.txt"
        if lbl_file.exists():
            shutil.copy2(str(lbl_file), str(dst_lbl_file))
        else:
            dst_lbl_file.write_text("", encoding="utf-8")

print(f"  Materialized {train_count:,} training images.")

print("2. Materializing 3F Validation Split (1,674 images, identical to 3E)...")
val_img_src = src_view / "images/val"
val_lbl_src = src_view / "labels/val"
val_img_dst = out_view / "images/val"
val_lbl_dst = out_view / "labels/val"

val_count = 0
for entry in val_img_src.iterdir():
    val_count += 1
    link_file(entry, val_img_dst / entry.name)
    lbl_file = val_lbl_src / f"{entry.stem}.txt"
    dst_lbl_file = val_lbl_dst / f"{entry.stem}.txt"
    if lbl_file.exists():
        shutil.copy2(str(lbl_file), str(dst_lbl_file))
    else:
        dst_lbl_file.write_text("", encoding="utf-8")

print(f"  Materialized {val_count:,} validation images.")

TAXONOMY = {
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

yaml_cfg = {
    "path": str(out_view.resolve()).replace("\\", "/"),
    "train": "images/train",
    "val": "images/val",
    "nc": 9,
    "names": TAXONOMY,
}

yaml_path = out_view / "dataset.yaml"
with open(yaml_path, "w", encoding="utf-8") as yf:
    yaml.safe_dump(yaml_cfg, yf, sort_keys=False)

print(f"Wrote dataset.yaml to: {yaml_path}")
print("Controlled 3F view construction COMPLETE.")
