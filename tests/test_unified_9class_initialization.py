"""
Unit Tests for Iteration 3D: Unified 9-Class Detector Initialization & Dataset Validation.
Verifies:
1. Production model immutability (YOLO-L-v002).
2. Source checkpoint immutability (yolov8m.pt and D1 best.pt).
3. Knowledge-preserving unified 9-class initialization checkpoint integrity, taxonomy, and inference smoke test.
4. Unified dataset view taxonomy and structure.
5. Holdout sequence isolation (zero leakage of UA-DETRAC test sequences).
"""

import hashlib
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

EXPECTED_TAXONOMY = {
    0: 'person',
    1: 'car',
    2: 'truck',
    3: 'bus',
    4: 'motorcycle',
    5: 'bicycle',
    6: 'animal',
    7: 'backpack',
    8: 'bag',
}


def test_production_model_immutability():
    """Confirms production YOLO-L-v002 weights exist and have not been altered."""
    prod_pt = ROOT_DIR / "models" / "registry" / "YOLO-L-v002" / "weights" / "best.pt"
    assert prod_pt.is_file(), f"Production model missing: {prod_pt}"
    expected_sha = "eba4580f75fd5ccd5bfb37f6289a903288f91f7eedda3024faa822f0db6fb943"
    actual_sha = hashlib.sha256(prod_pt.read_bytes()).hexdigest()
    assert actual_sha == expected_sha, "Production model weights were altered or overwritten!"


def test_source_checkpoints_immutability():
    """Confirms COCO yolov8m.pt and D1 best.pt exist and have not been altered."""
    coco_pt = ROOT_DIR / "yolov8m.pt"
    assert coco_pt.is_file(), f"yolov8m.pt missing: {coco_pt}"
    expected_coco_sha = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
    actual_coco_sha = hashlib.sha256(coco_pt.read_bytes()).hexdigest()
    assert actual_coco_sha == expected_coco_sha, "yolov8m.pt was altered!"

    d1_pt = ROOT_DIR / "training_lab" / "runs" / "D1_yolov8m_640_2class_full" / "weights" / "best.pt"
    assert d1_pt.is_file(), f"D1 best.pt missing: {d1_pt}"
    expected_d1_sha = "005b706408d49f52ab0a26795f8ea8ae2c0e7dc725445a616ddbb56cd4f7de3b"
    actual_d1_sha = hashlib.sha256(d1_pt.read_bytes()).hexdigest()
    assert actual_d1_sha == expected_d1_sha, "D1 best.pt was altered!"


def test_unified_initialization_checkpoint():
    """Verifies the derived 9-class initialization checkpoint loads, has nc=9, and infers without NaN."""
    from ultralytics import YOLO

    ckpt_path = ROOT_DIR / "training_lab" / "runs" / "unified_9class_initialization" / "weights" / "init_yolov8m_9class.pt"
    assert ckpt_path.is_file(), f"Unified initialization checkpoint missing: {ckpt_path}"

    model = YOLO(str(ckpt_path))
    assert model.model.nc == 9, f"Expected nc=9, got {model.model.nc}"
    assert model.names == EXPECTED_TAXONOMY, f"Taxonomy mismatch: {model.names}"

    # Smoke test inference on dummy frame
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    result = model(dummy, device=device, verbose=False)[0]

    assert result.boxes is not None
    assert not torch.isnan(result.boxes.data).any().item(), "NaN detected in initialization output!"


def test_unified_dataset_view_taxonomy():
    """Verifies that unified dataset view dataset.yaml exists and defines exactly the 9 classes."""
    yaml_path = ROOT_DIR / "training_lab" / "runs" / "unified_9class_dataset_view" / "dataset.yaml"
    assert yaml_path.is_file(), f"dataset.yaml missing: {yaml_path}"

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("nc") == 9, f"Expected nc=9 in yaml, got {cfg.get('nc')}"
    assert cfg.get("names") == EXPECTED_TAXONOMY, "Class names mismatch in dataset.yaml"


def test_ua_detrac_holdout_protection():
    """Verifies zero leakage of protected UA-DETRAC holdout sequences in the unified dataset view."""
    dataset_dir = ROOT_DIR / "training_lab" / "runs" / "unified_9class_dataset_view"
    assert dataset_dir.is_dir(), f"Unified dataset view missing: {dataset_dir}"

    holdout_sequences = [
        "MVI_40244",
        "MVI_40732",
        "MVI_40751",
        "MVI_40752",
        "MVI_40871",
        "MVI_40962",
        "MVI_40963",
    ]

    for split in ["train", "val", "test"]:
        lbl_dir = dataset_dir / "labels" / split
        if not lbl_dir.is_dir():
            continue
        for entry in lbl_dir.iterdir():
            for seq in holdout_sequences:
                assert seq not in entry.name, f"Holdout sequence {seq} leaked into {split} split ({entry.name})!"
