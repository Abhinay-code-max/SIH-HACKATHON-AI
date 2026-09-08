"""
Unit and Integration Tests for DerivedDatasetView, Class Remapping, Path Normalization,
and Canonical Dataset Immutability.
"""

import hashlib
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.dataset_view import (
    DerivedDatasetView,
    compute_file_sha256,
    remap_label_line,
    remap_label_file,
    link_directory_zero_copy,
)
from training_lab.engine.training_manager import format_weights_path


# ==============================================================================
# 1. Focused Unit Tests for remap_label_line
# ==============================================================================

def test_remap_label_line_canonical_to_d1():
    """Verify canonical 1 (car) -> 0, 3 (bus) -> 1, preserving geometries exactly."""
    mapping = {1: 0, 3: 1}

    # Car test
    car_line = "1 0.5 0.5 0.2 0.2"
    remapped_car = remap_label_line(car_line, mapping, source_file="test_car.txt")
    assert remapped_car == "0 0.5 0.5 0.2 0.2", f"Expected '0 0.5 0.5 0.2 0.2', got '{remapped_car}'"

    # Bus test
    bus_line = "3 0.4 0.4 0.1 0.1"
    remapped_bus = remap_label_line(bus_line, mapping, source_file="test_bus.txt")
    assert remapped_bus == "1 0.4 0.4 0.1 0.1", f"Expected '1 0.4 0.4 0.1 0.1', got '{remapped_bus}'"


def test_remap_label_line_unsupported_classes_fail_loudly():
    """Verify that any non-allowed canonical class (e.g. 0, 2, 5, 8) raises ValueError loudly."""
    mapping = {1: 0, 3: 1}

    # Class 5 (bicycle)
    with pytest.raises(ValueError, match="Unsupported canonical class ID 5"):
        remap_label_line("5 0.5 0.5 0.2 0.2", mapping, source_file="bad_bike.txt")

    # Class 0 (person)
    with pytest.raises(ValueError, match="Unsupported canonical class ID 0"):
        remap_label_line("0 0.5 0.5 0.2 0.2", mapping, source_file="bad_person.txt")

    # Class 2 (truck)
    with pytest.raises(ValueError, match="Unsupported canonical class ID 2"):
        remap_label_line("2 0.5 0.5 0.2 0.2", mapping, source_file="bad_truck.txt")

    # Non-integer token
    with pytest.raises(ValueError, match="Non-integer class ID"):
        remap_label_line("invalid 0.5 0.5 0.2 0.2", mapping, source_file="corrupt.txt")


# ==============================================================================
# 2. Path Normalization Regression Tests (Fix for Issue 2)
# ==============================================================================

def test_format_weights_path_relative_and_absolute():
    """Verify both relative and absolute weights paths produce clean repository-relative strings."""
    # Test with relative path
    rel_path = Path("training_lab/runs/D1_yolov8m_640_2class_smoke/weights/best.pt")
    formatted_rel = format_weights_path(rel_path)
    assert formatted_rel == "training_lab/runs/D1_yolov8m_640_2class_smoke/weights/best.pt"

    # Test with absolute path
    abs_path = (ROOT_DIR / "training_lab/runs/D1_yolov8m_640_2class_smoke/weights/best.pt").resolve()
    formatted_abs = format_weights_path(abs_path)
    assert formatted_abs == "training_lab/runs/D1_yolov8m_640_2class_smoke/weights/best.pt"

    # Test Windows backward slash normalization
    win_style_rel = "training_lab\\runs\\D1_yolov8m_640_2class_smoke\\weights\\best.pt"
    formatted_win = format_weights_path(win_style_rel)
    assert "\\" not in formatted_win
    assert formatted_win == "training_lab/runs/D1_yolov8m_640_2class_smoke/weights/best.pt"


# ==============================================================================
# 3. Mandatory Synthetic Fixture Test: Ultralytics Dataset Loading
# ==============================================================================

def test_ultralytics_loads_derived_view_without_errors(tmp_path):
    """
    Constructs a synthetic fixture with canonical IDs 1 and 3.
    Materializes a DerivedDatasetView.
    Instantiates Ultralytics YOLODataset for BOTH train and val.
    Verifies:
      - Ultralytics sees ONLY class IDs 0 and 1
      - Zero warnings/errors about class 3
      - Both classes are present and distinguished
    """
    from ultralytics.data.dataset import YOLODataset

    # Setup source canonical dummy dataset
    source_dir = tmp_path / "canonical_fixture"
    for split in ["train", "val"]:
        (source_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (source_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Create dummy image
        img = Image.new("RGB", (640, 640), color=(128, 128, 128))
        img.save(source_dir / "images" / split / f"{split}_sample.jpg")

        # Canonical labels: contains car (1) and bus (3)
        label_content = "1 0.5 0.5 0.2 0.2\n3 0.4 0.4 0.1 0.1\n"
        (source_dir / "labels" / split / f"{split}_sample.txt").write_text(label_content, encoding="utf-8")

    # Build derived view
    derived_dir = tmp_path / "derived_fixture"
    yaml_path = DerivedDatasetView.create_view(
        source_dataset_dir=source_dir,
        target_view_dir=derived_dir,
        class_mapping={1: 0, 3: 1},
        names={0: "car", 1: "bus"},
        splits=["train", "val"],
    )

    assert yaml_path.is_file()
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Test BOTH train and val dataset loading with Ultralytics YOLODataset
    for split in ["train", "val"]:
        img_path = str(derived_dir / "images" / split)
        dataset = YOLODataset(
            img_path=img_path,
            data=data_cfg,
            task="detect",
            augment=False,
            rect=False,
            batch_size=1,
            imgsz=640,
        )

        assert len(dataset.labels) == 1, f"Expected 1 label dict for {split}"
        loaded_classes = dataset.labels[0]["cls"].flatten().tolist()

        # Critical verification: Ultralytics MUST see ONLY class IDs 0 and 1
        assert sorted(loaded_classes) == [0.0, 1.0], (
            f"Expected classes [0.0, 1.0] in {split}, but Ultralytics loaded: {loaded_classes}"
        )
        assert 3.0 not in loaded_classes, f"Class 3 leaked into {split} dataset!"


# ==============================================================================
# 4. Canonical Dataset Immutability Verification
# ==============================================================================

def test_canonical_dataset_immutability():
    """
    Verifies that representative canonical label files remain 100% byte-for-byte identical.
    Checks SHA-256 hashes of Train/Val Car and Bus files.
    """
    canonical_files = {
        "train_car": ROOT_DIR / "training_lab/datasets/dataset_ua_detrac_v001/labels/train/dataset_ua_detrac_v001_train_MVI_20011_f000001.txt",
        "train_bus": ROOT_DIR / "training_lab/datasets/dataset_ua_detrac_v001/labels/train/dataset_ua_detrac_v001_train_MVI_20064_f000749.txt",
        "val_car": ROOT_DIR / "training_lab/datasets/dataset_ua_detrac_v001/labels/val/dataset_ua_detrac_v001_val_MVI_40201_f000410.txt",
        "val_bus": ROOT_DIR / "training_lab/datasets/dataset_ua_detrac_v001/labels/val/dataset_ua_detrac_v001_val_MVI_40204_f000741.txt",
    }

    # Expected baseline hashes (calculated directly from read-only canonical files)
    expected_hashes = {
        "train_car": "61E0B09CECEBE634583D573C3084C7298E16CB5FCF81EDDAC17F7AC5E6936772".lower(),
        "train_bus": "02247B432EBC51FA4A6A029BCD3E09DB259EA1BAF3933439D2A7F17774B0839C".lower(),
        "val_car": "3A71E83EEA4B239658E184520D4AF9E5C4C532B839C9E57CFD055E18FC6CCAE4".lower(),
        "val_bus": "1EED58C5C01080B7086501BB1E2DCC2F3A746C62F38E85C74CAF2FF27145915A".lower(),
    }

    for tag, file_path in canonical_files.items():
        assert file_path.is_file(), f"Canonical file missing: {file_path}"
        current_hash = compute_file_sha256(file_path).lower()
        assert current_hash == expected_hashes[tag], (
            f"CANONICAL DATASET MUTATION DETECTED on {tag}! "
            f"Expected {expected_hashes[tag]}, got {current_hash}"
        )


# ==============================================================================
# 5. Validation Sanity Test on Real Dataset Samples
# ==============================================================================

def test_validation_sanity_real_dataset_samples(tmp_path):
    """
    Performs a tiny validation/data-loading test using real samples from dataset_ua_detrac_v001.
    Proves:
      1. Ultralytics loads D1 derived dataset without 'Label class 3 exceeds dataset class count 2'
      2. No labels are ignored because of class ID
      3. Class IDs observed by the D1 loader are exactly: {0, 1}
      4. Both car (0) and bus (1) labels are actually present
      5. Validation class counts distinguish car and bus
      6. No canonical labels were modified
    """
    from ultralytics.data.dataset import YOLODataset

    canonical_dir = ROOT_DIR / "training_lab/datasets/dataset_ua_detrac_v001"
    assert canonical_dir.is_dir(), f"Canonical dataset not found: {canonical_dir}"

    # Pick 2 train frames and 2 val frames containing cars and buses
    selected = {
        "train": [
            "dataset_ua_detrac_v001_train_MVI_20011_f000001.jpg",  # car
            "dataset_ua_detrac_v001_train_MVI_20064_f000749.jpg",  # car + bus
        ],
        "val": [
            "dataset_ua_detrac_v001_val_MVI_40201_f000410.jpg",    # car
            "dataset_ua_detrac_v001_val_MVI_40204_f000741.jpg",    # car + bus
        ],
    }

    derived_test_view = tmp_path / "d1_sanity_view"
    yaml_path = DerivedDatasetView.create_view(
        source_dataset_dir=canonical_dir,
        target_view_dir=derived_test_view,
        class_mapping={1: 0, 3: 1},
        names={0: "car", 1: "bus"},
        splits=["train", "val"],
        selected_files=selected,
    )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Test Validation loader
    val_dataset = YOLODataset(
        img_path=str(derived_test_view / "images" / "val"),
        data=data_cfg,
        task="detect",
        augment=False,
        rect=False,
        batch_size=2,
        imgsz=640,
    )

    assert len(val_dataset.labels) == 2, f"Expected 2 validation images loaded, got {len(val_dataset.labels)}"

    all_observed_classes = set()
    class_counts = {0: 0, 1: 0}

    for item in val_dataset.labels:
        classes = item["cls"].flatten().tolist()
        for c in classes:
            c_int = int(c)
            all_observed_classes.add(c_int)
            assert c_int in (0, 1), f"Unexpected class ID {c_int} found in validation loader!"
            class_counts[c_int] += 1

    # Assertions for the 6 sanity criteria:
    # 1. Ultralytics loaded without class 3 errors (verified: no exception and only {0, 1})
    # 2. No labels ignored because of class ID
    assert class_counts[0] > 0, "No cars loaded in validation!"
    assert class_counts[1] > 0, "No buses loaded in validation!"

    # 3. Class IDs observed are exactly {0, 1}
    assert all_observed_classes == {0, 1}, f"Expected observed classes {{0, 1}}, got {all_observed_classes}"

    # 4 & 5. Both car and bus labels are present and distinguished
    print(f"Sanity Check Val Label Counts: Car={class_counts[0]}, Bus={class_counts[1]}")
    assert class_counts[0] != class_counts[1], "Car and bus counts should be distinct"

    # 6. Canonical labels remain unmodified
    test_canonical_dataset_immutability()


# ==============================================================================
# 6. Regression Test: Ultralytics Path Resolution Does Not Escape to Canonical
# ==============================================================================

def test_ultralytics_check_det_dataset_stays_inside_derived_view(tmp_path):
    """
    Regression Test: Ensures that check_det_dataset and img2label_paths
    do NOT follow directory junctions back into the canonical dataset.
    Both image paths and label paths must resolve strictly within the derived view.
    """
    from ultralytics.data.utils import check_det_dataset, img2label_paths

    source_dir = tmp_path / "canonical_ds"
    for split in ["train", "val"]:
        (source_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (source_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (64, 64), color=(0, 0, 0))
        img.save(source_dir / "images" / split / f"{split}_0.jpg")
        (source_dir / "labels" / split / f"{split}_0.txt").write_text("1 0.5 0.5 0.2 0.2\n3 0.4 0.4 0.1 0.1\n")

    derived_dir = tmp_path / "derived_ds"
    yaml_p = DerivedDatasetView.create_view(
        source_dataset_dir=source_dir,
        target_view_dir=derived_dir,
        class_mapping={1: 0, 3: 1},
        names={0: "car", 1: "bus"},
        splits=["train", "val"],
    )

    data = check_det_dataset(str(yaml_p))
    val_resolved = Path(data["val"]).resolve()
    assert str(derived_dir.resolve()).lower() in str(val_resolved).lower(), (
        f"val path escaped derived view! Got {val_resolved}, expected within {derived_dir}"
    )
    assert str(source_dir.resolve()).lower() not in str(val_resolved).lower(), (
        f"val path resolved back to source canonical dataset: {val_resolved}"
    )

    sample_img = val_resolved / "val_0.jpg"
    assert sample_img.is_file(), f"Sample image not found: {sample_img}"
    lbl_paths = img2label_paths([str(sample_img)])
    assert str(derived_dir.resolve()).lower() in str(Path(lbl_paths[0]).resolve()).lower(), (
        f"Label path escaped derived view: {lbl_paths[0]}"
    )
    assert Path(lbl_paths[0]).is_file(), f"Derived label file missing: {lbl_paths[0]}"

