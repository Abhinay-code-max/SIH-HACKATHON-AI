"""
Tests for Unified 9-Class Dataset & U1 Preflight Hardening Pipeline.
Validates:
1. D1 Baseline Immutability (weights exist, data.yaml SHA-256 byte-for-byte exact).
2. Unified Dataset Structure & 1:1 Image/Label Pairing.
3. Coordinate Normalization ([0, 1] range, no degenerate boxes).
4. Canonical 9-Class Taxonomy Adherence (all classes in {0..8} and represented in all splits).
5. Split Disjointness & Zero Cross-Split Leakage.
6. Manifest Integrity (data.yaml, dataset_manifest.json, split_manifest.json, source_manifest.json).
7. Forensics Report & Visual QA Artifacts.
8. U1 17-Point Preflight Runner Execution & Report.
"""

import hashlib
import json
from pathlib import Path
import pytest
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent

# D1 Baseline Constants
D1_WEIGHTS_PATH = ROOT_DIR / "training_lab" / "runs" / "D1_yolov8m_640_2class_full" / "weights" / "best.pt"
D1_DATA_YAML_PATH = ROOT_DIR / "training_lab" / "datasets" / "dataset_ua_detrac_v001" / "data.yaml"
D1_DATA_YAML_SHA = "594bf7a2b6a870faa1c176a1904c524cfb287010118f88ee0d0b1402c8eae6b9"

# Unified Dataset Constants
UNIFIED_DIR = ROOT_DIR / "training_lab" / "datasets" / "dataset_unified_v001"
EXPECTED_TRAIN = 2560
EXPECTED_VAL = 548
EXPECTED_HOLDOUT = 550
EXPECTED_TOTAL = 3658


def compute_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_d1_immutability():
    """Verify that D1 baseline remains strictly untouched and locked."""
    assert D1_WEIGHTS_PATH.is_file(), f"D1 best.pt missing at {D1_WEIGHTS_PATH}"
    assert D1_DATA_YAML_PATH.is_file(), f"D1 data.yaml missing at {D1_DATA_YAML_PATH}"
    actual_sha = compute_sha256(D1_DATA_YAML_PATH)
    assert actual_sha == D1_DATA_YAML_SHA, f"D1 data.yaml SHA mismatch! Expected {D1_DATA_YAML_SHA}, got {actual_sha}"


def test_unified_dataset_counts_and_pairing():
    """Verify split image counts and 100% label pairing."""
    assert UNIFIED_DIR.is_dir(), f"Unified dataset missing at {UNIFIED_DIR}"

    for split, expected in [("train", EXPECTED_TRAIN), ("val", EXPECTED_VAL), ("holdout", EXPECTED_HOLDOUT)]:
        img_dir = UNIFIED_DIR / "images" / split
        lbl_dir = UNIFIED_DIR / "labels" / split

        assert img_dir.is_dir(), f"Images dir missing for split {split}"
        assert lbl_dir.is_dir(), f"Labels dir missing for split {split}"

        imgs = sorted(list(img_dir.glob("*.jpg")))
        lbls = sorted(list(lbl_dir.glob("*.txt")))

        assert len(imgs) == expected, f"Split {split}: expected {expected} images, found {len(imgs)}"
        assert len(lbls) == expected, f"Split {split}: expected {expected} labels, found {len(lbls)}"

        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        assert img_stems == lbl_stems, f"Image/label pairing mismatch in split {split}"


def test_label_bounding_box_validity_and_taxonomy():
    """Verify bounding box coordinates are normalized and within [0, 1] with classes in {0..8}."""
    splits = ["train", "val", "holdout"]
    classes_seen_per_split = {s: set() for s in splits}

    for s in splits:
        lbl_dir = UNIFIED_DIR / "labels" / s
        for lbl_file in lbl_dir.glob("*.txt"):
            content = lbl_file.read_text(encoding="utf-8").strip()
            assert len(content) > 0, f"Empty label file: {lbl_file}"
            for line in content.splitlines():
                parts = line.strip().split()
                assert len(parts) == 5, f"Malformed row in {lbl_file}: {line}"

                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                assert 0 <= cls_id <= 8, f"Invalid class ID {cls_id} in {lbl_file}"
                assert 0.0 <= cx <= 1.0, f"cx out of range [0, 1]: {cx} in {lbl_file}"
                assert 0.0 <= cy <= 1.0, f"cy out of range [0, 1]: {cy} in {lbl_file}"
                assert 0.0 < w <= 1.0, f"w out of range (0, 1]: {w} in {lbl_file}"
                assert 0.0 < h <= 1.0, f"h out of range (0, 1]: {h} in {lbl_file}"

                classes_seen_per_split[s].add(cls_id)

    # Verify all 9 classes exist in every split
    expected_classes = set(range(9))
    for s in splits:
        assert classes_seen_per_split[s] == expected_classes, (
            f"Split {s} missing classes: {expected_classes - classes_seen_per_split[s]}"
        )


def test_split_disjointness():
    """Verify zero overlap between splits (train, val, holdout)."""
    train_stems = {p.stem for p in (UNIFIED_DIR / "images" / "train").glob("*.jpg")}
    val_stems = {p.stem for p in (UNIFIED_DIR / "images" / "val").glob("*.jpg")}
    holdout_stems = {p.stem for p in (UNIFIED_DIR / "images" / "holdout").glob("*.jpg")}

    assert len(train_stems & val_stems) == 0, "Leakage detected between train and val!"
    assert len(train_stems & holdout_stems) == 0, "Leakage detected between train and holdout!"
    assert len(val_stems & holdout_stems) == 0, "Leakage detected between val and holdout!"


def test_manifests():
    """Verify data.yaml and manifest JSONs exist, parse cleanly, and match specs."""
    data_yaml = UNIFIED_DIR / "data.yaml"
    assert data_yaml.is_file(), "data.yaml missing"
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert len(cfg.get("names", {})) == 9
    assert cfg["names"][0] == "person"
    assert cfg["names"][8] == "bag"

    # Dataset manifest
    dmanifest = UNIFIED_DIR / "dataset_manifest.json"
    assert dmanifest.is_file(), "dataset_manifest.json missing"
    with open(dmanifest, "r", encoding="utf-8") as f:
        d_data = json.load(f)
    assert d_data["splits"]["train"] == EXPECTED_TRAIN
    assert d_data["splits"]["val"] == EXPECTED_VAL
    assert d_data["splits"]["holdout"] == EXPECTED_HOLDOUT

    # Split manifest
    smanifest = UNIFIED_DIR / "split_manifest.json"
    assert smanifest.is_file(), "split_manifest.json missing"
    with open(smanifest, "r", encoding="utf-8") as f:
        s_data = json.load(f)
    assert s_data["counts"]["total"] == EXPECTED_TOTAL


def test_forensics_reports():
    """Verify forensic analysis outputs exist."""
    forensics_dir = ROOT_DIR / "training_lab" / "reports" / "dataset_unified_forensics_v001"
    assert forensics_dir.is_dir(), "Forensics reports directory missing"
    assert (forensics_dir / "DATASET_FORENSIC_REPORT.md").is_file()
    assert (forensics_dir / "chart_class_distribution.png").is_file()
    assert (forensics_dir / "chart_bbox_sizes.png").is_file()
    assert (forensics_dir / "chart_split_comparison.png").is_file()
    assert (forensics_dir / "visual_qa_montage_4x4.jpg").is_file()


def test_u1_preflight_report():
    """Verify U1 preflight verification report confirms all 17 checks passed."""
    preflight_json = ROOT_DIR / "training_lab" / "reports" / "u1_preflight_report.json"
    assert preflight_json.is_file(), "u1_preflight_report.json missing"
    with open(preflight_json, "r", encoding="utf-8") as f:
        p_data = json.load(f)
    assert p_data["status"] == "PREFLIGHT_PASSED"
    assert p_data["all_checks_passed"] is True
    assert len(p_data["checks"]) == 17
    for check_name, check_val in p_data["checks"].items():
        assert check_val["status"] == "PASS", f"Check {check_name} failed in preflight report!"
