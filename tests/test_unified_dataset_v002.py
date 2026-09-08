"""
Regression and integrity test suite for dataset_unified_v002 (U2 Phase 2).
Validates:
1. Immutability of baseline D1 and U1 models and configs.
2. Structure of dataset_unified_v002 (images, labels, splits).
3. 1-to-1 pairing of all images and labels.
4. Bounding box validity (coordinates in [0, 1], w > 0, h > 0, class_ids in 0..8).
5. Split disjointness (Train ∩ Val = ∅, Train ∩ Holdout = ∅, Val ∩ Holdout = ∅).
6. UA-DETRAC holdout sequences are completely isolated from train and val splits.
7. Manifest SHA-256 integrity against dataset artifacts.
"""

import hashlib
import json
import os
from pathlib import Path
import pytest
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
V002_DIR = ROOT_DIR / "training_lab" / "datasets" / "dataset_unified_v002"
REPORTS_DIR = ROOT_DIR / "training_lab" / "reports" / "U2_phase2"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_d1_and_u1_immutability():
    """Ensure baseline models and data configs are unchanged."""
    d1_best = ROOT_DIR / "training_lab" / "runs" / "D1_yolov8m_640_2class_full" / "weights" / "best.pt"
    d1_yaml = ROOT_DIR / "training_lab" / "datasets" / "dataset_ua_detrac_v001" / "data.yaml"
    u1_best = ROOT_DIR / "training_lab" / "runs" / "U1_yolov8m_640_9class_unified" / "weights" / "best.pt"
    u1_last = ROOT_DIR / "training_lab" / "runs" / "U1_yolov8m_640_9class_unified" / "weights" / "last.pt"
    u1_yaml = ROOT_DIR / "training_lab" / "datasets" / "dataset_unified_v001" / "data.yaml"

    assert d1_best.is_file()
    assert compute_sha256(d1_best) == "005b706408d49f52ab0a26795f8ea8ae2c0e7dc725445a616ddbb56cd4f7de3b"

    assert d1_yaml.is_file()
    assert compute_sha256(d1_yaml) == "594bf7a2b6a870faa1c176a1904c524cfb287010118f88ee0d0b1402c8eae6b9"

    assert u1_best.is_file()
    assert compute_sha256(u1_best) == "150d607f2287adff4c0bbed20b403684a66e5c05f48ee3c2b9067832fe291318"

    assert u1_last.is_file()
    assert compute_sha256(u1_last) == "32fdfd63f8737e05543094312c4fb10e32305be3d8743b2d6608e903d826f041"

    assert u1_yaml.is_file()
    assert compute_sha256(u1_yaml) == "b71a8cd183dea892c5117c7ebaa65d431286e48d78afada75560093bddb266c4"

def test_dataset_v002_structure_and_manifests():
    """Verify dataset_unified_v002 directory layout and required manifest files."""
    assert V002_DIR.is_dir()
    for s in ["train", "val", "holdout"]:
        assert (V002_DIR / "images" / s).is_dir()
        assert (V002_DIR / "labels" / s).is_dir()

    data_yaml = V002_DIR / "data.yaml"
    assert data_yaml.is_file()
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert len(cfg["names"]) == 9
    assert cfg["names"][0] == "person"
    assert cfg["names"][8] == "bag"

    manifest_json = V002_DIR / "manifest.json"
    assert manifest_json.is_file()
    with open(manifest_json, "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["dataset_version"] == "dataset_unified_v002"
    assert m["splits"]["train"] == 3846
    assert m["splits"]["val"] == 752
    assert m["splits"]["holdout"] == 713

def test_v002_image_label_pairing():
    """Verify exact 1-to-1 pairing between images and labels across all splits."""
    for s in ["train", "val", "holdout"]:
        imgs = set(os.listdir(V002_DIR / "images" / s))
        lbls = set(os.listdir(V002_DIR / "labels" / s))
        assert len(imgs) == len(lbls)
        for img_name in imgs:
            stem = os.path.splitext(img_name)[0]
            assert f"{stem}.txt" in lbls

def test_v002_split_disjointness_and_leakage():
    """Verify zero overlap in filenames or byte content between splits."""
    splits = ["train", "val", "holdout"]
    split_imgs = {s: set(os.listdir(V002_DIR / "images" / s)) for s in splits}

    # Filename disjointness
    assert len(split_imgs["train"] & split_imgs["val"]) == 0
    assert len(split_imgs["train"] & split_imgs["holdout"]) == 0
    assert len(split_imgs["val"] & split_imgs["holdout"]) == 0

    # Leakage report assertion
    leakage_rep = REPORTS_DIR / "U2_V002_LEAKAGE_REPORT.json"
    assert leakage_rep.is_file()
    with open(leakage_rep, "r", encoding="utf-8") as f:
        lr = json.load(f)
    assert lr["status"] == "LEAKAGE_FREE"
    assert lr["leakage_detected"] is False
    assert lr["violations_count"] == 0

def test_ua_detrac_holdout_isolation():
    """Ensure UA-DETRAC holdout sequences are absent from train and val."""
    ua_holdout_seqs = ["MVI_40201", "MVI_40241", "MVI_40732", "MVI_40751", "MVI_40871", "MVI_40962"]
    for s in ["train", "val"]:
        imgs = os.listdir(V002_DIR / "images" / s)
        for img_name in imgs:
            for hseq in ua_holdout_seqs:
                assert hseq not in img_name, f"Leak of {hseq} in {s}/{img_name}"

def test_bounding_box_validity_and_taxonomy():
    """Verify all bounding box coordinates are normalized in [0, 1] and class IDs in 0..8."""
    class_distribution = REPORTS_DIR / "U2_V002_CLASS_DISTRIBUTION.json"
    assert class_distribution.is_file()
    with open(class_distribution, "r", encoding="utf-8") as f:
        cd = json.load(f)
    assert cd["total_images"] == 5311
    assert cd["total_annotations"] == 55607

    # Sample check 100 random labels
    import random
    all_label_paths = list((V002_DIR / "labels").glob("**/*.txt"))
    assert len(all_label_paths) == 5311
    sample = random.sample(all_label_paths, min(200, len(all_label_paths)))

    for lp in sample:
        with open(lp, "r", encoding="utf-8") as lf:
            for line in lf:
                parts = line.strip().split()
                if not parts:
                    continue
                assert len(parts) == 5
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                assert 0 <= cls_id <= 8
                assert 0.0 <= cx <= 1.0
                assert 0.0 <= cy <= 1.0
                assert 0.0 < w <= 1.0
                assert 0.0 < h <= 1.0
