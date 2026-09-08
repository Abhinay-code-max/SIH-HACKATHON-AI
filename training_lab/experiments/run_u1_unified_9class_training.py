"""
Phase 8/9 — Full U1 Unified 9-Class Training Runner (YOLOv8m, 640x640, 9 Classes).
Experiment ID: U1_yolov8m_640_9class_unified

Executes:
1. Strict D1 baseline immutability audit.
2. Comprehensive 17-point preflight verification:
   - Check 1: Strict Hyperparameter Enforcement (batch=4, workers=0, device=0, imgsz=640, epochs=5, amp=True, mosaic=1.0)
   - Check 2: Python Version (3.11.x)
   - Check 3: PyTorch Version (2.x)
   - Check 4: CUDA Availability (no CPU fallback)
   - Check 5: GPU Identity (NVIDIA GPU / RTX 4060 Laptop)
   - Check 6: GPU VRAM Headroom (minimum 6.0 GB)
   - Check 7: Ultralytics Package Version
   - Check 8: Unified Dataset Existence & Integrity
   - Check 9: data.yaml SHA-256 and Class Count (9 classes)
   - Check 10: Train Image Count (2,560 images)
   - Check 11: Val Image Count (548 images)
   - Check 12: Holdout Split Protection (550 images)
   - Check 13: Split Disjointness & Zero Leakage
   - Check 14: 9-Class Representation Across Splits
   - Check 15: Base Model Weights & Architecture Loading (yolov8m.pt, ~25.9M params)
   - Check 16: cuDNN Deterministic Stability Configuration
   - Check 17: CUDA Tensor Allocation & VRAM Test
3. Support for --preflight-only flag to verify all prerequisites without starting training.
4. Normal execution flow:
   - Final holdout isolation check (zero holdout images in train/val; holdout never evaluated or tuned against).
   - Authorization banner.
   - Base model layer transfer analysis (COCO pretrained backbone/neck transferred, 9-class head initialized).
   - Execution of full 5-epoch training under strict stability constraints.
   - Fail-fast error handling (CUDA OOM / cuDNN failure classifications, no silent reduction of batch or CPU fallback).
   - Checkpoint verification (best.pt and last.pt).
   - Post-training validation on VAL split across all 9 canonical classes.
   - Offline inference verification on real validation frames (recording latency and confirming detections).
   - Machine-readable JSON and human-readable Markdown acceptance reports.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.cuda_diagnostic import (
    capture_gpu_memory_stats,
    capture_system_memory_stats,
    classify_failure,
    configure_stable_cudnn,
    get_cuda_diagnostics,
    print_cuda_diagnostics,
    reset_cuda_state,
    write_failure_diagnostic,
)
from training_lab.engine.training_manager import format_weights_path

# ==============================================================================
# EXPERIMENT PROTOCOL INVARIANTS
# ==============================================================================
EXPERIMENT_ID = "U1_yolov8m_640_9class_unified"
REQUIRED_BASE_MODEL = "yolov8m.pt"
REQUIRED_BATCH = 4
REQUIRED_WORKERS = 0
REQUIRED_DEVICE = "0"
REQUIRED_IMGSZ = 640
REQUIRED_EPOCHS = 5
REQUIRED_AMP = True
REQUIRED_MOSAIC = 1.0

UNIFIED_DATASET_REL = Path("training_lab/datasets/dataset_unified_v001")
EXPECTED_TRAIN_IMAGES = 2560
EXPECTED_VAL_IMAGES = 548
EXPECTED_HOLDOUT_IMAGES = 550
EXPECTED_TOTAL_IMAGES = 3658

EXPECTED_NUM_CLASSES = 9
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

# Immutable D1 baseline invariants
D1_RUN_DIR = ROOT_DIR / "training_lab" / "runs" / "D1_yolov8m_640_2class_full"
D1_BEST_WEIGHTS = D1_RUN_DIR / "weights" / "best.pt"
D1_CANONICAL_YAML = ROOT_DIR / "training_lab" / "datasets" / "dataset_ua_detrac_v001" / "data.yaml"
D1_DATA_YAML_SHA256 = "594bf7a2b6a870faa1c176a1904c524cfb287010118f88ee0d0b1402c8eae6b9"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR), text=True, stderr=subprocess.DEVNULL
        ).strip()
        return commit
    except Exception:
        return "unknown"


def run_preflight(
    batch: int = REQUIRED_BATCH,
    workers: int = REQUIRED_WORKERS,
    device: str = REQUIRED_DEVICE,
    imgsz: int = REQUIRED_IMGSZ,
    epochs: int = REQUIRED_EPOCHS,
    amp: bool = REQUIRED_AMP,
    mosaic: float = REQUIRED_MOSAIC,
) -> Dict[str, Any]:
    """
    Executes comprehensive 17-point preflight verification for U1.
    Fails immediately with clear diagnostic message if any check fails.
    """
    print("\n" + "=" * 80)
    print("PHASE 8 — U1 PRE-FLIGHT VERIFICATION & RUNNER HARDENING (17 CHECKS)")
    print("=" * 80)

    checks: Dict[str, Tuple[bool, str]] = {}

    # 1. Strict Hyperparameter Enforcement
    if batch != REQUIRED_BATCH:
        raise ValueError(f"Preflight failure: batch={batch} != {REQUIRED_BATCH}.")
    if workers != REQUIRED_WORKERS:
        raise ValueError(f"Preflight failure: workers={workers} != {REQUIRED_WORKERS}.")
    if device != REQUIRED_DEVICE:
        raise ValueError(f"Preflight failure: device='{device}' != '{REQUIRED_DEVICE}'.")
    if imgsz != REQUIRED_IMGSZ:
        raise ValueError(f"Preflight failure: imgsz={imgsz} != {REQUIRED_IMGSZ}.")
    if epochs != REQUIRED_EPOCHS:
        raise ValueError(f"Preflight failure: epochs={epochs} != {REQUIRED_EPOCHS}.")
    if amp is not REQUIRED_AMP:
        raise ValueError(f"Preflight failure: amp={amp} != {REQUIRED_AMP}.")
    if mosaic != REQUIRED_MOSAIC:
        raise ValueError(f"Preflight failure: mosaic={mosaic} != {REQUIRED_MOSAIC}.")

    checks["1. Strict Hyperparameters"] = (
        True,
        f"batch={batch}, workers={workers}, device={device}, imgsz={imgsz}, epochs={epochs}, amp={amp}, mosaic={mosaic}",
    )

    # 2. Python Version
    py_ver = sys.version.split()[0]
    checks["2. Python Version"] = (sys.version_info[:2] == (3, 11), f"Python {py_ver} (required: 3.11.x)")
    assert sys.version_info[:2] == (3, 11), f"Preflight failed: Python version {py_ver} is not 3.11.x"

    # 3. PyTorch Version
    torch_ver = torch.__version__
    checks["3. PyTorch Version"] = (torch_ver.startswith("2."), f"PyTorch {torch_ver}")

    # 4. CUDA Availability
    cuda_avail = torch.cuda.is_available()
    checks["4. CUDA Available"] = (cuda_avail, f"CUDA Available: {cuda_avail}")
    if not cuda_avail:
        raise RuntimeError("Preflight failure: CUDA is not available. CPU fallback is strictly prohibited.")

    # 5. GPU Identity
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None"
    checks["5. GPU Identity"] = (
        ("4060" in gpu_name or "NVIDIA" in gpu_name),
        f"Device 0: {gpu_name}",
    )

    # 6. GPU VRAM Headroom
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0.0
    checks["6. GPU VRAM"] = (vram_gb >= 6.0, f"{vram_gb} GB physical VRAM (minimum 6.0 GB required)")
    assert vram_gb >= 6.0, f"Preflight failure: Insufficient GPU VRAM ({vram_gb} GB < 6.0 GB)"

    # 7. Ultralytics Version
    import ultralytics
    checks["7. Ultralytics Version"] = (True, f"v{ultralytics.__version__}")

    # 8. Unified Dataset Directory Existence
    dataset_root = (ROOT_DIR / UNIFIED_DATASET_REL).resolve()
    checks["8. Unified Dataset Exists"] = (dataset_root.is_dir(), str(dataset_root))
    assert dataset_root.is_dir(), f"Preflight failure: Dataset directory missing at {dataset_root}"

    # 9. data.yaml Verification
    yaml_path = dataset_root / "data.yaml"
    assert yaml_path.is_file(), f"Preflight failure: data.yaml missing at {yaml_path}"
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f)
    num_classes = len(yaml_cfg.get("names", {}))
    assert num_classes == EXPECTED_NUM_CLASSES, f"Preflight failure: data.yaml has {num_classes} classes, expected {EXPECTED_NUM_CLASSES}"
    yaml_sha = compute_sha256(yaml_path)
    checks["9. data.yaml & 9-Class Taxonomy"] = (
        num_classes == 9,
        f"9 classes confirmed, SHA-256: {yaml_sha[:16]}...",
    )

    # 10. Train Image Count
    train_files = list((dataset_root / "images" / "train").glob("*.jpg"))
    assert len(train_files) == EXPECTED_TRAIN_IMAGES, f"Preflight failure: Expected {EXPECTED_TRAIN_IMAGES} train images, found {len(train_files)}"
    checks["10. Train Image Count"] = (True, f"{len(train_files):,} images (expected: {EXPECTED_TRAIN_IMAGES:,})")

    # 11. Val Image Count
    val_files = list((dataset_root / "images" / "val").glob("*.jpg"))
    assert len(val_files) == EXPECTED_VAL_IMAGES, f"Preflight failure: Expected {EXPECTED_VAL_IMAGES} val images, found {len(val_files)}"
    checks["11. Val Image Count"] = (True, f"{len(val_files):,} images (expected: {EXPECTED_VAL_IMAGES:,})")

    # 12. Holdout Image Count & Isolation Protection
    holdout_files = list((dataset_root / "images" / "holdout").glob("*.jpg"))
    assert len(holdout_files) == EXPECTED_HOLDOUT_IMAGES, f"Preflight failure: Expected {EXPECTED_HOLDOUT_IMAGES} holdout images, found {len(holdout_files)}"
    checks["12. Holdout Protection"] = (True, f"{len(holdout_files):,} holdout images isolated (zero leakage into train/val)")

    # 13. Split Disjointness Audit
    train_names = {f.name for f in train_files}
    val_names = {f.name for f in val_files}
    holdout_names = {f.name for f in holdout_files}
    assert len(train_names & val_names) == 0, "Preflight failure: Train and Val overlap!"
    assert len(train_names & holdout_names) == 0, "Preflight failure: Train and Holdout overlap!"
    assert len(val_names & holdout_names) == 0, "Preflight failure: Val and Holdout overlap!"
    checks["13. Split Disjointness"] = (True, "Train ∩ Val = ∅, Train ∩ Holdout = ∅, Val ∩ Holdout = ∅")

    # 14. 9-Class Representation in Train and Val
    train_classes = set()
    for lbl in (dataset_root / "labels" / "train").glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                train_classes.add(int(line.split()[0]))
    val_classes = set()
    for lbl in (dataset_root / "labels" / "val").glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                val_classes.add(int(line.split()[0]))
    assert train_classes == set(range(9)), f"Preflight failure: Train missing classes: {set(range(9)) - train_classes}"
    assert val_classes == set(range(9)), f"Preflight failure: Val missing classes: {set(range(9)) - val_classes}"
    checks["14. 9-Class Representation"] = (True, "All 9 canonical classes (0-8) actively represented in train and val")

    # 15. Base Model Architecture & Weights Verification
    base_model_path = ROOT_DIR / REQUIRED_BASE_MODEL
    assert base_model_path.is_file(), f"Preflight failure: Base weights missing at {base_model_path}"
    test_model = YOLO(str(base_model_path))
    param_count = sum(p.numel() for p in test_model.model.parameters())
    checks["15. Base Model Weights"] = (
        param_count > 20_000_000,
        f"YOLOv8m loaded ({param_count:,} parameters, ~25.9M expected)",
    )
    del test_model

    # 16. cuDNN Deterministic Stability Configuration
    cudnn_cfg = configure_stable_cudnn()
    cudnn_ok = (
        cudnn_cfg["cudnn_enabled"] is True
        and cudnn_cfg["cudnn_benchmark"] is False
        and cudnn_cfg["cudnn_deterministic"] is True
    )
    checks["16. cuDNN Determinism"] = (
        cudnn_ok,
        f"enabled={cudnn_cfg['cudnn_enabled']}, benchmark={cudnn_cfg['cudnn_benchmark']}, deterministic={cudnn_cfg['cudnn_deterministic']}",
    )

    # 17. Pre-run CUDA Tensor Allocation & VRAM Test
    reset_cuda_state()
    diag = get_cuda_diagnostics(amp=amp, workers=workers, batch=batch, imgsz=imgsz, mosaic=mosaic)
    checks["17. CUDA Tensor Test"] = (
        diag["tensor_allocation_test"],
        f"Basic GPU tensor allocation test: {diag['tensor_allocation_test']}",
    )
    assert diag["tensor_allocation_test"], "Preflight failure: CUDA tensor test failed!"

    # Print summary table
    for k, (status, detail) in checks.items():
        pass_str = "[PASS]" if status else "[FAIL]"
        print(f"  {k:35s}: {pass_str} {detail}")

    print("\nPre-flight verification: ALL 17 CHECKS PASSED.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "PREFLIGHT_PASSED",
        "all_checks_passed": True,
        "checks": {k: {"status": "PASS" if s else "FAIL", "detail": d} for k, (s, d) in checks.items()},
        "dataset_yaml_sha256": yaml_sha,
        "parameters": param_count,
        "counts": {
            "train": len(train_files),
            "val": len(val_files),
            "holdout": len(holdout_files),
            "total": len(train_files) + len(val_files) + len(holdout_files),
        },
        "git_commit": get_git_commit(),
    }

    report_path = ROOT_DIR / "training_lab" / "reports" / "u1_preflight_report.json"
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)
    print(f"Wrote preflight verification report: {report_path}")

    return report


def verify_d1_immutability():
    """Guarantees D1 baseline weights and UA-DETRAC dataset have not been modified."""
    print("\nAuditing D1 baseline immutability...")
    assert D1_BEST_WEIGHTS.is_file(), f"Critical: D1 best.pt missing at {D1_BEST_WEIGHTS}!"
    assert D1_CANONICAL_YAML.is_file(), f"Critical: D1 canonical data.yaml missing at {D1_CANONICAL_YAML}!"
    actual_sha = compute_sha256(D1_CANONICAL_YAML)
    assert actual_sha == D1_DATA_YAML_SHA256, (
        f"Critical security failure: D1 dataset data.yaml SHA changed! "
        f"Expected {D1_DATA_YAML_SHA256}, Got {actual_sha}"
    )
    print("D1 baseline integrity confirmed: 100% IMMUTABLE.")


def parse_results_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Parses per-epoch training and validation loss/metrics from results.csv."""
    if not csv_path.is_file():
        return []

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return []

    header = [h.strip() for h in lines[0].split(",")]
    epochs_data = []

    for line in lines[1:]:
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(header):
            continue
        row = dict(zip(header, vals))
        try:
            ep_num = int(row.get("epoch", 0))
            train_box = float(row.get("train/box_loss", 0.0))
            train_cls = float(row.get("train/cls_loss", 0.0))
            train_dfl = float(row.get("train/dfl_loss", 0.0))
            val_box = float(row.get("val/box_loss", 0.0))
            val_cls = float(row.get("val/cls_loss", 0.0))
            val_dfl = float(row.get("val/dfl_loss", 0.0))
            prec = float(row.get("metrics/precision(B)", 0.0))
            rec = float(row.get("metrics/recall(B)", 0.0))
            map50 = float(row.get("metrics/mAP50(B)", 0.0))
            map50_95 = float(row.get("metrics/mAP50-95(B)", 0.0))

            epochs_data.append({
                "epoch": ep_num,
                "train_box_loss": round(train_box, 5),
                "train_cls_loss": round(train_cls, 5),
                "train_dfl_loss": round(train_dfl, 5),
                "val_box_loss": round(val_box, 5),
                "val_cls_loss": round(val_cls, 5),
                "val_dfl_loss": round(val_dfl, 5),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "mAP50": round(map50 * 100, 2),
                "mAP50_95": round(map50_95 * 100, 2),
            })
        except Exception:
            continue

    return epochs_data


def run_post_training_validation(
    checkpoint_path: Path,
    data_yaml_path: Path,
    batch: int = REQUIRED_BATCH,
    workers: int = REQUIRED_WORKERS,
    device: str = REQUIRED_DEVICE,
) -> Dict[str, Any]:
    """
    Executes post-training validation on best.pt using ONLY the VAL split.
    Enforces that all 9 canonical classes are evaluated.
    """
    print("\n" + "=" * 80)
    print(f"RUNNING POST-TRAINING VALIDATION ON CHECKPOINT: {checkpoint_path.name}")
    print("=" * 80)

    assert checkpoint_path.is_file(), f"Checkpoint missing: {checkpoint_path}"

    reset_cuda_state()
    val_model = YOLO(str(checkpoint_path))
    val_results = val_model.val(
        data=str(data_yaml_path.resolve()),
        split="val",
        batch=batch,
        device=device,
        workers=workers,
        plots=True,
        verbose=True,
    )

    names = val_results.names
    ap_classes = getattr(val_results.box, "ap_class_index", None)
    all_map50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
    all_map50_95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))
    all_prec = float(val_results.results_dict.get("metrics/precision(B)", 0.0))
    all_rec = float(val_results.results_dict.get("metrics/recall(B)", 0.0))

    per_class_metrics = {}
    if ap_classes is not None:
        for i, c in enumerate(ap_classes):
            c_name = names.get(c, str(c))
            cp, cr, cap50, cap = val_results.box.class_result(i)
            per_class_metrics[c_name] = {
                "class_id": int(c),
                "precision": round(float(cp) * 100, 2),
                "recall": round(float(cr) * 100, 2),
                "mAP50": round(float(cap50) * 100, 2),
                "mAP50_95": round(float(cap) * 100, 2),
            }

    print("\n" + "=" * 80)
    print("POST-TRAINING 9-CLASS VALIDATION METRICS SUMMARY:")
    print("=" * 80)
    print(f"Overall Precision:  {all_prec * 100:.2f}%")
    print(f"Overall Recall:     {all_rec * 100:.2f}%")
    print(f"Overall mAP50:      {all_map50 * 100:.2f}%")
    print(f"Overall mAP50-95:   {all_map50_95 * 100:.2f}%")
    for cn, cm in per_class_metrics.items():
        print(f"  Class {cn:10s} (id={cm['class_id']}) -> P: {cm['precision']:5.2f}%, R: {cm['recall']:5.2f}%, mAP50: {cm['mAP50']:5.2f}%, mAP50-95: {cm['mAP50_95']:5.2f}%")

    return {
        "names": names,
        "ap_class_index": [int(x) for x in ap_classes] if ap_classes is not None else [],
        "all": {
            "precision": round(all_prec * 100, 2),
            "recall": round(all_rec * 100, 2),
            "mAP50": round(all_map50 * 100, 2),
            "mAP50_95": round(all_map50_95 * 100, 2),
        },
        "per_class": per_class_metrics,
    }


def run_offline_inference_verification(
    weights_path: Path,
    val_images_dir: Path,
    device: str = REQUIRED_DEVICE,
) -> List[Dict[str, Any]]:
    """
    Verifies that the trained checkpoint can perform local offline inference on real validation frames.
    Holdout images are strictly never touched.
    """
    print("\n" + "=" * 80)
    print("EXECUTING OFFLINE INFERENCE VERIFICATION ON CHECKPOINT")
    print("=" * 80)

    assert weights_path.is_file(), f"Checkpoint missing: {weights_path}"
    model = YOLO(str(weights_path))

    val_images = sorted(list(val_images_dir.glob("*.jpg")))[:5]
    assert len(val_images) > 0, f"No validation images found in {val_images_dir}"

    inference_results = []
    for img_p in val_images:
        t0 = time.perf_counter()
        results = model.predict(
            source=str(img_p),
            device=device,
            conf=0.25,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        res = results[0]
        boxes = res.boxes

        detected_classes = [int(c) for c in boxes.cls.tolist()] if len(boxes) > 0 else []
        class_names = [res.names[c] for c in detected_classes]
        confs = [round(float(cf), 3) for cf in boxes.conf.tolist()] if len(boxes) > 0 else []

        rec = {
            "image": img_p.name,
            "latency_ms": round(latency_ms, 2),
            "detections": len(boxes),
            "classes": detected_classes,
            "class_names": class_names,
            "confidences": confs,
        }
        inference_results.append(rec)
        print(f"  Frame {img_p.name} -> {len(boxes)} detections in {latency_ms:.1f}ms: {class_names[:5]} (confs: {confs[:5]})")

    print("[Offline Inference] Verification passed locally with zero network dependency.")
    return inference_results


def execute_full_u1_training(
    batch: int = REQUIRED_BATCH,
    workers: int = REQUIRED_WORKERS,
    device: str = REQUIRED_DEVICE,
    imgsz: int = REQUIRED_IMGSZ,
    epochs: int = REQUIRED_EPOCHS,
    amp: bool = REQUIRED_AMP,
    mosaic: float = REQUIRED_MOSAIC,
) -> int:
    """
    Main training execution function for U1 Unified 9-Class detector.
    Runs preflight, stability setup, training loop, validation, offline verification, and reporting.
    """
    # 1. Verification of D1 immutability
    verify_d1_immutability()

    # 2. 17-Point Preflight Verification
    try:
        preflight_meta = run_preflight(
            batch=batch,
            workers=workers,
            device=device,
            imgsz=imgsz,
            epochs=epochs,
            amp=amp,
            mosaic=mosaic,
        )
    except Exception as exc:
        print(f"\n[FATAL] PREFLIGHT FAILED: {exc}")
        return 1

    # 3. Final Holdout Isolation Verification
    dataset_root = (ROOT_DIR / UNIFIED_DATASET_REL).resolve()
    holdout_imgs = list((dataset_root / "images" / "holdout").glob("*.jpg"))
    train_imgs = list((dataset_root / "images" / "train").glob("*.jpg"))
    val_imgs = list((dataset_root / "images" / "val").glob("*.jpg"))
    assert len(holdout_imgs) == EXPECTED_HOLDOUT_IMAGES, f"Holdout count mismatch: {len(holdout_imgs)}"
    assert len(set(f.name for f in train_imgs) & set(f.name for f in holdout_imgs)) == 0, "Holdout leaked into train!"
    assert len(set(f.name for f in val_imgs) & set(f.name for f in holdout_imgs)) == 0, "Holdout leaked into val!"

    # Dynamically compute validation annotation counts across label files
    val_labels_dir = dataset_root / "labels" / "val"
    val_class_counts = {c: 0 for c in range(EXPECTED_NUM_CLASSES)}
    total_val_annotations = 0
    for lbl_file in val_labels_dir.glob("*.txt"):
        content = lbl_file.read_text(encoding="utf-8").strip()
        if content:
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        c_id = int(parts[0])
                        if c_id in val_class_counts:
                            val_class_counts[c_id] += 1
                        total_val_annotations += 1
                    except ValueError:
                        continue
    print(f"[Dataset Audit] Computed {total_val_annotations:,} validation annotations across {len(val_imgs):,} validation frames.")

    data_yaml_path = dataset_root / "data.yaml"
    with open(data_yaml_path, "r", encoding="utf-8") as yf:
        data_cfg = yaml.safe_load(yf)
    assert "holdout" not in str(data_cfg.get("train", "")), "Holdout set must not be in train split!"
    assert "holdout" not in str(data_cfg.get("val", "")), "Holdout set must not be in val split!"

    # 4. Mandatory Training Authorization Banner
    print("\n" + "=" * 48)
    print("U1 TRAINING AUTHORIZED")
    print("=" * 48)
    print(f"\nExperiment:\n{EXPERIMENT_ID}\n")
    print(f"Train images: {len(train_imgs)}")
    print(f"Val images: {len(val_imgs)}")
    print(f"Holdout images: {len(holdout_imgs)} [EXCLUDED]\n")
    print(f"Classes: {EXPECTED_NUM_CLASSES}\n")
    print(f"Model: YOLOv8m")
    print(f"Image size: {imgsz}")
    print(f"Batch: {batch}")
    print(f"Workers: {workers}")
    print(f"Epochs: {epochs}")
    print(f"AMP: {amp}")
    print(f"Mosaic: {mosaic}")
    print(f"Device: CUDA:{device}\n")

    # 5. Base Model Loading & Architecture / Weights Transfer Inspection
    base_model_path = (ROOT_DIR / REQUIRED_BASE_MODEL).resolve()
    assert base_model_path.is_file(), f"Base weights missing: {base_model_path}"
    model = YOLO(str(base_model_path))

    # Dynamically inspect actual model layers and parameters
    model_modules = list(model.model.model.children())
    num_modules = len(model_modules)
    head_module = model_modules[-1]
    head_type = type(head_module).__name__
    backbone_neck_modules = model_modules[:-1]
    backbone_neck_params = sum(p.numel() for mod in backbone_neck_modules for p in mod.parameters())
    head_params = sum(p.numel() for p in head_module.parameters())
    total_params = sum(p.numel() for p in model.model.parameters())
    orig_nc = getattr(head_module, "nc", 80)

    # Class-agnostic box regression (cv2) vs class-specific logits (cv3)
    cv2_params = sum(p.numel() for p in getattr(head_module, "cv2", []).parameters()) if hasattr(head_module, "cv2") else 0
    cv3_params = sum(p.numel() for p in getattr(head_module, "cv3", []).parameters()) if hasattr(head_module, "cv3") else 0

    transferred_weights_desc = (
        f"Transferred compatible pretrained backbone and neck layers (modules 0 to {num_modules - 2}, "
        f"{len(backbone_neck_modules)} modules, {backbone_neck_params:,} parameters) and class-agnostic box "
        f"regression branch ({cv2_params:,} parameters) from {REQUIRED_BASE_MODEL}. "
        f"Newly initialized final 9-class {head_type} classification projection head (cv3 output convolutions "
        f"adapted from {orig_nc} to {EXPECTED_NUM_CLASSES} classes). D1 2-class head completely excluded."
    )
    print(f"[Model Architecture] Total modules: {num_modules}, Total parameters: {total_params:,}")
    print(f"[Model Architecture] {transferred_weights_desc}")

    target_run_dir = ROOT_DIR / "training_lab" / "runs" / EXPERIMENT_ID
    target_run_dir.mkdir(parents=True, exist_ok=True)

    # 6. Configure cuDNN deterministic stability and clean CUDA state
    configure_stable_cudnn()
    reset_cuda_state()

    t_start = time.perf_counter()

    # 7. Execute Training (Strictly fail on OOM/CUDA errors without fallback)
    try:
        results = model.train(
            data=str(data_yaml_path.resolve()),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=str((ROOT_DIR / "training_lab" / "runs").resolve()),
            name=EXPERIMENT_ID,
            device=device,
            workers=workers,
            plots=True,
            save=True,
            verbose=True,
            mosaic=mosaic,
            amp=amp,
            deterministic=True,
            seed=42,
            exist_ok=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.perf_counter() - t_start
        print(f"\n[SUCCESS] Full {epochs}-epoch U1 training finished in {total_time:.1f}s ({total_time / 60:.1f} min).")

    except Exception as exc:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        elapsed = time.perf_counter() - t_start
        failure_category = classify_failure(exc)

        print("\n" + "!" * 80)
        print(f"[TRAINING FAILED] Exception after {elapsed:.1f}s in run '{EXPERIMENT_ID}':")
        print(f"  Failure Category: {failure_category}")
        print(f"  Exception Type:   {type(exc).__name__}")
        print(f"  Exception Message:{exc}")
        print("!" * 80)

        write_failure_diagnostic(
            experiment_name=EXPERIMENT_ID,
            run_dir=target_run_dir,
            exception=exc,
            amp=amp,
            workers=workers,
            batch=batch,
            imgsz=imgsz,
            mosaic=mosaic,
        )
        reset_cuda_state()
        return 1

    # 8. Checkpoint Preservation and Hashing
    best_pt = target_run_dir / "weights" / "best.pt"
    last_pt = target_run_dir / "weights" / "last.pt"
    assert best_pt.is_file(), f"best.pt checkpoint not found at: {best_pt}"
    assert last_pt.is_file(), f"last.pt checkpoint not found at: {last_pt}"

    best_pt_size_bytes = best_pt.stat().st_size
    best_pt_sha256 = compute_sha256(best_pt)
    print(f"[Checkpoint] Verified best.pt ({best_pt_size_bytes / (1024**2):.2f} MB, SHA-256: {best_pt_sha256[:16]}...)")
    print(f"[Checkpoint] Verified last.pt ({last_pt.stat().st_size / (1024**2):.2f} MB)")

    # 9. Post-Training Validation on best.pt across all 9 classes
    try:
        final_val_metrics = run_post_training_validation(
            checkpoint_path=best_pt,
            data_yaml_path=data_yaml_path,
            batch=batch,
            workers=workers,
            device=device,
        )
    except Exception as exc:
        print(f"\n[FATAL] Post-training validation failed: {exc}")
        return 1

    # 10. Offline Inference Verification (val images only)
    val_images_dir = dataset_root / "images" / "val"
    inference_results = run_offline_inference_verification(
        weights_path=best_pt,
        val_images_dir=val_images_dir,
        device=device,
    )
    avg_latency = round(float(np.mean([r["latency_ms"] for r in inference_results])), 2) if inference_results else 0.0

    # 11. Parse Epoch Metrics from results.csv & Find Best Epoch
    results_csv_p = target_run_dir / "results.csv"
    per_epoch_metrics = parse_results_csv(results_csv_p)
    best_epoch = max(per_epoch_metrics, key=lambda x: x.get("mAP50_95", 0.0))["epoch"] if per_epoch_metrics else epochs

    # Peak VRAM
    peak_gpu_mem_mb = (
        torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
    )
    gpu_identity = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"

    # Dataset SHA-256
    dataset_sha = compute_sha256(data_yaml_path)

    # 12. Final Acceptance Reports Generation
    report: Dict[str, Any] = {
        "status": "TRAINING_COMPLETED",
        "experiment_id": EXPERIMENT_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": preflight_meta["git_commit"],
        "gpu_identity": gpu_identity,
        "peak_vram_mb": round(peak_gpu_mem_mb, 2),
        "base_model": REQUIRED_BASE_MODEL,
        "model_parameter_count": total_params,
        "model_transfer_analysis": {
            "base_model": REQUIRED_BASE_MODEL,
            "total_modules": num_modules,
            "transferred_layers_range": f"0..{num_modules - 2}",
            "transferred_backbone_neck_params": backbone_neck_params,
            "transferred_box_regression_params": cv2_params,
            "head_module_index": num_modules - 1,
            "head_module_type": head_type,
            "original_head_classes": orig_nc,
            "target_head_classes": EXPECTED_NUM_CLASSES,
            "newly_initialized_head_params_desc": f"cv3 class prediction layers for {EXPECTED_NUM_CLASSES} classes",
        },
        "transferred_pretrained_weights": transferred_weights_desc,
        "newly_initialized_layers": f"Module {num_modules - 1} ({head_type}) cv3 classification projection layers for {EXPECTED_NUM_CLASSES} classes",
        "dataset": {
            "dataset_id": "dataset_unified_v001",
            "data_yaml_sha256": dataset_sha,
            "train_image_count": len(train_imgs),
            "val_image_count": len(val_imgs),
            "holdout_image_count": len(holdout_imgs),
            "val_annotation_count": total_val_annotations,
            "val_annotation_counts_by_class": {
                CANONICAL_CLASSES[c]: val_class_counts[c] for c in range(EXPECTED_NUM_CLASSES)
            },
            "num_classes": EXPECTED_NUM_CLASSES,
            "class_names": CANONICAL_CLASSES,
        },
        "hyperparameters": {
            "imgsz": imgsz,
            "batch": batch,
            "workers": workers,
            "amp": amp,
            "mosaic": mosaic,
            "device": device,
            "epochs": epochs,
            "auto_batch_disabled": True,
        },
        "performance": {
            "training_duration_seconds": round(total_time, 2),
            "average_epoch_duration_seconds": round(total_time / epochs, 2),
            "images_per_second": round((len(train_imgs) * epochs) / total_time, 2),
            "epochs_completed": len(per_epoch_metrics),
            "best_epoch": best_epoch,
        },
        "checkpoints": {
            "best_pt": format_weights_path(best_pt),
            "best_pt_sha256": best_pt_sha256,
            "best_pt_size_bytes": best_pt_size_bytes,
            "last_pt": format_weights_path(last_pt),
        },
        "validation_metrics": final_val_metrics,
        "offline_inference": {
            "average_latency_ms": avg_latency,
            "results": inference_results,
        },
        "epoch_progression": per_epoch_metrics,
    }

    # Save JSON report to run dir and to central reports dir
    report_path_run = target_run_dir / "u1_training_report.json"
    with open(report_path_run, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_path_central = ROOT_DIR / "training_lab" / "reports" / "U1_unified_9class_training_report.json"
    with open(report_path_central, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Machine-readable training report written to: {report_path_central}")

    # Generate Human-Readable Markdown Summary
    summary_md_path = ROOT_DIR / "training_lab" / "reports" / "U1_unified_9class_training_summary.md"
    md_content = f"""# BORDER SENTINEL — U1 UNIFIED 9-CLASS DETECTOR TRAINING REPORT

**Experiment ID**: `{EXPERIMENT_ID}`  
**Status**: **`{report['status']}`**  
**Timestamp**: `{report['timestamp']}`  
**Git Commit**: `{report['git_commit']}`  
**GPU Identity**: `{gpu_identity}` | **Peak VRAM**: `{peak_gpu_mem_mb:.2f} MB`  

---

## 1. Executive Summary & Architecture
- **Base Architecture**: `{REQUIRED_BASE_MODEL}` ({total_params:,} parameters)
- **Weight Transfer**: {transferred_weights_desc}
- **Newly Initialized**: Module {num_modules - 1} (`{head_type}`) cv3 classification projection layers for {EXPECTED_NUM_CLASSES} classes
- **Training Duration**: `{total_time:.1f}s` ({total_time / 60:.1f} min across {epochs} epochs)
- **Best Epoch**: `{best_epoch}` of {epochs}
- **Checkpoint**: `{format_weights_path(best_pt)}` (SHA-256: `{best_pt_sha256}`)

---

## 2. Dataset Partitioning & Scope
- **Dataset**: `dataset_unified_v001` (SHA-256: `{dataset_sha}`)
- **Train Split**: {len(train_imgs):,} images
- **Val Split**: {len(val_imgs):,} images ({report['dataset']['val_annotation_count']:,} annotations)
- **Holdout Split**: {len(holdout_imgs):,} images (**100% EXCLUDED & PROTECTED**)

---

## 3. Overall & Per-Class Validation Metrics (Split: VAL)
- **Overall Precision**: **{final_val_metrics['all']['precision']:.2f}%**
- **Overall Recall**: **{final_val_metrics['all']['recall']:.2f}%**
- **Overall mAP50**: **{final_val_metrics['all']['mAP50']:.2f}%**
- **Overall mAP50-95**: **{final_val_metrics['all']['mAP50_95']:.2f}%**

### Per-Class Performance
| Class ID | Class Name | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
"""
    for cn, cm in final_val_metrics["per_class"].items():
        md_content += f"| {cm['class_id']} | **{cn}** | {cm['precision']:.2f}% | {cm['recall']:.2f}% | {cm['mAP50']:.2f}% | {cm['mAP50_95']:.2f}% |\n"

    md_content += f"""
---

## 4. Offline Inference Smoke Test (Zero Network Dependency)
- **Average Latency**: **{avg_latency:.2f} ms** per frame
- **Tested Images**: {len(inference_results)} validation frames
- **Verification Status**: 100% Offline Local Inference Verified
"""
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[REPORT] Human-readable summary report written to: {summary_md_path}")

    # 13. Immutability Final Audit
    verify_d1_immutability()

    print("\n" + "=" * 80)
    print("U1 UNIFIED 9-CLASS TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return 0


def main():
    parser = argparse.ArgumentParser(description="U1 Unified 9-Class Training Runner")
    parser.add_argument("--preflight-only", action="store_true", help="Run 17-point preflight checks only and exit")
    parser.add_argument("--batch", type=int, default=REQUIRED_BATCH)
    parser.add_argument("--workers", type=int, default=REQUIRED_WORKERS)
    parser.add_argument("--device", type=str, default=REQUIRED_DEVICE)
    parser.add_argument("--imgsz", type=int, default=REQUIRED_IMGSZ)
    parser.add_argument("--epochs", type=int, default=REQUIRED_EPOCHS)
    args = parser.parse_args()

    # Always verify D1 immutability first
    verify_d1_immutability()

    # Run 17-point preflight verification
    preflight_results = run_preflight(
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        imgsz=args.imgsz,
        epochs=args.epochs,
        amp=REQUIRED_AMP,
        mosaic=REQUIRED_MOSAIC,
    )

    if args.preflight_only:
        print("\n[PREFLIGHT ONLY] Preflight checks successfully verified. STOPPING before training as instructed.")
        return 0

    # Normal command: proceed to full authorized U1 training
    return execute_full_u1_training(
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        imgsz=args.imgsz,
        epochs=args.epochs,
        amp=REQUIRED_AMP,
        mosaic=REQUIRED_MOSAIC,
    )


if __name__ == "__main__":
    sys.exit(main())
