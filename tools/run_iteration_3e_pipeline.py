"""
Iteration 3E — Unified 9-Class Detector Controlled Training, Evaluation, and Regression Pipeline.
Executes:
1. Integrity & Preflight Hash Verifications
2. Step 8: Pre-Training Baseline Validation on init_yolov8m_9class.pt
3. Steps 9 & 10: 5-Epoch Controlled Training of unified_9class_yolov8m_full
4. Steps 11 & 12: Post-Training Validation on best.pt
5. Step 13: UA-DETRAC Protected Holdout Evaluation (11,557 frames)
6. Step 14: Unified Multi-Class Holdout Evaluation (550 frames)
7. Step 15: Steady-State RTX 4060 Latency & Throughput Benchmark
8. Step 16: Offline Inference Verification
9. Step 18: Source Checkpoint Immutability Verification
10. Saves JSON report to training_lab/reports/iteration_3e_final_report.json
"""

import os
import sys
import time
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.cuda_diagnostic import configure_stable_cudnn, reset_cuda_state

INIT_WEIGHTS = ROOT_DIR / "training_lab/runs/unified_9class_initialization/weights/init_yolov8m_9class.pt"
DATASET_YAML = ROOT_DIR / "training_lab/runs/controlled_9class_dataset_view/dataset.yaml"
UA_DETRAC_DATA_YAML = ROOT_DIR / "training_lab/datasets/dataset_ua_detrac_v001/data.yaml"
UNIFIED_DATA_YAML = ROOT_DIR / "training_lab/datasets/dataset_unified_v001/data.yaml"

RUNS_DIR = ROOT_DIR / "training_lab/runs"
PROJECT_NAME = "unified_9class_yolov8m_full"
OUTPUT_DIR = RUNS_DIR / PROJECT_NAME
REPORT_PATH = ROOT_DIR / "training_lab/reports/iteration_3e_final_report.json"

PROD_MODEL_PATH = ROOT_DIR / "models/registry/YOLO-L-v002/weights/best.pt"
D1_MODEL_PATH = ROOT_DIR / "training_lab/runs/D1_yolov8m_640_2class_full/weights/best.pt"
COCO_MODEL_PATH = ROOT_DIR / "yolov8m.pt"

EXPECTED_INIT_HASH = "d48c5bd5ce1802bca021a3dd43d1f504a0fe8bfea747cb58fd03740510c3fc57"
EXPECTED_PROD_HASH = "eba4580f75fd5ccd5bfb37f6289a903288f91f7eedda3024faa822f0db6fb943"
EXPECTED_D1_HASH = "005b706408d49f52ab0a26795f8ea8ae2c0e7dc725445a616ddbb56cd4f7de3b"
EXPECTED_COCO_HASH = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"

CLASSES_9 = {
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

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

def extract_val_metrics(val_res) -> dict:
    names = val_res.names
    ap_classes = getattr(val_res.box, "ap_class_index", None)
    all_map50 = float(val_res.results_dict.get("metrics/mAP50(B)", 0.0))
    all_map50_95 = float(val_res.results_dict.get("metrics/mAP50-95(B)", 0.0))
    all_prec = float(val_res.results_dict.get("metrics/precision(B)", 0.0))
    all_rec = float(val_res.results_dict.get("metrics/recall(B)", 0.0))

    per_class = {}
    if ap_classes is not None:
        for i, c in enumerate(ap_classes):
            c_name = names.get(c, str(c))
            cp, cr, cap50, cap = val_res.box.class_result(i)
            per_class[c_name] = {
                "class_id": int(c),
                "precision": round(float(cp) * 100, 2),
                "recall": round(float(cr) * 100, 2),
                "mAP50": round(float(cap50) * 100, 2),
                "mAP50_95": round(float(cap) * 100, 2),
            }

    return {
        "overall": {
            "precision": round(all_prec * 100, 2),
            "recall": round(all_rec * 100, 2),
            "mAP50": round(all_map50 * 100, 2),
            "mAP50_95": round(all_map50_95 * 100, 2),
        },
        "per_class": per_class,
    }

def main():
    print("=" * 80)
    print("ITERATION 3E — UNIFIED 9-CLASS DETECTOR CONTROLLED PIPELINE")
    print("=" * 80)
    start_total_time = time.time()

    # Step 0: Pre-execution Hash & Safety Verification
    print("\n[Step 0] Verifying Checkpoint Immutability & Inputs...")
    init_hash = sha256_file(INIT_WEIGHTS)
    print(f"  init_yolov8m_9class.pt: {init_hash}")
    assert init_hash == EXPECTED_INIT_HASH, f"Init weights hash mismatch! Expected {EXPECTED_INIT_HASH}, got {init_hash}"

    prod_hash = sha256_file(PROD_MODEL_PATH)
    print(f"  YOLO-L-v002:           {prod_hash}")
    assert prod_hash == EXPECTED_PROD_HASH, f"Production model hash mismatch!"

    d1_hash = sha256_file(D1_MODEL_PATH)
    print(f"  D1 best.pt:            {d1_hash}")
    assert d1_hash == EXPECTED_D1_HASH, f"D1 best.pt hash mismatch!"

    coco_hash = sha256_file(COCO_MODEL_PATH)
    print(f"  yolov8m.pt:            {coco_hash}")
    assert coco_hash == EXPECTED_COCO_HASH, f"COCO base hash mismatch!"
    print("  All source checkpoint hashes verified 100% IMMUTABLE.")

    # Configure CUDA & cuDNN stability
    configure_stable_cudnn(seed=42)
    reset_cuda_state()

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": PROJECT_NAME,
        "base_model": "init_yolov8m_9class.pt",
        "init_weights_sha256": init_hash,
        "hyperparameters": {
            "epochs": 5,
            "batch": 4,
            "imgsz": 640,
            "device": "0",
            "workers": 0,
            "amp": True,
            "seed": 42,
            "mosaic": 1.0,
            "close_mosaic": 1,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "lr0": 0.01,
            "lrf": 0.01,
        },
    }

    # Step 8: Pre-Training Baseline Validation
    print("\n" + "=" * 80)
    print("[Step 8] Running Pre-Training Baseline Validation on init_yolov8m_9class.pt...")
    print("=" * 80)
    t0_pre_val = time.time()
    init_model = YOLO(str(INIT_WEIGHTS))
    pre_val_res = init_model.val(
        data=str(DATASET_YAML.resolve()),
        split="val",
        batch=4,
        workers=0,
        device="0",
        verbose=True,
    )
    pre_val_metrics = extract_val_metrics(pre_val_res)
    pre_val_duration = round(time.time() - t0_pre_val, 2)
    report_data["pre_training_baseline"] = {
        "duration_seconds": pre_val_duration,
        "metrics": pre_val_metrics,
    }
    print(f"Pre-Training Validation completed in {pre_val_duration}s:")
    print(f"  mAP50:    {pre_val_metrics['overall']['mAP50']}%")
    print(f"  mAP50-95: {pre_val_metrics['overall']['mAP50_95']}%")
    for cn, cd in pre_val_metrics["per_class"].items():
        print(f"    {cn:10s}: mAP50={cd['mAP50']:5.2f}%, mAP50-95={cd['mAP50_95']:5.2f}%")
    del init_model
    reset_cuda_state()

    # Steps 9 & 10: Execute Controlled 5-Epoch Training
    print("\n" + "=" * 80)
    print("[Steps 9 & 10] Starting Controlled 5-Epoch Training of unified_9class_yolov8m_full...")
    print("=" * 80)
    model = YOLO(str(INIT_WEIGHTS))
    t0_train = time.time()
    model.train(
        data=str(DATASET_YAML.resolve()),
        epochs=5,
        batch=4,
        imgsz=640,
        project=str(RUNS_DIR.resolve()),
        name=PROJECT_NAME,
        device="0",
        workers=0,
        amp=True,
        seed=42,
        mosaic=1.0,
        close_mosaic=1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        lr0=0.01,
        lrf=0.01,
        deterministic=True,
        exist_ok=True,
        verbose=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_duration = round(time.time() - t0_train, 2)
    print(f"\nTraining completed successfully in {training_duration}s ({training_duration/60:.2f} mins).")

    best_weights_path = OUTPUT_DIR / "weights/best.pt"
    last_weights_path = OUTPUT_DIR / "weights/last.pt"
    assert best_weights_path.is_file(), f"best.pt not found at {best_weights_path}!"
    assert last_weights_path.is_file(), f"last.pt not found at {last_weights_path}!"

    best_weights_hash = sha256_file(best_weights_path)
    best_weights_size = best_weights_path.stat().st_size
    print(f"Trained best.pt: {best_weights_path} ({best_weights_size / (1024**2):.2f} MB)")
    print(f"SHA-256: {best_weights_hash}")

    # Parse results.csv
    csv_path = OUTPUT_DIR / "results.csv"
    epoch_progression = []
    if csv_path.is_file():
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > 1:
            header = [h.strip() for h in lines[0].split(",")]
            for l in lines[1:]:
                parts = [p.strip() for p in l.split(",")]
                if len(parts) == len(header):
                    row = dict(zip(header, parts))
                    epoch_progression.append({
                        "epoch": int(row.get("epoch", 0)),
                        "train_box_loss": round(float(row.get("train/box_loss", 0)), 5),
                        "train_cls_loss": round(float(row.get("train/cls_loss", 0)), 5),
                        "train_dfl_loss": round(float(row.get("train/dfl_loss", 0)), 5),
                        "val_box_loss": round(float(row.get("val/box_loss", 0)), 5),
                        "val_cls_loss": round(float(row.get("val/cls_loss", 0)), 5),
                        "val_dfl_loss": round(float(row.get("val/dfl_loss", 0)), 5),
                        "precision": round(float(row.get("metrics/precision(B)", 0)) * 100, 2),
                        "recall": round(float(row.get("metrics/recall(B)", 0)) * 100, 2),
                        "mAP50": round(float(row.get("metrics/mAP50(B)", 0)) * 100, 2),
                        "mAP50_95": round(float(row.get("metrics/mAP50-95(B)", 0)) * 100, 2),
                    })

    report_data["training"] = {
        "duration_seconds": training_duration,
        "best_weights": str(best_weights_path),
        "best_weights_sha256": best_weights_hash,
        "best_weights_size_bytes": best_weights_size,
        "epoch_progression": epoch_progression,
    }
    del model
    reset_cuda_state()

    # Steps 11 & 12: Post-Training Validation on best.pt
    print("\n" + "=" * 80)
    print("[Steps 11 & 12] Running Post-Training Validation on best.pt...")
    print("=" * 80)
    t0_post_val = time.time()
    trained_model = YOLO(str(best_weights_path))
    post_val_res = trained_model.val(
        data=str(DATASET_YAML.resolve()),
        split="val",
        batch=4,
        workers=0,
        device="0",
        verbose=True,
    )
    post_val_metrics = extract_val_metrics(post_val_res)
    post_val_duration = round(time.time() - t0_post_val, 2)
    report_data["post_training_val"] = {
        "duration_seconds": post_val_duration,
        "metrics": post_val_metrics,
    }
    print(f"Post-Training Validation completed in {post_val_duration}s:")
    print(f"  mAP50:    {post_val_metrics['overall']['mAP50']}% (delta: {post_val_metrics['overall']['mAP50'] - pre_val_metrics['overall']['mAP50']:+.2f}%)")
    print(f"  mAP50-95: {post_val_metrics['overall']['mAP50_95']}% (delta: {post_val_metrics['overall']['mAP50_95'] - pre_val_metrics['overall']['mAP50_95']:+.2f}%)")
    for cn, cd in post_val_metrics["per_class"].items():
        pre_m = pre_val_metrics["per_class"].get(cn, {"mAP50": 0.0, "mAP50_95": 0.0})
        delta_50 = cd["mAP50"] - pre_m["mAP50"]
        print(f"    {cn:10s}: mAP50={cd['mAP50']:5.2f}% ({delta_50:+5.2f}%), mAP50-95={cd['mAP50_95']:5.2f}%")

    reset_cuda_state()

    # Step 13: Holdout Evaluation 1: UA-DETRAC Protected Holdout (11,557 frames)
    print("\n" + "=" * 80)
    print("[Step 13] Running Holdout Evaluation 1: UA-DETRAC Protected Holdout (11,557 frames)...")
    print("=" * 80)
    t0_ua_holdout = time.time()
    ua_holdout_res = trained_model.val(
        data=str(UA_DETRAC_DATA_YAML.resolve()),
        split="test",
        batch=4,
        workers=0,
        device="0",
        verbose=True,
    )
    ua_holdout_metrics = extract_val_metrics(ua_holdout_res)
    ua_holdout_duration = round(time.time() - t0_ua_holdout, 2)
    report_data["ua_detrac_holdout"] = {
        "duration_seconds": ua_holdout_duration,
        "metrics": ua_holdout_metrics,
    }
    print(f"UA-DETRAC Holdout completed in {ua_holdout_duration}s:")
    print(f"  Overall mAP50: {ua_holdout_metrics['overall']['mAP50']}%")
    for cn, cd in ua_holdout_metrics["per_class"].items():
        print(f"    {cn:10s}: mAP50={cd['mAP50']:5.2f}%, mAP50-95={cd['mAP50_95']:5.2f}%")

    reset_cuda_state()

    # Step 14: Holdout Evaluation 2: Unified Multi-Class Holdout (550 frames)
    print("\n" + "=" * 80)
    print("[Step 14] Running Holdout Evaluation 2: Unified Multi-Class Holdout (550 frames)...")
    print("=" * 80)
    t0_uni_holdout = time.time()
    uni_holdout_res = trained_model.val(
        data=str(UNIFIED_DATA_YAML.resolve()),
        split="test",
        batch=4,
        workers=0,
        device="0",
        verbose=True,
    )
    uni_holdout_metrics = extract_val_metrics(uni_holdout_res)
    uni_holdout_duration = round(time.time() - t0_uni_holdout, 2)
    report_data["unified_multi_class_holdout"] = {
        "duration_seconds": uni_holdout_duration,
        "metrics": uni_holdout_metrics,
    }
    print(f"Unified Holdout completed in {uni_holdout_duration}s:")
    print(f"  Overall mAP50: {uni_holdout_metrics['overall']['mAP50']}%")
    for cn, cd in uni_holdout_metrics["per_class"].items():
        print(f"    {cn:10s}: mAP50={cd['mAP50']:5.2f}%, mAP50-95={cd['mAP50_95']:5.2f}%")

    reset_cuda_state()

    # Step 15: Steady-State RTX 4060 Latency & Throughput Benchmark
    print("\n" + "=" * 80)
    print("[Step 15] Running Steady-State RTX 4060 Latency Benchmark...")
    print("=" * 80)
    dummy_input = torch.zeros((1, 3, 640, 640), device="cuda:0")

    # Cold start timing
    torch.cuda.synchronize()
    t0_cold = time.perf_counter()
    _ = trained_model.predict(dummy_input, device="0", verbose=False)
    torch.cuda.synchronize()
    cold_start_ms = round((time.perf_counter() - t0_cold) * 1000, 2)

    # Warmup 10 runs
    for _ in range(10):
        _ = trained_model.predict(dummy_input, device="0", verbose=False)
    torch.cuda.synchronize()

    # Timed 50 runs
    latencies = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = trained_model.predict(dummy_input, device="0", verbose=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_lat = round(float(np.mean(latencies)), 2)
    median_lat = round(float(np.median(latencies)), 2)
    p95_lat = round(float(np.percentile(latencies, 95)), 2)
    fps = round(1000.0 / mean_lat, 1) if mean_lat > 0 else 0.0

    report_data["latency_benchmark"] = {
        "device": torch.cuda.get_device_name(0),
        "cold_start_ms": cold_start_ms,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": median_lat,
        "p95_latency_ms": p95_lat,
        "throughput_fps": fps,
        "iterations": 50,
        "warmup_iterations": 10,
    }
    print(f"Latency Benchmark on {torch.cuda.get_device_name(0)}:")
    print(f"  Cold Start: {cold_start_ms} ms")
    print(f"  Mean:       {mean_lat} ms")
    print(f"  Median:     {median_lat} ms")
    print(f"  P95:        {p95_lat} ms")
    print(f"  FPS:        {fps} frames/sec")

    # Step 16: Offline Inference Verification on Real Frames
    print("\n" + "=" * 80)
    print("[Step 16] Running Offline Inference Verification on Real Validation Frames...")
    print("=" * 80)
    val_images_dir = ROOT_DIR / "training_lab/runs/controlled_9class_dataset_view/images/val"
    sample_imgs = sorted(list(val_images_dir.glob("*.jpg")))[:5]
    offline_results = []
    for s_img in sample_imgs:
        res = trained_model.predict(str(s_img), device="0", conf=0.25, verbose=False)[0]
        boxes = res.boxes
        detected_cids = [int(c) for c in boxes.cls.tolist()] if len(boxes) > 0 else []
        cnames = [res.names[c] for c in detected_cids]
        confs = [round(float(c), 3) for c in boxes.conf.tolist()] if len(boxes) > 0 else []
        rec = {
            "image": s_img.name,
            "detections": len(boxes),
            "classes": detected_cids,
            "class_names": cnames,
            "confidences": confs,
        }
        offline_results.append(rec)
        print(f"  {s_img.name}: {len(boxes)} detections -> {cnames[:5]}")
    report_data["offline_inference_verification"] = offline_results
    del trained_model
    reset_cuda_state()

    # Step 18: Final Immutability Verification
    print("\n" + "=" * 80)
    print("[Step 18] Final Source Checkpoint Immutability Audit...")
    print("=" * 80)
    final_prod_hash = sha256_file(PROD_MODEL_PATH)
    final_d1_hash = sha256_file(D1_MODEL_PATH)
    final_coco_hash = sha256_file(COCO_MODEL_PATH)

    assert final_prod_hash == EXPECTED_PROD_HASH, "CRITICAL: Production model hash modified!"
    assert final_d1_hash == EXPECTED_D1_HASH, "CRITICAL: D1 best.pt hash modified!"
    assert final_coco_hash == EXPECTED_COCO_HASH, "CRITICAL: COCO base hash modified!"
    report_data["immutability_verification"] = {
        "status": "PASS",
        "production_model_sha256": final_prod_hash,
        "d1_best_sha256": final_d1_hash,
        "coco_base_sha256": final_coco_hash,
    }
    print("  Production model, D1 baseline, and COCO weights: 100% UNTOUCHED.")

    total_pipeline_time = round(time.time() - start_total_time, 2)
    report_data["total_pipeline_duration_seconds"] = total_pipeline_time
    print(f"\nPipeline finished in {total_pipeline_time:.2f}s ({total_pipeline_time/60:.2f} mins).")

    # Save JSON Report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Wrote Iteration 3E JSON report to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
