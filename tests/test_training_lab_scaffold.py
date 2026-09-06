"""
Test Suite for AI Training, Validation & Simulation Lab Workspace Scaffold.

Verifies:
1. Directory Tree & Storage Verification (all 10 training_lab subdirectories and write capability).
2. GPU & CUDA Hardware Acceleration (NVIDIA RTX 4060 GPU and VRAM allocation).
3. Offline YOLO Model Loading via Ultralytics (local weights resolution, classes dictionary).
4. Synthetic Video Frame Buffer & Multi-Camera Tagging Data Structure.
"""

import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.inference.loader import find_local_model, get_device, load_local_model


def test_stage_1_directory_tree():
    print("\n[Stage 1/4] Verifying Training Lab Directory Tree & Permissions...")
    lab_root = ROOT_DIR / "training_lab"
    assert lab_root.is_dir(), f"Lab root missing: {lab_root}"

    required_subdirs = [
        "scenarios",
        "videos",
        "datasets",
        "annotations",
        "models",
        "experiments",
        "results",
        "engine",
        "ui",
        "api",
    ]

    for subdir in required_subdirs:
        dir_path = lab_root / subdir
        assert dir_path.is_dir(), f"Required directory missing: {dir_path}"
        print(f"  -> Verified Directory: training_lab/{subdir}/")

    # Sanity file write check
    sanity_file = lab_root / "results" / "sanity_check.json"
    sanity_payload = {
        "lab_status": "READY",
        "timestamp": time.time(),
        "verified_directories": required_subdirs,
    }
    with open(sanity_file, "w", encoding="utf-8") as f:
        json.dump(sanity_payload, f, indent=2)

    assert sanity_file.is_file(), "Sanity check file write failed"
    print(f"  -> Sanity check file successfully written: {sanity_file.name}")
    print("  --> PASS: Stage 1 Directory Tree Verification.")


def test_stage_2_gpu_environment():
    print("\n[Stage 2/4] Verifying PyTorch CUDA & GPU Hardware Acceleration...")
    cuda_ready = torch.cuda.is_available()
    assert cuda_ready is True, "CUDA is not available in PyTorch environment!"

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)

    print(f"  -> CUDA Available: True | Device Count: {device_count}")
    print(f"  -> Active GPU Model: {device_name}")
    print(f"  -> Dedicated VRAM: {vram_gb:.2f} GB")
    assert "RTX 4060" in device_name or "GeForce" in device_name, f"Unexpected GPU device: {device_name}"

    # Allocate a test tensor on cuda:0
    test_tensor = torch.zeros((100, 100), device="cuda:0")
    assert test_tensor.is_cuda, "Tensor was not allocated on CUDA"
    print("  -> Successfully allocated test tensor on cuda:0.")
    print("  --> PASS: Stage 2 GPU & CUDA Environment Verified.")


def test_stage_3_offline_yolo_import():
    print("\n[Stage 3/4] Verifying Offline Ultralytics YOLO Loading...")
    from ultralytics import YOLO

    model_path = find_local_model("yolov8l.pt")
    assert model_path.is_file(), f"Local model weights not found at: {model_path}"

    model = load_local_model(str(model_path))
    assert model is not None, "Failed to load local model"
    assert hasattr(model, "names") and len(model.names) > 0, "Model has no classes dictionary"

    print(f"  -> Loaded Model: {model_path.name} ({len(model.names)} classes)")
    sample_classes = [model.names[i] for i in range(min(5, len(model.names)))]
    print(f"  -> Sample Classes: {', '.join(sample_classes)}")
    print("  --> PASS: Stage 3 Offline YOLO Model Import Verified.")


def test_stage_4_synthetic_frame_and_tagging():
    print("\n[Stage 4/4] Verifying Synthetic Frame Buffer & Multi-Camera Tagging...")
    # Generate a synthetic 640x480 3-channel frame
    h, w = 480, 640
    frame = np.full((h, w, 3), (25, 30, 45), dtype=np.uint8)

    # Add visual test patterns
    cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "TRAINING LAB SIMULATION BUFFER",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    assert frame.shape == (480, 640, 3), f"Unexpected frame shape: {frame.shape}"

    # Multi-camera tagging data structure
    camera_tags = [
        {"camera_id": f"CAM_{i:02d}", "frame_idx": 0, "resolution": f"{w}x{h}", "timestamp": time.time()}
        for i in range(1, 6)
    ]

    assert len(camera_tags) == 5, "Expected 5 simulated camera tags"
    assert camera_tags[0]["camera_id"] == "CAM_01"
    assert camera_tags[4]["camera_id"] == "CAM_05"

    print(f"  -> Generated synthetic test frame: {w}x{h}x3")
    print(f"  -> Validated 5-camera simulation tag structure: {[c['camera_id'] for c in camera_tags]}")
    print("  --> PASS: Stage 4 Frame Buffer & Multi-Camera Tagging Verified.")


def run_all_scaffold_tests():
    print("=" * 75)
    print("AI TRAINING, VALIDATION & SIMULATION LAB — WORKSPACE SCAFFOLD TEST SUITE")
    print("=" * 75)

    test_stage_1_directory_tree()
    test_stage_2_gpu_environment()
    test_stage_3_offline_yolo_import()
    test_stage_4_synthetic_frame_and_tagging()

    print("\n" + "=" * 75)
    print("ALL 4 TRAINING LAB SCAFFOLD VERIFICATION STAGES PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    try:
        run_all_scaffold_tests()
        print("\nStatus: TRAINING LAB SCAFFOLD TEST SUITE SUCCESSFUL")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nStatus: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
