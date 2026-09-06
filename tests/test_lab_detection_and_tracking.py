"""
Automated 5-Stage Validation Suite for Lab Perception Engine.
Tests:
Stage 1: Model Initialization & GPU Acceleration (CUDA / RTX 4060).
Stage 2: 5-Camera Bundle Detection (detect_bundle across all 5 feeds with strict attribution & clamped bboxes).
Stage 3: Tracking ID Persistence (15 consecutive frames of CAM_01 maintaining persistent Track ID).
Stage 4: Multi-Camera Tracker Isolation (Dedicated per-camera ObjectTracker instances, zero track leakage).
Stage 5: Trajectory & Velocity Vector Validation (Accumulated points, positive displacement, dwell time, speed).
"""

from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.lab_detector import LabDetector
from training_lab.engine.lab_tracker import LabMultiCamTracker
from training_lab.engine.multi_cam_simulator import MultiCameraSimulator
from training_lab.engine.video_manager import ensure_demo_camera_videos


def test_stage_1_model_initialization_and_gpu():
    """Stage 1: Model Initialization & GPU Acceleration."""
    print("\n" + "=" * 80)
    print("[Stage 1/5] Testing Model Initialization & CUDA GPU Acceleration...")
    print("=" * 80)

    t0 = time.time()
    detector = LabDetector(model_name="auto")
    load_time = time.time() - t0

    assert torch.cuda.is_available(), "CUDA GPU is required but torch.cuda.is_available() is False"
    assert detector.device == "cuda", f"Expected device 'cuda', got '{detector.device}'"
    gpu_name = torch.cuda.get_device_name(0)
    assert "RTX 4060" in gpu_name or "NVIDIA" in gpu_name, f"Unexpected GPU: {gpu_name}"

    assert detector.model is not None, "YOLO model instance was not initialized"
    assert len(detector._class_map) > 0, "Model class map is empty"

    print(f"  [OK] Target Device: {detector.device} ({gpu_name})")
    print(f"  [OK] Resolved Weights: {detector.resolved_model_path}")
    print(f"  [OK] Model loaded in {load_time:.2f}s with {len(detector._class_map)} classes: {list(detector._class_map.values())[:9]}")
    print("  --> PASS: Stage 1 Model Initialization & GPU Acceleration Verified.")


def test_stage_2_five_camera_bundle_detection():
    """Stage 2: 5-Camera Bundle Detection."""
    print("\n" + "=" * 80)
    print("[Stage 2/5] Testing 5-Camera Bundle Detection with Strict Attribution...")
    print("=" * 80)

    ensure_demo_camera_videos()
    sim = MultiCameraSimulator(loop=True, fps=30.0)
    vids = {
        "CAM_01": ROOT_DIR / "training_lab" / "videos" / "cam_01_perimeter.mp4",
        "CAM_02": ROOT_DIR / "training_lab" / "videos" / "cam_02_roadway.mp4",
        "CAM_03": ROOT_DIR / "training_lab" / "videos" / "cam_03_restricted.mp4",
        "CAM_04": ROOT_DIR / "training_lab" / "videos" / "cam_04_terrain.mp4",
        "CAM_05": ROOT_DIR / "training_lab" / "videos" / "cam_05_secondary.mp4",
    }
    sim.bind_cameras(vids)

    bundle = sim.step()
    assert bundle is not None, "Simulator returned None on step"
    assert len(bundle["feeds"]) == 5, f"Expected 5 camera feeds, got {len(bundle['feeds'])}"

    detector = LabDetector(model_name="auto")
    bundle_dets = detector.detect_bundle(bundle, conf_threshold=0.35)

    expected_cams = ["CAM_01", "CAM_02", "CAM_03", "CAM_04", "CAM_05"]
    assert len(bundle_dets) == 5, f"Expected detections for 5 cameras, got {len(bundle_dets)}"

    total_detections = 0
    for cid in expected_cams:
        assert cid in bundle_dets, f"Missing detections for {cid}"
        dets = bundle_dets[cid]
        assert len(dets) > 0, f"Expected at least 1 detection for {cid}, got {len(dets)}"

        for d in dets:
            total_detections += 1
            assert d["camera_id"] == cid, f"Camera ID mismatch: {d['camera_id']} != {cid}"
            assert d["confidence"] >= 0.35, f"Confidence {d['confidence']} below threshold 0.35"
            assert "class_name" in d and len(d["class_name"]) > 0

            # Coordinate clamping validation [0, w] and [0, h]
            x1, y1, x2, y2 = d["bbox"]
            assert 0.0 <= x1 < x2 <= 640.0, f"{cid}: BBox x coords out of range [0, 640]: {d['bbox']}"
            assert 0.0 <= y1 < y2 <= 480.0, f"{cid}: BBox y coords out of range [0, 480]: {d['bbox']}"

            # Normalized coordinate validation [0.0, 1.0]
            nx1, ny1, nx2, ny2 = d["normalized_bbox"]
            assert 0.0 <= nx1 < nx2 <= 1.0, f"{cid}: Normalized bbox out of range: {d['normalized_bbox']}"
            assert 0.0 <= ny1 < ny2 <= 1.0, f"{cid}: Normalized bbox out of range: {d['normalized_bbox']}"

            print(f"  [OK] {cid}: [{d['detection_id']}] {d['class_name']} (conf={d['confidence']:.2f}) -> bbox={d['bbox']}")

    sim.close()
    assert total_detections >= 5, f"Expected at least 5 total detections across bundle, got {total_detections}"
    print("  --> PASS: Stage 2 5-Camera Bundle Detection Verified.")


def test_stage_3_tracking_id_persistence():
    """Stage 3: Tracking ID Persistence over 15 consecutive frames."""
    print("\n" + "=" * 80)
    print("[Stage 3/5] Testing Tracking ID Persistence over 15 Frames on CAM_01...")
    print("=" * 80)

    tracker = LabMultiCamTracker(model_name="auto")
    video_path = ROOT_DIR / "training_lab" / "videos" / "cam_01_perimeter.mp4"
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Failed to open {video_path}"

    persistent_id = None
    frames_tracked = 0

    for f_idx in range(15):
        ret, frame = cap.read()
        assert ret, f"Failed to read frame {f_idx}"

        tracks, ann_frame = tracker.update_frame(
            frame=frame,
            camera_id="CAM_01",
            conf_threshold=0.35,
            timestamp=round(f_idx / 30.0, 4),
        )

        assert len(tracks) > 0, f"No tracks detected at frame {f_idx}"
        track = tracks[0]

        if persistent_id is None:
            persistent_id = track["track_id"]
            print(f"  [Frame 00] Initialized track ID: {persistent_id} ({track['tracking_label']})")
        else:
            assert track["track_id"] == persistent_id, (
                f"Track ID reset at frame {f_idx}! Expected {persistent_id}, got {track['track_id']}"
            )

        assert track["camera_id"] == "CAM_01"
        assert track["class_name"] == "person"
        assert track["tracking_label"] == f"PERSON_{persistent_id:03d}"
        assert ann_frame.shape == (480, 640, 3)
        frames_tracked += 1

    cap.release()
    assert frames_tracked == 15, f"Expected 15 tracked frames, got {frames_tracked}"

    # Verify tracker internal state
    cam01_tracker = tracker.get_or_create_tracker("CAM_01")
    assert persistent_id in cam01_tracker.tracks, f"Track {persistent_id} not in CAM_01 tracker state"
    history_len = len(cam01_tracker.tracks[persistent_id]["history"])
    assert history_len == 15, f"Expected history length 15, got {history_len}"

    print(f"  -> Track ID {persistent_id} ('PERSON_{persistent_id:03d}') maintained continuously across {frames_tracked} frames without resetting.")
    print("  --> PASS: Stage 3 Tracking ID Persistence Verified.")


def test_stage_4_multicam_tracker_isolation():
    """Stage 4: Multi-Camera Tracker Isolation (CAM_01 vs CAM_02)."""
    print("\n" + "=" * 80)
    print("[Stage 4/5] Testing Multi-Camera Tracker State Isolation (CAM_01 vs CAM_02)...")
    print("=" * 80)

    tracker = LabMultiCamTracker(model_name="auto")
    cap1 = cv2.VideoCapture(str(ROOT_DIR / "training_lab" / "videos" / "cam_01_perimeter.mp4"))
    cap2 = cv2.VideoCapture(str(ROOT_DIR / "training_lab" / "videos" / "cam_02_roadway.mp4"))

    for step_i in range(5):
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        assert ret1 and ret2, "Failed to read frames from caps"

        t1, _ = tracker.update_frame(f1, "CAM_01", timestamp=round(step_i / 30.0, 4))
        t2, _ = tracker.update_frame(f2, "CAM_02", timestamp=round(step_i / 30.0, 4))

        for tr in t1:
            assert tr["camera_id"] == "CAM_01"
            assert "PERSON" in tr["tracking_label"]
        for tr in t2:
            assert tr["camera_id"] == "CAM_02"
            assert "CAR" in tr["tracking_label"] or "TRUCK" in tr["tracking_label"] or "VEHICLE" in tr["tracking_label"]

    cap1.release()
    cap2.release()

    # Verify dedicated separate ObjectTracker instances
    assert "CAM_01" in tracker.trackers
    assert "CAM_02" in tracker.trackers
    assert tracker.trackers["CAM_01"] is not tracker.trackers["CAM_02"], (
        "CRITICAL: CAM_01 and CAM_02 share the same ObjectTracker instance!"
    )

    cam1_tracks = tracker.trackers["CAM_01"].tracks
    cam2_tracks = tracker.trackers["CAM_02"].tracks

    for tid, info in cam1_tracks.items():
        assert info.get("class_name") == "person", f"CAM_01 contains non-person track: {info}"
    for tid, info in cam2_tracks.items():
        assert info.get("class_name") in ["car", "truck", "vehicle"], f"CAM_02 contains non-vehicle track: {info}"

    print("  [OK] Dedicated ObjectTracker instances confirmed for CAM_01 and CAM_02.")
    print("  [OK] Zero cross-camera track leakage verified across feeds.")
    print("  --> PASS: Stage 4 Multi-Camera Tracker Isolation Verified.")


def test_stage_5_trajectory_and_velocity_vectors():
    """Stage 5: Trajectory & Velocity Vector Validation."""
    print("\n" + "=" * 80)
    print("[Stage 5/5] Testing Trajectory Accumulation & Velocity Vectors...")
    print("=" * 80)

    tracker = LabMultiCamTracker(model_name="auto")
    cap = cv2.VideoCapture(str(ROOT_DIR / "training_lab" / "videos" / "cam_01_perimeter.mp4"))

    last_track = None
    for f_idx in range(15):
        ret, frame = cap.read()
        assert ret
        tracks, _ = tracker.update_frame(frame, "CAM_01", timestamp=round(f_idx / 30.0, 4))
        if tracks:
            last_track = tracks[0]
    cap.release()

    assert last_track is not None, "No track obtained"

    # Trajectory accumulation check
    traj = last_track["trajectory"]
    assert len(traj) == 15, f"Expected 15 trajectory points, got {len(traj)}"
    for pt in traj:
        assert len(pt) == 2, f"Invalid point: {pt}"

    # Displacement direction (CAM_01 pedestrian moves left to right, x increases)
    x_start = traj[0][0]
    x_end = traj[-1][0]
    displacement_x = x_end - x_start
    assert displacement_x > 15.0, f"Pedestrian should move forward, displacement was {displacement_x:.1f}px"

    # Velocity vector check
    vx, vy = last_track["velocity_vector"]
    assert vx > 0.0, f"Expected positive x velocity for forward motion, got {vx}"

    # Dwell time check
    dwell = last_track["dwell_seconds"]
    assert dwell >= 0.40, f"Expected ~0.47s dwell time, got {dwell}"

    # Speed check
    speed = last_track["speed_px_s"]
    assert speed > 0.0, f"Expected positive speed, got {speed}"

    print(f"  [OK] Trajectory Points: {len(traj)} points accumulated")
    print(f"  [OK] Net Displacement: dx=+{displacement_x:.1f}px (from x={x_start:.1f} to x={x_end:.1f})")
    print(f"  [OK] Velocity Vector: [{vx:.2f}, {vy:.2f}] px/step")
    print(f"  [OK] Calculated Speed: {speed:.1f} px/s")
    print(f"  [OK] Dwell Duration: {dwell:.2f}s (strictly non-negative)")
    print("  --> PASS: Stage 5 Trajectory & Velocity Vector Validation Verified.")


def run_all_stages():
    """Runs all 5 validation stages."""
    print("=" * 80)
    print("BORDER SENTINEL - LAB PERCEPTION ENGINE (YOLO & BYTETRACK) TEST SUITE")
    print("=" * 80)
    start_time = time.time()

    test_stage_1_model_initialization_and_gpu()
    test_stage_2_five_camera_bundle_detection()
    test_stage_3_tracking_id_persistence()
    test_stage_4_multicam_tracker_isolation()
    test_stage_5_trajectory_and_velocity_vectors()

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL 5 PERCEPTION STAGES PASSED SUCCESSFULLY in {total_time:.2f}s! [100% PASS RATE]")
    print("=" * 80)


if __name__ == "__main__":
    run_all_stages()
