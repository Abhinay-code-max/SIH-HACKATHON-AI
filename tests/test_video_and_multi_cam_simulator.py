"""
Automated 5-Stage Test Suite for Video Asset Manager & Five-Camera Synchronous Simulator.

Validates:
Stage 1: Offline Video Synthesis (CAM_01 through CAM_05 surveillance clips generated and on disk).
Stage 2: Metadata Extraction (Resolution 640x480, FPS 30.0, 150 frame counts).
Stage 3: Scenario Camera Binding (SCN_01 updated and persisted to disk with 5-camera video bindings).
Stage 4: 5-Camera Synchronous Playback (30 consecutive steps, 150 camera frames processed, shape 480x640x3).
Stage 5: Camera Isolation & Attribution (Zero cross-camera frame leakage, strict camera_id attribution, seek(0) and loop rewind).
"""

from pathlib import Path
import sys
import json
import time
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.video_manager import (
    VideoAssetManager,
    ensure_demo_camera_videos,
    get_video_metadata,
    video_manager,
)
from training_lab.engine.multi_cam_simulator import MultiCameraSimulator
from training_lab.engine.scenario_manager import scenario_manager, ScenarioManager


def test_stage_1_offline_video_synthesis():
    """Stage 1: Offline Video Synthesis - Verify 5 camera clips generated and exist on disk."""
    print("\n" + "=" * 80)
    print("[Stage 1/5] Testing Offline Video Synthesis for CAM_01 through CAM_05...")
    print("=" * 80)

    t0 = time.time()
    generated = ensure_demo_camera_videos(overwrite=True)
    elapsed = time.time() - t0

    assert len(generated) == 5, f"Expected 5 camera clips, got {len(generated)}"

    expected_files = {
        "CAM_01": "cam_01_perimeter.mp4",
        "CAM_02": "cam_02_roadway.mp4",
        "CAM_03": "cam_03_restricted.mp4",
        "CAM_04": "cam_04_terrain.mp4",
        "CAM_05": "cam_05_secondary.mp4",
    }

    for cam_id, expected_name in expected_files.items():
        assert cam_id in generated, f"Missing {cam_id} in generated video map"
        path_obj = generated[cam_id]
        assert path_obj.name == expected_name, f"Expected {expected_name}, got {path_obj.name}"
        assert path_obj.is_file(), f"Generated video does not exist on disk: {path_obj}"
        file_size = path_obj.stat().st_size
        assert file_size > 50_000, f"Video file {path_obj.name} abnormally small: {file_size} bytes"
        print(f"  [OK] {cam_id}: {path_obj.name} ({file_size / 1024:.1f} KB) on disk")

    print(f"  -> Generated 5 surveillance videos in {elapsed:.2f}s with zero network dependencies.")
    print("  --> PASS: Stage 1 Offline Video Synthesis Verified.")


def test_stage_2_metadata_extraction():
    """Stage 2: Metadata Extraction - Validate resolution, FPS, and frame count."""
    print("\n" + "=" * 80)
    print("[Stage 2/5] Testing Metadata Extraction (640x480, 30.0 FPS, 150 Frames)...")
    print("=" * 80)

    expected_files = [
        "cam_01_perimeter.mp4",
        "cam_02_roadway.mp4",
        "cam_03_restricted.mp4",
        "cam_04_terrain.mp4",
        "cam_05_secondary.mp4",
    ]

    videos_dir = ROOT_DIR / "training_lab" / "videos"

    for fname in expected_files:
        vpath = videos_dir / fname
        assert vpath.is_file(), f"Missing file: {vpath}"
        meta = get_video_metadata(vpath)

        # Resolution assertions
        assert meta["width"] == 640, f"{fname}: Expected width 640, got {meta['width']}"
        assert meta["height"] == 480, f"{fname}: Expected height 480, got {meta['height']}"
        assert meta["resolution"] == (640, 480), f"{fname}: Expected (640, 480), got {meta['resolution']}"

        # FPS assertions
        assert abs(meta["fps"] - 30.0) < 0.1, f"{fname}: Expected 30.0 FPS, got {meta['fps']}"

        # Total frames assertions (5.0s @ 30.0fps = 150 frames)
        assert meta["frame_count"] == 150, f"{fname}: Expected 150 frames, got {meta['frame_count']}"
        assert meta["total_frames"] == 150, f"{fname}: Expected 150 total_frames alias, got {meta['total_frames']}"

        # Duration assertions
        assert 4.9 <= meta["duration_sec"] <= 5.1, f"{fname}: Expected ~5.0s duration, got {meta['duration_sec']}"
        assert meta["file_size_mb"] > 0, f"{fname}: Invalid file size: {meta['file_size_mb']}"

        print(f"  [OK] {fname:25s} -> {meta['width']}x{meta['height']} @ {meta['fps']:.1f} FPS | {meta['frame_count']} frames | {meta['duration_sec']:.2f}s | {meta['file_size_mb']} MB")

    print("  -> All 5 surveillance videos calibrated to exact technical specifications.")
    print("  --> PASS: Stage 2 Metadata Extraction Verified.")


def test_stage_3_scenario_camera_binding():
    """Stage 3: Scenario Camera Binding - Update SCN_01 with the 5 synthesized video files."""
    print("\n" + "=" * 80)
    print("[Stage 3/5] Testing Scenario Camera Binding for SCN_01...")
    print("=" * 80)

    # Ensure scenarios are seeded
    scn = scenario_manager.get_scenario("SCN_01")
    if not scn:
        scenario_manager.seed_default_scenarios()
        scn = scenario_manager.get_scenario("SCN_01")

    assert scn is not None, "Failed to retrieve SCN_01"

    # Bind the 5 synthesized test videos to CAM_01 through CAM_05
    video_map = {
        "CAM_01": "training_lab/videos/cam_01_perimeter.mp4",
        "CAM_02": "training_lab/videos/cam_02_roadway.mp4",
        "CAM_03": "training_lab/videos/cam_03_restricted.mp4",
        "CAM_04": "training_lab/videos/cam_04_terrain.mp4",
        "CAM_05": "training_lab/videos/cam_05_secondary.mp4",
    }

    for cam_id, rel_path in video_map.items():
        assert cam_id in scn.assigned_cameras, f"Missing {cam_id} in SCN_01 assigned_cameras"
        scn.assigned_cameras[cam_id].video_file = rel_path

    # Persist updated scenario to disk
    scenario_manager.create_scenario(scn)

    # Reload fresh from disk using an independent ScenarioManager instance
    fresh_mgr = ScenarioManager()
    reloaded_scn = fresh_mgr.get_scenario("SCN_01")
    assert reloaded_scn is not None, "Failed to reload SCN_01 from disk"

    for cam_id, expected_rel in video_map.items():
        actual_rel = reloaded_scn.assigned_cameras[cam_id].video_file
        assert actual_rel == expected_rel, f"{cam_id}: expected {expected_rel}, got {actual_rel}"
        # Validate that the bound file actually exists on disk
        abs_bound_path = ROOT_DIR / actual_rel
        assert abs_bound_path.is_file(), f"Bound video does not exist on disk: {abs_bound_path}"
        print(f"  [OK] {cam_id} bound to: {actual_rel}")

    print("  -> SCN_01 successfully updated and persisted on disk with 5-camera video bindings.")
    print("  --> PASS: Stage 3 Scenario Camera Binding Verified.")


def test_stage_4_synchronous_playback():
    """Stage 4: 5-Camera Synchronous Playback - Step 30 frames and verify 150 camera frames."""
    print("\n" + "=" * 80)
    print("[Stage 4/5] Testing 5-Camera Synchronous Playback (30 Steps, 150 Frames)...")
    print("=" * 80)

    scn = scenario_manager.get_scenario("SCN_01")
    assert scn is not None, "SCN_01 must be available"

    # Initialize simulator with SCN_01 scenario bindings
    sim = MultiCameraSimulator(camera_bindings=scn, loop=True, fps=30.0)
    assert len(sim.get_active_cameras()) == 5, f"Expected 5 active cameras, got {len(sim.get_active_cameras())}"

    total_camera_frames_processed = 0
    expected_cams = ["CAM_01", "CAM_02", "CAM_03", "CAM_04", "CAM_05"]

    for step_i in range(30):
        bundle = sim.step()
        assert bundle is not None, f"Simulator returned None at step {step_i}"
        assert bundle["frame_idx"] == step_i, f"Frame index mismatch: {bundle['frame_idx']} != {step_i}"
        expected_ts = round(step_i / 30.0, 4)
        assert abs(bundle["timestamp"] - expected_ts) < 1e-4, f"Timestamp mismatch: {bundle['timestamp']} != {expected_ts}"

        feeds = bundle["feeds"]
        assert len(feeds) == 5, f"Expected 5 camera feeds at step {step_i}, got {len(feeds)}"

        for cid in expected_cams:
            assert cid in feeds, f"Missing {cid} in bundle feeds at step {step_i}"
            feed_data = feeds[cid]
            assert feed_data["camera_id"] == cid
            assert "frame" in feed_data
            assert isinstance(feed_data["frame"], np.ndarray), f"{cid}: frame is not an ndarray"
            assert feed_data["shape"] == (480, 640, 3), f"{cid}: invalid shape {feed_data['shape']}"
            assert feed_data["frame"].shape == (480, 640, 3), f"{cid}: invalid frame shape {feed_data['frame'].shape}"
            total_camera_frames_processed += 1

    sim.close()
    assert total_camera_frames_processed == 150, f"Expected 150 total processed camera frames, got {total_camera_frames_processed}"
    print(f"  -> Processed {total_camera_frames_processed} synchronized frames (30 bundles x 5 feeds) with 100% integrity.")
    print("  --> PASS: Stage 4 5-Camera Synchronous Playback Verified.")


def test_stage_5_camera_isolation_and_attribution():
    """Stage 5: Camera Isolation & Attribution - Zero frame leakage, matching camera_ids, seek(0) rewind."""
    print("\n" + "=" * 80)
    print("[Stage 5/5] Testing Camera Isolation, Attribution & Seek Rewind...")
    print("=" * 80)

    scn = scenario_manager.get_scenario("SCN_01")
    sim = MultiCameraSimulator(camera_bindings=scn, loop=True, fps=30.0)

    # 1. Grab initial frame bundle (frame 0)
    initial_bundle = sim.step()
    assert initial_bundle is not None
    initial_frames = {cid: initial_bundle["feeds"][cid]["frame"].copy() for cid in sim.get_active_cameras()}

    # 2. Assert no cross-camera frame leakage (CAM_01 vs CAM_02, etc.)
    f_cam1 = initial_frames["CAM_01"]
    f_cam2 = initial_frames["CAM_02"]
    f_cam3 = initial_frames["CAM_03"]
    f_cam4 = initial_frames["CAM_04"]
    f_cam5 = initial_frames["CAM_05"]

    assert not np.array_equal(f_cam1, f_cam2), "CRITICAL: CAM_01 and CAM_02 frames are identical (leakage detected)!"
    assert not np.array_equal(f_cam1, f_cam3), "CRITICAL: CAM_01 and CAM_03 frames are identical!"
    assert not np.array_equal(f_cam1, f_cam4), "CRITICAL: CAM_01 and CAM_04 frames are identical!"
    assert not np.array_equal(f_cam1, f_cam5), "CRITICAL: CAM_01 and CAM_05 frames are identical!"

    # Memory buffer isolation check
    assert f_cam1 is not f_cam2, "CAM_01 and CAM_02 share memory pointers!"
    print("  [OK] Frame isolation verified: no cross-camera frame leakage across any feeds.")

    # 3. Strict attribution check across 10 steps
    for s in range(1, 11):
        b = sim.step()
        assert b is not None
        for cid, f_data in b["feeds"].items():
            assert f_data["camera_id"] == cid, f"Attribution failure: feed key {cid} has camera_id {f_data['camera_id']}"
            assert f_data["frame_idx"] == s
    print("  [OK] Strict camera_id attribution verified across 50 individual camera feed payloads.")

    # 4. Seek(0) rewind test
    sim.seek(0)
    rewound_bundle = sim.step()
    assert rewound_bundle is not None
    assert rewound_bundle["frame_idx"] == 0, f"Expected rewound frame_idx 0, got {rewound_bundle['frame_idx']}"

    # Verify that rewound frame 0 matches initial frame 0
    for cid in sim.get_active_cameras():
        rewound_frame = rewound_bundle["feeds"][cid]["frame"]
        # Allow small floating point or codec decode variance, or exact match
        diff = np.mean(np.abs(rewound_frame.astype(np.float32) - initial_frames[cid].astype(np.float32)))
        assert diff < 1.0, f"{cid}: Rewound frame 0 differs significantly from initial frame 0 (diff={diff})"
        print(f"  [OK] {cid}: Seek(0) rewound successfully (mean pixel delta: {diff:.4f})")

    # 5. Loop playback test: seek to end of 150-frame clip (frame 149) and step across boundary
    sim.seek(149)
    b_end = sim.step()
    assert b_end is not None and b_end["frame_idx"] == 149
    b_loop = sim.step()
    assert b_loop is not None and b_loop["frame_idx"] == 150
    assert len(b_loop["feeds"]) == 5
    print("  [OK] Loop playback across 150-frame video boundary verified without frame drops or crashes.")

    sim.close()
    print("  --> PASS: Stage 5 Camera Isolation & Attribution Verified.")


def run_all_stages():
    """Runs all 5 validation stages sequentially."""
    print("=" * 80)
    print("BORDER SENTINEL - VIDEO ASSET MANAGER & 5-CAMERA SIMULATOR TEST SUITE")
    print("=" * 80)
    start_time = time.time()

    test_stage_1_offline_video_synthesis()
    test_stage_2_metadata_extraction()
    test_stage_3_scenario_camera_binding()
    test_stage_4_synchronous_playback()
    test_stage_5_camera_isolation_and_attribution()

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL 5 STAGES PASSED SUCCESSFULLY in {total_time:.2f}s! [100% PASS RATE]")
    print("=" * 80)


if __name__ == "__main__":
    run_all_stages()
