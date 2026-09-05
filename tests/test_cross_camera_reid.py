"""
Automated Test Suite for Phase v0.8: Cross-Camera Re-ID & Journey Tracking.
Validates:
1. Local deep feature extraction & spatial HSV histogram.
2. Spatio-temporal transit corridor validation & teleportation anomaly rejection.
3. GlobalSubjectManager lifecycle (registration, cross-camera transit handoff, EMA update).
4. Decoupled REST API endpoints (listing, dossier, transits, query search, reset).
"""

import io
from pathlib import Path
import sys
import time
import cv2
from fastapi.testclient import TestClient
from fastapi.testclient import TestClient
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.reid.extractor import feature_extractor
from ai.reid.association import cross_camera_associator
from ai.reid.manager import global_subject_manager
from backend.app.main import app


def reset_reid():
    """Resets Re-ID memory between tests."""
    global_subject_manager.subjects.clear()
    global_subject_manager.active_track_map.clear()
    global_subject_manager.last_embed_time.clear()
    global_subject_manager.transit_log.clear()
    global_subject_manager.next_subject_idx = 1
    global_subject_manager.transit_log.clear()


def test_feature_extractor():
    # Synthetic target crop (blue jacket, dark trousers)
    crop = np.zeros((160, 80, 3), dtype=np.uint8)
    crop[:90, :, :] = (180, 50, 20)  # blue upper body (BGR)
    crop[90:, :, :] = (30, 30, 30)   # dark lower body

    sig = feature_extractor.extract_signature(crop)
    assert sig.shape == (656,), f"Expected 656-dim feature signature, got {sig.shape}"
    norm = np.linalg.norm(sig)
    assert np.isclose(norm, 1.0, atol=1e-3), f"Signature must be unit-normalized, got {norm}"

    # Self-similarity should be ~1.0
    self_sim = feature_extractor.compute_similarity(sig, sig)
    assert np.isclose(self_sim, 1.0, atol=1e-3), f"Self-similarity should be 1.0, got {self_sim}"

    # Empty crop test
    empty_sig = feature_extractor.extract_signature(np.zeros((0, 0, 3), dtype=np.uint8))
    assert empty_sig.shape == (656,)
    assert np.all(empty_sig == 0.0)


def test_spatiotemporal_association_corridors():
    # 1. Valid transit corridor: CAM_01 -> CAM_02 in 10 seconds
    valid, verdict = cross_camera_associator.validate_transit("CAM_01", "CAM_02", 10.0)
    assert valid is True, f"Expected valid transit, got: {verdict}"
    assert "VALID_CORRIDOR" in verdict

    # 2. Teleportation anomaly: CAM_01 -> CAM_02 in 0.2 seconds (physically impossible)
    invalid, verdict = cross_camera_associator.validate_transit("CAM_01", "CAM_02", 0.2)
    assert invalid is False, "Expected teleportation anomaly rejection"
    assert "TELEPORTATION_ANOMALY" in verdict

    # 3. Expired transit window: CAM_01 -> CAM_02 after 500 seconds
    expired, verdict = cross_camera_associator.validate_transit("CAM_01", "CAM_02", 500.0)
    assert expired is False, "Expected expired transit rejection"
    assert "TRANSIT_EXPIRED" in verdict


def test_global_subject_lifecycle_and_transit():
    # Create synthetic frame with person
    frame_cam1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_cam1[100:300, 200:300] = (200, 100, 50)  # orange jacket

    bbox1 = [200.0, 100.0, 300.0, 300.0]

    # Step 1: Track appears on CAM_01
    subj1 = global_subject_manager.process_track(
        camera_id="CAM_01",
        local_track_id=101,
        class_name="person",
        bbox=bbox1,
        frame=frame_cam1,
        dwell_seconds=5.0,
        confidence=0.92,
    )

    assert subj1["subject_id"] == "SUBJ_0001"
    assert subj1["display_name"] == "[GLOBAL #01: PERSON]"
    assert subj1["last_camera_id"] == "CAM_01"
    assert len(global_subject_manager.subjects) == 1

    # Step 2: Same subject appears on CAM_02 after valid transit time (e.g. 8 seconds)
    # Simulate same person in CAM_02 frame
    frame_cam2 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_cam2[120:320, 250:350] = (200, 100, 50)  # same orange jacket
    bbox2 = [250.0, 120.0, 350.0, 320.0]

    # Advance time artificially on subject 1 to simulate travel time
    subj1["last_seen"] = time.time() - 8.0

    subj2 = global_subject_manager.process_track(
        camera_id="CAM_02",
        local_track_id=205,
        class_name="person",
        bbox=bbox2,
        frame=frame_cam2,
        dwell_seconds=1.0,
        confidence=0.90,
    )

    # Must be matched as the SAME Global Subject!
    assert subj2["subject_id"] == "SUBJ_0001", "Expected Re-ID to associate track 205 on CAM_02 with SUBJ_0001"
    assert subj2["last_camera_id"] == "CAM_02"
    assert len(global_subject_manager.subjects) == 1

    # Verify inter-camera transit handoff recorded
    transits = global_subject_manager.get_recent_transits()
    assert len(transits) >= 1, "Expected at least 1 inter-camera transit recorded"
    t = transits[0]
    assert t["subject_id"] == "SUBJ_0001"
    assert t["from_camera"] == "CAM_01"
    assert t["to_camera"] == "CAM_02"
    assert t["transit_duration_sec"] >= 7.0


def test_reid_api_endpoints():
    client = TestClient(app)

    # 1. Reset state
    res_reset = client.post("/api/reid/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "REID_STATE_RESET_SUCCESSFUL"

    # 2. Initially empty
    res_list = client.get("/api/reid/subjects")
    assert res_list.status_code == 200
    assert res_list.json() == []

    # 3. Simulate detection to register subject
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    subj = global_subject_manager.process_track(
        camera_id="CAM_01",
        local_track_id=7,
        class_name="car",
        bbox=[50.0, 50.0, 200.0, 150.0],
        frame=frame,
    )

    # 4. GET /api/reid/subjects
    res_list = client.get("/api/reid/subjects")
    assert res_list.status_code == 200
    subjects = res_list.json()
    assert len(subjects) == 1
    assert subjects[0]["subject_id"] == "SUBJ_0001"
    assert subjects[0]["class_name"] == "car"

    # 5. GET /api/reid/subjects/{id}
    res_dossier = client.get(f"/api/reid/subjects/{subj['subject_id']}")
    assert res_dossier.status_code == 200
    dossier = res_dossier.json()
    assert dossier["subject_id"] == "SUBJ_0001"
    assert len(dossier["sightings"]) >= 1

    # 6. GET 404 for invalid subject
    res_404 = client.get("/api/reid/subjects/SUBJ_9999")
    assert res_404.status_code == 404

    # 7. POST /api/reid/search with query image
    query_crop = np.full((100, 80, 3), 120, dtype=np.uint8)
    _, encoded_jpg = cv2.imencode(".jpg", query_crop)
    file_bytes = io.BytesIO(encoded_jpg.tobytes())

    res_search = client.post(
        "/api/reid/search",
        files={"file": ("query.jpg", file_bytes, "image/jpeg")},
    )
    assert res_search.status_code == 200
    search_data = res_search.json()
    assert search_data["total_matches"] >= 1
    top_match = search_data["results"][0]
    assert top_match["subject_id"] == "SUBJ_0001"
    assert top_match["similarity_score"] > 0.50


if __name__ == "__main__":
    print("\n=======================================================")
    print("RUNNING PHASE v0.8 CROSS-CAMERA RE-ID TEST SUITE")
    print("=======================================================")
    
    reset_reid()
    print("[1/4] Testing VisualFeatureExtractor (656-dim deep + color)...")
    test_feature_extractor()
    print("      --> PASS: 656-dim unit-normalized vector & self-similarity verified.")

    reset_reid()
    print("[2/4] Testing Spatio-Temporal Transit Corridors & Teleportation Rejection...")
    test_spatiotemporal_association_corridors()
    print("      --> PASS: Valid transit accepted, teleportation anomaly correctly rejected.")

    reset_reid()
    print("[3/4] Testing GlobalSubjectManager Cross-Camera Handoff (CAM_01 -> CAM_02)...")
    test_global_subject_lifecycle_and_transit()
    print("      --> PASS: Track handoff associated as SUBJ_0001; transit event logged.")

    reset_reid()
    print("[4/4] Testing Decoupled Re-ID REST API Endpoints...")
    test_reid_api_endpoints()
    print("      --> PASS: /api/reid/subjects, /api/reid/subjects/{id}, /api/reid/search verified.")

    print("\nSTATUS: ALL CROSS-CAMERA RE-ID TESTS PASSED SUCCESSFULLY! [4/4]")

