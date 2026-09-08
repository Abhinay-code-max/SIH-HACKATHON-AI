"""
Test Suite for Persistent Tracking, Geofence Tripwires, and Evidence Collection.
Verifies:
1. ByteTrack Object Tracking with persistent Track IDs.
2. Intelligence Engine rule triggers (Geofences, Tripwires, Loitering).
3. Evidence Capture (Snapshot, crop, and dossier JSON).
4. Operator Action Workflow (ACKNOWLEDGE, DISMISS, ESCALATE).
"""

from pathlib import Path
import sys
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from ai.tracking.tracker import ObjectTracker
from ai.events.intelligence_engine import intelligence_engine
from backend.app.services.evidence_engine import evidence_engine
from backend.app.main import app


def run_tracking_intelligence_tests():
    print("=" * 70)
    print("TESTING PERSISTENT TRACKING, GEOFENCES & EVIDENCE ENGINE")
    print("=" * 70)

    # 1. Test ObjectTracker
    print("\n[Stage 1/4] Testing ObjectTracker on synthetic video frame...")
    tracker = ObjectTracker(model_name="yolov8l.pt")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_frame[:] = (100, 100, 100)

    tracks, annotated_frame = tracker.update(dummy_frame)
    print(f"  -> Tracker initialized on {tracker.device}. Memory cleared.")

    # 1b. Test ObjectTracker with precomputed detections (zero YOLO inference)
    from backend.app.models.contracts import RawDetection
    precomputed = [
        RawDetection(
            detection_id="det_001",
            class_name="person",
            confidence=0.92,
            bbox=[100.0, 100.0, 180.0, 260.0],
            camera_id="CAM_01",
        ),
        RawDetection(
            detection_id="det_002",
            class_name="car",
            confidence=0.88,
            bbox=[300.0, 200.0, 450.0, 320.0],
            camera_id="CAM_01",
        ),
    ]
    tracks_pre, annotated_pre = tracker.update(dummy_frame, detections=precomputed)
    assert len(tracks_pre) == 2, f"Expected 2 tracks, got {len(tracks_pre)}"
    assert precomputed[0].object_id is not None, "Expected object_id to be populated on RawDetection"
    print(f"  -> Precomputed detection tracking SUCCESS: {len(tracks_pre)} persistent tracks established.")

    # 2. Test Intelligence Engine with simulated intrusion
    print("\n[Stage 2/4] Testing Geofence & Tripwire rule triggers...")
    simulated_tracks = [
        {
            "track_id": 27,
            "class_name": "person",
            "confidence": 0.94,
            "bbox": [50.0, 80.0, 150.0, 220.0],
            "center": [100.0, 150.0],  # Inside doorway polygon [0.02..0.35, 0.05..0.55]
            "normalized_center": [0.156, 0.312],
            "dwell_seconds": 10.5,  # Exceeds 8.0s dwell threshold
            "trajectory": [(150.0, 200.0), (180.0, 200.0), (230.0, 200.0)],  # Crosses x=192 tripwire line
        }
    ]

    events = intelligence_engine.evaluate_tracks(
        camera_id="CAM_01",
        tracks=simulated_tracks,
        frame=dummy_frame,
    )
    assert len(events) > 0, "No intelligence events triggered!"
    first_evt = events[0]
    print(f"  -> SUCCESS: Triggered {len(events)} security events:")
    for ev in events:
        print(f"     * [{ev['event_id']}] {ev['event_type']} (Track #{ev['target']['track_id']}) - Severity: {ev['severity']}")

    # 3. Test Evidence Dossier verification
    print("\n[Stage 3/4] Verifying forensic evidence generation...")
    dossier = evidence_engine.get_dossier(first_evt["event_id"])
    assert dossier is not None, "Dossier was not created!"
    snapshot_p = ROOT_DIR / "data" / dossier["evidence_files"]["snapshot"].lstrip("/")
    crop_p = ROOT_DIR / "data" / dossier["evidence_files"]["crop"].lstrip("/")
    assert snapshot_p.is_file(), f"Snapshot missing: {snapshot_p}"
    assert crop_p.is_file(), f"Crop missing: {crop_p}"
    print(f"  -> SUCCESS: Snapshot ({snapshot_p.name}) and Crop ({crop_p.name}) preserved on local disk.")

    # 4. Test Operator Workflow via FastAPI
    print("\n[Stage 4/4] Testing Operator Dispatch API (Acknowledge / Escalate)...")
    client = TestClient(app)

    # Telemetry
    r_telem = client.get("/api/cameras/telemetry")
    assert r_telem.status_code == 200
    print(f"  -> GET /api/cameras/telemetry: OK ({len(r_telem.json())} camera workers reported)")

    # List Evidence
    r_ev = client.get("/api/evidence")
    assert r_ev.status_code == 200
    evidence_list = r_ev.json()
    assert len(evidence_list) > 0, "Evidence API returned 0 dossiers!"
    print(f"  -> GET /api/evidence: OK ({len(evidence_list)} dossiers active)")

    # Take Operator Action
    r_act = client.post(
        f"/api/evidence/{first_evt['event_id']}/action",
        json={"action": "ACKNOWLEDGE", "operator_name": "Commander-Alpha", "notes": "Perimeter guard dispatched."},
    )
    assert r_act.status_code == 200
    updated = r_act.json()
    assert updated["operator_status"] == "ACKNOWLEDGE"
    print(f"  -> POST /api/evidence/.../action: OK (Status changed to {updated['operator_status']})")

    print("\n" + "=" * 70)
    print("ALL 4 TRACKING, GEOFENCE & EVIDENCE TESTS PASSED SUCCESSFULLY")
    print("=" * 70)
    return True


def test_tracking_intelligence():
    """PyTest entrypoint for tracking, geofence rules, and evidence dossiers."""
    assert run_tracking_intelligence_tests() is True


def test_drone_specialist_scenario_injection():
    """Verify Scenario 7 drone intrusion produces immediate DEFCON 1 / CRITICAL alert."""
    from tools.demo_scenario_injector import run_scenario_7_drone_intrusion
    from ai.events.incident_manager import incident_manager

    incident_manager.incidents.clear()
    run_scenario_7_drone_intrusion()

    defcon = incident_manager.get_current_system_defcon()
    assert defcon["defcon"]["level"] == "DEFCON_1"
    assert defcon["defcon"]["status"] == "CRITICAL"
    assert defcon["threat_score"] >= 75


if __name__ == "__main__":
    try:
        run_tracking_intelligence_tests()
        print("\nStatus: ADVANCED INTELLIGENCE SUITE SUCCESSFUL")
    except Exception as e:
        print(f"\nStatus: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
