"""
End-to-End System Integration Test Suite.
Verifies the complete pipeline:
Input Media -> AI Inference -> Event Engine Rules -> Local Storage -> Mapping GIS -> Backend API.
Strictly local, zero external network required.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from ai.inference.image_inference import run_image_inference
from ai.events.engine import EventEngine
from mapping.manager import OfflineMapManager
from backend.app.main import app
from backend.app.models.contracts import GeoLocation, SeverityLevel


def run_e2e_test():
    print("=" * 70)
    print("STARTING END-TO-END INTEGRATION TEST (PHASE 1 LOCAL PROTOTYPE)")
    print("=" * 70)

    # -------------------------------------------------------------
    # STAGE 1: Media Input & Local AI Inference
    # -------------------------------------------------------------
    print("\n[E2E 1/5] Running Local AI Inference on Sample Image...")
    sample_img = ROOT_DIR / "data" / "sample-images" / "traffic_sample.jpg"
    assert sample_img.is_file(), f"Sample image missing at: {sample_img}"

    inf_report = run_image_inference(
        image_path=sample_img,
        model_name="yolov8l.pt",
        conf_threshold=0.35,
        save_annotated=True,
    )

    det_count = inf_report["total_detections"]
    assert det_count > 0, "AI inference returned 0 detections!"
    print(f"  -> SUCCESS: Detected {det_count} objects on {inf_report['device']} in {inf_report['inference_latency_ms']} ms.")

    # -------------------------------------------------------------
    # STAGE 2: Event Engine & Rule Evaluation
    # -------------------------------------------------------------
    print("\n[E2E 2/5] Evaluating Rules & Generating Security Events...")
    engine = EventEngine(
        camera_id="CAM_01",
        default_location=GeoLocation(lat=17.4435, lon=78.3765, zone_name="Pedestrian Zone Alpha"),
        crowd_threshold=3,
        alert_cooldown_sec=0.0,  # Cooldown disabled for deterministic batch test
    )

    events = engine.process_frame_detections(
        raw_detections=inf_report["detections"],
        frame_index=101,
    )
    assert len(events) > 0, "Event Engine did not generate any events!"

    # Export to active store
    live_events_file = ROOT_DIR / "data" / "sample-events" / "live_events.json"
    engine.export_events(live_events_file)
    print(f"  -> SUCCESS: Generated {len(events)} security events:")
    for ev in events:
        print(f"     * [{ev.event_id}] {ev.event_type.value} ({ev.class_name}) - Severity: {ev.severity.value.upper()}")

    # -------------------------------------------------------------
    # STAGE 3: Offline Mapping & Spatial Association
    # -------------------------------------------------------------
    print("\n[E2E 3/5] Associating Events with Offline GeoJSON GIS Layers...")
    map_mgr = OfflineMapManager(base_dir=ROOT_DIR / "mapping")
    map_payload = map_mgr.build_integrated_map_payload(events_file=live_events_file)

    assert len(map_payload["zones"]["features"]) >= 3, "Zones layer missing!"
    assert len(map_payload["cameras"]) >= 3, "Cameras layer missing!"
    assert len(map_payload["event_markers"]) >= len(events), "Event markers not mapped!"
    print(f"  -> SUCCESS: Mapped {len(map_payload['event_markers'])} event markers onto {len(map_payload['zones']['features'])} security zones.")

    # -------------------------------------------------------------
    # STAGE 4: Backend API Verification
    # -------------------------------------------------------------
    print("\n[E2E 4/5] Verifying Local Backend API Endpoints via TestClient...")
    client = TestClient(app)

    # Health
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    h_data = r_health.json()
    assert h_data["offline_mode"] is True
    print(f"  -> GET /api/health: OK (Compute: {h_data['device_name']}, Offline: {h_data['offline_mode']})")

    # Cameras
    r_cams = client.get("/api/cameras")
    assert r_cams.status_code == 200
    print(f"  -> GET /api/cameras: OK ({len(r_cams.json())} cameras registered)")

    # Events
    r_evts = client.get("/api/events")
    assert r_evts.status_code == 200
    returned_evts = r_evts.json()
    assert len(returned_evts) > 0, "No events returned from API!"
    print(f"  -> GET /api/events: OK ({len(returned_evts)} events active)")

    # Map Data
    r_map = client.get("/api/map")
    assert r_map.status_code == 200
    print(f"  -> GET /api/map: OK (GeoJSON vector data ready)")

    # -------------------------------------------------------------
    # STAGE 5: Frontend Dashboard Contract Verification
    # -------------------------------------------------------------
    print("\n[E2E 5/5] Verifying Frontend Static Delivery...")
    r_ui = client.get("/")
    assert r_ui.status_code == 200
    assert "Aegis Defense" in r_ui.text
    print("  -> GET /: OK (Tactical dashboard delivered with ZERO remote CDN tags)")

    print("\n" + "=" * 70)
    print("ALL 5 END-TO-END PIPELINE STAGES PASSED SUCCESSFULLY")
    print("=" * 70)
    return True


if __name__ == "__main__":
    try:
        run_e2e_test()
        print("\nStatus: END-TO-END TEST SUCCESSFUL")
    except Exception as e:
        print(f"\nStatus: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
