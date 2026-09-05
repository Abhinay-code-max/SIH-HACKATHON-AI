"""
Automated Test Suite for Phase v0.9: Behavioral Anomaly AI & Incident Management.
Validates:
1. Sprinting / rapid displacement detection.
2. Unattended / abandoned baggage detection & proximity tracking.
3. Doorway tailgating / anti-piggybacking verification.
4. Compound threat risk scoring & DEFCON level determination.
5. Decoupled REST API endpoints (listing, report export, config tuning, webhooks).
"""

from pathlib import Path
import sys
import time
from fastapi.testclient import TestClient
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.events.behavioral_engine import behavioral_anomaly_engine
from ai.events.incident_manager import incident_manager
from backend.app.main import app


def reset_test_state():
    behavioral_anomaly_engine.baggage_isolated_start.clear()
    behavioral_anomaly_engine.tripwire_crossing_history.clear()
    behavioral_anomaly_engine.cooldowns.clear()
    incident_manager.incidents.clear()
    incident_manager.next_incident_idx = 1


def test_sprint_velocity_detection():
    reset_test_state()

    # Track 1: Fast sprinting (displacement of 120 px across 3 steps = 40 px/step)
    sprinting_track = {
        "track_id": 10,
        "class_name": "person",
        "confidence": 0.92,
        "bbox": [100.0, 100.0, 150.0, 200.0],
        "trajectory": [(100.0, 100.0), (140.0, 100.0), (180.0, 100.0), (220.0, 100.0)],
        "center": [220.0, 190.0],
    }

    # Track 2: Slow walking (displacement of 12 px across 3 steps = 4 px/step)
    walking_track = {
        "track_id": 20,
        "class_name": "person",
        "confidence": 0.88,
        "bbox": [300.0, 300.0, 350.0, 400.0],
        "trajectory": [(300.0, 300.0), (304.0, 300.0), (308.0, 300.0), (312.0, 300.0)],
        "center": [312.0, 390.0],
    }

    anomalies = behavioral_anomaly_engine.evaluate_motion_anomalies("CAM_01", [sprinting_track, walking_track])
    sprint_events = [a for a in anomalies if a["anomaly_type"] == "SPRINTING_DETECTED"]

    assert len(sprint_events) == 1, "Expected exactly 1 sprinting anomaly detected"
    assert sprint_events[0]["track_id"] == 10
    assert sprint_events[0]["velocity"] >= 35.0
    assert sprint_events[0]["severity"] == "HIGH"


def test_unattended_baggage_detection():
    reset_test_state()

    # Baggage track (stationary at (100, 100))
    bag_track = {
        "track_id": 44,
        "class_name": "backpack",
        "confidence": 0.94,
        "bbox": [85.0, 85.0, 115.0, 115.0],
        "trajectory": [(100.0, 100.0), (101.0, 100.0), (100.0, 101.0)],
        "center": [100.0, 100.0],
    }

    # Person track far away (at (450, 450), distance > 400 px)
    person_track = {
        "track_id": 7,
        "class_name": "person",
        "confidence": 0.90,
        "bbox": [430.0, 400.0, 470.0, 500.0],
        "trajectory": [(450.0, 450.0), (450.0, 450.0)],
        "center": [450.0, 450.0],
    }

    # Frame 1: Baggage first isolated
    t0 = time.time() - 9.0  # Artificially set 9s ago to trigger threshold (>7s)
    behavioral_anomaly_engine.baggage_isolated_start[("CAM_01", 44)] = t0

    anomalies = behavioral_anomaly_engine.evaluate_motion_anomalies("CAM_01", [bag_track, person_track])
    bag_events = [a for a in anomalies if a["anomaly_type"] == "UNATTENDED_BAGGAGE_ALERT"]

    assert len(bag_events) == 1, "Expected unattended baggage alarm triggered"
    assert bag_events[0]["track_id"] == 44
    assert bag_events[0]["severity"] == "CRITICAL"
    assert bag_events[0]["nearest_person_dist"] > 120.0


def test_tailgating_detection():
    reset_test_state()
    wire_id = "GATE_ENTRY_WIRE"

    now = time.time()
    # Step 1: Track #101 crosses tripwire
    a1 = behavioral_anomaly_engine.record_tripwire_crossing("CAM_02", wire_id, 101, timestamp=now)
    assert a1 is None, "First crossing must not trigger tailgating"

    # Step 2: Track #102 crosses 1.1s later (< 1.8s window) -> Tailgating!
    a2 = behavioral_anomaly_engine.record_tripwire_crossing("CAM_02", wire_id, 102, timestamp=now + 1.1)
    assert a2 is not None, "Second crossing within 1.1s must trigger tailgating breach"
    assert a2["anomaly_type"] == "TAILGATING_BREACH"
    assert a2["lead_track_id"] == 101
    assert a2["tail_track_id"] == 102
    assert a2["delta_seconds"] == 1.1

    # Step 3: Track #103 crosses 5.0s later (> 1.8s) -> Normal interval, no anomaly
    a3 = behavioral_anomaly_engine.record_tripwire_crossing("CAM_02", wire_id, 103, timestamp=now + 6.1)
    assert a3 is None, "Crossing after authorization interval must not trigger tailgating"


def test_incident_risk_scoring_and_defcon():
    reset_test_state()

    # 1. Low-level routine event
    routine_events = [{"event_type": "LOITERING_DETECTED", "severity": "MEDIUM", "class_name": "person"}]
    score_low = incident_manager.calculate_threat_score(routine_events)
    defcon_low = incident_manager.get_defcon_level(score_low)
    assert score_low < 45
    assert defcon_low["level"] == "DEFCON_3"

    # 2. Critical compound incident (unattended baggage + zone intrusion + multi-cam)
    critical_events = [
        {"event_type": "RESTRICTED_ZONE_INTRUSION", "severity": "CRITICAL", "class_name": "person"},
        {"anomaly_type": "UNATTENDED_BAGGAGE_ALERT", "severity": "CRITICAL", "class_name": "backpack"},
    ]
    score_high = incident_manager.calculate_threat_score(critical_events, has_multi_camera_transit=True)
    defcon_high = incident_manager.get_defcon_level(score_high)
    assert score_high >= 75
    assert defcon_high["level"] == "DEFCON_1"
    assert defcon_high["status"] == "CRITICAL"


def test_incident_api_endpoints():
    reset_test_state()
    client = TestClient(app)

    # 1. Register an incident
    inc = incident_manager.register_or_update_incident(
        camera_id="CAM_01",
        events=[{
            "event_type": "RESTRICTED_ZONE_INTRUSION",
            "severity": "CRITICAL",
            "class_name": "person",
            "track_id": 5,
            "description": "Perimeter breach",
        }],
        global_subject_id="SUBJ_0001",
    )
    inc_id = inc["incident_id"]

    # 2. GET /api/incidents
    res_list = client.get("/api/incidents")
    assert res_list.status_code == 200
    inc_list = res_list.json()
    assert len(inc_list) >= 1
    assert inc_list[0]["incident_id"] == inc_id

    # 3. GET /api/incidents/defcon
    res_defcon = client.get("/api/incidents/defcon")
    assert res_defcon.status_code == 200
    defcon_data = res_defcon.json()
    assert "threat_score" in defcon_data
    assert "defcon" in defcon_data

    # 4. GET /api/incidents/{id}
    res_dossier = client.get(f"/api/incidents/{inc_id}")
    assert res_dossier.status_code == 200
    dossier = res_dossier.json()
    assert dossier["incident_id"] == inc_id
    assert dossier["global_subject_id"] == "SUBJ_0001"

    # 5. GET /api/incidents/{id}/report (HTML Export)
    res_report = client.get(f"/api/incidents/{inc_id}/report")
    assert res_report.status_code == 200
    assert "BORDER SENTINEL // TACTICAL INCIDENT REPORT" in res_report.text
    assert inc_id in res_report.text

    # 6. GET & POST /api/incidents/anomalies/config
    res_cfg = client.get("/api/incidents/anomalies/config")
    assert res_cfg.status_code == 200
    assert "sprint_velocity_threshold" in res_cfg.json()

    res_cfg_update = client.post("/api/incidents/anomalies/config", json={"sprint_velocity_threshold": 45.0})
    assert res_cfg_update.status_code == 200
    assert res_cfg_update.json()["config"]["sprint_velocity_threshold"] == 45.0

    # 7. POST /api/incidents/webhooks/register
    res_hook = client.post("/api/incidents/webhooks/register", json={"webhook_url": "http://127.0.0.1:9999/dispatch"})
    assert res_hook.status_code == 200
    assert "http://127.0.0.1:9999/dispatch" in res_hook.json()["webhooks"]


if __name__ == "__main__":
    print("\n=======================================================")
    print("RUNNING PHASE v0.9 BEHAVIORAL AI & INCIDENTS TEST SUITE")
    print("=======================================================")

    print("[1/5] Testing Sprinting & Velocity Spike Detection...")
    test_sprint_velocity_detection()
    print("      --> PASS: Sprinting correctly distinguished from normal walking.")

    print("[2/5] Testing Unattended Baggage & Isolation Tracking...")
    test_unattended_baggage_detection()
    print("      --> PASS: Stationary isolated baggage alarm verified.")

    print("[3/5] Testing Doorway Tailgating & Anti-Piggybacking...")
    test_tailgating_detection()
    print("      --> PASS: Consecutive crossings under 1.8s flagged as breach.")

    print("[4/5] Testing Compound Threat Risk Scoring & DEFCON Levels...")
    test_incident_risk_scoring_and_defcon()
    print("      --> PASS: Threat scoring maps to DEFCON 1 (CRITICAL) and DEFCON 3 (NORMAL).")

    print("[5/5] Testing Decoupled Incident REST & Report Export APIs...")
    test_incident_api_endpoints()
    print("      --> PASS: /api/incidents, DEFCON, HTML reports, and config tuning verified.")

    print("\nSTATUS: ALL PHASE v0.9 BEHAVIORAL & INCIDENT TESTS PASSED! [5/5]")
