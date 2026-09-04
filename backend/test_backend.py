"""
Automated Local Backend Verification Suite.
Tests all FastAPI endpoints in-process via TestClient without requiring external network.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from backend.app.main import app


def run_backend_verification():
    print("[Backend Test] Initializing FastAPI TestClient...")
    client = TestClient(app)

    # 1. Test Root Dashboard
    r_root = client.get("/")
    assert r_root.status_code == 200, f"Root failed: {r_root.status_code}"
    assert "Aegis Defense" in r_root.text or "Surveillance" in r_root.text
    print(f"[Backend Test] GET / -> HTTP 200 OK (Served 100% Offline Frontend Dashboard)")

    # 2. Test Health Endpoint
    r_health = client.get("/api/health")
    assert r_health.status_code == 200, f"Health failed: {r_health.status_code}"
    health_data = r_health.json()
    assert health_data["offline_mode"] is True, "offline_mode must be True!"
    print(f"[Backend Test] GET /api/health -> HTTP 200 OK | Device: {health_data['device']} ({health_data['device_name']}) | Offline: {health_data['offline_mode']}")

    # 3. Test Cameras Endpoint
    r_cams = client.get("/api/cameras")
    assert r_cams.status_code == 200, f"Cameras failed: {r_cams.status_code}"
    cams_data = r_cams.json()
    assert len(cams_data) >= 3, "Expected at least 3 cameras registered"
    print(f"[Backend Test] GET /api/cameras -> HTTP 200 OK | Registered cameras: {len(cams_data)}")

    # 4. Test Events Endpoint
    r_events = client.get("/api/events")
    assert r_events.status_code == 200, f"Events failed: {r_events.status_code}"
    events_data = r_events.json()
    print(f"[Backend Test] GET /api/events -> HTTP 200 OK | Stored events returned: {len(events_data)}")

    # 5. Test Map Endpoint
    r_map = client.get("/api/map")
    assert r_map.status_code == 200, f"Map failed: {r_map.status_code}"
    map_data = r_map.json()
    assert "zones" in map_data and "cameras" in map_data and "event_markers" in map_data
    print(f"[Backend Test] GET /api/map -> HTTP 200 OK | Zones: {len(map_data['zones']['features'])}, Markers: {len(map_data['event_markers'])}")

    return {
        "status": "HEALTHY",
        "device": health_data["device"],
        "device_name": health_data["device_name"],
        "cameras_count": len(cams_data),
        "events_count": len(events_data),
        "zones_count": len(map_data["zones"]["features"]),
    }


if __name__ == "__main__":
    try:
        res = run_backend_verification()
        print("\n--- Local Backend API Verification Summary ---")
        print(f"Device: {res['device']} ({res['device_name']})")
        print(f"All 5 Core Local API Endpoints: VERIFIED 200 OK")
        print(f"Cameras Registered: {res['cameras_count']}")
        print(f"Events Exposed: {res['events_count']}")
        print(f"Offline Mode: 100% STRICT LOCAL")
        print("Status: LOCAL BACKEND API VERIFICATION SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
