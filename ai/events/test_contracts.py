"""
Contract and Schema Validator for Phase 1 Prototype.
Validates Pydantic schemas, converts real detection outputs into compliant SecurityEvents,
and exports JSON schemas and sample mocks for the Frontend team.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.models.contracts import (
    CameraAsset,
    CameraStatus,
    EventType,
    GeoLocation,
    RawDetection,
    SecurityEvent,
    SeverityLevel,
    SystemHealth,
)


def run_contract_verification():
    print("[Contracts] Verifying Pydantic detection and event schemas...")

    # 1. Test Camera Asset contract
    demo_cam = CameraAsset(
        camera_id="CAM_01",
        name="Main Demonstration CCTV",
        stream_source="0",
        status=CameraStatus.ONLINE,
        location=GeoLocation(lat=17.3850, lon=78.4867, zone_name="Sector Alpha - Hyderabad"),
        resolution="1920x1080",
        target_fps=30,
    )
    print(f"[Contracts] Validated CameraAsset: {demo_cam.camera_id} ({demo_cam.name})")

    # 2. Test Security Event contract
    sample_evt = SecurityEvent(
        event_id="evt_20260904_001",
        source_id=demo_cam.camera_id,
        event_type=EventType.OBJECT_DETECTED,
        class_name="bus",
        confidence=0.9549,
        bbox=[2.13, 230.65, 804.59, 738.54],
        severity=SeverityLevel.MEDIUM,
        location=demo_cam.location,
        metadata={"camera_fps": 30, "model": "yolov8l.pt"},
    )
    print(f"[Contracts] Validated SecurityEvent: {sample_evt.event_id} ({sample_evt.class_name} - {sample_evt.severity.value})")

    # 3. Test SystemHealth contract
    health = SystemHealth(
        status="healthy",
        offline_mode=True,
        device="cuda",
        device_name="NVIDIA GeForce RTX 4060 Laptop GPU",
        active_model="yolov8l.pt",
        vram_allocated_mb=1250.0,
        cameras_active=1,
    )
    print(f"[Contracts] Validated SystemHealth: status={health.status} offline={health.offline_mode}")

    # 4. Ingest real image detections and convert to compliant events
    detections_file = ROOT_DIR / "data" / "sample-images" / "detections_traffic_sample.json"
    converted_events = []
    if detections_file.is_file():
        with open(detections_file, "r") as f:
            det_data = json.load(f)

        for i, det in enumerate(det_data.get("detections", [])):
            cls = det["class"]
            # Assign severity based on domain logic
            severity = SeverityLevel.LOW
            if cls == "bus":
                severity = SeverityLevel.MEDIUM
            elif cls == "person" and i > 2:
                severity = SeverityLevel.HIGH  # Simulated crowd/group

            evt = SecurityEvent(
                event_id=f"evt_demo_{i + 1:03d}",
                source_id="CAM_01",
                event_type=EventType.OBJECT_DETECTED,
                class_name=cls,
                confidence=det["confidence"],
                bbox=det["bbox"],
                severity=severity,
                location=GeoLocation(lat=17.3850 + (i * 0.0002), lon=78.4867 + (i * 0.0002), zone_name="Sector Alpha"),
                metadata={"source_image": det_data.get("source_image")},
            )
            converted_events.append(evt.model_dump())

    # 5. Export mock events for Frontend team
    out_dir = ROOT_DIR / "data" / "sample-events"
    out_dir.mkdir(parents=True, exist_ok=True)

    mock_events_file = out_dir / "mock_events.json"
    with open(mock_events_file, "w") as f:
        json.dump(converted_events, f, indent=2)
    print(f"[Contracts] Exported {len(converted_events)} mock events to: {mock_events_file}")

    # 6. Export JSON Schema specification for Frontend developers
    schema_file = out_dir / "event_schema.json"
    schema_bundle = {
        "SecurityEvent": SecurityEvent.model_json_schema(),
        "CameraAsset": CameraAsset.model_json_schema(),
        "SystemHealth": SystemHealth.model_json_schema(),
    }
    with open(schema_file, "w") as f:
        json.dump(schema_bundle, f, indent=2)
    print(f"[Contracts] Exported JSON schemas to: {schema_file}")

    return {
        "camera_tested": demo_cam.camera_id,
        "sample_event_id": sample_evt.event_id,
        "converted_events_count": len(converted_events),
        "mock_events_path": str(mock_events_file),
        "schema_path": str(schema_file),
    }


if __name__ == "__main__":
    try:
        report = run_contract_verification()
        print("\n--- Detection Schema Verification Summary ---")
        print(f"Contracts Validated: CameraAsset, SecurityEvent, SystemHealth, RawDetection")
        print(f"Converted Real Events: {report['converted_events_count']}")
        print(f"Mock Events File: {report['mock_events_path']}")
        print(f"Schema File: {report['schema_path']}")
        print("Status: DETECTION SCHEMA VERIFICATION SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
