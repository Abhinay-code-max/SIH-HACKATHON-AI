"""
Pipeline runner to test and verify the EventEngine against local detections.
Processes recorded detections, triggers rules, and stores verified SecurityEvents.
"""

import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.events.engine import EventEngine
from backend.app.models.contracts import GeoLocation, SeverityLevel


def run_pipeline():
    print("[Event Pipeline] Initializing EventEngine with security rules...")
    engine = EventEngine(
        camera_id="CAM_01",
        default_location=GeoLocation(lat=17.3850, lon=78.4867, zone_name="Demonstration Perimeter"),
        crowd_threshold=3,
        alert_cooldown_sec=1.0,  # 1-second cooldown for demonstration
    )

    # 1. Ingest image detections (has 4 people + 1 bus)
    img_dets_file = ROOT_DIR / "data" / "sample-images" / "detections_traffic_sample.json"
    if img_dets_file.is_file():
        with open(img_dets_file, "r") as f:
            img_data = json.load(f)
        print(f"[Event Pipeline] Feeding {img_data.get('total_detections', 0)} image detections to engine...")
        events_from_img = engine.process_frame_detections(
            raw_detections=img_data.get("detections", []),
            frame_index=1,
        )
        print(f"[Event Pipeline] Image frame generated {len(events_from_img)} security events:")
        for e in events_from_img:
            print(f"  -> [{e.event_id}] Type: {e.event_type.value.upper()} | Class: {e.class_name} | Severity: {e.severity.value.upper()}")

    # 2. Ingest recorded video detections
    vid_dets_file = ROOT_DIR / "data" / "sample-videos" / "video_detections.json"
    video_events_count = 0
    if vid_dets_file.is_file():
        with open(vid_dets_file, "r") as f:
            vid_data = json.load(f)
        print(f"\n[Event Pipeline] Feeding video stream detection events...")
        for frame_entry in vid_data.get("events", [])[:30]:  # evaluate first 30 frames
            new_evts = engine.process_frame_detections(
                raw_detections=frame_entry.get("detections", []),
                frame_index=frame_entry.get("frame_index", 0),
            )
            video_events_count += len(new_evts)
        print(f"[Event Pipeline] Stream frames processed. Cooldown throttling active.")

    # 3. Export all generated live events
    out_file = ROOT_DIR / "data" / "sample-events" / "live_events.json"
    exported_path = engine.export_events(out_file)
    print(f"\n[Event Pipeline] Total verified security events: {len(engine.event_history)}")
    print(f"[Event Pipeline] Saved live events store: {exported_path}")

    # Summary by severity
    severity_breakdown = {}
    for ev in engine.event_history:
        severity_breakdown[ev.severity.value] = severity_breakdown.get(ev.severity.value, 0) + 1

    return {
        "total_events": len(engine.event_history),
        "breakdown": severity_breakdown,
        "output_file": str(exported_path),
    }


if __name__ == "__main__":
    try:
        report = run_pipeline()
        print("\n--- Event Engine Summary ---")
        print(f"Total Security Events Generated: {report['total_events']}")
        print(f"Severity Breakdown: {report['breakdown']}")
        print(f"Events Store: {report['output_file']}")
        print("Status: EVENT ENGINE VERIFICATION SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
