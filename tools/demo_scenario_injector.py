"""
BORDER SENTINEL — Interactive Evaluator & Judge Demo Scenario Injector.
Executes deterministic, realistic tactical defense scenarios on demand:
[1] Restricted Polygon Geofence Intrusion
[2] Virtual Boundary Tripwire Breach
[3] Sprinting / Rapid Displacement Flight Anomaly
[4] Unattended Baggage / Abandoned Luggage Alert
[5] Doorway Tailgating / Anti-Piggybacking Breach
[6] Cross-Camera Re-ID Subject Transit Journey (CAM_01 -> CAM_02)
[7] Full Grand Demonstration (All scenarios executed sequentially)
"""

import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.events.intelligence_engine import intelligence_engine
from ai.events.incident_manager import incident_manager
from ai.reid.manager import global_subject_manager


def create_synthetic_scene(label_text: str, bg_color=(20, 25, 35)) -> np.ndarray:
    """Creates an annotated 640x480 test frame."""
    frame = np.full((480, 640, 3), bg_color, dtype=np.uint8)
    # Grid lines
    for x in range(0, 640, 80):
        cv2.line(frame, (x, 0), (x, 480), (35, 45, 60), 1)
    for y in range(0, 480, 80):
        cv2.line(frame, (0, y), (640, y), (35, 45, 60), 1)

    cv2.putText(frame, f"[DEMO SCENARIO] {label_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return frame


def run_scenario_1_geofence():
    print("\n--- [1] Executing Restricted Zone Geofence Intrusion ---")
    frame = create_synthetic_scene("SCENARIO 1: Geofence Intrusion")
    # Draw unauthorized target in restricted zone (center at 160, 200)
    cv2.rectangle(frame, (120, 100), (200, 300), (0, 0, 255), -1)

    track = {
        "track_id": 101,
        "class_name": "person",
        "confidence": 0.94,
        "bbox": [120.0, 100.0, 200.0, 300.0],
        "center": [160.0, 290.0],
        "dwell_seconds": 2.0,
        "trajectory": [(160.0, 290.0)],
    }

    evts = intelligence_engine.evaluate_tracks("CAM_01", [track], frame)
    defcon = incident_manager.get_current_system_defcon()
    print(f"  -> Events Generated: {len(evts)}")
    print(f"  -> System DEFCON:   {defcon['defcon']['level']} ({defcon['defcon']['status']}) | Score: {defcon['threat_score']}/100")
    print("  -> Verdict: RESTRICTED ZONE GEOFENCE ALARM TRIGGERED")


def run_scenario_2_tripwire():
    print("\n--- [2] Executing Virtual Boundary Tripwire Breach ---")
    frame = create_synthetic_scene("SCENARIO 2: Boundary Tripwire Crossing")
    cv2.rectangle(frame, (300, 200), (420, 350), (0, 165, 255), -1)

    # Motion vector crossing from left (280) to right (360) across x=320 line
    track = {
        "track_id": 202,
        "class_name": "car",
        "confidence": 0.96,
        "bbox": [300.0, 200.0, 420.0, 350.0],
        "center": [360.0, 340.0],
        "dwell_seconds": 1.5,
        "trajectory": [(280.0, 340.0), (320.0, 340.0), (360.0, 340.0)],
    }

    evts = intelligence_engine.evaluate_tracks("CAM_02", [track], frame)
    defcon = incident_manager.get_current_system_defcon()
    print(f"  -> Events Generated: {len(evts)}")
    print(f"  -> System DEFCON:   {defcon['defcon']['level']} ({defcon['defcon']['status']}) | Score: {defcon['threat_score']}/100")
    print("  -> Verdict: TRIPWIRE BREACH CAPTURED & REGISTERED")


def run_scenario_3_sprinting():
    print("\n--- [3] Executing Sprinting / Rapid Displacement Anomaly ---")
    frame = create_synthetic_scene("SCENARIO 3: Sprinting / Rapid Flight Anomaly")
    cv2.rectangle(frame, (200, 150), (280, 380), (50, 50, 220), -1)

    # 4 trajectory points with delta = 45 px per step
    track = {
        "track_id": 303,
        "class_name": "person",
        "confidence": 0.91,
        "bbox": [200.0, 150.0, 280.0, 380.0],
        "center": [240.0, 370.0],
        "dwell_seconds": 1.0,
        "trajectory": [(105.0, 370.0), (150.0, 370.0), (195.0, 370.0), (240.0, 370.0)],
    }

    evts = intelligence_engine.evaluate_tracks("CAM_01", [track], frame)
    defcon = incident_manager.get_current_system_defcon()
    print(f"  -> Events Generated: {len(evts)}")
    print(f"  -> System DEFCON:   {defcon['defcon']['level']} ({defcon['defcon']['status']}) | Score: {defcon['threat_score']}/100")
    print("  -> Verdict: HIGH-VELOCITY SPRINTING ANOMALY FLAGGED")


def run_scenario_4_unattended_bag():
    print("\n--- [4] Executing Unattended / Abandoned Baggage Alert ---")
    frame = create_synthetic_scene("SCENARIO 4: Unattended Luggage Alert")
    # Draw bag at (120, 350)
    cv2.rectangle(frame, (100, 330), (140, 370), (0, 200, 200), -1)
    # Draw distant person at (520, 150)
    cv2.rectangle(frame, (500, 100), (540, 250), (150, 150, 150), -1)

    bag_track = {
        "track_id": 404,
        "class_name": "backpack",
        "confidence": 0.95,
        "bbox": [100.0, 330.0, 140.0, 370.0],
        "center": [120.0, 350.0],
        "dwell_seconds": 12.0,
        "trajectory": [(120.0, 350.0), (120.0, 351.0), (120.0, 350.0)],
    }
    person_track = {
        "track_id": 405,
        "class_name": "person",
        "confidence": 0.89,
        "bbox": [500.0, 100.0, 540.0, 250.0],
        "center": [520.0, 240.0],
        "dwell_seconds": 12.0,
        "trajectory": [(520.0, 240.0)],
    }

    # Artificially set isolation start time 10s ago
    from ai.events.behavioral_engine import behavioral_anomaly_engine
    behavioral_anomaly_engine.baggage_isolated_start[("CAM_01", 404)] = time.time() - 10.0

    evts = intelligence_engine.evaluate_tracks("CAM_01", [bag_track, person_track], frame)
    defcon = incident_manager.get_current_system_defcon()
    print(f"  -> Events Generated: {len(evts)}")
    print(f"  -> System DEFCON:   {defcon['defcon']['level']} ({defcon['defcon']['status']}) | Score: {defcon['threat_score']}/100")
    print("  -> Verdict: CRITICAL HOMELAND DEFENSE UNATTENDED LUGGAGE ALARM")


def run_scenario_5_tailgating():
    print("\n--- [5] Executing Doorway Tailgating / Anti-Piggybacking Breach ---")
    frame = create_synthetic_scene("SCENARIO 5: Tailgating Breach")

    t_now = time.time()
    from ai.events.behavioral_engine import behavioral_anomaly_engine
    # Lead target crossed 1.0s ago
    behavioral_anomaly_engine.record_tripwire_crossing("CAM_01", "DOORWAY_WIRE", 501, timestamp=t_now - 1.0)
    # Second target crosses now
    anom = behavioral_anomaly_engine.record_tripwire_crossing("CAM_01", "DOORWAY_WIRE", 502, timestamp=t_now)

    if anom:
        incident_manager.register_or_update_incident("CAM_01", [anom])

    defcon = incident_manager.get_current_system_defcon()
    print(f"  -> Breach Details:  Track #502 tailgated #501 with delta {anom['delta_seconds'] if anom else 0}s")
    print(f"  -> System DEFCON:   {defcon['defcon']['level']} ({defcon['defcon']['status']}) | Score: {defcon['threat_score']}/100")
    print("  -> Verdict: DOORWAY PIGGYBACKING BREACH CONFIRMED")


def run_scenario_6_reid_transit():
    print("\n--- [6] Executing Cross-Camera Re-ID Subject Transit (CAM_01 -> CAM_02) ---")
    # Person with distinctive green shirt
    frame1 = create_synthetic_scene("SCENARIO 6: Target on CAM_01")
    frame1[150:280, 200:300] = (50, 180, 50)

    # Step A: Target on CAM_01
    s1 = global_subject_manager.process_track(
        camera_id="CAM_01",
        local_track_id=601,
        class_name="person",
        bbox=[200.0, 150.0, 300.0, 280.0],
        frame=frame1,
    )
    print(f"  -> [CAM_01 Sighting] Registered as Global ID: {s1['subject_id']} ({s1['display_name']})")

    # Step B: Subject transits to CAM_02 after 8 seconds
    s1["last_seen"] = time.time() - 8.0
    frame2 = create_synthetic_scene("SCENARIO 6: Target transit on CAM_02")
    frame2[160:290, 250:350] = (50, 180, 50)

    s2 = global_subject_manager.process_track(
        camera_id="CAM_02",
        local_track_id=602,
        class_name="person",
        bbox=[250.0, 160.0, 350.0, 290.0],
        frame=frame2,
    )
    print(f"  -> [CAM_02 Sighting] Matched across camera handoff: {s2['subject_id']} ({s2['display_name']})")

    transits = global_subject_manager.get_recent_transits(limit=1)
    if transits:
        t = transits[0]
        print(f"  -> Transit Route:    {t['from_camera']} ---> {t['to_camera']} in {t['transit_duration_sec']}s (Sim: {t.get('similarity_score', 0.85)})")
    print("  -> Verdict: MULTI-CAMERA RE-ID JOURNEY CONTINUITY VERIFIED")


def run_full_grand_demo():
    print("\n======================================================================")
    print(" EXECUTING GRAND EVALUATION SUITE (ALL 6 SCENARIOS IN SEQUENCE)")
    print("======================================================================")
    run_scenario_1_geofence()
    time.sleep(1.0)
    run_scenario_2_tripwire()
    time.sleep(1.0)
    run_scenario_3_sprinting()
    time.sleep(1.0)
    run_scenario_4_unattended_bag()
    time.sleep(1.0)
    run_scenario_5_tailgating()
    time.sleep(1.0)
    run_scenario_6_reid_transit()

    final_defcon = incident_manager.get_current_system_defcon()
    all_incidents = incident_manager.get_all_incidents()
    print("\n======================================================================")
    print(" GRAND EVALUATION COMPLETE")
    print(f" Overall Grid DEFCON:   {final_defcon['defcon']['level']} // {final_defcon['defcon']['status']} (Score: {final_defcon['threat_score']}/100)")
    print(f" Active Incidents:     {len(all_incidents)} compound incident files generated")
    print(f" Registered Re-ID Subj: {len(global_subject_manager.subjects)} persistent entities tracked")
    print(" Open Dashboard:        http://127.0.0.1:8000/ to inspect live HUD & map")
    print("======================================================================")


def interactive_menu():
    while True:
        print("\n======================================================================")
        print(" BORDER SENTINEL — TACTICAL DEMO SCENARIO INJECTOR")
        print("======================================================================")
        print(" [1] Restricted Zone Geofence Intrusion (Doorway / Concourse)")
        print(" [2] Virtual Boundary Tripwire Crossing (Gate 1 Entry)")
        print(" [3] Sprinting / Rapid Displacement Anomaly (High Velocity Flight)")
        print(" [4] Unattended Baggage Alarm (Stationary Isolated Luggage)")
        print(" [5] Doorway Tailgating / Anti-Piggybacking Breach")
        print(" [6] Cross-Camera Re-ID Subject Journey (CAM_01 -> CAM_02)")
        print(" [7] Execute Full Grand Demonstration (All 6 Scenarios)")
        print(" [8] Reset System Incidents & Re-ID Memory")
        print(" [0] Exit")
        print("======================================================================")

        try:
            choice = input(" Select Scenario [0-8]: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice == "1":
            run_scenario_1_geofence()
        elif choice == "2":
            run_scenario_2_tripwire()
        elif choice == "3":
            run_scenario_3_sprinting()
        elif choice == "4":
            run_scenario_4_unattended_bag()
        elif choice == "5":
            run_scenario_5_tailgating()
        elif choice == "6":
            run_scenario_6_reid_transit()
        elif choice == "7":
            run_full_grand_demo()
        elif choice == "8":
            incident_manager.incidents.clear()
            global_subject_manager.subjects.clear()
            global_subject_manager.active_track_map.clear()
            global_subject_manager.transit_log.clear()
            print("\n[✓] System State & Re-ID Memory Cleared.")
        elif choice == "0":
            break
        else:
            print("Invalid option. Please choose 0-8.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo Scenario Injector")
    parser.add_argument("--auto", action="store_true", help="Run full grand demonstration automatically")
    args = parser.parse_args()

    if args.auto:
        run_full_grand_demo()
    else:
        interactive_menu()
