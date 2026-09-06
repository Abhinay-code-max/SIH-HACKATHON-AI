"""
Comprehensive Test Suite for Vehicle Behavior Intelligence, Border Heading Vectoring,
and Surveillance Intelligence Graph.

Validates:
1. MovementAnalyzer heading angle & border proximity vector classification (TOWARD, AWAY, PARALLEL, STATIONARY).
2. VehicleBehaviorAnalyzer detecting stationary vehicle stoppage (< 2.5 px/step) in restricted zones.
3. Vehicle loitering detection and compound DEFCON risk scoring integration.
4. SurveillanceIntelligenceGraph node/edge creation, journey tracing (CAM_01 -> CAM_02 -> CAM_03), and D3 export.
5. Topological path anomaly detection (teleportation rejection, checkpoint bypassing, circular patrolling).
"""

from pathlib import Path
import sys
import time
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.tracking.movement_analyzer import MovementAnalyzer, movement_analyzer
from ai.tracking.vehicle_analyzer import VehicleBehaviorAnalyzer, vehicle_analyzer
from ai.reid.intelligence_graph import SurveillanceIntelligenceGraph, surveillance_graph
from ai.events.intelligence_engine import intelligence_engine
from ai.events.incident_manager import incident_manager


def test_stage_1_movement_vectoring():
    print("\n[Stage 1/5] Testing Movement Heading & Border Vector Classification...")
    analyzer = MovementAnalyzer(stationary_threshold_px=2.0, default_border_normal_deg=90.0)

    # 1. Heading directly South (dy > 0, dx = 0) toward bottom border normal (90 deg)
    state, speed, heading, is_approaching = analyzer.classify_movement_vector(dx=0.0, dy=12.0)
    assert state == "TOWARD_BORDER", f"Expected TOWARD_BORDER, got {state}"
    assert is_approaching is True
    assert heading == 90.0
    print(f"  -> TOWARD_BORDER: Speed={speed} px/step, Heading={heading} deg, Approaching={is_approaching} [OK]")

    # 2. Heading East (dy = 0, dx = 15.0) parallel to horizontal border
    state, speed, heading, is_approaching = analyzer.classify_movement_vector(dx=15.0, dy=0.0)
    assert state == "PARALLEL_TO_BORDER", f"Expected PARALLEL_TO_BORDER, got {state}"
    assert heading == 0.0
    print(f"  -> PARALLEL_TO_BORDER: Speed={speed} px/step, Heading={heading} deg [OK]")

    # 3. Heading North (dy = -10.0, dx = 0.0) away from bottom border
    state, speed, heading, is_approaching = analyzer.classify_movement_vector(dx=0.0, dy=-10.0)
    assert state == "AWAY_FROM_BORDER", f"Expected AWAY_FROM_BORDER, got {state}"
    assert is_approaching is False
    assert heading == 270.0
    print(f"  -> AWAY_FROM_BORDER: Speed={speed} px/step, Heading={heading} deg [OK]")

    # 4. Stationary target (displacement < 2.0 px)
    state, speed, heading, is_approaching = analyzer.classify_movement_vector(dx=0.8, dy=0.6)
    assert state == "STATIONARY", f"Expected STATIONARY, got {state}"
    print(f"  -> STATIONARY: Displacement={speed} px [OK]")

    # 5. Full track dictionary analysis
    sample_track = {
        "track_id": 42,
        "class_name": "person",
        "trajectory": [(100.0, 100.0), (100.0, 120.0), (100.0, 145.0), (100.0, 175.0)],
    }
    telemetry = analyzer.analyze_track(sample_track, fps=30.0)
    assert telemetry["movement_state"] == "TOWARD_BORDER"
    assert telemetry["speed_px_s"] > 0
    print(f"  -> Track Telemetry: Speed={telemetry['speed_px_s']} px/s, State={telemetry['movement_state']} [OK]")
    print("  --> PASS: Stage 1 Movement Vectoring Verified.")


def test_stage_2_vehicle_stoppage():
    print("\n[Stage 2/5] Testing Vehicle Stoppage Detection in Restricted Zones...")
    v_analyzer = VehicleBehaviorAnalyzer(
        stoppage_speed_threshold_px=2.5,
        stoppage_threshold_seconds=4.0,
        cooldown_seconds=1.0,
    )

    dummy_zone = [{
        "camera_id": "CAM_02",
        "zone_id": "ZONE_VEHICLE_GATE",
        "name": "Gate 1 Restricted Lane",
        "polygon": [[100, 100], [500, 100], [500, 400], [100, 400]],
    }]

    stopped_vehicle_track = {
        "track_id": 88,
        "class_name": "truck",
        "confidence": 0.95,
        "bbox": [200.0, 200.0, 320.0, 300.0],
        "center": [260.0, 290.0],  # Inside zone
        "dwell_seconds": 1.0,
        "trajectory": [(260.0, 290.0), (260.5, 290.2), (260.8, 290.5)],  # speed < 1.0 px/step
    }

    t0 = 1000.0
    # First frame: Stoppage begins, no alert yet
    anoms = v_analyzer.evaluate_vehicles("CAM_02", [stopped_vehicle_track], zones=dummy_zone, timestamp=t0)
    assert len(anoms) == 0, f"Expected 0 anomalies at t0, got {len(anoms)}"
    print("  -> t = 0.0s: Vehicle stopped, timer initialized [OK]")

    # Second frame: 2.0s elapsed (< 4.0s threshold) -> still no alert
    anoms = v_analyzer.evaluate_vehicles("CAM_02", [stopped_vehicle_track], zones=dummy_zone, timestamp=t0 + 2.0)
    assert len(anoms) == 0, "Premature alert triggered before threshold"
    print("  -> t = 2.0s: Stoppage continuing, below threshold [OK]")

    # Third frame: 5.0s elapsed (>= 4.0s threshold) -> Alert triggered!
    anoms = v_analyzer.evaluate_vehicles("CAM_02", [stopped_vehicle_track], zones=dummy_zone, timestamp=t0 + 5.0)
    assert len(anoms) == 1, f"Expected 1 stoppage anomaly, got {len(anoms)}"
    anom = anoms[0]
    assert anom["anomaly_type"] == "VEHICLE_STOPPAGE_ALERT"
    assert anom["severity"] == "CRITICAL"  # truck in restricted zone = CRITICAL
    assert anom["stoppage_duration_sec"] >= 4.0
    print(f"  -> t = 5.0s: {anom['anomaly_type']} triggered! Severity={anom['severity']}, Duration={anom['stoppage_duration_sec']}s [OK]")
    print("  --> PASS: Stage 2 Vehicle Stoppage Verified.")


def test_stage_3_vehicle_loitering_and_intelligence():
    print("\n[Stage 3/5] Testing Vehicle Loitering & Intelligence Engine Pipeline...")
    v_analyzer = VehicleBehaviorAnalyzer(
        loitering_threshold_seconds=10.0,
        cooldown_seconds=1.0,
    )

    loitering_car = {
        "track_id": 99,
        "class_name": "car",
        "confidence": 0.92,
        "bbox": [150.0, 150.0, 250.0, 250.0],
        "center": [200.0, 240.0],
        "dwell_seconds": 12.5,  # Exceeds 10.0s threshold
        "trajectory": [(180.0, 240.0), (190.0, 240.0), (200.0, 240.0)],
    }

    anoms = v_analyzer.evaluate_vehicles("CAM_02", [loitering_car], zones=None, timestamp=time.time())
    assert len(anoms) == 1
    assert anoms[0]["anomaly_type"] == "VEHICLE_LOITERING_ALERT"
    print(f"  -> {anoms[0]['anomaly_type']} triggered: Dwell={anoms[0]['dwell_seconds']}s [OK]")

    # Verify IntelligenceEngine integration with synthetic frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sim_tracks = [
        {
            "track_id": 105,
            "class_name": "bus",
            "confidence": 0.96,
            "bbox": [200.0, 150.0, 400.0, 350.0],
            "center": [300.0, 340.0],  # Inside CAM_02 Gate Checkpoint polygon
            "dwell_seconds": 16.0,
            "trajectory": [(300.0, 340.0), (300.2, 340.1)],  # Stopped bus
        }
    ]

    events = intelligence_engine.evaluate_tracks("CAM_02", sim_tracks, dummy_frame)
    assert len(events) > 0, "No events triggered by IntelligenceEngine for vehicle"
    event_types = [e.get("event_type", "") for e in events]
    print(f"  -> IntelligenceEngine generated {len(events)} events: {event_types} [OK]")

    # Verify Threat Scoring with vehicle alerts
    score = incident_manager.calculate_threat_score(events)
    defcon = incident_manager.get_defcon_level(score)
    assert score >= 45, f"Expected elevated threat score (>= 45), got {score}"
    print(f"  -> Threat Score: {score}/100 | Level: {defcon['level']} ({defcon['status']}) [OK]")
    print("  --> PASS: Stage 3 Vehicle Intelligence Pipeline Verified.")


def test_stage_4_intelligence_graph():
    print("\n[Stage 4/5] Testing Surveillance Intelligence Graph & Path Tracing...")
    graph = SurveillanceIntelligenceGraph(storage_path=ROOT_DIR / "data" / "test_intel_graph.json")

    # Add Sightings across 3 cameras for Subject Alpha
    subj_id = "SUBJ_ALPHA_01"
    graph.add_sighting(subj_id, "CAM_01", timestamp=100.0, bbox=[50, 50, 100, 150], class_name="person")
    graph.add_sighting(subj_id, "CAM_01", timestamp=110.0, bbox=[60, 60, 110, 160], class_name="person")
    graph.add_sighting(subj_id, "CAM_02", timestamp=125.0, bbox=[80, 80, 130, 180], class_name="person")
    graph.add_sighting(subj_id, "CAM_03", timestamp=145.0, bbox=[90, 90, 140, 190], class_name="person")

    # Add Transits
    graph.add_transit("CAM_01", "CAM_02", subj_id, transit_sec=15.0, similarity=0.88, timestamp=125.0)
    graph.add_transit("CAM_02", "CAM_03", subj_id, transit_sec=20.0, similarity=0.84, timestamp=145.0)

    # Trace Journey
    journey = graph.trace_journey(subj_id)
    assert len(journey) == 3, f"Expected 3 camera hops in journey, got {len(journey)}"
    assert journey[0]["camera_id"] == "CAM_01"
    assert journey[1]["camera_id"] == "CAM_02"
    assert journey[2]["camera_id"] == "CAM_03"
    assert journey[0]["dwell_duration_sec"] == 10.0
    print(f"  -> Journey Traced: {' -> '.join([j['camera_id'] for j in journey])} [OK]")

    # Export to D3 Node-Link Schema
    d3_data = graph.to_node_link_json()
    assert d3_data["nodes_count"] >= 4  # 3 cameras + 1 entity
    assert d3_data["links_count"] >= 2
    assert "nodes" in d3_data and "links" in d3_data
    print(f"  -> Node-Link Export: {d3_data['nodes_count']} nodes, {d3_data['links_count']} links [OK]")

    # Test Graph Serialization
    save_path = graph.save_graph()
    assert save_path.is_file()
    print(f"  -> Graph persisted to disk: {save_path.name} [OK]")
    print("  --> PASS: Stage 4 Intelligence Graph Verified.")


def test_stage_5_topological_anomalies():
    print("\n[Stage 5/5] Testing Topological Path Anomaly Detection...")
    graph = SurveillanceIntelligenceGraph(storage_path=ROOT_DIR / "data" / "test_intel_graph.json")

    # Anomaly 1: Teleportation / Physical speed violation (0.3s between CAM_01 and CAM_02, min is 1.0s)
    bad_subj_1 = "SUBJ_TELEPORT_99"
    graph.add_transit("CAM_01", "CAM_02", bad_subj_1, transit_sec=0.3, similarity=0.91, timestamp=200.0)
    anoms_1 = graph.detect_path_anomalies(bad_subj_1)
    assert any(a["anomaly_type"] == "TELEPORTATION_VIOLATION" for a in anoms_1)
    print(f"  -> Anomaly 1 Caught: TELEPORTATION_VIOLATION (0.3s < min 1.0s) [OK]")

    # Anomaly 2: Checkpoint Skipping (CAM_01 -> CAM_03 skipping CAM_02)
    bad_subj_2 = "SUBJ_BYPASS_88"
    graph.add_transit("CAM_01", "CAM_03", bad_subj_2, transit_sec=10.0, similarity=0.85, timestamp=300.0)
    anoms_2 = graph.detect_path_anomalies(bad_subj_2)
    assert any(a["anomaly_type"] == "CHECKPOINT_SKIP_BREACH" for a in anoms_2)
    skip_anom = next(a for a in anoms_2 if a["anomaly_type"] == "CHECKPOINT_SKIP_BREACH")
    assert "CAM_02" in skip_anom["skipped_checkpoints"]
    print(f"  -> Anomaly 2 Caught: CHECKPOINT_SKIP_BREACH (Skipped: {skip_anom['skipped_checkpoints']}) [OK]")

    # Anomaly 3: Circular Patrolling / Oscillating Loiter (CAM_01 -> CAM_02 -> CAM_01 -> CAM_02)
    bad_subj_3 = "SUBJ_PATROL_77"
    graph.add_transit("CAM_01", "CAM_02", bad_subj_3, transit_sec=10.0, similarity=0.89, timestamp=400.0)
    graph.add_transit("CAM_02", "CAM_01", bad_subj_3, transit_sec=12.0, similarity=0.87, timestamp=415.0)
    graph.add_transit("CAM_01", "CAM_02", bad_subj_3, transit_sec=11.0, similarity=0.88, timestamp=430.0)
    anoms_3 = graph.detect_path_anomalies(bad_subj_3)
    assert any(a["anomaly_type"] == "CIRCULAR_PATROLLING_LOITER" for a in anoms_3)
    print(f"  -> Anomaly 3 Caught: CIRCULAR_PATROLLING_LOITER [OK]")
    print("  --> PASS: Stage 5 Path Anomalies Verified.")


def run_all_tests():
    print("=" * 75)
    print("PHASE 2: VEHICLE BEHAVIOR, BORDER VECTORS & INTELLIGENCE GRAPH TEST SUITE")
    print("=" * 75)

    test_stage_1_movement_vectoring()
    test_stage_2_vehicle_stoppage()
    test_stage_3_vehicle_loitering_and_intelligence()
    test_stage_4_intelligence_graph()
    test_stage_5_topological_anomalies()

    print("\n" + "=" * 75)
    print("ALL 5 PHASE 2 INTELLIGENCE STAGES PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    try:
        run_all_tests()
        print("\nStatus: VEHICLE & GRAPH INTELLIGENCE SUITE SUCCESSFUL")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nStatus: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
