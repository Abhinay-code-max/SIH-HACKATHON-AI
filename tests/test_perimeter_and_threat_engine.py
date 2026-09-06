"""
Automated Validation Suite for Phase 13, 14 & 15:
Perimeter Zone State Machine, Compound Threat Assessment, Operator Override & Event Timeline.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.perimeter_editor import (
    PerimeterEngine,
    LabPerimeterZone,
    ZoneType,
    ZoneTransition,
)
from training_lab.engine.threat_validator import ThreatValidator
from training_lab.engine.operator_override import OperatorOverrideEngine, EventTimeline


def test_stage_1_perimeter_state_machine():
    print("\n--- STAGE 1: Perimeter State Machine (ENTER, INSIDE, EXIT) ---")
    engine = PerimeterEngine()
    engine.clear_state()

    # Define a polygon zone in normalized/pixel coordinate space [100, 100] to [300, 300]
    zone = LabPerimeterZone(
        zone_id="ZONE_RESTRICTED_ALPHA",
        name="North Perimeter Sector",
        zone_type=ZoneType.RESTRICTED,
        camera_id="CAM_01",
        polygon=[(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)],
        color="#FF0000",
    )
    engine.add_zone(zone)

    # Frame 0: Target outside at (50, 50)
    events_f0 = engine.evaluate_track("CAM_01", track_id=42, class_name="person", ground_point=(50.0, 50.0), frame_idx=0)
    assert len(events_f0) == 0, f"Expected 0 events outside zone, got {len(events_f0)}"

    # Frame 1: Target enters zone at (150, 150) -> ENTER
    events_f1 = engine.evaluate_track("CAM_01", track_id=42, class_name="person", ground_point=(150.0, 150.0), frame_idx=1)
    assert len(events_f1) == 1
    assert events_f1[0].transition == ZoneTransition.ENTER
    assert events_f1[0].zone_id == "ZONE_RESTRICTED_ALPHA"
    print(f"  Emitted: {events_f1[0].transition.value} for track {events_f1[0].track_id} in {events_f1[0].zone_id}")

    # Frame 2: Target remains inside zone at (180, 180) -> INSIDE
    events_f2 = engine.evaluate_track("CAM_01", track_id=42, class_name="person", ground_point=(180.0, 180.0), frame_idx=2)
    assert len(events_f2) == 1
    assert events_f2[0].transition == ZoneTransition.INSIDE
    print(f"  Emitted: {events_f2[0].transition.value} for track {events_f2[0].track_id}")

    # Frame 3: Target exits zone at (350, 350) -> EXIT
    events_f3 = engine.evaluate_track("CAM_01", track_id=42, class_name="person", ground_point=(350.0, 350.0), frame_idx=3)
    assert len(events_f3) == 1
    assert events_f3[0].transition == ZoneTransition.EXIT
    print(f"  Emitted: {events_f3[0].transition.value} for track {events_f3[0].track_id}")

    # Frame 4: Target remains outside at (360, 360) -> No event
    events_f4 = engine.evaluate_track("CAM_01", track_id=42, class_name="person", ground_point=(360.0, 360.0), frame_idx=4)
    assert len(events_f4) == 0
    print("  [PASS] Stage 1: Successfully validated state machine sequence: ENTER -> INSIDE -> EXIT")


def test_stage_2_perimeter_human_validation():
    print("\n--- STAGE 2: Perimeter Decision Human Validation ---")
    engine = PerimeterEngine()

    val_record = engine.validate_perimeter_decision(
        event_id="evt_test_101",
        is_correct=False,
        human_decision="NO_BREACH",
        ai_decision="BREACH",
        notes="False alarm: Target was outside outer thermal sensor fence line.",
    )

    assert val_record.validation_id.startswith("pval_")
    assert val_record.is_correct is False
    assert val_record.human_decision == "NO_BREACH"
    assert val_record.ai_decision == "BREACH"
    assert engine.validations_file.exists()
    print(f"  [PASS] Stage 2: Perimeter human validation recorded and persisted (ID: {val_record.validation_id}, Human: {val_record.human_decision})")


def test_stage_3_compound_threat_assessment():
    print("\n--- STAGE 3: Compound Multi-Factor Threat Assessment ---")
    validator = ThreatValidator()

    # Scenario: person + RESTRICTED_ZONE (40) + moving inward (20) + dwell 10s (20) + baseline (15) = 95 -> CRITICAL (>= 85)
    threat = validator.assess_threat(
        class_name="person",
        zone_type="RESTRICTED",
        is_moving_inward=True,
        dwell_seconds=10.0,
        cluster_count=1,
    )

    assert threat["threat_score"] >= 85
    assert threat["threat_level"] == "CRITICAL"
    assert threat["factors"]["zone_pts"] == 40
    assert threat["factors"]["direction_pts"] == 20
    assert threat["factors"]["dwell_pts"] == 20

    # Test lower threat scenario: vehicle in SAFE zone, moving outward, dwell 2s
    low_threat = validator.assess_threat(
        class_name="car",
        zone_type="SAFE",
        is_moving_inward=False,
        dwell_seconds=2.0,
        cluster_count=1,
    )
    assert low_threat["threat_score"] == 15
    assert low_threat["threat_level"] == "LOW"

    print(f"  [PASS] Stage 3: Compound threat computed correctly (Critical Score={threat['threat_score']}, Level={threat['threat_level']}; Low Score={low_threat['threat_score']}, Level={low_threat['threat_level']})")


def test_stage_4_threat_disagreement_tracking():
    print("\n--- STAGE 4: Threat Validation Disagreement Tracking ---")
    validator = ThreatValidator()

    # Record two validations: one agreement, one disagreement
    val_1 = validator.validate_threat_assessment(
        assessment_id="th_001",
        ai_level="CRITICAL",
        human_level="CRITICAL",
        is_correct=True,
        notes="Confirmed breach with inward weapon carriage.",
    )
    val_2 = validator.validate_threat_assessment(
        assessment_id="th_002",
        ai_level="HIGH",
        human_level="MEDIUM",
        is_correct=False,
        notes="Validator downgraded: friendly patrol unit identified.",
    )

    metrics = validator.compute_threat_validation_metrics()
    assert metrics["total_validations"] >= 2
    assert "confusion_matrix" in metrics
    assert "disagreements" in metrics
    assert any(d["assessment_id"] == "th_002" for d in metrics["disagreements"])
    print(f"  [PASS] Stage 4: Threat validation metrics computed (Total={metrics['total_validations']}, Agreement={metrics['agreement_pct']}%, Disagreements logged={len(metrics['disagreements'])})")


def test_stage_5_operator_override_and_timeline():
    print("\n--- STAGE 5: Operator Override & Event Timeline ---")
    override_engine = OperatorOverrideEngine()
    timeline = EventTimeline()

    # Step 1: Set operator override (CRITICAL -> LOW)
    override = override_engine.set_override(
        automatic_level="CRITICAL",
        override_level="LOW",
        reason="Authorized UN border inspection convoy confirmed via radio dispatch",
        operator="Sector Commander Col. Verma",
    )
    assert override["is_active"] is True
    assert override["override_level"] == "LOW"

    # Step 2: Check effective threat
    effective = override_engine.get_effective_threat(current_ai_level="CRITICAL")
    assert effective["effective_level"] == "LOW"
    assert effective["source"] == "HUMAN_OVERRIDE"

    # Step 3: Clear override
    cleared = override_engine.clear_override()
    assert cleared["source"] == "AI_DECISION"
    effective_post = override_engine.get_effective_threat(current_ai_level="CRITICAL")
    assert effective_post["effective_level"] == "CRITICAL"
    assert effective_post["source"] == "AI_DECISION"

    # Step 4: Log chronological events into unified event timeline
    timeline.clear_timeline()
    timeline.log_event("DETECTION", "CAM_01", "Detected person bbox=[150, 150, 200, 260] conf=0.92")
    timeline.log_event("TRACKING", "CAM_01", "Assigned persistent ByteTrack ID 42")
    timeline.log_event("PERIMETER_ENTRY", "CAM_01", "Track 42 entered ZONE_RESTRICTED_ALPHA")
    timeline.log_event("THREAT_ESCALATION", "CAM_01", "Threat score escalated to 95 (CRITICAL)")
    timeline.log_event("OPERATOR_OVERRIDE", "CAM_01", "Operator Col. Verma downgraded threat to LOW")

    tl_events = timeline.get_timeline(limit=10)
    assert len(tl_events) == 5
    assert tl_events[0]["event_type"] == "DETECTION"
    assert tl_events[2]["event_type"] == "PERIMETER_ENTRY"
    assert tl_events[4]["event_type"] == "OPERATOR_OVERRIDE"

    print(f"  [PASS] Stage 5: Operator override verified ({effective['source']} -> {effective['effective_level']}) & Event Timeline verified ({len(tl_events)} chronological events)")


def run_all_stages():
    print("=" * 70)
    print("BORDER SENTINEL - TRAINING LAB: PHASE 13, 14 & 15 TEST SUITE")
    print("=" * 70)

    test_stage_1_perimeter_state_machine()
    test_stage_2_perimeter_human_validation()
    test_stage_3_compound_threat_assessment()
    test_stage_4_threat_disagreement_tracking()
    test_stage_5_operator_override_and_timeline()

    print("\n" + "=" * 70)
    print("ALL 5 VALIDATION STAGES PASSED SUCCESSFULLY (100% PASS RATE)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_stages()
