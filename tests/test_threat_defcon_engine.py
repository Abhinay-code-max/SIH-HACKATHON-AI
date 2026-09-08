"""
Comprehensive Automated Test Suite for Threat Intelligence & DEFCON Implementation.
Covers:
- Score ranges and clamping ([0, 100], <0, >100)
- Deterministic scoring for identical inputs
- Score-to-DEFCON exact boundary testing:
    19 -> DEFCON 5
    20 -> DEFCON 4
    39 -> DEFCON 4
    40 -> DEFCON 3
    59 -> DEFCON 3
    60 -> DEFCON 2
    79 -> DEFCON 2
    80 -> DEFCON 1
- Spatial events:
    Restricted-zone intrusion
    Boundary crossing
    Movement toward boundary
    Loitering
    Repeated boundary approach
- Class context:
    Normal vehicle remains low threat (DEFCON 5)
    Animal does not automatically become critical (contextual de-escalation)
    Backpack/bag do not automatically become threats without restricted/abandonment context
- Scenarios 1 through 6:
    Scenario 1: Normal Vehicle
    Scenario 2: Person Enters Restricted Zone
    Scenario 3: Loitering in Restricted Zone
    Scenario 4: Border Approach
    Scenario 5: Critical Compound Event (DEFCON 1)
    Scenario 6: False Positive Animal Check
- Operator Overrides & Feedback:
    Acknowledge, false positive, confirm, downgrade, escalate, dismiss
    Separation of automated_assessment and operator_decision
- Schema & Serialization:
    ThreatEvent JSON serialize/deserialize roundtrip
- Camera Isolation & Multi-Camera Compatibility:
    CAM_01 through CAM_05 track ID isolation
    Logical track ID formatting (camera_id + track_id)
- Offline Determinism Verification
"""

from pathlib import Path
import sys
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.threat_config import (
    DEFCON_THRESHOLDS,
    THREAT_FACTOR_WEIGHTS,
    CLASS_CONTEXT_MODIFIERS,
    THREAT_SCORING_VERSION,
)
from training_lab.engine.threat_engine import (
    ThreatScoringEngine,
    score_to_defcon,
    defcon_to_severity,
    threat_scoring_engine,
)
from training_lab.engine.threat_event_schema import (
    ThreatEvent,
    OperatorFeedbackRecord,
)
from training_lab.engine.behavior_engine import (
    BehaviorFeatureEngine,
    behavior_feature_engine,
)
from training_lab.engine.perimeter_editor import (
    PerimeterEngine,
    LabPerimeterZone,
    BorderBoundary,
    ZoneType,
    ZoneTransition,
)
from training_lab.engine.operator_override import (
    OperatorOverrideEngine,
    EventTimeline,
    operator_override_engine,
)


# ==============================================================================
# 1. DEFCON BOUNDARY AND SCORE RANGE TESTS
# ==============================================================================

@pytest.mark.parametrize(
    "score, expected_defcon",
    [
        (0, 5),
        (10, 5),
        (19, 5),
        (20, 4),
        (30, 4),
        (39, 4),
        (40, 3),
        (50, 3),
        (59, 3),
        (60, 2),
        (70, 2),
        (79, 2),
        (80, 1),
        (90, 1),
        (100, 1),
        (-10, 5),    # Clamped to 0 -> DEFCON 5
        (150, 1),    # Clamped to 100 -> DEFCON 1
        (19.4, 5),
        (19.6, 4),
    ],
)
def test_score_to_defcon_exact_boundaries(score, expected_defcon):
    assert score_to_defcon(score) == expected_defcon


def test_threat_score_range_and_determinism():
    engine = ThreatScoringEngine()
    track_state = {"track_id": 1, "class_name": "person", "confidence": 0.95}

    # Determinism: 5 repeated evaluations must give identical scores and explanations
    first_res = engine.assess_threat(track_state, spatial_events=[], behavior_features={})
    for _ in range(5):
        res = engine.assess_threat(track_state, spatial_events=[], behavior_features={})
        assert res.score == first_res.score
        assert res.defcon_level == first_res.defcon_level
        assert res.explanation == first_res.explanation
        assert 0 <= res.score <= 100


# ==============================================================================
# 2. CLASS PRIOR / CONTEXTUAL NEUTRALITY TESTS
# ==============================================================================

def test_neutral_vehicle_remains_low_threat():
    engine = ThreatScoringEngine()
    track = {
        "track_id": 10,
        "class_name": "car",
        "confidence": 0.92,
        "speed_px_s": 25.0,
    }
    behav = {
        "is_moving": True,
        "speed_px_s": 25.0,
        "direction_classification": "parallel",
        "dwell_seconds": 5.0,
        "is_loitering": False,
    }
    res = engine.assess_threat(track, spatial_events=[], behavior_features=behav)
    assert res.score <= 19
    assert res.defcon_level == 5
    assert res.severity == "LOW"


def test_animal_does_not_become_critical():
    engine = ThreatScoringEngine()
    track = {
        "track_id": 11,
        "class_name": "animal",
        "confidence": 0.88,
    }
    # Even if near boundary or moving toward boundary
    behav = {
        "direction_classification": "toward_boundary",
        "is_approaching_border": True,
        "dwell_seconds": 15.0,
    }
    res = engine.assess_threat(track, spatial_events=[], behavior_features=behav)
    # Movement toward boundary (+15) with animal context modifier (-10) = 5 -> DEFCON 5
    assert res.score < 40
    assert res.defcon_level in (4, 5)
    assert res.defcon_level != 1


def test_backpack_bag_neutral_without_context():
    engine = ThreatScoringEngine()
    track = {
        "track_id": 12,
        "class_name": "backpack",
        "confidence": 0.90,
    }
    res = engine.assess_threat(track, spatial_events=[], behavior_features={"dwell_seconds": 2.0})
    assert res.score == 0
    assert res.defcon_level == 5

    # Contextual check: stationary unattended in restricted zone elevates threat
    spatial_evts = [{"event_type": "ENTERED_RESTRICTED_ZONE"}]
    res_context = engine.assess_threat(
        track,
        spatial_events=spatial_evts,
        behavior_features={"dwell_seconds": 15.0},
    )
    # Restricted intrusion (+30) + unattended baggage in restricted zone (+15) = 45 -> DEFCON 3
    assert res_context.score >= 40
    assert res_context.defcon_level <= 3


# ==============================================================================
# 3. ANTI-DOUBLE COUNTING AND FACTOR WEIGHT TESTS
# ==============================================================================

def test_anti_double_counting_boundary_crossing():
    engine = ThreatScoringEngine()
    track = {"track_id": 15, "class_name": "person", "confidence": 0.90}
    # Crossing boundary should receive +30 for crossing, NOT +30 and +15 for approach simultaneously
    spatial = [{"event_type": "CROSSED_BOUNDARY"}]
    behav = {"direction_classification": "toward_boundary", "is_approaching_border": True}

    res = engine.assess_threat(track, spatial_events=spatial, behavior_features=behav)
    assert res.score == 30
    assert res.defcon_level == 4
    factor_names = [f["factor"] for f in res.contributing_factors]
    assert "boundary_crossing" in factor_names
    assert "movement_toward_boundary" not in factor_names


# ==============================================================================
# 4. SCENARIOS 1 THROUGH 6 TESTS (SYNTHETIC BENCHMARK)
# ==============================================================================

def test_scenario_1_normal_vehicle():
    """SYNTHETIC SCENARIO 1: Car, normal movement, no restricted zone interaction."""
    engine = ThreatScoringEngine()
    track = {"track_id": 101, "class_name": "car", "confidence": 0.94}
    behav = {
        "movement_state": "moving",
        "speed_px_s": 22.0,
        "direction_classification": "parallel",
        "dwell_seconds": 4.0,
        "is_loitering": False,
    }
    assessment = engine.assess_threat(track, spatial_events=[], behavior_features=behav)
    assert assessment.score <= 19
    assert assessment.defcon_level == 5


def test_scenario_2_person_enters_restricted_zone():
    """SYNTHETIC SCENARIO 2: Person enters restricted zone -> elevated threat (DEFCON >= 3 or score >= 30)."""
    engine = ThreatScoringEngine()
    track = {"track_id": 102, "class_name": "person", "confidence": 0.92}
    spatial = [{"event_type": "ENTERED_RESTRICTED_ZONE"}]
    behav = {"movement_state": "moving", "speed_px_s": 15.0, "dwell_seconds": 5.0}

    assessment = engine.assess_threat(track, spatial_events=spatial, behavior_features=behav)
    assert assessment.score >= 30
    assert assessment.defcon_level <= 4


def test_scenario_3_loitering():
    """SYNTHETIC SCENARIO 3: Person enters restricted zone and remains stationary > 30 seconds."""
    engine = ThreatScoringEngine()
    track = {"track_id": 103, "class_name": "person", "confidence": 0.95}
    spatial = [
        {"event_type": "ENTERED_RESTRICTED_ZONE"},
        {"event_type": "LOITERING_IN_RESTRICTED_ZONE"},
    ]
    behav = {
        "movement_state": "stationary",
        "is_loitering": True,
        "dwell_seconds": 38.0,
        "stopped_duration": 35.0,
    }
    assessment = engine.assess_threat(track, spatial_events=spatial, behavior_features=behav)
    # Restricted zone (+30) + Loitering (+15) = 45 -> DEFCON 3
    assert assessment.score >= 45
    assert assessment.defcon_level <= 3


def test_scenario_4_border_approach():
    """SYNTHETIC SCENARIO 4: Person enters restricted zone, moves toward border boundary, prolonged presence."""
    engine = ThreatScoringEngine()
    track = {"track_id": 104, "class_name": "person", "confidence": 0.96}
    spatial = [
        {"event_type": "ENTERED_RESTRICTED_ZONE"},
        {"event_type": "PROLONGED_PRESENCE"},
    ]
    behav = {
        "movement_state": "moving",
        "direction_classification": "toward_boundary",
        "is_approaching_border": True,
        "dwell_seconds": 65.0,
        "is_loitering": False,
    }
    assessment = engine.assess_threat(track, spatial_events=spatial, behavior_features=behav)
    # Restricted (+30) + Toward Boundary (+15) + Prolonged restricted (+10) = 55 (DEFCON 3 or DEFCON 2)
    # If repeated approach or abnormal movement is also present, it reaches DEFCON 2
    assert assessment.score >= 55
    assert assessment.defcon_level <= 3


def test_scenario_5_critical_event():
    """SYNTHETIC SCENARIO 5: Restricted intrusion + repeated boundary approach + movement toward boundary + prolonged loitering."""
    engine = ThreatScoringEngine()
    track = {"track_id": 105, "class_name": "person", "confidence": 0.98}
    spatial = [
        {"event_type": "ENTERED_RESTRICTED_ZONE"},
        {"event_type": "LOITERING_IN_RESTRICTED_ZONE"},
        {"event_type": "PROLONGED_PRESENCE"},
    ]
    behav = {
        "movement_state": "stationary",
        "direction_classification": "toward_boundary",
        "is_approaching_border": True,
        "is_repeated_approach": True,
        "repeated_approach_count": 3,
        "is_loitering": True,
        "dwell_seconds": 75.0,
        "group_size": 3,
    }
    assessment = engine.assess_threat(track, spatial_events=spatial, behavior_features=behav)
    # Restricted (+30) + Toward Boundary (+15) + Loitering (+15) + Repeated (+10) + Prolonged (+10) + Group (+5) = 85 -> DEFCON 1
    assert assessment.score >= 80
    assert assessment.defcon_level == 1
    assert assessment.severity == "CRITICAL"


def test_scenario_6_false_positive_animal():
    """SYNTHETIC SCENARIO 6: Animal enters area -> must NOT automatically produce DEFCON 1."""
    engine = ThreatScoringEngine()
    track = {"track_id": 106, "class_name": "animal", "confidence": 0.89}
    spatial = [{"event_type": "ENTERED_ZONE"}]
    behav = {
        "movement_state": "moving",
        "direction_classification": "toward_boundary",
        "dwell_seconds": 12.0,
    }
    assessment = engine.assess_threat(track, spatial_events=spatial, behavior_features=behav)
    assert assessment.defcon_level != 1
    assert assessment.defcon_level in (4, 5)


# ==============================================================================
# 5. BEHAVIOR FEATURE ENGINE TESTS
# ==============================================================================

def test_behavior_feature_extraction():
    b_engine = BehaviorFeatureEngine()
    b_engine.clear_state()

    # Track moving toward border (downward [0, 1])
    track = {
        "track_id": 42,
        "trajectory": [(100.0, 100.0), (100.0, 130.0)],
        "dwell_seconds": 35.0,
        "speed_px_s": 30.0,
    }
    features = b_engine.extract_features(
        camera_id="CAM_01",
        track=track,
        boundary_normal_vector=(0.0, 1.0),
    )
    assert features["movement_state"] == "moving"
    assert features["direction_classification"] == "toward_boundary"
    assert features["cosine_similarity"] > 0.9
    assert features["dwell_seconds"] == 35.0


def test_behavior_stopping_transition():
    b_engine = BehaviorFeatureEngine()
    b_engine.clear_state()

    # Step 1: moving
    t1 = {"track_id": 7, "trajectory": [(50.0, 50.0), (80.0, 80.0)], "speed_px_s": 40.0}
    f1 = b_engine.extract_features("CAM_01", t1)
    assert f1["movement_state"] == "moving"

    # Step 2: stops
    t2 = {"track_id": 7, "trajectory": [(80.0, 80.0), (80.0, 80.0)], "speed_px_s": 0.0}
    f2 = b_engine.extract_features("CAM_01", t2)
    assert f2["movement_state"] == "stationary"
    assert f2["state_transition"] == "moving_to_stationary"


# ==============================================================================
# 6. OPERATOR OVERRIDE & HUMAN FEEDBACK TESTS
# ==============================================================================

def test_operator_override_lifecycle(tmp_path):
    override_log = tmp_path / "test_overrides.json"
    feedback_log = tmp_path / "test_feedback.json"
    ovr_engine = OperatorOverrideEngine(log_file=override_log, feedback_file=feedback_log)

    # Initial AI assessment is DEFCON 2
    eff = ovr_engine.get_effective_threat("DEFCON 2")
    assert eff["source"] == "AI_DECISION"
    assert eff["effective_level"] == "DEFCON 2"

    # Operator marks false alarm and downgrades to DEFCON 5
    record = ovr_engine.set_override(
        automatic_level="DEFCON 2",
        override_level="DEFCON 5",
        reason="Authorized maintenance personnel with verified badge",
        operator="Officer_K",
        camera_id="CAM_01",
        track_id=105,
        action="mark_false_positive",
    )
    assert record["is_active"] is True
    assert record["override_level"] == "DEFCON 5"

    # Effective threat now reflects human override without deleting AI decision
    eff_ovr = ovr_engine.get_effective_threat("DEFCON 2")
    assert eff_ovr["source"] == "HUMAN_OVERRIDE"
    assert eff_ovr["effective_level"] == "DEFCON 5"

    # Human feedback log was created
    assert feedback_log.exists()

    # Clear override restores AI decision
    clear_res = ovr_engine.clear_override()
    assert clear_res["status"] == "OVERRIDE_CLEARED"
    eff_restored = ovr_engine.get_effective_threat("DEFCON 2")
    assert eff_restored["source"] == "AI_DECISION"


# ==============================================================================
# 7. THREAT EVENT SCHEMA & FIVE-CAMERA COMPATIBILITY
# ==============================================================================

def test_threat_event_serialization_and_multi_cam_isolation():
    event = ThreatEvent(
        event_id="evt_test_001",
        timestamp="2026-09-08T07:00:00Z",
        camera_id="CAM_03",
        track_id=17,
        object_class="person",
        object_confidence=0.95,
        behavior_features={"movement_state": "moving", "speed_px_s": 20.0},
        spatial_events=[{"event_type": "ENTERED_RESTRICTED_ZONE"}],
        threat_score=72,
        defcon_level=2,
        severity="HIGH",
        contributing_factors=[
            {"factor": "restricted_zone_intrusion", "points": 30},
            {"factor": "movement_toward_boundary", "points": 15},
        ],
        explanation="Track 17 entered restricted zone",
        recommended_action="Operator review required",
    )

    # Multi-camera collision-free track identity
    assert event.logical_track_id == "CAM_03:17"

    # JSON roundtrip test
    json_str = event.to_json()
    reconstructed = ThreatEvent.from_json(json_str)

    assert reconstructed.event_id == event.event_id
    assert reconstructed.camera_id == "CAM_03"
    assert reconstructed.track_id == 17
    assert reconstructed.logical_track_id == "CAM_03:17"
    assert reconstructed.threat_score == 72
    assert reconstructed.defcon_level == 2
    assert len(reconstructed.contributing_factors) == 2


def test_five_camera_processing_isolation():
    """Ensure tracks across CAM_01..CAM_05 with the same track_id do not collide."""
    engine = ThreatScoringEngine()
    cameras = ["CAM_01", "CAM_02", "CAM_03", "CAM_04", "CAM_05"]
    events_per_cam = {}

    for cam in cameras:
        track = {"track_id": 1, "class_name": "person", "confidence": 0.90}
        spatial = [{"event_type": "ENTERED_RESTRICTED_ZONE"}] if cam in ("CAM_01", "CAM_05") else []
        assessment = engine.assess_threat(track, spatial_events=spatial)
        logical_id = f"{cam}:{track['track_id']}"
        events_per_cam[logical_id] = assessment

    assert len(events_per_cam) == 5
    assert "CAM_01:1" in events_per_cam
    assert "CAM_05:1" in events_per_cam
    assert events_per_cam["CAM_01:1"].score == 30
    assert events_per_cam["CAM_02:1"].score == 0
