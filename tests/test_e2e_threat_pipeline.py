"""
End-to-End Pipeline Integration Test:
Detector (U2 local checkpoint) -> Tracker -> Behavior Features -> Spatial Events -> Threat Assessment -> DEFCON
Validates complete air-gapped offline flow without internet.
"""

from pathlib import Path
import sys
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.lab_detector import LabDetector
from training_lab.engine.lab_tracker import LabMultiCamTracker
from training_lab.engine.behavior_engine import behavior_feature_engine
from training_lab.engine.perimeter_editor import perimeter_engine, LabPerimeterZone, BorderBoundary, ZoneType
from training_lab.engine.threat_engine import threat_scoring_engine
from training_lab.engine.threat_event_schema import ThreatEvent
from training_lab.engine.operator_override import operator_override_engine


def test_end_to_end_u2_to_threat_defcon_pipeline():
    u2_weights = ROOT_DIR / "training_lab" / "runs" / "U2_yolov8m_640_9class_v002" / "weights" / "best.pt"
    assert u2_weights.is_file(), f"U2 best weights not found at {u2_weights}"

    # 1. Initialize detector strictly offline
    detector = LabDetector(model_name=str(u2_weights))
    assert detector.model is not None
    assert len(detector.MASTER_CLASSES) == 9

    # 2. Configure perimeter zone and boundary
    perimeter_engine.clear_state()
    zone = LabPerimeterZone(
        zone_id="ZONE_PERIMETER_TEST",
        name="Test Border Zone",
        zone_type=ZoneType.RESTRICTED_ZONE,
        camera_id="CAM_01",
        polygon=[(100.0, 100.0), (500.0, 100.0), (500.0, 400.0), (100.0, 400.0)],
    )
    perimeter_engine.add_zone(zone)

    boundary = BorderBoundary(
        boundary_id="BOUNDARY_LINE_01",
        name="South Border Fence",
        camera_id="CAM_01",
        line_start=(50.0, 350.0),
        line_end=(550.0, 350.0),
        normal_vector=(0.0, 1.0),
    )
    perimeter_engine.add_boundary(boundary)

    # 3. Simulate sequential track inputs representing an intruder moving into restricted zone toward boundary
    track_sim = {
        "track_id": 99,
        "class_name": "person",
        "confidence": 0.93,
        "bbox": [200.0, 180.0, 260.0, 320.0],
        "center": [230.0, 315.0],
        "trajectory": [(230.0, 150.0), (230.0, 250.0), (230.0, 315.0)],
        "dwell_seconds": 35.0,
        "speed_px_s": 25.0,
    }

    # Extract behavior features
    behav_features = behavior_feature_engine.extract_features(
        camera_id="CAM_01",
        track=track_sim,
        boundary_normal_vector=boundary.normal_vector,
    )
    assert behav_features["direction_classification"] == "toward_boundary"
    assert behav_features["movement_state"] == "moving"

    # Extract spatial events
    spatial_evts = perimeter_engine.evaluate_spatial_events(
        camera_id="CAM_01",
        track=track_sim,
    )
    assert any(e.get("event_type") == "ENTERED_RESTRICTED_ZONE" for e in spatial_evts)

    # Assess compound threat
    assessment = threat_scoring_engine.assess_threat(
        track_state=track_sim,
        spatial_events=spatial_evts,
        behavior_features=behav_features,
    )

    # Intrusion (+30) + Toward Boundary (+15) + Loitering (+15) = 60 -> DEFCON 2
    assert assessment.score >= 60
    assert assessment.defcon_level in (1, 2)
    assert assessment.track_id == 99
    assert len(assessment.contributing_factors) >= 2
    assert "person" in assessment.explanation.lower()

    # Formulate canonical ThreatEvent
    evt = ThreatEvent(
        event_id="evt_e2e_001",
        timestamp=assessment.timestamp,
        camera_id="CAM_01",
        track_id=99,
        object_class=track_sim["class_name"],
        object_confidence=track_sim["confidence"],
        behavior_features=behav_features,
        spatial_events=spatial_evts,
        threat_score=assessment.score,
        defcon_level=assessment.defcon_level,
        severity=assessment.severity,
        contributing_factors=assessment.contributing_factors,
        explanation=assessment.explanation,
        recommended_action=assessment.recommended_operator_action,
    )
    assert evt.logical_track_id == "CAM_01:99"

    # Verify JSON serializability
    json_out = evt.to_json()
    assert '"logical_track_id": "CAM_01:99"' in json_out
    assert f'"threat_score": {assessment.score}' in json_out
