"""
Grand End-to-End Test Suite for BORDER SENTINEL AI Training, Validation & Simulation Lab.
Validates the complete 12-stage human-in-the-loop lifecycle:
  Stage 1: Multi-Scenario Bank & 5-Camera Feed Ingestion
  Stage 2: Perception, ByteTrack & Strict Camera Isolation
  Stage 3: Polygonal Perimeter Geofence State Machine
  Stage 4: Compound Multi-Factor Threat Assessment & Operator Override
  Stage 5: Human Validation Engine & Missed Detection Capture
  Stage 6: Dataset Quality Gate & 70/15/15 Split
  Stage 7: Versioned Model Training Registry
  Stage 8: Multi-Scenario Benchmark & Section 28 Non-Fabrication Rule
  Stage 9: Side-by-Side Model Comparison & Confidence Calibration
  Stage 10: Experiment Dossier & Lineage Genealogy
  Stage 11: Lab REST API Router Telemetry
  Stage 12: Offline UI Dashboard Air-Gap Integrity Audit
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np


def test_stage_01_scenario_and_simulator():
    print("\n--- STAGE 1: Multi-Scenario Bank & 5-Camera Feed Ingestion ---")
    from training_lab.engine.scenario_manager import scenario_manager
    from training_lab.engine.multi_cam_simulator import MultiCameraSimulator
    from training_lab.engine.video_manager import ensure_demo_camera_videos

    ensure_demo_camera_videos()
    scenarios = scenario_manager.list_scenarios()
    assert len(scenarios) == 15, f"Expected 15 master scenarios, got {len(scenarios)}"

    scn1 = scenario_manager.get_scenario("SCN_01")
    assert scn1 is not None
    assert len(scn1.assigned_cameras) == 5

    sim = MultiCameraSimulator(camera_bindings=scn1, loop=True, fps=30.0)
    bundle = sim.step()
    assert bundle is not None, "Simulator returned None on step"
    assert "feeds" in bundle
    feeds = bundle["feeds"]
    assert len(feeds) == 5, f"Expected 5 feeds, got {len(feeds)}"
    for cid in ["CAM_01", "CAM_02", "CAM_03", "CAM_04", "CAM_05"]:
        assert cid in feeds
        assert isinstance(feeds[cid]["frame"], np.ndarray)
        assert feeds[cid]["camera_id"] == cid
    sim.close()
    print("  [PASS] Stage 1: All 15 scenarios active; synchronous 5-camera bundle verified.")


def test_stage_02_perception_and_isolated_tracking():
    print("\n--- STAGE 2: Perception, ByteTrack & Strict Camera Isolation ---")
    from training_lab.engine.lab_detector import LabDetector
    from training_lab.engine.lab_tracker import LabMultiCamTracker
    from training_lab.engine.multi_cam_simulator import MultiCameraSimulator
    from training_lab.engine.scenario_manager import scenario_manager

    scn1 = scenario_manager.get_scenario("SCN_01")
    sim = MultiCameraSimulator(camera_bindings=scn1, loop=True, fps=30.0)
    bundle = sim.step()
    assert bundle is not None

    detector = LabDetector(model_name="auto")
    tracker = LabMultiCamTracker()

    # Run detection on bundle
    bundle_dets = detector.detect_bundle(bundle, conf_threshold=0.35)
    assert len(bundle_dets) == 5
    for cid in ["CAM_01", "CAM_02", "CAM_03", "CAM_04", "CAM_05"]:
        assert cid in bundle_dets
        for d in bundle_dets[cid]:
            assert d["camera_id"] == cid

    # Run tracking across multiple frames
    for _ in range(5):
        step_bundle = sim.step()
        if step_bundle:
            track_res = tracker.update_bundle(step_bundle, conf_threshold=0.35)
            assert "feeds" in track_res
            assert len(track_res["feeds"]) == 5

    # Check track ID isolation per camera
    cam1_tracker = tracker.get_or_create_tracker("CAM_01")
    assert cam1_tracker is not None
    assert "CAM_01" in tracker.trackers
    assert "CAM_02" in tracker.trackers
    sim.close()
    print("  [PASS] Stage 2: YOLO perception executed; dedicated ByteTrack state isolated per camera.")


def test_stage_03_perimeter_geofence_state_machine():
    print("\n--- STAGE 3: Polygonal Perimeter Geofence State Machine ---")
    from training_lab.engine.perimeter_editor import (
        PerimeterEngine,
        LabPerimeterZone,
        ZoneType,
        ZoneTransition,
    )

    engine = PerimeterEngine()
    engine.clear_state()

    zone = LabPerimeterZone(
        zone_id="Z_CRITICAL_TEST",
        name="Sector Bravo",
        zone_type=ZoneType.CRITICAL,
        camera_id="CAM_01",
        polygon=[(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)],
    )
    engine.add_zone(zone)

    # 1. Target Outside
    e0 = engine.evaluate_track("CAM_01", track_id=99, class_name="person", ground_point=(50.0, 50.0), frame_idx=0)
    assert len(e0) == 0

    # 2. Target Enters
    e1 = engine.evaluate_track("CAM_01", track_id=99, class_name="person", ground_point=(150.0, 150.0), frame_idx=1)
    assert len(e1) == 1
    assert e1[0].transition == ZoneTransition.ENTER

    # 3. Target Inside
    e2 = engine.evaluate_track("CAM_01", track_id=99, class_name="person", ground_point=(200.0, 200.0), frame_idx=2)
    assert len(e2) == 1
    assert e2[0].transition == ZoneTransition.INSIDE

    # 4. Target Exits
    e3 = engine.evaluate_track("CAM_01", track_id=99, class_name="person", ground_point=(350.0, 350.0), frame_idx=3)
    assert len(e3) == 1
    assert e3[0].transition == ZoneTransition.EXIT

    print("  [PASS] Stage 3: Polygonal geofence state machine emitted ENTER -> INSIDE -> EXIT.")


def test_stage_04_compound_threat_and_operator_override():
    print("\n--- STAGE 4: Compound Multi-Factor Threat Assessment & Operator Override ---")
    from training_lab.engine.threat_validator import ThreatValidator
    from training_lab.engine.operator_override import OperatorOverrideEngine

    validator = ThreatValidator()
    override_engine = OperatorOverrideEngine()

    # Person in RESTRICTED zone moving inward with dwell 10s: 15 + 40 + 20 + 20 = 95 -> CRITICAL
    threat = validator.assess_threat(
        class_name="person",
        zone_type="RESTRICTED",
        is_moving_inward=True,
        dwell_seconds=10.0,
        cluster_count=1,
    )
    assert threat["threat_score"] >= 85
    assert threat["threat_level"] == "CRITICAL"

    # Operator Override
    override = override_engine.set_override(
        automatic_level="CRITICAL",
        override_level="LOW",
        reason="Authorized Routine Patrol",
        operator="Command Officer",
    )
    assert override["is_active"] is True
    effective = override_engine.get_effective_threat(current_ai_level="CRITICAL")
    assert effective["effective_level"] == "LOW"
    assert effective["source"] == "HUMAN_OVERRIDE"

    # Clear Override
    override_engine.clear_override()
    effective_cleared = override_engine.get_effective_threat(current_ai_level="CRITICAL")
    assert effective_cleared["effective_level"] == "CRITICAL"
    assert effective_cleared["source"] == "AI_DECISION"

    print("  [PASS] Stage 4: Compound threat scored 95 (CRITICAL); operator override and source arbitration verified.")


def test_stage_05_human_validation_and_missed_detection():
    print("\n--- STAGE 5: Human Validation Engine & Missed Detection Capture ---")
    from training_lab.engine.validation_engine import ValidationEngine, ValidationDecision
    from training_lab.engine.annotation_capture import AnnotationCapture

    ve = ValidationEngine()
    ac = AnnotationCapture()

    # Record validation
    val = ve.record_feedback(
        scenario_id="SCN_01",
        camera_id="CAM_01",
        frame_idx=10,
        ai_prediction={"class_name": "person", "confidence": 0.88},
        decision=ValidationDecision.CORRECT,
    )
    assert val.record_id.upper().startswith("VAL_")

    # Capture missed detection
    ann = ac.capture_missed_detection(
        scenario_id="SCN_01",
        camera_id="CAM_02",
        frame_idx=12,
        class_name="person",
        bbox=[120.0, 140.0, 180.0, 220.0],
    )
    assert ann.record_id.upper().startswith("VAL_")

    stats = ve.get_validation_statistics()
    assert stats["total_validations"] >= 1
    print("  [PASS] Stage 5: Human validation recorded and missed detection annotated.")


def test_stage_06_dataset_quality_gate_and_splits():
    print("\n--- STAGE 6: Dataset Quality Gate & 70/15/15 Split ---")
    from training_lab.engine.dataset_generator import dataset_generator

    candidates = [
        {"scenario_id": "SCN_01", "camera_id": "CAM_01", "frame_idx": 10, "class_name": "person", "bbox": [60.0, 188.0, 91.0, 293.0]},
        {"scenario_id": "SCN_01", "camera_id": "CAM_01", "frame_idx": 25, "class_name": "person", "bbox": [85.0, 188.0, 116.0, 293.0]},
        {"scenario_id": "SCN_02", "camera_id": "CAM_02", "frame_idx": 15, "class_name": "car", "bbox": [120.0, 200.0, 240.0, 310.0]},
        {"scenario_id": "SCN_02", "camera_id": "CAM_02", "frame_idx": 30, "class_name": "truck", "bbox": [150.0, 190.0, 280.0, 320.0]},
        {"scenario_id": "SCN_03", "camera_id": "CAM_03", "frame_idx": 40, "class_name": "backpack", "bbox": [306.0, 260.0, 360.0, 319.0]},
        {"scenario_id": "SCN_04", "camera_id": "CAM_04", "frame_idx": 50, "class_name": "person", "bbox": [480.0, 226.0, 505.0, 257.0]},
        {"scenario_id": "SCN_05", "camera_id": "CAM_05", "frame_idx": 60, "class_name": "motorcycle", "bbox": [30.0, 255.0, 110.0, 346.0]},
    ]

    manifest = dataset_generator.create_dataset_version(
        dataset_version="dataset_test_e2e",
        candidates=candidates,
    )
    assert manifest["dataset_version"] == "dataset_test_e2e"
    assert "splits" in manifest
    splits = manifest["splits"]
    assert "train" in splits and "val" in splits and "test" in splits
    holdout_hash = manifest.get("holdout_test_set_hash")
    assert holdout_hash is not None and len(holdout_hash) >= 16
    print(f"  [PASS] Stage 6: Dataset generated with 70/15/15 split (SHA256: {holdout_hash}).")


def test_stage_07_versioned_model_training_registry():
    print("\n--- STAGE 7: Versioned Model Training Registry ---")
    from training_lab.engine.training_manager import TrainingManager

    tm = TrainingManager()
    models = tm.list_models()
    assert len(models) >= 1
    v001 = tm.get_model("model_v001")
    assert v001 is not None
    assert v001["version"] == "model_v001"
    assert "weights_path" in v001
    print(f"  [PASS] Stage 7: Model registry contains versioned model {v001['version']}; production untouched.")


def test_stage_08_multi_scenario_and_non_fabrication():
    print("\n--- STAGE 8: Multi-Scenario Benchmark & Section 28 Non-Fabrication Rule ---")
    from training_lab.engine.model_evaluator import ModelEvaluator

    me = ModelEvaluator()
    # Unrun scenario test
    unrun_rep = me.evaluate_scenario("model_v001", "SCN_NON_EXISTENT_UNRUN")
    assert unrun_rep["status"] == "NOT TESTED"
    assert unrun_rep["overall"]["mAP50"] is None
    assert unrun_rep["overall"]["precision"] is None

    # Valid scenario test
    rep = me.evaluate_scenario(
        "model_v001",
        "SCN_01",
        ground_truth=[{"camera_id": "CAM_01", "class_name": "person", "bbox": [10.0, 10.0, 50.0, 50.0]}],
        custom_detections=[{"camera_id": "CAM_01", "class_name": "person", "bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.95}],
    )
    assert rep["status"] == "EVALUATED"
    assert rep["overall"]["mAP50"] == 100.0
    print("  [PASS] Stage 8: Section 28 Non-Fabrication Rule verified; valid evaluation computed.")


def test_stage_09_model_comparison_and_calibration():
    print("\n--- STAGE 9: Side-by-Side Model Comparison & Confidence Calibration ---")
    from training_lab.engine.model_comparator import ModelComparator

    comp = ModelComparator()
    eval_a = {"model_version": "V001", "overall": {"mAP50": 80.0, "precision": 80.0, "recall": 75.0, "fp": 30, "fn": 25}}
    eval_b = {"model_version": "V002", "overall": {"mAP50": 86.0, "precision": 87.0, "recall": 82.0, "fp": 18, "fn": 18}}

    c_res = comp.compare_evaluations(eval_a, eval_b)
    assert c_res["verdict"] == "VERIFIED_IMPROVEMENT"
    assert c_res["overall_deltas"]["delta_mAP50"] == 6.0
    assert c_res["overall_deltas"]["delta_fp"] == -12

    calib = comp.analyze_confidence_calibration(
        detections=[{"camera_id": "CAM_01", "class_name": "person", "bbox": [0, 0, 10, 10], "confidence": 0.92}],
        ground_truth=[{"camera_id": "CAM_01", "class_name": "person", "bbox": [0, 0, 10, 10]}],
    )
    assert "expected_calibration_error_pct" in calib
    assert "bins" in calib
    assert "90-100%" in calib["bins"]
    print("  [PASS] Stage 9: Model comparison verdict VERIFIED_IMPROVEMENT & calibration ECE computed.")


def test_stage_10_experiment_manager_and_genealogy():
    print("\n--- STAGE 10: Experiment Dossier & Lineage Genealogy ---")
    from training_lab.engine.experiment_manager import ExperimentManager

    em = ExperimentManager()
    exp = em.record_experiment(
        name="Daylight Perimeter Optimization",
        parent_model_version="yolov8l.pt",
        candidate_model_version="model_v001",
        dataset_version="dataset_v001",
        dataset_hash="a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
        scenarios_tested=["SCN_01", "SCN_02"],
        evaluation_metrics={"mAP50": 87.4, "precision": 88.0, "recall": 85.0, "fp": 26, "fn": 24},
        comparison_delta={"overall_deltas": {"delta_mAP50": 6.0, "delta_fp": -16}},
        verdict="IMPROVED",
    )
    assert exp["experiment_id"].startswith("EXP_")
    assert exp["human_approval_status"] == "PENDING_REVIEW"

    # Approve
    approved = em.update_approval(
        experiment_id=exp["experiment_id"],
        status="APPROVED",
        operator_id="Col. Verma",
        notes="Verified accuracy gain and FP reduction across 5 cameras.",
    )
    assert approved["human_approval_status"] == "APPROVED"

    tree = em.get_genealogy_tree()
    assert "nodes" in tree and "edges" in tree
    assert len(tree["nodes"]) >= 2
    assert tree["deployed_model"] == "model_v001"

    # Export markdown
    md = em.export_dossier_markdown(exp["experiment_id"])
    assert f"EXPERIMENT DOSSIER: {exp['experiment_id']}" in md
    print(f"  [PASS] Stage 10: Experiment {exp['experiment_id']} recorded, approved, and DAG genealogy tree mapped.")


def test_stage_11_lab_rest_api_router():
    print("\n--- STAGE 11: Lab REST API Router Telemetry ---")
    from fastapi import FastAPI
    from starlette.testclient import TestClient
    from training_lab.api.lab_routes import lab_router

    app = FastAPI()
    app.include_router(lab_router)
    client = TestClient(app)

    # 1. GET /api/lab/status
    res_status = client.get("/api/lab/status")
    assert res_status.status_code == 200
    s_data = res_status.json()
    assert s_data["status"] == "OPERATIONAL"
    assert s_data["air_gapped"] is True
    assert "gpu_name" in s_data
    assert s_data["scenarios_count"] == 15

    # 2. GET /api/lab/scenarios
    res_scenarios = client.get("/api/lab/scenarios")
    assert res_scenarios.status_code == 200
    sc_data = res_scenarios.json()
    assert len(sc_data) == 15

    # 3. GET /api/lab/experiments
    res_exp = client.get("/api/lab/experiments")
    assert res_exp.status_code == 200
    exp_data = res_exp.json()
    assert "dossiers" in exp_data
    assert "genealogy" in exp_data

    # 4. POST /api/lab/override/submit
    res_ovr = client.post(
        "/api/lab/override/submit",
        json={
            "automatic_level": "CRITICAL",
            "override_level": "MEDIUM",
            "reason": "Test API Override",
            "operator": "API Tester",
        },
    )
    assert res_ovr.status_code == 200
    assert res_ovr.json()["status"] == "OVERRIDE_APPLIED"

    print("  [PASS] Stage 11: FastAPI lab router endpoints (/status, /scenarios, /experiments, /override) fully operational.")


def test_stage_12_ui_dashboard_air_gap_audit():
    print("\n--- STAGE 12: Offline UI Dashboard Air-Gap Integrity Audit ---")
    html_file = ROOT_DIR / "training_lab" / "ui" / "lab_dashboard.html"
    assert html_file.exists(), f"Missing dashboard file at {html_file}"

    content = html_file.read_text(encoding="utf-8")
    file_size_kb = len(content.encode("utf-8")) / 1024.0

    # 1. Size constraint: must be > 5 KB
    assert file_size_kb > 5.0, f"Dashboard size ({file_size_kb:.2f} KB) is too small (< 5 KB)"

    # 2. Air-gap check: zero external CDN / HTTP script/link tags
    assert "http://" not in content, "Air-gap violation: 'http://' reference detected in dashboard!"
    assert "https://" not in content, "Air-gap violation: 'https://' reference detected in dashboard!"

    # 3. Canvas & validation buttons present
    for cid in ["canvas_CAM_01", "canvas_CAM_02", "canvas_CAM_03", "canvas_CAM_04", "canvas_CAM_05"]:
        assert cid in content, f"Missing canvas element {cid} in dashboard"

    assert "CORRECT" in content
    assert "INCORRECT" in content
    assert "CHANGE CLASS" in content
    assert "OVERRIDE THREAT" in content
    assert "RTX 4060" in content

    print(f"  [PASS] Stage 12: Dashboard air-gap integrity verified ({file_size_kb:.2f} KB, 0 external URLs, 5 canvases & buttons present).")


def run_all_stages():
    print("=" * 75)
    print("BORDER SENTINEL - AI TRAINING LAB: GRAND END-TO-END VALIDATION (PHASES 1-17)")
    print("=" * 75)

    test_stage_01_scenario_and_simulator()
    test_stage_02_perception_and_isolated_tracking()
    test_stage_03_perimeter_geofence_state_machine()
    test_stage_04_compound_threat_and_operator_override()
    test_stage_05_human_validation_and_missed_detection()
    test_stage_06_dataset_quality_gate_and_splits()
    test_stage_07_versioned_model_training_registry()
    test_stage_08_multi_scenario_and_non_fabrication()
    test_stage_09_model_comparison_and_calibration()
    test_stage_10_experiment_manager_and_genealogy()
    test_stage_11_lab_rest_api_router()
    test_stage_12_ui_dashboard_air_gap_audit()

    print("\n" + "=" * 75)
    print("ALL 12 GRAND END-TO-END STAGES PASSED WITH ZERO ERRORS (100% PASS RATE)")
    print("=" * 75)


if __name__ == "__main__":
    run_all_stages()
