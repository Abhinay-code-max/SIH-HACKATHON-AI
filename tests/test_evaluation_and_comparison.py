"""
Automated Validation Suite for Phase 11 & 12:
Multi-Scenario & Per-Camera Evaluator, Model Comparator & Confidence Calibration Analyzer.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.model_evaluator import ModelEvaluator
from training_lab.engine.model_comparator import ModelComparator


def test_stage_1_per_camera_and_class_eval():
    print("\n--- STAGE 1: Per-Camera & Per-Class Evaluation on model_v001 ---")
    evaluator = ModelEvaluator()

    # Provide synthetic test predictions and ground truth across all 5 cameras
    mock_preds = [
        {"camera_id": "CAM_01", "class_name": "person", "bbox": [60.0, 188.0, 91.0, 293.0], "confidence": 0.92},
        {"camera_id": "CAM_01", "class_name": "person", "bbox": [10.0, 20.0, 30.0, 40.0], "confidence": 0.40},  # FP
        {"camera_id": "CAM_02", "class_name": "car", "bbox": [0.0, 188.0, 41.0, 285.0], "confidence": 0.88},
        {"camera_id": "CAM_03", "class_name": "backpack", "bbox": [306.0, 260.0, 360.0, 319.0], "confidence": 0.82},
        {"camera_id": "CAM_04", "class_name": "person", "bbox": [480.0, 226.0, 505.0, 257.0], "confidence": 0.79},
        {"camera_id": "CAM_05", "class_name": "motorcycle", "bbox": [30.0, 255.0, 110.0, 346.0], "confidence": 0.85},
    ]

    mock_gt = [
        {"camera_id": "CAM_01", "class_name": "person", "bbox": [60.0, 188.0, 91.0, 293.0]},
        {"camera_id": "CAM_02", "class_name": "car", "bbox": [0.0, 188.0, 41.0, 285.0]},
        {"camera_id": "CAM_03", "class_name": "backpack", "bbox": [306.0, 260.0, 360.0, 319.0]},
        {"camera_id": "CAM_04", "class_name": "person", "bbox": [480.0, 226.0, 505.0, 257.0]},
        {"camera_id": "CAM_05", "class_name": "motorcycle", "bbox": [30.0, 255.0, 110.0, 346.0]},
    ]

    report = evaluator.evaluate_scenario(
        model_version="model_v001",
        scenario_id="SCN_01",
        ground_truth=mock_gt,
        custom_detections=mock_preds,
        test_frames=10,
    )

    assert report["status"] == "EVALUATED"
    assert report["model_version"] == "model_v001"
    assert "overall" in report
    assert report["overall"]["tp"] == 5
    assert report["overall"]["fp"] == 1
    assert report["overall"]["fn"] == 0
    assert report["overall"]["precision"] > 80.0
    assert report["overall"]["recall"] == 100.0
    assert report["overall"]["mAP50"] > 80.0
    assert report["overall"]["fps"] is not None
    assert report["overall"]["latency_ms"] is not None

    # Check all 5 cameras exist in scorecard
    for cam_id in ["CAM_01", "CAM_02", "CAM_03", "CAM_04", "CAM_05"]:
        assert cam_id in report["per_camera"], f"Missing camera {cam_id} in per_camera scorecard"
        cam_stats = report["per_camera"][cam_id]
        assert "precision" in cam_stats
        assert "recall" in cam_stats
        assert "mAP50" in cam_stats

    # Check per-class scorecard
    assert "person" in report["per_class"]
    assert "car" in report["per_class"]
    print(f"  [PASS] Stage 1: Successfully evaluated CAM_01-05 and classes (mAP50={report['overall']['mAP50']}%, TP={report['overall']['tp']}, FP={report['overall']['fp']})")


def test_stage_2_non_fabrication_rule():
    print("\n--- STAGE 2: Non-Fabrication Rule Enforcement (Section 28) ---")
    evaluator = ModelEvaluator()

    # Empty scenario without ground truth and without custom detections
    report = evaluator.evaluate_scenario(
        model_version="model_v001",
        scenario_id="SCN_EMPTY_TEST_UNRUN",
        ground_truth=None,
        custom_detections=None,
    )

    assert report["status"] == "NOT TESTED", f"Expected 'NOT TESTED', got {report['status']}"
    assert "reason" in report
    assert report["overall"]["mAP50"] is None, "mAP50 must be None under Section 28 Non-Fabrication Rule"
    assert report["overall"]["precision"] is None
    assert report["overall"]["recall"] is None
    print("  [PASS] Stage 2: Section 28 Non-Fabrication Rule strictly enforced (status='NOT TESTED', metrics=None)")


def test_stage_3_multi_scenario_and_environmental_breakdown():
    print("\n--- STAGE 3: Multi-Scenario Benchmark & Environmental Analysis ---")
    evaluator = ModelEvaluator()

    # Multi-scenario evaluation across SCN_01 (Day) and SCN_02 (Night)
    multi_rep = evaluator.evaluate_multi_scenario(
        model_version="model_v001",
        scenario_ids=["SCN_01", "SCN_02"],
    )

    assert multi_rep["model_version"] == "model_v001"
    assert len(multi_rep["scenarios_evaluated"]) == 2
    assert "environmental_breakdown" in multi_rep
    assert "day" in multi_rep["environmental_breakdown"]
    assert "night" in multi_rep["environmental_breakdown"]
    assert "per_camera" in multi_rep
    for cam_id in ["CAM_01", "CAM_02", "CAM_03", "CAM_04", "CAM_05"]:
        assert cam_id in multi_rep["per_camera"]

    day_cnt = multi_rep["environmental_breakdown"]["day"]["scenarios_count"]
    night_cnt = multi_rep["environmental_breakdown"]["night"]["scenarios_count"]
    print(f"  [PASS] Stage 3: Multi-scenario run completed (Day scenarios: {day_cnt}, Night scenarios: {night_cnt})")


def test_stage_4_model_comparator_and_verdict():
    print("\n--- STAGE 4: Side-by-Side Model Comparator & Decision Verdict ---")
    comparator = ModelComparator()

    eval_v001 = {
        "model_version": "model_v001",
        "status": "EVALUATED",
        "overall": {
            "mAP50": 81.0,
            "precision": 80.5,
            "recall": 78.0,
            "fp": 42,
            "fn": 38,
        },
        "per_camera": {
            "CAM_01": {"mAP50": 82.0, "precision": 81.0, "recall": 79.0, "fp": 8, "fn": 7},
            "CAM_02": {"mAP50": 80.0, "precision": 80.0, "recall": 77.0, "fp": 9, "fn": 8},
        },
        "per_class": {
            "person": {"mAP50": 83.0, "precision": 82.0, "recall": 80.0, "fp": 20, "fn": 18},
        },
    }

    eval_v002 = {
        "model_version": "model_v002",
        "status": "EVALUATED",
        "overall": {
            "mAP50": 87.0,
            "precision": 88.0,
            "recall": 85.0,
            "fp": 26,
            "fn": 24,
        },
        "per_camera": {
            "CAM_01": {"mAP50": 88.5, "precision": 89.0, "recall": 86.0, "fp": 4, "fn": 4},
            "CAM_02": {"mAP50": 85.5, "precision": 87.0, "recall": 84.0, "fp": 5, "fn": 5},
        },
        "per_class": {
            "person": {"mAP50": 89.0, "precision": 90.0, "recall": 87.0, "fp": 12, "fn": 10},
        },
    }

    comp = comparator.compare_evaluations(eval_v001, eval_v002)

    assert comp["verdict"] == "VERIFIED_IMPROVEMENT"
    assert comp["overall_deltas"]["delta_mAP50"] == 6.0
    assert comp["overall_deltas"]["delta_fp"] == -16
    assert comp["overall_deltas"]["delta_fn"] == -14
    assert comp["per_camera_deltas"]["CAM_01"]["delta_mAP50"] == 6.5
    assert comp["per_class_deltas"]["person"]["delta_mAP50"] == 6.0

    # Also test inconclusive comparison with NOT TESTED model
    eval_not_tested = {"model_version": "model_v003", "status": "NOT TESTED", "overall": {"mAP50": None}}
    comp_inconclusive = comparator.compare_evaluations(eval_v001, eval_not_tested)
    assert comp_inconclusive["verdict"] == "INCONCLUSIVE"
    assert comp_inconclusive["overall_deltas"]["delta_mAP50"] == "NOT APPLICABLE"

    print(f"  [PASS] Stage 4: Delta computation verified: Delta_mAP50=+{comp['overall_deltas']['delta_mAP50']}%, Delta_FP={comp['overall_deltas']['delta_fp']} -> Verdict: {comp['verdict']}")


def test_stage_5_confidence_calibration():
    print("\n--- STAGE 5: Confidence Calibration Analysis ---")
    comparator = ModelComparator()

    ground_truth = [
        {"camera_id": "CAM_01", "class_name": "person", "bbox": [10.0, 10.0, 50.0, 50.0]},
        {"camera_id": "CAM_01", "class_name": "car", "bbox": [100.0, 100.0, 200.0, 200.0]},
        {"camera_id": "CAM_02", "class_name": "person", "bbox": [20.0, 20.0, 60.0, 60.0]},
    ]

    detections = [
        # 90-100% bin: 2 predictions, both correct (100% empirical acc)
        {"camera_id": "CAM_01", "class_name": "person", "bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.95},
        {"camera_id": "CAM_01", "class_name": "car", "bbox": [100.0, 100.0, 200.0, 200.0], "confidence": 0.92},
        # 80-90% bin: 1 prediction, correct (100% empirical acc)
        {"camera_id": "CAM_02", "class_name": "person", "bbox": [20.0, 20.0, 60.0, 60.0], "confidence": 0.85},
        # 70-80% bin: 1 prediction, incorrect FP (0% empirical acc)
        {"camera_id": "CAM_02", "class_name": "person", "bbox": [500.0, 500.0, 600.0, 600.0], "confidence": 0.75},
        # <50% bin: 1 prediction, incorrect FP (0% empirical acc)
        {"camera_id": "CAM_03", "class_name": "dog", "bbox": [50.0, 50.0, 90.0, 90.0], "confidence": 0.35},
    ]

    cal_rep = comparator.analyze_confidence_calibration(detections, ground_truth)

    assert cal_rep["total_detections_analyzed"] == 5
    assert "expected_calibration_error_pct" in cal_rep
    assert "calibration_quality" in cal_rep
    bins = cal_rep["bins"]
    assert bins["90-100%"]["sample_count"] == 2
    assert bins["90-100%"]["correct_count"] == 2
    assert bins["90-100%"]["empirical_accuracy"] == 100.0

    assert bins["80-90%"]["sample_count"] == 1
    assert bins["80-90%"]["correct_count"] == 1
    assert bins["80-90%"]["empirical_accuracy"] == 100.0

    assert bins["70-80%"]["sample_count"] == 1
    assert bins["70-80%"]["correct_count"] == 0
    assert bins["70-80%"]["empirical_accuracy"] == 0.0

    print(f"  [PASS] Stage 5: Confidence calibration analyzed (Total={cal_rep['total_detections_analyzed']}, ECE={cal_rep['expected_calibration_error_pct']}%, Quality={cal_rep['calibration_quality']})")


def run_all_stages():
    print("=" * 70)
    print("BORDER SENTINEL - TRAINING LAB: PHASE 11 & 12 EVALUATION TEST SUITE")
    print("=" * 70)

    test_stage_1_per_camera_and_class_eval()
    test_stage_2_non_fabrication_rule()
    test_stage_3_multi_scenario_and_environmental_breakdown()
    test_stage_4_model_comparator_and_verdict()
    test_stage_5_confidence_calibration()

    print("\n" + "=" * 70)
    print("ALL 5 VALIDATION STAGES PASSED SUCCESSFULLY (100% PASS RATE)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_stages()
