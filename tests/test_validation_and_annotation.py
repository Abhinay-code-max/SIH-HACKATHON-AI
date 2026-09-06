"""
Automated 5-Stage Validation Suite for Human Validation & Annotation Capture Engines.
Tests:
Stage 1: True Positive Feedback (CORRECT validation on PERSON_001 -> TRUE_POSITIVE & training candidate).
Stage 2: Class Correction Feedback (CHANGE_CLASS car -> truck -> WRONG_CLASS & updated ground truth).
Stage 3: False Positive Feedback (INCORRECT on ghost box -> FALSE_POSITIVE & excluded from positive training).
Stage 4: Missed Detection Capture (AnnotationCapture manual box [100, 150, 180, 260] -> FALSE_NEGATIVE & crop on disk).
Stage 5: Error Taxonomy Statistics & Training Export (Breakdown across all 9 error types & ground truth export).
"""

from pathlib import Path
import sys
import time
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.validation_engine import (
    ErrorCategory,
    ValidationDecision,
    ValidationEngine,
    validation_engine,
)
from training_lab.engine.annotation_capture import (
    AnnotationCapture,
    annotation_capture,
)


def test_stage_1_true_positive_feedback():
    """Stage 1: True Positive Feedback - Record CORRECT validation on PERSON_001."""
    print("\n" + "=" * 80)
    print("[Stage 1/5] Testing True Positive Feedback Validation...")
    print("=" * 80)

    validation_engine.clear()

    rec = validation_engine.record_feedback(
        scenario_id="SCN_01",
        camera_id="CAM_01",
        frame_idx=12,
        decision=ValidationDecision.CORRECT,
        ai_prediction={
            "class_name": "person",
            "confidence": 0.94,
            "bbox": [60.0, 188.0, 91.0, 293.0],
        },
        tracking_id=1,
        notes="Verified standard daytime pedestrian detection",
    )

    assert rec.decision == ValidationDecision.CORRECT
    assert rec.error_category == ErrorCategory.TRUE_POSITIVE
    assert rec.human_verified_class == "person"
    assert rec.human_bbox == [60.0, 188.0, 91.0, 293.0]
    assert rec.is_training_candidate is True
    assert rec.camera_id == "CAM_01"
    assert rec.scenario_id == "SCN_01"
    assert rec.tracking_id == 1

    print(f"  [OK] Recorded {rec.record_id}: Decision={rec.decision.value} -> Category={rec.error_category.value}")
    print(f"  [OK] Ground-Truth Class: '{rec.human_verified_class}', BBox: {rec.human_bbox}")
    print(f"  [OK] Training Candidate: {rec.is_training_candidate}")
    print("  --> PASS: Stage 1 True Positive Feedback Verified.")


def test_stage_2_class_correction_feedback():
    """Stage 2: Class Correction Feedback - Record CHANGE_CLASS on car -> truck."""
    print("\n" + "=" * 80)
    print("[Stage 2/5] Testing Class Correction Feedback (car -> truck)...")
    print("=" * 80)

    rec = validation_engine.record_feedback(
        scenario_id="SCN_02",
        camera_id="CAM_02",
        frame_idx=45,
        decision=ValidationDecision.CHANGE_CLASS,
        ai_prediction={
            "class_name": "car",
            "confidence": 0.78,
            "bbox": [120.0, 200.0, 240.0, 310.0],
        },
        human_verified_class="truck",
        tracking_id=5,
        notes="Commercial flatbed misclassified as passenger car",
    )

    assert rec.decision == ValidationDecision.CHANGE_CLASS
    assert rec.error_category == ErrorCategory.WRONG_CLASS
    assert rec.human_verified_class == "truck"
    assert rec.human_bbox == [120.0, 200.0, 240.0, 310.0]
    assert rec.is_training_candidate is True
    assert rec.camera_id == "CAM_02"

    print(f"  [OK] Recorded {rec.record_id}: Decision={rec.decision.value} -> Category={rec.error_category.value}")
    print(f"  [OK] Reclassified: '{rec.ai_prediction['class_name']}' -> '{rec.human_verified_class}'")
    print(f"  [OK] Flagged as training candidate for hard-class fine tuning.")
    print("  --> PASS: Stage 2 Class Correction Feedback Verified.")


def test_stage_3_false_positive_feedback():
    """Stage 3: False Positive Feedback - Record INCORRECT on ghost detection."""
    print("\n" + "=" * 80)
    print("[Stage 3/5] Testing False Positive Feedback on Ghost Detection...")
    print("=" * 80)

    rec = validation_engine.record_feedback(
        scenario_id="SCN_03",
        camera_id="CAM_03",
        frame_idx=80,
        decision=ValidationDecision.INCORRECT,
        ai_prediction={
            "class_name": "person",
            "confidence": 0.39,
            "bbox": [400.0, 110.0, 435.0, 165.0],
        },
        notes="Shadow artifact falsely tagged as human intruder",
    )

    assert rec.decision == ValidationDecision.INCORRECT
    assert rec.error_category == ErrorCategory.FALSE_POSITIVE
    assert rec.is_training_candidate is False
    assert rec.camera_id == "CAM_03"

    print(f"  [OK] Recorded {rec.record_id}: Decision={rec.decision.value} -> Category={rec.error_category.value}")
    print(f"  [OK] Confirmed excluded from positive ground truth training (is_training_candidate=False).")
    print("  --> PASS: Stage 3 False Positive Feedback Verified.")


def test_stage_4_missed_detection_capture():
    """Stage 4: Missed Detection Capture - Inject manual bounding box via AnnotationCapture."""
    print("\n" + "=" * 80)
    print("[Stage 4/5] Testing Missed Detection Capture (Manual Addition with Crop)...")
    print("=" * 80)

    # 1. Coordinate size validation check (fails on < 10x10 px)
    try:
        annotation_capture.capture_missed_detection(
            scenario_id="SCN_01",
            camera_id="CAM_01",
            frame_idx=30,
            bbox=[100.0, 100.0, 105.0, 105.0],  # 5x5 px
            class_name="person",
        )
        assert False, "Should have raised ValueError on box smaller than 10x10 px"
    except ValueError as e:
        print(f"  [OK] Rejected sub-10px invalid bbox correctly: {e}")

    # 2. Valid missed detection capture with synthetic frame
    dummy_frame = np.full((480, 640, 3), 45, dtype=np.uint8)
    dummy_frame[150:260, 100:180] = (70, 130, 200)

    manual_box = [100.0, 150.0, 180.0, 260.0]
    rec = annotation_capture.capture_missed_detection(
        scenario_id="SCN_01",
        camera_id="CAM_01",
        frame_idx=30,
        bbox=manual_box,
        class_name="person",
        notes="Human validator drawn box for missed crawling subject",
        frame=dummy_frame,
    )

    assert rec.decision == ValidationDecision.MANUAL_ADDITION
    assert rec.error_category == ErrorCategory.FALSE_NEGATIVE
    assert rec.human_verified_class == "person"
    assert rec.human_bbox == manual_box
    assert rec.is_training_candidate is True
    assert rec.crop_path is not None, "Expected crop path to be set"

    crop_disk_path = ROOT_DIR / rec.crop_path
    assert crop_disk_path.is_file(), f"Saved crop does not exist on disk: {crop_disk_path}"
    assert crop_disk_path.stat().st_size > 100, "Crop file is empty"

    print(f"  [OK] Recorded {rec.record_id}: Decision={rec.decision.value} -> Category={rec.error_category.value}")
    print(f"  [OK] Verified False Negative Ground-Truth: '{rec.human_verified_class}' at {rec.human_bbox}")
    print(f"  [OK] Crop preserved on disk: {rec.crop_path} ({crop_disk_path.stat().st_size} bytes)")
    print("  --> PASS: Stage 4 Missed Detection Capture Verified.")


def test_stage_5_error_taxonomy_statistics_and_export():
    """Stage 5: Error Taxonomy Statistics & Training Export across 9-type taxonomy."""
    print("\n" + "=" * 80)
    print("[Stage 5/5] Testing 9-Type Error Taxonomy Statistics & Training Export...")
    print("=" * 80)

    # Inject feedback records covering remaining failure modes in the 9-type taxonomy
    remaining_modes = [
        ("SCN_04", "CAM_04", ErrorCategory.LOW_CONFIDENCE, ValidationDecision.IGNORE, "Vegetation glare below threshold"),
        ("SCN_05", "CAM_05", ErrorCategory.DUPLICATE_DETECTION, ValidationDecision.INCORRECT, "Double bbox on same motorcycle"),
        ("SCN_01", "CAM_01", ErrorCategory.TRACKING_ERROR, ValidationDecision.INCORRECT, "Track ID switch across occlusion"),
        ("SCN_02", "CAM_02", ErrorCategory.PERIMETER_ERROR, ValidationDecision.INCORRECT, "Target detected outside legal zone"),
        ("SCN_03", "CAM_03", ErrorCategory.THREAT_CLASSIFICATION_ERROR, ValidationDecision.INCORRECT, "Threat score under-weighted"),
    ]

    for scn_id, cam_id, err_cat, dec, note in remaining_modes:
        validation_engine.record_feedback(
            scenario_id=scn_id,
            camera_id=cam_id,
            frame_idx=50,
            decision=dec,
            error_category=err_cat,
            notes=note,
        )

    # 1. Verify Error Statistics
    stats = validation_engine.compute_error_statistics()
    assert stats["total_validations"] >= 9, f"Expected at least 9 validations, got {stats['total_validations']}"

    cat_counts = stats["counts_by_category"]
    print(f"  Total Validations Recorded: {stats['total_validations']}")
    print("  9-Type Failure Mode Breakdown:")
    for cat_name, count in cat_counts.items():
        print(f"    - {cat_name:28s}: {count}")
        assert cat_name in [c.value for c in ErrorCategory]

    # Verify all 9 error categories exist in stats
    for cat in ErrorCategory:
        assert cat.value in cat_counts, f"Missing category {cat.value} in statistics"

    assert stats["true_positives"] >= 1
    assert stats["false_positives"] >= 1
    assert stats["false_negatives"] >= 1
    assert stats["wrong_class_errors"] >= 1

    print(f"  [OK] Accuracy Pct: {stats['accuracy_pct']}% | FP Rate: {stats['false_positive_rate']}% | FN Rate: {stats['false_negative_rate']}%")

    # 2. Verify Training Candidates Ground-Truth Export
    candidates = validation_engine.export_verified_training_candidates()
    assert len(candidates) >= 3, f"Expected at least 3 training candidates, got {len(candidates)}"

    found_tp = any(c["error_category"] == "TRUE_POSITIVE" for c in candidates)
    found_wrong_cls = any(c["error_category"] == "WRONG_CLASS" for c in candidates)
    found_manual_add = any(c["error_category"] == "FALSE_NEGATIVE" for c in candidates)

    assert found_tp, "Missing TRUE_POSITIVE candidate in export"
    assert found_wrong_cls, "Missing WRONG_CLASS candidate in export"
    assert found_manual_add, "Missing FALSE_NEGATIVE manual addition in export"

    for c in candidates:
        assert "class_name" in c and len(c["class_name"]) > 0
        assert "bbox" in c and len(c["bbox"]) == 4
        assert c["bbox"][2] > c["bbox"][0] and c["bbox"][3] > c["bbox"][1]
        print(f"  [Exported Ground Truth] {c['record_id']} [{c['camera_id']}] {c['class_name']:10s} bbox={c['bbox']} ({c['error_category']})")

    # 3. Verify Disk Persistence & Independent Reload
    fresh_engine = ValidationEngine()
    reloaded_records = fresh_engine.get_validations()
    assert len(reloaded_records) == stats["total_validations"], (
        f"Disk persistence mismatch: {len(reloaded_records)} != {stats['total_validations']}"
    )
    print(f"  [OK] Disk persistence confirmed: {len(reloaded_records)} records reloaded from validation_log.json.")
    print("  --> PASS: Stage 5 Error Taxonomy Statistics & Training Export Verified.")


def run_all_stages():
    """Runs all 5 validation stages."""
    print("=" * 80)
    print("BORDER SENTINEL - HUMAN VALIDATION & ANNOTATION CAPTURE TEST SUITE")
    print("=" * 80)
    start_time = time.time()

    test_stage_1_true_positive_feedback()
    test_stage_2_class_correction_feedback()
    test_stage_3_false_positive_feedback()
    test_stage_4_missed_detection_capture()
    test_stage_5_error_taxonomy_statistics_and_export()

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL 5 VALIDATION & ANNOTATION STAGES PASSED in {total_time:.2f}s! [100% PASS RATE]")
    print("=" * 80)


if __name__ == "__main__":
    run_all_stages()
