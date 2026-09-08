"""
Unit Tests for AI Evaluation Module (ai/detection/evaluation.py).

Validates:
1. IoU calculation logic across edge cases (identical, disjoint, partial, contained).
2. Precision, Recall, and F1 calculations including zero-division edge cases.
3. Class remapping schema integrity (COCO 80 -> Custom 9 classes).
4. Custom-only class taxonomy isolation (animal, bag marked as non-comparable).
5. Comparison report generation schema and data structure contract.
6. Error handling for missing files and directories.

Operates purely on synthetic/mocked data; requires ZERO weights or GPU hardware.
"""

from pathlib import Path
import pytest
import sys

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.detection.evaluation import (
    CUSTOM_CLASSES,
    COCO_TO_CUSTOM_MAP,
    CUSTOM_ONLY_CLASSES,
    compute_iou,
    calculate_pr_f1,
    build_comparison_report,
)


def test_compute_iou_identical_boxes():
    """Identical bounding boxes must produce an IoU of exactly 1.0."""
    box = [10.0, 10.0, 50.0, 50.0]
    iou = compute_iou(box, box)
    assert pytest.approx(iou, 1e-5) == 1.0


def test_compute_iou_disjoint_boxes():
    """Completely separated bounding boxes must produce an IoU of 0.0."""
    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [20.0, 20.0, 30.0, 30.0]
    iou = compute_iou(box_a, box_b)
    assert iou == 0.0


def test_compute_iou_partial_overlap():
    """Partially overlapping boxes should match analytical IoU."""
    # Box A: [0, 0, 10, 10] -> Area = 100
    # Box B: [5, 0, 15, 10] -> Area = 100
    # Intersection: [5, 0, 10, 10] -> Area = 50
    # Union: 100 + 100 - 50 = 150
    # Expected IoU: 50 / 150 = 1/3 ~ 0.33333
    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [5.0, 0.0, 15.0, 10.0]
    iou = compute_iou(box_a, box_b)
    assert pytest.approx(iou, 1e-4) == (1.0 / 3.0)


def test_compute_iou_zero_area_box():
    """Zero-area boxes should safely return 0.0 without division by zero."""
    box_a = [10.0, 10.0, 10.0, 10.0]
    box_b = [0.0, 0.0, 20.0, 20.0]
    iou = compute_iou(box_a, box_b)
    assert iou == 0.0


def test_calculate_pr_f1_perfect_scores():
    """Zero false positives and zero false negatives yields 100% across all metrics."""
    p, r, f1 = calculate_pr_f1(tp=10, fp=0, fn=0)
    assert p == 100.0
    assert r == 100.0
    assert f1 == 100.0


def test_calculate_pr_f1_with_false_positives():
    """When TP=10, FP=10, FN=0 -> Precision=50%, Recall=100%, F1=66.7%."""
    p, r, f1 = calculate_pr_f1(tp=10, fp=10, fn=0)
    assert p == 50.0
    assert r == 100.0
    assert f1 == 66.7


def test_calculate_pr_f1_with_false_negatives():
    """When TP=10, FP=0, FN=10 -> Precision=100%, Recall=50%, F1=66.7%."""
    p, r, f1 = calculate_pr_f1(tp=10, fp=0, fn=10)
    assert p == 100.0
    assert r == 50.0
    assert f1 == 66.7


def test_calculate_pr_f1_all_zeros():
    """Zero TP, FP, and FN should not raise ZeroDivisionError."""
    p, r, f1 = calculate_pr_f1(tp=0, fp=0, fn=0)
    assert isinstance(p, float)
    assert isinstance(r, float)
    assert isinstance(f1, float)


def test_class_remapping_integrity():
    """Validates the 7 mapped classes correctly map between COCO and Custom schema."""
    assert len(CUSTOM_CLASSES) == 9
    assert len(COCO_TO_CUSTOM_MAP) == 7

    # Ensure all target IDs in COCO_TO_CUSTOM_MAP exist in CUSTOM_CLASSES
    for coco_id, custom_id in COCO_TO_CUSTOM_MAP.items():
        assert custom_id in CUSTOM_CLASSES
        assert isinstance(coco_id, int)
        assert isinstance(custom_id, int)

    # Specific known mappings
    assert COCO_TO_CUSTOM_MAP[0] == 0   # person
    assert COCO_TO_CUSTOM_MAP[1] == 5   # bicycle
    assert COCO_TO_CUSTOM_MAP[2] == 1   # car
    assert COCO_TO_CUSTOM_MAP[3] == 4   # motorcycle
    assert COCO_TO_CUSTOM_MAP[5] == 3   # bus
    assert COCO_TO_CUSTOM_MAP[7] == 2   # truck
    assert COCO_TO_CUSTOM_MAP[24] == 7  # backpack


def test_custom_only_classes_isolation():
    """Ensures animal (6) and bag (8) are excluded from baseline remapping."""
    assert len(CUSTOM_ONLY_CLASSES) == 2
    assert 6 in CUSTOM_ONLY_CLASSES
    assert 8 in CUSTOM_ONLY_CLASSES
    assert CUSTOM_ONLY_CLASSES[6] == "animal"
    assert CUSTOM_ONLY_CLASSES[8] == "bag"

    mapped_target_ids = set(COCO_TO_CUSTOM_MAP.values())
    assert 6 not in mapped_target_ids
    assert 8 not in mapped_target_ids


def test_build_comparison_report_schema():
    """Tests the structured comparison report contract with synthetic evaluation data."""
    synthetic_baseline_eval = {
        "avg_latency_ms": 50.0,
        "fps_throughput": 20.0,
        "false_positives_per_hour": 0.0,
        "scenarios": {
            "OVERALL": {"precision": 100.0, "recall": 100.0, "f1": 100.0, "tp_count": 10, "fp_count": 0, "fn_count": 0, "total_frames": 10},
            "DAY": {"precision": 100.0, "recall": 100.0, "f1": 100.0, "tp_count": 10, "fp_count": 0, "fn_count": 0, "total_frames": 10},
            "NIGHT_LOW_LIGHT": {"precision": 100.0, "recall": 100.0, "f1": 100.0, "tp_count": 0, "fp_count": 0, "fn_count": 0, "total_frames": 0},
            "DISTANT_OBJECTS": {"precision": 100.0, "recall": 100.0, "f1": 100.0, "tp_count": 0, "fp_count": 0, "fn_count": 0, "total_frames": 0},
        },
        "classes": {
            "0": {"name": "person", "precision": 100.0, "recall": 100.0, "f1": 100.0, "tp": 10, "fp": 0, "fn": 0},
        },
        "hard_negative_stats": {
            "total_negative_frames": 3,
            "false_positive_frames": 0,
            "total_false_detections": 0,
        },
    }

    synthetic_baseline_bench = {
        "warmup_passes": 5,
        "timed_passes": 50,
        "min_latency_ms": 48.0,
        "mean_latency_ms": 50.0,
        "p95_latency_ms": 55.0,
        "max_latency_ms": 60.0,
        "fps_throughput": 20.0,
        "peak_vram_mb": 250.0,
    }

    synthetic_custom_report = {
        "model_version": "YOLO-L-v002",
        "avg_latency_ms": 389.96,
        "fps_throughput": 2.6,
        "operational_metrics": {"false_positives_per_hour": 2215.4},
        "scenarios": {
            "OVERALL": {"precision": 81.0, "recall": 100.0, "f1": 89.5, "fp_count": 4},
        },
        "classes": {
            "0": {"name": "person", "precision": 100.0, "recall": 100.0},
        },
    }

    synthetic_custom_index = {
        "version": "YOLO-L-v002",
        "device": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "metrics": {
            "mAP50": 99.5,
            "mAP50-95": 98.63,
            "precision": 95.89,
            "recall": 95.88,
        },
    }

    report = build_comparison_report(
        baseline_eval=synthetic_baseline_eval,
        baseline_bench=synthetic_baseline_bench,
        custom_report=synthetic_custom_report,
        custom_index=synthetic_custom_index,
        dataset_version="v2",
    )

    # Validate structure
    assert "metadata" in report
    assert "summary" in report
    assert "latency_and_throughput_benchmark" in report
    assert "per_class_comparison" in report
    assert "scenario_breakdown" in report
    assert "hard_negative_rejection" in report
    assert "methodology_and_notes" in report

    # Validate all 9 classes present
    assert len(report["per_class_comparison"]) == 9
    assert "person" in report["per_class_comparison"]
    assert "animal" in report["per_class_comparison"]
    assert "bag" in report["per_class_comparison"]

    # Validate animal and bag are marked N/A for baseline
    animal_entry = report["per_class_comparison"]["animal"]["baseline_yolov8l"]
    bag_entry = report["per_class_comparison"]["bag"]["baseline_yolov8l"]
    assert "N/A — no equivalent COCO class" in animal_entry["precision"]
    assert "N/A — no equivalent COCO class" in bag_entry["precision"]

    # Validate summary fields
    summary = report["summary"]
    assert summary["baseline_model"]["version"] == "yolov8l.pt (COCO-pretrained)"
    assert summary["custom_model"]["version"] == "YOLO-L-v002"
    assert summary["custom_model"]["mAP50"] == 99.5
