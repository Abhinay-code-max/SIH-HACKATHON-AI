"""
Model Comparator & Confidence Calibration Analyzer.
Compares benchmark evaluations between models (e.g. V001 vs V002), calculates deltas
in mAP50, precision, recall, false positives, false negatives, assigns decision verdicts
(VERIFIED_IMPROVEMENT, REGRESSION_DETECTED, INCONCLUSIVE), and analyzes confidence
calibration across standard confidence bins.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.dataset_generator import compute_iou

RESULTS_DIR = ROOT_DIR / "training_lab" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ModelComparator:
    """
    Compares two model evaluations (Baseline vs Candidate) and assesses
    confidence calibration metrics.
    """

    CONFIDENCE_BINS = [
        ("90-100%", 0.90, 1.0001),
        ("80-90%", 0.80, 0.90),
        ("70-80%", 0.70, 0.80),
        ("50-70%", 0.50, 0.70),
        ("<50%", 0.00, 0.50),
    ]

    def __init__(self, results_dir: Optional[Union[str, Path]] = None):
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def compare_evaluations(
        self,
        eval_a: Dict[str, Any],
        eval_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compares Baseline (eval_a) vs Candidate (eval_b).
        Computes deltas: Delta = Candidate (B) - Baseline (A).
        Assigns decision verdict:
          - VERIFIED_IMPROVEMENT
          - REGRESSION_DETECTED
          - INCONCLUSIVE
          - NO_CHANGE
          - MARGINAL_TRADE_OFF
        """
        model_a = eval_a.get("model_version", "model_A")
        model_b = eval_b.get("model_version", "model_B")
        status_a = eval_a.get("status", "EVALUATED")
        status_b = eval_b.get("status", "EVALUATED")

        # Section 28 Non-Fabrication Rule: if either evaluation was NOT TESTED or missing mAP
        if (
            status_a == "NOT TESTED"
            or status_b == "NOT TESTED"
            or eval_a.get("overall", {}).get("mAP50") is None
            or eval_b.get("overall", {}).get("mAP50") is None
        ):
            return {
                "model_a": model_a,
                "model_b": model_b,
                "verdict": "INCONCLUSIVE",
                "reason": "One or both model evaluations were NOT TESTED or contain unmeasured metrics.",
                "overall_deltas": {
                    "delta_mAP50": "NOT APPLICABLE",
                    "delta_precision": "NOT APPLICABLE",
                    "delta_recall": "NOT APPLICABLE",
                    "delta_fp": "NOT APPLICABLE",
                    "delta_fn": "NOT APPLICABLE",
                },
                "per_camera_deltas": {},
                "per_class_deltas": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        oa = eval_a.get("overall", {})
        ob = eval_b.get("overall", {})

        map_a = float(oa.get("mAP50", 0.0))
        map_b = float(ob.get("mAP50", 0.0))
        prec_a = float(oa.get("precision", 0.0))
        prec_b = float(ob.get("precision", 0.0))
        rec_a = float(oa.get("recall", 0.0))
        rec_b = float(ob.get("recall", 0.0))
        fp_a = int(oa.get("fp", 0))
        fp_b = int(ob.get("fp", 0))
        fn_a = int(oa.get("fn", 0))
        fn_b = int(ob.get("fn", 0))

        delta_map50 = round(map_b - map_a, 2)
        delta_precision = round(prec_b - prec_a, 2)
        delta_recall = round(rec_b - rec_a, 2)
        delta_fp = fp_b - fp_a
        delta_fn = fn_b - fn_a

        # Determine decision verdict
        if delta_map50 > 0 and delta_fp <= 0:
            verdict = "VERIFIED_IMPROVEMENT"
        elif delta_map50 > 1.5 and delta_fp <= 5:
            # Significant mAP boost with negligible FP increase
            verdict = "VERIFIED_IMPROVEMENT"
        elif delta_map50 < -0.5 or (delta_fp > 10 and delta_map50 <= 0):
            verdict = "REGRESSION_DETECTED"
        elif delta_map50 == 0.0 and delta_fp == 0:
            verdict = "NO_CHANGE"
        elif delta_map50 > 0:
            verdict = "VERIFIED_IMPROVEMENT"
        else:
            verdict = "MARGINAL_TRADE_OFF"

        # Per-camera delta breakdown
        cams_a = eval_a.get("per_camera", {})
        cams_b = eval_b.get("per_camera", {})
        all_cam_keys = sorted(list(set(list(cams_a.keys()) + list(cams_b.keys()))))
        per_camera_deltas: Dict[str, Dict[str, Any]] = {}

        for c_id in all_cam_keys:
            ca = cams_a.get(c_id, {})
            cb = cams_b.get(c_id, {})
            per_camera_deltas[c_id] = {
                "delta_mAP50": round(float(cb.get("mAP50", 0.0)) - float(ca.get("mAP50", 0.0)), 2)
                if ("mAP50" in cb and "mAP50" in ca) else 0.0,
                "delta_precision": round(float(cb.get("precision", 0.0)) - float(ca.get("precision", 0.0)), 2),
                "delta_recall": round(float(cb.get("recall", 0.0)) - float(ca.get("recall", 0.0)), 2),
                "delta_fp": int(cb.get("fp", 0)) - int(ca.get("fp", 0)),
                "delta_fn": int(cb.get("fn", 0)) - int(ca.get("fn", 0)),
            }

        # Per-class delta breakdown
        cls_a = eval_a.get("per_class", {})
        cls_b = eval_b.get("per_class", {})
        all_cls_keys = sorted(list(set(list(cls_a.keys()) + list(cls_b.keys()))))
        per_class_deltas: Dict[str, Dict[str, Any]] = {}

        for cl_name in all_cls_keys:
            cla = cls_a.get(cl_name, {})
            clb = cls_b.get(cl_name, {})
            per_class_deltas[cl_name] = {
                "delta_mAP50": round(float(clb.get("mAP50", 0.0)) - float(cla.get("mAP50", 0.0)), 2)
                if ("mAP50" in clb and "mAP50" in cla) else 0.0,
                "delta_precision": round(float(clb.get("precision", 0.0)) - float(cla.get("precision", 0.0)), 2),
                "delta_recall": round(float(clb.get("recall", 0.0)) - float(cla.get("recall", 0.0)), 2),
                "delta_fp": int(clb.get("fp", 0)) - int(ca.get("fp", 0)),
                "delta_fn": int(clb.get("fn", 0)) - int(ca.get("fn", 0)),
            }

        comparison_report = {
            "model_a": model_a,
            "model_b": model_b,
            "verdict": verdict,
            "overall_deltas": {
                "delta_mAP50": delta_map50,
                "delta_precision": delta_precision,
                "delta_recall": delta_recall,
                "delta_fp": delta_fp,
                "delta_fn": delta_fn,
            },
            "baseline_summary": {
                "mAP50": map_a,
                "precision": prec_a,
                "recall": rec_a,
                "fp": fp_a,
                "fn": fn_a,
            },
            "candidate_summary": {
                "mAP50": map_b,
                "precision": prec_b,
                "recall": rec_b,
                "fp": fp_b,
                "fn": fn_b,
            },
            "per_camera_deltas": per_camera_deltas,
            "per_class_deltas": per_class_deltas,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save to disk
        dest_report = self.results_dir / f"compare_{model_a}_vs_{model_b}_{int(time.time())}.json"
        with open(dest_report, "w", encoding="utf-8") as rf:
            json.dump(comparison_report, rf, indent=2)

        return comparison_report

    def analyze_confidence_calibration(
        self,
        detections: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        iou_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Analyzes prediction confidence against empirical accuracy.
        Bins: '90-100%', '80-90%', '70-80%', '50-70%', '<50%'.
        Calculates Expected Calibration Error (ECE) and identifies calibration status.
        """
        # Match detections to ground truth to classify each detection as correct (TP) or incorrect (FP)
        matched_gt = set()
        det_evals = []

        sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)

        for d in sorted_dets:
            conf = float(d.get("confidence", 0.0))
            pred_box = d.get("bbox", [])
            pred_cls = str(d.get("class_name", "")).lower().strip()
            pred_cam = str(d.get("camera_id", ""))

            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, gt in enumerate(ground_truth):
                if g_idx in matched_gt:
                    continue
                gt_cam = str(gt.get("camera_id", ""))
                # If cameras specified on both, enforce camera match
                if pred_cam and gt_cam and pred_cam != gt_cam:
                    continue

                gt_cls = str(gt.get("class_name", "")).lower().strip()
                if pred_cls == gt_cls:
                    iou = compute_iou(pred_box, gt.get("bbox", []))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            is_correct = False
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                is_correct = True
                matched_gt.add(best_gt_idx)

            det_evals.append({"confidence": conf, "is_correct": is_correct})

        # Bin evaluation
        calibration_bins: Dict[str, Dict[str, Any]] = {}
        total_samples = len(det_evals)
        weighted_error_sum = 0.0

        for bin_name, lower, upper in self.CONFIDENCE_BINS:
            bin_items = [
                de for de in det_evals
                if lower <= de["confidence"] < upper or (upper == 1.0001 and de["confidence"] == 1.0)
            ]
            count = len(bin_items)
            correct_count = sum(1 for de in bin_items if de["is_correct"])

            empirical_acc = round((correct_count / count) * 100, 2) if count > 0 else 0.0
            mean_conf = round(float(np.mean([de["confidence"] for de in bin_items])) * 100, 2) if count > 0 else 0.0
            cal_err = round(abs(mean_conf - empirical_acc), 2)

            if total_samples > 0 and count > 0:
                weighted_error_sum += (count / total_samples) * cal_err

            calibration_bins[bin_name] = {
                "bin_range": f"[{int(lower*100)}%, {int(upper*100) if upper <= 1.0 else 100}%)",
                "sample_count": count,
                "correct_count": correct_count,
                "empirical_accuracy": empirical_acc,
                "mean_confidence": mean_conf,
                "calibration_error": cal_err,
            }

        ece = round(weighted_error_sum, 2)

        # Assessment
        if ece <= 5.0:
            calibration_quality = "WELL_CALIBRATED"
        elif ece <= 12.0:
            calibration_quality = "MODERATELY_CALIBRATED"
        else:
            calibration_quality = "POORLY_CALIBRATED"

        report = {
            "total_detections_analyzed": total_samples,
            "expected_calibration_error_pct": ece,
            "calibration_quality": calibration_quality,
            "bins": calibration_bins,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save calibration report
        dest_report = self.results_dir / f"calibration_report_{int(time.time())}.json"
        with open(dest_report, "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2)

        return report


model_comparator = ModelComparator()
