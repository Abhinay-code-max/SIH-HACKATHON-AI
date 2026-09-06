"""
Automatic Regression Testing & Deployment Gatekeeper.
Strictly assesses whether a candidate model regresses against the active approved baseline.
Enforces that a model cannot be promoted or approved if mAP drops, false alarms increase,
missed breaches increase, or critical perimeter defense rules fail.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RESULTS_DIR = ROOT_DIR / "training_lab" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class RegressionGateFailureError(Exception):
    """Raised when a candidate model fails regression testing against the active baseline."""
    pass


class RegressionGate:
    """
    Automated regression test gate enforcing strict quality thresholds
    prior to model approval or deployment.
    """

    def __init__(self, results_dir: Optional[Union[str, Path]] = None):
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR

    def evaluate_regression(
        self,
        candidate_eval: Dict[str, Any],
        baseline_eval: Dict[str, Any],
        critical_scenarios: Optional[List[str]] = None,
        max_map_drop_tolerance: float = 0.005,
        min_camera_map_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Runs automated regression tests comparing Candidate vs Baseline.

        Gates:
          1. Delta mAP50 >= -tolerance
          2. Delta False Positives <= 0
          3. Delta False Negatives <= 0
          4. Per-Camera mAP >= threshold
          5. Zero critical scenario regressions (e.g. SCN_01 fence breaches)
        """
        critical_scenarios = critical_scenarios or ["SCN_01"]

        cand_map = candidate_eval.get("mAP50", candidate_eval.get("metrics", {}).get("mAP50", 0.0)) or 0.0
        base_map = baseline_eval.get("mAP50", baseline_eval.get("metrics", {}).get("mAP50", 0.0)) or 0.0
        delta_map = float(cand_map - base_map)

        cand_fp = int(candidate_eval.get("false_positives", candidate_eval.get("metrics", {}).get("false_positives", 0)))
        base_fp = int(baseline_eval.get("false_positives", baseline_eval.get("metrics", {}).get("false_positives", 0)))
        delta_fp = cand_fp - base_fp

        cand_fn = int(candidate_eval.get("false_negatives", candidate_eval.get("metrics", {}).get("false_negatives", 0)))
        base_fn = int(baseline_eval.get("false_negatives", baseline_eval.get("metrics", {}).get("false_negatives", 0)))
        delta_fn = cand_fn - base_fn

        failures: List[str] = []

        # Gate 1: mAP50 Regression Check
        if delta_map < -max_map_drop_tolerance:
            failures.append(
                f"Overall mAP50 regressed by {delta_map:+.4f} (Candidate: {cand_map:.4f}, Baseline: {base_map:.4f})"
            )

        # Gate 2: False Positive Surge Check
        if delta_fp > 0:
            failures.append(
                f"False positive alarms increased by +{delta_fp} (Candidate: {cand_fp}, Baseline: {base_fp})"
            )

        # Gate 3: False Negative Missed Breach Check
        if delta_fn > 0:
            failures.append(
                f"False negative missed breaches increased by +{delta_fn} (Candidate: {cand_fn}, Baseline: {base_fn})"
            )

        # Gate 4: Per-Camera Minimum Performance Check
        cand_per_cam = candidate_eval.get("per_camera", {})
        for cam_id, cam_metrics in cand_per_cam.items():
            cam_map = cam_metrics.get("mAP50", 1.0)
            if cam_map is not None and cam_map < min_camera_map_threshold:
                failures.append(
                    f"Camera '{cam_id}' failed minimum mAP threshold: {cam_map:.3f} < {min_camera_map_threshold:.3f}"
                )

        # Gate 5: Critical Security Scenario Check
        cand_scenarios = candidate_eval.get("scenarios", {})
        base_scenarios = baseline_eval.get("scenarios", {})
        for scn_id in critical_scenarios:
            if scn_id in cand_scenarios and scn_id in base_scenarios:
                c_scn = cand_scenarios[scn_id]
                b_scn = base_scenarios[scn_id]
                c_fn = c_scn.get("false_negatives", 0)
                b_fn = b_scn.get("false_negatives", 0)
                if c_fn > b_fn:
                    failures.append(
                        f"Critical security scenario '{scn_id}' regressed: missed detections increased from {b_fn} to {c_fn}"
                    )

        passed = len(failures) == 0
        verdict = "REGRESSION_GATE_PASSED" if passed else "REGRESSION_GATE_FAILED"

        cand_ver = candidate_eval.get("model_version", "candidate")
        base_ver = baseline_eval.get("model_version", "baseline")

        report: Dict[str, Any] = {
            "candidate_model": cand_ver,
            "baseline_model": base_ver,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "passed": passed,
            "deltas": {
                "delta_mAP50": round(delta_map, 4),
                "delta_fp": delta_fp,
                "delta_fn": delta_fn,
            },
            "candidate_summary": {
                "mAP50": round(cand_map, 4),
                "fp": cand_fp,
                "fn": cand_fn,
            },
            "baseline_summary": {
                "mAP50": round(base_map, 4),
                "fp": base_fp,
                "fn": base_fn,
            },
            "failures_count": len(failures),
            "failures": failures,
            "policy_rule": "Model cannot be deployed or approved when verdict is REGRESSION_GATE_FAILED",
        }

        # Persist report
        report_file = self.results_dir / f"regression_gate_{cand_ver}_vs_{base_ver}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def assert_passed(
        self,
        candidate_eval: Dict[str, Any],
        baseline_eval: Dict[str, Any],
        critical_scenarios: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Asserts candidate passes regression test; raises RegressionGateFailureError if not."""
        report = self.evaluate_regression(candidate_eval, baseline_eval, critical_scenarios)
        if not report.get("passed", False):
            raise RegressionGateFailureError(
                f"Candidate '{report.get('candidate_model')}' failed regression test: {report.get('failures')}"
            )
        return report


regression_gate = RegressionGate()
