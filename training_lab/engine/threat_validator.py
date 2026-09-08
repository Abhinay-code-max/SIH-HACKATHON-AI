"""
Compound Threat Assessment Engine & Human Disagreement Validator.
Integrates deterministic DEFCON 5–1 scoring, explainable contributing factor logs,
and preserves human validation disagreement metrics and confusion matrices.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid
from typing import Any, Dict, List, Optional, Union

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.threat_engine import threat_scoring_engine, score_to_defcon, ThreatAssessment
from training_lab.engine.threat_config import THREAT_FACTOR_WEIGHTS, DEFCON_THRESHOLDS

ANNOTATIONS_DIR = ROOT_DIR / "training_lab" / "annotations"
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
THREAT_VALIDATIONS_FILE = ANNOTATIONS_DIR / "threat_validations.json"


class ThreatValidator:
    """
    Computes compound multi-factor threat scores and monitors human threat validations.
    """

    ZONE_SEVERITY_POINTS = {
        "SAFE": 0,
        "NORMAL_ZONE": 0,
        "WARNING": 20,
        "RESTRICTED": 40,
        "RESTRICTED_ZONE": 40,
        "CRITICAL": 60,
    }

    LEVEL_THRESHOLDS = [
        (85, "CRITICAL"),
        (60, "HIGH"),
        (30, "MEDIUM"),
        (0, "LOW"),
    ]

    def __init__(self, validations_file: Optional[Path] = None):
        self.validations_file = validations_file or THREAT_VALIDATIONS_FILE
        self._ensure_file()

    def _ensure_file(self):
        if not self.validations_file.exists():
            with open(self.validations_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def assess_threat(
        self,
        class_name: str,
        zone_type: Optional[str],
        is_moving_inward: bool,
        dwell_seconds: float,
        cluster_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Computes compound threat score:
        Score = Baseline(15) + ZonePts + DirectionPts + DwellPts + ClusterPts
        Clamped to [0, 100].
        Provides backward compatibility for existing test suite while mapping to DEFCON.
        """
        # 1. Baseline
        baseline = 15

        # 2. Zone Points
        z_type = (zone_type or "").upper()
        zone_pts = self.ZONE_SEVERITY_POINTS.get(z_type, 0)

        # 3. Trajectory Direction Points
        direction_pts = 20 if is_moving_inward else 0

        # 4. Dwell Persistence Points
        if dwell_seconds >= 10.0:
            dwell_pts = 20
        elif dwell_seconds >= 5.0:
            dwell_pts = 10
        else:
            dwell_pts = 0

        # 5. Cluster Density Points
        cluster_pts = 15 if cluster_count >= 2 else 0

        # Calculate Total Score
        raw_score = baseline + zone_pts + direction_pts + dwell_pts + cluster_pts
        score = max(0, min(100, raw_score))

        # Map to Threat Level
        threat_level = "LOW"
        for thresh, lvl in self.LEVEL_THRESHOLDS:
            if score >= thresh:
                threat_level = lvl
                break

        defcon = score_to_defcon(score)
        assessment_id = f"th_{uuid.uuid4().hex[:8]}"

        return {
            "assessment_id": assessment_id,
            "threat_score": score,
            "threat_level": threat_level,
            "defcon_level": defcon,
            "class_name": class_name,
            "zone_type": z_type or "UNASSIGNED",
            "factors": {
                "baseline_pts": baseline,
                "zone_pts": zone_pts,
                "direction_pts": direction_pts,
                "dwell_pts": dwell_pts,
                "cluster_pts": cluster_pts,
            },
            "is_moving_inward": is_moving_inward,
            "dwell_seconds": dwell_seconds,
            "cluster_count": cluster_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def assess_comprehensive_threat(
        self,
        track_state: Dict[str, Any],
        spatial_events: Optional[List[Dict[str, Any]]] = None,
        behavior_features: Optional[Dict[str, Any]] = None,
    ) -> ThreatAssessment:
        """Invokes the standard ThreatScoringEngine for comprehensive DEFCON assessment."""
        return threat_scoring_engine.assess_threat(
            track_state=track_state,
            spatial_events=spatial_events,
            behavior_features=behavior_features,
        )

    def validate_threat_assessment(
        self,
        assessment_id: str,
        ai_level: str,
        human_level: str,
        is_correct: bool,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Records human validation feedback on an AI threat level assessment.
        """
        record = {
            "validation_id": f"tval_{uuid.uuid4().hex[:8]}",
            "assessment_id": assessment_id,
            "ai_level": ai_level.upper(),
            "human_level": human_level.upper(),
            "is_correct": is_correct,
            "notes": notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            records = []
            if self.validations_file.exists():
                with open(self.validations_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            records.append(record)
            with open(self.validations_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception:
            pass

        return record

    def compute_threat_validation_metrics(self) -> Dict[str, Any]:
        """
        Calculates validation agreement percentage, confusion matrix, and disagreement logs.
        """
        records: List[Dict[str, Any]] = []
        if self.validations_file.exists():
            try:
                with open(self.validations_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = []

        total = len(records)
        if total == 0:
            return {
                "total_validations": 0,
                "agreement_pct": 100.0,
                "confusion_matrix": {},
                "disagreements": [],
            }

        correct_count = sum(1 for r in records if r.get("is_correct", False))
        agreement_pct = round((correct_count / total) * 100, 2)

        levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        confusion_matrix: Dict[str, Dict[str, int]] = {
            ai_lvl: {h_lvl: 0 for h_lvl in levels} for ai_lvl in levels
        }

        disagreements: List[Dict[str, Any]] = []

        for r in records:
            ai_l = r.get("ai_level", "").upper()
            hu_l = r.get("human_level", "").upper()

            if ai_l in confusion_matrix and hu_l in confusion_matrix[ai_l]:
                confusion_matrix[ai_l][hu_l] += 1

            if not r.get("is_correct", False) or ai_l != hu_l:
                disagreements.append(r)

        return {
            "total_validations": total,
            "agreement_count": correct_count,
            "agreement_pct": agreement_pct,
            "confusion_matrix": confusion_matrix,
            "disagreements": disagreements,
        }


threat_validator = ThreatValidator()
