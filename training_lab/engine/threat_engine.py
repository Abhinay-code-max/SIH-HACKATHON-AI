"""
Deterministic Threat Scoring Engine & DEFCON Mapper.
Implements:
- assess_threat(track_state, spatial_events, behavior_features) -> ThreatAssessment
- score_to_defcon(score) -> int (1..5)
- Configurable factor weights, anti-double-counting, and human-readable explainability logs.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import uuid

from training_lab.engine.threat_config import (
    DEFCON_THRESHOLDS,
    THREAT_FACTOR_WEIGHTS,
    CLASS_CONTEXT_MODIFIERS,
    THREAT_SCORING_VERSION,
)


@dataclass
class ThreatAssessment:
    score: int
    defcon_level: int
    severity: str
    contributing_factors: List[Dict[str, Any]]
    explanation: str
    recommended_operator_action: str
    timestamp: str
    track_id: int
    confidence: float
    rule_version: str = THREAT_SCORING_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def score_to_defcon(score: Union[int, float]) -> int:
    """
    Maps threat score to DEFCON level (1..5).
    Score clamped between [0, 100].
    0–19   -> DEFCON 5 (Normal surveillance)
    20–39  -> DEFCON 4 (Elevated awareness)
    40–59  -> DEFCON 3 (Suspicious activity)
    60–79  -> DEFCON 2 (High-risk event)
    80–100 -> DEFCON 1 (Critical event)
    """
    s = min(100, max(0, int(round(score))))
    for defcon, (low, high) in DEFCON_THRESHOLDS.items():
        if low <= s <= high:
            return defcon
    return 5


def defcon_to_severity(defcon: int) -> str:
    mapping = {
        5: "LOW",
        4: "ELEVATED",
        3: "MEDIUM",
        2: "HIGH",
        1: "CRITICAL",
    }
    return mapping.get(defcon, "LOW")


class ThreatScoringEngine:
    """
    Deterministic compound threat scoring engine.
    """

    def __init__(self, weights: Optional[Dict[str, int]] = None):
        self.weights = weights or THREAT_FACTOR_WEIGHTS

    def assess_threat(
        self,
        track_state: Dict[str, Any],
        spatial_events: Optional[List[Dict[str, Any]]] = None,
        behavior_features: Optional[Dict[str, Any]] = None,
    ) -> ThreatAssessment:
        spatial_evts = spatial_events or []
        behav = behavior_features or {}

        track_id = int(track_state.get("track_id", 0))
        class_name = str(track_state.get("class_name", "unknown")).lower()
        confidence = float(track_state.get("confidence", 0.90))

        contributing_factors: List[Dict[str, Any]] = []
        raw_score = 0

        # Event type sets for deduplication
        spatial_types = {e.get("event_type") for e in spatial_evts}

        # 1. Boundary Crossing vs Movement toward boundary
        # Anti-double counting: Crossing boundary gets full +30; approaching gets +15 only if NOT crossed
        has_crossed_boundary = "CROSSED_BOUNDARY" in spatial_types
        is_moving_toward_boundary = (behav.get("direction_classification") == "toward_boundary") or (behav.get("is_approaching_border", False))

        if has_crossed_boundary:
            pts = self.weights.get("boundary_crossing", 30)
            raw_score += pts
            contributing_factors.append({
                "factor": "boundary_crossing",
                "points": pts,
                "description": "Crossed configured border boundary line",
            })
        elif is_moving_toward_boundary:
            pts = self.weights.get("movement_toward_boundary", 15)
            raw_score += pts
            contributing_factors.append({
                "factor": "movement_toward_boundary",
                "points": pts,
                "description": "Moving directionally toward border boundary",
            })

        # 2. Restricted Zone Intrusion
        has_entered_restricted = "ENTERED_RESTRICTED_ZONE" in spatial_types or (track_state.get("zone_type") in ("RESTRICTED", "RESTRICTED_ZONE", "CRITICAL"))
        if has_entered_restricted:
            pts = self.weights.get("restricted_zone_intrusion", 30)
            raw_score += pts
            contributing_factors.append({
                "factor": "restricted_zone_intrusion",
                "points": pts,
                "description": "Entered restricted perimeter zone",
            })

        # 3. Loitering
        is_loitering = behav.get("is_loitering", False) or ("LOITERING_IN_RESTRICTED_ZONE" in spatial_types)
        dwell_seconds = float(behav.get("dwell_seconds", track_state.get("dwell_seconds", 0.0)))
        if is_loitering:
            pts = self.weights.get("loitering", 15)
            raw_score += pts
            contributing_factors.append({
                "factor": "loitering",
                "points": pts,
                "description": f"Loitering for {dwell_seconds:.1f} seconds",
            })

        # 4. Prolonged Restricted Presence (distinct from loitering when moving inside restricted zone > 60s)
        has_prolonged_presence = "PROLONGED_PRESENCE" in spatial_types or (dwell_seconds >= 60.0 and has_entered_restricted)
        if has_prolonged_presence:
            pts = self.weights.get("prolonged_restricted_presence", 10)
            raw_score += pts
            contributing_factors.append({
                "factor": "prolonged_restricted_presence",
                "points": pts,
                "description": f"Prolonged restricted presence exceeding 60s ({dwell_seconds:.1f}s)",
            })

        # 5. Repeated Boundary Approach
        is_repeated_approach = behav.get("is_repeated_approach", False)
        if is_repeated_approach:
            pts = self.weights.get("repeated_boundary_approach", 10)
            raw_score += pts
            contributing_factors.append({
                "factor": "repeated_boundary_approach",
                "points": pts,
                "description": f"Repeated boundary approach cycles: {behav.get('repeated_approach_count', 2)}",
            })

        # 6. Abnormal Movement (Sudden velocity spike or rapid sprint)
        speed = float(behav.get("speed_px_s", track_state.get("speed_px_s", 0.0)))
        if speed >= 100.0:
            pts = self.weights.get("abnormal_movement", 10)
            raw_score += pts
            contributing_factors.append({
                "factor": "abnormal_movement",
                "points": pts,
                "description": f"High velocity / sprint displacement ({speed:.1f} px/s)",
            })

        # 7. Group / Proximity Context
        group_size = int(behav.get("group_size", 1))
        if group_size >= 3:
            pts = self.weights.get("group_proximity_context", 5)
            raw_score += pts
            contributing_factors.append({
                "factor": "group_proximity_context",
                "points": pts,
                "description": f"Group proximity context with {group_size} associated tracks",
            })

        # 8. Object Class Prior / Context Modifier
        class_modifier = CLASS_CONTEXT_MODIFIERS.get(class_name, 0)
        # Contextual check: backpack / bag alone is neutral unless stationary in restricted zone
        if class_name in ("backpack", "bag"):
            if has_entered_restricted and dwell_seconds >= 10.0:
                raw_score += 15
                contributing_factors.append({
                    "factor": "unattended_baggage_in_restricted_zone",
                    "points": 15,
                    "description": f"Unattended {class_name} isolated in restricted zone",
                })
        elif class_modifier != 0:
            raw_score += class_modifier
            contributing_factors.append({
                "factor": "class_context_modifier",
                "points": class_modifier,
                "description": f"Context modifier for class '{class_name}' ({class_modifier:+d} pts)",
            })

        # Final Score Normalization / Clamping
        final_score = min(100, max(0, int(round(raw_score))))
        defcon = score_to_defcon(final_score)
        severity = defcon_to_severity(defcon)

        # Human-Readable Explanation
        if contributing_factors:
            factor_lines = [f"{f['points']:+d} {f['description']}" for f in contributing_factors]
            explanation_str = f"Track {track_id} ({class_name}): " + "; ".join(factor_lines)
        else:
            explanation_str = f"Track {track_id} ({class_name}): Normal baseline activity. No threat factors detected."

        # Operator Recommended Action
        if defcon == 1:
            action = "Immediate tactical escalation and physical intercept required."
        elif defcon == 2:
            action = "High-risk perimeter alert: immediate operator verification and dispatch."
        elif defcon == 3:
            action = "Operator review required: track suspicious movement patterns."
        elif defcon == 4:
            action = "Elevated surveillance awareness: maintain visual tracking."
        else:
            action = "Routine automated surveillance: no immediate action required."

        return ThreatAssessment(
            score=final_score,
            defcon_level=defcon,
            severity=severity,
            contributing_factors=contributing_factors,
            explanation=explanation_str,
            recommended_operator_action=action,
            timestamp=datetime.now(timezone.utc).isoformat(),
            track_id=track_id,
            confidence=confidence,
            rule_version=THREAT_SCORING_VERSION,
        )


threat_scoring_engine = ThreatScoringEngine()
