"""
Canonical Threat Event Schema and Serialization Engine.
Strictly local, deterministic JSON serialization.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

from training_lab.engine.threat_config import THREAT_SCORING_VERSION


@dataclass
class ThreatEvent:
    event_id: str
    timestamp: str
    camera_id: str
    track_id: int
    object_class: str
    object_confidence: float

    behavior_features: Dict[str, Any]
    spatial_events: List[Dict[str, Any]]

    threat_score: int
    defcon_level: int
    severity: str

    contributing_factors: List[Dict[str, Any]]
    explanation: str

    recommended_action: str

    operator_status: str = "PENDING"  # PENDING, ACKNOWLEDGED, OVERRIDDEN, DISMISSED
    operator_override: Optional[Dict[str, Any]] = None
    operator_note: str = ""

    scoring_version: str = THREAT_SCORING_VERSION

    @property
    def logical_track_id(self) -> str:
        """Global collision-free identity across multiple camera sensors."""
        return f"{self.camera_id}:{self.track_id}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["logical_track_id"] = self.logical_track_id
        return data

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreatEvent":
        clean_data = dict(data)
        clean_data.pop("logical_track_id", None)
        return cls(**clean_data)

    @classmethod
    def from_json(cls, json_str: str) -> "ThreatEvent":
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class OperatorFeedbackRecord:
    feedback_id: str
    event_id: str
    camera_id: str
    track_id: int
    automated_score: int
    automated_defcon: int
    operator_action: str  # acknowledge, mark_false_positive, confirm_threat, downgrade_threat, escalate_threat, dismiss
    operator_final_score: Optional[int]
    operator_final_defcon: Optional[int]
    operator_final_assessment: str
    operator_note: str
    event_features: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
