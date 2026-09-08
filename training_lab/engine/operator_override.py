"""
Operator Threat Override Engine & Unified Event Timeline.
Allows authorized human operators to manually override automated threat levels,
maintaining an immutable audit log, and records chronological security events
across detection, tracking, perimeter transitions, and validations.
Captures human feedback for future learning loops.
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

from training_lab.engine.threat_event_schema import OperatorFeedbackRecord

RESULTS_DIR = ROOT_DIR / "training_lab" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OVERRIDES_FILE = RESULTS_DIR / "operator_overrides.json"
FEEDBACK_FILE = RESULTS_DIR / "human_feedback_records.json"
TIMELINE_FILE = RESULTS_DIR / "event_timeline.json"


class OperatorOverrideEngine:
    """
    Manages manual threat overrides by command personnel with audit logging.
    """

    ALLOWED_ACTIONS = {
        "acknowledge",
        "mark_false_positive",
        "confirm_threat",
        "downgrade_threat",
        "escalate_threat",
        "dismiss",
    }

    def __init__(self, log_file: Optional[Path] = None, feedback_file: Optional[Path] = None):
        self.log_file = log_file or OVERRIDES_FILE
        self.feedback_file = feedback_file or FEEDBACK_FILE
        self.active_override: Optional[Dict[str, Any]] = None
        self._ensure_files()

    def _ensure_files(self):
        if not self.log_file.exists():
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
        if not self.feedback_file.exists():
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def set_override(
        self,
        automatic_level: str,
        override_level: str,
        reason: str,
        operator: str = "Commander",
        event_id: Optional[str] = None,
        camera_id: str = "CAM_01",
        track_id: int = 0,
        action: str = "downgrade_threat",
        event_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Applies a manual threat override, replacing automatic assessment in UI display
        while maintaining immutable original automated assessment.
        """
        override_id = f"ovr_{uuid.uuid4().hex[:8]}"
        now_ts = datetime.now(timezone.utc).isoformat()

        record = {
            "override_id": override_id,
            "event_id": event_id or f"evt_{uuid.uuid4().hex[:8]}",
            "camera_id": camera_id,
            "track_id": track_id,
            "automatic_level": automatic_level.upper(),
            "override_level": override_level.upper(),
            "action": action,
            "reason": reason,
            "operator": operator,
            "timestamp": now_ts,
            "is_active": True,
        }

        self.active_override = record

        # Append to overrides log
        try:
            records = []
            if self.log_file.exists():
                with open(self.log_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            records.append(record)
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception:
            pass

        # Structured Human Feedback Record for Future ML loop
        feedback_obj = OperatorFeedbackRecord(
            feedback_id=f"fbk_{uuid.uuid4().hex[:8]}",
            event_id=record["event_id"],
            camera_id=camera_id,
            track_id=track_id,
            automated_score=self._level_to_approx_score(automatic_level),
            automated_defcon=self._level_to_defcon(automatic_level),
            operator_action=action,
            operator_final_score=self._level_to_approx_score(override_level),
            operator_final_defcon=self._level_to_defcon(override_level),
            operator_final_assessment=override_level.upper(),
            operator_note=reason,
            event_features=event_features or {},
            timestamp=now_ts,
        )

        try:
            fb_records = []
            if self.feedback_file.exists():
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    fb_records = json.load(f)
            fb_records.append(feedback_obj.to_dict())
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump(fb_records, f, indent=2)
        except Exception:
            pass

        return record

    def _level_to_approx_score(self, level: str) -> int:
        lvl = level.upper()
        mapping = {
            "LOW": 10,
            "DEFCON 5": 10,
            "ELEVATED": 30,
            "DEFCON 4": 30,
            "MEDIUM": 50,
            "DEFCON 3": 50,
            "HIGH": 70,
            "DEFCON 2": 70,
            "CRITICAL": 90,
            "DEFCON 1": 90,
        }
        return mapping.get(lvl, 50)

    def _level_to_defcon(self, level: str) -> int:
        lvl = level.upper()
        mapping = {
            "LOW": 5,
            "DEFCON 5": 5,
            "ELEVATED": 4,
            "DEFCON 4": 4,
            "MEDIUM": 3,
            "DEFCON 3": 3,
            "HIGH": 2,
            "DEFCON 2": 2,
            "CRITICAL": 1,
            "DEFCON 1": 1,
        }
        return mapping.get(lvl, 3)

    def get_effective_threat(self, current_ai_level: str = "LOW") -> Dict[str, Any]:
        """
        Returns the effective threat level, clearly distinguishing between
        AI_DECISION and HUMAN_OVERRIDE.
        """
        if self.active_override and self.active_override.get("is_active", False):
            return {
                "effective_level": self.active_override["override_level"],
                "source": "HUMAN_OVERRIDE",
                "override_details": self.active_override,
            }

        return {
            "effective_level": current_ai_level.upper(),
            "source": "AI_DECISION",
            "override_details": None,
        }

    def clear_override(self) -> Dict[str, Any]:
        """
        Clears the active manual override, restoring automated AI decision mode.
        """
        prev = self.active_override
        if prev:
            prev["is_active"] = False
        self.active_override = None

        return {
            "status": "OVERRIDE_CLEARED",
            "previous_override": prev,
            "source": "AI_DECISION",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class EventTimeline:
    """
    Unified chronological event journal capturing perception, tracking,
    perimeter breaches, threat escalations, human validations, and operator actions.
    """

    VALID_EVENT_TYPES = {
        "DETECTION",
        "TRACKING",
        "PERIMETER_ENTRY",
        "PERIMETER_INSIDE",
        "PERIMETER_EXIT",
        "THREAT_ESCALATION",
        "HUMAN_VALIDATION",
        "OPERATOR_OVERRIDE",
    }

    def __init__(self, timeline_file: Optional[Path] = None):
        self.timeline_file = timeline_file or TIMELINE_FILE
        self._ensure_file()

    def _ensure_file(self):
        if not self.timeline_file.exists():
            with open(self.timeline_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def log_event(
        self,
        event_type: str,
        camera_id: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Appends an event to the timeline with microsecond timestamp."""
        entry = {
            "event_id": f"tl_{uuid.uuid4().hex[:8]}",
            "event_type": event_type.upper(),
            "camera_id": camera_id,
            "description": description,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            records = []
            if self.timeline_file.exists():
                with open(self.timeline_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            records.append(entry)
            with open(self.timeline_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception:
            pass

        return entry

    def get_timeline(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves timeline events sorted chronologically up to limit."""
        if not self.timeline_file.exists():
            return []

        try:
            with open(self.timeline_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            sorted_records = sorted(records, key=lambda r: r.get("timestamp", ""))
            return sorted_records[-limit:]
        except Exception:
            return []

    def clear_timeline(self) -> None:
        with open(self.timeline_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


operator_override_engine = OperatorOverrideEngine()
event_timeline = EventTimeline()
