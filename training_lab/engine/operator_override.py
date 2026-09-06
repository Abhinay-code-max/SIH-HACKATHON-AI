"""
Operator Threat Override Engine & Unified Event Timeline.
Allows authorized human operators to manually override automated threat levels,
maintaining an immutable audit log, and records chronological security events
across detection, tracking, perimeter transitions, and validations.
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

RESULTS_DIR = ROOT_DIR / "training_lab" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OVERRIDES_FILE = RESULTS_DIR / "operator_overrides.json"
TIMELINE_FILE = RESULTS_DIR / "event_timeline.json"


class OperatorOverrideEngine:
    """
    Manages manual threat overrides by command personnel with audit logging.
    """

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or OVERRIDES_FILE
        self.active_override: Optional[Dict[str, Any]] = None
        self._ensure_file()

    def _ensure_file(self):
        if not self.log_file.exists():
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def set_override(
        self,
        automatic_level: str,
        override_level: str,
        reason: str,
        operator: str = "Commander",
    ) -> Dict[str, Any]:
        """
        Applies a manual threat override, replacing automatic assessment.
        """
        record = {
            "override_id": f"ovr_{uuid.uuid4().hex[:8]}",
            "automatic_level": automatic_level.upper(),
            "override_level": override_level.upper(),
            "reason": reason,
            "operator": operator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_active": True,
        }

        self.active_override = record

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

        return record

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
        """
        Appends an event to the timeline with microsecond timestamp.
        """
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
        """
        Retrieves timeline events sorted chronologically up to limit.
        """
        if not self.timeline_file.exists():
            return []

        try:
            with open(self.timeline_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            # Sort chronologically by timestamp
            sorted_records = sorted(records, key=lambda r: r.get("timestamp", ""))
            return sorted_records[-limit:]
        except Exception:
            return []

    def clear_timeline(self) -> None:
        """Empties the event timeline."""
        with open(self.timeline_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


operator_override_engine = OperatorOverrideEngine()
event_timeline = EventTimeline()
