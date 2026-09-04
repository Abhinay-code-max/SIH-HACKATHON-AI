"""
Local Event Store Service for FastAPI.
Reads and updates security events locally.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.models.contracts import SecurityEvent, SeverityLevel


class EventService:
    def __init__(self, data_file: Path | None = None):
        if data_file is None:
            self.data_file = ROOT_DIR / "data" / "sample-events" / "live_events.json"
        else:
            self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

    def get_events(
        self,
        severity: Optional[SeverityLevel] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve stored events from local storage."""
        if not self.data_file.is_file():
            return []

        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                events = json.load(f)
        except Exception:
            return []

        if severity:
            events = [e for e in events if e.get("severity") == severity.value]

        return events[-limit:]

    def add_event(self, event: SecurityEvent) -> Dict[str, Any]:
        """Add a newly generated event to local storage."""
        events = self.get_events(limit=1000)
        serialized = event.model_dump()
        events.append(serialized)

        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)

        return serialized


event_service = EventService()
