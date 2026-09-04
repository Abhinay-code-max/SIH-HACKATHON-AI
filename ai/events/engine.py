"""
Local Rule-Based Event Engine.
Decoupled from the AI detection model.
Turns raw detections into high-level security/surveillance events with severity and alerts.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.models.contracts import (
    EventType,
    GeoLocation,
    RawDetection,
    SecurityEvent,
    SeverityLevel,
)


class EventEngine:
    """
    Evaluates raw detections against configurable security rules:
    1. High Crowd Density Rule (e.g., >= 3 persons)
    2. Restricted Vehicle Intrusion Rule (vehicles in pedestrian/restricted areas)
    3. High-Confidence Object Detection
    Applies cooldowns to avoid duplicate event flooding.
    """

    def __init__(
        self,
        camera_id: str = "CAM_01",
        default_location: Optional[GeoLocation] = None,
        crowd_threshold: int = 3,
        alert_cooldown_sec: float = 2.0,
    ):
        self.camera_id = camera_id
        self.location = default_location or GeoLocation(lat=17.3850, lon=78.4867, zone_name="Main Plaza")
        self.crowd_threshold = crowd_threshold
        self.alert_cooldown_sec = alert_cooldown_sec
        self.last_event_times: Dict[str, float] = {}
        self.event_counter = 0
        self.event_history: List[SecurityEvent] = []

    def _should_suppress(self, rule_key: str, now: float) -> bool:
        """Check if an event rule is within cooldown window."""
        last_time = self.last_event_times.get(rule_key, 0.0)
        if (now - last_time) < self.alert_cooldown_sec:
            return True
        self.last_event_times[rule_key] = now
        return False

    def process_frame_detections(
        self,
        raw_detections: List[Dict[str, Any]],
        frame_index: int = 0,
        timestamp_iso: Optional[str] = None,
    ) -> List[SecurityEvent]:
        """
        Ingests a list of raw detections from one frame and evaluates rules.
        Returns a list of newly triggered SecurityEvent objects.
        """
        now = time.time()
        ts = timestamp_iso or datetime.now(timezone.utc).isoformat()
        generated_events: List[SecurityEvent] = []

        # Count occurrences per class
        class_counts: Dict[str, int] = {}
        person_detections = []
        vehicle_detections = []

        for det in raw_detections:
            cls = det.get("class", det.get("class_name", "unknown"))
            class_counts[cls] = class_counts.get(cls, 0) + 1
            if cls == "person":
                person_detections.append(det)
            elif cls in {"car", "bus", "truck", "motorcycle"}:
                vehicle_detections.append(det)

        # -------------------------------------------------------------
        # RULE 1: Crowd Density Alert
        # -------------------------------------------------------------
        person_count = class_counts.get("person", 0)
        if person_count >= self.crowd_threshold:
            rule_key = f"{self.camera_id}_crowd_density"
            if not self._should_suppress(rule_key, now):
                self.event_counter += 1
                evt = SecurityEvent(
                    event_id=f"evt_{self.event_counter:04d}",
                    timestamp=ts,
                    source_id=self.camera_id,
                    event_type=EventType.CROWD_DENSITY,
                    class_name="person_group",
                    confidence=0.92,
                    bbox=[0.0, 0.0, 640.0, 480.0],  # covers zone
                    severity=SeverityLevel.HIGH if person_count >= (self.crowd_threshold + 2) else SeverityLevel.MEDIUM,
                    location=self.location,
                    metadata={
                        "rule": "CROWD_DENSITY_EXCEEDED",
                        "detected_count": person_count,
                        "threshold": self.crowd_threshold,
                        "frame_index": frame_index,
                    },
                )
                generated_events.append(evt)
                self.event_history.append(evt)

        # -------------------------------------------------------------
        # RULE 2: Heavy Vehicle / Vehicle Alert
        # -------------------------------------------------------------
        for v in vehicle_detections:
            v_cls = v.get("class", v.get("class_name"))
            rule_key = f"{self.camera_id}_vehicle_{v_cls}"
            if not self._should_suppress(rule_key, now):
                self.event_counter += 1
                severity = SeverityLevel.CRITICAL if v_cls in {"bus", "truck"} else SeverityLevel.MEDIUM
                evt = SecurityEvent(
                    event_id=f"evt_{self.event_counter:04d}",
                    timestamp=ts,
                    source_id=self.camera_id,
                    event_type=EventType.UNAUTHORIZED_VEHICLE,
                    class_name=v_cls,
                    confidence=float(v.get("confidence", 0.9)),
                    bbox=v.get("bbox", [0, 0, 0, 0]),
                    severity=severity,
                    location=self.location,
                    metadata={
                        "rule": "VEHICLE_SECURITY_EVENT",
                        "vehicle_type": v_cls,
                        "frame_index": frame_index,
                    },
                )
                generated_events.append(evt)
                self.event_history.append(evt)

        # -------------------------------------------------------------
        # RULE 3: Individual Person Detection (Informational / Low)
        # -------------------------------------------------------------
        if person_count < self.crowd_threshold and person_detections:
            rule_key = f"{self.camera_id}_single_person"
            if not self._should_suppress(rule_key, now):
                p = person_detections[0]
                self.event_counter += 1
                evt = SecurityEvent(
                    event_id=f"evt_{self.event_counter:04d}",
                    timestamp=ts,
                    source_id=self.camera_id,
                    event_type=EventType.OBJECT_DETECTED,
                    class_name="person",
                    confidence=float(p.get("confidence", 0.85)),
                    bbox=p.get("bbox", [0, 0, 0, 0]),
                    severity=SeverityLevel.LOW,
                    location=self.location,
                    metadata={"rule": "STANDARD_OBJECT_DETECTION", "frame_index": frame_index},
                )
                generated_events.append(evt)
                self.event_history.append(evt)

        return generated_events

    def export_events(self, output_file: Path | str) -> Path:
        out_p = Path(output_file).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        serialized = [e.model_dump() for e in self.event_history]
        with open(out_p, "w") as f:
            json.dump(serialized, f, indent=2)
        return out_p
