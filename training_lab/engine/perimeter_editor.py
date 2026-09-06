"""
Perimeter Zone State Machine & Validation Engine.
Manages polygonal zones (SAFE, WARNING, RESTRICTED, CRITICAL), tracks track transitions
(ENTER, INSIDE, EXIT) using OpenCV point-polygon hit testing, and validates perimeter decisions.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import sys
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ANNOTATIONS_DIR = ROOT_DIR / "training_lab" / "annotations"
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
VALIDATIONS_FILE = ANNOTATIONS_DIR / "perimeter_validations.json"


class ZoneType(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    RESTRICTED = "RESTRICTED"
    CRITICAL = "CRITICAL"


class ZoneTransition(str, Enum):
    ENTER = "ENTER"
    INSIDE = "INSIDE"
    EXIT = "EXIT"
    OUTSIDE = "OUTSIDE"


@dataclass
class LabPerimeterZone:
    zone_id: str
    name: str
    zone_type: ZoneType
    camera_id: str
    polygon: List[Tuple[float, float]]  # List of [x, y] coordinates
    color: Optional[str] = None


@dataclass
class PerimeterEvent:
    event_id: str
    camera_id: str
    track_id: int
    class_name: str
    zone_id: str
    zone_type: str
    transition: ZoneTransition
    ground_point: Tuple[float, float]
    frame_idx: int
    timestamp: str


@dataclass
class PerimeterValidation:
    validation_id: str
    event_id: str
    is_correct: bool
    human_decision: str
    ai_decision: str
    notes: str
    timestamp: str


class PerimeterEngine:
    """
    Perimeter zone geometry manager and track transition state machine.
    Tracks ENTER, INSIDE, and EXIT transitions per camera and track ID.
    """

    def __init__(self, validations_file: Optional[Path] = None):
        self.zones: Dict[str, LabPerimeterZone] = {}
        # active_tracks maps (camera_id, track_id) -> set of active zone_ids
        self.active_tracks: Dict[Tuple[str, int], Set[str]] = {}
        self.validations_file = validations_file or VALIDATIONS_FILE
        self._ensure_validations_file()

    def _ensure_validations_file(self):
        if not self.validations_file.exists():
            with open(self.validations_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def add_zone(self, zone: LabPerimeterZone) -> None:
        """Registers a perimeter zone."""
        self.zones[zone.zone_id] = zone

    def get_zones(self, camera_id: Optional[str] = None) -> List[LabPerimeterZone]:
        """Returns zones, optionally filtered by camera ID."""
        if camera_id is None:
            return list(self.zones.values())
        return [
            z for z in self.zones.values()
            if z.camera_id == camera_id or z.camera_id == "ALL"
        ]

    def clear_state(self) -> None:
        """Resets active track state."""
        self.active_tracks.clear()

    def evaluate_track(
        self,
        camera_id: str,
        track_id: int,
        class_name: str,
        ground_point: Tuple[float, float],
        frame_idx: int = 0,
        timestamp: Optional[str] = None,
    ) -> List[PerimeterEvent]:
        """
        Evaluates track position against all relevant zones for camera_id.
        Emits ENTER, INSIDE, and EXIT events according to state machine transitions.
        """
        now_ts = timestamp or datetime.now(timezone.utc).isoformat()
        track_key = (camera_id, track_id)
        active_zones = self.active_tracks.setdefault(track_key, set())

        events: List[PerimeterEvent] = []
        applicable_zones = self.get_zones(camera_id)

        gx, gy = float(ground_point[0]), float(ground_point[1])
        current_in_zones: Set[str] = set()

        for zone in applicable_zones:
            poly_np = np.array(zone.polygon, dtype=np.float32)
            # cv2.pointPolygonTest: >= 0 means inside or on boundary, < 0 means outside
            dist = cv2.pointPolygonTest(poly_np, (gx, gy), measureDist=False)
            is_inside = dist >= 0

            was_inside = zone.zone_id in active_zones

            if is_inside:
                current_in_zones.add(zone.zone_id)
                if not was_inside:
                    # Transition: ENTER
                    active_zones.add(zone.zone_id)
                    events.append(
                        PerimeterEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:8]}",
                            camera_id=camera_id,
                            track_id=track_id,
                            class_name=class_name,
                            zone_id=zone.zone_id,
                            zone_type=zone.zone_type.value if hasattr(zone.zone_type, "value") else str(zone.zone_type),
                            transition=ZoneTransition.ENTER,
                            ground_point=(gx, gy),
                            frame_idx=frame_idx,
                            timestamp=now_ts,
                        )
                    )
                else:
                    # Transition: INSIDE
                    events.append(
                        PerimeterEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:8]}",
                            camera_id=camera_id,
                            track_id=track_id,
                            class_name=class_name,
                            zone_id=zone.zone_id,
                            zone_type=zone.zone_type.value if hasattr(zone.zone_type, "value") else str(zone.zone_type),
                            transition=ZoneTransition.INSIDE,
                            ground_point=(gx, gy),
                            frame_idx=frame_idx,
                            timestamp=now_ts,
                        )
                    )
            else:
                if was_inside:
                    # Transition: EXIT
                    active_zones.remove(zone.zone_id)
                    events.append(
                        PerimeterEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:8]}",
                            camera_id=camera_id,
                            track_id=track_id,
                            class_name=class_name,
                            zone_id=zone.zone_id,
                            zone_type=zone.zone_type.value if hasattr(zone.zone_type, "value") else str(zone.zone_type),
                            transition=ZoneTransition.EXIT,
                            ground_point=(gx, gy),
                            frame_idx=frame_idx,
                            timestamp=now_ts,
                        )
                    )

        return events

    def validate_perimeter_decision(
        self,
        event_id: str,
        is_correct: bool,
        human_decision: str,
        notes: str = "",
        ai_decision: str = "BREACH",
    ) -> PerimeterValidation:
        """
        Records human validation feedback for a perimeter event and persists it to disk.
        """
        val_obj = PerimeterValidation(
            validation_id=f"pval_{uuid.uuid4().hex[:8]}",
            event_id=event_id,
            is_correct=is_correct,
            human_decision=human_decision,
            ai_decision=ai_decision,
            notes=notes,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        try:
            records = []
            if self.validations_file.exists():
                with open(self.validations_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            records.append(asdict(val_obj))
            with open(self.validations_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception:
            pass

        return val_obj


perimeter_engine = PerimeterEngine()
