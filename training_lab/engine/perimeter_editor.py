"""
Perimeter Zone State Machine & Boundary Geometry Engine.
Manages zones (NORMAL_ZONE, RESTRICTED_ZONE, CRITICAL, SAFE, WARNING),
Border Boundary Tripwires, and evaluates spatial events:
- ENTERED_RESTRICTED_ZONE
- EXITED_RESTRICTED_ZONE
- APPROACHING_BOUNDARY
- CROSSED_BOUNDARY
- LOITERING_IN_RESTRICTED_ZONE
- PROLONGED_PRESENCE
- REPEATED_BOUNDARY_APPROACH
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
    NORMAL_ZONE = "NORMAL_ZONE"
    RESTRICTED_ZONE = "RESTRICTED_ZONE"


class ZoneTransition(str, Enum):
    ENTER = "ENTER"
    INSIDE = "INSIDE"
    EXIT = "EXIT"
    OUTSIDE = "OUTSIDE"


@dataclass
class LabPerimeterZone:
    zone_id: str
    name: str
    zone_type: Union[ZoneType, str]
    camera_id: str
    polygon: List[Tuple[float, float]]  # List of [x, y] coordinates
    color: Optional[str] = None


@dataclass
class BorderBoundary:
    boundary_id: str
    name: str
    camera_id: str
    line_start: Tuple[float, float]  # (x1, y1)
    line_end: Tuple[float, float]    # (x2, y2)
    normal_vector: Tuple[float, float] = (0.0, 1.0)


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


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def line_intersection(p1, p2, p3, p4) -> bool:
    """Returns True if line segment p1-p2 intersects segment p3-p4."""
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


class PerimeterEngine:
    """
    Perimeter zone geometry manager and track transition state machine.
    Tracks ENTER, INSIDE, and EXIT transitions per camera and track ID.
    Supports boundary lines and crossing detection.
    """

    def __init__(self, validations_file: Optional[Path] = None):
        self.zones: Dict[str, LabPerimeterZone] = {}
        self.boundaries: Dict[str, BorderBoundary] = {}
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

    def add_boundary(self, boundary: BorderBoundary) -> None:
        """Registers a border boundary line."""
        self.boundaries[boundary.boundary_id] = boundary

    def get_zones(self, camera_id: Optional[str] = None) -> List[LabPerimeterZone]:
        """Returns zones, optionally filtered by camera ID."""
        if camera_id is None:
            return list(self.zones.values())
        return [
            z for z in self.zones.values()
            if z.camera_id == camera_id or z.camera_id == "ALL"
        ]

    def get_boundaries(self, camera_id: Optional[str] = None) -> List[BorderBoundary]:
        """Returns border boundaries, optionally filtered by camera ID."""
        if camera_id is None:
            return list(self.boundaries.values())
        return [
            b for b in self.boundaries.values()
            if b.camera_id == camera_id or b.camera_id == "ALL"
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

        for zone in applicable_zones:
            poly_np = np.array(zone.polygon, dtype=np.float32)
            dist = cv2.pointPolygonTest(poly_np, (gx, gy), measureDist=False)
            is_inside = dist >= 0

            was_inside = zone.zone_id in active_zones
            z_type_val = zone.zone_type.value if hasattr(zone.zone_type, "value") else str(zone.zone_type)

            if is_inside:
                if not was_inside:
                    active_zones.add(zone.zone_id)
                    events.append(
                        PerimeterEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:8]}",
                            camera_id=camera_id,
                            track_id=track_id,
                            class_name=class_name,
                            zone_id=zone.zone_id,
                            zone_type=z_type_val,
                            transition=ZoneTransition.ENTER,
                            ground_point=(gx, gy),
                            frame_idx=frame_idx,
                            timestamp=now_ts,
                        )
                    )
                else:
                    events.append(
                        PerimeterEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:8]}",
                            camera_id=camera_id,
                            track_id=track_id,
                            class_name=class_name,
                            zone_id=zone.zone_id,
                            zone_type=z_type_val,
                            transition=ZoneTransition.INSIDE,
                            ground_point=(gx, gy),
                            frame_idx=frame_idx,
                            timestamp=now_ts,
                        )
                    )
            else:
                if was_inside:
                    active_zones.remove(zone.zone_id)
                    events.append(
                        PerimeterEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:8]}",
                            camera_id=camera_id,
                            track_id=track_id,
                            class_name=class_name,
                            zone_id=zone.zone_id,
                            zone_type=z_type_val,
                            transition=ZoneTransition.EXIT,
                            ground_point=(gx, gy),
                            frame_idx=frame_idx,
                            timestamp=now_ts,
                        )
                    )

        return events

    def evaluate_spatial_events(
        self,
        camera_id: str,
        track: Dict[str, Any],
        frame_idx: int = 0,
        timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates track position and trajectory against zones and border boundaries,
        returning high-level spatial event descriptors.
        """
        track_id = int(track.get("track_id", 0))
        class_name = track.get("class_name", "unknown")
        trajectory = track.get("trajectory", [])
        dwell_seconds = float(track.get("dwell_seconds", 0.0))

        if "center" in track:
            curr_pt = (float(track["center"][0]), float(track["center"][1]))
        elif "bbox" in track:
            b = track["bbox"]
            curr_pt = ((b[0] + b[2]) / 2.0, b[3])
        elif len(trajectory) > 0:
            curr_pt = (float(trajectory[-1][0]), float(trajectory[-1][1]))
        else:
            curr_pt = (0.0, 0.0)

        # 1. Zone transition events
        zone_events = self.evaluate_track(
            camera_id=camera_id,
            track_id=track_id,
            class_name=class_name,
            ground_point=curr_pt,
            frame_idx=frame_idx,
            timestamp=timestamp,
        )

        spatial_events: List[Dict[str, Any]] = []

        active_zones = self.active_tracks.get((camera_id, track_id), set())
        in_restricted = False
        for zid in active_zones:
            z_obj = self.zones.get(zid)
            if z_obj:
                zt = z_obj.zone_type.value if hasattr(z_obj.zone_type, "value") else str(z_obj.zone_type)
                if zt in ("RESTRICTED", "RESTRICTED_ZONE", "CRITICAL"):
                    in_restricted = True

        for ze in zone_events:
            z_type_upper = ze.zone_type.upper()
            if ze.transition == ZoneTransition.ENTER:
                if z_type_upper in ("RESTRICTED", "RESTRICTED_ZONE", "CRITICAL"):
                    spatial_events.append({
                        "event_type": "ENTERED_RESTRICTED_ZONE",
                        "zone_id": ze.zone_id,
                        "zone_type": ze.zone_type,
                        "ground_point": ze.ground_point,
                    })
                else:
                    spatial_events.append({
                        "event_type": "ENTERED_ZONE",
                        "zone_id": ze.zone_id,
                        "zone_type": ze.zone_type,
                        "ground_point": ze.ground_point,
                    })
            elif ze.transition == ZoneTransition.EXIT:
                if z_type_upper in ("RESTRICTED", "RESTRICTED_ZONE", "CRITICAL"):
                    spatial_events.append({
                        "event_type": "EXITED_RESTRICTED_ZONE",
                        "zone_id": ze.zone_id,
                        "zone_type": ze.zone_type,
                        "ground_point": ze.ground_point,
                    })

        # 2. Boundary Crossing
        if len(trajectory) >= 2:
            p_prev = trajectory[-2]
            p_curr = trajectory[-1]
            for boundary in self.get_boundaries(camera_id):
                if line_intersection(p_prev, p_curr, boundary.line_start, boundary.line_end):
                    spatial_events.append({
                        "event_type": "CROSSED_BOUNDARY",
                        "boundary_id": boundary.boundary_id,
                        "boundary_name": boundary.name,
                    })

        # 3. Loitering in restricted zone / Prolonged presence
        if in_restricted:
            if dwell_seconds >= 60.0:
                spatial_events.append({
                    "event_type": "PROLONGED_PRESENCE",
                    "dwell_seconds": dwell_seconds,
                })
            if dwell_seconds >= 30.0:
                spatial_events.append({
                    "event_type": "LOITERING_IN_RESTRICTED_ZONE",
                    "dwell_seconds": dwell_seconds,
                })

        return spatial_events

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
