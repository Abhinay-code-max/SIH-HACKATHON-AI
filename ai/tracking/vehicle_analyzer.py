"""
Dedicated Vehicle Behavior & Stoppage Intelligence Engine.
Analyzes vehicle tracks (car, truck, bus, motorcycle, van) for:
1. Unauthorized stoppage in restricted sectors / checkpoints (< 2.5 px/step for >= threshold sec)
2. Excessive dwell time / loitering in transit corridors
3. Speed estimation and velocity profiling
"""

import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class VehicleBehaviorAnalyzer:
    """
    Monitors vehicular activity across camera zones:
    - Flags stopped vehicles in restricted perimeter lanes
    - Flags prolonged vehicle loitering at checkpoints
    - Enforces alert cooldown suppression to prevent alarm storms
    """

    VEHICLE_CLASSES: Set[str] = {"car", "truck", "bus", "motorcycle", "van", "unauthorized_vehicle"}

    def __init__(
        self,
        stoppage_speed_threshold_px: float = 2.5,
        stoppage_threshold_seconds: float = 5.0,  # 5.0s for rapid detection / demo
        loitering_threshold_seconds: float = 15.0,
        cooldown_seconds: float = 10.0,
    ):
        self.stoppage_speed_threshold_px = stoppage_speed_threshold_px
        self.stoppage_threshold_seconds = stoppage_threshold_seconds
        self.loitering_threshold_seconds = loitering_threshold_seconds
        self.cooldown_seconds = cooldown_seconds

        # (camera_id, track_id) -> timestamp when stoppage began
        self.stoppage_start_times: Dict[Tuple[str, int], float] = {}
        # rule_key -> last alert timestamp
        self.cooldowns: Dict[str, float] = {}

    def _should_suppress(self, rule_key: str, now: float) -> bool:
        last = self.cooldowns.get(rule_key, 0.0)
        if (now - last) < self.cooldown_seconds:
            return True
        self.cooldowns[rule_key] = now
        return False

    def is_vehicle(self, class_name: str) -> bool:
        return class_name.lower() in self.VEHICLE_CLASSES

    def calculate_recent_speed(self, trajectory: List[Tuple[float, float]], steps: int = 3) -> float:
        """Calculates average displacement per step over recent trajectory points."""
        if not trajectory or len(trajectory) < 2:
            return 0.0
        pts = trajectory[-steps:]
        if len(pts) < 2:
            return 0.0
        p_start = pts[0]
        p_end = pts[-1]
        dist = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
        return dist / float(len(pts) - 1)

    def is_point_in_zone(self, point: Tuple[float, float], polygon: List[List[float]], frame_shape: Tuple[int, int]) -> bool:
        """Checks if a point is inside a polygon (supports normalized [0..1] or pixel coords)."""
        if not polygon:
            return True
        h, w = frame_shape[:2]
        pts = []
        for p in polygon:
            # Check if normalized
            if p[0] <= 1.0 and p[1] <= 1.0:
                pts.append([int(p[0] * w), int(p[1] * h)])
            else:
                pts.append([int(p[0]), int(p[1])])
        poly_arr = np.array(pts, dtype=np.int32)
        return cv2.pointPolygonTest(poly_arr, (float(point[0]), float(point[1])), False) >= 0

    def evaluate_vehicles(
        self,
        camera_id: str,
        tracks: List[Dict[str, Any]],
        zones: Optional[List[Dict[str, Any]]] = None,
        frame_shape: Tuple[int, int] = (480, 640),
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates vehicle tracks against stoppage and loitering rules.
        Returns list of structured anomaly events.
        """
        now = timestamp if timestamp is not None else time.time()
        detected_anomalies: List[Dict[str, Any]] = []
        active_keys = set()

        for t in tracks:
            cls = t.get("class_name", "")
            if not self.is_vehicle(cls):
                continue

            track_id = t.get("track_id", 0)
            key = (camera_id, track_id)
            active_keys.add(key)

            trajectory = t.get("trajectory", [])
            center = t.get("center", [0.0, 0.0])
            dwell_sec = float(t.get("dwell_seconds", 0.0))
            conf = float(t.get("confidence", 0.9))
            bbox = t.get("bbox", [0.0, 0.0, 0.0, 0.0])

            speed = self.calculate_recent_speed(trajectory)

            # Check if track is inside any monitored zone
            in_restricted_zone = False
            matched_zone_id = "ZONE_RESTRICTED"
            matched_zone_name = "Restricted Sector"

            if zones:
                for z in zones:
                    if z.get("camera_id", camera_id) == camera_id:
                        poly = z.get("polygon", [])
                        if self.is_point_in_zone(center, poly, frame_shape):
                            in_restricted_zone = True
                            matched_zone_id = z.get("zone_id", matched_zone_id)
                            matched_zone_name = z.get("name", matched_zone_name)
                            break
            else:
                # If no explicit zones provided, treat all monitored vehicle tracks as monitored
                in_restricted_zone = True

            # -------------------------------------------------------------
            # RULE 1: VEHICLE STOPPAGE IN RESTRICTED SECTOR
            # -------------------------------------------------------------
            if in_restricted_zone and speed < self.stoppage_speed_threshold_px:
                if key not in self.stoppage_start_times:
                    self.stoppage_start_times[key] = now

                stoppage_duration = now - self.stoppage_start_times[key]

                if stoppage_duration >= self.stoppage_threshold_seconds:
                    rule_key = f"{camera_id}_veh_stoppage_{track_id}"
                    if not self._should_suppress(rule_key, now):
                        severity = "CRITICAL" if cls in {"truck", "bus"} else "HIGH"
                        detected_anomalies.append({
                            "anomaly_type": "VEHICLE_STOPPAGE_ALERT",
                            "severity": severity,
                            "camera_id": camera_id,
                            "track_id": track_id,
                            "class_name": cls,
                            "confidence": conf,
                            "bbox": bbox,
                            "trajectory": trajectory,
                            "speed_px_step": round(speed, 2),
                            "stoppage_duration_sec": round(stoppage_duration, 1),
                            "zone_id": matched_zone_id,
                            "zone_name": matched_zone_name,
                            "timestamp": now,
                            "description": (
                                f"Vehicle #{track_id} ({cls}) stopped in restricted area "
                                f"'{matched_zone_name}' (speed {speed:.1f} px/step) "
                                f"for {stoppage_duration:.1f}s (threshold: {self.stoppage_threshold_seconds:.1f}s)."
                            ),
                        })
            else:
                # Vehicle is moving or outside restricted zone; reset stoppage clock
                self.stoppage_start_times.pop(key, None)

            # -------------------------------------------------------------
            # RULE 2: SUSPICIOUS VEHICLE LOITERING
            # -------------------------------------------------------------
            if dwell_sec >= self.loitering_threshold_seconds:
                rule_key = f"{camera_id}_veh_loiter_{track_id}"
                if not self._should_suppress(rule_key, now):
                    detected_anomalies.append({
                        "anomaly_type": "VEHICLE_LOITERING_ALERT",
                        "severity": "HIGH",
                        "camera_id": camera_id,
                        "track_id": track_id,
                        "class_name": cls,
                        "confidence": conf,
                        "bbox": bbox,
                        "trajectory": trajectory,
                        "dwell_seconds": round(dwell_sec, 1),
                        "threshold": self.loitering_threshold_seconds,
                        "timestamp": now,
                        "description": f"Vehicle #{track_id} ({cls}) loitering in {camera_id} for {dwell_sec:.1f}s.",
                    })

        # Cleanup stale stopped vehicle state
        stale_keys = [k for k in self.stoppage_start_times.keys() if k[0] == camera_id and k not in active_keys]
        for k in stale_keys:
            self.stoppage_start_times.pop(k, None)

        return detected_anomalies


vehicle_analyzer = VehicleBehaviorAnalyzer()
