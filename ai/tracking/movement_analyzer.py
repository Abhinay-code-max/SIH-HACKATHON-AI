"""
Tactical Movement & Border Direction Vector Analyzer.
Calculates ground-contact displacement, instantaneous velocity, heading angles,
and categorizes trajectory vectors relative to perimeter border boundaries.
"""

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class MovementAnalyzer:
    """
    Analyzes trajectory movement vectors for persistent tracks.
    Classifies movement into tactical states:
    - TOWARD_BORDER: Moving within +/- 35 deg of border normal
    - AWAY_FROM_BORDER: Moving opposite to border normal
    - PARALLEL_TO_BORDER: Moving along the border line
    - STATIONARY: Displacement below stationary threshold (< 2.0 px/step)
    """

    def __init__(
        self,
        stationary_threshold_px: float = 2.0,
        default_border_normal_deg: float = 90.0,  # 90 deg = South / Down toward bottom border
        smoothing_window: int = 5,
        target_fps: float = 30.0,
    ):
        self.stationary_threshold_px = stationary_threshold_px
        self.default_border_normal_deg = default_border_normal_deg
        self.smoothing_window = smoothing_window
        self.target_fps = target_fps

    def calculate_heading_deg(self, dx: float, dy: float) -> float:
        """
        Calculates heading angle in degrees [0, 360).
        0 deg = East (+X)
        90 deg = South (+Y, down in screen coordinates)
        180 deg = West (-X)
        270 deg = North (-Y, up in screen coordinates)
        """
        rad = math.atan2(dy, dx)
        deg = math.degrees(rad) % 360.0
        return round(deg, 2)

    def calculate_angle_difference(self, angle1: float, angle2: float) -> float:
        """Computes minimum angular difference in range [0, 180] degrees."""
        diff = abs((angle1 - angle2 + 180.0) % 360.0 - 180.0)
        return diff

    def classify_movement_vector(
        self,
        dx: float,
        dy: float,
        border_normal_deg: Optional[float] = None,
    ) -> Tuple[str, float, float, bool]:
        """
        Classifies vector into (movement_state, speed_px_step, heading_deg, is_approaching_border).
        """
        normal_deg = border_normal_deg if border_normal_deg is not None else self.default_border_normal_deg
        dist = math.hypot(dx, dy)

        # 1. Check if stationary
        if dist < self.stationary_threshold_px:
            return "STATIONARY", round(dist, 2), 0.0, False

        heading = self.calculate_heading_deg(dx, dy)
        diff_to_normal = self.calculate_angle_difference(heading, normal_deg)

        # 2. Classification relative to border approach normal
        # Within +/- 35 degrees of normal -> directly heading toward border
        if diff_to_normal <= 35.0:
            state = "TOWARD_BORDER"
            is_approaching = True
        # Within +/- 35 degrees of opposite normal (diff > 145 deg) -> moving away from border
        elif diff_to_normal >= 145.0:
            state = "AWAY_FROM_BORDER"
            is_approaching = False
        # Within parallel bands (55 to 125 degrees difference)
        else:
            state = "PARALLEL_TO_BORDER"
            is_approaching = diff_to_normal < 90.0

        return state, round(dist, 2), heading, is_approaching

    def analyze_track(
        self,
        track: Dict[str, Any],
        border_normal_deg: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes a single track dictionary containing 'trajectory' or 'velocity_vector'.
        Returns structured movement telemetry.
        """
        track_id = track.get("track_id", 0)
        history = track.get("trajectory", [])
        current_fps = fps or self.target_fps

        dx, dy = 0.0, 0.0
        window = min(self.smoothing_window, len(history))

        if window >= 2:
            p_start = history[-window]
            p_end = history[-1]
            dx = (p_end[0] - p_start[0]) / float(window - 1)
            dy = (p_end[1] - p_start[1]) / float(window - 1)
        elif "velocity_vector" in track and track["velocity_vector"]:
            v = track["velocity_vector"]
            dx, dy = float(v[0]), float(v[1])

        state, speed_step, heading, is_approaching = self.classify_movement_vector(
            dx=dx,
            dy=dy,
            border_normal_deg=border_normal_deg,
        )

        speed_px_s = round(speed_step * current_fps, 1)

        return {
            "track_id": track_id,
            "speed_px_step": speed_step,
            "speed_px_s": speed_px_s,
            "heading_deg": heading,
            "movement_state": state,
            "is_approaching_border": is_approaching,
            "displacement_vector": [round(dx, 2), round(dy, 2)],
        }

    def analyze_tracks(
        self,
        tracks: List[Dict[str, Any]],
        border_normal_deg: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Batch analyzes active tracks and enriches track dicts with movement telemetry."""
        results = []
        for t in tracks:
            movement_info = self.analyze_track(t, border_normal_deg=border_normal_deg, fps=fps)
            t["movement"] = movement_info
            results.append(movement_info)
        return results


movement_analyzer = MovementAnalyzer()
