"""
Behavior Feature Extraction Engine.
Extracts deterministic behavioral features:
- Movement state (stationary, moving)
- Direction vector & directional alignment (toward_boundary, away_from_boundary, parallel, unknown)
- Dwell and stationary durations (loitering)
- Moving <-> stationary transitions
- Repeated boundary approach count
- Group / proximity context (nearby_track_count, group_size, proximity_duration)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from training_lab.engine.threat_config import (
    LOITERING_THRESHOLD_SECONDS,
    STATIONARY_SPEED_THRESHOLD_PX_S,
    PROXIMITY_DISTANCE_PX,
    COSINE_SIMILARITY_APPROACH_THRESHOLD,
)


class BehaviorFeatureEngine:
    def __init__(self):
        # Persistent state per logical track (camera_id, track_id)
        # track_key -> dict of historical transitions, approach counts, etc.
        self._track_states: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def clear_state(self):
        self._track_states.clear()

    def extract_features(
        self,
        camera_id: str,
        track: Dict[str, Any],
        all_tracks_in_frame: Optional[List[Dict[str, Any]]] = None,
        boundary_segment: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        boundary_normal_vector: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Extracts comprehensive behavior features for a given track in a camera feed.
        """
        track_id = int(track.get("track_id", 0))
        track_key = (camera_id, track_id)
        state_record = self._track_states.setdefault(track_key, {
            "was_moving": None,
            "stationary_duration": 0.0,
            "approach_cycles": 0,
            "prev_approaching": False,
            "proximity_duration": 0.0,
            "last_speed": 0.0,
        })

        # 1. Trajectory & Velocity
        trajectory = track.get("trajectory", [])
        dwell_seconds = float(track.get("dwell_seconds", 0.0))
        speed_px_s = float(track.get("speed_px_s", 0.0))

        # Displacement vector over recent trajectory points
        if len(trajectory) >= 2:
            dx = float(trajectory[-1][0] - trajectory[-2][0])
            dy = float(trajectory[-1][1] - trajectory[-2][1])
        elif "velocity_vector" in track and track["velocity_vector"]:
            dx = float(track["velocity_vector"][0])
            dy = float(track["velocity_vector"][1])
        else:
            dx, dy = 0.0, 0.0

        disp_mag = math.hypot(dx, dy)
        if speed_px_s == 0.0 and disp_mag > 0.0:
            speed_px_s = round(disp_mag * 30.0, 2)

        # 2. Movement State & Stopping Transitions
        is_moving = speed_px_s > STATIONARY_SPEED_THRESHOLD_PX_S
        movement_state = "moving" if is_moving else "stationary"

        state_transition = "none"
        if state_record["was_moving"] is not None:
            if not state_record["was_moving"] and is_moving:
                state_transition = "stationary_to_moving"
            elif state_record["was_moving"] and not is_moving:
                state_transition = "moving_to_stationary"

        state_record["was_moving"] = is_moving

        if not is_moving:
            state_record["stationary_duration"] += 1.0 / 30.0  # Approx step frame time
        else:
            state_record["stationary_duration"] = 0.0

        stopped_duration = round(state_record["stationary_duration"], 2)

        # 3. Loitering
        # Dwell stationary inside area
        is_loitering = (dwell_seconds >= LOITERING_THRESHOLD_SECONDS and not is_moving) or (dwell_seconds >= LOITERING_THRESHOLD_SECONDS * 1.5)

        # 4. Direction & Boundary Cosine Similarity
        # Boundary normal points toward the protected side
        # Default normal is downward [0, 1] if not specified
        b_normal = boundary_normal_vector or (0.0, 1.0)
        norm_len = math.hypot(b_normal[0], b_normal[1])
        if norm_len > 0:
            b_norm_unit = (b_normal[0] / norm_len, b_normal[1] / norm_len)
        else:
            b_norm_unit = (0.0, 1.0)

        cosine_similarity = 0.0
        if disp_mag > 0.1:
            dir_unit = (dx / disp_mag, dy / disp_mag)
            cosine_similarity = round(dir_unit[0] * b_norm_unit[0] + dir_unit[1] * b_norm_unit[1], 3)
            if cosine_similarity >= COSINE_SIMILARITY_APPROACH_THRESHOLD:
                direction_classification = "toward_boundary"
            elif cosine_similarity <= -COSINE_SIMILARITY_APPROACH_THRESHOLD:
                direction_classification = "away_from_boundary"
            else:
                direction_classification = "parallel"
        else:
            direction_classification = "unknown"

        # 5. Repeated Approach
        is_currently_approaching = (direction_classification == "toward_boundary")
        if is_currently_approaching and not state_record["prev_approaching"]:
            state_record["approach_cycles"] += 1
        state_record["prev_approaching"] = is_currently_approaching
        repeated_approach_count = state_record["approach_cycles"]
        is_repeated_approach = repeated_approach_count >= 2

        # 6. Group / Proximity Context
        nearby_track_count = 0
        group_size = 1
        current_center = track.get("center")
        if current_center is None and "bbox" in track:
            bx = track["bbox"]
            current_center = [(bx[0] + bx[2]) / 2.0, bx[3]]

        if current_center is not None and all_tracks_in_frame:
            for other in all_tracks_in_frame:
                if int(other.get("track_id", -1)) == track_id:
                    continue
                o_center = other.get("center")
                if o_center is None and "bbox" in other:
                    obx = other["bbox"]
                    o_center = [(obx[0] + obx[2]) / 2.0, obx[3]]
                if o_center is not None:
                    dist = math.hypot(current_center[0] - o_center[0], current_center[1] - o_center[1])
                    if dist <= PROXIMITY_DISTANCE_PX:
                        nearby_track_count += 1
            group_size = 1 + nearby_track_count

        if nearby_track_count > 0:
            state_record["proximity_duration"] += 1.0 / 30.0
        else:
            state_record["proximity_duration"] = 0.0

        return {
            "track_id": track_id,
            "movement_state": movement_state,
            "is_moving": is_moving,
            "speed_px_s": round(speed_px_s, 2),
            "image_velocity": [round(dx, 2), round(dy, 2)],
            "direction_vector": [round(dx, 2), round(dy, 2)],
            "boundary_normal_vector": list(b_norm_unit),
            "cosine_similarity": cosine_similarity,
            "direction_classification": direction_classification,
            "state_transition": state_transition,
            "stopped_duration": stopped_duration,
            "dwell_seconds": dwell_seconds,
            "is_loitering": is_loitering,
            "repeated_approach_count": repeated_approach_count,
            "is_repeated_approach": is_repeated_approach,
            "nearby_track_count": nearby_track_count,
            "group_size": group_size,
            "proximity_duration": round(state_record["proximity_duration"], 2),
        }


behavior_feature_engine = BehaviorFeatureEngine()
