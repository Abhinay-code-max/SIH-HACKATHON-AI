"""
Confidence Tracker & False Positive Suppression Module.
Tracks detections across consecutive frames before confirming them as valid targets.
Reduces transient false alarms (e.g. foliage flicker, lighting shifts, compression artifacts).
Decoupled from spatial tracker ID generation.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Tuple


@dataclass
class TrackedDetectionState:
    object_id: Any
    class_name: str
    consecutive_hits: int
    total_hits: int
    last_confidence: float
    max_confidence: float
    first_seen: float
    last_seen: float
    is_confirmed: bool


class ConfidenceTracker:
    """
    Suppresses transient false positives by requiring N consecutive detections
    above a confidence threshold before an object is considered 'confirmed'.
    """

    def __init__(
        self,
        consecutive_frames: int = 3,
        min_confidence: float = 0.35,
        max_history_age_sec: float = 5.0,
    ):
        self.consecutive_frames = max(1, consecutive_frames)
        self.min_confidence = min_confidence
        self.max_history_age_sec = max_history_age_sec
        # Key: (object_id, class_name) -> TrackedDetectionState
        self._states: Dict[Tuple[Any, str], TrackedDetectionState] = {}

    def update(
        self,
        object_id: Any,
        class_name: str,
        confidence: float,
        timestamp: Optional[float] = None,
        min_confidence: Optional[float] = None,
    ) -> bool:
        """
        Update the confidence state of a tracked target.

        Args:
            object_id: Identifier from tracker or upstream system.
            class_name: Object class label (e.g., 'person', 'vehicle').
            confidence: Detection confidence score [0.0 - 1.0].
            timestamp: Optional POSIX timestamp; defaults to time.time().
            min_confidence: Optional override for per-class minimum threshold.

        Returns:
            bool: True if the detection meets the confirmation criteria (>= consecutive_frames),
                  False otherwise.
        """
        now = timestamp if timestamp is not None else time.time()
        effective_min_conf = min_confidence if min_confidence is not None else self.min_confidence
        key = (object_id, class_name)

        if confidence < effective_min_conf:
            # Below confidence threshold: reset consecutive run if present
            if key in self._states:
                state = self._states[key]
                state.consecutive_hits = 0
                state.last_confidence = confidence
                state.last_seen = now
                state.is_confirmed = False
            return False

        if key not in self._states:
            self._states[key] = TrackedDetectionState(
                object_id=object_id,
                class_name=class_name,
                consecutive_hits=1,
                total_hits=1,
                last_confidence=confidence,
                max_confidence=confidence,
                first_seen=now,
                last_seen=now,
                is_confirmed=(self.consecutive_frames <= 1),
            )
        else:
            state = self._states[key]
            state.consecutive_hits += 1
            state.total_hits += 1
            state.last_confidence = confidence
            state.max_confidence = max(state.max_confidence, confidence)
            state.last_seen = now
            if state.consecutive_hits >= self.consecutive_frames:
                state.is_confirmed = True

        return self._states[key].is_confirmed

    def is_confirmed(self, object_id: Any, class_name: Optional[str] = None) -> bool:
        """Check if an object_id (or specific object_id + class_name) is confirmed."""
        if class_name is not None:
            state = self._states.get((object_id, class_name))
            return state.is_confirmed if state else False

        # If class_name not specified, check if any class for this object_id is confirmed
        for (oid, _), state in self._states.items():
            if oid == object_id and state.is_confirmed:
                return True
        return False

    def get_state(self, object_id: Any, class_name: str) -> Optional[TrackedDetectionState]:
        """Retrieve state for a given target."""
        return self._states.get((object_id, class_name))

    def prune_stale(self, max_age_sec: Optional[float] = None, current_time: Optional[float] = None) -> int:
        """
        Remove targets not observed within max_age_sec.

        Returns:
            Number of pruned records.
        """
        max_age = max_age_sec if max_age_sec is not None else self.max_history_age_sec
        now = current_time if current_time is not None else time.time()
        stale_keys = [
            k for k, state in self._states.items()
            if (now - state.last_seen) > max_age
        ]
        for k in stale_keys:
            del self._states[k]
        return len(stale_keys)

    def reset(self, object_id: Optional[Any] = None) -> None:
        """Reset internal states for a specific object_id or all objects."""
        if object_id is None:
            self._states.clear()
        else:
            keys_to_remove = [k for k in self._states if k[0] == object_id]
            for k in keys_to_remove:
                del self._states[k]
