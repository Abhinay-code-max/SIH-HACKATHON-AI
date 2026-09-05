"""
Behavioral Anomaly & Compound Motion Intelligence Engine.
Detects:
1. Sprinting & Sudden Velocity Spikes (Perimeter breach/flight attempts)
2. Unattended / Abandoned Baggage (Stationary luggage isolated from any person)
3. Doorway Tailgating / Anti-Piggybacking (Rapid consecutive tripwire crossings)
4. Crowd Surges & Congestion Clustering
"""

from collections import deque
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class BehavioralAnomalyEngine:
    def __init__(self):
        # Configurable anomaly parameters
        self.sprint_velocity_threshold = 32.0       # Pixels per step (approx > 1.8x walking speed)
        self.baggage_isolation_distance = 120.0     # Pixels distance from nearest person
        self.baggage_timeout_seconds = 7.0          # Seconds isolated before alarm triggers
        self.tailgating_window_seconds = 1.8        # Seconds between unauthorized consecutive crossings
        self.crowd_cluster_radius = 140.0           # Radius for density surge detection
        self.crowd_cluster_min_persons = 4          # Min persons clustered to trigger surge

        # Internal tracking state
        # (camera_id, track_id) -> isolation start timestamp
        self.baggage_isolated_start: Dict[Tuple[str, int], float] = {}
        # (camera_id, wire_id) -> list of (track_id, timestamp)
        self.tripwire_crossing_history: Dict[Tuple[str, str], deque] = {}
        # rule_key -> last trigger timestamp for cooldown suppression
        self.cooldowns: Dict[str, float] = {}

    def _should_suppress(self, rule_key: str, cooldown_sec: float = 5.0) -> bool:
        now = time.time()
        last = self.cooldowns.get(rule_key, 0.0)
        if (now - last) < cooldown_sec:
            return True
        self.cooldowns[rule_key] = now
        return False

    def evaluate_motion_anomalies(
        self,
        camera_id: str,
        tracks: List[Dict[str, Any]],
        frame_shape: Tuple[int, int] = (480, 640),
    ) -> List[Dict[str, Any]]:
        """
        Runs behavioral motion and object relationship checks across active tracks.
        Returns list of detected anomaly descriptors.
        """
        now = time.time()
        detected_anomalies = []

        # -----------------------------------------------------------------
        # 1. SPRINTING & RAPID VELOCITY DISPLACEMENT
        # -----------------------------------------------------------------
        for t in tracks:
            history = t.get("trajectory", [])
            if len(history) >= 4:
                # Calculate recent velocity over last 3 frame steps
                p_old = history[-4]
                p_new = history[-1]
                dx = p_new[0] - p_old[0]
                dy = p_new[1] - p_old[1]
                dist = math.hypot(dx, dy)
                avg_velocity = dist / 3.0

                if avg_velocity >= self.sprint_velocity_threshold:
                    rule_key = f"{camera_id}_sprint_{t['track_id']}"
                    if not self._should_suppress(rule_key, cooldown_sec=6.0):
                        detected_anomalies.append({
                            "anomaly_type": "SPRINTING_DETECTED",
                            "severity": "HIGH",
                            "camera_id": camera_id,
                            "track_id": t["track_id"],
                            "class_name": t["class_name"],
                            "confidence": t["confidence"],
                            "bbox": t["bbox"],
                            "trajectory": t["trajectory"],
                            "velocity": round(avg_velocity, 1),
                            "threshold": self.sprint_velocity_threshold,
                            "timestamp": now,
                            "description": f"Target #{t['track_id']} displaying rapid displacement ({avg_velocity:.1f} px/step). Possible perimeter flight or breach.",
                        })

        # -----------------------------------------------------------------
        # 2. UNATTENDED BAGGAGE / ABANDONED OBJECT DETECTION
        # -----------------------------------------------------------------
        baggage_classes = {"backpack", "handbag", "suitcase"}
        person_tracks = [t for t in tracks if t["class_name"] == "person"]
        baggage_tracks = [t for t in tracks if t["class_name"] in baggage_classes]

        active_bag_keys = set()
        for b in baggage_tracks:
            b_key = (camera_id, b["track_id"])
            active_bag_keys.add(b_key)
            bx, by = b["center"]

            # Check if baggage is stationary (trajectory points clustered within 18 px)
            history = b.get("trajectory", [])
            if len(history) >= 3:
                xs = [p[0] for p in history[-6:]]
                ys = [p[1] for p in history[-6:]]
                spread = max(max(xs) - min(xs), max(ys) - min(ys))
                is_stationary = spread <= 18.0
            else:
                is_stationary = True

            # Calculate distance to nearest person
            min_dist = float("inf")
            for p in person_tracks:
                px, py = p["center"]
                d = math.hypot(bx - px, by - py)
                if d < min_dist:
                    min_dist = d

            if is_stationary and min_dist >= self.baggage_isolation_distance:
                # Bag is isolated from all persons!
                if b_key not in self.baggage_isolated_start:
                    self.baggage_isolated_start[b_key] = now

                isolated_duration = now - self.baggage_isolated_start[b_key]
                if isolated_duration >= self.baggage_timeout_seconds:
                    rule_key = f"{camera_id}_unattended_{b['track_id']}"
                    if not self._should_suppress(rule_key, cooldown_sec=10.0):
                        detected_anomalies.append({
                            "anomaly_type": "UNATTENDED_BAGGAGE_ALERT",
                            "severity": "CRITICAL",
                            "camera_id": camera_id,
                            "track_id": b["track_id"],
                            "class_name": b["class_name"],
                            "confidence": b["confidence"],
                            "bbox": b["bbox"],
                            "trajectory": b["trajectory"],
                            "isolated_duration_sec": round(isolated_duration, 1),
                            "nearest_person_dist": round(min_dist, 1),
                            "timestamp": now,
                            "description": f"Unattended {b['class_name']} detected stationary with nearest person {min_dist:.0f}px away for {isolated_duration:.0f}s.",
                        })
            else:
                # A person is near the bag; reset isolation timer
                self.baggage_isolated_start.pop(b_key, None)

        # Cleanup disappeared bags
        stale_bag_keys = [k for k in self.baggage_isolated_start.keys() if k[0] == camera_id and k not in active_bag_keys]
        for k in stale_bag_keys:
            self.baggage_isolated_start.pop(k, None)

        # -----------------------------------------------------------------
        # 3. CROWD DENSITY SURGE / CHOKEPOINT CLUSTERING
        # -----------------------------------------------------------------
        if len(person_tracks) >= self.crowd_cluster_min_persons:
            # Check pairwise clustering
            for i, p1 in enumerate(person_tracks):
                cluster = [p1]
                p1_x, p1_y = p1["center"]
                for j, p2 in enumerate(person_tracks):
                    if i != j:
                        p2_x, p2_y = p2["center"]
                        if math.hypot(p1_x - p2_x, p1_y - p2_y) <= self.crowd_cluster_radius:
                            cluster.append(p2)

                if len(cluster) >= self.crowd_cluster_min_persons:
                    rule_key = f"{camera_id}_crowd_surge"
                    if not self._should_suppress(rule_key, cooldown_sec=15.0):
                        detected_anomalies.append({
                            "anomaly_type": "CROWD_DENSITY_SURGE",
                            "severity": "HIGH",
                            "camera_id": camera_id,
                            "track_id": p1["track_id"],
                            "class_name": "person_group",
                            "confidence": 0.95,
                            "bbox": p1["bbox"],
                            "cluster_size": len(cluster),
                            "timestamp": now,
                            "description": f"High crowd concentration ({len(cluster)} persons) clustered within {self.crowd_cluster_radius:.0f}px radius in {camera_id}.",
                        })
                    break

        return detected_anomalies

    def record_tripwire_crossing(
        self,
        camera_id: str,
        wire_id: str,
        track_id: int,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates doorway tailgating (anti-piggybacking).
        Triggers an anomaly if a second subject crosses shortly after the first.
        """
        now = timestamp or time.time()
        key = (camera_id, wire_id)

        if key not in self.tripwire_crossing_history:
            self.tripwire_crossing_history[key] = deque(maxlen=10)

        history = self.tripwire_crossing_history[key]
        anomaly = None

        if len(history) > 0:
            last_track_id, last_time = history[-1]
            delta_sec = now - last_time

            # If a different track crossed within the tailgating window
            if last_track_id != track_id and delta_sec <= self.tailgating_window_seconds:
                rule_key = f"{camera_id}_{wire_id}_tailgate_{last_track_id}_{track_id}"
                if not self._should_suppress(rule_key, cooldown_sec=10.0):
                    anomaly = {
                        "anomaly_type": "TAILGATING_BREACH",
                        "severity": "CRITICAL",
                        "camera_id": camera_id,
                        "wire_id": wire_id,
                        "lead_track_id": last_track_id,
                        "tail_track_id": track_id,
                        "delta_seconds": round(delta_sec, 2),
                        "window_threshold": self.tailgating_window_seconds,
                        "timestamp": now,
                        "description": f"Tailgating breach at {wire_id}: Track #{track_id} followed #{last_track_id} in {delta_sec:.2f}s without authorization interval.",
                    }

        history.append((track_id, now))
        return anomaly


behavioral_anomaly_engine = BehavioralAnomalyEngine()
