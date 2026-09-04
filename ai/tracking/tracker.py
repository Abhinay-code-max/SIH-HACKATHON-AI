"""
Persistent Object Tracking & Trajectory Engine.
Uses ByteTrack / BoT-SORT to maintain continuous Track IDs across frames (e.g. Person #21).
Calculates velocity, dwell time (loitering), and trajectory motion vectors for tripwires.
"""

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.inference.loader import get_device, load_local_model


class ObjectTracker:
    def __init__(self, model_name: str = "yolov8l.pt", max_history_points: int = 30):
        self.model_name = model_name
        self.model = None
        self.device = get_device()
        self.max_history_points = max_history_points
        # track_id -> {class_name, history: deque([(x, y)]), first_seen: float, last_seen: float, conf: float}
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def _ensure_model(self):
        if self.model is None:
            self.model = load_local_model(self.model_name)

    def update(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        tracker_type: str = "bytetrack.yaml",
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Runs tracking on a frame.
        Returns:
          (active_tracks, annotated_frame_with_trails)
        """
        self._ensure_model()
        now = time.time()
        # Filter strictly for surveillance classes (person, vehicles, animals, bags)
        # COCO IDs: 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck, 14-23:animals, 24:backpack, 26:handbag, 28:suitcase
        surveillance_class_ids = [0, 1, 2, 3, 5, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28]

        # Run tracking with persistent IDs restricted to surveillance classes
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=tracker_type,
            classes=surveillance_class_ids,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
        )

        active_tracks = []
        annotated = frame.copy()

        res = results[0]
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                # Track ID can be None if unassigned
                track_id = int(box.id[0].item()) if box.id is not None else None
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

                # Ground contact center point (bottom center of bbox)
                cx = (x1 + x2) / 2.0
                cy = y2 - 10.0  # ground contact point rather than face center

                if track_id is not None:
                    if track_id not in self.tracks:
                        self.tracks[track_id] = {
                            "track_id": track_id,
                            "class_name": cls_name,
                            "history": deque(maxlen=self.max_history_points),
                            "first_seen": now,
                            "last_seen": now,
                            "conf": conf,
                        }

                    t_info = self.tracks[track_id]
                    t_info["history"].append((round(cx, 1), round(cy, 1)))
                    t_info["last_seen"] = now
                    t_info["conf"] = conf
                    dwell_sec = now - t_info["first_seen"]

                    # Compute direction vector from history
                    dx, dy = 0.0, 0.0
                    if len(t_info["history"]) >= 5:
                        old_x, old_y = t_info["history"][0]
                        dx = cx - old_x
                        dy = cy - old_y

                    track_payload = {
                        "track_id": track_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "center": [round(cx, 1), round(cy, 1)],
                        "normalized_center": [round(cx / w, 4), round(cy / h, 4)],
                        "dwell_seconds": round(dwell_sec, 1),
                        "trajectory": list(t_info["history"]),
                        "velocity_vector": [round(dx, 1), round(dy, 1)],
                    }
                    active_tracks.append(track_payload)

                    # Draw Trajectory Motion Trail
                    pts = list(t_info["history"])
                    for i in range(1, len(pts)):
                        thickness = int(np.sqrt(self.max_history_points / float(i + 1)) * 2)
                        cv2.line(annotated, (int(pts[i - 1][0]), int(pts[i - 1][1])), (int(pts[i][0]), int(pts[i][1])), (0, 255, 255), thickness)

                    # Draw Bounding Box & Persistent Tag
                    color = (0, 255, 0) if cls_name == "person" else (255, 128, 0)
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"#{track_id} {cls_name} ({dwell_sec:.0f}s)"
                    cv2.putText(annotated, label, (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(annotated, label, (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # Purge stale tracks (> 10s inactivity)
        stale_ids = [tid for tid, info in self.tracks.items() if (now - info["last_seen"]) > 10.0]
        for tid in stale_ids:
            del self.tracks[tid]

        return active_tracks, annotated


tracker = ObjectTracker()
