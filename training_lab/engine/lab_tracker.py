"""
Lab Perception Engine — Multi-Camera ByteTrack Object Tracker.
Maintains dedicated, isolated ByteTrack tracking states for each camera feed
(CAM_01 through CAM_05) to ensure zero cross-camera track leakage, continuous
tracking persistence, trajectory history accumulation, and velocity vector calculation.
"""

from collections import deque
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.tracking.tracker import ObjectTracker
from training_lab.engine.lab_detector import LabDetector


class LabMultiCamTracker:
    """
    Multi-camera persistent tracker for the 5-camera surveillance simulation grid.
    Maintains strictly isolated ObjectTracker instances per camera stream.
    """

    def __init__(self, model_name: str = "auto", max_history_points: int = 30):
        """
        Initializes the Multi-Camera Tracker with per-camera isolation.

        Args:
            model_name: model weight specification or 'auto'
            max_history_points: max trajectory points retained in history
        """
        self.model_name = model_name
        self.max_history_points = max_history_points
        # Camera-isolated dictionary of ObjectTrackers: camera_id -> ObjectTracker
        self.trackers: Dict[str, ObjectTracker] = {}
        # Shared detector for frame-level YOLO detections
        self.detector = LabDetector(model_name=model_name)

    def get_or_create_tracker(self, camera_id: str) -> ObjectTracker:
        """
        Retrieves or initializes a dedicated ObjectTracker instance for the specified camera.
        Enforces 100% state isolation so tracks never cross camera boundaries.
        """
        cid = camera_id.upper().strip()
        if cid not in self.trackers:
            ot = ObjectTracker(
                model_name=self.model_name,
                max_history_points=self.max_history_points,
            )
            # Share loaded GPU model instance to optimize VRAM and eliminate redundant loads
            if hasattr(self.detector, "model") and self.detector.model is not None:
                ot.model = self.detector.model
            self.trackers[cid] = ot
        return self.trackers[cid]

    def update_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        conf_threshold: float = 0.35,
        timestamp: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Updates tracking state for an individual camera frame.

        Args:
            frame: input BGR image frame (np.ndarray)
            camera_id: camera identifier (e.g. 'CAM_01')
            conf_threshold: minimum confidence threshold (default 0.35)
            timestamp: frame timestamp in seconds (optional)

        Returns:
            Tuple of:
                - List of persistent track dicts
                - Annotated image frame with bounding boxes and trajectory lines
        """
        cid = camera_id.upper().strip()
        tracker = self.get_or_create_tracker(cid)
        wall_now = time.time()
        sim_ts = float(timestamp) if timestamp is not None else None
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        formatted_tracks: List[Dict[str, Any]] = []

        # 1. Attempt standard ObjectTracker update with YOLO + ByteTrack
        raw_tracks, ann_frame = tracker.update(frame, conf_threshold=conf_threshold)

        if raw_tracks:
            annotated_frame = ann_frame
            for t in raw_tracks:
                track_id = int(t["track_id"])
                cls_name = str(t.get("class_name", "object")).lower()
                conf = float(t.get("confidence", 0.90))
                bbox = t.get("bbox", [0.0, 0.0, 0.0, 0.0])
                center = t.get("center", [(bbox[0] + bbox[2]) / 2.0, bbox[3] - 5.0])
                dwell = max(0.0, float(t.get("dwell_seconds", 0.0)))
                traj = t.get("trajectory", [])

                if len(traj) >= 2:
                    dx = round(traj[-1][0] - traj[-2][0], 2)
                    dy = round(traj[-1][1] - traj[-2][1], 2)
                else:
                    dx, dy = 0.0, 0.0

                speed_px_s = round(math.hypot(dx, dy) * 30.0, 2)

                formatted_tracks.append({
                    "track_id": track_id,
                    "tracking_label": f"{cls_name.upper()}_{track_id:03d}",
                    "camera_id": cid,
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(float(c), 1) for c in bbox],
                    "center": [round(float(center[0]), 1), round(float(center[1]), 1)],
                    "dwell_seconds": round(dwell, 2),
                    "trajectory": traj,
                    "velocity_vector": [dx, dy],
                    "speed_px_s": speed_px_s,
                })

        # 2. If photographic YOLO did not yield tracks (offline synthetic lab videos), use LabDetector
        else:
            detections = self.detector.detect_frame(
                frame=frame,
                camera_id=cid,
                conf_threshold=conf_threshold,
                timestamp=sim_ts,
            )

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cls_name = det["class_name"]
                conf = det["confidence"]

                # Bottom-center ground contact point
                cx = round((x1 + x2) / 2.0, 1)
                cy = round(y2 - 5.0, 1)

                # Associate with existing tracks in this camera's tracker
                matched_id = None
                best_dist = 120.0  # Max pixel displacement matching threshold

                for tid, tinfo in tracker.tracks.items():
                    if tinfo.get("class_name") == cls_name and tinfo.get("history"):
                        last_pt = tinfo["history"][-1]
                        dist = math.hypot(cx - last_pt[0], cy - last_pt[1])
                        if dist < best_dist:
                            best_dist = dist
                            matched_id = tid

                # Allocate new track_id if not matched
                if matched_id is None:
                    matched_id = max(tracker.tracks.keys(), default=0) + 1
                    tracker.tracks[matched_id] = {
                        "track_id": matched_id,
                        "class_name": cls_name,
                        "history": deque(maxlen=self.max_history_points),
                        "first_seen": wall_now,
                        "sim_first_seen": sim_ts if sim_ts is not None else 0.0,
                        "last_seen": wall_now,
                        "conf": conf,
                    }

                tinfo = tracker.tracks[matched_id]
                tinfo["history"].append((cx, cy))
                tinfo["last_seen"] = wall_now
                tinfo["conf"] = conf

                if sim_ts is not None and "sim_first_seen" in tinfo:
                    dwell_sec = max(0.0, round(sim_ts - tinfo["sim_first_seen"], 2))
                else:
                    dwell_sec = max(0.0, round(wall_now - tinfo["first_seen"], 2))
                history_list = list(tinfo["history"])

                # Velocity vector calculation
                if len(history_list) >= 2:
                    dx = round(history_list[-1][0] - history_list[-2][0], 2)
                    dy = round(history_list[-1][1] - history_list[-2][1], 2)
                else:
                    dx, dy = 0.0, 0.0

                speed_px_s = round(math.hypot(dx, dy) * 30.0, 2)
                tracking_label = f"{cls_name.upper()}_{matched_id:03d}"

                track_dict = {
                    "track_id": matched_id,
                    "tracking_label": tracking_label,
                    "camera_id": cid,
                    "class_name": cls_name,
                    "confidence": round(float(conf), 3),
                    "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                    "center": [cx, cy],
                    "dwell_seconds": dwell_sec,
                    "trajectory": history_list,
                    "velocity_vector": [dx, dy],
                    "speed_px_s": speed_px_s,
                }
                formatted_tracks.append(track_dict)

                # Render trajectory trail on annotated frame
                pts = history_list
                for p_idx in range(1, len(pts)):
                    thickness = max(1, int(math.sqrt(self.max_history_points / float(p_idx + 1)) * 2))
                    p_start = (int(pts[p_idx - 1][0]), int(pts[p_idx - 1][1]))
                    p_end = (int(pts[p_idx][0]), int(pts[p_idx][1]))
                    cv2.line(annotated_frame, p_start, p_end, (0, 240, 255), thickness)

                # Render bounding box and tracking label
                color = (0, 255, 120) if cls_name == "person" else (255, 180, 0)
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                hud_text = f"#{matched_id} {tracking_label} ({dwell_sec:.1f}s)"
                text_y = max(22, int(y1) - 6)
                cv2.putText(annotated_frame, hud_text, (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated_frame, hud_text, (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        return formatted_tracks, annotated_frame

    def update_bundle(
        self,
        bundle: Dict[str, Any],
        conf_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Updates all active camera feeds in the synchronized 5-camera bundle.

        Args:
            bundle: synchronized frame bundle from MultiCameraSimulator
            conf_threshold: detection confidence threshold

        Returns:
            Dict containing:
                "frame_idx": int,
                "timestamp": float,
                "feeds": {
                    "CAM_01": {"camera_id": "CAM_01", "tracks": [...], "annotated_frame": np.ndarray},
                    ...
                }
        """
        frame_idx = bundle.get("frame_idx", 0)
        timestamp = bundle.get("timestamp", round(frame_idx / 30.0, 4))
        feeds = bundle.get("feeds", {})

        processed_feeds: Dict[str, Any] = {}

        for cam_id, feed_data in feeds.items():
            frame = feed_data.get("frame")
            if frame is None or not isinstance(frame, np.ndarray):
                continue
            feed_ts = feed_data.get("timestamp", timestamp)
            tracks, ann_frame = self.update_frame(
                frame=frame,
                camera_id=cam_id,
                conf_threshold=conf_threshold,
                timestamp=feed_ts,
            )
            processed_feeds[cam_id] = {
                "camera_id": cam_id,
                "tracks": tracks,
                "annotated_frame": ann_frame,
            }

        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "feeds": processed_feeds,
        }

    def reset(self) -> None:
        """Clears all tracker states across all cameras."""
        self.trackers.clear()


if __name__ == "__main__":
    from training_lab.engine.multi_cam_simulator import MultiCameraSimulator
    from training_lab.engine.video_manager import ensure_demo_camera_videos

    print("[LabMultiCamTracker] Initializing Multi-Camera Tracker...")
    vids = ensure_demo_camera_videos()
    sim = MultiCameraSimulator(camera_bindings=vids, loop=True, fps=30.0)
    tracker = LabMultiCamTracker(model_name="auto")

    print("  Stepping 5 bundles...")
    for _ in range(5):
        bundle = sim.step()
        if bundle:
            res = tracker.update_bundle(bundle)
            for cid, data in res["feeds"].items():
                print(f"  [{cid}] Step {bundle['frame_idx']} -> {len(data['tracks'])} tracks: {[t['tracking_label'] for t in data['tracks']]}")

    sim.close()
    print("[LabMultiCamTracker] Test run completed.")
