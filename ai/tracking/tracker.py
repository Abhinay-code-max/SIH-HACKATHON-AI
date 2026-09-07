"""
Persistent Object Tracking & Trajectory Engine.
Uses ByteTrack to maintain continuous Track IDs across frames (e.g. Person #21).
Consumes pre-computed detections from BaseDetector (RawDetection contracts),
eliminating duplicate YOLO inference and preventing VRAM exhaustion on 4GB hardware.
Calculates velocity, dwell time (loitering), and trajectory motion vectors for tripwires.
"""

from collections import deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.detection.detector import BaseDetector, YoloDetector
from ai.inference.loader import get_device
from backend.app.models.contracts import RawDetection

logger = logging.getLogger("ai.tracking.tracker")


class DetectionsAdapter:
    """
    Lightweight container adapting RawDetection objects / bounding boxes
    into the format expected by Ultralytics BYTETracker without running duplicate YOLO inference.
    Supports xywh, conf, and cls indexing.
    """

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx: Any) -> "DetectionsAdapter":
        return DetectionsAdapter(self.xywh[idx], self.conf[idx], self.cls[idx])


class ObjectTracker:
    """
    Persistent Object Tracking & Trajectory Engine.
    Consumes pre-computed detections from BaseDetector (RawDetection objects)
    and uses ByteTrack Kalman filter association to maintain continuous Track IDs across frames.

    Zero duplicate YOLO inference: does NOT load or run a secondary YOLO model in VRAM.
    """

    def __init__(
        self,
        detector: Optional[BaseDetector] = None,
        model_name: str = "auto",
        max_history_points: int = 30,
        track_high_thresh: float = 0.45,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.45,
        track_buffer: int = 30,
        match_thresh: float = 0.80,
    ):
        self.detector = detector
        self.model_name = model_name
        self.device = get_device()
        self.max_history_points = max_history_points
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh

        # Initialize ByteTracker (runs entirely on CPU Kalman filtering: 0 MB VRAM)
        self.byte_cfg = IterableSimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            fuse_score=True,
        )
        self.byte_tracker = BYTETracker(self.byte_cfg)

        # track_id -> {class_name, history: deque([(x, y)]), first_seen: float, last_seen: float, conf: float}
        self.tracks: Dict[int, Dict[str, Any]] = {}

    @property
    def model(self) -> Any:
        """Backward compatibility for external routes/inspectors checking tracker.model."""
        if self.detector is not None and hasattr(self.detector, "model"):
            return self.detector.model
        return None

    @model.setter
    def model(self, val: Any) -> None:
        if self.detector is not None and hasattr(self.detector, "model"):
            self.detector.model = val

    def _ensure_detector(self) -> BaseDetector:
        """Lazily initialize shared YoloDetector if no precomputed detections are provided."""
        if self.detector is None:
            self.detector = YoloDetector(model_name=self.model_name)
        return self.detector

    def reset(self) -> None:
        """Reset ByteTracker Kalman filter and active track history."""
        self.byte_tracker = BYTETracker(self.byte_cfg)
        self.tracks.clear()

    def update(
        self,
        frame: np.ndarray,
        detections: Optional[Union[List[RawDetection], List[Dict[str, Any]]]] = None,
        conf_threshold: Optional[float] = None,
        tracker_type: str = "bytetrack.yaml",
        camera_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Runs ByteTrack persistent tracking on a frame using pre-computed detections.

        Args:
            frame: Numpy BGR image array (H, W, C).
            detections: Pre-computed detections from BaseDetector (RawDetection or dicts).
                        If None, delegates detection to self.detector.
            conf_threshold: Optional confidence cutoff for tracking candidate filtering.
            tracker_type: Tracking configuration type (defaults to ByteTrack).
            camera_id: Optional camera identifier for stream routing.

        Returns:
            Tuple of (active_tracks, annotated_frame_with_trails)
        """
        if frame is None or frame.size == 0:
            return [], frame if frame is not None else np.zeros((0, 0, 3), dtype=np.uint8)

        now = time.time()
        h, w = frame.shape[:2]

        # 1. Resolve detections: consume pre-computed or delegate to BaseDetector
        if detections is None:
            det_engine = self._ensure_detector()
            raw_detections = det_engine.detect(frame, camera_id=camera_id)
        else:
            raw_detections = detections

        # 2. Extract bounding boxes and apply confidence + noise filtering
        candidates: List[Any] = []
        raw_boxes: List[Dict[str, Any]] = []

        min_conf = conf_threshold if conf_threshold is not None else self.track_low_thresh

        for det in raw_detections:
            if isinstance(det, RawDetection):
                conf = det.confidence
                cname = det.class_name
                coords = det.bbox
            elif isinstance(det, dict):
                conf = float(det.get("confidence", 0.0))
                cname = det.get("class_name", "unknown")
                coords = det.get("bbox", [0.0, 0.0, 0.0, 0.0])
            else:
                continue

            if conf < min_conf:
                continue

            x1, y1, x2, y2 = [float(c) for c in coords]

            # Filter out tiny noise specks (< 20px)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            candidates.append(det)
            raw_boxes.append({
                "det": det,
                "cls_name": cname,
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
            })

        # 3. Overlap & Containment Suppression:
        # If a chair detection is substantially contained inside or overlapping with a person's upper body,
        # suppress the chair box so it doesn't clutter the person's face/chest.
        person_boxes = [b["bbox"] for b in raw_boxes if b["cls_name"] == "person"]
        filtered_candidates: List[Any] = []
        filtered_boxes: List[Dict[str, Any]] = []

        for b in raw_boxes:
            if b["cls_name"] == "chair" and person_boxes:
                bx1, by1, bx2, by2 = b["bbox"]
                chair_cx, chair_cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                is_contained = any(
                    px1 <= chair_cx <= px2 and py1 <= chair_cy <= py2
                    for px1, py1, px2, py2 in person_boxes
                )
                if is_contained:
                    continue
            filtered_candidates.append(b["det"])
            filtered_boxes.append(b)

        # 4. Feed detections into ByteTracker
        if not filtered_boxes:
            tracked_boxes = np.zeros((0, 8), dtype=np.float32)
        else:
            xywh_list = []
            conf_list = []
            cls_list = []
            class_to_id: Dict[str, int] = {}
            id_to_class: Dict[int, str] = {}

            for b in filtered_boxes:
                cname = b["cls_name"]
                if cname not in class_to_id:
                    cid = len(class_to_id)
                    class_to_id[cname] = cid
                    id_to_class[cid] = cname

                x1, y1, x2, y2 = b["bbox"]
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                cx = x1 + bw / 2.0
                cy = y1 + bh / 2.0

                xywh_list.append([cx, cy, bw, bh])
                conf_list.append(b["conf"])
                cls_list.append(float(class_to_id[cname]))

            adapter = DetectionsAdapter(
                np.array(xywh_list, dtype=np.float32),
                np.array(conf_list, dtype=np.float32),
                np.array(cls_list, dtype=np.float32),
            )
            tracked_boxes = self.byte_tracker.update(adapter)

        # 5. Format active tracks & annotate frame
        active_tracks: List[Dict[str, Any]] = []
        annotated = frame.copy()

        for row in tracked_boxes:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            track_id = int(row[4])
            track_conf = float(row[5])
            cls_id = int(row[6])
            det_idx = int(row[7])

            # Resolve class name from original detection or id mapping
            cls_name = id_to_class.get(cls_id, "unknown")
            if 0 <= det_idx < len(filtered_candidates):
                matched_det = filtered_candidates[det_idx]
                if isinstance(matched_det, RawDetection):
                    cls_name = matched_det.class_name
                    # Populate object_id on the RawDetection contract object
                    matched_det.object_id = track_id
                elif isinstance(matched_det, dict) and "class_name" in matched_det:
                    cls_name = matched_det["class_name"]
                    matched_det["object_id"] = track_id
            elif track_id in self.tracks:
                cls_name = self.tracks[track_id]["class_name"]

            # Ground contact center point (bottom center of bbox)
            cx = (x1 + x2) / 2.0
            cy = y2 - 10.0

            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    "track_id": track_id,
                    "class_name": cls_name,
                    "history": deque(maxlen=self.max_history_points),
                    "first_seen": now,
                    "last_seen": now,
                    "conf": track_conf,
                }

            t_info = self.tracks[track_id]
            t_info["history"].append((round(cx, 1), round(cy, 1)))
            t_info["last_seen"] = now
            t_info["conf"] = track_conf
            dwell_sec = now - t_info["first_seen"]

            # Compute direction velocity vector from history
            dx, dy = 0.0, 0.0
            if len(t_info["history"]) >= 5:
                old_x, old_y = t_info["history"][0]
                dx = cx - old_x
                dy = cy - old_y

            # Backward-compat aliases: dwell_seconds (kept for camera_manager/intelligence_engine), trajectory (kept for camera_manager/intelligence_engine), velocity_vector (kept for downstream analyzers)
            track_payload = {
                "track_id": track_id,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(track_conf, 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "center": (round(cx, 1), round(cy, 1)),
                "normalized_center": [round(cx / max(1, w), 4), round(cy / max(1, h), 4)],
                "dwell_time_s": round(dwell_sec, 1),
                "dwell_seconds": round(dwell_sec, 1),
                "history": list(t_info["history"]),
                "trajectory": list(t_info["history"]),
                "velocity_vector": [round(dx, 1), round(dy, 1)],
            }
            active_tracks.append(track_payload)

            # Draw Trajectory Motion Trail
            pts = list(t_info["history"])
            for i in range(1, len(pts)):
                thickness = int(np.sqrt(self.max_history_points / float(i + 1)) * 2)
                cv2.line(
                    annotated,
                    (int(pts[i - 1][0]), int(pts[i - 1][1])),
                    (int(pts[i][0]), int(pts[i][1])),
                    (0, 255, 255),
                    thickness,
                )

            # Draw Bounding Box & Persistent Tag
            color = (0, 255, 0) if cls_name == "person" else (255, 128, 0)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"#{track_id} {cls_name} ({dwell_sec:.0f}s)"
            cv2.putText(
                annotated,
                label,
                (int(x1), max(20, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated,
                label,
                (int(x1), max(20, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
            )

        # 6. Purge stale tracks (> 10s inactivity)
        stale_ids = [tid for tid, info in self.tracks.items() if (now - info["last_seen"]) > 10.0]
        for tid in stale_ids:
            del self.tracks[tid]

        return active_tracks, annotated


tracker = ObjectTracker()
