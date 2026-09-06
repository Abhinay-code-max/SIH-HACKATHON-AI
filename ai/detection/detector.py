"""
Tactical Object Detection Engine & BaseDetector Interface.
Wraps YOLO inference behind a swappable interface with offline model resolution,
configurable thresholds, and RawDetection contract serialization.
Complies with Rule 2: Absolute Offline Edge Operation.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import yaml

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.detection.confidence_tracker import ConfidenceTracker
from ai.inference.loader import get_device, load_local_model
from backend.app.models.contracts import RawDetection

DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "detection_settings.yaml"


def _box_iou(box1: List[float], box2: List[float]) -> float:
    """Compute 2D Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0.0 else 0.0


def load_detection_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads detection settings from YAML configuration."""
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not cfg_file.is_file():
        return {}
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_registered_model(model_name: str = "auto") -> str:
    """
    Resolves the latest registered fine-tuned model path from registry_index.json,
    mirroring ObjectTracker resolution, or defaults to yolov8l.pt.
    """
    if model_name != "auto":
        return model_name

    index_file = ROOT_DIR / "models" / "registry" / "registry_index.json"
    if index_file.is_file():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                idx = json.load(f)
            runs = idx.get("models", [])
            if runs:
                latest = runs[-1]
                weights = ROOT_DIR / latest.get("weights_path", "")
                if weights.is_file():
                    return str(weights)
        except Exception:
            pass

    return "yolov8l.pt"


class BaseDetector(ABC):
    """
    Abstract Base Detector interface.
    Decouples calling code (trackers, stream workers, evaluators)
    from specific vision backends (YOLO, ONNX, TensorRT).
    """

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        camera_id: Optional[str] = None,
    ) -> List[RawDetection]:
        """
        Execute inference on a single BGR image frame.

        Args:
            frame: Numpy BGR image array (H, W, C).
            camera_id: Optional camera identifier (e.g. 'CAM_01').

        Returns:
            List of RawDetection objects matching the system data contract.
        """
        pass


class YoloDetector(BaseDetector):
    """
    Concrete YOLO detector implementation.
    Wraps Ultralytics YOLO with offline resolution, per-class filtering,
    multi-frame confidence confirmation tracking, and automatic compute device selection (CUDA / CPU).
    """

    def __init__(
        self,
        model_name: str = "auto",
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        config_path: Optional[Union[str, Path]] = None,
        enabled_classes: Optional[Set[str]] = None,
        confirmation_enabled: Optional[bool] = None,
        consecutive_frames: Optional[int] = None,
        filter_unconfirmed: Optional[bool] = None,
    ):
        self.config = load_detection_config(config_path)

        # 1. Resolve thresholds from config or defaults
        thresh_cfg = self.config.get("thresholds", {})
        self.conf_threshold = (
            conf_threshold
            if conf_threshold is not None
            else float(thresh_cfg.get("confidence", 0.35))
        )
        self.iou_threshold = (
            iou_threshold
            if iou_threshold is not None
            else float(thresh_cfg.get("iou", 0.70))
        )

        # 2. Resolve target classes
        self.classes_config = self.config.get("classes", {})
        if enabled_classes is not None:
            self.enabled_classes = enabled_classes
        elif self.classes_config:
            self.enabled_classes = {
                name for name, c_cfg in self.classes_config.items()
                if isinstance(c_cfg, dict) and c_cfg.get("enabled", True)
            }
        else:
            self.enabled_classes = None  # None indicates allow all model classes

        # 3. Resolve model path
        if model_name == "auto":
            profile_name = self.config.get("profiles", {}).get("active_profile", "command_center")
            profile_model = self.config.get("profiles", {}).get(profile_name, {}).get("model_name", "auto")
            self.model_path = resolve_registered_model(profile_model)
        else:
            self.model_path = resolve_registered_model(model_name)

        # 4. Resolve multi-frame confidence confirmation settings
        confirm_cfg = self.config.get("confirmation", {})
        self.confirmation_enabled = (
            confirmation_enabled
            if confirmation_enabled is not None
            else bool(confirm_cfg.get("enabled", True))
        )
        self.consecutive_frames = (
            consecutive_frames
            if consecutive_frames is not None
            else int(confirm_cfg.get("consecutive_frames", 3))
        )
        self.confirm_min_conf = float(confirm_cfg.get("min_confidence", self.conf_threshold))
        self.max_history_age_sec = float(confirm_cfg.get("max_history_age_sec", 5.0))
        self.filter_unconfirmed = (
            filter_unconfirmed
            if filter_unconfirmed is not None
            else bool(confirm_cfg.get("filter_unconfirmed", False))
        )
        self.spatial_match_iou = float(confirm_cfg.get("spatial_match_iou", 0.30))

        # Per-camera state isolation for multi-stream safety
        self._camera_trackers: Dict[str, ConfidenceTracker] = {}
        self._camera_active_tracks: Dict[str, List[Dict[str, Any]]] = {}
        self._camera_next_id: Dict[str, int] = {}

        self.device = get_device()
        self.model = None

    def _ensure_model(self) -> None:
        """Lazy-load local model weights strictly from disk."""
        if self.model is None:
            self.model = load_local_model(self.model_path)

    def _get_camera_tracker(self, camera_id: str) -> ConfidenceTracker:
        """Retrieve or create a ConfidenceTracker instance isolated to this camera."""
        if camera_id not in self._camera_trackers:
            self._camera_trackers[camera_id] = ConfidenceTracker(
                consecutive_frames=self.consecutive_frames,
                min_confidence=self.confirm_min_conf,
                max_history_age_sec=self.max_history_age_sec,
            )
            self._camera_active_tracks[camera_id] = []
            self._camera_next_id[camera_id] = 1
        return self._camera_trackers[camera_id]

    def reset_confirmation(self, camera_id: Optional[str] = None) -> None:
        """Reset confirmation tracking states for a specific camera or all cameras."""
        if camera_id is None:
            self._camera_trackers.clear()
            self._camera_active_tracks.clear()
            self._camera_next_id.clear()
        else:
            if camera_id in self._camera_trackers:
                self._camera_trackers[camera_id].reset()
            self._camera_active_tracks.pop(camera_id, None)
            self._camera_next_id.pop(camera_id, None)

    def detect(
        self,
        frame: np.ndarray,
        camera_id: Optional[str] = None,
    ) -> List[RawDetection]:
        """
        Runs local inference on frame and returns standardized RawDetection items.
        Applies multi-frame confidence confirmation tracking when enabled.
        """
        if frame is None or frame.size == 0:
            return []

        self._ensure_model()
        h, w = frame.shape[:2]
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        now_sec = now_dt.timestamp()

        # Execute local inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[RawDetection] = []
        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        # Preliminary candidates after per-class filtering
        candidates: List[Dict[str, Any]] = []
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0].item())

            # Filter by enabled classes if configured
            if self.enabled_classes is not None and cls_name not in self.enabled_classes:
                continue

            # Filter by per-class minimum confidence if configured
            class_cfg = self.classes_config.get(cls_name, {})
            if isinstance(class_cfg, dict):
                min_class_conf = class_cfg.get("min_confidence")
                if min_class_conf is not None and conf < float(min_class_conf):
                    continue

            coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = coords
            norm_center = [
                round(((x1 + x2) / 2.0) / max(1, w), 4),
                round(((y1 + y2) / 2.0) / max(1, h), 4),
            ]

            candidates.append({
                "idx": idx,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox": coords,
                "normalized_center": norm_center,
            })

        cam_key = camera_id or "default"

        if not self.confirmation_enabled:
            for cand in candidates:
                det = RawDetection(
                    detection_id=f"det_{camera_id or 'cam'}_{cand['idx'] + 1:04d}",
                    class_name=cand["class_name"],
                    confidence=cand["confidence"],
                    bbox=cand["bbox"],
                    normalized_center=cand["normalized_center"],
                    object_id=None,
                    camera_id=camera_id,
                    timestamp=now_iso,
                    confirmed=True,
                )
                detections.append(det)
            return detections

        # Multi-frame confidence confirmation tracking
        tracker = self._get_camera_tracker(cam_key)
        prev_tracks = self._camera_active_tracks.get(cam_key, [])

        # Prune stale active tracks
        tracker.prune_stale(max_age_sec=self.max_history_age_sec, current_time=now_sec)
        prev_tracks = [t for t in prev_tracks if (now_sec - t["last_seen"]) <= self.max_history_age_sec]

        # Spatial IoU matching across consecutive frames
        updated_tracks: List[Dict[str, Any]] = []
        matched_prev_indices: Set[int] = set()

        for cand in candidates:
            best_iou = 0.0
            best_prev_idx = -1

            for p_idx, prev in enumerate(prev_tracks):
                if p_idx in matched_prev_indices:
                    continue
                if prev["class_name"] != cand["class_name"]:
                    continue

                iou = _box_iou(cand["bbox"], prev["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_prev_idx = p_idx

            if best_iou >= self.spatial_match_iou and best_prev_idx >= 0:
                matched_prev_indices.add(best_prev_idx)
                assigned_id = prev_tracks[best_prev_idx]["track_id"]
                updated_tracks.append({
                    "track_id": assigned_id,
                    "class_name": cand["class_name"],
                    "bbox": cand["bbox"],
                    "last_seen": now_sec,
                })
            else:
                assigned_id = self._camera_next_id[cam_key]
                self._camera_next_id[cam_key] += 1
                updated_tracks.append({
                    "track_id": assigned_id,
                    "class_name": cand["class_name"],
                    "bbox": cand["bbox"],
                    "last_seen": now_sec,
                })

            is_confirmed = tracker.update(
                object_id=assigned_id,
                class_name=cand["class_name"],
                confidence=cand["confidence"],
                timestamp=now_sec,
            )

            det = RawDetection(
                detection_id=f"det_{camera_id or 'cam'}_{cand['idx'] + 1:04d}",
                class_name=cand["class_name"],
                confidence=cand["confidence"],
                bbox=cand["bbox"],
                normalized_center=cand["normalized_center"],
                object_id=None,  # Unassigned before spatial tracking
                camera_id=camera_id,
                timestamp=now_iso,
                confirmed=is_confirmed,
            )
            detections.append(det)

        # Update per-camera active tracks cache
        self._camera_active_tracks[cam_key] = updated_tracks

        # Optional filtering if requested via config or init parameter
        if self.filter_unconfirmed:
            detections = [d for d in detections if d.confirmed is True]

        return detections

    def detect_confirmed(
        self,
        frame: np.ndarray,
        camera_id: Optional[str] = None,
    ) -> List[RawDetection]:
        """Convenience method to execute detection and return only confirmed targets."""
        all_dets = self.detect(frame, camera_id=camera_id)
        return [d for d in all_dets if d.confirmed is True]
