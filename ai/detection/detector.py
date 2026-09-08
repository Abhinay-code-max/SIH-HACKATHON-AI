"""
Tactical Object Detection Engine & BaseDetector Interface.
Wraps YOLO inference behind a swappable interface with offline model resolution,
configurable thresholds, and RawDetection contract serialization.
Complies with Rule 2: Absolute Offline Edge Operation.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np
import yaml

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.detection.confidence_tracker import ConfidenceTracker
from ai.detection.preprocessing import FramePreprocessor
from ai.inference.loader import get_device, load_local_model
from backend.app.models.contracts import RawDetection

DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "detection_settings.yaml"
logger = logging.getLogger(__name__)


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


U2_DEFAULT_PATH = ROOT_DIR / "training_lab" / "runs" / "U2_yolov8m_640_9class_v002" / "weights" / "best.pt"
U2_EXPECTED_SHA256 = "cd8e50a9a84aef125450e81e79c8d4e5d8b1252604cd820167c5f094ac5e7930"


def resolve_registered_model(model_name: str = "auto") -> str:
    """
    Resolves the primary 9-class U2 detector path or registry-configured model.
    Prioritizes the accepted U2 candidate checkpoint:
    training_lab/runs/U2_yolov8m_640_9class_v002/weights/best.pt
    """
    if model_name not in ("auto", "u2", "U2"):
        p = Path(model_name)
        if p.is_file():
            return str(p)
        if (ROOT_DIR / p).is_file():
            return str(ROOT_DIR / p)
        return model_name

    # 1. Primary Accepted 9-Class Detector: U2
    if U2_DEFAULT_PATH.is_file():
        return str(U2_DEFAULT_PATH)

    # 2. Check registry index for active fine-tuned entries
    index_file = ROOT_DIR / "models" / "registry" / "registry_index.json"
    if index_file.is_file():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                idx = json.load(f)
            runs = idx.get("models", [])
            for entry in reversed(runs):
                if entry.get("status") == "active" or entry.get("version") in ("YOLO-U-v002", "U2"):
                    w = ROOT_DIR / entry.get("weights_path", "")
                    if w.is_file():
                        return str(w)
            for entry in reversed(runs):
                w = ROOT_DIR / entry.get("weights_path", "")
                if w.is_file():
                    return str(w)
        except Exception:
            pass

    # 3. Fallback to local base models if present
    for fallback_name in ("yolov8l.pt", "yolov8m.pt", "yolov8s.pt"):
        fb_path = ROOT_DIR / fallback_name
        if fb_path.is_file():
            return str(fb_path)

    # 4. Actionable error if U2 and local fallbacks are missing
    raise FileNotFoundError(
        f"U2 9-class model checkpoint is required at {U2_DEFAULT_PATH}. "
        "Please ensure the accepted U2 model weights are present. "
        "Air-gapped offline operation prohibits automatic model downloads."
    )


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
        half_precision: Optional[bool] = None,
        preprocessor: Optional[FramePreprocessor] = None,
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
        self.half_precision = (
            half_precision
            if half_precision is not None
            else bool(thresh_cfg.get("half_precision", False))
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
            # Prioritize primary accepted U2 model if available on disk
            if U2_DEFAULT_PATH.is_file():
                self.model_path = str(U2_DEFAULT_PATH)
            else:
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
        self.use_half = bool(self.half_precision and self.device == "cuda")
        if self.half_precision and self.device != "cuda":
            logger.warning(
                f"Half-precision (FP16) requested, but device is '{self.device}'. "
                "FP16 is only supported on CUDA devices; falling back to FP32."
            )
        self.model = None

        # 5. Resolve frame preprocessing pipeline (night/fog/IR enhancement)
        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            pre_cfg = self.config.get("preprocessing", {})
            self.preprocessor = FramePreprocessor(**pre_cfg)
        self.last_is_night_scene: bool = False

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

        # Apply optional frame preprocessing (night/fog/IR transforms) before inference
        if self.preprocessor is not None and getattr(self.preprocessor, "is_active", False):
            frame = self.preprocessor.preprocess(frame)

        # Task 18: Adaptive low-light / night scene trigger
        # Dynamically engage gamma brightening (1.5) when scene luminance < 60.0 without running CLAHE
        is_night_scene = False
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            if float(np.mean(gray)) < 60.0:
                is_night_scene = True
                if hasattr(self.preprocessor, "_apply_gamma"):
                    frame = self.preprocessor._apply_gamma(frame, gamma_override=1.5)
        except Exception as e:
            logger.warning(f"Adaptive low-light check failed: {e}")

        self.last_is_night_scene = is_night_scene

        self._ensure_model()
        h, w = frame.shape[:2]
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        now_sec = now_dt.timestamp()

        # Effective YOLO inference confidence floor (lowest among global and per-class thresholds)
        conf_floor = self.conf_threshold
        if self.classes_config:
            active_mins = [
                float(c.get("min_confidence", self.conf_threshold))
                for name, c in self.classes_config.items()
                if isinstance(c, dict) and (self.enabled_classes is None or name in self.enabled_classes)
            ]
            if active_mins:
                conf_floor = min(conf_floor, min(active_mins))

        predict_kwargs: Dict[str, Any] = {
            "source": frame,
            "conf": conf_floor,
            "iou": self.iou_threshold,
            "device": self.device,
            "verbose": False,
        }
        if self.use_half:
            predict_kwargs["half"] = True

        results = self.model.predict(**predict_kwargs)

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
                    is_night_scene=is_night_scene,
                    metadata={"is_night_scene": is_night_scene},
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
                is_night_scene=is_night_scene,
                metadata={"is_night_scene": is_night_scene},
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
