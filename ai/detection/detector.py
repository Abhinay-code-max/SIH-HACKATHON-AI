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
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import yaml

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.inference.loader import get_device, load_local_model
from backend.app.models.contracts import RawDetection

DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "detection_settings.yaml"


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
    and automatic compute device selection (CUDA / CPU).
    """

    def __init__(
        self,
        model_name: str = "auto",
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        config_path: Optional[Union[str, Path]] = None,
        enabled_classes: Optional[Set[str]] = None,
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

        self.device = get_device()
        self.model = None

    def _ensure_model(self) -> None:
        """Lazy-load local model weights strictly from disk."""
        if self.model is None:
            self.model = load_local_model(self.model_path)

    def detect(
        self,
        frame: np.ndarray,
        camera_id: Optional[str] = None,
    ) -> List[RawDetection]:
        """
        Runs local inference on frame and returns standardized RawDetection items.
        """
        if frame is None or frame.size == 0:
            return []

        self._ensure_model()
        h, w = frame.shape[:2]
        now_iso = datetime.now(timezone.utc).isoformat()

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

            det = RawDetection(
                detection_id=f"det_{camera_id or 'cam'}_{idx + 1:04d}",
                class_name=cls_name,
                confidence=round(conf, 4),
                bbox=coords,
                normalized_center=norm_center,
                object_id=None,  # Unassigned before spatial tracking
                camera_id=camera_id,
                timestamp=now_iso,
            )
            detections.append(det)

        return detections
