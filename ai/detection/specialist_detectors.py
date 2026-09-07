"""
Specialist Object Detection Modules for Border Sentinel.
Integrates community-sourced, fine-tuned YOLO specialist models:
1. WeaponSpecialistDetector (weapons: guns, rifles, pistols, knives)
2. DroneSpecialistDetector (airborne drones and UAVs)
3. FireSmokeSpecialistDetector (fire and smoke with explicit class index mapping)

Conforms to the abstract BaseDetector interface, serializing standardized
RawDetection contract objects. Offline-first: models are cached locally in
ai/models/specialists/ with zero network calls at runtime after first-run download.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.detection.detector import BaseDetector
from ai.inference.loader import get_device
from backend.app.models.contracts import RawDetection

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger("ai.detection.specialists")

DEFAULT_SPECIALISTS_DIR = ROOT_DIR / "ai" / "models" / "specialists"


class BaseSpecialistDetector(BaseDetector, ABC):
    """
    Base class for pretrained specialist YOLO detectors.
    Provides offline-first weight resolution, one-time HuggingFace download caching,
    model loading, and standardized RawDetection formatting.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.60,
        device: Optional[str] = None,
        half_precision: bool = False,
        auto_download: bool = False,
        enabled: bool = True,
    ):
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = device or get_device()
        self.half_precision = half_precision
        self.use_half = bool(self.half_precision and self.device == "cuda")
        self.auto_download = auto_download
        self.enabled = bool(enabled)
        self.status = "active" if self.enabled else "in_development"

        self.weights_path = Path(weights_path) if weights_path is not None else self.default_weights_path
        self.model = None

    @property
    @abstractmethod
    def default_weights_path(self) -> Path:
        """Default local path for cached model weights."""
        pass

    @property
    @abstractmethod
    def hf_repo_id(self) -> str:
        """HuggingFace repository ID."""
        pass

    @property
    @abstractmethod
    def hf_filename(self) -> str:
        """Target weights filename in the HuggingFace repository."""
        pass

    @property
    def hf_repo_type(self) -> str:
        """Repository type: 'model' or 'space'."""
        return "model"

    @property
    @abstractmethod
    def detector_tag(self) -> str:
        """Short identifier tag used in detection IDs (e.g. 'weap', 'drn', 'fs')."""
        pass

    @abstractmethod
    def _map_class_name(self, cls_id: int, raw_name: str) -> str:
        """Maps model-specific class ID or name to Border Sentinel taxonomy."""
        pass

    def _ensure_weights(self) -> Path:
        """Resolves local weights or downloads once from HuggingFace."""
        if self.weights_path.is_file():
            return self.weights_path

        if not self.auto_download:
            raise FileNotFoundError(
                f"[{self.__class__.__name__}] Specialist weights not found at: {self.weights_path}. "
                "Automatic download is disabled by default to enforce 100% offline tactical deployment (Rule 2). "
                f"Place model weights manually at {self.weights_path} or pass auto_download=True for one-time developer setup."
            )

        if hf_hub_download is None:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Weights missing at {self.weights_path} and 'huggingface_hub' "
                "is not installed. Install with 'pip install huggingface_hub' to enable one-time caching."
            )

        logger.info(
            f"[{self.__class__.__name__}] Downloading weights from HuggingFace "
            f"({self.hf_repo_id} / {self.hf_filename}, type={self.hf_repo_type})..."
        )
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            downloaded_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=self.hf_filename,
                repo_type=self.hf_repo_type,
            )
            shutil.copy2(downloaded_path, self.weights_path)
            logger.info(f"[{self.__class__.__name__}] Cached specialist weights to {self.weights_path}")
        except Exception as e:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Failed to download weights from {self.hf_repo_id}: {e}"
            ) from e

        return self.weights_path

    def _ensure_model(self) -> Any:
        """Lazy-loads local YOLO model weights."""
        if not self.enabled:
            logger.info(
                f"[{self.__class__.__name__}] Specialist is disabled (status={self.status}). Skipping model load (0 VRAM)."
            )
            return None

        if self.model is None:
            if YOLO is None:
                raise ImportError("ultralytics package is required to run specialist detectors.")
            weights_file = self._ensure_weights()
            self.model = YOLO(str(weights_file))
            # Output model names as explicitly requested
            print(f"[{self.__class__.__name__}] Loaded weights from {weights_file.name} | model.names: {self.model.names}")
            logger.info(f"[{self.__class__.__name__}] Loaded model names: {self.model.names}")
        return self.model

    def detect(
        self,
        frame: np.ndarray,
        camera_id: Optional[str] = None,
    ) -> List[RawDetection]:
        """
        Runs local specialist inference on frame and returns standardized RawDetection items.
        """
        if not self.enabled:
            return []

        if frame is None or frame.size == 0:
            return []

        if self._ensure_model() is None:
            return []
        h, w = frame.shape[:2]
        now_iso = datetime.now(timezone.utc).isoformat()

        predict_kwargs: Dict[str, Any] = {
            "source": frame,
            "conf": self.conf_threshold,
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

        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            raw_name = self.model.names.get(cls_id, f"class_{cls_id}")
            mapped_name = self._map_class_name(cls_id, raw_name)
            conf = float(box.conf[0].item())

            coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = coords
            norm_center = [
                round(((x1 + x2) / 2.0) / max(1, w), 4),
                round(((y1 + y2) / 2.0) / max(1, h), 4),
            ]

            det = RawDetection(
                detection_id=f"det_{camera_id or 'cam'}_{self.detector_tag}_{idx + 1:04d}",
                class_name=mapped_name,
                confidence=round(conf, 4),
                bbox=coords,
                normalized_center=norm_center,
                object_id=None,
                camera_id=camera_id,
                timestamp=now_iso,
                confirmed=True,
            )
            detections.append(det)

        return detections


class WeaponSpecialistDetector(BaseSpecialistDetector):
    """
    Specialist detector for weapons (guns, rifles, pistols, knives).
    Source model: Dricz/Weapon_Detection_YOLOv8 (HuggingFace Space).

    NOTE: Disabled by default due to high false-positive rate on real-world CCTV footage
    (7.1% - 21.4% across tested models with significant confidence score overlap against true positives).
    Retained for future fine-tuning or evaluation. Zero VRAM and zero inference latency incurred when disabled.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.60,
        device: Optional[str] = None,
        half_precision: bool = False,
        auto_download: bool = False,
        # Disabled by default due to high false-positive rate on real-world CCTV footage
        # (7.1% - 21.4% across tested models with significant confidence score overlap against true positives).
        # Retained for future fine-tuning or evaluation. Status: in_development.
        enabled: bool = False,
    ):
        super().__init__(
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            half_precision=half_precision,
            auto_download=auto_download,
            enabled=enabled,
        )

    @property
    def default_weights_path(self) -> Path:
        return DEFAULT_SPECIALISTS_DIR / "weapon_best.pt"

    @property
    def hf_repo_id(self) -> str:
        return "Dricz/Weapon_Detection_YOLOv8"

    @property
    def hf_filename(self) -> str:
        return "best (1).pt"

    @property
    def hf_repo_type(self) -> str:
        return "space"

    @property
    def detector_tag(self) -> str:
        return "weap"

    def _map_class_name(self, cls_id: int, raw_name: str) -> str:
        # All weapon categories (grenade, knife, pistol, rifle) unify to 'weapon'
        return "weapon"


class DroneSpecialistDetector(BaseSpecialistDetector):
    """
    Specialist detector for airborne drones and UAVs.
    Source model: TomSmail/drone-yolo-v1 (HuggingFace Model).
    Classes: 0: quadcopter, 1: fixed-wing.
    Maps all detected drone sub-types to standard class_name="drone".
    """

    @property
    def default_weights_path(self) -> Path:
        return DEFAULT_SPECIALISTS_DIR / "drone_best.pt"

    @property
    def hf_repo_id(self) -> str:
        return "TomSmail/drone-yolo-v1"

    @property
    def hf_filename(self) -> str:
        return "best.pt"

    @property
    def hf_repo_type(self) -> str:
        return "model"

    @property
    def detector_tag(self) -> str:
        return "drn"

    def _map_class_name(self, cls_id: int, raw_name: str) -> str:
        # All drone models (quadcopter, fixed-wing) unify to 'drone'
        return "drone"


class FireSmokeSpecialistDetector(BaseSpecialistDetector):
    """
    Specialist detector for fire and smoke.
    Source model: rabahdev/fire-smoke-yolov8n (HuggingFace Model).
    Source indexing: class 0 = smoke, class 1 = fire.
    Dynamically verifies class mapping against model.names at runtime.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._class_map: Optional[Dict[int, str]] = None

    @property
    def default_weights_path(self) -> Path:
        return DEFAULT_SPECIALISTS_DIR / "fire_smoke_best.pt"

    @property
    def hf_repo_id(self) -> str:
        return "rabahdev/fire-smoke-yolov8n"

    @property
    def hf_filename(self) -> str:
        return "best.pt"

    @property
    def hf_repo_type(self) -> str:
        return "model"

    @property
    def detector_tag(self) -> str:
        return "fs"

    def _ensure_model(self) -> Any:
        m = super()._ensure_model()
        if self._class_map is None:
            self._class_map = {}
            for cid, cname in m.names.items():
                lower = str(cname).strip().lower()
                if "smoke" in lower:
                    self._class_map[cid] = "smoke"
                elif "fire" in lower:
                    self._class_map[cid] = "fire"
                else:
                    # Fallback based on known index
                    self._class_map[cid] = "smoke" if cid == 0 else "fire"
            logger.info(f"[FireSmokeSpecialistDetector] Resolved runtime class map: {self._class_map}")
        return m

    def _map_class_name(self, cls_id: int, raw_name: str) -> str:
        if self._class_map and cls_id in self._class_map:
            return self._class_map[cls_id]
        # Direct name check fallback
        lower = str(raw_name).strip().lower()
        if "smoke" in lower:
            return "smoke"
        if "fire" in lower:
            return "fire"
        return "smoke" if cls_id == 0 else "fire"


class CompositeSpecialistDetector(BaseDetector):
    """
    Orchestrator that combines a primary BaseDetector (e.g. YoloDetector) with
    pluggable specialist detectors on the same frame, merging all detections into
    a single unified List[RawDetection].

    This is an optional extension that does not alter default YoloDetector pipelines.
    """

    def __init__(
        self,
        primary_detector: Optional[BaseDetector] = None,
        specialists: Optional[List[BaseDetector]] = None,
        auto_init_specialists: bool = False,
    ):
        self.primary_detector = primary_detector
        if specialists is not None:
            self.specialists = specialists
        elif auto_init_specialists:
            self.specialists = [
                WeaponSpecialistDetector(),
                DroneSpecialistDetector(),
                FireSmokeSpecialistDetector(),
            ]
        else:
            self.specialists = []

    def add_specialist(self, specialist: BaseDetector) -> None:
        """Register an additional specialist detector."""
        self.specialists.append(specialist)

    def detect(
        self,
        frame: np.ndarray,
        camera_id: Optional[str] = None,
    ) -> List[RawDetection]:
        """
        Executes primary detector and all registered specialists, returning
        a merged list of RawDetection objects.
        """
        if frame is None or frame.size == 0:
            return []

        all_detections: List[RawDetection] = []

        # 1. Primary detector pass (if present)
        if self.primary_detector is not None:
            try:
                primary_dets = self.primary_detector.detect(frame, camera_id=camera_id)
                all_detections.extend(primary_dets)
            except Exception as e:
                logger.error(f"Error executing primary detector: {e}")

        # 2. Specialist detector passes
        for idx, specialist in enumerate(self.specialists):
            # Skip disabled specialists immediately (0 VRAM, 0 ms latency)
            if not getattr(specialist, "enabled", True):
                continue
            try:
                spec_dets = specialist.detect(frame, camera_id=camera_id)
                all_detections.extend(spec_dets)
            except Exception as e:
                logger.error(f"Error executing specialist {specialist.__class__.__name__}: {e}")

        return all_detections

    def get_specialist_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns status of registered specialists for telemetry, API health, or UI demo layer.
        Reports whether each specialist is active or in_development (disabled).
        """
        status: Dict[str, Dict[str, Any]] = {}
        for s in self.specialists:
            name = s.__class__.__name__
            status[name] = {
                "enabled": getattr(s, "enabled", True),
                "status": getattr(s, "status", "active" if getattr(s, "enabled", True) else "in_development"),
                "weights_path": str(getattr(s, "weights_path", "")),
                "model_loaded": getattr(s, "model", None) is not None,
            }
        return status


def run_composite_detection(
    frame: np.ndarray,
    primary_detector: Optional[BaseDetector] = None,
    specialists: Optional[List[BaseDetector]] = None,
    camera_id: Optional[str] = None,
) -> List[RawDetection]:
    """
    Functional helper to execute primary and specialist detectors on a frame.
    """
    composite = CompositeSpecialistDetector(
        primary_detector=primary_detector,
        specialists=specialists,
    )
    return composite.detect(frame, camera_id=camera_id)
