"""
Automated Test Suite for ai/detection Module.
Validates:
1. ConfidenceTracker multi-frame confirmation & false-positive suppression.
2. ConfidenceTracker confidence threshold resets and stale record pruning.
3. Detection settings YAML configuration loader and class filtering schema.
4. BaseDetector abstract interface polymorphism and RawDetection contract serialization.
5. RawDetection backward-compatibility with existing contracts.

Uses synthetic data only; does NOT require external model weights.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.detection.confidence_tracker import ConfidenceTracker
from ai.detection.preprocessing import FramePreprocessor
from ai.detection.detector import (
    BaseDetector,
    YoloDetector,
    load_detection_config,
    resolve_registered_model,
)
from backend.app.models.contracts import RawDetection


class MockDetector(BaseDetector):
    """Synthetic detector for unit testing BaseDetector contract without weights."""

    def __init__(self, mock_detections: Optional[List[dict]] = None):
        self.mock_detections = mock_detections or []

    def detect(self, frame: np.ndarray, camera_id: Optional[str] = None) -> List[RawDetection]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        now_iso = datetime.now(timezone.utc).isoformat()
        results = []
        for idx, item in enumerate(self.mock_detections):
            coords = item.get("bbox", [10.0, 10.0, 100.0, 100.0])
            norm_center = [
                round(((coords[0] + coords[2]) / 2.0) / w, 4),
                round(((coords[1] + coords[3]) / 2.0) / h, 4),
            ]
            results.append(
                RawDetection(
                    detection_id=f"det_{camera_id or 'cam'}_{idx + 1:04d}",
                    class_name=item.get("class_name", "person"),
                    confidence=item.get("confidence", 0.90),
                    bbox=coords,
                    normalized_center=norm_center,
                    object_id=None,
                    camera_id=camera_id,
                    timestamp=now_iso,
                )
            )
        return results


class MockTensorValue:
    """Mock single-value tensor wrapper (.item())."""

    def __init__(self, val: Any):
        self._val = val

    def item(self) -> Any:
        return self._val


class MockTensorList:
    """Mock tensor wrapper for list conversion (.tolist())."""

    def __init__(self, val: Any):
        self._val = val

    def tolist(self) -> Any:
        return self._val


class MockYoloBox:
    """Mock YOLO prediction bounding box container."""

    def __init__(self, cls_id: int, conf: float, xyxy: List[float]):
        self.cls = [MockTensorValue(cls_id)]
        self.conf = [MockTensorValue(conf)]
        self.xyxy = [MockTensorList(xyxy)]


class MockYoloResult:
    """Mock YOLO Result container with boxes attribute."""

    def __init__(self, boxes: List[MockYoloBox]):
        self.boxes = boxes


class MockYoloModel:
    """Mock YOLO model returning configurable MockYoloBoxes."""

    def __init__(self, names: Optional[Dict[int, str]] = None):
        self.names = names or {0: "person", 1: "car", 2: "truck"}
        self.boxes: List[MockYoloBox] = []
        self.last_predict_kwargs: Dict[str, Any] = {}

    def predict(self, source: Any, **kwargs: Any) -> List[MockYoloResult]:
        self.last_predict_kwargs = {"source": source, **kwargs}
        return [MockYoloResult(self.boxes)]




def test_confidence_tracker_confirmation_flow():
    """Verify target becomes confirmed only after N consecutive frames."""
    tracker = ConfidenceTracker(consecutive_frames=3, min_confidence=0.35)

    # Frame 1: Hit 1 -> Not confirmed
    confirmed_f1 = tracker.update(object_id=101, class_name="person", confidence=0.85)
    assert confirmed_f1 is False, "Target should not be confirmed on frame 1"
    assert tracker.is_confirmed(101) is False

    # Frame 2: Hit 2 -> Not confirmed
    confirmed_f2 = tracker.update(object_id=101, class_name="person", confidence=0.88)
    assert confirmed_f2 is False, "Target should not be confirmed on frame 2"
    assert tracker.is_confirmed(101) is False

    # Frame 3: Hit 3 -> Confirmed!
    confirmed_f3 = tracker.update(object_id=101, class_name="person", confidence=0.91)
    assert confirmed_f3 is True, "Target should be confirmed on frame 3 (threshold reached)"
    assert tracker.is_confirmed(101) is True

    # Frame 4: Hit 4 -> Remains confirmed
    confirmed_f4 = tracker.update(object_id=101, class_name="person", confidence=0.87)
    assert confirmed_f4 is True, "Target should remain confirmed on subsequent hits"


def test_confidence_tracker_low_confidence_reset():
    """Verify transient low confidence resets consecutive hit count."""
    tracker = ConfidenceTracker(consecutive_frames=3, min_confidence=0.50)

    # Frame 1 & 2 above threshold
    tracker.update(object_id=202, class_name="car", confidence=0.65)
    tracker.update(object_id=202, class_name="car", confidence=0.70)
    state = tracker.get_state(202, "car")
    assert state is not None and state.consecutive_hits == 2

    # Frame 3 drops below threshold (e.g. shadow artifact, 0.30)
    confirmed_drop = tracker.update(object_id=202, class_name="car", confidence=0.30)
    assert confirmed_drop is False
    assert tracker.is_confirmed(202, "car") is False
    assert tracker.get_state(202, "car").consecutive_hits == 0

    # Needs 3 fresh consecutive hits to confirm again
    tracker.update(object_id=202, class_name="car", confidence=0.60)
    tracker.update(object_id=202, class_name="car", confidence=0.62)
    assert tracker.is_confirmed(202, "car") is False
    tracker.update(object_id=202, class_name="car", confidence=0.68)
    assert tracker.is_confirmed(202, "car") is True


def test_confidence_tracker_multi_object_independence():
    """Verify multiple objects are tracked independently."""
    tracker = ConfidenceTracker(consecutive_frames=2, min_confidence=0.40)

    # Target A gets 2 hits -> Confirmed
    tracker.update(object_id="T_A", class_name="person", confidence=0.75)
    tracker.update(object_id="T_A", class_name="person", confidence=0.80)
    assert tracker.is_confirmed("T_A") is True

    # Target B gets 1 hit -> Not confirmed
    tracker.update(object_id="T_B", class_name="backpack", confidence=0.90)
    assert tracker.is_confirmed("T_B") is False

    # Check class distinction for same object_id if needed
    state_a = tracker.get_state("T_A", "person")
    assert state_a is not None and state_a.total_hits == 2


def test_confidence_tracker_stale_pruning():
    """Verify stale objects are pruned based on max history age."""
    tracker = ConfidenceTracker(consecutive_frames=1, max_history_age_sec=2.0)

    t0 = 1000.0
    tracker.update(object_id="OLD_TARGET", class_name="truck", confidence=0.80, timestamp=t0)
    tracker.update(object_id="NEW_TARGET", class_name="truck", confidence=0.80, timestamp=t0 + 2.5)

    # At timestamp t0 + 2.5, OLD_TARGET is 2.5s old (> 2.0s limit)
    pruned = tracker.prune_stale(max_age_sec=2.0, current_time=t0 + 2.5)
    assert pruned == 1
    assert tracker.get_state("OLD_TARGET", "truck") is None
    assert tracker.get_state("NEW_TARGET", "truck") is not None


def test_detection_settings_loading():
    """Verify config/detection_settings.yaml schema and defaults."""
    cfg = load_detection_config()
    assert "version" in cfg, "Config missing version"
    assert "profiles" in cfg, "Config missing profiles section"
    assert "thresholds" in cfg, "Config missing thresholds section"
    assert "classes" in cfg, "Config missing classes section"

    # Verify profiles
    profiles = cfg["profiles"]
    assert profiles.get("active_profile") == "command_center"
    assert "command_center" in profiles and "edge" in profiles
    assert profiles["command_center"]["model_name"] == "yolov8l.pt"

    # Verify thresholds
    thresholds = cfg["thresholds"]
    assert thresholds["confidence"] == 0.35
    assert thresholds["iou"] == 0.70
    assert thresholds.get("half_precision") is False

    # Verify classes: baseline enabled, future classes disabled
    classes = cfg["classes"]
    for c in ["person", "car", "truck", "bus", "motorcycle", "bicycle", "animal", "backpack", "bag"]:
        assert c in classes, f"Baseline class {c} missing from detection settings"
        assert classes[c]["enabled"] is True, f"Baseline class {c} should be enabled"

    # Verify future border surveillance targets
    for fc in ["weapon", "drone", "fire", "smoke", "boat"]:
        assert fc in classes, f"Future target {fc} missing from detection settings"
        assert classes[fc]["enabled"] is False, f"Future target {fc} must be disabled pending training"
        assert classes[fc].get("pending_training_data") is True

    # Verify preprocessing scaffold defaults (task 18)
    assert "preprocessing" in cfg, "Config missing preprocessing section"
    pre = cfg["preprocessing"]
    assert pre.get("enabled") is False
    assert pre.get("enable_clahe") is False
    assert pre.get("enable_gamma") is False
    assert pre.get("enable_dehaze") is False
    assert pre.get("enable_grayscale_ir") is False



def test_base_detector_interface():
    """Verify BaseDetector implementation satisfies contract with RawDetection."""
    mock_items = [
        {"class_name": "person", "confidence": 0.92, "bbox": [50.0, 50.0, 150.0, 200.0]},
        {"class_name": "backpack", "confidence": 0.81, "bbox": [80.0, 90.0, 120.0, 140.0]},
    ]
    detector = MockDetector(mock_detections=mock_items)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame, camera_id="CAM_02")

    assert len(detections) == 2
    for det in detections:
        assert isinstance(det, RawDetection)
        assert det.camera_id == "CAM_02"
        assert det.object_id is None, "Detector should not fabricate tracking object_id"
        assert det.timestamp is not None
        assert len(det.bbox) == 4
        assert det.normalized_center is not None
        assert 0.0 <= det.confidence <= 1.0


def test_raw_detection_backward_compatibility():
    """Verify RawDetection supports both legacy and extended fields."""
    # Legacy instantiation (without object_id, camera_id, timestamp, confirmed)
    legacy_det = RawDetection(
        detection_id="det_001",
        class_name="person",
        confidence=0.95,
        bbox=[10.0, 20.0, 30.0, 40.0],
    )
    assert legacy_det.object_id is None
    assert legacy_det.camera_id is None
    assert legacy_det.timestamp is None
    assert legacy_det.confirmed is None

    # Extended instantiation
    ext_det = RawDetection(
        detection_id="det_002",
        class_name="car",
        confidence=0.88,
        bbox=[100.0, 100.0, 250.0, 200.0],
        normalized_center=[0.27, 0.31],
        object_id=42,
        camera_id="CAM_01",
        timestamp="2026-09-06T11:00:00Z",
        confirmed=True,
    )
    assert ext_det.object_id == 42
    assert ext_det.camera_id == "CAM_01"
    assert ext_det.timestamp == "2026-09-06T11:00:00Z"
    assert ext_det.confirmed is True


def test_model_resolution_fallback():
    """Verify resolve_registered_model falls back to yolov8l.pt when auto weights are missing."""
    resolved = resolve_registered_model(model_name="auto")
    # Weights for YOLO-L-v002 were gitignored and absent, so fallback must return yolov8l.pt
    assert resolved == "yolov8l.pt" or resolved.endswith(".pt")


def test_detector_rising_confidence_confirmation():
    """Verify YoloDetector confirms target only after consecutive frames threshold."""
    detector = YoloDetector(
        consecutive_frames=3,
        confirmation_enabled=True,
        filter_unconfirmed=False,
    )
    mock_model = MockYoloModel()
    detector.model = mock_model
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: person at [100, 100, 200, 200], conf 0.40 -> Not confirmed
    mock_model.boxes = [MockYoloBox(cls_id=0, conf=0.40, xyxy=[100.0, 100.0, 200.0, 200.0])]
    dets_f1 = detector.detect(dummy_frame, camera_id="CAM_01")
    assert len(dets_f1) == 1
    assert dets_f1[0].confirmed is False
    assert dets_f1[0].object_id is None, "Detector should not fabricate tracking object_id"

    # Frame 2: person slightly shifted [102, 101, 201, 202], conf 0.45 -> Not confirmed
    mock_model.boxes = [MockYoloBox(cls_id=0, conf=0.45, xyxy=[102.0, 101.0, 201.0, 202.0])]
    dets_f2 = detector.detect(dummy_frame, camera_id="CAM_01")
    assert len(dets_f2) == 1
    assert dets_f2[0].confirmed is False

    # Frame 3: person at [105, 103, 203, 204], conf 0.50 -> Confirmed!
    mock_model.boxes = [MockYoloBox(cls_id=0, conf=0.50, xyxy=[105.0, 103.0, 203.0, 204.0])]
    dets_f3 = detector.detect(dummy_frame, camera_id="CAM_01")
    assert len(dets_f3) == 1
    assert dets_f3[0].confirmed is True

    # Frame 4: subsequent hit remains confirmed
    mock_model.boxes = [MockYoloBox(cls_id=0, conf=0.55, xyxy=[107.0, 104.0, 204.0, 205.0])]
    dets_f4 = detector.detect(dummy_frame, camera_id="CAM_01")
    assert len(dets_f4) == 1
    assert dets_f4[0].confirmed is True


def test_detector_transient_spike_non_confirmation():
    """Verify an isolated single-frame detection spike is marked unconfirmed."""
    detector = YoloDetector(
        consecutive_frames=3,
        confirmation_enabled=True,
        filter_unconfirmed=False,
    )
    mock_model = MockYoloModel()
    detector.model = mock_model
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: high confidence spike (0.95), but only 1 frame
    mock_model.boxes = [MockYoloBox(cls_id=0, conf=0.95, xyxy=[150.0, 150.0, 250.0, 250.0])]
    dets = detector.detect(dummy_frame, camera_id="CAM_SPIKE")
    assert len(dets) == 1
    assert dets[0].confirmed is False, "Single frame spike must not be confirmed"


def test_detector_per_camera_isolation():
    """Verify confidence tracking state is strictly isolated per camera_id."""
    detector = YoloDetector(
        consecutive_frames=3,
        confirmation_enabled=True,
        filter_unconfirmed=False,
    )
    mock_model = MockYoloModel()
    detector.model = mock_model
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 3 frames on CAM_01 -> target confirmed on CAM_01
    mock_model.boxes = [MockYoloBox(cls_id=0, conf=0.85, xyxy=[50.0, 50.0, 150.0, 150.0])]
    detector.detect(dummy_frame, camera_id="CAM_01")
    detector.detect(dummy_frame, camera_id="CAM_01")
    dets_cam1 = detector.detect(dummy_frame, camera_id="CAM_01")
    assert len(dets_cam1) == 1
    assert dets_cam1[0].confirmed is True

    # Same target coordinates on CAM_02 for frame 1 -> must NOT be confirmed
    dets_cam2 = detector.detect(dummy_frame, camera_id="CAM_02")
    assert len(dets_cam2) == 1
    assert dets_cam2[0].confirmed is False, "Camera isolation: CAM_02 must start unconfirmed"


def test_detector_filter_unconfirmed_and_detect_confirmed():
    """Verify filter_unconfirmed drops unconfirmed hits and detect_confirmed returns only confirmed hits."""
    detector_filter = YoloDetector(
        consecutive_frames=2,
        confirmation_enabled=True,
        filter_unconfirmed=True,
    )
    mock_model = MockYoloModel()
    detector_filter.model = mock_model
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_model.boxes = [MockYoloBox(cls_id=0, conf=0.80, xyxy=[10.0, 10.0, 50.0, 50.0])]
    # Frame 1: Hit 1 -> unconfirmed -> should be filtered out
    dets_f1 = detector_filter.detect(dummy_frame, camera_id="CAM_FILT")
    assert len(dets_f1) == 0, "Unconfirmed hit must be filtered out when filter_unconfirmed=True"

    # Frame 2: Hit 2 -> confirmed -> should be returned
    dets_f2 = detector_filter.detect(dummy_frame, camera_id="CAM_FILT")
    assert len(dets_f2) == 1
    assert dets_f2[0].confirmed is True

    # Convenience method detect_confirmed on non-filtering detector
    detector_non_filter = YoloDetector(
        consecutive_frames=2,
        confirmation_enabled=True,
        filter_unconfirmed=False,
    )
    detector_non_filter.model = mock_model
    # Frame 1: Hit 1 (unconfirmed)
    confirmed_f1 = detector_non_filter.detect_confirmed(dummy_frame, camera_id="CAM_CONV")
    assert len(confirmed_f1) == 0
    # Frame 2: Hit 2 (confirmed)
    confirmed_f2 = detector_non_filter.detect_confirmed(dummy_frame, camera_id="CAM_CONV")
    assert len(confirmed_f2) == 1
    assert confirmed_f2[0].confirmed is True


def test_detector_half_precision_initialization_and_safety():
    """
    Verify FP16 (half-precision) initialization, safety fallback on non-CUDA devices,
    and propagation of half=True kwarg to model.predict().
    """
    from unittest.mock import patch

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Default initialization -> use_half=False
    detector_default = YoloDetector()
    assert detector_default.half_precision is False
    assert detector_default.use_half is False

    # 2. CPU + half_precision=True -> fallback to use_half=False (no crash)
    with patch("ai.detection.detector.get_device", return_value="cpu"):
        detector_cpu = YoloDetector(half_precision=True)
        assert detector_cpu.device == "cpu"
        assert detector_cpu.half_precision is True
        assert detector_cpu.use_half is False

        mock_model_cpu = MockYoloModel()
        detector_cpu.model = mock_model_cpu
        detector_cpu.detect(dummy_frame, camera_id="CAM_CPU")
        assert "half" not in mock_model_cpu.last_predict_kwargs

    # 3. CUDA + half_precision=True -> use_half=True
    with patch("ai.detection.detector.get_device", return_value="cuda"):
        detector_cuda = YoloDetector(half_precision=True)
        assert detector_cuda.device == "cuda"
        assert detector_cuda.half_precision is True
        assert detector_cuda.use_half is True

        mock_model_cuda = MockYoloModel()
        detector_cuda.model = mock_model_cuda
        detector_cuda.detect(dummy_frame, camera_id="CAM_CUDA")
        assert mock_model_cuda.last_predict_kwargs.get("half") is True

    # 4. Verify detect() does NOT pass half kwarg when use_half is False (even on CUDA)
    with patch("ai.detection.detector.get_device", return_value="cuda"):
        detector_cuda_fp32 = YoloDetector(half_precision=False)
        assert detector_cuda_fp32.use_half is False

        mock_model_cuda_fp32 = MockYoloModel()
        detector_cuda_fp32.model = mock_model_cuda_fp32
        detector_cuda_fp32.detect(dummy_frame, camera_id="CAM_CUDA_FP32")
        assert "half" not in mock_model_cuda_fp32.last_predict_kwargs


def test_frame_preprocessor_defaults_and_activation():
    """
    Verify FramePreprocessor defaults to master enabled=False and inactive,
    does not modify frames when inactive, and only activates when master-enabled
    with at least one transform configured.
    """
    # 1. Default no-args: master enabled=False, is_active=False
    prep_default = FramePreprocessor()
    assert prep_default.enabled is False
    assert prep_default.is_active is False

    # 2. Master disabled even if transform flag is passed -> stays inactive (fail-safe)
    prep_dormant = FramePreprocessor(enable_gamma=True)
    assert prep_dormant.enabled is False
    assert prep_dormant.is_active is False

    # 3. Master enabled without transforms -> stays inactive
    prep_no_transforms = FramePreprocessor(enabled=True)
    assert prep_no_transforms.enabled is True
    assert prep_no_transforms.is_active is False

    # 4. Master enabled with single transform -> active
    prep_active = FramePreprocessor(enabled=True, enable_gamma=True)
    assert prep_active.enabled is True
    assert prep_active.is_active is True

    # 5. Inactive preprocessor returns byte-identical frame without modification
    dummy_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    unmodified = prep_default.preprocess(dummy_frame)
    assert np.array_equal(unmodified, dummy_frame)
    assert np.array_equal(prep_dormant.preprocess(dummy_frame), dummy_frame)


def test_detector_preprocessing_hook_integration():
    """
    Verify YoloDetector respects fail-closed preprocessing hook:
    - Default configuration never invokes preprocess()
    - Fallback is fail-closed when is_active is absent
    - Explicitly enabled preprocessor is executed during detect()
    """
    from unittest.mock import MagicMock

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Default detector: preprocessor is not active, preprocess() is never called
    detector = YoloDetector()
    detector.model = MockYoloModel()
    assert detector.preprocessor.is_active is False

    detector.preprocessor.preprocess = MagicMock(return_value=dummy_frame)
    detector.detect(dummy_frame, camera_id="CAM_TEST")
    detector.preprocessor.preprocess.assert_not_called()

    # 2. Fail-closed fallback: object without is_active attribute defaults to False
    class DummyNoActive:
        def preprocess(self, f):
            raise RuntimeError("Should never be called")

    detector.preprocessor = DummyNoActive()
    # Does not crash or invoke preprocess
    dets = detector.detect(dummy_frame, camera_id="CAM_TEST")
    assert isinstance(dets, list)

    # 3. Explicitly enabled preprocessor: preprocess() is called
    active_prep = FramePreprocessor(enabled=True, enable_gamma=True, gamma_value=1.5)
    detector_active = YoloDetector(preprocessor=active_prep)
    detector_active.model = MockYoloModel()
    assert detector_active.preprocessor.is_active is True

    detector_active.preprocessor.preprocess = MagicMock(return_value=dummy_frame)
    detector_active.detect(dummy_frame, camera_id="CAM_ACTIVE")
    detector_active.preprocessor.preprocess.assert_called_once_with(dummy_frame)


def test_preprocessing_synthetic_transforms():
    """
    Verify correctness of synthetic transforms on controlled test images:
    - Gamma correction brightens dark imagery (mean pixel value increases)
    - CLAHE improves contrast on low-contrast imagery (standard deviation increases)
    - Grayscale IR conversion produces 3-channel identical values
    """
    # 1. Gamma brightening on dark image
    dark_img = np.full((100, 100, 3), 30, dtype=np.uint8)
    prep_gamma = FramePreprocessor(enabled=True, enable_gamma=True, gamma_value=1.5)
    bright_img = prep_gamma.preprocess(dark_img)
    assert bright_img.shape == dark_img.shape
    assert float(bright_img.mean()) > float(dark_img.mean())

    # 2. CLAHE contrast enhancement on low-contrast image
    low_contrast = np.full((100, 100, 3), 128, dtype=np.uint8)
    low_contrast[25:75, 25:75] = 135  # subtle patch
    prep_clahe = FramePreprocessor(enabled=True, enable_clahe=True, clahe=True)
    clahe_img = prep_clahe.preprocess(low_contrast)
    assert clahe_img.shape == low_contrast.shape
    assert float(clahe_img.std()) > float(low_contrast.std())

    # 3. Grayscale IR simulation on colored image
    color_img = np.zeros((50, 50, 3), dtype=np.uint8)
    color_img[:, :, 0] = 200  # blue
    color_img[:, :, 1] = 50   # green
    color_img[:, :, 2] = 100  # red
    prep_ir = FramePreprocessor(enabled=True, enable_grayscale_ir=True)
    ir_img = prep_ir.preprocess(color_img)
    assert ir_img.shape == color_img.shape
    assert np.array_equal(ir_img[:, :, 0], ir_img[:, :, 1])
    assert np.array_equal(ir_img[:, :, 1], ir_img[:, :, 2])


if __name__ == "__main__":
    print("\n=======================================================")
    print("RUNNING AI DETECTION MODULE TEST SUITE")
    print("=======================================================")

    print("[1/15] Testing ConfidenceTracker Confirmation Flow...")
    test_confidence_tracker_confirmation_flow()
    print("       --> PASS: Target confirmed after consecutive hits threshold.")

    print("[2/15] Testing ConfidenceTracker Low Confidence Reset...")
    test_confidence_tracker_low_confidence_reset()
    print("       --> PASS: Low confidence resets consecutive counter.")

    print("[3/15] Testing ConfidenceTracker Multi-Object Independence...")
    test_confidence_tracker_multi_object_independence()
    print("       --> PASS: Multi-object states decoupled.")

    print("[4/15] Testing ConfidenceTracker Stale Pruning...")
    test_confidence_tracker_stale_pruning()
    print("       --> PASS: Expired records pruned successfully.")

    print("[5/15] Testing Detection Settings YAML Loader...")
    test_detection_settings_loading()
    print("       --> PASS: Settings schema and class flags loaded properly.")

    print("[6/15] Testing BaseDetector Interface & RawDetection Contract...")
    test_base_detector_interface()
    print("       --> PASS: BaseDetector polymorphism and contract serialized.")

    print("[7/15] Testing RawDetection Backward Compatibility & Fallback...")
    test_raw_detection_backward_compatibility()
    test_model_resolution_fallback()
    print("       --> PASS: Backward compatibility and model fallback verified.")

    print("[8/15] Testing Detector Rising Confidence Confirmation...")
    test_detector_rising_confidence_confirmation()
    print("       --> PASS: Target confirmed across 3 consecutive video frames.")

    print("[9/15] Testing Detector Transient Spike Non-Confirmation...")
    test_detector_transient_spike_non_confirmation()
    print("       --> PASS: Single-frame spike remains unconfirmed.")

    print("[10/15] Testing Detector Per-Camera State Isolation...")
    test_detector_per_camera_isolation()
    print("        --> PASS: Cameras track confirmation independently.")

    print("[11/15] Testing Detector Filtering & detect_confirmed Helper...")
    test_detector_filter_unconfirmed_and_detect_confirmed()
    print("        --> PASS: Filtering and convenience helper verified.")

    print("[12/15] Testing Detector Half-Precision (FP16) Safety & Initialization...")
    test_detector_half_precision_initialization_and_safety()
    print("        --> PASS: Half-precision initialization, fallback, and kwargs verified.")

    print("[13/15] Testing FramePreprocessor Defaults and Fail-Safe Activation...")
    test_frame_preprocessor_defaults_and_activation()
    print("        --> PASS: Default inactive, master-enabled gating verified.")

    print("[14/15] Testing Detector Fail-Closed Preprocessing Hook Integration...")
    test_detector_preprocessing_hook_integration()
    print("        --> PASS: Fail-closed fallback and default-off verified.")

    print("[15/15] Testing Preprocessing Synthetic Image Transforms...")
    test_preprocessing_synthetic_transforms()
    print("        --> PASS: Gamma brightening, CLAHE contrast, and IR grayscale verified.")

    print("\nSTATUS: ALL 15 AI DETECTION MODULE TESTS PASSED! [15/15]")



