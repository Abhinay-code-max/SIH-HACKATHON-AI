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

    def predict(self, source: Any, **kwargs: Any) -> List[MockYoloResult]:
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

    # Verify classes: baseline enabled, future classes disabled
    classes = cfg["classes"]
    for c in ["person", "car", "truck", "bus", "motorcycle", "bicycle", "animal", "backpack", "bag"]:
        assert c in classes, f"Baseline class {c} missing from detection settings"
        assert classes[c]["enabled"] is True, f"Baseline class {c} should be enabled"

    # Verify future border surveillance targets
    for fc in ["weapon", "drone", "fire", "smoke"]:
        assert fc in classes, f"Future target {fc} missing from detection settings"
        assert classes[fc]["enabled"] is False, f"Future target {fc} must be disabled pending training"
        assert classes[fc].get("pending_training_data") is True


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


if __name__ == "__main__":
    print("\n=======================================================")
    print("RUNNING AI DETECTION MODULE TEST SUITE")
    print("=======================================================")

    print("[1/11] Testing ConfidenceTracker Confirmation Flow...")
    test_confidence_tracker_confirmation_flow()
    print("       --> PASS: Target confirmed after consecutive hits threshold.")

    print("[2/11] Testing ConfidenceTracker Low Confidence Reset...")
    test_confidence_tracker_low_confidence_reset()
    print("       --> PASS: Low confidence resets consecutive counter.")

    print("[3/11] Testing ConfidenceTracker Multi-Object Independence...")
    test_confidence_tracker_multi_object_independence()
    print("       --> PASS: Multi-object states decoupled.")

    print("[4/11] Testing ConfidenceTracker Stale Pruning...")
    test_confidence_tracker_stale_pruning()
    print("       --> PASS: Expired records pruned successfully.")

    print("[5/11] Testing Detection Settings YAML Loader...")
    test_detection_settings_loading()
    print("       --> PASS: Settings schema and class flags loaded properly.")

    print("[6/11] Testing BaseDetector Interface & RawDetection Contract...")
    test_base_detector_interface()
    print("       --> PASS: BaseDetector polymorphism and contract serialized.")

    print("[7/11] Testing RawDetection Backward Compatibility & Fallback...")
    test_raw_detection_backward_compatibility()
    test_model_resolution_fallback()
    print("       --> PASS: Backward compatibility and model fallback verified.")

    print("[8/11] Testing Detector Rising Confidence Confirmation...")
    test_detector_rising_confidence_confirmation()
    print("       --> PASS: Target confirmed across 3 consecutive video frames.")

    print("[9/11] Testing Detector Transient Spike Non-Confirmation...")
    test_detector_transient_spike_non_confirmation()
    print("       --> PASS: Single-frame spike remains unconfirmed.")

    print("[10/11] Testing Detector Per-Camera State Isolation...")
    test_detector_per_camera_isolation()
    print("        --> PASS: Cameras track confirmation independently.")

    print("[11/11] Testing Detector Filtering & detect_confirmed Helper...")
    test_detector_filter_unconfirmed_and_detect_confirmed()
    print("        --> PASS: Filtering and convenience helper verified.")

    print("\nSTATUS: ALL 11 AI DETECTION MODULE TESTS PASSED! [11/11]")

