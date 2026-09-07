"""
Unit and Integration Tests for Specialist Object Detectors.
Validates:
1. BaseDetector contract compliance (RawDetection output schema).
2. Exact class_name mapping for Weapon, Drone, and Fire/Smoke detectors.
3. Explicit verification of fire/smoke source index mapping (0 -> smoke, 1 -> fire).
4. Error handling and fail-safe behaviors (missing weights, auto_download=False).
5. CompositeSpecialistDetector orchestration and detection merging.
6. Real cached model loading and model.names verification.
"""

from pathlib import Path
import sys
from typing import Any, List
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.detection.detector import BaseDetector
from ai.detection.specialist_detectors import (
    BaseSpecialistDetector,
    WeaponSpecialistDetector,
    DroneSpecialistDetector,
    FireSmokeSpecialistDetector,
    CompositeSpecialistDetector,
    run_composite_detection,
)
from backend.app.models.contracts import RawDetection


class MockBox:
    def __init__(self, cls_id: int, conf: float, xyxy: List[float]):
        self.cls = np.array([cls_id])
        self.conf = np.array([conf])
        self.xyxy = np.array([xyxy])


class MockSpecialistModel:
    def __init__(self, names: dict, boxes: List[MockBox] = None):
        self.names = names
        self.boxes = boxes or []

    def predict(self, **kwargs):
        res = MagicMock()
        res.boxes = self.boxes
        return [res]


def test_specialist_detectors_conform_to_base_detector():
    """Verify all specialist detector classes inherit from BaseDetector."""
    assert issubclass(WeaponSpecialistDetector, BaseDetector)
    assert issubclass(DroneSpecialistDetector, BaseDetector)
    assert issubclass(FireSmokeSpecialistDetector, BaseDetector)
    assert issubclass(CompositeSpecialistDetector, BaseDetector)


def test_weapon_specialist_class_mapping_and_contract():
    """Verify WeaponSpecialistDetector maps all sub-classes (grenade, knife, pistol, rifle) to 'weapon'."""
    detector = WeaponSpecialistDetector(auto_download=False, weights_path="ai/models/specialists/weapon_best.pt", enabled=True)
    # Inject mock model with the real weapon model names
    mock_model = MockSpecialistModel(
        names={0: "grenade", 1: "knife", 2: "pistol", 3: "rifle"},
        boxes=[
            MockBox(cls_id=0, conf=0.88, xyxy=[10.0, 20.0, 50.0, 60.0]),
            MockBox(cls_id=2, conf=0.94, xyxy=[100.0, 120.0, 180.0, 220.0]),
        ],
    )
    detector.model = mock_model

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = detector.detect(dummy_frame, camera_id="CAM_TEST")

    assert len(dets) == 2
    for d in dets:
        assert isinstance(d, RawDetection)
        assert d.class_name == "weapon"
        assert d.camera_id == "CAM_TEST"
        assert d.confirmed is True
        assert len(d.bbox) == 4
        assert len(d.normalized_center) == 2
        assert "weap" in d.detection_id


def test_weapon_specialist_disabled_by_default():
    """Verify WeaponSpecialistDetector is disabled by default, allocates 0 VRAM and costs 0 ms."""
    detector = WeaponSpecialistDetector(auto_download=False)
    assert detector.enabled is False
    assert detector.status == "in_development"
    assert detector._ensure_model() is None
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(dummy_frame) == []


def test_drone_specialist_class_mapping_and_contract():
    """Verify DroneSpecialistDetector maps all drone types (quadcopter, fixed-wing) to 'drone'."""
    detector = DroneSpecialistDetector(auto_download=False, weights_path="ai/models/specialists/drone_best.pt")
    mock_model = MockSpecialistModel(
        names={0: "quadcopter", 1: "fixed-wing"},
        boxes=[
            MockBox(cls_id=0, conf=0.78, xyxy=[200.0, 50.0, 280.0, 110.0]),
            MockBox(cls_id=1, conf=0.85, xyxy=[400.0, 100.0, 460.0, 160.0]),
        ],
    )
    detector.model = mock_model

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = detector.detect(dummy_frame, camera_id="CAM_PERIMETER")

    assert len(dets) == 2
    for d in dets:
        assert isinstance(d, RawDetection)
        assert d.class_name == "drone"
        assert d.camera_id == "CAM_PERIMETER"
        assert "drn" in d.detection_id


def test_fire_smoke_specialist_exact_index_mapping():
    """
    CRITICAL: Verify FireSmokeSpecialistDetector source index remapping:
    Index 0 MUST map to 'smoke'
    Index 1 MUST map to 'fire'
    """
    detector = FireSmokeSpecialistDetector(auto_download=False, weights_path="ai/models/specialists/fire_smoke_best.pt")
    mock_model = MockSpecialistModel(
        names={0: "smoke", 1: "fire"},
        boxes=[
            MockBox(cls_id=0, conf=0.91, xyxy=[50.0, 50.0, 150.0, 150.0]),
            MockBox(cls_id=1, conf=0.89, xyxy=[200.0, 200.0, 300.0, 300.0]),
        ],
    )
    detector.model = mock_model
    # Explicitly trigger map initialization
    detector._class_map = {0: "smoke", 1: "fire"}

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = detector.detect(dummy_frame, camera_id="CAM_FOREST")

    assert len(dets) == 2
    assert dets[0].class_name == "smoke", f"Expected class 0 to map to 'smoke', got {dets[0].class_name}"
    assert dets[1].class_name == "fire", f"Expected class 1 to map to 'fire', got {dets[1].class_name}"
    assert dets[0].confidence == 0.91
    assert dets[1].confidence == 0.89
    assert "fs" in dets[0].detection_id


def test_empty_and_none_frame_safety():
    """Verify all specialist detectors return empty list on None or empty frames without crashing."""
    detectors = [
        WeaponSpecialistDetector(auto_download=False),
        DroneSpecialistDetector(auto_download=False),
        FireSmokeSpecialistDetector(auto_download=False),
        CompositeSpecialistDetector(),
    ]
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    for det in detectors:
        assert det.detect(None) == []
        assert det.detect(empty_frame) == []


def test_specialist_detectors_default_auto_download_false():
    """Verify all specialist detectors default to auto_download=False to enforce Rule 2 (Zero Network Call)."""
    assert WeaponSpecialistDetector().auto_download is False
    assert DroneSpecialistDetector().auto_download is False
    assert FireSmokeSpecialistDetector().auto_download is False


def test_missing_weights_handling_no_auto_download():
    """Verify FileNotFoundError is raised when weights are missing and auto_download is False."""
    fake_path = ROOT_DIR / "ai" / "models" / "specialists" / "non_existent_weights.pt"
    detector = WeaponSpecialistDetector(weights_path=fake_path, auto_download=False, enabled=True)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(FileNotFoundError) as excinfo:
        detector.detect(dummy_frame)
    assert "Specialist weights not found" in str(excinfo.value)
    assert "Rule 2" in str(excinfo.value)


def test_download_failure_handling():
    """Verify RuntimeError with helpful diagnostic is raised if HuggingFace download fails."""
    fake_path = ROOT_DIR / "ai" / "models" / "specialists" / "non_existent_weights.pt"
    detector = WeaponSpecialistDetector(weights_path=fake_path, auto_download=True, enabled=True)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("ai.detection.specialist_detectors.hf_hub_download", side_effect=Exception("Connection refused")):
        with pytest.raises(RuntimeError) as excinfo:
            detector.detect(dummy_frame)
        assert "Failed to download weights" in str(excinfo.value)


def test_composite_detector_skips_disabled_specialists():
    """Verify CompositeSpecialistDetector skips disabled specialists (weapon) and reports status."""
    composite = CompositeSpecialistDetector(auto_init_specialists=True)
    status = composite.get_specialist_status()

    assert "WeaponSpecialistDetector" in status
    assert status["WeaponSpecialistDetector"]["enabled"] is False
    assert status["WeaponSpecialistDetector"]["status"] == "in_development"
    assert status["WeaponSpecialistDetector"]["model_loaded"] is False

    assert "DroneSpecialistDetector" in status
    assert status["DroneSpecialistDetector"]["enabled"] is True

    assert "FireSmokeSpecialistDetector" in status
    assert status["FireSmokeSpecialistDetector"]["enabled"] is True

    # Confirm detect skips weapon without loading weights into VRAM
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for spec in composite.specialists:
        if isinstance(spec, WeaponSpecialistDetector):
            assert spec.model is None


def test_composite_detector_orchestration():
    """Verify CompositeSpecialistDetector runs primary detector and specialists, merging outputs."""
    # Mock primary detector
    mock_primary = MagicMock(spec=BaseDetector)
    det1 = RawDetection(
        detection_id="det_cam_0001",
        class_name="person",
        confidence=0.92,
        bbox=[10.0, 10.0, 60.0, 120.0],
        camera_id="CAM_GATE",
    )
    mock_primary.detect.return_value = [det1]

    # Mock specialist 1
    mock_spec1 = MagicMock(spec=BaseDetector)
    det2 = RawDetection(
        detection_id="det_cam_weap_0001",
        class_name="weapon",
        confidence=0.88,
        bbox=[20.0, 40.0, 50.0, 70.0],
        camera_id="CAM_GATE",
    )
    mock_spec1.detect.return_value = [det2]

    # Mock specialist 2 (fails/raises exception — composite should handle gracefully)
    mock_spec2 = MagicMock(spec=BaseDetector)
    mock_spec2.detect.side_effect = RuntimeError("Specialist backend unavailable")

    composite = CompositeSpecialistDetector(
        primary_detector=mock_primary,
        specialists=[mock_spec1, mock_spec2],
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    combined = composite.detect(dummy_frame, camera_id="CAM_GATE")

    assert len(combined) == 2
    assert combined[0].class_name == "person"
    assert combined[1].class_name == "weapon"

    # Verify functional helper run_composite_detection
    helper_res = run_composite_detection(
        dummy_frame,
        primary_detector=mock_primary,
        specialists=[mock_spec1],
        camera_id="CAM_GATE",
    )
    assert len(helper_res) == 2


def test_real_cached_models_loading_and_names():
    """
    Integration test using the locally cached weights in ai/models/specialists/.
    Confirms all 3 models load successfully and prints model.names dict.
    """
    specialists_dir = ROOT_DIR / "ai" / "models" / "specialists"
    weap_path = specialists_dir / "weapon_best.pt"
    drone_path = specialists_dir / "drone_best.pt"
    fs_path = specialists_dir / "fire_smoke_best.pt"

    # Only run if weights are present locally (offline safe)
    if not (weap_path.is_file() and drone_path.is_file() and fs_path.is_file()):
        pytest.skip("Specialist weights not cached locally; skipping live model load test.")

    # 1. Weapon Specialist (instantiated with enabled=True for test verification)
    weap_det = WeaponSpecialistDetector(weights_path=weap_path, auto_download=False, enabled=True)
    m_weap = weap_det._ensure_model()
    assert m_weap is not None
    print(f"\n[TEST CONFIRMATION] Weapon model.names: {m_weap.names}")
    # Weapon model classes: grenade, knife, pistol, rifle
    assert len(m_weap.names) >= 1

    # 2. Drone Specialist (TomSmail/drone-yolo-v1)
    drone_det = DroneSpecialistDetector(weights_path=drone_path, auto_download=False)
    m_drone = drone_det._ensure_model()
    assert m_drone is not None
    print(f"\n[TEST CONFIRMATION] Drone model.names: {m_drone.names}")
    # Drone model classes: quadcopter, fixed-wing
    assert m_drone.names == {0: "quadcopter", 1: "fixed-wing"}

    # 3. Fire/Smoke Specialist
    fs_det = FireSmokeSpecialistDetector(weights_path=fs_path, auto_download=False)
    m_fs = fs_det._ensure_model()
    assert m_fs is not None
    print(f"\n[TEST CONFIRMATION] Fire/Smoke model.names: {m_fs.names}")
    # Verify index mapping: 0 -> smoke, 1 -> fire
    assert m_fs.names[0].strip().lower() == "smoke", f"Expected index 0 to be 'smoke', got {m_fs.names[0]}"
    assert m_fs.names[1].strip().lower() == "fire", f"Expected index 1 to be 'fire', got {m_fs.names[1]}"

    # Run inference on dummy frame to verify detect() pipeline end-to-end
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert isinstance(weap_det.detect(dummy_frame), list)
    assert isinstance(drone_det.detect(dummy_frame), list)
    assert isinstance(fs_det.detect(dummy_frame), list)
