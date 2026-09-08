"""
Automated Test Suite for ai/tracking/tracker.py.
Validates:
1. ObjectTracker consuming precomputed BaseDetector RawDetection outputs.
2. Zero duplicate YOLO inference and zero VRAM allocation for tracking.
3. Object ID assignment back onto RawDetection contract objects.
4. Multi-frame persistent track continuity across consecutive frames.
5. Noise speck rejection (< 20px).
6. Chair containment suppression inside human bounding boxes.
7. Tracker reset state clearance.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import List
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.tracking.tracker import ObjectTracker, DetectionsAdapter
from backend.app.models.contracts import RawDetection


def test_tracker_with_precomputed_raw_detections():
    """Verify ObjectTracker consumes RawDetection list without running YOLO."""
    tracker = ObjectTracker()
    assert tracker.detector is None, "Detector should not be initialized before use"

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        RawDetection(
            detection_id="det_01",
            class_name="person",
            confidence=0.91,
            bbox=[100.0, 100.0, 180.0, 260.0],
            camera_id="CAM_01",
        ),
        RawDetection(
            detection_id="det_02",
            class_name="car",
            confidence=0.87,
            bbox=[300.0, 200.0, 460.0, 320.0],
            camera_id="CAM_01",
        ),
    ]

    tracks, annotated = tracker.update(frame, detections=detections)

    # 1. Verify detector was NOT loaded into memory
    assert tracker.detector is None, "Tracker should NOT instantiate detector when precomputed detections are provided"

    # 2. Verify tracks output format
    assert len(tracks) == 2, f"Expected 2 tracks, got {len(tracks)}"
    track_classes = {t["class_name"] for t in tracks}
    assert "person" in track_classes and "car" in track_classes

    # 3. Verify track payload structure (including contract additions and aliases)
    for t in tracks:
        assert "track_id" in t and isinstance(t["track_id"], int)
        assert "class_id" in t and isinstance(t["class_id"], int)
        assert "class_name" in t and isinstance(t["class_name"], str)
        assert "confidence" in t and isinstance(t["confidence"], float)
        assert "bbox" in t and len(t["bbox"]) == 4
        assert "center" in t and isinstance(t["center"], tuple) and len(t["center"]) == 2
        assert "normalized_center" in t and len(t["normalized_center"]) == 2
        assert "dwell_time_s" in t and isinstance(t["dwell_time_s"], float)
        assert "dwell_seconds" in t and t["dwell_seconds"] == t["dwell_time_s"]
        assert "history" in t and len(t["history"]) == 1 and isinstance(t["history"][0], tuple)
        assert "trajectory" in t and t["trajectory"] == t["history"]
        assert "velocity_vector" in t and len(t["velocity_vector"]) == 2

    # 4. Verify object_id was assigned back to the RawDetection contract instances
    assert detections[0].object_id is not None
    assert detections[1].object_id is not None
    assert detections[0].object_id != detections[1].object_id

    # 5. Verify annotated frame has correct dimensions
    assert annotated.shape == frame.shape


def test_tracker_multi_frame_persistence():
    """Verify track IDs persist across consecutive frames for moving objects."""
    tracker = ObjectTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Person at [100, 100, 180, 260]
    det_f1 = [
        RawDetection(
            detection_id="det_101",
            class_name="person",
            confidence=0.90,
            bbox=[100.0, 100.0, 180.0, 260.0],
        )
    ]
    tracks_f1, _ = tracker.update(frame, detections=det_f1)
    assert len(tracks_f1) == 1
    track_id_f1 = tracks_f1[0]["track_id"]

    # Frame 2: Person moves to [104, 102, 184, 262]
    det_f2 = [
        RawDetection(
            detection_id="det_102",
            class_name="person",
            confidence=0.92,
            bbox=[104.0, 102.0, 184.0, 262.0],
        )
    ]
    tracks_f2, _ = tracker.update(frame, detections=det_f2)
    assert len(tracks_f2) == 1
    assert tracks_f2[0]["track_id"] == track_id_f1, "Track ID must persist across consecutive frames"
    assert len(tracks_f2[0]["trajectory"]) == 2, "Trajectory history should accumulate points"
    assert len(tracks_f2[0]["history"]) == 2, "History alias should accumulate points"

    # Frame 3: Person moves to [108, 104, 188, 264]
    det_f3 = [
        RawDetection(
            detection_id="det_103",
            class_name="person",
            confidence=0.94,
            bbox=[108.0, 104.0, 188.0, 264.0],
        )
    ]
    tracks_f3, _ = tracker.update(frame, detections=det_f3)
    assert len(tracks_f3) == 1
    assert tracks_f3[0]["track_id"] == track_id_f1, "Track ID must remain continuous on Frame 3"
    assert len(tracks_f3[0]["trajectory"]) == 3
    assert len(tracks_f3[0]["history"]) == 3


def test_tracker_noise_speck_rejection():
    """Verify tiny noise boxes (< 20px) are ignored."""
    tracker = ObjectTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = [
        # Normal person
        RawDetection(detection_id="d1", class_name="person", confidence=0.85, bbox=[100.0, 100.0, 160.0, 240.0]),
        # Noise speck 10x10 px
        RawDetection(detection_id="d2", class_name="person", confidence=0.90, bbox=[50.0, 50.0, 60.0, 60.0]),
        # Noise speck width < 20 px
        RawDetection(detection_id="d3", class_name="backpack", confidence=0.88, bbox=[200.0, 200.0, 210.0, 250.0]),
    ]

    tracks, _ = tracker.update(frame, detections=detections)
    assert len(tracks) == 1
    assert tracks[0]["class_name"] == "person"
    assert tracks[0]["bbox"] == [100.0, 100.0, 160.0, 240.0]


def test_tracker_chair_containment_suppression():
    """Verify chair overlapping with person's upper body is suppressed."""
    tracker = ObjectTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = [
        # Person
        RawDetection(detection_id="d_person", class_name="person", confidence=0.92, bbox=[100.0, 100.0, 250.0, 400.0]),
        # Chair contained inside the person's upper body [140, 150, 210, 280]
        RawDetection(detection_id="d_chair", class_name="chair", confidence=0.75, bbox=[140.0, 150.0, 210.0, 280.0]),
        # Legitimate separate chair elsewhere in the frame
        RawDetection(detection_id="d_chair_ext", class_name="chair", confidence=0.82, bbox=[450.0, 200.0, 550.0, 350.0]),
    ]

    tracks, _ = tracker.update(frame, detections=detections)
    # The contained chair should be suppressed; person and external chair should be tracked
    assert len(tracks) == 2
    track_classes = [t["class_name"] for t in tracks]
    assert track_classes.count("person") == 1
    assert track_classes.count("chair") == 1


def test_tracker_reset():
    """Verify reset() clears tracks and re-initializes ByteTracker."""
    tracker = ObjectTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    det = [RawDetection(detection_id="d1", class_name="car", confidence=0.90, bbox=[50.0, 50.0, 150.0, 150.0])]
    tracker.update(frame, detections=det)
    assert len(tracker.tracks) == 1

    tracker.reset()
    assert len(tracker.tracks) == 0


if __name__ == "__main__":
    print("RUNNING TRACKER MODULE UNIT TESTS")
    test_tracker_with_precomputed_raw_detections()
    print("  -> test_tracker_with_precomputed_raw_detections: PASS")
    test_tracker_multi_frame_persistence()
    print("  -> test_tracker_multi_frame_persistence: PASS")
    test_tracker_noise_speck_rejection()
    print("  -> test_tracker_noise_speck_rejection: PASS")
    test_tracker_chair_containment_suppression()
    print("  -> test_tracker_chair_containment_suppression: PASS")
    test_tracker_reset()
    print("  -> test_tracker_reset: PASS")
    print("ALL TESTS PASSED!")
