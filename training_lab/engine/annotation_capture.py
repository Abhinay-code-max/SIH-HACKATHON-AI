"""
Annotation Capture Engine for Missed Detections (False Negatives).
Captures manual bounding boxes drawn by human validators when the AI fails to detect
an object (ADD DETECTION), validates coordinates and minimum 10x10 px size, extracts
and saves image crops/snapshots to disk, and records them as verified false negatives
for future retraining.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.validation_engine import (
    ErrorCategory,
    ValidatedRecord,
    ValidationDecision,
    ValidationEngine,
    validation_engine as default_validation_engine,
)

CROPS_DIR = ROOT_DIR / "training_lab" / "annotations" / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)


class AnnotationCapture:
    """
    Captures, validates, and archives manual bounding box additions from human validators.
    """

    def __init__(
        self,
        validation_engine_inst: Optional[ValidationEngine] = None,
        crops_dir: Optional[Union[str, Path]] = None,
    ):
        self.validation_engine = validation_engine_inst or default_validation_engine
        self.crops_dir = Path(crops_dir) if crops_dir else CROPS_DIR
        self.crops_dir.mkdir(parents=True, exist_ok=True)

    def save_frame_snapshot(
        self,
        frame: np.ndarray,
        scenario_id: str,
        camera_id: str,
        frame_idx: int,
        bbox: Optional[List[float]] = None,
    ) -> str:
        """
        Saves full frame snapshot or target crop to training_lab/annotations/crops/.

        Args:
            frame: input BGR frame
            scenario_id: scenario identifier (e.g. SCN_01)
            camera_id: camera identifier (e.g. CAM_01)
            frame_idx: video frame index
            bbox: optional [x1, y1, x2, y2] to crop specific target

        Returns:
            Relative path string to saved image file.
        """
        cid = camera_id.upper().strip()
        ts_suffix = int(time.time() * 1000) % 1_000_000

        if bbox:
            x1, y1, x2, y2 = [int(round(c)) for c in bbox]
            h, w = frame.shape[:2]
            cx1 = max(0, min(x1, w - 1))
            cy1 = max(0, min(y1, h - 1))
            cx2 = max(cx1 + 1, min(x2, w))
            cy2 = max(cy1 + 1, min(y2, h))

            crop_img = frame[cy1:cy2, cx1:cx2]
            if crop_img.size > 0:
                dest_file = self.crops_dir / f"{scenario_id}_{cid}_f{frame_idx:04d}_crop_{ts_suffix}.jpg"
                cv2.imwrite(str(dest_file), crop_img)
                return str(dest_file.relative_to(ROOT_DIR)).replace("\\", "/")

        # Fallback to saving whole frame
        dest_file = self.crops_dir / f"{scenario_id}_{cid}_f{frame_idx:04d}_snap_{ts_suffix}.jpg"
        cv2.imwrite(str(dest_file), frame)
        return str(dest_file.relative_to(ROOT_DIR)).replace("\\", "/")

    def capture_missed_detection(
        self,
        scenario_id: str,
        camera_id: str,
        frame_idx: int,
        bbox: List[float],
        class_name: str,
        notes: Optional[str] = None,
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> ValidatedRecord:
        """
        Captures manual bounding box drawn by a human validator when the AI missed an object.

        Validates:
          - Bounding box has 4 coordinates.
          - Minimum 10x10 px dimensions (x2 > x1 + 10 and y2 > y1 + 10).
          - Coordinates are clamped within image boundary [0, w] and [0, h].

        Tags record with:
          - decision: MANUAL_ADDITION
          - error_category: FALSE_NEGATIVE
          - is_training_candidate: True

        Returns:
            ValidatedRecord instance.
        """
        if len(bbox) != 4:
            raise ValueError(f"Bounding box must contain 4 coordinates [x1, y1, x2, y2], got {bbox}")

        raw_x1, raw_y1, raw_x2, raw_y2 = map(float, bbox)

        # Image boundary limits (default 640x480 if frame is not provided)
        if frame is not None and isinstance(frame, np.ndarray):
            h, w = float(frame.shape[0]), float(frame.shape[1])
        else:
            w, h = 640.0, 480.0

        # Boundary clamping
        cx1 = max(0.0, min(raw_x1, w))
        cy1 = max(0.0, min(raw_y1, h))
        cx2 = max(0.0, min(raw_x2, w))
        cy2 = max(0.0, min(raw_y2, h))

        # Order check
        if cx2 < cx1:
            cx1, cx2 = cx2, cx1
        if cy2 < cy1:
            cy1, cy2 = cy2, cy1

        # Size validation: minimum 10x10 px
        width_px = cx2 - cx1
        height_px = cy2 - cy1
        if width_px < 10.0 or height_px < 10.0:
            raise ValueError(
                f"Bounding box must satisfy minimum 10x10 px size requirement. Got width={width_px:.1f}px, height={height_px:.1f}px."
            )

        clamped_bbox = [round(cx1, 1), round(cy1, 1), round(cx2, 1), round(cy2, 1)]

        # Save crop or snapshot if frame is provided
        crop_rel_path = None
        if frame is not None and isinstance(frame, np.ndarray):
            crop_rel_path = self.save_frame_snapshot(
                frame=frame,
                scenario_id=scenario_id,
                camera_id=camera_id,
                frame_idx=frame_idx,
                bbox=clamped_bbox,
            )

        # Record feedback in ValidationEngine
        record = self.validation_engine.record_feedback(
            scenario_id=scenario_id,
            camera_id=camera_id,
            frame_idx=frame_idx,
            decision=ValidationDecision.MANUAL_ADDITION,
            human_verified_class=class_name.lower().strip(),
            human_bbox=clamped_bbox,
            error_category=ErrorCategory.FALSE_NEGATIVE,
            notes=notes or "Human validator manual addition (missed detection)",
            crop_path=crop_rel_path,
            timestamp=timestamp,
        )

        return record


# Singleton instance
annotation_capture = AnnotationCapture()


if __name__ == "__main__":
    print("[AnnotationCapture] Testing missed detection capture...")
    ac = AnnotationCapture()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_frame[150:260, 100:180] = (80, 120, 160)

    rec = ac.capture_missed_detection(
        scenario_id="SCN_01",
        camera_id="CAM_01",
        frame_idx=25,
        bbox=[100.0, 150.0, 180.0, 260.0],
        class_name="person",
        notes="Missed pedestrian near virtual line",
        frame=dummy_frame,
    )
    print(f"  Captured: {rec.record_id} -> {rec.error_category} ({rec.human_verified_class}) crop={rec.crop_path}")
