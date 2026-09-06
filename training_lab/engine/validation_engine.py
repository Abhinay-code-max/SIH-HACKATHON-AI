"""
Human Validation Engine & 9-Type Error Taxonomy.
Records feedback on AI predictions (CORRECT, INCORRECT, CHANGE_CLASS, IGNORE, MANUAL_ADDITION),
tracks fine-grained error categories across all 9 defined failure modes, computes error statistics,
and exports ground-truth training candidates for iterative model fine-tuning.
"""

from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ANNOTATIONS_DIR = ROOT_DIR / "training_lab" / "annotations"
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = ANNOTATIONS_DIR / "validation_log.json"


class ErrorCategory(str, Enum):
    """Fine-grained 9-type surveillance perception error taxonomy."""
    TRUE_POSITIVE = "TRUE_POSITIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    WRONG_CLASS = "WRONG_CLASS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    DUPLICATE_DETECTION = "DUPLICATE_DETECTION"
    TRACKING_ERROR = "TRACKING_ERROR"
    PERIMETER_ERROR = "PERIMETER_ERROR"
    THREAT_CLASSIFICATION_ERROR = "THREAT_CLASSIFICATION_ERROR"


class ValidationDecision(str, Enum):
    """Human validator review decisions."""
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    CHANGE_CLASS = "CHANGE_CLASS"
    IGNORE = "IGNORE"
    MANUAL_ADDITION = "MANUAL_ADDITION"


class ValidatedRecord(BaseModel):
    """Schema for an individual validated feedback record."""
    record_id: str
    scenario_id: str
    camera_id: str
    frame_idx: int
    timestamp: float
    tracking_id: Optional[Union[int, str]] = None
    ai_prediction: Optional[Dict[str, Any]] = None
    decision: ValidationDecision
    human_verified_class: Optional[str] = None
    human_bbox: Optional[List[float]] = None
    error_category: ErrorCategory
    is_training_candidate: bool = False
    validator_notes: Optional[str] = None
    crop_path: Optional[str] = None
    validated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ValidationEngine:
    """
    Engine for recording, querying, and analyzing human feedback on AI detections
    across the 9-type error taxonomy.
    """

    def __init__(self, log_file: Optional[Union[str, Path]] = None):
        self.log_file = Path(log_file) if log_file else DEFAULT_LOG_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: Dict[str, ValidatedRecord] = {}
        self._next_idx = 1
        self.reload_from_disk()

    def reload_from_disk(self) -> None:
        """Loads all validation records from disk into memory."""
        with self._lock:
            self._records.clear()
            if self.log_file.is_file():
                try:
                    with open(self.log_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for item in data:
                        rec = ValidatedRecord(**item)
                        self._records[rec.record_id] = rec
                    self._next_idx = len(self._records) + 1
                except Exception:
                    pass

    def _save_to_disk(self) -> None:
        """Atomically persists validation records to disk."""
        tmp_file = self.log_file.with_suffix(".tmp")
        data = [r.model_dump() for r in self._records.values()]
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_file.replace(self.log_file)

    def record_feedback(
        self,
        scenario_id: str,
        camera_id: str,
        frame_idx: int,
        decision: Union[ValidationDecision, str],
        ai_prediction: Optional[Dict[str, Any]] = None,
        human_verified_class: Optional[str] = None,
        human_bbox: Optional[List[float]] = None,
        tracking_id: Optional[Union[int, str]] = None,
        error_category: Optional[Union[ErrorCategory, str]] = None,
        notes: Optional[str] = None,
        crop_path: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> ValidatedRecord:
        """
        Records validator review feedback and assigns error taxonomy category.

        Decision Rules:
          - CORRECT: error_category = TRUE_POSITIVE, verified_class = ai_class, is_training_candidate = True.
          - INCORRECT: error_category = FALSE_POSITIVE, is_training_candidate = False.
          - CHANGE_CLASS: error_category = WRONG_CLASS, verified_class = corrected_class, is_training_candidate = True.
          - IGNORE: error_category = LOW_CONFIDENCE, is_training_candidate = False.
          - MANUAL_ADDITION: error_category = FALSE_NEGATIVE, is_training_candidate = True.
        """
        # Normalize decision
        if isinstance(decision, str):
            dec_enum = ValidationDecision(decision.upper().strip())
        else:
            dec_enum = decision

        # Determine ErrorCategory and Training Qualification based on decision rules
        verified_class = human_verified_class
        assigned_bbox = human_bbox
        is_candidate = False

        if dec_enum == ValidationDecision.CORRECT:
            cat = ErrorCategory.TRUE_POSITIVE
            if ai_prediction:
                verified_class = verified_class or ai_prediction.get("class_name")
                assigned_bbox = assigned_bbox or ai_prediction.get("bbox")
            is_candidate = True

        elif dec_enum == ValidationDecision.INCORRECT:
            cat = ErrorCategory.FALSE_POSITIVE
            is_candidate = False

        elif dec_enum == ValidationDecision.CHANGE_CLASS:
            cat = ErrorCategory.WRONG_CLASS
            if ai_prediction and not assigned_bbox:
                assigned_bbox = ai_prediction.get("bbox")
            is_candidate = True

        elif dec_enum == ValidationDecision.IGNORE:
            cat = ErrorCategory.LOW_CONFIDENCE
            is_candidate = False

        elif dec_enum == ValidationDecision.MANUAL_ADDITION:
            cat = ErrorCategory.FALSE_NEGATIVE
            is_candidate = True

        else:
            cat = ErrorCategory.TRUE_POSITIVE
            is_candidate = True

        # Allow explicit override if caller provided an explicit category
        if error_category:
            if isinstance(error_category, str):
                cat = ErrorCategory(error_category.upper().strip())
            else:
                cat = error_category

        ts = round(float(timestamp), 4) if timestamp is not None else round(frame_idx / 30.0, 4)

        with self._lock:
            rec_id = f"VAL_{self._next_idx:05d}"
            self._next_idx += 1

            record = ValidatedRecord(
                record_id=rec_id,
                scenario_id=scenario_id,
                camera_id=camera_id.upper().strip(),
                frame_idx=frame_idx,
                timestamp=ts,
                tracking_id=tracking_id,
                ai_prediction=ai_prediction,
                decision=dec_enum,
                human_verified_class=verified_class,
                human_bbox=assigned_bbox,
                error_category=cat,
                is_training_candidate=is_candidate,
                validator_notes=notes,
                crop_path=crop_path,
                validated_at=datetime.now(timezone.utc).isoformat(),
            )
            self._records[record.record_id] = record
            self._save_to_disk()

        return record

    def get_validations(
        self,
        scenario_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        error_category: Optional[Union[ErrorCategory, str]] = None,
    ) -> List[ValidatedRecord]:
        """Queries validations with optional filters."""
        with self._lock:
            records = list(self._records.values())

        if scenario_id:
            records = [r for r in records if r.scenario_id == scenario_id]
        if camera_id:
            cid_upper = camera_id.upper().strip()
            records = [r for r in records if r.camera_id == cid_upper]
        if error_category:
            cat_enum = ErrorCategory(error_category) if isinstance(error_category, str) else error_category
            records = [r for r in records if r.error_category == cat_enum]

        return sorted(records, key=lambda r: r.record_id)

    def compute_error_statistics(
        self,
        scenario_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Aggregates fine-grained error statistics across all 9 defined failure modes.
        """
        records = self.get_validations(scenario_id=scenario_id)
        total = len(records)

        # Initialize full 9-type taxonomy counts to 0
        cat_counts = {cat.value: 0 for cat in ErrorCategory}
        decision_counts = {dec.value: 0 for dec in ValidationDecision}

        for r in records:
            cat_counts[r.error_category.value] = cat_counts.get(r.error_category.value, 0) + 1
            decision_counts[r.decision.value] = decision_counts.get(r.decision.value, 0) + 1

        tp = cat_counts[ErrorCategory.TRUE_POSITIVE.value]
        fp = cat_counts[ErrorCategory.FALSE_POSITIVE.value]
        fn = cat_counts[ErrorCategory.FALSE_NEGATIVE.value]
        wrong_cls = cat_counts[ErrorCategory.WRONG_CLASS.value]

        accuracy_pct = round((tp / total) * 100, 2) if total > 0 else 0.0
        fp_rate = round((fp / total) * 100, 2) if total > 0 else 0.0
        fn_rate = round((fn / total) * 100, 2) if total > 0 else 0.0

        training_candidates = [r for r in records if r.is_training_candidate]

        return {
            "total_validations": total,
            "counts_by_category": cat_counts,
            "counts_by_decision": decision_counts,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "wrong_class_errors": wrong_cls,
            "accuracy_pct": accuracy_pct,
            "false_positive_rate": fp_rate,
            "false_negative_rate": fn_rate,
            "training_candidates_count": len(training_candidates),
        }

    def get_validation_statistics(
        self,
        scenario_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Alias for compute_error_statistics."""
        return self.compute_error_statistics(scenario_id=scenario_id)

    def export_verified_training_candidates(
        self,
        scenario_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Exports all validated positive samples (TP, Class Corrections, Manual Additions)
        formatted as complete ground-truth training payloads.
        """
        records = self.get_validations(scenario_id=scenario_id)
        candidates = [r for r in records if r.is_training_candidate]

        exported = []
        for r in candidates:
            # Resolve bounding box (human_bbox preferred, then ai_prediction bbox)
            bbox = r.human_bbox
            if not bbox and r.ai_prediction:
                bbox = r.ai_prediction.get("bbox")

            # Resolve target class (human_verified_class preferred, then ai_prediction class)
            class_name = r.human_verified_class
            if not class_name and r.ai_prediction:
                class_name = r.ai_prediction.get("class_name")

            if not bbox or not class_name:
                continue

            exported.append({
                "record_id": r.record_id,
                "scenario_id": r.scenario_id,
                "camera_id": r.camera_id,
                "frame_idx": r.frame_idx,
                "timestamp": r.timestamp,
                "tracking_id": r.tracking_id,
                "class_name": class_name,
                "bbox": [round(float(c), 1) for c in bbox],
                "decision": r.decision.value,
                "error_category": r.error_category.value,
                "crop_path": r.crop_path,
                "validator_notes": r.validator_notes,
                "validated_at": r.validated_at,
            })

        return exported

    def clear(self) -> None:
        """Clears records in memory and empties disk log (used for test isolation)."""
        with self._lock:
            self._records.clear()
            self._next_idx = 1
            if self.log_file.is_file():
                self.log_file.unlink()


# Singleton validation engine instance
validation_engine = ValidationEngine()


if __name__ == "__main__":
    print("[ValidationEngine] Initializing Human Validation Engine test...")
    engine = ValidationEngine()
    rec1 = engine.record_feedback(
        scenario_id="SCN_01",
        camera_id="CAM_01",
        frame_idx=10,
        decision=ValidationDecision.CORRECT,
        ai_prediction={"class_name": "person", "confidence": 0.94, "bbox": [60.0, 188.0, 91.0, 293.0]},
    )
    print(f"  Recorded feedback: {rec1.record_id} -> {rec1.error_category} (Training candidate: {rec1.is_training_candidate})")
    stats = engine.compute_error_statistics("SCN_01")
    print(f"  Error Stats: {stats['counts_by_category']}")
