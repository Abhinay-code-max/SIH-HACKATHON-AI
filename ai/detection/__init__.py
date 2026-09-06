"""
AI & Tactical Object Detection Module.
Provides BaseDetector interface, YoloDetector implementation,
and ConfidenceTracker for multi-frame false positive reduction.
"""

from ai.detection.confidence_tracker import ConfidenceTracker
from ai.detection.detector import (
    BaseDetector,
    YoloDetector,
    load_detection_config,
    resolve_registered_model,
)

__all__ = [
    "BaseDetector",
    "YoloDetector",
    "ConfidenceTracker",
    "load_detection_config",
    "resolve_registered_model",
]
