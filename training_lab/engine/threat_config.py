"""
Centralized Configuration for Threat Intelligence & DEFCON Assessment.
All weights, thresholds, and DEFCON calibrations are stored here in one place.
"""

from typing import Dict, Tuple

THREAT_SCORING_VERSION = "v1.0"

# DEFCON Calibration Thresholds (inclusive lower bound, inclusive upper bound)
DEFCON_THRESHOLDS: Dict[int, Tuple[int, int]] = {
    5: (0, 19),    # Normal surveillance
    4: (20, 39),   # Elevated awareness
    3: (40, 59),   # Suspicious activity requiring operator attention
    2: (60, 79),   # High-risk event requiring immediate operator attention
    1: (80, 100),  # Critical event requiring immediate escalation
}

# Maximum factor score contributions
THREAT_FACTOR_WEIGHTS: Dict[str, int] = {
    "restricted_zone_intrusion": 30,
    "boundary_crossing": 30,
    "movement_toward_boundary": 15,
    "loitering": 15,
    "repeated_boundary_approach": 10,
    "abnormal_movement": 10,
    "prolonged_restricted_presence": 10,
    "group_proximity_context": 5,
}

# Object class context modifiers (neutral baseline by default)
CLASS_CONTEXT_MODIFIERS: Dict[str, int] = {
    "person": 0,
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "animal": -10,      # De-escalation modifier for wildlife false-alarms
    "backpack": 0,    # Neutral baseline: only contextual events add risk
    "bag": 0,         # Neutral baseline: only contextual events add risk
}

# Project heuristic thresholds
LOITERING_THRESHOLD_SECONDS = 30.0
PROLONGED_PRESENCE_SECONDS = 60.0
STATIONARY_SPEED_THRESHOLD_PX_S = 5.0
SPRINT_SPEED_THRESHOLD_PX_S = 100.0
PROXIMITY_DISTANCE_PX = 80.0
REPEATED_APPROACH_MIN_CYCLES = 2
COSINE_SIMILARITY_APPROACH_THRESHOLD = 0.5  # cos(theta) > 0.5 => angle < 60 deg
