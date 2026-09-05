"""
Cross-Camera Spatio-Temporal Association Engine.
Validates physical transit corridors, enforces teleportation rejection,
and performs appearance-based bipartite matching between camera feeds.
"""

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.reid.extractor import feature_extractor


@dataclass
class TransitCorridor:
    cam_a: str
    cam_b: str
    min_transit_sec: float
    max_transit_sec: float
    corridor_name: str


class CrossCameraAssociator:
    def __init__(self, match_threshold: float = 0.62):
        self.match_threshold = match_threshold
        # Symmetrical corridor definitions
        self.corridors: Dict[Tuple[str, str], TransitCorridor] = {}
        self._register_default_corridors()

    def _register_default_corridors(self):
        corridor_list = [
            # CAM_01 (Concourse) <-> CAM_02 (Gate 1)
            TransitCorridor("CAM_01", "CAM_02", 1.0, 180.0, "Concourse to Gate 1 Corridor"),
            # CAM_02 (Gate 1) <-> CAM_03 (Perimeter Command)
            TransitCorridor("CAM_02", "CAM_03", 1.5, 240.0, "Gate 1 to Perimeter Sector Corridor"),
            # CAM_01 (Concourse) <-> CAM_03 (Perimeter Command)
            TransitCorridor("CAM_01", "CAM_03", 2.0, 300.0, "Concourse Direct to Perimeter Command Corridor"),
            # Same camera re-association after occlusion/loss
            TransitCorridor("CAM_01", "CAM_01", 0.5, 60.0, "CAM_01 Intra-Zone Occlusion Corridor"),
            TransitCorridor("CAM_02", "CAM_02", 0.5, 60.0, "CAM_02 Intra-Zone Occlusion Corridor"),
            TransitCorridor("CAM_03", "CAM_03", 0.5, 60.0, "CAM_03 Intra-Zone Occlusion Corridor"),
        ]
        for c in corridor_list:
            self.corridors[(c.cam_a, c.cam_b)] = c
            self.corridors[(c.cam_b, c.cam_a)] = c

    def get_corridor(self, cam_from: str, cam_to: str) -> Optional[TransitCorridor]:
        return self.corridors.get((cam_from, cam_to))

    def validate_transit(self, cam_from: str, cam_to: str, delta_sec: float) -> Tuple[bool, str]:
        """
        Validates whether movement between two cameras is physically possible.
        Rejects impossible speed (teleportation anomalies) and expired windows.
        """
        corridor = self.get_corridor(cam_from, cam_to)
        if corridor is None:
            # Default fallback for unconfigured camera pair
            if delta_sec < 0.8:
                return False, "PHYSICAL_ANOMALY_TELEPORTATION"
            if delta_sec > 600.0:
                return False, "TRANSIT_WINDOW_EXPIRED"
            return True, "DEFAULT_CORRIDOR_PERMITTED"

        if delta_sec < corridor.min_transit_sec:
            return False, f"TELEPORTATION_ANOMALY (delta {delta_sec:.1f}s < min {corridor.min_transit_sec:.1f}s)"
        if delta_sec > corridor.max_transit_sec:
            return False, f"TRANSIT_EXPIRED (delta {delta_sec:.1f}s > max {corridor.max_transit_sec:.1f}s)"

        return True, f"VALID_CORRIDOR: {corridor.corridor_name}"

    def match_candidate(
        self,
        query_signature: np.ndarray,
        query_class: str,
        query_camera_id: str,
        query_timestamp: float,
        candidates: List[dict],
    ) -> Optional[Tuple[dict, float, str]]:
        """
        Finds the best matching global subject among recent candidate sightings.
        Returns (matched_candidate, similarity_score, transit_verdict) or None.
        """
        if query_signature is None or len(query_signature) == 0:
            return None

        best_candidate = None
        best_sim = 0.0
        best_verdict = ""

        for cand in candidates:
            # 1. Target Class must match (person -> person, car -> car)
            cand_class = cand.get("class_name")
            if cand_class and cand_class != query_class:
                continue

            cand_camera_id = cand.get("last_camera_id", query_camera_id)
            cand_timestamp = cand.get("last_seen", query_timestamp)
            delta_sec = abs(query_timestamp - cand_timestamp)

            # 2. Spatio-temporal transit validation
            is_valid, verdict = self.validate_transit(cand_camera_id, query_camera_id, delta_sec)
            if not is_valid:
                continue

            # 3. Visual Appearance Cosine Similarity
            cand_sig = cand.get("signature")
            if cand_sig is None or len(cand_sig) == 0:
                continue

            sim = feature_extractor.compute_similarity(query_signature, cand_sig)
            if sim >= self.match_threshold and sim > best_sim:
                best_sim = sim
                best_candidate = cand
                best_verdict = verdict

        if best_candidate is not None:
            return best_candidate, round(best_sim, 3), best_verdict

        return None


cross_camera_associator = CrossCameraAssociator()
