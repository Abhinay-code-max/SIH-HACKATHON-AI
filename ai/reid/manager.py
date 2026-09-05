"""
Global Subject Manager & Cross-Camera Journey Tracker.
Assigns and maintains persistent Global Subject IDs (e.g. SUBJ_0001),
tracks multi-camera transit timelines, records visual evidence crops,
and performs appearance-based forensic query search.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.reid.extractor import feature_extractor
from ai.reid.association import cross_camera_associator

REID_CROPS_DIR = ROOT_DIR / "data" / "evidence" / "reid_crops"
REID_CROPS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_FILE = ROOT_DIR / "data" / "sample-events" / "global_subjects.json"


class GlobalSubjectManager:
    def __init__(self):
        self.next_subject_idx = 1
        # subject_id -> subject profile dict
        self.subjects: Dict[str, Dict[str, Any]] = {}
        # (camera_id, local_track_id) -> subject_id
        self.active_track_map: Dict[Tuple[str, int], str] = {}
        # Track last crop extraction timestamp per local track
        self.last_embed_time: Dict[Tuple[str, int], float] = {}
        # Chronological transit handoff events
        self.transit_log: List[Dict[str, Any]] = []
        self._load_persisted()

    def _load_persisted(self):
        """Loads previously recorded subjects if available."""
        if STORAGE_FILE.is_file():
            try:
                with open(STORAGE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for s in data.get("subjects", []):
                    sid = s["subject_id"]
                    # Signatures need to be stored as list and converted to numpy
                    if "signature" in s and s["signature"]:
                        s["signature"] = np.array(s["signature"], dtype=np.float32)
                    self.subjects[sid] = s
                self.next_subject_idx = data.get("next_subject_idx", len(self.subjects) + 1)
                self.transit_log = data.get("transits", [])
            except Exception:
                pass

    def _save_persisted(self):
        """Saves subjects and transits to JSON for air-gapped persistence."""
        try:
            ser_subjects = []
            for s in self.subjects.values():
                c = dict(s)
                if isinstance(c.get("signature"), np.ndarray):
                    c["signature"] = c["signature"].tolist()
                ser_subjects.append(c)

            payload = {
                "next_subject_idx": self.next_subject_idx,
                "subjects": ser_subjects,
                "transits": self.transit_log[-100:],  # keep last 100 transits
            }
            with open(STORAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def _extract_crop(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Safely extracts bbox crop with boundary clamping."""
        if frame is None or len(frame.shape) < 2:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 15 or (y2 - y1) < 20:
            return None
        return frame[y1:y2, x1:x2].copy()

    def process_track(
        self,
        camera_id: str,
        local_track_id: int,
        class_name: str,
        bbox: List[float],
        frame: np.ndarray,
        dwell_seconds: float = 0.0,
        confidence: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Main Re-ID pipeline invoked per tracked object per frame.
        Maps local track to Global Subject ID via continuous tracking or cross-camera Re-ID.
        """
        now = time.time()
        iso_now = datetime.now(timezone.utc).isoformat()
        key = (camera_id, local_track_id)

        # 1. Fast Path: Already mapped to a Global Subject ID
        if key in self.active_track_map:
            subject_id = self.active_track_map[key]
            if subject_id in self.subjects:
                subj = self.subjects[subject_id]
                subj["last_seen"] = now
                subj["last_camera_id"] = camera_id
                subj["last_local_track_id"] = local_track_id
                subj["is_active"] = True

                # Periodically update EMA signature (every 3 seconds) to capture pose/lighting shifts
                last_time = self.last_embed_time.get(key, 0.0)
                if (now - last_time) > 3.0:
                    crop = self._extract_crop(frame, bbox)
                    if crop is not None:
                        new_sig = feature_extractor.extract_signature(crop)
                        if new_sig is not None and len(new_sig) > 0:
                            old_sig = subj.get("signature")
                            if old_sig is not None and len(old_sig) == len(new_sig):
                                blended = 0.85 * old_sig + 0.15 * new_sig
                                subj["signature"] = blended / (np.linalg.norm(blended) + 1e-7)
                    self.last_embed_time[key] = now

                return subj

        # 2. Extract visual crop & appearance signature
        crop = self._extract_crop(frame, bbox)
        signature = None
        crop_rel_path = None

        if crop is not None:
            signature = feature_extractor.extract_signature(crop)

        # 3. Attempt Cross-Camera Association against all known subjects
        candidates = list(self.subjects.values())
        match_result = cross_camera_associator.match_candidate(
            query_signature=signature,
            query_class=class_name,
            query_camera_id=camera_id,
            query_timestamp=now,
            candidates=candidates,
        )

        if match_result is not None:
            # Found existing Global Subject across camera handoff!
            matched_subj, sim_score, verdict = match_result
            subject_id = matched_subj["subject_id"]
            prev_cam = matched_subj.get("last_camera_id", camera_id)

            # Record Inter-Camera Transit if moving across distinct cameras
            if prev_cam != camera_id:
                prev_time = matched_subj.get("last_seen", now)
                transit_sec = round(abs(now - prev_time), 1)
                transit_event = {
                    "transit_id": f"TR_{len(self.transit_log) + 1:04d}",
                    "subject_id": subject_id,
                    "display_name": matched_subj["display_name"],
                    "class_name": class_name,
                    "from_camera": prev_cam,
                    "to_camera": camera_id,
                    "transit_duration_sec": transit_sec,
                    "similarity_score": sim_score,
                    "verdict": verdict,
                    "timestamp": now,
                    "iso_time": iso_now,
                }
                self.transit_log.append(transit_event)
                matched_subj.setdefault("transits", []).append(transit_event)

            # Update subject state
            matched_subj["last_seen"] = now
            matched_subj["last_camera_id"] = camera_id
            matched_subj["last_local_track_id"] = local_track_id
            matched_subj["is_active"] = True
            matched_subj["sightings_count"] = matched_subj.get("sightings_count", 0) + 1

            # Update EMA signature
            if signature is not None and matched_subj.get("signature") is not None:
                blended = 0.80 * matched_subj["signature"] + 0.20 * signature
                matched_subj["signature"] = blended / (np.linalg.norm(blended) + 1e-7)

            self.active_track_map[key] = subject_id
            self.last_embed_time[key] = now
            self._save_persisted()
            return matched_subj

        # 4. No match found -> Register NEW Global Subject
        subject_id = f"SUBJ_{self.next_subject_idx:04d}"
        display_name = f"[GLOBAL #{self.next_subject_idx:02d}: {class_name.upper()}]"
        self.next_subject_idx += 1

        # Save initial evidence crop to disk
        if crop is not None:
            crop_filename = f"{subject_id}_init_{int(now)}.jpg"
            crop_abs_path = REID_CROPS_DIR / crop_filename
            cv2.imwrite(str(crop_abs_path), crop)
            crop_rel_path = f"/reid_crops/{crop_filename}"

        new_subject: Dict[str, Any] = {
            "subject_id": subject_id,
            "display_name": display_name,
            "class_name": class_name,
            "first_seen": now,
            "last_seen": now,
            "first_camera_id": camera_id,
            "last_camera_id": camera_id,
            "last_local_track_id": local_track_id,
            "is_active": True,
            "sightings_count": 1,
            "representative_crop": crop_rel_path,
            "signature": signature if signature is not None else np.zeros(656, dtype=np.float32),
            "sightings": [
                {
                    "camera_id": camera_id,
                    "local_track_id": local_track_id,
                    "timestamp": now,
                    "iso_time": iso_now,
                    "bbox": bbox,
                    "confidence": round(confidence, 3),
                    "crop_url": crop_rel_path,
                }
            ],
            "transits": [],
        }

        self.subjects[subject_id] = new_subject
        self.active_track_map[key] = subject_id
        self.last_embed_time[key] = now
        self._save_persisted()
        return new_subject

    def prune_stale_active_tracks(self, active_keys: List[Tuple[str, int]]):
        """Cleans up internal track mapping for lost local tracks."""
        active_set = set(active_keys)
        keys_to_remove = [k for k in self.active_track_map.keys() if k not in active_set]
        now = time.time()
        for k in keys_to_remove:
            sid = self.active_track_map.pop(k, None)
            self.last_embed_time.pop(k, None)
            if sid and sid in self.subjects:
                # If subject has no other active tracks, mark inactive
                other_active = any(v == sid for v in self.active_track_map.values())
                if not other_active and (now - self.subjects[sid]["last_seen"]) > 10.0:
                    self.subjects[sid]["is_active"] = False

    def get_all_subjects(self) -> List[Dict[str, Any]]:
        """Returns clean list of all subjects (signatures omitted for lightweight JSON)."""
        now = time.time()
        result = []
        for s in self.subjects.values():
            s_copy = dict(s)
            s_copy.pop("signature", None)
            s_copy["is_active"] = (now - s_copy.get("last_seen", 0.0)) <= 15.0
            result.append(s_copy)
        # Sort latest seen first
        return sorted(result, key=lambda x: x.get("last_seen", 0), reverse=True)

    def get_subject_dossier(self, subject_id: str) -> Optional[Dict[str, Any]]:
        if subject_id not in self.subjects:
            return None
        s = dict(self.subjects[subject_id])
        s.pop("signature", None)
        s["is_active"] = (time.time() - s.get("last_seen", 0.0)) <= 15.0
        return s

    def get_recent_transits(self, limit: int = 25) -> List[Dict[str, Any]]:
        return self.transit_log[-limit:][::-1]

    def search_by_image(self, query_img: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Forensic similarity search: given an uploaded image or crop,
        ranks matching Global Subjects and returns sightings timeline.
        """
        if query_img is None or query_img.size == 0:
            return []

        query_sig = feature_extractor.extract_signature(query_img)
        if query_sig is None or len(query_sig) == 0:
            return []

        results = []
        for s in self.subjects.values():
            cand_sig = s.get("signature")
            if cand_sig is None or len(cand_sig) == 0:
                continue
            sim = feature_extractor.compute_similarity(query_sig, cand_sig)
            if sim > 0.35:  # Relevance cutoff
                s_copy = dict(s)
                s_copy.pop("signature", None)
                s_copy["similarity_score"] = round(sim, 3)
                results.append(s_copy)

        results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]


global_subject_manager = GlobalSubjectManager()
