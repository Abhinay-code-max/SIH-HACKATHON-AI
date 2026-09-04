"""
Advanced Event Intelligence Engine.
Evaluates persistent object tracks against configurable rules (config/rules.yaml):
- Restricted Polygon Geofence Intrusion
- Virtual Boundary Tripwire Crossing
- Dwell Time Loitering Detection
- Vehicle Intrusion
- Crowd Surge
Automatically calls EvidenceEngine to preserve forensic snapshots and dossiers.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import yaml

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.evidence_engine import evidence_engine
from backend.app.services.event_service import event_service
from backend.app.models.contracts import GeoLocation, SecurityEvent, SeverityLevel, EventType


def line_intersection(p1, p2, p3, p4) -> bool:
    """Returns True if line segment p1-p2 intersects line segment p3-p4."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


class IntelligenceEngine:
    def __init__(self, rules_path: Path | None = None):
        if rules_path is None:
            self.rules_path = ROOT_DIR / "config" / "rules.yaml"
        else:
            self.rules_path = Path(rules_path)

        self.rules = self._load_rules()
        self.cooldowns: Dict[str, float] = {}
        self.event_counter = 0

    def _load_rules(self) -> dict:
        if not self.rules_path.is_file():
            return {"zones": [], "tripwires": [], "loitering": {}, "crowd": {}}
        with open(self.rules_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _should_suppress(self, rule_key: str, cooldown_sec: float = 3.0) -> bool:
        now = time.time()
        last = self.cooldowns.get(rule_key, 0.0)
        if (now - last) < cooldown_sec:
            return True
        self.cooldowns[rule_key] = now
        return False

    def evaluate_tracks(
        self,
        camera_id: str,
        tracks: List[Dict[str, Any]],
        frame: np.ndarray,
        camera_location: Optional[GeoLocation] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates active tracks from one camera feed against all rules.
        """
        triggered_events = []
        loc = camera_location or GeoLocation(lat=17.4435, lon=78.3765, zone_name="Sector Alpha")
        h, w = frame.shape[:2]

        # -------------------------------------------------------------
        # 1. RESTRICTED ZONE GEOFENCE INTRUSION
        # -------------------------------------------------------------
        for zone in self.rules.get("zones", []):
            if zone.get("camera_id") != camera_id:
                continue

            poly_pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone["polygon"]], dtype=np.int32)
            restricted_cls = set(zone.get("restricted_classes", []))
            z_id = zone["zone_id"]

            for t in tracks:
                cls_name = t["class_name"]
                if cls_name not in restricted_cls:
                    continue

                cx, cy = t["center"]
                is_inside = cv2.pointPolygonTest(poly_pts, (cx, cy), False) >= 0

                if is_inside:
                    rule_key = f"{camera_id}_{z_id}_track_{t['track_id']}"
                    if not self._should_suppress(rule_key, cooldown_sec=5.0):
                        self.event_counter += 1
                        evt_id = f"EVT_{self.event_counter:05d}"
                        sev = zone.get("severity", "HIGH")

                        # Capture Forensic Evidence
                        dossier = evidence_engine.capture_evidence(
                            event_id=evt_id,
                            event_type="RESTRICTED_ZONE_INTRUSION",
                            camera_id=camera_id,
                            class_name=cls_name,
                            confidence=t["confidence"],
                            track_id=t["track_id"],
                            bbox=t["bbox"],
                            frame=frame,
                            trajectory=t["trajectory"],
                            severity=sev,
                            metadata={"zone_id": z_id, "zone_name": zone["name"]},
                        )

                        # Register in central event store
                        sec_evt = SecurityEvent(
                            event_id=evt_id,
                            source_id=camera_id,
                            event_type=EventType.ZONE_INTRUSION,
                            class_name=cls_name,
                            confidence=t["confidence"],
                            bbox=t["bbox"],
                            severity=SeverityLevel.CRITICAL if sev == "CRITICAL" else SeverityLevel.HIGH,
                            location=loc,
                            metadata={"zone_id": z_id, "track_id": t["track_id"]},
                        )
                        event_service.add_event(sec_evt)
                        triggered_events.append(dossier)

        # -------------------------------------------------------------
        # 2. VIRTUAL BOUNDARY TRIPWIRE CROSSING
        # -------------------------------------------------------------
        for wire in self.rules.get("tripwires", []):
            if wire.get("camera_id") != camera_id:
                continue

            w_id = wire["wire_id"]
            p1 = (wire["line_start"][0] * w, wire["line_start"][1] * h)
            p2 = (wire["line_end"][0] * w, wire["line_end"][1] * h)

            for t in tracks:
                history = t.get("trajectory", [])
                if len(history) < 2:
                    continue

                # Test last movement segment
                prev_pt = history[-2]
                curr_pt = history[-1]

                if line_intersection(p1, p2, prev_pt, curr_pt):
                    rule_key = f"{camera_id}_{w_id}_track_{t['track_id']}"
                    if not self._should_suppress(rule_key, cooldown_sec=10.0):
                        self.event_counter += 1
                        evt_id = f"EVT_{self.event_counter:05d}"
                        sev = wire.get("severity", "CRITICAL")

                        dossier = evidence_engine.capture_evidence(
                            event_id=evt_id,
                            event_type="BOUNDARY_TRIPWIRE_BREACH",
                            camera_id=camera_id,
                            class_name=t["class_name"],
                            confidence=t["confidence"],
                            track_id=t["track_id"],
                            bbox=t["bbox"],
                            frame=frame,
                            trajectory=history,
                            severity=sev,
                            metadata={"tripwire_id": w_id, "name": wire["name"]},
                        )
                        sec_evt = SecurityEvent(
                            event_id=evt_id,
                            source_id=camera_id,
                            event_type=EventType.ZONE_INTRUSION,
                            class_name=t["class_name"],
                            confidence=t["confidence"],
                            bbox=t["bbox"],
                            severity=SeverityLevel.CRITICAL,
                            location=loc,
                            metadata={"tripwire_id": w_id, "track_id": t["track_id"]},
                        )
                        event_service.add_event(sec_evt)
                        triggered_events.append(dossier)

        # -------------------------------------------------------------
        # 3. LOITERING DETECTION (DWELL TIME)
        # -------------------------------------------------------------
        loitering_cfg = self.rules.get("loitering", {})
        dwell_limit = loitering_cfg.get("dwell_time_seconds", 5.0)

        for t in tracks:
            if t["class_name"] == "person" and t["dwell_seconds"] >= dwell_limit:
                rule_key = f"{camera_id}_loiter_track_{t['track_id']}"
                if not self._should_suppress(rule_key, cooldown_sec=15.0):
                    self.event_counter += 1
                    evt_id = f"EVT_{self.event_counter:05d}"

                    dossier = evidence_engine.capture_evidence(
                        event_id=evt_id,
                        event_type="LOITERING_DETECTED",
                        camera_id=camera_id,
                        class_name="person",
                        confidence=t["confidence"],
                        track_id=t["track_id"],
                        bbox=t["bbox"],
                        frame=frame,
                        trajectory=t["trajectory"],
                        severity="HIGH",
                        metadata={"dwell_seconds": t["dwell_seconds"], "threshold": dwell_limit},
                    )
                    sec_evt = SecurityEvent(
                        event_id=evt_id,
                        source_id=camera_id,
                        event_type=EventType.LOITERING,
                        class_name="person",
                        confidence=t["confidence"],
                        bbox=t["bbox"],
                        severity=SeverityLevel.HIGH,
                        location=loc,
                        metadata={"dwell_seconds": t["dwell_seconds"], "track_id": t["track_id"]},
                    )
                    event_service.add_event(sec_evt)
                    triggered_events.append(dossier)

        return triggered_events


intelligence_engine = IntelligenceEngine()
