"""
Evidence Capture & Forensic Dossier Engine.
Generates full-frame snapshots, cropped target images, and forensic JSON dossiers
for every security event. Manages operator review workflows (ACKNOWLEDGE, DISMISS, ESCALATE).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import cv2

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class EvidenceEngine:
    def __init__(self, evidence_dir: Path | None = None):
        if evidence_dir is None:
            self.evidence_dir = ROOT_DIR / "data" / "evidence"
        else:
            self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_file = ROOT_DIR / "data" / "operator_audit_log.json"

    def capture_evidence(
        self,
        event_id: str,
        event_type: str,
        camera_id: str,
        class_name: str,
        confidence: float,
        track_id: Optional[int],
        bbox: List[float],
        frame: Any,
        trajectory: Optional[List[List[float]]] = None,
        severity: str = "HIGH",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Saves full snapshot, cropped target, and dossier for human investigation.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, [max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])])

        # 1. Full Snapshot
        snapshot_filename = f"{event_id}_snapshot.jpg"
        snapshot_path = self.evidence_dir / snapshot_filename
        cv2.imwrite(str(snapshot_path), frame)

        # 2. Target Crop
        crop_filename = f"{event_id}_crop.jpg"
        crop_path = self.evidence_dir / crop_filename
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            cropped_img = frame[y1:y2, x1:x2]
            cv2.imwrite(str(crop_path), cropped_img)
        else:
            shutil.copy2(snapshot_path, crop_path)

        # 3. Evidence Dossier JSON
        dossier = {
            "event_id": event_id,
            "event_type": event_type,
            "camera_id": camera_id,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": {
                "class": class_name,
                "confidence": round(confidence, 3),
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "trajectory_trail": trajectory or [],
            },
            "evidence_files": {
                "snapshot": f"/evidence/{snapshot_filename}",
                "crop": f"/evidence/{crop_filename}",
            },
            "operator_status": "NEW",  # NEW | ACKNOWLEDGED | DISMISSED | ESCALATED
            "operator_action_history": [],
            "metadata": metadata or {},
        }

        dossier_filename = f"{event_id}_dossier.json"
        dossier_path = self.evidence_dir / dossier_filename
        with open(dossier_path, "w", encoding="utf-8") as f:
            json.dump(dossier, f, indent=2)

        return dossier

    def update_operator_action(
        self,
        event_id: str,
        action: str,  # ACKNOWLEDGE | DISMISS | ESCALATE
        operator_name: str = "Operator-1",
        notes: str = "",
    ) -> Dict[str, Any]:
        """Records operator intervention in the evidence dossier and audit log."""
        dossier_path = self.evidence_dir / f"{event_id}_dossier.json"
        if not dossier_path.is_file():
            raise FileNotFoundError(f"Evidence dossier not found for {event_id}")

        with open(dossier_path, "r", encoding="utf-8") as f:
            dossier = json.load(f)

        now_str = datetime.now(timezone.utc).isoformat()
        dossier["operator_status"] = action.upper()
        action_record = {
            "timestamp": now_str,
            "action": action.upper(),
            "operator": operator_name,
            "notes": notes,
        }
        dossier["operator_action_history"].append(action_record)

        with open(dossier_path, "w", encoding="utf-8") as f:
            json.dump(dossier, f, indent=2)

        # Append to system audit log
        audit_records = []
        if self.audit_log_file.is_file():
            try:
                with open(self.audit_log_file, "r", encoding="utf-8") as f:
                    audit_records = json.load(f)
            except Exception:
                audit_records = []

        audit_records.append({
            "event_id": event_id,
            "camera_id": dossier.get("camera_id"),
            **action_record,
        })

        with open(self.audit_log_file, "w", encoding="utf-8") as f:
            json.dump(audit_records, f, indent=2)

        return dossier

    def get_dossier(self, event_id: str) -> Optional[Dict[str, Any]]:
        dossier_path = self.evidence_dir / f"{event_id}_dossier.json"
        if not dossier_path.is_file():
            return None
        with open(dossier_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_all_evidence(self) -> List[Dict[str, Any]]:
        results = []
        for p in sorted(self.evidence_dir.glob("*_dossier.json"), reverse=True):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
            except Exception:
                pass
        return results


evidence_engine = EvidenceEngine()
