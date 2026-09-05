"""
Tactical Incident Management & Compound Threat Risk Scoring Engine.
Synthesizes multiple security events, behavioral anomalies, and cross-camera Re-ID
sightings into quantified Incident Dossiers (INC_0001) with DEFCON threat indexes (0-100).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.request

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

INCIDENTS_STORAGE = ROOT_DIR / "data" / "sample-events" / "incidents.json"


class IncidentManager:
    def __init__(self):
        self.next_incident_idx = 1
        # incident_id -> incident dict
        self.incidents: Dict[str, Dict[str, Any]] = {}
        # List of external registered webhook URLs
        self.registered_webhooks: List[str] = []
        self._load_persisted()

    def _load_persisted(self):
        if INCIDENTS_STORAGE.is_file():
            try:
                with open(INCIDENTS_STORAGE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.incidents = {i["incident_id"]: i for i in data.get("incidents", [])}
                self.next_incident_idx = data.get("next_incident_idx", len(self.incidents) + 1)
                self.registered_webhooks = data.get("webhooks", [])
            except Exception:
                pass

    def _save_persisted(self):
        try:
            payload = {
                "next_incident_idx": self.next_incident_idx,
                "incidents": list(self.incidents.values())[-50:],
                "webhooks": self.registered_webhooks,
            }
            with open(INCIDENTS_STORAGE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def calculate_threat_score(self, event_descriptors: List[Dict[str, Any]], has_multi_camera_transit: bool = False) -> int:
        """
        Computes composite tactical risk index (0–100).
        """
        score = 20  # Baseline presence score

        for ev in event_descriptors:
            ev_type = ev.get("event_type") or ev.get("anomaly_type", "")
            sev = ev.get("severity", "MEDIUM")

            if sev == "CRITICAL":
                score += 30
            elif sev == "HIGH":
                score += 15

            if ev_type == "UNATTENDED_BAGGAGE_ALERT":
                score += 35
            elif ev_type == "TAILGATING_BREACH":
                score += 25
            elif ev_type == "SPRINTING_DETECTED":
                score += 20
            elif ev_type == "RESTRICTED_ZONE_INTRUSION":
                score += 25
            elif ev_type == "LOITERING_DETECTED":
                score += 15

            if ev.get("class_name") in {"truck", "bus", "unauthorized_vehicle"}:
                score += 20

        if has_multi_camera_transit:
            score += 15  # Coordinated movement across sectors

        return min(100, max(0, score))

    def get_defcon_level(self, threat_score: int) -> Dict[str, str]:
        """Maps numerical threat score to defense condition level."""
        if threat_score >= 75:
            return {"level": "DEFCON_1", "status": "CRITICAL", "color": "#EF4444", "description": "Immediate Tactical Threat / Armed Escalation Required"}
        elif threat_score >= 45:
            return {"level": "DEFCON_2", "status": "ELEVATED", "color": "#F59E0B", "description": "Heightened Perimeter Alert / Intercept Patrol Dispatched"}
        else:
            return {"level": "DEFCON_3", "status": "NORMAL", "color": "#10B981", "description": "Routine Surveillance / Standard Monitoring"}

    def register_or_update_incident(
        self,
        camera_id: str,
        events: List[Dict[str, Any]],
        global_subject_id: Optional[str] = None,
        crop_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Groups real-time events into a compound Incident.
        Updates existing active incident or spawns new one.
        """
        now = time.time()
        iso_now = datetime.now(timezone.utc).isoformat()

        # Check if there is an ongoing active incident on this camera or global subject within 30 seconds
        active_inc = None
        for inc in reversed(list(self.incidents.values())):
            if inc["status"] in {"ACTIVE", "INVESTIGATING"}:
                time_diff = now - inc.get("updated_at", 0)
                if time_diff < 35.0:
                    if global_subject_id and inc.get("global_subject_id") == global_subject_id:
                        active_inc = inc
                        break
                    if inc.get("primary_camera_id") == camera_id:
                        active_inc = inc
                        break

        if active_inc is None:
            # Create New Incident
            inc_id = f"INC_{self.next_incident_idx:04d}"
            self.next_incident_idx += 1

            first_type = events[0].get("event_type") or events[0].get("anomaly_type", "SECURITY_ALERT")
            title = f"{first_type.replace('_', ' ').title()} ({camera_id})"

            threat_score = self.calculate_threat_score(events)
            defcon = self.get_defcon_level(threat_score)

            incident: Dict[str, Any] = {
                "incident_id": inc_id,
                "title": title,
                "threat_score": threat_score,
                "defcon": defcon,
                "primary_camera_id": camera_id,
                "involved_cameras": [camera_id],
                "global_subject_id": global_subject_id,
                "status": "ACTIVE",
                "created_at": now,
                "updated_at": now,
                "created_iso": iso_now,
                "events_count": len(events),
                "events_log": events,
                "crops": [crop_url] if crop_url else [],
                "operator_audit": [],
            }
            self.incidents[inc_id] = incident
            self._save_persisted()
            self._dispatch_webhooks(incident)
            return incident
        else:
            # Append to active incident
            active_inc["updated_at"] = now
            if camera_id not in active_inc["involved_cameras"]:
                active_inc["involved_cameras"].append(camera_id)
            if global_subject_id and not active_inc.get("global_subject_id"):
                active_inc["global_subject_id"] = global_subject_id
            if crop_url and crop_url not in active_inc["crops"]:
                active_inc["crops"].append(crop_url)

            active_inc["events_log"].extend(events)
            active_inc["events_count"] = len(active_inc["events_log"])

            # Recalculate threat score
            has_multi_cam = len(active_inc["involved_cameras"]) > 1
            active_inc["threat_score"] = self.calculate_threat_score(active_inc["events_log"], has_multi_cam)
            active_inc["defcon"] = self.get_defcon_level(active_inc["threat_score"])

            self._save_persisted()
            return active_inc

    def get_current_system_defcon(self) -> Dict[str, Any]:
        """Returns the highest active DEFCON state across the entire security grid."""
        now = time.time()
        active_scores = [
            inc["threat_score"]
            for inc in self.incidents.values()
            if inc.get("status") in {"ACTIVE", "INVESTIGATING"} and (now - inc.get("updated_at", 0)) < 40.0
        ]
        max_score = max(active_scores) if active_scores else 15
        defcon = self.get_defcon_level(max_score)
        return {
            "threat_score": max_score,
            "defcon": defcon,
            "active_incidents_count": len(active_scores),
        }

    def get_all_incidents(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        result = list(self.incidents.values())
        if status_filter:
            result = [i for i in result if i.get("status") == status_filter]
        return sorted(result, key=lambda x: x.get("updated_at", 0), reverse=True)

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        return self.incidents.get(incident_id)

    def register_webhook(self, url: str) -> List[str]:
        if url not in self.registered_webhooks:
            self.registered_webhooks.append(url)
            self._save_persisted()
        return self.registered_webhooks

    def _dispatch_webhooks(self, incident: Dict[str, Any]):
        """Dispatches notification to registered endpoints asynchronously (failsafe)."""
        if not self.registered_webhooks:
            return
        payload = json.dumps({
            "notification": "TACTICAL_SECURITY_INCIDENT_DISPATCH",
            "incident": incident,
        }).encode("utf-8")

        for url in self.registered_webhooks:
            try:
                req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=1.5)
            except Exception:
                pass

    def generate_html_report(self, incident_id: str) -> Optional[str]:
        """Generates self-contained, air-gapped HTML forensic report for printing or export."""
        inc = self.get_incident(incident_id)
        if not inc:
            return None

        crops_html = "".join([f'<img src="{c}" style="height:100px; border-radius:4px; border:1px solid #475569;"/>' for c in inc.get("crops", [])])
        events_rows = "".join([
            f"<tr><td>{ev.get('event_type') or ev.get('anomaly_type')}</td><td>{ev.get('camera_id')}</td><td>{ev.get('severity')}</td><td>{ev.get('description', 'Event recorded')}</td></tr>"
            for ev in inc.get("events_log", [])[:15]
        ])

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Incident Report — {inc['incident_id']}</title>
    <style>
        body {{ font-family: monospace, sans-serif; background: #0F172A; color: #F8FAFC; padding: 24px; }}
        .header {{ border-bottom: 2px solid #334155; padding-bottom: 12px; margin-bottom: 20px; }}
        .badge {{ padding: 4px 10px; border-radius: 4px; font-weight: bold; background: {inc['defcon']['color']}; color: #000; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 12px; }}
        th, td {{ border: 1px solid #334155; padding: 8px; text-align: left; }}
        th {{ background: #1E293B; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>BORDER SENTINEL // TACTICAL INCIDENT REPORT</h2>
        <div>Incident ID: <strong>{inc['incident_id']}</strong> | Status: <strong>{inc['status']}</strong></div>
        <div style="margin-top: 8px;"><span class="badge">{inc['defcon']['level']} — {inc['defcon']['status']} (SCORE: {inc['threat_score']}/100)</span></div>
    </div>
    <div>
        <p><strong>Primary Camera:</strong> {inc['primary_camera_id']} | <strong>Involved Corridors:</strong> {', '.join(inc['involved_cameras'])}</p>
        <p><strong>Global Subject:</strong> {inc.get('global_subject_id') or 'UNASSOCIATED'} | <strong>Time:</strong> {inc['created_iso']}</p>
        <div style="display:flex; gap:10px; margin: 12px 0;">{crops_html}</div>
    </div>
    <h3>Chronological Event & Anomaly Log</h3>
    <table>
        <tr><th>Type</th><th>Camera</th><th>Severity</th><th>Description</th></tr>
        {events_rows}
    </table>
</body>
</html>"""


incident_manager = IncidentManager()
