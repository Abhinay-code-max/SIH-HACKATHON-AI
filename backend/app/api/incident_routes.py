"""
Tactical Incident Management, Behavioral Anomaly & Webhook REST Gateway.
Exposes clean, decoupled endpoints for external UI and security operations:
- GET /api/incidents: List all compound incidents with DEFCON threat indexes.
- GET /api/incidents/defcon: Current system-wide defense readiness condition.
- GET /api/incidents/{id}: Full incident dossier and associated evidence.
- GET /api/incidents/{id}/report: Self-contained forensic HTML report.
- POST /api/incidents/{id}/status: Operator status update & dispatch notes.
- GET /api/anomalies/config: Read current anomaly detection thresholds.
- POST /api/anomalies/config: Update thresholds dynamically.
- POST /api/webhooks/register: Register external SOC dispatch webhooks.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.events.incident_manager import incident_manager
from ai.events.behavioral_engine import behavioral_anomaly_engine

router = APIRouter(prefix="/api/incidents", tags=["Incidents & Behavioral Anomalies"])


class StatusUpdateRequest(BaseModel):
    status: str  # ACTIVE, INVESTIGATING, ACKNOWLEDGED, RESOLVED, ESCALATED
    operator_name: str = "Commander"
    notes: Optional[str] = None


class AnomalyConfigModel(BaseModel):
    sprint_velocity_threshold: Optional[float] = None
    baggage_isolation_distance: Optional[float] = None
    baggage_timeout_seconds: Optional[float] = None
    tailgating_window_seconds: Optional[float] = None
    crowd_cluster_radius: Optional[float] = None
    crowd_cluster_min_persons: Optional[int] = None


class WebhookRegisterModel(BaseModel):
    webhook_url: str


@router.get("")
def list_incidents(
    status: Optional[str] = Query(None, description="Filter by status: ACTIVE, INVESTIGATING, ACKNOWLEDGED, RESOLVED, ESCALATED")
) -> List[Dict[str, Any]]:
    """Returns all compound incidents with threat scores and DEFCON classifications."""
    return incident_manager.get_all_incidents(status_filter=status)


@router.get("/defcon")
def get_grid_defcon_status() -> Dict[str, Any]:
    """Returns the highest active defense readiness condition across the camera grid."""
    return incident_manager.get_current_system_defcon()


@router.get("/{incident_id}")
def get_incident_details(incident_id: str) -> Dict[str, Any]:
    """Returns full incident dossier including crop evidence and event timelines."""
    inc = incident_manager.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    return inc


@router.get("/{incident_id}/report")
def export_incident_report(incident_id: str):
    """Generates an air-gapped, printable HTML incident report with embedded evidence."""
    html_content = incident_manager.generate_html_report(incident_id)
    if not html_content:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    return Response(content=html_content, media_type="text/html")


@router.post("/{incident_id}/status")
def update_incident_status(incident_id: str, req: StatusUpdateRequest):
    """Allows human operators to transition incident state and append dispatch notes."""
    inc = incident_manager.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    inc["status"] = req.status
    inc.setdefault("operator_audit", []).append({
        "status": req.status,
        "operator_name": req.operator_name,
        "notes": req.notes or "Status updated",
        "timestamp": incident_manager._load_persisted() or 0,
    })
    incident_manager._save_persisted()
    return {"status": "UPDATED", "incident_id": incident_id, "current_status": req.status}


@router.get("/anomalies/config")
def get_anomaly_configuration() -> Dict[str, Any]:
    """Returns active behavioral motion parameters."""
    return {
        "sprint_velocity_threshold": behavioral_anomaly_engine.sprint_velocity_threshold,
        "baggage_isolation_distance": behavioral_anomaly_engine.baggage_isolation_distance,
        "baggage_timeout_seconds": behavioral_anomaly_engine.baggage_timeout_seconds,
        "tailgating_window_seconds": behavioral_anomaly_engine.tailgating_window_seconds,
        "crowd_cluster_radius": behavioral_anomaly_engine.crowd_cluster_radius,
        "crowd_cluster_min_persons": behavioral_anomaly_engine.crowd_cluster_min_persons,
    }


@router.post("/anomalies/config")
def update_anomaly_configuration(cfg: AnomalyConfigModel):
    """Dynamically tunes behavioral anomaly detection parameters."""
    if cfg.sprint_velocity_threshold is not None:
        behavioral_anomaly_engine.sprint_velocity_threshold = cfg.sprint_velocity_threshold
    if cfg.baggage_isolation_distance is not None:
        behavioral_anomaly_engine.baggage_isolation_distance = cfg.baggage_isolation_distance
    if cfg.baggage_timeout_seconds is not None:
        behavioral_anomaly_engine.baggage_timeout_seconds = cfg.baggage_timeout_seconds
    if cfg.tailgating_window_seconds is not None:
        behavioral_anomaly_engine.tailgating_window_seconds = cfg.tailgating_window_seconds
    if cfg.crowd_cluster_radius is not None:
        behavioral_anomaly_engine.crowd_cluster_radius = cfg.crowd_cluster_radius
    if cfg.crowd_cluster_min_persons is not None:
        behavioral_anomaly_engine.crowd_cluster_min_persons = cfg.crowd_cluster_min_persons

    return {"status": "CONFIG_UPDATED", "config": get_anomaly_configuration()}


@router.post("/webhooks/register")
def register_soc_webhook(req: WebhookRegisterModel):
    """Registers an external webhook URL for real-time dispatch alerts."""
    hooks = incident_manager.register_webhook(req.webhook_url)
    return {"status": "WEBHOOK_REGISTERED", "webhooks": hooks}
