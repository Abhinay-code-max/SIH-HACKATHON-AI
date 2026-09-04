"""
Evidence & Operator Dispatch API Routes.
Exposes endpoints to view forensic dossiers, review snapshots, and execute operator actions
(ACKNOWLEDGE, DISMISS, ESCALATE).
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.evidence_engine import evidence_engine
from backend.app.services.camera_manager import camera_manager

router = APIRouter(prefix="/api")


class OperatorActionRequest(BaseModel):
    action: str  # "ACKNOWLEDGE" | "DISMISS" | "ESCALATE"
    operator_name: str = "Command-Officer-01"
    notes: Optional[str] = ""


@router.get("/evidence", response_model=List[Dict[str, Any]])
def list_evidence():
    """Lists all captured forensic dossiers."""
    return evidence_engine.list_all_evidence()


@router.get("/evidence/{event_id}")
def get_evidence_dossier(event_id: str):
    """Fetches details, snapshots, and trajectory for an alert."""
    dossier = evidence_engine.get_dossier(event_id)
    if not dossier:
        raise HTTPException(status_code=404, detail=f"Dossier {event_id} not found")
    return dossier


@router.post("/evidence/{event_id}/action")
def take_operator_action(event_id: str, req: OperatorActionRequest):
    """Executes human operator action (ACKNOWLEDGE / DISMISS / ESCALATE)."""
    try:
        updated = evidence_engine.update_operator_action(
            event_id=event_id,
            action=req.action,
            operator_name=req.operator_name,
            notes=req.notes or "",
        )
        return updated
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/cameras/telemetry")
def get_cameras_telemetry():
    """Returns real-time FPS, status, and track counts for all cameras."""
    return camera_manager.get_camera_status()
