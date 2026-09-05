"""
Cross-Camera Re-ID & Multi-Camera Journey REST Gateway.
Exposes clean, decoupled endpoints for external UI and forensic operators:
- GET /api/reid/subjects: List all global subjects across camera feeds.
- GET /api/reid/subjects/{id}: Complete multi-camera timeline and dossier.
- GET /api/reid/transits: Real-time inter-camera transit handoffs.
- POST /api/reid/search: Visual similarity query via uploaded image/crop.
- POST /api/reid/reset: Reset state for demo scenario replays.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.reid.manager import global_subject_manager

router = APIRouter(prefix="/api/reid", tags=["Re-ID & Cross-Camera Association"])


class SearchQueryPayload(BaseModel):
    crop_base64: Optional[str] = None
    top_k: int = 5


@router.get("/subjects")
def list_global_subjects(
    active_only: bool = Query(False, description="Filter only targets seen in the last 15 seconds"),
    class_name: Optional[str] = Query(None, description="Filter by object class (e.g. person, car)"),
) -> List[Dict[str, Any]]:
    """Returns all tracked entities identified across CCTV cameras."""
    subjects = global_subject_manager.get_all_subjects()
    if active_only:
        subjects = [s for s in subjects if s.get("is_active")]
    if class_name:
        subjects = [s for s in subjects if s.get("class_name") == class_name]
    return subjects


@router.get("/subjects/{subject_id}")
def get_subject_dossier(subject_id: str) -> Dict[str, Any]:
    """Returns complete multi-camera journey timeline and crop evidence for a subject."""
    dossier = global_subject_manager.get_subject_dossier(subject_id)
    if not dossier:
        raise HTTPException(status_code=404, detail=f"Subject {subject_id} not found in Re-ID database")
    return dossier


@router.get("/transits")
def get_inter_camera_transits(
    limit: int = Query(25, ge=1, le=100, description="Max transit handoff events to retrieve")
) -> List[Dict[str, Any]]:
    """Returns real-time log of subjects transitioning between different camera zones."""
    return global_subject_manager.get_recent_transits(limit=limit)


@router.post("/search")
async def search_subject_by_crop(
    file: UploadFile = File(..., description="Query image crop to match against CCTV sightings"),
    top_k: int = Query(5, ge=1, le=20),
):
    """
    Forensic Re-ID Search: Upload a suspicious target crop to find
    all historical sightings and transit routes across the camera network.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    query_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if query_img is None or query_img.size == 0:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file uploaded")

    matches = global_subject_manager.search_by_image(query_img, top_k=top_k)
    return {
        "query_filename": file.filename,
        "query_shape": [int(query_img.shape[1]), int(query_img.shape[0])],
        "total_matches": len(matches),
        "results": matches,
    }


@router.post("/reset")
def reset_reid_state():
    """Resets Re-ID memory and active tracks for clean demo testing."""
    global_subject_manager.subjects.clear()
    global_subject_manager.active_track_map.clear()
    global_subject_manager.last_embed_time.clear()
    global_subject_manager.transit_log.clear()
    global_subject_manager.next_subject_idx = 1
    global_subject_manager._save_persisted()
    return {"status": "REID_STATE_RESET_SUCCESSFUL"}
