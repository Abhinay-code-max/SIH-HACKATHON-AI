"""
FastAPI Routes for Local System APIs.
Zero external cloud dependencies.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import torch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.models.contracts import (
    CameraAsset,
    CameraStatus,
    GeoLocation,
    SecurityEvent,
    SeverityLevel,
    SystemHealth,
)
from backend.app.services.event_service import event_service
from backend.app.services.stream_service import stream_service
from mapping.manager import OfflineMapManager

router = APIRouter(prefix="/api")

map_manager = OfflineMapManager(base_dir=ROOT_DIR / "mapping")


@router.get("/health", response_model=SystemHealth)
def get_system_health() -> SystemHealth:
    """Returns local system health and GPU status."""
    is_cuda = torch.cuda.is_available()
    vram_mb = 0.0
    if is_cuda:
        vram_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)

    return SystemHealth(
        status="healthy",
        offline_mode=True,
        device="cuda" if is_cuda else "cpu",
        device_name=torch.cuda.get_device_name(0) if is_cuda else "Host CPU",
        active_model="yolov8l.pt",
        vram_allocated_mb=vram_mb,
        cameras_active=3,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/cameras", response_model=List[Dict[str, Any]])
def list_cameras() -> List[Dict[str, Any]]:
    """Lists registered local camera assets."""
    cam_dict = map_manager.get_camera_locations()
    return list(cam_dict.values())


@router.get("/events")
def list_events(
    severity: Optional[SeverityLevel] = Query(None, description="Filter by severity level"),
    limit: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """Queries security events from local storage."""
    return event_service.get_events(severity=severity, limit=limit)


@router.get("/map")
def get_map_data() -> Dict[str, Any]:
    """Returns bundled offline GeoJSON layers, camera locations, and event markers."""
    return map_manager.build_integrated_map_payload()


from backend.app.services.camera_manager import camera_manager

@router.get("/stream/{camera_id}")
def stream_camera(camera_id: str):
    """
    Streams live annotated MJPEG frames with persistent tracking (ByteTrack),
    virtual fence geofences, tripwires, and trajectory trails.
    """
    return StreamingResponse(
        camera_manager.generate_live_mjpeg(camera_id=camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
