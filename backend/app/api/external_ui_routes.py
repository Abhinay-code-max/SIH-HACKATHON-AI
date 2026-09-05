"""
External UI Integration and Decoupled Data Gateway.
Provides clean, standardized REST APIs and real-time WebSockets
for the under-development production UI to consume all camera feeds,
active tracks, forensic evidence, geofences, and model states.
"""

import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.camera_manager import camera_manager
from ai.events.intelligence_engine import intelligence_engine

router = APIRouter()


class ZoneCreateRequest(BaseModel):
    zone_id: str
    name: str
    camera_id: str
    severity: str = "HIGH"
    polygon: List[List[float]]
    restricted_classes: List[str] = ["person", "car", "truck"]
    alert_on_entry: bool = True


class TripwireCreateRequest(BaseModel):
    wire_id: str
    name: str
    camera_id: str
    severity: str = "HIGH"
    line_start: List[float]
    line_end: List[float]
    crossing_direction: str = "ANY"
    target_classes: List[str] = ["person", "car", "truck"]


class ModelActivateRequest(BaseModel):
    version: str


# =====================================================================
# 1. CAMERA STREAMS AND SNAPSHOTS
# =====================================================================

@router.get("/api/cameras/stream/{camera_id}")
def stream_camera_external(camera_id: str):
    """Streams live MJPEG frames with tracking and overlays."""
    return StreamingResponse(
        camera_manager.generate_live_mjpeg(camera_id=camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/api/cameras/{camera_id}/snapshot")
def get_camera_snapshot(camera_id: str):
    """Returns the latest single JPEG snapshot for camera cards or thumbnails."""
    jpeg_bytes = camera_manager.get_latest_snapshot_bytes(camera_id)
    if not jpeg_bytes:
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} has not captured any frames yet")
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get("/api/cameras/{camera_id}/info")
def get_camera_info(camera_id: str):
    """Returns full camera metadata, active model, zones, and tripwires."""
    info = camera_manager.get_full_camera_info(camera_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    return info


@router.get("/api/cameras/{camera_id}/tracks")
def get_camera_active_tracks(camera_id: str) -> List[Dict[str, Any]]:
    """Returns real-time structured active tracks for a camera."""
    return camera_manager.get_latest_tracks(camera_id)


# =====================================================================
# 2. INTELLIGENCE RULES AND GEOFENCE MANAGEMENT
# =====================================================================

@router.get("/api/rules")
def get_all_rules():
    """Returns all active zones, tripwires, loitering, and crowd limits."""
    return intelligence_engine.rules


@router.post("/api/rules/zones")
def create_or_update_zone(zone: ZoneCreateRequest):
    """Allows external UI to add or modify polygon geofences dynamically."""
    rules = intelligence_engine.rules
    zones = rules.get("zones", [])
    existing_idx = next((i for i, z in enumerate(zones) if z.get("zone_id") == zone.zone_id), None)
    if existing_idx is not None:
        zones[existing_idx] = zone.model_dump()
    else:
        zones.append(zone.model_dump())
    rules["zones"] = zones
    return {"status": "ZONE_SAVED", "zone": zone.model_dump(), "total_zones": len(zones)}


@router.post("/api/rules/tripwires")
def create_or_update_tripwire(wire: TripwireCreateRequest):
    """Allows external UI to add or modify virtual tripwires dynamically."""
    rules = intelligence_engine.rules
    wires = rules.get("tripwires", [])
    existing_idx = next((i for i, w in enumerate(wires) if w.get("wire_id") == wire.wire_id), None)
    if existing_idx is not None:
        wires[existing_idx] = wire.model_dump()
    else:
        wires.append(wire.model_dump())
    rules["tripwires"] = wires
    return {"status": "TRIPWIRE_SAVED", "tripwire": wire.model_dump(), "total_tripwires": len(wires)}


# =====================================================================
# 3. AI MODEL REGISTRY AND SELECTION
# =====================================================================

@router.get("/api/models")
def get_model_registry():
    """Returns all trained models, registered weights, and evaluation metrics."""
    index_file = ROOT_DIR / "models" / "registry" / "registry_index.json"
    if not index_file.is_file():
        return {"models": [], "active_model": "yolov8l.pt"}
    with open(index_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@router.post("/api/models/activate")
def activate_model(req: ModelActivateRequest):
    """Dynamically switches the active inference model across all camera workers."""
    for cid, worker in camera_manager.cameras.items():
        worker.tracker.model_name = req.version
        worker.tracker.model = None
    return {"status": "MODEL_ACTIVATED", "active_model": req.version}


# =====================================================================
# 4. REAL-TIME WEBSOCKET TELEMETRY STREAM FOR EXTERNAL UI
# =====================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


ws_manager = ConnectionManager()


@router.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """
    Real-time WebSocket feed for external UIs.
    Pushes live camera status, active tracks, and telemetry at 10 Hz.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            packet = {
                "timestamp": time.time(),
                "cameras": camera_manager.get_camera_status(),
                "tracks": {
                    cid: camera_manager.get_latest_tracks(cid)
                    for cid in camera_manager.cameras.keys()
                },
            }
            await websocket.send_json(packet)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
