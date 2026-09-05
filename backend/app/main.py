"""
Main FastAPI Application Entrypoint.
Serves Multi-Camera Live Intelligence, Command Dashboard, Evidence Viewer, and Annotation Studio.
Complies with Rule 2: Absolute Offline Operation (Zero internet, strictly localhost).
"""

from pathlib import Path
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.api.routes import router as api_router
from backend.app.api.annotation_routes import router as annotation_router
from backend.app.api.evidence_routes import router as evidence_router
from backend.app.api.external_ui_routes import router as external_ui_router

app = FastAPI(
    title="BORDER SENTINEL — Autonomous Defense & Intelligence Grid",
    description="Local edge system for multi-camera AI tracking, geofence tripwires, evidence collection, and external UI integration.",
    version="1.0.0",
)

# Enable CORS for localhost access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(annotation_router)
app.include_router(evidence_router)
app.include_router(external_ui_router)

# Mount evidence snapshots for operator review
EVIDENCE_DIR = ROOT_DIR / "data" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")

# Mount extracted frames for local annotation studio
EXTRACTED_FRAMES_DIR = ROOT_DIR / "dataset" / "extracted_frames"
if EXTRACTED_FRAMES_DIR.is_dir():
    app.mount("/dataset/extracted_frames", StaticFiles(directory=str(EXTRACTED_FRAMES_DIR)), name="extracted_frames")

FRONTEND_DIR = ROOT_DIR / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"
ANNOTATE_FILE = FRONTEND_DIR / "annotate.html"


@app.get("/")
def serve_dashboard():
    """Serves the BORDER SENTINEL multi-camera command dashboard."""
    if INDEX_FILE.is_file():
        return FileResponse(INDEX_FILE)
    return JSONResponse({"system": "BORDER SENTINEL", "status": "ONLINE_LOCAL"})


@app.get("/annotate")
def serve_annotation_studio():
    """Serves the local human-in-the-loop annotation studio."""
    if ANNOTATE_FILE.is_file():
        return FileResponse(ANNOTATE_FILE)
    return JSONResponse({"error": "annotate.html not found"})


@app.get("/api/info")
def get_system_info():
    """Returns local system overview."""
    return {
        "system": "BORDER SENTINEL Tactical Defense Grid",
        "version": "1.0.0",
        "network_mode": "100% AIR-GAPPED OFFLINE",
        "features": [
            "Multi-Camera Concurrency",
            "ByteTrack Persistent Object Tracking",
            "Geofence Polygon & Virtual Tripwire Intelligence",
            "Forensic Evidence Snapshotting & Dossiers",
            "Operator Dispatch (Acknowledge, Dismiss, Escalate)",
            "Zero-Tile Vector GIS Offline Map",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
