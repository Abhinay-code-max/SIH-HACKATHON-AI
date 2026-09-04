"""
Main FastAPI Application Entrypoint.
Serves both the Local API and the 100% Offline Frontend Dashboard.
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

app = FastAPI(
    title="Offline Security & Surveillance Grid",
    description="Local edge system API and UI for AI detections, offline maps, and surveillance streams.",
    version="0.1.0",
)

# Enable CORS for localhost frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

FRONTEND_DIR = ROOT_DIR / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"


@app.get("/")
def serve_dashboard():
    """Serves the self-contained offline surveillance dashboard."""
    if INDEX_FILE.is_file():
        return FileResponse(INDEX_FILE)
    return JSONResponse({
        "system": "Offline Surveillance & Security Prototype",
        "status": "ONLINE_LOCAL",
        "message": "Frontend index.html not found.",
        "api_docs": "/docs"
    })


@app.get("/api/info")
def get_system_info():
    """Returns local system overview."""
    return {
        "system": "Offline Surveillance & Security Prototype",
        "version": "0.1.0",
        "status": "ONLINE_LOCAL",
        "network_mode": "100% AIR-GAPPED OFFLINE (ZERO REMOTE APIS/CDNS)",
        "endpoints": {
            "dashboard": "/",
            "health": "/api/health",
            "cameras": "/api/cameras",
            "events": "/api/events",
            "map_data": "/api/map",
            "stream_cam01": "/api/stream/CAM_01",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
