# Offline Dependency Audit — Prototype v0.1

**Audit Date**: September 4, 2026  
**System Architecture**: Local Edge Surveillance & Defense Grid  
**Hardware Profile**: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.4), 8 GB VRAM, CPython 3.11  
**Runtime Requirement**: Zero Internet Connection (Air-Gapped)

---

## 1. Runtime Dependency Audit Table

| Component | Dependency / Asset | Purpose | Local File Location | Runtime Internet Required? |
| :--- | :--- | :--- | :--- | :--- |
| **AI Inference** | PyTorch (`torch 2.6.0+cu124`) | CUDA accelerated tensor operations | `.venv/Lib/site-packages/torch/` | **NO** |
| **AI Inference** | TorchVision (`torchvision 0.21.0+cu124`) | Computer vision primitives | `.venv/Lib/site-packages/torchvision/` | **NO** |
| **AI Inference** | Ultralytics (`ultralytics 8.4.138`) | YOLO engine & bounding box parsing | `.venv/Lib/site-packages/ultralytics/` | **NO** |
| **AI Inference** | OpenCV (`opencv-python 5.0.0`) | Video stream capture & JPEG encoding | `.venv/Lib/site-packages/cv2/` | **NO** |
| **AI Weights** | YOLOv8 Large (`yolov8l.pt`, 83.7 MB) | Object detection weights (80 classes) | `ai/models/yolov8l.pt` | **NO** |
| **Event Engine** | Pydantic (`pydantic 2.13.5`) | Typed event validation & contracts | `.venv/Lib/site-packages/pydantic/` | **NO** |
| **Event Store** | Local JSON Event Store | Persistence for live security alerts | `data/sample-events/live_events.json` | **NO** |
| **Mapping (GIS)**| GeoJSON Zones | Polygon vectors for security zones | `mapping/geojson/zones.geojson` | **NO** |
| **Mapping (GIS)**| GeoJSON Cameras | Point coordinates for registered CCTV | `mapping/geojson/cameras.geojson` | **NO** |
| **Mapping (GIS)**| GeoJSON Roads | Vector geometry for road corridors | `mapping/geojson/roads.geojson` | **NO** |
| **Mapping Renderer** | Native SVG Vector Renderer | Zero-tile client-side map rendering | Embedded in `frontend/index.html` | **NO** |
| **Backend API** | FastAPI (`fastapi 0.135.x`) | Local HTTP/REST & MJPEG stream server| `.venv/Lib/site-packages/fastapi/` | **NO** |
| **Backend Server**| Uvicorn (`uvicorn 0.41.x`) | ASGI server running on `127.0.0.1:8000` | `.venv/Lib/site-packages/uvicorn/` | **NO** |
| **Frontend UI** | HTML5 / Vanilla JS / Native CSS | Tactical dark-mode dashboard | `frontend/index.html` | **NO** |
| **Frontend Assets**| Inline SVG Icons | UI icons (cameras, alerts, badges) | Embedded in `frontend/index.html` | **NO** |
| **Frontend Fonts** | System UI Font Stack | Typography (`-apple-system, Segoe UI...`)| Built into Host Operating System | **NO** |
| **Media Input** | Local Camera / Sample Videos | Input video streams (webcam, MP4) | Hardware index `0` / `data/sample-videos/`| **NO** |

---

## 2. Cloud Fallback Verification
* **Cloud AI Fallbacks**: **NONE**. All inference runs strictly on the local RTX 4060 GPU (or CPU fallback). Automatic weight downloading at runtime is hard-blocked with custom `FileNotFoundError` handlers.
* **Map Tile CDNs**: **NONE**. No requests to Google Maps, Mapbox, OpenStreetMap tile servers, or Leaflet CDN. Vector layers render directly from local GeoJSON.
* **Frontend CDNs**: **NONE**. Zero `<script src="https://cdn...">` or `<link href="https://fonts.googleapis.com...">` tags.

---

## 3. Conclusion & Certification
The application complies 100% with **Rule 2 (Absolute Offline Operation)**. All runtime dependencies are bundled locally. The prototype runs in a complete air-gapped environment with no network access.
