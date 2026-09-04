# Offline Surveillance & Defense Grid — Prototype v0.1

An autonomous, **100% offline edge surveillance and security intelligence system**.  
Operates in completely **air-gapped environments with ZERO internet connection** (no Wi-Fi, no Ethernet, no DNS, no external APIs, and no cloud fallbacks).

---

## 1. System Architecture

```text
Local Video / Webcam Stream (CAM_01..03)
                 ↓
Local GPU AI Inference (YOLOv8 Large @ ~42 FPS on RTX 4060)
                 ↓
Rule-Based Security Event Engine (Crowd Density, Vehicle Intrusion, Cooldowns)
                 ↓
Offline Vector GIS Mapping (GeoJSON Zones, Roads, Camera Pins, Pulsing Alerts)
                 ↓
Local FastAPI REST & MJPEG Streaming Server (http://127.0.0.1:8000)
                 ↓
Self-Contained Tactical Dashboard (Zero CDNs, System Fonts, Inline SVGs)
```

---

## 2. Hardware Profile & Performance Benchmark

* **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
* **Compute Framework**: CUDA 12.4 with PyTorch 2.6
* **AI Model**: `yolov8l.pt` (43.7M parameters, 83.7 MB weights stored in `ai/models/`)
* **Inference Speed**: ~24 ms average latency (**~41.6 FPS**)
* **VRAM Footprint**: ~1.4 GB (well within 8 GB capacity)
* **Tested Network State**: **100% Offline / Air-Gapped** (Ping 8.8.8.8 unreachable)

---

## 3. Quickstart — Running Completely Offline

Once installed, **no internet is ever required**.

### Option A: Double-Click Launcher
Double-click `run_prototype.bat` in the project root directory.  
It will activate `.venv`, launch your browser, and host the dashboard on `http://127.0.0.1:8000`.

### Option B: Terminal Command
Open Command Prompt in this folder:
```cmd
call .venv\Scripts\activate.bat
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in Chrome, Edge, or Firefox.

---

## 4. Repository Structure

```text
sih hackathon/
├── run_prototype.bat            # One-click Windows launch script
├── PROJECT_RULES.md             # Core prompt/command-only & offline guidelines
├── README.md                    # System documentation and user manual
├── .gitignore                   # Ignores venvs, weights, caches, media
│
├── ai/
│   ├── models/
│   │   ├── yolov8l.pt           # Local YOLOv8 Large weights (83.7 MB)
│   │   └── download_weights.py  # Development-time weight downloader
│   ├── inference/
│   │   ├── loader.py            # Strict offline model loader (no auto-downloads)
│   │   ├── image_inference.py   # Standalone image inference script
│   │   └── video_inference.py   # Real-time video/webcam inference engine
│   └── events/
│       ├── engine.py            # Decoupled rule & security alert engine
│       ├── contracts.py         # Shared Pydantic data contracts
│       ├── test_contracts.py    # Schema verification script
│       └── run_event_pipeline.py# Event generation pipeline runner
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py        # REST endpoints & MJPEG video streaming
│   │   ├── models/
│   │   │   └── contracts.py     # Pydantic schema source of truth
│   │   ├── services/
│   │   │   ├── event_service.py # Local event storage & filtering
│   │   │   └── stream_service.py# Multipart MJPEG stream generator
│   │   └── main.py              # FastAPI app & static dashboard host
│   └── test_backend.py          # Automated API test suite
│
├── mapping/
│   ├── geojson/
│   │   ├── zones.geojson        # Security zone polygon vectors
│   │   ├── cameras.geojson      # CCTV coordinate points
│   │   └── roads.geojson        # Road corridor vector geometry
│   ├── processed/
│   │   ├── map_overlay.json     # Consolidated spatial payload
│   │   └── offline_map_preview.html # Self-contained offline map preview
│   └── manager.py               # Spatial mapping & GIS manager
│
├── frontend/
│   └── index.html               # Tactical dark-mode dashboard (Zero CDN/Internet)
│
├── data/
│   ├── sample-images/           # Deterministic test images & outputs
│   ├── sample-videos/           # Test surveillance videos & detections
│   └── sample-events/           # Mock events & live event stores
│
├── docs/
│   ├── PHASE1_PLAN.md           # Prototype architecture & roadmap document
│   └── DEPENDENCY_AUDIT.md      # Zero-cloud dependency audit
│
└── tests/
    └── test_e2e_pipeline.py     # Complete 5-stage automated integration test
```

---

## 5. Development Setup (Reproducing on a New Machine)

To prepare a new developer machine while connected to the internet:

```cmd
:: 1. Create Python 3.11 virtual environment
uv venv .venv --python 3.11
call .venv\Scripts\activate.bat

:: 2. Install PyTorch with CUDA acceleration
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

:: 3. Install core dependencies
uv pip install ultralytics opencv-python pydantic fastapi uvicorn

:: 4. Download and bundle model weights once
python ai/models/download_weights.py yolov8l

:: 5. Run end-to-end integration test
python tests/test_e2e_pipeline.py
```

After step 5, **disconnect the internet completely**; the prototype will run self-contained.

---

## 6. Verification Test Commands

| Test Stage | Command | Expected Output |
| :--- | :--- | :--- |
| **Model Loading** | `python ai/inference/loader.py yolov8l` | `Status: LOCAL MODEL LOAD SUCCESSFUL` |
| **Image Inference**| `python ai/inference/image_inference.py data/sample-images/traffic_sample.jpg yolov8l.pt 0.35` | `Status: IMAGE INFERENCE SUCCESSFUL` |
| **Webcam Stream** | `python ai/inference/video_inference.py 0 --model yolov8l.pt --show` | Live window, ~40 FPS, bounding boxes |
| **Event Contracts**| `python ai/events/test_contracts.py` | `Status: DETECTION SCHEMA VERIFICATION SUCCESSFUL` |
| **Event Pipeline** | `python ai/events/run_event_pipeline.py` | `Status: EVENT ENGINE VERIFICATION SUCCESSFUL` |
| **Offline Mapping**| `python mapping/manager.py` | `Status: OFFLINE MAPPING VERIFICATION SUCCESSFUL` |
| **Backend API** | `python backend/test_backend.py` | `Status: LOCAL BACKEND API VERIFICATION SUCCESSFUL` |
| **Full Pipeline** | `python tests/test_e2e_pipeline.py` | `Status: END-TO-END TEST SUCCESSFUL` |

---

## 7. Definition of Done Checklist (Phase 1)

- [x] Local video/image enters the pipeline deterministically.
- [x] Local YOLOv8 Large model produces real detections on RTX 4060 GPU (~42 FPS).
- [x] Raw detections are converted into structured `SecurityEvent` contracts.
- [x] Local FastAPI backend exposes health, cameras, events, and map data.
- [x] Tactical dashboard frontend consumes live data with ZERO remote CDNs.
- [x] Bundled vector GIS map renders zones, roads, and active markers offline.
- [x] Entire system executes seamlessly with Wi-Fi / Ethernet disconnected.
- [x] Dependency audit and reproduction documentation frozen.
