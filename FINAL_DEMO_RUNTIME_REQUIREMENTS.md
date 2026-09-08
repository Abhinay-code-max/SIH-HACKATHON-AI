# BORDER SENTINEL — FINAL DEMO RUNTIME REQUIREMENTS

## 1. Overview & Operational Profile
BORDER SENTINEL is an air-gapped, 100% offline tactical edge surveillance and perimeter defense grid.
The system integrates multi-camera CCTV tracking (ByteTrack), YOLO-based deep neural inference, behavioral motion anomaly evaluation, virtual geofencing and tripwires, compound tactical incident aggregation, dynamic DEFCON threat readiness calculation, and an offline GIS SVG tactical operations dashboard.

---

## 2. Hardware and Environment Requirements

### Hardware Specifications
- **Operating System:** Windows 10 / 11 (64-bit) or Linux (Ubuntu 22.04 LTS+)
- **GPU (Recommended):** NVIDIA GeForce RTX 3060 / 4060 or higher (CUDA 12.1+ / 12.4+ supported)
- **VRAM:** 4 GB minimum (single shared YOLO instance architecture across all concurrent feeds)
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** ~15 GB free disk space (models, local datasets, and synthetic video streams)
- **Display:** 1920x1080 resolution recommended for tactical dashboard operations

### Software Stack
- **Python:** 3.10 – 3.12 (managed via local virtual environment `.\.venv\`)
- **PyTorch:** 2.1+ with CUDA support
- **OpenCV:** 4.8+ (headless / GUI supported)
- **Ultralytics:** YOLOv8 engine
- **FastAPI / Uvicorn:** Local asynchronous API & single-page application gateway

---

## 3. Asset Classification Matrix (Git-Tracked vs Local-Runtime-Asset)

| Asset Path | Type / Category | Status | Storage / Git Policy | Purpose / Description |
|---|---|---|---|---|
| `ai/` | Source Code | GIT-TRACKED | Tracked | Detection, tracking, behavioral analysis, Re-ID engine |
| `backend/` | Source Code | GIT-TRACKED | Tracked | FastAPI routes, contracts, services, camera stream manager |
| `frontend/` | UI / Web Assets | GIT-TRACKED | Tracked | Standalone single-page tactical operations dashboard (`index.html`) |
| `mapping/` | GIS & GeoJSON | GIT-TRACKED | Tracked | Zero-tile vector geojson layers (`cameras.geojson`, `zones.geojson`, etc.) |
| `config/` | System Configuration| GIT-TRACKED | Tracked | Rules, detection settings, camera and tripwire coordinates |
| `training_lab/` | Code / Experiment Lab| GIT-TRACKED | Tracked | Automation scripts, evaluators, reporting engines, experiment runs |
| `run_sentinel.bat` / `.ps1` | Launcher Scripts | GIT-TRACKED | Tracked | Demo grid launcher with automated preflight validation |
| `training_lab/runs/U2_yolov8m_640_9class_v002/weights/best.pt` | Protected Weights (U2) | LOCAL-RUNTIME-ASSET | Ignored by Git | Active 9-class primary detector (`cd8e50a9a84aef12...`) |
| `training_lab/runs/U2_yolov8m_640_9class_v002/weights/last.pt` | Protected Weights (U2) | LOCAL-RUNTIME-ASSET | Ignored by Git | U2 final training checkpoint (`71f368ed1687392a...`) |
| `training_lab/runs/U1_yolov8m_640_9class_unified/weights/best.pt` | Protected Weights (U1) | LOCAL-RUNTIME-ASSET | Ignored by Git | U1 baseline candidate checkpoint (`150d607f2287adff...`) |
| `training_lab/runs/D1_yolov8m_640_2class_full/weights/best.pt` | Protected Weights (D1) | LOCAL-RUNTIME-ASSET | Ignored by Git | D1 detector baseline (`005b706408d49f52...`) |
| `training_lab/videos/CAM_01_gateway.mp4` | Synthesized Video Stream | LOCAL-RUNTIME-ASSET | Generated on-demand | Gateway sector video stream |
| `training_lab/videos/CAM_02_perimeter.mp4` | Synthesized Video Stream | LOCAL-RUNTIME-ASSET | Generated on-demand | Perimeter command video stream |
| `training_lab/videos/CAM_03_checkpoint.mp4` | Synthesized Video Stream | LOCAL-RUNTIME-ASSET | Generated on-demand | Security checkpoint video stream |
| `training_lab/videos/CAM_04_north_fence.mp4` | Synthesized Video Stream | LOCAL-RUNTIME-ASSET | Generated on-demand | North fence boundary crossing video stream |
| `training_lab/videos/CAM_05_logistics.mp4` | Synthesized Video Stream | LOCAL-RUNTIME-ASSET | Generated on-demand | Logistics road crossing video stream |
| `training_lab/datasets/` | Training Datasets | LOCAL-RUNTIME-ASSET | Ignored by Git | Air-gapped training data partitions |
| `.venv/` | Virtual Environment | LOCAL-RUNTIME-ASSET | Ignored by Git | Python 3.12 packages and CUDA runtime wheels |

---

## 4. Multi-Camera Grid Topology (`CAM_01` to `CAM_05`)

| Camera ID | Display Name | Video Source File / Device | Sector / Geofence | Stream Target FPS |
|---|---|---|---|---|
| `CAM_01` | Main Demonstration CCTV | Hardware Webcam (device index 0) with automatic video fallback | ZONE_01 (Pedestrian Plaza) | 25-30 FPS |
| `CAM_02` | Gate 1 Vehicle Entry | `data/sample-videos/sample_surveillance.mp4` | ZONE_02 (Gate Checkpoint) | 25-30 FPS |
| `CAM_03` | Perimeter Command CCTV | `data/sample-videos/annotated_surveillance.mp4` | ZONE_03 (Critical Infrastructure) | 25 FPS |
| `CAM_04` | North Fence Sector | `training_lab/videos/CAM_04_north_fence.mp4` | ZONE_03 (Perimeter Fence) | 25 FPS |
| `CAM_05` | Logistics Road Crossing | `training_lab/videos/CAM_05_logistics.mp4` | ZONE_02 (Logistics Corridors) | 25 FPS |

---

## 5. End-to-End Pipeline Architecture

```text
[Camera Stream Source]
  ├── CAM_01 (Webcam 0 / Gateway Video)
  ├── CAM_02 (Gate 1 Video)
  ├── CAM_03 (Perimeter Video)
  ├── CAM_04 (North Fence Video)
  └── CAM_05 (Logistics Crossing Video)
        │
        ▼
[Shared YOLO-U-v002 9-Class Detector]
  (Classes: 0: person, 1: car, 2: truck, 3: bus, 4: motorcycle, 5: bicycle, 6: animal, 7: backpack, 8: bag)
  (Confidence confirmation, FP16 CUDA execution, shared VRAM footprint)
        │
        ▼
[Multi-Object Tracker (ByteTrack)]
  (Persistent local track IDs, velocity smoothing, dwell time tracking)
        │
        ▼
[Behavioral Anomaly & Spatial Engine]
  ├── Geofence Polygon Intrusion (ZONE_01..03)
  ├── Virtual Boundary Tripwire Crossing
  ├── Sprinting & Rapid Acceleration Detection
  ├── Unattended Baggage / Object Isolation
  ├── Anti-Piggybacking / Tailgating
  └── Vehicle Stoppage & Loitering Analysis
        │
        ▼
[Threat Scoring & DEFCON State Engine]
  (Dynamic 0-100 composite tactical risk index)
  ├── 0  - 44: DEFCON 3 (NORMAL / Green)
  ├── 45 - 74: DEFCON 2 (ELEVATED / Amber)
  └── 75 - 100: DEFCON 1 (CRITICAL / Red)
        │
        ▼
[Compound Incident Dossier Dispatch]
  (INC_xxxx aggregation, forensic image crops, webhook broadcast, operator audit)
        │
        ▼
[FastAPI Asynchronous Gateway & Single-Page Dashboard]
  ├── Multi-cam MJPEG feeds (`/api/stream/{cam_id}`)
  ├── System DEFCON state (`/api/incidents/defcon`)
  ├── Real-time alerts & forensic dossiers (`/api/evidence`)
  ├── Zero-tile offline vector GIS map (`/api/map`)
  └── Cross-camera Re-ID transit timeline (`/api/reid/subjects`)
```

---

## 6. How to Launch the Demo

### Quick Start (Standard Demonstration)
Run the root batch script:
```bat
run_sentinel.bat
```
Or via PowerShell:
```powershell
.\run_sentinel.ps1
```

### Preflight Actions Executed Automatically:
1. Validates local Python virtual environment (`.\.venv\Scripts\python.exe`).
2. Checks model weights presence, verifying `U2_yolov8m_640_9class_v002` first.
3. Checks CUDA GPU device availability (`NVIDIA GeForce RTX 4060 Laptop GPU`).
4. Verifies sample video streams and GeoJSON map layers.
5. Launches FastAPI server on `http://127.0.0.1:8000`.
6. Opens the tactical command browser dashboard automatically.

---

## 7. Air-Gapped Offline Verification
- **External Network Dependency:** 0% (zero calls to internet, CDNs, or tile servers).
- **Map Rendering:** 100% vector SVG generated from local `cameras.geojson`, `zones.geojson`, and `roads.geojson`.
- **UI Assets:** Pure self-contained vanilla HTML5/CSS3/JavaScript embedded in `frontend/index.html`.
