# BORDER SENTINEL — Autonomous Defense & Tactical Intelligence Grid

> **SIH Hackathon Production Release v1.0.0**  
> **100% Air-Gapped & Offline Edge Architecture** | **Zero Internet, Cloud, or External API Dependencies**  
> **Hardware Acceleration**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) | PyTorch CUDA 12.4

---

## 1. System Overview

**BORDER SENTINEL** is an integrated tactical surveillance, perimeter defense, and automated threat intelligence grid. It processes concurrent CCTV camera feeds in real-time, executes persistent object tracking with motion vectors, correlates targets across camera handoffs using **deep visual appearance Re-Identification (Re-ID)**, detects complex **behavioral anomalies** (sprinting, abandoned luggage, doorway tailgating), and computes an automated **0–100 DEFCON threat risk score** with forensic evidence archiving and multi-channel dispatch.

```text
                      ┌────────────────────────────────────────┐
                      │          MULTI-CAMERA INGESTION        │
                      │  CAM_01 (Webcam) | CAM_02 (Gate 1 MP4) │
                      │       CAM_03 (Perimeter Command MP4)   │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    LOCAL GPU INFERENCE ACCELERATOR     │
                      │  YOLO-L-v002 Fine-Tuned (RTX 4060 GPU) │
                      │     99.5% mAP50 | 29.5ms (33.9 FPS)    │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    PERSISTENT TRACKING & TRAJECTORIES  │
                      │   ByteTrack Motion Vectors & Dwell Hub │
                      └───────────────────┬────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     │                                         │
                     ▼                                         ▼
   ┌───────────────────────────────────┐     ┌───────────────────────────────────┐
   │    CROSS-CAMERA RE-ID ENGINE      │     │     BEHAVIORAL ANOMALY AI         │
   │  • 512-dim YOLO Backbone Embed    │     │  • Sprinting / Rapid Flight (>32px)│
   │  • 144-dim Spatial HSV Histogram  │     │  • Unattended Baggage (>7s, >120px)│
   │  • Spatio-Temporal Corridors      │     │  • Tailgating / Anti-Piggybacking  │
   │  • Global Subject Journey Log     │     │  • Crowd Surge / Density Cluster  │
   └─────────────────┬─────────────────┘     └─────────────────┬─────────────────┘
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │   COMPOUND RISK SCORING & DEFCON HUB   │
                      │  0–100 Threat Index -> DEFCON 1/2/3    │
                      │  Forensic Evidence Snapshotting Engine │
                      └───────────────────┬────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     │                                         │
                     ▼                                         ▼
   ┌───────────────────────────────────┐     ┌───────────────────────────────────┐
   │    OFFLINE ZERO-TILE VECTOR GIS   │     │    DECOUPLED PRODUCTION GATEWAY   │
   │  • Pure SVG Local Map (Zones,     │     │  • 10 Hz WebSocket (/ws/telemetry)│
   │    Roads, Live Camera Pins)       │     │  • REST Endpoints (/api/incidents)│
   │  • Re-ID Inter-Camera Vectors     │     │  • Re-ID Search (/api/reid/search)│
   │  • Pulsing Threat Alert Rings     │     │  • Automated Webhook Dispatch     │
   └───────────────────────────────────┘     └───────────────────────────────────┘
```

---

## 2. Certified Performance Benchmarks (NVIDIA RTX 4060 GPU)

Certified on `2026-09-05` via `tools/benchmark_suite.py`:

| Performance Metric | Certified Result | Target Specification | Status |
| :--- | :--- | :--- | :--- |
| **YOLO-L Inference Latency (Mean)** | **`29.46 ms`** | `< 40 ms` | **PASS (EXCELLENT)** |
| **YOLO-L Inference Latency (P95)** | **`38.13 ms`** | `< 50 ms` | **PASS** |
| **Peak Detection Throughput** | **`33.9 FPS`** | `> 25 FPS` | **PASS (REAL-TIME)** |
| **Re-ID 656-dim Appearance Extractor** | **`47.77 ms`** | `< 60 ms` | **PASS** |
| **Cross-Camera Candidate Matching** | **`0.290 ms`** | `< 5 ms` | **PASS (INSTANTANEOUS)** |
| **Cold Start Weight Initialization** | **`0.67 s`** | `< 5 s` | **PASS** |
| **Active Dedicated VRAM Footprint** | **`340.6 MB`** | `< 2000 MB` | **PASS (< 5% of 8 GB)** |

---

## 3. Core Defense Subsystems

1. **Multi-Camera Concurrent Ingestion**:
   - Manages simultaneous feeds (`CAM_01` DirectShow webcam, `CAM_02` Gate 1 Checkpoint, `CAM_03` Perimeter Command).
   - Standardized 640x480 resolution with frame-drop tolerance and auto-failover.
2. **Fine-Tuned YOLO Large AI Detection**:
   - `YOLO-L-v002` trained with zero train/val temporal leakage and 15 hard negatives.
   - **99.5% mAP50**, **98.6% mAP50-95**, **95.9% Precision/Recall** across 9 defense classes.
3. **ByteTrack Persistent Tracking**:
   - Trajectory motion vectors, direction estimation, and loitering dwell timers.
4. **Cross-Camera Re-ID & Journey Association**:
   - Normalized 656-dim visual signature (512-dim deep backbone + 144-dim spatial HSV color histogram).
   - Spatio-temporal transit corridor matrix enforcing physical speed limits (rejection of impossible teleportation).
   - Persistent Global Subject IDs (`[GLOBAL #01: SUBJ_ALPHA]`) with multi-camera transit trails.
5. **Behavioral Anomaly AI**:
   - Sprinting/flight displacement ($> 32\text{ px/step}$).
   - Unattended stationary baggage isolated from persons ($> 120\text{ px}$, $> 7\text{s}$).
   - Doorway tailgating / anti-piggybacking ($\Delta t \le 1.8\text{s}$).
   - Crowd surge density clustering ($> 3$ persons within $140\text{ px}$).
6. **Compound Threat Scoring & DEFCON Levels**:
   - Quantified 0–100 threat risk index mapping to `DEFCON 1 (CRITICAL)`, `DEFCON 2 (ELEVATED)`, and `DEFCON 3 (NORMAL)`.
   - Printable air-gapped HTML forensic incident reports (`/api/incidents/{id}/report`).
7. **Offline Zero-Tile Vector GIS Tactical Map**:
   - 100% self-contained pure SVG local coordinate mapping engine (zero Leaflet, zero Mapbox, zero remote tiles).
   - Real-time animated Re-ID transit vectors connecting camera sectors.
8. **Decoupled Gateway for External Production UI**:
   - Full REST API suite and 10 Hz WebSocket telemetry (`/ws/telemetry`) ready for any React/Vue/Electron production UI.

---

## 4. Quick Start & Execution

### One-Click Launch (Windows)
Double-click `run_sentinel.bat` or run in terminal:
```cmd
run_sentinel.bat
```
or via PowerShell:
```powershell
.\run_sentinel.ps1
```
The script performs pre-flight GPU checks, clears lingering port 8000 processes, launches the server, and opens your browser to:
* **Command Dashboard**: `http://127.0.0.1:8000/`
* **OpenAPI Documentation**: `http://127.0.0.1:8000/docs`
* **Human-in-the-Loop AI Studio**: `http://127.0.0.1:8000/annotate`

### Clean Shutdown
```cmd
stop_sentinel.bat
```

---

## 5. Evaluator & Judge Demo Commands

### Automated Grand Demonstration (All 6 Scenarios)
Execute the automated evaluation suite to run all defense scenarios sequentially:
```powershell
.\.venv\Scripts\python.exe tools/demo_scenario_injector.py --auto
```

### Interactive Scenario Menu
```powershell
.\.venv\Scripts\python.exe tools/demo_scenario_injector.py
```
Allows triggering individual scenarios on demand:
- `[1]` Restricted Zone Geofence Intrusion
- `[2]` Virtual Boundary Tripwire Crossing
- `[3]` Sprinting / Rapid Displacement Anomaly
- `[4]` Unattended Baggage Alarm
- `[5]` Doorway Tailgating / Anti-Piggybacking
- `[6]` Cross-Camera Re-ID Subject Transit (`CAM_01` $\to$ `CAM_02`)

### Run Hardware Benchmark Suite
```powershell
.\.venv\Scripts\python.exe tools/benchmark_suite.py
```

### Run Full Test Regression Suite
```powershell
.\.venv\Scripts\python.exe tests/test_behavioral_incidents.py
.\.venv\Scripts\python.exe tests/test_cross_camera_reid.py
.\.venv\Scripts\python.exe tests/test_e2e_pipeline.py
.\.venv\Scripts\python.exe tests/test_tracking_intelligence.py
```

---

## 6. Project Documentation Directory

- **Technical Architecture Manual**: [`docs/DEFENSE_SYSTEM_MANUAL.md`](docs/DEFENSE_SYSTEM_MANUAL.md)
- **External UI Integration Contract**: [`docs/EXTERNAL_UI_INTEGRATION.md`](docs/EXTERNAL_UI_INTEGRATION.md)
- **Hardware Benchmark Report**: [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md)
- **Model Registry Index**: [`models/registry/registry_index.json`](models/registry/registry_index.json)
