# BORDER SENTINEL — Technical Architecture & Operational Defense Manual

**System Version**: `v1.0.0-PRODUCTION`  
**Security Classification**: 100% Air-Gapped / Strictly Offline Localhost  
**Primary Compute Platform**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) | CUDA 12.4 | Python 3.11  

---

## 1. Executive Summary & Problem Statement Alignment

BORDER SENTINEL is an autonomous, multi-camera tactical surveillance and defense grid engineered to address the critical challenges of border security, perimeter defense, and high-security facility monitoring. Traditional surveillance systems suffer from:
1. **Isolated Camera Silos**: Targets lose identity when transitioning between camera fields of view.
2. **Alert Fatigue**: False alarms from waving foliage, shadows, and domestic animals.
3. **Cloud Vulnerability**: Systems relying on external cloud APIs are vulnerable to jamming, network severance, and data exfiltration.
4. **Delayed Threat Correlation**: Human operators cannot mentally correlate multi-camera handoffs, loitering dwell times, and running anomalies in real time.

**BORDER SENTINEL** solves these challenges through a unified, 100% air-gapped system that runs entirely on local edge hardware with zero internet dependencies.

---

## 2. The 8-Layer Defense Architecture

```
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

## 3. Subsystem Specifications

### 3.1 Local AI Inference Engine
- **Weights Architecture**: Custom fine-tuned `YOLOv8l` (`models/registry/YOLO-L-v002/weights/best.pt`).
- **Classes (9 Master Defense Targets)**: `person`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `animal`, `backpack`, `bag`.
- **Hardware Acceleration**: NVIDIA GeForce RTX 4060 Laptop GPU via PyTorch CUDA 12.4.
- **Latency & Throughput**: Mean latency of **`29.46 ms`** (**`33.9 FPS`**) on 640x480 resolution.

### 3.2 Persistent Object Tracking (`ai/tracking/tracker.py`)
- **Tracker**: ByteTrack persistent Kalman filtering.
- **Trajectory Trails**: Real-time multi-point motion vector history.
- **Dwell Time Calculation**: Tracks stationary target presence in seconds to identify loitering.

### 3.3 Cross-Camera Re-ID & Journey Tracking (`ai/reid/`)
- **Appearance Extractor**: Composite **656-dimensional normalized visual signature**:
  - `512-dim` deep embedding from local YOLOv8l backbone via `m.embed()`.
  - `144-dim` 3-tier spatial HSV histogram (Head/Torso/Legs) for lighting invariance.
- **Spatio-Temporal Transit Verification**:
  - Defines physical travel corridors between `CAM_01`, `CAM_02`, and `CAM_03`.
  - Rejects impossible physical velocities ($\Delta t < 1.0\text{s}$, teleportation anomalies).
- **Global Subject Tracking**: Unifies local camera tracks into persistent entities (`[GLOBAL #01: PERSON]`) and records transit journeys across cameras.
- **Forensic Query Search**: `POST /api/reid/search` matches any query crop against CCTV history in $< 2\text{ ms}$.

### 3.4 Behavioral Anomaly AI (`ai/events/behavioral_engine.py`)
- **Sprinting Detection**: Displacements $> 32.0\text{ px/step}$ trigger flight anomaly alerts.
- **Unattended Baggage**: Stationary luggage isolated from all persons ($> 120\text{ px}$) for $> 7\text{s}$ triggers critical homeland defense alerts.
- **Doorway Tailgating**: Consecutive tripwire crossings within $\le 1.8\text{s}$ flag piggybacking breaches.
- **Crowd Surge**: Spatial clustering of $> 3$ persons within $140\text{ px}$ flags congestion.

### 3.5 Compound Threat Risk & DEFCON Scoring (`ai/events/incident_manager.py`)
- **Threat Index (0–100)**: Synthesizes target class lethality, zone severity, motion anomalies, and multi-camera transit history.
- **DEFCON Levels**:
  - **DEFCON 1 (CRITICAL, $\ge 75$)**: Armed response / immediate tactical escalation.
  - **DEFCON 2 (ELEVATED, $45 - 74$)**: Heightened perimeter alert / intercept patrol dispatched.
  - **DEFCON 3 (NORMAL, $< 45$)**: Routine surveillance monitoring.

---

## 4. Operational Launch & Evaluator Guide

### 4.1 One-Click Launch
To start the entire air-gapped system on Windows:
```cmd
run_sentinel.bat
```
or via PowerShell:
```powershell
.\run_sentinel.ps1
```
The script performs pre-flight GPU and weights verification, cleans port 8000, launches the backend, and opens `http://127.0.0.1:8000/`.

To cleanly terminate:
```cmd
stop_sentinel.bat
```

### 4.2 Automated Hardware Benchmark Suite
```powershell
.\.venv\Scripts\python.exe tools/benchmark_suite.py
```
Outputs `data/benchmark_report.json` and `docs/BENCHMARK_REPORT.md` confirming GPU throughput and memory allocation.

### 4.3 Deterministic Evaluator Demo Injector
```powershell
.\.venv\Scripts\python.exe tools/demo_scenario_injector.py --auto
```
Runs all 6 defense scenarios sequentially for judge evaluation without requiring live actors.

---

## 5. Decoupled Production Integration

For the external production UI currently under development:
- **WebSocket Telemetry**: `ws://127.0.0.1:8000/ws/telemetry` (10 Hz feed with live cameras, tracks, global Re-ID subjects, DEFCON condition, and active incidents).
- **Full API Documentation**: [`docs/EXTERNAL_UI_INTEGRATION.md`](file:///c:/Users/Abhinay%20Kandrika/OneDrive/Desktop/sih%20hackathon/docs/EXTERNAL_UI_INTEGRATION.md).
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`.
