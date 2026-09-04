# Offline Surveillance & Defense Grid — Prototype v0.2

An autonomous, **100% offline edge surveillance, defense intelligence, and dataset training system**.  
Operates in completely **air-gapped environments with ZERO internet connection** (no Wi-Fi, no Ethernet, no DNS, no external APIs, and no cloud fallbacks).

---

## 1. System Architecture

```text
CCTV Video / Webcam Stream (CAM_01..03)
                 ↓
Intelligent Frame Extractor & Anti-Redundancy Sampler (1-2 FPS, Motion Filter)
                 ↓
Local GPU AI Pre-Annotation (YOLOv8 Large with 35-65% Uncertainty Flagging)
                 ↓
Local Human-in-the-Loop Web Verification Studio (http://127.0.0.1:8000/annotate)
                 ↓
Leakage-Protected Dataset Builder (Temporal Block Splitting -> Dataset v1)
                 ↓
YOLO Large Training Engine & Model Registry (YOLO-L-v001, v002...)
                 ↓
Multi-Scenario Evaluation Suite (Day vs Night, Distant Targets, False Alarms/Hr)
                 ↓
Local FastAPI REST & Real-Time MJPEG Streaming Server (http://127.0.0.1:8000)
                 ↓
Self-Contained Tactical Surveillance Dashboard (Zero CDNs, System Fonts, GIS Map)
```

---

## 2. Hardware Profile & Performance Benchmark

* **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
* **Compute Framework**: CUDA 12.4 with PyTorch 2.6
* **Active Base Model**: `yolov8l.pt` (43.7M parameters, 83.7 MB)
* **Registered Fine-Tuned Model**: `YOLO-L-v001` (`models/registry/YOLO-L-v001/weights/best.pt`, 87.6 MB)
  * **Validation mAP50**: **98.17%**
  * **Precision**: **99.32%** | **Recall**: **87.5%**
* **Inference Speed**: ~24 ms latency (**~41.6 FPS**) on live camera feeds
* **Tested Network State**: **100% Offline / Air-Gapped** (Ping 8.8.8.8 unreachable)

---

## 3. Master Operational Detection Classes

Configured in `config/classes.yaml`:
```yaml
0: person
1: car
2: truck
3: bus
4: motorcycle
5: bicycle
6: animal
7: backpack
8: bag
```

---

## 4. Web Endpoints (100% Localhost)

Start the local server:
```cmd
run_prototype.bat
```
or
```cmd
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

* **`http://127.0.0.1:8000/`** — **Tactical Surveillance Dashboard**:
  * Live webcam / CCTV stream with real-time YOLO bounding boxes
  * Offline vector GIS map (zones, roads, camera pins, pulsing threat markers)
  * Real-time security alerts log
* **`http://127.0.0.1:8000/annotate`** — **Human-in-the-Loop Studio**:
  * Visual Canvas bounding box editor (add, resize, delete, change class)
  * Active learning uncertainty review queue (flags 35–65% confidence predictions)
  * Batch ground-truth confirmation button
  * Hard-negative tagging for empty scenes
* **`http://127.0.0.1:8000/docs`** — Interactive OpenAPI Documentation

---

## 5. Dataset & Training Pipeline Commands

| Pipeline Stage | Command | Purpose |
| :--- | :--- | :--- |
| **Directory Setup** | `python dataset/tools/dataset_manager.py` | Initializes directory tree & master classes |
| **Smart Frame Extraction** | `python dataset/tools/extract_frames.py <video.mp4> --camera CAM_01 --fps 2.0` | Samples frames, discards static/redundant footage |
| **YOLO Auto-Annotation** | `python dataset/tools/pre_annotate.py dataset/extracted_frames/CAM_01` | GPU auto-labeling & uncertainty mining |
| **Human Verification** | Open `http://127.0.0.1:8000/annotate` or `python dataset/tools/verify_human_loop.py` | Human-confirmed ground-truth locking |
| **Dataset Release** | `python dataset/tools/build_dataset_v1.py --version v1` | Temporal block splitting without train/val leakage |
| **Failure Mining** | `python dataset/tools/mine_failures.py --dataset v1` | Mines false positives, tiny objects, night scenes |
| **YOLO Training** | `python dataset/tools/train_yolo.py --dataset v1 --preset prototype_verification` | Trains & registers versioned model (`YOLO-L-v001`) |
| **Scenario Benchmark**| `python dataset/tools/evaluate_scenarios.py --model YOLO-L-v001` | Evaluates Day/Night, Distant, and False Positives/Hr |

---

## 6. Model Version Registry

Tracked in `models/registry/registry_index.json`:

| Model Tag | Dataset | Epochs | mAP50 | mAP50-95 | Precision | Recall | Registered Weights |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`YOLO-L-v001`** | `v1` | 3 | **98.17%** | **98.17%** | **99.32%** | **87.50%** | `models/registry/YOLO-L-v001/weights/best.pt` |

---

## 7. Operational Scenario Diagnostic Benchmark (`YOLO-L-v001`)

From `models/registry/evaluation_report_YOLO-L-v001.json`:
* **Overall F1 Score**: **82.4%** (Recall: 100.0%, Precision: 70.0%)
* **Night / Low Light**: **100% Precision, 100% Recall**
* **Distant Objects (< 35px)**: **100% Precision, 100% Recall**
* **Air-Gap Verification**: Complete (Zero cloud calls, zero external CDNs)
