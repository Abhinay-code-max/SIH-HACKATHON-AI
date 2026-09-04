# Offline Prototype — AI, Mapping & Local System Plan
**Prototype Planning Document — Phase 1**

## Purpose
Build the technical foundation for a completely local prototype: AI inference, video/image processing, offline mapping, local data flow, and the interface contract consumed by the frontend team. The first milestone is not maximum accuracy; it is a reliable local pipeline that demonstrates the idea end-to-end.

## Core Principle
Internet is optional for development but not required at runtime. The prototype should be capable of taking local camera/video input, running AI locally, generating structured detections/events, associating them with local coordinates or map regions, and exposing the results to the local UI.

---

## 1. Target Prototype Pipeline
```text
Local video/camera → preprocessing → local AI inference → event/detection engine → local storage → mapping layer → local API → offline frontend
```
* Input can initially be recorded surveillance footage or a local camera stream. Start with prerecorded data because it makes development deterministic.
* AI should initially produce a small set of clearly defined detection types relevant to the project. Do not train a huge custom model before proving the pipeline.
* The AI layer should return structured JSON-like events, not UI-specific objects.
* The mapping layer should work from locally stored geographic/map data and should not depend on online map tiles.
* The backend/API should run on localhost and act as the stable bridge between AI, map logic and frontend.

---

## 2. Divide the Work

| Area | Primary Responsibility | Phase 1 Deliverable |
| :--- | :--- | :--- |
| **AI / Computer Vision** | Model selection, inference, preprocessing | Local model runs on supplied video/images; detections have class, confidence, bounding box, timestamp |
| **Event / Logic Engine** | Rules and event generation | Turns raw detections into meaningful events/alerts with severity, timestamp and source |
| **Mapping / Geospatial** | Offline map and spatial association | Bundled local map data, camera/asset locations, markers and basic spatial overlays |
| **Backend / Integration** | Local API and contracts | Local endpoints connecting frontend ↔ AI ↔ event store ↔ mapping data |
| **Data / Testing** | Sample data and reproducibility | Small test videos/images, expected outputs, benchmark script and test cases |

---

## 3. AI Stack — Recommended Prototype Approach
* **Object detection**: Use a locally runnable YOLO-family detector or another permissively licensed model appropriate to the target classes and hardware.
* **Runtime**: Prefer an accelerated local runtime available on the target laptop/GPU. Keep a CPU fallback for development.
* **Video processing**: Use OpenCV/FFmpeg or the equivalent local media stack.
* **Model files**: Store/download model weights during setup; runtime inference must load them locally. Check model licensing before the final SIH product.
* **Prototype priority**: inference reliability → structured output → event logic → performance → accuracy tuning → custom training.

---

## 4. AI Implementation Steps
* **Step 1**: Define 3–6 prototype detection classes/events. Keep the list small.
* **Step 2**: Choose a lightweight model that can run comfortably on the team's available hardware.
* **Step 3**: Create a standalone inference script that accepts one image/video and writes detections to a local output format.
* **Step 4**: Add video-frame processing with configurable FPS/frame skipping.
* **Step 5**: Normalize output into a stable schema.
* **Step 6**: Add tracking only if needed for the demonstration; avoid complex tracking until detection is stable.
* **Step 7**: Build the event/rule layer separately from the model.
* **Step 8**: Add logging and performance metrics: FPS, inference latency, number of detections and dropped frames.

---

## 5. Stable Detection/Event Contract
Example conceptual event:
```json
{
  "event_id": "local-generated-id",
  "timestamp": "...",
  "source_id": "camera_01",
  "event_type": "object_detected",
  "class": "person",
  "confidence": 0.91,
  "bbox": [x1, y1, x2, y2],
  "severity": "medium",
  "location": {
    "lat": 17.0000,
    "lon": 78.0000
  },
  "metadata": {}
}
```
Keep this contract independent from React/UI components. The frontend team should be able to build against mock events using exactly the same structure.

---

## 6. Offline Mapping Strategy
Do not make an online map SDK the foundation of the prototype. For offline operation, use locally packaged map data. A practical architecture is a local map renderer consuming pre-downloaded OpenStreetMap-derived vector/raster data, with the exact renderer chosen according to the frontend platform.
* For the first prototype, use a limited geographic region rather than the entire country.
* Pre-package only the map area required for the demonstration.
* Store camera/asset coordinates locally and render them as markers.
* Represent zones/areas of interest as local polygons or GeoJSON.
* Represent events as markers/heat/alert overlays generated locally.
* If routing is required later, evaluate an offline routing engine rather than an online Directions API.

---

## 7. Mapping Implementation Steps
* **Step 1**: Choose the exact demonstration geographic area.
* **Step 2**: Obtain appropriately licensed OSM-derived map data for that area.
* **Step 3**: Convert/package it into the format supported by the selected offline renderer.
* **Step 4**: Create a local camera/asset registry with IDs and coordinates.
* **Step 5**: Add markers, status states and event overlays.
* **Step 6**: Connect event coordinates/asset IDs to the map.
* **Step 7**: Test with internet completely disabled.

---

## 8. Local Backend / Integration
* Use a lightweight local API service (for example FastAPI) to provide health, cameras, events, detections and map-data metadata.
* The API should never contain frontend rendering logic.
* The backend can initially read local JSON/SQLite files; a full database is not necessary unless the prototype requires persistence.
* Keep AI inference callable as a local module/service so it can later be moved to a separate process without changing the API contract.
* Add a health endpoint so the frontend can show whether AI/backend services are running.

---

## 9. Suggested Repository Structure
```text
prototype-root/
├── frontend/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── models/
│   │   ├── services/
│   │   └── main.py
├── ai/
│   ├── models/
│   ├── inference/
│   ├── tracking/
│   ├── events/
│   └── tests/
├── mapping/
│   ├── raw/
│   ├── processed/
│   └── geojson/
├── data/
│   ├── sample-videos/
│   ├── sample-images/
│   └── sample-events/
└── docs/
    └── README.md
```

---

## 10. Parallel Development Order
* **Day/Stage 1 — Contracts first**: Define event schema, camera schema, health status and map asset schema. Frontend builds mocks from these contracts.
* **Stage 2 — AI standalone**: AI developer proves local inference independently using a fixed test video.
* **Stage 3 — Map standalone**: Mapping developer proves local map rendering and camera markers.
* **Stage 4 — Backend**: Expose AI results and map/camera data through localhost endpoints.
* **Stage 5 — Integration**: Frontend switches from mock provider to local API provider without rewriting screens.
* **Stage 6 — Demonstration**: Run the complete pipeline on one machine with internet disabled.
* **Stage 7 — Freeze v0.1**: Tag the working prototype and document exactly how to reproduce it.

---

## 11. Git Rules for AI / Backend / Mapping
* Use the same repository and shared integration branch as the frontend.
* Do not commit secrets or cloud API keys.
* Do not commit huge datasets or model weights unless explicitly required; document where setup scripts obtain permitted model assets.
* Use separate feature branches for AI, backend, mapping and integration.
* Commit contracts/schema changes separately from implementation changes.
* Every integration change must include a reproducible local test command.

---

## 12. Definition of Done — Phase 1
* A local video/image can enter the pipeline.
* A local AI model produces real detections.
* Detections are converted into stable structured events.
* The local backend exposes those events.
* The frontend can consume them without internet.
* A bundled/local map can display at least the demonstration area.
* Camera/asset markers and event markers can be displayed.
* The complete demo works with network connectivity disabled.
* README documents setup, model installation, map-data preparation, run commands and troubleshooting.

---

## 13. What NOT to Build Yet
* Do not start custom model training before validating the pretrained-model pipeline.
* Do not build a nationwide offline map for the first milestone.
* Do not add cloud AI fallback if offline operation is a core requirement.
* Do not optimize for production-scale multi-camera deployment yet.
* Do not build advanced dashboards before the core detection → event → map workflow works.
* Do not tightly couple AI code to frontend components.

---

## 14. Phase 2 — Foundation Refinement
Once v0.1 works, refine rather than restart: improve model accuracy, add tracking and domain-specific rules, optimize GPU usage, improve offline map rendering, add persistence, strengthen API contracts, handle multiple camera streams, add proper configuration, test failure/recovery scenarios, and improve the UI based on the first demonstration.
