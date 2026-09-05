# BORDER SENTINEL — External UI Integration Contract & Data Gateway

> **Architectural Notice**: The built-in web frontend at `http://127.0.0.1:8000/` serves strictly as an internal test harness. All video feeds, structured tracking metadata, forensic dossiers, and geofence intelligence are completely decoupled via standard HTTP REST and WebSocket protocols for the production UI currently under development.

---

## 1. Network & Protocol Overview

- **Base URL**: `http://127.0.0.1:8000`
- **WebSocket URL**: `ws://127.0.0.1:8000`
- **CORS Policy**: Permissive (`*` for origins, headers, and methods) — ready for React/Vite (`localhost:5173`), Next.js (`localhost:3000`), Vue, Electron, or Flutter clients.
- **Interactive OpenAPI Explorer**: `http://127.0.0.1:8000/docs` (Swagger UI)
- **OpenAPI JSON Schema**: `http://127.0.0.1:8000/openapi.json`

---

## 2. Live Video Feeds

### Multi-Camera Video Streams (MJPEG)
Embeddable directly in standard HTML `<img>` elements or any canvas/video player in React/Electron:

| Endpoint | Description | Dimensions | Codec |
| :--- | :--- | :--- | :--- |
| `GET /api/cameras/stream/{camera_id}` | Live annotated video stream with ByteTrack trails, geofences, and persistent IDs | Standard 640x480 (4:3) | MJPEG |
| `GET /api/stream/{camera_id}` | Alias endpoint for live stream | Standard 640x480 (4:3) | MJPEG |
| `GET /api/cameras/{camera_id}/snapshot` | Instant single JPEG frame capture for thumbnail cards & preview grids | Standard 640x480 | JPEG |

#### React / JSX Embedding Example:
```jsx
// Real-time camera player component
function CameraPlayer({ cameraId, label }) {
  return (
    <div className="camera-card">
      <h3>{label}</h3>
      <img
        src={`http://127.0.0.1:8000/api/cameras/stream/${cameraId}`}
        alt={`${cameraId} Feed`}
        style={{ width: '100%', height: 'auto', borderRadius: '4px' }}
      />
    </div>
  );
}
```

---

## 3. Real-Time Telemetry & Tracking WebSocket

### `ws://127.0.0.1:8000/ws/telemetry`
High-frequency (10 Hz) continuous broadcast for external UI overlays, radar displays, and track visualization.

#### WebSocket Packet Payload Schema:
```json
{
  "timestamp": 1788598912.45,
  "cameras": [
    {
      "camera_id": "CAM_01",
      "name": "Main Demonstration CCTV / Webcam",
      "status": "ONLINE",
      "fps": 28.5,
      "active_tracks": 1
    },
    {
      "camera_id": "CAM_02",
      "name": "Gate 1 Vehicle Entry",
      "status": "ONLINE",
      "fps": 25.0,
      "active_tracks": 3
    }
  ],
  "tracks": {
    "CAM_01": [
      {
        "track_id": 5,
        "class_name": "person",
        "confidence": 0.942,
        "bbox": [120.5, 95.0, 480.0, 470.2],
        "normalized_center": [0.4691, 0.9583],
        "center": [300.25, 460.2],
        "dwell_seconds": 32.4,
        "velocity_vector": [0.0, 0.0],
        "trajectory": [[300.2, 460.0], [300.1, 460.1]],
        "global_subject_id": "SUBJ_0001",
        "global_display_name": "[GLOBAL #01: PERSON]"
      }
    ]
  },
  "global_subjects": [
    {
      "subject_id": "SUBJ_0001",
      "display_name": "[GLOBAL #01: PERSON]",
      "class_name": "person",
      "last_camera_id": "CAM_01",
      "is_active": true,
      "representative_crop": "/reid_crops/SUBJ_0001_init_1788599100.jpg"
    }
  ],
  "recent_transits": [
    {
      "transit_id": "TR_0001",
      "subject_id": "SUBJ_0001",
      "from_camera": "CAM_01",
      "to_camera": "CAM_02",
      "transit_duration_sec": 14.2
    }
  ]
}
```

#### JavaScript / WebSocket Client Snippet:
```typescript
const socket = new WebSocket("ws://127.0.0.1:8000/ws/telemetry");

socket.onmessage = (event) => {
  const telemetry = JSON.parse(event.data);
  console.log("Live Telemetry:", telemetry.cameras);
  
  // Render custom tactical bounding boxes or radar markers
  const cam1Tracks = telemetry.tracks["CAM_01"] || [];
  cam1Tracks.forEach(track => {
    console.log(`Track #${track.track_id} (${track.class_name}) at dwell ${track.dwell_seconds}s`);
  });
};
```

---

## 4. REST API Endpoints for External UI

### 4.1 Camera Management & Metadata
- **`GET /api/cameras`**: List all registered cameras, spatial locations, and status.
- **`GET /api/cameras/telemetry`**: Poll live FPS, online status, and track counts.
- **`GET /api/cameras/{camera_id}/info`**: Full camera specification, active rules, geofence polygons, and active model.
- **`GET /api/cameras/{camera_id}/tracks`**: Current active tracked objects for specific camera.

### 4.2 Security Events & Forensic Dossiers
- **`GET /api/events`**: Filterable security events log (`?severity=CRITICAL&limit=50`).
- **`GET /api/evidence`**: All captured forensic dossiers (includes snapshot URLs, crop URLs, and timestamps).
- **`GET /api/evidence/{event_id}`**: Full dossier with trajectory JSON and target metadata.
- **`POST /api/evidence/{event_id}/action`**: Submit human operator decision:
  ```json
  {
    "action": "ACKNOWLEDGE", // or "DISMISS" or "ESCALATE"
    "operator_name": "Sector-Commander-01",
    "notes": "Verified authorized patrol personnel."
  }
  ```

### 4.3 Interactive Rule Configuration (Geofences & Tripwires)
- **`GET /api/rules`**: Returns current geofence polygons and tripwires.
- **`POST /api/rules/zones`**: Create or modify a polygon geofence from the external UI canvas:
  ```json
  {
    "zone_id": "ZONE_CUSTOM_01",
    "name": "Armory Perimeter",
    "camera_id": "CAM_01",
    "severity": "CRITICAL",
    "polygon": [[0.05, 0.1], [0.4, 0.1], [0.4, 0.6], [0.05, 0.6]],
    "restricted_classes": ["person", "car"],
    "alert_on_entry": true
  }
  ```
- **`POST /api/rules/tripwires`**: Create or modify a virtual tripwire:
  ```json
  {
    "wire_id": "WIRE_CUSTOM_01",
    "name": "Gate Line",
    "camera_id": "CAM_02",
    "severity": "HIGH",
    "line_start": [0.25, 0.1],
    "line_end": [0.25, 0.9],
    "crossing_direction": "ANY",
    "target_classes": ["truck", "bus"]
  }
  ```

### 4.4 AI Model Management
- **`GET /api/models`**: List trained versions (`YOLO-L-v001`, `YOLO-L-v002`), metrics (mAP50, precision, recall), and active model.
- **`POST /api/models/activate`**: Switch active inference model across all camera streams on the fly:
  ```json
  {
    "version": "models/registry/YOLO-L-v002/weights/best.pt"
  }
  ```

### 4.5 Cross-Camera Re-ID & Multi-Camera Subject Journey Tracking
- **`GET /api/reid/subjects`**: List all global subjects tracked across cameras (`?active_only=true&class_name=person`).
- **`GET /api/reid/subjects/{subject_id}`**: Full chronological journey dossier across cameras with sighting crops, bboxes, dwell times, and transit records.
- **`GET /api/reid/transits`**: Real-time log of subjects transitioning between different camera sectors (`?limit=25`).
- **`POST /api/reid/search`**: Upload an arbitrary target crop to perform visual similarity matching across all historical CCTV sightings (returns ranked matches with cosine similarity scores).
- **`POST /api/reid/reset`**: Reset Re-ID state and journey history for test demonstrations.

#### Global Subject Schema:
```json
{
  "subject_id": "SUBJ_0001",
  "display_name": "[GLOBAL #01: PERSON]",
  "class_name": "person",
  "first_seen": 1788599100.12,
  "last_seen": 1788599145.80,
  "first_camera_id": "CAM_01",
  "last_camera_id": "CAM_02",
  "is_active": true,
  "sightings_count": 28,
  "representative_crop": "/reid_crops/SUBJ_0001_init_1788599100.jpg",
  "transits": [
    {
      "transit_id": "TR_0001",
      "from_camera": "CAM_01",
      "to_camera": "CAM_02",
      "transit_duration_sec": 14.2,
      "similarity_score": 0.842,
      "verdict": "VALID_CORRIDOR: Concourse to Gate 1 Corridor"
    }
  ]
}
```

---

## 5. Offline & Security Guarantees

1. **Air-Gapped Operation**: The backend initiates zero outbound internet requests at runtime.
2. **Local Storage**: All snapshots, crops, logs, and dossiers reside in `./data/evidence/` and `./data/sample-events/`.
3. **Decoupled Architecture**: You can replace or terminate `frontend/index.html` at any time without impacting any backend AI, tracking, or streaming functionality.
