"""
Multi-Camera Stream Orchestrator & Live Tracking Hub.
Manages concurrent camera feeds (Webcam, Gate video, Perimeter video).
Executes persistent tracking, draws geofences & tripwires, and streams live MJPEG.
"""

from pathlib import Path
import sys
import time
from typing import Dict, Generator, List, Optional
import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.tracking.tracker import ObjectTracker
from ai.events.intelligence_engine import intelligence_engine
from mapping.manager import OfflineMapManager


class CameraStreamWorker:
    def __init__(self, camera_id: str, source: str | int, name: str):
        self.camera_id = camera_id
        self.source = source
        self.name = name
        self.tracker = ObjectTracker(model_name="yolov8l.pt")
        self.status = "ONLINE"
        self.fps = 0.0
        self.active_tracks_count = 0
        self.last_frame = None

    def draw_overlays(self, frame: np.ndarray, tracks: List[dict]) -> np.ndarray:
        """Draws virtual zone polygons, tripwires, and camera HUD."""
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # 1. Draw Restricted Zone Geofences from rules
        rules = intelligence_engine.rules
        for zone in rules.get("zones", []):
            if zone.get("camera_id") == self.camera_id:
                pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone["polygon"]], dtype=np.int32)
                # Draw semi-transparent fill
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 200))
                cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
                cv2.polylines(annotated, [pts], True, (0, 0, 255), 2)
                z_name = zone.get("name", "Restricted Zone")
                cv2.putText(annotated, f"[GEOFENCE] {z_name}", (pts[0][0] + 10, pts[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 2. Draw Virtual Tripwires
        for wire in rules.get("tripwires", []):
            if wire.get("camera_id") == self.camera_id:
                p1 = (int(wire["line_start"][0] * w), int(wire["line_start"][1] * h))
                p2 = (int(wire["line_end"][0] * w), int(wire["line_end"][1] * h))
                cv2.line(annotated, p1, p2, (0, 165, 255), 2)
                cv2.circle(annotated, p1, 4, (0, 165, 255), -1)
                cv2.circle(annotated, p2, 4, (0, 165, 255), -1)
                cv2.putText(annotated, f"[TRIPWIRE] {wire['name']}", (p1[0], p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # 3. Bottom HUD Overlay Bar
        cv2.rectangle(annotated, (10, h - 36), (w - 10, h - 8), (10, 15, 25), -1)
        cv2.rectangle(annotated, (10, h - 36), (w - 10, h - 8), (40, 55, 80), 1)
        hud = f"{self.camera_id} // {self.name} | FPS: {self.fps:.1f} | TRACKED TARGETS: {self.active_tracks_count}"
        cv2.putText(annotated, hud, (20, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        return annotated


class MultiCameraManager:
    def __init__(self):
        self.cameras: Dict[str, CameraStreamWorker] = {
            "CAM_01": CameraStreamWorker("CAM_01", 0, "Main Demonstration CCTV / Webcam"),
            "CAM_02": CameraStreamWorker("CAM_02", str(ROOT_DIR / "data" / "sample-videos" / "sample_surveillance.mp4"), "Gate 1 Vehicle Entry"),
            "CAM_03": CameraStreamWorker("CAM_03", str(ROOT_DIR / "data" / "sample-videos" / "annotated_surveillance.mp4"), "Perimeter Command CCTV"),
        }

    def get_camera_status(self) -> List[dict]:
        return [
            {
                "camera_id": cid,
                "name": c.name,
                "status": c.status,
                "fps": round(c.fps, 1),
                "active_tracks": c.active_tracks_count,
            }
            for cid, c in self.cameras.items()
        ]

    def generate_live_mjpeg(self, camera_id: str) -> Generator[bytes, None, None]:
        """Streams live tracking feed with virtual fences and persistent IDs."""
        if camera_id not in self.cameras:
            camera_id = "CAM_01"

        worker = self.cameras[camera_id]
        src = worker.source

        cap = cv2.VideoCapture(src)
        if not cap.isOpened() and src == 0:
            # Fallback to sample video if webcam unavailable
            cap = cv2.VideoCapture(str(ROOT_DIR / "data" / "sample-videos" / "sample_surveillance.mp4"))

        target_fps = 25
        frame_interval = 1.0 / target_fps

        try:
            while cap.isOpened():
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    if not isinstance(src, int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                # 1. Update Object Tracking (confidence raised to 0.48 to reject false alarms)
                tracks, tracked_frame = worker.tracker.update(frame, conf_threshold=0.48)
                worker.active_tracks_count = len(tracks)

                # 2. Evaluate Intelligence Rules (Geofences, Tripwires, Loitering)
                intelligence_engine.evaluate_tracks(camera_id, tracks, frame)

                # 3. Draw Virtual Boundaries & HUD
                final_frame = worker.draw_overlays(tracked_frame, tracks)

                # Measure FPS
                elapsed = time.perf_counter() - t0
                worker.fps = 1.0 / elapsed if elapsed > 0 else target_fps

                # Encode JPEG
                _, jpeg_bytes = cv2.imencode(".jpg", final_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes.tobytes() + b"\r\n"
                )

                sleep_time = max(0.0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            cap.release()


camera_manager = MultiCameraManager()
