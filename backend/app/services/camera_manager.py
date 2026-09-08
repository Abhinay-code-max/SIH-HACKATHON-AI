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

from ai.detection.detector import BaseDetector, YoloDetector
from ai.tracking.tracker import ObjectTracker
from ai.events.intelligence_engine import intelligence_engine
from ai.reid.manager import global_subject_manager
from mapping.manager import OfflineMapManager


class CameraStreamWorker:
    def __init__(
        self,
        camera_id: str,
        source: str | int,
        name: str,
        detector: Optional[BaseDetector] = None,
    ):
        self.camera_id = camera_id
        self.source = source
        self.name = name
        self.detector = detector or YoloDetector(model_name="auto")
        self.tracker = ObjectTracker(detector=self.detector)
        self.status = "ONLINE"
        self.fps = 0.0
        self.active_tracks_count = 0
        self.last_frame = None
        self.latest_tracks: List[dict] = []
        self.last_jpeg: Optional[bytes] = None

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

        # 3. Global Re-ID Persistent Badges
        for t in tracks:
            gid = t.get("global_display_name")
            if gid and "bbox" in t:
                x1, y1 = int(t["bbox"][0]), int(t["bbox"][1])
                badge_y = max(35, y1 - 22)
                badge_w = len(gid) * 8 + 8
                cv2.rectangle(annotated, (x1, badge_y - 13), (x1 + badge_w, badge_y + 4), (15, 23, 42), -1)
                cv2.rectangle(annotated, (x1, badge_y - 13), (x1 + badge_w, badge_y + 4), (56, 189, 248), 1)
                cv2.putText(annotated, gid, (x1 + 4, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1)

        # 4. Bottom HUD Overlay Bar
        cv2.rectangle(annotated, (10, h - 36), (w - 10, h - 8), (10, 15, 25), -1)
        cv2.rectangle(annotated, (10, h - 36), (w - 10, h - 8), (40, 55, 80), 1)
        hud = f"{self.camera_id} // {self.name} | FPS: {self.fps:.1f} | LOCAL TRACKS: {self.active_tracks_count} | RE-ID: ACTIVE"
        cv2.putText(annotated, hud, (20, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        return annotated


class MultiCameraManager:
    def __init__(self):
        # Shared single YoloDetector instance across camera workers to conserve VRAM on 4GB hardware
        self.detector = YoloDetector(model_name="auto")
        self.cameras: Dict[str, CameraStreamWorker] = {
            "CAM_01": CameraStreamWorker("CAM_01", 0, "Main Demonstration CCTV / Webcam", detector=self.detector),
            "CAM_02": CameraStreamWorker("CAM_02", str(ROOT_DIR / "data" / "sample-videos" / "sample_surveillance.mp4"), "Gate 1 Vehicle Entry", detector=self.detector),
            "CAM_03": CameraStreamWorker("CAM_03", str(ROOT_DIR / "data" / "sample-videos" / "annotated_surveillance.mp4"), "Perimeter Command CCTV", detector=self.detector),
            "CAM_04": CameraStreamWorker("CAM_04", str(ROOT_DIR / "training_lab" / "videos" / "CAM_04_north_fence.mp4"), "North Fence Sector", detector=self.detector),
            "CAM_05": CameraStreamWorker("CAM_05", str(ROOT_DIR / "training_lab" / "videos" / "CAM_05_logistics.mp4"), "Logistics Road Crossing", detector=self.detector),
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

    def _open_capture(self, camera_id: str) -> cv2.VideoCapture:
        """Opens video capture for a camera with webcam fallback to sample video."""
        worker = self.cameras[camera_id]
        src = worker.source
        if src == 0:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            is_valid = False
            if cap.isOpened():
                ret_probe, frame_probe = cap.read()
                if ret_probe and frame_probe is not None:
                    is_valid = True
            if not is_valid:
                cap.release()
                fallback_video = ROOT_DIR / "data" / "sample-videos" / "annotated_surveillance.mp4"
                if not fallback_video.is_file():
                    fallback_video = ROOT_DIR / "data" / "sample-videos" / "sample_surveillance.mp4"
                cap = cv2.VideoCapture(str(fallback_video))
            return cap
        return cv2.VideoCapture(src)

    def generate_live_mjpeg(self, camera_id: str) -> Generator[bytes, None, None]:
        """Streams live tracking feed with virtual fences and persistent IDs."""
        if camera_id not in self.cameras:
            camera_id = "CAM_01"

        worker = self.cameras[camera_id]
        src = worker.source
        cap = self._open_capture(camera_id)

        target_fps = 25
        frame_interval = 1.0 / target_fps
        drop_count = 0

        try:
            while cap.isOpened():
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    if not isinstance(src, int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    drop_count += 1
                    if drop_count < 15:
                        time.sleep(0.04)
                        continue
                    # Switch to fallback if webcam drops permanently
                    cap.release()
                    cap = cv2.VideoCapture(str(ROOT_DIR / "data" / "sample-videos" / "annotated_surveillance.mp4"))
                    continue
                drop_count = 0

                # Standardize frame resolution to 640x480 landscape for aligned multi-cam grid
                if frame.shape[0] != 480 or frame.shape[1] != 640:
                    frame = cv2.resize(frame, (640, 480))

                # 1. Single Unified Detection pass via BaseDetector (FP16 & Confidence Confirmation applied)
                detections = worker.detector.detect(frame, camera_id=worker.camera_id)

                # 2. Update Object Tracking consuming pre-computed detections (Zero duplicate YOLO inference)
                tracks, tracked_frame = worker.tracker.update(
                    frame=frame,
                    detections=detections,
                    conf_threshold=0.48,
                    camera_id=worker.camera_id,
                )

                # 2. Cross-Camera Global Subject Association (Re-ID)
                for t in tracks:
                    try:
                        subj = global_subject_manager.process_track(
                            camera_id=camera_id,
                            local_track_id=t["track_id"],
                            class_name=t["class_name"],
                            bbox=t["bbox"],
                            frame=frame,
                            dwell_seconds=t.get("dwell_seconds", 0.0),
                            confidence=t.get("confidence", 0.8),
                        )
                        t["global_subject_id"] = subj["subject_id"]
                        t["global_display_name"] = subj["display_name"]
                    except Exception:
                        t["global_subject_id"] = f"SUBJ_{t['track_id']:04d}"
                        t["global_display_name"] = f"[GLOBAL: {t['class_name'].upper()}]"

                worker.active_tracks_count = len(tracks)
                worker.latest_tracks = tracks

                # 3. Evaluate Intelligence Rules (Geofences, Tripwires, Loitering)
                intelligence_engine.evaluate_tracks(camera_id, tracks, frame)

                # 4. Draw Virtual Boundaries, Re-ID Badges & HUD
                final_frame = worker.draw_overlays(tracked_frame, tracks)

                # Measure FPS
                elapsed = time.perf_counter() - t0
                worker.fps = 1.0 / elapsed if elapsed > 0 else target_fps

                # Encode JPEG
                _, jpeg_bytes = cv2.imencode(".jpg", final_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                worker.last_jpeg = jpeg_bytes.tobytes()

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + worker.last_jpeg + b"\r\n"
                )

                sleep_time = max(0.0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            cap.release()

    def get_latest_tracks(self, camera_id: str) -> List[dict]:
        if camera_id in self.cameras:
            return self.cameras[camera_id].latest_tracks
        return []

    def get_latest_snapshot_bytes(self, camera_id: str) -> Optional[bytes]:
        if camera_id in self.cameras:
            worker = self.cameras[camera_id]
            if worker.last_jpeg is not None:
                return worker.last_jpeg
            # On-demand single frame capture if stream has not started yet
            cap = self._open_capture(camera_id)
            try:
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        if frame.shape[0] != 480 or frame.shape[1] != 640:
                            frame = cv2.resize(frame, (640, 480))
                        detections = worker.detector.detect(frame, camera_id=worker.camera_id)
                        tracks, tracked_frame = worker.tracker.update(
                            frame=frame,
                            detections=detections,
                            camera_id=worker.camera_id,
                        )
                        final_frame = worker.draw_overlays(tracked_frame, tracks)
                        _, jpeg_bytes = cv2.imencode(".jpg", final_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        worker.last_jpeg = jpeg_bytes.tobytes()
                        return worker.last_jpeg
            finally:
                cap.release()
        return None

    def get_full_camera_info(self, camera_id: str) -> Optional[dict]:
        if camera_id not in self.cameras:
            return None
        w = self.cameras[camera_id]
        rules = intelligence_engine.rules
        return {
            "camera_id": w.camera_id,
            "name": w.name,
            "status": w.status,
            "fps": round(w.fps, 1),
            "active_tracks_count": w.active_tracks_count,
            "stream_url": f"/api/cameras/stream/{w.camera_id}",
            "snapshot_url": f"/api/cameras/{w.camera_id}/snapshot",
            "resolution": [640, 480],
            "active_model": getattr(w.tracker, "model_name", "YOLO-L-v002"),
            "zones": [z for z in rules.get("zones", []) if z.get("camera_id") == w.camera_id],
            "tripwires": [t for t in rules.get("tripwires", []) if t.get("camera_id") == w.camera_id],
        }


camera_manager = MultiCameraManager()
