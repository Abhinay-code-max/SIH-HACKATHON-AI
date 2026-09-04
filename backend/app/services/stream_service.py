"""
Local Video Streaming Service.
Generates MJPEG multipart frames with real-time YOLO bounding box overlays.
Runs 100% locally on localhost without cloud dependencies.
"""

from pathlib import Path
import sys
import time
from typing import Generator
import cv2

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.inference.loader import get_device, load_local_model

TARGET_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack"}


class StreamService:
    def __init__(self, model_name: str = "yolov8l.pt"):
        self.model_name = model_name
        self.model = None
        self.device = get_device()

    def _ensure_model(self):
        if self.model is None:
            self.model = load_local_model(self.model_name)

    def generate_mjpeg_stream(
        self,
        source: str | int = 0,
        conf_threshold: float = 0.35,
        target_fps: int = 25,
    ) -> Generator[bytes, None, None]:
        """
        Yields multipart JPEG frames for live browser rendering via standard <img> tag.
        """
        self._ensure_model()

        # Handle webcam vs file
        if isinstance(source, str) and source.isdigit():
            cam_src = int(source)
        elif isinstance(source, str):
            candidate = Path(source)
            if not candidate.is_absolute():
                candidate = ROOT_DIR / source
            cam_src = str(candidate) if candidate.is_file() else 0
        else:
            cam_src = source

        cap = cv2.VideoCapture(cam_src)
        frame_interval = 1.0 / target_fps

        try:
            while cap.isOpened():
                start_t = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    # Loop video if file source reached end
                    if not isinstance(cam_src, int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                # YOLO Inference
                results = self.model.predict(
                    source=frame,
                    conf=conf_threshold,
                    device=self.device,
                    verbose=False,
                )

                # Draw bounding boxes
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = self.model.names[cls_id]
                    if cls_name in TARGET_CLASSES:
                        conf = float(box.conf[0].item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        color = (0, 255, 0) if cls_name == "person" else (255, 120, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            f"{cls_name} {conf:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                # Encode frame to JPEG
                _, jpeg_bytes = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes.tobytes() + b"\r\n"
                )

                elapsed = time.perf_counter() - start_t
                sleep_time = max(0.0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            cap.release()


stream_service = StreamService()
