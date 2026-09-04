"""
Deterministic Sample Video Generator.
Creates a 5-second 1080p surveillance video locally using OpenCV.
Requires zero internet connection.
"""

from pathlib import Path
import cv2
import numpy as np

def generate_sample_surveillance_video(
    output_path: Path | str | None = None,
    duration_seconds: int = 5,
    fps: int = 30,
) -> Path:
    if output_path is None:
        target_dir = Path(__file__).resolve().parent
        output_file = target_dir / "sample_surveillance.mp4"
    else:
        output_file = Path(output_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use existing traffic_sample.jpg if available as realistic background
    sample_img_path = Path(__file__).resolve().parent.parent / "sample-images" / "traffic_sample.jpg"
    if sample_img_path.is_file():
        base_frame = cv2.imread(str(sample_img_path))
    else:
        base_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        base_frame[:] = (120, 120, 120)

    h, w = base_frame.shape[:2]
    total_frames = duration_seconds * fps

    # Setup VideoWriter with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_file), fourcc, fps, (w, h))

    print(f"[Video Gen] Generating {duration_seconds}s test video ({total_frames} frames, {fps} FPS, {w}x{h})...")

    # Simulate realistic surveillance panning and subtle motion
    for i in range(total_frames):
        # Subtle horizontal pan
        dx = int(15 * np.sin(2 * np.pi * i / total_frames))
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        frame = cv2.warpAffine(base_frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Add timestamp watermark
        timestamp_str = f"CAM-01 LOCAL STREAM - 2026-09-04 14:30:{i//fps:02d}.{int((i%fps)*(1000/fps)):03d}"
        cv2.putText(frame, timestamp_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, timestamp_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        out.write(frame)

    out.release()
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"[Video Gen] Video created successfully: {output_file} ({size_mb:.2f} MB)")
    return output_file

if __name__ == "__main__":
    generate_sample_surveillance_video()
