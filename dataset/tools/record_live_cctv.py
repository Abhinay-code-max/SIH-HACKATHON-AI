"""
Live CCTV / Webcam Collector for Training Dataset v2.
Records 15-30 seconds of live camera footage in your actual room/lighting conditions,
intelligently samples non-redundant frames, and queues them for human verification and fine-tuning.
"""

import argparse
from pathlib import Path
import sys
import time
import cv2

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.tools.extract_frames import extract_cctv_frames
from dataset.tools.pre_annotate import run_pre_annotation


def record_live_training_data(
    camera_index: int = 0,
    duration_seconds: int = 15,
    camera_id: str = "CAM_01",
) -> dict:
    """Records live camera footage and prepares frames for active learning."""
    out_dir = ROOT_DIR / "dataset" / "raw_footage"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{camera_id}_live_session.mp4"

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot access webcam at index {camera_index}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    print("=" * 70)
    print(f"RECORDING LIVE SURVEILLANCE FOOTAGE: {duration_seconds} SECONDS")
    print("=" * 70)
    print("Move around in your camera view, walk past the camera, or step into the scene.")
    print("Press 'q' in the window to stop recording early.\n")

    start_t = time.time()
    frames_recorded = 0

    while True:
        elapsed = time.time() - start_t
        if elapsed >= duration_seconds:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Display recording indicator
        display = frame.copy()
        remaining = max(0, int(duration_seconds - elapsed))
        cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(display, f"REC // {remaining}s REMAINING (Press 'q' to stop)", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(frame)
        frames_recorded += 1

        cv2.imshow("Live CCTV Dataset Recorder", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Recorder] Stopped by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n[Recorder] Recorded {frames_recorded} frames -> {video_path.name}")

    # Automatically extract non-redundant frames
    print("[Pipeline] Extracting non-redundant frames at 2.0 FPS...")
    extracted_res = extract_cctv_frames(
        video_path=video_path,
        camera_id=camera_id,
        sample_fps=2.0,
        motion_threshold=1.0,
        max_frames=60,
    )

    # Run GPU pre-annotation
    print("[Pipeline] Running YOLO Large pre-annotation on RTX 4060 GPU...")
    annot_res = run_pre_annotation(
        input_dir=extracted_res["output_dir"],
        camera_id=camera_id,
        model_name="yolov8l.pt",
        conf_min=0.30,
    )

    return {
        "video_path": str(video_path),
        "frames_extracted": extracted_res["saved_frames"],
        "boxes_generated": annot_res["total_boxes"],
        "uncertain_flagged": annot_res["uncertain_boxes"],
        "studio_url": "http://127.0.0.1:8000/annotate",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record live camera footage for dataset fine-tuning.")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    parser.add_argument("--camera", default="CAM_01", help="Camera ID")
    parser.add_argument("--device", type=int, default=0, help="Camera index")

    args = parser.parse_args()

    try:
        rep = record_live_training_data(
            camera_index=args.device,
            duration_seconds=args.duration,
            camera_id=args.camera,
        )
        print("\n" + "=" * 70)
        print("LIVE FOOTAGE COLLECTED & READY FOR VERIFICATION")
        print("=" * 70)
        print(f"Video Saved: {rep['video_path']}")
        print(f"Frames Sampled: {rep['frames_extracted']}")
        print(f"Target Objects Detected: {rep['boxes_generated']} (Uncertain for review: {rep['uncertain_flagged']})")
        print(f"Verify in Studio: {rep['studio_url']}")
        print("Status: LIVE DATASET COLLECTION SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
