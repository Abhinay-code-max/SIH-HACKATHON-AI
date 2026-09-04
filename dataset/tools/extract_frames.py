"""
Intelligent CCTV Video Sampler & Frame Extractor.
Anti-Redundancy Engine: Samples at configurable FPS (default 1-2 FPS) and detects
motion/scene changes to discard redundant, static frames.
Maintains camera-level metadata for leak-free train/val/test splitting.
"""

import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def calculate_motion_delta(prev_gray, curr_gray) -> float:
    """Calculates mean absolute difference between two grayscale frames (0.0 to 100.0)."""
    if prev_gray is None:
        return 100.0
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))


def extract_cctv_frames(
    video_path: Path | str,
    camera_id: str = "CAM_01",
    sample_fps: float = 1.0,
    motion_threshold: float = 1.5,
    max_frames: int | None = 200,
    output_dir: Path | None = None,
) -> dict:
    """
    Intelligently extracts non-redundant frames from CCTV footage.
    """
    v_path = Path(video_path).resolve()
    if not v_path.is_file():
        raise FileNotFoundError(f"Video file not found at: {v_path}")

    if output_dir is None:
        out_base = ROOT_DIR / "dataset" / "extracted_frames" / camera_id
    else:
        out_base = Path(output_dir) / camera_id
    out_base.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {v_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or native_fps > 120:
        native_fps = 30.0
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame step based on desired sampling FPS
    frame_step = max(1, int(round(native_fps / sample_fps)))

    print(f"\n[Sampler] Video: {v_path.name} ({width}x{height} @ {native_fps:.1f} native FPS)")
    print(f"[Sampler] Target sampling: {sample_fps} FPS (step: every {frame_step} frames)")
    print(f"[Sampler] Motion filter threshold: {motion_threshold} | Output: {out_base.relative_to(ROOT_DIR)}")

    extracted_records = []
    frame_idx = 0
    saved_count = 0
    discarded_redundant = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Only evaluate frames at the target sampling interval
        if frame_idx % frame_step != 0:
            continue

        # Resize small for fast motion delta calculation
        gray_small = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
        motion_score = calculate_motion_delta(prev_gray, gray_small)

        # Discard static/redundant frames
        if prev_gray is not None and motion_score < motion_threshold:
            discarded_redundant += 1
            continue

        prev_gray = gray_small
        saved_count += 1

        # Camera-tagged filename for leak-free dataset management
        filename = f"{camera_id}_{v_path.stem}_frame_{frame_idx:06d}.jpg"
        save_path = out_base / filename
        cv2.imwrite(str(save_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        extracted_records.append({
            "filename": filename,
            "path": str(save_path.relative_to(ROOT_DIR)),
            "camera_id": camera_id,
            "source_video": v_path.name,
            "frame_index": frame_idx,
            "timestamp_sec": round(frame_idx / native_fps, 2),
            "motion_score": round(motion_score, 2),
            "resolution": [width, height],
        })

        if max_frames and saved_count >= max_frames:
            print(f"[Sampler] Reached max frame limit ({max_frames}). Stopping extraction.")
            break

    cap.release()

    # Save manifest
    manifest_path = ROOT_DIR / "dataset" / "extracted_frames" / f"manifest_{camera_id}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "camera_id": camera_id,
            "source_video": v_path.name,
            "total_extracted": len(extracted_records),
            "discarded_redundant": discarded_redundant,
            "frames": extracted_records,
        }, f, indent=2)

    return {
        "camera_id": camera_id,
        "video_name": v_path.name,
        "total_source_frames": frame_idx,
        "saved_frames": saved_count,
        "discarded_redundant": discarded_redundant,
        "output_dir": str(out_base),
        "manifest_path": str(manifest_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract non-redundant CCTV frames.")
    parser.add_argument("video", nargs="?", default="data/sample-videos/sample_surveillance.mp4", help="Video path")
    parser.add_argument("--camera", default="CAM_01", help="Camera ID tag")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling FPS (e.g. 1.0, 2.0, 5.0)")
    parser.add_argument("--threshold", type=float, default=1.0, help="Motion delta threshold")
    parser.add_argument("--max", type=int, default=100, help="Max frames to extract")

    args = parser.parse_args()

    # Ensure sample video exists
    sample_vid = Path(args.video)
    if not sample_vid.is_file():
        # Fallback to recorded annotated video or generate
        alt_vid = ROOT_DIR / "data" / "sample-videos" / "annotated_surveillance.mp4"
        if alt_vid.is_file():
            sample_vid = alt_vid

    try:
        res = extract_cctv_frames(
            video_path=sample_vid,
            camera_id=args.camera,
            sample_fps=args.fps,
            motion_threshold=args.threshold,
            max_frames=args.max,
        )
        print("\n--- Frame Extraction Summary ---")
        print(f"Camera Tag: {res['camera_id']} | Source: {res['video_name']}")
        print(f"Frames Extracted (Saved): {res['saved_frames']}")
        print(f"Redundant Frames Skipped: {res['discarded_redundant']}")
        print(f"Output Directory: {res['output_dir']}")
        print(f"Manifest: {res['manifest_path']}")
        print("Status: INTELLIGENT FRAME EXTRACTION SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
