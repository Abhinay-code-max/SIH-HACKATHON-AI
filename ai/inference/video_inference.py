"""
Standalone Local Video Inference Engine.
Processes local video files or camera streams with configurable FPS / frame-skipping.
Outputs annotated video, structured detections JSON, and performance metrics (FPS/latency).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import torch
from ai.inference.loader import load_local_model, get_device

TARGET_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "traffic light",
}


def run_video_inference(
    source: str | int,
    model_name: str = "yolov8l.pt",
    conf_threshold: float = 0.35,
    process_every_n_frames: int = 1,
    max_frames: int | None = None,
    show_window: bool = False,
    output_video_path: str | Path | None = None,
    output_json_path: str | Path | None = None,
) -> dict:
    """
    Run local inference on a video stream or file.
    """
    model = load_local_model(model_name)
    device = get_device()

    # Determine if source is webcam index or file
    if isinstance(source, str) and source.isdigit():
        video_src = int(source)
    elif isinstance(source, str):
        video_src = str(Path(source).resolve())
        if not Path(video_src).is_file():
            raise FileNotFoundError(f"Video file not found: {video_src}")
    else:
        video_src = source

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0 or orig_fps > 120:
        orig_fps = 30.0
    total_file_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[Video Engine] Source: {source}")
    print(f"[Video Engine] Resolution: {orig_width}x{orig_height} @ {orig_fps:.1f} FPS (Total frames: {total_file_frames})")
    print(f"[Video Engine] Device: {device} | Model: {model_name} | Frame Step: {process_every_n_frames}")

    # Warmup GPU
    print("[Video Engine] Warming up GPU kernels...")
    dummy = torch.zeros((1, 3, 640, 640), device=device)
    try:
        model.model(dummy)
    except Exception:
        pass

    # Video Writer setup
    out_writer = None
    if output_video_path:
        out_path = Path(output_video_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Written video FPS matches the original FPS
        out_writer = cv2.VideoWriter(str(out_path), fourcc, orig_fps, (orig_width, orig_height))

    frame_idx = 0
    processed_count = 0
    skipped_count = 0
    latencies = []
    video_events = []
    last_detections = []

    start_wall_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        should_process = (frame_idx % process_every_n_frames == 0)

        if should_process:
            processed_count += 1
            t0 = time.perf_counter()
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                device=device,
                verbose=False,
            )
            infer_ms = (time.perf_counter() - t0) * 1000
            latencies.append(infer_ms)

            # Parse detections
            current_frame_dets = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                conf = float(box.conf[0].item())
                if cls_name in TARGET_CLASSES:
                    coords = [round(float(c), 1) for c in box.xyxy[0].tolist()]
                    current_frame_dets.append({
                        "class": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": coords,
                    })

            last_detections = current_frame_dets

            if current_frame_dets:
                video_events.append({
                    "frame_index": frame_idx,
                    "timestamp_sec": round(frame_idx / orig_fps, 3),
                    "detections_count": len(current_frame_dets),
                    "detections": current_frame_dets,
                })
        else:
            skipped_count += 1

        # Annotate frame
        annotated_frame = frame.copy()
        for det in last_detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = (0, 255, 0) if det["class"] == "person" else (255, 120, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_frame,
                f"{det['class']} {det['confidence']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Performance Overlay
        curr_fps = 1000 / latencies[-1] if latencies else orig_fps
        frame_label = f"Frame: {frame_idx}" if total_file_frames <= 0 else f"Frame: {frame_idx}/{total_file_frames}"
        info_text = f"{frame_label} | GPU FPS: {curr_fps:.1f} | Dets: {len(last_detections)} (Press 'q' to exit)"
        cv2.putText(annotated_frame, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if out_writer:
            out_writer.write(annotated_frame)

        if show_window:
            cv2.imshow("Local AI Video Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Video Engine] Stopped by user.")
                break

    total_wall_sec = time.perf_counter() - start_wall_time
    cap.release()
    if out_writer:
        out_writer.release()
    if show_window:
        cv2.destroyAllWindows()

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_fps = 1000 / avg_latency if avg_latency > 0 else 0.0
    overall_throughput_fps = frame_idx / total_wall_sec if total_wall_sec > 0 else 0.0

    summary = {
        "source": str(source),
        "resolution": f"{orig_width}x{orig_height}",
        "total_frames": frame_idx,
        "processed_frames": processed_count,
        "skipped_frames": skipped_count,
        "model": model_name,
        "device": device,
        "avg_inference_latency_ms": round(avg_latency, 2),
        "avg_inference_fps": round(avg_fps, 1),
        "overall_throughput_fps": round(overall_throughput_fps, 1),
        "total_event_frames": len(video_events),
        "events": video_events[:50],  # sample first 50 events
    }

    if output_video_path:
        summary["annotated_video_path"] = str(output_video_path)
    if output_json_path:
        json_p = Path(output_json_path).resolve()
        json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(json_p, "w") as f:
            json.dump(summary, f, indent=2)
        summary["json_summary_path"] = str(json_p)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local YOLO video inference.")
    parser.add_argument("source", nargs="?", default="data/sample-videos/sample_surveillance.mp4", help="Video file path or webcam index (0)")
    parser.add_argument("--model", default="yolov8l.pt", help="Model name or weights path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--step", type=int, default=1, help="Process every Nth frame (1 = all frames)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--show", action="store_true", help="Display live OpenCV window")
    parser.add_argument("--output-video", default="data/sample-videos/annotated_surveillance.mp4", help="Output annotated video path")
    parser.add_argument("--output-json", default="data/sample-videos/video_detections.json", help="Output detections JSON path")

    args = parser.parse_args()

    try:
        report = run_video_inference(
            source=args.source,
            model_name=args.model,
            conf_threshold=args.conf,
            process_every_n_frames=args.step,
            max_frames=args.max_frames,
            show_window=args.show,
            output_video_path=args.output_video,
            output_json_path=args.output_json,
        )
        print("\n--- Video Processing Performance Summary ---")
        print(f"Device: {report['device']} | Model: {report['model']}")
        print(f"Total Frames Processed: {report['processed_frames']} (Skipped: {report['skipped_frames']})")
        print(f"Avg Inference Latency: {report['avg_inference_latency_ms']} ms")
        print(f"Avg GPU Inference Speed: {report['avg_inference_fps']} FPS")
        print(f"Overall Pipeline Throughput: {report['overall_throughput_fps']} FPS")
        print(f"Annotated Video: {report.get('annotated_video_path')}")
        print(f"Detections JSON: {report.get('json_summary_path')}")
        print("Status: VIDEO INFERENCE SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
