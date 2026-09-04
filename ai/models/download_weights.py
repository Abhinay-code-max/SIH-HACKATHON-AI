"""
Development-time helper to fetch local YOLO model weights.
Supports: yolov8n, yolov8s (recommended), yolov8m, yolo11s, etc.
Usage:
  python ai/models/download_weights.py          # downloads yolov8s by default
  python ai/models/download_weights.py yolov8m  # downloads yolov8m
"""

import sys
from pathlib import Path
from ultralytics import YOLO

def setup_weights(model_name: str = "yolov8s"):
    target_dir = Path(__file__).resolve().parent
    target_file = target_dir / f"{model_name}.pt"

    print(f"[Setup] Target model: {model_name}")
    print(f"[Setup] Target path:  {target_file}")

    if target_file.exists():
        size_mb = target_file.stat().st_size / (1024 * 1024)
        print(f"[Setup] Model already present locally ({size_mb:.2f} MB). Skipping download.")
        return

    print(f"[Setup] Fetching {model_name}.pt weights for local offline bundling...")
    # Downloads model weights once during development setup
    model = YOLO(f"{model_name}.pt")

    # If downloaded to current working directory, relocate to ai/models/
    cwd_download = Path(f"{model_name}.pt")
    if cwd_download.exists() and cwd_download.resolve() != target_file:
        cwd_download.replace(target_file)

    size_mb = target_file.stat().st_size / (1024 * 1024)
    print(f"[Setup] Success! Weights bundled to: {target_file} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    requested_model = sys.argv[1] if len(sys.argv) > 1 else "yolov8s"
    setup_weights(requested_model)
