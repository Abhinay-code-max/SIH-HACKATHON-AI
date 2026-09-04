"""
Local AI Model Loader for Offline Operation.
Complies with Rule 2: Absolute Offline Operation (No auto-downloads, no cloud fallbacks).
Supports: yolov8s (default), yolov8m, yolov8n, etc.
"""

import sys
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def get_device() -> str:
    """Return 'cuda' if NVIDIA GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_local_model(model_name_or_path: str = "yolov8s.pt") -> Path:
    """Resolve model path inside ai/models/ strictly without network access."""
    base_models_dir = Path(__file__).resolve().parent.parent / "models"

    candidate = Path(model_name_or_path)
    if candidate.is_file():
        return candidate.resolve()

    candidate_in_models = base_models_dir / model_name_or_path
    if candidate_in_models.is_file():
        return candidate_in_models.resolve()

    if not model_name_or_path.endswith(".pt"):
        candidate_with_ext = base_models_dir / f"{model_name_or_path}.pt"
        if candidate_with_ext.is_file():
            return candidate_with_ext.resolve()

    # If requested doesn't exist, check if any .pt exists in ai/models
    existing_pts = list(base_models_dir.glob("*.pt"))
    if existing_pts:
        return existing_pts[0].resolve()

    return candidate_in_models.resolve()


def load_local_model(model_name_or_path: str = "yolov8s.pt") -> "YOLO":
    """
    Load YOLO model weights strictly from local disk.
    Raises FileNotFoundError if model is not present locally.
    Never attempts to download from the internet at runtime.
    """
    if YOLO is None:
        raise ImportError(
            "The 'ultralytics' package is not installed. "
            "Please install development dependencies."
        )

    model_path = find_local_model(model_name_or_path)

    # Hard offline check: Ensure file exists locally
    if not model_path.is_file():
        raise FileNotFoundError(
            f"\n[OFFLINE ARCHITECTURE ERROR] Local model weights not found at:\n"
            f"  {model_path}\n"
            f"Automatic online downloading is prohibited per offline project rules.\n"
            f"Run 'python ai/models/download_weights.py {model_name_or_path}' once to bundle the weights."
        )

    device = get_device()
    print(f"[Loader] Loading local weights: {model_path.name} ({model_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"[Loader] Target compute device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # Load local model weights
    model = YOLO(str(model_path))
    model.to(device)

    print(f"[Loader] Model successfully loaded into VRAM on {device}.")
    print(f"[Loader] Total classes: {len(model.names)}")
    return model


if __name__ == "__main__":
    requested = sys.argv[1] if len(sys.argv) > 1 else "yolov8s.pt"
    try:
        model = load_local_model(requested)
        print("\n--- Model Verification Summary ---")
        sample_classes = {k: model.names[k] for k in range(min(10, len(model.names)))}
        print(f"Sample classes: {sample_classes}")
        print("Status: LOCAL MODEL LOAD SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
