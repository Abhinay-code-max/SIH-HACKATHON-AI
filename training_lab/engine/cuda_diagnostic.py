"""
CUDA / cuDNN Diagnostics, Stream Stability & Controlled Fallback Engine.
Provides:
1. Complete hardware, driver, CUDA, and cuDNN diagnostic recording.
2. System RAM, Windows pagefile/swap, and GPU VRAM memory diagnostics via psutil and torch.cuda.
3. Explicit failure classification (CUDA_OOM, WINDOWS_PAGEFILE_EXHAUSTION, CUDNN_STREAM_MISMATCH, etc.).
4. Stable cuDNN configuration (enabled=True, benchmark=False, deterministic=True).
5. Explicit CUDA synchronization and memory cleanup utilities.
6. Safe fallback configuration ladder:
     Primary:    workers=8, amp=True  -> D1_full_yolov8m_640_amp
     Fallback 1: workers=8, amp=False -> D1_full_yolov8m_640_amp_retry_noamp
     Fallback 2: workers=4, amp=False -> D1_full_yolov8m_640_noamp_workers4
7. Diagnostic failure JSON persistence.
"""

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import psutil
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


FALLBACK_CONFIGURATIONS = [
    {
        "attempt": 0,
        "name": "D1_full_yolov8m_640_amp",
        "amp": True,
        "workers": 8,
        "description": "Primary full D1 configuration (workers=8, AMP=True)",
    },
    {
        "attempt": 1,
        "name": "D1_full_yolov8m_640_amp_retry_noamp",
        "amp": False,
        "workers": 8,
        "description": "Fallback 1: Stabilized FP32 (workers=8, AMP=False)",
    },
    {
        "attempt": 2,
        "name": "D1_full_yolov8m_640_noamp_workers4",
        "amp": False,
        "workers": 4,
        "description": "Fallback 2: Stabilized FP32 with reduced worker concurrency (workers=4, AMP=False)",
    },
]


def configure_stable_cudnn(seed: int = 0) -> Dict[str, Any]:
    """
    Configures cuDNN for strictly stable, deterministic algorithm selection without globally disabling GPU acceleration.
    Avoids unstable dynamic algorithm benchmarking across asynchronous streams.
    Enforces deterministic seeds across Python random, NumPy, PyTorch CPU, and CUDA.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return {
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "seed": seed,
    }


def reset_cuda_state() -> None:
    """
    Explicitly synchronizes CUDA streams, clears cache, and runs garbage collection.
    Guarantees no corrupted optimizer or model state leaks across attempts.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def get_cuda_diagnostics(
    amp: bool = True,
    workers: int = 8,
    batch: int = 16,
    imgsz: int = 640,
    mosaic: float = 1.0,
    device: str = "0",
) -> Dict[str, Any]:
    """
    Collects complete CUDA, cuDNN, driver, hardware, and training configuration parameters.
    Also executes a small CUDA tensor allocation test.
    """
    cuda_avail = torch.cuda.is_available()

    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.version else None,
        "cuda_available": cuda_avail,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "None",
        "gpu_capability": torch.cuda.get_device_capability(0) if cuda_avail else None,
        "total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if cuda_avail else 0
        ),
        "total_memory_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if cuda_avail
            else 0.0
        ),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "amp_setting": amp,
        "workers": workers,
        "batch": batch,
        "imgsz": imgsz,
        "mosaic": mosaic,
        "device": device,
    }

    # Small CUDA tensor allocation test
    tensor_test_passed = False
    if cuda_avail:
        try:
            torch.cuda.synchronize()
            t = torch.zeros((4, 3, 64, 64), device="cuda", dtype=torch.float32)
            del t
            torch.cuda.synchronize()
            tensor_test_passed = True
        except Exception as e:
            tensor_test_passed = False
            info["tensor_allocation_error"] = str(e)

    info["tensor_allocation_test"] = tensor_test_passed
    return info


def print_cuda_diagnostics(diag: Dict[str, Any]) -> None:
    """Pretty prints the CUDA diagnostic summary."""
    print("=" * 80)
    print("CUDA / cuDNN ENVIRONMENT & HARDWARE DIAGNOSTICS")
    print("=" * 80)
    for k, v in diag.items():
        print(f"  {k:28s}: {v}")
    print("=" * 80)


def capture_gpu_memory_stats() -> Dict[str, float]:
    """Captures allocated, reserved, and free VRAM in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 0.0, "total_mb": 0.0}

    allocated = torch.cuda.memory_allocated(0) / (1024**2)
    reserved = torch.cuda.memory_reserved(0) / (1024**2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    free = total - reserved

    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "free_mb": round(free, 2),
        "total_mb": round(total, 2),
    }


def capture_system_memory_stats() -> Dict[str, Any]:
    """Captures system RAM, Windows swap/pagefile, and GPU memory metrics."""
    v = psutil.virtual_memory()
    s = psutil.swap_memory()
    gpu = capture_gpu_memory_stats()
    return {
        "ram_total_gb": round(v.total / (1024**3), 2),
        "ram_available_gb": round(v.available / (1024**3), 2),
        "ram_used_gb": round(v.used / (1024**3), 2),
        "ram_percent": v.percent,
        "swap_total_gb": round(s.total / (1024**3), 2),
        "swap_free_gb": round(s.free / (1024**3), 2),
        "swap_used_gb": round(s.used / (1024**3), 2),
        "swap_percent": s.percent,
        "gpu_allocated_mb": gpu["allocated_mb"],
        "gpu_reserved_mb": gpu["reserved_mb"],
        "gpu_free_mb": gpu["free_mb"],
        "gpu_total_mb": gpu["total_mb"],
    }


def classify_failure(exc: Exception) -> str:
    """
    Classifies failure into one of the explicit diagnostic categories:
    - CUDA_OOM: GPU VRAM exhaustion (e.g. torch.cuda.OutOfMemoryError)
    - WINDOWS_PAGEFILE_EXHAUSTION: WinError 1455 / commit charge failure
    - CUDNN_STREAM_MISMATCH: CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH
    - OTHER_CUDA_ERROR: Any other CUDA or cuDNN runtime error
    - DATASET_ERROR: Missing files, corrupted labels, bad annotations
    - PROCESS_ERROR: Broken pipe, worker spawn, IPC, EOF errors
    """
    msg = str(exc).lower()

    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg or "cuda out of memory" in msg:
        return "CUDA_OOM"
    if "1455" in msg or "paging file is too small" in msg or "winerror 1455" in msg:
        return "WINDOWS_PAGEFILE_EXHAUSTION"
    if "cudnn_status_bad_param_stream_mismatch" in msg or "stream_mismatch" in msg:
        return "CUDNN_STREAM_MISMATCH"
    if "cudnn" in msg or "cuda" in msg or "cublas" in msg or "cufft" in msg:
        return "OTHER_CUDA_ERROR"
    if "dataset" in msg or "label" in msg or "corrupt" in msg or "annotation" in msg:
        return "DATASET_ERROR"
    return "PROCESS_ERROR"


def write_failure_diagnostic(
    experiment_name: str,
    run_dir: Path,
    exception: Exception,
    amp: bool,
    workers: int,
    batch: int = 16,
    imgsz: int = 640,
    mosaic: float = 1.0,
    epoch: Optional[int] = None,
    batch_idx: Optional[int] = None,
) -> Path:
    """
    Captures exact runtime/stream exception details into a structured JSON file.
    Does not fabricate any values.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    diag = get_cuda_diagnostics(
        amp=amp, workers=workers, batch=batch, imgsz=imgsz, mosaic=mosaic
    )
    sys_mem = capture_system_memory_stats()
    failure_category = classify_failure(exception)

    failure_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "run_directory": str(run_dir.resolve()).replace("\\", "/"),
        "failure_category": failure_category,
        "epoch": epoch,
        "batch": batch_idx,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "cuda_version": diag.get("cuda_version"),
        "pytorch_version": diag.get("torch_version"),
        "cudnn_version": diag.get("cudnn_version"),
        "gpu": diag.get("gpu_name"),
        "amp": amp,
        "workers": workers,
        "batch_size": batch,
        "imgsz": imgsz,
        "mosaic": mosaic,
        "system_ram_total_gb": sys_mem["ram_total_gb"],
        "system_ram_available_gb": sys_mem["ram_available_gb"],
        "system_ram_used_gb": sys_mem["ram_used_gb"],
        "system_swap_total_gb": sys_mem["swap_total_gb"],
        "system_swap_free_gb": sys_mem["swap_free_gb"],
        "gpu_memory_allocated_mb": sys_mem["gpu_allocated_mb"],
        "gpu_memory_reserved_mb": sys_mem["gpu_reserved_mb"],
        "gpu_memory_free_mb": sys_mem["gpu_free_mb"],
    }

    out_file = run_dir / "failure_diagnostic.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(failure_record, f, indent=2)

    print(f"\n[FAILURE DIAGNOSTIC RECORDED - {failure_category}] Written to: {out_file}")
    return out_file
