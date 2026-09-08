"""
Regression Tests for Phase D.4:
CUDA Diagnostics, cuDNN Deterministic Configuration, AMP/Worker Fallback Ladder,
Failure Diagnostic JSON Writing, Unique Retry Run Naming, and Dataset Immutability.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.cuda_diagnostic import (
    FALLBACK_CONFIGURATIONS,
    capture_gpu_memory_stats,
    capture_system_memory_stats,
    classify_failure,
    configure_stable_cudnn,
    get_cuda_diagnostics,
    reset_cuda_state,
    write_failure_diagnostic,
)
from training_lab.experiments.run_d1_gpu_diagnostic import (
    run_single_diagnostic,
)
from training_lab.experiments.run_d1_full_training import (
    CANONICAL_DATASET_SHA256,
    HOLDOUT_SEQUENCES,
    REQUIRED_AMP,
    REQUIRED_BATCH,
    REQUIRED_DEVICE,
    REQUIRED_EPOCHS,
    REQUIRED_IMGSZ,
    REQUIRED_MOSAIC,
    REQUIRED_WORKERS,
    extract_sequence_names,
    run_preflight,
)
from training_lab.engine.dataset_view import (
    DerivedDatasetView,
    remap_label_line,
    compute_file_sha256,
)


def test_cuda_diagnostics_and_allocation():
    """1. Test CUDA diagnostics collection and basic tensor allocation."""
    diag = get_cuda_diagnostics(amp=True, workers=8, batch=16, imgsz=640, mosaic=1.0)
    assert "torch_version" in diag
    assert "cuda_available" in diag
    assert "gpu_name" in diag
    assert "cudnn_version" in diag
    assert diag["amp_setting"] is True
    assert diag["workers"] == 8
    assert diag["batch"] == 16
    assert diag["imgsz"] == 640
    if torch.cuda.is_available():
        assert diag["tensor_allocation_test"] is True


def test_cudnn_configuration_explicitly_controlled():
    """2. Test that cuDNN configuration is explicitly controlled for stability."""
    cfg = configure_stable_cudnn()
    assert cfg["cudnn_enabled"] is True
    assert cfg["cudnn_benchmark"] is False
    assert cfg["cudnn_deterministic"] is True
    assert cfg["seed"] == 0
    assert os.environ.get("PYTHONHASHSEED") == "0"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert torch.backends.cudnn.enabled is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_amp_fallback_configuration_and_naming():
    """3 & 4. Test fallback configurations and unique retry run naming."""
    assert len(FALLBACK_CONFIGURATIONS) == 3

    # Primary
    assert FALLBACK_CONFIGURATIONS[0]["attempt"] == 0
    assert FALLBACK_CONFIGURATIONS[0]["amp"] is True
    assert FALLBACK_CONFIGURATIONS[0]["workers"] == 8
    assert FALLBACK_CONFIGURATIONS[0]["name"] == "D1_full_yolov8m_640_amp"

    # Fallback 1
    assert FALLBACK_CONFIGURATIONS[1]["attempt"] == 1
    assert FALLBACK_CONFIGURATIONS[1]["amp"] is False
    assert FALLBACK_CONFIGURATIONS[1]["workers"] == 8
    assert FALLBACK_CONFIGURATIONS[1]["name"] == "D1_full_yolov8m_640_amp_retry_noamp"

    # Fallback 2
    assert FALLBACK_CONFIGURATIONS[2]["attempt"] == 2
    assert FALLBACK_CONFIGURATIONS[2]["amp"] is False
    assert FALLBACK_CONFIGURATIONS[2]["workers"] == 4
    assert FALLBACK_CONFIGURATIONS[2]["name"] == "D1_full_yolov8m_640_noamp_workers4"

    # Verify all run names are strictly unique
    names = [c["name"] for c in FALLBACK_CONFIGURATIONS]
    assert len(names) == len(set(names)), "Run names in fallback ladder must be strictly unique!"


def test_failure_diagnostic_written(tmp_path):
    """5. Test that failure diagnostics JSON is cleanly written upon exception."""
    test_run_dir = tmp_path / "test_failed_run"
    dummy_exception = RuntimeError("cuDNN error: CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH")

    json_path = write_failure_diagnostic(
        experiment_name="D1_test_failure",
        run_dir=test_run_dir,
        exception=dummy_exception,
        amp=True,
        workers=8,
        batch=16,
        imgsz=640,
        mosaic=1.0,
        epoch=1,
        batch_idx=1267,
    )

    assert json_path.is_file(), "failure_diagnostic.json was not created!"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_name"] == "D1_test_failure"
    assert data["epoch"] == 1
    assert data["batch"] == 1267
    assert data["exception_type"] == "RuntimeError"
    assert "CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH" in data["exception_message"]
    assert data["amp"] is True
    assert data["workers"] == 8
    assert "gpu_memory_allocated_mb" in data


def test_derived_dataset_view_remains_correct(tmp_path):
    """6 & 7. Test DerivedDatasetView behavior and class remapping."""
    canonical_dummy = tmp_path / "dummy_ds"
    (canonical_dummy / "images" / "val").mkdir(parents=True, exist_ok=True)
    (canonical_dummy / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Create dummy car (canonical 1) and bus (canonical 3)
    (canonical_dummy / "images" / "val" / "img1.jpg").write_bytes(b"\x00" * 100)
    (canonical_dummy / "labels" / "val" / "img1.txt").write_text("1 0.5 0.5 0.2 0.2\n3 0.3 0.3 0.1 0.1\n")

    derived_dummy = tmp_path / "derived_ds"
    yaml_p = DerivedDatasetView.create_view(
        source_dataset_dir=canonical_dummy,
        target_view_dir=derived_dummy,
        class_mapping={1: 0, 3: 1},
        names={0: "car", 1: "bus"},
        splits=["val"],
    )

    assert yaml_p.is_file()
    remapped_lbl = derived_dummy / "labels" / "val" / "img1.txt"
    assert remapped_lbl.is_file()
    lines = remapped_lbl.read_text().strip().splitlines()
    assert lines[0].startswith("0 "), f"Expected class 0 for car, got: {lines[0]}"
    assert lines[1].startswith("1 "), f"Expected class 1 for bus, got: {lines[1]}"


def test_canonical_dataset_remains_untouched():
    """8. Test that the real canonical dataset remains 100% byte-for-byte immutable."""
    canonical_root = ROOT_DIR / "training_lab" / "datasets" / "dataset_ua_detrac_v001"
    yaml_p = canonical_root / "data.yaml"
    assert yaml_p.is_file(), "Canonical data.yaml missing!"

    sha = hashlib.sha256(yaml_p.read_bytes()).hexdigest()
    assert sha == "594bf7a2b6a870faa1c176a1904c524cfb287010118f88ee0d0b1402c8eae6b9", (
        f"Canonical dataset data.yaml SHA-256 mutated! Got {sha}"
    )

    train_imgs = len(list((canonical_root / "images" / "train").glob("*.jpg")))
    val_imgs = len(list((canonical_root / "images" / "val").glob("*.jpg")))
    assert train_imgs == 37935, f"Expected 37,935 train images, found {train_imgs}"
    assert val_imgs == 11130, f"Expected 11,130 val images, found {val_imgs}"


def test_system_memory_diagnostics():
    """9. Test system RAM, swap/pagefile, and GPU memory diagnostics via psutil."""
    mem = capture_system_memory_stats()
    assert "ram_total_gb" in mem and mem["ram_total_gb"] > 0
    assert "ram_available_gb" in mem and mem["ram_available_gb"] >= 0
    assert "ram_percent" in mem
    assert "swap_total_gb" in mem and mem["swap_total_gb"] >= 0
    assert "swap_free_gb" in mem and mem["swap_free_gb"] >= 0
    assert "gpu_allocated_mb" in mem
    assert "gpu_reserved_mb" in mem


def test_failure_classification_categories():
    """10. Test explicit failure classification against all known failure modes."""
    # CUDA OOM
    assert classify_failure(RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")) == "CUDA_OOM"

    # Windows Pagefile / WinError 1455
    win_err = OSError("[WinError 1455] The paging file is too small for this operation to complete")
    assert classify_failure(win_err) == "WINDOWS_PAGEFILE_EXHAUSTION"

    # cuDNN Stream Mismatch
    cudnn_err = RuntimeError("cuDNN error: CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH")
    assert classify_failure(cudnn_err) == "CUDNN_STREAM_MISMATCH"

    # Other CUDA error
    other_cuda = RuntimeError("CUDA error: an illegal memory access was encountered")
    assert classify_failure(other_cuda) == "OTHER_CUDA_ERROR"

    # Dataset error
    dataset_err = ValueError("Corrupt label line in dataset: non-numeric class index")
    assert classify_failure(dataset_err) == "DATASET_ERROR"

    # Process / Worker error
    proc_err = BrokenPipeError("DataLoader worker process exited unexpectedly")
    assert classify_failure(proc_err) == "PROCESS_ERROR"


def test_gpu_diagnostic_runner_parameters():
    """11. Test that the GPU diagnostic runner uses fixed batch=4, workers=0, and valid device."""
    import inspect
    sig = inspect.signature(run_single_diagnostic)
    params = sig.parameters

    assert params["batch_size"].default == 4, "Diagnostic batch size must default to 4 (no AutoBatch)"
    assert params["workers"].default == 0, "Diagnostic workers must default to 0 (no DataLoader child processes)"
    assert params["device"].default == "0", "Diagnostic device must default to GPU 0"
    assert params["target_batches"].default == 30, "Diagnostic target batches must be within 20-50 range"


def test_d1_full_runner_invariants():
    """12. Test that D1 full runner enforces verified hyperparameter invariants."""
    assert REQUIRED_BATCH == 4, "Batch size must be strictly 4"
    assert REQUIRED_WORKERS == 0, "Workers must be strictly 0 (no DataLoader child processes on Windows)"
    assert REQUIRED_DEVICE == "0", "Device must be strictly '0'"
    assert REQUIRED_IMGSZ == 640, "Image size must be 640"
    assert REQUIRED_EPOCHS == 5, "Full D1 training must be 5 epochs"
    assert REQUIRED_AMP is True, "AMP must be True for full D1 run"
    assert REQUIRED_MOSAIC == 1.0, "Mosaic probability must be 1.0"
    assert CANONICAL_DATASET_SHA256 == "594bf7a2b6a870faa1c176a1904c524cfb287010118f88ee0d0b1402c8eae6b9"


def test_d1_preflight_rejection_rules():
    """13. Test that preflight strictly rejects non-compliant parameters with ValueError."""
    with pytest.raises(ValueError, match="batch=8 != 4"):
        run_preflight(batch=8)

    with pytest.raises(ValueError, match="workers=4 != 0"):
        run_preflight(workers=4)

    with pytest.raises(ValueError, match="device='cpu' != '0'"):
        run_preflight(device="cpu")

    with pytest.raises(ValueError, match="amp=False != True"):
        run_preflight(amp=False)

    with pytest.raises(ValueError, match="epochs=1 != 5"):
        run_preflight(epochs=1)

    with pytest.raises(ValueError, match="imgsz=1280 != 640"):
        run_preflight(imgsz=1280)

    with pytest.raises(ValueError, match="mosaic=0.5 != 1.0"):
        run_preflight(mosaic=0.5)


def test_holdout_protection_and_sequence_separation():
    """14. Test that all 7 internal holdout sequences are 100% excluded from train and val."""
    assert len(HOLDOUT_SEQUENCES) == 7
    canonical_root = ROOT_DIR / "training_lab" / "datasets" / "dataset_ua_detrac_v001"
    train_files = [f.name for f in (canonical_root / "images" / "train").glob("*.jpg")]
    val_files = [f.name for f in (canonical_root / "images" / "val").glob("*.jpg")]

    for seq in HOLDOUT_SEQUENCES:
        train_matches = [f for f in train_files if seq in f]
        val_matches = [f for f in val_files if seq in f]
        assert len(train_matches) == 0, f"Holdout sequence {seq} found in train set!"
        assert len(val_matches) == 0, f"Holdout sequence {seq} found in val set!"

    train_seqs = extract_sequence_names(train_files)
    val_seqs = extract_sequence_names(val_files)
    assert len(train_seqs) == 32, f"Expected 32 train sequences, found {len(train_seqs)}"
    assert len(val_seqs) == 7, f"Expected 7 val sequences, found {len(val_seqs)}"
    assert set(train_seqs).isdisjoint(set(val_seqs)), "Train and val sequences must be strictly disjoint!"


def test_cuda_absence_and_cpu_fallback_rejected(monkeypatch):
    """15. Test that CUDA absence loudly fails preflight without falling back to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is strictly prohibited"):
        run_preflight()

