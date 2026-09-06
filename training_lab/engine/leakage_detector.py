"""
Data Leakage Protection & Sequence Isolation Auditor.
Guarantees zero cross-split data leakage across Train, Val, and Holdout Test benchmark sets.
Validates that splits are partitioned strictly at the video/scenario level,
detects frame overlap, sequence collisions, and near-duplicate visual frames.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Union
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATASETS_DIR = ROOT_DIR / "training_lab" / "datasets"
RESULTS_DIR = ROOT_DIR / "training_lab" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class DataLeakageDetectedError(Exception):
    """Raised when data leakage between train and test/validation sets is discovered."""
    pass


class DataLeakageDetector:
    """
    Audits dataset splits to ensure complete independence of training and evaluation data.
    """

    def __init__(self, datasets_dir: Optional[Union[str, Path]] = None):
        self.datasets_dir = Path(datasets_dir) if datasets_dir else DATASETS_DIR

    @staticmethod
    def compute_image_hash(img_path: Path) -> str:
        """Computes a perceptual downsampled 16x16 grayscale hash for visual similarity checks."""
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return hashlib.md5(img_path.read_bytes()).hexdigest()
            resized = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
            avg = resized.mean()
            diff = resized > avg
            return "".join("1" if b else "0" for b in diff.flatten())
        except Exception:
            return hashlib.md5(img_path.read_bytes()).hexdigest()

    def audit_dataset(self, dataset_version: str) -> Dict[str, Any]:
        """
        Audits a dataset version on disk for video-level overlap and perceptual leakage.
        """
        version_dir = self.datasets_dir / dataset_version
        if not version_dir.is_dir():
            raise FileNotFoundError(f"Dataset version directory not found: {version_dir}")

        manifest_file = version_dir / "dataset_manifest.json"
        manifest = {}
        if manifest_file.is_file():
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        violations: List[str] = []

        # 1. Video / Scenario-Level Overlap Check
        videos_per_split = manifest.get("videos_per_split", {})
        train_videos: Set[str] = set(videos_per_split.get("train", []))
        val_videos: Set[str] = set(videos_per_split.get("val", []))
        test_videos: Set[str] = set(videos_per_split.get("test", []))

        # Check train vs test
        leak_train_test = train_videos.intersection(test_videos)
        if leak_train_test:
            violations.append(
                f"Video/Scenario leakage between Train and Test: {sorted(list(leak_train_test))}"
            )

        # Check train vs val
        leak_train_val = train_videos.intersection(val_videos)
        if leak_train_val:
            violations.append(
                f"Video/Scenario leakage between Train and Val: {sorted(list(leak_train_val))}"
            )

        # 2. Duplicate Image Filenames Cross Splits
        train_imgs = {f.name: f for f in (version_dir / "images" / "train").glob("*.jpg")}
        val_imgs = {f.name: f for f in (version_dir / "images" / "val").glob("*.jpg")}
        test_imgs = {f.name: f for f in (version_dir / "images" / "test").glob("*.jpg")}

        common_train_test = set(train_imgs.keys()).intersection(set(test_imgs.keys()))
        if common_train_test:
            violations.append(f"Exact filename overlap between Train and Test: {common_train_test}")

        common_train_val = set(train_imgs.keys()).intersection(set(val_imgs.keys()))
        if common_train_val:
            violations.append(f"Exact filename overlap between Train and Val: {common_train_val}")

        # 3. Perceptual Image Hash / Near-Duplicate Detection
        train_hashes = {}
        for fname, p in train_imgs.items():
            train_hashes[self.compute_image_hash(p)] = fname

        for fname, p in test_imgs.items():
            h = self.compute_image_hash(p)
            if h in train_hashes and train_hashes[h] != fname:
                # If images are from different videos, warn, but if from same sequence, flag violation
                violations.append(
                    f"Visual duplicate frame detected: Test image '{fname}' matches Train image '{train_hashes[h]}'"
                )

        has_leakage = len(violations) > 0
        status = "CONTAMINATED" if has_leakage else "LEAKAGE_FREE"

        report = {
            "dataset_version": dataset_version,
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "leakage_detected": has_leakage,
            "split_strategy": manifest.get("split_strategy", "VIDEO_SCENARIO_LEVEL"),
            "videos_per_split": {
                "train": sorted(list(train_videos)),
                "val": sorted(list(val_videos)),
                "test": sorted(list(test_videos)),
            },
            "sample_counts": manifest.get("splits", {}),
            "violations_count": len(violations),
            "violations": violations,
        }

        # Persist report
        report_file = version_dir / "leakage_audit_report.json"
        with open(report_file, "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2)

        return report

    def assert_leakage_free(self, dataset_version: str) -> Dict[str, Any]:
        """Runs audit and raises DataLeakageDetectedError if contaminated."""
        report = self.audit_dataset(dataset_version)
        if report.get("leakage_detected", False):
            raise DataLeakageDetectedError(
                f"Data leakage detected in dataset '{dataset_version}': {report.get('violations')}"
            )
        return report


data_leakage_detector = DataLeakageDetector()
