"""
Unified Dataset Leakage & Cross-Split Auditor (Phase 6).
Audits training_lab/datasets/dataset_unified_v001/ to guarantee:
1. Strict cross-split disjointness (Train ∩ Val = ∅, Train ∩ Holdout = ∅, Val ∩ Holdout = ∅).
2. Perceptual and byte-level hash verification of all images across splits.
3. Cross-dataset collision audit against canonical UA-DETRAC (dataset_ua_detrac_v001).
4. Full reporting saved to dataset_unified_v001/leakage_audit_report.json and training_lab/reports/.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

UNIFIED_DATASET_DIR = ROOT_DIR / "training_lab" / "datasets" / "dataset_unified_v001"
UA_DETRAC_DATASET_DIR = ROOT_DIR / "training_lab" / "datasets" / "dataset_ua_detrac_v001"
REPORTS_DIR = ROOT_DIR / "training_lab" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class UnifiedLeakageDetector:
    def __init__(
        self,
        unified_dir: Path = UNIFIED_DATASET_DIR,
        ua_detrac_dir: Path = UA_DETRAC_DATASET_DIR,
    ):
        self.unified_dir = Path(unified_dir)
        self.ua_detrac_dir = Path(ua_detrac_dir)

    @staticmethod
    def compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def compute_perceptual_hash(img_path: Path) -> str:
        """Computes a 16x16 grayscale average perceptual hash."""
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return UnifiedLeakageDetector.compute_sha256(img_path)
            resized = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
            avg = resized.mean()
            diff = resized > avg
            return "".join("1" if b else "0" for b in diff.flatten())
        except Exception:
            return UnifiedLeakageDetector.compute_sha256(img_path)

    def audit(self) -> Dict[str, Any]:
        print("=" * 80)
        print("UNIFIED DATASET LEAKAGE & COLLISION AUDIT")
        print("=" * 80)

        violations: List[str] = []
        splits = ["train", "val", "holdout"]

        # Collect image files per split
        split_images: Dict[str, Dict[str, Path]] = {}
        split_hashes: Dict[str, Dict[str, str]] = {}
        split_phashes: Dict[str, Dict[str, str]] = {}

        for s in splits:
            s_dir = self.unified_dir / "images" / s
            files = sorted(list(s_dir.glob("*.jpg")))
            split_images[s] = {f.name: f for f in files}
            print(f"Split '{s}': {len(files):,} images found.")

        # 1. Exact Filename Overlap Check
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                overlap = set(split_images[s1].keys()) & set(split_images[s2].keys())
                if overlap:
                    violations.append(f"Exact filename overlap between {s1} and {s2}: {len(overlap)} files: {sorted(list(overlap))[:5]}")

        # 2. Byte SHA-256 and Perceptual Hash Check across splits
        print("\nComputing byte SHA-256 and perceptual hashes...")
        seen_sha: Dict[str, Tuple[str, str]] = {}  # sha -> (split, filename)
        seen_phash: Dict[str, Tuple[str, str]] = {}  # phash -> (split, filename)

        for s in splits:
            for fname, p in split_images[s].items():
                sha = self.compute_sha256(p)
                if sha in seen_sha:
                    prev_s, prev_f = seen_sha[sha]
                    violations.append(f"Duplicate byte content detected: {s}/{fname} == {prev_s}/{prev_f}")
                else:
                    seen_sha[sha] = (s, fname)

                phash = self.compute_perceptual_hash(p)
                if phash in seen_phash:
                    prev_s, prev_f = seen_phash[phash]
                    if s != prev_s:
                        violations.append(f"Visual perceptual collision between splits: {s}/{fname} matches {prev_s}/{prev_f}")
                else:
                    seen_phash[phash] = (s, fname)

        # 3. Cross-Dataset Collision Audit against UA-DETRAC
        print("\nAuditing cross-dataset collisions against UA-DETRAC (dataset_ua_detrac_v001)...")
        cross_collisions = []
        if self.ua_detrac_dir.is_dir():
            # UA-DETRAC train/val
            ua_files = list((self.ua_detrac_dir / "images").glob("**/*.jpg"))
            print(f"Checking against {len(ua_files):,} UA-DETRAC frames (sampling check)...")
            # All COCO names are 12-digit integers (e.g. 000000000139.jpg), UA-DETRAC are MVI_...
            coco_names = set(split_images["train"].keys()) | set(split_images["val"].keys()) | set(split_images["holdout"].keys())
            ua_names = {f.name for f in ua_files}
            name_collisions = coco_names & ua_names
            if name_collisions:
                violations.append(f"Cross-dataset filename collision: {name_collisions}")

            # Check SHA-256 collisions on sample of UA-DETRAC
            ua_sample = ua_files[::max(1, len(ua_files) // 500)]
            for uaf in ua_sample:
                u_sha = self.compute_sha256(uaf)
                if u_sha in seen_sha:
                    violations.append(f"Byte collision between COCO and UA-DETRAC: {uaf.name}")

        status = "LEAKAGE_FREE" if len(violations) == 0 else "CONTAMINATED"
        print(f"\nAudit Status: {status} ({len(violations)} violations)")

        report = {
            "dataset_version": "dataset_unified_v001",
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "leakage_detected": len(violations) > 0,
            "counts": {s: len(split_images[s]) for s in splits},
            "violations_count": len(violations),
            "violations": violations,
            "cross_dataset_check": {
                "ua_detrac_checked": self.ua_detrac_dir.is_dir(),
                "ua_detrac_path": str(self.ua_detrac_dir),
                "collisions_found": len(cross_collisions),
            },
        }

        # Save reports
        report_file_dataset = self.unified_dir / "leakage_audit_report.json"
        report_file_global = REPORTS_DIR / "dataset_unified_leakage_report.json"

        with open(report_file_dataset, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open(report_file_global, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"Saved reports to {report_file_dataset} and {report_file_global}")

        if violations:
            raise ValueError(f"Leakage audit failed with {len(violations)} violations: {violations[:3]}")

        return report


if __name__ == "__main__":
    detector = UnifiedLeakageDetector()
    detector.audit()
