"""
Dataset Generator & Data Quality Gate.
Audits human annotations, filters out corrupted/duplicate bounding boxes,
converts verified candidates into normalized YOLO format, and partitions data
into 70% Train / 15% Val / 15% Unseen Holdout Test benchmark sets.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATASETS_DIR = ROOT_DIR / "training_lab" / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
CLASSES_YAML_PATH = ROOT_DIR / "config" / "classes.yaml"


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union = areaA + areaB - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class DataQualityGate:
    """
    Validates annotation quality, rejects corrupted boundaries, filters sub-pixel
    noise, and flags duplicates and conflicting labels.
    """

    MIN_SIZE_PX = 10.0
    DUPLICATE_IOU_THRESHOLD = 0.95
    CONFLICT_IOU_THRESHOLD = 0.80

    @classmethod
    def audit_annotation(
        cls,
        bbox: List[float],
        image_shape: Tuple[int, ...],
        class_name: str,
        existing_boxes: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Audits an individual bounding box against data quality standards.

        Args:
            bbox: [x1, y1, x2, y2]
            image_shape: (height, width) or (h, w, c)
            class_name: target class name
            existing_boxes: list of already audited annotations on the same frame

        Returns:
            Tuple of (is_valid: bool, issues: List[str])
        """
        issues: List[str] = []

        if not bbox or len(bbox) != 4:
            return False, [f"Invalid bbox length: expected 4 coordinates, got {bbox}"]

        x1, y1, x2, y2 = map(float, bbox)
        h, w = float(image_shape[0]), float(image_shape[1])

        # 1. Coordinate ordering and inverted geometry
        if x2 <= x1:
            issues.append(f"Inverted horizontal coordinates: x1={x1} >= x2={x2}")
        if y2 <= y1:
            issues.append(f"Inverted vertical coordinates: y1={y1} >= y2={y2}")

        # 2. Boundary bounds clamping check
        if x1 < 0.0 or y1 < 0.0 or x2 > w or y2 > h:
            issues.append(f"Coordinates out of bounds: [{x1}, {y1}, {x2}, {y2}] exceeds image dimensions ({w}x{h})")

        # 3. Minimum dimension validation
        width_px = x2 - x1
        height_px = y2 - y1
        if width_px < cls.MIN_SIZE_PX or height_px < cls.MIN_SIZE_PX:
            issues.append(
                f"Bounding box smaller than {cls.MIN_SIZE_PX}x{cls.MIN_SIZE_PX} px minimum: {width_px:.1f}x{height_px:.1f} px"
            )

        # 4. Duplicate and conflicting class audit against existing boxes
        if existing_boxes:
            for other in existing_boxes:
                other_box = other.get("bbox")
                other_cls = other.get("class_name")
                if not other_box or len(other_box) != 4:
                    continue

                iou = compute_iou([x1, y1, x2, y2], other_box)

                # Duplicate detection (> 0.95 IoU)
                if iou >= cls.DUPLICATE_IOU_THRESHOLD:
                    issues.append(
                        f"Duplicate bounding box detected with IoU={iou:.3f} >= {cls.DUPLICATE_IOU_THRESHOLD}"
                    )

                # Conflicting class labels on overlapping object (> 0.80 IoU)
                elif iou >= cls.CONFLICT_IOU_THRESHOLD and other_cls and other_cls.lower() != class_name.lower():
                    issues.append(
                        f"Conflicting class labels for overlapping object (IoU={iou:.3f}): '{class_name}' vs '{other_cls}'"
                    )

        return (len(issues) == 0, issues)


class DatasetGenerator:
    """
    Generates versioned YOLO datasets partitioned into 70% Train / 15% Val / 15% Test.
    """

    def __init__(
        self,
        datasets_dir: Optional[Union[str, Path]] = None,
        classes_yaml_path: Optional[Union[str, Path]] = None,
    ):
        self.datasets_dir = Path(datasets_dir) if datasets_dir else DATASETS_DIR
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.classes_yaml_path = Path(classes_yaml_path) if classes_yaml_path else CLASSES_YAML_PATH
        self.class_to_id, self.id_to_class = self._load_master_classes()

    def _load_master_classes(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Loads master surveillance classes from config/classes.yaml."""
        if self.classes_yaml_path.is_file():
            try:
                with open(self.classes_yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                raw_classes = data.get("classes", {})
                id_to_cls = {int(k): str(v).lower().strip() for k, v in raw_classes.items()}
                cls_to_id = {v: k for k, v in id_to_cls.items()}
                return cls_to_id, id_to_cls
            except Exception:
                pass

        # Default master class fallback
        default_names = [
            "person", "car", "truck", "bus", "motorcycle",
            "bicycle", "animal", "backpack", "bag"
        ]
        cls_to_id = {name: i for i, name in enumerate(default_names)}
        id_to_cls = {i: name for i, name in enumerate(default_names)}
        return cls_to_id, id_to_cls

    def bbox_to_yolo_format(
        self,
        bbox: List[float],
        img_w: float,
        img_h: float,
        class_name: str,
    ) -> str:
        """Converts [x1, y1, x2, y2] to YOLO normalized string: 'cls_id x_c y_c w h'."""
        cls_id = self.class_to_id.get(class_name.lower().strip(), 0)
        x1, y1, x2, y2 = bbox

        x_c = ((x1 + x2) / 2.0) / img_w
        y_c = ((y1 + y2) / 2.0) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        # Clamp normalized coords to [0.0, 1.0]
        x_c = max(0.0, min(1.0, x_c))
        y_c = max(0.0, min(1.0, y_c))
        w = max(0.001, min(1.0, w))
        h = max(0.001, min(1.0, h))

        return f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

    def create_dataset_version(
        self,
        dataset_version: str,
        candidates: List[Dict[str, Any]],
        frame_provider_func: Optional[Callable[[Dict[str, Any]], Optional[np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """
        Builds a versioned dataset on disk partitioned into:
          - 70% Train
          - 15% Val
          - 15% Holdout Test (segregated benchmark)

        Directory Layout:
          training_lab/datasets/{dataset_version}/
            ├── data.yaml
            ├── dataset_manifest.json
            ├── images/ (train/, val/, test/)
            └── labels/ (train/, val/, test/)
        """
        version_dir = self.datasets_dir / dataset_version
        if version_dir.is_dir():
            shutil.rmtree(version_dir)

        # Create directory hierarchy
        for split in ["train", "val", "test"]:
            (version_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (version_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # 1. Audit Candidates via DataQualityGate
        audited_candidates: List[Dict[str, Any]] = []
        frame_group_audit: Dict[str, List[Dict[str, Any]]] = {}

        for c in candidates:
            bbox = c.get("bbox")
            cls_name = c.get("class_name", "person")
            shape = c.get("image_shape", (480, 640, 3))
            frame_key = f"{c.get('scenario_id', 'SCN')}_{c.get('camera_id', 'CAM')}_{c.get('frame_idx', 0)}"

            existing_on_frame = frame_group_audit.get(frame_key, [])
            is_valid, issues = DataQualityGate.audit_annotation(
                bbox=bbox,
                image_shape=shape,
                class_name=cls_name,
                existing_boxes=existing_on_frame,
            )

            if is_valid:
                audited_candidates.append(c)
                if frame_key not in frame_group_audit:
                    frame_group_audit[frame_key] = []
                frame_group_audit[frame_key].append({"bbox": bbox, "class_name": cls_name})

        total_samples = len(audited_candidates)
        if total_samples == 0:
            raise ValueError("No valid candidates passed DataQualityGate auditing.")

        # 2. Split Partitioning: Video / Scenario-Level Segregation (Zero Data Leakage)
        # Group candidates by unique video_id or scenario_id to prevent frame leakage
        video_groups: Dict[str, List[Dict[str, Any]]] = {}
        for c in audited_candidates:
            v_id = str(c.get("video_id") or c.get("scenario_id") or f"{c.get('scenario_id')}_{c.get('camera_id')}")
            if v_id not in video_groups:
                video_groups[v_id] = []
            video_groups[v_id].append(c)

        unique_videos = sorted(list(video_groups.keys()))
        splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        videos_per_split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

        if len(unique_videos) >= 3:
            split_strategy = "VIDEO_SCENARIO_LEVEL"
            n_test_v = max(1, int(round(len(unique_videos) * 0.15)))
            n_val_v = max(1, int(round(len(unique_videos) * 0.15)))
            n_train_v = len(unique_videos) - n_val_v - n_test_v
            if n_train_v < 1:
                n_train_v = 1
                if n_test_v > 1:
                    n_test_v -= 1
                elif n_val_v > 1:
                    n_val_v -= 1

            videos_per_split["train"] = unique_videos[:n_train_v]
            videos_per_split["val"] = unique_videos[n_train_v:n_train_v + n_val_v]
            videos_per_split["test"] = unique_videos[n_train_v + n_val_v:]

            for v in videos_per_split["train"]:
                splits["train"].extend(video_groups[v])
            for v in videos_per_split["val"]:
                splits["val"].extend(video_groups[v])
            for v in videos_per_split["test"]:
                splits["test"].extend(video_groups[v])
        else:
            # Fallback for single/dual video datasets: temporal chunking
            split_strategy = "TEMPORAL_SEQUENCE_LEVEL"
            if total_samples >= 3:
                n_test = max(1, int(round(total_samples * 0.15)))
                n_val = max(1, int(round(total_samples * 0.15)))
                n_train = total_samples - n_val - n_test
                if n_train < 1:
                    n_train = 1
                    if n_test > 1:
                        n_test -= 1
                    elif n_val > 1:
                        n_val -= 1
            elif total_samples == 2:
                n_train, n_val, n_test = 1, 0, 1
            else:
                n_train, n_val, n_test = 1, 0, 0

            splits["train"] = audited_candidates[:n_train]
            splits["val"] = audited_candidates[n_train:n_train + n_val]
            splits["test"] = audited_candidates[n_train + n_val:]
            videos_per_split["train"] = list({c.get("video_id") or c.get("scenario_id", "V1") for c in splits["train"]})
            videos_per_split["val"] = list({c.get("video_id") or c.get("scenario_id", "V1") for c in splits["val"]})
            videos_per_split["test"] = list({c.get("video_id") or c.get("scenario_id", "V1") for c in splits["test"]})

        train_samples = splits["train"]
        val_samples = splits["val"]
        test_samples = splits["test"]

        class_distribution: Dict[str, int] = {}
        source_scenarios: set = set()
        source_cameras: set = set()

        # 3. Write Images and Labels for each split
        for split_name, sample_list in splits.items():
            for idx, item in enumerate(sample_list):
                sample_id = f"{dataset_version}_{split_name}_{idx + 1:04d}"
                cls_name = item.get("class_name", "person").lower()
                bbox = item.get("bbox")
                scn_id = item.get("scenario_id", "SCN_01")
                cam_id = item.get("camera_id", "CAM_01")

                source_scenarios.add(scn_id)
                source_cameras.add(cam_id)
                class_distribution[cls_name] = class_distribution.get(cls_name, 0) + 1

                # Obtain Image
                img = None
                if frame_provider_func:
                    try:
                        img = frame_provider_func(item)
                    except Exception:
                        img = None

                if img is None and "frame" in item and isinstance(item["frame"], np.ndarray):
                    img = item["frame"]

                # If crop_path exists on disk, use crop or composite frame
                if img is None and item.get("crop_path"):
                    crop_file = ROOT_DIR / item["crop_path"]
                    if crop_file.is_file():
                        img = cv2.imread(str(crop_file))

                # Default fallback frame generator
                if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                    img = np.full((480, 640, 3), 42, dtype=np.uint8)
                    if bbox:
                        bx1, by1, bx2, by2 = map(int, bbox)
                        cv2.rectangle(img, (bx1, by1), (bx2, by2), (80, 120, 180), -1)

                h, w = img.shape[:2]

                # Save Image
                img_path = version_dir / "images" / split_name / f"{sample_id}.jpg"
                cv2.imwrite(str(img_path), img)

                # Save Normalized Label
                yolo_line = self.bbox_to_yolo_format(bbox, float(w), float(h), cls_name)
                lbl_path = version_dir / "labels" / split_name / f"{sample_id}.txt"
                with open(lbl_path, "w", encoding="utf-8") as lf:
                    lf.write(f"{yolo_line}\n")

        # 4. Generate data.yaml
        data_yaml_dict = {
            "path": str(version_dir.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in sorted(self.id_to_class.items())},
        }
        data_yaml_path = version_dir / "data.yaml"
        with open(data_yaml_path, "w", encoding="utf-8") as yf:
            yaml.dump(data_yaml_dict, yf, default_flow_style=False, sort_keys=False)

        # 5. Compute Holdout Test Set Hash for integrity auditing
        test_files = sorted((version_dir / "images" / "test").glob("*.jpg"))
        hasher = hashlib.sha256()
        for tf in test_files:
            hasher.update(tf.name.encode("utf-8"))
            lbl_f = version_dir / "labels" / "test" / f"{tf.stem}.txt"
            if lbl_f.is_file():
                hasher.update(lbl_f.read_bytes())
        holdout_hash = hasher.hexdigest()[:16]

        # 6. Generate dataset_manifest.json
        manifest = {
            "dataset_version": dataset_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": total_samples,
            "split_strategy": split_strategy,
            "videos_per_split": videos_per_split,
            "splits": {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
            "class_distribution": class_distribution,
            "source_scenarios": sorted(list(source_scenarios)),
            "source_cameras": sorted(list(source_cameras)),
            "holdout_test_set_hash": holdout_hash,
            "data_yaml_path": str(data_yaml_path.resolve()).replace("\\", "/"),
        }
        manifest_path = version_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

        return manifest


dataset_generator = DatasetGenerator()


if __name__ == "__main__":
    print("[DatasetGenerator] Testing DataQualityGate & DatasetGenerator...")
    sample_cands = [
        {"scenario_id": "SCN_01", "camera_id": "CAM_01", "frame_idx": 10, "class_name": "person", "bbox": [60.0, 188.0, 91.0, 293.0]},
        {"scenario_id": "SCN_02", "camera_id": "CAM_02", "frame_idx": 20, "class_name": "car", "bbox": [120.0, 200.0, 240.0, 310.0]},
        {"scenario_id": "SCN_03", "camera_id": "CAM_03", "frame_idx": 30, "class_name": "backpack", "bbox": [306.0, 260.0, 360.0, 319.0]},
        {"scenario_id": "SCN_04", "camera_id": "CAM_04", "frame_idx": 40, "class_name": "person", "bbox": [480.0, 226.0, 505.0, 257.0]},
        {"scenario_id": "SCN_05", "camera_id": "CAM_05", "frame_idx": 50, "class_name": "motorcycle", "bbox": [30.0, 255.0, 110.0, 346.0]},
        {"scenario_id": "SCN_01", "camera_id": "CAM_01", "frame_idx": 60, "class_name": "truck", "bbox": [150.0, 180.0, 280.0, 300.0]},
        {"scenario_id": "SCN_02", "camera_id": "CAM_02", "frame_idx": 70, "class_name": "person", "bbox": [80.0, 150.0, 120.0, 250.0]},
    ]
    res = dataset_generator.create_dataset_version("dataset_v001_demo", sample_cands)
    print(f"  Created dataset version: {res['dataset_version']} with {res['total_samples']} samples. Splits: {res['splits']}")
