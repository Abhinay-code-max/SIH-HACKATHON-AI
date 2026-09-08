"""
Dataset Generation Engine & Data Quality Gate.
Transforms raw bounding box annotations and surveillance image frames into
standardized YOLO format (normalized coordinates, class IDs, data.yaml),
enforcing quality gates, video-aware frame grouping, and strict sequence-level
partitioning (70% Train / 15% Val / 15% Unseen Holdout Test).
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
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
    if inter <= 0.0:
        return 0.0

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class DataQualityGate:
    """
    Validates annotation quality, rejects corrupted boundaries, filters sub-pixel
    noise, enforces master class validity, and flags duplicates and conflicting labels.
    """

    MIN_SIZE_PX = 10.0
    DUPLICATE_IOU_THRESHOLD = 0.95
    CONFLICT_IOU_THRESHOLD = 0.80

    @classmethod
    def clip_box(
        cls,
        bbox: List[float],
        image_shape: Tuple[int, ...],
    ) -> Tuple[List[float], bool]:
        """
        Clips [x1, y1, x2, y2] to image bounds [0, w] and [0, h].
        Returns (clipped_bbox, was_clipped).
        """
        h, w = float(image_shape[0]), float(image_shape[1])
        x1, y1, x2, y2 = map(float, bbox)
        cx1 = max(0.0, min(w, x1))
        cy1 = max(0.0, min(h, y1))
        cx2 = max(0.0, min(w, x2))
        cy2 = max(0.0, min(h, y2))
        was_clipped = (cx1 != x1) or (cy1 != y1) or (cx2 != x2) or (cy2 != y2)
        return [round(cx1, 2), round(cy1, 2), round(cx2, 2), round(cy2, 2)], was_clipped

    @classmethod
    def audit_annotation(
        cls,
        bbox: List[float],
        image_shape: Tuple[int, ...],
        class_name: str,
        existing_boxes: Optional[List[Dict[str, Any]]] = None,
        allowed_classes: Optional[Set[str]] = None,
        clip_to_bounds: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Audits an individual bounding box against data quality standards.

        Args:
            bbox: [x1, y1, x2, y2]
            image_shape: (height, width) or (h, w, c)
            class_name: target class name
            existing_boxes: list of already audited annotations on the same frame
            allowed_classes: optional set of allowed master class names
            clip_to_bounds: if True, clips box to [0, w] and [0, h] before validation

        Returns:
            Tuple of (is_valid: bool, issues: List[str])
        """
        is_valid, issues, _ = cls.audit_and_clean_annotation(
            bbox=bbox,
            image_shape=image_shape,
            class_name=class_name,
            existing_boxes=existing_boxes,
            allowed_classes=allowed_classes,
            clip_to_bounds=clip_to_bounds,
        )
        return is_valid, issues

    @classmethod
    def audit_and_clean_annotation(
        cls,
        bbox: List[float],
        image_shape: Tuple[int, ...],
        class_name: str,
        existing_boxes: Optional[List[Dict[str, Any]]] = None,
        allowed_classes: Optional[Set[str]] = None,
        clip_to_bounds: bool = True,
    ) -> Tuple[bool, List[str], List[float]]:
        """
        Audits and optionally clips bounding boxes against boundary and geometry standards.

        Returns:
            Tuple of (is_valid: bool, issues: List[str], cleaned_bbox: List[float])
        """
        issues: List[str] = []

        if not bbox or len(bbox) != 4:
            return False, [f"Invalid bbox length: expected 4 coordinates, got {bbox}"], bbox

        h, w = float(image_shape[0]), float(image_shape[1])
        x1, y1, x2, y2 = map(float, bbox)

        # 1. Check master class validity
        if allowed_classes is not None:
            norm_cls = class_name.lower().strip()
            if norm_cls not in allowed_classes:
                issues.append(
                    f"Unknown/unregistered class: '{class_name}' is not in master classes: {sorted(list(allowed_classes))}"
                )

        # 2. Coordinate ordering and inverted geometry
        if x2 <= x1:
            issues.append(f"Inverted horizontal coordinates: x1={x1} >= x2={x2}")
        if y2 <= y1:
            issues.append(f"Inverted vertical coordinates: y1={y1} >= y2={y2}")

        # If inverted, cannot proceed with clipping
        if issues:
            return False, issues, bbox

        # 3. Boundary bounds check & clipping
        cleaned_box = [x1, y1, x2, y2]
        if clip_to_bounds:
            cleaned_box, was_clipped = cls.clip_box(bbox, image_shape)
            x1, y1, x2, y2 = cleaned_box
        else:
            if x1 < 0.0 or y1 < 0.0 or x2 > w or y2 > h:
                issues.append(
                    f"Coordinates out of bounds: [{x1}, {y1}, {x2}, {y2}] exceeds image dimensions ({w}x{h})"
                )

        # 4. Minimum dimension validation (after clipping)
        width_px = x2 - x1
        height_px = y2 - y1
        if width_px < cls.MIN_SIZE_PX or height_px < cls.MIN_SIZE_PX:
            issues.append(
                f"Bounding box smaller than {cls.MIN_SIZE_PX}x{cls.MIN_SIZE_PX} px minimum: {width_px:.1f}x{height_px:.1f} px"
            )

        # 5. Duplicate and conflicting class audit against existing boxes on same frame
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

        return (len(issues) == 0, issues, cleaned_box)


class DatasetGenerator:
    """
    Generates versioned YOLO datasets partitioned into 70% Train / 15% Val / 15% Test.
    Enforces video-aware frame grouping so multiple targets in one physical frame
    are properly consolidated into one image and multi-line label file.
    """

    # Master taxonomy synonym mappings for compatibility with external datasets
    SYNONYM_MAPPINGS: Dict[str, str] = {
        "civilian_vehicle": "car",
        "military_vehicle": "truck",
        "patrol_unit": "person",
        "soldier": "person",
        "human": "person",
        "pedestrian": "person",
        "vehicle": "car",
        "van": "car",
        "auto": "car",
        "automobile": "car",
        "lorry": "truck",
        "pickup": "truck",
    }

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
        default_names = [
            "person", "car", "truck", "bus", "motorcycle",
            "bicycle", "animal", "backpack", "bag"
        ]
        active_set = set(default_names)

        if self.classes_yaml_path.is_file():
            try:
                with open(self.classes_yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                raw_classes = data.get("classes", {})
                id_to_cls = {
                    int(k): str(v).lower().strip()
                    for k, v in raw_classes.items()
                    if str(v).lower().strip() in active_set
                }
                if id_to_cls:
                    cls_to_id = {v: k for k, v in id_to_cls.items()}
                    return cls_to_id, id_to_cls
            except Exception:
                pass

        cls_to_id = {name: i for i, name in enumerate(default_names)}
        id_to_cls = {i: name for i, name in enumerate(default_names)}
        return cls_to_id, id_to_cls

    @property
    def classes(self) -> List[str]:
        """Returns the list of valid master class names."""
        return list(self.class_to_id.keys())

    def resolve_class_id(self, class_name: str) -> int:
        """
        Resolves class name to integer class ID.
        Strictly prevents unknown classes from silently defaulting to 0 (person).
        """
        cls_key = str(class_name).lower().strip()
        if cls_key in self.class_to_id:
            return self.class_to_id[cls_key]

        # Check synonym mapping
        if cls_key in self.SYNONYM_MAPPINGS:
            resolved = self.SYNONYM_MAPPINGS[cls_key]
            if resolved in self.class_to_id:
                return self.class_to_id[resolved]

        raise ValueError(
            f"Unknown/unregistered class '{class_name}'. "
            f"Must be one of master classes: {sorted(list(self.class_to_id.keys()))}"
        )

    def bbox_to_yolo_format(
        self,
        bbox: List[float],
        img_w: float,
        img_h: float,
        class_name: str,
    ) -> str:
        """
        Converts [x1, y1, x2, y2] to YOLO normalized string: 'cls_id x_c y_c w h'.
        Raises ValueError if class_name is unrecognized.
        """
        cls_id = self.resolve_class_id(class_name)
        x1, y1, x2, y2 = bbox

        x_c = ((x1 + x2) / 2.0) / img_w
        y_c = ((y1 + y2) / 2.0) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        # Clamp normalized coords strictly to [0.0, 1.0]
        x_c = max(0.0, min(1.0, x_c))
        y_c = max(0.0, min(1.0, y_c))
        w = max(0.0001, min(1.0, w))
        h = max(0.0001, min(1.0, h))

        return f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

    def create_dataset_version(
        self,
        dataset_version: str,
        candidates: List[Dict[str, Any]],
        frame_provider_func: Optional[Callable[[Dict[str, Any]], Optional[np.ndarray]]] = None,
        allow_synthetic_fallback: bool = True,
        clip_out_of_bounds: bool = True,
        video_splits: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
        source_accounting: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Builds a versioned dataset on disk partitioned into:
          - 70% Train (or explicit split)
          - 15% Val (or explicit split)
          - 15% Holdout Test (segregated benchmark / unseen holdout)

        Groups all bounding boxes belonging to the same physical frame (video_id + frame_idx)
        so that one physical image has all its valid annotations in one .txt file.

        Args:
            dataset_version: Target dataset name / directory (e.g. 'dataset_v001')
            candidates: List of candidate annotation dicts
            frame_provider_func: Callable returning BGR np.ndarray given a frame dict
            allow_synthetic_fallback: If False, raises FileNotFoundError if real frame is missing
            clip_out_of_bounds: If True, boundary coordinates are clipped to image shape
            video_splits: Optional explicit sequence-to-split mapping (e.g. {'seq1': 'train', ...}
                          or {'train': ['seq1'], 'val': [...], 'test': [...]})
        """
        version_dir = self.datasets_dir / dataset_version
        if version_dir.is_dir():
            shutil.rmtree(version_dir)

        # Create directory hierarchy
        for split in ["train", "val", "test"]:
            (version_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (version_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        allowed_cls = set(self.class_to_id.keys()).union(set(self.SYNONYM_MAPPINGS.keys()))

        # 1. Audit Candidates via DataQualityGate with video-aware frame grouping
        audited_candidates: List[Dict[str, Any]] = []
        rejected_candidates: List[Dict[str, Any]] = []
        frame_group_audit: Dict[str, List[Dict[str, Any]]] = {}

        for c in candidates:
            bbox = c.get("bbox")
            cls_name = c.get("class_name", "person")
            shape = c.get("image_shape", (540, 960, 3))
            v_id = str(c.get("video_id") or c.get("scenario_id") or "VID_01")
            f_idx = int(c.get("frame_idx", 0))

            # Video-aware frame key prevents collisions across multiple sequences
            frame_key = f"{v_id}_f{f_idx:06d}"

            existing_on_frame = frame_group_audit.get(frame_key, [])
            is_valid, issues, cleaned_bbox = DataQualityGate.audit_and_clean_annotation(
                bbox=bbox,
                image_shape=shape,
                class_name=cls_name,
                existing_boxes=existing_on_frame,
                allowed_classes=allowed_cls,
                clip_to_bounds=clip_out_of_bounds,
            )

            if is_valid:
                c_clean = dict(c)
                c_clean["bbox"] = cleaned_bbox
                c_clean["video_id"] = v_id
                c_clean["frame_idx"] = f_idx
                # Normalize synonym to canonical master class
                norm_cls = str(cls_name).lower().strip()
                if norm_cls in self.SYNONYM_MAPPINGS:
                    c_clean["class_name"] = self.SYNONYM_MAPPINGS[norm_cls]
                audited_candidates.append(c_clean)

                if frame_key not in frame_group_audit:
                    frame_group_audit[frame_key] = []
                frame_group_audit[frame_key].append({"bbox": cleaned_bbox, "class_name": c_clean["class_name"]})
            else:
                rejected_candidates.append({
                    "video_id": v_id,
                    "frame_idx": f_idx,
                    "class_name": cls_name,
                    "bbox": bbox,
                    "issues": issues,
                })

        if not audited_candidates:
            raise ValueError(
                f"No valid candidates passed DataQualityGate auditing. "
                f"Rejected {len(rejected_candidates)} candidates: {rejected_candidates[:3]}"
            )

        # 2. Frame-Level Grouping (Multiple objects per physical frame)
        frames_by_video: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for c in audited_candidates:
            v_id = c["video_id"]
            f_idx = c["frame_idx"]
            if v_id not in frames_by_video:
                frames_by_video[v_id] = {}
            if f_idx not in frames_by_video[v_id]:
                frames_by_video[v_id][f_idx] = {
                    "video_id": v_id,
                    "frame_idx": f_idx,
                    "scenario_id": c.get("scenario_id", "SCN_01"),
                    "camera_id": c.get("camera_id", "CAM_01"),
                    "image_shape": c.get("image_shape", (540, 960, 3)),
                    "crop_path": c.get("crop_path"),
                    "frame": c.get("frame"),
                    "annotations": [],
                }
            frames_by_video[v_id][f_idx]["annotations"].append({
                "bbox": c["bbox"],
                "class_name": c["class_name"],
            })

        # 3. Split Partitioning: Video / Sequence-Level Segregation (Zero Data Leakage)
        unique_videos = sorted(list(frames_by_video.keys()))
        splits_frames: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        videos_per_split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

        if video_splits is not None:
            split_strategy = "EXPLICIT_SEQUENCE_LEVEL"

            if not isinstance(video_splits, dict):
                raise ValueError(
                    f"video_splits must be a dictionary, got {type(video_splits).__name__}"
                )

            norm_splits: Dict[str, str] = {}
            # Detect whether format is Dict[str, List[str]] (e.g. {'train': [...], ...})
            # or Dict[str, str] (e.g. {'seq1': 'train', ...})
            is_group_format = any(k in ("train", "val", "test") for k in video_splits.keys()) and any(
                isinstance(v, (list, tuple, set)) for v in video_splits.values()
            )

            if is_group_format:
                for s_name, seq_list in video_splits.items():
                    if s_name not in ("train", "val", "test"):
                        raise ValueError(
                            f"Invalid split name '{s_name}' in video_splits. Allowed split names are: 'train', 'val', 'test'"
                        )
                    for sid in seq_list:
                        s_key = str(sid).strip()
                        if s_key in norm_splits:
                            raise ValueError(
                                f"Duplicate/conflicting split assignment: sequence '{s_key}' is assigned to both '{norm_splits[s_key]}' and '{s_name}'"
                            )
                        norm_splits[s_key] = s_name
            else:
                for sid, s_name in video_splits.items():
                    s_key = str(sid).strip()
                    s_val = str(s_name).strip().lower()
                    if s_val not in ("train", "val", "test"):
                        raise ValueError(
                            f"Invalid split name '{s_name}' for sequence '{s_key}'. Allowed split names are: 'train', 'val', 'test'"
                        )
                    norm_splits[s_key] = s_val

            # Validate against discovered unique videos in candidates
            discovered_videos_set = set(unique_videos)
            specified_videos_set = set(norm_splits.keys())

            # 1. Unknown sequences check
            unknown_videos = sorted(list(specified_videos_set - discovered_videos_set))
            if unknown_videos:
                raise ValueError(
                    f"Unknown sequence(s) present in video_splits that do not exist in dataset candidates: {unknown_videos}"
                )

            # 2. Missing sequences check
            missing_videos = sorted(list(discovered_videos_set - specified_videos_set))
            if missing_videos:
                raise ValueError(
                    f"Missing sequence(s) in video_splits: {missing_videos} must be explicitly assigned to 'train', 'val', or 'test'"
                )

            # Deterministic, mutually disjoint sequence assignment
            videos_per_split["train"] = sorted([v for v in unique_videos if norm_splits[v] == "train"])
            videos_per_split["val"] = sorted([v for v in unique_videos if norm_splits[v] == "val"])
            videos_per_split["test"] = sorted([v for v in unique_videos if norm_splits[v] == "test"])

            for v in videos_per_split["train"]:
                splits_frames["train"].extend(list(frames_by_video[v].values()))
            for v in videos_per_split["val"]:
                splits_frames["val"].extend(list(frames_by_video[v].values()))
            for v in videos_per_split["test"]:
                splits_frames["test"].extend(list(frames_by_video[v].values()))

        elif len(unique_videos) >= 3:
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
                splits_frames["train"].extend(list(frames_by_video[v].values()))
            for v in videos_per_split["val"]:
                splits_frames["val"].extend(list(frames_by_video[v].values()))
            for v in videos_per_split["test"]:
                splits_frames["test"].extend(list(frames_by_video[v].values()))
        else:
            # Fallback for single/dual video datasets: temporal sequence partitioning
            split_strategy = "TEMPORAL_SEQUENCE_LEVEL"
            all_frames: List[Dict[str, Any]] = []
            for v in unique_videos:
                for f_idx in sorted(frames_by_video[v].keys()):
                    all_frames.append(frames_by_video[v][f_idx])

            total_frames = len(all_frames)
            if total_frames >= 3:
                n_test = max(1, int(round(total_frames * 0.15)))
                n_val = max(1, int(round(total_frames * 0.15)))
                n_train = total_frames - n_val - n_test
                if n_train < 1:
                    n_train = 1
                    if n_test > 1:
                        n_test -= 1
                    elif n_val > 1:
                        n_val -= 1
            elif total_frames == 2:
                n_train, n_val, n_test = 1, 0, 1
            else:
                n_train, n_val, n_test = 1, 0, 0

            splits_frames["train"] = all_frames[:n_train]
            splits_frames["val"] = all_frames[n_train:n_train + n_val]
            splits_frames["test"] = all_frames[n_train + n_val:]
            videos_per_split["train"] = sorted(list({f["video_id"] for f in splits_frames["train"]}))
            videos_per_split["val"] = sorted(list({f["video_id"] for f in splits_frames["val"]}))
            videos_per_split["test"] = sorted(list({f["video_id"] for f in splits_frames["test"]}))

        # 4. Write Images and Labels (One physical image per frame, all bounding boxes in one .txt file)
        class_distribution: Dict[str, int] = {}
        source_scenarios: set = set()
        source_cameras: set = set()
        frame_counts_per_split: Dict[str, int] = {}
        annotation_counts_per_split: Dict[str, int] = {}
        total_annotations_written = 0

        for split_name, frame_list in splits_frames.items():
            frame_counts_per_split[split_name] = len(frame_list)
            annotation_counts_per_split[split_name] = sum(len(f["annotations"]) for f in frame_list)

            for idx, frame_item in enumerate(frame_list):
                v_id = frame_item["video_id"]
                f_idx = frame_item["frame_idx"]
                sample_id = f"{dataset_version}_{split_name}_{v_id}_f{f_idx:06d}"
                scn_id = frame_item.get("scenario_id", "SCN_01")
                cam_id = frame_item.get("camera_id", "CAM_01")
                source_scenarios.add(scn_id)
                source_cameras.add(cam_id)

                # Obtain Image via provider, buffer, crop, or synthetic fallback
                img = None
                src_file_path = None

                # Check if provider can supply physical file path for byte-identical copy
                if frame_provider_func and hasattr(frame_provider_func, "get_frame_path"):
                    try:
                        src_file_path = frame_provider_func.get_frame_path(v_id, f_idx)
                    except Exception:
                        src_file_path = None

                if frame_provider_func:
                    try:
                        img = frame_provider_func(frame_item)
                    except Exception as e:
                        if not allow_synthetic_fallback:
                            raise FileNotFoundError(
                                f"Failed to load real frame for video '{v_id}' frame {f_idx}: {e}"
                            )
                        img = None

                if img is None and "frame" in frame_item and isinstance(frame_item["frame"], np.ndarray):
                    img = frame_item["frame"]

                if img is None and frame_item.get("crop_path"):
                    crop_file = ROOT_DIR / frame_item["crop_path"]
                    if crop_file.is_file():
                        img = cv2.imread(str(crop_file))
                        src_file_path = crop_file

                # Handle missing image
                if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                    if not allow_synthetic_fallback:
                        raise FileNotFoundError(
                            f"Real frame image not found for video '{v_id}' frame {f_idx}. "
                            f"Synthetic placeholder fallback is disabled for real datasets."
                        )
                    # Synthetic fallback only when explicitly permitted
                    shape = frame_item.get("image_shape", (540, 960, 3))
                    img = np.full((int(shape[0]), int(shape[1]), 3), 42, dtype=np.uint8)
                    for ann in frame_item["annotations"]:
                        bx1, by1, bx2, by2 = map(int, ann["bbox"])
                        cv2.rectangle(img, (bx1, by1), (bx2, by2), (80, 120, 180), -1)

                h, w = img.shape[:2]

                # Write Physical Image ONCE (byte-identical copy if source file is available)
                img_path = version_dir / "images" / split_name / f"{sample_id}.jpg"
                if src_file_path and Path(src_file_path).is_file():
                    shutil.copy2(str(src_file_path), str(img_path))
                else:
                    cv2.imwrite(str(img_path), img)

                # Write All Bounding Boxes for this frame to .txt label file
                lbl_path = version_dir / "labels" / split_name / f"{sample_id}.txt"
                with open(lbl_path, "w", encoding="utf-8") as lf:
                    for ann in frame_item["annotations"]:
                        cls_name = ann["class_name"]
                        bbox = ann["bbox"]
                        yolo_line = self.bbox_to_yolo_format(bbox, float(w), float(h), cls_name)
                        lf.write(f"{yolo_line}\n")
                        class_distribution[cls_name] = class_distribution.get(cls_name, 0) + 1
                        total_annotations_written += 1

        # 5. Generate data.yaml and dataset.yaml
        data_yaml_path = version_dir / "data.yaml"
        data_yaml_dict = {
            "path": str(version_dir.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in sorted(self.id_to_class.items())},
        }
        for yml_name in ("data.yaml", "dataset.yaml"):
            with open(version_dir / yml_name, "w", encoding="utf-8") as yf:
                yaml.dump(data_yaml_dict, yf, default_flow_style=False, sort_keys=False)

        # 6. Compute Holdout Test Set Hash for integrity auditing
        test_files = sorted((version_dir / "images" / "test").glob("*.jpg"))
        hasher = hashlib.sha256()
        for tf in test_files:
            hasher.update(tf.name.encode("utf-8"))
            lbl_f = version_dir / "labels" / "test" / f"{tf.stem}.txt"
            if lbl_f.is_file():
                hasher.update(lbl_f.read_bytes())
        holdout_hash = hasher.hexdigest()[:16]

        total_frames = sum(frame_counts_per_split.values())

        # 7. Generate dataset_manifest.json
        manifest = {
            "dataset_version": dataset_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": total_frames,
            "total_frames": total_frames,
            "total_annotations": total_annotations_written,
            "split_strategy": split_strategy,
            "videos_per_split": videos_per_split,
            "splits": frame_counts_per_split,
            "annotation_splits": annotation_counts_per_split,
            "class_distribution": class_distribution,
            "rejected_annotations_count": len(rejected_candidates),
            "source_scenarios": sorted(list(source_scenarios)),
            "source_cameras": sorted(list(source_cameras)),
            "holdout_test_set_hash": holdout_hash,
            "data_yaml_path": str(data_yaml_path.resolve()).replace("\\", "/"),
        }
        if source_accounting is not None:
            manifest["source_accounting"] = dict(source_accounting)

        manifest_path = version_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

        return manifest

    def generate_yolo_dataset(
        self,
        dataset_version: str,
        candidates: List[Dict[str, Any]],
        frame_provider_func: Optional[Callable[[Dict[str, Any]], Optional[np.ndarray]]] = None,
        allow_synthetic_fallback: bool = True,
        clip_out_of_bounds: bool = True,
        video_splits: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Alias for create_dataset_version to support dataset importers."""
        return self.create_dataset_version(
            dataset_version=dataset_version,
            candidates=candidates,
            frame_provider_func=frame_provider_func,
            allow_synthetic_fallback=allow_synthetic_fallback,
            clip_out_of_bounds=clip_out_of_bounds,
            video_splits=video_splits,
            **kwargs,
        )

    @classmethod
    def load_ua_detrac_splits(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        as_sequence_map: bool = True,
    ) -> Union[Dict[str, str], Dict[str, List[str]]]:
        """
        Loads the canonical UA-DETRAC 32 Train / 7 Val / 7 Holdout split definition
        from config/ua_detrac_splits.yaml without hardcoding sequences in generator logic.

        Args:
            config_path: Optional custom path to yaml file.
            as_sequence_map: If True, returns Dict[video_id, split_name].
                             If False, returns Dict[split_name, List[video_id]].
        """
        cfg_p = Path(config_path) if config_path else ROOT_DIR / "config" / "ua_detrac_splits.yaml"
        if not cfg_p.is_file():
            raise FileNotFoundError(f"UA-DETRAC split configuration not found: {cfg_p}")

        with open(cfg_p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        splits_data = data.get("splits", {})
        if not as_sequence_map:
            return {
                "train": list(splits_data.get("train", [])),
                "val": list(splits_data.get("val", [])),
                "test": list(splits_data.get("test", [])),
            }

        seq_map: Dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            for seq_id in splits_data.get(split_name, []):
                seq_map[str(seq_id).strip()] = split_name
        return seq_map


def load_ua_detrac_split_config(
    config_path: Optional[Union[str, Path]] = None,
    as_sequence_map: bool = True,
) -> Union[Dict[str, str], Dict[str, List[str]]]:
    """Helper function to load the canonical UA-DETRAC sequence split configuration."""
    return DatasetGenerator.load_ua_detrac_splits(config_path=config_path, as_sequence_map=as_sequence_map)


dataset_generator = DatasetGenerator()

