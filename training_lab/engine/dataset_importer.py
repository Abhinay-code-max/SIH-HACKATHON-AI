"""
Real-World Dataset Ingestion Engine.
Supports ingestion of standard public and tactical surveillance datasets:
  - OD-VIRAT (Surveillance events & activity tracks)
  - UA-DETRAC (Vehicle surveillance & border checkpoint traffic)
  - MOT17 (Multi-Object Tracking pedestrian challenge format)
  - Custom Border CCTV (VOC XML, COCO JSON, CSV formats)

Converts annotations to standardized BORDER SENTINEL 9 master classes,
validates coordinates via DataQualityGate, attaches video/sequence metadata
for data leakage protection, and exports versioned YOLO datasets.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.dataset_generator import DataQualityGate, DatasetGenerator

DATASETS_DIR = ROOT_DIR / "training_lab" / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


class DatasetFormat:
    VIRAT = "VIRAT"
    UA_DETRAC = "UA_DETRAC"
    MOT17 = "MOT17"
    CUSTOM_CCTV = "CUSTOM_CCTV"


# Standardized Mapping from external taxonomies to BORDER SENTINEL 9 Master Classes
TAXONOMY_MAPPINGS: Dict[str, Dict[str, str]] = {
    DatasetFormat.VIRAT: {
        "person": "person",
        "pedestrian": "person",
        "human": "person",
        "vehicle": "civilian_vehicle",
        "car": "civilian_vehicle",
        "truck": "military_vehicle",
        "bus": "civilian_vehicle",
        "bike": "civilian_vehicle",
        "bag": "backpack",
        "backpack": "backpack",
        "object": "backpack",
    },
    DatasetFormat.UA_DETRAC: {
        "car": "car",
        "van": "car",
        "bus": "bus",
        "others": "car",
        "truck": "truck",
    },
    DatasetFormat.MOT17: {
        "1": "person",  # Pedestrian
        "2": "person",  # Person on vehicle
        "7": "civilian_vehicle",  # Static person / car
        "pedestrian": "person",
        "person": "person",
        "static_person": "person",
    },
    DatasetFormat.CUSTOM_CCTV: {
        "person": "person",
        "soldier": "patrol_unit",
        "patrol": "patrol_unit",
        "military_truck": "military_vehicle",
        "armored_car": "military_vehicle",
        "civilian_car": "civilian_vehicle",
        "vehicle": "civilian_vehicle",
        "rifle": "weapon",
        "weapon": "weapon",
        "drone": "drone",
        "uav": "drone",
        "dog": "animal",
        "camel": "animal",
        "backpack": "backpack",
        "camera": "optical_device",
        "binoculars": "optical_device",
    },
}


class DatasetImporter:
    """
    Ingests and normalizes external surveillance datasets into the Training Lab.
    Enforces quality filtering and video metadata tagging.
    """

    def __init__(self, datasets_dir: Optional[Union[str, Path]] = None):
        self.datasets_dir = Path(datasets_dir) if datasets_dir else DATASETS_DIR
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.generator = DatasetGenerator(self.datasets_dir)

    @classmethod
    def map_class(cls, raw_label: str, fmt: str) -> str:
        """Maps an external label string/ID to BORDER SENTINEL's master class.
        Returns 'unknown' for unmapped labels so DataQualityGate can audit and reject it.
        """
        mapping = TAXONOMY_MAPPINGS.get(fmt, {})
        key = str(raw_label).strip().lower()
        if key in mapping:
            return mapping[key]
        return "unknown"

    def parse_virat(
        self,
        content: Union[str, Path, List[Dict[str, Any]]],
        video_id: str = "VIRAT_S_0001",
        image_shape: Tuple[int, int, int] = (480, 640, 3),
    ) -> List[Dict[str, Any]]:
        """
        Parses OD-VIRAT style annotations.
        Format: track_id, length, frame_idx, top_left_x, top_left_y, width, height, [class_name]
        """
        candidates: List[Dict[str, Any]] = []

        if isinstance(content, (str, Path)) and Path(content).is_file():
            text = Path(content).read_text(encoding="utf-8")
            lines = text.strip().splitlines()
        elif isinstance(content, str):
            lines = content.strip().splitlines()
        elif isinstance(content, list):
            for item in content:
                bbox = item.get("bbox", [0, 0, 0, 0])
                raw_cls = item.get("class", item.get("class_name", "person"))
                mapped_cls = self.map_class(raw_cls, DatasetFormat.VIRAT)
                candidates.append({
                    "scenario_id": item.get("scenario_id", "SCN_VIRAT"),
                    "video_id": item.get("video_id", video_id),
                    "camera_id": item.get("camera_id", "CAM_VIRAT"),
                    "frame_idx": int(item.get("frame_idx", 0)),
                    "bbox": [float(b) for b in bbox],
                    "class_name": mapped_cls,
                    "confidence": float(item.get("confidence", 1.0)),
                    "image_shape": image_shape,
                })
            return candidates
        else:
            lines = []

        for line in lines:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 7:
                try:
                    f_idx = int(parts[2])
                    x1 = float(parts[3])
                    y1 = float(parts[4])
                    w = float(parts[5])
                    h = float(parts[6])
                    raw_cls = parts[7] if len(parts) > 7 else "person"
                    mapped_cls = self.map_class(raw_cls, DatasetFormat.VIRAT)
                    candidates.append({
                        "scenario_id": "SCN_VIRAT",
                        "video_id": video_id,
                        "camera_id": "CAM_VIRAT",
                        "frame_idx": f_idx,
                        "bbox": [x1, y1, x1 + w, y1 + h],
                        "class_name": mapped_cls,
                        "confidence": 1.0,
                        "image_shape": image_shape,
                    })
                except Exception:
                    continue

        return candidates

    def parse_ua_detrac(
        self,
        xml_content: Union[str, Path],
        video_id: Optional[str] = None,
        image_shape: Tuple[int, int, int] = (540, 960, 3),
        images_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parses UA-DETRAC XML tracking annotations.
        Enforces true default image dimensions (540, 960, 3) and clips bounding boxes.
        """
        candidates: List[Dict[str, Any]] = []

        if isinstance(xml_content, Path) or (isinstance(xml_content, str) and Path(xml_content).is_file()):
            tree = ET.parse(xml_content)
            root = tree.getroot()
        else:
            root = ET.fromstring(str(xml_content))

        seq_name = root.attrib.get("name", "")
        effective_video_id = video_id or (f"DETRAC_{seq_name}" if seq_name else "DETRAC_MVI_20011")

        # Dynamically infer image dimensions if images_dir is available
        actual_shape = image_shape
        if images_dir:
            images_path = Path(images_dir)
            test_names = [seq_name, effective_video_id, str(effective_video_id).replace("DETRAC_", "")]
            for s in test_names:
                if not s:
                    continue
                sample_img_path = images_path / s / "img00001.jpg"
                if sample_img_path.is_file():
                    img = cv2.imread(str(sample_img_path))
                    if img is not None:
                        actual_shape = img.shape
                    break

        img_h, img_w = int(actual_shape[0]), int(actual_shape[1])

        # Pre-index existing frames for the sequence if images_dir is provided
        valid_frame_nums: Optional[Set[int]] = None
        if images_dir:
            seq_disk_names = [seq_name, effective_video_id, str(effective_video_id).replace("DETRAC_", "")]
            for s in seq_disk_names:
                if not s:
                    continue
                s_dir = Path(images_dir) / s
                if s_dir.is_dir():
                    valid_frame_nums = set()
                    for img_p in s_dir.glob("img*.jpg"):
                        stem = img_p.stem.replace("img", "")
                        if stem.isdigit():
                            valid_frame_nums.add(int(stem))
                    break

        frames = root.findall(".//frame")
        for frame_elem in frames:
            frame_num = int(frame_elem.attrib.get("num", 1))
            if max_frames is not None and frame_num > max_frames:
                continue

            # If images_dir is known, skip frames that have no physical image on disk
            if valid_frame_nums is not None and frame_num not in valid_frame_nums:
                continue

            for target in frame_elem.findall(".//target"):
                box_elem = target.find("box")
                if box_elem is None:
                    continue
                left = float(box_elem.attrib.get("left", 0))
                top = float(box_elem.attrib.get("top", 0))
                width = float(box_elem.attrib.get("width", 0))
                height = float(box_elem.attrib.get("height", 0))

                attr_elem = target.find("attribute")
                vehicle_type = "car"
                if attr_elem is not None:
                    vehicle_type = attr_elem.attrib.get("vehicle_type", "car")

                mapped_cls = self.map_class(vehicle_type, DatasetFormat.UA_DETRAC)

                # 1-based (MATLAB continuous) to 0-based pixel continuous coordinate transformation
                x1 = max(0.0, min(float(img_w), left - 1.0))
                y1 = max(0.0, min(float(img_h), top - 1.0))
                x2 = max(0.0, min(float(img_w), x1 + width))
                y2 = max(0.0, min(float(img_h), y1 + height))

                candidates.append({
                    "scenario_id": f"SCN_{seq_name or 'DETRAC'}",
                    "video_id": effective_video_id,
                    "camera_id": "CAM_DETRAC",
                    "frame_idx": frame_num,
                    "bbox": [x1, y1, x2, y2],
                    "class_name": mapped_cls,
                    "confidence": 1.0,
                    "image_shape": actual_shape,
                })

        return candidates

    def parse_mot17(
        self,
        content: Union[str, Path],
        video_id: str = "MOT17_02_DPM",
        image_shape: Tuple[int, int, int] = (480, 640, 3),
    ) -> List[Dict[str, Any]]:
        """
        Parses MOT17 CSV lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """
        candidates: List[Dict[str, Any]] = []

        if isinstance(content, (str, Path)) and Path(content).is_file():
            text = Path(content).read_text(encoding="utf-8")
        else:
            text = str(content)

        for line in text.strip().splitlines():
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 7:
                try:
                    f_idx = int(parts[0])
                    x1 = float(parts[2])
                    y1 = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    conf = float(parts[6])
                    if conf > 0.0:
                        candidates.append({
                            "scenario_id": "SCN_MOT17",
                            "video_id": video_id,
                            "camera_id": "CAM_MOT17",
                            "frame_idx": f_idx,
                            "bbox": [x1, y1, x1 + w, y1 + h],
                            "class_name": "person",
                            "confidence": min(1.0, max(0.0, conf)),
                            "image_shape": image_shape,
                        })
                except Exception:
                    continue

        return candidates

    def parse_custom_cctv(
        self,
        records: List[Dict[str, Any]],
        video_id: str = "CCTV_BORDER_SEC_4",
        image_shape: Tuple[int, int, int] = (480, 640, 3),
    ) -> List[Dict[str, Any]]:
        """
        Parses custom Border CCTV structured annotations.
        """
        candidates: List[Dict[str, Any]] = []
        for r in records:
            raw_cls = r.get("class_name", r.get("label", "person"))
            mapped_cls = self.map_class(raw_cls, DatasetFormat.CUSTOM_CCTV)
            candidates.append({
                "scenario_id": r.get("scenario_id", "SCN_BORDER_CCTV"),
                "video_id": r.get("video_id", video_id),
                "camera_id": r.get("camera_id", "CAM_01"),
                "frame_idx": int(r.get("frame_idx", 0)),
                "bbox": [float(b) for b in r.get("bbox", [10, 10, 50, 50])],
                "class_name": mapped_cls,
                "confidence": float(r.get("confidence", 1.0)),
                "image_shape": image_shape,
            })
        return candidates

    @staticmethod
    def compute_ua_detrac_source_accounting(
        sequences: List[str],
        annotations_dir: Union[str, Path],
        images_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Computes exact, auditable source-accounting metrics for UA-DETRAC dataset sequences:
          - total_xml_frames: Number of frame entries declared by the selected XML files.
          - total_physical_frames_on_disk: Number of JPEG frames physically present on disk.
          - unannotated_disk_frames_skipped: Frames physically present that lack XML annotations.
          - missing_physical_frames_skipped: XML frame entries that lack physical JPEG files.
          - missing_physical_frames_by_sequence: Detailed per-sequence breakdown of missing frames.
        """
        ann_path = Path(annotations_dir)
        img_path = Path(images_dir)

        total_xml_frames = 0
        total_physical_frames = 0
        missing_physical_frames = 0
        unannotated_disk_frames = 0
        missing_by_seq: Dict[str, Dict[str, Any]] = {}

        for seq in sorted(list(set(sequences))):
            seq_clean = str(seq).replace("DETRAC_", "").replace("SCN_", "").strip()

            # 1. Inspect on-disk image frames
            s_img_dir = None
            for cand in [img_path / seq_clean, img_path / seq, img_path / "DETRAC-Images" / seq_clean]:
                if cand.is_dir():
                    s_img_dir = cand
                    break

            disk_fnums: Set[int] = set()
            if s_img_dir:
                for p in s_img_dir.glob("img*.jpg"):
                    stem = p.stem.replace("img", "")
                    if stem.isdigit():
                        disk_fnums.add(int(stem))

            n_disk = len(disk_fnums)
            total_physical_frames += n_disk

            # 2. Inspect XML frames
            s_xml_file = None
            for cand in [ann_path / f"{seq_clean}.xml", ann_path / f"{seq}.xml"]:
                if cand.is_file():
                    s_xml_file = cand
                    break

            xml_fnums: Set[int] = set()
            if s_xml_file:
                tree = ET.parse(str(s_xml_file))
                for f in tree.findall(".//frame"):
                    num = int(f.attrib.get("num", 1))
                    xml_fnums.add(num)

            n_xml = len(xml_fnums)
            total_xml_frames += n_xml

            # Missing source images that XML expected
            missing_fnums = sorted(list(xml_fnums - disk_fnums))
            if missing_fnums:
                missing_physical_frames += len(missing_fnums)
                missing_by_seq[seq_clean] = {
                    "xml_frames": n_xml,
                    "available_frames": n_disk,
                    "missing_frames": len(missing_fnums),
                    "missing_frame_range": [min(missing_fnums), max(missing_fnums)],
                }

            # Physical frames on disk that XML did not annotate
            unannotated = disk_fnums - xml_fnums
            unannotated_disk_frames += len(unannotated)

        return {
            "total_xml_frames": total_xml_frames,
            "total_physical_frames_on_disk": total_physical_frames,
            "unannotated_disk_frames_skipped": unannotated_disk_frames,
            "missing_physical_frames_skipped": missing_physical_frames,
            "missing_physical_frames_by_sequence": missing_by_seq,
        }

    def import_and_generate(
        self,
        dataset_version: str,
        candidates: List[Dict[str, Any]],
        frame_provider_func=None,
        allow_synthetic_fallback: bool = True,
        clip_out_of_bounds: bool = True,
        video_splits: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
        source_accounting: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs candidates through DataQualityGate and generates an audited,
        versioned YOLO dataset with video/scenario metadata attached.
        """
        for c in candidates:
            if "video_id" not in c:
                c["video_id"] = f"{c.get('scenario_id', 'SCN')}_{c.get('camera_id', 'CAM')}"

        manifest = self.generator.create_dataset_version(
            dataset_version=dataset_version,
            candidates=candidates,
            frame_provider_func=frame_provider_func,
            allow_synthetic_fallback=allow_synthetic_fallback,
            clip_out_of_bounds=clip_out_of_bounds,
            video_splits=video_splits,
            source_accounting=source_accounting,
        )

        manifest["imported_at"] = datetime.now(timezone.utc).isoformat()
        manifest["dataset_format"] = "NORMALIZED_BORDER_SENTINEL_YOLO"
        return manifest


class UADetracFrameProvider:
    """
    Frame provider for UA-DETRAC dataset sequences on local disk.
    Resolves sequence images in the format:
      <images_dir>/<sequence_name>/img%05d.jpg
    Returns BGR numpy array or None.
    """

    def __init__(self, images_dir: Union[str, Path]):
        self.images_dir = Path(images_dir)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"UA-DETRAC images directory does not exist: {self.images_dir}")

    def resolve_frame_path(self, video_id: str, frame_idx: int) -> Path:
        # Strip common prefixes like DETRAC_ or SCN_
        seq_name = str(video_id).replace("DETRAC_", "").replace("SCN_", "").strip()
        candidates = [
            self.images_dir / seq_name / f"img{frame_idx:05d}.jpg",
            self.images_dir / video_id / f"img{frame_idx:05d}.jpg",
            self.images_dir / "DETRAC-Images" / seq_name / f"img{frame_idx:05d}.jpg",
            self.images_dir / "DETRAC-Images" / video_id / f"img{frame_idx:05d}.jpg",
        ]
        for c in candidates:
            if c.is_file():
                return c
        return candidates[0]

    def has_sequence(self, video_id: str) -> bool:
        """Checks if physical images directory exists for video_id."""
        seq_name = str(video_id).replace("DETRAC_", "").replace("SCN_", "").strip()
        return (
            (self.images_dir / seq_name).is_dir()
            or (self.images_dir / video_id).is_dir()
            or (self.images_dir / "DETRAC-Images" / seq_name).is_dir()
        )

    def get_frame_path(
        self,
        frame_item_or_vid: Union[Dict[str, Any], str],
        frame_idx: Optional[int] = None,
    ) -> Optional[Path]:
        """Returns the physical Path of the frame on disk, or None if missing."""
        if isinstance(frame_item_or_vid, dict):
            v_id = frame_item_or_vid.get("video_id", "")
            f_idx = int(frame_item_or_vid.get("frame_idx", 1))
        else:
            v_id = str(frame_item_or_vid)
            f_idx = int(frame_idx if frame_idx is not None else 1)

        p = self.resolve_frame_path(v_id, f_idx)
        return p if p.is_file() else None

    def get_frame(
        self,
        frame_item_or_vid: Union[Dict[str, Any], str],
        frame_idx: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if isinstance(frame_item_or_vid, dict):
            v_id = frame_item_or_vid.get("video_id", "")
            f_idx = int(frame_item_or_vid.get("frame_idx", 1))
        else:
            v_id = str(frame_item_or_vid)
            f_idx = int(frame_idx if frame_idx is not None else 1)

        p = self.resolve_frame_path(v_id, f_idx)
        if not p.is_file():
            return None
        return cv2.imread(str(p))

    def __call__(
        self,
        frame_item_or_vid: Union[Dict[str, Any], str],
        frame_idx: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        return self.get_frame(frame_item_or_vid, frame_idx)

