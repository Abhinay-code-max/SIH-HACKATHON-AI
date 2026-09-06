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
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

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
        "car": "civilian_vehicle",
        "van": "civilian_vehicle",
        "bus": "civilian_vehicle",
        "others": "civilian_vehicle",
        "truck": "military_vehicle",
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
        """Maps an external label string/ID to BORDER SENTINEL's master class."""
        mapping = TAXONOMY_MAPPINGS.get(fmt, {})
        key = str(raw_label).strip().lower()
        return mapping.get(key, "person")

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
        video_id: str = "DETRAC_MVI_20011",
        image_shape: Tuple[int, int, int] = (480, 640, 3),
    ) -> List[Dict[str, Any]]:
        """
        Parses UA-DETRAC XML tracking annotations.
        """
        candidates: List[Dict[str, Any]] = []

        if isinstance(xml_content, Path) or (isinstance(xml_content, str) and Path(xml_content).is_file()):
            tree = ET.parse(xml_content)
            root = tree.getroot()
        else:
            root = ET.fromstring(str(xml_content))

        for frame_elem in root.findall(".//frame"):
            frame_num = int(frame_elem.attrib.get("num", 1))
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
                candidates.append({
                    "scenario_id": "SCN_DETRAC",
                    "video_id": video_id,
                    "camera_id": "CAM_DETRAC",
                    "frame_idx": frame_num,
                    "bbox": [left, top, left + width, top + height],
                    "class_name": mapped_cls,
                    "confidence": 1.0,
                    "image_shape": image_shape,
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

    def import_and_generate(
        self,
        dataset_version: str,
        candidates: List[Dict[str, Any]],
        frame_provider_func=None,
    ) -> Dict[str, Any]:
        """
        Runs candidates through DataQualityGate and generates an audited,
        versioned YOLO dataset with video/scenario metadata attached.
        """
        for c in candidates:
            if "video_id" not in c:
                c["video_id"] = f"{c.get('scenario_id', 'SCN')}_{c.get('camera_id', 'CAM')}"

        manifest = self.generator.generate_yolo_dataset(
            dataset_version=dataset_version,
            candidates=candidates,
            frame_provider_func=frame_provider_func,
        )

        manifest["imported_at"] = datetime.now(timezone.utc).isoformat()
        manifest["dataset_format"] = "NORMALIZED_BORDER_SENTINEL_YOLO"
        return manifest
