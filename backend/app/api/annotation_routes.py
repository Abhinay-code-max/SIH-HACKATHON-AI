"""
Annotation & Human Verification API Routes for Milestone v0.2.
Provides endpoints to review, adjust, add, delete, and verify pre-annotated bounding boxes.
Saves human-verified ground truth into dataset/verified_annotations/.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.tools.dataset_manager import DatasetManager

router = APIRouter(prefix="/api/annotation")

dm = DatasetManager()
CLASSES_DICT = dm.get_classes()


class BoundingBox(BaseModel):
    class_id: int
    class_name: str
    confidence: Optional[float] = 1.0
    bbox_xyxy: List[float]  # [x1, y1, x2, y2]
    human_verified: bool = True


class AnnotationPayload(BaseModel):
    image_filename: str
    camera_id: str = "CAM_01"
    status: str = "VERIFIED"  # "VERIFIED" | "HARD_NEGATIVE"
    annotations: List[BoundingBox]


@router.get("/classes")
def get_master_classes() -> Dict[int, str]:
    """Returns the 9 master classes."""
    return CLASSES_DICT


@router.get("/items")
def list_annotation_items(camera_id: str = "CAM_01") -> List[Dict[str, Any]]:
    """Lists all frames in the annotation pipeline with review status."""
    pre_dir = ROOT_DIR / "dataset" / "pre_annotations" / camera_id
    verified_dir = ROOT_DIR / "dataset" / "verified_annotations" / camera_id

    if not pre_dir.is_dir():
        return []

    items = []
    for json_p in sorted(pre_dir.glob("*.json")):
        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if already verified
        is_verified = (verified_dir / json_p.name).is_file()
        status = "VERIFIED" if is_verified else data.get("status", "PREDICTED")

        items.append({
            "stem": json_p.stem,
            "image_filename": data.get("image_filename"),
            "image_path": f"/dataset/extracted_frames/{camera_id}/{data.get('image_filename')}",
            "camera_id": camera_id,
            "total_objects": len(data.get("annotations", [])),
            "status": status,
            "has_uncertain": any(b.get("flag") == "NEEDS_HUMAN_REVIEW" for b in data.get("annotations", [])),
        })

    return items


@router.get("/item/{camera_id}/{stem}")
def get_annotation_item(camera_id: str, stem: str) -> Dict[str, Any]:
    """Fetches annotations for a specific image."""
    verified_file = ROOT_DIR / "dataset" / "verified_annotations" / camera_id / f"{stem}.json"
    pre_file = ROOT_DIR / "dataset" / "pre_annotations" / camera_id / f"{stem}.json"

    target_file = verified_file if verified_file.is_file() else pre_file
    if not target_file.is_file():
        raise HTTPException(status_code=404, detail=f"Annotation not found for {stem}")

    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["image_url"] = f"/dataset/extracted_frames/{camera_id}/{data.get('image_filename')}"
    return data


@router.post("/item/save")
def save_verified_annotation(payload: AnnotationPayload) -> Dict[str, Any]:
    """Saves human-verified ground truth into dataset/verified_annotations/."""
    cam_id = payload.camera_id
    stem = Path(payload.image_filename).stem
    out_dir = ROOT_DIR / "dataset" / "verified_annotations" / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get image resolution from extracted frame
    img_file = ROOT_DIR / "dataset" / "extracted_frames" / cam_id / payload.image_filename
    w, h = 640, 480
    if img_file.is_file():
        import cv2
        img = cv2.imread(str(img_file))
        if img is not None:
            h, w = img.shape[:2]

    # Format YOLO normalized format
    yolo_lines = []
    formatted_boxes = []

    for b in payload.annotations:
        x1, y1, x2, y2 = b.bbox_xyxy
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        bx = (x1 + x2) / (2.0 * w)
        by = (y1 + y2) / (2.0 * h)

        yolo_lines.append(f"{b.class_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")
        formatted_boxes.append({
            "class_id": b.class_id,
            "class_name": b.class_name,
            "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "bbox_yolo_normalized": [round(bx, 6), round(by, 6), round(bw, 6), round(bh, 6)],
            "human_verified": True,
        })

    # 1. Save JSON ground truth
    record = {
        "image_filename": payload.image_filename,
        "camera_id": cam_id,
        "resolution": [w, h],
        "status": "HARD_NEGATIVE" if not formatted_boxes else "VERIFIED_GROUND_TRUTH",
        "human_verified": True,
        "total_objects": len(formatted_boxes),
        "annotations": formatted_boxes,
    }

    json_out = out_dir / f"{stem}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    # 2. Save YOLO .txt label
    txt_out = out_dir / f"{stem}.txt"
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

    return {
        "status": "SAVED_GROUND_TRUTH",
        "saved_json": str(json_out),
        "saved_txt": str(txt_out),
        "objects_count": len(formatted_boxes),
    }


@router.post("/batch_approve")
def batch_approve_all(camera_id: str = "CAM_01") -> Dict[str, Any]:
    """
    Batch-approves all pre-annotated items from pre_annotations/ to verified_annotations/.
    Converts pre-labels into verified ground truth.
    """
    pre_dir = ROOT_DIR / "dataset" / "pre_annotations" / camera_id
    verified_dir = ROOT_DIR / "dataset" / "verified_annotations" / camera_id
    verified_dir.mkdir(parents=True, exist_ok=True)

    if not pre_dir.is_dir():
        raise HTTPException(status_code=404, detail="No pre-annotations found")

    count = 0
    for json_p in pre_dir.glob("*.json"):
        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["human_verified"] = True
        data["status"] = "VERIFIED_GROUND_TRUTH" if data.get("annotations") else "HARD_NEGATIVE"

        target_json = verified_dir / json_p.name
        with open(target_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Copy .txt label file
        txt_src = pre_dir / f"{json_p.stem}.txt"
        txt_dst = verified_dir / f"{json_p.stem}.txt"
        if txt_src.is_file():
            txt_dst.write_text(txt_src.read_text())
        else:
            txt_dst.touch()

        count += 1

    return {
        "status": "BATCH_VERIFICATION_COMPLETE",
        "verified_count": count,
        "destination_dir": str(verified_dir),
    }
