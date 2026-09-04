"""
YOLO Large Local Pre-Annotator & Active Learning Uncertainty Miner.
Uses local yolov8l.pt on RTX 4060 GPU to pre-generate bounding boxes.
Maps COCO classes to our 9 Master Classes and flags uncertain detections (0.35-0.65 conf)
for prioritized human verification.
"""

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
from ai.inference.loader import get_device, load_local_model
from dataset.tools.dataset_manager import DatasetManager

# Mapping from standard YOLO/COCO classes to our 9 Master Classes
COCO_TO_MASTER_MAP = {
    "person": 0,
    "car": 1,
    "truck": 2,
    "bus": 3,
    "motorcycle": 4,
    "bicycle": 5,
    # Animals map to master class 6
    "bird": 6,
    "cat": 6,
    "dog": 6,
    "horse": 6,
    "sheep": 6,
    "cow": 6,
    "elephant": 6,
    "bear": 6,
    "zebra": 6,
    "giraffe": 6,
    # Bags map to 7 and 8
    "backpack": 7,
    "handbag": 8,
    "suitcase": 8,
}


def run_pre_annotation(
    input_dir: Path | str,
    camera_id: str = "CAM_01",
    model_name: str = "yolov8l.pt",
    conf_min: float = 0.25,
    output_dir: Path | None = None,
) -> dict:
    """
    Pre-annotates all frames in input_dir with local YOLO Large model.
    """
    in_dir = Path(input_dir).resolve()
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Extracted frames directory not found: {in_dir}")

    dm = DatasetManager()
    master_classes = dm.get_classes()

    if output_dir is None:
        out_base = ROOT_DIR / "dataset" / "pre_annotations" / camera_id
    else:
        out_base = Path(output_dir) / camera_id
    out_base.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.png")))
    if not image_files:
        raise ValueError(f"No image files found in: {in_dir}")

    print(f"\n[Pre-Annotator] Loading {model_name} onto GPU...")
    model = load_local_model(model_name)
    device = get_device()

    print(f"[Pre-Annotator] Processing {len(image_files)} frames from camera {camera_id}...")

    total_boxes = 0
    uncertain_boxes = 0
    empty_frames = 0
    annotated_summary = []

    t0 = time.perf_counter()

    for idx, img_path in enumerate(image_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        results = model.predict(source=img, conf=conf_min, device=device, verbose=False)
        boxes_out = []
        yolo_lines = []

        for box in results[0].boxes:
            coco_cls_id = int(box.cls[0].item())
            coco_cls_name = model.names[coco_cls_id]
            conf = float(box.conf[0].item())

            if coco_cls_name in COCO_TO_MASTER_MAP:
                master_id = COCO_TO_MASTER_MAP[coco_cls_name]
                master_name = master_classes.get(master_id, coco_cls_name)
                coords_xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]

                # Calculate normalized YOLO format: [x_center, y_center, width, height]
                x1, y1, x2, y2 = coords_xyxy
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                bx = (x1 + x2) / (2.0 * w)
                by = (y1 + y2) / (2.0 * h)

                # Active learning flag: Uncertain predictions require human review
                is_uncertain = (0.35 <= conf <= 0.65)
                if is_uncertain:
                    uncertain_boxes += 1
                total_boxes += 1

                boxes_out.append({
                    "class_id": master_id,
                    "class_name": master_name,
                    "confidence": round(conf, 3),
                    "bbox_xyxy": coords_xyxy,
                    "bbox_yolo_normalized": [round(bx, 6), round(by, 6), round(bw, 6), round(bh, 6)],
                    "flag": "NEEDS_HUMAN_REVIEW" if is_uncertain else "PREDICTED_OK",
                    "human_verified": False,
                })

                yolo_lines.append(f"{master_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")

        if not boxes_out:
            empty_frames += 1

        # Save JSON annotation for local web UI
        json_file = out_base / f"{img_path.stem}.json"
        record = {
            "image_filename": img_path.name,
            "image_path": str(img_path.relative_to(ROOT_DIR)),
            "camera_id": camera_id,
            "resolution": [w, h],
            "total_objects": len(boxes_out),
            "status": "CANDIDATE_HARD_NEGATIVE" if not boxes_out else ("NEEDS_REVIEW" if any(b["flag"] == "NEEDS_HUMAN_REVIEW" for b in boxes_out) else "PREDICTED"),
            "annotations": boxes_out,
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        # Save standard YOLO .txt label
        txt_file = out_base / f"{img_path.stem}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        annotated_summary.append(record)

    total_time = time.perf_counter() - t0

    # Save master pre-annotation manifest
    manifest_file = ROOT_DIR / "dataset" / "pre_annotations" / f"pre_annotation_manifest_{camera_id}.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({
            "camera_id": camera_id,
            "total_images": len(image_files),
            "total_boxes_generated": total_boxes,
            "uncertain_boxes_flagged": uncertain_boxes,
            "empty_negative_frames": empty_frames,
            "avg_time_per_frame_ms": round((total_time / len(image_files)) * 1000, 2),
            "output_directory": str(out_base),
        }, f, indent=2)

    return {
        "camera_id": camera_id,
        "images_processed": len(image_files),
        "total_boxes": total_boxes,
        "uncertain_boxes": uncertain_boxes,
        "empty_frames": empty_frames,
        "manifest_path": str(manifest_file),
        "output_dir": str(out_base),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-annotate frames with YOLO Large.")
    parser.add_argument("input_dir", nargs="?", default="dataset/extracted_frames/CAM_01", help="Extracted frames dir")
    parser.add_argument("--camera", default="CAM_01", help="Camera ID tag")
    parser.add_argument("--model", default="yolov8l.pt", help="Model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Min confidence")

    args = parser.parse_args()

    try:
        res = run_pre_annotation(
            input_dir=args.input_dir,
            camera_id=args.camera,
            model_name=args.model,
            conf_min=args.conf,
        )
        print("\n--- YOLO Large Pre-Annotation Summary ---")
        print(f"Frames Processed: {res['images_processed']}")
        print(f"Auto-Generated Bounding Boxes: {res['total_boxes']}")
        print(f"Uncertain Detections Flagged for Review (35-65% conf): {res['uncertain_boxes']}")
        print(f"Candidate Negative Frames (Empty): {res['empty_frames']}")
        print(f"Output Annotations Dir: {res['output_dir']}")
        print(f"Pre-Annotation Manifest: {res['manifest_path']}")
        print("Status: YOLO LARGE PRE-ANNOTATION SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
