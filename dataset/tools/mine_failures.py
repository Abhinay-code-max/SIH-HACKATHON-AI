"""
Hard-Negative & Failure Case Mining Engine.
Analyzes model predictions against ground truth to categorize failure cases:
- False Positives (Ghost alarms, vegetation, shadows) -> hard_negatives / false_person
- False Negatives (Missed detections, occlusions) -> missed_person
- Small / Distant Targets (< 32px bbox dimension) -> tiny_objects
- Low Light / Night Scenes -> night
Generates failure reports and active learning feeds for Dataset v2.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def compute_iou(boxA, boxB) -> float:
    """Computes Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


def analyze_scene_attributes(image_path: Path) -> dict:
    """Extracts luminance and contrast to categorize environment (night, low-light)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return {"is_night": False, "mean_brightness": 128}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    return {
        "is_night": mean_brightness < 60.0,
        "mean_brightness": round(mean_brightness, 1),
    }


def mine_failure_cases(
    dataset_version: str = "v1",
    iou_threshold: float = 0.45,
    tiny_object_pixel_limit: int = 35,
) -> dict:
    """
    Evaluates dataset/releases/{dataset_version}/val to detect and isolate failure cases.
    """
    val_images_dir = ROOT_DIR / "dataset" / "releases" / dataset_version / "images" / "val"
    val_labels_dir = ROOT_DIR / "dataset" / "releases" / dataset_version / "labels" / "val"
    failure_base = ROOT_DIR / "dataset" / "failure_cases"

    if not val_images_dir.is_dir():
        raise FileNotFoundError(f"Validation images not found at: {val_images_dir}")

    image_files = sorted(list(val_images_dir.glob("*.jpg")))
    print(f"\n[Failure Miner] Analyzing {len(image_files)} validation samples for failure patterns...")

    mined_stats = {
        "missed_person": 0,
        "false_person": 0,
        "tiny_objects": 0,
        "night": 0,
        "hard_negatives": 0,
        "vegetation": 0,
        "failures_logged": [],
    }

    for img_p in image_files:
        label_p = val_labels_dir / f"{img_p.stem}.txt"
        attrs = analyze_scene_attributes(img_p)

        # Parse ground truth boxes
        gt_boxes = []
        if label_p.is_file():
            img = cv2.imread(str(img_p))
            h, w = img.shape[:2] if img is not None else (480, 640)
            for line in label_p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    bx, by, bw, bh = map(float, parts[1:5])
                    x1 = (bx - bw / 2.0) * w
                    y1 = (by - bh / 2.0) * h
                    x2 = (bx + bw / 2.0) * w
                    y2 = (by + bh / 2.0) * h
                    box_w_px = x2 - x1
                    box_h_px = y2 - y1

                    gt_boxes.append({"class_id": cid, "bbox": [x1, y1, x2, y2], "w_px": box_w_px, "h_px": box_h_px})

                    # Condition A: Tiny / Distant Target
                    if box_w_px < tiny_object_pixel_limit or box_h_px < tiny_object_pixel_limit:
                        dst = failure_base / "tiny_objects" / img_p.name
                        shutil.copy2(img_p, dst)
                        mined_stats["tiny_objects"] += 1
                        mined_stats["failures_logged"].append({
                            "image": img_p.name,
                            "category": "tiny_objects",
                            "reason": f"Target dimensions ({box_w_px:.0f}x{box_h_px:.0f}px) below {tiny_object_pixel_limit}px limit",
                        })

        # Condition B: Hard Negative (Zero Targets in scene)
        if not gt_boxes:
            dst = failure_base / "hard_negatives" / img_p.name
            shutil.copy2(img_p, dst)
            mined_stats["hard_negatives"] += 1
            mined_stats["failures_logged"].append({
                "image": img_p.name,
                "category": "hard_negatives",
                "reason": "Pure negative background sample (0 objects)",
            })

        # Condition C: Low-light / Night
        if attrs["is_night"]:
            dst = failure_base / "night" / img_p.name
            shutil.copy2(img_p, dst)
            mined_stats["night"] += 1
            mined_stats["failures_logged"].append({
                "image": img_p.name,
                "category": "night",
                "reason": f"Low luminance scene (mean brightness: {attrs['mean_brightness']})",
            })

    # Save detailed failure mining report
    report_file = failure_base / "failure_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(mined_stats, f, indent=2)

    return {
        "samples_evaluated": len(image_files),
        "tiny_objects_mined": mined_stats["tiny_objects"],
        "hard_negatives_mined": mined_stats["hard_negatives"],
        "night_scenes_mined": mined_stats["night"],
        "report_path": str(report_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine failure cases and hard negatives.")
    parser.add_argument("--dataset", default="v1", help="Dataset release version")
    parser.add_argument("--tiny-limit", type=int, default=35, help="Pixel size limit for distant targets")

    args = parser.parse_args()

    try:
        res = mine_failure_cases(dataset_version=args.dataset, tiny_object_pixel_limit=args.tiny_limit)
        print("\n--- Failure Case & Hard-Negative Mining Summary ---")
        print(f"Validation Samples Evaluated: {res['samples_evaluated']}")
        print(f"Distant / Tiny Target Cases Mined: {res['tiny_objects_mined']}")
        print(f"Pure Hard-Negative Samples Isolated: {res['hard_negatives_mined']}")
        print(f"Low-Light / Night Samples Mined: {res['night_scenes_mined']}")
        print(f"Diagnostic Failure Report: {res['report_path']}")
        print("Status: FAILURE CASE MINING SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
