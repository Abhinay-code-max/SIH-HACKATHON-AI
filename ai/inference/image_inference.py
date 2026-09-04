"""
Standalone Local Image Inference.
Runs local YOLO model on an image and outputs structured detections + annotated image.
Complies with Rule 2: Zero cloud fallback, fully local GPU execution.
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path so 'ai', 'backend', etc. resolve cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import torch
from ai.inference.loader import load_local_model, get_device

# Core surveillance classes for Phase 1 Prototype
TARGET_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "traffic light",
}


def run_image_inference(
    image_path: str | Path,
    model_name: str = "yolov8l.pt",
    conf_threshold: float = 0.35,
    save_annotated: bool = True,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Run local inference on a single image.
    Outputs structured detections conforming to prototype contracts.
    """
    img_path = Path(image_path).resolve()
    if not img_path.is_file():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    # Load model strictly from local weights
    model = load_local_model(model_name)
    device = get_device()

    # Read image locally with OpenCV
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to decode image from: {img_path}")
    height, width = img.shape[:2]

    print(f"\n[Inference] Running YOLO ({model_name}) on {img_path.name} ({width}x{height}) on {device}...")

    # Timed inference
    t0 = time.perf_counter()
    results = model.predict(
        source=img,
        conf=conf_threshold,
        device=device,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    result = results[0]
    detections = []
    annotated_img = img.copy()

    for idx, box in enumerate(result.boxes):
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        conf = float(box.conf[0].item())
        coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]

        # Filter for relevant prototype surveillance classes
        if cls_name in TARGET_CLASSES:
            detections.append({
                "detection_id": f"det_{idx + 1:03d}",
                "class": cls_name,
                "confidence": round(conf, 4),
                "bbox": coords,  # [x1, y1, x2, y2]
                "normalized_center": [
                    round(((coords[0] + coords[2]) / 2) / width, 4),
                    round(((coords[1] + coords[3]) / 2) / height, 4),
                ]
            })

            # Draw bounding box locally
            x1, y1, x2, y2 = map(int, coords)
            color = (0, 255, 0) if cls_name == "person" else (255, 100, 0)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                annotated_img,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    output_payload = {
        "source_image": str(img_path),
        "resolution": {"width": width, "height": height},
        "model_used": model_name,
        "device": device,
        "inference_latency_ms": round(latency_ms, 2),
        "fps_equivalent": round(1000 / latency_ms, 1) if latency_ms > 0 else 0,
        "total_detections": len(detections),
        "detections": detections,
    }

    # Save outputs if output_dir specified or default to data/sample-images/
    if output_dir is None:
        out_dir = img_path.parent
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_annotated:
        annotated_path = out_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(annotated_path), annotated_img)
        output_payload["annotated_image_path"] = str(annotated_path)
        print(f"[Inference] Annotated image saved: {annotated_path}")

    json_path = out_dir / f"detections_{img_path.stem}.json"
    with open(json_path, "w") as f:
        json.dump(output_payload, f, indent=2)
    output_payload["json_output_path"] = str(json_path)
    print(f"[Inference] Detections JSON saved: {json_path}")

    return output_payload


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai/inference/image_inference.py <path_to_image> [model_name] [conf_threshold]")
        print("Example: python ai/inference/image_inference.py data/sample-images/traffic_sample.jpg yolov8l.pt 0.35")
        sys.exit(1)

    image_file = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else "yolov8l.pt"
    confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35

    try:
        report = run_image_inference(image_file, model_name=model_file, conf_threshold=confidence)
        print("\n--- Inference Report Summary ---")
        print(f"Device: {report['device']}")
        print(f"Latency: {report['inference_latency_ms']} ms (~{report['fps_equivalent']} FPS)")
        print(f"Total relevant objects detected: {report['total_detections']}")
        for d in report["detections"]:
            print(f"  - [{d['detection_id']}] {d['class']} (conf: {d['confidence']:.2%}) bbox={d['bbox']}")
        print("Status: IMAGE INFERENCE SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
