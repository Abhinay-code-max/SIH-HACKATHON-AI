"""
Development-time helper to place a deterministic test image into data/sample-images/.
"""

from pathlib import Path
import urllib.request
import cv2
import numpy as np

def setup_sample_image():
    target_dir = Path(__file__).resolve().parent
    sample_file = target_dir / "traffic_sample.jpg"

    if sample_file.exists():
        print(f"[Sample Data] Sample already exists at: {sample_file}")
        return sample_file

    print("[Sample Data] Fetching standard traffic/street test image for deterministic verification...")
    # Standard multi-object test image (bus, people, street)
    test_url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
    try:
        urllib.request.urlretrieve(test_url, str(sample_file))
        print(f"[Sample Data] Downloaded sample image to: {sample_file}")
    except Exception as e:
        print(f"[Sample Data] Online fetch failed ({e}). Generating synthetic test image locally...")
        # Offline fallback during dev: generate a synthetic image with shapes
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[:] = (200, 200, 200) # light grey background
        # Draw some mock objects
        cv2.rectangle(img, (100, 150), (250, 450), (40, 40, 180), -1)
        cv2.rectangle(img, (300, 250), (500, 500), (180, 40, 40), -1)
        cv2.imwrite(str(sample_file), img)
        print(f"[Sample Data] Generated local synthetic image at: {sample_file}")

    return sample_file

if __name__ == "__main__":
    setup_sample_image()
