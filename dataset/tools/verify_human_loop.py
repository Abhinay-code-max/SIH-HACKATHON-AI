"""
Verification script for the Human-in-the-Loop Annotation Studio.
Tests loading items, uncertainty mining, and batch ground-truth approval.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from backend.app.main import app


def test_annotation_loop():
    print("[Human Loop] Testing Local Annotation & Verification Studio...")
    client = TestClient(app)

    # 1. Test Studio UI endpoint
    r_ui = client.get("/annotate")
    assert r_ui.status_code == 200, f"Annotation studio failed: {r_ui.status_code}"
    assert "HUMAN-IN-THE-LOOP" in r_ui.text
    print("  -> GET /annotate: 200 OK (Human Verification Studio served offline)")

    # 2. Test fetching items list
    r_items = client.get("/api/annotation/items?camera_id=CAM_01")
    assert r_items.status_code == 200
    items = r_items.json()
    assert len(items) > 0, "No pre-annotated frames returned!"
    uncertain_count = sum(1 for it in items if it.get("has_uncertain"))
    print(f"  -> GET /api/annotation/items: 200 OK ({len(items)} frames loaded, {uncertain_count} flagged for human review)")

    # 3. Test Batch Approving pre-labels to verified ground truth
    r_batch = client.post("/api/annotation/batch_approve?camera_id=CAM_01")
    assert r_batch.status_code == 200
    batch_res = r_batch.json()
    print(f"  -> POST /api/annotation/batch_approve: 200 OK ({batch_res['verified_count']} frames locked as verified ground truth)")

    # 4. Verify verified_annotations directory
    verified_dir = ROOT_DIR / "dataset" / "verified_annotations" / "CAM_01"
    json_count = len(list(verified_dir.glob("*.json")))
    txt_count = len(list(verified_dir.glob("*.txt")))
    assert json_count > 0 and txt_count > 0, "Verified ground truth files not created!"
    print(f"  -> Ground Truth Verified Files: {json_count} JSON records, {txt_count} YOLO .txt labels")

    return {
        "frames_loaded": len(items),
        "uncertain_flagged": uncertain_count,
        "verified_ground_truth": json_count,
        "verified_dir": str(verified_dir),
    }


if __name__ == "__main__":
    try:
        res = test_annotation_loop()
        print("\n--- Human-in-the-Loop Verification Summary ---")
        print(f"Frames Ingested: {res['frames_loaded']}")
        print(f"Uncertain Detections Prioritized: {res['uncertain_flagged']}")
        print(f"Verified Ground Truth Created: {res['verified_ground_truth']}")
        print(f"Verified Annotations Directory: {res['verified_dir']}")
        print("Status: HUMAN-IN-THE-LOOP VERIFICATION SUCCESSFUL")
    except Exception as e:
        print(f"Status: ERROR - {e}", file=sys.stderr)
        sys.exit(1)
