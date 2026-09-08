"""
BORDER SENTINEL — 5-Camera Surveillance Demo Video Generator.
100% Air-Gapped & Offline / Zero Internet Required.
Synthesizes calibrated test streams for CAM_01 through CAM_05 using OpenCV.
"""

from pathlib import Path
import sys
import shutil

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_lab.engine.video_synthesizer import VideoSynthesizer


def generate_all_demo_videos() -> dict[str, Path]:
    """Ensures calibrated video files are synthesized for CAM_01 through CAM_05."""
    print("======================================================================")
    print("   BORDER SENTINEL // 5-CAMERA SURVEILLANCE VIDEO GENERATOR")
    print("   100% Offline / Local OpenCV Generation / Zero External Downloads")
    print("======================================================================")
    print("")

    sample_dir = ROOT_DIR / "data" / "sample-videos"
    sample_dir.mkdir(parents=True, exist_ok=True)

    videos = VideoSynthesizer.synthesize_all_videos()

    # Mirror primary sample files into data/sample-videos if missing
    sample_surv = sample_dir / "sample_surveillance.mp4"
    if not sample_surv.exists() and videos.get("CAM_02", Path()).exists():
        shutil.copyfile(videos["CAM_02"], sample_surv)

    annotated_surv = sample_dir / "annotated_surveillance.mp4"
    if not annotated_surv.exists() and videos.get("CAM_03", Path()).exists():
        shutil.copyfile(videos["CAM_03"], annotated_surv)

    print("\n[OK] 5-Camera Surveillance Video Suite Ready:")
    for cam_id, vpath in videos.items():
        size_kb = vpath.stat().st_size / 1024 if vpath.exists() else 0
        print(f"    - {cam_id}: {vpath.name} ({size_kb:.1f} KB)")

    return videos


if __name__ == "__main__":
    generate_all_demo_videos()
