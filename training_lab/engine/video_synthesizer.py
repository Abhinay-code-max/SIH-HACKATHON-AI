"""
Offline Multi-Camera Video Synthesizer.
Generates deterministic synthetic surveillance test video feeds (640x480 MP4)
for CAM_01 through CAM_05 with simulated pedestrian, vehicular, and perimeter activity.
"""

from pathlib import Path
import sys
import time
from typing import Dict
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
VIDEOS_DIR = ROOT_DIR / "training_lab" / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


class VideoSynthesizer:
    """Generates synthetic surveillance video assets for offline simulation."""

    WIDTH = 640
    HEIGHT = 480
    FPS = 25
    TOTAL_FRAMES = 125  # 5 seconds per test video

    @staticmethod
    def draw_tactical_background(cam_id: str, sector_name: str, bg_tone: tuple = (30, 35, 45)) -> np.ndarray:
        """Draws base camera background with surveillance grid and camera HUD."""
        frame = np.full((VideoSynthesizer.HEIGHT, VideoSynthesizer.WIDTH, 3), bg_tone, dtype=np.uint8)

        # Draw optical grid lines
        for x in range(0, VideoSynthesizer.WIDTH, 80):
            cv2.line(frame, (x, 0), (x, VideoSynthesizer.HEIGHT), (45, 55, 65), 1)
        for y in range(0, VideoSynthesizer.HEIGHT, 80):
            cv2.line(frame, (0, y), (VideoSynthesizer.WIDTH, y), (45, 55, 65), 1)

        # Draw ground horizon
        cv2.line(frame, (0, 160), (VideoSynthesizer.WIDTH, 160), (60, 75, 90), 2)

        return frame

    @staticmethod
    def add_camera_hud(frame: np.ndarray, cam_id: str, sector_name: str, frame_idx: int):
        """Overlays standardized tactical CCTV HUD."""
        # Top banner
        cv2.putText(
            frame,
            f"BORDER SENTINEL // {cam_id} [{sector_name}]",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        # Frame counter & status
        cv2.putText(
            frame,
            f"REC  FRAME: {frame_idx:04d}/{VideoSynthesizer.TOTAL_FRAMES}  25.0 FPS  AIR-GAPPED",
            (15, VideoSynthesizer.HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        # Blinking record dot
        if (frame_idx // 10) % 2 == 0:
            cv2.circle(frame, (VideoSynthesizer.WIDTH - 25, 25), 6, (0, 0, 255), -1)

    @staticmethod
    def generate_cam_01_gateway() -> Path:
        """CAM_01: Concourse Main Gateway (Pedestrians walking across doorway)."""
        output_path = VIDEOS_DIR / "CAM_01_gateway.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, VideoSynthesizer.FPS, (VideoSynthesizer.WIDTH, VideoSynthesizer.HEIGHT))

        for f_idx in range(VideoSynthesizer.TOTAL_FRAMES):
            frame = VideoSynthesizer.draw_tactical_background("CAM_01", "CONCOURSE GATEWAY", (32, 38, 48))

            # Gateway doorway structure
            cv2.rectangle(frame, (100, 120), (250, 420), (50, 65, 80), 2)
            cv2.putText(frame, "GATEWAY ACCESS", (110, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 160, 220), 1)

            # Simulated Person 1: Walking left to right
            px1 = int(60 + (f_idx * 3.5))
            py1 = 280
            # Head, torso, legs
            cv2.circle(frame, (px1 + 15, py1 - 40), 12, (200, 210, 220), -1)
            cv2.rectangle(frame, (px1, py1 - 25), (px1 + 30, py1 + 45), (180, 120, 60), -1)
            cv2.line(frame, (px1 + 8, py1 + 45), (px1 + 5, py1 + 85), (80, 80, 160), 4)
            cv2.line(frame, (px1 + 22, py1 + 45), (px1 + 25, py1 + 85), (80, 80, 160), 4)

            # Simulated Person 2: Moving through gateway
            if f_idx > 30:
                px2 = int(140 + ((f_idx - 30) * 2.0))
                py2 = 240
                cv2.circle(frame, (px2 + 12, py2 - 35), 10, (210, 210, 210), -1)
                cv2.rectangle(frame, (px2, py2 - 20), (px2 + 25, py2 + 35), (60, 140, 180), -1)
                cv2.line(frame, (px2 + 6, py2 + 35), (px2 + 6, py2 + 70), (70, 70, 120), 3)
                cv2.line(frame, (px2 + 18, py2 + 35), (px2 + 18, py2 + 70), (70, 70, 120), 3)

            VideoSynthesizer.add_camera_hud(frame, "CAM_01", "CONCOURSE GATEWAY", f_idx)
            out.write(frame)

        out.release()
        return output_path

    @staticmethod
    def generate_cam_02_checkpoint() -> Path:
        """CAM_02: Gate 1 Vehicle Checkpoint (Vehicles stopping at inspection line)."""
        output_path = VIDEOS_DIR / "CAM_02_checkpoint.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, VideoSynthesizer.FPS, (VideoSynthesizer.WIDTH, VideoSynthesizer.HEIGHT))

        for f_idx in range(VideoSynthesizer.TOTAL_FRAMES):
            frame = VideoSynthesizer.draw_tactical_background("CAM_02", "GATE 1 CHECKPOINT", (28, 32, 40))

            # Roadway & checkpoint barrier
            cv2.rectangle(frame, (120, 200), (520, 460), (45, 48, 55), -1)
            cv2.line(frame, (320, 200), (320, 460), (0, 255, 255), 2)  # Stop line
            cv2.putText(frame, "STOP - INSPECTION ZONE", (210, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            # Simulated Vehicle (Truck/Car): approaches, stops at barrier, resumes
            if f_idx < 40:
                vx = int(80 + (f_idx * 4.0))  # Decelerating toward x=240
            elif f_idx < 90:
                vx = 240  # Stopped in inspection zone!
            else:
                vx = int(240 + ((f_idx - 90) * 4.5))  # Departed

            vy = 300
            # Draw vehicle body (truck chassis, cab, wheels)
            cv2.rectangle(frame, (vx, vy - 50), (vx + 110, vy + 30), (60, 100, 160), -1)
            cv2.rectangle(frame, (vx + 80, vy - 65), (vx + 130, vy + 20), (50, 85, 140), -1)
            cv2.circle(frame, (vx + 25, vy + 35), 14, (20, 20, 20), -1)
            cv2.circle(frame, (vx + 85, vy + 35), 14, (20, 20, 20), -1)
            cv2.circle(frame, (vx + 115, vy + 35), 14, (20, 20, 20), -1)

            VideoSynthesizer.add_camera_hud(frame, "CAM_02", "GATE 1 CHECKPOINT", f_idx)
            out.write(frame)

        out.release()
        return output_path

    @staticmethod
    def generate_cam_03_perimeter() -> Path:
        """CAM_03: Perimeter Command Outpost (Border fence buffer zone)."""
        output_path = VIDEOS_DIR / "CAM_03_perimeter.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, VideoSynthesizer.FPS, (VideoSynthesizer.WIDTH, VideoSynthesizer.HEIGHT))

        for f_idx in range(VideoSynthesizer.TOTAL_FRAMES):
            frame = VideoSynthesizer.draw_tactical_background("CAM_03", "PERIMETER COMMAND", (35, 42, 38))

            # Physical barbed fence line across middle
            cv2.line(frame, (0, 280), (VideoSynthesizer.WIDTH, 280), (100, 110, 120), 2)
            for fx in range(0, VideoSynthesizer.WIDTH, 30):
                cv2.line(frame, (fx, 250), (fx, 310), (120, 130, 140), 1)

            # Target moving along perimeter fence (parallel then turning inward)
            tx = int(500 - (f_idx * 2.8))
            ty = int(240 + (f_idx * 0.4))
            cv2.circle(frame, (tx + 10, ty - 25), 8, (210, 210, 200), -1)
            cv2.rectangle(frame, (tx, ty - 15), (tx + 20, ty + 25), (70, 90, 70), -1)
            cv2.line(frame, (tx + 5, ty + 25), (tx + 4, ty + 50), (40, 50, 40), 2)
            cv2.line(frame, (tx + 15, ty + 25), (tx + 16, ty + 50), (40, 50, 40), 2)

            VideoSynthesizer.add_camera_hud(frame, "CAM_03", "PERIMETER COMMAND", f_idx)
            out.write(frame)

        out.release()
        return output_path

    @staticmethod
    def generate_cam_04_north_fence() -> Path:
        """CAM_04: Secondary Fence North (Low-light Night / IR style feed)."""
        output_path = VIDEOS_DIR / "CAM_04_north_fence.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, VideoSynthesizer.FPS, (VideoSynthesizer.WIDTH, VideoSynthesizer.HEIGHT))

        for f_idx in range(VideoSynthesizer.TOTAL_FRAMES):
            # IR night palette (monochromatic green-gray)
            frame = VideoSynthesizer.draw_tactical_background("CAM_04", "NORTH FENCE (IR)", (18, 28, 22))

            # Outer fence posts
            for x in range(40, VideoSynthesizer.WIDTH, 70):
                cv2.line(frame, (x, 180), (x, 360), (40, 70, 50), 2)
            cv2.line(frame, (0, 240), (VideoSynthesizer.WIDTH, 240), (50, 80, 60), 1)

            # Target crawling or sprinting near fence
            nx = int(80 + (f_idx * 3.8))
            ny = 310
            cv2.circle(frame, (nx + 8, ny - 15), 7, (140, 200, 160), -1)
            cv2.rectangle(frame, (nx, ny - 8), (nx + 25, ny + 15), (100, 160, 120), -1)

            VideoSynthesizer.add_camera_hud(frame, "CAM_04", "NORTH FENCE (IR)", f_idx)
            out.write(frame)

        out.release()
        return output_path

    @staticmethod
    def generate_cam_05_logistics() -> Path:
        """CAM_05: Restricted Logistics Depot (Ammunition & Fuel Storage Bay)."""
        output_path = VIDEOS_DIR / "CAM_05_logistics.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, VideoSynthesizer.FPS, (VideoSynthesizer.WIDTH, VideoSynthesizer.HEIGHT))

        for f_idx in range(VideoSynthesizer.TOTAL_FRAMES):
            frame = VideoSynthesizer.draw_tactical_background("CAM_05", "LOGISTICS DEPOT", (30, 30, 42))

            # Storage structures & fuel tanks
            cv2.rectangle(frame, (50, 160), (220, 340), (60, 60, 75), -1)
            cv2.putText(frame, "FUEL BAY 01", (70, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 180, 100), 1)

            # Stationary unattended container / baggage
            cv2.rectangle(frame, (280, 320), (330, 360), (140, 90, 40), -1)
            cv2.putText(frame, "CARGO", (285, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            # Forklift / vehicle moving slowly
            fx = int(480 - (f_idx * 1.5))
            fy = 340
            cv2.rectangle(frame, (fx, fy - 35), (fx + 65, fy + 20), (220, 160, 20), -1)
            cv2.circle(frame, (fx + 15, fy + 25), 10, (20, 20, 20), -1)
            cv2.circle(frame, (fx + 50, fy + 25), 10, (20, 20, 20), -1)

            VideoSynthesizer.add_camera_hud(frame, "CAM_05", "LOGISTICS DEPOT", f_idx)
            out.write(frame)

        out.release()
        return output_path

    @classmethod
    def synthesize_all_videos(cls) -> Dict[str, Path]:
        """Generates all 5 test videos in training_lab/videos/."""
        print("[Video Synthesizer] Generating 5 synthetic surveillance camera streams...")
        t0 = time.time()
        v1 = cls.generate_cam_01_gateway()
        print(f"  -> Generated: {v1.name} ({v1.stat().st_size / 1024:.1f} KB)")
        v2 = cls.generate_cam_02_checkpoint()
        print(f"  -> Generated: {v2.name} ({v2.stat().st_size / 1024:.1f} KB)")
        v3 = cls.generate_cam_03_perimeter()
        print(f"  -> Generated: {v3.name} ({v3.stat().st_size / 1024:.1f} KB)")
        v4 = cls.generate_cam_04_north_fence()
        print(f"  -> Generated: {v4.name} ({v4.stat().st_size / 1024:.1f} KB)")
        v5 = cls.generate_cam_05_logistics()
        print(f"  -> Generated: {v5.name} ({v5.stat().st_size / 1024:.1f} KB)")
        elapsed = time.time() - t0
        print(f"[Video Synthesizer] All 5 camera streams synthesized in {elapsed:.2f}s.")
        return {
            "CAM_01": v1,
            "CAM_02": v2,
            "CAM_03": v3,
            "CAM_04": v4,
            "CAM_05": v5,
        }


if __name__ == "__main__":
    VideoSynthesizer.synthesize_all_videos()
