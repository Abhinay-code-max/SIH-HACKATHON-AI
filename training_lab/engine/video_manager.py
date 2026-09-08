"""
Video Asset Manager for AI Training, Validation & Simulation Lab.
Manages surveillance video assets, extracts video metadata, and synthesizes
calibrated surveillance test clips for CAM_01 through CAM_05 offline without
internet dependency.
"""

from pathlib import Path
import os
import sys
import time
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
VIDEOS_DIR = ROOT_DIR / "training_lab" / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


class VideoAssetManager:
    """
    Catalogues, validates, and synthesizes surveillance video assets for
    the Border Sentinel 5-camera simulation engine.
    """

    WIDTH = 640
    HEIGHT = 480
    FPS = 30.0
    DEFAULT_DURATION_SEC = 5.0

    def __init__(self, videos_dir: Optional[Union[str, Path]] = None):
        self.videos_dir = Path(videos_dir) if videos_dir else VIDEOS_DIR
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def list_videos(self) -> List[Dict[str, Any]]:
        """Catalogues all MP4 video clips present in the videos directory."""
        results = []
        for p in sorted(self.videos_dir.glob("*.mp4")):
            try:
                meta = self.get_video_metadata(p)
                results.append(meta)
            except Exception:
                continue
        return results

    def get_video_metadata(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Extracts stream metadata from a video file on disk.

        Returns:
            Dict containing:
                - filepath (str): absolute path to video
                - filename (str): filename
                - width (int): frame width (e.g. 640)
                - height (int): frame height (e.g. 480)
                - resolution (Tuple[int, int]): (width, height)
                - fps (float): frames per second (e.g. 30.0)
                - frame_count (int): total frames
                - total_frames (int): total frames alias
                - duration_sec (float): duration in seconds
                - duration_seconds (float): duration in seconds alias
                - file_size_mb (float): file size in megabytes
                - file_size_bytes (int): file size in bytes
        """
        p = Path(filepath).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Video file not found: {filepath}")

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            raise ValueError(f"Could not open video file: {filepath}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = round(frame_count / fps, 3) if fps > 0 else 0.0
            file_size_bytes = p.stat().st_size
            file_size_mb = round(file_size_bytes / (1024 * 1024), 3)
        finally:
            cap.release()

        return {
            "filepath": str(p),
            "filename": p.name,
            "width": width,
            "height": height,
            "resolution": (width, height),
            "fps": fps,
            "frame_count": frame_count,
            "total_frames": frame_count,
            "duration_sec": duration_sec,
            "duration_seconds": duration_sec,
            "file_size_mb": file_size_mb,
            "file_size_bytes": file_size_bytes,
        }

    def is_valid_video_file(
        self,
        filepath: Union[str, Path],
        expected_width: Optional[int] = WIDTH,
        expected_height: Optional[int] = HEIGHT,
        expected_fps: Optional[float] = FPS,
        min_frames: int = 1,
    ) -> bool:
        """
        Validates that an MP4 surveillance video on disk:
          - exists and has non-trivial size (>1024 bytes)
          - opens successfully via cv2.VideoCapture
          - matches expected width and height (default 640x480)
          - matches expected frame rate (default 30.0 FPS)
          - contains at least min_frames (non-zero frame count)
          - yields a valid readable first frame with matching dimensions
        """
        try:
            p = Path(filepath).resolve()
            if not p.is_file() or p.stat().st_size < 1024:
                return False

            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                return False

            try:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if expected_width is not None and width != expected_width:
                    return False
                if expected_height is not None and height != expected_height:
                    return False
                if expected_fps is not None and abs(fps - expected_fps) > 1.0:
                    return False
                if frame_count < min_frames:
                    return False

                ret, frame = cap.read()
                if not ret or frame is None:
                    return False
                if frame.shape[0] != height or frame.shape[1] != width:
                    return False

                return True
            finally:
                cap.release()
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Visual Synthesis Subroutines
    # -------------------------------------------------------------------------

    @staticmethod
    def _draw_tactical_hud(
        frame: np.ndarray,
        camera_id: str,
        sector_name: str,
        frame_idx: int,
        total_frames: int,
        fps: float = 30.0,
    ) -> None:
        """Draws standard tactical surveillance overlay HUD on top of frame."""
        h, w = frame.shape[:2]

        # Top banner background bar
        cv2.rectangle(frame, (0, 0), (w, 32), (18, 22, 28), -1)
        cv2.line(frame, (0, 32), (w, 32), (0, 180, 220), 1)

        # Header text
        title_str = f"BORDER SENTINEL GRID // {camera_id.upper()} [{sector_name.upper()}]"
        cv2.putText(
            frame,
            title_str,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )

        # Bottom banner background bar
        cv2.rectangle(frame, (0, h - 26), (w, h), (18, 22, 28), -1)
        cv2.line(frame, (0, h - 26), (w, h - 26), (45, 55, 65), 1)

        # Bottom info string
        status_str = f"AIR-GAPPED SIM // FRAME: {frame_idx + 1:04d}/{total_frames:04d} // {fps:.1f} FPS // CALIBRATED"
        cv2.putText(
            frame,
            status_str,
            (12, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )

        # Blinking REC dot
        if (frame_idx // 12) % 2 == 0:
            cv2.circle(frame, (w - 20, 16), 5, (0, 0, 230), -1)
            cv2.putText(
                frame,
                "REC",
                (w - 55, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 240),
                1,
                cv2.LINE_AA,
            )

    @classmethod
    def _render_cam_01_perimeter(cls, frame_idx: int, total_frames: int) -> np.ndarray:
        """
        CAM_01: Perimeter fence background with a moving rectangular
        'pedestrian' moving toward the virtual line.
        """
        w, h = cls.WIDTH, cls.HEIGHT
        frame = np.full((h, w, 3), (35, 40, 48), dtype=np.uint8)

        # Ground surface
        cv2.rectangle(frame, (0, 260), (w, h), (30, 34, 38), -1)
        cv2.line(frame, (0, 260), (w, 260), (70, 80, 90), 2)

        # Perimeter Fence line across middle
        for x_post in range(20, w, 70):
            cv2.line(frame, (x_post, 170), (x_post, 270), (90, 100, 110), 3)
            cv2.circle(frame, (x_post, 170), 3, (120, 130, 140), -1)
        # Horizontal fence cables
        cv2.line(frame, (0, 190), (w, 190), (80, 90, 100), 1)
        cv2.line(frame, (0, 220), (w, 220), (80, 90, 100), 1)
        cv2.line(frame, (0, 250), (w, 250), (80, 90, 100), 1)

        # Virtual Tripwire Line (dashed yellow/amber line at x=380)
        virtual_line_x = 380
        for y_seg in range(80, 420, 16):
            cv2.line(frame, (virtual_line_x, y_seg), (virtual_line_x, y_seg + 8), (0, 220, 255), 2)
        cv2.putText(
            frame,
            "VIRTUAL TRIPWIRE LINE",
            (virtual_line_x + 6, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

        # Moving pedestrian target: traverses from x=60 toward the virtual line
        progress = frame_idx / max(total_frames - 1, 1)
        ped_x = int(60 + progress * 330)
        ped_y = 250

        # Pedestrian figure: head, torso, legs
        leg_swing = math.sin(frame_idx * 0.45) * 8
        cv2.circle(frame, (ped_x + 15, ped_y - 45), 11, (210, 215, 220), -1)  # Head
        cv2.rectangle(frame, (ped_x + 4, ped_y - 32), (ped_x + 26, ped_y + 12), (50, 120, 200), -1)  # Torso
        # Legs
        cv2.line(frame, (ped_x + 9, ped_y + 12), (int(ped_x + 6 - leg_swing), ped_y + 42), (40, 50, 70), 3)
        cv2.line(frame, (ped_x + 21, ped_y + 12), (int(ped_x + 24 + leg_swing), ped_y + 42), (40, 50, 70), 3)

        # Bounding box tag
        cv2.rectangle(frame, (ped_x - 2, ped_y - 60), (ped_x + 32, ped_y + 44), (0, 255, 120), 1)
        cv2.putText(
            frame,
            "PERSON 0.94",
            (ped_x - 4, ped_y - 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )

        cls._draw_tactical_hud(frame, "CAM_01", "PERIMETER FENCE SECTOR", frame_idx, total_frames, cls.FPS)
        return frame

    @classmethod
    def _render_cam_02_roadway(cls, frame_idx: int, total_frames: int) -> np.ndarray:
        """
        CAM_02: Roadway background with moving 'vehicle' shape.
        """
        w, h = cls.WIDTH, cls.HEIGHT
        frame = np.full((h, w, 3), (40, 42, 45), dtype=np.uint8)

        # Asphalt roadway
        cv2.rectangle(frame, (0, 160), (w, 420), (32, 34, 38), -1)
        # Road edge white lines
        cv2.line(frame, (0, 160), (w, 160), (220, 220, 220), 2)
        cv2.line(frame, (0, 420), (w, 420), (220, 220, 220), 2)

        # Dashed yellow center line
        for x_dash in range(0, w, 40):
            cv2.line(frame, (x_dash, 290), (x_dash + 22, 290), (0, 215, 255), 2)

        # Moving Vehicle shape (SUV / tactical pickup)
        progress = frame_idx / max(total_frames - 1, 1)
        veh_x = int(-80 + progress * 720)
        veh_y = 240

        # Headlight illumination cone on roadway
        if veh_x + 130 < w:
            pts_light = np.array([
                [veh_x + 120, veh_y + 10],
                [min(veh_x + 240, w - 1), veh_y - 15],
                [min(veh_x + 240, w - 1), veh_y + 40],
            ], np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_light], (70, 75, 40))
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Vehicle body
        cv2.rectangle(frame, (veh_x, veh_y - 20), (veh_x + 115, veh_y + 25), (70, 95, 130), -1)
        # Cab / roof
        cv2.rectangle(frame, (veh_x + 35, veh_y - 45), (veh_x + 95, veh_y - 20), (55, 75, 105), -1)
        # Windows
        cv2.rectangle(frame, (veh_x + 40, veh_y - 40), (veh_x + 62, veh_y - 23), (170, 200, 220), -1)
        cv2.rectangle(frame, (veh_x + 68, veh_y - 40), (veh_x + 90, veh_y - 23), (170, 200, 220), -1)
        # Wheels
        cv2.circle(frame, (veh_x + 25, veh_y + 25), 13, (15, 15, 15), -1)
        cv2.circle(frame, (veh_x + 25, veh_y + 25), 5, (140, 140, 140), -1)
        cv2.circle(frame, (veh_x + 92, veh_y + 25), 13, (15, 15, 15), -1)
        cv2.circle(frame, (veh_x + 92, veh_y + 25), 5, (140, 140, 140), -1)
        # Headlights
        cv2.rectangle(frame, (veh_x + 112, veh_y - 8), (veh_x + 116, veh_y + 10), (0, 240, 255), -1)

        # Bounding box tag
        if 0 <= veh_x <= w - 40:
            cv2.rectangle(frame, (veh_x - 5, veh_y - 52), (veh_x + 120, veh_y + 40), (0, 255, 255), 1)
            cv2.putText(
                frame,
                "VEHICLE 0.91",
                (veh_x - 5, veh_y - 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cls._draw_tactical_hud(frame, "CAM_02", "GATEWAY ROADWAY CHECKPOINT", frame_idx, total_frames, cls.FPS)
        return frame

    @classmethod
    def _render_cam_03_restricted(cls, frame_idx: int, total_frames: int) -> np.ndarray:
        """
        CAM_03: Restricted zone with stationary 'backpack' box.
        """
        w, h = cls.WIDTH, cls.HEIGHT
        frame = np.full((h, w, 3), (32, 36, 44), dtype=np.uint8)

        # Security room / depot floor grid
        for x_g in range(0, w, 40):
            cv2.line(frame, (x_g, 100), (x_g, h - 30), (42, 46, 54), 1)
        for y_g in range(100, h - 30, 40):
            cv2.line(frame, (0, y_g), (w, y_g), (42, 46, 54), 1)

        # Restricted Zone Boundary (Red / Amber hazard boundary)
        zx1, zy1, zx2, zy2 = 180, 170, 460, 390
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (20, 20, 180), 2)

        # Hazard diagonal stripe pattern along zone top border
        for sx in range(zx1, zx2 - 12, 16):
            cv2.line(frame, (sx, zy1), (sx + 10, zy1 + 10), (0, 215, 255), 2)

        cv2.putText(
            frame,
            "RESTRICTED ZONE // NO ENTRY",
            (zx1 + 10, zy1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # Stationary "backpack" box in restricted zone
        bp_x, bp_y = 310, 275
        # Main pack
        cv2.rectangle(frame, (bp_x, bp_y), (bp_x + 44, bp_y + 36), (40, 75, 120), -1)
        # Front pocket
        cv2.rectangle(frame, (bp_x + 6, bp_y + 12), (bp_x + 38, bp_y + 32), (32, 60, 98), -1)
        # Zipper / straps
        cv2.line(frame, (bp_x + 8, bp_y + 12), (bp_x + 36, bp_y + 12), (180, 190, 200), 1)
        cv2.ellipse(frame, (bp_x + 22, bp_y), (10, 6), 0, 180, 360, (50, 90, 140), 2)  # Handle

        # Unattended object alarm box
        dwell_sec = frame_idx / cls.FPS
        cv2.rectangle(frame, (bp_x - 6, bp_y - 12), (bp_x + 50, bp_y + 42), (0, 0, 255), 1)
        cv2.putText(
            frame,
            f"SUSPICIOUS BAG // DWELL {dwell_sec:.1f}s",
            (bp_x - 30, bp_y - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        cls._draw_tactical_hud(frame, "CAM_03", "RESTRICTED DEPOT VAULT", frame_idx, total_frames, cls.FPS)
        return frame

    @classmethod
    def _render_cam_04_terrain(cls, frame_idx: int, total_frames: int) -> np.ndarray:
        """
        CAM_04: Open terrain with low-light night noise.
        """
        w, h = cls.WIDTH, cls.HEIGHT
        # Night thermal / low-lux dark greenish tint
        frame = np.full((h, w, 3), (16, 26, 20), dtype=np.uint8)

        # Distant terrain horizon hills
        pts_hill = np.array([
            [0, 220],
            [120, 190],
            [260, 210],
            [410, 180],
            [540, 205],
            [w, 185],
            [w, h],
            [0, h],
        ], np.int32)
        cv2.fillPoly(frame, [pts_hill], (12, 20, 15))

        # Terrain foreground contours and rocks
        cv2.line(frame, (0, 290), (w, 290), (22, 34, 25), 2)
        for rx, ry, rrad in [(80, 320, 10), (240, 350, 15), (460, 310, 12), (580, 370, 18)]:
            cv2.circle(frame, (rx, ry), rrad, (20, 30, 24), -1)

        # Distant target moving across open terrain (low-contrast nocturnal crawling/moving subject)
        progress = frame_idx / max(total_frames - 1, 1)
        tgt_x = int(480 - progress * 240)
        tgt_y = int(240 + progress * 30)

        # Silhouette
        cv2.circle(frame, (tgt_x + 8, tgt_y - 8), 6, (38, 58, 44), -1)
        cv2.rectangle(frame, (tgt_x, tgt_y - 2), (tgt_x + 24, tgt_y + 16), (32, 50, 38), -1)

        # Add calibrated low-light optical sensor noise
        # Seeded deterministically per frame so it is reproducible
        rng = np.random.RandomState(frame_idx + 1000)
        noise = rng.normal(0, 7.5, (h, w, 3)).astype(np.float32)
        noisy_frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        frame = noisy_frame

        # Thermal HUD indicators
        cv2.putText(
            frame,
            "IR SENSOR // AGC: HIGH // LUX: 0.02 // NIGHT-MODE",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (40, 190, 80),
            1,
            cv2.LINE_AA,
        )

        cls._draw_tactical_hud(frame, "CAM_04", "OPEN TERRAIN NORTH SECTOR", frame_idx, total_frames, cls.FPS)
        return frame

    @classmethod
    def _render_cam_05_secondary(cls, frame_idx: int, total_frames: int) -> np.ndarray:
        """
        CAM_05: Secondary gate with crossing 'motorcycle'.
        """
        w, h = cls.WIDTH, cls.HEIGHT
        frame = np.full((h, w, 3), (36, 38, 44), dtype=np.uint8)

        # Secondary access road
        cv2.rectangle(frame, (0, 210), (w, 410), (45, 48, 52), -1)
        cv2.line(frame, (0, 210), (w, 210), (80, 90, 100), 2)
        cv2.line(frame, (0, 410), (w, 410), (80, 90, 100), 2)

        # Checkpoint booth structure on the right
        cv2.rectangle(frame, (480, 140), (590, 310), (55, 60, 70), -1)
        cv2.rectangle(frame, (495, 160), (575, 220), (140, 170, 190), -1)  # Window
        cv2.putText(frame, "GATE 05", (505, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        # Barrier arm across road (partially raised)
        cv2.line(frame, (480, 280), (320, 210), (0, 0, 220), 4)
        for seg_x in range(330, 480, 30):
            cv2.line(frame, (seg_x, 210 + int((seg_x - 320) * 0.44)), (seg_x + 12, 210 + int((seg_x - 308) * 0.44)), (240, 240, 240), 4)

        # Crossing Motorcycle with Rider
        progress = frame_idx / max(total_frames - 1, 1)
        mc_x = int(40 + progress * 520)
        mc_y = 320

        # Motorcycle wheels
        cv2.circle(frame, (mc_x, mc_y), 14, (20, 20, 20), -1)
        cv2.circle(frame, (mc_x, mc_y), 14, (120, 120, 120), 2)
        cv2.circle(frame, (mc_x + 55, mc_y), 14, (20, 20, 20), -1)
        cv2.circle(frame, (mc_x + 55, mc_y), 14, (120, 120, 120), 2)

        # Frame and tank
        cv2.line(frame, (mc_x, mc_y), (mc_x + 28, mc_y - 12), (180, 30, 30), 4)
        cv2.line(frame, (mc_x + 28, mc_y - 12), (mc_x + 55, mc_y), (180, 30, 30), 4)
        cv2.rectangle(frame, (mc_x + 20, mc_y - 20), (mc_x + 42, mc_y - 10), (210, 40, 40), -1)

        # Rider (helmet head, leaning torso)
        cv2.circle(frame, (mc_x + 26, mc_y - 48), 9, (230, 230, 230), -1)  # Helmet
        cv2.rectangle(frame, (mc_x + 18, mc_y - 38), (mc_x + 36, mc_y - 18), (50, 70, 90), -1)  # Torso
        cv2.line(frame, (mc_x + 30, mc_y - 30), (mc_x + 48, mc_y - 22), (40, 40, 40), 3)  # Arm to handlebars

        # Bounding tag
        if 0 <= mc_x <= w - 30:
            cv2.rectangle(frame, (mc_x - 10, mc_y - 60), (mc_x + 70, mc_y + 18), (255, 180, 0), 1)
            cv2.putText(
                frame,
                "MOTORCYCLE 0.88",
                (mc_x - 10, mc_y - 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

        cls._draw_tactical_hud(frame, "CAM_05", "SECONDARY GATEWAY CORRIDOR", frame_idx, total_frames, cls.FPS)
        return frame

    # -------------------------------------------------------------------------
    # Public Synthesis Methods
    # -------------------------------------------------------------------------

    def _create_video_writer(
        self,
        output_path: Path,
        fps: float,
        resolution: Tuple[int, int],
    ) -> Tuple[cv2.VideoWriter, str]:
        """
        Initializes cv2.VideoWriter with MP4V encoding and fallback backends.
        Tries default backend, explicit FFMPEG, then MSMF (Windows).
        Returns (writer, backend_name).
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # 1. Default backend (standard OpenCV selection, typically FFMPEG on Windows)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
        if writer.isOpened():
            return writer, "mp4v/default"
        try:
            writer.release()
        except Exception:
            pass

        # 2. Explicit CAP_FFMPEG backend
        if hasattr(cv2, "CAP_FFMPEG"):
            writer = cv2.VideoWriter(str(output_path), cv2.CAP_FFMPEG, fourcc, fps, resolution)
            if writer.isOpened():
                return writer, "mp4v/ffmpeg"
            try:
                writer.release()
            except Exception:
                pass

        # 3. Windows Media Foundation (CAP_MSMF) fallback
        if sys.platform == "win32" and hasattr(cv2, "CAP_MSMF"):
            writer = cv2.VideoWriter(str(output_path), cv2.CAP_MSMF, fourcc, fps, resolution)
            if writer.isOpened():
                return writer, "mp4v/msmf"
            try:
                writer.release()
            except Exception:
                pass

            # 4. CAP_MSMF with H.264 (avc1) fallback
            fourcc_avc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(str(output_path), cv2.CAP_MSMF, fourcc_avc, fps, resolution)
            if writer.isOpened():
                return writer, "avc1/msmf"
            try:
                writer.release()
            except Exception:
                pass

        raise IOError(
            f"Failed to initialize VideoWriter for '{output_path}'. "
            f"fourcc='mp4v' (and fallback 'avc1') could not be initialized with available backends."
        )

    def generate_synthetic_surveillance_video(
        self,
        output_path: Union[str, Path],
        camera_id: str = "CAM_01",
        duration_sec: float = 5.0,
        scenario_type: Optional[str] = None,
    ) -> Path:
        """
        Generates a clean, 640x480 30FPS MP4 surveillance test clip using OpenCV.
        Offline, cross-platform, deterministic generation with zero internet dependency.
        Employs staging-file generation, post-generation validation, and safe atomic replacement
        to prevent partial/corrupted artifacts or Windows sharing violations.

        Args:
            output_path: destination path for MP4 file
            camera_id: e.g. CAM_01, CAM_02, CAM_03, CAM_04, CAM_05
            duration_sec: duration in seconds (default 5.0s -> 150 frames)
            scenario_type: perimeter, roadway, restricted, terrain, secondary
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        fps = self.FPS
        total_frames = max(1, int(round(duration_sec * fps)))
        resolution = (self.WIDTH, self.HEIGHT)

        # Staging file prevents in-place locking conflicts and partial corrupted target files
        staging_filename = f".tmp_{out_p.stem}_{os.getpid()}_{int(time.time() * 1000)}.mp4"
        staging_p = out_p.parent / staging_filename

        writer = None
        try:
            writer, backend_tag = self._create_video_writer(staging_p, fps, resolution)

            # Determine target renderer
            cam_upper = camera_id.upper()
            scen_lower = (scenario_type or "").lower()

            if "cam_01" in cam_upper or scen_lower == "perimeter":
                renderer = self._render_cam_01_perimeter
            elif "cam_02" in cam_upper or scen_lower == "roadway":
                renderer = self._render_cam_02_roadway
            elif "cam_03" in cam_upper or scen_lower == "restricted":
                renderer = self._render_cam_03_restricted
            elif "cam_04" in cam_upper or scen_lower == "terrain":
                renderer = self._render_cam_04_terrain
            elif "cam_05" in cam_upper or scen_lower == "secondary":
                renderer = self._render_cam_05_secondary
            else:
                renderer = self._render_cam_01_perimeter

            for frame_idx in range(total_frames):
                frame = renderer(frame_idx, total_frames)
                writer.write(frame)

        except Exception as e:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
                writer = None
            if staging_p.exists():
                try:
                    staging_p.unlink(missing_ok=True)
                except Exception:
                    pass
            raise IOError(f"Failed to synthesize surveillance video for {out_p}: {e}") from e
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
                writer = None

        # Validate that the generated staging video meets technical specifications before publishing
        if not self.is_valid_video_file(
            staging_p,
            expected_width=self.WIDTH,
            expected_height=self.HEIGHT,
            expected_fps=self.FPS,
            min_frames=total_frames,
        ):
            if staging_p.exists():
                try:
                    staging_p.unlink(missing_ok=True)
                except Exception:
                    pass
            raise IOError(
                f"Generated video artifact at {staging_p} failed post-synthesis validation "
                f"for destination {out_p}."
            )

        # Safely publish staging file to final destination
        try:
            _safe_replace(staging_p, out_p)
        except Exception as e:
            if staging_p.exists():
                try:
                    staging_p.unlink(missing_ok=True)
                except Exception:
                    pass
            raise IOError(f"Failed to publish synthesized video to {out_p}: {e}") from e

        return out_p

    def ensure_demo_camera_videos(
        self,
        duration_sec: float = 5.0,
        overwrite: bool = False,
    ) -> Dict[str, Path]:
        """
        Ensures all 5 standard calibrated surveillance test clips exist in training_lab/videos/:
          - cam_01_perimeter.mp4
          - cam_02_roadway.mp4
          - cam_03_restricted.mp4
          - cam_04_terrain.mp4
          - cam_05_secondary.mp4

        Validates that existing files are non-corrupted, readable, and match 640x480 @ 30 FPS.
        If an existing asset is missing, corrupted, or stale, it is regenerated deterministically.

        Returns:
            Dict mapping camera ID to generated Path.
        """
        targets = [
            ("CAM_01", "cam_01_perimeter.mp4", "perimeter"),
            ("CAM_02", "cam_02_roadway.mp4", "roadway"),
            ("CAM_03", "cam_03_restricted.mp4", "restricted"),
            ("CAM_04", "cam_04_terrain.mp4", "terrain"),
            ("CAM_05", "cam_05_secondary.mp4", "secondary"),
        ]

        expected_frames = max(1, int(round(duration_sec * self.FPS)))
        result: Dict[str, Path] = {}
        for cam_id, filename, scn_type in targets:
            dest = self.videos_dir / filename
            is_valid = False
            if dest.is_file() and not overwrite:
                is_valid = self.is_valid_video_file(
                    filepath=dest,
                    expected_width=self.WIDTH,
                    expected_height=self.HEIGHT,
                    expected_fps=self.FPS,
                    min_frames=expected_frames,
                )
            if not is_valid or overwrite:
                self.generate_synthetic_surveillance_video(
                    output_path=dest,
                    camera_id=cam_id,
                    duration_sec=duration_sec,
                    scenario_type=scn_type,
                )
            result[cam_id] = dest

        return result


def _safe_replace(src: Path, dst: Path, max_retries: int = 15, delay: float = 0.1) -> None:
    """
    Safely moves src to dst with retry logic to handle transient Windows file locks
    (e.g., cloud sync agents, search indexers, antivirus scanners).
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            if sys.platform == "win32" and dst.exists():
                try:
                    os.replace(str(src), str(dst))
                    return
                except (PermissionError, OSError) as pe:
                    last_exc = pe
                    try:
                        dst.unlink(missing_ok=True)
                    except Exception:
                        pass
                    src.replace(dst)
                    return
            else:
                src.replace(dst)
                return
        except (PermissionError, OSError) as exc:
            last_exc = exc
            time.sleep(delay)
    if last_exc:
        raise last_exc


# Singleton instance and module-level functions
video_manager = VideoAssetManager()


def get_video_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Module-level helper to extract video metadata."""
    return video_manager.get_video_metadata(filepath)


def is_valid_video_file(
    filepath: Union[str, Path],
    expected_width: Optional[int] = VideoAssetManager.WIDTH,
    expected_height: Optional[int] = VideoAssetManager.HEIGHT,
    expected_fps: Optional[float] = VideoAssetManager.FPS,
    min_frames: int = 1,
) -> bool:
    """Module-level helper to validate video file integrity and specifications."""
    return video_manager.is_valid_video_file(
        filepath=filepath,
        expected_width=expected_width,
        expected_height=expected_height,
        expected_fps=expected_fps,
        min_frames=min_frames,
    )


def generate_synthetic_surveillance_video(
    output_path: Union[str, Path],
    camera_id: str = "CAM_01",
    duration_sec: float = 5.0,
    scenario_type: Optional[str] = None,
) -> Path:
    """Module-level helper to generate synthetic surveillance test video."""
    return video_manager.generate_synthetic_surveillance_video(
        output_path=output_path,
        camera_id=camera_id,
        duration_sec=duration_sec,
        scenario_type=scenario_type,
    )


def ensure_demo_camera_videos(
    duration_sec: float = 5.0,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Module-level helper to generate all 5 standard demo camera test clips."""
    return video_manager.ensure_demo_camera_videos(
        duration_sec=duration_sec,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    print("[VideoManager] Synthesizing demo videos...")
    generated = ensure_demo_camera_videos(overwrite=True)
    for cid, p in generated.items():
        m = get_video_metadata(p)
        print(f"  {cid}: {p.name} -> {m['width']}x{m['height']} @ {m['fps']}fps, {m['frame_count']} frames ({m['file_size_mb']} MB)")
