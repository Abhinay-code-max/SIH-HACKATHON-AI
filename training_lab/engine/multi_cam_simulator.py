"""
Five-Camera Synchronous Simulation Engine.
Plays up to 5 camera feeds concurrently (CAM_01 through CAM_05) while preserving
strict camera attribution, multi-camera time synchronization, loop playback,
and frame-drop tolerance for the AI Training, Validation & Simulation Lab.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class MultiCameraSimulator:
    """
    Synchronous multi-camera playback simulator for border perimeter surveillance.
    Synchronizes up to 5 independent camera streams into bundled multi-feed frames.
    """

    SUPPORTED_CAMERAS = ["CAM_01", "CAM_02", "CAM_03", "CAM_04", "CAM_05"]

    def __init__(
        self,
        camera_bindings: Optional[Union[Dict[str, Any], Any]] = None,
        loop: bool = True,
        fps: float = 30.0,
    ):
        """
        Initializes the Multi-Camera Synchronous Simulator.

        Args:
            camera_bindings: Dict mapping camera_id (e.g. 'CAM_01') to video file path
                             or a Scenario object containing assigned_cameras.
            loop: Whether to loop camera feeds continuously upon reaching end of stream.
            fps: Expected simulation FPS rate (default 30.0).
        """
        self.loop = loop
        self.fps = float(fps)
        self._frame_idx = 0
        self._captures: Dict[str, cv2.VideoCapture] = {}
        self._video_paths: Dict[str, Path] = {}
        self._frame_counts: Dict[str, int] = {}
        self._stream_metadata: Dict[str, Dict[str, Any]] = {}
        self._last_frames: Dict[str, np.ndarray] = {}
        self._camera_finished: Dict[str, bool] = {}

        if camera_bindings:
            try:
                self.bind_cameras(camera_bindings)
            except Exception:
                self.close()
                raise

    def bind_camera(self, camera_id: str, video_path: Union[str, Path]) -> None:
        """
        Binds an individual camera stream (CAM_01 through CAM_05) to a video file.

        Args:
            camera_id: camera identifier (e.g. CAM_01)
            video_path: path to MP4 video file
        """
        cam_id = camera_id.upper().strip()
        path_obj = Path(video_path)
        if not path_obj.is_absolute():
            path_obj = (ROOT_DIR / path_obj).resolve()
        else:
            path_obj = path_obj.resolve()

        if not path_obj.is_file():
            raise FileNotFoundError(f"Video file for {cam_id} not found: {path_obj}")

        # Close existing capture if already bound
        if cam_id in self._captures:
            try:
                self._captures[cam_id].release()
            except Exception:
                pass

        cap = cv2.VideoCapture(str(path_obj))
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            raise ValueError(f"Failed to open video capture for {cam_id}: {path_obj}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_fps = float(cap.get(cv2.CAP_PROP_FPS)) or self.fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._captures[cam_id] = cap
        self._video_paths[cam_id] = path_obj
        self._frame_counts[cam_id] = frame_count
        self._camera_finished[cam_id] = False
        self._stream_metadata[cam_id] = {
            "camera_id": cam_id,
            "filepath": str(path_obj),
            "width": width,
            "height": height,
            "fps": cam_fps,
            "frame_count": frame_count,
        }

        # Prime initial reference frame
        ret, initial_frame = cap.read()
        if ret and initial_frame is not None:
            self._last_frames[cam_id] = initial_frame.copy()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            # Fallback zero placeholder frame
            self._last_frames[cam_id] = np.zeros((height, width, 3), dtype=np.uint8)

    def bind_cameras(self, camera_bindings: Union[Dict[str, Any], Any]) -> None:
        """
        Binds multiple camera streams simultaneously. Supports either a dictionary
        of {camera_id: path_or_dict} or a Scenario model instance.
        """
        # Handle Scenario Pydantic object or dict with assigned_cameras
        if hasattr(camera_bindings, "assigned_cameras"):
            assigned = camera_bindings.assigned_cameras
            for cam_id, cam_bind in assigned.items():
                vpath = getattr(cam_bind, "video_file", None)
                if vpath:
                    self.bind_camera(cam_id, vpath)
            return

        if isinstance(camera_bindings, dict):
            if "assigned_cameras" in camera_bindings:
                for cam_id, cdata in camera_bindings["assigned_cameras"].items():
                    vpath = cdata.get("video_file") if isinstance(cdata, dict) else getattr(cdata, "video_file", None)
                    if vpath:
                        self.bind_camera(cam_id, vpath)
            else:
                for cam_id, vpath in camera_bindings.items():
                    if isinstance(vpath, dict):
                        vpath = vpath.get("video_file") or vpath.get("filepath")
                    if vpath:
                        self.bind_camera(cam_id, vpath)

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Steps the multi-camera simulation forward by one synchronized frame bundle.

        Returns:
            Dict containing:
                "frame_idx": int,
                "timestamp": float,
                "feeds": {
                    "CAM_01": {"camera_id": "CAM_01", "frame": np.ndarray, "shape": (480, 640, 3)},
                    ...
                }
            Or None if loop=False and all camera streams have ended.
        """
        if not self._captures:
            return None

        # Check if all streams have finished under loop=False
        if not self.loop and all(self._camera_finished.values()):
            return None

        feeds: Dict[str, Dict[str, Any]] = {}
        curr_frame_idx = self._frame_idx
        timestamp = round(curr_frame_idx / self.fps, 4)

        for cam_id, cap in sorted(self._captures.items()):
            ret, frame = cap.read()

            # Handle end-of-stream or frame-drop tolerance
            if not ret or frame is None:
                if self.loop:
                    # Rewind to start and retry
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                if not ret or frame is None:
                    # Mark stream as ended or hold last frame
                    self._camera_finished[cam_id] = True
                    if cam_id in self._last_frames:
                        frame = self._last_frames[cam_id].copy()
                        ret = True
                    else:
                        h = self._stream_metadata[cam_id].get("height", 480)
                        w = self._stream_metadata[cam_id].get("width", 640)
                        frame = np.zeros((h, w, 3), dtype=np.uint8)
                        ret = True

            # If successfully retrieved or held from last frame
            if ret and frame is not None:
                # Retain copy for last frame cache and isolate array memory
                clean_frame = frame.copy()
                self._last_frames[cam_id] = clean_frame.copy()

                # Strict camera attribution
                feeds[cam_id] = {
                    "camera_id": cam_id,
                    "frame": clean_frame,
                    "shape": clean_frame.shape,
                    "timestamp": timestamp,
                    "frame_idx": curr_frame_idx,
                }

        # Check again if all cameras completed under loop=False
        if not self.loop and all(self._camera_finished.values()) and len(feeds) == 0:
            return None

        self._frame_idx += 1

        return {
            "frame_idx": curr_frame_idx,
            "timestamp": timestamp,
            "feeds": feeds,
        }

    def seek(self, frame_idx: int) -> bool:
        """
        Seeks all bound camera streams to a target frame index.

        Args:
            frame_idx: target frame position (0-indexed)

        Returns:
            bool: True if seek succeeded across all streams
        """
        target_idx = max(0, int(frame_idx))
        self._frame_idx = target_idx

        for cam_id, cap in self._captures.items():
            total = self._frame_counts.get(cam_id, 0)
            if total > 0:
                pos = target_idx % total if self.loop else min(target_idx, total - 1)
            else:
                pos = 0

            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            self._camera_finished[cam_id] = False

        return True

    def reset(self) -> None:
        """Resets all camera feeds to the beginning (frame 0)."""
        self.seek(0)

    def close(self) -> None:
        """Releases all open video capture streams."""
        for cam_id, cap in list(self._captures.items()):
            try:
                cap.release()
            except Exception:
                pass
        self._captures.clear()
        self._camera_finished.clear()
        self._last_frames.clear()

    def release(self) -> None:
        """Alias for close()."""
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def get_active_cameras(self) -> List[str]:
        """Returns list of currently active camera IDs."""
        return sorted(list(self._captures.keys()))

    def get_camera_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Returns metadata dictionary for all bound cameras."""
        return self._stream_metadata.copy()

    def is_running(self) -> bool:
        """Checks whether the simulator has active streams open."""
        return len(self._captures) > 0 and (self.loop or not all(self._camera_finished.values()))

    def __enter__(self) -> "MultiCameraSimulator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


if __name__ == "__main__":
    from training_lab.engine.video_manager import ensure_demo_camera_videos

    print("[MultiCameraSimulator] Initializing 5-camera test run...")
    demo_vids = ensure_demo_camera_videos()
    sim = MultiCameraSimulator(camera_bindings=demo_vids, loop=True, fps=30.0)

    print(f"  Bound cameras: {sim.get_active_cameras()}")
    for step_num in range(5):
        bundle = sim.step()
        if bundle:
            cams = list(bundle["feeds"].keys())
            shapes = {c: bundle["feeds"][c]["shape"] for c in cams}
            print(f"  Step {bundle['frame_idx']} @ {bundle['timestamp']}s: cameras={cams}, shapes={shapes}")

    sim.close()
    print("[MultiCameraSimulator] Test run completed.")
