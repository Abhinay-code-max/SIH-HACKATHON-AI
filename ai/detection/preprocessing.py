"""
Frame Preprocessing Pipeline for Tactical Detection.
Provides enhancement transforms for night, fog, and low-visibility conditions.
Operates strictly offline using NumPy and OpenCV (CPU-based, zero Torch dependency).
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


class FramePreprocessor:
    """Applies optional preprocessing transforms to frames before detection.

    Supports: histogram equalization, adaptive histogram (CLAHE), gamma correction,
    dehazing approximation, and grayscale conversion for IR-style inputs.
    """

    def __init__(
        self,
        enable_clahe: bool = False,   # Contrast Limited Adaptive Histogram Equalization
        enable_gamma: bool = False,    # Gamma correction
        gamma_value: float = 1.2,      # >1 brightens, <1 darkens
        enable_dehaze: bool = False,   # Simple dehazing via guided filter approximation
        enable_grayscale_ir: bool = False,  # Convert to 3-channel grayscale (simulate IR)
        resize_width: Optional[int] = None,  # Optional resize before detection
        resize_height: Optional[int] = None,
        enabled: bool = False,         # Master toggle
        **kwargs: Any,
    ):
        # Support aliases from YAML config if provided directly
        if "clahe" in kwargs:
            enable_clahe = bool(kwargs["clahe"])
        if "gamma" in kwargs:
            g_val = kwargs["gamma"]
            if isinstance(g_val, (int, float)):
                gamma_value = float(g_val)
                enable_gamma = (gamma_value != 1.0)
            elif isinstance(g_val, bool):
                enable_gamma = g_val
        if "dehaze" in kwargs:
            enable_dehaze = bool(kwargs["dehaze"])
        if "grayscale_ir" in kwargs:
            enable_grayscale_ir = bool(kwargs["grayscale_ir"])
        if "resize_width" in kwargs and resize_width is None:
            resize_width = kwargs["resize_width"]
        if "resize_height" in kwargs and resize_height is None:
            resize_height = kwargs["resize_height"]
        if "enabled" in kwargs:
            enabled = bool(kwargs["enabled"])

        self.enabled = bool(enabled)
        self.enable_clahe = bool(enable_clahe)
        self.enable_gamma = bool(enable_gamma)
        self.gamma_value = float(gamma_value)
        self.enable_dehaze = bool(enable_dehaze)
        self.enable_grayscale_ir = bool(enable_grayscale_ir)
        self.resize_width = int(resize_width) if resize_width is not None else None
        self.resize_height = int(resize_height) if resize_height is not None else None

        # Precompute gamma lookup table if enabled
        self._gamma_table = self._build_gamma_table(self.gamma_value) if self.enable_gamma else None

    @staticmethod
    def _build_gamma_table(gamma: float) -> Optional[np.ndarray]:
        if gamma <= 0 or gamma == 1.0:
            return None
        inv_gamma = 1.0 / gamma
        return np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)

    @property
    def is_active(self) -> bool:
        """Return True if preprocessor is master-enabled and at least one transform is active."""
        if not self.enabled:
            return False
        return (
            self.enable_clahe
            or self.enable_gamma
            or self.enable_dehaze
            or self.enable_grayscale_ir
            or (self.resize_width is not None and self.resize_height is not None)
        )

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply enabled transforms in order. Returns a copy of the frame."""
        if frame is None or frame.size == 0:
            return frame

        out = frame.copy()
        if not self.enabled:
            return out

        if self.enable_grayscale_ir:
            out = self._apply_grayscale_ir(out)
        if self.enable_dehaze:
            out = self._apply_dehaze(out)
        if self.enable_clahe:
            out = self._apply_clahe(out)
        if self.enable_gamma:
            out = self._apply_gamma(out)
        if self.resize_width is not None and self.resize_height is not None:
            out = self._apply_resize(out)

        return out

    def _apply_clahe(self, img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply CLAHE on the luminance channel (LAB color space for BGR, direct for grayscale)."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        if len(img.shape) == 2:
            return clahe.apply(img)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            return clahe.apply(img[:, :, 0])[:, :, np.newaxis]
        elif len(img.shape) == 3 and img.shape[2] == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img

    def _apply_gamma(self, img: np.ndarray) -> np.ndarray:
        """Apply gamma correction using precomputed LUT (gamma > 1 brightens, < 1 darkens)."""
        if self._gamma_table is None:
            self._gamma_table = self._build_gamma_table(self.gamma_value)
        if self._gamma_table is not None:
            return cv2.LUT(img, self._gamma_table)
        return img

    def _apply_dehaze(self, img: np.ndarray) -> np.ndarray:
        """Simple dehazing approximation via dark channel prior and guided filtering."""
        if len(img.shape) != 3 or img.shape[2] != 3:
            return img
        min_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        dark = cv2.erode(min_channel, kernel)
        transmission = 1.0 - 0.75 * (cv2.blur(dark, (11, 11)).astype(np.float32) / 255.0)
        transmission = np.clip(transmission, 0.2, 1.0)[:, :, np.newaxis]
        A = 220.0
        dehazed = (img.astype(np.float32) - A) / transmission + A
        return np.clip(dehazed, 0, 255).astype(np.uint8)

    def _apply_grayscale_ir(self, img: np.ndarray) -> np.ndarray:
        """Convert to 3-channel grayscale to simulate thermal/IR camera imagery."""
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return img

    def _apply_resize(self, img: np.ndarray) -> np.ndarray:
        """Resize frame to configured target dimensions."""
        if (
            self.resize_width is not None
            and self.resize_height is not None
            and self.resize_width > 0
            and self.resize_height > 0
        ):
            return cv2.resize(img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_LINEAR)
        return img

    def to_dict(self) -> Dict[str, Any]:
        """Serialise current settings for config persistence."""
        return {
            "enabled": self.enabled,
            "enable_clahe": self.enable_clahe,
            "enable_gamma": self.enable_gamma,
            "gamma_value": self.gamma_value,
            "enable_dehaze": self.enable_dehaze,
            "enable_grayscale_ir": self.enable_grayscale_ir,
            "resize_width": self.resize_width,
            "resize_height": self.resize_height,
        }
