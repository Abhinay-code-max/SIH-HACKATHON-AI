"""
Visual Appearance Feature Extractor for Re-Identification.
Combines:
1. 512-dimensional deep neural embedding from local YOLOv8l backbone.
2. Multi-zone spatial-color HSV histograms (head/torso/legs).
Strictly 100% offline, running on local hardware (NVIDIA RTX 4060 GPU).
"""

from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.inference.loader import get_device, load_local_model


class VisualFeatureExtractor:
    def __init__(self, model_path: Optional[str] = None):
        self.device = get_device()
        self.model_path = model_path or self._resolve_default_model()
        self._model = None

    def _resolve_default_model(self) -> str:
        """Finds latest registered model or default YOLO-L weights."""
        best_v2 = ROOT_DIR / "models" / "registry" / "YOLO-L-v002" / "weights" / "best.pt"
        if best_v2.is_file():
            return str(best_v2)
        best_v1 = ROOT_DIR / "models" / "registry" / "YOLO-L-v001" / "weights" / "best.pt"
        if best_v1.is_file():
            return str(best_v1)
        return "yolov8l.pt"

    def _ensure_model(self):
        if self._model is None:
            self._model = load_local_model(self.model_path)

    def extract_deep_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts 512-dim deep backbone embedding using local YOLO.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return np.zeros(512, dtype=np.float32)

        self._ensure_model()
        try:
            # Resize crop to standard Re-ID dimension (128w x 256h)
            resized = cv2.resize(crop, (128, 256))
            feats = self._model.embed(resized)
            if isinstance(feats, list) and len(feats) > 0:
                t = feats[0]
                if isinstance(t, torch.Tensor):
                    vec = t.detach().cpu().numpy().flatten().astype(np.float32)
                else:
                    vec = np.asarray(t, dtype=np.float32).flatten()
                norm = np.linalg.norm(vec)
                return vec / (norm + 1e-7)
        except Exception:
            pass

        return np.zeros(512, dtype=np.float32)

    def extract_spatial_color_histogram(self, crop: np.ndarray) -> np.ndarray:
        """
        Computes 3-zone spatial HSV color histogram:
        - Zone 1: Upper 35% (Head, hat, hair, collar)
        - Zone 2: Middle 40% (Torso, jacket, upper body)
        - Zone 3: Lower 25% (Legs, pants, footwear or chassis)
        """
        if crop is None or crop.size == 0:
            return np.zeros(144, dtype=np.float32)

        h, w = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Zone boundaries
        z1_end = max(1, int(h * 0.35))
        z2_end = max(z1_end + 1, int(h * 0.75))

        zones = [
            hsv[0:z1_end, :, :],
            hsv[z1_end:z2_end, :, :],
            hsv[z2_end:h, :, :],
        ]

        hist_vectors = []
        for zone in zones:
            if zone.size == 0:
                hist_vectors.append(np.zeros(48, dtype=np.float32))
                continue

            # 2D H-S histogram: 16 Hue bins (0-180), 3 Saturation bins (0-256) = 48 bins
            hist = cv2.calcHist([zone], [0, 1], None, [16, 3], [0, 180, 0, 256])
            hist = hist.flatten().astype(np.float32)
            norm = np.linalg.norm(hist)
            hist_vectors.append(hist / (norm + 1e-7))

        combined_hist = np.concatenate(hist_vectors)
        norm = np.linalg.norm(combined_hist)
        return combined_hist / (norm + 1e-7)

    def extract_signature(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts composite appearance signature vector (656-dim: 512 deep + 144 color).
        Weighted combination normalized to unit length.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return np.zeros(656, dtype=np.float32)

        deep_vec = self.extract_deep_embedding(crop)  # 512-dim
        color_vec = self.extract_spatial_color_histogram(crop)  # 144-dim

        # Weighted concatenation (deep 70%, color 30%)
        weighted_deep = deep_vec * 0.70
        weighted_color = color_vec * 0.30

        signature = np.concatenate([weighted_deep, weighted_color]).astype(np.float32)
        norm = np.linalg.norm(signature)
        return signature / (norm + 1e-7)

    @staticmethod
    def compute_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Calculates cosine similarity between two feature signatures [0.0, 1.0]."""
        if sig_a is None or sig_b is None or len(sig_a) == 0 or len(sig_b) == 0:
            return 0.0
        dot = float(np.dot(sig_a, sig_b))
        norm_a = float(np.linalg.norm(sig_a))
        norm_b = float(np.linalg.norm(sig_b))
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        sim = dot / (norm_a * norm_b)
        return max(0.0, min(1.0, sim))


# Global singleton instance
feature_extractor = VisualFeatureExtractor()
