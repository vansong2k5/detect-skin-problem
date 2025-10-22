"""Vision feature extraction utilities for the hybrid dermatology pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Optional

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


@dataclass(frozen=True)
class VisionConfig:
    """Configuration for the :class:`VisionFeatureExtractor`."""

    device: str = "cpu"
    half_precision: bool = False


class VisionFeatureExtractor(nn.Module):
    """Extract EfficientNet based embeddings and hand crafted signals."""

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        super().__init__()
        self.config = config or VisionConfig()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.preprocess = weights.transforms()
        backbone = efficientnet_b0(weights=weights)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to(self.config.device)
        self.eval()

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        features = self.backbone(batch)
        pooled = self.pool(features).flatten(1)
        if self.config.half_precision:
            pooled = pooled.half()
        return pooled

    def embed_array(self, img_bgr: np.ndarray) -> np.ndarray:
        tensor = self._prepare_tensor(img_bgr)
        embedding = self.forward(tensor).squeeze(0).cpu().numpy()
        return embedding

    def embed_path(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Unable to read image at path: {path}")
        return self.embed_array(image)

    def _prepare_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(img_rgb).unsqueeze(0).to(self.config.device)
        if self.config.half_precision:
            tensor = tensor.half()
        return tensor


def compute_image_signals(img_bgr: np.ndarray) -> Dict[str, float]:
    """Compute lightweight quantitative descriptors from a BGR image."""

    if img_bgr is None:
        raise ValueError("Input image array cannot be None")

    h, w = img_bgr.shape[:2]
    area = float(h * w)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_ratio = float(np.count_nonzero(red_mask)) / max(area, 1.0)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_irregularity = float(laplacian.var())

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(edges.mean())

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 5000
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(255 - gray)
    blister_count = int(len(keypoints))

    lesion_area = red_ratio * area

    return {
        "red_ratio": round(red_ratio, 6),
        "texture_irregularity": round(texture_irregularity, 6),
        "edge_density": round(edge_density, 6),
        "blister_count": blister_count,
        "lesion_area": round(lesion_area, 3),
    }


@lru_cache(maxsize=1)
def get_extractor(device: str = "cpu", half_precision: bool = False) -> VisionFeatureExtractor:
    """Return a cached :class:`VisionFeatureExtractor`."""

    return VisionFeatureExtractor(VisionConfig(device=device, half_precision=half_precision))


def batch_embed(paths: Iterable[str], device: str = "cpu") -> np.ndarray:
    """Embed a collection of image paths into a matrix."""

    extractor = get_extractor(device=device)
    vectors = [extractor.embed_path(path) for path in paths]
    return np.vstack(vectors)
