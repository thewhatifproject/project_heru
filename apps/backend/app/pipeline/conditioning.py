from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from app.pipeline.types import ConditioningSignals


class ConditioningExtractor:
    """Lightweight conditioning proxy.

    Placeholder module for webcam-focused pose/depth/seg pipelines.
    In production you can replace this with MediaPipe + depth estimator + segmentation model.
    """

    def __init__(self) -> None:
        self._last_luma_mean: float | None = None

    def extract(self, frame: Image.Image) -> ConditioningSignals:
        gray = frame.convert("L")
        luma = np.asarray(gray, dtype=np.float32) / 255.0
        luma_mean = float(luma.mean())

        if self._last_luma_mean is None:
            motion_score = 0.0
        else:
            motion_score = min(1.0, abs(luma_mean - self._last_luma_mean) * 8.0)

        edge = gray.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.asarray(edge, dtype=np.float32) / 255.0
        edge_density = float((edge_arr > 0.2).mean())

        self._last_luma_mean = luma_mean
        return ConditioningSignals(
            motion_score=motion_score,
            edge_density=edge_density,
            luma_mean=luma_mean,
        )
