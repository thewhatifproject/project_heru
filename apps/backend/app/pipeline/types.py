from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ConditioningSignals:
    motion_score: float
    edge_density: float
    luma_mean: float


@dataclass(slots=True)
class PipelineResult:
    image_bytes: bytes
    conditioning: ConditioningSignals
