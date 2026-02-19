from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import DEFAULT_PRESETS, RuntimeConfig, merge_runtime_config
from app.pipeline.runner import RealtimePipeline


@dataclass
class SessionMetrics:
    frames_in: int = 0
    frames_out: int = 0
    avg_latency_ms: float = 0.0


@dataclass
class RealtimeSession:
    session_id: str
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    revision: int = 1
    metrics: SessionMetrics = field(default_factory=SessionMetrics)

    def __post_init__(self) -> None:
        self._pipeline = RealtimePipeline()
        self._lock = asyncio.Lock()

    @property
    def runtime_status(self) -> dict[str, str | None]:
        return self._pipeline.runtime_status

    async def update_config(self, patch: dict[str, Any]) -> RuntimeConfig:
        async with self._lock:
            self.config = merge_runtime_config(self.config, patch)
            self.revision += 1
            return self.config

    async def apply_preset(self, preset_name: str) -> RuntimeConfig:
        preset = DEFAULT_PRESETS.get(preset_name.lower())
        if preset is None:
            raise ValueError(f"Unknown preset: {preset_name}")

        async with self._lock:
            self.config = preset.model_copy(deep=True)
            self.revision += 1
            return self.config

    async def process_frame(
        self,
        image_b64: str,
        timestamp_ms: int,
    ) -> dict[str, Any]:
        frame_bytes = base64.b64decode(image_b64)

        started = time.perf_counter()
        async with self._lock:
            config = self.config

        result = await asyncio.to_thread(self._pipeline.process, frame_bytes, config)
        latency_ms = (time.perf_counter() - started) * 1000

        self.metrics.frames_in += 1
        self.metrics.frames_out += 1
        self.metrics.avg_latency_ms = (
            self.metrics.avg_latency_ms * 0.9 + latency_ms * 0.1
            if self.metrics.frames_out > 1
            else latency_ms
        )

        return {
            "image_b64": base64.b64encode(result.image_bytes).decode("ascii"),
            "timestamp_ms": timestamp_ms,
            "latency_ms": round(latency_ms, 2),
            "config_revision": self.revision,
            "conditioning": {
                "motion_score": round(result.conditioning.motion_score, 4),
                "edge_density": round(result.conditioning.edge_density, 4),
                "luma_mean": round(result.conditioning.luma_mean, 4),
            },
        }


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, RealtimeSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str) -> RealtimeSession:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = RealtimeSession(session_id=session_id)
            return self._sessions[session_id]

    async def active_count(self) -> int:
        async with self._lock:
            return len(self._sessions)
