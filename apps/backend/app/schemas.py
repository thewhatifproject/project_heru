from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WsMessage(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ProcessedFramePayload(BaseModel):
    image_b64: str
    timestamp_ms: int
    latency_ms: float
    config_revision: int
    conditioning: dict[str, float]


class HealthResponse(BaseModel):
    status: str = "ok"
    active_sessions: int
