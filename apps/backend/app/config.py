from __future__ import annotations

from copy import deepcopy
import os
from typing import Any, Literal

from pydantic import BaseModel, Field


class OutputConfig(BaseModel):
    preview: bool = True
    virtual_camera: bool = False
    rtmp_enabled: bool = False
    rtmp_url: str | None = None


class RuntimeConfig(BaseModel):
    prompt: str = "futuristic cyborg portrait, cinematic lighting"
    negative_prompt: str = "blurry, low quality, beard, watermark"
    mode: Literal["performance", "balanced", "quality"] = "balanced"
    model_variant: Literal["wan-1.3b", "wan-14b"] = "wan-1.3b"
    inference_topology: Literal["single", "distributed"] = Field(
        default_factory=lambda: os.getenv("INFERENCE_TOPOLOGY", "single")
    )
    distributed_world_size: int = Field(
        default_factory=lambda: int(os.getenv("DISTRIBUTED_WORLD_SIZE", "2")), ge=1, le=8
    )
    distributed_master_addr: str = Field(
        default_factory=lambda: os.getenv("DISTRIBUTED_MASTER_ADDR", "127.0.0.1")
    )
    distributed_master_port: int = Field(
        default_factory=lambda: int(os.getenv("DISTRIBUTED_MASTER_PORT", "29501")),
        ge=1024,
        le=65535,
    )
    distributed_command_timeout_s: float = Field(default=120.0, ge=5.0, le=600.0)

    prompt_dominance: float = Field(default=0.82, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=6.5, ge=1.0, le=20.0)
    inference_steps: int = Field(default=2, ge=1, le=8)
    seed: int | None = None

    preserve_identity: bool = False
    identity_lock: float = Field(default=0.2, ge=0.0, le=1.0)
    pose_lock: float = Field(default=0.9, ge=0.0, le=1.0)
    depth_lock: float = Field(default=0.75, ge=0.0, le=1.0)
    segmentation_lock: float = Field(default=0.65, ge=0.0, le=1.0)
    motion_smoothing: float = Field(default=0.28, ge=0.0, le=1.0)

    target_fps: int = Field(default=30, ge=5, le=60)
    output_width: int = Field(default=768, ge=256, le=1920)
    output_height: int = Field(default=768, ge=256, le=1920)
    jpeg_quality: int = Field(default=85, ge=50, le=100)

    outputs: OutputConfig = Field(default_factory=OutputConfig)
    streamdiffusionv2_path: str | None = Field(
        default_factory=lambda: os.getenv("STREAMDIFFUSIONV2_PATH")
    )


DEFAULT_PRESETS: dict[str, RuntimeConfig] = {
    "a": RuntimeConfig(
        prompt="ultra-detailed cyberpunk cyborg, metallic skin, hard rim light",
        negative_prompt="beard, facial hair, blur, text, artifacts",
        mode="performance",
        model_variant="wan-1.3b",
        inference_topology="single",
        inference_steps=2,
        prompt_dominance=0.86,
    ),
    "b": RuntimeConfig(
        prompt="biomechanical android, clean face, synthetic materials, studio photo",
        negative_prompt="beard, moustache, blur, noisy background",
        mode="balanced",
        model_variant="wan-1.3b",
        inference_topology="single",
        inference_steps=3,
        prompt_dominance=0.8,
    ),
}


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_runtime_config(current: RuntimeConfig, patch: dict[str, Any]) -> RuntimeConfig:
    merged_dict = deep_merge(current.model_dump(), patch)
    return RuntimeConfig.model_validate(merged_dict)
