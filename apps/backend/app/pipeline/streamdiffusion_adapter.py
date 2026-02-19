from __future__ import annotations

import hashlib
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.config import RuntimeConfig
from app.pipeline.causal_wan_runtime import CausalWanRealtimeRunner, RuntimeLoadError
from app.pipeline.types import ConditioningSignals


class StreamDiffusionV2Adapter:
    """Adapter layer.

    - Uses vendored StreamDiffusionV2 core when runtime dependencies and checkpoints are available.
    - Falls back to a deterministic mock stylizer when runtime is unavailable.
    """

    def __init__(self) -> None:
        self._stream_runtime: CausalWanRealtimeRunner | None = None
        self._runtime_mode = "mock"
        self._runtime_error: str | None = None
        self._runtime_path: str | None = None
        self._core_import_ready = False
        self._bootstrap_runtime()

    def generate(
        self,
        frame: Image.Image,
        config: RuntimeConfig,
        conditioning: ConditioningSignals,
    ) -> Image.Image:
        if config.streamdiffusionv2_path and config.streamdiffusionv2_path != self._runtime_path:
            self._bootstrap_runtime(config.streamdiffusionv2_path)

        if self._core_import_ready:
            try:
                core_path = Path(self._runtime_path) if self._runtime_path else self._resolve_core_path(None)

                if self._stream_runtime is None:
                    self._stream_runtime = CausalWanRealtimeRunner(core_path=core_path, config=config)
                elif self._stream_runtime.requires_reload(config=config, core_path=core_path):
                    self._stream_runtime = CausalWanRealtimeRunner(core_path=core_path, config=config)

                output = self._stream_runtime.process_frame(frame, config, conditioning)
                self._runtime_mode = self._stream_runtime.mode
                self._runtime_error = None

                if output is not None:
                    return output
                if self._stream_runtime.last_output is not None:
                    return self._stream_runtime.last_output

            except RuntimeLoadError as error:
                self._runtime_error = str(error)
                self._runtime_mode = "mock"
                self._stream_runtime = None
            except Exception as error:  # pragma: no cover - runtime dependent
                self._runtime_error = f"runtime inference failed: {error}"
                self._runtime_mode = "mock"
                self._stream_runtime = None

        return self._run_mock_stylizer(frame, config, conditioning)

    @property
    def runtime_status(self) -> dict[str, str | int | bool | float | None]:
        payload: dict[str, str | int | bool | float | None] = {
            "mode": self._runtime_mode,
            "runtime_path": self._runtime_path,
            "error": self._runtime_error,
        }
        if self._stream_runtime is not None:
            payload.update(self._stream_runtime.stats)
        return payload

    def _bootstrap_runtime(self, explicit_path: str | None = None) -> None:
        core_path = self._resolve_core_path(explicit_path)
        self._runtime_path = str(core_path)
        self._runtime_error = None
        self._runtime_mode = "mock"
        self._core_import_ready = False
        self._stream_runtime = None

        if not core_path.exists():
            self._runtime_error = f"core path not found: {core_path}"
            return

        core_path_str = str(core_path)
        if core_path_str not in sys.path:
            sys.path.insert(0, core_path_str)

        try:
            from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline  # noqa: F401
            from omegaconf import OmegaConf  # noqa: F401
        except Exception as error:  # pragma: no cover - environment dependent
            self._runtime_error = f"failed to import StreamDiffusionV2 core: {error}"
            return

        self._core_import_ready = True
        self._runtime_mode = "core-imported"

    @staticmethod
    def _resolve_core_path(explicit_path: str | None) -> Path:
        if explicit_path:
            return Path(explicit_path).expanduser().resolve()

        env_path = os.getenv("STREAMDIFFUSIONV2_PATH")
        if env_path:
            return Path(env_path).expanduser().resolve()

        repo_root = Path(__file__).resolve().parents[4]
        return repo_root / "core" / "streamdiffusionv2"

    def _run_mock_stylizer(
        self,
        frame: Image.Image,
        config: RuntimeConfig,
        conditioning: ConditioningSignals,
    ) -> Image.Image:
        base = frame.resize((config.output_width, config.output_height), Image.Resampling.BILINEAR)

        # Prompt dominance influences how much the original identity gets abstracted away.
        prompt_mix = config.prompt_dominance
        identity_keep = config.identity_lock if config.preserve_identity else config.identity_lock * 0.35

        seed = config.seed if config.seed is not None else self._seed_from_prompt(config.prompt)
        rng = random.Random(seed)

        tint = np.array(
            [
                rng.uniform(0.15, 0.95),
                rng.uniform(0.15, 0.95),
                rng.uniform(0.15, 0.95),
            ],
            dtype=np.float32,
        )

        # Remove fine personal details when prompt dominance is high.
        blur_radius = max(0.0, (prompt_mix - identity_keep) * 2.2)
        stylized = base.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        stylized_arr = np.asarray(stylized, dtype=np.float32) / 255.0

        tinted = (stylized_arr * (1.0 - 0.45 * prompt_mix)) + (tint * (0.45 * prompt_mix))

        edge = base.convert("L").filter(ImageFilter.FIND_EDGES)
        edge_arr = np.asarray(edge, dtype=np.float32) / 255.0
        edge_rgb = np.repeat(edge_arr[..., None], 3, axis=2)

        structure_lock = max(config.pose_lock, config.depth_lock, config.segmentation_lock)
        motion_factor = 1.0 - (conditioning.motion_score * config.motion_smoothing)
        structure_mix = min(1.0, 0.25 + structure_lock * 0.55) * motion_factor

        composed = tinted * (1.0 - structure_mix) + edge_rgb * structure_mix
        composed = np.clip(composed, 0.0, 1.0)

        out = Image.fromarray((composed * 255).astype(np.uint8), mode="RGB")

        # Guidance/steps are mapped to contrast and sharpness in mock mode.
        contrast = 1.0 + (config.guidance_scale / 20.0)
        sharpness = 0.9 + (config.inference_steps * 0.08)

        out = ImageEnhance.Contrast(out).enhance(contrast)
        out = ImageEnhance.Sharpness(out).enhance(sharpness)
        return out

    @staticmethod
    def _seed_from_prompt(prompt: str) -> int:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16)
