from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import RuntimeConfig
from app.pipeline.types import ConditioningSignals


class RuntimeLoadError(RuntimeError):
    """Raised when the real StreamDiffusionV2 runtime cannot be initialized."""


@dataclass(frozen=True)
class RuntimeSignature:
    core_path: str
    model_variant: str
    output_width: int
    output_height: int


class CausalWanRealtimeRunner:
    """Thin realtime wrapper over StreamDiffusionV2 causal WAN inference.

    The runner keeps a streaming state:
    - warmup window for initial causal cache priming
    - frame chunks for iterative inference
    - prompt/steps-driven reset logic
    """

    def __init__(self, core_path: Path, config: RuntimeConfig) -> None:
        self.core_path = core_path
        self.output_width = self._snap_size(config.output_width)
        self.output_height = self._snap_size(config.output_height)
        self.signature = RuntimeSignature(
            core_path=str(core_path),
            model_variant=config.model_variant,
            output_width=self.output_width,
            output_height=self.output_height,
        )

        self._torch: Any = None
        self._pipeline: Any = None
        self._device: Any = None

        self._chunk_size = 4
        self._init_window = 5
        self._initial_frames: deque[Any] = deque(maxlen=self._init_window)
        self._pending_frames: list[Any] = []
        self._prepared = False
        self._processed_chunks = 0
        self._current_start = 0
        self._current_end = 0
        self._noise_scale = 0.68
        self._t_refresh = 50
        self._step_schedule_full: list[int] = [700, 500, 0]

        self._last_output: Image.Image | None = None
        self._prompt_key: tuple[str, str] | None = None
        self._steps_key: int | None = None

        self._load_runtime(config)
        self._reset_stream_state(config)

    @property
    def mode(self) -> str:
        return "core-runtime-active" if self._prepared else "core-warmup"

    @property
    def last_output(self) -> Image.Image | None:
        return self._last_output

    @property
    def stats(self) -> dict[str, str | int | bool | float]:
        return {
            "prepared": self._prepared,
            "processed_chunks": self._processed_chunks,
            "chunk_size": self._chunk_size,
            "warmup_frames": self._init_window,
            "noise_scale": round(float(self._noise_scale), 4),
        }

    def requires_reload(self, config: RuntimeConfig, core_path: Path) -> bool:
        desired = RuntimeSignature(
            core_path=str(core_path),
            model_variant=config.model_variant,
            output_width=self._snap_size(config.output_width),
            output_height=self._snap_size(config.output_height),
        )
        return desired != self.signature

    def process_frame(
        self,
        frame: Image.Image,
        config: RuntimeConfig,
        conditioning: ConditioningSignals,
    ) -> Image.Image | None:
        self._handle_dynamic_updates(config)

        frame_tensor = self._frame_to_tensor(frame)

        if not self._prepared:
            self._initial_frames.append(frame_tensor)
            if len(self._initial_frames) < self._init_window:
                return self._last_output

            with self._torch.inference_mode():
                init_video = self._torch.cat(list(self._initial_frames), dim=2)
                noisy_latents = self._encode_with_noise(init_video, config, conditioning)

                denoised_pred = self._pipeline.prepare(
                    text_prompts=[self._compose_prompt(config)],
                    device=self._device,
                    dtype=self._torch.bfloat16,
                    block_mode="input",
                    noise=noisy_latents,
                    current_start=self._current_start,
                    current_end=self._current_end,
                    batch_denoise=False,
                )
                output = self._decode_last_frame(denoised_pred)

            self._prepared = True
            self._processed_chunks = 1
            self._pending_frames.clear()
            self._current_start = self._current_end
            self._current_end += (self._chunk_size // 4) * self._pipeline.frame_seq_length
            self._last_output = output
            return output

        self._pending_frames.append(frame_tensor)
        if len(self._pending_frames) < self._chunk_size:
            return self._last_output

        with self._torch.inference_mode():
            chunk_video = self._torch.cat(self._pending_frames, dim=2)
            noisy_latents = self._encode_with_noise(chunk_video, config, conditioning)
            current_step = self._current_step(config, conditioning)

            if self._current_start // self._pipeline.frame_seq_length >= self._t_refresh:
                self._current_start = self._pipeline.kv_cache_length - self._pipeline.frame_seq_length
                self._current_end = self._current_start + (self._chunk_size // 4) * self._pipeline.frame_seq_length

            denoised_pred = self._pipeline.inference_wo_batch(
                noise=noisy_latents,
                current_start=self._current_start,
                current_end=self._current_end,
                current_step=current_step,
            )
            output = self._decode_last_frame(denoised_pred[[-1]])

        self._pending_frames.clear()
        self._processed_chunks += 1
        self._current_start = self._current_end
        self._current_end += (self._chunk_size // 4) * self._pipeline.frame_seq_length
        self._last_output = output
        return output

    def _load_runtime(self, config: RuntimeConfig) -> None:
        if not self.core_path.exists():
            raise RuntimeLoadError(f"core path not found: {self.core_path}")

        try:
            import torch
            from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
            from omegaconf import OmegaConf
        except Exception as error:  # pragma: no cover - env dependent
            raise RuntimeLoadError(f"failed to import runtime dependencies: {error}") from error

        if not torch.cuda.is_available():
            raise RuntimeLoadError("CUDA is required for StreamDiffusionV2 runtime")

        config_file = self._resolve_config_file(config.model_variant)
        cfg = OmegaConf.load(str(config_file))

        cfg.model_type = "T2V-14B" if config.model_variant == "wan-14b" else "T2V-1.3B"
        cfg.height = self.output_height
        cfg.width = self.output_width
        cfg.num_frame_per_block = 1
        cfg.warp_denoising_step = False
        cfg.denoising_step_list = self._steps_from_count(config.inference_steps)

        device = torch.device("cuda")

        pipeline = CausalStreamInferencePipeline(cfg, device=str(device))
        pipeline.to(device=str(device), dtype=torch.bfloat16)

        checkpoint_folder = self._resolve_checkpoint_folder(config.model_variant)
        self._load_checkpoint_state(torch, pipeline, checkpoint_folder)

        self._torch = torch
        self._pipeline = pipeline
        self._device = device
        self._chunk_size = 4 * self._pipeline.num_frame_per_block
        self._init_window = 1 + self._chunk_size
        self._initial_frames = deque(maxlen=self._init_window)
        self._step_schedule_full = self._steps_from_count(config.inference_steps)
        self._steps_key = config.inference_steps

    def _handle_dynamic_updates(self, config: RuntimeConfig) -> None:
        prompt_key = (config.prompt, config.negative_prompt)

        if self._steps_key != config.inference_steps:
            self._set_denoising_steps(config.inference_steps)
            self._reset_stream_state(config)
            return

        if self._prompt_key != prompt_key:
            self._reset_stream_state(config)

    def _set_denoising_steps(self, inference_steps: int) -> None:
        self._step_schedule_full = self._steps_from_count(inference_steps)
        self._pipeline.denoising_step_list = self._torch.tensor(
            self._step_schedule_full[:-1],
            dtype=self._torch.long,
            device=self._device,
        )
        self._steps_key = inference_steps

    def _reset_stream_state(self, config: RuntimeConfig) -> None:
        self._prepared = False
        self._processed_chunks = 0
        self._initial_frames.clear()
        self._pending_frames.clear()
        self._current_start = 0
        self._current_end = self._pipeline.frame_seq_length * (1 + self._chunk_size // 4)
        self._noise_scale = self._target_noise_scale(config, ConditioningSignals(0.0, 0.0, 0.0))
        self._prompt_key = (config.prompt, config.negative_prompt)

        # Force cache refresh after prompt/steps changes.
        self._pipeline.kv_cache1 = None
        self._pipeline.crossattn_cache = None
        self._pipeline.conditional_dict = None

    def _encode_with_noise(
        self,
        video_tensor: Any,
        config: RuntimeConfig,
        conditioning: ConditioningSignals,
    ) -> Any:
        latents = self._pipeline.vae.stream_encode(video_tensor)
        latents = latents.transpose(2, 1).contiguous().to(dtype=self._torch.bfloat16)

        target_noise = self._target_noise_scale(config, conditioning)
        self._noise_scale = self._noise_scale * 0.85 + target_noise * 0.15

        noise = self._torch.randn_like(latents)
        return noise * self._noise_scale + latents * (1.0 - self._noise_scale)

    def _current_step(self, config: RuntimeConfig, conditioning: ConditioningSignals) -> int:
        positive_steps = [step for step in self._step_schedule_full if step > 0]
        step_min = min(positive_steps)
        step_max = max(positive_steps)

        motion_term = conditioning.motion_score * (1.0 - config.motion_smoothing)
        base = int((self._noise_scale + 0.06 * motion_term) * 1000) - 100
        return max(step_min, min(step_max, base))

    def _frame_to_tensor(self, frame: Image.Image) -> Any:
        resized = frame.resize((self.output_width, self.output_height), Image.Resampling.BILINEAR)
        arr = np.asarray(resized.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2, 0, 1)).copy()
        tensor = self._torch.from_numpy(arr).unsqueeze(0).unsqueeze(2)
        return tensor.to(device=self._device, dtype=self._torch.bfloat16)

    def _decode_last_frame(self, latents: Any) -> Image.Image:
        decoded = self._pipeline.vae.stream_decode_to_pixel(latents)
        decoded = (decoded * 0.5 + 0.5).clamp(0.0, 1.0)
        frames = decoded[0].permute(0, 2, 3, 1).contiguous()
        frame = frames[-1].detach().float().cpu().numpy()
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(frame, mode="RGB")

    def _target_noise_scale(self, config: RuntimeConfig, conditioning: ConditioningSignals) -> float:
        noise = 0.42 + 0.46 * config.prompt_dominance
        identity_factor = config.identity_lock if config.preserve_identity else config.identity_lock * 0.3
        noise -= 0.22 * identity_factor
        noise += 0.12 * conditioning.motion_score
        noise = max(0.24, min(0.92, noise))
        return float(noise)

    def _compose_prompt(self, config: RuntimeConfig) -> str:
        prompt = config.prompt.strip()
        negative = config.negative_prompt.strip()
        if not negative:
            return prompt
        return f"{prompt}. avoid: {negative}"

    def _resolve_config_file(self, model_variant: str) -> Path:
        file_name = "wan_causal_dmd_v2v_14b.yaml" if model_variant == "wan-14b" else "wan_causal_dmd_v2v.yaml"
        config_path = self.core_path / "configs" / file_name
        if not config_path.exists():
            raise RuntimeLoadError(f"missing config file: {config_path}")
        return config_path

    def _resolve_checkpoint_folder(self, model_variant: str) -> Path:
        ckpt_root = self.core_path / "ckpts"
        if not ckpt_root.exists():
            raise RuntimeLoadError(f"missing checkpoint root: {ckpt_root}")

        preferred = (
            ["wan_causal_dmd_v2v_14b"]
            if model_variant == "wan-14b"
            else ["wan_causal_dmd_v2v", "wan_causal_dmd_warp_4step_cfg2"]
        )

        for folder_name in preferred:
            candidate = ckpt_root / folder_name
            if (candidate / "model.pt").is_file():
                return candidate
            if (candidate / "autoregressive_checkpoint" / "model.pt").is_file():
                return candidate / "autoregressive_checkpoint"

        discovered = sorted(ckpt_root.rglob("model.pt"))
        if discovered:
            return discovered[0].parent

        raise RuntimeLoadError(
            f"model.pt not found under {ckpt_root}. "
            "Expected ckpts/wan_causal_dmd_v2v/model.pt (or similar)."
        )

    @staticmethod
    def _load_checkpoint_state(torch: Any, pipeline: Any, checkpoint_folder: Path) -> None:
        ckpt_file = checkpoint_folder / "model.pt"
        if not ckpt_file.exists():
            raise RuntimeLoadError(f"checkpoint file not found: {ckpt_file}")

        checkpoint = torch.load(str(ckpt_file), map_location="cpu")
        if isinstance(checkpoint, dict):
            if "generator" in checkpoint:
                state_dict = checkpoint["generator"]
            elif "generator_ema" in checkpoint:
                state_dict = checkpoint["generator_ema"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        try:
            pipeline.generator.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            pipeline.generator.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _steps_from_count(inference_steps: int) -> list[int]:
        if inference_steps <= 1:
            return [700, 0]
        if inference_steps == 2:
            return [700, 500, 0]
        if inference_steps == 3:
            return [700, 600, 400, 0]
        return [700, 600, 500, 400, 0]

    @staticmethod
    def _snap_size(value: int) -> int:
        snapped = max(256, int(value) // 16 * 16)
        return snapped if snapped % 2 == 0 else snapped - 1
