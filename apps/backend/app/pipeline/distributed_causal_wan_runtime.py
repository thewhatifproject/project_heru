from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import socket
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image

from app.config import RuntimeConfig
from app.pipeline.causal_wan_runtime import CausalWanRealtimeRunner, RuntimeLoadError
from app.pipeline.types import ConditioningSignals


@dataclass(frozen=True)
class DistributedRuntimeSignature:
    core_path: str
    model_variant: str
    output_width: int
    output_height: int
    world_size: int


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_steps(inference_steps: int) -> list[int]:
    return CausalWanRealtimeRunner._steps_from_count(inference_steps)


def _resolve_block_intervals(total_blocks: int, world_size: int) -> list[list[int]]:
    if world_size == 2:
        split = max(1, min(total_blocks - 1, total_blocks // 2))
        return [[0, split], [split, total_blocks]]

    base = total_blocks // world_size
    rem = total_blocks % world_size
    start = 0
    intervals: list[list[int]] = []
    for rank in range(world_size):
        size = base + (1 if rank < rem else 0)
        end = start + size if rank < world_size - 1 else total_blocks
        intervals.append([start, end])
        start = end
    return intervals


def _decode_last_frame_to_np(torch: Any, pipeline: Any, latents: Any) -> np.ndarray:
    video = pipeline.vae.stream_decode_to_pixel(latents)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    frame = video[0, -1].permute(1, 2, 0).contiguous()
    arr = frame.float().detach().cpu().numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _frames_to_video_tensor(torch: Any, frames: np.ndarray, device: Any) -> Any:
    # frames: [T, H, W, C], uint8 -> [1, C, T, H, W], bfloat16 in [-1, 1]
    arr = frames.astype(np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (3, 0, 1, 2)).copy()
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.bfloat16)


def _release_latent_buffers(manager: Any, latent_data: Any) -> None:
    # The communication stack uses a reusable buffer pool.
    # Return allocated buffers after each iteration to avoid memory growth.
    try:
        manager.buffer_manager.return_buffer(latent_data.latents, "latent")
        manager.buffer_manager.return_buffer(latent_data.original_latents, "origin")
        if getattr(latent_data, "patched_x_shape", None) is not None:
            manager.buffer_manager.return_buffer(latent_data.patched_x_shape, "misc")
        if getattr(latent_data, "current_start", None) is not None:
            manager.buffer_manager.return_buffer(latent_data.current_start, "misc")
        if getattr(latent_data, "current_end", None) is not None:
            manager.buffer_manager.return_buffer(latent_data.current_end, "misc")
    except Exception:
        # Defensive: never fail command completion due to pool bookkeeping.
        return


def _worker_main(
    rank: int,
    world_size: int,
    control_queue: Any,
    result_queue: Any,
    error_queue: Any,
    core_path: str,
    model_variant: str,
    output_width: int,
    output_height: int,
    initial_steps: int,
    master_addr: str,
    master_port: int,
) -> None:
    manager = None
    dist = None
    try:
        import sys

        sys.path.insert(0, core_path)

        import torch
        import torch.distributed as dist_mod
        from omegaconf import OmegaConf
        from streamv2v.inference_pipe import InferencePipelineManager

        dist = dist_mod

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        config_file = (
            Path(core_path) / "configs" / ("wan_causal_dmd_v2v_14b.yaml" if model_variant == "wan-14b" else "wan_causal_dmd_v2v.yaml")
        )
        cfg = OmegaConf.load(str(config_file))
        cfg.model_type = "T2V-14B" if model_variant == "wan-14b" else "T2V-1.3B"
        cfg.height = output_height
        cfg.width = output_width
        cfg.num_frame_per_block = 1
        cfg.warp_denoising_step = False
        cfg.max_outstanding = 1
        cfg.denoising_step_list = _resolve_steps(initial_steps)

        manager = InferencePipelineManager(cfg, device=device, rank=rank, world_size=world_size)

        ckpt_root = Path(core_path) / "ckpts"
        preferred = ["wan_causal_dmd_v2v_14b"] if model_variant == "wan-14b" else ["wan_causal_dmd_v2v", "wan_causal_dmd_warp_4step_cfg2"]
        checkpoint_folder = None
        for folder_name in preferred:
            folder = ckpt_root / folder_name
            if (folder / "model.pt").is_file():
                checkpoint_folder = folder
                break
            if (folder / "autoregressive_checkpoint" / "model.pt").is_file():
                checkpoint_folder = folder / "autoregressive_checkpoint"
                break
        if checkpoint_folder is None:
            discovered = sorted(ckpt_root.rglob("model.pt"))
            checkpoint_folder = discovered[0].parent if discovered else None
        if checkpoint_folder is None:
            raise RuntimeError(f"No model.pt found under {ckpt_root}")

        manager.load_model(str(checkpoint_folder))

        total_blocks = manager.pipeline.num_transformer_blocks
        block_intervals = _resolve_block_intervals(total_blocks=total_blocks, world_size=world_size)
        block_num = torch.tensor(block_intervals, dtype=torch.int64, device=device)

        if rank == 0:
            block_mode = "input"
        elif rank == world_size - 1:
            block_mode = "output"
        else:
            block_mode = "middle"

        chunk_size = 4 * manager.pipeline.num_frame_per_block
        frame_seq = manager.pipeline.frame_seq_length

        current_start = 0
        current_end = frame_seq * (1 + chunk_size // 4)

        prompt_value = "futuristic cyborg portrait, cinematic lighting"
        steps_value = initial_steps
        num_steps = len(manager.pipeline.denoising_step_list)

        result_queue.put({"type": "ready", "rank": rank})

        while True:
            command = control_queue.get()
            cmd_type = command.get("type")
            cmd_id = int(command.get("cmd_id", -1))

            if cmd_type == "shutdown":
                result_queue.put({"type": "ack", "rank": rank, "cmd_id": cmd_id})
                break

            if cmd_type == "reset":
                manager.pipeline.kv_cache1 = None
                manager.pipeline.crossattn_cache = None
                manager.pipeline.conditional_dict = None
                manager.pipeline.hidden_states = None
                manager.pipeline.block_x = None
                current_start = 0
                current_end = frame_seq * (1 + chunk_size // 4)
                result_queue.put({"type": "ack", "rank": rank, "cmd_id": cmd_id})
                continue

            if cmd_type not in {"prepare", "infer"}:
                result_queue.put({"type": "ack", "rank": rank, "cmd_id": cmd_id})
                continue

            new_steps = int(command.get("inference_steps", steps_value))
            if new_steps != steps_value:
                steps_value = new_steps
                schedule = _resolve_steps(steps_value)
                manager.pipeline.denoising_step_list = torch.tensor(
                    schedule[:-1], dtype=torch.long, device=device
                )
                num_steps = len(manager.pipeline.denoising_step_list)

            prompt_value = str(command.get("prompt", prompt_value))
            noise_scale = float(command.get("noise_scale", 0.7))

            if cmd_type == "prepare":
                if rank == 0:
                    frames = np.asarray(command["frames"], dtype=np.uint8)
                    video_tensor = _frames_to_video_tensor(torch, frames, device)
                    latents = manager.pipeline.vae.stream_encode(video_tensor)
                    latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
                    noise = torch.randn_like(latents)
                    noisy_latents = noise * noise_scale + latents * (1.0 - noise_scale)
                    shape_tensor = torch.tensor(noisy_latents.shape, dtype=torch.int64, device=device)
                else:
                    shape_tensor = torch.zeros(5, dtype=torch.int64, device=device)
                    noisy_latents = None

                dist.broadcast(shape_tensor, src=0)
                if rank != 0:
                    noisy_latents = torch.zeros(tuple(shape_tensor.tolist()), dtype=torch.bfloat16, device=device)
                dist.broadcast(noisy_latents, src=0)

                denoised_pred = manager.prepare_pipeline(
                    text_prompts=[prompt_value],
                    noise=noisy_latents,
                    block_mode=block_mode,
                    current_start=current_start,
                    current_end=current_end,
                    block_num=block_num[rank],
                )

                if rank == world_size - 1:
                    out_np = _decode_last_frame_to_np(torch, manager.pipeline, denoised_pred)
                    result_queue.put({"type": "frame", "cmd_id": cmd_id, "frame": out_np})

                current_start = current_end
                current_end = current_end + (chunk_size // 4) * frame_seq
                result_queue.put({"type": "ack", "rank": rank, "cmd_id": cmd_id})
                continue

            # infer
            schedule_full = _resolve_steps(steps_value)
            positive_steps = [step for step in schedule_full if step > 0]
            step_min = min(positive_steps)
            step_max = max(positive_steps)
            current_step = int(noise_scale * 1000.0) - 100
            current_step = max(step_min, min(step_max, current_step))

            if rank == 0:
                frames = np.asarray(command["frames"], dtype=np.uint8)
                video_tensor = _frames_to_video_tensor(torch, frames, device)
                latents = manager.pipeline.vae.stream_encode(video_tensor)
                latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
                noise = torch.randn_like(latents)
                noisy_latents = noise * noise_scale + latents * (1.0 - noise_scale)

                denoised_pred, patched_x_shape = manager.pipeline.inference(
                    noise=noisy_latents,
                    current_start=current_start,
                    current_end=current_end,
                    current_step=current_step,
                    block_mode="input",
                    block_num=block_num[rank],
                )

                work_objects = manager.data_transfer.send_latent_data_async(
                    chunk_idx=cmd_id,
                    latents=denoised_pred,
                    original_latents=manager.pipeline.hidden_states,
                    patched_x_shape=patched_x_shape,
                    current_start=manager.pipeline.kv_cache_starts,
                    current_end=manager.pipeline.kv_cache_ends,
                    current_step=current_step,
                )
                for work in work_objects:
                    work.wait()

            elif rank == world_size - 1:
                latent_data = manager.data_transfer.receive_latent_data_async(num_steps)
                denoised_pred, _ = manager.pipeline.inference(
                    noise=latent_data.original_latents,
                    current_start=latent_data.current_start,
                    current_end=latent_data.current_end,
                    current_step=latent_data.current_step,
                    block_mode="output",
                    block_num=block_num[rank],
                    patched_x_shape=latent_data.patched_x_shape,
                    block_x=latent_data.latents,
                )

                out_np = _decode_last_frame_to_np(torch, manager.pipeline, denoised_pred[[-1]])
                result_queue.put({"type": "frame", "cmd_id": cmd_id, "frame": out_np})
                _release_latent_buffers(manager, latent_data)

            else:
                latent_data = manager.data_transfer.receive_latent_data_async(num_steps)
                denoised_pred, _ = manager.pipeline.inference(
                    noise=latent_data.original_latents,
                    current_start=latent_data.current_start,
                    current_end=latent_data.current_end,
                    current_step=latent_data.current_step,
                    block_mode="middle",
                    block_num=block_num[rank],
                    patched_x_shape=latent_data.patched_x_shape,
                    block_x=latent_data.latents,
                )
                work_objects = manager.data_transfer.send_latent_data_async(
                    chunk_idx=latent_data.chunk_idx,
                    latents=denoised_pred,
                    original_latents=latent_data.original_latents,
                    patched_x_shape=latent_data.patched_x_shape,
                    current_start=latent_data.current_start,
                    current_end=latent_data.current_end,
                    current_step=latent_data.current_step,
                )
                for work in work_objects:
                    work.wait()
                _release_latent_buffers(manager, latent_data)

            current_start = current_end
            current_end = current_end + (chunk_size // 4) * frame_seq
            result_queue.put({"type": "ack", "rank": rank, "cmd_id": cmd_id})

    except Exception as error:  # pragma: no cover - runtime dependent
        error_queue.put(
            {
                "type": "worker_error",
                "rank": rank,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        try:
            if manager is not None:
                manager.cleanup()
        except Exception:
            pass

        try:
            if dist is not None and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        except Exception:
            pass


class DistributedCausalWanRealtimeRunner:
    """Distributed realtime runner using multiple GPUs for a single stream."""

    def __init__(self, core_path: Path, config: RuntimeConfig) -> None:
        import torch

        if config.distributed_world_size < 2:
            raise RuntimeLoadError("distributed topology requires distributed_world_size >= 2")

        gpu_count = torch.cuda.device_count()
        if gpu_count < config.distributed_world_size:
            raise RuntimeLoadError(
                f"requested {config.distributed_world_size} GPUs but only {gpu_count} are visible"
            )

        self.core_path = core_path
        self.output_width = CausalWanRealtimeRunner._snap_size(config.output_width)
        self.output_height = CausalWanRealtimeRunner._snap_size(config.output_height)
        self.world_size = int(config.distributed_world_size)

        self.signature = DistributedRuntimeSignature(
            core_path=str(core_path),
            model_variant=config.model_variant,
            output_width=self.output_width,
            output_height=self.output_height,
            world_size=self.world_size,
        )

        self._ctx = mp.get_context("spawn")
        self._command_queues = [self._ctx.Queue(maxsize=8) for _ in range(self.world_size)]
        self._result_queue = self._ctx.Queue(maxsize=64)
        self._error_queue = self._ctx.Queue(maxsize=32)
        self._processes: list[Any] = []

        self._cmd_id = 0
        self._chunk_size = 4
        self._init_window = 5
        self._initial_frames: deque[np.ndarray] = deque(maxlen=self._init_window)
        self._pending_frames: list[np.ndarray] = []
        self._prepared = False
        self._processed_chunks = 0
        self._last_output: Image.Image | None = None

        self._prompt_key: tuple[str, str] | None = None
        self._steps_key: int | None = None
        self._noise_scale = 0.68

        self._master_addr = config.distributed_master_addr
        self._master_port = int(config.distributed_master_port or _find_free_port())
        self._timeout_s = float(config.distributed_command_timeout_s)

        self._start_workers(config)
        self._reset_stream_state(config)

    @property
    def mode(self) -> str:
        return "core-distributed-active" if self._prepared else "core-distributed-warmup"

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
            "distributed_world_size": self.world_size,
            "distributed_master_addr": self._master_addr,
            "distributed_master_port": self._master_port,
        }

    def requires_reload(self, config: RuntimeConfig, core_path: Path) -> bool:
        desired = DistributedRuntimeSignature(
            core_path=str(core_path),
            model_variant=config.model_variant,
            output_width=CausalWanRealtimeRunner._snap_size(config.output_width),
            output_height=CausalWanRealtimeRunner._snap_size(config.output_height),
            world_size=int(config.distributed_world_size),
        )
        return desired != self.signature

    def process_frame(
        self,
        frame: Image.Image,
        config: RuntimeConfig,
        conditioning: ConditioningSignals,
    ) -> Image.Image | None:
        self._handle_dynamic_updates(config)

        frame_np = self._frame_to_np(frame)

        if not self._prepared:
            self._initial_frames.append(frame_np)
            if len(self._initial_frames) < self._init_window:
                return self._last_output

            warmup_frames = np.stack(list(self._initial_frames), axis=0)
            self._noise_scale = self._target_noise_scale(config, conditioning)

            out_np = self._dispatch_command(
                command_type="prepare",
                frames=warmup_frames,
                config=config,
            )

            self._prepared = True
            self._processed_chunks = 1
            self._pending_frames.clear()

            if out_np is not None:
                self._last_output = Image.fromarray(out_np, mode="RGB")
            return self._last_output

        self._pending_frames.append(frame_np)
        if len(self._pending_frames) < self._chunk_size:
            return self._last_output

        chunk_frames = np.stack(self._pending_frames, axis=0)
        self._noise_scale = self._noise_scale * 0.85 + self._target_noise_scale(config, conditioning) * 0.15

        out_np = self._dispatch_command(
            command_type="infer",
            frames=chunk_frames,
            config=config,
        )

        self._pending_frames.clear()
        self._processed_chunks += 1

        if out_np is not None:
            self._last_output = Image.fromarray(out_np, mode="RGB")
        return self._last_output

    def close(self) -> None:
        self._cmd_id += 1
        cmd = {"type": "shutdown", "cmd_id": self._cmd_id}
        for queue in self._command_queues:
            queue.put(cmd)

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if all(not process.is_alive() for process in self._processes):
                break
            for process in self._processes:
                process.join(timeout=0.1)

        for process in self._processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)

    def _start_workers(self, config: RuntimeConfig) -> None:
        if not self.core_path.exists():
            raise RuntimeLoadError(f"core path not found: {self.core_path}")

        for rank in range(self.world_size):
            process = self._ctx.Process(
                target=_worker_main,
                args=(
                    rank,
                    self.world_size,
                    self._command_queues[rank],
                    self._result_queue,
                    self._error_queue,
                    str(self.core_path),
                    config.model_variant,
                    self.output_width,
                    self.output_height,
                    int(config.inference_steps),
                    self._master_addr,
                    self._master_port,
                ),
                daemon=True,
            )
            process.start()
            self._processes.append(process)

        ready_ranks: set[int] = set()
        deadline = time.monotonic() + 240.0

        while len(ready_ranks) < self.world_size:
            self._raise_worker_error_if_any()

            if time.monotonic() > deadline:
                raise RuntimeLoadError("timeout while starting distributed workers")

            try:
                message = self._result_queue.get(timeout=1.0)
            except Empty:
                continue

            if message.get("type") == "ready":
                ready_ranks.add(int(message["rank"]))

        if len(ready_ranks) != self.world_size:
            raise RuntimeLoadError("distributed workers did not initialize correctly")

    def _dispatch_command(
        self,
        command_type: str,
        frames: np.ndarray,
        config: RuntimeConfig,
    ) -> np.ndarray | None:
        self._cmd_id += 1
        cmd_id = self._cmd_id

        payload_common = {
            "type": command_type,
            "cmd_id": cmd_id,
            "prompt": self._compose_prompt(config),
            "inference_steps": int(config.inference_steps),
            "noise_scale": float(self._noise_scale),
        }

        for rank, queue in enumerate(self._command_queues):
            payload = dict(payload_common)
            if rank == 0:
                payload["frames"] = frames
            queue.put(payload)

        ack_ranks: set[int] = set()
        out_np: np.ndarray | None = None
        deadline = time.monotonic() + self._timeout_s

        while len(ack_ranks) < self.world_size:
            self._raise_worker_error_if_any()

            if time.monotonic() > deadline:
                raise RuntimeLoadError(
                    f"timeout waiting distributed command completion: {command_type}"
                )

            try:
                message = self._result_queue.get(timeout=1.0)
            except Empty:
                continue

            if message.get("cmd_id") != cmd_id:
                continue

            msg_type = message.get("type")
            if msg_type == "ack":
                ack_ranks.add(int(message["rank"]))
            elif msg_type == "frame":
                out_np = np.asarray(message["frame"], dtype=np.uint8)

        return out_np

    def _raise_worker_error_if_any(self) -> None:
        try:
            message = self._error_queue.get_nowait()
        except Empty:
            return

        rank = message.get("rank")
        error = message.get("error")
        trace = message.get("traceback", "")
        raise RuntimeLoadError(f"distributed worker {rank} failed: {error}\n{trace}")

    def _handle_dynamic_updates(self, config: RuntimeConfig) -> None:
        prompt_key = (config.prompt, config.negative_prompt)

        if self._steps_key != config.inference_steps or self._prompt_key != prompt_key:
            self._send_reset()
            self._reset_stream_state(config)

    def _send_reset(self) -> None:
        self._cmd_id += 1
        cmd_id = self._cmd_id
        payload = {"type": "reset", "cmd_id": cmd_id}

        for queue in self._command_queues:
            queue.put(payload)

        ack_ranks: set[int] = set()
        deadline = time.monotonic() + self._timeout_s
        while len(ack_ranks) < self.world_size:
            self._raise_worker_error_if_any()
            if time.monotonic() > deadline:
                raise RuntimeLoadError("timeout waiting distributed reset")
            try:
                message = self._result_queue.get(timeout=1.0)
            except Empty:
                continue
            if message.get("type") == "ack" and message.get("cmd_id") == cmd_id:
                ack_ranks.add(int(message["rank"]))

    def _reset_stream_state(self, config: RuntimeConfig) -> None:
        self._prepared = False
        self._processed_chunks = 0
        self._initial_frames.clear()
        self._pending_frames.clear()
        self._noise_scale = self._target_noise_scale(config, ConditioningSignals(0.0, 0.0, 0.0))
        self._prompt_key = (config.prompt, config.negative_prompt)
        self._steps_key = int(config.inference_steps)

    def _frame_to_np(self, frame: Image.Image) -> np.ndarray:
        resized = frame.resize((self.output_width, self.output_height), Image.Resampling.BILINEAR)
        return np.asarray(resized.convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _target_noise_scale(config: RuntimeConfig, conditioning: ConditioningSignals) -> float:
        noise = 0.42 + 0.46 * config.prompt_dominance
        identity_factor = config.identity_lock if config.preserve_identity else config.identity_lock * 0.3
        noise -= 0.22 * identity_factor
        noise += 0.12 * conditioning.motion_score
        return float(max(0.24, min(0.92, noise)))

    @staticmethod
    def _compose_prompt(config: RuntimeConfig) -> str:
        prompt = config.prompt.strip()
        negative = config.negative_prompt.strip()
        if not negative:
            return prompt
        return f"{prompt}. avoid: {negative}"

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
