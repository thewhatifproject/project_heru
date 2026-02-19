#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import RuntimeConfig
from app.pipeline.streamdiffusion_adapter import StreamDiffusionV2Adapter
from app.pipeline.types import ConditioningSignals


def build_frame(width: int, height: int, tick: int) -> Image.Image:
    """Generate a deterministic synthetic frame with visible motion."""
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    shift = (tick % 20) / 20.0
    r = np.mod(xx + shift, 1.0)
    g = np.mod(yy + shift * 0.7, 1.0)
    b = np.mod((xx * 0.5 + yy * 0.5) + shift * 0.4, 1.0)
    arr = np.stack([r, g, b], axis=-1)

    arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_uint8, mode="RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for realtime StreamDiffusionV2 adapter")
    parser.add_argument("--frames", type=int, default=10, help="Number of synthetic frames to process")
    parser.add_argument("--width", type=int, default=768, help="Output width")
    parser.add_argument("--height", type=int, default=768, help="Output height")
    parser.add_argument("--model", choices=["wan-1.3b", "wan-14b"], default="wan-1.3b")
    parser.add_argument("--steps", type=int, default=2, help="Inference steps")
    parser.add_argument("--save", type=str, default="", help="Optional path for last output image")
    parser.add_argument(
        "--require-core",
        action="store_true",
        help="Fail if runtime never enters core-warmup/core-runtime-active",
    )
    args = parser.parse_args()

    adapter = StreamDiffusionV2Adapter()
    status = adapter.runtime_status
    print(f"initial runtime: {status}")

    config = RuntimeConfig(
        model_variant=args.model,
        inference_steps=args.steps,
        output_width=args.width,
        output_height=args.height,
        prompt="cinematic cyborg portrait, clean face, metallic details",
        negative_prompt="beard, facial hair, blur, watermark",
        prompt_dominance=0.86,
        preserve_identity=False,
    )

    last_output: Image.Image | None = None
    saw_core_mode = False
    total_ms = 0.0

    for index in range(args.frames):
        frame = build_frame(args.width, args.height, index)
        conditioning = ConditioningSignals(
            motion_score=min(1.0, index / max(1, args.frames - 1)),
            edge_density=0.35,
            luma_mean=0.45,
        )

        start = time.perf_counter()
        out = adapter.generate(frame, config, conditioning)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        total_ms += elapsed_ms

        runtime_mode = str(adapter.runtime_status.get("mode"))
        if runtime_mode in {"core-warmup", "core-runtime-active"}:
            saw_core_mode = True

        print(
            f"frame={index + 1}/{args.frames} mode={runtime_mode} "
            f"latency_ms={elapsed_ms:.2f} status={adapter.runtime_status}"
        )

        last_output = out

    avg_ms = total_ms / max(1, args.frames)
    print(f"average latency per frame: {avg_ms:.2f} ms")

    if args.save and last_output is not None:
        output_path = Path(args.save).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        last_output.save(output_path, format="JPEG", quality=90)
        print(f"saved output: {output_path}")

    if args.require_core and not saw_core_mode:
        print("ERROR: runtime never reached core-warmup/core-runtime-active", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
