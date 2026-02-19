# Vast.ai Deployment Notes (H100)

## Recommended first setup

- 1x H100
- CUDA-compatible Ubuntu image
- open ports: `8000` (backend), optional `5173` (frontend dev)

For distributed single-stream inference:

- use a 2x GPU instance
- set session config to `inference_topology=distributed` and `distributed_world_size=2`

## Steps

1. Start instance and SSH in.
2. Clone this repository.
3. Install backend dependencies.
4. Export `STREAMDIFFUSIONV2_PATH=../../core/streamdiffusionv2` (when running from `apps/backend`).
5. Place WAN model weights under `core/streamdiffusionv2/wan_models/`.
6. Run backend service (prefer `tmux` or `systemd`).
7. Serve frontend either locally on your Mac or from the instance.

## Quick bootstrap

```bash
cd project_heru
./scripts/setup_h100_runtime.sh

conda activate heru
export STREAMDIFFUSIONV2_PATH=project_heru/core/streamdiffusionv2
cd project_heru/apps/backend
python -m app.main
```

## Cost guardrails (test budget 20-30 EUR)

- Use H100 only during profiling/tuning sessions
- Keep default test resolution moderate (e.g. 512-768)
- Use `wan-1.3b` + `1-2` steps while iterating

## Runtime profile suggestions

### Performance

- `model_variant=wan-1.3b`
- `inference_steps=1..2`
- `target_fps=30..45`
- `inference_topology=distributed` (when 2 GPUs are available)

### Balanced

- `model_variant=wan-1.3b`
- `inference_steps=2..3`
- `target_fps=24..35`

### Quality

- `model_variant=wan-14b` (if latency budget allows)
- `inference_steps=3..4`
- `target_fps=12..24`
