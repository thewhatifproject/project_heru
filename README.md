# Project Heru (Unified Fork)

This `README.md` is the single source of truth for project documentation.  
Legacy files under `docs/` now point here to avoid duplicated runbooks.

## Table of contents

- [Project scope](#project-scope)
- [Repository layout](#repository-layout)
- [Upstream relationship](#upstream-relationship)
- [Architecture](#architecture)
- [WebSocket protocol](#websocket-protocol)
- [Runtime modes](#runtime-modes)
- [Prerequisites](#prerequisites)
- [Local quickstart](#local-quickstart)
- [Session config and topology](#session-config-and-topology)
- [Vast deployment](#vast-deployment)
- [Guided H100 smoke test](#guided-h100-smoke-test)
- [Benchmark single vs distributed](#benchmark-single-vs-distributed)
- [SSH tunnel workflow](#ssh-tunnel-workflow)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Current status](#current-status)

## Project scope

Single project that combines:

1. Vendored StreamDiffusionV2 core (runtime/inference subset only)
2. App layer (`apps/backend` + `apps/web`) for realtime cam-to-inference control

## Repository layout

- `core/streamdiffusionv2`: upstream core subset (WAN causal runtime + streamv2v core)
- `apps/backend`: FastAPI realtime service (REST + WebSocket + runtime config)
- `apps/web`: React/Vite control UI
- `scripts`: bootstrap/sync helpers (`setup_h100_runtime.sh`, `sync_streamdiffusion_core.sh`)
- `docs`: compatibility entrypoints that redirect to this README

Core provenance and scope: `core/streamdiffusionv2/CORE_MANIFEST.md`.

## Upstream relationship

- Upstream repo: `https://github.com/chenfengxu714/StreamDiffusionV2`
- This repo vendors only core runtime/inference files.
- Excluded from vendoring: demo app, assets, examples, and training/eval utilities.
- Sync script: `scripts/sync_streamdiffusion_core.sh`
- Make target: `make sync-core`

## Architecture

High-level flow:

1. Browser captures webcam frames (JPEG)
2. Frames are sent over WebSocket to backend session
3. Backend applies conditioning + diffusion adapter
4. Processed frames are streamed back to browser preview
5. Runtime controls are patched in real-time via `config.update`

Core design goals:

- Prompt can dominate appearance when needed (`prompt_dominance`)
- Pose/depth/shape/motion remain stable via lock controls
- Every behavior is reversible/tunable at runtime

Session model:

- `RuntimeConfig`
- `RealtimePipeline`
- Metrics: `frames_in`, `frames_out`, `avg_latency_ms`
- Revision counter for deterministic config tracking

Backend modules:

- `apps/backend/app/main.py`: REST + WebSocket API
- `apps/backend/app/services/session.py`: session lifecycle and frame processing
- `apps/backend/app/config.py`: runtime config schema + deep merge
- `apps/backend/app/pipeline/conditioning.py`: conditioning signals
- `apps/backend/app/pipeline/streamdiffusion_adapter.py`: runtime adapter + fallback behavior

Frontend modules:

- `apps/web/src/App.tsx`: transport, webcam, controls
- `apps/web/src/styles.css`: responsive UI styling

## WebSocket protocol

Client -> Server message types:

- `config.update`
- `preset.apply`
- `frame`

Server -> Client message types:

- `config.current`
- `frame.processed`
- `stats`
- `error`

Example `config.update`:

```json
{
  "type": "config.update",
  "payload": {
    "prompt": "clean-face cyborg",
    "prompt_dominance": 0.9
  }
}
```

## Runtime modes

- `mock`: deterministic fallback stylizer (always available)
- `core-imported`: vendored StreamDiffusionV2 core is importable
- `core-warmup`: single-GPU runtime is priming
- `core-runtime-active`: single-GPU realtime inference is active
- `core-distributed-warmup`: distributed runtime is priming
- `core-distributed-active`: distributed realtime inference is active

Runtime status is exposed in `config.current.payload.runtime_status`.

## Prerequisites

Local development:

- Python 3.10+
- Node.js 20+

Remote GPU setup (recommended for real runtime):

- NVIDIA GPU host with `nvidia-smi`
- Linux environment suitable for CUDA
- SSH access to host

## Local quickstart

### Backend

```bash
cd apps/backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export STREAMDIFFUSIONV2_PATH=../../core/streamdiffusionv2
python -m app.main
```

Backend default: `http://127.0.0.1:8000`.

### Frontend

```bash
cd apps/web
npm install
npm run dev
```

Open: `http://127.0.0.1:5173`.

### Connect and stream

1. `Connect`
2. `Start Camera`
3. `Start Stream`
4. Tune prompt/controls live

### Optional Makefile shortcuts

```bash
make backend
make web
make smoke-realtime
```

### Optional Docker Compose

```bash
docker compose up --build
```

## Session config and topology

Main endpoints:

- `GET /api/health`
- `GET /api/session/{session_id}/config`
- `PUT /api/session/{session_id}/config`
- `WS /ws/session/{session_id}`

Switch to distributed 2-GPU topology:

```bash
curl -s -X PUT http://127.0.0.1:8000/api/session/main/config \
  -H "Content-Type: application/json" \
  -d '{"inference_topology":"distributed","distributed_world_size":2}'
```

Switch back to single GPU:

```bash
curl -s -X PUT http://127.0.0.1:8000/api/session/main/config \
  -H "Content-Type: application/json" \
  -d '{"inference_topology":"single"}'
```

Useful runtime fields in config:

- `model_variant`: `wan-1.3b` | `wan-14b`
- `inference_steps`: `1..8`
- `target_fps`: `5..60`
- `output_width` / `output_height`
- `streamdiffusionv2_path` (or env `STREAMDIFFUSIONV2_PATH`)

## Vast deployment

Recommended first setup:

- 1x H100 on CUDA-compatible Ubuntu image
- Open port `8000` (backend)
- Optional port `5173` (frontend dev)

For distributed single-stream inference:

- use 2x GPU instance
- set `inference_topology=distributed`
- set `distributed_world_size=2`

Quick bootstrap on remote host:

```bash
cd <repo-root>
./scripts/setup_h100_runtime.sh
```

Common options:

```bash
ENV_NAME=heru PYTHON_VERSION=3.10 DOWNLOAD_WAN_14B=1 ./scripts/setup_h100_runtime.sh
```

Hugging Face auth (if needed):

```bash
HF_TOKEN=hf_xxx ./scripts/setup_h100_runtime.sh
```

Start backend after bootstrap:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cam2inf
export STREAMDIFFUSIONV2_PATH=<REPO_ROOT>/core/streamdiffusionv2
cd <REPO_ROOT>/apps/backend
python -m app.main
```

Cost guardrails (20-30 EUR test budget):

- Keep H100 active only for profiling/tuning windows
- Use moderate resolution (512-768) during iteration
- Use `wan-1.3b` with `1-2` steps while tuning

Runtime profile suggestions:

- Performance: `wan-1.3b`, `1-2` steps, `target_fps=30..45`
- Balanced: `wan-1.3b`, `2-3` steps, `target_fps=24..35`
- Quality: `wan-14b`, `3-4` steps, `target_fps=12..24`

## Guided H100 smoke test

Validate end-to-end runtime directly on remote H100.

### 1) Connect and bootstrap

```bash
ssh -p <SSH_PORT> <SSH_USER>@<SSH_HOST>
git clone <your-fork-url> project-heru
cd project-heru
./scripts/setup_h100_runtime.sh
```

### 2) Activate env and export core path

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cam2inf
export STREAMDIFFUSIONV2_PATH=$(pwd)/core/streamdiffusionv2
```

### 3) Run adapter smoke test (single GPU)

```bash
cd apps/backend
python scripts/smoke_realtime.py \
  --frames 10 \
  --model wan-1.3b \
  --steps 2 \
  --require-core \
  --save /tmp/heru-smoke.jpg
```

Expected mode progression:

- `core-imported` (may appear first)
- `core-warmup`
- `core-runtime-active`

### 4) Optional distributed smoke (2 GPUs)

```bash
python scripts/smoke_realtime.py \
  --frames 12 \
  --model wan-1.3b \
  --topology distributed \
  --world-size 2 \
  --steps 2 \
  --require-core \
  --save /tmp/heru-smoke-distributed.jpg
```

Expected distributed progression:

- `core-distributed-warmup`
- `core-distributed-active`

### 5) API sanity checks

```bash
python -m app.main
```

From another shell:

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/session/main/config
```

## Benchmark single vs distributed

Use the same model/resolution/steps and compare logs.

From remote host:

```bash
cd <REPO_ROOT>/apps/backend
export STREAMDIFFUSIONV2_PATH=<REPO_ROOT>/core/streamdiffusionv2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Single GPU:

```bash
python scripts/smoke_realtime.py \
  --frames 40 \
  --model wan-1.3b \
  --topology single \
  --steps 1 \
  --width 512 \
  --height 512 \
  --require-core \
  --save /tmp/heru-bench-single.jpg | tee /tmp/heru-bench-single.log
```

Distributed (2 GPUs):

```bash
python scripts/smoke_realtime.py \
  --frames 40 \
  --model wan-1.3b \
  --topology distributed \
  --world-size 2 \
  --steps 1 \
  --width 512 \
  --height 512 \
  --require-core \
  --save /tmp/heru-bench-distributed.jpg | tee /tmp/heru-bench-distributed.log
```

Extract average latency:

```bash
grep -E "average latency per frame" /tmp/heru-bench-single.log /tmp/heru-bench-distributed.log
```

## SSH tunnel workflow

Recommended for local webcam + remote GPU backend without exposing public ports.

### Terminal A (remote): backend

```bash
ssh -p <SSH_PORT> <SSH_USER>@<SSH_HOST>
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cam2inf
export STREAMDIFFUSIONV2_PATH=<REMOTE_REPO_PATH>/core/streamdiffusionv2
cd <REMOTE_REPO_PATH>/apps/backend
python -m app.main
```

### Terminal B (local): tunnel

```bash
ssh -p <SSH_PORT> <SSH_USER>@<SSH_HOST> -N -L 8000:127.0.0.1:8000
```

If local `8000` is busy, use e.g. `-L 18000:127.0.0.1:8000`.

### Terminal C (local): frontend

```bash
cd <LOCAL_REPO_PATH>/apps/web
npm install
VITE_WS_BASE=ws://127.0.0.1:8000/ws/session npm run dev
```

Open: `http://127.0.0.1:5173`.

Then in UI:

1. `Connect`
2. `Start Camera`
3. `Start Stream`

Verify runtime from local machine:

```bash
curl -s http://127.0.0.1:8000/api/session/main/config
```

## Troubleshooting

- `runtime_status.mode` stays `mock`:
  - Check `runtime_status.error`
  - Verify `STREAMDIFFUSIONV2_PATH`
  - Verify model/checkpoint files under `core/streamdiffusionv2/wan_models` and `core/streamdiffusionv2/ckpts`
- `failed to import StreamDiffusionV2 core`:
  - Missing Python deps or invalid core path
- `CUDA is required`:
  - Host is not GPU-ready or driver stack is broken
- `model.pt not found`:
  - Missing checkpoint download
- `flash_attn` install issues:
  - Retry setup with `INSTALL_FLASH_ATTN=0` for temporary fallback tests
- Local API unreachable on `127.0.0.1:8000` with tunnel flow:
  - Tunnel not running or SSH host/port mismatch
- `vite: command not found`:
  - Run `npm install` in `apps/web`
- npm `EACCES` on cache:
  - `sudo chown -R "$(id -u):$(id -g)" ~/.npm`

## Roadmap

### Week 1

- Finalize runtime config schema and control taxonomy
- Integrate StreamDiffusionV2 runtime in adapter (Wan 1.3B default)
- Baseline latency/fps instrumentation

### Week 2

- Add real pose pipeline (MediaPipe)
- Add depth estimator and segmentation branch
- Introduce motion-aware blending and temporal smoothing presets

### Week 3

- Optimize transport path (binary WebSocket + optional WebRTC)
- Add adaptive quality controller (fps-first mode)
- Add queue backpressure and frame skipping policy

### Week 4

- Implement virtual camera sink for macOS via OBS bridge
- Implement RTMP output sink and connection health checks
- Add scene/preset bank management (A/B + quick recall)

### Week 5-6 (hardening)

- Vast deployment automation + observability
- Fallback profiles (A100 vs H100)
- Stability tests: movement stress, close-up transitions, low-light webcam

Acceptance targets:

- 30+ fps median on H100 in performance mode
- Prompt-dominant transformation with preserved movement semantics
- Stable UI controls with reversible runtime behavior

## Current status

- Unified structure (core fork + app layer) is complete.
- Reversible live controls are complete (prompt dominance, pose/depth/seg lock, presets A/B).
- Adapter includes tensor-level binding to `CausalStreamInferencePipeline` with fallback safety.
