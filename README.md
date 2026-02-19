# Cam2Inference Console (Unified Fork)

Single project that combines:

1. **Vendored StreamDiffusionV2 core** (no demo/utilities)
2. **App layer** (`apps/backend` + `apps/web`) for realtime cam-to-inference control

## Repository layout

- `core/streamdiffusionv2`: upstream core-only subset (WAN causal runtime + streamv2v core)
- `apps/backend`: FastAPI realtime service (websocket + runtime config)
- `apps/web`: React/Vite DJ console
- `docs`: architecture, roadmap, deployment notes

Core provenance and scope are documented in `core/streamdiffusionv2/CORE_MANIFEST.md`.

## Upstream relationship

- Upstream repo: `https://github.com/chenfengxu714/StreamDiffusionV2`
- This repo vendors only inference/runtime core files, excluding demo app, assets, examples, training/eval utilities.
- Sync script: `scripts/sync_streamdiffusion_core.sh`
- Make target: `make sync-core`

## Quickstart (local)

### 1) Backend

```bash
cd apps/backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export STREAMDIFFUSIONV2_PATH=../../core/streamdiffusionv2
python -m app.main
```

### 2) Frontend

```bash
cd apps/web
npm install
npm run dev
```

Open `http://localhost:5173`.

### 3) Connect and stream

- `Connect`
- `Start Camera`
- `Start Stream`
- Tune prompt + controls live

## Runtime modes

- `mock`: deterministic fallback stylizer (always available for end-to-end testing)
- `core-imported`: vendored StreamDiffusionV2 core is importable
- `core-warmup`: real runtime is active and priming causal cache
- `core-runtime-active`: real frame inference path is active

Runtime status is exposed in websocket payload (`config.current`) and shown in UI.

## H100 Bootstrap (Vast)

For a clean runtime setup on a remote Linux GPU:

```bash
cd <repo-root>
./scripts/setup_h100_runtime.sh
```

Optional flags:

```bash
ENV_NAME=cam2inf PYTHON_VERSION=3.10 DOWNLOAD_WAN_14B=1 ./scripts/setup_h100_runtime.sh
```

HF auth (if needed):

```bash
HF_TOKEN=hf_xxx ./scripts/setup_h100_runtime.sh
```

Guided validation checklist:

- `docs/H100_GUIDED_TEST.md`
- `docs/SSH_TUNNEL_RUNBOOK.md` (recommended flow for local webcam + remote backend)

## Status

- Unified structure (core fork + app layer) is complete.
- Reversible live controls are complete (prompt dominance, pose/depth/seg lock, presets A/B).
- Adapter now includes tensor-level binding to `CausalStreamInferencePipeline` with fallback safety.
