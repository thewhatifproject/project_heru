# Architecture

## High-level flow

1. Browser captures webcam frames (JPEG)
2. Frames are sent over websocket to backend session
3. Backend applies conditioning + diffusion adapter
4. Processed frames are streamed back to browser preview
5. Runtime controls are patched in real-time via `config.update`

## Core design goals

- Prompt dominates appearance when needed (`prompt_dominance` high)
- Pose/depth/shape/motion stay stable via lock controls
- Every behavior is reversible/tunable at runtime

## Session model

A session owns:

- `RuntimeConfig`
- `RealtimePipeline`
- metrics (`frames_in`, `frames_out`, `avg_latency_ms`)
- revision counter for deterministic config tracking

## Backend modules

- `app/config.py`: runtime config schema + deep merge
- `app/services/session.py`: session lifecycle and frame processing
- `app/pipeline/conditioning.py`: lightweight conditioning proxy
- `app/pipeline/streamdiffusion_adapter.py`: StreamDiffusionV2 adapter with runtime discovery
- `app/main.py`: REST + websocket API

## Vendored upstream core

- Path: `core/streamdiffusionv2`
- Scope: WAN causal inference core + streamv2v communication/inference modules
- Excluded: upstream demo app, training/eval helpers, non-core utilities
- Provenance: `core/streamdiffusionv2/CORE_MANIFEST.md`

## Frontend modules

- `src/App.tsx`: transport + webcam + control surface
- `src/styles.css`: responsive visual system and staged animations

## Websocket protocol

Client -> Server

- `config.update`
- `preset.apply`
- `frame`

Server -> Client

- `config.current`
- `frame.processed`
- `stats`
- `error`

`config.current` includes:

- `runtime_status.mode` (`mock`, `core-imported`, `core-warmup`, `core-runtime-active`)
- `runtime_status.runtime_path`
- `runtime_status.error` (if core import fails)
