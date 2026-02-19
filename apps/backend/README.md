# Backend (FastAPI)

Realtime websocket backend for camera-to-inference flows.

## Features

- Session-scoped runtime config (`prompt`, `dominance`, `pose/depth/seg locks`, outputs)
- Websocket protocol for low-latency frame processing
- Modular pipeline adapter with vendored StreamDiffusionV2 core discovery
- Preset switching (`A/B`) for DJ-style scene changes

## Run

```bash
cd apps/backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export STREAMDIFFUSIONV2_PATH=../../core/streamdiffusionv2
python -m app.main
```

Server default: `http://localhost:8000`

## Endpoints

- `GET /api/health`
- `GET /api/session/{session_id}/config`
- `PUT /api/session/{session_id}/config`
- `WS /ws/session/{session_id}`

## Websocket messages

### Client -> Server

```json
{"type":"config.update","payload":{"prompt":"clean-face cyborg","prompt_dominance":0.9}}
```

```json
{"type":"preset.apply","payload":{"name":"a"}}
```

```json
{"type":"frame","payload":{"timestamp_ms":1730000000000,"image_b64":"..."}}
```

### Server -> Client

- `config.current`
- `frame.processed`
- `stats`
- `error`

`config.current.payload.runtime_status` exposes:

- `mode`: `mock`, `core-imported`, `core-warmup`, `core-runtime-active`
- `runtime_path`
- `error`
