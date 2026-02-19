# SSH Tunnel Runbook (Local Webcam + Remote GPU)

This is the recommended workflow when your inference backend runs on a remote VM and your webcam is local.

## Prerequisites

- Remote backend machine reachable via SSH.
- Backend environment already bootstrapped (`setup_h100_runtime.sh`).
- Local machine with Node.js >= 20 for the web UI.

## Terminal A (remote shell): start backend

```bash
ssh -p <SSH_PORT> <SSH_USER>@<SSH_HOST>
source ~/miniconda3/etc/profile.d/conda.sh
conda activate heru
export STREAMDIFFUSIONV2_PATH=<REMOTE_REPO_PATH>/core/streamdiffusionv2
cd <REMOTE_REPO_PATH>/apps/backend
python -m app.main
```

Keep this terminal open.

## Terminal B (local shell): open SSH tunnel

```bash
ssh -p <SSH_PORT> <SSH_USER>@<SSH_HOST> -N -L 8000:127.0.0.1:8000
```

Keep this terminal open.

Notes:

- Password prompt is expected unless key-based auth is configured.
- If local port `8000` is busy, use another local port, e.g. `18000:127.0.0.1:8000`.

## Terminal C (local shell): run UI locally

```bash
cd <LOCAL_REPO_PATH>/apps/web
npm install
VITE_WS_BASE=ws://127.0.0.1:8000/ws/session npm run dev
```

Open:

- `http://127.0.0.1:5173`

Then in UI:

- `Connect`
- `Start Camera`
- `Start Stream`

## Runtime verification (local)

```bash
curl -s http://127.0.0.1:8000/api/session/main/config
```

Expected mode progression:

- `core-imported` (idle)
- `core-warmup` (initial buffering)
- `core-runtime-active` (live inference)

To enable 2-GPU distributed inference for the same stream:

```bash
curl -s -X PUT http://127.0.0.1:8000/api/session/main/config \
  -H 'Content-Type: application/json' \
  -d '{"inference_topology":"distributed","distributed_world_size":2}'
```

## Troubleshooting

- `vite: command not found`:
  - Run `npm install` inside `apps/web`.
- npm `EACCES` on cache:
  - `sudo chown -R "$(id -u):$(id -g)" ~/.npm`
- API unreachable on local `127.0.0.1:8000`:
  - Tunnel not running or wrong SSH host/port.
- Runtime stuck in `mock`:
  - Check `runtime_status.error` and verify remote `STREAMDIFFUSIONV2_PATH`.
