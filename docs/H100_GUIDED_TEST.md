# H100 Guided Setup + Smoke Test

This guide validates the unified repository end-to-end on a remote H100.

## 0) Connect to your Vast instance

```bash
ssh -p <SSH_PORT> <SSH_USER>@<SSH_HOST>
```

## 1) Clone repo and run bootstrap

```bash
git clone <your-fork-url> project-heru
cd project-heru
./scripts/setup_h100_runtime.sh
```

Notes:

- Optional: `DOWNLOAD_WAN_14B=1` if you want 14B assets now.

## 2) Activate env and export core path

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate heru
export STREAMDIFFUSIONV2_PATH=$(pwd)/core/streamdiffusionv2
```

## 3) Run adapter smoke test (real runtime path)

```bash
cd apps/backend
python scripts/smoke_realtime.py --frames 10 --model wan-1.3b --steps 2 --require-core --save /tmp/heru-smoke.jpg
```

Expected:

- Initial mode may be `core-imported`.
- Then mode should become `core-warmup` and finally `core-runtime-active`.
- Output file exists at `/tmp/heru-smoke.jpg`.

If it stays `mock`, inspect `status.error` shown per frame.

### Optional: distributed smoke (2 GPUs)

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

Expected runtime mode progression:

- `core-distributed-warmup`
- `core-distributed-active`

## 4) Launch backend API

```bash
python -m app.main
```

In another shell:

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/session/main/config
```

Expected in config response:

- `runtime_status.mode` in `core-imported|core-warmup|core-runtime-active`
- `runtime_status.error` should be `null`

## 5) Optional frontend on the same host

```bash
cd ../../apps/web
npm install
VITE_WS_BASE=ws://127.0.0.1:8000/ws/session npm run dev -- --host 0.0.0.0 --port 5173
```

Open your tunnel/public URL and start streaming.

## 6) Recommended: local webcam + SSH tunnel (no public exposure)

For webcam workflows, use local browser UI with a tunnel to remote backend.

See the dedicated runbook:

- `docs/SSH_TUNNEL_RUNBOOK.md`

## Troubleshooting quick map

- `failed to import StreamDiffusionV2 core`: missing Python deps or wrong `STREAMDIFFUSIONV2_PATH`.
- `CUDA is required`: not running on GPU instance or driver issue.
- `model.pt not found`: checkpoint not downloaded under `core/streamdiffusionv2/ckpts/...`.
- `flash_attn` build/install error: rerun setup with `INSTALL_FLASH_ATTN=0` for temporary fallback testing.
