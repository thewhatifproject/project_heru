# StreamDiffusionV2 Core Manifest

This folder vendors a **core-only subset** of upstream StreamDiffusionV2.

## Upstream

- Repository: `https://github.com/chenfengxu714/StreamDiffusionV2`
- Pinned commit: `bcfd9e7b182946b0030db28274a3e5121a5ac50f`
- Upstream commit date: `2026-02-18 10:33:00 +0800`
- License: Apache-2.0 (`LICENSE` copied as-is)

## Included (core)

- `causvid/models/model_interface.py`
- `causvid/models/wan/**` (excluding non-core helpers listed below)
- `causvid/scheduler.py`
- `causvid/data.py` (minimal `TextDataset` shim for streamv2v CLI)
- `streamv2v/inference.py`
- `streamv2v/inference_wo_batch.py`
- `streamv2v/communication/**` (excluding test file)
- `configs/wan_causal_dmd_v2v*.yaml`

## Excluded (non-core)

- Demo app/UI: `demo/**`
- Utilities/examples/assets unrelated to runtime core: `assets/**`, `examples/**`
- Training/evaluation scripts and datasets in `causvid/**`
- Non-core WAN scripts:
  - `causvid/models/wan/bidirectional_inference.py`
  - `causvid/models/wan/causal_inference.py`
  - `causvid/models/wan/generate_ode_pairs.py`

## Notes

- `causvid/models/__init__.py` is intentionally simplified to WAN/causal-WAN only.
- `streamv2v` CLI imports for `TextDataset` were made lazy to avoid requiring dataset modules for runtime import.
