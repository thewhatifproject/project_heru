# Roadmap (4-6 weeks)

## Week 1

- Finalize runtime config schema and control taxonomy
- Integrate StreamDiffusionV2 runtime into adapter (Wan 1.3B default)
- Baseline latency/fps instrumentation

## Week 2

- Add real pose pipeline (MediaPipe)
- Add depth estimator and segmentation branch
- Introduce motion-aware blending and temporal smoothing presets

## Week 3

- Optimize transport path (binary websocket + optional WebRTC)
- Add adaptive quality controller (fps-first mode)
- Add queue backpressure and frame skipping policy

## Week 4

- Implement Virtual Camera sink for macOS flow via OBS bridge
- Implement RTMP output sink and connection health checks
- Add scene/preset bank management (A/B + quick recall)

## Week 5-6 (hardening)

- Vast deployment automation + observability
- Fallback profiles (A100 vs H100)
- Stability tests: movement stress, close-up transitions, low-light webcam

## Acceptance targets

- 30+ fps median on H100 in performance mode
- prompt-dominant transformation preserving movement semantics
- stable UI controls with reversible runtime behavior
