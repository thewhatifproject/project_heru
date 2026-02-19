import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Mode = "performance" | "balanced" | "quality";
type ModelVariant = "wan-1.3b" | "wan-14b";
type InferenceTopology = "single" | "distributed";

type OutputConfig = {
  preview: boolean;
  virtual_camera: boolean;
  rtmp_enabled: boolean;
  rtmp_url: string | null;
};

type RuntimeConfig = {
  prompt: string;
  negative_prompt: string;
  mode: Mode;
  model_variant: ModelVariant;
  inference_topology: InferenceTopology;
  distributed_world_size: number;
  distributed_master_addr: string;
  distributed_master_port: number;
  distributed_command_timeout_s: number;
  prompt_dominance: number;
  guidance_scale: number;
  inference_steps: number;
  seed: number | null;
  preserve_identity: boolean;
  identity_lock: number;
  pose_lock: number;
  depth_lock: number;
  segmentation_lock: number;
  motion_smoothing: number;
  target_fps: number;
  output_width: number;
  output_height: number;
  jpeg_quality: number;
  outputs: OutputConfig;
  streamdiffusionv2_path: string | null;
};

type StatsPayload = {
  metrics: {
    frames_in: number;
    frames_out: number;
    avg_latency_ms: number;
  };
  revision: number;
};

type Conditioning = {
  motion_score: number;
  edge_density: number;
  luma_mean: number;
};

type RuntimeStatus = {
  mode: string;
  runtime_path: string | null;
  inference_topology?: string | null;
  distributed_world_size?: number;
  error: string | null;
};

type ConnectionState = "disconnected" | "connecting" | "connected";

const WS_BASE =
  (import.meta.env.VITE_WS_BASE as string | undefined) ?? "ws://localhost:8000/ws/session";

const DEFAULT_CONFIG: RuntimeConfig = {
  prompt: "ultra-detailed cyberpunk cyborg, metallic skin, clean face",
  negative_prompt: "beard, moustache, blur, text, watermark, low quality",
  mode: "balanced",
  model_variant: "wan-1.3b",
  inference_topology: "single",
  distributed_world_size: 2,
  distributed_master_addr: "127.0.0.1",
  distributed_master_port: 29501,
  distributed_command_timeout_s: 120,
  prompt_dominance: 0.82,
  guidance_scale: 6.5,
  inference_steps: 2,
  seed: null,
  preserve_identity: false,
  identity_lock: 0.2,
  pose_lock: 0.9,
  depth_lock: 0.75,
  segmentation_lock: 0.65,
  motion_smoothing: 0.28,
  target_fps: 30,
  output_width: 768,
  output_height: 768,
  jpeg_quality: 85,
  outputs: {
    preview: true,
    virtual_camera: false,
    rtmp_enabled: false,
    rtmp_url: null,
  },
  streamdiffusionv2_path: null,
};

const PRESET_LABELS: Record<"a" | "b", string> = {
  a: "Preset A · Cyborg Hard",
  b: "Preset B · Android Clean",
};

function round(value: number, precision = 2): number {
  const scalar = Math.pow(10, precision);
  return Math.round(value * scalar) / scalar;
}

function App() {
  const wsRef = useRef<WebSocket | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inputVideoRef = useRef<HTMLVideoElement | null>(null);
  const captureTimerRef = useRef<number | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const inFlightRef = useRef(false);

  const [sessionId, setSessionId] = useState("main");
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  const [config, setConfig] = useState<RuntimeConfig>(DEFAULT_CONFIG);
  const [configRevision, setConfigRevision] = useState(1);

  const [outputFrame, setOutputFrame] = useState<string | null>(null);
  const [lastLatencyMs, setLastLatencyMs] = useState(0);
  const [avgLatencyMs, setAvgLatencyMs] = useState(0);
  const [framesOut, setFramesOut] = useState(0);
  const [localFps, setLocalFps] = useState(0);
  const [conditioning, setConditioning] = useState<Conditioning>({
    motion_score: 0,
    edge_density: 0,
    luma_mean: 0,
  });
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus>({
    mode: "mock",
    runtime_path: null,
    error: null,
  });

  const frameTimesRef = useRef<number[]>([]);

  const wsUrl = useMemo(() => `${WS_BASE}/${encodeURIComponent(sessionId)}`, [sessionId]);

  const sendMessage = (data: object): boolean => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return false;
    }
    wsRef.current.send(JSON.stringify(data));
    return true;
  };

  const queueConfigPatch = (patch: Partial<RuntimeConfig>) => {
    sendMessage({ type: "config.update", payload: patch });
  };

  const connect = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionState("connecting");
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionState("connected");
      sendMessage({ type: "ping", payload: { at: Date.now() } });
    };

    ws.onclose = () => {
      setConnectionState("disconnected");
      if (captureTimerRef.current) {
        window.clearInterval(captureTimerRef.current);
        captureTimerRef.current = null;
        setIsStreaming(false);
      }
      inFlightRef.current = false;
    };

    ws.onerror = () => {
      setConnectionState("disconnected");
    };

    ws.onmessage = (event: MessageEvent) => {
      const message = JSON.parse(event.data) as { type: string; payload: any };

      if (message.type === "config.current") {
        const payload = message.payload as {
          config: RuntimeConfig;
          revision: number;
          metrics?: StatsPayload["metrics"];
          runtime_status?: RuntimeStatus;
        };
        setConfig(payload.config);
        setConfigRevision(payload.revision);
        if (payload.runtime_status) {
          setRuntimeStatus(payload.runtime_status);
        }
        if (payload.metrics) {
          setAvgLatencyMs(payload.metrics.avg_latency_ms ?? 0);
          setFramesOut(payload.metrics.frames_out ?? 0);
        }
        return;
      }

      if (message.type === "frame.processed") {
        const payload = message.payload as {
          image_b64: string;
          latency_ms: number;
          conditioning: Conditioning;
        };

        inFlightRef.current = false;
        setOutputFrame(`data:image/jpeg;base64,${payload.image_b64}`);
        setLastLatencyMs(payload.latency_ms);
        setConditioning(payload.conditioning);

        const now = performance.now();
        frameTimesRef.current = [...frameTimesRef.current.filter((ts) => now - ts < 1000), now];
        setLocalFps(frameTimesRef.current.length);
      }

      if (message.type === "stats") {
        const payload = message.payload as StatsPayload;
        setAvgLatencyMs(payload.metrics.avg_latency_ms ?? 0);
        setFramesOut(payload.metrics.frames_out ?? 0);
        setConfigRevision(payload.revision);
      }
    };
  };

  const disconnect = () => {
    wsRef.current?.close();
    wsRef.current = null;
  };

  const startCamera = async () => {
    if (cameraStreamRef.current) {
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });

    cameraStreamRef.current = stream;
    if (inputVideoRef.current) {
      inputVideoRef.current.srcObject = stream;
      await inputVideoRef.current.play();
    }
    setIsCameraActive(true);
  };

  const stopCamera = () => {
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((track) => track.stop());
      cameraStreamRef.current = null;
    }
    setIsCameraActive(false);
  };

  const sendFrame = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    if (!inputVideoRef.current || !captureCanvasRef.current || inFlightRef.current) {
      return;
    }

    const video = inputVideoRef.current;
    const canvas = captureCanvasRef.current;

    if (video.videoWidth <= 0 || video.videoHeight <= 0) {
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL("image/jpeg", config.jpeg_quality / 100);
    const image_b64 = dataUrl.split(",")[1];
    if (!image_b64) {
      return;
    }

    const sent = sendMessage({
      type: "frame",
      payload: {
        timestamp_ms: Date.now(),
        image_b64,
      },
    });

    if (sent) {
      inFlightRef.current = true;
    }
  };

  const startStreaming = () => {
    if (captureTimerRef.current || !isCameraActive || connectionState !== "connected") {
      return;
    }

    const intervalMs = Math.max(10, Math.floor(1000 / config.target_fps));
    captureTimerRef.current = window.setInterval(sendFrame, intervalMs);
    setIsStreaming(true);
  };

  const stopStreaming = () => {
    if (captureTimerRef.current) {
      window.clearInterval(captureTimerRef.current);
      captureTimerRef.current = null;
    }
    setIsStreaming(false);
    inFlightRef.current = false;
  };

  const applyPreset = (name: "a" | "b") => {
    sendMessage({ type: "preset.apply", payload: { name } });
  };

  const updateNumeric = <K extends keyof RuntimeConfig>(key: K, value: number) => {
    const updated = { ...config, [key]: value } as RuntimeConfig;
    setConfig(updated);
    queueConfigPatch({ [key]: value } as Partial<RuntimeConfig>);
  };

  const updateBoolean = <K extends keyof RuntimeConfig>(key: K, value: boolean) => {
    const updated = { ...config, [key]: value } as RuntimeConfig;
    setConfig(updated);
    queueConfigPatch({ [key]: value } as Partial<RuntimeConfig>);
  };

  const updateOutput = (key: keyof OutputConfig, value: string | boolean | null) => {
    const outputs = {
      ...config.outputs,
      [key]: value,
    };

    setConfig({ ...config, outputs });
    queueConfigPatch({ outputs });
  };

  const updatePrompt = (event: FormEvent<HTMLTextAreaElement>) => {
    const value = event.currentTarget.value;
    setConfig({ ...config, prompt: value });
  };

  const updateNegativePrompt = (event: FormEvent<HTMLTextAreaElement>) => {
    const value = event.currentTarget.value;
    setConfig({ ...config, negative_prompt: value });
  };

  const syncPrompts = () => {
    queueConfigPatch({
      prompt: config.prompt,
      negative_prompt: config.negative_prompt,
    });
  };

  useEffect(() => {
    return () => {
      stopStreaming();
      stopCamera();
      disconnect();
    };
  }, []);

  return (
    <div className="page-shell">
      <header className="topbar animated fade-up">
        <div>
          <h1>Project Heru</h1>
          <p>Realtime cam-to-inference control surface for prompt-dominant generation.</p>
        </div>

        <div className="connection-block">
          <label>
            Session
            <input
              value={sessionId}
              onChange={(event) => setSessionId(event.target.value.trim() || "main")}
            />
          </label>
          <div className="buttons-row">
            <button
              className="btn"
              disabled={connectionState === "connected"}
              onClick={connect}
              type="button"
            >
              Connect
            </button>
            <button className="btn secondary" onClick={disconnect} type="button">
              Disconnect
            </button>
          </div>
          <span className={`status ${connectionState}`}>{connectionState}</span>
        </div>
      </header>

      <main className="layout-grid">
        <section className="panel animated fade-up delay-1">
          <h2>Input / Output</h2>
          <div className="video-grid">
            <div>
              <h3>Camera Input</h3>
              <video ref={inputVideoRef} autoPlay muted playsInline className="video-frame" />
            </div>
            <div>
              <h3>Inference Output</h3>
              {outputFrame ? (
                <img src={outputFrame} className="video-frame" alt="Processed output" />
              ) : (
                <div className="video-frame placeholder">No output frame yet</div>
              )}
            </div>
          </div>

          <div className="buttons-row">
            <button className="btn" type="button" onClick={startCamera} disabled={isCameraActive}>
              Start Camera
            </button>
            <button className="btn secondary" type="button" onClick={stopCamera}>
              Stop Camera
            </button>
            <button
              className="btn accent"
              type="button"
              onClick={startStreaming}
              disabled={!isCameraActive || connectionState !== "connected" || isStreaming}
            >
              Start Stream
            </button>
            <button className="btn secondary" type="button" onClick={stopStreaming}>
              Stop Stream
            </button>
          </div>

          <canvas ref={captureCanvasRef} className="hidden-canvas" />

          <div className="metrics-grid">
            <article>
              <span>FPS (local)</span>
              <strong>{localFps}</strong>
            </article>
            <article>
              <span>Latency (last)</span>
              <strong>{round(lastLatencyMs)} ms</strong>
            </article>
            <article>
              <span>Latency (avg)</span>
              <strong>{round(avgLatencyMs)} ms</strong>
            </article>
            <article>
              <span>Frames out</span>
              <strong>{framesOut}</strong>
            </article>
            <article>
              <span>Motion score</span>
              <strong>{conditioning.motion_score}</strong>
            </article>
            <article>
              <span>Edge density</span>
              <strong>{conditioning.edge_density}</strong>
            </article>
            <article>
              <span>Luma mean</span>
              <strong>{conditioning.luma_mean}</strong>
            </article>
            <article>
              <span>Config rev</span>
              <strong>{configRevision}</strong>
            </article>
            <article>
              <span>Runtime mode</span>
              <strong>{runtimeStatus.mode}</strong>
            </article>
            <article>
              <span>Runtime topology</span>
              <strong>{runtimeStatus.inference_topology ?? config.inference_topology}</strong>
            </article>
          </div>
          {runtimeStatus.error ? (
            <p className="runtime-note">Core status: {runtimeStatus.error}</p>
          ) : null}
        </section>

        <section className="panel controls animated fade-up delay-2">
          <h2>Prompt + DJ Controls</h2>

          <label>
            Prompt
            <textarea value={config.prompt} onChange={updatePrompt} onBlur={syncPrompts} rows={3} />
          </label>

          <label>
            Negative Prompt
            <textarea
              value={config.negative_prompt}
              onChange={updateNegativePrompt}
              onBlur={syncPrompts}
              rows={3}
            />
          </label>

          <div className="buttons-row">
            <button className="btn" type="button" onClick={syncPrompts}>
              Sync Prompt
            </button>
            <button className="btn" type="button" onClick={() => applyPreset("a")}>
              {PRESET_LABELS.a}
            </button>
            <button className="btn" type="button" onClick={() => applyPreset("b")}>
              {PRESET_LABELS.b}
            </button>
          </div>

          <div className="control-grid">
            <label>
              Mode
              <select
                value={config.mode}
                onChange={(event) => {
                  const value = event.target.value as Mode;
                  setConfig({ ...config, mode: value });
                  queueConfigPatch({ mode: value });
                }}
              >
                <option value="performance">Performance</option>
                <option value="balanced">Balanced</option>
                <option value="quality">Quality</option>
              </select>
            </label>

            <label>
              Model
              <select
                value={config.model_variant}
                onChange={(event) => {
                  const value = event.target.value as ModelVariant;
                  setConfig({ ...config, model_variant: value });
                  queueConfigPatch({ model_variant: value });
                }}
              >
                <option value="wan-1.3b">Wan 1.3B</option>
                <option value="wan-14b">Wan 14B</option>
              </select>
            </label>

            <label>
              Inference topology
              <select
                value={config.inference_topology}
                onChange={(event) => {
                  const value = event.target.value as InferenceTopology;
                  setConfig({ ...config, inference_topology: value });
                  queueConfigPatch({ inference_topology: value });
                }}
              >
                <option value="single">Single GPU</option>
                <option value="distributed">Distributed (multi-GPU)</option>
              </select>
            </label>

            <label>
              Distributed world size
              <input
                type="number"
                min={1}
                max={8}
                value={config.distributed_world_size}
                disabled={config.inference_topology !== "distributed"}
                onChange={(event) => {
                  const value = Number.parseInt(event.target.value, 10);
                  if (!Number.isFinite(value)) {
                    return;
                  }
                  setConfig({ ...config, distributed_world_size: value });
                  queueConfigPatch({ distributed_world_size: value });
                }}
              />
            </label>

            <label>
              Distributed master addr
              <input
                type="text"
                value={config.distributed_master_addr}
                disabled={config.inference_topology !== "distributed"}
                onChange={(event) => {
                  const value = event.target.value.trim() || "127.0.0.1";
                  setConfig({ ...config, distributed_master_addr: value });
                  queueConfigPatch({ distributed_master_addr: value });
                }}
              />
            </label>

            <label>
              Distributed master port
              <input
                type="number"
                min={1024}
                max={65535}
                value={config.distributed_master_port}
                disabled={config.inference_topology !== "distributed"}
                onChange={(event) => {
                  const value = Number.parseInt(event.target.value, 10);
                  if (!Number.isFinite(value)) {
                    return;
                  }
                  setConfig({ ...config, distributed_master_port: value });
                  queueConfigPatch({ distributed_master_port: value });
                }}
              />
            </label>

            <label>
              Distributed timeout (s)
              <input
                type="number"
                min={5}
                max={600}
                step={1}
                value={config.distributed_command_timeout_s}
                disabled={config.inference_topology !== "distributed"}
                onChange={(event) => {
                  const value = Number.parseFloat(event.target.value);
                  if (!Number.isFinite(value)) {
                    return;
                  }
                  setConfig({ ...config, distributed_command_timeout_s: value });
                  queueConfigPatch({ distributed_command_timeout_s: value });
                }}
              />
            </label>

            <label>
              Seed (empty = auto)
              <input
                type="number"
                value={config.seed ?? ""}
                onChange={(event) => {
                  const raw = event.target.value.trim();
                  const value = raw.length ? Number.parseInt(raw, 10) : null;
                  setConfig({ ...config, seed: value });
                  queueConfigPatch({ seed: value });
                }}
              />
            </label>

            <label>
              Prompt dominance: {round(config.prompt_dominance)}
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={config.prompt_dominance}
                onChange={(event) =>
                  updateNumeric("prompt_dominance", Number.parseFloat(event.target.value))
                }
              />
            </label>

            <label>
              Guidance scale: {round(config.guidance_scale)}
              <input
                type="range"
                min={1}
                max={20}
                step={0.1}
                value={config.guidance_scale}
                onChange={(event) => updateNumeric("guidance_scale", Number.parseFloat(event.target.value))}
              />
            </label>

            <label>
              Inference steps: {config.inference_steps}
              <input
                type="range"
                min={1}
                max={8}
                step={1}
                value={config.inference_steps}
                onChange={(event) => updateNumeric("inference_steps", Number.parseInt(event.target.value, 10))}
              />
            </label>

            <label>
              Preserve identity
              <input
                type="checkbox"
                checked={config.preserve_identity}
                onChange={(event) => updateBoolean("preserve_identity", event.target.checked)}
              />
            </label>

            <label>
              Identity lock: {round(config.identity_lock)}
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={config.identity_lock}
                onChange={(event) => updateNumeric("identity_lock", Number.parseFloat(event.target.value))}
              />
            </label>

            <label>
              Pose lock: {round(config.pose_lock)}
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={config.pose_lock}
                onChange={(event) => updateNumeric("pose_lock", Number.parseFloat(event.target.value))}
              />
            </label>

            <label>
              Depth lock: {round(config.depth_lock)}
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={config.depth_lock}
                onChange={(event) => updateNumeric("depth_lock", Number.parseFloat(event.target.value))}
              />
            </label>

            <label>
              Seg lock: {round(config.segmentation_lock)}
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={config.segmentation_lock}
                onChange={(event) =>
                  updateNumeric("segmentation_lock", Number.parseFloat(event.target.value))
                }
              />
            </label>

            <label>
              Motion smoothing: {round(config.motion_smoothing)}
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={config.motion_smoothing}
                onChange={(event) =>
                  updateNumeric("motion_smoothing", Number.parseFloat(event.target.value))
                }
              />
            </label>

            <label>
              Target FPS: {config.target_fps}
              <input
                type="range"
                min={5}
                max={60}
                step={1}
                value={config.target_fps}
                onChange={(event) => updateNumeric("target_fps", Number.parseInt(event.target.value, 10))}
              />
            </label>

            <label>
              Output width
              <input
                type="number"
                min={256}
                max={1920}
                value={config.output_width}
                onChange={(event) => {
                  const value = Number.parseInt(event.target.value, 10);
                  if (!Number.isFinite(value)) {
                    return;
                  }
                  setConfig({ ...config, output_width: value });
                  if (value >= 256 && value <= 1920) {
                    queueConfigPatch({ output_width: value });
                  }
                }}
              />
            </label>

            <label>
              Output height
              <input
                type="number"
                min={256}
                max={1920}
                value={config.output_height}
                onChange={(event) => {
                  const value = Number.parseInt(event.target.value, 10);
                  if (!Number.isFinite(value)) {
                    return;
                  }
                  setConfig({ ...config, output_height: value });
                  if (value >= 256 && value <= 1920) {
                    queueConfigPatch({ output_height: value });
                  }
                }}
              />
            </label>

            <label>
              JPEG quality: {config.jpeg_quality}
              <input
                type="range"
                min={50}
                max={100}
                step={1}
                value={config.jpeg_quality}
                onChange={(event) => updateNumeric("jpeg_quality", Number.parseInt(event.target.value, 10))}
              />
            </label>
          </div>

          <h3>Outputs</h3>
          <div className="control-grid compact">
            <label>
              Preview
              <input
                type="checkbox"
                checked={config.outputs.preview}
                onChange={(event) => updateOutput("preview", event.target.checked)}
              />
            </label>
            <label>
              Virtual Camera
              <input
                type="checkbox"
                checked={config.outputs.virtual_camera}
                onChange={(event) => updateOutput("virtual_camera", event.target.checked)}
              />
            </label>
            <label>
              RTMP
              <input
                type="checkbox"
                checked={config.outputs.rtmp_enabled}
                onChange={(event) => updateOutput("rtmp_enabled", event.target.checked)}
              />
            </label>
            <label>
              RTMP URL
              <input
                type="text"
                value={config.outputs.rtmp_url ?? ""}
                onChange={(event) => updateOutput("rtmp_url", event.target.value || null)}
                placeholder="rtmp://..."
              />
            </label>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
