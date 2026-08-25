import { useCallback, useEffect, useState } from "react";
import type { CudaMapping, Health, ModelSummary } from "./api";
import { getJSON } from "./api";
import { useTelemetry } from "./useTelemetry";
import { GpuCard } from "./components/GpuCard";
import { ServerPanel } from "./components/ServerPanel";
import { ModelList } from "./components/ModelList";
import { ProbePanel } from "./components/ProbePanel";

interface GpuResponse {
  cuda_mapping: CudaMapping;
}

export function App() {
  const feed = useTelemetry();
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [mapping, setMapping] = useState<CudaMapping | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    void getJSON<Health>("/api/health").then(setHealth).catch(() => undefined);
    void getJSON<{ models: ModelSummary[] }>("/api/models")
      .then((d) => setModels(d.models))
      .catch((e: Error) => setLoadError(e.message));
    void getJSON<GpuResponse>("/api/gpus")
      .then((d) => setMapping(d.cuda_mapping))
      .catch(() => undefined);
  }, []);

  useEffect(refresh, [refresh]);

  const gpus = feed.data?.gpus ?? [];
  const server = feed.data?.server ?? null;
  const totalFree = gpus.reduce((sum, g) => sum + g.memory_free_mib, 0);
  const tightest = gpus.reduce<number | null>(
    (min, g) => (min === null || g.memory_free_mib < min ? g.memory_free_mib : min),
    null,
  );

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <h1>Headroom</h1>
          <div className="sub">
            {health?.registry ?? "loading registry…"}
          </div>
        </div>
        <span className={`conn ${feed.connection}`}>
          <span className="dot" />
          {feed.connection === "live"
            ? "live"
            : feed.connection === "connecting"
              ? "connecting"
              : `stale ${Math.round(feed.staleness)}s`}
        </span>
      </header>

      {/* The order-mismatch warning is the single most valuable thing this app
          can tell someone, so it sits above everything else rather than in a
          details panel. */}
      {mapping?.order_differs && (
        <div className="notice">
          <strong>Device order differs.</strong> On this machine{" "}
          <code>nvidia-smi</code> and llama.cpp number the GPUs differently, so{" "}
          <code>nvidia-smi -i 0</code> and <code>-dev CUDA0</code> refer to{" "}
          <em>different cards</em>. The <code>CUDA</code> tags below are
          llama.cpp&rsquo;s numbering — the one to use in your flags.
        </div>
      )}
      {mapping?.warning && (
        <div className="notice">
          <strong>CUDA mapping incomplete.</strong> {mapping.warning}
        </div>
      )}

      <section>
        <h2>
          GPUs
          {tightest !== null && (
            <> — {totalFree.toLocaleString()} MiB free total, tightest card {tightest.toLocaleString()} MiB</>
          )}
        </h2>
        {gpus.length === 0 ? (
          <div className="empty">
            {feed.connection === "lost"
              ? "Telemetry stream lost. Is the Headroom backend running?"
              : "Waiting for telemetry…"}
          </div>
        ) : (
          <div className="gpu-grid">
            {gpus.map((g) => (
              <GpuCard key={g.nvml_index} gpu={g} />
            ))}
          </div>
        )}
      </section>

      <section>
        <h2>Inference server</h2>
        {server ? (
          <ServerPanel server={server} onChanged={refresh} />
        ) : (
          <div className="empty">Waiting for server state…</div>
        )}
      </section>

      <section>
        <h2>Inspect a quant before downloading it</h2>
        <ProbePanel />
      </section>

      <section>
        <h2>Models</h2>
        {loadError ? <div className="error-line">{loadError}</div> : <ModelList models={models} />}
      </section>
    </div>
  );
}
