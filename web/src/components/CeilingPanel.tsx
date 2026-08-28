import { useCallback, useEffect, useState } from "react";
import type { ServerInfo } from "../api";
import { getJSON, postJSON } from "../api";

/**
 * Find the largest context this machine can actually hold.
 *
 * The registry records a context length and nothing verifies it. On the
 * development box the recorded 64K was arrived at by hand over three manual
 * reloads, for one model — and when this search was first run against it, 64K
 * turned out to leave the tightest card at 1,012 MiB, below the very threshold
 * the GPU panel grades as tight.
 *
 * Two things about the presentation follow from what the search costs and what
 * it means:
 *
 * - **Every probe is a real model load**, so the probe table is the receipt.
 *   Showing only the answer would hide that it cost four minutes of GPU and
 *   which contexts were actually tried.
 *
 * - **The answer is situational and is not saved.** It describes this machine
 *   as it was at that moment; another workload on these cards changes it. So
 *   the panel says so, and recording it in models.json stays a deliberate act.
 */

interface CeilingProbe {
  ctx: number;
  loaded: boolean;
  free_mib: number | null;
  breakdown: string | null;
  within_margin: boolean;
  seconds: number;
  error: string | null;
}

interface CeilingResult {
  found: boolean;
  ctx?: number;
  free_mib?: number;
  breakdown?: string | null;
  mib_per_token?: number | null;
  tokens_per_gib?: number | null;
  first_failure?: number | null;
  note: string;
}

interface CeilingInfo {
  id: string;
  model_key: string;
  status: "queued" | "running" | "complete" | "failed" | "cancelled";
  phase: string;
  margin_mib: number;
  probes: CeilingProbe[];
  probes_done: number;
  max_probes: number;
  best_ctx: number | null;
  mib_per_token: number | null;
  result: CeilingResult | null;
  error: string | null;
  elapsed_seconds: number;
}

const POLL_MS = 2000;

function isActive(s: CeilingInfo | null): boolean {
  return s !== null && (s.status === "running" || s.status === "queued");
}

export function CeilingPanel({ server }: { server: ServerInfo }) {
  const [current, setCurrent] = useState<CeilingInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const poll = useCallback(async () => {
    try {
      const d = await getJSON<{ searches: CeilingInfo[] }>("/api/ceiling");
      setCurrent(d.searches[0] ?? null);
    } catch {
      /* the backend may be restarting; the next tick picks it up */
    }
  }, []);

  useEffect(() => {
    void poll();
  }, [poll]);

  useEffect(() => {
    if (!isActive(current)) return;
    const timer = window.setInterval(() => void poll(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [current, poll]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      setCurrent(await postJSON<CeilingInfo>("/api/ceiling/start"));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (id: string) => {
    await fetch(`/api/ceiling/${id}`, { method: "DELETE" });
    void poll();
  };

  const active = isActive(current);
  // Refused while a server is up, because the search stops and starts one
  // repeatedly and the loaded model is somebody's working state.
  const stopped = server.status === "stopped";
  const result = current?.result ?? null;

  return (
    <div className="panel">
      <div className="server-row">
        <div className="server-facts">
          <div>
            <span className="k">looking for</span>
            largest context
          </div>
          <div>
            <span className="k">margin</span>
            {current?.margin_mib ?? 1200} MiB per card
          </div>
          <div>
            <span className="k">method</span>
            real loads, not arithmetic
          </div>
        </div>

        <div className="actions">
          <button className="primary" disabled={!stopped || active || busy} onClick={() => void start()}>
            {active ? "Searching…" : busy ? "Starting…" : "Find the ceiling"}
          </button>
          {active && current && (
            <button className="danger" onClick={() => void cancel(current.id)}>
              Cancel
            </button>
          )}
        </div>
      </div>

      {!stopped && (
        <div className="empty">
          Stop the server first. This starts and stops one several times, and it will not unload a
          model you are using.
        </div>
      )}

      {stopped && !current && (
        <div className="finding info">
          <div className="finding-detail">
            Tries context lengths for real — start, wait, measure free VRAM, stop — until it finds
            the largest that still leaves every card above the margin. Not the largest that
            <em> loads</em>: a context that fits with 200 MiB to spare is one browser tab from an
            OOM mid-generation. Expect a few minutes; each probe is a full model load.
          </div>
        </div>
      )}

      {active && current && (
        <div className="bench-progress">
          <div className="bench-phase">
            <span>{current.phase}</span>
            <span className="picker-meta">
              probe {current.probes_done} of at most {current.max_probes} ·{" "}
              {Math.round(current.elapsed_seconds)}s elapsed
            </span>
          </div>
          <div className="meter">
            <div
              className="used"
              style={{ width: `${(current.probes_done / current.max_probes) * 100}%` }}
            />
          </div>
        </div>
      )}

      {current && current.probes.length > 0 && (
        <table className="bench-tasks">
          <thead>
            <tr>
              <th>context</th>
              <th>free on tightest</th>
              <th>took</th>
              <th>verdict</th>
            </tr>
          </thead>
          <tbody>
            {current.probes.map((p) => (
              <tr key={p.ctx}>
                <td>{p.ctx.toLocaleString()}</td>
                <td>{p.free_mib === null ? "—" : `${p.free_mib} MiB`}</td>
                <td>{p.seconds.toFixed(1)}s</td>
                <td>
                  {!p.loaded ? (
                    <span className="thermal-hot">did not load</span>
                  ) : p.within_margin ? (
                    "within margin"
                  ) : (
                    <span className="thermal-warm">below margin</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {current?.status === "complete" && result && (
        <div className="bench-result">
          {result.found ? (
            <div className="measured-row">
              <div>
                <span className="k">ceiling</span>
                {result.ctx?.toLocaleString()}
                <span className="fig-sub">at this margin</span>
              </div>
              <div>
                <span className="k">free there</span>
                {result.free_mib} MiB
                <span className="fig-sub">tightest card</span>
              </div>
              <div>
                <span className="k">cost</span>
                {result.mib_per_token?.toFixed(4) ?? "—"}
                <span className="fig-sub">MiB per token, measured</span>
              </div>
              <div>
                <span className="k">per GiB</span>
                {result.tokens_per_gib?.toLocaleString() ?? "—"}
                <span className="fig-sub">tokens</span>
              </div>
            </div>
          ) : null}

          <div className="finding info">
            <div className="finding-title">What this means</div>
            <div className="finding-detail">{result.note}</div>
          </div>

          {result.breakdown && <div className="bench-caption">{result.breakdown}</div>}
        </div>
      )}

      {current?.status === "cancelled" && (
        <div className="finding info">
          <div className="finding-detail">
            Cancelled. {current.error} The probes above are still real measurements.
          </div>
        </div>
      )}

      {current?.status === "failed" && current.error && (
        <div className="error-line">{current.error}</div>
      )}
      {error && <div className="error-line">{error}</div>}
    </div>
  );
}
