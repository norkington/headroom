import { useCallback, useEffect, useRef, useState } from "react";
import type { BenchInfo, ServerInfo } from "../api";
import { getJSON, postJSON } from "../api";

/**
 * Measure the running model, and write the result into the registry.
 *
 * This panel exists because an entry added through the UI is marked NOT
 * MEASURED and, until now, there was no way to stop it being. An unmeasured
 * entry is not a small gap: every serve value in it is a guess inherited from a
 * template or a sibling, and the app's whole claim is that it distinguishes
 * those from figures someone actually observed.
 *
 * What is shown here follows from what the numbers mean:
 *
 * - **Decode never appears without its spread and its accept rate.** Decode
 *   tracks speculative-decoding acceptance, which varies by task; a bare decode
 *   figure invites reading a task difference as a regression. The ~6% rule is
 *   printed next to the result rather than left in a docstring, because the
 *   moment someone is about to compare two runs is the moment they need it.
 *
 * - **A prefill figure built from cached runs is called out.** The server caches
 *   prompts, and a run whose reps were served from cache has a far smaller
 *   sample than its rep count suggests.
 *
 * - **The write is stated, with its backup.** This writes to a file shared with
 *   the user's shell scripts; that should never be something they discover
 *   afterwards.
 */

const POLL_MS = 1000;

function isActive(b: BenchInfo | null): boolean {
  return b !== null && (b.status === "running" || b.status === "queued");
}

function num(value: number | null | undefined, digits: number, unit = ""): string {
  if (value == null) return "—";
  return `${value.toFixed(digits)}${unit}`;
}

export function BenchPanel({
  server,
  onRecorded,
}: {
  server: ServerInfo;
  onRecorded: () => void;
}) {
  const [current, setCurrent] = useState<BenchInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recorded = useRef<string | null>(null);

  const poll = useCallback(async () => {
    try {
      const d = await getJSON<{ benchmarks: BenchInfo[] }>("/api/bench");
      const latest = d.benchmarks[0] ?? null;
      setCurrent(latest);
      // Refresh the model list once when a run lands, not on every poll.
      if (latest && latest.status === "complete" && recorded.current !== latest.id) {
        recorded.current = latest.id;
        if (latest.written) onRecorded();
      }
    } catch {
      /* the backend may be restarting; the next tick picks it up */
    }
  }, [onRecorded]);

  useEffect(() => {
    void poll();
  }, [poll]);

  // Poll only while something is in flight. A benchmark is a rare, deliberate
  // action, so a permanent 1 Hz timer against an idle endpoint would be pure
  // noise next to the telemetry stream that actually needs one.
  useEffect(() => {
    if (!isActive(current)) return;
    const timer = window.setInterval(() => void poll(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [current, poll]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const started = await postJSON<BenchInfo>("/api/bench/start");
      setCurrent(started);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (id: string) => {
    await fetch(`/api/bench/${id}`, { method: "DELETE" });
    void poll();
  };

  const running = server.status === "running";
  const attributable = server.model_key !== null;
  const active = isActive(current);
  const canStart = running && attributable && !active && !busy;

  const result = current?.result ?? null;

  return (
    <div className="panel">
      <div className="server-row">
        <div className="server-facts">
          <div>
            <span className="k">target</span>
            {server.model_key ?? "—"}
          </div>
          <div>
            <span className="k">method</span>
            same as bin/bench.ps1
          </div>
          <div>
            <span className="k">records to</span>
            models.json
          </div>
        </div>

        <div className="actions">
          <button className="primary" disabled={!canStart} onClick={() => void start()}>
            {active ? "Running…" : busy ? "Starting…" : "Run benchmark"}
          </button>
          {active && current && (
            <button className="danger" onClick={() => void cancel(current.id)}>
              Cancel
            </button>
          )}
        </div>
      </div>

      {!running && (
        <div className="empty">
          Nothing is serving. Start a model to benchmark it — these figures are measured against a
          live server, never estimated.
        </div>
      )}

      {running && !attributable && (
        <div className="finding info">
          <div className="finding-detail">
            The loaded file is not in the registry, so there is no entry to record a measurement
            on. Headroom will not guess which entry the numbers belong to.
          </div>
        </div>
      )}

      {active && current && (
        <>
          <div className="bench-progress">
            <div className="bench-phase">
              <span>{current.phase}</span>
              <span className="picker-meta">
                run {current.runs_done} of {current.runs_total} · {Math.round(current.elapsed_seconds)}s
                elapsed
              </span>
            </div>
            <div className="meter">
              <div className="used" style={{ width: `${current.percent ?? 0}%` }} />
            </div>
          </div>
          <div className="finding info">
            <div className="finding-detail">
              This takes a few minutes and loads the GPUs continuously — the headroom figures above
              will read as busy for the duration, which is the point. The first few runs are
              warm-up and are thrown away: a model that has just loaded reports a decode rate it
              will not sustain.
            </div>
          </div>
        </>
      )}

      {current?.status === "cancelled" && (
        <div className="finding info">
          <div className="finding-detail">
            Cancelled. Nothing was recorded — a partial run is not a measurement.
          </div>
        </div>
      )}

      {current?.status === "complete" && result && (
        <div className="bench-result">
          <div className="measured-row">
            <div>
              <span className="k">decode</span>
              {num(result.decode_tok_s, 2, " tok/s")}
              {result.decode_sd != null && (
                <span className="picker-meta"> ± {result.decode_sd.toFixed(2)} SD</span>
              )}
            </div>
            <div>
              <span className="k">prefill</span>
              {num(result.prefill_tok_s, 1, " tok/s")}
            </div>
            <div>
              <span className="k">acceptance</span>
              {result.acceptance_range ?? "n/a"}
            </div>
            <div>
              <span className="k">context</span>
              {result.n_ctx?.toLocaleString() ?? "—"}
            </div>
            <div>
              <span className="k">free vram</span>
              {result.vram_free_mib != null ? `${result.vram_free_mib} MiB` : "—"}
            </div>
          </div>

          {/* Per card, because the total is the misleading half — several GiB
              "free" can be one comfortable card and one that is a browser tab
              away from an OOM. */}
          {result.vram_free_breakdown && (
            <div className="picker-meta" style={{ marginTop: 8, display: "block" }}>
              {result.vram_free_breakdown}
            </div>
          )}

          {Object.keys(current.per_task).length > 0 && (
            <table className="bench-tasks">
              <thead>
                <tr>
                  <th>task</th>
                  <th>decode</th>
                  <th>acceptance</th>
                  <th>runs</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(current.per_task).map(([name, t]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{num(t.decode_tok_s, 2)}</td>
                    <td>{num(t.acceptance, 3)}</td>
                    <td>{t.runs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <div className="finding info">
            <div className="finding-detail">
              {result.significance_note} Decode moves <em>with</em> acceptance, so check that
              before reading a difference between tasks as a property of the model.
            </div>
          </div>

          {result.prefill_tok_s == null && (
            <div className="error-line" style={{ borderLeftColor: "var(--tight)", background: "var(--tight-bg)" }}>
              Every prefill run was served from the prompt cache, so no honest prefill figure is
              available. Restart the server and run again.
            </div>
          )}

          {result.prefill_cached_runs > 0 && result.prefill_tok_s != null && (
            <div className="finding info">
              <div className="finding-detail">
                {result.prefill_cached_runs} prefill run
                {result.prefill_cached_runs === 1 ? "" : "s"} hit the prompt cache and{" "}
                {result.prefill_cached_runs === 1 ? "was" : "were"} discarded. The prefill figure
                rests on a smaller sample than the rep count suggests.
              </div>
            </div>
          )}

          {current.written ? (
            <div className="finding good">
              <div className="finding-detail">
                Recorded on <code>{current.model_key}</code> in models.json, which is no longer
                marked NOT MEASURED. The previous file was backed up alongside it as{" "}
                <code>models.json.bak</code>; the <code>serve</code> block was not touched.
              </div>
            </div>
          ) : (
            <div className="finding info">
              <div className="finding-detail">
                Measured but not recorded — the registry still shows whatever it showed before.
              </div>
            </div>
          )}
        </div>
      )}

      {current?.status === "failed" && current.error && (
        <div className="error-line">{current.error}</div>
      )}
      {error && <div className="error-line">{error}</div>}
    </div>
  );
}
