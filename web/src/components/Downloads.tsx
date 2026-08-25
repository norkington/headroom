import { useEffect, useState } from "react";
import { getJSON } from "../api";

/**
 * In-flight downloads.
 *
 * Shows **throughput over a short trailing window**, not an average since the
 * transfer began. A running average keeps reporting a healthy figure for
 * minutes after throughput has collapsed, which is exactly the reassurance that
 * makes a wedged transfer hard to notice — and a wedged transfer at 9 GiB into
 * a 15 GiB model is precisely the failure this project's download code exists
 * to prevent.
 *
 * Retries are surfaced rather than hidden. A transfer that has resumed four
 * times is technically fine and worth knowing about.
 */

interface DownloadInfo {
  id: string;
  filename: string;
  repo: string;
  status: "queued" | "running" | "complete" | "failed" | "cancelled";
  total_bytes: number | null;
  downloaded_bytes: number;
  percent: number | null;
  bytes_per_second: number | null;
  eta_seconds: number | null;
  attempt: number;
  error: string | null;
  elapsed_seconds: number;
}

function gib(bytes: number): string {
  return (bytes / 1024 ** 3).toFixed(2);
}

function rate(bps: number | null): string {
  if (bps == null) return "—";
  const mib = bps / 1024 ** 2;
  return `${mib.toFixed(1)} MiB/s`;
}

function eta(seconds: number | null): string {
  if (seconds == null) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function Downloads({ onComplete }: { onComplete?: () => void }) {
  const [items, setItems] = useState<DownloadInfo[]>([]);

  useEffect(() => {
    let completed = new Set<string>();
    const tick = async () => {
      try {
        const d = await getJSON<{ downloads: DownloadInfo[] }>("/api/downloads");
        setItems(d.downloads);
        // Tell the parent once per download, so the model list refreshes when
        // something finishes rather than on every poll.
        for (const item of d.downloads) {
          if (item.status === "complete" && !completed.has(item.id)) {
            completed = new Set(completed).add(item.id);
            onComplete?.();
          }
        }
      } catch {
        /* the backend may be restarting; the next tick will pick it up */
      }
    };
    void tick();
    const timer = window.setInterval(() => void tick(), 1000);
    return () => window.clearInterval(timer);
  }, [onComplete]);

  const cancel = async (id: string) => {
    await fetch(`/api/downloads/${id}`, { method: "DELETE" });
  };

  if (items.length === 0) return null;

  return (
    <section>
      <h2>Downloads</h2>
      {items.map((d) => (
        <div className={`download ${d.status}`} key={d.id}>
          <div className="download-head">
            <span className="dl-name">{d.filename}</span>
            <span className={`status-badge status-${d.status === "running" ? "loading" : d.status === "complete" ? "running" : "stopped"}`}>
              {d.status}
            </span>
          </div>

          <div className="meter">
            <div className="used" style={{ width: `${d.percent ?? 0}%` }} />
          </div>

          <div className="download-stats">
            <div>
              <span className="k">progress</span>
              {gib(d.downloaded_bytes)}
              {d.total_bytes ? ` / ${gib(d.total_bytes)} GiB` : " GiB"}
            </div>
            <div>
              <span className="k">rate</span>
              {rate(d.bytes_per_second)}
            </div>
            <div>
              <span className="k">eta</span>
              {eta(d.eta_seconds)}
            </div>
            <div>
              <span className="k">attempt</span>
              {d.attempt}
            </div>
            <div>
              {(d.status === "running" || d.status === "queued") && (
                <button onClick={() => void cancel(d.id)}>Cancel</button>
              )}
            </div>
          </div>

          {d.attempt > 1 && d.status === "running" && (
            <div className="finding info">
              <div className="finding-detail">
                Resumed {d.attempt - 1} time{d.attempt > 2 ? "s" : ""} after a stall. Progress is
                kept, so each retry continues rather than starting over.
              </div>
            </div>
          )}

          {d.error && <div className="error-line">{d.error}</div>}
        </div>
      ))}
    </section>
  );
}
