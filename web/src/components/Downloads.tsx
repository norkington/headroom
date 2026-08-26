import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getJSON } from "../api";

/**
 * Downloads: the ones in flight, and the ones this machine has already pulled.
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
 *
 * Records now outlive the backend process, which changes what this panel is.
 * Two consequences are deliberate:
 *
 * - **An interrupted transfer is offered back, not mourned.** Headroom being
 *   restarted mid-download never lost the bytes — the `.part` file is still
 *   there — but it used to lose the record naming the repo they came from,
 *   which left a 9 GiB orphan on disk and nothing pointing at it. That entry
 *   now comes back with a Resume button.
 *
 * - **Finished transfers fold away.** History is worth keeping and not worth
 *   looking at, so it goes behind a summary rather than pushing the live
 *   transfer down the page.
 */

type DownloadStatus =
  | "queued"
  | "running"
  | "complete"
  | "failed"
  | "cancelled"
  | "interrupted";

interface DownloadInfo {
  id: string;
  filename: string;
  repo: string;
  status: DownloadStatus;
  total_bytes: number | null;
  downloaded_bytes: number;
  percent: number | null;
  bytes_per_second: number | null;
  eta_seconds: number | null;
  attempt: number;
  error: string | null;
  /** Whether there is a transfer here that could be picked up again. */
  resumable: boolean;
  elapsed_seconds: number;
}

const LIVE: DownloadStatus[] = ["queued", "running", "interrupted"];

/**
 * Bytes at a unit that shows what is actually there.
 *
 * Fixed GiB reads "0.00" for anything under about 5 MiB, which is exactly the
 * wrong thing to print next to "resuming continues from there": a partial file
 * that is genuinely on disk looks like nothing at all, and the sentence
 * explaining why it is worth resuming argues against itself.
 */
function size(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024).toFixed(0)} KiB`;
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

function badgeClass(status: DownloadStatus): string {
  if (status === "running" || status === "queued") return "status-loading";
  if (status === "complete") return "status-running";
  if (status === "failed") return "status-orphaned";
  return "status-stopped";
}

export function Downloads({ onComplete }: { onComplete?: () => void }) {
  const [items, setItems] = useState<DownloadInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  const tick = useCallback(async () => {
    try {
      const d = await getJSON<{ downloads: DownloadInfo[] }>("/api/downloads");
      setItems(d.downloads);
      return d.downloads;
    } catch {
      /* the backend may be restarting; the next tick will pick it up */
      return null;
    }
  }, []);

  // Which completions the parent has already been told about. A ref rather
  // than state because it must survive the polling effect being rebuilt when
  // the cadence changes, and because writing to it should never re-render.
  const announced = useRef<Set<string> | null>(null);

  const active = useMemo(
    () => items.some((d) => d.status === "running" || d.status === "queued"),
    [items],
  );

  useEffect(() => {
    let stopped = false;

    const poll = async () => {
      const downloads = await tick();
      if (stopped || !downloads) return;
      // The first poll seeds the set rather than firing for everything in it.
      // Restored records are finished transfers from a previous run, and each
      // one announcing itself as freshly complete would refresh the model list
      // several times over for news that is days old.
      if (announced.current === null) {
        announced.current = new Set(
          downloads.filter((d) => d.status === "complete").map((d) => d.id),
        );
        return;
      }
      for (const item of downloads) {
        if (item.status === "complete" && !announced.current.has(item.id)) {
          announced.current.add(item.id);
          onComplete?.();
        }
      }
    };

    void poll();
    // A second is right for a transfer in flight and pointless for a list of
    // things that finished last week, so the timer follows the work.
    const timer = window.setInterval(() => void poll(), active ? 1000 : 5000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [tick, onComplete, active]);

  const cancel = async (id: string) => {
    await fetch(`/api/downloads/${id}`, { method: "DELETE" });
    void tick();
  };

  const resume = async (id: string) => {
    setError(null);
    const res = await fetch(`/api/downloads/${id}/resume`, { method: "POST" });
    if (!res.ok) {
      const body = (await res.json().catch(() => ({}))) as { detail?: string };
      setError(body.detail ?? res.statusText);
    }
    void tick();
  };

  const [live, earlier] = useMemo(
    () => [
      items.filter((d) => LIVE.includes(d.status)),
      items.filter((d) => !LIVE.includes(d.status)),
    ],
    [items],
  );

  if (items.length === 0) return null;

  const card = (d: DownloadInfo) => (
    <div className={`download ${d.status}`} key={d.id}>
      <div className="download-head">
        <span className="dl-name">{d.filename}</span>
        <span className={`status-badge ${badgeClass(d.status)}`}>{d.status}</span>
      </div>

      <div className="meter">
        <div className="used" style={{ width: `${d.percent ?? 0}%` }} />
      </div>

      <div className="download-stats">
        <div>
          <span className="k">progress</span>
          {size(d.downloaded_bytes)}
          {d.total_bytes ? ` / ${size(d.total_bytes)}` : ""}
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
          {d.resumable && <button onClick={() => void resume(d.id)}>Resume</button>}
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

      {d.status === "interrupted" && (
        <div className="finding info">
          <div className="finding-detail">
            Headroom stopped while this was transferring — nothing went wrong with the download
            itself. {size(d.downloaded_bytes)} is on disk in a <code>.part</code> file, and
            resuming continues from there rather than starting again. It came from{" "}
            <code>{d.repo}</code>.
          </div>
        </div>
      )}

      {d.error && <div className="error-line">{d.error}</div>}
    </div>
  );

  return (
    <section>
      <h2>Downloads</h2>
      {live.map(card)}
      {error && <div className="error-line">{error}</div>}
      {earlier.length > 0 && (
        <details className="dl-earlier">
          <summary>
            {earlier.length} finished transfer{earlier.length === 1 ? "" : "s"}
          </summary>
          {earlier.map(card)}
        </details>
      )}
    </section>
  );
}
