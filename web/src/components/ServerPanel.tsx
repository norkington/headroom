import { useState } from "react";
import type { ServerInfo } from "../api";
import { formatUptime, postJSON } from "../api";

/**
 * Server state and the two controls that matter.
 *
 * Both buttons are disabled while a request is in flight, because the expensive
 * mistakes here are double-clicks: two starts race for the same GPUs, and a
 * second stop lands on a server already shutting down.
 *
 * `loading` is shown as its own state rather than folded into stopped. A large
 * model takes tens of seconds to become ready, and a UI that says "stopped" for
 * that whole window invites the user to start a second one.
 */
export function ServerPanel({
  server,
  onChanged,
}: {
  server: ServerInfo;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState<null | "start" | "stop">(null);
  const [error, setError] = useState<string | null>(null);

  const act = async (which: "start" | "stop") => {
    setBusy(which);
    setError(null);
    try {
      await postJSON(`/api/server/${which}`);
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const canStart = server.status === "stopped" && busy === null;
  const canStop = (server.status === "running" || server.status === "loading") && busy === null;

  return (
    <div className="panel">
      <div className="server-row">
        <div className="server-facts">
          <div>
            <span className="k">status</span>
            <span className={`status-badge status-${server.status}`}>{server.status}</span>
          </div>
          <div>
            <span className="k">model</span>
            {server.model_name ?? "—"}
          </div>
          <div>
            <span className="k">context</span>
            {server.n_ctx?.toLocaleString() ?? "—"}
          </div>
          <div>
            <span className="k">vision</span>
            {server.vision ? "on" : "off"}
          </div>
          <div>
            <span className="k">uptime</span>
            {formatUptime(server.uptime_seconds)}
          </div>
          <div>
            <span className="k">pid</span>
            {server.pid ?? "—"}
          </div>
        </div>

        <div className="actions">
          <button className="primary" disabled={!canStart} onClick={() => void act("start")}>
            {busy === "start" ? "Starting…" : "Start"}
          </button>
          <button className="danger" disabled={!canStop} onClick={() => void act("stop")}>
            {busy === "stop" ? "Stopping…" : "Stop"}
          </button>
        </div>
      </div>

      {server.status === "loading" && (
        <div className="error-line" style={{ borderLeftColor: "var(--tight)", background: "var(--tight-bg)" }}>
          Model is loading. The process is up but not serving yet — this takes
          tens of seconds for a large model. Don&rsquo;t start a second one.
        </div>
      )}

      {server.error && <div className="error-line">{server.error}</div>}
      {error && <div className="error-line">{error}</div>}
    </div>
  );
}
