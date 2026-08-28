import { useEffect, useState } from "react";
import type { ModelSummary, ServerInfo } from "../api";
import { formatUptime, postJSON } from "../api";

/**
 * Server state, the model picker, and the two controls that matter.
 *
 * Both buttons are disabled while a request is in flight, because the expensive
 * mistakes here are double-clicks: two starts race for the same GPUs, and a
 * second stop lands on a server already shutting down.
 *
 * `loading` is shown as its own state rather than folded into stopped. A large
 * model takes tens of seconds to become ready, and a UI that says "stopped" for
 * that whole window invites the user to start a second one.
 *
 * The picker shows what *would* start, and disappears behind the running model
 * once something is up. Leaving a live dropdown next to a running server
 * suggests the selection describes what is loaded — it does not, and the gap
 * between the two is precisely where someone reads the wrong model's numbers.
 * While a server is running the panel reports the registry key it actually
 * resolved to, which comes from the file on its command line.
 */
export function ServerPanel({
  server,
  models,
  registryDefault,
  onChanged,
}: {
  server: ServerInfo;
  models: ModelSummary[];
  registryDefault: string | null;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState<null | "start" | "stop">(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string>("");
  const [vision, setVision] = useState(false);
  // Blank means "use the registry", which is the normal case. Only a deliberate
  // value overrides, so an experiment cannot silently become the default.
  const [ctx, setCtx] = useState<string>("");

  const startable = models.filter((m) => m.installed);
  const chosen = models.find((m) => m.key === selected) ?? null;
  // What would be used if the box is left blank. This mirrors `build_argv`'s
  // precedence deliberately -- vision.ctx, else serve.ctx -- because the number
  // shown beside the Start button and the number that reaches --ctx-size have
  // to be the same one. It read `serve.vision_ctx` until 2026-08-28: a key the
  // registry has never had, so a vision start silently displayed the text
  // context instead of the profile's.
  const registryCtx = chosen
    ? Number(
        (vision && chosen.vision_supported && (chosen.vision?.["ctx"] as number)) ||
          chosen.serve["ctx"] ||
          0,
      ) || null
    : null;
  // What a Start would actually load right now: the override if one is typed,
  // otherwise whatever the registry resolved to above. Worth stating outright,
  // because it was the one number the panel left the user to infer -- from a
  // placeholder, in a field too narrow to show it, and not shown at all once
  // vision was ticked and the profile's own context took over.
  const typedCtx = Number(ctx.trim()) || null;
  const effectiveCtx = ctx.trim() ? typedCtx : registryCtx;

  // Seed the selection once the registry has loaded: whatever is running, else
  // the registry default, else the first installed entry. Re-seeding on every
  // render would fight the user's own choice, so this only fills a blank.
  useEffect(() => {
    if (selected || models.length === 0) return;
    const seed =
      (server.model_key && models.some((m) => m.key === server.model_key) && server.model_key) ||
      (registryDefault && models.some((m) => m.key === registryDefault) && registryDefault) ||
      startable[0]?.key ||
      models[0]?.key ||
      "";
    if (seed) setSelected(seed);
  }, [models, registryDefault, server.model_key, selected, startable]);

  // A model without a projector cannot serve vision, and leaving the box ticked
  // across a change of selection would produce a start that fails at argv build
  // time for a reason the user did not choose.
  useEffect(() => {
    if (chosen && !chosen.vision_supported && vision) setVision(false);
  }, [chosen, vision]);

  const act = async (which: "start" | "stop") => {
    setBusy(which);
    setError(null);
    try {
      const trimmed = ctx.trim();
      const query =
        which === "start" && selected
          ? `?model=${encodeURIComponent(selected)}&vision=${vision ? "true" : "false"}` +
            (trimmed ? `&ctx=${encodeURIComponent(trimmed)}` : "")
          : "";
      await postJSON(`/api/server/${which}${query}`);
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const stopped = server.status === "stopped";
  const loading = server.status === "loading";
  const canStart = stopped && busy === null && chosen !== null && chosen.installed;
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
            {server.model_key ?? server.model_name ?? "—"}
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

      {stopped && (
        <div className="picker">
          <label className="picker-field">
            <span className="k">start</span>
            <select
              value={selected}
              disabled={busy !== null || models.length === 0}
              onChange={(e) => setSelected(e.target.value)}
            >
              {models.length === 0 && <option value="">no models in the registry</option>}
              {models.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.key}
                  {m.installed ? "" : " — not installed"}
                  {m.key === registryDefault ? " (default)" : ""}
                </option>
              ))}
            </select>
          </label>

          <label
            className={`picker-check${chosen?.vision_supported ? "" : " disabled"}`}
            title={
              chosen?.vision_supported
                ? "Loads the projector. A different operating point: less context, more VRAM."
                : "This model has no projector in the registry."
            }
          >
            <input
              type="checkbox"
              checked={vision}
              disabled={!chosen?.vision_supported || busy !== null}
              onChange={(e) => setVision(e.target.checked)}
            />
            vision
          </label>

          <label
            className="picker-field compact"
            title="Override the context length for this start only. Blank uses the registry."
          >
            <span className="k">ctx</span>
            <input
              type="number"
              className="ctx-input"
              min={512}
              step={1024}
              placeholder={registryCtx ? String(registryCtx) : "registry"}
              value={ctx}
              disabled={busy !== null}
              onChange={(e) => setCtx(e.target.value)}
            />
          </label>

          {chosen && (
            <span className="picker-meta">
              {chosen.size_gib.toFixed(2)} GiB · {chosen.arch}
              {vision && chosen.vision_supported ? " · vision profile" : ""}
              {effectiveCtx !== null && (
                <>
                  {" · "}
                  <strong className="picker-ctx">{effectiveCtx.toLocaleString()}</strong> ctx
                  {ctx.trim() ? " (override)" : ""}
                </>
              )}
            </span>
          )}
        </div>
      )}

      {/* Vision is not a flag on the same configuration — it is a different
          operating point that trades context and speed for the projector's
          VRAM. Saying so at the moment of choosing costs less than a failed
          load explains later. */}
      {stopped && vision && chosen?.vision_supported && chosen.vision_tuned && (
        <div className="finding info">
          <div className="finding-detail">
            The vision profile is its own operating point, not a flag:{" "}
            <code>{chosen.key}</code> carries a separate context length and tensor split for it,
            and the projector takes VRAM the text-only profile spends on context.
          </div>
        </div>
      )}

      {/* Whether that operating point is actually *known* is the part worth
          separating. An entry added through the UI gets a projector and nothing
          else, so vision starts at the text context -- an honest attempt that is
          often too tight, and far easier to understand before the load than
          after it. */}
      {stopped && vision && chosen?.vision_supported && !chosen.vision_tuned && (
        <div className="finding caution">
          <div className="finding-title">This vision profile is untuned</div>
          <div className="finding-detail">
            <code>{chosen.key}</code> has a projector but no measured operating point, so vision
            will start at the full text context
            {chosen.serve["ctx"] != null
              ? ` (${Number(chosen.serve["ctx"]).toLocaleString()})`
              : ""}
            . The projector&rsquo;s VRAM comes out of the context budget, so this may fail to
            allocate. If it does, shorten the context and record what worked in the
            registry&rsquo;s <code>vision</code> block — that is the measurement Headroom will
            not guess at.
          </div>
        </div>
      )}

      {/* An override is a one-off experiment, not a decision. Saying so at the
          moment of starting is what keeps someone from later reading a
          measurement taken at 96k as though the registry said 96k. */}
      {stopped && ctx.trim() && (
        <div className="finding info">
          <div className="finding-detail">
            Starting at <strong>{Number(ctx.trim()).toLocaleString()}</strong> context instead of
            the registry&rsquo;s{registryCtx ? ` ${registryCtx.toLocaleString()}` : ""}. This
            applies to this start only and is <em>not</em> written to models.json — if it turns
            out to be the right value, put it there deliberately. A context the cards cannot fit
            fails at load, which is the cheap way to find the ceiling.
          </div>
        </div>
      )}

      {stopped && chosen && !chosen.installed && (
        <div className="error-line">
          <code>{chosen.key}</code> is in the registry but its weights are not on disk. Download
          it first — starting would fail at the missing file.
        </div>
      )}

      {/* A server Headroom did not start, running a file that is not in the
          registry, is a legitimate state rather than an error — but nothing can
          be attributed to a registry entry while it is the case, so it is worth
          saying plainly rather than showing a blank field. */}
      {!stopped && server.model_key === null && server.model_name && (
        <div className="finding info">
          <div className="finding-detail">
            The loaded file <code>{server.model_name}</code> does not match any registry entry, so
            Headroom cannot attribute measurements to it. Add it to the registry if you want its
            figures recorded.
          </div>
        </div>
      )}

      {/* Loading is progress, not a fault, and it used to be presented as both:
          this notice styled as an error, with the raw probe failure printed
          underneath it. During a normal load that raw text is a 503 from /props
          plus a link to an HTTP status page, which reads like something broke
          at exactly the moment nothing has. */}
      {loading && (
        <div className="loading-line" role="status" aria-live="polite">
          <span className="loading-dot" aria-hidden="true" />
          <div>
            <strong>Model loading, please wait…</strong>
            <div className="loading-detail">
              The process is up and Headroom is watching for it in the background
              {server.uptime_seconds != null && ` — ${Math.round(server.uptime_seconds)}s so far`}
              . A large model takes tens of seconds, and it does not answer at all until the
              weights are resident. Don&rsquo;t start a second one.
            </div>
          </div>
        </div>
      )}

      {/* Suppressed while loading: during that window `error` is the expected
          probe failure, not a problem, and the notice above already says so.
          Kept for every other state, where it is the only explanation there is. */}
      {!loading && server.error && <div className="error-line">{server.error}</div>}
      {error && <div className="error-line">{error}</div>}
    </div>
  );
}
