import type { ModelSummary } from "../api";

/**
 * The registry, as the UI sees it.
 *
 * The one non-obvious decision here: a model whose performance figures were
 * *inherited* from a sibling build is labelled as such, rather than displaying
 * those numbers identically to measured ones. Presenting an inherited figure as
 * though it were measured is the kind of quiet dishonesty that makes an entire
 * dashboard untrustworthy — once you find one number that was not what it
 * claimed, you stop believing any of them.
 */
export function ModelList({ models }: { models: ModelSummary[] }) {
  if (models.length === 0) {
    return <div className="empty">No models in the registry.</div>;
  }

  return (
    <>
      {models.map((m) => {
        const decode = m.measured["decode_tok_s"];
        const prefill = m.measured["prefill_tok_s"];
        const vram = m.measured["vram_free_mib_at_64k"];
        const needle = m.verified["needle_score"];
        const hasNumbers = decode != null || prefill != null;

        return (
          <article className="model" key={m.key}>
            <div className="model-head">
              <div>
                <span className="model-name">{m.label}</span>
                <span className={`tag ${m.installed ? "installed" : "missing"}`}>
                  {m.installed ? "installed" : "not installed"}
                </span>
                {m.vision_supported && <span className="tag">vision</span>}
                {m.uncensored && <span className="tag">uncensored</span>}
                {hasNumbers && !m.measured_on_this_file && (
                  <span
                    className="tag inherited"
                    title="These figures came from a sibling build, not this file"
                  >
                    inherited numbers
                  </span>
                )}
              </div>
              <div className="model-meta">
                {m.size_gib.toFixed(2)} GiB · {m.arch}
                {m.license ? ` · ${m.license}` : ""}
              </div>
            </div>

            <div className="model-meta">{m.repo}</div>

            {hasNumbers && (
              <div className="measured-row">
                <div>
                  <span className="k">decode</span>
                  {decode != null ? `${String(decode)} tok/s` : "—"}
                </div>
                <div>
                  <span className="k">prefill</span>
                  {prefill != null ? `${String(prefill)} tok/s` : "—"}
                </div>
                <div>
                  <span className="k">free vram</span>
                  {vram != null ? `${String(vram)} MiB` : "—"}
                </div>
                <div>
                  <span className="k">needle</span>
                  {needle != null ? String(needle) : "—"}
                </div>
              </div>
            )}
          </article>
        );
      })}
    </>
  );
}
