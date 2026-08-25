import { useState } from "react";
import { getJSON, postJSON } from "../api";

/**
 * Inspect a quant before downloading it.
 *
 * This is the panel Headroom exists for. Two builds of the same model at the
 * same file size can differ a great deal in quality, and nothing on a model
 * page tells you which you are looking at — only the tensor table does, and it
 * sits at the front of the file, so a few megabytes answers the question that a
 * 15 GiB download would otherwise be needed to settle.
 *
 * The findings are shown above the raw tensor counts on purpose. The counts are
 * evidence; the findings are the answer to "should I download this", and burying
 * that under a table would make the user do the interpretation the app is for.
 */

interface Finding {
  level: "good" | "caution" | "info";
  title: string;
  detail: string;
}

interface ProbeResult {
  source: string;
  architecture: string;
  name: string;
  tensor_count: number;
  size_gib: number | null;
  bytes_read: number;
  has_mtp: boolean;
  mtp_tensor_count: number;
  families: Record<string, Record<string, number>>;
  findings: Finding[];
  free_vram_mib: number | null;
}

interface RepoFile {
  filename: string;
  size_gib: number;
}

export function ProbePanel({ onAdded }: { onAdded?: () => void }) {
  const [repo, setRepo] = useState("");
  const [files, setFiles] = useState<RepoFile[]>([]);
  const [selected, setSelected] = useState("");
  const [result, setResult] = useState<ProbeResult | null>(null);
  const [busy, setBusy] = useState<null | "list" | "probe" | "add">(null);
  const [error, setError] = useState<string | null>(null);
  const [added, setAdded] = useState<string | null>(null);
  const [key, setKey] = useState("");

  const listFiles = async () => {
    if (!repo.trim()) return;
    setBusy("list");
    setError(null);
    setFiles([]);
    setResult(null);
    setSelected("");
    try {
      const d = await getJSON<{ files: RepoFile[] }>(
        `/api/hf/files?repo=${encodeURIComponent(repo.trim())}`,
      );
      setFiles(d.files);
      if (d.files.length === 0) setError("No .gguf files in that repository.");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const probe = async (filename: string) => {
    setBusy("probe");
    setError(null);
    setAdded(null);
    setSelected(filename);
    // Suggest a registry key from the filename: lowercase, no extension, and
    // punctuation collapsed to hyphens so it is usable as a command argument.
    setKey(
      filename
        .replace(/\.gguf$/i, "")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 48),
    );
    try {
      const d = await getJSON<ProbeResult>(
        `/api/probe?repo=${encodeURIComponent(repo.trim())}&file=${encodeURIComponent(filename)}`,
      );
      setResult(d);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setBusy(null);
    }
  };

  const addToRegistry = async () => {
    setBusy("add");
    setError(null);
    try {
      const params = new URLSearchParams({
        key: key.trim(),
        repo: repo.trim(),
        file: selected,
        download: "true",
      });
      const d = await postJSON<{ added: string }>(`/api/registry/add?${params}`);
      setAdded(d.added);
      onAdded?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="panel">
      <div className="probe-form">
        <input
          type="text"
          value={repo}
          placeholder="huggingface repo, e.g. owner/model-GGUF"
          onChange={(e) => setRepo(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void listFiles();
          }}
          aria-label="HuggingFace repository"
        />
        <button className="primary" disabled={busy !== null || !repo.trim()} onClick={() => void listFiles()}>
          {busy === "list" ? "Looking…" : "List quants"}
        </button>
      </div>

      {error && <div className="error-line">{error}</div>}

      {files.length > 0 && (
        <div className="quant-list">
          {files.map((f) => (
            <button
              key={f.filename}
              className={`quant ${selected === f.filename ? "selected" : ""}`}
              disabled={busy !== null}
              onClick={() => void probe(f.filename)}
              title="Read this file's tensor table without downloading it"
            >
              <span className="qname">{f.filename}</span>
              <span className="qsize">{f.size_gib.toFixed(2)} GiB</span>
            </button>
          ))}
        </div>
      )}

      {busy === "probe" && <div className="probe-busy">Reading the tensor table…</div>}

      {result && (
        <div className="probe-result">
          <div className="probe-summary">
            <div>
              <span className="k">architecture</span>
              {result.architecture || "—"}
            </div>
            <div>
              <span className="k">tensors</span>
              {result.tensor_count.toLocaleString()}
            </div>
            <div>
              <span className="k">size</span>
              {result.size_gib ? `${result.size_gib.toFixed(2)} GiB` : "—"}
            </div>
            <div>
              <span className="k">downloaded</span>
              {(result.bytes_read / 1048576).toFixed(1)} MiB
            </div>
          </div>

          {result.findings.map((f, i) => (
            <div className={`finding ${f.level}`} key={i}>
              <div className="finding-title">{f.title}</div>
              <div className="finding-detail">{f.detail}</div>
            </div>
          ))}

          <div className="add-row">
            <input
              type="text"
              value={key}
              placeholder="registry key"
              onChange={(e) => setKey(e.target.value)}
              aria-label="Registry key for this model"
            />
            <button
              disabled={busy !== null || !key.trim() || added !== null}
              onClick={() => void addToRegistry()}
            >
              {busy === "add" ? "Adding…" : added ? "Added" : "Add to registry & download"}
            </button>
          </div>
          {added && (
            <div className="finding good">
              <div className="finding-title">Added as {added}</div>
              <div className="finding-detail">
                Downloading now &mdash; see below. The entry is marked{" "}
                <strong>not measured</strong>: its settings are defaults or inherited, never
                observed on this file. Benchmark it before trusting the numbers.
              </div>
            </div>
          )}

          <div className="families">
            {Object.entries(result.families).map(([family, counts]) => (
              <div className="family" key={family}>
                <span className="k">{family.replace("_", " ")}</span>
                <span className="family-counts">
                  {Object.entries(counts)
                    .map(([dtype, n]) => `${n}×${dtype}`)
                    .join("  ")}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
