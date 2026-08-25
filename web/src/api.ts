/**
 * Backend types and data hooks.
 *
 * The telemetry stream is the heart of this app, so its failure modes get more
 * care than the shape of the data. A monitoring dashboard that silently freezes
 * on a dead socket is worse than one that plainly says it lost connection: the
 * numbers stay on screen looking authoritative while being minutes stale, and
 * you make decisions on them.
 */

export type HeadroomState = "ok" | "tight" | "critical";
export type ServerStatus = "running" | "loading" | "stopped" | "orphaned";

export interface Gpu {
  nvml_index: number;
  cuda_index: number | null;
  name: string;
  label: string;
  memory_total_mib: number;
  memory_used_mib: number;
  memory_free_mib: number;
  headroom_state: HeadroomState;
  utilization_pct: number | null;
  power_watts: number | null;
  temperature_c: number | null;
}

export interface ServerInfo {
  status: ServerStatus;
  pid: number | null;
  model_name: string | null;
  n_ctx: number | null;
  vision: boolean;
  host_ram_mib: number | null;
  uptime_seconds: number | null;
  error: string | null;
}

export interface Telemetry {
  gpus: Gpu[];
  server: ServerInfo;
}

export interface CudaMapping {
  cuda_to_nvml: Record<string, number>;
  resolved: boolean;
  source: string;
  warning: string | null;
  /** True when nvidia-smi's numbering and llama.cpp's name different cards. */
  order_differs: boolean;
}

export interface ModelSummary {
  key: string;
  label: string;
  repo: string;
  file: string;
  size_gib: number;
  arch: string;
  installed: boolean;
  path: string;
  uncensored: boolean;
  license: string | null;
  vision_supported: boolean;
  why_this_build: string[];
  serve: Record<string, unknown>;
  measured: Record<string, unknown>;
  verified: Record<string, unknown>;
  /** Whether the numbers were measured on this artifact or inherited. */
  measured_on_this_file: boolean;
}

export interface Health {
  ok: boolean;
  version: string;
  gpu_backend_available: boolean;
  registry: string;
  registry_exists: boolean;
  llama_server: string;
  llama_server_exists: boolean;
}

export async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* response was not JSON; the status text is the best we have */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function postJSON<T>(path: string): Promise<T> {
  const res = await fetch(path, { method: "POST" });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* as above */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export function formatMiB(mib: number): string {
  if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GiB`;
  return `${mib} MiB`;
}

export function formatUptime(seconds: number | null): string {
  if (seconds == null) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}
