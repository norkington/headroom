import type { Gpu } from "../api";
import { formatMiB } from "../api";

/**
 * One GPU.
 *
 * Free VRAM is the headline figure, not used VRAM. "10.1 GiB used of 12" needs
 * arithmetic before it means anything; "1012 MiB free" is the decision directly.
 *
 * State is encoded three ways on purpose — a severity stripe, a text pill, and
 * colour — so it survives both a glance and colour-blindness.
 */
export function GpuCard({ gpu }: { gpu: Gpu }) {
  const usedPct = gpu.memory_total_mib
    ? (gpu.memory_used_mib / gpu.memory_total_mib) * 100
    : 0;

  const free = formatMiB(gpu.memory_free_mib);
  const [value, unit] = free.split(" ");

  return (
    <article className={`gpu ${gpu.headroom_state} state-${gpu.headroom_state}`}>
      <div className="gpu-head">
        <div>
          <div className="gpu-name">{gpu.name}</div>
        </div>
        {gpu.cuda_index !== null ? (
          <span className="cuda-tag" title="What llama.cpp calls this device">
            CUDA{gpu.cuda_index}
          </span>
        ) : (
          <span className="cuda-tag" title="llama.cpp device index could not be resolved">
            nvml {gpu.nvml_index}
          </span>
        )}
      </div>

      <div className="free-figure">
        <span className="value">{value}</span>
        <span className="unit">{unit} free</span>
      </div>
      <span className="state-pill">{gpu.headroom_state}</span>

      <div
        className={`meter ${gpu.headroom_state}`}
        role="img"
        aria-label={`${Math.round(usedPct)} percent of memory in use`}
      >
        <div className="used" style={{ width: `${usedPct}%` }} />
      </div>

      <div className="gpu-stats">
        <div>
          <span className="k">used</span>
          {formatMiB(gpu.memory_used_mib)}
        </div>
        <div>
          <span className="k">util</span>
          {gpu.utilization_pct ?? "—"}
          {gpu.utilization_pct !== null && "%"}
        </div>
        <div>
          <span className="k">power</span>
          {gpu.power_watts ?? "—"}
          {gpu.power_watts !== null && "W"}
        </div>
        <div>
          <span className="k">temp</span>
          {gpu.temperature_c ?? "—"}
          {gpu.temperature_c !== null && "°"}
        </div>
      </div>
    </article>
  );
}
