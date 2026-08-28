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
 *
 * A card holding a vision projector gets a second pill. Its free figure is an
 * upper bound rather than a reading: llama.cpp's image buffer is a retained
 * high-water mark, so the first large image takes several hundred MiB more and
 * never gives them back. The grade is not demoted for it — `ok` still means
 * what it measures, and flattening a 1.3 GiB card into the same bucket as a
 * 600 MiB one would lose the distinction that decides whether that first image
 * is survivable — but the number is labelled as unfinished, because reading it
 * as spare capacity is how the server gets OOMed by something that looked
 * affordable at the time.
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
      <div className="state-pills">
        <span className="state-pill">{gpu.headroom_state}</span>
        {gpu.headroom_provisional && (
          <span
            className="state-pill provisional"
            title="A vision projector is loaded on this card. llama.cpp's image buffer is a retained high-water mark, not a transient, so this figure is still on its way down."
          >
            provisional
          </span>
        )}
      </div>

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
          <span className={`thermal-${gpu.thermal_state}`}>
            {gpu.temperature_c ?? "—"}
            {gpu.temperature_c !== null && "°"}
          </span>
          {/* The margin, not just the reading. 71 C means nothing without
              knowing this card slows itself down at 96. */}
          {gpu.thermal_headroom_c !== null && gpu.temp_slowdown_c !== null && (
            <span className="fig-sub">{gpu.thermal_headroom_c}° to slowdown</span>
          )}
        </div>
      </div>

      {/* Throttling outranks the temperature: a card that has already been
          clamped is past the point where the reading is the interesting fact,
          and any throughput measured while it lasts describes the cooling. */}
      {gpu.throttling_thermally && (
        <div className="gpu-thermal throttling">
          <strong>Thermally throttling.</strong> This card is slowing itself down to shed heat, so
          it is not delivering the throughput its free VRAM implies. Benchmarks taken now measure
          the cooling as much as the model.
          {gpu.throttle_labels.length > 0 && ` (${gpu.throttle_labels.join(", ")})`}
        </div>
      )}

      {!gpu.throttling_thermally && gpu.thermal_state === "hot" && (
        <div className="gpu-thermal hot">
          <strong>{gpu.thermal_headroom_c}° from slowdown.</strong> Close enough that a sustained
          load will likely start clamping. Worth checking airflow before trusting a long benchmark.
        </div>
      )}

      {/* Power capping is normal on a stock card under sustained load, so it is
          reported without alarm -- but it does explain a throughput figure that
          looks low for the hardware. */}
      {!gpu.throttling_thermally && gpu.throttling_for_power && (
        <div className="gpu-thermal power">
          Power-capped at its board limit. Normal under sustained load, and it caps clocks — worth
          knowing if throughput reads lower than expected.
        </div>
      )}

      {gpu.headroom_provisional && (
        <div className="gpu-provisional">
          Projector resident — the first large image takes several hundred MiB more and never
          gives them back. Do not spend this.
        </div>
      )}
    </article>
  );
}
