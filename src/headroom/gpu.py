"""GPU telemetry.

Headroom's whole premise is that **free VRAM is the number that decides whether a
config is safe**, so this module is the core of the product rather than a
sidecar. Everything else is presentation.

Two things here are not obvious and are the reason this is a module rather than
three lines of `nvidia-smi` parsing:

1.  **NVML and llama.cpp disagree about which GPU is which.** NVML (like
    `nvidia-smi`) enumerates by PCI bus order. llama.cpp uses the CUDA runtime's
    default FASTEST_FIRST ordering. On a machine with a fast card in a later PCI
    slot these disagree, and a UI that shows one while the user types the other
    into `-dev`/`-ts` is actively harmful. `resolve_cuda_mapping()` reconciles
    them against llama.cpp's own `--list-devices` output, which is the only
    authoritative source.

2.  **Per-device headroom matters more than the total.** A model can have
    several GiB free "in aggregate" while the card that also drives the desktop
    sits at a few hundred MiB and is one browser tab away from OOMing the
    server mid-generation. `Gpu.headroom_state` grades each card on its own.

NVIDIA-only today. `GpuBackend` exists so ROCm or Metal can be added without
touching callers.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

# Below this many MiB free, anything else touching the card can OOM the server
# mid-generation. Not a hard limit -- a threshold for grading and colour.
HEADROOM_CRITICAL_MIB = 500
HEADROOM_TIGHT_MIB = 1200

# How llama.cpp is told which devices to use. Both spellings are accepted by
# llama-server, and a command line carrying neither means every visible device.
_DEVICE_FLAGS = ("-dev", "--device")

# NVML's clock-throttle bitmask. Defined here rather than imported because the
# library renamed them (`ClocksThrottleReason*` -> `ClocksEventReason*`) and a
# missing attribute on someone else's pynvml would take out the whole poll for a
# decorative field. The bit values are part of the driver ABI and stable.
THROTTLE_GPU_IDLE = 0x1
THROTTLE_APP_CLOCKS = 0x2
THROTTLE_SW_POWER_CAP = 0x4
THROTTLE_HW_SLOWDOWN = 0x8
THROTTLE_SYNC_BOOST = 0x10
THROTTLE_SW_THERMAL = 0x20
THROTTLE_HW_THERMAL = 0x40
THROTTLE_HW_POWER_BRAKE = 0x80
THROTTLE_DISPLAY_CLOCKS = 0x100

#: The card is slowing itself down because of heat. This is the one that
#: invalidates a benchmark: the number measured is the cooling, not the model.
THERMAL_THROTTLE_BITS = THROTTLE_SW_THERMAL | THROTTLE_HW_THERMAL

#: Clamped by the power budget rather than by heat. Common and often expected on
#: a stock card under sustained load, so it is reported and not alarmed about.
POWER_THROTTLE_BITS = THROTTLE_SW_POWER_CAP | THROTTLE_HW_POWER_BRAKE

# Not faults. GPU_IDLE is set on an idle card and means nothing is wrong, which
# is why a naive "throttle reasons != 0" check lights up on a machine doing
# nothing at all.
_BENIGN_BITS = THROTTLE_GPU_IDLE | THROTTLE_APP_CLOCKS | THROTTLE_DISPLAY_CLOCKS

_THROTTLE_LABELS: tuple[tuple[int, str], ...] = (
    (THROTTLE_HW_THERMAL, "hardware thermal slowdown"),
    (THROTTLE_SW_THERMAL, "software thermal slowdown"),
    (THROTTLE_HW_POWER_BRAKE, "power brake"),
    (THROTTLE_SW_POWER_CAP, "power cap"),
    (THROTTLE_HW_SLOWDOWN, "hardware slowdown"),
    (THROTTLE_SYNC_BOOST, "sync boost"),
)

# Thermal grading is done against the card's OWN slowdown threshold, never an
# absolute temperature. 83 C is comfortable on one card and throttling on
# another, and hardcoding a number is how a tool works only on its author's
# hardware -- the same failure `headroom.config` exists to avoid.
THERMAL_HOT_MARGIN_C = 5
THERMAL_WARM_MARGIN_C = 15

# Only for cards that will not report a threshold. Deliberately conservative.
FALLBACK_SLOWDOWN_C = 90


@dataclass(slots=True)
class Gpu:
    """One physical GPU, as both NVML and llama.cpp see it."""

    nvml_index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_pct: int | None = None
    power_watts: float | None = None
    temperature_c: int | None = None
    pcie_bus_id: str = ""
    pcie_link_width: int | None = None

    # Thermals. The thresholds are the card's own, read from the driver, so
    # grading works on hardware this was never tested on.
    temp_slowdown_c: int | None = None
    temp_shutdown_c: int | None = None
    fan_percent: int | None = None
    clock_sm_mhz: int | None = None
    clock_sm_max_mhz: int | None = None
    #: Raw NVML bitmask. Zero also means "not reported", which is why the
    #: properties below never treat 0 as evidence of health.
    throttle_reasons: int = 0

    # Filled in by resolve_cuda_mapping(); None until then, and None is honest
    # rather than a guess -- callers must not invent a mapping.
    cuda_index: int | None = None

    # Set by mark_vision_residency(): a multimodal projector is loaded on this
    # card right now. Not a property of the hardware, so it is filled in by the
    # caller that knows what is serving, the same way cuda_index is.
    vision_resident: bool = False

    @property
    def headroom_state(self) -> str:
        """`critical` | `tight` | `ok`, graded per-card, not against the total."""
        if self.memory_free_mib < HEADROOM_CRITICAL_MIB:
            return "critical"
        if self.memory_free_mib < HEADROOM_TIGHT_MIB:
            return "tight"
        return "ok"

    @property
    def headroom_provisional(self) -> bool:
        """Whether this card's free figure is still on its way down.

        A resident vision projector makes the number above an **upper bound**
        rather than a reading. The image buffer llama.cpp allocates is a
        retained high-water mark, not a transient: it grows the first time a
        large image is processed and is never given back for the life of the
        server. Measured on this project's development box, one 4K image took a
        card from 578 MiB free to 170 MiB and left it there.

        So a vision-loaded card sitting comfortably above the tight line has not
        finished falling, and reading its headroom as spare capacity -- starting
        a game, a diffusion run, a second model -- is how the server gets OOMed
        mid-generation by something that looked affordable at the time.

        The grade itself is deliberately NOT demoted. `ok` and `tight` still mean
        what they measure, and collapsing a 1.3 GiB card into the same bucket as
        a 600 MiB one would throw away the distinction that decides whether the
        first large image is survivable at all. What changes is that the figure
        is labelled as unfinished. A card already `critical` gets no label: there
        is no worse grade to warn about, and the warning would only dilute one
        that already says everything.
        """
        return self.vision_resident and self.headroom_state != "critical"

    @property
    def throttling_thermally(self) -> bool:
        """The card is slowing itself down because of heat, right now."""
        return bool(self.throttle_reasons & THERMAL_THROTTLE_BITS)

    @property
    def throttling_for_power(self) -> bool:
        """Clamped by its power budget. Normal under sustained load."""
        return bool(self.throttle_reasons & POWER_THROTTLE_BITS)

    @property
    def throttle_labels(self) -> tuple[str, ...]:
        """Active throttle reasons worth naming, benign ones excluded.

        `GpuIdle` is set on a card doing nothing, so a bare "reasons != 0" check
        reports a problem on an idle machine.
        """
        active = self.throttle_reasons & ~_BENIGN_BITS
        return tuple(label for bit, label in _THROTTLE_LABELS if active & bit)

    @property
    def thermal_headroom_c(self) -> int | None:
        """Degrees left before the card starts slowing itself down."""
        if self.temperature_c is None:
            return None
        return (self.temp_slowdown_c or FALLBACK_SLOWDOWN_C) - self.temperature_c

    @property
    def thermal_state(self) -> str:
        """`ok` | `warm` | `hot` | `throttling` | `unknown`.

        Graded against this card's own slowdown threshold rather than a fixed
        temperature, because the thresholds genuinely differ -- 95 C on one card
        here and 96 C on the other, with GPU_MAX at 93 and 90. A single hardcoded
        limit would be wrong on both.

        `throttling` outranks temperature: a card that has already been clamped
        is past the point where the reading is the interesting fact.
        """
        if self.throttling_thermally:
            return "throttling"
        if self.temperature_c is None:
            return "unknown"
        slowdown = self.temp_slowdown_c or FALLBACK_SLOWDOWN_C
        if self.temperature_c >= slowdown - THERMAL_HOT_MARGIN_C:
            return "hot"
        if self.temperature_c >= slowdown - THERMAL_WARM_MARGIN_C:
            return "warm"
        return "ok"

    @property
    def label(self) -> str:
        """How to name this card in the UI, without inventing a CUDA index."""
        if self.cuda_index is None:
            return f"{self.name} (nvml {self.nvml_index})"
        return f"{self.name} (CUDA{self.cuda_index})"


def devices_in_use(command_line: list[str]) -> list[int] | None:
    """CUDA indices named by `-dev`, or None when the command line names none.

    None is not an empty list. A server started without the flag uses every
    visible device, so "nothing named" and "nothing used" are opposite claims
    and must not share a representation.
    """
    for i, arg in enumerate(command_line):
        if arg in _DEVICE_FLAGS and i + 1 < len(command_line):
            value = command_line[i + 1]
        elif any(arg.startswith(f"{flag}=") for flag in _DEVICE_FLAGS):
            value = arg.split("=", 1)[1]
        else:
            continue
        indices = []
        for part in value.split(","):
            part = part.strip()
            if part.upper().startswith("CUDA") and part[4:].isdigit():
                indices.append(int(part[4:]))
        return indices or None
    return None


def mark_vision_residency(
    cards: list[Gpu], *, vision: bool, command_line: list[str] | None = None
) -> None:
    """Flag the cards a multimodal projector is currently loaded on.

    When the server names its devices and every named one resolves to a card,
    exactly those are flagged. Otherwise **every** card is, for two reasons that
    point the same way: a llama-server started without `-dev` really does use all
    of them, and an unresolved CUDA mapping means Headroom cannot rule any card
    out. Under-warning here costs someone a server mid-generation; over-warning
    costs them a label.
    """
    for card in cards:
        card.vision_resident = False
    if not vision:
        return

    named = devices_in_use(command_line or [])
    if named is not None:
        by_cuda = {c.cuda_index: c for c in cards if c.cuda_index is not None}
        if all(index in by_cuda for index in named):
            for index in named:
                by_cuda[index].vision_resident = True
            return

    for card in cards:
        card.vision_resident = True


class GpuBackend(Protocol):
    """So a ROCm or Metal backend can be added without changing callers."""

    def available(self) -> bool: ...
    def poll(self) -> list[Gpu]: ...
    def shutdown(self) -> None: ...


class NvmlBackend:
    """NVIDIA telemetry via NVML.

    Uses the library directly rather than shelling out to `nvidia-smi`: a
    subprocess per poll at 1 Hz is wasteful, and parsing its text output breaks
    silently across driver versions.
    """

    def __init__(self) -> None:
        self._nvml = None
        self._handles: list = []
        self._init_failed_reason: str | None = None
        # Temperature thresholds are properties of the card, not of the moment.
        # Read once and reused, the same reasoning as the CUDA mapping cache.
        self._thresholds: dict[int, tuple[int | None, int | None]] = {}

    def available(self) -> bool:
        return self._ensure_init()

    def _ensure_init(self) -> bool:
        if self._nvml is not None:
            return True
        if self._init_failed_reason is not None:
            return False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            count = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            log.info("NVML initialised, %d device(s)", count)
            return True
        except Exception as exc:  # noqa: BLE001 - any failure here means "no NVIDIA telemetry"
            self._init_failed_reason = str(exc)
            log.warning("NVML unavailable: %s", exc)
            return False

    def poll(self) -> list[Gpu]:
        if not self._ensure_init():
            return []
        n = self._nvml
        out: list[Gpu] = []

        for idx, handle in enumerate(self._handles):
            mem = n.nvmlDeviceGetMemoryInfo(handle)

            # Every field below the memory numbers is best-effort. Consumer cards,
            # WSL and older drivers omit some of them, and a missing power reading
            # must not take down the whole telemetry stream.
            util = _try(lambda h=handle: n.nvmlDeviceGetUtilizationRates(h).gpu)
            power = _try(lambda h=handle: n.nvmlDeviceGetPowerUsage(h) / 1000.0)
            temp = _try(lambda h=handle: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU))
            width = _try(lambda h=handle: n.nvmlDeviceGetCurrPcieLinkWidth(h))
            fan = _try(lambda h=handle: n.nvmlDeviceGetFanSpeed(h))
            clock = _try(lambda h=handle: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM))
            clock_max = _try(lambda h=handle: n.nvmlDeviceGetMaxClockInfo(h, n.NVML_CLOCK_SM))

            # Renamed across pynvml versions; try both rather than pin a version
            # for one field. Absent on some virtualised and WSL setups, where 0
            # simply means "not reported" -- see Gpu.throttle_labels.
            throttle = 0
            for fn_name in (
                "nvmlDeviceGetCurrentClocksEventReasons",
                "nvmlDeviceGetCurrentClocksThrottleReasons",
            ):
                fn = getattr(n, fn_name, None)
                if fn is not None:
                    throttle = _try(lambda f=fn, h=handle: f(h)) or 0
                    break

            if idx not in self._thresholds:
                self._thresholds[idx] = (
                    _try(
                        lambda h=handle: n.nvmlDeviceGetTemperatureThreshold(
                            h, n.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN
                        )
                    ),
                    _try(
                        lambda h=handle: n.nvmlDeviceGetTemperatureThreshold(
                            h, n.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN
                        )
                    ),
                )
            slowdown_c, shutdown_c = self._thresholds[idx]

            name = n.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()

            bus_id = _try(lambda h=handle: n.nvmlDeviceGetPciInfo(h).busId) or ""
            if isinstance(bus_id, bytes):
                bus_id = bus_id.decode()

            out.append(
                Gpu(
                    nvml_index=idx,
                    name=name,
                    memory_total_mib=mem.total // 1024 // 1024,
                    memory_used_mib=mem.used // 1024 // 1024,
                    memory_free_mib=mem.free // 1024 // 1024,
                    utilization_pct=util,
                    power_watts=round(power, 1) if power is not None else None,
                    temperature_c=temp,
                    pcie_bus_id=bus_id,
                    pcie_link_width=width,
                    temp_slowdown_c=slowdown_c,
                    temp_shutdown_c=shutdown_c,
                    fan_percent=fan,
                    clock_sm_mhz=clock,
                    clock_sm_max_mhz=clock_max,
                    throttle_reasons=throttle,
                )
            )
        return out

    def shutdown(self) -> None:
        if self._nvml is not None:
            _try(self._nvml.nvmlShutdown)
            self._nvml = None
            self._handles = []


def _try(fn):
    """Call fn, returning None on any failure.

    Telemetry is decorative for most fields; one unsupported query must never
    break the poll.
    """
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# CUDA index mapping
# --------------------------------------------------------------------------

_DEVICE_LINE = re.compile(
    r"^\s*(?P<dev>CUDA(?P<idx>\d+)):\s*(?P<name>.+?)\s*\(\s*(?P<total>\d+)\s*MiB"
    r"(?:\s*,\s*(?P<free>\d+)\s*MiB\s*free)?",
    re.MULTILINE,
)

# Two candidate cards are only told apart by free VRAM if their free figures
# differ by more than this. The reading from `--list-devices` and the reading
# from NVML are taken moments apart, and desktop compositing alone moves free
# memory by tens of MiB, so a near-tie is a tie.
AMBIGUOUS_FREE_MIB = 256


@dataclass(slots=True)
class CudaMapping:
    """Result of reconciling llama.cpp's device order with NVML's."""

    cuda_to_nvml: dict[int, int] = field(default_factory=dict)
    raw_output: str = ""
    source: str = "unresolved"
    warning: str | None = None
    #: CUDA indices whose physical card could not be established -- identical
    #: models with indistinguishable free VRAM. The mapping still contains an
    #: entry for them, because a partial mapping is worse than a guessed one,
    #: but that entry is a guess and callers must be able to know it.
    ambiguous: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return bool(self.cuda_to_nvml)

    @property
    def trustworthy(self) -> bool:
        """Resolved *and* every device pinned to a specific card.

        `resolved` alone answers "did anything come back", which on a rig of
        identical cards was true while the mapping was invented.
        """
        return self.resolved and not self.ambiguous


def resolve_cuda_mapping(llama_server_exe: str | Path, gpus: list[Gpu]) -> CudaMapping:
    """Ask llama.cpp which device it calls CUDA0, and match those to NVML indices.

    This exists because guessing is wrong often enough to matter. NVML orders by
    PCI bus; llama.cpp orders FASTEST_FIRST. A machine with the faster card in a
    higher-numbered slot gets the two orderings reversed, so `-dev CUDA0` and
    `nvidia-smi -i 0` refer to *different cards*. Anyone reading a dashboard that
    conflates them will tune the wrong device.

    Cache the result. It only changes when hardware or drivers change, and this
    spawns a subprocess.
    """
    exe = Path(llama_server_exe)
    if not exe.exists():
        return CudaMapping(warning=f"llama-server not found at {exe}; CUDA indices unresolved")

    try:
        proc = subprocess.run(
            [str(exe), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return CudaMapping(warning=f"could not run --list-devices: {exc}")

    text = (proc.stdout or "") + (proc.stderr or "")
    matches = list(_DEVICE_LINE.finditer(text))
    if not matches:
        return CudaMapping(
            raw_output=text[:2000],
            warning="--list-devices produced no parseable device lines",
        )

    # Match on (name, total memory), then on free memory where that is not
    # enough. Name and total are identical across a rig of identical cards, and
    # the first version took the first still-unclaimed card in NVML order --
    # producing an identity mapping that looked resolved and was invented.
    #
    # That is the worst available outcome on exactly the machines this function
    # exists for. Four RTX 3090s where llama.cpp enumerates them in a different
    # order than NVML would report `order_differs: False`, no warning, and every
    # per-card figure attributed to the wrong physical card.
    #
    # `--list-devices` prints free VRAM per device, and identical cards almost
    # never have identical free VRAM -- one drives a display, one holds a model.
    # So free memory is the tiebreak, and where even that cannot separate them
    # the mapping says so instead of guessing quietly.
    remaining = list(enumerate(gpus))
    mapping: dict[int, int] = {}
    unmatched: list[str] = []
    ambiguous: list[str] = []

    for m in matches:
        cuda_idx = int(m.group("idx"))
        name = m.group("name").strip()
        total = int(m.group("total"))
        free = int(m.group("free")) if m.group("free") else None

        candidates = [
            pos
            for pos, (_, gpu) in enumerate(remaining)
            if gpu.name.strip() == name and abs(gpu.memory_total_mib - total) <= 64
        ]
        if not candidates:
            candidates = [
                pos
                for pos, (_, gpu) in enumerate(remaining)
                if name in gpu.name or gpu.name in name
            ]
        if not candidates:
            unmatched.append(f"CUDA{cuda_idx}={name}")
            continue

        if len(candidates) == 1:
            hit = candidates[0]
        elif free is None:
            # Nothing left to separate them by. Take one, and say it is a guess.
            hit = candidates[0]
            ambiguous.append(f"CUDA{cuda_idx}")
        else:
            by_free = sorted(
                candidates, key=lambda pos: abs(remaining[pos][1].memory_free_mib - free)
            )
            hit = by_free[0]
            best = abs(remaining[by_free[0]][1].memory_free_mib - free)
            runner_up = abs(remaining[by_free[1]][1].memory_free_mib - free)
            if runner_up - best < AMBIGUOUS_FREE_MIB:
                ambiguous.append(f"CUDA{cuda_idx}")

        _, gpu = remaining.pop(hit)
        mapping[cuda_idx] = gpu.nvml_index
        gpu.cuda_index = cuda_idx

    notes = []
    if unmatched:
        notes.append("could not match to NVML: " + ", ".join(unmatched))
    if ambiguous:
        notes.append(
            f"{', '.join(ambiguous)} could not be pinned to a specific card: identical models "
            "with free VRAM too close to tell apart. Those CUDA indices are a guess, so do not "
            "trust per-card figures for them -- load one card (a model, or a full-screen "
            "window) and re-check to separate them."
        )

    return CudaMapping(
        cuda_to_nvml=mapping,
        raw_output=text[:2000],
        source="llama-server --list-devices",
        warning=" ".join(notes) if notes else None,
        ambiguous=tuple(ambiguous),
    )


def order_differs(mapping: CudaMapping) -> bool:
    """True when CUDA order and NVML order disagree.

    Worth surfacing prominently in the UI: it means `nvidia-smi -i N` and
    `-dev CUDAN` point at different cards on this machine, which is the single
    most expensive misunderstanding available here.
    """
    return any(cuda != nvml for cuda, nvml in mapping.cuda_to_nvml.items())
