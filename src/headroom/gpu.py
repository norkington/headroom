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

    # Filled in by resolve_cuda_mapping(); None until then, and None is honest
    # rather than a guess -- callers must not invent a mapping.
    cuda_index: int | None = None

    @property
    def headroom_state(self) -> str:
        """`critical` | `tight` | `ok`, graded per-card, not against the total."""
        if self.memory_free_mib < HEADROOM_CRITICAL_MIB:
            return "critical"
        if self.memory_free_mib < HEADROOM_TIGHT_MIB:
            return "tight"
        return "ok"

    @property
    def label(self) -> str:
        """How to name this card in the UI, without inventing a CUDA index."""
        if self.cuda_index is None:
            return f"{self.name} (nvml {self.nvml_index})"
        return f"{self.name} (CUDA{self.cuda_index})"


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
    r"^\s*(?P<dev>CUDA(?P<idx>\d+)):\s*(?P<name>.+?)\s*\(\s*(?P<total>\d+)\s*MiB",
    re.MULTILINE,
)


@dataclass(slots=True)
class CudaMapping:
    """Result of reconciling llama.cpp's device order with NVML's."""

    cuda_to_nvml: dict[int, int] = field(default_factory=dict)
    raw_output: str = ""
    source: str = "unresolved"
    warning: str | None = None

    @property
    def resolved(self) -> bool:
        return bool(self.cuda_to_nvml)


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

    # Match on (name, total memory). Name alone is ambiguous on identical-card
    # rigs; adding total VRAM disambiguates most of them. Where two cards are
    # genuinely indistinguishable the mapping is arbitrary but also harmless,
    # since the cards are interchangeable.
    remaining = list(enumerate(gpus))
    mapping: dict[int, int] = {}
    unmatched: list[str] = []

    for m in matches:
        cuda_idx = int(m.group("idx"))
        name = m.group("name").strip()
        total = int(m.group("total"))

        hit = None
        for pos, (_, gpu) in enumerate(remaining):
            if gpu.name.strip() == name and abs(gpu.memory_total_mib - total) <= 64:
                hit = pos
                break
        if hit is None:
            for pos, (_, gpu) in enumerate(remaining):
                if name in gpu.name or gpu.name in name:
                    hit = pos
                    break
        if hit is None:
            unmatched.append(f"CUDA{cuda_idx}={name}")
            continue

        _, gpu = remaining.pop(hit)
        mapping[cuda_idx] = gpu.nvml_index
        gpu.cuda_index = cuda_idx

    warning = None
    if unmatched:
        warning = "could not match to NVML: " + ", ".join(unmatched)

    return CudaMapping(
        cuda_to_nvml=mapping,
        raw_output=text[:2000],
        source="llama-server --list-devices",
        warning=warning,
    )


def order_differs(mapping: CudaMapping) -> bool:
    """True when CUDA order and NVML order disagree.

    Worth surfacing prominently in the UI: it means `nvidia-smi -i N` and
    `-dev CUDAN` point at different cards on this machine, which is the single
    most expensive misunderstanding available here.
    """
    return any(cuda != nvml for cuda, nvml in mapping.cuda_to_nvml.items())
