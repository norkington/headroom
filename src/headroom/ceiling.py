"""Finding the largest context this machine can actually hold.

The registry records a context length; nothing verifies it. On this project's
development box the recorded 64K was arrived at by hand, and the note beside it
says 96K OOMed while creating the speculative-decoding draft context and 80K
left about 1.2 GiB free — three data points bought with three manual reloads,
for one model, on one machine. Anyone else starts from nothing.

**The answer is measured, not modelled.** KV cache size per token is arithmetic
and the arithmetic is close, but it is not the whole allocation: compute buffers,
the draft context and the graph all scale too, and the failure they produce is a
server that exits during startup rather than a number that comes out slightly
high. So each candidate is tried for real — started, waited for, measured,
stopped — and the search learns the slope from its own probes rather than
trusting a formula.

What "largest" means here is deliberately not "largest that loads". A context
that fits with 200 MiB to spare is one browser tab from an OOM mid-generation,
which is precisely the failure this project exists to prevent. The target is the
largest context that still leaves every card above the margin, and the margin
defaults to the same threshold the GPU panel already grades against — so the
answer agrees with the colour the rest of the app shows.

Nothing is written to the registry. The search reports what it found and what it
cost; deciding that a number is now your configuration is a separate, deliberate
act, the same as for a benchmark.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)

#: Contexts are searched and reported on this grid. llama.cpp is happy with any
#: value, but a ceiling reported as 63,488 invites someone to wonder whether
#: 63,489 would also have worked; the answer is "yes, and it does not matter".
GRANULARITY = 1024

#: Stop when the bracket is this tight. Two probes to resolve the last 2k of
#: context is minutes of loading for a distinction nothing can act on.
SETTLE_TOKENS = 2048

#: A hard stop on probing. Each probe is a full model load, and an unbounded
#: search on a machine with lots of VRAM would run for a very long time.
MAX_PROBES = 8

#: Never propose less than this. Below it the model is not useful and the search
#: has clearly failed at something other than context.
MIN_CTX = 2048


class CeilingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Probe:
    """One context length, tried for real."""

    ctx: int
    loaded: bool
    #: Free VRAM on the *tightest* card once the model was resident. The tightest
    #: card is what binds: a total would hide the one card that is nearly out.
    free_mib: int | None = None
    breakdown: str | None = None
    #: True when it loaded AND left every card above the margin.
    within_margin: bool = False
    seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ctx": self.ctx,
            "loaded": self.loaded,
            "free_mib": self.free_mib,
            "breakdown": self.breakdown,
            "within_margin": self.within_margin,
            "seconds": round(self.seconds, 1),
            "error": self.error,
        }


def _round_to_grid(ctx: int) -> int:
    return max(MIN_CTX, (ctx // GRANULARITY) * GRANULARITY)


def measured_mib_per_token(probes: list[Probe]) -> float | None:
    """MiB of the tightest card consumed per token of context, from the probes.

    Measured rather than taken from the registry's `kv_bytes_per_token`, for two
    reasons. A new model has no recorded value at all, and the recorded one
    describes the KV cache alone while what actually binds is every allocation
    that scales with context. Using the observed slope means the search is
    calibrated to the thing it is steering.

    Uses the widest pair of successful probes available, because a slope taken
    from two nearby contexts is mostly noise.
    """
    usable = sorted((p for p in probes if p.loaded and p.free_mib is not None), key=lambda p: p.ctx)
    if len(usable) < 2:
        return None
    low, high = usable[0], usable[-1]
    if high.ctx == low.ctx:
        return None
    # free falls as ctx rises, so this is positive.
    slope = (low.free_mib - high.free_mib) / (high.ctx - low.ctx)
    return slope if slope > 0 else None


def plan_next_context(
    probes: list[Probe],
    *,
    margin_mib: int,
    max_ctx: int,
    hint_mib_per_token: float | None = None,
    max_probes: int = MAX_PROBES,
) -> int | None:
    """The next context worth trying, or None when the answer is settled.

    Pure, so the whole search strategy is testable without loading a model --
    which matters when a single integration run costs several minutes of GPU.

    Two things keep the probe count down, and both were added after watching a
    naive version take seven loads to answer a question worth two:

    **The registry's `kv_bytes_per_token` seeds the slope.** With one successful
    probe there is nothing to measure a slope from, and the obvious fallback --
    double the context -- throws a whole load away discovering that twice was
    far too much. A recorded cost per token turns that first step into a
    prediction, and the search typically lands on the answer immediately. It is
    only a seed: as soon as two real probes exist their measured slope replaces
    it, because the recorded figure covers the KV cache while what actually
    binds is every allocation that scales with context.

    **The recorded figure is also a TOTAL, while the slope here is per-card.**
    Measured on the development box: `kv_bytes_per_token` says 65,536 bytes
    (0.0625 MiB) and the observed slope on the tightest card was 0.0322 -- very
    nearly half, because the KV cache is split across two GPUs and the tightest
    one holds only its share. So the seed overestimates the per-card cost on any
    multi-card split, which makes the first predicted step SMALLER than it needs
    to be. That is the safe direction to be wrong in: it undershoots and probes
    again rather than overshooting into a load that fails.

    **Settling accounts for the margin, not just the bracket.** A bracket can
    still be thousands of tokens wide while the best probe already sits within
    one grid step of the margin -- every remaining candidate would breach it, so
    there is nothing left to learn and the loads would be spent proving it.
    """
    if len(probes) >= max_probes:
        return None

    good = [p.ctx for p in probes if p.within_margin]
    bad = [p.ctx for p in probes if not p.within_margin]
    best_ok = max(good) if good else None
    worst_bad = min(bad) if bad else None

    # Nothing tried yet: the caller's starting point is its own decision.
    if best_ok is None and worst_bad is None:
        return None

    if best_ok is None:
        # Everything tried has failed. Halve below the smallest failure.
        candidate = _round_to_grid(worst_bad // 2)
        return candidate if candidate >= MIN_CTX and candidate < worst_bad else None

    # Measured beats recorded; recorded beats nothing.
    slope = measured_mib_per_token(probes) or hint_mib_per_token
    headroom = next(
        (p.free_mib for p in probes if p.ctx == best_ok and p.free_mib is not None), None
    )

    # Already as close to the margin as the grid can express: one more step up
    # would breach it, so the answer is the context in hand.
    if slope and headroom is not None and (headroom - margin_mib) < slope * GRANULARITY:
        return None

    if worst_bad is None:
        # Everything has succeeded. Reach upward -- predicted where possible,
        # doubling only when nothing at all is known, never past the model's own
        # ceiling.
        if slope and headroom is not None:
            spare = headroom - margin_mib
            candidate = _round_to_grid(best_ok + int(spare / slope)) if spare > 0 else best_ok
            if candidate <= best_ok:
                candidate = _round_to_grid(best_ok + GRANULARITY * 4)
        else:
            candidate = _round_to_grid(best_ok * 2)
        candidate = min(candidate, _round_to_grid(max_ctx))
        return candidate if candidate > best_ok else None

    # Bracketed. Settle when the gap stops being worth a load.
    if worst_bad - best_ok <= SETTLE_TOKENS:
        return None

    if slope and headroom is not None and headroom > margin_mib:
        candidate = _round_to_grid(best_ok + int((headroom - margin_mib) / slope))
    else:
        candidate = _round_to_grid((best_ok + worst_bad) // 2)

    # Keep it strictly inside the bracket, or the search cannot converge.
    if candidate <= best_ok or candidate >= worst_bad:
        candidate = _round_to_grid((best_ok + worst_bad) // 2)
    if candidate <= best_ok or candidate >= worst_bad:
        return None
    return candidate


@dataclass(slots=True)
class CeilingSearch:
    """One search: what it is looking for, how far it has got, what it found."""

    id: str
    model_key: str
    port: int
    margin_mib: int
    max_ctx: int
    start_ctx: int
    status: CeilingStatus = CeilingStatus.QUEUED
    phase: str = "queued"
    probes: list[Probe] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def best(self) -> Probe | None:
        ok = [p for p in self.probes if p.within_margin]
        return max(ok, key=lambda p: p.ctx) if ok else None

    def to_dict(self) -> dict[str, Any]:
        best = self.best
        return {
            "id": self.id,
            "model_key": self.model_key,
            "status": self.status.value,
            "phase": self.phase,
            "margin_mib": self.margin_mib,
            "max_ctx": self.max_ctx,
            "probes": [p.to_dict() for p in self.probes],
            "probes_done": len(self.probes),
            "max_probes": MAX_PROBES,
            "best_ctx": best.ctx if best else None,
            "best_free_mib": best.free_mib if best else None,
            "mib_per_token": (
                round(measured_mib_per_token(self.probes), 4)
                if measured_mib_per_token(self.probes)
                else None
            ),
            "result": self.result,
            "error": self.error,
            "elapsed_seconds": round((self.finished_at or time.time()) - self.started_at, 1),
        }


def summarise(search: CeilingSearch) -> dict[str, Any]:
    """What the search found, in terms someone can act on."""
    best = search.best
    slope = measured_mib_per_token(search.probes)
    failed = [p for p in search.probes if not p.loaded]

    if best is None:
        return {
            "found": False,
            "note": (
                "No context in the searched range left every card above the "
                f"{search.margin_mib} MiB margin. The smallest tried was "
                f"{min((p.ctx for p in search.probes), default=search.start_ctx):,}. This model "
                "may simply not fit on these cards with room to work in."
            ),
        }

    tokens_per_gib = int(1024 / slope) if slope else None
    return {
        "found": True,
        "ctx": best.ctx,
        "free_mib": best.free_mib,
        "breakdown": best.breakdown,
        "mib_per_token": round(slope, 4) if slope else None,
        "tokens_per_gib": tokens_per_gib,
        "first_failure": min((p.ctx for p in failed), default=None),
        "note": (
            f"{best.ctx:,} is the largest context tried that left every card above "
            f"{search.margin_mib} MiB — {best.free_mib} MiB free on the tightest one. "
            + (
                f"Context costs about {slope:.3f} MiB per token here"
                + (f", so roughly {tokens_per_gib:,} tokens per GiB. " if tokens_per_gib else ". ")
                if slope
                else ""
            )
            + (
                f"{min(p.ctx for p in failed):,} did not load. "
                if failed
                else "Nothing in the range failed to load. "
            )
            + "Measured on this machine as it was at the time: another workload on these "
            "cards changes the answer. Nothing has been written to the registry."
        ),
    }


class CeilingRunner:
    """Runs one ceiling search at a time, against a real server.

    The probing itself is injected rather than imported. Starting a model needs
    the registry, the settings and the process layer, and pulling those in here
    would make the search untestable without a GPU -- which is exactly the
    property that matters, because a single real run costs minutes of loading.
    The caller supplies an async callable that tries one context and reports
    what happened; everything above it is decided by `plan_next_context`, which
    is pure.

    Serialised for the same reason benchmarks are, only more so: two searches
    would be starting and stopping the same llama-server underneath each other.
    """

    def __init__(self) -> None:
        self._searches: dict[str, CeilingSearch] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def list(self) -> list[CeilingSearch]:
        return sorted(self._searches.values(), key=lambda s: s.started_at, reverse=True)

    def get(self, search_id: str) -> CeilingSearch | None:
        return self._searches.get(search_id)

    def active(self) -> CeilingSearch | None:
        for s in self._searches.values():
            if s.status in (CeilingStatus.QUEUED, CeilingStatus.RUNNING):
                return s
        return None

    def start(
        self,
        *,
        model_key: str,
        port: int,
        start_ctx: int,
        margin_mib: int,
        max_ctx: int,
        probe_fn: Any,
        hint_mib_per_token: float | None = None,
    ) -> CeilingSearch:
        running = self.active()
        if running is not None:
            raise CeilingError(
                f"a ceiling search is already running against {running.model_key!r}. "
                "Two would be starting and stopping the same server underneath each other."
            )
        search = CeilingSearch(
            id=uuid.uuid4().hex[:12],
            model_key=model_key,
            port=port,
            margin_mib=margin_mib,
            max_ctx=max_ctx,
            start_ctx=start_ctx,
        )
        self._searches[search.id] = search
        self._tasks[search.id] = asyncio.create_task(
            self._run(search, hint_mib_per_token, probe_fn)
        )
        return search

    def cancel(self, search_id: str) -> bool:
        task = self._tasks.get(search_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def _run(self, search: CeilingSearch, hint: float | None, probe_fn: Any) -> None:
        try:
            search.status = CeilingStatus.RUNNING
            candidate: int | None = search.start_ctx

            while candidate is not None:
                search.phase = f"trying {candidate:,} context"
                log.info("ceiling %s: probing ctx %d", search.id, candidate)
                probe = await probe_fn(candidate, search.margin_mib)
                search.probes.append(probe)
                candidate = plan_next_context(
                    search.probes,
                    margin_mib=search.margin_mib,
                    max_ctx=search.max_ctx,
                    hint_mib_per_token=hint,
                )

            search.result = summarise(search)
            search.phase = "done"
            search.status = CeilingStatus.COMPLETE
            search.finished_at = time.time()

        except asyncio.CancelledError:
            search.status = CeilingStatus.CANCELLED
            search.phase = "cancelled"
            search.finished_at = time.time()
            # Whatever probes completed are still real measurements, so they are
            # kept -- but no ceiling is claimed from a search that was cut short.
            search.error = "cancelled; the server may need stopping by hand"
            raise
        except Exception as exc:
            search.status = CeilingStatus.FAILED
            search.phase = "failed"
            search.error = str(exc)
            search.finished_at = time.time()
            log.exception("ceiling search %s failed", search.id)


class CeilingError(RuntimeError):
    pass
