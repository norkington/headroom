"""Benchmarking a running server, and recording the result honestly.

This module is a port of ``bin/bench.ps1``, and the port is deliberately
faithful rather than tidied up. The numbers already in the registry were
produced by that script, so a benchmark that measures *differently* does not
produce a second opinion — it produces a number that cannot be compared to the
first one while looking exactly like it can.

Five rules are baked in here because each of them was learned by getting a
confident answer that turned out to be wrong:

1. **Warm-up runs are discarded.** After a model change the first few runs are
   not representative — one observed sequence climbed 19.61 → 27.03 → 30.33
   before settling near 30. Averaging those in reports a slow model.

2. **Prefill needs a long prompt.** The task prompts here are ~40 tokens, and at
   that length ``prompt_per_second`` measures per-request overhead rather than
   prefill throughput: it produced 54.9 tok/s with a standard deviation *larger
   than the mean*, against a true figure near 895 on the same box. Prefill
   therefore gets its own stage with a ~6000-token prompt and ``max_tokens=1``.

3. **Every prefill rep needs a unique prompt.** The server caches prompts, so an
   identical repeated prompt is served from cache. The first version of the
   shell stage kept 1 of 3 runs for exactly this reason and never said so. Each
   rep gets a nonce, and any run whose ``prompt_n`` shows it was cached is
   dropped rather than averaged in.

4. **Decode tracks draft acceptance.** With speculative decoding on, decode
   moves with the accept rate, which varies by task (0.366–0.605 measured
   here). A decode figure recorded without its acceptance range invites someone
   to read a task difference as a regression, so acceptance is always recorded
   alongside.

5. **Run-to-run SD is around 3%.** The standard deviation is reported so that a
   difference under ~6% can be recognised as not a result. A mean without a
   spread is what makes one-run comparisons look conclusive.

The field is ``draft_n_accepted``, not ``draft_accepted_n``. Getting it wrong
yields ``None`` silently and a blank acceptance column.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol

import httpx

from .store import JsonStore, prune

log = logging.getLogger(__name__)

DEFAULT_REPS = 3
DEFAULT_WARMUP = 3
DEFAULT_MAX_TOKENS = 512
DEFAULT_PREFILL_TOKENS = 6000

# A generation can legitimately take minutes on a large model at long context,
# so the per-request ceiling is generous. It exists to stop a wedged server
# hanging the job forever, not to bound normal slowness.
REQUEST_TIMEOUT = 900.0

# Below this many actually-processed prompt tokens, the run was served from the
# prompt cache and measures nothing. See rule 3.
CACHED_PROMPT_THRESHOLD = 1000

# The `measured` keys a benchmark run is the authority on. Everything else in
# that block -- a hand-written comparison, a note about where the context
# ceiling sits -- is the operator's, and a run that cannot produce it has no
# business deleting it.
#
# The distinction is not cosmetic. Replacing the whole block loses prose that
# took real work; merging the whole block lets a figure from an older run sit
# beside fresh ones looking equally current, which is the precise dishonesty
# this project exists to avoid. Owning a specific set is what separates the two.
OWNED_MEASURED_KEYS = frozenset(
    {
        "status",
        "decode_tok_s",
        "decode_sd",
        "decode_runs",
        "decode_note",
        "prefill_tok_s",
        "prefill_sd",
        "prefill_runs",
        "prefill_note",
        "mtp_acceptance_range",
        "vram_free_breakdown",
        "carried_forward",
    }
)


def owns_measured_key(name: str) -> bool:
    """Whether a benchmark run is the authority on this `measured` key.

    The `vram_free_mib*` prefix is matched rather than listed because the key
    carries the context length in its name -- `vram_free_mib_at_64k`. A run at a
    different context must *remove* the old figure, not leave it sitting beside
    results it was not measured with. Matching the family is what makes that
    happen; listing one literal key would silently keep the stale one.
    """
    return name in OWNED_MEASURED_KEYS or name.startswith("vram_free_mib")


class GpuReader(Protocol):
    """Just enough of the GPU backend for the one figure this module needs."""

    def __call__(self) -> list[Any]: ...


# Three prompts of deliberately different shape. Averaging over one shape would
# report a number that does not generalise to real work — and with speculative
# decoding the shape is exactly what moves the accept rate.
TASKS: tuple[tuple[str, str], ...] = (
    (
        "impl",
        (
            "Write a PowerShell function that takes a directory and returns the ten largest "
            "files under it, recursively, sorted descending, with sizes in GiB. No explanation."
        ),
    ),
    (
        "debug",
        (
            'A CUDA program fails with "out of memory" only when a second process starts, and '
            "works fine alone. List the five most likely causes, most likely first, one line "
            "each."
        ),
    ),
    (
        "refactor",
        (
            "Explain, in one paragraph, why quantization error in a recurrent (state-carrying) "
            "layer compounds with sequence length while error in a feed-forward layer does not."
        ),
    ),
)

# Repeated to build the long prefill prompt. Deliberately bland: the point is to
# occupy context, and anything interesting risks the model producing a long
# answer when the stage wants max_tokens=1.
_PREFILL_UNIT = (
    "The maintenance log records routine calibration of the secondary array. "
    "Ambient conditions remained within nominal bounds throughout the interval. "
)


class BenchStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # Headroom stopped mid-run. Kept as a record of an attempt, never as a
    # result: a run cut short measures the part of the workload it got to, and
    # rule 1 exists because that part is not representative.
    INTERRUPTED = "interrupted"


#: Nothing is working on these any more.
FINISHED = frozenset(
    {
        BenchStatus.COMPLETE.value,
        BenchStatus.FAILED.value,
        BenchStatus.CANCELLED.value,
        BenchStatus.INTERRUPTED.value,
    }
)


class BenchError(RuntimeError):
    pass


@dataclass(slots=True)
class Run:
    """One completed generation, as the server timed it."""

    decode_tok_s: float | None
    prefill_tok_s: float | None
    acceptance: float | None
    prompt_n: int | None


@dataclass(slots=True)
class Benchmark:
    """One benchmark job: what it is measuring, how far it has got, what it found."""

    id: str
    model_key: str
    model_path: str
    port: int
    reps: int
    warmup: int
    max_tokens: int
    prefill_tokens: int
    status: BenchStatus = BenchStatus.QUEUED
    phase: str = "queued"
    n_ctx: int | None = None
    runs_done: int = 0
    runs_total: int = 0
    per_task: dict[str, dict[str, Any]] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    written: bool = False
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def percent(self) -> float | None:
        if not self.runs_total:
            return None
        return min(100.0, self.runs_done / self.runs_total * 100.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_key": self.model_key,
            "model_path": self.model_path,
            "status": self.status.value,
            "phase": self.phase,
            "n_ctx": self.n_ctx,
            "runs_done": self.runs_done,
            "runs_total": self.runs_total,
            "percent": self.percent,
            "per_task": self.per_task,
            "result": self.result,
            "written": self.written,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round((self.finished_at or time.time()) - self.started_at, 1),
        }

    def to_record(self) -> dict[str, Any]:
        """The durable form: everything except the live progress fields.

        A finished run is worth keeping for the same reason its figures are
        worth reporting with a spread -- the next person to compare two numbers
        needs to see how each was reached, and the registry records only the
        result. `phase` and the elapsed clock describe a run in motion and are
        not carried.
        """
        return {
            "id": self.id,
            "model_key": self.model_key,
            "model_path": self.model_path,
            "port": self.port,
            "reps": self.reps,
            "warmup": self.warmup,
            "max_tokens": self.max_tokens,
            "prefill_tokens": self.prefill_tokens,
            "status": self.status.value,
            "n_ctx": self.n_ctx,
            "runs_done": self.runs_done,
            "runs_total": self.runs_total,
            "per_task": self.per_task,
            "result": self.result,
            "written": self.written,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Benchmark | None:
        """Rebuild from a saved record.

        A run that was still going when the process died comes back
        `interrupted` and **without a result**, even if some tasks had already
        produced figures. Those partial numbers are exactly what rules 1 and 3
        say to throw away: warm-up may not have finished, prefill certainly did
        not run, and a decode figure from two of three tasks is not the same
        measurement as one from three. Cancelling records nothing for the same
        reason, and a crash is not a better outcome than a cancel.
        """
        try:
            status = BenchStatus(record["status"])
            b = cls(
                id=str(record["id"]),
                model_key=str(record["model_key"]),
                model_path=str(record.get("model_path") or ""),
                port=int(record.get("port") or 0),
                reps=int(record.get("reps") or DEFAULT_REPS),
                warmup=int(record.get("warmup") or 0),
                max_tokens=int(record.get("max_tokens") or DEFAULT_MAX_TOKENS),
                prefill_tokens=int(record.get("prefill_tokens") or 0),
                status=status,
                n_ctx=record.get("n_ctx"),
                runs_done=int(record.get("runs_done") or 0),
                runs_total=int(record.get("runs_total") or 0),
                per_task=dict(record.get("per_task") or {}),
                result=record.get("result"),
                written=bool(record.get("written")),
                error=record.get("error"),
                started_at=float(record.get("started_at") or time.time()),
                finished_at=record.get("finished_at"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("ignoring unreadable benchmark record: %s", exc)
            return None

        b.phase = status.value
        if status in (BenchStatus.QUEUED, BenchStatus.RUNNING):
            b.status = BenchStatus.INTERRUPTED
            b.phase = "interrupted"
            b.result = None
            b.per_task = {}
            b.finished_at = b.finished_at or time.time()
            b.error = (
                "Headroom stopped while this run was in progress. Nothing was recorded -- a "
                "partial run is not a measurement."
            )
        return b


def stdev(values: list[float]) -> float:
    """Sample standard deviation. Zero for fewer than two values.

    Sample rather than population: these are repetitions drawn from a process,
    not the whole population of possible runs, and with n=3 the difference is
    not negligible.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def context_label(n_ctx: int | None) -> str:
    """`65536` -> `64k`, to match how the registry already names this figure."""
    if not n_ctx:
        return "unknown"
    if n_ctx % 1024 == 0:
        return f"{n_ctx // 1024}k"
    return str(n_ctx)


class BenchmarkRunner:
    """Runs benchmarks against an already-running server, one at a time.

    Serialised deliberately. Two concurrent benchmarks would contend for the
    same server's batch slots and each would measure the other's interference,
    producing two numbers that are both wrong and neither obviously so.
    """

    def __init__(self, read_gpus: GpuReader | None = None, store: JsonStore | None = None) -> None:
        self._jobs: dict[str, Benchmark] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        # Injected rather than imported so this module does not depend on how
        # telemetry is polled, and so a machine with no GPU backend simply
        # records no VRAM figure instead of failing the run.
        self._read_gpus = read_gpus
        self._store = store

    def _persist(self) -> None:
        if self._store is None:
            return
        self._store.save(prune([b.to_record() for b in self.list()], finished=FINISHED))

    def restore(self) -> int:
        """Reload runs from a previous process. Returns how many came back.

        Results outlive the process that produced them because they cost
        minutes of loaded GPU to obtain, and because the registry keeps only the
        figures -- not the rep count, the acceptance spread or the cached-prefill
        count that say how much weight the figures carry.
        """
        if self._store is None:
            return 0
        restored = 0
        for record in self._store.load():
            b = Benchmark.from_record(record)
            if b is None or b.id in self._jobs:
                continue
            self._jobs[b.id] = b
            restored += 1
        if restored:
            log.info("restored %d benchmark record(s)", restored)
            # The reconciled view goes straight back, so a record left saying
            # `running` does not outlive the process by another restart.
            self._persist()
        return restored

    def list(self) -> list[Benchmark]:
        return sorted(self._jobs.values(), key=lambda b: b.started_at, reverse=True)

    def get(self, bench_id: str) -> Benchmark | None:
        return self._jobs.get(bench_id)

    def active(self) -> Benchmark | None:
        for b in self._jobs.values():
            if b.status in (BenchStatus.QUEUED, BenchStatus.RUNNING):
                return b
        return None

    def start(
        self,
        *,
        model_key: str,
        model_path: str,
        port: int,
        reps: int = DEFAULT_REPS,
        warmup: int = DEFAULT_WARMUP,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        prefill_tokens: int = DEFAULT_PREFILL_TOKENS,
        on_complete: Any = None,
    ) -> Benchmark:
        running = self.active()
        if running is not None:
            raise BenchError(
                f"a benchmark is already running against {running.model_key!r}. "
                "Two at once would each measure the other's interference."
            )

        bench = Benchmark(
            id=uuid.uuid4().hex[:12],
            model_key=model_key,
            model_path=model_path,
            port=port,
            reps=max(1, reps),
            warmup=max(0, warmup),
            max_tokens=max(1, max_tokens),
            prefill_tokens=max(0, prefill_tokens),
        )
        bench.runs_total = bench.warmup + len(TASKS) * bench.reps + bench.reps
        self._jobs[bench.id] = bench
        self._tasks[bench.id] = asyncio.create_task(self._run(bench, on_complete))
        self._persist()
        return bench

    def cancel(self, bench_id: str) -> bool:
        task = self._tasks.get(bench_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    # ------------------------------------------------------------------ run
    async def _run(self, b: Benchmark, on_complete: Any) -> None:
        try:
            b.status = BenchStatus.RUNNING
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                b.n_ctx = await self._read_n_ctx(client, b.port)

                # Warm-up. Discarded, and counted towards progress so the UI does
                # not appear stuck for the first minute of a run.
                b.phase = "warming up"
                for _ in range(b.warmup):
                    await self._generate(client, b, TASKS[0][1], b.max_tokens)
                    b.runs_done += 1

                decode_all: list[float] = []
                acceptance_all: list[float] = []
                task_acceptance: dict[str, float] = {}

                for name, prompt in TASKS:
                    b.phase = f"task: {name}"
                    runs: list[Run] = []
                    for _ in range(b.reps):
                        runs.append(await self._generate(client, b, prompt, b.max_tokens))
                        b.runs_done += 1

                    decodes = [r.decode_tok_s for r in runs if r.decode_tok_s is not None]
                    accepts = [r.acceptance for r in runs if r.acceptance is not None]
                    decode_all += decodes
                    acceptance_all += accepts
                    task_mean_acceptance = mean(accepts)
                    if task_mean_acceptance is not None:
                        task_acceptance[name] = task_mean_acceptance

                    b.per_task[name] = {
                        "decode_tok_s": _round(mean(decodes), 2),
                        "acceptance": _round(task_mean_acceptance, 3),
                        "runs": len(decodes),
                    }

                prefill_values: list[float] = []
                prefill_cached = 0
                if b.prefill_tokens > 0:
                    b.phase = "prefill (long prompt)"
                    prefill_values, prefill_cached = await self._measure_prefill(client, b)
                prefill_value = mean(prefill_values)
                prefill_runs = len(prefill_values)

            # Sampled here: after the last generation, with the model still
            # resident and the server idle. That is the figure someone actually
            # wants -- "how much room is left while this model is loaded" -- and
            # it is the one number Headroom can take better than the shell
            # script, because it already has per-card telemetry rather than one
            # nvidia-smi line.
            b.phase = "reading VRAM"
            vram_total, vram_breakdown = self._read_free_vram()

            b.result = self._summarise(
                b,
                decode_all=decode_all,
                acceptance_all=acceptance_all,
                task_acceptance=task_acceptance,
                prefill_value=prefill_value,
                prefill_values=prefill_values,
                prefill_runs=prefill_runs,
                prefill_cached=prefill_cached,
                vram_total=vram_total,
                vram_breakdown=vram_breakdown,
            )
            b.phase = "done"
            b.status = BenchStatus.COMPLETE
            b.finished_at = time.time()

            if on_complete is not None:
                # Recording the result is the caller's business — this module
                # measures and does not decide what the measurement means for
                # the registry.
                try:
                    b.written = bool(on_complete(b))
                except Exception as exc:  # a write failure must not void the numbers
                    b.error = f"measured successfully, but recording it failed: {exc}"
                    log.exception("recording benchmark %s failed", b.id)

            # Saved after the registry write, not before: `written` is half of
            # what the record says, and a run kept as unwritten when it was
            # written sends someone looking for a figure that is already there.
            self._persist()

        except asyncio.CancelledError:
            b.status = BenchStatus.CANCELLED
            b.phase = "cancelled"
            b.finished_at = time.time()
            self._persist()
            raise
        except Exception as exc:
            b.status = BenchStatus.FAILED
            b.phase = "failed"
            b.error = str(exc)
            b.finished_at = time.time()
            self._persist()
            log.exception("benchmark %s failed", b.id)

    async def _read_n_ctx(self, client: httpx.AsyncClient, port: int) -> int | None:
        try:
            resp = await client.get(f"http://127.0.0.1:{port}/props", timeout=15.0)
            resp.raise_for_status()
            gen = resp.json().get("default_generation_settings") or {}
            return gen.get("n_ctx")
        except Exception:  # noqa: BLE001 - context length is nice to have, not required
            return None

    async def _generate(
        self, client: httpx.AsyncClient, b: Benchmark, prompt: str, max_tokens: int
    ) -> Run:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = await client.post(f"http://127.0.0.1:{b.port}/v1/chat/completions", json=body)
        resp.raise_for_status()
        timings = resp.json().get("timings") or {}

        draft_n = timings.get("draft_n")
        accepted = timings.get("draft_n_accepted")
        acceptance = None
        if draft_n and accepted is not None:
            acceptance = float(accepted) / float(draft_n)

        return Run(
            decode_tok_s=timings.get("predicted_per_second"),
            prefill_tok_s=timings.get("prompt_per_second"),
            acceptance=acceptance,
            prompt_n=timings.get("prompt_n"),
        )

    async def _measure_prefill(
        self, client: httpx.AsyncClient, b: Benchmark
    ) -> tuple[list[float], int]:
        """Measure prefill with a long prompt, discarding cache hits.

        Returns the kept per-run figures and how many runs were dropped for
        having been served from the prompt cache. The individual values come
        back rather than a mean so prefill gets a standard deviation like decode
        does; the dropped count comes back rather than being logged because a
        run where most reps were cached rests on a far smaller sample than the
        rep count suggests, and the UI has to be able to say so.
        """
        tokens_per_copy = await self._tokens_in(client, b.port, _PREFILL_UNIT * 4)
        copies = max(1, math.ceil(b.prefill_tokens / max(1, tokens_per_copy)))
        big = _PREFILL_UNIT * 4 * copies

        kept: list[float] = []
        cached = 0
        for i in range(b.reps):
            # A unique prefix per rep. Without it every rep after the first is a
            # cache hit, and the sample size collapses silently. See rule 3.
            nonce = f"Run {i} of {random.randint(100000, 999999)}. "
            run = await self._generate(
                client, b, f"{nonce}{big}\n\nReply with the single word: ok", 1
            )
            b.runs_done += 1
            if (run.prompt_n or 0) > CACHED_PROMPT_THRESHOLD and run.prefill_tok_s:
                kept.append(run.prefill_tok_s)
            else:
                cached += 1

        return kept, cached

    async def _tokens_in(self, client: httpx.AsyncClient, port: int, text: str) -> int:
        try:
            resp = await client.post(
                f"http://127.0.0.1:{port}/tokenize", json={"content": text}, timeout=30.0
            )
            resp.raise_for_status()
            count = len(resp.json().get("tokens") or [])
            return count if count > 0 else 30
        except Exception:  # noqa: BLE001
            # The shell script's fallback. Overestimating tokens-per-copy would
            # build too short a prompt and quietly reintroduce rule 2's bug, so
            # the fallback is deliberately low.
            return 30

    def _read_free_vram(self) -> tuple[int | None, str | None]:
        """Free VRAM per card, summed and described. Never raises.

        Reported per card as well as summed, because the total is the misleading
        half: a machine can show several gigabytes free "in total" while the card
        that also drives the desktop sits a browser tab away from an OOM. The
        breakdown names each card the way llama.cpp does, so the figure can be
        acted on.
        """
        if self._read_gpus is None:
            return None, None
        try:
            cards = self._read_gpus()
        except Exception:  # a telemetry failure must not void a good benchmark
            log.warning("could not read GPU telemetry for the VRAM figure", exc_info=True)
            return None, None
        if not cards:
            return None, None

        total = sum(int(c.memory_free_mib) for c in cards)

        # Ordered by CUDA index, not by the backend's enumeration order. On a
        # machine where the two disagree -- the case this project exists to
        # surface -- NVML order prints CUDA1 before CUDA0, and the reader has to
        # do the reconciliation the app is supposed to have done for them. Cards
        # with no resolved CUDA index sort last rather than being guessed at.
        def order(card: Any) -> tuple[int, int]:
            cuda = getattr(card, "cuda_index", None)
            return (1, 0) if cuda is None else (0, int(cuda))

        parts = [f"{int(c.memory_free_mib)} MiB on the {c.label}" for c in sorted(cards, key=order)]
        return total, " + ".join(parts)

    # ------------------------------------------------------------ summarise
    def _summarise(
        self,
        b: Benchmark,
        *,
        decode_all: list[float],
        acceptance_all: list[float],
        task_acceptance: dict[str, float],
        prefill_value: float | None,
        prefill_values: list[float],
        prefill_runs: int,
        prefill_cached: int,
        vram_total: int | None,
        vram_breakdown: str | None,
    ) -> dict[str, Any]:
        """Build the ``measured`` block, notes and caveats included.

        The notes are not decoration. Every figure here has a way of being
        misread, and the registry is consulted months later by someone who was
        not present for the run — including its author. A number that has to be
        interpreted travels with its interpretation or it eventually gets
        misused.
        """
        decode_mean = mean(decode_all)
        decode_sd = stdev(decode_all)

        measured: dict[str, Any] = {
            "status": (
                f"MEASURED on this exact file, {_today()}, "
                "via Headroom (same method as bin/bench.ps1)"
            ),
        }

        if decode_mean is not None:
            measured["decode_tok_s"] = round(decode_mean, 2)
            measured["decode_sd"] = round(decode_sd, 2)
            measured["decode_runs"] = len(decode_all)

        if task_acceptance:
            lo = min(task_acceptance.values())
            hi = max(task_acceptance.values())
            spread = ", ".join(
                f"{name} {b.per_task[name]['decode_tok_s']} @ {value:.3f}"
                for name, value in task_acceptance.items()
            )
            measured["mtp_acceptance_range"] = f"{lo:.3f} - {hi:.3f}"
            measured["decode_note"] = (
                f"Decode tracks MTP draft acceptance, which ranged {lo:.3f}-{hi:.3f} across the "
                f"three bench tasks ({spread}). Read decode WITH acceptance, never alone."
            )

        if prefill_value is not None:
            measured["prefill_tok_s"] = round(prefill_value, 1)
            measured["prefill_sd"] = round(stdev(prefill_values), 1)
            measured["prefill_runs"] = prefill_runs
            note = (
                f"Measured with a ~{b.prefill_tokens}-token prompt and max_tokens=1. "
                "Short-prompt prefill is meaningless here -- 40-token prompts measure request "
                "overhead, not prefill throughput."
            )
            if prefill_cached:
                note += (
                    f" {prefill_cached} of {prefill_runs + prefill_cached} prefill run(s) hit the "
                    "prompt cache and were discarded, so this figure rests on a smaller sample "
                    "than the rep count suggests."
                )
            measured["prefill_note"] = note
        else:
            measured["prefill_note"] = (
                "NOT MEASURED. Every prefill run was served from the prompt cache, so no honest "
                "figure is available. Restart the server and run again."
            )

        if vram_total is not None:
            # The context length lives in the key. A figure measured at 64K says
            # nothing about the same model at 96K, and a bare `vram_free_mib`
            # invites exactly that reading.
            suffix = f"_at_{context_label(b.n_ctx)}" if b.n_ctx else ""
            measured[f"vram_free_mib{suffix}"] = vram_total
            if vram_breakdown:
                measured["vram_free_breakdown"] = vram_breakdown

        return {
            "measured": measured,
            "n_ctx": b.n_ctx,
            "context_label": context_label(b.n_ctx),
            "vram_free_mib": vram_total,
            "vram_free_breakdown": vram_breakdown,
            "decode_tok_s": measured.get("decode_tok_s"),
            "decode_sd": measured.get("decode_sd"),
            "prefill_tok_s": measured.get("prefill_tok_s"),
            "acceptance_range": measured.get("mtp_acceptance_range"),
            "prefill_cached_runs": prefill_cached,
            # Repeated here so the UI can show it next to the result without
            # having to know the rule. A reader who sees only "27.6 vs 26.1"
            # will call it a regression; one who sees the SD will not.
            "significance_note": (
                "Run-to-run SD is around 3% on this class of machine. A difference under ~6% "
                "between two configurations is not a result."
            ),
        }


def _today() -> str:
    """Local calendar date. What a reader means by "when was this measured"."""
    return datetime.now(UTC).astimezone().date().isoformat()


def _round(value: float | None, places: int) -> float | None:
    return None if value is None else round(value, places)
