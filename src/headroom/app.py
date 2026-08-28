"""HTTP API.

Binds to loopback only. Headroom has no accounts, no telemetry and no cloud
dependency, and the bind address is the first place that promise is either kept
or quietly broken — so it is not configurable to anything routable without the
operator going out of their way.

The telemetry endpoint is Server-Sent Events rather than WebSockets. The data
flows one way at about 1 Hz, SSE reconnects on its own, and it survives proxies
that mangle upgrades. A WebSocket would be a strictly larger surface for no gain
here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import bench as bench_mod
from . import config as config_mod
from . import downloads as downloads_mod
from . import gguf as gguf_mod
from . import gpu as gpu_mod
from . import registry as registry_mod
from . import server as server_mod
from .store import JsonStore

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Settings:
    """Where things live, and how each location was decided.

    Nothing here is hardcoded to one machine. Paths come from
    :mod:`headroom.config`, which checks explicit arguments, then environment
    variables, then a config file, then conventional install locations -- and
    keeps a record of which of those answered, because "found nothing" and
    "found the wrong thing" need different fixes and look the same from outside.
    """

    registry_path: Path
    llama_server: Path | None
    log_dir: Path
    registry_resolution: config_mod.Resolution
    llama_resolution: config_mod.Resolution
    # Records of work Headroom itself did -- downloads and benchmark runs. Kept
    # apart from the registry, which is the operator's file and shared with
    # their scripts: nothing in here is a setting, and losing all of it costs a
    # transfer's progress and some history, never a configuration.
    state_dir: Path = field(default_factory=lambda: config_mod.data_dir() / "state")
    port: int = 8080
    poll_interval: float = 1.0

    @classmethod
    def resolve(
        cls,
        *,
        registry: str | None = None,
        llama_server: str | None = None,
        create_registry: bool = True,
        port: int = 8080,
    ) -> Settings:
        """Work out where everything is, creating a starter registry if needed.

        `create_registry` defaults to true so a fresh clone has somewhere to put
        its first model. An app that errors until the user reads the source is
        not a working app.
        """
        reg = config_mod.resolve_registry(registry, create=create_registry)
        llama = config_mod.resolve_llama_server(llama_server)
        log_dir = Path(
            (os.environ.get("HEADROOM_LOG_DIR") or "").strip() or (config_mod.data_dir() / "logs")
        )
        state_dir = Path(
            (os.environ.get("HEADROOM_STATE_DIR") or "").strip()
            or (config_mod.data_dir() / "state")
        )
        return cls(
            registry_path=reg.path or (config_mod.data_dir() / "models.json"),
            llama_server=llama.path if llama.exists else None,
            log_dir=log_dir,
            state_dir=state_dir,
            registry_resolution=reg,
            llama_resolution=llama,
            port=int((os.environ.get("HEADROOM_SERVER_PORT") or "").strip() or port),
            poll_interval=float((os.environ.get("HEADROOM_POLL_SECONDS") or "").strip() or 1.0),
        )


class State:
    """Process-wide state. Telemetry hardware handles, and the cached CUDA map.

    Downloads and benchmarks are given a store each, so both survive the process
    that started them. Neither resumes on its own: what comes back is a record,
    and restarting a transfer or a measurement is a decision someone makes.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend: gpu_mod.GpuBackend = gpu_mod.NvmlBackend()
        self.downloads = downloads_mod.DownloadManager(
            store=JsonStore(settings.state_dir / "downloads.json")
        )
        self.bench = bench_mod.BenchmarkRunner(
            read_gpus=self.gpus, store=JsonStore(settings.state_dir / "benchmarks.json")
        )
        self.cuda_mapping = gpu_mod.CudaMapping()
        self.mapping_resolved = False
        self._key_for_path: tuple[str, str] | None = None
        self.downloads.restore()
        self.bench.restore()

    def registry_key_for(self, model_path: str | None) -> str | None:
        """Which registry entry the running server loaded, if any.

        Resolved from the file the server actually has open, because Headroom
        attaches to servers it did not start -- one launched from a shell script
        an hour ago is a first-class citizen, and it has no idea what the UI
        currently has selected in a dropdown.

        Only successful resolutions are cached. A miss is re-checked on the next
        poll, so adding the running model to the registry starts being reflected
        immediately rather than after a restart.
        """
        if not model_path:
            return None
        if self._key_for_path and self._key_for_path[0] == model_path:
            return self._key_for_path[1]
        try:
            reg = registry_mod.load(self.settings.registry_path)
            entry = registry_mod.find_by_path(reg, model_path)
        except registry_mod.RegistryError:
            return None
        if entry is None:
            return None
        self._key_for_path = (model_path, entry.key)
        return entry.key

    def gpus(self) -> list[gpu_mod.Gpu]:
        """Poll, applying the cached CUDA mapping.

        The mapping is resolved once and reused: it spawns a subprocess, and it
        only changes when hardware or drivers do. Resolving it per poll would put
        a process launch on a 1 Hz timer for a value that never moves.
        """
        gpus = self.backend.poll()
        if not gpus:
            return gpus

        if not self.mapping_resolved:
            if self.settings.llama_server is None:
                # Without llama.cpp there is no authoritative device order. Saying
                # nothing is correct here -- guessing would be worse than a gap,
                # because the guess is wrong on exactly the machines that matter.
                self.cuda_mapping = gpu_mod.CudaMapping(
                    warning="llama-server not found, so CUDA device indices are unresolved"
                )
            else:
                self.cuda_mapping = gpu_mod.resolve_cuda_mapping(self.settings.llama_server, gpus)
            self.mapping_resolved = True
            if self.cuda_mapping.warning:
                log.warning("CUDA mapping: %s", self.cuda_mapping.warning)
            if gpu_mod.order_differs(self.cuda_mapping):
                log.warning(
                    "CUDA and NVML device order DISAGREE on this machine: %s. "
                    "nvidia-smi indices and -dev CUDAn refer to different cards.",
                    self.cuda_mapping.cuda_to_nvml,
                )
        else:
            nvml_to_cuda = {v: k for k, v in self.cuda_mapping.cuda_to_nvml.items()}
            for g in gpus:
                g.cuda_index = nvml_to_cuda.get(g.nvml_index)
        return gpus


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    app.state.hr = State(settings)
    log.info("registry: %s", settings.registry_path)
    log.info("llama-server: %s", settings.llama_server)
    yield
    app.state.hr.backend.shutdown()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.resolve()
    app = FastAPI(
        title="Headroom",
        description="Operations console for local LLM inference",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # ---------------------------------------------------------------- health
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        hr: State = app.state.hr
        problems: list[str] = []
        if not settings.registry_path.exists():
            problems.append(
                f"No model registry at {settings.registry_path}. Add a model through the UI, "
                "or point Headroom at an existing registry with --registry."
            )
        if settings.llama_server is None:
            problems.append(
                "llama-server was not found. Put it on your PATH, or pass "
                "--llama-server /path/to/llama-server. Serving and CUDA device "
                "identification need it; everything else works without it."
            )
        return {
            "ok": True,
            "version": "0.1.0",
            "gpu_backend_available": hr.backend.available(),
            "registry": str(settings.registry_path),
            "registry_exists": settings.registry_path.exists(),
            "registry_source": settings.registry_resolution.as_dict(),
            "llama_server": str(settings.llama_server) if settings.llama_server else None,
            "llama_server_exists": settings.llama_server is not None,
            "llama_server_source": settings.llama_resolution.as_dict(),
            "config_file": str(config_mod.config_file()),
            "state_dir": str(settings.state_dir),
            # Surfaced so the UI can tell a fresh install what it still needs,
            # instead of appearing broken.
            "problems": problems,
        }

    # ---------------------------------------------------------------- gpus
    @app.get("/api/gpus")
    async def gpus() -> dict[str, Any]:
        hr: State = app.state.hr
        cards = hr.gpus()
        # Probed here as well as in the stream, at the cost of one round trip on
        # a one-off request. A card's figure being provisional is part of what
        # that figure means, and an endpoint that quietly omitted it would be
        # handing out the optimistic half.
        state = await server_mod.probe(settings.port)
        gpu_mod.mark_vision_residency(cards, vision=state.vision, command_line=state.command_line)
        return {
            "gpus": [
                asdict(g)
                | {
                    "headroom_state": g.headroom_state,
                    "headroom_provisional": g.headroom_provisional,
                    "thermal_state": g.thermal_state,
                    "thermal_headroom_c": g.thermal_headroom_c,
                    "throttle_labels": list(g.throttle_labels),
                    "throttling_thermally": g.throttling_thermally,
                    "throttling_for_power": g.throttling_for_power,
                    "label": g.label,
                }
                for g in cards
            ],
            "cuda_mapping": {
                "cuda_to_nvml": hr.cuda_mapping.cuda_to_nvml,
                "resolved": hr.cuda_mapping.resolved,
                # Resolved says something came back; trustworthy says every
                # device was pinned to a specific card. On a rig of identical
                # GPUs those are different answers.
                "trustworthy": hr.cuda_mapping.trustworthy,
                "ambiguous": list(hr.cuda_mapping.ambiguous),
                "source": hr.cuda_mapping.source,
                "warning": hr.cuda_mapping.warning,
                # The UI should say this out loud when true. It means nvidia-smi's
                # numbering and llama.cpp's numbering name different cards.
                "order_differs": gpu_mod.order_differs(hr.cuda_mapping),
            },
        }

    # ---------------------------------------------------------------- telemetry
    @app.get("/api/telemetry")
    async def telemetry() -> StreamingResponse:
        hr: State = app.state.hr

        async def stream():
            # A client disconnect arrives as CancelledError and is ordinary, not
            # an error. It needs no handler, so there deliberately is not one --
            # catching it only to re-raise would be noise.
            while True:
                cards = hr.gpus()
                state = await server_mod.probe(settings.port)
                # Which cards are holding a projector, and are therefore still
                # on their way down. Done per poll rather than once, because a
                # vision server can be started and stopped under a UI that stays
                # open -- that is the whole point of attaching rather than owning.
                gpu_mod.mark_vision_residency(
                    cards, vision=state.vision, command_line=state.command_line
                )
                payload = {
                    "gpus": [
                        {
                            "nvml_index": g.nvml_index,
                            "cuda_index": g.cuda_index,
                            "name": g.name,
                            "label": g.label,
                            "memory_total_mib": g.memory_total_mib,
                            "memory_used_mib": g.memory_used_mib,
                            "memory_free_mib": g.memory_free_mib,
                            "headroom_state": g.headroom_state,
                            "vision_resident": g.vision_resident,
                            "headroom_provisional": g.headroom_provisional,
                            "utilization_pct": g.utilization_pct,
                            "power_watts": g.power_watts,
                            "temperature_c": g.temperature_c,
                            # Thermals travel with the memory figures because
                            # they explain them: a card that is clamping itself
                            # is not delivering the throughput its VRAM implies.
                            "thermal_state": g.thermal_state,
                            "thermal_headroom_c": g.thermal_headroom_c,
                            "temp_slowdown_c": g.temp_slowdown_c,
                            "fan_percent": g.fan_percent,
                            "throttle_labels": list(g.throttle_labels),
                            "throttling_thermally": g.throttling_thermally,
                            "throttling_for_power": g.throttling_for_power,
                        }
                        for g in cards
                    ],
                    "server": {
                        "status": state.status,
                        "pid": state.pid,
                        "model_name": state.model_name,
                        "model_key": hr.registry_key_for(state.model_path),
                        "n_ctx": state.n_ctx,
                        "vision": state.vision,
                        "host_ram_mib": state.host_ram_mib,
                        "uptime_seconds": state.uptime_seconds,
                        "error": state.error,
                    },
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(settings.poll_interval)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---------------------------------------------------------------- models
    @app.get("/api/models")
    async def models() -> dict[str, Any]:
        try:
            reg = registry_mod.load(settings.registry_path)
        except registry_mod.RegistryError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return {
            "registry_path": str(reg.path),
            "default": reg.default,
            "models": [
                {
                    "key": m.key,
                    "label": m.label,
                    "repo": m.repo,
                    "file": m.file,
                    "size_gib": m.size_gib,
                    "arch": m.arch,
                    "installed": m.installed,
                    "path": str(m.path),
                    "uncensored": m.uncensored,
                    "license": m.license,
                    "vision_supported": bool(m.vision.get("supported")),
                    "vision_tuned": m.vision_tuned,
                    "mmproj": m.mmproj,
                    "why_this_build": m.why_this_build,
                    "serve": m.serve,
                    "measured": m.measured,
                    "verified": m.verified,
                    # Surfaced so the UI can visually separate a measured number
                    # from an inherited one. Presenting the two identically is
                    # how a dashboard becomes untrustworthy.
                    "measured_on_this_file": m.measured_on_this_file,
                }
                for m in reg.models.values()
            ],
        }

    # ---------------------------------------------------------------- server
    @app.get("/api/server")
    async def server_state() -> dict[str, Any]:
        hr: State = app.state.hr
        state = await server_mod.probe(settings.port)
        return asdict(state) | {
            "status": state.status,
            "model_key": hr.registry_key_for(state.model_path),
        }

    @app.post("/api/server/stop")
    async def server_stop(force: bool = False) -> dict[str, Any]:
        state = await server_mod.stop(settings.port, force=force)
        return asdict(state) | {"status": state.status}

    @app.post("/api/server/start")
    async def server_start(
        model: str | None = None, vision: bool = False, ctx: int | None = None
    ) -> dict[str, Any]:
        existing = await server_mod.probe(settings.port)
        if existing.status in {"running", "loading"}:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"a server is already {existing.status} on port {settings.port} "
                    f"(pid {existing.pid}). Stop it before starting another — "
                    "two models will not fit."
                ),
            )

        if settings.llama_server is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "llama-server was not found, so there is nothing to start. Put it on "
                    "your PATH or pass --llama-server."
                ),
            )

        try:
            reg = registry_mod.load(settings.registry_path)
            entry = reg.get(model)
            argv = registry_mod.build_argv(
                entry,
                settings.llama_server,
                port=settings.port,
                vision=vision,
                # An explicit context beats both the serve block and the vision
                # profile, which is what makes it useful for finding a ceiling:
                # the registry value is the one being questioned. build_argv
                # already gives an override precedence over vision.ctx.
                overrides={"ctx": ctx} if ctx else None,
            )
        except registry_mod.RegistryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        log_path = settings.log_dir / f"llama-server-{entry.key}.log"
        try:
            pid = server_mod.spawn_detached(argv, log_path)
        except server_mod.SpawnError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "started": True,
            "pid": pid,
            "model": entry.key,
            "vision": vision,
            "ctx_override": ctx,
            "log": str(log_path),
            "argv": argv,
            # The caller polls /api/server or the telemetry stream from here.
            # Loading a large model takes tens of seconds and blocking the HTTP
            # request for that long would just time out somewhere unhelpful.
            "note": "loading; poll /api/server until status is 'running'",
        }

    # ---------------------------------------------------------------- probe
    @app.get("/api/hf/files")
    async def hf_files(repo: str) -> dict[str, Any]:
        """List the GGUF files in a HuggingFace repository, quants and projectors apart.

        Kept apart because they are not interchangeable and both are `.gguf`. A
        projector offered in a list of quants is a thing someone will eventually
        pick, and it will fail late -- at load, not at selection.
        """
        try:
            # Normalised first so the response can echo what was actually
            # queried. A pasted model-page URL is the common input, and showing
            # it back unchanged leaves the user unsure whether it was understood.
            repo = gguf_mod.normalise_repo(repo)
            files = await gguf_mod.list_repo_files(repo)
        except gguf_mod.GgufError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"could not reach the hub: {exc}") from exc
        return {
            "repo": repo,
            "files": [f for f in files if f["kind"] == "model"],
            "projectors": [f for f in files if f["kind"] == "projector"],
            # A suggestion the caller should show rather than apply silently:
            # which precision is wanted is a VRAM judgement, not a filename fact.
            "suggested_projector": gguf_mod.choose_projector(files),
        }

    @app.get("/api/probe")
    async def probe_gguf(repo: str | None = None, file: str | None = None, path: str | None = None):
        """Inspect a GGUF's tensor table without downloading the weights.

        Judged against **live free VRAM** where telemetry is available, because
        a size in gibibytes only means something relative to what this machine
        actually has spare right now.
        """
        hr: State = app.state.hr
        cards = hr.gpus()
        free_vram = sum(g.memory_free_mib for g in cards) if cards else None

        try:
            if path:
                analysis = gguf_mod.probe_local(path, free_vram_mib=free_vram)
            elif repo and file:
                analysis = await gguf_mod.probe_remote(repo, file, free_vram_mib=free_vram)
            else:
                raise HTTPException(
                    status_code=400, detail="provide either path, or both repo and file"
                )
        except gguf_mod.GgufError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"probe failed: {exc}") from exc

        return {
            "source": analysis.source,
            "architecture": analysis.architecture,
            "name": analysis.name,
            "tensor_count": analysis.tensor_count,
            "size_gib": analysis.size_gib,
            "bytes_read": analysis.bytes_read,
            "has_mtp": analysis.has_mtp,
            "mtp_tensor_count": len(analysis.mtp_tensors),
            "families": analysis.families,
            "metadata": analysis.metadata,
            "findings": [asdict(f) for f in analysis.findings],
            "free_vram_mib": free_vram,
        }

    # ---------------------------------------------------------------- downloads
    @app.get("/api/downloads")
    async def downloads_list() -> dict[str, Any]:
        hr: State = app.state.hr
        return {"downloads": [d.to_dict() for d in hr.downloads.list()]}

    @app.delete("/api/downloads/{download_id}")
    async def download_cancel(download_id: str) -> dict[str, Any]:
        hr: State = app.state.hr
        if not hr.downloads.cancel(download_id):
            raise HTTPException(status_code=404, detail="no such active download")
        # The partial file is kept deliberately, so resuming costs seconds
        # rather than starting the whole transfer again.
        return {"cancelled": True, "note": "partial file kept; resuming will continue from it"}

    @app.post("/api/downloads/{download_id}/resume")
    async def download_resume(download_id: str) -> dict[str, Any]:
        """Continue a transfer that stopped -- cancelled, failed, or interrupted.

        This is the path back for a download Headroom was in the middle of when
        it was shut down. The bytes are still in the `.part` file, so what is
        needed is the record that says which repo they came from, and that is
        what survives a restart now.
        """
        hr: State = app.state.hr
        try:
            download = hr.downloads.resume(download_id)
        except downloads_mod.DownloadError as exc:
            status = 404 if str(exc) == "no such download" else 409
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        return download.to_dict() | {"note": "resumed from the bytes already on disk"}

    # ---------------------------------------------------------------- bench
    @app.get("/api/bench")
    async def bench_list() -> dict[str, Any]:
        hr: State = app.state.hr
        return {"benchmarks": [b.to_dict() for b in hr.bench.list()]}

    @app.post("/api/bench/start")
    async def bench_start(
        reps: int = bench_mod.DEFAULT_REPS,
        warmup: int = bench_mod.DEFAULT_WARMUP,
        prefill_tokens: int = bench_mod.DEFAULT_PREFILL_TOKENS,
        write: bool = True,
    ) -> dict[str, Any]:
        """Benchmark whatever is currently serving, and record the result.

        There is deliberately no `model` parameter. The entry being measured is
        resolved from the file the running server actually loaded, so the numbers
        can only ever be attributed to the model that produced them. Letting the
        caller name a model would make it possible to benchmark one and write the
        result onto another -- silently, into a file shared with the user's shell
        scripts, in a project whose whole claim is that its numbers are honest.
        """
        hr: State = app.state.hr
        state = await server_mod.probe(settings.port)

        if state.status != "running":
            detail = {
                "loading": (
                    "the server is still loading. Benchmarking now would measure a model that "
                    "is not ready -- wait for it to finish."
                ),
                "stopped": (
                    "nothing is serving on port "
                    f"{settings.port}. Start a model before benchmarking it."
                ),
                "orphaned": (
                    f"something is listening on port {settings.port} but it is not a "
                    "recognisable llama-server, so there is no telling what would be measured."
                ),
            }.get(state.status, f"server status is {state.status!r}")
            raise HTTPException(status_code=409, detail=detail)

        try:
            reg = registry_mod.load(settings.registry_path)
        except registry_mod.RegistryError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        entry = registry_mod.find_by_path(reg, state.model_path or "")
        if entry is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"the running server loaded {state.model_path!r}, which is not in "
                    f"{settings.registry_path}. Headroom will not guess which entry these "
                    "numbers belong to -- add the file to the registry first."
                ),
            )

        def record(b: bench_mod.Benchmark) -> bool:
            if not write or not b.result:
                return False
            registry_mod.record_measurement(
                settings.registry_path,
                b.model_key,
                b.result["measured"],
                # A benchmark is the authority on throughput and on free VRAM.
                # It is not the authority on the operator's own notes, and a run
                # that cannot produce a key has no business deleting it.
                owns=bench_mod.owns_measured_key,
            )
            return True

        try:
            job = hr.bench.start(
                model_key=entry.key,
                model_path=str(entry.path),
                port=settings.port,
                reps=reps,
                warmup=warmup,
                prefill_tokens=prefill_tokens,
                on_complete=record if write else None,
            )
        except bench_mod.BenchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return job.to_dict() | {
            "note": (
                "running; poll /api/bench. Warm-up runs are discarded and prefill is measured "
                "separately with a long prompt -- see the benchmark module for why."
            ),
            "will_write": write,
        }

    @app.delete("/api/bench/{bench_id}")
    async def bench_cancel(bench_id: str) -> dict[str, Any]:
        hr: State = app.state.hr
        if not hr.bench.cancel(bench_id):
            raise HTTPException(status_code=404, detail="no such running benchmark")
        # Nothing is written on cancel: a partial run is not a measurement, and
        # recording one would be worse than having no number at all.
        return {"cancelled": True, "note": "no partial result was recorded"}

    # ---------------------------------------------------------------- registry write
    @app.post("/api/registry/add")
    async def registry_add(
        key: str,
        repo: str,
        file: str,
        label: str | None = None,
        download: bool = True,
        inherit_from: str | None = None,
        mmproj: str | None = None,
    ) -> dict[str, Any]:
        """Add a probed quant to the registry, and optionally fetch it.

        The entry is derived from a fresh probe rather than from user input, so
        architecture, size and the presence of a speculative-decoding head are
        read off the file itself instead of being asserted.
        """
        hr: State = app.state.hr
        cards = hr.gpus()
        free_vram = sum(g.memory_free_mib for g in cards) if cards else None

        try:
            analysis = await gguf_mod.probe_remote(repo, file, free_vram_mib=free_vram)
        except gguf_mod.GgufError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            reg = registry_mod.load(settings.registry_path)
        except registry_mod.RegistryError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # Where to put the weights: alongside the other models, in a directory
        # named for this entry.
        existing = next(iter(reg.models.values()), None)
        if existing is None:
            raise HTTPException(
                status_code=400,
                detail="the registry has no existing entry to infer a weights directory from",
            )
        weights_root = Path(existing.directory).parent
        target_dir = weights_root / key

        inherit = None
        if inherit_from:
            try:
                inherit = reg.get(inherit_from)
            except registry_mod.RegistryError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        entry = registry_mod.derive_entry(
            key=key,
            label=label or (analysis.name or file),
            repo=repo,
            filename=file,
            directory=str(target_dir).replace("\\", "/"),
            size_gib=analysis.size_gib or 0.0,
            architecture=analysis.architecture,
            has_mtp=analysis.has_mtp,
            template=(reg.raw.get("models") or {}).get("_template"),
            inherit_from=inherit,
            mmproj=mmproj,
        )

        try:
            registry_mod.add_entry(settings.registry_path, key, entry)
        except registry_mod.RegistryError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        result: dict[str, Any] = {
            "added": key,
            "entry": entry,
            "registry": str(settings.registry_path),
            "backup": str(settings.registry_path) + ".bak",
        }

        if download:
            result["download"] = hr.downloads.start(repo, file, target_dir / file).to_dict()
            if mmproj:
                # Fetched with the weights, not on first use. An entry that claims
                # vision and has no projector on disk fails at load -- minutes into
                # a start, long after the decision that caused it.
                result["projector_download"] = hr.downloads.start(
                    repo, mmproj, target_dir / mmproj
                ).to_dict()

        return result

    # ---------------------------------------------------------------- frontend
    # Mounted LAST so every /api route above wins. A StaticFiles mount at "/"
    # is greedy and would otherwise swallow them.
    #
    # The directory is generated by `npm run build` in web/ and is deliberately
    # not in version control. When it is absent -- a source checkout that has
    # never built the UI -- the root route says so in plain terms rather than
    # returning a bare 404 that looks like a broken install.
    static_dir = Path(__file__).parent / "static"
    if (static_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
        log.info("serving UI from %s", static_dir)
    else:

        @app.get("/")
        async def no_ui() -> dict[str, Any]:
            return {
                "ok": True,
                "ui": "not built",
                "detail": (
                    "The web UI has not been built. Run `npm install && npm run build` "
                    "in web/, then restart. The API under /api works regardless."
                ),
                "api_docs": "/docs",
            }

        log.warning("UI not built (%s missing); serving API only", static_dir)

    return app
