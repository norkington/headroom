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

from . import downloads as downloads_mod
from . import gguf as gguf_mod
from . import gpu as gpu_mod
from . import registry as registry_mod
from . import server as server_mod

log = logging.getLogger(__name__)


def _env_path(name: str, default: str) -> Path:
    """Read a path from the environment, tolerating stray whitespace.

    A trailing space in an environment variable is easy to produce on Windows --
    `set VAR=value && cmd` in cmd.exe captures the space before the `&&` -- and
    it fails in a way that is almost invisible. Windows will happily open
    ``"models.json "``, so the app reads the right file and looks correct, while
    every path *derived* from it inherits the space: a backup written alongside
    lands as ``models.json .bak`` instead of ``models.json.bak``.

    Found exactly that way. Stripping costs nothing and removes the whole class.
    """
    return Path(os.environ.get(name, default).strip())


@dataclass(slots=True)
class Settings:
    """Where things live. Every value is overridable by environment variable.

    Defaults point at the developer's own layout; nothing here is load-bearing
    for correctness, only for convenience on first run.
    """

    registry_path: Path = field(
        default_factory=lambda: _env_path("HEADROOM_REGISTRY", r"C:\AI\models\models.json")
    )
    llama_server: Path = field(
        default_factory=lambda: _env_path(
            "HEADROOM_LLAMA_SERVER",
            r"C:\src\llama.cpp\build\bin\Release\llama-server.exe",
        )
    )
    log_dir: Path = field(
        default_factory=lambda: _env_path(
            "HEADROOM_LOG_DIR",
            # Never under Documents: Controlled Folder Access silently blocks
            # writes there on Windows, and a blocked write looks like a bug in
            # whatever tried to log.
            os.path.join(os.environ.get("LOCALAPPDATA", "."), "headroom", "logs"),
        )
    )
    port: int = int(os.environ.get("HEADROOM_SERVER_PORT", "8080"))
    poll_interval: float = float(os.environ.get("HEADROOM_POLL_SECONDS", "1.0"))


class State:
    """Process-wide state. Telemetry hardware handles, and the cached CUDA map."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend: gpu_mod.GpuBackend = gpu_mod.NvmlBackend()
        self.downloads = downloads_mod.DownloadManager()
        self.cuda_mapping = gpu_mod.CudaMapping()
        self.mapping_resolved = False

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
    settings = settings or Settings()
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
        return {
            "ok": True,
            "version": "0.1.0",
            "gpu_backend_available": hr.backend.available(),
            "registry": str(settings.registry_path),
            "registry_exists": settings.registry_path.exists(),
            "llama_server": str(settings.llama_server),
            "llama_server_exists": settings.llama_server.exists(),
        }

    # ---------------------------------------------------------------- gpus
    @app.get("/api/gpus")
    async def gpus() -> dict[str, Any]:
        hr: State = app.state.hr
        cards = hr.gpus()
        return {
            "gpus": [
                asdict(g) | {"headroom_state": g.headroom_state, "label": g.label} for g in cards
            ],
            "cuda_mapping": {
                "cuda_to_nvml": hr.cuda_mapping.cuda_to_nvml,
                "resolved": hr.cuda_mapping.resolved,
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
                            "utilization_pct": g.utilization_pct,
                            "power_watts": g.power_watts,
                            "temperature_c": g.temperature_c,
                        }
                        for g in cards
                    ],
                    "server": {
                        "status": state.status,
                        "pid": state.pid,
                        "model_name": state.model_name,
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
        state = await server_mod.probe(settings.port)
        return asdict(state) | {"status": state.status}

    @app.post("/api/server/stop")
    async def server_stop(force: bool = False) -> dict[str, Any]:
        state = await server_mod.stop(settings.port, force=force)
        return asdict(state) | {"status": state.status}

    @app.post("/api/server/start")
    async def server_start(model: str | None = None, vision: bool = False) -> dict[str, Any]:
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

        try:
            reg = registry_mod.load(settings.registry_path)
            entry = reg.get(model)
            argv = registry_mod.build_argv(
                entry, settings.llama_server, port=settings.port, vision=vision
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
        """List the GGUF files in a HuggingFace repository."""
        try:
            files = await gguf_mod.list_repo_files(repo)
        except gguf_mod.GgufError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"could not reach the hub: {exc}") from exc
        return {"repo": repo, "files": files}

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
        return {"cancelled": True, "note": "partial file kept; starting again will resume"}

    # ---------------------------------------------------------------- registry write
    @app.post("/api/registry/add")
    async def registry_add(
        key: str,
        repo: str,
        file: str,
        label: str | None = None,
        download: bool = True,
        inherit_from: str | None = None,
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
            d = hr.downloads.start(repo, file, target_dir / file)
            result["download"] = d.to_dict()

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
