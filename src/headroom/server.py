"""llama-server process lifecycle.

**Headroom never owns the inference server.** This is the single most important
design decision in the project and everything in this module follows from it.

If the server were a child of the UI process, closing the window would kill the
model. Reloading a 15 GiB model costs minutes, and no dashboard should be able
to cost you that by being closed, crashing, or being restarted during
development. So:

- Headroom **spawns detached**. The child is deliberately orphaned.
- Headroom **attaches** to whatever is already listening, whether it started it
  or not. A server launched from a shell script an hour ago is a first-class
  citizen.
- State is **discovered, never remembered**. Everything reported comes from the
  live server's ``/props`` and its actual command line, so Headroom cannot drift
  into confidently describing a server that no longer exists.

The consequence worth stating plainly: Headroom is a *view onto* your inference
server, not a manager of it. Your CLI and this UI are peers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psutil

log = logging.getLogger(__name__)

SERVER_PROCESS_NAMES = {"llama-server", "llama-server.exe"}


@dataclass(slots=True)
class ServerState:
    """What is actually running, discovered fresh each time."""

    running: bool = False
    reachable: bool = False
    port: int = 8080
    pid: int | None = None
    model_path: str | None = None
    model_name: str | None = None
    n_ctx: int | None = None
    vision: bool = False
    command_line: list[str] = field(default_factory=list)
    host_ram_mib: int | None = None
    uptime_seconds: float | None = None
    error: str | None = None

    @property
    def status(self) -> str:
        """`running` | `loading` | `stopped` | `orphaned`.

        `loading` matters, and more than it first appears. Two separate gaps
        open during startup, measured on a 15.4 GiB model:

        - the process exists but nothing answers at all (~0-10 s), and
        - **`/health` answers while `/props` still does not** (~10-25 s).

        That second gap is the trap. Anything that waits on `/health` alone will
        conclude the server is ready and then read back a null model and a null
        context length. Readiness here therefore means `/props` answered, which
        is why `reachable` is set from that endpoint and not from `/health`.

        Reporting either gap as `stopped` would invite the user to start a
        second server, and two of these do not fit in memory at once.

        `orphaned` means the port answers but no matching process was found --
        typically something else is on that port.
        """
        if self.reachable:
            return "running"
        if self.pid is not None:
            return "loading"
        if self.running:
            return "orphaned"
        return "stopped"


async def probe(port: int = 8080, timeout: float = 3.0) -> ServerState:
    """Discover the current server state. Never raises.

    Two independent sources, deliberately:

    - ``/props`` is the documented surface and gives the loaded model and n_ctx.
    - The **process command line** is ground truth for how it was launched.

    They are not redundant. ``/props`` reports a ``vision`` capability flag that
    can be true on builds where the feature is compiled in but non-functional,
    whereas ``--mmproj`` on the command line means a projector was actually
    loaded. When a wrong answer is worse than no answer -- and for vision it very
    much is, since a text-only server will happily invent a description of an
    image it never received -- trust the command line.
    """
    state = ServerState(port=port)

    proc = find_server_process(port)
    if proc is not None:
        state.running = True
        state.pid = proc.pid
        try:
            state.command_line = proc.cmdline()
            state.host_ram_mib = proc.memory_info().rss // 1024 // 1024
            state.uptime_seconds = _uptime(proc)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"http://127.0.0.1:{port}/props")
            resp.raise_for_status()
            props = resp.json()
        state.reachable = True
        state.model_path = props.get("model_path")
        if state.model_path:
            state.model_name = Path(state.model_path).name
        gen = props.get("default_generation_settings") or {}
        state.n_ctx = gen.get("n_ctx")
    except Exception as exc:  # noqa: BLE001 - unreachable is a state, not an error
        if state.pid is not None:
            state.error = f"process {state.pid} is up but /props did not answer: {exc}"

    # Command line wins for vision. See the docstring.
    state.vision = any(a == "--mmproj" for a in state.command_line)

    return state


def find_server_process(port: int = 8080) -> psutil.Process | None:
    """Find the llama-server process, preferring one actually bound to `port`.

    Matching on the listening socket rather than the process name alone means a
    second server on another port is not mistaken for this one. Falls back to a
    name match, because connection enumeration needs privileges that are not
    always available.
    """
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.status != psutil.CONN_LISTEN or not conn.laddr:
                continue
            if conn.laddr.port != port or conn.pid is None:
                continue
            try:
                proc = psutil.Process(conn.pid)
                if proc.name() in SERVER_PROCESS_NAMES:
                    return proc
            except psutil.NoSuchProcess:
                continue
    except (psutil.AccessDenied, PermissionError):
        log.debug("connection enumeration denied; falling back to process name")

    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] in SERVER_PROCESS_NAMES:
                return proc
        except psutil.NoSuchProcess:
            continue
    return None


def _uptime(proc: psutil.Process) -> float | None:
    try:
        import time

        return round(time.time() - proc.create_time(), 1)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Spawning
# --------------------------------------------------------------------------


class SpawnError(RuntimeError):
    pass


def spawn_detached(argv: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    """Start llama-server so it OUTLIVES this process. Returns the child PID.

    On Windows this uses DETACHED_PROCESS with CREATE_BREAKAWAY_FROM_JOB. The
    breakaway flag matters more than it looks: when the parent sits inside a Job
    object configured to kill on close -- which is how many terminals, IDEs and
    agent harnesses run their children -- a merely detached child still dies with
    it. Breakaway opts out, where the job permits it.

    If the job forbids breakaway, `spawn_via_wmi` is the escape hatch: it asks
    the WMI provider to create the process, so the child is parented outside the
    job entirely.

    On POSIX, `start_new_session=True` calls setsid, which detaches from the
    controlling terminal and process group.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "ab", buffering=0)  # noqa: SIM115 - deliberately outlives this call

    kwargs: dict = {
        "stdout": handle,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if env is not None:
        kwargs["env"] = {**os.environ, **env}

    if sys.platform == "win32":
        flags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
        )
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(argv, **kwargs)
    except OSError as exc:
        handle.close()
        raise SpawnError(f"could not start llama-server: {exc}") from exc

    log.info("spawned llama-server pid=%s detached, logging to %s", proc.pid, log_path)
    return proc.pid


def spawn_via_wmi(argv: list[str], log_path: Path) -> int:
    """Windows fallback: create the process through the WMI provider.

    Use when `spawn_detached` produces a child that still dies with the parent,
    which happens inside Job objects that forbid breakaway. The WMI provider
    becomes the parent, so the new process is outside the job entirely.

    Costs a PowerShell round trip, so it is not the default.
    """
    if sys.platform != "win32":
        raise SpawnError("spawn_via_wmi is Windows-only")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    quoted = " ".join(f'"{a}"' if " " in a else a for a in argv)
    command = f'cmd.exe /c {quoted} > "{log_path}" 2>&1'
    ps = (
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        f"-Arguments @{{ CommandLine = '{command}' }}; "
        'Write-Output "$($r.ReturnValue) $($r.ProcessId)"'
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    parts = (result.stdout or "").split()
    if len(parts) != 2 or parts[0] != "0":
        raise SpawnError(f"WMI spawn failed: {result.stdout.strip()} {result.stderr.strip()}")
    return int(parts[1])


async def wait_until_ready(port: int, timeout: float = 300.0, poll: float = 2.0) -> ServerState:
    """Poll until the server answers, or give up.

    The default timeout is generous on purpose. A 15 GiB model off a cold page
    cache can take a while, and a UI that declares failure early tempts the user
    into starting a second server that will fight the first for VRAM.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    state = ServerState(port=port)

    while loop.time() < deadline:
        state = await probe(port)
        if state.reachable:
            return state
        if state.pid is None and state.error is None:
            # Nothing running at all: it died during startup rather than being slow.
            await asyncio.sleep(poll)
            state = await probe(port)
            if state.pid is None:
                state.error = "llama-server exited during startup; check the server log"
                return state
        await asyncio.sleep(poll)

    state.error = f"server did not become ready within {timeout:.0f}s"
    return state


async def stop(port: int = 8080, timeout: float = 30.0, force: bool = False) -> ServerState:
    """Stop the server and release the GPUs.

    Terminates politely first and escalates only if asked. Killing a server
    mid-generation loses the user's work, so `force` is opt-in rather than a
    silent fallback.

    Waits for the process to actually exit before returning: process death and
    VRAM release are not simultaneous, and reporting "stopped" while the memory
    is still held invites an immediate start that then OOMs.
    """
    state = await probe(port)
    if state.pid is None:
        return state

    try:
        proc = psutil.Process(state.pid)
    except psutil.NoSuchProcess:
        return await probe(port)

    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except psutil.TimeoutExpired:
        if not force:
            state.error = (
                f"llama-server (pid {state.pid}) did not exit within {timeout:.0f}s. "
                "It may be mid-generation. Retry with force to kill it."
            )
            return state
        proc.kill()
        try:
            proc.wait(timeout=10)
        except psutil.TimeoutExpired:
            state.error = f"could not kill pid {state.pid}"
            return state
    except psutil.AccessDenied:
        state.error = f"access denied stopping pid {state.pid}"
        return state

    return await probe(port)
