"""Does it actually start?

Every other test in this suite drives the app in-process, through FastAPI's
`TestClient`. That proves the routing and the logic, and it proves nothing about
the thing a new user does first: run the command and open the page. Between
those two lies real uvicorn startup, real port binding, the platform's data and
config directories, and creating a registry on a machine that has never run this
before -- none of which `TestClient` touches.

That gap matters most on the platform the author does not use. This project is
developed on Windows; CI runs the same suite on Linux, so putting the check here
rather than in the workflow means Linux startup is exercised on every push
without anyone maintaining a second copy of it in YAML. An earlier version of
this project did embed environment checks directly in ci.yml, where nothing
linted them, and they drifted against a changed constructor until CI caught it.

Deliberately tolerant about what it finds and strict about what it does. Whether
this machine has a GPU, a llama.cpp build, or a built frontend is not the
subject -- the app is supposed to come up regardless and say which of them are
missing. So the assertions are about *coming up and being honest*, not about the
host's inventory.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

# Generous: a cold CI runner importing torch-free but still substantial deps,
# on a shared vCPU, is slower than a warm laptop. The failure this guards is a
# process that never binds at all, and that is unambiguous well before here.
STARTUP_TIMEOUT = 60.0


def _free_port() -> int:
    """A port nothing is listening on, released immediately.

    There is an unavoidable race between releasing it and uvicorn binding it.
    Nothing else is starting servers on an ephemeral port during a test run, so
    the exposure is a few milliseconds wide and the alternative -- a fixed port
    -- collides with whatever the developer already has running.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    # Always a loopback URL this test built itself, never user input.
    with urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


@pytest.fixture
def running_app(tmp_path: Path):
    """The real command, in a real process, on a machine that has never run it.

    Every directory the app would touch is redirected into tmp_path, so this
    neither reads nor writes the developer's own registry -- which on the
    author's machine is shared with shell scripts and is not a test fixture.
    """
    port = _free_port()
    env = dict(os.environ)
    env.update(
        {
            "LOCALAPPDATA": str(tmp_path),  # Windows
            "XDG_DATA_HOME": str(tmp_path),  # Linux
            "XDG_CONFIG_HOME": str(tmp_path),
            "HEADROOM_STATE_DIR": str(tmp_path / "state"),
            "HEADROOM_LOG_DIR": str(tmp_path / "logs"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    # Not the console script. On the author's machine `headroom.exe` was observed
    # handing off through a chain of processes that ended on a different Python
    # than the one under test, which is precisely the ambiguity a startup test
    # must not have.
    proc = subprocess.Popen(
        [sys.executable, "-m", "headroom", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    "the app exited before serving anything.\n"
                    f"exit code {proc.returncode}\n{proc.stdout.read() if proc.stdout else ''}"
                )
            try:
                status, _ = _get(f"{base}/api/health")
                if status == 200:
                    break
            except (URLError, OSError, ConnectionError):
                time.sleep(0.4)
        else:
            proc.kill()
            raise AssertionError(
                f"nothing answered on {base} within {STARTUP_TIMEOUT:.0f}s.\n"
                f"{proc.stdout.read() if proc.stdout else ''}"
            )
        yield base, tmp_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_it_comes_up_on_a_machine_that_has_never_run_it(running_app) -> None:
    """The first-run path, end to end, as a real process.

    No config file, no registry, no state directory -- the state a fresh clone
    is in. Coming up degraded is fine and is the design; failing to come up is
    not.
    """
    base, home = running_app
    status, body = _get(f"{base}/api/health")
    health = json.loads(body)

    assert status == 200
    assert health["ok"] is True

    # A registry was created rather than the app erroring until someone reads
    # the source. Its location is reported, so this asserts against what the app
    # says rather than against a path this test guessed.
    registry = Path(health["registry"])
    assert registry.exists(), f"no registry was created at {registry}"
    assert str(home) in str(registry), "the app wrote outside the redirected home"

    # Where everything came from is part of the contract: "found nothing" and
    # "found the wrong thing" need different fixes and look identical otherwise.
    assert health["registry_source"]["source"]
    assert "state_dir" in health


def test_the_starter_registry_is_usable_rather_than_merely_present(running_app) -> None:
    """An empty file would satisfy "it exists" and help nobody."""
    base, _ = running_app
    _, body = _get(f"{base}/api/health")
    registry = Path(json.loads(body)["registry"])

    data = json.loads(registry.read_text(encoding="utf-8"))
    assert data["schema"] == 1
    assert "_template" in data["models"], "the starter registry has nothing to copy from"

    # And the app can read back what it just wrote.
    status, models_body = _get(f"{base}/api/models")
    assert status == 200
    # Keys beginning with _ are ignored by the loader, so a fresh registry lists
    # no models at all -- which is correct, not a failure.
    assert json.loads(models_body)["models"] == []


def test_telemetry_and_server_state_answer_without_hardware(running_app) -> None:
    """Neither a GPU nor llama.cpp is required for the app to be useful.

    On a runner there is no GPU; on the author's machine there is. Both must
    answer, so this asserts on the shape rather than the inventory.
    """
    base, _ = running_app

    status, body = _get(f"{base}/api/gpus")
    assert status == 200
    payload = json.loads(body)
    assert isinstance(payload["gpus"], list)
    assert "cuda_mapping" in payload

    status, body = _get(f"{base}/api/server")
    assert status == 200
    assert json.loads(body)["status"] in {"running", "loading", "stopped", "orphaned"}


def test_the_root_route_never_returns_a_bare_404(running_app) -> None:
    """With no frontend built, the root must explain itself.

    CI's Python job does not build the UI, so this runs against the un-built
    case there and the built one locally. A bare 404 in either would leave
    someone who followed the README staring at nothing.
    """
    base, _ = running_app
    status, body = _get(base)

    assert status == 200
    built = Path(__file__).resolve().parents[1] / "src" / "headroom" / "static" / "index.html"
    if built.exists():
        assert "<" in body, "a built frontend should serve markup"
    else:
        assert "npm run build" in body, "the un-built root route must say how to fix itself"
