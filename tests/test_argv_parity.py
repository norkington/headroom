"""The launcher-parity contract.

Headroom's promise is that its config *is* your config. That promise is only
worth anything if launching from the UI produces the same server as launching
from the shell script. If the two drift, every measurement taken through one
stops being comparable to the other, and the drift is silent — both launches
"work", they just aren't the same server.

So this compares Headroom's generated command line against the shell launcher's
own `-DryRun` output, flag by flag.

Ordering is deliberately ignored: llama-server does not care where `-dev`
appears, and asserting on order would make this fail for a reason that does not
matter. Values are compared exactly.

Skips cleanly when the local launcher or registry is absent, so the suite still
runs on a contributor's machine that has neither.

The three paths below are the author's layout as defaults and are overridable by
environment variable, because a test that hardcodes one machine's directories
would be the same works-on-one-computer assumption that :mod:`headroom.config`
exists to remove. Anyone with a launcher of their own can point this at it:

    HEADROOM_PARITY_REGISTRY=... HEADROOM_PARITY_LAUNCHER=... uv run pytest

Defaults rather than a plain skip, so the contract keeps being checked on the
machine it was written against without anyone remembering to set anything.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from headroom.registry import build_argv, load


def _path_from_env(name: str, default: str) -> Path:
    """An overridable path, with surrounding whitespace removed.

    Stripped for the same reason `headroom.config` strips: a trailing space is
    easy to produce on Windows and nearly invisible, and every path derived from
    the value inherits it.
    """
    raw = os.environ.get(name) or ""
    return Path(raw.strip() or default)


REGISTRY = _path_from_env("HEADROOM_PARITY_REGISTRY", r"C:\AI\models\models.json")
LAUNCHER = _path_from_env("HEADROOM_PARITY_LAUNCHER", r"C:\AI\models\bin\serve.ps1")
# Only ever argv[0]: the comparison is flag by flag and never looks at the
# executable, so this one does not need to exist for the test to be meaningful.
LLAMA_SERVER = _path_from_env(
    "HEADROOM_PARITY_LLAMA_SERVER", r"C:\src\llama.cpp\build\bin\Release\llama-server.exe"
)

# Flags that take no value; everything else is assumed to be `--flag value`.
BOOLEAN_FLAGS = {"--jinja", "--no-mmap", "--verbose"}


def _parse(argv: list[str]) -> tuple[str, dict[str, str], set[str]]:
    """Split a command line into (exe, valued flags, boolean flags)."""
    exe, rest = argv[0], argv[1:]
    valued: dict[str, str] = {}
    booleans: set[str] = set()

    i = 0
    while i < len(rest):
        tok = rest[i]
        if not tok.startswith("-"):
            i += 1
            continue
        if tok in BOOLEAN_FLAGS or i + 1 >= len(rest) or rest[i + 1].startswith("-"):
            # A following token that itself starts with "-" is the next flag, not
            # a value -- except for negative numbers, which none of these take.
            booleans.add(tok)
            i += 1
            continue
        valued[tok] = rest[i + 1]
        i += 2
    return exe, valued, booleans


def _command_line_from(stdout: str) -> list[str] | None:
    """Pull the launcher's command line out of its output.

    Not "the first line mentioning llama-server". When the GPUs are busy the
    launcher warns and lists the offending processes, and once a model is
    actually serving, one of those rows names llama-server itself -- so the
    naive match picked a line of `nvidia-smi` output and concluded the launcher
    had passed no flags at all. This test then failed for the sole reason that
    the thing it exists to compare against was running.

    A command line is identified by what it *is* -- a line that parses into
    flags -- rather than by text it happens to contain, so a future warning
    cannot shadow it either.
    """
    for line in stdout.splitlines():
        if "llama-server" not in line.lower():
            continue
        tokens = line.strip().split()
        if len(tokens) < 2:
            continue
        _, valued, _ = _parse(tokens)
        if valued:
            return tokens
    return None


def test_the_launcher_command_line_is_not_confused_with_a_warning() -> None:
    """Runs everywhere, unlike the parity test it protects.

    That test skips without a local launcher, so the regression it hit would be
    invisible in CI. This pins the selector against output of the shape the
    launcher really produces once a model is serving.
    """
    stdout = (
        "[warn] something is already using the GPUs:\n"
        "       17328, C:/llama.cpp/build/bin/Release/llama-server.exe, 8000 MiB\n"
        "       This model wants BOTH cards. Stop ComfyUI / training first.\n"
        "\n"
        "  model    Qwen3.8-27B Unleashed\n"
        "  ctx      65536  (KV q8_0/q8_0)   ub 512   MTP on   vision off\n"
        "\n"
        "C:/llama.cpp/bin/llama-server.exe -m C:/w/x.gguf --ctx-size 65536 -ngl 99\n"
    )
    tokens = _command_line_from(stdout)
    assert tokens is not None
    _, valued, _ = _parse(tokens)
    assert valued["-m"] == "C:/w/x.gguf"
    assert valued["--ctx-size"] == "65536"
    assert valued["-ngl"] == "99"


def test_output_with_no_command_line_yields_nothing_rather_than_junk() -> None:
    assert _command_line_from("[warn] 1, llama-server.exe, 8000 MiB\n") is None


needs_local = pytest.mark.skipif(
    not (REGISTRY.exists() and LAUNCHER.exists() and shutil.which("powershell")),
    # Naming the paths and the overrides, because "skipped" otherwise looks the
    # same whether the launcher is missing or the test is pointed at the wrong
    # place -- and those need different fixes.
    reason=(
        f"needs a shell launcher to compare against: registry {REGISTRY} "
        f"and launcher {LAUNCHER} (override with HEADROOM_PARITY_REGISTRY / "
        "HEADROOM_PARITY_LAUNCHER)"
    ),
)


@needs_local
def test_argv_matches_shell_launcher() -> None:
    registry = load(REGISTRY)
    entry = registry.get()
    mine = build_argv(entry, LLAMA_SERVER)

    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "-DryRun",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    theirs = _command_line_from(proc.stdout or "")
    assert theirs, (
        f"launcher produced no command line.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    _, mine_valued, mine_bool = _parse(mine)
    _, their_valued, their_bool = _parse(theirs)

    only_mine = {k: v for k, v in mine_valued.items() if their_valued.get(k) != v}
    only_theirs = {k: v for k, v in their_valued.items() if mine_valued.get(k) != v}

    assert not only_mine and not only_theirs, (
        "Headroom and the shell launcher disagree.\n"
        f"  differs (headroom): {only_mine}\n"
        f"  differs (launcher): {only_theirs}"
    )
    assert mine_bool == their_bool, (
        f"boolean flags differ: headroom={mine_bool - their_bool} launcher={their_bool - mine_bool}"
    )


@needs_local
def test_micro_batch_and_speculative_decoding_are_coupled() -> None:
    """Raising the micro-batch with MTP on must be refused, not attempted.

    The draft context's compute buffers scale with `-ub`, so a large micro-batch
    fails allocation at startup. Failing here with an explanation beats failing
    there with a raw CUDA out-of-memory message.
    """
    from headroom.registry import RegistryError

    registry = load(REGISTRY)
    entry = registry.get()
    if not entry.serve.get("mtp"):
        pytest.skip("default model does not use speculative decoding")

    with pytest.raises(RegistryError, match="micro-batch"):
        build_argv(entry, LLAMA_SERVER, overrides={"ubatch": 2048})


@needs_local
def test_tensor_split_must_match_device_count() -> None:
    """A mismatched split makes llama-server exit with only a parser error."""
    from headroom.registry import RegistryError

    registry = load(REGISTRY)
    entry = registry.get()

    with pytest.raises(RegistryError, match="ratio"):
        build_argv(entry, LLAMA_SERVER, overrides={"split": "0.5,0.3,0.2"})
