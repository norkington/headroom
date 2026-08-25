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
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from headroom.registry import build_argv, load

REGISTRY = Path(r"C:\AI\models\models.json")
LAUNCHER = Path(r"C:\AI\models\bin\serve.ps1")
LLAMA_SERVER = Path(r"C:\src\llama.cpp\build\bin\Release\llama-server.exe")

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


needs_local = pytest.mark.skipif(
    not (REGISTRY.exists() and LAUNCHER.exists() and shutil.which("powershell")),
    reason="local registry / shell launcher not present",
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
    line = next(
        (ln for ln in (proc.stdout or "").splitlines() if "llama-server" in ln.lower()),
        None,
    )
    assert line, (
        f"launcher produced no command line.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    theirs = line.strip().split()

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
