"""Finding things on a machine that is not the author's.

Headroom's first version hardcoded one developer's directory layout as its
defaults. That works exactly once, on one computer, and everywhere else it
produces an app that starts, looks healthy, and can do nothing — the worst
possible first impression for a tool someone just cloned.

Resolution order, highest priority first:

1. An explicit argument (``--registry``, ``--llama-server``)
2. An environment variable (``HEADROOM_REGISTRY``, ``HEADROOM_LLAMA_SERVER``)
3. The config file, ``config.toml`` in the platform config directory
4. Discovery: the executable search path, then conventional install locations
5. For the registry only: create a starter one, so a fresh clone has somewhere
   to put its first model rather than erroring until the user reads the source

Every step is reported, because "it found nothing" and "it found the wrong
thing" need different fixes and look identical from the outside.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

APP_NAME = "headroom"

# Where a llama.cpp build usually ends up, by platform. Ordered by how likely
# each is to be the one the user actually means.
_LLAMA_CANDIDATES_WINDOWS = (
    r"C:\src\llama.cpp\build\bin\Release\llama-server.exe",
    r"C:\llama.cpp\build\bin\Release\llama-server.exe",
    r"C:\Program Files\llama.cpp\llama-server.exe",
)
_LLAMA_CANDIDATES_POSIX = (
    "~/llama.cpp/build/bin/llama-server",
    "~/src/llama.cpp/build/bin/llama-server",
    "/usr/local/bin/llama-server",
    "/opt/llama.cpp/build/bin/llama-server",
)


def config_dir() -> Path:
    """Per-user config directory.

    On Windows this is ``%LOCALAPPDATA%`` rather than anywhere under
    ``Documents``: Controlled Folder Access protects Documents by default and
    blocks writes there *silently*, which turns a config save into a bug report
    about settings that never persist.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / APP_NAME


def config_file() -> Path:
    return config_dir() / "config.toml"


def data_dir() -> Path:
    """Where Headroom keeps things it owns: a starter registry, weights, logs."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / APP_NAME


def _read_config() -> dict[str, str]:
    path = config_file()
    if not path.exists():
        return {}
    try:
        import tomllib

        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:  # noqa: BLE001 - a broken config must not be fatal
        log.warning("ignoring unreadable config %s: %s", path, exc)
        return {}
    paths = data.get("paths")
    return {k: str(v) for k, v in paths.items()} if isinstance(paths, dict) else {}


def _env(name: str) -> str | None:
    """Environment variables, with surrounding whitespace removed.

    A trailing space is easy to produce on Windows (``set VAR=x && cmd`` in
    cmd.exe captures the space before the ``&&``) and nearly invisible: the file
    still opens, but every path derived from it inherits the space.
    """
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else None


@dataclass(slots=True)
class Resolution:
    """A resolved path, and how it was found — so failures are diagnosable."""

    path: Path | None
    source: str
    exists: bool
    searched: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "path": str(self.path) if self.path else None,
            "source": self.source,
            "exists": self.exists,
            "searched": list(self.searched),
        }


def resolve_llama_server(explicit: str | None = None) -> Resolution:
    if explicit:
        p = Path(explicit.strip())
        return Resolution(p, "argument", p.exists())

    if env := _env("HEADROOM_LLAMA_SERVER"):
        p = Path(env)
        return Resolution(p, "HEADROOM_LLAMA_SERVER", p.exists())

    cfg = _read_config()
    if cfg.get("llama_server"):
        p = Path(cfg["llama_server"])
        return Resolution(p, f"config file ({config_file()})", p.exists())

    # On PATH is the strongest signal: the user installed it deliberately.
    if found := shutil.which("llama-server"):
        return Resolution(Path(found), "found on PATH", True)

    candidates = _LLAMA_CANDIDATES_WINDOWS if sys.platform == "win32" else _LLAMA_CANDIDATES_POSIX
    searched = []
    for c in candidates:
        p = Path(os.path.expanduser(c))
        searched.append(str(p))
        if p.exists():
            return Resolution(p, "found in a conventional location", True, tuple(searched))

    return Resolution(None, "not found", False, tuple(searched))


def resolve_registry(explicit: str | None = None, *, create: bool = False) -> Resolution:
    if explicit:
        p = Path(explicit.strip())
        return Resolution(p, "argument", p.exists())

    if env := _env("HEADROOM_REGISTRY"):
        p = Path(env)
        return Resolution(p, "HEADROOM_REGISTRY", p.exists())

    cfg = _read_config()
    if cfg.get("registry"):
        p = Path(cfg["registry"])
        return Resolution(p, f"config file ({config_file()})", p.exists())

    # Only locations that mean something on *any* machine. An earlier version
    # listed the original author's own directory here, which is the same
    # works-on-one-computer problem this module exists to remove -- and it
    # silently wins over creating a starter registry for everyone who happens to
    # share that layout. Users with a registry elsewhere point at it with
    # --registry, the environment, or the config file, all of which rank higher.
    searched = []
    for candidate in (data_dir() / "models.json", Path.cwd() / "models.json"):
        searched.append(str(candidate))
        if candidate.exists():
            return Resolution(candidate, "found in a conventional location", True, tuple(searched))

    default = data_dir() / "models.json"
    if create:
        write_starter_registry(default)
        return Resolution(default, "created (no registry existed)", True, tuple(searched))

    return Resolution(default, "not found", False, tuple(searched))


STARTER_REGISTRY = """{
  "schema": 1,
  "_comment": [
    "Headroom model registry. This file is the single source of truth for model",
    "settings -- both this application and any launch scripts you write should",
    "read it, so the two cannot drift into disagreeing about what is running.",
    "",
    "Add a model through the UI (probe a quant, then 'Add to registry'), or copy",
    "_template below and fill it in by hand.",
    "",
    "One rule worth stating up front: TUNING DOES NOT TRANSFER BETWEEN",
    "ARCHITECTURES. A micro-batch or context size that is optimal for one model",
    "can be the value that fails on another, because the bottleneck moves. Do not",
    "copy a serve block from a model of a different architecture."
  ],
  "default": "",
  "models": {
    "_template": {
      "_use": [
        "Copy this block, rename it, and fill it in. Keys starting with _ are",
        "ignored by the loader."
      ],
      "label": "",
      "repo": "",
      "file": "",
      "mmproj": null,
      "dir": "",
      "size_gib": 0,
      "arch": "",
      "license": null,
      "uncensored": false,
      "why_this_build": [],
      "serve": {
        "ctx": 8192,
        "ubatch": 512,
        "batch": 2048,
        "ngl": 99,
        "devices": "",
        "split": "",
        "flash_attn": "on",
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "cache_ram": 8192,
        "parallel": 1,
        "mtp": false,
        "jinja": true,
        "chat_template_file": null,
        "sampling": {
          "temp": 0.7,
          "top_p": 0.95,
          "top_k": 40,
          "min_p": 0.05,
          "presence_penalty": 0.0
        }
      },
      "vision": { "supported": false, "on_demand": true, "ctx": 4096, "split": "", "image_min_tokens": 1024 },
      "measured": { "status": "NOT MEASURED -- every number here is a guess until you benchmark it" },
      "verified": { "header_probed": false, "loads": false, "benched": false, "needle_tested": false }
    }
  }
}
"""


def write_starter_registry(path: Path) -> None:
    """Create a registry with nothing but a template.

    Deliberately conservative: a small context, no tensor split, no speculative
    decoding. These are not good settings, they are *safe* ones — they will load
    on modest hardware, and the registry says plainly that nothing in it has been
    measured. Shipping optimistic defaults would produce a config that looks
    authoritative and fails on the first real model.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STARTER_REGISTRY, encoding="utf-8")
    log.info("created a starter registry at %s", path)


def save_config(registry: Path | None = None, llama_server: Path | None = None) -> Path:
    """Persist resolved paths so discovery does not have to run again."""
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Headroom configuration.", "", "[paths]"]
    if registry:
        lines.append(f'registry = "{str(registry).replace(chr(92), "/")}"')
    if llama_server:
        lines.append(f'llama_server = "{str(llama_server).replace(chr(92), "/")}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
