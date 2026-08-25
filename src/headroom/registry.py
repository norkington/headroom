"""The model registry.

Headroom does not invent a config format. It reads and writes the same
``models.json`` your launch scripts already use, because **two sources of truth
is how a CLI and a GUI drift into disagreeing about what is running**. If you
edit a value here it must be the same value the shell script picks up, and the
only way to guarantee that is to share the file.

`build_argv` is therefore held to a strict contract: given a registry entry, it
must produce the same llama-server command line the shell launcher would. Any
divergence is a bug, not a feature — a UI that quietly launches with different
flags than the CLI is worse than no UI, because the measurements stop comparing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Entries beginning with this are documentation or templates, never runnable.
PRIVATE_PREFIX = "_"


class RegistryError(RuntimeError):
    pass


@dataclass(slots=True)
class ModelEntry:
    """One registry entry, plus whatever provenance it carries."""

    key: str
    label: str
    repo: str
    file: str
    directory: str
    size_gib: float
    arch: str
    mmproj: str | None = None
    license: str | None = None
    uncensored: bool = False
    why_this_build: list[str] = field(default_factory=list)
    serve: dict[str, Any] = field(default_factory=dict)
    vision: dict[str, Any] = field(default_factory=dict)
    measured: dict[str, Any] = field(default_factory=dict)
    verified: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.directory) / self.file

    @property
    def installed(self) -> bool:
        return self.path.exists()

    @property
    def mmproj_path(self) -> Path | None:
        return Path(self.directory) / self.mmproj if self.mmproj else None

    @property
    def measured_on_this_file(self) -> bool:
        """Whether the numbers were measured on THIS artifact or inherited.

        Surfaced in the UI because an inherited number presented as measured is
        the kind of quiet dishonesty that makes a whole dashboard untrustworthy.
        """
        status = str(self.measured.get("status", "")).lower()
        return status.startswith("measured")


@dataclass(slots=True)
class Registry:
    path: Path
    default: str
    models: dict[str, ModelEntry]
    raw: dict[str, Any]

    def get(self, key: str | None = None) -> ModelEntry:
        key = key or self.default
        if key not in self.models:
            known = ", ".join(self.models) or "(none)"
            raise RegistryError(f"unknown model {key!r}. Known: {known}")
        return self.models[key]


def load(path: str | Path) -> Registry:
    path = Path(path)
    if not path.exists():
        raise RegistryError(f"registry not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{path} is not valid JSON: {exc}") from exc

    models: dict[str, ModelEntry] = {}
    for key, m in (raw.get("models") or {}).items():
        if key.startswith(PRIVATE_PREFIX):
            continue
        try:
            models[key] = ModelEntry(
                key=key,
                label=m.get("label", key),
                repo=m.get("repo", ""),
                file=m.get("file", ""),
                directory=m.get("dir", ""),
                size_gib=float(m.get("size_gib", 0) or 0),
                arch=m.get("arch", ""),
                mmproj=m.get("mmproj"),
                license=m.get("license"),
                uncensored=bool(m.get("uncensored", False)),
                why_this_build=list(m.get("why_this_build") or []),
                serve=dict(m.get("serve") or {}),
                vision=dict(m.get("vision") or {}),
                measured=dict(m.get("measured") or {}),
                verified=dict(m.get("verified") or {}),
            )
        except (TypeError, ValueError) as exc:
            log.warning("skipping malformed registry entry %r: %s", key, exc)

    default = raw.get("default") or next(iter(models), "")
    return Registry(path=path, default=default, models=models, raw=raw)


def build_argv(
    entry: ModelEntry,
    llama_server_exe: str | Path,
    *,
    port: int = 8080,
    vision: bool = False,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    """Build the llama-server command line for an entry.

    Mirrors the shell launcher exactly. Where a value is load-bearing for a
    non-obvious reason, the reason is recorded here so nobody "simplifies" it
    later:

    - **`-ub` and MTP are a package deal.** Speculative decoding builds a second
      context whose compute buffers scale with the micro-batch. Raising `-ub`
      with MTP enabled can fail allocation outright at startup, so they cannot be
      tuned independently.

    - **`--jinja` versus `--chat-template-file`.** Most GGUFs embed their chat
      template, and passing an external file then overrides the correct one with
      a guess. The registry sets `chat_template_file` only for models that
      genuinely ship a separate template.

    - **`--cache-ram` is a HOST RAM prompt cache**, not VRAM. It does not compete
      with the model for GPU memory, but on a model that keeps tens of GiB of
      experts in system RAM it absolutely competes there.
    """
    s = dict(entry.serve)
    if overrides:
        s.update(overrides)

    model_path = entry.path
    if not model_path.exists():
        raise RegistryError(f"model file missing: {model_path}")

    ctx = int(s.get("ctx", 4096))
    split = s.get("split") or ""
    devices = s.get("devices") or ""

    # The vision profile is a different operating point, not a flag. It trades
    # context and speed for the VRAM the projector needs, and the registry
    # carries its own ctx and tensor split for exactly that reason.
    if vision:
        if not entry.vision.get("supported"):
            raise RegistryError(f"{entry.key} has no vision support in the registry")
        if entry.vision.get("ctx") and not (overrides or {}).get("ctx"):
            ctx = int(entry.vision["ctx"])
        if entry.vision.get("split") and not (overrides or {}).get("split"):
            split = entry.vision["split"]

    ubatch = int(s.get("ubatch", 512))
    batch = max(int(s.get("batch", 2048)), ubatch)

    argv: list[str] = [
        str(llama_server_exe),
        "-m",
        str(model_path),
        "--ctx-size",
        str(ctx),
        "-ngl",
        str(s.get("ngl", 99)),
        "-fa",
        str(s.get("flash_attn", "on")),
        "-ub",
        str(ubatch),
        "-b",
        str(batch),
        "-np",
        str(s.get("parallel", 1)),
        "--cache-ram",
        str(s.get("cache_ram", 8192)),
        "-ctk",
        str(s.get("cache_type_k", "f16")),
        "-ctv",
        str(s.get("cache_type_v", "f16")),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    if devices:
        argv += ["-dev", devices]

    if split:
        # One ratio per device or llama-server refuses to load, reporting only a
        # parser error. Catch it here where we can say something useful.
        n_ratios = len([x for x in split.split(",") if x.strip()])
        n_devices = len([x for x in devices.split(",") if x.strip()]) if devices else 0
        if n_devices and n_ratios != n_devices:
            raise RegistryError(
                f"tensor split {split!r} has {n_ratios} ratio(s) but {n_devices} device(s) "
                "are selected; they must match"
            )
        argv += ["-ts", split]

    if s.get("jinja", True):
        argv.append("--jinja")
    if s.get("chat_template_file"):
        argv += ["--chat-template-file", str(s["chat_template_file"])]

    sampling = s.get("sampling") or {}
    for flag, key in (
        ("--temp", "temp"),
        ("--top-p", "top_p"),
        ("--top-k", "top_k"),
        ("--min-p", "min_p"),
        ("--presence-penalty", "presence_penalty"),
    ):
        if key in sampling:
            argv += [flag, str(sampling[key])]

    if s.get("mtp"):
        if ubatch > 512:
            raise RegistryError(
                f"micro-batch {ubatch} with speculative decoding enabled: the draft context's "
                "compute buffers scale with -ub and this is likely to fail allocation at "
                "startup. Use 512, or disable mtp."
            )
        argv += ["--spec-type", "draft-mtp"]

    if vision:
        mmproj = entry.mmproj_path
        if not mmproj or not mmproj.exists():
            raise RegistryError(f"projector missing: {mmproj}")
        argv += ["--mmproj", str(mmproj)]
        if entry.vision.get("image_min_tokens"):
            argv += ["--image-min-tokens", str(entry.vision["image_min_tokens"])]

    return argv


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def derive_entry(
    *,
    key: str,
    label: str,
    repo: str,
    filename: str,
    directory: str,
    size_gib: float,
    architecture: str,
    has_mtp: bool,
    template: dict[str, Any] | None = None,
    inherit_from: ModelEntry | None = None,
) -> dict[str, Any]:
    """Build a registry entry from a probe result.

    Two rules govern what goes in the ``serve`` block, and both exist because
    getting them wrong produces a config that looks authoritative and is not:

    **Tuning does not transfer across architectures.** A serve block is only
    inherited when the architecture matches exactly. Otherwise the template's
    conservative defaults are used, because a micro-batch or context size tuned
    for one architecture can be actively wrong for another -- the bottleneck
    moves, and the number that was optimal becomes the number that fails.

    **Inherited is not measured.** Even on an architecture match, ``measured``
    records that the figures came from somewhere else and were never observed on
    this file, and every ``verified`` flag stays false. A number presented as
    measured when it was copied is the kind of quiet dishonesty that makes an
    entire registry untrustworthy.

    ``mtp`` is the one setting derived from evidence rather than inherited: the
    probe read the tensor table, so whether a speculative-decoding head exists is
    a fact about this file, not a guess.
    """
    if inherit_from is not None and inherit_from.arch == architecture:
        serve = dict(inherit_from.serve)
        vision = dict(inherit_from.vision)
        provenance = (
            f"INHERITED from {inherit_from.key!r} (same architecture). NOT measured on this "
            "file -- run a benchmark and replace these values."
        )
    else:
        base = template or {}
        serve = dict(base.get("serve") or {})
        vision = dict(base.get("vision") or {})
        if inherit_from is not None:
            provenance = (
                f"NOT MEASURED. Defaults only -- {inherit_from.key!r} was not inherited because "
                f"its architecture ({inherit_from.arch!r}) differs from {architecture!r}, and "
                "tuning does not transfer across architectures."
            )
        else:
            provenance = "NOT MEASURED. Template defaults only."

    # Derived from the tensor table, so this one is evidence.
    serve["mtp"] = bool(has_mtp)
    if has_mtp:
        # Speculative decoding builds a second context whose compute buffers
        # scale with the micro-batch, so the two cannot be tuned independently.
        serve["ubatch"] = min(int(serve.get("ubatch", 512) or 512), 512)

    return {
        "label": label,
        "repo": repo,
        "file": filename,
        "mmproj": None,
        "dir": directory,
        "size_gib": round(size_gib, 3),
        "arch": architecture,
        "license": None,
        "uncensored": False,
        "why_this_build": [f"Added from a tensor-table probe of {repo}."],
        "serve": serve,
        "vision": vision or {"supported": False},
        "measured": {"status": provenance},
        "verified": {
            "header_probed": True,
            "mtp_tensors": 1 if has_mtp else 0,
            "loads": False,
            "benched": False,
            "needle_tested": False,
        },
    }


def add_entry(
    path: str | Path, key: str, entry: dict[str, Any], *, overwrite: bool = False
) -> None:
    """Add an entry to models.json, preserving everything already there.

    This file is shared with the user's own launch scripts, so it is treated as
    theirs rather than as this application's private state:

    - **A backup is written first**, next to the original, so a bad edit is
      always recoverable.
    - **The write is atomic** -- a temporary file is renamed into place, so a
      crash mid-write cannot leave a half-written registry that neither this app
      nor a shell script can parse.
    - **Existing keys are refused** unless overwrite is explicit. Silently
      replacing an entry would discard measurements someone earned.
    - Comment keys, the template, and unrelated entries are read and written
      back untouched.
    """
    path = Path(path)
    if not path.exists():
        raise RegistryError(f"registry not found: {path}")
    if key.startswith(PRIVATE_PREFIX):
        raise RegistryError(f"{key!r} starts with {PRIVATE_PREFIX!r}, which marks private entries")

    raw = json.loads(path.read_text(encoding="utf-8"))
    models = raw.setdefault("models", {})
    if key in models and not overwrite:
        raise RegistryError(
            f"{key!r} is already in the registry. Choose another name, or pass overwrite "
            "if you mean to replace it and lose its recorded measurements."
        )

    models[key] = entry
    if not raw.get("default"):
        raw["default"] = key

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    log.info("added %r to %s (backup at %s)", key, path, backup.name)
