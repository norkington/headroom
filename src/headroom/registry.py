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
        "-m", str(model_path),
        "--ctx-size", str(ctx),
        "-ngl", str(s.get("ngl", 99)),
        "-fa", str(s.get("flash_attn", "on")),
        "-ub", str(ubatch),
        "-b", str(batch),
        "-np", str(s.get("parallel", 1)),
        "--cache-ram", str(s.get("cache_ram", 8192)),
        "-ctk", str(s.get("cache_type_k", "f16")),
        "-ctv", str(s.get("cache_type_v", "f16")),
        "--host", "127.0.0.1",
        "--port", str(port),
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
