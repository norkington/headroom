"""Inspect a GGUF's tensor table — locally, or on a remote host without
downloading it.

This is the part of Headroom that has no equivalent elsewhere, and it exists
because **two builds of the same model at the same file size are not the same
quality, and the difference is invisible from the repo page.** It is visible in
the tensor table, and the tensor table sits at the *front* of a GGUF — so a
ranged request for the first few megabytes reads every tensor name and quant
type without transferring the weights.

The practical payoff: you can tell whether a 15 GiB download is worth making
before you make it.

What the analysis looks for
--------------------------

**A speculative-decoding head.** Some conversions silently drop it. Its absence
costs a large fraction of decode throughput and is not mentioned on most model
cards.

**How the recurrent layers were quantized.** On hybrid architectures most blocks
carry recurrent state rather than attention. That state is *carried forward*, so
quantization error in those layers accumulates with sequence length, while
feed-forward error is per-token and does not compound. A build that spends its
bits protecting `ssm_*` will hold up at long context where an all-4-bit build of
identical size degrades — and file size alone cannot tell you which you have.

**Whether it fits.** Compared against real free VRAM when the caller supplies
it, because "15.4 GiB" only means something relative to what you actually have.
"""

from __future__ import annotations

import logging
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import httpx

log = logging.getLogger(__name__)

# ggml type enum -> name. Only the ones that appear in shipped quants.
GGML_TYPES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
}

# Fixed-width metadata value types, by GGUF type id.
_FIXED_WIDTH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

# Start small and grow. A 4B model's table fits in a couple of MB; a 70B needs
# more. Fetching 48 MB unconditionally wastes bandwidth on most models.
FETCH_STEPS_MIB = (8, 24, 64)

# Names indicating a speculative-decoding / multi-token-prediction head.
_MTP_PATTERN = re.compile(r"nextn|\bmtp\b|eh_proj", re.IGNORECASE)

# Tensor family -> the substring identifying it.
_FAMILIES = {
    "recurrent": ".ssm_",
    "attention": ".attn_",
    "feed_forward": ".ffn_",
}

# Quant types considered "high precision" when judging how a family was treated.
_HIGH_PRECISION = {"F32", "F16", "BF16", "Q8_0", "Q6_K"}
_LOW_PRECISION = {
    "Q2_K",
    "Q3_K",
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ3_XXS",
    "IQ3_S",
}


class GgufError(RuntimeError):
    pass


class _Truncated(Exception):
    """The fetched prefix was too small to contain the whole tensor table."""


@dataclass(slots=True)
class Finding:
    """One interpreted observation. `level` drives presentation, not severity."""

    level: str  # "good" | "caution" | "info"
    title: str
    detail: str


@dataclass(slots=True)
class GgufAnalysis:
    source: str
    architecture: str = ""
    name: str = ""
    tensor_count: int = 0
    file_size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    mtp_tensors: list[str] = field(default_factory=list)
    families: dict[str, dict[str, int]] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    bytes_read: int = 0

    @property
    def has_mtp(self) -> bool:
        return bool(self.mtp_tensors)

    @property
    def size_gib(self) -> float | None:
        if self.file_size_bytes is None:
            return None
        return self.file_size_bytes / 1024**3


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read(f, 4))[0]


def _u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read(f, 8))[0]


def _read(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise _Truncated
    return b


def _string(f: BinaryIO) -> str:
    return _read(f, _u64(f)).decode("utf-8", "replace")


def _skip_value(f: BinaryIO, type_id: int) -> None:
    """Advance past a metadata value without materialising it.

    Metadata can contain very large arrays — token lists especially — and
    reading them costs memory for nothing. Only the handful of scalar keys the
    caller asks for get decoded.
    """
    if type_id in _FIXED_WIDTH:
        f.seek(_FIXED_WIDTH[type_id], 1)
    elif type_id == 8:  # string
        f.seek(_u64(f), 1)
    elif type_id == 9:  # array
        item_type = _u32(f)
        count = _u64(f)
        if item_type == 8:
            for _ in range(count):
                f.seek(_u64(f), 1)
        elif item_type in _FIXED_WIDTH:
            f.seek(_FIXED_WIDTH[item_type] * count, 1)
        elif item_type == 9:
            raise GgufError("nested metadata arrays are not supported")
        else:
            raise GgufError(f"unknown array item type {item_type}")
    else:
        raise GgufError(f"unknown metadata type {type_id}")


def _read_scalar(f: BinaryIO, type_id: int) -> Any:
    fmt = {
        0: "<b",
        1: "<B",
        2: "<h",
        3: "<H",
        4: "<i",
        5: "<I",
        6: "<f",
        7: "<?",
        10: "<q",
        11: "<Q",
        12: "<d",
    }.get(type_id)
    if fmt is None:
        return None
    return struct.unpack(fmt, _read(f, _FIXED_WIDTH[type_id]))[0]


def parse(buffer: BinaryIO, source: str, file_size: int | None = None) -> GgufAnalysis:
    """Parse a GGUF header and tensor table. Reads no tensor data."""
    if _read(buffer, 4) != b"GGUF":
        raise GgufError(
            "not a GGUF file. A gated or private repository returns an error page "
            "here instead of weights — check the model page in a browser."
        )
    _u32(buffer)  # version
    tensor_count = _u64(buffer)
    kv_count = _u64(buffer)

    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = _string(buffer)
        type_id = _u32(buffer)
        if type_id in _FIXED_WIDTH:
            metadata[key] = _read_scalar(buffer, type_id)
        elif type_id == 8:
            metadata[key] = _string(buffer)
        else:
            _skip_value(buffer, type_id)

    rows: list[tuple[str, str]] = []
    for _ in range(tensor_count):
        name = _string(buffer)
        n_dims = _u32(buffer)
        buffer.seek(8 * n_dims, 1)
        type_id = _u32(buffer)
        _u64(buffer)  # offset
        rows.append((name, GGML_TYPES.get(type_id, f"type{type_id}")))

    arch = str(metadata.get("general.architecture", "") or "")
    analysis = GgufAnalysis(
        source=source,
        architecture=arch,
        name=str(metadata.get("general.name", "") or ""),
        tensor_count=len(rows),
        file_size_bytes=file_size,
        metadata={
            k: v
            for k, v in metadata.items()
            if k.startswith("general.")
            or k.endswith((".block_count", ".context_length", ".attention.head_count_kv"))
        },
        mtp_tensors=[n for n, _ in rows if _MTP_PATTERN.search(n)],
    )

    for family, marker in _FAMILIES.items():
        counts = Counter(t for n, t in rows if marker in n)
        if counts:
            analysis.families[family] = dict(counts.most_common())

    return analysis


# --------------------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------------------


def _quantized_share(counts: dict[str, int]) -> tuple[int, int, int]:
    """(high precision, low precision, total) ignoring F32 norm/bias tensors.

    F32 entries in a family are almost always norms and biases, which are never
    quantized and would otherwise flatter every build equally.
    """
    high = low = total = 0
    for dtype, n in counts.items():
        if dtype == "F32":
            continue
        total += n
        if dtype in _HIGH_PRECISION:
            high += n
        elif dtype in _LOW_PRECISION:
            low += n
    return high, low, total


def interpret(analysis: GgufAnalysis, free_vram_mib: int | None = None) -> None:
    """Attach findings. Mutates `analysis.findings`."""
    findings = analysis.findings

    # --- speculative decoding -------------------------------------------------
    if analysis.has_mtp:
        findings.append(
            Finding(
                "good",
                "Speculative decoding head present",
                f"{len(analysis.mtp_tensors)} tensor(s) for multi-token prediction. "
                "Worth a large fraction of decode throughput, and it needs no separate "
                "draft model.",
            )
        )
    else:
        findings.append(
            Finding(
                "caution",
                "No speculative decoding head",
                "This conversion has no multi-token-prediction tensors. If the base model "
                "ships one, this build dropped it — decode will be substantially slower "
                "than a conversion that kept it.",
            )
        )

    # --- recurrent layers -----------------------------------------------------
    recurrent = analysis.families.get("recurrent")
    if recurrent:
        high, low, total = _quantized_share(recurrent)
        share = (high / total) if total else 0.0
        if share >= 0.5:
            findings.append(
                Finding(
                    "good",
                    "Recurrent layers are protected",
                    f"{high} of {total} quantized recurrent tensors are at high precision "
                    f"({_top(recurrent)}). Recurrent state is carried forward, so error there "
                    "accumulates with context length — spending bits here is what holds "
                    "quality together at long context.",
                )
            )
        elif share == 0.0:
            findings.append(
                Finding(
                    "caution",
                    "Recurrent layers are uniformly low precision",
                    f"All {total} quantized recurrent tensors sit at {_top(recurrent)}. "
                    "Because recurrent state is carried forward, this error compounds with "
                    "sequence length — expect it to degrade at long context relative to a "
                    "build of the same size that protects these layers.",
                )
            )
        else:
            findings.append(
                Finding(
                    "info",
                    "Recurrent layers are mixed precision",
                    f"{high} of {total} quantized recurrent tensors are high precision "
                    f"({_top(recurrent)}).",
                )
            )

    # --- attention ------------------------------------------------------------
    attention = analysis.families.get("attention")
    if attention:
        _, low, total = _quantized_share(attention)
        if total and low / total >= 0.5:
            findings.append(
                Finding(
                    "caution",
                    "Attention layers are aggressively quantized",
                    f"{low} of {total} quantized attention tensors are 3-bit or below "
                    f"({_top(attention)}). On a hybrid architecture only a minority of blocks "
                    "do full attention, which makes them a risky place to economise.",
                )
            )

    # --- fit ------------------------------------------------------------------
    if analysis.file_size_bytes and free_vram_mib is not None:
        weights_mib = analysis.file_size_bytes / 1024**2
        margin = free_vram_mib - weights_mib
        if margin < 0:
            findings.append(
                Finding(
                    "caution",
                    "Will not fit in free VRAM",
                    f"Weights alone are {weights_mib / 1024:.2f} GiB against "
                    f"{free_vram_mib / 1024:.2f} GiB free — short by "
                    f"{-margin / 1024:.2f} GiB, before any KV cache.",
                )
            )
        elif margin < 2048:
            findings.append(
                Finding(
                    "caution",
                    "Fits, but leaves little for KV cache",
                    f"Weights are {weights_mib / 1024:.2f} GiB, leaving {margin:.0f} MiB. "
                    "Context still has to come out of that, so a large window may not fit.",
                )
            )
        else:
            findings.append(
                Finding(
                    "good",
                    "Fits in free VRAM",
                    f"Weights are {weights_mib / 1024:.2f} GiB, leaving about "
                    f"{margin / 1024:.2f} GiB for KV cache and compute buffers.",
                )
            )


def _top(counts: dict[str, int], n: int = 4) -> str:
    """Describe a family's quantization, omitting F32.

    F32 entries are norms and biases, never quantized, and they are excluded
    from the ratios above -- so listing them here would describe a distribution
    the numbers alongside do not refer to.
    """
    quantized = {k: v for k, v in counts.items() if k != "F32"}
    if not quantized:
        return "unquantized"
    return " ".join(f"{v}x{k}" for k, v in list(quantized.items())[:n])


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def probe_local(path: str | Path, free_vram_mib: int | None = None) -> GgufAnalysis:
    p = Path(path)
    if not p.exists():
        raise GgufError(f"file not found: {p}")
    size = p.stat().st_size
    with p.open("rb") as f:
        try:
            analysis = parse(f, source=str(p), file_size=size)
        except _Truncated as exc:
            raise GgufError(f"{p} is truncated or not a complete GGUF") from exc
    analysis.bytes_read = size
    interpret(analysis, free_vram_mib)
    return analysis


async def probe_remote(
    repo: str,
    filename: str,
    *,
    free_vram_mib: int | None = None,
    endpoint: str = "https://huggingface.co",
    timeout: float = 60.0,
) -> GgufAnalysis:
    """Read a remote GGUF's tensor table with ranged requests.

    Grows the fetch window rather than guessing a single size: most models need
    only a few MB, and a fixed large request wastes bandwidth on all of them.
    """
    # Normalised here rather than at the call site, so every route into a probe
    # accepts what a person pastes. A URL left unparsed would be interpolated
    # straight into the resolve path and fail as a 404 on a file that exists.
    repo = normalise_repo(repo)
    url = f"{endpoint}/{repo}/resolve/main/{filename}"
    total_size: int | None = None
    last_error: Exception | None = None

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        head = await client.head(url)
        if head.status_code == 401:
            raise GgufError(
                f"{repo} is gated. Accept its terms on the model page and provide a token; "
                "Headroom will not work around access controls."
            )
        if head.status_code == 404:
            raise GgufError(f"not found: {repo}/{filename}")
        if head.is_success and "content-length" in head.headers:
            total_size = int(head.headers["content-length"])

        for mib in FETCH_STEPS_MIB:
            want = mib * 1024 * 1024
            if total_size is not None and want > total_size:
                want = total_size
            resp = await client.get(url, headers={"Range": f"bytes=0-{want - 1}"})
            if resp.status_code not in (200, 206):
                raise GgufError(f"HTTP {resp.status_code} fetching {filename}")
            data = resp.content
            try:
                analysis = parse(BytesIO(data), source=f"{repo}/{filename}", file_size=total_size)
            except _Truncated as exc:
                last_error = exc
                if total_size is not None and len(data) >= total_size:
                    raise GgufError("file ended before the tensor table did") from exc
                continue
            analysis.bytes_read = len(data)
            interpret(analysis, free_vram_mib)
            return analysis

    raise GgufError(
        f"tensor table did not fit in {FETCH_STEPS_MIB[-1]} MiB — unusually large metadata"
    ) from last_error


# Multimodal projectors ship alongside the weights in the same repository and are
# ordinary .gguf files, so they arrive mixed into the quant list. They are named
# by a convention llama.cpp's own conversion tooling follows.
PROJECTOR_MARKER = "mmproj"

# Preferred projector precision, best first. f16 rather than f32 because a
# projector is a few hundred MiB against a model's tens of gigabytes -- the extra
# precision buys nothing measurable and the VRAM it costs comes straight out of
# the context budget, which on a constrained box is the scarce thing. Quantized
# projectors are last: the saving is small and image fidelity is what pays.
_PROJECTOR_PRECISION = ("f16", "bf16", "f32", "q8_0")


def is_projector(filename: str) -> bool:
    """Whether this .gguf is a vision projector rather than model weights.

    Matched on the filename because that is what the convention actually is.
    Size would be a tempting proxy -- projectors are far smaller -- but a small
    quant of a small model is smaller still, and mistaking weights for a
    projector is worse than not detecting one.
    """
    return PROJECTOR_MARKER in filename.lower()


def choose_projector(files: list[dict[str, Any]]) -> str | None:
    """Pick the projector to pair with a model, or None if the repo has none.

    A suggestion, not a decision -- the caller is expected to show it and let it
    be changed. Repositories ship several precisions and which one is wanted is
    a judgement about VRAM, not something a filename can settle.
    """
    projectors = [f for f in files if f.get("kind") == "projector"]
    if not projectors:
        return None

    def rank(f: dict[str, Any]) -> tuple[int, float]:
        name = f["filename"].lower()
        for i, token in enumerate(_PROJECTOR_PRECISION):
            if token in name:
                return (i, -(f.get("size_bytes") or 0))
        # Unrecognised precision sorts after the known ones, larger first, on the
        # assumption that bigger means less quantized.
        return (len(_PROJECTOR_PRECISION), -(f.get("size_bytes") or 0))

    return min(projectors, key=rank)["filename"]


# What a repository identifier looks like once the noise is off it. The owner is
# optional: the hub's canonical models predate namespaces and are still served
# as a bare name -- `gpt2`, `bert-base-uncased`. Requiring owner/name rejected
# those as "not a HuggingFace repository", which is simply false.
_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*(/[A-Za-z0-9][\w.-]*)?$")

# Everything a hub URL can carry after the repository name.
_REPO_SUFFIXES = ("tree", "blob", "resolve", "raw", "commit", "discussions")


def normalise_repo(raw: str) -> str:
    """Turn what people actually paste into ``owner/name``.

    Nobody types a repository identifier. They copy the address bar, and every
    form of that used to fail in a way that blamed them or blamed the hub:

    - a full URL reported "repository not found", pointing at the repo rather
      than at the input that was never a repo name;
    - a URL with ``/tree/main`` on it made the hub return a JSON *list* instead
      of an object, which crashed on ``.get("siblings")`` and surfaced as
      "could not reach the hub" -- an outage message for a typo;
    - a trailing space, which is free when pasting, produced a 401 and the
      advice to go and accept the repo's terms.

    So the parsing happens here, once, before anything is fetched.
    """
    repo = (raw or "").strip()
    repo = repo.split("?", 1)[0].split("#", 1)[0]
    repo = re.sub(r"^https?://", "", repo, flags=re.IGNORECASE)
    repo = re.sub(r"^(www\.)?(huggingface\.co|hf\.co)/", "", repo, flags=re.IGNORECASE)
    repo = re.sub(r"^models/", "", repo, flags=re.IGNORECASE)
    repo = repo.strip("/")

    # owner/name/tree/main -> owner/name, and gpt2/tree/main -> gpt2. Cut at the
    # first known hub path segment rather than at a fixed offset, so both the
    # namespaced and the bare-name forms survive. Only at a *known* segment:
    # truncating blindly would quietly accept nonsense as success.
    parts = repo.split("/")
    for i, part in enumerate(parts):
        if i > 0 and part.lower() in _REPO_SUFFIXES:
            repo = "/".join(parts[:i])
            break

    if not _REPO_RE.match(repo):
        raise GgufError(
            f"{raw.strip()!r} is not a HuggingFace repository. Expected owner/name, "
            "for example unsloth/Qwen3-8B-GGUF -- a full model-page URL works too."
        )
    return repo


async def list_repo_files(
    repo: str,
    *,
    endpoint: str = "https://huggingface.co",
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """List the GGUF files in a repository, largest first."""
    repo = normalise_repo(repo)
    url = f"{endpoint}/api/models/{repo}"
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url, params={"blobs": "true"})
        # 401 and 404 are the SAME answer from this hub. A private or missing
        # repository both come back 401, deliberately -- otherwise the status
        # code would confirm that a private repo exists. So the message has to
        # cover both, and the old one ("is gated; accept its terms") sent people
        # to a terms page for repositories that had simply been mistyped.
        if resp.status_code in (401, 403, 404):
            raise GgufError(
                f"{repo} could not be read. Either it does not exist -- check the spelling -- "
                "or it is private or gated, in which case accept its terms on the model page "
                "while signed in."
            )
        resp.raise_for_status()
        payload = resp.json()

    # The hub returns an object for a repository and a list for a search. Asking
    # for a URL that is not a repository lands in the second case, and reaching
    # blindly for .get() raised an AttributeError that the API reported as
    # "could not reach the hub" -- describing an outage that had not happened.
    if not isinstance(payload, dict):
        raise GgufError(
            f"{repo} did not come back as a repository. Check that owner/name is right."
        )

    files = [
        {
            "filename": s["rfilename"],
            "size_bytes": s.get("size"),
            "size_gib": round((s.get("size") or 0) / 1024**3, 3),
            "kind": "projector" if is_projector(s["rfilename"]) else "model",
        }
        for s in payload.get("siblings", [])
        if s.get("rfilename", "").endswith(".gguf")
    ]
    files.sort(key=lambda f: f["size_bytes"] or 0, reverse=True)
    return files
