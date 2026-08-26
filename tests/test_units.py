"""Portable unit tests.

These run anywhere — no GPU, no llama.cpp, no local registry — which is the
point. `test_argv_parity.py` covers the contract that matters most, but it can
only run on a machine with the real shell launcher installed, so in CI it skips.
A pipeline whose only green signal is "lint passed and everything skipped" is
decoration, not evidence.

So these exercise the actual decision logic against synthetic inputs: the
headroom grading, the CUDA-order reconciliation, the server state machine, and
the argv builder's guard rails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.gguf import GgufAnalysis
from headroom.gpu import (
    _DEVICE_LINE,
    HEADROOM_CRITICAL_MIB,
    HEADROOM_TIGHT_MIB,
    CudaMapping,
    Gpu,
    order_differs,
)
from headroom.registry import RegistryError, build_argv, load
from headroom.server import ServerState


def make_gpu(**kw) -> Gpu:
    base = {
        "nvml_index": 0,
        "name": "NVIDIA GeForce RTX 4070 SUPER",
        "memory_total_mib": 12282,
        "memory_used_mib": 1000,
        "memory_free_mib": 11282,
    }
    base.update(kw)
    return Gpu(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------- headroom


@pytest.mark.parametrize(
    ("free", "expected"),
    [
        (0, "critical"),
        (HEADROOM_CRITICAL_MIB - 1, "critical"),
        (HEADROOM_CRITICAL_MIB, "tight"),
        (HEADROOM_TIGHT_MIB - 1, "tight"),
        (HEADROOM_TIGHT_MIB, "ok"),
        (11282, "ok"),
    ],
)
def test_headroom_grading_boundaries(free: int, expected: str) -> None:
    assert make_gpu(memory_free_mib=free).headroom_state == expected


def test_grading_is_per_card_not_aggregate() -> None:
    """The premise of the project in one assertion.

    Two cards can total plenty of free memory while one of them is nearly out.
    Grading on the total would report this pair as healthy.
    """
    roomy = make_gpu(nvml_index=0, memory_free_mib=2200)
    starved = make_gpu(nvml_index=1, name="NVIDIA GeForce RTX 3060", memory_free_mib=400)

    total_free = roomy.memory_free_mib + starved.memory_free_mib
    assert total_free > HEADROOM_TIGHT_MIB, "the aggregate looks fine"
    assert roomy.headroom_state == "ok"
    assert starved.headroom_state == "critical", "but one card is nearly out"


def test_label_never_invents_a_cuda_index() -> None:
    """An unresolved mapping must not be presented as CUDA0.

    Guessing here would be worse than admitting ignorance: the whole reason this
    mapping exists is that the obvious assumption is often wrong.
    """
    unmapped = make_gpu()
    assert "CUDA" not in unmapped.label
    assert "nvml 0" in unmapped.label

    unmapped.cuda_index = 1
    assert "CUDA1" in unmapped.label


# ---------------------------------------------------------------- cuda order


def test_device_line_parses_llama_cpp_output() -> None:
    sample = """
Available devices:
  CUDA0: NVIDIA GeForce RTX 4070 SUPER (12281 MiB, 11069 MiB free)
  CUDA1: NVIDIA GeForce RTX 3060 (12287 MiB, 11253 MiB free)
"""
    matches = list(_DEVICE_LINE.finditer(sample))
    assert [m.group("idx") for m in matches] == ["0", "1"]
    assert matches[0].group("name") == "NVIDIA GeForce RTX 4070 SUPER"
    assert matches[1].group("total") == "12287"


def test_order_differs_detects_the_reversal() -> None:
    """The condition that makes `-dev CUDA0` and `nvidia-smi -i 0` disagree."""
    same = CudaMapping(cuda_to_nvml={0: 0, 1: 1})
    reversed_ = CudaMapping(cuda_to_nvml={0: 1, 1: 0})

    assert order_differs(same) is False
    assert order_differs(reversed_) is True


def test_unresolved_mapping_is_not_treated_as_agreement() -> None:
    """An empty mapping means "unknown", which must not read as "they agree"."""
    empty = CudaMapping()
    assert empty.resolved is False
    assert order_differs(empty) is False  # nothing known, so nothing claimed


# ---------------------------------------------------------------- server state


def test_status_distinguishes_loading_from_stopped() -> None:
    """Loading must not read as stopped, or the user starts a second server."""
    assert ServerState().status == "stopped"
    assert ServerState(running=True, pid=123).status == "loading"
    assert ServerState(running=True, pid=123, reachable=True).status == "running"


def test_status_orphaned_when_reachable_process_is_unknown() -> None:
    """Something answers the port but no matching process was found."""
    assert ServerState(running=True, pid=None).status == "orphaned"


# ---------------------------------------------------------------- argv


@pytest.fixture
def fake_registry(tmp_path: Path) -> Path:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "fake.gguf").write_bytes(b"GGUF")
    (weights / "mmproj.gguf").write_bytes(b"GGUF")

    doc = {
        "default": "demo",
        "models": {
            "_template": {"label": "ignored"},
            "demo": {
                "label": "Demo Model",
                "repo": "example/demo-GGUF",
                "file": "fake.gguf",
                "mmproj": "mmproj.gguf",
                "dir": str(weights).replace("\\", "/"),
                "size_gib": 1.0,
                "arch": "demo-arch",
                "serve": {
                    "ctx": 8192,
                    "ubatch": 512,
                    "batch": 2048,
                    "ngl": 99,
                    "devices": "CUDA0,CUDA1",
                    "split": "",
                    "flash_attn": "on",
                    "cache_type_k": "q8_0",
                    "cache_type_v": "q8_0",
                    "cache_ram": 32768,
                    "parallel": 1,
                    "mtp": True,
                    "jinja": True,
                    "chat_template_file": None,
                    "sampling": {"temp": 1.0, "top_p": 0.95},
                },
                "vision": {
                    "supported": True,
                    "ctx": 4096,
                    "split": "0.4,0.6",
                    "image_min_tokens": 1024,
                },
                "measured": {"status": "MEASURED on this file"},
                "verified": {},
            },
        },
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_template_entries_are_not_runnable(fake_registry: Path) -> None:
    reg = load(fake_registry)
    assert "_template" not in reg.models
    assert reg.default == "demo"


def test_argv_carries_the_registry_values(fake_registry: Path) -> None:
    reg = load(fake_registry)
    argv = build_argv(reg.get(), "llama-server")

    def value_after(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert value_after("--ctx-size") == "8192"
    assert value_after("-ub") == "512"
    assert value_after("-ctk") == "q8_0"
    assert value_after("-dev") == "CUDA0,CUDA1"
    assert value_after("--temp") == "1.0"
    assert "--jinja" in argv
    assert "--spec-type" in argv, "speculative decoding was enabled in the registry"
    assert "--mmproj" not in argv, "vision was not requested"


def test_vision_applies_its_own_operating_point(fake_registry: Path) -> None:
    """Vision is a different profile, not a flag: it carries its own ctx and split."""
    reg = load(fake_registry)
    argv = build_argv(reg.get(), "llama-server", vision=True)

    assert argv[argv.index("--ctx-size") + 1] == "4096", "vision ctx overrides the text default"
    assert argv[argv.index("-ts") + 1] == "0.4,0.6"
    assert "--mmproj" in argv
    assert argv[argv.index("--image-min-tokens") + 1] == "1024"


def test_explicit_override_beats_the_vision_profile(fake_registry: Path) -> None:
    reg = load(fake_registry)
    argv = build_argv(reg.get(), "llama-server", vision=True, overrides={"ctx": 16384})
    assert argv[argv.index("--ctx-size") + 1] == "16384"


def test_large_micro_batch_with_mtp_is_refused(fake_registry: Path) -> None:
    reg = load(fake_registry)
    with pytest.raises(RegistryError, match="micro-batch"):
        build_argv(reg.get(), "llama-server", overrides={"ubatch": 2048})


def test_split_device_count_mismatch_is_refused(fake_registry: Path) -> None:
    reg = load(fake_registry)
    with pytest.raises(RegistryError, match="ratio"):
        build_argv(reg.get(), "llama-server", overrides={"split": "0.5,0.3,0.2"})


def test_missing_model_file_is_reported_clearly(fake_registry: Path, tmp_path: Path) -> None:
    reg = load(fake_registry)
    entry = reg.get()
    entry.file = "does-not-exist.gguf"
    with pytest.raises(RegistryError, match="missing"):
        build_argv(entry, "llama-server")


def test_measured_provenance_is_distinguished(fake_registry: Path) -> None:
    """Inherited numbers must not be indistinguishable from measured ones."""
    reg = load(fake_registry)
    entry = reg.get()
    assert entry.measured_on_this_file is True

    entry.measured = {"status": "INHERITED from a sibling build"}
    assert entry.measured_on_this_file is False


# ---------------------------------------------------------------- gguf analysis


def _analysis(**kw) -> GgufAnalysis:
    base = {"source": "test/model.gguf", "architecture": "demo", "tensor_count": 100}
    base.update(kw)
    return GgufAnalysis(**base)  # type: ignore[arg-type]


def _titles(a) -> list[str]:
    return [f.title for f in a.findings]


def _level_of(a, needle: str) -> str:
    from headroom.gguf import Finding

    match: Finding = next(f for f in a.findings if needle.lower() in f.title.lower())
    return match.level


def test_missing_speculative_head_is_flagged() -> None:
    from headroom.gguf import interpret

    a = _analysis(mtp_tensors=[])
    interpret(a)
    assert _level_of(a, "speculative") == "caution"

    b = _analysis(mtp_tensors=["blk.64.nextn.eh_proj"])
    interpret(b)
    assert _level_of(b, "speculative") == "good"


def test_protected_recurrent_layers_read_as_good() -> None:
    """The distinction the whole probe exists to draw."""
    from headroom.gguf import interpret

    protected = _analysis(families={"recurrent": {"F32": 192, "Q8_0": 96, "Q5_K": 33}})
    interpret(protected)
    assert _level_of(protected, "recurrent") == "good"

    uniform = _analysis(families={"recurrent": {"F32": 192, "Q4_K": 144}})
    interpret(uniform)
    assert _level_of(uniform, "recurrent") == "caution"


def test_f32_is_excluded_from_the_quantization_description() -> None:
    """F32 tensors are norms and biases, and the ratios already exclude them.

    Listing them in the prose would describe a distribution the adjacent numbers
    do not refer to.
    """
    from headroom.gguf import interpret

    a = _analysis(families={"recurrent": {"F32": 192, "Q4_K": 144}})
    interpret(a)
    detail = next(f.detail for f in a.findings if "recurrent" in f.title.lower())
    assert "144xQ4_K" in detail
    assert "F32" not in detail


def test_aggressively_quantized_attention_is_flagged() -> None:
    from headroom.gguf import interpret

    a = _analysis(families={"attention": {"F32": 99, "IQ3_S": 96, "Q8_0": 34}})
    interpret(a)
    assert _level_of(a, "attention") == "caution"


def test_fit_is_judged_against_real_free_vram() -> None:
    """A size in gibibytes means nothing without the machine it has to fit on."""
    from headroom.gguf import interpret

    size = 15 * 1024**3

    roomy = _analysis(file_size_bytes=size)
    interpret(roomy, free_vram_mib=22000)
    assert _level_of(roomy, "fit") == "good"

    snug = _analysis(file_size_bytes=size)
    interpret(snug, free_vram_mib=16000)
    assert _level_of(snug, "leaves little") == "caution"

    too_big = _analysis(file_size_bytes=size)
    interpret(too_big, free_vram_mib=8000)
    assert _level_of(too_big, "not fit") == "caution"


def test_fit_is_silent_without_telemetry() -> None:
    """With no GPU data, saying nothing beats guessing."""
    from headroom.gguf import interpret

    a = _analysis(file_size_bytes=15 * 1024**3)
    interpret(a, free_vram_mib=None)
    assert not any("fit" in t.lower() for t in _titles(a))


def test_non_gguf_input_explains_the_likely_cause() -> None:
    from io import BytesIO

    from headroom.gguf import GgufError, parse

    with pytest.raises(GgufError, match="gated"):
        parse(BytesIO(b"<html>401 Unauthorized</html>"), source="x")


# ---------------------------------------------------------------- registry writes


def test_adding_an_entry_preserves_everything_else(fake_registry: Path) -> None:
    """models.json belongs to the user and is shared with their shell scripts.

    Comment blocks, the template, and unrelated entries must survive a write
    untouched -- this app is a guest in that file.
    """
    from headroom.registry import add_entry

    original = json.loads(fake_registry.read_text(encoding="utf-8"))
    original["_comment"] = ["a comment the user wrote"]
    fake_registry.write_text(json.dumps(original, indent=2), encoding="utf-8")

    add_entry(fake_registry, "another", {"label": "Another", "serve": {}})

    after = json.loads(fake_registry.read_text(encoding="utf-8"))
    assert after["_comment"] == ["a comment the user wrote"]
    assert "_template" in after["models"], "the template must survive"
    assert after["models"]["demo"] == original["models"]["demo"], "existing entry changed"
    assert "another" in after["models"]


def test_a_backup_is_written_before_the_edit(fake_registry: Path) -> None:
    from headroom.registry import add_entry

    before = fake_registry.read_text(encoding="utf-8")
    add_entry(fake_registry, "another", {"label": "Another"})

    backup = fake_registry.with_suffix(fake_registry.suffix + ".bak")
    assert backup.exists(), "no backup was written"
    assert backup.read_text(encoding="utf-8") == before, "backup does not match the pre-edit file"


def test_existing_keys_are_refused(fake_registry: Path) -> None:
    """Silently replacing an entry would discard measurements someone earned."""
    from headroom.registry import RegistryError, add_entry

    with pytest.raises(RegistryError, match="already in the registry"):
        add_entry(fake_registry, "demo", {"label": "clobbered"})


def test_private_keys_are_refused(fake_registry: Path) -> None:
    from headroom.registry import RegistryError, add_entry

    with pytest.raises(RegistryError, match="private"):
        add_entry(fake_registry, "_sneaky", {"label": "x"})


def test_serve_block_is_not_inherited_across_architectures(fake_registry: Path) -> None:
    """The rule the whole registry design exists to enforce.

    A micro-batch or context size tuned for one architecture can be actively
    wrong for another, because the bottleneck moves. Copying it over would
    produce a config that looks authoritative and is not.
    """
    from headroom.registry import derive_entry, load

    reg = load(fake_registry)
    donor = reg.get("demo")
    assert donor.serve["ctx"] == 8192

    entry = derive_entry(
        key="different",
        label="Different Architecture",
        repo="x/y",
        filename="y.gguf",
        directory="/tmp/y",
        size_gib=5.0,
        architecture="some-other-arch",
        has_mtp=False,
        template={"serve": {"ctx": 4096}},
        inherit_from=donor,
    )

    assert entry["serve"]["ctx"] == 4096, "template default should win, not the donor's 8192"
    assert "does not transfer" in entry["measured"]["status"]


def test_serve_block_is_inherited_within_an_architecture_but_marked(fake_registry: Path) -> None:
    from headroom.registry import derive_entry, load

    reg = load(fake_registry)
    donor = reg.get("demo")

    entry = derive_entry(
        key="sibling",
        label="Same Architecture",
        repo="x/y",
        filename="y.gguf",
        directory="/tmp/y",
        size_gib=5.0,
        architecture=donor.arch,
        has_mtp=True,
        template={"serve": {"ctx": 4096}},
        inherit_from=donor,
    )

    assert entry["serve"]["ctx"] == 8192, "same architecture should inherit"
    assert "INHERITED" in entry["measured"]["status"]
    assert "NOT measured" in entry["measured"]["status"]
    assert entry["verified"]["benched"] is False, "inherited numbers are not verification"


def test_mtp_comes_from_the_probe_not_from_the_donor(fake_registry: Path) -> None:
    """Whether a speculative head exists is a fact about the file, not a guess."""
    from headroom.registry import derive_entry, load

    reg = load(fake_registry)
    donor = reg.get("demo")
    assert donor.serve["mtp"] is True

    entry = derive_entry(
        key="nomtp",
        label="No Speculative Head",
        repo="x/y",
        filename="y.gguf",
        directory="/tmp/y",
        size_gib=5.0,
        architecture=donor.arch,
        has_mtp=False,
        inherit_from=donor,
    )
    assert entry["serve"]["mtp"] is False, "the probe found no head; the donor's true must not win"


def test_speculative_decoding_forces_a_small_micro_batch(fake_registry: Path) -> None:
    """The draft context's buffers scale with -ub, so the two are coupled."""
    from headroom.registry import derive_entry

    entry = derive_entry(
        key="spec",
        label="Speculative",
        repo="x/y",
        filename="y.gguf",
        directory="/tmp/y",
        size_gib=5.0,
        architecture="a",
        has_mtp=True,
        template={"serve": {"ubatch": 4096}},
    )
    assert entry["serve"]["ubatch"] == 512


def test_env_paths_tolerate_trailing_whitespace(monkeypatch) -> None:
    """A trailing space in an env var must not leak into derived paths.

    `set VAR=value && cmd` in cmd.exe captures the space before the `&&`.
    Windows still opens the file, so nothing looks wrong -- but every derived
    path inherits the space, and a backup lands as "models.json .bak".
    """
    from headroom.app import Settings

    monkeypatch.setenv("HEADROOM_REGISTRY", "C:/models/models.json   ")
    settings = Settings.resolve(create_registry=False)

    assert str(settings.registry_path).endswith("models.json")
    derived = settings.registry_path.with_suffix(settings.registry_path.suffix + ".bak")
    assert derived.name == "models.json.bak"


# ---------------------------------------------------------------- discovery


def test_explicit_argument_beats_the_environment(monkeypatch, tmp_path: Path) -> None:
    from headroom.config import resolve_registry

    monkeypatch.setenv("HEADROOM_REGISTRY", str(tmp_path / "from-env.json"))
    r = resolve_registry(str(tmp_path / "explicit.json"))
    assert r.path is not None and r.path.name == "explicit.json"
    assert r.source == "argument"


def test_environment_beats_discovery(monkeypatch, tmp_path: Path) -> None:
    from headroom.config import resolve_registry

    target = tmp_path / "from-env.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HEADROOM_REGISTRY", str(target))
    r = resolve_registry()
    assert r.path == target
    assert r.source == "HEADROOM_REGISTRY"
    assert r.exists


def test_a_fresh_machine_gets_a_usable_registry(monkeypatch, tmp_path: Path) -> None:
    """The whole point of this module.

    With nothing configured and nothing installed, Headroom must still come up
    with somewhere to put a model. An app that errors until the user reads the
    source is not a working app.
    """
    from headroom.config import resolve_registry
    from headroom.registry import load

    monkeypatch.delenv("HEADROOM_REGISTRY", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    r = resolve_registry(create=True)
    assert r.exists, "no registry was created"
    assert r.path is not None

    # And it must be loadable, not merely present.
    reg = load(r.path)
    assert reg.models == {}, "a starter registry has no real models, only a template"


def test_a_missing_llama_server_is_reported_not_guessed(monkeypatch, tmp_path: Path) -> None:
    """Reporting nothing beats inventing a path that does not exist."""
    from headroom.config import resolve_llama_server

    monkeypatch.delenv("HEADROOM_LLAMA_SERVER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    r = resolve_llama_server()
    if r.exists:
        pytest.skip("a real llama-server is installed in a conventional location")
    assert r.path is None, "a not-found result must not carry a fabricated path"
    assert r.source == "not found"
    assert r.searched, "the search locations should be reported so the user can fix it"


def test_an_empty_registry_says_so_rather_than_reporting_a_typo(tmp_path: Path) -> None:
    """A fresh install is not a mistyped model name."""
    from headroom.config import write_starter_registry
    from headroom.registry import RegistryError, load

    path = tmp_path / "models.json"
    write_starter_registry(path)
    reg = load(path)

    with pytest.raises(RegistryError, match="no models in the registry yet"):
        reg.get()


# ---------------------------------------------------------------- degraded environments


def test_the_app_serves_without_a_gpu_or_a_registry(monkeypatch, tmp_path: Path) -> None:
    """Headroom must come up on a machine with none of its dependencies.

    This lived as an inline Python snippet inside the CI workflow, where it was
    unlinted, untested locally, and free to drift -- which it promptly did,
    calling a constructor signature that had changed. YAML is a poor place to
    keep code. Here it runs on every machine, including the developer's.
    """
    from fastapi.testclient import TestClient

    from headroom.app import Settings, create_app

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("HEADROOM_REGISTRY", raising=False)
    monkeypatch.delenv("HEADROOM_LLAMA_SERVER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    # Pinned, because "degraded install" is the subject here and "nothing is
    # serving" is only the backdrop. Left to the real probe this asserted a fact
    # about the developer's machine instead: on a box with a model up -- which
    # is the normal state while working on the benchmark -- it failed for a
    # reason unrelated to anything it tests.
    async def nothing_serving(_port, timeout: float = 3.0):
        return ServerState()

    monkeypatch.setattr("headroom.server.probe", nothing_serving)

    settings = Settings.resolve()

    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()
        assert health["ok"] is True, health
        assert "problems" in health, "a degraded install must say what it is missing"

        # Whatever the hardware, this must be a list rather than an error.
        gpus = client.get("/api/gpus").json()
        assert isinstance(gpus["gpus"], list)

        # No server running, and no llama.cpp to start one with.
        assert client.get("/api/server").json()["status"] == "stopped"

        # The registry was created, so this is 200 with nothing in it -- not a 404.
        models = client.get("/api/models")
        assert models.status_code == 200, models.text
        assert models.json()["models"] == []


def test_starting_without_llama_cpp_explains_itself(monkeypatch, tmp_path: Path) -> None:
    """503 with a reason beats a traceback or a silent no-op."""
    from fastapi.testclient import TestClient

    from headroom.app import Settings, create_app

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("HEADROOM_LLAMA_SERVER", raising=False)
    monkeypatch.delenv("HEADROOM_REGISTRY", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    settings = Settings.resolve()
    if settings.llama_server is not None:
        pytest.skip("llama-server is installed in a conventional location here")

    with TestClient(create_app(settings)) as client:
        resp = client.post("/api/server/start")
        assert resp.status_code == 503
        assert "llama-server" in resp.json()["detail"]
