"""The benchmark, and what it is allowed to write down.

Every rule tested here was learned by believing a number that was wrong. The
tests are written against those specific failures rather than against the code's
current shape, so that a rewrite which reintroduces one of them fails:

- warm-up runs dragging the mean down,
- prefill measured on a cached prompt and reported as a real figure,
- draft acceptance silently reading `None` because the field was misspelt,
- a measurement landing on the wrong registry entry,
- a benchmark rewriting the tuning it was supposed to be measuring.

The server is faked with an httpx MockTransport so these run anywhere. What is
faked is only the transport: the runner's own arithmetic, discarding and
provenance logic is the code under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from headroom import bench as bench_mod
from headroom.registry import RegistryError, find_by_path, load, record_measurement

# --------------------------------------------------------------- statistics


def test_standard_deviation_is_the_sample_one() -> None:
    # Population SD of this set is 0.816; sample is 1.0. With n=3 the difference
    # is not academic, and the ~6% significance rule is read against this number.
    assert bench_mod.stdev([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_a_single_run_reports_no_spread_rather_than_failing() -> None:
    assert bench_mod.stdev([27.6]) == 0.0


@pytest.mark.parametrize(
    ("n_ctx", "expected"),
    [(65536, "64k"), (4096, "4k"), (5000, "5000"), (None, "unknown")],
)
def test_context_label_matches_how_the_registry_names_it(n_ctx, expected) -> None:
    assert bench_mod.context_label(n_ctx) == expected


# ------------------------------------------------------------- fake server


class FakeServer:
    """A llama-server that reports whatever these tests need it to.

    Warm-up generations are answered with a deliberately terrible decode rate so
    that a run which fails to discard them cannot accidentally pass.
    """

    def __init__(
        self,
        *,
        warmup_decode: float = 5.0,
        task_decode: float = 30.0,
        warmup_calls: int = 3,
        prefill_cached_after: int = 1,
        draft_field: str = "draft_n_accepted",
    ) -> None:
        self.warmup_decode = warmup_decode
        self.task_decode = task_decode
        self.warmup_calls = warmup_calls
        self.prefill_cached_after = prefill_cached_after
        self.draft_field = draft_field
        self.generations = 0
        self.prefills = 0
        self.prompts: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 65536}})
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(30))})
        if path == "/v1/chat/completions":
            return self._generate(json.loads(request.content))
        return httpx.Response(404)

    def _generate(self, body: dict) -> httpx.Response:
        prompt = body["messages"][0]["content"]
        self.prompts.append(prompt)

        if body["max_tokens"] == 1:
            # The prefill stage. After the first, pretend the prompt cache served
            # it: prompt_n collapses to a handful of tokens even though the
            # reported rate looks spectacular. That combination is exactly what
            # made a real run silently rest on one sample.
            self.prefills += 1
            cached = self.prefills > self.prefill_cached_after
            return httpx.Response(
                200,
                json={
                    "timings": {
                        "prompt_n": 3 if cached else 6000,
                        "prompt_per_second": 99999.0 if cached else 900.0,
                        "predicted_per_second": 1.0,
                    }
                },
            )

        self.generations += 1
        warm = self.generations <= self.warmup_calls
        return httpx.Response(
            200,
            json={
                "timings": {
                    "predicted_per_second": self.warmup_decode if warm else self.task_decode,
                    "prompt_per_second": 54.9,
                    "prompt_n": 40,
                    "draft_n": 100,
                    self.draft_field: 50,
                }
            },
        )


# Captured before any patching: `headroom.bench.httpx` is the httpx module
# itself, so a factory that calls `httpx.AsyncClient` after the patch calls
# itself.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _mock_client_factory(handler):
    def factory(*_args, **_kwargs):
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), timeout=5.0)

    return factory


@pytest.fixture
def patch_httpx(monkeypatch):
    def install(server: FakeServer):
        monkeypatch.setattr(
            "headroom.bench.httpx.AsyncClient", _mock_client_factory(server.handler)
        )

    return install


async def _run_bench(runner: bench_mod.BenchmarkRunner, **kw) -> bench_mod.Benchmark:
    job = runner.start(model_key="demo", model_path="/w/fake.gguf", port=8080, **kw)
    task = runner._tasks[job.id]
    await task
    return job


# ------------------------------------------------------------------- runner


@pytest.mark.asyncio
async def test_warmup_runs_are_discarded_from_the_mean(patch_httpx) -> None:
    server = FakeServer(warmup_decode=5.0, task_decode=30.0, warmup_calls=3)
    patch_httpx(server)

    job = await _run_bench(bench_mod.BenchmarkRunner(), reps=3, warmup=3)

    assert job.status is bench_mod.BenchStatus.COMPLETE
    # Nine kept task runs at 30, three discarded at 5. Including the warm-up
    # would give 23.75 -- a plausible-looking number that is simply wrong.
    assert job.result["decode_tok_s"] == pytest.approx(30.0)
    assert job.result["measured"]["decode_runs"] == 9
    assert job.result["decode_sd"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_prefill_runs_served_from_cache_are_thrown_away(patch_httpx) -> None:
    server = FakeServer(prefill_cached_after=1)
    patch_httpx(server)

    job = await _run_bench(bench_mod.BenchmarkRunner(), reps=3, warmup=0)

    # One honest run at 900; two cache hits reporting 99999 were dropped rather
    # than averaged in, which would have produced ~67000 tok/s.
    assert job.result["prefill_tok_s"] == pytest.approx(900.0)
    assert job.result["prefill_cached_runs"] == 2
    assert job.result["measured"]["prefill_runs"] == 1
    # And the shortfall is stated, because a figure resting on one sample of a
    # requested three must not look like three.
    assert "smaller sample" in job.result["measured"]["prefill_note"]


@pytest.mark.asyncio
async def test_every_prefill_rep_gets_a_different_prompt(patch_httpx) -> None:
    server = FakeServer()
    patch_httpx(server)

    await _run_bench(bench_mod.BenchmarkRunner(), reps=3, warmup=0)

    long_prompts = [p for p in server.prompts if len(p) > 2000]
    assert len(long_prompts) == 3
    # Identical prompts are what let the cache collapse the sample in the first
    # place. Uniqueness is the fix, so it is asserted rather than assumed.
    assert len(set(long_prompts)) == 3


@pytest.mark.asyncio
async def test_all_prefill_cached_reports_no_figure_rather_than_a_bad_one(patch_httpx) -> None:
    server = FakeServer(prefill_cached_after=0)
    patch_httpx(server)

    job = await _run_bench(bench_mod.BenchmarkRunner(), reps=2, warmup=0)

    assert job.result["prefill_tok_s"] is None
    assert "prefill_tok_s" not in job.result["measured"]
    assert "NOT MEASURED" in job.result["measured"]["prefill_note"]


@pytest.mark.asyncio
async def test_draft_acceptance_is_read_from_the_right_field(patch_httpx) -> None:
    server = FakeServer()
    patch_httpx(server)

    job = await _run_bench(bench_mod.BenchmarkRunner(), reps=2, warmup=0)

    assert job.result["acceptance_range"] == "0.500 - 0.500"
    assert "Read decode WITH acceptance" in job.result["measured"]["decode_note"]


@pytest.mark.asyncio
async def test_a_misspelt_acceptance_field_does_not_silently_pass(patch_httpx) -> None:
    # `draft_accepted_n` is the plausible wrong spelling. It must produce no
    # acceptance figure at all rather than a confident-looking zero.
    server = FakeServer(draft_field="draft_accepted_n")
    patch_httpx(server)

    job = await _run_bench(bench_mod.BenchmarkRunner(), reps=2, warmup=0)

    assert job.result["acceptance_range"] is None
    assert "decode_note" not in job.result["measured"]


@pytest.mark.asyncio
async def test_decode_is_never_recorded_without_its_spread(patch_httpx) -> None:
    server = FakeServer(task_decode=27.6)
    patch_httpx(server)

    job = await _run_bench(bench_mod.BenchmarkRunner(), reps=3, warmup=0)
    measured = job.result["measured"]

    assert "decode_tok_s" in measured
    assert "decode_sd" in measured
    assert "decode_runs" in measured
    assert "under ~6%" in job.result["significance_note"]


@pytest.mark.asyncio
async def test_two_benchmarks_at_once_are_refused(patch_httpx) -> None:
    server = FakeServer()
    patch_httpx(server)
    runner = bench_mod.BenchmarkRunner()

    first = runner.start(model_key="demo", model_path="/w/fake.gguf", port=8080, reps=1, warmup=0)
    with pytest.raises(bench_mod.BenchError, match="already running"):
        runner.start(model_key="demo", model_path="/w/fake.gguf", port=8080)

    await runner._tasks[first.id]


@pytest.mark.asyncio
async def test_a_failing_server_fails_the_job_rather_than_recording_nothing(monkeypatch) -> None:
    monkeypatch.setattr(
        "headroom.bench.httpx.AsyncClient",
        _mock_client_factory(lambda _r: httpx.Response(500)),
    )

    job = await _run_bench(bench_mod.BenchmarkRunner(), reps=1, warmup=0)

    assert job.status is bench_mod.BenchStatus.FAILED
    assert job.result is None
    assert job.error


# ----------------------------------------------------------- recording it


@pytest.fixture
def registry_with_unmeasured_entry(tmp_path: Path) -> Path:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "fake.gguf").write_bytes(b"GGUF")
    doc = {
        "default": "demo",
        "_comment": "kept verbatim",
        "models": {
            "_template": {"label": "ignored"},
            "demo": {
                "label": "Demo Model",
                "file": "fake.gguf",
                "dir": str(weights).replace("\\", "/"),
                "size_gib": 1.0,
                "arch": "demo-arch",
                "serve": {"ctx": 8192, "ubatch": 512, "mtp": True},
                "measured": {"status": "NOT MEASURED. Template defaults only."},
                "verified": {"benched": False, "loads": False, "needle_tested": False},
            },
            "other": {
                "label": "Untouched",
                "file": "other.gguf",
                "dir": str(weights).replace("\\", "/"),
                "arch": "demo-arch",
                "measured": {"status": "MEASURED on this exact file, 2026-01-01"},
                "verified": {"benched": True},
            },
        },
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_recording_clears_not_measured(registry_with_unmeasured_entry: Path) -> None:
    before = load(registry_with_unmeasured_entry)
    assert not before.models["demo"].measured_on_this_file

    record_measurement(
        registry_with_unmeasured_entry,
        "demo",
        {"status": "MEASURED on this exact file, 2026-08-25, via Headroom", "decode_tok_s": 30.0},
    )

    after = load(registry_with_unmeasured_entry)
    assert after.models["demo"].measured_on_this_file
    assert after.models["demo"].measured["decode_tok_s"] == 30.0


def test_a_benchmark_does_not_rewrite_the_tuning_it_measured(
    registry_with_unmeasured_entry: Path,
) -> None:
    # The serve block is the user's configuration. A benchmark observes a
    # configuration; it does not get to change one, or the next run measures
    # something else and the comparison is meaningless.
    record_measurement(registry_with_unmeasured_entry, "demo", {"status": "MEASURED"})

    entry = load(registry_with_unmeasured_entry).models["demo"]
    assert entry.serve == {"ctx": 8192, "ubatch": 512, "mtp": True}
    assert entry.label == "Demo Model"


def test_recording_asserts_only_what_a_benchmark_shows(
    registry_with_unmeasured_entry: Path,
) -> None:
    record_measurement(registry_with_unmeasured_entry, "demo", {"status": "MEASURED"})
    verified = load(registry_with_unmeasured_entry).models["demo"].verified

    # It ran, so it loads and it has been benched.
    assert verified["loads"] is True
    assert verified["benched"] is True
    # It says nothing whatsoever about long-context retrieval.
    assert verified["needle_tested"] is False


def test_recording_leaves_every_other_entry_alone(registry_with_unmeasured_entry: Path) -> None:
    record_measurement(registry_with_unmeasured_entry, "demo", {"status": "MEASURED"})
    raw = json.loads(registry_with_unmeasured_entry.read_text(encoding="utf-8"))

    assert raw["_comment"] == "kept verbatim"
    assert raw["models"]["_template"] == {"label": "ignored"}
    assert raw["models"]["other"]["measured"]["status"].startswith("MEASURED on this exact file")
    assert raw["default"] == "demo"


def test_the_replaced_figures_are_handed_back(registry_with_unmeasured_entry: Path) -> None:
    # So a UI can show what a measurement overwrote instead of a number simply
    # changing under the user.
    previous = record_measurement(
        registry_with_unmeasured_entry, "other", {"status": "MEASURED today"}
    )
    assert previous["status"] == "MEASURED on this exact file, 2026-01-01"


def test_a_backup_is_written_before_recording(registry_with_unmeasured_entry: Path) -> None:
    original = registry_with_unmeasured_entry.read_text(encoding="utf-8")
    record_measurement(registry_with_unmeasured_entry, "demo", {"status": "MEASURED"})

    backup = registry_with_unmeasured_entry.with_suffix(
        registry_with_unmeasured_entry.suffix + ".bak"
    )
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == original


def test_recording_on_an_unknown_key_says_what_it_knows(
    registry_with_unmeasured_entry: Path,
) -> None:
    with pytest.raises(RegistryError, match="unknown model 'ghost'"):
        record_measurement(registry_with_unmeasured_entry, "ghost", {"status": "MEASURED"})


# --------------------------------------------------------- attribution


def test_the_entry_is_found_from_the_file_the_server_loaded(
    registry_with_unmeasured_entry: Path,
) -> None:
    reg = load(registry_with_unmeasured_entry)
    loaded = str(reg.models["demo"].path)

    # The same file arrives with different separators and casing depending on
    # whether it came from the registry or from a Windows command line.
    assert find_by_path(reg, loaded).key == "demo"
    assert find_by_path(reg, loaded.replace("/", "\\")).key == "demo"
    assert find_by_path(reg, loaded.upper()).key == "demo"


def test_an_unregistered_file_matches_nothing_rather_than_the_default(
    registry_with_unmeasured_entry: Path,
) -> None:
    # Falling back to the default here is the bug this guards: it would write one
    # model's measurements onto another entry, silently.
    reg = load(registry_with_unmeasured_entry)
    assert find_by_path(reg, "/somewhere/else/stranger.gguf") is None


# ------------------------------------------------------------ the endpoint


def _client(monkeypatch, tmp_path: Path, registry: Path):
    from fastapi.testclient import TestClient

    from headroom.app import Settings, create_app

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("HEADROOM_REGISTRY", str(registry))
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app(Settings.resolve()))


def _serving(model_path: str | None):
    """A probe result standing in for a live server."""
    from headroom.server import ServerState

    async def probe(_port, timeout: float = 3.0):
        return ServerState(
            running=True, reachable=True, pid=4242, model_path=model_path, n_ctx=65536
        )

    return probe


def test_benchmarking_nothing_is_refused_with_a_reason(
    monkeypatch, tmp_path: Path, registry_with_unmeasured_entry: Path
) -> None:
    from headroom.server import ServerState

    async def stopped(_port, timeout: float = 3.0):
        return ServerState()

    # Pinned rather than left to the machine: on a developer box something is
    # often serving on 8080, and a test that passes only when it is not is not
    # a test.
    monkeypatch.setattr("headroom.server.probe", stopped)
    with _client(monkeypatch, tmp_path, registry_with_unmeasured_entry) as client:
        resp = client.post("/api/bench/start")
        assert resp.status_code == 409
        assert "Start a model" in resp.json()["detail"]


def test_benchmarking_a_loading_server_is_refused(
    monkeypatch, tmp_path: Path, registry_with_unmeasured_entry: Path
) -> None:
    from headroom.server import ServerState

    async def loading(_port, timeout: float = 3.0):
        # Process up, /props not answering yet: the gap that tempts a UI into
        # measuring a model that is not ready.
        return ServerState(running=True, reachable=False, pid=99)

    monkeypatch.setattr("headroom.server.probe", loading)
    with _client(monkeypatch, tmp_path, registry_with_unmeasured_entry) as client:
        resp = client.post("/api/bench/start")
        assert resp.status_code == 409
        assert "still loading" in resp.json()["detail"]


def test_an_unregistered_running_model_is_refused_rather_than_attributed(
    monkeypatch, tmp_path: Path, registry_with_unmeasured_entry: Path
) -> None:
    monkeypatch.setattr("headroom.server.probe", _serving("/somewhere/stranger.gguf"))
    with _client(monkeypatch, tmp_path, registry_with_unmeasured_entry) as client:
        resp = client.post("/api/bench/start")
        assert resp.status_code == 400
        assert "not in" in resp.json()["detail"]
        assert "will not guess" in resp.json()["detail"]


def test_the_endpoint_targets_the_file_the_server_loaded(
    monkeypatch, tmp_path: Path, registry_with_unmeasured_entry: Path
) -> None:
    # Even though 'demo' is the registry default, the job must name whichever
    # entry owns the loaded file -- here, deliberately the other one.
    reg = load(registry_with_unmeasured_entry)
    monkeypatch.setattr("headroom.server.probe", _serving(str(reg.models["other"].path)))

    server = FakeServer()
    monkeypatch.setattr("headroom.bench.httpx.AsyncClient", _mock_client_factory(server.handler))

    with _client(monkeypatch, tmp_path, registry_with_unmeasured_entry) as client:
        resp = client.post("/api/bench/start?reps=1&warmup=0&prefill_tokens=0")
        assert resp.status_code == 200, resp.text
        assert resp.json()["model_key"] == "other"


def test_the_running_model_is_reported_as_a_registry_key(
    monkeypatch, tmp_path: Path, registry_with_unmeasured_entry: Path
) -> None:
    reg = load(registry_with_unmeasured_entry)
    monkeypatch.setattr("headroom.server.probe", _serving(str(reg.models["demo"].path)))

    with _client(monkeypatch, tmp_path, registry_with_unmeasured_entry) as client:
        assert client.get("/api/server").json()["model_key"] == "demo"


def test_a_file_outside_the_registry_reports_no_key_rather_than_a_wrong_one(
    monkeypatch, tmp_path: Path, registry_with_unmeasured_entry: Path
) -> None:
    monkeypatch.setattr("headroom.server.probe", _serving("/somewhere/stranger.gguf"))

    with _client(monkeypatch, tmp_path, registry_with_unmeasured_entry) as client:
        assert client.get("/api/server").json()["model_key"] is None


def test_writing_does_not_rewrite_the_line_endings(registry_with_unmeasured_entry: Path) -> None:
    """A one-entry edit must produce a one-entry diff.

    `Path.write_text` translates newlines on Windows, which turned every write
    of this file into a whole-file change and left the backup differing
    byte-for-byte from the original it copies. models.json is shared with the
    user's launch scripts, so that is not a cosmetic problem.
    """
    original = registry_with_unmeasured_entry.read_bytes()
    assert b"\r\n" not in original, "fixture should be LF, or this proves nothing"

    record_measurement(registry_with_unmeasured_entry, "demo", {"status": "MEASURED"})

    assert b"\r\n" not in registry_with_unmeasured_entry.read_bytes()
    backup = registry_with_unmeasured_entry.with_suffix(
        registry_with_unmeasured_entry.suffix + ".bak"
    )
    # The backup is a copy, so it is byte-identical -- not merely equivalent.
    assert backup.read_bytes() == original


# ------------------------------------------------------------------- vram


class FakeCard:
    def __init__(self, free: int, label: str, cuda_index: int | None = None) -> None:
        self.memory_free_mib = free
        self.label = label
        self.cuda_index = cuda_index


async def test_free_vram_is_recorded_per_card_not_just_summed(patch_httpx) -> None:
    patch_httpx(FakeServer())
    runner = bench_mod.BenchmarkRunner(
        read_gpus=lambda: [
            FakeCard(2105, "NVIDIA GeForce RTX 4070 SUPER (CUDA0)", cuda_index=0),
            FakeCard(993, "NVIDIA GeForce RTX 3060 (CUDA1)", cuda_index=1),
        ]
    )
    job = await _run_bench(runner, reps=1, warmup=0)
    measured = job.result["measured"]

    # The total is the misleading half on a machine like this one: 3098 MiB
    # "free" is two cards, and only one of them also drives the desktop.
    assert measured["vram_free_mib_at_64k"] == 3098
    assert measured["vram_free_breakdown"] == (
        "2105 MiB on the NVIDIA GeForce RTX 4070 SUPER (CUDA0) + "
        "993 MiB on the NVIDIA GeForce RTX 3060 (CUDA1)"
    )


async def test_the_breakdown_is_ordered_by_cuda_index(patch_httpx) -> None:
    """This box enumerates the cards in the opposite order to llama.cpp.

    Printing them in backend order puts CUDA1 before CUDA0 and leaves the reader
    doing the reconciliation the app exists to do for them.
    """
    patch_httpx(FakeServer())
    runner = bench_mod.BenchmarkRunner(
        read_gpus=lambda: [
            FakeCard(994, "RTX 3060 (CUDA1)", cuda_index=1),
            FakeCard(2044, "RTX 4070 SUPER (CUDA0)", cuda_index=0),
        ]
    )
    job = await _run_bench(runner, reps=1, warmup=0)

    assert job.result["measured"]["vram_free_breakdown"] == (
        "2044 MiB on the RTX 4070 SUPER (CUDA0) + 994 MiB on the RTX 3060 (CUDA1)"
    )


async def test_an_unmapped_card_sorts_last_rather_than_being_guessed(patch_httpx) -> None:
    patch_httpx(FakeServer())
    runner = bench_mod.BenchmarkRunner(
        read_gpus=lambda: [
            FakeCard(100, "mystery (nvml 2)", cuda_index=None),
            FakeCard(200, "known (CUDA0)", cuda_index=0),
        ]
    )
    job = await _run_bench(runner, reps=1, warmup=0)

    assert job.result["measured"]["vram_free_breakdown"].startswith("200 MiB on the known (CUDA0)")


async def test_the_vram_key_carries_the_context_it_was_measured_at(patch_httpx) -> None:
    patch_httpx(FakeServer())
    runner = bench_mod.BenchmarkRunner(read_gpus=lambda: [FakeCard(1000, "card (CUDA0)")])
    job = await _run_bench(runner, reps=1, warmup=0)

    # A figure taken at 64K says nothing about the same model at 96K.
    assert "vram_free_mib_at_64k" in job.result["measured"]
    assert "vram_free_mib" not in job.result["measured"]


async def test_no_gpu_backend_records_no_vram_rather_than_a_zero(patch_httpx) -> None:
    patch_httpx(FakeServer())
    job = await _run_bench(bench_mod.BenchmarkRunner(read_gpus=None), reps=1, warmup=0)

    assert job.status is bench_mod.BenchStatus.COMPLETE
    assert job.result["vram_free_mib"] is None
    assert not [k for k in job.result["measured"] if k.startswith("vram_free")]


async def test_a_telemetry_failure_does_not_void_the_benchmark(patch_httpx) -> None:
    def explode():
        raise RuntimeError("NVML is having a day")

    patch_httpx(FakeServer())
    job = await _run_bench(bench_mod.BenchmarkRunner(read_gpus=explode), reps=1, warmup=0)

    # The throughput figures were honestly earned; losing them to a telemetry
    # hiccup would waste four minutes of GPU time for nothing.
    assert job.status is bench_mod.BenchStatus.COMPLETE
    assert job.result["decode_tok_s"] is not None
    assert job.result["vram_free_mib"] is None


async def test_prefill_gets_a_spread_like_decode_does(patch_httpx) -> None:
    patch_httpx(FakeServer(prefill_cached_after=99))
    job = await _run_bench(bench_mod.BenchmarkRunner(), reps=3, warmup=0)

    assert job.result["measured"]["prefill_sd"] == pytest.approx(0.0)
    assert job.result["measured"]["prefill_runs"] == 3


# --------------------------------------------------------- carrying prose


def test_notes_a_benchmark_cannot_produce_survive_a_run(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "demo": {
                        "measured": {
                            "status": "MEASURED 2026-01-01",
                            "decode_tok_s": 27.64,
                            "comparison_to_base": "PREFILL IS EQUIVALENT; decode is not comparable.",
                            "ctx_ceiling_note": "96K OOMs creating the MTP draft context.",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    record_measurement(
        path,
        "demo",
        {"status": "MEASURED today", "decode_tok_s": 25.32},
        owns=bench_mod.owns_measured_key,
    )
    measured = json.loads(path.read_text(encoding="utf-8"))["models"]["demo"]["measured"]

    # Prose the benchmark has nothing to say about is kept...
    assert measured["comparison_to_base"].startswith("PREFILL IS EQUIVALENT")
    assert measured["ctx_ceiling_note"].startswith("96K OOMs")
    # ...the figures it does produce are replaced, not merged...
    assert measured["decode_tok_s"] == 25.32
    assert measured["status"] == "MEASURED today"
    # ...and the file says which ones this run does not stand behind.
    assert measured["carried_forward"] == ["comparison_to_base", "ctx_ceiling_note"]


def test_a_stale_vram_figure_from_another_context_is_dropped(tmp_path: Path) -> None:
    """The bug the prefix rule exists to prevent.

    A previous run measured free VRAM at 64K. This run is at 48K and reports its
    own. Carrying the 64K key forward would leave two figures side by side, both
    looking current, differing only in a suffix nobody reads.
    """
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "demo": {
                        "measured": {
                            "vram_free_mib_at_64k": 3098,
                            "vram_free_breakdown": "old breakdown",
                            "ctx_ceiling_note": "keep me",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    record_measurement(
        path,
        "demo",
        {"vram_free_mib_at_48k": 5200, "vram_free_breakdown": "new breakdown"},
        owns=bench_mod.owns_measured_key,
    )
    measured = json.loads(path.read_text(encoding="utf-8"))["models"]["demo"]["measured"]

    assert "vram_free_mib_at_64k" not in measured
    assert measured["vram_free_mib_at_48k"] == 5200
    assert measured["vram_free_breakdown"] == "new breakdown"
    assert measured["ctx_ceiling_note"] == "keep me"
    assert "carried_forward" in measured


def test_carried_forward_does_not_accumulate_across_runs(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps({"models": {"demo": {"measured": {"note": "mine", "decode_tok_s": 1.0}}}}),
        encoding="utf-8",
    )

    for _ in range(3):
        record_measurement(path, "demo", {"decode_tok_s": 2.0}, owns=bench_mod.owns_measured_key)

    measured = json.loads(path.read_text(encoding="utf-8"))["models"]["demo"]["measured"]
    # `carried_forward` is itself owned, so it is recomputed rather than piling
    # up a record of every previous run's carry.
    assert measured["carried_forward"] == ["note"]


def test_without_an_owner_the_whole_block_is_still_replaced(
    registry_with_unmeasured_entry: Path,
) -> None:
    # The default stays a clean replace: a caller that has not thought about
    # ownership should not silently inherit merge semantics.
    record_measurement(registry_with_unmeasured_entry, "other", {"status": "MEASURED today"})
    measured = load(registry_with_unmeasured_entry).models["other"].measured
    assert measured == {"status": "MEASURED today"}
