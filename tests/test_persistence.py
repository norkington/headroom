"""What survives a restart, and what deliberately does not.

Downloads and benchmarks used to live only in the process that started them.
Restarting the backend lost a 9 GiB transfer's identity — the bytes stayed in
the `.part` file, but nothing in the UI knew which repo they came from — and
lost the run record behind a figure that had already been written into
`models.json`.

The tests here are written against the two ways this could go wrong rather than
against the current shape of the code:

- **Trusting the file over the disk.** A saved byte count is at best equal to
  the `.part` file and at worst stale, because the process can die between a
  chunk landing and a record being saved. Every restored progress figure is
  re-read from disk.
- **Restoring a partial run as if it were a result.** A benchmark cut short
  measures the part of the workload it reached, which is precisely what the
  warm-up and prefill rules say not to average. A crash is not a better outcome
  than a cancel, and a cancel records nothing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from headroom import bench as bench_mod
from headroom import downloads as dl_mod
from headroom.store import KEEP_FINISHED, JsonStore, prune

_REAL_ASYNC_CLIENT = httpx.AsyncClient


# ----------------------------------------------------------------- the store


def test_a_state_file_round_trips(tmp_path: Path) -> None:
    store = JsonStore(tmp_path / "downloads.json")
    store.save([{"id": "abc", "status": "complete"}])

    assert store.load() == [{"id": "abc", "status": "complete"}]


def test_a_missing_state_file_is_simply_empty(tmp_path: Path) -> None:
    assert JsonStore(tmp_path / "nothing.json").load() == []


def test_an_unreadable_state_file_does_not_stop_headroom_starting(tmp_path: Path) -> None:
    # A state file is a convenience, not something the user typed. Coming up
    # empty is recoverable; refusing to start over a bookkeeping file is not.
    path = tmp_path / "downloads.json"
    path.write_text("{ this is not json", encoding="utf-8")
    store = JsonStore(path)

    assert store.load() == []
    # Moved aside rather than deleted: whatever was in it may still explain
    # something, and silently destroying it would be the second surprise.
    assert (tmp_path / "downloads.json.corrupt").exists()


def test_saving_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    store = JsonStore(tmp_path / "downloads.json")
    store.save([{"id": "abc", "status": "running"}])
    store.save([{"id": "abc", "status": "complete"}])

    assert sorted(p.name for p in tmp_path.iterdir()) == ["downloads.json"]


def test_pruning_never_drops_a_record_that_still_has_work_attached() -> None:
    records = [{"id": f"f{i}", "status": "complete"} for i in range(KEEP_FINISHED + 20)]
    records += [{"id": "live", "status": "running"}, {"id": "part", "status": "interrupted"}]

    kept = prune(records, finished={"complete"})

    assert sum(1 for r in kept if r["status"] == "complete") == KEEP_FINISHED
    assert {"live", "part"} <= {r["id"] for r in kept}


def test_pruning_keeps_the_most_recent_finished_records() -> None:
    # Records arrive newest first, so the survivors are the head of the list.
    records = [{"id": f"f{i}", "status": "complete"} for i in range(KEEP_FINISHED + 5)]

    kept = prune(records, finished={"complete"})

    assert [r["id"] for r in kept] == [f"f{i}" for i in range(KEEP_FINISHED)]


# -------------------------------------------------------------- downloads


def _write_download_record(store: JsonStore, **overrides) -> dict:
    record = {
        "id": "d1",
        "repo": "unsloth/Qwen3.8-27B-GGUF",
        "filename": "Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "dest": "",
        "status": "running",
        "total_bytes": 8192,
        "downloaded_bytes": 0,
        "attempt": 1,
        "error": None,
        "started_at": time.time() - 60,
        "finished_at": None,
    }
    record.update(overrides)
    store.save([record])
    return record


def test_a_transfer_that_was_running_comes_back_interrupted(tmp_path: Path) -> None:
    dest = tmp_path / "model.gguf"
    part = tmp_path / "model.gguf.part"
    part.write_bytes(b"x" * 4096)
    store = JsonStore(tmp_path / "downloads.json")
    # The saved figure is deliberately wrong. The process can die between a
    # chunk landing and the record being written, so the file is the authority
    # and this number must not be the one that comes back.
    _write_download_record(store, dest=str(dest), downloaded_bytes=17)

    manager = dl_mod.DownloadManager(store=store)
    assert manager.restore() == 1

    (d,) = manager.list()
    assert d.status is dl_mod.DownloadStatus.INTERRUPTED
    assert d.downloaded_bytes == 4096
    assert d.resumable is True
    # `interrupted` is not `failed`: nothing went wrong with the transfer, and
    # an error line would send someone looking for a fault that is not there.
    assert d.error is None


def test_a_transfer_whose_file_landed_before_the_crash_comes_back_complete(
    tmp_path: Path,
) -> None:
    # The rename into place happened; the save did not. The file is on disk and
    # usable, so offering to resume it would be an invitation to redownload
    # something the user already has.
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"x" * 8192)
    store = JsonStore(tmp_path / "downloads.json")
    _write_download_record(store, dest=str(dest))

    manager = dl_mod.DownloadManager(store=store)
    manager.restore()

    (d,) = manager.list()
    assert d.status is dl_mod.DownloadStatus.COMPLETE
    assert d.downloaded_bytes == 8192


def test_restoring_rewrites_the_file_so_nothing_still_claims_to_be_running(
    tmp_path: Path,
) -> None:
    store = JsonStore(tmp_path / "downloads.json")
    _write_download_record(store, dest=str(tmp_path / "model.gguf"))

    dl_mod.DownloadManager(store=store).restore()

    (saved,) = store.load()
    assert saved["status"] == "interrupted"


def test_a_record_that_cannot_be_read_is_skipped_not_fatal(tmp_path: Path) -> None:
    store = JsonStore(tmp_path / "downloads.json")
    store.save(
        [
            {"id": "broken"},
            {
                "id": "ok",
                "repo": "r",
                "filename": "f",
                "dest": str(tmp_path / "a.gguf"),
                "status": "complete",
            },
        ]
    )

    manager = dl_mod.DownloadManager(store=store)

    assert manager.restore() == 1
    assert manager.list()[0].id == "ok"


@pytest.mark.asyncio
async def test_download_records_are_saved_as_they_finish(tmp_path: Path) -> None:
    store = JsonStore(tmp_path / "downloads.json")
    manager = dl_mod.DownloadManager(store=store)
    # A file already at the destination is the shortest complete path through
    # the runner, and it still has to leave a record behind.
    dest = tmp_path / "already-here.gguf"
    dest.write_bytes(b"x" * 32)

    d = manager.start("some/repo", "already-here.gguf", dest)
    await manager._tasks[d.id]

    (saved,) = store.load()
    assert saved["status"] == "complete"
    assert saved["repo"] == "some/repo"


# ------------------------------------------------------------------ resuming


def _serving(body: bytes, seen: list[httpx.Request]):
    """A range-aware host holding `body`, recording what was asked of it."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        start = 0
        if (rng := request.headers.get("Range")) is not None:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
        chunk = body[start:]
        return httpx.Response(
            206 if start else 200,
            headers={"content-length": str(len(chunk))},
            content=chunk,
        )

    return handler


@pytest.fixture
def patch_httpx(monkeypatch):
    def install(handler, module: str = "headroom.downloads"):
        def factory(*_args, **_kwargs):
            return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), timeout=5.0)

        monkeypatch.setattr(f"{module}.httpx.AsyncClient", factory)

    return install


@pytest.mark.asyncio
async def test_resuming_continues_from_the_bytes_already_on_disk(
    tmp_path: Path, patch_httpx
) -> None:
    body = bytes(range(256)) * 32  # 8 KiB
    seen: list[httpx.Request] = []
    patch_httpx(_serving(body, seen))

    dest = tmp_path / "model.gguf"
    (tmp_path / "model.gguf.part").write_bytes(body[:5000])
    store = JsonStore(tmp_path / "downloads.json")
    _write_download_record(store, dest=str(dest), total_bytes=len(body))

    manager = dl_mod.DownloadManager(store=store)
    manager.restore()
    (restored,) = manager.list()
    manager.resume(restored.id)
    await manager._tasks[restored.id]

    # The point of the whole exercise: 5000 bytes were not fetched again.
    assert seen[0].headers["Range"] == "bytes=5000-"
    assert dest.read_bytes() == body
    assert restored.status is dl_mod.DownloadStatus.COMPLETE
    assert store.load()[0]["status"] == "complete"


@pytest.mark.asyncio
async def test_resuming_keeps_the_original_record_rather_than_making_a_second(
    tmp_path: Path, patch_httpx
) -> None:
    # A fresh entry claiming to be a new download would hide the history that
    # explains why there are already bytes on disk.
    body = b"y" * 4096
    patch_httpx(_serving(body, []))
    dest = tmp_path / "model.gguf"
    (tmp_path / "model.gguf.part").write_bytes(body[:1000])
    store = JsonStore(tmp_path / "downloads.json")
    _write_download_record(store, dest=str(dest), total_bytes=len(body), attempt=4)

    manager = dl_mod.DownloadManager(store=store)
    manager.restore()
    (restored,) = manager.list()
    resumed = manager.resume(restored.id)
    await manager._tasks[restored.id]

    assert resumed.id == restored.id
    assert len(manager.list()) == 1


def test_a_download_that_is_already_running_cannot_be_resumed(tmp_path: Path) -> None:
    store = JsonStore(tmp_path / "downloads.json")
    manager = dl_mod.DownloadManager(store=store)
    d = dl_mod.Download(
        id="live",
        repo="r",
        filename="f.gguf",
        dest=tmp_path / "f.gguf",
        status=dl_mod.DownloadStatus.RUNNING,
    )
    manager._downloads[d.id] = d

    with pytest.raises(dl_mod.DownloadError, match="cannot be resumed"):
        manager.resume("live")


def test_resuming_something_that_never_existed_says_so(tmp_path: Path) -> None:
    manager = dl_mod.DownloadManager(store=JsonStore(tmp_path / "downloads.json"))

    with pytest.raises(dl_mod.DownloadError, match="no such download"):
        manager.resume("nope")


# -------------------------------------------------------------- benchmarks


def _bench_record(**overrides) -> dict:
    record = {
        "id": "b1",
        "model_key": "qwen38-unleashed",
        "model_path": "C:/models/q.gguf",
        "port": 8080,
        "reps": 3,
        "warmup": 3,
        "max_tokens": 512,
        "prefill_tokens": 6000,
        "status": "complete",
        "n_ctx": 65536,
        "runs_done": 15,
        "runs_total": 15,
        "per_task": {"impl": {"decode_tok_s": 25.32, "acceptance": 0.41, "runs": 3}},
        "result": {"decode_tok_s": 25.32, "decode_sd": 4.05, "prefill_tok_s": 893.6},
        "written": True,
        "error": None,
        "started_at": time.time() - 300,
        "finished_at": time.time() - 43,
    }
    record.update(overrides)
    return record


def test_a_finished_benchmark_survives_a_restart(tmp_path: Path) -> None:
    # The registry keeps the figures. It does not keep the rep count, the
    # acceptance spread or the per-task breakdown that say how much weight
    # those figures carry -- which is the half a reader needs to compare runs.
    store = JsonStore(tmp_path / "benchmarks.json")
    store.save([_bench_record()])

    runner = bench_mod.BenchmarkRunner(store=store)
    assert runner.restore() == 1

    (b,) = runner.list()
    assert b.status is bench_mod.BenchStatus.COMPLETE
    assert b.result["prefill_tok_s"] == 893.6
    assert b.per_task["impl"]["runs"] == 3
    assert b.written is True


def test_a_run_interrupted_by_a_restart_is_kept_as_an_attempt_not_a_result(
    tmp_path: Path,
) -> None:
    store = JsonStore(tmp_path / "benchmarks.json")
    store.save(
        [
            _bench_record(
                status="running",
                runs_done=7,
                result={"decode_tok_s": 25.32},
                per_task={"impl": {"decode_tok_s": 25.32, "runs": 3}},
                finished_at=None,
            )
        ]
    )

    runner = bench_mod.BenchmarkRunner(store=store)
    runner.restore()

    (b,) = runner.list()
    assert b.status is bench_mod.BenchStatus.INTERRUPTED
    # Cancelling records nothing because a partial run is not a measurement. A
    # crash is not a better outcome than a cancel, so it gets the same answer:
    # warm-up may not have finished, prefill certainly did not run, and a
    # decode figure from some of the tasks is a different measurement from one
    # taken across all of them.
    assert b.result is None
    assert b.per_task == {}
    assert "partial run is not a measurement" in b.error


def test_an_interrupted_run_does_not_block_the_next_one(tmp_path: Path) -> None:
    # `active()` gates starting a benchmark. A restored record left looking
    # `running` would refuse every future run, in a process that has nothing
    # running at all.
    store = JsonStore(tmp_path / "benchmarks.json")
    store.save([_bench_record(status="running", finished_at=None)])

    runner = bench_mod.BenchmarkRunner(store=store)
    runner.restore()

    assert runner.active() is None


@pytest.mark.asyncio
async def test_a_run_is_saved_when_it_starts_not_only_when_it_finishes(
    tmp_path: Path, patch_httpx
) -> None:
    # Saving only on completion would leave exactly the case this feature is
    # for -- a run killed halfway -- with nothing on disk to restore.
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    patch_httpx(refuse, module="headroom.bench")
    store = JsonStore(tmp_path / "benchmarks.json")
    runner = bench_mod.BenchmarkRunner(store=store)

    job = runner.start(model_key="demo", model_path="/w/fake.gguf", port=8080, reps=1, warmup=0)
    assert store.load()[0]["status"] in ("queued", "running")

    await runner._tasks[job.id]
    (saved,) = store.load()
    assert saved["status"] == "failed"
    assert saved["result"] is None


# ------------------------------------------------------------------- the API


def _client(monkeypatch, tmp_path: Path):
    from fastapi.testclient import TestClient

    from headroom.app import Settings, create_app

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("HEADROOM_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app(Settings.resolve()))


def test_the_api_offers_an_interrupted_download_back(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (tmp_path / "model.gguf.part").write_bytes(b"z" * 2048)
    JsonStore(state / "downloads.json").save(
        [
            {
                "id": "d1",
                "repo": "unsloth/Qwen3.8-27B-GGUF",
                "filename": "model.gguf",
                "dest": str(tmp_path / "model.gguf"),
                "status": "running",
                "total_bytes": 8192,
                "downloaded_bytes": 0,
                "attempt": 2,
                "error": None,
                "started_at": time.time() - 90,
                "finished_at": None,
            }
        ]
    )

    with _client(monkeypatch, tmp_path) as client:
        body = client.get("/api/downloads").json()
        (d,) = body["downloads"]
        assert d["status"] == "interrupted"
        assert d["resumable"] is True
        assert d["downloaded_bytes"] == 2048
        assert d["percent"] == pytest.approx(25.0)

        # And the state directory is reported, for the same reason every other
        # path Headroom resolved is.
        assert client.get("/api/health").json()["state_dir"] == str(state)


def test_resuming_a_download_the_api_has_never_heard_of_is_a_404(
    monkeypatch, tmp_path: Path
) -> None:
    with _client(monkeypatch, tmp_path) as client:
        resp = client.post("/api/downloads/nope/resume")

    assert resp.status_code == 404


def test_the_state_directory_is_not_created_until_there_is_something_to_put_in_it(
    monkeypatch, tmp_path: Path
) -> None:
    # A fresh install should not accumulate empty scaffolding for features it
    # has not used.
    with _client(monkeypatch, tmp_path) as client:
        client.get("/api/downloads")

    assert not (tmp_path / "state").exists()


def test_a_state_file_is_json_a_person_can_read(tmp_path: Path) -> None:
    store = JsonStore(tmp_path / "downloads.json")
    store.save([{"id": "abc", "status": "complete"}])

    text = (tmp_path / "downloads.json").read_text(encoding="utf-8")
    assert "\n" in text
    assert json.loads(text)["records"][0]["id"] == "abc"


def test_a_record_with_no_destination_does_not_become_a_phantom(tmp_path: Path) -> None:
    # An empty path resolves to the current directory, which exists -- so a
    # naive existence check would report this as a completed download of
    # something nobody asked for.
    store = JsonStore(tmp_path / "downloads.json")
    _write_download_record(store, dest="")

    manager = dl_mod.DownloadManager(store=store)

    assert manager.restore() == 0
    assert manager.list() == []


def test_a_byte_order_mark_does_not_make_a_state_file_unreadable(tmp_path: Path) -> None:
    # Windows puts one there for free: PowerShell 5.1's `Out-File -Encoding
    # utf8` and Notepad both prepend a BOM. Read as plain utf-8 that is a parse
    # error, and the download the file named vanishes from the UI over one
    # invisible byte. Found by seeding a state file from PowerShell.
    path = tmp_path / "downloads.json"
    path.write_text('{"records": [{"id": "abc", "status": "complete"}]}', encoding="utf-8-sig")

    assert JsonStore(path).load() == [{"id": "abc", "status": "complete"}]
    assert not (tmp_path / "downloads.json.corrupt").exists()
