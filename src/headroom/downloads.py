"""Resumable model downloads with stall detection.

Written the way it is because of a specific failure. A `huggingface_hub`
snapshot download on the development machine **stalled dead at 9.3 GiB**: the
process stayed alive, the file stopped growing, network throughput sat at zero,
nothing timed out and nothing raised. It would have hung forever, and the only
symptom was a progress bar that had quietly stopped.

That is the failure mode this module is built to make impossible.

- **Progress is judged by bytes written, not by the socket being open.** A
  connection that is technically alive while delivering nothing is the exact case
  that hung, so a transfer that stops growing for `STALL_SECONDS` is aborted and
  retried rather than waited on.
- **Every attempt resumes.** Partial data lands in a `.part` file and the next
  attempt continues from that offset, so a stall costs seconds rather than
  gigabytes.
- **The final name appears only on success.** A file is renamed into place after
  its size is verified, so a half-transferred model can never be mistaken for a
  complete one — by this app, by a shell script, or by llama.cpp.
- **A restart does not lose the transfer.** Records are written to disk as they
  change (:mod:`headroom.store`), and one that was in flight when the process
  died comes back as ``interrupted`` rather than vanishing. Its progress is
  re-read from the ``.part`` file, never trusted from the saved number: the
  process may have died between a chunk landing and the record being saved, so
  the file is at best equal to what was recorded and at worst ahead of it.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

from .store import JsonStore, prune

log = logging.getLogger(__name__)

# No growth for this long means the transfer is wedged, not merely slow.
STALL_SECONDS = 30.0
MAX_ATTEMPTS = 20
CHUNK_BYTES = 1024 * 1024


class DownloadStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # Headroom stopped while this was transferring. Distinct from `failed`,
    # which means the transfer itself gave up: nothing went wrong here, and the
    # bytes already on disk are still good.
    INTERRUPTED = "interrupted"


#: Nothing is working on these any more.
FINISHED = frozenset(
    {
        DownloadStatus.COMPLETE.value,
        DownloadStatus.FAILED.value,
        DownloadStatus.CANCELLED.value,
        DownloadStatus.INTERRUPTED.value,
    }
)

#: These can be picked up again, from whatever the `.part` file already holds.
RESUMABLE = frozenset({DownloadStatus.INTERRUPTED, DownloadStatus.FAILED, DownloadStatus.CANCELLED})


@dataclass(slots=True)
class Download:
    id: str
    repo: str
    filename: str
    dest: Path
    status: DownloadStatus = DownloadStatus.QUEUED
    total_bytes: int | None = None
    downloaded_bytes: int = 0
    attempt: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    _recent: list[tuple[float, int]] = field(default_factory=list, repr=False)

    @property
    def part_path(self) -> Path:
        return self.dest.with_suffix(self.dest.suffix + ".part")

    @property
    def percent(self) -> float | None:
        if not self.total_bytes:
            return None
        return min(100.0, self.downloaded_bytes / self.total_bytes * 100.0)

    @property
    def bytes_per_second(self) -> float | None:
        """Throughput over a short trailing window.

        A running average over the whole transfer would keep reporting a healthy
        figure for minutes after throughput collapsed, which is precisely the
        reassurance that made the original hang hard to spot.
        """
        if len(self._recent) < 2:
            return None
        (t0, b0), (t1, b1) = self._recent[0], self._recent[-1]
        span = t1 - t0
        return (b1 - b0) / span if span > 0 else None

    @property
    def eta_seconds(self) -> float | None:
        rate = self.bytes_per_second
        if not rate or rate <= 0 or not self.total_bytes:
            return None
        return max(0.0, (self.total_bytes - self.downloaded_bytes) / rate)

    def note_progress(self) -> None:
        now = time.time()
        self._recent.append((now, self.downloaded_bytes))
        # Keep roughly the last 15 seconds.
        while len(self._recent) > 2 and now - self._recent[0][0] > 15:
            self._recent.pop(0)

    @property
    def resumable(self) -> bool:
        return self.status in RESUMABLE

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "repo": self.repo,
            "filename": self.filename,
            "dest": str(self.dest),
            "status": self.status.value,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "percent": self.percent,
            "bytes_per_second": self.bytes_per_second,
            "eta_seconds": self.eta_seconds,
            "attempt": self.attempt,
            "error": self.error,
            "resumable": self.resumable,
            "elapsed_seconds": round((self.finished_at or time.time()) - self.started_at, 1),
        }

    def to_record(self) -> dict:
        """The durable half: what was asked for and how far it got.

        Rate and ETA are deliberately absent. They describe a transfer that is
        moving, and reloading them after a restart would put a throughput figure
        next to a download that has not sent a byte in three days.
        """
        return {
            "id": self.id,
            "repo": self.repo,
            "filename": self.filename,
            "dest": str(self.dest),
            "status": self.status.value,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "attempt": self.attempt,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_record(cls, record: dict) -> Download | None:
        """Rebuild from a saved record, reconciled against the disk.

        The disk wins. A record saying `running` only means the process did not
        live long enough to write anything else, and its byte count is whatever
        was true at the last save -- so the file is consulted for both. A
        transfer that actually finished (the rename landed, the save did not)
        comes back complete rather than as a phantom to resume.
        """
        try:
            # An empty destination would resolve to the current directory, which
            # exists -- and a record whose file "already exists" comes back
            # complete. A malformed record must not become a phantom download of
            # something nobody asked for.
            if not str(record.get("dest") or "").strip():
                raise ValueError("record has no destination path")
            status = DownloadStatus(record["status"])
            d = cls(
                id=str(record["id"]),
                repo=str(record["repo"]),
                filename=str(record["filename"]),
                dest=Path(record["dest"]),
                status=status,
                total_bytes=record.get("total_bytes"),
                downloaded_bytes=int(record.get("downloaded_bytes") or 0),
                attempt=int(record.get("attempt") or 0),
                error=record.get("error"),
                started_at=float(record.get("started_at") or time.time()),
                finished_at=record.get("finished_at"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("ignoring unreadable download record: %s", exc)
            return None

        if status in (DownloadStatus.QUEUED, DownloadStatus.RUNNING):
            if d.dest.exists():
                size = d.dest.stat().st_size
                d.status = DownloadStatus.COMPLETE
                d.total_bytes = d.total_bytes or size
                d.downloaded_bytes = size
                d.finished_at = d.finished_at or time.time()
            else:
                d.status = DownloadStatus.INTERRUPTED
                d.downloaded_bytes = d.part_path.stat().st_size if d.part_path.exists() else 0
                d.finished_at = d.finished_at or time.time()
                d.error = None
        return d


class DownloadManager:
    def __init__(
        self, endpoint: str = "https://huggingface.co", store: JsonStore | None = None
    ) -> None:
        self.endpoint = endpoint
        self._downloads: dict[str, Download] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._store = store

    # ------------------------------------------------------------- persistence
    def _persist(self) -> None:
        if self._store is None:
            return
        self._store.save(prune([d.to_record() for d in self.list()], finished=FINISHED))

    def restore(self) -> int:
        """Reload records written by a previous run. Returns how many came back.

        Nothing is restarted here. A download that was in flight comes back as
        `interrupted` and waits to be resumed deliberately: the process may have
        been stopped hours ago, and a backend that starts pulling gigabytes off
        the network because of something someone clicked on Tuesday is not a
        backend anyone would leave running.
        """
        if self._store is None:
            return 0
        restored = 0
        for record in self._store.load():
            d = Download.from_record(record)
            if d is None or d.id in self._downloads:
                continue
            self._downloads[d.id] = d
            restored += 1
        if restored:
            interrupted = sum(
                1 for d in self._downloads.values() if d.status is DownloadStatus.INTERRUPTED
            )
            log.info(
                "restored %d download record(s); %d interrupted and resumable",
                restored,
                interrupted,
            )
            # Write the reconciled view straight back, so the file stops
            # claiming that something is running inside a process that is gone.
            self._persist()
        return restored

    # ------------------------------------------------------------------- query
    def list(self) -> list[Download]:
        return sorted(self._downloads.values(), key=lambda d: d.started_at, reverse=True)

    def get(self, download_id: str) -> Download | None:
        return self._downloads.get(download_id)

    def active_for(self, dest: Path) -> Download | None:
        for d in self._downloads.values():
            if d.dest == dest and d.status in (DownloadStatus.QUEUED, DownloadStatus.RUNNING):
                return d
        return None

    # ------------------------------------------------------------------ change
    def start(self, repo: str, filename: str, dest: Path) -> Download:
        existing = self.active_for(dest)
        if existing:
            return existing

        download = Download(id=uuid.uuid4().hex[:12], repo=repo, filename=filename, dest=dest)
        self._downloads[download.id] = download
        self._launch(download)
        return download

    def resume(self, download_id: str) -> Download:
        """Pick a stopped transfer up again, on its original record.

        Deliberately not a new record. The `.part` file, the repo and the
        destination are the same ones, and a second entry claiming to be a fresh
        download of the same file would hide the history that explains why there
        are already 9 GiB on disk.
        """
        d = self._downloads.get(download_id)
        if d is None:
            raise DownloadError("no such download")
        if not d.resumable:
            raise DownloadError(f"a {d.status.value} download cannot be resumed")
        if (other := self.active_for(d.dest)) is not None:
            raise DownloadError(f"{other.filename} is already downloading to that path")

        d.status = DownloadStatus.QUEUED
        d.error = None
        d.finished_at = None
        d.attempt = 0
        d.downloaded_bytes = d.part_path.stat().st_size if d.part_path.exists() else 0
        self._launch(d)
        return d

    def _launch(self, d: Download) -> None:
        self._tasks[d.id] = asyncio.create_task(self._run(d))
        self._persist()

    def cancel(self, download_id: str) -> bool:
        task = self._tasks.get(download_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def _run(self, d: Download) -> None:
        url = f"{self.endpoint}/{d.repo}/resolve/main/{d.filename}"
        d.dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            if d.dest.exists():
                d.status = DownloadStatus.COMPLETE
                d.total_bytes = d.downloaded_bytes = d.dest.stat().st_size
                d.finished_at = time.time()
                self._persist()
                return

            d.status = DownloadStatus.RUNNING
            self._persist()
            for attempt in range(1, MAX_ATTEMPTS + 1):
                d.attempt = attempt
                try:
                    await self._attempt(d, url)
                except asyncio.CancelledError:
                    raise
                except _Stalled:
                    log.warning(
                        "download %s stalled at %d bytes; resuming (attempt %d)",
                        d.id,
                        d.downloaded_bytes,
                        attempt,
                    )
                    await asyncio.sleep(min(3 * attempt, 15))
                    continue
                except (httpx.HTTPError, OSError) as exc:
                    d.error = str(exc)
                    log.warning("download %s attempt %d failed: %s", d.id, attempt, exc)
                    await asyncio.sleep(min(3 * attempt, 15))
                    continue

                # Verify before promoting the .part to its real name. A truncated
                # file with the right name is worse than an obvious failure.
                size = d.part_path.stat().st_size
                if d.total_bytes and size != d.total_bytes:
                    d.error = f"size mismatch: got {size}, expected {d.total_bytes}"
                    continue

                d.part_path.replace(d.dest)
                d.status = DownloadStatus.COMPLETE
                d.downloaded_bytes = size
                d.error = None
                d.finished_at = time.time()
                self._persist()
                log.info("download %s complete: %s", d.id, d.dest)
                return

            d.status = DownloadStatus.FAILED
            d.error = d.error or f"gave up after {MAX_ATTEMPTS} attempts"
            d.finished_at = time.time()
            self._persist()

        except asyncio.CancelledError:
            d.status = DownloadStatus.CANCELLED
            d.finished_at = time.time()
            self._persist()
            # The .part file is deliberately kept: a cancelled download resumes
            # from where it stopped rather than starting over.
            log.info(
                "download %s cancelled at %d bytes (partial file kept)", d.id, d.downloaded_bytes
            )
            raise
        except Exception as exc:
            d.status = DownloadStatus.FAILED
            d.error = str(exc)
            d.finished_at = time.time()
            self._persist()
            log.exception("download %s failed", d.id)

    async def _attempt(self, d: Download, url: str) -> None:
        resume_from = d.part_path.stat().st_size if d.part_path.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        d.downloaded_bytes = resume_from

        timeout = httpx.Timeout(connect=30.0, read=STALL_SECONDS, write=30.0, pool=30.0)
        async with (
            httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client,
            client.stream("GET", url, headers=headers) as resp,
        ):
            if resp.status_code == 416:
                # Range beyond the end: the part file is already complete or
                # corrupt. Treat as complete and let the size check decide.
                d.total_bytes = d.total_bytes or resume_from
                return
            if resp.status_code not in (200, 206):
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )

            if resp.status_code == 200 and resume_from:
                # Server ignored the range header; start over rather than
                # appending fresh bytes onto old ones and corrupting the file.
                log.warning("download %s: server ignored resume, restarting", d.id)
                d.part_path.unlink(missing_ok=True)
                resume_from = 0
                d.downloaded_bytes = 0

            content_length = resp.headers.get("content-length")
            if content_length:
                d.total_bytes = int(content_length) + resume_from
                # Saved now rather than at the end: it is what turns a restored
                # byte count back into a percentage, and it is known here.
                self._persist()

            last_growth = time.monotonic()
            mode = "ab" if resume_from else "wb"
            with d.part_path.open(mode) as fh:
                async for chunk in resp.aiter_bytes(CHUNK_BYTES):
                    if chunk:
                        fh.write(chunk)
                        d.downloaded_bytes += len(chunk)
                        d.note_progress()
                        last_growth = time.monotonic()
                    elif time.monotonic() - last_growth > STALL_SECONDS:
                        raise _Stalled


class DownloadError(RuntimeError):
    pass


class _Stalled(Exception):
    """The transfer stopped growing while the connection stayed open."""
