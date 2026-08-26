"""Surviving a restart.

Downloads and benchmarks lived only in the process that started them. Restarting
the backend lost both, and the two losses are not equally recoverable:

- A download's ``.part`` file survives on disk, so the *bytes* were never at
  risk — but the record naming them was, and a 9 GiB orphan with nothing in the
  UI pointing at it is indistinguishable from junk. The next attempt had to be
  started from the probe panel, by someone who remembered which repo it came
  from.
- A benchmark takes minutes of loaded GPU to produce, and its ``measured`` block
  had already been written into the registry. Losing the run record left the
  figure in ``models.json`` with nothing left to say how it was reached.

So state is written to disk as it changes. Two rules keep that from becoming its
own source of bugs:

**The file is a record, never the authority.** What is on disk wins on restore —
a download's progress is re-read from the ``.part`` file rather than trusted from
the number that was saved, because the process may have died between a write and
a save. The saved figure is at best equal to the file and at worst stale.

**A broken state file must never stop the app from starting.** It is a
convenience, not data the user typed. An unreadable one is moved aside and
reported, and Headroom comes up empty rather than not at all.

Nothing here locks: a second Headroom against the same state directory would
have its writes overwritten by the first. That is out of scope deliberately —
two instances would already be fighting over the same ``.part`` files and the
same registry, and a lock would make that look supported.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# How many finished records to keep. Enough to compare a handful of runs
# against each other, few enough that the file stays something a person can
# read. Unfinished records are never pruned -- they are the ones with work
# still attached to them.
KEEP_FINISHED = 50


class JsonStore:
    """A list of records in a JSON file, written atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            # utf-8-sig, not utf-8: these files are small enough to edit by hand
            # and this is Windows, where `Out-File -Encoding utf8` and Notepad
            # both prepend a BOM. Reading as plain utf-8 turns that invisible
            # byte into "unreadable state file", and the download it named
            # disappears from the UI for a reason nobody can see.
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            spoiled = self.path.with_suffix(self.path.suffix + ".corrupt")
            log.warning("unreadable state file %s (%s); moved to %s", self.path, exc, spoiled)
            try:
                self.path.replace(spoiled)
            except OSError:
                pass
            return []
        records = data.get("records") if isinstance(data, dict) else data
        if not isinstance(records, list):
            log.warning("ignoring state file %s: expected a list of records", self.path)
            return []
        return [r for r in records if isinstance(r, dict)]

    def save(self, records: list[dict[str, Any]]) -> None:
        """Replace the file's contents. Failure is logged, never raised.

        A state save happens on the same path as real work -- a download
        finishing, a benchmark starting -- and none of that should fail because
        a bookkeeping file could not be written.
        """
        payload = {"saved_at": time.time(), "records": records}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # newline="" so the file is not rewritten line-by-line on Windows.
            # It is compared against itself often enough (and by eye) that a
            # whole-file diff on every save is worth avoiding.
            with tmp.open("w", encoding="utf-8", newline="") as fh:
                json.dump(payload, fh, indent=2)
                fh.write("\n")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("could not save state to %s: %s", self.path, exc)
            tmp.unlink(missing_ok=True)


def prune(
    records: list[dict[str, Any]], *, finished: set[str], keep: int = KEEP_FINISHED
) -> list[dict[str, Any]]:
    """Keep every unfinished record, and the most recent `keep` finished ones.

    Records arrive newest first. Unfinished ones are exempt from the limit
    because each still has something attached to it -- a `.part` file to resume,
    a run to explain -- and dropping one to make room for history would discard
    the half that is still actionable.
    """
    kept: list[dict[str, Any]] = []
    seen_finished = 0
    for record in records:
        if record.get("status") in finished:
            seen_finished += 1
            if seen_finished > keep:
                continue
        kept.append(record)
    return kept
