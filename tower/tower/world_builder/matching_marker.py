"""The background solve's "matching committed" marker, and its log-only probes.

WHY (opt run 2026-11-03, lever L2; `TOWER_WORLD_BG_STOP_AT_MATCH`). At Stop the builder waits up
to 120 s for the background solve that is running and then kills it. Whatever that solve maps is
never used: the final solve empties `sparse/` and maps from scratch (`global_solve._map_candidate`).
What reaches the final solve is the feature database, which the background child extracts and
matches into, and its mapping only reads. So once a background child has committed everything it
writes into the database, the rest of the wait buys nothing the final solve can see, and the child
can be ended at once.

THE MARKER. With the switch on, the builder gives each background child it launches a fresh token
(`TOKEN_ENV`, in that child's environment only) and removes any earlier marker first. The child
(`scripts/world_solve.py`) writes `MARKER_FILENAME` into its solve workspace, atomically, once
`global_solve.solve` says everything it writes into the walk database is committed (`on_matched`):
the token, its pid, the database's digest (`global_solve.database_digest`) and when. The builder's
wait (`BackgroundSolver.wait`) reads it each poll and ends the child when the token is the one it
gave THIS child. A marker from another solve -- an earlier launch whose removal failed, a hand-run
solve -- carries another token, or none, and is ignored. No marker is today's wait, to the same cap.

SAFE WHEREVER THE KILL LANDS. Before the marker exists the wait is exactly today's (it may kill
the child at the cap, mid-matching, as today). After it, the child only reads the database (maps)
and publishes a background solution the final solve replaces. A kill between the commit and the
marker -- while the digest is computed, or mid atomic write -- leaves no marker, and so today's
wait, over a database that is already committed. The marker is written through
`storage.write_json_atomic` (staging file + replace, retried while a Windows reader holds the
destination) and read through `storage.read_json_closed`, so a reader never sees a torn one.

THE PROBES (`TOWER_WORLD_BG_STOP_AT_MATCH_PROBE`, debug only, LOG ONLY). Every solve child records
the walk database as it opens it, at its matching commit and when it finishes: the digest, the
SHA-256 of the database file and of its SQLite companions. Appended to `PROBE_FILENAME` beside the
database and logged to the child's stderr (the solve log). They change nothing a solve writes.

Nothing here raises: a marker or a probe that fails is a log line, and the solve goes on.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# The builder -> its own background child. Internal: never set by hand.
TOKEN_ENV = "TOWER_WORLD_BG_MATCH_TOKEN"
MARKER_FILENAME = "bg_matching_committed.json"
MARKER_RECORD = "wb-bg-matching-committed/1"
PROBE_FILENAME = "bg_stop_at_match.probe.jsonl"
PROBE_RECORD = "wb-bg-stop-at-match-probe/1"
DATABASE_FILENAME = "database.db"
# The database and the companions SQLite may leave beside it. `-shm` is a shared-memory index
# (rebuilt by the next opener), so only its size is recorded.
_COMPANIONS = ("", "-wal", "-journal", "-shm")
_HASHED = ("", "-wal", "-journal")
_CHUNK = 1 << 20


def marker_path(workspace_root) -> Path:
    return Path(workspace_root) / MARKER_FILENAME


def clear(workspace_root) -> None:
    """Remove the marker before a new launch. Never raises: a marker that cannot be removed
    carries another launch's token and is ignored (`read`)."""
    try:
        marker_path(workspace_root).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("matching marker: could not remove %s (%s); its token is not the next "
                       "launch's, so it is ignored", marker_path(workspace_root), exc)


def read(workspace_root, token: str | None) -> dict | None:
    """The marker THIS launch wrote (its token), or None: absent, unreadable, torn, another
    record, or another launch's token. Never raises."""
    if not token:
        return None
    path = marker_path(workspace_root)
    if not path.exists():
        return None
    try:
        from tower.storage import read_json_closed  # noqa: PLC0415

        record = read_json_closed(path)
    except Exception:  # noqa: BLE001 -- unreadable is "no marker": today's wait
        return None
    if not isinstance(record, dict) or record.get("record") != MARKER_RECORD:
        return None
    if record.get("token") != token:
        return None
    return record


def file_identity(database) -> dict:
    """{companion: {"size", "sha256"} or None} for the database file and its SQLite companions:
    what is on disk, byte for byte. Never raises (an unreadable file is `{"error": ...}`)."""
    out: dict = {}
    for suffix in _COMPANIONS:
        path = Path(str(database) + suffix)
        key = suffix or "db"
        try:
            size = path.stat().st_size
        except OSError:
            out[key] = None
            continue
        entry: dict = {"size": size}
        if suffix in _HASHED:
            try:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(_CHUNK), b""):
                        digest.update(chunk)
                entry["sha256"] = digest.hexdigest()
            except OSError as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
        out[key] = entry
    return out


def _digest(database) -> dict | None:
    from tower.world_builder.global_solve import database_digest  # noqa: PLC0415

    try:
        return database_digest(database)
    except Exception as exc:  # noqa: BLE001 -- evidence only
        return {"error": f"{type(exc).__name__}: {exc}"}


class ChildHook:
    """What one solve child does about the marker and the probes. `for_child` builds it."""

    def __init__(self, workspace_root, *, token: str | None, probe: bool, final: bool):
        self.root = Path(workspace_root)
        self.database = self.root / DATABASE_FILENAME
        self.token = token
        self.probe_on = probe
        self.final = final

    def solve_kwargs(self) -> dict:
        """The keyword `global_solve.solve` takes: its call once the database is committed."""
        return {"on_matched": self.matched}

    def opened(self) -> None:
        """Probe: the database as this solve finds it, before it touches it."""
        if self.probe_on:
            self._probe("opened")

    def matched(self, database_path=None) -> None:
        """Everything this solve writes into the walk database is committed. Write the marker
        (a background child the builder gave a token), then the probe. Never raises."""
        try:
            started = time.perf_counter()
            digest = _digest(self.database)
            digest_s = round(time.perf_counter() - started, 3)
            files = files_s = None
            if self.probe_on:
                # Before the marker: once the marker is there the builder may end this child, and
                # the bytes at the commit are the evidence. Its cost is recorded (`files_s`).
                started = time.perf_counter()
                files = file_identity(self.database)
                files_s = round(time.perf_counter() - started, 3)
            if self.token:
                from tower.storage import write_json_atomic  # noqa: PLC0415

                record = {"record": MARKER_RECORD, "token": self.token, "pid": os.getpid(),
                          "committed_at": time.time(), "database": self.database.name,
                          "mapped": Path(database_path).name if database_path else None,
                          "digest": digest, "digest_s": digest_s}
                if files is not None:
                    record["files"] = files
                    record["files_s"] = files_s
                write_json_atomic(marker_path(self.root), record)
            if self.probe_on:
                self._probe("matched", digest=digest, files=files,
                            seconds=digest_s + (files_s or 0.0))
        except Exception:  # noqa: BLE001 -- no marker is today's wait; the solve goes on
            logger.exception("matching marker: could not record the matching commit")

    def finished(self) -> None:
        """Probe: the database as this solve leaves it (after mapping and publishing)."""
        if self.probe_on:
            self._probe("finished")

    def _probe(self, stage: str, *, digest=None, files=None, seconds=None) -> None:
        try:
            started = time.perf_counter()
            if digest is None:
                digest = _digest(self.database)
            if files is None:
                files = file_identity(self.database)
            if seconds is None:
                seconds = round(time.perf_counter() - started, 3)
            record = {"probe": PROBE_RECORD, "stage": stage, "final": self.final,
                      "token": self.token, "pid": os.getpid(), "at": time.time(),
                      "digest": digest, "files": files, "seconds": seconds}
            from tower.storage import append_jsonl  # noqa: PLC0415

            append_jsonl(self.root / PROBE_FILENAME, record)
            db = (files or {}).get("db") or {}
            logger.warning(
                "[bg-stop-at-match probe] %s final=%s pid=%s content=%s verified_pairs=%s "
                "db_sha256=%s db_size=%s wal=%s journal=%s (%.2fs)",
                stage, self.final, os.getpid(), (digest or {}).get("content"),
                (digest or {}).get("verified_pairs"), db.get("sha256"), db.get("size"),
                (files or {}).get("-wal"), (files or {}).get("-journal"), seconds)
        except Exception:  # noqa: BLE001 -- a probe never changes the solve
            logger.exception("matching marker: probe %s failed", stage)


def for_child(store, world_id: str, session_id: str, *, final: bool) -> ChildHook | None:
    """The hook for a `world_solve.py` child, or None when it has nothing to do: no token (a
    final solve never takes one) and the probe off. None means the child runs exactly as before."""
    from tower.config import world_bg_stop_at_match_probe_setting  # noqa: PLC0415
    from tower.world_builder.global_solve import workspace_for  # noqa: PLC0415

    token = None if final else (os.environ.get(TOKEN_ENV) or "").strip() or None
    probe = world_bg_stop_at_match_probe_setting()
    if token is None and not probe:
        return None
    return ChildHook(workspace_for(store, world_id, session_id).root, token=token, probe=probe,
                     final=final)
