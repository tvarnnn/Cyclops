"""The background solve's "matching committed" marker, its horizon snapshot, and its log-only probes.

WHY (opt run 2026-11-03, lever L2; `TOWER_WORLD_BG_STOP_AT_MATCH`). At Stop the builder waits up
to 120 s for the background solve that is running and then kills it. Whatever that solve maps is
never used by the final solve, which empties `sparse/` and maps from scratch
(`global_solve._map_candidate`). What reaches the final solve is the feature database, which the
background child extracts and matches into and its mapping only reads. So once a background child
has committed everything it writes into the database, the rest of the wait buys the final solve
nothing, and the child can be ended at once.

THE MARKER. With the switch on, the builder gives each background child it launches a fresh token
(`TOKEN_ENV`, in that child's environment only) and removes any earlier marker first. The child
(`scripts/world_solve.py`) writes `MARKER_FILENAME` into its solve workspace, atomically, when
`global_solve.solve` calls `on_matched`: the matching call has RETURNED WITHOUT ERROR (an exception
never reaches the hook) and every write the solve makes into its databases is behind it. The hook
then proves no connection still holds the database for writing (`writer_free`: a zero-timeout
`BEGIN IMMEDIATE`, rolled back) and only then writes the marker: the token, its pid, when, and what a
deferred map of this horizon needs (the keyframe ids it read, its timings). Anything that fails
first -- the writer check, the digest in probe mode, the write itself -- leaves NO marker, which is
today's wait (fail closed). The builder's wait (`BackgroundSolver.wait`) accepts a marker only with
THIS child's token and every field `read` checks.

THE HORIZON SNAPSHOT AND THE DEFERRED MAP. Ending the child early must never leave an older world
standing than the switch off would. If the final solve then fails, is withheld or produces nothing,
the default path could have published THIS child's map (it would have finished inside the cap). So
the builder snapshots the committed database right after the child and its descendants have exited
(`snapshot_horizon`, before the final solve opens it), and in those cases maps that snapshot with
the child's own keyframes (`global_solve.map_committed_horizon`, through `world_solve.py
--deferred-map`) inside the time the default path would have given the child. See
`BackgroundSolver.run_final`.

SAFE WHEREVER THE KILL LANDS. Before the marker exists the wait is exactly today's (it may kill the
child at the cap, mid-matching, as today). After it, the child only reads the database (maps) and
publishes a background solution the final solve replaces. A kill between the commit and the marker
leaves no marker -- today's wait, over a database that is already committed -- and at most a staging
file the next solve sweeps. The marker is written through `storage.write_json_atomic` (staging file
+ replace, retried while a Windows reader holds the destination) and read through
`storage.read_json_closed`.

THE PROBES (`TOWER_WORLD_BG_STOP_AT_MATCH_PROBE`, debug only, LOG ONLY). Every solve child records
the walk database as it opens it, at its matching commit and when it finishes: the content digest
(`global_solve.database_digest`), the SHA-256 of the database file and of its SQLite companions.
Appended to `PROBE_FILENAME` beside the database and logged to the child's stderr (the solve log).
With the probe on, the marker also carries the digest and the file hashes, and a digest that cannot
be taken means no marker. With it off, no digest is computed anywhere.

Nothing here raises: a marker or a probe that fails is a log line, and the solve goes on.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sqlite3
import time
from contextlib import closing
from pathlib import Path

logger = logging.getLogger(__name__)

# The builder -> its own background child. Internal: never set by hand.
TOKEN_ENV = "TOWER_WORLD_BG_MATCH_TOKEN"
MARKER_FILENAME = "bg_matching_committed.json"
MARKER_RECORD = "wb-bg-matching-committed/2"
PROBE_FILENAME = "bg_stop_at_match.probe.jsonl"
PROBE_RECORD = "wb-bg-stop-at-match-probe/1"
DATABASE_FILENAME = "database.db"
# The committed database of a background child ended at its marker, kept until the final solve
# has decided whether a deferred map of it is owed. And the order the builder hands that map.
SNAPSHOT_FILENAME = "database.bg-horizon.db"
DEFERRED_ORDER_FILENAME = "bg_deferred_map.json"
DEFERRED_ORDER_RECORD = "wb-bg-deferred-map/1"
# The database and the companions SQLite may leave beside it. `-shm` is a shared-memory index
# (rebuilt by the next opener), so only its size is recorded.
_COMPANIONS = ("", "-wal", "-journal", "-shm")
_HASHED = ("", "-wal", "-journal")
_CHUNK = 1 << 20


def marker_path(workspace_root) -> Path:
    return Path(workspace_root) / MARKER_FILENAME


def snapshot_path(workspace_root) -> Path:
    return Path(workspace_root) / SNAPSHOT_FILENAME


def clear(workspace_root) -> None:
    """Remove the marker before a new launch. Never raises: a marker that cannot be removed
    carries another launch's token and is ignored (`read`)."""
    try:
        marker_path(workspace_root).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("matching marker: could not remove %s (%s); its token is not the next "
                       "launch's, so it is ignored", marker_path(workspace_root), exc)


def _valid_digest(digest) -> bool:
    return isinstance(digest, dict) and isinstance(digest.get("content"), str) and bool(digest["content"])


def read(workspace_root, token: str | None) -> dict | None:
    """The marker THIS launch wrote (its token), whole, or None: absent, unreadable, torn, another
    record or version, another launch's token, or missing a field a marker must carry (the writer
    check, the keyframe ids; with a digest, a valid one). Never raises."""
    if not token:
        return None
    try:
        path = marker_path(workspace_root)
        if not path.exists():
            return None
        from tower.storage import read_json_closed  # noqa: PLC0415

        record = read_json_closed(path)
        if not isinstance(record, dict) or record.get("record") != MARKER_RECORD:
            return None
        if record.get("token") != token:
            return None
        if record.get("writer_free") is not True:
            return None
        if not isinstance(record.get("pid"), int) or not isinstance(record.get("committed_at"),
                                                                       (int, float)):
            return None
        ids = record.get("keyframe_ids")
        if not isinstance(ids, list) or len(ids) < 2 or not all(isinstance(i, str) for i in ids):
            return None
        if "digest" in record and not _valid_digest(record["digest"]):
            return None
        return record
    except Exception:  # noqa: BLE001 -- unreadable is "no marker": today's wait
        return None


def writer_free(database) -> None:
    """Raise unless no connection holds `database` for writing: a zero-timeout `BEGIN IMMEDIATE`
    that is rolled back at once (it writes nothing). On Windows the lock is per handle, so a
    connection the matcher left open in this very process is caught too."""
    with closing(sqlite3.connect(str(database), timeout=0)) as connection:
        connection.execute("begin immediate")
        connection.rollback()


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


def snapshot_horizon(workspace_root) -> Path | None:
    """Copy the walk database AS IT IS NOW (the file and a non-empty WAL beside it) to
    `SNAPSHOT_FILENAME`, through a staging name, and return its path; None when it cannot be made.
    Called by the builder only when no process of the ended child is left and before the final
    solve starts, so nothing writes the database while it is copied. Never raises."""
    root = Path(workspace_root)
    database = root / DATABASE_FILENAME
    target = snapshot_path(root)
    try:
        discard_snapshot(root)
        if not database.is_file():
            return None
        from tower.storage import replace_with_retry, staging_path  # noqa: PLC0415

        wal = Path(str(database) + "-wal")
        if wal.is_file() and wal.stat().st_size > 0:
            # The WAL first, under its final name: SQLite pairs a database with `<name>-wal`, and
            # the snapshot's database file only appears (atomically) once its WAL is in place.
            shutil.copyfile(wal, Path(str(target) + "-wal"))
        staging = staging_path(target)
        try:
            shutil.copyfile(database, staging)
            replace_with_retry(staging, target)
        finally:
            staging.unlink(missing_ok=True)
        return target
    except Exception:  # noqa: BLE001 -- no snapshot: no deferred map, logged
        logger.exception("matching marker: could not snapshot the committed database")
        discard_snapshot(root)
        return None


def discard_snapshot(workspace_root) -> None:
    """Remove the horizon snapshot, its SQLite companions and the deferred-map order. Never raises."""
    target = snapshot_path(workspace_root)
    for path in (target, *(Path(str(target) + s) for s in ("-wal", "-shm", "-journal")),
                 Path(workspace_root) / DEFERRED_ORDER_FILENAME):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("matching marker: could not remove %s (%s)", path, exc)


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

    def matched(self, database_path=None, context: dict | None = None) -> None:
        """The matching call returned and every write this solve makes is behind it. Prove the
        database is free of writers, then write the marker (a background child the builder gave a
        token) and the probe. Any failure first means no marker (fail closed). Never raises."""
        try:
            digest = files = None
            seconds = 0.0
            started = time.perf_counter()
            writer_free(self.database)
            if self.probe_on:
                digest = _digest(self.database)
                files = file_identity(self.database)
                seconds = round(time.perf_counter() - started, 3)
                if not _valid_digest(digest):
                    logger.warning("matching marker: the database digest could not be taken (%s); "
                                   "no marker", digest)
                    self._probe("matched", digest=digest, files=files, seconds=seconds)
                    return
            if self.token:
                from tower.storage import write_json_atomic  # noqa: PLC0415

                context = dict(context or {})
                record = {"record": MARKER_RECORD, "token": self.token, "pid": os.getpid(),
                          "committed_at": time.time(), "database": self.database.name,
                          "mapped": Path(database_path).name if database_path else None,
                          "writer_free": True,
                          "keyframe_ids": list(context.get("keyframe_ids") or []),
                          "solve": {k: v for k, v in context.items() if k != "keyframe_ids"}}
                if digest is not None:
                    record["digest"] = digest
                    record["files"] = files
                    record["probe_s"] = seconds
                write_json_atomic(marker_path(self.root), record)
            if self.probe_on:
                self._probe("matched", digest=digest, files=files, seconds=seconds)
        except Exception:  # noqa: BLE001 -- no marker is today's wait; the solve goes on
            logger.exception("matching marker: could not record the matching commit; no marker")

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
