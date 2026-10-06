"""Freeze a stopped live builder immediately before its final solve.

The caller has observed capture Stop and waited for its background solver.
SQLite's backup API takes a transactionally consistent view of database.db,
including committed WAL pages, even if another connection still holds it.
The backup excludes uncommitted pages. PRESTOP-PIN.json is written last and
is the completion marker; an interrupted copy is never treated as a pin.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_digest(journal: Path) -> str:
    digest = hashlib.sha256()
    count = 0
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        keyframe_id = json.loads(line)["keyframe_id"]
        digest.update(keyframe_id.encode("utf-8"))
        digest.update(b"\0")
        count += 1
    if not count:
        raise ValueError(f"empty keyframe journal: {journal}")
    return digest.hexdigest()


def snapshot_at_stop(root: Path, world: str, session: str, destination: Path) -> dict:
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError("snapshot and source world root must be separate")
    if destination.exists():
        raise FileExistsError(f"snapshot destination already exists: {destination}")
    relative_journal = Path("worlds") / world / "sessions" / session / "keyframes.jsonl"
    relative_db = Path("worlds") / world / "solve" / session / "database.db"
    journal, db = root / relative_journal, root / relative_db
    if not journal.is_file():
        raise FileNotFoundError(journal)
    if not db.is_file():
        raise FileNotFoundError(db)
    source_journal_sha = sha256(journal)
    # sqlite backup, not a raw file copy: the database may be in WAL mode.
    # Exclude sidecars as well so a copied WAL cannot override the backup.
    def ignore(_directory, names):
        return {name for name in names if name in {"database.db", "database.db-wal", "database.db-shm"}}

    copied = destination / "world_builder"
    shutil.copytree(root, copied, ignore=ignore)
    target_db = copied / relative_db
    target_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(target_db) as target:
            source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise RuntimeError("snapshot database integrity check failed")
    target_journal = copied / relative_journal
    if sha256(journal) != source_journal_sha or sha256(target_journal) != source_journal_sha:
        raise RuntimeError("keyframe journal changed during snapshot")
    pin = {
        "world": world,
        "session": session,
        "keyframes_sha256": source_journal_sha,
        "input_digest": _input_digest(target_journal),
        "database_sha256": sha256(target_db),
        "database_copy_method": "sqlite3.Connection.backup from read-only source",
    }
    (destination / "PRESTOP-PIN.json").write_text(
        json.dumps(pin, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return pin


def verify_snapshot(destination: Path) -> dict:
    """Verify the completion pin against copied bytes, before an A/B arm uses it."""
    destination = Path(destination)
    pin = json.loads((destination / "PRESTOP-PIN.json").read_text(encoding="utf-8"))
    world, session = pin["world"], pin["session"]
    if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value)
               for value in (world, session)):
        raise ValueError("invalid snapshot world/session path")
    root = destination / "world_builder" / "worlds" / world
    journal = root / "sessions" / session / "keyframes.jsonl"
    db = root / "solve" / session / "database.db"
    if sha256(journal) != pin["keyframes_sha256"]:
        raise ValueError("keyframe SHA-256 mismatch")
    if _input_digest(journal) != pin["input_digest"]:
        raise ValueError("Tower input digest mismatch")
    if sha256(db) != pin["database_sha256"]:
        raise ValueError("database SHA-256 mismatch")
    if pin.get("database_copy_method") != "sqlite3.Connection.backup from read-only source":
        raise ValueError("unrecognized database copy method")
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as source:
        if source.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("snapshot database integrity check failed")
    return pin
