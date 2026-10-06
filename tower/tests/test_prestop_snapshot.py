import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.prestop_snapshot import snapshot_at_stop, verify_snapshot


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_snapshot_copies_session_and_uses_sqlite_backup(tmp_path):
    root = tmp_path / "tower" / "world_builder"
    world, session = "world-1", "session-1"
    session_dir = root / "worlds" / world / "sessions" / session
    solve_dir = root / "worlds" / world / "solve" / session
    session_dir.mkdir(parents=True)
    solve_dir.mkdir(parents=True)
    (session_dir / "images").mkdir()
    (session_dir / "images" / "00000001.jpg").write_bytes(b"jpeg")
    (root / "intrinsics").mkdir()
    (root / "intrinsics" / "360x640.json").write_text("{}")
    journal = session_dir / "keyframes.jsonl"
    journal.write_text(json.dumps({"keyframe_id": "session-1:00000001"}) + "\n")
    db = solve_dir / "database.db"
    writer = sqlite3.connect(db)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE matches (value TEXT)")
    writer.commit()
    writer.execute("INSERT INTO matches VALUES ('committed')")
    writer.commit()
    writer.execute("INSERT INTO matches VALUES ('uncommitted')")
    destination = tmp_path / "snapshot"
    pin = snapshot_at_stop(root, world, session, destination)
    assert pin == json.loads((destination / "PRESTOP-PIN.json").read_text())
    copied = destination / "world_builder"
    assert (copied / "worlds" / world / "sessions" / session / "images" / "00000001.jpg").read_bytes() == b"jpeg"
    assert (copied / "intrinsics" / "360x640.json").exists()
    assert pin["keyframes_sha256"] == _sha(journal)
    assert pin["input_digest"] == hashlib.sha256(b"session-1:00000001\0").hexdigest()
    assert pin["database_sha256"] == _sha(copied / "worlds" / world / "solve" / session / "database.db")
    assert verify_snapshot(destination) == pin
    with sqlite3.connect(copied / "worlds" / world / "solve" / session / "database.db") as check:
        assert check.execute("SELECT value FROM matches").fetchall() == [("committed",)]
        assert check.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    writer.rollback()
    writer.close()


def test_snapshot_refuses_existing_destination_and_missing_db(tmp_path):
    root = tmp_path / "world_builder"
    journal = root / "worlds" / "w" / "sessions" / "s" / "keyframes.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text('{"keyframe_id":"s:1"}\n')
    destination = tmp_path / "snap"
    with pytest.raises(FileNotFoundError):
        snapshot_at_stop(root, "w", "s", destination)
    assert not destination.exists()
    db = root / "worlds" / "w" / "solve" / "s" / "database.db"
    db.parent.mkdir(parents=True)
    sqlite3.connect(db).close()
    destination.mkdir()
    with pytest.raises(FileExistsError):
        snapshot_at_stop(root, "w", "s", destination)


def test_pin_verification_rejects_tampered_database(tmp_path):
    root = tmp_path / "world_builder"
    journal = root / "worlds" / "w" / "sessions" / "s" / "keyframes.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text('{"keyframe_id":"s:1"}\n')
    db = root / "worlds" / "w" / "solve" / "s" / "database.db"
    db.parent.mkdir(parents=True)
    sqlite3.connect(db).close()
    snapshot = tmp_path / "snap"
    snapshot_at_stop(root, "w", "s", snapshot)
    with (snapshot / "world_builder" / "worlds" / "w" / "solve" / "s" / "database.db").open("ab") as target:
        target.write(b"tampered")
    with pytest.raises(ValueError, match="database SHA-256"):
        verify_snapshot(snapshot)
