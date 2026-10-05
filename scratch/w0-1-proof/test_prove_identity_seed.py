"""Frozen verification seed is independent of the published consensus winner."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).with_name("prove_identity.py")
SPEC = importlib.util.spec_from_file_location("prove_identity", SCRIPT)
PROOF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROOF)


def _frozen_world(tmp_path, monkeypatch, *, chosen_seed, verification_seed):
    world_id = "world"
    session_id = "session"
    workspace = tmp_path / "worlds" / world_id / "solve" / session_id
    workspace.mkdir(parents=True)
    database = workspace / "database.db"
    database.write_bytes(b"frozen")
    solution = SimpleNamespace(solve={
        "matching": "frozen",
        "seed": chosen_seed,
        "verification_seed": verification_seed,
        "database": database.name,
    })
    monkeypatch.setattr(PROOF.GS, "load_solution", lambda *args: solution)
    monkeypatch.setattr(PROOF.GS, "database_digest", lambda path: {"content": "known-digest"})
    return world_id, session_id, database, solution


@pytest.mark.parametrize("chosen_seed", [0, 1, 2])
def test_frozen_world_accepts_any_published_winner_with_verification_seed_zero(
        tmp_path, monkeypatch, chosen_seed):
    world_id, session_id, database, solution = _frozen_world(
        tmp_path, monkeypatch, chosen_seed=chosen_seed, verification_seed=0)

    _, _, loaded, named_database, digest = PROOF._load(tmp_path, world_id, session_id, 0)

    assert loaded is solution
    assert named_database == database
    assert digest == "known-digest"


def test_frozen_world_rejects_different_verification_seed_even_when_winner_matches(
        tmp_path, monkeypatch):
    world_id, session_id, _, _ = _frozen_world(
        tmp_path, monkeypatch, chosen_seed=0, verification_seed=1)

    with pytest.raises(RuntimeError, match="verification seed.*expected 0"):
        PROOF._load(tmp_path, world_id, session_id, 0)
