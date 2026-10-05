"""U-PARTIAL, switch OFF: every status payload and every `GET /worlds` row is byte for byte as before.

Spec `U-PARTIAL-SPEC-20261004.md` §5.1 A. `TOWER_WORLD_PARTIAL_STATE` and `TOWER_WORLD_PARTIAL_FRAMES`
are both default off (WORLD-BUILDER-COMPONENTS.md §3.1a, §7 rule 7). This file proves "off" over a
matrix of record shapes, and it is written to be COPIED ONTO THE BASE TREE AND RUN UNCHANGED: it imports
only the status producer, the listing and the store API, and passes `capture_root` only where the
producer accepts it. Run it on both trees with `-s` and compare the printed `OFF-PIN sha256` lines; they
must be identical (`U_PARTIAL_PIN_DUMP=<file>` also writes the per-cell canonical JSON for a diff).

The matrix (one world root per cell, holding one world with one session; every timestamp fixed; clock
`lambda: 1000.0`). One root per cell because `resolve` lists every world of its root on each snapshot, so a
shared root makes the matrix quadratic (measured: 222 s for one pass over one shared root, against seconds):

| Dimension | Values |
|---|---|
| `end_reason` | `stop`, `interrupted`, `error`, open |
| finalization | none, pending/pending, complete/solved, complete/skipped, complete/failed, interrupted/None |
| notice | absent, or the closed-set `depth-unavailable` sentence |
| tree | figures, empty, absent |
| photographic | `never_recorded`, `complete`, `failed`, `owed` |
| capture.json | absent, equal counts, fewer observed |
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import os

import pytest

from tower.results.world_builder import WorldBuilderStatusProducer
from tower.results.world_builder_library import build_world_listing
from tower.world_builder.coherence_publish import NOTICE_SENTENCES
from tower.world_builder.events import WorldEvent
from tower.world_builder.records import Keyframe, Session, World
from tower.world_builder.store import WorldStore, compute_input_digest

CLOCK = 1000.0

# A closed-set notice the gate really writes (present, unchanged, on the base tree too).
DEPTH_UNAVAILABLE_NOTICE = NOTICE_SENTENCES["depth-unavailable"]

END_REASONS = ("stop", "interrupted", "error", None)
FINALIZATIONS = {
    "none": None,
    "pending": {"state": "pending", "final_solve": "pending"},
    "complete-solved": {"state": "complete", "final_solve": "solved"},
    "complete-skipped": {"state": "complete", "final_solve": "skipped"},
    "complete-failed": {"state": "complete", "final_solve": "failed"},
    "interrupted": {"state": "interrupted", "final_solve": None},
}
NOTICES = ("absent", "depth-unavailable")
TREES = ("figures", "empty", "absent")
PHOTOGRAPHIC = {
    "never_recorded": None,
    "complete": {"surface": {"state": "ok"}, "appearance": {"state": "ok"}},
    "failed": {"surface": {"state": "ok"}, "appearance": {"state": "failed", "detail": "boom"}},
    "owed": {"surface": {"state": "ok"}, "appearance": {"state": "running"}},
}
CAPTURES = ("absent", "equal", "fewer-observed")

FRAMES_OBSERVED = 805
FRAMES_WRITTEN_FEWER = 3197

# Every spelling the switch must read as OFF (spec §5.1 A), each with the F switch unset and on.
OFF_VALUES = (None, "", "0", "off", "false", "garbage")
FRAMES_VALUES = (None, "1")


def cells():
    """Every matrix cell, in a fixed order, keyed by a readable name."""
    for end, fin, notice, tree, photo, capture in itertools.product(
        END_REASONS, FINALIZATIONS, NOTICES, TREES, PHOTOGRAPHIC, CAPTURES
    ):
        # A notice is something a finalization record carries; without one there is nowhere to put it.
        if FINALIZATIONS[fin] is None and notice != "absent":
            continue
        yield {
            "end_reason": end, "finalization": fin, "notice": notice,
            "tree": tree, "photographic": photo, "capture": capture,
        }


def _keyframe(session_id: str, seq: int, segment_index: int) -> Keyframe:
    return Keyframe(
        keyframe_id=f"{session_id}:{seq:08d}", session_id=session_id, source_seq=seq,
        received_at=1000.0 + seq, image_relpath=f"images/{seq:08d}.jpg", width=360, height=640,
        byte_count=1234, segment_index=segment_index,
    )


def _write_tree(store, world_id: str, session_id: str, tree: str) -> None:
    if tree == "absent":
        return
    figures = tree == "figures"
    poses = [
        {"keyframe_id": f"{session_id}:00000001", "segment_index": 0, "status": "anchor",
         "degeneracy": "", "rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
        {"keyframe_id": f"{session_id}:00000002", "segment_index": 0, "status": "solved",
         "degeneracy": "", "rotation": [0.0, 1.0, 0.0, 0.0], "translation": [1.0, 2.0, 3.0]},
    ] if figures else []
    points = [
        {"segment_index": 0, "xyz": [1.0, 2.0, 3.0]},
        {"segment_index": 0, "xyz": [-1.0, 0.0, 5.0]},
    ] if figures else []
    digest = compute_input_digest(store.read_keyframes(world_id, session_id))
    manifest = {
        "schema_version": 1, "input_digest": digest, "built_at": 3.0,
        "backend_id": "classical-sfm", "session_id": session_id,
        "keyframes": 2, "poses_solved": 1 if figures else 0, "poses_refused": 0,
        "poses_anchor": 1 if figures else 0, "poses_positioned": 2 if figures else 0,
        "points": 2 if figures else 0, "segments": 1, "scale_state": "unknown", "global_solve": None,
    }
    store.write_derived(world_id, session_id, poses=poses, points=points, manifest=manifest)


def _write_capture(capture_root, capture_id: str, capture: str) -> None:
    if capture == "absent":
        return
    written = FRAMES_OBSERVED if capture == "equal" else FRAMES_WRITTEN_FEWER
    folder = capture_root / "captures" / capture_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "capture.json").write_text(json.dumps({
        "schema_version": 1, "capture_id": capture_id, "started_at": 1.0, "ended_at": 5.0,
        "end_reason": "stop", "frames_written": written, "bytes_written": 1000,
        "continues_capture": None,
    }), encoding="utf-8")


def build_cell(base, index: int, cell: dict, capture_root, *, notice: str | None = None):
    """One world root holding one world with one session, shaped as `cell` says. `notice` replaces the
    closed-set notice text when given. Returns `(store, world_id, session_id)`."""
    store = WorldStore(base / f"r{index:04d}")
    world_id = f"w{index:04d}"
    session_id = f"s{index:04d}"
    capture_id = f"c{index:04d}"
    store.write_world(World(world_id=world_id, created_at=1.0 + index / 1000.0,
                            updated_at=2.0 + index / 1000.0, session_ids=(session_id,)))
    for seq in (1, 2):
        store.append_keyframe(world_id, _keyframe(session_id, seq, 0))
    finalization = FINALIZATIONS[cell["finalization"]]
    if finalization is not None:
        finalization = {**finalization, "started_at": 2.0, "updated_at": 3.0, "detail": None}
        if cell["notice"] == "depth-unavailable":
            finalization["notice"] = notice if notice is not None else DEPTH_UNAVAILABLE_NOTICE
    ended = cell["end_reason"] is not None
    store.write_session(Session(
        session_id=session_id, world_id=world_id, started_at=1.0,
        ended_at=2.0 if ended else None, end_reason=cell["end_reason"],
        frame_source="phone", capture_id=capture_id,
        frames_observed=FRAMES_OBSERVED, keyframes_accepted=2,
        finalization=finalization, stages=PHOTOGRAPHIC[cell["photographic"]],
    ))
    if ended:
        store.append_event(world_id, session_id,
                           WorldEvent(event_id=1, kind="session_stopped", at=2.0, payload={}))
    _write_tree(store, world_id, session_id, cell["tree"])
    _write_capture(capture_root, capture_id, cell["capture"])
    return store, world_id, session_id


def build_matrix(base, capture_root):
    """Every cell of `cells()`, each in its own world root. Returns `[(store, world_id, session_id, cell)]`."""
    return [
        (*build_cell(base, index, cell, capture_root), cell)
        for index, cell in enumerate(cells())
    ]


def _accepts_capture_root(fn) -> bool:
    return "capture_root" in inspect.signature(fn).parameters


def surface(store, capture_root, world_id, session_id) -> tuple[dict, dict]:
    """`(status payload, GET /worlds listing)` for one cell, as the head or the base tree builds them."""
    kwargs = {"capture_root": capture_root} if _accepts_capture_root(WorldBuilderStatusProducer) else {}
    producer = WorldBuilderStatusProducer(store.root, lambda: CLOCK, **kwargs)
    snapshot = producer.snapshot(world_id, session_id)
    payload = getattr(snapshot, "payload", snapshot)
    listing_kwargs = (
        {"capture_root": capture_root} if _accepts_capture_root(build_world_listing) else {}
    )
    return payload, build_world_listing(store, **listing_kwargs)


def surfaces(capture_root, built) -> dict[str, str]:
    """Canonical JSON (`sort_keys=True`) of every cell's status payload and listing, by session id."""
    out = {}
    for store, world_id, session_id, _cell in built:
        payload, listing = surface(store, capture_root, world_id, session_id)
        out[session_id] = json.dumps({"payload": payload, "listing": listing}, sort_keys=True)
    return out


def _digest(dumped: dict[str, str]) -> str:
    h = hashlib.sha256()
    for key in sorted(dumped):
        h.update(key.encode())
        h.update(b"\0")
        h.update(dumped[key].encode())
        h.update(b"\0")
    return h.hexdigest()


def _set_env(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    root = tmp_path_factory.mktemp("upartial-off-pin")
    capture_root = root / "capture"
    return capture_root, build_matrix(root / "worlds", capture_root)


def test_the_matrix_covers_every_dimension(matrix):
    _capture_root, built = matrix
    seen = {key: {cell[key] for *_rest, cell in built} for key in built[0][3]}
    assert seen["end_reason"] == set(END_REASONS)
    assert seen["finalization"] == set(FINALIZATIONS)
    assert seen["notice"] == set(NOTICES)
    assert seen["tree"] == set(TREES)
    assert seen["photographic"] == set(PHOTOGRAPHIC)
    assert seen["capture"] == set(CAPTURES)


def test_every_payload_and_row_is_identical_for_every_off_spelling(matrix, monkeypatch):
    """Unset, blank, `0`, `off`, `false` and garbage, each with `TOWER_WORLD_PARTIAL_FRAMES` unset and
    on: one canonical dump, and its digest is printed for the A/B against the base tree."""
    capture_root, built = matrix
    reference = None
    for state, frames in itertools.product(OFF_VALUES, FRAMES_VALUES):
        _set_env(monkeypatch, "TOWER_WORLD_PARTIAL_STATE", state)
        _set_env(monkeypatch, "TOWER_WORLD_PARTIAL_FRAMES", frames)
        dumped = surfaces(capture_root, built)
        if reference is None:
            reference = dumped
            continue
        changed = sorted(k for k in dumped if dumped[k] != reference[k])
        assert not changed, (state, frames, changed[:5])
    digest = _digest(reference)
    print(f"\nOFF-PIN sha256 {digest} cells={len(built)}")
    out = os.environ.get("U_PARTIAL_PIN_DUMP")
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(reference, handle, sort_keys=True, indent=0)
    # No cell may carry the key on any surface while the switch is off.
    assert all('"walk"' not in text for text in reference.values())
