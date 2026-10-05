"""U-PARTIAL, switch ON: a world the Tower finished from only part of its walk never looks complete.

Spec `U-PARTIAL-SPEC-20261004.md` §5.1 C and D; WORLD-BUILDER-COMPONENTS.md §3.1a (v12, PROPOSED);
WORLDS §2 and §3 rule 7. Module under test: `tower/results/world_builder_partial.py`, asked by both the
status producer (`results/world_builder.py`, `/ws`) and the saved-worlds listing
(`results/world_builder_library.py`, `GET /worlds`).

- **Rule R** (`TOWER_WORLD_PARTIAL_STATE`): a closed record, finalization `complete` and `solved`, and
  `end_reason` `interrupted` or `error`.
- **Rule F** (`TOWER_WORLD_PARTIAL_FRAMES`, only with R's switch on): a `stop` record whose own capture's
  closed `capture.json` recorded more frames than the session looked at.

What changes is exactly: READY/`complete` becomes `interrupted` with the walk's sentence as the reason,
and `finalization` gains `walk` and the sentence first in `notice`. Nothing is written to disk.

The phone half (§3.3, "old-client decode compatibility") is pinned at the bottom with a Python port of
the decoder rules of `origin/ios/ux-v1` @ `0f1249e`, which this lane cannot compile.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import time

import pytest

from tests import test_world_builder_partial_state_off_pin as pin
from tower.results import world_builder as wb
from tower.results import world_builder_library as wbl
from tower.results import world_builder_partial as wbp
from tower.results.world_builder import WorldBuilderStatusProducer
from tower.results.world_builder_library import build_world_listing
from tower.results.world_builder_partial import (
    WALK_NOTICE_SENTENCES,
    partial_walk,
    present_finalization,
    recorded_frames,
)
from tower.world_builder import coherence_publish as CP
from tower.world_builder.records import Session

EARLY = WALK_NOTICE_SENTENCES["walk-ended-early"]
INCIDENT = (
    "only part of this walk is in this world: the Tower looked at 805 of the 3197 frames it recorded "
    "for the walk; an owner can re-capture this walk"
)
GATE_NOTICE = CP.NOTICE_SENTENCES["depth-unavailable"]


# -- helpers ------------------------------------------------------------------------------------------


def _cell(end_reason="interrupted", finalization="complete-solved", notice="absent", tree="figures",
          photographic="never_recorded", capture="absent"):
    return {"end_reason": end_reason, "finalization": finalization, "notice": notice, "tree": tree,
            "photographic": photographic, "capture": capture}


class World:
    """One cell of the OFF pin's matrix, on disk, with both surfaces one call away."""

    def __init__(self, tmp_path, *, index=0, notice_text=None, **cell):
        self.capture_root = tmp_path / "capture"
        self.cell = _cell(**cell)
        self.store, self.world_id, self.session_id = pin.build_cell(
            tmp_path / "worlds", index, self.cell, self.capture_root, notice=notice_text)
        self.capture_id = f"c{index:04d}"

    @property
    def manifest_path(self):
        return self.capture_root / "captures" / self.capture_id / "capture.json"

    def payload(self, *, capture_root="default", clock=lambda: pin.CLOCK):
        root = self.capture_root if capture_root == "default" else capture_root
        producer = WorldBuilderStatusProducer(self.store.root, clock, capture_root=root)
        snapshot = producer.snapshot(self.world_id, self.session_id)
        return getattr(snapshot, "payload", snapshot)

    def row(self, *, capture_root="default"):
        root = self.capture_root if capture_root == "default" else capture_root
        listing = build_world_listing(self.store, capture_root=root)
        return listing["worlds"][0]["sessions"][0]

    def rewrite(self, **fields):
        import dataclasses

        session = self.store.read_session(self.world_id, self.session_id)
        self.store.write_session(dataclasses.replace(session, **fields))


@pytest.fixture
def switch(monkeypatch):
    """`switch(state, frames)` sets both environment variables; None unsets one."""
    def set_(state="1", frames=None):
        for name, value in (("TOWER_WORLD_PARTIAL_STATE", state), ("TOWER_WORLD_PARTIAL_FRAMES", frames)):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
    set_(None, None)
    return set_


def _canon(value) -> str:
    return json.dumps(value, sort_keys=True)


def _diff_paths(a, b, path=""):
    """Every leaf path where two JSON values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            out += _diff_paths(a.get(key, "<absent>"), b.get(key, "<absent>"), f"{path}.{key}")
        return out
    return [] if a == b else [path]


def _dead_pid() -> int:
    import psutil

    return next(pid for pid in range(100_000, 200_000) if not psutil.pid_exists(pid))


# -- C1-C3: rule R, the plain shapes ------------------------------------------------------------------


@pytest.mark.parametrize("end_reason, reason", [("interrupted", "session-interrupted"),
                                                ("error", "session-error")])
def test_a_finished_world_from_an_interrupted_walk_is_interrupted_on_both_surfaces(
        tmp_path, switch, end_reason, reason):
    world = World(tmp_path, end_reason=end_reason)
    off_payload, off_row = world.payload(), world.row()
    assert off_payload["lifecycle"]["state"] == "ready", "the fixture must start from the false complete"
    assert off_payload["model_state"] == "finalized"
    assert off_row["state"] == "complete"

    switch("1")
    payload, row = world.payload(), world.row()
    lifecycle = payload["lifecycle"]
    assert lifecycle["state"] == "interrupted"
    assert payload["model_state"] == "interrupted"
    assert lifecycle["reason"] == payload["model_state_reason"] == EARLY
    assert lifecycle["finalization"]["walk"] == {"saved": "part", "reason": reason}
    assert lifecycle["finalization"]["notice"] == EARLY
    assert lifecycle["build_in_progress"] is False
    assert lifecycle["evidence"].endswith(
        f"; this world was finished from part of its walk ({reason})")
    assert row["state"] == "interrupted"
    assert row["finalization"] == lifecycle["finalization"]
    # Nothing else on either surface moved.
    # `.world_snapshot.revision` is the content hash of the payload (`compute_revision`), so it moves
    # whenever anything in it does.
    allowed = {".lifecycle.state", ".lifecycle.evidence", ".lifecycle.reason", ".model_state",
               ".model_state_reason", ".lifecycle.finalization.walk", ".lifecycle.finalization.notice",
               ".world_snapshot.revision"}
    assert set(_diff_paths(off_payload, payload)) <= allowed, _diff_paths(off_payload, payload)
    assert set(_diff_paths(off_row, row)) == {".state", ".finalization.walk", ".finalization.notice"}


def test_the_walk_sentence_comes_before_the_gate_notice_and_the_record_keeps_its_own(tmp_path, switch):
    world = World(tmp_path, notice="depth-unavailable")
    session_json = world.store.session_path(world.world_id, world.session_id)
    before = session_json.read_bytes()

    switch("1")
    payload, row = world.payload(), world.row()
    composed = f"{EARLY}; {GATE_NOTICE}"
    assert payload["lifecycle"]["finalization"]["notice"] == composed
    assert row["finalization"]["notice"] == composed
    assert row["finalization"] == payload["lifecycle"]["finalization"]
    # The record on disk still holds the gate notice alone, and not a byte of it moved.
    assert session_json.read_bytes() == before
    assert world.store.read_session(world.world_id, world.session_id).finalization["notice"] == GATE_NOTICE


# -- C4-C6: the photographic room and the tree ----------------------------------------------------------


def test_a_partial_world_still_building_its_photographic_room_is_improving_then_interrupted(
        tmp_path, switch):
    """`finalizing` still wins over the walk on both surfaces while the room is owed, running or
    unobservable; `walk` rides along; once the stage settles the world reads `interrupted`."""
    switch("1")
    world = World(tmp_path, photographic="owed")
    payload, row = world.payload(), world.row()
    assert payload["lifecycle"]["state"] == "finalizing"
    assert payload["model_state"] == "finalizing"
    assert payload["lifecycle"]["finalization"]["walk"]["saved"] == "part"
    assert row["state"] == "finalizing"
    assert row["finalization"] == payload["lifecycle"]["finalization"]

    # Running: a live stage writer. The reason carries the settled reason -- the walk sentence.
    world.rewrite(stages={"surface": {"state": "ok"}, "appearance": {"state": "ok"}})
    status = world.store.world_dir(world.world_id) / "surface" / world.session_id
    status.mkdir(parents=True, exist_ok=True)
    (status / "status.json").write_text(json.dumps(
        {"state": "running", "pid": os.getpid(), "updated_at": time.time()}), encoding="utf-8")
    payload = world.payload(clock=time.time)
    lifecycle = payload["lifecycle"]
    assert lifecycle["state"] == "finalizing"
    assert lifecycle["build_in_progress"] is True
    assert lifecycle["reason"].endswith(f"What is already recorded about this session: {EARLY}")
    assert lifecycle["finalization"]["walk"]["reason"] == "session-interrupted"
    assert world.row()["state"] == "finalizing"

    # Settled: the stage finished.
    (status / "status.json").unlink()
    payload, row = world.payload(), world.row()
    assert payload["lifecycle"]["state"] == "interrupted"
    assert payload["lifecycle"]["reason"] == EARLY
    assert row["state"] == "interrupted"


def test_a_partial_world_whose_photographic_probe_breaks_is_improving_with_the_walk(
        tmp_path, switch, monkeypatch):
    import tower.results.world_builder_render as render

    def boom(*args, **kwargs):
        raise OSError("the liveness probe is broken")

    monkeypatch.setattr(render, "_stage_running", boom)
    switch("1")
    world = World(tmp_path, photographic="owed")
    payload, row = world.payload(), world.row()
    assert payload["lifecycle"]["photographic"]["state"] == "unobservable"
    assert payload["lifecycle"]["state"] == "finalizing"
    assert row["state"] == "finalizing"
    assert payload["lifecycle"]["finalization"]["walk"] == row["finalization"]["walk"]


def test_a_partial_world_whose_photographic_build_failed_is_interrupted(tmp_path, switch):
    switch("1")
    world = World(tmp_path, photographic="failed")
    payload, row = world.payload(), world.row()
    assert payload["lifecycle"]["state"] == "interrupted"
    assert payload["lifecycle"]["reason"] == EARLY
    assert payload["lifecycle"]["photographic"]["state"] == "failed"
    assert row["state"] == "interrupted"
    assert row["finalization"]["walk"]["saved"] == "part"


@pytest.mark.parametrize("tree", ["absent", "empty"])
def test_a_partial_world_with_nothing_to_open_keeps_todays_reason_and_carries_the_walk(
        tmp_path, switch, tree):
    world = World(tmp_path, tree=tree)
    off = world.payload()
    switch("1")
    payload, row = world.payload(), world.row()
    assert row["has_geometry"] is False
    assert row["state"] == "interrupted"
    assert payload["model_state"] == "interrupted"
    assert payload["lifecycle"]["state"] == "interrupted"
    assert payload["lifecycle"]["finalization"]["walk"] == {"saved": "part", "reason": "session-interrupted"}
    assert payload["lifecycle"]["finalization"]["notice"] == EARLY
    assert row["finalization"] == payload["lifecycle"]["finalization"]
    if off["lifecycle"]["state"] == "interrupted":
        # Today's arm, today's reason (`the mapping session ended with 'interrupted'`).
        assert payload["lifecycle"]["reason"] == off["lifecycle"]["reason"]
        assert payload["lifecycle"]["reason"].startswith("the mapping session ended with 'interrupted'")
    else:
        # An empty tree: the panel's override asks whether a tree EXISTS, the row whether it can be
        # drawn (spec §2.4, "the no-geometry edge"). Both end `interrupted` with the same finalization.
        assert off["lifecycle"]["state"] == "ready"
        assert payload["lifecycle"]["reason"] == EARLY


# -- C7-C8: everything that is not partial is untouched ------------------------------------------------


@pytest.mark.parametrize("cell", [
    dict(end_reason="interrupted", finalization="complete-skipped"),
    dict(end_reason="interrupted", finalization="pending"),
    dict(end_reason="error", finalization="complete-failed"),
    dict(end_reason="stop", finalization="complete-solved"),
    dict(end_reason="stop", finalization="complete-solved", capture="fewer-observed"),
    dict(end_reason=None, finalization="none"),
])
def test_a_session_rule_r_does_not_name_is_byte_identical_with_the_switch_on(tmp_path, switch, cell):
    world = World(tmp_path, **cell)
    off = _canon([world.payload(), world.row()])
    switch("1")
    assert _canon([world.payload(), world.row()]) == off


def test_a_dead_lock_is_byte_identical_with_the_switch_on(tmp_path, switch):
    world = World(tmp_path, end_reason=None, finalization="none")
    lock = world.store.lock_path(world.world_id)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": _dead_pid()}))
    off = _canon([world.payload(), world.row()])
    assert world.payload()["lifecycle"]["state"] == "interrupted"
    switch("1", "1")
    assert _canon([world.payload(), world.row()]) == off


def _expected_walk(cell, *, frames_on):
    if cell["end_reason"] in ("interrupted", "error") and cell["finalization"] == "complete-solved":
        return {"saved": "part", "reason": f"session-{cell['end_reason']}"}
    if (frames_on and cell["end_reason"] == "stop" and cell["finalization"].startswith("complete")
            and cell["capture"] == "fewer-observed"):
        return {"saved": "part", "reason": "frames-not-read"}
    return None


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    root = tmp_path_factory.mktemp("upartial-on")
    capture_root = root / "capture"
    return capture_root, pin.build_matrix(root / "worlds", capture_root)


@pytest.mark.parametrize("frames", [None, "1"], ids=["R", "R+F"])
def test_every_matrix_cell_changes_exactly_as_the_rule_says(matrix, monkeypatch, frames):
    """Every non-partial cell: ON bytes equal OFF bytes. Every partial cell: only the state word (READY
    / `complete` only), the reason that goes with it, and `finalization.walk`/`notice` move."""
    capture_root, built = matrix
    monkeypatch.delenv("TOWER_WORLD_PARTIAL_STATE", raising=False)
    monkeypatch.delenv("TOWER_WORLD_PARTIAL_FRAMES", raising=False)
    off = {sid: pin.surface(store, capture_root, wid, sid) for store, wid, sid, _ in built}
    monkeypatch.setenv("TOWER_WORLD_PARTIAL_STATE", "1")
    if frames:
        monkeypatch.setenv("TOWER_WORLD_PARTIAL_FRAMES", frames)
    partial = 0
    for store, wid, sid, cell in built:
        payload, listing = pin.surface(store, capture_root, wid, sid)
        off_payload, off_listing = off[sid]
        expected = _expected_walk(cell, frames_on=bool(frames))
        if expected is None:
            assert _canon([payload, listing]) == _canon([off_payload, off_listing]), cell
            continue
        partial += 1
        lifecycle, off_lifecycle = payload["lifecycle"], off_payload["lifecycle"]
        row = listing["worlds"][0]["sessions"][0]
        off_row = off_listing["worlds"][0]["sessions"][0]
        assert lifecycle["finalization"]["walk"] == expected, cell
        assert row["finalization"] == lifecycle["finalization"], cell
        sentence = wbp.partial_walk(store.read_session(wid, sid),
                                    recorded=pin.FRAMES_WRITTEN_FEWER)["sentence"]
        assert lifecycle["finalization"]["notice"].startswith(sentence), cell
        if off_lifecycle["state"] == "ready":
            assert lifecycle["state"] == "interrupted", cell
            assert lifecycle["reason"] == payload["model_state_reason"] == sentence, cell
        else:
            assert lifecycle["state"] == off_lifecycle["state"], cell
            assert lifecycle["reason"] == off_lifecycle["reason"], cell
        assert row["state"] == ("interrupted" if off_row["state"] == "complete" else off_row["state"]), cell
        assert payload["model_state"] not in ("finalized",), cell
        assert row["state"] != "complete", cell
        allowed = {".lifecycle.state", ".lifecycle.evidence", ".lifecycle.reason", ".model_state",
                   ".model_state_reason", ".lifecycle.finalization.walk",
                   ".lifecycle.finalization.notice", ".world_snapshot.revision"}
        assert set(_diff_paths(off_payload, payload)) <= allowed, (cell, _diff_paths(off_payload, payload))
    # 2 end reasons x 2 notices x 3 trees x 4 rooms x 3 captures under R; F adds 3 complete
    # finalizations x 2 notices x 3 trees x 4 rooms of `stop` with fewer frames observed.
    assert partial == (144 + 72 if frames else 144)


# -- C9: nothing is written ---------------------------------------------------------------------------


def test_reading_a_partial_world_writes_nothing(tmp_path, switch):
    switch("1", "1")
    worlds = [World(tmp_path / "r", index=0),
              World(tmp_path / "f", index=1, end_reason="stop", capture="fewer-observed")]
    files = []
    for world in worlds:
        files += [world.store.session_path(world.world_id, world.session_id),
                  world.store.events_path(world.world_id, world.session_id)]
    files.append(worlds[1].manifest_path)

    def state():
        return [(p, p.read_bytes(), p.stat().st_mtime_ns) for p in files]

    before = state()
    for world in worlds:
        assert world.payload()["lifecycle"]["finalization"]["walk"]["saved"] == "part"
        assert world.row()["finalization"]["walk"]["saved"] == "part"
    assert state() == before


# -- C10: the phone's text guard and the length bound --------------------------------------------------


def looks_like_machine_output(text: str) -> bool:
    """A Python port of `WorldTowerText.looksLikeMachineOutput` (`WorldSelection.swift`, `origin/ios/ux-v1`
    @ `0f1249e`): the phone replaces any Tower text this matches with a generic sentence."""
    if len(text) > 700:
        return True
    if "\\" in text or "\n" in text or "\r" in text:
        return True
    trimmed = text.strip()
    if trimmed.startswith("Traceback") or "Traceback (most recent call last)" in text:
        return True
    if re.search(r'File "[^"]*", line \d+', text):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)\s*[:(]", trimmed):
        return True
    if re.match(r"^(Error|Exception)\b", trimmed):
        return True
    if re.search(r"\b[A-Z][A-Za-z0-9]*(Error|Exception)\b", text):
        return True
    if re.search(r"\b[a-z_][a-z0-9_]*=\S", text):
        return True
    if re.search(r"\{\s*[\"']", text):
        return True
    if re.search(r"(^|[\s(\"'\[=])~?/[A-Za-z0-9._-]+/", text):
        return True
    return False


def test_the_port_of_the_phone_guard_still_catches_what_it_catches():
    for text in ("RuntimeError: boom", r"C:\Users\x", "a\nb", "keyframes=24", '{"state": 1}',
                 "/Users/x/y", "Traceback (most recent call last)", "x" * 701):
        assert looks_like_machine_output(text), text
    assert not looks_like_machine_output("the depth stage did not finish")


@pytest.mark.parametrize("count", [0, 1, 12, 999999])
def test_both_walk_sentences_pass_the_phones_guard_verbatim(count):
    filled = WALK_NOTICE_SENTENCES["walk-frames-not-read"].format(observed=count, recorded=count)
    for text in (EARLY, filled):
        assert not looks_like_machine_output(text), text
        assert "\n" not in text and "\r" not in text
    assert re.findall(r"\d+", EARLY) == []
    assert re.findall(r"\d+", filled) == [str(count), str(count)]


def _closed_set_compositions():
    """Every notice the Tower can write today, filled at its widest: each sentence alone, masks then gate,
    masks + excluded + gate, and the finisher's three sentences around the longest clause."""
    from scripts import world_finish_pending as wfp

    fill = dict(unmasked=999999, images=999999, excluded=999999)
    sentences = {cause: text.format(**fill) for cause, text in CP.NOTICE_SENTENCES.items()}
    masks = [text for cause, text in sentences.items() if cause.startswith("masks-")]
    gates = [text for cause, text in sentences.items() if not cause.startswith("masks-")]
    clauses = list(CP.NOTICE_CLAUSES.values())
    out = list(sentences.values())
    out += [f"{m}; {g}" for m in masks for g in gates]
    out += [f"{m}; {sentences['masks-excluded']}; {g}" for m in masks for g in gates]
    finisher = [wfp.REGATE_GIVEN_UP.format(what=c, attempts=999999) for c in clauses]
    finisher += [wfp.REGATE_REFUSED_NOTICE.format(what=c) for c in clauses]
    finisher.append(wfp.REFINISH_PARKED_NOTICE)
    out += finisher
    out += [f"{m}; {sentences['masks-excluded']}; {f}" for m in masks for f in finisher]
    return out


def test_every_composition_with_a_walk_sentence_stays_under_the_bound():
    widest_walk = WALK_NOTICE_SENTENCES["walk-frames-not-read"].format(observed=999999, recorded=999999)
    assert len(widest_walk) == 148 and len(EARLY) == 133
    longest = 0
    for composition in _closed_set_compositions():
        for walk in (EARLY, widest_walk):
            composed = f"{walk}; {composition}"
            longest = max(longest, len(composed))
            assert len(composed) <= CP.FINALIZATION_TEXT_MAX_CHARS, composed
            assert not looks_like_machine_output(composed), composed
            out = present_finalization({"state": "complete", "notice": composition},
                                       {"saved": "part", "reason": "x", "sentence": walk})
            assert out["notice"] == composed
    assert longest < 700


def test_a_legacy_notice_too_long_to_compose_yields_the_walk_sentence_alone(tmp_path, switch):
    legacy = "an old notice " * 60
    world = World(tmp_path, notice="depth-unavailable", notice_text=legacy[:700])
    switch("1")
    payload, row = world.payload(), world.row()
    assert payload["lifecycle"]["finalization"]["notice"] == EARLY
    assert row["finalization"]["notice"] == EARLY
    # A composition of exactly 700 is kept; 701 is not.
    walk = {"saved": "part", "reason": "x", "sentence": EARLY}
    fits = "y" * (700 - len(EARLY) - 2)
    assert present_finalization({"notice": fits}, walk)["notice"] == f"{EARLY}; {fits}"
    assert present_finalization({"notice": fits + "y"}, walk)["notice"] == EARLY


def test_present_finalization_is_the_same_object_with_no_walk_and_a_copy_with_one():
    record = {"state": "complete", "final_solve": "solved", "notice": "  "}
    assert present_finalization(record, None) is record
    assert present_finalization(None, None) is None
    out = present_finalization(record, {"saved": "part", "reason": "session-error", "sentence": EARLY})
    assert out is not record and record == {"state": "complete", "final_solve": "solved", "notice": "  "}
    assert out["notice"] == EARLY, "a blank record notice is not composed"
    assert out["walk"] == {"saved": "part", "reason": "session-error"}


# -- C11: the payload bound ----------------------------------------------------------------------------


def test_the_status_envelope_with_a_walk_stays_under_the_channel_bound(tmp_path, monkeypatch):
    """CARTRIDGE-RESULTS §8: a snapshot is measured < 8 KB (`test_result_channel_protocol`). The incident
    shape, with a gate notice, end to end over `/ws` with both switches on and the capture root the Tower
    was configured with."""
    from tests.result_channel_fixtures import drain, make_client, subscribe

    world = World(tmp_path, end_reason="stop", capture="fewer-observed", notice="depth-unavailable")
    monkeypatch.setenv("TOWER_WORLD_PARTIAL_STATE", "1")
    monkeypatch.setenv("TOWER_WORLD_PARTIAL_FRAMES", "1")
    monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(world.capture_root))
    client = make_client(monkeypatch, world.store.root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws, world_id=world.world_id, session_id=world.session_id)
        envelope = drain(ws, expect="cartridge_result")
    lifecycle = envelope["payload"]["lifecycle"]
    assert lifecycle["finalization"]["walk"] == {"saved": "part", "reason": "frames-not-read"}
    assert lifecycle["finalization"]["notice"] == f"{INCIDENT}; {GATE_NOTICE}"
    encoded = json.dumps(envelope)
    assert len(encoded) < 8000, f"snapshot grew to {len(encoded)} bytes"

    # And `GET /worlds` through the route, which hands the app's capture root to rule F.
    row = client.get("/worlds").json()["worlds"][0]["sessions"][0]
    assert row["state"] == "interrupted"
    assert row["finalization"] == lifecycle["finalization"]


# -- C12: switch parsing -------------------------------------------------------------------------------


@pytest.mark.parametrize("value, on", [("1", True), ("true", True), ("yes", True), ("on", True),
                                       ("ON", True), (" True ", True), (None, False), ("", False),
                                       ("0", False), ("off", False), ("false", False),
                                       ("garbage", False)])
def test_the_switch_reads_the_way_every_tower_flag_reads(tmp_path, switch, value, on):
    from tower import config

    switch(value, None)
    assert config.world_partial_state_setting() is on
    world = World(tmp_path)
    assert world.payload()["lifecycle"]["state"] == ("interrupted" if on else "ready")
    assert world.row()["state"] == ("interrupted" if on else "complete")


def test_both_switches_are_off_by_default(switch):
    from tower import config

    assert config.world_partial_state_setting() is False
    assert config.world_partial_frames_setting() is False


def test_the_frames_switch_alone_changes_nothing(tmp_path, switch):
    worlds = [World(tmp_path / "f", index=0, end_reason="stop", capture="fewer-observed"),
              World(tmp_path / "r", index=1)]
    off = _canon([[w.payload(), w.row()] for w in worlds])
    switch(None, "1")
    assert _canon([[w.payload(), w.row()] for w in worlds]) == off


# -- D: rule F ------------------------------------------------------------------------------------------


def test_the_incident_shape_is_a_walk_whose_frames_were_not_read(tmp_path, switch):
    """2026-10-02, world b08c294b, session 8140a195: `stop`, complete/solved, 805 of 3,197 frames."""
    world = World(tmp_path, end_reason="stop", capture="fewer-observed")
    session = world.store.read_session(world.world_id, world.session_id)
    assert (session.end_reason, session.frames_observed) == ("stop", 805)
    switch("1", "1")
    payload, row = world.payload(), world.row()
    lifecycle = payload["lifecycle"]
    assert lifecycle["state"] == "interrupted" and payload["model_state"] == "interrupted"
    assert lifecycle["finalization"]["walk"] == {"saved": "part", "reason": "frames-not-read"}
    assert lifecycle["reason"] == payload["model_state_reason"] == INCIDENT
    assert lifecycle["finalization"]["notice"] == INCIDENT
    assert row["state"] == "interrupted"
    assert row["finalization"] == lifecycle["finalization"]

    # R alone does not see it.
    switch("1", None)
    assert world.payload()["lifecycle"]["state"] == "ready"
    assert world.row()["state"] == "complete"


def test_equal_counts_are_a_whole_walk(tmp_path, switch):
    world = World(tmp_path, end_reason="stop", capture="equal")
    off = _canon([world.payload(), world.row()])
    switch("1", "1")
    assert _canon([world.payload(), world.row()]) == off


def _manifest(world, **fields):
    body = {"schema_version": 1, "capture_id": world.capture_id, "started_at": 1.0, "ended_at": 5.0,
            "end_reason": "stop", "frames_written": 3197}
    body.update(fields)
    world.manifest_path.write_text(json.dumps(body), encoding="utf-8")


@pytest.mark.parametrize("damage", [
    "open", "other-id", "truncated", "not-json", "a-list", "missing", "bool", "negative", "float",
    "string-ended", "bool-ended", "no-capture-root", "no-capture-id", "capture-id-escapes",
])
def test_a_capture_manifest_that_cannot_judge_decides_nothing(tmp_path, switch, damage):
    world = World(tmp_path, end_reason="stop", capture="fewer-observed")
    capture_root = "default"
    if damage == "open":
        _manifest(world, ended_at=None)
    elif damage == "other-id":
        _manifest(world, capture_id="c9999")
    elif damage == "truncated":
        world.manifest_path.write_text('{"capture_id": "c0000", "ended_at": 5.0, "frames_wr', encoding="utf-8")
    elif damage == "not-json":
        world.manifest_path.write_bytes(b"\xff\xfe\x00garbage")
    elif damage == "a-list":
        world.manifest_path.write_text("[1, 2, 3]", encoding="utf-8")
    elif damage == "missing":
        world.manifest_path.unlink()
    elif damage == "bool":
        _manifest(world, frames_written=True)
    elif damage == "negative":
        _manifest(world, frames_written=-1)
    elif damage == "float":
        _manifest(world, frames_written=3197.0)
    elif damage == "string-ended":
        _manifest(world, ended_at="5.0")
    elif damage == "bool-ended":
        _manifest(world, ended_at=True)
    elif damage == "no-capture-root":
        capture_root = None
    elif damage == "no-capture-id":
        world.rewrite(capture_id=None)
    elif damage == "capture-id-escapes":
        # A record naming a path outside `captures/` is not followed, even to a manifest that would judge.
        escape = tmp_path / "capture" / "elsewhere"
        escape.mkdir(parents=True)
        (escape / "capture.json").write_text(json.dumps(
            {"capture_id": "../elsewhere", "ended_at": 5.0, "frames_written": 3197}), encoding="utf-8")
        world.rewrite(capture_id="../elsewhere")
    off = _canon([world.payload(capture_root=capture_root), world.row(capture_root=capture_root)])
    switch("1", "1")
    on = _canon([world.payload(capture_root=capture_root), world.row(capture_root=capture_root)])
    assert on == off
    session = world.store.read_session(world.world_id, world.session_id)
    root = world.capture_root if capture_root == "default" else capture_root
    assert recorded_frames(root, session) is None


def test_recorded_frames_never_raises(tmp_path):
    class Odd:
        capture_id = "c0000"

    folder = tmp_path / "captures" / "c0000"
    folder.mkdir(parents=True)
    (folder / "capture.json").mkdir()  # a directory where the file should be
    assert recorded_frames(tmp_path, Odd()) is None
    assert recorded_frames(tmp_path, object()) is None
    assert recorded_frames(object(), Odd()) is None


def test_a_reconnect_lineage_is_judged_by_its_first_capture_alone(tmp_path, switch):
    """The documented false negative (spec §1.5): observed above the first capture's count."""
    world = World(tmp_path, end_reason="stop", capture="fewer-observed")
    _manifest(world, frames_written=600)
    off = _canon([world.payload(), world.row()])
    switch("1", "1")
    assert _canon([world.payload(), world.row()]) == off


def test_rule_f_needs_only_a_complete_finalization_and_a_closed_record(tmp_path, switch):
    switch("1", "1")
    world = World(tmp_path, end_reason="stop", finalization="complete-skipped", capture="fewer-observed")
    assert world.payload()["lifecycle"]["finalization"]["walk"]["reason"] == "frames-not-read"
    world.rewrite(finalization={"state": "pending", "final_solve": "pending"})
    assert "walk" not in (world.payload()["lifecycle"]["finalization"] or {})
    world.rewrite(finalization={"state": "complete", "final_solve": "solved"}, ended_at=None)
    assert "walk" not in (world.payload()["lifecycle"]["finalization"] or {})


def test_the_producer_reads_the_capture_manifest_once_per_change(tmp_path, switch, monkeypatch):
    world = World(tmp_path, end_reason="stop", capture="fewer-observed")
    calls = []

    def spy(capture_root, session):
        calls.append(session.capture_id)
        return recorded_frames(capture_root, session)

    monkeypatch.setattr(wb, "recorded_frames", spy)
    monkeypatch.setattr(wbl, "recorded_frames", spy)
    switch("1", "1")
    producer = WorldBuilderStatusProducer(world.store.root, lambda: pin.CLOCK,
                                          capture_root=world.capture_root)
    for _ in range(3):
        producer.snapshot(world.world_id, world.session_id)
    assert calls == ["c0000"]
    _manifest(world, frames_written=4000, bytes_written=123456)
    snapshot = producer.snapshot(world.world_id, world.session_id)
    assert calls == ["c0000", "c0000"]
    assert "805 of the 4000 frames" in snapshot.payload["lifecycle"]["reason"]

    # Off, the manifest is never read at all.
    switch(None, "1")
    producer.snapshot(world.world_id, world.session_id)
    build_world_listing(world.store, capture_root=world.capture_root)
    assert calls == ["c0000", "c0000"]


def test_the_hub_hands_the_capture_root_to_the_producer(tmp_path, switch):
    from tower.results import make_snapshot_for
    from tower.results.contracts import CARTRIDGE_WORLD_BUILDER, RESULT_TYPE_STATUS

    world = World(tmp_path, end_reason="stop", capture="fewer-observed")
    switch("1", "1")
    with_root = make_snapshot_for(world.store.root, lambda: pin.CLOCK, capture_root=world.capture_root)
    without = make_snapshot_for(world.store.root, lambda: pin.CLOCK)
    a = with_root(CARTRIDGE_WORLD_BUILDER, RESULT_TYPE_STATUS, world.world_id, world.session_id).payload
    b = without(CARTRIDGE_WORLD_BUILDER, RESULT_TYPE_STATUS, world.world_id, world.session_id).payload
    assert a["lifecycle"]["finalization"]["walk"]["reason"] == "frames-not-read"
    assert "walk" not in b["lifecycle"]["finalization"]


# -- partial_walk, pure -----------------------------------------------------------------------------------


def _session(**fields):
    base = dict(session_id="s", world_id="w", started_at=1.0, ended_at=2.0, end_reason="interrupted",
                capture_id="c", frames_observed=10,
                finalization={"state": "complete", "final_solve": "solved"})
    base.update(fields)
    return Session(**base)


@pytest.mark.parametrize("fields, recorded, reason", [
    ({}, None, "session-interrupted"),
    ({"end_reason": "error"}, None, "session-error"),
    ({"end_reason": "interrupted"}, 99, "session-interrupted"),
    ({"end_reason": "stop"}, 11, "frames-not-read"),
    ({"end_reason": "stop", "finalization": {"state": "complete", "final_solve": "skipped"}}, 11,
     "frames-not-read"),
    ({"end_reason": "stop"}, 10, None),
    ({"end_reason": "stop"}, 9, None),
    ({"end_reason": "stop"}, None, None),
    ({"end_reason": "stop", "capture_id": None}, 11, None),
    ({"end_reason": "stop", "capture_id": ""}, 11, None),
    ({"end_reason": "stop", "frames_observed": True}, 11, None),
    ({"end_reason": "stop", "frames_observed": -1}, 11, None),
    ({"end_reason": "stop"}, True, None),
    ({"end_reason": "disconnect"}, 11, None),
    ({"end_reason": None}, None, None),
    ({"ended_at": None}, None, None),
    ({"finalization": None}, None, None),
    ({"finalization": "complete"}, None, None),
    ({"finalization": {"state": "pending", "final_solve": "solved"}}, None, None),
    ({"finalization": {"state": "complete", "final_solve": "skipped"}}, None, None),
    ({"finalization": {"state": "complete", "final_solve": "failed"}}, None, None),
    ({"finalization": {"state": "complete"}}, None, None),
])
def test_partial_walk_is_the_rule_and_nothing_else(fields, recorded, reason):
    walk = partial_walk(_session(**fields), recorded=recorded)
    if reason is None:
        assert walk is None
    else:
        assert walk["saved"] == "part" and walk["reason"] == reason
        assert walk["sentence"] == (
            WALK_NOTICE_SENTENCES["walk-frames-not-read"].format(observed=10, recorded=recorded)
            if reason == "frames-not-read" else EARLY)


# -- the old client: `origin/ios/ux-v1` @ 0f1249e, decoded in Python -------------------------------------

# `WorldBuilderResultDecoder.modelState(from:)` (TowerWorldBuilderClient.swift): any other word fails the
# whole panel as `.undecodableResponse`.
PHONE_MODEL_STATES = {"unsupported", "idle", "receiving", "finalizing", "finalized", "interrupted", "failed"}
# `WorldListingSessionState` (WorldLibrary.swift): an unknown word survives but prints raw.
PHONE_ROW_STATES = {"receiving", "finalizing", "complete", "interrupted", "unbuilt"}


def _phone_finalization(value):
    """`WorldFinalizationReport.init?(json:)`: nil without a string `state`; named keys only."""
    if not isinstance(value, dict) or not isinstance(value.get("state"), str):
        return None
    notice = value.get("notice")
    notice = notice if isinstance(notice, str) and notice.strip() else None
    return {"state": value["state"], "final_solve": value.get("final_solve"), "notice": notice}


def _phone_panel(payload):
    """What the installed phone shows for a status payload: (headline stage, reason as drawn)."""
    state = payload["model_state"]
    assert state in PHONE_MODEL_STATES, f"{state!r} would fail the whole panel on an installed phone"
    if state == "interrupted":
        assert payload["world_snapshot"] is not None, "interrupted with no snapshot reads as a failure"
    reason = payload["model_state_reason"]
    drawn = "The Tower gave a technical reason, which is not shown here." if (
        reason is not None and looks_like_machine_output(reason)) else reason
    return state, drawn


def _phone_row(row):
    """`WorldListingSession.init?(json:)`: nil without these four, typed."""
    assert isinstance(row["session_id"], str)
    assert isinstance(row["started_at"], (int, float)) and not isinstance(row["started_at"], bool)
    assert isinstance(row["frame_source"], str)
    assert isinstance(row["has_geometry"], bool)
    return row["state"], _phone_finalization(row["finalization"])


@pytest.mark.parametrize("cell, frames", [
    (dict(end_reason="interrupted"), None),
    (dict(end_reason="error", notice="depth-unavailable"), None),
    (dict(end_reason="stop", capture="fewer-observed", notice="depth-unavailable"), "1"),
    (dict(end_reason="interrupted", tree="absent"), None),
    (dict(end_reason="interrupted", photographic="owed"), None),
])
def test_an_installed_phone_decodes_a_partial_world_and_never_calls_it_saved(tmp_path, switch, cell, frames):
    from tower.results.contracts import WORLD_BUILDER_STATUS_CONTRACT
    from tower.results.world_builder_library import WORLDS_CONTRACT

    # No identifier moves (spec §3.3, Q4): the phone compares both for equality.
    assert WORLD_BUILDER_STATUS_CONTRACT == "world_builder.status/2026-09-10"
    assert WORLDS_CONTRACT == "world_builder.worlds/2026-09-10"

    world = World(tmp_path, **cell)
    switch("1", frames)
    payload = world.payload()
    listing = build_world_listing(world.store, capture_root=world.capture_root)
    assert listing["contract"] == WORLDS_CONTRACT

    state, drawn = _phone_panel(payload)
    assert state != "finalized", "an installed phone would say Saved"
    assert state in {"interrupted", "finalizing"}
    assert drawn == payload["model_state_reason"], "the phone's guard would hide the Tower's sentence"
    finalization = _phone_finalization(payload["lifecycle"]["finalization"])
    assert finalization is not None and finalization["state"] == "complete"
    assert finalization["notice"] == payload["lifecycle"]["finalization"]["notice"]
    assert not looks_like_machine_output(finalization["notice"]), "the room would hide the notice"
    assert finalization["notice"].startswith("only part of this walk is in this world: ")

    row_state, row_finalization = _phone_row(listing["worlds"][0]["sessions"][0])
    assert row_state in PHONE_ROW_STATES and row_state != "complete"
    assert row_finalization == finalization
    assert {"interrupted": "interrupted", "finalizing": "finalizing"}[state] == row_state


def test_every_matrix_cell_still_decodes_on_an_installed_phone_with_both_switches_on(matrix, monkeypatch):
    capture_root, built = matrix
    monkeypatch.setenv("TOWER_WORLD_PARTIAL_STATE", "1")
    monkeypatch.setenv("TOWER_WORLD_PARTIAL_FRAMES", "1")
    for store, wid, sid, cell in itertools.islice(built, 0, None, 7):
        payload, listing = pin.surface(store, capture_root, wid, sid)
        state, _ = _phone_panel(payload)
        row_state, _ = _phone_row(listing["worlds"][0]["sessions"][0])
        assert row_state in PHONE_ROW_STATES
        if "walk" in (payload["lifecycle"]["finalization"] or {}):
            assert state != "finalized" and row_state != "complete", cell
