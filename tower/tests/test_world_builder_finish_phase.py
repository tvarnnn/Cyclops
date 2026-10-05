"""T-UX1: what the builder is doing between Stop and the end of its final solve, and `basis`.

Spec: `RUN\\lead\\specs\\U06-TUX1-PROGRESS-SPEC-20261004.md` §3 and §7.1 (T1, T3-T13; T2, the
OFF identity pin against 5c0dc12, is `test_world_builder_finish_stages_identity.py`).
Contracts: WORLDS §2b and §4a `basis`; CARTRIDGE-RESULTS §10.1 `lifecycle.processing`.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_world_builder_finish_stages_identity import (  # noqa: F401 -- fixtures
    EPOCH,
    LOCKS,
    PHASES,
    RECORDS,
    _make_cell,
    _run_builder,
    base_world,
    processes,
    rendered,
)
from tower import config
from tower.world_builder import finish_phase as FP

ON = "on"
SWITCHES = (config.WORLD_FINISH_STAGES_ENV, config.WORLD_PICTURE_BASIS_ENV)


def _on(monkeypatch, finish=True, basis=True):
    for name, value in zip(SWITCHES, (finish, basis)):
        if value:
            monkeypatch.setenv(name, ON)
        else:
            monkeypatch.delenv(name, raising=False)


def _off(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# T1 switches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    (None, False), ("", False), ("  ", False), ("0", False), ("off", False), ("no", False),
    ("false", False), ("garbage", False), ("1", True), ("true", True), ("TRUE", True),
    ("yes", True), ("On", True), (" on ", True)])
def test_t1_both_switches_parse_like_every_flag_and_default_off(monkeypatch, value, expected):
    for name, read in ((config.WORLD_FINISH_STAGES_ENV, config.world_finish_stages_setting),
                       (config.WORLD_PICTURE_BASIS_ENV, config.world_picture_basis_setting)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
        assert read() is expected, (name, value)
    assert config.WORLD_FINISH_STAGES_ENV == "TOWER_WORLD_FINISH_STAGES"
    assert config.WORLD_PICTURE_BASIS_ENV == "TOWER_WORLD_PICTURE_BASIS"


# ---------------------------------------------------------------------------
# T3 the projection
# ---------------------------------------------------------------------------

OWNER = {"pid": 4212, "created_at": 1790620497.11}
CHILD = {"pid": 4560, "created_at": 1790620930.20}
SID = "s" * 32


def _doc(**kw):
    doc = {"format": FP.FORMAT, "session_id": SID, "epoch": EPOCH, "owner": dict(OWNER),
           "writer": dict(CHILD), "stage": "checking", "step": {"n": 2, "of": 3},
           "written_at": EPOCH + 9.0}
    doc.update(kw)
    return doc


def _session(**fin):
    finalization = {"state": "pending", "final_solve": "pending", "started_at": EPOCH,
                    "updated_at": EPOCH, "detail": None}
    finalization.update(fin)
    return SimpleNamespace(ended_at=EPOCH - 1.0, finalization=finalization)


HOLDER = {"pid": OWNER["pid"], "alive": True, "unreadable": False}


def _running(*alive):
    names = {json.dumps(i, sort_keys=True) for i in alive}
    return lambda identity: json.dumps(identity, sort_keys=True) in names


def _project(doc=None, session=None, holder=HOLDER, running=(OWNER, CHILD)):
    return FP.project(_doc() if doc is None else doc, session_id=SID,
                      session=_session() if session is None else session, holder=holder,
                      is_running=_running(*running))


def test_t3_projected_only_when_every_proof_holds():
    assert _project() == {"stage": "checking", "step": {"n": 2, "of": 3}}
    assert list(_project()) == ["stage", "step"], "no timestamp, no writer, no epoch on the wire"
    # the writer IS the owner (the builder parent's own marks): one probe is enough
    assert _project(_doc(writer=dict(OWNER), stage="assembling", step=None),
                    running=(OWNER,)) == {"stage": "assembling"}


@pytest.mark.parametrize("case, kw", [
    ("writer dead (child died, parent alive)", {"running": (OWNER,)}),
    ("owner dead", {"running": (CHILD,)}),
    ("owner is not the holder (finisher / world_finalize / new walk)",
     {"holder": {"pid": 9999, "alive": True, "unreadable": False}}),
    ("holder dead", {"holder": dict(HOLDER, alive=False)}),
    ("holder absent (lock does not speak for the session)", {"holder": None}),
    ("holder alive is truthy, not True", {"holder": dict(HOLDER, alive=1)}),
    ("holder pid is a bool", {"holder": {"pid": True, "alive": True}}),
    ("epoch mismatch", {"doc": _doc(epoch=EPOCH - 60.0)}),
    ("finalization complete", {"session": _session(state="complete")}),
    ("finalization interrupted", {"session": _session(state="interrupted")}),
    ("finalization None", {"session": SimpleNamespace(ended_at=EPOCH, finalization=None)}),
    ("started_at missing", {"session": _session(started_at=None)}),
    ("started_at a bool", {"session": _session(started_at=True)}),
    ("record open", {"session": SimpleNamespace(ended_at=None, finalization=_session().finalization)}),
    ("unknown stage", {"doc": _doc(stage="replaying")}),
    ("other session", {"doc": _doc(session_id="t" * 32)}),
    ("wrong format", {"doc": _doc(format="wb-finish-phase/2")}),
    ("owner missing", {"doc": {k: v for k, v in _doc().items() if k != "owner"}}),
    ("writer missing", {"doc": {k: v for k, v in _doc().items() if k != "writer"}}),
    ("owner created_at missing", {"doc": _doc(owner={"pid": OWNER["pid"]})}),
    ("writer pid a bool", {"doc": _doc(writer={"pid": True, "created_at": 1.0})}),
    ("recycled owner pid (alive, other start time)",
     {"running": ({"pid": OWNER["pid"], "created_at": OWNER["created_at"] + 50}, CHILD)}),
    ("not a dict", {"doc": ["checking"]}),
])
def test_t3_absent_for_each_single_violation(case, kw):
    assert _project(**kw) is None, case


@pytest.mark.parametrize("step, kept", [
    ({"n": 1, "of": 3}, {"n": 1, "of": 3}), ({"n": 3, "of": 3}, {"n": 3, "of": 3}),
    ({"n": 7, "of": 7}, {"n": 7, "of": 7}), ({"n": 4, "of": 3}, None), ({"n": 1, "of": 9}, None),
    ({"n": 0, "of": 3}, None), ({"n": True, "of": 3}, None), ({"n": 1, "of": True}, None),
    ({"n": 1.0, "of": 3}, None), ({"n": "1", "of": 3}, None), ([1, 3], None), (None, None)])
def test_t3_step_kept_only_in_range_and_the_stage_survives_a_bad_one(step, kept):
    expected = {"stage": "checking"} if kept is None else {"stage": "checking", "step": kept}
    assert _project(_doc(step=step)) == expected
    # never on another stage
    assert _project(_doc(stage="placing", step=step)) == {"stage": "placing"}


def test_t3_truncated_and_malformed_files_read_as_no_phase(tmp_path):
    path = tmp_path / FP.FILENAME
    assert FP.read_phase(path) is None
    path.write_text(json.dumps(_doc())[:40], encoding="utf-8")
    assert FP.read_phase(path) is None
    path.write_text("[]", encoding="utf-8")
    assert FP.read_phase(path) is None
    path.write_text(json.dumps(_doc(epoch=float("nan")).__repr__()), encoding="utf-8")
    assert FP.read_phase(path) is None
    path.write_text(json.dumps(_doc()), encoding="utf-8")
    assert FP.read_phase(path) == _doc()


def test_t3_a_real_finished_process_is_not_running(processes):
    from tower.world_builder.store import process_is_running

    assert process_is_running(processes["live"]) is True
    assert process_is_running(processes["dead"]) is False
    doc = _doc(owner=processes["live"], writer=processes["dead"])
    holder = {"pid": processes["live"]["pid"], "alive": True}
    session = _session()
    assert FP.project(doc, session_id=SID, session=session, holder=holder) is None
    doc["writer"] = processes["live"]
    assert FP.project(doc, session_id=SID, session=session, holder=holder) == {
        "stage": "checking", "step": {"n": 2, "of": 3}}


# ---------------------------------------------------------------------------
# T4-T6, T12, T13: the status channel and the listing, on the identity matrix
# ---------------------------------------------------------------------------


def _status(root, world_id, session_id, producer=None):
    from tower.results.world_builder import WorldBuilderStatusProducer

    producer = producer or WorldBuilderStatusProducer(root, lambda: 1_790_000_500.0)
    return producer.snapshot(world_id=world_id, session_id=session_id)


def _row(root, world_id, session_id):
    from tower.results.world_builder_library import build_world_listing
    from tower.world_builder.store import WorldStore

    listing = build_world_listing(WorldStore(root))
    (row,) = [s for w in listing["worlds"] if w["world_id"] == world_id
              for s in w["sessions"] if s["session_id"] == session_id]
    return row


def test_t4_t6_processing_only_on_the_proved_cell_and_row_equals_panel(base_world, processes,
                                                                      tmp_path, monkeypatch):
    seen = {}
    for record in RECORDS:
        for lock in LOCKS:
            for phase in PHASES:
                cell = f"{record}_{lock}_{phase}"
                # short directory names: Windows MAX_PATH (the world tree adds ~150 characters)
                root, w, s = _make_cell(base_world, processes, tmp_path / f"c{len(seen)}",
                                        record, lock, phase)
                _off(monkeypatch)
                off = _status(root, w, s)
                off_row = _row(root, w, s)
                _on(monkeypatch, basis=False)
                on = _status(root, w, s)
                on_row = _row(root, w, s)
                processing = on.payload["lifecycle"].get("processing")
                seen[cell] = processing
                # T6: the row says exactly what the panel says
                assert on_row.get("processing") == processing, cell
                # T4: nothing else moved
                # (the snapshot's revision IS the envelope's, which moves with the stage: T5)
                stripped = json.loads(json.dumps(on.payload))
                stripped["lifecycle"].pop("processing", None)
                stripped["world_snapshot"].pop("revision")
                off_payload = json.loads(json.dumps(off.payload))
                off_payload["world_snapshot"].pop("revision")
                assert stripped == off_payload, cell
                assert on.payload["world_snapshot"]["revision"] == on.revision
                row_stripped = dict(on_row)
                row_stripped.pop("processing", None)
                assert row_stripped == off_row, cell
                if processing is not None:
                    assert list(on.payload["lifecycle"])[-1] == "processing", cell
                    assert list(on_row)[-1] == "processing", cell
                    assert on.payload["lifecycle"]["state"] == "finalizing"
                    assert on_row["state"] == "finalizing"
                    assert on.revision != off.revision
                else:
                    assert on.revision == off.revision, cell
    proved = {k: v for k, v in seen.items() if v is not None}
    assert proved == {"pending_live_valid": {"stage": "checking", "step": {"n": 2, "of": 3}}}


def _rewrite_phase(root, world_id, session_id, **kw):
    path = root / "worlds" / world_id / "solve" / session_id / FP.FILENAME
    if not path.exists():
        path = next(root.rglob(FP.FILENAME))
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc.update(kw)
    doc = {k: v for k, v in doc.items() if v is not None}
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_t5_revision_moves_with_the_stage_and_the_pass_and_never_between(base_world, processes,
                                                                         tmp_path, monkeypatch):
    from tower.results import world_builder as WB
    from tower.results.world_builder import WorldBuilderStatusProducer

    root, w, s = _make_cell(base_world, processes, tmp_path / "c", "pending", "live", "valid")
    _on(monkeypatch, basis=False)
    producer = WorldBuilderStatusProducer(root, lambda: 1_790_000_500.0)
    first = _status(root, w, s, producer)
    assert _status(root, w, s, producer).revision == first.revision
    # a rewrite with only a new `written_at` is the same stage: same revision
    _rewrite_phase(root, w, s, written_at=EPOCH + 500.0)
    assert _status(root, w, s, producer).revision == first.revision
    _rewrite_phase(root, w, s, step={"n": 3, "of": 3})
    third = _status(root, w, s, producer)
    assert third.payload["lifecycle"]["processing"] == {"stage": "checking", "step": {"n": 3, "of": 3}}
    assert third.revision != first.revision
    _rewrite_phase(root, w, s, stage="assembling", step=None)
    fourth = _status(root, w, s, producer)
    assert fourth.payload["lifecycle"]["processing"] == {"stage": "assembling"}
    assert fourth.revision not in (first.revision, third.revision)
    assert not any("processing" in path for path in WB.VOLATILE_PATHS)


def test_t12_the_payload_with_processing_stays_inside_the_bound(base_world, processes, tmp_path,
                                                               monkeypatch):
    max_payload_bytes = 8 * 1024        # CARTRIDGE-RESULTS §8: "measured < 8 KB"

    root, w, s = _make_cell(base_world, processes, tmp_path / "c", "pending", "live", "valid")
    _on(monkeypatch)
    payload = _status(root, w, s).payload
    assert payload["lifecycle"]["processing"]
    assert len(json.dumps(payload).encode("utf-8")) < max_payload_bytes
    assert len(json.dumps(payload["lifecycle"]["processing"])) <= 60


def test_t13_the_file_is_parsed_once_per_change_and_probed_twice_per_poll_at_most(
        base_world, processes, tmp_path, monkeypatch):
    from tower.results.world_builder import WorldBuilderStatusProducer
    from tower.world_builder import store as store_module

    root, w, s = _make_cell(base_world, processes, tmp_path / "c", "pending", "live", "valid")
    # the writer is a child of the owner, so both identities are probed
    import os

    from tower.world_builder.store import process_identity

    _rewrite_phase(root, w, s, writer=process_identity(os.getpid()))
    _on(monkeypatch, basis=False)
    reads, probes = [], []
    real_read, real_probe = FP.read_phase, store_module.process_is_running
    monkeypatch.setattr(FP, "read_phase", lambda path: reads.append(path) or real_read(path))
    monkeypatch.setattr(store_module, "process_is_running",
                        lambda identity: probes.append(identity) or real_probe(identity))
    producer = WorldBuilderStatusProducer(root, lambda: 1_790_000_500.0)
    for _ in range(3):
        before = len(probes)
        assert _status(root, w, s, producer).payload["lifecycle"]["processing"]
        assert len(probes) - before == 2
    assert len(reads) == 1, "the unchanged phase was parsed again"
    # off the lock-held finalizing arm nothing is probed
    root2, w2, s2 = _make_cell(base_world, processes, tmp_path / "d", "complete-solved", "none", "valid")
    reads.clear()
    probes.clear()
    _status(root2, w2, s2)
    _row(root2, w2, s2)
    assert reads == [] and probes == []


# ---------------------------------------------------------------------------
# T7 the writer
# ---------------------------------------------------------------------------


class _Store:
    def __init__(self, root, session=None):
        self.root = Path(root)
        self.session = session

    def world_dir(self, world_id):
        return self.root / world_id

    def read_session(self, world_id, session_id):
        if self.session is None:
            raise FileNotFoundError(session_id)
        return self.session


def _writer(tmp_path, session=None, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv(config.WORLD_FINISH_STAGES_ENV, ON)
    return FP.PhaseWriter.for_builder(_Store(tmp_path, session or _session()), "w1", SID)


def test_t7_for_builder_needs_the_switch_and_a_pending_record(tmp_path, monkeypatch):
    monkeypatch.delenv(config.WORLD_FINISH_STAGES_ENV, raising=False)
    assert FP.PhaseWriter.for_builder(_Store(tmp_path, _session()), "w1", SID) is None
    monkeypatch.setenv(config.WORLD_FINISH_STAGES_ENV, ON)
    for session in (_session(state="complete"), _session(started_at=None),
                    _session(started_at=float("inf")),
                    SimpleNamespace(ended_at=1.0, finalization=None)):
        assert FP.PhaseWriter.for_builder(_Store(tmp_path, session), "w1", SID) is None
    assert FP.PhaseWriter.for_builder(_Store(tmp_path, None), "w1", SID) is None
    writer = FP.PhaseWriter.for_builder(_Store(tmp_path, _session()), "w1", SID)
    from tower.world_builder.store import process_identity
    import os

    assert writer.owner == writer.writer == process_identity(os.getpid())
    assert writer.epoch == EPOCH
    assert writer.path == tmp_path / "w1" / "solve" / SID / FP.FILENAME


def test_t7_mark_writes_atomically_dedupes_and_ignores_unknown_words(tmp_path, monkeypatch):
    writer = _writer(tmp_path, monkeypatch=monkeypatch)
    writes = []
    real = FP.write_json_atomic
    monkeypatch.setattr(FP, "write_json_atomic", lambda p, d: writes.append(d) or real(p, d))
    writer.mark("replaying")
    assert writes == [] and not writer.path.exists()
    writer.mark(FP.PREPARING)
    writer.mark(FP.PREPARING)
    assert len(writes) == 1
    doc = json.loads(writer.path.read_text(encoding="utf-8"))
    assert doc["stage"] == "preparing" and "step" not in doc and doc["epoch"] == EPOCH
    assert doc["owner"] == doc["writer"] == writer.owner and doc["format"] == FP.FORMAT
    assert FP.read_phase(writer.path) == doc
    writer.mark(FP.CHECKING, (1, 3))
    writer.mark(FP.CHECKING, (1, 3))
    writer.mark(FP.CHECKING, (2, 3))
    assert [d.get("step") for d in writes] == [None, {"n": 1, "of": 3}, {"n": 2, "of": 3}]
    writer.mark(FP.PLACING, (1, 3))                     # a step on another stage is dropped
    assert "step" not in json.loads(writer.path.read_text(encoding="utf-8"))
    writer.mark(FP.CHECKING, (4, 3))                    # out of range: the stage, no step
    assert "step" not in json.loads(writer.path.read_text(encoding="utf-8"))
    assert not list(writer.path.parent.glob("*.tmp*")), "no staging file left behind"


def test_t7_mark_and_clear_never_raise_and_warn_once(tmp_path, monkeypatch, caplog):
    blocker = tmp_path / "w1"
    blocker.write_text("a file where the world directory should be", encoding="utf-8")
    writer = FP.PhaseWriter(blocker / "solve" / SID / FP.FILENAME, world_id="w1", session_id=SID,
                            epoch=EPOCH, owner=OWNER, writer=OWNER)
    with caplog.at_level(logging.WARNING, logger=FP.__name__):
        writer.mark(FP.PREPARING)
        writer.mark(FP.MATCHING)
        writer.mark(FP.CHECKING, ("x", None))
        writer.clear()
    assert len([r for r in caplog.records if r.name == FP.__name__]) == 1
    # mark with a broken step tuple does not raise either
    writer.mark(FP.CHECKING, None)


def test_t7_clear_unlinks_and_is_idempotent_and_marks_again_after(tmp_path, monkeypatch):
    writer = _writer(tmp_path, monkeypatch=monkeypatch)
    writer.mark(FP.ASSEMBLING)
    assert writer.path.exists()
    writer.clear()
    writer.clear()
    assert not writer.path.exists()
    writer.mark(FP.ASSEMBLING)          # the dedupe memory went with the file
    assert writer.path.exists()


def test_t7_the_token_round_trip_and_its_refusals(tmp_path, monkeypatch):
    parent = _writer(tmp_path, monkeypatch=monkeypatch)
    env = parent.child_environment()
    assert list(env) == [FP.TOKEN_ENV]
    store = _Store(tmp_path, _session())
    monkeypatch.delenv(config.WORLD_FINISH_STAGES_ENV)     # the token carries the switch
    child = FP.writer_from_environment(store, "w1", SID, environ=env)
    assert child is not None and child.owner == parent.owner and child.epoch == EPOCH
    assert child.path == parent.path
    token = json.loads(env[FP.TOKEN_ENV])
    bad = [
        {}, {FP.TOKEN_ENV: ""}, {FP.TOKEN_ENV: "not json"}, {FP.TOKEN_ENV: "[]"},
        {FP.TOKEN_ENV: json.dumps(dict(token, format="wb-finish-phase/0"))},
        {FP.TOKEN_ENV: json.dumps(dict(token, world_id="w2"))},
        {FP.TOKEN_ENV: json.dumps(dict(token, session_id="x"))},
        {FP.TOKEN_ENV: json.dumps(dict(token, epoch=EPOCH + 1))},
        {FP.TOKEN_ENV: json.dumps(dict(token, epoch="now"))},
        {FP.TOKEN_ENV: json.dumps(dict(token, owner={"pid": 1}))},
    ]
    for environ in bad:
        assert FP.writer_from_environment(store, "w1", SID, environ=environ) is None, environ
    for session in (_session(state="complete"), _session(started_at=EPOCH + 1.0)):
        assert FP.writer_from_environment(_Store(tmp_path, session), "w1", SID, environ=env) is None
    assert FP.writer_from_environment(_Store(tmp_path, None), "w1", SID, environ=env) is None


def test_t7_phase_path_is_the_solve_workspace():
    from tower.world_builder import global_solve as GS
    from tower.world_builder.store import WorldStore

    store = WorldStore(Path("C:/nowhere/worlds"))
    assert FP.phase_path(store, "w1", SID) == GS.workspace_for(store, "w1", SID).root / FP.FILENAME


def test_t7_process_identity_and_process_is_running(processes):
    import os

    from tower.world_builder.store import process_identity, process_is_running

    me = process_identity(os.getpid())
    assert me["pid"] == os.getpid() and isinstance(me["created_at"], float)
    assert process_is_running(me) is True
    assert process_is_running(processes["dead"]) is False
    assert process_is_running(dict(me, created_at=me["created_at"] + 100.0)) is False   # recycled
    for bad in (None, [], {}, {"pid": os.getpid()}, {"pid": True, "created_at": 1.0},
                {"pid": "1", "created_at": 1.0}, {"pid": os.getpid(), "created_at": True},
                {"pid": os.getpid(), "created_at": float("nan")}, {"pid": -1, "created_at": 1.0}):
        assert process_is_running(bad) is False, bad


def test_t7_module_mark_is_a_no_op_without_an_installed_writer(tmp_path, monkeypatch):
    FP.mark(FP.PREPARING)               # nothing installed: nothing happens
    writer = _writer(tmp_path, monkeypatch=monkeypatch)
    token = FP.install(writer)
    try:
        FP.mark(FP.PLACING)
    finally:
        FP.uninstall(token)
    assert json.loads(writer.path.read_text(encoding="utf-8"))["stage"] == "placing"
    writer.clear()
    FP.mark(FP.MATCHING)
    assert not writer.path.exists()
    FP.uninstall(object())              # never raises


# ---------------------------------------------------------------------------
# T8 the builder
# ---------------------------------------------------------------------------

T8_STUB = r'''
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.getcwd())
argv = sys.argv[1:]
final = "--final" in argv
out = Path(os.environ["WB_FS_STUB_OUT"])
record = {"argv": argv, "env": dict(os.environ), "final": final}
if final:
    from tower.world_builder import finish_phase as FP
    from tower.world_builder.store import WorldStore
    store = WorldStore(Path(argv[argv.index("--root") + 1]))
    writer = FP.writer_from_environment(store, argv[argv.index("--world") + 1],
                                        argv[argv.index("--session") + 1])
    record["writer"] = writer is not None
    seen = []
    if writer is not None:
        doc = FP.read_phase(writer.path)
        seen.append([doc["stage"], doc.get("step"), doc["writer"] == doc["owner"]])
        for stage, step in ((FP.PREPARING, None), (FP.MATCHING, None), (FP.PLACING, None),
                            (FP.CHECKING, (1, 3)), (FP.CHECKING, (2, 3)), (FP.CHECKING, (3, 3))):
            writer.mark(stage, step)
            doc = FP.read_phase(writer.path)
            seen.append([doc["stage"], doc.get("step"), doc["writer"] == doc["owner"]])
    record["seen"] = seen
with out.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(record) + "\n")
if not final:
    # still running at Stop: the builder waits (`waiting`), then ends it
    time.sleep(float(os.environ.get("WB_FS_STUB_SLEEP", "0")))
print(json.dumps({"solved": False, "reason": "stub"}))
'''


def _t8_run(order, rendered, workdir, monkeypatch, *, on, background):
    import scripts.world_build_session as builder_script

    if on:
        _on(monkeypatch, basis=False)
    else:
        _off(monkeypatch)
    marks, at_release = [], []
    real_mark, real_release = FP.PhaseWriter.mark, builder_script.WorldBuilderEngine.release_world

    def mark(self, stage, step=None):
        marks.append([stage, list(step) if step else None])
        return real_mark(self, stage, step)

    def release(self, world_id):
        at_release.append(sorted(p.name for p in workdir.rglob(FP.FILENAME)))
        return real_release(self, world_id)

    monkeypatch.setattr(FP.PhaseWriter, "mark", mark)
    monkeypatch.setattr(builder_script.WorldBuilderEngine, "release_world", release)
    monkeypatch.setenv("WB_FS_STUB_SLEEP", "30" if background else "0")
    _run_builder(order, rendered, workdir, monkeypatch, solve_every=2 if background else 0,
                 stub_source=T8_STUB, extra_argv=("--solve-wait-seconds", "1"))
    calls = [json.loads(line) for line in
             (workdir / "stub_calls.jsonl").read_text(encoding="utf-8").splitlines()] \
        if (workdir / "stub_calls.jsonl").exists() else []
    return marks, at_release, calls, sorted(p.name for p in workdir.rglob(FP.FILENAME))


def test_t8_the_builder_records_its_finish_and_clears_it_before_the_lock(rendered, tmp_path,
                                                                        monkeypatch):
    marks, at_release, calls, left = _t8_run("soft", rendered, tmp_path / "on", monkeypatch,
                                             on=True, background=True)
    background = [c for c in calls if not c["final"]]
    (final,) = [c for c in calls if c["final"]]
    assert background, "no background solve ran at Stop: the test did not reach `waiting`"
    assert all(FP.TOKEN_ENV not in c["env"] for c in background), "a background solve got the token"
    assert FP.TOKEN_ENV in final["env"] and final["writer"] is True
    # parent: waiting (a background solve was running at Stop), preparing, assembling
    assert marks == [["waiting", None], ["preparing", None], ["assembling", None]]
    # the child saw the parent's `preparing`, then wrote its own stages as itself
    assert final["seen"][0] == ["preparing", None, True]
    assert [s[:2] for s in final["seen"][1:]] == [
        ["preparing", None], ["matching", None], ["placing", None],
        ["checking", {"n": 1, "of": 3}], ["checking", {"n": 2, "of": 3}],
        ["checking", {"n": 3, "of": 3}]]
    assert all(s[2] is False for s in final["seen"][1:]), "the child wrote as the owner"
    assert at_release == [[]], "the phase was still on disk when the lock was released"
    assert left == []

    # off: no token, no marks, the same argv
    marks_off, _, calls_off, left_off = _t8_run("soft", rendered, tmp_path / "off", monkeypatch,
                                                on=False, background=False)
    (final_off,) = [c for c in calls_off if c["final"]]
    assert marks_off == [] and left_off == [] and FP.TOKEN_ENV not in final_off["env"]
    assert final_off["seen"] == []

    def masked(argv, root):
        return [a.replace(str(root), "<WD>") for a in argv]

    assert masked(final_off["argv"], tmp_path / "off") == masked(final["argv"], tmp_path / "on")


@pytest.mark.parametrize("on", [False, True])
def test_t8_off_the_final_solve_is_called_exactly_as_before(tmp_path, monkeypatch, on):
    """Off, `run_final` is called with the keywords it always had -- a stand-in with the old
    signature (as `test_world_builder_finalization_notice.py` installs) still works. On, the
    token travels as `phase_env`. (The full suite caught the first version passing
    `phase_env=None` unconditionally.)"""
    import contextlib
    import io

    import scripts.world_build_session as wbs

    if on:
        _on(monkeypatch, basis=False)
    else:
        _off(monkeypatch)
    calls, marks = [], []
    real_mark = FP.PhaseWriter.mark
    monkeypatch.setattr(FP.PhaseWriter, "mark",
                        lambda self, stage, step=None: marks.append(stage) or real_mark(self, stage, step))
    monkeypatch.setattr(wbs, "prewarm_world_builder", lambda *a, **k: ())
    monkeypatch.setattr(wbs.StopRequest, "install", lambda self, **k: None)
    if on:
        monkeypatch.setattr(wbs.BackgroundSolver, "run_final",
                            lambda self, store, sources, should_stop=None, phase_env=None:
                            calls.append(phase_env) or {"attempted": True, "solved": False,
                                                        "reason": "stub"})
    else:
        monkeypatch.setattr(wbs.BackgroundSolver, "run_final",
                            lambda self, store, sources, should_stop=None:
                            calls.append("old") or {"attempted": True, "solved": False,
                                                    "reason": "stub"})
    with contextlib.redirect_stdout(io.StringIO()):
        code = wbs.main(["--synthetic", "--synthetic-frames", "10", "--root", str(tmp_path / "wb"),
                         "--solve", "--solve-every", "0", "--format", "json"])
    assert code == 0
    if on:
        (env,) = calls
        assert list(env) == [FP.TOKEN_ENV]
        # no background solve was running at Stop, so no `waiting`
        assert marks == ["preparing", "assembling"]
    else:
        assert calls == ["old"] and marks == []
    assert not list(tmp_path.rglob(FP.FILENAME))


def test_t8_a_hard_stop_says_assembling_and_clears(rendered, tmp_path, monkeypatch):
    marks, at_release, calls, left = _t8_run("hard", rendered, tmp_path / "hard", monkeypatch,
                                             on=True, background=False)
    assert marks == [["assembling", None]]
    assert [c for c in calls if c["final"]] == []
    assert at_release == [[]] and left == []


# ---------------------------------------------------------------------------
# T9 the solve's marks, T10 the re-gate writes nothing
# ---------------------------------------------------------------------------


class _Recording(FP.PhaseWriter):
    def __init__(self, path):
        super().__init__(path, world_id="w", session_id="s", epoch=EPOCH, owner=OWNER,
                         writer=CHILD)
        self.seen = []

    def mark(self, stage, step=None):
        self.seen.append((stage, tuple(step) if step else None))
        super().mark(stage, step)


@pytest.fixture
def no_links(monkeypatch):
    """`test_world_builder_global_solve_v10`'s autouse fixture, applied only where it is used."""
    from tower.world_builder import relocalizer

    monkeypatch.setattr(relocalizer, "revisit_pairs", lambda session_dir: [])
    monkeypatch.setenv("TOWER_WORLD_TRANSIENTS", "off")


def _solve_with(walk, writer, **kw):
    from tests.test_world_builder_global_solve_v10 import _finish

    token = FP.install(writer)
    try:
        return _finish(walk, **kw)
    finally:
        FP.uninstall(token)


from tests.test_world_builder_reproducible_finish import engines, walk  # noqa: E402,F401
from tests.test_world_builder_solve_consensus import SID as CONSENSUS_SID  # noqa: E402
from tests.test_world_builder_solve_consensus import world  # noqa: E402,F401
from tests.test_world_builder_solve_masks import colmap, session  # noqa: E402,F401


def test_t9a_t9b_a_consensus_of_three_marks_every_pass_and_a_frozen_solve_never_matches(
        walk, engines, colmap, no_links, tmp_path):
    from tower.world_builder import global_solve as GS

    writer = _Recording(tmp_path / FP.FILENAME)
    summary = _solve_with(walk, writer)
    assert summary["solved"]
    assert writer.seen == [("preparing", None), ("matching", None), ("placing", None),
                           ("checking", (1, 3)), ("checking", (2, 3)), ("checking", (3, 3))]
    again = _Recording(tmp_path / "again.json")
    summary = _solve_with(walk, again)
    published = json.loads(GS.workspace_for(walk.store, walk.world_id, walk.session_id)
                           .solution_path.read_text(encoding="utf-8"))
    assert published["solve"]["matching"] == GS.MATCHING_FROZEN, "the second solve did not freeze"
    assert ("matching", None) not in again.seen
    assert again.seen[0] == ("preparing", None) and again.seen[1] == ("placing", None)


def test_t9c_an_ungated_final_solve_has_no_checking(session, colmap, no_links, tmp_path):
    from tower.world_builder import global_solve as GS

    writer = _Recording(tmp_path / FP.FILENAME)
    token = FP.install(writer)
    try:
        summary = GS.solve(session.store, session.world_id, session.session_id, final=True,
                           gate=False, masks=False, seed=None, loop_detection=False)
    finally:
        FP.uninstall(token)
    assert summary["solved"]
    assert writer.seen == [("preparing", None), ("matching", None), ("placing", None)]
    # a background solve never marks, even with a writer installed
    writer.seen.clear()
    token = FP.install(writer)
    try:
        GS.solve(session.store, session.world_id, session.session_id, final=False)
    finally:
        FP.uninstall(token)
    assert writer.seen == []


def test_t9d_t9e_product_bytes_identical_with_and_without_a_writer(walk, engines, colmap, no_links,
                                                                  tmp_path, monkeypatch):
    import time as time_module

    from tower.world_builder import global_solve as GS
    from tower.world_builder.store import WorldStore

    monkeypatch.setattr(time_module, "time", lambda: 1234.5)
    monkeypatch.setattr(time_module, "perf_counter", lambda: 20.0)
    pristine = tmp_path / "pristine"
    shutil.copytree(walk.store.root, pristine)

    def run(name, with_writer):
        root = tmp_path / name
        shutil.copytree(pristine, root)
        store = WorldStore(root)
        copy = SimpleNamespace(store=store, world_id=walk.world_id, session_id=walk.session_id)
        writer = None
        if with_writer:
            writer = FP.PhaseWriter(FP.phase_path(store, walk.world_id, walk.session_id),
                                    world_id=walk.world_id, session_id=walk.session_id,
                                    epoch=EPOCH, owner=OWNER, writer=CHILD)
        summary = _solve_with(copy, writer)
        ws = GS.workspace_for(store, walk.world_id, walk.session_id).root
        # The mask filter's database copy is named per process and per call (a staging name):
        # the one value two identical solves cannot share.
        product = {p.name: re.sub(rb"database\.masked\.p\d+\.[0-9a-f]+\.db", b"<masked-db>",
                                  p.read_bytes())
                   for p in ws.iterdir() if p.is_file() and p.name in {
                       "solution.json", "components.json", "points.bin", "observations.bin"}}
        events = (store.session_dir(walk.world_id, walk.session_id) / "events.jsonl").read_bytes()
        return summary, product, events, (ws / FP.FILENAME).exists()

    run("warm", False)      # the depth stage's prediction cache is shared: both runs hit it
    off = run("without", False)
    on = run("with", True)
    assert off[0]["solved"] and on[0]["solved"]
    assert "solution.json" in off[1] and "components.json" in off[1]
    for name in off[1]:
        if on[1].get(name) != off[1][name]:
            x, y = on[1].get(name) or b"", off[1][name]
            at = next(i for i in range(min(len(x), len(y)) + 1) if x[i:i + 1] != y[i:i + 1])
            pytest.fail(f"{name} differs at {at}: {x[max(0, at - 300):at + 120]!r} vs "
                        f"{y[max(0, at - 300):at + 120]!r}")
    assert on[1] == off[1], "a finish phase changed what the solve published"
    assert on[2] == off[2]
    assert off[3] is False, "a solve with no writer installed wrote a phase"
    assert on[3] is True


def test_t10_the_finishers_re_gate_writes_nothing(world, tmp_path, monkeypatch):
    """`world_finish_pending.finish_regate` re-gates through `regate_published` (it holds no
    `finish_phase` call of its own): the consensus loop's marks reach no writer."""
    from tests.test_world_builder_consensus_v10 import _depth, _mapper, _published_store
    from tower.world_builder import coherence_publish as CP
    from tower.world_builder import global_solve as GS

    gate = {"state": CP.GATE_STATE_APPLIED, "retryable": True, "cause": CP.CAUSE_DEPTH_UNAVAILABLE,
            "attach": False, "consensus": {"state": CP.CONSENSUS_DEFERRED, "requested": 3,
                                           "seeds": [7, 8, 9]}}
    store, ws = _published_store(tmp_path, monkeypatch, world, gate, solve_seed=7)
    _depth(monkeypatch)
    monkeypatch.setattr(GS, "frozen_draw_mapper", _mapper([]))
    calls = []
    real = FP.mark
    monkeypatch.setattr(FP, "mark", lambda stage, step=None: calls.append((stage, step)) or real(stage, step))
    CP.regate_published(store, "w1", CONSENSUS_SID)
    assert calls, "the re-gate did not reach the consensus loop"
    assert not list(Path(tmp_path).rglob(FP.FILENAME))


# ---------------------------------------------------------------------------
# T11 basis
# ---------------------------------------------------------------------------


def _appearance(monkeypatch, manifest):
    from tower.results import world_builder_appearance as WBA

    def servable(store, world_id, session_id):
        if manifest is None:
            raise WBA.AppearanceNotServed("no appearance for this session")
        return world_id, manifest

    monkeypatch.setattr(WBA, "_servable_manifest", servable)


@pytest.mark.parametrize("manifest, expected", [
    ({"quality": "live"}, "walk"),
    ({"quality": "live", "proxy": {"source": {"surface_quality": "final"}}}, "walk"),
    ({"quality": "final", "proxy": {"source": {"surface_quality": "live"}}}, "walk"),
    ({"quality": "final", "proxy": {"source": {"surface_quality": "final"}}}, "final"),
    ({"quality": "final", "proxy": {"source": {}}}, "final"),
    ({"quality": "final"}, "final"),
    ({"quality": "final", "proxy": {"source": {"surface_quality": "other"}}}, None),
    ({}, None), ({"quality": None}, None), ({"quality": "draft"}, None), (None, None)])
def test_t11_appearance_basis(monkeypatch, manifest, expected):
    from tower.results.world_builder_appearance import appearance_basis
    from tower.results.world_builder_render import served_basis

    _appearance(monkeypatch, manifest)
    assert appearance_basis(object(), "w", "s") == expected
    assert served_basis(object(), "w", "s", "appearance") == expected


def test_t11_surface_and_record_basis(base_world, processes, tmp_path):
    from tower.results.world_builder_render import served_basis
    from tower.world_builder.store import WorldStore

    expected = {"open": "walk", "pending": "walk", "complete-solved": "final",
                "complete-skipped": "walk", "interrupted": "walk", "closed-no-finalization": None}
    for record, want in expected.items():
        root, w, s = _make_cell(base_world, processes, tmp_path / f"r{list(expected).index(record)}",
                                record, "none", "absent")
        store = WorldStore(root)
        assert served_basis(store, w, s, "sparse") == want, record
        assert served_basis(store, w, s, "dense") == want, record
        surface = store.world_dir(w) / "surface" / s / "manifest.json"
        assert served_basis(store, w, s, "surface") is None
        surface.parent.mkdir(parents=True, exist_ok=True)
        for params, value in (({"quality": "live"}, "walk"), ({"quality": "final"}, "final"),
                              ({}, None), ({"quality": 3}, None)):
            surface.write_text(json.dumps({"params": params}), encoding="utf-8")
            assert served_basis(store, w, s, "surface") == value, params
        surface.write_text("{not json", encoding="utf-8")
        assert served_basis(store, w, s, "surface") is None
    assert served_basis(WorldStore(tmp_path / "none"), "w", "s", "sparse") is None


def test_t11_basis_is_the_last_key_only_with_its_switch_and_never_in_the_revision(
        base_world, processes, tmp_path, monkeypatch):
    from tower.results.world_builder_render import build_area_revision, build_render_revision
    from tower.world_builder.store import WorldStore

    root, w, s = _make_cell(base_world, processes, tmp_path / "c", "complete-solved", "none", "absent")
    store = WorldStore(root)
    _off(monkeypatch)
    off = {v: build_render_revision(store, w, s, viewer=v) for v in (None, "appearance-1")}
    off_diag = build_render_revision(store, w, s, view="diagnostics")
    _on(monkeypatch, finish=False, basis=True)
    for viewer, body in off.items():
        on = build_render_revision(store, w, s, viewer=viewer)
        assert list(on)[-1] == "basis" and on["basis"] == "final"
        assert {k: v for k, v in on.items() if k != "basis"} == body
        assert list(on)[:-1] == list(body)
    on_diag = build_render_revision(store, w, s, view="diagnostics")
    assert on_diag["basis"] == "final" and on_diag["representation"] == "sparse"
    assert {k: v for k, v in on_diag.items() if k != "basis"} == off_diag
    # the area route is unchanged
    import inspect

    assert "basis" not in inspect.getsource(build_area_revision)
