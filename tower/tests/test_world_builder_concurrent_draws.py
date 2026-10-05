"""W0-1 concurrent consensus draws: selection, the private children, and every way a child can
fail. Every outcome must publish what the switch OFF publishes (Tier A); the real-child tests
map a small synthetic full-schema COLMAP database (no imagery) on CPU."""

import contextlib
import hashlib
import ctypes
import dataclasses
import json
import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from tower.config import world_solve_consensus_concurrent_setting
from tower.world_builder import global_solve as GS
from tower.world_builder import coherence_publish as CP
from tower.world_builder import stage_timing
from tower.world_builder.records import Keyframe
from tower.world_builder.store import WorldStore
from tests.test_world_builder_reproducible_finish import engines, walk  # noqa: F401
from tests.test_world_builder_solve_masks import colmap  # noqa: F401
from tests.test_world_builder_solve_consensus import (  # noqa: F401
    PIECES, SID, _candidate, _Store, world)

GIB = 1024 ** 3


_HELD = []


@pytest.fixture(autouse=True)
def _session_writer(tmp_path):
    """Every final solve holds its session's writer lock (`GS.session_writer_lock`, review
    W01F-FIX MED-2), and the concurrent mapper runs only under it. These tests drive the mapper
    directly, as such a solve does, so they hold it (`_hold`) for the sessions they use, until
    the test ends."""
    with contextlib.ExitStack() as stack:
        _HELD.append(stack)
        try:
            store = WorldStore(tmp_path)
            for world_id, session_id in (("w", "s"), ("w1", SID)):
                _hold(store, world_id, session_id)
            yield
        finally:
            _HELD.remove(stack)


def _hold(store, world_id, session_id):
    _HELD[-1].enter_context(GS.session_writer_lock(GS.workspace_for(store, world_id, session_id).root))


def _base():
    return GS.Solution(
        solver="glomap", solved_at=1.0, input_digest="input", keyframe_ids=[],
        poses={}, components=[], xyz=np.zeros((0, 3), np.float32),
        rgb=np.zeros((0, 3), np.uint8), component=np.zeros(0, np.int32),
        first_keyframe=np.zeros(0, np.int32), track_length=np.zeros(0, np.int32),
        error=np.zeros(0, np.float32), observations=np.zeros((0, 3), np.int32),
        camera=dict(fx=1, fy=1, cx=0, cy=0, width=1, height=1),
        solve={"seed": 3, "threads": 1}, timing={"map_s": 99},
        transients={"state": "applied"})


def _posed():
    """A child's candidate that posed something (an empty one is re-mapped: STD MED-1)."""
    candidate = _base()
    candidate.keyframe_ids = ["k0"]
    candidate.components = [{"index": 0, "images": 1, "images_supported": 1, "points": 0}]
    candidate.poses = {"k0": {"component": 0, "rotation": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
                              "translation": [0.0, 0.0, 0.0], "observations": 40}}
    return candidate


class _Process:
    pid = 0

    def __init__(self, code=0):
        self.code = code
        self.timeouts = []

    def wait(self, timeout=None):
        self.timeouts.append(timeout)
        return self.code

    def poll(self):
        return self.code


_ExitProcess = _Process


class _Job:
    def close(self):
        pass


def _journal(ws):
    path = ws.root / "consensus_concurrent.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _rig(tmp_path, monkeypatch, *, bad_copy=False, bad_child=False, tamper=False,
         empty=False, no_result=False, source_changes=False):
    store = WorldStore(tmp_path)
    _hold(store, "w", "s")
    ws = GS.workspace_for(store, "w", "s")
    ws.root.mkdir(parents=True)
    ws.database_path.write_bytes(b"db")
    launched = []
    serial = []
    sweeps = []
    events = []
    processes = {}

    def digest(path):
        data = Path(path).read_bytes()
        return {"content": hashlib.sha1(data).hexdigest()}

    copies = []

    def copy(source, destination):
        Path(destination).write_bytes(b"bad" if bad_copy else Path(source).read_bytes())
        copies.append(destination)
        if source_changes and len(copies) == 2:   # both copies made; then a writer changes it
            Path(source).write_bytes(b"db, written again by another writer")

    def launch(context, database, seed, sparse, output, expected_digest):
        launched.append((seed, Path(database)))
        content = digest(database)["content"]
        if not no_result:
            with Path(output).open("wb") as handle:
                candidate = _base() if empty else _posed()
                candidate.solve = {"child_seed": seed}
                pickle.dump(GS.draw_child_result(candidate, seed=seed, before=content,
                                                 after="bad" if bad_child else content,
                                                 map_ms=1250), handle)
        if tamper:   # the child reports an unchanged copy, but it changed (only the parent's own digest sees it)
            with Path(database).open("ab") as handle:
                handle.write(b"migrated")
        processes[seed] = _Process()
        return processes[seed], _Job()

    def frozen(*args, **kwargs):
        sweeps.append(kwargs.get("sweep", True))

        def draw(seed):
            serial.append(seed)
            return _base()
        return draw

    monkeypatch.setattr(GS, "database_digest", digest)
    monkeypatch.setattr(GS, "_private_draw_database", copy)
    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 32 * GIB)
    monkeypatch.setattr(GS, "_draw_free_commit", lambda: 32 * GIB)
    monkeypatch.setattr(GS, "frozen_draw_mapper", frozen)
    monkeypatch.setattr(stage_timing, "concurrent_draw_event", events.append)
    rig = types.SimpleNamespace(store=store, ws=ws, launched=launched, serial=serial,
                                sweeps=sweeps, events=events, processes=processes)
    return store, ws, launched, serial, events, rig


def _mapper(store, ws, seeds=(4, 5), **kwargs):
    kwargs.setdefault("keyframes", [])
    return GS.concurrent_draw_mapper(store, "w", "s", ws.database_path, _base(), seeds=seeds,
                                     **kwargs)


# ---------------------------------------------------------------------------
# selection: the switch, final solves only
# ---------------------------------------------------------------------------


def test_config_defaults_and_rejects_later_value(monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT", raising=False)
    assert world_solve_consensus_concurrent_setting() == "off"
    monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT", "after-draw-0")
    assert world_solve_consensus_concurrent_setting() == "after-draw-0"
    monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT", "at-freeze")
    assert world_solve_consensus_concurrent_setting() == "off"


def test_final_solve_selects_only_explicit_concurrent_mode(walk, engines, colmap, monkeypatch):
    from tests.test_world_builder_solve_masks import StubDetector

    chosen = []

    def mapper(label):
        def make(store, world_id, session_id, database_path, base, **kwargs):
            chosen.append(label)
            return lambda seed: base
        return make

    monkeypatch.setattr(GS, "frozen_draw_mapper", mapper("serial"))
    monkeypatch.setattr(GS, "concurrent_draw_mapper", mapper("concurrent"))
    for setting in ("off", "after-draw-0"):
        monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT", setting)
        GS.solve(walk.store, walk.world_id, walk.session_id, final=True, masks=True,
                 seed=0, gate=True, consensus=3, transient_backend_factory=StubDetector(),
                 mask_device_probe=lambda: None)
    # A gated, seeded consensus that is NOT the final solve never maps in children.
    GS.solve(walk.store, walk.world_id, walk.session_id, final=False, masks=True,
             seed=0, gate=True, consensus=3, transient_backend_factory=StubDetector(),
             mask_device_probe=lambda: None)
    assert chosen == ["serial", "concurrent", "serial"]


def test_switch_off_solve_leaves_no_concurrent_trace(walk, engines, colmap, monkeypatch):
    """OFF through the whole final solve: no I0 key, no journal, no private scratch (M03b)."""
    from tests.test_world_builder_solve_masks import StubDetector

    monkeypatch.setenv("TOWER_WORLD_STAGE_TIMING", "on")
    monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT", "off")
    GS.solve(walk.store, walk.world_id, walk.session_id, final=True, masks=True,
             seed=0, gate=True, consensus=3, transient_backend_factory=StubDetector(),
             mask_device_probe=lambda: None)
    timing = walk.store.session_dir(walk.world_id, walk.session_id) / stage_timing.FILENAME
    doc = json.loads(timing.read_text(encoding="utf-8"))
    assert doc["finishes"] and "consensus_concurrent" not in timing.read_text(encoding="utf-8")
    assert not (walk.workspace.root / "consensus_concurrent.jsonl").exists()
    assert not (walk.workspace.root / GS.CONSENSUS_SPARSE_DIRNAME).exists()


def test_switch_off_does_not_write_concurrent_i0_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_STAGE_TIMING", "on")
    monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT", "off")
    store = WorldStore(tmp_path)
    store.session_dir("w", "s").mkdir(parents=True)

    @stage_timing.timed("solve")
    def solve(store, world_id, session_id):
        return None

    solve(store, "w", "s")
    record = json.loads((store.session_dir("w", "s") /
                         stage_timing.FILENAME).read_text())["finishes"][0]["stages"]["solve"][0]
    assert "consensus_concurrent" not in record


def test_publish_draw_0_first_always_closes_the_mapper(tmp_path, monkeypatch):
    """`_publish_draw_0_first` closes a concurrent mapper after the consensus, and when it
    raises (A16: a mapper never closed leaves its children and scratch behind)."""
    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w", "s")
    for fail in (False, True):
        calls, closed = [], []

        def gate_and_publish(store, world_id, session_id, workspace, solution, **kw):
            calls.append(kw)
            if len(calls) == 1:
                return solution, {"state": CP.GATE_STATE_APPLIED, "attach": True}
            if fail:
                raise RuntimeError("the consensus gate failed")
            return solution, {}

        def draw(seed):
            return _base()

        draw.close = lambda: closed.append(1)
        monkeypatch.setattr(CP, "gate_and_publish", gate_and_publish)
        plan = types.SimpleNamespace(map_draw=draw)
        if fail:
            with pytest.raises(RuntimeError, match="consensus gate failed"):
                GS._publish_draw_0_first(store, "w", "s", ws, _base(), plan, final=True,
                                         gate=True, database_path=ws.database_path, keyframes=[])
        else:
            GS._publish_draw_0_first(store, "w", "s", ws, _base(), plan, final=True,
                                     gate=True, database_path=ws.database_path, keyframes=[])
        assert len(calls) == 2 and closed == [1]


# ---------------------------------------------------------------------------
# the concurrent path, and each child outcome
# ---------------------------------------------------------------------------


def test_lazy_private_launch_and_seed_order(tmp_path, monkeypatch):
    store, ws, launched, serial, events, _ = _rig(tmp_path, monkeypatch)
    mapper = _mapper(store, ws)
    assert launched == []  # draw 0 did not attach or consensus did not ask for draw 1
    first = mapper(4)
    assert first.solve["seed"] == 4 and first.timing["map_s"] == 1.25
    assert [seed for seed, _ in launched] == [4, 5]
    assert len({path for _, path in launched}) == 2
    assert all(path != ws.database_path for _, path in launched)
    # Each child's private root is `child-<k>`, never the serial sparse dir `seed-<k>`.
    assert [path.parent.name for _, path in launched] == ["child-4", "child-5"]
    root = ws.root / GS.CONSENSUS_SPARSE_DIRNAME
    # A collected child's private root is swept at once; the sibling's is still in use.
    assert not (root / "child-4").exists() and (root / "child-5").exists()
    assert mapper(5).solve["seed"] == 5
    assert serial == [] and events == ["after-draw-0"] and _journal(ws) == []
    mapper.close()
    assert not root.exists()


@pytest.mark.parametrize("cause,reason", [
    ("ram", "ram-refusal"),
    ("copy", "database-copy-digest-mismatch"),
    ("source", "database-source-digest-mismatch"),
    ("child", "database-child-digest-mismatch"),
    ("tamper", "database-child-digest-mismatch"),
    ("empty", "child-empty-candidate"),
    ("no-result", "child-result-FileNotFoundError"),
])
def test_resource_digest_or_result_refusal_uses_serial(tmp_path, monkeypatch, cause, reason):
    store, ws, launched, serial, events, rig = _rig(
        tmp_path, monkeypatch, bad_copy=cause == "copy", bad_child=cause == "child",
        tamper=cause == "tamper", empty=cause == "empty", no_result=cause == "no-result",
        source_changes=cause == "source")
    if cause == "ram":
        monkeypatch.setattr(GS, "_draw_free_ram", lambda: 0)
    mapper = _mapper(store, ws)
    assert mapper(4).solve["seed"] == 3  # sentinel from the unchanged serial mapper
    assert serial == [4]
    assert events[-1] == f"{reason}:seed-4"
    journal = _journal(ws)
    assert [(j["seed"], j["reason"], j["outcome"]) for j in journal] == [(4, reason, "serial")]
    at_launch = cause in ("ram", "copy", "source")
    assert len(launched) == (0 if at_launch else 2)
    if not at_launch:
        # Only that seed is re-mapped, beside its sibling, whose root is never swept under it.
        assert rig.sweeps == [False]
        assert not (ws.root / GS.CONSENSUS_SPARSE_DIRNAME / "child-4").exists()
        assert (ws.root / GS.CONSENSUS_SPARSE_DIRNAME / "child-5").exists()
        mapper(5)
        assert serial == [4, 5]   # the sibling failed the same way: re-mapped too
    mapper.close()
    assert not (ws.root / GS.CONSENSUS_SPARSE_DIRNAME).exists()


@pytest.mark.parametrize("stage", ["scratch-sweep", "resource-budget", "ram-probe", "commit-probe",
                                   "source-digest"])
def test_preflight_error_is_recorded_and_maps_serially(tmp_path, monkeypatch, stage):
    """Not an ownership hazard, so not fatal: OFF's mapper maps every seed (STD LOW-3)."""
    store, ws, launched, serial, _, rig = _rig(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        raise OSError("probe unavailable")

    target = {"scratch-sweep": "_sweep_draw_root", "resource-budget": "_draw_ram_needed",
              "ram-probe": "_draw_free_ram", "commit-probe": "_draw_free_commit",
              "source-digest": "database_digest"}[stage]
    monkeypatch.setattr(GS, target, fail)
    mapper = _mapper(store, ws, seeds=(8, 9))
    assert mapper(8).solve["seed"] == 3 and mapper(9).solve["seed"] == 3
    assert serial == [8, 9] and launched == [] and rig.sweeps == [True]
    journal = _journal(ws)
    assert [(j["reason"], j["seed"], j["outcome"]) for j in journal] == [
        (f"preflight-{stage}-OSError", 8, "serial")]
    mapper.close()


def test_ram_refusal_is_journaled_with_its_numbers(tmp_path, monkeypatch):
    store, ws, _, serial, _, _ = _rig(tmp_path, monkeypatch)
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 6 * GIB)
    mapper = _mapper(store, ws, keyframes=[None] * 997)
    mapper(4)
    mapper.close()
    (entry,) = _journal(ws)
    assert entry["reason"] == "ram-refusal" and entry["outcome"] == "serial"
    assert entry["free_bytes"] == 6 * GIB and entry["walk_size"] == 997 and entry["children"] == 2
    assert entry["need_bytes"] == GS._draw_ram_needed(997, 997, 2)
    assert serial == [4]


# ---------------------------------------------------------------------------
# the RAM gate and the wait: from the measured child
# ---------------------------------------------------------------------------


def test_ram_budget_is_the_measured_line_times_the_margin():
    measured_398 = int(1.09 * GIB)          # claude-W0-1-REVIEW-ADV-20260928 E2
    measured_997 = 1_506_500_608            # claude-W01F-6b41add-REVIEW-ADV-20261004 section 0
    assert abs(GS._draw_commit_budget(398, 398) - 1.5 * measured_398) <= 1
    assert abs(GS._draw_commit_budget(997, 997) - 1.5 * measured_997) <= 1
    # keyframes or images, whichever is larger, sizes the walk
    assert GS._draw_commit_budget(10, 997) == GS._draw_commit_budget(997, 10)
    assert GS._draw_commit_budget(997, 997) > GS._draw_commit_budget(398, 398)
    # the floor: an empty walk still costs the interpreter and the mapper's fixed state
    assert abs(GS._draw_commit_budget(0, 0) - 1.5 * 0.882 * GIB) < 0.001 * GIB
    # one budget per child and one for the parent
    assert GS._draw_ram_needed(997, 997, 2) == 3 * GS._draw_commit_budget(997, 997)
    assert round(GS._draw_ram_needed(997, 997, 2) / GIB, 2) == 6.31
    assert round(GS._draw_ram_needed(398, 398, 2) / GIB, 2) == 4.90


@pytest.mark.parametrize("free_gib,admitted", [
    (10.9, True),          # this host, Tower idle, walk 5: admitted (the old gate wanted 12.29)
    (6.4, True),
    (6.2, False),          # below 3 x 1.5 x 1.403 GiB = 6.31 GiB
    (3.0, False),
])
def test_ram_gate_admits_two_walk5_children_where_they_fit(tmp_path, monkeypatch, free_gib,
                                                          admitted):
    store, ws, launched, serial, events, _ = _rig(tmp_path, monkeypatch)
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: int(free_gib * GIB))
    mapper = _mapper(store, ws, keyframes=[None] * 997)
    mapper(4)
    mapper.close()
    assert (events == ["after-draw-0"]) is admitted
    assert len(launched) == (2 if admitted else 0)
    assert serial == ([] if admitted else [4])


def test_ram_gate_boundary_is_exact(tmp_path, monkeypatch):
    need = GS._draw_ram_needed(997, 997, 2)
    for free, admitted in ((need, True), (need - 1, False)):
        store, ws, launched, _, events, _ = _rig(tmp_path / str(free), monkeypatch)
        monkeypatch.setattr(GS, "_draw_free_ram", lambda: free)
        mapper = _mapper(store, ws, keyframes=[None] * 997)
        mapper(4)
        mapper.close()
        assert (len(launched) == 2) is admitted


def test_wait_bound_arithmetic():
    assert GS._draw_wait_seconds(0, 0) == 600.0                       # the floor
    assert GS._draw_wait_seconds(398, 398) == 600.0                   # 3 x 175 s = 525 s < 600
    assert abs(GS._draw_wait_seconds(997, 997) - 3 * 175.0 / 398 * 997) < 1e-9
    assert round(GS._draw_wait_seconds(997, 997)) == 1315             # 3.3x the 397 s measured
    assert GS._draw_wait_seconds(997, 997) > 3 * 397.2                # generous over the measured child
    assert GS._draw_wait_seconds(2000, 10) == GS._draw_wait_seconds(10, 2000)
    assert round(GS._draw_wait_seconds(2000, 2000)) == 2638


def test_child_wait_is_the_derived_bound_from_launch(tmp_path, monkeypatch):
    store, ws, _, _, _, rig = _rig(tmp_path, monkeypatch)
    mapper = _mapper(store, ws, keyframes=[None] * 997)
    started = time.monotonic()
    mapper(4)
    later = time.monotonic()
    (timeout,) = rig.processes[4].timeouts
    bound = GS._draw_wait_seconds(997, 997)
    assert timeout is not None and bound - (later - started) - 1 <= timeout <= bound
    mapper.close()


def test_child_wait_runs_from_each_childs_own_launch(tmp_path, monkeypatch):
    """W01F-FIX ADV LOW-4 (R04 survived): exactly, on a clock the test drives -- child 4
    launched (its launch returning) at 1100 s, child 5 at 1200 s, collected at 1200 s and
    1500 s."""
    store, ws, _, _, _, rig = _rig(tmp_path, monkeypatch)
    clock = {"now": 1000.0}
    monkeypatch.setattr(GS.time, "monotonic", lambda: clock["now"])
    real_launch = GS._launch_draw_child

    def launch(*args):
        launched = real_launch(*args)
        clock["now"] += 100.0
        return launched

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    mapper = _mapper(store, ws, keyframes=[None] * 997)
    bound = GS._draw_wait_seconds(997, 997)
    mapper(4)                                   # collected at 1200
    clock["now"] = 1500.0
    mapper(5)
    mapper.close()
    assert rig.processes[4].timeouts == [pytest.approx(bound - 100.0)]
    assert rig.processes[5].timeouts == [pytest.approx(bound - 300.0)]


def test_child_timeout_remaps_only_that_seed_and_keeps_its_sibling(tmp_path, monkeypatch):
    store, ws, launched, serial, events, _ = _rig(tmp_path, monkeypatch)
    stopped = []
    real_launch = GS._launch_draw_child

    class Blocking(_Process):
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("fake draw", timeout)

        def poll(self):
            return None

    def launch(context, database, seed, sparse, output, expected_digest):
        process, job = real_launch(context, database, seed, sparse, output, expected_digest)
        return (Blocking() if seed == 4 else process), job

    def stop(children, root=None):
        stopped.append(tuple(children))
        return True

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    monkeypatch.setattr(GS, "_stop_draw_children", stop)
    calls = []

    def should_stop():
        calls.append(1)
        return True

    mapper = _mapper(store, ws, should_stop=should_stop)
    assert mapper(4).solve["seed"] == 3
    assert serial == [4] and [seed for seed, _ in launched] == [4, 5]
    assert stopped == [(4,)] and events == ["after-draw-0", "child-timeout:seed-4"]
    (entry,) = _journal(ws)
    assert entry["reason"] == "child-timeout" and entry["wait_s"] == 600.0
    # The sibling kept mapping: its own candidate is draw 2.
    assert mapper(5).solve["seed"] == 5 and serial == [4]
    assert calls == []  # an ordinary soft stop does not kill a running draw
    mapper.close()


def test_soft_stop_during_successful_child_wait_keeps_child(tmp_path, monkeypatch):
    store, ws, launched, serial, events, _ = _rig(tmp_path, monkeypatch)
    stopped = []
    monkeypatch.setattr(GS, "_stop_draw_children",
                        lambda children, root=None: stopped.append(tuple(children)) or True)
    mapper = _mapper(store, ws, should_stop=lambda: True)
    assert mapper(4).solve["seed"] == 4
    assert serial == [] and events == ["after-draw-0"]
    assert stopped == []
    mapper.close()
    assert stopped == [(5,)]


def test_child_crash_remaps_same_seed_serially(tmp_path, monkeypatch):
    store, ws, launched, serial, events, _ = _rig(tmp_path, monkeypatch)
    monkeypatch.setattr(GS, "_launch_draw_child", lambda *args: (_Process(7), _Job()))
    mapper = _mapper(store, ws)
    assert mapper(4).solve["seed"] == 3
    assert serial == [4]
    assert events[-1] == "child-exit-7:seed-4"
    assert not (ws.root / "sparse-draws" / "child-4").exists()
    assert mapper(5).solve["seed"] == 3 and serial == [4, 5]
    journal = _journal(ws)
    assert [(j["seed"], j["reason"]) for j in journal] == [(4, "child-exit-7"), (5, "child-exit-7")]
    mapper.close()
    assert not (ws.root / "sparse-draws").exists()


def test_child_log_reaches_stderr_and_the_journal(tmp_path, monkeypatch, capsys):
    """The child's own lines are not swept with its private root (ADV LOW-3)."""
    store, ws, _, serial, _, _ = _rig(tmp_path, monkeypatch)

    def launch(context, database, seed, sparse, output, expected_digest):
        (Path(output).parent / "child.log").write_text(
            "Traceback (most recent call last):\nMemoryError: std::bad_alloc in seed %d\n" % seed)
        return _Process(1), _Job()

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    mapper = _mapper(store, ws, seeds=(4,))
    mapper(4)
    mapper.close()
    (entry,) = _journal(ws)
    assert entry["reason"] == "child-exit-1"
    assert "MemoryError: std::bad_alloc in seed 4" in entry["child_log_tail"]
    err = capsys.readouterr().err
    assert "[consensus draw seed 4: child log]" in err and "std::bad_alloc in seed 4" in err
    assert serial == [4]


@pytest.mark.parametrize("fault,reason", [
    ("wait", "child-wait-OSError"),
    ("job-close", "child-job-close-OSError"),
    ("pickle", "child-result-ModuleNotFoundError"),
    ("pickle-attr", "child-result-AttributeError"),
])
def test_postlaunch_child_error_remaps_that_seed(tmp_path, monkeypatch, fault, reason):
    """Brief item 1 / ADV LOW-5: an unreadable result, or a wait or Job error whose child is
    then confirmed gone, re-maps that seed; it does not abort the finish."""
    store, ws, _, serial, _, _ = _rig(tmp_path, monkeypatch)
    if fault == "wait":
        class BadWait(_Process):
            def wait(self, timeout=None):
                raise OSError("process handle unreadable")
        monkeypatch.setattr(GS, "_launch_draw_child", lambda *args: (BadWait(), _Job()))
    elif fault == "job-close":
        class FlakyJob(_Job):
            calls = 0

            def close(self):
                self.calls += 1
                if self.calls == 1:
                    raise OSError("job handle unreadable")
        monkeypatch.setattr(GS, "_launch_draw_child",
                            lambda *args: (_Process(), FlakyJob() if args[2] == 8 else _Job()))
    else:
        error = ModuleNotFoundError("old class") if fault == "pickle" else AttributeError("gone")
        monkeypatch.setattr(GS.pickle, "load", lambda *args: (_ for _ in ()).throw(error))
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                       _base(), seeds=(8, 9), keyframes=[])
    try:
        assert mapper(8).solve["seed"] == 3
    finally:
        mapper.close()
    (entry, *_) = _journal(ws)
    assert entry["reason"] == reason and entry["seed"] == 8 and entry["outcome"] == "serial"
    assert serial[0] == 8
    assert not (ws.root / "sparse-draws").exists()


def test_wait_error_whose_child_cannot_be_stopped_aborts(tmp_path, monkeypatch):
    from tower import process_ownership

    store, ws, _, serial, _, _ = _rig(tmp_path, monkeypatch)

    class Hung(_Process):
        def wait(self, timeout=None):
            raise OSError("process handle unreadable")

        def poll(self):
            return None

    monkeypatch.setattr(GS, "_launch_draw_child", lambda *args: (Hung(), _Job()))
    monkeypatch.setattr(process_ownership, "terminate_tree", lambda *args, **kwargs: False)
    before = len(GS._UNCONFIRMED_DRAW_CHILDREN)
    mapper = _mapper(store, ws)
    try:
        with pytest.raises(CP.ConsensusAuditError, match="could not be confirmed stopped"):
            mapper(4)
        with pytest.raises(CP.ConsensusAuditError):
            mapper.close()
        assert serial == []
        assert (ws.root / "sparse-draws").exists()   # never swept under an unconfirmed child
        assert [(j["reason"], j["outcome"]) for j in _journal(ws)] == [
            ("child-wait-OSError", "abort")]
    finally:
        del GS._UNCONFIRMED_DRAW_CHILDREN[before:]
        GS._LIVE_DRAW_ROOTS.discard(GS._draw_root_key(ws.root / "sparse-draws"))


def test_child_timing_failure_is_not_a_reason(tmp_path, monkeypatch):
    store, ws, _, serial, _, _ = _rig(tmp_path, monkeypatch)
    monkeypatch.setattr(stage_timing, "child_draw_timing",
                        lambda *args: (_ for _ in ()).throw(OSError("timing unavailable")))
    mapper = _mapper(store, ws)
    assert mapper(4).solve["seed"] == 4
    mapper.close()
    assert serial == [] and _journal(ws) == []


def test_seed_asked_twice_maps_serially_like_off(tmp_path, monkeypatch):
    store, ws, _, serial, _, _ = _rig(tmp_path, monkeypatch)
    mapper = _mapper(store, ws)
    assert mapper(4).solve["seed"] == 4
    assert mapper(4).solve["seed"] == 3
    assert [(j["reason"], j["outcome"]) for j in _journal(ws)] == [("child-missing", "serial")]
    assert serial == [4]
    mapper.close()
    assert not (ws.root / "sparse-draws").exists()


def test_consensus_record_takes_a_concurrent_draws_own_map_time(world):
    """ADV LOW-4: what the consensus waited for a child's draw is not its map."""
    def plan(declared):
        def draw(seed):
            candidate = _candidate(tuple(PIECES))
            candidate.timing = {"map_s": 100.0 + seed}
            return candidate
        if declared:
            draw.reports_map_s = True
        return CP.ConsensusPlan(draws=3, seed=7, map_draw=draw)

    def map_seconds(declared):
        result = CP.gate_by_consensus(_Store(), "w1", SID, _candidate(tuple(PIECES)),
                                      plan=plan(declared), database_path="db",
                                      keyframes=world.keyframes)
        return [d["map_s"] for d in result.record["consensus"]["draws"][1:]]

    assert map_seconds(True) == [108.0, 109.0]
    assert all(s < 100.0 for s in map_seconds(False))     # OFF: the call, as before


# ---------------------------------------------------------------------------
# the journal and the audit (fail closed only for hazards OFF cannot have)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cause", ["ram", "copy", "child"])
def test_unwritable_fallback_journal_prevents_serial_publish(tmp_path, monkeypatch, cause):
    store, ws, _, serial, events, _ = _rig(tmp_path, monkeypatch, bad_copy=cause == "copy")
    if cause == "ram":
        monkeypatch.setattr(GS, "_draw_free_ram", lambda: 0)
    elif cause == "child":
        monkeypatch.setattr(GS, "_launch_draw_child", lambda *args: (_Process(7), _Job()))

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(GS, "append_jsonl", fail_write)
    mapper = _mapper(store, ws)
    with pytest.raises(RuntimeError, match="fallback journal write failed"):
        mapper(4)
    assert serial == []
    assert any(word in events[-1] for word in ("refusal", "mismatch", "child-exit"))
    # A launch fallback sweeps its private copies BEFORE it journals, so even an unwritable
    # journal leaves none behind; one child's re-map leaves its sibling's root to close().
    assert (ws.root / "sparse-draws").exists() is (cause == "child")
    mapper.close()
    assert not (ws.root / "sparse-draws").exists()


@pytest.mark.parametrize("fault", ["launch", "job"])
def test_a_launch_fallback_stops_the_children_already_launched(tmp_path, monkeypatch, fault):
    if fault == "job" and os.name != "nt":
        pytest.skip("a Job is mandatory only on Windows")
    store, ws, launched, serial, _, rig = _rig(tmp_path, monkeypatch)
    real_launch = GS._launch_draw_child
    real_stop = GS._stop_draw_children
    stopped = []

    def launch(context, database, seed, sparse, output, expected_digest):
        if fault == "launch" and seed == 5:
            raise RuntimeError("the second child could not be spawned")
        process, job = real_launch(context, database, seed, sparse, output, expected_digest)
        return process, (None if fault == "job" and seed == 5 else job)

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    order = []
    monkeypatch.setattr(GS, "_stop_draw_children",
                        lambda children, root=None: stopped.append(sorted(children))
                        or order.append("stop") or real_stop(children, root))
    real_release = GS._release_draw_root
    monkeypatch.setattr(GS, "_release_draw_root",
                        lambda root: order.append("sweep") or real_release(root))
    mapper = _mapper(store, ws)
    assert mapper(4).solve["seed"] == 3
    assert stopped == [[4] if fault == "launch" else [4, 5]]   # at once, not at close()
    assert order == ["stop", "sweep"]          # never a root swept under a running child (R06)
    # Nothing of it still runs, so the seed maps with OFF's mapper exactly (it sweeps first).
    assert serial == [4] and rig.sweeps == [True]
    assert not (ws.root / "sparse-draws").exists()
    mapper.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows delete semantics")
def test_a_root_that_cannot_be_swept_loses_its_marker(tmp_path, monkeypatch):
    """`close()` is never fatal for a sweep, and a leftover it could not sweep must not read as
    a live writer's to the next finish (another process would see this pid alive)."""
    store, ws, _, serial, events, _ = _rig(tmp_path, monkeypatch)
    root = ws.root / "sparse-draws"
    mapper = _mapper(store, ws)
    mapper(4)
    mapper(5)
    held = open(root / "concurrent-context.pkl", "rb")
    try:
        mapper.close()
        assert root.exists() and not (root / "writer.json").exists()
        assert GS._draw_root_key(root) not in GS._LIVE_DRAW_ROOTS
        events.clear()
        again = _mapper(store, ws)
        again(4)                                   # still held: OFF's mapper, not an abort
        again.close()
        assert events[0].startswith("preflight-scratch-sweep-PermissionError") and serial == [4]
    finally:
        held.close()
    events.clear()
    last = _mapper(store, ws)
    last(4)
    last.close()
    assert events == ["after-draw-0"] and not root.exists()


def test_a_concurrent_draws_map_time_in_the_record_is_the_childs(world, tmp_path, monkeypatch):
    """ADV LOW-4 through the real concurrent mapper: draws[k].map_s is the child's map."""
    rig = _draw_rig(tmp_path, monkeypatch, map_ms=4321.0)
    on = GS.concurrent_draw_mapper(rig.store, "w1", SID, rig.ws.database_path, rig.base,
                                   seeds=(8, 9), keyframes=[])
    try:
        result = CP.gate_by_consensus(_Store(), "w1", SID, _cand(7),
                                      plan=CP.ConsensusPlan(draws=3, seed=7, map_draw=on),
                                      database_path="db", keyframes=world.keyframes)
    finally:
        on.close()
    assert rig.events == ["after-draw-0"]
    assert [d["map_s"] for d in result.record["consensus"]["draws"][1:]] == [4.321, 4.321]


def test_unwritable_fallback_journal_aborts_consensus(world, tmp_path, monkeypatch):
    store, ws, _, serial, _, _ = _rig(tmp_path, monkeypatch)
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 0)

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(GS, "append_jsonl", fail_write)
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                       _base(), seeds=(8, 9), keyframes=[])
    plan = CP.ConsensusPlan(draws=3, seed=7, map_draw=mapper)
    try:
        with pytest.raises(RuntimeError, match="fallback journal write failed"):
            CP.gate_by_consensus(_Store(), "w1", SID, _candidate(tuple(PIECES)), plan=plan,
                                 database_path="db", keyframes=world.keyframes)
    finally:
        mapper.close()
    assert serial == []


class _Alive(_Process):
    def poll(self):
        return None


def test_unconfirmed_prior_child_of_this_root_maps_every_seed_beside_it(world, tmp_path, monkeypatch):
    """W01F-FIX ADV MED-1, the ON side. An earlier ON finish of THIS session could not confirm
    a child stopped: it aborted the next finish. Now OFF's mapper maps every seed, without its
    sweep, so the child's `child-<k>/` is untouched and the consensus is OFF's."""
    store, ws, launched, serial, _, rig = _rig(tmp_path, monkeypatch)
    root = ws.root / GS.CONSENSUS_SPARSE_DIRNAME
    (root / "child-8").mkdir(parents=True)
    (root / "child-8" / "database.db").write_bytes(b"the unconfirmed child's")
    before = len(GS._UNCONFIRMED_DRAW_CHILDREN)
    GS._UNCONFIRMED_DRAW_CHILDREN.append((_Alive(), None, GS._draw_root_key(root)))
    try:
        mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                           _base(), seeds=(8, 9), keyframes=[])
        plan = CP.ConsensusPlan(draws=3, seed=7, map_draw=mapper)
        try:
            CP.gate_by_consensus(_Store(), "w1", SID, _candidate(tuple(PIECES)), plan=plan,
                                 database_path="db", keyframes=world.keyframes)
        finally:
            mapper.close()
        (entry,) = _journal(ws)
        assert (entry["reason"], entry["outcome"], entry["seed"]) == (
            "unconfirmed-prior-child", "serial", 8)
        assert serial == [8, 9] and launched == [] and rig.sweeps == [False, False]
        assert (root / "child-8" / "database.db").read_bytes() == b"the unconfirmed child's"
    finally:
        del GS._UNCONFIRMED_DRAW_CHILDREN[before:]


def test_another_roots_unconfirmed_child_does_not_hold_this_finish_back(tmp_path, monkeypatch):
    """Keyed by root (W01F-FIX ADV MED-1): another world's unstoppable child says nothing about
    this session's scratch, so this finish maps concurrently."""
    store, ws, launched, serial, events, _ = _rig(tmp_path, monkeypatch)
    before = len(GS._UNCONFIRMED_DRAW_CHILDREN)
    GS._UNCONFIRMED_DRAW_CHILDREN.append(
        (_Alive(), None, GS._draw_root_key(tmp_path / "another-world" / "sparse-draws")))
    try:
        mapper = _mapper(store, ws)
        assert mapper(4).solve["seed"] == 4
        mapper.close()
        assert events == ["after-draw-0"] and [s for s, _ in launched] == [4, 5] and serial == []
    finally:
        del GS._UNCONFIRMED_DRAW_CHILDREN[before:]


def test_off_mapper_is_the_base_even_while_an_unconfirmed_child_lives(tmp_path, monkeypatch):
    """W01F-FIX ADV MED-1 / Codex M4: OFF's mapper never reads the unconfirmed registry. With an
    earlier ON finish's child still alive -- of this root, of another, of none known -- it is made
    without raising and sweeps `sparse-draws/` exactly as the base's (44fbd13) does."""
    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w", "s")
    root = ws.root / GS.CONSENSUS_SPARSE_DIRNAME
    before = len(GS._UNCONFIRMED_DRAW_CHILDREN)
    for key in (GS._draw_root_key(root), "another-root"):
        GS._UNCONFIRMED_DRAW_CHILDREN.append((_Alive(), None, key))
    GS._UNCONFIRMED_DRAW_CHILDREN.append((_Alive(), None))
    try:
        (root / "seed-4" / "0").mkdir(parents=True)
        (root / "seed-4" / "0" / "cameras.bin").write_bytes(b"a killed serial draw's")
        mapper = GS.frozen_draw_mapper(store, "w", "s", ws.database_path, _base(), keyframes=[])
        assert callable(mapper) and not root.exists()
    finally:
        del GS._UNCONFIRMED_DRAW_CHILDREN[before:]


def test_unconfirmed_child_stop_is_visible_when_mapper_closes(tmp_path, monkeypatch):
    store, ws, _, _, _, _ = _rig(tmp_path, monkeypatch)
    mapper = _mapper(store, ws)
    mapper(4)  # a sibling is still owned by the mapper
    monkeypatch.setattr(GS, "_stop_draw_children", lambda children, root=None: False)
    with pytest.raises(CP.ConsensusAuditError, match="could not be confirmed stopped"):
        mapper.close()
    assert (ws.root / "sparse-draws").exists()  # do not delete an unconfirmed child's DB
    with pytest.raises(CP.ConsensusAuditError, match="could not be confirmed stopped"):
        mapper.close()
    GS._LIVE_DRAW_ROOTS.discard(GS._draw_root_key(ws.root / "sparse-draws"))


def test_journal_and_child_stop_failure_still_raise_audit_error(tmp_path, monkeypatch):
    store, ws, _, serial, _, _ = _rig(tmp_path, monkeypatch)
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 0)
    monkeypatch.setattr(GS, "append_jsonl",
                        lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(GS, "_stop_draw_children",
                        lambda children, root=None: (_ for _ in ()).throw(OSError("stop unavailable")))
    mapper = _mapper(store, ws)
    with pytest.raises(CP.ConsensusAuditError):
        mapper(4)
    assert serial == []


def test_child_poll_error_retains_unconfirmed_ownership(tmp_path, monkeypatch):
    from tower import process_ownership

    store, ws, _, _, _, _ = _rig(tmp_path, monkeypatch)
    original_launch = GS._launch_draw_child

    class BadPoll(_Process):
        def poll(self):
            raise OSError("process handle unreadable")

    def launch(*args):
        process, job = original_launch(*args)
        return (BadPoll() if args[2] == 5 else process), job

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    monkeypatch.setattr(process_ownership, "terminate_tree", lambda *args, **kwargs: False)
    mapper = _mapper(store, ws)
    mapper(4)
    before = len(GS._UNCONFIRMED_DRAW_CHILDREN)
    try:
        with pytest.raises(CP.ConsensusAuditError):
            mapper.close()
        assert len(GS._UNCONFIRMED_DRAW_CHILDREN) == before + 1
        assert (ws.root / "sparse-draws").exists()
    finally:
        del GS._UNCONFIRMED_DRAW_CHILDREN[before:]
        GS._LIVE_DRAW_ROOTS.discard(GS._draw_root_key(ws.root / "sparse-draws"))


def test_unconfirmed_new_child_aborts_without_serial_or_scratch_sweep(
        world, tmp_path, monkeypatch):
    store, ws, _, serial, _, _ = _rig(tmp_path, monkeypatch)

    class Alive(_Process):
        def poll(self):
            return None

    def launch(*args):
        GS._UNCONFIRMED_DRAW_CHILDREN.append((Alive(), None))
        raise RuntimeError("a consensus child could not be confirmed stopped")

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                       _base(), seeds=(8, 9), keyframes=[])
    plan = CP.ConsensusPlan(draws=3, seed=7, map_draw=mapper)
    before = len(GS._UNCONFIRMED_DRAW_CHILDREN)
    try:
        with pytest.raises(CP.ConsensusAuditError, match="launched consensus child"):
            CP.gate_by_consensus(_Store(), "w1", SID, _candidate(tuple(PIECES)), plan=plan,
                                 database_path="db", keyframes=world.keyframes)
        with pytest.raises(CP.ConsensusAuditError):
            mapper.close()
        assert len(GS._UNCONFIRMED_DRAW_CHILDREN) == before + 1
        assert (ws.root / "sparse-draws").exists()
        (entry,) = _journal(ws)
        assert entry["reason"] == "child-launch-unconfirmed" and entry["outcome"] == "abort"
        assert serial == []
    finally:
        del GS._UNCONFIRMED_DRAW_CHILDREN[before:]
        GS._LIVE_DRAW_ROOTS.discard(GS._draw_root_key(ws.root / "sparse-draws"))


@pytest.mark.parametrize("fault", ["assign-raises", "job-none", "resume-raises"])
def test_post_spawn_stop_uncertainty_retains_global_child(tmp_path, monkeypatch, fault):
    from tower import process_ownership

    class Spawned:
        pass

    spawned = Spawned()
    monkeypatch.setattr(GS.subprocess, "Popen", lambda *args, **kwargs: spawned)
    if fault == "assign-raises":
        monkeypatch.setattr(process_ownership, "assign_to_job",
                            lambda process: (_ for _ in ()).throw(RuntimeError("assign failed")))
        monkeypatch.setattr(process_ownership, "terminate_tree",
                            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("stop failed")))
    else:
        monkeypatch.setattr(process_ownership, "assign_to_job",
                            lambda process: None if fault == "job-none" else _Job())
        monkeypatch.setattr(process_ownership, "terminate_tree", lambda *args, **kwargs: False)
        if fault == "resume-raises":
            monkeypatch.setattr(GS, "_resume_draw_child",
                                lambda process: (_ for _ in ()).throw(OSError("resume failed")))
    before = len(GS._UNCONFIRMED_DRAW_CHILDREN)
    try:
        with pytest.raises(RuntimeError, match="could not be confirmed stopped"):
            GS._launch_draw_child(tmp_path / "context.pkl", tmp_path / "db", 4,
                                  tmp_path / "sparse", tmp_path / "candidate.pkl", "digest")
        assert len(GS._UNCONFIRMED_DRAW_CHILDREN) == before + 1
        assert GS._UNCONFIRMED_DRAW_CHILDREN[-1][0] is spawned
    finally:
        del GS._UNCONFIRMED_DRAW_CHILDREN[before:]


@pytest.mark.parametrize("fault", ["assign-runtime", "resume-oserror"])
def test_confirmed_post_spawn_stop_uses_recorded_serial(tmp_path, monkeypatch, fault):
    from tower import process_ownership

    original_launch = GS._launch_draw_child
    store, ws, _, serial, _, _ = _rig(tmp_path, monkeypatch)
    monkeypatch.setattr(GS, "_launch_draw_child", original_launch)
    monkeypatch.setattr(GS.subprocess, "Popen", lambda *args, **kwargs: _Process())
    stopped = []
    monkeypatch.setattr(process_ownership, "terminate_tree",
                        lambda *args, **kwargs: stopped.append(1) or True)
    if fault == "assign-runtime":
        monkeypatch.setattr(process_ownership, "assign_to_job",
                            lambda process: (_ for _ in ()).throw(RuntimeError("assign failed")))
    else:
        monkeypatch.setattr(process_ownership, "assign_to_job", lambda process: _Job())
        monkeypatch.setattr(GS, "_resume_draw_child",
                            lambda process: (_ for _ in ()).throw(OSError("resume failed")))
    before = len(GS._UNCONFIRMED_DRAW_CHILDREN)
    mapper = _mapper(store, ws)
    assert mapper(4).solve["seed"] == 3
    assert serial == [4] and stopped == [1]
    assert len(GS._UNCONFIRMED_DRAW_CHILDREN) == before
    assert not (ws.root / "sparse-draws").exists()
    (entry,) = _journal(ws)
    assert entry["reason"] == (
        "child-launch-RuntimeError" if fault == "assign-runtime"
        else "launch-refusal:OSError")
    mapper.close()


# ---------------------------------------------------------------------------
# draw scratch left behind: never fatal, never a live writer's (W01F STD MED-2 / ADV MED-1)
# ---------------------------------------------------------------------------


def _fake_map_candidate(candidate):
    """`_map_candidate`'s first two lines, verbatim, and then a given candidate."""
    def mapped(pycolmap, database_path, workspace, sparse_dir, keyframes, *, seed, **kwargs):
        shutil.rmtree(sparse_dir, ignore_errors=True)
        Path(sparse_dir).mkdir(parents=True)
        return candidate(seed)
    return mapped


_DRAWS = [(), ("A",), ()]


def _cand(seed):
    return _candidate(tuple(PIECES), rotated=_DRAWS[seed - 7])


def _publish(plan, world_fixture):
    result = CP.gate_by_consensus(_Store(), "w1", SID, _cand(7), plan=plan,
                                  database_path="db", keyframes=world_fixture.keyframes)
    consensus = result.record["consensus"]
    return (consensus["state"], (consensus.get("votes") or {}).get("draws"),
            sorted(k for k, p in result.solution.poses.items() if p["component"] == 0))


def _draw_rig(tmp_path, monkeypatch, map_ms=1.0):
    """The real mappers (OFF's and the concurrent one) over a fake `_map_candidate`; the
    concurrent children run in-process from the private copy the parent made."""
    store = WorldStore(tmp_path)
    _hold(store, "w1", SID)
    ws = GS.workspace_for(store, "w1", SID)
    ws.root.mkdir(parents=True)
    ws.database_path.write_bytes(b"db")
    (ws.root / "camera.json").write_text(json.dumps(dict(fx=1, fy=1, cx=0, cy=0, width=1, height=1)))
    monkeypatch.setattr(GS, "_map_candidate", _fake_map_candidate(_cand))
    monkeypatch.setattr(GS, "database_digest",
                        lambda p: {"content": hashlib.sha1(Path(p).read_bytes()).hexdigest()})
    monkeypatch.setattr(GS, "_private_draw_database",
                        lambda s, d: Path(d).write_bytes(Path(s).read_bytes()))
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 64 * GIB)
    monkeypatch.setattr(GS, "_draw_free_commit", lambda: 64 * GIB)
    events = []
    monkeypatch.setattr(stage_timing, "concurrent_draw_event", events.append)
    exits = {}

    def launch(context, database, seed, sparse, output, expected):
        with Path(output).open("wb") as handle:
            pickle.dump(GS.draw_child_result(_cand(seed), seed=seed, before=expected,
                                             after=expected, map_ms=map_ms), handle)
        return _Process(exits.get(seed, 0)), _Job()

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    base = _cand(7)
    base.solve = {"seed": 7}
    return types.SimpleNamespace(store=store, ws=ws, base=base, events=events, exits=exits,
                                 root=ws.root / GS.CONSENSUS_SPARSE_DIRNAME)


def _off(rig, world_fixture):
    off = GS.frozen_draw_mapper(rig.store, "w1", SID, rig.ws.database_path, rig.base, keyframes=[])
    return _publish(CP.ConsensusPlan(draws=3, seed=7, map_draw=off), world_fixture)


def _on(rig, world_fixture):
    on = GS.concurrent_draw_mapper(rig.store, "w1", SID, rig.ws.database_path, rig.base,
                                   seeds=(8, 9), keyframes=[])
    try:
        return _publish(CP.ConsensusPlan(draws=3, seed=7, map_draw=on), world_fixture)
    finally:
        on.close()


def test_markerless_leftover_of_a_killed_serial_draw_is_swept(world, tmp_path, monkeypatch):
    """A hard kill during a SERIAL draw (OFF, a re-gate, or ON's own fallback) leaves
    `sparse-draws/seed-k/` with no marker. ON sweeps it and maps concurrently, publishing
    exactly what OFF publishes (it used to raise and leave draw 0 deferred)."""
    rig = _draw_rig(tmp_path, monkeypatch)

    def leave_killed_serial_draw():
        (rig.root / "seed-8" / "0").mkdir(parents=True, exist_ok=True)
        (rig.root / "seed-8" / "0" / "cameras.bin").write_bytes(b"partial")

    leave_killed_serial_draw()
    off = _off(rig, world)
    assert off[0] == CP.CONSENSUS_APPLIED and off[1] == 3 and len(off[2]) == 90
    leave_killed_serial_draw()
    assert _on(rig, world) == off
    assert rig.events == ["after-draw-0"] and not rig.root.exists()
    assert not (rig.ws.root / "consensus_concurrent.jsonl").exists()


def test_stale_writers_are_swept_but_a_live_writers_root_is_never_touched(tmp_path, monkeypatch):
    import psutil

    store, ws, launched, serial, events, rig = _rig(tmp_path, monkeypatch)
    root = ws.root / "sparse-draws"
    stale = [
        {"pid": 99999999, "created_at": 1.0},                                # gone
        {"pid": os.getpid(), "created_at": psutil.Process().create_time()},  # this process, no open mapper
        None,                                                                # no marker at all
    ]
    for marker in stale:
        root.mkdir()
        if marker is not None:
            (root / "writer.json").write_text(json.dumps(marker))
        (root / "stale.bin").write_bytes(b"stale")
        launched.clear()
        events.clear()
        mapper = _mapper(store, ws, seeds=(4,))
        mapper(4)
        mapper.close()
        assert events == ["after-draw-0"] and len(launched) == 1, marker
        assert not root.exists()
    assert serial == []

    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        root.mkdir()
        (root / "writer.json").write_text(json.dumps(
            {"pid": other.pid, "created_at": psutil.Process(other.pid).create_time()}))
        (root / "live.bin").write_bytes(b"live")
        launched.clear()
        mapper = _mapper(store, ws, seeds=(4,))
        assert mapper(4).solve["seed"] == 3                       # OFF's mapper, beside it
        mapper.close()
        assert launched == [] and serial == [4] and rig.sweeps == [False]
        assert (root / "live.bin").read_bytes() == b"live"
        assert (root / "writer.json").exists()
        assert _journal(ws)[-1]["reason"] == "draw-scratch-owned"
    finally:
        other.kill()
        other.wait()


def test_scratch_made_after_the_sweep_is_mapped_beside_not_swept(tmp_path, monkeypatch):
    store, ws, _, serial, _, rig = _rig(tmp_path, monkeypatch)
    root = ws.root / "sparse-draws"
    root.mkdir()
    (root / "theirs.bin").write_bytes(b"theirs")
    monkeypatch.setattr(GS, "_sweep_draw_root", lambda *args, **kwargs: True)
    mapper = _mapper(store, ws, seeds=(8, 9))
    assert mapper(8).solve["seed"] == 3
    mapper.close()
    assert [(j["reason"], j["outcome"]) for j in _journal(ws)] == [("draw-scratch-owned", "serial")]
    assert serial == [8] and rig.sweeps == [False]
    assert (root / "theirs.bin").read_bytes() == b"theirs"


@pytest.mark.skipif(os.name != "nt", reason="Windows delete semantics")
def test_held_private_database_never_costs_a_vote_or_a_finish(world, tmp_path, monkeypatch):
    """ADV LOW-2 and MED-1's second origin. A handle held on a failed child's private copy (a
    scanner, an indexer) used to make the same-seed re-map's `mkdir` fail (2 voters, room 70)
    and then leave a markerless leftover that aborted the next finish. Now: OFF's result,
    the marker goes, the next finish maps serially while the handle is held and concurrently
    once it is released."""
    rig = _draw_rig(tmp_path, monkeypatch)
    off = _off(rig, world)
    assert off[0] == CP.CONSENSUS_APPLIED and off[1] == 3 and len(off[2]) == 90
    held = []
    real_launch = GS._launch_draw_child

    def launch(context, database, seed, sparse, output, expected):
        process, job = real_launch(context, database, seed, sparse, output, expected)
        if seed == 9:
            held.append(open(database, "rb"))
        return process, job

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    rig.exits[9] = 1
    try:
        assert _on(rig, world) == off
        # The root could not be swept, so its marker went: the leftover is stale, not "owned".
        assert (rig.root / "child-9" / "database.db").exists()
        assert not (rig.root / "writer.json").exists()
        rig.events.clear()
        rig.exits.clear()
        assert _on(rig, world) == off        # still held: the sweep fails, OFF's mapper maps
        assert rig.events[0] == "preflight-scratch-sweep-PermissionError:seed-8"
    finally:
        for handle in held:
            handle.close()
        monkeypatch.setattr(GS, "_launch_draw_child", real_launch)   # nothing holds it now
    rig.events.clear()
    assert _on(rig, world) == off
    assert rig.events == ["after-draw-0"] and not rig.root.exists()


def test_child_exit_preserves_three_voter_published_room(world, tmp_path, monkeypatch):
    rig = _draw_rig(tmp_path, monkeypatch)
    off = _off(rig, world)
    rig.exits[9] = 1
    concurrent = _on(rig, world)
    assert concurrent == off
    assert concurrent[0] == CP.CONSENSUS_APPLIED
    assert concurrent[1] == 3 and len(concurrent[2]) == 90
    assert not rig.root.exists()
    journal = (rig.ws.root / "consensus_concurrent.jsonl").read_text()
    assert "child-exit-1" in journal and '"seed": 9' in journal


# ---------------------------------------------------------------------------
# the solve: OFF against ON, byte for byte, in every outcome (Tier A)
# ---------------------------------------------------------------------------


def _world_files(walk):
    root = walk.store.world_dir(walk.world_id)
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_on_publishes_the_world_off_publishes_in_every_outcome(walk, engines, colmap, monkeypatch):
    """The REAL child script's `main()` runs in-process from the private copy (the fake
    pycolmap of these fixtures cannot cross a process), through `solve()` ->
    `_publish_draw_0_first` -> `gate_by_consensus`. Draw 1 detaches 25 room keyframes, so
    the vote really splits. Every file of the world must equal OFF's, but the ON journal."""
    import importlib.util
    from tower.world_builder import solve_masks as SM
    from tests.test_world_builder_solve_masks import StubDetector

    monkeypatch.setattr(time, "perf_counter", lambda: 100.0)
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)
    monkeypatch.setattr(SM, "new_masked_database_path",
                        lambda workspace: workspace.root / "database.masked.pX.fixed.db")
    base_map = GS._map_candidate

    def seeded_map(pycolmap, database_path, workspace, sparse_dir, keyframes, *, seed, **kw):
        try:   # the mapper call, as `_map_candidate` makes it: a failure is an empty model
            pycolmap.global_mapping(database_path, workspace.images_dir, sparse_dir)
        except Exception:  # noqa: BLE001
            pass
        sol = base_map(pycolmap, database_path, workspace, sparse_dir, keyframes, seed=seed, **kw)
        if seed == 1:
            for kid in sorted(sol.poses)[5:30]:
                sol.poses[kid] = dict(sol.poses[kid], observations=0)
        return sol

    monkeypatch.setattr(GS, "_map_candidate", seeded_map)
    script = Path(GS.__file__).resolve().parents[2] / "scripts" / "world_solve_draw.py"
    arm = {"exit": {}, "raises": set()}
    launched = []

    class Raising:
        def __init__(self, module):
            self._module = module

        def __getattr__(self, name):
            if name in ("global_mapping", "incremental_mapping"):
                def boom(*args, **kwargs):
                    raise MemoryError("std::bad_alloc")
                return boom
            return getattr(self._module, name)

    def launch(context, database, seed, sparse, output, expected):
        spec = importlib.util.spec_from_file_location("w01_world_solve_draw", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.global_solve = GS
        module.pycolmap = (Raising(sys.modules["pycolmap"]) if int(seed) in arm["raises"]
                           else sys.modules["pycolmap"])
        launched.append(int(seed))
        if int(seed) in arm["exit"]:
            return _Process(arm["exit"][int(seed)]), _Job()
        code = module.main(["--context", str(context), "--database", str(database),
                            "--seed", str(seed), "--sparse", str(sparse),
                            "--output", str(output), "--expected-digest", expected])
        return _Process(code), _Job()

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 64 * GIB)
    monkeypatch.setattr(GS, "_draw_free_commit", lambda: 64 * GIB)
    events = []
    real_event = stage_timing.concurrent_draw_event
    monkeypatch.setattr(stage_timing, "concurrent_draw_event",
                        lambda e: events.append(e) or real_event(e))
    journal = f"solve/{walk.session_id}/consensus_concurrent.jsonl"

    def finish(setting):
        monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT", setting)
        events.clear()
        launched.clear()
        GS.solve(walk.store, walk.world_id, walk.session_id, final=True, masks=True, seed=0,
                 gate=True, consensus=3, transient_backend_factory=StubDetector(),
                 mask_device_probe=lambda: None, input_digest="walk-digest")
        assert not (walk.workspace.root / GS.CONSENSUS_SPARSE_DIRNAME).exists()
        files = _world_files(walk)
        files.pop(journal, None)
        # The masked SQLite file's bytes differ OFF against OFF too; its content digest is
        # what the solve records, and solution.json carries it.
        return {k: v for k, v in files.items() if not k.endswith(".fixed.db")}

    for _ in range(3):
        finish("off")                    # the world's first finishes: it matches, then freezes
    off = finish("off")
    consensus = json.loads(off[f"solve/{walk.session_id}/solution.json"])["gate"]["consensus"]
    assert consensus["state"] == CP.CONSENSUS_APPLIED and consensus["votes"]["draws"] == 3
    outcomes = {}

    def why():
        path = walk.workspace.root / "consensus_concurrent.jsonl"
        return path.read_text(encoding="utf-8") if path.exists() else "no journal"

    outcomes["success"] = finish("after-draw-0")
    assert events == ["after-draw-0"] and launched == [1, 2], why()
    for seed in (1, 2):
        arm["exit"] = {seed: 1}
        outcomes[f"child {seed} exits 1"] = finish("after-draw-0")
        assert events[-1] == f"child-exit-1:seed-{seed}"
    arm["exit"] = {}
    arm["raises"] = {1}
    outcomes["child 1's mapper raises"] = finish("after-draw-0")
    assert events[-1] == f"child-exit-{GS.DRAW_CHILD_MAPPER_RAISED_EXIT}:seed-1"
    arm["raises"] = set()
    stale = walk.workspace.root / GS.CONSENSUS_SPARSE_DIRNAME / "seed-1" / "sparse" / "0"
    stale.mkdir(parents=True)
    (stale / "images.bin").write_bytes(b"half-written by a killed serial draw")
    outcomes["a markerless leftover"] = finish("after-draw-0")
    assert events == ["after-draw-0"], why()
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 0)
    outcomes["the RAM gate refuses"] = finish("after-draw-0")
    assert events == ["ram-refusal:seed-1"]
    outcomes["OFF again"] = finish("off")
    # An earlier ON finish in this interpreter could not confirm a child stopped (W01F-FIX ADV
    # MED-1 / Codex M4): OFF is the base's (it raised), and ON maps beside the child.
    before = len(GS._UNCONFIRMED_DRAW_CHILDREN)
    GS._UNCONFIRMED_DRAW_CHILDREN.append(
        (_Alive(), None, GS._draw_root_key(walk.workspace.root / GS.CONSENSUS_SPARSE_DIRNAME)))
    try:
        outcomes["OFF after an ON finish left an unconfirmed child"] = finish("off")
        outcomes["ON while that child lives"] = finish("after-draw-0")
        assert events == ["unconfirmed-prior-child:seed-1"]
    finally:
        del GS._UNCONFIRMED_DRAW_CHILDREN[before:]
    for name, files in outcomes.items():
        assert sorted(files) == sorted(off), name
        assert [k for k in off if files[k] != off[k]] == [], name
    reasons = [json.loads(line)["reason"] for line in
               (walk.workspace.root / "consensus_concurrent.jsonl").read_text().splitlines()]
    assert reasons == ["child-exit-1", "child-exit-1",
                       f"child-exit-{GS.DRAW_CHILD_MAPPER_RAISED_EXIT}", "ram-refusal",
                       "unconfirmed-prior-child"]


# ---------------------------------------------------------------------------
# real children: a synthetic full-schema COLMAP database GLOMAP maps (no imagery)
# ---------------------------------------------------------------------------


def _synthetic_walk(path, n_images=10, n_points=400, seed=0):
    """A noise-light synthetic scene: cameras on an arc looking at a point cloud, projected
    into keypoints, with matches and calibrated two-view geometries between near views.
    Written by pycolmap, so the schema is its own complete one and mapping leaves the
    content digest unchanged."""
    import pycolmap

    rng = np.random.default_rng(seed)
    width, height, focal = 640, 480, 500.0
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    points = rng.uniform([-2, -1.5, 4], [2, 1.5, 8], size=(n_points, 3))
    poses = []
    for i in range(n_images):
        a = np.deg2rad(-12 + 24 * i / (n_images - 1))
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        centre = np.array([-1.0 + 2.0 * i / (n_images - 1), 0.05 * np.sin(i), 0.0])
        poses.append((R, -R @ centre))
    names, seen = [], []
    with pycolmap.Database.open(str(path)) as db:
        camera = pycolmap.Camera(model="PINHOLE", width=width, height=height,
                                 params=[focal, focal, width / 2, height / 2])
        camera.camera_id = db.write_camera(camera)
        rig = pycolmap.Rig()
        rig.add_ref_sensor(camera.sensor_id)
        rig_id = db.write_rig(rig)
        image_ids = []
        for i, (R, t) in enumerate(poses):
            in_camera = (R @ points.T).T + t
            uv = (K @ in_camera.T).T
            uv = uv[:, :2] / uv[:, 2:3] + rng.normal(0, 0.3, (n_points, 2))
            visible = np.flatnonzero((in_camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                                     & (uv[:, 1] >= 0) & (uv[:, 1] < height))
            name = f"{i:08d}.jpg"
            image = pycolmap.Image(name=name, camera_id=camera.camera_id)
            image.image_id = db.write_image(image)
            frame = pycolmap.Frame()
            frame.rig_id = rig_id
            frame.add_data_id(image.data_id)
            db.write_frame(frame)
            db.write_keypoints(image.image_id, uv[visible].astype(np.float32))
            image_ids.append(image.image_id)
            names.append(name)
            seen.append({int(p): k for k, p in enumerate(visible)})
        for a in range(n_images):
            for b in range(a + 1, min(n_images, a + 4)):
                common = sorted(set(seen[a]) & set(seen[b]))
                matches = np.array([[seen[a][p], seen[b][p]] for p in common], dtype=np.uint32)
                db.write_matches(image_ids[a], image_ids[b], matches)
                (Ra, ta), (Rb, tb) = poses[a], poses[b]
                R = Rb @ Ra.T
                t = tb - R @ ta
                t = t / np.linalg.norm(t)
                E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R
                geometry = pycolmap.TwoViewGeometry()
                geometry.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
                geometry.inlier_matches = matches
                geometry.E = E
                geometry.F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
                geometry.cam2_from_cam1 = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
                db.write_two_view_geometry(image_ids[a], image_ids[b], geometry)
    return [Keyframe(keyframe_id=f"k{i}", session_id="s", source_seq=i, received_at=1000.0 + i,
                     image_relpath=f"images/{name}", width=width, height=height, byte_count=1)
            for i, name in enumerate(names)]


def _real_rig(tmp_path, monkeypatch):
    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w", "s")
    ws.root.mkdir(parents=True)
    ws.images_dir.mkdir(parents=True)
    keyframes = _synthetic_walk(ws.database_path)
    events = []
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 32 * GIB)
    monkeypatch.setattr(GS, "_draw_free_commit", lambda: 32 * GIB)
    monkeypatch.setattr(stage_timing, "concurrent_draw_event", events.append)
    return store, ws, keyframes, events


def _same_candidate(left, right):
    for field in dataclasses.fields(left):
        if field.name in ("solved_at", "timing"):
            continue
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape, field.name
            assert a.tobytes() == b.tobytes(), field.name
        else:
            assert a == b, field.name


@pytest.mark.slow
@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ownership")
def test_real_children_map_a_synthetic_walk_exactly_as_in_process(tmp_path, monkeypatch):
    import psutil
    from tower import process_ownership

    store, ws, keyframes, events = _real_rig(tmp_path, monkeypatch)
    source_before = ws.database_path.read_bytes()
    order = []
    assign = process_ownership.assign_to_job
    resume = GS._resume_draw_child

    def assigned(process):
        order.append(("assigned", psutil.Process(process.pid).status()))
        return assign(process)

    def resumed(process):
        order.append(("resumed", None))
        return resume(process)

    monkeypatch.setattr(process_ownership, "assign_to_job", assigned)
    monkeypatch.setattr(GS, "_resume_draw_child", resumed)
    base = _base()
    base.keyframe_ids = [k.keyframe_id for k in keyframes]
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path, base,
                                       seeds=(4, 5), keyframes=keyframes)
    try:
        concurrent = {4: mapper(4), 5: mapper(5)}
    finally:
        mapper.close()
    # The concurrent path ran: no fallback, no journal, nothing left.
    assert events == ["after-draw-0"]
    assert not (ws.root / "consensus_concurrent.jsonl").exists()
    assert not (ws.root / GS.CONSENSUS_SPARSE_DIRNAME).exists()
    # Each child ran no instruction outside its Job: suspended when assigned, then resumed.
    assert order == [("assigned", psutil.STATUS_STOPPED), ("resumed", None)] * 2
    assert ws.database_path.read_bytes() == source_before
    serial = GS.frozen_draw_mapper(store, "w", "s", ws.database_path, base, keyframes=keyframes)
    for seed, candidate in concurrent.items():
        assert candidate.solve["seed"] == seed
        assert len(candidate.poses) == 10 and len(candidate.xyz) > 100   # a real, non-empty map
        _same_candidate(serial(seed), candidate)


@pytest.mark.slow
def test_real_child_whose_mapper_raises_exits_23_and_the_seed_is_remapped(tmp_path, monkeypatch):
    """STD MED-1: a child's pycolmap raising (here MemoryError, as commit exhaustion does)
    used to be an empty candidate the parent accepted. Now the child exits 23 and the parent
    maps that seed itself, with OFF's mapper."""
    store, ws, keyframes, events = _real_rig(tmp_path, monkeypatch)
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "sitecustomize.py").write_text(
        "import pycolmap\n"
        "def boom(*args, **kwargs):\n"
        "    raise MemoryError('std::bad_alloc (injected)')\n"
        "pycolmap.global_mapping = boom\n"
        "pycolmap.incremental_mapping = boom\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(shim) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    base = _base()
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path, base,
                                       seeds=(4,), keyframes=keyframes)
    try:
        remapped = mapper(4)
    finally:
        mapper.close()
    assert events == ["after-draw-0", "child-exit-23:seed-4"]
    (entry,) = _journal(ws)
    assert entry["reason"] == "child-exit-23" and entry["outcome"] == "serial"
    assert "MemoryError" in entry["child_log_tail"]
    assert len(remapped.poses) == 10
    serial = GS.frozen_draw_mapper(store, "w", "s", ws.database_path, base, keyframes=keyframes)
    _same_candidate(serial(4), remapped)


def _run_child(tmp_path, ws, *, expected, shim_code=None, seed="17"):
    context = tmp_path / "context.pkl"
    with context.open("wb") as handle:
        pickle.dump((ws, [], GS.PinholeCamera.from_json_dict(_base().camera),
                     "input", GS.MIN_IMAGE_OBSERVATIONS), handle)
    path = [str(Path(GS.__file__).parents[2])]
    if shim_code is not None:
        shim = tmp_path / "shim"
        shim.mkdir()
        (shim / "sitecustomize.py").write_text(shim_code, encoding="utf-8")
        path.insert(0, str(shim))
    observed = tmp_path / "observed.json"
    env = dict(os.environ, W01_OBSERVED=str(observed), CUDA_VISIBLE_DEVICES="-1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join(path))
    output = tmp_path / "candidate.pkl"
    script = Path(GS.__file__).parents[2] / "scripts" / "world_solve_draw.py"
    result = subprocess.run([
        sys.executable, str(script), "--context", str(context),
        "--database", str(ws.database_path), "--seed", seed,
        "--sparse", str(tmp_path / "sparse"), "--output", str(output),
        "--expected-digest", expected], env=env, capture_output=True, text=True, timeout=60)
    return result, output, observed


_OBSERVING_SHIM = (
    "import json, os, sqlite3\n"
    "from pathlib import Path\n"
    "from tower.world_builder import global_solve as GS\n"
    "def mapped(*args, **kwargs):\n"
    "    Path(os.environ['W01_OBSERVED']).write_text(json.dumps("
    "{'seed': kwargs['seed'], 'threads': kwargs['threads']}))\n"
    "    db = sqlite3.connect(str(args[1]))\n"
    "    db.execute('create table after_probe (id integer)')\n"
    "    db.commit(); db.close()\n"
    "    return None\n"
    "GS._map_candidate = mapped\n")


def _child_ws(tmp_path):
    import pycolmap

    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w", "s")
    ws.root.mkdir(parents=True)
    ws.images_dir.mkdir(parents=True)
    with pycolmap.Database.open(str(ws.database_path)):
        pass
    return ws


@pytest.mark.slow
def test_child_script_uses_requested_seed_one_thread_and_after_digest(tmp_path):
    ws = _child_ws(tmp_path)
    expected = GS.database_digest(ws.database_path)["content"]
    result, output, observed = _run_child(tmp_path, ws, expected=expected,
                                          shim_code=_OBSERVING_SHIM)
    assert result.returncode == 0, result.stderr
    assert json.loads(observed.read_text()) == {"seed": 17, "threads": 1}
    with output.open("rb") as handle:
        child = pickle.load(handle)
    assert child["before"] == expected
    assert child["after"] != expected


@pytest.mark.slow
def test_child_script_never_maps_a_database_that_is_not_the_expected_one(tmp_path):
    ws = _child_ws(tmp_path)
    result, output, observed = _run_child(tmp_path, ws, expected="not-this-database",
                                          shim_code=_OBSERVING_SHIM)
    assert result.returncode == 0, result.stderr
    assert not observed.exists()                       # the mapper never ran
    with output.open("rb") as handle:
        child = pickle.load(handle)
    assert child["candidate"] is None and child["before"] != "not-this-database"


@pytest.mark.slow
def test_child_script_exits_23_when_the_mapper_raises(tmp_path):
    """23, not 3: a C-runtime abort() exits 3 too (review W01F-FIX ADV LOW-5)."""
    ws = _child_ws(tmp_path)
    expected = GS.database_digest(ws.database_path)["content"]
    result, output, _ = _run_child(tmp_path, ws, expected=expected, shim_code=(
        "import pycolmap\n"
        "def boom(*args, **kwargs):\n"
        "    raise MemoryError('std::bad_alloc (injected)')\n"
        "pycolmap.global_mapping = boom\n"
        "pycolmap.incremental_mapping = boom\n"))
    assert result.returncode == GS.DRAW_CHILD_MAPPER_RAISED_EXIT == 23
    assert "global_mapping: MemoryError" in result.stderr
    assert "incremental_mapping: MemoryError" in result.stderr
    assert not output.exists()


def test_private_copy_carries_a_wal_the_main_file_lacks(tmp_path):
    """The SQLite backup, not a file copy: a frozen database's WAL is part of it (M23)."""
    source = tmp_path / "walk.db"
    writer = sqlite3.connect(source)
    try:
        writer.execute("pragma journal_mode=wal")
        writer.execute("pragma wal_autocheckpoint=0")
        writer.execute("create table pairs (v integer)")
        writer.execute("insert into pairs values (42)")
        writer.commit()                      # in the WAL, not yet in the main file
        GS._private_draw_database(source, tmp_path / "private.db")
    finally:
        writer.close()
    copy = sqlite3.connect(tmp_path / "private.db")
    try:
        assert copy.execute("select v from pairs").fetchall() == [(42,)]
    finally:
        copy.close()


# ---------------------------------------------------------------------------
# process ownership: suspended launch, the Job, parent death, cancel
# ---------------------------------------------------------------------------


def test_parent_thread_counts_are_untouched(tmp_path, monkeypatch):
    import torch

    # This pinned NumPy bundles OpenBLAS under numpy.libs; query its actual pool,
    # since threadpoolctl is not installed in the Tower venv.
    blas = next((Path(np.__path__[0]).parent / "numpy.libs").glob("*openblas*.dll"))
    get_threads = ctypes.CDLL(str(blas)).scipy_openblas_get_num_threads64_
    get_threads.restype = ctypes.c_int

    store, ws, _, _, _, _ = _rig(tmp_path, monkeypatch)
    before = (torch.get_num_threads(), torch.get_num_interop_threads())
    blas_before = get_threads()
    numpy_pool = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    numpy_before = {name: os.environ.get(name) for name in numpy_pool}
    mapper = _mapper(store, ws)
    mapper(4)
    mapper.close()
    assert (torch.get_num_threads(), torch.get_num_interop_threads()) == before
    assert get_threads() == blas_before
    assert {name: os.environ.get(name) for name in numpy_pool} == numpy_before


def test_child_timing_joins_in_seed_order_and_fallback_stays_in_i0(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_STAGE_TIMING", "on")
    store = WorldStore(tmp_path)
    store.session_dir("w", "s").mkdir(parents=True)

    @stage_timing.timed("gate")
    def gate(store, world_id, session_id):
        stage_timing.child_draw_timing(4, 1200)
        stage_timing.child_draw_timing(5, 900)
        stage_timing.concurrent_draw_event("ram-refusal")

    @stage_timing.timed("solve")
    def solve(store, world_id, session_id):
        gate(store, world_id, session_id)

    solve(store, "w", "s")
    doc = json.loads((store.session_dir("w", "s") / stage_timing.FILENAME).read_text())
    stages = doc["finishes"][0]["stages"]
    assert [(d["seed"], d["draw_index"], d["wall_ms"]) for d in stages["solve_draw"]] == [
        (4, 0, 1200), (5, 1, 900)]
    assert stages["solve"][0]["consensus_concurrent"] == "ram-refusal"


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ownership")
def test_job_object_kills_draw_child_when_parent_exits(tmp_path):
    import psutil

    pid_file = tmp_path / "child-pid.txt"
    parent_code = (
        "import os, subprocess, sys\n"
        "from tower.process_ownership import assign_to_job\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)'])\n"
        "job = assign_to_job(child)\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid) + ' ' + str(job is not None))\n"
        "os._exit(0)\n"
    )
    subprocess.run([sys.executable, "-c", parent_code], check=True, timeout=5)
    pid, owned = pid_file.read_text().split()
    assert owned == "True"
    deadline = time.monotonic() + 3
    while psutil.pid_exists(int(pid)) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not psutil.pid_exists(int(pid))


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ownership")
def test_product_launcher_job_and_stop_end_real_child(tmp_path, monkeypatch):
    import psutil
    from tower import process_ownership

    monkeypatch.setattr(process_ownership, "interpreter_argv",
                        lambda *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    output = tmp_path / "seed" / "candidate.pkl"
    output.parent.mkdir()
    process, job = GS._launch_draw_child(tmp_path / "context", tmp_path / "db", 4,
                                         tmp_path / "sparse", output, "digest")
    try:
        assert job is not None and psutil.pid_exists(process.pid)
    finally:
        assert GS._stop_draw_children({4: (process, job)})
    assert process.poll() is not None


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ownership")
def test_child_is_suspended_until_its_job_is_assigned(tmp_path, monkeypatch):
    """The child runs no instruction outside the Job (A01/M07: launched not suspended)."""
    import psutil
    from tower import process_ownership

    states = []
    real_assign = process_ownership.assign_to_job

    def assign(process):
        states.append(psutil.Process(process.pid).status())
        return real_assign(process)

    monkeypatch.setattr(process_ownership, "assign_to_job", assign)
    monkeypatch.setattr(process_ownership, "interpreter_argv",
                        lambda *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    output = tmp_path / "seed" / "candidate.pkl"
    output.parent.mkdir()
    process, job = GS._launch_draw_child(tmp_path / "context", tmp_path / "db", 4,
                                         tmp_path / "sparse", output, "digest")
    try:
        running = psutil.Process(process.pid).status()
    finally:
        assert GS._stop_draw_children({4: (process, job)})
    assert states == [psutil.STATUS_STOPPED]
    assert running != psutil.STATUS_STOPPED


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ownership")
def test_parent_hard_exit_kills_launched_child_and_grandchild(tmp_path):
    """Through the product launcher: a parent that dies with no cleanup at all (a hard stop)
    takes the draw child and its own child with it (STD LOW-6)."""
    import psutil

    pids = (tmp_path / "pids.json").as_posix()
    grand_txt = (tmp_path / "grand.txt").as_posix()
    seed_out = (tmp_path / "seed" / "candidate.pkl").as_posix()
    grand = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; "
             f"time.sleep(60)']); open({grand_txt!r},'w').write(str(p.pid)); time.sleep(60)")
    code = (
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "from tower import process_ownership\n"
        "from tower.world_builder import global_solve as GS\n"
        f"grand = {grand!r}\n"
        "process_ownership.interpreter_argv = lambda *a: [sys.executable, '-c', grand]\n"
        f"out = Path({seed_out!r})\n"
        "out.parent.mkdir()\n"
        "p, job = GS._launch_draw_child(Path('c'), Path('d'), 4, Path('s'), out, 'x')\n"
        "deadline = time.time() + 10\n"
        f"while not Path({grand_txt!r}).exists() and time.time() < deadline: time.sleep(0.05)\n"
        f"open({pids!r}, 'w').write(json.dumps([p.pid, job is not None]))\n"
        "os._exit(0)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30, env=dict(os.environ))
    child, owned = json.loads(Path(pids).read_text())
    grandchild = int((tmp_path / "grand.txt").read_text())
    assert owned
    deadline = time.monotonic() + 5
    while ((psutil.pid_exists(child) or psutil.pid_exists(grandchild))
           and time.monotonic() < deadline):
        time.sleep(0.05)
    assert not psutil.pid_exists(child) and not psutil.pid_exists(grandchild)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ownership")
def test_cancel_during_the_wait_kills_real_children_and_sweeps(tmp_path, monkeypatch):
    """A console event while the parent waits is not swallowed, and `close()` (as
    `_publish_draw_0_first`'s finally calls it) kills both real children and sweeps."""
    import psutil
    from tower import process_ownership

    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w", "s")
    ws.root.mkdir(parents=True)
    ws.database_path.write_bytes(b"db")
    monkeypatch.setattr(process_ownership, "interpreter_argv",
                        lambda *args: [sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(GS, "database_digest",
                        lambda p: {"content": hashlib.sha1(Path(p).read_bytes()).hexdigest()})
    monkeypatch.setattr(GS, "_private_draw_database",
                        lambda s, d: Path(d).write_bytes(Path(s).read_bytes()))
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 64 * GIB)
    monkeypatch.setattr(GS, "_draw_free_commit", lambda: 64 * GIB)
    pids = []
    fired = []
    real_launch = GS._launch_draw_child

    def launch(*args):
        process, job = real_launch(*args)
        pids.append(process.pid)
        real_wait = process.wait

        def wait(timeout=None):
            if not fired:
                fired.append(1)
                raise KeyboardInterrupt("console control event")
            return real_wait(timeout)
        process.wait = wait
        return process, job

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    mapper = _mapper(store, ws)
    with pytest.raises(KeyboardInterrupt):
        try:
            mapper(4)
        finally:
            mapper.close()
    deadline = time.monotonic() + 5
    while any(psutil.pid_exists(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(pids) == 2 and not any(psutil.pid_exists(p) for p in pids)
    assert not (ws.root / "sparse-draws").exists()


# ---------------------------------------------------------------------------
# W01F-FIX4: the session writer lock (MED-2 / Codex H1, H2)
# ---------------------------------------------------------------------------


def _record_unlocked_solve(monkeypatch, seen, store):
    def unlocked(store_, world_id, session_id, **options):
        seen.append((world_id, session_id, options.get("final", False),
                     GS.session_writer_lock_held(GS.workspace_for(store_, world_id, session_id).root)))
        return {"solved": False}

    monkeypatch.setattr(GS, "_solve_unlocked", unlocked)


def test_every_final_solve_holds_its_sessions_writer_lock(tmp_path, monkeypatch):
    """`solve(final=True)` -- and any solve asked for a consensus -- runs under the lock; a
    background solve does not take it, and the lock is gone once the solve returns."""
    store = WorldStore(tmp_path)
    seen = []
    _record_unlocked_solve(monkeypatch, seen, store)
    GS.solve(store, "w7", "s7", final=True)
    GS.solve(store, "w7", "s7", final=False, consensus=3)
    GS.solve(store, "w7", "s7", final=False)
    assert seen == [("w7", "s7", True, True), ("w7", "s7", False, True), ("w7", "s7", False, False)]
    assert not GS.session_writer_lock_held(GS.workspace_for(store, "w7", "s7").root)


def test_world_solve_script_and_the_builders_final_solve_take_the_lock(tmp_path, monkeypatch):
    """The entry points the review named: a hand-run `world_solve.py --final` (it took no lock
    at all) and the live builder's in-process `solve_session`. The finisher and `world_finalize.py`
    call the same `global_solve.solve(final=True)`."""
    import importlib.util

    store = WorldStore(tmp_path)
    seen = []
    _record_unlocked_solve(monkeypatch, seen, store)
    scripts = Path(GS.__file__).resolve().parents[2] / "scripts"
    spec = importlib.util.spec_from_file_location("fix4_world_solve", scripts / "world_solve.py")
    world_solve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(world_solve)
    monkeypatch.setattr(world_solve, "global_solve", GS)
    assert world_solve.main(["--root", str(tmp_path), "--world", "w7", "--session", "s7",
                             "--final"]) == 0
    assert world_solve.main(["--root", str(tmp_path), "--world", "w7", "--session", "s7"]) == 0
    spec = importlib.util.spec_from_file_location("fix4_build_session", scripts / "world_build_session.py")
    build_session = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_session)
    monkeypatch.setattr(GS, "write_sources", lambda *args, **kwargs: None)
    build_session.solve_session(store, "w7", "s8", capture_dirs=[], sources={})
    assert [(w, s, final, held) for w, s, final, held in seen] == [
        ("w7", "s7", True, True), ("w7", "s7", False, False), ("w7", "s8", True, True)]


def test_a_regate_in_place_holds_the_session_writer_lock(tmp_path, monkeypatch):
    store = WorldStore(tmp_path)
    seen = []

    def regate(store_, world_id, session_id, **kwargs):
        seen.append(GS.session_writer_lock_held(GS.workspace_for(store_, world_id, session_id).root))
        return {"regated": True}

    monkeypatch.setattr(CP, "_regate_published", regate)
    assert CP.regate_published(store, "w7", "s7") == {"regated": True}
    assert seen == [True]
    assert not GS.session_writer_lock_held(GS.workspace_for(store, "w7", "s7").root)


_HOLDER = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "from tower.world_builder import global_solve as GS\n"
    "with GS.session_writer_lock(Path(sys.argv[1])):\n"
    "    print('held', flush=True)\n"
    "    release = Path(sys.argv[2])\n"
    "    while not release.exists():\n"
    "        time.sleep(0.05)\n")


def _hold_in_another_process(tmp_path, root):
    release = tmp_path / "release-the-lock"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1",
               PYTHONPATH=str(Path(GS.__file__).resolve().parents[2]))
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(root), str(release)],
                              stdout=subprocess.PIPE, text=True, env=env)
    assert holder.stdout.readline().strip() == "held"
    return holder, release


def _enter_in_thread(fn):
    import threading

    entered = threading.Event()
    done = threading.Event()
    errors = []

    def run():
        try:
            fn(entered)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, entered, done, errors


def test_a_second_final_solve_of_the_session_waits_for_the_first(tmp_path, monkeypatch):
    """Codex H1/H2's precondition, removed: run B (a final solve here) cannot start while run A
    (another process: a builder's final solve, a hand-run `world_solve.py --final`) holds the
    session, so it can never map in, sweep or close A's live `sparse-draws/`. Another session
    of the same world is not held back."""
    store = WorldStore(tmp_path)
    monkeypatch.setattr(GS, "_SESSION_LOCK_POLL_S", 0.05)
    root = GS.workspace_for(store, "w7", "s7").root
    holder, release = _hold_in_another_process(tmp_path, root)
    try:
        def final_solve(world_id, session_id):
            def run(entered):
                monkeypatch.setattr(GS, "_solve_unlocked",
                                    lambda *args, **kwargs: entered.set() or {"solved": False})
                GS.solve(store, world_id, session_id, final=True)
            return run

        other = _enter_in_thread(final_solve("w7", "another-session"))
        assert other[1].wait(10)                        # another session: not held back
        other[0].join(10)
        thread, entered, done, errors = _enter_in_thread(final_solve("w7", "s7"))
        assert not entered.wait(1.5)                    # A holds the session: B waits
        release.write_text("go")
        assert holder.wait(10) == 0
        assert entered.wait(10) and done.wait(10) and errors == []
    finally:
        release.write_text("go")
        if holder.poll() is None:
            holder.kill()
        holder.wait()


def test_a_killed_holder_never_strands_the_session(tmp_path, monkeypatch):
    """The builder hard-kills a solve child that outstays its stop budget: the OS frees its
    lock, so the next final solve of the session is not left waiting on a dead process."""
    store = WorldStore(tmp_path)
    monkeypatch.setattr(GS, "_SESSION_LOCK_POLL_S", 0.05)
    root = GS.workspace_for(store, "w7", "s7").root
    holder, release = _hold_in_another_process(tmp_path, root)
    holder.kill()
    holder.wait()

    def take(entered):
        with GS.session_writer_lock(root):
            entered.set()

    thread, entered, done, errors = _enter_in_thread(take)
    assert entered.wait(10) and done.wait(10) and errors == []


def test_another_thread_of_this_process_waits_too_and_the_lock_is_reentrant(tmp_path, monkeypatch):
    store = WorldStore(tmp_path)
    monkeypatch.setattr(GS, "_SESSION_LOCK_POLL_S", 0.05)
    root = GS.workspace_for(store, "w7", "s7").root

    def take(entered):
        with GS.session_writer_lock(root):
            entered.set()

    with GS.session_writer_lock(root):
        with GS.session_writer_lock(root):              # a re-gate inside this thread's solve
            assert GS.session_writer_lock_held(root)
        assert GS.session_writer_lock_held(root)
        thread, entered, done, errors = _enter_in_thread(take)
        assert not entered.wait(1.0)
    assert entered.wait(10) and done.wait(10) and errors == []
    assert not GS.session_writer_lock_held(root)


def test_without_the_session_lock_the_concurrent_mapper_is_offs(tmp_path, monkeypatch):
    """W01F-FIX MED-2's minimum: outside the session writer lock nothing of W0-1 runs -- no
    child, no marker, no ON sweep: OFF's own mapper, which sweeps as the base does, maps every
    seed. (The lock is this thread's; another thread does not hold it.)"""
    store, ws, launched, serial, events, rig = _rig(tmp_path, monkeypatch)
    results = []

    def run(entered):
        mapper = _mapper(store, ws)
        results.extend([mapper(4).solve["seed"], mapper(5).solve["seed"]])
        mapper.close()
        entered.set()

    thread, entered, done, errors = _enter_in_thread(run)
    assert done.wait(30) and errors == []
    assert results == [3, 3] and serial == [4, 5] and launched == [] and rig.sweeps == [True]
    assert events == ["session-lock-not-held:seed-4"]
    (entry,) = _journal(ws)
    assert (entry["reason"], entry["outcome"]) == ("session-lock-not-held", "serial")


# ---------------------------------------------------------------------------
# W01F-FIX4: the commit gate (Codex H3 / ADV LOW-2)
# ---------------------------------------------------------------------------


def test_commit_refusal_is_journaled_with_its_numbers(tmp_path, monkeypatch):
    """Ample free RAM, too little commit: the children would fail to allocate where a serial
    OFF draw fits. Refused, and every seed maps with OFF's mapper."""
    store, ws, launched, serial, events, _ = _rig(tmp_path, monkeypatch)
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 64 * GIB)
    need = GS._draw_ram_needed(997, 997, 2)
    monkeypatch.setattr(GS, "_draw_free_commit", lambda: need - 1)
    mapper = _mapper(store, ws, keyframes=[None] * 997)
    assert mapper(4).solve["seed"] == 3 and mapper(5).solve["seed"] == 3
    mapper.close()
    assert launched == [] and serial == [4, 5] and events == ["commit-refusal:seed-4"]
    (entry,) = _journal(ws)
    assert entry["reason"] == "commit-refusal" and entry["outcome"] == "serial"
    assert (entry["free_commit_bytes"], entry["need_bytes"], entry["walk_size"],
            entry["children"]) == (need - 1, need, 997, 2)
    monkeypatch.setattr(GS, "_draw_free_commit", lambda: need)          # exactly enough
    events.clear()
    again = _mapper(store, ws, keyframes=[None] * 997)
    again(4)
    again.close()
    assert events == ["after-draw-0"]


@pytest.mark.skipif(os.name != "nt", reason="GlobalMemoryStatusEx")
def test_free_commit_is_the_systems_commit_headroom():
    import psutil

    commit = GS._draw_free_commit()
    assert isinstance(commit, int) and commit > 0
    # The commit limit is physical memory plus the page files: its headroom is at most that.
    assert commit <= psutil.virtual_memory().total + psutil.swap_memory().total + GIB


# ---------------------------------------------------------------------------
# W01F-FIX4: a partial child Solution never votes (Codex M5)
# ---------------------------------------------------------------------------


def _partial(fault, candidate, result):
    if fault == "not-complete":
        result.pop("complete")
    elif fault == "complete-false":
        result["complete"] = False
    elif fault == "another-seed":
        result["seed"] = 99
    elif fault == "another-walk":
        candidate.input_digest = "another walk"
        result["fingerprint"] = GS._candidate_fingerprint(candidate)
    elif fault == "changed-after-attest":
        candidate.poses = dict(candidate.poses, k1=dict(candidate.poses["k0"]))
        candidate.keyframe_ids = ["k0", "k1"]
    elif fault == "pose-off-the-walk":
        candidate.poses = dict(candidate.poses, stranger=dict(candidate.poses["k0"]))
        result["fingerprint"] = GS._candidate_fingerprint(candidate)
    elif fault == "missing-component":
        candidate.poses = {"k0": dict(candidate.poses["k0"], component=1)}
        result["fingerprint"] = GS._candidate_fingerprint(candidate)
    elif fault == "torn-arrays":
        candidate.xyz = np.zeros((3, 3), np.float32)
        result["fingerprint"] = GS._candidate_fingerprint(candidate)
    elif fault == "unknown-solver":
        candidate.solver = "half-a-mapper"
        result["fingerprint"] = GS._candidate_fingerprint(candidate)


@pytest.mark.parametrize("fault,reason", [
    ("not-complete", "child-incomplete"),
    ("complete-false", "child-incomplete"),
    ("another-seed", "child-seed-mismatch"),
    ("another-walk", "child-candidate-walk"),
    ("changed-after-attest", "child-candidate-fingerprint"),
    ("pose-off-the-walk", "child-candidate-poses"),
    ("missing-component", "child-candidate-components"),
    ("torn-arrays", "child-candidate-arrays"),
    ("unknown-solver", "child-candidate-solver"),
])
def test_a_partial_child_solution_never_votes(tmp_path, monkeypatch, fault, reason):
    """Codex M5: a non-empty candidate votes only as the whole of a map that RETURNED, for its
    seed, of this walk, internally whole. Anything else re-maps that seed with OFF's mapper;
    its sibling's whole candidate is still used."""
    store, ws, launched, serial, events, _ = _rig(tmp_path, monkeypatch)
    real_launch = GS._launch_draw_child

    def launch(context, database, seed, sparse, output, expected):
        process, job = real_launch(context, database, seed, sparse, output, expected)
        if seed == 4:
            with Path(output).open("rb") as handle:
                result = pickle.load(handle)
            _partial(fault, result["candidate"], result)
            with Path(output).open("wb") as handle:
                pickle.dump(result, handle)
        return process, job

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    mapper = _mapper(store, ws)
    assert mapper(4).solve["seed"] == 3                 # OFF's mapper's sentinel
    assert mapper(5).solve["seed"] == 5                 # the whole sibling votes
    mapper.close()
    assert serial == [4] and events == ["after-draw-0", f"{reason}:seed-4"]
    (entry,) = _journal(ws)
    assert (entry["reason"], entry["seed"], entry["outcome"]) == (reason, 4, "serial")


def test_a_whole_child_result_is_attested_by_the_child():
    candidate = _posed()
    result = GS.draw_child_result(candidate, seed=4, before="d", after="d", map_ms=1.0)
    assert result["complete"] is True and result["seed"] == 4
    assert GS._child_partial(result, candidate, seed=4, input_digest="input") is None
    assert GS.draw_child_result(None, seed=4, before="d", after="d", map_ms=1.0)["complete"] is False


# ---------------------------------------------------------------------------
# W01F-FIX4: what the fix-round mutant pass left unpinned (ADV LOW-3, LOW-4)
# ---------------------------------------------------------------------------


def test_a_reused_pid_is_no_live_writer(tmp_path):
    """R02: the marker names a LIVE process, but one started at another time: stale."""
    import psutil

    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        marker = tmp_path / "sparse-draws" / "writer.json"
        marker.parent.mkdir()
        started = psutil.Process(other.pid).create_time()
        marker.write_text(json.dumps({"pid": other.pid, "created_at": started}))
        assert GS._draw_writer_live(marker)
        marker.write_text(json.dumps({"pid": other.pid, "created_at": started - 5.0}))
        assert not GS._draw_writer_live(marker)
    finally:
        other.kill()
        other.wait()


def test_an_open_mappers_own_root_is_live_until_it_closes(tmp_path, monkeypatch):
    """R10 / R13: an OPEN concurrent mapper registers its root, so its own marker is live to
    this process -- never swept under its running children -- and stale once it closes."""
    store, ws, _, _, _, _ = _rig(tmp_path, monkeypatch)
    root = ws.root / GS.CONSENSUS_SPARSE_DIRNAME
    mapper = _mapper(store, ws)
    mapper(4)                                           # child 5 still "running"
    assert GS._draw_writer_live(root / "writer.json")
    assert GS._sweep_draw_root(root) is False and (root / "child-5").exists()
    mapper.close()
    assert not root.exists()
    root.mkdir()
    (root / "writer.json").write_text(json.dumps(
        {"pid": os.getpid(), "created_at": __import__("psutil").Process().create_time()}))
    assert not GS._draw_writer_live(root / "writer.json")


def test_an_unreadable_marker_is_stale_under_the_session_lock(tmp_path, monkeypatch):
    """ADV LOW-3: under the session writer lock no live writer's marker can be unreadable, so it
    no longer makes every later ON finish map beside it: it is swept, and the draws are
    concurrent."""
    store, ws, launched, serial, events, _ = _rig(tmp_path, monkeypatch)
    root = ws.root / GS.CONSENSUS_SPARSE_DIRNAME
    root.mkdir()
    (root / "writer.json").write_text("{torn")
    mapper = _mapper(store, ws)
    mapper(4)
    mapper.close()
    assert events == ["after-draw-0"] and serial == [] and not root.exists()


def test_a_parent_digest_that_raises_is_not_an_unchanged_database(tmp_path, monkeypatch):
    """R08: the parent's own (third) digest of a child's private copy raising cannot show the
    copy unchanged: that seed is re-mapped."""
    store, ws, _, serial, events, _ = _rig(tmp_path, monkeypatch)
    real_digest = GS.database_digest
    calls = {}

    def digest(path):
        if Path(path).name == "database.db" and Path(path).parent.name == "child-4":
            calls[path] = calls.get(path, 0) + 1
            if calls[path] == 2:                        # the copy check passed; then collection
                raise OSError("the private copy cannot be read")
        return real_digest(path)

    monkeypatch.setattr(GS, "database_digest", digest)
    mapper = _mapper(store, ws)
    assert mapper(4).solve["seed"] == 3 and serial == [4]
    mapper.close()
    assert events == ["after-draw-0", "database-child-digest-mismatch:seed-4"]
