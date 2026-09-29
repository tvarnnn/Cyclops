"""Stop mask child ownership, isolation and cache handoff (W0-3,
`TOWER_WORLD_SOLVE_MASKS_AT_STOP`).

The two rulings (`world_build_session.STOP_MASKS_JOIN_ENDS_ON_SOFT_STOP`,
`STOP_MASKS_UNCONFIRMED_CHILD_SKIPS_FINAL`) each have a test here that runs
with the constant's DEFAULT, so a flipped default fails; the real soft stop
(a closed stdin) is in `test_world_builder_lifecycle.py`.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import world_build_session as B
from scripts import world_solve_masks as M
from tower.world_builder import global_solve, solve_masks as SM
from tower.world_builder.engine import WorldBuilderEngine
from tower.world_builder.records import (
    FINAL_SOLVE_SKIPPED,
    FINAL_SOLVE_UNAVAILABLE,
    CameraIntrinsics,
    Keyframe,
)
from tower.world_builder.store import WorldStore
from tests.test_world_builder_solve_masks import StubDetector

LOGGER = B.logger.name


class Job:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class Process:
    def __init__(self, rc=None):
        self.pid = os.getpid()
        self.returncode = rc

    def poll(self):
        return self.returncode


def solver(root):
    class Solver:
        running = True
        world_id = "w"
        session_id = "s"

        def __init__(self):
            self.root = root

        def wait(self, timeout, *, should_stop):
            self.timeout = timeout

    return Solver()


def _live(root, world="w", session="s"):
    return global_solve.workspace_for(WorldStore(root), world, session)


def _write_stage(ws, images: dict, *, available=True, masked=None):
    """What a child that exited would have left: a result and one npz per image."""
    stage = ws.root / M.STAGE_PARENT_DIRNAME / M.STAGE_DIRNAME
    (stage / "transients").mkdir(parents=True, exist_ok=True)
    for name, sha in images.items():
        (stage / "transients" / f"{Path(name).stem}.{sha[:12]}.gdsam.npz").write_bytes(b"m")
    (stage / "result.json").write_text(json.dumps({
        "images": images, "masked": len(images) if masked is None else masked,
        "cache_hits": 0, "available": available}), encoding="utf-8")
    return stage


# -- the switch -------------------------------------------------------------


def test_stop_mask_switch_defaults_off(monkeypatch):
    from tower.config import world_solve_masks_at_stop_setting
    monkeypatch.delenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", raising=False)
    assert world_solve_masks_at_stop_setting() is False
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", "on")
    assert world_solve_masks_at_stop_setting() is True


def test_stop_mask_switch_logs_an_unrecognised_value_once(monkeypatch, caplog):
    """C25 review L6: a typo reads as off, and says so."""
    from tower import config
    monkeypatch.setattr(config, "_world_solve_masks_at_stop_warned_values", set())
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", "enabled")
    with caplog.at_level(logging.WARNING, logger=config.logger.name):
        assert config.world_solve_masks_at_stop_setting() is False
        assert config.world_solve_masks_at_stop_setting() is False
    lines = [r.getMessage() for r in caplog.records if "SOLVE_MASKS_AT_STOP" in r.getMessage()]
    assert len(lines) == 1 and "'enabled'" in lines[0]
    caplog.clear()
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", "false")
    with caplog.at_level(logging.WARNING, logger=config.logger.name):
        assert config.world_solve_masks_at_stop_setting() is False
    assert not caplog.records


# -- launch -----------------------------------------------------------------


def test_launch_contract_and_job_assignment(tmp_path, monkeypatch, caplog):
    process, job, seen = Process(rc=1), Job(), {}
    monkeypatch.setattr(B, "assign_to_job", lambda p: job if p is process else None)
    monkeypatch.setattr(B, "python_executable", lambda: "python")
    monkeypatch.setattr(B, "child_environment", lambda: {"sentinel": "yes"})

    def spawn(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return process

    child = B.StopSolverMasks(solver(tmp_path), script=Path("stub.py"), spawn=spawn)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert child.launch()
        assert child.join(should_stop=lambda: False)
    assert seen["argv"] == ["python", "stub.py", "--root", str(tmp_path),
                             "--world", "w", "--session", "s"]
    assert seen["kwargs"]["stdin"] == B.subprocess.DEVNULL
    assert seen["kwargs"]["stderr"] == B.subprocess.STDOUT
    assert seen["kwargs"]["env"]["sentinel"] == "yes"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        assert seen["kwargs"]["env"][name] == "2", name
    if os.name == "nt":
        assert seen["kwargs"]["creationflags"] == B.subprocess.BELOW_NORMAL_PRIORITY_CLASS
    assert job.closed
    log = tmp_path / "worlds" / "w" / "solve" / "s" / "solve_masks_at_stop.log"
    assert log.exists() and seen["kwargs"]["stdout"].name == str(log)
    # Observability (C25 review M3): the launch, and the exit with its code,
    # duration, images masked and cache hits.
    messages = [r.getMessage() for r in caplog.records]
    assert f"[Tower][WorldBuilder] Stop mask prefill launched (pid {process.pid})" in messages
    exits = [m for m in messages if m.startswith("[Tower][WorldBuilder] Stop mask prefill exited 1 in ")]
    assert len(exits) == 1 and exits[0].endswith("; images masked unknown, cache hits unknown")
    assert any("exited 1; nothing it staged is promoted, and the stage is swept" in m
               for m in messages)


def test_argv_round_trips_through_the_child_parser_and_caps_threads(tmp_path, monkeypatch):
    """C25 review L3 (M11) and the audit's B2(c): the argv the parent builds
    parses, through the child's OWN parser, to this world and session; the
    child warms its native stack, then caps cv2 and torch at 2 threads."""
    import torch

    seen = {}
    monkeypatch.setattr(B, "assign_to_job", lambda p: Job())
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda argv, **k: (
        seen.update(argv=argv) or Process(rc=1)))
    assert child.launch()
    child.join(should_stop=lambda: False)
    argv = seen["argv"]
    assert Path(argv[1]) == B.TOWER_ROOT / "scripts" / "world_solve_masks.py"
    calls = []
    monkeypatch.setattr(M, "prewarm_world_builder", lambda: calls.append("prewarm"))
    monkeypatch.setattr(cv2, "setNumThreads", lambda n: calls.append(("cv2", n)))
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(("torch", n)))
    monkeypatch.setattr(M, "prefill", lambda root, world, session: (
        calls.append((Path(root).resolve(), world, session)) or 0))
    assert M.main(argv[2:]) == 0
    assert calls == ["prewarm", ("cv2", 2), ("torch", 2), (tmp_path.resolve(), "w", "s")]


def test_job_assignment_failure_disables_prefill(tmp_path, monkeypatch):
    process = Process()
    monkeypatch.setattr(B, "assign_to_job", lambda p: None)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: setattr(process, "returncode", 1) or True)
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process)
    assert not child.launch()
    assert process.poll() == 1 and child._child is None


def test_launch_exception_leaves_background_wait_available(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "python_executable", lambda: (_ for _ in ()).throw(OSError("bad python")))
    s = solver(tmp_path)
    assert B.wait_for_background_solve_at_stop(s, 0.2, should_stop=lambda: False,
                                               stop_at=time.monotonic())
    assert 0.0 < s.timeout <= 0.2 + 1e-6


def test_a_stale_stage_is_swept_before_the_child_starts(tmp_path, monkeypatch):
    ws = _live(tmp_path)
    _write_stage(ws, {"a.jpg": "0" * 40})
    (ws.root / M.STAGE_PARENT_DIRNAME / "q1234abcd").mkdir()
    at_spawn = {}

    def spawn(*a, **k):
        at_spawn["left"] = (ws.root / M.STAGE_PARENT_DIRNAME).exists()
        return Process(rc=1)

    monkeypatch.setattr(B, "assign_to_job", lambda p: Job())
    child = B.StopSolverMasks(solver(tmp_path), spawn=spawn)
    assert child.launch()
    assert at_spawn == {"left": False}
    child.join(should_stop=lambda: False)


# -- the 120-s cutoff: one Stop instant on both paths ------------------------


class Clock:
    """`world_build_session`'s `time`, frozen: sleeping is what moves it."""

    def __init__(self):
        self.now = 5000.0

    def monotonic(self):
        return self.now

    perf_counter = monotonic

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_child_startup_consumes_original_stop_deadline(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(B, "time", clock)

    class SlowLaunch:
        _child = None

        def __init__(self, *args, **kwargs):
            pass

        def launch(self):
            clock.sleep(0.08)
            return True

        def join(self, **kwargs):
            return True

    monkeypatch.setattr(B, "StopSolverMasks", SlowLaunch)
    s = solver(tmp_path)
    assert B.wait_for_background_solve_at_stop(
        s, 0.05, should_stop=lambda: False, stop_at=clock.monotonic())
    assert s.timeout == 0.0


@pytest.mark.parametrize("path", ["off", "on"])
@pytest.mark.parametrize("finishes_after", [119.9, 120.1])
def test_the_120s_cutoff_boundary_is_the_same_on_both_paths(tmp_path, monkeypatch, path,
                                                            finishes_after):
    """C25x H3 / M6: a background solve that finishes 0.1 s inside the
    120-s wait is kept on BOTH paths, and one 0.1 s outside it is terminated
    on BOTH -- although the switch-on path spends 3 s starting its child
    first. The real `BackgroundSolver.wait`, on a frozen clock."""
    clock = Clock()
    monkeypatch.setattr(B, "time", clock)
    stop = clock.monotonic()
    terminated = []

    class Solve:
        pid = 1

        def __init__(self):
            self.returncode = None

        def poll(self):
            if self.returncode is None and clock.now >= stop + finishes_after:
                self.returncode = 0
            return self.returncode

    def terminate(process):
        terminated.append(clock.now - stop)
        process.returncode = 1

    monkeypatch.setattr(B, "_terminate_process_tree", terminate)
    background = B.BackgroundSolver(root=tmp_path, world_id="w", session_id="s", every=1,
                                    capture_dirs=[])
    background._child = Solve()

    class SlowLaunch:
        _child = None

        def __init__(self, *a, **k):
            pass

        def launch(self):
            clock.sleep(3.0)        # Popen + Job Object + interpreter start-up
            return True

        def join(self, **kwargs):
            return True

    monkeypatch.setattr(B, "StopSolverMasks", SlowLaunch)
    if path == "off":
        background.wait(120.0, should_stop=lambda: False)     # exactly main's off path
    else:
        assert B.wait_for_background_solve_at_stop(
            background, 120.0, should_stop=lambda: False, stop_at=stop)
    if finishes_after < 120.0:
        assert terminated == []
    else:
        assert len(terminated) == 1 and 120.0 <= terminated[0] < 120.0 + B.CHILD_POLL_S + 1e-9


# -- the join: bounded; a HARD stop ends it; RULING 1 -------------------------


@pytest.mark.parametrize("reason", ["timeout", "hard"])
def test_join_terminates_hung_child_before_return(tmp_path, monkeypatch, reason, caplog):
    class Hung(Process):
        """Hung, as far as the join can tell -- except that, so a join that
        never ends it FAILS this test instead of hanging the suite, it gives
        up by itself after 200 polls, exiting 0."""
        polls = 0

        def poll(self):
            self.polls += 1
            if self.returncode is None and self.polls > 200:
                self.returncode = 0
            return self.returncode

    process, job = Hung(), Job()
    monkeypatch.setattr(B, "CHILD_POLL_S", 0.001)
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    calls = []

    def terminate(p, **kw):
        calls.append(kw)
        p.returncode = 1
        return True

    monkeypatch.setattr(B, "terminate_tree", terminate)
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process,
                              join_timeout=-1 if reason == "timeout" else 1800)
    assert child.launch()
    stage = _write_stage(_live(tmp_path), {"a.jpg": "0" * 40})   # what it staged so far
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert child.join(should_stop=lambda: reason == "hard")
    assert calls and calls[0]["hard"] and calls[0]["job"] is job
    assert process.poll() == 1 and job.closed
    assert not stage.parent.exists(), "a terminated child's stage must be swept"
    assert not (_live(tmp_path).root / "transients").exists()
    why = "timed out after -1s" if reason == "timeout" else "a stop was requested"
    assert any(f"join ended ({why}); terminating pid" in r.getMessage() for r in caplog.records)


def test_ruling_1_a_soft_stop_does_not_end_the_join(tmp_path, monkeypatch, caplog):
    """RULING 1, at the DEFAULT: the soft stop every ordinary walk sets is
    already true when the join starts; the join carries on, and the child's
    work is promoted. (The real stdin close: the lifecycle test.)"""
    assert B.STOP_MASKS_JOIN_ENDS_ON_SOFT_STOP is False
    process, job = Process(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: pytest.fail("a soft stop killed it"))
    monkeypatch.setattr(B, "CHILD_POLL_S", 0.001)
    ws = _live(tmp_path)
    ws.images_dir.mkdir(parents=True)
    (ws.images_dir / "a.jpg").write_bytes(b"pixels")
    polls = []

    def soft():
        polls.append(1)
        if len(polls) == 5:     # the child finishes while the soft stop stands
            _write_stage(ws, {"a.jpg": hashlib.sha1(b"pixels").hexdigest()})
            process.returncode = 0
        return True

    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process)
    assert child.launch()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert child.join(should_stop=lambda: False, should_soft_stop=soft)
    assert len(list((ws.root / "transients").glob("a.*.gdsam.npz"))) == 1
    notes = [r.getMessage() for r in caplog.records if "soft stop arrived" in r.getMessage()]
    assert len(notes) == 1 and notes[0].endswith("so it does not end the join")


def test_ruling_1_flipped_a_soft_stop_ends_the_join(tmp_path, monkeypatch):
    """The other ruling is the one constant: brief item 4 / C25x H1."""
    monkeypatch.setattr(B, "STOP_MASKS_JOIN_ENDS_ON_SOFT_STOP", True)
    process, job = Process(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree",
                        lambda p, **k: setattr(p, "returncode", 1) or True)
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process)
    assert child.launch()
    assert child.join(should_stop=lambda: False, should_soft_stop=lambda: True)
    assert process.poll() == 1 and job.closed


# -- a child that cannot be confirmed dead; RULING 2 --------------------------


def test_failed_termination_retains_owned_child_and_never_promotes(tmp_path, monkeypatch,
                                                                   caplog):
    process, job = Process(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: False)
    ws = _live(tmp_path)
    ws.images_dir.mkdir(parents=True)
    (ws.images_dir / "a.jpg").write_bytes(b"pixels")
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process,
                              join_timeout=-1)
    assert child.launch()
    stage = _write_stage(ws, {"a.jpg": hashlib.sha1(b"pixels").hexdigest()})
    try:
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            assert child.join(should_stop=lambda: False) is False
        assert child in B._owned_stalled_mask_children
        assert not job.closed and child._child is process
        assert stage.exists(), "a child that may still write keeps its stage"
        assert not (ws.root / "transients").exists(), "nothing of it is promoted"
        assert any("could not be confirmed dead" in r.getMessage() for r in caplog.records)
    finally:
        process.returncode = 1
        child.close()
        B._owned_stalled_mask_children.remove(child)
    assert not stage.parent.exists() and job.closed


def _frames_dir(tmp_path):
    from tower.capture import CaptureRecorder
    from tests.test_world_builder_lifecycle import _frames, _write

    recorder = CaptureRecorder(tmp_path / "capture")
    capture_id = recorder.start()
    _write(recorder, _frames(10))
    recorder.stop()
    return recorder.capture_dir(capture_id) / "frames"


def _main_with_a_running_background_solve(tmp_path, monkeypatch, *, at_stop=True):
    """`main`, in process, over a replay whose background solve is still
    running at Stop. Returns (exit code, store, world, session, events)."""
    monkeypatch.setattr(B.StopRequest, "install", lambda self, **_: None)
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS", "on")
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", "on" if at_stop else "off")
    events = tmp_path / "events.txt"
    stub = tmp_path / "stub_solve.py"
    stub.write_text(
        "import json, pathlib, sys, time\n"
        f"events = pathlib.Path({str(events)!r})\n"
        "if '--final' in sys.argv:\n"
        "    with events.open('a') as f: f.write('final\\n')\n"
        "    print(json.dumps({'solved': False, 'reason': 'stub'}))\n"
        "else:\n"
        "    with events.open('a') as f: f.write('background\\n')\n"
        "    time.sleep(60)\n", encoding="utf-8")
    root = tmp_path / "worlds"
    code = B.main(["--frames", str(_frames_dir(tmp_path)), "--root", str(root),
                   "--rebuild-every", "2", "--format", "json", "--solve",
                   "--solve-every", "2", "--solve-wait-seconds", "0.3",
                   "--solve-script", str(stub)])
    store = WorldStore(root)
    (world,) = store.list_world_ids()
    (session,) = store.list_session_ids(world)
    order = events.read_text().splitlines() if events.exists() else []
    return code, store, world, session, order


@pytest.mark.parametrize("skips", [None, True])
def test_ruling_2_unkillable_child_final_solve_still_runs(tmp_path, monkeypatch, caplog, skips):
    """RULING 2, at the DEFAULT (`skips=None`): a Stop mask child that cannot
    be confirmed dead does not cost the wearer the final solve; nothing it
    staged is promoted, and its stage is left where nothing reads it. `True`
    is the other ruling, one constant away: C25f's skip."""
    if skips is None:
        assert B.STOP_MASKS_UNCONFIRMED_CHILD_SKIPS_FINAL is False
    else:
        monkeypatch.setattr(B, "STOP_MASKS_UNCONFIRMED_CHILD_SKIPS_FINAL", skips)
    never, job, made = Process(), Job(), []
    real = B.StopSolverMasks

    def spawn(*a, **k):
        # A child that wrote a complete stage for a live image and then hung.
        ws = _live(made[0].solver.root, made[0].solver.world_id, made[0].solver.session_id)
        ws.images_dir.mkdir(parents=True, exist_ok=True)
        (ws.images_dir / "zz.jpg").write_bytes(b"pixels")
        _write_stage(ws, {"zz.jpg": hashlib.sha1(b"pixels").hexdigest()})
        return never

    def factory(s, *, script=None):
        made.append(real(s, script=script, spawn=spawn, join_timeout=0.3))
        return made[-1]

    monkeypatch.setattr(B, "StopSolverMasks", factory)
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: False)
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            code, store, world, session, order = _main_with_a_running_background_solve(
                tmp_path, monkeypatch)
        assert code == 0
        record = store.read_session(world, session).finalization
        ws = global_solve.workspace_for(store, world, session)
        assert made and made[0] in B._owned_stalled_mask_children
        assert not list((ws.root / "transients").glob("zz.*.npz")), "promoted"
        assert (ws.root / M.STAGE_PARENT_DIRNAME / M.STAGE_DIRNAME / "result.json").exists()
        messages = [r.getMessage() for r in caplog.records]
        # `background` is the stub's own first line, and a builder that reaches Stop
        # first may end it before it runs: only what follows is asserted.
        order = [event for event in order if event != "background"]
        if skips is None:
            assert order == ["final"]
            assert record["final_solve"] == FINAL_SOLVE_UNAVAILABLE
            assert any("could not be confirmed dead; nothing it staged is promoted, and the "
                       "final solve runs anyway" in m for m in messages)
        else:
            assert order == []
            assert record["final_solve"] == FINAL_SOLVE_SKIPPED
            assert "Stop mask child could not be stopped" in record["detail"]
    finally:
        never.returncode = 1
        for child in made:
            child.close()
            if child in B._owned_stalled_mask_children:
                B._owned_stalled_mask_children.remove(child)


def test_main_passes_the_stop_instant_and_the_hard_stop(tmp_path, monkeypatch):
    """`main`'s wiring (audit B2(a)): the wrapper gets the Stop instant, taken
    immediately before it, and the background wait's deadline is that
    instant plus `--solve-wait-seconds` although the child's start-up ran first."""
    real_wrapper, real_wait = B.wait_for_background_solve_at_stop, B.BackgroundSolver.wait
    seen, waits = {}, []

    def wrapper(s, timeout, **kw):
        seen.update(kw, timeout=timeout, entered=time.monotonic())
        return real_wrapper(s, timeout, **kw)

    def wait(self, timeout, should_stop=None):
        waits.append((time.monotonic(), timeout, should_stop))
        return real_wait(self, timeout, should_stop=should_stop)

    class SlowChild:
        _child = None

        def __init__(self, *a, **k):
            pass

        def launch(self):
            time.sleep(0.2)
            return True

        def join(self, **kw):
            return True

    monkeypatch.setattr(B, "wait_for_background_solve_at_stop", wrapper)
    monkeypatch.setattr(B.BackgroundSolver, "wait", wait)
    monkeypatch.setattr(B, "StopSolverMasks", SlowChild)
    code, store, world, session, order = _main_with_a_running_background_solve(
        tmp_path, monkeypatch)
    assert code == 0 and [e for e in order if e != "background"] == ["final"]
    assert seen["timeout"] == 0.3
    assert 0.0 <= seen["entered"] - seen["stop_at"] < 0.1
    assert seen["should_stop"].__name__ == "hard_asked_for"
    assert seen["should_soft_stop"].__name__ == "asked_for"
    (started, remaining, should_stop), = [w for w in waits if w[0] >= seen["stop_at"]
                                          and w[1] is not None and w[1] > 0][:1]
    assert remaining <= 0.3 - 0.2 + 0.05
    assert abs((started + remaining) - (seen["stop_at"] + 0.3)) < 0.02
    assert should_stop.__name__ == "hard_asked_for"


def test_main_switch_off_calls_the_original_wait_directly(tmp_path, monkeypatch):
    """C25x M6: OFF is the original `solver.wait(args.solve_wait_seconds,
    should_stop=stop_request.hard_asked_for)`; the wrapper is never entered."""
    real_wait = B.BackgroundSolver.wait
    waits = []

    def wait(self, *args, **kwargs):
        waits.append((args, kwargs))
        return real_wait(self, *args, **kwargs)

    monkeypatch.setattr(B.BackgroundSolver, "wait", wait)
    monkeypatch.setattr(B, "wait_for_background_solve_at_stop",
                        lambda *a, **k: pytest.fail("the switch-off path entered the wrapper"))
    code, store, world, session, order = _main_with_a_running_background_solve(
        tmp_path, monkeypatch, at_stop=False)
    assert code == 0 and [e for e in order if e != "background"] == ["final"]
    stop_waits = [(a, k) for a, k in waits if a == (0.3,)]
    assert len(stop_waits) == 1
    assert set(stop_waits[0][1]) == {"should_stop"}
    assert stop_waits[0][1]["should_stop"].__name__ == "hard_asked_for"


# -- ownership across process death -----------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_job_kills_mask_child_when_parent_dies(tmp_path):
    import psutil

    pidfile = tmp_path / "child.pid"
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import os, pathlib, subprocess, sys\n"
        "from tower.process_ownership import assign_to_job\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],\n"
        "                     stdin=subprocess.DEVNULL)\n"
        "job = assign_to_job(p)\n"
        "if job is None: p.kill(); sys.exit(3)\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid))\n"
        "os._exit(0)\n", encoding="utf-8")
    completed = subprocess.run([sys.executable, str(parent)], timeout=20,
                               check=False, capture_output=True)
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 10
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not psutil.pid_exists(pid), "Job close left a mask child alive"


def test_wait_exception_kills_child(tmp_path, monkeypatch):
    process, job = Process(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: setattr(p, "returncode", 1) or True)
    real_child = B.StopSolverMasks
    monkeypatch.setattr(B, "StopSolverMasks", lambda *a, **k:
                        real_child(*a, spawn=lambda *x, **y: process, **k))
    s = solver(tmp_path)
    s.wait = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        B.wait_for_background_solve_at_stop(s, 1, should_stop=lambda: False,
                                            stop_at=time.monotonic())
    assert process.poll() == 1 and job.closed


# -- the child: a snapshot, a stage, and a promotion ---------------------------


@pytest.fixture
def images(tmp_path, monkeypatch):
    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    world = engine.create_world("test")
    intrinsics = CameraIntrinsics(source="self_calibrated", model="pinhole_radtan",
                                  fx=50, fy=50, cx=32, cy=24,
                                  dist_coeffs=(0, 0, 0, 0, 0),
                                  calibrated_width=64, calibrated_height=48)
    session = engine.start_session(world, frame_source="synthetic", intrinsics=intrinsics)
    ws = global_solve.workspace_for(store, world, session)
    ws.images_dir.mkdir(parents=True)
    ws.camera_path.write_text(json.dumps(global_solve.PinholeCamera(
        50, 50, 32, 24, 64, 48).to_json_dict()), encoding="utf-8")
    for i in range(3):
        name = f"{i:08d}.jpg"
        if i < 2:
            rgb = np.full((48, 64, 3), i * 10 + 100, np.uint8)
            ok, encoded = cv2.imencode(".jpg", rgb)
            assert ok
            (ws.images_dir / name).write_bytes(encoded.tobytes())
        store.append_keyframe(world, Keyframe(
            keyframe_id=f"{session}:{i}", session_id=session, source_seq=i,
            received_at=float(i), image_relpath=f"images/{name}", width=64,
            height=48, byte_count=1, segment_index=0))
    ws.database_path.write_bytes(b"walk database sentinel")
    monkeypatch.setattr(SM.T, "BACKEND_FACTORY", StubDetector())
    monkeypatch.setattr(SM, "cuda_unavailable_reason", lambda: None)
    monkeypatch.setattr(global_solve, "prepare_images", lambda *a, **k: pytest.fail(
        "Stop child must not prepare solver images"))
    return tmp_path, store, world, session, ws


def _join(images, rc):
    """The parent's side of a child that exited `rc`: the real `join`."""
    root, _store, world, session, _ws = images
    child = B.StopSolverMasks(solver(root))
    child.solver.world_id, child.solver.session_id = world, session
    child._child = Process(rc=rc)
    child._started = time.monotonic()
    return child.join(should_stop=lambda: False)


def _files(root: Path) -> set:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_the_stage_is_outside_every_path_the_final_solve_or_a_refinish_reads(images):
    """C25 review L4 / audit CL-L4: the child writes ONLY under
    `solve/<session>/stop_masks/`, which is neither the live cache, nor
    `masks/`, nor anything `world_refinish` copies into a fresh solve."""
    from scripts.world_refinish import SOLVE_COPY_BACK

    root, _store, world, session, ws = images
    before = _files(ws.root)
    assert M.prefill(root, world, session) == 0
    added = _files(ws.root) - before
    assert added and all(p.startswith(f"{M.STAGE_PARENT_DIRNAME}/{M.STAGE_DIRNAME}/")
                         for p in added), sorted(added)
    assert M.STAGE_PARENT_DIRNAME not in SOLVE_COPY_BACK
    assert M.STAGE_PARENT_DIRNAME not in (SM.CACHE_DIRNAME, SM.MASKS_DIRNAME)
    assert (_files(ws.root) - added) == before


def test_child_uses_staging_and_never_touches_database(images, caplog):
    root, store, world, session, ws = images
    before = hashlib.sha256(ws.database_path.read_bytes()).hexdigest()
    assert M.prefill(root, world, session) == 0
    assert hashlib.sha256(ws.database_path.read_bytes()).hexdigest() == before
    assert not (ws.root / "masks").exists() and not (ws.root / "transients").exists()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert _join(images, 0)
    # Promoted into a live cache that did not exist yet; nothing else promoted.
    assert len(list((ws.root / "transients").glob("*.npz"))) == 4
    assert not (ws.root / "transients" / "index.json").exists()
    assert not (ws.root / "masks").exists()
    assert not (ws.root / M.STAGE_PARENT_DIRNAME).exists(), "the stage is swept"
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("[Tower][WorldBuilder] Stop mask prefill exited 0 in ")
               and m.endswith("; images masked 2, cache hits 0") for m in messages)
    assert ("[Tower][WorldBuilder] Stop mask prefill promoted 4 cache files: 2 images masked, "
            "0 cache hits in the child; 2 of 2 images still match at the final solve") in messages
    ids = {global_solve.keyframe_image_name(k): k.keyframe_id
           for k in store.read_keyframes(world, session)}
    ok, encoded = cv2.imencode(".jpg", np.full((48, 64, 3), 130, np.uint8))
    assert ok
    (ws.images_dir / "00000002.jpg").write_bytes(encoded.tobytes())
    final = SM.ensure_solver_masks(ws, ["00000000.jpg", "00000001.jpg", "00000002.jpg"],
                                   keyframe_ids=ids, shape=(48, 64),
                                   backend_factory=StubDetector(), device_probe=lambda: None)
    assert final.cache_hits == 2 and final.computed == 1
    assert (ws.root / "masks" / "00000002.jpg.png").exists()


def test_replaced_image_is_never_promoted_under_old_hash(images, monkeypatch):
    root, _store, world, session, ws = images
    name = "00000000.jpg"
    original_sha = hashlib.sha1((ws.images_dir / name).read_bytes()).hexdigest()
    decode = SM._read_solver_image
    changed = [False]

    def replace_at_decode(path):
        if not changed[0] and Path(path).name == name:
            changed[0] = True
            ok, encoded = cv2.imencode(".jpg", np.full((48, 64, 3), 222, np.uint8))
            assert ok
            (ws.images_dir / name).write_bytes(encoded.tobytes())
        return decode(path)

    monkeypatch.setattr(SM, "_read_solver_image", replace_at_decode)
    assert M.prefill(root, world, session) == 0
    stage = ws.root / M.STAGE_PARENT_DIRNAME / M.STAGE_DIRNAME
    manifest = json.loads((stage / "result.json").read_text())
    assert manifest["images"][name] == original_sha
    assert _join(images, 0)
    assert not list((ws.root / "transients").glob(f"{Path(name).stem}.*.npz"))
    final = SM.ensure_solver_masks(ws, [name, "00000001.jpg"], shape=(48, 64),
                                   backend_factory=StubDetector(), device_probe=lambda: None)
    assert final.cache_hits == 1 and final.computed == 1


def test_recalibration_after_the_snapshot_promotes_nothing(images):
    """C25x H2's recalibration corner: the background solve rebuilt the
    images (and the camera) after the child's snapshot. Every hash then
    differs, so nothing is promoted and the final solve masks every image."""
    root, _store, world, session, ws = images
    assert M.prefill(root, world, session) == 0
    for i in range(2):
        ok, encoded = cv2.imencode(".jpg", np.full((48, 64, 3), 7 + i, np.uint8))
        assert ok
        (ws.images_dir / f"{i:08d}.jpg").write_bytes(encoded.tobytes())
    ws.camera_path.write_text(json.dumps(global_solve.PinholeCamera(
        51, 51, 32, 24, 64, 48).to_json_dict()), encoding="utf-8")
    assert _join(images, 0)
    assert not list((ws.root / "transients").glob("*.npz"))
    final = SM.ensure_solver_masks(ws, ["00000000.jpg", "00000001.jpg"], shape=(48, 64),
                                   backend_factory=StubDetector(), device_probe=lambda: None)
    assert final.cache_hits == 0 and final.computed == 2


@pytest.mark.parametrize("failure", ["late failure", "CUDA out of memory"])
def test_partial_detector_failure_is_quarantined(images, monkeypatch, failure, tmp_path):
    """C25x M5: the child fails AFTER its first component wrote. Nothing is
    promoted, the stage is swept, and the final solve's record, PNGs and cache
    are exactly those of a final solve that never had a child."""
    root, _store, world, session, ws = images
    old = global_solve.SolveWorkspace(tmp_path / "old")
    shutil.copytree(ws.root, old.root)          # the switch-off twin, before the child
    good = StubDetector()

    def factory(component):
        backend = good(component)
        if component == SM.T.COMPONENT_ONEFORMER:
            backend.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(failure))
        return backend

    monkeypatch.setattr(SM.T, "BACKEND_FACTORY", factory)
    assert M.prefill(root, world, session) == 1
    stage = ws.root / M.STAGE_PARENT_DIRNAME / M.STAGE_DIRNAME
    assert list((stage / "transients").glob("*.gdsam.npz"))
    assert _join(images, 1)
    assert not (ws.root / "transients").exists()
    assert not (ws.root / M.STAGE_PARENT_DIRNAME).exists()
    monkeypatch.setattr(SM.T, "BACKEND_FACTORY", StubDetector())
    records = []
    for workspace in (ws, old):
        final = SM.ensure_solver_masks(workspace, ["00000000.jpg", "00000001.jpg"],
                                       shape=(48, 64), backend_factory=StubDetector(),
                                       device_probe=lambda: None)
        record = final.record()
        record.pop("seconds")
        records.append(record)
    assert records[0] == records[1] and records[0]["state"] == "applied"
    assert records[0]["cache_hits"] == 0 and records[0]["computed"] == 2
    for sub in ("masks", "transients"):
        assert _files(ws.root / sub) == _files(old.root / sub)
    for png in (ws.root / "masks").glob("*.png"):
        assert png.read_bytes() == (old.root / "masks" / png.name).read_bytes()


def test_an_exit_0_without_a_complete_result_promotes_nothing(tmp_path, caplog):
    ws = _live(tmp_path)
    ws.images_dir.mkdir(parents=True)
    (ws.images_dir / "a.jpg").write_bytes(b"pixels")
    _write_stage(ws, {"a.jpg": hashlib.sha1(b"pixels").hexdigest()}, available=False)
    child = B.StopSolverMasks(solver(tmp_path))
    child._child, child._started = Process(rc=0), time.monotonic()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert child.join(should_stop=lambda: False)
    assert not (ws.root / "transients").exists()
    assert not (ws.root / M.STAGE_PARENT_DIRNAME).exists()
    assert any("without a complete result" in r.getMessage() for r in caplog.records)


# -- the child's 2-thread cap: an identity assertion (manager 150) ------------

# A detector whose "inference" runs on the CPU through cv2's and torch's THREADED
# kernels (upscale, blur, downscale, bilinear interpolation), thresholded into masks.
# It stands in for the real models, which run on the GPU: see the test's docstring.
_CAP_DRIVER = r'''
import json, os, sys
from pathlib import Path
import numpy as np, cv2, torch
from scripts import world_solve_masks as M
from tower.world_builder import solve_masks as SM, transients as T


class CpuDetector(T.ComponentBackend):
    def __init__(self, component):
        self.component = component

    def probe(self):
        return None

    def run(self, items, params, emit, should_stop=None):
        for i, rgb, _unobserved in items:
            h, w = rgb.shape[:2]
            big = cv2.resize(rgb, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
            blur = cv2.GaussianBlur(big, (31, 31), 6.0)
            small = cv2.resize(blur, (w, h), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(small).permute(2, 0, 1)[None].float()
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear",
                                                align_corners=False)
            x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear",
                                                align_corners=False)
            gray = x[0].mean(dim=0).numpy()
            if self.component == T.COMPONENT_GDSAM:
                hand = gray > np.median(gray)
            else:
                hand = np.zeros((h, w), bool)
            phone = gray < np.percentile(gray, 12)
            emit(i, hand, phone, 0.001)
        return {"frames": len(items)}


T.BACKEND_FACTORY = CpuDetector
SM.cuda_unavailable_reason = lambda: None
arm, root, world, session, out = sys.argv[1:6]
if arm == "capped":
    # The child's own entry point: prewarm, `_cap_threads`, `prefill`.
    rc = M.main(["--root", root, "--world", world, "--session", session])
else:
    M.prewarm_world_builder()
    rc = M.prefill(Path(root), world, session)
Path(out).write_text(json.dumps({
    "rc": rc, "torch": torch.get_num_threads(), "cv2": cv2.getNumThreads(),
    "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}))
'''


def _decoded_stage(stage: Path) -> dict:
    out = {}
    for path in sorted((stage / "transients").glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            key = json.loads(bytes(z["key"]).decode())
            key.pop("computed_at")
            key.pop("seconds")
            out[path.name] = {"hand": z["hand"].tobytes(), "phone": z["phone"].tobytes(),
                              "shape": z["shape"].tobytes(), "key": key}
    for path in sorted((stage / "masks").glob("*.png")):
        out[path.name] = path.read_bytes()
    return out


def test_the_child_thread_cap_leaves_its_masks_bit_identical(tmp_path):
    """Manager 150: the child's 2-thread cap gets an identity assertion.

    The SAME world is masked twice, in two fresh processes: once through the
    child's own entry point under its exact cap (the parent's four env caps,
    then `_cap_threads`: cv2 and torch at 2), once uncapped (default
    threads, no env caps). The cache arrays (hand, phone, shape), the cache
    keys (without `computed_at` and `seconds`) and the COLMAP PNG bytes must
    be BIT-IDENTICAL.

    WHAT IT PROVES: every step of the child's own path -- the snapshot, the
    decode, the shape check, the cache write, the composition and the PNG
    encode -- and a detector whose inference runs through cv2's and torch's
    multi-threaded CPU kernels, give the same bits at 2 threads as at the
    machine's default. WHAT IT DOES NOT PROVE: anything about the real
    models (Grounding DINO, SAM 2.1, OneFormer), which run on the GPU. Stage 0
    measured those bit-identical under this exact cap (STAGE0.md §6.6, row T:
    200/200 arrays, 50/50 PNGs); `RUN/experiments/C25f-FINISH/cap_identity_real.py`
    re-checks them through this child's code, under gpulock. A threaded GEMM or
    reduction on the CPU is NOT thread-invariant in general (Stage 0: DINOv2 on
    the CPU, 0/320 rows identical under the cap), which is why the cap is kept
    to a child whose models run on the GPU."""
    import psutil

    rng = np.random.default_rng(7)
    width, height, count = 640, 480, 3
    roots = {}
    for arm in ("capped", "uncapped"):
        root = tmp_path / arm
        store = WorldStore(root)
        engine = WorldBuilderEngine(store)
        world = engine.create_world("cap")
        intrinsics = CameraIntrinsics(source="self_calibrated", model="pinhole_radtan",
                                      fx=500, fy=500, cx=320, cy=240,
                                      dist_coeffs=(0, 0, 0, 0, 0),
                                      calibrated_width=width, calibrated_height=height)
        session = engine.start_session(world, frame_source="synthetic", intrinsics=intrinsics)
        roots[arm] = (root, store, world, session)
    images = []
    for i in range(count):
        ramp = np.linspace(0, 200, width, dtype=np.float32)[None, :, None]
        noise = rng.normal(0, 30, (height, width, 3)).astype(np.float32)
        rgb = np.clip(ramp + noise + 20 * i, 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".jpg", rgb)
        assert ok
        images.append(encoded.tobytes())
    for arm, (root, store, world, session) in roots.items():
        ws = global_solve.workspace_for(store, world, session)
        ws.images_dir.mkdir(parents=True)
        ws.camera_path.write_text(json.dumps(global_solve.PinholeCamera(
            500, 500, 320, 240, width, height).to_json_dict()), encoding="utf-8")
        for i, data in enumerate(images):
            name = f"{i:08d}.jpg"
            (ws.images_dir / name).write_bytes(data)
            store.append_keyframe(world, Keyframe(
                keyframe_id=f"{session}:{i}", session_id=session, source_seq=i,
                received_at=float(i), image_relpath=f"images/{name}", width=width,
                height=height, byte_count=len(data), segment_index=0))
    driver = tmp_path / "cap_driver.py"
    driver.write_text(_CAP_DRIVER, encoding="utf-8")
    seen, stages = {}, {}
    for arm, (root, store, world, session) in roots.items():
        env = {k: v for k, v in os.environ.items() if k not in B.STOP_MASKS_THREAD_ENV}
        env.update(PYTHONPATH=str(B.TOWER_ROOT), PYTHONDONTWRITEBYTECODE="1",
                   CUDA_VISIBLE_DEVICES="-1")
        if arm == "capped":
            env.update({name: "2" for name in B.STOP_MASKS_THREAD_ENV})   # as `launch` does
        out = tmp_path / f"{arm}.json"
        done = subprocess.run([sys.executable, str(driver), arm, str(root), world, session,
                               str(out)], cwd=str(B.TOWER_ROOT), env=env, timeout=300,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-3000:]
        seen[arm] = json.loads(out.read_text())
        assert seen[arm]["rc"] == 0
        ws = global_solve.workspace_for(store, world, session)
        stages[arm] = _decoded_stage(ws.root / M.STAGE_PARENT_DIRNAME / M.STAGE_DIRNAME)
    # The cap was in force in one process and not in the other.
    assert seen["capped"]["torch"] == 2 and seen["capped"]["cv2"] == 2
    assert set(seen["capped"]["env"].values()) == {"2"}
    assert set(seen["uncapped"]["env"].values()) == {None}
    if (psutil.cpu_count(logical=True) or 1) > 2:
        assert seen["uncapped"]["torch"] > 2 and seen["uncapped"]["cv2"] > 2
    # Every image masked, both components, and every bit the same.
    npz = [name for name in stages["capped"] if name.endswith(".npz")]
    assert len(npz) == 2 * count and len(stages["capped"]) == 3 * count
    assert stages["capped"] == stages["uncapped"]
