"""Stop mask child ownership, isolation and cache handoff (W0-3,
`TOWER_WORLD_SOLVE_MASKS_AT_STOP`).

The three rulings (`world_build_session.STOP_MASKS_JOIN_ENDS_ON_SOFT_STOP`,
`STOP_MASKS_UNCONFIRMED_CHILD_SKIPS_FINAL`, `STOP_MASKS_PROMOTE_PARTIAL_ON_TIMEOUT`)
each have a test here that runs with the constant's DEFAULT, so a flipped
default fails, and one with it flipped; the real soft stop (a closed stdin)
is in `test_world_builder_lifecycle.py`.
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


# The solver camera's (height, width) in the tests that fabricate a stage.
SHAPE = (4, 8)


def _camera(ws, shape=SHAPE):
    """The live solver camera: promotion verifies every file against its shape."""
    ws.root.mkdir(parents=True, exist_ok=True)
    if not ws.camera_path.exists():
        ws.camera_path.write_text(json.dumps(global_solve.PinholeCamera(
            8, 8, shape[1] / 2, shape[0] / 2, shape[1], shape[0]).to_json_dict()),
            encoding="utf-8")


def _component(cache: Path, name: str, sha: str, component=SM.T.COMPONENT_GDSAM, *,
               shape=SHAPE, key=None):
    """One cache file exactly as the child writes it (`transients.write_component`),
    VALID for `name` at `sha` unless `key` or `shape` says otherwise."""
    hand = np.zeros(shape, bool)
    hand[0, 0] = True
    path = SM.component_path(cache, name, component, sha)
    SM.T.write_component(path, key or SM.mask_key(component, SM.solver_params(), name, sha),
                         hand, np.zeros(shape, bool), image_sha1=sha)
    return path


def _write_stage(ws, images: dict, *, available=True, masked=None, snapshot=None):
    """What a child that exited would have left: a result and one VALID gdsam cache
    file per image (promotion verifies each), and the live camera it verifies against.
    `snapshot` (name -> bytes): the images it had copied into its stage, which is what
    a join that times out promotes from (RULING 3)."""
    stage = ws.root / M.STAGE_PARENT_DIRNAME / M.STAGE_DIRNAME
    (stage / "transients").mkdir(parents=True, exist_ok=True)
    for name, data in (snapshot or {}).items():
        (stage / "images").mkdir(exist_ok=True)
        (stage / "images" / name).write_bytes(data)
    _camera(ws)
    for name, sha in images.items():
        _component(stage / "transients", name, sha)
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
    process, job, seen, order = Process(rc=1), Job(), {}, []
    marker = M.writer_marker(_live(tmp_path).root, process.pid)

    def assign(p):
        order.append(("job", p is process))
        return job if p is process else None

    monkeypatch.setattr(B, "assign_to_job", assign)
    monkeypatch.setattr(B, "_resume_suspended",
                        lambda p: order.append(("resume", marker.exists())) or True)
    monkeypatch.setattr(B, "python_executable", lambda: "python")
    monkeypatch.setattr(B, "child_environment", lambda: {"sentinel": "yes"})

    def spawn(argv, **kwargs):
        order.append(("spawn", kwargs["creationflags"]))
        seen.update(argv=argv, kwargs=kwargs)
        return process

    child = B.StopSolverMasks(solver(tmp_path), script=Path("stub.py"), spawn=spawn)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert child.launch()
        assert child.join(should_stop=lambda: False)
    # C25x2 HIGH-1: spawned SUSPENDED, put in the Job, marked, and only then resumed.
    if os.name == "nt":
        assert order == [("spawn", B.subprocess.BELOW_NORMAL_PRIORITY_CLASS | 0x4),
                         ("job", True), ("resume", True)]
    assert not marker.parent.exists(), "the marker is swept with the stage"
    assert seen["argv"] == ["python", "stub.py", "--root", str(tmp_path),
                             "--world", "w", "--session", "s"]
    assert seen["kwargs"]["stdin"] == B.subprocess.DEVNULL
    assert seen["kwargs"]["stderr"] == B.subprocess.STDOUT
    assert seen["kwargs"]["env"]["sentinel"] == "yes"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        assert seen["kwargs"]["env"][name] == "2", name
    if os.name == "nt":
        assert seen["kwargs"]["creationflags"] == (B.subprocess.BELOW_NORMAL_PRIORITY_CLASS
                                                   | B._CREATE_SUSPENDED)
        assert B._CREATE_SUSPENDED == 0x4
    assert job.closed
    log = tmp_path / "worlds" / "w" / "solve" / "s" / "solve_masks_at_stop.log"
    assert log.exists() and seen["kwargs"]["stdout"].name == str(log)
    # Observability (C25 review M3): the launch, and the exit with its code,
    # duration, images masked and cache hits.
    messages = [r.getMessage() for r in caplog.records]
    assert (f"[Tower][WorldBuilder] Stop mask prefill launched (pid {process.pid}); join bound "
            "1800s for 0 solver images, or 120s without progress") in messages
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
    monkeypatch.setattr(B, "_resume_suspended", lambda p: pytest.fail("resumed without a Job"))
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: setattr(process, "returncode", 1) or True)
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process)
    assert not child.launch()
    assert process.poll() == 1 and child._child is None


def test_no_job_and_no_kill_the_child_never_runs_and_is_never_joined(tmp_path, monkeypatch,
                                                                      caplog):
    """C25x2 HIGH-1 and C25f review LOW-2 (mutant V2): Job assignment fails AND
    the termination cannot be confirmed. The child was created suspended and is
    NEVER resumed, so it runs nothing beside the final solve; it is tracked and
    says so honestly (not "its Job Object closes with this builder": it has
    none); the wrapper never joins it, the background wait still runs on the
    original deadline, and the answer is False (RULING 2 then runs the final)."""
    process, resumed, joined, made = Process(), [], [], []
    monkeypatch.setattr(B, "assign_to_job", lambda p: None)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: False)
    monkeypatch.setattr(B, "_resume_suspended", lambda p: resumed.append(p) or True)
    monkeypatch.setattr(B.StopSolverMasks, "join", lambda self, **k: joined.append(self) or True)
    real = B.StopSolverMasks

    def factory(s, *, script=None):
        made.append(real(s, script=script, spawn=lambda *a, **k: process))
        return made[-1]

    monkeypatch.setattr(B, "StopSolverMasks", factory)
    s = solver(tmp_path)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        answer = B.wait_for_background_solve_at_stop(s, 0.2, should_stop=lambda: False,
                                                     stop_at=time.monotonic())
    try:
        assert answer is False
        assert 0.0 < s.timeout <= 0.2 + 1e-6, "the background wait still ran"
        assert joined == [], "a child that was never owned was joined"
        assert resumed == [], "a child with no Job Object was resumed"
        assert made[0] in B._owned_stalled_mask_children and made[0]._child is process
        assert not M.writer_marker(_live(tmp_path).root, process.pid).exists()
        messages = [r.getMessage() for r in caplog.records]
        if os.name == "nt":
            assert any("has no Job Object; disabling it (it is never resumed)" in m
                       for m in messages)
            assert any("could not be confirmed dead; it was created suspended and never "
                       "resumed, so it has run none of its own code" in m for m in messages)
        assert not any("its Job Object closes with this builder" in m for m in messages)
    finally:
        process.returncode = 1
        made[0].close()
    assert made[0] not in B._owned_stalled_mask_children


@pytest.mark.skipif(os.name != "nt", reason="CREATE_SUSPENDED and Job Objects are Windows")
def test_a_real_child_that_cannot_be_put_in_a_job_never_runs(tmp_path, monkeypatch):
    """C25x2 HIGH-1 with a REAL process: `launch` spawns the child suspended;
    Job assignment fails and termination is refused. The child's first line
    would write a file -- and it never does, because it never runs."""
    import psutil

    ran = tmp_path / "ran.txt"
    script = tmp_path / "stub_mask_child.py"
    script.write_text(f"import pathlib\npathlib.Path({str(ran)!r}).write_text('ran')\n",
                      encoding="utf-8")
    monkeypatch.setattr(B, "assign_to_job", lambda p: None)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: False)
    child = B.StopSolverMasks(solver(tmp_path), script=script)
    try:
        assert child.launch() is False
        process = child._child
        assert isinstance(process, subprocess.Popen) and process.poll() is None
        time.sleep(2.0)
        assert not ran.exists(), "a child with no Job Object ran its own code"
        assert process.poll() is None
        assert psutil.Process(process.pid).status() == psutil.STATUS_STOPPED
    finally:
        if child._child is not None:
            child._child.kill()
            child._child.wait(10)
        child.close()
    assert not ran.exists()
    assert child not in B._owned_stalled_mask_children


@pytest.mark.skipif(os.name != "nt", reason="only a Windows child is created suspended")
def test_a_failed_resume_kills_the_child_through_its_job_and_falls_back(tmp_path, monkeypatch,
                                                                        caplog):
    """C25f-FIX2 review L-1 (mutant V6): the child IS in its Job, but
    `NtResumeProcess` fails. It must be terminated through that Job (it is still
    suspended, so it never ran), its marker and stage swept, and the wrapper must
    FALL BACK: the background wait on the original deadline, no join of a live
    child -- it would wait the whole bound for a child that never runs -- and
    True, since the child is confirmed dead and the final solve computes every
    mask."""
    process, job, kills, joined, made = Process(), Job(), [], [], []
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "_resume_suspended", lambda p: False)

    def terminate(p, **kw):
        kills.append(kw)
        p.returncode = 1
        return True

    monkeypatch.setattr(B, "terminate_tree", terminate)
    real, real_join = B.StopSolverMasks, B.StopSolverMasks.join

    def join(self, **kwargs):
        # The child the join would WAIT for. The real join of a launch that failed
        # and was confirmed has none, and returns at once; one with a child here
        # would wait the whole bound for a child that never runs (not run: True).
        joined.append(self._child)
        return real_join(self, **kwargs) if self._child is None else True

    monkeypatch.setattr(B.StopSolverMasks, "join", join)

    def factory(s, *, script=None):
        made.append(real(s, script=script, spawn=lambda *a, **k: process))
        return made[-1]

    monkeypatch.setattr(B, "StopSolverMasks", factory)
    s = solver(tmp_path)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        answer = B.wait_for_background_solve_at_stop(s, 0.2, should_stop=lambda: False,
                                                     stop_at=time.monotonic())
    assert answer is True, "a confirmed-dead child must leave the ordinary final solve"
    assert all(c is None for c in joined), "a child whose resume failed was joined"
    assert len(kills) == 1 and kills[0]["job"] is job and kills[0]["hard"]
    assert process.poll() == 1 and job.closed
    assert made[0]._child is None and made[0] not in B._owned_stalled_mask_children
    assert 0.0 < s.timeout <= 0.2 + 1e-6, "the background wait still ran"
    assert not (_live(tmp_path).root / M.STAGE_PARENT_DIRNAME).exists(), "marker not swept"
    assert any("could not be resumed; disabling it" in r.getMessage() for r in caplog.records)


@pytest.mark.skipif(os.name != "nt", reason="CREATE_SUSPENDED and Job Objects are Windows")
def test_a_real_child_whose_resume_fails_dies_in_its_job_without_running(tmp_path, monkeypatch):
    """L-1 with a REAL process, a REAL Job and the real termination: only the
    resume fails. `launch` answers False with the child confirmed dead, and its
    first line -- which would write a file -- never ran."""
    ran = tmp_path / "ran.txt"
    script = tmp_path / "stub_mask_child.py"
    script.write_text(f"import pathlib\npathlib.Path({str(ran)!r}).write_text('ran')\n",
                      encoding="utf-8")
    spawned, jobs, real_assign = [], [], B.assign_to_job
    monkeypatch.setattr(B, "assign_to_job", lambda p: jobs.append(real_assign(p)) or jobs[-1])
    monkeypatch.setattr(B, "_resume_suspended", lambda p: False)
    child = B.StopSolverMasks(solver(tmp_path), script=script,
                              spawn=lambda *a, **k: spawned.append(subprocess.Popen(*a, **k))
                              or spawned[-1])
    try:
        assert child.launch() is False
        assert len(jobs) == 1 and jobs[0] is not None, "no real Job: this proves nothing"
        assert jobs[0].closed
        assert child._child is None and child not in B._owned_stalled_mask_children
        (process,) = spawned
        assert process.poll() is not None, "the Job did not take the suspended child"
        time.sleep(0.5)
        assert not ran.exists(), "a child whose resume failed ran its own code"
        assert not (_live(tmp_path).root / M.STAGE_PARENT_DIRNAME).exists()
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait(10)


def test_close_trusts_the_process_not_terminate_trees_answer(tmp_path, monkeypatch):
    """C25f review LOW-4 (mutant V3): `terminate_tree` says gone, `poll()` says
    alive. The child is NOT confirmed: retained, its Job kept open, its stage kept."""
    process, job = Process(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: True)
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process)
    assert child.launch()
    stage = _write_stage(_live(tmp_path), {"a.jpg": "0" * 40})
    try:
        assert child.close() is False
        assert child in B._owned_stalled_mask_children
        assert not job.closed and child._child is process and stage.exists()
    finally:
        process.returncode = 1
        assert child.close()
    assert child not in B._owned_stalled_mask_children and job.closed
    assert not stage.parent.exists()


def test_launch_exception_leaves_background_wait_available(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "python_executable", lambda: (_ for _ in ()).throw(OSError("bad python")))
    s = solver(tmp_path)
    assert B.wait_for_background_solve_at_stop(s, 0.2, should_stop=lambda: False,
                                               stop_at=time.monotonic())
    assert 0.0 < s.timeout <= 0.2 + 1e-6


@pytest.mark.parametrize("confirmed", [True, False])
def test_a_join_that_raises_answers_only_what_close_confirms(tmp_path, monkeypatch, caplog,
                                                             confirmed):
    """C25f-FIX2 review L-9 (mutant V10): the join raises while the child still
    runs (here the stop channel itself breaks). The wrapper's exception branch
    ends the child and answers what `close` CONFIRMED: True only for a child
    confirmed dead. An unkillable one is False -- under a flipped RULING 2 that
    is what keeps the final solve from running beside it -- stays registered,
    and its Job stays open."""
    process, job, made = Process(), Job(), []
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: (
        setattr(p, "returncode", 1) or True) if confirmed else False)
    real = B.StopSolverMasks

    def factory(s, *, script=None):
        made.append(real(s, script=script, spawn=lambda *a, **k: process))
        return made[-1]

    def broken_stop_channel():
        raise RuntimeError("injected: the stop channel broke")

    monkeypatch.setattr(B, "StopSolverMasks", factory)
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            answer = B.wait_for_background_solve_at_stop(
                solver(tmp_path), 0.0, should_stop=broken_stop_channel,
                stop_at=time.monotonic())
        messages = [r.getMessage() for r in caplog.records]
        assert any("Stop mask prefill failed: injected: the stop channel broke" in m
                   for m in messages)
        assert answer is confirmed
        if confirmed:
            assert made[0]._child is None and job.closed
        else:
            assert made[0] in B._owned_stalled_mask_children and not job.closed
            assert any("could not be confirmed dead" in m for m in messages)
    finally:
        process.returncode = 1
        made[0].close()
    assert made[0] not in B._owned_stalled_mask_children


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


def test_the_join_bound_scales_with_the_solver_images(tmp_path, monkeypatch, caplog):
    """C25f review MED-1 (mutant V4, the 30-s default): the bound is
    max(1800 s, 4.0 x 0.538 s x images), from the images present at launch.
    0.538 s/image is walk 5's MEASURED mask rate; the factor must cover Stage
    0's worst measured slowdowns, 2.7x (a GPU neighbour) x 1.06x (the cap)."""
    assert B.STOP_MASKS_SECONDS_PER_IMAGE == 0.538
    assert B.STOP_MASKS_JOIN_FACTOR == 4.0 and B.STOP_MASKS_JOIN_FACTOR >= 2.7 * 1.06
    assert B.STOP_MASKS_JOIN_FLOOR_S == 1800.0
    assert B.stop_masks_join_bound(0) == 1800.0
    assert B.stop_masks_join_bound(836) == 1800.0
    assert B.stop_masks_join_bound(997) == pytest.approx(4.0 * 0.538 * 997)     # ~2146 s
    assert B.stop_masks_join_bound(5000) == pytest.approx(10760.0)
    # `launch` applies it to the images it finds; an explicit bound wins.
    ws = _live(tmp_path)
    ws.images_dir.mkdir(parents=True)
    for i in range(5):
        (ws.images_dir / f"{i:08d}.jpg").write_bytes(b"x")
    monkeypatch.setattr(B, "STOP_MASKS_JOIN_FLOOR_S", 0.0)
    monkeypatch.setattr(B, "assign_to_job", lambda p: Job())
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: Process(rc=1))
    assert child.join_timeout is None
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert child.launch()
    assert child.join_timeout == pytest.approx(4.0 * 0.538 * 5)
    assert any(r.getMessage().endswith("; join bound 11s for 5 solver images, or 120s without "
                                       "progress") for r in caplog.records)
    child.join(should_stop=lambda: False)
    fixed = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: Process(rc=1),
                              join_timeout=7.0)
    assert fixed.launch() and fixed.join_timeout == 7.0
    fixed.join(should_stop=lambda: False)


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


# -- the stall bound: a HUNG child does not hold the final for the join bound (L-2) --


def test_the_stall_bound_is_declared_beside_the_join_bound(tmp_path, monkeypatch, caplog):
    """C25f-FIX2 review L-2: max(120 s, 4.0 x 0.0024 s x images). 120 s is 55
    images' worth at the declared worst per-image rate (4.0 x 0.538 s) with no
    beat at all; 0.0024 s/image is the MEASURED cost of the passes that have no
    per-image beat (lookup + two read passes, walk 5). The same factor as the
    join bound, which it must stay well below."""
    assert B.STOP_MASKS_STALL_FLOOR_S == 120.0
    assert B.STOP_MASKS_SILENT_S_PER_IMAGE == 0.0024
    assert B.STOP_MASKS_PROGRESS_POLL_S == 5.0
    assert B.STOP_MASKS_STALL_FLOOR_S >= 50 * B.STOP_MASKS_JOIN_FACTOR * B.STOP_MASKS_SECONDS_PER_IMAGE
    assert B.stop_masks_stall_bound(0) == 120.0
    assert B.stop_masks_stall_bound(997) == 120.0                  # walk 5
    assert B.stop_masks_stall_bound(12_000) == 120.0
    assert B.stop_masks_stall_bound(20_000) == pytest.approx(4.0 * 0.0024 * 20_000)
    for images in (0, 997, 3_300, 20_000):
        assert B.stop_masks_stall_bound(images) < B.stop_masks_join_bound(images) / 10
    # `launch` applies it to the images it finds; an explicit bound wins.
    ws = _live(tmp_path)
    ws.images_dir.mkdir(parents=True)
    (ws.images_dir / "a.jpg").write_bytes(b"x")
    monkeypatch.setattr(B, "STOP_MASKS_STALL_FLOOR_S", 0.0)
    monkeypatch.setattr(B, "assign_to_job", lambda p: Job())
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: Process(rc=1))
    assert child.stall_timeout is None
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert child.launch()
    assert child.stall_timeout == pytest.approx(4.0 * 0.0024)
    assert any(r.getMessage().endswith(" for 1 solver images, or 0s without progress")
               for r in caplog.records)
    child.join(should_stop=lambda: False)
    fixed = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: Process(rc=1),
                              stall_timeout=7.0)
    assert fixed.launch() and fixed.stall_timeout == 7.0
    fixed.join(should_stop=lambda: False)


@pytest.mark.parametrize("behaviour", ["stalls", "keeps progressing"])
def test_a_child_without_progress_is_ended_at_the_stall_bound_not_the_join_bound(
        tmp_path, monkeypatch, caplog, behaviour):
    """C25f-FIX2 review L-2, on a frozen clock. The child beats every 30 s. One
    that STALLS -- its last beat at 570 s into the join, then silence, as in a
    CUDA or teardown hang -- is terminated through its Job at the STALL bound
    (120 s later, give or take one look at the stage), not at the 1800-s join
    bound, and handled exactly as a timeout: the files it finished are
    verified and promoted (RULING 3), and the record says "stalled". One that
    KEEPS PROGRESSING is never ended by the stall bound; only the join bound
    ends it."""
    clock = Clock()
    monkeypatch.setattr(B, "time", clock)
    ws = _live(tmp_path)
    ws.images_dir.mkdir(parents=True)
    _camera(ws)
    data = _jpeg(90)
    (ws.images_dir / "a.jpg").write_bytes(data)
    sha = hashlib.sha1(data).hexdigest()
    stage = ws.root / M.STAGE_PARENT_DIRNAME / M.STAGE_DIRNAME
    kills = []

    class Child(Process):
        beats = 0

        def poll(self):
            elapsed = clock.now - t0
            if (self.returncode is None and elapsed >= 30 * self.beats
                    and (behaviour == "keeps progressing" or elapsed < 600)):
                self.beats += 1
                (stage / M.PROGRESS_FILENAME).write_text(str(self.beats), encoding="ascii")
            return self.returncode

    def terminate(p, **kw):
        kills.append((clock.now - t0, kw["job"]))
        p.returncode = 1
        return True

    process, job = Child(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", terminate)
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process)
    assert child.launch()
    assert (child.join_timeout, child.stall_timeout) == (1800.0, 120.0)
    # What it finished before the join: a snapshot and image a's two cache files.
    (stage / "images").mkdir(parents=True)
    (stage / "images" / "a.jpg").write_bytes(data)
    good = sorted(_component(stage / "transients", "a.jpg", sha, component).name
                  for component in (SM.T.COMPONENT_GDSAM, SM.T.COMPONENT_ONEFORMER))
    t0 = clock.now
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert child.join(should_stop=lambda: False)
    (ended, by), = kills
    assert by is job and job.closed and process.poll() == 1
    assert not stage.parent.exists(), "the stage is swept"
    messages = [r.getMessage() for r in caplog.records]
    (record,) = _promotion_records(ws)
    assert sorted(p.name for p in (ws.root / "transients").glob("*.npz")) == good
    if behaviour == "stalls":
        last_beat = 570.0
        assert (last_beat + 120.0 <= ended
                <= last_beat + 120.0 + B.STOP_MASKS_PROGRESS_POLL_S + B.CHILD_POLL_S + 1e-6)
        assert ended < 1800.0 - 1000.0, "ended by the join bound, not the stall bound"
        assert any("join ended (no progress for 120s); terminating pid" in m
                   and m.endswith("only those are promoted") for m in messages)
        assert record["why"] == "stalled" and record["verified"] == 2
        assert ("[Tower][WorldBuilder] Stop mask prefill promoted 2 verified cache files from a "
                "child that stalled, for 1 snapshot images") in messages
    else:
        assert 1800.0 <= ended <= 1800.0 + B.CHILD_POLL_S + 1e-6
        assert process.beats > 1800 // 30, "it made progress throughout"
        assert any("join ended (timed out after 1800s); terminating pid" in m for m in messages)
        assert record["why"] == "timed out" and record["verified"] == 2


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
    # A complete, VALID stage for a live image, snapshot included: only the failed
    # termination keeps it from being promoted after the timeout (RULING 3).
    stage = _write_stage(ws, {"a.jpg": hashlib.sha1(b"pixels").hexdigest()},
                         snapshot={"a.jpg": b"pixels"})
    try:
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            assert child.join(should_stop=lambda: False) is False
        assert child in B._owned_stalled_mask_children
        assert not job.closed and child._child is process
        assert stage.exists(), "a child that may still write keeps its stage"
        assert not (ws.root / "transients").exists(), "nothing of it is promoted"
        assert any("could not be confirmed dead; it stays owned: its Job Object closes with "
                   "this builder" in r.getMessage() for r in caplog.records)
    finally:
        process.returncode = 1
        child.close()
    assert child not in B._owned_stalled_mask_children, "a confirmed child leaves the registry"
    assert not stage.parent.exists() and job.closed


def test_join_reports_a_child_that_died_after_a_failed_termination_as_gone(tmp_path,
                                                                           monkeypatch):
    """C25f review LOW-5: the join's termination is refused, and the child dies
    a moment later, before the join returns. The join's `finally` confirms and
    sweeps it -- and the answer is now True (it said False, so `main` logged
    "could not be confirmed dead" for a dead, swept child)."""

    class DiesLate(Process):
        polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls <= 2 else 1

    process, job = DiesLate(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: False)
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process, join_timeout=-1)
    assert child.launch()
    stage = _write_stage(_live(tmp_path), {"a.jpg": "0" * 40})
    assert child.join(should_stop=lambda: False) is True
    assert child._child is None and child not in B._owned_stalled_mask_children
    assert job.closed and not stage.parent.exists()


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


# -- RULING 3: a timed-out child's FINISHED files, verified (manager 154 §1) ---


def _jpeg(value, shape=SHAPE) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((*shape, 3), value, np.uint8))
    assert ok
    return encoded.tobytes()


def _promotion_records(ws) -> list:
    log = ws.root / B.STOP_MASKS_LOG_NAME
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    tag = "Stop mask prefill promotion record: "
    return [json.loads(line.split(tag, 1)[1]) for line in lines if tag in line]


@pytest.mark.parametrize("reason", ["timeout", "hard"])
def test_ruling_3_a_timed_out_child_promotes_only_verified_files(tmp_path, monkeypatch, caplog,
                                                                 reason):
    """RULING 3, at the DEFAULT. A join that TIMES OUT kills the child, then
    promotes only what it FINISHED and what VERIFIES for the live bytes (a):
    image a's two files. Quarantined, never promoted: a torn archive, a file
    under another key, a file of the wrong shape, a writer's `.tmp`. Image d
    was rebuilt after the snapshot, so its file names bytes that are gone: not
    a candidate at all. The set is recorded (b). A HARD stop promotes nothing."""
    assert B.STOP_MASKS_PROMOTE_PARTIAL_ON_TIMEOUT is True
    process, job = Process(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: setattr(p, "returncode", 1) or True)
    ws = _live(tmp_path)
    ws.images_dir.mkdir(parents=True)
    _camera(ws)
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process,
                              join_timeout=-1 if reason == "timeout" else 1800)
    assert child.launch()
    # What the child had done by the time it was killed.
    stage = ws.root / M.STAGE_PARENT_DIRNAME / M.STAGE_DIRNAME
    staged = stage / "transients"
    (stage / "images").mkdir(parents=True)
    staged.mkdir()
    sha = {}
    for i, name in enumerate(["a.jpg", "b.jpg", "c.jpg", "d.jpg"]):
        data = _jpeg(40 * i + 10)
        (ws.images_dir / name).write_bytes(data)
        (stage / "images" / name).write_bytes(data)
        sha[name] = hashlib.sha1(data).hexdigest()
    G, O = SM.T.COMPONENT_GDSAM, SM.T.COMPONENT_ONEFORMER
    good = [_component(staged, "a.jpg", sha["a.jpg"], G),
            _component(staged, "a.jpg", sha["a.jpg"], O)]
    torn = _component(staged, "b.jpg", sha["b.jpg"], G)
    torn.write_bytes(torn.read_bytes()[:40])
    other_key = _component(staged, "b.jpg", sha["b.jpg"], O, key=dict(
        SM.mask_key(O, SM.solver_params(), "b.jpg", sha["b.jpg"]), input_rule="another"))
    wrong_shape = _component(staged, "c.jpg", sha["c.jpg"], G, shape=(2, 8))
    stale = _component(staged, "d.jpg", sha["d.jpg"], G)
    tmp = staged / f"e.{'0' * 12}.gdsam.npz.p123.tmp"
    tmp.write_bytes(b"half an archive")
    (ws.images_dir / "d.jpg").write_bytes(_jpeg(250))       # rebuilt after the snapshot
    seen = {}
    sweep = child._sweep_stage

    def look_then_sweep():
        quarantine = stage / B.STOP_MASKS_QUARANTINE_DIRNAME
        seen.setdefault("quarantine", sorted(p.name for p in quarantine.iterdir())
                        if quarantine.exists() else [])
        seen.setdefault("left", sorted(p.name for p in staged.iterdir()))
        sweep()

    monkeypatch.setattr(child, "_sweep_stage", look_then_sweep)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert child.join(should_stop=lambda: reason == "hard")
    assert process.poll() == 1 and job.closed
    assert not stage.parent.exists(), "the stage is swept"
    live = ws.root / "transients"
    promoted = sorted(p.name for p in live.glob("*.npz")) if live.exists() else []
    messages = [r.getMessage() for r in caplog.records]
    if reason == "hard":
        assert promoted == [] and _promotion_records(ws) == []
        assert any("join ended (a stop was requested); terminating pid" in m
                   and m.endswith("nothing it staged is promoted") for m in messages)
        return
    assert promoted == sorted(p.name for p in good)
    for component in (G, O):
        assert SM.T.read_component(SM.component_path(live, "a.jpg", component, sha["a.jpg"]),
                                   SM.mask_key(component, SM.solver_params(), "a.jpg",
                                               sha["a.jpg"]), SHAPE) is not None
    assert seen["quarantine"] == sorted([torn.name, other_key.name, wrong_shape.name, tmp.name])
    assert seen["left"] == [stale.name], "a file for replaced bytes is no candidate"
    (record,) = _promotion_records(ws)
    assert record == {"why": "timed out", "state": "complete", "verified": 2,
                      "promoted": sorted(p.name for p in good), "error": None,
                      "quarantined": sorted([torn.name, other_key.name, wrong_shape.name,
                                             tmp.name])}
    assert any("join ended (timed out after -1s); terminating pid" in m
               and m.endswith("the cache files it finished are verified, and only those are "
                              "promoted") for m in messages)
    assert ("[Tower][WorldBuilder] Stop mask prefill promoted 2 verified cache files from a "
            "child that timed out, for 4 snapshot images; 4 staged files did not verify or were "
            "incomplete, and were quarantined, never promoted") in messages
    # The final solve reads a's files as hits and computes b, c and d.
    final = SM.ensure_solver_masks(ws, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"], shape=SHAPE,
                                   backend_factory=StubDetector(), device_probe=lambda: None)
    assert final.cache_hits == 1 and final.computed == 3 and len(final.masked) == 4


def test_ruling_3_flipped_a_timed_out_child_promotes_nothing(tmp_path, monkeypatch, caplog):
    """The other ruling is the one constant: C25f's discard on timeout."""
    monkeypatch.setattr(B, "STOP_MASKS_PROMOTE_PARTIAL_ON_TIMEOUT", False)
    process, job = Process(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: setattr(p, "returncode", 1) or True)
    ws = _live(tmp_path)
    ws.images_dir.mkdir(parents=True)
    (ws.images_dir / "a.jpg").write_bytes(b"pixels")
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process, join_timeout=-1)
    assert child.launch()
    stage = _write_stage(ws, {"a.jpg": hashlib.sha1(b"pixels").hexdigest()})
    (stage / "images").mkdir()
    (stage / "images" / "a.jpg").write_bytes(b"pixels")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert child.join(should_stop=lambda: False)
    assert not (ws.root / "transients").exists() and _promotion_records(ws) == []
    assert not stage.parent.exists()
    assert any(r.getMessage().endswith("nothing it staged is promoted") for r in caplog.records)


def _hung_child_with_a_finished_stage(tmp_path, monkeypatch, *, result=None, **bounds):
    """A launched child that never exits by itself, whose stage holds a snapshot of
    one live image and that image's two VALID cache files -- everything RULING 3
    would promote -- and, if `result` is given, a result with `available: result`.
    Returns (child, process, job, ws, stage, promotable names)."""
    process, job = Process(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: setattr(p, "returncode", 1) or True)
    monkeypatch.setattr(B, "CHILD_POLL_S", 0.001)
    ws = _live(tmp_path)
    ws.images_dir.mkdir(parents=True)
    (ws.images_dir / "a.jpg").write_bytes(b"pixels")
    sha = hashlib.sha1(b"pixels").hexdigest()
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process, **bounds)
    assert child.launch()
    stage = _write_stage(ws, {"a.jpg": sha}, available=bool(result),
                         snapshot={"a.jpg": b"pixels"})
    _component(stage / "transients", "a.jpg", sha, SM.T.COMPONENT_ONEFORMER)
    if result is None:
        (stage / M.RESULT_FILENAME).unlink()
    names = sorted(p.name for p in (stage / "transients").glob("*.npz"))
    return child, process, job, ws, stage, names


@pytest.mark.parametrize("bound", ["timeout", "stall"])
def test_a_hard_stop_at_the_timeout_instant_promotes_nothing(tmp_path, monkeypatch, caplog,
                                                             bound):
    """C25f-FIX2 review L-7 (mutant V8): the join times out -- or stalls -- at
    the very poll where a HARD stop arrives. The stop wins: nothing is promoted
    (`main` then skips the final solve), exactly as the RULING 3 comment says,
    although the same stage alone WOULD be promoted on a timeout."""
    child, process, job, ws, stage, _names = _hung_child_with_a_finished_stage(
        tmp_path, monkeypatch, join_timeout=-1 if bound == "timeout" else 1800,
        stall_timeout=-1 if bound == "stall" else 120)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert child.join(should_stop=lambda: True)
    assert process.poll() == 1 and job.closed
    assert not (ws.root / "transients").exists() and _promotion_records(ws) == []
    assert not stage.parent.exists()
    assert any("join ended (a stop was requested); terminating pid" in r.getMessage()
               and r.getMessage().endswith("nothing it staged is promoted")
               for r in caplog.records)


@pytest.mark.parametrize("how,result,promoted", [
    ("exited 1", False, False),        # the reference: a non-zero exit promotes nothing
    ("timed out", False, False),       # L-4 (i): reported incomplete, then hung
    ("stalled", False, False),
    ("timed out", True, True),         # a COMPLETE report, then a teardown hang: RULING 3
    ("timed out", None, True),         # no report at all: RULING 3
])
def test_a_child_that_reported_an_incomplete_pass_is_never_promoted_however_it_ends(
        tmp_path, monkeypatch, caplog, how, result, promoted):
    """C25f-FIX2 review L-4 (i). A child whose result says its pass was NOT
    complete exits 1 (`world_solve_masks.prefill`), and a non-zero exit promotes
    nothing. If that same child then HANGS instead -- in teardown, say -- the
    join ends it (a timeout or a stall), and it must promote nothing either,
    although every file it staged would verify. A child that reported a
    COMPLETE pass, or nothing at all, is promoted as RULING 3 says."""
    child, process, job, ws, stage, names = _hung_child_with_a_finished_stage(
        tmp_path, monkeypatch, result=result, join_timeout=-1 if how == "timed out" else 1800,
        stall_timeout=-1 if how == "stalled" else 120)
    if how == "exited 1":
        process.returncode = 1
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert child.join(should_stop=lambda: False)
    assert process.poll() == 1 and job.closed and not stage.parent.exists()
    live = sorted(p.name for p in (ws.root / "transients").glob("*.npz"))
    messages = [r.getMessage() for r in caplog.records]
    if promoted:
        assert live == names and len(names) == 2
        (record,) = _promotion_records(ws)
        assert record["why"] == how and record["promoted"] == names
    else:
        assert live == [] and _promotion_records(ws) == []
    reported = any("had reported an incomplete pass before it was terminated; as for a "
                   "non-zero exit, nothing it staged is promoted" in m for m in messages)
    assert reported is (how != "exited 1" and result is False)


@pytest.mark.parametrize("rc", [1, 0])
def test_a_child_that_exits_at_the_timeout_instant_is_judged_by_its_own_exit(
        tmp_path, monkeypatch, caplog, rc):
    """C25f-FIX2 review L-4 (ii). The child exits by itself between the join's
    poll and its timeout decision. It is not "terminated", and not promoted as
    a timeout: its own exit code decides, as for any exit -- 1 promotes nothing,
    0 promotes from its result."""

    class ExitsAtTheDeadline(Process):
        polls = 0

        def poll(self):
            self.polls += 1
            if self.polls >= 2 and self.returncode is None:
                self.returncode = rc
            return self.returncode

    kills = []
    child, _process, job, ws, stage, names = _hung_child_with_a_finished_stage(
        tmp_path, monkeypatch, result=rc == 0, join_timeout=-1)
    process = ExitsAtTheDeadline()
    child._child = process
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: kills.append(p) or True)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert child.join(should_stop=lambda: False)
    assert kills == [], "a child that had exited by itself was terminated as timed out"
    assert job.closed and not stage.parent.exists()
    messages = [r.getMessage() for r in caplog.records]
    assert not any("join ended" in m for m in messages)
    assert any(m.startswith(f"[Tower][WorldBuilder] Stop mask prefill exited {rc} in ")
               for m in messages)
    live = sorted(p.name for p in (ws.root / "transients").glob("*.npz"))
    if rc == 0:
        (record,) = _promotion_records(ws)
        assert record["why"] == "exited 0" and live == names
    else:
        assert live == [] and _promotion_records(ws) == []


def test_a_stop_at_the_instant_the_child_exits_0_still_promotes_nothing(tmp_path, monkeypatch,
                                                                        caplog):
    """The other side of L-4 (ii): the re-check that lets a child's own exit
    decide is for a TIMEOUT only. A hard stop that arrives as the child exits 0
    with a complete result promotes nothing (`main` skips the final solve)."""

    class ExitsAsTheStopArrives(Process):
        polls = 0

        def poll(self):
            self.polls += 1
            if self.polls >= 2 and self.returncode is None:
                self.returncode = 0
            return self.returncode

    child, _process, job, ws, stage, _names = _hung_child_with_a_finished_stage(
        tmp_path, monkeypatch, result=True)
    child._child = ExitsAsTheStopArrives()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert child.join(should_stop=lambda: True)
    assert job.closed and not stage.parent.exists()
    assert not list((ws.root / "transients").glob("*.npz")) and _promotion_records(ws) == []
    assert any("join ended (a stop was requested)" in r.getMessage() for r in caplog.records)


# -- ownership across process death -----------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_job_kills_mask_child_when_parent_dies(tmp_path):
    """A RUNNING child in a KILL_ON_JOB_CLOSE Job dies with the process that owns the Job.

    HARDENED AGAINST LOAD (C25f-FIX2 review L-5), proving the same thing. It flaked when
    the parent spawned `sys.executable` -- the venv LAUNCHER, which starts the real
    interpreter with silent breakaway -- and assigned the Job after: under load the
    sleeper escaped the Job and held the test's stdout pipe (`TimeoutExpired`). The child
    is now the one `StopSolverMasks.launch` makes: ONE process
    (`interpreter_executable`), created SUSPENDED, put in the Job, and only then resumed
    (`NtResumeProcess`, as `world_build_session._resume_suspended`). It must be RUNNING --
    it writes its started-file -- before the parent dies, it holds none of the parent's
    pipes (so an escape fails the assertion below instead of timing out), and the
    time limits allow for a loaded host."""
    import psutil

    pidfile, started = tmp_path / "child.pid", tmp_path / "child.started"
    parent = tmp_path / "parent.py"
    child_code = (f"import pathlib, time; pathlib.Path({str(started)!r}).write_text('ran'); "
                  "time.sleep(120)")
    parent.write_text(
        "import ctypes, os, pathlib, subprocess, sys, time\n"
        "from tower.process_ownership import (assign_to_job, interpreter_environment,\n"
        "                                     interpreter_executable)\n"
        f"p = subprocess.Popen([interpreter_executable(), '-c', {child_code!r}],\n"
        "                     env=interpreter_environment(), stdin=subprocess.DEVNULL,\n"
        "                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
        f"                     creationflags={B._CREATE_SUSPENDED})\n"
        "job = assign_to_job(p)\n"
        "if job is None: p.kill(); sys.exit(3)\n"
        "import psutil\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(\n"
        "    f'{p.pid} {psutil.Process(p.pid).create_time()!r}')\n"
        "ntdll = ctypes.WinDLL('ntdll')\n"
        "ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]\n"
        "ntdll.NtResumeProcess.restype = ctypes.c_long\n"
        "if ntdll.NtResumeProcess(ctypes.c_void_p(int(p._handle))) != 0: p.kill(); sys.exit(4)\n"
        "deadline = time.monotonic() + 90\n"
        f"while not pathlib.Path({str(started)!r}).exists():\n"
        "    if p.poll() is not None or time.monotonic() > deadline: sys.exit(5)\n"
        "    time.sleep(0.05)\n"
        "os._exit(0)\n", encoding="utf-8")
    completed = subprocess.run([sys.executable, str(parent)], timeout=150,
                               check=False, capture_output=True)
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    pid, created = pidfile.read_text().split()
    assert started.read_text() == "ran", "the child never ran: nothing proven for a running child"

    def child_alive() -> bool:
        # THAT process, not a later one the OS gave its pid to (also a load flake).
        try:
            return psutil.Process(int(pid)).create_time() == float(created)
        except psutil.Error:
            return False

    deadline = time.monotonic() + 30
    while child_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not child_alive(), "Job close left a mask child alive"


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


# -- a stage a dead builder left behind (C25x2 MED-3) --------------------------


def _dead_pid() -> int:
    import psutil

    pid = 2_147_483_644          # the largest pid psutil takes here (a C long), never live
    assert not psutil.pid_exists(pid)
    return pid


def _leftover_stage(store, world, session, pid=None) -> Path:
    """A dead builder's `stop_masks/`: its marker (if any), a snapshot image, a cache file."""
    ws = global_solve.workspace_for(store, world, session)
    stage = ws.root / M.STAGE_PARENT_DIRNAME / M.STAGE_DIRNAME
    (stage / "images").mkdir(parents=True)
    (stage / "images" / "00000000.jpg").write_bytes(b"snapshot")
    (stage / "transients").mkdir()
    (stage / "transients" / "00000000.0123456789ab.gdsam.npz").write_bytes(b"npz")
    if pid is not None:
        M.writer_marker(ws.root, pid).write_bytes(b"")
    return stage.parent


def test_stale_stages_are_swept_only_when_their_writer_is_gone(tmp_path, caplog):
    """The sweep follows the product's precedent for its own leavings
    (`storage.sweep_abandoned_staging`): a DEAD writer's stage is deleted; a
    live writer's (a child its builder could not confirm dead, RULING 2) and an
    unattributable one are left alone; a session with none is not touched."""
    from tower.world_builder.store import SESSION_FILENAME

    store = WorldStore(tmp_path)
    for session in ("dead", "live", "unmarked", "none"):
        folder = store.world_dir("w") / "sessions" / session
        folder.mkdir(parents=True)
        (folder / SESSION_FILENAME).write_text("{}", encoding="utf-8")
    dead = _leftover_stage(store, "w", "dead", _dead_pid())
    live = _leftover_stage(store, "w", "live", os.getpid())
    unmarked = _leftover_stage(store, "w", "unmarked")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert B.sweep_stale_stop_mask_stages(store, "w") == ["dead"]
    assert not dead.exists() and live.exists() and unmarked.exists()
    assert not (global_solve.workspace_for(store, "w", "none").root).exists()
    assert any("swept 1 stale Stop mask stage(s)" in r.getMessage() for r in caplog.records)
    assert B.sweep_stale_stop_mask_stages(store, "w") == []


def test_a_writer_pid_that_cannot_be_checked_is_treated_as_alive(tmp_path, monkeypatch):
    """C25f-FIX2 review L-8 (mutant V9): `_pid_alive` follows
    `solve_masks._pid_alive`'s rule, UNKNOWN IS ALIVE. When psutil cannot answer,
    the stage is KEPT -- a leak at worst -- and never swept from under a writer
    that may still be alive."""
    import psutil

    from tower.world_builder.store import SESSION_FILENAME

    store = WorldStore(tmp_path)
    folder = store.world_dir("w") / "sessions" / "s"
    folder.mkdir(parents=True)
    (folder / SESSION_FILENAME).write_text("{}", encoding="utf-8")
    pid = _dead_pid()
    stage = _leftover_stage(store, "w", "s", pid)

    def cannot_tell(_pid):
        raise psutil.AccessDenied(_pid)

    monkeypatch.setattr(psutil, "pid_exists", cannot_tell)
    assert B._pid_alive(pid) is True
    assert B.sweep_stale_stop_mask_stages(store, "w") == []
    assert stage.exists() and M.writer_pids(stage) == [pid]
    monkeypatch.undo()
    assert B.sweep_stale_stop_mask_stages(store, "w") == ["s"], "the dead writer's, once known"


def test_the_next_builder_start_sweeps_a_dead_builders_stage(tmp_path, monkeypatch):
    """The call site: a builder that died in its Stop left `stop_masks/` in its
    session; the next builder START for that world -- here with the switch off --
    sweeps it, holding the world's writer lock."""
    monkeypatch.setattr(B.StopRequest, "install", lambda self, **_: None)
    monkeypatch.delenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", raising=False)
    frames, root = _frames_dir(tmp_path), tmp_path / "worlds"
    assert B.main(["--frames", str(frames), "--root", str(root), "--format", "json"]) == 0
    store = WorldStore(root)
    (world,) = store.list_world_ids()
    (first,) = store.list_session_ids(world)
    left = _leftover_stage(store, world, first, _dead_pid())
    assert B.main(["--frames", str(frames), "--root", str(root), "--world", world,
                   "--format", "json"]) == 0
    assert len(store.list_session_ids(world)) == 2
    assert not left.exists(), "the next builder start did not sweep a dead builder's stage"


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


@pytest.mark.parametrize("failure", ["late failure", "CUDA out of memory",
                                     "CUDA out of memory once, then success"])
def test_partial_detector_failure_is_quarantined(images, monkeypatch, failure, tmp_path):
    """C25x M5: the child fails AFTER its first component wrote. Nothing is
    promoted, the stage is swept, and the final solve's record, PNGs and cache
    are exactly those of a final solve that never had a child. The third case
    (C25f review LOW-3, mutant V1) SUCCEEDS on its one OOM retry -- over a
    different item set -- and is quarantined all the same: `retries` makes the
    pass incomplete."""
    root, _store, world, session, ws = images
    old = global_solve.SolveWorkspace(tmp_path / "old")
    shutil.copytree(ws.root, old.root)          # the switch-off twin, before the child
    good = StubDetector()

    def factory(component):
        backend = good(component)
        if component == SM.T.COMPONENT_ONEFORMER:
            run, calls = backend.run, []

            def failing(*a, **k):
                calls.append(1)
                if failure.endswith("then success") and len(calls) > 1:
                    return run(*a, **k)
                raise RuntimeError(failure)

            backend.run = failing
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


@pytest.mark.parametrize("camera", ["missing", "unreadable"])
def test_no_live_camera_at_promotion_promotes_nothing_unverified(tmp_path, caplog, camera):
    """C25f-FIX2 review L-6 (mutant V7): a complete, VALID stage from an exit 0,
    but the parent cannot read the live solver camera when it promotes. With no
    expected shape nothing can be verified (manager 154 (a)), so NOTHING is
    promoted: every candidate is quarantined and recorded, and the final solve
    computes it."""
    ws = _live(tmp_path)
    ws.images_dir.mkdir(parents=True)
    (ws.images_dir / "a.jpg").write_bytes(b"pixels")
    stage = _write_stage(ws, {"a.jpg": hashlib.sha1(b"pixels").hexdigest()})
    candidate = next((stage / "transients").glob("*.npz")).name
    if camera == "missing":
        ws.camera_path.unlink()
    else:
        ws.camera_path.write_text("{", encoding="utf-8")
    child = B.StopSolverMasks(solver(tmp_path))
    child._child, child._started = Process(rc=0), time.monotonic()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert child.join(should_stop=lambda: False)
    assert not list((ws.root / "transients").glob("*.npz")), "promoted without verification"
    (record,) = _promotion_records(ws)
    assert record == {"why": "exited 0", "state": "complete", "verified": 0, "promoted": [],
                      "quarantined": [candidate], "error": None}
    assert not stage.parent.exists()
    assert any("1 staged files did not verify or were incomplete, and were quarantined, never "
               "promoted" in r.getMessage() for r in caplog.records)


def _decoded_cache(cache: Path) -> dict:
    out = {}
    for path in sorted(cache.glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            key = json.loads(bytes(z["key"]).decode())
            key.pop("computed_at")
            key.pop("seconds")
            out[path.name] = (z["hand"].tobytes(), z["phone"].tobytes(), z["shape"].tobytes(),
                              key)
    return out


def test_a_promotion_that_stops_halfway_is_recorded_and_harmless(images, monkeypatch, tmp_path,
                                                                  caplog):
    """C25x2 MED-2 / RULING 3 (b). The third of four promotions fails (a full
    disk). The first two stay promoted -- each VERIFIED and whole (one atomic
    replace each) -- and the partial set is RECORDED. The final solve reads them
    as hits, computes the rest, and ends with the SAME masks, PNG bytes and
    cache arrays as a final solve that never had a child; only its counters
    differ (one image cached instead of computed)."""
    root, _store, world, session, ws = images
    old = global_solve.SolveWorkspace(tmp_path / "old")
    shutil.copytree(ws.root, old.root)          # the switch-off twin, before the child
    assert M.prefill(root, world, session) == 0
    live = ws.root / "transients"
    real_replace, calls, armed = os.replace, [], [True]

    def replace(src, dst):
        if armed[0] and Path(dst).parent == live:
            calls.append(Path(dst).name)
            if len(calls) == 3:
                raise OSError(28, "injected: no space left on device")
        return real_replace(src, dst)

    monkeypatch.setattr(B.os, "replace", replace)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert _join(images, 0)
    armed[0] = False
    promoted = sorted(p.name for p in live.glob("*.npz"))
    assert promoted == sorted(calls[:2]) and len(promoted) == 2
    assert not (ws.root / M.STAGE_PARENT_DIRNAME).exists(), "the stage is swept"
    (record,) = _promotion_records(ws)
    assert record["state"] == "partial" and record["why"] == "exited 0"
    assert record["verified"] == 4 and sorted(record["promoted"]) == promoted
    assert "injected: no space left on device" in record["error"]
    assert any("promotion stopped after 2 of 4 verified cache files" in r.getMessage()
               and "the final solve computes the rest" in r.getMessage() for r in caplog.records)
    for name in promoted:                          # each one individually valid
        stem, sha12, component, _npz = name.split(".")
        image = f"{stem}.jpg"
        full = hashlib.sha1((ws.images_dir / image).read_bytes()).hexdigest()
        assert full.startswith(sha12)
        assert SM.T.read_component(live / name, SM.mask_key(component, SM.solver_params(),
                                                            image, full), (48, 64)) is not None
    records = []
    for workspace in (ws, old):
        final = SM.ensure_solver_masks(workspace, ["00000000.jpg", "00000001.jpg"],
                                       shape=(48, 64), backend_factory=StubDetector(),
                                       device_probe=lambda: None)
        records.append(final.record())
    assert (records[0]["cache_hits"], records[0]["computed"]) == (1, 1)
    assert (records[1]["cache_hits"], records[1]["computed"]) == (0, 2)
    for record in records:
        for counter in ("cache_hits", "computed", "seconds"):
            record.pop(counter)
    assert records[0] == records[1] and records[0]["state"] == "applied"
    assert _files(ws.root / "masks") == _files(old.root / "masks")
    for png in (ws.root / "masks").glob("*.png"):
        assert png.read_bytes() == (old.root / "masks" / png.name).read_bytes()
    assert _decoded_cache(live) == _decoded_cache(old.root / "transients")


def test_no_camera_at_stop_is_a_clean_exit_that_promotes_nothing(tmp_path, monkeypatch, caplog):
    """C25f review LOW-6: no solver camera yet at Stop. The child says so in a
    result like any other exit 0 -- nothing to promote -- instead of exiting 0
    with no result, which the parent logged as a failed prefill."""
    real, made = B.StopSolverMasks, []

    def spawn(*a, **k):
        assert M.prefill(tmp_path, "w", "s") == 0        # the child's own code
        return Process(rc=0)

    def factory(s, *, script=None):
        made.append(real(s, script=script, spawn=spawn))
        return made[-1]

    monkeypatch.setattr(B, "StopSolverMasks", factory)
    monkeypatch.setattr(B, "assign_to_job", lambda p: Job())
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert B.wait_for_background_solve_at_stop(solver(tmp_path), 0.0,
                                                   should_stop=lambda: False,
                                                   stop_at=time.monotonic())
    messages = [r.getMessage() for r in caplog.records]
    assert made and not any("prefill failed" in m for m in messages)
    assert ("[Tower][WorldBuilder] Stop mask prefill promoted 0 cache files: 0 images masked, "
            "0 cache hits in the child; 0 of 0 images still match at the final solve") in messages
    assert not (_live(tmp_path).root / M.STAGE_PARENT_DIRNAME).exists()
    assert not (_live(tmp_path).root / "transients").exists()


# -- the child's progress beat (L-2, the child's half) --------------------------


class _CheckingDetector(SM.T.ComponentBackend):
    """Like the real backends: `should_stop` is checked before every image, and the
    answer is honoured. `gdsam` marks a rectangle, `oneformer` nothing."""

    answers = []

    def __init__(self, component):
        self.component = component

    def probe(self):
        return None

    def run(self, items, params, emit, should_stop=None):
        for i, rgb, _unobserved in items:
            answer = should_stop() if should_stop is not None else None
            _CheckingDetector.answers.append(answer)
            if answer:
                return {"stopped": True}
            hand = np.zeros(rgb.shape[:2], bool)
            if self.component == SM.T.COMPONENT_GDSAM:
                hand[4:20, 8:30] = True
            emit(i, hand, np.zeros(rgb.shape[:2], bool), 0.001)
        return {"frames": len(items)}


def test_the_child_beats_per_image_and_per_mask_and_computes_the_same_masks(images, monkeypatch,
                                                                            tmp_path):
    """The child counts every image each model checks in (`should_stop`) and
    every mask it emits, into `stop_masks/s/progress`, which the join watches.
    The backend's `should_stop` always answers "carry on" (`ensure_solver_masks`
    passes none), so the masks, keys and PNGs are exactly those of the same
    detector run WITHOUT the beat; and the beat file is swept with the stage."""
    root, _store, world, session, ws = images
    twin = global_solve.SolveWorkspace(tmp_path / "twin")
    shutil.copytree(ws.root, twin.root)              # the same images, no beat
    monkeypatch.setattr(SM.T, "BACKEND_FACTORY", _CheckingDetector)
    monkeypatch.setattr(M, "PROGRESS_BEAT_S", 0.0)   # write every beat
    monkeypatch.setattr(_CheckingDetector, "answers", [])
    assert M.prefill(root, world, session) == 0
    stage = ws.root / M.STAGE_PARENT_DIRNAME / M.STAGE_DIRNAME
    # 1 at the stage's creation; 2 images x 2 models checked in; 2 x 2 masks emitted.
    assert (stage / M.PROGRESS_FILENAME).read_text(encoding="ascii") == str(1 + 4 + 4)
    assert _CheckingDetector.answers == [False] * 4, "the backend was told to stop"
    _CheckingDetector.answers.clear()
    names = ["00000000.jpg", "00000001.jpg"]
    direct = SM.ensure_solver_masks(twin, names, shape=(48, 64),
                                    backend_factory=_CheckingDetector)
    assert direct.computed == 2 and _CheckingDetector.answers == [None] * 4
    assert _decoded_cache(stage / "transients") == _decoded_cache(twin.root / "transients")
    assert _files(stage / "masks") == _files(twin.root / "masks")
    for png in (stage / "masks").glob("*.png"):
        assert png.read_bytes() == (twin.root / "masks" / png.name).read_bytes()
    assert _join(images, 0)
    assert not (ws.root / M.STAGE_PARENT_DIRNAME).exists(), "the beat is swept with the stage"


def test_the_beat_is_rate_limited_and_never_fails_the_pass(tmp_path, monkeypatch):
    """At most one write per PROGRESS_BEAT_S (1 s): a beat costs one small write
    a second, not one per image. A beat that cannot be written is skipped."""
    clock = Clock()
    monkeypatch.setattr(M, "time", clock)
    assert M.PROGRESS_BEAT_S == 1.0
    path = tmp_path / "progress"
    beat = M._ProgressBeat(path)
    beat()
    assert path.read_text() == "1"
    clock.sleep(0.5)
    beat()
    assert path.read_text() == "1" and beat.count == 2
    clock.sleep(0.5)
    beat()
    assert path.read_text() == "3"
    lost = M._ProgressBeat(tmp_path / "no such directory" / "progress")
    lost()
    assert lost.count == 1 and not (tmp_path / "no such directory").exists()


def test_the_beating_backend_honours_a_callers_stop(tmp_path, monkeypatch):
    """The wrapper answers exactly what the caller's `should_stop` says -- here
    "stop" at the second image -- and still beats for each check."""
    monkeypatch.setattr(_CheckingDetector, "answers", [])
    beat = M._ProgressBeat(tmp_path / "progress")
    wrapped = M._beating(_CheckingDetector, beat)(SM.T.COMPONENT_GDSAM)
    assert wrapped.component == SM.T.COMPONENT_GDSAM and wrapped.probe() is None
    calls, emitted = [], []
    items = [(i, np.zeros((4, 8, 3), np.uint8), None) for i in range(3)]
    out = wrapped.run(items, None, lambda *a, **k: emitted.append(a[0]),
                      should_stop=lambda: calls.append(1) or len(calls) >= 2)
    assert out == {"stopped": True} and emitted == [0] and beat.count == 3
    assert _CheckingDetector.answers == [False, True]


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

    WHAT IT PROVES -- PLUMBING, NOT MODEL IDENTITY (C25f review LOW-7): the
    cap is IN FORCE in the child's process (env caps 2, torch and cv2 at 2)
    and NOT in the other (more than 2 threads on this host); and the child's
    own path -- the snapshot, the decode, the shape check, the cache write, the
    composition and the PNG encode -- is deterministic across two fresh
    processes, one per thread count.

    WHAT IT DOES NOT PROVE: that anything thread-count-DEPENDENT is identical
    under the cap. Its stand-in detector's kernels (cubic and area resize, a
    Gaussian blur, bilinear interpolation, a per-pixel channel mean) compute
    each output element independently, so threads only partition the output:
    they cannot show a difference even where one exists. There is no
    reduction and no GEMM here, and that is where thread counts change bits
    (Stage 0: DINOv2 on the CPU, 0/320 rows identical under this cap). Nor
    does it say anything about the real models (Grounding DINO, SAM 2.1,
    OneFormer). The identity evidence for those is Stage 0 row T (STAGE0.md
    §6.6: 200/200 arrays, 50/50 PNGs under this exact cap) and, through this
    child's own code, `RUN/experiments/C25f-FINISH/cap_identity_real.py` with
    its same-condition control arm -- which manager 154 §1 (c) requires to
    pass before any enable claim."""
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
