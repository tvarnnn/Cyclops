"""TOWER_WORLD_MERGE_AFTER_STOP: one non-blocking poll for a landed solve after the frame loop."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.world_build_session import (  # noqa: E402
    BackgroundSolver, merge_landed_solve_after_stop,
)
from tower.config import WORLD_MERGE_AFTER_STOP_ENV, world_merge_after_stop_setting  # noqa: E402


class FakeChild:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.pid = 1
        self.waited = False

    def poll(self):
        return self.returncode

    def wait(self, *a, **k):  # a blocking wait would be a bug here
        self.waited = True
        raise AssertionError("merge-after-stop must never wait")


class FakeEngine:
    def __init__(self, fail=False):
        self.builds = []
        self.fail = fail

    def build(self, world_id, session_id):
        self.builds.append((world_id, session_id))
        if self.fail:
            raise OSError("locked")
        return SimpleNamespace(keyframes=5, poses_solved=5, points=9)


def solver_with(child):
    solver = BackgroundSolver(root=Path("."), world_id="w", session_id="s", every=10,
                              capture_dirs=[])
    solver._child = child
    return solver


def test_finished_solve_is_merged_once():
    solver, engine = solver_with(FakeChild(returncode=0)), FakeEngine()
    assert merge_landed_solve_after_stop(solver, engine, "w", "s", 20) is True
    assert engine.builds == [("w", "s")]
    # Already merged: a second poll (or the in-loop one) finds nothing.
    assert merge_landed_solve_after_stop(solver, engine, "w", "s", 20) is False
    assert solver.finished() is False
    assert engine.builds == [("w", "s")]


def test_unfinished_solve_is_not_waited_for_or_merged():
    child = FakeChild(returncode=None)
    solver, engine = solver_with(child), FakeEngine()
    assert merge_landed_solve_after_stop(solver, engine, "w", "s", 20) is False
    assert engine.builds == [] and not child.waited
    assert solver.running  # still running, untouched


def test_already_merged_in_loop_is_not_remerged():
    solver, engine = solver_with(FakeChild(returncode=0)), FakeEngine()
    assert solver.finished() is True  # the in-loop poll consumed it
    assert merge_landed_solve_after_stop(solver, engine, "w", "s", 20) is False
    assert engine.builds == []


def test_hard_stop_too_few_keyframes_and_no_solver_do_nothing():
    engine = FakeEngine()
    assert not merge_landed_solve_after_stop(
        solver_with(FakeChild(0)), engine, "w", "s", 20, hard_stop=True)
    assert not merge_landed_solve_after_stop(solver_with(FakeChild(0)), engine, "w", "s", 1)
    assert not merge_landed_solve_after_stop(None, engine, "w", "s", 20)
    assert engine.builds == []


def test_build_failure_is_swallowed():
    assert merge_landed_solve_after_stop(
        solver_with(FakeChild(0)), FakeEngine(fail=True), "w", "s", 20) is False


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("off", False), ("garbage", False), ("on", True),
])
def test_switch_default_off(monkeypatch, value, expected):
    monkeypatch.delenv(WORLD_MERGE_AFTER_STOP_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(WORLD_MERGE_AFTER_STOP_ENV, value)
    assert world_merge_after_stop_setting() is expected
