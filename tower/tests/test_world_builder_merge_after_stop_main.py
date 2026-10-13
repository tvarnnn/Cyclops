"""TOWER_WORLD_MERGE_AFTER_STOP through the builder's real `main` (cand5 test gap).

`test_world_builder_merge_after_stop.py` pins the helper on its own. This drives the
whole lifecycle -- a synthetic walk through `wbs.main` with the real engine, the real
`StopRequest` and the real finalization record -- with a SCRIPTED solver in place of
`BackgroundSolver`, so the ordering of everything after the frame loop is observable:

  * a solve that landed after the last in-loop poll is merged exactly once, AFTER the
    frame loop and BEFORE the final solve launches -- with `TOWER_WORLD_FINAL_IMMEDIATE`
    on and off, each keeping today's own wait / stop_background step;
  * nothing is merged when nothing landed or a solve is still running, and the
    post-loop trace is then today's (the only addition is one `finished()` poll; no
    wait is added, so the final is not delayed);
  * no interim build lands after the final solve, and no landing is built twice;
  * an exception in the interim build does not block the final;
  * a hard stop skips the merge; a hard stop that arrives DURING the merge build is not
    cancelled (review HIGH-2), but the final solve is then skipped and the session
    still closes with its record and final build;
  * switch off == today: no post-loop poll and no build at all.

No child process, no GPU, no Tower.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_build_session as wbs  # noqa: E402
from scripts.world_build_session import StopRequest  # noqa: E402
from tower.config import WORLD_MERGE_AFTER_STOP_ENV  # noqa: E402
from tower.world_builder.engine import WorldBuilderEngine  # noqa: E402
from tower.world_builder.records import (  # noqa: E402
    FINAL_SOLVE_SKIPPED,
    FINALIZATION_COMPLETE,
)
from tower.world_builder.store import WorldStore  # noqa: E402

FINAL_IMMEDIATE_ENV = "TOWER_WORLD_FINAL_IMMEDIATE"
SOLVE_WAIT = 7.5
# The originals, so a test that walks twice never wraps its own wrapper.
_REAL_BOUNDED = StopRequest.bounded
_REAL_BUILD = WorldBuilderEngine.build

LOOP_END = "loop-end"
POLL_AFTER = "finished-after-loop"
BUILD = "build"
RUN_FINAL = "run_final"


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(wbs, "prewarm_world_builder", lambda *a, **k: ())
    monkeypatch.setattr(wbs.StopRequest, "install", lambda self, **k: None)
    for name in (WORLD_MERGE_AFTER_STOP_ENV, FINAL_IMMEDIATE_ENV, "TOWER_WORLD_POSE_SANITY",
                 "TOWER_WORLD_SURFACE_FAR_DEPTH_CAP", "TOWER_WORLD_FINISH_STAGES"):
        monkeypatch.delenv(name, raising=False)


class ScriptedSolver:
    """Stands in for `BackgroundSolver`. `finished()` answers from a script:
    `in_loop` is the set of in-loop poll numbers (1-based) that report a landing;
    `after_stop` says whether the first poll after the frame loop does."""

    def __init__(self, events, *, in_loop=(), after_stop=False, running=False, **_kw):
        self.events = events
        self.in_loop = set(in_loop)
        self.after_stop = after_stop
        self._running = running
        self.polls = 0
        self.launches = 0
        self.final_immediate_fallback = False

    @property
    def running(self):
        return self._running

    def finished(self):
        if LOOP_END in self.events:
            self.events.append(POLL_AFTER)
            landed, self.after_stop = self.after_stop, False
            if landed:
                self._running = False
            return landed
        self.polls += 1
        self.events.append("finished")
        return self.polls in self.in_loop

    def maybe_launch(self, store, accepted, sources):
        self.events.append("maybe_launch")
        return False

    def stop_background(self, store):
        self.events.append("stop_background")
        self._running = False

    def wait(self, seconds, should_stop=None):
        self.events.append(f"wait({seconds:g})")
        self._running = False

    def run_final(self, store, sources, *, should_stop=None, **kwargs):
        self.events.append(RUN_FINAL)
        return {"attempted": True, "solved": False, "reason": "scripted"}

    def close(self):
        self.events.append("close")


def _walk(tmp_path, monkeypatch, *, merge=True, final_immediate=False, in_loop=(),
          after_stop=False, running=False, build_hook=None, hard_stop_after_frames=None,
          frames=10, extra_args=(), events=None):
    """One synthetic walk through `wbs.main`. Returns (events, exit code, store, ids)."""
    if merge:
        monkeypatch.setenv(WORLD_MERGE_AFTER_STOP_ENV, "on")
    if final_immediate:
        monkeypatch.setenv(FINAL_IMMEDIATE_ENV, "on")
    events = [] if events is None else events
    state = {"stop": None, "solver": None}

    def make_solver(**kw):
        state["solver"] = ScriptedSolver(events, in_loop=in_loop, after_stop=after_stop,
                                         running=running, **kw)
        return state["solver"]

    monkeypatch.setattr(wbs, "BackgroundSolver", make_solver)

    real_bounded = _REAL_BOUNDED

    def bounded(self, frames_, **kw):
        state["stop"] = self
        n = 0
        for frame in real_bounded(self, frames_, **kw):
            n += 1
            if hard_stop_after_frames is not None and n > hard_stop_after_frames:
                self.request(StopRequest.HARD, "test-hard")
            yield frame
        events.append(LOOP_END)

    monkeypatch.setattr(StopRequest, "bounded", bounded)

    real_build = _REAL_BUILD

    def build(self, world_id, session_id):
        events.append(BUILD)
        if build_hook is not None:
            build_hook(events, state)
        return real_build(self, world_id, session_id)

    monkeypatch.setattr(WorldBuilderEngine, "build", build)

    root = tmp_path / "wb"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = wbs.main(["--synthetic", "--synthetic-frames", str(frames), "--root", str(root),
                         "--solve", "--solve-every", "3",
                         "--solve-wait-seconds", str(SOLVE_WAIT), "--format", "json",
                         *extra_args])
    store = WorldStore(root)
    (world_id,) = store.list_world_ids()
    (session_id,) = store.list_session_ids(world_id)
    return events, code, store, world_id, session_id


def _after_loop(events):
    return events[events.index(LOOP_END):]


def _today_wait(final_immediate):
    return "stop_background" if final_immediate else f"wait({SOLVE_WAIT:g})"


# ---------------------------------------------------------------------------
# a landing after Stop is merged once, before the final
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("final_immediate", [False, True], ids=["fi-off", "fi-on"])
def test_a_solve_that_landed_after_stop_is_merged_once_before_the_final(
        tmp_path, monkeypatch, final_immediate):
    events, code, store, w, s = _walk(tmp_path, monkeypatch, final_immediate=final_immediate,
                                      after_stop=True)
    assert code == 0
    tail = _after_loop(events)
    # exactly: one poll, the interim build, today's wait/stop step, the final, its build
    assert tail == [LOOP_END, POLL_AFTER, BUILD, _today_wait(final_immediate), RUN_FINAL,
                    BUILD, "close"], tail
    # no build in the loop (nothing landed there): the merge build is the only interim one
    assert events.count(BUILD) == 2
    assert store.read_session(w, s).finalization["state"] == FINALIZATION_COMPLETE


@pytest.mark.parametrize("final_immediate", [False, True], ids=["fi-off", "fi-on"])
def test_no_interim_build_lands_after_the_final_and_no_landing_is_built_twice(
        tmp_path, monkeypatch, final_immediate):
    """A landing consumed IN the loop, and a second one after Stop: two interim builds,
    one per landing, both before `run_final`; nothing after the final build."""
    events, code, *_ = _walk(tmp_path, monkeypatch, final_immediate=final_immediate,
                             in_loop={2}, after_stop=True)
    assert code == 0
    final_at = events.index(RUN_FINAL)
    builds = [i for i, e in enumerate(events) if e == BUILD]
    assert len(builds) == 3, events          # in-loop landing, post-Stop landing, final
    assert builds[0] < events.index(LOOP_END) < builds[1] < final_at < builds[2]
    assert BUILD not in events[builds[2] + 1:]
    assert events.count(POLL_AFTER) == 1


def test_a_landing_consumed_in_the_loop_is_not_merged_again(tmp_path, monkeypatch):
    """The in-loop poll on the last keyframe consumed the landing; the post-loop poll
    finds nothing and builds nothing."""
    events, code, *_ = _walk(tmp_path, monkeypatch, in_loop=set(range(1, 50)),
                             after_stop=False)
    assert code == 0
    tail = _after_loop(events)
    assert tail == [LOOP_END, POLL_AFTER, _today_wait(False), RUN_FINAL, BUILD, "close"], tail


# ---------------------------------------------------------------------------
# nothing landed / still running: nothing merged, today's trace, not delayed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("final_immediate", [False, True], ids=["fi-off", "fi-on"])
@pytest.mark.parametrize("running", [False, True], ids=["none-finished", "still-running"])
def test_nothing_landed_merges_nothing_and_is_todays_trace(tmp_path, monkeypatch,
                                                           final_immediate, running):
    off, code_off, *_ = _walk(tmp_path / "off", monkeypatch, merge=False,
                              final_immediate=final_immediate, running=running)
    on, code_on, *_ = _walk(tmp_path / "on", monkeypatch, merge=True,
                            final_immediate=final_immediate, running=running)
    assert code_off == code_on == 0
    # The only difference is the one non-blocking poll; no build, no extra wait.
    assert [e for e in on if e != POLL_AFTER] == off
    assert on.count(POLL_AFTER) == 1
    tail = _after_loop(on)
    assert tail == [LOOP_END, POLL_AFTER, _today_wait(final_immediate), RUN_FINAL, BUILD,
                    "close"], tail
    # The wait (fi-off) is today's own bounded wait, not a second one.
    assert sum(e.startswith("wait(") for e in on) == sum(e.startswith("wait(") for e in off)


@pytest.mark.parametrize("final_immediate", [False, True], ids=["fi-off", "fi-on"])
def test_switch_off_is_today_even_when_a_solve_landed(tmp_path, monkeypatch, final_immediate):
    events, code, *_ = _walk(tmp_path, monkeypatch, merge=False,
                             final_immediate=final_immediate, after_stop=True)
    assert code == 0
    assert POLL_AFTER not in events
    assert _after_loop(events) == [LOOP_END, _today_wait(final_immediate), RUN_FINAL, BUILD,
                                   "close"]


@pytest.mark.parametrize("value", ["", "off", "0", "garbage"])
def test_any_off_spelling_adds_nothing(tmp_path, monkeypatch, value):
    monkeypatch.setenv(WORLD_MERGE_AFTER_STOP_ENV, value)
    events, code, *_ = _walk(tmp_path, monkeypatch, merge=False, after_stop=True)
    assert code == 0 and POLL_AFTER not in events and events.count(BUILD) == 1


def test_final_solve_only_walks_never_poll(tmp_path, monkeypatch):
    """`--solve-every 0`: no background solves, so the switch has nothing to merge and
    the helper is not even called."""
    monkeypatch.setenv(WORLD_MERGE_AFTER_STOP_ENV, "on")
    events: list[str] = []
    monkeypatch.setattr(wbs, "BackgroundSolver",
                        lambda **kw: ScriptedSolver(events, after_stop=True))
    root = tmp_path / "wb"
    with contextlib.redirect_stdout(io.StringIO()):
        assert wbs.main(["--synthetic", "--synthetic-frames", "8", "--root", str(root),
                         "--solve", "--solve-every", "0", "--format", "json"]) == 0
    assert "finished" not in events and RUN_FINAL in events


# ---------------------------------------------------------------------------
# failure and stops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("final_immediate", [False, True], ids=["fi-off", "fi-on"])
@pytest.mark.parametrize("error", [RuntimeError("boom"), OSError("locked"),
                                   ValueError("bad solution")],
                         ids=["runtime", "oserror", "valueerror"])
def test_an_exception_in_the_interim_merge_does_not_block_the_final(
        tmp_path, monkeypatch, final_immediate, error):
    def fail_the_merge_build(events, state):
        if LOOP_END in events and RUN_FINAL not in events:
            raise error

    events, code, store, w, s = _walk(tmp_path, monkeypatch, final_immediate=final_immediate,
                                      after_stop=True, build_hook=fail_the_merge_build)
    assert code == 0
    tail = _after_loop(events)
    assert tail == [LOOP_END, POLL_AFTER, BUILD, _today_wait(final_immediate), RUN_FINAL,
                    BUILD, "close"], tail
    fin = store.read_session(w, s).finalization
    assert fin["state"] == FINALIZATION_COMPLETE


def test_a_hard_stop_skips_the_merge(tmp_path, monkeypatch):
    events, code, store, w, s = _walk(tmp_path, monkeypatch, after_stop=True,
                                      hard_stop_after_frames=4)
    assert code == 0
    tail = _after_loop(events)
    # the helper returns before polling; the final solve is skipped as today
    assert POLL_AFTER not in tail and RUN_FINAL not in tail
    assert tail.count(BUILD) == 1             # only the final build
    assert store.read_session(w, s).finalization["final_solve"] == FINAL_SOLVE_SKIPPED


def test_a_hard_stop_during_the_merge_build_is_not_cancelled_but_skips_the_final(
        tmp_path, monkeypatch):
    """Review HIGH-2, CURRENT behaviour: the interim build has no cancellation path, so a
    hard stop arriving inside it waits for it to finish (its duration is added to Stop). The
    session then sees the hard stop: the final solve is skipped, the final build still runs,
    and the record is closed normally -- in-process nothing is lost. (Whether the Tower's
    own kill grace outlasts a slow real build is not testable here.)"""
    def hard_stop_inside_the_merge(events, state):
        if LOOP_END in events and RUN_FINAL not in events and "merge-done" not in events:
            events.append("merge-done")
            state["stop"].request(StopRequest.HARD, "test-hard-during-merge")

    events, code, store, w, s = _walk(tmp_path, monkeypatch, after_stop=True,
                                      build_hook=hard_stop_inside_the_merge)
    assert code == 0
    tail = _after_loop(events)
    assert tail[:4] == [LOOP_END, POLL_AFTER, BUILD, "merge-done"], tail
    assert RUN_FINAL not in tail and tail.count(BUILD) == 2
    fin = store.read_session(w, s).finalization
    assert fin["state"] == FINALIZATION_COMPLETE
    assert fin["final_solve"] == FINAL_SOLVE_SKIPPED
    assert "hard stop" in (fin.get("detail") or "")


def test_too_few_keyframes_merges_nothing(tmp_path, monkeypatch):
    events, code, *_ = _walk(tmp_path, monkeypatch, after_stop=True, frames=1)
    assert code == 0
    assert POLL_AFTER not in events and events.count(BUILD) == 1


def test_the_report_is_unchanged_by_the_merge(tmp_path, monkeypatch):
    """The JSON report and the finalization row do not mention the interim build: the
    merge publishes a derived tree only, and the final overwrites it."""
    def report(tmp, merge):
        events, code, store, w, s = _walk(tmp, monkeypatch, merge=merge, after_stop=True)
        fin = dict(store.read_session(w, s).finalization)
        for volatile in ("updated_at", "at", "finished_at", "started_at", "pid"):
            fin.pop(volatile, None)
        return fin

    off = report(tmp_path / "off", False)
    on = report(tmp_path / "on", True)
    assert set(on) == set(off)
    assert on["state"] == off["state"] and on.get("final_solve") == off.get("final_solve")
