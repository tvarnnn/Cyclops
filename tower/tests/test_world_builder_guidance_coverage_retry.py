"""Bounded retry after a transient coverage-worker failure (FOW amendment 2026-10-07).

Walk 6 kept the 221-keyframe block for the rest of the walk: the 1,090
landing missed the 250 ms budget once and the worker never offered it again.
Transient failures (deadline miss, tree changed mid-read, unsettled clock)
now retry the same candidate after 5 s, then 20 s; permanent ones do not.
All cooldowns run on an injected clock; no test sleeps more than a few ms.
"""

import json
import logging
import threading
import time

import pytest

from tower.world_builder import guidance_coverage as coverage
from tests.test_world_builder_guidance_coverage import _block, _inputs


def _until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.005)
    assert predicate()


class _Clock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def advance(self, worker, seconds):
        with worker._condition:
            self.value += seconds
            worker._condition.notify_all()


def _offer(worker, horizon, tag=None):
    return worker.offer("w", "s", float(horizon), horizon, (horizon, horizon),
                        f"g{horizon}", tag or (horizon, "d"))


def _miss_deadline(budget):
    """The real budget's own deadline error, as the worker raises it."""
    budget.deadline = 0.0
    budget.check()


def _dispositions(caplog):
    return [r.getMessage().rsplit("disposition=", 1)[1] for r in caplog.records
            if "candidate=" in r.getMessage() and "disposition=" in r.getMessage()]


def _stop(worker):
    worker.shutdown()
    worker._thread.join(1)
    assert not worker._thread.is_alive()


def test_deadline_miss_retries_same_candidate_and_advances_block(monkeypatch, tmp_path):
    """Fails on 5a8efa4: the 1,090 candidate is never attempted again."""
    clock = _Clock()
    calls = []

    def compute(root, world, session, expected, revision, wall, tag, budget):
        calls.append(expected[1])
        if len(calls) == 2:
            _miss_deadline(budget)
        return {"horizon_keyframes": expected[1]}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path)
    worker._retry_clock = clock.now  # attribute, so the base runs this test too
    try:
        _offer(worker, 221)
        _until(lambda: worker.latest("w", "s") is not None)
        _offer(worker, 1090)
        _until(lambda: calls == [221, 1090] and worker._completed[3] == 1090)
        assert worker.latest("w", "s")["horizon_keyframes"] == 221  # old block kept
        clock.advance(worker, 4.9)
        threading.Event().wait(0.05)
        assert calls == [221, 1090]
        clock.advance(worker, 0.1)
        _until(lambda: worker.latest("w", "s")["horizon_keyframes"] == 1090)
        assert calls == [221, 1090, 1090]
    finally:
        _stop(worker)


def test_timeout_gives_up_after_exactly_two_retries(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=coverage.__name__)
    clock = _Clock()
    calls = []

    def compute(root, world, session, expected, revision, wall, tag, budget):
        calls.append(1)
        _miss_deadline(budget)

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path, retry_clock=clock.now)
    try:
        _offer(worker, 1090)
        _until(lambda: worker._pending is not None and worker._pending[3] == 1)
        assert worker._pending[4] == 5.0
        clock.advance(worker, 5.0)
        _until(lambda: worker._pending is not None and worker._pending[3] == 2)
        assert worker._pending[4] == 25.0
        clock.advance(worker, 20.0)
        _until(lambda: len(calls) == 3 and worker._pending is None and
               "gave up" in _dispositions(caplog))
        clock.advance(worker, 1000.0)
        _offer(worker, 1090)
        threading.Event().wait(0.05)
        assert calls == [1, 1, 1]
        assert _dispositions(caplog) == ["retry 1", "retry 2", "gave up"]
        message = [r.getMessage() for r in caplog.records if "retry 1" in r.getMessage()][0]
        assert "1090" in message and "elapsed_ms=" in message and "exceeded 250 ms" in message
    finally:
        _stop(worker)


def test_new_tree_during_cooldown_is_attempted_without_resetting_the_cap(monkeypatch, tmp_path,
                                                                        caplog):
    """A rebuilt tree re-attempts at once (as on the base); the cap stays per candidate."""
    caplog.set_level(logging.INFO, logger=coverage.__name__)
    clock = _Clock()
    calls = []

    def compute(root, world, session, expected, revision, wall, tag, budget):
        calls.append(tag)
        raise coverage._TransientCoverageError("derived tree changed during guidance read")

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path, retry_clock=clock.now)
    try:
        _offer(worker, 1090, tag=(1, "d"))
        _until(lambda: worker._pending is not None and worker._pending[3] == 1)
        _offer(worker, 1090, tag=(2, "d"))
        _until(lambda: worker._pending is not None and worker._pending[3] == 2)
        clock.advance(worker, 20.0)
        _until(lambda: len(calls) == 3 and worker._pending is None)
        assert calls == [(1, "d"), (2, "d"), (2, "d")]
        assert _dispositions(caplog) == ["retry 1", "retry 2", "gave up"]
    finally:
        _stop(worker)


def test_rebuilt_tree_offered_during_a_failing_attempt_runs_without_cooldown(monkeypatch,
                                                                            tmp_path):
    """A failing attempt must not replace an already-queued fresh job with a timed retry."""
    clock = _Clock()
    calls, running, release = [], threading.Event(), threading.Event()

    def compute(root, world, session, expected, revision, wall, tag, budget):
        calls.append(tag)
        if tag == (1, "d"):
            running.set()
            release.wait(1.0)
            raise coverage._TransientCoverageError("derived tree changed during guidance read")
        return {"horizon_keyframes": expected[1]}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path, retry_clock=clock.now)
    try:
        _offer(worker, 1090, tag=(1, "d"))
        assert running.wait(1.0)
        _offer(worker, 1090, tag=(2, "d"))  # rebuilt tree while attempt 1 runs
        release.set()
        _until(lambda: worker.latest("w", "s") is not None)  # clock never advanced
        assert calls == [(1, "d"), (2, "d")]
        assert worker._pending is None
    finally:
        _stop(worker)


def _land(root, session, solved_at, horizon):
    """A discoverable landing: what `_discover` globs and reads, nothing more."""
    world = root / "worlds" / "w"
    solution = world / "solve" / session / "solution.json"
    solution.parent.mkdir(parents=True, exist_ok=True)
    solution.write_text(json.dumps({"solved_at": solved_at, "pad": "x" * horizon}),
                        encoding="utf-8")
    manifest = world / "derived" / session / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"global_solve": {"solved_at": solved_at,
                                                     "horizon_keyframes": horizon},
                                    "built_at": solved_at + 1, "input_digest": "d"}),
                        encoding="utf-8")


def test_discovery_keeps_running_during_a_retry_cooldown(monkeypatch, tmp_path):
    """A newer landing written during a 5 s cooldown is attempted within ~0.5 s, not 5 s."""
    clock = _Clock()
    calls = []

    def compute(root, world, session, expected, *args):
        calls.append(expected[1])
        if expected[1] == 221:
            _miss_deadline(args[-1])
        return {"horizon_keyframes": expected[1]}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    _land(tmp_path, "s", 1000.0, 221)
    worker = coverage.CoverageWorker(tmp_path, retry_clock=clock.now)
    try:
        worker.request("w", "s")
        _until(lambda: worker._pending is not None and worker._pending[3] == 1)
        _land(tmp_path, "s", 2000.0, 1090)  # the next solve lands; nothing offered by hand
        worker.request("w", "s")
        _until(lambda: worker.latest("w", "s") is not None)  # clock never advanced
        assert worker.latest("w", "s")["horizon_keyframes"] == 1090
        assert calls == [221, 1090]
    finally:
        _stop(worker)


def test_newer_candidate_supersedes_pending_retry(monkeypatch, tmp_path):
    clock = _Clock()
    calls = []

    def compute(root, world, session, expected, *args):
        calls.append(expected[1])
        if expected[1] == 221:
            raise coverage._TransientCoverageError("derived tree changed during guidance read")
        return {"horizon_keyframes": expected[1]}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path, retry_clock=clock.now)
    try:
        _offer(worker, 221)
        _until(lambda: worker._pending is not None and worker._pending[3] == 1)
        _offer(worker, 1090)
        _until(lambda: worker.latest("w", "s") is not None)
        clock.advance(worker, 30.0)
        threading.Event().wait(0.05)
        _offer(worker, 221)
        assert worker.latest("w", "s")["horizon_keyframes"] == 1090
        assert calls == [221, 1090]
        assert worker._pending is None
    finally:
        _stop(worker)


@pytest.mark.parametrize("message", ["invalid solve clock", "invalid solve horizon",
                                     "guidance file exceeds size limit"])
def test_permanent_validation_failure_is_not_retried(monkeypatch, tmp_path, message):
    clock = _Clock()
    calls = []

    def compute(*args):
        calls.append(1)
        raise ValueError(message)

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path, retry_clock=clock.now)
    try:
        _offer(worker, 1090)
        _until(lambda: worker._completed is not None)
        assert worker._pending is None
        clock.advance(worker, 30.0)
        _offer(worker, 1090)
        threading.Event().wait(0.05)
        assert calls == [1]
        assert worker.latest("w", "s") is None
    finally:
        _stop(worker)


def test_shutdown_cancels_pending_retry_without_timer_or_thread(monkeypatch, tmp_path):
    clock = _Clock()
    calls = []

    def compute(*args):
        calls.append(1)
        raise coverage._TransientCoverageError("solve clock not yet settled")

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    before = set(threading.enumerate())
    worker = coverage.CoverageWorker(tmp_path, retry_clock=clock.now)
    _offer(worker, 1090)
    _until(lambda: worker._pending is not None and worker._pending[3] == 1)
    _stop(worker)
    assert worker._pending is None
    assert set(threading.enumerate()) <= before
    clock.advance(worker, 30.0)
    _offer(worker, 1090)
    assert calls == [1]


def test_failure_classification():
    data = _inputs()
    data["computed_at"] = data["solution"]["solved_at"] - 0.001
    with pytest.raises(coverage._TransientCoverageError, match="not yet settled"):
        _block(data)
    data = _inputs()
    data["computed_at"] = True
    with pytest.raises(ValueError, match="invalid solve clock") as refused:
        _block(data)
    assert not isinstance(refused.value, coverage._TransientCoverageError)
    with pytest.raises(coverage._TransientCoverageError, match="250 ms"):
        _miss_deadline(coverage._Budget())
    with pytest.raises(ValueError, match="cancelled") as cancelled:
        stop = threading.Event()
        stop.set()
        coverage._Budget(stop).check()
    assert not isinstance(cancelled.value, coverage._TransientCoverageError)
