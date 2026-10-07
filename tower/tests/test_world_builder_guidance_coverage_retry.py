"""Bounded retry behavior for the optional coverage worker."""

import threading
import time

from tower.world_builder import guidance_coverage as coverage


def _until(predicate):
    deadline = time.monotonic() + 1.0
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


def _offer(worker, horizon):
    return worker.offer("w", "s", float(horizon), horizon, (horizon, horizon),
                        f"g{horizon}", (horizon, "d"))


def test_deadline_miss_retries_same_candidate_and_advances_block(monkeypatch, tmp_path):
    retry_clock = _Clock()
    calls = []

    def compute(root, world, session, expected, revision, clock, tag, budget):
        calls.append(expected[1])
        if len(calls) == 2:
            raise getattr(coverage, "_TransientCoverageError", ValueError)(
                "coverage worker exceeded 250 ms")
        return {"horizon_keyframes": expected[1]}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path, retry_clock=retry_clock.now)
    try:
        _offer(worker, 221)
        _until(lambda: worker.latest("w", "s") is not None)
        assert worker.latest("w", "s")["horizon_keyframes"] == 221
        _offer(worker, 1090)
        _until(lambda: worker._pending is not None and worker._pending[3] == 1)
        assert worker.latest("w", "s")["horizon_keyframes"] == 221
        assert calls == [221, 1090]
        retry_clock.advance(worker, 5.0)
        _until(lambda: worker.latest("w", "s")["horizon_keyframes"] == 1090)
        assert calls == [221, 1090, 1090]
    finally:
        worker.shutdown()
        worker._thread.join(1)
        assert not worker._thread.is_alive()


def test_timeout_gives_up_after_exactly_two_retries(monkeypatch, tmp_path, caplog):
    retry_clock = _Clock()
    calls = []

    def compute(*args):
        calls.append(1)
        raise coverage._TransientCoverageError("coverage worker exceeded 250 ms")

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path, retry_clock=retry_clock.now)
    try:
        _offer(worker, 1090)
        _until(lambda: worker._pending is not None and worker._pending[3] == 1)
        assert worker._pending[4] == 5.0
        worker.offer("w", "s", 1090.0, 1090, (1090, 1090), "g-new", (2, "d"))
        assert worker._pending[2:] == ((2, "d"), 1, 5.0)
        retry_clock.advance(worker, 5.0)
        _until(lambda: worker._pending is not None and worker._pending[3] == 2)
        assert worker._pending[4] == 25.0
        retry_clock.advance(worker, 20.0)
        _until(lambda: len(calls) == 3 and worker._pending is None and
               any("disposition=gave up" in r.message for r in caplog.records))
        assert calls == [1, 1, 1]
        assert [r.message.rsplit("disposition=", 1)[1] for r in caplog.records
                if "candidate=" in r.message] == ["retry 1", "retry 2", "gave up"]
        retry_clock.advance(worker, 100.0)
        worker.offer("w", "s", 1090.0, 1090, (1090, 1090), "g-newer", (3, "d"))
        assert calls == [1, 1, 1]
    finally:
        worker.shutdown()
        worker._thread.join(1)
        assert not worker._thread.is_alive()


def test_newer_candidate_supersedes_pending_retry(monkeypatch, tmp_path):
    retry_clock = _Clock()
    calls = []

    def compute(root, world, session, expected, *args):
        calls.append(expected[1])
        if expected[1] == 221:
            raise coverage._TransientCoverageError("derived tree changed during guidance read")
        return {"horizon_keyframes": expected[1]}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path, retry_clock=retry_clock.now)
    try:
        _offer(worker, 221)
        _until(lambda: worker._pending is not None and worker._pending[3] == 1)
        _offer(worker, 1090)
        _until(lambda: worker.latest("w", "s") is not None)
        retry_clock.advance(worker, 30.0)
        _offer(worker, 221)
        assert worker.latest("w", "s")["horizon_keyframes"] == 1090
        assert calls == [221, 1090]
        assert worker._pending is None
    finally:
        worker.shutdown()
        worker._thread.join(1)
        assert not worker._thread.is_alive()


def test_permanent_validation_failure_is_not_retried(monkeypatch, tmp_path):
    retry_clock = _Clock()
    calls = []

    def compute(*args):
        calls.append(1)
        raise ValueError("invalid solve horizon")

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path, retry_clock=retry_clock.now)
    try:
        _offer(worker, 1090)
        _until(lambda: worker._completed is not None)
        assert worker._pending is None
        retry_clock.advance(worker, 30.0)
        _offer(worker, 1090)
        assert calls == [1]
        assert worker.latest("w", "s") is None
    finally:
        worker.shutdown()
        worker._thread.join(1)
        assert not worker._thread.is_alive()


def test_shutdown_cancels_pending_retry_without_timer_or_thread(monkeypatch, tmp_path):
    retry_clock = _Clock()
    calls = []

    def compute(*args):
        calls.append(1)
        raise coverage._TransientCoverageError("solve clock not yet settled")

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path, retry_clock=retry_clock.now)
    _offer(worker, 1090)
    _until(lambda: worker._pending is not None and worker._pending[3] == 1)
    worker.shutdown()
    worker._thread.join(1)
    assert not worker._thread.is_alive()
    assert worker._pending is None
    retry_clock.advance(worker, 30.0)
    _offer(worker, 1090)
    assert calls == [1]
