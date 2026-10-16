"""B2 tile job budget: CPU time, a wall-clock cap, and one retry per receipt.

e2e10 (2026-10-12 20:47:10): a ~2 ms tile job on the below-normal-priority
coverage worker missed the 80 ms *wall* budget under a live walk's load, so
that landing never got photos ("Photos unavailable for this update").
"""

from __future__ import annotations

import threading
import time

import pytest

from tower.world_builder import guidance_coverage as coverage
from tower.world_builder import guidance_imagery as imagery


class _Store:
    redaction = "faces-blurred"

    def __init__(self, root):
        pass

    def keyframe_image_set(self, world, session):
        return type("ImageSet", (), {"redaction": self.redaction})()


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(imagery, "WorldStore", _Store)
    monkeypatch.setattr(imagery, "label_is_trusted", lambda label: label == "faces-blurred")
    return imagery.ImageryState()


RECEIPT = {"solved_at": 1.0, "geometry_revision": "g-1", "frame_revision": 1}


def _dial(n):
    return {"reference_segment": 0, "sectors": tuple((s, (f"k{s}",)) for s in range(n)),
            "latest": {"component_reference_segment": 0, "sector": 0, "keyframe_id": "k0"},
            "motion_sector": None}


def _burn(seconds):
    started = time.thread_time()
    while time.thread_time() - started < seconds:
        pass


def test_a_starved_job_that_needs_little_cpu_still_publishes(state, monkeypatch):
    # Every wall-clock read jumps 50 ms: the job "takes" ~0.5 s of wall time
    # waiting for the CPU, but its own CPU time is microseconds.
    wall = [100.0]

    def starved():
        wall[0] += .050
        return wall[0]

    monkeypatch.setattr(imagery, "_tile", lambda *args: b"\xff\xd8\xff" + args[3].encode())
    monkeypatch.setattr(imagery.time, "monotonic", starved)
    outcome = state.publish_landed("root", "home", "s9", RECEIPT, [], dial=_dial(4))
    assert outcome == imagery.LANDED_PUBLISHED
    assert wall[0] - 100.0 > .080  # the old wall budget would have omitted it
    assert [e["sector"] for e in state.landed("home", "s9")["entries"]] == [0, 1, 2, 3]


def test_a_cpu_heavy_job_is_still_cut(state, monkeypatch):
    monkeypatch.setattr(imagery, "_tile", lambda *args: b"\xff\xd8\xff" + args[3].encode())
    state.publish_landed("root", "home", "s9", RECEIPT, [], dial=_dial(1))
    previous = state.landed("home", "s9")

    def heavy(*args):
        _burn(.030)  # 30 ms of real CPU per tile: 12 sectors need ~360 ms
        return b"\xff\xd8\xff" + args[3].encode()

    monkeypatch.setattr(imagery, "_tile", heavy)
    started = time.thread_time()
    outcome = state.publish_landed("root", "home", "s9", {**RECEIPT, "solved_at": 2.0}, [],
                                   dial=_dial(12))
    assert outcome == imagery.LANDED_OVER_BUDGET
    assert time.thread_time() - started < .200  # stopped near 80 ms, not after all 12
    assert state.landed("home", "s9") == previous


def test_the_wall_clock_cap_still_bounds_a_stalled_job(state, monkeypatch):
    wall = [100.0]

    def stalled():
        wall[0] += .600
        return wall[0]

    monkeypatch.setattr(imagery, "_tile", lambda *args: b"\xff\xd8\xff" + args[3].encode())
    monkeypatch.setattr(imagery.time, "monotonic", stalled)
    outcome = state.publish_landed("root", "home", "s9", RECEIPT, [], dial=_dial(4))
    assert outcome == imagery.LANDED_OVER_BUDGET
    assert state.landed("home", "s9") is None


def test_no_tile_is_skipped_not_over_budget(state, monkeypatch):
    monkeypatch.setattr(imagery, "_tile", lambda *args: None)
    assert state.publish_landed("root", "home", "s9", RECEIPT, [], dial=_dial(4)) == \
        imagery.LANDED_SKIPPED
    _Store.redaction = "untrusted"
    try:
        assert state.publish_landed("root", "home", "s9", RECEIPT, [], dial=_dial(4)) == \
            imagery.LANDED_SKIPPED
    finally:
        _Store.redaction = "faces-blurred"


# -- the coverage worker's one retry ---------------------------------------------


class _Sink:
    """An imagery state whose publish outcomes are scripted."""

    def __init__(self, outcomes, delay=0.0):
        self.outcomes = list(outcomes)
        self.calls = []
        self.delay = delay
        self.in_publish = threading.Event()

    def begin_landed(self, *args):
        pass

    def finish_landed(self, *args):
        pass

    def publish_landed(self, root, world, session, block, eligible, **kwargs):
        self.calls.append(block["solved_at"])
        if self.delay and len(self.calls) > 1:
            self.in_publish.set()
            time.sleep(self.delay)
        return self.outcomes.pop(0) if self.outcomes else imagery.LANDED_PUBLISHED


def _worker(monkeypatch, tmp_path, sink):
    monkeypatch.setattr(imagery, "state_for", lambda root: sink)

    def compute(root, world, session, expected, revision, clock, tag, budget, **kwargs):
        if kwargs.get("dial_out") is not None:
            kwargs["dial_out"].update(_dial(1))
        return {"solved_at": expected[0], "geometry_revision": revision, "frame_revision": 1}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    return coverage.CoverageWorker(tmp_path)


def _wait(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.01)
    return False


def test_an_over_budget_landing_is_retried_exactly_once(monkeypatch, tmp_path):
    sink = _Sink([imagery.LANDED_OVER_BUDGET, imagery.LANDED_OVER_BUDGET,
                  imagery.LANDED_OVER_BUDGET])
    worker = _worker(monkeypatch, tmp_path, sink)
    try:
        worker.offer("home", "s9", 1.0, 1, (1, 1), "g1", (1, "d"))
        assert _wait(lambda: len(sink.calls) == 2)
        time.sleep(1.5)  # three more worker ticks
        assert sink.calls == [1.0, 1.0]
        assert worker._landed_retry is None
    finally:
        worker.shutdown()
        worker._thread.join(1)


def test_a_successful_retry_publishes_the_same_receipt(monkeypatch, tmp_path):
    sink = _Sink([imagery.LANDED_OVER_BUDGET, imagery.LANDED_PUBLISHED])
    worker = _worker(monkeypatch, tmp_path, sink)
    try:
        worker.offer("home", "s9", 1.0, 1, (1, 1), "g1", (1, "d"))
        assert _wait(lambda: len(sink.calls) == 2)
        assert sink.calls == [1.0, 1.0]
    finally:
        worker.shutdown()
        worker._thread.join(1)


def test_published_or_skipped_landings_are_not_retried(monkeypatch, tmp_path):
    sink = _Sink([imagery.LANDED_SKIPPED])
    worker = _worker(monkeypatch, tmp_path, sink)
    try:
        worker.offer("home", "s9", 1.0, 1, (1, 1), "g1", (1, "d"))
        assert _wait(lambda: sink.calls)
        time.sleep(1.2)
        assert sink.calls == [1.0]
    finally:
        worker.shutdown()
        worker._thread.join(1)


def test_the_retry_is_dropped_when_a_newer_receipt_lands(monkeypatch, tmp_path):
    sink = _Sink([imagery.LANDED_OVER_BUDGET, imagery.LANDED_PUBLISHED])
    worker = _worker(monkeypatch, tmp_path, sink)
    try:
        worker.offer("home", "s9", 1.0, 1, (1, 1), "g1", (1, "d"))
        assert _wait(lambda: sink.calls == [1.0])
        # The newer solve lands before the retry's tick (0.5 s later).
        worker.offer("home", "s9", 2.0, 2, (1, 2), "g2", (2, "d"))
        assert _wait(lambda: len(sink.calls) == 2)
        time.sleep(1.2)
        assert sink.calls == [1.0, 2.0]  # the 1.0 retry never ran
    finally:
        worker.shutdown()
        worker._thread.join(1)


def test_a_retry_owed_to_a_superseded_receipt_is_dropped_at_its_tick(monkeypatch, tmp_path):
    # Direct check of the tick-time guard: the published receipt changed
    # (e.g. offer() re-published another candidate) after the retry was owed.
    sink = _Sink([imagery.LANDED_OVER_BUDGET])
    worker = _worker(monkeypatch, tmp_path, sink)
    try:
        worker.offer("home", "s9", 1.0, 1, (1, 1), "g1", (1, "d"))
        assert _wait(lambda: sink.calls == [1.0])
        with worker._condition:
            worker._published = {**worker._published, ("home", "s9"): object()}
        time.sleep(1.2)
        assert sink.calls == [1.0]
        assert worker._landed_retry is None
    finally:
        worker.shutdown()
        worker._thread.join(1)


def test_status_is_never_blocked_by_a_retry(monkeypatch, tmp_path):
    sink = _Sink([imagery.LANDED_OVER_BUDGET, imagery.LANDED_PUBLISHED], delay=.5)
    worker = _worker(monkeypatch, tmp_path, sink)
    try:
        worker.offer("home", "s9", 1.0, 1, (1, 1), "g1", (1, "d"))
        assert sink.in_publish.wait(3)  # the retry is now inside its tile job
        started = time.perf_counter()
        assert worker.latest("home", "s9") is not None
        assert worker.request("home", "s9") is not None
        worker.offer("home", "s9", 1.0, 1, (1, 1), "g1", (1, "d"))
        assert time.perf_counter() - started < .050
    finally:
        worker.shutdown()
        worker._thread.join(2)


def test_a_failing_retry_never_ends_the_worker(monkeypatch, tmp_path):
    # Review HIGH (1e2c7f5): an exception anywhere in the retry must not kill the coverage thread.
    sink = _Sink([imagery.LANDED_OVER_BUDGET, imagery.LANDED_PUBLISHED])
    worker = _worker(monkeypatch, tmp_path, sink)

    def boom():
        raise RuntimeError("retry bug")

    monkeypatch.setattr(worker, "_retry_landed", boom)
    try:
        worker.offer("home", "s9", 1.0, 1, (1, 1), "g1", (1, "d"))
        assert _wait(lambda: sink.calls == [1.0])
        assert _wait(lambda: worker._landed_retry is None)
        assert worker._thread.is_alive()
        worker.offer("home", "s9", 2.0, 2, (2, 2), "g2", (2, "d"))
        assert _wait(lambda: sink.calls == [1.0, 2.0])
        assert worker._thread.is_alive()
    finally:
        worker.shutdown()
        worker._thread.join(1)
