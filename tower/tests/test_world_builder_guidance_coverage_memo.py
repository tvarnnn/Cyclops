"""Requested FOW work is bounded by distinct solve/tree identities."""

import threading

from tower.world_builder import guidance_coverage as coverage
from tests.test_world_builder_guidance_coverage_retry import _land


def _until(predicate):
    for _ in range(400):
        if predicate():
            return
        threading.Event().wait(0.005)
    assert predicate()


def _offer(worker, world, tag=(1, "digest"), horizon=4):
    return worker.offer(world, "session", 1000.0, horizon, (1, horizon), "geometry", tag)


def test_twelve_rotating_worlds_compute_once_and_publish_from_memo(monkeypatch, tmp_path):
    calls = []

    def compute(root, world, session, expected, revision, clock, tag, budget):
        calls.append((world, tag))
        return {"world": world, "horizon_keyframes": expected[1]}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path)
    try:
        for world in (f"world-{n}" for n in range(12)):
            _offer(worker, world)
            _until(lambda world=world: worker.latest(world, "session") is not None)
        for _ in range(100):
            for world in (f"world-{n}" for n in range(12)):
                assert _offer(worker, world)["world"] == world
        assert calls == [(f"world-{n}", (1, "digest")) for n in range(12)]
    finally:
        worker.shutdown()
        worker._thread.join(1)


def test_same_solve_new_tree_recomputes_and_publishes_new_receipt(monkeypatch, tmp_path):
    calls = []

    def compute(root, world, session, expected, revision, clock, tag, budget):
        calls.append(tag)
        return {"tag": tag}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path)
    try:
        _offer(worker, "world")
        _until(lambda: worker.latest("world", "session") is not None)
        _offer(worker, "world", (2, "digest"))
        _until(lambda: worker.latest("world", "session").get("tag") == (2, "digest"))
        assert calls == [(1, "digest"), (2, "digest")]
    finally:
        worker.shutdown()
        worker._thread.join(1)


def test_no_store_sweep_until_status_requests_a_target(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(coverage, "compute_from_tree", lambda *args: calls.append(args))
    _land(tmp_path, "s", 1000.0, 221)
    worker = coverage.CoverageWorker(tmp_path)
    try:
        threading.Event().wait(0.65)
        assert calls == []
    finally:
        worker.shutdown()
        worker._thread.join(1)


def test_active_landing_precedes_rotating_old_worlds(monkeypatch, tmp_path):
    calls = []
    entered, release = threading.Event(), threading.Event()

    def compute(root, world, session, expected, revision, clock, tag, budget):
        calls.append(world)
        if world == "old-0":
            entered.set()
            assert release.wait(1)
        return {"world": world}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path)
    try:
        _offer(worker, "old-0")
        assert entered.wait(1)
        for n in range(1, 12):
            _offer(worker, f"old-{n}")
        _offer(worker, "active", horizon=100)
        release.set()
        _until(lambda: worker.latest("active", "session") is not None)
        assert calls.index("active") <= 1
        _until(lambda: all(worker.latest(f"old-{n}", "session") is not None
                           for n in range(12)))
        assert len(calls) == 13
    finally:
        release.set()
        worker.shutdown()
        worker._thread.join(1)


def test_thirty_seconds_of_rotating_polls_has_bounded_compute_calls(monkeypatch, tmp_path):
    calls = []

    def compute(root, world, session, expected, revision, clock, tag, budget):
        calls.append(world)
        return {"world": world}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path)
    try:
        for second in range(30):
            for n in range(12):
                _offer(worker, f"old-{n}")
        _until(lambda: all(worker.latest(f"old-{n}", "session") is not None
                           for n in range(12)))
        for second in range(30):
            for n in range(12):
                _offer(worker, f"old-{n}")
        assert len(calls) == 12
    finally:
        worker.shutdown()
        worker._thread.join(1)


def test_new_tree_does_not_publish_older_inflight_receipt(monkeypatch, tmp_path):
    calls = []
    old_entered, release_old = threading.Event(), threading.Event()
    new_entered, release_new = threading.Event(), threading.Event()

    def compute(root, world, session, expected, revision, clock, tag, budget):
        calls.append(tag)
        if tag == (1, "digest"):
            old_entered.set()
            assert release_old.wait(1)
        else:
            new_entered.set()
            assert release_new.wait(1)
        return {"tag": tag}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    worker = coverage.CoverageWorker(tmp_path)
    try:
        _offer(worker, "active", (1, "digest"))
        assert old_entered.wait(1)
        _offer(worker, "active", (2, "digest"))
        release_old.set()
        assert new_entered.wait(1)
        assert worker.latest("active", "session") is None
        release_new.set()
        _until(lambda: worker.latest("active", "session") is not None)
        assert worker.latest("active", "session")["tag"] == (2, "digest")
        assert calls == [(1, "digest"), (2, "digest")]
    finally:
        release_old.set()
        release_new.set()
        worker.shutdown()
        worker._thread.join(1)
