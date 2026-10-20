"""When a Scene Understanding session runs, and when it does not.

The rule, in one line: **a scene session runs while somebody is watching
AND somebody is streaming.** Two independent facts, neither sufficient on
its own.

Why neither is enough. The stream is the FEED: with no phone sending
frames there is nothing to observe, and a session left running after its
stream closed holds a model and keeps serving a scene of a room the wearer
has left -- integration finding 15, which sat as a strict xfail in
`test_live_session_lifecycle_races.py` until this rule replaced the
owner-based one. The watcher is the DEMAND: Scene Understanding is the one
cartridge whose subject is detecting people, and it should not run because
a phone happened to open a camera stream for World Builder or the CV Lab.
A result-channel subscription to the live scene is the phone saying "I am
showing this to a person right now"; when the last subscriber leaves, the
detector is released and the GPU is handed back.

The operator path is deliberately separate. `POST /scene/start` from a
Mac or curl starts the session at once, with or without a stream, so a
physical test can be driven without a phone build -- and `POST /scene/stop`
ends that hold. What the operator path no longer does is survive its
stream closing: frames come only from the stream, so a session with no
stream has nothing to do, whoever started it.
"""

import threading
import time

import pytest

from tower.live_session import (
    LiveSession,
    STATE_FAILED,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
)
from tower.scene.live import SceneLive


class _Engine:
    made = 0
    released = 0

    def __init__(self) -> None:
        _Engine.made += 1
        self.observed = 0

    def load(self):
        return None

    def observe(self, frame, received_at=None, source_seq=None):
        self.observed += 1
        return None

    def release(self):
        _Engine.released += 1


class _EngineThatFailsToLoad(_Engine):
    def load(self):
        raise RuntimeError("no weights")


def _await_state(session, want, timeout=10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if session.state == want:
            return True
        time.sleep(0.005)
    return False


_MADE: list = []


def _session(factory=_Engine, **kwargs):
    kwargs.setdefault("follow_stream", True)
    session = SceneLive(factory, decode=lambda raw: raw, **kwargs)
    _MADE.append(session)
    return session


@pytest.fixture(autouse=True)
def _retire_workers():
    """Stop every session this test made and retire its parked worker.

    A `SceneLive` keeps one worker thread parked for the life of the
    process -- correct for the one session a Tower holds, and a thread
    per test here. Retiring them keeps the suite's thread count honest
    for the tests that assert on it.
    """
    yield
    while _MADE:
        session = _MADE.pop()
        session.stop()
        worker = getattr(session, "_worker", None)
        if worker is not None:
            worker.retire()


class TestAStreamAloneDoesNotStartTheScene:
    def test_a_stream_with_no_watcher_leaves_the_session_stopped(self):
        session = _session()
        session.stream_opened(owner="phone")
        assert session.state == STATE_STOPPED
        assert session.status()["demand"]["streams"] == 1
        assert session.status()["demand"]["watchers"] == 0

    def test_a_watcher_with_no_stream_leaves_the_session_stopped(self):
        session = _session()
        session.watcher_joined("phone:sub-1", owner="phone")
        assert session.state == STATE_STOPPED
        assert session.status()["demand"]["watchers"] == 1

    def test_stream_then_watcher_starts_it(self):
        session = _session()
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        assert _await_state(session, STATE_RUNNING)

    def test_watcher_then_stream_starts_it(self):
        session = _session()
        session.watcher_joined("phone:sub-1", owner="phone")
        session.stream_opened(owner="phone")
        assert _await_state(session, STATE_RUNNING)


class TestTheLastWatcherLeavingStopsIt:
    def test_the_last_watcher_leaving_stops_and_discards(self):
        session = _session()
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        session.watcher_joined("mac:sub-1", owner="mac")
        assert _await_state(session, STATE_RUNNING)
        before = _Engine.released

        session.watcher_left("phone:sub-1")
        assert session.state == STATE_RUNNING, "one watcher remains"

        session.watcher_left("mac:sub-1")
        assert _await_state(session, STATE_STOPPED)
        assert _Engine.released == before + 1, "the engine is released"
        assert session.latest() == (None, None, None)

    def test_a_connection_closing_removes_every_watcher_it_held(self):
        session = _session()
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        session.watcher_joined("phone:sub-2", owner="phone")
        assert _await_state(session, STATE_RUNNING)

        session.watchers_left(owner="phone")
        assert _await_state(session, STATE_STOPPED)
        assert session.status()["demand"]["watchers"] == 0

    def test_a_watcher_rejoining_restarts_it_on_a_fresh_session(self):
        session = _session()
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        assert _await_state(session, STATE_RUNNING)
        first = session.status()["session_id"]
        session.watcher_left("phone:sub-1")
        assert _await_state(session, STATE_STOPPED)

        session.watcher_joined("phone:sub-2", owner="phone")
        assert _await_state(session, STATE_RUNNING)
        assert session.status()["session_id"] == first + 1


class TestTheStreamClosingAlwaysStopsIt:
    def test_the_last_stream_closing_stops_it_while_watchers_remain(self):
        session = _session()
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        assert _await_state(session, STATE_RUNNING)

        session.stream_closed(owner="phone")
        assert _await_state(session, STATE_STOPPED)
        # The watcher is still there, so the next stream restarts it.
        session.stream_opened(owner="phone-2")
        assert _await_state(session, STATE_RUNNING)

    def test_a_restarted_session_is_still_stopped_by_its_stream_closing(self):
        """Integration finding 15, closed.

        Stop -> Start over HTTP used to leave the session owned by nobody:
        the stream that fed it could close and it would keep running.
        Open streams are now tracked apart from who started the session,
        so the stream closing stops it whoever pressed Start.
        """
        session = _session()
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        assert _await_state(session, STATE_RUNNING)

        session.stop()
        session.start()
        assert _await_state(session, STATE_RUNNING)

        session.stream_closed(owner="phone")
        assert _await_state(session, STATE_STOPPED, timeout=5.0)

    def test_a_connection_that_never_streamed_does_not_stop_it(self):
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)
        session.stream_closed(owner="a-passing-connection")
        assert session.state == STATE_RUNNING

    def test_with_two_streams_the_first_to_close_does_not_stop_it(self):
        session = _session()
        session.watcher_joined("mac:sub-1", owner="mac")
        session.stream_opened(owner="phone-a")
        session.stream_opened(owner="phone-b")
        assert _await_state(session, STATE_RUNNING)
        session.stream_closed(owner="phone-a")
        assert session.state == STATE_RUNNING
        session.stream_closed(owner="phone-b")
        assert _await_state(session, STATE_STOPPED)


class TestTheOperatorPath:
    def test_http_start_runs_without_a_stream_or_a_watcher(self):
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)
        assert session.status()["demand"]["operator_hold"] is True

    def test_http_stop_releases_the_hold(self):
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)
        session.stop()
        assert session.state == STATE_STOPPED
        assert session.status()["demand"]["operator_hold"] is False

    def test_an_operator_hold_survives_the_last_watcher_leaving(self):
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)
        session.watcher_joined("phone:sub-1", owner="phone")
        session.watcher_left("phone:sub-1")
        assert session.state == STATE_RUNNING

    def test_stream_closure_releases_hold_for_a_later_watched_session(self):
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)
        session.stream_opened(owner="operator-phone")
        session.stream_closed(owner="operator-phone")
        assert _await_state(session, STATE_STOPPED)

        session.stream_opened(owner="watcher-phone")
        session.watcher_joined("watcher-phone:sub-1", owner="watcher-phone")
        assert _await_state(session, STATE_RUNNING)
        session.watcher_left("watcher-phone:sub-1")

        assert _await_state(session, STATE_STOPPED, timeout=0.5)
        assert session.status()["demand"]["operator_hold"] is False

    @staticmethod
    def _watched_then_unwatched_session():
        session = _session()
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        assert _await_state(session, STATE_RUNNING)
        assert session.status()["demand"]["operator_hold"] is False
        return session

    def test_an_operator_start_after_a_demand_stop_decision_wins(self, monkeypatch):
        # The order R2 found, forced: the watcher path decides to stop
        # under `_demand`, the operator starts, THEN the stop runs.
        session = self._watched_then_unwatched_session()
        decided = session._stop_then_reconcile

        def operator_starts_between(*args, **kwargs):
            session.start()
            return decided(*args, **kwargs)

        monkeypatch.setattr(session, "_stop_then_reconcile", operator_starts_between)
        session.watcher_left("phone:sub-1")

        assert session.status()["demand"]["operator_hold"] is True
        assert not _await_state(session, STATE_STOPPED, timeout=0.3)
        assert session.state == STATE_RUNNING

    def test_an_operator_start_during_a_demand_stop_is_restarted(self, monkeypatch):
        # The narrower window: the operator starts after the stop path
        # re-checked the hold but before the base stop closed the door.
        session = self._watched_then_unwatched_session()
        base_stop = LiveSession.stop
        fired = []

        def operator_starts_first(self_):
            if self_ is session and not fired:
                fired.append(True)
                session.start()
            return base_stop(self_)

        monkeypatch.setattr(LiveSession, "stop", operator_starts_first)
        session.watcher_left("phone:sub-1")

        assert fired
        assert session.status()["demand"]["operator_hold"] is True
        assert _await_state(session, STATE_RUNNING)

    def test_a_stream_alone_does_not_start_it_after_an_http_stop(self):
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)
        session.stop()
        session.stream_opened(owner="phone")
        assert session.state == STATE_STOPPED


class TestOperatorDecisionsArriveTogether:
    """An operator Start and Stop, or a Start and the last stream closing,
    at the same moment.

    Each decision sets the hold and THEN makes its base transition. If a
    second decision can land between the two, the hold and the session
    disagree once both calls have returned. Each window is forced with
    events: the first caller is parked on its way into the base call, and
    the second is given `_WINDOW_S` to land inside it. Whether the second
    lands there or is made to wait (and the window expires), once both
    have returned the later decision must have won and the hold must
    agree with the session.
    """

    _WINDOW_S = 1.0

    @staticmethod
    def _park(monkeypatch, verb, thread_name):
        """Park `thread_name` once, on its way into the base `verb`."""
        real = getattr(LiveSession, verb)
        arrived = threading.Event()
        release = threading.Event()

        def parked(self_, *args, **kwargs):
            if threading.current_thread().name == thread_name and not arrived.is_set():
                arrived.set()
                release.wait(10)
            return real(self_, *args, **kwargs)

        monkeypatch.setattr(LiveSession, verb, parked)
        return arrived, release

    @staticmethod
    def _call(thread_name, function, *args):
        returned = threading.Event()

        def run():
            try:
                function(*args)
            finally:
                returned.set()

        thread = threading.Thread(target=run, name=thread_name, daemon=True)
        thread.start()
        return thread, returned

    def _race(self, monkeypatch, *, park, first, second):
        """`first` parked at the base `park`; `second` let into its window."""
        arrived, release = self._park(monkeypatch, park, "first")
        threads = [self._call("first", *first)[0]]
        try:
            assert arrived.wait(5), "the first call never reached its base call"
            second_thread, second_returned = self._call("second", *second)
            threads.append(second_thread)
            second_returned.wait(self._WINDOW_S)
        finally:
            release.set()
        for thread in threads:
            thread.join(10)
            assert not thread.is_alive(), f"{thread.name} never returned"

    def test_a_stop_inside_a_start_leaves_it_stopped_and_unheld(self, monkeypatch):
        # hold-set / hold-clear / base-stop / base-start.
        session = _session()
        self._race(
            monkeypatch,
            park="start",
            first=(session.start,),
            second=(session.stop,),
        )
        # The Stop is the later decision, and it wins.
        assert session.status()["demand"]["operator_hold"] is False
        assert session.state == STATE_STOPPED

    def test_a_start_inside_a_stop_leaves_it_running_and_held(self, monkeypatch):
        # hold-clear / hold-set / base-start / base-stop.
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)
        self._race(
            monkeypatch,
            park="stop",
            first=(session.stop,),
            second=(session.start,),
        )
        # The Start is the later decision, and it wins.
        assert session.status()["demand"]["operator_hold"] is True
        assert session.state in (STATE_STARTING, STATE_RUNNING)

    def test_the_last_stream_closing_inside_a_start_leaves_it_stopped(
        self, monkeypatch
    ):
        # The demand stop path. The stream closes after the Start set its
        # hold, so the closure is the later decision and, by the rule in
        # `TestTheStreamClosingAlwaysStopsIt`, ends the session. Landing
        # inside the Start's window left a session running with no hold,
        # no stream and no watcher: owned by nobody, ended by nothing.
        session = _session()
        session.stream_opened(owner="phone")
        self._race(
            monkeypatch,
            park="start",
            first=(session.start,),
            second=(session.stream_closed, "phone"),
        )
        assert session.status()["demand"]["operator_hold"] is False
        assert session.state == STATE_STOPPED

    def test_a_start_inside_a_demand_stop_still_wins(self, monkeypatch):
        # The window `_stop_then_reconcile` already covered, now from
        # another thread: whichever way the two run, the Start is the
        # later decision.
        session = TestTheOperatorPath._watched_then_unwatched_session()
        self._race(
            monkeypatch,
            park="stop",
            first=(session.watcher_left, "phone:sub-1"),
            second=(session.start,),
        )
        assert session.status()["demand"]["operator_hold"] is True
        assert _await_state(session, STATE_RUNNING)


class TestDemandNeverUndoesAPause:
    def test_a_new_watcher_does_not_resume_a_paused_session(self):
        session = _session()
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        assert _await_state(session, STATE_RUNNING)
        session.pause()
        session.watcher_joined("phone:sub-2", owner="phone")
        session.stream_opened(owner="phone-2")
        assert session.state == STATE_PAUSED

    def test_the_last_watcher_leaving_stops_a_paused_session_too(self):
        """A paused scene is still a held scene, and nobody is looking."""
        session = _session()
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        assert _await_state(session, STATE_RUNNING)
        session.pause()
        session.watcher_left("phone:sub-1")
        assert _await_state(session, STATE_STOPPED)
        assert session.latest() == (None, None, None)


class TestFailureAndRetry:
    def test_a_failed_load_is_retried_by_the_next_demand_event_only(self):
        session = _session(_EngineThatFailsToLoad)
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        assert _await_state(session, STATE_FAILED)
        failed_session = session.status()["session_id"]

        # Nothing retries on its own.
        time.sleep(0.05)
        assert session.status()["session_id"] == failed_session

        # A NEW watcher is a new request, and gets one new attempt.
        session.watcher_joined("phone:sub-2", owner="phone")
        assert _await_state(session, STATE_FAILED)
        assert session.status()["session_id"] == failed_session + 1


class TestManualMode:
    def test_with_autostart_off_only_the_operator_can_start_it(self):
        session = _session(follow_stream=False)
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        assert session.state == STATE_STOPPED
        session.start()
        assert _await_state(session, STATE_RUNNING)
        # But the stream closing still ends it: no stream, no frames.
        session.stream_closed(owner="phone")
        assert _await_state(session, STATE_STOPPED)


class TestTheStatusSaysWhy:
    def test_demand_is_on_the_status(self):
        session = _session()
        demand = session.status()["demand"]
        assert demand == {
            "streams": 0,
            "watchers": 0,
            "operator_hold": False,
            "runs_when": "stream-and-watcher-or-operator",
        }

    def test_offering_frames_from_many_threads_while_demand_changes_never_raises(
        self,
    ):
        session = _session()
        stop = threading.Event()
        errors: list = []

        def churn():
            i = 0
            while not stop.is_set():
                try:
                    session.stream_opened(owner="p")
                    session.watcher_joined(f"p:{i}", owner="p")
                    session.offer_frame(b"x")
                    session.watcher_left(f"p:{i}")
                    session.stream_closed(owner="p")
                except Exception as exc:  # pragma: no cover - the assertion
                    errors.append(exc)
                i += 1

        threads = [threading.Thread(target=churn) for _ in range(4)]
        for thread in threads:
            thread.start()
        time.sleep(0.3)
        stop.set()
        for thread in threads:
            thread.join(5)
        session.stop()
        assert not errors
        assert session.state == STATE_STOPPED


class TestDemandEventsNeverWaitOnAStop:
    """A reviewer's SEV-1: `stream_opened` and `watcher_joined` run on the
    event loop, and a stop joins its worker for up to 5 s. The demand lock
    must not be held across that join."""

    def test_stream_opened_returns_while_a_stop_is_joining(self):
        entered = threading.Event()
        release = threading.Event()

        class _SlowEngine(_Engine):
            def observe(self, frame, received_at=None, source_seq=None):
                entered.set()
                release.wait(10)
                return None

        session = _session(_SlowEngine, stop_join_timeout_s=3.0)
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        assert _await_state(session, STATE_RUNNING)
        session.offer_frame(b"frame")
        assert entered.wait(5), "the worker never took the frame"

        # A stop that will block on the join until `release` is set.
        stopper = threading.Thread(target=session.watcher_left, args=("phone:sub-1",))
        stopper.start()
        time.sleep(0.1)

        started = time.monotonic()
        session.stream_opened(owner="phone-2")
        session.watcher_joined("mac:sub-1", owner="mac")
        elapsed = time.monotonic() - started

        release.set()
        stopper.join(10)
        assert elapsed < 0.5, f"a demand event waited {elapsed:.2f}s on a stop"
        # And the facts still say run: a watcher and a stream exist.
        assert _await_state(session, STATE_RUNNING, timeout=10.0)

    def test_a_watcher_joining_during_a_stop_ends_up_running(self):
        """The race the lock used to prevent, closed by reconciling after
        the stop instead."""
        session = _session()
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        assert _await_state(session, STATE_RUNNING)
        session.watcher_left("phone:sub-1")
        session.watcher_joined("phone:sub-2", owner="phone")
        assert _await_state(session, STATE_RUNNING)


class TestAFailingDetectorDoesNotFloodTheLog:
    def test_the_failure_is_logged_once_then_by_decade(self, caplog):
        import logging

        import numpy as np

        from tower.scene.engine import SceneEngine

        class _Exploding:
            name = "exploding"
            score_threshold = 0.5

            def load(self):
                return None

            def detect(self, frame):
                raise RuntimeError("poisoned")

            def release(self):
                return None

        engine = SceneEngine(_Exploding(), clock=lambda: 0.0)
        engine.load()
        with caplog.at_level(logging.ERROR, logger="tower.scene.engine"):
            for index in range(120):
                engine.observe(np.zeros((8, 8, 3), np.uint8), received_at=index * 0.1)
        assert engine.detect_failures == 120
        assert len(caplog.records) == 3  # 1st, 10th, 100th
