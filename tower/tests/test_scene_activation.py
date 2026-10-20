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
from tower.cartridge_runtime import LiveCartridges
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

    Each stop is bounded, so a regression that leaves a lock held fails
    the test here rather than hanging the suite.
    """
    yield
    stuck = []
    while _MADE:
        session = _MADE.pop()
        stopper = threading.Thread(target=session.stop, daemon=True)
        stopper.start()
        stopper.join(10)
        if stopper.is_alive():
            stuck.append(session)
            continue
        worker = getattr(session, "_worker", None)
        if worker is not None:
            worker.retire()
    assert not stuck, "a session's stop() never returned: a lock is still held"


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


class _QueueWatch:
    """A session's `_demand`, reporting when the thread named `second`
    finds it held by another thread and has to wait for it."""

    def __init__(self, inner, queued) -> None:
        self._inner = inner
        self._queued = queued

    def __enter__(self):
        if not self._inner.acquire(blocking=False):
            if threading.current_thread().name == "second":
                self._queued.set()
            self._inner.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self._inner.release()


class TestOperatorDecisionsArriveTogether:
    """Operator and demand decisions arriving at the same moment.

    Each verb records its decision and then makes its base transition.
    Every test parks the FIRST caller at its base call (on the way in, or
    once it has returned) and lets it go only when the SECOND has either
    returned or is queued behind it on `_demand`. Both are signalled,
    never timed, so the window is always exercised. Once both have
    returned, the later decision must have won and the hold must agree
    with the session. An error in either thread is raised here.
    """

    @staticmethod
    def _park(monkeypatch, verb, *, after=False):
        """Park the thread named `first` once at the base `verb`."""
        real = getattr(LiveSession, verb)
        arrived = threading.Event()
        release = threading.Event()

        def hold_here():
            arrived.set()
            assert release.wait(10), "the test never released the parked call"

        def parked(self_, *args, **kwargs):
            mine = threading.current_thread().name == "first" and not arrived.is_set()
            if mine and not after:
                hold_here()
            out = real(self_, *args, **kwargs)
            if mine and after:
                hold_here()
            return out

        monkeypatch.setattr(LiveSession, verb, parked)
        return arrived, release

    @staticmethod
    def _call(name, function, *args, landed=None):
        outcome = {}

        def run():
            try:
                outcome["value"] = function(*args)
            except BaseException as exc:  # raised again in the test thread
                outcome["error"] = exc
            finally:
                if landed is not None:
                    landed.set()

        thread = threading.Thread(target=run, name=name, daemon=True)
        thread.start()
        return thread, outcome

    def _race(self, monkeypatch, session, *, park, first, second, after=False):
        """`first` parked at the base `park`; `second` lands or queues.

        Returns what `first` returned."""
        arrived, release = self._park(monkeypatch, park, after=after)
        landed = threading.Event()
        monkeypatch.setattr(session, "_demand", _QueueWatch(session._demand, landed))
        calls = [self._call("first", *first)]
        try:
            assert arrived.wait(5), "the first call never reached its base call"
            calls.append(self._call("second", *second, landed=landed))
            assert landed.wait(5), "the second call neither returned nor queued"
        finally:
            release.set()
        for thread, outcome in calls:
            thread.join(10)
            assert not thread.is_alive(), f"{thread.name} never returned"
            if "error" in outcome:
                raise outcome["error"]
        return calls[0][1].get("value")

    @staticmethod
    def _held_and_state(session):
        status = session.status()
        return status["demand"]["operator_hold"], status["state"]

    def test_a_stop_inside_a_start_leaves_it_stopped_and_unheld(self, monkeypatch):
        # hold-set / hold-clear / base-stop / base-start.
        session = _session()
        self._race(
            monkeypatch,
            session,
            park="start",
            first=(session.start,),
            second=(session.stop,),
        )
        # The Stop is the later decision, and it wins.
        assert self._held_and_state(session) == (False, STATE_STOPPED)

    def test_a_start_inside_a_stop_leaves_it_running_and_held(self, monkeypatch):
        # hold-clear / hold-set / base-start / base-stop.
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)
        returned = self._race(
            monkeypatch,
            session,
            park="stop",
            first=(session.stop,),
            second=(session.start,),
        )
        # The Start is the later decision, and it wins.
        held, state = self._held_and_state(session)
        assert held is True
        assert state in (STATE_STARTING, STATE_RUNNING)
        # And the Stop says so, rather than what its base stop left.
        assert returned["state"] in (STATE_STARTING, STATE_RUNNING)

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
            session,
            park="start",
            first=(session.start,),
            second=(session.stream_closed, "phone"),
        )
        assert self._held_and_state(session) == (False, STATE_STOPPED)

    def test_a_start_inside_a_demand_stop_still_wins(self, monkeypatch):
        # The window `_stop_then_reconcile` already covered, now from
        # another thread: whichever way the two run, the Start is the
        # later decision.
        session = TestTheOperatorPath._watched_then_unwatched_session()
        self._race(
            monkeypatch,
            session,
            park="stop",
            first=(session.watcher_left, "phone:sub-1"),
            second=(session.start,),
        )
        assert session.status()["demand"]["operator_hold"] is True
        assert _await_state(session, STATE_RUNNING)

    def test_a_later_stop_is_not_undone_by_an_older_one(self, monkeypatch):
        # Stop A, Start B, Stop C: C is the latest, and A landing last
        # must not bring B back.
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)

        def start_then_stop():
            session.start()
            session.stop()

        self._race(
            monkeypatch,
            session,
            park="stop",
            first=(session.stop,),
            second=(start_then_stop,),
        )
        assert self._held_and_state(session) == (False, STATE_STOPPED)

    def test_a_later_stream_closing_is_not_undone_by_an_older_stop(
        self, monkeypatch
    ):
        # Stop A, Start B, then the last stream closes: the closure is the
        # latest and releases B's hold. A landing last must not restart a
        # session nobody holds.
        session = _session()
        session.start()
        session.stream_opened(owner="phone")
        assert _await_state(session, STATE_RUNNING)

        def start_then_stream_closes():
            session.start()
            session.stream_closed(owner="phone")

        self._race(
            monkeypatch,
            session,
            park="stop",
            first=(session.stop,),
            second=(start_then_stream_closes,),
        )
        assert self._held_and_state(session) == (False, STATE_STOPPED)

    def test_a_pause_after_a_newer_start_survives_an_older_stop(self, monkeypatch):
        # Stop A, Start B, Pause P, then A's base stop: P is the latest,
        # so the session ends paused, not running and not stopped.
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)

        def start_then_pause():
            session.start()
            session.pause()

        self._race(
            monkeypatch,
            session,
            park="stop",
            first=(session.stop,),
            second=(start_then_pause,),
        )
        assert self._held_and_state(session) == (True, STATE_PAUSED)

    def test_a_pause_after_an_older_stops_base_stop_is_not_resumed(
        self, monkeypatch
    ):
        # A's base stop lands first, then Start B and Pause P, then A goes
        # on: settling A must leave P's pause alone.
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)

        def start_then_pause():
            session.start()
            assert _await_state(session, STATE_RUNNING)
            session.pause()

        self._race(
            monkeypatch,
            session,
            park="stop",
            after=True,
            first=(session.stop,),
            second=(start_then_pause,),
        )
        assert self._held_and_state(session) == (True, STATE_PAUSED)

    def test_a_pause_after_an_older_stop_does_not_revive_it(self, monkeypatch):
        # Stop A, then Pause P while A's base stop is still to land: in
        # that order the Pause found nothing to pause.
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)
        self._race(
            monkeypatch,
            session,
            park="stop",
            first=(session.stop,),
            second=(session.pause,),
        )
        assert self._held_and_state(session) == (False, STATE_STOPPED)

    def test_a_resume_after_an_older_pause_wins(self, monkeypatch):
        # Pause A, then Resume R before A's base pause lands: R is the
        # latest, so the session ends running, not paused.
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)
        self._race(
            monkeypatch,
            session,
            park="pause",
            # Late-bound: on a tree where `pause` is the base method, a
            # bound method taken now would miss the parking patch.
            first=(lambda: session.pause(),),
            second=(session.resume,),
        )
        assert self._held_and_state(session) == (True, STATE_RUNNING)

    def test_a_start_racing_shutdown_leaves_nothing_running(self, monkeypatch):
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)
        before = session.status()["session_id"]
        live = LiveCartridges(frame_consumers=[session], scene=session)
        self._race(
            monkeypatch,
            session,
            park="stop",
            first=(live.shutdown,),
            second=(session.start,),
        )
        assert self._held_and_state(session) == (False, STATE_STOPPED)
        assert session.status()["session_id"] == before, "a session began"
        # And nothing begins one afterwards.
        session.start()
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        session.resume()
        assert self._held_and_state(session) == (False, STATE_STOPPED)
        assert session.status()["session_id"] == before

    def test_a_watcher_arriving_inside_an_operator_stop_is_served(
        self, monkeypatch
    ):
        # Stop, then a stream and a watcher: in that order the facts say
        # run, but their `_reconcile` found the session still running.
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)

        def stream_and_watch():
            session.stream_opened(owner="phone")
            session.watcher_joined("phone:sub-1", owner="phone")

        self._race(
            monkeypatch,
            session,
            park="stop",
            first=(session.stop,),
            second=(stream_and_watch,),
        )
        held, state = self._held_and_state(session)
        assert held is False
        assert state in (STATE_STARTING, STATE_RUNNING)


    @staticmethod
    def _demand_session(held=False):
        session = _session()
        session.stream_opened(owner="phone")
        session.watcher_joined("phone:sub-1", owner="phone")
        if held:
            session.start()
        assert _await_state(session, STATE_RUNNING)
        return session

    @pytest.mark.parametrize("held", [False, True])
    def test_a_pause_after_a_watcher_inside_an_operator_stop_is_kept(
        self, monkeypatch, held
    ):
        # Stop A, a watcher joins, Pause P, then A's base stop. In that
        # order the watcher restarted the session and P paused it: P is
        # the newest decision, and the detector must not end up running.
        session = self._demand_session(held)

        def watcher_then_pause():
            session.watcher_joined("phone:sub-2", owner="phone")
            session.pause()

        self._race(
            monkeypatch,
            session,
            park="stop",
            first=(session.stop,),
            second=(watcher_then_pause,),
        )
        assert self._held_and_state(session) == (False, STATE_PAUSED)

    def test_a_watcher_after_a_pause_inside_an_operator_stop_runs(
        self, monkeypatch
    ):
        # The control: Stop A, Pause P (nothing left to pause), then the
        # watcher joins. The watcher is newest, so the session runs.
        session = self._demand_session()

        def pause_then_watcher():
            session.pause()
            session.watcher_joined("phone:sub-2", owner="phone")

        self._race(
            monkeypatch,
            session,
            park="stop",
            first=(session.stop,),
            second=(pause_then_watcher,),
        )
        held, state = self._held_and_state(session)
        assert held is False
        assert state in (STATE_STARTING, STATE_RUNNING)

    def test_a_pause_between_two_watchers_inside_an_operator_stop_is_kept(
        self, monkeypatch
    ):
        # Stop A, watcher W1, Pause P, watcher W2, then A's base stop. In
        # that order W1 restarted the session, P paused it and W2 could
        # not resume it: the newest arrival does not make this the J3 case.
        session = self._demand_session()

        def watcher_pause_watcher():
            session.watcher_joined("phone:sub-2", owner="phone")
            session.pause()
            session.watcher_joined("phone:sub-3", owner="phone")

        self._race(
            monkeypatch,
            session,
            park="stop",
            first=(session.stop,),
            second=(watcher_pause_watcher,),
        )
        assert self._held_and_state(session) == (False, STATE_PAUSED)

    def test_a_stream_alone_before_a_pause_inside_a_stop_does_not_pause_it(
        self, monkeypatch
    ):
        # Stop A (the watcher is gone), a stream with no watcher, Pause P,
        # then a watcher. Only the watcher could start the session, and it
        # is newer than P, so in order the session runs.
        session = _session()
        session.start()
        assert _await_state(session, STATE_RUNNING)

        def stream_pause_watcher():
            session.stream_opened(owner="phone")
            session.pause()
            session.watcher_joined("phone:sub-1", owner="phone")

        self._race(
            monkeypatch,
            session,
            park="stop",
            first=(session.stop,),
            second=(stream_pause_watcher,),
        )
        held, state = self._held_and_state(session)
        assert held is False
        assert state in (STATE_STARTING, STATE_RUNNING)

    def test_a_pause_after_a_watcher_inside_a_demand_stop_is_kept(
        self, monkeypatch
    ):
        # The demand-stop path: the last watcher leaves (stop D), a new
        # watcher joins, Pause P, then D's base stop. P is newest.
        session = self._demand_session()

        def watcher_then_pause():
            session.watcher_joined("phone:sub-2", owner="phone")
            session.pause()

        self._race(
            monkeypatch,
            session,
            park="stop",
            first=(session.watcher_left, "phone:sub-1"),
            second=(watcher_then_pause,),
        )
        assert self._held_and_state(session) == (False, STATE_PAUSED)


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


# -- every short ordering, checked against decision order ---------------
#
# A small-scope exhaustive check. Each kind of call that leaves `_demand`
# for a blocking base call is parked there; every ordering of up to three
# further events lands inside its window; then it goes on. The session
# must end where `_Model` -- the module's rules, written down here
# independently of `SceneLive` -- says those events leave it when taken
# one at a time in the order they were decided.

_WINDOW_EVENTS = (
    "start",
    "stop",
    "pause",
    "resume",
    "stream_open",
    "stream_close",
    "watcher_join",
    "watcher_leave",
    "shutdown",
)

# Parked call -> (where it is parked, the event it is). "stop" and
# "pause" are the base calls; "decided" is a demand stop just after it
# decides, before `_stop_then_reconcile` looks at the target again.
_PARKED = {
    "operator_stop": ("stop", "stop"),
    "operator_pause": ("pause", "pause"),
    "last_watcher_leaves": ("stop", "watcher_leave"),
    "last_stream_closes": ("stop", "stream_close"),
    "last_watcher_leaves_decided": ("decided", "watcher_leave"),
    "last_stream_closes_decided": ("decided", "stream_close"),
}
_PARK_AT = {
    "stop": (LiveSession, "stop"),
    "pause": (LiveSession, "pause"),
    "decided": (SceneLive, "_stop_then_reconcile"),
}


def _windows(longest=3):
    import itertools

    for length in range(longest + 1):
        yield from itertools.product(_WINDOW_EVENTS, repeat=length)


class _Model:
    """The rules, from the module's docstrings, one event at a time.

    Start holds the session and runs it (resuming a pause); Stop releases
    the hold and stops it; Pause pauses a running session; Resume resumes
    a paused one. A stream or watcher arriving starts a stopped session
    when there is a stream AND a watcher, and never resumes a pause. The
    last stream closing stops the session whoever started it; the last
    watcher leaving stops it unless the operator holds it. Shutdown stops
    it for good: nothing starts it afterwards.
    """

    def __init__(self) -> None:
        self.state = STATE_STOPPED
        self.hold = False
        self.closed = False
        self.streams: set = set()
        self.watchers: set = set()

    def _stop(self) -> None:
        self.hold = False
        self.state = STATE_STOPPED

    def _arrived(self) -> None:
        if not self.closed and self.streams and self.watchers:
            if self.state == STATE_STOPPED:
                self.state = STATE_RUNNING

    def apply(self, event, arg) -> None:
        if event == "start":
            if not self.closed:
                self.hold = True
                self.state = STATE_RUNNING
        elif event == "stop":
            self._stop()
        elif event == "pause":
            if self.state == STATE_RUNNING:
                self.state = STATE_PAUSED
        elif event == "resume":
            if not self.closed and self.state == STATE_PAUSED:
                self.state = STATE_RUNNING
        elif event == "stream_open":
            self.streams.add(arg)
            self._arrived()
        elif event == "watcher_join":
            self.watchers.add(arg)
            self._arrived()
        elif event == "stream_close":
            was = arg in self.streams
            self.streams.discard(arg)
            if was and not self.streams:
                self._stop()
        elif event == "watcher_leave":
            self.watchers.discard(arg)
            if not self.hold and not self.watchers:
                self._stop()
        elif event == "shutdown":
            self.closed = True
            self._stop()
        else:  # pragma: no cover - a typo in the table
            raise ValueError(event)


def _resolve(event, ids):
    """The argument an event takes. New streams and watchers get new
    names; a close or leave takes the newest one still there."""
    if event == "stream_open":
        ids["n"] += 1
        ids["streams"].append(f"s{ids['n']}")
        return ids["streams"][-1]
    if event == "watcher_join":
        ids["n"] += 1
        ids["watchers"].append(f"w{ids['n']}")
        return ids["watchers"][-1]
    if event == "stream_close":
        return ids["streams"].pop() if ids["streams"] else "s-none"
    if event == "watcher_leave":
        return ids["watchers"].pop() if ids["watchers"] else "w-none"
    return None


def _do(session, event, arg) -> None:
    if event == "start":
        session.start()
    elif event == "stop":
        session.stop()
    elif event == "pause":
        session.pause()
    elif event == "resume":
        session.resume()
    elif event == "stream_open":
        session.stream_opened(owner=arg)
    elif event == "stream_close":
        session.stream_closed(owner=arg)
    elif event == "watcher_join":
        session.watcher_joined(arg, owner="phone")
    elif event == "watcher_leave":
        session.watcher_left(arg)
    elif event == "shutdown":
        LiveCartridges(frame_consumers=[session], scene=session).shutdown()
    else:  # pragma: no cover
        raise ValueError(event)


def _in_thread(name, function, timeout=10.0):
    """Run `function` on a thread called `name`; its error, or a hang."""
    outcome = {}

    def run():
        try:
            function()
        except BaseException as exc:  # reported, never swallowed
            outcome["error"] = exc

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread, outcome


def _settled(session, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if session.state != STATE_STARTING:
            return True
        time.sleep(0.001)
    return False


def _check_case(parked, window, *, sequential=False):
    """Run one case. None if the session ends where `_Model` says, else
    a description. `sequential`: no parking, every event in turn (checks
    the model against the session's own behaviour without a race)."""
    where, first_event = _PARKED[parked]
    owner, verb = _PARK_AT[where]
    session = SceneLive(_Engine, decode=lambda raw: raw, follow_stream=True)
    real = owner.__dict__[verb]
    arrived, release = threading.Event(), threading.Event()
    model = _Model()
    ids = {"n": 0, "streams": [], "watchers": []}
    problem = None
    try:
        for event in ("stream_open", "watcher_join"):
            arg = _resolve(event, ids)
            _do(session, event, arg)
            model.apply(event, arg)
        if not _settled(session) or session.state != STATE_RUNNING:
            return f"setup did not run: {session.state}"

        first_arg = _resolve(first_event, ids)
        model.apply(first_event, first_arg)
        plan = [(event, _resolve(event, ids)) for event in window]
        for event, arg in plan:
            model.apply(event, arg)

        if sequential:
            _do(session, first_event, first_arg)
            for event, arg in plan:
                _do(session, event, arg)
        else:

            def parked_call(self_, *args, **kwargs):
                if threading.current_thread().name == "first" and not arrived.is_set():
                    arrived.set()
                    release.wait(10)
                return real(self_, *args, **kwargs)

            setattr(owner, verb, parked_call)
            first, first_outcome = _in_thread(
                "first", lambda: _do(session, first_event, first_arg)
            )
            try:
                if not arrived.wait(5):
                    return "the parked call never reached its base call"
                rest, rest_outcome = _in_thread(
                    "window", lambda: [_do(session, e, a) for e, a in plan]
                )
                rest.join(10)
                if rest.is_alive():
                    return "the window's events never returned"
            finally:
                release.set()
                setattr(owner, verb, real)
            first.join(10)
            if first.is_alive():
                return "the parked call never returned"
            for outcome in (first_outcome, rest_outcome):
                if "error" in outcome:
                    return f"raised {outcome['error']!r}"

        if not _settled(session):
            return "never settled"
        status = session.status()
        got = (status["demand"]["operator_hold"], status["state"])
        want = (model.hold, model.state)
        if got != want:
            problem = f"(hold, state) {got}, decision order gives {want}"
        return problem
    finally:
        setattr(owner, verb, real)
        stopper, _ = _in_thread("cleanup", session.stop)
        stopper.join(10)
        worker = getattr(session, "_worker", None)
        if worker is not None:
            worker.retire()


def _failures(parked, *, sequential=False):
    failures = []
    for window in _windows():
        problem = _check_case(parked, window, sequential=sequential)
        if problem is not None:
            failures.append((window, problem))
    return failures


class TestEveryShortOrderingEndsInDecisionOrder:
    """820 orderings per parked call (every sequence of up to three of
    the nine events), 4920 in all."""

    @pytest.mark.parametrize("parked", sorted(_PARKED))
    def test_the_model_is_the_sessions_own_behaviour_without_a_race(self, parked):
        failures = _failures(parked, sequential=True)
        assert not failures, f"{len(failures)} of 820 differ: {failures[:5]}"

    @pytest.mark.parametrize("parked", sorted(_PARKED))
    def test_a_window_of_up_to_three_events_ends_in_decision_order(self, parked):
        failures = _failures(parked)
        assert not failures, f"{len(failures)} of 820 differ: {failures[:5]}"


class TestAStartAfterAFailedLoadRecordsItsTarget:
    def test_a_start_that_keeps_a_pause_still_retries_a_failed_load_running(self):
        """A paused session whose load then fails is begun afresh by
        `start(resume_paused=False)`, and runs. Its decision-order target
        must say so, or the next `_drive` -- which every stop and pause
        ends with -- pauses the fresh session again."""
        loading = threading.Event()

        class _FailsOnce(_Engine):
            failed = False

            def load(self):
                if not _FailsOnce.failed:
                    _FailsOnce.failed = True
                    assert loading.wait(5), "the test never released the load"
                    raise RuntimeError("no weights")
                return None

        session = _session(_FailsOnce)
        session.start()
        session.pause()  # while the first load is in flight
        assert session.state == STATE_PAUSED
        loading.set()
        assert _await_state(session, STATE_FAILED)

        session.start(resume_paused=False)
        assert _await_state(session, STATE_RUNNING)
        session._drive()
        assert session.state == STATE_RUNNING
