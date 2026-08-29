"""Start, Pause, Stop for a cartridge that produces out of process.

The state machine exists because of a gap measured on real hardware. On
2026-08-26 a Ray-Ban walk produced 2,203 frames and 64 remembered
observations, and every one of those observations existed because a
human opened a second terminal, listed `data/captures` by modification
time, copied a hex id, and ran a producer script against it. There was
no Start. There was no Pause. There was nothing to press.

What a session is NOT:

  * it is not a capture. The capture is minted by the phone's
    `stream_start` and is shared by every cartridge; a session says what
    THIS cartridge does about it.
  * it is not persisted. A Tower that restarts comes back `stopped`,
    because resuming a memory of what a camera sees without anybody
    asking again is the wrong direction to fail in.
  * it is not the producer. It starts and stops one; whether the process
    is alive is the supervisor's answer, and the session reports it
    rather than believing it.
"""

import pytest

from tower.cartridge_session import (
    ACTIVE,
    PAUSED,
    STOPPED,
    CartridgeSession,
    SessionRefused,
)


class FakeSupervisor:
    """Records attach/detach without running anything.

    The real supervisor is tested against a fake Popen in
    `test_capture_workers_multi_spec.py`. Doubling it here keeps these
    tests about the STATE MACHINE, which is the part that decides whether
    a wearer's Pause is obeyed.
    """

    def __init__(self, *, attach_result=True, names=("worker",)):
        self.attached = []
        self.detached = []
        self.attach_result = attach_result
        self._names = names
        self._following = []
        # A clock, because `following(since=...)` is answered against the
        # SUPERVISOR's, never the session's -- the real ones default to
        # `time.monotonic` and `time.time` respectively, and comparing
        # across them compares an uptime with a Unix epoch. Frozen here:
        # nothing in this file is about ordering, and a frozen mark makes
        # every worker "started at or after" every activation, which is
        # what these cases mean by "the session's".
        self.now = 0.0

    def worker_names(self):
        return tuple(self._names)

    def mark(self):
        return self.now

    def attach(self, name, capture_id, capture_dir):
        self.attached.append((name, capture_id, str(capture_dir)))
        if self.attach_result:
            self._following.append(capture_id)
        return self.attach_result

    def detach(self, name, grace_seconds=10.0):
        self.detached.append((name, grace_seconds))
        count = len(self._following)
        self._following = []
        return count

    def following(self, name, *, since=None):
        return list(self._following)


def _session(supervisor=None, open_capture=None, clock=None):
    ticks = iter(range(1, 10_000))
    return CartridgeSession(
        cartridge="a_cartridge",
        worker="worker",
        supervisor=supervisor if supervisor is not None else FakeSupervisor(),
        open_capture=open_capture if open_capture is not None else (lambda: None),
        clock=clock if clock is not None else (lambda: float(next(ticks))),
    )


# -- the resting state -------------------------------------------------


def test_a_new_session_is_stopped():
    session = _session()

    assert session.state == STOPPED
    assert session.is_active() is False
    assert session.snapshot()["state"] == STOPPED


def test_a_stopped_session_gates_the_worker_closed():
    """The gate is the whole mechanism, so it is asserted directly.

    `main.py` hands `is_active` to the worker spec as its gate. If this
    ever returned True while stopped, a capture opening would attach a
    producer nobody started.
    """
    session = _session()

    assert session.is_active() is False


# -- starting ----------------------------------------------------------


def test_start_with_no_capture_open_waits_rather_than_failing():
    """Pressing Start before the camera is a normal order of operations.

    Refusing here would make the wearer press Start twice for reasons
    they cannot see, so the session goes ACTIVE and the gate opens; the
    next capture to open finds it open.
    """
    supervisor = FakeSupervisor()
    session = _session(supervisor)

    result = session.start()

    assert result["state"] == ACTIVE
    assert result["changed"] is True
    assert session.is_active() is True
    assert supervisor.attached == []
    assert result["attached_capture_id"] is None


def test_start_attaches_to_a_capture_that_is_already_recording(tmp_path):
    supervisor = FakeSupervisor()
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))

    result = session.start()

    assert supervisor.attached == [("worker", "cap1", str(tmp_path))]
    assert result["attached_capture_id"] == "cap1"


def test_start_twice_is_idempotent(tmp_path):
    """A double tap must not put two producers on one store."""
    supervisor = FakeSupervisor()
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))
    session.start()

    result = session.start()

    assert result["changed"] is False
    assert len(supervisor.attached) == 1


def test_start_records_when_it_started():
    session = _session(clock=lambda: 1000.0)

    session.start()

    assert session.snapshot()["started_at"] == 1000.0


# -- pausing and resuming ----------------------------------------------


def test_pause_detaches_the_worker_and_closes_the_gate(tmp_path):
    supervisor = FakeSupervisor()
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))
    session.start()

    result = session.pause()

    assert result["state"] == PAUSED
    assert session.is_active() is False
    assert [name for name, _ in supervisor.detached] == ["worker"]


def test_resume_reattaches_to_whatever_is_recording_now(tmp_path):
    """Resume attaches to the CURRENT capture, not the one Start found.

    A pause long enough to matter is a pause long enough for the phone to
    have reconnected, and a session that re-attached to the capture it
    remembered would follow a directory nothing is writing to any more.
    """
    open_id = "cap1"
    supervisor = FakeSupervisor()
    session = _session(supervisor, open_capture=lambda: (open_id, tmp_path))
    session.start()
    session.pause()
    open_id = "cap2"

    result = session.resume()

    assert result["state"] == ACTIVE
    assert supervisor.attached[-1] == ("worker", "cap2", str(tmp_path))


def test_pause_keeps_the_session_identity(tmp_path):
    """Pause is a gap in one session, not the end of it."""
    supervisor = FakeSupervisor()
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))
    started = session.start()["session_id"]
    session.pause()

    assert session.resume()["session_id"] == started


def test_pause_when_stopped_is_refused():
    """There is nothing to pause, and pretending otherwise hides a bug.

    A UI that shows Pause on a stopped cartridge has a state bug, and a
    Tower that answers "paused" to it would make the bug invisible.
    """
    session = _session()

    with pytest.raises(SessionRefused):
        session.pause()


def test_pause_twice_is_idempotent(tmp_path):
    supervisor = FakeSupervisor()
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))
    session.start()
    session.pause()

    result = session.pause()

    assert result["changed"] is False
    assert len(supervisor.detached) == 1


def test_resume_when_stopped_is_refused():
    session = _session()

    with pytest.raises(SessionRefused):
        session.resume()


# -- stopping ----------------------------------------------------------


def test_stop_detaches_and_forgets_the_session(tmp_path):
    supervisor = FakeSupervisor()
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))
    session.start()

    result = session.stop()

    assert result["state"] == STOPPED
    assert session.is_active() is False
    assert supervisor.detached != []
    assert session.snapshot()["session_id"] is None
    assert session.snapshot()["started_at"] is None


def test_stop_from_paused_works(tmp_path):
    supervisor = FakeSupervisor()
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))
    session.start()
    session.pause()

    assert session.stop()["state"] == STOPPED


def test_stop_when_already_stopped_is_idempotent():
    """Stop is the one action that is never refused.

    Whatever state a confused client believes it is in, "stop" is a
    request to end up stopped, and it always can be honoured.
    """
    session = _session()

    result = session.stop()

    assert result["state"] == STOPPED
    assert result["changed"] is False


def test_a_second_start_after_stop_is_a_new_session(tmp_path):
    supervisor = FakeSupervisor()
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))
    first = session.start()["session_id"]
    session.stop()

    assert session.start()["session_id"] != first


# -- what the snapshot may claim ---------------------------------------


def test_the_snapshot_reports_captures_this_session_touched(tmp_path):
    """Which recordings this session's producer actually followed.

    Accumulated from the supervisor rather than from what the session
    asked for: a worker the supervisor declined to start, or one that
    died, must not appear here as a capture that was remembered.
    """
    supervisor = FakeSupervisor()
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))
    session.start()

    assert session.snapshot()["captures"] == ["cap1"]


def test_a_capture_attached_by_the_gate_still_reaches_the_snapshot(tmp_path):
    """The gate path does not go through `start`, and must still be seen.

    A session started before the walk attaches nothing itself: the
    capture opens later and the supervisor consults the gate. The session
    learns about that capture the only honest way available to it, by
    asking the supervisor what it is actually following.
    """
    supervisor = FakeSupervisor()
    session = _session(supervisor)
    session.start()
    supervisor._following.append("cap-opened-later")

    assert session.snapshot()["captures"] == ["cap-opened-later"]


def test_the_snapshot_says_whether_a_producer_is_running(tmp_path):
    supervisor = FakeSupervisor()
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))

    assert session.snapshot()["following"] == []
    session.start()
    assert session.snapshot()["following"] == ["cap1"]


def test_an_active_session_whose_worker_died_says_so(tmp_path):
    """ACTIVE is a claim about intent; `following` is a claim about fact.

    Collapsing them would let the phone show "remembering" for the rest
    of a walk whose producer exited in the first ten seconds -- which is
    precisely the "looks successful but does nothing" failure the whole
    state surface exists to make visible.
    """
    supervisor = FakeSupervisor()
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))
    session.start()
    supervisor._following = []

    snapshot = session.snapshot()

    assert snapshot["state"] == ACTIVE
    assert snapshot["following"] == []


def test_a_failed_attach_is_reported_rather_than_swallowed(tmp_path):
    supervisor = FakeSupervisor(attach_result=False)
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))

    result = session.start()

    assert result["state"] == ACTIVE
    assert result["attached_capture_id"] is None


def test_an_exploding_capture_lookup_does_not_take_the_session_down(tmp_path):
    """A broken recorder must not make Start unpressable.

    The session still goes ACTIVE, because the gate is the part that
    matters and the next capture to open will find it open.
    """

    def boom():
        raise RuntimeError("no recorder")

    session = _session(open_capture=boom)

    assert session.start()["state"] == ACTIVE


def test_an_exploding_detach_still_changes_state(tmp_path):
    """Pause must be obeyed even if stopping the process fails.

    The gate closing is what stops the NEXT capture being followed. A
    session that refused to move because a terminate failed would leave
    the cartridge attaching to every future capture.
    """

    class BrokenSupervisor(FakeSupervisor):
        def detach(self, name, grace_seconds=10.0):
            raise RuntimeError("cannot stop")

    supervisor = BrokenSupervisor()
    session = _session(supervisor, open_capture=lambda: ("cap1", tmp_path))
    session.start()

    assert session.pause()["state"] == PAUSED
    assert session.is_active() is False


def test_a_session_for_a_worker_the_supervisor_does_not_have_is_honest():
    """A Tower with the cartridge disabled must not offer a working Start.

    `supported` is False, and starting is refused with a reason -- not
    accepted into a state where nothing can ever attach.
    """
    supervisor = FakeSupervisor(names=())
    session = _session(supervisor)

    assert session.snapshot()["supported"] is False
    with pytest.raises(SessionRefused):
        session.start()
