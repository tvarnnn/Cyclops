"""A liveness claim must belong to the session making it.

THE DEFECT, WHICH WAS ALREADY DOCUMENTED AND NOT YET FIXED.

`docs/contracts/OBJECT-MEMORY.md` section 9.1 carries this warning:

    `following` and `captures` are supervisor-scoped, not
    session-scoped. Start a NEW session and it will report the OLD
    session's capture under the new `session_id`, having attached
    nothing. Under this defect that rule produces a FALSE POSITIVE: a
    brand-new session that attached nothing renders as recording.

It is reachable by the ordinary route. `_stop_worker` leaves a worker it
could not kill in the registry -- deliberately, so it stays visible to
`/health` and to the next shutdown -- and `following` then names its
capture forever. Press Stop against one of those, press Start again, and
the phone draws "remembering" for a session that started nothing and is
writing nothing.

THE FIX IS ADDITIVE, AND THAT IS THE POINT.

`following` keeps its exact meaning: EVERY live producer for this
worker. Narrowing it would hide the un-killable one, and an un-killable
producer is what "the Stop button failed open" looks like -- the single
worst thing this cartridge can do to a person and the one thing that
must never become silent. `intent_contradicts_liveness` is drawn from
it and stays drawn from it.

What is new is `following_this_session`: the subset started at or after
this session went active. That is the field a "you are being recorded by
what you just started" claim may be drawn from, and the two together say
something neither says alone.
"""

import pytest

from tower.cartridge_session import ACTIVE, PAUSED, STOPPED, CartridgeSession


class Supervisor:
    """A supervisor with a clock, which is the half that made this hard.

    `CaptureWorkerSupervisor._clock` defaults to `time.monotonic` and
    `CartridgeSession._clock` to `time.time`. Comparing a `started_at`
    from one against a timestamp from the other compares an uptime with a
    Unix epoch; the first implementation did exactly that, every worker
    looked older than every session, and the correct-looking code
    reported nothing at all. `mark()` exists so a caller cannot get the
    clock wrong, and this fake has one for the same reason the real thing
    does.
    """

    def __init__(self):
        self.now = 100.0
        self.workers: list[tuple[str, float]] = []
        self.unkillable = False
        self.detached: list[tuple[str, float]] = []
        self.attach_result = True

    # -- the supervisor surface a session uses --

    def worker_names(self):
        return ("worker",)

    def mark(self) -> float:
        return self.now

    def attach(self, name, capture_id, capture_dir):
        if not self.attach_result:
            return False
        self.workers.append((capture_id, self.now))
        return True

    def detach(self, name, grace_seconds=10.0):
        self.detached.append((name, grace_seconds))
        if self.unkillable:
            return 0
        self.workers = []

    def following(self, name, *, since=None):
        return [
            capture_id
            for capture_id, started_at in self.workers
            if since is None or started_at >= since
        ]

    # -- what a walk does to it --

    def tick(self, seconds=1.0):
        self.now += seconds

    def capture_opened(self, capture_id):
        """The supervisor spawning a worker because the gate was open.

        Not routed through `attach`, deliberately: this is the ordinary
        path and it does not consult the session at all, which is why a
        session that counted only its own `attach` returns would miss
        almost every real attachment.
        """
        self.workers.append((capture_id, self.now))


def _session(supervisor):
    ticks = iter(range(1, 10_000))
    return CartridgeSession(
        cartridge="object_memory",
        worker="worker",
        supervisor=supervisor,
        open_capture=lambda: None,
        clock=lambda: float(next(ticks)),
    )


class TestTheFalsePositive:
    def test_a_new_session_does_not_inherit_an_unkillable_producer(self):
        supervisor = Supervisor()
        session = _session(supervisor)

        # A first walk, remembered.
        session.apply("start")
        supervisor.capture_opened("cap-1")
        assert session.snapshot()["following_this_session"] == ["cap-1"]

        # Stop, against a producer that will not die.
        supervisor.unkillable = True
        session.apply("stop")
        stopped = session.snapshot()
        assert stopped["state"] == STOPPED
        # The alarm survives. This is the whole reason `following` is not
        # narrowed.
        assert stopped["following"] == ["cap-1"]
        # And it is still THIS session's producer. A Stop that failed to
        # kill one does not make it somebody else's problem.
        assert stopped["following_this_session"] == ["cap-1"]

        # A second walk. Nothing new attaches -- the gate is open but no
        # capture has opened yet, which is the normal state one moment
        # after pressing Start.
        supervisor.tick()
        session.apply("start")
        snapshot = session.snapshot()

        assert snapshot["state"] == ACTIVE
        assert snapshot["following"] == ["cap-1"], (
            "the leftover must stay visible somewhere"
        )
        assert snapshot["following_this_session"] == [], (
            "a session that attached nothing must not render as recording"
        )

    def test_a_capture_opened_after_start_is_this_session_s(self):
        """The ordinary path, and the one a naive fix breaks.

        Start before the camera is the documented normal order: the
        session goes active with `attached_capture_id: null` and the next
        capture to open finds the gate open. Nothing calls back into the
        session when that happens, so scoping by "ids this object
        returned from `_attach`" would report nothing for almost every
        real walk.
        """
        supervisor = Supervisor()
        session = _session(supervisor)

        session.apply("start")
        supervisor.tick()
        supervisor.capture_opened("cap-1")

        assert session.snapshot()["following_this_session"] == ["cap-1"]

    def test_a_capture_opened_in_the_same_instant_as_start_counts(self):
        """The mark is taken BEFORE the attach, not after.

        A mark taken afterwards can be later than the `started_at` of the
        worker the same call spawned, and the session would then disown
        the producer it had just started.
        """
        supervisor = Supervisor()
        session = _session(supervisor)

        # `open_capture` returning a live capture is what makes `_attach`
        # spawn during `start` itself, with no clock tick in between.
        supervisor_capture = ("cap-live", "/tmp/cap-live")
        session = CartridgeSession(
            cartridge="object_memory",
            worker="worker",
            supervisor=supervisor,
            open_capture=lambda: supervisor_capture,
            clock=lambda: 1.0,
        )

        result = session.apply("start")

        assert result["attached_capture_id"] == "cap-live"
        assert session.snapshot()["following_this_session"] == ["cap-live"]


class TestPauseAndResume:
    def test_resume_does_not_claim_a_producer_the_pause_failed_to_kill(self):
        supervisor = Supervisor()
        session = _session(supervisor)

        session.apply("start")
        supervisor.capture_opened("cap-1")
        supervisor.unkillable = True
        session.apply("pause")

        paused = session.snapshot()
        assert paused["state"] == PAUSED
        # The contradiction a client must show loudly: paused, and still
        # following.
        assert paused["following"] == ["cap-1"]
        assert paused["following_this_session"] == ["cap-1"], (
            "a producer a Pause did not kill is still this session's, and "
            "saying otherwise would hide it from the person who paused"
        )

        supervisor.tick()
        session.apply("resume")

        resumed = session.snapshot()
        assert resumed["state"] == ACTIVE
        # The Resume re-marked, so the survivor is no longer counted as
        # something this activation started -- but `following` still
        # names it, which is where the alarm lives.
        assert resumed["following_this_session"] == []
        assert resumed["following"] == ["cap-1"]

    def test_a_resume_that_really_attaches_is_claimed(self):
        supervisor = Supervisor()
        session = _session(supervisor)

        session.apply("start")
        supervisor.capture_opened("cap-1")
        session.apply("pause")
        supervisor.tick()
        session.apply("resume")
        supervisor.capture_opened("cap-2")

        assert session.snapshot()["following_this_session"] == ["cap-2"]


class TestTheHistory:
    def test_captures_records_only_what_this_session_followed(self):
        supervisor = Supervisor()
        session = _session(supervisor)

        session.apply("start")
        supervisor.capture_opened("cap-1")
        session.snapshot()
        supervisor.unkillable = True
        session.apply("stop")
        supervisor.tick()

        session.apply("start")
        supervisor.capture_opened("cap-2")
        snapshot = session.snapshot()

        assert snapshot["captures"] == ["cap-2"], (
            "the history is this session's, and a leftover it never "
            "started is not part of it"
        )


class TestDegradingSafely:
    """A supervisor without the new methods says NULL, not "none".

    THIS WAS `[]` FOR ONE ROUND, AND A REVIEWER WAS RIGHT ABOUT IT.

    The first version returned the empty list here, reasoning that a
    claim which cannot be scoped must not be made. The reasoning was
    right and the encoding was wrong: a client reads the empty list as a
    POSITIVE claim -- "none of the producers you can see are mine" -- and
    draws a loud warning from the difference between it and `following`.
    So a Tower that merely could not answer would have told a person that
    a producer they did not start was recording them and that Stop would
    not reach it, about a producer they had started themselves.

    Three values, and each says a different thing:

    - a list  -- these, and no others, are mine;
    - `[]`    -- I started nothing that is still running. A claim;
    - `null`  -- I cannot scope this. Fall back to `following`.
    """

    class Old:
        def __init__(self):
            self.following_ids = ["cap-1"]

        def worker_names(self):
            return ("worker",)

        def attach(self, name, capture_id, capture_dir):
            return False

        def detach(self, name, grace_seconds=10.0):
            return 0

        def following(self, name):
            return list(self.following_ids)

    def test_a_supervisor_with_no_mark_answers_null_rather_than_empty(self):
        supervisor = self.Old()
        session = _session(supervisor)

        session.apply("start")
        snapshot = session.snapshot()

        assert snapshot["following"] == ["cap-1"], (
            "nothing may become invisible"
        )
        assert snapshot["following_this_session"] is None, (
            "an unanswerable question is null; the empty list is an answer "
            "and would be the wrong one"
        )

    def test_the_history_still_accumulates_when_nothing_can_be_scoped(self):
        """`captures` is what it was before scoping existed.

        Accumulating from the scoped list would leave the history empty
        forever on a Tower that cannot scope, which is a second thing
        going quiet for one thing being unanswerable.
        """
        supervisor = self.Old()
        session = _session(supervisor)

        session.apply("start")

        assert session.snapshot()["captures"] == ["cap-1"]

    def test_a_session_that_was_never_started_claims_nothing(self):
        """The one case in which the empty list is unambiguous.

        Not null, deliberately, and not even on a supervisor that cannot
        scope: a session nobody has ever pressed Start on has started
        nothing, and that is knowable without a clock.
        """
        supervisor = self.Old()
        session = _session(supervisor)

        snapshot = session.snapshot()

        assert snapshot["state"] == STOPPED
        assert snapshot["following"] == ["cap-1"]
        assert snapshot["following_this_session"] == []


class TestTheWireStillCarriesBoth:
    @pytest.mark.parametrize("action", ["start", "pause", "stop"])
    def test_every_payload_carries_the_new_field(self, action):
        """A client that switches on its presence must never see it
        missing on one verb and present on another."""
        supervisor = Supervisor()
        session = _session(supervisor)
        session.apply("start")

        result = session.apply(action)

        assert "following" in result
        assert "following_this_session" in result
        assert result["following_this_session"] is not None, (
            "a supervisor that CAN scope must never answer null; null is "
            "reserved for a Tower that cannot answer the question at all"
        )


class TestASupervisorThatCannotAnswerAScopedQuestion:
    """A `TypeError` from `following(since=...)` must not widen the claim.

    `_following` widens on purpose -- it retries unscoped so an older
    supervisor can still answer the PUBLIC `following` field, and
    over-reporting what is running is safe there. Handing that same
    widened list back as `following_this_session` is the opposite: it
    tells a client "these are the producers YOU started" when the answer
    is "every producer on this Tower", and raises the loud
    something-else-is-recording warning about a recording the person
    started themselves.

    A reviewer found that the fallback did exactly that.
    """

    class Narrow:
        """Accepts `since` and then raises, which is the shape of an old
        supervisor seen through a modern call."""

        def __init__(self):
            self.calls = []

        def worker_names(self):
            return ("worker",)

        def mark(self):
            return 100.0

        def attach(self, name, capture_id, capture_dir):
            return False

        def detach(self, name, grace_seconds=10.0):
            return 0

        def following(self, name, *args, **kwargs):
            self.calls.append(kwargs)
            if "since" in kwargs:
                raise TypeError("following() got an unexpected keyword 'since'")
            return ["someone-elses-producer"]

    def test_it_answers_null_rather_than_the_unscoped_list(self):
        supervisor = self.Narrow()
        session = _session(supervisor)

        session.apply("start")
        snapshot = session.snapshot()

        assert snapshot["following"] == ["someone-elses-producer"], (
            "the wide field still reports everything, so nothing goes invisible"
        )
        assert snapshot["following_this_session"] is None, (
            "a claim that could not be scoped must not be answered with a "
            "list that was not scoped"
        )

    def test_the_scoped_question_was_actually_asked(self):
        supervisor = self.Narrow()
        session = _session(supervisor)
        session.apply("start")
        session.snapshot()

        assert any("since" in call for call in supervisor.calls), (
            "the session must try the scoped call before giving up on it"
        )
