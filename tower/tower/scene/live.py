"""Scene Understanding as a live session: Start, Pause, Stop, and a clock.

`SceneEngine` is pull-shaped -- hand it a decoded frame, get a state back,
keep nothing. That is the right shape for a script and the wrong shape for
a web process, which owns neither the loop nor the timing.

The lifecycle, the single-slot frame path and the abandoned-load latch are
all `tower/live_session.py`, shared with Document Memory because both
cartridges need exactly the same three properties around an engine that is
synchronous, blocking and expensive. What is HERE is only what is specific
to this cartridge, and two things are.

**Stop must mean stopped.** On `stop()` the last scene is DISCARDED, not
retained. A scene held past the end of a session and served to a client is
the failure this cartridge is least able to survive: it is a claim about
what is around a person right now, and it would be wrong. No staleness
number is large enough to make it safe, because a client that renders
counts above staleness shows the room first.

`pause()` is the deliberately different case -- the scene survives and the
payload says it is no longer being refreshed.
`ios/docs/agent-handoffs/IOS-TO-TOWER.md` §4.7 asks the Tower to keep
`observing` and `lastKnown` apart rather than flatten them, and this is that
distinction.

Document Memory's Stop does the opposite and keeps what it recorded. That
asymmetry is the difference between the two cartridges, not an
inconsistency: a scene expires the moment nobody is looking, and a record
of what was read does not.

WHEN IT RUNS: STREAM AND WATCHER, OR OPERATOR

A scene session runs while **somebody is streaming AND somebody is
watching**. Two facts, tracked apart, neither sufficient alone:

  * the STREAM is the feed. `stream_opened` / `stream_closed` arrive from
    the connection that sends frames. With no stream there is nothing to
    observe, so the last stream closing stops the session whoever started
    it (otherwise a Stop -> Start over HTTP leaves a session owned by
    nobody that its own stream can no longer end).
  * a WATCHER is the demand. `watcher_joined` / `watcher_left` arrive from
    the result channel when a client subscribes to the live scene. A phone
    streaming for World Builder or the CV Lab is not a request to detect
    people in the room; a subscription to this cartridge's live result is.
    When the last watcher leaves, the detector is released and the GPU is
    handed back.

The OPERATOR path is separate and deliberately so. `start()` -- reached
from `POST /scene/start` -- runs the session at once, with or without a
stream, so a physical test can be driven from a Mac or curl without a
phone build; `stop()` ends that hold. `follow_stream=False`
(`TOWER_SCENE_AUTOSTART=false`) keeps ONLY the operator path.

No demand event ever resumes a Pause. A Pause is a person's decision; a
socket connecting or a screen opening is not.

PERSISTS NOTHING, ENFORCED

`test_scene_understanding_persists_nothing` AST-walks every file on this
cartridge's wire path -- this one, the adapter, the route, the runtime
factory and the shared session base -- and fails on any call that could
write. Frames arrive here as bytes, are decoded into an array, are
counted, and are dropped.
"""

import threading

from tower.live_session import (
    LIFECYCLE_STATES,
    LOAD_OVERDUE_S,
    STATE_FAILED,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
    STOP_JOIN_TIMEOUT_S,
    LiveSession,
    decode_frame,
)

# Published on the status so a client can see WHY a session is or is not
# running, in one word it can pin.
RUNS_WHEN = "stream-and-watcher-or-operator"

# The events `decision_order_state` knows.
EVENT_START = "start"
EVENT_STOP = "stop"
EVENT_PAUSE = "pause"
EVENT_RESUME = "resume"
EVENT_DEMAND = "demand"


def decision_order_state(
    state: str, event: str, *, can_run: bool = False, resume_paused: bool = True
) -> str:
    """Where `event` leaves a session that the events before it left in
    `state`, taking every event in the order it was decided. Pure.

    This is the one rule. `SceneLive` folds it over its events as each
    one is decided under `_demand`, into `_target`, and every path that
    finishes a stop or a pause outside that lock brings the session to
    `_target` and nothing else (`_drive`).

    `state` and the result are `STATE_STOPPED`, `STATE_RUNNING` or
    `STATE_PAUSED`. The events:

      EVENT_START   the operator's Start: running, or still paused with
                    `resume_paused=False`.
      EVENT_STOP    any stop: the operator's, a demand stop (the last
                    watcher or stream gone) or shutdown. Stopped.
      EVENT_PAUSE   pauses a running session; nothing to pause otherwise.
      EVENT_RESUME  resumes a paused session; nothing otherwise.
      EVENT_DEMAND  a stream or watcher arrived. Starts a stopped session
                    when it leaves a stream AND a watcher (`can_run`), and
                    never resumes a paused one: a Pause is a person's
                    decision; a socket connecting is not.
    """
    if event == EVENT_STOP:
        return STATE_STOPPED
    if event == EVENT_START:
        if state == STATE_PAUSED and not resume_paused:
            return STATE_PAUSED
        return STATE_RUNNING
    if event == EVENT_PAUSE:
        return STATE_PAUSED if state == STATE_RUNNING else state
    if event == EVENT_RESUME:
        return STATE_RUNNING if state == STATE_PAUSED else state
    if event == EVENT_DEMAND:
        return STATE_RUNNING if state == STATE_STOPPED and can_run else state
    raise ValueError(f"unknown event {event!r}")


__all__ = [
    "LIFECYCLE_STATES",
    "RUNS_WHEN",
    "LOAD_OVERDUE_S",
    "STATE_FAILED",
    "STATE_PAUSED",
    "STATE_RUNNING",
    "STATE_STARTING",
    "STATE_STOPPED",
    "STOP_JOIN_TIMEOUT_S",
    "SceneLive",
    "decision_order_state",
    "decode_frame",
]


class SceneLive(LiveSession):
    """One Scene Understanding session, driven by frames from elsewhere.

    Constructed with a factory rather than an engine, because `start()`
    after a `stop()` must build a fresh engine: `SceneEngine.release()`
    releases the models and resets the tracker, and re-loading a released
    detector is not a path this repository supports -- `tower/loading.py`
    is explicit that "FAILED is terminal in this repo's lifecycle". A new
    session being a new object graph is also what "session-scoped track
    ids" already promised.
    """

    name = "Scene"

    def __init__(
        self,
        engine_factory,
        *,
        decode=decode_frame,
        follow_stream: bool = True,
        **kwargs,
    ) -> None:
        # `follow_stream` defaults ON here and OFF on the base, and the
        # difference is the cartridge. Enabling Scene Understanding is
        # already the opt-in and it writes nothing; iOS sends nothing when
        # a cartridge is opened, so a second manual step would leave the
        # product path dead. TOWER_SCENE_AUTOSTART=false restores manual
        # control, and the five routes work either way.
        #
        # Document Memory deliberately defaults it OFF: it writes to
        # disk, and a session that persists gets an explicit start.
        super().__init__(follow_stream=follow_stream, **kwargs)
        self._engine_factory = engine_factory
        self._decode = decode
        self._latest = None
        self._latest_observed_at: float | None = None
        self._latest_computed_at: float | None = None
        self._decode_failures = 0
        # Demand. Held under `_demand`, a lock of their own, and acted on
        # with that lock still held so two demand events cannot interleave
        # into a decision neither of them made -- a watcher leaving and
        # another joining, resolved as "stop" then "start", must not land
        # as "start" then "stop". `stop()` and `start()` take it first
        # too, so the order is always demand -> lifecycle -> condition and
        # can never invert.
        self._demand = threading.RLock()
        # Connections currently streaming frames. NOT cleared by a stop:
        # a stream that is open stays open whether or not anyone is
        # watching it.
        self._open_streams: set = set()
        # Result-channel subscriptions to the live scene, token -> owner
        # connection, so a connection closing can drop every watcher it
        # held without knowing their tokens.
        self._watchers: dict = {}
        # Whether an operator started this by hand. Cleared by `stop()`, by
        # a demand stop and by `refuse_starts()`.
        self._operator_hold = False
        # Where the session's events leave it in decision order:
        # `decision_order_state` folded over every event as it is decided
        # under `_demand`. See DECISION ORDER below.
        self._target = STATE_STOPPED
        # Set once, by `refuse_starts()`, before the shutdown stop. Nothing
        # starts a worker after it.
        self._closed = False

    def latest(self):
        """The most recent `SceneState`, or None, with when it was taken.

        Returns `(state, observed_at, computed_at)`. `None` is a real
        answer and means one of four things the caller must keep apart:
        the session is stopped, it is still loading, it failed, or it is
        running and has not finished a frame yet. `status()` says which.

        The state is safe to hold: `SceneEngine.observe` copies the
        tracker's tracks (`tests/test_scene_snapshot_isolation.py`), so
        nothing this returns changes underneath a publisher.
        """
        with self._condition:
            return (self._latest, self._latest_observed_at, self._latest_computed_at)

    # -- demand --------------------------------------------------------
    #
    # `_demand` guards the three facts and the DECISION. It is never held
    # across `super().stop()` or `super().pause()`: both wait on
    # `_lifecycle`, which a stop holds across its join for up to
    # `STOP_JOIN_TIMEOUT_S`, and `stream_opened` / `watcher_joined` are
    # called inline on the event loop -- so a lock held across the join
    # would freeze every connection's frames, pings and `/health` while
    # one phone's session wound down. Only a base start or resume, which
    # return at once (the load is off-thread), is made under it.
    #
    # DECISION ORDER. Every event -- the operator's Start, Stop, Pause and
    # Resume, a stream or watcher arriving, a demand stop, shutdown --
    # records its effect on `_target` under `_demand`, with
    # `decision_order_state`, and makes its base call there if it does
    # not block. A base stop or pause runs outside the lock, so it can
    # land after NEWER events and undo them. Whoever made it then calls
    # `_drive`, which brings the session to `_target`. The final state is
    # the one the events give in the order they were decided; at worst
    # one session churns.

    def start(self, *, resume_paused: bool = True) -> dict:
        """The operator's Start. Runs at once, stream or no stream.

        Decided and made in one step under `_demand`, so no stop can
        decide between them. Refused after `refuse_starts()`.
        """
        with self._demand:
            if self._closed:
                return self.status()
            self._operator_hold = True
            # A failed session is begun afresh by the base start, paused
            # target or not, so it is folded from stopped.
            prior = STATE_STOPPED if self.state == STATE_FAILED else self._target
            self._target = decision_order_state(
                prior, EVENT_START, resume_paused=resume_paused
            )
            return super().start(resume_paused=resume_paused)

    def stop(self) -> dict:
        """The operator's Stop, and any stop but a demand stop. Releases the
        hold. Returns the status once `_drive` has run, not as the base
        stop left it.
        """
        with self._demand:
            self._operator_hold = False
            self._target = decision_order_state(self._target, EVENT_STOP)
        super().stop()
        self._drive()
        return self.status()

    def pause(self) -> dict:
        """The operator's Pause. See `LiveSession.pause`."""
        with self._demand:
            if self.state != STATE_FAILED:
                # A failed load has nothing to pause, as in the base.
                self._target = decision_order_state(self._target, EVENT_PAUSE)
        super().pause()
        self._drive()
        return self.status()

    def resume(self) -> dict:
        """The operator's Resume. Refused after `refuse_starts()`."""
        with self._demand:
            if self._closed:
                return self.status()
            self._target = decision_order_state(self._target, EVENT_RESUME)
            return super().resume()

    def refuse_starts(self) -> None:
        """The app is shutting down: no later start, operator or demand,
        starts a worker. `LiveCartridges.shutdown()` calls it, then `stop()`.

        Set with the hold released and the target stopped, under
        `_demand`, so a Start racing the shutdown is either stopped by that
        stop or refused. Terminal for this object, which `create_app()`
        builds once per app: a second lifespan on the same app would get a
        Scene that never starts.
        """
        with self._demand:
            self._closed = True
            self._operator_hold = False
            self._target = decision_order_state(self._target, EVENT_STOP)

    def _drive(self) -> None:
        """After a base stop or pause made OUTSIDE `_demand`, bring the
        session to `_target`.

        Running: resume a pause, or start a stopped session. Paused: pause
        it, starting it first if it was stopped. Stopped: nothing; the
        newest stop's own base stop has landed or is still to land. A
        failed load is left failed: nothing here retries it, though an
        older stop landing after it turns it into STOPPED, which is
        started again (a churn, like any restart here). Every start is made
        under `_demand`, never after `refuse_starts()`. A pause waits on
        `_lifecycle`, so it runs outside the lock and the loop asks again;
        each pass that does not return answers a stop that landed since.
        """
        while True:
            with self._demand:
                if self._closed:
                    return
                state = self.state
                target = self._target
                if target == STATE_STOPPED or state == STATE_FAILED:
                    return
                if target == STATE_RUNNING:
                    if state == STATE_PAUSED:
                        super().resume()
                    elif state == STATE_STOPPED:
                        super().start(resume_paused=False)
                    return
                if state == STATE_PAUSED:
                    return
                if state == STATE_STOPPED:
                    super().start(resume_paused=False)
            super().pause()

    def stream_opened(self, owner=None) -> None:
        """A connection began streaming frames. Run if somebody is watching."""
        with self._demand:
            self._open_streams.add(owner)
            self._reconcile()

    def stream_closed(self, owner=None) -> None:
        """A streaming connection ended. The last one out stops the session.

        Whoever started it. Frames arrive only from a stream, so a session
        with no stream has nothing left to do -- and, for this cartridge,
        a scene kept past its feed is a claim about a room the wearer has
        left. A connection that never streamed is not an owner and does
        not count, so a passing socket cannot end an operator's session.
        """
        with self._demand:
            was_streaming = owner in self._open_streams
            self._open_streams.discard(owner)
            last_one_out = was_streaming and not self._open_streams
            if last_one_out:
                self._demand_stop_decided()
        if last_one_out:
            self._stop_then_reconcile()

    def watcher_joined(self, token, *, owner=None) -> None:
        """A client subscribed to the live scene. Run if a stream is open."""
        with self._demand:
            self._watchers[token] = owner
            self._reconcile()

    def watcher_left(self, token) -> None:
        """That subscription ended. The last watcher out stops the session.

        Unless an operator is holding it: a Mac driving a physical test
        keeps its session across the phone's screen changes.
        """
        with self._demand:
            self._watchers.pop(token, None)
            unwatched = not self._operator_hold and not self._watchers
            if unwatched:
                self._demand_stop_decided()
        if unwatched:
            self._stop_then_reconcile()

    def watchers_left(self, *, owner) -> None:
        """A connection closed; every watcher it held leaves with it."""
        with self._demand:
            for token in [t for t, o in self._watchers.items() if o == owner]:
                del self._watchers[token]
            unwatched = not self._operator_hold and not self._watchers
            if unwatched:
                self._demand_stop_decided()
        if unwatched:
            self._stop_then_reconcile()

    def _demand_stop_decided(self) -> None:
        """A demand stop, decided. Under `_demand`."""
        self._operator_hold = False
        self._target = decision_order_state(self._target, EVENT_STOP)

    def _stop_then_reconcile(self) -> None:
        """A demand stop's base stop, OUTSIDE the demand lock, then `_drive`.

        The base stop is skipped when an event since the decision has
        already set the target running or paused again, so a Start or a
        watcher landing in between costs no churn. `_drive` is not: a
        Pause in between can have paused a session that a later watcher
        set running again, and its base start does not resume.
        """
        with self._demand:
            skip = self._target != STATE_STOPPED
        if not skip:
            super().stop()
        self._drive()

    def _reconcile(self) -> None:
        """A stream or watcher arrived: record it, and start if the facts
        say so. Under `_demand`.

        `resume_paused=False`, and a session whose target is paused is
        not started: a demand event is a socket or a screen, not a person,
        and must not undo a Pause. Starting from FAILED is allowed because
        each call here is one discrete request -- a new watcher, a new
        stream -- and gets one new attempt; nothing here loops, so a load
        that keeps failing fails once per request. `start()` itself
        returns at once (the load is off-thread), so holding the lock
        across it costs nothing. Never after `refuse_starts()`.
        """
        if not self._follow_stream or self._closed:
            return
        can_run = bool(self._open_streams and self._watchers)
        self._target = decision_order_state(
            self._target, EVENT_DEMAND, can_run=can_run
        )
        if can_run and self._target == STATE_RUNNING:
            super().start(resume_paused=False)

    # -- hooks ---------------------------------------------------------

    def _create(self):
        engine = self._engine_factory()
        engine.load()
        return engine

    def _engine_name(self, engine):
        return getattr(getattr(engine, "_detector", None), "name", None)

    def _consume(self, engine, raw_bytes, received_at, source_seq):
        frame = self._decode(raw_bytes)
        if frame is None:
            with self._condition:
                self._decode_failures += 1
            # `None` travels on like any other result and `_publish`
            # ignores it. Counting the failure and continuing is the same
            # answer `SceneEngine._detect` already gives a bad frame.
            return None
        return engine.observe(frame, received_at=received_at)

    def _publish(self, result, received_at: float, now: float) -> None:
        if result is None:
            # A frame that would not decode is not an observation. Undo
            # the increment the base made on our behalf, so
            # `frames_observed` counts frames that produced a scene
            # rather than frames that were taken off the slot.
            self._frames_observed -= 1
            return
        self._latest = result
        self._latest_observed_at = received_at
        self._latest_computed_at = now

    def _on_start_locked(self) -> None:
        self._decode_failures = 0
        self._latest = None
        self._latest_observed_at = None
        self._latest_computed_at = None

    def _on_stop_locked(self) -> None:
        """The load-bearing three lines in this class.

        See the module header. A cartridge whose entire claim is "what is
        around you NOW" must not answer that question after it stopped
        looking, and a state left in a field is exactly how it would.
        """
        self._latest = None
        self._latest_observed_at = None
        self._latest_computed_at = None

    def _extra_status(self) -> dict:
        now = self._clock()
        return {
            "detector": getattr(self, "_engine_label", None),
            "decode_failures": self._decode_failures,
            # When the FRAME was received, not when the detector finished
            # with it. A 30 ms detection would otherwise make every scene
            # look 30 ms fresher than it is, and the error grows with the
            # cost of the model.
            "observed_at": self._latest_observed_at,
            "computed_at": self._latest_computed_at,
            "staleness_seconds": (
                None
                if self._latest_observed_at is None
                else max(now - self._latest_observed_at, 0.0)
            ),
            "has_state": bool(self._latest is not None),
            "follows_stream": bool(self._follow_stream),
            # Why the session is, or is not, running. Counts rather than
            # tokens: a subscription id or a connection token on the wire
            # would be a handle to something this payload should not name.
            "demand": {
                "streams": len(self._open_streams),
                "watchers": len(self._watchers),
                "operator_hold": bool(self._operator_hold),
                "runs_when": RUNS_WHEN,
            },
        }
