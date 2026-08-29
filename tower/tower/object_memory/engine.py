"""One frame in, zero or more observations out.

The producer's whole logic, deliberately separate from the script that
drives it. `scripts/object_memory_session.py` chooses where frames come
from and prints a report; everything that decides what gets REMEMBERED
lives here, where it can be tested against detections a test wrote down.

Nothing here is a `Module`. Object Memory produces out of process, by
tailing a capture journal, exactly as World Builder, Scene and Document
Memory do -- so it needs no module lifecycle, no `_do_load` timeout and
no second module slot, and the unresolved lifecycle-versus-model-load
question does not stand in its way. The Tower now ATTACHES that process
automatically (`tower/capture_workers.py`), which is a different question
and was the actual product gap.

THE FUNNEL, AND WHERE EACH STAGE'S COST GOES.

    every frame        decode + detector          40-47 ms, CPU, measured
    every detection    four deterministic gates   microseconds
    every sighting     one record, once           an append
    some sightings     a semantic second opinion  off the frame path

The validated capture is 186 seconds long and produced 4,287 detections.
The stage that costs real money is asked ONCE PER SIGHTING and only for
classes the detector cannot be trusted to name -- 53 times across the
whole 18,821-frame corpus, one call per 355 frames, and four times on
that one capture. Every one of those figures is counted at runtime and
printed, because a funnel whose narrowing is not measured is a funnel
that has quietly stopped narrowing.

The range is the measurement. Five consecutive replays of the same
capture with no other work of ours running spanned **39.8 to 46.9
ms/frame**; two earlier runs that overlapped our own benchmarks gave 66
and 69. This host carries several autonomous agent lanes at once, so even
the lower figures are not an idle-machine number -- and an earlier draft
of this docstring quoting a flat "68 ms" was quoting one of the contended
runs as if it were.

WHY A RECORD IS WRITTEN EARLY AND UPDATED LATER.

A sighting is only complete when it ends, and a session can be killed at
any moment. So the record is written as soon as the sighting is old
enough to be real -- three frames, ~250 ms -- stamped with the FIRST
frame, and what the sighting later becomes is folded back into it. The
alternative, writing at the end, loses the whole sighting whenever a walk
is interrupted, which on this hardware is often.

Updates are deliberately rare: a stronger look, a slow tick, and the end
of the sighting. `ObservationStore.update_sighting` rewrites the file, so
updating per frame would make the store O(n) per frame for no gain.

WHERE THE ONE PERSISTED PICTURE IS MADE.

This engine used to persist no pixels. It now writes, per recorded
sighting and only when a keyframe store is configured, a single small
filtered crop under the OBSERVATION root -- so a 30-day record stops
depending on a capture directory nobody promised to keep. That happens
at exactly one point, `_settle`, between the final `_refresh` and
`forget_imagery`, which is the one moment the record names the strongest
look AND the pixels of that look are still in hand. See `keyframes.py`
for what may be written and what may not.
"""

import logging
import time

import cv2
import numpy as np

from tower.object_memory.records import (
    Confidence,
    ObjectObservation,
    observation_id_for,
    privacy_tags_for,
)
from tower.object_memory.relevance import (
    DROP_REASONS,
    RECORD,
    TOO_BRIEF,
    UNVERIFIED,
    RelevanceFilter,
    RelevancePolicy,
)
from tower.object_memory.classes import tier_of
from tower.object_memory.sightings import Look, SightingTracker

logger = logging.getLogger(__name__)

MODULE_ID = "object-memory"

# Rule 16. There is no on-glasses capture timestamp anywhere on this
# wire -- `tower/frames.py`'s REQUIRED_FIELDS carries no time field -- so
# every record this slice writes is stamped with the clock it can
# actually know. If a capture timestamp is ever threaded through, new
# records say "capture" and these stay correctly labelled.
TIME_BASIS = "tower-receipt"

# How often an already-written record is refreshed with what its sighting
# has since become, in seconds of the sighting's own clock.
#
# Ten seconds, because the cost is a whole-file rewrite and the benefit
# is only visible if the session dies mid-sighting. A sighting that ends
# normally is updated at its end regardless; this bounds what an
# interrupted one loses to ten seconds of duration, not to the sighting.
UPDATE_EVERY_SECONDS = 10.0

# How much context is included around a box before a crop is handed to a
# verifier. A tight crop of a 3%-of-frame object is unreadable, and the
# surroundings are most of what distinguishes a remote on a sofa from a
# laptop keyboard on a bed.
CROP_PADDING = 0.35


class ObjectMemoryEngine:
    """Detect, group into sightings, filter, persist. Counts every refusal.

    `verification` is optional and its absence is a real configuration,
    not a degraded one: with no verifier the `verify` tier is never
    written, which reproduces exactly the behaviour that shipped and was
    physically validated.

    `keyframes` and `face_filter` are optional in exactly the same sense,
    and default to today's behaviour: with no keyframe store this engine
    persists no pixels at all, which is what every test written before
    `keyframes.py` existed asserts and what the shipped Tower did. With
    one, each recorded sighting leaves behind a small filtered crop under
    the OBSERVATION root, governed by the observation store's retention
    rather than by the capture's -- see `keyframes.py` for why a durable
    record pointing into an ephemeral store had to stop being the whole
    answer.

    Both are needed together. A keyframe store with no usable filter
    writes nothing and counts the refusal; there is no configuration in
    which an unfiltered crop reaches disk.
    """

    def __init__(
        self,
        store,
        detector,
        *,
        policy: RelevancePolicy | None = None,
        verification=None,
        keyframes=None,
        face_filter=None,
        source: str = "glasses-camera",
        session_id: str | None = None,
        clock=time.time,
    ) -> None:
        self._store = store
        self._detector = detector
        self._policy = policy or RelevancePolicy()
        self._relevance = RelevanceFilter(self._policy)
        self._verification = verification
        self._keyframes = keyframes
        self._face_filter = face_filter
        self._source = source
        self._session_id = session_id
        self._clock = clock
        self._tracker = SightingTracker(
            gap_seconds=self._policy.gap_seconds,
            min_frames=self._policy.min_frames,
        )
        # Sighting -> (best score at last write, sighting-clock time of
        # last write). Kept beside the sighting rather than on it,
        # because it is bookkeeping about the STORE and a sighting is a
        # record of what the camera did.
        self._last_written: dict[int, tuple[float, float]] = {}
        # Sightings that ENDED while a verdict was still in flight.
        #
        # Without this they are lost in silence. `_collect_verdicts` runs
        # at the top of every frame and a verdict that arrives between
        # frames reaches its sighting -- but a sighting whose class has
        # been out of view for longer than the gap window is closed and
        # dropped from the tracker, and a verdict arriving after that
        # lands on an object nothing looks at again.
        #
        # It cannot happen at the measured rate: verdicts take 128 ms on
        # CUDA and the gap window is three seconds. It CAN happen on a
        # verifier running on CPU at 2.5 seconds a crop with a queue in
        # front of it, which is a supported configuration, and "supported
        # but silently lossy" is not a state worth shipping.
        #
        # Bounded by the queue's own backlog limit, because nothing gets
        # in here that was not submitted.
        self._awaiting_verdict: list = []

        self.frames_observed = 0
        self.frames_undecodable = 0
        self.detections_seen = 0
        self.sightings_opened = 0
        self.sightings_closed = 0
        self.observations_recorded = 0
        self.write_failures = 0
        self.recorded_by_class: dict[str, int] = {}
        self.sighting_updates = 0
        self.update_failures = 0
        self.verification_requested = 0
        # How many records got a picture of their own, and why the rest
        # did not.
        #
        # A dict keyed by reason rather than a single failure count, for
        # the same reason `dropped` is: "wrote 9 keyframes" means nothing
        # without "and refused 2 because this host has no face-detection
        # weights", which is an operator-fixable configuration problem
        # and not the same event as a full disk. The keys are
        # `keyframes.REFUSAL_REASONS`; it starts empty rather than
        # zero-filled because a run with no keyframe store configured
        # should report nothing rather than a table of zeroes implying it
        # tried.
        self.keyframes_written = 0
        self.keyframes_refused: dict[str, int] = {}
        # Why detections did NOT become observations. Reported rather
        # than discarded: "the producer wrote 11 records" means nothing
        # without "and declined 4,000, mostly for classes it has no
        # evidence it can read".
        self.dropped: dict[str, int] = {reason: 0 for reason in DROP_REASONS}

    # -- lifecycle ----------------------------------------------------

    def load(self) -> None:
        self._detector.load()
        if self._verification is not None:
            self._verification.start()

    def release(self) -> None:
        """Close every open sighting, collect late verdicts, then let go.

        Order matters and it is the opposite of the obvious one. Stopping
        the verifier first would throw away verdicts already paid for;
        closing sightings first gives every mature one its last chance to
        be written.
        """
        try:
            self.finish()
        finally:
            if self._verification is not None:
                self._verification.stop()
            self._detector.release()

    def finish(self) -> None:
        """End of frames. Settle everything still open.

        Separate from `release` so a test -- and a replay -- can settle
        the session without tearing down a model it may want again.
        """
        if self._verification is not None:
            # One last chance for the queue to catch up before anything
            # is closed, so a sighting that matured on the final frame is
            # not refused for a verdict that was already in flight.
            self._verification.wait_idle()
            self._collect_verdicts()
        # Captured before `close_all` empties the tracker, for the same
        # reason as in `observe`.
        open_classes = self._open_classes()
        for sighting in self._tracker.close_all():
            self.sightings_closed += 1
            self._settle(sighting, open_classes)
        # Anything still parked never got its verdict -- the queue
        # dropped it, or the verifier hung past `wait_idle`. Settled
        # anyway, so it is refused for a reason the counters record
        # rather than vanishing from the accounting entirely.
        for sighting in list(self._awaiting_verdict):
            self._awaiting_verdict.remove(sighting)
            self._settle(sighting, open_classes)

    # -- the frame path ------------------------------------------------

    def observe(
        self,
        raw_bytes: bytes,
        *,
        received_at: float | None = None,
        source_seq: int | None = None,
        relpath: str | None = None,
    ) -> list[ObjectObservation]:
        frame = self._decode(raw_bytes)
        if frame is None:
            self.frames_undecodable += 1
            return []
        self.frames_observed += 1

        height, width = frame.shape[:2]
        # received_at is the capture journal's receipt time, which is the
        # same clock TIME_BASIS names. A source with no timestamps -- a
        # directory of loose jpegs -- gets the processing clock instead;
        # inventing an interval to make it look like a real one would
        # fabricate a clock (Rule 3).
        observed_at = self._clock() if received_at is None else received_at

        # Verdicts first: one that arrived since the last frame may be
        # what lets a sighting be written on this one.
        self._collect_verdicts()
        # Then stale sightings, using the FRAME's clock rather than the
        # wall clock, so a replay closes them at the times the recording
        # implies.
        #
        # The open set is captured BEFORE the close. `close_stale`
        # removes what it closes, and `_settle` asks whether a part's
        # whole was in view -- against an empty set that question always
        # answers no, and the duplicate the part-of rule exists to
        # prevent gets written at the end of every sighting.
        open_classes = self._open_classes()
        for sighting in self._tracker.close_stale(observed_at):
            self.sightings_closed += 1
            self._settle(sighting, open_classes)

        recorded = []
        for detection in self._detector.detect(frame):
            self.detections_seen += 1
            verdict = self._relevance.decide(detection.label, detection.score)
            if verdict != RECORD:
                self.dropped[verdict] = self.dropped.get(verdict, 0) + 1
                continue
            x1, y1, x2, y2 = detection.box
            look = Look(
                score=detection.score,
                # Stored as a fraction of the frame, not in pixels. The
                # record carries no resolution, so a pixel box would mean
                # different things in different captures and nothing
                # would say which. This is still a box in an IMAGE -- it
                # is not a position in a room.
                box=(x1 / width, y1 / height, x2 / width, y2 / height),
                at=observed_at,
                frame_seq=source_seq,
                relpath=relpath,
                width=width,
                height=height,
            )
            # The crop is made HERE, on the frame the look came from,
            # and kept only if this look turns out to be the best one.
            # Cropping later -- at verification time, or at the end of
            # the sighting -- would use whatever frame happened to be
            # current and hand a model, or a wearer, a picture of
            # something else.
            #
            # MADE FOR EVERY SIGHTING NOW, not only verify-tier ones.
            #
            # This used to read `if self._verification is not None and
            # self._relevance.needs_verification(...)`, which was right
            # while a crop's only consumer was a verifier. Its effect was
            # that `laptop` and `cell phone` -- the two REMEMBERED-tier
            # classes, the only two a Tower with no verifier writes, and
            # the two the physical walk actually produced -- never held a
            # crop at all. A keyframe store hung off that condition would
            # have had nothing to write for exactly the records that
            # exist.
            #
            # Every detection that reaches this line has already passed
            # `RelevanceFilter.decide` with RECORD, so its class is one
            # this cartridge may persist; there is no widening here of
            # what may be remembered, only of which admitted sightings
            # hold their pixels. The per-sighting bound is unchanged --
            # one crop per OPEN sighting, replaced only by a stronger
            # look, released at `_settle` -- so the memory ceiling is
            # still "one crop per class currently in view".
            #
            # It costs one padded `numpy` copy per admitted detection.
            # The frame path already pays 40-47 ms for decode and
            # detection; a sub-frame copy is not visible against that,
            # and the corpus admits ~1.3 detections per frame after the
            # gates rather than the 4,287 it saw.
            crop = self._crop(frame, look.box)
            sighting, opened = self._tracker.observe(detection.label, look, crop)
            if opened:
                self.sightings_opened += 1
            observation = self._consider(sighting)
            if observation is not None:
                recorded.append(observation)
        return recorded

    # -- deciding ------------------------------------------------------

    def _consider(self, sighting) -> ObjectObservation | None:
        """Write, request a second opinion, refresh, or do nothing."""
        verdict = self._relevance.decide_sighting(sighting, self._open_classes())
        if verdict == RECORD:
            return self._write(sighting)
        if verdict == UNVERIFIED:
            self._request_verification(sighting)
            self.dropped[UNVERIFIED] = self.dropped.get(UNVERIFIED, 0) + 1
            return None
        if verdict == TOO_BRIEF:
            # Not counted as a drop. A sighting one frame old has not
            # been refused; it has not finished being made. Counting it
            # would report thousands of refusals per walk that describe
            # nothing.
            return None
        if sighting.recorded:
            self._refresh(sighting)
            return None
        self.dropped[verdict] = self.dropped.get(verdict, 0) + 1
        return None

    def _open_classes(self) -> frozenset:
        """What else is in view right now.

        Only the classes, not the sightings: the one rule that reads this
        asks whether a whole is present while a part is being considered,
        and handing it the sightings would invite something to start
        comparing boxes -- which is a spatial claim this cartridge does
        not make.
        """
        return frozenset(self._tracker.open_sightings)

    def _request_verification(self, sighting) -> None:
        """Ask once per sighting, when it matures, with its best look.

        Once, not once per frame. A sighting that has been asked about is
        in flight until a verdict arrives, and re-asking every frame in
        that window would queue the same crop a dozen times -- which is
        how a funnel becomes a fan, and how a bounded queue starts
        dropping work it has already done.
        """
        if self._verification is None:
            return
        if sighting.verification_requested or sighting.verdict is not None:
            return
        if sighting.best_crop is None:
            return
        sighting.verification_requested = True
        self.verification_requested += 1
        self._verification.submit(sighting, sighting.best_crop)

    def _collect_verdicts(self) -> None:
        """Attach arrived verdicts, and settle anything that was waiting.

        A sighting still open is reconsidered on its next frame, as
        usual. A sighting that ENDED while its verdict was in flight has
        no next frame, so it is settled here -- which is the whole reason
        `_awaiting_verdict` exists.
        """
        if self._verification is None:
            return
        for sighting, verdict in self._verification.drain():
            sighting.verdict = verdict.to_json_dict()
            if sighting in self._awaiting_verdict:
                self._awaiting_verdict.remove(sighting)
                # This sighting has already left the tracker, so what is
                # open now is the honest answer to "is its whole still in
                # view" -- the alternative would be to remember the set
                # from the moment it closed, which is state with a
                # shorter half-life than the bug it would prevent.
                self._settle(sighting, self._open_classes())

    # -- writing -------------------------------------------------------

    def _write(self, sighting) -> ObjectObservation | None:
        observation = self._observation(sighting)
        try:
            self._store.append(observation)
        except OSError:
            # `recorded` is deliberately NOT set: the producer must not
            # believe it wrote a sighting that never reached disk, or the
            # next frame suppresses the retry and the observation is lost
            # in silence.
            self.write_failures += 1
            logger.warning(
                "[Tower][ObjectMemory] failed to persist a %s observation",
                sighting.object_class,
            )
            return None
        sighting.recorded = True
        self._last_written[id(sighting)] = (sighting.best.score, sighting.last.at)
        self.observations_recorded += 1
        self.recorded_by_class[sighting.object_class] = (
            self.recorded_by_class.get(sighting.object_class, 0) + 1
        )
        return observation

    def _refresh(self, sighting, *, force: bool = False) -> None:
        """Fold the sighting's progress back into the record on disk.

        Rate-limited, because each call rewrites the store. `force` is
        the sighting's end, which is always worth one write: it is what
        turns "seen at 14:03" into "seen at 14:03, for 4.4 seconds,
        across 29 frames".
        """
        key = id(sighting)
        written_score, written_at = self._last_written.get(key, (0.0, 0.0))
        improved = sighting.best.score > written_score
        stale = (sighting.last.at - written_at) >= UPDATE_EVERY_SECONDS
        if not (force or improved or stale):
            return
        try:
            changed = self._store.update_sighting(
                sighting.object_class,
                sighting.first.at,
                best_score=sighting.best.score,
                last_seen_at=sighting.last.at,
                frame_count=sighting.frame_count,
                best_frame_seq=sighting.best.frame_seq,
                best_relpath=sighting.best.relpath,
                best_bounding_box=sighting.best.box,
                verification=sighting.verdict,
            )
        except OSError:
            # Counted, not swallowed, and separate from write_failures: a
            # failed update loses an improvement to an honest record
            # already safely on disk, which is not the same accident as
            # losing the record itself.
            self.update_failures += 1
            logger.warning(
                "[Tower][ObjectMemory] failed to update a %s observation",
                sighting.object_class,
            )
            return
        self._last_written[key] = (sighting.best.score, sighting.last.at)
        if changed:
            self.sighting_updates += 1

    def _settle(self, sighting, open_classes) -> None:
        """A sighting has ended. Write it if it earned it, then let it go.

        `open_classes` is captured by the CALLER, before the close that
        emptied the tracker. Recomputing it here would ask "is this
        part's whole in view" of a set that no longer contains anything,
        which always answers no.

        Unless the sighting is still waiting for a verdict it has already
        paid for, in which case it is parked until the verdict arrives
        and settled then. Deciding now would refuse it for `unverified`
        when the answer is already in flight.
        """
        if (
            sighting.verification_requested
            and sighting.verdict is None
            and self._verification is not None
            and sighting not in self._awaiting_verdict
        ):
            self._awaiting_verdict.append(sighting)
            return
        if self._relevance.decide_sighting(sighting, open_classes) == RECORD:
            self._write(sighting)
        if sighting.recorded:
            self._refresh(sighting, force=True)
            # HERE, and only here, because this is the one moment both
            # halves are true at once: the record on disk now names the
            # strongest look (that is what the forced `_refresh` above
            # just did), and the pixels of that look are still in hand
            # (the `forget_imagery` below is what ends that). Writing
            # earlier would pin a picture a later frame improves on;
            # writing later would have nothing to write.
            self._write_keyframe(sighting)
        self._last_written.pop(id(sighting), None)
        # The pixels this cartridge holds in memory, released as soon as
        # the sighting they belong to can no longer use them. What
        # survives, when a keyframe store is configured, is the filtered
        # copy `_write_keyframe` just made -- never this array.
        sighting.forget_imagery()

    def _write_keyframe(self, sighting) -> None:
        """Give this record a picture of its own, or count why it has none.

        Never raises and never blocks the caller on a failure: a keyframe
        is an improvement to a record that is already safely on disk, and
        a walk must not end because a JPEG could not be written. Every
        outcome lands in a counter, so a run that wrote 11 records and no
        pictures says so in its report rather than looking identical to
        one that wrote 11 of each.

        The observation id is DERIVED from the same three values
        `_observation` stamps the record with -- session, class, and the
        first look's time -- which is the same derivation
        `records.observation_id_for` and the HTTP handle use. That is
        what makes the file findable by a reader that has only the
        record: there is no second identifier to keep in step.
        """
        if self._keyframes is None:
            return
        observation_id = observation_id_for(
            self._session_id, sighting.object_class, sighting.first.at
        )
        # CONTAINED, because the docstring above promises it and because
        # the cost of being wrong is not a missing thumbnail. This runs
        # inside `_settle`, which runs inside the frame loop: an escape
        # here ends the producer and takes `engine.release()` with it, so
        # one unwritable keyframe would cost every sighting still open --
        # which is the failure the whole graceful-stop change exists to
        # prevent, arriving by a different door.
        #
        # `KeyframeStore.write` is itself exception-tight now. This is
        # the second wall, and a reviewer found the first one had gaps:
        # a filter returning an unexpected shape raised straight through.
        try:
            result = self._keyframes.write(
                observation_id,
                sighting.best_crop,
                self._face_filter,
                source_capture=self._session_id,
                source_relpath=sighting.best.relpath,
                written_at=self._clock(),
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[Tower][ObjectMemory] the keyframe store raised for a %s "
                "record; the record itself is written and this walk goes on",
                sighting.object_class,
            )
            self.keyframes_refused["store-raised"] = (
                self.keyframes_refused.get("store-raised", 0) + 1
            )
            return
        if result.written:
            self.keyframes_written += 1
            return
        reason = result.reason or "unknown"
        self.keyframes_refused[reason] = self.keyframes_refused.get(reason, 0) + 1

    def _observation(self, sighting) -> ObjectObservation:
        first = sighting.first
        best = sighting.best
        return ObjectObservation(
            object_class=sighting.object_class,
            # PROVENANCE: how confident the detector was in the frame
            # that FIRST brought this class into view -- the one frame
            # this record's observed_at, frame_seq and bounding_box all
            # describe. It never moves.
            detector_score=first.score,
            # Derived, never asserted -- and derived from the best look,
            # which at the first write is usually a later frame than the
            # first. A stronger look still to come moves it, in
            # ObservationStore.update_sighting.
            confidence=Confidence.from_score(best.score),
            observed_at=first.at,
            time_basis=TIME_BASIS,
            recorded_at=self._clock(),
            source=self._source,
            module_id=MODULE_ID,
            session_id=self._session_id,
            frame_seq=first.frame_seq,
            bounding_box=first.box,
            retention_tag="default",
            # Not a flat ("derived-only",): the record holds no imagery,
            # but session_id + frame_seq resolves to a frame under
            # data/captures/ that this cartridge's retention does not
            # govern. See records.privacy_tags_for.
            privacy_tags=privacy_tags_for(self._session_id, first.frame_seq),
            # A box in an IMAGE is not a position in a room, and this is
            # the only place that could ever put one on disk. The wire
            # nulls it unconditionally, the shipped iOS decoder REFUSES a
            # populated value, and prune preserves unknown keys -- so a
            # value written here would be invisible to every reader and
            # permanent. It stays None until a world exists to anchor it
            # in and a contract exists to carry it.
            spatial_ref=None,
            external_refs=(),
            best_score=best.score,
            last_seen_at=sighting.last.at,
            frame_count=sighting.frame_count,
            # The strongest look, which is usually a different frame from
            # the first and is the one worth showing a person.
            best_frame_seq=best.frame_seq,
            best_relpath=best.relpath,
            best_bounding_box=best.box,
            tier=tier_of(sighting.object_class),
            verification=sighting.verdict,
        )

    # -- frames --------------------------------------------------------

    @staticmethod
    def _crop(frame, box):
        """The padded region a box names, in pixels, or None.

        Padded because a tight crop of a 3%-of-frame object is
        unreadable, and because the published evidence on small-crop
        classification is unusually consistent: pad and upscale, never
        crop tight.
        """
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (
            box[0] * width,
            box[1] * height,
            box[2] * width,
            box[3] * height,
        )
        box_w, box_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        x1 = int(max(0, x1 - CROP_PADDING * box_w))
        y1 = int(max(0, y1 - CROP_PADDING * box_h))
        x2 = int(min(width, x2 + CROP_PADDING * box_w))
        y2 = int(min(height, y2 + CROP_PADDING * box_h))
        if x2 <= x1 or y2 <= y1:
            return None
        # Copied, not sliced: the crop outlives this frame on a queue,
        # and a numpy view would keep the whole frame alive with it --
        # eight pending crops holding eight full frames is a slow leak
        # with a semantic model attached to it.
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def _decode(raw_bytes: bytes):
        """Decode, or return None. A bad frame must not end a session.

        The corpus benchmark decoded all 18,821 real frames without a
        failure, so this is not a hot path -- but a producer that dies on
        one truncated JPEG loses the rest of the walk.
        """
        if not raw_bytes:
            return None
        array = np.frombuffer(raw_bytes, dtype=np.uint8)
        try:
            return cv2.imdecode(array, cv2.IMREAD_COLOR)
        except cv2.error:
            return None

    # -- reporting -----------------------------------------------------

    def counters(self) -> dict:
        report = {
            "frames_observed": self.frames_observed,
            "frames_undecodable": self.frames_undecodable,
            "detections_seen": self.detections_seen,
            "sightings_opened": self.sightings_opened,
            "sightings_closed": self.sightings_closed,
            "observations_recorded": self.observations_recorded,
            "recorded_by_class": dict(self.recorded_by_class),
            "sighting_updates": self.sighting_updates,
            "verification_requested": self.verification_requested,
            "declined": dict(self.dropped),
            "write_failures": self.write_failures,
            "update_failures": self.update_failures,
        }
        if self._keyframes is not None:
            # Only when a keyframe store is configured. A run that was
            # not asked to keep imagery should not report a zero that
            # reads as a failure to keep it.
            report["keyframes_written"] = self.keyframes_written
            report["keyframes_refused"] = dict(self.keyframes_refused)
        if self._verification is not None:
            report["verification"] = self._verification.counters()
        return report
