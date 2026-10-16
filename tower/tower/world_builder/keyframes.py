"""Which frames become permanent observations, and why.

Capture rate and mapping rate are different things. At the delivered
~3.3 fps most frames carry no new information, and at a future 12 fps
almost none will. The system, not the wearer, decides which ordinary
frames earn the expensive path -- that is what reconciles "the wearer
behaves normally" with "we cannot afford to process everything".

Three thresholds -- `min_survival_ratio`, `loss_survival_ratio` and
`min_overlap_ratio` -- are measured on one physical walk and documented
with those numbers at their definitions; they are still not "tuned",
because one walk is one walk. The rest are unmeasured starting points, in
the same spirit as tower/object_memory/relevance.py's confidence buckets,
and the V0.9.3 acceptance gate forbids citing synthetic numbers as
validation for the platform's own camera.

## Why this policy does NOT decide degeneracy

It does not gate on the homography residual from
frontend.summarise_motion. The idea is attractive -- a homography exactly
explains pure rotation, so a near-zero residual would mean "no baseline,
do not bother" -- but it does not work.

Measured residual / median-displacement across motion types:

    rotation only,  1 deg          0.01110
    rotation only,  4 deg          0.00758
    rotation only, 12 deg          0.00581
    strafe 0.05 m                  0.17785   <- real translation
    strafe 0.30 m                  0.01299   <- real translation
    rotation 2 deg + 0.15 m        0.00812   <- real translation
    rotation 4 deg + 0.25 m        0.00500   <- real translation, LOWEST

The pure-rotation and genuinely-translating distributions OVERLAP, so no
threshold separates them. The reason is the same one that makes ORB-SLAM's
r_H saturate in a room: when a head both turns and moves, the rotation
dominates the pixel motion and a homography absorbs it, leaving a residual
that reflects only the comparatively small translation-induced parallax.
Realistic wearable motion is rotation-dominant: V0.9.3 measured 53.69%
of consecutive pairs degenerate.

So the responsibilities split:

- this policy answers "is this frame usable, and has enough happened?"
  from cheap, reliable, intrinsics-free signals -- sharpness, track
  survival, overlap, displacement;
- the geometry backend answers "does this pair carry triangulatable
  geometry?" using median triangulation angle, which needs intrinsics and
  separates cleanly (0.11-0.42 deg degenerate vs 0.54+ usable).

When intrinsics are unknown the second question is simply unanswerable,
which is precisely why the uncalibrated backend returns no poses rather
than guessing. The residual is still measured and persisted per keyframe,
because it is a real observation and costs nothing -- it is just not a gate.
"""

from dataclasses import dataclass

from tower.world_builder.frontend import FrameQuality, MotionSummary

# Decision outcomes.
ACCEPT = "accept"
SKIP = "skip"
REJECT = "reject"
TRACKING_LOST = "tracking_lost"

# Reasons. Every decision carries one, and they are counted into the
# session record as a rejection histogram, the tuning instrument.
REASON_SESSION_SEED = "session_seed"
REASON_PARALLAX = "parallax"
REASON_OVERLAP_FLOOR = "overlap_floor"
REASON_BLURRED = "blurred"
REASON_TRACKING_DEGRADED = "tracking_degraded"
REASON_TRACKING_LOST = "tracking_lost"
# Below the loss floor, but inside the grace window -- the track may yet
# recover. Counted under its own name rather than folded into
# `tracking_degraded`, so how often grace fires is visible.
REASON_TRACKING_HELD = "tracking_held"
REASON_INSUFFICIENT_MOTION = "insufficient_motion"
REASON_RATE_LIMITED = "rate_limited"
REASON_NO_MOTION_EVIDENCE = "no_motion_evidence"
# A frame that failed the blur gate, accepted anyway because the view has
# turned far enough since the last keyframe that skipping it would leave a
# gap no matcher can cross. Only ever produced with
# `KeyframePolicy.turn_bridge` on.
REASON_TURN_BRIDGE = "turn_bridge"


@dataclass(frozen=True)
class KeyframePolicy:
    """Thresholds for keyframe selection.

    Frozen so a policy is a value that can be recorded alongside the
    results it produced -- a sweep that cannot say which policy produced
    which numbers is not a sweep.
    """

    # Absolute blur floor. Measured: 662 sharp, 38 at a 5-px blur kernel.
    # Set well below the sharp value because absolute variance-of-Laplacian
    # is strongly scene-dependent; the ratio test below does the real work.
    min_sharpness: float = 25.0

    # Consecutive frames below `loss_survival_ratio` required before a
    # segment is actually broken.
    #
    # At 1 a single frame ends the segment, and on a replayed walk (a
    # different capture from the 1395-frame walk below) that verdict is
    # usually wrong: of the 50 declared losses, 47 still had survival above
    # the floor measured against the PREVIOUS frame, and 40 were above
    # 0.20. Only 3 were genuine. The rest were a stale reference asked to
    # cross a gap in one hop, where the next frame would have tracked fine.
    #
    # DEFAULT 1 -- the mechanism is here and tested, but 3 was measured
    # and REJECTED.
    #
    # Segment count said 3 was a clear win: 130 -> 99 across the eight
    # largest captures. The full solve says otherwise. Against reach-only,
    # on five real captures:
    #
    #             segments   poses_solved   points
    #   reach       114          265         42100
    #   + grace 3    96          178         27262
    #
    # It buys 18 segments and destroys a third of the reconstruction. The
    # mechanism is clear: a held frame neither advances the reference nor
    # becomes a keyframe, so holding through three frames means more
    # staleness and fewer keyframes at exactly the moment correspondence is
    # already struggling. The segment survives; the geometry inside it does
    # not.
    #
    # Segment count alone cannot tell which you have: a loss floor of 0.0
    # gives ONE segment and 37 keyframes, and one segment containing
    # nothing is worse than 51 containing something. Anything tuning this
    # must report poses_solved and points beside it. 2 is untested.
    loss_grace_frames: int = 1
    # Reject a frame markedly blurrier than its recent neighbours, which
    # catches motion blur during a head turn even in a scene whose
    # absolute sharpness is low throughout.
    min_sharpness_ratio: float = 0.55
    sharpness_window: int = 30

    # Track survival, and the overlap floor that rescues a decaying
    # chain. Below `min` the measurements are untrustworthy; below `loss`
    # the reference frame is simply gone; below `overlap` correspondence
    # is about to be lost and a weak keyframe beats a broken chain.
    #
    # They are NOT three independent knobs. Measured on one physical walk
    # (Ray-Ban Meta -> iPhone -> Tower, 1395 frames, replayed
    # bit-identically through the real FrameTracker and the real
    # KeyframeSelector: same 155 keyframes, same 35 losses, same four-way
    # rejection histogram, zero delta against every persisted sharpness,
    # survival_ratio and overlap_ratio):
    #
    #   overlap 0.45 / minsurv 0.35 / loss 0.15   36 segs, 155 kf, 10 single-kf
    #   overlap 0.75 / minsurv 0.20 / loss 0.05   20 segs, 260 kf,  4 single-kf
    #
    # `overlap_ratio` is not an independent signal from `survival_ratio`
    # on real footage: the two are EQUAL in 1283 of 1358 measured frames,
    # max gap 0.029, because tracks die rather than leave frame. An
    # overlap floor of 0.45 sitting above a survival reject at 0.35 can
    # therefore only fire in the band survival in [0.35, 0.45) -- 36
    # frames out of 1395, of which 28 were already being accepted for
    # parallax anyway. At those values the "a usable weak link beats a
    # broken chain" gate was very nearly dead, and the chain broke 36
    # times in one 122-second walk.
    #
    # So the rescue window is widened from the TOP, not the bottom.
    # Raising the overlap floor takes the rescue keyframe while tracking
    # is decaying but still alive; lowering the reject and loss floors
    # alone would only postpone the break, because by the time survival
    # reaches 0.15 the useful correspondence has already gone. Lowering
    # them as well keeps still-measurable frames out of a segment break,
    # but the top of the window is what does the work.
    #
    # NOT loosened, on the same measurement: the blur thresholds below
    # and the gate ORDER in evaluate(). `min_sharpness_ratio` 0.45 gives
    # 43 segments, 0.00 (blur gate off) gives 49, and moving the
    # survival/overlap gates ahead of blur gives 40 -- all WORSE than the
    # 36-segment baseline. 77% of blur rejections occur when survival is
    # ALREADY below 0.15: blur was masking losses that had already
    # happened, not causing them. The standing hypothesis, "spurious
    # `blurred` rejections cascading into `tracking_lost`", is refuted.
    #
    # A hypothesis with evidence behind it, not a tuned constant:
    #
    #   * ONE walk, one room, one wearer, one lighting condition. A
    #     second walk in a different space should reproduce the ordering
    #     before these numbers are treated as settled.
    #   * It costs +68% keyframes (155 -> 260) and that cost is
    #     UNBUDGETED. It falls on keyframe JPEG storage, on the 20.5 ms
    #     per keyframe of face redaction that sits on the write path, and
    #     on build time, which is roughly O(N^1.2) per rebuild.
    #   * It shifts the dominant promotion path from parallax to
    #     track-decay: overlap_floor accepts go from 28 of 155 to 198 of
    #     260. Whether track-decay keyframes triangulate as well as
    #     parallax keyframes is UNMEASURED, and was unmeasurable on this
    #     walk -- intrinsics were unknown, so BACKEND_AUTO selected the
    #     unposed backend and the world has 0 solved poses to check
    #     against. An ORB plus fundamental-matrix proxy says the extra
    #     pairs are not garbage (median inlier ratio 0.627 -> 0.649, no
    #     pair below MIN_INLIERS); that is not the same as saying the
    #     reconstruction is better.
    min_survival_ratio: float = 0.20
    loss_survival_ratio: float = 0.05

    # The top of the rescue window. See the measurement above: this is
    # the constant that closed 16 of the 36 segment breaks, and the
    # ordering loss < min_survival < min_overlap is what makes the
    # window exist at all.
    min_overlap_ratio: float = 0.75

    # Motion, in fractions of the image diagonal, so the policy transfers
    # across resolutions.
    target_displacement_frac: float = 0.035
    min_displacement_frac: float = 0.010

    # Deliberately no rotation/degeneracy gate here (see the module
    # docstring): the geometry backend decides, with intrinsics and
    # triangulation angle.

    # Rate limiter only -- never an accept trigger. See below.
    min_frame_gap: int = 1
    # Stall detector. Crossing it reports that mapping has stalled; it
    # does NOT promote a frame.
    max_frame_gap: int = 90

    # THE TURN BRIDGE. The World Builder engine's default policy turns it
    # on (engine.py); the field stays off by default so a bare
    # `KeyframePolicy()` is the policy above exactly.
    #
    # Measured on walk 9: the blur gate runs before the
    # overlap-floor rescue, so in a head turn every frame between two sharp
    # ones is rejected and consecutive keyframes end up 52-391 deg apart.
    # The camera's horizontal FOV is only ~45 deg; under 15 deg of rotation
    # 99-100% of consecutive pairs verify, over 50 deg 11%. All 9 of walk
    # 9's component breaks sit on such a gap. Half the floor's rejects are
    # low-texture (white wall, carpet), not blurred at all.
    #
    # The bridge applies the "a weak link beats a broken chain" rule to the
    # frames the blur gate rejected: a blurred frame is accepted when the
    # view has turned far enough since the last keyframe that waiting for a
    # sharp one would lose correspondence. "Far enough" is measured with
    # the tracker's own numbers (no extra work on the ingest path): track
    # overlap against the last keyframe below `bridge_overlap_ratio` (with
    # a ~45 deg FOV, overlap 0.5 is roughly 20-25 deg of yaw), or a median
    # track displacement above `bridge_displacement_frac` of the diagonal.
    # When nothing can be measured (no reference, too few tracks) a run of
    # `bridge_max_frames` blurred frames takes one anyway, unless the
    # tracker positively says the camera is still.
    turn_bridge: bool = False
    # A sane absolute minimum: pure smear (variance of Laplacian ~1 at
    # 120 deg/s on walk 9) is never taken, bridge or not.
    bridge_min_sharpness: float = 3.0
    # Never bridge sooner than this many frames after a keyframe: in a
    # sustained blur the tracker against a blurred reference reads low
    # survival on every frame, and this is what stops a keyframe per frame.
    bridge_min_frames: int = 2
    bridge_overlap_ratio: float = 0.5
    bridge_displacement_frac: float = 0.15
    # The time bound, in frames (0.5 s at 12 fps), for when motion cannot
    # be measured or has not yet crossed the bounds above.
    bridge_max_frames: int = 6


@dataclass(frozen=True)
class KeyframeDecision:
    outcome: str
    reason: str

    @property
    def accepted(self) -> bool:
        return self.outcome == ACCEPT

    @property
    def lost(self) -> bool:
        return self.outcome == TRACKING_LOST


class KeyframeSelector:
    """Stateful selector. Deliberately separate from the store.

    `note_accepted` is called by the caller only after a keyframe has
    actually been persisted. Folding it into `evaluate` would let a failed
    write leave the selector believing it succeeded, so the next frames
    would be measured against a keyframe that does not exist.
    """

    def __init__(self, policy: KeyframePolicy | None = None) -> None:
        self._policy = policy or KeyframePolicy()
        self._recent_sharpness: list[float] = []
        self._frames_since_keyframe = 0
        self._has_keyframe = False
        # Consecutive frames currently below the loss floor. Reset by any
        # frame that tracks, and by a declared loss, so a fresh segment
        # starts with a whole grace window rather than a spent one.
        self._consecutive_loss = 0
        # Consecutive frames the blur gate has rejected. Read only by the
        # turn bridge; counted regardless, which changes no decision.
        self._blur_run = 0

    @property
    def policy(self) -> KeyframePolicy:
        return self._policy

    @property
    def frames_since_keyframe(self) -> int:
        return self._frames_since_keyframe

    def note_frame(self, quality: FrameQuality) -> None:
        self._recent_sharpness.append(quality.sharpness)
        if len(self._recent_sharpness) > self._policy.sharpness_window:
            self._recent_sharpness.pop(0)
        self._frames_since_keyframe += 1

    def note_accepted(self) -> None:
        self._frames_since_keyframe = 0
        self._has_keyframe = True

    def note_lost(self) -> None:
        self._has_keyframe = False
        self._frames_since_keyframe = 0
        self._consecutive_loss = 0

    @property
    def is_stalled(self) -> bool:
        return (
            self._has_keyframe
            and self._frames_since_keyframe > self._policy.max_frame_gap
        )

    def _is_sharp_enough(self, quality: FrameQuality) -> bool:
        policy = self._policy
        if quality.sharpness < policy.min_sharpness:
            return False
        if len(self._recent_sharpness) >= 5:
            reference = sorted(self._recent_sharpness)[len(self._recent_sharpness) // 2]
            if reference > 0 and quality.sharpness / reference < policy.min_sharpness_ratio:
                return False
        return True

    def _turn_bridge_due(
        self, quality: FrameQuality, motion: MotionSummary | None
    ) -> bool:
        """Should this blur-rejected frame be taken as a turn bridge?

        Uses only what evaluate() already has: no image work. See the
        `turn_bridge` block in KeyframePolicy for the measurement behind it.
        """
        policy = self._policy
        if quality.sharpness < policy.bridge_min_sharpness:
            return False
        if motion is None or not self._has_keyframe:
            # Nothing to measure against (session start, or after a loss):
            # seed a segment only after a sustained blurred run.
            return self._blur_run >= policy.bridge_max_frames
        if self._frames_since_keyframe < policy.bridge_min_frames:
            return False
        displacement_frac = (
            (motion.median_displacement_px or 0.0) / quality.diagonal_px
        )
        if (
            motion.tracked_count > 0
            and motion.overlap_ratio >= policy.min_overlap_ratio
            and displacement_frac < policy.min_displacement_frac
        ):
            # The tracker says the camera is still -- on however few tracks
            # a low-texture view has: a blurred frame of a scene already
            # keyframed bridges nothing. This is what keeps a static
            # low-texture view from becoming a keyframe stream.
            return False
        if motion.overlap_ratio < policy.bridge_overlap_ratio:
            return True
        if displacement_frac >= policy.bridge_displacement_frac:
            return True
        return self._blur_run >= policy.bridge_max_frames

    def evaluate(
        self, quality: FrameQuality, motion: MotionSummary | None
    ) -> KeyframeDecision:
        """Decide what to do with one frame: accept, skip, reject or tracking lost.

        Inputs are the frame's `FrameQuality` and its `MotionSummary` against the
        last keyframe (None at session start or after a loss). Checks run in order:
        a blurred frame is rejected (unless the turn bridge takes it); the first
        frame of a segment is accepted as the session seed; a feature survival
        ratio below the loss threshold for `loss_grace_frames` frames in a row
        returns `tracking_lost`, a shorter dip is rejected as held; a weak
        survival is rejected as degraded; no motion evidence or a too-recent last
        keyframe is skipped; overlap falling below the floor is accepted; too
        little displacement is then skipped; and displacement reaching the
        target is accepted (anything in between is skipped). Returns a
        `KeyframeDecision` (decision, reason) and keeps the blur-run and
        loss-run counters for the next call.

        Blur is checked first, even before the seed, because anchoring a session
        on a smeared frame poisons every later measurement. Loss needs sustained
        evidence because a break is permanent.
        """
        policy = self._policy

        # Blur is checked before anything else, including the session
        # seed: anchoring a whole session on a smeared frame poisons every
        # measurement taken against it afterwards.
        if not self._is_sharp_enough(quality):
            self._blur_run += 1
            if policy.turn_bridge and self._turn_bridge_due(quality, motion):
                self._blur_run = 0
                self._consecutive_loss = 0
                return KeyframeDecision(ACCEPT, REASON_TURN_BRIDGE)
            return KeyframeDecision(REJECT, REASON_BLURRED)
        self._blur_run = 0

        if motion is None or not self._has_keyframe:
            return KeyframeDecision(ACCEPT, REASON_SESSION_SEED)

        if motion.survival_ratio < policy.loss_survival_ratio:
            # A break is permanent -- poses either side share no frame --
            # so it takes sustained evidence, not one bad hop. 47 of the
            # 50 losses on the real walk recovered on the very next frame.
            self._consecutive_loss += 1
            if self._consecutive_loss >= policy.loss_grace_frames:
                self._consecutive_loss = 0
                return KeyframeDecision(TRACKING_LOST, REASON_TRACKING_LOST)
            return KeyframeDecision(REJECT, REASON_TRACKING_HELD)

        # Anything that tracked at all spends the grace window's credit.
        # Without this, two unrelated bad hops minutes apart would add up
        # to a break that neither of them earned.
        self._consecutive_loss = 0

        if motion.survival_ratio < policy.min_survival_ratio:
            return KeyframeDecision(REJECT, REASON_TRACKING_DEGRADED)

        if not motion.has_motion_evidence:
            return KeyframeDecision(SKIP, REASON_NO_MOTION_EVIDENCE)

        # Rate limiting comes after the reject gates so that a burst of
        # blurred frames is still counted as blurred rather than hidden
        # behind "rate limited".
        if self._frames_since_keyframe < policy.min_frame_gap:
            return KeyframeDecision(SKIP, REASON_RATE_LIMITED)

        # Correspondence is about to be lost. Take the keyframe now even
        # at low parallax -- a usable weak link beats a broken chain.
        # With the floor at 0.75 this is the DOMINANT promotion path on
        # real footage (198 of 260 accepts on the 1395-frame walk measured
        # in KeyframePolicy), not a rare rescue.
        if motion.overlap_ratio < policy.min_overlap_ratio:
            return KeyframeDecision(ACCEPT, REASON_OVERLAP_FLOOR)

        diagonal = quality.diagonal_px
        displacement_frac = (motion.median_displacement_px or 0.0) / diagonal

        if displacement_frac < policy.min_displacement_frac:
            return KeyframeDecision(SKIP, REASON_INSUFFICIENT_MOTION)

        if displacement_frac >= policy.target_displacement_frac:
            return KeyframeDecision(ACCEPT, REASON_PARALLAX)

        return KeyframeDecision(SKIP, REASON_INSUFFICIENT_MOTION)
