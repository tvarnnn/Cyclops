"""The cheap per-frame path: decode, score quality, track, summarise motion.

Runs on every delivered frame, at roughly 5 ms per 360x640 frame against
a ~300 ms delivered interval, so nothing here trades clarity for speed.

Two disciplines govern this module:

**Nothing here reads a clock.** Frame spacing is variable, the sender
decimates frames, and there is no capture timestamp anywhere on the wire.
Every threshold is in pixels or ratios. A time-based rule would silently
mean something different at 3 fps than at 12 fps, and would confuse "the
user turned quickly" with "the network stalled".

**Nothing here needs intrinsics.** Everything measured is image-space, so
the same measurements are available whether or not the camera has ever
been calibrated -- which is what lets the uncalibrated backend still
report real evidence instead of nothing.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Feature seeding for tracking. Shi-Tomasi rather than ORB here: this is
# a tracking seed, not a matching problem, and goodFeaturesToTrack gives
# better-conditioned corners for Lucas-Kanade than ORB's FAST keypoints.
MAX_TRACK_POINTS = 600
TRACK_QUALITY_LEVEL = 0.01
TRACK_MIN_DISTANCE = 7
LK_WINDOW = (21, 21)
# Level 4, not 3. Lucas-Kanade's capture range is roughly
# winSize/2 * 2**maxLevel: ~80 px at level 3, ~160 px at level 4.
# Displacement p99 on a real walk is 85 px, just past level 3.
#
# Displacement is that large because the reference does NOT advance every
# frame. It advances only on an accepted keyframe, so a run of blurred
# frames freezes it while the camera keeps moving: median 7, mean 12.4,
# max 89 frames stale at the moment loss was declared. The tracker is then
# asked to cross that entire gap in one call.
#
# Measured on that walk: 51 -> 40 segments and one added keyframe, at
# 4.85 ms per frame either way (the extra pyramid level is free at this
# resolution).
LK_MAX_LEVEL = 4

# Forward-backward consistency threshold in pixels. A track that does not
# land back where it started under the reverse flow is not a track, it is
# a coincidence.
#
# 3.0, not 1.0. Round-trip error grows with the distance travelled, and a
# stale reference (see LK_MAX_LEVEL above) makes long hops routine, so a
# 1.0 px budget calibrated for adjacent frames rejects real tracks and turns
# recoverable drift into declared loss: 47 of 50 losses on a real walk still
# had frame-to-frame survival above the 0.05 floor, and only 3 were genuine.
# 3.0 is still an eighth of the 21 px search window, so it discriminates.
# Together with the pyramid change: 51 -> 33 segments on that walk, and
# 171 -> 130 across eight captures.
#
# It is NOT free. Against a known warp, 1.0 -> 3.0 at a FIXED pyramid level
# roughly doubles the gross-outlier rate: 0.97% -> 1.74% of surviving tracks
# land beyond 20 px of truth. The pyramid change cuts that rate about 6x
# (6.27% -> 0.97%), so the pair is a net ~3.6x purity gain, but this is the
# weaker and costlier half of it, buying only 40 -> 33 segments.
#
# Two things bound the damage. LK tracks never reach the solve:
# `backends/classical.py` re-detects ORB and matches independently, so this
# tolerance decides which frames are PROMOTED, not which correspondences are
# triangulated. And it does not splice: on unrelated frames the combined
# change discriminates BETTER than the baseline, not worse.
#
# Not evidence for this value: intra-segment solvable pairs rose 46% -> 53%
# and triangulation angle 0.43 -> 0.63 deg, but that is confounded. Fewer
# segments means longer segments and a wider-baseline pair population
# (median keyframe gap 6 -> 9), and triangulation angle rises with baseline
# by construction; zero change in track quality would produce the same.
FORWARD_BACKWARD_MAX_PX = 3.0

# Below this many surviving tracks nothing downstream is meaningful.
MIN_TRACKS_FOR_MOTION = 12

HOMOGRAPHY_RANSAC_PX = 3.0


@dataclass(frozen=True)
class FrameQuality:
    """Per-frame measurements that need no reference frame."""

    width: int
    height: int
    sharpness: float

    @property
    def diagonal_px(self) -> float:
        return float(np.hypot(self.width, self.height))


@dataclass(frozen=True)
class MotionSummary:
    """How the camera moved relative to the current reference keyframe.

    `homography_residual_px` is the load-bearing field and deserves its
    own explanation. It is the median distance between where a homography
    fitted to the tracks predicts each point should land and where it
    actually landed.

    A homography exactly explains any motion that is pure rotation about
    the camera centre, whatever the scene. So residual near zero means
    "this motion carries no baseline" -- which is the one thing a keyframe
    policy most needs to know and cannot get from displacement.

    Measured: 0.107 and 0.120 px for genuinely zero-baseline pairs, versus
    0.305 to 0.765 px once the camera translated. Raw displacement was
    12.4 px under that SAME pure rotation -- larger than the 4.3 px of a
    real 5 cm sideways step -- so displacement alone would promote exactly
    the frames that carry no geometry.

    It is a rotation DETECTOR, not a parallax magnitude: it is not
    monotonic in baseline, because at wide baselines the surviving tracks
    concentrate on whatever plane dominates the view. Use it to reject,
    not to rank.
    """

    seeded_count: int
    tracked_count: int
    survival_ratio: float
    overlap_ratio: float
    median_displacement_px: float | None
    homography_residual_px: float | None

    @property
    def has_motion_evidence(self) -> bool:
        return (
            self.tracked_count >= MIN_TRACKS_FOR_MOTION
            and self.median_displacement_px is not None
        )


def decode_gray(raw_bytes: bytes) -> np.ndarray:
    """Decode a JPEG straight to grayscale.

    Raises ValueError on an undecodable payload so the caller can drop one
    frame rather than take the session down -- a malformed frame is a
    frame-scoped failure, matching how the existing Tower frame path
    already distinguishes the two.
    """
    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("frame did not decode as an image")
    return image


def measure_sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: high for crisp, low for blurred.

    Measured on identical content at increasing blur: 662.5 sharp, 38.0
    at a 5-px kernel, 10.1 at 9 px, 3.5 at 15 px. A 17x separation at the
    first step, for well under a millisecond, which is why blur is
    rejected here rather than discovered later by an expensive stage.

    The absolute value is scene-dependent, so callers compare against a
    rolling median of recent frames as well as an absolute floor.

    WHY CV_16S AND meanStdDev RATHER THAN CV_64F AND .var()

    Cost: this runs on EVERY delivered frame. A 640x360 float64 Laplacian
    is a 1.8 MB allocation per frame, and reducing it in numpy costs more
    than computing it; the CV_64F form was 7.6% of a full replay. On 120
    real Ray-Ban frames this form measured 1.429 -> 0.193 ms per frame (7.40x),
    ~2.3 s of a 32 s replay of the canonical capture.

    The intermediate is EXACT, not merely close. The input is 8-bit and
    `ksize` defaults to 1, so the kernel is [[0,1,0],[1,-4,1],[0,1,0]] and
    the output is bounded by +/-4*255 = +/-1020 against int16's +/-32767 --
    saturation is unreachable. Verified as well as argued:
    `np.array_equal(Laplacian(g, CV_64F), Laplacian(g, CV_16S))` is True
    on real frames, whose observed range was -497..324.

    The variance then differs only in the last bits of a float64 -- max
    relative error 6.22e-16 across 9,372 corpus frames -- because
    `meanStdDev` accumulates in float64 exactly as `.var()` does.

    That cannot flip a threshold comparison (absolute floor or rolling
    ratio), and the reason is stronger than the empirical check: the CV_16S
    Laplacian is INTEGER, so the variance of N such values lies on a lattice
    of spacing 1/N**2 -- 1.884e-11 at 360x640, about 5,300x coarser than one
    ULP at the 25.0 floor. Empirically, across the same 9,372 frames: zero
    flips, closest approach 24.96532, minimum safety factor 4.9e12 against
    the floor.

    THAT GUARANTEE IS RESOLUTION-DEPENDENT. The spacing is 1/N**2, so it
    shrinks as the frame grows: comfortable at 360x640, weaker at 1080p,
    gone above roughly 30 megapixels. DAT's adaptive ladder changes
    resolution mid-stream, so re-check this if a much larger frame size is
    adopted -- the empirical margin would survive, the proof would not.
    """
    if gray.dtype != np.uint8 or gray.ndim != 2:
        # NOT a defensive nicety. Violating this fails THREE ways, and the
        # worst one is silent. Measured against the CV_64F/.var() form on
        # the same array:
        #
        #   3-channel colour  109635.12 -> 110448.25   7.4e-3 relative
        #   int16            6.89e9    -> 9.06e8       8.7e-1 relative
        #   uint16 / float64                           raises cv2.error
        #
        # Colour is the quiet one: `meanStdDev` returns a PER-CHANNEL
        # deviation and `[0, 0]` takes channel zero, where `.var()` pooled
        # all three -- a plausible number, twelve orders of magnitude outside
        # the 6.2e-16 agreement above. `decode_gray` uses IMREAD_GRAYSCALE so
        # the product path cannot get here, and a test pins that; this guards
        # a public function annotated only `np.ndarray`.
        raise ValueError(
            "measure_sharpness requires 8-bit single-channel input; got "
            f"dtype={gray.dtype}, ndim={gray.ndim}. The CV_16S intermediate "
            "is only exact for uint8, and colour input would silently "
            "return a channel-0 answer."
        )
    _, deviation = cv2.meanStdDev(cv2.Laplacian(gray, cv2.CV_16S))
    return float(deviation[0, 0] ** 2)


def analyse_frame(gray: np.ndarray) -> FrameQuality:
    height, width = gray.shape[:2]
    return FrameQuality(
        width=width, height=height, sharpness=measure_sharpness(gray)
    )


def seed_tracks(gray: np.ndarray) -> np.ndarray | None:
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_TRACK_POINTS,
        qualityLevel=TRACK_QUALITY_LEVEL,
        minDistance=TRACK_MIN_DISTANCE,
    )


def track_forward_backward(
    reference_gray: np.ndarray, current_gray: np.ndarray, seeds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Track seeds and keep only forward-backward consistent survivors."""
    forward, status_f, _ = cv2.calcOpticalFlowPyrLK(
        reference_gray, current_gray, seeds, None,
        winSize=LK_WINDOW, maxLevel=LK_MAX_LEVEL,
    )
    if forward is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    backward, status_b, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, reference_gray, forward, None,
        winSize=LK_WINDOW, maxLevel=LK_MAX_LEVEL,
    )
    if backward is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    round_trip = np.linalg.norm(seeds - backward, axis=2).ravel()
    keep = (
        (status_f.ravel() == 1)
        & (status_b.ravel() == 1)
        & (round_trip < FORWARD_BACKWARD_MAX_PX)
    )
    return seeds[keep].reshape(-1, 2), forward[keep].reshape(-1, 2)


def summarise_motion(
    reference_gray: np.ndarray,
    current_gray: np.ndarray,
    seeds: np.ndarray | None,
) -> MotionSummary:
    height, width = current_gray.shape[:2]

    if seeds is None or len(seeds) == 0:
        return MotionSummary(0, 0, 0.0, 0.0, None, None)

    seeded_count = int(len(seeds))
    origin, moved = track_forward_backward(reference_gray, current_gray, seeds)
    tracked_count = int(len(moved))
    survival_ratio = tracked_count / seeded_count

    if tracked_count == 0:
        return MotionSummary(seeded_count, 0, 0.0, 0.0, None, None)

    inside = (
        (moved[:, 0] >= 0)
        & (moved[:, 0] < width)
        & (moved[:, 1] >= 0)
        & (moved[:, 1] < height)
    )
    overlap_ratio = float(inside.sum()) / seeded_count
    displacement = float(np.median(np.linalg.norm(origin - moved, axis=1)))

    residual = None
    if tracked_count >= MIN_TRACKS_FOR_MOTION:
        homography, _ = cv2.findHomography(
            origin, moved, cv2.RANSAC, HOMOGRAPHY_RANSAC_PX
        )
        if homography is not None:
            predicted = cv2.perspectiveTransform(
                origin.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            residual = float(
                np.median(np.linalg.norm(predicted - moved, axis=1))
            )

    return MotionSummary(
        seeded_count=seeded_count,
        tracked_count=tracked_count,
        survival_ratio=survival_ratio,
        overlap_ratio=overlap_ratio,
        median_displacement_px=displacement,
        homography_residual_px=residual,
    )


class FrameTracker:
    """Holds the reference frame that motion is measured against.

    The reference advances only when a keyframe is accepted, so motion is
    always measured against the last thing actually persisted rather than
    against the previous frame. That is what makes the measurements
    directly comparable to the keyframe decision they inform.
    """

    def __init__(self) -> None:
        self._reference_gray: np.ndarray | None = None
        self._seeds: np.ndarray | None = None

    @property
    def has_reference(self) -> bool:
        return self._reference_gray is not None

    def set_reference(self, gray: np.ndarray) -> None:
        self._reference_gray = gray.copy()
        self._seeds = seed_tracks(self._reference_gray)

    def reset(self) -> None:
        """Forget the reference. Used on tracking loss, where continuing to
        measure against a frame we can no longer see would be meaningless."""
        self._reference_gray = None
        self._seeds = None

    def measure(self, gray: np.ndarray) -> MotionSummary | None:
        if self._reference_gray is None:
            return None
        return summarise_motion(self._reference_gray, gray, self._seeds)
