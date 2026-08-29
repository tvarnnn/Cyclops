"""How much is the scene moving, and how coherently.

Sparse Lucas-Kanade rather than dense Farneback. Measured on this host at
640x360: 2.9 ms against 25.4 ms for the same headline number. The dense
field is a better picture and an 8.8x worse deal, and the benchmark keeps
the comparison so the choice stays evidence rather than habit.

Stateful, because flow is a statement about a PAIR of frames. The first
frame of a session therefore has no answer, and this reports that
honestly -- zero tracked, magnitude zero, `has_reference` 0 -- rather
than inventing a still scene.

Forward-backward error is measured, not assumed: a track is kept only if
following it back lands near where it started. Without that check, flow
in a textureless region returns confident nonsense.

A STALE reference is treated as no reference. The module is
process-scoped, so without this the first frame of a new wearer session
is silently diffed against the last frame of the previous one -- possibly
minutes old, possibly a different room -- and reported with
`has_reference: 1.0` as though nothing were wrong. Holding a frame of
wearer imagery indefinitely is also the wrong posture for a module
declaring `retains_raw_imagery=False`.

The residual case this does NOT cover: a new session starting inside the
staleness window still inherits the previous one's frame. Closing that
needs a session-boundary hook on the module contract, which is the
blocked V1.0/V1.1 work. The gap is named rather than papered over.
"""

import time

import cv2
import numpy as np

from tower.experiments import (
    ExperimentPreview,
    ExperimentResult,
    ExperimentSettings,
    FlowPreview,
    MetricKind,
    ScenePreview,
    decode_gray,
    scene_structure,
)
from tower.instrumentation import StageTimer

# `has_reference`, `resolution_changed` and `reference_stale` are 0/1
# flags, and a flag's SUM is the number of frames it fired on -- which is
# exactly the question worth asking of a corpus, so they are counts.
# `tracked_fraction` is the metric that was being summed to ~768 over the
# real corpus for a quantity that cannot exceed 1.
# `dominant_direction_deg` is circular and has no mean; see
# MetricKind.UNAGGREGATED.
METRIC_KINDS: dict[str, MetricKind] = {
    "median_flow_px": MetricKind.RATE,
    "mean_flow_px": MetricKind.RATE,
    "max_flow_px": MetricKind.RATE,
    "tracked_fraction": MetricKind.RATE,
    "tracked_count": MetricKind.COUNT,
    "seeded_count": MetricKind.COUNT,
    "median_forward_backward_px": MetricKind.RATE,
    "rejected_by_forward_backward": MetricKind.COUNT,
    "direction_coherence": MetricKind.RATE,
    "dominant_direction_deg": MetricKind.UNAGGREGATED,
    "has_reference": MetricKind.COUNT,
    "resolution_changed": MetricKind.COUNT,
    "reference_stale": MetricKind.COUNT,
    "seconds_since_reference": MetricKind.RATE,
}

MAX_CORNERS = 300
CORNER_QUALITY = 0.01
MIN_CORNER_DISTANCE = 8
# A track that does not return to within this many pixels of its origin
# after being followed forward and then back is not a track.
MAX_FORWARD_BACKWARD_PX = 1.0
# Beyond this gap the retained frame is not a reference to anything. At
# the 3.3 fps the glasses deliver the interval is ~300 ms and at 12 fps
# ~83 ms, so two seconds is roughly 7x the slowest expected spacing --
# loose enough never to fire during normal streaming, tight enough that a
# reconnect, a walk out of range, or a new session does not silently
# become a measurement.
MAX_REFERENCE_AGE_S = 2.0
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class OpticalFlowExperiment:
    name = "optical_flow"

    def __init__(self, clock=time.monotonic) -> None:
        self._previous = None
        self._previous_at = None
        self._preview = ExperimentPreview()
        # Monotonic, not wall clock: this measures the gap between two
        # calls, and an NTP correction must not be able to make a live
        # reference look stale or a stale one look fresh.
        self._clock = clock

    def load(self, settings: ExperimentSettings | None = None) -> None:
        return None

    def set_preview_capture(self, enabled: bool) -> None:
        self._preview.set_preview_capture(enabled)

    def take_preview(self):
        return self._preview.take_preview()

    def release(self) -> None:
        # Holds one grayscale frame, not a model. Dropping it is the whole
        # of teardown -- but it must still happen, or a stopped experiment
        # keeps a frame of wearer imagery alive in memory.
        self._previous = None
        self._previous_at = None
        self._preview.set_preview_capture(False)

    def run(self, raw_bytes: bytes) -> ExperimentResult:
        timer = StageTimer()

        with timer.stage("decode"):
            gray = decode_gray(raw_bytes)

        # Offered here, empty, and offered AGAIN at the bottom with the
            # tracks on it. This method has four exits -- no reference, a
            # stale reference, no seeds, nothing tracked -- and each of
            # them is a real thing to look at: a picture of the room with
            # "no reference frame" on it says something a blank panel does
            # not. Offering twice costs one overwritten slot and covers
            # every exit; adding an offer to each of the four would be
            # four places for the next person to forget one.
        scene = None
        if self._preview.wanted:
            with timer.stage("preview"):
                # The structure is the whole measured cost. Assembling the
                # overlay below is numpy views over arrays that already
                # exist, in microseconds, and `StageTimer` overwrites a
                # stage by name rather than accumulating -- so timing it
                # separately would replace this number with a smaller and
                # far less interesting one.
                scene = ScenePreview(structure=scene_structure(gray))
            self._preview.offer(
                FlowPreview(
                    scene=scene,
                    origins=np.empty((0, 2), np.float32),
                    displacements=np.empty((0, 2), np.float32),
                    rejected=np.empty((0, 2), np.float32),
                    tracked_count=0,
                    seeded_count=0,
                    median_flow_px=0.0,
                )
            )

        now = self._clock()
        previous = self._previous
        age = None if self._previous_at is None else now - self._previous_at
        self._previous = gray
        self._previous_at = now

        stale = age is not None and age > MAX_REFERENCE_AGE_S
        resolution_changed = previous is not None and previous.shape != gray.shape
        if previous is None or stale or resolution_changed:
            # A resolution change lands here too. DAT's adaptive ladder can
            # change resolution mid-stream, and comparing frames of
            # different sizes would produce a large, meaningless flow.
            return ExperimentResult(
                result_value=0.0,
                result_label="median_flow_px",
                processing_ms=timer.total_ms,
                stage_ms=timer.snapshot(),
                metrics={
                    "median_flow_px": 0.0,
                    "tracked_fraction": 0.0,
                    "seeded_count": 0.0,
                    "tracked_count": 0.0,
                    "has_reference": 0.0,
                    "resolution_changed": 1.0 if resolution_changed else 0.0,
                    "reference_stale": 1.0 if stale else 0.0,
                    "seconds_since_reference": -1.0 if age is None else age,
                },
            )

        with timer.stage("seed"):
            seeds = cv2.goodFeaturesToTrack(
                previous,
                maxCorners=MAX_CORNERS,
                qualityLevel=CORNER_QUALITY,
                minDistance=MIN_CORNER_DISTANCE,
            )

        if seeds is None or len(seeds) == 0:
            return ExperimentResult(
                result_value=0.0,
                result_label="median_flow_px",
                processing_ms=timer.total_ms,
                stage_ms=timer.snapshot(),
                metrics={
                    "median_flow_px": 0.0,
                    "tracked_fraction": 0.0,
                    "seeded_count": 0.0,
                    "tracked_count": 0.0,
                    "has_reference": 1.0,
                    "resolution_changed": 0.0,
                    "reference_stale": 0.0,
                    "seconds_since_reference": age,
                },
            )

        with timer.stage("track"):
            forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
                previous, gray, seeds, None, **LK_PARAMS
            )
            backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
                gray, previous, forward, None, **LK_PARAMS
            )

        with timer.stage("summarize"):
            seeded = seeds.reshape(-1, 2)
            forward_points = forward.reshape(-1, 2)
            backward_points = backward.reshape(-1, 2)
            ok = (
                status_forward.reshape(-1).astype(bool)
                & status_backward.reshape(-1).astype(bool)
            )
            fb_error = np.linalg.norm(backward_points - seeded, axis=1)
            kept = ok & (fb_error <= MAX_FORWARD_BACKWARD_PX)
            # Reported over every track LK claimed to have followed, not
            # only those that then passed the <=1.0 px filter. Measured
            # over the survivors it could never exceed the threshold by
            # construction, so it would read "excellent" on every frame
            # regardless of how badly the tracking actually went.
            attempted = fb_error[ok.astype(bool)]

            tracked_count = int(kept.sum())
            seeded_count = int(len(seeded))
            if tracked_count == 0:
                metrics = {
                    "median_flow_px": 0.0,
                    "tracked_fraction": 0.0,
                    "seeded_count": float(seeded_count),
                    "tracked_count": 0.0,
                    "has_reference": 1.0,
                    "resolution_changed": 0.0,
                    "reference_stale": 0.0,
                    "seconds_since_reference": age,
                }
                return ExperimentResult(
                    result_value=0.0,
                    result_label="median_flow_px",
                    processing_ms=timer.total_ms,
                    stage_ms=timer.snapshot(),
                    metrics=metrics,
                )

            displacement = forward_points[kept] - seeded[kept]
            magnitudes = np.linalg.norm(displacement, axis=1)
            median_magnitude = float(np.median(magnitudes))

            # Direction coherence via the mean resultant length of the
            # displacement angles: 1.0 means every track agrees, 0.0 means
            # they cancel out. A mean of raw angles would be wrong -- the
            # average of 179 and -179 degrees is not 0.
            angles = np.arctan2(displacement[:, 1], displacement[:, 0])
            resultant = float(
                np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles)))
            )
            dominant_direction_deg = float(
                np.degrees(
                    np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
                )
            )

        if scene is not None:
            # In `scene` pixels, not frame pixels. The scale is applied
            # once, here, so no renderer is ever handed a second one --
            # `IOS-to-Tower.md` 2.5 refuses annotation geometry precisely
            # because "a wrong convention renders confidently in the wrong
            # place", and one conversion in one place is how that stays
            # true of a picture the Tower draws itself.
            flow_scale = scene.size[0] / float(gray.shape[1] or 1)
            self._preview.offer(
                FlowPreview(
                    scene=scene,
                    origins=(seeded[kept] * flow_scale).astype(np.float32),
                    displacements=(displacement * flow_scale).astype(np.float32),
                    rejected=(seeded[~kept] * flow_scale).astype(np.float32),
                    tracked_count=tracked_count,
                    seeded_count=seeded_count,
                    median_flow_px=median_magnitude,
                )
            )

        return ExperimentResult(
            result_value=median_magnitude,
            result_label="median_flow_px",
            processing_ms=timer.total_ms,
            stage_ms=timer.snapshot(),
            metrics={
                "median_flow_px": median_magnitude,
                "mean_flow_px": float(magnitudes.mean()),
                "max_flow_px": float(magnitudes.max()),
                "tracked_fraction": tracked_count / seeded_count,
                "tracked_count": float(tracked_count),
                "seeded_count": float(seeded_count),
                "median_forward_backward_px": (
                    float(np.median(attempted)) if attempted.size else 0.0
                ),
                "rejected_by_forward_backward": float(
                    int(ok.astype(bool).sum()) - tracked_count
                ),
                "direction_coherence": resultant,
                "dominant_direction_deg": dominant_direction_deg,
                "has_reference": 1.0,
                "resolution_changed": 0.0,
                "reference_stale": 0.0,
                "seconds_since_reference": age,
            },
        )
