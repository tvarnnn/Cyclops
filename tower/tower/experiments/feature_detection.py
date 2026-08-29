"""How much trackable texture does this frame actually contain.

A count alone is misleading. A thousand keypoints piled into one corner
of the image is a worse frame for geometry than three hundred spread
evenly, because a solver needs constraints from across the view, not a
dense clump in one place. So the headline is the count and the number
that qualifies it is `spatial_coverage` -- the fraction of an 8x8 grid
that contains at least one keypoint.

This is the measurement that says whether a real indoor scene gives us
anything to work with. Today the answer is synthetic-only.

ORB is used rather than SIFT because it is what the platform already
depends on and what World Builder's geometry uses; measuring a detector
the rest of the system does not use would tell us about the wrong thing.
Deliberately implemented here with cv2 directly rather than by importing
World Builder's helper -- the Lab must not depend on a cartridge, and a
test enforces that.
"""

import cv2
import numpy as np

from tower.experiments import (
    ORB_MIN_DIMENSION,
    ExperimentPreview,
    ExperimentResult,
    ExperimentSettings,
    KeypointPreview,
    MetricKind,
    ScenePreview,
    decode_gray,
    scene_structure,
)
from tower.instrumentation import StageTimer

# Counts of keypoints are counts; the response and size statistics are
# already per-frame means and must be averaged again, not added.
# `requested_features` is ORB_FEATURES, the same number on every frame.
METRIC_KINDS: dict[str, MetricKind] = {
    "keypoint_count": MetricKind.COUNT,
    "descriptor_count": MetricKind.COUNT,
    "spatial_coverage": MetricKind.RATE,
    "mean_response": MetricKind.RATE,
    "mean_keypoint_size": MetricKind.RATE,
    "requested_features": MetricKind.CONSTANT,
}

ORB_FEATURES = 1000
COVERAGE_GRID = 8

# How many keypoints survive into the picture, per grid cell. Two per
# cell over an 8x8 grid is at most 128 markers, which is legible on a
# 320-pixel panel; a thousand is a grey rectangle.
#
# Bucketing rather than "the top 128 by response", and rather than the
# proper answer, which is adaptive non-maximal suppression. Top-N by
# response returns whatever the textured corner of the room contains and
# would draw a picture that contradicts the coverage number sitting next
# to it. ANMS is what you would use if this were the front end of a SLAM
# system; it is O(n log n) with real per-pair work, and every real front
# end that has met this problem at frame rate -- ORB-SLAM's grid
# extractor, VINS-Mono's per-cell cap -- reaches for the grid instead.
# The grid is also the one this experiment already has.
PREVIEW_PER_CELL = 2


def run(raw_bytes: bytes) -> ExperimentResult:
    """One frame, measured. See `FeatureDetection` for the registered form."""
    result, _preview = _measure(raw_bytes, preview=False)
    return result


def _measure(raw_bytes: bytes, *, preview: bool):
    timer = StageTimer()

    with timer.stage("decode"):
        image = decode_gray(raw_bytes, min_dimension=ORB_MIN_DIMENSION)

    with timer.stage("detect"):
        orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        keypoints = orb.detect(image, None)

    with timer.stage("describe"):
        keypoints, descriptors = orb.compute(image, keypoints)

    with timer.stage("summarize"):
        count = len(keypoints)
        height, width = image.shape[:2]
        cells = set()
        if count:
            responses = np.asarray([kp.response for kp in keypoints], dtype=np.float64)
            sizes = np.asarray([kp.size for kp in keypoints], dtype=np.float64)
            xs = np.asarray([kp.pt[0] for kp in keypoints])
            ys = np.asarray([kp.pt[1] for kp in keypoints])
            cells = {
                (
                    min(int(x * COVERAGE_GRID / width), COVERAGE_GRID - 1),
                    min(int(y * COVERAGE_GRID / height), COVERAGE_GRID - 1),
                )
                for x, y in zip(xs, ys)
            }
            coverage = len(cells) / (COVERAGE_GRID * COVERAGE_GRID)
            mean_response = float(responses.mean())
            mean_size = float(sizes.mean())
        else:
            coverage = 0.0
            mean_response = 0.0
            mean_size = 0.0

        described = 0 if descriptors is None else int(len(descriptors))

    payload = None
    if preview:
        # A named stage, so the cost of the picture is visible beside the
        # cost of the measurement rather than hidden inside it. It runs
        # only on frames the Lab asked for one, so the run's average
        # `processing_ms` moves by the throttled fraction of this, and
        # anybody comparing against a pre-preview baseline can subtract
        # exactly this line.
        with timer.stage("preview"):
            payload = _preview_payload(image, keypoints, cells, coverage)

    return ExperimentResult(
        result_value=float(count),
        result_label="keypoint_count",
        processing_ms=timer.total_ms,
        stage_ms=timer.snapshot(),
        metrics={
            "keypoint_count": float(count),
            "descriptor_count": float(described),
            "spatial_coverage": coverage,
            "mean_response": mean_response,
            "mean_keypoint_size": mean_size,
            "requested_features": float(ORB_FEATURES),
        },
    ), payload


def _preview_payload(image, keypoints, cells, coverage):
    """The thinned keypoints, in the line drawing's coordinates."""
    structure = scene_structure(image)
    height, width = image.shape[:2]
    scale = structure.shape[1] / float(width) if width else 1.0

    buckets: dict = {}
    for keypoint in keypoints:
        x, y = keypoint.pt
        cell = (
            min(int(x * COVERAGE_GRID / width), COVERAGE_GRID - 1),
            min(int(y * COVERAGE_GRID / height), COVERAGE_GRID - 1),
        )
        bucket = buckets.setdefault(cell, [])
        # Kept sorted by insertion rather than sorted at the end: at most
        # `PREVIEW_PER_CELL` entries ever live in a bucket, so this is a
        # constant-size insert instead of sorting a thousand keypoints.
        if len(bucket) < PREVIEW_PER_CELL:
            bucket.append(keypoint)
            bucket.sort(key=lambda kp: kp.response, reverse=True)
        elif keypoint.response > bucket[-1].response:
            bucket[-1] = keypoint
            bucket.sort(key=lambda kp: kp.response, reverse=True)

    drawn = [kp for bucket in buckets.values() for kp in bucket]
    xy = np.asarray(
        [(kp.pt[0] * scale, kp.pt[1] * scale) for kp in drawn], dtype=np.float32
    ).reshape(-1, 2)
    return KeypointPreview(
        scene=ScenePreview(structure=structure),
        xy=xy,
        detected=len(keypoints),
        coverage_cells=tuple(cells),
        coverage_grid=COVERAGE_GRID,
        coverage=float(coverage),
    )


class FeatureDetection:
    """`_measure`, plus somewhere to put the keypoints down.

    Stateless in every sense that matters -- its answer depends on this
    frame and nothing before it, so its metadata still says
    `stateful=False`. What it gained is a `self` to hold one thinned
    keypoint set on, which a bare function could not have.
    """

    name = "feature_detection"

    def __init__(self) -> None:
        self._preview = ExperimentPreview()

    def load(self, settings: ExperimentSettings | None = None) -> None:
        return None

    def run(self, raw_bytes: bytes) -> ExperimentResult:
        result, payload = _measure(raw_bytes, preview=self._preview.wanted)
        self._preview.offer(payload)
        return result

    def set_preview_capture(self, enabled: bool) -> None:
        self._preview.set_preview_capture(enabled)

    def take_preview(self):
        return self._preview.take_preview()

    def release(self) -> None:
        self._preview.set_preview_capture(False)
