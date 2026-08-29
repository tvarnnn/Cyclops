"""How usable is this frame, before anything expensive looks at it.

Every cartridge has some version of this question and World Builder
already answers a private one. The Lab's job is to make the signals
measurable side by side so a threshold gets chosen from a distribution
rather than from taste -- and to make the cost of asking visible, since a
gate that costs more than the work it skips is not a gate.

All six measurements are computed from one decode and one grayscale
conversion. Sharpness is the headline because it is the one that
separates most sharply: a Laplacian variance falls roughly 17x between a
sharp frame and a blurred one, which is a far bigger margin than any of
the others offer.
"""

import cv2
import numpy as np

from tower.experiments import (
    ExperimentPreview,
    ExperimentResult,
    ExperimentSettings,
    MetricKind,
    QualityPreview,
    ScenePreview,
    decode_color,
    scene_structure,
)
from tower.instrumentation import StageTimer

# What each number means when many frames are combined. Seven of these
# are per-frame measurements whose corpus answer is a mean; `width` and
# `height` are neither summable nor averageable -- a corpus of 9,199
# frames does not have a mean width, it has the sizes it was shot at.
METRIC_KINDS: dict[str, MetricKind] = {
    "sharpness_laplacian_var": MetricKind.RATE,
    "gradient_energy": MetricKind.RATE,
    "entropy_bits": MetricKind.RATE,
    "contrast_std": MetricKind.RATE,
    "edge_density": MetricKind.RATE,
    "overexposed_fraction": MetricKind.RATE,
    "underexposed_fraction": MetricKind.RATE,
    "width": MetricKind.CONSTANT,
    "height": MetricKind.CONSTANT,
}

# 8-bit imagery, so these are absolute levels rather than fractions.
# Chosen at the ends of the range: a pixel at 250+ has almost certainly
# clipped and a pixel at 5 or below carries no recoverable detail.
OVEREXPOSED_LEVEL = 250
UNDEREXPOSED_LEVEL = 5


def run(raw_bytes: bytes) -> ExperimentResult:
    """One frame, measured. See `FrameQuality` for the registered form."""
    result, _preview = _measure(raw_bytes, preview=False)
    return result


def _measure(raw_bytes: bytes, *, preview: bool):
    timer = StageTimer()

    with timer.stage("decode"):
        gray = cv2.cvtColor(decode_color(raw_bytes), cv2.COLOR_BGR2GRAY)

    with timer.stage("sharpness"):
        # CV_16S and meanStdDev rather than CV_64F and .var(). A 640x360
        # float64 Laplacian is a 1.8 MB allocation per frame, and reducing
        # it in numpy costs more than computing it.
        #
        # MEASURED on 1,500 real corpus frames, same-session A/B with
        # alternating arm order: 1.4885 ms -> 0.3173 ms per frame, 4.69x.
        # This experiment runs SYNCHRONOUSLY ON THE EVENT LOOP whenever it
        # is the selected one, so the saving is loop time every connection
        # shares.
        #
        # The intermediate is EXACT here, not merely close, for the same
        # reason it is in `world_builder/frontend.py: measure_sharpness`,
        # whose docstring carries the full argument: 8-bit input with
        # ksize=1 bounds the Laplacian at +/-1020 against int16's +/-32767,
        # so saturation is unreachable. Verified on this corpus rather
        # than assumed -- `np.array_equal(Laplacian(g, CV_64F),
        # Laplacian(g, CV_16S))` held on 400 frames, observed range
        # -538..392 -- and the variance then agreed to a max relative
        # difference of 5.053e-16 across all 1,500.
        #
        # DUPLICATED FROM THAT FUNCTION RATHER THAN IMPORTED. The Lab is a
        # sandbox that may be thrown away and must not import another
        # cartridge; `test_scene_understanding_does_not_import_another_
        # cartridge` states the rule for its neighbour and the reasoning
        # is the same one `cartridge_runtime._resolve_device` gives for
        # duplicating in spirit. No dtype guard is needed here because
        # `gray` is produced two lines above by `cvtColor(..., BGR2GRAY)`
        # and is uint8 single-channel by construction; the shared function
        # needs one only because it is public.
        _, deviation = cv2.meanStdDev(cv2.Laplacian(gray, cv2.CV_16S))
        sharpness = float(deviation[0, 0] ** 2)

    with timer.stage("gradient"):
        # Scharr rather than a 3x3 Sobel: same cost, better rotational
        # symmetry, so the number does not depend on which way an edge
        # happens to run.
        #
        # `cv2.magnitude(dx, dy)` was MEASURED as a replacement for the
        # numpy chain below and REFUSED. It is 1.72x on this stage
        # (1.4551 -> 0.8441 ms/frame over 1,500 real frames), not the 5.36x
        # an earlier estimate suggested, and unlike the sharpness change it
        # is NOT exact: max relative difference 2.203e-07, which is float32
        # epsilon rather than float64 agreement. `gradient_energy` is a
        # REPORTED measurement whose whole purpose is that thresholds get
        # chosen from its distribution, so 0.6 ms per frame on a sandbox
        # experiment does not buy a perturbation in its seventh
        # significant digit. Recorded because it is the intuitive next
        # step and the numbers argue against it.
        dx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        dy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        gradient_energy = float(np.mean(np.sqrt(dx * dx + dy * dy)))

    with timer.stage("exposure"):
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total = float(gray.size)
        overexposed = float(histogram[OVEREXPOSED_LEVEL:].sum()) / total
        underexposed = float(histogram[: UNDEREXPOSED_LEVEL + 1].sum()) / total
        mean_intensity = float(gray.mean())
        contrast = float(gray.std())

        # Shannon entropy of the intensity histogram, in bits. A frame of
        # flat wall and a frame of textured room can share a mean and a
        # standard deviation; they do not share an entropy.
        probabilities = histogram / total
        nonzero = probabilities[probabilities > 0]
        entropy = float(-(nonzero * np.log2(nonzero)).sum())

    with timer.stage("edges"):
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(edges)) / edges.size

    payload = None
    if preview:
        with timer.stage("preview"):
            # `scene_structure` rather than the `edges` array two lines
            # up, even though that array is free. Those thresholds are
            # tuned for the FULL decode resolution and this picture is
            # 320 px wide: downscaling a 100/200 edge map produces a
            # sparse dotted room, and the structure helper re-runs Canny
            # at the size it will actually be looked at. The saving would
            # have been half a millisecond and the cost would have been
            # the picture.
            payload = QualityPreview(
                scene=ScenePreview(structure=scene_structure(gray)),
                histogram=histogram,
                overexposed_level=OVEREXPOSED_LEVEL,
                underexposed_level=UNDEREXPOSED_LEVEL,
                overexposed_fraction=overexposed,
                underexposed_fraction=underexposed,
            )

    return ExperimentResult(
        result_value=sharpness,
        result_label="sharpness_laplacian_var",
        processing_ms=timer.total_ms,
        stage_ms=timer.snapshot(),
        mean_intensity=mean_intensity,
        metrics={
            "sharpness_laplacian_var": sharpness,
            "gradient_energy": gradient_energy,
            "entropy_bits": entropy,
            "contrast_std": contrast,
            "edge_density": edge_density,
            "overexposed_fraction": overexposed,
            "underexposed_fraction": underexposed,
            "width": float(gray.shape[1]),
            "height": float(gray.shape[0]),
        },
    ), payload


class FrameQuality:
    """`_measure`, plus somewhere to put the histogram down.

    Still `stateful=False`: nothing it reports depends on a previous
    frame, and the preview slot holds one frame's derived picture rather
    than any part of the measurement.
    """

    name = "frame_quality"

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
