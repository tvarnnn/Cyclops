"""Canny edge density, and the edge map it was counted from.

The number this reports has not changed and the thresholds have not
moved: `edge_density` is still `count_nonzero(Canny(blur(gray), 100,
200)) / size`, measured the same way it was when it was the only thing
this file produced.

What changed is that the array is no longer thrown away. `cv2.Canny`
already built it -- counting its non-zero pixels is the last thing that
happens to it -- and the whole of the live preview for this experiment is
the decision not to let it go out of scope. Nothing is recomputed, no
second Canny runs, no colour conversion is added, and the frame path
gains one attribute assignment whose cost does not appear in the timings.

The array is offered rather than kept: `ExperimentPreview` holds it only
after the Lab has said somebody is watching, so a Tower with previews off
retains exactly what it retained before, which is nothing.
"""

import cv2
import numpy as np

from tower.experiments import (
    ExperimentPreview,
    ExperimentResult,
    ExperimentSettings,
    MetricKind,
    decode_color,
)
from tower.instrumentation import StageTimer

# The headline `edge_density` says everything this experiment measures,
# so there is no metrics bag. A metric added below without a line here
# raises rather than being guessed at.
METRIC_KINDS: dict[str, MetricKind] = {}


def run(raw_bytes: bytes) -> ExperimentResult:
    """One frame, measured. Kept as a free function on purpose.

    `EdgeDetection` is the registered experiment now, but the measurement
    is still a pure function of the bytes, and a benchmark or a corpus
    harness that wants the number without a lifecycle should not have to
    construct an object and remember to release it.

    `edges` is discarded here, exactly as it always was. The class below
    is the one that offers it onwards, so that the difference between
    "measuring" and "being watched" stays visible in the code rather than
    living in a flag.
    """
    result, _edges = _measure(raw_bytes)
    return result


def _measure(raw_bytes: bytes):
    """The measurement, and the array it came from.

    Returns both because the caller decides what happens to the second
    one. The stage names are unchanged -- `decode`, `blur`, `canny`,
    `summarize` -- so a run of this experiment is still comparable
    against every timing already recorded for it.
    """
    timer = StageTimer()

    with timer.stage("decode"):
        image = decode_color(raw_bytes)

    with timer.stage("blur"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    with timer.stage("canny"):
        edges = cv2.Canny(blurred, 100, 200)

    with timer.stage("summarize"):
        edge_density = float(np.count_nonzero(edges)) / edges.size

    return (
        ExperimentResult(
            result_value=edge_density,
            result_label="edge_density",
            processing_ms=timer.total_ms,
            stage_ms=timer.snapshot(),
        ),
        edges,
    )


class EdgeDetection:
    """`_measure`, plus somewhere to put the edge map down.

    Holds no model, allocates nothing at load, and answers each frame
    from that frame alone -- which is why its metadata still declares
    `stateful=False`. The only thing it carries between frames is the
    preview slot, and the preview slot has no effect on any number this
    reports.
    """

    name = "edge_detection"

    def __init__(self) -> None:
        self._preview = ExperimentPreview()

    def load(self, settings: ExperimentSettings | None = None) -> None:
        return None

    def run(self, raw_bytes: bytes) -> ExperimentResult:
        result, edges = _measure(raw_bytes)
        # After the measurement and outside every timed stage. The
        # preview must not appear in `stage_ms`: those numbers are the
        # experiment's cost and are compared against runs recorded before
        # this file grew a preview at all.
        self._preview.offer(edges)
        return result

    def set_preview_capture(self, enabled: bool) -> None:
        self._preview.set_preview_capture(enabled)

    def take_preview(self):
        return self._preview.take_preview()

    def release(self) -> None:
        # Safe twice, and safe after a partial load, like every other
        # experiment's -- and the one thing it has to actually do is drop
        # the array, because `release()` runs on the FAILED transition and
        # a failed module is never released again.
        self._preview.set_preview_capture(False)
