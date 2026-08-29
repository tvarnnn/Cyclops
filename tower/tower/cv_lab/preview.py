"""One derived picture at a time, and no way for it to slow a frame down.

What this is for
----------------
The Lab was, until this module, a screen of numbers. `edge_density:
0.071` is a true statement about a frame and it is almost useless to the
person wearing the glasses, who wants to know whether the algorithm can
see the doorway. This turns the array the experiment already computed
into something a person can look at.

The one rule
------------
**Visualisation may never make the CV path slower.** Everything below is
shaped by that and by nothing else.

    frame  ->  experiment.run()          the authoritative work
           ->  LivePreview.capture()     two attribute assignments and
                                          a clock comparison
           ->  ...
           ->  GET /cv-lab/preview       resize, colourise, encode --
                                          on a worker thread, and only
                                          when somebody actually asked

`capture()` does no image work at all: it takes the array the experiment
computed -- by reference, no copy, no resize, no encode -- and puts it in
a slot. Everything expensive happens later, in `render()`, which
Starlette runs in its thread pool because the route that calls it is a
plain `def`. That is the same reason every disk-touching route in this
Tower is a `def`, and it is the whole answer to "can the viewer
backpressure the pipeline": **the event loop never runs the encoder.**

The frame path is not, however, entirely free, and the honest version is
worth stating rather than rounding to zero. Two experiments hand over an
array they had already built for their own metric and cost nothing:
`edge_detection`'s Canny map and `depth`'s inference output. The other
five have no such array and derive one -- a line drawing, a thinned
keypoint set, boxes pulled off a tensor -- inside `run()`, on the loop.
Measured, per captured frame: `frame_quality` 0.17 ms, `optical_flow`
0.27 ms, `feature_detection` 0.74 ms, `redaction_impact` 1.07 ms. It
happens only on frames the 20 Hz throttle allowed, it appears as a named
`preview` stage in `stage_ms`, and it is therefore visible and
subtractable rather than hidden inside a total. At the throttle ceiling
that is 3-21 ms of loop time per second of wall clock, worst case.

If nobody ever fetches, nothing is ever encoded. The work is not queued
for later -- it simply does not happen.

Bounded by shape
----------------
There is ONE slot and ONE cached encoding. `capture()` overwrites;
`render()` replaces. There is no list, no deque, no `maxlen` to get
wrong, and no file. A consumer that stops consuming costs nothing that
grows: it costs the same one array and the same one buffer it cost while
it was keeping up, and the frames it missed were dropped at the moment
they were replaced rather than accumulated against its return.

That matters more here than it sounds. `handoff.md` 9.3 says a
`stream_stop` MAY NEVER ARRIVE, so "for the length of a run" means "for
as long as the Tower is up". Anything per-frame would be unbounded.

Staleness is answered three ways, on purpose
--------------------------------------------
1. **Run identity.** Every capture carries the `run_id` it was produced
   under. A caller may name the run it is watching, and a preview from
   another run is refused rather than served -- the "stop A, start B, A's
   last frame arrives" case, closed at the Tower rather than trusted to
   the phone.
2. **An epoch.** Every stop, pause and release bumps a counter. A render
   that began before one and finished after it is discarded, because the
   bytes it holds are of a run that is no longer live.
3. **Age.** A capture older than `PREVIEW_MAX_AGE_S` is refused. A phone
   showing a four-second-old edge map while its wearer turns their head
   is showing a lie about where they are looking, and "the picture
   stopped" is a much better thing for a person to see than "the picture
   is wrong".

Failure is contained
--------------------
`render()` never raises. An encoder that fails produces a refusal with a
reason, the counters record it, and the experiment neither knows nor
cares -- it has already returned its result and its numbers are already
on the wire. A preview is a convenience, and no convenience gets to end
a run.

Threads
-------
`begin`, `suspend`, `end` and `capture` run on the event loop, from the
Lab. `render`, `descriptor` and `stats` run on worker threads. The
handover is a single attribute assignment of an immutable object, which
is atomic in CPython and needs no lock -- the same technique
`CVLab._last_frame_provenance` already uses, for the same reason.
`_stats_guard` covers only the counters the serving side writes and the
status document reads; the frame path never touches it, because a lock
on the measured path to make a diagnostic one frame fresher is paying in
the wrong currency. The one piece of genuinely shared MUTABLE state is
`_DepthNormaliser`, which carries its own lock and explains why.
"""

import functools
import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from tower.cv_lab.contracts import (
    ARTIFACT_KIND_LIVE_PREVIEW,
    PREVIEW_CONTRACT,
    PREVIEW_DISABLED,
    PREVIEW_FACE_FILTER_NONE,
    PREVIEW_JPEG_QUALITY,
    PREVIEW_MAX_AGE_S,
    PREVIEW_MAX_EDGE_PX,
    PREVIEW_MIN_INTERVAL_S,
    PREVIEW_NONE_YET,
    PREVIEW_NOT_VISUAL,
    PREVIEW_PATH,
    PREVIEW_PERSISTENCE_NONE,
    PREVIEW_PNG_COMPRESSION,
    PREVIEW_POLL_INTERVAL_S,
    PREVIEW_RENDER_FAILED,
    PREVIEW_RUN_CHANGED,
    PREVIEW_STALE,
    TREATMENT_RAW_EPHEMERAL,
)
from tower.cv_lab.run import Running
from tower.experiments import (
    PREVIEW_KIND_DETECTIONS,
    PREVIEW_KIND_EDGE_MAP,
    PREVIEW_KIND_FLOW_TRACKS,
    PREVIEW_KIND_FRAME_QUALITY,
    PREVIEW_KIND_KEYPOINTS,
    PREVIEW_KIND_REDACTION,
    PREVIEW_KIND_RELATIVE_DEPTH,
)

logger = logging.getLogger(__name__)

MEDIA_TYPE_PNG = "image/png"
MEDIA_TYPE_JPEG = "image/jpeg"


@dataclass(frozen=True)
class PreviewPolicy:
    """What this Tower will and will not draw.

    Frozen and passed in rather than read from the environment here, so
    a test can build one and a second Lab in the same process cannot
    change the first one's mind.
    """

    enabled: bool = True
    max_edge_px: int = PREVIEW_MAX_EDGE_PX
    min_interval_s: float = PREVIEW_MIN_INTERVAL_S
    max_age_s: float = PREVIEW_MAX_AGE_S
    poll_interval_s: float = PREVIEW_POLL_INTERVAL_S
    jpeg_quality: int = PREVIEW_JPEG_QUALITY
    png_compression: int = PREVIEW_PNG_COMPRESSION

    @classmethod
    def from_settings(cls, settings) -> "PreviewPolicy":
        """Built from `tower.config.Settings`, which owns the defaults.

        `getattr` with a default rather than attribute access, because
        several tests build a `Settings`-shaped stub and a preview policy
        is not a reason for any of them to grow three fields.
        """
        return cls(
            enabled=bool(getattr(settings, "cv_preview", True)),
            max_edge_px=int(
                getattr(settings, "cv_preview_max_edge_px", PREVIEW_MAX_EDGE_PX)
            ),
            min_interval_s=float(
                getattr(
                    settings, "cv_preview_min_interval_s", PREVIEW_MIN_INTERVAL_S
                )
            ),
        )


@dataclass(frozen=True)
class Capture:
    """One frame's derived array, and whose frame it was.

    Immutable, and handed over as a whole. That is what lets the slot be
    a plain attribute rather than a lock: a reader either sees the old
    object or the new one, never half of either.
    """

    kind: str
    array: object
    run_id: str | None
    result_seq: int
    captured_at: float
    epoch: int


@dataclass(frozen=True)
class RenderedPreview:
    """Bytes a client may draw, and everything it needs to place them."""

    image_bytes: bytes
    media_type: str
    width: int
    height: int
    kind: str
    run_id: str | None
    result_seq: int
    captured_at: float
    age_s: float
    render_ms: float
    etag: str
    # Repeated from the descriptor onto the bytes, so a client that
    # fetched the image without ever reading the status document still
    # learns this is untreated, live-view-only imagery it must not
    # persist. An image whose treatment travels separately is an image
    # whose treatment can be lost.
    treatment: str = TREATMENT_RAW_EPHEMERAL


@dataclass(frozen=True)
class PreviewNotModified:
    """The caller already has this exact frame."""

    etag: str
    run_id: str | None
    result_seq: int


@dataclass(frozen=True)
class PreviewRefusal:
    """No picture, and which of the closed reasons it is."""

    reason: str
    message: str
    current_run_id: str | None = None


class _DepthNormaliser:
    """Relative inverse depth to 0-255, without the frame-to-frame flicker.

    MiDaS's own `write_depth` normalises each frame between its own min
    and its own max. For a still that is right; for video it is the
    flicker mechanism, because one pixel of specular glare moving through
    the frame rescales the whole picture's brightness and a wall appears
    to pulse while the wearer stands still.

    Two cheap changes fix it and neither needs a second model:

    1. **Percentiles, not extremes.** The 2nd and 98th percentiles of the
       frame instead of its min and max, so a handful of outlier pixels
       cannot set the scale for the other fifty thousand.
    2. **Smoothed bounds.** Those percentiles are folded into an
       exponential moving average with `alpha = 0.2` -- about a
       five-frame time constant -- so the SCALE moves slowly while the
       picture moves at full rate. The smoothing is on the two bounds and
       never on the pixels: smoothing pixels would ghost, and a depth map
       that ghosts is worse than one that flickers.

    Anything outside the smoothed bounds is clipped rather than allowed
    to widen them, which is what makes a hand entering frame brighten
    without washing the room out.

    The scale is per-RUN and reset with it. Carrying it across a stop
    would be the previous experiment deciding what this one's near and
    far look like.
    """

    __slots__ = ("_low", "_high", "_guard")

    # Chosen rather than tuned by eye: 2/98 is the conventional robust
    # stretch, and 0.2 is about a five-frame time constant, which settles
    # a scene change inside half a second at the rates this Lab actually
    # runs at while damping a single bright frame almost entirely.
    ALPHA = 0.2
    LOW_PCT = 2.0
    HIGH_PCT = 98.0
    # The smallest span the normaliser will divide by. A frame of one
    # value -- a lens cap, a blank wall at the model's resolution -- has
    # no range at all, and dividing by its zero span produces `inf` and
    # then a `ValueError` out of `astype`.
    EPS = 1e-6

    def __init__(self) -> None:
        self._low: float | None = None
        self._high: float | None = None
        # THE one piece of mutable state this module shares between
        # threads, and it needs a lock where nothing else here does.
        #
        # Everything else crossing the loop/worker boundary is an
        # immutable object handed over by a single atomic assignment, so
        # a reader sees the old one or the new one and never half of
        # either. This is different: `to_uint8` READS `_low`/`_high`,
        # computes from them, and WRITES them back, and two `render()`
        # calls can be inside it at once -- a phone polling while
        # somebody runs `curl`, or one retried request -- because the
        # encoded-bytes cache that would normally make the second call
        # free is not populated until the first one finishes.
        #
        # Measured with the window widened: two threads on one fresh
        # capture both read the same bounds and one update was silently
        # lost. The damage is cosmetic -- a slightly wrong brightness on
        # one frame of a scale that re-converges within about five --
        # and it never reaches `mean_relative_depth` or decides which
        # bytes are served under which run. It is still a lost write,
        # and the argument that made a lock unnecessary everywhere else
        # simply does not apply here.
        self._guard = threading.Lock()

    def to_uint8(self, depth):
        if getattr(depth, "size", 0) == 0:
            # `np.percentile` on an empty array raises `IndexError: index -1
            # is out of bounds`, which is contained but tells a reader
            # nothing. The NaN path below was hardened for the same reason
            # and this is the same failure with a worse message.
            raise ValueError("depth frame is empty")
        low, high = np.percentile(depth, (self.LOW_PCT, self.HIGH_PCT))
        low = float(low)
        high = float(high)
        if not np.isfinite(low) or not np.isfinite(high):
            # A frame carrying a NaN poisons the percentiles, and every
            # arithmetic step after this one. Refusing here is what makes
            # `render()`'s failure legible instead of a `ValueError`
            # thrown by `astype` several lines later.
            raise ValueError("depth frame has no finite percentiles")
        if high - low < self.EPS:
            high = low + self.EPS
        with self._guard:
            # Read, fold, write and re-read the span as ONE critical
            # section. Splitting it is the race: two threads that both
            # read the old bounds both write a fold of the old bounds,
            # and one of the two updates is gone.
            if self._low is None or self._high is None:
                # The first frame of a run has no history to average
                # against, and seeding the average with a zero would
                # spend the first second of every run fading up from
                # black.
                self._low, self._high = low, high
            else:
                self._low = (1.0 - self.ALPHA) * self._low + self.ALPHA * low
                self._high = (1.0 - self.ALPHA) * self._high + self.ALPHA * high
                if self._high - self._low < self.EPS:
                    # A long run of degenerate frames can collapse the
                    # smoothed span even though no single frame did.
                    self._high = self._low + self.EPS
            low_bound, high_bound = self._low, self._high
        span = high_bound - low_bound
        if span <= self.EPS * 2.0:
            # A frame with no relief at all -- a blank wall at the model's
            # resolution, a lens cap. The arithmetic below would divide by
            # the widened epsilon and clip every pixel to the LOW end,
            # rendering solid black. Black in this colour map means "very
            # far away", which is a confident statement about a frame that
            # contains no depth information at all. Mid-grey says nothing,
            # which is the honest answer.
            return np.full(depth.shape, 128, np.uint8)
        # The bounds read out of the critical section above, not
        # `self._low`/`self._high` again: another thread may have folded
        # a newer frame in between, and normalising against bounds that
        # are not the ones this span was computed from is the same bug
        # from the other end.
        scaled = (np.clip(depth, low_bound, high_bound) - low_bound) / span
        return (scaled * 255.0).astype(np.uint8)


def _resize_to_bound(array, max_edge_px: int, interpolation):
    """Down to `max_edge_px` on the longest side. Never up.

    MiDaS-small's own transform already caps its output near 256x192, and
    stretching that to 320 would spend bytes inventing pixels the phone
    can invent for free -- for a picture containing no more information.
    """
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_edge_px or longest == 0:
        return array
    scale = max_edge_px / float(longest)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(array, target, interpolation=interpolation)


def _encode_edge_map(edges, *, policy: PreviewPolicy):
    """A Canny map, small, as PNG. White on black.

    PNG rather than JPEG, and the reason is the content: a Canny map is
    binary, and JPEG's block transform rings around every hard
    transition -- so the cheaper-sounding format produces a halo on each
    of several thousand edges AND a file several times larger. Measured
    against this repository's own venv, PNG came out 3-5x smaller than
    JPEG q80 for the same edge map, with nothing to explain away.

    Resized with INTER_AREA and then re-thresholded, rather than with
    INTER_NEAREST. Nearest-neighbour on one-pixel-wide lines drops about
    half of them at a 2x reduction, which renders a doorway as a dashed
    suggestion of a doorway. Area-averaging touches every contributing
    pixel, and `> 0` keeps any output pixel that any edge reached -- so
    the lines stay continuous, stay binary, and stay the colour the
    algorithm actually produced.
    """
    small = _resize_to_bound(edges, policy.max_edge_px, cv2.INTER_AREA)
    if small is not edges:
        small = np.where(small > 0, np.uint8(255), np.uint8(0))
    ok, buffer = cv2.imencode(
        ".png", small, [int(cv2.IMWRITE_PNG_COMPRESSION), policy.png_compression]
    )
    if not ok:
        raise RuntimeError("cv2.imencode refused the edge map")
    return bytes(buffer), MEDIA_TYPE_PNG, int(small.shape[1]), int(small.shape[0])


def _encode_depth(depth, normaliser, *, policy: PreviewPolicy):
    """Relative inverse depth, colourised, as JPEG.

    INFERNO because that is what MiDaS's and DPT's own visualisers use,
    so a preview here looks like every published picture of this model's
    output; because it is monotonic in perceived lightness, so
    near-versus-far survives being read by somebody colourblind or on a
    phone in sunlight; and because `cv2.applyColorMap` already has it, so
    nothing new is installed for it. JET was rejected for the reason
    Google's Turbo write-up gives: its luminance is not monotonic, so it
    invents banding that reads as structure.

    JPEG here and PNG for edges, for symmetric reasons -- a colourised
    depth map is smooth and photographic, which is what JPEG is for and
    what PNG is worst at.
    """
    small = _resize_to_bound(depth, policy.max_edge_px, cv2.INTER_AREA)
    # Percentiles on the SMALL array. It is the only step whose cost
    # scales with pixel count rather than being near-constant, and
    # resizing first makes several times the difference to it, for a
    # scale nobody reading a phone-sized panel can see.
    levels = normaliser.to_uint8(small)
    coloured = cv2.applyColorMap(levels, cv2.COLORMAP_INFERNO)
    ok, buffer = cv2.imencode(
        ".jpg", coloured, [int(cv2.IMWRITE_JPEG_QUALITY), policy.jpeg_quality]
    )
    if not ok:
        raise RuntimeError("cv2.imencode refused the depth map")
    return (
        bytes(buffer),
        MEDIA_TYPE_JPEG,
        int(coloured.shape[1]),
        int(coloured.shape[0]),
    )


# -- drawing an overlay over a line drawing -----------------------------
#
# Five of the seven kinds are "structure, plus what the algorithm found on
# top of it". They share a canvas, a caption, a colour discipline and an
# encoder, because five hand-rolled versions of those would drift into
# five different-looking screens in one app.
#
# Colour discipline, and it is the load-bearing part:
#
#   dim grey     the scene. Recedes.
#   green        the algorithm succeeded / kept this.
#   red          the algorithm rejected this, or it is a leak.
#   amber        a boundary case worth a second look.
#   per-class    object detection only, where identity IS the question.
#
# Everything is drawn with `LINE_8` rather than `LINE_AA`. Antialiasing
# would produce hundreds of intermediate colours on an otherwise
# three-colour image, and PNG's filters are exactly the thing that turns
# a three-colour image into two kilobytes.

# The scene. Dark enough to sit behind an overlay, light enough that a
# doorway is still a doorway.
_STRUCTURE_BGR = (70, 70, 70)
_GOOD_BGR = (90, 230, 90)
_REJECTED_BGR = (70, 70, 235)
_BOUNDARY_BGR = (0, 175, 255)
_REGION_BGR = (235, 90, 235)
_GRID_BGR = (42, 42, 42)
_INK_BGR = (245, 245, 245)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CAPTION_SCALE = 0.34

# Below this many pixels of displacement, a flow track is drawn as a dot
# rather than as an arrow. Half of `optical_flow`'s own
# `MAX_FORWARD_BACKWARD_PX` tolerance: a track the experiment accepts as
# good can still have moved by a fraction of a pixel, and there is no
# direction in a fraction of a pixel worth drawing an arrow for.
_FLOW_MIN_DRAWN_PX = 0.5


def _canvas(scene):
    """The line drawing, as a colour image ready to be drawn on."""
    structure = scene.structure
    height, width = structure.shape[:2]
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[structure > 0] = _STRUCTURE_BGR
    return canvas


def _caption(canvas, text: str, line: int = 0) -> None:
    """One line of small white text, legible over anything.

    Drawn twice -- black one pixel down and right, then white -- because
    a caption over an edge map has no guaranteed background and white on
    white is the one failure a caption cannot recover from.
    """
    origin = (5, 13 + line * 12)
    shadow = (origin[0] + 1, origin[1] + 1)
    cv2.putText(canvas, text, shadow, _FONT, _CAPTION_SCALE, (0, 0, 0), 1)
    cv2.putText(canvas, text, origin, _FONT, _CAPTION_SCALE, _INK_BGR, 1)


# Golden-angle hue stepping rather than a hash. `TRACKED_CLASSES` is six
# names plus whatever else COCO returns, and for a small set an even
# spread separates colours better than hashing does -- a hash is what you
# reach for when the vocabulary is open and unknown, which this one is
# not.
_GOLDEN_ANGLE_DEG = 137.508


@functools.lru_cache(maxsize=128)
def _class_colour(name: str) -> tuple:
    """A stable colour per class name. Same name, same colour, always.

    Cached rather than recomputed: the same six or seven names arrive
    every frame, and a colour that changed between frames would make a
    tracked object look like a new one each time.
    """
    index = sum(ord(character) for character in name)
    hue = int((index * _GOLDEN_ANGLE_DEG) % 180)
    swatch = cv2.cvtColor(np.uint8([[[hue, 190, 255]]]), cv2.COLOR_HSV2BGR)
    return tuple(int(channel) for channel in swatch[0, 0])


def _contrast_ink(bgr: tuple) -> tuple:
    """Black or white, whichever can be read on `bgr`."""
    blue, green, red = bgr
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def _encode_canvas(canvas, policy: PreviewPolicy):
    """PNG, for every overlay kind. Not JPEG, and the reason is measured.

    An overlay canvas is a handful of flat colours on black, which is the
    case PNG's filters were designed for and the case JPEG is worst at:
    JPEG would ring around every one of several thousand edge pixels and
    produce a LARGER file doing it. The only kind that gets JPEG is the
    colourised depth map, which is genuinely smooth and genuinely
    photographic in its statistics.
    """
    ok, buffer = cv2.imencode(
        ".png", canvas, [int(cv2.IMWRITE_PNG_COMPRESSION), policy.png_compression]
    )
    if not ok:
        raise RuntimeError("cv2.imencode refused the overlay canvas")
    return (
        bytes(buffer),
        MEDIA_TYPE_PNG,
        int(canvas.shape[1]),
        int(canvas.shape[0]),
    )


def _encode_keypoints(payload, *, policy: PreviewPolicy):
    """ORB keypoints, and the grid the coverage figure is counted on.

    Two drawings of one fact, because `keypoint_count: 995,
    spatial_coverage: 0.57` is two facts and a person needs both to read
    either. The dots say how much texture; the tinted cells say whether it
    is spread out or piled in one corner -- which is the whole reason
    `feature_detection` reports coverage beside the count at all.

    The experiment has already thinned the points. Drawing a thousand
    markers on a 320-pixel panel produces a grey rectangle, and every
    SLAM front end that has met this problem -- ORB-SLAM's grid extractor,
    VINS-Mono's 100-300 cap -- solves it by bucketing rather than by
    drawing everything smaller.
    """
    canvas = _canvas(payload.scene)
    width, height = payload.scene.size
    grid = payload.coverage_grid

    if grid > 0:
        cells = set(payload.coverage_cells or ())
        for index in range(1, grid):
            x = int(index * width / grid)
            y = int(index * height / grid)
            cv2.line(canvas, (x, 0), (x, height), _GRID_BGR, 1)
            cv2.line(canvas, (0, y), (width, y), _GRID_BGR, 1)
        # Occupied cells tinted rather than outlined. An outline competes
        # with the scene's own lines; a wash does not.
        tint = canvas.copy()
        for column, row in cells:
            x0 = int(column * width / grid)
            y0 = int(row * height / grid)
            x1 = int((column + 1) * width / grid)
            y1 = int((row + 1) * height / grid)
            cv2.rectangle(tint, (x0, y0), (x1, y1), (0, 90, 0), -1)
        # 0.12, not more. Rendered at 0.18 the wash reached the structure's
        # own white lines and turned the whole room green, which made the
        # coverage signal louder than the scene it is a statement about.
        cv2.addWeighted(tint, 0.12, canvas, 0.88, 0.0, canvas)

    for x, y in np.asarray(payload.xy, dtype=np.int32):
        cv2.circle(canvas, (int(x), int(y)), 2, _GOOD_BGR, -1)

    shown = 0 if payload.xy is None else len(payload.xy)
    _caption(
        canvas,
        f"{payload.detected} keypoints ({shown} drawn)  coverage "
        f"{payload.coverage:.2f}",
    )
    return _encode_canvas(canvas, policy)


def _encode_detections(payload, *, policy: PreviewPolicy):
    """Boxes, class names and scores -- including the unconvincing ones.

    Detections BELOW the metrics' threshold are drawn, thin and faded.
    They are the whole reason this view exists: physical testing produced
    160 `person` detections in a room with nobody in it, and the only way
    to find out what the model was looking at is to see where it drew the
    box. A viewer that hid the near-misses would have hidden the evidence.

    Thickness 2 for accepted and 1 for below-threshold follows the
    adaptive line width every annotator converges on at this size
    (Ultralytics computes `round(sum(shape[:2]) / 2 * 0.003)`, which is 1
    at 320x240 and floors to 2). The label chip flips INSIDE the box when
    there is no room above it, which is the standard behaviour and the
    difference between a readable top row and clipped text.
    """
    canvas = _canvas(payload.scene)
    width, height = payload.scene.size
    boxes = np.asarray(payload.boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(payload.scores, dtype=np.float32).reshape(-1)
    accepted = 0

    for box, label, score in zip(boxes, payload.labels, scores):
        x0 = int(max(0, min(width - 1, box[0])))
        y0 = int(max(0, min(height - 1, box[1])))
        x1 = int(max(0, min(width - 1, box[2])))
        y1 = int(max(0, min(height - 1, box[3])))
        low = bool(score < payload.threshold)
        if not low:
            accepted += 1
        colour = _class_colour(str(label))
        if low:
            # Faded towards mid-grey rather than dashed. Dashing costs a
            # loop of `cv2.line` per box for a distinction that a 320-pixel
            # panel renders as "slightly broken", and the fade reads at
            # this size where the dashes do not.
            colour = tuple(int(channel * 0.45 + 55) for channel in colour)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 1 if low else 2)

        text = f"{label} {score:.2f}"
        chip_height = 12
        chip_width = min(int(7.0 * len(text)) + 6, width - x0)
        chip_y0 = y0 - chip_height if y0 - chip_height >= 0 else y0
        cv2.rectangle(
            canvas,
            (x0, chip_y0),
            (x0 + chip_width, chip_y0 + chip_height),
            colour,
            -1,
        )
        cv2.putText(
            canvas,
            text,
            (x0 + 3, chip_y0 + 9),
            _FONT,
            0.30,
            _contrast_ink(colour),
            1,
        )

    below = len(scores) - accepted
    # `payload.accepted_total`, not a `getattr` with a fallback:
    # `DetectionPreview` requires the field, so a payload without one is a
    # TypeError at construction rather than a caption that quietly lies.
    total = payload.accepted_total
    shown = (
        f"{accepted} over {payload.threshold:.2f}"
        if total <= accepted
        # The case that matters: say what was left out, in the picture, so
        # it cannot silently disagree with the `detections` metric beside
        # it.
        else f"{accepted} of {total} over {payload.threshold:.2f} drawn"
    )
    _caption(canvas, shown + (f"  (+{below} below, faded)" if below else ""))
    return _encode_canvas(canvas, policy)


# Hue by direction, precomputed once. The formula is OpenCV's own
# `draw_hsv` from `samples/python/opt_flow.py`: angle mapped onto hue's
# 0-179 range, so a camera panning right colours every arrow the same and
# a scene of independently moving things does not.
_FLOW_HUES = cv2.cvtColor(
    np.dstack(
        (
            np.arange(180, dtype=np.uint8),
            np.full(180, 200, np.uint8),
            np.full(180, 255, np.uint8),
        )
    ),
    cv2.COLOR_HSV2BGR,
)[0]


def _encode_flow(payload, *, policy: PreviewPolicy):
    """Where each tracked point went, and which tracks were thrown away.

    Arrows rather than accumulated polylines. OpenCV's `lk_track.py`
    sample draws a ten-frame history per point, which is the better
    picture and needs state this experiment deliberately does not keep --
    `optical_flow` holds exactly one previous frame and says so. What it
    does have is a one-step displacement per point, which is the shape of
    `opt_flow.py`'s grid-of-arrows sample, applied to an irregular point
    set instead of a lattice.

    Arrow LENGTH is floored. Real displacements are frequently sub-pixel
    at this scale, and an arrow shorter than its own head is a dot with no
    direction in it -- which would make a slow pan and a still camera look
    identical. The honest magnitude is in the caption.

    Rejected seeds are drawn in red. "It tracked nothing" and "it tracked
    confident nonsense and the forward-backward check caught it" are the
    same number of tracked points and completely different situations.
    """
    canvas = _canvas(payload.scene)
    origins = np.asarray(payload.origins, dtype=np.float32).reshape(-1, 2)
    displacements = np.asarray(payload.displacements, dtype=np.float32).reshape(-1, 2)

    for origin, displacement in zip(origins, displacements):
        magnitude = float(np.hypot(displacement[0], displacement[1]))
        angle = float(np.arctan2(displacement[1], displacement[0])) + np.pi
        colour = tuple(int(c) for c in _FLOW_HUES[int(angle * 90.0 / np.pi) % 180])
        if magnitude < _FLOW_MIN_DRAWN_PX:
            # A dot, not a floored arrow, and the threshold is not zero.
            #
            # Lucas-Kanade on a still camera returns sub-pixel jitter --
            # measured around 0.02 to 0.3 px, well inside the 1.0 px
            # forward-backward tolerance that already calls those tracks
            # good -- with random directions. Floored to a five-pixel
            # minimum those become a dense field of confident, colourful,
            # contradictory arrows, which reads as a scene full of
            # independent motion. The caption said "median 0.1px" and
            # nobody was reading the caption.
            cv2.circle(canvas, (int(origin[0]), int(origin[1])), 1, colour, -1)
            continue
        drawn = float(min(max(magnitude, 5.0), 14.0))
        tip = (
            int(origin[0] + displacement[0] / magnitude * drawn),
            int(origin[1] + displacement[1] / magnitude * drawn),
        )
        cv2.arrowedLine(
            canvas,
            (int(origin[0]), int(origin[1])),
            tip,
            colour,
            1,
            tipLength=0.35,
        )

    for x, y in np.asarray(payload.rejected, dtype=np.int32).reshape(-1, 2):
        cv2.circle(canvas, (int(x), int(y)), 2, _REJECTED_BGR, -1)

    rejected = 0 if payload.rejected is None else len(payload.rejected)
    _caption(
        canvas,
        f"tracked {payload.tracked_count}/{payload.seeded_count}  median "
        f"{payload.median_flow_px:.1f}px",
    )
    if rejected:
        _caption(canvas, f"{rejected} rejected by forward-backward", line=1)
    return _encode_canvas(canvas, policy)


def _encode_redaction(payload, *, policy: PreviewPolicy):
    """The blurred rectangle, and what the detector could still see in it.

    Read this one carefully, because the honest version is not the obvious
    version. `redaction_impact` detects keypoints TWICE -- once on the
    frame, once on the blurred copy -- and never matches them, so there is
    no such thing as "this exact point was lost". Claiming one would be
    inventing a correspondence the experiment never computed.

    So: every pre-redaction keypoint is a dim grey dot meaning "there was
    texture here", and survivors are drawn on top in a colour that says
    where they are. A grey dot with nothing on it reads as lost without
    anybody having asserted it.

        magenta rectangle   what was blurred
        amber rectangle     the boundary band, one ORB patch wide
        red dot             survived INSIDE the blur -- the leak
        amber dot           survived on the boundary -- an artefact of
                            the edge, not texture from the scene
        green dot           survived outside, which is the control

    The background is the Canny of the ALREADY-BLURRED copy, so the blur
    is visible as an absence of lines rather than being something a person
    has to take on trust.
    """
    canvas = _canvas(payload.scene)
    x0, y0, x1, y1 = (int(round(value)) for value in payload.region)
    margin = int(round(payload.boundary_margin_px))

    for x, y in np.asarray(payload.before, dtype=np.int32).reshape(-1, 2):
        cv2.circle(canvas, (int(x), int(y)), 1, (105, 105, 105), -1)
    cv2.rectangle(canvas, (x0 - margin, y0 - margin), (x1 + margin, y1 + margin),
                  _BOUNDARY_BGR, 1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), _REGION_BGR, 2)

    for points, colour in (
        (payload.survived_outside, _GOOD_BGR),
        (payload.survived_on_boundary, _BOUNDARY_BGR),
        (payload.survived_inside, _REJECTED_BGR),
    ):
        for x, y in np.asarray(points, dtype=np.int32).reshape(-1, 2):
            cv2.circle(canvas, (int(x), int(y)), 2, colour, -1)

    inside = len(np.asarray(payload.survived_inside).reshape(-1, 2))
    boundary = len(np.asarray(payload.survived_on_boundary).reshape(-1, 2))
    _caption(canvas, f"{inside} survived inside the blur, {boundary} on its edge")
    return _encode_canvas(canvas, policy)


def _encode_frame_quality(payload, *, policy: PreviewPolicy):
    """The frame's structure, and the histogram the exposure figures came from.

    NO VERDICT IS DRAWN, and that is the point rather than an omission.
    Variance-of-Laplacian has no portable threshold -- the standard
    reference for the technique says the value has to be tuned per dataset
    -- and this Lab has exactly one physical run to calibrate against,
    which is none. So the picture shows the distribution the numbers were
    counted off, the run document says where this frame sits inside the
    run's OWN observed range, and neither of them says "blurry".

    The histogram is the array `frame_quality` already built for its
    exposure metrics, redrawn. The two vertical markers are the exact
    levels it counts clipping at, so a person can see the spike sitting on
    one instead of reading `overexposed_fraction: 0.083` and wondering
    what 8.3% looks like.
    """
    canvas = _canvas(payload.scene)
    width, height = payload.scene.size
    strip_height = max(24, height // 5)
    top = height - strip_height

    histogram = np.asarray(payload.histogram, dtype=np.float64).reshape(-1)
    peak = float(histogram.max()) if histogram.size else 0.0
    cv2.rectangle(canvas, (0, top), (width, height), (16, 16, 16), -1)
    if peak > 0:
        for x in range(width):
            level = int(round(x * 255 / max(width - 1, 1)))
            bar = int(histogram[level] / peak * (strip_height - 3))
            if bar > 0:
                cv2.line(canvas, (x, height - 1), (x, height - 1 - bar), (150, 150, 150), 1)
    for level, colour in (
        (payload.underexposed_level, _BOUNDARY_BGR),
        (payload.overexposed_level, _REJECTED_BGR),
    ):
        # `round`, matching the bar loop above. Two independent
        # truncations put a marker up to a pixel off its own bar, which is
        # imperceptible and is also the kind of thing somebody eventually
        # spends an hour on.
        x = int(round(level * (width - 1) / 255))
        cv2.line(canvas, (x, top), (x, height), colour, 1)
    cv2.line(canvas, (0, top), (width, top), (60, 60, 60), 1)

    _caption(
        canvas,
        f"clipped {payload.overexposed_fraction * 100:.1f}% bright / "
        f"{payload.underexposed_fraction * 100:.1f}% dark",
    )
    return _encode_canvas(canvas, policy)


# What each kind renders as. Adding a visual experiment is a line here
# plus a `preview_kind` on its metadata -- not a new endpoint, not a
# second contract, and not another copy of the freshness and identity
# rules, which are the parts that are easy to get subtly wrong.
_RENDERERS = {
    PREVIEW_KIND_EDGE_MAP: _encode_edge_map,
    PREVIEW_KIND_RELATIVE_DEPTH: _encode_depth,
    PREVIEW_KIND_KEYPOINTS: _encode_keypoints,
    PREVIEW_KIND_DETECTIONS: _encode_detections,
    PREVIEW_KIND_FLOW_TRACKS: _encode_flow,
    PREVIEW_KIND_REDACTION: _encode_redaction,
    PREVIEW_KIND_FRAME_QUALITY: _encode_frame_quality,
}

_MEDIA_TYPES = {
    PREVIEW_KIND_EDGE_MAP: MEDIA_TYPE_PNG,
    PREVIEW_KIND_RELATIVE_DEPTH: MEDIA_TYPE_JPEG,
    PREVIEW_KIND_KEYPOINTS: MEDIA_TYPE_PNG,
    PREVIEW_KIND_DETECTIONS: MEDIA_TYPE_PNG,
    PREVIEW_KIND_FLOW_TRACKS: MEDIA_TYPE_PNG,
    PREVIEW_KIND_REDACTION: MEDIA_TYPE_PNG,
    PREVIEW_KIND_FRAME_QUALITY: MEDIA_TYPE_PNG,
}

_KIND_DESCRIPTIONS = {
    PREVIEW_KIND_KEYPOINTS: (
        "The ORB keypoints this experiment counted, thinned so they can be "
        "seen, over a line drawing of the frame and the 8x8 grid the "
        "coverage figure is measured on."
    ),
    PREVIEW_KIND_DETECTIONS: (
        "What the detector found, with each box's class and score, over a "
        "line drawing of the frame. Boxes that came at least halfway to the "
        "threshold the metrics use are drawn faded rather than hidden; "
        "anything below that the detector had already dismissed."
    ),
    PREVIEW_KIND_FLOW_TRACKS: (
        "One arrow per tracked point, coloured by direction, over a line "
        "drawing of the frame. Red dots are seeds the forward-backward "
        "check rejected."
    ),
    PREVIEW_KIND_REDACTION: (
        "The blurred rectangle and the keypoints that survived it, over a "
        "line drawing of the ALREADY-BLURRED frame. Red means texture the "
        "blur failed to destroy."
    ),
    PREVIEW_KIND_FRAME_QUALITY: (
        "A line drawing of the frame and the luminance histogram the "
        "exposure figures were counted from, with the clipping levels "
        "marked. No verdict is drawn: these metrics have no calibrated "
        "threshold on this camera yet."
    ),
    PREVIEW_KIND_EDGE_MAP: (
        "The Canny edge map this experiment measured its density from, "
        "white on black. Not a photograph and not a filtered one -- the "
        "only pixels here are the ones the algorithm called an edge."
    ),
    PREVIEW_KIND_RELATIVE_DEPTH: (
        "Relative inverse depth, colourised. Bright is nearer and dark is "
        "farther, on a scale the Tower rebuilds continuously. These are "
        "NOT metres, and the difference between two pixels is not a "
        "distance."
    ),
}


def _renderer_args(kind: str, normaliser):
    """The extra positional a renderer needs, if it needs one.

    Only depth carries state between frames, so only depth is handed
    anything. A table where every entry took the same unused argument
    would be a table pretending the two are more alike than they are.
    """
    if kind == PREVIEW_KIND_RELATIVE_DEPTH:
        return (normaliser,)
    return ()


def _etag(run_id: str | None, result_seq: int) -> str:
    """The identity of one preview, as an HTTP validator.

    Strong rather than weak, and entitled to be: the bytes for a given
    `(run_id, result_seq)` are computed once and cached, so two responses
    carrying this tag are byte-identical. A new run mints a new `run_id`,
    so a tag cannot outlive the run it names.
    """
    return f'"{run_id or "none"}:{result_seq}"'


def _etag_matches(header: str, etag: str) -> bool:
    """`If-None-Match`, tolerantly. Never raises on a hostile header."""
    for candidate in header.split(","):
        candidate = candidate.strip()
        if candidate in ("*", etag):
            return True
        if candidate.startswith("W/") and candidate[2:] == etag:
            return True
    return False


def _refresh_age(rendered: RenderedPreview, now: float) -> RenderedPreview:
    """The cached bytes with an honest age rather than the one they were
    encoded at. Nothing else about a preview changes after it is made."""
    return RenderedPreview(
        image_bytes=rendered.image_bytes,
        media_type=rendered.media_type,
        width=rendered.width,
        height=rendered.height,
        kind=rendered.kind,
        run_id=rendered.run_id,
        result_seq=rendered.result_seq,
        captured_at=rendered.captured_at,
        age_s=round(now - rendered.captured_at, 4),
        render_ms=rendered.render_ms,
        etag=rendered.etag,
        treatment=rendered.treatment,
    )


class LivePreview:
    """The slot: one capture, one encoding, and the counters for both."""

    def __init__(self, policy: PreviewPolicy | None = None, clock=time.time) -> None:
        self._policy = policy or PreviewPolicy()
        self._clock = clock

        # THE slot. Assigned as a whole, read as a whole, never mutated.
        self._latest: Capture | None = None
        # THE cache. One encoding of one capture, so a phone polling
        # faster than the Tower produces costs a round trip rather than
        # an encode per request.
        self._rendered: RenderedPreview | None = None

        self._kind: str | None = None
        self._run_id: str | None = None
        self._epoch = 0
        self._live = False
        self._last_capture_at = 0.0
        self._normaliser = _DepthNormaliser()

        # Written on the frame path only, read anywhere. Plain ints: a
        # single load is atomic, and a diagnostic that is one frame
        # behind is a diagnostic rather than a bug.
        self._offered = 0
        self._captured = 0
        self._skipped_by_throttle = 0
        self._empty_takes = 0
        self._replaced_unread = 0
        # NOT one of those, despite sitting with them. This one is
        # written by `render()` on a WORKER thread and read by
        # `capture()` on the loop, which is the reverse direction from
        # every counter above it. A single int store is still atomic, so
        # nothing tears -- but a `begin()` landing between a render's
        # epoch check and its write here can leave a value from a run
        # that `begin()` had just reset, which makes `replaced_unread`
        # off by one for one frame. Tolerated, named, and confined to a
        # diagnostic: no client ever sees a picture because of it.
        self._last_rendered_seq = -1

        # Written by whichever worker thread is serving, and read when
        # the status document is built -- which happens BOTH on a worker
        # thread (the result channel's poller) and on the event loop (a
        # `cv_lab_status` reply). So the loop can briefly wait on this.
        #
        # Held only for counter arithmetic: five scalar stores and two
        # `Running.add`. NEVER around an encode, never around anything
        # that touches a device or a file. The wait is microseconds, and
        # the alternative -- unguarded reads of a `Running` mid-update --
        # would publish a mean computed from a total and a count that do
        # not belong to each other.
        #
        # The FRAME PATH never touches it, and that is the part that
        # matters: a lock on the measured path to make a diagnostic one
        # frame fresher would be paying in the wrong currency.
        self._stats_guard = threading.Lock()
        self._encoded = 0
        self._encode_failures = 0
        self._served = 0
        self._not_modified = 0
        self._refused = 0
        self._render_ms = Running()
        self._payload_bytes = Running()

    @property
    def policy(self) -> PreviewPolicy:
        return self._policy

    @property
    def is_live(self) -> bool:
        """Whether a capture would be kept if one were offered.

        False for a Tower with previews off, for an experiment with no
        visual output, and for a run that is not RUNNING. Checked by the
        frame path before it asks the experiment for anything, so the
        three cheapest reasons to do nothing cost one attribute read.
        """
        return self._live

    # -- lifecycle, from the Lab, on the event loop ---------------------

    def begin(self, run_id: str | None, kind: str | None) -> bool:
        """A run went RUNNING. Returns whether anything will be captured.

        Everything is reset, the depth normaliser included: carrying a
        previous run's near-and-far into this one would be the old
        experiment deciding what the new one looks like.
        """
        self._epoch += 1
        self._latest = None
        self._rendered = None
        self._normaliser = _DepthNormaliser()
        self._run_id = run_id
        self._kind = kind
        self._last_capture_at = 0.0
        self._last_rendered_seq = -1
        self._live = bool(self._policy.enabled and kind in _RENDERERS)
        return self._live

    def suspend(self) -> None:
        """A run left RUNNING -- paused, stopped, failed or released.

        The picture goes with it, immediately, rather than ageing out. A
        frozen last frame under a "paused" label reads as live to anybody
        not reading the label, and `raw_ephemeral` promises live-view-only
        in both directions: the phone will not store it, and the Tower
        will not go on serving it once it stopped being a view of
        anything.
        """
        self._epoch += 1
        self._live = False
        self._latest = None
        self._rendered = None

    def end(self) -> None:
        """The Lab is gone. `suspend`, named for the call site."""
        self.suspend()
        self._run_id = None
        self._kind = None

    # -- the frame path -------------------------------------------------

    def wants_capture(self, now: float) -> bool:
        """Whether to bother asking the experiment for its array.

        Checked BEFORE `take_preview()`, so a throttled frame costs one
        subtraction and one comparison and the experiment goes on holding
        the array it already had.
        """
        if not self._live:
            return False
        return (now - self._last_capture_at) >= self._policy.min_interval_s

    def capture(self, array, *, run_id: str | None, result_seq: int, now: float):
        """Take this frame's array. Two assignments and a counter.

        Called from `CVLab.process()`, on the event loop, with no lock
        held and nothing here that can block. There is no resize, no
        colour conversion, no encode and no copy: the array the
        experiment computed is the array this holds, and the only work
        the frame path does for the viewer is deciding to keep it.
        """
        self._offered += 1
        if not self._live:
            return
        previous = self._latest
        if previous is not None and previous.result_seq != self._last_rendered_seq:
            # Dropped because a newer one arrived. The intended
            # behaviour, counted rather than hidden: it is how a person
            # reads "the phone saw one preview in nine" afterwards
            # instead of guessing.
            self._replaced_unread += 1
        self._latest = Capture(
            kind=self._kind,
            array=array,
            run_id=run_id,
            result_seq=result_seq,
            captured_at=now,
            epoch=self._epoch,
        )
        self._last_capture_at = now
        self._captured += 1

    def note_throttled(self) -> None:
        """A frame was processed and deliberately not captured."""
        self._offered += 1
        self._skipped_by_throttle += 1

    def note_empty(self) -> None:
        """The experiment offered nothing. Not an error, and not silent."""
        self._offered += 1
        self._empty_takes += 1

    # -- serving, on a worker thread ------------------------------------

    def render(self, *, run_id: str | None = None, if_none_match: str | None = None):
        """The newest preview, encoded. Never raises.

        Returns a `RenderedPreview`, a `PreviewNotModified`, or a
        `PreviewRefusal` naming one of the closed reasons. A caller that
        gets a refusal has not been handed a stale picture, an empty one,
        or half of one -- which is the only behaviour that makes "what
        the algorithm sees" worth trusting.
        """
        if not self._policy.enabled:
            return self._refuse(
                PREVIEW_DISABLED,
                "this Tower serves no CV Lab previews; they are switched "
                "off with TOWER_CV_PREVIEW",
            )
        kind = self._kind
        if kind is None or kind not in _RENDERERS:
            return self._refuse(
                PREVIEW_NOT_VISUAL,
                "the running experiment produces no visual output",
            )
        current_run = self._run_id
        if run_id is not None and run_id != current_run:
            return self._refuse(
                PREVIEW_RUN_CHANGED,
                f"there is no preview for run {run_id!r}; the Lab is now on "
                f"{current_run!r}",
                current_run_id=current_run,
            )

        epoch = self._epoch
        capture = self._latest
        if capture is None or not self._live:
            return self._refuse(
                PREVIEW_NONE_YET,
                "no frame has produced a preview for this run yet",
            )
        now = self._clock()
        age = now - capture.captured_at
        if age > self._policy.max_age_s:
            return self._refuse(
                PREVIEW_STALE,
                f"the newest preview is {age:.1f}s old, past the "
                f"{self._policy.max_age_s:g}s this Tower will serve",
                current_run_id=current_run,
            )

        etag = _etag(capture.run_id, capture.result_seq)
        if if_none_match is not None and _etag_matches(if_none_match, etag):
            with self._stats_guard:
                self._not_modified += 1
            return PreviewNotModified(
                etag=etag, run_id=capture.run_id, result_seq=capture.result_seq
            )

        cached = self._rendered
        if cached is not None and cached.etag == etag:
            with self._stats_guard:
                self._served += 1
            return _refresh_age(cached, now)

        started = time.perf_counter()
        try:
            image_bytes, media_type, width, height = _RENDERERS[capture.kind](
                capture.array,
                *_renderer_args(capture.kind, self._normaliser),
                policy=self._policy,
            )
        except BaseException:
            # Contained here and nowhere else. The experiment has already
            # returned its result, its numbers are already on the wire,
            # and a broken encoder must cost a picture rather than a run.
            with self._stats_guard:
                self._encode_failures += 1
                self._refused += 1
            logger.exception(
                "[Tower][CVLab] rendering a %s preview failed; the run is "
                "unaffected and telemetry continues",
                capture.kind,
            )
            return PreviewRefusal(
                reason=PREVIEW_RENDER_FAILED,
                message=(
                    "this preview could not be rendered; the experiment is "
                    "unaffected and its figures are still current"
                ),
                current_run_id=current_run,
            )
        render_ms = (time.perf_counter() - started) * 1000.0

        if self._epoch != epoch:
            # Stopped, paused or restarted while we were encoding. These
            # bytes belong to a run that is no longer live, and the whole
            # point of the epoch is that they never get served under the
            # next run's name.
            return self._refuse(
                PREVIEW_RUN_CHANGED,
                "the run ended while its preview was being rendered",
                current_run_id=self._run_id,
            )

        rendered = RenderedPreview(
            image_bytes=image_bytes,
            media_type=media_type,
            width=width,
            height=height,
            kind=capture.kind,
            run_id=capture.run_id,
            result_seq=capture.result_seq,
            captured_at=capture.captured_at,
            age_s=round(self._clock() - capture.captured_at, 4),
            render_ms=round(render_ms, 4),
            etag=etag,
        )
        # The cache and the read-marker, in that order. Both are single
        # assignments and neither can be observed half-written.
        self._rendered = rendered
        self._last_rendered_seq = capture.result_seq
        with self._stats_guard:
            self._encoded += 1
            self._served += 1
            self._render_ms.add(render_ms)
            self._payload_bytes.add(float(len(image_bytes)))
        return rendered

    def _refuse(self, reason: str, message: str, current_run_id=None):
        with self._stats_guard:
            self._refused += 1
        return PreviewRefusal(
            reason=reason, message=message, current_run_id=current_run_id
        )

    # -- what the status document says ----------------------------------

    def descriptor(self) -> dict | None:
        """The `artifact` block, or `None` -- see `why_none` for the words.

        Deliberately free of anything that changes per frame. This rides
        the result channel, whose `revision` is a hash of the payload,
        and a `result_seq` in here would make every poll report news
        about a Lab that had merely kept running. The per-frame identity
        travels on the bytes instead, in headers, where the client that
        is actually holding a picture can read it.
        """
        kind = self._kind
        if not self._policy.enabled or kind is None or kind not in _RENDERERS:
            return None
        return {
            "contract": PREVIEW_CONTRACT,
            "kind": ARTIFACT_KIND_LIVE_PREVIEW,
            # How to READ the picture. Never what it means.
            "visual_kind": kind,
            "description": _KIND_DESCRIPTIONS[kind],
            # -- the privacy half, which is not optional ----------------
            #
            # `raw_ephemeral` is iOS's own value with iOS's own
            # definition: "Untreated imagery. Permitted only for the
            # live, in-memory view of what the wearer currently sees --
            # never for anything persisted, and never for anything a
            # cartridge stored and re-served later." Every clause of that
            # is true here, and none of it is a concession: it is the
            # STRICT answer rather than a lenient one, and it is one a
            # phone already knows how to obey.
            #
            # It is not `redacted`, because nothing was redacted. No face
            # detector runs on this path. An edge map keeps a jawline, a
            # hairline and the frames of a pair of glasses; a depth map
            # keeps a silhouette. Derived is not unrecognisable, and the
            # one thing this contract must never do is let "it is only a
            # Canny map" become a privacy claim.
            "treatment": TREATMENT_RAW_EPHEMERAL,
            # A PROCESS claim, in the discipline
            # `world_builder/redaction.py` set: it says what was DONE,
            # never what the result is safe for. "none" is the whole
            # sentence.
            "face_filter": PREVIEW_FACE_FILTER_NONE,
            "persistence": PREVIEW_PERSISTENCE_NONE,
            "derived_from": "one frame, transiently, in memory",
            # -- how to fetch it ----------------------------------------
            #
            # A PATH, not a URL. The Tower does not know what address a
            # phone reached it on, and a client that resolved a base URL
            # to ask for this document can resolve this against the same
            # one.
            "path": PREVIEW_PATH,
            "media_type": _MEDIA_TYPES[kind],
            "run_id": self._run_id,
            "max_age_s": self._policy.max_age_s,
            "poll_interval_s": self._policy.poll_interval_s,
            "max_edge_px": self._policy.max_edge_px,
        }

    def why_none(self) -> str | None:
        """The sentence for `artifact_unavailable_reason`, or `None`."""
        if not self._policy.enabled:
            return (
                "this Tower serves no CV Lab previews; they are switched off "
                "with TOWER_CV_PREVIEW"
            )
        kind = self._kind
        if kind is None or kind not in _RENDERERS:
            return (
                "this experiment produces no visual output, so there is no "
                "picture to serve -- which is not the same as one being "
                "withheld"
            )
        return None

    def stats(self) -> dict:
        """The counters, for the run document's diagnostics.

        Every one of them exists to answer a question that would
        otherwise be answered by guessing: how much of the CV rate the
        phone is actually seeing, what a preview costs, how big one is,
        and whether the encoder has ever failed.
        """
        with self._stats_guard:
            encoded = self._encoded
            encode_failures = self._encode_failures
            served = self._served
            not_modified = self._not_modified
            refused = self._refused
            render_avg = (
                round(self._render_ms.average, 4) if self._render_ms.count else None
            )
            render_max = (
                round(self._render_ms.maximum, 4) if self._render_ms.count else None
            )
            payload_avg = (
                int(self._payload_bytes.average) if self._payload_bytes.count else None
            )
            payload_max = (
                int(self._payload_bytes.maximum) if self._payload_bytes.count else None
            )
            payload_last = (
                int(self._payload_bytes.last) if self._payload_bytes.count else None
            )
        return {
            "enabled": bool(self._policy.enabled),
            "live": bool(self._live),
            "visual_kind": self._kind,
            # Frames the Lab finished and told the preview about.
            "frames_offered": self._offered,
            # ...of which this many became the newest preview.
            "captured": self._captured,
            # ...and this many were never even asked for, because the
            # throttle said the last capture was too recent. This is the
            # figure that says visualisation runs at its own rate.
            "skipped_by_throttle": self._skipped_by_throttle,
            # The experiment had nothing to hand over. Normally zero.
            "empty_takes": self._empty_takes,
            # Captures overwritten before anything rendered them. The
            # intended behaviour, counted: `captured - replaced_unread`
            # is roughly what the phone actually saw.
            "replaced_unread": self._replaced_unread,
            # Encodes actually performed -- far fewer than `captured`,
            # and the gap between the two IS the design.
            "encoded": encoded,
            "encode_failures": encode_failures,
            "served": served,
            # Fetches answered 304 because the phone already had that
            # frame. A high figure means the phone is polling faster than
            # the Tower is producing, which costs a round trip and no
            # encode.
            "not_modified": not_modified,
            "refused": refused,
            "render_ms": render_avg,
            "render_ms_max": render_max,
            "payload_bytes": payload_avg,
            "payload_bytes_max": payload_max,
            "payload_bytes_last": payload_last,
            "max_edge_px": self._policy.max_edge_px,
            "min_interval_s": self._policy.min_interval_s,
            "max_age_s": self._policy.max_age_s,
        }
