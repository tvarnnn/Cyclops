"""Resolving a record back to the picture it was derived from.

WHY THIS IS WORTH BUILDING, AND IT IS NOT OBVIOUS.

The strongest published evidence on wearable memory aids says the IMAGE
is the product and the label is not. MemPal (15 adults aged 62-96, in
their own homes, objects hidden and retrieved after a 40-minute delay)
measured its own answers as correct only 72% of the time in audio form
and its last-seen images as showing the true location only 53% of the
time -- and users still went from 0.81 to 0.95-0.97 retrieval accuracy,
searching 1.1 rooms instead of 1.9. A wrong-but-plausible cue plus a
human closes the gap. A confident sentence does not.

This cartridge already had the pointer: every record carries a capture
id, a frame number and a box. What it had was a DIAGNOSTIC -- the iOS
surface renders it as "Frame reference: capture 22e9d428..., frame 3410"
inside a disclosure -- which is a correct thing to show a developer and
close to useless to a wearer. This module is what turns the pointer into
the picture.

IT IS A DISPLAY FILTER, NOT A PRIVACY TRANSFORMATION, AND IT SAYS SO.

`tower/world_builder/redaction.py` makes exactly this distinction and is
right: redacting on the way out leaves the raw frames on disk, where a
later bug, a backup or a forensic recovery can still reach them. That
module runs BEFORE persistence, at the one choke point every persisted
pixel passes through, and it earns the name privacy transformation.

Object Memory has no such choke point over the CAPTURE. It serves frames
out of `data/captures/`, which it does not own, did not write, and must
not modify -- the manifests there record `redaction: "none"` and
rewriting them would destroy the corpus every measurement in this
repository is made against. So what happens here is a filter applied on
READ, the raw frame stays exactly where it was, and the label says
`display-filter/...` rather than anything that could be read as "this
image is safe".

WHERE THAT IS NO LONGER THE WHOLE STORY.

Since `keyframes.py`, this cartridge does own one small picture per
record: a filtered crop written under the OBSERVATION root, so a 30-day
memory stops depending on a capture directory nobody promised to keep.
That store DOES have a choke point -- `KeyframeStore.write` is the only
way a pixel gets in, and it is arranged so that the bytes written are
always the filter's output -- and it is still not called a privacy
transformation, for the same reason as everything else here: YuNet's
blind spots do not close because a file was written instead of served.

The consequence for this module is one asymmetry, in `render`. A CROP
prefers the owned keyframe and is served from it WITHOUT being filtered
again, because it was filtered before it reached disk, which is the
stronger of the two guarantees. A whole FRAME still comes from the
capture and is still filtered on read -- a keyframe is a crop, and there
is nothing to build a context view out of.

WHY THE CODE IS DUPLICATED RATHER THAN SHARED.

`tower/detection.py` records the rule this follows: shared code was
promoted when a THIRD consumer appeared, and until then "the duplication
was the honest answer and two boundary tests said so in their own
docstrings". There are two consumers of YuNet now. A cartridge may not
import another cartridge (`test_a_cartridge_does_not_import_another_
cartridge` is symmetric and exhaustive), and World Builder is frozen to
another lane, so promoting its file is not this lane's to do.

The constants below are World Builder's, measured by it, and they are
copied deliberately rather than tuned: two copies that disagree about a
face-detection threshold would be worse than two copies that agree. The
promotion trigger -- a third consumer -- is recorded in
`docs/agent-handoffs/OBJECT-MEMORY-MAC-HANDOFF.md`.

WHAT MAY BE CLAIMED.

Not "faces removed". YuNet has measured hard false negatives: a face
occluded more than about 60%, and a face rotated about 90 degrees in
plane. Profile and rear views are a known blind spot of this detector
class. So the label records WHAT RAN -- the detector's identity and its
threshold -- and never "redacted", "anonymised" or "privacy-safe".
Bodies, clothing, room contents, screens and any undetected face are all
still in the picture.

AND IF THE FILTER CANNOT RUN, NOTHING IS SERVED.

This is a NEW surface exposing pixels that were previously only pointed
at. A missing model means the honest answer is a refusal, not a raw
first-person frame with an apologetic header on it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Every constant below is World Builder's, measured by it at 640x360 on
# this corpus, and copied unchanged. See its module docstring for the
# rejected alternatives (scikit-image's LBP cascade at 300-524 ms and
# blind above 15 degrees of tilt; COCO person + pose keypoints at 998 ms
# and unable to see a face with no body in frame; mediapipe and
# facenet-pytorch, both rejected on dependency collisions).
DETECTOR_ID = "yunet-2023mar"
CONFIDENCE = 0.30
NMS_THRESHOLD = 0.30
TOP_K = 5000
UPSCALE = 2
# The longest side the detector is ever handed, after upscaling.
#
# `UPSCALE = 2` was measured at 640x360, where it produces 1280x720. At
# that size the filter costs ~22 ms. Applied blindly to a larger frame it
# is quadratic: a 4000x4000 image upscaled 2x holds the shared lock for
# **2.18 seconds**, and every other request for a picture waits behind
# it. That is not hypothetical -- raising capture resolution is the one
# change this cartridge's own roadmap entry recommends.
#
# So the upscale is a TARGET rather than a constant: whatever factor gets
# the long side to 1280, capped at 2 and never below 1. At the corpus's
# 360x640 that is exactly 2, so nothing measured here changes; at 720x1280
# it is 1, and the work stays bounded.
TARGET_LONG_SIDE = 1280
HEAD_DILATION = 1.6
# Solid fill, not blur: blur is partially invertible, and it is not
# cheaper.
FILL_VALUE = 0
JPEG_QUALITY = 90

DEFAULT_MODEL_PATH = Path("models") / "face_detection_yunet_2023mar.onnx"

# What is served, said as what RAN rather than as what was achieved.
FILTER_PREFIX = "display-filter"

# How much context is included around a box in a crop. The same figure
# the producer uses when it hands a crop to a verifier, and for the same
# reason: a tight crop of a 3%-of-frame object is unreadable, and the
# surroundings are most of what makes it recognisable.
CROP_PADDING = 0.35

# Which store the served bytes came out of, and therefore whose
# retention governs them. A value on the result rather than something a
# caller infers from the route it called, because the answer depends on
# what is on disk: the same `/crop` request is answered from the owned
# keyframe when there is one and from the capture frame when there is
# not.
SOURCE_CAPTURE = "capture-frame"
SOURCE_KEYFRAME = "object-memory-keyframe"

# Why a picture could not be served. Values a client switches on, not
# sentences it displays -- the wording belongs to whoever is speaking to
# the wearer.
NOT_FOUND = "no-such-observation"
NO_CAPTURE_ROOT = "no-capture-root-configured"
NO_FRAME_REFERENCE = "record-has-no-frame-reference"
IMAGERY_EXPIRED = "imagery-no-longer-available"
FILTER_UNAVAILABLE = "display-filter-unavailable"
UNREADABLE = "frame-unreadable"


@dataclass(frozen=True)
class Imagery:
    """A picture, or the reason there is not one.

    One type for both outcomes on purpose. A caller that had to check for
    `None` and then separately ask why would be a caller that can forget
    to ask, and "the memory is retained but the picture is gone" is a
    sentence a wearer needs to hear rather than an empty response.
    """

    image_bytes: bytes | None
    reason: str | None
    filter_label: str | None = None
    regions_filled: int = 0
    relpath: str | None = None
    # How much of the RECORD'S OWN BOX the filter covered, 0.0 to 1.0.
    #
    # This exists because it happened. On frame 2708 of the validated
    # capture -- a desk with a monitor, a lit keyboard and a red gaming
    # mouse, and no person anywhere in it -- YuNet fired twice, and one
    # of the filled regions landed squarely on the mouse. The record is
    # correct, the verifier agreed with it, and the crop served for it is
    # a black rectangle.
    #
    # The response is NOT to weaken the filter. A face-detection
    # threshold is not a picture-quality knob, and trading detection
    # sensitivity for a nicer thumbnail is exactly the trade a privacy
    # filter must never make. So the fraction is measured and REPORTED,
    # and a client that knows the subject was covered can say so, or show
    # the context frame instead of the crop.
    subject_obscured: float = 0.0
    # Which store these bytes came out of, or None when there are none.
    #
    # It exists so the payload can report `imagery_retention` truthfully
    # instead of asserting one answer for every response. A crop served
    # from the owned keyframe is governed by THIS cartridge's retention
    # -- it expires with the record and `purge()` deletes it -- while a
    # frame served out of `data/captures/` is governed by capture-side
    # retention this cartridge neither sets nor enforces. Those are
    # different promises, and a client caching or captioning a picture
    # deserves to know which one it is holding.
    source: str | None = None

    @property
    def available(self) -> bool:
        return self.image_bytes is not None


def detector_scale(height: int, width: int) -> float:
    """The factor an image is resized by before detection.

    A CAP, not a floor: whatever puts the long side at
    `TARGET_LONG_SIDE`, never more than `UPSCALE`. A frame LARGER than
    the target is scaled DOWN, and that is the only way the work under
    the shared lock is actually bounded -- merely refusing to upscale a
    4000x4000 frame still leaves a 4000x4000 detection holding it, which
    a reviewer measured at 2.18 seconds while every other request for a
    picture waited.

    It costs small faces in a very large frame. A very large frame is
    also not something this cartridge can receive: DAT caps capture at
    720x1280, which lands exactly on the target, and at the corpus's
    360x640 the factor is exactly the 2 every constant here was measured
    at.

    A free function so the decision can be asserted without a model.
    """
    return min(float(UPSCALE), TARGET_LONG_SIDE / max(height, width, 1))


def model_path() -> Path | None:
    """Where the detector weights are, or None.

    `TOWER_FACE_REDACTION_MODEL` overrides -- the same variable World
    Builder reads, deliberately, so an operator who points one at a
    vendored model has pointed both.
    """
    override = os.environ.get("TOWER_FACE_REDACTION_MODEL")
    if override:
        candidate = Path(override.strip())
        return candidate if candidate.exists() else None
    return DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else None


class FaceFilter:
    """Fills detected face regions in an image on its way to a client.

    Constructed once and reused. Holds one detector; a frame-size change
    re-targets it rather than rebuilding it, because a capture's
    resolution can change mid-stream.

    SERIALISED, AND THAT IS NOT A PERFORMANCE CHOICE.

    One instance lives on `app.state` and the routes that use it are
    declared sync `def` -- deliberately, so a blocking read stays off the
    event loop -- which means FastAPI runs them CONCURRENTLY in its
    threadpool. `cv2.FaceDetectorYN` holds mutable inference state and is
    not thread-safe, and `_size` here is mutable too.

    Measured, over HTTP, against the real app: eight concurrent clients
    asking for the same frame, and **171 of 200 responses came back
    200 OK reporting `regions_filled: 0`** on a frame that serially
    always yields one filled region. Others reported 106, 24 and 23
    regions -- another request's detections, painted onto this one's
    image. Nothing raised. It failed OPEN: those responses were
    unfiltered first-person frames with a label saying they had been
    filtered.

    The constants in this file were copied from
    `tower/world_builder/redaction.py`, which builds one redactor per
    session on one thread. The code came across; the concurrency context
    did not. A lock costs ~27 ms of serialisation on a request that is
    already doing a JPEG decode, a detection and a re-encode, and it is
    the only thing standing between a shared detector and a wearer's
    unfiltered frame.
    """

    def __init__(self, path=None) -> None:
        # An explicitly supplied path is checked too, not just the
        # default. Trusting it produced a redactor that reported itself
        # AVAILABLE and then failed on every frame, in the module this
        # one is copied from.
        #
        # A BLANK path means "no model", explicitly. `Path("")` is
        # `Path(".")`, and `Path(".").exists()` is True -- so a filter
        # constructed with `path=""` to be deliberately unavailable
        # reported itself AVAILABLE, and only refused because
        # `cv2.FaceDetectorYN.create(".")` happened to raise. A refusal
        # that works by accident is a refusal that stops working when the
        # accident does.
        if path is not None and not str(path).strip():
            candidate = None
        else:
            candidate = Path(path) if path is not None else model_path()
        if candidate is not None and not candidate.exists():
            candidate = None
        # Non-reentrant: `apply` is the only public method that touches
        # the detector, and it never calls itself.
        self._lock = threading.Lock()
        self._path = candidate
        self._detector = None
        self._size = None
        self._unavailable_reason: str | None = None
        if self._path is None:
            self._unavailable_reason = (
                "no face-detection model is available on this Tower, so no "
                "frame will be served; set TOWER_FACE_REDACTION_MODEL or "
                f"vendor {DEFAULT_MODEL_PATH.as_posix()}"
            )

    @property
    def available(self) -> bool:
        return self._unavailable_reason is None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    @property
    def label(self) -> str:
        """What a served image may say about itself.

        Names the detector and its threshold rather than asserting an
        outcome, and prefixes it with `display-filter` so nothing here
        can be mistaken for the privacy transformation World Builder
        applies before persistence.
        """
        if not self.available:
            return FILTER_PREFIX + "/none"
        return f"{FILTER_PREFIX}/{DETECTOR_ID}@{CONFIDENCE:.2f}"

    def apply(self, image) -> tuple[object, list]:
        """Fill every detected face in a decoded image. Never raises.

        Returns the image and the PIXEL BOXES that were filled, rather
        than a count. The caller needs to know WHERE, not only how many:
        a filled region that lands on the object a record is about turns
        a useful crop into a black rectangle, and that has to be
        reportable.
        """
        with self._lock:
            boxes = self._detect(image)
        if not boxes:
            return image, []
        height, width = image.shape[:2]
        filled = []
        for x, y, w, h in boxes:
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(width, int(x + w)), min(height, int(y + h))
            if x1 > x0 and y1 > y0:
                image[y0:y1, x0:x1] = FILL_VALUE
                filled.append((x0, y0, x1, y1))
        return image, filled

    def _detect(self, image) -> list:
        import cv2

        height, width = image.shape[:2]
        upscale = detector_scale(height, width)
        scaled = cv2.resize(
            image,
            (max(1, int(width * upscale)), max(1, int(height * upscale))),
            interpolation=cv2.INTER_CUBIC,
        )
        size = (scaled.shape[1], scaled.shape[0])
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(self._path), "", size, CONFIDENCE, NMS_THRESHOLD, TOP_K
            )
            self._size = size
        elif size != self._size:
            self._detector.setInputSize(size)
            self._size = size

        _, faces = self._detector.detect(scaled)
        if faces is None:
            return []

        boxes = []
        for face in faces:
            x, y, w, h = (float(value) / upscale for value in face[:4])
            # Dilate about the centre: a face box is not a head.
            cx, cy = x + w / 2.0, y + h / 2.0
            w *= HEAD_DILATION
            h *= HEAD_DILATION
            boxes.append((cx - w / 2.0, cy - h / 2.0, w, h))
        return boxes


# A capture is not a walk.
#
# iOS reconnects about half a second after a WiFi hiccup and re-sends
# `stream_start`; `CaptureRecorder` mints a NEW capture directory and its
# manifest names the old one in `continues_capture`; and
# `CaptureFollower` walks the producer straight into it without missing a
# frame. The producer's `session_id`, though, was fixed once at spawn --
# it is `--follow-capture <dir>`'s basename and nothing updates it -- so a
# sighting that opened before the hiccup and had its STRONGEST look after
# it carries the OLD capture's id beside the NEW capture's frame.
#
# Measured against this host's own store on 2026-08-29, over 116 records
# and 71 capture directories: 90 resolve inside their own capture and
# **26 do not** -- 22.4% of every memory in the store. Every one of those
# 26 frames is on disk, further along the same walk. Nothing had expired;
# this repository has no capture pruner at all, so nothing CAN expire.
# The route was answering 410 "the memory is kept, the picture is gone"
# about pictures that were sitting right there.
#
# (An earlier note here said 74 records, 70 and 4. That was the store as
# it stood a few hours before, and it was left stale for long enough for
# a reviewer to catch it -- so the numbers above are the ones the shipped
# code was re-run against, not the ones first written down.)
#
# So resolution follows the lineage, not the single directory.
#
# HOW LONG A LINEAGE ACTUALLY IS, BECAUSE THE FIRST GUESS WAS WRONG.
#
# This started with a fixed `MAX_LINEAGE_STEPS = 8`, on the strength of
# "reconnect chains of five are in this corpus". Then the constant was
# checked against the corpus instead of remembered: of 32 lineage roots on
# this host, one walk is **eighteen captures long** -- 05320786 -> 664f2867
# -> ... -> 2fd9da90, covering source frames 1 to 8,690 with seventeen
# reconnects in a single sitting. A bound of eight silently truncated it,
# and four records in that walk were still answering 410 about frames that
# were on disk, which is the exact defect this whole function exists to
# remove.
#
# A fixed constant was the wrong SHAPE, not merely the wrong number. What
# actually terminates this walk is the `seen` set: `_successors` builds a
# map over directories that exist, each step consumes one edge, and an
# edge cannot be consumed twice. A manifest that names itself -- from a
# restore, a merge, or a future writer -- is one edge to a node already
# seen, and stops on the first step. A cycle of any length stops likewise.
# Cost is one dict lookup and at most two `exists()` calls per step, and
# even the eighteen-long walk is under a millisecond.
#
# So there is no constant here any more, and the thing that used to need
# one is asserted directly in `test_object_memory_imagery.py`.


def _successors(root: Path) -> dict[str, str]:
    """`{predecessor_id: successor_id}` for every capture under `root`.

    One directory listing and one small manifest read per capture,
    computed ONLY when a frame was not found where the record says it is.
    The ordinary case -- a record whose capture never got interrupted --
    never reaches this function, which is why it is not cached: a cache
    here would have to be invalidated by a capture that opens while the
    Tower is running, which is exactly when it would be wrong.
    """
    index: dict[str, str] = {}
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return index
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
            manifest = json.loads(
                (entry / "capture.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        predecessor = manifest.get("continues_capture")
        # First writer wins. Two captures claiming one predecessor is not
        # a shape the recorder can produce, and picking arbitrarily
        # between them would make this function's answer depend on
        # filesystem ordering.
        if isinstance(predecessor, str) and predecessor not in index:
            index[predecessor] = entry.name
    return index


def _lineage(root: Path, session_id: str) -> list[str]:
    """The capture ids one walk was recorded into, oldest reference first.

    Always starts with `session_id` itself, so a record whose capture was
    never interrupted costs nothing beyond the `exists()` it already paid.
    Only SUCCESSORS are followed: a producer's `session_id` is the capture
    it was spawned on, and a follower only ever moves forward.

    Measured on this host: 32 lineage roots, and the longest walk is
    eighteen captures.
    """
    chain = [session_id]
    index = _successors(root)
    seen = {session_id}
    current = session_id
    # Bounded by `seen`, and by nothing else. See the comment above
    # `_successors` for why a fixed step count was the wrong shape: this
    # loop cannot run longer than there are captures on disk, because each
    # iteration consumes one edge of a finite map and no node is entered
    # twice.
    while True:
        nxt = index.get(current)
        if nxt is None or nxt in seen:
            return chain
        chain.append(nxt)
        seen.add(nxt)
        current = nxt


def _frame_in(root: Path, capture_id: str, observation) -> Path | None:
    """The picture inside ONE capture directory, or None.

    Two ways to find it, in order of how much they trust:

    1. `best_relpath`, written onto records since sightings existed. It
       is what the producer actually read, so it cannot be wrong about
       the naming convention.
    2. The convention itself, `frames/<source_seq:08d>.jpg`, for the 64
       records written before that field existed. Coupled to
       `tower/capture.py`, and that coupling is the reason it is second
       rather than first.
    """
    directory = root / capture_id

    relpath = observation.best_relpath
    if relpath:
        candidate = _contained(root, directory / relpath)
        if candidate is not None and candidate.exists():
            return candidate

    frame_seq = (
        observation.best_frame_seq
        if observation.best_frame_seq is not None
        else observation.frame_seq
    )
    if frame_seq is None:
        return None
    candidate = _contained(root, directory / "frames" / f"{int(frame_seq):08d}.jpg")
    return candidate if candidate is not None and candidate.exists() else None


def frame_path(capture_root, observation) -> Path | None:
    """Where the picture behind a record is, or None if it is not there.

    Searches the record's own capture first and then the reconnect
    lineage after it -- see `_successors` for the measurement that made the
    second half necessary, and for why it carries no step limit.

    Returns None for a record with no frame reference at all, and for a
    walk whose directories are no longer on disk. The caller turns that
    into `IMAGERY_EXPIRED`, and `capture_lineage_present` is what lets it
    tell "the recording was deleted" apart from "the recording is here and
    this frame is not" in the log without changing what a client is told.
    """
    if capture_root is None:
        return None
    session_id = observation.session_id
    if session_id is None:
        return None
    root = Path(capture_root) / "captures"

    # The overwhelmingly common case, and it must stay free of the
    # directory scan `_lineage` pays for.
    found = _frame_in(root, session_id, observation)
    if found is not None:
        return found

    for capture_id in _lineage(root, session_id)[1:]:
        found = _frame_in(root, capture_id, observation)
        if found is not None:
            logger.info(
                "[Tower][ObjectMemory] frame for a %s record resolved in %s, "
                "the capture that continues %s",
                observation.object_class,
                capture_id,
                session_id,
            )
            return found
    return None


def capture_lineage_present(capture_root, observation) -> bool:
    """Whether any directory of this record's walk is still on disk.

    Not on the wire, and deliberately: a client is told the same thing
    either way -- the memory is kept and the picture is gone -- because
    there is nothing a wearer can do differently about the two. It exists
    so the LOG can distinguish a recording that was deleted from a pointer
    that does not resolve into a recording that is still there, which is
    the defect this module shipped with and could not see.
    """
    if capture_root is None:
        return False
    session_id = observation.session_id
    if session_id is None:
        return False
    root = Path(capture_root) / "captures"
    # Contained, like every other path this module builds out of a
    # record. `session_id` comes off a JSONL file and is used to BUILD A
    # PATH; `_frame_in` runs it through `_contained` and this did not,
    # which is an inconsistency a reviewer was right to call out even
    # though the value here only steers a log line.
    own = _contained(root, root / session_id)
    if own is None:
        return False
    # The record's own directory first, and it answers on its own in the
    # overwhelmingly common case. This runs immediately after a
    # `frame_path` that already paid for one directory scan, and asking
    # `_lineage` again would pay for a second one to answer a question
    # the first `is_dir` has usually already settled.
    if own.is_dir():
        return True
    return any(
        (root / capture_id).is_dir() for capture_id in _lineage(root, session_id)
    )


def _contained(root: Path, candidate: Path) -> Path | None:
    """The path, if it really is under the capture root. Otherwise None.

    `session_id` and `best_relpath` both come off a record in a JSONL
    file, and both are used to BUILD A PATH. Nothing in the pipeline that
    writes them can produce a `..` -- `CaptureRecorder` mints the id and
    names every frame `frames/<seq:08d>.jpg` -- so this is defence in
    depth rather than a fix for a reachable bug.

    It is cheap defence in depth against a real class of problem, though.
    A store file is a plain text file on disk; an operator restoring one
    from a backup, a future producer, or a merge of two stores could all
    introduce a path this route would happily follow out of the capture
    tree and serve over HTTP. Resolving and comparing is four lines.

    `resolve()` on both sides, so a symlink cannot be used to step out
    either.
    """
    try:
        resolved = candidate.resolve()
        base = root.resolve()
    except OSError:
        return None
    return resolved if resolved.is_relative_to(base) else None


def _bounding_box(observation):
    """The box to crop, preferring the strongest look.

    `bounding_box` describes the FIRST frame of the sighting and
    `best_bounding_box` the strongest; the two go with different frames,
    and `frame_path` prefers the strongest, so this must agree with it or
    a crop would cut the wrong part of the right picture.
    """
    if observation.best_bounding_box is not None and (
        observation.best_frame_seq is not None or observation.best_relpath
    ):
        return observation.best_bounding_box
    return observation.bounding_box


def _from_keyframe(keyframes, observation) -> Imagery | None:
    """The crop this cartridge owns, or None if it does not own one.

    NOT RE-FILTERED, and that is the point rather than an optimisation.
    These bytes were filtered BEFORE they were written -- `KeyframeStore.
    write` has no path that puts an unfiltered crop on disk, and `read`
    refuses an image whose sidecar is missing, so the sidecar's presence
    is what makes the claim checkable. Filtering again at read time would
    be strictly weaker: it would run the detector on an image whose faces
    have already been filled, find nothing, and report a fresh label that
    describes this read rather than what actually protected the file.

    So the label, the region count and the obscured fraction all come out
    of the sidecar. They describe the run that produced the file, which
    is the honest thing for them to describe.

    Returns None -- not a refusal -- when there is no keyframe, because
    "no keyframe" is not a failure. It is the ordinary state of every
    record written before this existed, and the caller's answer is to
    fall back to the capture frame.
    """
    if keyframes is None:
        return None
    try:
        found = keyframes.read(observation.observation_id)
    except Exception:  # noqa: BLE001
        # A keyframe store that misbehaves must not take the capture-side
        # path down with it. The fallback here is another FILTERED
        # picture, not a raw one, so degrading to it costs nothing but
        # the ownership.
        logger.exception(
            "[Tower][ObjectMemory] the keyframe store failed for a %s "
            "record; falling back to the capture frame",
            observation.object_class,
        )
        return None
    if found is None:
        return None
    image_bytes, sidecar = found
    return Imagery(
        image_bytes=image_bytes,
        reason=None,
        filter_label=sidecar.get("filter_label"),
        regions_filled=_as_int(sidecar.get("regions_filled")),
        # The frame this crop was taken from, named for the same reason
        # the capture-side path names one: a diagnostic can say which
        # picture without a client learning a path on the Tower. It
        # survives the frame itself, which is the whole point.
        relpath=sidecar.get("source_relpath"),
        subject_obscured=_as_float(sidecar.get("subject_obscured")),
        source=SOURCE_KEYFRAME,
    )


def _as_int(value) -> int:
    """A sidecar field, as a number, or 0. A sidecar is a file on disk."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _as_float(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def render(
    capture_root, observation, face_filter, *, crop: bool, keyframes=None
) -> Imagery:
    """The picture behind a record, filtered, or the reason there is none.

    Never raises, and never returns an unfiltered frame. The refusal
    reasons are values rather than sentences so a client can render them
    in its own words, and so a test can assert on which one happened
    rather than on how it was phrased.

    TWO SOURCES, AND ONLY THE CROP HAS TWO.

    A crop prefers the keyframe this cartridge owns and falls back to the
    capture. A whole frame has only ever one source -- the capture --
    because a keyframe IS a crop and there is nothing to synthesise a
    context view out of. That asymmetry is the product shape: the owned
    picture is the durable one, and the capture frame remains the source
    of the wider view for as long as capture-side retention keeps it.
    """
    import cv2

    # BEFORE the filter and capture-root checks, deliberately.
    #
    # An owned keyframe needs neither. It does not need a capture root
    # because it is not in the capture tree -- which is the entire reason
    # it exists, and what lets a memory keep its picture after the whole
    # recording has been deleted. And it does not need a working face
    # filter because it was filtered before it was written; refusing to
    # serve it on a Tower whose ONNX weights went missing would be
    # withholding a picture on the strength of a check that has already
    # been passed, more thoroughly, at write time.
    if crop:
        owned = _from_keyframe(keyframes, observation)
        if owned is not None:
            return owned

    if not face_filter.available:
        return Imagery(None, FILTER_UNAVAILABLE)
    if capture_root is None:
        return Imagery(None, NO_CAPTURE_ROOT)

    path = frame_path(capture_root, observation)
    if path is None:
        if observation.session_id is None and observation.frame_seq is None:
            return Imagery(None, NO_FRAME_REFERENCE)
        # The pointer is intact and the picture is not, and the wearer is
        # told the one thing that is true either way: the memory is here
        # and the picture is not.
        #
        # The LOG says which, because the two have opposite fixes and this
        # module shipped unable to tell them apart. A walk whose
        # directories are gone is retention (or a human with rm) doing its
        # job. A walk that is STILL ON DISK and does not contain the frame
        # a record names is a defect in whoever wrote the pointer, and it
        # spent two days being reported as expiry.
        if capture_lineage_present(capture_root, observation):
            logger.warning(
                "[Tower][ObjectMemory] a %s record points at %s in capture %s "
                "and the recording is still on disk without it; reporting the "
                "memory kept and the picture gone, but this is a POINTER "
                "defect rather than retention",
                observation.object_class,
                observation.best_relpath or observation.best_frame_seq,
                observation.session_id,
            )
        else:
            logger.info(
                "[Tower][ObjectMemory] the recording behind a %s record is no "
                "longer on disk; the memory is kept and the picture is gone",
                observation.object_class,
            )
        return Imagery(None, IMAGERY_EXPIRED)

    try:
        image = cv2.imread(str(path))
    except Exception:  # noqa: BLE001
        logger.exception("[Tower][ObjectMemory] could not read %s", path)
        image = None
    if image is None:
        return Imagery(None, UNREADABLE)

    try:
        filtered, filled = face_filter.apply(image)
    except Exception:  # noqa: BLE001
        # A filter that failed has said nothing about this frame, and an
        # unfiltered first-person frame is not the fallback. Refusing
        # costs a picture; serving would cost the promise.
        logger.exception(
            "[Tower][ObjectMemory] the display filter failed on %s; serving "
            "nothing rather than an unfiltered frame",
            path,
        )
        return Imagery(None, FILTER_UNAVAILABLE)

    # Measured on the FULL frame, before any crop, because the box and
    # the filled regions are both in full-frame coordinates and comparing
    # them after a crop would compare two different origins.
    obscured = _obscured_fraction(
        filtered.shape, _bounding_box(observation), filled
    )

    if crop:
        filtered = _cropped(filtered, _bounding_box(observation))
        if filtered is None:
            return Imagery(None, NO_FRAME_REFERENCE)

    ok, encoded = cv2.imencode(
        ".jpg", filtered, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        return Imagery(None, UNREADABLE)
    return Imagery(
        image_bytes=encoded.tobytes(),
        reason=None,
        filter_label=face_filter.label,
        regions_filled=len(filled),
        # Relative to the capture directory, so a diagnostic can name the
        # frame without a client learning an absolute path on the Tower.
        relpath=path.name,
        subject_obscured=obscured,
        source=SOURCE_CAPTURE,
    )


def _obscured_fraction(shape, box, filled) -> float:
    """How much of the record's own box the filter covered.

    Approximated as the largest single overlap rather than the union of
    all of them. Two filled regions overlapping each other would be
    double-counted by a naive sum, and a fraction above 1.0 would be
    worse than an under-estimate -- this figure exists so a client can
    say "the subject was covered", and it should never be able to say
    something impossible.
    """
    if box is None or not filled:
        return 0.0
    height, width = shape[:2]
    bx0, by0 = box[0] * width, box[1] * height
    bx1, by1 = box[2] * width, box[3] * height
    area = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    if area <= 0:
        return 0.0
    largest = 0.0
    for x0, y0, x1, y1 in filled:
        overlap_w = max(0.0, min(bx1, x1) - max(bx0, x0))
        overlap_h = max(0.0, min(by1, y1) - max(by0, y0))
        largest = max(largest, overlap_w * overlap_h)
    return round(min(1.0, largest / area), 4)


def _cropped(image, box):
    """The padded region a normalised box names, or None.

    The box is a fraction of the frame because a stored pixel box would
    mean different things at different capture resolutions and nothing
    would say which. It becomes pixels here, once, against the image it
    is actually being applied to.
    """
    if box is None:
        return None
    height, width = image.shape[:2]
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
    return image[y1:y2, x1:x2]
