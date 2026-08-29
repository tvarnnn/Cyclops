"""A sighting: one category, in view, from when it appeared to when it left.

WHAT THIS REPLACES, AND WHY.

The first slice suppressed repeats with a 30-second resample window: the
first detection of a class was recorded, and every detection of that
class for the next 30 seconds was discarded. It was explicitly labelled a
starting point "to be revisited against measured retrieval behavior", and
the measurement now exists.

Grouping the same corpus into temporally contiguous runs instead --
18,821 frames, every detection at score >= 0.5, a run broken by a gap of
more than 3 seconds -- gives **763 sightings, 499 of them at least three
frames long, 404 of those excluding `person`**, over 28 classes rather
than two.

Of those 499, the classes this cartridge may actually write account for
**211**. The corpus is 1,942 seconds of recording, so that is **one
memory every 9.2 seconds of walking** -- about 380 an hour, which is a
shape a wearer could scroll and a store that stays small for months.

The 30-second window cannot express any of that. It is an interval with
no relationship to what the camera did: an object glanced at twice in one
second produces one record, and an object watched continuously for four
minutes produces eight, and neither number means anything. A sighting is
an EVENT -- "this came into view at 14:03:12 and stayed for 4.4 seconds
across 29 frames" -- and every field on it is something that happened.

WHY THREE FRAMES.

Of the 763 sightings, 264 are one or two frames long -- 35%. Those are
flickers: a class that fired once and never again. Writing them would
make a third of the memory noise. Requiring three frames costs two
inter-frame gaps, about **170-210 ms** depending on whether the
denominator is the measured 83.5 ms delivered interval or the corpus's
own 9.7 frames a second averaged across whole captures. It is the only
threshold here chosen from a distribution rather than from taste.

The record is still stamped with the FIRST frame of the sighting, so
`observed_at` still means "when it came into view" and a session killed
mid-sighting loses at most those first 250 ms.

ONE SIGHTING PER CLASS AT A TIME, DELIBERATELY.

Two laptops in one frame are one `laptop` sighting. The contract this
cartridge serves says `identity: "category-not-instance"` and means it:
a record is evidence that the CATEGORY was in view, and splitting one
category into two concurrent sightings would be the first step towards
implying otherwise without any of the evidence that would justify it.
"""

from dataclasses import dataclass, field, replace

# A sighting survives this long a dropout before it is treated as
# finished. Three seconds is roughly thirty delivered frames, which
# covers a head turn away and back -- the commonest reason a class
# vanishes for a moment on head-mounted footage. Shorter, and one glance
# aside becomes two memories; longer, and two genuinely separate visits
# to the same object merge into one.
GAP_SECONDS = 3.0

# See the module docstring. 264 of 763 sightings in the corpus are
# shorter than this.
MIN_FRAMES = 3


@dataclass(frozen=True)
class Look:
    """One frame's view of an object: what was seen, where, and how well."""

    score: float
    # Normalised to the frame, not pixels: a stored box must still mean
    # the same thing at a different capture resolution, and the frame
    # sizes this corpus carries are already not guaranteed to be stable.
    box: tuple[float, float, float, float]
    at: float
    frame_seq: int | None
    # Where the pixels are, when they are still there. `relpath` points
    # inside the capture directory; it is the only thing that makes a
    # representative crop retrievable later, and it is why the crop can
    # be served without this cartridge storing a single pixel of its own.
    relpath: str | None = None
    width: int | None = None
    height: int | None = None

    @property
    def area_fraction(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Sighting:
    """One category's continuous presence, accumulating as frames arrive.

    Mutable, unlike everything this cartridge persists: it is working
    state that exists only while a class is in view. What gets written
    from it is an immutable record.
    """

    object_class: str
    first: Look
    best: Look
    last: Look
    frame_count: int = 1
    # Whether a record for this sighting has already reached disk. The
    # producer sets it; the sighting only remembers it, so that a
    # sighting which crosses MIN_FRAMES a second time (it cannot) or is
    # re-closed (it can, at session end) does not write twice.
    recorded: bool = False
    # Set once the sighting has stopped growing. A closed sighting is
    # complete evidence; an open one is a claim still being made.
    closed: bool = False
    # Filled by whatever agreed the label is right, if anything did.
    # Deliberately a free-form dict rather than a type: this module must
    # keep working when no verifier exists, and typing an absence is how
    # a null becomes a required field.
    verdict: dict | None = field(default=None)
    # Whether this sighting was ever seen while a class it is a PART of
    # was also in view.
    #
    # Recorded on the sighting rather than recomputed, because by the
    # time a sighting is settled it has been removed from the tracker and
    # so has everything that was open alongside it -- so the check would
    # run against an empty set and the suppression would evaporate at the
    # end of every sighting, writing the duplicate it exists to prevent.
    # A reviewer reproduced exactly that.
    #
    # A latch, not a live check: "this was only in view because its whole
    # was" is a fact about what happened, and it does not stop being true
    # when the whole leaves the frame.
    suppressed_as_part: bool = False
    # Whether a second opinion has already been ASKED FOR. Distinct from
    # `verdict`, which is whether one has arrived: between the two the
    # sighting is in flight, and without this flag every frame in that
    # window would queue the same crop again -- which is how a funnel
    # becomes a fan.
    verification_requested: bool = False
    # The strongest look's pixels, kept only while the sighting is open.
    #
    # Held here rather than re-cropped at verification time because the
    # best frame is usually NOT the current one: cropping the current
    # frame with the best frame's box would hand a model a picture of
    # something else entirely.
    #
    # It is the only imagery this cartridge holds IN MEMORY, and it is
    # dropped when the sighting closes. It used to be true that it never
    # reached disk, and since `keyframes.py` it is not: when a keyframe
    # store is configured, `ObjectMemoryEngine._settle` hands this crop
    # to it -- once, at the end of the sighting, through a face filter
    # that must have run -- and the resulting JPEG becomes the one piece
    # of imagery Object Memory's own retention governs. The crop itself
    # still dies with the sighting; what outlives it is a filtered,
    # downscaled copy under the observation root.
    #
    # It is also made for MORE sightings than it used to be. It was built
    # only for `verify`-tier classes with a verifier attached, which
    # meant `laptop` and `cell phone` -- the two classes this Tower
    # actually records without one -- never held a crop at all, and so
    # could never have had a keyframe. It is now made for every sighting
    # of a persistable class. The bound is unchanged: one crop per open
    # sighting, replaced only by a stronger look.
    best_crop: object = field(default=None, repr=False)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.last.at - self.first.at)

    @property
    def mature(self) -> bool:
        """Long enough to be worth a record. See MIN_FRAMES."""
        return self.frame_count >= MIN_FRAMES

    def extend(self, look: Look, crop=None) -> None:
        self.frame_count += 1
        self.last = look
        if look.score > self.best.score:
            self.best = look
            # Replaced together with the look it belongs to, so the two
            # can never describe different frames.
            self.best_crop = crop

    def forget_imagery(self) -> None:
        """Drop the held crop. Called when the sighting can no longer use it."""
        self.best_crop = None


class SightingTracker:
    """Turns a stream of per-frame detections into sightings.

    Holds no images and no model. It is fed a class, a score and a box,
    and it answers with the sighting that detection belongs to -- which
    is the whole of the temporal reasoning this cartridge does.

    NOT a multi-object tracker, and the distinction is measured rather
    than stylistic. `EgoTracks` (Meta, 5,708 egocentric videos, 602.9
    hours) reports that off-the-shelf trackers score 20-37 average
    overlap on first-person footage, with recall far below precision --
    they give up rather than misfire. A tracker's motion model assumes
    the camera is roughly still and the object moves; on head-mounted
    footage the opposite is true. Associating by CLASS over time asks
    much less and cannot be wrong in the way a broken motion model is.
    """

    def __init__(
        self, *, gap_seconds: float = GAP_SECONDS, min_frames: int = MIN_FRAMES
    ) -> None:
        self._gap_seconds = gap_seconds
        self._min_frames = min_frames
        self._open: dict[str, Sighting] = {}

    @property
    def open_sightings(self) -> dict[str, Sighting]:
        return dict(self._open)

    def observe(
        self, object_class: str, look: Look, crop=None
    ) -> tuple[Sighting, bool]:
        """Fold one detection in. Returns the sighting and whether it is new.

        The caller decides what to do with a new sighting; this only
        decides whether it IS one. Keeping that split is what lets the
        producer write on maturity, the store enforce its own allowlist,
        and a test drive the grouping with no store at all.
        """
        existing = self._open.get(object_class)
        if existing is not None and (look.at - existing.last.at) <= self._gap_seconds:
            existing.extend(look, crop)
            return existing, False
        sighting = Sighting(
            object_class=object_class,
            first=look,
            best=look,
            last=look,
            best_crop=crop,
        )
        self._open[object_class] = sighting
        return sighting, True

    def close_stale(self, now: float) -> list[Sighting]:
        """Every sighting whose class has not been seen for the gap window.

        Driven by the caller's clock rather than by a timer, so a replay
        of a recorded capture closes sightings at the times the RECORDING
        implies and not at the times the replay happens to run. A
        producer that only closed sightings when a class reappeared would
        hold every sighting of the walk open until the walk ended.
        """
        stale = []
        for object_class, sighting in list(self._open.items()):
            if (now - sighting.last.at) > self._gap_seconds:
                sighting.closed = True
                stale.append(sighting)
                del self._open[object_class]
        return stale

    def close_all(self) -> list[Sighting]:
        """End of session. Everything still open is as complete as it will get."""
        remaining = list(self._open.values())
        for sighting in remaining:
            sighting.closed = True
        self._open.clear()
        return remaining


def summarise(sighting: Sighting) -> dict:
    """The sighting as plain data, for a report or a test.

    A function rather than a method so `Sighting` stays a record of what
    happened rather than something with opinions about how it is shown.
    """
    return {
        "object_class": sighting.object_class,
        "first_seen": sighting.first.at,
        "last_seen": sighting.last.at,
        "duration_seconds": round(sighting.duration_seconds, 3),
        "frame_count": sighting.frame_count,
        "best_score": sighting.best.score,
        "first_score": sighting.first.score,
        "best_area_fraction": round(sighting.best.area_fraction, 6),
        "recorded": sighting.recorded,
        "closed": sighting.closed,
    }


def rebased(look: Look, *, width: int, height: int) -> Look:
    """The same look with its box expressed as fractions of the frame.

    Detections arrive in pixels because that is what a detector reports
    about the image it was given. Only the thing that writes a durable
    record has to care that a stored box must still mean something at a
    different capture resolution -- so the conversion happens once, here,
    on the way in.
    """
    x1, y1, x2, y2 = look.box
    return replace(
        look,
        box=(x1 / width, y1 / height, x2 / width, y2 / height),
        width=width,
        height=height,
    )
