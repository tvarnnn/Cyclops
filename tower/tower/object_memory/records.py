import hashlib
from dataclasses import dataclass

# Confidence moved to tower/confidence.py when World Builder became a
# second consumer. Re-exported here so this module's public surface is
# unchanged and no caller has to care where it lives.
from tower.confidence import (  # noqa: F401
    LOW_CONFIDENCE_MAX,
    MEDIUM_CONFIDENCE_MAX,
    Confidence,
)


# --- What the privacy tags on a record mean -------------------------------
#
# DERIVED_ONLY is a claim about the record's CONTENT: a class label, a
# score and a box, with no pixels, no crop and no embedding of the frame.
# That much was always true.
#
# It was read as a claim about REACH, and there it was false.
# `session_id` + `frame_seq` is an exact pointer into
# `data/captures/<session_id>/frames/`, where the JPEG this record was
# derived from is still sitting. Object Memory's retention governs THIS
# store and nothing else: purging every observation here leaves the
# imagery untouched, and any record that survives resolves straight back
# to a frame.
#
# The provenance is worth keeping -- it is what makes a record checkable
# against the footage instead of merely believable. So the fix is to stop
# the tag overclaiming rather than to drop the fields: a record that
# carries a frame pointer says FRAME_REFERENCED as well, so a reader can
# see that its privacy footprint is bounded by capture-side retention and
# not by ours.
DERIVED_ONLY = "derived-only"
FRAME_REFERENCED = "frame-referenced"


def observation_id_for(
    session_id: str | None, object_class: str, observed_at: float
) -> str:
    """A stable handle for one record, derived rather than minted.

    Needed because a record has to be ADDRESSABLE: `GET
    /object-memory/observations/{id}/frame` cannot exist without one, and
    "the laptop record whose observed_at is 1787806912.4471" is not a URL.

    DERIVED, not a uuid4, and that is the whole design. Sixty-four
    observations are already on disk, written before this field existed.
    A random id would have to be minted at read time -- so it would
    change on every read, and a link to a record would stop working the
    moment the Tower restarted -- or written back, which means rewriting
    a wearer's memory to add a field. Deriving it from the three values
    that already identify a sighting gives every existing record a
    permanent id it never had, with no migration and no write.

    The inputs are exactly the ones `ObservationStore.update_sighting`
    matches on, so an id and an update can never disagree about which
    record they mean.

    Truncated to 16 hex characters. This is a handle within one store,
    not a global identifier: 64 bits is far more than enough to keep a
    few thousand records apart, and a shorter string is a friendlier URL.
    """
    # An explicit separator, so ("ab", "c") and ("a", "bc") cannot hash
    # to the same handle. A pipe, because a capture id is hex and no
    # COCO label contains one -- and because a printable separator
    # survives being copied between a file, a diff and a terminal,
    # which a control character does not.
    material = " | ".join((session_id or "", object_class, repr(observed_at)))
    return hashlib.blake2b(material.encode("utf-8"), digest_size=8).hexdigest()


def privacy_tags_for(
    session_id: str | None, frame_seq: int | None
) -> tuple[str, ...]:
    """The tags a record with this provenance may honestly carry.

    Either half alone is a reach claim worth making: a session id narrows
    the imagery to one capture directory even with no frame number.
    """
    if session_id is None and frame_seq is None:
        return (DERIVED_ONLY,)
    return (DERIVED_ONLY, FRAME_REFERENCED)


@dataclass(frozen=True)
class ObjectObservation:
    """One "this category was visible at this time" record.

    Deliberately NOT a claim that the object is present now, or that it is
    a specific instance ("my keys" vs "keys"). See 07-PLATFORM-CONSTRAINTS.md
    Core Principle 3 and OBJECT-MEMORY.md's Identity vs. Category section.

    observed_at is qualified by time_basis: this slice can only know
    tower-receipt time, never on-glasses capture time (Rule 16 -- these
    must not be conflated). There is no soft-delete flag by design;
    06-PRIVACY-DATA.md requires real deletion.

    spatial_ref and external_refs are reserved-but-unused: they are carried
    so a later cross-module need does not require rewriting already-persisted
    records.

    THREE FIELDS ABOUT STRENGTH, AND WHICH TO TRUST FOR WHAT.

    detector_score is PROVENANCE: how confident the detector was in the
    frame that FIRST brought this class into view -- the one frame this
    record's observed_at, frame_seq and bounding_box all describe. It is
    what makes the record auditable, and it never moves.

    best_score is EVIDENCE: the strongest score seen while that same
    sighting stayed in view, filled in afterwards by the producer as the
    sighting continues. Also never revised downwards.

    confidence is the INTERPRETATION, and it is the field a consumer
    reads. The claim a record makes is "this category was in view", so it
    is derived from best_score -- the best evidence for that claim --
    and moves with it. A record first seen at 0.601 and then seen at
    0.962 is a HIGH-confidence record about a laptop, with 0.601 still on
    it saying how the sighting started. At the first write the two scores
    are equal, so the initial label is derived from either.

    None of the three is a calibrated probability. They are detector
    output, and an interpretation of detector output.
    """

    object_class: str
    detector_score: float | None
    confidence: Confidence
    observed_at: float
    time_basis: str
    recorded_at: float
    source: str
    module_id: str
    session_id: str | None
    frame_seq: int | None
    bounding_box: tuple[float, float, float, float] | None
    retention_tag: str
    privacy_tags: tuple[str, ...]
    spatial_ref: None
    external_refs: tuple[()]
    # Last, with defaults, so every record persisted before these fields
    # existed still constructs -- and so a reader of those records gets an
    # honest "not tracked" rather than an invented number.
    best_score: float | None = None
    # WHEN THE SIGHTING ENDED, and how much of it there was.
    #
    # `observed_at` says when the category came into view and never
    # moves. These two accumulate while it stays in view, and are what
    # turn a record from "seen at 14:03" into "seen at 14:03, for 4.4
    # seconds, across 29 frames" -- which is the difference between a
    # detection and an observation. None means the record predates
    # sighting tracking; never 0, which would claim a sighting of no
    # duration.
    last_seen_at: float | None = None
    frame_count: int | None = None
    # Which frame the representative crop comes from, and where in it.
    #
    # `frame_seq` and `bounding_box` describe the FIRST frame, because
    # that is the frame `observed_at` is about and the record must stay
    # auditable against it. This describes the BEST frame -- the
    # strongest look during the sighting -- which is the one worth
    # showing a person. They are usually different frames and the record
    # needs both.
    best_frame_seq: int | None = None
    best_relpath: str | None = None
    best_bounding_box: tuple[float, float, float, float] | None = None
    # Which policy tier admitted this record, and what (if anything)
    # agreed with the detector's label. `verification` is None when
    # nothing was asked -- a REMEMBERED class on this Tower -- and a
    # dict when something was. Never a bare bool: "a model agreed" is
    # not a claim worth carrying without saying which model and how
    # strongly.
    tier: str | None = None
    verification: dict | None = None

    def to_json_dict(self) -> dict:
        return {
            "object_class": self.object_class,
            "detector_score": self.detector_score,
            "confidence": self.confidence.value,
            "observed_at": self.observed_at,
            "time_basis": self.time_basis,
            "recorded_at": self.recorded_at,
            "source": self.source,
            "module_id": self.module_id,
            "session_id": self.session_id,
            "frame_seq": self.frame_seq,
            "bounding_box": (
                list(self.bounding_box) if self.bounding_box is not None else None
            ),
            "retention_tag": self.retention_tag,
            "privacy_tags": list(self.privacy_tags),
            "spatial_ref": self.spatial_ref,
            "external_refs": list(self.external_refs),
            "best_score": self.best_score,
            "last_seen_at": self.last_seen_at,
            "frame_count": self.frame_count,
            "best_frame_seq": self.best_frame_seq,
            "best_relpath": self.best_relpath,
            "best_bounding_box": (
                list(self.best_bounding_box)
                if self.best_bounding_box is not None
                else None
            ),
            "tier": self.tier,
            "verification": self.verification,
            "observation_id": self.observation_id,
        }

    @property
    def observation_id(self) -> str:
        """Derived on demand, never stored as a separate source of truth.

        Written into `to_json_dict` so a reader that never calls this
        still sees it, and recomputed on read rather than trusted -- a
        record whose stored id disagreed with its own fields would be a
        record two different lookups could reach differently.
        """
        return observation_id_for(self.session_id, self.object_class, self.observed_at)


def object_observation_from_json_dict(data: dict) -> ObjectObservation:
    box = data.get("bounding_box")
    return ObjectObservation(
        object_class=data["object_class"],
        detector_score=data["detector_score"],
        confidence=Confidence(data["confidence"]),
        observed_at=data["observed_at"],
        time_basis=data["time_basis"],
        recorded_at=data["recorded_at"],
        source=data["source"],
        module_id=data["module_id"],
        # Required-key access, not .get(): to_json_dict always writes
        # these keys, so a missing key means a malformed record, not a
        # None value -- matching detector_score above.
        session_id=data["session_id"],
        frame_seq=data["frame_seq"],
        bounding_box=tuple(box) if box is not None else None,
        retention_tag=data["retention_tag"],
        privacy_tags=tuple(data["privacy_tags"]),
        spatial_ref=None,
        external_refs=(),
        # .get(), unlike every required field above, and deliberately:
        # the records already on disk were written before best_score
        # existed. A required key would make _parse_observations treat
        # every one of them as a schema mismatch and skip it, deleting
        # the wearer's memory to add a field. Every field added since
        # follows the same rule, for the same reason.
        best_score=data.get("best_score"),
        last_seen_at=data.get("last_seen_at"),
        frame_count=data.get("frame_count"),
        best_frame_seq=data.get("best_frame_seq"),
        best_relpath=data.get("best_relpath"),
        best_bounding_box=(
            tuple(data["best_bounding_box"])
            if data.get("best_bounding_box") is not None
            else None
        ),
        tier=data.get("tier"),
        verification=data.get("verification"),
    )
