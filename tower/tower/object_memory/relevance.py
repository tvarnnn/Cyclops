"""What reaches the store, and what is refused, and why.

WHY A CLASS TABLE, AND WHY SIGHTINGS.

  * Classes are admitted from evidence, not from a score histogram, which
    describes the detector's opinion of ITSELF. Reading the actual crops
    (over all 18,821 real frames, versus 9,199 when the first rules were
    written) found a ceiling fan detected as `airplane` at 0.99 and as
    `scissors` at 0.93, a white door as `refrigerator` at 0.95, a phone
    in a hand as `chair` at 0.94, and a laptop keyboard as `remote` at
    0.87. The per-class evidence lives in `classes.py`.
  * The unit of memory is a SIGHTING (`sightings.py`): an event that
    happened rather than a timer that expired. A fixed resample window
    has no relationship to what the camera did: an object glanced at
    twice in a second produced one record and an object watched for four
    minutes produced eight.

FOUR GATES, IN THIS ORDER, AND THE ORDER IS LOAD-BEARING.

  1. EXCLUDED. `person`, and anything else the privacy policy names.
     Checked first so an excluded class can never occupy a slot in any
     table downstream, and so the accounting never reports a refusal
     under the wrong reason.
  2. TIER. What `classes.py` says this class is worth. A class this
     cartridge has no evidence it can read correctly is dropped here,
     before a score is looked at.
  3. SCORE. Weak detections are not evidence.
  4. MATURITY, then VERIFICATION. A sighting must last long enough to be
     real, and a `verify`-tier class must additionally have been agreed
     with by something other than the detector that proposed it.

THE PRIVACY POLICY IS NOT A MODEL'S TO OVERTURN.

Gates 1 and 2 are deterministic tables. A verifier is consulted only at
gate 4, only for classes gate 2 has already admitted, and only to CONFIRM
or REFUSE -- there is no path by which any model introduces a class the
tables did not already allow, and none by which one re-admits `person`.

A TOWER WITH NO VERIFIER BEHAVES AS THE `remembered`-ONLY TOWER.

With no verifier configured, gate 4 refuses every `verify`-tier class, so
what is written is the `remembered` tier: `laptop` and `cell phone`, the
two classes the corpus supports on the detector's word alone. That is not
a fallback, and the machinery is in place to widen it the moment
something can tell a remote from a keyboard.
"""

from dataclasses import dataclass

from tower.object_memory.classes import (
    CONTEXT,
    IGNORED,
    PERSISTABLE_CLASSES,
    PERSISTED_CLASSES,  # noqa: F401
    REMEMBERED,
    VERIFY,
    classes_in,
    is_excluded,
    tier_of,
    wholes_of,
)
from tower.object_memory.sightings import GAP_SECONDS, MIN_FRAMES

# What `decide` returns. Strings rather than an enum because their only
# consumers are a counter and a report line -- and a counter keyed by
# these names IS the report the producer prints.
RECORD = "record"
EXCLUDED = "excluded"
NOT_WHITELISTED = "not-whitelisted"
CONTEXT_ONLY = "context-only"
BELOW_MIN_SCORE = "below-min-score"
TOO_BRIEF = "too-brief"
UNVERIFIED = "unverified"
PART_OF_ANOTHER = "part-of-another-sighting"
ALREADY_RECORDED = "already-recorded"

DROP_REASONS = (
    EXCLUDED,
    NOT_WHITELISTED,
    CONTEXT_ONLY,
    BELOW_MIN_SCORE,
    TOO_BRIEF,
    UNVERIFIED,
    PART_OF_ANOTHER,
    ALREADY_RECORDED,
)


@dataclass(frozen=True)
class RelevancePolicy:
    """Thresholds for turning detections into stored observations.

    `min_score` is 0.5, and the corpus supports it: 78,546 detections at 0.15 fall to 30,727 at 0.4
    and 24,028 at 0.5, and below 0.5 the sighting structure is dominated
    by one- and two-frame flickers. It is still not a probability, and
    raising it would not make it one -- a ceiling fan scores 0.99.

    `gap_seconds` and `min_frames` come from `sightings.py` and are the
    two numbers that were chosen from a measured distribution.

    `allowed_classes` is different in kind from all three. It is not a
    threshold to tune but a decision about what this system is allowed to
    remember, and it defaults to the OUTER bound -- every class the
    tables could ever admit. What is actually written is narrower and
    depends on whether a verifier exists.
    """

    min_score: float = 0.5
    gap_seconds: float = GAP_SECONDS
    min_frames: int = MIN_FRAMES
    allowed_classes: tuple[str, ...] = PERSISTABLE_CLASSES
    # Whether anything is available to second-guess the detector. False
    # is the default and the honest one: a Tower with no semantic
    # model cannot tell a remote from a laptop keyboard, and the right
    # response to that is to remember neither rather than to remember
    # both and call the result a memory.
    verification_available: bool = False


def recordable_classes(verification_available: bool) -> tuple[str, ...]:
    """What this Tower will actually write, right now.

    Distinct from `PERSISTABLE_CLASSES`, which is the store's outer
    bound, and the distinction reaches the wire. `recorded_classes` on
    every payload is the universe of what could ever appear below it, and
    a client uses it to tell "looked for and not seen" from "never looked
    for". Naming a class there that nothing will ever write would turn
    the weaker silence into the stronger one.
    """
    if verification_available:
        return classes_in(REMEMBERED) + classes_in(VERIFY)
    return classes_in(REMEMBERED)


class RelevanceFilter:
    """Gates 1 to 3, per detection; gate 4, per sighting.

    Stateless: the temporal state lives in `SightingTracker`, where it
    describes something that happened. A stateless filter cannot go
    stale, cannot survive a restart wrongly, and can be tested one call
    at a time.
    """

    def __init__(self, policy: RelevancePolicy) -> None:
        self._policy = policy
        self._allowed = frozenset(policy.allowed_classes)

    @property
    def policy(self) -> RelevancePolicy:
        return self._policy

    def decide(self, object_class: str, score: float) -> str:
        """Whether this detection may enter a sighting at all, and why not.

        RECORD here means "worth tracking", which is weaker than "will be
        written": maturity and verification are decided later, on the
        sighting, because neither is knowable from one frame.
        """
        if is_excluded(object_class):
            return EXCLUDED
        tier = tier_of(object_class)
        if tier == CONTEXT:
            return CONTEXT_ONLY
        if tier == IGNORED or object_class not in self._allowed:
            return NOT_WHITELISTED
        if score < self._policy.min_score:
            return BELOW_MIN_SCORE
        return RECORD

    def decide_sighting(self, sighting, open_classes=()) -> str:
        """Whether a sighting has become worth a record.

        Called on every frame of an open sighting, which is why
        ALREADY_RECORDED is a verdict rather than an assertion: on the
        second and every later frame the common answer is "that one is
        already on disk".

        `open_classes` is what else is in view at this moment. It is
        passed in rather than reached for, because this filter holds no
        state -- the temporal state lives in `SightingTracker`, where it
        describes something that happened.
        """
        if sighting.recorded:
            return ALREADY_RECORDED
        verdict = self.decide(sighting.object_class, sighting.best.score)
        if verdict != RECORD:
            return verdict
        if sighting.frame_count < self._policy.min_frames:
            return TOO_BRIEF
        # A part that is only in view because its whole is. Checked
        # against what is open RIGHT NOW, so a keyboard on a desk with no
        # laptop near it is still a memory and a laptop's own keyboard is
        # not a second one.
        #
        # LIVE, not latched: latching would let one sub-maturity frame of
        # a laptop permanently suppress a keyboard for the rest of an
        # unbounded sighting, so five minutes of a keyboard alone would
        # write nothing at all. Callers pass the open set captured BEFORE
        # closing the sighting, because once it has left the tracker so
        # has everything that was open beside it.
        if any(whole in open_classes for whole in wholes_of(sighting.object_class)):
            return PART_OF_ANOTHER
        if tier_of(sighting.object_class) == VERIFY:
            agreed = sighting.verdict is not None and sighting.verdict.get("agrees")
            if not agreed:
                return UNVERIFIED
        return RECORD

    def needs_verification(self, object_class: str) -> bool:
        """Whether a second opinion is required before this class is written.

        The answer is the class tier and nothing else: a verifier cannot
        make itself required, and cannot make itself optional. Not called
        by the engine, which learns it from `decide_sighting` returning
        `UNVERIFIED`.
        """
        return tier_of(object_class) == VERIFY
