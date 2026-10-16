"""Asking the scene a question, and getting a refusal when it cannot answer.

The brief's four example questions, and one rule that governs all of them:
**an answer must never be more confident than the evidence.**

    How many people are in this room?          -> count, from TRACKS
    Where is the desk?                         -> camera-relative position
    Where is a chair?                          -> same
    How many appear to be facing my direction? -> orientation, or a refusal

The third question hides a trap the others do not. If orientation is not
running -- no face model vendored, `TOWER_SCENE_ORIENTATION` off, or an
estimator that has never once succeeded -- then "how many people are
facing me" has **no answer**, and returning 0 would be a lie of exactly
the kind Core Principle 3 forbids: an observation gap reported as an
observation of absence. (The keypoint model this stage replaced cost
956 ms on CPU against an 83.5 ms frame interval; the face-visibility
stage costs ~9 ms per face.)

Nothing here reads or writes a store. There is no store.
"""

from dataclasses import dataclass

from tower.scene.records import FACING_UNKNOWN
from tower.scene.state import REFUSED_RELATIONSHIPS, SceneState, describe_position


@dataclass(frozen=True)
class Answer:
    """A result, or an honest refusal. Never a guess.

    `answered` is the field that matters. `False` plus a `reason` is a
    complete, correct response, and a caller rendering it as "I can't tell
    you that" is telling the truth.
    """

    question: str
    answered: bool
    value: object = None
    reason: str = ""
    detail: dict | None = None

    def to_json_dict(self) -> dict:
        return {
            "question": self.question,
            "answered": self.answered,
            "value": self.value,
            "reason": self.reason,
            "detail": self.detail or {},
        }


class SceneQuery:
    """Read-only questions about one scene state."""

    def __init__(self, state: SceneState) -> None:
        self._state = state

    # -- counting ------------------------------------------------------

    def count(self, label: str) -> Answer:
        """How many of these are around the wearer.

        From confirmed TRACKS. Summing detections would report a count
        that flickers with the detector, which is the specific error the
        brief calls out.

        A zero here is honest in a way the facing question's zero would
        not be: the detector ran, looked for this class, and found none.
        The reason says which of the two it is.
        """
        found = self._state.count(label)
        return Answer(
            question=f"how many {label}",
            answered=True,
            value=found,
            reason=(
                f"{found} confirmed track(s), from tracking rather than raw "
                f"detections, at score threshold "
                f"{self._state.score_threshold}"
            ),
            detail={
                "label": label,
                "frames_observed": self._state.frames_observed,
                "detector": self._state.detector,
                "note": (
                    "absence of a detection is not evidence of absence: an "
                    "occluded or out-of-frame object is simply not seen"
                ),
            },
        )

    # -- location ------------------------------------------------------

    def where_is(self, label: str) -> Answer:
        """Where something is, camera-relative and labelled as such."""
        tracks = self._state.of_class(label)
        if not tracks:
            return Answer(
                question=f"where is the {label}",
                answered=False,
                reason=(
                    f"no confirmed {label} in view. That is a statement "
                    "about what is currently visible, not about the room"
                ),
            )
        return Answer(
            question=f"where is the {label}",
            answered=True,
            value=[
                describe_position(
                    track, self._state.frame_width, self._state.frame_height
                )
                for track in tracks
            ],
            reason="camera-relative position; there is no live world pose",
            detail={"track_ids": [track.track_id for track in tracks]},
        )

    # -- orientation ---------------------------------------------------

    def facing_wearer(self) -> Answer:
        """How many people appear to be facing the wearer's direction.

        **Refuses when orientation is disabled.** Returning 0 would be an
        observation gap reported as an observation of absence -- the exact
        error Core Principle 3 names, and the one most likely to be
        mistaken for a real answer because zero looks like data.

        And even when enabled it is not gaze. A person facing the wearer
        may be looking past them; the camera cannot tell.
        """
        if not self._state.orientation_enabled:
            return Answer(
                question="how many people appear to be facing my direction",
                answered=False,
                reason=(
                    "orientation has never produced an estimate -- either "
                    "it is disabled, or the pose model has not once "
                    "succeeded -- so this was never measured. Reporting 0 "
                    "would be an observation gap presented as an "
                    "observation of absence"
                ),
                detail={
                    "people_in_view": self._state.count("person"),
                    "how_to_enable": (
                        "supply a pose estimator; it costs ~956 ms per call "
                        "on CPU and ~43 ms on CUDA, and the default device "
                        "is CPU, which is why it is off by default"
                    ),
                },
            )

        people = self._state.of_class("person")
        facing = self._state.facing_wearer()
        unknown = [
            track for track in people if track.facing.state == FACING_UNKNOWN
        ]
        return Answer(
            question="how many people appear to be facing my direction",
            answered=True,
            value=len(facing),
            reason=(
                "coarse head orientation from facial-keypoint visibility. "
                "This is NOT gaze: the camera cannot establish attention"
            ),
            detail={
                "people_in_view": len(people),
                "orientation_unknown": len(unknown),
                # None when nothing has an estimate; `or 0.0` would read
                # that as "zero seconds old", freshness never measured.
                "oldest_estimate_seconds": max(
                    (
                        track.facing.age_seconds
                        for track in people
                        if track.facing.age_seconds is not None
                    ),
                    default=None,
                ),
                "states": {
                    track.track_id: track.facing.to_json_dict()
                    for track in people
                },
            },
        )

    # -- relationships -------------------------------------------------

    def relationships(self) -> Answer:
        return Answer(
            question="what is where, relative to what",
            answered=True,
            value=[relation.to_json_dict() for relation in self._state.relations],
            reason="camera-relative relations from 2-D boxes",
            detail={"refused": REFUSED_RELATIONSHIPS},
        )

    def why_not(self, relationship: str) -> Answer:
        """Why a relationship is not asserted.

        A refusal that can explain itself is worth far more than a silent
        omission: it stops the next cartridge re-deriving the same
        conclusion, and it names the measurement that would change it.
        """
        reason = REFUSED_RELATIONSHIPS.get(relationship)
        if reason is None:
            return Answer(
                question=f"why is {relationship!r} not reported",
                answered=False,
                reason=(
                    f"{relationship!r} is not a relationship this cartridge "
                    "knows about, refused or otherwise"
                ),
            )
        return Answer(
            question=f"why is {relationship!r} not reported",
            answered=True,
            value=reason,
            reason="deliberately refused for lack of evidence",
        )
