"""A world the Tower finished from only PART of its walk (U-PARTIAL; WORLD-BUILDER-COMPONENTS.md §3.1a).

A session can end `interrupted` or `error` and still have a final solve that succeeds: the wearer left
World Builder while the camera was recording, the recorder stopped itself at its bound, a reconnect was
abandoned, the closed-capture drain guard counted frames the stop never let the session read, or the
builder failed mid-walk and the world was repaired afterwards. The status producer's "a completed
finalization outranks how the capture ended" arm then reported it `ready` -- "the world was finished
afterwards and is complete" -- and the phone said **Saved** over part of a walk. This module is the one
place that decides the session was saved in part, and both producers (the status channel and the
`GET /worlds` row) ask it, so the two surfaces cannot disagree.

It reads the record it is handed and, for rule F only, one capture manifest. It never writes: the
record on disk is evidence, and `walk` exists only in what the Tower sends.

Two switches, both default OFF (`tower.config`): `TOWER_WORLD_PARTIAL_STATE` (rule R) and
`TOWER_WORLD_PARTIAL_FRAMES` (rule F, only with the first on). Off, the producers never call
`partial_walk`, and `present_finalization(x, None) is x`, so every payload is byte for byte as before.
"""

from __future__ import annotations

from pathlib import Path

from tower.storage import read_json_closed
from tower.world_builder.records import FINAL_SOLVE_SOLVED, FINALIZATION_COMPLETE

WALK_SAVED_PART = "part"
WALK_REASON_SESSION_INTERRUPTED = "session-interrupted"
WALK_REASON_SESSION_ERROR = "session-error"
WALK_REASON_FRAMES_NOT_READ = "frames-not-read"

_REASON_BY_END = {
    "interrupted": WALK_REASON_SESSION_INTERRUPTED,
    "error": WALK_REASON_SESSION_ERROR,
}

# The closed set's walk members (COMPONENTS v12 §3.1a), beside `coherence_publish.NOTICE_SENTENCES`.
# `{observed}` and `{recorded}` are integer frame counts, written as plain digits.
WALK_NOTICE_SENTENCES = {
    "walk-ended-early": (
        "only part of this walk is in this world: the Tower stopped before it had looked at all "
        "of the walk; an owner can re-capture this walk"),
    "walk-frames-not-read": (
        "only part of this walk is in this world: the Tower looked at {observed} of the {recorded} "
        "frames it recorded for the walk; an owner can re-capture this walk"),
}

# The capture recorder's manifest name and layout (`tower/capture.py`: `CAPTURE_FILENAME`,
# `CaptureRecorder.capture_dir`). Spelled here rather than imported so the result producers do not pull
# the recorder's module in for one file name.
_CAPTURE_MANIFEST = "capture.json"

# The bound every client copy of a finalization text respects (`coherence_publish.FINALIZATION_TEXT_MAX_CHARS`,
# the phone's `WorldTowerText.maximumLength`). Imported lazily in `present_finalization`, as the status
# producer imports `coherence_publish`.


def _count(value) -> int | None:
    """A frame count as a record holds it, or None: an int >= 0, and never a bool."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _capture_name(capture_id) -> str | None:
    """`capture_id` when it is a non-empty string that names ONE directory, else None. A record is not
    allowed to point this read outside `<capture_root>/captures/`."""
    if not isinstance(capture_id, str) or not capture_id.strip():
        return None
    if capture_id in (".", "..") or "/" in capture_id or "\\" in capture_id or Path(capture_id).name != capture_id:
        return None
    return capture_id


def capture_manifest_path(capture_root, session) -> Path | None:
    """`<capture_root>/captures/<capture_id>/capture.json` for the session's OWN capture, or None."""
    if capture_root is None:
        return None
    name = _capture_name(getattr(session, "capture_id", None))
    if name is None:
        return None
    return Path(capture_root) / "captures" / name / _CAPTURE_MANIFEST


def recorded_frames(capture_root, session) -> int | None:
    """`frames_written` of the session's OWN capture, or None (= not judged). Never raises.

    None unless the manifest is a dict naming the same `capture_id`, is closed (`ended_at` a number) and
    `frames_written` is an int >= 0 (a bool is not). An unreadable, open or absent manifest decides
    nothing: that is the old world, never an error."""
    try:
        path = capture_manifest_path(capture_root, session)
        if path is None:
            return None
        manifest = read_json_closed(path)
        if not isinstance(manifest, dict):
            return None
        if manifest.get("capture_id") != session.capture_id:
            return None
        ended_at = manifest.get("ended_at")
        if isinstance(ended_at, bool) or not isinstance(ended_at, (int, float)):
            return None
        return _count(manifest.get("frames_written"))
    except Exception:  # noqa: BLE001 -- a manifest that cannot be read is "not judged", by contract
        return None


def partial_walk(session, *, recorded: int | None = None) -> dict | None:
    """`{"saved", "reason", "sentence"}` for a walk saved in part (rules R and F, spec §1.3), else None.

    Pure: reads nothing but its arguments. `recorded` is None unless `TOWER_WORLD_PARTIAL_FRAMES` is on.

    Precondition for both rules: the record is closed (`ended_at` set) and its finalization is a dict
    whose `state` is `complete`.

    - **R.** `end_reason` is `interrupted` or `error`, and `final_solve` is `solved`.
    - **F.** `end_reason` is `stop`, the session names a capture, and its `frames_observed` is an
      integer strictly below the capture's `recorded` frame count.
    """
    if session is None or getattr(session, "ended_at", None) is None:
        return None
    finalization = getattr(session, "finalization", None)
    if not isinstance(finalization, dict) or finalization.get("state") != FINALIZATION_COMPLETE:
        return None
    end_reason = getattr(session, "end_reason", None)
    reason = _REASON_BY_END.get(end_reason) if isinstance(end_reason, str) else None
    if reason is not None:
        if finalization.get("final_solve") != FINAL_SOLVE_SOLVED:
            return None
        return {
            "saved": WALK_SAVED_PART,
            "reason": reason,
            "sentence": WALK_NOTICE_SENTENCES["walk-ended-early"],
        }
    if end_reason != "stop" or _capture_name(getattr(session, "capture_id", None)) is None:
        return None
    observed = _count(getattr(session, "frames_observed", None))
    written = _count(recorded)
    if observed is None or written is None or not observed < written:
        return None
    return {
        "saved": WALK_SAVED_PART,
        "reason": WALK_REASON_FRAMES_NOT_READ,
        "sentence": WALK_NOTICE_SENTENCES["walk-frames-not-read"].format(
            observed=str(observed), recorded=str(written)),
    }


def present_finalization(finalization, walk):
    """The outbound finalization: what the row and `lifecycle.finalization` carry.

    - `walk` is None: `finalization` ITSELF, the same object, so the switch off is byte-identical.
    - otherwise a COPY with `walk: {saved, reason}` and the walk's sentence first in `notice`, then
      `"; "` and the record's own notice when it has a non-blank one. A composition longer than
      `coherence_publish.FINALIZATION_TEXT_MAX_CHARS` (700) sends the walk sentence alone: only a legacy
      notice outside the closed set can be that long (560 is the closed-set worst case).

    Applied AFTER `client_safe_finalization`, so the record's notice is already scrubbed; the walk
    sentence is a closed-set member and needs no scrubbing. The record on disk is never touched."""
    if walk is None:
        return finalization
    from tower.world_builder.coherence_publish import FINALIZATION_TEXT_MAX_CHARS  # noqa: PLC0415

    base = finalization if isinstance(finalization, dict) else {}
    sentence = walk["sentence"]
    own = base.get("notice")
    notice = sentence
    if isinstance(own, str) and own.strip():
        composed = f"{sentence}; {own}"
        if len(composed) <= FINALIZATION_TEXT_MAX_CHARS:
            notice = composed
    return {**base, "notice": notice, "walk": {"saved": walk["saved"], "reason": walk["reason"]}}


__all__ = [
    "WALK_NOTICE_SENTENCES",
    "WALK_REASON_FRAMES_NOT_READ",
    "WALK_REASON_SESSION_ERROR",
    "WALK_REASON_SESSION_INTERRUPTED",
    "WALK_SAVED_PART",
    "capture_manifest_path",
    "partial_walk",
    "present_finalization",
    "recorded_frames",
]
