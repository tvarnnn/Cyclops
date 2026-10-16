"""The generic envelope every cartridge result travels in.

Small on purpose. `ios/docs/agent-handoffs/IOS-TO-TOWER.md` section 6 lists
seventeen things iOS deliberately did not assume, and most of them are
payload-shaped (geometry representation, pose schema, relation predicates,
metric names). A big universal schema would have to decide those centrally, on
behalf of cartridges that have not been written yet. So the envelope
carries only what is true of EVERY result -- who produced it, under which
agreement, in what order, when -- and the payload stays each cartridge's
own business.

The envelope is deliberately NOT a delta format. See publisher.py for why
snapshots are the whole design rather than an implementation choice.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field


def json_safe(value):
    """Replace every non-finite float with None, recursively.

    `NaN` and `Infinity` are not JSON. Python's `json.dumps` emits them as
    bare `NaN` / `Infinity` because `allow_nan` defaults to True, and
    Starlette's `send_json` takes that default -- so ONE non-finite float
    anywhere in a payload makes the ENTIRE message unparseable by a strict
    decoder. Swift's JSONDecoder is strict. A `NaN` reprojection RMS from
    a calibration run, sent on a real socket, made a conforming parser
    reject the whole frame: not one field degraded, every
    `cartridge_result` for that session lost.

    Applied at the envelope boundary rather than at each producer, because
    the failure is catastrophic and total, and the next cartridge must not
    have to remember. `None` is already this contract's word for "not
    established", which is the honest reading of a non-finite measurement.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value

from tower.results.contracts import ENVELOPE_CONTRACT, TIME_BASIS


@dataclass(frozen=True)
class ResultEnvelope:
    """One structured result, addressed to one subscription.

    `seq` is dense per subscription and assigned at SEND time, starting
    at 1. It is the ordering guarantee and nothing else: because every
    result is a complete snapshot, a client never needs to detect a gap
    to know whether it missed information -- it did not, the newest
    snapshot supersedes everything before it. `coalesced` reports how many
    snapshots were superseded while the client was not reading, so a slow
    consumer can SEE that it was slow instead of inferring it from a
    sequence gap.

    That split is the point. A single gappy sequence number would conflate
    "you missed data" with "you were sent less data because you did not
    need it", and only the first deserves alarm.
    """

    cartridge: str
    result_type: str
    contract: str
    subscription_id: str
    seq: int
    revision: str
    revision_changed: bool
    tower_sent_at: float
    payload: dict
    coalesced: int = 0
    # Whether the client's `since_revision`, if it sent one, was
    # recognised. See publisher.py: an unrecognised cursor is never an
    # error, because the reply is a complete snapshot either way.
    cursor_status: str | None = None
    envelope_contract: str = ENVELOPE_CONTRACT
    time_basis: str = TIME_BASIS
    # Always true in V1 and stated rather than implied. A consumer that
    # reads this field and finds `false` one day will know to look for
    # delta-merge rules; one that never sees the field at all would
    # quietly assume whichever it was written against.
    snapshot: bool = True

    def to_json_dict(self) -> dict:
        return json_safe({
            "type": "cartridge_result",
            "envelope_contract": self.envelope_contract,
            "subscription_id": self.subscription_id,
            "cartridge": self.cartridge,
            "result_type": self.result_type,
            "contract": self.contract,
            "seq": self.seq,
            "revision": self.revision,
            "revision_changed": self.revision_changed,
            "coalesced": self.coalesced,
            "cursor_status": self.cursor_status,
            "snapshot": self.snapshot,
            "tower_sent_at": self.tower_sent_at,
            "time_basis": self.time_basis,
            "payload": self.payload,
        })


@dataclass(frozen=True)
class Snapshot:
    """What a producer returns: a payload plus its change identity.

    `revision` is computed from the payload with the volatile fields
    excluded, so that a figure which advances on every poll -- elapsed
    mapping seconds, most obviously -- does not make every update look
    like a change. `ios/docs/agent-handoffs/IOS-TO-TOWER.md` §1.2 asks for
    exactly this:

        "A monotonic revision or counter, so the UI can tell **new data
         from repeated data** without diffing geometry. Without it, every
         update looks like a change"

    and section 6 item 17 settles its type: an opaque string compared for
    equality, "because inequality is the entire requirement".

    Deriving it by hashing the payload rather than incrementing a counter
    is what stops it drifting from the content. A hand-maintained counter
    is a second source of truth about whether something changed, and the
    two disagree the first time someone adds a field and forgets to bump.
    """

    payload: dict
    revision: str
    volatile_fields: tuple[str, ...] = field(default=())


def compute_revision(payload: dict, volatile_paths: tuple[str, ...] = ()) -> str:
    """A stable, opaque change identity for a payload.

    `volatile_paths` are dotted paths (``"progress.mapping_seconds"``)
    removed before hashing. They are the fields whose value advances
    without anything having actually happened.

    Truncated to 16 hex characters. This is a change detector, not a
    security primitive: iOS compares it for equality and nothing else, and
    64 bits makes an accidental collision between two states of one
    session not worth reasoning about.
    """
    reduced = json_safe(_without(payload, volatile_paths))
    canonical = json.dumps(reduced, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _without(payload: dict, dotted_paths: tuple[str, ...]) -> dict:
    """Copy `payload` with the named dotted paths removed.

    A path naming something absent is not an error: producers share a
    volatile-path list across payload shapes that legitimately differ
    (a live session has progress figures, an idle one does not), and
    raising would make the honest "this field is not present right now"
    case fail.
    """
    import copy

    reduced = copy.deepcopy(payload)
    for path in dotted_paths:
        parts = path.split(".")
        cursor = reduced
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                cursor = None
                break
            cursor = cursor[part]
        if isinstance(cursor, dict):
            cursor.pop(parts[-1], None)
    return reduced
