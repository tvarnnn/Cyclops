"""The CV Lab's status, as a result-channel `Snapshot`.

Thin on purpose. `CVLab.status()` already builds the one document every
CV Lab surface serves -- `GET /cv-lab`, the socket's `cv_lab_status`
reply, and this -- and a producer that reshaped it here would create a
third version of the truth that only the result channel ever sees.

**Unlike World Builder's producer, this one reads LIVE state rather than
files.** It is still read-only, still writes nothing, still runs no
cartridge and still touches no frame. World Builder's producer reads a
directory another process wrote; this one reads an object in this process
that the frame path mutates. The distinction that matters is not
disk-versus-memory, it is that neither producer can alter what it
reports -- `test_the_result_channel_never_writes` and
`test_the_module_container_is_untouched_by_a_subscription` are the two
gates on that.

The concurrency this introduces is real and is handled where the state
is, not here: `ResultHub.poll_once` computes snapshots with
`asyncio.to_thread`, so `status()` runs on a worker thread while the
event loop is mutating the Lab. `CVLab` takes a lock around every state
transition and around building the document. This module inherits that
and adds nothing of its own.

`revision` is derived from the payload rather than counted, so it cannot
drift from the content -- but three of the figures in it advance with
WALL CLOCK rather than with anything happening, and those are excluded
before hashing.

`elapsed_s` and the two throughput rates derived from it change on every
read, so left in the hash `compute_revision` would produce a new value on
every poll and `revision_changed: true` -- the field the channel defines
as "news, not a heartbeat" -- would fire twice a second on a Lab that had
seen no frame at all (measured over the wire without the exclusion: four
polls, zero new frames, four distinct revisions).

`capacity_fps` is NOT excluded: it is derived from measured per-frame
cost alone and moves only when a frame is processed. Nor is
`source.receiving_frames`, which flips without a frame arriving but flips
precisely BECAUSE frames stopped arriving -- that is the news.
"""

from tower.results.envelope import Snapshot, compute_revision

# Dotted paths removed before hashing. Each one advances with the clock
# and with nothing else, so a change in it is not a change in what the
# Tower knows.
VOLATILE_PATHS = (
    "run.elapsed_s",
    "run.throughput.processed_fps",
    "run.throughput.offered_fps",
)


class ExperimentalCVStatusProducer:
    """Turns the Lab's own document into a channel snapshot.

    Holds the Lab, not a copy of anything it says. A cached payload would
    be a second answer to "what is the Lab doing", and the first thing
    that goes wrong with two answers is that one of them is older.
    """

    def __init__(self, lab) -> None:
        self._lab = lab

    def snapshot(self) -> Snapshot:
        payload = self._lab.status()
        return Snapshot(
            payload=payload,
            revision=compute_revision(payload, VOLATILE_PATHS),
            volatile_fields=VOLATILE_PATHS,
        )


def unavailable_payload(reason: str) -> dict:
    """What this Tower says when it has no Lab at all.

    Shaped like a status document with everything hollow rather than as a
    bare error, so a client decodes one thing whichever it gets. The
    lifecycle state is `unavailable`, which iOS renders as "the Tower
    cannot do this" rather than as "wait a moment" -- a different
    instruction to a person, and the right one.

    The contract identifiers are REAL, not null. They describe what this
    BUILD speaks, which is true whether or not a Lab is loaded -- and the
    whole point of iOS's third state ("offered, implemented, unreachable")
    is that those are separable. Nulling them would hand a client an
    identifier to compare for equality that could never equal anything.

    Used by the result channel's producer and by the control-plane
    handler, so that a refusal on a Tower with no Lab still carries a
    document.
    """
    from tower.cv_lab.contracts import (
        CONTROL_CONTRACT,
        FRAME_RESULT_CONTRACT,
        STATUS_CONTRACT,
        TIME_BASIS,
    )

    return {
        "contract": STATUS_CONTRACT,
        "control_contract": CONTROL_CONTRACT,
        "frame_result_contract": FRAME_RESULT_CONTRACT,
        "tower_instance_id": None,
        "time_basis": TIME_BASIS,
        "lifecycle": {
            "state": "unavailable",
            "reason": reason,
            "since": None,
            "run_id": None,
        },
        "available": [],
        "selected": None,
        "default_experiment": None,
        "device_requested": None,
        "run": None,
        "source": {
            "clients_connected": None,
            "receiving_frames": False,
            "last_frame_at": None,
            "frames_offered_total": 0,
            "frames_rejected_before_lab": 0,
            "idle_after_s": None,
        },
    }
