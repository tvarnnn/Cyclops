#!/usr/bin/env python
r"""Ask Object Memory when it last saw something.

Deliberately independent of any voice path, like Document Memory's query
CLI: a Siri shortcut or a wake word would sit ABOVE this, and building
the voice layer first would have made the memory untestable.

    --last-seen laptop   when was a laptop last in view
    --purge-all          really delete every observation

ON "WHERE"

The obvious question is "when and WHERE did I last see my laptop", and
half of it cannot be answered. Observations carry `spatial_ref: None` --
the field is reserved, never populated, and actively nulled on read.
Nothing in this slice knows where anything is in a room.

So "where" is answered as a FRAME REFERENCE: which capture, which frame
sequence number, which camera. That is a pointer back into the recording,
not a place. This CLI says so on every answer rather than letting a
capture id be mistaken for a location.

Nor is it a claim the object is still there. It is a record that a
CATEGORY was visible once -- not that this is YOUR laptop, and not that
absence of a record means absence of the laptop.

ON RETENTION

`--retention-days` is a request, not an authority. The store records the
window it was WRITTEN under, and every read is clamped to
min(persisted, requested): this CLI can narrow the window it sees and
cannot widen it. Asking for 3650 days, or 0 meaning forever, against a
store written under the 30-day default still gets you 30 days.

ON WHAT DELETION REACHES

`--purge-all` deletes every observation this cartridge holds, AND the
small filtered crop each one owns under `<root>/keyframes/`. It does not
touch `data/captures/`, and a record's session_id + frame_seq points
straight into it. That frame is governed by capture-side retention,
which is not Object Memory's to give away or to promise.

It also REPORTS what it could not delete, and exits non-zero when
anything survived. A purge that prints a count while a directory of
crops is still on disk is the false claim of deletion
`CARTRIDGE-GROUNDWORK.md` calls worse than an honest failure, and this
is the command a wearer's erasure request actually runs.

    .venv\Scripts\python.exe scripts/object_query.py --last-seen laptop
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.config import DEFAULT_OBSERVATION_ROOT  # noqa: E402
from tower.object_memory.relevance import PERSISTED_CLASSES  # noqa: E402
from tower.object_memory.store import (  # noqa: E402
    DEFAULT_RETENTION_DAYS,
    ObservationStore,
)

# THE SAME CONSTANT the producer and the read routes use, resolved from
# `tower/config.py` rather than spelled again here.
#
# This used to be `Path("data/object_memory")` -- relative, resolved
# against whatever directory the operator happened to be standing in, and
# never consulting TOWER_OBSERVATION_ROOT at all. `config.py` exists to
# stop exactly that, and says why at length: on 2026-08-26 the producer
# defaulted to one directory and the web process to another, and a real
# 2,203-frame walk was remembered into a store every HTTP request
# answered 404 about. "Two defaults for one directory is not a
# configuration choice; it is a bug with a settings file in front of it."
#
# `object_memory_session.py` was fixed then. This file was not, and it is
# the worse place to miss, because THIS is where deletion lives. A wearer
# asks for their object memory to be erased; an operator runs
# `--purge-all` from the tower root; `ObservationStore.purge()` on a
# directory nothing ever wrote to removes nothing and reports success,
# because `unlink(missing_ok=True)` cannot tell "already gone" from "never
# here". Exit code 0, `observations_removed: 0`, and the records still on
# disk and still being served.
#
# An explicit `--root` still wins, and `TOWER_OBSERVATION_ROOT` still wins
# over the default, so an operator who chose a directory is obeyed.
DEFAULT_ROOT = Path(
    os.environ.get("TOWER_OBSERVATION_ROOT", "").strip() or DEFAULT_OBSERVATION_ROOT
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Query observed objects (read-only unless --purge-all)."
    )
    parser.add_argument(
        "--root", type=artifact_root_arg, default=str(DEFAULT_ROOT)
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--last-seen", default=None, metavar="OBJECT_CLASS")
    parser.add_argument(
        "--retention-days",
        type=float,
        default=DEFAULT_RETENTION_DAYS,
        help=(
            "Narrow the window this read may see. The store persists the "
            "window it was written under and reads clamp to "
            "min(persisted, requested), so this can only ever serve LESS. "
            "0 means 'no limit of my own', not 'keep forever'."
        ),
    )
    parser.add_argument(
        "--purge-all",
        action="store_true",
        help="Really delete every stored observation.",
    )
    parser.add_argument("--now", type=float, default=None, help="Override the clock.")
    args = parser.parse_args(argv)

    if (args.last_seen is not None) == args.purge_all:
        parser.error("exactly one of --last-seen or --purge-all is required")

    retention = None if args.retention_days <= 0 else args.retention_days * 86400.0
    now = time.time() if args.now is None else args.now
    store = ObservationStore(args.root, retention_seconds=retention, clock=lambda: now)

    if args.purge_all:
        removed = store.purge()
        # THE PICTURES ARE PART OF THE ANSWER, and so is any the
        # filesystem refused to delete.
        #
        # Since `keyframes.py` a record may own a small crop under
        # `<root>/keyframes/`, and `purge()` deletes those with the
        # records. Printing only the observation count would let a run
        # that left a directory of crops behind -- a Windows sharing
        # violation is enough to cause one -- look identical to a run
        # that removed everything.
        keyframes_removed, retained = store.last_keyframe_purge
        # The store's own artifacts as well as the crops. Both halves
        # report, because "everything was deleted except the file with
        # every observation in it" is the answer a person asking for
        # erasure most needs and least expects.
        store_retained = store.last_purge_retained
        report = {
            "observations_removed": removed,
            "keyframes_removed": keyframes_removed,
            "keyframes_not_removed": list(retained),
            "store_files_not_removed": list(store_retained),
        }
        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            print(f"observations removed  {removed}")
            print(f"keyframes removed     {keyframes_removed}")
            if store_retained:
                print(
                    f"store files NOT removed {len(store_retained)}: "
                    + ", ".join(store_retained)
                )
            if retained:
                print(
                    f"keyframes NOT removed {len(retained)}: "
                    + ", ".join(retained)
                )
                print(
                    "Those files are still on disk. This command has NOT "
                    "deleted everything it was asked to."
                )
        # Non-zero when something survived, so a caller acting on a
        # wearer's behalf cannot read a complete deletion out of exit 0.
        return 1 if (retained or store_retained) else 0

    observation = store.last_seen(args.last_seen)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "object_class": args.last_seen,
                    # "observed", never "present": the record says a
                    # category was visible once, not that it is there.
                    "observed": observation is not None,
                    "observation": (
                        observation.to_json_dict() if observation else None
                    ),
                    # Named so a consumer cannot read a capture id as a
                    # place. There is no spatial answer in this slice.
                    "where": (
                        {
                            "kind": "frame-reference",
                            "spatial_ref": None,
                            "session_id": observation.session_id,
                            "frame_seq": observation.frame_seq,
                            "source": observation.source,
                            # The pointer resolves into data/captures/,
                            # whose lifetime this cartridge neither sets
                            # nor enforces. Purging every observation
                            # here leaves the imagery where it is.
                            "imagery_retention": "capture-side",
                        }
                        if observation
                        else None
                    ),
                },
                indent=2,
            )
        )
        return 0 if observation else 1

    if observation is None:
        print(f"No record of observing a {args.last_seen}.")
        print("(that is a statement about what was captured, not about the world)")
        if args.last_seen not in PERSISTED_CLASSES:
            print(
                f"\nNothing ever records {args.last_seen!r}: this slice only "
                f"remembers {', '.join(PERSISTED_CLASSES)}."
            )
        return 1

    age_minutes = (now - observation.observed_at) / 60.0
    best = "not tracked" if observation.best_score is None else observation.best_score
    print(f"=== last observed {observation.object_class} ===")
    print(f"  when          {age_minutes:.1f} min ago (tower-receipt time)")
    # Three fields about strength, in the order a reader should trust
    # them: the interpretation first, then the two raw numbers it is
    # accountable to. confidence follows the BEST look, because the claim
    # is "this was in view" and that is the best evidence for it; the
    # first-sighting score stays visible so the record is auditable.
    # Records written before best_score existed say "not tracked" rather
    # than borrowing the other number.
    print(f"  confidence    {observation.confidence.value} (from the best look)")
    print(f"  score         {observation.detector_score} (when it came into view)")
    print(f"  best score    {best} (strongest look while in view)")
    print(f"  capture       {observation.session_id}")
    print(f"  frame_seq     {observation.frame_seq}")
    print(f"  camera        {observation.source}")
    print(
        "\nWHERE: a frame reference, not a place. This slice stores no "
        "spatial position (spatial_ref is null) and cannot say which room "
        "or surface the object was on."
    )
    print(
        "OBSERVED, NOT PRESENT: a category was visible in that frame. Not "
        "a claim it is still there, and not a claim it is YOUR "
        f"{observation.object_class}."
    )
    print(
        "PROVENANCE: the capture and frame_seq above resolve to a stored "
        "frame under data/captures/. This record holds no imagery, but "
        "deleting it does not delete that frame -- capture retention "
        "governs the image, and Object Memory does not set it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
