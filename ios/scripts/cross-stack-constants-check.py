#!/usr/bin/env python3
"""Does the Swift name every constant the Tower actually sends?

WHY THIS EXISTS, AND IT IS A REAL BREAK RATHER THAN A HYPOTHETICAL ONE.

`contract-drift-check.py` beside this one compares contract IDENTIFIERS
against a live Tower. That catches a version skew and nothing else. On
2026-08-29 a Tower change split `filter_means` into two values -- one for
a picture filtered on read, one for a picture filtered before it was
written -- and the shipped iOS decoder compared that field against a
single constant. Every payload carrying the new value would have failed
the parse outright, and the whole owned-keyframe feature would have been
invisible on the phone. Both halves were written the same day, by
different agents, and neither test suite could see the other.

This is the check that would have caught it, and it needs no running
Tower: it imports the Tower's own constants and greps the Swift for them.

WHAT IT CANNOT DO.

It proves the strings are PRESENT, not that they are used correctly -- a
constant declared and never compared is still a pass here. It is a
tripwire for a value that moved on one side only, which is the failure
that actually happens when two lanes ship the same day. Everything else
is the type checker's job, and the type checker lives on a Mac.

    python ios/scripts/cross-stack-constants-check.py

Exit codes: 0 agreement, 1 something the Tower sends is not named in the
Swift, 2 the Tower package could not be imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SWIFT_ROOT = REPO / "ios" / "Glasses" / "Workspaces" / "ObjectMemory"

# Wire KEYS, which are not Python constants anywhere -- they are dict
# literals in the route adapters. Listed by hand, which is the honest
# shape: a key that stops being sent is a contract change, and it should
# take an edit here to notice.
WIRE_KEYS = (
    "following_this_session",
    "frame_available",
    "imagery_source",
    "imagery_retention",
    "memory_retained",
    "filter_means",
)


def tower_constants() -> dict[str, str]:
    sys.path.insert(0, str(REPO / "tower"))
    from tower.object_memory.imagery import SOURCE_CAPTURE, SOURCE_KEYFRAME
    from tower.results.object_memory import (
        FILTER_MEANS_BEFORE_WRITE,
        FILTER_MEANS_ON_READ,
        IMAGERY_CLAIM,
        IMAGERY_CONTRACT,
        RETENTION_CAPTURE_SIDE,
        RETENTION_OBJECT_MEMORY,
    )
    from tower.routes.sessions import SESSION_CONTRACT, STATE_MEANS

    return {
        "imagery contract": IMAGERY_CONTRACT,
        "imagery claim": IMAGERY_CLAIM,
        "filter_means (on read)": FILTER_MEANS_ON_READ,
        "filter_means (before write)": FILTER_MEANS_BEFORE_WRITE,
        "imagery_source capture": SOURCE_CAPTURE,
        "imagery_source keyframe": SOURCE_KEYFRAME,
        "session contract": SESSION_CONTRACT,
        "state_means": STATE_MEANS,
        "imagery_retention capture-side": RETENTION_CAPTURE_SIDE,
        "imagery_retention object-memory": RETENTION_OBJECT_MEMORY,
    }


def main() -> int:
    try:
        constants = tower_constants()
    except Exception as exc:  # noqa: BLE001
        print(f"could not import the Tower package: {exc.__class__.__name__}: {exc}")
        print("run this from a checkout with tower/ present and importable")
        return 2

    swift = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(SWIFT_ROOT.glob("*.swift"))
    )
    if not swift:
        print(f"no Swift found under {SWIFT_ROOT}")
        return 2

    missing: list[str] = []
    for name, value in constants.items():
        present = value in swift
        print(("  ok      " if present else "  ABSENT  ") + f"{name:32s} {value!r}")
        if not present:
            missing.append(f"{name} = {value!r}")

    print()
    for key in WIRE_KEYS:
        present = key in swift
        print(("  ok      " if present else "  ABSENT  ") + f"wire key {key}")
        if not present:
            missing.append(f"wire key {key}")

    print()
    if missing:
        print("THE TOWER SENDS SOMETHING THE SWIFT DOES NOT NAME:")
        for item in missing:
            print("  " + item)
        return 1
    print("agreement: the Swift names every constant and key the Tower sends")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
