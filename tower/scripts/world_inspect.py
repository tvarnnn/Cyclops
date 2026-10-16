#!/usr/bin/env python
"""Inspect a saved World Builder world, offline.

Reads a world back cold, with no live session and no shared state, so a saved
world can be inspected without the session that produced it.

Reports an unavailable figure as `unknown`, never as `0`. "No poses were
solved" and "poses were never attempted" are different facts, and once
both render as 0 the difference is gone.

    .venv\\Scripts\\python.exe scripts/world_inspect.py --list
    .venv\\Scripts\\python.exe scripts/world_inspect.py --world <id>
    .venv\\Scripts\\python.exe scripts/world_inspect.py --world <id> --format json
    .venv\\Scripts\\python.exe scripts/world_inspect.py --world <id> --trajectory
    .venv\\Scripts\\python.exe scripts/world_inspect.py --world <id> --verify
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder.inspect import WorldView, render_text  # noqa: E402
from tower.world_builder.store import WorldStore  # noqa: E402

DEFAULT_ROOT = Path("data/world_builder")


def verify(store: WorldStore, view: WorldView) -> tuple[list[str], list[str]]:
    """Check every journal line has its image, and list orphan images.

    A journal line pointing at a missing image is corruption; an image
    with no journal line is a harmless orphan. Reporting them separately
    keeps that asymmetry visible instead of collapsing both into "world
    is broken".
    """
    missing: list[str] = []
    orphans: list[str] = []
    for session_id in view.session_ids():
        session_dir = store.session_dir(view.world_id, session_id)
        referenced = set()
        for keyframe in store.read_keyframes(view.world_id, session_id):
            path = session_dir / keyframe.image_relpath
            referenced.add(path.name)
            if not path.exists():
                missing.append(keyframe.keyframe_id)
        images = store.images_dir(view.world_id, session_id)
        if images.exists():
            orphans.extend(
                image.name
                for image in sorted(images.glob("*.jpg"))
                if image.name not in referenced
            )
    return missing, orphans


def follow(
    view: WorldView,
    *,
    session_id: str | None,
    as_json: bool,
    poll_seconds: float,
    max_idle_polls: int | None,
    sleep=time.sleep,
    out=None,
) -> int:
    """Print a session's events as they are appended.

    This is the read half of live viewing, and it needs nothing the
    persistence design did not already have: the journal is written while
    the walk happens, `event_id` is dense so a gap means a genuinely
    dropped event, and a cursor turns an archive into a stream. No socket,
    no subscription lifecycle, no backpressure policy, no risk of an
    unbounded queue.

    The writer is a different process and this reader never touches its
    state, which is exactly why following a live session cannot perturb
    the world being built.
    """
    stream = out if out is not None else sys.stdout
    session_ids = view.session_ids()
    if not session_ids:
        print(f"world {view.world_id} has no sessions", file=sys.stderr)
        return 2
    if session_id is None:
        session_id = session_ids[-1]
    elif session_id not in session_ids:
        print(f"no session {session_id!r} in world {view.world_id}", file=sys.stderr)
        return 2

    if not as_json:
        print(f"=== Following {view.world_id} / {session_id} ===", file=stream)

    cursor = None
    idle_polls = 0
    try:
        while True:
            events = view.events(session_id, after_event_id=cursor)
            for event in events:
                cursor = event["event_id"]
                if as_json:
                    print(json.dumps(event), file=stream, flush=True)
                else:
                    payload = " ".join(
                        f"{key}={value}"
                        for key, value in sorted(event.get("payload", {}).items())
                    )
                    print(
                        f"  [{event['event_id']:>4}] {event['kind']:<18} {payload}",
                        file=stream,
                        flush=True,
                    )
                if event["kind"] == "session_stopped":
                    return 0

            if events:
                idle_polls = 0
            else:
                idle_polls += 1
                if max_idle_polls is not None and idle_polls >= max_idle_polls:
                    return 0
            sleep(poll_seconds)
    except KeyboardInterrupt:
        # Walking away from a viewer is a normal way to end it, not a
        # crash, and it must never look like the session failed.
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a saved World Builder world (read-only)."
    )
    parser.add_argument(
        "--root", type=artifact_root_arg, default=str(DEFAULT_ROOT)
    )
    parser.add_argument("--world", help="World id to inspect.")
    parser.add_argument(
        "--list", action="store_true", help="List world ids and exit."
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Print the ordered camera trajectory per session.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check every keyframe image referenced by a journal exists.",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help=(
            "Tail a session's event journal, printing each update as it "
            "happens. Works on a session that is still open."
        ),
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Which session to follow. Defaults to the most recent one.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.25,
        help="How often --follow checks for new events.",
    )
    parser.add_argument(
        "--max-idle-polls",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop --follow after N quiet polls. Unset follows until the "
            "session reports that it stopped."
        ),
    )
    args = parser.parse_args(argv)

    store = WorldStore(args.root)

    if args.list:
        world_ids = store.list_world_ids()
        if args.format == "json":
            print(json.dumps({"worlds": world_ids}, indent=2))
        else:
            print("\n".join(world_ids) if world_ids else "(no worlds)")
        return 0

    if not args.world:
        parser.error("--world is required unless --list is given")

    if args.world not in store.list_world_ids():
        print(f"no world {args.world!r} under {args.root}", file=sys.stderr)
        return 2

    view = WorldView(store, args.world)

    if args.follow:
        return follow(
            view,
            session_id=args.session,
            as_json=args.format == "json",
            poll_seconds=args.poll_seconds,
            max_idle_polls=args.max_idle_polls,
        )

    report = view.to_report()

    if args.trajectory:
        report["trajectory"] = {
            session_id: view.trajectory(session_id)
            for session_id in view.session_ids()
        }

    if args.verify:
        missing, orphans = verify(store, view)
        report["verify"] = {
            "missing_images": missing,
            "orphan_images": orphans,
            "ok": not missing,
        }

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
        if args.trajectory:
            for session_id, rows in report["trajectory"].items():
                print(f"\n=== Trajectory {session_id} ===")
                for row in rows:
                    position = (
                        "unknown"
                        if row["pose"] is None
                        else ", ".join(
                            f"{value:+.3f}" for value in row["pose"]["translation"]
                        )
                    )
                    print(
                        f"  seq {row['source_seq']:>7}  seg {row['segment_index']}  "
                        f"{row['status']:<12} [{position}]  {row['image_relpath']}"
                    )
        if args.verify:
            result = report["verify"]
            print("\n=== Verify ===")
            print(f"missing images  {len(result['missing_images'])}")
            print(f"orphan images   {len(result['orphan_images'])}")
            print(f"ok              {result['ok']}")

    if args.verify and not report["verify"]["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
