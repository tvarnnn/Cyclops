#!/usr/bin/env python
"""Stage a stopped session's transient masks while its final solve runs.

The child `world_build_session.MasksAtStop` starts after Stop when
`TOWER_WORLD_MASKS_AT_STOP` is on; the builder terminates it when the final
solve has returned. Everything it writes goes to a run
directory under `<store root>/masks_at_stop/`, never into the mask cache: the
final surface stage moves a staged mask into the cache only after checking it
against its own inputs (`world_builder/masks_at_stop.py`).

    .venv\\Scripts\\python.exe scripts/world_masks_at_stop.py --root <root> --world <id> --session <id> [--parent-pid N]

Reads the raw-frame paths the builder handed it from
`masks_at_stop.sources_path` (the same dict the final solve's `sources.json` is
written from). `--parent-pid`: stop at the next keyframe once that process is gone (the
builder passes its own; under the Tower's supervisor its Job Object ends this
child anyway, but a builder run by hand or by a harness has no such job). Exit 0
whatever it managed to stage; 1 only when it crashed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.power_throttling import apply_world_build_power_policy  # noqa: E402
from tower.world_builder import masks_at_stop  # noqa: E402
from tower.world_builder.store import WorldStore  # noqa: E402


def parent_gone(pid: int | None):
    """A `should_stop` that turns true once process `pid` is gone (or replaced: the same pid
    with another start time), checked at most once a second. Never true without a pid."""
    if pid is None:
        return lambda: False
    import psutil  # noqa: PLC0415

    try:
        created = psutil.Process(pid).create_time()
    except psutil.Error:
        return lambda: True
    state = {"at": 0.0, "gone": False}

    def gone() -> bool:
        now = time.monotonic()
        if not state["gone"] and now - state["at"] >= 1.0:
            state["at"] = now
            try:
                state["gone"] = psutil.Process(pid).create_time() != created
            except psutil.Error:
                state["gone"] = True
        return state["gone"]

    return gone


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=artifact_root_arg, required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--parent-pid", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("PIL", "urllib3", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # TOWER_WORLD_NO_POWER_THROTTLING (default off): never let Windows throttle this
    # process's CPU speed. See tower/power_throttling.py. Computes nothing differently.
    apply_world_build_power_policy("masks")
    store = WorldStore(Path(args.root))
    sources = masks_at_stop.read_sources(store, args.world, args.session)
    try:
        summary = masks_at_stop.prefill(store, args.world, args.session, sources=sources,
                                        should_stop=parent_gone(args.parent_pid))
    except Exception:  # noqa: BLE001 -- the stage computes whatever is missing
        logging.getLogger(__name__).exception("%s the worker failed", masks_at_stop.LOG)
        return 1
    print(json.dumps({k: summary.get(k) for k in ("state", "detail", "staged", "chunks", "fills",
                                                   "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
