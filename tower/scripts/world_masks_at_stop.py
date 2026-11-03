#!/usr/bin/env python
"""Stage a stopped session's transient masks while its final solve runs.

The child `world_build_session.MasksAtStop` starts after Stop when
`TOWER_WORLD_MASKS_AT_STOP` is on; the builder terminates it when the final
solve has returned. Everything it writes goes to
`<world>/dense/<session>/work/masks_at_stop/`, never into the mask cache: the
final surface stage moves a staged mask into the cache only after checking it
against its own inputs (`world_builder/masks_at_stop.py`).

    .venv\\Scripts\\python.exe scripts/world_masks_at_stop.py --root <root> --world <id> --session <id>

Reads the raw-frame paths the builder handed it from `sources.json` in the
staging directory (the same dict the final solve's `sources.json` is written
from). Exit 0 whatever it managed to stage; 1 only when it crashed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder import masks_at_stop  # noqa: E402
from tower.world_builder.store import WorldStore  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=artifact_root_arg, required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("PIL", "urllib3", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    store = WorldStore(Path(args.root))
    staging = masks_at_stop.staging_dir(store, args.world, args.session)
    try:
        sources = json.loads((staging / masks_at_stop.SOURCES_FILENAME).read_text(encoding="utf-8"))
        sources = dict(sources.get("sources") or {})
    except (OSError, ValueError, AttributeError):
        sources = {}
    try:
        summary = masks_at_stop.prefill(store, args.world, args.session, sources=sources)
    except Exception:  # noqa: BLE001 -- the stage computes whatever is missing
        logging.getLogger(__name__).exception("%s the worker failed", masks_at_stop.LOG)
        return 1
    print(json.dumps({k: summary.get(k) for k in ("state", "detail", "staged", "chunks", "fills",
                                                   "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
