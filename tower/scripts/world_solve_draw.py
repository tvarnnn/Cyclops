#!/usr/bin/env python
"""Map one private, digest-checked consensus draw for a final solve."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import native libraries on the main thread, before any possible stdin watcher.
import numpy  # noqa: E402,F401
import pycolmap  # noqa: E402

from tower.storage import write_bytes_atomic  # noqa: E402
from tower.world_builder import global_solve  # noqa: E402


class _RecordingPycolmap:
    """pycolmap, with every exception a mapper raises remembered.

    `_map_candidate` turns a mapper's exception into an empty candidate, as the serial path
    does. A CHILD must not report that as its draw: a `MemoryError` (Windows commit
    exhaustion) or any other pycolmap failure here may be this process's alone. It exits
    `DRAW_CHILD_MAPPER_RAISED_EXIT` instead, and the parent re-maps the seed in-process and
    journals it (review W01F STD MED-1)."""

    def __init__(self, module):
        self._module = module
        self.raised = []

    def __getattr__(self, name):
        value = getattr(self._module, name)
        if name not in ("global_mapping", "incremental_mapping"):
            return value

        def mapper(*args, **kwargs):
            try:
                return value(*args, **kwargs)
            except BaseException as exc:
                self.raised.append(f"{name}: {type(exc).__name__}: {exc}")
                raise
        return mapper


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-digest", required=True)
    args = parser.parse_args(argv)

    with args.context.open("rb") as handle:
        workspace, keyframes, camera, input_digest, min_observations = pickle.load(handle)
    before = (global_solve.database_digest(args.database) or {}).get("content")
    if before is None:
        raise RuntimeError("private consensus database has no content digest")
    if before != args.expected_digest:
        write_bytes_atomic(args.output, lambda handle: pickle.dump(
            {"before": before, "after": before, "map_ms": 0, "candidate": None},
            handle, protocol=pickle.HIGHEST_PROTOCOL))
        return 0
    global_solve._quiet_pycolmap()
    started = time.perf_counter()
    recording = _RecordingPycolmap(pycolmap)
    candidate = global_solve._map_candidate(
        recording, args.database, workspace, args.sparse, keyframes,
        seed=args.seed, threads=1, input_digest=input_digest,
        min_image_observations=min_observations, camera=camera)
    if recording.raised:
        print("consensus draw child: the mapper raised, so this seed is re-mapped by the "
              "parent:\n  " + "\n  ".join(recording.raised), file=sys.stderr, flush=True)
        return global_solve.DRAW_CHILD_MAPPER_RAISED_EXIT
    map_ms = round((time.perf_counter() - started) * 1000, 3)
    after = (global_solve.database_digest(args.database) or {}).get("content")
    # Only here -- the mapper returned, none raised -- is the result `complete`
    # (`global_solve._child_partial`; review W01F-FIX Codex M5).
    result = global_solve.draw_child_result(candidate, seed=args.seed, before=before,
                                            after=after, map_ms=map_ms)
    write_bytes_atomic(args.output, lambda handle: pickle.dump(
        result, handle, protocol=pickle.HIGHEST_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
