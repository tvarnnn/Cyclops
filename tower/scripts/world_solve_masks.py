#!/usr/bin/env python
"""Prefill masks for solver images already prepared at Stop.

This child reads the session keyframes, camera and images. Its mask routine
writes only the transient cache, COLMAP masks, and the transient index. The
final solve owns image preparation and every database operation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.native_prewarm import prewarm_world_builder  # noqa: E402


def prefill(root: Path, world_id: str, session_id: str) -> int:
    from tower.world_builder import global_solve, solve_masks
    from tower.world_builder.store import WorldStore
    from tower.storage import read_json_closed

    store = WorldStore(root)
    workspace = global_solve.workspace_for(store, world_id, session_id)
    if not workspace.camera_path.is_file():
        return 0  # no committed solver camera yet; the final solve will mask
    camera = global_solve.PinholeCamera.from_json_dict(
        read_json_closed(workspace.camera_path))
    keyframes = store.read_keyframes(world_id, session_id)
    ids = {global_solve.keyframe_image_name(k): k.keyframe_id for k in keyframes}
    names = [name for name in ids if (workspace.images_dir / name).is_file()]
    if names:
        result = solve_masks.ensure_solver_masks(
            workspace, names, keyframe_ids=ids, shape=(camera.height, camera.width))
        return 0 if result.available else 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=artifact_root_arg, required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args(argv)
    # No stdin watcher is installed here. Warm the native stack before any
    # future watcher could be added (tower/native_prewarm.py).
    prewarm_world_builder()
    return prefill(Path(args.root), args.world, args.session)


if __name__ == "__main__":
    raise SystemExit(main())
