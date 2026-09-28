#!/usr/bin/env python
"""Prefill masks for solver images already prepared at Stop.

This child reads the session keyframes, camera and images. Its mask routine
writes only the transient cache, COLMAP masks, and the transient index. The
final solve owns image preparation and every database operation.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.native_prewarm import prewarm_world_builder  # noqa: E402


def _stage_path(workspace) -> Path:
    path = workspace.root / "transients" / "s"
    return Path("\\\\?\\" + str(path)) if os.name == "nt" else path


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
        # One read per image: inference decodes these exact bytes, and the cache
        # key hashes them. This path is used only by the Stop child.
        images = {}
        for name in names:
            try:
                images[name] = (workspace.images_dir / name).read_bytes()
            except OSError:
                pass
        names = [name for name in names if name in images]
        stage = global_solve.SolveWorkspace(_stage_path(workspace))
        if stage.root.exists():
            stage.root.rename(stage.root.with_name(f"q{uuid.uuid4().hex[:8]}"))
        stage.root.mkdir(parents=True, exist_ok=True)
        stage.images_dir.mkdir(exist_ok=True)
        for name, data in images.items():
            (stage.images_dir / name).write_bytes(data)
        staged_cache = stage.root / "transients"
        staged_cache.mkdir(exist_ok=True)
        for source in (workspace.root / "transients").glob("*.npz"):
            target = staged_cache / source.name
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        result = solve_masks.ensure_solver_masks(
            stage, names, keyframe_ids=ids, shape=(camera.height, camera.width))
        complete = (result.available and not result.partial and not result.unmasked
                    and not result.cache_write_failed and not result.retries)
        (stage.root / "result.json").write_text(json.dumps({
            "images": {name: hashlib.sha1(images[name]).hexdigest() for name in names},
            "masked": len(result.masked), "cache_hits": result.cache_hits,
            "available": complete,
        }), encoding="utf-8")
        return 0 if complete else 1
    stage = _stage_path(workspace)
    if stage.exists():
        stage.rename(stage.with_name(f"q{uuid.uuid4().hex[:8]}"))
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "result.json").write_text(json.dumps({
        "images": {}, "masked": 0, "cache_hits": 0, "available": True,
    }), encoding="utf-8")
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
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    return prefill(Path(args.root), args.world, args.session)


if __name__ == "__main__":
    raise SystemExit(main())
