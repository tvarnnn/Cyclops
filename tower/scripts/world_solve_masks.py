#!/usr/bin/env python
"""Prefill masks for solver images already prepared at Stop.

This child reads the session keyframes, camera and images. It masks an
immutable snapshot of those images in its own STAGE, and writes nothing else:
the parent promotes the stage's cache files after it has confirmed this
process exited 0 (`world_build_session.StopSolverMasks`). The final solve owns
image preparation, the live mask cache, `masks/` and every database operation.
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

# THE STAGE LIVES OUTSIDE `transients/` (C25f audit CL-L4). `transients/` is the live
# mask cache: the final solve reads it, and `world_refinish.SOLVE_COPY_BACK` copies it
# into every re-finish -- so a stage kept there (a full copy of the solver images plus
# hard links into the cache) would ride forward into every re-finish, the links turning
# into copies. `solve/<session>/stop_masks/` is read by nothing but the parent, which
# sweeps it on every exit it can confirm. The names are SHORT on purpose: the stage
# nests its own `transients/` cache, and a longer stage name put the cache's staging
# files past MAX_PATH under pytest's temporary directory.
STAGE_PARENT_DIRNAME = "stop_masks"
STAGE_DIRNAME = "s"
RESULT_FILENAME = "result.json"

# The child's CPU thread cap. Stage 0 measured the solver masks (Grounding DINO + SAM
# 2.1 + OneFormer: hand/phone arrays and the COLMAP PNGs) BIT-IDENTICAL under exactly
# this cap -- OMP / MKL / OPENBLAS = 2, torch and cv2 = 2 -- against default threads
# (RUN experiments/W0-STAGE0/STAGE0.md §6.6, row T: 200/200 arrays, 50/50 PNGs). It
# was NOT for DINOv2, which this child never runs.
CHILD_THREADS = 2


def _long_path(path) -> Path:
    """`path` with the Windows extended-length prefix, so the stage's nested cache
    is not cut at MAX_PATH. Absolute first: the prefix on a relative path is not a
    path. Unchanged on POSIX, and for a path that already carries a prefix."""
    if os.name != "nt":
        return Path(path)
    text = os.path.abspath(str(path))
    if text.startswith("\\\\"):
        return Path(text)       # already extended, or a UNC share: leave it as it is
    return Path("\\\\?\\" + text)


def stage_parent(workspace_root) -> Path:
    """`solve/<session>/stop_masks`: everything the Stop child ever writes."""
    return _long_path(Path(workspace_root) / STAGE_PARENT_DIRNAME)


def stage_path(workspace_root) -> Path:
    """The one stage a Stop child masks into. The parent and the child both call
    this; it is the single definition of the layout."""
    return stage_parent(workspace_root) / STAGE_DIRNAME


def _fresh_stage(workspace_root) -> Path:
    """An empty stage. One left behind (a parent that died before it could sweep)
    is moved aside inside `stop_masks/`, never reused, never under `transients/`."""
    stage = stage_path(workspace_root)
    if stage.exists():
        stage.rename(stage.with_name(f"q{uuid.uuid4().hex[:8]}"))
    stage.mkdir(parents=True, exist_ok=True)
    return stage


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
        stage = global_solve.SolveWorkspace(_fresh_stage(workspace.root))
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
        (stage.root / RESULT_FILENAME).write_text(json.dumps({
            "images": {name: hashlib.sha1(images[name]).hexdigest() for name in names},
            "masked": len(result.masked), "cache_hits": result.cache_hits,
            "available": complete,
        }), encoding="utf-8")
        return 0 if complete else 1
    stage = _fresh_stage(workspace.root)
    (stage / RESULT_FILENAME).write_text(json.dumps({
        "images": {}, "masked": 0, "cache_hits": 0, "available": True,
    }), encoding="utf-8")
    return 0


def _cap_threads() -> None:
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415
    cv2.setNumThreads(CHILD_THREADS)
    torch.set_num_threads(CHILD_THREADS)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=artifact_root_arg, required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args(argv)
    # No stdin watcher is installed here. Warm the native stack before any
    # future watcher could be added (tower/native_prewarm.py).
    prewarm_world_builder()
    _cap_threads()
    return prefill(Path(args.root), args.world, args.session)


if __name__ == "__main__":
    raise SystemExit(main())
