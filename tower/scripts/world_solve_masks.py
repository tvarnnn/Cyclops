#!/usr/bin/env python
"""Prefill masks for solver images already prepared at Stop.

This child reads the session keyframes, camera and images. It masks an
immutable snapshot of those images in its own STAGE, and writes nothing else:
the parent promotes the stage's cache files after it has confirmed this
process exited 0 -- or, once it is confirmed dead, after the join's bound
killed it (manager 154 §1) -- and only the files that verify for the live
image bytes (`world_build_session.StopSolverMasks`). The final solve owns
image preparation, the live mask cache, `masks/` and every database operation.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
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
# `stop_masks/writer.p<pid>`: which process may write this stage. The parent writes it for
# the child it launched BEFORE that child runs a single instruction (it is created
# suspended), so a stage a dead builder left behind is attributable -- and sweepable once
# that pid is gone (`world_build_session.sweep_stale_stop_mask_stages`, C25x2 MED-3).
WRITER_MARKER_PREFIX = "writer.p"
# `stop_masks/s/progress`: the child's PROGRESS BEAT (C25f-FIX2 review L-2). The parent ends
# the join once the stage has shown no progress for its stall bound
# (`world_build_session.STOP_MASKS_STALL_FLOOR_S`), and finished files alone go silent for
# minutes while Grounding DINO runs over every image. So the child counts every image each
# model finishes -- the backends' own per-image `should_stop` check, and each cache file
# emitted -- and rewrites the count here at most once per PROGRESS_BEAT_S. It changes
# nothing the models compute: the check it answers always says "carry on".
PROGRESS_FILENAME = "progress"
PROGRESS_BEAT_S = 1.0

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


def writer_marker(workspace_root, pid: int) -> Path:
    """`stop_masks/writer.p<pid>`: the marker naming the process that may write the stage."""
    return stage_parent(workspace_root) / f"{WRITER_MARKER_PREFIX}{int(pid)}"


def writer_pids(parent: Path) -> list:
    """The pids the markers in `parent` (a `stop_masks/`) name, in name order."""
    pids = []
    for marker in sorted(Path(parent).glob(f"{WRITER_MARKER_PREFIX}*")):
        digits = marker.name[len(WRITER_MARKER_PREFIX):]
        if digits.isdigit():
            pids.append(int(digits))
    return pids


def _fresh_stage(workspace_root) -> Path:
    """An empty stage. One left behind (a parent that died before it could sweep)
    is moved aside inside `stop_masks/`, never reused, never under `transients/`."""
    stage = stage_path(workspace_root)
    if stage.exists():
        stage.rename(stage.with_name(f"q{uuid.uuid4().hex[:8]}"))
    stage.mkdir(parents=True, exist_ok=True)
    return stage


class _ProgressBeat:
    """Counts finished steps; writes the count to `path` at most once per PROGRESS_BEAT_S.
    A beat that cannot be written is skipped: it must never fail the pass."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.count = 0
        self._written = None

    def __call__(self) -> None:
        self.count += 1
        now = time.monotonic()
        if self._written is not None and now - self._written < PROGRESS_BEAT_S:
            return
        self._written = now
        try:
            self.path.write_text(str(self.count), encoding="ascii")
        except Exception:  # noqa: BLE001 -- a beat is never a reason to fail
            pass


class _BeatingBackend:
    """A mask backend that beats once per image it checks in and once per mask it emits.
    It hands the backend a `should_stop` that beats and then answers exactly what the
    caller's would (`ensure_solver_masks` passes none: always "carry on"), so the backend
    computes what it would have computed without it."""

    def __init__(self, backend, beat: _ProgressBeat):
        self._backend, self._beat = backend, beat
        self.component = getattr(backend, "component", None)

    def probe(self):
        return self._backend.probe()

    def run(self, items, params, emit, should_stop=None):
        def check() -> bool:
            self._beat()
            return bool(should_stop()) if should_stop is not None else False

        def emitted(*args, **kwargs):
            emit(*args, **kwargs)
            self._beat()

        return self._backend.run(items, params, emitted, should_stop=check)


def _beating(factory, beat: _ProgressBeat):
    """`factory` (a component -> backend callable), with every backend beating."""
    return lambda component: _BeatingBackend(factory(component), beat)


def prefill(root: Path, world_id: str, session_id: str) -> int:
    from tower.world_builder import global_solve, solve_masks, transients
    from tower.world_builder.store import WorldStore
    from tower.storage import read_json_closed

    store = WorldStore(root)
    workspace = global_solve.workspace_for(store, world_id, session_id)
    if not workspace.camera_path.is_file():
        # No committed solver camera yet: nothing to mask, and the final solve will mask
        # everything. Said in a RESULT like every other exit 0, so the parent promotes
        # nothing and reports no failure (C25f review LOW-6: it read a missing result as a
        # failed prefill).
        _write_result(_fresh_stage(workspace.root), {})
        return 0
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
        beat = _ProgressBeat(stage.root / PROGRESS_FILENAME)
        beat()
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
            stage, names, keyframe_ids=ids, shape=(camera.height, camera.width),
            backend_factory=_beating(transients.BACKEND_FACTORY, beat))
        complete = (result.available and not result.partial and not result.unmasked
                    and not result.cache_write_failed and not result.retries)
        _write_result(stage.root,
                      {name: hashlib.sha1(images[name]).hexdigest() for name in names},
                      masked=len(result.masked), cache_hits=result.cache_hits,
                      available=complete)
        return 0 if complete else 1
    _write_result(_fresh_stage(workspace.root), {})
    return 0


def _write_result(stage: Path, images: dict, *, masked: int = 0, cache_hits: int = 0,
                  available: bool = True) -> None:
    """The child's last word: the image bytes it masked (by SHA-1), and whether the pass
    was COMPLETE. The parent promotes nothing from a stage without one."""
    (Path(stage) / RESULT_FILENAME).write_text(json.dumps({
        "images": images, "masked": masked, "cache_hits": cache_hits,
        "available": available,
    }), encoding="utf-8")


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
