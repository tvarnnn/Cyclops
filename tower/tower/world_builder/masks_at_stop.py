"""Transient masks at Stop: the final build's masks, computed while the final solve runs.

Switch: `TOWER_WORLD_MASKS_AT_STOP` (`config.world_masks_at_stop_setting`), default off. Off,
nothing in this module runs and the builder and the stages are byte for byte as before.

WHY. After Stop the builder waits for the background solve (up to 120 s) and then runs the
final global solve, which on this Tower is CPU work (pycolmap has no CUDA build on Windows).
On walk 10.2 that was 556 s with the GPU idle. Then, after the solve, the surface stage
computes the transient masks of every fused keyframe and the appearance stage those of its
extra candidates (`transients.ensure_transient_masks`): 397 + 42 s of GPU on that walk, on
the critical path. Nearly everything a mask is made from is fixed at Stop -- the stored
keyframe JPEG, the redaction label `stop_session` just wrote, the session calibration, the
models -- so those masks can be computed in the idle window and only looked up afterwards.

TWO HALVES, AND THE CACHE IS NEVER WRITTEN BY THE FIRST.

1. `prefill` runs in a child the builder starts right after `stop_session` and stops when
   the final solve returns (`scripts/world_masks_at_stop.py`). It calls the transient stage
   UNCHANGED -- `ensure_transient_masks`, the same backends, batch 1, the same fp16 -- on
   chunks of keyframes, most-likely-needed first, with `depth_dir` = a STAGING directory,
   `dense/<sid>/work/masks_at_stop/`. The masks it writes there are the cache's own files
   under the cache's own key (`transients.component_key`), so the manifest's
   `frames_digest` of a build that uses them is the one it would have had anyway.
2. `promote` runs in the final surface stage after its depth stage, just before its
   transient step. A staged mask moves into the real cache ONLY when the final's own inputs
   are the staged ones: the same key (the keyframe at this index of the FINAL solve, the
   SHA-1 of its stored JPEG, the label, the rule, the models); the same calibration and
   camera size; the final depth record naming the same image bytes under `FILL_RULE`; and
   the final `_fill.npy` equal to the staged one. Anything else stays staged and the stage
   computes it exactly as it does today. Then `ensure_transient_masks` finds hits.

THE ONE INPUT NOT FIXED AT STOP IS THE FILL MASK. Grounding DINO + SAM drops a mask that
lies mostly inside redaction fill (`filter_instance_masks`), so the cached gdsam masks depend
on `_fill.npy`, which the final depth stage writes after the solve. The fill is a function of
the keyframe bytes and its raw frame only (no pose), and the depth stage takes it from the
live depth worker's sidecar, else from the last live depth stage, else computes it
(`surface_pipeline.ensure_depth_stage`, `dense_pipeline.run_depth_stage`). `prefill` stages
the fill the final is expected to use by the same three rules, and `promote` checks the
expectation against the file the final actually wrote: a wrong guess costs a recomputation,
never a different mask.

WHAT IS LEFT TO THE STAGE, AS TODAY: every keyframe the worker did not reach, every staged
mask the checks refuse, and every case the worker declines -- a label that is not trusted at
Stop (the re-redaction path's fill is not the stored one), the research bypass
(`raw-local-research`), a detector that is unavailable. Mode `off` stages nothing.

GPU. One child, models loaded one at a time inside `ensure_transient_masks` as in the stage
(at most ~2.6 GB on the card). Chunks of `CHUNK_FRAMES`: each chunk reloads the three models
(about 5 s on 10.2: Grounding DINO 2.9, SAM 0.6, OneFormer 1.6), and a stop in a chunk's
Grounding DINO pass loses that pass (its boxes are emitted only after SAM). 256 frames is
about 85 s of GPU: reload overhead ~6% and an expected loss at the stop of ~9 s.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from tower.world_builder import transients as T

logger = logging.getLogger(__name__)

SCHEMA = 1
STAGING_DIRNAME = "masks_at_stop"
INDEX_FILENAME = "index.json"
SOURCES_FILENAME = "sources.json"
CHUNK_FRAMES = 256

LOG = "[Tower][WorldBuilder][masks-at-stop]"


def staging_dir(store, world_id: str, session_id: str) -> Path:
    """`dense/<sid>/work/masks_at_stop`: inside the depth work, so `prune_intermediates`
    removes what is left of it with the rest of the work."""
    from tower.world_builder.dense_pipeline import dense_dir  # noqa: PLC0415

    return dense_dir(store, world_id, session_id) / "work" / STAGING_DIRNAME


def log_path(store, world_id: str, session_id: str) -> Path:
    """The worker's own log, `dense/<sid>/masks_at_stop.log`: beside the work, not in it, so
    it outlives the prune and says afterwards what the worker staged."""
    from tower.world_builder.dense_pipeline import dense_dir  # noqa: PLC0415

    return dense_dir(store, world_id, session_id) / "masks_at_stop.log"


def intrinsics_record(intrinsics) -> dict:
    """The calibration the undistortion maps are made from, as plain JSON."""
    out = {}
    for k in ("fx", "fy", "cx", "cy", "dist_coeffs", "calibrated_width", "calibrated_height"):
        v = getattr(intrinsics, k, None)
        out[k] = list(v) if isinstance(v, (tuple, list)) else v
    return out


def keyframe_order(keyframe_ids: list, solution) -> list:
    """Indices into `keyframe_ids` (the final solve's `ki`), most likely needed first.

    The final needs the keyframes it poses in its main component (the surface fuses the
    depth-gated ones, the appearance offers a few more). At Stop the best evidence is the
    latest background solution: (1) posed in its largest component, (2) newer than it,
    (3) posed in another of its components, (4) in it but unposed. Index order within each.
    No solution: index order."""
    if solution is None:
        return list(range(len(keyframe_ids)))
    known = set(solution.keyframe_ids)
    groups: tuple = ([], [], [], [])
    for ki, kid in enumerate(keyframe_ids):
        pose = solution.poses.get(kid)
        if kid not in known:
            groups[1].append(ki)
        elif not pose or pose.get("rotation") is None:
            groups[3].append(ki)
        elif int(pose.get("component", 0)) == 0:
            groups[0].append(ki)
        else:
            groups[2].append(ki)
    return [ki for g in groups for ki in g]


@dataclasses.dataclass
class _Camera:
    """What `compatible_live_records` reads of a solution: the final's index space and camera."""
    keyframe_ids: list
    camera: dict


def _expected_fills(store, world_id, session_id, keyframe_ids, camera, intrinsics, *,
                    imagery_source) -> dict:
    """`{ki: (fill file, image_sha1)}`: the `_fill.npy` the final depth stage is expected to
    REUSE for keyframe `ki` -- the live depth worker's when its record matches
    (`live_depth.compatible_live_records`, which wins: `offer_live_predictions` copies it over),
    else the last live depth stage's (`dense_pipeline.reusable_predictions`, with its
    prediction, fill and undistorted frame on disk). Read only."""
    import math  # noqa: PLC0415

    from tower.world_builder.dense import DenseParams  # noqa: PLC0415
    from tower.world_builder.dense_pipeline import (  # noqa: PLC0415
        dense_dir,
        depth_trust_now,
        reusable_predictions,
    )
    from tower.world_builder.live_depth import compatible_live_records  # noqa: PLC0415

    root = dense_dir(store, world_id, session_id)
    work = root / "work"
    out: dict = {}
    try:
        fov = round(math.degrees(2.0 * math.atan(int(camera["width"]) / (2.0 * float(camera["fx"])))), 6)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        fov = None
    prior = reusable_predictions(root / "align.json", DenseParams().backend, imagery_source, known_fov=fov)
    for ki, (kid, sha) in prior.items():
        if (0 <= ki < len(keyframe_ids) and keyframe_ids[ki] == kid
                and (work / "depth" / f"{ki:05d}_pred.npy").is_file()
                and (work / "depth" / f"{ki:05d}_fill.npy").is_file()
                and (work / "undist" / f"{ki:05d}.jpg").is_file()):
            out[ki] = (work / "depth" / f"{ki:05d}_fill.npy", sha)
    trust = depth_trust_now(store, world_id, session_id, imagery_source=imagery_source)
    live = compatible_live_records(store, world_id, session_id, _Camera(list(keyframe_ids), camera),
                                   intrinsics, root, imagery_source=imagery_source, trust=trust)
    for ki, (_kid, sha, old_ki) in live.items():
        out[ki] = (root / "live_depth" / "work" / "depth" / f"{int(old_ki):05d}_fill.npy", sha)
    return out


def _fresh_fill(store, world_id, session_id, kid, source_path, image_set, intrinsics, imagery_source):
    """`(fill_u, image_sha1)` as `run_depth_stage` computes a frame it does not reuse, for a
    trusted label (the only one the worker runs under): the stored bytes, the fill against the
    raw frame (`keyframe_image_bytes`), dilated 3x3 three times, else the shape-gated guess,
    remapped nearest-neighbour into the solve camera. (None, None) when the stage would
    refuse the frame. `promote` compares the result with the final's own file."""
    import cv2  # noqa: PLC0415

    from tower.world_builder.dense_pipeline import (  # noqa: PLC0415
        keyframe_image_bytes,
        redaction_fill_mask,
    )
    from tower.world_builder.global_solve import _undistort_maps  # noqa: PLC0415

    data, _origin, exact_fill = keyframe_image_bytes(
        store, world_id, session_id, kid, source_path, None, keyframes_are_redacted=True,
        image_set=image_set, imagery_source=imagery_source)
    if data is None:
        return None, None
    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        return None, None
    m1, m2, (x0, y0, rw, rh), _cam = _undistort_maps(intrinsics, raw.shape[1], raw.shape[0])
    fill = (cv2.dilate(exact_fill.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3).astype(bool)
            if exact_fill is not None else redaction_fill_mask(raw, None))
    fill_u = cv2.remap(fill.astype(np.uint8) * 255, m1, m2, cv2.INTER_NEAREST)[y0:y0 + rh, x0:x0 + rw] > 0
    return fill_u, hashlib.sha1(data).hexdigest()


def _save_atomic(path: Path, array) -> None:
    tmp = path.with_name(f"{path.name}.p{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def _write_json(path: Path, doc: dict) -> None:
    tmp = path.with_name(f"{path.name}.p{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _write_index(staging: Path, doc: dict) -> None:
    _write_json(staging / INDEX_FILENAME, doc)


def write_sources(staging: Path, sources: dict) -> None:
    """The builder's `keyframe_id -> raw frame path`, as `run_final` hands it to the final
    solve's `sources.json`: the worker reads raw frames exactly where the depth stage will."""
    _write_json(staging / SOURCES_FILENAME, {"sources": {k: str(v) for k, v in sources.items()}})


def prefill(store, world_id: str, session_id: str, *, sources: dict | None = None,
            should_stop=None, backend_factory=None, chunk: int = CHUNK_FRAMES) -> dict:
    """Stage the final build's transient masks in `staging_dir`, most-likely-needed first,
    until done or `should_stop()`. Never writes the real cache. Returns a summary dict
    (also the staging index). Raises nothing a caller must handle: a failure is recorded."""
    from tower.world_builder import appearance as A  # noqa: PLC0415
    from tower.world_builder.dense_pipeline import FILL_RULE  # noqa: PLC0415
    from tower.world_builder.global_solve import _undistort_maps, load_solution  # noqa: PLC0415
    from tower.world_builder.surface import SurfaceParams  # noqa: PLC0415

    t0 = time.time()
    staging = staging_dir(store, world_id, session_id)
    staging.mkdir(parents=True, exist_ok=True)
    surface_params = SurfaceParams()
    params = T.TransientParams(mode=surface_params.transient_detector)
    imagery_source = surface_params.imagery_source
    summary = {"schema": SCHEMA, "world_id": world_id, "session_id": session_id,
               "rule": params.rule_id(), "started_at": t0, "state": "running"}

    def done(state, detail=None, **more):
        summary.update(state=state, detail=detail, seconds=round(time.time() - t0, 3), **more)
        _write_index(staging, summary)
        logger.info("%s %s/%s: %s%s", LOG, world_id, session_id, state,
                    f" ({detail})" if detail else "")
        return summary

    if params.mode == T.MODE_OFF:
        return done("declined", "transient detection is off for the final build")
    session = store.read_session(world_id, session_id)
    intrinsics = session.intrinsics
    keyframes = store.read_keyframes(world_id, session_id)
    if not keyframes or intrinsics is None:
        return done("declined", "no keyframes or no calibration")
    policy = A.resolve_label_policy(store, world_id, session_id, imagery_source=imagery_source)
    if policy.raw or not policy.trusted:
        return done("declined", f"label {policy.effective!r}: only a trusted, redacted build is staged")
    first = keyframes[0]
    _m1, _m2, (_x, _y, rw, rh), cam = _undistort_maps(intrinsics, first.width, first.height)
    camera = {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy, "width": rw, "height": rh}
    keyframe_ids = [k.keyframe_id for k in keyframes]
    order = keyframe_order(keyframe_ids, load_solution(store, world_id, session_id))
    expected = _expected_fills(store, world_id, session_id, keyframe_ids, camera, intrinsics,
                               imagery_source=imagery_source)
    summary.update(intrinsics=intrinsics_record(intrinsics), camera=camera,
                   redaction_effective=policy.effective, keyframes=len(keyframe_ids),
                   expected_fills=len(expected), chunk=chunk, staged=0, chunks=0,
                   fills={"reused": 0, "fresh": 0, "none": 0})
    _write_index(staging, summary)
    image_set = policy.image_set
    sources = sources or {}
    computed = 0
    for start in range(0, len(order), chunk):
        if should_stop is not None and should_stop():
            return done("stopped", staged=computed)
        frames, records = [], {}
        for ki in order[start:start + chunk]:
            kid = keyframe_ids[ki]
            stored = image_set.directory / f"{kid.rsplit(':', 1)[-1]}.jpg"
            try:
                sha = hashlib.sha1(stored.read_bytes()).hexdigest()
            except OSError:
                continue
            fill = None
            got = expected.get(ki)
            if got is not None and got[1] == sha:
                try:
                    fill = np.load(got[0])
                    summary["fills"]["reused"] += 1
                except (OSError, ValueError):
                    fill = None
            if fill is None:
                fill, data_sha = _fresh_fill(store, world_id, session_id, kid, sources.get(kid),
                                             image_set, intrinsics, imagery_source)
                if fill is None or data_sha != sha:
                    summary["fills"]["none"] += 1
                    continue
                summary["fills"]["fresh"] += 1
            _save_atomic(staging / f"{ki:05d}_fill.npy", np.asarray(fill, bool))
            records[ki] = {"ki": ki, "kid": kid, "ok": True, "fill_rule": FILL_RULE, "image_sha1": sha}
            frames.append((ki, kid))
        report = T.ensure_transient_masks(
            store, world_id, session_id, frames, intrinsics=intrinsics, camera=camera,
            align_records=records, depth_dir=staging, params=params, policy=policy,
            backend_factory=backend_factory, should_stop=should_stop, donor=False)
        computed += report.computed
        summary["chunks"] += 1
        summary["staged"] = computed
        summary["gpu_peak_mb"] = max(summary.get("gpu_peak_mb") or 0, report.gpu_peak_mb or 0) or None
        _write_index(staging, summary)
        if report.state == T.STATE_STOPPED:
            return done("stopped", staged=computed)
        if report.state != T.STATE_OK:
            return done(report.state, report.detail, staged=computed)
        logger.info("%s %s/%s: chunk %d: %d keyframes staged (%d computed) in %.1f s since Stop",
                    LOG, world_id, session_id, summary["chunks"], len(report.keys), report.computed,
                    time.time() - t0)
    return done("done", staged=computed)


def _same_array(a: Path, b: Path) -> bool:
    try:
        x, y = np.load(a), np.load(b)
    except (OSError, ValueError):
        return False
    return x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)


def promote(store, world_id: str, session_id: str, *, depth_dir, align_records: dict, solution,
            intrinsics, params: T.TransientParams, policy) -> dict | None:
    """Move every staged mask whose inputs are the final's own into the real cache.

    None when nothing was staged. Otherwise a record of what happened, also logged:
    `promoted`, `already` (the cache held it under the final key), `refused` by reason. A
    refused mask stays where it is and the transient stage computes it as it always has."""
    from tower.world_builder import appearance as A  # noqa: PLC0415
    from tower.world_builder.dense_pipeline import FILL_RULE  # noqa: PLC0415

    depth_dir = Path(depth_dir)
    staging = depth_dir.parent / STAGING_DIRNAME
    try:
        index = json.loads((staging / INDEX_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    out = {"staged": 0, "promoted": 0, "already": 0, "refused": {}}

    def refuse(why):
        out["refused"][why] = out["refused"].get(why, 0) + 1

    files = sorted(staging.glob("*_transient.*.npz"))
    out["staged"] = len(files)
    camera = solution.camera or {}
    H, W = int(camera.get("height", 0)), int(camera.get("width", 0))
    staged_camera = index.get("camera") or {}
    if (policy.raw or index.get("intrinsics") != intrinsics_record(intrinsics)
            or (staged_camera.get("width"), staged_camera.get("height")) != (W, H)):
        for _ in files:
            refuse("camera" if not policy.raw else "imagery")
        _log_promote(world_id, session_id, out)
        return out
    undistorter = A.Undistorter(intrinsics, camera)
    source_sha1: dict = {}
    for path in files:
        try:
            ki = int(path.name.split("_", 1)[0])
            component = path.name.split(".")[1]
        except (ValueError, IndexError):
            refuse("name")
            continue
        if component not in params.components or not 0 <= ki < len(solution.keyframe_ids):
            refuse("not-this-build")
            continue
        kid = solution.keyframe_ids[ki]
        if ki not in source_sha1:
            src = A.keyframe_source(store, world_id, session_id, kid, ki, policy=policy, align_record=None,
                                    depth_dir=None, undistorter=undistorter, hash_only=True)
            source_sha1[ki] = None if src.refused else src.source_sha1
        if source_sha1[ki] is None:
            refuse("source")
            continue
        key = T.component_key(component, params, keyframe_id=kid, source_sha1=source_sha1[ki],
                              redaction_effective=policy.effective)
        target = T.cache_path(depth_dir, ki, component)
        if T.read_component(target, key, (H, W)) is not None:
            out["already"] += 1
            continue
        got = T.read_component(path, key, (H, W))
        if got is None:
            refuse("key")
            continue
        rec = align_records.get(ki)
        if not (isinstance(rec, dict) and rec.get("fill_rule") == FILL_RULE
                and rec.get("image_sha1") == got[2].get("image_sha1")):
            refuse("image")
            continue
        if not _same_array(staging / f"{ki:05d}_fill.npy", depth_dir / f"{ki:05d}_fill.npy"):
            refuse("fill")
            continue
        try:
            os.replace(path, target)
        except OSError:
            refuse("write")
            continue
        out["promoted"] += 1
    _log_promote(world_id, session_id, out)
    return out


def _log_promote(world_id, session_id, out) -> None:
    logger.info("%s %s/%s: %d staged masks, %d promoted into the cache, %d already there, refused %s",
                LOG, world_id, session_id, out["staged"], out["promoted"], out["already"],
                json.dumps(out["refused"], sort_keys=True))


def promote_quietly(*args, **kwargs) -> dict | None:
    """`promote`, never raising: a failure here is a stage that computes its masks as today."""
    try:
        return promote(*args, **kwargs)
    except Exception:  # noqa: BLE001 -- an optimisation, never a failure of the stage
        logger.exception("%s promoting the staged masks failed; the stage computes them", LOG)
        return None
