"""FOW v1.2 photo floor plan: landing inputs, landing builder, side worker, published state.

Design: FOW-V12-PHOTO-FLOORPLAN-DESIGN-20261012 ("Landing, live stretch and cost";
"Wire, identity and iOS"). Wire: mail 303, tower/docs/contracts/
WORLD-BUILDER-GUIDANCE-IMAGERY.md "Photo floor plan v1.2".
Raster core: `tower.world_builder.photo_floorplan` (pure).

OFF unless `TOWER_WORLD_PHOTO_FLOORPLAN` is exactly `on` (with
`TOWER_WORLD_GUIDANCE_COVERAGE=on`). Then the `CoverageWorker` owns ONE
`PhotoFloorplanWorker` and, after it has published a NEW coverage receipt, hands it
that landing with `offer()` -- a constant-time enqueue. Everything below runs on this
module's own daemon thread:

  - never on a status request, the capture path, the coverage worker's 250 ms
    budget or the imagery worker's 80 ms budget;
  - one landing at a time, with a 1.5 s WALL cap from the start of its job. The cap
    is cooperative: `check()` runs between components, between frames and around
    each read, so an overrun stops at the next check, publishes nothing, and the
    prior dated plan stays. A single uninterruptible step (one image decode, one
    `np.load` of the solve arrays) can overshoot the cap by that step;
  - a newer landing for the same world/session cancels the in-flight job;
    `shutdown()` cancels everything and returns without waiting.

Identity (exactly the v8 coverage receipt's): (world, session, solved_at,
geometry_revision, frame_revision). Publication is one dict replacement per
landing (manifest block and its tile index together, atomically). The state
refuses a landing that is not strictly newer (`solved_at`) than the one it holds,
so a local rebuild can never re-date or re-place a published plan, and a failed
solve (no new receipt) leaves the prior plan with its prior date. A job also
publishes nothing unless its receipt is still the coverage worker's latest.

PER-LANDING SfM DATA. The job reads `solve/<session>/solution.npz` -- the arrays of
the solve the landing came from -- and refuses them unless they provably are:
`solution.json` must still have the stat the coverage candidate named (before and
after the arrays are read), the arrays must not be newer than that file
(`write_solution` writes the arrays first, then the metadata), and the per-component
point counts in that `solution.json` must equal the arrays'. A newer solve being
written therefore skips this landing (its own landing will follow) rather than
mixing two solves.

IMAGERY: the solve's own undistorted solver images (`solve/<session>/images`), which
are RAW camera pixels: `imagery_source` is "unredacted". Privacy is deferred for the
walk-9 demo; redaction is a TODO, never an implied property of these bytes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from tower.world_builder import guidance_coverage as gc
from tower.world_builder.photo_floorplan import (
    LOWER_IMAGE_FRACTION,
    KeyframeSample,
    PhotoFrame,
    build_homography_raster,
    build_raster,
    encode_tile,
    estimate_floor_plane,
    frame_gate,
    lower_band_centre,
)

logger = logging.getLogger(__name__)

MAX_COMPONENTS = 16
MANIFEST_MAX_BYTES = 8 * 1024
# Manager 323 (walk-8 e2e, 2026-10-12): 1.5 s skipped 3 of 4 live landings under load. The job is a side thread and
# landings are >= 15 s apart, so 8 s cannot pile up; a newer landing still cancels the job in flight.
LANDING_WALL_CAP_S = 8.0
MAX_SELECTED = 64
TARGET_PIXEL_BUDGET = 256 * 144
MIN_TRACK_LENGTH = 3
MAX_POINT_ERROR_PX = 2.0
ARRAYS_MAX_BYTES = 64 * 1024 * 1024
IMAGERY_SOURCE = "unredacted"
METHOD_HOMOGRAPHY = "homography"
METHOD_DEPTH = "depth"
# Option (2)'s per-keyframe SfM depth fit (unchanged from the slice-1 probe).
MIN_FIT_POINTS = 12
MAX_HELD_OUT_REL = 0.15


class LandingSkipped(Exception):
    """This landing publishes nothing; the prior dated plan stays."""


class LandingOverrun(LandingSkipped):
    """The landing wall cap elapsed (or the job was cancelled)."""


# ---------------------------------------------------------------------------
# landing inputs (shared with scripts/fow_v12_probe.py -- one copy, not a fork)
# ---------------------------------------------------------------------------


def select_spatial(entries, forward, right, limit=MAX_SELECTED):
    """`entries`: [(kid, point_xyz, ...), ...]. Greedy farthest-point sampling in the
    component's own (x, y) projection of `entry[1]`, for spatial distribution."""
    if len(entries) <= limit:
        return entries
    xy = np.stack([[np.dot(e[1], forward), np.dot(e[1], right)] for e in entries])
    centroid = xy.mean(axis=0)
    start = int(np.argmin(np.sum((xy - centroid) ** 2, axis=1)))
    chosen = [start]
    min_dist = np.full(len(entries), np.inf)
    while len(chosen) < limit:
        last = xy[chosen[-1]]
        d = np.sum((xy - last) ** 2, axis=1)
        min_dist = np.minimum(min_dist, d)
        min_dist[chosen] = -1.0
        nxt = int(np.argmax(min_dist))
        if min_dist[nxt] < 0:
            break
        chosen.append(nxt)
    return [entries[i] for i in chosen]


def _tree_index(tree):
    index = tree.get("_index")
    if index is None:
        index = tree["_index"] = (
            {r["keyframe_id"]: r for r in tree["poses"]},
            {p["segment_index"]: p for p in tree["placements"] if p.get("state") == "registered"})
    return index


def solve_to_output(tree, kid):
    """(M, scale, position_out, centre_solve, solve_component) for one eligible
    keyframe, such that X_out = position_out + scale * M @ (X_solve - centre_solve).

    The same algebra as `guidance_coverage._solution_relative` (solve world -> that
    keyframe's camera -> its segment-local frame -> the registered placement),
    REUSING coverage's `_rotate`/`_placed` rather than re-deriving the conventions."""
    poses_by_id, placement_by_segment = _tree_index(tree)
    solution_pose = tree["solution"]["poses"][kid]
    row = poses_by_id[kid]
    placement = placement_by_segment[row["segment_index"]]
    R = np.asarray(solution_pose["rotation"], np.float64).reshape(3, 3)
    t = np.asarray(solution_pose["translation"], np.float64)
    centre = -R.T @ t
    M = np.stack([gc._rotate(placement["rotation_wxyz"], gc._rotate(row["rotation"], list(R[:, j])))
                  for j in range(3)], axis=1)
    position, _, _ = gc._placed(row, placement)
    return (M, float(placement["scale"]), np.asarray(position, np.float64), centre,
            int(solution_pose["component"]))


def placed_entries(tree, eligible):
    """{reference_segment: [(kid, position, up, forward, scale), ...]} for coverage's
    own eligible (kid, ref) pairs, placed with coverage's own `_placed`."""
    poses_by_id, placement_by_segment = _tree_index(tree)
    by_ref = {}
    for kid, ref in eligible:
        row = poses_by_id.get(kid)
        placement = placement_by_segment.get(row["segment_index"]) if row else None
        if row is None or placement is None:
            continue
        try:
            position, up_world, forward_world = gc._placed(row, placement)
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
        by_ref.setdefault(ref, []).append(
            (kid, np.asarray(position, np.float64), np.asarray(up_world, np.float64),
             np.asarray(forward_world, np.float64), float(placement["scale"])))
    return by_ref


def load_frame_image(images_dir: Path, kid: str, camera: dict):
    """The solve's own undistorted image, area-resampled to <= 256x144 pixels, with
    its intrinsics rescaled to match (pixel-centre convention). None if absent or
    not the solve camera's size."""
    import cv2

    path = Path(images_dir) / f"{kid.split(':')[-1]}.jpg"
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if (w, h) != (int(camera["width"]), int(camera["height"])):
        return None
    s = min(1.0, (TARGET_PIXEL_BUDGET / float(h * w)) ** 0.5)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    rgb = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    sx, sy = nw / w, nh / h
    return (np.ascontiguousarray(rgb), camera["fx"] * sx, camera["fy"] * sy,
            (camera["cx"] + 0.5) * sx - 0.5, (camera["cy"] + 0.5) * sy - 0.5)


def homography_component(*, receipt_component, entries, tree, arrays, keyframe_index,
                         camera, images_dir, check=None, furniture_mask=None):
    """Fallback (3) for one displayed component.

    Returns (RasterResult, FloorPlane, frames, stats)."""
    forward = np.asarray(receipt_component["forward_xyz"], np.float64)
    up = np.asarray(receipt_component["up_xyz"], np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    stats = {"eligible": len(entries)}
    plane = None
    frames = []
    if entries:
        transforms = [solve_to_output(tree, e[0]) for e in entries]
        M, scale, pos0, c0, solve_component = transforms[0]
        worst = 0.0
        for (_, position, _, _, _), (_, _, _, centre, comp) in zip(entries, transforms):
            if comp != solve_component:
                raise LandingSkipped("eligible keyframes of one reference span two solve components")
            worst = max(worst, float(np.linalg.norm(pos0 + scale * M @ (centre - c0) - position)))
        stats["sim3_consistency_max"] = worst
        in_comp = np.asarray(arrays["component"]) == solve_component
        good = (in_comp & (np.asarray(arrays["track_length"]) >= MIN_TRACK_LENGTH)
                & (np.asarray(arrays["error"]) <= MAX_POINT_ERROR_PX))
        xyz = np.asarray(arrays["xyz"], np.float64)
        out_xyz = np.full_like(xyz, np.nan)
        out_xyz[good] = pos0 + scale * (M @ (xyz[good] - c0).T).T
        stats["points_quality"] = int(good.sum())
        cams = np.stack([e[1] for e in entries])
        t_plane = time.perf_counter()
        plane = estimate_floor_plane(points_xyz=out_xyz[good], camera_positions=cams,
                                     up_xyz=up, forward_xyz=forward,
                                     bounds_xy=receipt_component["bounds_xy"], check=check)
        stats["plane_ms"] = round((time.perf_counter() - t_plane) * 1000.0, 1)
        stats["plane"] = plane.stats
        stats["plane_ok"] = plane.ok
        stats["plane_reason"] = plane.reason
        gate_counts = {}
        if plane.ok:
            admissible = []
            for e in entries:
                ok, reason, _, _ = frame_gate(plane, e[1], e[3], e[2])
                gate_counts[reason] = gate_counts.get(reason, 0) + 1
                if not ok:
                    continue
                centre = lower_band_centre(plane, e[1], e[3], e[2], camera["fx"], camera["fy"],
                                           camera["cx"], camera["cy"], camera["width"],
                                           camera["height"])
                if centre is None:
                    gate_counts["band centre misses plane"] = gate_counts.get(
                        "band centre misses plane", 0) + 1
                    continue
                admissible.append((e[0], centre, e))
            selected = select_spatial(admissible, forward, right)
            observations = np.asarray(arrays["observations"]).reshape(-1, 3)
            obs_xy = np.asarray(arrays["observation_xy"]).reshape(-1, 2)
            band_top = (1.0 - LOWER_IMAGE_FRACTION) * float(camera["height"])
            by_frame = np.argsort(observations[:, 0], kind="stable")
            starts = np.searchsorted(observations[by_frame, 0], np.arange(len(keyframe_index) + 1))
            for kid, _, (_, position, up_world, forward_world, _) in selected:
                if check is not None:
                    check()
                loaded = load_frame_image(images_dir, kid, camera)
                if loaded is None or kid not in keyframe_index:
                    gate_counts["image missing"] = gate_counts.get("image missing", 0) + 1
                    continue
                rgb, fx, fy, cx, cy = loaded
                ki = keyframe_index[kid]
                rows = by_frame[starts[ki]:starts[ki + 1]]
                rows = rows[obs_xy[rows, 1] >= band_top]
                pts = out_xyz[observations[rows, 2]]
                keep = np.all(np.isfinite(pts), axis=1)
                # Observed pixel positions, mapped to the sample size (pixel centres).
                sx = rgb.shape[1] / float(camera["width"])
                sy = rgb.shape[0] / float(camera["height"])
                uv = (obs_xy[rows].astype(np.float64) + 0.5) * np.array([sx, sy]) - 0.5
                frames.append(PhotoFrame(kid=kid, position_xyz=tuple(position),
                                         forward_xyz=tuple(forward_world),
                                         up_xyz=tuple(up_world), fx=fx, fy=fy, cx=cx, cy=cy,
                                         color=rgb, order=ki, check_points_xyz=pts[keep],
                                         check_points_uv=uv[keep]))
            stats["admissible_frames"] = len(admissible)
            stats["selected_frames"] = len(selected)
        stats["frame_gate_counts"] = gate_counts
    if plane is None:
        from tower.world_builder.photo_floorplan import FloorPlane
        plane = FloorPlane(ok=False, reason="no eligible keyframes")
        stats["plane_ok"], stats["plane_reason"] = False, plane.reason
    t_warp = time.perf_counter()
    result = build_homography_raster(origin_xyz=receipt_component["origin_xyz"], up_xyz=up,
                                     forward_xyz=forward,
                                     bounds_xy=receipt_component["bounds_xy"], plane=plane,
                                     frames=frames, check=check,
                                     **({} if furniture_mask is None
                                        else {"furniture_mask": bool(furniture_mask)}))
    stats["warp_ms"] = round((time.perf_counter() - t_warp) * 1000.0, 1)
    return result, plane, frames, stats


# -- option (2): the Exp1 live-depth cache, fit to this solve -----------------


class DepthCache:
    def __init__(self, dense_root: Path):
        self.root = Path(dense_root) / "live_depth"
        doc = json.loads((self.root / "align.json").read_text(encoding="utf-8"))
        self.by_kid = {r["kid"]: r for r in doc.get("records", [])
                       if isinstance(r, dict) and r.get("kid")}
        self.imagery_source = doc.get("imagery_source")
        self.backend = doc.get("backend")

    def has(self, kid) -> bool:
        return kid in self.by_kid

    def load(self, kid):
        import cv2

        record = self.by_kid[kid]
        ki = int(record["ki"])
        pred = np.load(self.root / "work" / "depth" / f"{ki:05d}_pred.npy").astype(np.float64)
        fill_path = self.root / "work" / "depth" / f"{ki:05d}_fill.npy"
        fill = np.load(fill_path).astype(bool) if fill_path.is_file() else np.zeros_like(pred, bool)
        bgr = cv2.imread(str(self.root / "work" / "undist" / f"{ki:05d}.jpg"))
        rgb = bgr[:, :, ::-1] if bgr is not None else np.zeros((*pred.shape, 3), np.uint8)
        return pred, fill, rgb


@dataclasses.dataclass
class SolveArrays:
    camera: dict
    keyframe_index: dict
    poses: dict
    xyz: np.ndarray
    observations: np.ndarray
    observation_xy: np.ndarray


def solve_arrays_from(meta: dict, arrays: dict) -> SolveArrays:
    return SolveArrays(camera=meta["camera"],
                       keyframe_index={kid: i for i, kid in enumerate(meta["keyframe_ids"])},
                       poses=meta["poses"], xyz=np.asarray(arrays["xyz"], np.float64),
                       observations=np.asarray(arrays["observations"], np.int64).reshape(-1, 3),
                       observation_xy=np.asarray(arrays["observation_xy"], np.float64).reshape(-1, 2))


def fit_keyframe(kid: str, solve: SolveArrays, pred: np.ndarray, fill: np.ndarray):
    """(ok, a, b, n_clean, held_out_rel, reason): the cached prediction fit to THIS
    solve's own sparse observations of that keyframe."""
    from tower.world_builder.dense import align_frame, project

    ki = solve.keyframe_index.get(kid)
    if ki is None:
        return False, None, None, 0, None, "keyframe not in this solve's horizon"
    mask = solve.observations[:, 0] == ki
    if int(mask.sum()) < MIN_FIT_POINTS:
        return False, None, None, int(mask.sum()), None, "too few sparse observations"
    pt_idx = solve.observations[mask, 2]
    X = solve.xyz[pt_idx]
    pose = solve.poses.get(kid)
    if pose is None:
        return False, None, None, 0, None, "no local pose for this keyframe"
    R = np.asarray(pose["rotation"], np.float64).reshape(3, 3)
    t = np.asarray(pose["translation"], np.float64)
    cam = solve.camera
    K = np.array([[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]])
    uv, zc = project(R, t, K, X)
    H, W = pred.shape
    g = (np.isfinite(zc) & (zc > 1e-3) & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
         & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1))
    if int(g.sum()) < MIN_FIT_POINTS:
        return False, None, None, int(g.sum()), None, "too few sparse points inside the frame"
    ui = np.clip(np.rint(uv[g, 0]).astype(int), 0, W - 1)
    vi = np.clip(np.rint(uv[g, 1]).astype(int), 0, H - 1)
    clean = ~fill[vi, ui]
    n_clean = int(clean.sum())
    if n_clean < MIN_FIT_POINTS:
        return False, None, None, n_clean, None, "too few clean (non-fill) anchors"
    disp_at_points = pred[vi[clean], ui[clean]].astype(np.float64)
    z_sfm = zc[g][clean]
    a, b, ho = align_frame(disp_at_points, z_sfm, kind="depth")
    if not (np.isfinite(a) and np.isfinite(b)):
        return False, a, b, n_clean, ho, "non-finite fit"
    if a <= 0:
        return False, a, b, n_clean, ho, "negative/zero scale"
    if ho is not None and ho > MAX_HELD_OUT_REL:
        return False, a, b, n_clean, ho, f"held-out residual {ho:.3f} > {MAX_HELD_OUT_REL}"
    return True, a, b, n_clean, ho, "ok"


def resize_sample(depth, valid, color, fx, fy, cx, cy):
    """Resample a fitted depth/valid/colour triple to the <= 256x144 budget."""
    import cv2

    h, w = depth.shape
    scale = min(1.0, (TARGET_PIXEL_BUDGET / float(h * w)) ** 0.5)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    if (new_w, new_h) == (w, h):
        return depth, valid, color, fx, fy, cx, cy
    depth_r = cv2.resize(depth.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    valid_r = cv2.resize(valid.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
    color_r = cv2.resize(color, (new_w, new_h), interpolation=cv2.INTER_AREA)
    sx, sy = new_w / w, new_h / h
    return depth_r.astype(np.float64), valid_r, color_r, fx * sx, fy * sy, cx * sx, cy * sy


def depth_component(*, receipt_component, entries, cache: DepthCache, solve: SolveArrays,
                    check=None):
    """Option (2) for one displayed component (non-default method)."""
    from tower.world_builder.dense import depth_from_prediction

    forward = np.asarray(receipt_component["forward_xyz"], np.float64)
    up = np.asarray(receipt_component["up_xyz"], np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    selected = select_spatial([e for e in entries if cache.has(e[0])], forward, right)
    samples = []
    for kid, position, up_world, forward_world, scale in selected:
        if check is not None:
            check()
        pred, fill, rgb = cache.load(kid)
        ok, a, b, _, _, _ = fit_keyframe(kid, solve, pred, fill)
        if not ok:
            continue
        depth = depth_from_prediction(pred, a, b, kind="depth")
        valid = np.isfinite(depth) & (depth > 1e-6) & ~fill
        cam = solve.camera
        d_r, v_r, c_r, fx, fy, cx, cy = resize_sample(depth, valid, rgb, cam["fx"], cam["fy"],
                                                      cam["cx"], cam["cy"])
        samples.append(KeyframeSample(kid=kid, position_xyz=tuple(position),
                                      forward_xyz=tuple(forward_world), up_xyz=tuple(up_world),
                                      scale=scale, fx=fx, fy=fy, cx=cx, cy=cy, depth=d_r,
                                      valid=v_r, color=c_r,
                                      order=solve.keyframe_index.get(kid, 0)))
    result = build_raster(origin_xyz=receipt_component["origin_xyz"], up_xyz=up,
                          forward_xyz=forward, bounds_xy=receipt_component["bounds_xy"],
                          keyframes=samples, check=check)
    return result, {"eligible": len(entries), "selected": len(selected), "fitted": len(samples)}


# ---------------------------------------------------------------------------
# one landing
# ---------------------------------------------------------------------------


def _stat(path: Path):
    st = path.stat()
    return (st.st_size, st.st_mtime_ns)


def read_landing_arrays(solve_dir: Path, solution_stat, solution_meta, check=None):
    """The solve arrays of THE landing's solve, or LandingSkipped (see module doc)."""
    json_path, npz_path = Path(solve_dir) / "solution.json", Path(solve_dir) / "solution.npz"
    expected = tuple(solution_stat)
    try:
        if _stat(json_path) != expected:
            raise LandingSkipped("solution.json is no longer the landing's solve")
        npz_before = _stat(npz_path)
        if npz_before[0] > ARRAYS_MAX_BYTES:
            raise LandingSkipped("solution.npz exceeds its size limit")
        if npz_before[1] > expected[1]:
            raise LandingSkipped("solution.npz is newer than the landing's solution.json")
        if check is not None:
            check()
        data = npz_path.read_bytes()
        if _stat(json_path) != expected or _stat(npz_path) != npz_before:
            raise LandingSkipped("the solve changed while its arrays were read")
    except OSError as exc:
        raise LandingSkipped(f"solve arrays unreadable: {exc}") from None
    if check is not None:
        check()
    names = ("xyz", "component", "track_length", "error", "observations", "observation_xy")
    try:
        with np.load(io.BytesIO(data)) as npz:
            arrays = {k: np.asarray(npz[k]) for k in names}
    except (KeyError, ValueError, OSError) as exc:
        raise LandingSkipped(f"solve arrays malformed: {exc}") from None
    arrays["observations"] = arrays["observations"].reshape(-1, 3)
    arrays["observation_xy"] = arrays["observation_xy"].reshape(-1, 2)
    components = solution_meta.get("components")
    if (isinstance(components, (list, tuple)) and components and
            all(isinstance(c, dict) and isinstance(c.get("index"), int) and
                isinstance(c.get("points"), int) for c in components)):
        counts = np.bincount(arrays["component"].astype(np.int64),
                             minlength=max(c["index"] for c in components) + 1)
        if any(c["index"] < 0 or int(counts[c["index"]]) != c["points"] for c in components):
            raise LandingSkipped("solve arrays do not match the landing's solution.json")
    obs = arrays["observations"]
    if len(obs) and (int(obs[:, 0].max()) >= len(solution_meta["keyframe_ids"]) or
                     int(obs[:, 2].max()) >= len(arrays["xyz"])):
        raise LandingSkipped("solve arrays index outside the landing's solve")
    return arrays


def build_landing(root, world_id, session_id, receipt, eligible, tree, solution_stat,
                  method=METHOD_HOMOGRAPHY, check=None):
    """(manifest block, {digest: (bytes, mime)}, per-component stats) for one landing."""
    solve_dir = Path(root) / "worlds" / world_id / "solve" / session_id
    solution = tree["solution"]
    camera = solution.get("camera")
    if not isinstance(camera, dict) or not all(
            k in camera for k in ("fx", "fy", "cx", "cy", "width", "height")):
        raise LandingSkipped("the landing's solve records no camera")
    arrays = read_landing_arrays(solve_dir, solution_stat, solution, check)
    keyframe_index = {kid: i for i, kid in enumerate(solution["keyframe_ids"])}
    by_ref = placed_entries(tree, eligible)
    cache = solve = None
    if method == METHOD_DEPTH:
        try:
            cache = DepthCache(Path(root) / "worlds" / world_id / "dense" / session_id)
        except (OSError, ValueError) as exc:
            raise LandingSkipped(f"no live-depth cache for the depth method: {exc}") from None
        solve = solve_arrays_from(solution, arrays)
    components, tiles, stats = [], {}, []
    receipt_components = sorted(receipt["components"], key=lambda c: c["reference_segment"])
    for rc in receipt_components[:MAX_COMPONENTS]:
        if check is not None:
            check()
        ref = rc["reference_segment"]
        entries = by_ref.get(ref, [])
        if method == METHOD_DEPTH:
            result, cstats = depth_component(receipt_component=rc, entries=entries,
                                             cache=cache, solve=solve, check=check)
        else:
            result, _, _, cstats = homography_component(
                receipt_component=rc, entries=entries, tree=tree, arrays=arrays,
                keyframe_index=keyframe_index, camera=camera,
                images_dir=solve_dir / "images", check=check)
        if check is not None:
            check()
        tile = encode_tile(result.rgba)
        cstats.update(reference_segment=ref, opaque_fraction=result.opaque_fraction,
                      projected_keyframes=len(result.contributing_keyframes))
        if tile is None:
            cstats["tile"] = "omitted (> 180 KiB)"
            stats.append(cstats)
            logger.warning("world photo floorplan: component %s omitted, tile > 180 KiB", ref)
            continue
        digest = hashlib.sha256(tile["data"]).hexdigest()
        cstats.update(tile=tile["encoding"], bytes=tile["bytes"])
        stats.append(cstats)
        tiles[digest] = (tile["data"], tile["mime"])
        components.append({
            "reference_segment": int(ref),
            "origin_xyz": [float(v) for v in rc["origin_xyz"]],
            "up_xyz": [float(v) for v in rc["up_xyz"]],
            "forward_xyz": [float(v) for v in rc["forward_xyz"]],
            # 1e-6 is far below a raster cell; it keeps 16 components inside 8 KiB.
            "raster_bounds_xy": [round(float(v), 6) for v in result.raster_bounds_xy],
            "width": int(tile["width"]),
            "height": int(tile["height"]),
            "digest": digest,
            "bytes": int(tile["bytes"]),
            "projected_keyframes": len(result.contributing_keyframes),
            "imagery_source": IMAGERY_SOURCE,
        })
    block = {"solved_at": receipt["solved_at"],
             "geometry_revision": receipt["geometry_revision"],
             "frame_revision": receipt["frame_revision"],
             "horizon_keyframes": receipt["horizon_keyframes"],
             "components": components}
    if len(manifest_bytes(block)) > MANIFEST_MAX_BYTES:
        raise LandingSkipped("manifest exceeds 8 KiB")
    return block, tiles, stats


def manifest_bytes(landed):
    return json.dumps({"version": 1, "landed": landed}, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def identity_of(block):
    """(solved_at, geometry_revision, frame_revision) of a receipt or manifest block."""
    if block is None:
        return None
    return (float(block["solved_at"]), block["geometry_revision"], block["frame_revision"])


# ---------------------------------------------------------------------------
# published state (read by the route adapter)
# ---------------------------------------------------------------------------


class PhotoFloorplanState:
    """Per-root memory only. A landing replaces its manifest block and tile index together."""

    def __init__(self):
        self._lock = threading.Lock()
        self._landed = {}
        self._inflight = {}

    def begin(self, world_id, session_id):
        with self._lock:
            self._inflight.setdefault((world_id, session_id), threading.Event())

    def finish(self, world_id, session_id):
        with self._lock:
            event = self._inflight.pop((world_id, session_id), None)
        if event is not None:
            event.set()

    def publish(self, world_id, session_id, block, tiles) -> bool:
        """Atomically replace this pair's landing, only with a STRICTLY newer solve:
        never re-dated, never re-placed, never rolled back."""
        pair = (world_id, session_id)
        with self._lock:
            previous = self._landed.get(pair)
            if previous is not None and float(previous[0]["solved_at"]) >= float(block["solved_at"]):
                return False
            self._landed[pair] = (block, dict(tiles))
            return True

    def landed(self, world_id, session_id, *, wait_s=0.0):
        """The published block, or None. With `wait_s`, a manifest request (never a
        status poll) waits up to that long for an in-flight landing job."""
        with self._lock:
            event = self._inflight.get((world_id, session_id))
        if wait_s > 0 and event is not None:
            event.wait(wait_s)
        with self._lock:
            row = self._landed.get((world_id, session_id))
            return row[0] if row else None

    def tile(self, world_id, session_id, digest):
        """(bytes, mime) of a tile in the CURRENT landing, else None (unknown or superseded)."""
        with self._lock:
            row = self._landed.get((world_id, session_id))
            hit = row[1].get(digest) if row else None
        if hit is not None and hashlib.sha256(hit[0]).hexdigest() == digest:
            return hit
        return None


_states = {}
_states_lock = threading.Lock()


def state_for(root):
    with _states_lock:
        return _states.setdefault(str(root), PhotoFloorplanState())


# ---------------------------------------------------------------------------
# the side worker
# ---------------------------------------------------------------------------


class PhotoFloorplanWorker:
    """One daemon builds each newly published coverage landing's photo plan."""

    def __init__(self, root, *, latest=None, state=None, method=None,
                 wall_cap_s=LANDING_WALL_CAP_S, build=None):
        if method is None:
            from tower.config import world_photo_floorplan_method_setting
            method = world_photo_floorplan_method_setting()
        self.root = root
        self.method = method
        self.wall_cap_s = float(wall_cap_s)
        self._latest = latest
        self._state = state if state is not None else state_for(root)
        self._build = build or build_landing
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._jobs = OrderedDict()
        self._running = None
        self.outcomes = {}
        self._thread = threading.Thread(target=self._run, name="world-photo-floorplan",
                                        daemon=True)
        self._thread.start()

    def offer(self, world_id, session_id, receipt, eligible, tree, solution_stat):
        """Constant-time enqueue from the coverage worker thread; never blocks on a job."""
        pair = (world_id, session_id)
        job = dict(world_id=world_id, session_id=session_id, receipt=receipt,
                   eligible=list(eligible or ()), tree=tree, solution_stat=solution_stat)
        with self._condition:
            if self._stop.is_set():
                return
            self._jobs.pop(pair, None)
            self._jobs[pair] = job
            if self._running is not None and self._running[0] == pair:
                self._running[1].set()   # a newer landing supersedes the in-flight one
            self._state.begin(world_id, session_id)
            self._condition.notify()

    def shutdown(self):
        """Cancel queued and in-flight work and return without waiting."""
        self._stop.set()
        with self._condition:
            pairs = list(self._jobs)
            self._jobs.clear()
            if self._running is not None:
                self._running[1].set()
            self._condition.notify_all()
        for world_id, session_id in pairs:
            self._state.finish(world_id, session_id)

    def idle(self) -> bool:
        with self._condition:
            return not self._jobs and self._running is None

    def _run(self):
        if os.name == "nt":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetThreadPriority(kernel32.GetCurrentThread(), -1)
            except (AttributeError, OSError):
                logger.warning("world photo floorplan: could not lower worker priority")
        while not self._stop.is_set():
            with self._condition:
                while not self._jobs and not self._stop.is_set():
                    self._condition.wait(timeout=0.5)
                if self._stop.is_set():
                    break
                pair, job = self._jobs.popitem(last=False)
                cancel = threading.Event()
                self._running = (pair, cancel)
            try:
                self._one(pair, job, cancel)
            except Exception:  # noqa: BLE001 -- the worker never dies on one landing
                logger.exception("world photo floorplan: landing job failed")
            finally:
                with self._condition:
                    self._running = None
                    more = pair in self._jobs
                if not more:
                    self._state.finish(*pair)

    def _one(self, pair, job, cancel):
        started = time.monotonic()
        deadline = started + self.wall_cap_s

        def check():
            if self._stop.is_set() or cancel.is_set():
                raise LandingOverrun("cancelled (superseded or shutdown)")
            if time.monotonic() >= deadline:
                raise LandingOverrun(f"exceeded the {self.wall_cap_s:.1f} s wall cap")

        receipt = job["receipt"]
        identity = identity_of(receipt)
        disposition, block, stats = "ok", None, None
        try:
            block, tiles, stats = self._build(self.root, job["world_id"], job["session_id"],
                                              receipt, job["eligible"], job["tree"],
                                              job["solution_stat"], method=self.method,
                                              check=check)
            check()
        except LandingSkipped as exc:
            disposition = f"skipped: {exc}"
        except Exception as exc:  # noqa: BLE001 -- one bad landing never stops the worker
            logger.exception("world photo floorplan: build failed")
            disposition = f"failed: {type(exc).__name__}: {exc}"
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if disposition == "ok":
            with self._condition:
                current = self._latest(*pair) if self._latest is not None else receipt
                if self._stop.is_set() or cancel.is_set():
                    disposition = "skipped: cancelled"
                elif identity_of(current) != identity:
                    disposition = "skipped: superseded by a newer coverage landing"
                elif time.monotonic() - started > self.wall_cap_s:
                    disposition = f"skipped: exceeded the {self.wall_cap_s:.1f} s wall cap"
                elif not self._state.publish(*pair, block, tiles):
                    disposition = "skipped: not newer than the published landing (never re-dated)"
        self.outcomes = {**self.outcomes, pair: {"identity": identity, "disposition": disposition,
                                                 "elapsed_ms": round(elapsed_ms, 1),
                                                 "components": stats}}
        log = logger.info if disposition == "ok" else logger.warning
        log("world photo floorplan: world=%s session=%s landing=%r method=%s elapsed_ms=%.1f "
            "disposition=%s", pair[0], pair[1], identity, self.method, elapsed_ms, disposition)
