"""FOW v1.2 slice 1 replay PROBE (Tower side), run OFFLINE on a finished replay.

Design: FOW-V12-PHOTO-FLOORPLAN-DESIGN-20261012.md, "Raster and evidence".
Wire:   303-v12-wire-contract.md.
Core:   tower.world_builder.photo_floorplan (pure; this script is the only
        place that touches disk, the depth-fit, and the coverage internals).

For each distinct `guidance.coverage` landing recorded in a replay run's
`status-pushes.jsonl`, and for each of that landing's displayed registered
components, this script:

  1. takes the landing's own receipt basis and (unpadded) bounds verbatim;
  2. reconstructs that landing's eligible, posed keyframes by re-running
     `guidance_coverage.compute_coverage` against the replay's
     `tree-snapshots/<geometry_revision>/` copy of the five files it reads,
     and recovers each keyframe's placed world position/up/forward by
     calling `guidance_coverage._placed` on the SAME (row, placement) pairs
     coverage used -- i.e. REUSING coverage's own geometry, not re-deriving
     it;
  3. intersects that set with the session's live-depth cache (`align.json`)
     by keyframe id;
  4. selects <= 64 spatially distributed cache-hit keyframes;
  5. fits each selected keyframe's cached MoGe prediction to THIS solve's
     own SfM observations (`dense.align_frame`), accepting only a positive,
     finite fit with enough clean (non-fill) anchors and an acceptable
     held-out residual;
  6. hands the fitted, placed, cache-image samples to
     `photo_floorplan.build_raster` and writes the RGBA PNG plus a gray-plate
     composite preview.

`--method homography` (fallback (3), no depth cache, no GPU) instead, for the
FINAL landing only (it needs the solve's 3-D points, kept only for the final
solve):

  a. the same eligible, placed keyframes (step 2 above);
  b. this component's solve points, mapped into the output frame by the same
     Sim3 algebra as `guidance_coverage._solution_relative`, quality-filtered
     (track length >= 3, reprojection error <= 2 px);
  c. `photo_floorplan.estimate_floor_plane` (lower envelope + camera check);
  d. downward-looking frames, <= 64 spread over where their bottom band meets
     the floor, read from the solve's own undistorted `solve/<s>/images`
     with `camera.json` intrinsics, area-resampled to <= 256x144;
  e. `photo_floorplan.build_homography_raster`, then `encode_tile`
     (PNG, else WebP with lossless alpha, else 384 px; <= 180 KiB or omitted).

KNOWN DATA-AVAILABILITY GAP (see PROBE-REPORT.md "open problems" for the
full discussion): step 5 needs this solve's own sparse 3-D observations
(`solution.npz`), which the replay harness only preserved for the FINAL
solve -- the five intermediate `solution-snapshots/*.json` are metadata
only. Earlier landings therefore get real cache-hit/selection numbers but
no fit/raster; this is reported, not hidden or faked.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np  # noqa: E402

from tower.world_builder import guidance_coverage as gc  # noqa: E402
from tower.world_builder.dense import align_frame, depth_from_prediction, project  # noqa: E402
from tower.world_builder.photo_floorplan import (  # noqa: E402
    KeyframeSample,
    LOWER_IMAGE_FRACTION,
    PhotoFrame,
    build_homography_raster,
    build_raster,
    encode_tile,
    estimate_floor_plane,
    frame_gate,
    lower_band_centre,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("fow_v12_probe")

MAX_SELECTED = 64
TARGET_PIXEL_BUDGET = 256 * 144
MIN_FIT_POINTS = 12
MAX_HELD_OUT_REL = 0.15
GRAY_PLATE = (0x70, 0x75, 0x7B)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# landings
# ---------------------------------------------------------------------------


def extract_landings(run_dir: Path):
    """Distinct (solved_at, geometry_revision, frame_revision) coverage
    receipts, in the order they were first pushed, each with its own
    `received_at` (the walk-clock "landing time" for the report/caption)."""
    landings = []
    seen_keys = set()
    path = run_dir / "status-pushes.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        push = json.loads(line)
        payload = push.get("envelope", {}).get("payload", {})
        guidance = payload.get("guidance")
        coverage = guidance.get("coverage") if guidance else None
        if not coverage:
            continue
        key = (coverage["solved_at"], coverage["geometry_revision"], coverage["frame_revision"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        landings.append({"key": key, "receipt": coverage,
                         "received_at": push.get("received_at")})
    return landings


def tree_snapshot_dir(run_dir: Path, geometry_revision: str) -> Path:
    return run_dir / "tree-snapshots" / geometry_revision


def load_tree_snapshot(run_dir: Path, geometry_revision: str):
    """The five files `compute_coverage` reads, as they were at this
    landing's geometry revision. Returns None if the snapshot is missing or
    incomplete (a failed/unverified capture)."""
    d = tree_snapshot_dir(run_dir, geometry_revision)
    names = ("solution.json", "manifest.json", "placements.json", "poses.json", "keyframes.jsonl")
    if not all((d / n).is_file() for n in names):
        return None
    try:
        solution = load_json(d / "solution.json")
        manifest = load_json(d / "manifest.json")
        placements = load_json(d / "placements.json")["placements"]
        poses = load_json(d / "poses.json")["poses"]
        keyframes = load_jsonl(d / "keyframes.jsonl")
    except (OSError, ValueError, KeyError):
        return None
    return dict(solution=solution, manifest=manifest, placements=placements,
               poses=poses, keyframes=keyframes)


def eligible_placed_keyframes(tree):
    """Re-run coverage's own eligibility+placement math (reused, not
    re-derived) and return {reference_segment: [(kid, position, up_world,
    forward_world, scale), ...]}."""
    captured = []
    block = gc.compute_coverage(
        solution=tree["solution"], manifest=tree["manifest"],
        placements=tree["placements"], poses=tree["poses"],
        keyframes=tree["keyframes"],
        geometry_revision=_revision_of(tree),
        computed_at=float(tree["manifest"]["global_solve"]["solved_at"]) + 0.01,
        eligible_out=captured)
    poses_by_id = {row["keyframe_id"]: row for row in tree["poses"]}
    placement_by_segment = {p["segment_index"]: p for p in tree["placements"]
                            if p.get("state") == "registered"}
    by_ref = {}
    for kid, ref in captured:
        row = poses_by_id.get(kid)
        placement = placement_by_segment.get(row["segment_index"]) if row else None
        if row is None or placement is None:
            continue
        try:
            position, up_world, forward_world = gc._placed(row, placement)
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
        scale = float(placement["scale"])
        by_ref.setdefault(ref, []).append(
            (kid, np.asarray(position, np.float64), np.asarray(up_world, np.float64),
             np.asarray(forward_world, np.float64), scale))
    return block, by_ref


def _revision_of(tree):
    from tower.results.world_builder import geometry_revision_from_manifest
    return geometry_revision_from_manifest(tree["manifest"])


# ---------------------------------------------------------------------------
# live-depth cache
# ---------------------------------------------------------------------------


class DepthCache:
    def __init__(self, dense_root: Path):
        self.root = dense_root / "live_depth"
        doc = load_json(self.root / "align.json")
        self.by_kid = {r["kid"]: r for r in doc.get("records", [])
                      if isinstance(r, dict) and r.get("kid")}
        self.imagery_source = doc.get("imagery_source")
        self.backend = doc.get("backend")

    def has(self, kid) -> bool:
        return kid in self.by_kid

    def load(self, kid):
        record = self.by_kid[kid]
        ki = int(record["ki"])
        pred = np.load(self.root / "work" / "depth" / f"{ki:05d}_pred.npy").astype(np.float64)
        fill_path = self.root / "work" / "depth" / f"{ki:05d}_fill.npy"
        fill = np.load(fill_path).astype(bool) if fill_path.is_file() else np.zeros_like(pred, bool)
        undist_path = self.root / "work" / "undist" / f"{ki:05d}.jpg"
        import cv2
        bgr = cv2.imread(str(undist_path))
        rgb = bgr[:, :, ::-1] if bgr is not None else np.zeros((*pred.shape, 3), np.uint8)
        return pred, fill, rgb


# ---------------------------------------------------------------------------
# spatial selection
# ---------------------------------------------------------------------------


def select_spatial(entries, forward, right, limit=MAX_SELECTED):
    """`entries`: [(kid, position, up, fwd, scale), ...], all cache hits for
    one component. Greedy farthest-point sampling in the component's own
    (x, y) projection, for spatial distribution across the room."""
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


# ---------------------------------------------------------------------------
# SfM depth fit (needs this solve's own sparse points -- see module docstring)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SolveArrays:
    camera: dict
    keyframe_index: dict
    poses: dict
    xyz: np.ndarray
    observations: np.ndarray
    observation_xy: np.ndarray


def load_solve_arrays(solve_session_dir: Path) -> "SolveArrays | None":
    meta_path = solve_session_dir / "solution.json"
    npz_path = solve_session_dir / "solution.npz"
    if not meta_path.is_file() or not npz_path.is_file():
        return None
    meta = load_json(meta_path)
    with np.load(npz_path) as arrays:
        xyz = np.asarray(arrays["xyz"], np.float64)
        observations = np.asarray(arrays["observations"], np.int64).reshape(-1, 3)
        observation_xy = np.asarray(arrays["observation_xy"], np.float64).reshape(-1, 2)
    keyframe_index = {kid: i for i, kid in enumerate(meta["keyframe_ids"])}
    return SolveArrays(camera=meta["camera"], keyframe_index=keyframe_index,
                       poses=meta["poses"], xyz=xyz, observations=observations,
                       observation_xy=observation_xy)


def fit_keyframe(kid: str, solve: SolveArrays, pred: np.ndarray, fill: np.ndarray):
    """(ok, a, b, n_clean, held_out_rel, reason)."""
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


# ---------------------------------------------------------------------------
# resizing to the design's <= 256x144 sample budget
# ---------------------------------------------------------------------------


def resize_sample(depth, valid, color, fx, fy, cx, cy):
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


# ---------------------------------------------------------------------------
# main per-landing/per-component pipeline
# ---------------------------------------------------------------------------


def process_component(*, component_receipt, cache: DepthCache, placed_entries,
                      solve_arrays, order_base, png_path, preview_path,
                      landing_label, is_final):
    t0 = time.perf_counter()
    ref = component_receipt["reference_segment"]
    forward = np.asarray(component_receipt["forward_xyz"], np.float64)
    up = np.asarray(component_receipt["up_xyz"], np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    bounds_xy = component_receipt["bounds_xy"]
    origin = component_receipt["origin_xyz"]

    eligible = placed_entries
    cache_hit_entries = [e for e in eligible if cache.has(e[0])]
    cache_hit_fraction = (len(cache_hit_entries) / len(eligible)) if eligible else 0.0
    selected = select_spatial(cache_hit_entries, forward, right)

    stats = dict(reference_segment=ref, eligible=len(eligible),
                cache_hits=len(cache_hit_entries), cache_hit_fraction=cache_hit_fraction,
                selected=len(selected), landing=landing_label)

    if solve_arrays is None:
        stats.update(fit_attempted=0, fit_success=0, fit_success_fraction=None,
                    opaque_fraction=None, cpu_ms=None, png_bytes=None,
                    reason="no SfM arrays for this landing (see open problems)")
        return stats, None

    samples = []
    fit_reasons = {}
    for i, (kid, position, up_world, forward_world, scale) in enumerate(selected):
        record = cache.by_kid[kid]
        pred, fill, rgb_full = cache.load(kid)
        ok, a, b, n_clean, ho, reason = fit_keyframe(kid, solve_arrays, pred, fill)
        fit_reasons[kid] = reason
        if not ok:
            continue
        full_depth = depth_from_prediction(pred, a, b, kind="depth")
        full_valid = np.isfinite(full_depth) & (full_depth > 1e-6) & ~fill
        cam = solve_arrays.camera
        depth_r, valid_r, color_r, fx, fy, cx, cy = resize_sample(
            full_depth, full_valid, rgb_full, cam["fx"], cam["fy"], cam["cx"], cam["cy"])
        samples.append(KeyframeSample(
            kid=kid, position_xyz=tuple(position), forward_xyz=tuple(forward_world),
            up_xyz=tuple(up_world), scale=scale, fx=fx, fy=fy, cx=cx, cy=cy,
            depth=depth_r, valid=valid_r, color=color_r, order=order_base + i))

    fit_attempted = len(selected)
    fit_success = len(samples)
    result = build_raster(origin_xyz=origin, up_xyz=up, forward_xyz=forward,
                          bounds_xy=bounds_xy, keyframes=samples)
    cpu_ms = (time.perf_counter() - t0) * 1000.0

    from PIL import Image
    img = Image.fromarray(result.rgba, mode="RGBA")
    img.save(png_path, format="PNG")
    png_bytes = png_path.stat().st_size

    _write_composite_preview(result, preview_path, component_ref=ref, label=landing_label)

    stats.update(fit_attempted=fit_attempted, fit_success=fit_success,
                fit_success_fraction=(fit_success / fit_attempted) if fit_attempted else None,
                opaque_fraction=result.opaque_fraction, cpu_ms=round(cpu_ms, 1),
                png_bytes=png_bytes, multi_view_cells=result.multi_view_cells,
                single_view_cells=result.single_view_cells,
                raster_bounds_xy=result.raster_bounds_xy,
                fit_failure_reasons={r: sum(1 for v in fit_reasons.values() if v == r)
                                     for r in set(fit_reasons.values()) if r != "ok"})
    return stats, result


def _write_composite_preview(result, path, *, component_ref, label):
    from PIL import Image, ImageDraw

    plate = Image.new("RGB", (result.width, result.height), GRAY_PLATE)
    photo = Image.fromarray(result.rgba, mode="RGBA")
    plate.paste(photo, (0, 0), photo)
    canvas = Image.new("RGB", (result.width, result.height + 24), GRAY_PLATE)
    canvas.paste(plate, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, result.height + 4), f"component {component_ref} - {label}", fill=(255, 255, 255))
    canvas.save(path, format="PNG")


# ---------------------------------------------------------------------------
# --method homography (fallback (3))
# ---------------------------------------------------------------------------

MIN_TRACK_LENGTH = 3
MAX_POINT_ERROR_PX = 2.0


def solve_to_output(tree, kid):
    """(M, scale, position_out, centre_solve, solve_component) for one eligible
    keyframe, such that X_out = position_out + scale * M @ (X_solve - centre_solve).

    The same algebra as `guidance_coverage._solution_relative` (solve world ->
    that keyframe's camera -> its segment-local frame -> the registered
    placement), REUSING coverage's `_rotate`/`_placed` rather than re-deriving
    the conventions."""
    index = tree.get("_index")
    if index is None:
        index = tree["_index"] = (
            {r["keyframe_id"]: r for r in tree["poses"]},
            {p["segment_index"]: p for p in tree["placements"] if p.get("state") == "registered"})
    solution_pose = tree["solution"]["poses"][kid]
    row = index[0][kid]
    placement = index[1][row["segment_index"]]
    R = np.asarray(solution_pose["rotation"], np.float64).reshape(3, 3)
    t = np.asarray(solution_pose["translation"], np.float64)
    centre = -R.T @ t
    M = np.stack([gc._rotate(placement["rotation_wxyz"], gc._rotate(row["rotation"], list(R[:, j])))
                  for j in range(3)], axis=1)
    position, _, _ = gc._placed(row, placement)
    return (M, float(placement["scale"]), np.asarray(position, np.float64), centre,
            int(solution_pose["component"]))


def _load_frame_image(images_dir: Path, kid: str, camera: dict):
    """The solve's own undistorted image, area-resampled to <= 256x144 pixels,
    with its intrinsics rescaled to match (pixel-centre convention)."""
    import cv2

    path = images_dir / f"{kid.split(':')[-1]}.jpg"
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


def process_component_homography(*, component_receipt, placed_entries, tree, solve_dir,
                                  npz, keyframe_index, camera, out_prefix, landing_label,
                                  run_label):
    t_wall = time.perf_counter()
    t_cpu = time.process_time()
    ref = component_receipt["reference_segment"]
    forward = np.asarray(component_receipt["forward_xyz"], np.float64)
    up = np.asarray(component_receipt["up_xyz"], np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    stats = dict(reference_segment=ref, landing=landing_label, eligible=len(placed_entries),
                 method="homography")
    if not placed_entries:
        stats["reason"] = "no eligible keyframes"
        return stats, None

    # (b) this component's solve points in the output frame.
    transforms = [solve_to_output(tree, e[0]) for e in placed_entries]
    M, scale, pos0, c0, solve_component = transforms[0]
    worst = 0.0
    for (kid, position, _, _, _), (_, _, _, centre, comp) in zip(placed_entries, transforms):
        if comp != solve_component:
            raise ValueError("eligible keyframes of one reference span two solve components")
        worst = max(worst, float(np.linalg.norm(pos0 + scale * M @ (centre - c0) - position)))
    stats["sim3_consistency_max"] = worst
    in_comp = np.asarray(npz["component"]) == solve_component
    good = (in_comp & (np.asarray(npz["track_length"]) >= MIN_TRACK_LENGTH)
            & (np.asarray(npz["error"]) <= MAX_POINT_ERROR_PX))
    xyz = np.asarray(npz["xyz"], np.float64)
    out_xyz = np.full_like(xyz, np.nan)
    out_xyz[good] = pos0 + scale * (M @ (xyz[good] - c0).T).T
    stats["points_component"] = int(in_comp.sum())
    stats["points_quality"] = int(good.sum())

    # (c) one floor plane.
    t_core = time.perf_counter()
    cams = np.stack([e[1] for e in placed_entries])
    plane = estimate_floor_plane(points_xyz=out_xyz[good], camera_positions=cams,
                                 up_xyz=up, forward_xyz=forward,
                                 bounds_xy=component_receipt["bounds_xy"])
    plane_ms = (time.perf_counter() - t_core) * 1000.0
    stats["plane"] = plane.stats
    stats["plane_ok"] = plane.ok
    stats["plane_reason"] = plane.reason

    # (d) downward-looking frames, spread over the floor they show.
    frames = []
    gate_counts = {}
    if plane.ok:
        admissible = []
        for e in placed_entries:
            ok, reason, _, _ = frame_gate(plane, e[1], e[3], e[2])
            gate_counts[reason] = gate_counts.get(reason, 0) + 1
            if not ok:
                continue
            centre = lower_band_centre(plane, e[1], e[3], e[2], camera["fx"], camera["fy"],
                                       camera["cx"], camera["cy"], camera["width"], camera["height"])
            if centre is None:
                gate_counts["band centre misses plane"] = gate_counts.get("band centre misses plane", 0) + 1
                continue
            admissible.append((e[0], centre, e))
        selected = select_spatial(admissible, forward, right)
        observations = np.asarray(npz["observations"])
        obs_xy = np.asarray(npz["observation_xy"])
        band_top = (1.0 - LOWER_IMAGE_FRACTION) * float(camera["height"])
        by_frame = np.argsort(observations[:, 0], kind="stable")
        starts = np.searchsorted(observations[by_frame, 0], np.arange(len(keyframe_index) + 1))
        for kid, _, (_, position, up_world, forward_world, _) in selected:
            loaded = _load_frame_image(solve_dir / "images", kid, camera)
            if loaded is None:
                gate_counts["image missing"] = gate_counts.get("image missing", 0) + 1
                continue
            rgb, fx, fy, cx, cy = loaded
            ki = keyframe_index[kid]
            rows = by_frame[starts[ki]:starts[ki + 1]]
            rows = rows[obs_xy[rows, 1] >= band_top]
            pts = out_xyz[observations[rows, 2]]
            pts = pts[np.all(np.isfinite(pts), axis=1)]
            frames.append(PhotoFrame(kid=kid, position_xyz=tuple(position),
                                     forward_xyz=tuple(forward_world), up_xyz=tuple(up_world),
                                     fx=fx, fy=fy, cx=cx, cy=cy, color=rgb, order=ki,
                                     check_points_xyz=pts))
        stats["admissible_frames"] = len(admissible)
        stats["selected_frames"] = len(selected)
    stats["frame_gate_counts"] = gate_counts

    # (e) warp, compose, encode.
    t_warp = time.perf_counter()
    result = build_homography_raster(origin_xyz=component_receipt["origin_xyz"], up_xyz=up,
                                     forward_xyz=forward,
                                     bounds_xy=component_receipt["bounds_xy"], plane=plane,
                                     frames=frames)
    warp_ms = (time.perf_counter() - t_warp) * 1000.0
    t_enc = time.perf_counter()
    tile = encode_tile(result.rgba)
    encode_ms = (time.perf_counter() - t_enc) * 1000.0
    cpu_ms = (time.process_time() - t_cpu) * 1000.0
    wall_ms = (time.perf_counter() - t_wall) * 1000.0

    from PIL import Image
    Image.fromarray(result.rgba, mode="RGBA").save(f"{out_prefix}_raster.png", format="PNG")
    tile_path = None
    if tile is not None:
        ext = "png" if tile["mime"] == "image/png" else "webp"
        tile_path = Path(f"{out_prefix}.{ext}")
        tile_path.write_bytes(tile["data"])
    _write_tile_preview(tile, result, Path(f"{out_prefix}_preview.png"),
                        caption=f"{run_label} component {ref} - homography - {landing_label}")

    stats.update(
        frames_warped=result.stats.get("frames_warped", 0),
        frames_rejected=result.stats.get("frames_rejected", {}),
        cells_rejected=result.stats.get("cells_rejected", {}),
        raster_reason=result.stats.get("reason"),
        projected_keyframes=len(result.contributing_keyframes),
        opaque_fraction=result.opaque_fraction,
        multi_view_cells=result.multi_view_cells, single_view_cells=result.single_view_cells,
        raster_bounds_xy=result.raster_bounds_xy,
        band_pixels_hitting_plane=result.stats.get("band_pixels_hitting_plane"),
        band_pixels_outside_bounds=result.stats.get("band_pixels_outside_bounds"),
        tile_path=str(tile_path) if tile_path else None,
        tile_bytes=tile["bytes"] if tile else None,
        tile_encoding=tile["encoding"] if tile else "omitted (> 180 KiB)",
        tile_size=[tile["width"], tile["height"]] if tile else None,
        plane_ms=round(plane_ms, 1), warp_ms=round(warp_ms, 1), encode_ms=round(encode_ms, 1),
        cpu_ms=round(cpu_ms, 1), wall_ms=round(wall_ms, 1),
        imagery_source="unredacted")

    # Diagnostic only: the runner-up plane, if the estimate was ambiguous.
    if plane.ok and plane.stats.get("ambiguous"):
        from tower.world_builder.photo_floorplan import FloorPlane
        alt_stats = dict(plane.stats)
        alt_n = np.asarray(plane.stats["alt_plane_normal"])
        alt_d = float(plane.stats["alt_plane_offset"])
        alt_stats["camera_height_median"] = float(np.median(cams @ alt_n + alt_d))
        alt = FloorPlane(ok=True, reason="ok", normal=tuple(alt_n), offset=alt_d, stats=alt_stats)
        alt_result = build_homography_raster(origin_xyz=component_receipt["origin_xyz"],
                                             up_xyz=up, forward_xyz=forward,
                                             bounds_xy=component_receipt["bounds_xy"],
                                             plane=alt, frames=frames)
        _write_tile_preview(None, alt_result, Path(f"{out_prefix}_ALTPLANE-diagnostic_preview.png"),
                            caption=f"DIAGNOSTIC runner-up plane, NOT the rule's choice - {run_label} c{ref}")
        stats["alt_plane_opaque_fraction"] = alt_result.opaque_fraction
    return stats, result


def _write_tile_preview(tile, result, path, *, caption):
    """The DECODED tile (what a client would show) over the #70757B plate."""
    import io

    from PIL import Image, ImageDraw

    if tile is not None:
        photo = Image.open(io.BytesIO(tile["data"])).convert("RGBA")
    else:
        photo = Image.fromarray(result.rgba, mode="RGBA")
    w, h = photo.size
    plate = Image.new("RGB", (w, h), GRAY_PLATE)
    plate.paste(photo, (0, 0), photo)
    canvas = Image.new("RGB", (w, h + 24), GRAY_PLATE)
    canvas.paste(plate, (0, 0))
    ImageDraw.Draw(canvas).text((4, h + 4), caption, fill=(255, 255, 255))
    canvas.save(path, format="PNG")


def run_homography(*, run_dir, world_dir, session_id, out_dir, run_label):
    solve_dir = world_dir / "solve" / session_id
    camera = load_json(solve_dir / "camera.json")
    meta = load_json(solve_dir / "solution.json")
    keyframe_index = {kid: i for i, kid in enumerate(meta["keyframe_ids"])}
    with np.load(solve_dir / "solution.npz") as arrays:
        npz = {k: np.asarray(arrays[k]) for k in
               ("xyz", "component", "track_length", "error", "observations", "observation_xy")}
    npz["observations"] = npz["observations"].reshape(-1, 3)
    npz["observation_xy"] = npz["observation_xy"].reshape(-1, 2)
    # Warm the PIL/WebP/cv2 imports once, as a resident side worker would
    # have them, so the first component's timing is not an import benchmark.
    import cv2  # noqa: F401
    encode_tile(np.random.default_rng(0).integers(0, 256, (64, 64, 4), dtype=np.uint8),
                target_bytes=1, cap_bytes=1 << 20)
    landings = extract_landings(run_dir)
    rows = []
    for idx, landing in enumerate(landings):
        label = f"landing{idx}" + ("-final" if idx == len(landings) - 1 else "")
        solved_at, geometry_revision, _ = landing["key"]
        if idx != len(landings) - 1:
            rows.append(dict(landing=label, reference_segment=None, solved_at=solved_at,
                             reason="non-final landing: no solution.npz (3-D points) kept"))
            continue
        if float(meta["solved_at"]) != float(solved_at):
            raise SystemExit(f"final landing solved_at {solved_at} != solve dir {meta['solved_at']}")
        tree = load_tree_snapshot(run_dir, geometry_revision)
        if tree is None:
            rows.append(dict(landing=label, reference_segment=None,
                             reason="tree snapshot missing for the final landing"))
            continue
        _, by_ref = eligible_placed_keyframes(tree)
        for component_receipt in landing["receipt"]["components"]:
            ref = component_receipt["reference_segment"]
            prefix = out_dir / f"{run_label}_{label}_component{ref}_homography"
            stats, _ = process_component_homography(
                component_receipt=component_receipt, placed_entries=by_ref.get(ref, []),
                tree=tree, solve_dir=solve_dir, npz=npz, keyframe_index=keyframe_index,
                camera=camera, out_prefix=prefix, landing_label=label, run_label=run_label)
            stats["solved_at"] = solved_at
            stats["geometry_revision"] = geometry_revision
            rows.append(stats)
            log.info("%s %s component %s: plane=%s opaque=%.3f frames=%s cpu=%.0fms bytes=%s",
                     run_label, label, ref, stats.get("plane_reason"),
                     stats.get("opaque_fraction") or 0.0, stats.get("frames_warped"),
                     stats.get("cpu_ms") or 0.0, stats.get("tile_bytes"))
    (out_dir / f"{run_label}-homography-stats.json").write_text(
        json.dumps(rows, indent=1, default=float), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("depth", "homography"), default="depth")
    parser.add_argument("--run-label", default=None,
                        help="homography: output file prefix (default: the run dir's name)")
    parser.add_argument("--run-dir", default=str(Path(
        r"C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\experiments\W8LOOP\runs\g2")))
    parser.add_argument("--data-root", default=str(Path(
        r"C:\Users\tvllo\Projects\Glasses-scratch\wbcpt\w8-g2-d\world_builder")))
    parser.add_argument("--out-dir", default=str(Path(
        r"C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\experiments\W8LOOP\p2probe")))
    parser.add_argument("--world-id", default="467e6db5e9664cd1b3d9ed0a4f071911")
    parser.add_argument("--session-id", default="856b0df35403423a999fd64082aaa1e2")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    world_dir = data_root / "worlds" / args.world_id
    if args.method == "homography":
        run_homography(run_dir=run_dir, world_dir=world_dir, session_id=args.session_id,
                       out_dir=out_dir, run_label=args.run_label or run_dir.name)
        return
    dense_root = world_dir / "dense" / args.session_id
    solve_session_dir = world_dir / "solve" / args.session_id

    cache = DepthCache(dense_root)
    landings = extract_landings(run_dir)
    log.info("found %d distinct coverage landings", len(landings))

    final_key = landings[-1]["key"] if landings else None
    solve_arrays_final = load_solve_arrays(solve_session_dir)

    report_rows = []
    order_base = 0
    for idx, landing in enumerate(landings):
        solved_at, geometry_revision, frame_revision = landing["key"]
        is_final = landing["key"] == final_key
        label = f"landing{idx}" + ("-final" if is_final else "")
        tree = load_tree_snapshot(run_dir, geometry_revision)
        if tree is None:
            report_rows.append(dict(landing=label, reference_segment=None,
                                    reason="tree snapshot missing/incomplete for this geometry_revision"))
            log.warning("%s: tree snapshot missing for revision %s", label, geometry_revision)
            continue
        try:
            block, by_ref = eligible_placed_keyframes(tree)
        except Exception as exc:  # noqa: BLE001 -- a probe, never crash the whole run
            report_rows.append(dict(landing=label, reference_segment=None,
                                    reason=f"recompute_coverage failed: {exc}"))
            log.exception("%s: recompute_coverage failed", label)
            continue
        receipt = landing["receipt"]
        solve_arrays = solve_arrays_final if is_final else None
        for component_receipt in receipt["components"]:
            ref = component_receipt["reference_segment"]
            entries = by_ref.get(ref, [])
            png_path = out_dir / f"{label}_component{ref}.png"
            preview_path = out_dir / f"{label}_component{ref}_preview.png"
            stats, _ = process_component(
                component_receipt=component_receipt, cache=cache, placed_entries=entries,
                solve_arrays=solve_arrays, order_base=order_base,
                png_path=png_path, preview_path=preview_path,
                landing_label=label, is_final=is_final)
            order_base += 1
            stats["solved_at"] = solved_at
            stats["geometry_revision"] = geometry_revision
            stats["received_at"] = landing["received_at"]
            report_rows.append(stats)
            log.info("%s component %s: %s", label, ref, stats)

    write_report(out_dir, report_rows)


def write_report(out_dir: Path, rows):
    lines = ["# PROBE-REPORT: FOW v1.2 slice 1, walk 8 / G2 replay\n",
            "One row per (landing, displayed component). `fit_success_fraction` and "
            "`opaque_fraction` are None where no SfM arrays were available for that "
            "landing (see PROBE's module docstring / the open problems note below).\n",
            "| landing | solved_at | ref | eligible | cache_hit_frac | selected | "
            "fit_success_frac | opaque_frac | cpu_ms | png_bytes | multi/single cells |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("reference_segment") is None and "reason" in r and "eligible" not in r:
            lines.append(f"| {r['landing']} | - | - | - | - | - | - | - | - | - | {r['reason']} |")
            continue
        chf = r.get("cache_hit_fraction")
        fsf = r.get("fit_success_fraction")
        of = r.get("opaque_fraction")
        mv, sv = r.get("multi_view_cells"), r.get("single_view_cells")
        lines.append(
            f"| {r['landing']} | {r.get('solved_at')} | {r['reference_segment']} | "
            f"{r.get('eligible')} | {chf:.2f} | {r.get('selected')} | "
            f"{'%.2f' % fsf if fsf is not None else 'n/a'} | "
            f"{'%.3f' % of if of is not None else 'n/a'} | "
            f"{r.get('cpu_ms') if r.get('cpu_ms') is not None else 'n/a'} | "
            f"{r.get('png_bytes') if r.get('png_bytes') is not None else 'n/a'} | "
            f"{mv if mv is not None else '-'}/{sv if sv is not None else '-'} |")
    (out_dir / "PROBE-REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", out_dir / "PROBE-REPORT.md")


if __name__ == "__main__":
    main()
