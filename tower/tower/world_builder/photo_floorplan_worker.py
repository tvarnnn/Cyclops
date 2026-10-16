"""FOW v1.2 photo floor plan: landing inputs, landing builder, side worker, published state.

Contract: tower/docs/contracts/WORLD-BUILDER-GUIDANCE-IMAGERY.md, "Photo floor plan
v1.2". Raster core: `tower.world_builder.photo_floorplan` (pure).

Every `CoverageWorker` (the status producer builds one) owns ONE `PhotoFloorplanWorker`
and, after it has published a NEW coverage receipt, hands it that landing with
`offer()` -- a constant-time enqueue. Everything below runs on this module's own daemon thread:

  - never on a status request, the capture path, the coverage worker's 250 ms
    budget or the imagery worker's 80 ms budget;
  - one landing at a time, with an 8 s WALL cap (`LANDING_WALL_CAP_S`) from the start
    of its job. The cap is cooperative: `check()` runs between components, between frames and around
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
are RAW camera pixels: `imagery_source` is "unredacted", and redaction is never an
implied property of these bytes.
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
    MIN_FRAME_PITCH_DEG,
    fallback_floor_plane,
    PhotoFrame,
    build_homography_raster,
    encode_tile,
    estimate_floor_plane,
    frame_gate,
    lower_band_centre,
)

logger = logging.getLogger(__name__)

MAX_COMPONENTS = 16
MANIFEST_MAX_BYTES = 8 * 1024
# A 1.5 s cap skipped 3 of 4 live landings under load (walk-8 live run). The job is a side thread and
# landings are >= 15 s apart, so 8 s cannot pile up; a newer landing still cancels the job in flight.
LANDING_WALL_CAP_S = 8.0
MAX_SELECTED = 64
TARGET_PIXEL_BUDGET = 256 * 144
# A solve point is used only with a track length >= MIN_TRACK_LENGTH and a
# reprojection error <= MAX_POINT_ERROR_PX.
MIN_TRACK_LENGTH = 3
MAX_POINT_ERROR_PX = 2.0
ARRAYS_MAX_BYTES = 64 * 1024 * 1024
IMAGERY_SOURCE = "unredacted"
DIAG_INPUTS = "_diag_inputs"
DIAG_MAX_WALKS = 8   # walks whose diagnostics records the worker keeps in memory (LRU)


PLANE_REUSE_MIN_SHARED = 10
PLANE_REUSE_MAX_RESID_FRAC = 0.05
PLANE_REUSE_MAX_LANDINGS = 2
"""Plane REUSE across landings. Each new solve has its own gauge, so the previous
landing's plane is carried over by a Sim(3) fitted (Umeyama) on the camera centres
of keyframes both landings placed: >= 10 shared, median residual <= 0.05 of the
previous camera height. The carried plane must still pass the camera checks against
THIS landing's cameras (>= 95% above, robust CV <= 0.25). Only an ESTIMATED plane is
carried (never a fallback), and a fresh fit is forced after 2 consecutive reuses, so
a plane is never more than 2 landings stale."""


REGRESS_MIN_RATIO = 0.25
REGRESS_MIN_PRIOR_OPAQUE = 0.02
# Escape valve: after this many CONSECUTIVE regressed skips for one session the next regressed
# landing publishes anyway -- a fresh but worse plan beats a plan frozen for the rest of the
# walk -- and its opacity becomes the new baseline.
REGRESS_MAX_CONSECUTIVE_SKIPS = 3
"""NEVER-REGRESS guard (live run e2e4: a landing published a 0.4%-opaque main room
over a 55% one, and the phone showed gray for ~5 min). A landing is skipped -- the
prior plan stays, with its prior date -- if any component it shares (same
`reference_segment`) with the last PUBLISHED landing comes out below 25% of that
component's published opaque fraction (when that was at least 2%), unless the
new fraction is at least 2% and its opaque raster area has not fallen. A tile
is never kept across landings piecemeal: a new solve changes the receipt basis."""


class LandingSkipped(Exception):
    """This landing publishes nothing; the prior dated plan stays."""


class LandingOverrun(LandingSkipped):
    """The landing wall cap elapsed (or the job was cancelled)."""


# ---------------------------------------------------------------------------
# landing inputs
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
    """(poses by keyframe id, registered placements by segment index) of `tree`,
    cached under `tree["_index"]` (a mutation of the caller's dict; harmless
    because the tree is a private copy per job)."""
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


def _umeyama(src, dst):
    """`(scale, R, t)` of the similarity with dst ~ scale * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S_, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ D @ Vt
    var = float(xs.var(0).sum())
    scale = float(np.trace(np.diag(S_) @ D) / var) if var > 0 else 0.0
    return scale, R, mu_d - scale * R @ mu_s


def plane_memory_reuses(stats):
    """The reuse count AFTER this landing: `_reuses_before` plus one."""
    return int(stats.get("_reuses_before", 0)) + 1


def reuse_plane(memory, entries):
    """The previous landing's estimated plane carried into this landing's gauge, or
    (None, reason)."""
    from tower.world_builder.photo_floorplan import (
        CAMERA_ABOVE_MIN_FRAC, CAMERA_HEIGHT_MAX_CV, FloorPlane)

    if not memory:
        return None, "no previous plane"
    if memory.get("reuses", 0) >= PLANE_REUSE_MAX_LANDINGS:
        return None, "refit due"
    prior = memory["plane"]
    now = {e[0]: e[1] for e in entries}
    shared = [k for k in memory["positions"] if k in now]
    if len(shared) < PLANE_REUSE_MIN_SHARED:
        return None, f"{len(shared)} shared keyframes"
    src = np.stack([memory["positions"][k] for k in shared])
    dst = np.stack([now[k] for k in shared])
    scale, R, t = _umeyama(src, dst)
    if not np.isfinite(scale) or scale <= 0:
        return None, "degenerate Sim(3)"
    h_prev = float(prior.stats["camera_height_median"])
    resid = float(np.median(np.linalg.norm((scale * (R @ src.T)).T + t - dst, axis=1)))
    if resid > PLANE_REUSE_MAX_RESID_FRAC * h_prev * scale:
        return None, f"Sim(3) residual {resid:.3f} too large"
    n_prev = np.asarray(prior.normal, dtype=np.float64)
    n = R @ n_prev
    p0 = scale * (R @ (-prior.offset * n_prev)) + t
    offset = -float(n @ p0)
    cams = np.stack([e[1] for e in entries])
    above = cams @ n + offset
    med = float(np.median(above))
    cv = float(1.4826 * np.median(np.abs(above - med)) / med) if med > 0 else float("inf")
    if med <= 0 or float(np.mean(above > 0)) < CAMERA_ABOVE_MIN_FRAC or cv > CAMERA_HEIGHT_MAX_CV:
        return None, "carried plane fails the camera checks"
    stats = dict(prior.stats, source="reused_sim3", camera_height_median=med,
                 camera_height_cv=cv, camera_above_fraction=float(np.mean(above > 0)),
                 reuse_shared=len(shared), reuse_resid=resid, reason="ok (reused)")
    return FloorPlane(ok=True, reason="ok (reused from the previous landing)",
                      normal=tuple(float(v) for v in n), offset=offset, stats=stats), "ok"


def homography_component(*, receipt_component, entries, tree, arrays, keyframe_index,
                         camera, images_dir, check=None, furniture_mask=None,
                         plane_memory=None, diag_out=None):
    """Place one displayed component's eligible keyframes, carry or estimate the
    floor plane, choose and load frames, and warp them onto the plane.

    Returns (RasterResult, FloorPlane, frames, stats). `plane_memory` (a dict, mutated:
    cleared or updated) carries the previous landing's plane for reuse. `diag_out` (a
    dict, diagnostics only) receives this component's bottom-band SfM point reader."""
    forward = np.asarray(receipt_component["forward_xyz"], np.float64)
    up = np.asarray(receipt_component["up_xyz"], np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    stats = {"eligible": len(entries)}
    ms = stats["ms"] = {}
    plane = None
    frames = []
    if entries:
        t_stage = time.perf_counter()
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
        ms["points"] = round((time.perf_counter() - t_stage) * 1000.0, 1)
        t_plane = time.perf_counter()
        stats["_reuses_before"] = (plane_memory or {}).get("reuses", 0)
        reused, reuse_reason = (reuse_plane(plane_memory, entries)
                                if plane_memory is not None else (None, "off"))
        stats["plane_reuse"] = reuse_reason
        if reused is not None:
            plane = reused
        else:
            rights = np.stack([np.cross(e[3], e[2]) for e in entries])
            plane = estimate_floor_plane(points_xyz=out_xyz[good], camera_positions=cams,
                                         up_xyz=up, forward_xyz=forward,
                                         bounds_xy=receipt_component["bounds_xy"], check=check,
                                         camera_rights=rights,
                                         camera_order=[keyframe_index.get(e[0], 0)
                                                       for e in entries])
        observations = np.asarray(arrays["observations"]).reshape(-1, 3)
        obs_xy = np.asarray(arrays["observation_xy"]).reshape(-1, 2)
        band_top = (1.0 - LOWER_IMAGE_FRACTION) * float(camera["height"])
        by_frame = np.argsort(observations[:, 0], kind="stable")
        starts = np.searchsorted(observations[by_frame, 0], np.arange(len(keyframe_index) + 1))

        def band_points(kid):
            ki = keyframe_index.get(kid)
            if ki is None or ki + 1 >= len(starts):
                return np.zeros((0, 2)), np.zeros((0, 3))
            rows = by_frame[starts[ki]:starts[ki + 1]]
            rows = rows[obs_xy[rows, 1] >= band_top]
            return obs_xy[rows], out_xyz[observations[rows, 2]]

        if diag_out is not None:
            # Diagnostics only (photo_floorplan_diagnostics' floor-level check), read after
            # the landing's disposition is final.
            diag_out["band_points"] = band_points

        if not plane.ok and plane.stats.get("reference_normal_xyz") is not None:
            # A sparse live solve: never go blank -- a level plane from the camera
            # trajectory and the drops the downward frames themselves observed.
            n_ref = np.asarray(plane.stats["reference_normal_xyz"])
            drops, drop_ids = [], []
            for e in entries:
                pitch = float(np.degrees(np.arcsin(np.clip(-float(e[3] @ n_ref), -1.0, 1.0))))
                if pitch < MIN_FRAME_PITCH_DEG:
                    continue
                _, pts = band_points(e[0])
                pts = pts[np.all(np.isfinite(pts), axis=1)]
                drops.append(float(e[1] @ n_ref) - pts @ n_ref)
                drop_ids.append(e[0])
            plane = fallback_floor_plane(failed=plane, camera_positions=cams,
                                         band_drops_by_frame=drops, frame_ids=drop_ids)
            stats["plane"] = plane.stats
            stats["plane_ok"] = plane.ok
            stats["plane_reason"] = plane.reason
        stats["plane_ms"] = ms["plane"] = round((time.perf_counter() - t_plane) * 1000.0, 1)
        t_stage = time.perf_counter()
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
            ms["gate"] = round((time.perf_counter() - t_stage) * 1000.0, 1)
            t_stage = time.perf_counter()
            for kid, _, (_, position, up_world, forward_world, _) in selected:
                if check is not None:
                    check()
                loaded = load_frame_image(images_dir, kid, camera)
                if loaded is None or kid not in keyframe_index:
                    gate_counts["image missing"] = gate_counts.get("image missing", 0) + 1
                    continue
                rgb, fx, fy, cx, cy = loaded
                ki = keyframe_index[kid]
                band_xy, pts = band_points(kid)
                keep = np.all(np.isfinite(pts), axis=1)
                # Observed pixel positions, mapped to the sample size (pixel centres).
                sx = rgb.shape[1] / float(camera["width"])
                sy = rgb.shape[0] / float(camera["height"])
                uv = (band_xy.astype(np.float64) + 0.5) * np.array([sx, sy]) - 0.5
                frames.append(PhotoFrame(kid=kid, position_xyz=tuple(position),
                                         forward_xyz=tuple(forward_world),
                                         up_xyz=tuple(up_world), fx=fx, fy=fy, cx=cx, cy=cy,
                                         color=rgb, order=ki, check_points_xyz=pts[keep],
                                         check_points_uv=uv[keep]))
            stats["admissible_frames"] = len(admissible)
            stats["selected_frames"] = len(selected)
            ms["load"] = round((time.perf_counter() - t_stage) * 1000.0, 1)
        stats["frame_gate_counts"] = gate_counts
    if plane is None:
        from tower.world_builder.photo_floorplan import FloorPlane
        plane = FloorPlane(ok=False, reason="no eligible keyframes")
        stats["plane_ok"], stats["plane_reason"] = False, plane.reason
    if plane_memory is not None and entries:
        source = plane.stats.get("source")
        if plane.ok and source in (None, "reused_sim3"):
            plane_memory.clear()
            plane_memory.update(plane=plane, positions={e[0]: e[1] for e in entries},
                                reuses=(plane_memory_reuses(stats) if source else 0))
        else:
            plane_memory.clear()
    t_warp = time.perf_counter()
    result = build_homography_raster(origin_xyz=receipt_component["origin_xyz"], up_xyz=up,
                                     forward_xyz=forward,
                                     bounds_xy=receipt_component["bounds_xy"], plane=plane,
                                     frames=frames, check=check,
                                     **({} if furniture_mask is None
                                        else {"furniture_mask": bool(furniture_mask)}))
    stats["warp_ms"] = ms["warp"] = round((time.perf_counter() - t_warp) * 1000.0, 1)
    return result, plane, frames, stats


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
                  check=None, plane_memory=None):
    """Build the photo floor plan of one coverage landing: returns (manifest block,
    {digest: (bytes, mime)}, per-component stats).

    WHAT: reads the landing's own solve arrays (`read_landing_arrays`, guarded by the
    stat of `solution.json`), takes up to `MAX_COMPONENTS` (16) receipt components in
    reference-segment order, places each component's eligible keyframes
    (`placed_entries`), runs `homography_component`, encodes one tile per component
    (`encode_tile`; a tile over the cap is omitted and logged), content-addresses it by
    SHA-256 and assembles the manifest block, whose identity is copied from the receipt
    and which is refused above `MANIFEST_MAX_BYTES` (8 KiB).

    Inputs: `receipt` (the coverage block), `eligible` ([(kid, ref)]), `tree` (the parsed
    files coverage computed from), `solution_stat`, `check` (cancellation and wall-cap
    hook), `plane_memory` (per-reference memory, mutated). `LandingSkipped` publishes
    nothing; `check` raises `LandingOverrun` on the wall cap or a cancellation.

    WHY: the stats carry `DIAG_INPUTS` references for the post-landing diagnostics;
    `arrays_ms` is charged to the first component only; coordinates are rounded to 1e-6
    to keep 16 components inside 8 KiB."""
    solve_dir = Path(root) / "worlds" / world_id / "solve" / session_id
    solution = tree["solution"]
    camera = solution.get("camera")
    if not isinstance(camera, dict) or not all(
            k in camera for k in ("fx", "fy", "cx", "cy", "width", "height")):
        raise LandingSkipped("the landing's solve records no camera")
    t_arrays = time.perf_counter()
    arrays = read_landing_arrays(solve_dir, solution_stat, solution, check)
    arrays_ms = round((time.perf_counter() - t_arrays) * 1000.0, 1)
    keyframe_index = {kid: i for i, kid in enumerate(solution["keyframe_ids"])}
    by_ref = placed_entries(tree, eligible)
    components, tiles, stats = [], {}, []
    receipt_components = sorted(receipt["components"], key=lambda c: c["reference_segment"])
    for rc in receipt_components[:MAX_COMPONENTS]:
        if check is not None:
            check()
        ref = rc["reference_segment"]
        entries = by_ref.get(ref, [])
        diag_out = {}
        result, plane, frames, cstats = homography_component(
            receipt_component=rc, entries=entries, tree=tree, arrays=arrays,
            keyframe_index=keyframe_index, camera=camera,
            images_dir=solve_dir / "images", check=check,
            plane_memory=(plane_memory.setdefault(ref, {})
                          if plane_memory is not None else None),
            diag_out=diag_out)
        # New-scene diagnostics (photo_floorplan_diagnostics): REFERENCES the worker reads
        # only after this landing's disposition is final; nothing here reads them back.
        cstats[DIAG_INPUTS] = {
            "plane": plane, "entries": entries, "receipt_component": rc,
            "band_points": diag_out.get("band_points"),
            "frames": [(f.kid, f.check_points_xyz) for f in frames],
            "warped_kids": result.stats.get("warped_kids"),
            "raster_bounds_xy": result.raster_bounds_xy, "raster_width": result.width}
        if check is not None:
            check()
        t_encode = time.perf_counter()
        tile = encode_tile(result.rgba)
        cstats.setdefault("ms", {})["encode"] = round((time.perf_counter() - t_encode) * 1000.0, 1)
        if not stats:
            cstats["ms"]["arrays"] = arrays_ms   # charged once, to the first component
        x0, y0, x1, y1 = result.raster_bounds_xy
        cstats.update(reference_segment=ref, opaque_fraction=result.opaque_fraction,
                      opaque_area=result.opaque_fraction * (x1 - x0) * (y1 - y0),
                      projected_keyframes=len(result.contributing_keyframes),
                      frames_warped=result.stats.get("frames_warped"),
                      frames_rejected=result.stats.get("frames_rejected"))
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


def opaque_by_ref(stats):
    return {c["reference_segment"]: float(c["opaque_fraction"]) for c in stats or ()
            if c.get("reference_segment") is not None and c.get("opaque_fraction") is not None}


def opaque_area_by_ref(stats):
    return {c["reference_segment"]: float(c["opaque_area"]) for c in stats or ()
            if c.get("reference_segment") is not None and c.get("opaque_area") is not None}


def regression(prior, stats, prior_area=None):
    """A description of the first regressed component, or '' (see REGRESS_MIN_RATIO)."""
    if not prior:
        return ""
    now_area = opaque_area_by_ref(stats)
    for ref, now in sorted(opaque_by_ref(stats).items()):
        before = prior.get(ref)
        if (before is not None and before >= REGRESS_MIN_PRIOR_OPAQUE
                and now < REGRESS_MIN_RATIO * before
                and not (now >= REGRESS_MIN_PRIOR_OPAQUE
                         and ref in now_area and ref in (prior_area or {})
                         and now_area[ref] >= prior_area[ref])):
            return f"ref {ref}: opaque {now:.3f} < {REGRESS_MIN_RATIO} x published {before:.3f}"
    return ""


def component_summary(stats):
    """One compact JSON line per landing for the disposition log: every landing logs
    why a plane or tile came out as it did, so an `ok` landing that projected nothing
    is never opaque."""
    rows = []
    for c in stats or ():
        plane = c.get("plane") or {}
        rows.append({
            "ref": c.get("reference_segment"), "eligible": c.get("eligible"),
            "points": c.get("points_quality"), "plane_ok": c.get("plane_ok"),
            "plane_reason": c.get("plane_reason"), "reference": plane.get("reference_normal"),
            "plane_source": plane.get("source", "estimated" if c.get("plane_ok") else None),
            "plane_reuse": c.get("plane_reuse"),
            "candidates": plane.get("candidate_points"), "inliers": plane.get("inliers"),
            "gates": c.get("frame_gate_counts"), "selected": c.get("selected_frames"),
            "warped": c.get("frames_warped"), "frames_rejected": c.get("frames_rejected"),
            "projected": c.get("projected_keyframes"),
            "opaque_area": (round(c["opaque_area"], 3) if c.get("opaque_area") is not None
                            else None),
            "opaque":(round(c["opaque_fraction"], 3) if c.get("opaque_fraction") is not None
                       else None),
            "tile": c.get("tile"), "bytes": c.get("bytes"), "ms": c.get("ms")})
    return json.dumps(rows, separators=(",", ":"), default=str)


@dataclasses.dataclass
class _DiagWalk:
    """One walk's diagnostics records and how many are in its JSONL."""

    records: list = dataclasses.field(default_factory=list)
    written: int = 0


class PhotoFloorplanWorker:
    """One daemon builds each newly published coverage landing's photo plan."""

    def __init__(self, root, *, latest=None, state=None,
                 wall_cap_s=LANDING_WALL_CAP_S, build=None):
        self.root = root
        self.wall_cap_s = float(wall_cap_s)
        self._latest = latest
        self._state = state if state is not None else state_for(root)
        self._build = build or build_landing
        import inspect
        self._build_takes_memory = "plane_memory" in inspect.signature(self._build).parameters
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._jobs = OrderedDict()
        self._running = None
        self.outcomes = {}
        self._plane_memory = {}   # (world, session) -> {reference_segment: memory}
        self._published_opaque = {}   # (world, session) -> {reference_segment: opaque}
        self._published_opaque_area = {}  # (world, session) -> {reference_segment: area}
        self._regress_streak = {}     # (world, session) -> consecutive regressed skips
        # New-scene diagnostics (photo_floorplan_diagnostics): bounded per-walk landing records
        # (pair -> _DiagWalk), written to the session when `session_write_gate` allows.
        self._diag_records = OrderedDict()
        self._diag_pending = None
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
                pending, self._diag_pending = self._diag_pending, None
                if pending is not None:
                    try:
                        self._diagnostics(*pending)
                    except Exception:  # noqa: BLE001 -- never stops the worker
                        logger.exception("world photo floorplan: diagnostics failed")

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
        trial = None
        regressed = ""
        try:
            if self._build_takes_memory:
                memory = self._plane_memory.get(pair, {})
                trial = {ref: dict(m) for ref, m in memory.items()}
                kwargs = {"plane_memory": trial}
            else:
                kwargs = {}
            block, tiles, stats = self._build(self.root, job["world_id"], job["session_id"],
                                              receipt, job["eligible"], job["tree"],
                                              job["solution_stat"], check=check, **kwargs)
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
                elif ((regressed := regression(self._published_opaque.get(pair), stats,
                                               self._published_opaque_area.get(pair)))
                      and self._regress_streak.get(pair, 0) < REGRESS_MAX_CONSECUTIVE_SKIPS):
                    self._regress_streak[pair] = self._regress_streak.get(pair, 0) + 1
                    disposition = (f"skipped: regressed ({regressed}; "
                                   f"{self._regress_streak[pair]}/{REGRESS_MAX_CONSECUTIVE_SKIPS})")
                elif not self._state.publish(*pair, block, tiles):
                    disposition = "skipped: not newer than the published landing (never re-dated)"
                else:
                    # Only a PUBLISHED landing's planes are carried to the next one.
                    if trial is not None:
                        self._plane_memory[pair] = trial
                    self._published_opaque[pair] = opaque_by_ref(stats)
                    self._published_opaque_area[pair] = opaque_area_by_ref(stats)
                    if regressed:
                        logger.warning("world photo floorplan: publishing a regressed landing after %d "
                                       "consecutive skips (escape valve): %s",
                                       self._regress_streak.get(pair, 0), regressed)
                    self._regress_streak[pair] = 0
        self.outcomes = {**self.outcomes, pair: {"identity": identity, "disposition": disposition,
                                                 "elapsed_ms": round(elapsed_ms, 1),
                                                 "components": stats}}
        log = logger.info if disposition == "ok" else logger.warning
        # `method=homography` stays in the line: scripts/photo_floorplan_diagnostics.py parses it.
        log("world photo floorplan: world=%s session=%s landing=%r method=homography "
            "elapsed_ms=%.1f disposition=%s components=%s", pair[0], pair[1], identity,
            elapsed_ms, disposition, component_summary(stats))
        # New-scene diagnostics run in `_run` AFTER this job is finished and its state released:
        # the disposition above is final, the wall cap and the manifest wait never see them.
        self._diag_pending = (pair, identity, receipt, disposition, bool(regressed), elapsed_ms,
                              stats, time.time())

    def _diagnostics(self, pair, identity, receipt, disposition, regressed, elapsed_ms, stats,
                     landed_at):
        record = self._diagnose(pair, receipt, disposition, regressed, elapsed_ms, stats,
                                landed_at)
        if record is not None:
            self._report_diagnostics(pair, identity, record)

    def _diagnose(self, pair, receipt, disposition, regressed, elapsed_ms, stats, landed_at):
        """The landing's diagnostics record (photo_floorplan_diagnostics), or None. Strips the
        build's `_diag_inputs` from `stats` either way; never raises."""
        record = None
        try:
            from tower.world_builder import photo_floorplan_diagnostics as diag
            record = diag.landing_record(
                world_id=pair[0], session_id=pair[1], receipt=receipt, disposition=disposition,
                published=disposition == "ok", escaped=disposition == "ok" and regressed,
                elapsed_ms=elapsed_ms, stats=stats, now=landed_at)
        except Exception:  # noqa: BLE001 -- a diagnostic never touches a landing
            logger.exception("world photo floorplan: diagnostics failed")
        for c in stats or ():
            if isinstance(c, dict):
                c.pop(DIAG_INPUTS, None)
        return record

    def _report_diagnostics(self, pair, identity, record):
        """One bounded log line per landing, plus the session's JSONL and per-walk summary.
        Never raises."""
        try:
            from tower.world_builder import photo_floorplan_diagnostics as diag
            logger.info("world photo floorplan diagnostics: world=%s session=%s landing=%r "
                        "record=%s", pair[0], pair[1], identity, diag._dumps(record))
            walk = self._diag_records.get(pair)
            if walk is None:
                walk = self._diag_records[pair] = _DiagWalk()
            self._diag_records.move_to_end(pair)
            while len(self._diag_records) > DIAG_MAX_WALKS:
                self._diag_records.popitem(last=False)
            if len(walk.records) >= diag.MAX_RECORDS:
                return
            walk.records.append(record)
            summary = diag.walk_summary(walk.records, world_id=pair[0], session_id=pair[1],
                                        generated_at=time.time())
            # Every record not yet in the file goes in with this write: a landing whose
            # write was skipped (`session_write_gate`) is appended by the next permitted one.
            written, why = diag.write_files(self.root, pair[0], pair[1],
                                            walk.records[walk.written:], summary)
            if written:
                walk.written = len(walk.records)
            if not written or why.startswith(diag.SUMMARY_NOT_REPLACED):
                logger.info("world photo floorplan diagnostics files skipped: world=%s "
                            "session=%s landing=%r: %s", pair[0], pair[1], identity, why)
        except Exception:  # noqa: BLE001 -- a diagnostic never touches a landing
            logger.exception("world photo floorplan: diagnostics report failed")
