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

FOOTPRINT BOUNDS (opt-in, TOWER_WORLD_PHOTO_FLOORPLAN_FOOTPRINT_BOUNDS exactly `on`,
homography method only): each component's `raster_bounds_xy` grows from the padded
camera square to the floor its selected frames can paint (see
`photo_floorplan.FOOTPRINT_TRIM_PERCENTILE`). Off, nothing below changes.

KEPT NOTICE (opt-in, TOWER_WORLD_PHOTO_FLOORPLAN_KEPT exactly `on`): a landing that publishes
nothing while an earlier plan is published records why in the state, and the manifest serves it
beside the untouched plan as `kept` (see `KEPT_REASONS`). Off, nothing is recorded or served.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import math
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
    KeyframeSample,
    PhotoFrame,
    build_homography_raster,
    build_raster,
    encode_tile,
    estimate_floor_plane,
    frame_gate,
    lower_band_centre,
    padded_square_bounds,
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
DIAG_INPUTS = "_diag_inputs"
DIAG_MAX_WALKS = 8   # walks whose diagnostics records the worker keeps in memory (LRU)
METHOD_DEPTH = "depth"
# Option (2)'s per-keyframe SfM depth fit (unchanged from the slice-1 probe).
MIN_FIT_POINTS = 12
MAX_HELD_OUT_REL = 0.15


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
# Escape valve (guard review MED, manager 336): after this many CONSECUTIVE regressed skips for one
# session the next regressed landing publishes anyway -- a fresh but worse plan beats a plan frozen
# for the rest of the walk -- and its opacity becomes the new baseline.
REGRESS_MAX_CONSECUTIVE_SKIPS = 3
"""NEVER-REGRESS guard (e2e4 10:37:47: a landing published a 0.4%-opaque main room
over a 55% one, and the phone showed gray for ~5 min). A landing is skipped -- the
prior plan stays, with its prior date -- if any component it shares (same
`reference_segment`) with the last PUBLISHED landing comes out below 25% of that
component's published opaque fraction (when that was at least 2%), unless the
new fraction is at least 2% and its opaque raster area has not fallen. A tile
is never kept across landings piecemeal: a new solve changes the receipt basis."""


REGRESS_V2_MIN_RETAINED = 0.5
REGRESS_V2_MIN_SHARED = PLANE_REUSE_MIN_SHARED
REGRESS_V2_MAX_RESID_FRAC = 0.25
REGRESS_V2_MAX_ANGLE_DEG = 5.0
REGRESS_V2_MIN_AGREE = 0.5
REGRESS_V2_MIN_SPREAD_H = 0.05
REGRESS_V2_MAX_SCALE_ERR = 0.05
REGRESS_V2_MIN_LATERAL_H = 0.5
REGRESS_V2_MIN_LATERAL_RATIO = 0.2
REGRESS_V2_MIN_COMMON_SHARE = 0.25
REGRESS_V2_MAX_CELLS = 65536
REGRESS_V2_MAX_DILATE_CELLS = 8
REGRESS_V2_EXTENT_GROWTH = 1.5
REGRESS_V2_EXTENT_MIN_NOVEL = 10
REGRESS_V2_EXTENT_NOVEL_MARGIN_FRAC = 0.5
REGRESS_V2_EXTENT_MIN_NET = 1.0
REGRESS_V2_ESCAPE_AFTER = REGRESS_MAX_CONSECUTIVE_SKIPS
REGRESS_V2_ESCAPE_MIN_PROJECTED = 8
REGRESS_V2_ESCAPE_PLANE_SOURCES = ("estimated", "reused_sim3")
REGRESS_V2_MAX_FROZEN_S = 300.0
"""NEVER-REGRESS GUARD V2 (TOWER_WORLD_PHOTO_FLOORPLAN_REGRESS_V2, exactly `on`; brief
CX-GEN-B-REGRESS, audit item 5). Each live solve has its own gauge, so neither an opaque
FRACTION (it falls whenever the camera square grows) nor an opaque AREA in solve units
(e2e6: 1,019 -> 146 between two good landings) compares two landings.

COMMON FOOTPRINT. Per shared `reference_segment`, the Sim(3) prev -> new is fitted on the
keyframes both landings placed (>= 10), robustly (`robust_sim3`): the ROTATION from the
cameras' own orientations (up + forward of every shared keyframe, Kabsch -- fixed for any
trajectory shape, a straight hallway included), scale and translation from their
positions; a trimmed refit, then the inliers within 0.25 x the new camera height and 5
degrees. >= 50% of the shared keyframes must agree, and the agreeing cameras must have
moved: a predicted scale error (median residual / RMS spread) <= 5% -- self-calibrating,
because camera height is itself a noisy gauge (room side / h ranged 1.2-3x on walk 8) --
and a numerical floor of 0.05 camera heights of spread; else the scale is unobservable
(e2e5 kf 520: spread 0.54 h, error 0.9%). A footprint without orientations falls back to
the positional Umeyama only when the agreeing cameras are genuinely 2-D: lateral RMS
spread (2nd singular value) >= 0.5 camera heights and >= 0.2 x the along-track one; a
hallway leaves the roll about its axis free, so it is refused (review HIGH-2).
Every opaque cell of the PUBLISHED raster is lifted onto its floor plane, carried into the
new basis and looked up in the new raster (dilated by the residual, 1..8 cells). Over the
published floor owned by agreeing cameras that falls inside the new raster square:

  retained = share of it the new landing still paints   (a cell-count ratio: scale-free)

A landing is REGRESSED if retained < 0.5, i.e. it no longer shows half of the floor the
phone already shows where both plans can show floor -- unless the extent GENUINELY
changed: the camera hull grew >= 1.5x (both hulls in the new gauge, so the ratio is
scale-free), >= 10 eligible keyframes stand outside the old hull by > 0.5 camera heights,
and the supported floor did not shrink (net = new area / (s^2 x published area) >= 1).
Without a usable common footprint the component falls back to the legacy rule above, so
V2 is never more permissive than today blind.

ESCAPE. After 3 consecutive regressed skips a regressed landing publishes only if every
regressed component meets a floor of its own: >= 2% opaque, a plane that is estimated or
reused (not the camera-drop fallback) and >= 8 projected keyframes. (A genuine extent
change -- growth AND net >= 1 -- is never regressed in the first place; growth alone
buys nothing: review HIGH-1.) Otherwise it is skipped as STALLED. The freeze is bounded:
once the plan has been held for 300 s since the first regressed skip, the next landing
with usable floor -- plane ok, >= 2% opaque, >= 8 projected keyframes in every regressed
component -- publishes. The bound deliberately accepts the camera-drop fallback plane:
since e2e4 it must pass the same camera checks as an estimated plane, and it is the only
plane a sparse or low-texture scene may get for minutes, so refusing it would make the
freeze unbounded exactly there; what hurt in e2e4 was a blank result (0.4%), which the
2% and 8-frame floors refuse. A blank landing never publishes. Every V2 decision is
logged and kept in `outcomes`."""


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


def _umeyama(src, dst):
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
                         plane_memory=None, footprint_bounds=False, diag_out=None):
    """Fallback (3) for one displayed component.

    Returns (RasterResult, FloorPlane, frames, stats). `diag_out` (a dict, diagnostics only)
    receives this component's bottom-band SfM point reader."""
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
    options = {} if furniture_mask is None else {"furniture_mask": bool(furniture_mask)}
    if footprint_bounds:
        # The plane (estimated, reused or fallback) is settled above; the footprint is
        # derived FROM it and never feeds back into plane estimation or reuse.
        options["footprint_bounds"] = True
    result = build_homography_raster(origin_xyz=receipt_component["origin_xyz"], up_xyz=up,
                                     forward_xyz=forward,
                                     bounds_xy=receipt_component["bounds_xy"], plane=plane,
                                     frames=frames, check=check, **options)
    stats["warp_ms"] = ms["warp"] = round((time.perf_counter() - t_warp) * 1000.0, 1)
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
                  method=METHOD_HOMOGRAPHY, check=None, plane_memory=None, footprints=None,
                  footprint_bounds=False):
    """(manifest block, {digest: (bytes, mime)}, per-component stats) for one landing.

    `footprints` (V2 only): a dict filled with {reference_segment: Footprint}.

    `footprint_bounds` (homography only) grows each raster past the padded camera
    square; each component's stats then also carry `guard_opaque_fraction`, its opaque
    AREA over the padded square's area -- the never-regress guard's ratio input, so
    bounds growth alone can never read as a regression (see `regression`)."""
    footprint_bounds = bool(footprint_bounds) and method != METHOD_DEPTH
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
        plane = None
        if method == METHOD_DEPTH:
            result, cstats = depth_component(receipt_component=rc, entries=entries,
                                             cache=cache, solve=solve, check=check)
        else:
            diag_out = {}
            result, plane, frames, cstats = homography_component(
                receipt_component=rc, entries=entries, tree=tree, arrays=arrays,
                keyframe_index=keyframe_index, camera=camera,
                images_dir=solve_dir / "images", check=check,
                plane_memory=(plane_memory.setdefault(ref, {})
                              if plane_memory is not None else None),
                **({"footprint_bounds": True} if footprint_bounds else {}),
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
        if footprints is not None:
            footprints[ref] = footprint_of(result, entries, plane, rc)
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
        if footprint_bounds:
            sx0, sy0, sx1, sy1 = padded_square_bounds(rc["bounds_xy"])
            cstats["guard_opaque_fraction"] = cstats["opaque_area"] / max(
                (sx1 - sx0) * (sy1 - sy0), 1e-12)
            cstats["footprint"] = result.stats.get("footprint")
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


def manifest_bytes(landed, kept=None):
    """The manifest body. `kept` (TOWER_WORLD_PHOTO_FLOORPLAN_KEPT only) is appended as an
    optional top-level key; without it the bytes are exactly the v1.2 manifest's."""
    doc = {"version": 1, "landed": landed}
    if kept is not None:
        doc["kept"] = kept
    return json.dumps(doc, separators=(",", ":"), allow_nan=False).encode("utf-8")


def identity_of(block):
    """(solved_at, geometry_revision, frame_revision) of a receipt or manifest block."""
    if block is None:
        return None
    return (float(block["solved_at"]), block["geometry_revision"], block["frame_revision"])


KEPT_REGRESSED = "regressed"
KEPT_STALLED = "stalled"
KEPT_WALL_CAP = "wall_cap"
KEPT_SUPERSEDED = "superseded"
KEPT_FAILED = "failed"
KEPT_NOT_NEWER = "not_newer"
KEPT_REASONS = (KEPT_REGRESSED, KEPT_STALLED, KEPT_WALL_CAP, KEPT_SUPERSEDED, KEPT_FAILED,
                KEPT_NOT_NEWER)
"""THE KEPT NOTICE (TOWER_WORLD_PHOTO_FLOORPLAN_KEPT, exactly `on`; manager mail 354). When the
newest landing processed for a session publishes no plan while an earlier plan IS published, the
served manifest carries

  "kept": {"for_landing": {solved_at, geometry_revision, frame_revision},   # the skipped landing
           "reason": one of KEPT_REASONS,
           "plan_landing": {solved_at, geometry_revision, frame_revision}}  # == landed's identity

so the phone can keep drawing the held plan for its own (newer) receipt instead of falling back
to fog after its 45 s hold. Reasons (closed set):

  regressed   the never-regress guard (legacy, or V2 "skip") refused it;
  stalled     V2: regressed past the skip count and the escape was refused;
  wall_cap    the landing job overran its wall cap;
  superseded  a newer coverage landing replaced it before it could publish (its own job follows);
  failed      its inputs could not be read or the build failed (any other skip or error);
  not_newer   its solve is not newer than the served plan's (a local rebuild): never re-dated.

The served plan (`landed`, its identity and tiles) is never touched. The block is replaced by the
next skip, removed atomically with the next PUBLISHED landing, absent while nothing is published,
and dropped (never the plan) if it would push the manifest past 8 KiB. Off: nothing is recorded."""


def _kept_identity(block):
    return {"solved_at": block["solved_at"], "geometry_revision": block["geometry_revision"],
            "frame_revision": block["frame_revision"]}


# ---------------------------------------------------------------------------
# published state (read by the route adapter)
# ---------------------------------------------------------------------------


class PhotoFloorplanState:
    """Per-root memory only. A landing replaces its manifest block and tile index together."""

    def __init__(self):
        self._lock = threading.Lock()
        self._landed = {}
        self._inflight = {}
        self._kept = {}   # (world, session) -> the KEPT notice; only with the KEPT switch on

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
            self._kept.pop(pair, None)   # a published landing ends any kept notice, atomically
            return True

    def keep(self, world_id, session_id, for_landing, reason) -> bool:
        """Record that `for_landing` (a receipt or block) published no plan and the served plan
        is kept (see KEPT_REASONS). Replaces any earlier notice. Nothing is recorded -- and an
        earlier notice is dropped -- while no plan is published, for the served plan's own
        identity, or if the notice would push the manifest past 8 KiB. Never touches the plan."""
        if reason not in KEPT_REASONS:
            raise ValueError(f"unknown kept reason {reason!r}")
        pair = (world_id, session_id)
        with self._lock:
            row = self._landed.get(pair)
            if row is None or identity_of(for_landing) == identity_of(row[0]):
                self._kept.pop(pair, None)
                return False
            notice = {"for_landing": _kept_identity(for_landing), "reason": reason,
                      "plan_landing": _kept_identity(row[0])}
            if len(manifest_bytes(row[0], notice)) > MANIFEST_MAX_BYTES:
                self._kept.pop(pair, None)
                logger.warning("world photo floorplan: kept notice dropped for world=%s "
                               "session=%s: the manifest would exceed 8 KiB", world_id, session_id)
                return False
            self._kept[pair] = notice
            return True

    def _wait_inflight(self, world_id, session_id, wait_s):
        with self._lock:
            event = self._inflight.get((world_id, session_id))
        if wait_s > 0 and event is not None:
            event.wait(wait_s)

    def landed(self, world_id, session_id, *, wait_s=0.0):
        """The published block, or None. With `wait_s`, a manifest request (never a
        status poll) waits up to that long for an in-flight landing job."""
        self._wait_inflight(world_id, session_id, wait_s)
        with self._lock:
            row = self._landed.get((world_id, session_id))
            return row[0] if row else None

    def served(self, world_id, session_id, *, wait_s=0.0):
        """(published block or None, kept notice or None), read together under one lock so a
        notice is never served beside a plan it does not describe. Waits as `landed` does."""
        self._wait_inflight(world_id, session_id, wait_s)
        with self._lock:
            pair = (world_id, session_id)
            row = self._landed.get(pair)
            return (row[0] if row else None), (self._kept.get(pair) if row else None)

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
    """The guard's per-component opaque fraction. With footprint bounds it is the
    square-normalised `guard_opaque_fraction` (opaque area / padded-square area), which
    is exactly `opaque_fraction` whenever the raster IS the padded square."""
    out = {}
    for c in stats or ():
        value = c.get("guard_opaque_fraction", c.get("opaque_fraction"))
        if c.get("reference_segment") is not None and value is not None:
            out[c["reference_segment"]] = float(value)
    return out


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


@dataclasses.dataclass
class Footprint:
    """One component's painted floor, in its own solve gauge (V2 only)."""
    positions: dict          # kid -> camera position (output frame) of the eligible keyframes
    up_xyz: tuple
    forward_xyz: tuple
    raster_bounds_xy: tuple  # the padded square the raster covers
    opaque: np.ndarray       # bool (height, width); row = right index, col = forward index
    plane_normal: tuple = None
    plane_offset: float = None
    camera_height: float = None
    origin_xyz: tuple = None
    axes: dict = None        # kid -> (up, forward) unit vectors (output frame); None = unknown


def footprint_of(result, entries, plane, receipt_component):
    """The V2 footprint of one built component (None if it cannot be described)."""
    try:
        ok = plane is not None and plane.ok
        height = float(plane.stats.get("camera_height_median") or 0.0) if ok else 0.0
        return Footprint(
            positions={e[0]: np.asarray(e[1], np.float64) for e in entries},
            up_xyz=tuple(float(v) for v in receipt_component["up_xyz"]),
            forward_xyz=tuple(float(v) for v in receipt_component["forward_xyz"]),
            raster_bounds_xy=tuple(float(v) for v in result.raster_bounds_xy),
            opaque=np.asarray(result.rgba)[:, :, 3] > 0,
            plane_normal=tuple(float(v) for v in plane.normal) if ok else None,
            plane_offset=float(plane.offset) if ok else None,
            camera_height=height if np.isfinite(height) and height > 0 else None,
            origin_xyz=tuple(float(v) for v in receipt_component["origin_xyz"]),
            axes={e[0]: (np.asarray(e[2], np.float64), np.asarray(e[3], np.float64))
                  for e in entries})
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None


def _basis(fp):
    up = np.asarray(fp.up_xyz, np.float64)
    forward = np.asarray(fp.forward_xyz, np.float64)
    up, forward = up / np.linalg.norm(up), forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    return up, forward, right / np.linalg.norm(right)


def _grid(fp):
    h, w = fp.opaque.shape
    x0, y0, x1, y1 = fp.raster_bounds_xy
    return x0, y0, (x1 - x0) / max(w - 1, 1), (y1 - y0) / max(h - 1, 1), w, h


def _cells_to_floor(fp, rows, cols):
    """3-D floor points (own gauge) of raster cells: on the plane, else at the origin height."""
    up, forward, right = _basis(fp)
    x0, y0, sx, sy, _, _ = _grid(fp)
    x = x0 + cols * sx
    y = y0 + rows * sy
    if fp.plane_normal is not None:
        n = np.asarray(fp.plane_normal, np.float64)
        n_up = float(n @ up)
        if n_up > 0.1:
            bx = forward - (float(n @ forward) / n_up) * up
            by = right - (float(n @ right) / n_up) * up
            p0 = -(float(fp.plane_offset) / n_up) * up
            return x[:, None] * bx + y[:, None] * by + p0
    z = float(np.asarray(fp.origin_xyz or (0.0, 0.0, 0.0), np.float64) @ up)
    return x[:, None] * forward + y[:, None] * right + z * up


def _hull_area(xy):
    import cv2

    if len(xy) < 3:
        return 0.0, None
    hull = cv2.convexHull(np.asarray(xy, np.float32).reshape(-1, 1, 2))
    return float(cv2.contourArea(hull)), hull


def _kabsch_directions(src_dirs, dst_dirs):
    """The rotation taking unit directions src -> dst (least squares)."""
    U, _, Vt = np.linalg.svd(dst_dirs.T @ src_dirs)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    return U @ D @ Vt


def _scale_translation(src, dst, R):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    den = float(np.sum(xs * xs))
    s = float(np.sum(xd * (R @ xs.T).T)) / den if den > 0 else 0.0
    return s, mu_d - s * R @ mu_s


def robust_sim3(src, dst, tol, min_inliers=REGRESS_V2_MIN_SHARED, src_dirs=None,
                dst_dirs=None, max_angle_deg=REGRESS_V2_MAX_ANGLE_DEG):
    """Sim(3) src -> dst fitted on the keyframes the two solves AGREE on: a trimmed refit
    (best half), then the inliers within `tol` (and `max_angle_deg`), refitted. A new loop
    that bends part of the trajectory (e2e5 kf 520: median 8.6 keyframe steps over all
    shared, 0.56 over the consistent majority) then does not hide the unchanged part.

    With `src_dirs`/`dst_dirs` (n x 2 x 3: each camera's up and forward) the ROTATION comes
    from the orientations and only scale/translation from the positions, so the fit is
    determined for any trajectory shape (review HIGH-2: positions on a line leave the roll
    about it free). Returns (s, R, t, inlier mask, residuals) or None."""
    use_dirs = src_dirs is not None and dst_dirs is not None
    cos_tol = math.cos(math.radians(max_angle_deg))

    def fit(keep):
        if use_dirs:
            R = _kabsch_directions(src_dirs[keep].reshape(-1, 3), dst_dirs[keep].reshape(-1, 3))
            s, t = _scale_translation(src[keep], dst[keep], R)
            return s, R, t
        return _umeyama(src[keep], dst[keep])

    def residuals(s, R, t):
        r = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        if not use_dirs:
            return r, np.ones(len(src), bool)
        rotated = np.einsum("ij,nkj->nki", R, src_dirs)
        cos = np.sum(rotated * dst_dirs, axis=2).min(axis=1)
        return r, cos

    keep = np.ones(len(src), bool)
    for _ in range(4):
        s, R, t = fit(keep)
        if not np.isfinite(s) or s <= 0:
            return None
        r, cos = residuals(s, R, t)
        if use_dirs:
            keep = (r <= max(float(np.median(r)), tol)) & (cos >= min(float(np.median(cos)),
                                                                       cos_tol))
        else:
            keep = r <= max(float(np.median(r)), tol)
        if int(keep.sum()) < min_inliers:
            return None
    s, R, t = fit(keep)
    if not np.isfinite(s) or s <= 0:
        return None
    r, cos = residuals(s, R, t)
    inlier = r <= tol
    if use_dirs:
        inlier &= cos >= cos_tol
    return s, R, t, inlier, r


def compare_footprints(prev, new):
    """The scale-free common-footprint comparison of a published and a new component.

    Returns a record; `basis` is "common" when it holds, else "legacy: <why>"."""
    import cv2
    from scipy.spatial import cKDTree

    if prev is None or new is None:
        return {"basis": "legacy: no footprint"}
    shared = [k for k in prev.positions if k in new.positions]
    if len(shared) < REGRESS_V2_MIN_SHARED:
        return {"basis": f"legacy: {len(shared)} shared keyframes", "shared": len(shared)}
    gauge = new.camera_height
    if gauge is None:
        return {"basis": "legacy: no camera height", "shared": len(shared)}
    src = np.stack([prev.positions[k] for k in shared])
    dst = np.stack([new.positions[k] for k in shared])
    oriented = (prev.axes is not None and new.axes is not None
                and all(k in prev.axes and k in new.axes for k in shared))
    dirs = {}
    if oriented:
        dirs = dict(src_dirs=np.stack([np.stack(prev.axes[k]) for k in shared]),
                    dst_dirs=np.stack([np.stack(new.axes[k]) for k in shared]))
    fit = robust_sim3(src, dst, REGRESS_V2_MAX_RESID_FRAC * gauge, **dirs)
    if fit is None:
        return {"basis": "legacy: no consistent Sim(3)", "shared": len(shared)}
    s, R, t, inlier, resid_all = fit
    rec = {"shared": len(shared), "scale": round(float(s), 4),
           "agree": round(float(inlier.mean()), 3),
           "rotation": "orientations" if oriented else "positions"}
    if int(inlier.sum()) < REGRESS_V2_MIN_SHARED or inlier.mean() < REGRESS_V2_MIN_AGREE:
        return {**rec, "basis": f"legacy: {int(inlier.sum())} of {len(shared)} shared "
                                f"keyframes agree within {REGRESS_V2_MAX_RESID_FRAC} h"}
    # Conditioning of the agreeing cameras, in camera heights (scale-free): RMS extent
    # along the trajectory's principal axes (singular values / sqrt(n)).
    agreeing = dst[inlier] - dst[inlier].mean(0)
    sv = np.linalg.svd(agreeing, compute_uv=False) / math.sqrt(len(agreeing)) / gauge
    rec["spread_h"] = [round(float(v), 3) for v in sv[:2]]
    resid = float(np.median(resid_all[inlier]))
    if sv[0] < REGRESS_V2_MIN_SPREAD_H or resid / gauge > REGRESS_V2_MAX_SCALE_ERR * sv[0]:
        return {**rec, "basis": f"legacy: shared cameras span {sv[0]:.2f} h (scale "
                                f"unobservable below {REGRESS_V2_MIN_SPREAD_H} h or a "
                                f"{REGRESS_V2_MAX_SCALE_ERR:.0%} scale error)"}
    if not oriented and (sv[1] < REGRESS_V2_MIN_LATERAL_H
                         or sv[1] < REGRESS_V2_MIN_LATERAL_RATIO * sv[0]):
        return {**rec, "basis": f"legacy: shared cameras near-collinear (lateral {sv[1]:.2f} h, "
                                f"{sv[1] / sv[0]:.2f} of along-track) and no orientations"}
    rec["resid_h"] = round(resid / gauge, 4)
    up_p, fwd_p, right_p = _basis(prev)
    up_n, fwd_n, right_n = _basis(new)
    nx0, ny0, nsx, nsy, nw, nh = _grid(new)
    rows, cols = np.nonzero(prev.opaque)
    if len(rows) == 0:
        return {**rec, "basis": "legacy: published component unpainted"}
    if len(rows) > REGRESS_V2_MAX_CELLS:   # a deterministic, uniform subsample
        pick = np.linspace(0, len(rows) - 1, REGRESS_V2_MAX_CELLS).astype(np.int64)
        rows, cols = rows[pick], cols[pick]
    floor_prev = _cells_to_floor(prev, rows, cols)
    # The common footprint: published floor whose nearest published camera is one both
    # solves agree on, and that lies inside the new raster square.
    kids = list(prev.positions)
    cams = np.stack([prev.positions[k] for k in kids])
    owner = cKDTree(np.c_[cams @ fwd_p, cams @ right_p]).query(
        np.c_[floor_prev @ fwd_p, floor_prev @ right_p])[1]
    agreed = {k for k, ok in zip(shared, inlier) if ok}
    owned = np.array([k in agreed for k in kids])[owner]
    floor = (s * (R @ floor_prev.T)).T + t
    cn = (floor @ fwd_n - nx0) / nsx
    rn = (floor @ right_n - ny0) / nsy
    inside = owned & (cn >= -0.5) & (cn <= nw - 0.5) & (rn >= -0.5) & (rn <= nh - 0.5)
    common = int(inside.sum())
    rec["common_share"] = round(common / len(rows), 4)
    if rec["common_share"] < REGRESS_V2_MIN_COMMON_SHARE:
        return {**rec, "basis": f"legacy: common footprint {rec['common_share']:.2f} of the "
                                "published floor"}
    k = int(min(REGRESS_V2_MAX_DILATE_CELLS, max(1, math.ceil(resid / min(nsx, nsy)))))
    painted = cv2.dilate(new.opaque.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    ri = np.clip(np.rint(rn[inside]).astype(int), 0, nh - 1)
    ci = np.clip(np.rint(cn[inside]).astype(int), 0, nw - 1)
    rec["retained"] = round(float(painted[ri, ci].mean()), 4)
    rec["dilate_cells"] = k
    # Extent: both camera hulls in the NEW gauge, so their ratio is scale-free.
    prev_xy_3d = (s * (R @ np.stack(list(prev.positions.values())).T)).T + t
    prev_xy = np.c_[prev_xy_3d @ fwd_n, prev_xy_3d @ right_n]
    new_pts = np.stack(list(new.positions.values()))
    new_xy = np.c_[new_pts @ fwd_n, new_pts @ right_n]
    prev_hull_area, prev_hull = _hull_area(prev_xy)
    new_hull_area, _ = _hull_area(new_xy)
    rec["growth"] = round(new_hull_area / max(prev_hull_area, gauge * gauge), 3)
    margin = REGRESS_V2_EXTENT_NOVEL_MARGIN_FRAC * gauge
    if prev_hull is None:
        novel = len(new_xy)
    else:
        novel = sum(cv2.pointPolygonTest(prev_hull, (float(x), float(y)), True) < -margin
                    for x, y in new_xy)
    rec["novel_keyframes"] = int(novel)
    rec["footprint_grew"] = bool(rec["growth"] >= REGRESS_V2_EXTENT_GROWTH
                                 and novel >= REGRESS_V2_EXTENT_MIN_NOVEL)
    rec["basis"] = "common"
    rec["_scale"] = float(s)
    return rec


def _plane_source(c):
    if not c.get("plane_ok"):
        return None
    return (c.get("plane") or {}).get("source") or "estimated"


def regress_v2_assess(published_fraction, published_area, published_footprints, stats,
                      footprints):
    """Per shared component: the V2 record (see REGRESS_V2_MIN_RETAINED). Pure."""
    records = []
    now_fraction, now_area = opaque_by_ref(stats), opaque_area_by_ref(stats)
    by_ref = {c.get("reference_segment"): c for c in stats or ()}
    for ref, now in sorted(now_fraction.items()):
        before = (published_fraction or {}).get(ref)
        if before is None:
            continue   # a NEW component never blocks
        rec = {"ref": ref, "fraction": round(now, 4), "published_fraction": round(before, 4)}
        if before < REGRESS_MIN_PRIOR_OPAQUE:
            records.append({**rec, "basis": "none: published component unpainted",
                            "regressed": False})
            continue
        rec.update(compare_footprints((published_footprints or {}).get(ref),
                                      (footprints or {}).get(ref)))
        scale = rec.pop("_scale", None)
        if rec["basis"] == "common":
            prior_area = (published_area or {}).get(ref)
            net = (now_area[ref] / (scale * scale * prior_area)
                   if ref in now_area and prior_area else None)
            rec["net"] = round(net, 3) if net is not None else None
            rec["extent_change"] = bool(rec["footprint_grew"] and net is not None
                                        and net >= REGRESS_V2_EXTENT_MIN_NET)
            rec["regressed"] = bool(rec["retained"] < REGRESS_V2_MIN_RETAINED
                                    and not rec["extent_change"])
            if rec["regressed"]:
                rec["why"] = (f"ref {ref}: retained {rec['retained']:.2f} < "
                              f"{REGRESS_V2_MIN_RETAINED} of the published floor on the common "
                              f"footprint")
        else:
            legacy = regression({ref: before}, [{"reference_segment": ref,
                                                 "opaque_fraction": now,
                                                 "opaque_area": now_area.get(ref)}],
                                {ref: published_area[ref]}
                                if published_area and ref in published_area else None)
            rec["regressed"] = bool(legacy)
            if legacy:
                rec["why"] = legacy
        c = by_ref.get(ref, {})
        rec["projected"] = c.get("projected_keyframes")
        rec["plane_source"] = _plane_source(c)
        records.append(rec)
    return records


def regress_v2_escape(records, stats):
    """('escape' | 'stalled', why) for a regressed landing past the skip count."""
    by_ref = {c.get("reference_segment"): c for c in stats or ()}
    failures = []
    for rec in records:
        if not rec.get("regressed"):
            continue
        c = by_ref.get(rec["ref"], {})
        fraction = float(c.get("opaque_fraction") or 0.0)
        if fraction < REGRESS_MIN_PRIOR_OPAQUE:
            failures.append(f"ref {rec['ref']}: opaque {fraction:.3f} < {REGRESS_MIN_PRIOR_OPAQUE}")
            continue
        if rec.get("extent_change"):   # never true for a regressed record; explicit anyway
            continue
        source = _plane_source(c)
        projected = int(c.get("projected_keyframes") or 0)
        if source not in REGRESS_V2_ESCAPE_PLANE_SOURCES:
            failures.append(f"ref {rec['ref']}: plane {source or 'none'}")
        elif projected < REGRESS_V2_ESCAPE_MIN_PROJECTED:
            failures.append(f"ref {rec['ref']}: projected {projected} < "
                            f"{REGRESS_V2_ESCAPE_MIN_PROJECTED}")
    return ("stalled", "; ".join(failures)) if failures else ("escape", "quality floor met")


def regress_v2_usable(records, stats):
    """The freeze bound's floor: every regressed component has a plane (the camera-drop
    fallback included -- see REGRESS_V2_MAX_FROZEN_S), >= 2% opaque and >= 8 projected
    keyframes."""
    by_ref = {c.get("reference_segment"): c for c in stats or ()}

    def usable(c):
        return (bool(c.get("plane_ok"))
                and float(c.get("opaque_fraction") or 0.0) >= REGRESS_MIN_PRIOR_OPAQUE
                and int(c.get("projected_keyframes") or 0) >= REGRESS_V2_ESCAPE_MIN_PROJECTED)

    return all(usable(by_ref.get(r["ref"], {})) for r in records if r.get("regressed"))


def component_summary(stats):
    """One compact JSON line per landing for the disposition log: a live run must never
    be blind again (e2e2: three `ok` landings that projected nothing, with no clue why)."""
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
        footprint = c.get("footprint")
        if footprint:   # only with footprint bounds on: OFF's log line is unchanged
            rows[-1]["footprint"] = {"reason": footprint.get("reason"),
                                     "growth": footprint.get("growth"),
                                     "size": [footprint.get("width"), footprint.get("height")]}
    return json.dumps(rows, separators=(",", ":"), default=str)


@dataclasses.dataclass
class _DiagWalk:
    """One walk's diagnostics records (files on only) and how many are in its JSONL."""

    records: list = dataclasses.field(default_factory=list)
    written: int = 0


class PhotoFloorplanWorker:
    """One daemon builds each newly published coverage landing's photo plan."""

    def __init__(self, root, *, latest=None, state=None, method=None,
                 wall_cap_s=LANDING_WALL_CAP_S, build=None, regress_v2=None,
                 clock=time.monotonic, footprint_bounds=None, diagnostics_files=None,
                 kept=None):
        if kept is None:
            from tower.config import world_photo_floorplan_kept_setting
            kept = world_photo_floorplan_kept_setting()
        self.kept = bool(kept)
        if method is None:
            from tower.config import world_photo_floorplan_method_setting
            method = world_photo_floorplan_method_setting()
        if regress_v2 is None:
            from tower.config import world_photo_floorplan_regress_v2_setting
            regress_v2 = world_photo_floorplan_regress_v2_setting()
        if footprint_bounds is None:
            from tower.config import world_photo_floorplan_footprint_bounds_setting
            footprint_bounds = world_photo_floorplan_footprint_bounds_setting()
        self.regress_v2 = bool(regress_v2)
        self._clock = clock
        self.root = root
        self.method = method
        self.footprint_bounds = bool(footprint_bounds)
        self.wall_cap_s = float(wall_cap_s)
        self._latest = latest
        self._state = state if state is not None else state_for(root)
        self._build = build or build_landing
        import inspect
        params = inspect.signature(self._build).parameters
        self._build_takes_memory = "plane_memory" in params
        self._build_takes_footprints = "footprints" in params
        self._build_takes_footprint = "footprint_bounds" in params
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._jobs = OrderedDict()
        self._running = None
        self.outcomes = {}
        self._plane_memory = {}   # (world, session) -> {reference_segment: memory}
        self._published_opaque = {}   # (world, session) -> {reference_segment: opaque}
        self._published_opaque_area = {}  # (world, session) -> {reference_segment: area}
        self._regress_streak = {}     # (world, session) -> consecutive regressed skips
        self._published_footprints = {}   # V2: (world, session) -> {reference_segment: Footprint}
        self._frozen_since = {}           # V2: (world, session) -> clock() at the first regressed skip
        # New-scene diagnostics (photo_floorplan_diagnostics): bounded per-walk landing records
        # (pair -> _DiagWalk), kept -- and written to the session -- only with
        # TOWER_WORLD_PHOTO_FLOORPLAN_DIAGNOSTICS exactly `on`; off, the log line is all.
        self._diag_records = OrderedDict()
        self._diag_pending = None
        if diagnostics_files is None:
            from tower.config import world_photo_floorplan_diagnostics_setting
            diagnostics_files = world_photo_floorplan_diagnostics_setting()
        self._diag_files = bool(diagnostics_files)
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

    def _v2_holds(self, pair, v2, stats):
        """The V2 decision for an otherwise publishable landing (under the condition lock).
        True: skip it (`v2["disposition"]` says why). False: publish (`v2["decision"]`)."""
        records = v2["components"]
        regressed = [r for r in records if r.get("regressed")]
        if not regressed:
            return False
        why = "; ".join(r.get("why", f"ref {r['ref']}") for r in regressed)
        v2["why"] = why
        now = self._clock()
        since = self._frozen_since.get(pair)
        frozen = 0.0 if since is None else now - since
        v2["frozen_s"] = round(frozen, 1)
        streak = self._regress_streak.get(pair, 0)
        if streak >= REGRESS_V2_ESCAPE_AFTER:
            verdict, detail = regress_v2_escape(records, stats)
            if verdict == "escape":
                v2["decision"], v2["why"] = "escape", f"{why}; {detail}"
                return False
        if frozen >= REGRESS_V2_MAX_FROZEN_S and regress_v2_usable(records, stats):
            v2["decision"] = "freeze-bound"
            v2["why"] = f"{why}; held {frozen:.0f} s >= {REGRESS_V2_MAX_FROZEN_S:.0f} s"
            return False
        if since is None:
            self._frozen_since[pair] = now
        self._regress_streak[pair] = streak + 1
        if streak >= REGRESS_V2_ESCAPE_AFTER:
            v2["decision"], v2["stalled"] = "stalled", detail
            v2["disposition"] = (f"skipped: regressed, stalled ({why}; {streak + 1} skips, "
                                 f"held {frozen:.0f} s; escape refused: {detail})")
            logger.warning("world photo floorplan: plan STALLED for world=%s session=%s: %d "
                           "regressed skips, held %.0f s (bound %.0f s): %s", pair[0], pair[1],
                           streak + 1, frozen, REGRESS_V2_MAX_FROZEN_S, detail)
        else:
            v2["decision"] = "skip"
            v2["disposition"] = (f"skipped: regressed ({why}; "
                                 f"{streak + 1}/{REGRESS_V2_ESCAPE_AFTER})")
        return True

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
        kept_reason = None   # why this landing published nothing (KEPT_REASONS); None if it did
        try:
            if self._build_takes_memory:
                memory = self._plane_memory.get(pair, {})
                trial = {ref: dict(m) for ref, m in memory.items()}
                kwargs = {"plane_memory": trial}
            else:
                kwargs = {}
            footprints = None
            if self.regress_v2 and self._build_takes_footprints:
                footprints = kwargs["footprints"] = {}
            if self.footprint_bounds and self._build_takes_footprint:
                kwargs["footprint_bounds"] = True
            block, tiles, stats = self._build(self.root, job["world_id"], job["session_id"],
                                              receipt, job["eligible"], job["tree"],
                                              job["solution_stat"], method=self.method,
                                              check=check, **kwargs)
            check()
        except LandingSkipped as exc:
            disposition = f"skipped: {exc}"
            if isinstance(exc, LandingOverrun):
                kept_reason = KEPT_SUPERSEDED if cancel.is_set() else KEPT_WALL_CAP
            else:
                kept_reason = KEPT_FAILED
        except Exception as exc:  # noqa: BLE001 -- one bad landing never stops the worker
            logger.exception("world photo floorplan: build failed")
            disposition = f"failed: {type(exc).__name__}: {exc}"
            kept_reason = KEPT_FAILED
        elapsed_ms = (time.monotonic() - started) * 1000.0
        v2 = None
        if disposition == "ok" and self.regress_v2:
            # Pure, and on this worker thread -- the only writer of the published baseline --
            # so it runs before the lock is taken.
            v2 = {"decision": "publish",
                  "components": regress_v2_assess(self._published_opaque.get(pair),
                                                  self._published_opaque_area.get(pair),
                                                  self._published_footprints.get(pair),
                                                  stats, footprints)}
        if disposition == "ok":
            with self._condition:
                current = self._latest(*pair) if self._latest is not None else receipt
                if self._stop.is_set() or cancel.is_set():
                    disposition = "skipped: cancelled"
                    kept_reason = KEPT_SUPERSEDED
                elif identity_of(current) != identity:
                    disposition = "skipped: superseded by a newer coverage landing"
                    kept_reason = KEPT_SUPERSEDED
                elif time.monotonic() - started > self.wall_cap_s:
                    disposition = f"skipped: exceeded the {self.wall_cap_s:.1f} s wall cap"
                    kept_reason = KEPT_WALL_CAP
                elif v2 is not None and self._v2_holds(pair, v2, stats):
                    disposition = v2["disposition"]
                    kept_reason = KEPT_STALLED if v2["decision"] == "stalled" else KEPT_REGRESSED
                elif v2 is None and ((regressed := regression(
                        self._published_opaque.get(pair), stats,
                        self._published_opaque_area.get(pair)))
                      and self._regress_streak.get(pair, 0) < REGRESS_MAX_CONSECUTIVE_SKIPS):
                    self._regress_streak[pair] = self._regress_streak.get(pair, 0) + 1
                    disposition = (f"skipped: regressed ({regressed}; "
                                   f"{self._regress_streak[pair]}/{REGRESS_MAX_CONSECUTIVE_SKIPS})")
                    kept_reason = KEPT_REGRESSED
                elif not self._state.publish(*pair, block, tiles):
                    disposition = "skipped: not newer than the published landing (never re-dated)"
                    kept_reason = KEPT_NOT_NEWER
                else:
                    # Only a PUBLISHED landing's planes are carried to the next one.
                    if trial is not None:
                        self._plane_memory[pair] = trial
                    self._published_opaque[pair] = opaque_by_ref(stats)
                    self._published_opaque_area[pair] = opaque_area_by_ref(stats)
                    if v2 is not None:
                        self._published_footprints[pair] = dict(footprints or {})
                        self._frozen_since.pop(pair, None)
                        if v2["decision"] in ("escape", "freeze-bound"):
                            logger.warning("world photo floorplan: publishing a regressed landing "
                                           "after %d consecutive skips (%s: %s)",
                                           self._regress_streak.get(pair, 0), v2["decision"],
                                           v2.get("why"))
                    elif regressed:
                        logger.warning("world photo floorplan: publishing a regressed landing after %d "
                                       "consecutive skips (escape valve): %s",
                                       self._regress_streak.get(pair, 0), regressed)
                    self._regress_streak[pair] = 0
        outcome = {"identity": identity, "disposition": disposition,
                   "elapsed_ms": round(elapsed_ms, 1), "components": stats}
        if self.kept:
            # On this worker thread -- the only writer of the published state -- after the
            # disposition is final and before `_run` releases a waiting manifest request. A
            # published landing already cleared the notice inside `publish`.
            outcome["kept"] = None
            if disposition != "ok" and kept_reason is not None and not self._stop.is_set():
                if self._state.keep(*pair, receipt, kept_reason):
                    outcome["kept"] = kept_reason
                    logger.info("world photo floorplan kept: world=%s session=%s landing=%r "
                                "reason=%s", pair[0], pair[1], identity, kept_reason)
        if self.regress_v2:
            if v2 is None:
                v2 = {"decision": "not assessed", "components": []}
            elif disposition != "ok" and v2["decision"] in ("publish", "escape", "freeze-bound"):
                # superseded, cancelled, wall cap or never re-dated after the V2 verdict
                v2["decision"] = f"not published (V2 said {v2['decision']})"
            outcome["regress_v2"] = v2
            logger.info("world photo floorplan regress-v2: world=%s session=%s landing=%r "
                        "decision=%s streak=%d frozen_s=%s components=%s", pair[0], pair[1],
                        identity, v2["decision"], self._regress_streak.get(pair, 0),
                        v2.get("frozen_s"), json.dumps(v2["components"], separators=(",", ":"),
                                                       default=str))
        self.outcomes = {**self.outcomes, pair: outcome}
        log = logger.info if disposition == "ok" else logger.warning
        log("world photo floorplan: world=%s session=%s landing=%r method=%s elapsed_ms=%.1f "
            "disposition=%s components=%s", pair[0], pair[1], identity, self.method, elapsed_ms,
            disposition, component_summary(stats))
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
        """One bounded log line per landing; with TOWER_WORLD_PHOTO_FLOORPLAN_DIAGNOSTICS=on also
        the session's JSONL and per-walk summary. Never raises."""
        try:
            from tower.world_builder import photo_floorplan_diagnostics as diag
            logger.info("world photo floorplan diagnostics: world=%s session=%s landing=%r "
                        "record=%s", pair[0], pair[1], identity, diag._dumps(record))
            if not self._diag_files:
                return          # the log line is the record; nothing is retained
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
