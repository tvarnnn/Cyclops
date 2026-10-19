"""Re-place, don't cut: register the keyframes the final solve could not place into the finished room.

WHY. A walk has short low-texture, low-parallax stretches (a plain wall, a wall top, a door corner: 24-90
observations per keyframe against 300-780 elsewhere, 3-13 s of mostly head rotation). GLOMAP's global
positioning starts from random camera positions, and on such a stretch it sometimes collapses the camera
centres onto one point: the stretch's own matches barely constrain its translation, and a collapsed stretch
satisfies them too. Its rotations stay good (rotation averaging does not need parallax); its centres do not.
The final scale guard then isolates the stretch (`final-scale-outlier`), or its cameras fall below the
support floor, so its frames are never fused and what only they saw is void. The same database gives a
good draw or a collapsed one by chance (walk 10.1: void 5.0 % or 15.1 % from one database).

WHAT. Every such keyframe is offered to COLMAP's incremental registration against the FINISHED room, the way
COLMAP's incremental mapper registers the next image: PnP of the image's 2-D features against the 3-D points
the room's cameras triangulated. PnP needs no parallax among the stretch's own frames -- exactly what global
positioning lacked -- and refuses an image it cannot pose instead of collapsing it.

  1. TARGETS (`targets_of`): every keyframe the published room would not hold as a supported camera --
     the guard's isolated pieces, the solver room's cameras below the support floor
     (`global_solve.MIN_IMAGE_OBSERVATIONS`), and keyframes the solver did not pose. A keyframe of another
     solver component is not a target: that is another decision, not this failure.
  2. THE ROOM MODEL (`register`): the kept room's cameras, FIXED, and the points they triangulate, with
     COLMAP's own point filters on the kept observers (reprojection <= `filter_max_reproj_error` 4 px,
     >= 2 observers, triangulation angle >= `filter_min_tri_angle` 1.5 deg). A target's old observations
     are not used: they belong to the collapsed geometry.
  3. REGISTRATION (`register`): `pycolmap.IncrementalMapper` continues that model with
     `fix_existing_frames` -- next-best-image ranking, PnP by LO-RANSAC and refinement, then triangulation
     and local bundle adjustment of the newly registered image. COLMAP's registration gates, unchanged:
     >= 30 inliers (`abs_pose_min_num_inliers`), inlier ratio >= 0.25, <= 12 px (`abs_pose_max_error`),
     at most 3 trials an image. Intrinsics are fixed, as in the solve. Seeded: one input, one result.
  4. GATES (`replace_cut`), each a rule that already exists; a keyframe that fails one stays where it was
     (cut, unsupported or unposed -- today's outcome):
       a. the support floor: >= `MIN_IMAGE_OBSERVATIONS` 3-D observations, the solve's own rule for a camera
          that is a measurement;
       b. the guard's pose screen (`final_scale_guard._pose_screen`) over the room and the re-placed
          cameras in capture order: a re-placed camera in a step beyond the walking allowance
          (2.0 m/s x dt + 0.3 m) or a height jump, in a hard step, with a room defect, or outside the core
          cluster is refused (consistency with its neighbours in time); and one that no chain of steps the
          screen can test (<= 3 journal indices, <= 2 s) joins to a room camera is refused: the screen
          cannot place an island relative to the room;
       c. the guard's scale screen, re-run (`assess_v2`) on the room plus the re-placed cameras with their
          metric ratios recomputed from the live depth cache: a re-placed camera the guard would isolate is
          refused (the rule that cut the stretch must now keep it);
       d. downstream, unchanged: the depth stage's per-frame fit (held-out <= 8 %) decides what is fused.
     Without metric evidence there is no scale screen, so nothing is re-placed.
  5. MERGE (`merge`): a re-placed keyframe joins component 0 with its registered pose; its old observations
     are dropped; its observations of room points and the points newly triangulated with it are added. The
     room's own points and poses are never moved.

It never raises: any failure leaves the guard's decision as it was, with the error in the record. With
`TOWER_WORLD_REPLACE_CUT=off` it does not run and the publication is today's, byte for byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

STEP_ID = "replace-cut/1"

# Why a target stayed where it was (the audit's `refused` counts).
NOT_REGISTERED = "not-registered"        # COLMAP's registration found no pose that passes its gates
BELOW_FLOOR = "below-support-floor"      # registered, but < min_obs 3-D observations
MOTION = "motion-screen"                 # the guard's pose screen refuses it
SCALE = "scale-screen"                   # the guard's scale screen would isolate it

# The uncertified kind of a re-placed camera no scale screen judged (components record, placed component).
UNCERTIFIED_REPLACED = "re-placed"


@dataclass(frozen=True)
class ReplaceParams:
    """The step's own settings. Every registration threshold is COLMAP's default
    (`pycolmap.IncrementalPipelineOptions()`), recorded in the audit, not restated here."""

    # global_solve.MIN_IMAGE_OBSERVATIONS: the floor for a camera that is a measurement.
    min_obs: int = 30
    # RANSAC and the mapper are seeded so that one solve and one database give one result.
    seed: int = 20261019
    # A bound on the cost, not a quality knob: images not registered by then stay where they were.
    max_seconds: float = 180.0
    # The registration loop. "pnp": register (PnP), triangulate, local BA, until a pass registers nothing.
    # "colmap": COLMAP's own incremental loop (`IncrementalPipeline`): after a pass that registers nothing,
    # one global refinement (track completion, retriangulation, global BA with the room fixed) and another
    # pass; and, when no image has visible 3-D points, COLMAP 4's structure-less registration from 2-D-2-D
    # correspondences to registered images (`structure_less_registration_fallback`, on by default).
    loop: str = "pnp"
    # Whether the guard's per-camera scale screen also judges a re-placed camera (gate c).
    scale_screen: bool = True

    def digest(self) -> str:
        return hashlib.sha1(json.dumps({"step": STEP_ID, **asdict(self)}, sort_keys=True).encode()
                            ).hexdigest()[:16]


@dataclass
class Registration:
    """What `register` placed. Point refs: ("o", index into the solution's points) for a room point,
    ("n", id) for a point triangulated during registration."""

    poses: dict = field(default_factory=dict)       # kid -> (R_cw (3, 3), t_cw (3,))
    obs: dict = field(default_factory=dict)         # kid -> [(feature idx, ref)]
    points: dict = field(default_factory=dict)      # new id -> {"xyz", "error", "track": [(kid, feature)]}
    xy: dict = field(default_factory=dict)          # (kid, feature) -> (x, y) undistorted px
    record: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def targets_of(solution, pieces_kids, *, min_obs: int) -> list[str]:
    """The keyframes to offer, in the solution's keyframe order: isolated by the guard, in the solver's
    room below the support floor, or not posed at all."""
    poses = solution.poses or {}
    out = []
    for kid in solution.keyframe_ids:
        p = poses.get(kid)
        if kid in pieces_kids or p is None:
            out.append(kid)
        elif int(p.get("component", 0)) == 0 and int(p.get("observations", 0)) < min_obs:
            out.append(kid)
    return out


# ---------------------------------------------------------------------------
# the room model and the registration
# ---------------------------------------------------------------------------


def _pose(p: dict) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(p["rotation"], np.float64).reshape(3, 3), np.asarray(p["translation"], np.float64))


def room_points(solution, room_kids, *, max_error: float, min_angle_deg: float) -> dict:
    """{point index: [(kid, feature)]}: the solution's points as the ROOM's cameras alone triangulate them,
    with COLMAP's point filters applied to those observers: each observation within `max_error` px, at least
    two observers, a triangulation angle of at least `min_angle_deg`."""
    cam = solution.camera or {}
    fx, fy, cx, cy = (float(cam[k]) for k in ("fx", "fy", "cx", "cy"))
    kidx = {kid: i for i, kid in enumerate(solution.keyframe_ids)}
    room_idx = np.array(sorted(kidx[k] for k in room_kids), np.int64)
    n_kf = len(solution.keyframe_ids)
    R = np.full((n_kf, 3, 3), np.nan)
    t = np.full((n_kf, 3), np.nan)
    for k in room_kids:
        R[kidx[k]], t[kidx[k]] = _pose(solution.poses[k])
    obs = np.asarray(solution.observations, np.int64).reshape(-1, 3)
    xy = np.asarray(solution.observation_xy, np.float64).reshape(-1, 2)
    if not len(obs) or len(xy) != len(obs) or not len(room_idx):
        return {}
    sel = np.isin(obs[:, 0], room_idx)
    obs, xy = obs[sel], xy[sel]
    X = np.asarray(solution.xyz, np.float64)[obs[:, 2]]
    Xc = np.einsum("nij,nj->ni", R[obs[:, 0]], X) + t[obs[:, 0]]
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * Xc[:, 0] / z + cx
        v = fy * Xc[:, 1] / z + cy
    err = np.hypot(u - xy[:, 0], v - xy[:, 1])
    ok = (z > 0) & np.isfinite(err) & (err <= max_error)
    obs = obs[ok]
    centres = -np.einsum("nji,nj->ni", R, t)       # C = -R^T t, per keyframe
    order = np.argsort(obs[:, 2], kind="stable")
    obs = obs[order]
    starts = np.flatnonzero(np.r_[True, obs[1:, 2] != obs[:-1, 2]]) if len(obs) else np.zeros(0, np.int64)
    ends = np.r_[starts[1:], len(obs)] if len(obs) else np.zeros(0, np.int64)
    cos_max = math.cos(math.radians(min_angle_deg))
    names = solution.keyframe_ids
    out = {}
    xyz = np.asarray(solution.xyz, np.float64)
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        rows = obs[s:e]
        rays = xyz[rows[0, 2]] - centres[rows[:, 0]]
        n = np.linalg.norm(rays, axis=1)
        if not np.all(n > 0):
            continue
        rays = rays / n[:, None]
        if float((rays @ rays.T).min()) > cos_max:
            continue
        out[int(rows[0, 2])] = [(names[int(r[0])], int(r[1])) for r in rows]
    return out


def _pipeline_options(pycolmap, params: ReplaceParams):
    opts = pycolmap.IncrementalPipelineOptions()
    # The solve keeps the calibrated pinhole fixed (global_solve._map_candidate); so does this.
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    opts.mapper.abs_pose_refine_focal_length = False
    opts.mapper.abs_pose_refine_extra_params = False
    # The room's cameras are measurements already published; only the new images move.
    opts.fix_existing_frames = True
    opts.random_seed = int(params.seed)
    opts.mapper.random_seed = int(params.seed)
    opts.triangulation.random_seed = int(params.seed)
    return opts


def register(solution, *, room_kids, target_kids, database_path, name_of: dict,
             params: ReplaceParams | None = None) -> Registration:
    """Register `target_kids` into the room made of `room_kids` (poses fixed) and the points they
    triangulate, with COLMAP's incremental mapper over the solve database. Reads the database only."""
    import pycolmap  # noqa: PLC0415 -- optional, as everywhere in the solve

    params = params or ReplaceParams()
    started = time.perf_counter()
    reg = Registration()
    opts = _pipeline_options(pycolmap, params)
    mopts = opts.get_mapper()
    pycolmap.set_random_seed(int(params.seed))
    room_kids = [k for k in solution.keyframe_ids if k in set(room_kids)]
    target_kids = [k for k in solution.keyframe_ids if k in set(target_kids)]
    names = {name_of[k] for k in room_kids + target_kids if k in name_of}
    db = pycolmap.Database.open(str(database_path))
    try:
        copts = pycolmap.DatabaseCacheOptions()
        copts.min_num_matches = int(opts.min_num_matches)
        copts.ignore_watermarks = bool(opts.ignore_watermarks)
        copts.image_names = names
        cache = pycolmap.DatabaseCache.create(db, copts)
    finally:
        db.close()
    rec = pycolmap.Reconstruction()
    rec.load(cache)
    image_of = {img.name: img.image_id for img in rec.images.values()}
    kid_of_image = {image_of[name_of[k]]: k for k in room_kids + target_kids if name_of.get(k) in image_of}
    cam = solution.camera or {}
    params4 = [float(cam[k]) for k in ("fx", "fy", "cx", "cy")]
    for camera in rec.cameras.values():
        camera.params = params4
    for k in room_kids:
        iid = image_of.get(name_of.get(k))
        if iid is None:
            continue
        R, t = _pose(solution.poses[k])
        img = rec.image(iid)
        rec.frame(img.frame_id).rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
        rec.register_frame(img.frame_id)
    kept = room_points(solution, room_kids, max_error=float(mopts.filter_max_reproj_error),
                       min_angle_deg=float(mopts.filter_min_tri_angle))
    orig_of: dict[int, int] = {}
    xyz = np.asarray(solution.xyz, np.float64)
    for p, elems in kept.items():
        track = pycolmap.Track()
        for k, f in elems:
            iid = image_of.get(name_of.get(k))
            if iid is not None:
                track.add_element(iid, f)
        if track.length() >= 2:
            orig_of[rec.add_point3D(xyz[p], track)] = p
    mapper = pycolmap.IncrementalMapper(cache)
    mapper.begin_reconstruction(rec)
    target_images = {image_of[name_of[k]] for k in target_kids if name_of.get(k) in image_of}
    tri = opts.get_triangulation()
    ba = opts.get_local_bundle_adjustment()
    pnp_inliers: dict[int, int] = {}
    attempts = 0
    structure_less = 0
    global_refinements = 0
    stopped_by_budget = False
    colmap_loop = params.loop == "colmap"
    prev_success = True
    while True:
        if time.perf_counter() - started > params.max_seconds:
            stopped_by_budget = True
            break
        nxt = [i for i in mapper.find_next_images(mopts, False) if i in target_images]
        less = False
        if not nxt and colmap_loop and opts.structure_less_registration_fallback:
            nxt = [i for i in mapper.find_next_images(mopts, True) if i in target_images]
            less = True
        placed = None
        for iid in nxt:
            attempts += 1
            ok = (mapper.register_next_structure_less_image(mopts, iid) if less
                  else mapper.register_next_image(mopts, iid))
            if ok:
                placed = iid
                break
        if placed is not None:
            structure_less += int(less)
            pnp_inliers[placed] = int(rec.image(placed).num_points3D)
            mapper.triangulate_image(tri, placed)
            mapper.iterative_local_refinement(int(opts.ba_local_max_refinements),
                                              float(opts.ba_local_max_refinement_change), mopts, ba, tri, placed)
            prev_success = True
            continue
        if colmap_loop and prev_success:
            mapper.iterative_global_refinement(int(opts.ba_global_max_refinements),
                                               float(opts.ba_global_max_refinement_change), mopts,
                                               opts.get_global_bundle_adjustment(), tri, False)
            global_refinements += 1
            prev_success = False
            continue
        break
    # The room is fixed by construction (`fix_existing_frames`); checked, because every re-placed pose is
    # read in the room's frame.
    moved = 0.0
    for k in room_kids:
        iid = image_of.get(name_of.get(k))
        if iid is None:
            continue
        R0, t0 = _pose(solution.poses[k])
        cfw = rec.image(iid).cam_from_world()
        moved = max(moved, float(np.abs(np.asarray(cfw.translation) - t0).max()),
                    float(np.abs(np.asarray(cfw.rotation.matrix()) - R0).max()))
    if moved > 1e-6:
        raise RuntimeError(f"the room's fixed cameras moved during registration ({moved:.3g})")
    registered = [iid for iid in target_images if rec.image(iid).has_pose]
    new_ids: set[int] = set()
    for iid in registered:
        k = kid_of_image[iid]
        img = rec.image(iid)
        cfw = img.cam_from_world()
        reg.poses[k] = (np.asarray(cfw.rotation.matrix(), np.float64), np.asarray(cfw.translation, np.float64))
        rows = []
        for f, p2d in enumerate(img.points2D):
            if not p2d.has_point3D():
                continue
            pid = int(p2d.point3D_id)
            reg.xy[(k, f)] = (float(p2d.xy[0]), float(p2d.xy[1]))
            if pid in orig_of:
                rows.append((f, ("o", orig_of[pid])))
            else:
                rows.append((f, ("n", pid)))
                new_ids.add(pid)
        reg.obs[k] = rows
    for pid in sorted(new_ids):
        pt = rec.point3D(pid)
        track = []
        for el in pt.track.elements:
            k = kid_of_image.get(int(el.image_id))
            if k is None:
                continue
            track.append((k, int(el.point2D_idx)))
            xy = rec.image(int(el.image_id)).points2D[int(el.point2D_idx)].xy
            reg.xy[(k, int(el.point2D_idx))] = (float(xy[0]), float(xy[1]))
        reg.points[pid] = {"xyz": np.asarray(pt.xyz, np.float64), "error": float(pt.error), "track": track}
    reg.record = {
        "room_cameras": len(room_kids), "room_points": len(orig_of), "targets": len(target_kids),
        "target_images": len(target_images), "registration_attempts": attempts, "registered": len(registered),
        "pnp_inliers": {kid_of_image[i]: n for i, n in sorted(pnp_inliers.items())},
        "existing_frames_fixed": len(mapper.existing_frame_ids),
        "loop": params.loop, "structure_less_registered": structure_less,
        "global_refinements": global_refinements, "room_moved": moved,
        "stopped_by_budget": stopped_by_budget, "seconds": round(time.perf_counter() - started, 3),
        "colmap": {"mapper": {k: v for k, v in mopts.todict().items() if isinstance(v, (int, float, bool))},
                   "triangulation": {k: v for k, v in tri.todict().items() if isinstance(v, (int, float, bool))},
                   "min_num_matches": int(opts.min_num_matches)},
    }
    mapper.end_reconstruction(False)
    return reg


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------


def merge(solution, reg: Registration, accepted, *, room_kids):
    """The solution with every `accepted` keyframe re-placed: in component 0 at its registered pose, its old
    observation rows dropped (a point left with none is dropped too), its rows to room points and the points
    triangulated with it added. A new point keeps only observations of room or accepted keyframes that no
    other point already holds, and needs two. The room's points and poses are not moved."""
    accepted = set(accepted)
    if not accepted:
        return solution
    room = set(room_kids)
    kidx = {kid: i for i, kid in enumerate(solution.keyframe_ids)}
    acc_idx = np.array(sorted(kidx[k] for k in accepted), np.int64)
    obs = np.asarray(solution.observations, np.int64).reshape(-1, 3)
    xy = np.asarray(solution.observation_xy, np.float32).reshape(-1, 2)
    has_xy = len(xy) == len(obs)
    keep = ~np.isin(obs[:, 0], acc_idx)
    n_old = len(solution.xyz)
    seen_before = np.zeros(n_old, bool)
    seen_before[obs[:, 2]] = True
    obs, xy = obs[keep], (xy[keep] if has_xy else xy)
    alive = ~seen_before
    alive[obs[:, 2]] = True
    new_index = np.cumsum(alive) - 1
    obs = obs.copy()
    obs[:, 2] = new_index[obs[:, 2]]
    taken = set(zip(obs[:, 0].tolist(), obs[:, 1].tolist()))
    xyz = [np.asarray(solution.xyz, np.float32)[alive]]
    rgb = [np.asarray(solution.rgb, np.uint8).reshape(-1, 3)[alive]]
    comp = [np.asarray(solution.component, np.int32)[alive]]
    tlen = np.asarray(solution.track_length, np.int32)[alive].copy()
    err = [np.asarray(solution.error, np.float32)[alive]]
    n_alive = int(alive.sum())
    rows, rows_xy = [], []
    for k in sorted(accepted, key=kidx.get):
        ki = kidx[k]
        for f, (kind, ref) in reg.obs.get(k, []):
            if kind != "o" or not alive[ref] or (ki, f) in taken:
                continue
            rows.append((ki, f, int(new_index[ref])))
            rows_xy.append(reg.xy[(k, f)])
            taken.add((ki, f))
            tlen[int(new_index[ref])] += 1
    new_xyz, new_err = [], []
    for pid in sorted(reg.points):
        pt = reg.points[pid]
        elems = []
        for k, f in pt["track"]:
            if (k in accepted or k in room) and (kidx[k], f) not in taken:
                elems.append((k, f))
        if len(elems) < 2 or not any(k in accepted for k, _ in elems):
            continue
        p = n_alive + len(new_xyz)
        for k, f in elems:
            rows.append((kidx[k], f, p))
            rows_xy.append(reg.xy[(k, f)])
            taken.add((kidx[k], f))
        new_xyz.append(pt["xyz"])
        new_err.append(pt["error"])
    n_new = len(new_xyz)
    obs_all = np.concatenate([obs, np.asarray(rows, np.int64).reshape(-1, 3)]) if rows else obs
    if has_xy:
        xy_all = np.concatenate([xy, np.asarray(rows_xy, np.float32).reshape(-1, 2)]) if rows else xy
    else:
        xy_all = xy
    xyz_all = np.concatenate(xyz + [np.asarray(new_xyz, np.float32).reshape(-1, 3)])
    n_pts = len(xyz_all)
    counts = np.bincount(obs_all[:, 2], minlength=n_pts)
    tlen_all = np.concatenate([tlen, counts[n_alive:].astype(np.int32)])
    first = np.full(n_pts, np.iinfo(np.int32).max, np.int64)
    np.minimum.at(first, obs_all[:, 2], obs_all[:, 0])
    old_first = np.asarray(solution.first_keyframe, np.int64)[alive]
    first[:n_alive] = np.where(first[:n_alive] == np.iinfo(np.int32).max, old_first, first[:n_alive])
    per_kf = np.bincount(obs_all[:, 0], minlength=len(solution.keyframe_ids))
    poses = {kid: dict(p) for kid, p in (solution.poses or {}).items()}
    for k in accepted:
        R, t = reg.poses[k]
        row = dict(poses.get(k) or {})
        if "component" in row:
            row.setdefault("solver_component", int(row["component"]))
        row.update(component=0, rotation=[float(v) for v in R.reshape(-1)],
                   translation=[float(v) for v in t.reshape(-1)], registration=STEP_ID)
        poses[k] = row
    for kid in accepted | room:
        if kid in poses:
            poses[kid]["observations"] = int(per_kf[kidx[kid]])
    from tower.world_builder.global_solve import withheld_rgb  # noqa: PLC0415

    return dataclasses.replace(
        solution, poses=poses,
        xyz=xyz_all.astype(np.float32),
        rgb=np.concatenate(rgb + [withheld_rgb(n_new)]).reshape(-1, 3),
        component=np.concatenate(comp + [np.zeros(n_new, np.int32)]).astype(np.int32),
        first_keyframe=first.astype(np.int32),
        track_length=tlen_all.astype(np.int32),
        error=np.concatenate(err + [np.asarray(new_err, np.float32)]).astype(np.float32),
        observations=obs_all.astype(np.int32),
        observation_xy=np.asarray(xy_all, np.float32).reshape(-1, 2),
    )


# ---------------------------------------------------------------------------
# the gates and the step
# ---------------------------------------------------------------------------


def motion_refusals(room_cams, new_cams, mpu: float, gparams) -> dict:
    """{kid: why} for the re-placed cameras the guard's pose screen, run over the room and the re-placed
    cameras in capture order, does not accept. Each round: a re-placed camera outside the screen's core
    cluster or with a room defect is refused; only when there is none, the re-placed camera(s) in the most
    steps beyond the walking allowance (or hard steps, or height jumps) -- a room camera is never blamed,
    and a lone outlier between two re-placed neighbours takes the blame alone. Then the screen runs again
    without the refused cameras, until it accepts every one left."""
    from tower.world_builder import final_scale_guard as FSG  # noqa: PLC0415

    out: dict = {}
    pending = list(new_cams)
    while pending:
        new = {c.kid for c in pending}
        cams = sorted(list(room_cams) + pending, key=lambda c: c.index)
        kid_at = {c.index: c.kid for c in cams}
        screen = FSG._pose_screen(cams, mpu, gparams)
        core = {c.kid for c in screen["room"]}
        hit = {c.kid: "outside-core-cluster" for c in pending if c.kid not in core}
        for k in screen.get("defects") or {}:
            if k in new:
                hit.setdefault(k, "room-defect")
        if not hit:
            count: dict = {}
            why: dict = {}
            for item in list(screen["flags"]) + list(screen["warnings"]):
                for i in item.get("indices", []):
                    k = kid_at.get(i)
                    if k in new:
                        count[k] = count.get(k, 0) + 1
                        why.setdefault(k, item["kind"])
            if count:
                most = max(count.values())
                hit = {k: why[k] for k, n in count.items() if n == most}
        if not hit:
            break
        out.update(hit)
        pending = [c for c in pending if c.kid not in hit]
    # ANCHORED: the screen tests only steps of <= `step_max_index_gap` journal indices and <= `step_max_dt_s`
    # ("elapsed time alone never certifies where an island sits relative to the room"), and an untestable
    # step never splits its runs. So a re-placed camera is kept only when a chain of TESTABLE steps through
    # kept re-placed cameras joins it to a room camera on at least one side; an island the screen cannot
    # test against the room is refused.
    cams = sorted(list(room_cams) + pending, key=lambda c: c.index)
    new = {c.kid for c in pending}

    def testable(a, b):
        if a.t is None or b.t is None:
            return False
        dt = float(b.t - a.t)
        return b.index - a.index <= gparams.step_max_index_gap and 0 < dt <= gparams.step_max_dt_s

    anchored = set()
    for seq in (cams, cams[::-1]):
        prev = None
        reach = False
        for c in seq:
            if c.kid not in new:
                reach = True
            else:
                a, b = (prev, c) if seq is cams else (c, prev)
                reach = reach and prev is not None and testable(a, b)
                if reach:
                    anchored.add(c.kid)
            prev = c
    for c in pending:
        if c.kid not in anchored:
            out[c.kid] = "unanchored"
    return out


def _piece_record(p: dict, ids: list, kidx: dict) -> dict:
    idx = sorted(kidx[k] for k in ids)
    first = next(k for k in ids if kidx[k] == idx[0])
    last = next(k for k in ids if kidx[k] == idx[-1])
    return dict(p, keyframe_ids=ids, keyframes=len(ids), first_index=idx[0], last_index=idx[-1],
                first_keyframe_id=first, last_keyframe_id=last,
                replaced=len(p["keyframe_ids"]) - len(ids))


def replace_cut(store, world_id, session_id, solution, assessment: dict, *, cams, keyframes, name_of: dict,
                database_path, gparams, groups_of=None, live_metric_fn=None,
                params: ReplaceParams | None = None, register_fn=None):
    """(solution, assessment, record): the candidate with every target that passes the gates re-placed into
    the room, the guard's assessment without them in its pieces, and what happened. `cams` are the guard's
    room cameras (`final_scale_guard.room_cameras`) of `solution`; `assessment` its `assess_v2` decision.
    Returns the inputs unchanged and record None when there is nothing to offer, and unchanged with a record
    saying why when nothing passes or anything fails. Never raises."""
    from tower.world_builder import final_scale_guard as FSG  # noqa: PLC0415

    params = params or ReplaceParams()
    started = time.perf_counter()
    pieces_kids = {k for p in assessment.get("pieces") or [] for k in p["keyframe_ids"]}
    targets = targets_of(solution, pieces_kids, min_obs=params.min_obs)
    if not targets:
        return solution, assessment, None
    record: dict = {"id": STEP_ID, "params": asdict(params), "params_digest": params.digest(),
                    "targets": len(targets), "targets_isolated": len(pieces_kids),
                    "targets_below_floor": sum(1 for k in targets
                                               if k not in pieces_kids and k in (solution.poses or {})),
                    "targets_unposed": sum(1 for k in targets if k not in (solution.poses or {})),
                    "replaced": 0, "replaced_keyframe_ids": [], "refused": {}, "state": None}

    def done(state, **extra):
        record.update(state=state, seconds=round(time.perf_counter() - started, 3), **extra)
        return record

    try:
        if not assessment.get("reference") or not (assessment.get("pose") or {}).get("metres_per_unit"):
            return solution, assessment, done("skipped", why="no metric evidence, so no scale screen")
        mpu = float(assessment["pose"]["metres_per_unit"])
        room_cams = [c for c in cams if c.kid not in pieces_kids and FSG._finite_pose(c)]
        room = {c.kid for c in room_cams}
        reg = (register_fn or register)(solution, room_kids=room, target_kids=targets,
                                        database_path=database_path, name_of=name_of, params=params)
        record["registration"] = reg.record
        accepted = set(reg.poses)
        refused: dict[str, str] = {k: NOT_REGISTERED for k in targets if k not in accepted}

        def screen_floor_and_motion(acc):
            while True:
                cand = merge(solution, reg, acc, room_kids=room)
                floor = {k for k in acc if int(cand.poses[k]["observations"]) < params.min_obs}
                for k in floor:
                    refused[k] = BELOW_FLOOR
                acc = acc - floor
                cand = merge(solution, reg, acc, room_kids=room) if floor else cand
                new_cams = [c for c in FSG.room_cameras(cand, keyframes, name_of, groups_of=groups_of,
                                                        min_obs=params.min_obs) if c.kid in acc]
                motion = motion_refusals(room_cams, new_cams, mpu, gparams)
                for k, why in motion.items():
                    refused[k] = f"{MOTION}:{why}"
                acc = acc - set(motion)
                if not floor and not motion:
                    return cand, acc

        cand, accepted = screen_floor_and_motion(accepted)
        scale_record = None
        if accepted and params.scale_screen:
            metric_log, inputs = FSG.read_evidence(store, world_id, session_id, cand, None,
                                                   database_path=database_path, name_of=name_of,
                                                   live_metric_fn=live_metric_fn)
            keep = room | accepted
            cams2 = [c for c in FSG.room_cameras(cand, keyframes, name_of, groups_of=groups_of,
                                                 min_obs=params.min_obs) if c.kid in keep]
            if not metric_log:
                for k in accepted:
                    refused[k] = f"{SCALE}:no-evidence"
                accepted = set()
                scale_record = {"source": inputs.get("source")}
            else:
                a2 = FSG.assess_v2(cams2, metric_log, gparams)
                iso2 = {k for p in a2["pieces"] for k in p["keyframe_ids"]}
                for k in accepted & iso2:
                    refused[k] = SCALE
                scale_record = {"source": inputs.get("source"), "reference": a2.get("reference"),
                                "pieces": [{"first_index": p["first_index"], "last_index": p["last_index"],
                                            "keyframes": len(p["keyframe_ids"]), "ratio": p.get("ratio"),
                                            "replaced_in_piece": len(set(p["keyframe_ids"]) & accepted)}
                                           for p in a2["pieces"]],
                                "room_isolated_by_rerun": len(iso2 & room)}
                accepted = accepted - iso2
                if accepted:
                    cand, accepted = screen_floor_and_motion(accepted)
                    uncertified_new = []
                    for u in a2.get("uncertified") or []:
                        ids = [k for k in u["keyframe_ids"] if k in accepted]
                        if ids:
                            uncertified_new.append(dict(u, keyframe_ids=ids, keyframes=len(ids), replaced=True))
                    scale_record["uncertified"] = uncertified_new
        record["scale_screen"] = scale_record
        record["refused"] = _counts_of(refused)
        record["refused_keyframe_ids"] = {why: sorted((k for k, w in refused.items() if w == why),
                                                      key=solution.keyframe_ids.index)
                                          for why in sorted(set(refused.values()))}
        if not accepted:
            return solution, assessment, done("none-passed")
        kidx = {kid: i for i, kid in enumerate(solution.keyframe_ids)}
        out = dict(assessment)
        pieces = []
        for p in assessment.get("pieces") or []:
            ids = [k for k in p["keyframe_ids"] if k not in accepted]
            if ids:
                pieces.append(_piece_record(p, ids, kidx) if len(ids) < len(p["keyframe_ids"]) else p)
        out["pieces"] = pieces
        out["room_keyframes"] = sorted(list(assessment.get("room_keyframes") or []) + list(accepted),
                                       key=kidx.get)
        extra = (scale_record or {}).get("uncertified") or []
        if not params.scale_screen:
            extra = _replaced_spans(sorted(accepted, key=kidx.get), kidx)
        if extra:
            out["uncertified"] = list(assessment.get("uncertified") or []) + extra
            out["uncertified_keyframes"] = sum(len(u["keyframe_ids"]) for u in out["uncertified"])
            if out.get("scale_status") == FSG.SCALE_CERTIFIED:
                out["scale_status"] = FSG.SCALE_UNCERTIFIED
                out["why"] = f"{sum(len(u['keyframe_ids']) for u in extra)} re-placed keyframes are not certified"
        ordered = sorted(accepted, key=kidx.get)
        done("replaced", replaced=len(ordered), replaced_keyframe_ids=ordered)
        out["replace_cut"] = record
        return cand, out, record
    except Exception as exc:  # noqa: BLE001 -- an optional improvement: failing it is today's outcome
        logger.exception("[Tower][WorldBuilder] re-placing the cut keyframes of %s/%s failed; the guard's "
                         "decision stands", world_id, session_id)
        return solution, assessment, done("error", error=f"{type(exc).__name__}: {exc}")


def _replaced_spans(ordered: list, kidx: dict) -> list[dict]:
    """Uncertified rows for re-placed cameras no scale screen judged: one per run of consecutive keyframes."""
    runs: list[list[str]] = []
    for k in ordered:
        if runs and kidx[k] == kidx[runs[-1][-1]] + 1:
            runs[-1].append(k)
        else:
            runs.append([k])
    return [{"group": "room", "first_index": kidx[r[0]], "last_index": kidx[r[-1]], "first_keyframe_id": r[0],
             "last_keyframe_id": r[-1], "keyframes": len(r), "measured": 0, "class": None, "kind": "re-placed",
             "ratio": None, "decision": "attached", "reason": None, "uncertified": UNCERTIFIED_REPLACED,
             "keyframe_ids": r, "replaced": True} for r in runs]


def _counts_of(refused: dict) -> dict:
    out: dict = {}
    for why in refused.values():
        out[why] = out.get(why, 0) + 1
    return dict(sorted(out.items()))


def summary(record: dict | None) -> dict | None:
    """What the solution's guard summary says about this step: no ratio, no figure beyond counts."""
    if not record:
        return None
    return {"id": STEP_ID, "state": record.get("state"), "targets": record.get("targets"),
            "replaced": record.get("replaced", 0), "refused": record.get("refused") or {},
            "seconds": record.get("seconds")}
