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
     at most 3 trials an image. Intrinsics are fixed, as in the solve. Seeded and on one thread, as the
     seeded solve maps: one input, one result. In a child process (`register`), stopped at the step's
     deadline (`ReplaceParams.max_seconds` from the step's start; then nothing is re-placed); inside it a
     soft deadline `soft_margin_seconds` earlier, read before every pass and image attempt and capping
     every bundle adjustment, keeps what is registered by then.
  4. GATES (`replace_cut`), each a rule that already exists; a keyframe that fails one stays where it was
     (cut, unsupported or unposed -- today's outcome):
       a. the support floor: >= `MIN_IMAGE_OBSERVATIONS` (30) 3-D observations, the solve's own rule for a
          camera that is a measurement, counted on what the frame keeps (`anchors`: with "room", its
          observations of the ROOM's points only);
       b. the guard's pose screen (`final_scale_guard._pose_screen`) over the room and the re-placed
          cameras in capture order: a re-placed camera in a step beyond the walking allowance
          (2.0 m/s x dt + 0.3 m) or a height jump, in a hard step, with a room defect, or outside the core
          cluster is refused (consistency with its neighbours in time); and one that no chain of steps the
          screen can test (<= 3 journal indices, <= 2 s) joins to an anchor -- a room camera the screen can
          test against another room camera -- is refused: the screen cannot place an island relative to the
          room, and a lone room camera inside a collapsed stretch is no anchor;
       c. with `anchors="own"` only, the guard's scale screen, re-run (`assess_v2`) on the room plus the
          re-placed cameras with their metric ratios recomputed from the live depth cache: a re-placed camera
          the guard would isolate is refused (the rule that cut the stretch must now keep it). With "room"
          the frame brings no scale of its own (every 3-D point it keeps is the room's), so there is none to
          screen, and it is recorded `re-placed` uncertified;
       d. downstream, unchanged: the depth stage's per-frame fit (held-out <= 8 %) decides what is fused.
     Without metric evidence (the assessment's reference and metres per unit) the pose screen has no
     walking allowance in metres, so nothing is re-placed.
  5. MERGE (`merge`): a re-placed keyframe joins component 0 with its registered pose; its old observations
     are dropped; its observations of room points and the points newly triangulated with it are added. The
     room's own points and poses are never moved.

It never raises: any failure -- the deadline passed, its registration process crashed or failing -- leaves the
guard's decision as it was, with the error in the record; and if publishing its candidate fails afterwards
(the guard's `apply`, or writing its records), the guard's own cut is published instead
(`final_scale_guard.guard_publication`, `coherence_publish._records_before_publish`; record `withdrawn`). With `TOWER_WORLD_REPLACE_CUT=off`
it does not run and the publication is today's, byte for byte.

ANCHORS (`ReplaceParams.anchors`, default "room"). A re-placed stretch frame's pose is right where a room camera
vouches for it (cm level against a good draw of the same frames), but most of the points triangulated while it
was registered come from the stretch's own low-parallax pairs: on walk 10.1's live draw their depth scale read
x1.6-1.8 and, anchored on them, the depth stage pushed the surface back (a phantom). "room" keeps only a
re-placed frame's observations of room points, so it brings no scale of its own (no scale screen), and its
depth comes from the room -- its own fit on room points, or a fit borrowed from its fused neighbours
(`neighbour_fit`). "own" (gates a-c) is the measured alternative (patchy run 2026-10-19, lane R).

THE FLOOR UNDER "room" counts room points, because COLMAP's >= 30 PnP inliers may be mostly the stretch's
own chained points: measured against a good draw of the same frames (patchy run 2026-10-19, lane X), the
re-placed frames with < 30 room-point observations were less accurate than the room's own draw-to-draw
spread -- live 10.1 vs pre3, local fits: centre median 4.3 cm (room 0.8 cm), rotation median 1.3 deg (room
0.2 deg); pre1's bad draw vs its good draw: centre median 7.0 cm (room 7.4 cm, global fit) but rotation 1.9
deg (room 0.1 deg, local) -- while those with >= 30 matched the room (live: 0.2 cm, 0.1 deg). So a frame
below the floor on room points stays where it was.
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
    # THE STEP'S DEADLINE, from its start, a bound on the cost, not a quality knob. Everything counts
    # against it: the registration's input write, its child's start and the wait for it (the child is
    # STOPPED there, `_in_child` / `_stop`), and every gate round after it (`replace_cut`, `in_time`).
    # Past it nothing is re-placed and the guard's cut stands. It can be overrun only by the one piece
    # running when it passes, none of which can be interrupted: stopping the child (0.2 s measured; the
    # stop gives up after 30 s of tree kill plus 10 s of confirming the interpreter gone), or reading a
    # result that arrived in time (0.06 s on live 10.1) and one gate round (all the gates, ~1 s there).
    max_seconds: float = 180.0
    # The soft deadline inside the child, `max_seconds - soft_margin_seconds`: no pass or image attempt
    # starts after it and every bundle adjustment is capped at it, so a long registration returns what it
    # has registered instead of being killed. The margin is for the one call already running when it
    # passes (an image attempt, a triangulation, a refinement, the database read), none of which can be
    # interrupted: the longest measured on the 9 patchy-run worlds is recorded with every run
    # (`registration.longest_call_s`); 30 s is a multiple of it, so the kill is for a hang, not for a
    # registration that is merely long.
    soft_margin_seconds: float = 30.0
    # The registration loop. "pnp": register (PnP), triangulate, local BA, until a pass registers nothing.
    # "colmap": COLMAP's own incremental loop (`IncrementalPipeline`): after a pass that registers nothing,
    # one global refinement (track completion, retriangulation, global BA with the room fixed) and another
    # pass; and, when no image has visible 3-D points, COLMAP 4's structure-less registration from 2-D-2-D
    # correspondences to registered images (`structure_less_registration_fallback`, on by default).
    loop: str = "pnp"
    # Whether the guard's per-camera scale screen also judges a re-placed camera (gate c).
    scale_screen: bool = True
    # What a re-placed keyframe brings into the solution as its 3-D observations (its depth stage anchors).
    # "own": its observations of room points AND the points triangulated with it during registration.
    # "room": its observations of room points only; the points triangulated during registration (from the
    # stretch's own low-parallax pairs) are used to chain the registration and then dropped, so the room's
    # geometry is unchanged and the depth stage scales a re-placed frame only on room points (or, with too
    # few, borrows a fit from its fused neighbours where the surface does that). The support floor then
    # counts its ROOM-point observations (COLMAP's >= 30 PnP inliers may be mostly the stretch's own chained
    # points; measured, see the module's THE FLOOR UNDER "room"); no scale screen: it brings no scale.
    anchors: str = "room"

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
    from tower.world_builder.global_solve import _seed_incremental_options  # noqa: PLC0415

    opts = pycolmap.IncrementalPipelineOptions()
    # The solve keeps the calibrated pinhole fixed (global_solve._map_candidate); so does this.
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    opts.mapper.abs_pose_refine_focal_length = False
    opts.mapper.abs_pose_refine_extra_params = False
    # The room's cameras are measurements already published; only the new images move.
    opts.fix_existing_frames = True
    # Seeded, so ONE thread, exactly as the seeded solve maps (`_map_candidate`: `inc.num_threads` is the
    # mapper's threads, 1 when seeded; `_seed_incremental_options`: every seed and `mapper.num_threads` 1).
    # `num_threads` is what `get_mapper()` and the bundle adjusters read; its default, -1, is every core
    # of the Tower, inside the final solve.
    opts.num_threads = 1
    _seed_incremental_options(opts, int(params.seed))
    return opts


def register(solution, *, room_kids, target_kids, database_path, name_of: dict,
             params: ReplaceParams | None = None, deadline: float | None = None) -> Registration:
    """Register `target_kids` into the room made of `room_kids` (poses fixed) and the points they
    triangulate, with COLMAP's incremental mapper over the solve database. Reads the database only.

    BOUNDED by `deadline` (a `time.perf_counter()` instant; None: `params.max_seconds` from now; the step
    passes its own, `max_seconds` from ITS start): `_register` runs in a child process (`_in_child`)
    whose input write, spawn and wait all count against it, and which is stopped there (`TimeoutError`;
    the step then re-places nothing). Inside, `_register` keeps the soft deadline: `soft_margin_seconds`
    before it. One input, one result: the child runs the same seeded, one-thread code on the same inputs.
    The record gets `child_seconds` and `transfer_s` (input write, result read)."""
    params = params or ReplaceParams()
    started = time.perf_counter()
    if deadline is None:
        deadline = started + params.max_seconds
    timings: dict = {}
    reg = _in_child(f"{__name__}:_register", kwargs=dict(
        solution=solution, room_kids=list(room_kids), target_kids=list(target_kids),
        database_path=str(database_path), name_of=dict(name_of), params=params,
        deadline_s=deadline - started - params.soft_margin_seconds), deadline=deadline, timings=timings)
    reg.record["child_seconds"] = round(time.perf_counter() - started, 3)
    reg.record["transfer_s"] = {"write": timings.get("write_s"), "read": timings.get("read_s")}
    return reg


def _register(solution, *, room_kids, target_kids, database_path, name_of: dict,
              params: ReplaceParams | None = None, deadline_s: float | None = None) -> Registration:
    """`register`'s work, in this process: stops starting work at `deadline_s` after it started (None:
    `params.max_seconds - params.soft_margin_seconds`)."""
    import pycolmap  # noqa: PLC0415 -- optional, as everywhere in the solve

    params = params or ReplaceParams()
    if deadline_s is None:
        deadline_s = params.max_seconds - params.soft_margin_seconds
    started = time.perf_counter()
    longest: dict[str, float] = {}

    def timed(kind, fn, *a):
        t0 = time.perf_counter()
        try:
            return fn(*a)
        finally:
            longest[kind] = max(longest.get(kind, 0.0), time.perf_counter() - t0)

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
        cache = timed("database", pycolmap.DatabaseCache.create, db, copts)
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
    colmap_loop = params.loop == "colmap"
    prev_success = True

    # THE SOFT DEADLINE: read before every pass and every image attempt (one attempt is one COLMAP pose
    # estimate: RANSAC under its own trial cap, then a refinement), and every bundle adjustment is told
    # what is left (Ceres' `max_solver_time_in_seconds`, read only when it is reached). An image not
    # registered by then stays where it was. A call already running cannot be interrupted: that is what
    # the margin to the hard kill (`register`) is for. Finishing in time, nothing here changes the result.
    def left() -> float:
        return deadline_s - (time.perf_counter() - started)

    def capped(options):
        options.ceres.solver_options.max_solver_time_in_seconds = max(left(), 1e-3)
        return options

    stopped_by_budget = left() <= 0
    while not stopped_by_budget:
        nxt = [i for i in mapper.find_next_images(mopts, False) if i in target_images]
        less = False
        if not nxt and colmap_loop and opts.structure_less_registration_fallback:
            nxt = [i for i in mapper.find_next_images(mopts, True) if i in target_images]
            less = True
        placed = None
        for iid in nxt:
            if left() <= 0:
                stopped_by_budget = True
                break
            attempts += 1
            ok = timed("attempt", mapper.register_next_structure_less_image if less
                       else mapper.register_next_image, mopts, iid)
            if ok:
                placed = iid
                break
        if placed is not None:
            structure_less += int(less)
            pnp_inliers[placed] = int(rec.image(placed).num_points3D)
            timed("triangulate", mapper.triangulate_image, tri, placed)
            timed("refine", mapper.iterative_local_refinement, int(opts.ba_local_max_refinements),
                  float(opts.ba_local_max_refinement_change), mopts, capped(ba), tri, placed)
            prev_success = True
            stopped_by_budget = left() <= 0
            continue
        if stopped_by_budget:
            break
        if colmap_loop and prev_success:
            timed("global_refine", mapper.iterative_global_refinement, int(opts.ba_global_max_refinements),
                  float(opts.ba_global_max_refinement_change), mopts,
                  capped(opts.get_global_bundle_adjustment()), tri, False)
            global_refinements += 1
            prev_success = False
            stopped_by_budget = left() <= 0
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
        "longest_call_s": {k: round(v, 3) for k, v in sorted(longest.items())},
        "colmap": {"mapper": {k: v for k, v in mopts.todict().items() if isinstance(v, (int, float, bool))},
                   "triangulation": {k: v for k, v in tri.todict().items() if isinstance(v, (int, float, bool))},
                   "min_num_matches": int(opts.min_num_matches)},
    }
    mapper.end_reconstruction(False)
    return reg


def _in_child(target: str, *, args: tuple = (), kwargs: dict | None = None, timeout: float | None = None,
              deadline: float | None = None, timings: dict | None = None):
    """`module:function`(*args, **kwargs) in a child Python process, its result returned.

    BOUNDED by `deadline` (a `time.perf_counter()` instant), else `timeout` seconds from the call: writing
    the input counts against it (past it, no child is started), and the child is STOPPED there
    (`TimeoutError`, `_stop`). Reading a result that arrived in time is the one step after it; its time is
    in `timings["read_s"]` (and the input write's in `"write_s"`). A crash or non-zero exit is a
    `RuntimeError`. Inputs and the result travel as pickles in a private temporary directory, which is
    also the child's working directory; the child imports this tree's `tower`; its output (COLMAP's log)
    goes to a file there, and its tail is in any error.

    The INTERPRETER is not always the process started: on Windows a venv's python.exe is a launcher that
    runs the interpreter as its child. So the child writes its own PID first, and the parent opens a
    handle to it (`_Interpreter`), which also keeps the PID from being reused; `_stop` uses it to confirm
    the interpreter is gone, killing it by its own handle if the tree kill did not."""
    import os  # noqa: PLC0415
    import pickle  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    started = time.perf_counter()
    end = deadline if deadline is not None else started + float(timeout)
    timings = timings if timings is not None else {}
    root = str(Path(__file__).resolve().parents[2])        # the directory that holds the `tower` package
    paths = [root] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != root]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths))
    with tempfile.TemporaryDirectory(prefix="replace-cut-", ignore_cleanup_errors=True) as tmp:
        inp, out, log, pid = (Path(tmp) / n for n in ("in.pkl", "out.pkl", "child.log", "pid"))
        inp.write_bytes(pickle.dumps((target, tuple(args), dict(kwargs or {})), protocol=pickle.HIGHEST_PROTOCOL))
        timings["write_s"] = round(time.perf_counter() - started, 3)
        if time.perf_counter() >= end:
            raise TimeoutError(f"{target}: no time left to start it")
        code = ("import os, sys; open(sys.argv[3] + '.tmp', 'w').write(str(os.getpid())); "
                "os.replace(sys.argv[3] + '.tmp', sys.argv[3]); "
                "from tower.world_builder.replace_cut import _child_main; _child_main(sys.argv[1], sys.argv[2])")
        interpreter = None
        with open(log, "wb") as sink:
            proc = subprocess.Popen([sys.executable, "-c", code, str(inp), str(out), str(pid)], cwd=tmp, env=env,
                                    stdin=subprocess.DEVNULL, stdout=sink, stderr=subprocess.STDOUT,
                                    start_new_session=os.name != "nt")
            try:
                while True:
                    if interpreter is None:
                        interpreter = _Interpreter.of(proc, pid)
                    left = end - time.perf_counter()
                    if left <= 0:
                        timings.update(_stop(proc, interpreter, pid))
                        raise TimeoutError(f"{target} ran past its deadline and was stopped")
                    try:
                        status = proc.wait(timeout=min(0.05, left))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if interpreter is not None:
                    interpreter.close()
        t_read = time.perf_counter()
        if status != 0 or not out.exists():
            tail = log.read_text(encoding="utf-8", errors="replace")[-400:]
            raise RuntimeError(f"{target} failed in its process (exit {status}): {tail}")
        ok, value = pickle.loads(out.read_bytes())
        timings["read_s"] = round(time.perf_counter() - t_read, 3)
    if not ok:
        raise RuntimeError(value)
    return value


class _Interpreter:
    """The child's interpreter process, from the PID it wrote, held by a handle so the PID cannot be
    reused while the parent may still kill it. On POSIX the interpreter is the process started."""

    def __init__(self, pid: int, handle=None, proc=None):
        self.pid, self._handle, self._proc = pid, handle, proc

    @classmethod
    def of(cls, proc, pid_file) -> "_Interpreter | None":
        import os  # noqa: PLC0415

        if os.name != "nt":
            return cls(proc.pid, proc=proc)
        try:
            pid = int(pid_file.read_text())
        except (OSError, ValueError):
            return None                           # not written yet
        k32 = _kernel32()
        handle = k32.OpenProcess(0x0001 | 0x00100000, False, pid)     # PROCESS_TERMINATE | SYNCHRONIZE
        return cls(pid, handle=handle) if handle else cls(pid)        # no handle: it has already exited

    def gone(self, wait_s: float = 0.0) -> bool:
        if self._proc is not None:
            try:
                self._proc.wait(timeout=wait_s)
                return True
            except Exception:  # noqa: BLE001 -- subprocess.TimeoutExpired
                return False
        if not self._handle:
            return True
        return _kernel32().WaitForSingleObject(self._handle, int(wait_s * 1000)) == 0    # WAIT_OBJECT_0

    def kill(self) -> None:
        if self._proc is not None:
            self._proc.kill()
        elif self._handle:
            _kernel32().TerminateProcess(self._handle, 1)

    def close(self) -> None:
        if self._handle:
            _kernel32().CloseHandle(self._handle)
            self._handle = None


def _kernel32():
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.WaitForSingleObject.restype = wintypes.DWORD
    k32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    k32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    return k32


def _stop(proc, interpreter, pid_file) -> dict:
    """Stop the child at its deadline and CONFIRM its interpreter is gone: the tree kill (`_kill_tree`),
    then, if the interpreter is still there -- the tree kill failed, or the launcher was killed before
    the interpreter wrote its PID -- the interpreter killed by its own handle. Raises when it cannot be
    confirmed gone. Returns what it took ({stop_s, tree_killed, killed_by_pid, interpreter_pid})."""
    t0 = time.perf_counter()
    tree = _kill_tree(proc)
    if interpreter is None:                       # killed while it was starting: wait for its PID briefly
        for _ in range(40):
            interpreter = _Interpreter.of(proc, pid_file)
            if interpreter is not None:
                break
            time.sleep(0.05)
    by_pid = False
    try:
        if interpreter is not None and not interpreter.gone(0.5):
            interpreter.kill()
            by_pid = True
            if not interpreter.gone(10.0):
                raise RuntimeError(f"the child interpreter {interpreter.pid} could not be stopped")
    finally:
        if interpreter is not None:
            interpreter.close()
        if proc.poll() is None:                   # a launcher whose interpreter is gone exits by itself
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001 -- subprocess.TimeoutExpired
                proc.kill()
                proc.wait()
    return {"stop_s": round(time.perf_counter() - t0, 3), "tree_killed": tree, "killed_by_pid": by_pid,
            "interpreter_pid": None if interpreter is None else interpreter.pid}


def _kill_tree(proc) -> bool:
    """Kill `proc` and every process it started (best effort; True when the tree kill reported success),
    then reap `proc` itself."""
    import os  # noqa: PLC0415
    import signal  # noqa: PLC0415

    ok = _taskkill_tree(proc.pid) if os.name == "nt" else _killpg(proc.pid, signal.SIGKILL)
    try:
        proc.kill()
    except OSError:
        pass
    proc.wait()
    return ok


def _taskkill_tree(pid: int) -> bool:
    import subprocess  # noqa: PLC0415

    try:
        done = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=30)
        return done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _killpg(pid: int, sig) -> bool:
    import os  # noqa: PLC0415

    try:
        os.killpg(pid, sig)                       # its own session (`start_new_session`)
        return True
    except OSError:
        return False


def _child_main(inp: str, out: str) -> None:
    """The child side of `_in_child`: run the pickled call, pickle (ok, result or the error's text)."""
    import importlib  # noqa: PLC0415
    import pickle  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    target, args, kwargs = pickle.loads(Path(inp).read_bytes())
    module, name = target.split(":")
    try:
        result = (True, getattr(importlib.import_module(module), name)(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 -- reported to the parent, which raises it
        result = (False, f"{type(exc).__name__}: {exc}")
    Path(out).write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------


def merge(solution, reg: Registration, accepted, *, room_kids, new_points: bool = True):
    """The solution with every `accepted` keyframe re-placed: in component 0 at its registered pose, its old
    observation rows dropped (a point left with none is dropped too), its rows to room points and the points
    triangulated with it added. A new point keeps only observations of room or accepted keyframes that no
    other point already holds, and needs two. The room's points and poses are not moved. Track lengths
    follow the rows: a kept point loses one per row dropped and gains one per row added."""
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
    # every observation row dropped here leaves its point's track: a point keeps the support it still has
    # (the photo floorplan selects points by track length)
    removed = np.bincount(obs[~keep, 2], minlength=n_old)
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
    tlen = (np.asarray(solution.track_length, np.int64) - removed)[alive].astype(np.int32)
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
    for pid in sorted(reg.points if new_points else ()):
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


def motion_refusals(room_cams, new_cams, mpu: float, gparams, deadline: float | None = None) -> dict:
    """{kid: why} for the re-placed cameras the guard's pose screen, run over the room and the re-placed
    cameras in capture order, does not accept. Each round: a re-placed camera outside the screen's core
    cluster or with a room defect is refused; only when there is none, the re-placed camera(s) in the most
    steps beyond the walking allowance (or hard steps, or height jumps) -- a room camera is never blamed,
    and a lone outlier between two re-placed neighbours takes the blame alone. Then the screen runs again
    without the refused cameras, until it accepts every one left. Past `deadline` (a `time.perf_counter()`
    instant) it raises `TimeoutError` before starting another round."""
    from tower.world_builder import final_scale_guard as FSG  # noqa: PLC0415

    out: dict = {}
    pending = list(new_cams)
    while pending:
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("the step ran past its deadline (in the pose screen)")
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

    # An ANCHOR is a room camera the screen can test against another room camera: a room camera that is
    # itself an island in time (a lone survivor of a collapsed stretch) carries a chain but cannot start one.
    room_seq = sorted(room_cams, key=lambda c: c.index)
    anchors = {c.kid for i, c in enumerate(room_seq)
               if (i > 0 and testable(room_seq[i - 1], c)) or (i + 1 < len(room_seq) and testable(c, room_seq[i + 1]))}
    anchored = set()
    for seq in (cams, cams[::-1]):
        prev = None
        reach = False
        for c in seq:
            a, b = (prev, c) if seq is cams else (c, prev)
            linked = prev is not None and testable(a, b)
            reach = c.kid in anchors or (reach and linked)
            if reach and c.kid in new:
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
    # THE STEP'S DEADLINE, from its start: the registration's child is stopped there (`register`), and
    # every gate round after it checks it, so a step past it re-places nothing (the guard's cut).
    deadline = started + params.max_seconds

    def in_time(stage: str) -> None:
        if time.perf_counter() > deadline:
            raise TimeoutError(f"the step ran past its {params.max_seconds:g} s deadline ({stage})")

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
            return solution, assessment, done("skipped", why="no metric evidence: the pose screen needs metres per unit")
        mpu = float(assessment["pose"]["metres_per_unit"])
        room_cams = [c for c in cams if c.kid not in pieces_kids and FSG._finite_pose(c)]
        room = {c.kid for c in room_cams}
        reg = (register_fn or register)(solution, room_kids=room, target_kids=targets,
                                        database_path=database_path, name_of=name_of, params=params,
                                        deadline=deadline)
        record["registration"] = reg.record
        in_time("after the registration")
        accepted = set(reg.poses)
        refused: dict[str, str] = {k: NOT_REGISTERED for k in targets if k not in accepted}

        own = params.anchors == "own"
        kidx0 = {kid: i for i, kid in enumerate(solution.keyframe_ids)}
        t_of = {k.keyframe_id: float(k.received_at) for k in keyframes}

        def screen_floor_and_motion(acc):
            while True:
                in_time("in the gates")
                cand = merge(solution, reg, acc, room_kids=room, new_points=own)
                # the floor on what the frame keeps: with "room", its observations of room points only
                floor = {k for k in acc if int(cand.poses[k]["observations"]) < params.min_obs}
                for k in floor:
                    refused[k] = BELOW_FLOOR
                acc = acc - floor
                cand = merge(solution, reg, acc, room_kids=room, new_points=own) if floor else cand
                new_cams = [FSG.RoomCamera(kid=k, name=name_of.get(k, k), index=kidx0[k], t=t_of.get(k),
                                           R=reg.poses[k][0], tvec=reg.poses[k][1],
                                           group=(groups_of or {}).get(name_of.get(k), "room"))
                            for k in sorted(acc, key=kidx0.get)]
                motion = motion_refusals(room_cams, new_cams, mpu, gparams, deadline=deadline)
                for k, why in motion.items():
                    refused[k] = f"{MOTION}:{why}"
                acc = acc - set(motion)
                if not floor and not motion:
                    return cand, acc

        cand, accepted = screen_floor_and_motion(accepted)
        scale_record = None
        if accepted and params.scale_screen and own:
            metric_log, inputs = FSG.read_evidence(store, world_id, session_id, cand, None,
                                                   database_path=database_path, name_of=name_of,
                                                   live_metric_fn=live_metric_fn)
            in_time("after the scale evidence")
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
        if not (params.scale_screen and own):
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


def publication_identity(solution) -> str | None:
    """Which publication of a solve this is, when re-placement changed it; None when it did not (no pose
    row carries `registration`), so every cache identity of an untouched solve stays as it was.

    WHY. `input_digest` names the keyframes only, and the step changes poses and observations without
    changing it: the guard's cut and the re-placed publication of ONE solve (the switch off or on, a
    re-gate) share it, so a surface or a depth stage keyed by it would be reused across them. This hashes
    everything the step writes (`merge`), which includes all a depth fit reads (`dense_pipeline._fit_record`:
    poses, observation rows, their image coordinates, points): the solve's identity
    (`global_solve.solve_identity`: `solved_at`, every pose and its component), each pose's observation
    count, and every point and observation array. 16 hex."""
    poses = getattr(solution, "poses", None) or {}
    if not any((p or {}).get("registration") for p in poses.values()):
        return None
    from tower.world_builder.global_solve import solve_identity  # noqa: PLC0415

    h = hashlib.sha1(solve_identity(solution).encode())
    h.update(json.dumps({k: int((p or {}).get("observations", 0)) for k, p in poses.items()},
                        sort_keys=True).encode())
    for name, dtype in _PUBLISHED_ARRAYS:
        h.update(name.encode())
        h.update(np.ascontiguousarray(np.asarray(getattr(solution, name), dtype)).tobytes())
    return h.hexdigest()[:16]


# Every array of the solution `merge` writes, with the dtype it writes it in (`publication_identity`).
_PUBLISHED_ARRAYS = (("xyz", np.float32), ("rgb", np.uint8), ("component", np.int32),
                     ("first_keyframe", np.int32), ("track_length", np.int32), ("error", np.float32),
                     ("observations", np.int32), ("observation_xy", np.float32))


def withdrawn(record: dict, why: str) -> dict:
    """The record of a re-placement that was not published (its candidate failed after the step, in the
    guard's publication): state `error`, nothing re-placed, and the keyframes it had re-placed."""
    return dict(record, state="error", error=why, replaced=0, replaced_keyframe_ids=[],
                withdrawn_keyframe_ids=list(record.get("replaced_keyframe_ids") or []))


def summary(record: dict | None) -> dict | None:
    """What the solution's guard summary says about this step: no ratio, no figure beyond counts."""
    if not record:
        return None
    return {"id": STEP_ID, "state": record.get("state"), "targets": record.get("targets"),
            "replaced": record.get("replaced", 0), "refused": record.get("refused") or {},
            "seconds": record.get("seconds")}
