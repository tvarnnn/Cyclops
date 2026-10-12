"""Opt-in, bounded background mapping with a guarded full-prefix publication.

The final solver never calls this module. Its feature database and mapper workspace
stay separate from the live window's rolling workspace.
"""

from __future__ import annotations

import dataclasses
from contextlib import closing
import math
import os
import sqlite3
import time
import uuid

import numpy as np

from tower.world_builder import global_solve
from tower.world_builder.store import compute_input_digest


WINDOW_KEYFRAMES = 100
# The window always re-solves this many already-published keyframes before the
# newest frames, so the active window and its anchors share views by construction.
OVERLAP_KEYFRAMES = 20
MAX_ANCHORS = 6
# Anchors come from the published poses at most this far before the window start.
# W1 (2026-10-12) drew them from 150 frames back; two of four windows then lost an
# anchor ("anchor missing or disconnected"). Near anchors are co-visible anchors.
ANCHOR_LOOKBACK = 40
# A Sim(3) needs three; four registered anchors leave one redundant check.
MIN_ALIGNED_ANCHORS = 4
MIN_ANCHOR_OBSERVATIONS = global_solve.MIN_IMAGE_OBSERVATIONS
# EVERY LENGTH TEST IS A FRACTION OF THE SCENE DEPTH: the median distance from the
# anchor cameras to the points they observe, in the published solution. The published
# gauge is as arbitrary as the window's, so an absolute 0.35 meant a different thing
# in every world; and the anchor spread is no unit either, because a wearer who turns
# more than walks leaves the anchors almost on one point. Offline on W2 (2026-10-12,
# seeded GLOMAP, final feature database): anchors spread 1.26 against a depth of
# 12.2, and honest windows fitted at 0.9-29% of the spread but 0.1-3% of the depth.
# A residual of 5% of the depth is a camera displaced by ~3 degrees as seen from what
# it looks at. Without observations (no points) the anchor spread stands in.
MAX_ANCHOR_RESIDUAL_FRACTION = 0.05
MAX_ANCHOR_ROTATION_DEG = 8.0
# Anchors on one point fix no scale; judged against the scene depth (gauge-free).
DEGENERATE_SPREAD_FRACTION = 1e-4
# A fresh window map has its own arbitrary gauge (GLOMAP normalises every map), so
# the anchor scale is NOT expected near 1: W1's solve 5 aligned at 0.0601 with a
# 0.0012 anchor residual. Only a degenerate fit is refused here; agreement is
# judged by the residuals after alignment, in the published solution's units.
MIN_SCALE = 0.001
MAX_SCALE = 1000.0
MAX_ACTIVE_FAILURE_FRACTION = 0.25
MAX_POSE_RESIDUAL_FRACTION = 0.05
MAX_POSE_ROTATION_DEG = 8.0
# The floor of the per-keyframe step a new camera may take from its trusted
# predecessor, as a fraction of the scene depth. Keyframe steps are heavy-tailed: in
# W2's final solve the median step is 0.144 and the p99 is 5.52 (38x), so "4x the
# recent median" alone refused W2's honest keyframe 137 (3.3 final units after a
# turn) and, through the support chain, every camera after it. Per keyframe, against
# each camera's own scene depth, an honest step in that final solve is 0.026 at the
# median, 0.39 at p99 and 1.06 at most (over 900 steps): the floor sits above.
MIN_STEP_LIMIT_FRACTION = 1.5
# A published camera whose centre coincides with another's is a collapsed pose, not
# a validated one: W2's 21-keyframe bootstrap put 12 of its 21 cameras on one point,
# and three of them became anchors.
COLLAPSED_CENTRE_FRACTION = 1e-6
MIN_NEW_CAMERA_POINTS = 6
MAX_REPROJECTION_MEDIAN_PX = 3.0
MAX_REPROJECTION_P95_PX = 6.0


class WindowRefused(ValueError):
    """A candidate must not replace the last published solution."""


@dataclasses.dataclass(frozen=True)
class WindowSelection:
    active_ids: list[str]
    anchor_ids: list[str]
    solve_ids: list[str]


def prepare_window_database(window_workspace, canonical_workspace, image_names: set[str]) -> dict:
    """Keep incremental overlap matches while removing every out-of-window pair.

    The first window can reuse the ordinary background solve's feature database.
    Later windows prune their own database. COLMAP's sequential matcher sees only
    rows in ``images``, not the ``image_names`` extraction argument.
    """
    started = time.perf_counter()
    database = window_workspace.database_path
    if not database.exists():
        if not canonical_workspace.database_path.exists():
            return {"copy_s": 0.0, "prune_s": 0.0,
                    "removed_images": 0, "kept_images": 0}
        window_workspace.root.mkdir(parents=True, exist_ok=True)
        try:
            with closing(sqlite3.connect(canonical_workspace.database_path)) as source:
                with closing(sqlite3.connect(database)) as target:
                    source.backup(target)
        except sqlite3.Error as exc:
            raise WindowRefused(f"window feature database could not be copied: {exc}") from exc
    copied = time.perf_counter()
    removed = set()
    try:
        with closing(sqlite3.connect(database)) as con:
            with con:
                images = con.execute("SELECT image_id, name FROM images").fetchall()
                removed = {int(image_id) for image_id, name in images if name not in image_names}
                if removed:
                    for table in ("matches", "two_view_geometries"):
                        pair_ids = [int(row[0]) for row in con.execute(f"SELECT pair_id FROM {table}")]
                        discarded = [(pid,) for pid in pair_ids
                                     if pid // 2147483647 in removed or pid % 2147483647 in removed]
                        con.executemany(f"DELETE FROM {table} WHERE pair_id = ?", discarded)
                    for table in ("keypoints", "descriptors", "images"):
                        con.executemany(f"DELETE FROM {table} WHERE image_id = ?",
                                        [(image_id,) for image_id in removed])
                count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
                if count > len(image_names):
                    raise WindowRefused("window feature database exceeds selected images")
    except (sqlite3.Error, OSError) as exc:
        raise WindowRefused(f"window feature database could not be pruned: {exc}") from exc
    return {"copy_s": round(copied - started, 3),
            "prune_s": round(time.perf_counter() - copied, 3),
            "removed_images": len(removed), "kept_images": count}


def _validated_published(previous: global_solve.Solution) -> set[str]:
    """Published poses a window may start from or anchor on.

    Enough observations, and a centre of its own: a camera on exactly the same
    point as another published camera is a collapsed pose (W2's bootstrap), not
    two measurements that happen to agree.
    """
    observed = {kid for kid, pose in previous.poses.items()
                if int(pose.get("observations") or 0) >= MIN_ANCHOR_OBSERVATIONS}
    return observed - _collapsed(previous) if observed else set()


def _collapsed(solution: global_solve.Solution) -> set[str]:
    """Cameras sharing one centre with another camera of the same solution.

    GLOMAP's global positioning can put a cluster of cameras, and the points they
    see, on a single point: W2's 21-keyframe bootstrap held 12 such cameras, and an
    offline W2 window held 3 with their points 0.00018 from the camera against a
    depth of ~100 elsewhere. Their reprojection stays good, so only this finds them.
    """
    centres = {kid: global_solve._pose_matrices(pose)[1] for kid, pose in solution.poses.items()}
    if len(centres) < 2:
        return set()
    stacked = np.asarray(list(centres.values()), np.float64)
    extent = float(np.max(np.ptp(stacked, axis=0)))
    tolerance = COLLAPSED_CENTRE_FRACTION * extent
    if not math.isfinite(tolerance) or tolerance <= 0:
        return set()
    cells: dict[tuple, list[str]] = {}
    for kid, centre in centres.items():
        cells.setdefault(tuple(np.floor(centre / tolerance).astype(np.int64)), []).append(kid)
    collapsed = set()
    for (x, y, z), members in cells.items():
        neighbours = [kid for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                      for kid in cells.get((x + dx, y + dy, z + dz), ())]
        for kid in members:
            if any(other != kid and
                   np.linalg.norm(centres[other] - centres[kid]) <= tolerance
                   for other in neighbours):
                collapsed.add(kid)
    return collapsed


def select_window(ids: list[str], previous: global_solve.Solution,
                  *, active_size: int = WINDOW_KEYFRAMES) -> WindowSelection:
    """Choose the frames after the published frontier and six anchors before it.

    The FRONTIER is the last keyframe with a validated published pose. When the
    newest keyframe is within ``active_size`` of the frontier's overlap, the
    active window is the newest ``active_size`` frames, as before. Otherwise it
    is a LAGGED window: ``active_size`` frames starting ``OVERLAP_KEYFRAMES``
    before the frontier, which advances from the frontier instead of jumping to
    the newest frames. A jump is what W1 did
    (2026-10-12): its second solve mapped keyframes 336-435 against anchors from
    a 70-keyframe solution, shared no view with them, and was refused; so were
    the three after it. Anchors are fixed for this solve from the *published*
    solution, within ``ANCHOR_LOOKBACK`` frames before the window start; bins
    favour observations without letting one textured patch supply every anchor.
    """
    if previous.keyframe_ids != ids[:len(previous.keyframe_ids)]:
        raise WindowRefused("previous solution is not the accepted prefix")
    trusted = _validated_published(previous)
    frontier = max((i for i, kid in enumerate(previous.keyframe_ids) if kid in trusted),
                   default=-1)
    if frontier < 0:
        raise WindowRefused("no published poses to anchor the window")
    overlap = min(OVERLAP_KEYFRAMES, max(1, active_size // 5))
    start = max(0, frontier - overlap + 1)
    if len(ids) - start > active_size:
        active = ids[start:start + active_size]
    else:
        active = ids[-active_size:]
    candidates = [(i, kid, pose) for i, kid in enumerate(previous.keyframe_ids)
                  if max(0, start - ANCHOR_LOOKBACK) <= i <= frontier
                  if kid in trusted
                  for pose in [previous.poses[kid]]]
    if not candidates:
        raise WindowRefused("no validated anchor poses in the previous solution")
    counts: dict[int, int] = {}
    for _, _, pose in candidates:
        comp = int(pose["component"])
        counts[comp] = counts.get(comp, 0) + 1
    main = max(counts, key=lambda comp: (counts[comp], -comp))
    candidates = [item for item in candidates if int(item[2]["component"]) == main]
    if len(candidates) < 3:
        raise WindowRefused("fewer than three validated anchors in one component")
    n = min(MAX_ANCHORS, len(candidates))
    chosen = []
    for bin_index in range(n):
        lo = bin_index * len(candidates) // n
        hi = (bin_index + 1) * len(candidates) // n
        chosen.append(max(candidates[lo:hi],
                          key=lambda item: (int(item[2]["observations"]), item[0]))[1])
    selected = set(active) | set(chosen)
    return WindowSelection(active, chosen, [kid for kid in ids if kid in selected])


def _aligned_anchors(previous, candidate, anchors):
    """The anchors the fit may use: registered, in one candidate component.

    Every anchor must be a published pose of one component. At least
    ``MIN_ALIGNED_ANCHORS`` of them (all of them when fewer were chosen) must be
    registered in the candidate, with enough observations, in the candidate
    component holding most of them. An anchor the window failed to register is
    left out of the fit instead of refusing the window; the fit on the others
    is still checked anchor by anchor below.
    """
    if any(kid not in previous.poses for kid in anchors):
        raise WindowRefused("anchor missing in the published solution")
    old_components = {int(previous.poses[kid]["component"]) for kid in anchors}
    if len(old_components) != 1:
        raise WindowRefused("anchors span published components")
    collapsed = _collapsed(candidate)
    registered = [kid for kid in anchors
                  if kid in candidate.poses and kid not in collapsed and
                  int(candidate.poses[kid].get("observations") or 0) >= MIN_ANCHOR_OBSERVATIONS]
    counts: dict[int, int] = {}
    for kid in registered:
        comp = int(candidate.poses[kid]["component"])
        counts[comp] = counts.get(comp, 0) + 1
    need = min(MIN_ALIGNED_ANCHORS, len(anchors))
    if not counts:
        raise WindowRefused("anchor missing or disconnected in window solve")
    new_component = max(counts, key=lambda comp: (counts[comp], -comp))
    used = [kid for kid in registered if int(candidate.poses[kid]["component"]) == new_component]
    if len(used) < need:
        raise WindowRefused("anchor missing or disconnected in window solve")
    return old_components.pop(), new_component, used


def _alignment(previous, candidate, anchors):
    old_centres, new_centres, old_rotations, new_rotations = [], [], [], []
    old_component, new_component, anchors = _aligned_anchors(previous, candidate, anchors)
    for kid in anchors:
        old = previous.poses[kid]
        new = candidate.poses[kid]
        old_r, old_c = global_solve._pose_matrices(old)
        new_r, new_c = global_solve._pose_matrices(new)
        old_centres.append(old_c)
        new_centres.append(new_c)
        old_rotations.append(old_r)
        new_rotations.append(new_r)
    old_centres = np.asarray(old_centres)
    new_centres = np.asarray(new_centres)
    old_mean = old_centres.mean(axis=0)
    new_mean = new_centres.mean(axis=0)
    old_zero = old_centres - old_mean
    new_zero = new_centres - new_mean
    span = float(np.max(np.linalg.norm(old_zero, axis=1)))
    depth = _scene_depth(previous, anchors)
    unit = depth if math.isfinite(depth) and depth > 0 else span
    # Degenerate is judged without a gauge: anchors on one point, against the scene
    # depth in the published solution and against nothing in the window's.
    if not (span > DEGENERATE_SPREAD_FRACTION * unit and
            float(np.max(np.linalg.norm(new_zero, axis=1))) > 0):
        raise WindowRefused("anchor centres are geometrically degenerate")
    # THE ROTATION COMES FROM THE ANCHORS' ORIENTATIONS, not their centres. A wearer
    # turns far more than he walks, so the anchor centres sit in a small cluster
    # (offline W2: 10% of the scene depth) and a Procrustes fit of the centres alone
    # was 56 degrees off, while each camera's own orientation is measured well: the
    # same window then fitted every anchor orientation within 0.5 degrees. The scale
    # and translation that remain are a linear least-squares fit of the centres.
    rotation_sum = sum(old_r @ new_r.T for old_r, new_r in zip(old_rotations, new_rotations))
    u, _singular, vt = np.linalg.svd(rotation_sum)
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    turned = (rotation @ new_zero.T).T
    variance = float(np.sum(turned * turned))
    scale = float(np.sum(old_zero * turned) / variance) if variance > 0 else float("nan")
    if not math.isfinite(scale) or not MIN_SCALE <= scale <= MAX_SCALE:
        raise WindowRefused("anchor scale is outside the accepted interval")
    translation = old_mean - scale * rotation @ new_mean
    aligned = scale * (rotation @ new_centres.T).T + translation
    position_residual = np.linalg.norm(aligned - old_centres, axis=1)
    if float(np.max(position_residual)) > MAX_ANCHOR_RESIDUAL_FRACTION * unit:
        raise WindowRefused("anchor position residual is too high")
    for old_r, new_r in zip(old_rotations, new_rotations):
        delta = old_r.T @ rotation @ new_r
        angle = math.degrees(math.acos(float(np.clip((np.trace(delta)-1) / 2, -1, 1))))
        if angle > MAX_ANCHOR_ROTATION_DEG:
            raise WindowRefused("anchor orientation residual is too high")
    return (old_component, new_component, scale, rotation, translation,
            float(max(position_residual)), anchors, unit)


def _scene_depth(solution, kids) -> float:
    """Median camera-to-point distance over the observations of `kids`; NaN if none."""
    index = {kid: i for i, kid in enumerate(solution.keyframe_ids)}
    rows = solution.observations
    if not len(rows) or not len(solution.xyz):
        return float("nan")
    distances = []
    for kid in kids:
        if kid not in index or kid not in solution.poses:
            continue
        mine = rows[(rows[:, 0] == index[kid]) & (rows[:, 2] >= 0) & (rows[:, 2] < len(solution.xyz))]
        if len(mine):
            centre = global_solve._pose_matrices(solution.poses[kid])[1]
            distances.append(np.linalg.norm(solution.xyz[mine[:, 2]].astype(np.float64) - centre,
                                            axis=1))
    if not distances:
        return float("nan")
    return float(np.median(np.concatenate(distances)))


def _aligned_pose(pose, scale, rotation, translation, component):
    r_wc, centre = global_solve._pose_matrices(pose)
    centre = scale * rotation @ centre + translation
    r_cw = (rotation @ r_wc).T
    return {**pose, "component": component, "rotation": r_cw.reshape(-1).tolist(),
            "translation": (-r_cw @ centre).tolist()}


def _pose_agrees(old, aligned, unit):
    old_r, old_c = global_solve._pose_matrices(old)
    new_r, new_c = global_solve._pose_matrices(aligned)
    cosine = float(np.clip((np.trace(old_r.T @ new_r) - 1) / 2, -1, 1))
    return (np.linalg.norm(old_c - new_c) <= MAX_POSE_RESIDUAL_FRACTION * unit and
            math.degrees(math.acos(cosine)) <= MAX_POSE_ROTATION_DEG)


def _camera_point_errors(candidate, kid):
    if candidate.camera is None or len(candidate.observation_xy) != len(candidate.observations):
        return {}
    index = candidate.keyframe_ids.index(kid)
    pose = candidate.poses[kid]
    r = np.asarray(pose["rotation"], np.float64).reshape(3, 3)
    t = np.asarray(pose["translation"], np.float64)
    cam = candidate.camera
    rows = candidate.observations
    selected = ((rows[:, 0] == index) & (rows[:, 2] >= 0) &
                (rows[:, 2] < len(candidate.xyz)))
    point_ids = rows[selected, 2]
    if not len(point_ids):
        return {}
    positions = (r @ candidate.xyz[point_ids].astype(np.float64).T).T + t
    depth = positions[:, 2]
    valid = depth > 1e-9
    pixels = candidate.observation_xy[selected]
    x = cam["fx"] * positions[valid, 0] / depth[valid] + cam["cx"]
    y = cam["fy"] * positions[valid, 1] / depth[valid] + cam["cy"]
    errors = {}
    for point, error in zip(point_ids[valid], np.hypot(x - pixels[valid, 0],
                                                      y - pixels[valid, 1])):
        point = int(point)
        error = float(error)
        if math.isfinite(error):
            errors[point] = min(errors.get(point, math.inf), error)
    return errors


def _validated_active(previous, candidate, ids, selection, new_comp,
                      scale, rotation, translation, old_comp, unit):
    # Only a VALIDATED published pose is a reference: a collapsed or weakly observed
    # one is re-placed like a new camera, on the same evidence, instead of being
    # either kept or allowed to veto its correction.
    validated = _validated_published(previous)
    collapsed = _collapsed(candidate)
    aligned = {}
    failures = []
    # FAILED IS TWO THINGS. A camera that CONTRADICTS the evidence -- a re-solved
    # published pose that moved, a step no walk takes, observations that do not
    # reproject, a collapse onto another camera's centre -- says the window is
    # wrong. A camera that is merely UNVERIFIED -- not registered, in another
    # piece, or past a break in the chain of shared points -- says nothing against
    # it and is simply not published yet. Offline on W2, an honest window (anchors within 0.2% of the
    # depth) lost 19 cameras after a turn to "no shared points" and was refused
    # whole; only contradictions now refuse a window.
    contradictions = []
    for kid in selection.active_ids:
        pose = candidate.poses.get(kid)
        if pose is None or int(pose["component"]) != new_comp or kid in collapsed:
            failures.append(kid)
            if pose is not None and kid in collapsed:
                contradictions.append(kid)
            continue
        transformed = _aligned_pose(pose, scale, rotation, translation, old_comp)
        if kid in validated and not _pose_agrees(previous.poses[kid], transformed, unit):
            failures.append(kid)
            contradictions.append(kid)
            continue
        aligned[kid] = transformed
    # Previously published cameras are the trusted bridge to the anchors.
    trusted = {kid for kid in aligned if kid in validated}
    point_support = {}
    for kid in trusted | set(selection.anchor_ids):
        if kid not in candidate.poses:
            continue
        for point, error in _camera_point_errors(candidate, kid).items():
            if error <= MAX_REPROJECTION_MEDIAN_PX:
                point_support.setdefault(point, set()).add(kid)
    historical_steps = []
    for left, right in zip(previous.keyframe_ids, previous.keyframe_ids[1:]):
        if left in validated and right in validated:
            a = global_solve._pose_matrices(previous.poses[left])[1]
            b = global_solve._pose_matrices(previous.poses[right])[1]
            historical_steps.append(float(np.linalg.norm(a - b)))
    floor = MIN_STEP_LIMIT_FRACTION * unit
    step_limit = (max(floor, 4.0 * float(np.median(historical_steps[-30:])))
                  if historical_steps else floor)
    for kid in selection.active_ids:
        if kid not in aligned or kid in validated:
            continue
        index = ids.index(kid)
        predecessor = next((ids[j] for j in range(index - 1, -1, -1)
                            if ids[j] in trusted), None)
        errors = _camera_point_errors(candidate, kid)
        supported = [error for point, error in errors.items()
                     if len(point_support.get(point, ())) >= 2]
        plausible = False
        if predecessor is not None:
            centre = global_solve._pose_matrices(aligned[kid])[1]
            prior = global_solve._pose_matrices(aligned[predecessor])[1]
            plausible = float(np.linalg.norm(centre - prior)) <= step_limit * (index - ids.index(predecessor))
        contradicted = ((predecessor is not None and not plausible) or
                        (len(supported) >= MIN_NEW_CAMERA_POINTS and
                         (float(np.median(supported)) > MAX_REPROJECTION_MEDIAN_PX or
                          float(np.percentile(supported, 95)) > MAX_REPROJECTION_P95_PX)))
        if contradicted or predecessor is None or len(supported) < MIN_NEW_CAMERA_POINTS:
            del aligned[kid]
            failures.append(kid)
            if contradicted:
                contradictions.append(kid)
            continue
        trusted.add(kid)
        for point, error in errors.items():
            if error <= MAX_REPROJECTION_MEDIAN_PX:
                point_support.setdefault(point, set()).add(kid)
    # Of the cameras the evidence could judge, at most a quarter may contradict it.
    if len(contradictions) > MAX_ACTIVE_FAILURE_FRACTION * (len(contradictions) + len(aligned)):
        raise WindowRefused("too many active cameras failed validation")
    return aligned, failures, contradictions


def stitch(previous: global_solve.Solution, candidate: global_solve.Solution,
           ids: list[str], selection: WindowSelection) -> global_solve.Solution:
    """Align validated poses and points; retain the earlier published prefix."""
    if (previous.keyframe_ids != ids[:len(previous.keyframe_ids)] or
            candidate.keyframe_ids != selection.solve_ids or
            len(selection.anchor_ids) < 3):
        raise WindowRefused("anchor or prefix identity changed during window solve")
    old_comp, new_comp, scale, rotation, translation, residual, used_anchors, unit = _alignment(
        previous, candidate, selection.anchor_ids)
    cumulative = float(((previous.solve or {}).get("live_window") or {}).get("cumulative_scale", 1.0)) * scale
    validated = _validated_published(previous)
    aligned, failures, contradictions = _validated_active(previous, candidate, ids, selection,
                                          new_comp, scale, rotation, translation, old_comp,
                                          unit)
    active = set(aligned)
    poses = dict(previous.poses)
    poses.update(aligned)
    if not any(kid not in validated and kid in poses for kid in active):
        raise WindowRefused("window solve added no anchored poses")

    active_observations = {}
    for camera_index, _feature, point_index in candidate.observations:
        if selection.solve_ids[int(camera_index)] in active:
            active_observations.setdefault(int(point_index), set()).add(int(camera_index))
    new_keep = np.array([int(c) == new_comp and
                         len(active_observations.get(p, ())) >= 2
                         for p, c in enumerate(candidate.component)], bool)
    valid_candidate = active | (set(used_anchors) - set(failures))
    # Feature indices in shared keyframes identify points across the two
    # solutions. Replace a prior point only when at least two observations
    # agree on the same candidate point and their aligned 3-D positions agree.
    old_features = {}
    for camera_index, feature, point in previous.observations:
        point = int(point)
        if int(previous.component[point]) == old_comp:
            old_features[(previous.keyframe_ids[int(camera_index)], int(feature))] = point
    matches = {}
    for camera_index, feature, point in candidate.observations:
        kid = selection.solve_ids[int(camera_index)]
        point = int(point)
        if kid not in valid_candidate or not new_keep[point]:
            continue
        old_point = old_features.get((kid, int(feature)))
        if old_point is not None:
            key = (old_point, point)
            matches[key] = matches.get(key, 0) + 1
    old_keep = np.ones(len(previous.xyz), bool)
    replaced_old = {}
    old_claims, new_claims = {}, {}
    for (old_point, new_point), votes in matches.items():
        if votes >= 2:
            old_claims[old_point] = old_claims.get(old_point, 0) + 1
            new_claims[new_point] = new_claims.get(new_point, 0) + 1
    for (old_point, new_point), votes in matches.items():
        if (votes < 2 or old_claims[old_point] != 1 or
                new_claims[new_point] != 1):
            continue
        aligned_xyz = scale * rotation @ candidate.xyz[new_point] + translation
        if np.linalg.norm(previous.xyz[old_point] - aligned_xyz) <= MAX_POSE_RESIDUAL_FRACTION * unit:
            old_keep[old_point] = False
            replaced_old[old_point] = new_point
    old_indices = np.flatnonzero(old_keep)
    new_indices = np.flatnonzero(new_keep)
    old_map = {int(p): j for j, p in enumerate(old_indices)}
    new_map = {int(p): len(old_indices) + j for j, p in enumerate(new_indices)}
    old_map.update({old: new_map[new] for old, new in replaced_old.items()})
    id_index = {kid: i for i, kid in enumerate(ids)}
    new_owner = {}
    for camera_index, _feature, point_index in candidate.observations:
        kid = selection.solve_ids[int(camera_index)]
        point = int(point_index)
        if point in new_map and kid in valid_candidate:
            new_owner[point] = min(new_owner.get(point, len(ids)), id_index[kid])
    first = np.concatenate((previous.first_keyframe[old_keep],
                            np.array([new_owner[p]
                                      for p in new_indices], np.int32)))
    xyz = np.concatenate((previous.xyz[old_keep],
                          (scale * (rotation @ candidate.xyz[new_keep].T).T + translation).astype(np.float32)))
    component = np.concatenate((previous.component[old_keep],
                                np.full(len(new_indices), old_comp, np.int32)))
    observations = []
    xy = []
    for source, point_map, source_ids, discard_active in (
            (previous, old_map, previous.keyframe_ids, True),
            (candidate, new_map, candidate.keyframe_ids, False)):
        for row, pixel in zip(source.observations, source.observation_xy):
            kf, feature, point = (int(v) for v in row)
            kid = source_ids[kf]
            if (point in point_map and
                    ((kid not in active and
                      (point not in replaced_old or kid not in valid_candidate))
                     if discard_active else kid in valid_candidate)):
                observations.append((id_index[kid], feature, point_map[point]))
                xy.append(pixel)
    observation_rows = np.asarray(observations, np.int32).reshape(-1, 3)
    track_length = np.bincount(observation_rows[:, 2], minlength=len(xyz)).astype(np.int32)
    counts = {c: sum(int(p["component"]) == c for p in poses.values())
              for c in set(int(p["component"]) for p in poses.values())}
    components = [dict(c, images=counts.get(int(c["index"]), 0),
                       images_supported=sum(int(p["component"]) == int(c["index"]) and
                                            int(p["observations"]) >= MIN_ANCHOR_OBSERVATIONS
                                            for p in poses.values()),
                       points=int(np.count_nonzero(component == int(c["index"]))))
                  for c in previous.components]
    info = {"active_keyframes": len(selection.active_ids),
            "anchor_ids": selection.anchor_ids, "aligned_anchor_ids": used_anchors,
            "anchor_max_residual": residual, "length_unit": unit,
            "anchor_residual_fraction": residual / unit if unit > 0 else None,
            "scale": scale, "cumulative_scale": cumulative,
            "failed_active_ids": failures,
            "contradicted_active_ids": contradictions,
            "reused_prefix_keyframes": len(previous.keyframe_ids)}
    return dataclasses.replace(
        previous, solver=candidate.solver, solved_at=candidate.solved_at,
        input_digest=compute_input_digest([_Id(kid) for kid in ids]),
        keyframe_ids=list(ids), poses=poses, components=components,
        xyz=xyz, rgb=global_solve.withheld_rgb(len(xyz)), component=component,
        first_keyframe=first,
        track_length=track_length,
        error=np.concatenate((previous.error[old_keep], candidate.error[new_keep])),
        observations=observation_rows,
        observation_xy=np.asarray(xy, np.float32).reshape(-1, 2),
        camera=candidate.camera, timing=candidate.timing, transients=candidate.transients,
        solve={**(candidate.solve or {}), "live_window": info}, gate=None,
    )


@dataclasses.dataclass(frozen=True)
class _Id:
    keyframe_id: str


class _WindowStore:
    def __init__(self, base, world_id, session_id, keyframes):
        self.base = base
        self.world_id = world_id
        self.session_id = session_id
        self.keyframes = keyframes

    def world_dir(self, world_id):
        return self.base.world_dir(world_id) / "live-window"

    def read_keyframes(self, world_id, session_id):
        return self.keyframes

    def read_session(self, world_id, session_id):
        return self.base.read_session(world_id, session_id)

    def session_dir(self, world_id, session_id):
        return self.base.session_dir(world_id, session_id)


def _commit_window_metadata(workspace, metadata):
    global_solve.write_json_atomic(workspace.solution_path, metadata)


def publish_window_solution(workspace, solution):
    """Commit metadata last; it names an immutable arrays generation.

    The ordinary full solve still writes the historical pair. A crashed
    window writer can leave an unreferenced archive, never a mixed pair.
    """
    generation = uuid.uuid4().hex
    stage = global_solve.SolveWorkspace(workspace.root / f".live-stage-{generation}")
    arrays = workspace.root / f"solution-live-{generation}.npz"
    try:
        global_solve.write_solution(stage, solution)
        metadata = global_solve.read_json_closed(stage.solution_path)
        os.replace(stage.arrays_path, arrays)
        metadata["arrays_file"] = arrays.name
        _commit_window_metadata(workspace, metadata)
    finally:
        for path in (stage.arrays_path, stage.solution_path):
            path.unlink(missing_ok=True)
        if stage.root.exists():
            stage.root.rmdir()


REFUSALS_FILE = "refusals.json"
REBOOTSTRAP_AFTER_REFUSALS = 3


def _refusals_against(path, previous) -> int:
    """Consecutive window refusals against THIS published solution (its solved_at)."""
    if previous is None:
        return 0
    try:
        state = global_solve.read_json_closed(path)
    except Exception:  # noqa: BLE001 -- absent or unreadable: no refusals recorded
        return 0
    if state.get("solved_at") != previous.solved_at:
        return 0
    count = state.get("count")
    return count if isinstance(count, int) and count > 0 else 0


def _count_refusal(path, previous) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        global_solve.write_json_atomic(path, {
            "solved_at": previous.solved_at,
            "count": _refusals_against(path, previous) + 1})
    except OSError:
        pass  # a lost count delays a re-bootstrap; it never publishes anything


def _clear_refusals(path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def solve(store, world_id, session_id, *, capture_dirs=(), final=False,
          num_threads=None, min_image_observations=global_solve.MIN_IMAGE_OBSERVATIONS,
          loop_detection=True, input_digest=None):
    """Run only the in-walk bounded solve; publish after anchor validation."""
    if final:
        raise ValueError("the final solve must use global_solve.solve directly")
    started = time.perf_counter()
    keyframes = store.read_keyframes(world_id, session_id)
    ids = [kf.keyframe_id for kf in keyframes]
    previous = global_solve.load_solution(store, world_id, session_id)
    state_path = global_solve.workspace_for(
        _WindowStore(store, world_id, session_id, []), world_id, session_id).root / REFUSALS_FILE
    refusals = _refusals_against(state_path, previous)
    if previous is None or refusals >= REBOOTSTRAP_AFTER_REFUSALS:
        # The bootstrap, or a RE-bootstrap: the ordinary full-prefix background solve.
        # Windows refused this many times in a row against one published solution
        # say that solution, not the windows, is what is wrong (W2: a collapsed
        # 21-keyframe bootstrap refused all 8). It replaces the reference wholesale,
        # exactly as every background solve does with the switch off.
        summary = global_solve.solve(store, world_id, session_id,
                                     capture_dirs=capture_dirs, final=False,
                                     num_threads=num_threads,
                                     min_image_observations=min_image_observations,
                                     loop_detection=loop_detection, input_digest=input_digest)
        _clear_refusals(state_path)
        if previous is not None:
            summary = {**summary, "rebootstrap": {"after_refusals": refusals}}
        return summary
    try:
        selection = select_window(ids, previous)
    except WindowRefused as exc:
        _count_refusal(state_path, previous)
        return {"solved": False, "reason": str(exc), "window_refused": True}
    # What this window was, on every outcome, for the replay's solve.log: W1 had no
    # way to tell a lagged window, its anchors or its wall time from the summary.
    window = {"keyframes": len(ids),
              "active_first": ids.index(selection.active_ids[0]),
              "active_last": ids.index(selection.active_ids[-1]),
              "lagged": selection.active_ids[-1] != ids[-1],
              "anchors": len(selection.anchor_ids),
              "solve_images": len(selection.solve_ids),
              "started_at": round(time.time() - (time.perf_counter() - started), 3)}

    def _done(result):
        if result.get("solved"):
            _clear_refusals(state_path)
        else:
            _count_refusal(state_path, previous)
        return {**result, "window": {**window, "ended_at": round(time.time(), 3),
                                     "total_s": round(time.perf_counter() - started, 3)}}
    selected = set(selection.solve_ids)
    window_store = _WindowStore(store, world_id, session_id,
                                [kf for kf in keyframes if kf.keyframe_id in selected])
    window_workspace = global_solve.workspace_for(window_store, world_id, session_id)
    canonical = global_solve.workspace_for(store, world_id, session_id)
    wanted_images = {global_solve.keyframe_image_name(kf) for kf in window_store.keyframes}
    try:
        db_timing = prepare_window_database(window_workspace, canonical, wanted_images)
    except WindowRefused as exc:
        return _done({"solved": False, "reason": str(exc), "window_refused": True})
    if window_workspace.images_dir.exists():
        for image in window_workspace.images_dir.iterdir():
            if image.is_file() and image.name not in wanted_images:
                try:
                    image.unlink()
                except OSError as exc:
                    return _done({"solved": False, "reason": f"old window image could not be removed: {exc}",
                            "window_refused": True})
    # Preserve the builder's exact raw-frame source mapping in the isolated workspace.
    source_path = canonical.root / global_solve.SOURCES_FILENAME
    if source_path.exists():
        sources = global_solve.read_json_closed(source_path).get("sources", {})
        global_solve.write_sources(window_store, world_id, session_id,
                                   {kid: sources[kid] for kid in selection.solve_ids if kid in sources})
    prepared = time.perf_counter()
    summary = global_solve.solve(
        window_store, world_id, session_id, capture_dirs=capture_dirs, final=False,
        num_threads=num_threads, min_image_observations=min_image_observations,
        loop_detection=loop_detection)
    if not summary.get("solved"):
        return _done({**summary, "window_database": db_timing,
                "window_setup_s": round(prepared - started, 3)})
    mapped = time.perf_counter()
    candidate = global_solve.load_solution(window_store, world_id, session_id)
    if candidate is None:
        return _done({"solved": False, "reason": "window solution could not be loaded",
                "window_refused": True})
    try:
        published = stitch(previous, candidate, ids, selection)
    except WindowRefused as exc:
        return _done({"solved": False, "reason": str(exc), "window_refused": True,
                "window_database": db_timing,
                "window_setup_s": round(prepared - started, 3),
                "timing": summary.get("timing")})
    publish_window_solution(canonical, published)
    return _done({**summary, "keyframes": len(ids),
            "window_database": db_timing,
            "timing": {**(summary.get("timing") or {}),
                       "window_setup_s": round(prepared - started, 3),
                       "window_stitch_s": round(time.perf_counter() - mapped, 3)},
            "keyframes_posed": sum(int(p["observations"]) >= min_image_observations
                                   for p in published.poses.values()),
            "components": published.components, "points": len(published.xyz),
            "solve": published.solve, "workspace": str(canonical.root)})
