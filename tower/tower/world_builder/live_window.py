"""Opt-in, bounded background mapping with a guarded full-prefix publication.

The final solver never calls this module. Its feature database and mapper workspace
stay separate from the live window's rolling workspace.
"""

from __future__ import annotations

import dataclasses
from contextlib import closing
import math
import sqlite3

import numpy as np

from tower.world_builder import global_solve
from tower.world_builder.store import compute_input_digest


WINDOW_KEYFRAMES = 100
MAX_ANCHORS = 6
ANCHOR_LOOKBACK = 150
MIN_ANCHOR_OBSERVATIONS = global_solve.MIN_IMAGE_OBSERVATIONS
MAX_ANCHOR_POSITION_RESIDUAL = 0.35
MAX_ANCHOR_ROTATION_DEG = 8.0
MIN_SCALE = 0.67
MAX_SCALE = 1.5


class WindowRefused(ValueError):
    """A candidate must not replace the last published solution."""


@dataclasses.dataclass(frozen=True)
class WindowSelection:
    active_ids: list[str]
    anchor_ids: list[str]
    solve_ids: list[str]


def prepare_window_database(window_workspace, canonical_workspace, image_names: set[str]) -> None:
    """Keep incremental overlap matches while removing every out-of-window pair.

    The first window can reuse the ordinary background solve's feature database.
    Later windows prune their own database. COLMAP's sequential matcher sees only
    rows in ``images``, not the ``image_names`` extraction argument.
    """
    database = window_workspace.database_path
    if not database.exists():
        if not canonical_workspace.database_path.exists():
            return
        window_workspace.root.mkdir(parents=True, exist_ok=True)
        try:
            with closing(sqlite3.connect(canonical_workspace.database_path)) as source:
                with closing(sqlite3.connect(database)) as target:
                    source.backup(target)
        except sqlite3.Error as exc:
            raise WindowRefused(f"window feature database could not be copied: {exc}") from exc
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


def select_window(ids: list[str], previous: global_solve.Solution,
                  *, active_size: int = WINDOW_KEYFRAMES) -> WindowSelection:
    """Choose recent frames and six high-observation poses spread in time.

    Anchors are fixed for this solve from the *published* solution. The last
    150 previous frames bridge the preceding and current windows; bins favour
    observations without letting one textured patch supply every anchor.
    """
    if previous.keyframe_ids != ids[:len(previous.keyframe_ids)]:
        raise WindowRefused("previous solution is not the accepted prefix")
    active = ids[-active_size:]
    candidates = [(i, kid, pose) for i, kid in enumerate(previous.keyframe_ids)
                  if i >= max(0, len(previous.keyframe_ids) - ANCHOR_LOOKBACK)
                  for pose in [previous.poses.get(kid)]
                  if pose is not None and
                  int(pose.get("observations") or 0) >= MIN_ANCHOR_OBSERVATIONS]
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


def _alignment(previous, candidate, anchors):
    old_centres, new_centres, old_rotations, new_rotations = [], [], [], []
    if anchors[0] not in previous.poses or anchors[0] not in candidate.poses:
        raise WindowRefused("anchor missing in window solve")
    old_component = int(previous.poses[anchors[0]]["component"])
    new_component = int(candidate.poses[anchors[0]]["component"])
    for kid in anchors:
        old = previous.poses.get(kid)
        new = candidate.poses.get(kid)
        if (old is None or new is None or int(old["component"]) != old_component or
                int(new["component"]) != new_component or
                int(new.get("observations") or 0) < MIN_ANCHOR_OBSERVATIONS):
            raise WindowRefused("anchor missing or disconnected in window solve")
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
    if (np.linalg.matrix_rank(old_zero, tol=1e-5) < 2 or
            np.linalg.matrix_rank(new_zero, tol=1e-5) < 2):
        raise WindowRefused("anchor centres are geometrically degenerate")
    u, singular, vt = np.linalg.svd(old_zero.T @ new_zero / len(anchors))
    sign = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rotation = u @ sign @ vt
    variance = float(np.sum(new_zero * new_zero) / len(anchors))
    scale = float(np.sum(singular * np.diag(sign)) / variance)
    if not math.isfinite(scale) or not MIN_SCALE <= scale <= MAX_SCALE:
        raise WindowRefused("anchor scale is outside the accepted interval")
    translation = old_mean - scale * rotation @ new_mean
    aligned = scale * (rotation @ new_centres.T).T + translation
    position_residual = np.linalg.norm(aligned - old_centres, axis=1)
    span = float(np.max(np.linalg.norm(old_zero, axis=1)))
    limit = min(MAX_ANCHOR_POSITION_RESIDUAL, max(0.05, 0.05 * span))
    if float(np.max(position_residual)) > limit:
        raise WindowRefused("anchor position residual is too high")
    for old_r, new_r in zip(old_rotations, new_rotations):
        delta = old_r.T @ rotation @ new_r
        angle = math.degrees(math.acos(float(np.clip((np.trace(delta)-1) / 2, -1, 1))))
        if angle > MAX_ANCHOR_ROTATION_DEG:
            raise WindowRefused("anchor orientation residual is too high")
    return old_component, new_component, scale, rotation, translation, float(max(position_residual))


def _aligned_pose(pose, scale, rotation, translation, component):
    r_wc, centre = global_solve._pose_matrices(pose)
    centre = scale * rotation @ centre + translation
    r_cw = (rotation @ r_wc).T
    return {**pose, "component": component, "rotation": r_cw.reshape(-1).tolist(),
            "translation": (-r_cw @ centre).tolist()}


def stitch(previous: global_solve.Solution, candidate: global_solve.Solution,
           ids: list[str], selection: WindowSelection) -> global_solve.Solution:
    """Align validated poses and points; retain the earlier published prefix."""
    if (previous.keyframe_ids != ids[:len(previous.keyframe_ids)] or
            candidate.keyframe_ids != selection.solve_ids or
            len(selection.anchor_ids) < 3):
        raise WindowRefused("anchor or prefix identity changed during window solve")
    old_comp, new_comp, scale, rotation, translation, residual = _alignment(
        previous, candidate, selection.anchor_ids)
    active = set(selection.active_ids)
    poses = dict(previous.poses)
    for kid in selection.active_ids:
        pose = candidate.poses.get(kid)
        if pose is not None and int(pose["component"]) == new_comp:
            poses[kid] = _aligned_pose(pose, scale, rotation, translation, old_comp)
    if not any(kid not in previous.poses and kid in poses for kid in active):
        raise WindowRefused("window solve added no anchored poses")

    old_keep = np.array([ids[int(i)] not in active or int(c) != old_comp
                         for i, c in zip(previous.first_keyframe, previous.component)], bool)
    new_keep = np.array([selection.solve_ids[int(i)] in active and int(c) == new_comp
                         for i, c in zip(candidate.first_keyframe, candidate.component)], bool)
    old_indices = np.flatnonzero(old_keep)
    new_indices = np.flatnonzero(new_keep)
    old_map = {int(p): j for j, p in enumerate(old_indices)}
    new_map = {int(p): len(old_indices) + j for j, p in enumerate(new_indices)}
    id_index = {kid: i for i, kid in enumerate(ids)}
    first = np.concatenate((previous.first_keyframe[old_keep],
                            np.array([id_index[selection.solve_ids[int(candidate.first_keyframe[p])]]
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
            if point in point_map and (not discard_active or kid not in active):
                observations.append((id_index[kid], feature, point_map[point]))
                xy.append(pixel)
    counts = {c: sum(int(p["component"]) == c for p in poses.values())
              for c in set(int(p["component"]) for p in poses.values())}
    components = [dict(c, images=counts.get(int(c["index"]), 0),
                       images_supported=sum(int(p["component"]) == int(c["index"]) and
                                            int(p["observations"]) >= MIN_ANCHOR_OBSERVATIONS
                                            for p in poses.values()),
                       points=int(np.count_nonzero(component == int(c["index"]))))
                  for c in previous.components]
    info = {"active_keyframes": len(selection.active_ids),
            "anchor_ids": selection.anchor_ids, "anchor_max_residual": residual,
            "scale": scale, "reused_prefix_keyframes": len(previous.keyframe_ids)}
    return dataclasses.replace(
        previous, solver=candidate.solver, solved_at=candidate.solved_at,
        input_digest=compute_input_digest([_Id(kid) for kid in ids]),
        keyframe_ids=list(ids), poses=poses, components=components,
        xyz=xyz, rgb=global_solve.withheld_rgb(len(xyz)), component=component,
        first_keyframe=first,
        track_length=np.concatenate((previous.track_length[old_keep], candidate.track_length[new_keep])),
        error=np.concatenate((previous.error[old_keep], candidate.error[new_keep])),
        observations=np.asarray(observations, np.int32).reshape(-1, 3),
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


def solve(store, world_id, session_id, *, capture_dirs=(), final=False,
          num_threads=None, min_image_observations=global_solve.MIN_IMAGE_OBSERVATIONS,
          loop_detection=True, input_digest=None):
    """Run only the in-walk bounded solve; publish after anchor validation."""
    if final:
        raise ValueError("the final solve must use global_solve.solve directly")
    keyframes = store.read_keyframes(world_id, session_id)
    ids = [kf.keyframe_id for kf in keyframes]
    previous = global_solve.load_solution(store, world_id, session_id)
    if previous is None:
        if len(keyframes) > WINDOW_KEYFRAMES:
            return {"solved": False, "reason": "no published anchors for bounded window"}
        return global_solve.solve(store, world_id, session_id,
                                  capture_dirs=capture_dirs, final=False,
                                  num_threads=num_threads,
                                  min_image_observations=min_image_observations,
                                  loop_detection=loop_detection, input_digest=input_digest)
    try:
        selection = select_window(ids, previous)
    except WindowRefused as exc:
        return {"solved": False, "reason": str(exc), "window_refused": True}
    selected = set(selection.solve_ids)
    window_store = _WindowStore(store, world_id, session_id,
                                [kf for kf in keyframes if kf.keyframe_id in selected])
    window_workspace = global_solve.workspace_for(window_store, world_id, session_id)
    canonical = global_solve.workspace_for(store, world_id, session_id)
    wanted_images = {global_solve.keyframe_image_name(kf) for kf in window_store.keyframes}
    try:
        prepare_window_database(window_workspace, canonical, wanted_images)
    except WindowRefused as exc:
        return {"solved": False, "reason": str(exc), "window_refused": True}
    if window_workspace.images_dir.exists():
        for image in window_workspace.images_dir.iterdir():
            if image.is_file() and image.name not in wanted_images:
                try:
                    image.unlink()
                except OSError as exc:
                    return {"solved": False, "reason": f"old window image could not be removed: {exc}",
                            "window_refused": True}
    # Preserve the builder's exact raw-frame source mapping in the isolated workspace.
    source_path = canonical.root / global_solve.SOURCES_FILENAME
    if source_path.exists():
        sources = global_solve.read_json_closed(source_path).get("sources", {})
        global_solve.write_sources(window_store, world_id, session_id,
                                   {kid: sources[kid] for kid in selection.solve_ids if kid in sources})
    summary = global_solve.solve(
        window_store, world_id, session_id, capture_dirs=capture_dirs, final=False,
        num_threads=num_threads, min_image_observations=min_image_observations,
        loop_detection=loop_detection)
    if not summary.get("solved"):
        return summary
    candidate = global_solve.load_solution(window_store, world_id, session_id)
    if candidate is None:
        return {"solved": False, "reason": "window solution could not be loaded",
                "window_refused": True}
    try:
        published = stitch(previous, candidate, ids, selection)
    except WindowRefused as exc:
        return {"solved": False, "reason": str(exc), "window_refused": True}
    global_solve.write_solution(canonical, published)
    return {**summary, "keyframes": len(ids),
            "keyframes_posed": sum(int(p["observations"]) >= min_image_observations
                                   for p in published.poses.values()),
            "components": published.components, "points": len(published.xyz),
            "solve": published.solve, "workspace": str(canonical.root)}
