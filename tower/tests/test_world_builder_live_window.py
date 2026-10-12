"""The opt-in in-walk window never weakens the published prefix."""

import hashlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from tower.world_builder import global_solve, guidance_coverage, live_window
from tower.world_builder.store import WorldStore
from scripts import world_solve
sys.path.insert(0, str(Path(__file__).parent))
from test_world_builder_guidance_coverage import _inputs


def _digest(ids):
    return hashlib.sha256(b"".join(k.encode() + b"\0" for k in ids)).hexdigest()


def _pose(x, y=0.0, *, observations=50, component=0):
    return {"component": component, "rotation": np.eye(3).reshape(-1).tolist(),
            "translation": [-x, -y, 0.0], "observations": observations}


def _solution(ids, *, solved_at=100.0, offset=0.0, outlier=False):
    poses = {kid: _pose(float(i) + offset, float(i % 3)) for i, kid in enumerate(ids)}
    if outlier:
        poses[ids[2]] = _pose(1000.0, 1000.0)
    n = len(ids)
    # Eight landmarks jointly observed by every camera make support checkable.
    xyz = np.array([[offset + j - 4.0, 0.0, 10.0] for j in range(8)], np.float32)
    obs = np.array([(i, j, j) for i in range(n) for j in range(8)], np.int32)
    xy = np.array([[100.0 * (xyz[j, 0] - (i + offset)) / 10.0 + 320.0,
                    100.0 * (xyz[j, 1] - (i % 3)) / 10.0 + 240.0]
                   for i in range(n) for j in range(8)], np.float32)
    return global_solve.Solution(
        solver="synthetic", solved_at=solved_at, input_digest=_digest(ids),
        keyframe_ids=list(ids), poses=poses,
        components=[{"index": 0, "images": n, "images_supported": n,
                     "points": n, "mean_error_px": 0.0}],
        xyz=xyz, rgb=np.zeros((8, 3), np.uint8), component=np.zeros(8, np.int32),
        first_keyframe=np.zeros(8, dtype=np.int32),
        track_length=np.full(8, n, np.int32), error=np.zeros(8, np.float32),
        observations=obs, observation_xy=xy,
        camera={"fx": 100.0, "fy": 100.0, "cx": 320.0, "cy": 240.0},
        solve={"final": False},
    )


def _refresh_observations(solution):
    xy = []
    for camera_index, _feature, point_index in solution.observations:
        kid = solution.keyframe_ids[int(camera_index)]
        pose = solution.poses[kid]
        r = np.asarray(pose["rotation"]).reshape(3, 3)
        t = np.asarray(pose["translation"])
        p = r @ solution.xyz[int(point_index)] + t
        xy.append([100.0 * p[0] / p[2] + 320.0,
                   100.0 * p[1] / p[2] + 240.0])
    solution.observation_xy = np.asarray(xy, np.float32).reshape(-1, 2)


def test_window_is_bounded_and_anchors_are_spread():
    ids = [f"k{i}" for i in range(1000)]
    old = _solution(ids[:950])
    selection = live_window.select_window(ids, old)
    assert len(selection.active_ids) == live_window.WINDOW_KEYFRAMES
    assert len(selection.solve_ids) <= live_window.WINDOW_KEYFRAMES + live_window.MAX_ANCHORS
    assert selection.solve_ids == [kid for kid in ids if kid in set(selection.solve_ids)]
    assert len(selection.anchor_ids) >= 3
    assert min(ids.index(k) for k in selection.anchor_ids) < 900
    assert max(ids.index(k) for k in selection.anchor_ids) >= 900


def test_residual_refusal_keeps_last_published_solution(tmp_path):
    ids = [f"k{i}" for i in range(15)]
    old = _solution(ids[:12])
    window = _solution(ids[3:], solved_at=101.0, outlier=True)
    store = WorldStore(tmp_path)
    workspace = global_solve.workspace_for(store, "w", "s")
    global_solve.write_solution(workspace, old)
    before = workspace.solution_path.read_bytes()
    selection = live_window.select_window(ids, old, active_size=12)
    with pytest.raises(live_window.WindowRefused, match="anchor"):
        live_window.stitch(old, window, ids, selection)
    assert workspace.solution_path.read_bytes() == before


def test_stitched_published_solution_passes_coverage_prefix_contract(tmp_path):
    ids = [f"k{i}" for i in range(15)]
    old = _solution(ids[:12])
    selection = live_window.select_window(ids, old, active_size=12)
    window = _solution(selection.solve_ids, solved_at=101.0, offset=3.0)
    # Synthetic camera centres follow global indices even when anchors are inserted.
    for kid in selection.solve_ids:
        i = ids.index(kid)
        window.poses[kid] = _pose(float(i) + 3.0, float(i % 3))
    _refresh_observations(window)
    stitched = live_window.stitch(old, window, ids, selection)
    store = WorldStore(tmp_path)
    workspace = global_solve.workspace_for(store, "w", "s")
    global_solve.write_solution(workspace, stitched)
    loaded = global_solve.load_solution(store, "w", "s")
    assert loaded is not None
    assert loaded.keyframe_ids == ids
    assert loaded.input_digest == _digest(ids)
    assert loaded.poses["k0"] == old.poses["k0"]
    assert loaded.poses["k14"]["translation"][0] == pytest.approx(-14.0)
    data = _inputs(((0, 15),), solved_at=stitched.solved_at)
    data["solution"].update(keyframe_ids=loaded.keyframe_ids,
                            input_digest=loaded.input_digest,
                            poses=loaded.poses)
    data["manifest"]["input_digest"] = _digest(ids)
    data["manifest"]["global_solve"]["horizon_keyframes"] = len(ids)
    data["manifest"]["keyframes"] = len(ids)
    data["keyframes"] = [{"keyframe_id": kid} for kid in ids]
    data["placements"][0]["input_digest"] = _digest(ids)
    data["manifest"]["global_solve"]["solve"] = stitched.solve
    data["solution"]["solve"] = stitched.solve
    assert guidance_coverage.compute_coverage(**data)["horizon_keyframes"] == len(ids)


def test_off_and_final_use_the_original_solver_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(WorldStore, "read_keyframes", lambda *_: [])
    monkeypatch.setattr(global_solve, "solve", lambda *a, **kw: calls.append(("full", kw)) or
                        {"solved": False})
    monkeypatch.setattr(live_window, "solve", lambda *a, **kw: calls.append(("window", kw)) or
                        {"solved": False})
    base = ["--root", str(tmp_path), "--world", "w", "--session", "s"]
    monkeypatch.delenv("TOWER_WORLD_LIVE_WINDOW", raising=False)
    world_solve.main(base)
    off = calls[-1][1]
    assert calls[-1][0] == "full"
    world_solve.main(base + ["--final"])
    final_off = calls[-1][1]
    monkeypatch.setenv("TOWER_WORLD_LIVE_WINDOW", "on")
    world_solve.main(base)
    assert calls[-1][0] == "window"
    assert calls[-1][1] == off
    world_solve.main(base + ["--final"])
    assert calls[-1][0] == "full"
    assert calls[-1][1] == final_off


def test_run_window_limits_mapper_inputs_and_resets_matching_database(tmp_path, monkeypatch):
    ids = [f"k{i}" for i in range(180)]
    store = WorldStore(tmp_path)
    monkeypatch.setattr(store, "read_keyframes", lambda *_: [_Keyframe(kid) for kid in ids])
    canonical = global_solve.workspace_for(store, "w", "s")
    global_solve.write_solution(canonical, _solution(ids[:120]))
    selected = live_window.select_window(ids, global_solve.load_solution(store, "w", "s"))
    window_store = live_window._WindowStore(store, "w", "s", [])
    workspace = global_solve.workspace_for(window_store, "w", "s")
    workspace.root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(workspace.database_path) as con:
        con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
        con.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY)")
        con.execute("CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY)")
        con.execute("CREATE TABLE matches (pair_id INTEGER PRIMARY KEY)")
        con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO images VALUES (1, 'old.jpg')")
    mapped = []

    def map_window(target, _world, _session, **kwargs):
        selected_ids = [kf.keyframe_id for kf in target.read_keyframes("w", "s")]
        assert selected_ids == selected.solve_ids
        assert len(selected_ids) <= live_window.WINDOW_KEYFRAMES + live_window.MAX_ANCHORS
        with sqlite3.connect(global_solve.workspace_for(target, "w", "s").database_path) as con:
            assert con.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0
        candidate = _solution(selected_ids, solved_at=101.0)
        for kid in selected_ids:
            i = ids.index(kid)
            candidate.poses[kid] = _pose(float(i) + 3.0, float(i % 3))
        candidate.xyz[:, 0] += 3.0
        _refresh_observations(candidate)
        global_solve.write_solution(global_solve.workspace_for(target, "w", "s"), candidate)
        mapped.append(selected_ids)
        return {"solved": True, "solver": "synthetic", "timing": {}}

    monkeypatch.setattr(global_solve, "solve", map_window)
    result = live_window.solve(store, "w", "s")
    assert result["solved"] is True
    assert len(mapped) == 1
    assert global_solve.load_solution(store, "w", "s").keyframe_ids == ids


def test_database_pruning_removes_old_pairs_and_preserves_overlap(tmp_path):
    window = global_solve.SolveWorkspace(tmp_path / "window")
    canonical = global_solve.SolveWorkspace(tmp_path / "canonical")
    window.root.mkdir()
    with sqlite3.connect(window.database_path) as con:
        con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
        con.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY)")
        con.execute("CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY)")
        con.execute("CREATE TABLE matches (pair_id INTEGER PRIMARY KEY)")
        con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY)")
        for image_id, name in ((1, "old.jpg"), (2, "anchor.jpg"), (3, "recent.jpg")):
            con.execute("INSERT INTO images VALUES (?, ?)", (image_id, name))
            con.execute("INSERT INTO keypoints VALUES (?)", (image_id,))
            con.execute("INSERT INTO descriptors VALUES (?)", (image_id,))
        for a, b in ((1, 2), (2, 3)):
            pair = a * 2147483647 + b
            con.execute("INSERT INTO matches VALUES (?)", (pair,))
            con.execute("INSERT INTO two_view_geometries VALUES (?)", (pair,))
    live_window.prepare_window_database(window, canonical, {"anchor.jpg", "recent.jpg"})
    with sqlite3.connect(window.database_path) as con:
        assert con.execute("SELECT name FROM images ORDER BY image_id").fetchall() == [
            ("anchor.jpg",), ("recent.jpg",)]
        assert con.execute("SELECT pair_id FROM matches").fetchall() == [(2 * 2147483647 + 3,)]
        assert con.execute("SELECT pair_id FROM two_view_geometries").fetchall() == [
            (2 * 2147483647 + 3,)]
        assert con.execute("SELECT image_id FROM keypoints ORDER BY image_id").fetchall() == [
            (2,), (3,)]


def test_first_background_solve_falls_back_to_full_prefix(tmp_path, monkeypatch):
    store = WorldStore(tmp_path)
    monkeypatch.setattr(store, "read_keyframes", lambda *_: [_Keyframe(f"k{i}") for i in range(500)])
    called = []
    monkeypatch.setattr(global_solve, "solve", lambda target, *_a, **_kw:
                        called.append(len(target.read_keyframes("w", "s"))) or {"solved": True})
    result = live_window.solve(store, "w", "s")
    assert result["solved"] is True
    assert called == [500]


def test_non_anchor_wrong_published_pose_is_not_replaced():
    ids = [f"k{i}" for i in range(15)]
    old = _solution(ids[:12])
    selection = live_window.select_window(ids, old, active_size=12)
    new = _solution(selection.solve_ids, solved_at=101.0, offset=3.0)
    for kid in selection.solve_ids:
        i = ids.index(kid)
        new.poses[kid] = _pose(float(i) + 3.0, float(i % 3))
    _refresh_observations(new)
    new.poses["k10"] = _pose(1000.0, 1000.0)
    stitched = live_window.stitch(old, new, ids, selection)
    assert stitched.poses["k10"] == old.poses["k10"]
    assert "k14" in stitched.poses
    assert np.count_nonzero(stitched.observations[:, 0] == 10) == 8
    assert np.array_equal(stitched.track_length,
                          np.bincount(stitched.observations[:, 2],
                                      minlength=len(stitched.xyz)))


def test_new_camera_needs_anchor_supported_observations_and_plausible_step():
    ids = [f"k{i}" for i in range(15)]
    old = _solution(ids[:12])
    selection = live_window.select_window(ids, old, active_size=12)
    new = _solution(selection.solve_ids, solved_at=101.0, offset=3.0)
    for kid in selection.solve_ids:
        i = ids.index(kid)
        new.poses[kid] = _pose(float(i) + 3.0, float(i % 3))
    _refresh_observations(new)
    # One isolated jump, and one camera whose observations have no support.
    new.poses["k13"] = _pose(1000.0, 0.0)
    index = selection.solve_ids.index("k14")
    new.observations = new.observations[new.observations[:, 0] != index]
    new.observation_xy = np.delete(new.observation_xy,
                                   np.arange(index * 8, (index + 1) * 8), axis=0)
    stitched = live_window.stitch(old, new, ids, selection)
    assert "k13" not in stitched.poses
    assert "k14" not in stitched.poses
    assert not np.any(np.isin(stitched.observations[:, 0], [13, 14]))


def test_too_many_failed_active_cameras_refuses_window():
    ids = [f"k{i}" for i in range(15)]
    old = _solution(ids[:12])
    selection = live_window.select_window(ids, old, active_size=12)
    new = _solution(selection.solve_ids, solved_at=101.0, offset=3.0)
    for kid in selection.solve_ids:
        i = ids.index(kid)
        new.poses[kid] = _pose(float(i) + 3.0, float(i % 3))
    _refresh_observations(new)
    for kid in (kid for kid in selection.active_ids if kid not in selection.anchor_ids):
        new.poses[kid] = _pose(1000.0, 0.0)
    with pytest.raises(live_window.WindowRefused, match="active cameras"):
        live_window.stitch(old, new, ids, selection)


def test_anchor_first_point_seen_by_two_active_cameras_is_retained():
    ids = [f"k{i}" for i in range(15)]
    old = _solution(ids[:12])
    selection = live_window.select_window(ids, old, active_size=12)
    new = _solution(selection.solve_ids, solved_at=101.0, offset=3.0)
    for kid in selection.solve_ids:
        i = ids.index(kid)
        new.poses[kid] = _pose(float(i) + 3.0, float(i % 3))
    _refresh_observations(new)
    new.xyz = np.vstack((new.xyz, [20.0, 0.0, 10.0])).astype(np.float32)
    new.rgb = np.vstack((new.rgb, [0, 0, 0])).astype(np.uint8)
    new.component = np.append(new.component, 0).astype(np.int32)
    new.first_keyframe = np.append(new.first_keyframe, 0).astype(np.int32)
    new.track_length = np.append(new.track_length, 3).astype(np.int32)
    new.error = np.append(new.error, 0).astype(np.float32)
    for kid in (selection.anchor_ids[0], "k12", "k13"):
        j = selection.solve_ids.index(kid)
        new.observations = np.vstack((new.observations, [j, 8, 8])).astype(np.int32)
        new.observation_xy = np.vstack((new.observation_xy,
            [100.0 * (20.0 - (ids.index(kid) + 3.0)) / 10.0 + 320.0,
             100.0 * (0.0 - ids.index(kid) % 3) / 10.0 + 240.0])).astype(np.float32)
    stitched = live_window.stitch(old, new, ids, selection)
    assert len(stitched.xyz) == 9
    extra = np.flatnonzero(np.isclose(stitched.xyz[:, 0], 17.0))
    assert len(extra) == 1
    assert ids[int(stitched.first_keyframe[extra[0]])] == selection.anchor_ids[0]


def test_window_publish_crash_before_metadata_keeps_old_pair(tmp_path, monkeypatch):
    store = WorldStore(tmp_path)
    workspace = global_solve.workspace_for(store, "w", "s")
    old = _solution(["k0", "k1"])
    new = _solution(["k0", "k1", "k2"], solved_at=101.0)
    global_solve.write_solution(workspace, old)
    original_meta = workspace.solution_path.read_bytes()
    def crash(*_args, **_kwargs):
        raise OSError("crash before commit")
    monkeypatch.setattr(live_window, "_commit_window_metadata", crash, raising=False)
    with pytest.raises(OSError, match="crash before commit"):
        live_window.publish_window_solution(workspace, new)
    loaded = global_solve.load_solution(store, "w", "s")
    assert loaded.keyframe_ids == old.keyframe_ids
    assert len(loaded.xyz) == len(old.xyz)
    assert workspace.solution_path.read_bytes() == original_meta


def test_repeated_window_scale_drift_falls_back_to_full_prefix(tmp_path, monkeypatch):
    ids = [f"k{i}" for i in range(15)]
    store = WorldStore(tmp_path)
    monkeypatch.setattr(store, "read_keyframes", lambda *_: [_Keyframe(kid) for kid in ids])
    old = _solution(ids[:12])
    old.solve["live_window"] = {"cumulative_scale": 1.24}
    global_solve.write_solution(global_solve.workspace_for(store, "w", "s"), old)
    selection = live_window.select_window(ids, old)
    candidate = _solution(selection.solve_ids)
    real_load = global_solve.load_solution
    monkeypatch.setattr(global_solve, "load_solution", lambda target, w, s:
                        candidate if isinstance(target, live_window._WindowStore)
                        else real_load(target, w, s))
    def too_large(*_args, **_kw):
        raise live_window.WindowRefused("cumulative scale outside first solve bound")
    monkeypatch.setattr(live_window, "stitch", too_large)
    called = []
    monkeypatch.setattr(global_solve, "solve", lambda target, *_a, **_kw:
                        called.append(target) or {"solved": True, "solver": "synthetic"})
    result = live_window.solve(store, "w", "s")
    assert result["solved"] is True
    assert called[-1] is store


def test_cumulative_scale_is_bounded_against_first_full_solve():
    ids = [f"k{i}" for i in range(15)]
    old = _solution(ids[:12])
    old.solve["live_window"] = {"cumulative_scale": 1.24}
    selection = live_window.select_window(ids, old, active_size=12)
    new = _solution(selection.solve_ids, solved_at=101.0)
    for kid in selection.solve_ids:
        i = ids.index(kid)
        new.poses[kid] = _pose(float(i) / 1.1, float(i % 3) / 1.1)
    new.xyz /= 1.1
    _refresh_observations(new)
    with pytest.raises(live_window.WindowRefused, match="cumulative scale"):
        live_window.stitch(old, new, ids, selection)


class _Keyframe:
    def __init__(self, kid):
        self.keyframe_id = kid
        self.image_relpath = f"images/{kid}.jpg"
