"""The opt-in in-walk window never weakens the published prefix."""

import hashlib
import ast
import copy
import dataclasses
import json
import sqlite3
import subprocess
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


def test_lagged_window_advances_from_last_published_pose():
    ids = [f"k{i}" for i in range(500)]
    old = _solution(ids[:70])
    first = live_window.select_window(ids, old)
    assert first.active_ids[0] == "k50"
    assert first.active_ids[-1] == "k149"
    assert len(first.solve_ids) <= 106
    # Publishing a full prefix can leave the newest keyframes unposed. The
    # following solve must continue from its posed frontier, not skip to 499.
    old.keyframe_ids = ids[:500]
    old.poses.update({kid: _pose(float(i), float(i % 3))
                      for i, kid in enumerate(ids[70:150], start=70)})
    second = live_window.select_window(ids, old)
    assert second.active_ids[0] == "k130"
    assert second.active_ids[-1] == "k229"


def test_consistent_sim3_can_land_despite_large_arbitrary_mapper_scale():
    ids = [f"k{i}" for i in range(150)]
    old = _solution(ids[:70])
    selection = live_window.select_window(ids, old)
    candidate = _solution(selection.solve_ids, solved_at=101.0)
    for kid in selection.solve_ids:
        i = ids.index(kid)
        candidate.poses[kid] = _pose(float(i) / 0.06, float(i % 3) / 0.06)
    candidate.xyz /= 0.06
    _refresh_observations(candidate)
    stitched = live_window.stitch(old, candidate, ids, selection)
    assert stitched.poses["k149"]["translation"][0] == pytest.approx(-149.0)
    assert stitched.keyframe_ids == ids


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


def test_switch_off_reads_legacy_writer_arrays_identically(tmp_path, monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_LIVE_WINDOW", raising=False)
    store = WorldStore(tmp_path)
    workspace = global_solve.workspace_for(store, "w", "s")
    old = _solution(["k0", "k1", "k2"])
    global_solve.write_solution(workspace, old)
    metadata = json.loads(workspace.solution_path.read_text())
    assert "arrays_file" not in metadata
    loaded = global_solve.load_solution(store, "w", "s")
    assert loaded.keyframe_ids == old.keyframe_ids
    assert loaded.poses == old.poses
    np.testing.assert_array_equal(loaded.rgb, global_solve.withheld_rgb(len(old.xyz)))
    for name in ("xyz", "component", "first_keyframe", "track_length",
                 "error", "observations", "observation_xy"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(old, name))


def test_switch_off_does_not_select_versioned_window_arrays(tmp_path, monkeypatch):
    store = WorldStore(tmp_path)
    workspace = global_solve.workspace_for(store, "w", "s")
    live_window.publish_window_solution(workspace, _solution(["k0", "k1", "k2"]))
    monkeypatch.setenv("TOWER_WORLD_LIVE_WINDOW", "on")
    assert global_solve.load_solution(store, "w", "s") is not None
    monkeypatch.delenv("TOWER_WORLD_LIVE_WINDOW", raising=False)
    assert global_solve.load_solution(store, "w", "s") is None


def test_final_writer_bytes_match_6978a93_on_same_solution(tmp_path):
    baseline = subprocess.run(
        ["git", "show", "6978a93:tower/tower/world_builder/global_solve.py"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
        encoding="utf-8", check=True,
    ).stdout
    writer_node = next(node for node in ast.parse(baseline).body
                       if isinstance(node, ast.FunctionDef) and node.name == "write_solution")
    namespace = dict(vars(global_solve))
    exec(compile(ast.Module(body=[writer_node], type_ignores=[]),
                 "6978a93:global_solve.py", "exec"), namespace)
    old_writer = namespace["write_solution"]
    solution = _solution(["k0", "k1", "k2"], solved_at=101.0)
    old = global_solve.SolveWorkspace(tmp_path / "old")
    new = global_solve.SolveWorkspace(tmp_path / "new")
    old_writer(old, solution)
    global_solve.write_solution(new, solution)
    assert old.solution_path.read_bytes() == new.solution_path.read_bytes()
    assert old.arrays_path.read_bytes() == new.arrays_path.read_bytes()


def test_run_window_limits_mapper_inputs_and_resets_matching_database(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_LIVE_WINDOW", "on")
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
    assert result["timing"]["window_setup_s"] >= 0
    assert result["timing"]["window_stitch_s"] >= 0
    record = result["window"]
    assert record["keyframes"] == 180 and record["lagged"] is False
    assert record["active_last"] == 179
    assert record["solve_images"] == len(selected.solve_ids)
    assert record["ended_at"] >= record["started_at"] and record["total_s"] >= 0
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
    db = live_window.prepare_window_database(window, canonical,
                                             {"anchor.jpg", "recent.jpg"})
    assert db["removed_images"] == 1
    assert db["kept_images"] == 2
    assert db["copy_s"] >= 0
    assert db["prune_s"] >= 0
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


def test_refused_window_does_not_launch_unbounded_full_prefix(tmp_path, monkeypatch):
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
    def refused(*_args, **_kw):
        raise live_window.WindowRefused("anchor residual is too high")
    monkeypatch.setattr(live_window, "stitch", refused)
    called = []
    monkeypatch.setattr(global_solve, "solve", lambda target, *_a, **_kw:
                        called.append(target) or {"solved": True, "solver": "synthetic"})
    result = live_window.solve(store, "w", "s")
    assert result["solved"] is False
    assert result["window_refused"] is True
    assert len(called) == 1 and isinstance(called[0], live_window._WindowStore)
    assert global_solve.load_solution(store, "w", "s").keyframe_ids == old.keyframe_ids


def test_anchor_validated_scale_is_recorded_even_when_cumulative_value_is_large():
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
    stitched = live_window.stitch(old, new, ids, selection)
    assert stitched.solve["live_window"]["cumulative_scale"] == pytest.approx(1.364)


class _Keyframe:
    def __init__(self, kid):
        self.keyframe_id = kid
        self.image_relpath = f"images/{kid}.jpg"


# ---- FIX2 (walk-8 replay W1, 2026-10-12) ---------------------------------------------

BASELINE = "6978a93"


def _baseline_source(relpath):
    return subprocess.run(
        ["git", "show", f"{BASELINE}:tower/{relpath}"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
        encoding="utf-8", check=True,
    ).stdout


def _baseline_function(relpath, name, module):
    """`name` exactly as it was at 6978a93, run against `module`'s namespace."""
    node = next(n for n in ast.parse(_baseline_source(relpath)).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"{BASELINE}:{relpath}", "exec"),
         namespace)
    return namespace[name]


def _outcome(fn, **kwargs):
    try:
        return ("ok", fn(**kwargs))
    except Exception as exc:  # noqa: BLE001 -- the comparison is of the outcome
        return (type(exc).__name__, str(exc))


def test_w1_jump_becomes_a_lagged_window_next_to_the_published_frontier():
    # W1 solve 2: 436 accepted, a published solution of 70. f245dad mapped 336-435
    # against anchors from 0-69 and was refused ("anchor missing or disconnected").
    ids = [f"k{i}" for i in range(436)]
    old = _solution(ids[:70])
    selection = live_window.select_window(ids, old)
    index = [ids.index(k) for k in selection.active_ids]
    start = 69 - live_window.OVERLAP_KEYFRAMES + 1
    assert index == list(range(start, start + live_window.WINDOW_KEYFRAMES))
    anchors = [ids.index(k) for k in selection.anchor_ids]
    assert all(start - live_window.ANCHOR_LOOKBACK <= a <= 69 for a in anchors)
    assert len(selection.anchor_ids) == live_window.MAX_ANCHORS


def test_frontier_is_the_last_validated_pose_not_any_pose():
    ids = [f"k{i}" for i in range(400)]
    old = _solution(ids[:200])
    for i, kid in enumerate(ids[60:200], start=60):
        old.poses[kid] = _pose(float(i), observations=5)
    selection = live_window.select_window(ids, old)
    assert ids.index(selection.active_ids[0]) == 59 - live_window.OVERLAP_KEYFRAMES + 1
    assert all(int(old.poses[k]["observations"]) >= live_window.MIN_ANCHOR_OBSERVATIONS
               for k in selection.anchor_ids)


def test_window_is_bounded_for_any_backlog_and_any_horizon():
    for horizon, published in ((150, 70), (5000, 70), (5000, 4990), (20000, 9000)):
        ids = [f"k{i}" for i in range(horizon)]
        selection = live_window.select_window(ids, _solution(ids[:published]))
        assert len(selection.active_ids) <= live_window.WINDOW_KEYFRAMES
        assert len(selection.solve_ids) <= live_window.WINDOW_KEYFRAMES + live_window.MAX_ANCHORS


def _mapped(ids, selection, *, scale=1.0, drop=()):
    """The window as the mapper returned it, without the anchors in `drop`."""
    mapped = dataclasses.replace(
        selection, solve_ids=[k for k in selection.solve_ids if k not in drop])
    candidate = _solution(mapped.solve_ids, solved_at=101.0)
    for kid in mapped.solve_ids:
        i = ids.index(kid)
        candidate.poses[kid] = _pose(float(i) / scale, float(i % 3) / scale)
    candidate.xyz /= scale
    _refresh_observations(candidate)
    return mapped, candidate


def test_an_unregistered_anchor_is_left_out_of_the_fit_not_fatal():
    ids = [f"k{i}" for i in range(150)]
    old = _solution(ids[:70])
    selection = live_window.select_window(ids, old)
    lost = [k for k in selection.anchor_ids if k not in selection.active_ids][:1]
    assert lost
    mapped, candidate = _mapped(ids, selection, scale=0.06, drop=lost)
    stitched = live_window.stitch(old, candidate, ids, mapped)
    info = stitched.solve["live_window"]
    assert set(info["aligned_anchor_ids"]) == set(selection.anchor_ids) - set(lost)
    assert stitched.poses["k149"]["translation"][0] == pytest.approx(-149.0)


def test_too_few_registered_anchors_still_refuses():
    ids = [f"k{i}" for i in range(150)]
    old = _solution(ids[:70])
    selection = live_window.select_window(ids, old)
    lost = selection.anchor_ids[:len(selection.anchor_ids) - live_window.MIN_ALIGNED_ANCHORS + 1]
    mapped, candidate = _mapped(ids, selection, drop=lost)
    with pytest.raises(live_window.WindowRefused, match="anchor missing or disconnected"):
        live_window.stitch(old, candidate, ids, mapped)


def test_scale_gauge_is_free_but_a_degenerate_scale_is_refused():
    ids = [f"k{i}" for i in range(150)]
    old = _solution(ids[:70])
    selection = live_window.select_window(ids, old)
    # W1 solve 5 aligned at 0.0601 with a 0.0012 anchor residual and was refused
    # by the 0.67-1.5 band; the gauge of a fresh GLOMAP map is arbitrary.
    for scale in (0.0601, 16.0):
        _, candidate = _mapped(ids, selection, scale=scale)
        stitched = live_window.stitch(old, candidate, ids, selection)
        assert stitched.solve["live_window"]["scale"] == pytest.approx(scale)
    _, candidate = _mapped(ids, selection, scale=5000.0)
    with pytest.raises(live_window.WindowRefused, match="scale"):
        live_window.stitch(old, candidate, ids, selection)


def _legacy_fixtures(root):
    """Workspaces as the 6978a93 writer leaves them, plus the ways a read can fail."""
    def store_with(name, mutate=None):
        store = WorldStore(root / name)
        workspace = global_solve.workspace_for(store, "w", "s")
        global_solve.write_solution(workspace, _solution(["k0", "k1", "k2"]))
        if mutate is not None:
            mutate(workspace)
        return store

    def wrong_schema(ws):
        meta = json.loads(ws.solution_path.read_text())
        meta["schema_version"] = 999
        ws.solution_path.write_text(json.dumps(meta))
    return {
        "published": store_with("published"),
        "arrays missing": store_with("arrays", lambda ws: ws.arrays_path.unlink()),
        "metadata missing": store_with("meta", lambda ws: ws.solution_path.unlink()),
        "wrong schema": store_with("schema", wrong_schema),
        "torn arrays": store_with("torn", lambda ws: ws.arrays_path.write_bytes(
            ws.arrays_path.read_bytes()[:40])),
        "torn metadata": store_with("tornmeta", lambda ws: ws.solution_path.write_text("{")),
        "nothing": WorldStore(root / "empty"),
    }


def _solution_fields(solution):
    if solution is None:
        return None
    return {name: ((value.tobytes(), value.dtype.str, value.shape)
                   if isinstance(value, np.ndarray) else value)
            for name, value in vars(solution).items()}


@pytest.mark.parametrize("switch", [None, "off", "1"])
def test_switch_off_reader_is_the_6978a93_reader_on_every_legacy_case(tmp_path, monkeypatch,
                                                                       switch):
    if switch is None:
        monkeypatch.delenv("TOWER_WORLD_LIVE_WINDOW", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_LIVE_WINDOW", switch)
    old_reader = _baseline_function("tower/world_builder/global_solve.py", "load_solution",
                                    global_solve)
    warnings = []
    monkeypatch.setattr(global_solve.logger, "warning",
                        lambda message, *args, **_kw: warnings.append(message % args))
    for case, store in _legacy_fixtures(tmp_path).items():
        warnings.clear()
        before = _solution_fields(old_reader(store, "w", "s"))
        old_warnings = list(warnings)
        warnings.clear()
        after = _solution_fields(global_solve.load_solution(store, "w", "s"))
        assert after == before, case
        assert warnings == old_warnings, case


def test_switch_off_coverage_is_the_6978a93_coverage(monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_LIVE_WINDOW", raising=False)
    old = _baseline_function("tower/world_builder/guidance_coverage.py", "compute_coverage",
                             guidance_coverage)

    def cases():
        yield _inputs()
        data = _inputs()
        data["solution"]["solved_at"] += 1.0
        yield data
        data = _inputs()
        data["placements"][0]["input_digest"] = "0" * 64
        yield data
        data = _inputs()
        data["solution"]["solved_at"] = float("nan")
        yield data
        data = _inputs()
        data["computed_at"] = data["solution"]["solved_at"] - 1.0
        yield data
        data = _inputs(((0, 4), (7, 4)))
        data["placements"][1]["reference_segment"] = 99
        yield data
    outcomes = []
    for data in cases():
        new = _outcome(guidance_coverage.compute_coverage, **copy.deepcopy(data))
        assert new == _outcome(old, **copy.deepcopy(data))
        outcomes.append(new[0])
    assert outcomes[0] == "ok" and "ValueError" in outcomes


def test_final_solve_code_is_the_6978a93_code_except_the_gated_reader():
    """Every statement of global_solve.py but `load_solution` is as it was at 6978a93."""
    def units(text):
        return {getattr(node, "name", None) or ast.dump(node): ast.dump(node)
                for node in ast.parse(text).body}
    old = units(_baseline_source("tower/world_builder/global_solve.py"))
    new = units(Path(global_solve.__file__).read_text(encoding="utf-8"))
    assert {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)} == {
        "load_solution"}


def test_final_write_after_a_window_publication_is_byte_identical_on_and_off(tmp_path,
                                                                             monkeypatch):
    """world_solve --final, switch on after a window publication and switch off: same bytes."""
    final = _solution(["k0", "k1", "k2"], solved_at=200.0)
    final.solve = {"final": True}
    old_writer = _baseline_function("tower/world_builder/global_solve.py", "write_solution",
                                    global_solve)

    def fake_final(store, world_id, session_id, **kwargs):
        assert kwargs["final"] is True
        global_solve.write_solution(global_solve.workspace_for(store, world_id, session_id),
                                    final)
        return {"solved": True}

    monkeypatch.setattr(global_solve, "solve", fake_final)
    monkeypatch.setattr(WorldStore, "read_keyframes", lambda *_: [])
    outputs = {}
    for switch in ("on", "off"):
        root = tmp_path / switch
        store = WorldStore(root)
        workspace = global_solve.workspace_for(store, "w", "s")
        global_solve.write_solution(workspace, _solution(["k0", "k1"]))
        if switch == "on":
            monkeypatch.setenv("TOWER_WORLD_LIVE_WINDOW", "on")
            live_window.publish_window_solution(
                workspace, _solution(["k0", "k1", "k2"], solved_at=150.0))
            assert global_solve.load_solution(store, "w", "s").solved_at == 150.0
        else:
            monkeypatch.delenv("TOWER_WORLD_LIVE_WINDOW", raising=False)
        world_solve.main(["--root", str(root), "--world", "w", "--session", "s", "--final"])
        outputs[switch] = (workspace.solution_path.read_bytes(),
                           workspace.arrays_path.read_bytes())
        assert global_solve.load_solution(store, "w", "s").solved_at == 200.0
    reference = global_solve.SolveWorkspace(tmp_path / BASELINE)
    old_writer(reference, final)
    assert outputs["on"] == outputs["off"] == (reference.solution_path.read_bytes(),
                                               reference.arrays_path.read_bytes())
