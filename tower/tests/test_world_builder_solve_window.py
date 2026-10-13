"""The live solve window keeps old geometry and leaves the final path alone."""

import dataclasses
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.world_build_session import BackgroundSolver, BackgroundSurface
from tower.world_builder.guidance_coverage import photo_eligible_for_window
from tower.world_builder.global_solve import (
    Solution, merge, merge_windowed, window_bridge_start, window_keyframes,
    window_mapping_database,
)
from tower.world_builder.records import Keyframe


def _keyframe(i, segment):
    return Keyframe(keyframe_id=f"s:{i:08d}", session_id="s", source_seq=i,
                    received_at=float(i), image_relpath=f"images/{i:08d}.jpg",
                    width=360, height=640, byte_count=1, segment_index=segment)


@pytest.fixture
def scene():
    keyframes = [_keyframe(i, i // 2) for i in range(6)]
    poses = {}
    for i in range(4):
        poses[keyframes[i].keyframe_id] = {
            "component": 0, "rotation": np.eye(3).reshape(-1).tolist(),
            "translation": [-float(i), 0., 0.], "observations": 100}
    solution = Solution(
        solver="glomap", solved_at=1., input_digest="old",
        keyframe_ids=[k.keyframe_id for k in keyframes[:4]], poses=poses,
        components=[{"index": 0, "images": 4, "points": 2}],
        xyz=np.asarray([[0., 0., 10.], [2., 0., 10.]], dtype=np.float32),
        rgb=np.full((2, 3), 138, dtype=np.uint8),
        component=np.zeros(2, dtype=np.int32),
        first_keyframe=np.asarray([0, 2], dtype=np.int32),
        track_length=np.ones(2, dtype=np.int32),
        error=np.zeros(2, dtype=np.float32),
        observations=np.zeros((0, 3), dtype=np.int32))
    local_poses = [{"keyframe_id": k.keyframe_id, "segment_index": k.segment_index,
                    "status": "anchor" if i % 2 == 0 else "unavailable",
                    "degeneracy": "", "rotation": [1., 0., 0., 0.] if i % 2 == 0 else None,
                    "translation": [0., 0., 0.] if i % 2 == 0 else None}
                   for i, k in enumerate(keyframes)]
    return keyframes, None, solution, local_poses, [], []


def _argv(tmp_path, monkeypatch, value, *, final=False):
    if value is None:
        monkeypatch.delenv("TOWER_WORLD_SOLVE_WINDOW", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_SOLVE_WINDOW", value)
    solver = BackgroundSolver(root=tmp_path, world_id="w", session_id="s",
                              every=50, capture_dirs=[])
    return solver._argv(final=final)


def test_off_and_invalid_are_the_exact_old_argv(tmp_path, monkeypatch):
    off = _argv(tmp_path, monkeypatch, None)
    for value in ("", "off", "0", "-1", "1.5", " on "):
        assert _argv(tmp_path, monkeypatch, value) == off


def test_only_non_final_argv_receives_the_window(tmp_path, monkeypatch):
    off_final = _argv(tmp_path, monkeypatch, None, final=True)
    live = _argv(tmp_path, monkeypatch, "180")
    assert live[-2:] == ["--solve-window", "180"]
    assert _argv(tmp_path, monkeypatch, "180", final=True) == off_final


def test_partial_window_does_not_replace_a_full_live_surface(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_SOLVE_WINDOW", "180")
    path = tmp_path / "worlds" / "w" / "solve" / "s" / "solution.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"solve": {"window_applied": True}}))
    surface = BackgroundSurface(root=tmp_path, world_id="w", session_id="s")
    assert surface.solve_landed(None) is False
    assert surface._stale is False


def test_photo_worker_gets_only_current_single_component_source_poses():
    eligible = [("old", 0), ("a", 0), ("b", 0), ("c", 0)]
    solution = {"solve": {"window_applied": True}, "poses": {
        "a": {"component": 1}, "b": {"component": 1}, "c": {"component": 2}}}
    assert photo_eligible_for_window(eligible, solution) == [("a", 0), ("b", 0)]
    solution["solve"]["window_applied"] = False
    assert photo_eligible_for_window(eligible, solution) is eligible


def test_window_keeps_its_boundary_segment_whole():
    keyframes = [_keyframe(i, i // 3) for i in range(9)]
    assert window_keyframes(keyframes, 0) is keyframes
    assert [k.keyframe_id for k in window_keyframes(keyframes, 4)] == [
        k.keyframe_id for k in keyframes[3:]]


def test_window_bridges_to_the_last_placed_segment_when_new_frames_exceed_n():
    keyframes = [_keyframe(i, i // 10) for i in range(40)]
    previous = SimpleNamespace(poses={k.keyframe_id: {"observations": 40}
                                      for k in keyframes[:13]})
    placements = [SimpleNamespace(segment_index=1, state="registered")]
    start = window_bridge_start(keyframes, previous, placements)
    assert start == 10
    assert window_keyframes(keyframes, 18, start) == keyframes[10:]


def test_mapping_copy_prunes_old_images_and_pairs_without_touching_final_database(tmp_path):
    source, target = tmp_path / "database.db", tmp_path / "window.db"
    with sqlite3.connect(source) as db:
        db.execute("create table images (image_id integer primary key, name text)")
        db.execute("create table keypoints (image_id integer, rows integer)")
        db.execute("create table descriptors (image_id integer, rows integer)")
        db.execute("create table matches (pair_id integer primary key, rows integer)")
        db.execute("create table two_view_geometries (pair_id integer primary key, rows integer)")
        db.executemany("insert into images values (?, ?)", [(1, "a"), (2, "b"), (3, "c")])
        for table in ("keypoints", "descriptors"):
            db.executemany(f"insert into {table} values (?, 1)", [(1,), (2,), (3,)])
        for table in ("matches", "two_view_geometries"):
            db.executemany(f"insert into {table} values (?, 1)",
                           [(1 * 2147483647 + 2,), (2 * 2147483647 + 3,)])
    window_mapping_database(source, target, ["b", "c"])
    with sqlite3.connect(source) as db:
        assert db.execute("select count(*) from images").fetchone()[0] == 3
        assert db.execute("select count(*) from matches").fetchone()[0] == 2
    with sqlite3.connect(target) as db:
        assert db.execute("select name from images order by image_id").fetchall() == [("b",), ("c",)]
        for table in ("keypoints", "descriptors"):
            assert db.execute(f"select image_id from {table} order by image_id").fetchall() == [(2,), (3,)]
        for table in ("matches", "two_view_geometries"):
            assert db.execute(f"select pair_id from {table}").fetchall() == [(2 * 2147483647 + 3,)]


def _late_solution(scene):
    keyframes, _, solution, *_ = scene
    ids = [k.keyframe_id for k in keyframes[2:6]]
    late_poses = {kid: row for kid, row in solution.poses.items() if kid in ids}
    for i in (4, 5):
        late_poses[keyframes[i].keyframe_id] = {
            "component": 0, "rotation": np.eye(3).reshape(-1).tolist(),
            "translation": [-float(i), 0., 0.], "observations": 100}
    return dataclasses.replace(
        solution, keyframe_ids=ids,
        poses=late_poses,
        xyz=np.asarray([[2., 0., 10.], [4., 0., 10.]], dtype=np.float32),
        rgb=np.full((2, 3), 138, dtype=np.uint8),
        component=np.zeros(2, dtype=np.int32),
        first_keyframe=np.asarray([0, 2], dtype=np.int32),
        track_length=np.ones(2, dtype=np.int32),
        error=np.zeros(2, dtype=np.float32),
        observations=np.zeros((0, 3), dtype=np.int32),
        observation_xy=np.zeros((0, 2), dtype=np.float32),
        solve={"window_keyframes": 4},
    )


def test_window_merge_keeps_old_segment_and_registers_new_one_in_same_frame(scene):
    keyframes, _, solution, poses, points, support = scene
    old = merge(keyframes, poses, points, support, solution, input_digest="old")
    new = merge_windowed(
        keyframes, poses, points, support, _late_solution(scene), input_digest="new",
        previous={"poses": old.pose_rows, "points": old.point_rows,
                  "support": old.support_rows},
        previous_placements=old.placements,
        previous_summary={"segments": {str(k): v for k, v in old.segments.items()}},
    )
    assert [r for r in new.pose_rows if r["segment_index"] == 0] == [
        r for r in old.pose_rows if r["segment_index"] == 0]
    assert [r for r in new.point_rows if r["segment_index"] == 0] == [
        r for r in old.point_rows if r["segment_index"] == 0]
    placements = {p.segment_index: p for p in new.placements}
    assert placements[0].translation == old.placements[0].translation
    assert placements[0].input_digest == "new"
    assert placements[1].reference_segment == placements[0].reference_segment == 0
    assert placements[1].frame_revision == placements[0].frame_revision
    assert placements[2].reference_segment == placements[0].reference_segment
    assert new.summary["window_placement_wobble"]["shared_cameras"] == 2
    assert new.summary["window_placement_wobble"]["max_translation"] < 1e-8


def test_insufficient_overlap_does_not_move_an_older_placement(scene):
    keyframes, _, solution, poses, points, support = scene
    old = merge(keyframes, poses, points, support, solution, input_digest="old")
    late = _late_solution(scene)
    late.poses.pop(keyframes[3].keyframe_id)
    new = merge_windowed(
        keyframes, poses, points, support, late, input_digest="new",
        previous={"poses": old.pose_rows, "points": old.point_rows,
                  "support": old.support_rows}, previous_placements=old.placements,
        previous_summary={"segments": {str(k): v for k, v in old.segments.items()}},
    )
    placements = {p.segment_index: p for p in new.placements}
    assert placements[0].translation == old.placements[0].translation
    assert placements[1].translation == old.placements[1].translation
    assert 2 not in placements
    assert [r for r in new.pose_rows if r["segment_index"] == 1] == [
        r for r in old.pose_rows if r["segment_index"] == 1]
