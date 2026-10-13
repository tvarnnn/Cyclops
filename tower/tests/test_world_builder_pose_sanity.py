"""The final publication's far-pose quarantine and its existing component consumers."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np

from tests.test_world_builder_final_scale_guard_seam import _candidate
from tower.world_builder import coherence_publish as CP
from tower.world_builder import final_scale_guard as FSG
from tower.world_builder import global_solve as GS
from tower.world_builder import pose_sanity as PS
from tower.world_builder import surface_render as SR
from tower.world_builder import surface_pipeline as SP
from tower.world_builder.surface import SurfaceParams


def _scene(centres):
    n = len(centres)
    keyframes = [SimpleNamespace(keyframe_id=f"s:{i:04d}", received_at=1000 + i * 0.5,
                                 image_relpath=f"images/{i:08d}.jpg")
                 for i in range(n)]
    camera = SimpleNamespace(to_json_dict=lambda: {"fx": 50, "fy": 50, "cx": 32, "cy": 24,
                                                     "width": 64, "height": 48})
    solution = _candidate(keyframes, camera, centers=[(x, 0, 0) for x in centres], islands=[(0, n)])
    return solution, keyframes


def test_switch_requires_exact_on(monkeypatch):
    for value in (None, "", "off", "ON", "true", "1", "on "):
        if value is None:
            monkeypatch.delenv("TOWER_WORLD_POSE_SANITY", raising=False)
        else:
            monkeypatch.setenv("TOWER_WORLD_POSE_SANITY", value)
        assert PS.enabled() is False
    monkeypatch.setenv("TOWER_WORLD_POSE_SANITY", "on")
    assert PS.enabled() is True


def test_runaway_cameras_leave_room_and_viewer_path():
    solution, keyframes = _scene([float(i % 12) for i in range(60)] + [3000.0, 4000.0, 5000.0])
    published, result, audit = PS.apply(solution, None, keyframes=keyframes,
                                        session_id="s", started_at=1000.0)
    assert audit["state"] == "quarantined"
    assert audit["quarantined"] == 3
    assert audit["capture_spans_s"] == [[30.0, 31.0]]
    assert audit["max_distance"] > 4000
    assert len(SR._room_poses(published, fallback=False)) == 60
    assert all(published.poses[f"s:{i:04d}"]["component"] != 0 for i in range(60, 63))
    assert {p["component"] for p in published.poses.values()} == {0, 1}
    assert result.components["components"][0]["state"] == "placed"
    assert result.components["components"][1]["reason"] == "pose-sanity-outlier"
    assert published.solve["pose_sanity"]["quarantined"] == 3


def test_normal_draw_is_identical_when_enabled():
    solution, keyframes = _scene([float(i % 12) for i in range(63)])
    before = json.dumps(solution.poses, sort_keys=True)
    published, result, audit = PS.apply(solution, None, keyframes=keyframes,
                                        session_id="s", started_at=1000.0)
    assert published is solution
    assert result is None
    assert audit["quarantined"] == 0
    assert json.dumps(solution.poses, sort_keys=True) == before


def test_small_room_cap_publishes_unchanged_with_warning():
    solution, keyframes = _scene([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1000.0])
    before = copy.deepcopy(solution.poses)
    published, result, audit = PS.apply(solution, None, keyframes=keyframes,
                                        session_id="s", started_at=1000.0)
    assert published.poses == before
    assert result is None
    assert audit["state"] == "cap-exceeded"
    assert audit["candidate_count"] == 1
    assert audit["quarantined"] == 0


def test_quarantined_points_and_dense_inputs_exclude_far_observers():
    solution, keyframes = _scene([float(i % 12) for i in range(60)] + [3000.0, 4000.0, 5000.0])
    published, _result, audit = PS.apply(solution, None, keyframes=keyframes,
                                         session_id="s", started_at=1000.0)
    far_indices = {60, 61, 62}
    point_ids = set(int(row[2]) for row in solution.observations if int(row[0]) in far_indices)
    assert point_ids
    assert all(int(published.component[p]) != 0 for p in point_ids)
    assert sum(int(p["component"]) == 0 for p in published.poses.values()) == 60
    assert audit["quarantined"] == 3


def test_publish_seam_quarantines_when_scale_guard_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_POSE_SANITY", "on")
    monkeypatch.delenv("TOWER_WORLD_FINAL_SCALE_GUARD", raising=False)
    solution, keyframes = _scene([float(i % 12) for i in range(60)] + [3000.0, 4000.0, 5000.0])
    store = SimpleNamespace(read_session=lambda _w, _s: SimpleNamespace(started_at=1000.0))
    written = []
    published, record = CP.gate_and_publish(
        store, "w", "s", SimpleNamespace(root=tmp_path), solution, final=True, gate=False,
        database_path=tmp_path / "database.db", keyframes=keyframes,
        write=lambda _workspace, value: written.append(value))
    assert published is written[0]
    assert published.solve["pose_sanity"]["quarantined"] == 3
    assert len(SR._room_poses(published, fallback=False)) == 60
    assert json.loads((tmp_path / CP.COMPONENTS_FILENAME).read_text())["components"][1]["reason"] == PS.REASON
    assert record is None


def test_scale_guard_assesses_pose_sanitized_candidate_first(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_POSE_SANITY", "on")
    solution, keyframes = _scene([float(i % 12) for i in range(60)] + [3000.0, 4000.0, 5000.0])
    store = SimpleNamespace(read_session=lambda _w, _s: SimpleNamespace(started_at=1000.0))
    monkeypatch.setattr(FSG, "read_evidence", lambda *_args, **_kwargs: ({}, {"source": "none"}))
    outcome = FSG.guard_publication(store, "w", "s", solution, None, mode_=FSG.MODE_ON,
                                    database_path=tmp_path / "database.db", keyframes=keyframes)
    assert outcome.decision == FSG.DECISION_PUBLISHED
    assert outcome.solution.solve["pose_sanity"]["quarantined"] == 3
    assert outcome.audit["pose_sanity"]["quarantined"] == 3
    assert len(SR._room_poses(outcome.solution, fallback=False)) == 60
    far_points = {int(row[2]) for row in solution.observations if int(row[0]) in {60, 61, 62}}
    assert all(int(outcome.solution.component[p]) != 0 for p in far_points)


def test_pose_guard_failure_names_itself_in_solve_summary(tmp_path):
    record = {"pose_sanity": {"decision": "withheld", "why": "invalid room pose"}}
    summary = GS._withheld_summary(record, SimpleNamespace(root=tmp_path))
    assert summary["reason"] == "pose sanity failed: invalid room pose"
    assert summary["pose_sanity"] == record["pose_sanity"]
    assert "final_scale_guard" not in summary


def test_pose_failure_does_not_publish_through_scale_shadow(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_POSE_SANITY", "on")
    solution, keyframes = _scene([float(i) for i in range(8)])
    store = SimpleNamespace(read_session=lambda _w, _s: SimpleNamespace(started_at=1000.0))
    def fail(*_args, **_kwargs):
        raise ValueError("invalid pose")
    monkeypatch.setattr(PS, "apply", fail)
    outcome = FSG.guard_publication(store, "w", "s", solution, None, mode_=FSG.MODE_SHADOW,
                                    database_path=tmp_path / "database.db", keyframes=keyframes)
    assert outcome.decision == FSG.DECISION_WITHHELD
    assert outcome.solution is None


def test_pose_quarantine_writes_components_before_scale_shadow_solution(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_POSE_SANITY", "on")
    monkeypatch.setenv("TOWER_WORLD_FINAL_SCALE_GUARD", "shadow")
    solution, keyframes = _scene([float(i % 12) for i in range(60)] + [3000.0, 4000.0, 5000.0])
    store = SimpleNamespace(read_session=lambda _w, _s: SimpleNamespace(started_at=1000.0))
    monkeypatch.setattr(FSG, "read_evidence", lambda *_args, **_kwargs: ({}, {"source": "none"}))
    seen = []
    def write(_workspace, published):
        seen.append((tmp_path / CP.COMPONENTS_FILENAME).exists())
    CP.gate_and_publish(store, "w", "s", SimpleNamespace(root=tmp_path), solution,
                        final=True, gate=False, database_path=tmp_path / "database.db",
                        keyframes=keyframes, write=write)
    assert seen == [True]


def test_reapplying_published_quarantine_does_not_shrink_room_again():
    solution, keyframes = _scene([float(i % 12) for i in range(60)] + [22.0, 1000.0])
    first, result, audit = PS.apply(solution, None, keyframes=keyframes,
                                    session_id="s", started_at=1000.0)
    assert audit["quarantined"] == 1
    again, again_result, again_audit = PS.apply(first, result, keyframes=keyframes,
                                                session_id="s", started_at=1000.0)
    assert again is first
    assert again_result is result
    assert again_audit == audit


def test_surface_rejects_a_cached_depth_fit_from_a_quarantined_camera(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_POSE_SANITY", "on")
    solution, keyframes = _scene([float(i % 12) for i in range(60)] + [3000.0, 4000.0, 5000.0])
    published, _result, _audit = PS.apply(solution, None, keyframes=keyframes,
                                          session_id="s", started_at=1000.0)
    work = tmp_path / "work"
    (work / "depth").mkdir(parents=True)
    for ki in (0, 60):
        np.save(work / "depth" / f"{ki:05d}.npy", np.ones((2, 2), np.float32))
    align = {"camera": published.camera, "records": [
        {"ki": ki, "kid": f"s:{ki:04d}", "ok": True, "a": 1.0, "b": 0.0,
         "held_out_rel": 0.01} for ki in (0, 60)]}
    frames = SP._Frames(align, work, published, SurfaceParams())
    assert [item[0] for item in frames.items] == [0]


def test_appearance_ignores_a_cached_fit_from_an_unplaced_camera(tmp_path, monkeypatch):
    from tests.test_world_builder_appearance import World, _never_redact
    from tower.world_builder import appearance_pipeline as AP

    monkeypatch.setenv("TOWER_WORLD_POSE_SANITY", "on")
    world = World(tmp_path / "appearance")
    world.solution.poses[world.kids[-1]]["component"] = 1
    world._write_solution(world._workspace, world.solution)
    result = world.build(redactor_factory=_never_redact)
    assert result.state == AP.STATE_OK, result.detail
    assert {row["ki"] for row in world.manifest()["keyframes"]} == set(range(7))


def test_dense_fuse_ignores_a_cached_fit_from_an_unplaced_camera(tmp_path, monkeypatch):
    from tests.test_world_builder_appearance import World, W, H
    from tower.world_builder import dense_pipeline as DP

    monkeypatch.setenv("TOWER_WORLD_POSE_SANITY", "on")
    world = World(tmp_path / "dense")
    world.solution.poses[world.kids[-1]]["component"] = 1
    work = world.dense / "work"
    (work / "undist").mkdir(parents=True)
    for ki in range(7):
        np.save(work / "depth" / f"{ki:05d}.npy", world.depth[ki][:H, :W])
        (work / "undist" / f"{ki:05d}.jpg").write_bytes(
            (world.images / f"{ki + 1:08d}.jpg").read_bytes())
    align = json.loads((world.dense / "align.json").read_text())
    fused = DP.run_fuse_stage(world.solution, align, world.dparams, world.dense)
    assert fused["frames_used"] == 7


def test_regate_candidate_can_reapply_without_duplicate_component_labels():
    solution, keyframes = _scene([float(i % 12) for i in range(60)] + [3000.0, 4000.0, 5000.0])
    first, _result, audit = PS.apply(solution, None, keyframes=keyframes,
                                     session_id="s", started_at=1000.0)
    candidate = CP.solver_candidate(first)
    again, result2, audit2 = PS.apply(candidate, None, keyframes=keyframes,
                                      session_id="s", started_at=1000.0)
    labels = [entry["index"] for entry in again.components]
    assert audit2["keyframe_ids"] == audit["keyframe_ids"]
    assert len(labels) == len(set(labels))
    assert len(result2.components["components"]) == 2
