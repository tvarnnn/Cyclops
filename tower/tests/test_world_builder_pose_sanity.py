"""The final publication's far-pose quarantine and its existing component consumers."""

from __future__ import annotations

import copy
import dataclasses
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
    monkeypatch.setenv("TOWER_WORLD_FINAL_SCALE_GUARD", "shadow")
    solution, keyframes = _scene([float(i) for i in range(8)])
    store = SimpleNamespace(read_session=lambda _w, _s: SimpleNamespace(started_at=1000.0))
    def fail(*_args, **_kwargs):
        raise ValueError("invalid pose")
    monkeypatch.setattr(PS, "apply", fail)
    monkeypatch.setattr(FSG, "read_evidence", lambda *_args, **_kwargs: ({}, {"source": "none"}))
    written = []
    published, record = CP.gate_and_publish(
        store, "w", "s", SimpleNamespace(root=tmp_path), solution, final=True, gate=False,
        database_path=tmp_path / "database.db", keyframes=keyframes,
        write=lambda _workspace, value: written.append(value))
    assert published is None
    assert written == []
    assert record["pose_sanity"]["decision"] == "withheld"


def test_shadow_scale_guard_leaves_pose_sanity_to_the_seam(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_POSE_SANITY", "on")
    solution, keyframes = _scene([float(i % 12) for i in range(60)] + [3000.0, 4000.0, 5000.0])
    store = SimpleNamespace(read_session=lambda _w, _s: SimpleNamespace(started_at=1000.0))
    monkeypatch.setattr(FSG, "read_evidence", lambda *_args, **_kwargs: ({}, {"source": "none"}))
    outcome = FSG.guard_publication(store, "w", "s", solution, None, mode_=FSG.MODE_SHADOW,
                                    database_path=tmp_path / "database.db", keyframes=keyframes)
    assert outcome.decision == FSG.DECISION_SHADOW
    assert outcome.solution is solution


def _persisted(root, monkeypatch, guard_mode):
    if guard_mode is None:
        monkeypatch.delenv("TOWER_WORLD_FINAL_SCALE_GUARD", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_FINAL_SCALE_GUARD", guard_mode)
    solution, keyframes = _scene([float(i % 12) for i in range(60)] + [3000.0, 4000.0, 5000.0])
    store = SimpleNamespace(read_session=lambda _w, _s: SimpleNamespace(started_at=1000.0))
    root.mkdir(parents=True)
    CP.gate_and_publish(store, "w", "s", SimpleNamespace(root=root), solution, final=True, gate=False,
                        database_path=root / "database.db", keyframes=keyframes,
                        write=lambda _workspace, value: GS.write_solution(GS.SolveWorkspace(root), value))
    meta = json.loads((root / "solution.json").read_text())
    with np.load(root / "solution.npz") as arrays:
        component = arrays["component"].tolist() if "component" in arrays.files else None
    return meta, component, json.loads((root / CP.COMPONENTS_FILENAME).read_text())


def test_scale_shadow_persists_exactly_what_scale_off_persists_with_pose_sanity_on(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_POSE_SANITY", "on")
    monkeypatch.setattr(FSG, "read_evidence", lambda *_args, **_kwargs: ({}, {"source": "none"}))
    off_meta, off_component, off_doc = _persisted(tmp_path / "off", monkeypatch, None)
    sh_meta, sh_component, sh_doc = _persisted(tmp_path / "shadow", monkeypatch, "shadow")
    assert off_meta["solve"]["pose_sanity"]["quarantined"] == 3
    assert sum(int(p["component"]) == 0 for p in off_meta["poses"].values()) == 60
    assert sh_meta["poses"] == off_meta["poses"]
    assert sh_meta["solve"] == off_meta["solve"]
    assert sh_component == off_component
    assert sh_doc["components"] == off_doc["components"]
    assert "final_scale_guard" not in sh_meta["solve"]


def test_sparse_quarantine_uses_each_cameras_own_label():
    solution, keyframes = _scene([float(i % 12) for i in range(60)] + [3000.0, 4000.0, 5000.0])
    published, _result, audit = PS.apply(solution, None, keyframes=keyframes,
                                         session_id="s", started_at=1000.0)
    # A later relabel (the scale guard's) splits the quarantined cameras across two labels.
    restored = dataclasses.replace(published, component=np.asarray(solution.component).copy(),
                                   poses=copy.deepcopy(published.poses))
    restored.poses["s:0060"]["component"] = 7
    restored.poses["s:0061"]["component"] = 8
    restored.poses["s:0062"]["component"] = 8
    out = PS.enforce_sparse_quarantine(restored, audit["keyframe_ids"])
    seen = {}
    for cam, _feature, point in np.asarray(solution.observations)[:, :3].astype(np.int64):
        seen.setdefault(int(point), set()).add(int(cam))
    only_60 = [p for p, cams in seen.items() if 60 in cams and not cams & {61, 62}]
    only_late = [p for p, cams in seen.items() if cams & {61, 62} and 60 not in cams]
    assert only_60 and only_late
    assert all(int(out.component[p]) == 7 for p in only_60)
    assert all(int(out.component[p]) == 8 for p in only_late)


def test_radius_threshold_survives_contamination_at_the_cap():
    # 6 of 60 (10%, the cap) are far: five runaways and one camera 18.5 from the centre.
    # Untrimmed, the runaways drag the p90 into the gap and 3 x p90 = 20.4 keeps it; the
    # trimmed p90 ignores the runaways (3 x 5.5 = 16.5) and the whole far set goes.
    centres = [float(i % 12) for i in range(54)] + [24.0] + [3000.0 + i for i in range(5)]
    solution, keyframes = _scene(centres)
    published, _result, audit = PS.apply(solution, None, keyframes=keyframes,
                                         session_id="s", started_at=1000.0)
    distances = np.abs(np.asarray(centres) - np.median(centres))
    assert np.percentile(distances, 90) * PS.RADIUS_MULTIPLE > distances[54]  # untrimmed rule misses it
    assert audit["radius_trimmed"] == 5
    assert audit["p90_radius"] == 5.5
    assert audit["state"] == "quarantined"
    assert audit["radius_candidates"] == [f"s:{i:04d}" for i in range(54, 60)]
    assert len(SR._room_poses(published, fallback=False)) == 54


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
    solution, keyframes = _scene([float(i % 12) for i in range(60)] + [14.0, 1000.0])
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


# --- temporal-jump rule (same switch, same cap) ---------------------------------

def _walk(n, step=0.1):
    return [round(i * step, 6) for i in range(n)]


def test_isolated_temporal_spike_is_quarantined():
    centres = _walk(60)
    centres[30] += 5.0  # 50 median steps out and straight back; inside 3 x p90
    solution, keyframes = _scene(centres)
    published, result, audit = PS.apply(solution, None, keyframes=keyframes,
                                        session_id="s", started_at=1000.0)
    assert audit["radius_candidates"] == []
    assert audit["jump_candidates"] == ["s:0030"]
    assert audit["state"] == "quarantined"
    assert audit["keyframe_ids"] == ["s:0030"]
    assert audit["capture_spans_s"] == [[15.0, 15.0]]
    assert published.poses["s:0030"]["component"] != 0
    assert len(SR._room_poses(published, fallback=False)) == 59
    assert result.components["components"][1]["reason"] == PS.REASON


def test_genuine_fast_segment_is_untouched():
    centres, x = [], 0.0
    for i in range(60):
        centres.append(x)
        x += 3.0 if 28 <= i < 32 else 0.1  # four consecutive 30x steps, one direction
    solution, keyframes = _scene(centres)
    published, result, audit = PS.apply(solution, None, keyframes=keyframes,
                                        session_id="s", started_at=1000.0)
    assert audit["jump"]["max_spike_speed"] is None or audit["jump"]["max_spike_speed"] < PS.JUMP_SPEED_MULTIPLE
    assert audit["candidate_count"] == 0
    assert published is solution and result is None


def test_slow_out_and_back_is_untouched():
    # A spike-shaped path walked at plausible speed (long receipt gaps) is not a jump.
    centres = _walk(60)
    centres[30] += 5.0
    solution, keyframes = _scene(centres)
    for i, kf in enumerate(keyframes):
        kf.received_at = 1000 + i * 0.5 + (60.0 if i >= 30 else 0.0) + (60.0 if i >= 31 else 0.0)
    _published, _result, audit = PS.apply(solution, None, keyframes=keyframes,
                                          session_id="s", started_at=1000.0)
    assert audit["jump_candidates"] == []


def test_capture_order_follows_source_seq_over_receipt():
    centres = _walk(60)
    centres[30] += 5.0
    solution, keyframes = _scene(centres)
    for i, kf in enumerate(keyframes):
        kf.source_seq = i
    # Receipts for 29 and 30 arrive swapped; capture order still puts 30 between 29 and 31.
    keyframes[29].received_at, keyframes[30].received_at = keyframes[30].received_at, keyframes[29].received_at
    _published, _result, audit = PS.apply(solution, None, keyframes=keyframes,
                                          session_id="s", started_at=1000.0)
    assert audit["jump"]["order"] == "source_seq"
    assert audit["jump_candidates"] == ["s:0030"]


def test_radius_and_jump_candidates_share_one_cap():
    centres = _walk(30)
    centres[8] += 2.4
    centres[18] += 2.4
    centres += [1000.0, 1001.0]
    solution, keyframes = _scene(centres)
    before = copy.deepcopy(solution.poses)
    published, result, audit = PS.apply(solution, None, keyframes=keyframes,
                                        session_id="s", started_at=1000.0)
    assert len(audit["radius_candidates"]) == 2
    assert audit["jump_candidates"] == ["s:0008", "s:0018"]
    assert len(audit["radius_candidates"]) <= PS.MAX_ROOM_FRACTION * 32
    assert len(audit["jump_candidates"]) <= PS.MAX_ROOM_FRACTION * 32
    assert audit["candidate_count"] == 4 > PS.MAX_ROOM_FRACTION * 32
    assert audit["state"] == "cap-exceeded"
    assert published.poses == before
    assert result is None


def test_spike_is_untouched_when_switch_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_POSE_SANITY", raising=False)
    monkeypatch.delenv("TOWER_WORLD_FINAL_SCALE_GUARD", raising=False)
    centres = _walk(60)
    centres[30] += 5.0
    solution, keyframes = _scene(centres)
    before = json.dumps(solution.poses, sort_keys=True)
    store = SimpleNamespace(read_session=lambda _w, _s: SimpleNamespace(started_at=1000.0))
    written = []
    published, _record = CP.gate_and_publish(
        store, "w", "s", SimpleNamespace(root=tmp_path), solution, final=True, gate=False,
        database_path=tmp_path / "database.db", keyframes=keyframes,
        write=lambda _workspace, value: written.append(value))
    assert "pose_sanity" not in (published.solve or {})
    assert json.dumps(published.poses, sort_keys=True) == before
    assert published.poses["s:0030"]["component"] == 0
