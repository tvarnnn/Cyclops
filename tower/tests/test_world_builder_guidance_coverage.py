"""Frozen FOW v1 Tower contract, including a baseline-source byte golden."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tower.results.world_builder import WorldBuilderStatusProducer
from tower.world_builder.events import EventLog
from tower.world_builder.guidance_coverage import compute_coverage, compute_from_tree
from tower.world_builder import guidance_coverage as coverage_module
from tower.world_builder.records import Keyframe, Session, World
from tower.world_builder.store import WorldStore

GOLDEN = Path(__file__).parent / "golden" / "world_builder_guidance_off_324f4a6.json"
Q_NORTH = [math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0]


def _source_transform(q, centre):
    axes = [coverage_module._rotate(q, axis) for axis in ([1.0, 0.0, 0.0],
                                                          [0.0, 1.0, 0.0],
                                                          [0.0, 0.0, 1.0])]
    r_cw = [[axes[i][j] for j in range(3)] for i in range(3)]
    translation = [-sum(r_cw[i][j]*centre[j] for j in range(3)) for i in range(3)]
    return [v for row in r_cw for v in row], translation


def _digest(ids):
    return hashlib.sha256(b"".join(k.encode()+b"\0" for k in ids)).hexdigest()


def _inputs(groups=((0, 4),), *, pending=0, unposed_in_horizon=0, solved_at=1000.0):
    ids, keyframes, poses, source_poses, placements, components, segments = [], [], [], {}, [], [], {}
    for ref, count in groups:
        components.append({"reference_segment": ref})
        segments[str(ref)] = {"replaced": True}
        placements.append({"segment_index": ref, "state": "registered", "reference_segment": ref,
                           "rotation_wxyz": Q_NORTH, "translation": [0.0]*3,
                           "scale": 1.0, "frame_revision": 1,
                           "evidence": {"solver": "synthetic", "component": ref, "coverage": "confident"}})
        for j in range(count):
            kid = f"k{ref}-{j}"
            ids.append(kid)
            # Two cameras per end cell support the same heading. No frustum
            # or ray distance is inferred from their pointing direction.
            position = [0.0, 0.0, float(j//2)*2]
            world_position = coverage_module._rotate(Q_NORTH, position)
            source_rotation, source_translation = _source_transform(Q_NORTH, world_position)
            keyframes.append({"keyframe_id": kid})
            poses.append({"keyframe_id": kid, "segment_index": ref,
                          "status": "anchor" if j == 0 else "solved",
                          "rotation": [1.0, 0.0, 0.0, 0.0],
                          "translation": position, "observations": 30})
            source_poses[kid] = {"component": ref, "observations": 30,
                                 "rotation": source_rotation, "translation": source_translation}
    for j in range(unposed_in_horizon):
        kid = f"unposed-{j}"
        ids.append(kid)
        keyframes.append({"keyframe_id": kid})
    for j in range(pending):
        keyframes.append({"keyframe_id": f"pending-{j}"})
    for placement in placements:
        placement["input_digest"] = _digest([row["keyframe_id"] for row in keyframes])
    solution = {"solved_at": solved_at, "solver": "synthetic", "keyframe_ids": ids, "poses": source_poses,
                "input_digest": _digest(ids)}
    manifest = {"keyframes": len(keyframes), "input_digest": _digest(
        [row["keyframe_id"] for row in keyframes]), "global_solve": {
            "solved_at": solved_at, "horizon_keyframes": len(ids), "solver": "synthetic",
            "components": components, "segments": segments}}
    return dict(solution=solution, manifest=manifest, placements=placements,
                poses=poses, keyframes=keyframes, geometry_revision="g-synthetic",
                computed_at=solved_at+0.12)


def _block(data=None):
    return compute_coverage(**(data or _inputs()))


def _raw(snapshot):
    return json.dumps({"payload": snapshot.payload, "revision": snapshot.revision,
                       "volatile_fields": snapshot.volatile_fields},
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _status_matrix(root, producer_class=WorldBuilderStatusProducer):
    store = WorldStore(root)
    world_id, session_id = "fow-world", "fow-session"
    producer = producer_class(root, lambda: 1000.0)
    result = [_raw(producer.snapshot(None, None))]
    store.write_world(World(world_id=world_id, created_at=900.0, updated_at=900.0,
                            session_ids=()))
    result.append(_raw(producer.snapshot(world_id, None)))
    session = Session(session_id=session_id, world_id=world_id, started_at=900.0)
    store.write_session(session)
    store.write_world(World(world_id=world_id, created_at=900.0, updated_at=900.0,
                            session_ids=(session_id,)))
    events = EventLog(store, world_id, session_id, lambda: 900.0)
    events.append("session_started")
    result.append(_raw(producer.snapshot(world_id, session_id)))
    store.write_derived(world_id, session_id, poses=[], points=[], manifest={
        "schema_version": 1, "input_digest": _digest([]), "session_id": session_id, "keyframes": 0,
        "points": 0, "poses_solved": 0, "poses_refused": 0, "segments": 0,
        "built_at": 950.0, "backend_id": "synthetic", "scale_state": "unknown"})
    result.append(_raw(producer.snapshot(world_id, session_id)))
    events.append("session_stopped")
    store.write_session(replace(session, ended_at=990.0, end_reason="user_stop"))
    result.append(_raw(producer.snapshot(world_id, session_id)))
    return result


def test_off_exact_bytes_twice_and_one_byte_mutant(tmp_path, monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_GUIDANCE_COVERAGE", raising=False)
    source = os.environ.get("FOW_BASELINE_SOURCE")
    if source:
        import types
        baseline = types.ModuleType("tower.results.world_builder_baseline")
        exec(Path(source).read_text(encoding="utf-8-sig"), baseline.__dict__)
        producer_class = baseline.WorldBuilderStatusProducer
    else:
        producer_class = WorldBuilderStatusProducer
    first = _status_matrix(tmp_path / "first", producer_class)
    second = _status_matrix(tmp_path / "second", producer_class)
    assert first == second, [(i, next(((j, a[j-30:j+80], b[j-30:j+80]) for j in range(min(len(a), len(b))) if a[j] != b[j]), None)) for i, (a, b) in enumerate(zip(first, second)) if a != b]
    if os.environ.get("FOW_RECORD_GOLDEN"):
        GOLDEN.write_text(json.dumps([r.decode() for r in first], indent=2)+"\n", encoding="utf-8")
    golden = [r.encode() for r in json.loads(GOLDEN.read_text(encoding="utf-8"))]
    assert first == golden
    mutant = bytearray(first[2]); mutant[0] ^= 1
    assert bytes(mutant) != golden[2]


def test_three_synthetic_guidance_values_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    early = {"guidance": json.loads(_status_matrix(tmp_path / "early")[2])["payload"]["guidance"]}
    mid = {"guidance": {"coverage": _block(_inputs(
        ((0, 82), (17, 19)), unposed_in_horizon=12, pending=14))}}
    stop = {"guidance": {"coverage": _block(_inputs(
        ((0, 616),), unposed_in_horizon=368))}}
    for fixture in (early, mid, stop):
        decoded = json.loads(json.dumps(fixture, allow_nan=False))
        assert decoded == fixture
    assert mid["guidance"]["coverage"]["keyframes_pending"] == 14
    assert mid["guidance"]["coverage"]["horizon_keyframes"] == 113
    assert mid["guidance"]["coverage"]["keyframes_now"] == 127
    assert mid["guidance"]["coverage"]["components_total"] == 2
    assert stop["guidance"]["coverage"]["keyframes_now"] == 984
    assert 1090-stop["guidance"]["coverage"]["horizon_keyframes"] == 106
    for fixture in (mid, stop):
        for component in fixture["guidance"]["coverage"]["components"]:
            assert component["stations"]
            for station in component["stations"]:
                assert station["weak_mask"] & station["supported_mask"] == 0


def test_no_pointing_only_and_distinct_support_threshold():
    one = _block(_inputs(((0, 1),)))
    assert one["components"] == []  # a single-point map is not a room
    data = _inputs(((0, 4),))
    for row in data["poses"]:
        row["status"] = "unavailable"
    assert _block(data)["components"] == []
    data = _inputs(((0, 4),))
    data["poses"][1]["status"] = "unavailable"
    stations = _block(data)["components"][0]["stations"]
    assert any(s["weak_mask"] for s in stations)
    assert any(s["supported_mask"] for s in stations)
    assert all(s["weak_mask"] & s["supported_mask"] == 0 for s in stations)


def test_anchor_requires_solution_and_accepted_join():
    data = _inputs()
    data["solution"]["poses"].pop("k0-0")
    assert _block(data)["components"] == []
    data = _inputs()
    data["keyframes"] = data["keyframes"][:-1]
    with pytest.raises(ValueError, match="journal|digest"):
        _block(data)


def test_unresolved_partial_and_component_isolation():
    data = _inputs(((0, 4), (17, 4)))
    data["placements"][0]["evidence"]["coverage"] = "unresolved"
    result = _block(data)
    assert [c["reference_segment"] for c in result["components"]] == [17]
    assert result["components_omitted"] == 1
    data["placements"][0]["evidence"]["coverage"] = "partial"
    result = _block(data)
    assert [c["reference_segment"] for c in result["components"]] == [0, 17]
    assert result["components"][0]["stations"] != []


def test_omission_reasons_and_global_station_cap():
    data = _inputs(tuple((i, 4) for i in range(17)))
    result = _block(data)
    assert result["components_total"] == 17
    assert result["components_omitted"] >= 1  # valid island loses global top 16
    assert result["stations_omitted"] > 0
    data = _inputs(((0, 4), (17, 4)))
    for row in data["poses"][:4]:
        row["rotation"] = [float("nan")]*4
    result = _block(data)
    assert result["components_omitted"] == 1  # malformed basis / pose evidence


def test_refuse_mixed_frame_source_clock_and_nonfinite():
    data = _inputs(((0, 4), (17, 4)))
    data["placements"][1]["frame_revision"] = 2
    with pytest.raises(ValueError, match="consistent"):
        _block(data)
    data = _inputs()
    data["placements"][0]["evidence"]["solver"] = "other"
    with pytest.raises(ValueError, match="source"):
        _block(data)
    data = _inputs()
    data["solution"].pop("input_digest")
    with pytest.raises(ValueError, match="solution input digest"):
        _block(data)
    data = _inputs()
    data["computed_at"] = data["solution"]["solved_at"]-0.001
    with pytest.raises(ValueError, match="clock"):
        _block(data)
    data = _inputs()
    data["poses"][1]["translation"] = [float("inf"), 0, 0]
    assert sum(s["keyframes"] for s in _block(data)["components"][0]["stations"]) == 3
    data = _inputs()
    data["solution"]["poses"]["k0-2"]["translation"][2] -= 1.0
    with pytest.raises(ValueError, match="disagree"):
        _block(data)


def test_compass_cell_edges_and_mask_bounds():
    data = _inputs()
    result = _block(data)
    assert result["components"][0]["forward_xyz"][1] == pytest.approx(1)
    assert result["components"][0]["up_xyz"][2] == pytest.approx(1)
    assert {s["x"] for s in result["components"][0]["stations"]} == {0, 7}
    for station in result["components"][0]["stations"]:
        assert 0 <= station["weak_mask"] <= 4095
        assert 0 <= station["supported_mask"] <= 4095
        assert station["supported_mask"] == 1


def test_clockwise_north_east_and_thirty_degree_boundary():
    up = [0.0, 0.0, 1.0]
    theta = math.radians(30)
    entries = [
        ("a", [0.0, 0.0, 0.0], up, [0.0, 1.0, 0.0]),
        ("b", [0.0, 0.0, 0.0], up, [0.0, 1.0, 0.0]),
        ("c", [0.0, 2.0, 0.0], up, [1.0, 0.0, 0.0]),
        ("d", [0.0, 2.0, 0.0], up, [math.sin(theta), math.cos(theta), 0.0]),
    ]
    _, stations = coverage_module._component(0, entries)
    assert any(s["supported_mask"] == 1 for s in stations)
    assert any(s["weak_mask"] == (1 << 1) | (1 << 3) for s in stations)


def test_door_wall_negative_stays_unconfirmed_from_other_island_and_rejected_views():
    data = _inputs(((0, 4), (17, 4)))
    south = [0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)]
    for row in data["poses"][4:]:
        row["rotation"] = [0.0, 0.0, 1.0, 0.0]
        position = coverage_module._rotate(Q_NORTH, row["translation"])
        rotation, translation = _source_transform(south, position)
        data["solution"]["poses"][row["keyframe_id"]]["rotation"] = rotation
        data["solution"]["poses"][row["keyframe_id"]]["translation"] = translation
    # A pointing-only frame and a local-only row cannot join the accepted
    # journal and named solve, even if their heading is into the grey half.
    data["poses"].append({"keyframe_id": "raw-pointing", "segment_index": 0,
                          "status": "solved", "rotation": south,
                          "translation": [0.0, 0.0, 0.0], "observations": 30})
    data["poses"][1]["status"] = "unavailable"
    block = _block(data)
    room = next(c for c in block["components"] if c["reference_segment"] == 0)
    assert all(s["supported_mask"] & (1 << 6) == 0 for s in room["stations"])
    bathroom = next(c for c in block["components"] if c["reference_segment"] == 17)
    assert bathroom["stations"] and room["stations"]
    assert all(s["weak_mask"] & s["supported_mask"] == 0 for c in block["components"]
               for s in c["stations"])


def test_four_kib_cap():
    # The cap is enforced on UTF-8 serialized bytes, not Python character count.
    data = _inputs(tuple((i, 4) for i in range(16)))
    data["geometry_revision"] = "z"*128
    block = _block(data)
    assert coverage_module._MAX_BYTES == 4096
    assert len(json.dumps(block, separators=(",", ":")).encode()) <= 4096
    old_cap = coverage_module._MAX_BYTES
    coverage_module._MAX_BYTES = 100
    try:
        with pytest.raises(ValueError, match="4096"):
            _block(data)
    finally:
        coverage_module._MAX_BYTES = old_cap


def test_on_selected_without_landing_is_null_and_off_omits(tmp_path, monkeypatch):
    root = tmp_path / "worlds"
    store = WorldStore(root)
    store.write_world(World(world_id="w", created_at=1, updated_at=1, session_ids=("s",)))
    store.write_session(Session(session_id="s", world_id="w", started_at=1))
    producer = WorldBuilderStatusProducer(root, lambda: 1000)
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    assert producer.snapshot("w", "s").payload["guidance"] == {"coverage": None}
    store.write_world(World(world_id="empty", created_at=1, updated_at=1))
    assert "guidance" not in producer.snapshot("empty", None).payload
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "garbage")
    assert "guidance" not in producer.snapshot("w", "s").payload


def _landed_tree(root, data):
    store = WorldStore(root)
    world_id, session_id = "fow-world", "fow-session"
    store.write_world(World(world_id=world_id, created_at=900, updated_at=950,
                            session_ids=(session_id,)))
    store.write_session(Session(session_id=session_id, world_id=world_id, started_at=900))
    events = EventLog(store, world_id, session_id, lambda: 950)
    events.append("session_started")
    for j, row in enumerate(data["keyframes"]):
        store.append_keyframe(world_id, Keyframe(
            keyframe_id=row["keyframe_id"], session_id=session_id, source_seq=j,
            received_at=901+j, image_relpath=f"images/{j}.jpg", width=1, height=1,
            byte_count=1))
        events.append("keyframe_accepted", {"keyframe_id": row["keyframe_id"]})
    manifest = {**data["manifest"], "schema_version": 1, "session_id": session_id,
                "points": 0, "poses_solved": len(data["poses"]), "poses_refused": 0,
                "segments": len(data["placements"]), "built_at": 950.0,
                "backend_id": "synthetic", "scale_state": "unknown"}
    store.write_derived(world_id, session_id, poses=data["poses"], points=[], manifest=manifest)
    derived = store.derived_dir(world_id) / session_id
    (derived / "placements.json").write_text(json.dumps({"placements": data["placements"]}), encoding="utf-8")
    solution_path = store.world_dir(world_id) / "solve" / session_id / "solution.json"
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    solution_path.write_text(json.dumps(data["solution"]), encoding="utf-8")
    return world_id, session_id, store


def test_worker_publishes_only_completed_receipt_and_keeps_revision(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    data = _inputs()
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, data)
    producer = WorldBuilderStatusProducer(root, lambda: 1000.12)
    first = producer.snapshot(world, session)
    assert first.payload["guidance"] == {"coverage": None}
    worker = producer._coverage_worker
    assert worker is not None
    deadline = threading.Event()
    for _ in range(100):
        with worker._condition:
            if worker._completed is not None:
                break
        deadline.wait(0.01)
    second = producer.snapshot(world, session)
    assert second.payload["guidance"]["coverage"] is not None
    assert second.revision != first.revision
    assert second.payload["geometry"]["revision"] == first.payload["geometry"]["revision"]
    assert producer.snapshot(world, session).revision == second.revision
    assert worker._pending is None  # duplicate landing did not enqueue another job


def test_tree_snapshot_refuses_torn_identity_and_placement(tmp_path, monkeypatch):
    data = _inputs()
    root = tmp_path / "worlds"
    world, session, store = _landed_tree(root, data)
    solution = store.world_dir(world) / "solve" / session / "solution.json"
    stat = solution.stat()
    block = compute_from_tree(root, world, session, (1000.0, 4, (stat.st_size, stat.st_mtime_ns)),
                              "g-tree", lambda: 1000.12)
    assert block["horizon_keyframes"] == 4
    with pytest.raises(ValueError, match="changed"):
        compute_from_tree(root, world, session, (1000.0, 4, (0, 0)), "g-tree", lambda: 1000.12)
    original_json = coverage_module._json
    manifest_path = store.derived_dir(world) / session / "manifest.json"
    touched = False
    def change_during_read(path):
        nonlocal touched
        result = original_json(path)
        if path.name == "placements.json" and not touched:
            manifest_path.write_text(manifest_path.read_text(encoding="utf-8")+" ", encoding="utf-8")
            touched = True
        return result
    with monkeypatch.context() as patch:
        patch.setattr(coverage_module, "_json", change_during_read)
        with pytest.raises(ValueError, match="changed"):
            compute_from_tree(root, world, session, (1000.0, 4, (stat.st_size, stat.st_mtime_ns)),
                              "g-tree", lambda: 1000.12)
    placement = store.derived_dir(world) / session / "placements.json"
    doc = json.loads(placement.read_text(encoding="utf-8"))
    doc["placements"][0]["input_digest"] = "stale"
    placement.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        compute_from_tree(root, world, session, (1000.0, 4, (stat.st_size, stat.st_mtime_ns)),
                          "g-tree", lambda: 1000.12)


def test_stale_receipt_publishes_no_directional_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, _inputs(((0, 22),)))
    producer = WorldBuilderStatusProducer(root, lambda: 1040.0)
    producer.snapshot(world, session)
    worker = producer._coverage_worker
    for _ in range(100):
        with worker._condition:
            if worker._completed is not None:
                break
        threading.Event().wait(0.01)
    guidance = producer.snapshot(world, session).payload["guidance"]
    assert guidance["coverage"]["solved_at"] == 1000.0
    assert 1040.0-guidance["coverage"]["solved_at"] > 30
    assert set(guidance) == {"coverage"}  # optional prompt is not shipped


def test_slow_failed_worker_keeps_previous_receipt_without_blocking_offer(monkeypatch, tmp_path):
    from tower.world_builder.guidance_coverage import CoverageWorker
    first = _block()
    calls = []
    def job(*args):
        calls.append(args)
        if len(calls) == 1:
            return first
        time.sleep(0.27)
        return _block(_inputs(solved_at=1001.0))
    monkeypatch.setattr(coverage_module, "compute_from_tree", job)
    worker = CoverageWorker(tmp_path, lambda: 1000.12)
    assert worker.offer("w", "s", 1000.0, 4, (1, 1), "g1", (1, "d")) is None
    for _ in range(100):
        with worker._condition:
            if worker._completed is not None:
                break
        threading.Event().wait(0.01)
    assert worker.offer("w", "s", 1000.0, 4, (1, 1), "g1", (1, "d")) == first
    start = time.monotonic()
    assert worker.offer("w", "s", 1001.0, 5, (2, 2), "g2", (2, "d")) == first
    assert time.monotonic()-start < 0.05
    for _ in range(100):
        with worker._condition:
            if worker._completed and worker._completed[2] == 1001.0:
                break
        threading.Event().wait(0.01)
    assert worker.offer("w", "s", 1001.0, 5, (2, 2), "g2", (2, "d")) == first
    assert len(calls) == 2


def test_two_pinned_sessions_do_not_resubmit_completed_landings(monkeypatch, tmp_path):
    from tower.world_builder.guidance_coverage import CoverageWorker
    calls = []
    def job(root, world, session, *args):
        calls.append(session)
        return _block()
    monkeypatch.setattr(coverage_module, "compute_from_tree", job)
    worker = CoverageWorker(tmp_path, lambda: 1000.12)
    for session in ("a", "b"):
        worker.offer("w", session, 1000.0, 4, (1, 1), "g", (1, "d"))
        for _ in range(100):
            with worker._condition:
                if worker._completed is not None and worker._completed[1] == session:
                    break
            threading.Event().wait(0.01)
    assert worker.offer("w", "a", 1000.0, 4, (1, 1), "g", (1, "d")) is not None
    assert worker.offer("w", "b", 1000.0, 4, (1, 1), "g", (1, "d")) is not None
    assert calls == ["a", "b"]


@pytest.mark.parametrize("mutant", [
    "reverse_cross", "zero_dot", "huge_epsilon", "tiny_cap",
    "zero_integer", "wrong_up", "no_rotation", "promote_weak",
])
def test_eight_guidance_mutants_are_killed(monkeypatch, mutant):
    """Each local mutation changes new-code behavior; the same semantic oracle kills it."""
    if mutant == "reverse_cross":
        original = coverage_module._cross
        monkeypatch.setattr(coverage_module, "_cross", lambda a, b: original(b, a))
    elif mutant == "zero_dot":
        monkeypatch.setattr(coverage_module, "_dot", lambda a, b: 0)
    elif mutant == "huge_epsilon":
        monkeypatch.setattr(coverage_module, "_EPS", 100)
    elif mutant == "tiny_cap":
        monkeypatch.setattr(coverage_module, "_MAX_BYTES", 10)
    elif mutant == "zero_integer":
        monkeypatch.setattr(coverage_module, "_integer", lambda value, maximum: 0)
    elif mutant == "wrong_up":
        monkeypatch.setattr(coverage_module, "_unit", lambda v: [1.0, 0.0, 0.0])
    elif mutant == "no_rotation":
        monkeypatch.setattr(coverage_module, "_rotate", lambda q, v: v)
    else:
        original = coverage_module._component
        def promote(*args):
            component, stations = original(*args)
            for station in stations:
                station["supported_mask"] |= station["weak_mask"]
            return component, stations
        monkeypatch.setattr(coverage_module, "_component", promote)
    data = _inputs()
    if mutant == "promote_weak":
        data["poses"][1]["status"] = "unavailable"
    try:
        block = _block(data)
        component = block["components"][0]
        assert block["horizon_keyframes"] == 4
        assert component["up_xyz"][2] == pytest.approx(1)
        assert component["forward_xyz"][1] == pytest.approx(1)
        assert {s["x"] for s in component["stations"]} == {0, 7}
        assert all(s["weak_mask"] & s["supported_mask"] == 0 for s in component["stations"])
    except (ValueError, TypeError, IndexError, ZeroDivisionError, AssertionError):
        return
    pytest.fail(f"mutant survived: {mutant}")
