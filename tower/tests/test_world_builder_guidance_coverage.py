"""Frozen FOW v1 Tower contract, including a baseline-source byte golden."""

from __future__ import annotations

import hashlib
import asyncio
import json
import math
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tower.results.world_builder import WorldBuilderStatusProducer, _geometry_block
from tower.results import world_builder as status_module
from tower.results import build_hub, make_snapshot_for
from tower.results.contracts import CARTRIDGE_WORLD_BUILDER, RESULT_TYPE_STATUS
from tower.results.envelope import ResultEnvelope
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


def test_missing_reference_placement_omits_only_that_island():
    data = _inputs(((0, 4), (17, 4)))
    data["placements"] = data["placements"][:1]
    block = _block(data)
    assert block["components_total"] == 2
    assert block["components_omitted"] == 1
    assert [row["reference_segment"] for row in block["components"]] == [0]


def test_unreliable_solve_placement_cannot_support_a_sector():
    data = _inputs()
    data["placements"][0]["evidence"]["placement_reliable"] = False
    block = _block(data)
    assert block["components_total"] == 1
    assert block["components_omitted"] == 1
    assert block["components"] == []


def test_nonfinite_derived_metadata_is_rejected_at_component_boundary():
    entries = [("a", [0.0, 0.0, 1e308], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
               ("b", [0.0, 2.0, 1e308], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])]
    with pytest.raises(ValueError, match="nonfinite"):
        coverage_module._component(0, entries)


def test_nonfinite_component_metadata_omits_only_that_island(monkeypatch):
    original = coverage_module._component
    def overflow_one(reference, entries, check=lambda: None):
        component, stations = original(reference, entries, check)
        if reference == 0:
            component["origin_xyz"][2] = float("inf")
        return component, stations
    monkeypatch.setattr(coverage_module, "_component", overflow_one)
    block = _block(_inputs(((0, 4), (17, 4))))
    assert block["components_total"] == 2
    assert block["components_omitted"] == 1
    assert [row["reference_segment"] for row in block["components"]] == [17]


def test_boolean_computed_at_is_not_a_json_number():
    data = _inputs(solved_at=0.0)
    data["computed_at"] = True
    with pytest.raises(ValueError, match="clock"):
        _block(data)


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
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    root = tmp_path / "worlds"
    store = WorldStore(root)
    store.write_world(World(world_id="w", created_at=1, updated_at=1, session_ids=("s",)))
    store.write_session(Session(session_id="s", world_id="w", started_at=1))
    producer = WorldBuilderStatusProducer(root, lambda: 1000)
    assert producer.snapshot("w", "s").payload["guidance"] == {"coverage": None}
    store.write_world(World(world_id="empty", created_at=1, updated_at=1))
    assert "guidance" not in producer.snapshot("empty", None).payload
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "garbage")
    assert producer.snapshot("w", "s").payload["guidance"] == {"coverage": None}
    off_producer = WorldBuilderStatusProducer(root, lambda: 1000,
                                             coverage_enabled=False)
    assert "guidance" not in off_producer.snapshot("w", "s").payload


def test_switch_latched_at_setup_never_starts_worker_on_later_poll(tmp_path, monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_GUIDANCE_COVERAGE", raising=False)
    root = tmp_path / "worlds"
    store = WorldStore(root)
    store.write_world(World(world_id="w", created_at=1, updated_at=1,
                            session_ids=("s",)))
    store.write_session(Session(session_id="s", world_id="w", started_at=1))
    snapshot_for = make_snapshot_for(root, lambda: 1000)
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", lambda *_: pytest.fail(
            "status poll started a guidance worker"))
        snapshot = snapshot_for(CARTRIDGE_WORLD_BUILDER, RESULT_TYPE_STATUS, "w", "s")
    assert "guidance" not in snapshot.payload


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
    release = threading.Event()
    original = coverage_module.compute_from_tree

    def delayed(*args, **kwargs):
        release.wait(timeout=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(coverage_module, "compute_from_tree", delayed)
    producer = WorldBuilderStatusProducer(root, lambda: 1000.12)
    first = producer.snapshot(world, session)
    assert first.payload["guidance"] == {"coverage": None}
    worker = producer._coverage_worker
    assert worker is not None
    release.set()
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


def test_guidance_poll_never_stats_or_starts_a_thread(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    producer = WorldBuilderStatusProducer(tmp_path, lambda: 1000.12)
    receipt = {"geometry_revision": "g", "solved_at": 1000.0}
    class Published:
        def latest(self, world, session):
            assert (world, session) == ("w", "s")
            return receipt
    producer._coverage_worker = Published()
    def forbidden(*args, **kwargs):
        raise AssertionError("guidance poll performed I/O or started a thread")
    with monkeypatch.context() as patch:
        patch.setattr(os, "stat", forbidden)
        patch.setattr(threading.Thread, "start", forbidden)
        assert producer._coverage(None, "w", "s", None, None) is receipt


def test_slow_file_read_does_not_block_status_or_shutdown_and_cannot_publish(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, _inputs())
    entered, release = threading.Event(), threading.Event()
    original = coverage_module._read_bounded

    def slow_read(*args, **kwargs):
        entered.set()
        release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(coverage_module, "_read_bounded", slow_read)
    producer = WorldBuilderStatusProducer(root, lambda: 1000.12)
    worker = producer._coverage_worker
    assert entered.wait(1)
    start = time.monotonic()
    assert producer.snapshot(world, session).payload["guidance"]["coverage"] is None
    assert time.monotonic() - start < 0.1
    start = time.monotonic()
    worker.shutdown()
    assert time.monotonic() - start < 0.1
    release.set()
    worker._thread.join(1)
    assert not worker._thread.is_alive()
    assert worker.latest(world, session) is None


def test_slow_os_read_releases_the_file_so_the_final_solve_write_lands(monkeypatch, tmp_path):
    """The final solve publishes by replacing the derived files. On Windows a
    replace onto a file a reader holds open is refused (WinError 5), and the
    writer's `replace_with_retry` gives up after 2 s. A guidance read slowed
    at the OS level must therefore close its handle at the next budget check,
    let the write land, keep the status poll fast, and publish nothing."""
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    root = tmp_path / "worlds"
    data = _inputs()
    world, session, store = _landed_tree(root, data)
    derived = store.derived_dir(world) / session
    poses_path = derived / "poses.json"
    manifest = json.loads((derived / "manifest.json").read_text(encoding="utf-8"))
    # Many small chunks, each slower than the whole 250 ms budget.
    monkeypatch.setattr(coverage_module, "_READ_CHUNK_BYTES", 16)
    assert poses_path.stat().st_size > 20 * 16
    entered, fast = threading.Event(), threading.Event()
    times = {}
    real_open = Path.open

    class SlowStream:
        def __init__(self, stream):
            self.stream = stream

        def read(self, n=-1):
            if not fast.is_set():
                time.sleep(0.3)
            return self.stream.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.stream.close()
            times.setdefault("closed", time.monotonic())

    def open_(self, *args, **kwargs):
        stream = real_open(self, *args, **kwargs)
        if threading.current_thread().name == "world-guidance" and self == poses_path:
            times.setdefault("opened", time.monotonic())
            entered.set()
            return SlowStream(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_)
    producer = WorldBuilderStatusProducer(root, lambda: 1000.12)
    worker = producer._coverage_worker
    try:
        assert entered.wait(2)
        start = time.monotonic()
        assert producer.snapshot(world, session).payload["guidance"]["coverage"] is None
        assert time.monotonic() - start < 0.1
        # The final solve's publication: the same atomic writes a merge makes.
        start = time.monotonic()
        store.write_derived(world, session, poses=data["poses"], points=[], manifest=manifest)
        assert time.monotonic() - start < 1.5
        assert "closed" in times and times["closed"] - times["opened"] < 1.0
        assert worker.latest(world, session) is None
    finally:
        worker.shutdown()
        fast.set()
        worker._thread.join(2)
    assert not worker._thread.is_alive()
    assert worker.latest(world, session) is None


def test_oversized_guidance_file_is_rejected_before_read(monkeypatch, tmp_path):
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, _inputs())
    path = root / "worlds" / world / "derived" / session / "manifest.json"
    with path.open("ab") as stream:
        stream.write(b" " * (1024 * 1024 + 1))  # past the 1 MiB manifest cap
    with pytest.raises(ValueError, match="size"):
        compute_from_tree(root, world, session, (1000.0, 4, (0, 0)), None)
    # The cap also bounds the one uninterruptible step, a single json.loads
    # (25-29 ms measured at 4 MiB): no file may be capped above that.
    assert max(coverage_module._FILE_LIMITS.values()) <= 4 * 1024 * 1024


def test_result_hub_shutdown_never_raises_from_the_guidance_hook():
    from tower.results.publisher import ResultHub

    def snapshot_for(*_):
        raise AssertionError("no snapshot is built during shutdown")

    def failing_hook():
        raise RuntimeError("guidance hook failed")

    snapshot_for.shutdown_guidance = failing_hook
    hub = ResultHub(snapshot_for, clock=lambda: 1000.0)
    asyncio.run(hub.shutdown())


def test_tower_result_hub_shutdown_cancels_guidance_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    hub = build_hub(tmp_path, lambda: 1000.12)
    # The producer was created by make_snapshot_for before any subscription.
    from tower.world_builder import guidance_coverage
    seen = []
    original = guidance_coverage.CoverageWorker.shutdown
    def shutdown(self):
        seen.append(self)
        original(self)
    monkeypatch.setattr(guidance_coverage.CoverageWorker, "shutdown", shutdown)
    asyncio.run(hub.shutdown())
    assert len(seen) == 1 and seen[0]._stop.is_set()
    seen[0]._thread.join(1)
    assert not seen[0]._thread.is_alive()


def test_on_worker_starts_before_poll_and_discovers_landing(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, _inputs())
    producer = WorldBuilderStatusProducer(root, lambda: 1000.12)
    worker = producer._coverage_worker
    assert worker is not None and worker._thread.is_alive()
    for _ in range(200):
        if worker.latest(world, session) is not None:
            break
        threading.Event().wait(0.01)
    assert worker.latest(world, session) is not None
    assert producer.snapshot(world, session).payload["guidance"]["coverage"] is not None
    with pytest.raises(TypeError, match="immutable"):
        worker.latest(world, session)["solved_at"] = 0


def test_worker_recovers_from_discovery_exception(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, _inputs())
    original = coverage_module.CoverageWorker._discover
    calls = 0

    def discover(worker):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient discovery failure")
        return original(worker)

    monkeypatch.setattr(coverage_module.CoverageWorker, "_discover", discover)
    worker = coverage_module.CoverageWorker(root, lambda: 1000.12)
    for _ in range(200):
        if worker.latest(world, session) is not None:
            break
        threading.Event().wait(0.01)
    assert calls >= 2
    assert worker.latest(world, session) is not None


def test_tower_result_setup_starts_on_worker_before_first_status(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, _inputs())
    before = {id(thread) for thread in threading.enumerate()}
    snapshot_for = make_snapshot_for(root, lambda: 1000.12)
    assert any(thread.name == "world-guidance" and id(thread) not in before
               for thread in threading.enumerate())
    for _ in range(200):
        snapshot = snapshot_for(CARTRIDGE_WORLD_BUILDER, RESULT_TYPE_STATUS, world, session)
        if snapshot.payload["guidance"]["coverage"] is not None:
            break
        threading.Event().wait(0.01)
    assert snapshot.payload["guidance"]["coverage"] is not None


def test_whole_status_envelope_with_near_cap_coverage_fits_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, _inputs())
    producer = WorldBuilderStatusProducer(root, lambda: 1000.12)
    block = _block(_inputs(tuple((i, 4) for i in range(16))))
    assert len(json.dumps(block, separators=(",", ":")).encode("utf-8")) > 3500
    producer._coverage_worker._published = {(world, session): coverage_module._freeze(block)}
    snapshot = producer.snapshot(world, session)
    envelope = ResultEnvelope(cartridge=CARTRIDGE_WORLD_BUILDER,
                              result_type=RESULT_TYPE_STATUS,
                              contract="world_builder.status/2026-09-10",
                              subscription_id="s", seq=1, revision=snapshot.revision,
                              revision_changed=True, tower_sent_at=1000.12,
                              payload=snapshot.payload)
    size = len(json.dumps(envelope.to_json_dict(), separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8"))
    assert size <= 16 * 1024


def test_largest_realistic_fow_status_uses_full_allowance(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    root = tmp_path / "worlds"
    world, session, store = _landed_tree(root, _inputs())
    producer = WorldBuilderStatusProducer(root, lambda: 1000.12)
    block = _block(_inputs(tuple((i, 4) for i in range(16))))
    producer._coverage_worker._published = {(world, session): coverage_module._freeze(block)}
    # Maximum contract station/component rows plus a large real display name
    # carried through the status and its iOS projection.
    low, high = 0, status_module._MAX_STATUS_WITH_COVERAGE_BYTES
    while low + 1 < high:
        mid = (low + high) // 2
        store.write_world(replace(store.read_world(world), display_name="x" * mid))
        payload = producer.snapshot(world, session).payload
        if payload["guidance"]["coverage"] is None:
            high = mid
        else:
            low = mid
    store.write_world(replace(store.read_world(world), display_name="x" * low))
    payload = producer.snapshot(world, session).payload
    assert payload["guidance"]["coverage"] is not None
    size = len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    assert status_module._MAX_STATUS_WITH_COVERAGE_BYTES - 32 <= size <= status_module._MAX_STATUS_WITH_COVERAGE_BYTES
    store.write_world(replace(store.read_world(world), display_name="x" * high))
    assert producer.snapshot(world, session).payload["guidance"]["coverage"] is None


REAL_LARGEST_STATUS = Path(__file__).parent / "golden" / "world_builder_status_walk6_largest_20261006.json"


def test_largest_realistic_status_with_cap_size_coverage_fits_the_15_kib_allowance():
    """The largest status measured on a real walk (5,147 B, walk 6, 1,090
    keyframes, finalizing with the longest reason sentences), given a
    64-character two-byte world name in both copies and a coverage block
    padded to its own 4,096-byte cap, still fits the FOW-specific 15 KiB
    allowance, so coverage is kept, and the complete envelope fits 16 KiB."""
    fixture = json.loads(REAL_LARGEST_STATUS.read_text(encoding="utf-8"))
    payload = fixture["payload"]
    assert "guidance" not in payload
    assert len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
               .encode("utf-8")) == fixture["payload_compact_bytes"] == 5147
    name = "é" * 64
    payload["world"]["display_name"] = name
    payload["world_snapshot"]["name"] = name
    block = _block(_inputs(tuple((i, 4) for i in range(16))))
    block_bytes = len(json.dumps(block, separators=(",", ":")).encode("utf-8"))
    assert 3500 < block_bytes <= coverage_module._MAX_BYTES
    payload["guidance"] = {"coverage": block}
    size = len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    worst = size + (coverage_module._MAX_BYTES - block_bytes)
    assert worst <= status_module._MAX_STATUS_WITH_COVERAGE_BYTES
    assert status_module._coverage_status_fits(payload)
    envelope = ResultEnvelope(cartridge=CARTRIDGE_WORLD_BUILDER,
                              result_type=RESULT_TYPE_STATUS,
                              contract="world_builder.status/2026-09-10",
                              subscription_id="sub-1", seq=1190, revision="7ed5b1bcc48926d7",
                              revision_changed=True, tower_sent_at=1791329472.3530006,
                              payload=payload)
    envelope_size = len(json.dumps(envelope.to_json_dict(), separators=(",", ":"),
                                   ensure_ascii=False).encode("utf-8"))
    assert envelope_size + (coverage_module._MAX_BYTES - block_bytes) <= 16 * 1024
    assert status_module._MAX_STATUS_WITH_COVERAGE_BYTES == 15 * 1024


def test_coverage_falls_back_to_null_if_complete_status_exceeds_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    root = tmp_path / "worlds"
    world, session, store = _landed_tree(root, _inputs())
    producer = WorldBuilderStatusProducer(root, lambda: 1000.12)
    block = _block(_inputs(tuple((i, 4) for i in range(16))))

    class Published:
        def latest(self, *_):
            return block

    producer._coverage_worker = Published()
    base = producer.snapshot(world, session).payload
    base["guidance"]["coverage"] = None
    base_size = len(json.dumps(base, separators=(",", ":"),
                               ensure_ascii=False).encode("utf-8"))
    # The iOS projection repeats the world name, so reserve for both copies.
    padding = (status_module._MAX_STATUS_WITH_COVERAGE_BYTES - base_size - 64) // 2
    assert padding > 0
    store.write_world(replace(store.read_world(world), display_name="x" * padding))
    snapshot = producer.snapshot(world, session)
    assert snapshot.payload["guidance"] == {"coverage": None}
    assert len(json.dumps(snapshot.payload, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")) <= status_module._MAX_STATUS_WITH_COVERAGE_BYTES
    assert snapshot.revision == status_module.compute_revision(snapshot.payload,
                                                               snapshot.volatile_fields)


def test_tree_snapshot_refuses_torn_identity_and_placement(tmp_path, monkeypatch):
    data = _inputs()
    root = tmp_path / "worlds"
    world, session, store = _landed_tree(root, data)
    solution = store.world_dir(world) / "solve" / session / "solution.json"
    stat = solution.stat()
    manifest = json.loads((store.derived_dir(world) / session / "manifest.json").read_text(encoding="utf-8"))
    revision = _geometry_block(manifest, True, 4, has_session_geometry=True)["revision"]
    block = compute_from_tree(root, world, session, (1000.0, 4, (stat.st_size, stat.st_mtime_ns)),
                              revision, lambda: 1000.12)
    assert block["horizon_keyframes"] == 4
    with pytest.raises(ValueError, match="changed"):
        compute_from_tree(root, world, session, (1000.0, 4, (0, 0)), "g-tree", lambda: 1000.12)
    original_json = coverage_module._json
    manifest_path = store.derived_dir(world) / session / "manifest.json"
    touched = False
    def change_during_read(path, budget=None):
        nonlocal touched
        result = original_json(path, budget)
        if path.name == "placements.json" and not touched:
            manifest_path.write_text(manifest_path.read_text(encoding="utf-8")+" ", encoding="utf-8")
            touched = True
        return result
    with monkeypatch.context() as patch:
        patch.setattr(coverage_module, "_json", change_during_read)
        with pytest.raises(ValueError, match="changed"):
            compute_from_tree(root, world, session, (1000.0, 4, (stat.st_size, stat.st_mtime_ns)),
                              revision, lambda: 1000.12)
    placement = store.derived_dir(world) / session / "placements.json"
    doc = json.loads(placement.read_text(encoding="utf-8"))
    doc["placements"][0]["input_digest"] = "stale"
    placement.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        compute_from_tree(root, world, session, (1000.0, 4, (stat.st_size, stat.st_mtime_ns)),
                          revision, lambda: 1000.12)


def test_tree_snapshot_accepts_live_tail_after_frozen_build(tmp_path):
    data = _inputs()
    root = tmp_path / "worlds"
    world, session, store = _landed_tree(root, data)
    solution = store.world_dir(world) / "solve" / session / "solution.json"
    stat = solution.stat()
    # The build and solve froze four inputs. The accepted journal grows before
    # guidance reads it; no derived file is rewritten for the fifth keyframe.
    store.append_keyframe(world, Keyframe(
        keyframe_id="live-tail", session_id=session, source_seq=4,
        received_at=905, image_relpath="images/4.jpg", width=1, height=1,
        byte_count=1))
    block = compute_from_tree(root, world, session,
                              (1000.0, 4, (stat.st_size, stat.st_mtime_ns)),
                              None, lambda: 1000.12)
    assert (block["horizon_keyframes"], block["keyframes_now"],
            block["keyframes_pending"]) == (4, 5, 1)


def test_tree_snapshot_refuses_other_session_and_solve_with_live_tail(tmp_path):
    data = _inputs(pending=1)
    root = tmp_path / "worlds"
    world, session, store = _landed_tree(root, data)
    solution = store.world_dir(world) / "solve" / session / "solution.json"
    stat = solution.stat()
    expected = (1000.0, 4, (stat.st_size, stat.st_mtime_ns))
    manifest_path = store.derived_dir(world) / session / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["session_id"] = "another-session"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="session"):
        compute_from_tree(root, world, session, expected, None, lambda: 1000.12)
    manifest["session_id"] = session
    manifest["global_solve"]["solve"] = {"run": "another-solve"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="solution identity"):
        compute_from_tree(root, world, session, expected, None, lambda: 1000.12)


def test_frozen_inputs_refuse_changed_prefix_and_bad_solve_digest():
    data = _inputs(pending=1)
    data["keyframes"][0]["keyframe_id"] = "another-walk"
    with pytest.raises(ValueError, match="accepted journal mismatch") as refused:
        _block(data)
    assert not isinstance(refused.value, coverage_module._TransientCoverageError)
    data = _inputs(pending=1)
    # Walk 6's 501 solution declared the digest of 499 IDs but listed 501.
    data["solution"]["input_digest"] = _digest(data["solution"]["keyframe_ids"][:-1])
    with pytest.raises(ValueError, match="solution input digest mismatch") as refused:
        _block(data)
    assert not isinstance(refused.value, coverage_module._TransientCoverageError)


def test_journal_behind_frozen_build_is_retryable():
    data = _inputs(pending=1)
    data["keyframes"].pop()
    with pytest.raises(coverage_module._TransientCoverageError,
                       match="accepted journal mismatch"):
        _block(data)


def test_tree_snapshot_refuses_other_session_journal(tmp_path):
    root = tmp_path / "worlds"
    world, session, store = _landed_tree(root, _inputs())
    journal = store.keyframes_path(world, session)
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    rows[0]["session_id"] = "another-session"
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    solution = store.world_dir(world) / "solve" / session / "solution.json"
    stat = solution.stat()
    with pytest.raises(ValueError, match="session"):
        compute_from_tree(root, world, session,
                          (1000.0, 4, (stat.st_size, stat.st_mtime_ns)),
                          None, lambda: 1000.12)


def test_live_tail_still_refuses_wrong_build_digest():
    data = _inputs(pending=1)
    data["manifest"]["input_digest"] = _digest(data["solution"]["keyframe_ids"])
    with pytest.raises(ValueError, match="manifest input digest mismatch"):
        _block(data)


def test_tree_snapshot_rejects_same_tag_with_different_geometry_revision(tmp_path):
    data = _inputs()
    root = tmp_path / "worlds"
    world, session, store = _landed_tree(root, data)
    path = store.derived_dir(world) / session / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    revision = _geometry_block(manifest, True, 4, has_session_geometry=True)["revision"]
    solution = store.world_dir(world) / "solve" / session / "solution.json"
    stat = solution.stat()
    expected = (1000.0, 4, (stat.st_size, stat.st_mtime_ns))
    assert compute_from_tree(root, world, session, expected, revision,
                             lambda: 1000.12)["geometry_revision"] == revision
    manifest["points"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry revision"):
        compute_from_tree(root, world, session, expected, revision, lambda: 1000.12,
                          (manifest["built_at"], manifest["input_digest"]))


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
