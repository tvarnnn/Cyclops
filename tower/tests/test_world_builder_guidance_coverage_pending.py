"""Coverage counts posed keyframes of pending segments.

The merge replaces a tracker segment only when every member is inside the
solve's horizon (`global_solve.merge`, mode `pending`). Counting a keyframe
only once its segment is replaced leaves a walk with the keyframe turn
bridge's long segments nearly empty (e2e10: 69 of 327 posed keyframes, 4/12
sectors). So a pending segment's keyframe counts when the named solve posed it
(>= 30 observations) in a component that maps to a registered, anchored
reference, at the solve's own pose carried through that reference's anchor.
`_off` (`include_pending=False`) is the replaced-segments-only baseline.
"""

from __future__ import annotations

import json
import math
import threading
import time

import pytest

from tests.test_world_builder_guidance_coverage import (
    _digest, _inputs, _landed_tree, _source_transform)
from tests.test_world_builder_guidance_dial import _camera
from tower.world_builder import guidance_coverage as coverage

# The retired switch; the live .env may still set it, and it must change nothing.
ENV = "TOWER_WORLD_GUIDANCE_COVERAGE_PENDING"


def _walk(frames, *, pending=(), refused=(), beyond=(), components=None):
    """compute_coverage inputs from [(segment, component, kid, (x, y), yaw, observations)].

    Journal order is list order. A segment in `pending` is merge's `pending`:
    no placement, its LOCAL pose rows kept (here a stale identity pose that
    must never be used), and `beyond` holds more of its keyframes, journalled
    past the solve's horizon. Every other segment is replaced and registered
    with the identity into the lowest replaced segment of its component, its
    rows already in the output frame (the first row of each is its anchor).
    `components` overrides the manifest's component rows.
    """
    pending, refused = set(pending), set(refused)
    ids, journal, poses, source = [], [], [], {}
    reference, anchored = {}, set()
    for segment, component, *_ in frames:
        if segment not in pending | refused:
            reference[component] = min(reference.get(component, segment), segment)
    for segment, component, kid, (x, y), yaw, observations in frames:
        q = _camera(yaw)
        position = [float(x), float(y), 0.0]
        rotation, translation = _source_transform(q, position)
        ids.append(kid)
        journal.append({"keyframe_id": kid})
        source[kid] = {"component": component, "observations": observations,
                       "rotation": rotation, "translation": translation}
        if segment in pending | refused:
            poses.append({"keyframe_id": kid, "segment_index": segment, "status": "solved",
                          "rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0]*3,
                          "observations": 0})
        else:
            poses.append({"keyframe_id": kid, "segment_index": segment,
                          "status": "solved" if segment in anchored else "anchor",
                          "rotation": q, "translation": position,
                          "observations": observations})
            anchored.add(segment)
    for segment, kid in beyond:
        journal.append({"keyframe_id": kid})
        poses.append({"keyframe_id": kid, "segment_index": segment, "status": "solved",
                      "rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0]*3,
                      "observations": 0})
    digest = _digest([row["keyframe_id"] for row in journal])
    segments, placements = {}, []
    for segment, component, *_ in frames:
        if str(segment) in segments:
            continue
        if segment in pending:
            segments[str(segment)] = {"replaced": False, "state": "pending"}
        elif segment in refused:
            segments[str(segment)] = {"replaced": False, "state": "refused"}
        else:
            segments[str(segment)] = {"replaced": True, "state": "registered"}
            placements.append({
                "segment_index": segment, "state": "registered",
                "reference_segment": reference[component],
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0], "translation": [0.0]*3, "scale": 1.0,
                "frame_revision": 1, "input_digest": digest,
                "evidence": {"solver": "synthetic", "component": component,
                             "coverage": "confident"}})
    if components is None:
        components = [{"component": c, "reference_segment": r}
                      for c, r in sorted(reference.items(), key=lambda item: item[1])]
    solution = {"solved_at": 1000.0, "solver": "synthetic", "keyframe_ids": ids,
                "poses": source, "input_digest": _digest(ids)}
    manifest = {"keyframes": len(journal), "input_digest": digest, "global_solve": {
        "solved_at": 1000.0, "horizon_keyframes": len(ids), "solver": "synthetic",
        "components": components, "segments": segments}}
    return dict(solution=solution, manifest=manifest, placements=placements, poses=poses,
                keyframes=journal, geometry_revision="g-pending", computed_at=1000.1)


def _union(block, ref=0):
    union = 0
    for component in block["components"]:
        if component["reference_segment"] == ref:
            for station in component["stations"]:
                union |= station["weak_mask"] | station["supported_mask"]
    return bin(union).count("1")


def _posed(block, ref=0):
    return next((c["posed_keyframes"] for c in block["components"]
                 if c["reference_segment"] == ref), 0)


def _bytes(block):
    return json.dumps(block, separators=(",", ":"), allow_nan=False).encode()


# A 4 m room, reference segment 0 (component 0) pinned at two corners and
# looking +y; segment 4 (also component 0) is still being walked.
ROOM = [(0, 0, "a0", (0, 0), 0, 30), (0, 0, "a1", (4, 4), 0, 30)]
PENDING = [(4, 0, f"b{j}", (2, 2), yaw, 30) for j, yaw in enumerate((95, 95, 185, 275))]


def _small(**kwargs):
    frames = kwargs.pop("frames", ROOM + PENDING)
    kwargs.setdefault("pending", {4})
    kwargs.setdefault("beyond", [(4, "c0"), (4, "c1")])
    return _walk(frames, **kwargs)


def _on(data, **kwargs):
    return coverage.compute_coverage(**data, **kwargs)


def _off(data, **kwargs):
    return coverage.compute_coverage(**data, include_pending=False, **kwargs)


# --- no pending segment ------------------------------------------------------------------

@pytest.mark.parametrize("data", [
    _inputs(), _inputs(((0, 82), (17, 19)), unposed_in_horizon=12, pending=14),
    _inputs(((0, 616),), unposed_in_horizon=368)], ids=["one", "mid", "stop"])
def test_on_changes_nothing_when_no_segment_is_pending(data):
    assert _bytes(_on(data)) == _bytes(_off(data))


# --- ON: what counts ---------------------------------------------------------------------

def test_on_counts_the_pending_segments_posed_keyframes():
    data = _small()
    eligible = []
    block = _on(data, eligible_out=eligible)
    assert _posed(block) == 6
    assert sorted(kid for kid, _ in eligible) == ["a0", "a1", "b0", "b1", "b2", "b3"]
    assert {ref for _, ref in eligible} == {0}
    assert _union(block) > _union(_off(data))
    # The receipt still describes the solve and the journal as they are.
    assert block["horizon_keyframes"] == 6 and block["keyframes_now"] == 8


def test_a_pending_keyframe_counts_exactly_as_it_will_once_replaced():
    """The solve's own pose through the reference anchor: the receipt is the one
    the next solve gives the same keyframes after it replaces their segment."""
    pending = _on(_small(beyond=[]))
    replaced = _off(_walk(ROOM + PENDING))
    assert _posed(replaced) == 6
    assert len(pending["components"]) == len(replaced["components"]) == 1
    got, want = pending["components"][0], replaced["components"][0]
    assert got["stations"] == want["stations"]
    assert got["posed_keyframes"] == want["posed_keyframes"]
    for name in ("bounds_xy", "origin_xyz", "up_xyz", "forward_xyz"):
        assert got[name] == pytest.approx(want[name], abs=1e-9)
    assert got["cell_size"] == pytest.approx(want["cell_size"], abs=1e-9)


def test_the_stale_local_pose_is_never_used_or_checked():
    data = _small()
    for row in data["poses"]:
        if row["segment_index"] == 4:
            row["status"] = "unavailable"
            row["rotation"] = row["translation"] = None
    assert _posed(_on(data)) == 6


def test_pending_keyframes_count_under_their_own_components_reference():
    frames = ROOM + [(7, 4, "d0", (0, 0), 0, 30), (7, 4, "d1", (1, 1), 0, 30)]
    frames += [(4, 4, f"b{j}", (0.5, 0.5), yaw, 30) for j, yaw in enumerate((95, 185))]
    block = _on(_walk(frames, pending={4}))
    assert _posed(block, 0) == 2
    assert _posed(block, 7) == 4


# --- ON: what never counts ---------------------------------------------------------------

def test_an_unmapped_component_is_not_counted():
    frames = ROOM + [(4, 9, kid, xy, yaw, obs) for _, _, kid, xy, yaw, obs in PENDING]
    assert _posed(_on(_small(frames=frames))) == 2


def test_a_reference_without_an_anchor_counts_nothing():
    data = _small()
    data["placements"][0]["evidence"]["placement_reliable"] = False
    assert _on(data)["components"] == []
    data = _small()
    data["poses"][1]["status"] = "anchor"  # two anchor rows: no unique anchor
    assert _on(data)["components"] == []


def test_a_manifest_component_row_that_disagrees_maps_nothing():
    data = _small(components=[{"component": 3, "reference_segment": 0}])
    assert _posed(_on(data)) == 2


@pytest.mark.parametrize("claimed, posed", [(0, 2), (5, 6)])
def test_a_component_two_references_claim_maps_to_neither(claimed, posed):
    """Reference 7 is registered and anchored; claiming component 0 as well as
    reference 0 does makes component 0 ambiguous, so no pending keyframe counts."""
    data = _small(frames=ROOM + [(4, 0, f"b{j}", (1+j, 2), 95, 30) for j in range(4)])
    data["placements"].append({**data["placements"][0], "segment_index": 7,
                               "reference_segment": 7})
    data["placements"][-1]["evidence"] = {**data["placements"][0]["evidence"],
                                          "component": claimed}
    data["manifest"]["global_solve"]["components"].append({"reference_segment": 7})
    data["manifest"]["global_solve"]["segments"]["7"] = {"replaced": True}
    data["poses"].append({**data["poses"][0], "keyframe_id": "z", "segment_index": 7})
    data["solution"]["poses"]["z"] = dict(data["solution"]["poses"]["a0"])
    assert sum(c["posed_keyframes"] for c in _on(data)["components"]) == posed


def test_a_reference_registered_into_another_maps_nothing():
    """A manifest reference whose own placement registers it into a different
    reference is not the frame of a component: its pending keyframes do not count."""
    data = _small(frames=ROOM + [(4, 5, f"b{j}", (1+j, 2), 95, 30) for j in range(4)])
    data["placements"].append({**data["placements"][0], "segment_index": 7,
                               "reference_segment": 0})
    data["placements"][-1]["evidence"] = {**data["placements"][0]["evidence"], "component": 5}
    data["manifest"]["global_solve"]["components"].append({"reference_segment": 7})
    data["manifest"]["global_solve"]["segments"]["7"] = {"replaced": True}
    data["poses"].append({**data["poses"][0], "keyframe_id": "z", "segment_index": 7})
    data["solution"]["poses"]["z"] = dict(data["solution"]["poses"]["a0"])
    assert sum(c["posed_keyframes"] for c in _on(data)["components"]) == 2


@pytest.mark.parametrize("observations", [29, 0, None, True, 30.0, "30"])
def test_fewer_than_thirty_observations_or_a_malformed_count_is_not_counted(observations):
    data = _small()
    for kid in ("b0", "b1", "b2", "b3"):
        data["solution"]["poses"][kid]["observations"] = observations
    assert _posed(_on(data)) == 2


def test_a_keyframe_the_solve_did_not_pose_is_not_counted():
    data = _small()
    data["solution"]["poses"].pop("b3")
    assert _posed(_on(data)) == 5


def test_a_malformed_solve_pose_is_skipped_not_fatal():
    data = _small()
    data["solution"]["poses"]["b0"]["rotation"] = [1.0]*9
    data["solution"]["poses"]["b1"]["component"] = True
    assert _posed(_on(data)) == 4


@pytest.mark.parametrize("summary", [
    None, {"replaced": False, "state": "refused"}, {"replaced": True, "state": "pending"},
    {"replaced": False}, {"state": "pending"}, "pending"])
def test_only_a_pending_segment_is_eligible(summary):
    data = _small()
    segments = data["manifest"]["global_solve"]["segments"]
    if summary is None:
        segments.pop("4")
    else:
        segments["4"] = summary
    assert _posed(_on(data)) == 2


def test_a_refused_segment_is_not_counted():
    assert _posed(_on(_small(pending=(), refused={4}))) == 2


# --- the dial (S3) -----------------------------------------------------------------------

def test_the_dial_sees_pending_keyframes():
    data = _small()
    off_dial, on_dial = {}, {}
    _off(data, dial_out=off_dial)
    block = _on(data, dial_out=on_dial)
    sectors = dict(on_dial["sectors"])
    assert len(sectors) == _union(block) > len(dict(off_dial["sectors"]))
    assert {"b0", "b1", "b2", "b3"} <= {kid for kids in sectors.values() for kid in kids}
    assert on_dial["latest"]["keyframe_id"] == "b3"
    assert on_dial["reference_segment"] == 0


# --- e2e10's shape: a short replaced head, one long pending segment -----------------------

def _e2e10_shape():
    """Segment 0 (12 keyframes) replaced, looking one way from one corner; segment 4
    (300 in the horizon, 40 more journalled past it) walks a loop of the room,
    turning through every heading. Segment 4 cannot be replaced until all 340 are in."""
    frames = [(0, 0, f"a{j:03d}", (0.1*(j % 3), 0.1*(j//3)), 10*(j % 4), 120)
              for j in range(12)]
    for j in range(300):
        theta = 2*math.pi*j/300
        frames.append((4, 0, f"b{j:03d}", (2+1.8*math.cos(theta), 2+1.8*math.sin(theta)),
                       (360*j/300 + 90) % 360, 60))
    return _walk(frames, pending={4}, beyond=[(4, f"c{j:03d}") for j in range(40)])


def test_e2e10_shape_goes_from_few_to_many_sectors_within_caps_and_budget():
    data = _e2e10_shape()
    off_dial, on_dial = {}, {}
    off = _off(data, dial_out=off_dial)
    started = time.perf_counter()
    on = _on(data, dial_out=on_dial, check=coverage._Budget().check)
    elapsed = time.perf_counter() - started
    assert _posed(off) == 12 and _posed(on) == 312
    assert _union(off) <= 4
    assert _union(on) >= max(6, 2*_union(off))
    assert len(dict(on_dial["sectors"])) == _union(on) > len(dict(off_dial["sectors"]))
    # Caps: 16 stations across all components, 4,096 bytes, 250 ms.
    assert sum(len(c["stations"]) for c in on["components"]) <= 16
    assert on["stations_omitted"] > 0
    assert len(_bytes(on)) <= 4096
    assert elapsed < coverage._LANDING_BUDGET_S
    assert on["horizon_keyframes"] == 312 and on["keyframes_pending"] == 40


def test_every_pending_keyframe_is_a_budget_check():
    data = _e2e10_shape()
    calls = {"off": 0, "on": 0}

    def counter(key):
        def check():
            calls[key] += 1
        return check
    _off(data, check=counter("off"))
    _on(data, check=counter("on"))
    assert calls["on"] - calls["off"] >= 300

    def expired():
        raise coverage._TransientCoverageError("coverage worker exceeded 250 ms")
    with pytest.raises(coverage._TransientCoverageError):
        _on(data, check=expired)


def test_pending_cannot_push_the_receipt_over_4096_bytes():
    # The 16-station cap bounds the block whatever is eligible: the station
    # count, not the keyframe count, sets its size.
    data = _e2e10_shape()
    on = _on(data)
    assert len(on["components"]) == 1
    assert len(on["components"][0]["stations"]) == 16


# --- wiring ------------------------------------------------------------------------------

def test_compute_from_tree_counts_pending_by_default(tmp_path):
    data = _small()
    root = tmp_path / "worlds"
    world, session, store = _landed_tree(root, data)
    stat = (store.world_dir(world) / "solve" / session / "solution.json").stat()
    expected = (1000.0, 6, (stat.st_size, stat.st_mtime_ns))
    off = coverage.compute_from_tree(root, world, session, expected, None, lambda: 1000.12,
                                     include_pending=False)
    on = coverage.compute_from_tree(root, world, session, expected, None, lambda: 1000.12)
    assert _posed(off) == 2 and _posed(on) == 6


def _worker_kwargs(monkeypatch, tmp_path, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    monkeypatch.delenv("TOWER_WORLD_PHOTO_FLOORPLAN", raising=False)
    seen = []

    def job(*args, **kwargs):
        seen.append(kwargs)
        return _off(_inputs())
    monkeypatch.setattr(coverage, "compute_from_tree", job)
    worker = coverage.CoverageWorker(tmp_path, lambda: 1000.12)
    try:
        worker.offer("w", "s", 1000.0, 4, (1, 1), "g", (1, "d"))
        for _ in range(200):
            with worker._condition:
                if worker._completed is not None:
                    break
            threading.Event().wait(0.01)
    finally:
        worker.shutdown()
        worker._thread.join(1)
    assert len(seen) == 1
    return seen[0]


def test_worker_takes_the_pending_default_whatever_the_env(monkeypatch, tmp_path):
    for value in (None, "on", "off"):
        assert "include_pending" not in _worker_kwargs(monkeypatch, tmp_path, value)
