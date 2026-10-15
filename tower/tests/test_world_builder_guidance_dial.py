"""FOW capture dial (S3): B2 per-sector selection, `latest` and `motion_sector`.

The coverage worker always computes the dial for B2; the coverage receipt
itself must not change by a byte.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_world_builder_guidance_coverage import (
    Q_NORTH, _digest, _inputs, _landed_tree, _source_transform)
from tower.results import world_builder_guidance_imagery as route
from tower.world_builder import guidance_coverage as coverage
from tower.world_builder import guidance_imagery as imagery

UP = [0.0, 0.0, 1.0]


def _ray(yaw):
    """A level camera ray, `yaw` degrees clockwise (seen from above) from +y."""
    theta = math.radians(yaw)
    return [math.sin(theta), math.cos(theta), 0.0]


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2]


def _camera(yaw):
    """Camera-to-world rotation: up +z, forward `_ray(yaw)`."""
    half = math.radians(-yaw)/2
    return _quat_mul([math.cos(half), 0.0, 0.0, math.sin(half)], Q_NORTH)


def _world(frames, *, solved_at=1000.0):
    """compute_coverage inputs from [(ref, kid, (x, y), yaw)] in journal order.

    The registered placement is the identity, so each pose row is already in
    the output frame; the first frame of each ref is that component's anchor.
    """
    ids, keyframes, poses, source, refs = [], [], [], {}, []
    for ref, kid, (x, y), yaw in frames:
        q = _camera(yaw)
        position = [float(x), float(y), 0.0]
        rotation, translation = _source_transform(q, position)
        ids.append(kid)
        keyframes.append({"keyframe_id": kid})
        poses.append({"keyframe_id": kid, "segment_index": ref,
                      "status": "solved" if ref in refs else "anchor",
                      "rotation": q, "translation": position, "observations": 30})
        source[kid] = {"component": ref, "observations": 30,
                       "rotation": rotation, "translation": translation}
        if ref not in refs:
            refs.append(ref)
    placements = [{"segment_index": ref, "state": "registered", "reference_segment": ref,
                   "rotation_wxyz": [1.0, 0.0, 0.0, 0.0], "translation": [0.0]*3,
                   "scale": 1.0, "frame_revision": 1, "input_digest": _digest(ids),
                   "evidence": {"solver": "synthetic", "component": ref, "coverage": "confident"}}
                  for ref in refs]
    solution = {"solved_at": solved_at, "solver": "synthetic", "keyframe_ids": ids,
                "poses": source, "input_digest": _digest(ids)}
    manifest = {"keyframes": len(ids), "input_digest": _digest(ids), "global_solve": {
        "solved_at": solved_at, "horizon_keyframes": len(ids), "solver": "synthetic",
        "components": [{"reference_segment": ref} for ref in refs],
        "segments": {str(ref): {"replaced": True} for ref in refs}}}
    return dict(solution=solution, manifest=manifest, placements=placements, poses=poses,
                keyframes=keyframes, geometry_revision="g-dial", computed_at=solved_at+0.1)


def _dial(data):
    dial = {}
    block = coverage.compute_coverage(**data, dial_out=dial)
    return block, dial


def _union(component):
    union = 0
    for station in component["stations"]:
        union |= station["weak_mask"] | station["supported_mask"]
    return {sector for sector in range(12) if union >> sector & 1}


def _bytes(block):
    return json.dumps(block, separators=(",", ":"), allow_nan=False).encode()


# A 4 m square room: corners pin cell_size to 0.5 m (side 4 m / 8). Their ids sort
# first, so their heading (+y) is the sector frame (first qualified forward).
ROOM = [(0, "00", (0, 0), 0), (0, "01", (4, 4), 0)]


def test_yaw_fixture_bins_where_the_masks_say():
    block, dial = _dial(_world(ROOM + [(0, "a", (2, 2), 95), (0, "b", (2, 2), 95)]))
    component = block["components"][0]
    assert component["forward_xyz"] == pytest.approx([0.0, 1.0, 0.0])
    assert component["cell_size"] == pytest.approx(0.5)
    assert _union(component) == {0, 3}
    assert dict(dial["sectors"])[3] == ("b", "a")


def test_one_candidate_list_per_set_sector_and_none_on_grey():
    frames = ROOM + [(0, f"k{yaw}", (1, 3), yaw) for yaw in (15, 75, 195, 315)]
    block, dial = _dial(_world(frames))
    sectors = dict(dial["sectors"])
    assert set(sectors) == _union(block["components"][0]) == {0, 2, 6, 10}
    assert sectors[2] == ("k75",) and sectors[6] == ("k195",) and sectors[10] == ("k315",)
    assert sectors[0] == ("k15", "01", "00")
    for grey in set(range(12)) - set(sectors):
        assert grey not in sectors


def test_newest_wins_by_journal_order_not_keyframe_id_order():
    # "s9:10" sorts before "s9:9" as a string; the journal has it later.
    frames = ROOM + [(0, "s9:9", (2, 2), 100), (0, "s9:10", (2, 2), 100)]
    _, dial = _dial(_world(frames))
    assert dict(dial["sectors"])[3] == ("s9:10", "s9:9")
    assert dial["latest"] == {"component_reference_segment": 0, "sector": 3,
                              "keyframe_id": "s9:10"}


def test_only_drawn_stations_contribute():
    # Sixteen two-keyframe stations fill the 16-station cap; a seventeenth,
    # one-keyframe station holding the newest keyframe is not drawn. Its
    # sector is neither a candidate nor a bit of the union, though `latest`
    # (an orientation anchor, not a photo) still names it.
    frames = []
    for x in range(8):
        for y in (0, 7):
            frames += [(0, f"s{x}{y}a", (x, y), 0), (0, f"s{x}{y}b", (x, y), 200)]
    frames.append((0, "t-newest", (3, 3), 100))
    block, dial = _dial(_world(frames))
    assert block["stations_omitted"] == 1
    drawn = block["components"][0]
    assert _union(drawn) == {0, 6}
    assert set(dict(dial["sectors"])) == {0, 6}
    kids = {kid for _, kids in dial["sectors"] for kid in kids}
    assert "t-newest" not in kids and len(kids) == 32
    assert dict(dial["sectors"])[6][0] == "s77b"
    assert dial["latest"] == {"component_reference_segment": 0, "sector": 3,
                              "keyframe_id": "t-newest"}


def test_hero_is_the_newest_keyframes_component():
    frames = (ROOM + [(17, "b0", (10, 10), 0), (17, "b1", (12, 12), 0),
                      (0, "a-late", (2, 2), 90), (17, "b-newest", (11, 11), 180)])
    block, dial = _dial(_world(frames))
    assert [c["reference_segment"] for c in block["components"]] == [0, 17]
    assert dial["reference_segment"] == 17
    assert dial["latest"]["component_reference_segment"] == 17
    assert all(kid.startswith("b") for _, kids in dial["sectors"] for kid in kids)


def _bins(ref, entries):
    bins = {}
    component, stations = coverage._component(ref, entries, bins_out=bins)
    component["stations"] = sorted(stations, key=lambda s: (s["y"], s["x"]))
    return component, bins


def test_degenerate_forward_is_skipped_for_latest_and_sectors():
    entries = [("a", [0.0, 0.0, 0.0], UP, _ray(0)), ("b", [0.0, 4.0, 0.0], UP, _ray(0)),
               ("c", [0.0, 2.0, 0.0], UP, _ray(100)), ("d", [0.0, 2.0, 0.0], UP, list(UP))]
    eligible = {0: list(entries)}
    component, bins = _bins(0, eligible[0])
    assert bins["d"][1] is None
    dial = coverage._dial([component], eligible, bins, ["a", "b", "c", "d"], lambda: None)
    assert dial["latest"] == {"component_reference_segment": 0, "sector": 3, "keyframe_id": "c"}
    assert "d" not in {kid for _, kids in dial["sectors"] for kid in kids}


def test_no_usable_forward_anywhere_has_no_latest():
    entries = [("a", [0.0, 0.0, 0.0], UP, _ray(0)), ("b", [0.0, 4.0, 0.0], UP, _ray(0))]
    eligible = {0: list(entries)}
    component, bins = _bins(0, eligible[0])
    bins = {kid: (cell, None) for kid, (cell, _) in bins.items()}
    component["stations"] = [{**s, "weak_mask": 0, "supported_mask": 0}
                             for s in component["stations"]]
    dial = coverage._dial([component], eligible, bins, ["a", "b"], lambda: None)
    assert dial["latest"] is None and dial["sectors"] == ()
    assert dial["reference_segment"] == 0


@pytest.mark.parametrize("step, expected", [(0.13, 3), (0.12, None), (-0.13, 9)])
def test_motion_threshold_is_a_quarter_cell_over_the_newest_24(step, expected):
    # cell_size 0.5 m, so the threshold is 0.125 m: 24 frames walking along +x
    # (the right of +y) by `step` in total. An older far frame, the 25th
    # newest, is ignored (were it counted, 0.12 m would not be null).
    walk = [(0, f"w{i:02d}", (2 + step*i/23, 2), 0) for i in range(24)]
    frames = ROOM + [(0, "old-far", (0, 4), 0)] + walk
    _, dial = _dial(_world(frames))
    assert coverage._DIAL_MOTION_KEYFRAMES == 24
    assert coverage._DIAL_MOTION_MIN_CELLS == 0.25
    assert dial["motion_sector"] == expected


def test_motion_window_reaches_past_the_newest_eight():
    # Walked 1 m right over 16 frames, then stood for 8: the 8-frame window
    # of S3 v1 saw nothing; the ~5 s window still reports the walk.
    walk = [(0, f"w{i:02d}", (1 + min(i, 15)/15, 2), 0) for i in range(24)]
    _, dial = _dial(_world(ROOM + walk))
    assert dial["motion_sector"] == 3


def test_motion_is_scale_free():
    # The same walk at 10x the scale gives the same answer: the threshold is
    # a fraction of the component's own cell, never metres.
    def run(scale):
        room = [(0, "00", (0, 0), 0), (0, "01", (4*scale, 4*scale), 0)]
        walk = [(0, f"w{i:02d}", ((2 + 0.13*i/23)*scale, 2*scale), 0) for i in range(24)]
        return _dial(_world(room + walk))[1]["motion_sector"]
    assert run(1) == run(10) == run(0.1) == 3


def test_motion_needs_two_keyframes_and_is_null_when_standing():
    _, dial = _dial(_world(ROOM + [(0, f"s{i:02d}", (2, 2), 15*i) for i in range(24)]))
    assert dial["motion_sector"] is None
    entries = [("a", [0.0, 0.0, 0.0], UP, _ray(0))]
    component = {"reference_segment": 0, "stations": [], "forward_xyz": [0.0, 1.0, 0.0],
                 "up_xyz": UP, "cell_size": 0.5}
    bins = {"a": ((0, 0), 0)}
    assert coverage._dial([component], {0: entries}, bins, ["a"], lambda: None)["motion_sector"] is None


def test_motion_forward_is_sector_zero():
    walk = [(0, f"w{i:02d}", (2, 1 + i/23), 0) for i in range(24)]
    _, dial = _dial(_world(ROOM + walk))
    assert dial["motion_sector"] == 0


def test_receipt_is_byte_identical_with_or_without_dial():
    for data in (_inputs(), _inputs(((0, 82), (17, 19)), unposed_in_horizon=12, pending=14),
                 _inputs(tuple((n, 4) for n in range(17))),
                 _world(ROOM + [(0, f"k{i}", (i % 4, i // 4), 37*i) for i in range(16)])):
        plain = coverage.compute_coverage(**json.loads(json.dumps(data)))
        eligible = []
        with_dial, dial = _dial(json.loads(json.dumps(data)))
        assert _bytes(plain) == _bytes(with_dial)
        coverage.compute_coverage(**json.loads(json.dumps(data)), eligible_out=eligible)
        drawn = {c["reference_segment"] for c in with_dial["components"]}
        assert dial["reference_segment"] in drawn
        assert {ref for _, ref in eligible} == drawn


def test_a_failing_dial_never_costs_the_coverage_receipt(monkeypatch):
    data = _inputs()
    plain = coverage.compute_coverage(**json.loads(json.dumps(data)))

    def boom(*args, **kwargs):
        raise RuntimeError("dial bug")

    monkeypatch.setattr(coverage, "_dial", boom)
    dial = {"stale": True}
    with_dial = coverage.compute_coverage(**json.loads(json.dumps(data)), dial_out=dial)
    assert _bytes(plain) == _bytes(with_dial)
    assert dial == {}


def test_a_dial_cannot_swallow_a_cancellation(monkeypatch):
    data = _inputs()
    cancelled = threading.Event()

    def cancel_then_fail(*args, **kwargs):
        cancelled.set()
        raise ValueError("coverage worker cancelled")

    monkeypatch.setattr(coverage, "_dial", cancel_then_fail)
    budget = coverage._Budget(cancelled)
    with pytest.raises(ValueError, match="cancelled"):
        coverage.compute_coverage(**json.loads(json.dumps(data)), dial_out={}, check=budget.check)


# -- imagery (B2) --------------------------------------------------------------


class _Store:
    redaction = "faces-blurred"

    def __init__(self, root):
        pass

    def read_session(self, world, session):
        return type("Session", (), {"world_id": world})()

    def read_keyframes(self, world, session):
        return []

    def read_derived(self, world, session, verify=False):
        return {"poses": []}

    def keyframe_image_set(self, world, session):
        return type("ImageSet", (), {"redaction": self.redaction})()


def _empty_live(monkeypatch):
    """The route always builds B1 too: let it reach `_Store`, whose empty
    journal makes the `live` block empty, so these tests see B2 alone."""
    monkeypatch.setattr(imagery, "contained_world_id", lambda store, world: world)
    monkeypatch.setattr(imagery, "contained_session_id", lambda store, world, session: session)


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(imagery, "WorldStore", _Store)
    monkeypatch.setattr(imagery, "label_is_trusted", lambda label: label == "faces-blurred")
    return imagery.ImageryState()


RECEIPT = {"solved_at": 1791240313.0, "geometry_revision": "g-1", "frame_revision": 1}


def _tiles(monkeypatch, missing=()):
    reads = []

    def tile(store, world, session, kid, image_set):
        reads.append(kid)
        return None if kid in missing else hashlib.sha256(kid.encode()).digest()*8
    monkeypatch.setattr(imagery, "_tile", tile)
    return reads


def _dial_block(sectors, latest=None, motion=None, ref=17):
    return {"reference_segment": ref, "sectors": tuple(sectors),
            "latest": latest or {"component_reference_segment": ref, "sector": 0,
                                 "keyframe_id": "x"},
            "motion_sector": motion}


def test_b2_one_tile_per_set_sector_with_its_sector(state, monkeypatch):
    _tiles(monkeypatch)
    dial = _dial_block([(1, ("n1", "o1")), (4, ("n4",)), (11, ("n11", "o11"))],
                       latest={"component_reference_segment": 17, "sector": 4,
                               "keyframe_id": "n4"}, motion=5)
    state.publish_landed("root", "home", "s9", RECEIPT, [], dial=dial)
    landed = state.landed("home", "s9")
    assert [(e["sector"], e["keyframe_id"]) for e in landed["entries"]] == [
        (1, "n1"), (4, "n4"), (11, "n11")]
    assert all(e["component_reference_segment"] == 17 and e["segment_index"] is None and
               e["pose_source"] == "landed_global_solve" and e["redaction"] == "redacted"
               for e in landed["entries"])
    assert landed["latest"] == {"component_reference_segment": 17, "sector": 4,
                                "keyframe_id": "n4"}
    assert landed["motion_sector"] == 5
    assert {k: landed[k] for k in RECEIPT} == RECEIPT
    for entry in landed["entries"]:
        blob = state.tile("home", "s9", entry["digest"], False, True)
        assert hashlib.sha256(blob).hexdigest() == entry["digest"]


def test_b2_missing_tile_falls_back_to_next_newest_three_at_most(state, monkeypatch):
    reads = _tiles(monkeypatch, missing={"n1", "n2", "o2", "p2"})
    dial = _dial_block([(1, ("n1", "o1")), (2, ("n2", "o2", "p2", "q2"))])
    state.publish_landed("root", "home", "s9", RECEIPT, [], dial=dial)
    landed = state.landed("home", "s9")
    assert [(e["sector"], e["keyframe_id"]) for e in landed["entries"]] == [(1, "o1")]
    assert reads == ["n1", "o1", "n2", "o2", "p2"]  # q2 is never read


def test_b2_at_most_twelve_tiles_and_null_motion(state, monkeypatch):
    _tiles(monkeypatch)
    dial = _dial_block([(s, (f"k{s}",)) for s in range(12)] + [(12, ("extra",))])
    state.publish_landed("root", "home", "s9", RECEIPT, [], dial=dial)
    landed = state.landed("home", "s9")
    assert len(landed["entries"]) == 12
    assert [e["sector"] for e in landed["entries"]] == list(range(12))
    assert landed["motion_sector"] is None


def test_b2_no_tile_keeps_previous_landing(state, monkeypatch):
    _tiles(monkeypatch, missing={"gone"})
    state.publish_landed("root", "home", "s9", RECEIPT, [], dial=_dial_block([(0, ("a",))]))
    previous = state.landed("home", "s9")
    newer = {**RECEIPT, "solved_at": RECEIPT["solved_at"] + 1}
    state.publish_landed("root", "home", "s9", newer, [], dial=_dial_block([(0, ("gone",))]))
    assert state.landed("home", "s9") == previous
    state.publish_landed("root", "home", "s9", newer, [], dial={})
    assert state.landed("home", "s9") == previous


def test_b2_dial_landing_keeps_the_80_ms_budget(state, monkeypatch):
    _tiles(monkeypatch)
    state.publish_landed("root", "home", "s9", RECEIPT, [], dial=_dial_block([(0, ("a",))]))
    previous = state.landed("home", "s9")
    ticks = iter([0.0, 0.0, 0.0, .081])
    # The budget is the job's CPU time (thread_time), not wall time.
    monkeypatch.setattr(imagery.time, "thread_time", lambda: next(ticks, .081))
    state.publish_landed("root", "home", "s9", {**RECEIPT, "solved_at": 2e9}, [],
                         dial=_dial_block([(0, ("b",)), (1, ("c",))]))
    assert state.landed("home", "s9") == previous


def test_b2_without_dial_is_the_pre_s3_block(state, monkeypatch):
    _tiles(monkeypatch)
    state.publish_landed("root", "home", "s9", RECEIPT, [(f"s9:{i}", 0) for i in range(40)])
    landed = state.landed("home", "s9")
    assert set(landed) == {"solved_at", "geometry_revision", "frame_revision", "entries"}
    assert all("sector" not in e for e in landed["entries"])


def test_b2_dial_wire_through_the_route(state, monkeypatch):
    _tiles(monkeypatch)
    monkeypatch.setattr(route, "store_from_root", _Store)
    monkeypatch.setattr(route, "contained_world_id", lambda store, world: world)
    monkeypatch.setattr(route, "contained_session_id", lambda store, world, session: session)
    monkeypatch.setattr(route, "label_is_trusted", lambda label: label == "faces-blurred")
    monkeypatch.setattr(route, "state_for", lambda root: state)
    _empty_live(monkeypatch)
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    state.publish_landed("root", "home", "s9", RECEIPT, [],
                         dial=_dial_block([(3, ("k3",))], motion=None))
    app = FastAPI()
    app.state.world_root = "synthetic-root"
    app.include_router(route.router)
    payload = TestClient(app).get("/worlds/home/guidance/s9/imagery/manifest").json()
    assert payload["version"] == 1
    assert payload["landed"]["entries"][0]["sector"] == 3
    assert payload["landed"]["latest"] == {"component_reference_segment": 17, "sector": 0,
                                           "keyframe_id": "x"}
    assert payload["landed"]["motion_sector"] is None


# -- worker --------------------------------------------------------------------


class _Sink:
    def __init__(self):
        self.published = []

    def begin_landed(self, *args):
        pass

    def finish_landed(self, *args):
        pass

    def publish_landed(self, *args, **kwargs):
        self.published.append((args, kwargs))


def _wait(predicate):
    for _ in range(300):
        if predicate():
            return True
        threading.Event().wait(.01)
    return False


def test_worker_hands_the_receipts_own_dial_to_b2(monkeypatch, tmp_path):
    sink = _Sink()
    monkeypatch.setattr(imagery, "state_for", lambda root: sink)
    data = _world(ROOM + [(0, "a", (2, 2), 95), (0, "b", (2, 2), 95)])
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, data)
    solution = root / "worlds" / world / "solve" / session / "solution.json"
    stat = solution.stat()
    worker = coverage.CoverageWorker(root, clock=lambda: 1000.1)
    try:
        worker.offer(world, session, 1000.0, 4, (stat.st_size, stat.st_mtime_ns), None,
                     (950.0, data["manifest"]["input_digest"]))
        assert _wait(lambda: sink.published)
        (args, kwargs), = sink.published
        dial = kwargs["dial"]
        assert dial["reference_segment"] == 0
        assert dict(dial["sectors"])[3] == ("b", "a")
        assert dial["latest"]["keyframe_id"] == "b"
        assert "dial" not in json.dumps(dict(worker.latest(world, session)))
    finally:
        worker.shutdown()
        worker._thread.join(1)


# -- the phone's decoder (ios/fow-dial 7ea559c, WorldImagery.swift) -----------
#
# WorldImageryManifest(json:) refuses the whole manifest for: `entries` outside
# 1...16; a `latest` missing a field, with an empty keyframe_id or a sector
# outside 0...11; a `motion_sector` outside 0...11. WorldCoverageReader.integer
# refuses a JSON bool. The overlay is used only when the landing triple equals
# the current coverage receipt's; an entry without `sector` falls back to the
# S2 heading join.


def _strict_int(value, low, high):
    return type(value) is int and low <= value <= high


def _phone_accepts(landed, coverage_block):
    """The S3 phone contract, for one published B2 block and its receipt."""
    assert 1 <= len(landed["entries"]) <= 16
    for key in ("solved_at", "geometry_revision", "frame_revision"):
        assert landed[key] == coverage_block[key]
        assert type(landed[key]) is type(coverage_block[key])
    drawn = {c["reference_segment"]: c for c in coverage_block["components"]}
    latest = landed["latest"]
    if latest is not None:
        assert set(latest) == {"component_reference_segment", "sector", "keyframe_id"}
        assert _strict_int(latest["sector"], 0, 11)
        assert isinstance(latest["keyframe_id"], str) and latest["keyframe_id"]
        assert latest["component_reference_segment"] in drawn
    motion = landed["motion_sector"]
    assert motion is None or _strict_int(motion, 0, 11)
    heroes = {e["component_reference_segment"] for e in landed["entries"]}
    assert len(heroes) == 1
    hero = heroes.pop()
    assert hero in drawn
    assert latest is None or latest["component_reference_segment"] == hero
    union = _union(drawn[hero])
    for entry in landed["entries"]:
        assert _strict_int(entry["sector"], 0, 11)  # always present: no S2 fallback
        assert entry["sector"] in union  # never on grey
        assert isinstance(entry["keyframe_id"], str) and entry["keyframe_id"]


def _jpeg_tile(kid):
    import cv2
    import numpy as np
    image = np.full((90, 160, 3), sum(kid.encode()) % 256, dtype=np.uint8)
    ok, data = cv2.imencode(".jpg", image)
    assert ok
    return data.tobytes()


def _real_imagery_state(monkeypatch):
    state = imagery.ImageryState()
    monkeypatch.setattr(imagery, "state_for", lambda root: state)
    monkeypatch.setattr(imagery, "WorldStore", _Store)
    monkeypatch.setattr(imagery, "label_is_trusted", lambda label: label == "faces-blurred")
    monkeypatch.setattr(imagery, "_tile",
                        lambda store, world, session, kid, image_set: _jpeg_tile(kid))
    return state


def test_phone_contract_end_to_end_worker_state_and_route(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    state = _real_imagery_state(monkeypatch)
    frames = ROOM + [(0, f"k{i:02d}", (1 + i % 3, 1 + i // 3 % 3), 37*i) for i in range(30)]
    data = _world(frames)
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, data)
    stat = (root / "worlds" / world / "solve" / session / "solution.json").stat()
    worker = coverage.CoverageWorker(root, clock=lambda: 1000.1)
    try:
        worker.offer(world, session, 1000.0, len(frames), (stat.st_size, stat.st_mtime_ns), None,
                     (950.0, data["manifest"]["input_digest"]))
        assert _wait(lambda: state.landed(world, session) is not None)
        receipt = json.loads(json.dumps(worker.latest(world, session)))
    finally:
        worker.shutdown()
        worker._thread.join(1)
    for name, value in (("store_from_root", _Store), ("state_for", lambda root: state),
                        ("contained_world_id", lambda store, w: w),
                        ("contained_session_id", lambda store, w, s: s),
                        ("label_is_trusted", lambda label: label == "faces-blurred")):
        monkeypatch.setattr(route, name, value)
    _empty_live(monkeypatch)
    app = FastAPI()
    app.state.world_root = str(root)
    app.include_router(route.router)
    client = TestClient(app)
    response = client.get(f"/worlds/{world}/guidance/{session}/imagery/manifest")
    assert response.status_code == 200
    landed = json.loads(response.text)["landed"]
    assert '"entries":[]' not in json.dumps(landed, separators=(",", ":"))
    _phone_accepts(landed, receipt)
    assert len(landed["entries"]) == len(_union(receipt["components"][0]))  # one per set sector
    for entry in landed["entries"]:
        tile = client.get(f"/worlds/{world}/guidance/{session}/imagery/tile/{entry['digest']}")
        assert tile.status_code == 200
        assert hashlib.sha256(tile.content).hexdigest() == entry["digest"]


def test_no_tile_landing_keeps_previous_or_null_never_an_empty_list(state, monkeypatch):
    _tiles(monkeypatch, missing={"gone"})
    state.publish_landed("root", "home", "s9", RECEIPT, [], dial=_dial_block([(0, ("gone",))]))
    assert state.landed("home", "s9") is None  # the route serves `landed: null`
    state.publish_landed("root", "home", "s9", RECEIPT, [], dial=_dial_block([(0, ("a",))]))
    previous = state.landed("home", "s9")
    assert previous["entries"]
    for dial in ({}, _dial_block([]), _dial_block([(0, ("gone",))])):
        state.publish_landed("root", "home", "s9", {**RECEIPT, "solved_at": 2e9}, [], dial=dial)
        assert state.landed("home", "s9") == previous


def test_latest_never_names_a_component_the_station_cap_dropped():
    # Ref 0 fills all 16 drawn stations. Ref 17 holds the newest keyframes,
    # with usable headings, but its one-keyframe stations lose the cap, so
    # ref 17 is not a drawn piece: latest and H come from ref 0 instead.
    frames = []
    for x in range(8):
        for y in (0, 7):
            frames += [(0, f"s{x}{y}a", (x, y), 0), (0, f"s{x}{y}b", (x, y), 200)]
    frames += [(17, "z0", (20, 20), 90), (17, "z1", (24, 24), 90)]
    block, dial = _dial(_world(frames))
    assert [c["reference_segment"] for c in block["components"]] == [0]
    assert block["components_omitted"] == 1
    assert dial["reference_segment"] == 0
    assert dial["latest"]["component_reference_segment"] == 0
    assert dial["latest"]["keyframe_id"] == "s77b"


def test_sector_bins_rebuild_the_coverage_masks_exactly(monkeypatch):
    # Headings on exact 30-degree boundaries and between, across many
    # stations: counting the dial's own bins per drawn station reproduces
    # each station's weak and supported masks bit for bit.
    captured = {}
    original = coverage._dial

    def capture(components, eligible, bins, ids, check):
        captured.update(eligible=eligible, bins=dict(bins))
        return original(components, eligible, bins, ids, check)

    monkeypatch.setattr(coverage, "_dial", capture)
    frames = ROOM + [(0, f"k{i:03d}", (i % 5, i // 5 % 5), (15*i) % 360) for i in range(120)]
    block, dial = _dial(_world(frames))
    component = block["components"][0]
    counts = {}
    for kid, _, _, _ in captured["eligible"][0]:
        cell, sector = captured["bins"][kid]
        if sector is not None:
            assert type(sector) is int and 0 <= sector <= 11
            counts.setdefault(cell, [0]*12)[sector] += 1
    assert len(component["stations"]) > 4
    for station in component["stations"]:
        per = counts.get((station["y"], station["x"]), [0]*12)
        assert station["weak_mask"] == sum(1 << s for s, n in enumerate(per) if n == 1)
        assert station["supported_mask"] == sum(1 << s for s, n in enumerate(per) if n >= 2)
    for sector, kids in dial["sectors"]:
        assert all(captured["bins"][kid][1] == sector for kid in kids)


def test_latest_requires_a_non_empty_string_id():
    entries = [("a", [0.0, 0.0, 0.0], UP, _ray(0)), ("b", [0.0, 4.0, 0.0], UP, _ray(0)),
               ("", [0.0, 2.0, 0.0], UP, _ray(100))]
    eligible = {0: list(entries)}
    component, bins = _bins(0, eligible[0])
    dial = coverage._dial([component], eligible, bins, ["a", "b", ""], lambda: None)
    assert dial["latest"]["keyframe_id"] == "b"


def test_failed_dial_never_publishes_sectorless_entries(monkeypatch, tmp_path):
    state = _real_imagery_state(monkeypatch)

    def boom(*args):
        raise RuntimeError("dial bug")

    monkeypatch.setattr(coverage, "_dial", boom)
    data = _world(ROOM + [(0, "a", (2, 2), 95), (0, "b", (2, 2), 95)])
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, data)
    stat = (root / "worlds" / world / "solve" / session / "solution.json").stat()
    worker = coverage.CoverageWorker(root, clock=lambda: 1000.1)
    try:
        worker.offer(world, session, 1000.0, 4, (stat.st_size, stat.st_mtime_ns), None,
                     (950.0, data["manifest"]["input_digest"]))
        assert _wait(lambda: worker.latest(world, session) is not None)
        threading.Event().wait(.2)
        assert state.landed(world, session) is None  # never the stride's sector-less entries
    finally:
        worker.shutdown()
        worker._thread.join(1)
