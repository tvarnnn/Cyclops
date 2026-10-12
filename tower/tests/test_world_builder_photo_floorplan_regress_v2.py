"""Photo floor plan never-regress guard V2 (TOWER_WORLD_PHOTO_FLOORPLAN_REGRESS_V2=on).

Brief CX-GEN-B-REGRESS (audit item 5): compare supported floor on the COMMON footprint of
the published and the new landing, in a gauge-independent way; record extent changes; a
4th regressed landing publishes only through a conditional escape or the freeze bound.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tower import config
from tower.world_builder import photo_floorplan_worker as pfw
from tower.world_builder.photo_floorplan import padded_square_bounds
from tests.test_world_builder_photo_floorplan_worker import (
    SESSION, WORLD, _fake_build_factory, _receipt, _synthetic_solve, _wait)

GOLDEN = Path(__file__).parent / "golden" / "fow_v12_regress_v2"
H = 1.5   # canonical camera height above the floor


# ---------------------------------------------------------------------------
# a synthetic landing: one canonical room, seen through an arbitrary solve gauge
# ---------------------------------------------------------------------------


def _rotation(yaw_deg, tilt_deg):
    a, b = math.radians(yaw_deg), math.radians(tilt_deg)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return rx @ rz


def _grid_cams(x0, x1, y0, y1, n, prefix="k"):
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    return {f"{prefix}{i}_{j}": (x, y) for i, x in enumerate(xs) for j, y in enumerate(ys)}


def _landing(cams, paint, *, scale=1.0, yaw=0.0, tilt=0.0, shift=(0.0, 0.0, 0.0), size=256,
             projected=40, source="estimated", plane_ok=True, axes=True):
    """(stats row, Footprint) of a landing whose cameras are `cams` {kid: (x, y)} at height H
    and whose raster paints canonical floor where `paint(x, y)` is true."""
    R = _rotation(yaw, tilt)
    t = np.asarray(shift, np.float64)

    def gauge(p):
        return scale * (R @ np.asarray(p, np.float64)) + t

    positions = {k: gauge((x, y, H)) for k, (x, y) in cams.items()}
    up, forward = R @ np.array([0.0, 0.0, 1.0]), R @ np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up)
    pts = np.stack(list(positions.values()))
    bx, by = pts @ forward, pts @ right
    bounds = padded_square_bounds((bx.min(), by.min(), bx.max(), by.max()))
    x0, y0, x1, y1 = bounds
    cols, rows = np.meshgrid(np.arange(size), np.arange(size))
    x = x0 + cols * (x1 - x0) / (size - 1)
    y = y0 + rows * (y1 - y0) / (size - 1)
    floor_z = float(up @ t)
    world = (x[..., None] * forward + y[..., None] * right + floor_z * up - t) @ R / scale
    opaque = paint(world[..., 0], world[..., 1]) & plane_ok
    extra = {}
    if axes:   # per-camera (up, forward), looking around and 30 deg down, in this gauge
        extra["axes"] = {}
        for i, k in enumerate(cams):
            a = math.radians(37.0 * i)
            f = np.array([math.cos(a), math.sin(a), -0.6])
            f /= np.linalg.norm(f)
            u = np.array([0.0, 0.0, 1.0]) - f[2] * f
            extra["axes"][k] = (R @ (u / np.linalg.norm(u)), R @ f)
    fp = pfw.Footprint(**extra, positions=positions, up_xyz=tuple(up), forward_xyz=tuple(forward),
                       raster_bounds_xy=tuple(bounds), opaque=opaque,
                       plane_normal=tuple(up) if plane_ok else None,
                       plane_offset=-floor_z if plane_ok else None,
                       camera_height=scale * H if plane_ok else None, origin_xyz=tuple(t))
    fraction = float(opaque.mean())
    row = {"reference_segment": 0, "opaque_fraction": fraction,
           "opaque_area": fraction * (x1 - x0) * (y1 - y0), "projected_keyframes": projected,
           "plane_ok": plane_ok, "plane": {"source": None if source == "estimated" else source}}
    return row, fp


def _room(x0, x1, y0, y1):
    return lambda x, y: (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


ROOM = _grid_cams(0.5, 3.5, 0.5, 3.5, 6)                 # 36 keyframes in a 4 x 4 room
FULL = _room(0.0, 4.0, 0.0, 4.0)
NEW_GAUGE = dict(scale=0.3, yaw=70.0, tilt=25.0, shift=(4.0, -2.0, 7.0))


def _assess(prev, new):
    (prev_row, prev_fp), (new_row, new_fp) = prev, new
    return pfw.regress_v2_assess({0: prev_row["opaque_fraction"]}, {0: prev_row["opaque_area"]},
                                 {0: prev_fp}, [new_row], {0: new_fp})[0]


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "off", "1", "true", "yes", "ON", " on", "on "])
def test_only_exactly_on_enables_v2(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(config.WORLD_PHOTO_FLOORPLAN_REGRESS_V2_ENV, raising=False)
    else:
        monkeypatch.setenv(config.WORLD_PHOTO_FLOORPLAN_REGRESS_V2_ENV, value)
    assert config.world_photo_floorplan_regress_v2_setting() is False
    worker = pfw.PhotoFloorplanWorker("synthetic", method="homography",
                                      build=_fake_build_factory())
    try:
        assert worker.regress_v2 is False
    finally:
        worker.shutdown()
    monkeypatch.setenv(config.WORLD_PHOTO_FLOORPLAN_REGRESS_V2_ENV, "on")
    assert config.world_photo_floorplan_regress_v2_setting() is True


# ---------------------------------------------------------------------------
# the common-footprint comparison
# ---------------------------------------------------------------------------


def test_a_gauge_change_alone_is_not_a_regression():
    # e2e6 12:55:35 -> 12:56:11: the same room, opaque area 1,019 -> 146 in solve units.
    prev = _landing(ROOM, FULL)
    new = _landing(ROOM, FULL, **NEW_GAUGE)
    assert new[0]["opaque_area"] < 0.1 * prev[0]["opaque_area"]      # solve units mislead
    rec = _assess(prev, new)
    assert rec["basis"] == "common" and not rec["regressed"]
    assert rec["scale"] == pytest.approx(0.3, rel=1e-6)
    assert rec["retained"] > 0.99 and rec["common_share"] > 0.99
    assert rec["net"] == pytest.approx(1.0, abs=0.05)                 # gauge-free area
    assert not rec["extent_change"]


def test_a_true_degradation_is_refused_where_todays_rule_lets_it_through():
    prev = _landing(ROOM, FULL)
    new = _landing(ROOM, _room(0.0, 1.2, 0.0, 4.0), **NEW_GAUGE)    # 30% of the same floor
    rec = _assess(prev, new)
    assert rec["basis"] == "common" and rec["regressed"]
    assert rec["retained"] < 0.4 and "common footprint" in rec["why"]
    # Today's rule: 0.30 x the fraction is above 0.25 x, so it would have published.
    assert pfw.regression({0: prev[0]["opaque_fraction"]}, [new[0]],
                          {0: prev[0]["opaque_area"]}) == ""


def test_an_extent_growth_publishes_and_is_recorded():
    # The walk reaches a second room; the refit paints less of the first one.
    hall = _grid_cams(6.0, 11.0, 0.5, 3.5, 5, prefix="b")
    prev = _landing(ROOM, FULL)
    new = _landing({**ROOM, **hall},
                   lambda x, y: _room(0.0, 1.2, 0.0, 4.0)(x, y) | _room(5.5, 11.5, 0.0, 4.0)(x, y),
                   **NEW_GAUGE)
    rec = _assess(prev, new)
    assert rec["retained"] < pfw.REGRESS_V2_MIN_RETAINED            # would be a regression...
    assert rec["footprint_grew"] and rec["growth"] >= pfw.REGRESS_V2_EXTENT_GROWTH
    assert rec["novel_keyframes"] >= pfw.REGRESS_V2_EXTENT_MIN_NOVEL
    assert rec["net"] >= 1.0
    assert rec["extent_change"] and not rec["regressed"]              # ...but the extent grew


def test_growth_without_supported_floor_is_not_an_extent_change():
    hall = _grid_cams(6.0, 11.0, 0.5, 3.5, 5, prefix="b")
    prev = _landing(ROOM, FULL)
    new = _landing({**ROOM, **hall}, _room(0.0, 1.2, 0.0, 4.0), **NEW_GAUGE)
    rec = _assess(prev, new)
    assert rec["footprint_grew"] and rec["net"] < 1.0
    assert not rec["extent_change"] and rec["regressed"]


def test_e2e4_near_blank_landing_is_refused():
    # e2e4 10:37:47: a few runaway cameras blew the square up ~70x in area; the fallback
    # plane painted 0.4% of it, almost none on the room the phone already showed.
    runaway = {f"r{i}": (40.0 + 3.0 * i, 30.0 - 2.0 * i) for i in range(6)}
    prev = _landing(ROOM, _room(0.0, 4.0, 0.0, 2.2))                  # ~55% of its square
    new = _landing({**ROOM, **runaway}, _room(20.0, 26.0, 10.0, 14.0), projected=53,
                   source="fallback_camera_drop", **NEW_GAUGE)
    assert new[0]["opaque_fraction"] < 0.01
    rec = _assess(prev, new)
    assert rec["basis"] == "common" and rec["regressed"] and rec["retained"] < 0.05
    assert not rec["extent_change"]
    # With no plane at all (today's fallback refusal), it falls back to the legacy rule.
    blank = _landing({**ROOM, **runaway}, FULL, plane_ok=False, **NEW_GAUGE)
    rec = _assess(prev, blank)
    assert rec["basis"] == "legacy: no camera height" and rec["regressed"]


def test_e2e5_kf520_real_landing_publishes():
    # e2e5 11:14:34, horizon 520 vs the published 313-kf landing: fraction 0.194 < 0.25 x
    # 0.799 was skipped live. Rebuilt offline from the two landings' own live poses with
    # the worker's homography (replay in Glasses-scratch/gen-b-regress-20261012).
    z = np.load(GOLDEN / "e2e5_kf520_ref0.npz")

    def load(tag):
        shape = tuple(z[f"{tag}_shape"])
        up, forward, origin = z[f"{tag}_basis"]
        plane = z[f"{tag}_plane"]
        fp = pfw.Footprint(
            positions=dict(zip(z[f"{tag}_kids"].tolist(), z[f"{tag}_positions"])),
            up_xyz=tuple(up), forward_xyz=tuple(forward),
            raster_bounds_xy=tuple(z[f"{tag}_bounds"]),
            opaque=np.unpackbits(z[f"{tag}_opaque"])[:shape[0] * shape[1]].reshape(shape) > 0,
            plane_normal=tuple(plane[:3]), plane_offset=float(plane[3]),
            camera_height=float(plane[4]), origin_xyz=tuple(origin),
            axes={k: tuple(a) for k, a in zip(z[f"{tag}_kids"].tolist(), z[f"{tag}_axes"])})
        row = dict(json.loads(str(z[f"{tag}_stats"])), reference_segment=0)
        return row, fp

    prev, new = load("prev"), load("new")
    assert new[0]["opaque_fraction"] < 0.25 * prev[0]["opaque_fraction"]   # today: skipped
    rec = _assess(prev, new)
    assert rec["basis"] == "common" and not rec["regressed"]
    assert rec["rotation"] == "orientations"
    assert rec["agree"] > 0.85 and rec["resid_h"] < 0.05
    assert rec["retained"] >= pfw.REGRESS_V2_MIN_RETAINED
    assert rec["extent_change"] and rec["growth"] > 5 and rec["net"] > 1.0


def test_too_few_agreeing_keyframes_fall_back_to_the_legacy_rule():
    prev = _landing(ROOM, FULL)
    few = dict(list(ROOM.items())[:6])
    new = _landing(few, FULL, **NEW_GAUGE)
    rec = _assess(prev, new)
    assert rec["basis"].startswith("legacy: 6 shared keyframes")
    assert not rec["regressed"]       # legacy: 1.0 vs 0.something -- no fraction drop


def test_a_bent_trajectory_is_compared_on_the_part_both_solves_agree_on():
    prev = _landing(ROOM, FULL)
    new_row, new_fp = _landing(ROOM, FULL, **NEW_GAUGE)
    bent = dict(new_fp.positions)
    for k in list(bent)[:8]:               # a new loop moved 8 of 36 keyframes by ~2 m
        bent[k] = bent[k] + 0.3 * 2.0 * np.asarray(new_fp.forward_xyz)
    new_fp.positions = bent
    rec = _assess(prev, (new_row, new_fp))
    assert rec["basis"] == "common" and 0.7 < rec["agree"] < 0.8
    assert rec["resid_h"] < 0.01 and rec["retained"] > 0.95 and not rec["regressed"]


# ---------------------------------------------------------------------------
# the worker: decisions, escape, freeze bound, logging
# ---------------------------------------------------------------------------


def _v2_build(plan):
    def build(root, world, session, receipt, eligible, tree, stat, method, check,
              footprints=None):
        block, tiles, _ = _fake_build_factory()(root, world, session, receipt, eligible, tree,
                                                stat, method=method, check=check)
        row, fp = plan[receipt["solved_at"]]
        if footprints is not None:
            footprints[0] = fp
        return block, tiles, [dict(row)]
    return build


def _run(plan, *, clock_at=None, regress_v2=True):
    """Offer each landing in order; returns (worker outcomes per landing, state)."""
    latest, now = {}, [0.0]
    state = pfw.PhotoFloorplanState()
    worker = pfw.PhotoFloorplanWorker("synthetic", latest=lambda w, s: latest.get((w, s)),
                                      state=state, method="homography", build=_v2_build(plan),
                                      regress_v2=regress_v2, clock=lambda: now[0])
    outcomes = []
    try:
        for i, solved_at in enumerate(sorted(plan)):
            now[0] = (clock_at or {}).get(solved_at, float(i))
            receipt = _receipt(solved_at, revision=f"g{i}")
            latest[(WORLD, SESSION)] = receipt
            worker.offer(WORLD, SESSION, receipt, [("k", 0)], {}, (1, 1))
            assert _wait(lambda: worker.outcomes.get((WORLD, SESSION), {}).get("identity")
                         == (solved_at, f"g{i}", 1))
            outcomes.append(worker.outcomes[(WORLD, SESSION)])
    finally:
        worker.shutdown()
    return outcomes, state


def test_worker_v2_publishes_the_e2e5_kind_of_landing_and_refuses_a_degradation():
    hall = _grid_cams(6.0, 11.0, 0.5, 3.5, 5, prefix="b")
    plan = {1000.0: _landing(ROOM, FULL),
            2000.0: _landing(ROOM, _room(0.0, 1.2, 0.0, 4.0), **NEW_GAUGE),       # degraded
            3000.0: _landing({**ROOM, **hall}, lambda x, y: _room(0.0, 1.2, 0.0, 4.0)(x, y)
                             | _room(5.5, 11.5, 0.0, 4.0)(x, y), **NEW_GAUGE)}  # grew
    outcomes, state = _run(plan)
    assert [o["disposition"] == "ok" for o in outcomes] == [True, False, True]
    assert outcomes[1]["disposition"].startswith("skipped: regressed (ref 0: retained")
    assert outcomes[1]["regress_v2"]["decision"] == "skip"
    assert outcomes[2]["regress_v2"]["components"][0]["extent_change"] is True
    assert state.landed(WORLD, SESSION)["solved_at"] == 3000.0


def _degraded(**kw):
    return _landing(ROOM, _room(0.0, 1.0, 0.0, 4.0), **{**NEW_GAUGE, **kw})


def test_conditional_escape_publishes_a_landing_that_meets_the_quality_floor():
    assert pfw.REGRESS_V2_ESCAPE_AFTER == 3
    plan = {1000.0: _landing(ROOM, FULL), **{t: _degraded() for t in (2000.0, 3000.0, 4000.0,
                                                                      5000.0)}}
    outcomes, state = _run(plan)
    assert [o["regress_v2"]["decision"] for o in outcomes] == [
        "publish", "skip", "skip", "skip", "escape"]
    assert outcomes[3]["disposition"].endswith("3/3)")
    assert outcomes[4]["disposition"] == "ok"
    assert state.landed(WORLD, SESSION)["solved_at"] == 5000.0


@pytest.mark.parametrize("weak", [dict(source="fallback_camera_drop"), dict(projected=5)])
def test_conditional_escape_refuses_a_weak_landing_and_reports_it_stalled(weak):
    plan = {1000.0: _landing(ROOM, FULL),
            **{t: _degraded(**weak) for t in (2000.0, 3000.0, 4000.0, 5000.0, 6000.0)}}
    outcomes, state = _run(plan)
    decisions = [o["regress_v2"]["decision"] for o in outcomes]
    assert decisions == ["publish", "skip", "skip", "skip", "stalled", "stalled"]
    assert outcomes[4]["disposition"].startswith("skipped: regressed, stalled")
    assert "escape refused" in outcomes[4]["disposition"]
    assert state.landed(WORLD, SESSION)["solved_at"] == 1000.0       # the useful plan stays


def test_the_freeze_bound_publishes_a_usable_landing_but_never_a_blank_one():
    weak = _degraded(source="fallback_camera_drop")
    blank = _landing(ROOM, _room(0.0, 0.05, 0.0, 0.05), source="fallback_camera_drop",
                     **NEW_GAUGE)
    assert blank[0]["opaque_fraction"] < pfw.REGRESS_MIN_PRIOR_OPAQUE
    plan = {1000.0: _landing(ROOM, FULL), 2000.0: blank, 3000.0: blank, 4000.0: blank,
            5000.0: blank, 6000.0: weak}
    clock_at = {1000.0: 0.0, 2000.0: 10.0, 3000.0: 60.0, 4000.0: 120.0,
                5000.0: 10.0 + pfw.REGRESS_V2_MAX_FROZEN_S + 1, 6000.0: 10.0 + 400.0}
    outcomes, state = _run(plan, clock_at=clock_at)
    decisions = [o["regress_v2"]["decision"] for o in outcomes]
    assert decisions == ["publish", "skip", "skip", "skip", "stalled", "freeze-bound"]
    assert outcomes[4]["regress_v2"]["frozen_s"] > pfw.REGRESS_V2_MAX_FROZEN_S   # blank: held
    assert outcomes[5]["disposition"] == "ok"
    assert state.landed(WORLD, SESSION)["solved_at"] == 6000.0


def test_every_v2_decision_is_logged(caplog):
    plan = {1000.0: _landing(ROOM, FULL), 2000.0: _degraded()}
    with caplog.at_level("INFO", logger="tower.world_builder.photo_floorplan_worker"):
        _run(plan)
    lines = [r.getMessage() for r in caplog.records if "regress-v2:" in r.getMessage()]
    assert len(lines) == 2
    assert "decision=publish" in lines[0] and "decision=skip" in lines[1]
    record = json.loads(lines[1].split("components=", 1)[1])[0]
    assert record["basis"] == "common" and record["regressed"] is True


# ---------------------------------------------------------------------------
# OFF identity
# ---------------------------------------------------------------------------


def test_off_never_asks_for_footprints_and_keeps_todays_decisions():
    seen = []

    def build(root, world, session, receipt, eligible, tree, stat, method, check,
              footprints=None):
        seen.append(footprints)
        return _v2_build(plan)(root, world, session, receipt, eligible, tree, stat, method,
                               check)

    # A true degradation (30% of the floor) that today's rule lets through (0.30 > 0.25).
    plan = {1000.0: _landing(ROOM, FULL),
            2000.0: _landing(ROOM, _room(0.0, 1.2, 0.0, 4.0), **NEW_GAUGE)}
    latest = {}
    state = pfw.PhotoFloorplanState()
    worker = pfw.PhotoFloorplanWorker("synthetic", latest=lambda w, s: latest.get((w, s)),
                                      state=state, method="homography", build=build,
                                      regress_v2=False)
    try:
        for i, solved_at in enumerate(sorted(plan)):
            receipt = _receipt(solved_at, revision=f"g{i}")
            latest[(WORLD, SESSION)] = receipt
            worker.offer(WORLD, SESSION, receipt, [("k", 0)], {}, (1, 1))
            assert _wait(lambda: worker.outcomes.get((WORLD, SESSION), {}).get("identity")
                         == (solved_at, f"g{i}", 1))
        outcome = worker.outcomes[(WORLD, SESSION)]
    finally:
        worker.shutdown()
    assert seen == [None, None]
    assert "regress_v2" not in outcome
    assert outcome["disposition"] == "ok"                # today's rule: published
    assert state.landed(WORLD, SESSION)["solved_at"] == 2000.0


def test_build_landing_output_is_unchanged_by_the_footprint_hook(tmp_path):
    solve_dir, tree, receipt, eligible, stat = _synthetic_solve(tmp_path)
    plain = pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree, stat)
    footprints = {}
    hooked = pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree, stat,
                               footprints=footprints)
    assert hooked[0] == plain[0] and hooked[1] == plain[1]

    def strip(stats):
        # DIAG_INPUTS (lane A diagnostics) carries live arrays the worker strips later.
        return [{k: v for k, v in c.items()
                 if k not in ("ms", pfw.DIAG_INPUTS) and not k.endswith("_ms")}
                for c in stats]

    assert strip(hooked[2]) == strip(plain[2])
    ref = receipt["components"][0]["reference_segment"]
    fp = footprints[ref]
    assert fp.camera_height and fp.plane_normal is not None
    assert int(fp.opaque.sum()) == pytest.approx(
        plain[2][0]["opaque_fraction"] * fp.opaque.size, abs=1)
    # A landing compared with itself retains everything.
    rec = pfw.compare_footprints(fp, fp)
    assert rec["basis"] == "common" and rec["retained"] == 1.0 and rec["scale"] == pytest.approx(1)


# ---------------------------------------------------------------------------
# review round (lead, b291377): HIGH-1 escape bypass, HIGH-2 degenerate Sim(3)
# ---------------------------------------------------------------------------


def test_escape_refuses_a_grown_footprint_without_supported_floor():
    # HIGH-1: hull grew (footprint_grew) but the floor shrank (net < 1, so no extent
    # change); a fallback plane over 2 projected frames must not escape.
    hall = _grid_cams(6.0, 11.0, 0.5, 3.5, 5, prefix="b")
    weak = _landing({**ROOM, **hall}, _room(0.0, 1.2, 0.0, 4.0), source="fallback_camera_drop",
                    projected=2, **NEW_GAUGE)
    prev = _landing(ROOM, FULL)
    rec = _assess(prev, weak)
    assert rec["footprint_grew"] and not rec["extent_change"] and rec["regressed"]
    plan = {1000.0: prev, **{t: weak for t in (2000.0, 3000.0, 4000.0, 5000.0)}}
    outcomes, state = _run(plan)
    assert [o["regress_v2"]["decision"] for o in outcomes] == [
        "publish", "skip", "skip", "skip", "stalled"]
    assert state.landed(WORLD, SESSION)["solved_at"] == 1000.0


def test_the_freeze_bound_needs_projected_frames_not_just_a_sliver():
    sliver = _degraded(source="fallback_camera_drop", projected=2)
    plan = {1000.0: _landing(ROOM, FULL), 2000.0: sliver, 3000.0: sliver}
    clock_at = {1000.0: 0.0, 2000.0: 10.0, 3000.0: 10.0 + pfw.REGRESS_V2_MAX_FROZEN_S + 1}
    outcomes, state = _run(plan, clock_at=clock_at)
    assert [o["regress_v2"]["decision"] for o in outcomes] == ["publish", "skip", "skip"]
    assert state.landed(WORLD, SESSION)["solved_at"] == 1000.0


HALL = {f"h{i}": (float(i), 0.0) for i in range(12)}            # exactly collinear, 11 m
STRIP = _room(0.0, 11.0, 1.5, 2.5)                               # floor 1.5-2.5 m off the axis


def test_a_collinear_hallway_without_orientations_is_not_compared():
    # HIGH-2: positions on a line leave the roll about that line free -- a perfect fit with
    # a wrong rotation would carry off-axis floor to the wrong place.
    rec = _assess(_landing(HALL, STRIP, axes=False),
                  _landing(HALL, STRIP, axes=False, **NEW_GAUGE))
    assert rec["basis"].startswith("legacy: shared cameras")
    assert not rec["regressed"]                                  # legacy: same fraction


def test_a_collinear_hallway_is_compared_through_camera_orientations():
    prev = _landing(HALL, STRIP, axes=True)
    new = _landing(HALL, STRIP, axes=True, **NEW_GAUGE)
    rec = _assess(prev, new)
    assert rec["basis"] == "common" and rec["rotation"] == "orientations"
    assert rec["scale"] == pytest.approx(0.3, rel=1e-6)
    assert rec["retained"] > 0.99 and not rec["regressed"]
    # The strip mirrored across the hallway (the wrong roll) is a real loss, and is seen.
    mirrored = _landing(HALL, _room(0.0, 11.0, -2.5, -1.5), axes=True, **NEW_GAUGE)
    rec = _assess(prev, mirrored)
    assert rec["basis"] == "common" and rec["retained"] < 0.05 and rec["regressed"]


def test_orientations_that_disagree_with_the_positions_are_refused():
    prev = _landing(ROOM, FULL, axes=True)
    row, fp = _landing(ROOM, FULL, axes=True, **NEW_GAUGE)
    flip = _rotation(0.0, 40.0)       # every camera rolled 40 deg in the new solve only
    fp.axes = {k: (flip @ u, flip @ f) for k, (u, f) in fp.axes.items()}
    rec = _assess(prev, (row, fp))
    assert rec["basis"].startswith("legacy:")


def test_cameras_that_barely_moved_give_no_scale():
    huddle = {f"q{i}": (2.0 + 0.02 * math.cos(i), 2.0 + 0.02 * math.sin(i)) for i in range(20)}
    rec = _assess(_landing(huddle, FULL, axes=True), _landing(huddle, FULL, axes=True,
                                                               **NEW_GAUGE))
    assert rec["basis"].startswith("legacy: shared cameras")


def test_a_two_dimensional_walk_without_orientations_still_compares_by_positions():
    rec = _assess(_landing(ROOM, FULL, axes=False), _landing(ROOM, FULL, axes=False, **NEW_GAUGE))
    assert rec["basis"] == "common" and rec["rotation"] == "positions"
    assert rec["retained"] > 0.99 and not rec["regressed"]
