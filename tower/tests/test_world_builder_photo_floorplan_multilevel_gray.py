"""Slice-0 gray truth correction through the real synthetic solve and publication path."""

import hashlib
import io
import time

import numpy as np
import pytest
from PIL import Image

from tower.world_builder import photo_floorplan_worker as pfw
from tower.world_builder import photo_floorplan_diagnostics as diag
from tower import config
from tests.test_world_builder_photo_floorplan_diagnostics import (
    CAM_HEIGHT, SESSION, WORLD, _floor, _grid_cams, _write_scene,
    kneel_in_one_level_room, split_level, long_hallway, large_open_room,
    low_texture_floor,
)


def _build(root, cams, points, *, gray):
    tree, receipt, eligible, stat = _write_scene(root, cams, points)
    block, tiles, stats = pfw.build_landing(
        root, WORLD, SESSION, receipt, eligible, tree, stat,
        plane_memory={}, multilevel_gray=gray,
    )
    return tree, receipt, eligible, stat, block, tiles, stats


def _alpha(tiles):
    (data, _mime), = tiles.values()
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))[:, :, 3]


def _step_scene(*, continuous=False, connector=False):
    rng = np.random.default_rng(82 if continuous else 81)
    if connector:
        xs = np.r_[np.linspace(0.4, 3.5, 7), np.linspace(3.8, 4.8, 5),
                   np.linspace(5.1, 8.4, 7)]
        def height(x):
            return 0.0 if x < 3.8 else (1.0 if x >= 5.1 else 0.2 * (1 + int((x - 3.8) / 0.26)))
    else:
        xs = np.linspace(0.4, 8.4, 18)
        def height(x):
            return x / 8.8 if continuous else 0.2 * min(5, int(x / 1.7))
    cams = _grid_cams(xs, (-1.0, 0.0, 1.0),
                      z_of=lambda x, _y: CAM_HEIGHT + height(x), seed=81)
    # Points cover each local walking surface; there is no invented second floor.
    xy = np.c_[rng.uniform(0, 8.8, 2800), rng.uniform(-2.0, 2.0, 2800)]
    points = np.c_[xy, [height(x) for x in xy[:, 0]] + rng.normal(0, 0.01, len(xy))]
    return cams, points


def _stacked_scene():
    rng = np.random.default_rng(84)
    low = _grid_cams(np.linspace(-2.2, 2.2, 5), (-1.5, 0, 1.5), seed=84)
    high = _grid_cams(np.linspace(-2.2, 2.2, 5), (-1.5, 0, 1.5),
                      z_of=lambda _x, _y: CAM_HEIGHT + 1.1, seed=85)
    points = np.vstack((_floor(rng, 1600, -3, 3, -2.5, 2.5),
                        _floor(rng, 1600, -3, 3, -2.5, 2.5, z=1.1)))
    return low + high, points


def _carry_scene():
    rng = np.random.default_rng(86)
    cams = _grid_cams(np.linspace(-2.5, 2.5, 7), np.linspace(-2.5, 2.5, 6),
                      z_of=lambda x, _y: CAM_HEIGHT + (0.5 if x > 0 else 0), seed=86)
    return cams, _floor(rng, 2500, -3.5, 3.5, -3.5, 3.5)


def _sit_scene():
    cams, points = kneel_in_one_level_room()
    return [(p if p[2] > 1.1 else (p[0], p[1], 0.75), y, pitch)
            for p, y, pitch in cams], points


SCENES = {
    "staircase": lambda: _step_scene(),
    "ramp": lambda: _step_scene(continuous=True),
    "plateaus_connector": lambda: _step_scene(connector=True),
    "stacked": _stacked_scene,
    "carry": _carry_scene,
    "kneel": kneel_in_one_level_room,
    "sit": _sit_scene,
    "hallway": long_hallway,
    "large_room": large_open_room,
    "low_texture": low_texture_floor,
    "split_level": split_level,
}


# Characterization of c3e468a. A stepped surface currently becomes one tilted
# raster, a continuous ramp becomes one plane, and stacked floors are rejected
# before the floor-confirmed diagnostic can identify two levels.
CURRENT = {
    "staircase": (set(), True, False),
    "ramp": (set(), True, False),
    "plateaus_connector": (set(), True, False),
    "stacked": ({"low_floor_support"}, False, False),
    "carry": (set(), True, False),
    "kneel": (set(), True, False),
    "sit": (set(), True, False),
    "hallway": ({"collinear_trajectory", "large_extent"}, True, False),
    "large_room": ({"large_extent"}, True, False),
    "low_texture": ({"low_floor_support"}, False, False),
    "split_level": ({"multi_level_suspected"}, True, True),
}


@pytest.mark.parametrize("name", sorted(SCENES))
def test_scene_matrix_diagnostics_raster_manifest_and_gate(tmp_path, name):
    cams, points = SCENES[name]()
    _, receipt, _, _, off, off_tiles, off_stats = _build(tmp_path, cams, points, gray=False)
    _, _, _, _, on, on_tiles, on_stats = _build(tmp_path, cams, points, gray=True)
    flags, photo, gate = CURRENT[name]
    component = diag.component_record(off_stats[0])
    assert set(component["flags"]) == flags
    assert bool(np.count_nonzero(_alpha(off_tiles)) > 0) is photo
    assert bool(on_stats[0].get("truth_correction")) is gate
    assert on["solved_at"] == receipt["solved_at"]
    assert on["components"][0]["raster_bounds_xy"] == off["components"][0]["raster_bounds_xy"]
    assert bool(np.count_nonzero(_alpha(on_tiles)) > 0) is (photo and not gate)
    if not gate:
        assert pfw.manifest_bytes(on) == pfw.manifest_bytes(off)
        assert on_tiles == off_tiles
    if name in ("staircase", "plateaus_connector"):
        # Known limitation: one non-horizontal plane is painted across distinct steps.
        normal = off_stats[0]["_diag_inputs"]["plane"].normal
        assert abs(float(normal[0])) > 0.05
    if name == "ramp":
        # Current whole-ramp photo is one tilted plane; transition withholding is slice 1.
        assert abs(float(off_stats[0]["_diag_inputs"]["plane"].normal[0])) > 0.05
    if name in ("carry", "kneel", "sit", "stacked"):
        assert component.get("camera_split_unconfirmed") or name == "stacked"


@pytest.mark.parametrize("value,expected", [(None, False), ("on", True), ("ON", False),
                                             ("1", False), ("true", False), (" on", False)])
def test_gray_switch_is_exactly_on(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(config.WORLD_PHOTO_FLOORPLAN_MULTILEVEL_GRAY_ENV, raising=False)
    else:
        monkeypatch.setenv(config.WORLD_PHOTO_FLOORPLAN_MULTILEVEL_GRAY_ENV, value)
    assert config.world_photo_floorplan_multilevel_gray_setting() is expected


@pytest.mark.parametrize("variant", ["carry_height", "density", "pose_noise", "frame_order",
                                      "sim3_half", "sim3_double"])
def test_split_classification_survives_perturbation(tmp_path, variant):
    cams, points = split_level()
    rng = np.random.default_rng(92)
    if variant == "carry_height":
        cams = [((p[0], p[1], p[2] + 0.12), yaw, pitch) for p, yaw, pitch in cams]
    elif variant == "density":
        points = points[::2]
    elif variant == "pose_noise":
        cams = [(tuple(np.asarray(p) + rng.normal(0, 0.015, 3)), yaw, pitch)
                for p, yaw, pitch in cams]
    elif variant == "frame_order":
        rng.shuffle(cams)
    elif variant.startswith("sim3_"):
        scale = 0.5 if variant == "sim3_half" else 2.0
        cams = [(tuple(scale * np.asarray(p)), yaw, pitch) for p, yaw, pitch in cams]
        points = scale * points
    _, _, _, _, _, tiles, stats = _build(tmp_path, cams, points, gray=True)
    component = diag.component_record(stats[0])
    assert "multi_level_suspected" in component["flags"]
    assert stats[0]["truth_correction"] == "multi_level_suspected"
    assert np.count_nonzero(_alpha(tiles)) == 0


def test_default_off_is_byte_identical_to_explicit_off(tmp_path):
    cams, points = split_level()
    tree, receipt, eligible, stat = _write_scene(tmp_path, cams, points)
    default_block, default_tiles, _ = pfw.build_landing(
        tmp_path, WORLD, SESSION, receipt, eligible, tree, stat, plane_memory={})
    off_block, off_tiles, _ = pfw.build_landing(
        tmp_path, WORLD, SESSION, receipt, eligible, tree, stat,
        plane_memory={}, multilevel_gray=False)
    assert pfw.manifest_bytes(default_block) == pfw.manifest_bytes(off_block)
    assert default_tiles == off_tiles


def test_gray_check_stays_within_landing_cap(tmp_path):
    cams, points = split_level()
    _, _, _, _, _, _, stats = _build(tmp_path, cams, points, gray=True)
    assert stats[0]["ms"]["multilevel_gray_check"] < 8000.0


def test_floor_confirmation_checks_the_landing_budget_between_reads(tmp_path):
    cams, points = split_level()
    _, _, _, _, _, _, stats = _build(tmp_path, cams, points, gray=False)
    inputs = stats[0]["_diag_inputs"]
    calls = 0

    def deadline():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise pfw.LandingOverrun("cap reached during floor check")

    with pytest.raises(pfw.LandingOverrun, match="cap reached during floor check"):
        diag.multilevel_floor_confirmation(
            inputs["plane"], inputs["entries"], inputs["receipt_component"],
            inputs["band_points"], check=deadline)
    assert calls == 5


def test_confirmed_split_publishes_dated_gray_with_same_bounds(tmp_path):
    cams, points = split_level()
    _, receipt, _, _, photo, photo_tiles, _ = _build(tmp_path, cams, points, gray=False)
    _, _, _, _, gray, gray_tiles, stats = _build(tmp_path, cams, points, gray=True)
    assert np.count_nonzero(_alpha(photo_tiles)) > 0
    assert np.count_nonzero(_alpha(gray_tiles)) == 0
    assert gray["solved_at"] == receipt["solved_at"]
    assert gray["components"][0]["raster_bounds_xy"] == photo["components"][0]["raster_bounds_xy"]
    assert gray["components"][0]["digest"] == hashlib.sha256(
        next(iter(gray_tiles.values()))[0]).hexdigest()
    assert stats[0]["truth_correction"] == "multi_level_suspected"
    assert stats[0]["opaque_fraction"] == 0


def test_unconfirmed_camera_split_keeps_photo(tmp_path):
    cams, points = kneel_in_one_level_room()
    _, _, _, _, off, off_tiles, _ = _build(tmp_path, cams, points, gray=False)
    _, _, _, _, on, on_tiles, stats = _build(tmp_path, cams, points, gray=True)
    assert pfw.manifest_bytes(on) == pfw.manifest_bytes(off)
    assert on_tiles == off_tiles
    assert "truth_correction" not in stats[0]


def test_floor_check_on_weak_plane_does_not_certify_a_second_level():
    # Walk-8 e2e3's one-level main component had a spurious camera/floor split
    # on a plane with only five support cells. A weak fitted reference cannot
    # authorize replacing its published photo with a gray truth correction.
    rec = {"plane": {"ok": True, "source": "estimated", "inliers": 214,
                     "support_cells": 5},
           "cams": {"bimodal": True, "gap_h": 0.413, "separation": 4.25,
                    "low_share": 0.151, "floor_check": {"result": "confirmed", "moved": 0.871}},
           "band": {"frames": 0}, "area": {}}
    assert "multi_level_suspected" not in diag.component_flags(rec)


@pytest.mark.parametrize("v2", [False, True])
def test_gray_truth_correction_publishes_past_guards_and_clears_kept(tmp_path, v2, caplog):
    cams, points = split_level()
    tree, receipt, eligible, stat = _write_scene(tmp_path, cams, points)
    state = pfw.PhotoFloorplanState()
    latest = {"receipt": receipt}
    worker = pfw.PhotoFloorplanWorker(
        tmp_path, latest=lambda _w, _s: latest["receipt"], state=state,
        method="homography", regress_v2=v2, kept=True, multilevel_gray=True,
    )
    try:
        # The prior tile is the known-wrong single-plane image.
        prior, prior_tiles, prior_stats = pfw.build_landing(
            tmp_path, WORLD, SESSION, receipt, eligible, tree, stat, plane_memory={})
        assert np.count_nonzero(_alpha(prior_tiles)) > 0
        assert state.publish(WORLD, SESSION, prior, prior_tiles)
        pair = (WORLD, SESSION)
        worker._published_opaque[pair] = pfw.opaque_by_ref(prior_stats)
        worker._published_opaque_area[pair] = pfw.opaque_area_by_ref(prior_stats)
        assert state.keep(WORLD, SESSION, dict(receipt, solved_at=1001.0), pfw.KEPT_REGRESSED)
        latest["receipt"] = dict(receipt, solved_at=1002.0)
        with caplog.at_level("INFO", logger="tower.world_builder.photo_floorplan_worker"):
            worker.offer(WORLD, SESSION, latest["receipt"], eligible, tree, stat)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and worker.outcomes.get(pair, {}).get("identity") != pfw.identity_of(latest["receipt"]):
                time.sleep(0.02)
        outcome = worker.outcomes[pair]
        assert outcome["disposition"] == "ok"
        assert outcome["truth_correction"] == [0]
        assert outcome["components"][0]["truth_correction"] == "multi_level_suspected"
        landed, kept = state.served(WORLD, SESSION)
        assert landed["solved_at"] == 1002.0 and kept is None
        assert np.count_nonzero(_alpha({landed["components"][0]["digest"]: state.tile(
            WORLD, SESSION, landed["components"][0]["digest"])})) == 0
        assert worker._regress_streak.get(pair, 0) == 0
        assert pair not in worker._frozen_since
        assert any("floor confirmed" in r.getMessage() and "truth_correction_reason" in r.getMessage()
                   for r in caplog.records)
    finally:
        worker.shutdown()
