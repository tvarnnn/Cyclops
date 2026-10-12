"""Photo floor plan NEW-SCENE DIAGNOSTICS (manager 349 lane A): synthetic scenes the walks 3-8
apartment never offered -- a large open room, a long straight hallway, a split level, a
low-texture floor -- each run through the REAL landing build (`build_landing`: solve arrays,
undistorted images, plane fit, warp, tile), then through the diagnostics.

Two things are asserted per scene:
  1. the diagnostics flag the scene's condition, and only the conditions it has;
  2. the landing itself is what the code produced BEFORE the diagnostics existed: the plane,
     the frame gates, the warped frames and the opaque raster cells are pinned to the values
     the base commit (f5b3211) produced on this same fixture, and the worker publishes
     byte-identical manifests and tiles with the diagnostics files on and off.
"""

from __future__ import annotations

import io
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from tower import config
from tower.world_builder import photo_floorplan_diagnostics as diag
from tower.world_builder import photo_floorplan_worker as pfw
from tests.test_world_builder_photo_floorplan import _pitched_camera, _render

WORLD, SESSION = "diag-world", "diag-session"
W_PX, H_PX, F_PX = 72, 128, 90.0
CAMERA = {"fx": F_PX, "fy": F_PX, "cx": (W_PX - 1) / 2.0, "cy": (H_PX - 1) / 2.0,
          "width": W_PX, "height": H_PX}
CAM_HEIGHT = 1.5


def _quat_wxyz(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    return [w, x, y, z]


def _write_scene(root, cams, points, *, solved_at=1000.0, max_range=6.0):
    """A world whose solve/<session> holds what a real landing reads: arrays (with each
    camera's own observations of the points it sees), metadata, and undistorted images.

    `cams`: [(position, yaw_deg, pitch_deg)]; `points`: (N, 3) world points. Returns
    (tree, receipt, eligible, solution_stat)."""
    solve_dir = Path(root) / "worlds" / WORLD / "solve" / SESSION
    (solve_dir / "images").mkdir(parents=True, exist_ok=True)
    (Path(root) / "worlds" / WORLD / "sessions" / SESSION).mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    kids, rows, poses, obs, obs_xy = [], [], {}, [], []
    centres = []
    for i, (pos, yaw, pitch) in enumerate(cams):
        pos, fwd, cam_up = _pitched_camera(pos=pos, yaw_deg=yaw, pitch_deg=pitch)
        right = np.cross(fwd, cam_up)
        R_wc = np.stack([right, -cam_up, fwd], axis=1)
        kid = f"{SESSION}:{i:08d}"
        kids.append(kid)
        img, _ = _render(pos, fwd, cam_up, w=W_PX, h=H_PX, f=F_PX)
        cv2.imwrite(str(solve_dir / "images" / f"{i:08d}.jpg"), img[:, :, ::-1])
        rows.append({"keyframe_id": kid, "segment_index": 0, "status": "solved",
                     "rotation": _quat_wxyz(R_wc), "translation": list(map(float, pos))})
        R_cw = R_wc.T
        poses[kid] = {"component": 0, "observations": 50, "rotation": list(R_cw.reshape(-1)),
                      "translation": list(-R_cw @ pos)}
        xc = (pts - pos) @ R_wc            # camera coordinates: x right, y down, z forward
        z = xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = F_PX * xc[:, 0] / z + CAMERA["cx"]
            v = F_PX * xc[:, 1] / z + CAMERA["cy"]
        seen = ((z > 0.05) & (u >= 0) & (u <= W_PX - 1) & (v >= 0) & (v <= H_PX - 1)
                & (np.linalg.norm(pts - pos, axis=1) <= max_range))
        for j in np.flatnonzero(seen):
            obs.append([i, len(obs), int(j)])
            obs_xy.append([u[j], v[j]])
        centres.append(pos)
    arrays = dict(xyz=pts.astype(np.float32), rgb=np.zeros((len(pts), 3), np.uint8),
                  component=np.zeros(len(pts), np.int32),
                  first_keyframe=np.zeros(len(pts), np.int32),
                  track_length=np.full(len(pts), 5, np.int32),
                  error=np.full(len(pts), 0.5, np.float32),
                  observations=np.asarray(obs, np.int32).reshape(-1, 3),
                  observation_xy=np.asarray(obs_xy, np.float32).reshape(-1, 2))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    (solve_dir / "solution.npz").write_bytes(buf.getvalue())
    solution = {"solved_at": solved_at, "keyframe_ids": kids, "poses": poses, "camera": CAMERA,
                "components": [{"index": 0, "points": int(len(pts))}]}
    time.sleep(0.02)  # the metadata is written after the arrays, as write_solution does
    (solve_dir / "solution.json").write_text(json.dumps(solution), encoding="utf-8")
    st = (solve_dir / "solution.json").stat()
    tree = {"solution": solution, "poses": rows,
            "placements": [{"segment_index": 0, "state": "registered", "reference_segment": 0,
                            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0], "translation": [0.0] * 3,
                            "scale": 1.0}]}
    c = np.asarray(centres)
    xs, ys = c[:, 0], -c[:, 1]          # receipt x = p . forward, y = p . (forward x up) = -Y
    receipt = {"solved_at": solved_at, "computed_at": solved_at + 0.4,
               "geometry_revision": "g-diag", "frame_revision": 1,
               "horizon_keyframes": len(cams), "keyframes_pending": 0,
               "stations_omitted": 3, "components_omitted": 0, "components_total": 1,
               "components": [{"reference_segment": 0,
                               "origin_xyz": [float(c[:, 0].mean()), float(c[:, 1].mean()),
                                              CAM_HEIGHT],
                               "up_xyz": [0.0, 0.0, 1.0], "forward_xyz": [1.0, 0.0, 0.0],
                               "bounds_xy": [float(xs.min()), float(ys.min()),
                                             float(xs.max()), float(ys.max())]}]}
    return tree, receipt, [(k, 0) for k in kids], (st.st_size, st.st_mtime_ns)


# ---------------------------------------------------------------------------
# the four scenes (deterministic)
# ---------------------------------------------------------------------------


def _grid_cams(xs, ys, *, z_of=lambda x, y: CAM_HEIGHT, seed=0, pitch=(35.0, 55.0)):
    rng = np.random.default_rng(seed)
    cams = []
    for x in xs:
        for y in ys:
            z = z_of(x, y) + float(rng.normal(0.0, 0.02))
            cams.append(((float(x), float(y), z), float(rng.uniform(0, 360)),
                         float(rng.uniform(*pitch))))
    return cams


def _floor(rng, n, x0, x1, y0, y1, z=0.0, noise=0.01):
    return np.c_[rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), z + rng.normal(0.0, noise, n)]


def large_open_room():
    """A 16 x 16 open floor walked on a grid: the camera square is ~12 camera heights wide."""
    rng = np.random.default_rng(10)
    cams = _grid_cams(np.linspace(-7, 7, 10), np.linspace(-7, 7, 10), seed=10)
    return cams, _floor(rng, 6000, -8.5, 8.5, -8.5, 8.5)


def long_hallway():
    """A 2-unit-wide, 26-unit-long corridor walked straight down its middle and back."""
    rng = np.random.default_rng(20)
    cams = []
    for i, x in enumerate(np.linspace(0.0, 24.0, 90)):
        yaw = 0.0 if i % 2 == 0 else 180.0
        cams.append(((float(x), float(rng.normal(0.0, 0.01)),
                      CAM_HEIGHT + float(rng.normal(0.0, 0.02))),
                     yaw + float(rng.normal(0.0, 3.0)), float(rng.uniform(35.0, 55.0))))
    return cams, _floor(rng, 3000, -1.0, 25.0, -1.0, 1.0)


SPLIT_X, SPLIT_RISE = 4.5, 1.0


def split_level():
    """Two floor levels 1.0 apart (0.67 camera heights) joined at x = 4.5, in an ordinary-
    sized 9 x 5 space; the walker carries the camera at the same height above whichever
    level they stand on."""
    rng = np.random.default_rng(30)

    def z_of(x, y):
        return CAM_HEIGHT + (SPLIT_RISE if x >= SPLIT_X else 0.0)

    cams = _grid_cams(np.linspace(0.3, 4.2, 7), np.linspace(-2, 2, 6), z_of=z_of, seed=30)
    cams += _grid_cams(np.linspace(4.8, 8.5, 5), np.linspace(-2, 2, 6), z_of=z_of, seed=31)
    pts = np.vstack([_floor(rng, 2200, -1.0, SPLIT_X, -3.0, 3.0),
                     _floor(rng, 1500, SPLIT_X, 10.0, -3.0, 3.0, z=SPLIT_RISE)])
    return cams, pts


def low_texture_floor():
    """A normal 6 x 6 room whose floor yields almost no stable SfM points (polished concrete)."""
    rng = np.random.default_rng(40)
    cams = _grid_cams(np.linspace(-2.5, 2.5, 6), np.linspace(-2.5, 2.5, 6), seed=40)
    return cams, _floor(rng, 30, -3.5, 3.5, -3.5, 3.5)


SCENES = {"large_open_room": large_open_room, "long_hallway": long_hallway,
          "split_level": split_level, "low_texture_floor": low_texture_floor}

EXPECTED_FLAGS = {
    "large_open_room": {"large_extent"},
    "long_hallway": {"large_extent", "collinear_trajectory"},
    "split_level": {"multi_level_suspected"},
    "low_texture_floor": {"low_floor_support"},
}


def _build(tmp_path, name):
    cams, pts = SCENES[name]()
    tree, receipt, eligible, stat = _write_scene(tmp_path, cams, pts)
    block, tiles, stats = pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree,
                                            stat, plane_memory={})
    return receipt, block, tiles, stats


def _record(receipt, stats):
    return diag.landing_record(world_id=WORLD, session_id=SESSION, receipt=receipt,
                               disposition="ok", published=True, escaped=False,
                               elapsed_ms=1.0, stats=stats, now=receipt["solved_at"] + 2.0)


# ---------------------------------------------------------------------------
# 1. each scene fires its own flag, and only it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SCENES))
def test_each_new_scene_fires_its_own_flags_and_no_other(tmp_path, name):
    receipt, _, _, stats = _build(tmp_path, name)
    record = _record(receipt, stats)
    (comp,) = record["components"]
    assert set(comp["flags"]) == EXPECTED_FLAGS[name], json.dumps(comp, indent=1)
    summary = diag.walk_summary([record], world_id=WORLD, session_id=SESSION)
    assert set(summary["flags_fired"]) == EXPECTED_FLAGS[name]
    for flag in EXPECTED_FLAGS[name]:
        assert summary["flags"][flag]["main_component"] is True
        assert summary["flags"][flag]["landings_fired"] == 1


def test_the_large_room_is_flagged_for_resolution_not_for_support(tmp_path):
    receipt, _, _, stats = _build(tmp_path, "large_open_room")
    (comp,) = _record(receipt, stats)["components"]
    assert comp["plane"]["ok"] and comp["plane"]["source"] == "estimated"
    assert comp["area"]["side_H"] > diag.LARGE_EXTENT_CELL_FRAC * diag.MAX_DIM
    assert comp["area"]["selected"] == 64 and comp["area"]["admissible"] > 64
    assert comp["band"]["frames"] >= 3 and comp["band"]["on_median"] > 0.9   # a bare floor
    assert comp["cams"]["bimodal"] is False


def test_the_hallway_has_no_usable_camera_plane(tmp_path):
    receipt, _, _, stats = _build(tmp_path, "long_hallway")
    (comp,) = _record(receipt, stats)["components"]
    assert comp["plane"]["cam_spread"] < 0.05
    assert comp["plane"]["reference"] in ("roll_free", "receipt_up")
    assert comp["plane"]["ok"]                     # the floor itself is still found


def test_the_split_level_is_two_camera_levels_and_the_floor_moved_with_them(tmp_path):
    receipt, _, _, stats = _build(tmp_path, "split_level")
    (comp,) = _record(receipt, stats)["components"]
    cams = comp["cams"]
    assert cams["bimodal"] is True
    assert cams["gap_h"] == pytest.approx(SPLIT_RISE / CAM_HEIGHT, abs=0.1)
    assert 0.3 < cams["low_share"] < 0.7
    assert cams["floor_check"]["result"] == "confirmed"
    assert cams["floor_check"]["moved"] == pytest.approx(1.0, abs=0.2)


def test_the_low_texture_floor_is_gray_or_fallback(tmp_path):
    receipt, _, _, stats = _build(tmp_path, "low_texture_floor")
    (comp,) = _record(receipt, stats)["components"]
    assert (not comp["plane"]["ok"]) or comp["plane"]["source"] == "fallback_camera_drop"
    assert comp["plane"]["inliers"] is None or comp["plane"]["inliers"] < 40


# ---------------------------------------------------------------------------
# 2. the landing is unchanged (base f5b3211 values on these fixtures; files on vs off)
# ---------------------------------------------------------------------------

# Taken by running `_build` on the base commit f5b3211 (before the diagnostics existed):
# the plane decision, the frame gates and the opaque raster cells of each scene.
BASE_F5B3211 = {'large_open_room': {'gates': {'ok': 100},
                                    'normal': [-9.0326e-05, -0.000270355, 0.999999959],
                                    'opaque_cells': 55154,
                                    'plane_ok': True,
                                    'plane_reason': 'ok',
                                    'projected': 64,
                                    'rejected': {},
                                    'selected': 64,
                                    'source': None,
                                    'warped': 64},
                'long_hallway': {'gates': {'ok': 90},
                                 'normal': [0.0, 0.0, 1.0],
                                 'opaque_cells': 9397,
                                 'plane_ok': True,
                                 'plane_reason': 'ok',
                                 'projected': 64,
                                 'rejected': {},
                                 'selected': 64,
                                 'source': None,
                                 'warped': 64},
                'low_texture_floor': {'gates': {},
                                      'normal': [-0.001974327, 0.000977316, 0.999997573],
                                      'opaque_cells': 0,
                                      'plane_ok': False,
                                      'plane_reason': 'inadequate support: 29 points below the cameras inside '
                                                      'the padded bounds (< 40); no fallback: 0 downward '
                                                      'frames with >= 5 bottom-band points (< 3)',
                                      'projected': 0,
                                      'rejected': None,
                                      'selected': None,
                                      'source': None,
                                      'warped': None},
                'split_level': {'gates': {'ok': 72},
                                'normal': [-0.165219355, 0.001779639, 0.98625524],
                                'opaque_cells': 142342,
                                'plane_ok': True,
                                'plane_reason': 'ok',
                                'projected': 64,
                                'rejected': {},
                                'selected': 64,
                                'source': None,
                                'warped': 64}}


def _decision(block, tiles, stats):
    (c,) = stats
    plane = c.get("plane") or {}
    return {"plane_ok": c["plane_ok"], "plane_reason": c["plane_reason"],
            "source": plane.get("source"),
            "normal": [round(v, 9) for v in (plane.get("reference_normal_xyz") or [])],
            "gates": c.get("frame_gate_counts"), "selected": c.get("selected_frames"),
            "warped": c.get("frames_warped"), "rejected": c.get("frames_rejected"),
            "opaque_cells": int(round(c["opaque_fraction"] * 512 * 512)),
            "projected": c["projected_keyframes"]}


@pytest.mark.parametrize("name", sorted(SCENES))
def test_the_landing_is_what_the_base_commit_built(tmp_path, name):
    receipt, block, tiles, stats = _build(tmp_path, name)
    expected = BASE_F5B3211[name]
    assert expected is not None
    assert _decision(block, tiles, stats) == expected


@pytest.mark.parametrize("name", ["split_level", "large_open_room"])
def test_the_worker_publishes_the_same_bytes_with_diagnostics_files_on_and_off(
        tmp_path, name, caplog):
    cams, pts = SCENES[name]()
    tree, receipt, eligible, stat = _write_scene(tmp_path, cams, pts)
    published = {}
    for files in (False, True):
        state = pfw.PhotoFloorplanState()
        worker = pfw.PhotoFloorplanWorker(tmp_path, latest=lambda w, s: receipt, state=state,
                                          method="homography", diagnostics_files=files)
        try:
            with caplog.at_level("INFO", logger="tower.world_builder.photo_floorplan_worker"):
                worker.offer(WORLD, SESSION, receipt, eligible, tree, stat)
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and not (
                        worker.idle() and (WORLD, SESSION) in worker.outcomes
                        and worker._diag_pending is None
                        and worker._diag_records.get((WORLD, SESSION))):
                    time.sleep(0.02)
        finally:
            worker.shutdown()
        block = state.landed(WORLD, SESSION)
        assert block is not None, worker.outcomes
        digests = [c["digest"] for c in block["components"]]
        published[files] = (pfw.manifest_bytes(block),
                            [state.tile(WORLD, SESSION, d) for d in digests],
                            worker.outcomes[(WORLD, SESSION)]["disposition"])
        # the build's references are dropped once the diagnostics have read them
        assert all(pfw.DIAG_INPUTS not in c
                   for c in worker.outcomes[(WORLD, SESSION)]["components"])
        session = diag.session_dir(tmp_path, WORLD, SESSION)
        assert (session / diag.SUMMARY_FILENAME).is_file() is files
        assert (session / diag.RECORDS_FILENAME).is_file() is files
    assert published[True] == published[False]
    summary = json.loads((diag.session_dir(tmp_path, WORLD, SESSION) / diag.SUMMARY_FILENAME)
                         .read_text(encoding="utf-8"))
    assert set(summary["flags_fired"]) == EXPECTED_FLAGS[name]
    assert summary["landings"]["published"] == 1
    assert summary["landing_lag_s"]["last"] > 0
    assert summary["final_landing"]["covers_all_accepted"] is True
    # the log line is there either way, and bounded
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("world photo floorplan diagnostics:")]
    assert len(lines) == 2
    assert all(len(line.split("record=", 1)[1]) <= diag.MAX_RECORD_BYTES for line in lines)


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------


def test_only_exactly_on_enables_the_diagnostics_files(monkeypatch):
    for value, expected in ((None, False), ("", False), ("1", False), ("true", False),
                            ("ON", False), (" on", False), ("on", True)):
        if value is None:
            monkeypatch.delenv(config.WORLD_PHOTO_FLOORPLAN_DIAGNOSTICS_ENV, raising=False)
        else:
            monkeypatch.setenv(config.WORLD_PHOTO_FLOORPLAN_DIAGNOSTICS_ENV, value)
        assert config.world_photo_floorplan_diagnostics_setting() is expected


def test_a_one_level_walk_never_splits_into_two(monkeypatch):
    rng = np.random.default_rng(0)
    for heights in (rng.normal(1.5, 0.1, 2000),            # Otsu cut of a normal: ~2.65 SD
                    rng.uniform(1.2, 1.8, 2000),             # of a uniform: ~3.46 SD
                    rng.laplace(1.5, 0.08, 2000)):
        out = diag.camera_height_split(heights, 1.5)
        assert out["bimodal"] is False, out
        assert out["separation"] < diag.MULTILEVEL_MIN_SEPARATION


def test_a_camera_split_needs_h_a_gap_and_two_real_clusters():
    rng = np.random.default_rng(1)
    two = np.r_[rng.normal(1.5, 0.03, 60), rng.normal(2.5, 0.03, 40)]
    assert diag.camera_height_split(two, 1.5)["bimodal"] is True
    assert diag.camera_height_split(two, None)["bimodal"] is None          # no H: not judged
    small_step = np.r_[rng.normal(1.5, 0.01, 60), rng.normal(1.6, 0.01, 40)]
    out = diag.camera_height_split(small_step, 1.5)                        # 0.07 H: a bob
    assert out["separation"] > diag.MULTILEVEL_MIN_SEPARATION and out["bimodal"] is False
    crouch = np.r_[rng.normal(1.5, 0.03, 95), rng.normal(0.9, 0.03, 5)]   # 5%: a crouch
    assert diag.camera_height_split(crouch, 1.5)["bimodal"] is False
    assert diag.camera_height_split(two[:10], 1.5)["bimodal"] is None       # too few cameras


def test_a_carrying_height_change_on_one_floor_is_not_a_second_level():
    """The cameras split in two, but every frame's own floor stays at z = 0."""
    rng = np.random.default_rng(2)
    entries, frames = [], []
    for i in range(60):
        z = 1.5 if i < 36 else 1.9
        kid = f"k{i}"
        pos = np.array([i * 0.1, 0.0, z])
        _, fwd, up = _pitched_camera(pos=pos, yaw_deg=0.0, pitch_deg=40.0)
        entries.append((kid, pos, up, fwd, 1.0))
        frames.append((kid, np.c_[rng.uniform(0, 6, 12), rng.uniform(-1, 1, 12),
                                  rng.normal(0, 0.01, 12)]))
    heights = np.array([e[1][2] for e in entries])
    split = diag.camera_height_split(heights, 1.5)
    assert split["bimodal"] is True
    check = diag.floor_level_check(entries, frames, (0.0, 0.0, 1.0), 0.0, split)
    assert check["result"] == "contradicted" and abs(check["moved"]) < 0.2
    rec = {"plane": {"ok": True, "source": "estimated", "inliers": 500, "support_cells": 20},
           "cams": dict(split, floor_check=check), "band": {"frames": 0}, "area": {}}
    assert "multi_level_suspected" not in diag.component_flags(rec)


def test_a_supported_layer_below_the_floor_is_a_second_level_but_one_above_is_furniture():
    base = {"ok": True, "source": "estimated", "inliers": 500, "support_cells": 20,
            "ambiguous": True, "alt_ratio": 0.9}
    below = {"plane": dict(base, alt_h=-0.6), "cams": {}, "band": {"frames": 0}, "area": {}}
    above = {"plane": dict(base, alt_h=0.2), "cams": {}, "band": {"frames": 0}, "area": {}}
    assert "multi_level_suspected" in diag.component_flags(below)
    assert "multi_level_suspected" not in diag.component_flags(above)


def test_furniture_bands_need_off_floor_mass_above_not_below():
    plane = {"ok": True, "source": "estimated", "inliers": 500, "support_cells": 20}
    furniture = {"plane": plane, "cams": {}, "area": {},
                 "band": {"frames": 30, "on_median": 0.1, "above_median": 0.8,
                          "below_median": 0.0}}
    pose_error = {"plane": plane, "cams": {}, "area": {},
                  "band": {"frames": 30, "on_median": 0.1, "above_median": 0.1,
                           "below_median": 0.7}}
    floor = {"plane": plane, "cams": {}, "area": {},
             "band": {"frames": 30, "on_median": 0.6, "above_median": 0.4, "below_median": 0.0}}
    assert "furniture_dominated_bands" in diag.component_flags(furniture)
    assert "furniture_dominated_bands" not in diag.component_flags(pose_error)
    assert "furniture_dominated_bands" not in diag.component_flags(floor)


def test_low_support_marks_fallback_rejected_and_marginal_planes():
    def flags(**plane):
        return diag.component_flags({"plane": plane, "cams": {}, "band": {"frames": 0},
                                     "area": {}})
    assert "low_floor_support" in flags(ok=False, reason="inadequate support")
    assert "low_floor_support" in flags(ok=True, source="fallback_camera_drop", inliers=23)
    assert "low_floor_support" in flags(ok=True, source="estimated", inliers=60, support_cells=9)
    assert "low_floor_support" in flags(ok=True, source="estimated", inliers=300, support_cells=4)
    assert "low_floor_support" not in flags(ok=True, source="estimated", inliers=300,
                                            support_cells=12)


def test_the_landing_record_and_summary_are_bounded():
    comp = {"reference_segment": 0, "eligible": 10, "plane_ok": False,
            "plane_reason": "x" * 5000, "plane": {"reference_normal": "receipt_up"},
            "opaque_fraction": 0.0}
    stats = [dict(comp, reference_segment=i, eligible=10 + i) for i in range(64)]
    rec = diag.landing_record(world_id=WORLD, session_id=SESSION, receipt={"solved_at": 1.0},
                              disposition="ok", published=True, escaped=False, elapsed_ms=1.0,
                              stats=stats, now=2.0)
    assert len(diag._dumps(rec).encode()) <= diag.MAX_RECORD_BYTES
    assert rec["components_dropped"] > 0
    assert rec["components"][0]["eligible"] == 73          # the largest are kept
    summary = diag.walk_summary([rec] * 600)
    assert len(diag._dumps(summary).encode()) <= diag.MAX_SUMMARY_BYTES


def test_a_broken_component_never_breaks_the_record():
    rec = diag.landing_record(world_id=WORLD, session_id=SESSION, receipt={"solved_at": 1.0},
                              disposition="skipped: regressed (ref 0: ...; 1/3)", published=False,
                              escaped=False, elapsed_ms=1.0,
                              stats=[{"reference_segment": 0, pfw.DIAG_INPUTS: {"entries": 5}}],
                              now=2.0)
    assert rec["class"] == "regressed" and "error" in rec["components"][0]


def test_files_are_written_only_into_an_existing_session(tmp_path):
    rec = {"solved_at": 1.0, "components": []}
    assert diag.write_files(tmp_path, WORLD, SESSION, rec, {"k": 1}) is False
    assert not (tmp_path / "worlds").exists()                  # nothing created
    session = diag.session_dir(tmp_path, WORLD, SESSION)
    session.mkdir(parents=True)
    assert diag.write_files(tmp_path, WORLD, SESSION, rec, {"k": 1}) is True
    assert diag.write_files(tmp_path, WORLD, SESSION, dict(rec, solved_at=2.0), {"k": 2})
    assert [r["solved_at"] for r in diag.read_records(session / diag.RECORDS_FILENAME)] == [1.0, 2.0]
    assert json.loads((session / diag.SUMMARY_FILENAME).read_text(encoding="utf-8")) == {"k": 2}
    assert sorted(p.name for p in session.iterdir()) == sorted(
        [diag.RECORDS_FILENAME, diag.SUMMARY_FILENAME])        # no temp file left behind


def test_the_walk_summary_counts_skips_escapes_and_lag():
    def rec(t, disposition, published, escaped=False, comps=()):
        return {"solved_at": t, "disposition": disposition,
                "class": diag._disposition_class(disposition), "published": published,
                "escaped": escaped, "elapsed_ms": 100.0, "landing_lag_s": 3.0,
                "stations_omitted": 5, "components": list(comps)}
    comp = {"ref": 0, "share": 1.0, "plane": {"source": "estimated"},
            "flags": {"low_floor_support": "x"}}
    records = [rec(100.0, "ok", True, comps=[comp]),
               rec(150.0, "skipped: regressed (ref 0: opaque 0.1 < 0.25 x published 0.5; 1/3)",
                   False, comps=[comp]),
               rec(160.0, "skipped: exceeded the 8.0 s wall cap", False),
               rec(220.0, "ok", True, escaped=True, comps=[dict(comp, flags={})])]
    s = diag.walk_summary(records)
    assert s["landings"] == {"total": 4, "built": 3, "published": 2, "escaped": 1,
                             "by_class": {"ok": 2, "regressed": 1, "wall_cap": 1}}
    assert s["landing_gap_s"]["max"] == 60.0
    assert s["final_landing"]["solved_at"] == 220.0
    assert s["flags"]["low_floor_support"]["fired"] is False          # judged on the last
    assert s["flags"]["low_floor_support"]["landings_fired"] == 2


# ---------------------------------------------------------------------------
# the offline CLI reads both a diagnostics-era log and an older one
# ---------------------------------------------------------------------------


def test_the_offline_cli_reads_diagnostics_and_legacy_log_lines(tmp_path):
    from scripts import photo_floorplan_diagnostics as cli

    legacy_comp = [{"ref": 0, "eligible": 29, "points": 1888, "plane_ok": True,
                    "plane_reason": "ok (fallback: camera trajectory)", "reference": "roll_free",
                    "plane_source": "fallback_camera_drop", "plane_reuse": "no previous plane",
                    "candidates": 69, "inliers": 24, "gates": {"ok": 9}, "selected": 9,
                    "warped": 9, "projected": 9, "opaque": 0.4, "opaque_area": 10.0}]
    record = {"v": 1, "world": "w", "session": "s", "solved_at": 2000.0, "class": "ok",
              "disposition": "ok", "published": True, "escaped": True, "landing_lag_s": 4.0,
              "components": [{"ref": 0, "share": 1.0, "plane": {"source": "estimated"},
                              "flags": {"large_extent": "x"}}]}
    lines = [
        "2026-10-12 11:11:07,747 INFO tower.world_builder.photo_floorplan_worker world photo "
        "floorplan: world=w session=s landing=(1000.0, 'g1', 1) method=homography "
        f"elapsed_ms=125.0 disposition=ok components={json.dumps(legacy_comp)}",
        "2026-10-12 11:12:00,000 WARNING tower.world_builder.photo_floorplan_worker world photo "
        "floorplan: publishing a regressed landing after 3 consecutive skips (escape valve): x",
        "2026-10-12 11:12:00,001 INFO tower.world_builder.photo_floorplan_worker world photo "
        "floorplan: world=w session=s landing=(1100.0, 'g2', 1) method=homography "
        f"elapsed_ms=99.0 disposition=ok components={json.dumps(legacy_comp)}",
        "2026-10-12 11:12:00,002 INFO tower.world_builder.photo_floorplan_worker world photo "
        "floorplan: world=other session=s landing=(1200.0, 'g3', 1) method=homography "
        f"elapsed_ms=99.0 disposition=ok components={json.dumps(legacy_comp)}",
    ]
    log = tmp_path / "tower.err.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    new, legacy = cli.records_from_log(log, "w", "s")
    assert new == [] and [r["solved_at"] for r in legacy] == [1000.0, 1100.0]
    assert [r["escaped"] for r in legacy] == [False, True]
    (comp,) = legacy[0]["components"]
    assert comp["plane"]["source"] == "fallback_camera_drop"
    assert set(comp["flags"]) == {"low_floor_support"}
    # a diagnostics-era log: the record is taken as written
    log.write_text(lines[0] + "\n" + "2026-10-12 11:13:00,000 INFO x world photo floorplan "
                   "diagnostics: world=w session=s landing=(2000.0, 'g4', 1) record="
                   + json.dumps(record) + "\n", encoding="utf-8")
    new, legacy = cli.records_from_log(log, "w", "s")
    assert new == [record] and len(legacy) == 1
    out = tmp_path / "summary.json"
    assert cli.main(["--log", str(log), "--world", "w", "--session", "s", "--out", str(out)]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["flags_fired"] == ["large_extent"] and summary["landings"]["escaped"] == 1
