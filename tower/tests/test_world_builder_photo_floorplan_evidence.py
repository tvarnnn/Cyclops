"""Lane C: per-pixel floor evidence for the photo floor plan's homography warp.

TOWER_WORLD_PHOTO_FLOORPLAN_EVIDENCE (exactly `on`, default off). Off is today's fixed
bottom-40% band byte for byte; on, a pixel is warped only with positive evidence that it
sees the floor (live depth, else the frame's own sparse points, minus vetoes and
transient masks) and everything else stays gray.

Scene: the floor is world Z=0, component basis forward=+X, up=+Z (raster x = world X,
raster y = -world Y), cameras 1.5 above the floor (H = 1.5).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tower import config
from tower.world_builder import photo_floorplan_evidence as fe
from tower.world_builder import photo_floorplan_worker as pfw
from tower.world_builder.photo_floorplan import FloorPlane, PhotoFrame, build_homography_raster
from tests.test_world_builder_photo_floorplan import _pitched_camera
from tests.test_world_builder_photo_floorplan_worker import SESSION, WORLD, _synthetic_solve

H = 1.5
W_PX, H_PX, F = 72, 128, 90.0
BOUNDS = (-3.0, -3.0, 3.0, 3.0)
FLOOR_A, FLOOR_B = (200, 170, 120), (150, 120, 80)     # carpet checker
TABLE = (0, 255, 0)                                      # pure green: never floor
PERSON = (255, 0, 255)
WALL = (240, 240, 240)


def _plane(h=H):
    return FloorPlane(ok=True, reason="ok", normal=(0.0, 0.0, 1.0), offset=0.0,
                      stats={"camera_height_median": h})


def _scene(pos, forward, cam_up, *, table=None, wall_x=None, sticker=None):
    """Ray-cast floor (checker), an optional table top (x0, x1, y0, y1, z) and an
    optional wall at x = wall_x. Returns (rgb, depth = camera z, intrinsics)."""
    cx, cy = (W_PX - 1) / 2.0, (H_PX - 1) / 2.0
    right = np.cross(forward, cam_up)
    down = -np.asarray(cam_up)
    vv, uu = np.mgrid[0:H_PX, 0:W_PX].astype(float)
    rays = (((uu - cx) / F)[..., None] * right + ((vv - cy) / F)[..., None] * down
            + np.asarray(forward))
    t_best = np.full((H_PX, W_PX), np.inf)
    rgb = np.zeros((H_PX, W_PX, 3), np.uint8)
    rgb[:] = WALL
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -pos[2] / rays[..., 2]
    hit = (rays[..., 2] < -1e-9) & (t > 0)
    p = pos + np.where(hit, t, 0)[..., None] * rays
    even = (np.floor(p[..., 0] / 0.5) + np.floor(p[..., 1] / 0.5)).astype(int) % 2 == 0
    colour = np.where(even[..., None], np.array(FLOOR_A), np.array(FLOOR_B))
    if sticker is not None:
        x0, x1, y0, y1 = sticker
        on = (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)
        colour = np.where(on[..., None], np.array(PERSON), colour)
    rgb[hit] = colour[hit]
    t_best[hit] = t[hit]
    if table is not None:
        x0, x1, y0, y1, z = table
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (z - pos[2]) / rays[..., 2]
        q = pos + np.where(np.isfinite(tt), tt, 0)[..., None] * rays
        on = ((tt > 0) & (tt < t_best) & (q[..., 0] >= x0) & (q[..., 0] <= x1)
              & (q[..., 1] >= y0) & (q[..., 1] <= y1))
        rgb[on] = TABLE
        t_best[on] = tt[on]
    if wall_x is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            tw = (wall_x - pos[0]) / rays[..., 0]
        on = (tw > 0) & (tw < t_best)
        rgb[on] = WALL
        t_best[on] = tw[on]
    depth = np.where(np.isfinite(t_best), t_best, np.nan)   # rays have z_cam = 1
    return rgb, depth, (F, F, cx, cy)


def _frame(kid, cam, *, order=0, evidence=None, **scene):
    pos, fwd, up = cam
    rgb, depth, (fx, fy, cx, cy) = _scene(pos, fwd, up, **scene)
    fr = PhotoFrame(kid=kid, position_xyz=tuple(pos), forward_xyz=tuple(fwd), up_xyz=tuple(up),
                    fx=fx, fy=fy, cx=cx, cy=cy, color=rgb, order=order,
                    floor_evidence=evidence)
    return fr, depth


def _depth_evidence(fr, depth, plane=None):
    plane = plane or _plane()
    floor, known, info = fe.depth_floor_evidence(
        depth, np.isfinite(depth), scale=1.0, position_xyz=fr.position_xyz,
        forward_xyz=fr.forward_xyz, up_xyz=fr.up_xyz, fx=fr.fx, fy=fr.fy, cx=fr.cx, cy=fr.cy,
        normal=plane.normal, offset=plane.offset, h_med=plane.stats["camera_height_median"])
    return floor, known, info


def _raster(frames, **kwargs):
    return build_homography_raster(origin_xyz=(0, 0, H), up_xyz=(0, 0, 1), forward_xyz=(1, 0, 0),
                                   bounds_xy=BOUNDS, plane=kwargs.pop("plane", _plane()),
                                   frames=frames, width=128, height=128, **kwargs)


def _painted(result, colour, tol=40):
    """Opaque cells of `colour`, at full saturation (multi-view) or pulled 40% to gray
    (single view)."""
    a = result.rgba[..., 3] == 255
    rgb = result.rgba[..., :3].astype(int)
    c = np.array(colour)
    d = np.minimum(np.abs(rgb - c).max(-1), np.abs(rgb - (c * 0.6 + 51.2)).max(-1))
    return int((a & (d <= tol)).sum())


CAMS = [_pitched_camera(pos=(-1.5, 0.0, H), yaw_deg=0, pitch_deg=55),
        _pitched_camera(pos=(-1.2, 0.6, H), yaw_deg=-15, pitch_deg=50),
        _pitched_camera(pos=(-1.4, -0.5, H), yaw_deg=10, pitch_deg=60)]
TABLE_TOP = (-1.3, -0.6, -0.5, 0.5, 0.55)     # 0.37 H: a low table / bed top


# ---------------------------------------------------------------------------
# furniture in the band
# ---------------------------------------------------------------------------


def test_a_table_top_in_the_band_stays_gray():
    frames, depths = zip(*[_frame(f"k{i}", c, order=i, table=TABLE_TOP) for i, c in enumerate(CAMS)])
    before = _raster(list(frames))
    assert _painted(before, TABLE) > 100, "the fixture must reproduce today's smear"
    for fr, d in zip(frames, depths):
        floor, known, info = _depth_evidence(fr, d)
        assert info["layer"] == pytest.approx(0.0, abs=0.02)
        table_px = np.all(fr.color == np.array(TABLE), axis=-1)
        assert table_px.any() and not (floor & table_px).any()
        fr.floor_evidence = floor
    after = _raster(list(frames), floor_evidence=True)
    assert _painted(after, TABLE) == 0
    assert after.opaque_cells > 500              # the floor around it is still painted


def test_a_wall_in_the_band_stays_gray():
    fr, d = _frame("w", CAMS[0], wall_x=-0.9)
    before = _raster([fr])
    floor, _, _ = _depth_evidence(fr, d)
    wall_px = np.all(fr.color == np.array(WALL), axis=-1)
    assert not (floor & wall_px).any()
    fr.floor_evidence = floor
    after = _raster([fr], floor_evidence=True)
    assert _painted(before, WALL, tol=10) > 0
    assert _painted(after, WALL, tol=10) == 0


def test_a_frame_that_sees_only_furniture_has_no_floor_layer_and_paints_nothing():
    cam = _pitched_camera(pos=(0.0, 0.0, H), yaw_deg=0, pitch_deg=80)
    fr, d = _frame("t", cam, table=(-2.0, 2.0, -2.0, 2.0, 0.55))
    floor, known, info = _depth_evidence(fr, d)
    assert known.any() and info["layer"] is None and not floor.any()
    fr.floor_evidence = floor
    after = _raster([fr], floor_evidence=True)
    assert after.opaque_cells == 0
    assert after.stats["frames_rejected"] == {"no floor evidence": 1}


def test_the_floor_layer_follows_a_plane_that_is_a_little_off():
    """The fitted plane 0.15 H below the floor this frame sees (walk 8: one plane for a
    not-quite-flat room): the frame's own floor layer is found, the table still is not."""
    fr, d = _frame("o", CAMS[0], table=TABLE_TOP)
    low = FloorPlane(ok=True, reason="ok", normal=(0.0, 0.0, 1.0), offset=0.15 * H,
                     stats={"camera_height_median": H})
    floor, _, info = _depth_evidence(fr, d, plane=low)
    assert info["layer"] == pytest.approx(0.15, abs=0.02)
    floor_px = ~np.all(fr.color == np.array(TABLE), axis=-1) & ~np.all(fr.color == np.array(WALL), axis=-1)
    assert floor[floor_px].mean() > 0.9
    assert not (floor & ~floor_px).any()


def test_a_layer_outside_the_window_is_not_floor():
    fr, d = _frame("x", CAMS[0])
    sunk = FloorPlane(ok=True, reason="ok", normal=(0.0, 0.0, 1.0), offset=0.6 * H,
                      stats={"camera_height_median": H})   # the walk-8 misplaced closet
    floor, known, info = _depth_evidence(fr, d, plane=sunk)
    assert known.any() and info["layer"] is None and not floor.any()


# ---------------------------------------------------------------------------
# a person: the transient mask wins over every other source
# ---------------------------------------------------------------------------


def test_a_masked_person_is_never_painted_even_where_the_depth_says_floor():
    # Worst case: the "person" is a flat sticker ON the floor, so depth calls it floor.
    sticker = (-0.2, 0.3, -0.25, 0.25)
    fr, d = _frame("p", CAMS[0], sticker=sticker)
    floor, known, _ = _depth_evidence(fr, d)
    person_px = np.all(fr.color == np.array(PERSON), axis=-1)
    assert (floor & person_px).sum() == person_px.sum() > 50
    transient = fe.resample_transient(cv2.dilate(person_px.astype(np.uint8), np.ones((3, 3))),
                                      W_PX, H_PX)
    ev, _ = fe.combine(depth_floor=floor, depth_known=known, transient=transient,
                       shape=floor.shape)
    fr.floor_evidence = ev
    after = _raster([fr], floor_evidence=True)
    assert _painted(after, PERSON, tol=60) == 0
    assert after.opaque_cells > 300


def test_the_worker_reads_the_solve_transient_mask(tmp_path):
    """(c) through the worker: `<solve>/masks/<image>.png`, 0 = transient, at the solve
    camera's size; sparse evidence elsewhere (no depth cache)."""
    pos, fwd, up = CAMS[0]
    rgb, depth, (fx, fy, cx, cy) = _scene(pos, fwd, up)
    camera = {"fx": fx * 2, "fy": fy * 2, "cx": cx * 2 + 0.5, "cy": cy * 2 + 0.5,
              "width": W_PX * 2, "height": H_PX * 2}
    # Sparse floor observations on a grid of the lower image (full-size pixel coords).
    vv, uu = np.mgrid[70:128:6, 2:72:6]
    uu, vv = uu.ravel().astype(float), vv.ravel().astype(float)
    right = np.cross(fwd, up)
    rays = (((uu - cx) / fx)[:, None] * right + ((vv - cy) / fy)[:, None] * -np.asarray(up)
            + np.asarray(fwd))
    pts = pos + (-pos[2] / rays[:, 2])[:, None] * rays
    obs_xy = np.c_[(uu + 0.5) * 2 - 0.5, (vv + 0.5) * 2 - 0.5]
    mask = np.full((H_PX * 2, W_PX * 2), 255, np.uint8)
    mask[:, : W_PX] = 0                                  # left half: a person
    (tmp_path / "masks").mkdir()
    cv2.imwrite(str(tmp_path / "masks" / "00000007.jpg.png"), mask)
    common = dict(kid="s:00000007", rgb=rgb, fx=fx, fy=fy, cx=cx, cy=cy, position=pos,
                  up_world=up, forward_world=fwd, scale=1.0, plane=_plane(), obs_xy=obs_xy,
                  obs_xyz=pts, camera=camera)
    ev_masked, info = pfw.frame_floor_evidence(**common, masks_dir=tmp_path / "masks")
    ev_plain, info_plain = pfw.frame_floor_evidence(**common, masks_dir=None)
    assert info["source"] == info_plain["source"] == "sparse"
    assert ev_plain[:, : W_PX // 2 - 1].any()
    assert not ev_masked[:, : W_PX // 2].any()
    assert (ev_masked[:, W_PX // 2 + 1:] == ev_plain[:, W_PX // 2 + 1:]).all()
    assert info["transient_px"] > 0


# ---------------------------------------------------------------------------
# a floor-only band is unchanged
# ---------------------------------------------------------------------------


def _floor_frames():
    return [_frame(f"f{i}", c, order=i)[0] for i, c in enumerate(CAMS)]


def test_evidence_equal_to_the_band_reproduces_todays_raster():
    """Evidence = the band's own pixels gives today's raster, up to the band's top edge
    (the band edge is continuous, evidence is per nearest pixel)."""
    off = _raster(_floor_frames())
    frames = _floor_frames()
    v_top = 0.6 * H_PX - 0.5
    for fr in frames:
        fr.floor_evidence = np.broadcast_to((np.arange(H_PX) >= v_top)[:, None], (H_PX, W_PX)).copy()
    on = _raster(frames, floor_evidence=True)
    same = (on.rgba == off.rgba).all(-1)
    a_off = off.rgba[..., 3] == 255
    assert a_off.sum() > 400
    assert same[a_off].mean() > 0.97
    assert (same | a_off).all()        # nothing painted that today leaves gray


def test_a_floor_only_scene_keeps_every_band_cell_and_its_colour():
    off = _raster(_floor_frames())
    frames, depths = zip(*[_frame(f"f{i}", c, order=i) for i, c in enumerate(CAMS)])
    for fr, d in zip(frames, depths):
        fr.floor_evidence = _depth_evidence(fr, d)[0]
    on = _raster(list(frames), floor_evidence=True)
    kept = off.rgba[..., 3] == 255
    assert (on.rgba[..., 3][kept] == 255).mean() > 0.99
    diff = np.abs(on.rgba[..., :3].astype(int) - off.rgba[..., :3].astype(int)).max(-1)
    # Same floor colour wherever both paint (a different winning view may blend an edge).
    assert np.median(diff[kept & (on.rgba[..., 3] == 255)]) <= 2


# ---------------------------------------------------------------------------
# fail to gray, never extrapolate
# ---------------------------------------------------------------------------


def test_a_frame_without_evidence_contributes_nothing():
    frames = _floor_frames()
    frames[1].floor_evidence = None
    frames[2].floor_evidence = np.zeros((H_PX, W_PX), bool)
    frames[0].floor_evidence = np.zeros((H_PX, W_PX), bool)
    frames[0].floor_evidence[100:110, 30:40] = True
    on = _raster(frames, floor_evidence=True)
    assert on.contributing_keyframes == ["f0"]
    assert on.stats["frames_rejected"] == {"no floor evidence": 2}
    assert 0 < on.opaque_cells < 400


def test_sparse_seeds_grow_only_through_their_own_colour_and_stop_at_vetoes():
    img = np.zeros((60, 60, 3), np.uint8)
    img[:, :30] = (200, 170, 120)
    img[:, 30:] = (40, 40, 40)
    floor, veto, n = fe.sparse_floor_evidence(img, [[10, 30]], [0.0])
    assert n == 1 and floor[30, 10] and not floor[:, 30:].any()
    r = round(fe.GROW_RADIUS_FRAC * 60)
    assert not floor[30, 10 + r + 2:30].any()          # bounded reach
    floor2, veto2, n2 = fe.sparse_floor_evidence(img, [[10, 30], [14, 30]], [0.0, 0.5])
    assert veto2[30, 14] and not floor2[30, 14]
    none, _, n0 = fe.sparse_floor_evidence(img, np.zeros((0, 2)), [])
    assert n0 == 0 and not none.any()


def test_the_depth_scale_fit_refuses_a_bad_fit():
    rng = np.random.default_rng(0)
    z = rng.uniform(1, 4, 40)
    ok, s, ho, _ = fe.fit_depth_scale(z * 0.5, z)
    assert ok and s == pytest.approx(2.0) and ho == pytest.approx(0.0, abs=1e-12)
    ok, *_ , why = fe.fit_depth_scale(rng.uniform(1, 4, 40), z)
    assert not ok and "held-out" in why
    ok, *_ , why = fe.fit_depth_scale(z[:5], z[:5])
    assert not ok and "too few" in why


# ---------------------------------------------------------------------------
# switch, OFF identity, the cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, on", [(None, False), ("", False), ("1", False), ("true", False),
                                       ("ON", False), ("on ", False), ("on", True)])
def test_only_exactly_on_enables_floor_evidence(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv("TOWER_WORLD_PHOTO_FLOORPLAN_EVIDENCE", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_PHOTO_FLOORPLAN_EVIDENCE", value)
    assert config.world_photo_floorplan_evidence_setting() is on


def test_off_ignores_any_evidence_a_frame_carries():
    plain = _raster(_floor_frames())
    frames = _floor_frames()
    for fr in frames:
        fr.floor_evidence = np.zeros((H_PX, W_PX), bool)
    off = _raster(frames)
    off_explicit = _raster(_floor_frames(), floor_evidence=False)
    assert np.array_equal(off.rgba, plain.rgba) and np.array_equal(off_explicit.rgba, plain.rgba)
    assert "floor_evidence" not in off.stats and off.stats == plain.stats


def test_off_build_landing_is_byte_identical_and_reads_no_cache(tmp_path, monkeypatch):
    solve_dir, tree, receipt, eligible, stat = _synthetic_solve(tmp_path)
    monkeypatch.delenv("TOWER_WORLD_PHOTO_FLOORPLAN_EVIDENCE", raising=False)
    opened = []
    monkeypatch.setattr(pfw, "DepthCache", lambda *a, **k: opened.append(a) or None)
    block, tiles, stats = pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree, stat)
    block2, tiles2, stats2 = pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree,
                                               stat, floor_evidence=False)
    assert opened == []
    assert block == block2 and tiles == tiles2
    assert "evidence" not in stats[0] and "evidence" not in stats[0]["ms"]
    assert '"evidence"' not in pfw.component_summary(stats)


def test_on_build_landing_without_cache_or_points_is_all_gray(tmp_path, monkeypatch):
    """The synthetic solve has no live-depth cache and no sparse observations: on, there
    is no evidence anywhere, so nothing is painted (never extrapolated)."""
    solve_dir, tree, receipt, eligible, stat = _synthetic_solve(tmp_path)
    monkeypatch.setenv("TOWER_WORLD_PHOTO_FLOORPLAN_EVIDENCE", "on")
    block, tiles, stats = pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree, stat)
    (c,) = stats
    assert c["opaque_fraction"] == 0.0
    assert c["evidence"]["depth_cache"] is False
    assert c["evidence"]["frames_none"] == c["selected_frames"] > 0
    assert "evidence" in c["ms"]
    assert '"evidence"' in pfw.component_summary(stats)


def test_the_wall_cap_still_stops_an_evidence_build_between_frames(tmp_path):
    solve_dir, tree, receipt, eligible, stat = _synthetic_solve(tmp_path)
    calls = []

    def check():
        calls.append(1)
        if len(calls) > 3:
            raise pfw.LandingOverrun("exceeded the 8.0 s wall cap")

    with pytest.raises(pfw.LandingOverrun):
        pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree, stat, check=check,
                          floor_evidence=True)


def test_sparse_evidence_cost_is_capped_by_max_seeds():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (144, 256, 3), dtype=np.uint8)
    uv = np.c_[rng.uniform(0, 255, 20000), rng.uniform(0, 143, 20000)]
    _, _, n = fe.sparse_floor_evidence(img, uv, np.zeros(len(uv)))
    assert n <= fe.MAX_SEEDS


def test_the_landing_cap_is_unchanged():
    assert pfw.LANDING_WALL_CAP_S == 8.0
