"""Focused unit tests for the FOW v1.2 slice-1 pure raster core.

Scene convention used throughout: component basis `forward=(1,0,0)` (world
X), `up=(0,0,1)` (world Z) -> `right = forward x up = (0,-1,0)` (= -world Y).
So raster x = world X, raster y = -world Y, raster "height" = world Z. A
camera with world `forward_xyz=(0,0,-1)` and `up_xyz=(0,1,0)` looks straight
down (-Z); for a flat floor at world Z=0, the per-pixel camera-frame depth
is CONSTANT (`camera_z / scale`) because the floor is perpendicular to the
optical axis -- which is what makes a closed-form synthetic scene tractable.
"""

import numpy as np
import pytest

from tower.world_builder.photo_floorplan import (
    KeyframeSample,
    build_raster,
    padded_square_bounds,
)

FORWARD = (1.0, 0.0, 0.0)
UP = (0.0, 0.0, 1.0)
BOUNDS = (-1.2, -1.2, 1.2, 1.2)


def _straight_down_camera(*, pos_x, pos_y, pos_z, color, size=100, fx=100.0, fy=100.0,
                          kid="k", order=0, scale=1.0):
    """A camera looking straight down (-Z world) at a flat floor at world Z=0.

    Returns a KeyframeSample whose depth is the constant value that puts
    every pixel's back-projection exactly on that floor.
    """
    cx = cy = (size - 1) / 2.0
    depth_value = pos_z / scale
    depth = np.full((size, size), depth_value, dtype=np.float64)
    valid = np.ones((size, size), dtype=bool)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, :] = color
    return KeyframeSample(
        kid=kid, position_xyz=(pos_x, pos_y, pos_z),
        forward_xyz=(0.0, 0.0, -1.0), up_xyz=(0.0, 1.0, 0.0),
        scale=scale, fx=fx, fy=fy, cx=cx, cy=cy,
        depth=depth, valid=valid, color=rgb, order=order,
    )


def _cell_of(x_comp, y_comp, bounds=BOUNDS, width=64, height=64, pad_frac=0.15):
    min_x, min_y, max_x, max_y = padded_square_bounds(bounds, pad_frac)
    col = round((x_comp - min_x) / (max_x - min_x) * (width - 1))
    row = round((y_comp - min_y) / (max_y - min_y) * (height - 1))
    return row, col


class TestPaddedSquareBounds:
    def test_squares_a_non_square_box_about_its_own_centre_then_pads(self):
        # width 4 (x: -1..3), height 2 (y: 0..2); side = max(4, 2) = 4.
        bounds = padded_square_bounds((-1.0, 0.0, 3.0, 2.0), pad_frac=0.15)
        mid_x, mid_y = 1.0, 1.0
        half = 4.0 * 1.30 / 2.0  # side * (1 + 2*pad_frac) / 2
        assert bounds == pytest.approx((mid_x - half, mid_y - half, mid_x + half, mid_y + half))

    def test_padding_zero_returns_the_squared_box_unpadded(self):
        bounds = padded_square_bounds((-1.0, -1.0, 1.0, 1.0), pad_frac=0.0)
        assert bounds == pytest.approx((-1.0, -1.0, 1.0, 1.0))

    def test_rejects_nonfinite_bounds(self):
        with pytest.raises(ValueError):
            padded_square_bounds((float("nan"), 0.0, 1.0, 1.0))


class TestEmptyCase:
    def test_no_keyframes_gives_an_all_alpha_zero_raster_of_the_requested_size(self):
        result = build_raster(origin_xyz=(0, 0, 0), up_xyz=UP, forward_xyz=FORWARD,
                              bounds_xy=BOUNDS, keyframes=[], width=32, height=48)
        assert result.width == 32 and result.height == 48
        assert result.rgba.shape == (48, 32, 4)
        assert int(result.rgba[:, :, 3].sum()) == 0
        assert result.opaque_cells == 0
        assert result.opaque_fraction == 0.0

    def test_width_and_height_are_capped_at_512(self):
        result = build_raster(origin_xyz=(0, 0, 0), up_xyz=UP, forward_xyz=FORWARD,
                              bounds_xy=BOUNDS, keyframes=[], width=10_000, height=10_000)
        assert result.width == 512 and result.height == 512


class TestFlatFloorTwoCameras:
    """Two straight-down cameras, overlapping footprints, a shared flat floor."""

    def _build(self, **kwargs):
        cam_a = _straight_down_camera(pos_x=-0.3, pos_y=0.0, pos_z=1.5,
                                      color=(200, 40, 40), kid="a", order=0)
        cam_b = _straight_down_camera(pos_x=0.3, pos_y=0.0, pos_z=1.5,
                                      color=(40, 40, 200), kid="b", order=1)
        defaults = dict(origin_xyz=(0, 0, 1.5), up_xyz=UP, forward_xyz=FORWARD,
                        bounds_xy=BOUNDS, keyframes=[cam_a, cam_b], width=64, height=64)
        defaults.update(kwargs)
        return build_raster(**defaults)

    def test_the_overlap_region_is_opaque_and_multi_view(self):
        result = self._build()
        # Camera A's footprint is world x in [-1.05, 0.45]; camera B's is
        # [-0.45, 1.05]. x_comp = world x, y_comp = -world y, so the centre
        # (world x=0, y=0) is seen by both.
        row, col = _cell_of(0.0, 0.0)
        assert result.rgba[row, col, 3] == 255
        assert result.multi_view_cells > 0
        assert {"a", "b"} <= set(result.contributing_keyframes)

    def test_a_camera_only_region_is_opaque_single_view_with_that_cameras_colour(self):
        result = self._build()
        # world x = -0.9 is inside A's footprint [-1.05, -0.45) only.
        row, col = _cell_of(-0.9, 0.0)
        assert result.rgba[row, col, 3] == 255
        r, g, b, a = (int(v) for v in result.rgba[row, col])
        # Single-view cells are desaturated toward gray (128), not full colour.
        assert r > g and r > b  # still recognisably "camera A's red"
        assert not (r, g, b) == (200, 40, 40)  # but not the raw, saturated colour either

    def test_far_outside_either_footprint_is_gray_alpha_zero(self):
        result = self._build()
        row, col = _cell_of(1.15, 1.15)  # near the padded bounds' far corner
        assert result.rgba[row, col, 3] == 0

    def test_opaque_region_is_a_strict_subset_of_the_raster(self):
        result = self._build()
        assert 0 < result.opaque_cells < result.total_cells

    def test_raster_bounds_xy_is_the_padded_square_not_the_raw_receipt_bounds(self):
        result = self._build()
        assert result.raster_bounds_xy != BOUNDS
        assert result.raster_bounds_xy == pytest.approx(padded_square_bounds(BOUNDS))


class TestSingleViewRequiresNonGrazingAndConsistency:
    def test_a_lone_camera_on_a_flat_floor_still_paints_with_one_view(self):
        # Only one camera: every cell in its footprint needs the single-view
        # path (non-grazing + 3x3 patch consistency), which a flat floor
        # viewed head-on satisfies everywhere.
        cam = _straight_down_camera(pos_x=0.0, pos_y=0.0, pos_z=1.5,
                                    color=(10, 200, 10), kid="solo")
        result = build_raster(origin_xyz=(0, 0, 1.5), up_xyz=UP, forward_xyz=FORWARD,
                              bounds_xy=BOUNDS, keyframes=[cam], width=64, height=64)
        row, col = _cell_of(0.0, 0.0)
        assert result.rgba[row, col, 3] == 255
        assert result.single_view_cells > 0
        assert result.multi_view_cells == 0


class TestVerticalSurfaceRejected:
    def test_a_wall_facing_camera_contributes_no_opaque_cells(self):
        # A camera looking along world +X (horizontal) at a vertical wall:
        # the wall's normal is horizontal, so |normal . up| ~= 0 and the
        # surface-normal gate (>= 0.6) must reject every pixel.
        size = 40
        cx = cy = (size - 1) / 2.0
        depth = np.full((size, size), 2.0, dtype=np.float64)
        valid = np.ones((size, size), dtype=bool)
        rgb = np.full((size, size, 3), 128, dtype=np.uint8)
        wall_cam = KeyframeSample(
            kid="wall", position_xyz=(-2.0, 0.0, 1.0),
            forward_xyz=(1.0, 0.0, 0.0), up_xyz=(0.0, 0.0, 1.0),
            scale=1.0, fx=100.0, fy=100.0, cx=cx, cy=cy,
            depth=depth, valid=valid, color=rgb, order=0,
        )
        result = build_raster(origin_xyz=(0, 0, 1.0), up_xyz=UP, forward_xyz=FORWARD,
                              bounds_xy=BOUNDS, keyframes=[wall_cam], width=64, height=64)
        assert result.opaque_cells == 0
        assert int(result.rgba[:, :, 3].sum()) == 0


class TestInvalidAndOutOfBoundsPixelsAreExcluded:
    def test_an_all_invalid_keyframe_contributes_nothing(self):
        cam = _straight_down_camera(pos_x=0.0, pos_y=0.0, pos_z=1.5, color=(1, 2, 3))
        cam.valid[:, :] = False
        result = build_raster(origin_xyz=(0, 0, 1.5), up_xyz=UP, forward_xyz=FORWARD,
                              bounds_xy=BOUNDS, keyframes=[cam], width=32, height=32)
        assert result.opaque_cells == 0

    def test_nonfinite_or_nonpositive_depth_pixels_are_excluded(self):
        cam = _straight_down_camera(pos_x=0.0, pos_y=0.0, pos_z=1.5, color=(1, 2, 3))
        cam.depth[0, 0] = float("nan")
        cam.depth[0, 1] = -1.0
        cam.depth[0, 2] = 0.0
        # Should not raise, and should simply treat those pixels as absent.
        result = build_raster(origin_xyz=(0, 0, 1.5), up_xyz=UP, forward_xyz=FORWARD,
                              bounds_xy=BOUNDS, keyframes=[cam], width=32, height=32)
        assert result.opaque_cells >= 0  # ran to completion; no crash on bad pixels


# ===========================================================================
# Fallback (3): lower-image floor homography.
#
# Same scene convention as above: component forward = world +X, up = world
# +Z, so raster x = world X and raster y = -world Y. The floor is world Z=0.
# ===========================================================================

import io  # noqa: E402
import math  # noqa: E402

from PIL import Image  # noqa: E402

from tower.world_builder.photo_floorplan import (  # noqa: E402
    GRAZING_COS_MIN,
    FloorPlane,
    PhotoFrame,
    build_homography_raster,
    encode_tile,
    estimate_floor_plane,
)

RED = (255, 0, 0)
BLUE = (0, 0, 255)
SKY = (255, 0, 255)
IMG_W, IMG_H, F = 72, 128, 90.0


def _checker(x, y, period=0.5):
    """World-floor colour: RED where floor(x/p)+floor(y/p) is even, else BLUE."""
    even = (np.floor(x / period) + np.floor(y / period)).astype(int) % 2 == 0
    return np.where(even[..., None], np.array(RED), np.array(BLUE)).astype(np.uint8)


def _pitched_camera(*, pos, yaw_deg=0.0, pitch_deg=45.0):
    """A camera at `pos`, heading `yaw_deg` from world +X toward +Y, pitched down."""
    p, yw = math.radians(pitch_deg), math.radians(yaw_deg)
    h = np.array([math.cos(yw), math.sin(yw), 0.0])
    forward = math.cos(p) * h + np.array([0.0, 0.0, -math.sin(p)])
    cam_up = math.sin(p) * h + np.array([0.0, 0.0, math.cos(p)])
    return np.asarray(pos, float), forward, cam_up


def _render(pos, forward, cam_up, *, w=IMG_W, h=IMG_H, f=F, colour=_checker):
    """Ray-cast the synthetic floor: what a rectified pinhole camera sees."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    right = np.cross(forward, cam_up)
    down = -cam_up
    vv, uu = np.mgrid[0:h, 0:w].astype(float)
    rays = (((uu - cx) / f)[..., None] * right + ((vv - cy) / f)[..., None] * down + forward)
    hit = rays[..., 2] < -1e-9
    t = np.where(hit, -pos[2] / np.where(hit, rays[..., 2], -1.0), 0.0)
    pts = pos + t[..., None] * rays
    img = colour(pts[..., 0], pts[..., 1])
    img[~hit] = SKY
    return img, (f, f, cx, cy)


def _frame(kid, pos, forward, cam_up, order=0, img=None, check_points=None):
    rendered, (fx, fy, cx, cy) = _render(pos, forward, cam_up)
    return PhotoFrame(kid=kid, position_xyz=tuple(pos), forward_xyz=tuple(forward),
                      up_xyz=tuple(cam_up), fx=fx, fy=fy, cx=cx, cy=cy,
                      color=rendered if img is None else img, order=order,
                      check_points_xyz=check_points)


def _floor_plane(cam_height=1.5):
    return FloorPlane(ok=True, reason="ok", normal=(0.0, 0.0, 1.0), offset=0.0,
                      stats={"camera_height_median": cam_height})


def _cell_world(result, rows, cols):
    min_x, min_y, max_x, max_y = result.raster_bounds_xy
    x = min_x + cols * (max_x - min_x) / (result.width - 1)
    y_r = min_y + rows * (max_y - min_y) / (result.height - 1)
    return x, -y_r  # raster y = p . (forward x up) = -world Y


HBOUNDS = (-3.0, -3.0, 3.0, 3.0)


def _homography(frames, plane=None, **kwargs):
    defaults = dict(origin_xyz=(0, 0, 1.5), up_xyz=UP, forward_xyz=FORWARD,
                    bounds_xy=HBOUNDS, plane=plane or _floor_plane(), frames=frames,
                    width=128, height=128)
    defaults.update(kwargs)
    return build_homography_raster(**defaults)


class TestHomographyPlacesTheFloorInTheRightCells:
    def test_tilted_cameras_paint_the_checkerboard_where_it_lies(self):
        frames = [
            _frame("a", *_pitched_camera(pos=(-1.5, 0.0, 1.5), yaw_deg=0, pitch_deg=50), order=0),
            _frame("b", *_pitched_camera(pos=(1.5, 0.5, 1.5), yaw_deg=180, pitch_deg=40), order=1),
            _frame("c", *_pitched_camera(pos=(0.0, -1.5, 1.5), yaw_deg=90, pitch_deg=60), order=2),
        ]
        result = _homography(frames)
        assert result.opaque_cells > 500
        assert set(result.contributing_keyframes) == {"a", "b", "c"}
        rows, cols = np.nonzero(result.rgba[:, :, 3] == 255)
        x, y = _cell_world(result, rows, cols)
        # Skip cells within ~1.5 cells of a checker edge (bilinear blending).
        cell = (HBOUNDS[2] - HBOUNDS[0]) * 1.3 / 127
        fx_, fy_ = np.mod(x, 0.5), np.mod(y, 0.5)
        interior = ((np.minimum(fx_, 0.5 - fx_) > 1.5 * cell)
                    & (np.minimum(fy_, 0.5 - fy_) > 1.5 * cell))
        expected_red = _checker(x, y)[:, 0] > 128
        got = result.rgba[rows, cols, :3].astype(int)
        got_red = got[:, 0] > got[:, 2]
        assert interior.sum() > 200
        agreement = float(np.mean(got_red[interior] == expected_red[interior]))
        assert agreement > 0.98

    def test_nothing_is_painted_behind_the_camera(self):
        frames = [_frame("a", *_pitched_camera(pos=(-1.5, 0.0, 1.5), pitch_deg=50))]
        result = _homography(frames)
        rows, cols = np.nonzero(result.rgba[:, :, 3] == 255)
        x, _ = _cell_world(result, rows, cols)
        assert len(x) > 0 and (x > -1.5).all()
        assert result.raster_bounds_xy == pytest.approx(padded_square_bounds(HBOUNDS))


class TestOnlyTheBottomFortyPercent:
    def test_the_upper_sixty_percent_never_reaches_the_raster(self):
        pos, fwd, cam_up = _pitched_camera(pos=(0.0, 0.0, 1.5), pitch_deg=70)
        img, _ = _render(pos, fwd, cam_up, colour=lambda x, y: np.full(x.shape + (3,), 40, np.uint8))
        img[: int(0.6 * IMG_H)] = (0, 255, 0)  # the top 60% is pure green floor
        result = _homography([_frame("a", pos, fwd, cam_up, img=img)])
        painted = result.rgba[:, :, 3] == 255
        assert painted.sum() > 100
        greenish = result.rgba[:, :, 1].astype(int) > result.rgba[:, :, 0].astype(int) + 30
        assert not (painted & greenish).any()
        assert result.stats["cells_rejected"]["upper_image"] > 0


class TestHorizonIncidenceAndRangeGates:
    def test_a_level_camera_is_not_downward_looking(self):
        result = _homography([_frame("a", *_pitched_camera(pos=(0, 0, 1.5), pitch_deg=0))])
        assert result.opaque_cells == 0
        assert result.stats["frames_rejected"] == {"not downward-looking": 1}

    def test_near_horizon_and_grazing_rays_are_rejected(self):
        # A level camera with the frame gate lifted: the bottom band's top
        # edge is 8.1 degrees below the axis here (row 76.3, cy 63.5, f 90),
        # so the band spans depressions ~8..35 degrees and both gates bind.
        frame = _frame("a", *_pitched_camera(pos=(0.0, 0.0, 1.5), pitch_deg=0))
        result = _homography([frame], bounds_xy=(-12.0, -12.0, 12.0, 12.0), min_pitch_deg=0.0,
                             max_range_frac=100.0)
        rejected = result.stats["cells_rejected"]
        assert rejected["horizon"] > 0
        assert rejected["incidence"] > 0
        rows, cols = np.nonzero(result.rgba[:, :, 3] == 255)
        x, y = _cell_world(result, rows, cols)
        dist = np.sqrt(x ** 2 + y ** 2 + 1.5 ** 2)
        # Single view: painted only where cos(incidence) >= GRAZING_COS_MIN
        # (a cell-size slack for the cell-centre vs. sample position).
        assert result.opaque_cells > 0
        assert (1.5 / dist >= GRAZING_COS_MIN - 0.02).all()

    def test_range_gate_bounds_the_hit_distance(self):
        frame = _frame("a", *_pitched_camera(pos=(0.0, 0.0, 1.5), pitch_deg=20))
        result = _homography([frame], bounds_xy=(-12.0, -12.0, 12.0, 12.0), max_range_frac=2.0)
        assert result.stats["cells_rejected"]["range"] > 0
        rows, cols = np.nonzero(result.rgba[:, :, 3] == 255)
        x, y = _cell_world(result, rows, cols)
        cell = 24.0 * 1.3 / 127
        assert len(x) > 0
        assert (np.sqrt(x ** 2 + y ** 2 + 1.5 ** 2) <= 2.0 * 1.5 + 1.5 * cell).all()

    def test_two_views_admit_cells_one_view_would_call_grazing(self):
        # The same pose twice: every admitted cell is multi-view, so the
        # grazing band (cos 0.26-0.35, depression 15-20.5 degrees) paints
        # too. Pitch 7 puts the band's top edge at ~15 degrees.
        pos, fwd, cam_up = _pitched_camera(pos=(0.0, 0.0, 1.5), pitch_deg=7)
        big = (-12.0, -12.0, 12.0, 12.0)
        kw = dict(bounds_xy=big, max_range_frac=100.0, min_pitch_deg=0.0)
        one = _homography([_frame("a", pos, fwd, cam_up)], **kw)
        two = _homography([_frame("a", pos, fwd, cam_up, order=0),
                           _frame("b", pos, fwd, cam_up, order=1)], **kw)
        assert one.stats["single_view_grazing_cells"] > 0
        assert two.opaque_cells > one.opaque_cells
        assert two.single_view_cells == 0

    def test_a_frame_whose_own_points_lie_below_the_floor_is_rejected(self):
        below = np.array([[1.0, 0.0, -0.8]] * 10)  # 0.8 below a floor seen from 1.5
        frame = _frame("a", *_pitched_camera(pos=(0, 0, 1.5), pitch_deg=50), check_points=below)
        result = _homography([frame])
        assert result.opaque_cells == 0
        assert list(result.stats["frames_rejected"]) == [
            "pose disagrees with plane (own lower-band points below it)"]

    def test_a_rejected_plane_paints_nothing_and_says_why(self):
        frame = _frame("a", *_pitched_camera(pos=(0, 0, 1.5), pitch_deg=50))
        plane = FloorPlane(ok=False, reason="inadequate support: 3 inlier points (< 40)")
        result = _homography([frame], plane=plane)
        assert int(result.rgba[:, :, 3].sum()) == 0
        assert "inadequate support" in result.stats["reason"]


def _room(seed=0, *, n_floor=600, n_table=1500, n_below=40, cam_height=1.5):
    """Floor Z=0 over 6x6, a dense tabletop at Z=0.75 (MORE points than the
    floor, as on walk 8's desk), a wall, a few outliers below, and a camera
    loop at a hand-held height with noise."""
    rng = np.random.default_rng(seed)
    floor = np.c_[rng.uniform(-3, 3, (n_floor, 2)), rng.normal(0, 0.01, n_floor)]
    table = np.c_[rng.uniform(0.5, 2.0, (n_table, 2)), 0.75 + rng.normal(0, 0.01, n_table)]
    wall = np.c_[np.full(400, 3.0), rng.uniform(-3, 3, 400), rng.uniform(0, 2.5, 400)]
    below = np.c_[rng.uniform(-3, 3, (n_below, 2)), rng.uniform(-1.5, -0.3, n_below)]
    ang = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    cams = np.c_[2.0 * np.cos(ang), 2.0 * np.sin(ang), cam_height + rng.normal(0, 0.05, 120)]
    return np.vstack([floor, table, wall, below]), cams


class TestFloorPlaneEstimation:
    # A receipt `up` tilted 8 degrees (the mean camera up axis is biased by
    # the downward pitch), as on walk 8.
    TILTED_UP = (math.sin(math.radians(8)), 0.0, math.cos(math.radians(8)))
    CAM_BOUNDS = (-2.0, -2.0, 2.0, 2.0)

    def _estimate(self, points, cams, **kwargs):
        up = np.asarray(self.TILTED_UP)
        forward = np.asarray(FORWARD) - (np.asarray(FORWARD) @ up) * up
        return estimate_floor_plane(points_xyz=points, camera_positions=cams, up_xyz=up,
                                    forward_xyz=forward / np.linalg.norm(forward),
                                    bounds_xy=self.CAM_BOUNDS, **kwargs)

    def test_finds_the_floor_not_the_denser_table_nor_the_outliers_below(self):
        points, cams = _room()
        plane = self._estimate(points, cams)
        assert plane.ok, plane.reason
        normal = np.asarray(plane.normal)
        assert math.degrees(math.acos(min(1.0, normal @ np.array(UP)))) < 2.0
        assert abs(plane.offset) < 0.05  # Z=0
        assert plane.stats["camera_height_median"] == pytest.approx(1.5, abs=0.05)
        assert plane.stats["reference_normal"] == "camera_plane"
        assert plane.stats["inliers"] >= 400
        assert plane.stats["camera_above_fraction"] == 1.0

    def test_is_deterministic(self):
        points, cams = _room()
        a, b = self._estimate(points, cams), self._estimate(points, cams)
        assert a.normal == b.normal and a.offset == b.offset

    def test_rejects_a_plane_with_inadequate_support(self):
        points, cams = _room(n_floor=12, n_table=0, n_below=0)
        plane = self._estimate(points, cams)
        assert not plane.ok
        assert "inadequate support" in plane.reason

    def test_rejects_when_the_cameras_have_no_consistent_height(self):
        points, cams = _room()
        rng = np.random.default_rng(3)
        cams[:, 2] = rng.uniform(0.2, 3.0, len(cams))  # no hand-held height at all
        plane = self._estimate(points, cams)
        assert not plane.ok
        assert "camera_height_consistency" in plane.reason

    def test_rejects_when_too_few_cameras(self):
        points, cams = _room()
        plane = self._estimate(points, cams[:2])
        assert not plane.ok and "fewer than 3" in plane.reason


class TestEncodeTile:
    def test_small_simple_raster_is_lossless_png(self):
        rgba = np.zeros((512, 512, 4), np.uint8)
        rgba[100:200, 100:300] = (120, 90, 60, 255)
        tile = encode_tile(rgba)
        assert tile["mime"] == "image/png" and tile["bytes"] <= 150 * 1024
        decoded = np.asarray(Image.open(io.BytesIO(tile["data"])).convert("RGBA"))
        assert (decoded == rgba).all()

    def test_a_noisy_raster_steps_down_and_keeps_alpha_binary(self):
        rng = np.random.default_rng(0)
        rgba = rng.integers(0, 256, (512, 512, 4), dtype=np.uint8)
        rgba[:, :, 3] = np.where(rng.random((512, 512)) > 0.3, 255, 0)
        tile = encode_tile(rgba)
        assert tile is not None and tile["bytes"] <= 180 * 1024
        assert tile["encoding"] != "png"
        decoded = np.asarray(Image.open(io.BytesIO(tile["data"])).convert("RGBA"))
        assert decoded.shape[:2] == (tile["height"], tile["width"])
        if (tile["width"], tile["height"]) == (512, 512):
            assert (decoded[:, :, 3] == rgba[:, :, 3]).all()  # WebP alpha is lossless
        assert set(np.unique(decoded[:, :, 3])) <= {0, 255}

    def test_returns_none_when_nothing_fits_the_cap(self):
        rng = np.random.default_rng(1)
        rgba = rng.integers(0, 256, (512, 512, 4), dtype=np.uint8)
        assert encode_tile(rgba, target_bytes=100, cap_bytes=200) is None


class TestFurnitureMask:
    """A green 'tabletop' patch in the bottom band whose own SfM points sit 0.6 above
    a floor seen from 1.5 (0.4 H, well over the 0.12 H tolerance)."""

    def _scene(self, *, above=5, on=0):
        pos, fwd, cam_up = _pitched_camera(pos=(0.0, 0.0, 1.5), pitch_deg=70)
        img, _ = _render(pos, fwd, cam_up, colour=lambda x, y: np.full(x.shape + (3,), 90, np.uint8))
        img[96:112, 16:32] = (0, 255, 0)            # exactly one 16x16 block
        xyz = [[0.3, 0.0, 0.6]] * above + [[0.3, 0.0, 0.0]] * on
        uv = [[24.0, 104.0]] * (above + on)
        return _frame("a", pos, fwd, cam_up, img=img,
                      check_points=np.array(xyz, float).reshape(-1, 3)), np.array(uv, float)

    def _greenish(self, result):
        rgba = result.rgba.astype(int)
        return int(((rgba[:, :, 3] == 255) & (rgba[:, :, 1] > rgba[:, :, 0] + 40)).sum())

    def test_a_block_whose_points_sit_above_the_floor_is_not_painted(self):
        frame, uv = self._scene()
        frame.check_points_uv = uv
        masked = _homography([frame], furniture_mask=True, width=512, height=512)
        unmasked = _homography([frame], furniture_mask=False, width=512, height=512)
        assert self._greenish(unmasked) > 20
        # Only bilinear bleed across the block's one-pixel edge may survive.
        assert self._greenish(masked) <= 0.1 * self._greenish(unmasked)
        assert masked.stats["cells_rejected"]["furniture_mask"] > 0
        assert masked.stats["furniture_masked_blocks"] == 1
        assert masked.opaque_cells > 100                 # the rest of the band still paints

    def test_a_block_with_more_floor_points_than_furniture_points_is_kept(self):
        frame, uv = self._scene(above=1, on=3)
        frame.check_points_uv = uv
        result = _homography([frame], furniture_mask=True)
        assert result.stats["furniture_masked_blocks"] == 0
        assert self._greenish(result) > 0

    def test_without_observed_pixels_nothing_is_masked(self):
        frame, _ = self._scene()
        result = _homography([frame], furniture_mask=True)   # check_points_uv is None
        assert result.stats["furniture_masked_blocks"] == 0
        assert self._greenish(result) > 0

    def test_the_mask_is_opt_in(self):
        frame, uv = self._scene()
        frame.check_points_uv = uv
        result = _homography([frame], width=512, height=512)
        assert result.stats["furniture_mask"] is False
        assert result.stats["furniture_masked_blocks"] == 0
