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
