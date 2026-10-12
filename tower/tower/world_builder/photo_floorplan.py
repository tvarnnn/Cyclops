"""FOW v1.2 slice 1: the photo floor-plan raster, pure core.

Two paths share the basis, bounds, orientation, per-cell selection, alpha
and encoding rules below: option (2), the depth-projected raster
(`build_raster`, documented in this docstring), and fallback (3), the
lower-image floor homography (`estimate_floor_plane` +
`build_homography_raster`, documented at its own section further down).

Design: ``FOW-V12-PHOTO-FLOORPLAN-DESIGN-20261012.md``, "Raster and evidence".
Wire: ``303-v12-wire-contract.md`` (the geometry this module must match exactly).

This module does no I/O, touches no Tower state and makes no network calls.
Every input is a plain value: one component's basis and (unpadded) bounds, as
published by ``guidance_coverage``'s v8 receipt, and a list of keyframe
samples that are *already* undistorted, *already* depth-fit to this solve's
SfM observations (metres, in the solve's local/un-placed scale) and already
placed (camera pose expressed in the SAME output/world frame as the
component basis). Fitting the depth to SfM, reading the cache, and loading
images are all somebody else's job -- the CLI's.

GEOMETRY CONVENTIONS (must match the coverage receipt and the wire exactly):

  - ``right = forward x up`` (component basis), unit.
  - A world point P projects to raster coordinates
    ``x = P . forward``, ``y = P . right`` -- the SAME absolute coordinates
    as v8's ``bounds_xy``. Column 0 = min_x (left), row 0 = min_y (top),
    increasing right / down.
  - ``raster_bounds_xy`` is the *padded* (15%) SQUARE footprint: the v8
    receipt's ``bounds_xy`` is squared (side = max width/height) about its
    own centre, then padded, matching the design's "solved camera-centre
    square, padded 15%". This is NOT v8's unpadded ``bounds_xy``.
  - Each camera's own axes (``forward_xyz``/``up_xyz`` on a
    ``KeyframeSample``) are in the SAME output frame as the component basis
    (that is what ``guidance_coverage._placed`` already returns): camera
    local +Z -> ``forward_xyz``, camera local -Y -> ``up_xyz``. The camera's
    local +X ("right") is recovered as ``forward_xyz x up_xyz`` (cyclic
    y x z = x for a right-handed x=right, y=down, z=forward camera frame,
    with down = -up).
  - A camera-local point ``(x, y, z) = ray * depth`` reaches the output frame
    as ``position + scale * (x * right - y * up + z * forward)``, where
    ``scale`` is the placement Sim3 scale (local metres -> output units);
    rotation-only directions (surface normals) use the same combination
    without the ``scale`` factor and without ``position``.

GATES (design "Raster and evidence", starting values, unvalidated on a real
walk -- see the module docstring's thresholds section for exactly which ones
this probe uses and why):

  - reject a pixel with no finite, positive depth, or marked invalid (fill,
    cache miss, failed per-frame fit);
  - reject a pixel outside the padded bounds;
  - reject a surface whose normal is less than ``normal_up_min`` (design:
    0.6) aligned with the component's ``up`` -- keeps horizontal surfaces,
    drops walls;
  - admit a horizontal surface only between a robust floor-height estimate
    (a low percentile of admitted heights, minus a margin for estimation
    noise) and the median camera height plus a headroom fraction of the
    camera-to-floor height -- keeps the floor and tabletops, drops ceilings;
  - a raster cell is opaque only with >= 2 distinct contributing keyframes,
    or one non-grazing view whose local 3x3 depth patch is height-consistent;
  - per cell, keep the highest admitted surface (floor vs. a tabletop above
    it), breaking ties by lower viewing obliqueness then a newer keyframe.

No dilation, no hole filling: a cell with no admitted sample is alpha 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9
MAX_DIM = 512
DEFAULT_PAD_FRAC = 0.15

# -- design "starting values, unvalidated on walk 9" ------------------------
NORMAL_UP_MIN = 0.6
"""A surface whose normal is less than this aligned with `up` is not floor
or tabletop; it is rejected before it can paint a cell (design section
"Raster and evidence", paragraph 2)."""

FLOOR_HEIGHT_PERCENTILE = 5.0
"""The floor height is read off the low tail of admitted horizontal-surface
heights rather than the bare minimum, so one stray low point (noise, a
back-projected speck under the true floor) cannot relocate the whole plane."""

FLOOR_HEIGHT_MARGIN_FRAC = 0.05
"""Slack below the estimated floor height, as a fraction of camera-to-floor
height, so the floor estimate's own noise does not clip the floor it just
estimated."""

FLOOR_ADMIT_HEADROOM_FRAC = 0.20
"""Horizontal surfaces are admitted up to camera height plus this fraction of
camera-to-floor height (design: "no higher than the median camera height
plus 20% of estimated camera-to-floor height"), which keeps tabletops and
rejects ceilings."""

GRAZING_COS_MIN = 0.35
"""A single-view admission requires a "non-grazing" view: the cosine between
the viewing ray and the surface normal must be at least this. (The design
names the requirement without a number; this probe picks one and says so.)"""

PATCH_CONSISTENCY_FRAC = 0.05
"""A single-view admission also requires its local 3x3 depth patch (in the
KEYFRAME's own pixel grid, not the raster) to agree in height to within this
fraction of camera-to-floor height."""


@dataclass
class KeyframeSample:
    """One eligible, posed, cache-hit keyframe, already fit to this solve.

    All geometry is in the OUTPUT (component/world) frame except `depth`,
    which is this keyframe's own camera-frame depth in metres (already fit;
    NOT yet placed -- `scale`/`position`/`forward_xyz`/`up_xyz` do that).
    """

    kid: str
    position_xyz: tuple
    forward_xyz: tuple
    up_xyz: tuple
    scale: float
    fx: float
    fy: float
    cx: float
    cy: float
    depth: np.ndarray       # (h, w) float, metres; <=0 / non-finite = invalid
    valid: np.ndarray       # (h, w) bool: usable (not fill, cache-valid, in frame)
    color: np.ndarray       # (h, w, 3) uint8 RGB
    order: int = 0          # higher = newer; a tie-break only


@dataclass
class RasterResult:
    rgba: np.ndarray
    raster_bounds_xy: tuple
    width: int
    height: int
    opaque_cells: int
    total_cells: int
    multi_view_cells: int
    single_view_cells: int
    contributing_keyframes: list
    stats: dict = field(default_factory=dict)

    @property
    def opaque_fraction(self) -> float:
        return self.opaque_cells / self.total_cells if self.total_cells else 0.0


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < EPS:
        raise ValueError("degenerate basis vector")
    return v / n


def padded_square_bounds(bounds_xy, pad_frac: float = DEFAULT_PAD_FRAC):
    """The design's "solved camera-centre square, padded 15%": square the v8
    receipt's bounds about their own centre (side = max width/height), then
    pad every side by `pad_frac` of that side."""
    min_x, min_y, max_x, max_y = (float(v) for v in bounds_xy)
    if not all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y)):
        raise ValueError("nonfinite bounds")
    side = max(max_x - min_x, max_y - min_y, EPS)
    mid_x, mid_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    half = side * (1.0 + 2.0 * pad_frac) / 2.0
    return (mid_x - half, mid_y - half, mid_x + half, mid_y + half)


def _paint_cells(rgba, rows, cols, colors, multi_view):
    """The shared presentation rule for both paths: a cell seen by >= 2
    distinct keyframes keeps its camera colour; a single-view cell is pulled
    40% toward mid-gray ("uncertain one-view pixels at reduced saturation").
    Painted cells get alpha 255; every other cell keeps alpha 0 (gray)."""
    colors = np.asarray(colors, dtype=np.float64)
    desaturated = colors * 0.6 + 128.0 * 0.4
    final_colors = np.where(np.asarray(multi_view, bool)[:, None], colors, desaturated)
    rgba[rows, cols, 0:3] = np.clip(final_colors, 0, 255).astype(np.uint8)
    rgba[rows, cols, 3] = 255


def _empty_result(bounds, width, height):
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    return RasterResult(rgba=rgba, raster_bounds_xy=tuple(float(v) for v in bounds),
                        width=width, height=height, opaque_cells=0,
                        total_cells=width * height, multi_view_cells=0,
                        single_view_cells=0, contributing_keyframes=[],
                        stats={"reason": "no eligible keyframes"})


def build_raster(*, origin_xyz, up_xyz, forward_xyz, bounds_xy, keyframes,
                 width: int = MAX_DIM, height: int = MAX_DIM,
                 pad_frac: float = DEFAULT_PAD_FRAC,
                 normal_up_min: float = NORMAL_UP_MIN,
                 floor_height_percentile: float = FLOOR_HEIGHT_PERCENTILE,
                 floor_height_margin_frac: float = FLOOR_HEIGHT_MARGIN_FRAC,
                 floor_admit_headroom_frac: float = FLOOR_ADMIT_HEADROOM_FRAC,
                 grazing_cos_min: float = GRAZING_COS_MIN,
                 patch_consistency_frac: float = PATCH_CONSISTENCY_FRAC,
                 check=None) -> RasterResult:
    """Build one component's RGBA photo raster from already-fit, posed samples.

    `origin_xyz` is accepted (it is part of the receipt's basis and the wire
    contract requires components to carry it) but, like `guidance_coverage`,
    is not used in the x/y projection itself -- the wire's formula is an
    absolute dot product, not an offset from `origin_xyz`.
    """
    width = max(1, min(int(width), MAX_DIM))
    height = max(1, min(int(height), MAX_DIM))
    up = _unit(up_xyz)
    forward = _unit(forward_xyz)
    right = np.cross(forward, up)
    rn = float(np.linalg.norm(right))
    if rn < EPS:
        raise ValueError("degenerate component basis: forward parallel to up")
    right = right / rn
    bounds = padded_square_bounds(bounds_xy, pad_frac)
    min_x, min_y, max_x, max_y = bounds
    span_x, span_y = max(max_x - min_x, EPS), max(max_y - min_y, EPS)

    if not keyframes:
        return _empty_result(bounds, width, height)

    # -- pass 1: per keyframe, back-project every pixel into the component
    # plane and keep the ones with a horizontal-aligned normal.
    cand_h, cand_x, cand_y, cand_ob, cand_kid, cand_order = [], [], [], [], [], []
    cand_rgb, cand_row_px, cand_col_px, cand_ki = [], [], [], []
    cam_heights = []
    per_kf_grids = []  # (h_val, mask, obliqueness) per keyframe index; pass 3's patch check
    for ki, kf in enumerate(keyframes):
        if check is not None:
            check()
        try:
            kf_up = _unit(kf.up_xyz)
            kf_forward = _unit(kf.forward_xyz)
        except ValueError:
            continue
        kf_right = np.cross(kf_forward, kf_up)
        rn2 = float(np.linalg.norm(kf_right))
        if rn2 < EPS:
            continue
        kf_right = kf_right / rn2
        down = -kf_up
        position = np.asarray(kf.position_xyz, dtype=np.float64)
        scale = float(kf.scale)
        if not math.isfinite(scale) or scale <= 0:
            continue
        cam_heights.append(float(np.dot(position, up)))

        depth = np.asarray(kf.depth, dtype=np.float64)
        valid = np.asarray(kf.valid, dtype=bool)
        h_px, w_px = depth.shape
        if valid.shape != depth.shape or h_px < 3 or w_px < 3:
            continue
        uu, vv = np.meshgrid(np.arange(w_px, dtype=np.float64),
                             np.arange(h_px, dtype=np.float64))
        finite = np.isfinite(depth) & (depth > 1e-6)
        ok = valid & finite
        x_local = (uu - kf.cx) / kf.fx * depth
        y_local = (vv - kf.cy) / kf.fy * depth
        z_local = depth

        # World point for every pixel (cheap: it is just a linear combination).
        world_x = (position[0] + scale * (x_local * kf_right[0] + y_local * down[0]
                                          + z_local * kf_forward[0]))
        world_y = (position[1] + scale * (x_local * kf_right[1] + y_local * down[1]
                                          + z_local * kf_forward[1]))
        world_z = (position[2] + scale * (x_local * kf_right[2] + y_local * down[2]
                                          + z_local * kf_forward[2]))

        # Local surface normal from finite differences of the LOCAL camera-frame
        # point (translation/scale cancel in a difference), one-sided at the
        # last row/col so every pixel gets an estimate.
        px = np.stack([x_local, y_local, z_local], axis=-1)
        dpdu = np.empty_like(px)
        dpdu[:, :-1] = px[:, 1:] - px[:, :-1]
        dpdu[:, -1] = dpdu[:, -2]
        dpdv = np.empty_like(px)
        dpdv[:-1, :] = px[1:, :] - px[:-1, :]
        dpdv[-1, :] = dpdv[-2, :]
        normal_local = np.cross(dpdu, dpdv)
        nnorm = np.linalg.norm(normal_local, axis=-1)
        safe_nnorm = np.where(nnorm > 1e-9, nnorm, 1.0)
        normal_local = normal_local / safe_nnorm[..., None]
        ray = np.stack([(uu - kf.cx) / kf.fx, (vv - kf.cy) / kf.fy,
                        np.ones_like(uu)], axis=-1)
        ray_norm = np.linalg.norm(ray, axis=-1)
        # cos(angle) between the surface normal and the viewing ray: 1 is
        # head-on (least grazing), 0 is edge-on (most grazing).
        obliqueness = np.abs((normal_local * ray).sum(-1)) / np.maximum(ray_norm, 1e-9)

        # Normal in world/component frame: same rotation-only combination.
        nx, ny, nz = normal_local[..., 0], normal_local[..., 1], normal_local[..., 2]
        world_nx = nx * kf_right[0] + ny * down[0] + nz * kf_forward[0]
        world_ny = nx * kf_right[1] + ny * down[1] + nz * kf_forward[1]
        world_nz = nx * kf_right[2] + ny * down[2] + nz * kf_forward[2]
        dot_up = world_nx * up[0] + world_ny * up[1] + world_nz * up[2]
        horizontal = np.abs(dot_up) >= normal_up_min

        h_val = world_x * up[0] + world_y * up[1] + world_z * up[2]
        x_comp = world_x * forward[0] + world_y * forward[1] + world_z * forward[2]
        y_comp = world_x * right[0] + world_y * right[1] + world_z * right[2]

        mask = ok & horizontal & np.isfinite(h_val) & np.isfinite(x_comp) & np.isfinite(y_comp)
        kf_index = len(per_kf_grids)
        per_kf_grids.append((h_val, mask, obliqueness))
        if not mask.any():
            continue
        rows_px, cols_px = np.nonzero(mask)
        cand_h.append(h_val[mask])
        cand_x.append(x_comp[mask])
        cand_y.append(y_comp[mask])
        cand_ob.append(obliqueness[mask])
        n_sel = int(mask.sum())
        cand_kid.extend([kf.kid] * n_sel)
        cand_order.extend([int(kf.order)] * n_sel)
        cand_row_px.append(rows_px)
        cand_col_px.append(cols_px)
        cand_ki.extend([kf_index] * n_sel)
        color = np.asarray(kf.color)
        cand_rgb.append(color[mask])

    if not cand_h:
        return _empty_result(bounds, width, height)

    h_all = np.concatenate(cand_h)
    x_all = np.concatenate(cand_x)
    y_all = np.concatenate(cand_y)
    ob_all = np.concatenate(cand_ob)
    order_all = np.asarray(cand_order, dtype=np.int64)
    kid_all = np.asarray(cand_kid)
    rgb_all = np.concatenate(cand_rgb, axis=0)
    row_px_all = np.concatenate(cand_row_px)
    col_px_all = np.concatenate(cand_col_px)
    ki_all = np.asarray(cand_ki, dtype=np.int64)

    # -- pass 2: robust floor height and the admission band.
    floor_height = float(np.percentile(h_all, floor_height_percentile))
    median_cam_height = (float(np.median(cam_heights)) if cam_heights
                         else float(np.median(h_all)))
    cam_to_floor = max(median_cam_height - floor_height, EPS)
    lower = floor_height - floor_height_margin_frac * cam_to_floor
    upper = median_cam_height + floor_admit_headroom_frac * cam_to_floor

    in_bounds = ((x_all >= min_x) & (x_all <= max_x) &
                (y_all >= min_y) & (y_all <= max_y))
    admitted = in_bounds & (h_all >= lower) & (h_all <= upper)
    if not admitted.any():
        return _empty_result(bounds, width, height)

    col = np.clip(np.round((x_all - min_x) / span_x * (width - 1)), 0, width - 1).astype(np.int64)
    row = np.clip(np.round((y_all - min_y) / span_y * (height - 1)), 0, height - 1).astype(np.int64)
    flat = row * width + col

    sel = np.nonzero(admitted)[0]
    flat_s, h_s, ob_s, order_s = flat[sel], h_all[sel], ob_all[sel], order_all[sel]
    kid_s, rgb_s = kid_all[sel], rgb_all[sel]
    row_px_s, col_px_s, ki_s = row_px_all[sel], col_px_all[sel], ki_all[sel]

    # -- 3x3 patch consistency, per ADMITTED candidate (post height-band/bounds
    # filtering), from that pixel's own keyframe grid. Grouped by keyframe so
    # each lookup stays a cheap vectorised gather.
    patch_ok = np.zeros(len(sel), dtype=bool)
    for kf_index, (h_grid, mask_grid, ob_grid) in enumerate(per_kf_grids):
        in_this_kf = ki_s == kf_index
        if not in_this_kf.any():
            continue
        rows_px, cols_px = row_px_s[in_this_kf], col_px_s[in_this_kf]
        H, W = h_grid.shape
        local_patch_ok = np.zeros(len(rows_px), dtype=bool)
        for shift_r, shift_c in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = np.clip(rows_px + shift_r, 0, H - 1)
            nc = np.clip(cols_px + shift_c, 0, W - 1)
            neighbor_h = h_grid[nr, nc]
            neighbor_valid = mask_grid[nr, nc]
            diff_ok = neighbor_valid & (np.abs(neighbor_h - h_grid[rows_px, cols_px])
                                        <= patch_consistency_frac * cam_to_floor)
            local_patch_ok |= diff_ok
        candidate_grazing_ok = ob_grid[rows_px, cols_px] >= grazing_cos_min
        patch_ok[in_this_kf] = local_patch_ok & candidate_grazing_ok

    # -- grouping: best admitted sample per cell, and distinct-keyframe count.
    group_ids, group_of = np.unique(flat_s, return_inverse=True)
    n_groups = len(group_ids)
    kid_codes, kid_of = np.unique(kid_s, return_inverse=True)
    pair = np.stack([group_of, kid_of], axis=1)
    unique_pairs = np.unique(pair, axis=0)
    distinct_count = np.bincount(unique_pairs[:, 0], minlength=n_groups)
    patch_ok_per_group = np.zeros(n_groups, dtype=bool)
    np.logical_or.at(patch_ok_per_group, group_of, patch_ok)
    eligible_group = (distinct_count >= 2) | patch_ok_per_group

    # `ob_s` is cos(angle) between the surface normal and the viewing ray: 1 is
    # head-on, 0 is grazing. "Lower obliqueness" (design wording) means a more
    # head-on view, i.e. a HIGHER cosine, hence `-ob_s` here.
    sort_idx = np.lexsort((-order_s, -ob_s, -h_s, group_of))
    sorted_group = group_of[sort_idx]
    first_mask = np.ones(len(sorted_group), dtype=bool)
    first_mask[1:] = sorted_group[1:] != sorted_group[:-1]
    best_local_idx = sort_idx[first_mask]
    best_group = sorted_group[first_mask]

    keep = eligible_group[best_group]
    best_local_idx = best_local_idx[keep]
    best_group = best_group[keep]
    multi_view = distinct_count[best_group] >= 2

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rows_out = group_ids[best_group] // width
    cols_out = group_ids[best_group] % width
    _paint_cells(rgba, rows_out, cols_out, rgb_s[best_local_idx], multi_view)

    contributing = sorted({str(k) for k in kid_s[best_local_idx]})
    stats = {
        "floor_height": floor_height,
        "median_camera_height": median_cam_height,
        "camera_to_floor_height": cam_to_floor,
        "admit_band": [lower, upper],
        "candidate_pixels": int(len(h_all)),
        "admitted_pixels": int(admitted.sum()),
    }
    return RasterResult(rgba=rgba, raster_bounds_xy=tuple(float(v) for v in bounds),
                        width=width, height=height,
                        opaque_cells=int(len(best_group)), total_cells=width * height,
                        multi_view_cells=int(multi_view.sum()),
                        single_view_cells=int((~multi_view).sum()),
                        contributing_keyframes=contributing, stats=stats)


# ===========================================================================
# Fallback (3): the lower-image FLOOR HOMOGRAPHY.
#
# Design: "Decision" table, row "Fallback (3)", plus the shared "Raster and
# evidence" rules (basis, padded square bounds, column/row orientation,
# alpha 0 = unseen, >= 2 keyframes or one non-grazing view, single-view
# desaturation -- all reused from option (2) above, not forked).
#
# No depth model, no GPU. Per displayed registered component:
#
#   1. `estimate_floor_plane`: ONE plane from the lower envelope of the
#      solve's own 3-D points (already in the component's output frame),
#      cross-checked against the camera centres (every camera above it, at a
#      consistent height). A plane without adequate support, or one the
#      cameras contradict, is REJECTED with a reason -- the component then
#      stays all gray.
#   2. `build_homography_raster`: for each downward-looking, rectified frame,
#      every raster cell is mapped through the plane back into the frame
#      (inverse warp: plane point -> camera -> pixel, i.e. the plane-induced
#      homography) and sampled only if it lands in the BOTTOM 40% of the
#      image and passes the horizon / incidence / range gates. Inverse
#      mapping is used (not pixel splatting) so a 144x256 sample can fill a
#      512x512 raster without speckle; it also makes the bounds gate
#      structural (a hit outside the padded bounds has no cell).
#   3. Per cell: the existing option-(2) selection -- the most head-on view
#      wins (ties: newer keyframe), opaque with >= 2 distinct keyframes or
#      one view at least GRAZING_COS_MIN head-on; alpha 0 elsewhere.
#
# Heights here are SIGNED DISTANCES TO THE PLANE (`normal . p + offset`), not
# `p . up`: the receipt's `up` is the mean camera up axis, which on walk 8 is
# 3-13 degrees off the camera-centre plane's normal (cameras pitch down on
# average), and over a 16-unit room that tilt alone is a ~2-unit height
# error. The raster's x/y are still exactly the receipt's `p . forward`,
# `p . (forward x up)`.
#
# All distance thresholds are FRACTIONS OF THE MEDIAN CAMERA-TO-PLANE HEIGHT:
# the solve is unitless (GLOMAP scale, placement scale 1.0 on walk 8), so an
# absolute metre threshold would be meaningless.
# ===========================================================================

# -- floor-plane estimation -------------------------------------------------
FLOOR_PLANE_MAX_TILT_DEG = 10.0
"""A floor hypothesis may tilt at most this far from the reference normal
(the camera-centre plane's normal when that is well conditioned, else the
receipt's `up`). A hand-held walk keeps the phone at a roughly constant
height, so its centres span a plane parallel to the floor; 10 degrees admits
real height drift without admitting a wall."""

FLOOR_INLIER_TOL_FRAC = 0.06
"""A point is ON the plane within this fraction of the median camera height
(~8 cm at a 1.35 m camera height): about SfM point noise on carpet, and well
under a bed/sofa top (~0.3-0.5 of camera height)."""

FLOOR_BELOW_MAX_FRAC = 0.10
"""Lower envelope: at most this fraction of the candidate points may lie
more than one tolerance BELOW the plane. Without an allowance one bad track
pushes the plane down; with too large an allowance a tabletop could win with
the real floor 'below' it."""

FLOOR_MIN_INLIERS = 40
"""Minimum number of 3-D points on the plane ("inadequate support")."""

FLOOR_SUPPORT_GRID = 8
FLOOR_MIN_SUPPORT_CELLS = 3
"""The inliers must be spread over at least 3 cells of an 8x8 grid on the
padded bounds (>= 3 inliers each): one box top in a corner is not a floor."""

FLOOR_RANSAC_HYPOTHESES = 512
FLOOR_RANSAC_SCORE_POINTS = 4000
"""Random 3-point hypotheses (deterministic seed) plus 61 level planes along
the reference normal; scored on a deterministic subsample of at most 4000
candidate points to stay inside the CPU budget."""

CAMERA_ABOVE_MIN_FRAC = 0.95
"""At least this fraction of the component's cameras must be above the plane."""

CAMERA_HEIGHT_MAX_CV = 0.25
"""Robust coefficient of variation (1.4826 * MAD / median) of the camera
heights above the plane. A hand-held walk keeps the camera within roughly
+-25% of its median height; a wrongly TILTED plane makes the heights trend
across the room and fails this."""

CAMERA_PLANE_MIN_SPREAD = 0.05
CAMERA_PLANE_MAX_FLATNESS = 0.25
CAMERA_PLANE_MAX_TILT_DEG = 25.0
"""The camera-centre plane is the reference normal only if the centres span
two dimensions (second/first eigenvalue >= 0.05), are flat (third/second
<= 0.25) and lie within 25 degrees of the receipt `up`; a walk along one line
leaves its plane normal undetermined about that line."""

FLOOR_PLANE_MAX_TILT_LOOSE_DEG = 20.0
"""The tilt window when neither the camera-centre plane NOR the roll-free vertical is
usable (only the receipt `up` is left). Original rationale: when the camera-centre plane is NOT usable (a partial live
trajectory: walk 8 / e2e2 landings 0-2 had spread 0.03-0.05, flatness 0.70 or a
min-variance axis 54 deg off up). The fallback references are biased: the receipt
`up` (mean camera up axis) sat 10-13 deg off the camera plane on e2e2, the roll-free
vertical 7-8 deg. A 10 deg window could exclude the true floor; 20 deg holds it,
and the camera-height consistency check still rejects a wrongly tilted plane."""

FLOOR_POINTS_BUDGET = 20000
NEAREST_CAMERA_BUDGET = 256
"""Cost bounds for `estimate_floor_plane`, flat in walk length: at most 20k of the
component's points (a deterministic subsample, seed 0) and at most 256 cameras (an
even stride along the walk) for the nearest-camera candidate test. The 4000-point
RANSAC scoring subsample is drawn from these."""

DECOY_LAYER_MIN_POINTS = 10
DECOY_LAYER_MIN_FRACTION = 0.5
DECOY_LAYER_MIN_CELLS = 3
DECOY_LAYER_MIN_BREADTH = 0.5
DECOY_MAX_ROUNDS = 4
"""Furniture DECOY rule for the LOOSE path (no usable camera plane; review MED-2).
The lower-envelope allowance lets up to 10% of the candidates lie below a plane, so
a long, dense table can beat a sparsely textured floor. On the loose path a plane is
refused when the points more than one tolerance below it form a LAYER -- >= 10
points, >= half of them within one tolerance of their median height, spread over
>= 3 cells of the 8x8 support grid AND over >= half as many cells as the plane's own
inliers (a floor is broader than the furniture on it; a small lower patch under a
real floor -- e2e2 kf 353: ~23 points in a few cells under a 13-cell floor -- is not
a decoy) -- evaluated over every hypothesis on the
scoring subsample, and again on the REFINED plane against all candidates (a tilted
hypothesis can hide the floor's layer; its least-squares refinement cannot). A refused
height excludes every hypothesis within three tolerances of it; at most 4 rounds,
then the plane is refused. Scattered noise below a floor is not a layer. On the camera-plane path
the rule is not applied: there a layer below is walk 8's misplaced closet, and the
room floor above it is the right answer (PROBE-REPORT p2probe-h)."""

ROLL_FREE_MIN_SPREAD = 0.05
ROLL_FREE_MAX_FLATNESS = 0.1
ROLL_FREE_MAX_TILT_DEG = 30.0
"""Roll-free vertical: a hand-held / head-worn camera has ~zero roll, so the
vertical is perpendicular to every camera RIGHT axis -- the least-variance
direction of the right vectors. Used when the right axes span the horizontal
(second/first eigenvalue >= 0.05, third/second <= 0.1) and it is within 30 deg
of the receipt `up`. It does not depend on the trajectory's shape."""

FLOOR_AMBIGUITY_RATIO = 0.5
"""Reported, not acted on: a second VALID plane more than three tolerances
away with at least half the chosen plane's support marks the estimate
`ambiguous` (e.g. a mis-placed closet floor below the room floor)."""

# -- warp ---------------------------------------------------------------------
LOWER_IMAGE_FRACTION = 0.40
"""Only the bottom 40% of each rectified image is warped (design)."""

MIN_FRAME_PITCH_DEG = 15.0
"""A frame is "downward-looking" if its optical axis is at least this far
below the plane. With walk 8's ~69 degree vertical field of view the bottom
40% band of such a frame spans depressions of roughly 22-49 degrees, i.e.
mostly inside the incidence gate."""

HORIZON_MIN_DEPRESSION_DEG = 10.0
"""Reject rays less than 10 degrees below the plane's horizon (the hit is
then > 5.7 camera heights away and a 1 degree pose error moves it by more
than half a camera height). On a single plane this equals an incidence of
80 degrees, so with MAX_INCIDENCE_DEG = 75 it only guards the far tail; it is
counted separately so a report can see whether it ever binds."""

MAX_INCIDENCE_DEG = 75.0
"""Reject plane hits whose ray meets the floor more than 75 degrees from its
normal: a pixel's footprint is stretched > 3.9x there. Between ~69.5 and 75
degrees (cos 0.26-0.35) a cell is admitted only if a second keyframe also
sees it (GRAZING_COS_MIN is the existing single-view rule)."""

MAX_RANGE_FRAC = 4.0
"""Reject plane hits farther than 4 MEDIAN camera heights from the camera.
For a camera at the median height the incidence gate binds first (3.86); this
catches a frame whose OWN height above the plane is anomalous (pose error)."""

FRAME_BELOW_PLANE_TOL_FRAC = 0.12
FRAME_BELOW_PLANE_MAX_FRAC = 0.5
FRAME_MIN_CHECK_POINTS = 5
"""Per-frame pose cross-check: if more than half of the SfM points THIS frame
observed inside its own bottom-40% band lie more than 2 inlier tolerances
BELOW the plane, the frame's pose and the plane disagree (nothing visible is
below a floor) and the frame is rejected. Points above the plane are allowed
(furniture); the opt-in furniture mask below can remove some of them."""

FURNITURE_MASK_DEFAULT = False
FURNITURE_ABOVE_TOL_FRAC = 0.12
FURNITURE_BLOCK_PX = 16
"""Bottom-band FURNITURE MASK (OPT-IN, default OFF; Phase B item 6). Before/after on
walk 8 (p2probe-h/maskband/before_after_side_by_side.jpg) was NOT a clear improvement:
it removed most of R0's closet-clothes smear but cut blocky, staircase holes, left the
desk and bookshelf smears largely in place, and changed G2 little -- so, per the lead's
rule (mail 308), it is not on by default. The bottom 40% of most
walk-8 downward frames shows furniture, not floor (R0: a median 3% of a frame's own
band points lie on the plane), and a homography flattens it onto the floor. Each
frame's band is cut into 16x16-pixel blocks at the <= 256x144 sample size (about
40x40 at walk 8's full resolution: SfM points are sparse, so one observation has to
speak for its neighbourhood). A block is MASKED when its own SfM observations that
sit clearly above the plane (more than 0.12 H: the pose gate's tolerance, about a
16 cm step at H = 1.35 m) outnumber those on it (within 0.12 H). A block with no
observation is kept: carpet is low-texture, so silence is not evidence of
furniture. Masked cells simply contribute nothing (they may still be painted by
another frame that sees floor there)."""


@dataclass
class FloorPlane:
    """`normal . p + offset` is the signed height of `p` above the floor."""

    ok: bool
    reason: str
    normal: tuple = None
    offset: float = None
    stats: dict = field(default_factory=dict)


@dataclass
class PhotoFrame:
    """One eligible posed keyframe for the homography path.

    Pose in the component's OUTPUT frame exactly as `guidance_coverage._placed`
    returns it (camera +Z -> `forward_xyz`, camera -Y -> `up_xyz`). `color` is
    the RECTIFIED (undistorted) image, already sampled at <= 256x144, and
    `fx/fy/cx/cy` are its intrinsics at that sample size. `check_points_xyz`
    (optional) are the solve's 3-D points this keyframe observed inside its
    own bottom-40% band, in the output frame (the per-frame pose gate), and
    `check_points_uv` (optional, same order) their observed pixel positions AT
    THE SAMPLE SIZE (the furniture mask)."""

    kid: str
    position_xyz: tuple
    forward_xyz: tuple
    up_xyz: tuple
    fx: float
    fy: float
    cx: float
    cy: float
    color: np.ndarray
    order: int = 0
    check_points_xyz: np.ndarray = None
    check_points_uv: np.ndarray = None


def _furniture_blocks(frame, n, d, h_med, h_px, w_px, block_px, above_tol_frac):
    """(masked-block boolean grid, number of masked blocks) for one frame, or None."""
    if frame.check_points_xyz is None or frame.check_points_uv is None:
        return None
    pts = np.asarray(frame.check_points_xyz, dtype=np.float64).reshape(-1, 3)
    uv = np.asarray(frame.check_points_uv, dtype=np.float64).reshape(-1, 2)
    if len(pts) != len(uv) or len(pts) == 0:
        return None
    ok = np.all(np.isfinite(pts), axis=1) & np.all(np.isfinite(uv), axis=1)
    pts, uv = pts[ok], uv[ok]
    rows = -(-h_px // block_px)
    cols = -(-w_px // block_px)
    bu = np.clip(np.floor((uv[:, 0] + 0.5) / block_px).astype(int), 0, cols - 1)
    bv = np.clip(np.floor((uv[:, 1] + 0.5) / block_px).astype(int), 0, rows - 1)
    r = pts @ n + d
    tol = above_tol_frac * h_med
    above = np.zeros((rows, cols), np.int32)
    on = np.zeros((rows, cols), np.int32)
    np.add.at(above, (bv[r > tol], bu[r > tol]), 1)
    np.add.at(on, (bv[np.abs(r) <= tol], bu[np.abs(r) <= tol]), 1)
    masked = (above > 0) & (above > on)
    return masked, int(masked.sum())


def _reject_plane(reason, stats):
    stats = dict(stats)
    stats["reason"] = reason
    return FloorPlane(ok=False, reason=reason, stats=stats)


def _camera_plane_reference(cameras, up):
    """The reference normal: the camera-centre plane if well conditioned."""
    info = {"reference_normal": "receipt_up"}
    if len(cameras) < 3:
        return up, info
    centred = cameras - cameras.mean(axis=0)
    w, v = np.linalg.eigh(centred.T @ centred)  # ascending
    if not np.all(np.isfinite(w)) or w[2] <= EPS:
        return up, info
    normal = v[:, 0] if float(v[:, 0] @ up) >= 0 else -v[:, 0]
    spread = float(w[1] / w[2])
    flatness = float(w[0] / max(w[1], EPS))
    tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(normal @ up)))))
    info.update(camera_plane_spread=spread, camera_plane_flatness=flatness,
                camera_plane_tilt_from_up_deg=tilt)
    if (spread >= CAMERA_PLANE_MIN_SPREAD and flatness <= CAMERA_PLANE_MAX_FLATNESS
            and tilt <= CAMERA_PLANE_MAX_TILT_DEG):
        info["reference_normal"] = "camera_plane"
        return normal, info
    return up, info


def _roll_free_reference(camera_rights, up):
    """(normal, info) of the roll-free vertical, or (None, info) if it is not usable."""
    info = {}
    if camera_rights is None:
        return None, info
    r = np.asarray(camera_rights, dtype=np.float64).reshape(-1, 3)
    norms = np.linalg.norm(r, axis=1)
    r = r[np.isfinite(norms) & (norms > EPS)]
    if len(r) < 3:
        return None, info
    r = r / np.linalg.norm(r, axis=1)[:, None]
    w, v = np.linalg.eigh(r.T @ r)
    normal = v[:, 0] if float(v[:, 0] @ up) >= 0 else -v[:, 0]
    spread = float(w[1] / max(w[2], EPS))
    flatness = float(w[0] / max(w[1], EPS))
    tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(normal @ up)))))
    info.update(roll_free_spread=spread, roll_free_flatness=flatness,
                roll_free_tilt_from_up_deg=tilt)
    if (spread >= ROLL_FREE_MIN_SPREAD and flatness <= ROLL_FREE_MAX_FLATNESS
            and tilt <= ROLL_FREE_MAX_TILT_DEG):
        return normal, info
    return None, info


def _below_nearest_camera(points, cameras, n_ref, chunk=4096):
    """Each point's height minus the height of the camera nearest to it ACROSS the
    reference normal (horizontally). A uniform tilt of the reference cancels locally,
    which a single global camera-height cut does not on a long, tilted trajectory."""
    a = np.cross(n_ref, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 0.1:
        a = np.cross(n_ref, [0.0, 1.0, 0.0])
    a = a / np.linalg.norm(a)
    b = np.cross(n_ref, a)
    cam_xy = np.stack([cameras @ a, cameras @ b], axis=1)
    cam_h = cameras @ n_ref
    pt_xy = np.stack([points @ a, points @ b], axis=1)
    pt_h = points @ n_ref
    out = np.empty(len(points))
    for i in range(0, len(points), chunk):
        d = ((pt_xy[i:i + chunk, None, :] - cam_xy[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = pt_h[i:i + chunk] - cam_h[np.argmin(d, axis=1)]
    return out


def _score_planes(points, cameras, normals, offsets, n_ref, *, max_tilt_deg, tol_frac,
                  below_max_frac, camera_above_min_frac, camera_height_max_cv):
    """Vectorised constraint check and inlier count for many plane hypotheses."""
    cam_h = cameras @ normals.T + offsets[None, :]                 # (ncam, nh)
    h_med = np.median(cam_h, axis=0)
    safe_med = np.where(h_med > EPS, h_med, 1.0)
    tol = tol_frac * safe_med
    res = points @ normals.T + offsets[None, :]                    # (npts, nh)
    inliers = (np.abs(res) <= tol[None, :]).sum(axis=0)
    below = (res < -tol[None, :]).mean(axis=0)
    above = (cam_h > 0).mean(axis=0)
    cv = 1.4826 * np.median(np.abs(cam_h - h_med[None, :]), axis=0) / safe_med
    tilt_ok = normals @ n_ref >= math.cos(math.radians(max_tilt_deg)) - 1e-12
    checks = {
        "tilt": tilt_ok,
        "camera_above": (h_med > EPS) & (above >= camera_above_min_frac),
        "lower_envelope": below <= below_max_frac,
        "camera_height_consistency": cv <= camera_height_max_cv,
    }
    valid = np.ones(len(offsets), dtype=bool)
    for ok in checks.values():
        valid &= ok
    return dict(inliers=inliers, below=below, above=above, cv=cv, h_med=h_med,
                checks=checks, valid=valid)


def estimate_floor_plane(*, points_xyz, camera_positions, up_xyz, forward_xyz, bounds_xy,
                         pad_frac: float = DEFAULT_PAD_FRAC, seed: int = 0,
                         max_tilt_deg: float = FLOOR_PLANE_MAX_TILT_DEG,
                         tol_frac: float = FLOOR_INLIER_TOL_FRAC,
                         below_max_frac: float = FLOOR_BELOW_MAX_FRAC,
                         min_inliers: int = FLOOR_MIN_INLIERS,
                         min_support_cells: int = FLOOR_MIN_SUPPORT_CELLS,
                         camera_above_min_frac: float = CAMERA_ABOVE_MIN_FRAC,
                         camera_height_max_cv: float = CAMERA_HEIGHT_MAX_CV,
                         hypotheses: int = FLOOR_RANSAC_HYPOTHESES, check=None,
                         camera_rights=None, camera_order=None) -> FloorPlane:
    """ONE floor plane per component from the lower envelope of its 3-D points.

    `points_xyz` are the solve's (already quality-filtered) 3-D points of this
    component, and `camera_positions` its eligible camera centres, both in the
    component's output frame. `camera_rights` (optional, same order) are the
    cameras' right axes (`forward x up`), for the roll-free reference;
    `camera_order` (optional, same order) their walk order (keyframe index in the
    solve), which the nearest-camera stride uses. Returns a
    rejected `FloorPlane` (with a reason) rather than raising when the data cannot
    support a plane."""
    up = _unit(up_xyz)
    forward = _unit(forward_xyz)
    right = np.cross(forward, up)
    right = right / max(float(np.linalg.norm(right)), EPS)
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    # ORDER-INVARIANT (review MED-4): canonicalise the points by an exact lexsort on
    # xyz before the budget subsample and the RANSAC draws, so a permuted input
    # gives the same plane.
    pts = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    cams = np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3)
    finite_cams = np.all(np.isfinite(cams), axis=1)
    # WALK ORDER (review MED-1): the nearest-camera stride below assumes cameras in
    # walk order, so sort them by the caller's `camera_order` (keyframe index in the
    # solve), else canonically by position -- never by whatever order arrived.
    if camera_order is not None and len(camera_order) == len(cams):
        order_key = np.asarray(camera_order, dtype=np.float64)[finite_cams]
        cams = cams[finite_cams][np.argsort(order_key, kind="stable")]
    else:
        cams = cams[finite_cams]
        cams = cams[np.lexsort((cams[:, 2], cams[:, 1], cams[:, 0]))]
    stats = {"points_total": int(len(pts)), "cameras": int(len(cams))}
    if len(pts) > FLOOR_POINTS_BUDGET:
        keep = np.random.default_rng(seed).choice(len(pts), FLOOR_POINTS_BUDGET, replace=False)
        pts = pts[np.sort(keep)]
        stats["points_used"] = int(len(pts))
    if len(cams) < 3:
        return _reject_plane("fewer than 3 posed cameras", stats)
    n_ref, ref_info = _camera_plane_reference(cams, up)
    stats.update(ref_info)
    if ref_info["reference_normal"] != "camera_plane":
        # A partial trajectory: no usable camera plane. Prefer the roll-free
        # vertical over the pitch-biased receipt `up`, and widen the tilt window.
        roll_free, rf_info = _roll_free_reference(camera_rights, up)
        stats.update(rf_info)
        if roll_free is not None:
            n_ref = roll_free
            stats["reference_normal"] = "roll_free"
        else:
            # Only the pitch-biased receipt `up` is left: widen the window. With the
            # roll-free vertical the normal 10 deg window holds (review MED-2: a
            # near-collinear walk cannot constrain tilt about its own axis, and a
            # 15-20 deg plane can straddle a table top and the floor).
            max_tilt_deg = max(max_tilt_deg, FLOOR_PLANE_MAX_TILT_LOOSE_DEG)
    stats["max_tilt_deg"] = max_tilt_deg
    stats["reference_normal_xyz"] = [float(v) for v in n_ref]

    bounds = padded_square_bounds(bounds_xy, pad_frac)
    min_x, min_y, max_x, max_y = bounds
    px, py = pts @ forward, pts @ right
    in_bounds = (px >= min_x) & (px <= max_x) & (py >= min_y) & (py <= max_y)
    candidate = in_bounds.copy()
    if in_bounds.any():
        near = cams[::max(1, -(-len(cams) // NEAREST_CAMERA_BUDGET))]
        candidate[in_bounds] = _below_nearest_camera(pts[in_bounds], near, n_ref) < 0.0
    cand = pts[candidate]
    stats["candidate_points"] = int(len(cand))
    if len(cand) < min_inliers:
        return _reject_plane(
            f"inadequate support: {len(cand)} points below the cameras inside the "
            f"padded bounds (< {min_inliers})", stats)

    rng = np.random.default_rng(seed)
    score_pts = cand
    if len(cand) > FLOOR_RANSAC_SCORE_POINTS:
        score_pts = cand[np.sort(rng.choice(len(cand), FLOOR_RANSAC_SCORE_POINTS, replace=False))]

    # Hypotheses: random triples, plus level planes along the reference normal
    # through the low quantiles (so a near-level floor is always on the table).
    tri = rng.integers(0, len(cand), size=(hypotheses, 3))
    a, b, c = cand[tri[:, 0]], cand[tri[:, 1]], cand[tri[:, 2]]
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1)
    keep = norms > EPS
    normals = normals[keep] / norms[keep, None]
    normals *= np.where(normals @ n_ref < 0, -1.0, 1.0)[:, None]
    offsets = -np.sum(normals * a[keep], axis=1)
    level_h = np.percentile(cand @ n_ref, np.linspace(0.0, 60.0, 61))
    normals = np.vstack([normals, np.tile(n_ref, (len(level_h), 1))])
    offsets = np.concatenate([offsets, -level_h])

    kwargs = dict(max_tilt_deg=max_tilt_deg, tol_frac=tol_frac, below_max_frac=below_max_frac,
                  camera_above_min_frac=camera_above_min_frac,
                  camera_height_max_cv=camera_height_max_cv)
    if check is not None:
        check()
    sc = _score_planes(score_pts, cams, normals, offsets, n_ref, **kwargs)
    if check is not None:
        check()
    stats["hypotheses"] = int(len(offsets))
    stats["hypotheses_valid"] = int(sc["valid"].sum())
    if not sc["valid"].any():
        # Name the constraint that failed for the best-supported tilt-valid hypothesis.
        pool = np.nonzero(sc["checks"]["tilt"])[0]
        if len(pool) == 0:
            return _reject_plane("no plane hypothesis within the tilt limit", stats)
        best = pool[int(np.argmax(sc["inliers"][pool]))]
        failed = [k for k, ok in sc["checks"].items() if not ok[best]]
        stats.update(best_invalid_below_frac=float(sc["below"][best]),
                     best_invalid_camera_above_frac=float(sc["above"][best]),
                     best_invalid_camera_height_cv=float(sc["cv"][best]))
        return _reject_plane("no plane satisfies the camera/lower-envelope checks "
                             f"(best-supported hypothesis failed: {', '.join(failed)})", stats)
    score = np.where(sc["valid"], sc["inliers"], -1)
    best = int(np.argmax(score))
    loose = stats["reference_normal"] != "camera_plane"
    if loose:
        # Vectorised over every hypothesis, on the scoring subsample: does a LAYER
        # of points lie more than one tolerance below it?
        tol_h = tol_frac * np.where(sc["h_med"] > EPS, sc["h_med"], 1.0)
        res = score_pts @ normals.T + offsets[None, :]
        below = res < -tol_h[None, :]
        n_below = below.sum(axis=0)
        masked = np.where(below, res, np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(n_below[None, :] > 0, masked, 0.0), axis=0)
        near = below & (np.abs(res - med[None, :]) <= tol_h[None, :])
        layer_frac = near.sum(axis=0) / np.maximum(n_below, 1)
        span_g = max(max_x - min_x, EPS)
        gx = np.clip(((score_pts @ forward - min_x) / span_g * FLOOR_SUPPORT_GRID).astype(int),
                     0, FLOOR_SUPPORT_GRID - 1)
        gy = np.clip(((score_pts @ right - min_y) / span_g * FLOOR_SUPPORT_GRID).astype(int),
                     0, FLOOR_SUPPORT_GRID - 1)
        one_hot = np.zeros((len(score_pts), FLOOR_SUPPORT_GRID ** 2), dtype=np.float32)
        one_hot[np.arange(len(score_pts)), gy * FLOOR_SUPPORT_GRID + gx] = 1.0
        cells = ((one_hot.T @ below.astype(np.float32)) > 0).sum(axis=0)
        on = np.abs(res) <= tol_h[None, :]
        own_cells = ((one_hot.T @ on.astype(np.float32)) > 0).sum(axis=0)
        decoy = ((n_below >= DECOY_LAYER_MIN_POINTS) & (layer_frac >= DECOY_LAYER_MIN_FRACTION)
                 & (cells >= np.maximum(DECOY_LAYER_MIN_CELLS,
                                        DECOY_LAYER_MIN_BREADTH * own_cells)))
        stats["decoy_refused"] = int((sc["valid"] & decoy).sum())
        score = np.where(sc["valid"] & ~decoy, sc["inliers"], -1)
    centroid_cam = cams.mean(axis=0)
    hyp_height = centroid_cam @ normals.T + offsets      # camera-centroid height per hypothesis
    span_g = max(max_x - min_x, EPS)
    cand_cell = (np.clip(((cand @ right - min_y) / span_g * FLOOR_SUPPORT_GRID).astype(int),
                         0, FLOOR_SUPPORT_GRID - 1) * FLOOR_SUPPORT_GRID
                 + np.clip(((cand @ forward - min_x) / span_g * FLOOR_SUPPORT_GRID).astype(int),
                           0, FLOOR_SUPPORT_GRID - 1))
    for _attempt in range(DECOY_MAX_ROUNDS if loose else 1):
        best = int(np.argmax(score))
        if score[best] < 0:
            return _reject_plane("every supported plane has a layer of points below it "
                                 "(a furniture decoy on a partial trajectory)", stats)
        normal, offset = normals[best].copy(), float(offsets[best])

        # Least-squares refinement on the full candidate set's inliers; kept only
        # if the refined plane still passes every check.
        h_med = float(sc["h_med"][best])
        res = cand @ normal + offset
        inl = np.abs(res) <= tol_frac * h_med
        stats["refined"] = False
        if inl.sum() >= 3:
            centroid = cand[inl].mean(axis=0)
            _, _, vt = np.linalg.svd(cand[inl] - centroid, full_matrices=False)
            rn = vt[-1] if float(vt[-1] @ n_ref) >= 0 else -vt[-1]
            ro = -float(rn @ centroid)
            rsc = _score_planes(cand, cams, rn[None, :], np.array([ro]), n_ref, **kwargs)
            if bool(rsc["valid"][0]):
                normal, offset = rn, ro
                stats["refined"] = True
        if not loose:
            break
        # MED-2, after refinement: the refined plane lies parallel to the surface it
        # fits (a table top is parallel to the floor), so a floor below it is a layer
        # in ITS residuals even if the raw hypothesis was tilted enough to hide it.
        tol_f = tol_frac * h_med
        r_all = cand @ normal + offset
        below = r_all < -tol_f
        if int(below.sum()) >= DECOY_LAYER_MIN_POINTS:
            rb = r_all[below]
            own = len(np.unique(cand_cell[np.abs(r_all) <= tol_f]))
            if (float(np.mean(np.abs(rb - np.median(rb)) <= tol_f)) >= DECOY_LAYER_MIN_FRACTION
                    and len(np.unique(cand_cell[below])) >= max(DECOY_LAYER_MIN_CELLS,
                                                                DECOY_LAYER_MIN_BREADTH * own)):
                # A decoy: exclude every hypothesis at this height and choose again.
                refused_h = float(centroid_cam @ normal + offset)
                score = np.where(np.abs(hyp_height - refused_h) <= 3 * tol_f, -1, score)
                stats["decoy_refused"] = stats.get("decoy_refused", 0) + 1
                continue
        break
    else:
        return _reject_plane("every supported plane has a layer of points below it "
                             "(a furniture decoy on a partial trajectory)", stats)

    # Final statistics on the full candidate set.
    fin = _score_planes(cand, cams, normal[None, :], np.array([offset]), n_ref, **kwargs)
    h_med = float(fin["h_med"][0])
    tol = tol_frac * h_med
    res = cand @ normal + offset
    inl = np.abs(res) <= tol
    span = max(max_x - min_x, EPS)
    gx = np.clip(((px[candidate][inl] - min_x) / span * FLOOR_SUPPORT_GRID).astype(int),
                 0, FLOOR_SUPPORT_GRID - 1)
    gy = np.clip(((py[candidate][inl] - min_y) / span * FLOOR_SUPPORT_GRID).astype(int),
                 0, FLOOR_SUPPORT_GRID - 1)
    cell_counts = np.bincount(gy * FLOOR_SUPPORT_GRID + gx, minlength=FLOOR_SUPPORT_GRID ** 2)
    support_cells = int((cell_counts >= 3).sum())
    cam_h = cams @ normal + offset
    stats.update(
        inliers=int(inl.sum()),
        inlier_fraction=float(inl.mean()),
        residual_rms_frac=(float(np.sqrt(np.mean((res[inl] / h_med) ** 2))) if inl.any() else None),
        below_fraction=float(fin["below"][0]),
        support_cells=support_cells,
        camera_height_median=h_med,
        camera_height_cv=float(fin["cv"][0]),
        camera_above_fraction=float(fin["above"][0]),
        camera_height_p5_p95=[float(v) for v in np.percentile(cam_h, [5, 95])],
        tilt_from_up_deg=math.degrees(math.acos(min(1.0, float(normal @ up)))),
        tilt_from_reference_deg=math.degrees(math.acos(min(1.0, float(normal @ n_ref)))),
    )

    # Ambiguity: the best other VALID plane, well away from this one at the
    # camera centroid, and how its support compares (reported, not acted on).
    centroid_c = cams.mean(axis=0)
    chosen_h = float(centroid_c @ normal + offset)
    other_h = centroid_c @ normals.T + offsets
    far = sc["valid"] & (np.abs(other_h - chosen_h) > 3 * tol)
    if far.any():
        alt = int(np.argmax(np.where(far, sc["inliers"], -1)))
        ratio = float(sc["inliers"][alt]) / max(float(sc["inliers"][best]), 1.0)
        stats.update(alt_plane_normal=[float(v) for v in normals[alt]],
                     alt_plane_offset=float(offsets[alt]),
                     alt_plane_height_above_chosen_frac=float((chosen_h - other_h[alt]) / h_med),
                     alt_plane_support_ratio=ratio,
                     ambiguous=bool(ratio >= FLOOR_AMBIGUITY_RATIO))
    else:
        stats["ambiguous"] = False

    if stats["inliers"] < min_inliers:
        return _reject_plane(f"inadequate support: {stats['inliers']} inlier points "
                             f"(< {min_inliers})", stats)
    if support_cells < min_support_cells:
        return _reject_plane(f"inadequate support: inliers cover {support_cells} grid cells "
                             f"(< {min_support_cells})", stats)
    if not bool(fin["valid"][0]):
        failed = [k for k, ok in fin["checks"].items() if not ok[0]]
        return _reject_plane(f"plane fails: {', '.join(failed)}", stats)
    stats["reason"] = "ok"
    return FloorPlane(ok=True, reason="ok", normal=tuple(float(v) for v in normal),
                      offset=float(offset), stats=stats)


FALLBACK_BAND_PERCENTILE = 80.0
FALLBACK_MIN_POINTS_PER_FRAME = 5
FALLBACK_MIN_FRAMES = 3
"""Camera-trajectory FALLBACK plane (live landings whose solve is still too sparse for
`estimate_floor_plane`; e2e2 landing 0: 29 cameras, 14 inliers). Scale-free: the
camera height H is read off the solve itself -- for every downward frame, the drop
(along the reference normal) from the camera to the 3-D points it observed in its
OWN bottom 40%, taken at the 80th percentile within that frame (the deepest typical
surface, i.e. the floor when the floor is visible), then the median over frames.
The plane is level along the reference normal, H below the cameras' median height.
It needs >= 3 such frames with >= 5 points each, and it must pass the SAME camera
checks as an estimated plane (>= 95% of cameras above it, height CV <= 0.25);
otherwise there is no plane and the component stays gray. It is reported as `source = fallback_camera_drop`."""


def fallback_floor_plane(*, failed, camera_positions, band_drops_by_frame, frame_ids=None):
    """A level plane from the camera trajectory, or a still-rejected plane.

    `band_drops_by_frame`: for each downward frame, the drops (camera height minus
    point height, along `failed.stats['reference_normal_xyz']`) of the 3-D points it
    observed in its own bottom band."""
    n = failed.stats.get("reference_normal_xyz")
    cams = np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3)
    if n is None or len(cams) < 3:
        return failed
    n = np.asarray(n, dtype=np.float64)
    ids = list(frame_ids) if frame_ids is not None else [None] * len(band_drops_by_frame)
    per_frame, contributors = [], []
    for fid, d in zip(ids, band_drops_by_frame):
        if len(d) < FALLBACK_MIN_POINTS_PER_FRAME:
            continue
        v = float(np.percentile(d, FALLBACK_BAND_PERCENTILE))
        if math.isfinite(v) and v > 0:
            per_frame.append(v)
            contributors.append(fid)
    stats = dict(failed.stats)
    stats.update(fallback_from=failed.reason, fallback_frames=len(per_frame))
    if len(per_frame) < FALLBACK_MIN_FRAMES:
        reason = (f"{failed.reason}; no fallback: {len(per_frame)} downward frames with "
                  f">= {FALLBACK_MIN_POINTS_PER_FRAME} bottom-band points "
                  f"(< {FALLBACK_MIN_FRAMES})")
        stats["reason"] = reason
        return FloorPlane(ok=False, reason=reason, stats=stats)
    height = float(np.median(per_frame))
    cam_h = cams @ n
    offset = -(float(np.median(cam_h)) - height)
    above = cam_h + offset
    med = float(np.median(above))
    stats["fallback_frame_ids"] = [c for c in contributors if c is not None]
    above_frac = float(np.mean(above > 0))
    cv = float(1.4826 * np.median(np.abs(above - med)) / max(med, EPS))
    if med <= EPS or above_frac < CAMERA_ABOVE_MIN_FRAC or cv > CAMERA_HEIGHT_MAX_CV:
        # e2e4 landing 4: a solve whose cameras are not at one height above ANY level
        # plane (CV 0.52, 11% below it) -- the fallback then warped 53 frames into
        # 0.4% opaque cells and replaced a good plan. Refuse it, like an estimated plane.
        reason = (f"{failed.reason}; fallback refused: cameras inconsistent with it "
                  f"(above {above_frac:.2f} < {CAMERA_ABOVE_MIN_FRAC} or height CV "
                  f"{cv:.2f} > {CAMERA_HEIGHT_MAX_CV})")
        stats.update(camera_height_median=med, camera_above_fraction=above_frac,
                     camera_height_cv=cv, reason=reason)
        return FloorPlane(ok=False, reason=reason, stats=stats)
    stats.update(source="fallback_camera_drop", camera_height_median=med,
                 camera_above_fraction=float(np.mean(above > 0)),
                 camera_height_cv=float(1.4826 * np.median(np.abs(above - med)) / max(med, EPS)),
                 reason="ok (fallback: camera trajectory)")
    return FloorPlane(ok=True, reason="ok (fallback: camera trajectory)",
                      normal=tuple(float(v) for v in n), offset=offset, stats=stats)


def _frame_axes(forward_xyz, up_xyz):
    fwd = _unit(forward_xyz)
    cam_up = _unit(up_xyz)
    right = np.cross(fwd, cam_up)
    rn = float(np.linalg.norm(right))
    if rn < EPS:
        raise ValueError("degenerate camera basis")
    return fwd, right / rn, -cam_up


def frame_gate(plane: FloorPlane, position_xyz, forward_xyz, up_xyz,
               min_pitch_deg: float = MIN_FRAME_PITCH_DEG):
    """(ok, reason, camera_height, pitch_deg) for one frame against the plane."""
    try:
        fwd, _, _ = _frame_axes(forward_xyz, up_xyz)
    except ValueError:
        return False, "degenerate camera basis", None, None
    n = np.asarray(plane.normal, dtype=np.float64)
    position = np.asarray(position_xyz, dtype=np.float64)
    cam_h = float(n @ position + plane.offset)
    pitch = math.degrees(math.asin(min(1.0, max(-1.0, -float(n @ fwd)))))
    if not math.isfinite(cam_h) or cam_h <= EPS:
        return False, "camera not above the floor plane", cam_h, pitch
    if pitch < min_pitch_deg:
        return False, "not downward-looking", cam_h, pitch
    return True, "ok", cam_h, pitch


def lower_band_centre(plane: FloorPlane, position_xyz, forward_xyz, up_xyz, fx, fy, cx, cy,
                      width_px, height_px, lower_fraction: float = LOWER_IMAGE_FRACTION):
    """Where the centre of the frame's bottom band meets the plane (3-D, output
    frame), or None if that ray does not hit it. Used to spread the selected
    frames over the FLOOR they show, not over where the camera stood."""
    fwd, right, down = _frame_axes(forward_xyz, up_xyz)
    n = np.asarray(plane.normal, dtype=np.float64)
    position = np.asarray(position_xyz, dtype=np.float64)
    u = (width_px - 1) / 2.0
    v = (1.0 - lower_fraction / 2.0) * height_px - 0.5
    ray = ((u - cx) / fx) * right + ((v - cy) / fy) * down + fwd
    denom = float(n @ ray)
    cam_h = float(n @ position + plane.offset)
    if denom >= -EPS or cam_h <= EPS:
        return None
    return position + (cam_h / -denom) * ray


def _bilinear(flat_img, h, w, u, v):
    """Bilinear sample of a (h*w, 3) float32 image at pixel coords (u, v)."""
    u = np.clip(u, 0.0, w - 1.0).astype(np.float32)
    v = np.clip(v, 0.0, h - 1.0).astype(np.float32)
    u0 = u.astype(np.int32)
    v0 = v.astype(np.int32)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]
    i00 = v0 * w + u0
    i01 = i00 + (u0 < w - 1)
    i10 = i00 + (v0 < h - 1) * w
    i11 = i10 + (u0 < w - 1)
    top = flat_img[i00] * (1 - du) + flat_img[i01] * du
    bot = flat_img[i10] * (1 - du) + flat_img[i11] * du
    return top * (1 - dv) + bot * dv


def build_homography_raster(*, origin_xyz, up_xyz, forward_xyz, bounds_xy, plane: FloorPlane,
                            frames, width: int = MAX_DIM, height: int = MAX_DIM,
                            pad_frac: float = DEFAULT_PAD_FRAC,
                            lower_fraction: float = LOWER_IMAGE_FRACTION,
                            min_pitch_deg: float = MIN_FRAME_PITCH_DEG,
                            horizon_min_depression_deg: float = HORIZON_MIN_DEPRESSION_DEG,
                            max_incidence_deg: float = MAX_INCIDENCE_DEG,
                            max_range_frac: float = MAX_RANGE_FRAC,
                            grazing_cos_min: float = GRAZING_COS_MIN,
                            furniture_mask: bool = FURNITURE_MASK_DEFAULT,
                            furniture_above_tol_frac: float = FURNITURE_ABOVE_TOL_FRAC,
                            furniture_block_px: int = FURNITURE_BLOCK_PX,
                            check=None) -> RasterResult:
    """One component's RGBA raster by warping the bottom band of each frame
    onto `plane`. Same basis, bounds, orientation and alpha rules as
    `build_raster`; `origin_xyz` is carried, not used (see `build_raster`)."""
    width = max(1, min(int(width), MAX_DIM))
    height = max(1, min(int(height), MAX_DIM))
    up = _unit(up_xyz)
    forward = _unit(forward_xyz)
    right = np.cross(forward, up)
    rn = float(np.linalg.norm(right))
    if rn < EPS:
        raise ValueError("degenerate component basis: forward parallel to up")
    right = right / rn
    bounds = padded_square_bounds(bounds_xy, pad_frac)
    min_x, min_y, max_x, max_y = bounds
    step_x = (max_x - min_x) / max(width - 1, 1)
    step_y = (max_y - min_y) / max(height - 1, 1)

    if plane is None or not plane.ok:
        result = _empty_result(bounds, width, height)
        result.stats = {"reason": "floor plane rejected: " + (plane.reason if plane else "none"),
                        "plane": dict(plane.stats) if plane else {}}
        return result
    n = np.asarray(plane.normal, dtype=np.float64)
    d = float(plane.offset)
    n_up = float(n @ up)
    # Frames whose own drops BUILT a fallback plane are exempt from the pose gate
    # (review MED-3); every other frame is checked against it as usual.
    fallback_ids = (set(plane.stats.get("fallback_frame_ids") or ())
                    if plane.stats.get("source") == "fallback_camera_drop" else set())
    if n_up <= 0.1:
        raise ValueError("floor plane normal nearly perpendicular to the component up")
    h_med = float(plane.stats.get("camera_height_median") or 0.0)
    if not math.isfinite(h_med) or h_med <= EPS:
        raise ValueError("floor plane carries no camera height")
    n_f, n_r = float(n @ forward), float(n @ right)
    xs = min_x + np.arange(width) * step_x
    ys = min_y + np.arange(height) * step_y
    sin_horizon = math.sin(math.radians(horizon_min_depression_deg))
    cos_incidence = math.cos(math.radians(max_incidence_deg))
    max_range = max_range_frac * h_med

    n_cells = width * height
    best_cos = np.full(n_cells, -1.0)
    best_order = np.full(n_cells, np.iinfo(np.int64).min, dtype=np.int64)
    best_rgb = np.zeros((n_cells, 3), dtype=np.float32)
    best_frame = np.full(n_cells, -1, dtype=np.int64)
    count = np.zeros(n_cells, dtype=np.int32)

    frame_reasons = {}
    rejected_cells = {"upper_image": 0, "horizon": 0, "incidence": 0, "range": 0,
                      "furniture_mask": 0}
    masked_blocks = 0
    band_pixels = 0
    band_pixels_outside_bounds = 0
    used_frames = 0
    warped_kids = []   # diagnostics only (the band floor fraction); never read back here

    def _reject(reason):
        frame_reasons[reason] = frame_reasons.get(reason, 0) + 1

    for fi, fr in enumerate(frames):
        if check is not None:
            check()   # the caller's wall cap / cancellation; raises to abandon
        ok, reason, cam_h, _pitch = frame_gate(plane, fr.position_xyz, fr.forward_xyz,
                                               fr.up_xyz, min_pitch_deg)
        if not ok:
            _reject(reason)
            continue
        # The per-frame pose gate checks a frame against an INDEPENDENT plane. A
        # camera-trajectory fallback plane is built from some frames' own drops; such
        # a frame glimpsing deeper floor would only vote itself out, so only THOSE
        # frames skip it.
        if fr.check_points_xyz is not None and fr.kid not in fallback_ids:
            cp = np.asarray(fr.check_points_xyz, dtype=np.float64).reshape(-1, 3)
            if len(cp) >= FRAME_MIN_CHECK_POINTS:
                below = float(np.mean(cp @ n + d < -FRAME_BELOW_PLANE_TOL_FRAC * h_med))
                if below > FRAME_BELOW_PLANE_MAX_FRAC:
                    _reject("pose disagrees with plane (own lower-band points below it)")
                    continue
        img = np.asarray(fr.color)
        if img.ndim != 3 or img.shape[2] < 3 or img.shape[0] < 2 or img.shape[1] < 2:
            _reject("bad image")
            continue
        h_px, w_px = img.shape[:2]
        fwd_c, right_c, down_c = _frame_axes(fr.forward_xyz, fr.up_xyz)
        cam = np.asarray(fr.position_xyz, dtype=np.float64)
        v_top = (1.0 - lower_fraction) * h_px - 0.5   # the band's upper pixel edge

        # Footprint bbox: the range disc, tightened by the band's corner hits
        # when all four corners hit the plane (the footprint is then the
        # convex quad of those hits).
        cx_c, cy_c = float(cam @ forward), float(cam @ right)
        lo_x, hi_x = cx_c - max_range, cx_c + max_range
        lo_y, hi_y = cy_c - max_range, cy_c + max_range
        corners = [(-0.5, v_top), (w_px - 0.5, v_top), (-0.5, h_px - 0.5), (w_px - 0.5, h_px - 0.5)]
        hits = []
        for (cu, cv) in corners:
            ray = ((cu - fr.cx) / fr.fx) * right_c + ((cv - fr.cy) / fr.fy) * down_c + fwd_c
            den = float(n @ ray)
            if den >= -EPS:
                hits = None
                break
            p = cam + (cam_h / -den) * ray
            hits.append((float(p @ forward), float(p @ right)))
        if hits:
            hx = [h[0] for h in hits]
            hy = [h[1] for h in hits]
            lo_x, hi_x = max(lo_x, min(hx)), min(hi_x, max(hx))
            lo_y, hi_y = max(lo_y, min(hy)), min(hi_y, max(hy))

        # Forward-map the band's own pixels once, only to REPORT how much of
        # the band lands outside the padded bounds (structural in the inverse
        # warp below: such a hit has no cell).
        bv, bu = np.mgrid[int(math.ceil(v_top)):h_px, 0:w_px]
        rays = (((bu - fr.cx) / fr.fx)[..., None] * right_c + ((bv - fr.cy) / fr.fy)[..., None]
                * down_c + fwd_c)
        den = rays @ n
        hit = den < -EPS
        t = np.where(hit, cam_h / np.where(hit, -den, 1.0), 0.0)
        hp = cam + t[..., None] * rays
        hpx, hpy = hp @ forward, hp @ right
        band_pixels += int(hit.sum())
        band_pixels_outside_bounds += int((hit & ((hpx < min_x) | (hpx > max_x) |
                                                  (hpy < min_y) | (hpy > max_y))).sum())

        if hi_x < min_x or lo_x > max_x or hi_y < min_y or lo_y > max_y:
            _reject("footprint outside the padded bounds")
            continue
        c0 = max(0, int(math.floor((lo_x - min_x) / step_x)) - 1)
        c1 = min(width - 1, int(math.ceil((hi_x - min_x) / step_x)) + 1)
        r0 = max(0, int(math.floor((lo_y - min_y) / step_y)) - 1)
        r1 = min(height - 1, int(math.ceil((hi_y - min_y) / step_y)) + 1)
        # float32 grid: coordinates are tens of units, so ~1e-6 error, far below a cell.
        gx = xs[c0:c1 + 1][None, :].astype(np.float32)
        gy = ys[r0:r1 + 1][:, None].astype(np.float32)
        # The plane point of cell (x, y) is affine in (x, y):
        #   P = x * bx + y * by + p0, with its height along `up` solved from
        #   the plane equation; so are its camera coordinates (the
        #   plane-induced homography's numerator and denominator).
        bx = forward - (n_f / n_up) * up
        by = right - (n_r / n_up) * up
        v0 = -(d / n_up) * up - cam
        zc = float(v0 @ fwd_c) + gx * float(bx @ fwd_c) + gy * float(by @ fwd_c)
        xc = float(v0 @ right_c) + gx * float(bx @ right_c) + gy * float(by @ right_c)
        yc = float(v0 @ down_c) + gx * float(bx @ down_c) + gy * float(by @ down_c)
        front = zc > EPS
        safe_z = np.where(front, zc, 1.0)
        uu = fr.fx * xc / safe_z + fr.cx
        vv = fr.fy * yc / safe_z + fr.cy
        in_image = front & (uu >= -0.5) & (uu <= w_px - 0.5) & (vv >= -0.5) & (vv <= h_px - 0.5)
        in_band = in_image & (vv >= v_top)
        dist = np.sqrt(xc * xc + yc * yc + zc * zc)   # the camera axes are orthonormal
        cos_inc = cam_h / np.maximum(dist, EPS)       # = sin(depression) for a plane hit
        horizon_ok = cos_inc >= sin_horizon
        incidence_ok = cos_inc >= cos_incidence
        range_ok = dist <= max_range
        rejected_cells["upper_image"] += int((in_image & ~in_band).sum())
        rejected_cells["horizon"] += int((in_band & ~horizon_ok).sum())
        rejected_cells["incidence"] += int((in_band & horizon_ok & ~incidence_ok).sum())
        rejected_cells["range"] += int((in_band & horizon_ok & incidence_ok & ~range_ok).sum())
        ok_cells = in_band & horizon_ok & incidence_ok & range_ok
        blocks = (_furniture_blocks(fr, n, d, h_med, h_px, w_px, furniture_block_px,
                                    furniture_above_tol_frac) if furniture_mask else None)
        if blocks is not None and blocks[1]:
            grid, n_masked = blocks
            masked_blocks += n_masked
            bu = np.clip(np.floor((uu + 0.5) / furniture_block_px).astype(int), 0, grid.shape[1] - 1)
            bv = np.clip(np.floor((vv + 0.5) / furniture_block_px).astype(int), 0, grid.shape[0] - 1)
            furniture = ok_cells & grid[bv, bu]
            rejected_cells["furniture_mask"] += int(furniture.sum())
            ok_cells &= ~furniture
        if not ok_cells.any():
            _reject("no admissible floor cell")
            continue
        rr, cc = np.nonzero(ok_cells)
        flat = (rr + r0) * width + (cc + c0)
        cos_s = cos_inc[rr, cc]
        order = int(fr.order)
        better = (cos_s > best_cos[flat]) | ((cos_s == best_cos[flat]) & (order > best_order[flat]))
        upd = flat[better]
        best_cos[upd] = cos_s[better]
        best_order[upd] = order
        if upd.size:
            # Sample colour only where this frame wins: a cell another frame sees
            # more head-on never needs this frame's pixel.
            flat_img = np.ascontiguousarray(img[:, :, :3], dtype=np.float32).reshape(-1, 3)
            best_rgb[upd] = _bilinear(flat_img, h_px, w_px, uu[rr[better], cc[better]],
                                      vv[rr[better], cc[better]])
        best_frame[upd] = fi
        count[flat] += 1
        used_frames += 1
        warped_kids.append(str(fr.kid))

    opaque = (count >= 2) | ((count == 1) & (best_cos >= grazing_cos_min))
    multi = count >= 2
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    idx = np.nonzero(opaque)[0]
    _paint_cells(rgba, idx // width, idx % width, best_rgb[idx], multi[idx])
    contributing = sorted({str(frames[i].kid) for i in np.unique(best_frame[idx])})
    stats = {
        "reason": "ok",
        "frames_given": len(frames),
        "frames_warped": used_frames,
        "warped_kids": warped_kids,
        "frames_rejected": frame_reasons,
        "cells_rejected": rejected_cells,
        "single_view_grazing_cells": int(((count == 1) & (best_cos < grazing_cos_min)).sum()),
        "furniture_mask": bool(furniture_mask),
        "furniture_masked_blocks": masked_blocks,
        "band_pixels_hitting_plane": band_pixels,
        "band_pixels_outside_bounds": band_pixels_outside_bounds,
        "plane": dict(plane.stats),
    }
    return RasterResult(rgba=rgba, raster_bounds_xy=tuple(float(v) for v in bounds),
                        width=width, height=height, opaque_cells=int(len(idx)),
                        total_cells=n_cells, multi_view_cells=int(multi[idx].sum()),
                        single_view_cells=int((~multi[idx]).sum()),
                        contributing_keyframes=contributing, stats=stats)


# -- tile encoding (shared by both paths) -------------------------------------
TILE_TARGET_BYTES = 150 * 1024
TILE_CAP_BYTES = 180 * 1024
WEBP_QUALITIES = (90, 80, 70, 60, 50)
FALLBACK_DIM = 384


def _downscale_rgba(rgba, dim):
    """Area-downscale with premultiplied colour; a cell stays opaque only if
    at least half of its source area was opaque (no alpha dilation)."""
    from PIL import Image

    h, w = rgba.shape[:2]
    scale = dim / float(max(h, w))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    pre = rgba[:, :, :3].astype(np.float32) * alpha[:, :, None]
    chans = [np.asarray(Image.fromarray(np.ascontiguousarray(pre[:, :, i]), mode="F")
                        .resize((nw, nh), Image.BOX)) for i in range(3)]
    a_small = np.asarray(Image.fromarray(alpha, mode="F").resize((nw, nh), Image.BOX))
    out = np.zeros((nh, nw, 4), dtype=np.uint8)
    safe = np.maximum(a_small, 1e-6)
    for i in range(3):
        out[:, :, i] = np.clip(chans[i] / safe, 0, 255).astype(np.uint8)
    keep = a_small >= 0.5
    out[:, :, 3] = np.where(keep, 255, 0)
    out[~keep, :3] = 0
    return out


def encode_tile(rgba, *, target_bytes: int = TILE_TARGET_BYTES, cap_bytes: int = TILE_CAP_BYTES):
    """Encode an RGBA raster under the wire's byte limits.

    Ladder: lossless PNG; then WebP (lossy colour, LOSSLESS alpha:
    `alpha_quality=100`) at falling quality; then the same at 384 px. The
    first encoding <= `target_bytes` wins; failing that, the smallest one
    <= `cap_bytes`; failing that, None (omit the tile). Returns a dict with
    `data`, `mime`, `width`, `height`, `encoding`, `bytes`."""
    import io

    from PIL import Image

    attempts = []

    def _try(arr, fmt, quality=None):
        buf = io.BytesIO()
        img = Image.fromarray(np.ascontiguousarray(arr), mode="RGBA")
        if fmt == "PNG":
            img.save(buf, format="PNG")  # `optimize` costs ~5x the time for ~3% bytes
            label, mime = "png", "image/png"
        else:
            img.save(buf, format="WEBP", quality=quality, alpha_quality=100, method=4)
            label, mime = f"webp-q{quality}", "image/webp"
        data = buf.getvalue()
        entry = {"data": data, "mime": mime, "width": int(arr.shape[1]),
                 "height": int(arr.shape[0]), "encoding": label, "bytes": len(data)}
        attempts.append(entry)
        return entry

    rgba = np.asarray(rgba, dtype=np.uint8)
    # Transparent cells carry no colour; zero them so they cost nothing.
    rgba = rgba.copy()
    rgba[rgba[:, :, 3] == 0, :3] = 0
    sizes = [rgba]
    if max(rgba.shape[:2]) > FALLBACK_DIM:
        sizes.append(_downscale_rgba(rgba, FALLBACK_DIM))
    for arr in sizes:
        e = _try(arr, "PNG")
        if e["bytes"] <= target_bytes:
            return e
        for q in WEBP_QUALITIES:
            e = _try(arr, "WEBP", q)
            if e["bytes"] <= target_bytes:
                return e
    fitting = [e for e in attempts if e["bytes"] <= cap_bytes]
    if not fitting:
        return None
    return min(fitting, key=lambda e: e["bytes"])
