"""FOW v1.2 slice 1: the depth-projected photo raster, pure core.

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
                 patch_consistency_frac: float = PATCH_CONSISTENCY_FRAC) -> RasterResult:
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
    colors = rgb_s[best_local_idx].astype(np.float64)
    desaturated = colors * 0.6 + 128.0 * 0.4
    final_colors = np.where(multi_view[:, None], colors, desaturated)
    rgba[rows_out, cols_out, 0:3] = np.clip(final_colors, 0, 255).astype(np.uint8)
    rgba[rows_out, cols_out, 3] = 255

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
