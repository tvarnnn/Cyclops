"""Per-pixel FLOOR EVIDENCE for the photo floor plan's homography warp (lane C).

OFF unless `TOWER_WORLD_PHOTO_FLOORPLAN_EVIDENCE` is exactly `on`; off, nothing here is
imported and the warp is today's fixed bottom-40% band.

WHY. The homography method used to warp the fixed bottom 40% of every downward frame
onto the floor plane. On walk 8 the median selected band had 3% of its sparse points on
the floor (p2probe-h PROBE-REPORT): the desk, the closet clothes, the bed and the walls
were painted as carpet. On, a pixel is warped only where there is POSITIVE evidence
that it sees the floor plane; a pixel without evidence stays alpha 0 (gray). Nothing is
dilated, hole-filled or extrapolated past the evidence.

THE EVIDENCE, per selected frame, at the warp's sample size (<= 256x144), in this order:

  (b) LIVE DEPTH, where the Exp1 live-depth cache holds this keyframe (same model, same
      solve-camera FoV and size) and a SCALE-ONLY fit of it to THIS solve's own sparse
      observations of the frame passes (`fit_depth_scale`: >= 12 clean anchors, held-out
      median relative error <= `DEPTH_MAX_HELD_OUT_REL`). Every depth pixel gets a height
      above the plane (along one ray height falls linearly with depth, so this is the
      depth agreeing with the ray-plane intersection depth: scale-free, the same at every
      range) and a surface normal. The frame's FLOOR LAYER is the lowest horizontal layer
      with enough pixels inside `LAYER_WINDOW_FRAC` of the plane; a pixel is floor iff it
      is horizontal and within `EVIDENCE_TOL_FRAC` of that layer. Where the depth is
      known but says "not floor" -- or the frame has no floor layer at all -- the pixel
      is gray: the sparse points do not overrule it.
  (a) SPARSE SfM, only where (b) is silent (no cache entry, a refused fit, redaction
      fill): the frame's OWN observations of the solve's 3-D points. A point within
      `EVIDENCE_TOL_FRAC` of the floor (the frame's layer if (b) found one, else the
      plane) is a SEED; any other point (above: furniture, walls, people; below: a
      pose/plane disagreement or a reflection) is a VETO. Each seed grows a
      colour-consistent region (flood fill, fixed range around the seed's colour) of at
      most `GROW_RADIUS_FRAC` of the image's shorter side, never into a veto disc. No
      seed, no evidence.
  (c) TRANSIENT MASKS, where the solve has them (`<solve>/masks/<image>.png`, 0 =
      transient): every sample pixel touching a masked one is dropped, whatever (a)/(b)
      said. (Walk-8 live solves have none; the path is exercised by tests only.)

Veto discs apply to (b) as well: the depth model smooths over thin chair legs and table
edges that a triangulated point pins down.

TOLERANCES. All in units of the plane's median camera height H, like every other
threshold of the homography path (the solve is unitless):

  * `EVIDENCE_TOL_FRAC = 0.12`: the per-frame pose gate's own tolerance and the old
    furniture mask's (about 16 cm at H = 1.35 m). The lowest furniture that smears
    (bed, sofa seat, low table: 0.3-0.45 m, 0.25-0.35 H; a desk 0.55 H) sits well
    above it; carpet SfM noise plus the plane's own inlier tolerance (0.06 H) sit
    inside it.
  * `DEPTH_MAX_HELD_OUT_REL = 0.10`: along a floor ray a relative depth error IS a
    height error in units of H, so a fit worse than 0.10 would scatter floor pixels
    across the 0.12 tolerance; that frame falls back to (a).
  * `LAYER_WINDOW_FRAC`, `LAYER_MIN_FRAC`, `NORMAL_MIN_COS`: below, with the evidence
    for each.

COST (walk 8, 64 frames, one idle CPU): (b)+(a)+(c) ~7 ms per frame, ~0.4-0.6 s per
landing, reported as `ms["evidence"]`; the warp itself gets cheaper (a frame's footprint
is bounded by its evidence pixels, not the band's corners).
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-9

EVIDENCE_TOL_FRAC = 0.12
DEPTH_MAX_HELD_OUT_REL = 0.10
VETO_RADIUS_FRAC = 0.02
"""A veto disc's radius, as a fraction of the sample's shorter side (about 3 px of 144)."""
GROW_RADIUS_FRAC = 0.08
"""How far one seed's region may reach, as a fraction of the sample's shorter side (about
11 px of 144, i.e. ~30 px at walk 8's 359 px width). Carpet is low-texture, so seeds are
sparse; much further and a seed would speak for floor it never saw."""
GROW_COLOR_TOL = 10
"""Per-channel fixed range (0-255) around the seed's colour (3x3-blurred image)."""
MAX_SEEDS = 600
"""Cost bound: at most this many seeds per frame (an even stride over them)."""

LAYER_WINDOW_FRAC = (-0.30, 0.20)
"""Where a frame's own floor LAYER may sit relative to the component plane, in H. On
walk 8 the lowest horizontal layer of frames that plainly see carpet sits anywhere from
-0.17 to +0.18 H of the one fitted plane (the solve is not flat to 0.12 H across a room,
and a pose's pitch error tilts it further), so a fixed +-0.12 H band around the plane lost
most of the real floor. The window is asymmetric: below, a misplaced island (the walk-8
closet sits 0.6 H under the room) must stay out; above, +0.20 H (~27 cm at H = 1.35 m)
stays under the lowest furniture tops that smeared (bed, sofa seat, low table: >= 0.3 H).
A frame whose lowest horizontal layer is outside the window has no depth evidence."""
LAYER_BIN_FRAC = 0.05
LAYER_MIN_FRAC = 0.03
"""A layer needs >= 3% of the frame's depth-known pixels (about 1,100 of 36,864): a floor
glimpse smaller than that is left to gray rather than taken from a few noisy pixels."""
NORMAL_MIN_COS = 0.7
"""Floor pixels must be horizontal: the depth surface normal within ~45 deg of the plane
normal. This removes the strip of wall (baseboard) that crosses the layer's height."""
SCALE_MIN_ANCHORS = 12
"""The evidence depth fit (`fit_depth_scale`): MoGe-2 predicts depth up to ONE scale, so
the evidence fits only that scale (robust median of z_sfm / pred over the frame's clean
anchors) -- the depth method's affine fit adds a shift that bends near-range pixels, and
on walk 8 the scale-only fit put more frames' floor layers within 0.12 H of the plane
(36% vs 33% of frames on e2e4's final; median |offset| 0.10 vs 0.17 H)."""


def _axes(forward_xyz, up_xyz):
    fwd = np.asarray(forward_xyz, dtype=np.float64)
    fwd = fwd / max(float(np.linalg.norm(fwd)), EPS)
    cam_up = np.asarray(up_xyz, dtype=np.float64)
    cam_up = cam_up / max(float(np.linalg.norm(cam_up)), EPS)
    right = np.cross(fwd, cam_up)
    rn = float(np.linalg.norm(right))
    if rn < EPS:
        raise ValueError("degenerate camera basis")
    return fwd, right / rn, -cam_up


def depth_floor_evidence(depth, valid, *, scale, position_xyz, forward_xyz, up_xyz,
                         fx, fy, cx, cy, normal, offset, h_med,
                         tol_frac: float = EVIDENCE_TOL_FRAC,
                         layer_window=LAYER_WINDOW_FRAC, layer_bin: float = LAYER_BIN_FRAC,
                         layer_min_frac: float = LAYER_MIN_FRAC,
                         normal_min_cos: float = NORMAL_MIN_COS):
    """(floor, known, info): per pixel, whether the fitted depth says this pixel sees the
    frame's floor layer, and whether the depth can speak for the pixel at all.

    The floor LAYER is the lowest substantial horizontal layer of the frame's own depth
    heights inside `layer_window` of the plane (see `LAYER_WINDOW_FRAC`); a pixel is
    floor iff its surface is horizontal and within `tol_frac` of that layer. No layer,
    no depth evidence: the frame stays gray where its depth is known.

    `depth` is camera-frame z in the SOLVE's units (the sparse fit's), `scale` the
    placement scale that takes solve units to the output frame the plane lives in;
    `fx..cy` are the intrinsics at `depth`'s own size."""
    depth = np.asarray(depth, dtype=np.float64)
    h_px, w_px = depth.shape
    fwd, right, down = _axes(forward_xyz, up_xyz)
    n = np.asarray(normal, dtype=np.float64)
    h_med = float(h_med)
    cam_h = float(n @ np.asarray(position_xyz, dtype=np.float64) + float(offset))
    vv, uu = np.mgrid[0:h_px, 0:w_px].astype(np.float64)
    rx, ry = (uu - cx) / fx, (vv - cy) / fy
    # n . ray for the ray (x/z, y/z, 1) in the output frame; < 0 points at the floor.
    den = rx * float(n @ right) + ry * float(n @ down) + float(n @ fwd)
    known = (np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 1e-6)
             & (den < -EPS))
    z = np.where(known, depth, 0.0)
    height = (cam_h + float(scale) * z * den) / max(h_med, EPS)
    # Surface normal from central differences of the camera-frame points (2-px steps,
    # one-sided at the border); only its direction matters, so `scale` cancels.
    pts = np.stack([rx * z, ry * z, z], axis=-1)
    du = np.empty_like(pts)
    dv = np.empty_like(pts)
    if w_px >= 3 and h_px >= 3:
        du[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
        du[:, 0], du[:, -1] = du[:, 1], du[:, -2]
        dv[1:-1] = pts[2:] - pts[:-2]
        dv[0], dv[-1] = dv[1], dv[-2]
        nc = np.cross(du, dv)
        n_cam = np.array([float(n @ right), float(n @ down), float(n @ fwd)])
        cos_up = np.abs(nc @ n_cam) / np.maximum(np.linalg.norm(nc, axis=-1), EPS)
        nb_known = known.copy()
        nb_known[:, 1:-1] &= known[:, 2:] & known[:, :-2]
        nb_known[1:-1] &= known[2:] & known[:-2]
        horizontal = nb_known & (cos_up >= normal_min_cos)
    else:
        horizontal = np.zeros_like(known)
    info = {"layer": None}
    lo, hi = layer_window
    hv = height[horizontal & (height >= lo - tol_frac) & (height <= hi + tol_frac)]
    n_known = int(known.sum())
    need = max(1, int(math.ceil(layer_min_frac * n_known)))
    floor = np.zeros_like(known)
    if n_known and len(hv) >= need:
        edges = np.arange(lo, hi + 1e-9, layer_bin)
        # Each candidate level counts the horizontal pixels within `tol_frac` of it; the
        # LOWEST level with enough of them is the floor (nothing real is below it).
        counts = np.array([int((np.abs(hv - c) <= tol_frac).sum()) for c in edges])
        hit = np.nonzero(counts >= need)[0]
        if len(hit):
            centre = float(edges[hit[0]])
            layer = centre
            for _ in range(3):   # re-centre on the layer itself, not the first level
                sel = hv[np.abs(hv - layer) <= tol_frac]
                if not len(sel):
                    break
                layer = float(np.median(sel))
            if lo - 1e-9 <= layer <= hi + 1e-9:
                info["layer"] = layer
                floor = horizontal & (np.abs(height - layer) <= tol_frac)
    return floor, known, info


def _disc(radius):
    r = int(radius)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= radius * radius + 1e-9


def sparse_floor_evidence(color, uv, height_frac, *, tol_frac: float = EVIDENCE_TOL_FRAC,
                          veto_radius_frac: float = VETO_RADIUS_FRAC,
                          grow_radius_frac: float = GROW_RADIUS_FRAC,
                          color_tol: int = GROW_COLOR_TOL, max_seeds: int = MAX_SEEDS):
    """(floor, veto, n_seeds) from the frame's own sparse observations.

    `uv`: observed pixel positions AT THE SAMPLE SIZE of `color`; `height_frac`: the
    observed 3-D points' signed heights above the plane in units of H (non-finite =
    unknown, ignored)."""
    import cv2

    img = np.ascontiguousarray(np.asarray(color)[:, :, :3], dtype=np.uint8)
    h_px, w_px = img.shape[:2]
    floor = np.zeros((h_px, w_px), dtype=bool)
    veto = np.zeros((h_px, w_px), dtype=np.uint8)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    hf = np.asarray(height_frac, dtype=np.float64).reshape(-1)
    ok = np.all(np.isfinite(uv), axis=1) & np.isfinite(hf)
    ui = np.rint(uv[:, 0]).astype(np.int64)
    vi = np.rint(uv[:, 1]).astype(np.int64)
    ok &= (ui >= 0) & (ui < w_px) & (vi >= 0) & (vi < h_px)
    ui, vi, hf = ui[ok], vi[ok], hf[ok]
    short = float(min(h_px, w_px))
    r_veto = max(1, int(round(veto_radius_frac * short)))
    r_grow = max(1, int(round(grow_radius_frac * short)))
    on = np.abs(hf) <= tol_frac
    for u, v in zip(ui[~on], vi[~on]):
        cv2.circle(veto, (int(u), int(v)), r_veto, 1, -1)
    veto_b = veto.astype(bool)
    seeds = np.nonzero(on & ~veto_b[vi, ui])[0]
    if len(seeds) > max_seeds:
        seeds = seeds[np.linspace(0, len(seeds) - 1, max_seeds).round().astype(np.int64)]
    if len(seeds) == 0:
        return floor, veto_b, 0
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    outside = ~_disc(r_grow)
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (2 << 8)
    diff = (int(color_tol),) * 3
    for s in seeds:
        u, v = int(ui[s]), int(vi[s])
        if floor[v, u]:
            continue        # an earlier seed's region already holds it
        u0, u1 = max(0, u - r_grow), min(w_px, u + r_grow + 1)
        v0, v1 = max(0, v - r_grow), min(h_px, v + r_grow + 1)
        win = np.ascontiguousarray(blurred[v0:v1, u0:u1])
        mask = np.ones((v1 - v0 + 2, u1 - u0 + 2), dtype=np.uint8)
        barrier = veto[v0:v1, u0:u1] | outside[v0 - (v - r_grow):v1 - (v - r_grow),
                                                u0 - (u - r_grow):u1 - (u - r_grow)]
        mask[1:-1, 1:-1] = barrier.astype(np.uint8)
        cv2.floodFill(win, mask, (u - u0, v - v0), 0, diff, diff, flags)
        floor[v0:v1, u0:u1] |= mask[1:-1, 1:-1] == 2
    return floor, veto_b, int(len(seeds))


def resample_depth(depth, valid, out_w, out_h):
    """Area-resample a fitted depth map to the sample size: a sample pixel is valid only
    if EVERY full-resolution pixel under it was (no averaging across a fill edge)."""
    import cv2

    depth = np.asarray(depth)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 1e-6)
    if depth.shape == (out_h, out_w):
        return depth.astype(np.float64), valid
    filled = np.where(valid, depth, 0.0).astype(np.float32)
    frac = cv2.resize(valid.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
    total = cv2.resize(filled, (out_w, out_h), interpolation=cv2.INTER_AREA)
    ok = frac >= 0.999
    out = np.where(ok, total / np.maximum(frac, 1e-6), np.nan)
    return out.astype(np.float64), ok


def resample_transient(transient, out_w, out_h):
    """A sample pixel is transient if ANY full-resolution pixel under it is."""
    import cv2

    t = np.asarray(transient, dtype=np.float32)
    if t.shape == (out_h, out_w):
        return t > 0
    return cv2.resize(t, (out_w, out_h), interpolation=cv2.INTER_AREA) > 0


def combine(*, depth_floor=None, depth_known=None, sparse_floor=None, veto=None,
            transient=None, shape):
    """The frame's floor evidence: depth where it is known, sparse elsewhere, minus
    vetoes and transients. Returns (evidence, per-pixel source counts)."""
    ev = np.zeros(shape, dtype=bool)
    from_depth = np.zeros(shape, dtype=bool)
    if depth_floor is not None:
        from_depth = np.asarray(depth_floor, bool) & np.asarray(depth_known, bool)
        ev |= from_depth
    from_sparse = np.zeros(shape, dtype=bool)
    if sparse_floor is not None:
        silent = (~np.asarray(depth_known, bool) if depth_known is not None
                  else np.ones(shape, dtype=bool))
        from_sparse = np.asarray(sparse_floor, bool) & silent
        ev |= from_sparse
    if veto is not None:
        ev &= ~np.asarray(veto, bool)
    if transient is not None:
        ev &= ~np.asarray(transient, bool)
    return ev, {"depth_px": int((from_depth & ev).sum()), "sparse_px": int((from_sparse & ev).sum())}


def fit_depth_scale(pred_at_anchors, z_sfm, *, min_anchors: int = SCALE_MIN_ANCHORS,
                    max_held_out_rel: float = DEPTH_MAX_HELD_OUT_REL):
    """(ok, s, held_out_rel, reason): depth = s * pred. `s` is the median of z / pred over
    the anchors; it is scored on the odd anchors with a scale fitted on the even ones
    only (median relative depth error), as `dense.align_frame` scores its fit."""
    p = np.asarray(pred_at_anchors, dtype=np.float64).reshape(-1)
    z = np.asarray(z_sfm, dtype=np.float64).reshape(-1)
    ok = np.isfinite(p) & np.isfinite(z) & (p > 1e-6) & (z > 1e-6)
    p, z = p[ok], z[ok]
    if len(p) < min_anchors:
        return False, None, None, "too few clean anchors"
    s = float(np.median(z / p))
    idx = np.arange(len(p))
    even, odd = idx % 2 == 0, idx % 2 == 1
    s_even = float(np.median(z[even] / p[even]))
    held_out = float(np.median(np.abs(s_even * p[odd] - z[odd]) / z[odd]))
    if not (math.isfinite(s) and s > 0):
        return False, s, held_out, "non-finite scale"
    if not held_out <= max_held_out_rel:
        return False, s, held_out, f"held-out residual {held_out:.3f} > {max_held_out_rel}"
    return True, s, held_out, "ok"
