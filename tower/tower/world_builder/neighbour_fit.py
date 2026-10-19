"""Depth fits borrowed from the room, for posed frames whose own fit is missing or refused.

WHY THIS EXISTS

A keyframe's monocular depth enters the surface only through an affine fit to the solve's
sparse points in that frame (`dense_pipeline._fit_record`). A frame that sees few features --
a plain wall, the top of a wall, a frame mostly covered by redaction fill -- has fewer than
`DenseParams.min_sparse_points` anchors and gets no fit at all; one whose few anchors are
clustered or carry an outlier track gets a fit the held-out gate refuses. Either way the frame
is posed, in the room, and fused nowhere. On walk 10.1 that was 165 of the 1042 frames of the
placed component (79 without a fit, 86 refused), at the same rate as the pre-cleanup control
(191 of 1270), so it is a property of the frames, not of one solve.

THE IDEA

The room's fused frames have already measured depth there. Their depth, back-projected and
projected into the unfitted frame, gives that frame thousands of anchors where the sparse solve
gave it a dozen, and its prediction is fitted to them by the SAME robust affine and scored by
the SAME held-out split (`dense.align_frame`) as every other frame.

WHAT KEEPS IT FROM INVENTING GEOMETRY

  * Only frames the solve posed in the placed component are candidates (the depth stage's own
    targets), and never one the pose gate dropped.
  * The anchors are depth the room's fused frames measured: valid under the fusion's own mask,
    and bracketed by that frame's own sparse anchors (`z <= z_sparse_max`), so an anchor is
    never itself an extrapolation.
  * The fit must pass the gate every fused frame passes: a physical slope and a held-out error
    within `gate_rel`, on at least `min_sparse_points` anchors, which must come from at least
    `min_support_frames` distinct fused frames.
  * The anchors must be in the ROOM'S GAUGE, against the frame's own metric prediction (below).
  * The frame is fused only on its ANCHORED FOOTPRINT (below): nowhere its fit was not scored.
  * After adoption the frame is an ordinary fused frame: the consistency field, the evidence
    filter (>= 2 supporting frames, the contradiction ratio, back-facing) and the component
    filter all apply to what it contributes. Its far bound is `anchor_depth_multiple` x its
    farthest anchor, as an SfM-anchored frame's.

What the held-out error does NOT do. A wrong pose that moves the anchors across the scene's
structure (a corner, an edge) puts them on pixels the prediction cannot follow, and the error
rises. But the candidates are, by selection, frames that see little structure, and the held-out
anchors are the odd entries of the same dense grid as the fitted ones. On a plain wall a pose
error along the view is absorbed by the affine shift and one parallel to the wall moves nothing
the fit can see (a CPU probe of review R1, 2026-10-19: poses 0.6-2 m wrong adopted at 0.07 %);
and anchors that share an error with the candidate -- a stretch of the solve drawn in its own
gauge -- agree with it by construction. The two rules below close what it cannot see.

Occlusion: a fused frame can see a surface the candidate cannot (behind something nearer in
the candidate's view). Per small cell of the candidate's image only anchors within the fusion's
truncation band of the nearest anchor in that cell are kept -- the band is what the fusion
itself treats as one surface -- and the robust fit absorbs the rest.

THE ROOM'S GAUGE (`neighbour-fit/2`)

The held-out error measures whether the candidate's prediction can follow its ANCHORS. It
cannot tell whether the anchors themselves are where the room is. On a broken solve draw
(walk 10.1 replay cc3) a short stretch of keyframes sat in its own gauge, x4.16 the room's (the
final scale guard measured it and left it attached, uncertified). Three frames of that stretch
had own fits, consistent with each other in that gauge; seven unfitted frames of the same
stretch took 100 % of their borrowed anchors from those three (the nearest fused frames: the
room's own frames, 2-4 m away, lent nothing), fitted with a ~= 0.09 against the room's 0.39 and
a held-out error of 0.6-1.5 %, and passed every rule above -- "two distinct fused frames"
included. Fused, they multiplied the support of the stretch's surface and grew it from 4.5 to
26 m^2, a sheet 2-7 m from every camera (20.5 % of the surface).

A gauge error is visible to one independent witness: the candidate's own METRIC prediction.
The depth stage runs a metric backend (MoGe-2, "depth" kind) told the solve camera's FoV, so
for a frame in the room's gauge `z_room / z_metric` is the room's units per metre -- the very
statistic the final scale guard screens every camera with (`coherence_scale`: per camera, the
median of z_sfm / z_metric). A borrowed fit is therefore admitted only if the median of
`z_anchor / pred` over its anchors, against the room's reference (the guard's own estimator,
`dominant_level`, over the same statistic of the placed frames' own fits), is not a gross scale
error by the guard's own rule: inside the guard's CLEAR band [0.75, 1.333] (default; the band
outside which the guard calls a measured ratio a `final-scale-outlier`), or, as the stricter
setting, its SCREEN band [0.8, 1.25] (`SurfaceParams.neighbour_fit_scale_band`). The same
witness also sees a pose error along the view direction, which the affine shift absorbs: the
anchors' depth then moves by that error and the metric prediction does not. Without a metric
prediction ("disparity" backends) there is no such witness and nothing is borrowed.

THE ANCHORED FOOTPRINT

The fit was scored only where the anchors are. Beyond them -- the ceiling over a wall the donors
saw, the far side of a doorway -- the frame's depth is its own prediction carried through its
pose, which nothing checked: a pose error parallel to a plain wall moves that geometry by the
error, and on cc3 28 % of the far-surface support of the stretch's adopted frames came from such
pixels. And inside the anchors the robust fit is free to leave a minority behind: with the pose
0.6 m too low on a plain wall, the room's wall points land on pixels that truly see the ceiling,
the fit follows the wall majority and those pixels keep the ceiling's depth, 0.6 m off (review
R1's probe: 2016 of 5088 ceiling pixels). An adopted frame's prediction is therefore kept only
on the cells of the anchor grid (`ANCHOR_STRIDE`) whose anchors its fitted depth agrees with
within the fusion's truncation band (what the fusion calls one surface), dilated by one cell (the
grid's own spacing: a donor's 4-px samples projected into another view leave one-cell gaps) but
never into a cell holding an anchor it disagrees with; it is unknown elsewhere for every consumer
(`anchored_footprint`, `surface_pipeline._Frames.load`). In R1's probe that leaves 96 ceiling
pixels in every case, the dilation's fringe, which the true pose fuses correctly too. Measured before the rule, the
surface that ONLY such pixels supported was 0.5 % of the area on walk 10.1 and 0.5 % on the
pre1-u2 draw (laneG `phantom_who.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEIGHBOUR_FIT_ID = "neighbour-fit/2"

# The backends whose prediction carries a metric level (`dense.MoGeBackend`; the depth stage of
# the surface always tells it the solve camera's FoV, `surface_pipeline.ensure_depth_stage`).
METRIC_KINDS = ("depth",)

# The anchor grid: one anchor per `ANCHOR_STRIDE` pixels of a fused frame, and the candidate's
# z-buffer cell. 4 px keeps ~14k anchors per 352x640 neighbour -- three orders of magnitude
# above the 20-anchor floor -- at a cost of milliseconds per frame.
ANCHOR_STRIDE = 4


@dataclass
class NeighbourPoints:
    """A fused frame's measured, bracketed depth as world points."""

    ki: int
    centre: np.ndarray
    axis: np.ndarray
    X: np.ndarray


def frame_points(ki, z, valid, R, t, K, *, z_cap: float | None,
                 stride: int = ANCHOR_STRIDE) -> NeighbourPoints:
    """World points of the valid pixels of one fused frame on a `stride` grid, limited to depth
    its own sparse anchors bracketed (`z <= z_cap`) when a cap is known."""
    z = np.asarray(z, np.float64)
    valid = np.asarray(valid, bool)
    H, W = z.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    zz = z[vs, us]
    m = valid[vs, us] & np.isfinite(zz) & (zz > 1e-6)
    if z_cap is not None and z_cap > 0:
        m &= zz <= float(z_cap)
    u = us[m] + 0.5
    v = vs[m] + 0.5
    d = zz[m]
    Xc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], 1)
    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64)
    X = (Xc - t) @ R
    return NeighbourPoints(int(ki), -R.T @ t, R[2].copy(), X)


def pick_neighbours(centre, axis, centres, axes, n: int) -> list[int]:
    """Indices of the `n` fused frames nearest the candidate's centre among those facing the
    same half-space (optical axes less than 90 degrees apart: frames facing away from each other
    cannot share a view). `centres` and `axes` are (N, 3)."""
    C = np.asarray(centres, np.float64).reshape(-1, 3)
    if not len(C) or n <= 0:
        return []
    A = np.asarray(axes, np.float64).reshape(-1, 3)
    facing = A @ np.asarray(axis, np.float64) > 0.0
    d2 = ((C - np.asarray(centre, np.float64)) ** 2).sum(1)
    d2 = np.where(facing, d2, np.inf)
    order = np.argsort(d2, kind="stable")[:n]
    return [int(i) for i in order if np.isfinite(d2[i])]


def borrowed_anchors(K, shape, R, t, neighbours: list[NeighbourPoints], *, band,
                     blocked=None, cell: int = ANCHOR_STRIDE):
    """The neighbours' points as anchors of the candidate frame (R, t).

    Returns (u, v, z, src): integer pixels, candidate-frame depth, and the index into
    `neighbours` each anchor came from. Points behind the camera, outside the image or on a
    `blocked` pixel (the candidate's redaction fill and its margin) are dropped, and per
    `cell` x `cell` block only points within `band(z_near)` of the nearest are kept."""
    H, W = shape
    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64)
    us, vs, zs, ss = [], [], [], []
    for i, nb in enumerate(neighbours):
        if not len(nb.X):
            continue
        pc = nb.X @ R.T + t
        z = pc[:, 2]
        ok = z > 1e-6
        if not ok.any():
            continue
        pc, z = pc[ok], z[ok]
        u = np.floor(pc[:, 0] / z * K[0, 0] + K[0, 2]).astype(np.int64)
        v = np.floor(pc[:, 1] / z * K[1, 1] + K[1, 2]).astype(np.int64)
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not inside.any():
            continue
        us.append(u[inside])
        vs.append(v[inside])
        zs.append(z[inside])
        ss.append(np.full(int(inside.sum()), i, np.int64))
    if not us:
        e = np.zeros(0, np.int64)
        return e, e, np.zeros(0), e
    u, v, z, src = (np.concatenate(us), np.concatenate(vs), np.concatenate(zs),
                    np.concatenate(ss))
    if blocked is not None:
        keep = ~np.asarray(blocked, bool)[v, u]
        u, v, z, src = u[keep], v[keep], z[keep], src[keep]
    if not len(z):
        return u, v, z, src
    wc = (W + cell - 1) // cell
    cid = (v // cell) * wc + (u // cell)
    near = np.full(int(cid.max()) + 1, np.inf)
    np.minimum.at(near, cid, z)
    zn = near[cid]
    keep = z <= zn + band(zn)
    return u[keep], v[keep], z[keep], src[keep]


def scale_log(z, pred) -> float | None:
    """log of the median of `z / pred` over the finite pairs with both positive: depth in the
    solve's units per metre of the metric prediction (the final scale guard's per-camera
    statistic). None without such a pair."""
    z = np.asarray(z, np.float64).ravel()
    p = np.asarray(pred, np.float64).ravel()
    ok = np.isfinite(z) & np.isfinite(p) & (z > 0) & (p > 0)
    if not ok.any():
        return None
    return float(np.log(np.median(z[ok] / p[ok])))


def anchored_footprint(u, v, shape, *, agree=None, cell: int = ANCHOR_STRIDE) -> np.ndarray:
    """(H, W) bool: the `cell` x `cell` blocks of the candidate's image that hold an anchor
    (`u`, `v` integer pixels) the fit agrees with, dilated by one block, minus every block that
    holds an anchor it disagrees with (`agree`, per anchor; None: all agree). Module docstring,
    THE ANCHORED FOOTPRINT."""
    H, W = int(shape[0]), int(shape[1])
    hc, wc = (H + cell - 1) // cell, (W + cell - 1) // cell
    u = np.asarray(u, np.int64)
    v = np.asarray(v, np.int64)
    ok = np.ones(len(u), bool) if agree is None else np.asarray(agree, bool)
    good = np.zeros((hc + 2, wc + 2), bool)
    good[v[ok] // cell + 1, u[ok] // cell + 1] = True
    d = np.zeros((hc, wc), bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            d |= good[dy:dy + hc, dx:dx + wc]
    d[v[~ok] // cell, u[~ok] // cell] = False
    return np.repeat(np.repeat(d, cell, 0), cell, 1)[:H, :W]


def fit_from_neighbours(pred, kind: str, u, v, z, src, *, gate_rel: float, min_anchors: int,
                        min_frames: int, room_scale_log: float,
                        scale_band: tuple[float, float]) -> dict:
    """Fit the candidate's prediction to the borrowed anchors and decide. The record always says
    why; `ok` is True only when the fit passes every rule a fused frame's fit must pass AND the
    anchors are in the room's gauge: `exp(scale_log(anchors) - room_scale_log)` inside
    `scale_band` (low, high), inclusive (module docstring, THE ROOM'S GAUGE)."""
    from tower.world_builder.dense import align_frame

    n = int(len(z))
    frames = int(len(np.unique(src))) if n else 0
    rec = {"anchors": n, "anchor_frames": frames, "ok": False}
    if n < int(min_anchors):
        rec.update(code="anchors", why=f"only {n} borrowed anchors")
        return rec
    if frames < int(min_frames):
        rec.update(code="frames", why=f"anchors from only {frames} fused frame(s)")
        return rec
    p = np.asarray(pred, np.float64)[v, u]
    finite = np.isfinite(p)
    if int(finite.sum()) < int(min_anchors):
        rec.update(code="nonfinite", why="the prediction is not finite at the anchors")
        return rec
    # The anchors arrive grouped by neighbour in scan order, so `align_frame`'s even/odd
    # held-out split puts every other anchor of EVERY neighbour in each half.
    zf = np.asarray(z, np.float64)[finite]
    a, b, ho = align_frame(p[finite], zf, kind)
    rec.update(a=float(a), b=float(b), held_out_rel=None if ho is None else float(ho),
               z_anchor_min=float(np.min(zf)), z_anchor_max=float(np.max(zf)))
    if not (np.isfinite(a) and np.isfinite(b) and a > 0):
        rec.update(code="inverted", why="the borrowed fit is not physical")
        return rec
    if ho is None:
        rec.update(code="unscorable", why="the borrowed fit's held-out half is not physical")
        return rec
    if ho > gate_rel:
        rec.update(code="gate", why=f"held-out error {ho:.3f} over the {gate_rel:.0%} gate")
        return rec
    s = scale_log(zf, p[finite])
    if s is None:
        rec.update(code="scale", why="no anchor has a positive metric prediction")
        return rec
    d = s - float(room_scale_log)
    rec.update(scale_ratio=float(np.exp(d)))
    lo, hi = scale_band
    # Compared in log, as the guard compares (`final_scale_guard.classify`).
    if d < np.log(lo) or d > np.log(hi):
        rec.update(code="scale", why=f"the anchors are x{np.exp(d):.2f} the room's scale, "
                                     f"outside [{lo}, {hi}]")
        return rec
    rec.update(ok=True, code="adopted")
    return rec
