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
  * Only CERTIFIED DONORS lend anchors (below): frames the final scale guard kept in the room.
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
gauge -- agree with it by construction. The two rules below close what it cannot see: the
donors must be placed in the room, and the candidate then only re-states what they measured.

Occlusion: a fused frame can see a surface the candidate cannot (behind something nearer in
the candidate's view). Per small cell of the candidate's image only anchors within the fusion's
truncation band of the nearest anchor in that cell are kept -- the band is what the fusion
itself treats as one surface -- and the robust fit absorbs the rest.

CERTIFIED DONORS (`neighbour-fit/3`)

The held-out error measures whether the candidate's prediction can follow its ANCHORS. It
cannot tell whether the anchors themselves are where the room is. On a broken solve draw
(walk 10.1 replay cc3) a short stretch of keyframes (347-358) sat in its own gauge, x4.16 the
room's scale and turned ~90 degrees (against a good draw of the same frames); the final scale
guard measured it and kept it ATTACHED but uncertified (an excursion too short to cut). Three
frames of that stretch had own fits, consistent with each other; seven unfitted frames of the
same stretch took 100 % of their borrowed anchors from those three, fitted at a held-out error
of 0.6-1.5 % on 2-3 donor frames, and passed every rule above. Fused, they gave the stretch's
surface seven more supporters and grew it from 4.5 to 26 m^2, a sheet 2-7 m from every camera.

The pipeline's authority on whether a camera is placed in the room's gauge is the final scale
guard (`final_scale_guard.py`): it splits the room's cameras into plateaus of their metric scale
and KEEPS the ones inside its screen band (`decision: "kept"`), certified; excursions, off-band
plateaus and unmeasured stretches stay attached, uncertified. So only a fused frame inside a
plateau the guard kept may lend anchors (`certified_keyframes`); a frame the guard did not
certify may still be fused with its own fit, as before, but it may not vouch for another frame.
Without the guard's audit nothing is borrowed. No constant: the guard's own decision.

Tried first and replaced (patchy run lane G, 2026-10-19): testing each candidate's anchors
against its OWN metric prediction (median z_anchor / pred against the room's, inside the guard's
clear band). It removed the cc3 sheet, but per-frame metric depth misreads -- night plain walls
at x0.5-0.7, a near occluder up to x3.6 -- and of the frames it refused on two broken draws most
were posed within 4 cm of a good draw of the same frames. The guard's plateaus smooth that
per-frame noise; a single frame cannot.

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

NEIGHBOUR_FIT_ID = "neighbour-fit/3"

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


def certified_keyframes(audit: dict | None) -> set[int]:
    """The keyframe indices inside a plateau the final scale guard KEPT (its certified room
    stretches; module docstring, CERTIFIED DONORS), from its audit document
    (`final_scale_guard.json`). Empty without an assessment."""
    plateaus = ((audit or {}).get("assessment") or {}).get("plateaus") or []
    out: set[int] = set()
    for p in plateaus:
        if p.get("decision") != "kept":
            continue
        try:
            out.update(range(int(p["first_index"]), int(p["last_index"]) + 1))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def fit_from_neighbours(pred, kind: str, u, v, z, src, *, gate_rel: float, min_anchors: int,
                        min_frames: int) -> dict:
    """Fit the candidate's prediction to the borrowed anchors and decide. The record always says
    why; `ok` is True only when the fit passes every rule a fused frame's fit must pass."""
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
    rec.update(ok=True, code="adopted")
    return rec
