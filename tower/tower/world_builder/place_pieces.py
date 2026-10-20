"""Depth-placed pieces: put a piece the final scale guard cut back into the room, by its measured scale.

WHY. A walk sometimes holds a stretch that matches the room through ONE rotation-only hinge (a head
panning across a narrow doorway into a small white room). Rotation averaging places its rotations, but
nothing fixes its distance from the hinge, and GLOMAP's global positioning can blow the whole stretch up
about it (walk 10.2: x4,400, cameras thousands of units out). The final scale guard measures that
correctly -- per camera log(z_sfm / z_metric) against the frames' own metric depth -- and cuts the
stretch out of the room (`final-scale-outlier`). What only that stretch saw (10.2: the far half of the
bathroom) is then void. Matching cannot rejoin it: the stretch shares no verified pair with the room
(bug-fix run 2026-10-20, LANE-B.md section 2).

WHAT. The cut is a rigid similarity of the truth (LANE-B section 2). Its three parts are measured or kept:
  * SCALE: 1 / the guard's own measured ratio (the piece's median log ratio against the room's reference).
  * ROTATION: the solver's (rotation averaging does not need parallax).
  * TRANSLATION: where the piece's own metric depth (the live depth cache, MoGe-2) agrees with the room's:
    a grid search, coarse to fine, around the time-adjacent anchor estimate (a room camera next in time to
    the piece stood where the piece's camera stood, within walking distance).
Every frame's depth is taken in its own solve units by its own measured ratio (the guard's per-camera
log(z_sfm / z_metric), held inside the guard's screen band around its group's level): MoGe-2's metric
scale alone varies by about +-20 % from frame to frame within one room (10.2: p5 / p95 of a certified
room frame's ratio to the reference -0.17 / +0.23 in log).
The ROOM here is not the published surface (it does not exist at publish time): it is the certified
room cameras' own live depth, voxelised, a voxel a surface when two or more frames put a sample in it and
the room's frames looked through it no more often than they saw it (a door shut for a while, a person, a
depth hallucination is outvoted), and a voxel FREE when more frames looked through it than saw it.

THE ROOM'S OWN CONTROLS. That evidence is noisy: MoGe-2 depth disagrees with itself across a room by more
than any fixed tolerance. So the rule is not "few contradictions" but "no more contradictions than the
room's own frames have, found the way the piece is found": `control_segments` stretches of certified room
frames, spread over the walk, are left out of the evidence (and out of the anchor pool) and put through
EXACTLY the piece's procedure under the piece's prior error: searched over a region of the piece's radius,
each starting from its own truth displaced by a share of that radius (a sixth of it to all of it: the
piece's own starting error is unknown anywhere in its region). They give, per room and per prior,
how well a stretch that IS where the solve put it agrees with the rest, and whether the rule would accept a
stretch the search put in the wrong place.

THE SEARCH REGION. A piece is searched over the whole region the walk allows: the ball about its tightest
anchor's estimate whose radius is that anchor's bound (2.0 m/s x dt + 0.3 m, the guard's pose screen, plus
the scale interval's lever arm), on a capped lattice (counted before it is built); a larger region is
refused. Every basin of the lattice's landscape is refined and compared on the fine evidence; a region with
more basins than the step can refine is refused (`coverage-not-shown`). Every search, the room's evidence
and the certification stop at the step's wall-time bound (`time-bound`).

CERTIFICATION, every test a refusal; a refused piece stays cut (today's outcome):
  R0 the room's evidence holds under the piece's prior: all control_segments controls measured, at least
     min_good_controls (and half) GOOD (within max_control_miss_m of their own poses, pinned and unique), a
     rigidity bound, and no control the search put farther than that passing R3-R5 (R6 not credited).
  R1 scale: the guard's bootstrap interval of the piece's ratio, as the guard keeps it UNROUNDED in log
     against the core of the round that isolated the piece, narrower than the guard's own screen band
     (hi / lo <= `max_ci_width`, x1.25).
  R2 evidence: >= `min_frames` piece frames with depth (the guard's own floor for a measurement).
  R3 rigid: the piece's own frames agree with each other at least as well as the least rigid good control's.
  R4 consistent with the room, both ways, at its placement, no worse than the worst good control: its
     agreement with the room's surface at least the lowest control's, its share seeing through the room's
     surface and its share inside the room's observed free space at most the highest control's.
  R5 pinned and unique: moving the placement `profile_m` along any axis loses at least `min_peak_drop` of
     its height above a typical placement in the region (both by the fine scorer), and no other refined
     basin in the whole region scores within `min_peak_drop` of it on the same fine evidence: the depth
     gives one answer inside the region.
  R6 every anchor's bound holds, and the placement is CORROBORATED independently of the depth fit: an anchor
     that is not the region's centre and whose check could have failed (its bound is below the largest
     miss the search's reach allows from its own estimate) agrees; or, chaining only from such a piece,
     another placed piece's new geometry coincides with this one's. Two pieces never vouch for each other
     alone.
GROUPS. Pieces of one blow-up -- each passing R1 on its own, joined in the solve by shared 3-D points
through keyframes that are not the room's, with overlapping unrounded ratio intervals -- are ALSO measured
as one rigid body: the pooled scale, the union of the members' intervals, every member's anchors under the
one similarity (every bound enforced; the tightest is the search's centre, the others checks that can
fail), R5 on the pooled evidence, controls under the group's prior, R6 by an independent member anchor. A
group that is not one rigid body (R3) leaves the members' own decisions standing; otherwise a member anchor
disagreeing refuses the group and every member; a placed group places every member; any other group
refusal leaves the members' own decisions standing.

Then the piece's cameras and its points (those whose observers are mostly the piece's) take the
similarity and join the room, recorded `uncertified` (`depth-placed`): the guard's scale rule did not
certify them. The record says what was placed and why each refusal happened.

It never raises: any failure leaves every piece cut, with the error in the record. With
`TOWER_WORLD_PLACE_PIECES` off (the default) it does not run.

numpy only (scipy's KD tree for the cross-piece check, its graph components for groups). CPU. Reads the
live depth cache in place.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

PLACE_ID = "place-pieces/1"
UNCERTIFIED_DEPTH_PLACED = "depth-placed"

# Why a piece was not placed (the record's `refused`).
REFUSE_ROOM = "room-evidence"
REFUSE_CI = "scale-interval"
REFUSE_FRAMES = "too-few-depth-frames"
REFUSE_NO_ANCHOR = "no-anchor"
REFUSE_RIGID = "not-rigid"
REFUSE_OVERLAP = "no-overlap"
REFUSE_AGREE = "agrees-less-than-the-room"
REFUSE_THROUGH = "sees-through-room"
REFUSE_FREE = "in-room-free-space"
REFUSE_PEAK = "not-pinned"
REFUSE_AMBIGUOUS = "ambiguous"
REFUSE_CORROBORATION = "not-corroborated"
REFUSE_REGION = "anchor-region-too-large"
REFUSE_TOO_LARGE = "too-large"
REFUSE_TIME = "time-bound"
REFUSE_NO_CONTROLS = "no-matched-controls"
REFUSE_GROUP_ANCHORS = "group-anchors-disagree"
REFUSE_COVERAGE = "coverage-not-shown"


@dataclass(frozen=True)
class PlaceParams:
    """Every threshold of the step. Recorded with its digest."""

    # One metric depth d against a surface at depth z along the same ray: AGREE when |log(d/z)| < agree_log
    # (MoGe-2's relative error is a few per cent; 10 % holds that plus a 5 cm voxel at room distances);
    # SEES THROUGH when d > through_ratio x z (the guard's screen_high: beyond it, not the same surface).
    agree_log: float = 0.10
    through_ratio: float = 1.25
    # The search score: (closeness - through_weight x (through + in free space)) / the piece's valid samples.
    # 2: a contradiction is a measured fact against the placement, an agreement only consistent with it; at
    # weight 2 a placement that sees through as much as it agrees scores below one that meets nothing (lane
    # B's prototype's weight, kept).
    through_weight: float = 2.0
    # A placement where fewer than this share of the piece's samples meet the room is not evidence: below a
    # tenth, a handful of frames' edges decide the score.
    min_hit_share: float = 0.10
    # Depth samples used: inside (min, max) metres, not masked, not filled, not at an edge. 8 m: MoGe-2's
    # metric depth beyond a room's far wall is extrapolation (and the guard's own valid range is wider only
    # for the RATIO, a median); edge_px 2: a neighbour two pixels away, past the network's blending at a
    # discontinuity, differing by more than the agree band.
    depth_min_m: float = 0.10
    depth_max_m: float = 8.0
    edge_px: int = 2
    # Room: every room_stride_px-th pixel of each certified room frame, voxelised at voxel_m; a voxel is a
    # room surface when >= min_voxel_frames frames put a sample in it and it survives the visibility vote.
    room_stride_px: int = 8
    voxel_m: float = 0.05
    min_voxel_frames: int = 2
    # The room's observed FREE space: along every free_stride_px-th pixel's ray of each certified room frame,
    # from its camera to 1 / through_ratio of its depth (short of the surface by the same band), voxelised at
    # free_voxel_m; a voxel is free when >= min_free_frames frames looked through it and more frames looked
    # through it than saw a surface in it.
    free_stride_px: int = 16
    free_voxel_m: float = 0.10
    min_free_frames: int = 2
    # Piece samples for the rigidity and cross-piece tests.
    piece_stride_px: int = 16
    # Search: the whole REGION the walk allows -- the ball about the tightest anchor's estimate whose radius is
    # that anchor's bound (walking allowance + the scale interval's lever arm) -- on a lattice of coarse_step_m
    # (coarse voxels, coarse cells, at most coarse_frames frames; each pass's bands widen by the lattice's
    # reach, step sqrt(3) / 2, so no basin falls between lattice points), then each fine step around the best
    # (fine voxels, fine cells, at most fine_frames frames). At most max_region_evaluations lattice points:
    # a region larger than that is refused, not sampled more coarsely. The region's size follows from the
    # anchor's walking allowance; no other search size exists. 0.3 m: R5's profile scale (a basin is judged
    # at +-0.3 m), and the widened bands cover any point within the lattice's reach (0.26 m). 2,000 points:
    # a region up to ~2.3 m in radius, the allowance of an anchor ~1 s away.
    coarse_step_m: float = 0.30
    fine_steps_m: tuple = (0.10, 0.05, 0.025)
    coarse_voxel_m: float = 0.10
    coarse_stride_px: int = 16
    coarse_frames: int = 12
    fine_stride_px: int = 8
    fine_frames: int = 16
    max_region_evaluations: int = 2000
    # Every BASIN of the coarse landscape is refined on the fine evidence and the basins are compared there
    # (R5's rival is the best refined basin away from the winner, scored by the same fine scorer). A basin
    # is a local maximum of the coarse lattice -- scored optimistically where the lattice cannot see
    # between its points: depth bands widened by its reach, no hit-share cut, no free-space penalty -- above
    # the region's median. Each refinement costs ~180 fine evaluations; a region with more basins than this
    # cannot be shown covered inside the step's wall time and is refused (`coverage-not-shown`).
    max_basins: int = 8
    # R5's profile is scaled by a typical placement scored by the fine scorer at this many lattice points.
    typical_points: int = 64
    # Bounded wall time for the whole step: past it, the pieces not yet decided stay cut (`time-bound`).
    max_seconds: float = 900.0
    # Bounded work: a piece of more than max_piece_keyframes keyframes is refused; at most max_measure_frames
    # of its frames (evenly spread) are measured.
    max_piece_keyframes: int = 400
    max_measure_frames: int = 40
    # The room's evidence is built from at most this many certified frames (evenly spread): its cost is
    # linear in frames, and ~1,000 frames is a long walk's room.
    max_room_frames: int = 1500
    # The room's own controls: exactly this many stretches of control_frames consecutive certified room frames
    # with depth, evenly spread over the walk, left out of the evidence (and of the anchor pool) and placed
    # like pieces under each piece's matched prior; a room too short to spare them all places nothing. 6 x 20:
    # twenty frames is a piece's typical size (the guard needs ten ratios), six stretches sample the walk's
    # beginning, middle and end twice, and 120 frames held out of a ~1,000-frame room cost it ~12 %.
    control_segments: int = 6
    control_frames: int = 20
    # R0 needs at least this many GOOD controls (and all control_segments formed): an envelope is the
    # extreme of the good ones, and fewer than three is hardly an extreme of anything.
    min_good_controls: int = 3
    # A control the search puts within this of its own pose is placed (GOOD, with R5); farther, misplaced.
    # The agree band (10 %) at the 1.5 m of a doorway or a small room. Second setting, reported: 0.30.
    max_control_miss_m: float = 0.15
    # Certification (module docstring R1-R6).
    max_ci_width: float = 1.25
    min_frames: int = 10
    profile_m: float = 0.30
    min_peak_drop: float = 0.10
    anchor_dt_s: float = 2.0
    speed_mps: float = 2.0
    speed_margin_m: float = 0.3
    cross_new_m: float = 0.10
    cross_overlap_m: float = 0.30
    # A cross-piece coincidence needs >= 50 overlapping new points: fewer is a few frames' edges.
    cross_min_points: int = 50
    cross_median_m: float = 0.10
    # The guard's room radius: a camera farther than this from the room's median centre is not room evidence.
    room_radius_m: float = 15.0

    def to_json(self) -> dict:
        d = asdict(self)
        d["fine_steps_m"] = list(self.fine_steps_m)
        return d

    def digest(self) -> str:
        doc = {"place": PLACE_ID, **self.to_json()}
        return hashlib.sha1(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Pinhole:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def of(cls, camera: dict) -> "Pinhole":
        return cls(float(camera["fx"]), float(camera["fy"]), float(camera["cx"]), float(camera["cy"]),
                   int(camera["width"]), int(camera["height"]))


@dataclass
class Frame:
    """One camera with metric depth. `R` world->camera, `c` its centre in the solve's coordinates, `s` the
    factor from the depth's metres to that camera's own solve units (its measured scale)."""

    kid: str
    R: np.ndarray
    c: np.ndarray
    s: float
    t: float | None = None
    index: int = -1


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def centre_of(pose: dict) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(pose["rotation"], np.float64).reshape(3, 3)
    t = np.asarray(pose["translation"], np.float64).reshape(3)
    return R, -R.T @ t


def _grid(cam: Pinhole, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel rows and columns sampled every `stride` pixels (centred in each block)."""
    return np.arange(stride // 2, cam.height, stride), np.arange(stride // 2, cam.width, stride)


def _rays(cam: Pinhole, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(len(rows) * len(cols), 3) camera-frame ray directions with z = 1 (pixel centres, COLMAP's convention:
    pixel i spans [i, i+1))."""
    gv, gu = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([(gu.ravel() + 0.5 - cam.cx) / cam.fx, (gv.ravel() + 0.5 - cam.cy) / cam.fy,
                     np.ones(gv.size)], 1)


def sampled_depth(D: np.ndarray, rows: np.ndarray, cols: np.ndarray, params: PlaceParams) -> np.ndarray:
    """`D` (metres) at the sampled pixels, NaN where invalid or at a depth EDGE: a neighbour `edge_px` away
    (up, down, left or right) differs by more than the agree band. A depth network blends across a
    discontinuity, so samples there float between the two surfaces; one surface does not change depth by
    more than the agree band over a couple of pixels except at a grazing angle."""
    d = D[np.ix_(rows, cols)].astype(np.float64)
    ok = np.isfinite(d) & (d > params.depth_min_m) & (d < params.depth_max_m)
    e = params.edge_px
    H, W = D.shape
    for dr, dc in ((e, 0), (-e, 0), (0, e), (0, -e)):
        n = D[np.ix_(np.clip(rows + dr, 0, H - 1), np.clip(cols + dc, 0, W - 1))].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            step = np.abs(np.log(n / d))
        ok &= ~(np.isfinite(n) & (n > 0) & (step > params.agree_log))
    return np.where(ok, d, np.nan)


# ---------------------------------------------------------------------------
# the room's evidence
# ---------------------------------------------------------------------------

_KEY_BITS = 21


class _OutOfTime(Exception):
    """The step's wall-time bound passed inside a measurement (raised by `_check`)."""


def _check(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise _OutOfTime


def _voxel_keys(X: np.ndarray, size: float, origin: np.ndarray, clip: bool = False) -> np.ndarray:
    q = np.floor((X - origin) / size).astype(np.int64)
    if clip:  # outside the grid: a key no voxel has
        bad = ((q < 0) | (q >= (1 << _KEY_BITS))).any(1)
        q = np.where(bad[:, None], 0, q)
        return np.where(bad, -1, (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2])
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def _count_at(keys: np.ndarray, counts: np.ndarray):
    """A lookup: the count of each query key in sorted unique `keys` (0 when absent)."""
    def at(q: np.ndarray) -> np.ndarray:
        if not len(keys):
            return np.zeros(len(q), np.int64)
        i = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
        return np.where(keys[i] == q, counts[i], 0)
    return at


@dataclass
class Room:
    """The certified room's metric evidence, in room units."""

    points: np.ndarray          # (N, 3) surface voxels (mean of their samples)
    coarse: np.ndarray          # (n, 3) the same at coarse_voxel_m, for the coarse search pass
    free_keys: np.ndarray       # sorted keys of the observed free-space voxels (free_voxel_m)
    origin: np.ndarray          # the voxel grids' origin
    free_size: float            # free_voxel_m in room units
    units_per_m: float
    frames: int

    def in_free_space(self, X: np.ndarray) -> np.ndarray:
        """Whether each point lies in a voxel the room's frames looked through more often than they saw."""
        if not len(self.free_keys) or not len(X):
            return np.zeros(len(X), bool)
        k = _voxel_keys(X, self.free_size, self.origin, clip=True)
        i = np.minimum(np.searchsorted(self.free_keys, k), len(self.free_keys) - 1)
        return self.free_keys[i] == k


def build_room(frames: list[Frame], depth_of: Callable[[str], np.ndarray | None], cam: Pinhole,
               units_per_m: float, params: PlaceParams, deadline: float | None = None) -> Room | None:
    """The room's surface voxels and its observed free space (from at most max_room_frames frames, evenly
    spread), or None with no usable frame."""
    if not frames:
        return None
    frames = [frames[i] for i in _spread(len(frames), params.max_room_frames)]
    rows, cols = _grid(cam, params.room_stride_px)
    rays = _rays(cam, rows, cols)
    frows, fcols = _grid(cam, params.free_stride_px)
    frays = _rays(cam, frows, fcols)
    mid = np.median(np.stack([f.c for f in frames]), axis=0)
    reach = (params.room_radius_m + params.depth_max_m) * units_per_m
    origin = mid - reach
    size = params.voxel_m * units_per_m
    csize = params.coarse_voxel_m * units_per_m
    fsize = params.free_voxel_m * units_per_m
    if 2 * reach / min(size, csize, fsize) >= (1 << _KEY_BITS):
        raise ValueError("the room's reach does not fit the voxel key")
    keys, means, ckeys, cmeans, skeys, fkeys = [], [], [], [], [], []
    used = 0
    for f in frames:
        _check(deadline)
        D = depth_of(f.kid)
        if D is None:
            continue
        d = sampled_depth(D, rows, cols, params)
        ok = np.isfinite(d)
        if not ok.any():
            continue
        used += 1
        z = d[ok].ravel() * f.s
        X = (f.R.T @ (rays[ok.ravel()] * z[:, None]).T).T + f.c
        X = X[np.all(np.abs(X - mid) < reach, axis=1)]
        for sz, ks, ms in ((size, keys, means), (csize, ckeys, cmeans)):
            k, inv = np.unique(_voxel_keys(X, sz, origin), return_inverse=True)
            m = np.zeros((len(k), 3))
            np.add.at(m, inv, X)
            ks.append(k)
            ms.append(m / np.bincount(inv, minlength=len(k))[:, None])
        skeys.append(np.unique(_voxel_keys(X, fsize, origin)))
        # Free space: from the camera to depth / through_ratio along the sparser rays.
        fd = sampled_depth(D, frows, fcols, params).ravel()
        fok = np.isfinite(fd)
        if fok.any():
            stop = fd[fok] * f.s / params.through_ratio
            steps = np.arange(1, int(math.ceil(stop.max() / fsize)) + 1) * fsize
            sd = steps[None, :] < stop[:, None]
            dirs = (f.R.T @ frays[fok].T).T                        # z-unit rays, world frame
            P = f.c + (dirs[:, None, :] * steps[None, :, None])[sd]
            P = P[np.all(np.abs(P - mid) < reach, axis=1)]
            fkeys.append(np.unique(_voxel_keys(P, fsize, origin)))
    if not used:
        return None
    # Visibility vote at the free grid: per voxel, how many frames put a surface in it and how many looked
    # through it. A surface the room's frames looked through more often than they saw it is a transient (a
    # door that was shut for a while, a person) or a depth hallucination: not a room surface for these tests.
    su, sn = np.unique(np.concatenate(skeys), return_counts=True)
    fu, fn = np.unique(np.concatenate(fkeys), return_counts=True) if fkeys else (np.zeros(0, np.int64),
                                                                               np.zeros(0, np.int64))
    s_at = _count_at(su, sn)
    f_at = _count_at(fu, fn)
    free = fu[(fn >= params.min_free_frames) & (fn > s_at(fu))]

    # A voxel is a surface when enough frames put a sample in it and it survives the vote; it stands at the
    # mean of its samples (each frame's own mean, averaged), not at its centre, so the grid adds no bias.
    def surface(ks, ms):
        u, inv, n = np.unique(np.concatenate(ks), return_inverse=True, return_counts=True)
        pts = np.zeros((len(u), 3))
        np.add.at(pts, inv, np.concatenate(ms))
        pts = (pts / n[:, None])[n >= params.min_voxel_frames]
        fk = _voxel_keys(pts, fsize, origin)
        return pts[s_at(fk) >= f_at(fk)]

    return Room(points=surface(keys, means), coarse=surface(ckeys, cmeans), free_keys=np.sort(free),
                origin=origin, free_size=fsize, units_per_m=units_per_m, frames=used)


# ---------------------------------------------------------------------------
# a piece, and its score at a translation
# ---------------------------------------------------------------------------


@dataclass
class Piece:
    """A cut piece: its frames (centres c / ratio once placed in room units), their depth in metres and the
    factor from those metres to ROOM units."""

    frames: list[Frame]
    depths: list               # per frame: (H, W) float32 metres
    scales: list               # per frame: metres -> room units
    ratio: float
    cam: Pinhole

    @property
    def centres(self) -> np.ndarray:
        return np.stack([f.c / self.ratio for f in self.frames])

    def sampled(self, stride: int, params: PlaceParams, keep=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rows, cols, (n, h, w) z in room units at those pixels, NaN invalid or at an edge)."""
        rows, cols = _grid(self.cam, stride)
        idx = range(len(self.frames)) if keep is None else keep
        z = np.stack([sampled_depth(self.depths[k], rows, cols, params) * self.scales[k] for k in idx])
        return rows, cols, z


def _spread(n: int, k: int | None) -> list[int]:
    """At most k of range(n), evenly spread (all of them when k is None or n <= k)."""
    if not k or n <= k:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


class Scorer:
    """The piece's agreement with the room at translations near `t_ref`.

    The room's voxels are z-buffered into each placed piece frame on a grid of `stride`-pixel cells; at each
    cell with a voxel and a piece depth d: AGREE when |log(d / z_first)| < agree_log, THROUGH when
    d > through_ratio x z_first (the piece sees past the room's nearest surface), SHORT when
    d < z_first / through_ratio (something nearer than the room's surface). And every piece sample is FREE
    when it lies in space the room's frames looked through. Voxels are pre-selected once per frame for every
    translation within `reach` of `t_ref`; with `prune`, voxels deeper than through_ratio x the frame's
    95th-percentile depth (plus `reach`) are dropped (they can only make a cell SHORT)."""

    def __init__(self, piece: Piece, room: Room, points: np.ndarray, t_ref: np.ndarray, reach: float,
                 stride: int, params: PlaceParams, max_frames: int | None = None, prune: bool = True):
        cam = piece.cam
        self.cam, self.params, self.stride, self.room = cam, params, stride, room
        self.t_ref = np.asarray(t_ref, np.float64)
        keep = _spread(len(piece.frames), max_frames)
        rows, cols, z = piece.sampled(stride, params, keep)
        self.h, self.w = len(rows), len(cols)
        self.dS = z.reshape(-1)
        self.n_valid = int(np.isfinite(self.dS).sum())
        frames = [piece.frames[k] for k in keep]
        rays = _rays(cam, rows, cols)
        W0 = []
        for k, f in enumerate(frames):
            zz = z[k].ravel()
            ok = np.isfinite(zz)
            W0.append((f.R.T @ (rays[ok] * zz[ok, None]).T).T + f.c / piece.ratio + self.t_ref)
        self.W0 = np.concatenate(W0) if W0 else np.zeros((0, 3))   # the piece's samples, placed at t_ref
        hx = (cam.width / 2.0 + 1) / cam.fx
        hy = (cam.height / 2.0 + 1) / cam.fy
        Y0, fi = [], []
        for k, f in enumerate(frames):
            zk = z[k][np.isfinite(z[k])]
            if not len(zk):
                continue
            zmax = (float(np.percentile(zk, 95)) * params.through_ratio if prune else float(zk.max())) + reach
            Y = (f.R @ (points - (f.c / piece.ratio + self.t_ref)).T).T
            zz = Y[:, 2]
            sel = (zz > -reach) & (zz < zmax)
            sel &= np.abs(Y[:, 0]) <= (np.maximum(zz, 0) + reach) * hx + reach
            sel &= np.abs(Y[:, 1]) <= (np.maximum(zz, 0) + reach) * hy + reach
            Y0.append(Y[sel])
            fi.append(np.full(int(sel.sum()), k, np.int64))
        self.Y0 = np.concatenate(Y0) if Y0 else np.zeros((0, 3))
        self.fi = np.concatenate(fi) if fi else np.zeros(0, np.int64)
        self.Rs = np.stack([f.R for f in frames]) if frames else np.zeros((0, 3, 3))
        self.n_frames = len(frames)
        self.lo = math.log(params.through_ratio)
        self.zmin = params.depth_min_m * room.units_per_m

    def counts(self, t: np.ndarray, slack: float = 0.0) -> dict:
        """{agree, through, short, hit, free, valid} sample counts at translation `t`, and `closeness`: the
        agreeing samples each weighted 1 - |log(d/z)| / band (1 on the surface, 0 at the band's edge).
        `slack` (room units) widens every band by slack / z (a coarse search pass)."""
        out = {"agree": 0, "through": 0, "short": 0, "hit": 0, "free": 0, "closeness": 0.0, "valid": self.n_valid}
        if not self.n_valid:
            return out
        d_t = np.asarray(t, np.float64) - self.t_ref
        out["free"] = int(self.room.in_free_space(self.W0 + d_t).sum())
        if not len(self.Y0):
            return out
        Y = self.Y0 - np.einsum("kij,j->ki", self.Rs, d_t)[self.fi]
        z = Y[:, 2]
        ok = z > self.zmin
        zs = np.where(ok, z, 1.0)
        ci = np.floor((self.cam.fx * Y[:, 0] / zs + self.cam.cx) / self.stride).astype(np.int64)
        ri = np.floor((self.cam.fy * Y[:, 1] / zs + self.cam.cy) / self.stride).astype(np.int64)
        inside = ok & (ci >= 0) & (ci < self.w) & (ri >= 0) & (ri < self.h)
        cell = (self.fi[inside] * self.h + ri[inside]) * self.w + ci[inside]
        zb = np.full(self.n_frames * self.h * self.w, np.inf)
        np.minimum.at(zb, cell, z[inside])
        hit = np.isfinite(zb) & np.isfinite(self.dS)
        r = np.log(self.dS[hit] / zb[hit])
        ar = np.abs(r)
        # A search pass on a grid of step g cannot land closer than ~g sqrt(3) / 2 to the true placement: its
        # bands widen by that much (`slack`, room units) so the true basin is not missed between grid points.
        extra = slack / zb[hit] if slack else 0.0
        band = self.params.agree_log + extra
        near = ar < band
        out.update(agree=int(near.sum()), through=int((r > self.lo + extra).sum()),
                   short=int((r < -self.lo - extra).sum()), hit=int(hit.sum()),
                   closeness=float((1.0 - (ar / band)[near]).sum()))
        return out


def _score(c: dict, params: PlaceParams) -> float:
    """(closeness - through_weight x (through + free)) over the piece's valid samples: how much of the piece
    lands on the room's surface, and how precisely (an agreeing sample weighs 1 on the surface, 0 at the edge
    of the agree band, so the peak is the alignment and not anywhere inside the band), less the hard
    contradictions both ways (it sees through the room's surface; it puts a surface where the room looked
    through). Over valid samples, not over those that meet the room, so a placement that meets little of
    the room cannot outscore one that meets much of it; -inf when fewer than min_hit_share of the samples
    meet the room at all (not evidence)."""
    if not c["valid"] or not c["hit"] or c["hit"] < params.min_hit_share * c["valid"]:
        return -math.inf
    return (c["closeness"] - params.through_weight * (c["through"] + c["free"])) / c["valid"]


def _grid_search(scorer: Scorer, centre: np.ndarray, half: int, step_units: float, params: PlaceParams,
                 landscape: list | None = None, deadline: float | None = None) -> tuple[np.ndarray, float, int]:
    offs = np.arange(-half, half + 1) * step_units
    slack = step_units * math.sqrt(3.0) / 2.0
    best_t, best_s, n = np.asarray(centre, np.float64), -math.inf, 0
    for dx in offs:
        for dy in offs:
            for dz in offs:
                _check(deadline)
                t = centre + np.array([dx, dy, dz])
                s = _score(scorer.counts(t, slack), params)
                n += 1
                if landscape is not None:
                    landscape.append((t, s))
                if s > best_s:
                    best_s, best_t = s, t
    return best_t, best_s, n


def _fine_halves(params: PlaceParams) -> list[int]:
    """Steps either side of the previous best at each fine pass: +-2 for the first, +-1 after."""
    return [2] + [1] * (len(params.fine_steps_m) - 1)


def region_lattice(radius_units: float, step_units: float) -> np.ndarray:
    """Offsets of the coarse lattice that cover a ball of `radius_units`: every lattice point within the
    radius plus the lattice's own reach (step sqrt(3) / 2), so every point of the ball is within that reach
    of one of them."""
    n = int(math.ceil(radius_units / step_units)) + 1
    ax = np.arange(-n, n + 1) * step_units
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    return g[np.linalg.norm(g, axis=1) <= radius_units + step_units * math.sqrt(3.0) / 2.0]


def lattice_size(radius_units: float, step_units: float, cap: int) -> int:
    """`region_lattice`'s point count, or cap + 1 when its bounding cube alone holds more than 8 x cap points
    (counted before anything is allocated: a ball fills more than a third of its cube once the cube is that
    large, so the ball is then past the cap too)."""
    n = int(math.ceil(radius_units / step_units)) + 1
    if (2 * n + 1) ** 3 > 8 * cap:
        return cap + 1
    return len(region_lattice(radius_units, step_units))


def _optimistic(c: dict, params: PlaceParams) -> float:
    """A coarse lattice point's score, optimistic where the lattice cannot see what lies between its points:
    no hit-share cut (the share of samples meeting the room changes between points) and no free-space
    penalty (a sample can leave free space within the lattice's reach); the depth bands are already widened
    by that reach (`Scorer.counts`' slack)."""
    if not c["valid"]:
        return -math.inf
    return (c["closeness"] - params.through_weight * c["through"]) / c["valid"]


_NEIGHBOURS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]


def basins(offs: np.ndarray, step_units: float, scores: list[float]) -> list[int]:
    """The lattice points that are local maxima of `scores` over their lattice neighbours (up to 26): at
    least every neighbour and above one of them (a flat stretch is no basin)."""
    key = {tuple(int(x) for x in np.rint(o / step_units)): i for i, o in enumerate(offs)}
    out = []
    for k, i in key.items():
        s = scores[i]
        if s == -math.inf:
            continue
        ns = [scores[j] for d in _NEIGHBOURS
              if (j := key.get((k[0] + d[0], k[1] + d[1], k[2] + d[2]))) is not None]
        if all(s >= x for x in ns) and any(s > x for x in ns):
            out.append(i)
    return out


def search(piece: Piece, room: Room, centre: np.ndarray, radius_units: float, params: PlaceParams,
           deadline: float | None = None) -> tuple:
    """(best translation, the fine scorer around it, evaluations, uniqueness, typical, basins).

    Every point of the coarse lattice covering the ball of `radius_units` about `centre` (the region the walk
    allows) is scored optimistically (`_optimistic`) on coarse voxels, cells and frames. Every BASIN of that
    landscape (`basins`) above its median is refined on the fine evidence -- a fine scorer around it, each
    finer step in turn (+-2 steps for the first, +-1 after), each pass's bands widened by its own lattice's
    reach -- and scored there without widening. The winner is the best refined basin. UNIQUENESS =
    (winner - best rival) / (winner - typical), all on the fine evidence: the rival the best refined basin
    farther than profile_m + coarse_step_m from the winner, `typical` the fine scorer's median over
    `typical_points` lattice points (a placement anywhere in the region; also the profile's scale): 1 when
    no other basin stands out, 0 when one is as good (None when the winner is not above typical). With more
    basins than max_basins nothing is refined: the translation is None (`coverage-not-shown`)."""
    upm = room.units_per_m
    centre = np.asarray(centre, np.float64)
    step = params.coarse_step_m * upm
    offs = region_lattice(radius_units, step)
    coarse = Scorer(piece, room, room.coarse, centre, radius_units + 2 * step, params.coarse_stride_px, params,
                    max_frames=params.coarse_frames)
    slack = step * math.sqrt(3.0) / 2.0
    land = []
    for o in offs:
        _check(deadline)
        land.append(_optimistic(coarse.counts(centre + o, slack), params))
    evals = len(land)
    finite = [sc for sc in land if sc > -math.inf]
    middle = float(np.median(finite)) if finite else -math.inf
    found = sorted((i for i in basins(offs, step, land) if land[i] > middle), key=lambda i: -land[i])
    if not found and finite:
        found = [int(np.argmax(land))]            # a landscape with no basin above its median: its best point
    if len(found) > params.max_basins:
        return None, None, evals, None, None, len(found)
    wide = Scorer(piece, room, room.points, centre, radius_units + 2 * step, params.fine_stride_px, params,
                  max_frames=params.fine_frames)
    pick = _spread(len(offs), params.typical_points)
    fine_typ = []
    for i in pick:
        _check(deadline)
        sc = _score(wide.counts(centre + offs[i]), params)
        if sc > -math.inf:
            fine_typ.append(sc)
    typical_fine = float(np.median(fine_typ)) if fine_typ else -math.inf
    evals += len(pick)
    halves = _fine_halves(params)
    extent = sum(h * st for h, st in zip(halves, params.fine_steps_m)) * math.sqrt(3)
    refined = []
    for i in found:
        t = centre + offs[i]
        fine = Scorer(piece, room, room.points, t, (extent + params.profile_m * math.sqrt(3) + 0.2) * upm,
                      params.fine_stride_px, params, max_frames=params.fine_frames)
        for half, step_m in zip(halves, params.fine_steps_m):
            t, _s, k = _grid_search(fine, t, half, step_m * upm, params, deadline=deadline)
            evals += k
        refined.append((t, fine, _score(fine.counts(t), params)))
        evals += 1
    if not refined:
        return centre, Scorer(piece, room, room.points, centre, 0.05 * upm, params.fine_stride_px, params,
                              max_frames=params.fine_frames), evals, None, typical_fine, 0
    t, fine, s_win = max(refined, key=lambda r: r[2])
    far = (params.profile_m + params.coarse_step_m) * upm
    rival = max((sc for tt, _f, sc in refined if np.linalg.norm(tt - t) > far), default=-math.inf)
    span = s_win - typical_fine
    unique = (None if not (math.isfinite(s_win) and span > 0) else
              (1.0 if rival == -math.inf else (s_win - max(rival, typical_fine)) / span))
    return t, fine, evals, unique, typical_fine, len(found)


def profile(scorer: Scorer, t: np.ndarray, units_per_m: float, params: PlaceParams, typical: float) -> dict:
    """The score's drop at t +- profile_m along each axis, as a share of how far the placement stands above
    a typical placement in the cube ((s0 - s) / (s0 - typical); 1 when the shifted placement no longer meets
    the room; 0 when s0 is not above typical)."""
    s0 = _score(scorer.counts(t), params)
    span = s0 - typical if math.isfinite(typical) else s0
    out = {}
    for ax in range(3):
        for sgn in (-1, 1):
            e = np.zeros(3)
            e[ax] = sgn * params.profile_m * units_per_m
            s = _score(scorer.counts(t + e), params)
            out[f"{'xyz'[ax]}{'+' if sgn > 0 else '-'}"] = (
                1.0 if s == -math.inf else (s0 - s) / span) if span > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# rigidity: the piece's frames against each other
# ---------------------------------------------------------------------------


def seen_through(points: np.ndarray, cams_R: np.ndarray, cams_c: np.ndarray, maps: np.ndarray,
                 rows: np.ndarray, cols: np.ndarray, cam: Pinhole, params: PlaceParams,
                 zmin_units: float) -> tuple[np.ndarray, np.ndarray]:
    """Per point: (how many cameras' sampled depth at the point's pixel lies beyond it by through_ratio --
    the camera looked past it --, how many agree with it). `maps` hold z at pixel `rows` x `cols`."""
    n = len(points)
    through = np.zeros(n, np.int64)
    agree = np.zeros(n, np.int64)
    if not n or not len(cams_R):
        return through, agree
    stride_r = int(rows[1] - rows[0]) if len(rows) > 1 else cam.height
    stride_c = int(cols[1] - cols[0]) if len(cols) > 1 else cam.width
    h, w = maps.shape[1:]
    lo = math.log(params.through_ratio)
    for k in range(len(cams_R)):
        Y = (cams_R[k] @ (points - cams_c[k]).T).T
        z = Y[:, 2]
        ok = np.flatnonzero(z > zmin_units)
        if not len(ok):
            continue
        u = cam.fx * Y[ok, 0] / z[ok] + cam.cx
        v = cam.fy * Y[ok, 1] / z[ok] + cam.cy
        inside = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        ok, u, v = ok[inside], u[inside], v[inside]
        ci = np.minimum((u // stride_c).astype(np.int64), w - 1)
        ri = np.minimum((v // stride_r).astype(np.int64), h - 1)
        D = maps[k][ri, ci].astype(np.float64)
        good = np.isfinite(D)
        r = np.log(D[good] / z[ok[good]])
        through[ok[good][r > lo]] += 1
        agree[ok[good][np.abs(r) < params.agree_log]] += 1
    return through, agree


def piece_points(piece: Piece, params: PlaceParams, t: np.ndarray, stride: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(the piece's sampled depth points in room units at translation t, the frame each came from)."""
    rows, cols, z = piece.sampled(stride, params)
    rays = _rays(piece.cam, rows, cols)
    P, fi = [], []
    for k, f in enumerate(piece.frames):
        zz = z[k].ravel()
        ok = np.isfinite(zz)
        P.append((f.R.T @ (rays[ok] * zz[ok, None]).T).T + f.c / piece.ratio + t)
        fi.append(np.full(int(ok.sum()), k, np.int64))
    if not P:
        return np.zeros((0, 3)), np.zeros(0, np.int64)
    return np.concatenate(P), np.concatenate(fi)


def rigidity(piece: Piece, params: PlaceParams, units_per_m: float) -> dict:
    """The piece's own frames against each other at their scaled poses (no translation involved): of the
    comparisons that are not occlusions, the share where one frame sees through another frame's surface."""
    P, fi = piece_points(piece, params, np.zeros(3), params.piece_stride_px)
    rows, cols, maps = piece.sampled(params.room_stride_px, params)
    Rs = np.stack([f.R for f in piece.frames])
    Cs = piece.centres
    agree = through = 0
    for k in range(len(piece.frames)):
        others = np.flatnonzero(np.arange(len(piece.frames)) != k)
        sel = fi == k
        if not len(others) or not sel.any():
            continue
        thr, agr = seen_through(P[sel], Rs[others], Cs[others], maps[others], rows, cols, piece.cam, params,
                                params.depth_min_m * units_per_m)
        agree += int(agr.sum())
        through += int(thr.sum())
    tot = agree + through
    return {"agree": agree, "through": through, "through_share": through / tot if tot else None}


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------


def anchors(piece: Piece, room_frames: list[Frame], params: PlaceParams) -> list[dict]:
    """The time-adjacent anchor estimates: for the piece's first camera, the latest anchor-eligible room
    camera before it, and for its last, the earliest after it, each within anchor_dt_s. Each gives the
    translation that puts the piece's camera on the room camera, and the walking allowance in metres."""
    timed = [f for f in room_frames if f.t is not None]
    out = []
    fr = sorted((f for f in piece.frames if f.t is not None), key=lambda f: f.t)
    if not fr or not timed:
        return out
    first, last = fr[0], fr[-1]
    mine = {f.kid for f in piece.frames}
    before = [f for f in timed if f.t < first.t and first.t - f.t <= params.anchor_dt_s and f.kid not in mine]
    after = [f for f in timed if f.t > last.t and f.t - last.t <= params.anchor_dt_s and f.kid not in mine]
    for side, pc, cands in (("before", first, before), ("after", last, after)):
        if not cands:
            continue
        rc = max(cands, key=lambda f: f.t) if side == "before" else min(cands, key=lambda f: f.t)
        dt = abs(pc.t - rc.t)
        out.append({"side": side, "piece_kid": pc.kid, "room_kid": rc.kid, "dt_s": dt,
                    "t": rc.c - pc.c / piece.ratio,
                    "allowance_m": params.speed_mps * dt + params.speed_margin_m,
                    "piece_centre": pc.c / piece.ratio})
    return out


# ---------------------------------------------------------------------------
# measuring one piece (or one control), and certifying it
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Placement:
    """One piece's measurements and decision. `t` the translation (room units) when searched."""

    label: str
    kids: list[str]
    ratio: float
    ci: list | None
    placed: bool = False
    refused: list = dataclasses.field(default_factory=list)
    t: np.ndarray | None = None
    stats: dict = dataclasses.field(default_factory=dict)
    new_points: np.ndarray | None = None

    def record(self) -> dict:
        out = {"label": self.label, "first_keyframe_id": self.kids[0] if self.kids else None,
               "last_keyframe_id": self.kids[-1] if self.kids else None, "keyframes": len(self.kids),
               "ratio": None if not math.isfinite(self.ratio) else round(self.ratio, 6), "ci": self.ci,
               "placed": self.placed, "refused": list(self.refused),
               "translation": None if self.t is None else [float(x) for x in self.t]}
        out.update({k: v for k, v in self.stats.items()})
        return out


def _reach_m(radius_m: float, params: PlaceParams) -> float:
    """How far (metres) the search can put a placement from its region's centre: the region, the coarse
    lattice's reach past it, and the fine passes' drift."""
    drift = sum(h * st for h, st in zip(_fine_halves(params), params.fine_steps_m)) * math.sqrt(3)
    return radius_m + params.coarse_step_m * math.sqrt(3) / 2 + drift


def regions(piece: Piece, ci_log: tuple | None, anc: list[dict], params: PlaceParams, upm: float) -> dict | None:
    """Each anchor's bound -- its walking allowance plus the scale interval's lever arm (the distance from the
    anchor's piece camera to where the piece meets the room, its samples' centroid, times the interval's
    largest log half-width; translation-free) -- written into the anchor, and the REGION: the tightest
    anchor's ball (the centre anchor). None without an anchor."""
    if not anc:
        return None
    lever = 0.0 if ci_log is None else max(abs(ci_log[0]), abs(ci_log[1]))
    P0, _fi0 = piece_points(piece, params, np.zeros(3), params.piece_stride_px)
    centroid0 = P0.mean(0) if len(P0) else np.zeros(3)
    for a in anc:
        a["bound_m"] = a["allowance_m"] + float(np.linalg.norm(a["piece_centre"] - centroid0) / upm) * lever
    return min(anc, key=lambda a: a["bound_m"])


def measure(label: str, piece: Piece, ci: list | None, room: Room, anc: list[dict], region: dict,
            params: PlaceParams, radius_m: float | None = None, deadline: float | None = None) -> Placement:
    """Everything the certification reads, for a piece or a control, searched over `region`'s ball (radius
    `radius_m`, default its bound): rigidity, the search, the full counts at the placement (every measured
    frame, unpruned), the profile, uniqueness, the new geometry and every anchor's check. Refuses only what
    cannot be measured (too few frames, a region past the lattice cap, basins past max_basins, no overlap).
    Raises _OutOfTime past `deadline`."""
    upm = room.units_per_m
    pl = Placement(label=label, kids=[f.kid for f in piece.frames], ratio=piece.ratio, ci=ci)
    st = pl.stats
    st["frames_with_depth"] = len(piece.frames)
    if len(piece.frames) < params.min_frames:
        pl.refused.append(REFUSE_FRAMES)
        return pl
    t0 = time.perf_counter()
    radius_m = region["bound_m"] if radius_m is None else radius_m
    centre = region["t"]
    lattice = lattice_size(radius_m * upm, params.coarse_step_m * upm, params.max_region_evaluations)
    st["region"] = {"side": region["side"], "dt_s": round(region["dt_s"], 3), "radius_m": round(radius_m, 3),
                    "lattice_points": lattice}
    if lattice > params.max_region_evaluations:
        pl.refused.append(REFUSE_REGION)
        return pl
    _check(deadline)
    st["rigidity"] = rigidity(piece, params, upm)
    t, fine, evals, unique, typical, nb = search(piece, room, centre, radius_m * upm, params, deadline=deadline)
    st["basins"] = nb
    st["evaluations"] = evals
    if t is None:
        pl.refused.append(REFUSE_COVERAGE)
        st["seconds"] = round(time.perf_counter() - t0, 3)
        return pl
    pl.t = t
    st["uniqueness"] = None if unique is None else round(unique, 4)
    st["shift_from_anchor_m"] = float(np.linalg.norm(t - centre) / upm)
    full = Scorer(piece, room, room.points, t, 0.05 * upm, params.fine_stride_px, params, prune=False)
    c = full.counts(t)
    st.update({k: c[k] for k in ("agree", "through", "short", "hit", "free", "valid")})
    st["hit_share"] = c["hit"] / c["valid"] if c["valid"] else None
    st["agree_share"] = c["agree"] / c["hit"] if c["hit"] else None
    st["through_share"] = c["through"] / c["hit"] if c["hit"] else None
    st["short_share"] = c["short"] / c["hit"] if c["hit"] else None
    st["free_share"] = c["free"] / c["valid"] if c["valid"] else None
    if _score(c, params) == -math.inf:
        pl.refused.append(REFUSE_OVERLAP)
        st["seconds"] = round(time.perf_counter() - t0, 3)
        return pl
    st["profile_drop"] = {k: round(v, 4) for k, v in profile(fine, t, upm, params, typical).items()}
    P, _fi = piece_points(piece, params, t, params.piece_stride_px)
    free = room.in_free_space(P)
    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415

        rest = P[~free]
        dist = (cKDTree(room.points).query(rest, k=1)[0] / upm if len(room.points) and len(rest)
                else np.full(len(rest), np.inf))
        pl.new_points = rest[dist > params.cross_new_m]
    except Exception:  # noqa: BLE001 -- the cross-piece record then has nothing from this piece
        pl.new_points = np.zeros((0, 3))
    st["new_points"] = int(len(pl.new_points))
    # R6's evidence: every anchor's room camera within its bound of the placed piece camera. An anchor is
    # INDEPENDENT evidence only when it is not the region's centre AND its check could have failed: its bound
    # is below the largest miss the search's reach allows from its own estimate.
    reach = _reach_m(radius_m, params)
    checks = []
    for a in anc:
        miss = float(np.linalg.norm(t - a["t"]) / upm)
        worst = float(np.linalg.norm(centre - a["t"]) / upm) + reach
        independent = a is not region and a["bound_m"] < worst
        checks.append({"side": a["side"], "dt_s": round(a["dt_s"], 3), "miss_m": round(miss, 3),
                       "bound_m": round(a["bound_m"], 3), "worst_m": round(worst, 3), "centre": a is region,
                       "independent": bool(independent), "ok": bool(miss <= a["bound_m"])})
    st["anchor_check"] = checks
    st["seconds"] = round(time.perf_counter() - t0, 3)
    return pl


def envelope(controls: list[Placement], units_per_m: float, params: PlaceParams) -> dict:
    """What the room's own controls show, under a prior matched to the piece's. A control is GOOD when the
    search put it back within max_control_miss_m of its own pose and the placement is pinned and unique
    (R5). The envelope is the good controls' least agreement, most seeing-through, most free space and least
    rigidity. `usable` needs every one of control_segments controls measured, at least min_good_controls
    (and half) of them good, a rigidity bound, and NO misplaced control (farther than max_control_miss_m)
    that passes R3-R5: the depth tests alone must refuse every stretch of the room's own frames the search
    would put in the wrong place (R6 is not credited: a piece can be corroborated by another piece, which no
    control can be tested on)."""
    for c in controls:
        if c.t is not None:
            truth = np.asarray(c.stats.get("true_translation") or np.zeros(3), np.float64)
            c.stats["miss_m"] = round(float(np.linalg.norm(c.t - truth)) / units_per_m, 4)

    def pinned(c):
        return (min((c.stats.get("profile_drop") or {"-": 0.0}).values()) >= params.min_peak_drop
                and (c.stats.get("uniqueness") or 0.0) >= params.min_peak_drop)

    good = [c for c in controls if c.t is not None and c.stats.get("agree_share") is not None
            and not c.refused and c.stats["miss_m"] <= params.max_control_miss_m and pinned(c)]
    out = {"controls": len(controls), "good": len(good), "usable": False}
    if len(controls) < params.control_segments:
        out["why"] = f"{len(controls)} of the {params.control_segments} controls could be formed"
        return out
    if len(good) < max(params.min_good_controls, (len(controls) + 1) // 2):
        out["why"] = "too few of the room's own stretches are placed back where they are"
        return out
    rig = [c.stats["rigidity"]["through_share"] for c in good if c.stats["rigidity"]["through_share"] is not None]
    if not rig:
        out["why"] = "no good control's rigidity could be measured"
        return out
    out.update(min_agree_share=min(c.stats["agree_share"] for c in good),
               max_through_share=max(c.stats["through_share"] for c in good),
               max_free_share=max(c.stats["free_share"] for c in good),
               max_rigidity_through=max(rig))
    bad = [c for c in controls if c.t is not None and c.stats.get("miss_m", 0.0) > params.max_control_miss_m]
    accepted = []
    for c in bad:
        probe = Placement(label=c.label, kids=c.kids, ratio=1.0, ci=None, t=c.t, stats=c.stats,
                          refused=list(c.refused))
        certify(probe, dict(out, usable=True), params, r6=False)
        if not probe.refused:
            accepted.append(c.label)
    out["misplaced_accepted"] = accepted
    out["usable"] = not accepted
    if accepted:
        out["why"] = "the rule would accept a stretch of the room's own frames in the wrong place"
    return out


def piece_tests(pl: Placement, params: PlaceParams) -> None:
    """R5 and R6's contradiction half on a measured piece -- the tests that need no controls."""
    st = pl.stats
    if pl.t is None or pl.refused:
        return
    if min((st.get("profile_drop") or {"-": 0.0}).values()) < params.min_peak_drop:
        pl.refused.append(REFUSE_PEAK)
    if st.get("uniqueness") is None or st["uniqueness"] < params.min_peak_drop:
        pl.refused.append(REFUSE_AMBIGUOUS)
    if not all(a["ok"] for a in st.get("anchor_check") or [{"ok": False}]):
        pl.refused.append(REFUSE_CORROBORATION)      # an anchor contradicts the placement


def certify(pl: Placement, env: dict, params: PlaceParams, r6: bool = True) -> None:
    """R0 and R3-R5 (and R6's contradiction half with `r6`) on a measured piece. `r6=False` for the
    controls' false-accept probe in R0, which must not credit R6."""
    st = pl.stats
    if not env.get("usable"):
        pl.refused.append(REFUSE_ROOM)
        return
    if pl.t is None or REFUSE_OVERLAP in pl.refused or st.get("agree_share") is None:
        if REFUSE_OVERLAP not in pl.refused:
            pl.refused.append(REFUSE_OVERLAP)
        return
    rig = st["rigidity"]["through_share"]
    if rig is None or rig > env["max_rigidity_through"]:
        pl.refused.append(REFUSE_RIGID)
    if st["agree_share"] < env["min_agree_share"]:
        pl.refused.append(REFUSE_AGREE)
    if st["through_share"] > env["max_through_share"]:
        pl.refused.append(REFUSE_THROUGH)
    if st["free_share"] > env["max_free_share"]:
        pl.refused.append(REFUSE_FREE)
    if min((st.get("profile_drop") or {"-": 0.0}).values()) < params.min_peak_drop:
        pl.refused.append(REFUSE_PEAK)
    if st.get("uniqueness") is None or st["uniqueness"] < params.min_peak_drop:
        pl.refused.append(REFUSE_AMBIGUOUS)
    if r6 and not all(a["ok"] for a in st.get("anchor_check") or [{"ok": False}]):
        pl.refused.append(REFUSE_CORROBORATION)
    pl.refused = list(dict.fromkeys(pl.refused))


def corroborate(placements: list[Placement], params: PlaceParams, units_per_m: float) -> None:
    """R6, corroboration: a piece that passed R0-R5 is placed when an INDEPENDENT anchor (not its region's
    centre, a check that could have failed) agrees with it, or -- chaining only from such a piece -- when its
    new geometry coincides with an already corroborated piece's (median nearest neighbour <= cross_median_m
    over >= cross_min_points points within cross_overlap_m). Two pieces never vouch for each other alone."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    ok = [p for p in placements if not p.refused]
    done = []
    for p in ok:
        if any(a["ok"] and a.get("independent") for a in p.stats.get("anchor_check") or []):
            p.stats["corroborated_by"] = "independent-anchor"
            done.append(p)

    def coincides(p, q) -> dict:
        if p.new_points is None or q.new_points is None or not len(p.new_points) or not len(q.new_points):
            return {"with": q.label, "overlap_points": 0, "median_m": None, "ok": False}
        d = cKDTree(q.new_points).query(p.new_points, k=1)[0] / units_per_m
        near = d[d <= params.cross_overlap_m]
        return {"with": q.label, "overlap_points": int(len(near)),
                "median_m": round(float(np.median(near)), 4) if len(near) else None,
                "ok": bool(len(near) >= params.cross_min_points and np.median(near) <= params.cross_median_m)}

    grew = True
    while grew:
        grew = False
        for p in ok:
            if p in done:
                continue
            cross = [coincides(p, q) for q in done]
            p.stats["cross_check"] = cross
            if any(c["ok"] for c in cross):
                p.stats["corroborated_by"] = "chained:" + next(c["with"] for c in cross if c["ok"])
                done.append(p)
                grew = True
    for p in ok:
        if p not in done:
            p.refused.append(REFUSE_CORROBORATION)
    for p in placements:
        p.placed = not p.refused


# ---------------------------------------------------------------------------
# the step at the guard seam
# ---------------------------------------------------------------------------


def _metres_per_unit(assessment: dict) -> float | None:
    ref = (assessment.get("reference") or {}).get("log")
    return math.exp(-float(ref)) if ref is not None and math.isfinite(float(ref)) else None


def _frame_scale(v: float | None, level: float, band: tuple[float, float]) -> float:
    """exp(the frame's own log ratio, held inside its group's level +- the guard's screen band)."""
    if v is None or not math.isfinite(v):
        return math.exp(level)
    return math.exp(min(max(v, level + band[0]), level + band[1]))


def solve_components(solution, room_kids: set) -> np.ndarray:
    """Per keyframe index, the id of its connected set in the solve's tracks with every room keyframe left
    out (keyframe -- point -- keyframe), or -1 for a room keyframe: two pieces in one set are joined by
    shared 3-D points through keyframes that are not the room's."""
    from scipy.sparse import coo_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

    kids = list(solution.keyframe_ids)
    nk = len(kids)
    obs = np.asarray(solution.observations, np.int64).reshape(-1, 3)
    room = np.array([k in room_kids for k in kids], bool)
    keep = (obs[:, 0] >= 0) & (obs[:, 0] < nk)
    keep[keep] &= ~room[obs[keep, 0]]
    rows, pts = obs[keep, 0], obs[keep, 2]
    npts = int(np.asarray(solution.xyz).shape[0])
    n = nk + npts
    g = coo_matrix((np.ones(len(rows)), (rows, nk + pts)), shape=(n, n))
    _ncomp, lab = connected_components(g, directed=False)
    out = lab[:nk].astype(np.int64)
    out[room] = -1
    return out


def control_kids(room_frames: list[Frame], has_depth: Callable[[str], bool], params: PlaceParams
                 ) -> list[list[str]]:
    """`control_segments` runs of `control_frames` consecutive (capture order) room frames with depth,
    evenly spread over the walk; none when the room is too short to spare them all."""
    seq = [f.kid for f in sorted(room_frames, key=lambda f: (f.index, f.kid)) if has_depth(f.kid)]
    n, k = params.control_frames, params.control_segments
    if len(seq) < (k + 2) * n:
        return []      # too short to spare them all: the room places nothing (R0)
    starts = np.linspace(n, len(seq) - 2 * n, k).astype(int)
    return [seq[s:s + n] for s in starts]


def plan(solution, assessment: dict, metric_log: dict, name_of: dict, times: dict,
         depth_of: Callable[[str], np.ndarray | None], params: PlaceParams | None = None,
         guard_params=None) -> dict:
    """Decide every isolated piece. Pure apart from `depth_of`. Returns the record, with `placements`
    (Placement objects) for `apply_placements`.

    Order (bounded work): the room's evidence; every piece's R1 (the guard's unrounded interval), size
    bound, R2 and anchors; the measurable pieces searched over their regions and tested on R5 and R6's
    contradictions; the groups, measured the same way; only then, for the pieces and groups still standing,
    the room's controls under each one's prior error (its region's radius, starts spanning it) and R0, R3,
    R4; last, R6's corroboration and the groups' decisions. Past max_seconds -- checked inside every
    search, the room's evidence and before each certification -- the pieces not yet decided stay cut."""
    from tower.world_builder.final_scale_guard import GuardParams  # noqa: PLC0415

    params = params or PlaceParams()
    gp = guard_params or GuardParams()
    started = time.perf_counter()
    rec = {"id": PLACE_ID, "params": params.to_json(), "params_digest": params.digest(), "pieces": []}
    mpu = _metres_per_unit(assessment)
    pieces = assessment.get("pieces") or []
    if mpu is None or not pieces:
        rec["why"] = "no reference scale" if mpu is None else "no isolated piece"
        rec["placements"], rec["placed"] = [], 0
        rec["seconds"] = round(time.perf_counter() - started, 3)
        return rec
    upm = 1.0 / mpu
    ref = math.log(upm)
    band = (math.log(gp.screen_low), math.log(gp.screen_high))
    cam = Pinhole.of(solution.camera)
    poses = solution.poses or {}
    index = {k: i for i, k in enumerate(solution.keyframe_ids)}

    deadline = started + params.max_seconds

    def over_time() -> bool:
        return time.perf_counter() > deadline

    def frame(kid: str, level: float) -> Frame | None:
        p = poses.get(kid)
        if not p or p.get("rotation") is None or p.get("translation") is None:
            return None
        R, c = centre_of(p)
        if not (np.isfinite(R).all() and np.isfinite(c).all()):
            return None
        return Frame(kid=kid, R=R, c=c, s=_frame_scale(metric_log.get(name_of.get(kid, kid)), level, band),
                     t=times.get(kid), index=index.get(kid, -1))

    def piece_of(kids, level, ratio) -> Piece:
        frames, depths, scales = [], [], []
        for k in [kids[i] for i in _spread(len(kids), params.max_measure_frames)]:
            f = frame(k, level)
            D = depth_of(k) if f is not None else None
            if f is None or D is None:
                continue
            frames.append(f)
            depths.append(D)
            scales.append(f.s / ratio)   # metres -> this frame's own solve units -> room units
        return Piece(frames=frames, depths=depths, scales=scales, ratio=ratio, cam=cam)

    def log_values(kids) -> np.ndarray:
        v = [metric_log.get(name_of.get(k, k)) for k in kids]
        return np.asarray([x for x in v if x is not None and math.isfinite(x)], np.float64)

    # The room's evidence: certified room cameras (in a kept plateau), inside the guard's room radius; the
    # controls are left out of it, and out of the one anchor pool everything uses. Anchors carry their own
    # in-band ratio (measured, not inherited).
    uncert = {k for u in assessment.get("uncertified") or [] for k in u.get("keyframe_ids") or []}
    certified = [k for k in assessment.get("room_keyframes") or [] if k not in uncert]
    core = log_values(certified)
    room_frames = [f for f in (frame(k, ref) for k in certified) if f is not None]
    if room_frames:
        mid = np.median(np.stack([f.c for f in room_frames]), axis=0)
        room_frames = [f for f in room_frames if np.linalg.norm(f.c - mid) * mpu <= params.room_radius_m]
    ctrl = control_kids(room_frames, lambda k: depth_of(k) is not None, params)
    left_out = {k for seg in ctrl for k in seg}
    anchor_pool = [f for f in room_frames if f.kid not in left_out
                   and (v := metric_log.get(name_of.get(f.kid, f.kid))) is not None and math.isfinite(v)
                   and band[0] <= v - ref <= band[1]]
    t_room = time.perf_counter()
    room_late = False
    try:
        room = build_room([f for f in room_frames if f.kid not in left_out], depth_of, cam, upm, params,
                          deadline=deadline)
    except _OutOfTime:
        room, room_late = None, True
    rec["room"] = {"seconds": round(time.perf_counter() - t_room, 3), "frames": 0 if room is None else room.frames,
                   "voxels": 0 if room is None else len(room.points),
                   "free_voxels": 0 if room is None else len(room.free_keys),
                   "anchor_cameras": len(anchor_pool), "metres_per_unit": mpu, "control_segments": len(ctrl)}

    # 1. Every piece: R1 on the unrounded interval, the size bound, R2, anchors; then the search and R5/R6.
    placements = []
    measured = []
    info: dict = {}
    for n, p in enumerate(pieces):
        kids = list(p.get("keyframe_ids") or [])
        vals = log_values(kids)
        label = f"piece-{n}"
        if not len(vals) or not len(core) or room is None:
            pl = Placement(label=label, kids=kids, ratio=float("nan"), ci=p.get("ci"))
            pl.refused.append(REFUSE_FRAMES if not len(vals) else (REFUSE_TIME if room_late else REFUSE_ROOM))
            placements.append(pl)
            continue
        level = float(np.median(vals))
        ratio = math.exp(level - ref)
        pl = Placement(label=label, kids=kids, ratio=ratio, ci=p.get("ci"))
        placements.append(pl)
        # R1 and the lever read the guard's own interval, unrounded, against the core of the round that
        # isolated the piece (`assess_v2` keeps both); without it R1 cannot be decided.
        if p.get("ci_log") is None or p.get("core_log_median") is None:
            pl.refused.append(REFUSE_CI)
            pl.stats["why"] = "the guard kept no unrounded interval for this piece"
            continue
        ci_log = (float(p["ci_log"][0]), float(p["ci_log"][1]))
        core_med = float(p["core_log_median"])
        d_hat = level - core_med
        ci_rel = (ci_log[0] - d_hat, ci_log[1] - d_hat)
        pl.stats["ci_log"] = [round(ci_log[0], 6), round(ci_log[1], 6)]
        if ci_log[1] - ci_log[0] > math.log(params.max_ci_width):
            pl.refused.append(REFUSE_CI)
            continue
        # A group member must have a scale of its own (R1): an uncertain one cannot be said to share it.
        info[n] = {"kids": kids, "ci_log": ci_log, "abs": (ci_log[0] + core_med, ci_log[1] + core_med),
                   "ci": p.get("ci")}
        if len(kids) > params.max_piece_keyframes:
            pl.refused.append(REFUSE_TOO_LARGE)
            continue
        piece = piece_of(kids, level, ratio)
        if len(piece.frames) < params.min_frames:
            pl.refused.append(REFUSE_FRAMES)
            pl.stats["frames_with_depth"] = len(piece.frames)
            continue
        anc = anchors(piece, anchor_pool, params)
        region = regions(piece, ci_rel, anc, params, upm)
        pl.stats["anchors"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in a.items()
                                if k not in ("piece_centre", "t")} for a in anc]
        if region is None:
            pl.refused.append(REFUSE_NO_ANCHOR)
            continue
        if over_time():
            pl.refused.append(REFUSE_TIME)
            continue
        try:
            m = measure(label, piece, p.get("ci"), room, anc, region, params, deadline=deadline)
        except _OutOfTime:
            pl.refused.append(REFUSE_TIME)
            continue
        m.stats.update(ci_log=pl.stats["ci_log"], anchors=pl.stats["anchors"])
        m.kids = kids
        piece_tests(m, params)
        placements[-1] = m
        measured.append((m, region))

    # 1b. Pieces of one blow-up, measured as ONE rigid body (module docstring, GROUPS).
    groups = []
    if room is not None and len(info) >= 2:
        comp = solve_components(solution, set(assessment.get("room_keyframes") or []))
        sets = {n: {int(comp[index[k]]) for k in d["kids"] if k in index and comp[index[k]] >= 0}
                for n, d in info.items()}
        ns = sorted(info)
        parent = {n: n for n in ns}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in ns:
            for j in ns:
                if j <= i:
                    continue
                a, b = info[i]["abs"], info[j]["abs"]
                if sets[i] & sets[j] and max(a[0], b[0]) <= min(a[1], b[1]):
                    parent[find(i)] = find(j)
        members: dict = {}
        for n in ns:
            members.setdefault(find(n), []).append(n)
        for gi, mem in enumerate(m for m in members.values() if len(m) >= 2):
            g = _measure_group(f"group-{gi}", mem, info, params, ref, upm, room, anchor_pool, index, piece_of,
                               log_values, over_time, deadline)
            groups.append(g)
            if not g["placement"].refused:
                measured.append((g["placement"], g["region"]))

    # 2. The room's controls, under each standing piece's matched prior, and R0, R3, R4.
    cache: dict = {}
    rec["controls"] = {}
    for m, region in measured:
        if m.refused:
            continue
        key = round(m.stats["region"]["radius_m"], 2)
        if key not in cache:
            controls = []
            for i, seg in enumerate(ctrl):
                if over_time():
                    break
                try:
                    c = _matched_control(f"control-{i}", i, seg, key, room, params, upm, ref, log_values, piece_of,
                                         frame, deadline)
                except _OutOfTime:
                    break
                if c is not None:
                    controls.append(c)
            cache[key] = (controls, envelope(controls, upm, params))
            rec["controls"][f"r{key}"] = {
                "controls": [_jsonable({k: v for k, v in c.record().items() if k not in ("ci", "placed")})
                             for c in controls], "envelope": cache[key][1]}
        if over_time():                                # nothing is certified past the bound
            m.refused.append(REFUSE_TIME)
            continue
        m.stats["envelope"] = f"r{key}"
        certify(m, cache[key][1], params, r6=not m.stats.get("group"))

    # 3. R6's corroboration: pieces (anchors, then chains); groups by an independent member anchor only.
    if room is not None:
        corroborate(placements, params, upm)
    for g in groups:
        gp_ = g["placement"]
        if not gp_.refused and not gp_.stats.get("anchors_disagree"):
            if any(a["ok"] and a.get("independent") for a in gp_.stats.get("anchor_check") or []):
                gp_.stats["corroborated_by"] = "independent-anchor"
            else:
                gp_.refused.append(REFUSE_CORROBORATION)
        _group_decision(g, placements, info)
    rec["groups"] = [_jsonable(dict(g["record"], members=[f"piece-{n}" for n in g["members"]],
                                    decision=g["decision"], **{k: v for k, v in g["placement"].record().items()
                                                               if k not in ("ci", "placed")}))
                     for g in groups]
    for pl in placements:
        pl.placed = not pl.refused
    rec["pieces"] = [_jsonable(pl.record()) for pl in placements]
    rec["placed"] = sum(1 for pl in placements if pl.placed)
    rec["placements"] = placements
    rec["seconds"] = round(time.perf_counter() - started, 3)
    return rec


def _measure_group(label, mem, info, params, ref, upm, room, anchor_pool, index, piece_of, log_values,
                   over_time, deadline=None) -> dict:
    """One group measured exactly as a piece: the pooled scale; as its interval (R1, the lever) the UNION of
    its members' guard intervals -- never narrower than any member's (pooling them would need each isolation
    round's core, which the guard's audit does not keep); the group's
    keyframe cap, frames spread over the union (R2), every member's anchors under the group's similarity
    (all bounds enforced; the tightest is the search's centre, the rest are checks that can fail), the
    search, R5 and R6's contradictions on the pooled evidence."""
    kids = sorted({k for n in mem for k in info[n]["kids"]}, key=lambda k: index.get(k, 0))
    vals = log_values(kids)
    level = float(np.median(vals))
    ratio = math.exp(level - ref)
    lo = min(info[n]["abs"][0] for n in mem)
    hi = max(info[n]["abs"][1] for n in mem)
    ci_log = (lo - level, hi - level)
    ci_rel = ci_log
    pl = Placement(label=label, kids=kids, ratio=ratio, ci=None)
    pl.stats["ci_log"] = [round(ci_log[0], 6), round(ci_log[1], 6)]
    out = {"members": list(mem), "placement": pl, "region": None,
           "record": {"connected_in_solve": True, "intervals_overlap": True}}
    if ci_log[1] - ci_log[0] > math.log(params.max_ci_width):
        pl.refused.append(REFUSE_CI)
        return out
    if len(kids) > params.max_piece_keyframes:
        pl.refused.append(REFUSE_TOO_LARGE)
        return out
    union = piece_of(kids, level, ratio)
    if len(union.frames) < params.min_frames:
        pl.refused.append(REFUSE_FRAMES)
        return out
    anc = []
    for n in mem:
        member = piece_of(info[n]["kids"], level, ratio)
        for a in anchors(member, anchor_pool, params):
            a["member"] = f"piece-{n}"
            anc.append(a)
    region = regions(union, ci_rel, anc, params, upm)
    pl.stats["anchors"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in a.items()
                            if k not in ("piece_centre", "t")} for a in anc]
    if region is None:
        pl.refused.append(REFUSE_NO_ANCHOR)
        return out
    if over_time():
        pl.refused.append(REFUSE_TIME)
        return out
    try:
        m = measure(label, union, None, room, anc, region, params, deadline=deadline)
    except _OutOfTime:
        pl.refused.append(REFUSE_TIME)
        return out
    m.stats.update(ci_log=pl.stats["ci_log"], anchors=pl.stats["anchors"], group=True)
    m.kids = kids
    piece_tests(m, params)
    if REFUSE_CORROBORATION in m.refused:
        # A member anchor contradicts the one similarity. Whether that refuses the members depends on R3,
        # which needs the controls: kept aside, and decided in _group_decision.
        m.refused.remove(REFUSE_CORROBORATION)
        m.stats["anchors_disagree"] = True
    out.update(placement=m, region=region)
    return out


def _group_decision(g: dict, placements: list[Placement], info: dict) -> None:
    """Not one rigid body (R3 failed): the members' own decisions stand. Else an anchor of any member disagreeing
    with the group's placement: the group AND every member refused (the cluster's placement evidence
    contradicts itself). Else placed: every member takes the group's similarity. Any other refusal: the
    members' own decisions stand."""
    gpl = g["placement"]
    if REFUSE_RIGID in gpl.refused:
        g["decision"] = "not one rigid body: members decided alone"
        return
    if gpl.stats.get("anchors_disagree"):
        gpl.refused.append(REFUSE_GROUP_ANCHORS)
        g["decision"] = "refused, members refused (anchors disagree)"
        for n in g["members"]:
            if REFUSE_GROUP_ANCHORS not in placements[n].refused:
                placements[n].refused.append(REFUSE_GROUP_ANCHORS)
        return
    if not gpl.refused:
        g["decision"] = "placed"
        for n in g["members"]:
            m = Placement(label=f"piece-{n}", kids=list(info[n]["kids"]), ratio=gpl.ratio, ci=info[n].get("ci"),
                          t=gpl.t)
            m.stats.update(placed_with=gpl.label, ci_log=[round(x, 6) for x in info[n]["ci_log"]])
            placements[n] = m
        return
    g["decision"] = "refused: members decided alone"


_START_DIRECTIONS = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                              [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])


def _matched_control(label, i, seg, radius, room, params, upm, ref, log_values, piece_of, frame, deadline=None):
    """Control `i` under a piece's prior error. A piece's starting error is unknown anywhere in its region
    (up to its radius), so the controls' starting errors span it: control i starts from its own true
    translation displaced by radius x (i + 1) / control_segments (a sixth of the radius to all of it), along
    the solve's six axis directions in turn, and is searched over a ball of the piece's radius about that
    start. A control carries every error a piece carries, its scale too: its level is its own frames' median
    ratio, so it is scaled by exp(level - ref) about the solve's origin and its true translation is what
    undoes that at its own cameras. None when it has no frame with depth."""
    vals = log_values(seg)
    level = float(np.median(vals)) if len(vals) else ref
    ratio = math.exp(level - ref)
    piece = piece_of(seg, level, ratio)
    cc = [f.c for f in (frame(k, ref) for k in seg) if f is not None]
    if not piece.frames or not cc:
        return None
    truth = np.mean(cc, axis=0) * (1.0 - 1.0 / ratio)
    offset = _START_DIRECTIONS[i % len(_START_DIRECTIONS)] * radius * upm * (i + 1) / params.control_segments
    first = piece.frames[0]
    a = {"side": "start", "piece_kid": first.kid, "room_kid": None, "dt_s": 0.0, "t": truth + offset,
         "allowance_m": radius, "piece_centre": first.c / ratio, "bound_m": radius}
    c = measure(label, piece, None, room, [a], a, params, radius_m=radius, deadline=deadline)
    c.stats["true_translation"] = truth.tolist()
    c.stats["ratio"] = ratio
    c.stats["start_error_m"] = round(float(np.linalg.norm(offset)) / upm, 4)
    return c


def apply_placements(solution, assessment: dict, placements: list[Placement]):
    """(the solution with each placed piece's cameras and points moved into the room, the assessment with
    those pieces no longer cut and recorded `uncertified` `depth-placed`). Points move with the piece whose
    cameras observe them most (the guard's own point rule, `coherence_publish._point_labels`)."""
    from tower.world_builder import coherence_publish as CP  # noqa: PLC0415

    placed = [pl for pl in placements if pl.placed]
    if not placed:
        return solution, assessment
    kidx = {k: i for i, k in enumerate(solution.keyframe_ids)}
    # The guard's labels: every posed keyframe its solver component, each isolated piece its own label.
    kf_label = np.full(len(solution.keyframe_ids), -1, np.int64)
    for kid, p in (solution.poses or {}).items():
        if kid in kidx:
            kf_label[kidx[kid]] = int(p.get("component", 0))
    nxt = int(kf_label.max(initial=0)) + 1
    label_of_piece = {}
    for n, p in enumerate(assessment.get("pieces") or []):
        for kid in p.get("keyframe_ids") or []:
            if kid in kidx:
                kf_label[kidx[kid]] = nxt + n
        label_of_piece[f"piece-{n}"] = nxt + n
    point_label = CP._point_labels(solution, kf_label)
    poses = {k: dict(v) for k, v in (solution.poses or {}).items()}
    xyz = np.asarray(solution.xyz, np.float64).copy()
    for pl in placed:
        s, t = 1.0 / pl.ratio, np.asarray(pl.t, np.float64)
        for kid in pl.kids:
            if kid not in poses:
                continue
            R, c = centre_of(poses[kid])
            # Provenance on the pose: a re-gate must not read a placed camera as solved (the solver's own
            # translation is kept, so the placement can be undone).
            poses[kid]["placed_by_depth"] = {"id": PLACE_ID, "solver_translation": list(poses[kid]["translation"]),
                                             "ratio": float(pl.ratio), "t": [float(x) for x in t]}
            poses[kid]["translation"] = [float(x) for x in (-R @ (c * s + t))]
        sel = point_label == label_of_piece[pl.label]
        xyz[sel] = xyz[sel] * s + t
    new_solution = dataclasses.replace(solution, poses=poses, xyz=xyz.astype(np.float32))
    placed_kids = {k for pl in placed for k in pl.kids}
    keep = [p for p in assessment.get("pieces") or [] if not set(p.get("keyframe_ids") or []) & placed_kids]
    unc = list(assessment.get("uncertified") or [])
    by_first = {(p.get("keyframe_ids") or [None])[0]: p for p in assessment.get("pieces") or []}
    for pl in placed:
        src = by_first.get(pl.kids[0], {})
        row = {k: src.get(k) for k in ("group", "first_index", "last_index", "measured", "class", "ratio", "reason")
               if k in src}
        row.update({"first_keyframe_id": pl.kids[0], "last_keyframe_id": pl.kids[-1], "keyframes": len(pl.kids),
                    "kind": "depth-placed", "decision": "depth-placed", "uncertified": UNCERTIFIED_DEPTH_PLACED,
                    "keyframe_ids": list(pl.kids)})
        unc.append(row)
    placed_firsts = {pl.kids[0] for pl in placed}
    plateaus = [dict(r, decision="depth-placed") if r.get("first_keyframe_id") in placed_firsts
                and r.get("decision") == "isolated" else r for r in assessment.get("plateaus") or []]
    order = {k: i for i, k in enumerate(solution.keyframe_ids)}
    room = sorted(set(assessment.get("room_keyframes") or []) | placed_kids, key=lambda k: order.get(k, len(order)))
    n_unc = sum(len(u.get("keyframe_ids") or []) for u in unc)
    new_assessment = dict(assessment, pieces=keep, uncertified=unc, uncertified_keyframes=n_unc, plateaus=plateaus,
                          room_keyframes=room, scale_status="uncertified",
                          why=f"{n_unc} attached keyframes are not certified")
    return new_solution, new_assessment


def rescale_depth_handoff(depth: dict | None, placements: list[Placement]) -> dict | None:
    """The gate's depth hand-off (`GateResult.depth`) with every PLACED frame's fit carried through its
    piece's similarity, or `depth` unchanged when nothing was placed or no depth ran.

    The gate fitted each frame's prediction to the candidate's sparse points, so a placed frame's fit is in
    its piece's old units. `coherence_publish.hand_depth_to_surface` stamps the records with the published
    solve's identity and the surface REUSES them, so they must describe the published geometry. A placed
    frame's depths all scale by s = 1 / ratio (cameras and points take X' = s X + t; rotations are kept), so
    with z' = s z: a "depth" fit pred = a z + b becomes a' = a / s, b' = b; a "disparity" fit
    pred = a / z + b becomes a' = a s, b' = b; the frame's sparse depth range z_sparse_min / max scales by s;
    the held-out residual is relative and unchanged. Dropping the records instead would drop the frames from
    the fusion (`_Frames` fuses only the records it is given). An unknown fit kind raises: the caller then
    leaves every piece cut."""
    placed = [pl for pl in placements if pl.placed]
    if not placed or not depth or not (depth.get("align") or {}).get("records"):
        return depth
    align = depth["align"]
    kind = align.get("kind")
    if kind not in ("depth", "disparity"):
        raise ValueError(f"the depth hand-off's fit kind is {kind!r}: a placed frame's fit cannot be carried")
    s_of = {k: 1.0 / pl.ratio for pl in placed for k in pl.kids}
    records = []
    for r in align["records"]:
        s = s_of.get(r.get("kid"))
        if s is None:
            records.append(r)
            continue
        r = dict(r)
        if r.get("a") is not None:
            r["a"] = float(r["a"]) / s if kind == "depth" else float(r["a"]) * s
        for key in ("z_sparse_min", "z_sparse_max"):
            if r.get(key) is not None:
                r[key] = float(r[key]) * s
        r["placed_by_depth"] = PLACE_ID
        records.append(r)
    return dict(depth, align=dict(align, records=records))


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (np.integer, np.bool_)):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


# ---------------------------------------------------------------------------
# depth from the live cache
# ---------------------------------------------------------------------------


def live_depth_source(store, world_id: str, session_id: str, solution) -> Callable[[str], np.ndarray | None]:
    """`depth_of(kid)` over the session's compatible live depth cache (read in place): MoGe-2's metric z at the
    solve camera's resolution, NaN where the network masked the pixel or the redaction fill covers it.
    Without a compatible cache every call returns None."""
    from tower.world_builder import live_depth as LD  # noqa: PLC0415
    from tower.world_builder.coherence_publish import _depth_params  # noqa: PLC0415
    from tower.world_builder.dense_pipeline import dense_dir, depth_trust_now  # noqa: PLC0415

    root = dense_dir(store, world_id, session_id)
    if not (root / "live_depth" / "align.json").is_file():
        return lambda kid: None
    imagery = _depth_params().imagery_source
    trust = depth_trust_now(store, world_id, session_id, imagery_source=imagery)
    session = store.read_session(world_id, session_id)
    matches = LD.compatible_live_records(store, world_id, session_id, solution, session.intrinsics, root,
                                         imagery_source=imagery, trust=trust)
    work = root / "live_depth" / "work" / "depth"
    live_ki = {kid: old_ki for _ki, (kid, _sha1, old_ki) in matches.items()}
    return lambda kid: load_depth(work, live_ki.get(kid))


def load_depth(work: Path, ki) -> np.ndarray | None:
    """`<ki>_pred.npy` with `<ki>_fill.npy` masked out (NaN), or None."""
    if ki is None:
        return None
    p = Path(work) / f"{int(ki):05d}_pred.npy"
    if not p.is_file():
        return None
    try:
        d = np.load(p).astype(np.float32)
        fp = Path(work) / f"{int(ki):05d}_fill.npy"
        if fp.is_file():
            fill = np.load(fp)
            if fill.shape != d.shape:
                return None
            d = np.where(fill.astype(bool), np.nan, d)
        return d
    except (OSError, ValueError):
        return None
