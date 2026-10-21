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
REFUSE_WORK = "work-bound"
REFUSE_ABORT = "aborted"           # the wall-clock backstop fired: an operational abort, not a certification result
REFUSE_NO_CONTROLS = "no-matched-controls"
REFUSE_GROUP_ANCHORS = "group-anchors-disagree"
REFUSE_COVERAGE = "coverage-not-shown"
REFUSE_MEMBER = "member-not-one-body"


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
    max_basins: int = 16
    # R5's profile is scaled by a typical placement scored by the fine scorer at this many lattice points.
    typical_points: int = 64
    # Bounded WORK for the whole step, counted, so a walk decides the same on a busy Tower as on an idle one:
    # every depth sample and room voxel a measurement touches (each score evaluation, scorer, rigidity
    # comparison, room frame and its consolidation, corroboration query) spends a unit. Past the budget the
    # pieces not yet decided stay cut (`work-bound`), and nothing is placed once it has run out. 1.1e10: the
    # 900 s the step was allowed before, at the Tower's measured 1.2e7 units/s (10.2 and 10.1, LANE-B2
    # section 13.3). The wall clock is only a backstop well above that (it fires on a stalled machine, and the
    # record says it fired).
    max_work: float = 1.1e10
    max_seconds: float = 3600.0
    # Bounded work: a piece of more than max_piece_keyframes keyframes is refused; at most max_measure_frames
    # of its frames (evenly spread) are measured.
    max_piece_keyframes: int = 400
    max_measure_frames: int = 40
    # The room's evidence is built from at most this many certified frames (evenly spread): its cost is
    # linear in frames, and ~1,000 frames is a long walk's room.
    max_room_frames: int = 1500
    # ...and from at most this many voxel keys in all (surface at both sizes and free space; ~8 bytes each
    # plus a surface voxel's mean): frames past it are left out, in capture order, and the record says so.
    max_room_keys: int = 40_000_000
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
    # The WALK'S OWN motion, from ALL its certified room cameras (each camera against the one nearest in time
    # to t + dt, within walk_gap_tol_s: half the ~0.2 s keyframe spacing), on gaps every walk_grid_s and made
    # monotone (an upper envelope: a longer gap never allows less), never more than the walking limit above
    # (2.0 m/s x dt + 0.3 m), which is used where fewer than walk_min_pairs pairs exist. Two uses, two numbers:
    #   * an anchor's CHECK: the walk_quantile (0.99) of that motion. A correct placement must hold EVERY
    #     anchor of a group (up to ~5): it is falsely refused at most ~5 % of the time -- an availability
    #     choice, and the check is corroboration, so it may be tight;
    #   * the search REGION: region_quantile (1.0: the largest move the walk itself made at that gap). The
    #     region must CONTAIN the truth -- a safety property: a truth outside it leaves only wrong basins to
    #     rank, which no control can test -- so it is sized by the extreme the walker demonstrably reached.
    walk_quantile: float | None = 0.99
    region_quantile: float = 1.0
    walk_gap_tol_s: float = 0.1
    walk_grid_s: float = 0.05
    walk_min_pairs: int = 50
    # Groups. G1: two pieces are one body in the solve only when they triangulate points of their own
    # together -- at least min_shared_points points seen by both and by no room camera (a point the room
    # also sees ties each piece to the room, not to each other; one shared track is not a relative pose).
    # 10: the guard's own floor for a measurement. Each member keeps at least min_frames measured frames
    # with depth, and is tested on its own at the group's placement: R4 against the envelope; its views
    # compared with the other members', and AGREEING with them, on at least min_hit_share of its samples; and
    # COVERAGE spread over its frames -- at least min_member_frames_share of its measured frames must each
    # meet the room (hit share >= min_hit_share) and agree with it no worse than the room's own controls.
    # One half: the member's similarity is what is certified, so a majority of its own frames must support
    # it; an aligned minority (a doorway) cannot carry the rest.
    min_shared_points: int = 10
    min_member_frames_share: float = 0.5
    # R0 at the prior's error: at least min_edge_good GOOD controls among those started at the region's edge
    # (at least edge_start of its radius: with six controls, the two outermost, in two different directions)
    # -- the piece's own starting error may be the whole radius --, and the false-accept test exercised on
    # SCORED wrong placements from at least min_safety_controls different controls (their misplaced
    # winners, and each control's best refined basin away from its truth, judged as if the truth were
    # withheld).
    min_edge_good: int = 2
    edge_start: float = 0.8
    min_safety_controls: int = 2
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


class _OutOfWork(Exception):
    """The step's work budget (or its wall-clock backstop) ran out inside a measurement."""


class Budget:
    """The step's work bound (PlaceParams.max_work): counted units, so the same input decides the same. The
    wall clock is a separate backstop: when it fires the step is ABORTED (an operational abort, recorded as
    such, not a certification result). `spend` raises _OutOfWork once either has run out, and keeps
    raising; `fired` says which."""

    def __init__(self, units: float, backstop_s: float):
        self.units = float(units)
        self.backstop_s = float(backstop_s)
        self.used = 0.0
        self.started = time.perf_counter()
        self.fired: str | None = None

    def spend(self, n: float = 0.0) -> None:
        self.used += float(n)
        if self.fired is None:
            if self.used > self.units:
                self.fired = "work"
            elif time.perf_counter() - self.started > self.backstop_s:
                self.fired = "wall-clock backstop"
        if self.fired is not None:
            raise _OutOfWork(self.fired)

    def out(self) -> bool:
        try:
            self.spend(0.0)
        except _OutOfWork:
            return True
        return False

    @property
    def refusal(self) -> str:
        """`work-bound` when the deterministic budget ran out; `aborted` when the wall-clock backstop fired
        (an operational abort: the step stopped, it did not certify anything either way)."""
        return REFUSE_ABORT if self.fired == "wall-clock backstop" else REFUSE_WORK


def _spend(budget: Budget | None, n: float = 0.0) -> None:
    if budget is not None:
        budget.spend(n)


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
    keys: int = 0
    truncated: bool = False

    def in_free_space(self, X: np.ndarray) -> np.ndarray:
        """Whether each point lies in a voxel the room's frames looked through more often than they saw."""
        if not len(self.free_keys) or not len(X):
            return np.zeros(len(X), bool)
        k = _voxel_keys(X, self.free_size, self.origin, clip=True)
        i = np.minimum(np.searchsorted(self.free_keys, k), len(self.free_keys) - 1)
        return self.free_keys[i] == k


def build_room(frames: list[Frame], depth_of: Callable[[str], np.ndarray | None], cam: Pinhole,
               units_per_m: float, params: PlaceParams, budget: Budget | None = None) -> Room | None:
    """The room's surface voxels and its observed free space (from at most max_room_frames frames, evenly
    spread, and at most max_room_keys voxel keys: later frames past it are left out, `truncated`), or None
    with no usable frame. Every frame and the consolidation spend from `budget`."""
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
    n_keys = 0
    truncated = False
    for f in frames:
        _spend(budget)
        if n_keys > params.max_room_keys:
            truncated = True
            break
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
        n_keys += len(keys[-1]) + len(ckeys[-1]) + len(skeys[-1])
        _spend(budget, len(X))
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
            n_keys += len(fkeys[-1])
            _spend(budget, len(P))
    if not used:
        return None
    _spend(budget, 4 * n_keys)                 # the consolidation: sorting and counting every key
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
                origin=origin, free_size=fsize, units_per_m=units_per_m, frames=used, keys=n_keys,
                truncated=truncated)


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
                 stride: int, params: PlaceParams, max_frames: int | None = None, prune: bool = True,
                 budget: Budget | None = None):
        cam = piece.cam
        self.budget = budget
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
        _spend(budget, len(frames) * len(points) + len(self.W0))

    def counts(self, t: np.ndarray, slack: float = 0.0) -> dict:
        """{agree, through, short, hit, free, valid} sample counts at translation `t`, and `closeness`: the
        agreeing samples each weighted 1 - |log(d/z)| / band (1 on the surface, 0 at the band's edge).
        `slack` (room units) widens every band by slack / z (a coarse search pass)."""
        out = {"agree": 0, "through": 0, "short": 0, "hit": 0, "free": 0, "closeness": 0.0, "valid": self.n_valid}
        _spend(self.budget, self.n_valid + len(self.Y0))
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
                 landscape: list | None = None) -> tuple[np.ndarray, float, int]:
    offs = np.arange(-half, half + 1) * step_units
    slack = step_units * math.sqrt(3.0) / 2.0
    best_t, best_s, n = np.asarray(centre, np.float64), -math.inf, 0
    for dx in offs:
        for dy in offs:
            for dz in offs:
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
    """One lattice point for every local maximum of `scores` over its lattice neighbours (up to 26): a
    point at least as high as every neighbour; a connected stretch of such points at one height (a plateau)
    is one maximum, given by its first point."""
    key = {tuple(int(x) for x in np.rint(o / step_units)): i for i, o in enumerate(offs)}
    cand = {}
    for k, i in key.items():
        sc = scores[i]
        if sc == -math.inf:
            continue
        if all(sc >= scores[j] for d in _NEIGHBOURS
               if (j := key.get((k[0] + d[0], k[1] + d[1], k[2] + d[2]))) is not None):
            cand[k] = i
    out, seen = [], set()
    for k, i in sorted(cand.items(), key=lambda kv: kv[1]):
        if k in seen:
            continue
        out.append(i)
        stack = [k]
        seen.add(k)
        while stack:
            q = stack.pop()
            for d in _NEIGHBOURS:
                r = (q[0] + d[0], q[1] + d[1], q[2] + d[2])
                if r in cand and r not in seen and scores[cand[r]] == scores[i]:
                    seen.add(r)
                    stack.append(r)
    return out


def search(piece: Piece, room: Room, centre: np.ndarray, radius_units: float, params: PlaceParams,
           budget: Budget | None = None) -> tuple:
    """(best translation, the fine scorer around it, evaluations, uniqueness, typical, basins, refined) --
    `refined` every refined basin as (translation, its fine scorer, its fine score).

    Every point of the coarse lattice covering the ball of `radius_units` about `centre` (the region the walk
    allows) is scored optimistically (`_optimistic`) on coarse voxels, cells and frames. EVERY local maximum
    of that landscape (`basins`), however low, is refined on the fine evidence -- a fine scorer around it, each
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
                    max_frames=params.coarse_frames, budget=budget)
    slack = step * math.sqrt(3.0) / 2.0
    land = [_optimistic(coarse.counts(centre + o, slack), params) for o in offs]
    evals = len(land)
    found = sorted(basins(offs, step, land), key=lambda i: -land[i])
    if len(found) > params.max_basins:
        return None, None, evals, None, None, len(found), []
    wide = Scorer(piece, room, room.points, centre, radius_units + 2 * step, params.fine_stride_px, params,
                  max_frames=params.fine_frames, budget=budget)
    pick = _spread(len(offs), params.typical_points)
    fine_typ = []
    for i in pick:
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
                      params.fine_stride_px, params, max_frames=params.fine_frames, budget=budget)
        for half, step_m in zip(halves, params.fine_steps_m):
            t, _s, k = _grid_search(fine, t, half, step_m * upm, params)
            evals += k
        refined.append((t, fine, _score(fine.counts(t), params)))
        evals += 1
    if not refined:
        return centre, Scorer(piece, room, room.points, centre, 0.05 * upm, params.fine_stride_px, params,
                              max_frames=params.fine_frames, budget=budget), evals, None, typical_fine, 0, []
    t, fine, s_win = max(refined, key=lambda r: r[2])
    far = (params.profile_m + params.coarse_step_m) * upm
    rival = max((sc for tt, _f, sc in refined if np.linalg.norm(tt - t) > far), default=-math.inf)
    span = s_win - typical_fine
    unique = (None if not (math.isfinite(s_win) and span > 0) else
              (1.0 if rival == -math.inf else (s_win - max(rival, typical_fine)) / span))
    return t, fine, evals, unique, typical_fine, len(found), refined


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


def rigidity(piece: Piece, params: PlaceParams, units_per_m: float, budget: Budget | None = None,
             member: list | None = None) -> dict:
    """The piece's own frames against each other at their scaled poses (no translation involved): of the
    comparisons that are not occlusions, the share where one frame sees through another frame's surface.
    With `member` (per frame, the group member it belongs to), only frames of DIFFERENT members are compared,
    and `members` gives each member's share of samples compared at all ("overlap") and its own counts."""
    P, fi = piece_points(piece, params, np.zeros(3), params.piece_stride_px)
    rows, cols, maps = piece.sampled(params.room_stride_px, params)
    Rs = np.stack([f.R for f in piece.frames])
    Cs = piece.centres
    agree = through = 0
    per: dict = {}
    idx = np.arange(len(piece.frames))
    for k in range(len(piece.frames)):
        others = np.flatnonzero(idx != k) if member is None else np.flatnonzero(
            np.asarray([m != member[k] for m in member]))
        sel = fi == k
        if member is not None:
            d = per.setdefault(member[k], {"points": 0, "compared": 0, "agreeing": 0, "agree": 0, "through": 0})
            d["points"] += int(sel.sum())
        if not len(others) or not sel.any():
            continue
        _spend(budget, int(sel.sum()) * len(others))
        thr, agr = seen_through(P[sel], Rs[others], Cs[others], maps[others], rows, cols, piece.cam, params,
                                params.depth_min_m * units_per_m)
        agree += int(agr.sum())
        through += int(thr.sum())
        if member is not None:
            d["compared"] += int(((thr + agr) > 0).sum())
            d["agreeing"] += int((agr > 0).sum())
            d["agree"] += int(agr.sum())
            d["through"] += int(thr.sum())
    tot = agree + through
    out = {"agree": agree, "through": through, "through_share": through / tot if tot else None}
    if member is not None:
        out["members"] = {m: dict(d, overlap=round(d["compared"] / d["points"], 4) if d["points"] else 0.0,
                                  agreeing_share=round(d["agreeing"] / d["points"], 4) if d["points"] else 0.0)
                          for m, d in per.items()}
    return out


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------


def walk_motion(frames: list[Frame], params: PlaceParams, units_per_m: float) -> Callable[[float], tuple]:
    """allowance(dt) -> (check metres, region metres, pairs, source): how far the walk's own certified room
    cameras moved over a time gap of dt -- each camera against the camera nearest in time to t + dt (within
    walk_gap_tol_s) -- as the walk_quantile (the anchor's CHECK) and the region_quantile (the search REGION,
    containment), on gaps every walk_grid_s up to anchor_dt_s, each made a monotone upper envelope over the
    gaps and read at the first grid gap at or past dt; never more than the walking limit (speed_mps x dt +
    speed_margin_m), which is also what a gap with fewer than walk_min_pairs pairs, or no quantile, uses."""
    timed = sorted((f for f in frames if f.t is not None), key=lambda f: f.t)
    ts = np.asarray([f.t for f in timed], np.float64)
    cs = np.stack([f.c for f in timed]) if timed else np.zeros((0, 3))

    def walking(dt: float) -> float:
        return params.speed_mps * dt + params.speed_margin_m

    grid = np.arange(1, int(math.ceil(params.anchor_dt_s / params.walk_grid_s - 1e-9)) + 1) * params.walk_grid_s
    rows = []
    for g in grid:
        n, q, m = 0, walking(g), walking(g)
        if params.walk_quantile is not None and len(ts) >= 2:
            target = ts + g
            j = np.clip(np.searchsorted(ts, target), 1, len(ts) - 1)
            j = np.where(np.abs(ts[j - 1] - target) <= np.abs(ts[j] - target), j - 1, j)
            ok = (np.abs(ts[j] - target) <= params.walk_gap_tol_s) & (j != np.arange(len(ts)))
            n = int(ok.sum())
            if n >= params.walk_min_pairs:
                d = np.linalg.norm(cs[j[ok]] - cs[ok], axis=1) / units_per_m
                q = min(float(np.quantile(d, params.walk_quantile)), walking(g))
                m = min(float(np.quantile(d, params.region_quantile)), walking(g))
        rows.append((q, m, n))
    # The upper envelope runs over the MEASURED gaps only (a gap too short to pair cameras is not evidence of
    # motion; it falls back to the walking limit on its own and is not carried to longer gaps).
    q_env, m_env, top_q, top_m = [], [], 0.0, 0.0
    for q, m, n in rows:
        if n >= params.walk_min_pairs:
            top_q, top_m = max(top_q, q), max(top_m, m)
            q_env.append(top_q)
            m_env.append(top_m)
        else:
            q_env.append(q)
            m_env.append(m)

    def allowance(dt: float) -> tuple:
        if not len(grid) or dt > grid[-1] + 1e-9:
            return walking(dt), walking(dt), 0, "walking-limit"
        k = int(np.searchsorted(grid, dt - 1e-9))
        q, m, n = float(q_env[k]), float(m_env[k]), rows[k][2]
        return q, m, n, ("walk" if n >= params.walk_min_pairs else "walking-limit")

    return allowance


def anchors(piece: Piece, room_frames: list[Frame], params: PlaceParams, motion=None) -> list[dict]:
    """The time-adjacent anchor estimates: for the piece's first camera, the latest anchor-eligible room
    camera before it, and for its last, the earliest after it, each within anchor_dt_s. Each gives the
    translation that puts the piece's camera on the room camera, and the allowance in metres: the walk's own
    motion over that gap (`motion`, `walk_motion`), or the walking limit without one."""
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
        walking = params.speed_mps * dt + params.speed_margin_m
        allow, contain, pairs, source = (motion(dt) if motion is not None else (walking, walking, 0, "walking-limit"))
        out.append({"side": side, "piece_kid": pc.kid, "room_kid": rc.kid, "dt_s": dt,
                    "t": rc.c - pc.c / piece.ratio, "allowance_m": allow, "containment_m": contain,
                    "allowance_pairs": pairs, "allowance_source": source, "piece_centre": pc.c / piece.ratio})
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
    refined: list = dataclasses.field(default_factory=list)     # the search's refined basins (not recorded)
    typical: float | None = None

    def record(self) -> dict:
        out = {"label": self.label, "first_keyframe_id": self.kids[0] if self.kids else None,
               "last_keyframe_id": self.kids[-1] if self.kids else None, "keyframes": len(self.kids),
               "ratio": None if not math.isfinite(self.ratio) else round(self.ratio, 6), "ci": self.ci,
               "placed": self.placed, "refused": list(self.refused),
               "translation": None if self.t is None else [float(x) for x in self.t]}
        out.update({k: v for k, v in self.stats.items()})
        return out


def regions(piece: Piece, ci_log: tuple | None, anc: list[dict], params: PlaceParams, upm: float) -> dict | None:
    """Each anchor's CHECK bound (its allowance) and REGION bound (its containment), each plus the scale
    interval's lever arm (the distance from the anchor's piece camera to where the piece meets the room, its
    samples' centroid, times the interval's largest log half-width; translation-free), written into the
    anchor (`bound_m`, `region_m`), and the REGION: the anchor with the tightest region bound (the centre
    anchor), searched over a ball of that radius. None without an anchor."""
    if not anc:
        return None
    lever = 0.0 if ci_log is None else max(abs(ci_log[0]), abs(ci_log[1]))
    P0, _fi0 = piece_points(piece, params, np.zeros(3), params.piece_stride_px)
    centroid0 = P0.mean(0) if len(P0) else np.zeros(3)
    for a in anc:
        arm = float(np.linalg.norm(a["piece_centre"] - centroid0) / upm) * lever
        a["bound_m"] = a["allowance_m"] + arm
        a["region_m"] = a.get("containment_m", a["allowance_m"]) + arm
    return min(anc, key=lambda a: a["region_m"])


def measure(label: str, piece: Piece, ci: list | None, room: Room, anc: list[dict], region: dict,
            params: PlaceParams, radius_m: float | None = None, budget: Budget | None = None) -> Placement:
    """Everything the certification reads, for a piece or a control, searched over `region`'s ball (radius
    `radius_m`, default its bound): rigidity, the search, the full counts at the placement (every measured
    frame, unpruned), the profile, uniqueness, the new geometry and every anchor's check. Refuses only what
    cannot be measured (too few frames, a region past the lattice cap, basins past max_basins, no overlap).
    Raises _OutOfWork when `budget` runs out."""
    upm = room.units_per_m
    pl = Placement(label=label, kids=[f.kid for f in piece.frames], ratio=piece.ratio, ci=ci)
    st = pl.stats
    st["frames_with_depth"] = len(piece.frames)
    if len(piece.frames) < params.min_frames:
        pl.refused.append(REFUSE_FRAMES)
        return pl
    t0 = time.perf_counter()
    radius_m = region.get("region_m", region["bound_m"]) if radius_m is None else radius_m
    centre = region["t"]
    lattice = lattice_size(radius_m * upm, params.coarse_step_m * upm, params.max_region_evaluations)
    st["region"] = {"side": region["side"], "dt_s": round(region["dt_s"], 3), "radius_m": round(radius_m, 3),
                    "lattice_points": lattice}
    if lattice > params.max_region_evaluations:
        pl.refused.append(REFUSE_REGION)
        return pl
    _spend(budget)
    st["rigidity"] = rigidity(piece, params, upm, budget)
    t, fine, evals, unique, typical, nb, refined = search(piece, room, centre, radius_m * upm, params,
                                                          budget=budget)
    pl.refined, pl.typical = refined, typical
    st["basins"] = nb
    st["evaluations"] = evals
    if t is None:
        pl.refused.append(REFUSE_COVERAGE)
        st["seconds"] = round(time.perf_counter() - t0, 3)
        return pl
    pl.t = t
    st["uniqueness"] = None if unique is None else round(unique, 4)
    st["shift_from_anchor_m"] = float(np.linalg.norm(t - centre) / upm)
    full = Scorer(piece, room, room.points, t, 0.05 * upm, params.fine_stride_px, params, prune=False,
                  budget=budget)
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
    # R6's evidence: every anchor's room camera within its CHECK bound of the placed piece camera. An anchor
    # is INDEPENDENT evidence only when it is not the region's centre AND its check could have failed INSIDE
    # the region: some lattice points within the region's radius (not the lattice's overhang past it, which
    # the region excludes anyway) lie outside its bound (the share is kept).
    checks = []
    for a in anc:
        miss = float(np.linalg.norm(t - a["t"]) / upm)
        outside = outside_share(centre, radius_m * upm, params.coarse_step_m * upm, a["t"], a["bound_m"] * upm)
        independent = a is not region and outside > 0.0
        checks.append({"side": a["side"], "dt_s": round(a["dt_s"], 3), "miss_m": round(miss, 3),
                       "bound_m": round(a["bound_m"], 3), "outside_share": round(outside, 3),
                       "centre": a is region, "member": a.get("member"),
                       "independent": bool(independent), "ok": bool(miss <= a["bound_m"])})
    st["anchor_check"] = checks
    st["seconds"] = round(time.perf_counter() - t0, 3)
    return pl


def outside_share(centre: np.ndarray, radius_units: float, step_units: float, at: np.ndarray,
                  bound_units: float) -> float:
    """The share of the region's lattice points WITHIN its radius (not the lattice's overhang past it, which
    the region excludes anyway) that lie farther than `bound_units` from `at`: what an anchor's check would
    refuse inside the region."""
    offs = region_lattice(radius_units, step_units)
    pts = centre + offs[np.linalg.norm(offs, axis=1) <= radius_units + 1e-9]
    if not len(pts):
        return 0.0
    return float(np.mean(np.linalg.norm(pts - np.asarray(at, np.float64), axis=1) > bound_units))


def envelope(controls: list[Placement], units_per_m: float, params: PlaceParams) -> dict:
    """What the room's own controls show, under a prior matched to the piece's. A control is GOOD when the
    search put it back within max_control_miss_m of its own pose and the placement is pinned and unique
    (R5). The envelope is the good controls' least agreement, most seeing-through, most free space and least
    rigidity. `usable` needs:
      * every one of control_segments controls measured, at least min_good_controls (and half) of them good;
      * at least min_edge_good good among the controls started at the region's EDGE (at least edge_start of
        its radius; a piece's own starting error may be the whole radius), from different directions;
      * a rigidity bound;
      * the false-accept test passed and exercised: every WRONG placement -- a control's winner farther than
        max_control_miss_m from its truth, and each control's best refined basin that far from its truth,
        judged as if the truth were withheld (`withheld_probe`) -- is refused by R3-R5 (R6 is not credited:
        a piece can be corroborated by another piece, which no control can be tested on), with SCORED wrong
        placements (ones that met the room) from at least min_safety_controls different controls (safety
        must not rest on one, nor on a probe refused only because it met nothing)."""
    for c in controls:
        if c.t is not None:
            truth = np.asarray(c.stats.get("true_translation") or np.zeros(3), np.float64)
            c.stats["miss_m"] = round(float(np.linalg.norm(c.t - truth)) / units_per_m, 4)

    def pinned(c):
        return (min((c.stats.get("profile_drop") or {"-": 0.0}).values()) >= params.min_peak_drop
                and (c.stats.get("uniqueness") or 0.0) >= params.min_peak_drop)

    good = [c for c in controls if c.t is not None and c.stats.get("agree_share") is not None
            and not c.refused and c.stats["miss_m"] <= params.max_control_miss_m and pinned(c)]
    edge = [c for c in controls if (c.stats.get("start_error_m") or 0.0)
            >= params.edge_start * ((c.stats.get("region") or {}).get("radius_m") or 0.0) - 1e-6]
    good_edge = [c for c in good if c in edge]
    out = {"controls": len(controls), "good": len(good), "edge": len(edge), "good_edge": len(good_edge),
           "edge_directions": len({tuple(np.sign(np.round(c.stats.get("start_direction") or [0, 0, 0], 6)))
                                   for c in good_edge}), "usable": False}
    if len(controls) < params.control_segments:
        out["why"] = f"{len(controls)} of the {params.control_segments} controls could be formed"
        return out
    if len(good) < max(params.min_good_controls, (len(controls) + 1) // 2):
        out["why"] = "too few of the room's own stretches are placed back where they are"
        return out
    if len(good_edge) < params.min_edge_good or out["edge_directions"] < min(2, params.min_edge_good):
        out["why"] = "the stretches started at the region's edge are not placed back (from more than one direction)"
        return out
    rig = [c.stats["rigidity"]["through_share"] for c in good if c.stats["rigidity"]["through_share"] is not None]
    if not rig:
        out["why"] = "no good control's rigidity could be measured"
        return out
    out.update(min_agree_share=min(c.stats["agree_share"] for c in good),
               max_through_share=max(c.stats["through_share"] for c in good),
               max_free_share=max(c.stats["free_share"] for c in good),
               max_rigidity_through=max(rig))
    probes = []
    for c in controls:
        if c.t is not None and c.stats.get("miss_m", 0.0) > params.max_control_miss_m:
            probes.append((c.label, c.label, dict(c.stats), list(c.refused)))
        if c.stats.get("withheld_probe"):
            probes.append((c.label + "/withheld", c.label, dict(c.stats["withheld_probe"]), []))
    accepted = []
    scored = set()
    for label, src, st, refused in probes:
        probe = Placement(label=label, kids=[], ratio=1.0, ci=None, t=np.zeros(3), stats=st, refused=refused)
        certify(probe, dict(out, usable=True), params, r6=False)
        if not probe.refused:
            accepted.append(label)
        if st.get("agree_share") is not None:
            scored.add(src)
    out["misplaced_accepted"] = accepted
    out["safety_probes"] = len(probes)
    out["safety_controls"] = len(scored)
    if accepted:
        out["why"] = "the rule would accept a stretch of the room's own frames in the wrong place"
    elif out["safety_controls"] < params.min_safety_controls:
        out["why"] = (f"the false-accept test was exercised by {out['safety_controls']} control(s); "
                      f"{params.min_safety_controls} are needed")
    else:
        out["usable"] = True
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
    st["r3_decided"] = True
    # A group: every member on its own at the group's placement, and its views against the other members'.
    for ms in (st.get("member_checks") or {}).values():
        ct = ms.get("cross_through_share")
        fs = ms.get("frame_shares") or []
        covered = (sum(1 for h, a in fs if h >= params.min_hit_share and a >= env["min_agree_share"]) / len(fs)
                   if fs else 0.0)
        ms["frames_covered"] = round(covered, 4)
        if (ms.get("hit_share") is None or ms["hit_share"] < params.min_hit_share
                or covered < params.min_member_frames_share
                or ms.get("cross_agreeing", 0.0) < params.min_hit_share
                or ms.get("agree_share") is None or ms["agree_share"] < env["min_agree_share"]
                or ms["through_share"] > env["max_through_share"] or ms["free_share"] > env["max_free_share"]
                or ms.get("overlap", 0.0) < params.min_hit_share
                or ct is None or ct > env["max_rigidity_through"]):
            pl.refused.append(REFUSE_MEMBER)
            break
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


def corroborate(placements: list[Placement], params: PlaceParams, units_per_m: float,
                budget: Budget | None = None) -> None:
    """R6, corroboration: a piece that passed R0-R5 is placed when an INDEPENDENT anchor (not its region's
    centre, a check that could have failed) agrees with it, or -- chaining only from such a piece -- when its
    new geometry coincides with an already corroborated piece's (median nearest neighbour <= cross_median_m
    over >= cross_min_points points within cross_overlap_m). Two pieces never vouch for each other alone."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    ok = [p for p in placements if not p.refused]
    done = [p for p in ok if str(p.stats.get("corroborated_by") or "").startswith("group:")]
    for p in ok:
        if p in done:
            continue
        if any(a["ok"] and a.get("independent") for a in p.stats.get("anchor_check") or []):
            p.stats["corroborated_by"] = "independent-anchor"
            done.append(p)

    def coincides(p, q) -> dict:
        if p.new_points is None or q.new_points is None or not len(p.new_points) or not len(q.new_points):
            return {"with": q.label, "overlap_points": 0, "median_m": None, "ok": False}
        _spend(budget, len(p.new_points) + len(q.new_points))
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


def own_points(solution, kids: list, room_kids: set, index: dict) -> np.ndarray:
    """The 3-D points `kids` observe that no room keyframe observes (sorted unique indices): what the piece
    triangulates without the room."""
    obs = np.asarray(solution.observations, np.int64).reshape(-1, 3)
    nk = len(solution.keyframe_ids)
    obs = obs[(obs[:, 0] >= 0) & (obs[:, 0] < nk)]
    room = np.zeros(nk, bool)
    room[[index[k] for k in room_kids if k in index]] = True
    mine = np.zeros(nk, bool)
    mine[[index[k] for k in kids if k in index]] = True
    seen_by_room = np.unique(obs[room[obs[:, 0]], 2])
    return np.setdiff1d(np.unique(obs[mine[obs[:, 0]], 2]), seen_by_room, assume_unique=True)


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
    R4; then the groups' decisions; last, R6's corroboration of the single pieces. Every measurement spends
    from one counted work budget (max_work): past it the pieces not yet decided stay cut (`work-bound`), and
    when it has run out by the end nothing is placed. A wall-clock backstop firing aborts the step
    (`aborted`, recorded in `rec["aborted"]`): nothing is placed, and that is not a certification result."""
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

    budget = Budget(params.max_work, params.max_seconds)

    def frame(kid: str, level: float) -> Frame | None:
        p = poses.get(kid)
        if not p or p.get("rotation") is None or p.get("translation") is None:
            return None
        R, c = centre_of(p)
        if not (np.isfinite(R).all() and np.isfinite(c).all()):
            return None
        return Frame(kid=kid, R=R, c=c, s=_frame_scale(metric_log.get(name_of.get(kid, kid)), level, band),
                     t=times.get(kid), index=index.get(kid, -1))

    def piece_of(kids, level, ratio, cap=None) -> Piece:
        frames, depths, scales = [], [], []
        for k in [kids[i] for i in _spread(len(kids), params.max_measure_frames if cap is None else cap)]:
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
                          budget=budget)
    except _OutOfWork:
        room, room_late = None, True
    motion = walk_motion(room_frames, params, upm)       # ALL certified room cameras: poses held out of the
    #                                                      depth evidence still show how the walker moved
    rec["room"] = {"seconds": round(time.perf_counter() - t_room, 3), "frames": 0 if room is None else room.frames,
                   "voxels": 0 if room is None else len(room.points),
                   "free_voxels": 0 if room is None else len(room.free_keys),
                   "keys": 0 if room is None else room.keys, "truncated": bool(room is not None and room.truncated),
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
            pl.refused.append(REFUSE_FRAMES if not len(vals) else (budget.refusal if room_late else REFUSE_ROOM))
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
        anc = anchors(piece, anchor_pool, params, motion)
        region = regions(piece, ci_rel, anc, params, upm)
        pl.stats["anchors"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in a.items()
                                if k not in ("piece_centre", "t")} for a in anc]
        if region is None:
            pl.refused.append(REFUSE_NO_ANCHOR)
            continue
        try:
            m = measure(label, piece, p.get("ci"), room, anc, region, params, budget=budget)
        except _OutOfWork:
            pl.refused.append(budget.refusal)
            continue
        m.stats.update(ci_log=pl.stats["ci_log"], anchors=pl.stats["anchors"])
        m.kids = kids
        piece_tests(m, params)
        placements[-1] = m
        measured.append((m, region))

    # 1b. Pieces of one blow-up, measured as ONE rigid body (module docstring, GROUPS). G1: they triangulate at
    # least min_shared_points points together that no room camera sees; G2: their intervals overlap.
    groups = []
    rec["group_links"] = []
    if room is not None and len(info) >= 2:
        room_kids = set(assessment.get("room_keyframes") or [])
        own = {n: own_points(solution, d["kids"], room_kids, index) for n, d in info.items()}
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
                a_, b_ = info[i]["abs"], info[j]["abs"]
                shared = int(len(np.intersect1d(own[i], own[j], assume_unique=True)))
                overlap = max(a_[0], b_[0]) <= min(a_[1], b_[1])
                if shared or overlap:
                    rec["group_links"].append({"pieces": [f"piece-{i}", f"piece-{j}"], "shared_points": shared,
                                               "intervals_overlap": bool(overlap)})
                if shared >= params.min_shared_points and overlap:
                    parent[find(i)] = find(j)
        members: dict = {}
        for n in ns:
            members.setdefault(find(n), []).append(n)
        has_depth = lambda k: depth_of(k) is not None and frame(k, ref) is not None  # noqa: E731
        for gi, mem in enumerate(m for m in members.values() if len(m) >= 2):
            g = _measure_group(f"group-{gi}", mem, info, params, ref, upm, room, anchor_pool, index, piece_of,
                               log_values, has_depth, motion, budget)
            groups.append(g)
            if not g["placement"].refused:
                measured.append((g["placement"], g["region"]))

    # 2. The room's controls, under each standing item's prior error, and R0, R3, R4 -- each prior's radius
    # rounded UP to 0.1 m so priors share their controls (a control then faces a region at least the item's
    # own: never easier). GROUPS FIRST, and decided (their R6: an independent member anchor) before any single
    # piece is: a placed group decides its members, whose own measurements are then moot and spend no work.
    cache: dict = {}
    rec["controls"] = {}

    def judge(m) -> None:
        key = math.ceil(round(m.stats["region"]["radius_m"] * 10.0, 6)) / 10.0
        if key not in cache:
            controls = []
            for i, seg in enumerate(ctrl):
                try:
                    c = _matched_control(f"control-{i}", i, seg, key, room, params, upm, ref, log_values, piece_of,
                                         frame, budget)
                except _OutOfWork:
                    break
                if c is not None:
                    controls.append(c)
            cache[key] = (controls, envelope(controls, upm, params))
            rec["controls"][f"r{key}"] = {
                "controls": [_jsonable({k: v for k, v in c.record().items() if k not in ("ci", "placed")})
                             for c in controls], "envelope": cache[key][1]}
        if budget.out():                               # nothing is certified past the bound
            m.refused.append(budget.refusal)
            return
        m.stats["envelope"] = f"r{key}"
        certify(m, cache[key][1], params, r6=not m.stats.get("group"))

    for m, _region in measured:
        if m.stats.get("group") and not m.refused:
            judge(m)
    for g in groups:
        _group_decision(g, placements, info)
    for m, _region in measured:
        # a single piece still standing, and still itself (not replaced by its group's placement)
        if not m.stats.get("group") and not m.refused and any(pl is m for pl in placements):
            judge(m)

    # 3. R6's corroboration of the single pieces (after the groups' decisions: a member a group refuses never
    # vouches for another piece).
    if room is not None:
        try:
            corroborate(placements, params, upm, budget)
        except _OutOfWork:
            for pl in placements:
                if not pl.refused and not str(pl.stats.get("corroborated_by") or "").startswith("group:"):
                    pl.refused.append(budget.refusal)
    # The last check: nothing is placed once the budget (or its backstop) has run out.
    if budget.out():
        for pl in placements:
            if not pl.refused:
                pl.refused.append(budget.refusal)
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
    rec["work"] = {"used": budget.used, "budget": budget.units, "fired": budget.fired}
    if budget.fired == "wall-clock backstop":
        # Not a certification result: the same input on a machine that did not stall decides by its work alone.
        rec["aborted"] = {"by": "wall-clock backstop", "after_s": params.max_seconds}
    return rec


def _measure_group(label, mem, info, params, ref, upm, room, anchor_pool, index, piece_of, log_values,
                   has_depth, motion=None, budget=None) -> dict:
    """One group measured as a piece, with every member's own evidence kept:
      * the pooled scale; as its interval (R1, the lever) the UNION of its members' guard intervals -- never
        narrower than any member's (pooling them would need each isolation round's core, which the guard's
        audit does not keep); the keyframe cap on the whole group;
      * measured frames RESERVED per member: at least min_frames with depth each (and its share of
        max_measure_frames); a member without them refuses the group (R2);
      * every member's anchors under the one similarity (all bounds enforced; the tightest is the search's
        centre, the rest are checks that can fail);
      * the search, R5 and R6's contradictions on the pooled evidence;
      * at the group's placement, every MEMBER's counts (R4, hit share) and its views against the other
        members' (`rigidity(member=...)`: overlap share and through share), judged in `certify`."""
    kids = sorted({k for n in mem for k in info[n]["kids"]}, key=lambda k: index.get(k, 0))
    vals = log_values(kids)
    level = float(np.median(vals))
    ratio = math.exp(level - ref)
    lo = min(info[n]["abs"][0] for n in mem)
    hi = max(info[n]["abs"][1] for n in mem)
    ci_log = (lo - level, hi - level)
    pl = Placement(label=label, kids=kids, ratio=ratio, ci=None)
    pl.stats["ci_log"] = [round(ci_log[0], 6), round(ci_log[1], 6)]
    out = {"members": list(mem), "placement": pl, "region": None, "record": {}}
    if ci_log[1] - ci_log[0] > math.log(params.max_ci_width):
        pl.refused.append(REFUSE_CI)
        return out
    if len(kids) > params.max_piece_keyframes:
        pl.refused.append(REFUSE_TOO_LARGE)
        return out
    total = sum(len(info[n]["kids"]) for n in mem)
    chosen = []
    for n in mem:
        kd = [k for k in info[n]["kids"] if has_depth(k)]
        want = min(len(kd), max(params.min_frames,
                                int(round(params.max_measure_frames * len(info[n]["kids"]) / total))))
        chosen += [kd[i] for i in _spread(len(kd), want)]
    chosen = sorted(set(chosen), key=lambda k: index.get(k, 0))
    union = piece_of(chosen, level, ratio, cap=len(chosen))
    member_of = {k: n for n in mem for k in info[n]["kids"]}
    per = {n: sum(1 for f in union.frames if member_of.get(f.kid) == n) for n in mem}
    out["record"]["measured_frames"] = {f"piece-{n}": per[n] for n in mem}
    if any(v < params.min_frames for v in per.values()):
        pl.refused.append(REFUSE_FRAMES)
        return out
    anc = []
    for n in mem:
        member = piece_of(info[n]["kids"], level, ratio)
        for a in anchors(member, anchor_pool, params, motion):
            a["member"] = f"piece-{n}"
            anc.append(a)
    region = regions(union, ci_log, anc, params, upm)
    pl.stats["anchors"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in a.items()
                            if k not in ("piece_centre", "t")} for a in anc]
    if region is None:
        pl.refused.append(REFUSE_NO_ANCHOR)
        return out
    try:
        m = measure(label, union, None, room, anc, region, params, budget=budget)
        m.stats.update(ci_log=pl.stats["ci_log"], anchors=pl.stats["anchors"], group=True)
        m.kids = kids
        piece_tests(m, params)
        if REFUSE_CORROBORATION in m.refused:
            # A member anchor contradicts the one similarity. Whether that refuses the members depends on R3
            # and R5, which need the controls: kept aside, and decided in _group_decision.
            m.refused.remove(REFUSE_CORROBORATION)
            m.stats["anchors_disagree"] = True
        if m.t is not None and REFUSE_OVERLAP not in m.refused:
            flist = [member_of[f.kid] for f in union.frames]
            cross = rigidity(union, params, upm, budget, member=flist)
            m.stats["cross_rigidity"] = {k: cross[k] for k in ("agree", "through", "through_share")}
            mstats = {}
            for n in mem:
                sel = [i for i, x in enumerate(flist) if x == n]
                sub = Piece(frames=[union.frames[i] for i in sel], depths=[union.depths[i] for i in sel],
                            scales=[union.scales[i] for i in sel], ratio=union.ratio, cam=union.cam)
                c = Scorer(sub, room, room.points, m.t, 0.05 * upm, params.fine_stride_px, params, prune=False,
                           budget=budget).counts(m.t)
                frames = []
                for i in sel:                      # coverage, frame by frame
                    one = Piece(frames=[union.frames[i]], depths=[union.depths[i]], scales=[union.scales[i]],
                                ratio=union.ratio, cam=union.cam)
                    cf = Scorer(one, room, room.points, m.t, 0.05 * upm, params.fine_stride_px, params,
                                prune=False, budget=budget).counts(m.t)
                    frames.append([round(cf["hit"] / cf["valid"], 4) if cf["valid"] else 0.0,
                                   round(cf["agree"] / cf["hit"], 4) if cf["hit"] else 0.0])
                cm = (cross.get("members") or {}).get(n) or {}
                ct = cm.get("agree", 0) + cm.get("through", 0)
                mstats[f"piece-{n}"] = {
                    "frames": len(sel), "hit_share": c["hit"] / c["valid"] if c["valid"] else None,
                    "agree_share": c["agree"] / c["hit"] if c["hit"] else None,
                    "through_share": c["through"] / c["hit"] if c["hit"] else None,
                    "free_share": c["free"] / c["valid"] if c["valid"] else None,
                    "overlap": cm.get("overlap", 0.0), "cross_agreeing": cm.get("agreeing_share", 0.0),
                    "cross_through_share": cm.get("through", 0) / ct if ct else None, "frame_shares": frames}
            m.stats["member_checks"] = mstats
    except _OutOfWork:
        pl.refused.append(budget.refusal)
        return out
    out.update(placement=m, region=region)
    return out


_R5 = (REFUSE_PEAK, REFUSE_AMBIGUOUS)


def _group_decision(g: dict, placements: list[Placement], info: dict) -> None:
    """In the order declared (LANE-B2 sections 12.1 and 13):
      1. not one rigid body -- R3 on the union, or a member failing on its own at the group's placement or
         unseen by the other members (`member-not-one-body`): the members' own decisions stand;
      2. R3 never decided (the group stopped before its controls: R5, overlap, R0, work): the members' own
         decisions stand -- an anchor disagreeing with an undecided or ambiguous winner says nothing;
      3. R5 failed: the members' own decisions stand;
      4. an anchor of any member disagreeing with the group's placement: the group AND every member refused;
      5. any other refusal: the members' own decisions stand;
      6. R6: an independent member anchor (it could have failed inside the region) agrees -- else the
         members' own decisions stand; with it, every member takes the group's similarity."""
    gpl = g["placement"]
    st = gpl.stats
    alone = "refused: members decided alone"
    if REFUSE_RIGID in gpl.refused or REFUSE_MEMBER in gpl.refused:
        g["decision"] = "not one rigid body: members decided alone"
        return
    if not st.get("r3_decided") or any(r in gpl.refused for r in _R5):
        g["decision"] = alone
        return
    if st.get("anchors_disagree"):
        gpl.refused.append(REFUSE_GROUP_ANCHORS)
        g["decision"] = "refused, members refused (anchors disagree)"
        for n in g["members"]:
            if REFUSE_GROUP_ANCHORS not in placements[n].refused:
                placements[n].refused.append(REFUSE_GROUP_ANCHORS)
        return
    if gpl.refused:
        g["decision"] = alone
        return
    if not any(a["ok"] and a.get("independent") for a in st.get("anchor_check") or []):
        gpl.refused.append(REFUSE_CORROBORATION)
        g["decision"] = alone
        return
    st["corroborated_by"] = "independent-anchor"
    g["decision"] = "placed"
    for n in g["members"]:
        m = Placement(label=f"piece-{n}", kids=list(info[n]["kids"]), ratio=gpl.ratio, ci=info[n].get("ci"),
                      t=gpl.t)
        m.stats.update(placed_with=gpl.label, corroborated_by=f"group:{gpl.label}",
                       ci_log=[round(x, 6) for x in info[n]["ci_log"]])
        placements[n] = m


_START_DIRECTIONS = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                              [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])


def _matched_control(label, i, seg, radius, room, params, upm, ref, log_values, piece_of, frame, budget=None):
    """Control `i` under a piece's prior error. A piece's starting error is unknown anywhere in its region
    (up to its radius), so the controls' starting errors span it: control i starts from its own true
    translation displaced by radius x (i + 1) / control_segments (a sixth of the radius to all of it), along
    the solve's six axis directions in turn, and is searched over a ball of the piece's radius about that
    start. A control carries every error a piece carries, its scale too: its level is its own frames' median
    ratio, so it is scaled by exp(level - ref) about the solve's origin and its true translation is what
    undoes that at its own cameras. Its WITHHELD-TRUTH probe: its best refined basin farther than
    max_control_miss_m from its truth, measured as the winner would be were the truth not there (counts,
    profile, uniqueness among the other such basins). None when it has no frame with depth."""
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
    c = measure(label, piece, None, room, [a], a, params, radius_m=radius, budget=budget)
    c.stats["true_translation"] = truth.tolist()
    c.stats["ratio"] = ratio
    c.stats["start_error_m"] = round(float(np.linalg.norm(offset)) / upm, 4)
    c.stats["start_direction"] = _START_DIRECTIONS[i % len(_START_DIRECTIONS)].tolist()
    wrong = [(t, sc, s) for t, sc, s in c.refined
             if math.isfinite(s) and np.linalg.norm(t - truth) / upm > params.max_control_miss_m]
    if wrong and c.typical is not None and math.isfinite(c.typical):
        t1, sc1, s1 = max(wrong, key=lambda r: r[2])
        far = (params.profile_m + params.coarse_step_m) * upm
        rival = max((s for t, _sc, s in wrong if np.linalg.norm(t - t1) > far), default=-math.inf)
        span = s1 - c.typical
        unique = None if not span > 0 else (1.0 if rival == -math.inf else (s1 - max(rival, c.typical)) / span)
        full = Scorer(piece, room, room.points, t1, 0.05 * upm, params.fine_stride_px, params, prune=False,
                      budget=budget).counts(t1)
        probe = {"miss_m": round(float(np.linalg.norm(t1 - truth)) / upm, 4), "rigidity": c.stats.get("rigidity"),
                 "uniqueness": None if unique is None else round(unique, 4)}
        if full["hit"] and _score(full, params) > -math.inf:
            probe.update(agree_share=full["agree"] / full["hit"], through_share=full["through"] / full["hit"],
                         free_share=full["free"] / full["valid"], hit_share=full["hit"] / full["valid"],
                         profile_drop={k: round(v, 4) for k, v in profile(sc1, t1, upm, params, c.typical).items()})
        c.stats["withheld_probe"] = probe
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
    moved = []
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
        moved.append({"id": PLACE_ID, "piece": pl.label, "ratio": float(pl.ratio), "t": [float(x) for x in t],
                      "points": np.flatnonzero(sel).astype(int).tolist()})
    # Exactly which points each placement moved, beside the solution: a re-gate reverses only these
    # (`coherence_publish.solver_candidate`).
    new_solution = dataclasses.replace(solution, poses=poses, xyz=xyz.astype(np.float32),
                                       solve=dict(solution.solve or {}, placed_by_depth=moved))
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
