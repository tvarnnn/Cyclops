"""Depth-placed pieces (`world_builder/place_pieces.py`): the placement maths, the room's own controls, every
refusal path, applying a placement, and the step at the guard seam behind its switch.

The world: a closed box room with furniture, its depth rendered exactly and then given a depth network's
error (an affine error per frame and per-pixel noise; the guard's per-camera ratio measures the frame's
scale). ROOM cameras walk one half of it; a PIECE of cameras looks at the other half. The piece's solve
positions are blown up x`K` about its first camera (the walk 10.2 mechanism: one hinge, a similarity about
it) and its metric log says so. The search must put every piece camera back where it stood. Whether a
placement is ACCEPTED is decided against the room's own controls, whose noise the real walks set (DEV:
walk 10.2 places its three bathroom pieces, LANE-B2.md); here the decision logic is pinned on explicit
envelopes.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import time
import types

import numpy as np
import pytest

from tower.world_builder import final_scale_guard as FSG
from tower.world_builder import global_solve as GS
from tower.world_builder import place_pieces as PP

UPM = 2.0          # room units per metre (the room's reference log = log 2)
K = 50.0           # the piece's blow-up: its ratio to the room
CAM = {"fx": 40.0, "fy": 40.0, "cx": 32.0, "cy": 24.0, "width": 64, "height": 48}
ROOM_BOX = (np.array([0.0, 0.0, 0.0]), np.array([4.0, 2.5, 3.0]))
FURNITURE = [(np.array([0.4, 1.7, 0.3]), np.array([1.1, 2.5, 1.0])),
             (np.array([3.2, 1.9, 2.2]), np.array([4.0, 2.5, 3.0])),
             (np.array([2.9, 0.0, 0.0]), np.array([4.0, 0.6, 0.5]))]
# Small images: sample densely so the tests see enough of the room.
# The synthetic room is small (60 frames): two controls of ten frames, and two good ones suffice here.
PARAMS = PP.PlaceParams(room_stride_px=2, piece_stride_px=2, fine_stride_px=2, coarse_stride_px=4, edge_px=1,
                        free_stride_px=2, voxel_m=0.10, coarse_voxel_m=0.20, coarse_frames=12,
                        control_segments=2, control_frames=10, min_good_controls=2,
                        # The synthetic room's cameras barely move (p99 ~0.16 m over 0.3 s) while its pieces stand
                        # 0.3 m from their anchors: the walking limit is these tests' allowance. The walk's own
                        # motion has its own tests (WALK below).
                        walk_quantile=None)
WALK = dataclasses.replace(PARAMS, walk_quantile=0.99)
N_ROOM = 60
SID = "s"


def _look_at(c, target):
    """World -> camera rotation (rows x right, y down, z forward) for a camera at `c` looking at `target`;
    the world's down is +y (the floor is y = 2.5)."""
    z = np.asarray(target, float) - c
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def _noisy(depth, rng, on=True):
    """A depth network's error, roughly as MoGe-2's: an affine error per frame (scale off by ~10 %, offset by
    ~8 cm) and 2 % per-pixel noise. Returns (the frame's measured log(true / predicted) -- what the guard's
    per-camera ratio reads, a median over the frame --, the prediction)."""
    f = float(np.exp(0.10 * rng.standard_normal()))
    b = 0.08 * rng.standard_normal()
    pred = (depth * f + b) * (1.0 + 0.02 * rng.standard_normal(depth.shape))
    if not on:
        return 0.0, depth.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmedian(np.log(depth / pred))), pred.astype(np.float32)


def _render(R, c, solids=FURNITURE, room=ROOM_BOX):
    """Metric z depth (H, W) seen from (R, c) inside `room` with `solids` (axis-aligned boxes)."""
    H, W = CAM["height"], CAM["width"]
    gv, gu = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    r = np.stack([(gu - CAM["cx"]) / CAM["fx"], (gv - CAM["cy"]) / CAM["fy"], np.ones_like(gu)], -1).reshape(-1, 3)
    d = r @ R  # world directions (unnormalised; the camera z component is 1, so t along d is z depth)
    lo, hi = room
    with np.errstate(divide="ignore", invalid="ignore"):
        t_exit = np.where(d > 0, (hi - c) / d, np.where(d < 0, (lo - c) / d, np.inf)).min(1)
        best = t_exit
        for blo, bhi in solids:
            t1 = (blo - c) / d
            t2 = (bhi - c) / d
            tn = np.nanmax(np.minimum(t1, t2), 1)
            tf = np.nanmin(np.maximum(t1, t2), 1)
            hit = (tf >= tn) & (tn > 0)
            best = np.where(hit & (tn < best), tn, best)
    return best.reshape(H, W).astype(np.float32)


def _pose(R, c):
    return {"component": 0, "rotation": R.reshape(-1).tolist(), "translation": (-R @ c).tolist(),
            "observations": 100}


class World:
    """Room cameras (kids r00..), piece cameras (p00..), their depth, times, metric log and the assessment."""

    def __init__(self, *, n_piece=12, piece_jitter_m=0.0, anchor_offset_m=(0.3, 0.0, 0.0), piece_dt=0.3,
                 ci=None, piece_looks=None, piece_forward=None, piece_step=(0.08, 0.0, 0.05), room_solids=None,
                 piece_solids=None, noise=True, tail_room=0, tail_offset_m=(0.1, 0.0, 0.0), piece_log_jitter=0.0,
                 tail_dt=None, split=None, link_pieces=True, second_log_offset=0.0, second_log_jitter=0.0,
                 second_looks=None, second_offset_m=None, second_offset_from=0):
        rng = np.random.default_rng(7)
        self.depth, self.poses, self.times, self.metric = {}, {}, {}, {}
        self.true = {}
        room_kids = []
        # Room: N_ROOM cameras in the near half, sweeping their gaze round the room twice.
        for i in range(N_ROOM):
            c = np.array([1.0 + 1.5 * i / (N_ROOM - 1), 1.2, 1.0 + 0.6 * math.sin(i / 8)])
            ang = 4 * math.pi * i / N_ROOM
            # a little downward, as a walker looks: the floor fixes the vertical
            R = _look_at(c, c + np.array([math.cos(ang), 0.35 + 0.15 * math.sin(3 * ang), math.sin(ang)]))
            kid = f"{SID}:r{i:02d}"
            room_kids.append(kid)
            self.true[kid] = c
            self.poses[kid] = _pose(R, c * UPM)
            m, self.depth[kid] = _noisy(_render(R, c, solids=room_solids if room_solids is not None else FURNITURE),
                                        rng, noise)
            self.times[kid] = 0.2 * i                 # keyframes ~5 a second, as a walk's
            self.metric[kid] = math.log(UPM) + m      # log(z_sfm / z_metric), as the guard measures it
        last_room = self.true[room_kids[-1]]
        # Piece: cameras in the far half, looking at the far corner, the floor and the furniture there.
        first = last_room + np.asarray(anchor_offset_m)
        looks = piece_looks or [np.array([4.0, 2.1, 3.0]), np.array([3.5, 2.5, 2.5]), np.array([4.0, 1.9, 1.5]),
                                np.array([2.5, 2.2, 3.0])]
        hinge = first * UPM
        self.piece_kids = []
        for j in range(n_piece):
            c = first + np.asarray(piece_step) * j
            if split is not None and j >= split and second_looks is not None:
                R = _look_at(c, np.asarray(second_looks[j % len(second_looks)], float))
            elif piece_forward is not None:
                R = _look_at(c, c + np.asarray(piece_forward, float))
            else:
                R = _look_at(c, looks[j % len(looks)] + np.array([0.0, 0.1 * (j % 3), 0.0]))
            kid = f"{SID}:p{j:02d}"
            self.piece_kids.append(kid)
            self.true[kid] = c
            blown = (c * UPM - hinge) * K + hinge
            if piece_jitter_m:
                blown = blown + rng.normal(scale=piece_jitter_m * UPM * K, size=3)
            if split is not None and j >= split + second_offset_from and second_offset_m is not None:
                blown = blown + np.asarray(second_offset_m, float) * UPM * K   # the solve put this half elsewhere
            self.poses[kid] = _pose(R, blown)
            m, self.depth[kid] = _noisy(_render(R, c, solids=piece_solids if piece_solids is not None else FURNITURE),
                                        rng, noise)
            self.times[kid] = 0.2 * (N_ROOM - 1) + piece_dt + 0.2 * j
            self.metric[kid] = math.log(UPM * K) + m + (piece_log_jitter * rng.standard_normal()
                                                         if piece_log_jitter else 0.0)
            if split is not None and j >= split:
                self.metric[kid] += second_log_offset + second_log_jitter * rng.standard_normal()
        # Room frames AFTER the piece (the walker back in the room): an anchor on the piece's other side.
        tail_kids = []
        last_piece = self.true[self.piece_kids[-1]] if self.piece_kids else last_room
        for i in range(tail_room):
            c = last_piece + np.asarray(tail_offset_m) + np.array([-0.05 * i, 0.0, -0.03 * i])
            ang = 2 * math.pi * i / max(tail_room, 1) + 2.0
            R = _look_at(c, c + np.array([math.cos(ang), 0.35, math.sin(ang)]))
            kid = f"{SID}:t{i:02d}"
            tail_kids.append(kid)
            self.true[kid] = c
            self.poses[kid] = _pose(R, c * UPM)
            m, self.depth[kid] = _noisy(_render(R, c, solids=room_solids if room_solids is not None else FURNITURE),
                                        rng, noise)
            self.times[kid] = self.times[self.piece_kids[-1]] + (piece_dt if tail_dt is None else tail_dt) + 0.2 * i
            self.metric[kid] = math.log(UPM) + m
        self.room_kids = room_kids + tail_kids
        self.kids = room_kids + self.piece_kids + tail_kids
        # The room's sparse points: what each room camera sees on a pixel grid, observed by that camera (the
        # depth stage fits each room frame's prediction to them).
        xyz, obs, xy = [], [], []
        for kid in self.room_kids:
            i = self.kids.index(kid)
            R, c = PP.centre_of(self.poses[kid])
            D = self.depth[kid]
            for v in range(3, CAM["height"], 6):
                for u in range(3, CAM["width"], 6):
                    ray = np.array([(u + 0.5 - CAM["cx"]) / CAM["fx"], (v + 0.5 - CAM["cy"]) / CAM["fy"], 1.0])
                    obs.append((i, len(xy), len(xyz)))
                    xyz.append(c + R.T @ (ray * D[v, u] * UPM))
                    xy.append((u + 0.49, v + 0.49))
        # The piece's own tracks: a point seen by each pair of consecutive piece frames (in the blown-up solve's
        # units). With `split` and not `link_pieces`, no track joins the two halves.
        for j in range(len(self.piece_kids) - 1 if split is not None else 0):
            if split is not None and not link_pieces and j + 1 == split:
                continue
            ka, kb = self.piece_kids[j], self.piece_kids[j + 1]
            Ra, ca = PP.centre_of(self.poses[ka])
            for q in range(12):                       # twelve points the pair triangulates on its own
                obs.append((self.kids.index(ka), len(xy), len(xyz)))
                xy.append((32.0, 24.0))
                obs.append((self.kids.index(kb), len(xy), len(xyz)))
                xy.append((32.0, 24.0))
                xyz.append(ca + Ra.T @ np.array([0.01 * q, 0.0, 1.0]) * UPM * K)
        if split is not None and tail_kids:
            # Twelve points both halves see that a room camera sees too (a doorway both stretches look through):
            # they tie each half to the room, not the halves to each other.
            for q in range(12):
                for kb in (self.piece_kids[split - 1], self.piece_kids[split], tail_kids[0]):
                    obs.append((self.kids.index(kb), len(xy), len(xyz)))
                    xy.append((32.0, 24.0))
                xyz.append(PP.centre_of(self.poses[self.piece_kids[split]])[1] + q)
        self.points = (np.asarray(xyz), np.asarray(obs), np.asarray(xy))
        self.solution = types_solution(self.kids, self.poses, xyz=xyz, obs=obs, xy=xy)
        self.assessment = {
            "reference": {"log": math.log(UPM)}, "room_keyframes": list(self.room_kids), "uncertified": [],
            "uncertified_keyframes": 0, "scale_status": "certified", "why": None, "evidence": FSG.EVIDENCE_LIVE,
            "pieces": [{"keyframe_ids": list(self.piece_kids), "ci": ci or [K * 0.97, K * 1.03],
                        "reason": FSG.REASON_SCALE_OUTLIER, "first_index": N_ROOM}]}
        if split is not None:
            self.assessment["pieces"] = [
                {"keyframe_ids": list(self.piece_kids[:split]), "ci": [K * 0.97, K * 1.03],
                 "reason": FSG.REASON_SCALE_OUTLIER, "first_index": N_ROOM},
                {"keyframe_ids": list(self.piece_kids[split:]), "ci": [K * 0.97, K * 1.03],
                 "reason": FSG.REASON_SCALE_OUTLIER, "first_index": N_ROOM + split}]
        # What the guard keeps for each piece: its interval unrounded, against its round's core.
        core = np.asarray([self.metric[k] for k in self.room_kids])
        for p in self.assessment["pieces"]:
            vals = np.asarray([self.metric[k] for k in p["keyframe_ids"]])
            p["ci_log"] = list(FSG.bootstrap_interval_log(vals, core, FSG.GuardParams(), p["first_index"]))
            p["core_log_median"] = float(np.median(core))

    def depth_of(self, kid):
        return self.depth.get(kid)

    def plan(self, params=PARAMS):
        return PP.plan(self.solution, self.assessment, self.metric, {k: k for k in self.kids}, self.times,
                       self.depth_of, params)


def types_solution(kids, poses, xyz=None, obs=None, xy=None):
    xyz = np.zeros((0, 3), np.float32) if xyz is None else np.asarray(xyz, np.float32)
    obs = np.zeros((0, 3), np.int32) if obs is None else np.asarray(obs, np.int32)
    xy = np.zeros((len(obs), 2), np.float32) if xy is None else np.asarray(xy, np.float32)
    p = len(xyz)
    first = np.zeros(p, np.int32)
    for row in obs[::-1]:
        first[row[2]] = row[0]
    return GS.Solution(solver="glomap", solved_at=0.0, input_digest=None, keyframe_ids=list(kids), poses=poses,
                       components=[{"index": 0, "images": len(kids), "points": p}], xyz=xyz,
                       rgb=np.zeros((p, 3), np.uint8), component=np.zeros(p, np.int32), first_keyframe=first,
                       track_length=np.full(p, 2, np.int32), error=np.full(p, 0.5, np.float32),
                       observations=obs, observation_xy=xy, camera=dict(CAM))


def _placed_centres(world, pl):
    return {k: (PP.centre_of(world.poses[k])[1] / pl.ratio + pl.t) / UPM for k in world.piece_kids}


def _room(w, params=PARAMS, leave_out=()):
    frames = [PP.Frame(kid=k, R=PP.centre_of(w.poses[k])[0], c=PP.centre_of(w.poses[k])[1],
                       s=math.exp(w.metric[k]), t=w.times[k], index=i)
              for i, k in enumerate(w.kids) if k not in w.piece_kids]
    room = PP.build_room([f for f in frames if f.kid not in leave_out], w.depth_of, PP.Pinhole.of(CAM), UPM, params)
    return room, frames


def _piece(w, kids=None):
    kids = kids or w.piece_kids
    level = float(np.median([w.metric[k] for k in kids]))
    ratio = math.exp(level - math.log(UPM))
    frames = [PP.Frame(kid=k, R=PP.centre_of(w.poses[k])[0], c=PP.centre_of(w.poses[k])[1],
                       s=math.exp(w.metric[k]), t=w.times[k]) for k in kids]
    return PP.Piece(frames=frames, depths=[w.depth[k] for k in kids], scales=[f.s / ratio for f in frames],
                    ratio=ratio, cam=PP.Pinhole.of(CAM))


def _measure(w, params=PARAMS, ci_rel=(-0.03, 0.03)):
    """A piece measured the way plan measures it: anchors, the tightest anchor's region, the search."""
    room, frames = _room(w, params)
    piece = _piece(w)
    anc = PP.anchors(piece, frames, params)
    region = PP.regions(piece, ci_rel, anc, params, UPM)
    return PP.measure("piece-0", piece, [K * 0.97, K * 1.03], room, anc, region, params)


ENV_OK = {"usable": True, "controls": 6, "good": 6, "min_agree_share": 0.5, "max_through_share": 0.07,
          "max_free_share": 0.12, "max_rigidity_through": 0.05}


def _measured(**over):
    """A measured placement that passes ENV_OK and R5."""
    pl = PP.Placement(label="piece-0", kids=["a", "b"], ratio=2.0, ci=[1.9, 2.1], t=np.zeros(3))
    pl.stats.update({"rigidity": {"through_share": 0.01}, "agree_share": 0.7, "through_share": 0.03,
                     "free_share": 0.05, "profile_drop": {"x-": 0.5, "x+": 0.5, "y-": 0.5, "y+": 0.5, "z-": 0.5,
                                                          "z+": 0.5}, "uniqueness": 0.4,
                     "anchor_check": [{"ok": True, "centre": True, "independent": False},
                                      {"ok": True, "centre": False, "independent": True}]})
    pl.stats.update(over)
    return pl


# ---------------------------------------------------------------------------
# the maths
# ---------------------------------------------------------------------------


def test_the_search_puts_a_blown_up_piece_back_where_it_stood():
    w = World(noise=False)
    assert _piece(w).ratio == pytest.approx(K, rel=1e-9)
    pl = _measure(w)
    assert not pl.refused
    err = [np.linalg.norm(c - w.true[k]) for k, c in _placed_centres(w, pl).items()]
    assert max(err) < 0.05, err                      # metres; the finest search step is 2.5 cm
    st = pl.stats
    assert st["agree_share"] > 0.9 and st["through_share"] < 0.01 and st["free_share"] < 0.01
    assert all(a["ok"] for a in st["anchor_check"])
    assert min(st["profile_drop"].values()) >= PARAMS.min_peak_drop and st["uniqueness"] >= PARAMS.min_peak_drop


def test_the_search_still_finds_it_through_a_depth_networks_error():
    w = World()
    pl = _measure(w)
    err = [np.linalg.norm(c - w.true[k]) for k, c in _placed_centres(w, pl).items()]
    assert max(err) < 0.25, err                      # the piece's own measured scale is off by a few per cent


def test_the_anchor_estimate_is_where_the_room_camera_stood():
    w = World(anchor_offset_m=(0.0, 0.0, 0.0), noise=False)
    _room_, frames = _room(w)
    anc = PP.anchors(_piece(w), frames, PARAMS)
    assert [a["side"] for a in anc] == ["before"]
    a = anc[0]
    assert a["dt_s"] == pytest.approx(0.3)
    assert a["allowance_m"] == pytest.approx(2.0 * 0.3 + 0.3)
    # The estimate puts the piece's first camera exactly on the room camera (no walking offset here).
    first = PP.centre_of(w.poses[w.piece_kids[0]])[1] / K + a["t"]
    np.testing.assert_allclose(first / UPM, w.true[w.piece_kids[0]], atol=1e-9)


def test_a_rigid_piece_agrees_with_itself_and_a_jittered_one_does_not():
    assert PP.rigidity(_piece(World()), PARAMS, UPM)["through_share"] < 0.01
    assert PP.rigidity(_piece(World(piece_jitter_m=0.25)), PARAMS, UPM)["through_share"] > 0.2


def test_the_score_needs_the_piece_to_meet_the_room():
    c = {"agree": 5, "closeness": 5.0, "through": 0, "short": 0, "hit": 5, "free": 0, "valid": 100}
    assert PP._score(c, PARAMS) == -math.inf          # 5 % of the samples meet the room: no evidence
    c.update(hit=20, agree=18, closeness=12.0, through=1, free=2)
    assert PP._score(c, PARAMS) == pytest.approx((12.0 - 2 * (1 + 2)) / 100)
    # Meeting more of the room as closely scores higher; meeting it less closely scores lower.
    assert PP._score(dict(c, hit=40, agree=36, closeness=24.0), PARAMS) > PP._score(c, PARAMS)
    assert PP._score(dict(c, closeness=6.0), PARAMS) < PP._score(c, PARAMS)


def test_a_depth_edge_is_not_a_sample():
    D = np.full((10, 10), 2.0, np.float32)
    D[:, 5:] = 4.0                                     # a step from 2 m to 4 m between columns 4 and 5
    rows, cols = np.arange(10), np.arange(10)
    s = PP.sampled_depth(D, rows, cols, dataclasses.replace(PARAMS, edge_px=1))
    assert np.isnan(s[:, 4]).all() and np.isnan(s[:, 5]).all()
    assert np.isfinite(s[:, :4]).all() and np.isfinite(s[:, 6:]).all()


def test_a_surface_the_room_looked_through_more_than_it_saw_is_not_room_evidence():
    # A cabinet stood in the room for a few frames only (a door shut for a while): the rest looked through it.
    from scipy.spatial import cKDTree

    cabinet = (np.array([2.9, 0.8, 1.5]), np.array([3.6, 2.5, 2.2]))
    clean = World(noise=False)
    w = World(noise=False)
    for k in [k for k in w.kids if k not in w.piece_kids][:4]:
        R, c = PP.centre_of(w.poses[k])
        w.depth[k] = _render(R, c / UPM, solids=FURNITURE + [cabinet])
    seen = PP.build_room([f for f in _room(w)[1]][:4], w.depth_of, PP.Pinhole.of(CAM), UPM,
                         dataclasses.replace(PARAMS, min_voxel_frames=1))
    extra = cKDTree(_room(clean)[0].points).query(seen.points)[0] / UPM > 0.15
    assert extra.sum() > 50                            # those frames did put the cabinet in the room ...
    room, _ = _room(w)
    gone = cKDTree(_room(clean)[0].points).query(room.points)[0] / UPM > 0.15
    assert gone.sum() < 0.1 * extra.sum()              # ... and the vote takes it out again


# ---------------------------------------------------------------------------
# the room's own controls and the certification
# ---------------------------------------------------------------------------


def test_controls_are_spread_held_out_and_searched_under_the_pieces_prior():
    w = World(tail_room=6)
    rec = w.plan()
    assert rec["room"]["frames"] == len(w.room_kids) - PARAMS.control_segments * PARAMS.control_frames
    (prior,) = rec["controls"].values()               # one standing piece, one matched prior
    ctrl = prior["controls"]
    assert len(ctrl) == PARAMS.control_segments and all(c["keyframes"] == PARAMS.control_frames for c in ctrl)
    piece = rec["pieces"][0]
    # the controls' region: the piece's radius rounded UP to 0.1 m (never easier than the piece's own)
    radius = math.ceil(round(piece["region"]["radius_m"] * 10, 6)) / 10
    assert radius >= piece["region"]["radius_m"]
    # their starting errors span that region, up to all of its radius
    assert sorted(c["start_error_m"] for c in ctrl) == pytest.approx(
        [radius * (i + 1) / PARAMS.control_segments for i in range(PARAMS.control_segments)], abs=0.01)
    for c in ctrl:
        # searched over that radius, about a start that far from the control's truth
        assert c["region"]["radius_m"] == pytest.approx(radius, abs=0.01)
        assert c["anchor_check"][0]["miss_m"] <= radius + 0.01
        assert c.get("miss_m") is not None
        # a control carries its own measured scale, as a piece does
        kids = w.kids[w.kids.index(c["first_keyframe_id"]):w.kids.index(c["last_keyframe_id"]) + 1]
        assert c["ratio"] == pytest.approx(math.exp(float(np.median([w.metric[k] for k in kids])) - math.log(UPM)))
        assert c["ratio"] != pytest.approx(1.0, abs=1e-6)


# A wrong placement R5 refuses (its best refined basin away from its truth, judged as if the truth were withheld).
WITHHELD_REFUSED = {"miss_m": 0.8, "rigidity": {"through_share": 0.01}, "agree_share": 0.7, "through_share": 0.03,
                    "free_share": 0.05, "hit_share": 0.5, "uniqueness": 0.02,
                    "profile_drop": {"x-": 0.5, "x+": 0.5, "y-": 0.5, "y+": 0.5, "z-": 0.5, "z+": 0.5}}


_CONTROL_IDS = itertools.count()


def _control(miss_m, start=1.0, withheld=WITHHELD_REFUSED, **stats):
    """A measured control `miss_m` from its truth, started `start` x the region's radius (1 m) from it, along
    the six start directions in turn."""
    c = _measured(**stats)
    i = next(_CONTROL_IDS)
    c.label = f"control-{i}"
    c.t = np.array([miss_m * UPM, 0.0, 0.0])
    c.stats.update(true_translation=[0.0, 0.0, 0.0], start_error_m=start, region={"radius_m": 1.0},
                   start_direction=PP._START_DIRECTIONS[i % 6].tolist())
    if withheld is not None:
        c.stats["withheld_probe"] = dict(withheld)
    return c


def test_the_envelope_is_the_good_controls_and_refuses_a_room_it_cannot_trust():
    P6 = dataclasses.replace(PARAMS, control_segments=6, min_good_controls=3)
    good = [_control(0.05, agree_share=0.6), _control(0.1, through_share=0.06), _control(0.02, free_share=0.11),
            _control(0.03), _control(0.04)]
    env = PP.envelope(good + [_control(0.9, agree_share=0.2)], UPM, P6)
    assert env["usable"] and env["good"] == 5 and env["min_agree_share"] == 0.6
    assert env["max_through_share"] == 0.06 and env["max_free_share"] == 0.11
    # A misplaced control the depth tests (R3-R5) would ACCEPT: unusable -- whatever its anchor says.
    for anchor in ([{"ok": True, "independent": True}], [{"ok": False, "independent": True}], []):
        sneaky = _control(0.9, anchor_check=anchor)
        env = PP.envelope(good + [sneaky], UPM, P6)
        assert not env["usable"] and env["misplaced_accepted"] == [sneaky.label]
    # Fewer than min_good_controls good (here 2 of 6): unusable.
    bad = [_control(0.9, agree_share=0.2) for _ in range(4)]
    assert not PP.envelope(good[:2] + bad, UPM, P6)["usable"]
    # Fewer controls than control_segments: unusable.
    assert not PP.envelope(good, UPM, P6)["usable"]
    # Half good is not enough when that is fewer than min_good_controls (2 of 4 here).
    P4 = dataclasses.replace(P6, control_segments=4)
    assert not PP.envelope(good[:2] + bad[:2], UPM, P4)["usable"]
    assert PP.envelope(good[:3] + bad[:1], UPM, P4)["usable"]
    # No good control's rigidity: no R3 bound, unusable.
    norig = [_control(0.02, rigidity={"through_share": None}) for _ in range(5)]
    assert not PP.envelope(norig + [_control(0.9, agree_share=0.2)], UPM, P6)["usable"]


def test_r0_needs_recoveries_at_the_regions_edge_from_more_than_one_direction():
    # Five good controls, but at the edge (at least 0.8 of the radius) only one is good -- the other edge
    # start is misplaced: the room has not shown it recovers a stretch from the piece's largest possible error.
    P6 = dataclasses.replace(PARAMS, control_segments=6, min_good_controls=3, min_edge_good=2)
    inner = [_control(0.05, start=s) for s in (0.17, 0.33, 0.5, 0.67)]
    env = PP.envelope(inner + [_control(0.05, start=0.83), _control(0.9, start=1.0, agree_share=0.2)], UPM, P6)
    assert env["good"] == 5 and env["good_edge"] == 1 and not env["usable"]
    env = PP.envelope(inner[:3] + [_control(0.05, start=0.83), _control(0.05, start=1.0),
                                   _control(0.9, start=0.67, agree_share=0.2)], UPM, P6)
    assert env["good_edge"] == 2 and env["edge_directions"] == 2 and env["usable"]
    # Two good edge recoveries from ONE direction are not enough.
    a, b = _control(0.05, start=0.83), _control(0.05, start=1.0)
    b.stats["start_direction"] = a.stats["start_direction"]
    env = PP.envelope(inner[:3] + [a, b, _control(0.9, start=0.67, agree_share=0.2)], UPM, P6)
    assert env["edge_directions"] == 1 and not env["usable"]


def test_r0_safety_must_be_exercised_by_more_than_one_control():
    P6 = dataclasses.replace(PARAMS, control_segments=6, min_good_controls=3, min_safety_controls=2)
    good = [_control(0.05, withheld=None) for _ in range(5)]
    # One misplaced control refused, and nothing else wrong ever tried: the false-accept test rests on it.
    env = PP.envelope(good + [_control(0.9, agree_share=0.2, withheld=None)], UPM, P6)
    assert env["safety_controls"] == 1 and not env["usable"]
    # A second control's withheld-truth basin, refused by R5: exercised twice, and usable.
    good[0].stats["withheld_probe"] = dict(WITHHELD_REFUSED)
    env = PP.envelope(good + [_control(0.9, agree_share=0.2, withheld=None)], UPM, P6)
    assert env["safety_controls"] == 2 and env["usable"]
    # A withheld-truth basin the depth tests would ACCEPT: the rule would place a stretch in the wrong place.
    good[1].stats["withheld_probe"] = dict(WITHHELD_REFUSED, uniqueness=0.5)
    env = PP.envelope(good + [_control(0.9, agree_share=0.2, withheld=None)], UPM, P6)
    assert not env["usable"] and env["misplaced_accepted"] == [good[1].label + "/withheld"]


def test_a_misplaced_control_with_nothing_measured_does_not_break_the_room():
    # A control the search could not score (no samples met the room): it carries its own refusal into R0's
    # probe, and it does not raise.
    P6 = dataclasses.replace(PARAMS, control_segments=6, min_good_controls=3)
    good = [_control(0.05) for _ in range(5)]
    blind = _control(0.9, agree_share=None, through_share=None, free_share=None)
    blind.refused.append(PP.REFUSE_OVERLAP)
    env = PP.envelope(good + [blind], UPM, P6)
    assert env["usable"] and env["misplaced_accepted"] == []
    # A misplaced control refused while measured (its samples barely met the room) stays refused in the
    # probe, even when its shares, as far as they go, look fine.
    thin = _control(0.9)
    thin.refused.append(PP.REFUSE_OVERLAP)
    env = PP.envelope(good + [thin], UPM, P6)
    assert env["usable"] and env["misplaced_accepted"] == []


def test_certify_passes_a_piece_as_consistent_as_the_room():
    pl = _measured()
    PP.certify(pl, ENV_OK, PARAMS)
    assert pl.refused == []


@pytest.mark.parametrize("over, reason", [
    ({"rigidity": {"through_share": 0.2}}, PP.REFUSE_RIGID),
    ({"agree_share": 0.4}, PP.REFUSE_AGREE),
    ({"through_share": 0.1}, PP.REFUSE_THROUGH),
    ({"free_share": 0.2}, PP.REFUSE_FREE),
    ({"profile_drop": {"x-": 0.5, "x+": 0.05, "y-": 0.5, "y+": 0.5, "z-": 0.5, "z+": 0.5}}, PP.REFUSE_PEAK),
    ({"uniqueness": 0.05}, PP.REFUSE_AMBIGUOUS),
    ({"uniqueness": None}, PP.REFUSE_AMBIGUOUS),
])
def test_certify_refuses_each_way_a_piece_is_worse_than_the_room(over, reason):
    pl = _measured(**over)
    PP.certify(pl, ENV_OK, PARAMS)
    assert pl.refused == [reason]


def test_nothing_is_placed_in_a_room_whose_evidence_cannot_place_its_own_frames():
    pl = _measured()
    PP.certify(pl, dict(ENV_OK, usable=False), PARAMS)
    assert pl.refused == [PP.REFUSE_ROOM]


# ---------------------------------------------------------------------------
# refusals end to end: every one leaves the piece cut
# ---------------------------------------------------------------------------


def test_a_scale_known_no_better_than_the_band_is_refused_and_no_control_is_paid_for():
    w = World(piece_log_jitter=0.6)                  # the piece's frames' ratios scatter: a wide interval
    rec = w.plan()
    lo, hi = rec["pieces"][0]["ci_log"]
    assert hi - lo > math.log(PARAMS.max_ci_width)
    assert rec["pieces"][0]["refused"] == [PP.REFUSE_CI] and rec["placed"] == 0
    assert rec["controls"] == {}                      # nothing standing: the controls are not run


def test_r1_reads_the_guards_unrounded_interval_not_its_display():
    w = World(ci=[K / 1.4, K * 1.1], tail_room=6)    # a wide DISPLAY interval; the frames' ratios agree
    rec = w.plan()
    assert PP.REFUSE_CI not in rec["pieces"][0]["refused"]
    w = World(ci=[K * 0.999, K * 1.001], piece_log_jitter=0.6)    # a tight display, scattered ratios
    assert w.plan()["pieces"][0]["refused"] == [PP.REFUSE_CI]
    # The kept interval is what is read -- not one recomputed against another core -- and without it, R1
    # cannot be decided.
    w = World(tail_room=6)
    w.assessment["pieces"][0]["ci_log"] = [0.0, 0.5]
    assert w.plan()["pieces"][0]["refused"] == [PP.REFUSE_CI]
    del w.assessment["pieces"][0]["ci_log"]
    assert w.plan()["pieces"][0]["refused"] == [PP.REFUSE_CI]


def test_the_interval_is_the_guards_bootstrap_unrounded():
    gp = FSG.GuardParams()
    rng = np.random.default_rng(5)
    core = rng.normal(0.0, 0.05, size=300)
    piece = math.log(0.0009) + rng.normal(0.0, 0.15, size=15)
    lo, hi = FSG.bootstrap_interval_log(piece, core, gp, 7)
    shown = FSG.bootstrap_interval(piece, core, gp, salt=7)            # the guard's, rounded to 4 decimals
    assert shown[0] == pytest.approx(round(math.exp(lo), 4)) and shown[1] == pytest.approx(round(math.exp(hi), 4))
    # At collapsed ratios the display rounding flips R1 (x1.25) for some pieces; the unrounded one decides.
    flips = 0
    for k in range(40):
        piece = math.log(0.0004 + 0.00005 * k) + rng.normal(0.0, 0.12, size=15)
        lo, hi = FSG.bootstrap_interval_log(piece, core, gp, k)
        a, b = FSG.bootstrap_interval(piece, core, gp, salt=k)
        flips += (hi - lo <= math.log(1.25)) != (a > 0 and b / a <= 1.25)
    assert flips > 0


def test_too_few_frames_with_depth_are_refused():
    w = World(n_piece=6)
    assert PP.REFUSE_FRAMES in w.plan()["pieces"][0]["refused"]


def test_a_piece_no_room_camera_was_near_in_time_is_refused():
    w = World(piece_dt=30.0)
    assert PP.REFUSE_NO_ANCHOR in w.plan()["pieces"][0]["refused"]


def test_a_piece_that_is_not_a_similarity_is_refused():
    w = World(piece_jitter_m=0.25, tail_room=6)
    rec = w.plan()
    assert rec["pieces"][0]["rigidity"]["through_share"] > 0.2 and rec["placed"] == 0


def _counts_at_truth(w):
    room, _frames = _room(w)
    piece = _piece(w)
    k0 = w.piece_kids[0]
    t_true = w.true[k0] * UPM - PP.centre_of(w.poses[k0])[1] / piece.ratio
    c = PP.Scorer(piece, room, room.points, t_true, 0.1 * UPM, PARAMS.fine_stride_px, PARAMS, prune=False).counts(t_true)
    return c


def test_a_piece_that_sees_through_the_rooms_surface_is_refused():
    # The room was captured with a cabinet the piece never saw (it looks through where the cabinet stood).
    cabinet = (np.array([3.0, 0.8, 1.8]), np.array([3.9, 2.5, 2.9]))
    w = World(room_solids=FURNITURE + [cabinet], noise=False, tail_room=6)
    c = _counts_at_truth(w)
    assert c["through"] / c["hit"] > 0.1             # even where it truly stood
    rec = w.plan()
    assert rec["pieces"][0]["refused"] and rec["placed"] == 0


def test_a_piece_with_geometry_where_the_room_saw_empty_space_is_refused():
    # The piece's frames hold a cabinet where the room's cameras looked through empty space.
    cabinet = (np.array([3.0, 0.8, 1.8]), np.array([3.9, 2.5, 2.9]))
    w = World(piece_solids=FURNITURE + [cabinet], noise=False, tail_room=6)
    c = _counts_at_truth(w)
    assert c["free"] / c["valid"] > 0.1              # even where it truly stood
    rec = w.plan()
    assert rec["pieces"][0]["refused"] and rec["placed"] == 0


def test_a_piece_that_sees_one_plane_is_not_pinned():
    # A bare room; every piece camera faces the far wall square on from 1.2 m: it can slide along the wall.
    w = World(anchor_offset_m=(0.3, 0.0, -0.29), piece_forward=(1.0, 0.0, 0.0), piece_step=(0.0, 0.0, 0.05),
              room_solids=[], piece_solids=[], noise=False)
    pl = _measure(w)
    PP.certify(pl, ENV_OK, PARAMS)
    assert PP.REFUSE_PEAK in pl.refused and PP.REFUSE_AMBIGUOUS in pl.refused
    rec = w.plan()
    assert rec["placed"] == 0 and PP.REFUSE_PEAK in rec["pieces"][0]["refused"]


def test_a_piece_its_anchor_puts_elsewhere_is_not_corroborated():
    # The room camera next in time stood 1.24 m from the piece's first camera, 0.3 s earlier: no walk does
    # that. The depth fit still finds where the piece stood (past the region's edge, by the lattice's reach
    # and the fine passes), and the anchor's check then FAILS: R6 can fail.
    w = World(anchor_offset_m=(0.3, 0.0, -1.2), noise=False)
    pl = _measure(w)
    (a,) = pl.stats["anchor_check"]
    assert a["centre"] and not a["ok"] and a["miss_m"] > a["bound_m"]
    PP.certify(pl, ENV_OK, PARAMS)
    assert PP.REFUSE_CORROBORATION in pl.refused


def test_the_region_is_the_tightest_anchors_ball_and_its_lattice_covers_it():
    step = 0.3
    for r in (0.5, 1.0, 2.0):
        g = PP.region_lattice(r, step)
        # every point of the ball is within the lattice's reach of a lattice point
        rng = np.random.default_rng(1)
        v = rng.normal(size=(2000, 3))
        v = v / np.linalg.norm(v, axis=1)[:, None] * r * rng.uniform(0, 1, size=(2000, 1)) ** (1 / 3)
        from scipy.spatial import cKDTree
        assert cKDTree(g).query(v)[0].max() <= step * math.sqrt(3) / 2 + 1e-9
    w = World(noise=False, tail_room=4, tail_dt=0.6)  # a room camera 0.3 s before, another 0.6 s after
    pl = _measure(w)
    reg = pl.stats["region"]
    assert reg["side"] == "before"                   # the tighter of the two
    assert reg["dt_s"] == pytest.approx(0.3) and 0.9 <= reg["radius_m"] < 1.0      # allowance + the lever arm
    assert pl.stats["evaluations"] >= reg["lattice_points"]


def test_only_an_independent_anchor_corroborates():
    # One anchor: the region's own centre. Its check holds by construction -- it is no corroboration.
    pl = _measure(World(noise=False))
    (a,) = pl.stats["anchor_check"]
    assert a["centre"] and a["ok"] and not a["independent"]
    PP.certify(pl, ENV_OK, PARAMS)
    assert pl.refused == []
    PP.corroborate([pl], PARAMS, UPM)
    assert pl.refused == [PP.REFUSE_CORROBORATION] and not pl.placed
    # A room camera on the piece's other side, 0.3 s after it: a check the search could have failed.
    pl = _measure(World(noise=False, tail_room=4))
    other = [a for a in pl.stats["anchor_check"] if not a["centre"]]
    assert other and other[0]["independent"] and other[0]["ok"] and other[0]["outside_share"] > 0
    PP.certify(pl, ENV_OK, PARAMS)
    PP.corroborate([pl], PARAMS, UPM)
    assert pl.placed and pl.stats["corroborated_by"] == "independent-anchor"
    # The same camera 1.5 s later: its ball holds every lattice point of the region, so no placement the
    # search could rank would fail it -- it agrees, and corroborates nothing.
    pl = _measure(World(noise=False, tail_room=4, tail_dt=1.5))
    other = [a for a in pl.stats["anchor_check"] if not a["centre"]]
    assert other and other[0]["ok"] and other[0]["outside_share"] == 0 and not other[0]["independent"]
    PP.certify(pl, ENV_OK, PARAMS)
    PP.corroborate([pl], PARAMS, UPM)
    assert pl.refused == [PP.REFUSE_CORROBORATION]


def test_a_region_larger_than_the_search_may_cover_is_refused_not_sampled_coarser():
    # The room camera next in time is 1.5 s away: a 3.3 m walking ball, more lattice points than the cap.
    pl = _measure(World(noise=False, piece_dt=1.5))
    assert pl.refused == [PP.REFUSE_REGION] and pl.t is None
    assert pl.stats["region"]["lattice_points"] > PARAMS.max_region_evaluations


def test_an_anchor_the_placement_disagrees_with_is_a_refusal():
    pl = _measured(anchor_check=[{"ok": True, "centre": True}, {"ok": False, "centre": False, "independent": True}])
    PP.certify(pl, ENV_OK, PARAMS)
    assert pl.refused == [PP.REFUSE_CORROBORATION]
    # ... decided before any control is paid for.
    pl = _measured(anchor_check=[{"ok": True, "centre": True}, {"ok": False, "centre": False, "independent": True}])
    PP.piece_tests(pl, PARAMS)
    assert pl.refused == [PP.REFUSE_CORROBORATION]


def test_a_parallel_wall_alias_is_refused_or_placed_at_the_truth():
    # A partition parallel to the far wall, 1.3 m in front of it, over half the wall; the piece faces the far
    # wall and its anchor puts it 1.3 m back (the walker stepped 1.3 m in 0.6 s through a doorway). Both the
    # truth and the alias are inside the region. The rule must refuse it or place it where it stood.
    partition = (np.array([2.6, 0.0, 0.0]), np.array([2.7, 2.5, 1.4]))
    solids = FURNITURE + [partition]
    w = World(anchor_offset_m=(1.3, 0.0, 0.2), piece_dt=0.6, piece_forward=(1.0, 0.0, 0.0),
              piece_step=(0.0, 0.0, 0.05), room_solids=solids, piece_solids=solids, noise=False, tail_room=4,
              tail_offset_m=(-1.3, 0.0, 0.0))
    rec = w.plan()
    pl = rec["placements"][0]
    if pl.placed:
        err = max(np.linalg.norm(c - w.true[k]) for k, c in _placed_centres(w, pl).items())
        assert err < 0.1, (err, pl.refused)
    else:
        assert pl.refused


def test_plan_places_a_piece_end_to_end_when_the_room_vouches(monkeypatch):
    # Every step of plan() on the real search: R1 on the recomputed interval, R2, anchors, the region, R5,
    # R6 by the independent anchor, then the placement. The envelope's statistics are pinned on their own
    # above; here the room's controls are given a realistic envelope (a near-noise-free synthetic room's own
    # controls agree ~99 % and leave no room for any other view of it).
    w = World(tail_room=6)
    real = PP.envelope
    seen = []

    def envelope(controls, upm, params):
        seen.append(len(controls))
        return dict(real(controls, upm, params), **ENV_OK)

    monkeypatch.setattr(PP, "envelope", envelope)
    rec = w.plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    pl = rec["placements"][0]
    assert seen == [PARAMS.control_segments]                 # the matched-prior controls did run
    assert pl.placed, pl.refused
    assert pl.stats["corroborated_by"] == "independent-anchor"
    err = max(np.linalg.norm(c - w.true[k]) for k, c in _placed_centres(w, pl).items())
    assert err < 0.2                                         # the depth network's affine error, ~8 cm offsets


def test_cross_piece_corroboration_only_chains_from_an_independently_anchored_piece():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 1, size=(400, 3)) * UPM

    def piece(label, independent, shift=0.0):
        p = PP.Placement(label=label, kids=[label], ratio=1.0, ci=[1, 1])
        p.new_points = pts + shift * UPM
        p.stats["anchor_check"] = [{"ok": True, "centre": True, "independent": False}] + (
            [{"ok": True, "centre": False, "independent": True}] if independent else [])
        return p

    # Two pieces with no independent anchor never vouch for each other, however well they agree.
    a, b = piece("a", False), piece("b", False, 0.02)
    PP.corroborate([a, b], PARAMS, UPM)
    assert not a.placed and not b.placed
    # One independently anchored piece corroborates the other through coincident new geometry (2 cm) ...
    a, b = piece("a", True), piece("b", False, 0.02)
    PP.corroborate([a, b], PARAMS, UPM)
    assert a.placed and b.placed and b.stats["corroborated_by"] == "chained:a"
    # ... but not geometry 25 cm away.
    a, c = piece("a", True), piece("c", False, 0.25)
    PP.corroborate([a, c], PARAMS, UPM)
    assert a.placed and not c.placed


# ---------------------------------------------------------------------------
# applying a placement
# ---------------------------------------------------------------------------


def _placed_record(w):
    pl = PP.Placement(label="piece-0", kids=list(w.piece_kids), ratio=K, ci=[K * 0.97, K * 1.03], placed=True)
    k0 = w.piece_kids[0]
    pl.t = w.true[k0] * UPM - PP.centre_of(w.poses[k0])[1] / K
    return pl


def test_placing_moves_the_pieces_cameras_and_points_and_nothing_else():
    w = World(noise=False)
    kids = w.kids
    ki = {k: i for i, k in enumerate(kids)}
    p0, r0 = ki[w.piece_kids[0]], ki[kids[0]]
    # Two more points. A: seen by three piece cameras and one room camera -> the piece's. B: the room's.
    rx, ro, rxy = w.points
    n = len(rx)
    xyz = np.concatenate([rx, [[100.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
    obs = np.concatenate([ro, [[p0, 0, n], [p0 + 1, 0, n], [p0 + 2, 0, n], [r0, 0, n], [r0, 1, n + 1],
                               [r0 + 1, 1, n + 1]]])
    xy = np.concatenate([rxy, np.full((6, 2), 5.0)])
    w.solution = types_solution(kids, w.poses, xyz=xyz, obs=obs, xy=xy)
    pl = _placed_record(w)
    sol, ass = PP.apply_placements(w.solution, w.assessment, [pl])
    for k in w.piece_kids:
        R0, c0 = PP.centre_of(w.solution.poses[k])
        R1, c1 = PP.centre_of(sol.poses[k])
        np.testing.assert_allclose(R1, R0)
        np.testing.assert_allclose(c1, c0 / pl.ratio + pl.t, atol=1e-6)
        np.testing.assert_allclose(c1 / UPM, w.true[k], atol=1e-6)     # where it stood
        assert sol.poses[k]["component"] == 0
    for k in kids[:N_ROOM]:
        assert sol.poses[k] == w.solution.poses[k]
    np.testing.assert_allclose(sol.xyz[n], xyz[n] / pl.ratio + pl.t, rtol=1e-6)
    np.testing.assert_allclose(sol.xyz[:n], np.asarray(xyz[:n], np.float32))
    np.testing.assert_allclose(sol.xyz[n + 1], xyz[n + 1])
    assert ass["pieces"] == []
    assert ass["scale_status"] == FSG.SCALE_UNCERTIFIED
    assert ass["uncertified"][-1]["uncertified"] == PP.UNCERTIFIED_DEPTH_PLACED
    assert set(w.piece_kids) <= set(ass["room_keyframes"])
    # Nothing placed: both come back unchanged, the same objects.
    pl.placed = False
    assert PP.apply_placements(w.solution, w.assessment, [pl]) == (w.solution, w.assessment)


# ---------------------------------------------------------------------------
# the switch at the guard seam
# ---------------------------------------------------------------------------


def test_with_the_switch_off_the_guard_seam_is_untouched(monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_PLACE_PIECES", raising=False)
    w = World(noise=False)
    called = []
    monkeypatch.setattr(PP, "plan", lambda *a, **k: called.append(1))
    out = FSG._place_pieces(None, "w", SID, w.solution, w.assessment, w.metric, [], {}, "result")
    assert out[0] is w.solution and out[1] is w.assessment and out[2] == "result" and not called
    # Off, the guard's summary carries nothing of this step (the output is today's but for elapsed time).
    assert "pieces_placed" not in FSG.summary(w.assessment, mode_=FSG.MODE_ON, decision=FSG.DECISION_PUBLISHED,
                                              seconds=0.0)


def test_with_the_switch_on_a_failure_leaves_every_piece_cut(monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_PLACE_PIECES", "true")
    w = World(noise=False)
    monkeypatch.setattr(PP, "live_depth_source", lambda *a, **k: w.depth_of)

    def boom(*a, **k):
        raise RuntimeError("no")

    monkeypatch.setattr(PP, "plan", boom)
    sol, ass, res = FSG._place_pieces(None, "w", SID, w.solution, w.assessment, w.metric, [], {}, "result")
    assert sol is w.solution and ass["pieces"] == w.assessment["pieces"] and res == "result"
    assert ass["place_pieces"]["placed"] == 0 and "RuntimeError" in ass["place_pieces"]["error"]
    summ = FSG.summary(ass, mode_=FSG.MODE_ON, decision=FSG.DECISION_PUBLISHED, seconds=0.0)
    assert summ["pieces_placed"] == 0 and "RuntimeError" in summ["place_pieces_error"]


def test_with_the_switch_on_an_import_failure_leaves_every_piece_cut(monkeypatch):
    import sys

    import tower.world_builder as WB

    monkeypatch.setenv("TOWER_WORLD_PLACE_PIECES", "true")
    monkeypatch.delattr(WB, "place_pieces", raising=False)
    monkeypatch.setitem(sys.modules, "tower.world_builder.place_pieces", None)
    w = World(noise=False)
    sol, ass, res = FSG._place_pieces(None, "w", SID, w.solution, w.assessment, w.metric, [], {}, None)
    assert sol is w.solution and ass["pieces"] == w.assessment["pieces"]
    assert ass["place_pieces"]["placed"] == 0 and "Error" in ass["place_pieces"]["error"]


def _gate_result(w, kind):
    """A gate result whose depth hand-off holds a fit for every keyframe, in the CANDIDATE's units."""
    from tower.world_builder.coherence_publish import GateResult

    records = []
    for i, k in enumerate(w.kids):
        R, c = PP.centre_of(w.poses[k])
        records.append({"ki": i, "kid": k, "ok": True, "a": 0.8 if kind == "depth" else 2.0, "b": 0.1,
                        "held_out_rel": 0.03, "z_sparse_min": 1.0 * K if k in w.piece_kids else 1.0,
                        "z_sparse_max": 4.0 * K if k in w.piece_kids else 4.0})
    return GateResult(solution=w.solution, record={}, components=None,
                      depth={"align": {"kind": kind, "records": records}, "work": None, "dparams": None})


@pytest.mark.parametrize("kind", ["depth", "disparity"])
def test_with_the_switch_on_the_guard_seam_places_records_and_carries_the_depth_fits(monkeypatch, kind):
    from tower.world_builder.dense import depth_from_prediction

    monkeypatch.setenv("TOWER_WORLD_PLACE_PIECES", "true")
    w = World(noise=False)
    monkeypatch.setattr(PP, "live_depth_source", lambda *a, **k: w.depth_of)
    seen = {}

    def plan(solution, assessment, metric_log, name_of, times, depth_of, *a, **k):
        seen["times"], seen["depth"] = times, depth_of
        pl = _placed_record(w)
        return {"id": PP.PLACE_ID, "placed": 1, "pieces": [PP._jsonable(pl.record())], "placements": [pl]}

    monkeypatch.setattr(PP, "plan", plan)
    kfs = [type("K", (), {"keyframe_id": k, "received_at": w.times[k]})() for k in w.kids]
    gate = _gate_result(w, kind)
    sol, ass, res = FSG._place_pieces(None, "w", SID, w.solution, w.assessment, w.metric, kfs,
                                      {k: k for k in w.kids}, gate)
    assert seen["times"] == w.times and seen["depth"] == w.depth_of
    assert ass["place_pieces"]["placed"] == 1 and "placements" not in ass["place_pieces"]
    assert ass["pieces"] == [] and sol is not w.solution
    assert FSG.summary(ass, mode_=FSG.MODE_ON, decision=FSG.DECISION_PUBLISHED, seconds=0.0)["pieces_placed"] == 1
    # The surface REUSES the gate's fits after the hand-off: a placed frame's fit must give its depth in the
    # PUBLISHED units -- the candidate's depth times 1 / ratio -- and nobody else's fit may change.
    pred = np.array([0.5, 1.0, 2.0])
    old = {r["kid"]: r for r in gate.depth["align"]["records"]}
    for r in res.depth["align"]["records"]:
        o = old[r["kid"]]
        z_old = depth_from_prediction(pred, o["a"], o["b"], kind)
        z_new = depth_from_prediction(pred, r["a"], r["b"], kind)
        if r["kid"] in w.piece_kids:
            np.testing.assert_allclose(z_new, z_old / K, rtol=1e-12)
            assert r["z_sparse_min"] == pytest.approx(o["z_sparse_min"] / K)
            assert r["z_sparse_max"] == pytest.approx(o["z_sparse_max"] / K)
        else:
            assert r == o
    assert gate.depth["align"]["records"][N_ROOM]["a"] == old[w.piece_kids[0]]["a"]      # input untouched


def test_a_depth_hand_off_of_an_unknown_fit_kind_leaves_every_piece_cut(monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_PLACE_PIECES", "true")
    w = World(noise=False)
    monkeypatch.setattr(PP, "live_depth_source", lambda *a, **k: w.depth_of)
    monkeypatch.setattr(PP, "plan", lambda *a, **k: {"id": PP.PLACE_ID, "placed": 1, "pieces": [],
                                                     "placements": [_placed_record(w)]})
    gate = _gate_result(w, "depth")
    gate.depth["align"]["kind"] = "mystery"
    sol, ass, res = FSG._place_pieces(None, "w", SID, w.solution, w.assessment, w.metric, [], {}, gate)
    assert sol is w.solution and res is gate and ass["place_pieces"]["placed"] == 0


# ---------------------------------------------------------------------------
# bounds and the room's controls (round 2)
# ---------------------------------------------------------------------------


def test_a_room_too_short_for_every_control_places_nothing():
    w = World(tail_room=6)
    rec = w.plan(dataclasses.replace(PARAMS, control_segments=6, control_frames=10))     # needs 80 frames
    assert rec["room"]["control_segments"] == 0
    assert PP.REFUSE_ROOM in rec["pieces"][0]["refused"] and rec["placed"] == 0


def test_no_held_out_frame_anchors_anything():
    w = World(tail_room=6)
    rec = w.plan()
    ctrl = next(iter(rec["controls"].values()))["controls"]
    held = {k for c in ctrl for k in w.kids[w.kids.index(c["first_keyframe_id"]):
                                            w.kids.index(c["last_keyframe_id"]) + 1]}
    used = {a["room_kid"] for r in ctrl + rec["pieces"] for a in r.get("anchors") or []}
    used |= {r["anchors"][0]["room_kid"] for r in ctrl if r.get("anchors")}
    assert used and not used & held


def test_work_is_bounded_and_a_piece_past_the_bound_is_refused():
    w = World(tail_room=6)
    rec = w.plan(dataclasses.replace(PARAMS, max_measure_frames=10))
    assert rec["pieces"][0]["frames_with_depth"] == 10
    rec = w.plan(dataclasses.replace(PARAMS, max_piece_keyframes=11))
    assert rec["pieces"][0]["refused"] == [PP.REFUSE_TOO_LARGE]
    # Past the work budget nothing is decided, deterministically: the same budget, the same decisions.
    small = dataclasses.replace(PARAMS, max_work=2.0e7)
    a, b = w.plan(small), w.plan(small)
    assert a["pieces"][0]["refused"] == [PP.REFUSE_WORK] and a["placed"] == 0 and a["work"]["fired"] == "work"
    assert a["work"]["used"] == b["work"]["used"] and a["pieces"] == b["pieces"]
    # The wall clock is only a backstop: an operational abort, recorded as such (not `work-bound`).
    rec = w.plan(dataclasses.replace(PARAMS, max_seconds=0.0))
    assert rec["pieces"][0]["refused"] == [PP.REFUSE_ABORT] and rec["work"]["fired"] == "wall-clock backstop"


def test_the_step_runs_only_on_live_depth_evidence(monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_PLACE_PIECES", "true")
    w = World(noise=False)
    called = []
    monkeypatch.setattr(PP, "plan", lambda *a, **k: called.append(1))
    ass = dict(w.assessment, evidence=FSG.EVIDENCE_GATE)
    sol, out, res = FSG._place_pieces(None, "w", SID, w.solution, ass, w.metric, [], {}, None)
    assert sol is w.solution and out["pieces"] == ass["pieces"] and not called
    assert out["place_pieces"]["placed"] == 0 and "live depth cache" in out["place_pieces"]["skipped"]


# ---------------------------------------------------------------------------
# one blow-up as one rigid body (round 3)
# ---------------------------------------------------------------------------


SPLIT = {"n_piece": 24, "split": 12, "piece_step": (0.04, 0.0, 0.025), "tail_room": 6}


def _vouching_envelope(monkeypatch):
    real = PP.envelope
    monkeypatch.setattr(PP, "envelope", lambda controls, upm, params: dict(real(controls, upm, params), **ENV_OK))


def test_two_pieces_of_one_blow_up_are_placed_only_together(monkeypatch):
    # Twelve frames each, 2.7 s apart at their far ends: alone, each has only the room camera on its own side,
    # which is its search's centre -- nothing independent can corroborate it. Together, one similarity, the
    # room cameras on both sides, and the second checks the first.
    _vouching_envelope(monkeypatch)
    w = World(**SPLIT)
    rec = w.plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    a, b = rec["placements"]
    assert a.placed and b.placed, (rec["groups"], a.refused, b.refused)
    assert a.stats["placed_with"] == b.stats["placed_with"] == "group-0"
    np.testing.assert_allclose(a.t, b.t)
    assert a.ratio == b.ratio
    (g,) = rec["groups"]
    assert g["members"] == ["piece-0", "piece-1"] and g["decision"] == "placed"
    assert g["corroborated_by"] == "independent-anchor"
    assert {x["member"] for x in g["anchors"]} == {"piece-0", "piece-1"}
    for pl in (a, b):
        err = max(np.linalg.norm((PP.centre_of(w.poses[k])[1] / pl.ratio + pl.t) / UPM - w.true[k]) for k in pl.kids)
        assert err < 0.2


def test_scale_matched_pieces_the_solve_does_not_connect_are_not_grouped(monkeypatch):
    _vouching_envelope(monkeypatch)
    rec = World(**SPLIT, link_pieces=False).plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    assert rec["groups"] == [] and rec["placed"] == 0
    assert all(PP.REFUSE_CORROBORATION in p["refused"] for p in rec["pieces"])


def test_a_member_whose_own_scale_is_uncertain_is_not_grouped(monkeypatch):
    _vouching_envelope(monkeypatch)
    rec = World(**SPLIT, second_log_jitter=1.0).plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    assert PP.REFUSE_CI in rec["pieces"][1]["refused"]
    assert rec["groups"] == [] and rec["placed"] == 0


def test_a_group_with_only_its_centre_anchor_is_not_placed(monkeypatch):
    # No room camera after the second half: the camera before the first is the search's centre and nothing
    # independent checks it.
    _vouching_envelope(monkeypatch)
    rec = World(**dict(SPLIT, tail_room=0)).plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    (g,) = rec["groups"]
    assert PP.REFUSE_CORROBORATION in g["refused"] and g["decision"] == "refused: members decided alone"
    assert rec["placed"] == 0


def test_the_keyframe_cap_counts_the_whole_group(monkeypatch):
    _vouching_envelope(monkeypatch)
    rec = World(**SPLIT).plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25, max_piece_keyframes=20))
    (g,) = rec["groups"]
    assert g["refused"] == [PP.REFUSE_TOO_LARGE] and rec["placed"] == 0


def test_connected_pieces_at_different_scales_are_not_grouped(monkeypatch):
    _vouching_envelope(monkeypatch)
    w = World(**SPLIT, second_log_offset=0.5)       # the second half x1.65 off the first
    rec = w.plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    assert rec["groups"] == [] and rec["placed"] == 0


def test_a_group_whose_member_anchors_disagree_is_refused_with_its_members(monkeypatch):
    # The room camera after the second half stood 1.5 m from it, 0.4 s later (a bound near 1.2 m): placed by
    # the camera before the first half (0.3 s, the tighter bound and so the search's centre), the group puts
    # the second half where that camera could not have reached it. One similarity cannot satisfy both sides,
    # so the cluster's evidence contradicts itself.
    _vouching_envelope(monkeypatch)
    w = World(**dict(SPLIT, tail_offset_m=(-1.4, 0.0, -0.6), tail_dt=0.4))
    rec = w.plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    (g,) = rec["groups"]
    assert "anchors disagree" in g["decision"]
    assert any(not x["ok"] for x in g["anchor_check"])
    assert all(PP.REFUSE_GROUP_ANCHORS in p["refused"] for p in rec["pieces"]) and rec["placed"] == 0


def test_a_group_that_is_not_one_rigid_body_leaves_the_members_alone():
    # Even with its anchors disagreeing: one similarity cannot fit what is not one body, so the disagreement
    # says nothing against the members.
    pl = PP.Placement(label="group-0", kids=["a"], ratio=1.0, ci=None)
    pl.refused.append(PP.REFUSE_RIGID)
    pl.stats.update(anchor_check=[{"ok": False}], anchors_disagree=True)
    m0 = PP.Placement(label="piece-0", kids=["a"], ratio=1.0, ci=None)
    m0.refused.append(PP.REFUSE_FRAMES)
    placements = [m0]
    g = {"members": [0], "placement": pl}
    PP._group_decision(g, placements, {0: {"kids": ["a"], "ci_log": (0.0, 0.0)}})
    assert placements[0] is m0 and placements[0].refused == [PP.REFUSE_FRAMES]
    assert "not one rigid body" in g["decision"]


# ---------------------------------------------------------------------------
# coverage of the region, matched starts, bounded work (round 3, R2b review)
# ---------------------------------------------------------------------------


def test_basins_are_every_local_maximum_and_a_plateau_is_one():
    step = 1.0
    offs = np.array([[x, 0.0, 0.0] for x in range(7)])
    assert PP.basins(offs, step, [0.1, 0.5, 0.2, 0.1, 0.3, 0.9, 0.4]) == [1, 5]
    assert PP.basins(offs, step, [0.1, 0.2, 0.1, 0.8, 0.9, 1.0, 0.9]) == [1, 5]    # one far below the median
    assert PP.basins(offs, step, [0.2] * 7) == [0]                                 # flat: one plateau
    assert PP.basins(offs, step, [-math.inf, 0.5, 0.5, 0.1, -math.inf, 0.3, -math.inf]) == [1, 5]


def test_every_basin_is_refined_and_compared_on_the_fine_evidence():
    # The parallel-wall alias: the truth and the partition 1.3 m in front of the far wall are two basins of
    # one region; both are refined, and uniqueness compares them on the fine scorer.
    partition = (np.array([2.6, 0.0, 0.0]), np.array([2.7, 2.5, 1.4]))
    solids = FURNITURE + [partition]
    w = World(anchor_offset_m=(1.3, 0.0, 0.2), piece_dt=0.6, piece_forward=(1.0, 0.0, 0.0),
              piece_step=(0.0, 0.0, 0.05), room_solids=solids, piece_solids=solids, noise=False, tail_room=4,
              tail_offset_m=(-1.3, 0.0, 0.0))
    rec = w.plan(dataclasses.replace(PARAMS, max_basins=32))
    piece = rec["pieces"][0]
    assert piece["basins"] >= 2
    assert piece["uniqueness"] < PARAMS.min_peak_drop          # the alias's basin is the winner's rival
    # and a region with more basins than the step can refine is refused, not searched in part
    rec = w.plan(dataclasses.replace(PARAMS, max_basins=1))
    assert rec["pieces"][0]["refused"] == [PP.REFUSE_COVERAGE] and rec["pieces"][0]["translation"] is None


def test_a_region_past_the_cap_is_refused_before_its_lattice_is_built(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("allocated")

    monkeypatch.setattr(PP.np, "meshgrid", boom)
    assert PP.lattice_size(500.0, 0.3, 2000) == 2001
    monkeypatch.undo()
    assert PP.lattice_size(1.0, 0.3, 2000) == len(PP.region_lattice(1.0, 0.3))


def test_the_rooms_evidence_is_built_from_a_bounded_number_of_frames():
    w = World(tail_room=6)
    room, frames = _room(w, dataclasses.replace(PARAMS, max_room_frames=12))
    assert room.frames == 12 < len(frames)


def test_searches_and_the_room_stop_when_the_work_runs_out():
    w = World(tail_room=6)
    room, frames = _room(w)
    with pytest.raises(PP._OutOfWork):
        PP.search(_piece(w), room, np.zeros(3), 0.5 * UPM, PARAMS, budget=PP.Budget(1.0, 3600.0))
    with pytest.raises(PP._OutOfWork):
        PP.build_room(frames, w.depth_of, PP.Pinhole.of(CAM), UPM, PARAMS, budget=PP.Budget(1.0, 3600.0))
    # The room's keys are capped: frames past the cap are left out, and the room says so.
    room = PP.build_room(frames, w.depth_of, PP.Pinhole.of(CAM), UPM, dataclasses.replace(PARAMS, max_room_keys=1))
    assert room.truncated and room.frames == 1


class _Tracked(PP.Budget):
    made: list = []

    def __init__(self, *a):
        super().__init__(*a)
        _Tracked.made.append(self)


def test_nothing_is_certified_past_the_work_bound(monkeypatch):
    # The budget runs out while the controls run: the piece is not certified on them.
    _Tracked.made = []
    monkeypatch.setattr(PP, "Budget", _Tracked)
    real = PP.envelope

    def late(*a, **k):
        _Tracked.made[-1].used = _Tracked.made[-1].units + 1.0
        return dict(real(*a, **k), **ENV_OK)

    monkeypatch.setattr(PP, "envelope", late)
    rec = World(tail_room=6).plan()
    assert rec["pieces"][0]["refused"] == [PP.REFUSE_WORK] and rec["placed"] == 0


def test_nothing_is_placed_when_the_work_runs_out_during_corroboration(monkeypatch):
    _Tracked.made = []
    monkeypatch.setattr(PP, "Budget", _Tracked)
    real_env, real_cor = PP.envelope, PP.corroborate
    monkeypatch.setattr(PP, "envelope", lambda *a, **k: dict(real_env(*a, **k), **ENV_OK))

    def late(placements, params, upm, budget=None):
        budget.used = budget.units + 1.0
        return real_cor(placements, params, upm, budget)

    monkeypatch.setattr(PP, "corroborate", late)
    rec = World(tail_room=6).plan()
    assert rec["placed"] == 0 and PP.REFUSE_WORK in rec["pieces"][0]["refused"]
    # ...where the same run with the work left places it.
    monkeypatch.setattr(PP, "corroborate", real_cor)
    assert World(tail_room=6).plan()["placed"] == 1


def test_a_groups_interval_is_the_union_of_its_members(monkeypatch):
    _vouching_envelope(monkeypatch)
    w = World(**SPLIT)
    rec = w.plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    (g,) = rec["groups"]
    a, b = (w.assessment["pieces"][n] for n in (0, 1))
    lo = min(p["ci_log"][0] + p["core_log_median"] for p in (a, b))
    hi = max(p["ci_log"][1] + p["core_log_median"] for p in (a, b))
    level = float(np.median([w.metric[k] for k in w.piece_kids]))
    assert g["ci_log"] == pytest.approx([lo - level, hi - level], abs=1e-6)


# ---------------------------------------------------------------------------
# round 4: the walk's own motion, per-member evidence, coverage of every basin, R0 at the prior's error
# ---------------------------------------------------------------------------


def test_an_anchors_allowance_is_the_walks_own_motion():
    # Cameras 0.2 s apart walking 0.5 m/s with one 0.8 m lurch: the 0.99 quantile of the moves over 0.4 s
    # (0.2 m; the lurch's two pairs are the top 1 %), never above the walking limit; with too few pairs, the
    # walking limit.
    frames = [PP.Frame(kid=f"r{i}", R=np.eye(3), c=np.array([0.1 * i * UPM + (0.8 * UPM if i == 50 else 0.0), 0, 0]),
                       s=1.0, t=0.2 * i) for i in range(200)]
    allow = PP.walk_motion(frames, WALK, UPM)
    q, m, pairs, source = allow(0.4)
    # (the lurch's few pairs are the top 1-3 % at the gaps up to 0.4 s, so the check lies between the walk's
    # 0.2 m and the lurch's moves; the envelope carries the highest of the shorter gaps)
    assert source == "walk" and pairs > 150 and 0.2 <= q <= 0.4
    # the REGION's containment is the largest move at that gap (the lurch), the CHECK the 0.99 quantile
    assert m == pytest.approx(1.0, abs=0.01) and m < PARAMS.speed_mps * 0.4 + PARAMS.speed_margin_m
    q, m, pairs, source = PP.walk_motion(frames[:20], WALK, UPM)(0.4)
    assert source == "walking-limit" and q == m == pytest.approx(PARAMS.speed_mps * 0.4 + PARAMS.speed_margin_m)


def test_the_walks_motion_never_allows_less_at_a_longer_gap():
    # A walker who darts 0.5 m between two frames 0.2 s apart and otherwise stands still: at 0.4 s and beyond
    # the pairs show almost no motion, but a longer gap can always contain the shorter one's move.
    frames = [PP.Frame(kid=f"r{i}", R=np.eye(3), c=np.array([(0.5 * UPM if i % 2 else 0.0), 0, 0]),
                       s=1.0, t=0.2 * i) for i in range(400)]
    allow = PP.walk_motion(frames, WALK, UPM)
    q2, m2, _n, _s = allow(0.2)
    q4, m4, _n, _s = allow(0.4)
    assert q2 == pytest.approx(0.5, abs=0.01) and q4 >= q2 and m4 >= m2


def test_pieces_that_share_too_few_points_of_their_own_are_not_grouped(monkeypatch):
    _vouching_envelope(monkeypatch)
    rec = World(**SPLIT).plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25, min_shared_points=13))
    assert rec["group_links"][0]["shared_points"] == 12 and rec["groups"] == []


def test_a_member_the_others_never_see_is_not_one_body_with_them(monkeypatch):
    # The solve joins the halves (twelve points of their own), but the second half looks back the other way:
    # no frame of either half sees what the other saw, so R3 cannot test their relative pose. The group is not
    # placed; each half decides alone.
    _vouching_envelope(monkeypatch)
    w = World(**SPLIT, second_looks=[np.array([0.0, 1.5, 0.0]), np.array([0.2, 2.0, 0.3])])
    rec = w.plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    (g,) = rec["groups"]
    assert PP.REFUSE_MEMBER in g["refused"] and g["decision"].startswith("not one rigid body")
    assert min(m["overlap"] for m in g["member_checks"].values()) < PARAMS.min_hit_share
    assert not any(p.get("placed_with") for p in rec["pieces"])


def test_a_misplaced_member_cannot_hide_in_the_pool(monkeypatch):
    # The solve put the second half 0.6 m from where it stood relative to the first; one similarity cannot
    # place both. Its own counts, or its views against the first half's, refuse the group.
    _vouching_envelope(monkeypatch)
    w = World(**SPLIT, second_offset_m=(0.0, 0.0, 0.6))
    rec = w.plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    assert all(g["decision"] != "placed" for g in rec["groups"])
    assert not any(p.get("placed_with") for p in rec["pieces"])


def test_every_member_keeps_its_own_measured_frames():
    # A 20-frame member and a 10-frame member under a 12-frame measurement cap: each keeps min_frames.
    w = World(n_piece=30, split=20, piece_step=(0.03, 0.0, 0.02), tail_room=6)
    rec = w.plan(dataclasses.replace(PARAMS, max_measure_frames=12))
    (g,) = rec["groups"]
    assert all(v >= PARAMS.min_frames for v in g["measured_frames"].values())
    assert PP.REFUSE_FRAMES not in g["refused"]


def test_a_fine_winner_whose_coarse_basin_is_below_the_median_is_still_found(monkeypatch):
    # The coarse landscape is made to peak at a wrong place and to hold the truth's lattice point only as a
    # low local maximum (below the median); the fine evidence then picks the truth, because EVERY local
    # maximum is refined.
    w = World(noise=False, tail_room=6)
    room, frames = _room(w)
    piece = _piece(w)
    k0 = w.piece_kids[0]
    t_true = w.true[k0] * UPM - PP.centre_of(w.poses[k0])[1] / piece.ratio
    radius = 1.2 * UPM
    step = PARAMS.coarse_step_m * UPM
    centre = t_true + np.array([1.0, 0.0, 0.0]) * UPM
    offs = PP.region_lattice(radius, step)
    near = centre + offs[np.argmin(np.linalg.norm(centre + offs - t_true, axis=1))]
    wrong = centre + np.array([1.1, 0.0, 0.0]) * UPM
    real_counts = PP.Scorer.counts

    def counts(self, t, slack=0.0):
        c = real_counts(self, t, slack)
        c["_t"] = np.asarray(t, float)
        return c

    def optimistic(c, params):
        t = c["_t"]
        return -float(np.linalg.norm(t - wrong)) + (0.7 if np.allclose(t, near) else 0.0)

    monkeypatch.setattr(PP.Scorer, "counts", counts)
    monkeypatch.setattr(PP, "_optimistic", optimistic)
    land = [optimistic({"_t": centre + o}, PARAMS) for o in offs]
    assert optimistic({"_t": near}, PARAMS) < np.median(land)              # the premise
    t, _f, _e, _u, _typ, nb, refined = PP.search(piece, room, centre, radius, PARAMS)
    assert nb == 2 and np.linalg.norm(t - t_true) / UPM < 0.1


def test_a_placed_group_moves_every_member_by_the_one_similarity(monkeypatch):
    _vouching_envelope(monkeypatch)
    w = World(**dict(SPLIT, noise=False))
    rec = w.plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    placements = rec["placements"]
    assert all(p.placed for p in placements) and len({tuple(p.t) for p in placements}) == 1
    sol, ass = PP.apply_placements(w.solution, w.assessment, placements)
    for p in placements:
        for k in p.kids:
            np.testing.assert_allclose(PP.centre_of(sol.poses[k])[1],
                                       PP.centre_of(w.poses[k])[1] / p.ratio + p.t, atol=1e-6)
            assert np.linalg.norm(PP.centre_of(sol.poses[k])[1] / UPM - w.true[k]) < 0.2
    depth = PP.rescale_depth_handoff(_gate_result(w, "depth").depth, placements)
    for r in depth["align"]["records"]:
        if r["kid"] in w.piece_kids:
            assert r["a"] == pytest.approx(0.8 * placements[0].ratio) and r["placed_by_depth"] == PP.PLACE_ID
    assert ass["pieces"] == [] and {u["uncertified"] for u in ass["uncertified"]} == {PP.UNCERTIFIED_DEPTH_PLACED}


def test_a_re_gate_sees_the_solvers_geometry_not_the_depth_placement():
    from tower.world_builder.coherence_publish import solver_candidate

    w = World(noise=False)
    ki = {k: i for i, k in enumerate(w.kids)}
    p0, r0 = ki[w.piece_kids[0]], ki[w.kids[0]]
    rx, ro, rxy = w.points
    n = len(rx)
    # A point the piece observes most (three piece cameras, one room camera): it moves with the piece.
    xyz = np.concatenate([rx, [[100.0, 0.0, 0.0]]])
    obs = np.concatenate([ro, [[p0, 0, n], [p0 + 1, 0, n], [p0 + 2, 0, n], [r0, 0, n]]])
    w.solution = types_solution(w.kids, w.poses, xyz=xyz, obs=obs, xy=np.concatenate([rxy, np.full((4, 2), 5.0)]))
    pl = _placed_record(w)
    sol, _ass = PP.apply_placements(w.solution, w.assessment, [pl])
    assert not np.allclose(sol.xyz[n], w.solution.xyz[n])
    cand = solver_candidate(sol)
    for k in w.kids:
        np.testing.assert_allclose(cand.poses[k]["translation"], w.solution.poses[k]["translation"], rtol=1e-9)
        assert "placed_by_depth" not in cand.poses[k]
    np.testing.assert_allclose(cand.xyz, np.asarray(w.solution.xyz), rtol=1e-4, atol=1e-3)


def test_switch_off_the_audit_is_what_it_was():
    # The guard keeps two fields on each isolated piece for the placement; with the step not run, the
    # audited assessment is byte for byte the one without them.
    ass = dict(World().assessment)
    ass["pieces"] = [dict(p) for p in ass["pieces"]]
    old = dict(ass, pieces=[{k: v for k, v in p.items() if k not in ("ci_log", "core_log_median")}
                            for p in ass["pieces"]])
    kw = {"mode_": FSG.MODE_ON, "decision": FSG.DECISION_PUBLISHED, "solution": None, "inputs": {}, "seconds": 1.0}
    off = json.dumps(FSG.audit_document(ass, **kw), sort_keys=True)
    assert off == json.dumps(FSG.audit_document(old, **kw), sort_keys=True) and "ci_log" not in off
    on = FSG.audit_document(dict(ass, place_pieces={"placed": 0}), **kw)
    assert "ci_log" in on["assessment"]["pieces"][0]


def test_an_anchor_disagreeing_with_an_ambiguous_winner_does_not_refuse_the_members():
    gpl = PP.Placement(label="group-0", kids=["a"], ratio=1.0, ci=None, t=np.zeros(3))
    gpl.refused.append(PP.REFUSE_AMBIGUOUS)
    gpl.stats.update(anchors_disagree=True, r3_decided=True, anchor_check=[{"ok": False}])
    m0 = PP.Placement(label="piece-0", kids=["a"], ratio=1.0, ci=None)
    placements = [m0]
    g = {"members": [0], "placement": gpl}
    PP._group_decision(g, placements, {0: {"kids": ["a"], "ci_log": (0.0, 0.0)}})
    assert placements[0].refused == [] and g["decision"] == "refused: members decided alone"
    # ...nor one whose R3 was never decided (it stopped before its controls).
    gpl.refused, gpl.stats["r3_decided"] = [], False
    PP._group_decision(g, placements, {0: {"kids": ["a"], "ci_log": (0.0, 0.0)}})
    assert placements[0].refused == []
    # With R3 and R5 decided and passed, the disagreement refuses the group and its members.
    gpl.stats["r3_decided"] = True
    PP._group_decision(g, placements, {0: {"kids": ["a"], "ci_log": (0.0, 0.0)}})
    assert placements[0].refused == [PP.REFUSE_GROUP_ANCHORS]


def test_the_plan_takes_each_anchors_allowance_from_the_walk():
    w = World(tail_room=6)
    rec = w.plan(dataclasses.replace(WALK, walk_min_pairs=20))
    (a, *_rest) = rec["pieces"][0]["anchors"]
    assert a["allowance_source"] == "walk" and a["allowance_pairs"] >= 20
    assert a["allowance_m"] < PARAMS.speed_mps * a["dt_s"] + PARAMS.speed_margin_m


def test_groups_are_decided_before_single_pieces_are_corroborated(monkeypatch):
    # A member a group refuses must not have vouched for another piece first.
    _vouching_envelope(monkeypatch)
    order = []
    real_g, real_c = PP._group_decision, PP.corroborate
    monkeypatch.setattr(PP, "_group_decision", lambda *a, **k: (order.append("group"), real_g(*a, **k))[1])
    monkeypatch.setattr(PP, "corroborate", lambda *a, **k: (order.append("corroborate"), real_c(*a, **k))[1])
    World(**SPLIT).plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    assert order == ["group", "corroborate"]


def test_an_anchor_that_excludes_only_the_lattices_overhang_corroborates_nothing():
    # The region: a ball of 1 m about the origin, on its 0.3 m lattice (which reaches 0.26 m past it).
    step = 0.3 * UPM
    centre, radius = np.zeros(3), 1.0 * UPM
    # An anchor at the centre whose bound is a little past the radius refuses only overhang points.
    assert PP.outside_share(centre, radius, step, centre, radius + 0.1 * UPM) == 0.0
    assert PP.outside_share(centre, radius, step, centre, radius - 0.2 * UPM) > 0.0


def test_a_re_gate_moves_back_exactly_the_points_a_group_moved():
    # A point seen once by each of two grouped pieces and once by a room camera: placing (each piece its own
    # label, the three-way tie to the room) left it where it was, so the re-gate must too -- though the two
    # pieces share one similarity.
    from tower.world_builder.coherence_publish import solver_candidate

    w = World(**dict(SPLIT, noise=False))
    ki = {k: i for i, k in enumerate(w.kids)}
    a, b, r = ki[w.piece_kids[0]], ki[w.piece_kids[SPLIT["split"]]], ki[w.kids[0]]
    rx, ro, rxy = w.points
    n = len(rx)
    xyz = np.concatenate([rx, [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]])
    obs = np.concatenate([ro, [[a, 0, n], [b, 0, n], [r, 0, n], [a, 1, n + 1], [a + 1, 1, n + 1], [a + 2, 1, n + 1]]])
    w.solution = types_solution(w.kids, w.poses, xyz=xyz, obs=obs, xy=np.concatenate([rxy, np.full((6, 2), 5.0)]))
    t = w.true[w.piece_kids[0]] * UPM - PP.centre_of(w.poses[w.piece_kids[0]])[1] / K
    split = SPLIT["split"]
    pls = [PP.Placement(label="piece-0", kids=list(w.piece_kids[:split]), ratio=K, ci=None, placed=True, t=t),
           PP.Placement(label="piece-1", kids=list(w.piece_kids[split:]), ratio=K, ci=None, placed=True, t=t)]
    sol, _ass = PP.apply_placements(w.solution, w.assessment, pls)
    np.testing.assert_allclose(sol.xyz[n], [1.0, 1.0, 1.0])                 # the tie stayed with the room
    assert not np.allclose(sol.xyz[n + 1], [5.0, 5.0, 5.0])                  # piece-0's own point moved
    cand = solver_candidate(sol)
    np.testing.assert_allclose(cand.xyz[n], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(cand.xyz[n + 1], [5.0, 5.0, 5.0], rtol=1e-4)
    assert "placed_by_depth" not in (cand.solve or {})


def test_a_member_aligned_only_at_a_doorway_is_refused(monkeypatch):
    # The second half's first three frames stand where the solve put them (a doorway the first half also saw);
    # the rest of it the solve put 0.6 m off. Its aligned minority cannot carry it.
    _vouching_envelope(monkeypatch)
    w = World(**SPLIT, second_offset_m=(0.0, 0.0, 0.6), second_offset_from=3)
    rec = w.plan(dataclasses.replace(PARAMS, max_control_miss_m=0.25))
    assert all(g["decision"] != "placed" for g in rec["groups"])
    assert not any(p.get("placed_with") for p in rec["pieces"])


def test_a_stalled_machine_aborts_and_says_so():
    rec = World(tail_room=6).plan(dataclasses.replace(PARAMS, max_seconds=0.0))
    assert rec["pieces"][0]["refused"] == [PP.REFUSE_ABORT] and rec["placed"] == 0
    assert rec["aborted"]["by"] == "wall-clock backstop"
