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
import math

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
                        control_segments=2, control_frames=10, min_good_controls=2)
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
                 tail_dt=None):
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
            if piece_forward is not None:
                R = _look_at(c, c + np.asarray(piece_forward, float))
            else:
                R = _look_at(c, looks[j % len(looks)] + np.array([0.0, 0.1 * (j % 3), 0.0]))
            kid = f"{SID}:p{j:02d}"
            self.piece_kids.append(kid)
            self.true[kid] = c
            blown = (c * UPM - hinge) * K + hinge
            if piece_jitter_m:
                blown = blown + rng.normal(scale=piece_jitter_m * UPM * K, size=3)
            self.poses[kid] = _pose(R, blown)
            m, self.depth[kid] = _noisy(_render(R, c, solids=piece_solids if piece_solids is not None else FURNITURE),
                                        rng, noise)
            self.times[kid] = 0.2 * (N_ROOM - 1) + piece_dt + 0.2 * j
            self.metric[kid] = math.log(UPM * K) + m + (piece_log_jitter * rng.standard_normal()
                                                         if piece_log_jitter else 0.0)
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
        self.points = (np.asarray(xyz), np.asarray(obs), np.asarray(xy))
        self.solution = types_solution(self.kids, self.poses, xyz=xyz, obs=obs, xy=xy)
        self.assessment = {
            "reference": {"log": math.log(UPM)}, "room_keyframes": list(self.room_kids), "uncertified": [],
            "uncertified_keyframes": 0, "scale_status": "certified", "why": None, "evidence": FSG.EVIDENCE_LIVE,
            "pieces": [{"keyframe_ids": list(self.piece_kids), "ci": ci or [K * 0.97, K * 1.03],
                        "reason": FSG.REASON_SCALE_OUTLIER, "first_index": N_ROOM}]}

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
    for c in ctrl:
        # anchored at least as far back as the piece's centre anchor, searched over the piece's radius
        assert c["anchor_dt_s"] >= piece["region"]["dt_s"] - 1e-9
        assert c["region"]["radius_m"] == pytest.approx(piece["region"]["radius_m"], abs=0.01)
        assert c.get("miss_m") is not None
        # a control carries its own measured scale, as a piece does
        kids = w.kids[w.kids.index(c["first_keyframe_id"]):w.kids.index(c["last_keyframe_id"]) + 1]
        assert c["ratio"] == pytest.approx(math.exp(float(np.median([w.metric[k] for k in kids])) - math.log(UPM)))
        assert c["ratio"] != pytest.approx(1.0, abs=1e-6)


def _control(miss_m, **stats):
    c = _measured(**stats)
    c.t = np.array([miss_m * UPM, 0.0, 0.0])
    c.stats["true_translation"] = [0.0, 0.0, 0.0]
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


def test_r1_reads_the_recomputed_interval_not_the_guards_display():
    w = World(ci=[K / 1.4, K * 1.1], tail_room=6)    # a wide DISPLAY interval; the frames' ratios agree
    rec = w.plan()
    assert PP.REFUSE_CI not in rec["pieces"][0]["refused"]
    w = World(ci=[K * 0.999, K * 1.001], piece_log_jitter=0.6)    # a tight display, scattered ratios
    assert w.plan()["pieces"][0]["refused"] == [PP.REFUSE_CI]


def test_the_interval_is_the_guards_bootstrap_unrounded():
    gp = FSG.GuardParams()
    rng = np.random.default_rng(5)
    core = rng.normal(0.0, 0.05, size=300)
    piece = math.log(0.0009) + rng.normal(0.0, 0.15, size=15)
    lo, hi = PP.scale_interval_log(piece, core, 7, gp)
    shown = FSG.bootstrap_interval(piece, core, gp, salt=7)            # the guard's, rounded to 4 decimals
    assert shown[0] == pytest.approx(round(math.exp(lo), 4)) and shown[1] == pytest.approx(round(math.exp(hi), 4))
    # At collapsed ratios the display rounding flips R1 (x1.25) for some pieces; the unrounded one decides.
    flips = 0
    for k in range(40):
        piece = math.log(0.0004 + 0.00005 * k) + rng.normal(0.0, 0.12, size=15)
        lo, hi = PP.scale_interval_log(piece, core, k, gp)
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
    assert other and other[0]["independent"] and other[0]["ok"] and other[0]["bound_m"] < other[0]["worst_m"]
    PP.certify(pl, ENV_OK, PARAMS)
    PP.corroborate([pl], PARAMS, UPM)
    assert pl.placed and pl.stats["corroborated_by"] == "independent-anchor"


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
    rec = w.plan(dataclasses.replace(PARAMS, max_seconds=0.0))
    assert rec["pieces"][0]["refused"] == [PP.REFUSE_TIME] and rec["placed"] == 0


def test_the_step_runs_only_on_live_depth_evidence(monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_PLACE_PIECES", "true")
    w = World(noise=False)
    called = []
    monkeypatch.setattr(PP, "plan", lambda *a, **k: called.append(1))
    ass = dict(w.assessment, evidence=FSG.EVIDENCE_GATE)
    sol, out, res = FSG._place_pieces(None, "w", SID, w.solution, ass, w.metric, [], {}, None)
    assert sol is w.solution and out["pieces"] == ass["pieces"] and not called
    assert out["place_pieces"]["placed"] == 0 and "live depth cache" in out["place_pieces"]["skipped"]
