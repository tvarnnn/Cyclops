"""Depth fits borrowed from the room (`neighbour_fit.py`, `_Frames.adopt_neighbour_fits`).

A posed frame whose own sparse fit is missing or refused is fitted to the depth its fused
neighbours measured, and fused only when that fit passes the gate every fused frame passes.
The scene is a synthetic box room with exact depth, so "the right fit" and "the wrong pose"
are known by construction. `TestTheRoomsGauge` adds what a box room alone never exercises: a
stretch of the solve drawn in its OWN gauge (its poses and its own fits scaled together, as on
the 10.1 replay cc3), which is internally consistent and therefore invisible to the held-out
error. `TestCertifiedDonors` is what stops it: only frames the final scale guard kept lend.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tower.world_builder import neighbour_fit as NF
from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP

ROOM = 3.0
W, H = 96, 128            # portrait, like the glasses
K = np.array([[110.0, 0, W / 2], [0, 110.0, H / 2], [0, 0, 1.0]])
A0, B0 = 0.8, 0.3         # the network's affine: pred = A0 * z + B0 ("depth" kind)


def _look_from(eye, target, up=(0.0, -1.0, 0.0)):
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1).T
    return R, -R @ eye


def _depth(R, t, slab=None):
    """Exact depth of the inside of the box room (and an optional box `slab`)."""
    C = -R.T @ t
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    d = dc @ R
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    best = np.full((H, W), np.inf)
    boxes = [(np.full(3, -ROOM), np.full(3, ROOM))]
    if slab is not None:
        boxes.append((np.asarray(slab[0], float), np.asarray(slab[1], float)))
    for lo, hi in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t0, t1 = (lo - C) / d, (hi - C) / d
        tmin = np.minimum(t0, t1).max(-1)
        tmax = np.maximum(t0, t1).min(-1)
        hit = tmax >= np.maximum(tmin, 1e-6)
        cand = np.where(hit, np.where(tmin > 1e-6, tmin, tmax), np.inf)
        best = np.minimum(best, cand)
    return (best * dc[..., 2] / np.linalg.norm(dc, axis=-1)).astype(np.float64)


def _view(eye, target, slab=None):
    R, t = _look_from(eye, target)
    return R, t, _depth(R, t, slab)


def _neighbour(ki, eye, target, slab=None, z_cap=None):
    R, t, z = _view(eye, target, slab)
    return NF.frame_points(ki, z, np.isfinite(z), R, t, K, z_cap=z_cap)


def _band(z):
    return np.maximum(0.05, 0.06 * z)


CORNER = (ROOM, 0.0, ROOM)      # a corner: two walls at different depths in one view
EYES = [(-0.6, 0.0, -0.5), (-0.2, 0.1, -0.7), (0.3, -0.1, -0.6), (0.6, 0.0, -0.4)]


def _anchors_for(R, t, neighbours, **kw):
    return NF.borrowed_anchors(K, (H, W), R, t, neighbours, band=_band, **kw)


def _fit(pred, R, t, neighbours, **kw):
    u, v, z, src = _anchors_for(R, t, neighbours)
    return NF.fit_from_neighbours(pred, "depth", u, v, z, src, gate_rel=0.08,
                                  min_anchors=20, min_frames=2, **kw)


class TestTheBorrowedFit:
    def test_a_frame_with_no_anchors_of_its_own_recovers_its_affine(self):
        nbs = [_neighbour(i, e, CORNER) for i, e in enumerate(EYES)]
        R, t, z = _view((0.0, 0.0, -0.55), CORNER)
        fit = _fit(A0 * z + B0, R, t, nbs)
        assert fit["ok"] and fit["code"] == "adopted"
        # An anchor lands on the pixel its point projects into, whose centre is up to
        # half a pixel away: that, not the fit, is the residual error.
        assert fit["a"] == pytest.approx(A0, rel=5e-3)
        assert fit["b"] == pytest.approx(B0, abs=5e-3)
        assert fit["held_out_rel"] < 2e-3
        assert fit["anchor_frames"] == len(nbs) and fit["anchors"] > 1000

    @pytest.mark.parametrize("wrong", ["shifted", "turned"])
    def test_a_wrong_pose_cannot_follow_the_room_and_is_refused(self, wrong):
        """The prediction is made from where the camera really was; the pose the
        anchors are projected through is not that. The room's points land on the
        wrong pixels and no affine of the prediction follows them."""
        nbs = [_neighbour(i, e, CORNER) for i, e in enumerate(EYES)]
        eye = np.array([0.0, 0.0, -0.55])
        R, t, z = _view(eye, CORNER)
        if wrong == "shifted":
            Rw, tw = _look_from(eye + np.array([1.2, 0.0, 0.9]), np.asarray(CORNER) + [1.2, 0, 0.9])
        else:
            Rw, tw = _look_from(eye, (ROOM, 0.0, 0.0))
        fit = _fit(A0 * z + B0, Rw, tw, nbs)
        assert not fit["ok"]
        assert fit["code"] in ("gate", "inverted", "unscorable")

    def test_one_fused_frame_is_not_evidence(self):
        nbs = [_neighbour(0, EYES[0], CORNER)]
        R, t, z = _view((0.0, 0.0, -0.55), CORNER)
        fit = _fit(A0 * z + B0, R, t, nbs)
        assert not fit["ok"] and fit["code"] == "frames"

    def test_no_overlap_means_no_fit(self):
        nbs = [_neighbour(i, e, CORNER) for i, e in enumerate(EYES)]
        R, t, z = _view((0.0, 0.0, 0.0), (-ROOM, 0.0, -ROOM))    # the opposite corner
        fit = _fit(A0 * z + B0, R, t, nbs)
        assert not fit["ok"] and fit["code"] == "anchors"

    def test_redaction_fill_lends_no_anchor(self):
        nbs = [_neighbour(i, e, CORNER) for i, e in enumerate(EYES)]
        R, t, _z = _view((0.0, 0.0, -0.55), CORNER)
        blocked = np.zeros((H, W), bool)
        blocked[:, : W // 2] = True
        u, _v, _zz, _s = _anchors_for(R, t, nbs, blocked=blocked)
        assert len(u) and (u >= W // 2).all()


def _in_gauge(eye, s):
    """Where a solve that drew this stretch `s` times the room's scale about the origin puts a
    camera standing at `eye`: the same rotation, the centre scaled."""
    return np.asarray(eye, float) * s


def _stretch_neighbour(ki, eye, target, s):
    """A fused frame of a stretch drawn in its own gauge: posed at `_in_gauge(eye, s)`, and its
    own sparse fit -- to points triangulated in that gauge -- turns its prediction into `s` x the
    true depth. Internally consistent: its points are the room scaled by `s`."""
    R, _t, z = _view(eye, target)
    t = -R @ _in_gauge(eye, s)
    return NF.frame_points(ki, s * z, np.isfinite(z), R, t, K, z_cap=None)


STRETCH = [(-0.3, 0.0, -0.6), (0.0, 0.1, -0.7), (0.3, -0.1, -0.6)]


class TestAStretchInItsOwnGauge:
    """The cc3 mechanism: a stretch the solve drew in its own gauge lends anchors that agree with
    each other, so every rule of the fit passes. What stops it is not the fit (see
    `TestCertifiedDonors`)."""

    @pytest.mark.parametrize("s", [4.16, 0.07])
    def test_its_own_frames_vouch_for_each_other(self, s):
        """x4.16 is cc3's excursion 347-358; x0.07 its collapsed stretches."""
        nbs = [_stretch_neighbour(i, e, CORNER, s) for i, e in enumerate(STRETCH)]
        eye = (0.0, 0.0, -0.55)
        R, _t, z = _view(eye, CORNER)
        fit = _fit(A0 * z + B0, R, -R @ _in_gauge(eye, s), nbs)
        assert fit["ok"] and fit["anchor_frames"] == 3 and fit["held_out_rel"] < 0.01
        assert fit["a"] == pytest.approx(A0 / s, rel=0.02)


class TestTheAnchorsAreMeasuredSurface:
    def test_the_nearest_surface_in_a_cell_wins(self):
        """One neighbour saw a slab, another saw the wall behind it. In the candidate,
        which sees the slab, the wall points behind it are not anchors."""
        slab = ((1.0, -0.4, 1.4), (1.6, 0.4, 1.6))
        target = (1.3, 0.0, ROOM)
        R, t, z = _view((1.3, 0.0, -0.4), target, slab=slab)
        near = _neighbour(0, (1.25, 0.0, -0.3), target, slab=slab)
        # A frame whose view of the wall misses the slab: it stands beside it.
        R2, t2 = _look_from((2.6, 0.0, 1.2), target)
        z2 = _depth(R2, t2)
        far = NF.frame_points(1, z2, np.isfinite(z2), R2, t2, K, z_cap=None)
        u, v, za, src = _anchors_for(R, t, [near, far])
        truth = z[v, u]
        assert len(za) > 100
        # Anchors in front of or on the candidate's own surface, never behind it.
        assert np.all(za <= truth + _band(truth) + 1e-6)

    def test_an_anchor_is_never_itself_an_extrapolation(self):
        R, t, z = _view(EYES[0], CORNER)
        cap = float(np.nanpercentile(z, 50))
        p = NF.frame_points(0, z, np.isfinite(z), R, t, K, z_cap=cap)
        zc = (p.X @ R.T + t)[:, 2]
        assert len(zc) and zc.max() <= cap + 1e-9

    def test_neighbours_facing_away_are_not_consulted(self):
        centres = np.array([[0.1, 0, 0], [0.2, 0, 0], [5.0, 0, 0]])
        axes = np.array([[0, 0, -1.0], [0, 0, 1.0], [0, 0, 1.0]])
        assert NF.pick_neighbours(np.zeros(3), np.array([0, 0, 1.0]), centres, axes, 10) == [1, 2]
        assert NF.pick_neighbours(np.zeros(3), np.array([0, 0, 1.0]), centres, axes, 1) == [1]


# ---------------------------------------------------------------------------
# the surface stage's frames
# ---------------------------------------------------------------------------

CAMERA = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "width": W, "height": H}


def _pose(R, t, component=0):
    return {"rotation": R.ravel().tolist(), "translation": t.tolist(), "component": component}


def _world(tmp_path, specs):
    """`specs`: (ki, eye, target, record kind, component[, gauge]). Every frame's prediction is
    the exact affine of its true depth; 'ok' frames carry that fit. A frame of a stretch the
    solve drew in its own `gauge` (default 1: the room's) is posed at `_in_gauge(eye, gauge)`,
    and an 'ok' one's own fit turns its prediction into `gauge` x the true depth."""
    import cv2

    work = tmp_path / "work"
    (work / "depth").mkdir(parents=True)
    (work / "undist").mkdir(parents=True)
    poses, records, kids = {}, [], []
    for ki, eye, target, kind, comp, *gauge in specs:
        g = float(gauge[0]) if gauge else 1.0
        R, _t, z = _view(eye, target)
        t = -R @ _in_gauge(eye, g)
        kid = f"s:{ki:08d}"
        kids.append(kid)
        poses[kid] = _pose(R, t, comp)
        pred = (A0 * z + B0).astype(np.float16)
        np.save(work / "depth" / f"{ki:05d}_pred.npy", pred)
        np.save(work / "depth" / f"{ki:05d}_fill.npy", np.zeros((H, W), bool))
        cv2.imwrite(str(work / "undist" / f"{ki:05d}.jpg"), np.full((H, W, 3), 128, np.uint8))
        if kind == "ok":
            np.save(work / "depth" / f"{ki:05d}.npy", pred)
            records.append({"ki": ki, "kid": kid, "ok": True, "a": A0 / g, "b": B0,
                            "held_out_rel": 0.01, "z_sparse_max": g * float(np.nanmax(z))})
        elif kind == "refused":
            np.save(work / "depth" / f"{ki:05d}.npy", pred)
            records.append({"ki": ki, "kid": kid, "ok": True, "a": A0 * 1.4, "b": B0,
                            "held_out_rel": 0.2, "z_sparse_max": 2.0})
        else:
            records.append({"ki": ki, "kid": kid, "ok": False,
                            "why": "only 3 sparse observations"})
    solution = SimpleNamespace(poses=poses, camera=CAMERA, keyframe_ids=kids)
    align = {"kind": "depth", "camera": CAMERA, "records": records}
    params = S.SurfaceParams()
    frames = SP._Frames(align, work, solution, params)
    return frames, align, solution, params


def _adopt(frames, align, solution, params, certified=None):
    """`certified`: the keyframe indices the final scale guard certified; None certifies every
    frame fused with its own fit."""
    if certified is None:
        certified = {int(it[0]) for it in frames.items}
    return frames.adopt_neighbour_fits(align, solution, params, median_depth=3.0,
                                       voxel=params.voxel_frac * 3.0, certified=certified)


class TestAdoption:
    def test_unfitted_and_refused_frames_of_the_room_are_adopted_with_their_true_affine(
            self, tmp_path):
        specs = [(i, e, CORNER, "ok", 0) for i, e in enumerate(EYES)]
        specs += [(10, (0.0, 0.0, -0.55), CORNER, "no-fit", 0),
                  (11, (0.1, 0.05, -0.45), CORNER, "refused", 0)]
        frames, align, solution, params = _world(tmp_path, specs)
        assert sorted(it[0] for it in frames.items) == [0, 1, 2, 3]
        rec = _adopt(frames, align, solution, params)
        assert rec["candidates"] == 2 and rec["adopted"] == 2
        assert rec["by_prior"] == {"no-fit": {"candidates": 1, "adopted": 1},
                                   "refused": {"candidates": 1, "adopted": 1}}
        items = {it[0]: it for it in frames.items}
        for ki in (10, 11):
            _ki, a, b, _R, _t, ho, zmax = items[ki]
            # float16 storage of the prediction is the only error.
            assert a == pytest.approx(A0, rel=5e-3) and b == pytest.approx(B0, abs=5e-3)
            assert ho < 0.01 and zmax > 0
        # The adopted frame's depth is read from the raw prediction.
        pred, img, fill = frames.load(10)
        assert pred.shape == (H, W) and img is not None and fill is not None
        # ... and the manifest hands the appearance stage records shaped like the depth
        # stage's own `ok` records, with the fit the surface fused.
        recs = SP.adopted_fits({"detail": {"neighbour_fit": rec}})
        assert set(recs) == {10, 11}
        for ki in (10, 11):
            r = recs[ki]
            assert r["ok"] and r["fit"] == "neighbour" and r["kid"] == f"s:{ki:08d}"
            assert r["a"] == pytest.approx(items[ki][1]) and r["b"] == pytest.approx(items[ki][2])
        assert SP.adopted_fits({"detail": {}}) == {} and SP.adopted_fits(None) == {}

    def test_another_component_is_never_a_candidate(self, tmp_path):
        specs = [(i, e, CORNER, "ok", 0) for i, e in enumerate(EYES)]
        specs += [(10, (0.0, 0.0, -0.55), CORNER, "no-fit", 3)]
        frames, align, solution, params = _world(tmp_path, specs)
        rec = _adopt(frames, align, solution, params)
        assert rec["candidates"] == 0 and len(frames) == 4

    def test_a_pose_the_gate_dropped_is_not_readmitted(self, tmp_path):
        specs = [(i, e, CORNER, "ok", 0) for i, e in enumerate(EYES)]
        frames, align, solution, params = _world(tmp_path, specs)
        frames.pose_gated.add(2)
        frames.items = [it for it in frames.items if it[0] != 2]
        rec = _adopt(frames, align, solution, params)
        assert rec["candidates"] == 0 and 2 not in {it[0] for it in frames.items}

    def test_a_frame_whose_pose_disagrees_with_the_room_stays_out(self, tmp_path):
        specs = [(i, e, CORNER, "ok", 0) for i, e in enumerate(EYES)]
        specs += [(10, (0.0, 0.0, -0.55), CORNER, "no-fit", 0)]
        frames, align, solution, params = _world(tmp_path, specs)
        # The solve placed it looking at another wall than the one it photographed.
        R, t = _look_from((0.0, 0.0, -0.55), (ROOM, 0.0, 0.0))
        solution.poses["s:00000010"] = _pose(R, t)
        rec = _adopt(frames, align, solution, params)
        assert rec["adopted"] == 0 and sum(rec["refused"].values()) == 1
        assert 10 not in {it[0] for it in frames.items}


class TestCertifiedDonors:
    """CERTIFIED DONORS (`neighbour_fit.py`): the cc3 layout in miniature -- the room's frames,
    and beside them a stretch the solve drew x4.16 the room's scale whose own fits agree with each
    other; one unfitted frame in each. The final scale guard kept the room and left the stretch
    attached, uncertified."""

    OPP = (-ROOM, 0.0, -ROOM)
    ROOM_KI = {0, 1, 2, 3, 4, 21}

    def _specs(self, gauge=4.16):
        room = [(i, (-e[0], e[1], -e[2]), self.OPP, "ok", 0) for i, e in enumerate(EYES)]
        room += [(4, (0.1, 0.0, 0.45), self.OPP, "ok", 0)]
        stretch = [(10 + i, e, CORNER, "ok", 0, gauge) for i, e in enumerate(STRETCH)]
        return room + stretch + [(20, (0.0, 0.0, -0.55), CORNER, "no-fit", 0, gauge),
                                 (21, (0.0, 0.05, 0.55), self.OPP, "no-fit", 0)]

    def test_an_uncertified_stretch_cannot_vouch_for_its_own_frames(self, tmp_path):
        frames, align, solution, params = _world(tmp_path, self._specs())
        assert sorted(it[0] for it in frames.items) == [0, 1, 2, 3, 4, 10, 11, 12]
        rec = _adopt(frames, align, solution, params, certified=self.ROOM_KI)
        assert rec["candidates"] == 2 and rec["adopted"] == 1
        assert [f["ki"] for f in rec["adopted_frames"]] == [21]
        assert sum(rec["refused"].values()) == 1 and 20 not in {it[0] for it in frames.items}
        assert rec["donors"] == {"rule": "final-scale-guard kept plateaus", "fused": 8,
                                 "certified": 5}
        # The stretch's own-fit frames are still fused: own fits are not this rule's.
        assert {10, 11, 12} <= {it[0] for it in frames.items}

    def test_what_neighbour_fit_1_did_with_the_same_stretch(self, tmp_path):
        """Every fused frame lending, as before: the stretch's frame is adopted on its own
        stretch's anchors, at the stretch's scale."""
        frames, align, solution, params = _world(tmp_path, self._specs())
        rec = _adopt(frames, align, solution, params)
        assert rec["adopted"] == 2
        a20 = next(f for f in rec["adopted_frames"] if f["ki"] == 20)
        assert a20["a"] == pytest.approx(A0 / 4.16, rel=0.05)

    def test_a_certified_stretch_lends(self, tmp_path):
        frames, align, solution, params = _world(tmp_path, self._specs(gauge=1.0))
        rec = _adopt(frames, align, solution, params, certified=self.ROOM_KI | {10, 11, 12})
        assert rec["adopted"] == 2 and rec["refused"] == {}

    @pytest.mark.parametrize("certified", [set(), None])
    def test_without_a_certified_donor_nothing_is_borrowed(self, tmp_path, certified):
        specs = [(i, e, CORNER, "ok", 0) for i, e in enumerate(EYES)]
        specs += [(10, (0.0, 0.0, -0.55), CORNER, "no-fit", 0)]
        frames, align, solution, params = _world(tmp_path, specs)
        rec = frames.adopt_neighbour_fits(align, solution, params, median_depth=3.0,
                                          voxel=params.voxel_frac * 3.0, certified=certified)
        assert rec["adopted"] == 0 and rec["refused"] == {"no-certified-donor": 1}
        assert sorted(it[0] for it in frames.items) == [0, 1, 2, 3]

    def test_the_certified_keyframes_are_the_guards_judged_kept_cameras(self):
        kids = [f"s:{i}" for i in range(400)]
        audit = {"assessment": {
            "room_keyframes": [k for i, k in enumerate(kids) if i not in (5, 83)],
            "plateaus": [
                {"first_index": 0, "last_index": 74, "decision": "kept", "class": "in"},
                {"first_index": 75, "last_index": 81, "decision": "attached", "class": "high"},
                {"first_index": 82, "last_index": 120, "decision": "kept", "class": "in"},
                {"first_index": 347, "last_index": 358, "decision": "attached", "class": "high",
                 "kind": "excursion"}],
            "pieces": [{"first_index": 100, "last_index": 110, "decision": "isolated",
                        "keyframe_ids": kids[100:111]}]}}
        want = (set(range(0, 75)) | set(range(82, 121))) - {5, 83} - set(range(100, 111))
        assert NF.certified_keyframes(audit, kids) == want
        assert NF.certified_keyframes(None, kids) == set()
        assert NF.certified_keyframes({"assessment": None, "error": "x"}, kids) == set()
        assert NF.certified_keyframes(audit, []) == set()

    def test_the_surface_stage_reads_the_guards_audit(self, tmp_path):
        from tower.world_builder import final_scale_guard as FSG

        root = tmp_path / "worlds" / "w"
        store = SimpleNamespace(world_dir=lambda wid: tmp_path / "worlds" / wid)
        kids = ["a", "b", "c", "d", "e", "f"]
        assert SP.guard_certified(store, "w", "s", kids) == set()
        (root / "solve" / "s").mkdir(parents=True)
        (root / "solve" / "s" / FSG.AUDIT_FILENAME).write_text(json.dumps({"assessment": {
            "room_keyframes": ["c", "d", "f"],
            "plateaus": [{"first_index": 2, "last_index": 4, "decision": "kept"}]}}))
        assert SP.guard_certified(store, "w", "s", kids) == {2, 3}
        (root / "solve" / "s" / FSG.AUDIT_FILENAME).write_text("{not json")
        assert SP.guard_certified(store, "w", "s", kids) == set()


class TestTheAnchoredFootprint:
    """THE ANCHORED FOOTPRINT (`neighbour_fit.py`): an adopted frame's prediction exists only
    where its fit was scored."""

    def test_the_cells_holding_an_anchor_dilated_by_one_cell(self):
        fp = NF.anchored_footprint(np.array([9]), np.array([17]), (32, 24))
        want = np.zeros((32, 24), bool)
        want[12:24, 4:16] = True        # cell (4, 2) and its eight neighbours, 4 px each
        assert (fp == want).all()
        corner = NF.anchored_footprint(np.array([0]), np.array([0]), (10, 10))
        want = np.zeros((10, 10), bool)
        want[:8, :8] = True
        assert (corner == want).all()
        assert not NF.anchored_footprint(np.zeros(0, int), np.zeros(0, int), (8, 8)).any()

    def test_a_cell_whose_anchor_the_fit_left_behind_is_not_fused(self):
        """The dilation never reaches into a block holding an anchor the fit disagrees with,
        and that block's own anchors do not seed it."""
        u = np.array([9, 13, 5])
        v = np.array([17, 17, 17])
        fp = NF.anchored_footprint(u, v, (32, 24), agree=np.array([True, False, True]))
        want = np.zeros((32, 24), bool)
        want[12:24, 0:16] = True        # cells (4, 1) and (4, 2) agree: columns 0..3 dilated
        want[16:20, 12:16] = False      # cell (4, 3) holds the disagreeing anchor
        assert (fp == want).all()

    def test_a_pose_too_low_on_a_plain_wall_keeps_only_what_agrees(self):
        """Review R1's probe in miniature: the room's wall points land, through a pose 0.6 m too
        low, on pixels that truly see the ceiling. The fit follows the wall; those pixels do not
        agree with their anchors and are not fused."""
        ceiling = 1.2
        wall = (0.0, 0.0, ROOM)

        def depth_with_ceiling(R, t):
            z = _depth(R, t)
            C = -R.T @ t
            uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
            dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                           np.ones_like(uu)], -1)
            d = dc @ R
            with np.errstate(divide="ignore", invalid="ignore"):
                tc = (-ceiling - C[1]) / d[..., 1]          # y is down: the ceiling at y = -1.2
            zc = np.where(tc > 0, tc * dc[..., 2] / np.linalg.norm(dc, axis=-1) *
                          np.linalg.norm(d, axis=-1), np.inf)
            return np.minimum(z, zc), zc < z

        nbs = []
        for i, e in enumerate([(-0.4, 0.0, 0.2), (0.0, 0.1, 0.0), (0.4, 0.0, 0.1),
                               (0.2, -0.1, 0.3)]):
            R, t = _look_from(e, (e[0], 0.6, ROOM))
            z, on_ceiling = depth_with_ceiling(R, t)
            nbs.append(NF.frame_points(i, z, np.isfinite(z) & ~on_ceiling, R, t, K, z_cap=None))
        eye = np.array([0.0, 0.0, 0.5])
        R, t = _look_from(eye, (0.0, -0.9, ROOM))
        z, on_ceiling = depth_with_ceiling(R, t)
        Rw, tw = _look_from(eye + [0.0, 0.6, 0.0], (0.0, -0.3, ROOM))
        u, v, za, src = _anchors_for(Rw, tw, nbs)
        fit = NF.fit_from_neighbours(z, "depth", u, v, za, src, gate_rel=0.08, min_anchors=20,
                                     min_frames=2)
        assert fit["ok"]                    # the held-out error does not see it
        zf = (z[v, u] - fit["b"]) / fit["a"]
        agree = np.abs(zf - za) <= _band(za)
        assert (~agree).sum() > 50
        loose = NF.anchored_footprint(u, v, (H, W))
        tight = NF.anchored_footprint(u, v, (H, W), agree=agree)
        assert (loose & on_ceiling).sum() > 4 * (tight & on_ceiling).sum()

    def test_an_adopted_frame_is_fused_only_where_its_neighbours_measured(self, tmp_path):
        """The donors saw one wall of the corner; the candidate sees both. Its other wall is
        geometry no anchor checked -- a pose error parallel to a plain wall would move it
        unseen -- so it does not exist for the surface."""
        wall = (ROOM, 0.0, 1.0)
        donors = [(i, (1.4, dy, 1.0 + dz), wall, "ok", 0)
                  for i, (dy, dz) in enumerate([(0.0, -0.3), (0.2, 0.0), (-0.2, 0.2), (0.0, 0.4)])]
        frames, align, solution, params = _world(
            tmp_path, donors + [(10, (0.0, 0.0, -0.55), CORNER, "no-fit", 0)])
        rec = _adopt(frames, align, solution, params)
        assert rec["adopted"] == 1
        fp = frames.footprint[10]
        share = rec["adopted_frames"][0]["footprint_share"]
        assert share == pytest.approx(float(fp.mean()), abs=1e-4) and 0.05 < share < 0.9
        pred, _img, _fill = frames.load(10, image=False)
        assert np.isnan(pred[~fp]).all() and np.isfinite(pred[fp]).all()
        # Where it exists it is the room: the borrowed affine recovers the true depth.
        R, _t, z = _view((0.0, 0.0, -0.55), CORNER)
        _ki, a, b, *_ = next(it for it in frames.items if it[0] == 10)
        zf = SP._depth_from_prediction(pred, a, b, "depth")
        assert np.nanmedian(np.abs(zf[fp] - z[fp]) / z[fp]) < 0.01
        # A frame fused with its own fit is untouched.
        own, _i, _f = frames.load(0, image=False)
        assert np.isfinite(own).all()


class TestTheFootprintSwitch:
    def test_off_fuses_the_whole_frame_and_is_named_in_the_digest(self, tmp_path):
        wall = (ROOM, 0.0, 1.0)
        donors = [(i, (1.4, dy, 1.0 + dz), wall, "ok", 0)
                  for i, (dy, dz) in enumerate([(0.0, -0.3), (0.2, 0.0), (-0.2, 0.2), (0.0, 0.4)])]
        frames, align, solution, _p = _world(
            tmp_path, donors + [(10, (0.0, 0.0, -0.55), CORNER, "no-fit", 0)])
        params = S.SurfaceParams(neighbour_fit_footprint=False)
        rec = _adopt(frames, align, solution, params)
        assert rec["adopted"] == 1 and frames.footprint == {}
        pred, _img, _fill = frames.load(10, image=False)
        assert np.isfinite(pred).all()
        assert ("neighbour-fit-footprint", False) in params.digest_fields()
        assert not any(isinstance(f, tuple) and f and f[0] == "neighbour-fit-footprint"
                       for f in S.SurfaceParams().digest_fields())


class TestTheParameter:
    def test_on_by_default_and_named_in_the_digest(self):
        p = S.SurfaceParams()
        assert p.neighbour_fit
        assert ("neighbour-fit", NF.NEIGHBOUR_FIT_ID, 10) in p.digest_fields()

    @pytest.mark.parametrize("extra", [{}, {"depth_known_fov": True}, {"fill_gap_frac": 0.024}])
    def test_off_leaves_the_digest_as_it_was(self, extra):
        off = S.SurfaceParams(neighbour_fit=False, **extra).digest_fields()
        on = S.SurfaceParams(**extra).digest_fields()
        assert not any(isinstance(f, tuple) and f and f[0] == "neighbour-fit" for f in off)
        assert [f for f in on if not (isinstance(f, tuple) and f and f[0] == "neighbour-fit")] \
            == list(off)

    def test_the_live_preset_does_not_borrow(self):
        assert not S.SurfaceParams.live().neighbour_fit


class TestTheSolverFrameScope:
    """`neighbour_fit_scope="solver-frame"`: the guard's `final-scale-outlier` pieces
    posed in the placed component's own solver frame, and nothing else."""

    def _solution(self):
        R, t = _look_from((0.0, 0.0, -0.5), CORNER)
        comps = [
            {"index": 0, "gate_state": "placed", "gate_reasons": [], "solver_components": [0]},
            {"index": 1, "gate_state": "unplaced", "gate_reasons": ["solved-separately"],
             "solver_components": [1]},
            {"index": 2, "gate_state": "unplaced", "gate_reasons": ["final-scale-outlier"],
             "solver_components": [0]},
        ]
        poses = {"s:0": {**_pose(R, t, 0), "solver_component": 0},
                 "s:1": {**_pose(R, t, 1), "solver_component": 1},
                 "s:2": {**_pose(R, t, 2), "solver_component": 0},
                 "s:3": {**_pose(R, t, 2), "solver_component": 0}}
        return SimpleNamespace(components=comps, poses=poses, keyframe_ids=list(poses))

    def test_only_scale_isolated_pieces_of_the_placed_solver_frame_are_wanted(
            self, tmp_path, monkeypatch):
        from tower.world_builder import dense_pipeline, live_depth

        monkeypatch.setattr(dense_pipeline, "depth_trust_now", lambda *a, **k: "t")
        monkeypatch.setattr(live_depth, "compatible_live_records", lambda *a, **k: {})
        work = tmp_path / "dense" / "work"
        work.mkdir(parents=True)
        params = S.SurfaceParams(neighbour_fit_scope="solver-frame")
        out, rec = SP.isolated_candidates(None, "w", "s", self._solution(), None, work, params)
        assert rec == {"pieces": 1, "keyframes": 2, "staged": 0} and out == []

    def test_the_scope_is_named_in_the_digest_only_when_widened(self):
        assert ("neighbour-fit-scope", "solver-frame") in S.SurfaceParams(
            neighbour_fit_scope="solver-frame").digest_fields()
        assert not any(isinstance(f, tuple) and f and f[0] == "neighbour-fit-scope"
                       for f in S.SurfaceParams().digest_fields())
