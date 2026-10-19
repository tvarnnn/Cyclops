"""Depth fits borrowed from the room (`neighbour_fit.py`, `_Frames.adopt_neighbour_fits`).

A posed frame whose own sparse fit is missing or refused is fitted to the depth its fused
neighbours measured, and fused only when that fit passes the gate every fused frame passes.
The scene is a synthetic box room with exact depth, so "the right fit" and "the wrong pose"
are known by construction. `TestTheRoomsGauge` adds what a box room alone never exercises: a
stretch of the solve drawn in its OWN gauge (its poses and its own fits scaled together, as on
the 10.1 replay cc3), which is internally consistent and therefore invisible to the held-out
error.
"""

from __future__ import annotations

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


# The final scale guard's bands (`final_scale_guard.GuardParams`), as `SurfaceParams.neighbour_fit_scale_band`
# names them.
CLEAR, SCREEN = SP._scale_band("clear"), SP._scale_band("screen")
NO_GAUGE = (1e-12, 1e12)    # the rule switched off: what `neighbour-fit/1` admitted


def _room_scale_log():
    """The room's units per metre of prediction, as a room frame reads it."""
    _R, _t, z = _view(EYES[0], CORNER)
    return NF.scale_log(z, A0 * z + B0)


def _fit(pred, R, t, neighbours, *, band=CLEAR, ref=None, **kw):
    u, v, z, src = _anchors_for(R, t, neighbours)
    return NF.fit_from_neighbours(pred, "depth", u, v, z, src, gate_rel=0.08,
                                  min_anchors=20, min_frames=2,
                                  room_scale_log=_room_scale_log() if ref is None else ref,
                                  scale_band=band, **kw)


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


class TestTheRoomsGauge:
    """THE ROOM'S GAUGE (`neighbour_fit.py`): the cc3 mechanism. A stretch the solve drew in its
    own gauge lends anchors that agree with each other, so the held-out error passes; the
    candidate's metric prediction is the witness that they are not where the room is."""

    def _candidate(self, s):
        eye = (0.0, 0.0, -0.55)
        R, _t, z = _view(eye, CORNER)
        return A0 * z + B0, R, -R @ _in_gauge(eye, s)

    @pytest.mark.parametrize("s", [4.16, 0.07])
    def test_a_stretch_in_its_own_gauge_cannot_vouch_for_its_own_frames(self, s):
        """x4.16 is cc3's excursion 347-358; x0.07 its collapsed stretches."""
        nbs = [_stretch_neighbour(i, e, CORNER, s) for i, e in enumerate(STRETCH)]
        pred, R, t = self._candidate(s)
        # What neighbour-fit/1 did: two or more consistent donors, a near-perfect held-out
        # error, a slope `s` times off -- admitted.
        old = _fit(pred, R, t, nbs, band=NO_GAUGE)
        assert old["ok"] and old["anchor_frames"] == 3 and old["held_out_rel"] < 0.01
        assert old["a"] == pytest.approx(A0 / s, rel=0.02)
        # The anchors read `s` against the candidate's metric prediction: refused.
        for band in (CLEAR, SCREEN):
            fit = _fit(pred, R, t, nbs, band=band)
            assert not fit["ok"] and fit["code"] == "scale"
            assert fit["scale_ratio"] == pytest.approx(s, rel=0.05)

    @pytest.mark.parametrize("s, clear, screen", [(1.0, True, True), (1.1, True, True),
                                                  (1.3, True, False), (0.77, True, False),
                                                  (1.45, False, False), (0.7, False, False)])
    def test_the_guards_bands_decide(self, s, clear, screen):
        """Inside the guard's screen band the gauge agrees; between screen and clear it is the
        guard's borderline (admitted by the default `clear`); beyond clear, a gross error."""
        nbs = [_stretch_neighbour(i, e, CORNER, s) for i, e in enumerate(STRETCH)]
        pred, R, t = self._candidate(s)
        assert _fit(pred, R, t, nbs, band=CLEAR)["ok"] is clear
        assert _fit(pred, R, t, nbs, band=SCREEN)["ok"] is screen

    def test_a_pose_error_along_the_view_is_absorbed_by_the_shift_but_not_by_the_witness(self):
        """A frame facing a plain wall square on: the solve put it 1.5 m too far back. The
        room's anchors on the wall are all at one depth, so an affine shift follows them
        exactly and the held-out error cannot see the error; the metric prediction can."""
        target = (0.0, 0.0, ROOM)
        eyes = [(-0.4, 0.0, -0.5), (0.4, 0.0, -0.5), (0.0, 0.3, -0.6), (0.0, -0.3, -0.4)]
        nbs = [_neighbour(i, e, target) for i, e in enumerate(eyes)]
        eye = np.array([0.0, 0.0, -0.5])
        _R, _t, z = _view(eye, target)
        pred = A0 * z + B0
        R, t = _look_from(eye - np.array([0.0, 0.0, 1.5]), target)
        ref = NF.scale_log(z, pred)
        old = _fit(pred, R, t, nbs, band=NO_GAUGE, ref=ref)
        assert old["ok"] and old["held_out_rel"] < 0.01
        fit = _fit(pred, R, t, nbs, ref=ref)
        assert not fit["ok"] and fit["code"] == "scale"
        assert fit["scale_ratio"] == pytest.approx((ROOM + 0.5 + 1.5) / (ROOM + 0.5), rel=0.05)

    def test_scale_log_is_the_guards_per_camera_statistic(self):
        z = np.array([2.0, 4.0, 6.0, np.nan, 1.0])
        p = np.array([1.0, 2.0, 3.0, 1.0, -1.0])
        assert NF.scale_log(z, p) == pytest.approx(np.log(2.0))
        assert NF.scale_log(np.array([1.0]), np.array([0.0])) is None


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


def _adopt(frames, align, solution, params):
    return frames.adopt_neighbour_fits(align, solution, params, median_depth=3.0,
                                       voxel=params.voxel_frac * 3.0)


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


class TestTheRoomsGaugeInTheSurfaceStage:
    """The cc3 layout in miniature: the room's frames, and beside them a stretch the solve drew
    x4.16 the room's scale whose own fits agree with each other; one unfitted frame in each."""

    OPP = (-ROOM, 0.0, -ROOM)

    def _specs(self, gauge=4.16):
        room = [(i, (-e[0], e[1], -e[2]), self.OPP, "ok", 0) for i, e in enumerate(EYES)]
        room += [(4, (0.1, 0.0, 0.45), self.OPP, "ok", 0)]
        stretch = [(10 + i, e, CORNER, "ok", 0, gauge) for i, e in enumerate(STRETCH)]
        return room + stretch + [(20, (0.0, 0.0, -0.55), CORNER, "no-fit", 0, gauge),
                                 (21, (0.0, 0.05, 0.55), self.OPP, "no-fit", 0)]

    def test_the_stretch_cannot_lend_its_gauge_to_its_own_frames(self, tmp_path):
        frames, align, solution, params = _world(tmp_path, self._specs())
        assert sorted(it[0] for it in frames.items) == [0, 1, 2, 3, 4, 10, 11, 12]
        rec = _adopt(frames, align, solution, params)
        assert rec["candidates"] == 2 and rec["adopted"] == 1
        assert rec["refused"] == {"scale": 1}
        assert [f["ki"] for f in rec["adopted_frames"]] == [21]
        assert rec["adopted_frames"][0]["scale_ratio"] == pytest.approx(1.0, abs=0.05)
        # The reference is the room's: five frames outvote the stretch's three.
        assert rec["scale"]["state"] == "measured" and rec["scale"]["band"] == "clear"
        assert rec["scale"]["reference_units_per_metre"] == pytest.approx(
            np.exp(_room_scale_log()), rel=0.05)
        assert 20 not in {it[0] for it in frames.items}

    def test_a_stretch_in_the_rooms_gauge_still_lends(self, tmp_path):
        frames, align, solution, params = _world(tmp_path, self._specs(gauge=1.0))
        rec = _adopt(frames, align, solution, params)
        assert rec["adopted"] == 2 and rec["refused"] == {}

    def test_without_a_metric_prediction_nothing_is_borrowed(self, tmp_path):
        specs = [(i, e, CORNER, "ok", 0) for i, e in enumerate(EYES)]
        specs += [(10, (0.0, 0.0, -0.55), CORNER, "no-fit", 0)]
        frames, align, solution, params = _world(tmp_path, specs)
        frames.kind = "disparity"
        rec = _adopt(frames, align, solution, params)
        assert rec["adopted"] == 0 and rec["refused"] == {"no-room-scale": 1}
        assert rec["scale"]["state"] == "no-metric-depth"
        assert sorted(it[0] for it in frames.items) == [0, 1, 2, 3]

    def test_the_screen_band_is_stricter_and_named_in_the_digest(self, tmp_path):
        frames, align, solution, params = _world(tmp_path, self._specs(gauge=1.3))
        assert _adopt(frames, align, solution, params)["adopted"] == 2
        frames, align, solution, _p = _world(tmp_path / "screen", self._specs(gauge=1.3))
        rec = _adopt(frames, align, solution, S.SurfaceParams(neighbour_fit_scale_band="screen"))
        assert rec["adopted"] == 1 and rec["refused"] == {"scale": 1}
        assert ("neighbour-fit-scale-band", "screen") in S.SurfaceParams(
            neighbour_fit_scale_band="screen").digest_fields()
        assert not any(isinstance(f, tuple) and f and f[0] == "neighbour-fit-scale-band"
                       for f in S.SurfaceParams().digest_fields())
        with pytest.raises(ValueError):
            SP._scale_band("loose")


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
