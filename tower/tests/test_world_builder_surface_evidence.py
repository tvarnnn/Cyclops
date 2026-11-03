"""The surface review's findings, each pinned on a scene whose answer is known.

`Glasses-scratch/wb-final-recon/review-recon/REVIEW.md` measured four ways the
fused surface said something the cameras did not:

  M1  a phantom second wall behind a real one, in space nothing measured
  B1  a far wall dozens of frames measured, deleted by a scene-relative clip
  B2  a length scale that moved with which frames a sampling stride hit, so a
      build over part of the walk and the final build disagreed
  M2  a thing near the camera that one frame saw -- or that other frames saw
      straight through -- surviving as geometry

Each test builds the situation from ray-cast depth, so there is no
reconstruction in the ground truth, and asserts the outcome.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_world_builder_surface import ROOM, _camera, _fuse, _look_from, _render_box_depth
from tests.test_world_builder_surface_pipeline import SESSION, WORLD, _params, _synthetic_world

from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP


def test_evidence_cull_preserves_every_counter_and_decision(monkeypatch):
    import torch

    # Contiguous face groups: visible, beyond the right edge, and behind.
    V = np.array([[-0.1, -0.1, 5], [0.1, -0.1, 5], [0, 0.1, 5],
                  [20, -0.1, 5], [20.2, -0.1, 5], [20.1, 0.1, 5],
                  [-0.1, -0.1, -5], [0.1, -0.1, -5], [0, 0.1, -5]], np.float32)
    F = np.concatenate([np.tile([[i, i + 1, i + 2]], (65_536, 1))
                        for i in (0, 3, 6)]).astype(np.int64)
    K = np.array([[40., 0., 8.], [0., 40., 8.], [0., 0., 1.]])
    depth = torch.full((16, 16), 5.)
    views = [(d, torch.ones_like(d, dtype=torch.bool), torch.eye(3),
              torch.zeros(3)) for d in (depth, depth + 1)]
    views.append((depth, torch.ones_like(depth, dtype=torch.bool),
                  torch.diag(torch.tensor([-1., 1., -1.])), torch.zeros(3)))
    params = S.SurfaceParams(min_support_frames=1)
    weak = np.zeros(len(F), bool)
    weak[::65_536] = True
    projected = []
    original_matmul = torch.Tensor.__matmul__

    def observed_matmul(a, b):
        if a.ndim == 2 and a.shape[1] == 3 and len(a) >= 65_536:
            projected.append(len(a))
        return original_matmul(a, b)

    monkeypatch.setattr(torch.Tensor, "__matmul__", observed_matmul)
    run = lambda: S.evidence_filter(V, F, views, K, lambda d: torch.full_like(d, .1),
                                    params, torch.device("cpu"), chunk=65_536,
                                    weak=weak, return_counters=True)
    monkeypatch.delenv("TOWER_WORLD_MESH_EVIDENCE_CULL", raising=False)
    old = run()
    old_projected = sum(projected)
    projected.clear()
    monkeypatch.setenv("TOWER_WORLD_MESH_EVIDENCE_CULL", "off")
    off = run()
    assert np.array_equal(old[0], off[0]) and old[1] == off[1]
    assert all(old[2][key].tobytes() == off[2][key].tobytes() for key in old[2])
    projected.clear()
    monkeypatch.setenv("TOWER_WORLD_MESH_EVIDENCE_CULL", "on")
    new = run()
    assert sum(projected) < old_projected
    assert np.array_equal(old[0], new[0])
    assert old[1] == new[1]
    for name in old[2]:
        assert old[2][name].tobytes() == new[2][name].tobytes(), name
    assert old[2]["support"][0] == 1
    assert old[2]["through"][0] == 1
    assert old[2]["support"][65_536] == 0
    assert old[2]["support"][-1] == 1


def _cull_run(monkeypatch, setting, *args, **kwargs):
    """`evidence_filter` with the cull switch set, and every (storage offset,
    shape) a 2-D matrix product saw: the cull may skip products, never change
    one, since a product's kernel may depend on both."""
    import torch

    seen = []
    original_matmul = torch.Tensor.__matmul__

    def observed_matmul(a, b):
        if a.ndim == 2:
            seen.append((a.storage_offset(), tuple(a.shape)))
        return original_matmul(a, b)

    monkeypatch.setattr(torch.Tensor, "__matmul__", observed_matmul)
    monkeypatch.setenv("TOWER_WORLD_MESH_EVIDENCE_CULL", setting)
    try:
        return S.evidence_filter(*args, **kwargs), seen
    finally:
        monkeypatch.setattr(torch.Tensor, "__matmul__", original_matmul)


def _devices():
    import torch

    return ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="no CUDA device"))]


@pytest.mark.parametrize("device", _devices())
def test_evidence_cull_keeps_the_uncut_chunks_and_every_counter(monkeypatch, device):
    """Chunk edges that fall inside face groups (the build's 2 M chunk is no
    multiple of the 65,536-face group), a group straddling the image edge, and
    groups wholly off-image or behind: every counter bit-identical, and every
    matrix product the cull still runs is one the uncut loop runs."""
    import torch

    g = S.CULL_GROUP_FACES
    V = np.array([[-0.1, -0.1, 5], [0.1, -0.1, 5], [0, 0.1, 5],          # in view
                  [20, -0.1, 5], [20.2, -0.1, 5], [20.1, 0.1, 5],        # off the right edge
                  [-0.1, -0.1, -5], [0.1, -0.1, -5], [0, 0.1, -5]],      # behind
                 np.float32)
    inside, right, behind = [0, 1, 2], [3, 4, 5], [6, 7, 8]
    straddle = np.tile([inside, right], (g // 2, 1))
    F = np.concatenate([np.tile([inside], (g, 1)), straddle, np.tile([right], (g, 1)),
                        np.tile([behind], (g, 1))]).astype(np.int64)
    K = np.array([[40., 0., 8.], [0., 40., 8.], [0., 0., 1.]])
    depth = torch.full((16, 16), 5.)
    ok = torch.ones_like(depth, dtype=torch.bool)
    views = [(depth, ok, torch.eye(3), torch.zeros(3)),
             (depth + 1, ok, torch.eye(3), torch.zeros(3)),
             (depth, ok, torch.diag(torch.tensor([-1., 1., -1.])), torch.zeros(3))]
    weak = np.zeros(len(F), bool)
    weak[::4099] = True
    args = (V, F, views, K, lambda d: torch.full_like(d, .1),
            S.SurfaceParams(min_support_frames=1), torch.device(device))
    kw = dict(chunk=100_000, weak=weak, return_counters=True)

    off, off_seen = _cull_run(monkeypatch, "off", *args, **kw)
    on, on_seen = _cull_run(monkeypatch, "on", *args, **kw)
    assert set(on_seen) <= set(off_seen)
    assert len(on_seen) < len(off_seen), "nothing was culled"
    assert np.array_equal(off[0], on[0]) and off[1] == on[1]
    for name in off[2]:
        assert off[2][name].tobytes() == on[2][name].tobytes(), name
    # the premise: the straddling group holds faces that did vote
    assert off[2]["support"][g] > 0 and off[2]["support"][g + 1] == 0


def test_evidence_cull_bound_holds_far_from_the_origin(monkeypatch):
    """A float32 pixel that rounds INTO the image although the exact one lies
    half a pixel outside it, 10^6 units from the origin (R2's case on L5): a
    fixed 1e-4 slack culled the face and lost its support."""
    import torch

    V = np.array([[-1e-3, 0, 0], [1e-3, 0, 0], [0, 1e-3, 0]], np.float32)
    F = np.array([[0, 1, 2]], np.int64)
    K = np.array([[40., 0., 8.], [0., 40., 8.], [0., 0., 1.]])
    depth = torch.full((16, 17), 1e6)
    t = torch.tensor([212500.015625, 0., 1e6])
    views = [(depth, torch.ones_like(depth, dtype=torch.bool), torch.eye(3), t)]
    args = (V, F, views, K, lambda d: torch.full_like(d, 1.),
            S.SurfaceParams(min_support_frames=1), torch.device("cpu"))
    off, _ = _cull_run(monkeypatch, "off", *args, return_counters=True)
    on, _ = _cull_run(monkeypatch, "on", *args, return_counters=True)
    assert off[2]["support"][0] == 1, "the premise: float32 puts the centroid in the image"
    for name in off[2]:
        assert off[2][name].tobytes() == on[2][name].tobytes(), name


def test_evidence_cull_stands_down_under_reduced_precision_products(monkeypatch):
    """The bound covers IEEE float32 products only; TF32/bfloat16 err ~1e-3."""
    import torch

    g = S.CULL_GROUP_FACES
    V = np.array([[-0.1, -0.1, 5], [0.1, -0.1, 5], [0, 0.1, 5],
                  [20, -0.1, 5], [20.2, -0.1, 5], [20.1, 0.1, 5]], np.float32)
    F = np.concatenate([np.tile([[0, 1, 2]], (g, 1)), np.tile([[3, 4, 5]], (g, 1))])
    K = np.array([[40., 0., 8.], [0., 40., 8.], [0., 0., 1.]])
    depth = torch.full((16, 16), 5.)
    views = [(depth, torch.ones_like(depth, dtype=torch.bool), torch.eye(3), torch.zeros(3))]
    args = (V, F, views, K, lambda d: torch.full_like(d, .1),
            S.SurfaceParams(min_support_frames=1), torch.device("cpu"))
    _, culled = _cull_run(monkeypatch, "on", *args, chunk=g)
    _, uncut = _cull_run(monkeypatch, "off", *args, chunk=g)
    assert len(culled) < len(uncut)

    def refuses():
        raise RuntimeError("mix of the legacy and new APIs")

    for patch in (lambda: "high", lambda: "medium", refuses):
        monkeypatch.setattr(torch, "get_float32_matmul_precision", patch)
        assert not S._float32_products_are_ieee()
        out, seen = _cull_run(monkeypatch, "on", *args, chunk=g)
        assert len(seen) == len(uncut)
    monkeypatch.undo()
    assert S._float32_products_are_ieee()
    # torch >= 2.9's per-backend setting, faked: setting it for real would
    # leave the process's precision state mixed for the tests after this one
    from types import SimpleNamespace
    monkeypatch.setattr(torch.backends.cuda, "matmul", SimpleNamespace(fp32_precision="tf32"))
    assert not S._float32_products_are_ieee()


def _views_tensors(views, device="cpu"):
    import torch

    K, w, h = _camera()
    out = []
    for eye, target, slab in views:
        R, t = _look_from(eye, target)
        d = torch.as_tensor(_render_box_depth(R, t, K, w, h, slab=slab))
        out.append((d, torch.isfinite(d) & (d > 1e-3), torch.as_tensor(R).float(),
                    torch.as_tensor(t).float()))
    return out, K


class TestNoPhantomWallBehindAMeasuredOne:
    """M1. Six clean frames look straight at a wall. Behind the wall nothing was
    measured, so nothing may be emitted there. The old extractor checked the
    evidence only at the voxel nearest each vertex; the far end of an edge at
    the back of the truncation band was an unobserved +1, and marching cubes
    built a complete second sheet ~7 voxels behind the wall."""

    def test_only_the_wall_is_emitted(self):
        voxel = 0.05
        offs = ((0, 0), (0.2, 0), (0, 0.2), (-0.2, 0), (0, -0.2), (0.15, 0.15))
        views = [(np.array([dx, dy, -1.0]), np.array([dx, dy, 5.0]), None)
                 for dx, dy in offs]
        # 7.58 voxels: the shipped canonical ratio of truncation to voxel
        vol = _fuse(views, voxel=voxel, trunc=7.58 * voxel,
                    params=S.SurfaceParams(depth_falloff=False))
        V, F, _ = vol.extract_mesh(min_weight=2.0)
        assert len(F) > 0, "the measured wall was not emitted at all"
        assert (np.abs(V[:, 2] - ROOM) < 2 * voxel).mean() > 0.9
        behind = V[:, 2] > ROOM + 3 * voxel
        assert not behind.any(), (
            f"{int(behind.sum())} vertices behind the wall, where no camera "
            f"measured anything (deepest at +{(V[:, 2].max() - ROOM) / voxel:.1f} voxels)")


class TestAFarMeasuredWallIsKept:
    """B1. Most of every frame is a close object; the room's far wall fills the
    rest at 5.5x that depth. The old clip, 2.6x the scene median, deleted the
    wall although every frame measured it and the solve's anchors lie on it.
    The bound is now the frame's own farthest anchor x 1.5."""

    SLAB = ((-1.2, -1.2, -1.6), (0.1, 1.2, -1.5))

    def _views(self):
        offs = ((0.0, 0.0), (0.12, 0.0), (0.0, 0.12), (-0.1, -0.08), (0.08, 0.1), (-0.12, 0.05))
        return [(np.array([dx, dy, -2.5]), np.array([dx, dy, 5.0]), self.SLAB)
                for dx, dy in offs]

    def test_the_depth_bound_follows_the_frames_anchors(self):
        p = S.SurfaceParams()
        assert S.depth_bound(p, 1.0, 5.5) == pytest.approx(5.5 * p.anchor_depth_multiple)
        # no anchor range recorded: the generous scene-relative fallback
        assert S.depth_bound(p, 1.0, None) == pytest.approx(p.max_depth_frac)

    def test_the_far_wall_survives_the_build(self, tmp_path):
        store = _synthetic_world(tmp_path, views=self._views(), anchors=True)
        # the premise: the near object is most of what the frames measured
        K, w, h = _camera()
        R, t = _look_from(*self._views()[0][:2])
        d = _render_box_depth(R, t, K, w, h, slab=self.SLAB)
        assert np.nanmedian(d) * 2.6 < ROOM + 2.5 - 0.1

        result = SP.surfacify(store, WORLD, SESSION, params=_params(min_support_frames=2))
        assert result.state == SP.STATE_OK, result.detail
        man = SP.read_surface_manifest(store, WORLD, SESSION)
        V, F, _, _ = S.read_mesh_bytes(SP.read_surface_level(store, WORLD, SESSION, 0,
                                                             manifest=man))
        wall = np.abs(V[:, 2] - ROOM) < 0.15
        assert wall.sum() > 50, (
            f"{int(wall.sum())} vertices on the far wall every frame measured")


class TestTheScaleDoesNotDependOnWhichFramesAreSampled:
    """B2. The old scale took every n-th frame's dense depth, n = frames // 24,
    so which frames it saw depended on how long the walk was. A walk that
    alternates close-ups and room views then got one scale from a prefix
    (stride 1: both kinds) and another from the whole walk (stride 2: only
    close-ups). On the canonical world prefixes ran 1.20-1.43x the final. The
    scale is now pooled over every gated frame's sparse observations."""

    SLAB = ((-1.5, -1.5, -1.6), (1.5, 1.5, -1.5))

    def _views(self, n):
        views = []
        for i in range(n):
            dx = 0.02 * (i % 5)
            near = i % 2 == 0
            views.append((np.array([dx, 0.0, -2.5]), np.array([dx, 0.0, 5.0]),
                          self.SLAB if near else None))
        return views

    def _scale(self, tmp_path, n):
        from tower.world_builder.global_solve import load_solution

        store = _synthetic_world(tmp_path / str(n), views=self._views(n), anchors=True)
        solution = load_solution(store, WORLD, SESSION)
        dense = store.world_dir(WORLD) / "dense" / SESSION
        import json

        align = json.loads((dense / "align.json").read_text())
        frames = SP._Frames(align, dense / "work", solution, S.SurfaceParams())
        return SP._scene_scale(frames, solution)

    def test_a_prefix_and_the_whole_walk_agree(self, tmp_path):
        prefix, src_p = self._scale(tmp_path, 24)
        full, src_f = self._scale(tmp_path, 48)
        assert src_p == src_f == "sparse-observation-depth"
        assert prefix == pytest.approx(full, rel=0.05), (prefix, full)


class TestANearGhostIsNotEmitted:
    """M2. Weight is a sum. With the near-depth boost one close frame reaches
    `min_weight` by itself, and carving only reaches `max_carve_voxels` in
    front of each frame's own surface, so the frames that look past a hand at
    a wall three units away never carve it. On the canonical capture every
    face within 0.5 units of the walked path was a hand or a lap."""

    WALL = [(np.array([dx, dy, -1.0]), np.array([dx, dy, 5.0]))
            for dx, dy in ((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25))]

    def _extract(self, views, voxel=0.02):
        import torch

        params = S.SurfaceParams()
        vol = _fuse(views, voxel=voxel, trunc=3 * voxel, params=params)
        V, F, C = vol.extract_mesh(min_weight=params.min_weight)
        tv, K = _views_tensors(views)
        keep, stats = S.evidence_filter(V, F, tv, K, vol.trunc_at, params,
                                        torch.device("cpu"))
        return V, F, keep, stats

    @staticmethod
    def _in(V, F, slab, pad=0.04):
        c = V[F].mean(axis=1)
        return np.all((c > np.asarray(slab[0]) - pad) & (c < np.asarray(slab[1]) + pad), axis=1)

    def test_a_thing_one_close_frame_saw_is_not_emitted(self):
        slab = ((-0.06, -0.06, -0.52), (0.06, 0.06, -0.48))
        views = [(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 5.0]), slab)]
        views += [(e, tg, None) for e, tg in self.WALL]
        V, F, keep, stats = self._extract(views)
        ghost = self._in(V, F, slab)
        assert ghost.sum() > 0, "the fixture did not produce a near ghost to remove"
        assert not (ghost & keep).any(), (
            f"{int((ghost & keep).sum())} faces of a thing one frame saw survived")
        wall = np.abs(V[F].mean(axis=1)[:, 2] - ROOM) < 0.1
        assert (keep[wall]).mean() > 0.9, "the filter removed the wall everyone agreed on"

    def test_a_thing_other_frames_saw_through_is_not_emitted(self):
        slab = ((-0.06, -0.06, -0.52), (0.06, 0.06, -0.48))
        # two frames see it, so the frame count alone would keep it...
        views = [(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 5.0]), slab),
                 (np.array([0.02, 0.0, -1.0]), np.array([0.02, 0.0, 5.0]), slab)]
        # ...and five frames behind them look straight through where it was
        views += [(np.array([dx, dy, -2.0]), np.array([dx, dy, 5.0]), None)
                  for dx, dy in ((0, 0), (0.01, 0), (0, 0.01), (-0.01, 0), (0, -0.01))]
        V, F, keep, stats = self._extract(views)
        ghost = self._in(V, F, slab)
        assert ghost.sum() > 0, "the fixture did not produce a near ghost to remove"
        assert not (ghost & keep).any(), (
            f"{int((ghost & keep).sum())} faces of a thing five frames saw through survived")


class TestTheContradictionTestIsARatio:
    """Review 2, S2. The contract said a hand other frames saw through "does not
    survive". The rule is `through >= contradiction_ratio x support`, and
    consecutive keyframes often hold the same hand. Pinned both ways so the
    contract's wording cannot drift from it again."""

    SLAB = ((-0.06, -0.06, -0.52), (0.06, 0.06, -0.48))

    def _hand(self, n_saw, n_through):
        views = [(np.array([dx, 0.0, -1.0]), np.array([dx, 0.0, 5.0]), self.SLAB)
                 for dx in (0.0, 0.02, -0.02)[:n_saw]]
        views += [(np.array([dx, dy, -2.0]), np.array([dx, dy, 5.0]), None)
                  for dx, dy in ((0, 0), (0.01, 0), (0, 0.01), (-0.01, 0),
                                 (0, -0.01))[:n_through]]
        V, F, keep, stats = TestANearGhostIsNotEmitted()._extract(views)
        hand = TestANearGhostIsNotEmitted._in(V, F, self.SLAB)
        assert hand.sum() > 0, "the fixture did not produce the near thing"
        return float(keep[hand].mean())

    def test_two_that_saw_it_against_four_that_saw_past_it_is_removed(self):
        assert self._hand(2, 4) == 0.0

    def test_three_that_saw_it_against_five_that_saw_past_it_stays(self):
        # 5 < 2 x 3: the documented limit of the rule, not a wish.
        assert self._hand(3, 5) > 0.9


class TestASurfaceIsKeptOnlyFromItsFront:
    """Review 2, S5. `drop_back_facing` took the shipped canonical area seen from
    behind from 18% to 0.6%, and no test failed with it switched off."""

    @staticmethod
    def _wall_views():
        return [(np.array([dx, dy, -1.0]), np.array([dx, dy, 5.0]), None)
                for dx, dy in ((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25))]

    def _filter(self, V, F, **kw):
        import torch

        tv, K = _views_tensors(self._wall_views())
        voxel = 0.02
        params = S.SurfaceParams(**kw)
        return S.evidence_filter(np.asarray(V, np.float32), np.asarray(F), tv, K,
                                 lambda d: torch.full_like(d, 3 * voxel), params,
                                 torch.device("cpu"))

    def test_a_face_every_supporting_camera_sees_from_behind_is_removed(self):
        s = 0.05
        # Two triangles on the measured wall, the same place, opposite windings.
        # The one whose normal points at the cameras (-z) is the wall's front.
        V = [(-s, -s, ROOM), (s, -s, ROOM), (-s, s, ROOM),
             (-s, -s, ROOM), (-s, s, ROOM), (s, -s, ROOM)]
        F = [(0, 1, 2), (3, 4, 5)]
        n0 = np.cross(np.subtract(V[1], V[0]), np.subtract(V[2], V[0]))
        n1 = np.cross(np.subtract(V[4], V[3]), np.subtract(V[5], V[3]))
        assert n0[2] > 0 > n1[2], "fixture: face 0 faces away from the cameras"

        keep, stats = self._filter(V, F)
        assert list(keep) == [False, True], stats
        assert stats["dropped_back_facing"] == 1
        # and it is that rule, not the support count, that removed it
        keep_off, _ = self._filter(V, F, drop_back_facing=False)
        assert list(keep_off) == [True, True]

    def test_a_thin_board_seen_from_both_sides_keeps_both_faces(self):
        import torch

        from tests.test_world_builder_surface import _fuse

        voxel = 0.01
        thick = 2 * voxel
        board = ((-0.5, -0.5, 0.0), (0.5, 0.5, thick))
        offs = ((0, 0), (0.05, 0), (0, 0.05), (-0.05, -0.03))
        views = [(np.array([dx, dy, -1.2]), np.array([dx, dy, 5.0]), board) for dx, dy in offs]
        views += [(np.array([dx, dy, 1.2]), np.array([dx, dy, -5.0]), board) for dx, dy in offs]
        params = S.SurfaceParams()
        vol = _fuse(views, voxel=voxel, trunc=params.trunc_voxels * voxel, params=params)
        V, F, _C = vol.extract_mesh(params.min_weight)
        V, F, _C, _ = S.weld_mesh(V, F, _C, voxel * 1e-3)
        tv, K = _views_tensors(views)
        keep, stats = S.evidence_filter(V, F, tv, K, vol.trunc_at, params,
                                        torch.device("cpu"))
        c = V[F].mean(axis=1)
        centre = ((np.abs(c[:, 0]) < 0.3) & (np.abs(c[:, 1]) < 0.3)
                  & (np.abs(c[:, 2] - thick / 2) < 0.1))
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        toward_minus = centre & (n[:, 2] < 0)
        toward_plus = centre & (n[:, 2] > 0)
        assert toward_minus.sum() > 100 and toward_plus.sum() > 100, "fixture"
        assert keep[toward_minus].mean() > 0.9, stats
        assert keep[toward_plus].mean() > 0.9, stats
