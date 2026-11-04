"""One stray pose or a spread of far depth must not cut a whole room's resolution.

WHY THIS FILE EXISTS

Replay int4-w102-2 of walk 10.2 (2026-11-03, FIX-BOUND) built its room at a voxel
3.54x coarser than every other replay of the same walk -- 618 K faces instead of
4.1-4.7 M. Nothing was wrong with the room. Seven of its 959 fused frames were:
four whose every sparse anchor the solver had placed 17-99 units out (in a room of
radius ~6), and three cameras the solve put 19-24 units out, just inside the pose
gate. Each keyed 0.36-1.06 M blocks against a median frame's 8.5 K, the walk
needed 4.09 M blocks against the field's budget of 360 K, and the budget's one
answer was to coarsen the voxel of the whole room. Separately, the fusion bound
had been widened from 74 to 89 units by sparse points of OTHER solver
components, which the surface never fuses.

`TOWER_WORLD_SURFACE_BOUND_FIX` (default off) changes both:

  * the bound's point envelope is built from the fused component's points only
    (`SurfaceParams.bound_from_fused_component`);
  * a walk over the budget first loses the blocks no kept face can come from,
    then the blocks keyed only by frames the solve's own sparse points
    contradict, and only then a coarser voxel (`budget_spares_the_room`,
    `budget_drops_contradicted`, `surface_pipeline._spare_the_room`). By
    evidence, never by distance: a far wall healthy frames see is always kept.

Off, every surface is byte for byte what it was.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_world_builder_surface_pipeline import (
    SESSION,
    WORLD,
    _manifest,
    _params,
    _synthetic_world,
)
from tower.config import WORLD_SURFACE_BOUND_FIX_ENV, world_surface_bound_fix_setting
from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------


class TestTheSwitch:

    def test_it_is_off_by_default(self, monkeypatch):
        monkeypatch.delenv(WORLD_SURFACE_BOUND_FIX_ENV, raising=False)
        assert world_surface_bound_fix_setting() is False

    @pytest.mark.parametrize("value", ["on", "1", "true", "yes"])
    def test_it_turns_on(self, monkeypatch, value):
        monkeypatch.setenv(WORLD_SURFACE_BOUND_FIX_ENV, value)
        assert world_surface_bound_fix_setting() is True

    def test_off_the_cache_identity_is_unchanged(self):
        """A surface built before the switch existed must not rebuild because
        it did: off adds nothing to the digest."""
        off = S.SurfaceParams().digest_fields()
        assert S.SurfaceParams(bound_from_fused_component=False, budget_spares_the_room=False,
                               budget_drops_contradicted=False).digest_fields() == off
        assert not [f for f in off if isinstance(f, tuple) and f and f[0] in ("bound-points", "budget")]

    def test_on_the_cache_identity_says_so_and_only_appends(self):
        off = S.SurfaceParams().digest_fields()
        on = S.SurfaceParams(bound_from_fused_component=True, budget_spares_the_room=True,
                             budget_drops_contradicted=True).digest_fields()
        assert on[:len(off)] == off
        assert on[len(off):] == (("bound-points", "fused-component"),
                                 ("budget", "unsupported-then-voxel"),
                                 ("budget", "contradicted-frames", S.CONTRADICTED_DEPTH_FACTOR))

    def test_the_switch_turns_on_all_three(self, monkeypatch):
        """`surfacify` reads the switch per build; checked through the params it
        hands `_build` (a stand-in that stops there)."""
        seen = {}

        def stop(root, frames, params, *a, **k):
            seen["params"] = params
            raise RuntimeError("stop")

        monkeypatch.setenv(WORLD_SURFACE_BOUND_FIX_ENV, "on")
        monkeypatch.setattr(SP, "_build", stop)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            store = _synthetic_world(Path(d))
            SP.surfacify(store, WORLD, SESSION, params=_params())
        p = seen["params"]
        assert p.bound_from_fused_component and p.budget_spares_the_room and p.budget_drops_contradicted


# ---------------------------------------------------------------------------
# the bound's points
# ---------------------------------------------------------------------------


class _FramesAt:
    """What `_fusion_bound` reads of `_Frames`: the items' poses."""

    def __init__(self, centres):
        I = np.eye(3)
        self.items = [(i, 1.0, 0.0, I, -np.asarray(c, float), 0.01, None)
                      for i, c in enumerate(centres)]

    def __len__(self):
        return len(self.items)


def _frames_at(centres):
    return _FramesAt(centres)


def _ring(n=24, r=3.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), np.zeros(n), r * np.sin(a)], 1)


class TestTheBoundIsBuiltFromTheFusedComponent:
    """Points of another component are where the solver put a piece it could
    not place relative to the fused one: they say nothing about where the
    fused frames' surfaces are."""

    def _solution(self, with_component=True):
        room = _ring(64, 3.0)
        # A second component, in its own gauge: inside the 10x-median filter,
        # far outside the room.
        other = _ring(16, 25.0) + np.array([20.0, 0.0, 0.0])
        xyz = np.concatenate([room, other]).astype(np.float32)
        comp = np.concatenate([np.zeros(len(room), np.int32), np.ones(len(other), np.int32)])
        return SimpleNamespace(xyz=xyz, component=comp if with_component else None)

    def test_off_every_component_widens_the_bound(self):
        frames, sol = _frames_at(_ring(12, 1.0)), self._solution()
        off = SP._fusion_bound(frames, sol, S.SurfaceParams(), median_depth=1.0)
        assert off.hi[0] > 40.0

    def test_on_only_the_fused_component_does(self):
        frames, sol = _frames_at(_ring(12, 1.0)), self._solution()
        params = S.SurfaceParams(bound_from_fused_component=True)
        on = SP._fusion_bound(frames, sol, params, median_depth=1.0)
        alone = S.robust_fusion_bound(sol.xyz[sol.component == 0], _ring(12, 1.0),
                                      reach=params.max_depth_frac * 1.0)
        np.testing.assert_allclose(on.lo, alone.lo)
        np.testing.assert_allclose(on.hi, alone.hi)
        assert on.hi[0] < 20.0
        assert "component 0 only (64 of 80)" in on.detail

    def test_a_solution_without_components_keeps_every_point(self):
        """Standing down only ever grows the envelope, the safe direction."""
        frames = _frames_at(_ring(12, 1.0))
        params = S.SurfaceParams(bound_from_fused_component=True)
        on = SP._fusion_bound(frames, self._solution(with_component=False), params, 1.0)
        off = SP._fusion_bound(frames, self._solution(), S.SurfaceParams(), 1.0)
        np.testing.assert_allclose(on.hi, off.hi)
        assert "only" not in on.detail


# ---------------------------------------------------------------------------
# the budget's answers, in order
# ---------------------------------------------------------------------------


def _keys(coords):
    return S.block_key(torch.as_tensor(np.asarray(coords, np.int64)))


def _pass(frame_coords):
    """One allocation pass from each frame's block coordinates: (unique keys,
    frames per key, each frame's keys), as `_build` computes them."""
    fk = [torch.unique(_keys(c)) for c in frame_coords]
    allk, n = torch.unique(torch.cat(fk), return_counts=True)
    return allk, n, fk


class TestTheBudgetAnswersInOrder:
    """`_spare_the_room`: by evidence, never by distance."""

    def _params(self, **kw):
        base = dict(max_blocks=4, min_support_frames=2, budget_spares_the_room=True,
                    budget_drops_contradicted=True)
        base.update(kw)
        return S.SurfaceParams(**base)

    def test_blocks_fewer_frames_key_than_a_face_needs_go_first(self):
        allk, n, fk = _pass([[[0, 0, 0], [0, 0, 1], [0, 0, 2]],
                             [[0, 0, 0], [0, 0, 1], [0, 0, 3]]])
        asked = []
        kept, dropped = SP._spare_the_room(allk, n, fk, params=self._params(max_blocks=2),
                                           contradicted=lambda: asked.append(1) or [False, False])
        assert dropped == {"unsupported": 2, "contradicted": 0, "frames_contradicted": 0}
        assert sorted(kept.tolist()) == sorted(_keys([[0, 0, 0], [0, 0, 1]]).tolist())
        assert not asked, "the evidence is not even read once answer 1 is enough"

    def test_a_count_of_one_drops_nothing(self):
        allk, n, fk = _pass([[[0, 0, i] for i in range(6)]])
        kept, dropped = SP._spare_the_room(allk, n, fk, params=self._params(min_support_frames=1,
                                                                            budget_drops_contradicted=False))
        assert dropped["unsupported"] == 0 and kept.numel() == 6

    def test_blocks_only_contradicted_frames_key_go_next(self):
        room = [[0, 0, i] for i in range(3)]
        ghost = [[0, 50, i] for i in range(4)]          # keyed by the two contradicted frames only
        allk, n, fk = _pass([room, room, ghost, ghost])
        kept, dropped = SP._spare_the_room(allk, n, fk, params=self._params(max_blocks=3),
                                           contradicted=lambda: [False, False, True, True])
        assert dropped == {"unsupported": 0, "contradicted": 4, "frames_contradicted": 2}
        assert sorted(kept.tolist()) == sorted(_keys(room).tolist())

    def test_one_healthy_frame_vouches_for_a_block(self):
        """A block a contradicted frame keys is kept if any uncontradicted frame
        keys it too: only blocks NO healthy frame measured go."""
        shared = [[0, 0, 0], [0, 0, 1]]
        allk, n, fk = _pass([shared, shared + [[9, 9, 9]], [[9, 9, 9]]])
        kept, dropped = SP._spare_the_room(allk, n, fk, params=self._params(max_blocks=1),
                                           contradicted=lambda: [True, False, True])
        assert dropped["contradicted"] == 0 and kept.numel() == 3

    def test_a_far_wall_healthy_frames_see_is_kept_however_far(self):
        """The case a distance rule gets wrong: a wall a hundred blocks from
        every camera, measured by several healthy frames, over the budget."""
        near = [[0, 0, i] for i in range(3)]
        wall = [[100, j, 400] for j in range(5)]
        allk, n, fk = _pass([near + wall, near + wall, wall])
        kept, dropped = SP._spare_the_room(allk, n, fk, params=self._params(max_blocks=2),
                                           contradicted=lambda: [False, False, False])
        assert dropped == {"unsupported": 0, "contradicted": 0, "frames_contradicted": 0}
        assert kept.numel() == 8, "over the budget, the caller coarsens; nothing here drops the wall"

    def test_without_the_evidence_step_contradicted_frames_keep_their_blocks(self):
        ghost = [[0, 50, i] for i in range(4)]
        allk, n, fk = _pass([ghost, ghost])
        kept, dropped = SP._spare_the_room(allk, n, fk, params=self._params(budget_drops_contradicted=False,
                                                                            max_blocks=1),
                                           contradicted=lambda: [True, True])
        assert dropped["contradicted"] == 0 and kept.numel() == 4


# ---------------------------------------------------------------------------
# which frames the solve's own points contradict
# ---------------------------------------------------------------------------


def _plane_frame(depth, *, w=64, h=48, f=60.0, eye=(0.0, 0.0, 0.0)):
    """A camera at `eye` looking down +z at a constant-depth plane: the cached
    tuple `_build` holds, plus K."""
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    z = torch.full((h, w), float(depth))
    ok = torch.ones((h, w), dtype=torch.bool)
    R = torch.eye(3)
    t = -torch.as_tensor(np.asarray(eye, np.float32))
    return (z, ok, None, R, t, None), K


def _points_on(depth, n=8, half=0.3):
    g = np.linspace(-half, half, n)
    X, Y = np.meshgrid(g * depth, g * depth)
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(depth))], 1)


class TestContradictedFrames:

    def _flags(self, frames, points, K, scale=1.0, **kw):
        params = S.SurfaceParams(**kw)
        return SP._contradicted_frames([f for f in frames], np.asarray(points) * scale, K,
                                       params=params, trel=0.05, floor=0.01 * scale,
                                       device=torch.device("cpu"))

    def test_a_frame_that_measures_through_the_solves_surface_is_contradicted(self):
        good, K = _plane_frame(2.0)
        stretched, _ = _plane_frame(16.0)        # its fit put the same surface 8x too far
        assert self._flags([good, stretched], _points_on(2.0), K) == [False, True]

    def test_a_far_wall_whose_points_agree_is_not(self):
        wall, K = _plane_frame(50.0)
        assert self._flags([wall], _points_on(50.0), K) == [False]

    def test_a_point_behind_the_surface_is_no_evidence_against_it(self):
        near, K = _plane_frame(2.0)              # an occluder in front of the points
        assert self._flags([near], _points_on(10.0), K) == [False]

    def test_too_few_contradicting_points_to_judge(self):
        stretched, K = _plane_frame(16.0)
        few = _points_on(2.0, n=4)               # 16 points: under `min_sparse_points`
        assert self._flags([stretched], few, K) == [False]

    def test_mild_disagreement_is_not_contradiction(self):
        off_by_half, K = _plane_frame(3.0)       # 1.5x the points' depth: under the 2x line
        assert self._flags([off_by_half], _points_on(2.0), K) == [False]

    def test_it_is_gauge_free(self):
        good, K = _plane_frame(2000.0)
        stretched, _ = _plane_frame(16000.0)
        assert self._flags([good, stretched], _points_on(2.0), K, scale=1000.0) == [False, True]


# ---------------------------------------------------------------------------
# answer 1 and the allocation's pixel subsample
# ---------------------------------------------------------------------------


class TestAThinFeatureIsCountedByEveryFrameThatSeesIt:
    """`blocks_for_depth` keys a subsample of a frame's pixels (to 60,000) and
    face support (`evidence_filter`) reads every pixel, so answer 1 is only
    safe if every block a frame's full-resolution depth lands in is among its
    keys. Checked at the corpus' frame size (359 x 639: one pixel in three),
    on a thin post in front of a far wall, from two viewpoints."""

    W, H, F = 359, 639, 465.0

    def _frame(self, eye_x, md, voxel):
        K = np.array([[self.F, 0, self.W / 2], [0, self.F, self.H / 2], [0, 0, 1.0]])
        uu = torch.arange(self.W, dtype=torch.float32).view(1, -1).expand(self.H, self.W)
        x_at_post = (uu - K[0, 2]) / self.F          # ray x per unit depth
        depth = torch.full((self.H, self.W), 6.0 * md * 0.9)   # the wall, near the far bound
        post_x, post_z, half = 0.0, 1.5 * md, 0.012 * md       # a post ~5 px wide
        on_post = ((x_at_post * post_z + eye_x) - post_x).abs() <= half
        depth = torch.where(on_post, torch.full_like(depth, post_z), depth)
        params = S.SurfaceParams()
        ok, _cos = S.depth_validity(depth, K, params, md)
        R = torch.eye(3)
        t = -torch.tensor([eye_x, 0.0, 0.0])
        return depth, ok, R, t, K, on_post

    def test_every_full_resolution_block_is_keyed(self):
        md = 3.84
        voxel = 0.0051 * md
        vol = S.SurfaceVolume(voxel, 3 * voxel, device=torch.device("cpu"), trunc_rel=0.053,
                              trunc_max=12 * voxel)
        post_blocks = []
        for eye_x in (0.0, 0.07):
            depth, ok, R, t, K, on_post = self._frame(eye_x, md, voxel)
            keys = vol.blocks_for_depth(depth, ok, R, t, K)
            assert ok.sum() > 120_000, "the subsample must actually thin this frame"
            vy, vx = torch.nonzero(ok, as_tuple=True)
            z = depth[vy, vx]
            X = torch.stack([(vx - K[0, 2]) / K[0, 0] * z, (vy - K[1, 2]) / K[1, 1] * z, z], 1) - t
            full = torch.unique(S.block_key(torch.floor(X / (voxel * S.BLOCK)).to(torch.int64)))
            assert bool(torch.isin(full, keys).all()), "a full-resolution block was not keyed"
            pv = ok & on_post
            vy, vx = torch.nonzero(pv, as_tuple=True)
            assert vy.numel() > 0, "the post must have valid depth to be a test"
            z = depth[vy, vx]
            X = torch.stack([(vx - K[0, 2]) / K[0, 0] * z, (vy - K[1, 2]) / K[1, 1] * z, z], 1) - t
            post_blocks.append((torch.unique(S.block_key(torch.floor(X / (voxel * S.BLOCK)).to(torch.int64))),
                                keys))
        (post0, keys0), (post1, keys1) = post_blocks
        # The same post, seen from two places: each frame keys every block the
        # other's full-resolution post depth lands in, so the count is 2.
        assert bool(torch.isin(post0, keys1).all()) and bool(torch.isin(post1, keys0).all())


# ---------------------------------------------------------------------------
# through the stage
# ---------------------------------------------------------------------------


def _with_far_frame(tmp_path):
    """The ring fixture, with frame 0's fit stretched 8x and its anchor range
    with it: a frame whose depth the solve's own anchors put far outside the
    room, as int4-w102-2's frames 471-474 were."""
    store = _synthetic_world(tmp_path, anchors=True)
    align_p = store.world_dir(WORLD) / "dense" / SESSION / "align.json"
    align = json.loads(align_p.read_text())
    r = align["records"][0]
    r["a"] = 1.0 / 8.0          # kind "depth": z = (pred - b) / a, so 8x farther
    r["z_sparse_min"] = 8.0 * r["z_sparse_min"]
    r["z_sparse_max"] = 8.0 * r["z_sparse_max"]
    align_p.write_text(json.dumps(align))
    return store


class TestThroughTheStage:

    def test_off_nothing_new_is_recorded(self, tmp_path, monkeypatch):
        monkeypatch.delenv(WORLD_SURFACE_BOUND_FIX_ENV, raising=False)
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        man = _manifest(store)
        assert not [k for k in man["outliers"] if k.startswith("blocks_dropped_")]
        assert "bound-points" not in man["params_digest"]

    def test_on_a_walk_within_the_budget_builds_the_same_mesh(self, tmp_path, monkeypatch):
        monkeypatch.delenv(WORLD_SURFACE_BOUND_FIX_ENV, raising=False)
        store = _synthetic_world(tmp_path)
        SP.surfacify(store, WORLD, SESSION, params=_params())
        off = _manifest(store)
        off_mesh = SP.read_surface_level(store, WORLD, SESSION, 0)
        monkeypatch.setenv(WORLD_SURFACE_BOUND_FIX_ENV, "on")
        result = SP.surfacify(store, WORLD, SESSION, params=_params())
        assert result.state == SP.STATE_OK, result.detail
        on = _manifest(store)
        assert on["params_digest"] != off["params_digest"], "on rebuilds"
        assert on["voxel"] == off["voxel"] and on["faces"] == off["faces"]
        assert SP.read_surface_level(store, WORLD, SESSION, 0) == off_mesh
        for k in ("unsupported", "contradicted", "frames_contradicted"):
            assert on["outliers"][f"blocks_dropped_{k}"] == 0

    def _far(self, tmp_path):
        store = _with_far_frame(tmp_path)
        # The consistency field would pull the stretched frame back towards its
        # (unstretched) anchors; these tests are about the budget, so it is off.
        params = dict(depth_consistency=False)
        free = SP.surfacify(store, WORLD, SESSION, params=_params(**params))
        assert free.state == SP.STATE_OK, free.detail
        return store, params, free, _manifest(store)["voxel"]

    def test_a_contradicted_frame_over_the_budget_costs_its_own_blocks_not_the_room(
            self, tmp_path, monkeypatch):
        monkeypatch.delenv(WORLD_SURFACE_BOUND_FIX_ENV, raising=False)
        store, params, free, free_voxel = self._far(tmp_path)
        budget = free.blocks // 2

        old = SP.surfacify(store, WORLD, SESSION, force=True,
                           params=_params(max_blocks=budget, **params))
        assert old.state == SP.STATE_OK, old.detail
        assert json.loads(old.detail)["voxel_coarsened_by"] > 1.0, "the old answer: the room"

        monkeypatch.setenv(WORLD_SURFACE_BOUND_FIX_ENV, "on")
        new = SP.surfacify(store, WORLD, SESSION, force=True,
                           params=_params(max_blocks=budget, **params))
        assert new.state == SP.STATE_OK, new.detail
        man = _manifest(store)
        assert json.loads(new.detail)["voxel_coarsened_by"] == 1.0
        assert man["voxel"] == free_voxel
        assert new.blocks <= budget
        assert man["outliers"]["blocks_dropped_frames_contradicted"] == 1, "the stretched frame only"
        assert man["outliers"]["blocks_dropped_contradicted"] > 0
        assert man["outliers"]["voxel_coarsened_for_block_budget"] == 1.0
        assert man["faces"] > 0

    def test_what_is_left_over_the_budget_is_coarsened_by_what_is_left(self, tmp_path,
                                                                       monkeypatch):
        """The kitchen's path: both answers taken and still over the budget.
        The voxel is coarsened, by less than the old answer at the same budget,
        and the reserved pass fits."""
        monkeypatch.delenv(WORLD_SURFACE_BOUND_FIX_ENV, raising=False)
        store, params, free, free_voxel = self._far(tmp_path)
        monkeypatch.setenv(WORLD_SURFACE_BOUND_FIX_ENV, "on")
        first = SP.surfacify(store, WORLD, SESSION, force=True,
                             params=_params(max_blocks=free.blocks // 2, **params))
        o = _manifest(store)["outliers"]
        left = free.blocks - o["blocks_dropped_unsupported"] - o["blocks_dropped_contradicted"]
        assert first.state == SP.STATE_OK and left > 0
        budget = max(1, left // 3)

        new = SP.surfacify(store, WORLD, SESSION, force=True,
                           params=_params(max_blocks=budget, **params))
        assert new.state == SP.STATE_OK, new.detail
        on = _manifest(store)
        assert new.blocks <= budget
        monkeypatch.delenv(WORLD_SURFACE_BOUND_FIX_ENV, raising=False)
        old = SP.surfacify(store, WORLD, SESSION, force=True,
                           params=_params(max_blocks=budget, **params))
        assert old.state == SP.STATE_OK, old.detail
        assert 1.0 < on["outliers"]["voxel_coarsened_for_block_budget"] \
            < _manifest(store)["outliers"]["voxel_coarsened_for_block_budget"]

    def test_a_far_wall_healthy_frames_see_is_kept_over_the_budget(self, tmp_path, monkeypatch):
        """World Builder must work anywhere: a wall twelve scene depths from
        every camera, measured by several healthy frames, in a walk over the
        budget, is kept -- coarsened with the room if it must be, never cut."""
        slab = ([-1.0, -1.0, 0.25], [1.0, 1.0, 0.35])          # what most frames look at, close
        views = [(np.array([dx, dy, 0.0]), np.array([dx, dy, 1.0]), slab)
                 for dx in (-0.06, 0.0, 0.06) for dy in (-0.04, 0.04)]
        views += [(np.array([dx, 0.0, 0.0]), np.array([dx, 0.0, -1.0]), slab)
                  for dx in (-0.06, -0.02, 0.02, 0.06)]          # four look back at the far wall
        monkeypatch.delenv(WORLD_SURFACE_BOUND_FIX_ENV, raising=False)
        store = _synthetic_world(tmp_path, views=views, anchors=True)
        p = dict(min_support_frames=2, depth_consistency=False)
        free = SP.surfacify(store, WORLD, SESSION, params=_params(**p))
        assert free.state == SP.STATE_OK, free.detail
        md = _manifest(store)["median_scene_depth"]
        assert 6.0 * md < 2.9, "the wall must be beyond the far bound of every camera to test this"

        monkeypatch.setenv(WORLD_SURFACE_BOUND_FIX_ENV, "on")
        new = SP.surfacify(store, WORLD, SESSION, force=True,
                           params=_params(max_blocks=free.blocks // 2, **p))
        assert new.state == SP.STATE_OK, new.detail
        man = _manifest(store)
        assert man["outliers"]["blocks_dropped_frames_contradicted"] == 0
        assert man["outliers"]["blocks_dropped_contradicted"] == 0
        V, F, _C, _N = S.read_mesh_bytes(SP.read_surface_level(store, WORLD, SESSION, 0))
        cz = np.asarray(V)[np.asarray(F)].mean(1)[:, 2]
        assert (cz < -2.5).sum() > 0, "the far wall is in the mesh"
