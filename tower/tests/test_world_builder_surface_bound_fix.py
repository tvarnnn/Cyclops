"""One stray pose or a spread of far depth must not cut a whole room's resolution.

WHY THIS FILE EXISTS

Replay int4-w102-2 of walk 10.2 (2026-11-03, FIX-BOUND) built its room at a voxel
3.54x coarser than every other replay of the same walk -- 618 K faces instead of
4.1-4.7 M, the worst haze of any run. Nothing was wrong with the room. Seven of
its 959 fused frames were: four whose every sparse anchor the solver had placed
17-99 units out (in a room of radius ~6), and three cameras the solve put 19-24
units out, just inside the pose gate. Each keyed 0.36-1.06 M blocks against a
median frame's 8.7 K, the walk needed 4.09 M blocks against the field's budget of
360 K, and the budget's one answer was to coarsen the voxel of the whole room.
Separately, the fusion bound had been widened from 74 to 89 units by sparse
points of OTHER solver components, which the surface never fuses.

`TOWER_WORLD_SURFACE_BOUND_FIX` (default off) changes both:

  * the bound's point envelope is built from the fused component's points only
    (`SurfaceParams.bound_from_fused_component`);
  * a walk over the budget first loses the blocks no kept face can come from,
    then the far periphery, and only then a coarser voxel
    (`SurfaceParams.budget_spares_the_room`, `surface_pipeline._spare_the_room`).

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
        assert S.SurfaceParams(bound_from_fused_component=False,
                               budget_spares_the_room=False).digest_fields() == off
        assert not [f for f in off if isinstance(f, tuple) and f and f[0] in ("bound-points", "budget")]

    def test_on_the_cache_identity_says_so_and_only_appends(self):
        off = S.SurfaceParams().digest_fields()
        on = S.SurfaceParams(bound_from_fused_component=True,
                             budget_spares_the_room=True).digest_fields()
        assert on[:len(off)] == off
        assert on[len(off):] == (("bound-points", "fused-component"),
                                 ("budget", "unsupported-then-periphery-then-voxel"))


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


class TestTheBudgetSparesTheRoom:
    VOXEL = 0.1          # a block is 0.8 units

    def _params(self, **kw):
        base = dict(max_blocks=4, min_support_frames=2, max_depth_frac=6.0)
        base.update(kw)
        return S.SurfaceParams(**base)

    def _spare(self, coords, counts, centres, **kw):
        keys = _keys(coords)
        order = torch.argsort(keys)
        return SP._spare_the_room(keys[order], torch.as_tensor(counts)[order], voxel=self.VOXEL,
                                  centres=torch.as_tensor(np.asarray(centres, np.float32)),
                                  params=self._params(**kw), median_depth=1.0)

    def test_blocks_fewer_frames_key_than_a_face_needs_go_first(self):
        coords = [[0, 0, i] for i in range(6)]
        kept, dropped = self._spare(coords, [1, 2, 5, 1, 3, 2], [[0, 0, 0]])
        assert dropped == {"unsupported": 2, "periphery": 0}
        assert sorted(kept.tolist()) == sorted(_keys([[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 5]]).tolist())

    def test_the_periphery_goes_only_while_the_budget_still_binds(self):
        # Six supported blocks against a budget of four: the two far ones go.
        near = [[0, 0, i] for i in range(4)]
        far = [[0, 0, 100], [0, 0, 101]]           # 80 units out; reach is 6
        kept, dropped = self._spare(near + far, [3] * 6, [[0, 0, 0]])
        assert dropped == {"unsupported": 0, "periphery": 2}
        assert sorted(kept.tolist()) == sorted(_keys(near).tolist())

    def test_a_walk_within_the_budget_after_the_first_answer_keeps_its_periphery(self):
        near = [[0, 0, i] for i in range(3)]
        far = [[0, 0, 100]]
        junk = [[5, 5, 5], [6, 6, 6]]
        kept, dropped = self._spare(near + far + junk, [3, 3, 3, 3, 1, 1], [[0, 0, 0]])
        assert dropped == {"unsupported": 2, "periphery": 0}
        assert kept.numel() == 4

    def test_near_ANY_camera_is_not_periphery(self):
        """The far periphery is far from every place the wearer stood, not
        from the middle of the walk: a long walk's far end is near its own
        cameras."""
        blocks = [[0, 0, i] for i in range(4)] + [[0, 0, 100], [0, 0, 101]]
        kept, dropped = self._spare(blocks, [3] * 6, [[0, 0, 0], [0, 0, 80.5]])
        assert dropped == {"unsupported": 0, "periphery": 0}
        assert kept.numel() == 6

    def test_a_count_of_one_drops_nothing(self):
        coords = [[0, 0, i] for i in range(6)]
        kept, dropped = self._spare(coords, [1] * 6, [[0, 0, 0]], min_support_frames=1)
        assert dropped["unsupported"] == 0 and kept.numel() == 6

    def test_the_reach_is_the_scene_relative_far_bound(self):
        """A block whose centre is within `max_depth_frac` x the scene scale
        (plus half a block diagonal) of a camera is kept; one beyond is not."""
        # Block i along z has its centre at (i + 0.5) * 0.8: block 7 at 6.0.
        blocks = [[0, 0, i] for i in range(12)]
        kept, dropped = self._spare(blocks, [3] * 12, [[0.4, 0.4, 0.0]], max_blocks=1)
        lim = 6.0 + 8 * self.VOXEL * 0.8660254
        centres = (np.arange(12) + 0.5) * 0.8
        expect = int(np.sum(np.sqrt(centres ** 2) <= lim))
        assert kept.numel() == expect
        assert dropped["periphery"] == 12 - expect


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
    r["a"] = 8.0
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
        assert on["outliers"]["blocks_dropped_unsupported"] == 0
        assert on["outliers"]["blocks_dropped_periphery"] == 0

    def test_a_far_frame_over_the_budget_costs_the_periphery_not_the_room(self, tmp_path,
                                                                          monkeypatch):
        monkeypatch.delenv(WORLD_SURFACE_BOUND_FIX_ENV, raising=False)
        store = _with_far_frame(tmp_path)
        # The consistency field would pull the stretched frame back towards its
        # (unstretched) anchors; this test is about the budget, so it is off.
        params = dict(depth_consistency=False)
        free = SP.surfacify(store, WORLD, SESSION, params=_params(**params))
        assert free.state == SP.STATE_OK, free.detail
        free_voxel = _manifest(store)["voxel"]
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
        assert man["outliers"]["blocks_dropped_periphery"] > 0
        assert man["outliers"]["voxel_coarsened_for_block_budget"] == 1.0
        assert man["faces"] > 0
