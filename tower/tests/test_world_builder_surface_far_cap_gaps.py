"""SURFACE_FAR_DEPTH_CAP: the gaps the cand5 cross-review left (CX-CAND5-TESTGAP).

`test_world_builder_surface_far_cap.py` pins what the cap is for. This adds:

  * OFF == MAIN through the whole budget loop -- not only the first allocation:
    a reference copy of main's (26e394e) allocation loop, run side by side with
    `_allocate_blocks(far_cap=None)` on the same frames, must agree on keys (and
    their order), voxel, coarsening, key coarsening and refusal -- on real fused
    frames (fits / coarsens for the budget) and on a scripted volume (key-range
    coarsening, key-range refusal past `max_key_coarsening`, budget refusal);
  * under budget the armed cap is `not-needed` and allocates exactly as off;
  * over budget with far spray: `applied`, and the budget is STILL ENFORCED --
    when the capped walk is still over budget it coarsens to within it, and it
    still refuses past the attempts; the record fields are right;
  * the stand-down boundary (> max_drop stands down, == max_drop does not);
  * edge cases: empty cache, no valid sample, NaN/zero depth, zero / NaN /
    negative scene scale, a single frame;
  * the key-range interaction (review HIGH-1): the CURRENT behaviour is
    documented by a passing test, and the behaviour the review asks for is a
    strict xfail, so a fix flips it and forces this file to be updated.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tests.test_world_builder_surface_far_cap import (  # noqa: E402
    CAP_ENVS,
    CPU,
    K,
    MAX_DROP,
    _allocate,
    _lean,
    _prepare,
    _room_blocks,
    _world,
)
from tests.test_world_builder_surface_pipeline import (  # noqa: E402
    SESSION,
    WORLD,
    _manifest,
    _params,
)
from tower import config as C  # noqa: E402
from tower.world_builder import surface_pipeline as SP  # noqa: E402
from tower.world_builder.surface import (  # noqa: E402
    SurfaceUnavailable,
    key_range_refusal,
    truncation_rel,
)


@pytest.fixture(autouse=True)
def _cap_unset(monkeypatch):
    for name in CAP_ENVS + ("TOWER_WORLD_POSE_SANITY", "TOWER_WORLD_MERGE_AFTER_STOP"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# main's allocation loop, verbatim in its decisions (26e394e, `_build`)
# ---------------------------------------------------------------------------


def _main_allocate(vol, cached, Kmat, params, voxel, new_volume, device=CPU):
    """Main's loop with the volume constructor passed in. Returns what
    `_Allocation` carries: (keys, voxel, coarsened, key_coarsened, refusal)."""
    coarsened = 1.0
    key_coarsened = 1.0
    attempts = 12
    cap = params.max_key_coarsening
    allk = None
    for attempt in range(attempts):
        try:
            keys = [vol.blocks_for_depth(z.to(device).float(), ok.to(device), R, tt, Kmat)
                    for z, ok, _img, R, tt, _w in cached]
            allk = torch.unique(torch.cat(keys))
            del keys
        except SurfaceUnavailable as exc:
            over = exc.key_overshoot
            factor = max(1.1, float(over) * 1.05) if over is not None else 0.0
            if over is None or attempt == attempts - 1 or (
                    cap > 0 and key_coarsened * factor > cap):
                reason = exc.reason
                if over is not None:
                    reason += (
                        f"; bringing it inside needs the voxel coarsened "
                        f"x{key_coarsened * factor:.2f}, past the "
                        f"x{cap:g} a surface may be coarsened for the key "
                        "range (max_key_coarsening). A scene that wide after "
                        "the pose gate and the fusion bound is a solve that "
                        "came apart, and a coarser room would be published as "
                        "though it were the room"
                        if cap > 0 and factor and key_coarsened * factor > cap else
                        f"; coarsening the voxel x{key_coarsened:.2f} over "
                        f"{attempt + 1} attempts did not bring it inside")
                return None, voxel, coarsened, key_coarsened, reason
            coarsened *= factor
            key_coarsened *= factor
            voxel *= factor
            vol = new_volume(voxel)
            continue
        if params.max_blocks <= 0 or allk.numel() <= params.max_blocks:
            break
        if attempt == attempts - 1:
            reason = (f"the walk still needs {allk.numel()} blocks after "
                      f"coarsening the voxel x{coarsened:.2f} over {attempts} "
                      f"attempts, over the block budget of {params.max_blocks}; "
                      "not built")
            return None, voxel, coarsened, key_coarsened, reason
        factor = max(1.1, math.sqrt(allk.numel() / params.max_blocks) * 1.05)
        coarsened *= factor
        voxel *= factor
        vol = new_volume(voxel)
    return allk, voxel, coarsened, key_coarsened, None


def _same(alloc, ref):
    keys, voxel, coarsened, key_coarsened, refusal = ref
    assert alloc.refusal == refusal
    assert alloc.voxel == voxel
    assert alloc.coarsened == coarsened and alloc.key_coarsened == key_coarsened
    assert not alloc.far_cap_engaged
    if keys is None:
        assert alloc.keys is None
    else:
        assert torch.equal(alloc.keys, keys)          # same keys, same order


# ---------------------------------------------------------------------------
# a scripted volume: blocks are depth bins; the key range is a bin limit
# ---------------------------------------------------------------------------

KEY_LIMIT = 100000      # a bin index at or past this cannot be keyed (1000 m at 0.01)
FRAME_STRIDE = 10 ** 7    # keeps each frame's bins distinct


class BinVolume:
    """`blocks_for_depth` -> one key per depth bin of width `voxel`, per frame.
    A bin past KEY_LIMIT raises the real `key_range_refusal` with its overshoot."""

    def __init__(self, voxel):
        self.voxel = float(voxel)

    def blocks_for_depth(self, depth, valid, R, t, K):
        z = depth[valid]
        if z.numel() == 0:
            return torch.zeros(0, dtype=torch.int64)
        bins = torch.floor(z / self.voxel).to(torch.int64)
        top = int(bins.max())
        if top >= KEY_LIMIT:
            raise key_range_refusal(top / float(KEY_LIMIT))
        return torch.unique(bins + int(R) * FRAME_STRIDE)


def _frame(index, depths, ok=None):
    z = torch.tensor(depths, dtype=torch.float32).reshape(1, -1)
    okm = torch.ones_like(z, dtype=torch.bool) if ok is None else \
        torch.tensor(ok, dtype=torch.bool).reshape(1, -1)
    return (z.to(torch.float16), okm, None, index, None, None)


def _room(n_frames=10, n=200, lo=1.0, hi=3.0):
    step = (hi - lo) / (n - 1)
    return [_frame(i, [lo + j * step for j in range(n)]) for i in range(n_frames)]


def _bin_allocate(monkeypatch, cached, params, voxel=0.01, cap=None, always=False):
    monkeypatch.setattr(SP, "_new_volume",
                        lambda params, trel, vx, md, mho, dev: BinVolume(vx))
    return SP._allocate_blocks(BinVolume(voxel), cached, None, params, voxel, 0.0, 2.0,
                               None, CPU, far_cap=cap, far_cap_always=always)


def _bin_main(cached, params, voxel=0.01):
    return _main_allocate(BinVolume(voxel), cached, None, params, voxel, BinVolume)


# ---------------------------------------------------------------------------
# off == main
# ---------------------------------------------------------------------------


class TestOffIsMainThroughTheWholeLoop:

    @pytest.mark.parametrize("budget_of_room", [0, 2.0, 1.2, 0.5],
                             ids=["no-budget", "fits", "coarsens-1.2", "coarsens-0.5"])
    def test_real_frames(self, tmp_path, budget_of_room):
        store = _world(tmp_path, far=(8.0, 8.0))
        frames, scale, cached = _prepare(store)
        budget = int(_room_blocks(store) * budget_of_room)
        params = _params(max_blocks=budget)
        trel = truncation_rel(params, frames.median_held_out)

        def main_volume(vx):          # main's own constructor expression
            if trel > 0:
                band = params.trunc_voxels * vx
            else:
                band = SP.truncation_for(params, vx, scale, frames.median_held_out)
            return SP.SurfaceVolume(vx, band, device=CPU, trunc_rel=trel,
                                    trunc_max=params.trunc_max_voxels * vx)

        voxel = params.voxel_frac * scale
        ref = _main_allocate(main_volume(voxel), cached, frames.K, params, voxel, main_volume)
        off = _allocate(frames, scale, cached, params)
        _same(off, ref)
        if budget_of_room and budget_of_room < 1.5:
            assert off.coarsened > 1.0, "fixture: this budget coarsens"
            assert off.keys.numel() <= budget

    @pytest.mark.parametrize("case", ["fits", "budget-coarsens", "budget-refuses",
                                      "key-coarsens", "key-refuses-past-cap"])
    def test_scripted_volume(self, monkeypatch, case):
        room = _room()
        cached, params = room, _params(max_blocks=0)
        if case == "fits":
            params = _params(max_blocks=10 ** 6)
        elif case == "budget-coarsens":
            params = _params(max_blocks=1000)
        elif case == "budget-refuses":
            params = _params(max_blocks=3)
        elif case == "key-coarsens":
            cached = room + [_frame(99, [1.0, 2.0, 1500.0])]        # x1.5 past the key range
        else:
            cached = room + [_frame(99, [1.0, 2.0, 50000.0])]       # x50, past the x4 cap
        ref = _bin_main(cached, params)
        off = _bin_allocate(monkeypatch, cached, params)
        _same(off, ref)
        if case == "budget-refuses" or case == "key-refuses-past-cap":
            assert off.refusal
        if case == "key-coarsens":
            assert off.key_coarsened > 1.0 and off.refusal is None


# ---------------------------------------------------------------------------
# under budget, over budget, the budget still enforced
# ---------------------------------------------------------------------------


class TestUnderAndOverBudget:

    def test_under_budget_the_armed_cap_is_not_needed_and_allocates_as_off(self, monkeypatch):
        cached = _room() + [_frame(50, [5.0] * 20 + [40.0] * 5)]
        params = _params(max_blocks=10 ** 6)
        far = SP.FarDepthCap(k=K, max_drop=MAX_DROP)
        cap = far.measure(cached, 2.0)
        assert cap == pytest.approx(12.0) and far.samples_beyond == 5
        on = _bin_allocate(monkeypatch, cached, params, cap=cap)
        off = _bin_allocate(monkeypatch, cached, params)
        assert not on.far_cap_engaged
        assert torch.equal(on.keys, off.keys) and on.voxel == off.voxel
        assert on.coarsened == off.coarsened == 1.0
        far.engaged = on.far_cap_engaged
        assert far.record(on.coarsened)["state"] == "not-needed"

    def _spray(self):
        # A room of 10 frames x 200 bins, plus one frame spraying 400 far bins.
        far = [12.5 + 0.01 * j for j in range(400)]
        return _room() + [_frame(50, [1.5] * 1300 + far)]

    def test_over_budget_with_spray_applies_and_keeps_the_room_voxel(self, monkeypatch):
        cached = self._spray()
        params = _params(max_blocks=2100)
        far = SP.FarDepthCap(k=K, max_drop=MAX_DROP)
        cap = far.measure(cached, 2.0)
        assert cap is not None
        off = _bin_allocate(monkeypatch, cached, params)
        on = _bin_allocate(monkeypatch, cached, params, cap=cap)
        assert off.coarsened > 1.0, "fixture: the spray pushes the room over budget"
        assert on.far_cap_engaged and on.coarsened == 1.0 and on.refusal is None
        assert on.keys.numel() <= params.max_blocks
        far.engaged = True
        rec = far.record(on.coarsened)
        assert rec["state"] == "applied" and rec["rule"] == SP.FAR_DEPTH_CAP_ID
        assert rec["k"] == K and rec["scene_scale"] == 2.0 and rec["cap"] == 12.0
        assert rec["max_drop_fraction"] == MAX_DROP
        assert rec["samples_valid"] == 2000 + 1700 and rec["samples_beyond_cap"] == 400
        assert rec["fraction_beyond_cap"] == round(400 / 3700, 6)
        assert rec["frames_measured"] == 11 and rec["frames_wholly_beyond_cap"] == 0
        assert rec["voxel_coarsened_by"] == 1.0

    def test_over_budget_even_capped_still_coarsens_to_within_the_budget(self, monkeypatch):
        cached = self._spray()
        params = _params(max_blocks=1500)                 # the room alone is ~2000 bins
        cap = SP.FarDepthCap(k=K, max_drop=MAX_DROP).measure(cached, 2.0)
        off = _bin_allocate(monkeypatch, cached, params)
        on = _bin_allocate(monkeypatch, cached, params, cap=cap)
        assert on.far_cap_engaged and on.refusal is None
        assert 1.0 < on.coarsened < off.coarsened
        assert on.keys.numel() <= params.max_blocks
        # and every coarsening pass after engaging stayed capped: no far bin survived
        assert int((on.keys % FRAME_STRIDE).max()) * on.voxel < cap

    def test_the_budget_refusal_still_happens_with_the_cap_engaged(self, monkeypatch):
        cached = self._spray()
        params = _params(max_blocks=3)
        cap = SP.FarDepthCap(k=K, max_drop=MAX_DROP).measure(cached, 2.0)
        on = _bin_allocate(monkeypatch, cached, params, cap=cap)
        assert on.far_cap_engaged and on.keys is None
        assert on.refusal and "over the block budget of 3" in on.refusal

    def test_the_build_over_budget_even_capped(self, tmp_path, monkeypatch):
        """End to end on real frames: a budget below the room's own blocks. The
        cap applies, the voxel still coarsens for the budget, and the record says
        both -- less coarsening than off, more than none."""
        store = _world(tmp_path, far=(8.0, 8.0))
        budget = int(_room_blocks(store) * 0.7)
        off = SP.surfacify(store, WORLD, SESSION, params=_lean(max_blocks=budget))
        assert off.state == SP.STATE_OK, off.detail
        off_c = _manifest(store)["detail"]["voxel_coarsened_by"]
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, "on")
        on = SP.surfacify(store, WORLD, SESSION, force=True, params=_lean(max_blocks=budget))
        assert on.state == SP.STATE_OK, on.detail
        man = _manifest(store)
        rec = man["outliers"]["far_depth_cap"]
        assert rec["state"] == "applied"
        assert 1.0 < rec["voxel_coarsened_by"] == man["detail"]["voxel_coarsened_by"] < off_c


# ---------------------------------------------------------------------------
# measure(): the stand-down boundary and the edge cases
# ---------------------------------------------------------------------------


class TestMeasureEdges:

    def _cap(self, max_drop=MAX_DROP):
        return SP.FarDepthCap(k=K, max_drop=max_drop)

    def test_exactly_max_drop_beyond_is_armed_and_one_more_stands_down(self):
        near = [1.0] * 75
        at = self._cap()
        assert at.measure([_frame(0, near + [100.0] * 25)], 2.0) == 12.0   # 25% == 0.25
        assert not at.stood_down
        over = self._cap()
        assert over.measure([_frame(0, near + [100.0] * 26)], 2.0) is None  # 25.7%
        assert over.stood_down and over.record(1.0)["state"] == "stood-down"

    def test_a_sample_exactly_at_the_cap_is_inside(self):
        far = self._cap()
        assert far.measure([_frame(0, [1.0, 12.0])], 2.0) == 12.0
        assert far.samples_beyond == 0

    def test_empty_cache_stands_down(self):
        far = self._cap()
        assert far.measure([], 2.0) is None
        assert far.stood_down and far.samples_valid == 0
        rec = far.record(1.0)
        assert rec["fraction_beyond_cap"] == 0.0 and rec["frames_measured"] == 0

    def test_no_valid_sample_stands_down_and_measures_no_frame(self):
        far = self._cap()
        assert far.measure([_frame(0, [1.0, 50.0], ok=[False, False])], 2.0) is None
        assert far.stood_down and far.frames_measured == 0

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")], ids=["zero", "neg", "nan"])
    def test_a_degenerate_scene_scale_stands_down(self, scale):
        far = self._cap()
        assert far.measure(_room(n_frames=2, n=10), scale) is None
        assert far.stood_down
        assert far.record(1.0)["state"] == "stood-down"

    def test_masked_nan_depth_is_ignored_and_zero_depth_is_inside(self):
        far = self._cap()
        cached = [_frame(0, [float("nan"), 0.0, 1.0, 50.0], ok=[False, True, True, True])]
        assert far.measure(cached, 2.0) is None          # 1 of 3 beyond > 0.25
        assert far.samples_valid == 3 and far.samples_beyond == 1
        far = self._cap(max_drop=0.5)
        assert far.measure(cached, 2.0) == 12.0

    def test_unmasked_nan_depth_counts_as_beyond(self):
        """CURRENT behaviour: `within_far_cap` is `z <= cap`, so a NaN the mask let
        through is counted beyond (and would allocate nothing when engaged).
        `depth_validity` is what keeps NaN out of `ok` upstream."""
        far = self._cap(max_drop=1.0)
        far.measure([_frame(0, [1.0, 1.0, float("nan")])], 2.0)
        assert far.samples_beyond == 1

    def test_float16_depth_far_past_the_cap_is_beyond_not_overflowed(self):
        far = self._cap(max_drop=1.0)
        far.measure([_frame(0, [1.0, 60000.0, float("inf")])], 2.0)
        assert far.samples_valid == 3 and far.samples_beyond == 2

    def test_a_single_frame(self):
        part = self._cap()
        assert part.measure([_frame(0, [1.0] * 9 + [99.0])], 2.0) == 12.0
        assert part.frames_measured == 1 and part.frames_wholly_beyond == 0
        whole = self._cap(max_drop=1.0)
        assert whole.measure([_frame(0, [99.0] * 10)], 2.0) is None
        assert whole.stood_down and whole.frames_wholly_beyond == 1

    def test_a_stood_down_cap_allocates_exactly_as_off(self, monkeypatch):
        cached = _room() + [_frame(50, [40.0] * 2000)]
        params = _params(max_blocks=1000)
        far = self._cap()
        assert far.measure(cached, 2.0) is None
        _same(_bin_allocate(monkeypatch, cached, params, cap=None), _bin_main(cached, params))


# ---------------------------------------------------------------------------
# the key-range branch (review HIGH-1)
# ---------------------------------------------------------------------------


def _key_range_walk():
    """A room that fits the budget, plus a few far samples (x2 past the key range at the
    initial voxel) that trip the key range. Capped, every bin is keyable."""
    return _room() + [_frame(50, [1.5] * 200 + [2000.0] * 5)]


class TestKeyRangeInteraction:

    def test_current_behaviour_the_cap_does_not_rescue_a_key_range_overshoot(
            self, monkeypatch):
        cached = _key_range_walk()
        params = _params(max_blocks=10 ** 6)
        far = SP.FarDepthCap(k=K, max_drop=MAX_DROP)
        cap = far.measure(cached, 2.0)
        assert cap == 12.0 and not far.stood_down and far.samples_beyond == 5
        # capped from the first pass, the walk is keyable at the room's voxel
        always = _bin_allocate(monkeypatch, cached, params, cap=cap, always=True)
        assert always.coarsened == 1.0 and always.refusal is None
        # but the build's path keys UNCAPPED first; the refusal goes to the
        # key-range coarsening branch before the over-budget check can arm the cap
        on = _bin_allocate(monkeypatch, cached, params, cap=cap)
        off = _bin_allocate(monkeypatch, cached, params)
        assert on.key_coarsened > 1.0 and on.coarsened == off.coarsened
        assert not on.far_cap_engaged
        assert torch.equal(on.keys, off.keys)

    def test_current_behaviour_past_max_key_coarsening_the_capped_walk_is_refused(
            self, monkeypatch):
        cached = _room() + [_frame(50, [1.5] * 200 + [60000.0] * 5)]  # x60 overshoot
        params = _params(max_blocks=10 ** 6)
        cap = SP.FarDepthCap(k=K, max_drop=MAX_DROP).measure(cached, 2.0)
        assert cap == 12.0
        on = _bin_allocate(monkeypatch, cached, params, cap=cap)
        assert on.refusal and "max_key_coarsening" in on.refusal and on.keys is None

    @pytest.mark.xfail(strict=True, reason=(
        "review HIGH-1: _allocate_blocks keys uncapped first, so a far tail that trips "
        "the key range coarsens (or refuses) the room before the armed cap can exclude it"))
    def test_wanted_an_armed_cap_keeps_the_room_voxel_through_a_key_range_overshoot(
            self, monkeypatch):
        cached = _key_range_walk()
        params = _params(max_blocks=10 ** 6)
        cap = SP.FarDepthCap(k=K, max_drop=MAX_DROP).measure(cached, 2.0)
        on = _bin_allocate(monkeypatch, cached, params, cap=cap)
        assert on.refusal is None and on.coarsened == 1.0 and on.key_coarsened == 1.0
