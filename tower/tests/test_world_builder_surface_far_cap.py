"""The far-depth cap: `TOWER_WORLD_SURFACE_FAR_DEPTH_CAP`, `surface_pipeline.FarDepthCap`.

Far, unreliable depth -- far-looking keyframes the turn bridge adds, depth seen
through doorways -- allocated blocks past `SurfaceParams.max_blocks`, and the
budget then coarsened the WHOLE room's voxel (w9b 2.9x, walk 9.2 1.65x, the
kitchen 3.3x). With the cap on, depth deeper than k x the scene scale
allocates no block. Pinned here, on a synthetic room:

  * spray coarsens the room without the cap, and does not with it;
  * a walk within the budget is built as with the cap off;
  * a real far wall inside the cap is still allocated, block for block, and
    only the samples beyond it lose their blocks;
  * the cap stands down rather than eat the scene, and so can never be what
    empties every frame;
  * off is main: nothing read, nothing recorded, the same keys;
  * the settings parse like their neighbours;
  * the recount script runs.

The end-to-end builds use `_lean` parameters -- no decimation, no hidden-face
raycast -- so they run where open3d does not; neither touches allocation.
"""

from __future__ import annotations

import json
import logging
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.test_world_builder_surface_pipeline import (  # noqa: E402
    SESSION,
    WORLD,
    _manifest,
    _params,
    _synthetic_world,
)
from tower import config as C  # noqa: E402
from tower.world_builder import surface_pipeline as SP  # noqa: E402
from tower.world_builder.dense_pipeline import dense_dir  # noqa: E402
from tower.world_builder.global_solve import load_solution  # noqa: E402
from tower.world_builder.surface import truncation_rel  # noqa: E402

CPU = torch.device("cpu")
N_RING = 24
K = C.WORLD_SURFACE_FAR_DEPTH_CAP_K_DEFAULT
MAX_DROP = C.WORLD_SURFACE_FAR_DEPTH_CAP_MAX_DROP_DEFAULT
CAP_ENVS = (C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, C.WORLD_SURFACE_FAR_DEPTH_CAP_K_ENV,
            C.WORLD_SURFACE_FAR_DEPTH_CAP_MAX_DROP_ENV)


@pytest.fixture(autouse=True)
def _cap_unset(monkeypatch):
    for name in CAP_ENVS:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# a box room ringed by cameras, plus far-looking frames
# ---------------------------------------------------------------------------


def _ring(n, *, phase=0.0, radius=1.2):
    views = []
    for i in range(n):
        a = 2 * np.pi * i / n + phase
        eye = np.array([radius * np.cos(a), 0.0, radius * np.sin(a)])
        views.append((eye, eye * 3.0, None))
    return views


def _scale_frame(store, ki, factor, *, right_half=False):
    """Scale one frame's depth (or only its right half) -- and its anchor
    range, so the per-frame far bound does not clip it: a real frame looking
    through a doorway has far anchors too."""
    root = dense_dir(store, WORLD, SESSION)
    path = root / "work" / "depth" / f"{ki:05d}.npy"
    depth = np.load(path).astype(np.float32)
    if right_half:
        depth[:, depth.shape[1] // 2:] *= factor
    else:
        depth *= factor
    np.save(path, depth.astype(np.float16))
    align = json.loads((root / "align.json").read_text())
    align["records"][ki]["z_sparse_max"] *= factor
    (root / "align.json").write_text(json.dumps(align))


def _world(tmp_path, far=()):
    """24 cameras ringing the room, then one extra frame per entry of `far`,
    its depth scaled by that factor."""
    store = _synthetic_world(tmp_path, views=_ring(N_RING) + _ring(len(far), phase=np.pi / N_RING),
                             anchors=True)
    for j, factor in enumerate(far):
        _scale_frame(store, N_RING + j, factor)
    return store


def _prepare(store, params=None):
    """The frames as `_build` caches them, prepared the way `surfacify` does."""
    params = params or _params()
    solution = load_solution(store, WORLD, SESSION)
    root = dense_dir(store, WORLD, SESSION)
    frames = SP._Frames(json.loads((root / "align.json").read_text()), root / "work",
                        solution, params)
    scale, _source = SP._scene_scale(frames, solution)
    frames.bound = SP._fusion_bound(frames, solution, params, scale)
    cached = [(z.to(torch.float16).cpu(), ok.cpu(), None, R, tt, None)
              for z, ok, _img, R, tt, _w in frames.prepared(params, scale, CPU)]
    return frames, scale, cached


def _allocate(frames, scale, cached, params, cap=None, *, always=False):
    trel = truncation_rel(params, frames.median_held_out)
    voxel = params.voxel_frac * scale
    vol = SP._new_volume(params, trel, voxel, scale, frames.median_held_out, CPU)
    return SP._allocate_blocks(vol, cached, frames.K, params, voxel, trel, scale,
                               frames.median_held_out, CPU, far_cap=cap,
                               far_cap_always=always)


def _keys(allocation) -> set:
    return set(allocation.keys.tolist())


def _lean(**kw):
    return _params(low_weight_hidden_test=False, lod_face_targets=(0,), canonical_level=0,
                   mobile_level=0, mobile_page_bytes=0, **kw)


def _room_blocks(store) -> int:
    frames, scale, cached = _prepare(store)
    return _allocate(frames, scale, cached[:N_RING], _params(max_blocks=0)).keys.numel()


# ---------------------------------------------------------------------------
# what it is for
# ---------------------------------------------------------------------------


class TestFarSprayNoLongerCoarsensTheRoom:

    def test_spray_coarsens_the_room_without_the_cap_and_not_with_it(self, tmp_path):
        store = _world(tmp_path, far=(8.0, 8.0))
        frames, scale, cached = _prepare(store)
        room = _keys(_allocate(frames, scale, cached[:N_RING], _params(max_blocks=0)))
        params = _params(max_blocks=int(len(room) * 1.2))

        off = _allocate(frames, scale, cached, params)
        far = SP.FarDepthCap(k=K, max_drop=MAX_DROP)
        on = _allocate(frames, scale, cached, params, far.measure(cached, scale))

        assert off.coarsened > 1.5, "fixture: two far frames push the room over budget"
        assert on.refusal is None and on.coarsened == 1.0 and on.far_cap_engaged
        assert not far.stood_down and far.frames_wholly_beyond == 2
        record = far.record(on.coarsened)
        assert 0.0 < record["fraction_beyond_cap"] < MAX_DROP
        assert record["cap"] == pytest.approx(K * scale)
        # nothing the room's own frames allocate is lost
        assert room <= _keys(on)

    def test_the_build_records_what_the_cap_did(self, tmp_path, monkeypatch):
        store = _world(tmp_path, far=(8.0, 8.0))
        budget = int(_room_blocks(store) * 1.2)

        off = SP.surfacify(store, WORLD, SESSION, params=_lean(max_blocks=budget))
        assert off.state == SP.STATE_OK, off.detail
        off_outliers = _manifest(store)["outliers"]
        assert off_outliers["voxel_coarsened_for_block_budget"] > 1.5

        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, "on")
        on = SP.surfacify(store, WORLD, SESSION, force=True, params=_lean(max_blocks=budget))
        assert on.state == SP.STATE_OK, on.detail
        man = _manifest(store)
        rec = man["outliers"]["far_depth_cap"]
        assert rec["state"] == "applied"
        assert rec["k"] == K and rec["max_drop_fraction"] == MAX_DROP
        assert rec["cap"] == pytest.approx(K * man["median_scene_depth"])
        assert 0.0 < rec["fraction_beyond_cap"] < MAX_DROP
        assert rec["voxel_coarsened_by"] == 1.0 == man["detail"]["voxel_coarsened_by"]
        assert man["outliers"]["voxel_coarsened_for_block_budget"] == 1.0
        assert man["voxel"] < off.voxel
        assert f"far-depth-cap/1:k={K:g}:max-drop={MAX_DROP:g}" in man["params_digest"]


class TestAWalkWithinBudgetIsUntouched:

    def test_the_cap_does_not_engage_when_the_walk_fits(self, tmp_path):
        store = _world(tmp_path, far=(8.0, 8.0))
        frames, scale, cached = _prepare(store)
        params = _params()                      # the 360k budget: this walk fits
        off = _allocate(frames, scale, cached, params)
        far = SP.FarDepthCap(k=K, max_drop=MAX_DROP)
        on = _allocate(frames, scale, cached, params, far.measure(cached, scale))
        assert not far.stood_down and not on.far_cap_engaged
        assert torch.equal(on.keys, off.keys) and on.voxel == off.voxel
        assert _keys(_allocate(frames, scale, cached, params, far.cap, always=True)) < \
            _keys(off), "fixture: the cap would have dropped blocks"

    def test_the_build_says_it_was_not_needed(self, tmp_path, monkeypatch):
        store = _world(tmp_path, far=(8.0, 8.0))
        assert SP.surfacify(store, WORLD, SESSION, params=_lean()).state == SP.STATE_OK
        off = _manifest(store)
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, "on")
        assert SP.surfacify(store, WORLD, SESSION, force=True,
                            params=_lean()).state == SP.STATE_OK
        on = _manifest(store)
        assert on["outliers"]["far_depth_cap"]["state"] == "not-needed"
        assert on["voxel"] == off["voxel"] and on["faces"] == off["faces"]


class TestARealFarWallIsKept:

    def test_a_wall_at_four_and_a_half_scene_depths_is_allocated_block_for_block(self, tmp_path):
        """9 m away at a 2 m median: a normal room's far wall must survive."""
        store = _world(tmp_path, far=(1.0,))
        frames, scale, cached = _prepare(store)
        z, ok = cached[N_RING][0].float(), cached[N_RING][1]
        _scale_frame(store, N_RING, 4.5 * scale / float(z[ok].max()))
        frames, scale2, cached = _prepare(store)
        assert scale2 == pytest.approx(scale), "fixture: the scene scale is the room's"
        z, ok = cached[N_RING][0].float(), cached[N_RING][1]
        assert float(z[ok].max()) == pytest.approx(4.5 * scale, rel=0.01)

        params = _params(max_blocks=0)
        far = SP.FarDepthCap(k=K, max_drop=MAX_DROP)
        cap = far.measure(cached, scale)
        assert cap is not None and far.samples_beyond == 0
        wall = _keys(_allocate(frames, scale, [cached[N_RING]], params))
        assert wall == _keys(_allocate(frames, scale, [cached[N_RING]], params, cap,
                                       always=True))
        assert _keys(_allocate(frames, scale, cached, params)) == \
            _keys(_allocate(frames, scale, cached, params, cap, always=True))

    def test_only_the_samples_beyond_the_cap_lose_their_blocks(self, tmp_path):
        """A frame half on the room, half through a doorway: the near half is
        allocated block for block, the far half not at all."""
        store = _world(tmp_path, far=(1.0,))
        _scale_frame(store, N_RING, 8.0, right_half=True)
        frames, scale, cached = _prepare(store)
        z, ok = cached[N_RING][0], cached[N_RING][1]
        half = z.shape[1] // 2
        near, far_half = ok.clone(), ok.clone()
        near[:, half:] = False
        far_half[:, :half] = False
        cap = K * scale
        assert bool((z.float()[near] <= cap).all()), "fixture: the left half is the room"
        assert bool((z.float()[far_half] > cap).all()), "fixture: the right half is far"
        assert int(near.sum()) and int(far_half.sum())

        params = _params(max_blocks=0)
        vol = SP._new_volume(params, truncation_rel(params, frames.median_held_out),
                             params.voxel_frac * scale, scale, frames.median_held_out, CPU)
        R, tt = cached[N_RING][3], cached[N_RING][4]
        near_keys = set(vol.blocks_for_depth(z.float(), near, R, tt, frames.K).tolist())
        far_keys = set(vol.blocks_for_depth(z.float(), far_half, R, tt, frames.K).tolist())
        capped = _keys(_allocate(frames, scale, [cached[N_RING]], params, cap, always=True))
        assert capped == near_keys
        assert far_keys - near_keys and not (far_keys - near_keys) & capped

    def test_spray_does_not_take_the_far_wall_with_it(self, tmp_path):
        store = _world(tmp_path, far=(1.0, 8.0, 8.0))
        frames, scale, cached = _prepare(store)
        z, ok = cached[N_RING][0].float(), cached[N_RING][1]
        _scale_frame(store, N_RING, 4.5 * scale / float(z[ok].max()))
        frames, scale, cached = _prepare(store)
        params = _params(max_blocks=0)
        far = SP.FarDepthCap(k=K, max_drop=MAX_DROP)
        capped = _keys(_allocate(frames, scale, cached, params, far.measure(cached, scale),
                                 always=True))
        kept = _keys(_allocate(frames, scale, cached[:N_RING + 1], params))
        spray = _keys(_allocate(frames, scale, cached[N_RING + 1:], params))
        assert kept <= capped
        assert len(spray - kept) > 0 and not (spray - kept) & capped


class TestItStandsDownRatherThanEatTheScene:

    def test_a_cap_past_the_drop_limit_allocates_as_off_and_says_so(self, tmp_path):
        """A third of the frames far is not a far tail; the scene is out there."""
        store = _world(tmp_path, far=(8.0,) * 12)
        frames, scale, cached = _prepare(store)
        far = SP.FarDepthCap(k=K, max_drop=MAX_DROP)
        assert far.measure(cached, scale) is None
        assert far.stood_down
        assert far.record(1.0)["state"] == "stood-down"
        assert far.record(1.0)["fraction_beyond_cap"] > MAX_DROP

    def test_a_cap_that_would_empty_every_frame_stands_down(self, tmp_path):
        store = _world(tmp_path)
        frames, scale, cached = _prepare(store)
        far = SP.FarDepthCap(k=0.1, max_drop=1.0)
        assert far.measure(cached, scale) is None
        assert far.stood_down and far.frames_wholly_beyond == far.frames_measured == len(cached)

    def test_so_the_build_never_loses_every_frame_to_it(self, tmp_path, monkeypatch):
        store = _world(tmp_path)
        monkeypatch.setattr(SP, "_far_depth_cap_setting", lambda: {"k": 0.1, "max_drop": 1.0})
        result = SP.surfacify(store, WORLD, SESSION, params=_lean())
        assert result.state == SP.STATE_OK, result.detail
        assert _manifest(store)["outliers"]["far_depth_cap"]["state"] == "stood-down"


class TestOffIsMain:

    def test_off_allocates_the_keys_main_computes(self, tmp_path):
        """`_allocate_blocks` with no cap against main's expression, inline."""
        store = _world(tmp_path, far=(8.0, 8.0))
        frames, scale, cached = _prepare(store)
        params = _params(max_blocks=0)
        trel = truncation_rel(params, frames.median_held_out)
        vol = SP._new_volume(params, trel, params.voxel_frac * scale, scale,
                             frames.median_held_out, CPU)
        main = torch.unique(torch.cat([
            vol.blocks_for_depth(z.to(CPU).float(), ok.to(CPU), R, tt, frames.K)
            for z, ok, _img, R, tt, _w in cached]))
        off = _allocate(frames, scale, cached, params)
        assert torch.equal(off.keys, main)
        assert off.voxel == params.voxel_frac * scale and off.coarsened == 1.0

    @pytest.mark.parametrize("value", [None, "", "off", "0", "false", "no"])
    def test_off_records_nothing_and_leaves_the_digest_alone(self, tmp_path, monkeypatch,
                                                             value):
        if value is not None:
            monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, value)
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_K_ENV, "3")
        store = _world(tmp_path, far=(8.0,))
        assert SP.surfacify(store, WORLD, SESSION, params=_lean()).state == SP.STATE_OK
        man = _manifest(store)
        assert "far_depth_cap" not in man["outliers"]
        assert "far-depth-cap" not in man["params_digest"]


# ---------------------------------------------------------------------------
# the settings
# ---------------------------------------------------------------------------


class TestTheSettings:

    def test_unset_is_off(self):
        assert C.world_surface_far_depth_cap_setting() is None

    @pytest.mark.parametrize("value", ["on", "1", "true", "yes", "ON"])
    def test_on_takes_the_defaults(self, monkeypatch, value):
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, value)
        assert C.world_surface_far_depth_cap_setting() == {"k": 6.0, "max_drop": 0.25}

    def test_the_knobs_are_read(self, monkeypatch):
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, "on")
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_K_ENV, "4")
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_MAX_DROP_ENV, "1")
        assert C.world_surface_far_depth_cap_setting() == {"k": 4.0, "max_drop": 1.0}

    def test_a_typo_is_off_and_logged(self, monkeypatch, caplog):
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, "maybe")
        with caplog.at_level(logging.WARNING, logger="tower.config"):
            assert C.world_surface_far_depth_cap_setting() is None
        assert C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV in caplog.text

    @pytest.mark.parametrize("value", ["", "off", "0", "false", "no"])
    def test_off_is_silent(self, monkeypatch, caplog, value):
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, value)
        with caplog.at_level(logging.WARNING, logger="tower.config"):
            assert C.world_surface_far_depth_cap_setting() is None
        assert not caplog.records

    @pytest.mark.parametrize("k", ["1.9", "51", "abc", "nan", "inf", "-6"])
    def test_a_bad_k_is_the_default_and_logged(self, monkeypatch, caplog, k):
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, "on")
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_K_ENV, k)
        with caplog.at_level(logging.WARNING, logger="tower.config"):
            assert C.world_surface_far_depth_cap_setting()["k"] == 6.0
        assert C.WORLD_SURFACE_FAR_DEPTH_CAP_K_ENV in caplog.text

    @pytest.mark.parametrize("drop", ["0", "1.5", "-0.1", "x"])
    def test_a_bad_max_drop_is_the_default_and_logged(self, monkeypatch, caplog, drop):
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, "on")
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_MAX_DROP_ENV, drop)
        with caplog.at_level(logging.WARNING, logger="tower.config"):
            assert C.world_surface_far_depth_cap_setting()["max_drop"] == 0.25
        assert C.WORLD_SURFACE_FAR_DEPTH_CAP_MAX_DROP_ENV in caplog.text

    def test_the_knobs_are_not_read_with_the_cap_off(self, monkeypatch, caplog):
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_K_ENV, "garbage")
        with caplog.at_level(logging.WARNING, logger="tower.config"):
            assert C.world_surface_far_depth_cap_setting() is None
        assert not caplog.records


# ---------------------------------------------------------------------------
# the recount script
# ---------------------------------------------------------------------------


class TestTheRecountScript:

    def test_it_recounts_each_k_with_the_builds_arithmetic(self, tmp_path, capsys):
        from scripts import surface_far_cap_recount as R

        _world(tmp_path, far=(8.0, 8.0))
        assert R.main(["--root", str(tmp_path), "--world", WORLD, "--k", "4,6,off",
                       "--json"]) == 0
        result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        rows = {r["k"]: r for r in result["rows"]}
        assert set(rows) == {4.0, 6.0, "off"}
        assert result["frames"] == N_RING + 2
        assert result["consistency_field"] == "plain affine"
        for k in (4.0, 6.0):
            assert rows[k]["cap"] == pytest.approx(k * result["scene_scale"])
            assert 0.0 < rows[k]["fraction_beyond_cap"] < MAX_DROP
            assert rows[k]["frames_wholly_beyond_cap"] == 2 and not rows[k]["stood_down"]
            assert rows[k]["blocks_at_voxel"] < rows["off"]["blocks_at_voxel"]
            # the final preset's budget holds this walk: a build would not engage it
            assert rows[k]["engaged"] is False and rows[k]["voxel_coarsened_by"] == 1.0
        assert "cap" not in rows["off"]

    def test_the_off_row_is_what_a_build_coarsens_by(self, tmp_path, monkeypatch):
        """The recount's arithmetic is the build's: its `off` row's coarsening
        is the manifest's, on the same frames and budget."""
        from scripts import surface_far_cap_recount as R

        store = _world(tmp_path, far=(8.0, 8.0))
        budget = int(_room_blocks(store) * 1.2)
        params = _lean(max_blocks=budget, depth_consistency=False, transient_detector="off")
        assert SP.surfacify(store, WORLD, SESSION, params=params).state == SP.STATE_OK
        rows = R.recount(store, WORLD, SESSION, [None, K], params=params,
                         max_drop=MAX_DROP)["rows"]
        man = _manifest(store)
        assert rows[0]["voxel_coarsened_by"] == man["detail"]["voxel_coarsened_by"] > 1.5
        assert rows[1]["voxel_coarsened_by"] == 1.0 and rows[1]["engaged"] is True

    def test_a_pruned_session_is_reported_not_crashed_on(self, tmp_path, capsys):
        from scripts import surface_far_cap_recount as R

        store = _world(tmp_path)
        shutil.rmtree(dense_dir(store, WORLD, SESSION) / "work")
        assert R.main(["--root", str(tmp_path), "--world", WORLD]) == 1
        assert "pruned" in capsys.readouterr().out
