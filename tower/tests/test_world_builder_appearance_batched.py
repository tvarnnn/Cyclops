"""The batched appearance projection (`TOWER_WORLD_APPEARANCE_BATCHED`)
against the per-group loops it replaces, inside one process.

The batched path changes how many views one projection holds, never what is
computed per element: each view group's matrix product is the call the loop
makes, and everything after it is elementwise or a per-element sample. So the
ordered exposure observation stream, every keyframe's colour-vote stack and,
given the same gains, every vote grid must be byte-identical. The only part of
the stage that is not reproducible on CUDA is the exposure solve's
`index_add_` (atomic float adds), which this change does not touch: with
deterministic algorithms it is byte-identical too, and without them the gains
of the two paths may differ only by that solve's own run-to-run spread."""

import numpy as np
import pytest
import torch

from tests.test_world_builder_appearance_groups import _assert_same, _host, _run_traced, _trace_at
from tower.config import world_appearance_batched_setting
from tower.world_builder import appearance as A

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]

# Gains of the batched path against the per-group path when the exposure solve
# is NOT made deterministic (CUDA atomics). The two paths hand the solve the
# same bytes, so they may differ only by its own run-to-run spread. The bound is
# checked on the small synthetic scene below (49 views), whose solve noise is
# far smaller than a walk's; walk 10.2 is context, not its basis (there, two
# solves on one stream differ by up to 1.3e-4 in log gain and separate runs
# spread over +-8e-4: APPEAR-VEC.md 3.1, 3.5). What it catches is a changed
# stream: the scene's stream is subsampled (`exposure_max_observations` 4,000),
# so one observation more or less redraws the whole subsample.
GAIN_LOG_TOL = 1e-4


@pytest.mark.parametrize("value, expected", [(None, False), ("", False),
                                              ("off", False), ("on", True)])
def test_appearance_batched_switch_is_default_off(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("TOWER_WORLD_APPEARANCE_BATCHED", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_APPEARANCE_BATCHED", value)
    assert world_appearance_batched_setting() is expected


def test_batch_views_are_whole_groups_within_the_pair_budget():
    for points in (1, 7, 880, 3600, 50_000, 10 ** 7):
        for group in (32, 48):
            views = A._batch_views(points, group)
            assert views % group == 0 and views >= group
            if points * group <= A.BATCH_PAIRS:
                assert views * points <= A.BATCH_PAIRS
                assert (views + group) * points > A.BATCH_PAIRS


def test_exposure_field_with_held_coordinates_is_the_same_field():
    w, h = 37, 23
    coords = A.exposure_pixel_coordinates(w, h)
    for slope, vignette in ((None, None), ((0.031, -0.02), None), (None, (-0.16, 0.05)),
                            ((-0.07, 0.11), (-0.162, 0.0538))):
        old = A.exposure_field(slope, vignette, w, h)
        new = A.exposure_field(slope, vignette, w, h, coordinates=coords)
        assert old.dtype == new.dtype == np.float32
        assert old.tobytes() == new.tobytes()


def _scene(n=49, h=24, w=32):
    """A textured wall seen by `n` keyframes that pan and slide, with what the
    filters must drop: a hole in the proxy, a transparent patch, clipped and
    near-black texels, a source with no valid points, a near occluder in the
    proxy of one view, and a transient (a red patch) in one keyframe. Two grey
    ramps run through the filter bands, so a slip in any of those thresholds
    moves samples across them: a dark one through `exposure_dark` (0.02) and
    `transient_dark` (0.03), a bright one through `transient_saturated` (0.92)
    and `exposure_saturated` (0.97). Non-square so a W/H mix-up shows. 49
    views cross both group sizes (32 and 48) and leave the last 48-view group
    holding one keyframe."""
    rng = np.random.default_rng(7)
    y, x = np.mgrid[:h, :w]
    base = np.stack((60 + 4 * x, 80 + 5 * y, 100 + 2 * (x + y)), -1)
    rgbs, opaque, zps, Rs, ts = [], [], [], [], []
    for i in range(n):
        img = np.clip(base + rng.integers(-3, 4, base.shape) + (i % 5), 0, 255).astype(np.uint8)
        img[0:3, 0:4] = 255          # clipped
        img[-3:, -4:] = 0            # near black
        img[12:18, 24:32] = (2 + (x[12:18, 24:32] - 24) + i % 3)[..., None]          # 2-11 of 255
        img[18:24, 12:20] = (228 + 3 * (x[18:24, 12:20] - 12) + i % 3)[..., None]    # 228-251
        rgbs.append(img)
        op = np.ones((h, w), bool)
        op[5:8, 20:26] = False       # a transparent patch
        opaque.append(op)
        z = np.full((h, w), 2.0, np.float32) + 0.002 * x.astype(np.float32)
        z[18:22, 2:6] = np.nan       # a hole in the proxy
        zps.append(z)
        ang = 0.02 * np.sin(i / 3)
        c, s = np.cos(ang), np.sin(ang)
        Rs.append(np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), np.float32))
        ts.append(np.array((i / 150 - 0.15, 0.03 * np.cos(i / 4), 0.0), np.float32))
    rgbs[7][10:16, 10:16] = (200, 40, 40)   # the transient
    opaque[0][:] = False                     # no valid points
    zps[11][2:12, 2:12] = 1.2                # a near occluder in one proxy
    K = np.array(((24, 0, w / 2), (0, 24, h / 2), (0, 0, 1)), np.float32)
    return rgbs, opaque, zps, Rs, ts, K


def _params():
    return A.AppearanceParams(exposure_grid_px=3, exposure_border_px=2, exposure_box_px=3,
                              exposure_max_observations=4000, exposure_cg_outer=1,
                              exposure_cg_iterations=5, transient_grid_px=3,
                              transient_box_px=3, transient_shift_px=1)


def _assert_stacks_match(old_stacks, new_stacks):
    """The batched stack has every view, with the source's own row all NaN;
    without that row it is the per-group stack, byte for byte."""
    assert [s["a"] for s in old_stacks] == [s["a"] for s in new_stacks]
    for old, new in zip(old_stacks, new_stacks):
        a, C = new["a"], new["C"]
        assert np.isnan(C[a]).all()
        _assert_same(old["C"], np.ascontiguousarray(np.delete(C, a, axis=0)), f"stack of {a}")


# 49 views: one batch of two groups (default), or one group per batch (1).
# 97 views: 32-view groups 32+32+32+1 and 48-view groups 48+48+1; with 9,000
# pairs (~85 points a source) a batch holds three exposure groups or two vote
# groups, so batches of several groups follow one another.
@pytest.mark.parametrize("n, pairs", [(49, None), (49, 1), (97, None), (97, 9000)])
@pytest.mark.parametrize("device", DEVICES)
def test_batched_matches_the_per_group_loops_in_one_process(monkeypatch, device, n, pairs):
    if pairs is not None:
        monkeypatch.setattr(A, "BATCH_PAIRS", pairs)
    scene = _scene(n)
    params = _params()
    # index_add_ on CUDA (the IRLS) is atomic, so the gains are not reproducible
    # run to run without this; warn_only because nanmedian's indices alert.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.delenv("TOWER_WORLD_APPEARANCE_GROUPS", raising=False)
    old_mode = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        results = []
        for setting in ("off", "on", "off"):  # the second "off" is the control
            monkeypatch.setenv("TOWER_WORLD_APPEARANCE_BATCHED", setting)
            results.append(_run_traced(monkeypatch, *scene, params, device))
    finally:
        torch.use_deterministic_algorithms(old_mode[0], warn_only=old_mode[1])

    old, new, control = results
    assert old["raw_observations"]["P"].size > 0
    assert len(old["vote_stacks"]) == n - 1  # every keyframe but the one with no points
    assert old["votes"][7][0].any()  # the transient was found
    _assert_same(old, control, "off vs off")
    _assert_same(old["raw_observations"], new["raw_observations"], "observation stream")
    _assert_same(old["spatial_solve"], new["spatial_solve"], "spatial solve")
    _assert_stacks_match(old["vote_stacks"], new["vote_stacks"])
    for key in ("gains", "counts", "record", "votes"):
        _assert_same(old[key], new[key], key)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_projects_each_view_group_as_the_loop_does(monkeypatch, device):
    """The exactness argument rests on each view group's matrix product being
    the loop's own call: the same view indices in the same order, one call per
    group, the source left out of its own group in the votes. One einsum over a
    whole batch is not guaranteed to give the same bits (measured: 1 of 126,585
    elements differ on a CPU), and the equivalence tests cannot always see it."""
    rgbs, opaque, zps, Rs, ts, K = _scene(97)
    params = _params()
    monkeypatch.delenv("TOWER_WORLD_APPEARANCE_GROUPS", raising=False)
    monkeypatch.setattr(A, "BATCH_PAIRS", 9000)  # batches of several groups, several batches
    gains = np.ones((len(rgbs), 3), np.float32)
    original = A._project_group
    for fn, args in ((A.solve_gains, (rgbs, opaque, zps, Rs, ts, K, params)),
                     (A.transient_votes, (rgbs, opaque, zps, gains, Rs, ts, K, params))):
        # the loop's own view groups, read where it calls the einsum
        monkeypatch.setenv("TOWER_WORLD_APPEARANCE_BATCHED", "off")
        start, stop, seen = _trace_at(fn, 'pc = torch.einsum("sij,pj->spi", R[cs], X)', ("cs",))
        start()
        try:
            fn(*args, device=device)
        finally:
            stop()
        loop = [s["cs"].tolist() for s in seen]
        calls = []

        def spy(R, t, X, cs, calls=calls):
            calls.append(cs.tolist())
            return original(R, t, X, cs)

        monkeypatch.setattr(A, "_project_group", spy)
        monkeypatch.setenv("TOWER_WORLD_APPEARANCE_BATCHED", "on")
        try:
            fn(*args, device=device)
        finally:
            monkeypatch.setattr(A, "_project_group", original)
        assert len(loop) > len(rgbs), fn.__name__  # several groups per source
        assert calls == loop, fn.__name__


@pytest.mark.parametrize("device", DEVICES)
def test_batched_without_deterministic_algorithms_differs_only_by_the_solve(monkeypatch, device):
    """As the stage runs: no deterministic mode. The observation stream and,
    given the same gains, the vote stacks and grids are still byte-identical;
    the gains differ at most by the exposure solve's own spread."""
    rgbs, opaque, zps, Rs, ts, K = _scene()
    params = _params()
    monkeypatch.delenv("TOWER_WORLD_APPEARANCE_GROUPS", raising=False)
    runs = []
    for setting in ("off", "on"):
        monkeypatch.setenv("TOWER_WORLD_APPEARANCE_BATCHED", setting)
        start, stop, raw = _trace_at(A.solve_gains, "cnt = torch.bincount(P.to(torch.int64)",
                                     ("P", "S", "C", "UV"))
        start()
        try:
            gains, counts, record = A.solve_gains(rgbs, opaque, zps, Rs, ts, K, params, device=device)
        finally:
            stop()
        runs.append({"raw": raw[0], "gains": gains, "counts": counts, "record": _host(record)})
    old, new = runs
    _assert_same(old["raw"], new["raw"], "observation stream")
    _assert_same(old["counts"], new["counts"], "counts")
    assert np.abs(np.log(new["gains"]) - np.log(old["gains"])).max() <= GAIN_LOG_TOL
    assert np.abs(new["record"]["slopes"] - old["record"]["slopes"]).max() <= GAIN_LOG_TOL
    assert np.abs(np.subtract(new["record"]["vignette"], old["record"]["vignette"])).max() <= GAIN_LOG_TOL

    votes = []
    for setting in ("off", "on"):
        monkeypatch.setenv("TOWER_WORLD_APPEARANCE_BATCHED", setting)
        start, stop, stacks = _trace_at(A.transient_votes, "n = torch.isfinite(C[..., 0]).sum(0)",
                                        ("a", "C"))
        start()
        try:
            out = A.transient_votes(rgbs, opaque, zps, old["gains"], Rs, ts, K, params,
                                    device=device, slopes=old["record"]["slopes"],
                                    vignette=old["record"]["vignette"])
        finally:
            stop()
        votes.append((stacks, _host(out)))
    _assert_stacks_match(votes[0][0], votes[1][0])
    _assert_same(votes[0][1], votes[1][1], "votes")


def test_batched_stops_like_the_loops(monkeypatch):
    rgbs, opaque, zps, Rs, ts, K = _scene()
    monkeypatch.setenv("TOWER_WORLD_APPEARANCE_BATCHED", "on")
    calls = []

    def stop_on_third():
        calls.append(1)
        return len(calls) >= 3

    assert A.solve_gains(rgbs, opaque, zps, Rs, ts, K, _params(), device="cpu",
                         should_stop=stop_on_third) == (None, None, None)
    calls.clear()
    assert A.transient_votes(rgbs, opaque, zps, np.ones((len(rgbs), 3), np.float32), Rs, ts, K,
                             _params(), device="cpu", should_stop=stop_on_third) is None
