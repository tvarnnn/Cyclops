"""DET-GAINS (manager 146 §1): the appearance's exposure-gain solve made reproducible on CUDA, behind
`TOWER_WORLD_APPEARANCE_DETERMINISTIC_GAINS` (off by default).

RUN experiments/W0-STAGE0/STAGE0.md §3.4 found the one nondeterministic op in the appearance: 15 CUDA
`index_add_` call sites in `appearance.solve_gains` / `_solve_spatial_exposure`, float32 atomics whose
summation order changes run to run. §6.6 measured the fix, `index_put_(accumulate=True)`, bit-identical on
CUDA across repeats and processes. It is NOT reproducible on the CPU, so on a CPU device the switch keeps
`index_add_` -- the OFF op.

FINAL BUILDS ONLY (manager 148 §4): `AppearanceParams.live()` -- the walk's live appearance, built by
`scripts/world_surface.py --live --appearance` -- sets the param False, so with the switch on a live build still
runs `index_add_`, byte for byte the OFF solve, under the OFF params record and digest.

THE GOLDEN (`golden/world_builder_appearance_gains_44fbd13.json`) was recorded from the UNEDITED 44fbd13
`appearance.py` by RUN/experiments/DET-GAINS/golden_record.py. With the switch unset, blank, `off` or garbage:
  - a small CPU exposure solve runs the same torch ops in the same order (every torch function and Tensor
    method, traced) and returns the same bytes (on the recording's torch/numpy/platform; to 1e-6 elsewhere);
  - `AppearanceParams().as_dict()` -- the manifest's `params` and the params digest's input -- is the same
    dict, key for key, value for value and in order.

CPU only. The two CUDA checks at the end run only with CUDA present AND `TOWER_TEST_CUDA=1`, which a caller
sets under the run's gpulock: a test never takes the GPU because it happens to be visible.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode, resolve_name

from tests.test_world_builder_appearance import World, _exposure_inputs, _never_redact
from tower import config
from tower.world_builder import appearance as A
from tower.world_builder import appearance_pipeline as AP

ENV = "TOWER_WORLD_APPEARANCE_DETERMINISTIC_GAINS"
KEY = "exposure_deterministic_gains"
GOLDEN = Path(__file__).parent / "golden" / "world_builder_appearance_gains_44fbd13.json"
N_SITES = 15
# The 15 sites by module-level function: the spatial IRLS (fw, d_la, d_lg, d_sl x2, and in its nested At():
# ga, gg, gs x2) and the rest of solve_gains (the warm start: acc, ws, g, gw; the "before" residual: la0, c0).
SITES = {"_solve_spatial_exposure": 9, "solve_gains": 6}
# ... and by the frame that runs them (At() is a closure of _solve_spatial_exposure).
SITES_BY_FRAME = {"_solve_spatial_exposure": 5, "At": 4, "solve_gains": 6}
CUDA_OPT_IN = "TOWER_TEST_CUDA"

# -- KEEP IN STEP with RUN/experiments/DET-GAINS/golden_record.py: the recorded solve ------------------------

# Two reweightings of three CG steps each: no CG stopping test can end a step early on this problem, so the op
# sequence is fixed by the code alone -- the same on every machine.
GOLDEN_PARAMS = dict(exposure_grid_px=6, exposure_border_px=4, exposure_cg_outer=2, exposure_cg_iterations=3)


def golden_inputs():
    """Six keyframes of the synthetic box room through a known photometric model."""
    n = 6
    gains = [[1.0, 1.0, 1.0]] * n
    gains[2] = [0.7, 0.75, 0.8]
    slopes = [(0.0, 0.0)] * n
    slopes[4] = (0.2, -0.1)
    return _exposure_inputs(gains, slopes, (-0.3, 0.0), n)


def index_sums(params) -> int:
    """How many sums the spatial solve does with `params` when no CG step ends early: the 10-round warm
    start (4 each), each reweighting's 5 plus At() (4) before the loop and once per CG step, then 2."""
    return 10 * 4 + params.exposure_cg_outer * (5 + 4 * (1 + params.exposure_cg_iterations)) + 2


class _OpTrace(TorchFunctionMode):
    """Every torch function and Tensor method a block calls, by name, in order."""

    def __init__(self):
        super().__init__()
        self.ops: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.ops.append(resolve_name(func) or getattr(func, "__qualname__", repr(func)))
        return func(*args, **(kwargs or {}))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_traced(module, params) -> dict:
    """`module.solve_gains` on `golden_inputs()`, on the CPU with ONE torch thread (so a host's core count
    cannot move a reduction's order), traced: the op sequence it ran and what it returned."""
    inputs = golden_inputs()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with _OpTrace() as trace:
            gains, per_frame, record = module.solve_gains(*inputs, params, device="cpu")
    finally:
        torch.set_num_threads(threads)
    record = dict(record)
    slopes = np.asarray(record.pop("slopes"), np.float32)
    gains = np.asarray(gains, np.float32)
    return {
        "ops": len(trace.ops),
        "ops_sha256": _sha256("\n".join(trace.ops).encode()),
        "op_counts": dict(sorted(Counter(trace.ops).items())),
        "gains_sha256": _sha256(gains.tobytes()),
        "slopes_sha256": _sha256(slopes.tobytes()),
        "gains": gains.tolist(),
        "slopes": slopes.tolist(),
        "per_frame": np.asarray(per_frame).astype(int).tolist(),
        "record": json.loads(json.dumps(record, sort_keys=True)),
    }


def params_items(params) -> list:
    """`params.as_dict()` as JSON would write it, in order."""
    return json.loads(json.dumps(list(params.as_dict().items())))


# -----------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _exact_build(golden) -> bool:
    return (golden["torch"], golden["numpy"], golden["machine"], golden["system"]) == (
        torch.__version__, np.__version__, platform.machine(), platform.system())


def _set(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)


def _count_index_ops(monkeypatch) -> Counter:
    """Count the Python-level `Tensor.index_add_` and `Tensor.index_put_` calls from here on."""
    counts: Counter = Counter()
    add, put = torch.Tensor.index_add_, torch.Tensor.index_put_

    def counted_add(self, *a, **k):
        counts["index_add_"] += 1
        return add(self, *a, **k)

    def counted_put(self, *a, **k):
        counts["index_put_"] += 1
        return put(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "index_add_", counted_add)
    monkeypatch.setattr(torch.Tensor, "index_put_", counted_put)
    return counts


def _solve(params):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return A.solve_gains(*golden_inputs(), params, device="cpu")
    finally:
        torch.set_num_threads(threads)


# -- the switch ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   ", "0", "false", "no", "off", "OFF", " Off "])
def test_unset_blank_and_off_are_off_and_say_nothing(monkeypatch, caplog, value):
    monkeypatch.setattr(config, "_world_appearance_deterministic_gains_warned", set())
    _set(monkeypatch, value)
    with caplog.at_level(logging.WARNING, logger="tower.config"):
        assert config.world_appearance_deterministic_gains_setting() is False
    assert not [r for r in caplog.records if ENV in r.getMessage()]


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "ON", " True "])
def test_the_flag_spellings_of_true_are_on(monkeypatch, value):
    _set(monkeypatch, value)
    assert config.world_appearance_deterministic_gains_setting() is True


def test_garbage_is_off_and_logged_once(monkeypatch, caplog):
    monkeypatch.setattr(config, "_world_appearance_deterministic_gains_warned", set())
    monkeypatch.setenv(ENV, "sometimes")
    with caplog.at_level(logging.WARNING, logger="tower.config"):
        assert config.world_appearance_deterministic_gains_setting() is False
        assert config.world_appearance_deterministic_gains_setting() is False
    assert sum(1 for r in caplog.records if ENV in r.getMessage()) == 1


def test_the_switch_is_read_when_the_params_are_built_and_an_explicit_value_wins(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    off = A.AppearanceParams()
    assert off.exposure_deterministic_gains is False
    assert A.AppearanceParams.live().exposure_deterministic_gains is False
    monkeypatch.setenv(ENV, "on")
    assert A.AppearanceParams().exposure_deterministic_gains is True
    # FINAL builds only (manager 148 §4): the live preset never takes the switch ...
    assert A.AppearanceParams.live().exposure_deterministic_gains is False
    # ... and, like every preset value, an explicit override of it wins
    assert A.AppearanceParams.live(exposure_deterministic_gains=True).exposure_deterministic_gains is True
    assert A.AppearanceParams(exposure_deterministic_gains=False).exposure_deterministic_gains is False
    # read once, at construction: a params object never changes under a build
    assert off.exposure_deterministic_gains is False
    assert dataclasses.replace(off, seed=1).exposure_deterministic_gains is False
    monkeypatch.setenv(ENV, "off")
    assert A.AppearanceParams(exposure_deterministic_gains=True).exposure_deterministic_gains is True


@pytest.mark.parametrize("bad", ["off", "on", 1, 0, None])
def test_the_param_is_a_bool_not_anything_truthy(bad):
    with pytest.raises(ValueError):
        A.AppearanceParams(exposure_deterministic_gains=bad)


# -- the params record and the params digest --------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "off", "garbage"])
def test_off_leaves_the_params_record_exactly_as_44fbd13_wrote_it(monkeypatch, golden, value):
    _set(monkeypatch, value)
    assert params_items(A.AppearanceParams()) == golden["params_final"]
    assert params_items(A.AppearanceParams.live()) == golden["params_live"]
    assert KEY not in A.AppearanceParams().as_dict()


def test_on_adds_exactly_the_switch_to_the_params_record(monkeypatch, golden):
    monkeypatch.setenv(ENV, "on")
    items = params_items(A.AppearanceParams())
    assert [k for k, _v in items].count(KEY) == 1 and dict(items)[KEY] is True
    assert [kv for kv in items if kv[0] != KEY] == golden["params_final"]


def test_a_build_with_the_switch_never_reuses_one_without_it_and_the_reverse(tmp_path):
    """The params digest (`appearance_pipeline`, over `params.as_dict()`) is what `_already_built` matches."""
    w = World(tmp_path)
    off = A.AppearanceParams(selection_samples=4000, exposure_deterministic_gains=False)
    on = A.AppearanceParams(selection_samples=4000, exposure_deterministic_gains=True)
    assert w.build(params=off, redactor_factory=_never_redact).state == AP.STATE_OK
    m_off = w.manifest()
    assert KEY not in m_off["params"]
    assert w.build(params=off, redactor_factory=_never_redact).detail == AP.ALREADY_BUILT

    r_on = w.build(params=on, redactor_factory=_never_redact)
    assert r_on.state == AP.STATE_OK and r_on.detail != AP.ALREADY_BUILT
    m_on = w.manifest()
    assert m_on["params"][KEY] is True
    assert m_on["params_digest"] != m_off["params_digest"]
    assert {k: v for k, v in m_on["params"].items() if k != KEY} == m_off["params"]
    # On this CPU device the switch keeps `index_add_`: the gains are the OFF gains, bit for bit.
    assert [k["gain"] for k in m_on["keyframes"]] == [k["gain"] for k in m_off["keyframes"]]
    assert w.build(params=on, redactor_factory=_never_redact).detail == AP.ALREADY_BUILT

    r_back = w.build(params=off, redactor_factory=_never_redact)
    assert r_back.state == AP.STATE_OK and r_back.detail != AP.ALREADY_BUILT
    assert w.manifest()["params_digest"] == m_off["params_digest"]


# -- the helper ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(7,), (7, 3)])
def test_the_helper_is_index_add_when_off_and_the_same_sum_when_on(shape):
    rng = np.random.default_rng(3)
    idx = torch.as_tensor(rng.integers(0, shape[0], 500), dtype=torch.int64)
    # Terms in eighths, small: every summation order gives the same float32, so the two ops must agree exactly.
    src = torch.as_tensor(rng.integers(-64, 64, (500,) + shape[1:]) / 8.0, dtype=torch.float32)
    ref = np.zeros(shape)
    np.add.at(ref, idx.numpy(), src.numpy().astype(np.float64))

    out_off = torch.zeros(shape)
    assert A._index_accumulate(out_off, idx, src, False) is out_off
    out_on = torch.zeros(shape)
    assert A._index_accumulate(out_on, idx, src, True) is out_on
    assert torch.equal(out_off, torch.zeros(shape).index_add_(0, idx, src))
    assert np.array_equal(out_off.numpy(), ref.astype(np.float32))
    assert np.array_equal(out_on.numpy(), ref.astype(np.float32))


def test_the_helper_sums_ordinary_floats_alike_both_ways():
    rng = np.random.default_rng(4)
    idx = torch.as_tensor(rng.integers(0, 40, 20000), dtype=torch.int64)
    src = torch.as_tensor(rng.normal(0, 1, (20000, 3)), dtype=torch.float32)
    off = A._index_accumulate(torch.zeros((40, 3)), idx, src, False)
    on = A._index_accumulate(torch.zeros((40, 3)), idx, src, True)
    ref = np.zeros((40, 3))
    np.add.at(ref, idx.numpy(), src.numpy().astype(np.float64))
    assert np.allclose(off.numpy(), ref, rtol=0, atol=1e-4)
    assert np.allclose(on.numpy(), ref, rtol=0, atol=1e-4)
    empty = torch.zeros(0, dtype=torch.int64)
    for det in (False, True):
        assert torch.equal(A._index_accumulate(torch.zeros((5, 3)), empty, torch.zeros((0, 3)), det),
                           torch.zeros((5, 3)))


def test_the_deterministic_op_is_taken_only_on_cuda_and_only_when_asked():
    on = A.AppearanceParams(exposure_deterministic_gains=True)
    off = A.AppearanceParams(exposure_deterministic_gains=False)
    assert A._deterministic_accumulate(on, torch.device("cuda")) is True
    assert A._deterministic_accumulate(on, torch.device("cuda", 0)) is True
    assert A._deterministic_accumulate(on, "cuda:0") is True
    assert A._deterministic_accumulate(on, torch.device("cpu")) is False
    assert A._deterministic_accumulate(on, "cpu") is False
    assert A._deterministic_accumulate(on, torch.device("meta")) is False
    assert A._deterministic_accumulate(off, torch.device("cuda")) is False
    assert A._deterministic_accumulate(object(), torch.device("cuda")) is False


def test_the_helper_is_the_only_place_either_op_is_named_and_it_has_15_callers():
    tree = ast.parse(inspect.getsource(A))
    lines, start = inspect.getsourcelines(A._index_accumulate)
    helper = range(start, start + len(lines))
    ops = [(n.lineno, n.func.attr) for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr in ("index_add_", "index_add", "index_put_", "index_put",
                               "scatter_add_", "scatter_add", "put_")]
    assert sorted(op for _ln, op in ops) == ["index_add_", "index_put_"]
    assert all(ln in helper for ln, _op in ops)
    callers: Counter = Counter()
    for fn in tree.body:                       # module-level functions; a nested At() counts for its parent
        if isinstance(fn, ast.FunctionDef):
            for n in ast.walk(fn):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id == "_index_accumulate"):
                    callers[fn.name] += 1
    assert dict(callers) == SITES and sum(callers.values()) == N_SITES


# -- the solve, off: the 44fbd13 solve ------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "off", "garbage"])
def test_off_is_the_44fbd13_solve_op_for_op_and_byte_for_byte(monkeypatch, golden, value):
    _set(monkeypatch, value)
    params = A.AppearanceParams(**GOLDEN_PARAMS)
    assert params.exposure_deterministic_gains is False
    got, want = run_traced(A, params), golden["solve"]
    assert got["op_counts"] == want["op_counts"]
    assert (got["ops"], got["ops_sha256"]) == (want["ops"], want["ops_sha256"])
    assert got["per_frame"] == want["per_frame"]
    if _exact_build(golden):
        for k in ("gains_sha256", "slopes_sha256", "gains", "slopes", "record"):
            assert got[k] == want[k], k
    else:
        assert np.allclose(got["gains"], want["gains"], rtol=0, atol=1e-6)
        assert np.allclose(got["slopes"], want["slopes"], rtol=0, atol=1e-6)


def test_off_still_sums_with_index_add_and_never_index_put(monkeypatch, golden):
    monkeypatch.delenv(ENV, raising=False)
    params = A.AppearanceParams(**GOLDEN_PARAMS)
    counts = _count_index_ops(monkeypatch)
    _solve(params)
    assert counts["index_put_"] == 0
    assert counts["index_add_"] == index_sums(params)
    assert counts["index_add_"] == golden["solve"]["op_counts"]["torch.Tensor.index_add_"]


def test_every_one_of_the_15_sites_is_reached_through_the_helper(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    sites: Counter = Counter()
    real = A._index_accumulate

    def spy(out, index, source, deterministic):
        caller = inspect.currentframe().f_back
        sites[(caller.f_code.co_name, caller.f_lineno)] += 1
        return real(out, index, source, deterministic)

    monkeypatch.setattr(A, "_index_accumulate", spy)
    counts = _count_index_ops(monkeypatch)
    params = A.AppearanceParams(**GOLDEN_PARAMS)
    _solve(params)
    assert len(sites) == N_SITES
    assert Counter(fn for fn, _line in sites) == Counter(SITES_BY_FRAME)
    # every index_add_ of the solve came through the helper
    assert sum(sites.values()) == counts["index_add_"] == index_sums(params)


# -- the solve, on ---------------------------------------------------------------------------------------------


def test_on_every_sum_is_index_put_accumulate_and_none_is_index_add(monkeypatch):
    """The CUDA branch, taken on this CPU host by forcing the device decision: which op each site runs is
    the point here, not the values (CPU `index_put_` accumulate is not the op the product runs there)."""
    monkeypatch.setattr(A, "_deterministic_accumulate", lambda params, dev: True)
    params = A.AppearanceParams(**GOLDEN_PARAMS, exposure_deterministic_gains=True)
    counts = _count_index_ops(monkeypatch)
    _solve(params)
    assert counts["index_add_"] == 0
    assert counts["index_put_"] == index_sums(params)


def test_on_a_cpu_device_the_switch_keeps_index_add_and_the_off_bytes(monkeypatch):
    monkeypatch.setenv(ENV, "on")
    on = A.AppearanceParams(**GOLDEN_PARAMS)
    assert on.exposure_deterministic_gains is True
    counts = _count_index_ops(monkeypatch)
    g1, o1, r1 = _solve(on)
    assert counts["index_put_"] == 0 and counts["index_add_"] == index_sums(on)
    g0, o0, r0 = _solve(A.AppearanceParams(**GOLDEN_PARAMS, exposure_deterministic_gains=False))
    assert g1.tobytes() == g0.tobytes() and np.array_equal(o1, o0)
    assert r1["slopes"].tobytes() == r0["slopes"].tobytes() and r1["vignette"] == r0["vignette"]


# -- FINAL builds only (manager 148 §4): with the switch on, a live build is the OFF build --------------------


def _as_if_cuda(monkeypatch):
    """The device half of the decision taken as if the solve ran on CUDA -- the one device where the switch
    changes the op -- while the params half stays the product's. On this CPU host, which op each site runs is
    the point; the values are the CPU's."""
    real = A._deterministic_accumulate
    cuda_device = torch.device("cuda")   # made here, once: a torch.device() inside a traced solve is an op
    monkeypatch.setattr(A, "_deterministic_accumulate", lambda params, dev: real(params, cuda_device))


@pytest.mark.parametrize("value", ["on", "1", "true"])
def test_live_params_with_the_switch_on_are_the_off_live_params(monkeypatch, golden, value):
    monkeypatch.setenv(ENV, value)
    live, final = A.AppearanceParams.live(), A.AppearanceParams()
    assert (live.exposure_deterministic_gains, final.exposure_deterministic_gains) == (False, True)
    # the live params record -- and so the live params digest -- is 44fbd13's, key for key and in order
    assert KEY not in live.as_dict()
    assert params_items(live) == golden["params_live"]
    assert A._deterministic_accumulate(live, torch.device("cuda")) is False
    assert A._deterministic_accumulate(final, torch.device("cuda")) is True


def test_live_with_the_switch_on_runs_the_off_ops_byte_identical_even_on_cuda(monkeypatch, golden):
    _as_if_cuda(monkeypatch)
    monkeypatch.delenv(ENV, raising=False)
    off = run_traced(A, A.AppearanceParams.live(**GOLDEN_PARAMS))
    monkeypatch.setenv(ENV, "on")
    live_on = A.AppearanceParams.live(**GOLDEN_PARAMS)
    on = run_traced(A, live_on)
    assert (on["ops"], on["ops_sha256"]) == (off["ops"], off["ops_sha256"])
    assert on["op_counts"] == off["op_counts"]
    assert "torch.Tensor.index_put_" not in on["op_counts"]
    assert on["op_counts"]["torch.Tensor.index_add_"] == index_sums(live_on)
    for k in ("gains_sha256", "slopes_sha256", "gains", "slopes", "per_frame", "record"):
        assert on[k] == off[k], k
    # The live preset differs from the final one only in fields this spatial solve does not read, so the live
    # solve with the switch on is also the recorded 44fbd13 solve, op for op.
    want = golden["solve"]
    assert (on["ops"], on["ops_sha256"], on["op_counts"]) == (want["ops"], want["ops_sha256"],
                                                              want["op_counts"])
    if _exact_build(golden):
        assert (on["gains_sha256"], on["slopes_sha256"]) == (want["gains_sha256"], want["slopes_sha256"])
    counts = _count_index_ops(monkeypatch)
    _solve(live_on)
    assert counts["index_put_"] == 0 and counts["index_add_"] == index_sums(live_on)


def test_final_with_the_switch_on_takes_the_deterministic_op_at_every_site(monkeypatch):
    """The control for the test above: the same as-if-CUDA decision DOES reach `index_put_` for a final build."""
    _as_if_cuda(monkeypatch)
    monkeypatch.setenv(ENV, "on")
    final = A.AppearanceParams(**GOLDEN_PARAMS)
    counts = _count_index_ops(monkeypatch)
    _solve(final)
    assert counts["index_add_"] == 0 and counts["index_put_"] == index_sums(final)


def test_a_live_build_with_the_switch_on_is_served_as_the_live_build_without_it(monkeypatch, tmp_path):
    w = World(tmp_path)
    monkeypatch.delenv(ENV, raising=False)
    live_off = A.AppearanceParams.live(selection_samples=3000)
    assert w.build(params=live_off, redactor_factory=_never_redact).state == AP.STATE_OK
    m_off = w.manifest()
    assert m_off["quality"] == "live" and KEY not in m_off["params"]

    monkeypatch.setenv(ENV, "on")
    r_live = w.build(params=A.AppearanceParams.live(selection_samples=3000), redactor_factory=_never_redact)
    assert r_live.detail == AP.ALREADY_BUILT
    assert w.manifest()["params_digest"] == m_off["params_digest"]

    r_final = w.build(params=A.AppearanceParams(selection_samples=4000), redactor_factory=_never_redact)
    assert r_final.state == AP.STATE_OK and r_final.detail != AP.ALREADY_BUILT
    m_final = w.manifest()
    assert m_final["quality"] == "final" and m_final["params"][KEY] is True


def test_an_area_is_shaded_by_the_rooms_own_final_stages_so_it_takes_the_switch_as_the_room_does():
    """A2 (manager 148 §4), the wiring half of why the room-only GPU proof covers the areas: an area's appearance
    is `area_build.build_area` running THE SAME `world_build_session.final_surface_stages` the room's finish runs
    (the finisher passes that very function; `world_refinish` imports it), whose appearance is the final preset
    `AppearanceParams()` -- the one place the switch is read -- and never the live one."""
    import scripts.world_build_session as B
    import scripts.world_finish_pending as F
    from tower.world_builder import area_build as AB

    assert F.final_surface_stages is B.final_surface_stages
    assert "final_surface_stages=final_surface_stages" in inspect.getsource(F.build_session_areas)
    code = "\n".join(line.split("#")[0] for line in inspect.getsource(B.final_surface_stages).splitlines())
    assert "params=AppearanceParams()," in code and "AppearanceParams.live" not in code
    assert "final_surface_stages(" in inspect.getsource(AB.build_area)


class _Captured(Exception):
    pass


@pytest.mark.parametrize("live, expected", [(True, False), (False, True)])
def test_the_surface_cli_builds_the_walks_appearance_without_it_and_a_final_one_with_it(
        monkeypatch, live, expected):
    """`scripts/world_surface.py --live --appearance` is the walk's live appearance (the builder's live child);
    without `--live` it is a final build. The switch is on for both."""
    import scripts.world_surface as CLI

    def build_appearance(store, world_id, sid, *, params=None, **kw):
        raise _Captured(params)

    monkeypatch.setenv(ENV, "on")
    monkeypatch.setattr(AP, "build_appearance", build_appearance)
    with pytest.raises(_Captured) as got:
        CLI._build_appearance(None, "w", "s", live=live)
    params = got.value.args[0]
    assert params.quality == ("live" if live else "final")
    assert params.exposure_deterministic_gains is expected


# -- CUDA (opt-in; never run by an agent outside gpulock) ------------------------------------------------------

cuda = pytest.mark.skipif(
    not (os.environ.get(CUDA_OPT_IN) == "1" and torch.cuda.is_available()),
    reason=f"CUDA, and only with {CUDA_OPT_IN}=1 (a caller holding the run's gpulock)")


@cuda
def test_on_cuda_the_deterministic_sum_is_repeat_identical():
    rng = np.random.default_rng(5)
    idx = torch.as_tensor(rng.integers(0, 477, 2_000_000), dtype=torch.int64, device="cuda")
    src = torch.as_tensor(rng.normal(0, 1, (2_000_000, 3)), dtype=torch.float32, device="cuda")
    outs = {A._index_accumulate(torch.zeros((477, 3), device="cuda"), idx, src, True).cpu().numpy().tobytes()
            for _ in range(3)}
    assert len(outs) == 1


@cuda
def test_on_cuda_the_exposure_solve_is_repeat_identical_with_the_switch_on():
    inputs = golden_inputs()
    params = A.AppearanceParams(exposure_grid_px=6, exposure_border_px=4, exposure_deterministic_gains=True)
    runs = [A.solve_gains(*inputs, params, device="cuda") for _ in range(3)]
    for g, o, r in runs[1:]:
        assert g.tobytes() == runs[0][0].tobytes() and np.array_equal(o, runs[0][1])
        assert r["slopes"].tobytes() == runs[0][2]["slopes"].tobytes()


@cuda
def test_on_cuda_with_the_switch_on_a_final_solve_is_repeat_identical_and_a_live_one_keeps_index_add(
        monkeypatch):
    monkeypatch.setenv(ENV, "on")
    inputs = golden_inputs()
    final = A.AppearanceParams(**GOLDEN_PARAMS)
    assert final.exposure_deterministic_gains is True
    runs = [A.solve_gains(*inputs, final, device="cuda") for _ in range(3)]
    for g, o, r in runs[1:]:
        assert g.tobytes() == runs[0][0].tobytes() and np.array_equal(o, runs[0][1])
        assert r["slopes"].tobytes() == runs[0][2]["slopes"].tobytes()
    live = A.AppearanceParams.live(**GOLDEN_PARAMS)
    counts = _count_index_ops(monkeypatch)
    A.solve_gains(*inputs, live, device="cuda")
    assert counts["index_put_"] == 0 and counts["index_add_"] == index_sums(live)
