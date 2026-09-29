"""DET-GAINS F2 (Claude review of 2026-09-28): the device the ON decision is made on.

The product never passes a device to the exposure solve: `appearance_pipeline.build_appearance` calls
`A.solve_gains(..., device=device)` with its own default `device=None`, and the solve resolves the device itself
(`A._torch_device`). Every ON test in `test_world_builder_appearance_det_gains.py` either forces
`_deterministic_accumulate` or passes `device=` explicitly, so two regressions of that resolution survived it:

  det-raw-device    `solve_gains` decides on the RAW `device` argument (None) instead of the device it resolved:
                    on CUDA the switch silently becomes a no-op while the manifest still records it on;
  spatial-det-cpu   `_solve_spatial_exposure` decides on a fixed CPU device: 9 of the 15 sums stay atomic.

Here the decision is watched through `device=None`, the product's own path:
  - on any host, with CUDA hidden from the solve: both decisions (`solve_gains` and `_solve_spatial_exposure`) are
    made on the very device object the solve resolved, and an as-if-CUDA answer given for THAT object -- and only
    for it -- reaches all 15 sums (a live or an OFF solve under the same answer reaches none); and a whole
    `build_appearance(device=None)` hands the solve `device=None` and gets the same answer;
  - with CUDA and `TOWER_TEST_CUDA=1` (a caller holding the run's gpulock): the real thing, through `solve_gains`
    and through a whole `build_appearance`, each with `device=None` -- resolved to CUDA, every sum `index_put_`,
    repeat-identical to the last bit. Keep these as the torch-upgrade canary.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from tests.test_world_builder_appearance import World, _never_redact
from tests.test_world_builder_appearance_det_gains import (
    CUDA_OPT_IN,
    ENV,
    GOLDEN_PARAMS,
    KEY,
    _count_index_ops,
    golden_inputs,
    index_sums,
)
from tower.world_builder import appearance as A
from tower.world_builder import appearance_pipeline as AP

CUDA_DEVICE = torch.device("cuda")      # a device object only: making it touches no GPU


def _hide_cuda(monkeypatch):
    """The solve resolves `device=None` to the CPU on every host (a CPU test never takes a visible GPU)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _watch(monkeypatch, *, as_if_cuda: bool) -> dict:
    """Record every device the solve resolves, every device a decision is made on, and every sum's
    (device, op). With `as_if_cuda`, a decision made on a device object the solve itself resolved is answered as
    if that device were CUDA; a decision made on anything else is answered as the product answers it."""
    seen: dict = {"resolved": [], "decided": [], "summed": []}
    real_device, real_decide, real_sum = A._torch_device, A._deterministic_accumulate, A._index_accumulate

    def torch_device(device=None):
        dev = real_device(device)
        seen["resolved"].append(dev)
        return dev

    def decide(params, dev):
        seen["decided"].append(dev)
        if as_if_cuda and any(dev is r for r in seen["resolved"]):
            return real_decide(params, CUDA_DEVICE)
        return real_decide(params, dev)

    def accumulate(out, index, source, deterministic):
        seen["summed"].append((out.device, deterministic))
        return real_sum(out, index, source, deterministic)

    monkeypatch.setattr(A, "_torch_device", torch_device)
    monkeypatch.setattr(A, "_deterministic_accumulate", decide)
    monkeypatch.setattr(A, "_index_accumulate", accumulate)
    return seen


def _solve_device_none(params):
    """`solve_gains` exactly as the pipeline calls it -- no device -- on one torch thread."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return A.solve_gains(*golden_inputs(), params)
    finally:
        torch.set_num_threads(threads)


def _made_on_a_resolved_device(seen) -> bool:
    return bool(seen["decided"]) and all(any(d is r for r in seen["resolved"]) for d in seen["decided"])


# -- any host (CUDA hidden) ---------------------------------------------------------------------------------------


def test_through_device_none_the_cuda_decision_is_made_on_the_device_the_solve_resolved(monkeypatch):
    _hide_cuda(monkeypatch)
    monkeypatch.setenv(ENV, "on")
    params = A.AppearanceParams(**GOLDEN_PARAMS)
    assert params.exposure_deterministic_gains is True
    seen = _watch(monkeypatch, as_if_cuda=False)
    _solve_device_none(params)
    assert seen["resolved"] == [torch.device("cpu")]
    resolved = seen["resolved"][0]
    # one decision in solve_gains, one in _solve_spatial_exposure: both on the resolved object itself
    assert len(seen["decided"]) == 2
    assert all(d is resolved for d in seen["decided"]), seen["decided"]
    # every sum ran on that device, with the CPU's answer (the OFF op)
    assert len(seen["summed"]) == index_sums(params)
    assert all(dev == resolved and det is False for dev, det in seen["summed"])


def test_through_device_none_an_as_if_cuda_resolution_reaches_all_15_sites_and_no_live_or_off_solve(monkeypatch):
    _hide_cuda(monkeypatch)
    monkeypatch.setenv(ENV, "on")
    final = A.AppearanceParams(**GOLDEN_PARAMS)
    seen = _watch(monkeypatch, as_if_cuda=True)
    counts = _count_index_ops(monkeypatch)
    _solve_device_none(final)
    assert _made_on_a_resolved_device(seen)
    assert counts["index_add_"] == 0 and counts["index_put_"] == index_sums(final)
    assert len(seen["summed"]) == index_sums(final) and all(det is True for _dev, det in seen["summed"])
    # the controls: the same as-if-CUDA answer never reaches a live solve, nor one with the switch off
    before = Counter(counts)
    live = A.AppearanceParams.live(**GOLDEN_PARAMS)
    _solve_device_none(live)
    monkeypatch.delenv(ENV)
    off = A.AppearanceParams(**GOLDEN_PARAMS)
    assert off.exposure_deterministic_gains is False
    _solve_device_none(off)
    assert counts["index_put_"] == before["index_put_"]
    assert counts["index_add_"] - before["index_add_"] == index_sums(live) + index_sums(off)


def test_the_pipeline_hands_the_solve_device_none_and_the_decision_is_made_on_what_the_solve_resolved(
        monkeypatch, tmp_path):
    """The product's own path end to end, on any host: `build_appearance(device=None)` reaches
    `A.solve_gains(..., device=None)`, and an as-if-CUDA answer for the device THE SOLVE resolved turns every sum
    of that real build's exposure solve into `index_put_`; the manifest records the switch."""
    _hide_cuda(monkeypatch)
    monkeypatch.setenv(ENV, "on")
    w = World(tmp_path)
    seen = _watch(monkeypatch, as_if_cuda=True)
    counts = _count_index_ops(monkeypatch)
    solves = []
    real_solve = A.solve_gains

    def solve_gains(*args, **kwargs):
        before = Counter(counts)
        out = real_solve(*args, **kwargs)
        solves.append({"device": kwargs.get("device", "absent"),
                       "index_add_": counts["index_add_"] - before["index_add_"],
                       "index_put_": counts["index_put_"] - before["index_put_"]})
        return out

    monkeypatch.setattr(A, "solve_gains", solve_gains)
    params = A.AppearanceParams(selection_samples=4000)
    assert params.exposure_deterministic_gains is True and params.exposure_model == A.EXPOSURE_MODEL_SPATIAL
    result = w.build(params=params, redactor_factory=_never_redact, device=None)
    assert result.state == AP.STATE_OK, result.detail
    assert len(solves) == 1 and solves[0]["device"] is None
    assert solves[0]["index_add_"] == 0 and solves[0]["index_put_"] > 0
    # one decision in solve_gains, one in _solve_spatial_exposure, each on the object the solve resolved
    assert len(seen["decided"]) == 2 and _made_on_a_resolved_device(seen)
    assert all(d.type == "cpu" for d in seen["decided"])
    assert w.manifest()["params"][KEY] is True


# -- CUDA (opt-in; never run by an agent outside gpulock) -----------------------------------------------------------

cuda = pytest.mark.skipif(
    not (__import__("os").environ.get(CUDA_OPT_IN) == "1" and torch.cuda.is_available()),
    reason=f"CUDA, and only with {CUDA_OPT_IN}=1 (a caller holding the run's gpulock)")


def _raw(gains, record) -> tuple:
    return (np.asarray(gains, np.float32).tobytes(), np.asarray(record["slopes"], np.float32).tobytes(),
            tuple(record.get("vignette") or ()))


@cuda
def test_on_cuda_through_device_none_every_sum_is_index_put_and_the_solve_repeats_bit_for_bit(monkeypatch):
    monkeypatch.setenv(ENV, "on")
    final = A.AppearanceParams(**GOLDEN_PARAMS)
    seen = _watch(monkeypatch, as_if_cuda=False)
    counts = _count_index_ops(monkeypatch)
    runs = [A.solve_gains(*golden_inputs(), final) for _ in range(3)]
    assert len(seen["resolved"]) == 3 and all(r.type == "cuda" for r in seen["resolved"])
    assert len(seen["decided"]) == 6 and _made_on_a_resolved_device(seen)
    assert all(dev.type == "cuda" and det is True for dev, det in seen["summed"])
    assert counts["index_add_"] == 0 and counts["index_put_"] == 3 * index_sums(final)
    ref = runs[0]
    for g, o, r in runs[1:]:
        assert _raw(g, r) == _raw(ref[0], ref[2]) and np.array_equal(o, ref[1])
    # a live solve through the same path keeps the OFF op on CUDA
    before = Counter(counts)
    live = A.AppearanceParams.live(**GOLDEN_PARAMS)
    A.solve_gains(*golden_inputs(), live)
    assert counts["index_put_"] == before["index_put_"]
    assert counts["index_add_"] - before["index_add_"] == index_sums(live)


@cuda
def test_on_cuda_a_final_build_through_the_pipelines_device_none_is_deterministic_and_repeats(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, "on")
    w = World(tmp_path)
    seen = _watch(monkeypatch, as_if_cuda=False)
    counts = _count_index_ops(monkeypatch)
    solves = []
    real_solve = A.solve_gains

    def solve_gains(*args, **kwargs):
        assert kwargs.get("device", "absent") is None      # the pipeline's default reaches the solve as None
        before = Counter(counts)
        out = real_solve(*args, **kwargs)
        solves.append({"raw": _raw(out[0], out[2]),
                       "index_add_": counts["index_add_"] - before["index_add_"],
                       "index_put_": counts["index_put_"] - before["index_put_"]})
        return out

    monkeypatch.setattr(A, "solve_gains", solve_gains)
    params = A.AppearanceParams(selection_samples=4000)
    assert params.exposure_deterministic_gains is True
    first = w.build(params=params, redactor_factory=_never_redact, device=None)
    assert first.state == AP.STATE_OK, first.detail
    m1 = w.manifest()
    again = w.build(params=params, redactor_factory=_never_redact, device=None, force=True)
    assert again.state == AP.STATE_OK and again.detail != AP.ALREADY_BUILT
    m2 = w.manifest()
    assert m1["params"][KEY] is True
    assert len(solves) == 2
    for s in solves:
        assert s["index_add_"] == 0 and s["index_put_"] > 0
    assert _made_on_a_resolved_device(seen)
    assert all(d.type == "cuda" for d in seen["decided"])
    assert solves[0]["raw"] == solves[1]["raw"]
    assert [k["gain"] for k in m1["keyframes"]] == [k["gain"] for k in m2["keyframes"]]
    assert [c["digest"] for c in m1["chunks"]] == [c["digest"] for c in m2["chunks"]]
