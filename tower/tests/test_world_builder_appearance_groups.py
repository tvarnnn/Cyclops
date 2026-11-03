"""Byte equality of the opt-in appearance index reuse and the original path."""

import inspect
import sys

import numpy as np
import pytest
import torch

from tower.config import world_appearance_groups_setting
from tower.world_builder import appearance as A


@pytest.mark.parametrize("value, expected", [(None, False), ("", False),
                                              ("off", False), ("on", True)])
def test_appearance_groups_switch_is_default_off(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("TOWER_WORLD_APPEARANCE_GROUPS", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_APPEARANCE_GROUPS", value)
    assert world_appearance_groups_setting() is expected


def _host(value):
    """Tensors to host arrays, recursively, so a result can be compared byte for byte."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: _host(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_host(v) for v in value]
    return value


def _assert_same(old, new, where="result"):
    if isinstance(old, np.ndarray):
        assert isinstance(new, np.ndarray), where
        assert (old.dtype, old.shape) == (new.dtype, new.shape), where
        assert old.tobytes() == new.tobytes(), where
    elif isinstance(old, dict):
        assert old.keys() == new.keys(), where
        for k in old:
            _assert_same(old[k], new[k], f"{where}[{k!r}]")
    elif isinstance(old, list):
        assert len(old) == len(new), where
        for i, (a, b) in enumerate(zip(old, new)):
            _assert_same(a, b, f"{where}[{i}]")
    else:
        assert old == new, where


def _trace_at(fn, marker, names):
    """A tracer for `fn` that keeps `names` (on the host) each time the line
    holding `marker` is about to run. Returns (start, stop, captured)."""
    lines, first = inspect.getsourcelines(fn)
    line = first + next(i for i, s in enumerate(lines) if marker in s)
    code = fn.__code__
    captured = []

    def local(frame, event, arg):
        if event == "line" and frame.f_lineno == line:
            captured.append(_host({k: frame.f_locals.get(k) for k in names}))
        return local

    previous = []

    def start():
        previous.append(sys.gettrace())
        sys.settrace(lambda frame, event, arg: local if frame.f_code is code else None)

    def stop():
        sys.settrace(previous.pop())

    return start, stop, captured


def _run_traced(monkeypatch, rgbs, opaque, zps, Rs, ts, K, params, device):
    """`solve_gains` then `transient_votes` as the pipeline calls them, keeping
    the ORDERED raw exposure observations (P, S, C, UV as they stand before the
    per-point view count, the filter and the sample), everything the spatial
    exposure solve is handed and returns, and every keyframe's vote stack
    (views x points x 3, before the median)."""
    start_g, stop_g, raw = _trace_at(A.solve_gains, "cnt = torch.bincount(P.to(torch.int64)",
                                     ("P", "S", "C", "UV"))
    start_v, stop_v, stacks = _trace_at(A.transient_votes, "n = torch.isfinite(C[..., 0]).sum(0)",
                                        ("a", "C"))
    spatial = []
    original = A._solve_spatial_exposure

    def spy(*a):
        out = original(*a)
        spatial.append(_host({"in": list(a[:8]), "out": list(out)}))
        return out

    monkeypatch.setattr(A, "_solve_spatial_exposure", spy)
    start_g()
    try:
        gains, counts, record = A.solve_gains(rgbs, opaque, zps, Rs, ts, K, params, device=device)
    finally:
        stop_g()
        monkeypatch.setattr(A, "_solve_spatial_exposure", original)
    start_v()
    try:
        votes = A.transient_votes(rgbs, opaque, zps, gains, Rs, ts, K, params, device=device,
                                  slopes=record["slopes"], vignette=record["vignette"])
    finally:
        stop_v()
    assert len(raw) == 1 and len(spatial) == 1  # the stream and the spatial solve were seen
    return {"raw_observations": raw[0], "spatial_solve": spatial[0], "vote_stacks": stacks,
            "gains": gains, "counts": counts, "record": _host(record), "votes": _host(votes)}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_appearance_groups_match_the_original_in_one_process(monkeypatch, device):
    n, h, w = 49, 32, 32  # cross both 32-view and 48-view group boundaries
    y, x = np.mgrid[:h, :w]
    rgb = np.stack((90 + x, 105 + y, 120 + (x + y) // 2), -1).astype(np.uint8)
    rgbs = [np.ascontiguousarray(rgb + (i % 3)) for i in range(n)]
    rgbs[7][10:18, 10:18] = (200, 40, 40)  # a transient the votes must find
    opaque = [np.ones((h, w), bool) for _ in range(n)]
    opaque[0][:] = False  # also exercise a source with no valid points
    zps = [np.full((h, w), 2.0, np.float32) for _ in range(n)]
    Rs = [np.eye(3, dtype=np.float32) for _ in range(n)]
    ts = [np.array((i / 200, 0, 0), np.float32) for i in range(n)]
    K = np.array(((32, 0, w / 2), (0, 32, h / 2), (0, 0, 1)), np.float32)
    params = A.AppearanceParams(exposure_grid_px=4, exposure_border_px=2,
                                exposure_box_px=3, exposure_max_observations=3000,
                                exposure_cg_outer=1, exposure_cg_iterations=5,
                                transient_grid_px=4, transient_box_px=3,
                                transient_shift_px=1)

    # index_add_ on CUDA (the IRLS) is atomic, so the gains are not reproducible
    # run to run without this; warn_only because nanmedian's indices alert.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old_mode = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        results = []
        for setting in ("off", "on", "off"):  # the second "off" is the control
            monkeypatch.setenv("TOWER_WORLD_APPEARANCE_GROUPS", setting)
            results.append(_run_traced(monkeypatch, rgbs, opaque, zps, Rs, ts, K, params, device))
    finally:
        torch.use_deterministic_algorithms(old_mode[0], warn_only=old_mode[1])

    old, new, control = results
    assert old["raw_observations"]["P"].size > 0
    assert len(old["vote_stacks"]) == n - 1  # every keyframe but the one with no points
    assert old["votes"][7][0].any()  # the transient was found
    _assert_same(old, control, "off vs off")
    _assert_same(old, new, "off vs on")
