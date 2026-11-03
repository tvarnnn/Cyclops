"""Byte equality of the opt-in appearance index reuse and the original path."""

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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_appearance_groups_match_the_original_in_one_process(monkeypatch, device):
    n, h, w = 49, 32, 32  # cross both 32-view and 48-view group boundaries
    y, x = np.mgrid[:h, :w]
    rgb = np.stack((90 + x, 105 + y, 120 + (x + y) // 2), -1).astype(np.uint8)
    rgbs = [np.ascontiguousarray(rgb + (i % 3)) for i in range(n)]
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

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        results = []
        for setting in ("off", "on"):
            monkeypatch.setenv("TOWER_WORLD_APPEARANCE_GROUPS", setting)
            gains, counts, record = A.solve_gains(
                rgbs, opaque, zps, Rs, ts, K, params, device=device)
            votes = A.transient_votes(
                rgbs, opaque, zps, gains, Rs, ts, K, params, device=device,
                slopes=record["slopes"], vignette=record["vignette"])
            results.append((gains, counts, record, votes))
    finally:
        torch.use_deterministic_algorithms(old_deterministic)

    old, new = results
    np.testing.assert_array_equal(old[0], new[0])
    assert old[0].tobytes() == new[0].tobytes()
    np.testing.assert_array_equal(old[1], new[1])
    assert old[1].tobytes() == new[1].tobytes()
    assert old[2].keys() == new[2].keys()
    for key in old[2]:
        if isinstance(old[2][key], np.ndarray):
            np.testing.assert_array_equal(old[2][key], new[2][key])
            assert old[2][key].tobytes() == new[2][key].tobytes()
        else:
            assert old[2][key] == new[2][key]
    for (old_grid, old_record), (new_grid, new_record) in zip(old[3], new[3]):
        np.testing.assert_array_equal(old_grid, new_grid)
        assert old_grid.tobytes() == new_grid.tobytes()
        assert old_record == new_record
