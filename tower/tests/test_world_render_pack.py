"""world_render_pack: the pure parts a review pack can lie about.

  * stations: the start is always station 0, the most-revisited place is found,
    and the rest are spread through the walk, not stacked on one spot;
  * orientation: a box room is upright under its true up and is reported as
    tilted under a tilted one; the floor and the camera height are found;
  * the stage copy never carries a COLMAP database;
  * the CLI's path flags go through the artifact-root guard.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_render_pack as P  # noqa: E402


def _walk_with_a_revisit():
    # out along x for 20 s, a detour along y, then back through x = 2 at t = 50
    out = [(x, 0.0, 0.0) for x in np.linspace(0, 4, 21)]
    side = [(4.0, y, 0.0) for y in np.linspace(0.2, 3, 15)]
    back = [(x, 3.0 - 3.0 * (4 - x) / 2, 0.0) if x > 2 else (x, 0.0, 0.0) for x in np.linspace(4, 0, 21)]
    C = np.array(out + side + back, float)
    T = np.arange(len(C), dtype=float)
    return C, T


def test_stations_start_first_then_revisited_then_spread():
    C, T = _walk_with_a_revisit()
    st = P.choose_stations(C, T, radius=0.5, n=4)
    assert st[0]["index"] == 0 and st[0]["role"].startswith("start")
    assert any(s["role"] == "most-revisited" for s in st)
    rev = next(s for s in st if s["role"] == "most-revisited")
    assert rev["passes"] >= 2
    assert len(st) == 4
    idx = [s["index"] for s in st]
    assert len(set(idx)) == 4
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.linalg.norm(C[idx[i]] - C[idx[j]]) > 0.5


def test_revisit_counts_separates_passes_by_time():
    C = np.array([[0, 0, 0], [0.1, 0, 0], [5, 0, 0], [0.05, 0, 0]], float)
    T = np.array([0.0, 1.0, 20.0, 40.0])
    rc = P.revisit_counts(C, T, radius=0.3, gap_s=8.0)
    assert rc[0] == 2 and rc[2] == 1


def _box(up_tilt_deg=0.0, size=(4.0, 3.0, 2.5), floor_tiles=5):
    """A closed axis-aligned box room (y up) as a triangle mesh."""
    sx, sy, sz = size
    V = np.array([[x, y, z] for x in (0, sx) for y in (0, sy) for z in (0, sz)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    F = []
    for a, b, c, d in quads:
        F += [(a, b, c), (a, c, d)]
    F = np.array(F)
    # subdivide nothing; the orientation checks are area-weighted, so 12 faces suffice
    # the floor needs >= 20 faces for the floor finder: tile the floor
    fx = np.linspace(0, sx, floor_tiles + 1)
    fz = np.linspace(0, sz, floor_tiles + 1)
    extra_v, extra_f = [], []
    base = len(V)
    grid = {}
    for i, x in enumerate(fx):
        for k, z in enumerate(fz):
            grid[i, k] = base + len(extra_v)
            extra_v.append([x, 0.0, z])
    for i in range(floor_tiles):
        for k in range(floor_tiles):
            a, b, c, d = grid[i, k], grid[i + 1, k], grid[i + 1, k + 1], grid[i, k + 1]
            extra_f += [(a, b, c), (a, c, d)]
    V = np.vstack([V, extra_v])
    F = np.vstack([F, extra_f])
    return V, F


def test_orientation_upright_box():
    V, F = _box()
    cams = np.array([[2.0, 1.6, 1.0], [2.5, 1.6, 1.2]])
    o = P.orientation_checks(V, F, [0, 1, 0], cams, metres_per_unit=1.0)
    assert o["wall_tilt_deg"]["area_weighted_mean"] < 0.5
    assert o["floor"]["found"]
    assert o["floor"]["plane_fit_tilt_deg"] < 0.5
    assert o["floor"]["camera_height_above_floor_m"] == pytest.approx(1.6, abs=0.01)


def test_floor_is_the_dominant_height_not_every_low_sheet():
    V, F = _box()
    # a stray, smaller horizontal sheet far below the floor (a mis-scaled piece)
    sheet = np.array([[10, -6, 10], [12, -6, 10], [12, -6.5, 11], [10, -6.5, 11]], float)
    base = len(V)
    V = np.vstack([V, sheet])
    F = np.vstack([F, [(base, base + 1, base + 2), (base, base + 2, base + 3)]])
    cams = np.array([[2.0, 1.6, 1.0]])
    o = P.orientation_checks(V, F, [0, 1, 0], cams, metres_per_unit=1.0)
    assert o["floor"]["camera_height_above_floor_m"] == pytest.approx(1.6, abs=0.01)
    assert o["floor"]["plane_fit_tilt_deg"] < 0.5


def test_floor_ignores_large_stray_geometry_far_from_the_walk():
    # W9-DIAG (kitchen walk 9.1): stray far geometry, larger than the floor, ~28
    # units down in a huge fuse box made the camera "18.3 m above the floor"
    V, F = _box(floor_tiles=20)
    base = len(V)
    big = np.array([[30, -28, 30], [70, -28, 30], [70, -28, 70], [30, -28, 70]], float)
    V = np.vstack([V, big])
    F = np.vstack([F, [(base, base + 1, base + 2), (base, base + 2, base + 3)] * 12])
    cams = np.array([[1.0, 1.6, 0.6], [2.0, 1.7, 1.2], [3.0, 1.5, 1.8]])
    o = P.orientation_checks(V, F, [0, 1, 0], cams, metres_per_unit=1.0)
    assert o["floor"]["found"]
    assert 1.4 < o["floor"]["camera_height_above_floor_m"] < 1.8
    assert o["floor"]["camera_height_plausible"]
    # and without a scale, the same answer in units
    o = P.orientation_checks(V, F, [0, 1, 0], cams, metres_per_unit=None)
    assert o["floor"]["found"]
    assert 1.4 < o["floor"]["camera_height_above_floor_units"] < 1.8


def test_orientation_reports_a_tilted_up():
    V, F = _box()
    tilt = math.radians(10)
    up = [math.sin(tilt), math.cos(tilt), 0.0]
    cams = np.array([[2.0, 1.6, 1.0]])
    o = P.orientation_checks(V, F, up, cams, metres_per_unit=1.0)
    assert o["floor"]["plane_fit_tilt_deg"] == pytest.approx(10, abs=0.5)
    assert o["wall_tilt_deg"]["area_weighted_mean"] > 3


def test_orientation_classifies_dominant_planes():
    V, F = _box()
    planes = [{"normal": [0, 1, 0], "area": 10}, {"normal": [1, 0, 0], "area": 8},
              {"normal": [0.7, 0.7, 0], "area": 2}]
    o = P.orientation_checks(V, F, [0, 1, 0], None, None, planes)
    kinds = [p["kind"] for p in o["dominant_planes"]]
    assert kinds == ["horizontal", "vertical", "oblique"]


def test_stage_copy_leaves_the_colmap_database_behind(tmp_path):
    src = tmp_path / "store"
    w, s = "w" * 32, "s" * 32
    solve = src / "worlds" / w / "solve" / s
    (solve / "images").mkdir(parents=True)
    (solve / "images" / "a.jpg").write_bytes(b"x")
    (solve / "database.db").write_bytes(b"sqlite")
    (solve / "solution.json").write_text("{}")
    (src / "worlds" / w / "dense" / s / "live_depth" / "work").mkdir(parents=True)
    (src / "worlds" / w / "dense" / s / "live_depth" / "align.json").write_text("{}")
    stage = P.stage_world(src, tmp_path / "stage", w, s)
    dst = stage / "worlds" / w
    assert (dst / "solve" / s / "solution.json").exists()
    assert not (dst / "solve" / s / "database.db").exists()
    assert not (dst / "solve" / s / "images").exists()
    assert (dst / "dense" / s / "live_depth" / "align.json").exists()
    assert not (dst / "dense" / s / "live_depth" / "work").exists()
    # the source is untouched
    assert (solve / "database.db").read_bytes() == b"sqlite"


def test_server_refuses_the_live_port(tmp_path):
    with pytest.raises(SystemExit):
        P.PackServer(tmp_path, P.LIVE_TOWER_PORT, None)


@pytest.mark.parametrize("flag", ["--world-root", "--out-root", "--stage-root"])
def test_path_flags_are_guarded(flag, tmp_path):
    drive_child = str(Path(Path.cwd().anchor) / "wb-render-pack-never")
    argv = {"--world-root": str(tmp_path), "--out-root": str(tmp_path), "--stage-root": str(tmp_path)}
    argv[flag] = drive_child
    with pytest.raises(SystemExit):
        P.parse_args([*sum(([k, v] for k, v in argv.items()), []), "--world", "w"])
    assert not Path(drive_child).exists()
