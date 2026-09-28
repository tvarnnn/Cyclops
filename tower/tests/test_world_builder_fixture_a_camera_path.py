"""FX1-A: saved viewer paths exclude impossible room poses and unsupported riders.

The synthetic solve is image-free. The frozen solves stay in Glasses-scratch;
their NPZ files are loaded in memory and never copied into this repository.
To run the frozen cases, set TOWER_FX1A_W4_WORLD_DIR and
TOWER_FX1A_W3_WORLD_DIR to their respective world directories. The lead's
default copies are under Glasses-scratch/wb-coherence-run-2026-09-23/
physical-test/scored/<short world ID>/worlds/<full world ID>.
A frozen case skips when its copy is absent (variable unset, files missing, or
an older solution schema), unless WB_FROZEN_FIXTURES_REQUIRED=1, which makes
each of those a failure; a run that sets it must set both variables too.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests import wb_pose_quarantine_fixtures as Q
from tests.wb_anchor_verify_fixtures import Store
from tower.config import WORLD_POSE_QUARANTINE_ENV
from tower.world_builder import appearance_pipeline as AP
from tower.world_builder import appearance_render as AR
from tower.world_builder import surface as Surface
from tower.world_builder import surface_pipeline as SP
from tower.world_builder import surface_render as SR
from tower.world_builder.global_solve import SOLUTION_SCHEMA_VERSION, SolveWorkspace, load_solution, write_solution


W4 = ("c81766a3", "c81766a3bd3a4eeb9739aa4271fdbb75", "9b03ac81387b471bb90645f6c74ddc03")
W3 = ("4f5d0b15", "4f5d0b15ce314e80a11fe031707767dc", "a9c7dcba37a0404285dce0bc952656f4")
W4_BAD = (2026, 2031, 2037, 2047, 2050, 2053, 2055, 2056, 3090, 3092, 3095, 3097, 3098)
# LABEL4 independently names W4 s1617 as bed and s2847 as desk content.
# W3's retained controls are a solve/path characterization, not visual labels.
GOOD = {"w4": (1617, 2847), "w3": (5827, 6505)}
# Opt-in, for the lead's integration run on the host that holds the copies, as
# in fixture E: "1" turns a frozen case that cannot run into a failure, so a
# moved or renamed copy cannot quietly turn it into a skip (C9 LOW).
FROZEN_REQUIRED_ENV = "WB_FROZEN_FIXTURES_REQUIRED"


def _frozen_copy_unusable(reason: str):
    if os.environ.get(FROZEN_REQUIRED_ENV) == "1":
        pytest.fail(f"{FROZEN_REQUIRED_ENV}=1 but {reason}")
    pytest.skip(reason)


def _kid(session_id: str, seq: int) -> str:
    return f"{session_id}:{seq:08d}"


def _camera(pose: dict) -> list[float]:
    """The persisted pose's centre and forward vector in the viewer wire format."""
    R = np.asarray(pose["rotation"], float).reshape(3, 3)
    centre = -R.T @ np.asarray(pose["translation"], float)
    forward = R.T @ np.array([0.0, 0.0, 1.0])
    return [round(float(v), 5) for v in (*centre, *forward)]


def _synthetic_solution():
    # ORACLE: 12 supported room cameras, two impossible but well-observed poses,
    # and two riders below the 30-observation publication floor.
    entries = [(_kid(Q.SID, 100 + 10 * i), (float(i + 1), 1.0, 1.0), 60, 0) for i in range(12)]
    entries += [(_kid(Q.SID, 230), (1_000_000.0, 1.0, 1.0), 100, 0),
                (_kid(Q.SID, 240), (2_000_000.0, 1.0, 1.0), 100, 0),
                (_kid(Q.SID, 250), (2.5, 1.0, 1.0), 2, 0),
                (_kid(Q.SID, 260), (500.0, 1.0, 1.0), 0, 0)]
    solution = Q.pose_solution(entries)
    for kid, centre, _, _ in entries:
        solution.poses[kid]["rotation"] = np.eye(3).ravel().tolist()
        solution.poses[kid]["translation"] = [-float(v) for v in centre]
    return solution


def _mock_page_artifacts(monkeypatch):
    """Feed both page builders a valid tiny mesh while keeping the real path call."""
    mesh = (np.zeros((3, 3), np.float32), np.array([[0, 1, 2]], np.int32), None, None)
    monkeypatch.setattr(SP, "read_surface_manifest", lambda *a: {"levels": [{"level": 0, "bytes": 4}]})
    monkeypatch.setattr(SP, "read_surface_level", lambda *a, **k: b"mesh")
    monkeypatch.setattr(SP, "surface_currency", lambda *a: {"current": True})
    monkeypatch.setattr(SR, "read_mesh_bytes", lambda raw: mesh)
    monkeypatch.setattr(SR, "_world_up", lambda *a: None)
    monkeypatch.setattr(AP, "read_appearance_manifest", lambda *a: {
        "keyframes": [{"tier": "phone"}], "proxy": {"digest": "synthetic"}})
    monkeypatch.setattr(AP, "label_matches", lambda *a: True)
    monkeypatch.setattr(AP, "read_appearance_file", lambda *a: b"mesh")
    monkeypatch.setattr(AP, "appearance_currency", lambda *a: {"current": True})
    monkeypatch.setattr(Surface, "read_mesh_bytes", lambda raw: mesh)


@pytest.mark.parametrize("page", ["surface", "appearance"])
@pytest.mark.parametrize("switch", ["off", "path"])
def test_synthetic_saved_path_at_both_page_builders(tmp_path, monkeypatch, page, switch):
    solution = _synthetic_solution()
    ws = SolveWorkspace(tmp_path / "synthetic" / "solve" / Q.SID)
    write_solution(ws, solution)
    # CHARACTERIZATION: both files are required; a missing NPZ makes load_solution
    # return None and an exclusion-only assertion would pass on an empty path.
    assert ws.solution_path.is_file() and ws.arrays_path.is_file()
    assert load_solution(Store(tmp_path), "synthetic", Q.SID) is not None
    monkeypatch.setenv(WORLD_POSE_QUARANTINE_ENV, switch)
    _mock_page_artifacts(monkeypatch)
    store = Store(tmp_path)
    if page == "surface":
        _, config = SR.build_surface_payload(store, "synthetic", Q.SID)
    else:
        config = AR.build_appearance_config(store, "synthetic", Q.SID)
    path = config["cameras"]
    # ORACLE: both served pages retain named supported room cameras and a path.
    assert path and all(_camera(solution.poses[_kid(Q.SID, seq)]) in path for seq in (100, 210))
    if switch == "path":
        # ORACLE: no impossible room pose or under-supported rider is navigable.
        assert len(path) == 12
        assert all(_camera(solution.poses[_kid(Q.SID, seq)]) not in path for seq in (230, 240, 250, 260))
    else:
        # CHARACTERIZATION: off preserves today's 16-camera wire list byte for byte.
        expected = [[float(i + 1), 1.0, 1.0, 0.0, 0.0, 1.0] for i in range(12)]
        expected += [[1_000_000.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                     [2_000_000.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                     [2.5, 1.0, 1.0, 0.0, 0.0, 1.0], [500.0, 1.0, 1.0, 0.0, 0.0, 1.0]]
        assert json.dumps(path).encode() == json.dumps(expected).encode()


def test_dwell_and_small_room_keep_a_nonempty_path(tmp_path, monkeypatch):
    monkeypatch.setenv(WORLD_POSE_QUARANTINE_ENV, "path")
    # CHARACTERIZATION: the radius gate keeps its eight-pose floor. With seven
    # supported poses it preserves even the far pose; a small room is not emptied.
    small = Q.pose_solution([(_kid(Q.SID, i), (float(i), 0.0, 0.0), 60, 0) for i in range(6)] +
                            [(_kid(Q.SID, 99), (1_000_000.0, 0.0, 0.0), 60, 0)])
    write_solution(SolveWorkspace(tmp_path / "small" / "solve" / Q.SID), small)
    small_path = SR._camera_path(Store(tmp_path), "small", Q.SID)
    assert len(small_path) == 7 and _camera(small.poses[_kid(Q.SID, 99)]) in small_path
    # ORACLE: the zero-radius stand-down retains all 12 supported dwell cameras.
    dwell = Q.pose_solution([(_kid(Q.SID, i), (1.0, 0.0, 0.0), 60, 0) for i in range(12)])
    write_solution(SolveWorkspace(tmp_path / "dwell" / "solve" / Q.SID), dwell)
    assert len(SR._camera_path(Store(tmp_path), "dwell", Q.SID)) == 12


class _ReadOnlyFrozenStore:
    """Expose only the path loader needs; no method can create or modify a world."""

    def __init__(self, world_dir: Path, world_id: str):
        self._world_dir, self._world_id = world_dir, world_id

    def world_dir(self, world_id: str) -> Path:
        assert world_id == self._world_id
        return self._world_dir


@pytest.mark.parametrize("case", ["w4", "w3"])
def test_frozen_world_saved_path(case, monkeypatch):
    _, world_id, session_id = W4 if case == "w4" else W3
    variable = f"TOWER_FX1A_{case.upper()}_WORLD_DIR"
    override = os.environ.get(variable)
    if not override:
        _frozen_copy_unusable(f"{variable} is not set")
    world_dir = Path(override)
    solve_dir = world_dir / "solve" / session_id
    solution_path = solve_dir / "solution.json"
    if not solution_path.is_file() or not (solve_dir / "solution.npz").is_file():
        _frozen_copy_unusable(f"local frozen {case.upper()} solution.json and solution.npz are absent")
    schema = json.loads(solution_path.read_text(encoding="utf-8")).get("schema_version")
    if schema != SOLUTION_SCHEMA_VERSION:
        _frozen_copy_unusable(f"local frozen {case.upper()} solution schema {schema!r} != {SOLUTION_SCHEMA_VERSION}; re-freeze the copy")
    store = _ReadOnlyFrozenStore(world_dir, world_id)
    solution = load_solution(store, world_id, session_id)
    # CHARACTERIZATION: the frozen inputs load completely, so the path test
    # cannot pass through load_solution's missing-file fallback.
    assert solution is not None
    monkeypatch.setenv(WORLD_POSE_QUARANTINE_ENV, "path")
    bad = {_kid(session_id, seq) for seq in (W4_BAD if case == "w4" else (2400,))}
    good = {_kid(session_id, seq) for seq in GOOD[case]}
    # CHARACTERIZATION: all bad poses belong to today's full room list, so
    # the rule must remove them; decimation alone cannot satisfy this check.
    assert bad <= set(SR._room_poses(solution))
    eligible = SR.viewable_poses(SR._room_poses(solution, fallback=False))
    # ORACLE (DIAG4 Q3 / LABEL4): impossible poses are not places to stand;
    # the named bed/desk cameras are. W3's good controls are characterization.
    assert not bad & set(eligible)
    assert good <= set(eligible)
    path = SR._camera_path(store, world_id, session_id)
    # The served list is non-empty and contains only eligible cameras,
    # regardless of which eligible positions survive decimation.
    assert path
    assert {tuple(camera) for camera in path} <= {tuple(_camera(pose)) for pose in eligible.values()}
