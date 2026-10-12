"""TOWER_WORLD_UP_ESTIMATE=camera_right_all: the page's vertical from the level head.

Walk 9's room piece faces the desk, looking down (yaw concentration 0.68, median
pitch -19 deg), and its mean camera up leaned 15 deg from the head-level vertical
(RUN review/W9-DIAG.md cause 6). The option measures the vertical as the direction
the cameras' right axes are perpendicular to, over every camera of the room's
solver frame -- including the views the gate or the final scale guard cut from the
room -- and lets the surface refine it only a little. Off is today, byte for byte.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tower import config as CFG
from tower.world_builder import appearance_pipeline as AP
from tower.world_builder import global_solve as GS
from tower.world_builder import surface_pipeline as SP
from tower.world_builder import surface_render as SR

ENV = CFG.WORLD_UP_ESTIMATE_ENV
ON = CFG.WORLD_UP_ESTIMATE_CAMERA_RIGHT_ALL


# ---------------------------------------------------------------------------
# synthetic cameras
# ---------------------------------------------------------------------------


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _angle(a, b):
    return math.degrees(math.acos(min(1.0, max(-1.0, float(_unit(a) @ _unit(b))))))


def _axis_rot(axis, deg):
    axis = _unit(axis)
    a = math.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _frame(up):
    """Two horizontal unit vectors for `up`."""
    up = _unit(up)
    a = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 0.1:
        a = np.cross(up, [0.0, 1.0, 0.0])
    a = _unit(a)
    return a, np.cross(up, a)


def camera(up, yaw_deg, pitch_deg, roll_deg=0.0):
    """R_cw (OpenCV: x right, y down, z forward) of a camera in a world whose
    vertical is `up`. Negative pitch looks down."""
    up = _unit(up)
    e1, e2 = _frame(up)
    yaw = math.radians(yaw_deg)
    f = math.cos(yaw) * e1 + math.sin(yaw) * e2
    y = -up
    x = np.cross(y, f)
    C = np.stack([x, y, f], axis=1)                      # camera -> world, level
    # Pitch about the camera's own x, roll about its own z.
    p = -math.radians(pitch_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    r = math.radians(roll_deg)
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    C = C @ Rx @ Rz
    assert np.allclose(C.T @ C, np.eye(3), atol=1e-9)
    return C.T


def mean_camera_up(rotations):
    return _unit(np.mean([R.T @ np.array([0.0, -1.0, 0.0]) for R in rotations], axis=0))


UP = _unit([0.31, -0.88, 0.36])  # an arbitrary solve gauge: the vertical is not an axis


def desk_walk(n=300, *, yaw_sd=20.0, pitch=-20.0, pitch_sd=8.0, roll_sd=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return [camera(UP, rng.normal(0, yaw_sd), rng.normal(pitch, pitch_sd), rng.normal(0, roll_sd))
            for _ in range(n)]


def test_the_synthetic_camera_looks_down_when_pitched_down():
    R = camera(UP, 0.0, -20.0)
    forward = R.T @ np.array([0.0, 0.0, 1.0])
    assert forward @ UP == pytest.approx(-math.sin(math.radians(20.0)), abs=1e-9)


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------


def test_one_heading_looking_down_biases_the_mean_but_not_the_level_head():
    rots = desk_walk()
    assert _angle(mean_camera_up(rots), UP) > 10.0, "today's estimate leans toward the desk"
    up = SR.camera_right_up(rots)
    assert up is not None
    assert _angle(up, UP) < 1.5
    assert np.dot(up, UP) > 0, "signed by the mean camera up: never upside down"


def test_random_headings_and_pitches():
    rng = np.random.default_rng(1)
    rots = [camera(UP, rng.uniform(0, 360), rng.uniform(-45, 30), rng.normal(0, 3)) for _ in range(200)]
    up = SR.camera_right_up(rots)
    assert up is not None and _angle(up, UP) < 1.0


def test_roll_noise_and_a_few_rolled_heads():
    rots = desk_walk(yaw_sd=35.0, roll_sd=6.0, seed=2)
    rng = np.random.default_rng(3)
    rots += [camera(UP, rng.uniform(0, 360), -10.0, rng.choice([-35.0, 35.0])) for _ in range(20)]
    up = SR.camera_right_up(rots)
    assert up is not None and _angle(up, UP) < 2.5


def test_every_camera_facing_one_way_does_not_measure_a_vertical():
    """Right axes along one line leave the vertical free about it: refuse."""
    rots = [camera(UP, 0.0, p) for p in np.linspace(-40, 10, 50)]
    assert SR.camera_right_up(rots) is None


def test_one_heading_with_roll_noise_is_refused_not_turned_sideways():
    """One heading plus roll: the right axes span the plane of the roll, whose normal
    is the cameras' forward direction, not the vertical. Refused."""
    rng = np.random.default_rng(4)
    rots = [camera(UP, rng.normal(0, 0.5), -20.0, rng.normal(0, 8.0)) for _ in range(100)]
    assert SR.camera_right_up(rots) is None


def test_too_few_or_bad_cameras_are_refused():
    rots = desk_walk(n=4)
    assert SR.camera_right_up(rots) is None
    assert SR.camera_right_up([]) is None
    bad = desk_walk(n=20)
    bad[3] = np.full((3, 3), np.nan)
    assert SR.camera_right_up(bad) is None


# ---------------------------------------------------------------------------
# the page's vertical
# ---------------------------------------------------------------------------


def _quad(corners):
    return np.asarray(corners, float), np.array([[0, 1, 2], [0, 2, 3]])


def room_surface(up, tilt_deg=0.0):
    """A floor, a ceiling and two walls around the origin; the floor and ceiling
    tilted `tilt_deg` off the true level (a bent solve), the walls plumb."""
    e1, e2 = _frame(up)
    B = np.stack([e1, e2, _unit(up)], axis=1)            # local (x, y, up) -> world
    T = _axis_rot([1.0, 0.0, 0.0], tilt_deg)
    parts = [
        (_quad([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]]), T),
        (_quad([[-2, -2, 3], [-2, 2, 3], [2, 2, 3], [2, -2, 3]]), T),
        (_quad([[-2, -2, 0], [-2, -2, 3], [2, -2, 3], [2, -2, 0]]), np.eye(3)),
        (_quad([[-2, -2, 0], [-2, 2, 0], [-2, 2, 3], [-2, -2, 3]]), np.eye(3)),
    ]
    Vs, Fs, base = [], [], 0
    for (V, F), M in parts:
        Vs.append((V @ M.T) @ B.T)
        Fs.append(F + base)
        base += len(V)
    return np.concatenate(Vs), np.concatenate(Fs)


def _pose(R, component, solver_component=None, observations=200):
    p = {"component": component, "rotation": [float(v) for v in np.asarray(R).ravel()],
         "translation": [0.0, 0.0, 0.0], "observations": observations}
    if solver_component is not None:
        p["solver_component"] = solver_component
    return p


def guarded_walk():
    """Walk 7 replay n1's shape: the room (component 0) kept the desk views; the
    guard cut the turned-around views to component 6 of the SAME solver frame; a
    separately solved piece (solver component 1) is in a gauge of its own."""
    rng = np.random.default_rng(5)
    poses = {}
    for i in range(240):                                   # the desk, looking down
        poses[f"s:{i:08d}"] = _pose(camera(UP, rng.normal(0, 15), rng.normal(-22, 6), rng.normal(0, 2)), 0, 0)
    for i in range(240, 420):                              # the rest of the room
        poses[f"s:{i:08d}"] = _pose(camera(UP, rng.uniform(60, 300), rng.normal(-5, 8), rng.normal(0, 2)), 6, 0)
    other = _axis_rot([1.0, 0.0, 0.0], 70.0)               # another gauge entirely
    for i in range(420, 470):
        poses[f"s:{i:08d}"] = _pose(camera(UP, rng.uniform(0, 360), 0.0) @ other, 1, 1)
    return _solution(poses)


def _solution(poses):
    return SimpleNamespace(poses=poses, keyframe_ids=sorted(poses), solved_at=100.0, input_digest="kf")


@pytest.fixture
def solution(monkeypatch):
    """The solve, and a surface and an appearance built from it (after it)."""
    holder = {"solution": guarded_walk(),
              "surface": {"built_at": 200.0, "input_digest": "kf"},
              "appearance": {"built_at": 210.0}}
    monkeypatch.setattr(GS, "load_solution", lambda store, w, s: holder["solution"])
    monkeypatch.setattr(SP, "read_surface_manifest", lambda store, w, s: holder["surface"])
    monkeypatch.setattr(AP, "read_appearance_manifest", lambda store, w, s: holder["appearance"])
    return holder


def test_off_is_todays_estimate_exactly(monkeypatch, solution):
    V, F = room_surface(UP, tilt_deg=8.0)
    seed = SR._camera_up(None, "w", "s")
    refined = SR.surface_up(V, F, seed)
    today = seed if refined is None else refined
    calls = []
    monkeypatch.setattr(SR, "_world_up_camera_right", lambda *a: calls.append(a) or [0.0, 0.0, 1.0])
    for value in (None, "", "off", "on", "1", "true", "CAMERA_RIGHT_ALL", " camera_right_all", "camera_right"):
        if value is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, value)
        assert CFG.world_up_estimate_setting() is None
        assert SR._world_up(None, "w", "s", V, F) == today
    assert calls == [], "the option's code does not run when it is off"


def test_on_uses_every_camera_of_the_room_solver_frame(monkeypatch, solution):
    monkeypatch.setenv(ENV, ON)
    room = [np.array(p["rotation"]).reshape(3, 3) for p in solution["solution"].poses.values()
            if p["component"] == 0]
    assert _angle(mean_camera_up(room), UP) > 15.0, "the trimmed room alone leans toward the desk"
    frame = SR._room_frame_rotations(solution["solution"])
    assert len(frame) == 420, "the guard's pieces are in; the separately solved piece is not"
    V, F = room_surface(UP, tilt_deg=0.0)
    up = SR._world_up(None, "w", "s", V, F)
    assert _angle(up, UP) < 0.5


def test_on_bounds_the_surface_refinement(monkeypatch, solution):
    """The surface's level faces 8 deg from the cameras' level head: today's
    refinement follows the surface wholesale; the option keeps the head-level
    vertical. 2 deg is still the surface's to fine-tune."""
    seed = SR.camera_right_up(SR._room_frame_rotations(solution["solution"]))
    across = _frame(UP)[0]
    surface_up8 = _axis_rot(across, 8.0) @ UP
    V, F = room_surface(surface_up8)
    monkeypatch.delenv(ENV, raising=False)
    off = SR._world_up(None, "w", "s", V, F)
    assert _angle(off, surface_up8) < 0.5
    monkeypatch.setenv(ENV, ON)
    on = SR._world_up(None, "w", "s", V, F)
    assert on == seed and _angle(on, UP) < 1.0

    surface_up2 = _axis_rot(across, 2.0) @ UP
    V, F = room_surface(surface_up2)
    on = SR._world_up(None, "w", "s", V, F)
    assert on != seed and _angle(on, surface_up2) < 0.5


def test_a_bent_solve_keeps_its_plumb_walls_under_the_option(monkeypatch, solution):
    """Floor and ceiling 8 deg off, walls plumb (a bent solve): today levels to the
    floor and leans the walls; the option keeps the walls plumb."""
    V, F = room_surface(UP, tilt_deg=8.0)
    monkeypatch.delenv(ENV, raising=False)
    assert _angle(SR._world_up(None, "w", "s", V, F), UP) > 7.0
    monkeypatch.setenv(ENV, ON)
    assert _angle(SR._world_up(None, "w", "s", V, F), UP) < 1.0


def test_on_falls_back_to_today_when_the_cameras_measure_nothing(monkeypatch, solution):
    poses = {f"s:{i:08d}": _pose(camera(UP, 0.0, p), 0) for i, p in enumerate(np.linspace(-40, 0, 60))}
    solution["solution"] = _solution(poses)
    V, F = room_surface(UP, tilt_deg=8.0)
    monkeypatch.delenv(ENV, raising=False)
    off = SR._world_up(None, "w", "s", V, F)
    monkeypatch.setenv(ENV, ON)
    assert SR._world_up(None, "w", "s", V, F) == off


def test_on_without_a_solution_falls_back_to_today(monkeypatch, solution):
    solution["solution"] = None
    monkeypatch.setenv(ENV, ON)
    V, F = room_surface(UP)
    assert SR._world_up(None, "w", "s", V, F) is None


def test_an_ungated_solve_reads_its_component_as_the_solver_frame(monkeypatch, solution):
    """No `solver_component` on any pose (a solve the gate never relabelled): the
    room's component is its solver frame, and other components are not in it."""
    sol = guarded_walk()
    for p in sol.poses.values():
        p.pop("solver_component")
    frame = SR._room_frame_rotations(sol)
    assert len(frame) == 240


def test_looking_steeply_down_one_way_is_refused():
    """One heading, looking at the floor, with roll: the right axes spread only by
    the roll, so their least direction is the cameras' FORWARD, 90 deg from the mean
    camera up. Refused (the tilt gate; the spread gate would refuse it too)."""
    rng = np.random.default_rng(6)
    rots = [camera(UP, rng.normal(0, 0.3), -65.0, rng.normal(0, 8.0)) for _ in range(100)]
    assert SR.camera_right_up(rots) is None


def test_unsupported_riders_are_left_out_when_there_are_enough_supported_cameras():
    rng = np.random.default_rng(7)
    poses = {f"s:{i:08d}": _pose(camera(UP, rng.uniform(0, 360), -10.0), 0, 0) for i in range(100)}
    for i in range(100, 160):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        poses[f"s:{i:08d}"] = _pose(q, 0, 0, observations=2)
    assert len(SR._room_frame_rotations(_solution(poses))) == 100


@pytest.mark.parametrize("stale", [
    {"surface": {"built_at": 50.0, "input_digest": "kf"}},           # the final solve landed after it
    {"surface": {"built_at": 200.0, "input_digest": "older"}},       # built from other keyframes
    {"surface": {"input_digest": "kf"}},                             # no build time: not known current
    {"surface": None},
    {"appearance": {"built_at": 50.0}},                              # the proxy predates the solve
])
def test_a_mesh_older_than_the_solve_falls_back_to_today(monkeypatch, solution, stale):
    """W7REG o1 was copied between its final solve and the surface rebuilt from it:
    the cameras' vertical was 16 deg off the mesh drawn. Today's estimate measures the
    mesh it draws, so it stands."""
    across = _frame(UP)[0]
    V, F = room_surface(_axis_rot(across, 20.0) @ UP)     # the mesh is another gauge
    monkeypatch.delenv(ENV, raising=False)
    off = SR._world_up(None, "w", "s", V, F)
    monkeypatch.setenv(ENV, ON)
    assert SR._world_up(None, "w", "s", V, F) != off, "current: the option decides"
    solution.update(stale)
    assert SR._world_up(None, "w", "s", V, F) == off


def test_no_appearance_is_not_stale(monkeypatch, solution):
    solution["appearance"] = None
    monkeypatch.setenv(ENV, ON)
    V, F = room_surface(UP)
    assert _angle(SR._world_up(None, "w", "s", V, F), UP) < 0.5
    assert SR._mesh_predates_solve(None, "w", "s", solution["solution"]) is False
