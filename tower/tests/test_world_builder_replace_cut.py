"""Re-place, don't cut (`world_builder/replace_cut.py`).

Pinned:
  * the targets are the guard's isolated keyframes, the solver room's cameras below the support floor and
    the unposed keyframes -- never a keyframe of another solver component;
  * the merge re-places a keyframe without moving the room: its old observations go (and a point only it
    saw), its registered pose and its new observations come in, a new point never takes an observation
    another point holds;
  * the guard's pose screen refuses a re-placed camera beyond the walking allowance and keeps one within it;
  * the step: nothing to offer changes nothing; no metric evidence re-places nothing; a camera that passes
    every gate leaves its piece; one the scale screen would isolate stays cut; a failure leaves the guard's
    decision as it was; the switch reads off by default;
  * at the publish seam (`guard_publication`): off is the guard alone; on, a re-placed keyframe is in the
    published room and out of the isolated piece, and the summary and the audit say so;
  * with real pycolmap (skipped without it): COLMAP's registration puts collapsed cameras of a synthetic
    scene back where they belong, with the room's cameras fixed.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from tower.world_builder import coherence_publish as CP
from tower.world_builder import final_scale_guard as FSG
from tower.world_builder import global_solve as GS
from tower.world_builder import replace_cut as RC
from tower.world_builder.records import Keyframe

P = FSG.GuardParams()
CAM = {"fx": 300.0, "fy": 300.0, "cx": 180.0, "cy": 320.0, "width": 360, "height": 640}
CLOCK0 = 1_790_000_000.0
STEP = 0.05          # solve units per keyframe
DT = 0.25            # seconds per keyframe


@pytest.fixture(autouse=True)
def _replace_cut_on(monkeypatch):
    # conftest turns the step off for every other test; these tests are about it.
    monkeypatch.setenv("TOWER_WORLD_REPLACE_CUT", "on")


def _kid(i):
    return f"s:{i:08d}"


def _keyframes(n):
    return [Keyframe(keyframe_id=_kid(i), session_id="s", source_seq=i, received_at=CLOCK0 + DT * i,
                     image_relpath=f"images/{i:08d}.jpg", width=360, height=640, byte_count=1)
            for i in range(n)]


def _project(R, t, X):
    Xc = X @ R.T + t
    u = CAM["fx"] * Xc[:, 0] / Xc[:, 2] + CAM["cx"]
    v = CAM["fy"] * Xc[:, 1] / Xc[:, 2] + CAM["cy"]
    ok = (Xc[:, 2] > 0) & (u >= 0) & (u < CAM["width"]) & (v >= 0) & (v < CAM["height"])
    return u, v, ok


def _scene(n=40, *, collapsed=(), unposed=(), other=(), seed=0):
    """A walk along x facing a textured wall: every camera sees most points. `collapsed` keyframes keep
    their rotation but sit on keyframe `collapsed[0]`'s centre with few observations of their own;
    `unposed` have no pose; `other` are a second solver component."""
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.uniform(-2.0, n * STEP + 2.0, 900), rng.uniform(-1.5, 1.5, 900), rng.uniform(3.5, 4.5, 900)]
    kids = [_kid(i) for i in range(n)]
    poses, obs, xy = {}, [], []
    truth = {}
    for i, k in enumerate(kids):
        R = np.eye(3)
        C = np.array([STEP * i, 0.0, 0.0])
        truth[k] = (R, -R @ C)
        if i in unposed:
            continue
        u, v, ok = _project(R, -R @ C, pts)
        idx = np.flatnonzero(ok)
        if i in collapsed:
            idx = idx[:12]          # a collapsed stretch: few observations, all a few cm from the camera
            C = np.array([STEP * collapsed[0], 0.0, 0.0])
        for f, p in enumerate(idx):
            obs.append([i, f, p])
            xy.append([u[p], v[p]])
        poses[k] = {"component": 1 if i in other else 0, "rotation": R.reshape(-1).tolist(),
                    "translation": (-R @ C).tolist(), "observations": int(len(idx))}
    obs = np.asarray(obs, np.int32)
    first = np.full(len(pts), 0, np.int32)
    np.minimum.at(first, obs[:, 2], obs[:, 0])
    sol = GS.Solution(solver="glomap", solved_at=CLOCK0, input_digest="d", keyframe_ids=kids, poses=poses,
                      components=[{"index": 0, "images": n, "images_supported": n, "points": len(pts),
                                   "mean_error_px": 0.5}],
                      xyz=pts.astype(np.float32), rgb=GS.withheld_rgb(len(pts)),
                      component=np.zeros(len(pts), np.int32), first_keyframe=first,
                      track_length=np.bincount(obs[:, 2], minlength=len(pts)).astype(np.int32),
                      error=np.full(len(pts), 0.5, np.float32), observations=obs,
                      observation_xy=np.asarray(xy, np.float32), camera=dict(CAM))
    return sol, truth, pts


def _registration(sol, truth, pts, kids, *, offset=None):
    """What `register` would return for `kids`: their true pose (or shifted by `offset`), an observation of
    every room point they see (feature ids past the solution's own), and one new point each."""
    reg = RC.Registration()
    room_pts = sorted({int(p) for p in sol.observations[:, 2] if p < len(pts)})
    for k in kids:
        R, t = truth[k]
        if offset is not None:
            t = t - R @ np.asarray(offset, np.float64)
        reg.poses[k] = (R, t)
        u, v, ok = _project(*truth[k], pts)
        rows = []
        for f, p in enumerate(p for p in room_pts if ok[p]):
            rows.append((1000 + f, ("o", p)))
            reg.xy[(k, 1000 + f)] = (float(u[p]), float(v[p]))
        reg.obs[k] = rows
    return reg


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def test_targets_are_the_isolated_the_unsupported_and_the_unposed_never_another_component():
    sol, _, _ = _scene(30, collapsed=(10, 11, 12), unposed=(20,), other=(25, 26))
    pieces = {_kid(5), _kid(6)}
    got = RC.targets_of(sol, pieces, min_obs=30)
    assert got == [_kid(i) for i in (5, 6, 10, 11, 12, 20)]


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------


def test_merge_re_places_a_keyframe_without_moving_the_room():
    sol, truth, pts = _scene(30, collapsed=(10,))
    k10 = _kid(10)
    room = {k for k in sol.keyframe_ids if k != k10}
    reg = _registration(sol, truth, pts, [k10])
    # one new point seen by k10 and by room camera 11, plus an element room camera 11 already holds
    held = tuple(int(v) for v in sol.observations[sol.observations[:, 0] == 11][0][:2])
    reg.points[7] = {"xyz": np.array([0.3, 0.1, 2.0]), "error": 0.4,
                     "track": [(k10, 5000), (_kid(11), 5001), (_kid(11), held[1])]}
    reg.obs[k10].append((5000, ("n", 7)))
    reg.xy.update({(k10, 5000): (10.0, 20.0), (_kid(11), 5001): (11.0, 21.0), (_kid(11), held[1]): (0.0, 0.0)})
    out = RC.merge(sol, reg, {k10}, room_kids=room)
    p = out.poses[k10]
    assert p["component"] == 0 and p["registration"] == RC.STEP_ID and p["solver_component"] == 0
    assert np.allclose(p["translation"], truth[k10][1]) and np.allclose(p["rotation"], truth[k10][0].reshape(-1))
    # the room did not move: its poses and the xyz of every point it holds are the solution's own
    for k in room:
        assert out.poses[k]["rotation"] == sol.poses[k]["rotation"]
        assert out.poses[k]["translation"] == sol.poses[k]["translation"]
    old_rows = sol.observations[sol.observations[:, 0] != 10]
    assert np.allclose(out.xyz[:len(sol.xyz)][np.unique(old_rows[:, 2])], sol.xyz[np.unique(old_rows[:, 2])])
    # k10's rows are exactly the registration's: its old rows (feature ids < 1000) are gone
    rows10 = out.observations[out.observations[:, 0] == 10]
    assert len(rows10) == len(reg.obs[k10]) and (rows10[:, 1] >= 1000).all()
    # the new point holds k10 and camera 11's free feature, never the one camera 11 already gave a point
    new_p = len(out.xyz) - 1
    assert sorted(map(tuple, out.observations[out.observations[:, 2] == new_p][:, :2].tolist())) == [(10, 5000),
                                                                                                    (11, 5001)]
    assert out.track_length[new_p] == 2 and out.first_keyframe[new_p] == 10
    # every count is the rows'
    per_kf = np.bincount(out.observations[:, 0], minlength=30)
    assert all(out.poses[k]["observations"] == per_kf[i] for i, k in enumerate(out.keyframe_ids))
    assert len(out.observation_xy) == len(out.observations)
    assert set(map(tuple, out.observations[:, :2].tolist())).__len__() == len(out.observations)


def test_merge_drops_a_point_only_the_re_placed_keyframe_saw():
    sol, truth, pts = _scene(20)
    k = _kid(4)
    # a point only keyframe 4 observes
    sol.xyz = np.vstack([sol.xyz, [[9.0, 9.0, 9.0]]]).astype(np.float32)
    sol.rgb = GS.withheld_rgb(len(sol.xyz))
    for name, v in (("component", 0), ("first_keyframe", 4), ("track_length", 1)):
        setattr(sol, name, np.r_[getattr(sol, name), v].astype(np.int32))
    sol.error = np.r_[sol.error, 0.5].astype(np.float32)
    lone = len(sol.xyz) - 1
    sol.observations = np.vstack([sol.observations, [[4, 999, lone]]]).astype(np.int32)
    sol.observation_xy = np.vstack([sol.observation_xy, [[1.0, 1.0]]]).astype(np.float32)
    reg = _registration(sol, truth, pts, [k])
    out = RC.merge(sol, reg, {k}, room_kids={x for x in sol.keyframe_ids if x != k})
    assert len(out.xyz) == len(sol.xyz) - 1
    assert not np.isclose(out.xyz, [9.0, 9.0, 9.0]).all(axis=1).any()


# ---------------------------------------------------------------------------
# the motion screen
# ---------------------------------------------------------------------------


def _cam(i, centre):
    c = np.asarray(centre, np.float64)
    return FSG.RoomCamera(kid=_kid(i), name=f"{i:08d}.jpg", index=i, t=CLOCK0 + DT * i, R=np.eye(3), tvec=-c,
                          group="room")


def test_the_motion_screen_refuses_a_camera_beyond_the_walking_allowance_and_keeps_one_within():
    room = [_cam(i, [STEP * i, 0, 0]) for i in range(40) if i not in (20, 21)]
    near = [_cam(20, [STEP * 20, 0.02, 0.0]), _cam(21, [STEP * 21, 0.0, 0.0])]
    assert RC.motion_refusals(room, near, 1.0, P) == {}
    far = [_cam(20, [STEP * 20, 0.0, 3.0]), _cam(21, [STEP * 21, 0.0, 0.0])]
    got = RC.motion_refusals(room, far, 1.0, P)
    assert _kid(20) in got and _kid(21) not in got


def test_an_island_the_screen_cannot_test_against_the_room_is_refused():
    # room cameras 0-19 and 40-59; re-placed 20-22 (joined to 19 by testable steps) and 30-33, an island:
    # its nearest room cameras are 10 journal indices (2.5 s) away, beyond the screen's step bounds
    room = [_cam(i, [STEP * i, 0, 0]) for i in list(range(20)) + list(range(40, 60))]
    new = [_cam(i, [STEP * i, 0, 0]) for i in (20, 21, 22, 30, 31, 32, 33)]
    got = RC.motion_refusals(room, new, 1.0, P)
    assert got == {_kid(i): "unanchored" for i in (30, 31, 32, 33)}


def test_a_lone_room_camera_inside_a_stretch_is_no_anchor():
    # room 0-19 and 40-59, plus a lone room camera 30 (its nearest room cameras are 10 indices away);
    # re-placed 31-33 are joined to 30 by testable steps, but 30 is no anchor
    room = [_cam(i, [STEP * i, 0, 0]) for i in list(range(20)) + [30] + list(range(40, 60))]
    new = [_cam(i, [STEP * i, 0, 0]) for i in (20, 31, 32, 33)]
    got = RC.motion_refusals(room, new, 1.0, P)
    assert got == {_kid(i): "unanchored" for i in (31, 32, 33)}
    # a chain may pass THROUGH a lone room camera to an anchor
    new = [_cam(i, [STEP * i, 0, 0]) for i in range(20, 30)] + [_cam(i, [STEP * i, 0, 0]) for i in (31, 32, 33)]
    assert RC.motion_refusals(room, new, 1.0, P) == {}


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------


def _assessed(sol, kfs, log_of):
    """The guard's cameras and v2 assessment for `sol` with per-keyframe log ratios `log_of(i)`."""
    name_of = CP._image_names(kfs)
    cams = FSG.room_cameras(sol, kfs, name_of)
    ml = {c.name: log_of(c.index) for c in cams}
    return cams, FSG.assess_v2(cams, ml, P), name_of


def _evidence(log_of):
    def read(store, world_id, session_id, solution, result, *, database_path, name_of, live_metric_fn=None):
        cams = FSG.room_cameras(solution, _keyframes(len(solution.keyframe_ids)), name_of)
        return {c.name: log_of(c.index) for c in cams}, {"source": FSG.EVIDENCE_LIVE}
    return read


def _isolated_world(n=60, piece=range(20, 35)):
    """A walk whose keyframes `piece` the guard isolates (ratio x0.05), with full observations."""
    sol, truth, pts = _scene(n)
    kfs = _keyframes(n)
    cams, a, name_of = _assessed(sol, kfs, lambda i: math.log(0.05) if i in piece else 0.0)
    assert {k for p in a["pieces"] for k in p["keyframe_ids"]} == {_kid(i) for i in piece}
    return sol, truth, pts, kfs, cams, a, name_of


def _step(sol, a, cams, kfs, name_of, register_fn, **kw):
    return RC.replace_cut(None, "w", "s", sol, a, cams=cams, keyframes=kfs, name_of=name_of,
                          database_path="unused.db", gparams=P, register_fn=register_fn, **kw)


def test_nothing_to_offer_changes_nothing():
    sol, _, _ = _scene(40)
    kfs = _keyframes(40)
    cams, a, name_of = _assessed(sol, kfs, lambda i: 0.0)
    out_sol, out_a, record = _step(sol, a, cams, kfs, name_of, register_fn=lambda *a_, **k: pytest.fail("ran"))
    assert out_sol is sol and out_a is a and record is None


def test_without_metric_evidence_nothing_is_re_placed():
    sol, _, _ = _scene(40, collapsed=(10, 11))
    kfs = _keyframes(40)
    name_of = CP._image_names(kfs)
    cams = FSG.room_cameras(sol, kfs, name_of)
    a = FSG.assess_v2(cams, {}, P)
    out_sol, out_a, record = _step(sol, a, cams, kfs, name_of, register_fn=lambda *a_, **k: pytest.fail("ran"))
    assert out_sol is sol and out_a is a and record["state"] == "skipped"


def test_a_camera_that_passes_every_gate_leaves_its_piece(monkeypatch):
    sol, truth, pts, kfs, cams, a, name_of = _isolated_world()
    monkeypatch.setattr(FSG, "read_evidence", _evidence(lambda i: 0.0))
    piece = [_kid(i) for i in range(20, 35)]
    out_sol, out_a, record = _step(sol, a, cams, kfs, name_of,
                                   register_fn=lambda s, **k: _registration(s, truth, pts, piece))
    assert record["state"] == "replaced" and record["replaced"] == 15 and record["refused"] == {}
    assert out_a["pieces"] == [] and set(piece) <= set(out_a["room_keyframes"])
    assert all(out_sol.poses[k]["registration"] == RC.STEP_ID for k in piece)
    assert out_a["replace_cut"] is record


def test_what_the_scale_screen_would_isolate_stays_cut(monkeypatch):
    sol, truth, pts, kfs, cams, a, name_of = _isolated_world()
    piece = [_kid(i) for i in range(20, 35)]
    # re-placed, the piece still reads x0.05: the rule that cut it cuts it again
    monkeypatch.setattr(FSG, "read_evidence", _evidence(lambda i: math.log(0.05) if 20 <= i < 35 else 0.0))
    out_sol, out_a, record = _step(sol, a, cams, kfs, name_of,
                                   register_fn=lambda s, **k: _registration(s, truth, pts, piece))
    assert out_sol is sol and out_a is a
    assert record["state"] == "none-passed" and record["refused"] == {RC.SCALE: 15}


def test_a_camera_off_its_neighbours_is_refused_by_the_motion_screen(monkeypatch):
    sol, truth, pts, kfs, cams, a, name_of = _isolated_world()
    monkeypatch.setattr(FSG, "read_evidence", _evidence(lambda i: 0.0))
    piece = [_kid(i) for i in range(20, 35)]

    def register(s, **k):
        reg = _registration(s, truth, pts, piece)
        R, t = reg.poses[_kid(27)]
        reg.poses[_kid(27)] = (R, t - R @ np.array([0.0, 0.0, 2.5]))   # 2.5 m forward in 0.25 s
        return reg

    _, out_a, record = _step(sol, a, cams, kfs, name_of, register_fn=register)
    assert record["refused_keyframe_ids"] == {f"{RC.MOTION}:outside-core-cluster": [_kid(27)]}
    assert [p["keyframe_ids"] for p in out_a["pieces"]] == [[_kid(27)]]
    assert out_a["pieces"][0]["replaced"] == 14


def test_a_lone_speed_outlier_between_re_placed_neighbours_takes_the_blame_alone(monkeypatch):
    sol, truth, pts, kfs, cams, a, name_of = _isolated_world()
    monkeypatch.setattr(FSG, "read_evidence", _evidence(lambda i: 0.0))
    piece = [_kid(i) for i in range(20, 35)]

    def register(s, **k):
        reg = _registration(s, truth, pts, piece)
        R, t = reg.poses[_kid(27)]
        reg.poses[_kid(27)] = (R, t - R @ np.array([0.0, 0.0, 1.0]))   # beyond 0.8 m, within the hard 1.6 m
        return reg

    _, _, record = _step(sol, a, cams, kfs, name_of, register_fn=register)
    assert record["refused_keyframe_ids"] == {f"{RC.MOTION}:speed": [_kid(27)]} and record["replaced"] == 14


def test_a_camera_below_the_support_floor_is_refused(monkeypatch):
    sol, truth, pts, kfs, cams, a, name_of = _isolated_world()
    monkeypatch.setattr(FSG, "read_evidence", _evidence(lambda i: 0.0))
    piece = [_kid(i) for i in range(20, 35)]

    def register(s, **k):
        reg = _registration(s, truth, pts, piece)
        reg.obs[_kid(30)] = reg.obs[_kid(30)][:10]
        return reg

    _, _, record = _step(sol, a, cams, kfs, name_of, register_fn=register)
    assert record["refused_keyframe_ids"] == {RC.BELOW_FLOOR: [_kid(30)]} and record["replaced"] == 14


def test_without_the_scale_screen_the_re_placed_cameras_are_recorded_uncertified(monkeypatch):
    sol, truth, pts, kfs, cams, a, name_of = _isolated_world()
    monkeypatch.setattr(FSG, "read_evidence", lambda *a_, **k: pytest.fail("the scale screen ran"))
    piece = [_kid(i) for i in range(20, 35)]
    _, out_a, record = _step(sol, a, cams, kfs, name_of,
                             register_fn=lambda s, **k: _registration(s, truth, pts, piece),
                             params=RC.ReplaceParams(scale_screen=False))
    assert record["replaced"] == 15 and out_a["scale_status"] == FSG.SCALE_UNCERTIFIED
    spans = [u for u in out_a["uncertified"] if u.get("replaced")]
    assert [(u["first_index"], u["last_index"], u["uncertified"]) for u in spans] == [(20, 34,
                                                                                        RC.UNCERTIFIED_REPLACED)]


def test_room_anchors_drop_the_registrations_own_points_and_judge_no_floor(monkeypatch):
    sol, truth, pts, kfs, cams, a, name_of = _isolated_world()
    monkeypatch.setattr(FSG, "read_evidence", lambda *a_, **k: pytest.fail("the scale screen ran"))
    piece = [_kid(i) for i in range(20, 35)]

    def register(s, **k):
        reg = _registration(s, truth, pts, piece)
        for j, kk in enumerate(piece):
            reg.obs[kk] = reg.obs[kk][:5] + [(9000 + j, ("n", 77))]   # 5 room points, one new point
            reg.xy[(kk, 9000 + j)] = (1.0, 1.0)
        reg.points[77] = {"xyz": np.array([0.0, 0.0, 4.0]), "error": 0.5,
                          "track": [(kk, 9000 + j) for j, kk in enumerate(piece)]}
        return reg

    out_sol, out_a, record = _step(sol, a, cams, kfs, name_of, register_fn=register,
                                   params=RC.ReplaceParams(anchors="room"))
    assert record["replaced"] == 15 and out_a["pieces"] == []
    assert len(out_sol.xyz) == len(sol.xyz)          # no new point
    assert all(out_sol.poses[k]["observations"] == 5 for k in piece)
    assert [u["uncertified"] for u in out_a["uncertified"] if u.get("replaced")] == [RC.UNCERTIFIED_REPLACED]


def test_a_failure_leaves_the_guards_decision_as_it_was():
    sol, _, _, kfs, cams, a, name_of = _isolated_world()

    def boom(*a_, **k):
        raise RuntimeError("no database")

    out_sol, out_a, record = _step(sol, a, cams, kfs, name_of, register_fn=boom)
    assert out_sol is sol and out_a is a
    assert record["state"] == "error" and "no database" in record["error"]


def test_the_switch_is_off_by_default_and_reads_on(monkeypatch):
    from tower.config import world_replace_cut_setting

    monkeypatch.delenv("TOWER_WORLD_REPLACE_CUT", raising=False)
    assert world_replace_cut_setting() is False
    monkeypatch.setenv("TOWER_WORLD_REPLACE_CUT", "on")
    assert world_replace_cut_setting() is True


# ---------------------------------------------------------------------------
# at the publish seam
# ---------------------------------------------------------------------------


def _guard(monkeypatch, sol, kfs, *, replace, log_of, register):
    monkeypatch.setattr(FSG, "read_evidence", _evidence(log_of))
    monkeypatch.setattr(RC, "register", register)
    store = types.SimpleNamespace(read_session=lambda w, s: types.SimpleNamespace(started_at=CLOCK0))
    return FSG.guard_publication(store, "w", "s", sol, None, database_path="unused.db", keyframes=kfs,
                                 replace=replace)


def test_at_the_seam_off_is_the_guard_alone_and_on_re_places(monkeypatch):
    sol, truth, pts = _scene(60)
    kfs = _keyframes(60)
    piece = [_kid(i) for i in range(20, 35)]
    calls = []

    def log_of(i):
        # the stored solve's piece reads x0.05; once re-placed (registered pose) it reads x1
        return math.log(0.05) if (20 <= i < 35 and not calls) else 0.0

    def register(s, **k):
        calls.append(1)
        return _registration(s, truth, pts, piece)

    off = _guard(monkeypatch, sol, kfs, replace=False, log_of=log_of, register=register)
    assert off.decision == FSG.DECISION_PUBLISHED and "replace_cut" not in off.summary and not calls
    assert off.summary["keyframes_isolated"] == 15
    assert "replace_cut" not in off.audit["assessment"]
    on = _guard(monkeypatch, sol, kfs, replace=True, log_of=log_of, register=register)
    assert on.decision == FSG.DECISION_PUBLISHED and calls
    assert on.summary["keyframes_isolated"] == 0 and on.summary["replace_cut"]["replaced"] == 15
    assert on.audit["assessment"]["replace_cut"]["state"] == "replaced"
    assert all(int(on.solution.poses[k]["component"]) == 0 for k in piece)
    placed = next(c for c in on.result.components["components"] if c["state"] == CP.CG_PLACED)
    assert set(piece) <= set(placed["keyframe_ids"])


# ---------------------------------------------------------------------------
# real pycolmap: COLMAP's registration against a fixed room
# ---------------------------------------------------------------------------


def test_colmap_registration_puts_collapsed_cameras_back_with_the_room_fixed(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    pycolmap.set_random_seed(1)
    opts = pycolmap.SyntheticDatasetOptions()
    opts.num_rigs = 1
    opts.num_cameras_per_rig = 1
    opts.num_frames_per_rig = 16
    opts.num_points3D = 400
    opts.camera_model_id = pycolmap.CameraModelId.PINHOLE
    opts.camera_width, opts.camera_height = 640, 480
    opts.camera_params = [500.0, 500.0, 320.0, 240.0]
    opts.num_points2D_without_point3D = 20
    db_path = tmp_path / "database.db"
    db = pycolmap.Database.open(str(db_path))
    truth_rec = pycolmap.synthesize_dataset(opts, db)
    db.close()
    images = sorted(truth_rec.images.values(), key=lambda im: im.name)
    kfs = [Keyframe(keyframe_id=_kid(i), session_id="s", source_seq=i, received_at=CLOCK0 + DT * i,
                    image_relpath=f"images/{im.name}", width=640, height=480, byte_count=1)
           for i, im in enumerate(images)]
    sol = GS._solution_from_reconstructions({0: truth_rec}, kfs, solver="glomap", input_digest="d",
                                            min_image_observations=30)
    sol.camera = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
    truth = {k: (np.asarray(p["rotation"]).reshape(3, 3), np.asarray(p["translation"])) for k, p in sol.poses.items()}
    collapsed = [_kid(i) for i in (6, 7, 8)]
    # collapse them: every centre onto keyframe 6's
    R6, t6 = truth[_kid(6)]
    c6 = -R6.T @ t6
    for k in collapsed:
        R, _t = truth[k]
        sol.poses[k] = dict(sol.poses[k], translation=(-R @ c6).tolist())
    room = [k for k in sol.keyframe_ids if k not in collapsed]
    name_of = CP._image_names(kfs)
    reg = RC.register(sol, room_kids=room, target_kids=collapsed, database_path=db_path, name_of=name_of)
    assert set(reg.poses) == set(collapsed) and reg.record["room_moved"] <= 1e-6
    extent = np.ptp(np.stack([-truth[k][0].T @ truth[k][1] for k in room]), axis=0).max()
    for k in collapsed:
        R, t = reg.poses[k]
        c, c_true = -R.T @ t, -truth[k][0].T @ truth[k][1]
        assert np.linalg.norm(c - c_true) < 1e-3 * extent
        assert np.allclose(R, truth[k][0], atol=1e-3)
        assert len(reg.obs[k]) >= 30
    out = RC.merge(sol, reg, set(collapsed), room_kids=set(room))
    assert all(out.poses[k]["observations"] >= 30 for k in collapsed)
    for k in room:
        assert out.poses[k]["translation"] == sol.poses[k]["translation"]
