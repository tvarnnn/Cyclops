"""The pose quarantine (`TOWER_WORLD_POSE_QUARANTINE`; RUN P5-PQ RULE.md; manager 095 §2 as corrected by 096 §2).

Pinned here:
  THE GOLDEN: with the switch unset, blank, `off` or `0`, the gate's, the publish step's, the anchor verification's and
      the camera path's outputs are what the product wrote BEFORE P5-PQ (6db75f3, `golden/world_builder_pose_
      quarantine_off.json`, recorded by RUN/experiments/P5-PQ/golden_record.py).
  (a) the viewer's camera list keeps only published, supported room poses inside the surface's own radius gate:
      walk 4's six listed impossible poses (2031, 2047, 2053, 2056, 3092, 3097) and walk 3's s2400, at their recorded
      distances, are out; every published room pose inside the gate stays.
  (b) the gate's rider rule: a rider never defaults to the room label; it takes a label only through >= 3 shared
      points with one supported camera; nothing published changes.
  (c) the anchor verification: an impossible-speed motion flag seals an image-unverifiable group (walk 4's bed
      stretch), `link-contradicted`; a flag between two unverifiable groups, a head-turn flag, or a confirmed group
      seals nothing.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests import wb_anchor_verify_fixtures as F
from tests import wb_pose_quarantine_fixtures as Q
from tower.config import WORLD_POSE_QUARANTINE_ENV, world_pose_quarantine_setting
from tower.world_builder import anchor_verify as AV
from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP
from tower.world_builder.surface_render import viewable_poses

GOLDEN = Path(__file__).parent / "golden" / "world_builder_pose_quarantine_off.json"
ENV = WORLD_POSE_QUARANTINE_ENV


def _round_floats(obj, nd=9):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, nd) for v in obj]
    return obj


# ---------------------------------------------------------------------------------------------------------------
# the switch


@pytest.mark.parametrize("value", [None, "", "  ", "off", "OFF", "0", "false", "no"])
def test_the_switch_is_off_unless_it_says_on(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert world_pose_quarantine_setting() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " ON "])
def test_the_switch_is_on_for_flag_spellings_of_true(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert world_pose_quarantine_setting() is True


def test_a_typo_is_off_and_logged(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "onn")
    with caplog.at_level(logging.WARNING, logger="tower.config"):
        assert world_pose_quarantine_setting() is False
    assert ENV in caplog.text


# ---------------------------------------------------------------------------------------------------------------
# THE GOLDEN: off is today's output, byte for byte


@pytest.mark.parametrize("value", [None, "", "off", "0"])
def test_golden_with_the_switch_unset_or_off_every_output_is_todays(tmp_path, monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    monkeypatch.delenv("TOWER_WORLD_ANCHOR_VERIFY", raising=False)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    now = json.loads(json.dumps(Q.golden_outputs(tmp_path), sort_keys=True, default=str))
    expected = golden["outputs"]
    assert sorted(now) == sorted(expected)
    for key in expected:
        if (golden["cv2"], golden["numpy"]) == (cv2.__version__, np.__version__):
            assert now[key] == expected[key], key            # value for value
        else:  # another OpenCV/NumPy build: last digits may move, nothing else may
            assert _round_floats(now[key]) == _round_floats(expected[key]), key


def test_the_golden_exercises_what_it_claims():
    out = json.loads(GOLDEN.read_text(encoding="utf-8"))["outputs"]
    # today: every rider in the room, the one sharing nothing too (the walk-4 leak)
    assert set(out["publish_riders"]["poses"].values()) == {0}
    # today: the bed stretch is flagged and unverifiable, and stays in the room
    av = out["publish_bed_av_on"]["record"]["anchor_verify"]
    assert len(av["motion_flags"]) == 8 and av["sealed_kf"] == 0 and av["room_after"] == 96
    assert [g["verdict"] for g in av["image_verification"]["groups"]].count(AV.VERDICT_UNVERIFIABLE) == 1
    # today: the walk-4-like page lists impossible poses; walk 3's s2400 falls between the kept ones by decimation
    far = lambda path: sum(1 for c in path if np.linalg.norm(c[:3]) > 100)  # noqa: E731
    assert far(out["camera_path_w4"]) == 5 and far(out["camera_path_riders"]) == 2
    assert far(out["camera_path_w3"]) == 0
    assert "pose_quarantine" not in json.dumps(out) and "motion_seal" not in json.dumps(out)


# ---------------------------------------------------------------------------------------------------------------
# (a) the viewer's camera list


def _seq(kid: str) -> int:
    return int(kid.split(":")[1])


def test_a_walk_4s_impossible_poses_leave_the_viewer_list_and_the_room_stays():
    sol = Q.walk_like_solution(Q.W4_ROOM, Q.W4_MEDIAN_RADIUS, {**Q.W4_BED, **Q.W4_RIDERS})
    kept = viewable_poses(sol.poses)
    seqs = {_seq(k) for k in kept}
    assert not seqs & (set(Q.W4_BED) | set(Q.W4_RIDERS))
    assert not seqs & {2031, 2047, 2053, 2056, 3092, 3097}            # the six in walk 4's CONFIG.cameras
    assert len(kept) == Q.W4_ROOM                                      # every published room pose stays


def test_a_walk_3s_s2400_leaves_the_viewer_list():
    sol = Q.walk_like_solution(Q.W3_ROOM, Q.W3_MEDIAN_RADIUS, Q.W3_FAR)
    kept = viewable_poses(sol.poses)
    assert 2400 not in {_seq(k) for k in kept} and len(kept) == Q.W3_ROOM


def test_a_riders_never_reach_the_viewer_list_however_near():
    sol = Q.rider_solution(Q.RIDER_N, Q.RIDERS)
    kept = viewable_poses(sol.poses)
    assert sorted(kept) == sorted(sol.keyframe_ids[:Q.RIDER_N])


@pytest.mark.parametrize("value, far_expected, listed", [(None, 5, 188), ("on", 0, 177)])
def test_a_camera_path_is_filtered_only_with_the_switch_on(tmp_path, monkeypatch, value, far_expected, listed):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    sol = Q.walk_like_solution(Q.W4_ROOM, Q.W4_MEDIAN_RADIUS, {**Q.W4_BED, **Q.W4_RIDERS})
    path = Q.camera_path_of(tmp_path, sol)
    assert (tmp_path / "w1" / "solve" / F.SID / "solution.npz").is_file()
    assert sum(1 for c in path if np.linalg.norm(c[:3]) > 100) == far_expected
    assert len(path) == listed                                        # today 375 -> 188; on 354 kept -> 177
    # a named room camera is on the path either way (the first pose, which decimation always keeps)
    first_room = next(k for k in sol.keyframe_ids if int(sol.poses[k]["observations"]) >= 30)
    R = np.asarray(sol.poses[first_room]["rotation"]).reshape(3, 3)
    c = -R.T @ np.asarray(sol.poses[first_room]["translation"])
    assert any(np.allclose(p[:3], c, atol=1e-4) for p in path)


def test_a_the_surface_gate_is_blind_to_a_far_share_above_five_percent():
    """THE RULE'S KNOWN LIMIT, recorded: the surface gate is relative to the poses' own p95. With 8 impossible poses
    of 96 (8.3 %) the p95 radius is itself an impossible pose, so 2.5 x p95 clears every one of them. Walk 4's
    share is 8 of 354 (2.3 %). (c) is what removes such a stretch from the room itself."""
    room = {k: p for k, p in Q.bed_solution().poses.items() if p["component"] == 0}
    kept = viewable_poses(room)
    bed = {f"{F.SID}:{i:08d}" for i in range(*Q.BED)}
    assert len(room) == 96 and len(set(kept) & bed) == 8


def test_a_an_empty_or_riders_only_room_lists_nothing():
    sol = Q.pose_solution([(f"s1:{i:08d}", (float(i), 0.0, 0.0), 5, 0) for i in range(4)])
    assert viewable_poses(sol.poses) == {}
    assert viewable_poses({}) == {}


# ---------------------------------------------------------------------------------------------------------------
# (b) the gate's rider rule


def _labels(res, n):
    return [res["labels"][F.name(i)] for i in range(n)]


def test_b_today_every_rider_takes_the_room_label():
    model, links, rots, levels = Q.rider_model()
    res = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True)
    assert set(_labels(res, Q.RIDER_N + len(Q.RIDERS))) == {0}


def test_b_with_the_hook_a_rider_takes_a_label_only_through_three_shared_points():
    model, links, rots, levels = Q.rider_model()
    res = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True,
                        rider_min_shared=CP.RIDER_MIN_SHARED)
    got = _labels(res, Q.RIDER_N + len(Q.RIDERS))
    assert set(got[:Q.RIDER_N]) == {0}
    riders = got[Q.RIDER_N:]
    # shares 0 (0 obs), 2, 5, 0 (12 of its own)
    assert riders[2] == 0 and riders[0] == riders[1] == riders[3] != 0
    (piece,) = [c for c in res["components"] if c["label"] == riders[0]]
    assert (piece["state"], piece["reasons"], piece["supported"], piece["cameras"]) == \
        ("unplaced", [CG.REASON_NO_VERIFIED_LINK], 0, 3)
    assert [rd for rd in res["rounds"] if rd.get("quarantined")][0]["riders"] == 3


def test_b_m_one_is_todays_argmax_without_the_default():
    model, links, rots, levels = Q.rider_model()
    res = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True, rider_min_shared=1)
    riders = _labels(res, Q.RIDER_N + len(Q.RIDERS))[Q.RIDER_N:]
    assert riders[1] == riders[2] == 0 and riders[0] == riders[3] != 0


def test_b_the_hook_changes_nothing_but_the_riders():
    model, links, rots, levels = Q.rider_model()
    today = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True)
    hooked = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True,
                           rider_min_shared=CP.RIDER_MIN_SHARED)
    assert _labels(today, Q.RIDER_N) == _labels(hooked, Q.RIDER_N)
    assert today["groups"] == hooked["groups"] and today["params_digest"] == hooked["params_digest"]
    assert today["components"][0] | {"cameras": Q.RIDER_N + 1} == hooked["components"][0]


def test_b_end_to_end_the_riders_leave_the_room_and_nothing_published_changes(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    off = Q.publish_riders(tmp_path / "off")
    monkeypatch.setenv(ENV, "on")
    on = Q.publish_riders(tmp_path / "on")
    riders = [f"{F.SID}:{Q.RIDER_N + k:08d}" for k in range(len(Q.RIDERS))]
    assert [off["poses"][k] for k in riders] == [0, 0, 0, 0]
    assert [on["poses"][k] == 0 for k in riders] == [False, False, True, False]
    assert on["components.json"]["components"] == off["components.json"]["components"]
    assert on["record"]["pose_quarantine"] == {"rider_min_shared": 3, "riders_unplaced": 3}
    assert "pose_quarantine" not in off["record"]


# ---------------------------------------------------------------------------------------------------------------
# (c) the anchor verification: impossible speed seals an unverifiable group


MP = AV.MotionParams()


def _flag(a, b, metres, dt=0.2, deg=3.0):
    return {"a": a, "b": b, "dt": dt, "metres": metres, "deg": deg}


GROUPS = [np.arange(0, 10), np.arange(10, 18), np.arange(18, 30), np.arange(30, 40)]
U, C, X = AV.VERDICT_UNVERIFIABLE, AV.VERDICT_CONFIRMED, AV.VERDICT_CONTRADICTED


def test_c_a_flag_inside_an_unverifiable_group_seals_it():
    hit = AV.motion_sealed_groups(GROUPS, [C, U, C, C], [_flag(11, 12, 5e4)], MP)
    assert list(hit) == [1]


def test_c_a_flag_from_a_confirmed_group_into_an_unverifiable_one_seals_the_unverifiable_side():
    hit = AV.motion_sealed_groups(GROUPS, [C, U, C, C], [_flag(17, 18, 5e4)], MP)
    assert list(hit) == [1]
    hit = AV.motion_sealed_groups(GROUPS, [C, C, U, C], [_flag(17, 18, 5e4)], MP)
    assert list(hit) == [2]


def test_c_a_flag_between_two_unverifiable_groups_seals_neither():
    assert AV.motion_sealed_groups(GROUPS, [C, U, U, C], [_flag(17, 18, 5e4)], MP) == {}


def test_c_confirmed_or_contradicted_groups_are_never_motion_sealed():
    assert AV.motion_sealed_groups(GROUPS, [C, C, C, C], [_flag(11, 12, 5e4)], MP) == {}
    assert AV.motion_sealed_groups(GROUPS, [C, X, U, C], [_flag(17, 18, 5e4)], MP) == {}


def test_c_a_head_turn_flag_is_not_an_impossible_speed():
    # dt 0.2 s: the speed bound is 2.0 * 0.2 + 0.3 = 0.7 m; 0.5 m is a walk, flagged only for its 120 deg turn
    assert not AV.impossible_speed(_flag(11, 12, 0.5, deg=120.0), MP)
    assert AV.impossible_speed(_flag(11, 12, 0.71), MP)
    assert AV.motion_sealed_groups(GROUPS, [C, U, C, C], [_flag(11, 12, 0.5, deg=120.0)], MP) == {}


def test_c_a_camera_outside_every_group_seals_nothing():
    assert AV.motion_sealed_groups(GROUPS, [C, U, C, C], [_flag(17, 45, 5e4)], MP) == {}


def test_c_end_to_end_walk_4s_bed_stretch_is_sealed_link_contradicted(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    out = Q.publish_bed(tmp_path)
    av = out["record"]["anchor_verify"]
    bed = [f"{F.SID}:{i:08d}" for i in range(*Q.BED)]
    assert av["state"] == AV.STATE_APPLIED and av["sealed_kf"] == 8 and av["room_after"] == 88
    assert sorted(av["sealed"][CG.REASON_LINK_CONTRADICTED]["keyframe_ids"]) == bed
    (sg,) = av["sealed_groups"]
    assert sg["source"] == AV.SOURCE_MOTION and sg["reason"] == CG.REASON_LINK_CONTRADICTED and sg["flags"] == 8
    assert av["motion_seal"]["speed_flags"] == 8 and av["motion_seal"]["groups"] == 1
    assert av["collateral"]["keyframes"] == 0
    comps = out["components.json"]["components"]
    assert (comps[0]["keyframes"], comps[0]["shown_as"]) == (88, "room")
    sealed = [e for e in comps if set(e["keyframe_ids"]) <= set(bed)]
    assert sum(e["keyframes"] for e in sealed) == 8
    assert all(e["reasons"] == [CG.REASON_LINK_CONTRADICTED] and e["shown_as"] == "none" for e in sealed)
    assert all(out["poses"][k] != 0 for k in bed)
    assert "anchor-motion" not in json.dumps(out["components.json"])      # the source never reaches the wire


def test_c_an_internal_flag_seals_even_when_the_run_after_is_unverifiable(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    out = Q.publish_bed(tmp_path, confirm_after=False)
    av = out["record"]["anchor_verify"]
    assert av["sealed_kf"] == 8 and av["motion_seal"]["groups"] == 1


@pytest.mark.parametrize("parts", ["scale,images", "images", "motion"])
def test_c_without_both_motion_and_images_nothing_is_motion_sealed(tmp_path, monkeypatch, parts):
    monkeypatch.setenv(ENV, "on")
    out = Q.publish_bed(tmp_path, anchor_verify=parts)
    av = out["record"]["anchor_verify"]
    assert av["sealed_kf"] == 0
    assert all("motion_seal" not in rd or rd["motion_seal"]["groups"] == 0 for rd in av["rounds"])


def test_c_off_the_bed_stretch_stays_and_no_key_is_added(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    out = Q.publish_bed(tmp_path)
    av = out["record"]["anchor_verify"]
    assert av["sealed_kf"] == 0 and av["room_after"] == 96
    assert "motion_seal" not in av and "motion_seal_rule" not in av
    assert all("motion_seal" not in rd for rd in av["rounds"])


# ---------------------------------------------------------------------------------------------------------------
# phase R: the same inputs give the same bytes with the switch on


def test_on_the_same_inputs_give_the_same_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    first = {"riders": Q.publish_riders(tmp_path / "r1"), "bed": Q.publish_bed(tmp_path / "b1"),
             "path": Q.camera_path_of(tmp_path / "c1", Q.walk_like_solution(Q.W4_ROOM, Q.W4_MEDIAN_RADIUS,
                                                                           {**Q.W4_BED, **Q.W4_RIDERS}))}
    again = {"riders": Q.publish_riders(tmp_path / "r2"), "bed": Q.publish_bed(tmp_path / "b2"),
             "path": Q.camera_path_of(tmp_path / "c2", Q.walk_like_solution(Q.W4_ROOM, Q.W4_MEDIAN_RADIUS,
                                                                           {**Q.W4_BED, **Q.W4_RIDERS}))}
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(again, sort_keys=True, default=str)
    assert "seconds" not in json.dumps(first["bed"]["record"]["anchor_verify"].get("motion_seal"))


# ---------------------------------------------------------------------------------------------------------------
# (a) at BOTH call sites of the camera list, on a real built world (the appearance page's synthetic solved,
# surfaced world): `appearance_render.build_appearance_config` and `surface_render.build_surface_payload`. Every
# assertion also names cameras that MUST be there, so an empty list (no `solution.npz`, say) fails rather than passes.


@pytest.fixture
def built_world(tmp_path):
    from tests.test_world_builder_appearance import World, _never_redact

    world = World(tmp_path)
    world.build(redactor_factory=_never_redact)
    return world


def _add_room_far_and_rider(world):
    """The built world's solution, plus 20 published room poses among its own, one published pose flown 100,000
    units out and one rider (2 observations) inside the room. Returns (room centres, far centre, rider centre)."""
    import dataclasses

    base = dict(world.poses)
    first = next(iter(base.values()))
    R = np.asarray(first["rotation"], float).reshape(3, 3)
    centres = [-np.asarray(p["rotation"], float).reshape(3, 3).T @ np.asarray(p["translation"], float)
               for p in base.values()]
    c0 = np.mean(centres, axis=0)
    spread = max(float(np.max(np.linalg.norm(np.asarray(centres) - c0, axis=1))), 0.05)

    def pose(c, obs):
        return {"component": 0, "observations": obs, "rotation": R.ravel().tolist(), "translation": (-R @ c).tolist()}

    extra, room = {}, list(centres)
    for i in range(20):
        c = c0 + spread * 0.5 * np.array([math.cos(i), 0.1 * math.sin(3 * i), math.sin(i)])
        extra[f"{world.kids[0].split(':')[0]}:{9000 + i:08d}"] = pose(c, 50)
        room.append(c)
    far, rider = c0 + np.array([1.0e5, 0.0, 0.0]), c0 + np.array([0.3 * spread, 0.0, 0.0])
    sid = world.kids[0].split(":")[0]
    extra[f"{sid}:{9100:08d}"] = pose(far, 60)
    extra[f"{sid}:{9101:08d}"] = pose(rider, 2)
    world._write_solution(world._workspace, dataclasses.replace(
        world.solution, keyframe_ids=[*world.kids, *extra], poses={**base, **extra}))
    return room, far, rider


def _has(cameras, c) -> bool:
    return any(np.allclose(cam[:3], c, atol=1e-4) for cam in cameras)


@pytest.mark.parametrize("site", ["appearance", "surface"])
def test_a_both_viewer_lists_drop_the_far_pose_and_the_rider_and_keep_the_room(built_world, monkeypatch, site):
    from tests.test_world_builder_appearance import SESSION, WORLD
    from tower.world_builder.appearance_render import build_appearance_config
    from tower.world_builder.surface_render import build_surface_payload

    room, far, rider = _add_room_far_and_rider(built_world)
    assert (built_world._workspace.root / "solution.npz").is_file()

    def cameras():
        if site == "appearance":
            return build_appearance_config(built_world.store, WORLD, SESSION)["cameras"]
        return build_surface_payload(built_world.store, WORLD, SESSION)[1]["cameras"]

    monkeypatch.delenv(ENV, raising=False)
    today = cameras()
    assert len(today) == len(room) + 2 and _has(today, far) and _has(today, rider)
    monkeypatch.setenv(ENV, "on")
    on = cameras()
    assert len(on) == len(room) and all(_has(on, c) for c in room)          # non-empty: every room camera
    assert not _has(on, far) and not _has(on, rider)


# ---------------------------------------------------------------------------------------------------------------
# (b) with the hook: the contract that replaces `test_posed_cameras_always_have_a_room_label` when the switch is on
# (that test pins today's rule and stays, unchanged, for the switch off)


def test_b_on_every_posed_camera_still_has_a_label_but_an_unattached_rider_not_the_rooms():
    model, links, rots, levels = Q.rider_model()
    res = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True,
                        rider_min_shared=CP.RIDER_MIN_SHARED)
    assert sorted(res["labels"]) == sorted(model.names) and min(res["labels"].values()) >= 0
    labels = [c["label"] for c in res["components"]]
    assert all(res["labels"][n] in labels for n in model.names)
    unattached = [F.name(Q.RIDER_N + k) for k in (0, 1, 3)]
    assert all(res["labels"][n] != 0 for n in unattached)


def test_b_on_a_solve_with_no_supported_camera_is_still_one_label():
    model = CG.SolveModel(names=[F.name(i) for i in range(3)], component=np.zeros(3, np.int64),
                          R_cw=np.stack([np.eye(3)] * 3), n_obs=np.array([3, 2, 1]),
                          obs_image=np.array([0, 1, 2]), obs_point=np.array([0, 0, 0]), n_points=1)
    out = CG.apply_gate(model, {}, {}, link_rotations={}, masks_applied=True, rider_min_shared=CP.RIDER_MIN_SHARED)
    assert set(out["labels"].values()) == {0} and [c["label"] for c in out["components"]] == [0]


def test_b_a_genuine_under_floor_room_view_with_its_own_attachment_keeps_the_room_label():
    """Walk 4's keyframe 406 (s2706: the bedroom floor by the entry corner, 24 observations) is a GENUINE room view
    under the 30-observation floor. A rider like it -- 24 observations, 20 of them points a room camera also sees --
    keeps the room label with the hook on."""
    riders = ((12, 20, 4, (0.6, 0.0, 0.3)),)
    sol = Q.rider_solution(Q.RIDER_N, riders)
    model, links, rots, levels = Q.rider_model(riders=riders)
    res = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True,
                        rider_min_shared=CP.RIDER_MIN_SHARED)
    assert int(sol.poses[sol.keyframe_ids[-1]]["observations"]) == 24
    assert res["labels"][F.name(Q.RIDER_N)] == 0
    assert not [rd for rd in res["rounds"] if rd.get("quarantined")]


def test_the_gate_params_and_their_digest_are_unchanged():
    assert CG.GateParams().digest() == "6ce602286efb999f"
    assert "rider" not in json.dumps(CG.GateParams().to_json())
