"""The pose quarantine (`TOWER_WORLD_POSE_QUARANTINE`; RUN P5-PQ RULE.md; manager 095 §2 as corrected by 096 §2, made
granular by 098: parts `path` (a), `riders` (b), `seal` (c'), or `on`; parsed exactly as TOWER_WORLD_ANCHOR_VERIFY).

Pinned here:
  THE SWITCH: its parser is ANCHOR_VERIFY's, shape for shape (`1`, `true`, `yes` and `0` are words it does not know:
      off, and logged). `path` is render-only; `riders` and `seal` change the PUBLISHED solution, and a camera path
      read from it follows it (review V18, LOW-3) -- each part alone changes only its own output otherwise.
  THE GOLDEN: with the switch unset, blank, `off`, or a value it reads as off, the gate's, the publish step's, the
      anchor verification's and the camera path's outputs are what the product wrote BEFORE P5-PQ (6db75f3,
      `golden/world_builder_pose_quarantine_off.json`, recorded by RUN/experiments/P5-PQ/golden_record.py).
  (a') the viewer's camera list (RULE.md (a') v4, manager 102): component 0 only (no fallback: empty if none),
      published and supported poses only, none beyond 10 x their median radius -- the surface gate without its 5 %
      stand-down and its p95 detachment term, its 8-pose floor kept. Walk 4's six listed impossible poses (2031, 2047,
      2053, 2056, 3092, 3097), walk 3's s2400, and a 96-pose room's 8 impossible poses (8.3 %) are out; every good
      pose stays.
  (b) the gate's rider rule: a rider never defaults to the room label; it takes a label only through >= 3 shared
      points with one supported camera; nothing published changes, and every supported camera keeps today's label,
      the room's included (V17 LOW-2; V18 MED-2's widened differential). An all-rider solve keeps label 0, `placed`,
      publishes nothing, and has an empty path with `path` on (RVPQ LOW; V18 LOW-4).
  (c') the anchor verification (RULE.md (c') v3, manager 100): an impossible-speed motion flag seals an
      image-unverifiable group whose flag partner is in a DIFFERENT, image-CONFIRMED group (walk 4's bed stretch,
      through 2056 -> 2061), `link-contradicted`; a flag inside one group (v2's withdrawn clause: walk 3's desk), a
      flag between two unverifiable groups, a head-turn flag, or a confirmed group seals nothing.
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


@pytest.mark.parametrize("value", [None, "", "   ", "off", "OFF", " off "])
def test_unset_blank_and_off_are_off(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert world_pose_quarantine_setting() == frozenset()


@pytest.mark.parametrize("value, parts", [
    ("on", {"path", "riders", "seal"}),
    ("ON", {"path", "riders", "seal"}),
    ("path", {"path"}),
    ("riders", {"riders"}),
    ("seal", {"seal"}),                              # no part implies another
    ("path,riders", {"path", "riders"}),
    (" Path , SEAL ", {"path", "seal"}),
    ("path,riders,seal", {"path", "riders", "seal"}),
    ("seal,seal", {"seal"}),
])
def test_on_and_comma_lists_turn_on_their_parts(monkeypatch, value, parts):
    monkeypatch.setenv(ENV, value)
    assert world_pose_quarantine_setting() == frozenset(parts)


@pytest.mark.parametrize("value", ["yes", "1", "true", "0", "false", "no", "path,foo", "off,path", "on,path", "pth",
                                   ",", "onn"])
def test_garbage_is_off_and_logged(monkeypatch, caplog, value):
    monkeypatch.setenv(ENV, value)
    with caplog.at_level(logging.WARNING, logger="tower.config"):
        assert world_pose_quarantine_setting() == frozenset()
    assert ENV in caplog.text


@pytest.mark.parametrize("template", [None, "", "  ", "off", " OFF ", "on", "On", "{a}", "{a},{b}", " {A} , {b} ",
                                      "{a},{a}", "on,{a}", "off,{a}", "{a},foo", "foo", ",", "1", "true", "yes", "0",
                                      "{a};{b}", "{a} {b}"])
def test_the_parser_is_anchor_verifys_word_for_word(monkeypatch, caplog, template):
    """Manager 098 / the lead: TOWER_WORLD_POSE_QUARANTINE parses exactly as TOWER_WORLD_ANCHOR_VERIFY does -- the
    same value SHAPE (a part standing for a part) is off, all, or a list, and logged, alike."""
    from tower import config

    def run(env, parts, alias):
        if template is None:
            monkeypatch.delenv(env, raising=False)
        else:
            monkeypatch.setenv(env, template.format(a=parts[0], b=parts[1], A=parts[0].upper()))
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="tower.config"):
            got = alias()
        return ("all" if got == frozenset(parts_all(env)) else len(got)), bool(caplog.records)

    def parts_all(env):
        return (config.WORLD_POSE_QUARANTINE_PARTS if env == ENV
                else config.WORLD_ANCHOR_VERIFY_PARTS)

    ours = run(ENV, ("path", "seal"), config.world_pose_quarantine_setting)
    # `scale` and `motion`: two anchor-verify parts that imply nothing, like every pose-quarantine part
    theirs = run(config.WORLD_ANCHOR_VERIFY_ENV, ("scale", "motion"), config.world_anchor_verify_setting)
    assert ours == theirs


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


def test_a_v4_a_96_pose_room_with_8_impossible_poses_lists_0_of_them_and_all_88_good_ones(tmp_path, monkeypatch):
    """RULE.md (a') v4 (manager 102 §2; RVPQ HIGH-2). 8 impossible poses of 96 is 8.3 %: the surface gate's p95 is
    then itself an impossible pose, so its 2.5 x p95 detachment term cleared all 8 (round 1 pinned that), and a larger
    share trips its 5 % stand-down. The viewer list drops both: 0 of the 8, all 88 good ones, at both levels -- the
    pose set and the page's camera path."""
    room = {k: p for k, p in Q.bed_solution().poses.items() if p["component"] == 0}
    bed = {f"{F.SID}:{i:08d}" for i in range(*Q.BED)}
    kept = viewable_poses(room)
    assert len(room) == 96 and not set(kept) & bed and set(kept) == set(room) - bed
    # the same through `_camera_path` (the list the pages carry: 88 poses, every one near, decimation step 1)
    monkeypatch.setenv(ENV, "path")
    path = Q.camera_path_of(tmp_path, Q.bed_solution())
    assert len(path) == 88 and all(np.linalg.norm(c[:3]) < 100 for c in path)
    monkeypatch.delenv(ENV)
    assert sum(1 for c in Q.camera_path_of(tmp_path / "off", Q.bed_solution()) if np.linalg.norm(c[:3]) > 100) == 8


def test_a_v4_the_surface_gate_itself_is_unchanged():
    """Only the viewer list drops the stand-downs: the SURFACE's reconstruction gate still clears the 8 (its
    detachment term), as it always has."""
    from tower.world_builder.surface import robust_pose_outliers

    room = {k: p for k, p in Q.bed_solution().poses.items() if p["component"] == 0}
    C = [-np.asarray(p["rotation"]).reshape(3, 3).T @ np.asarray(p["translation"]) for p in room.values()]
    assert robust_pose_outliers(np.asarray(C)).gated == 0


@pytest.mark.parametrize("n_far, far_expected", [(5, 5), (40, 40)])
def test_a_v4_any_minority_share_of_impossible_poses_is_dropped(n_far, far_expected):
    """A clump as tight as a sub-model flown off whole (all at ~one distance, so its own p95 is itself): 5 or 40 of 88
    good poses. The median holds while the impossible poses are a minority, and every one is dropped."""
    entries = [(f"{F.SID}:{i:08d}", c, 60, 0) for i, c in enumerate(Q.ring_centres(88, radius=1.7))]
    entries += [(f"{F.SID}:{100 + k:08d}", (2000.0 + k, 0.0, 0.0), 60, 0) for k in range(n_far)]
    kept = viewable_poses(Q.pose_solution(entries).poses)
    assert len(kept) == 88 and not any(int(k.split(":")[1]) >= 100 for k in kept)


def test_a_v4_the_floor_is_kept_under_eight_poses():
    entries = [(f"{F.SID}:{i:08d}", (float(i), 0.0, 0.0), 60, 0) for i in range(6)]
    entries.append((f"{F.SID}:{99:08d}", (1.0e6, 0.0, 0.0), 60, 0))
    assert len(viewable_poses(Q.pose_solution(entries).poses)) == 7        # 7 poses: a median describes nothing


@pytest.mark.parametrize("value, expect_today", [(None, True), ("path", False), ("riders,seal", True)])
def test_a_v4_no_room_component_is_an_empty_path_only_with_path_on(tmp_path, monkeypatch, value, expect_today):
    """RVPQ MED (manager 102 §3): a solution with only supported component-1 poses (an area's, say) gave a room
    page that area's camera path through `_room_poses`' fallback. With `path` on it is empty; off, today's list."""
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    entries = [(f"{F.SID}:{i:08d}", c, 60, 1) for i, c in enumerate(Q.ring_centres(20))]
    path = Q.camera_path_of(tmp_path, Q.pose_solution(entries))
    assert (tmp_path / "w1" / "solve" / F.SID / "solution.npz").is_file()
    assert len(path) == (20 if expect_today else 0)


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


def test_c_v3_a_flag_inside_one_group_seals_nothing_whatever_its_verdict():
    """RULE.md (c') v3 (manager 100): v2's internal-flag clause is withdrawn. A flag with both ends in one group -- the
    merge of small runs joined it across the flag -- says one of its two poses is wrong, not that the group is. On walk
    3 the clause sealed nine probably-correct desk keyframes (group 62-73, flag 70->72, 2.92 m in 0.53 s)."""
    for verdicts in ([C, U, C, C], [U, U, U, U], [C, C, C, C]):
        assert AV.motion_sealed_groups(GROUPS, verdicts, [_flag(11, 12, 5e4)], MP) == {}


def test_c_v3_walk_3s_desk_group_is_not_sealed():
    """Walk 3's replay, reduced: group 62-73 (unverifiable) holds the internal flag 70->72; its neighbours 54-61
    (unverifiable) and 91-110 (confirmed) share no flag with it except 61->62 (unverifiable on both sides)."""
    groups = [np.arange(54, 62), np.array([62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73]), np.arange(91, 111)]
    flags = [_flag(61, 62, 1.81, dt=0.474), _flag(70, 72, 2.921, dt=0.526), _flag(88, 91, 2.66, dt=1.089)]
    assert AV.motion_sealed_groups(groups, [U, U, C], flags, MP) == {}


def test_c_a_flag_from_a_confirmed_group_into_an_unverifiable_one_seals_the_unverifiable_side():
    hit = AV.motion_sealed_groups(GROUPS, [C, U, C, C], [_flag(17, 18, 5e4)], MP)
    assert list(hit) == [1]
    hit = AV.motion_sealed_groups(GROUPS, [C, C, U, C], [_flag(17, 18, 5e4)], MP)
    assert list(hit) == [2]


def test_c_a_flag_between_two_unverifiable_groups_seals_neither():
    assert AV.motion_sealed_groups(GROUPS, [C, U, U, C], [_flag(17, 18, 5e4)], MP) == {}


def test_c_v3_a_flag_between_two_different_confirmed_groups_seals_neither():
    """Review V18, MED-1: the GT shape (A0/A0h 6839fb8f: one speed flag, both ends in confirmed groups). The images
    placed both sides, so nothing is sealed -- a rule that sealed on "the other side is confirmed" alone would seal
    both groups here."""
    assert AV.motion_sealed_groups(GROUPS, [C, C, C, C], [_flag(17, 18, 5e4)], MP) == {}
    assert AV.motion_sealed_groups(GROUPS, [C, C, U, C], [_flag(9, 10, 5e4)], MP) == {}


def test_c_v3_mixed_flags_seal_only_the_unverifiable_side_of_a_confirmed_partner():
    """One U|C flag, one C|U flag, one C|C flag, an internal flag in a confirmed and in an unverifiable group, and a
    head turn: only the unverifiable side of a flag to a confirmed group is sealed, and only by such flags."""
    verdicts = [C, U, C, U]
    flags = [_flag(9, 10, 5e4),                   # C (0) | U (1): seals 1
             _flag(17, 18, 5e4),                  # U (1) | C (2): seals 1 again -- one group, two flags
             _flag(29, 30, 5e4),                  # C (2) | U (3): seals 3
             _flag(3, 4, 5e4),                    # inside 0 (confirmed)
             _flag(31, 32, 5e4),                  # inside 3 (unverifiable): no seal of its own
             _flag(0, 39, 0.5, deg=120.0)]        # C (0) | U (3), a head turn: not a speed
    hit = AV.motion_sealed_groups(GROUPS, verdicts, flags, MP)
    assert sorted(hit) == [1, 3]
    assert [(f["a"], f["b"]) for f in hit[1]] == [(9, 10), (17, 18)]
    assert [(f["a"], f["b"]) for f in hit[3]] == [(29, 30)]
    assert AV.motion_sealed_groups(GROUPS, [C, C, C, C], flags, MP) == {}


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
    # v3: sealed by its ONE boundary flag into the confirmed run after it (77 -> 78), not by the 7 inside it
    assert sg["source"] == AV.SOURCE_MOTION and sg["reason"] == CG.REASON_LINK_CONTRADICTED and sg["flags"] == 1
    assert sg["flag_keyframes"] == [f"{F.SID}:{77:08d}", f"{F.SID}:{78:08d}"]
    assert av["motion_seal"]["speed_flags"] == 8 and av["motion_seal"]["groups"] == 1
    assert av["collateral"]["keyframes"] == 0
    comps = out["components.json"]["components"]
    assert (comps[0]["keyframes"], comps[0]["shown_as"]) == (88, "room")
    sealed = [e for e in comps if set(e["keyframe_ids"]) <= set(bed)]
    assert sum(e["keyframes"] for e in sealed) == 8
    assert all(e["reasons"] == [CG.REASON_LINK_CONTRADICTED] and e["shown_as"] == "none" for e in sealed)
    assert all(out["poses"][k] != 0 for k in bed)
    assert "anchor-motion" not in json.dumps(out["components.json"])      # the source never reaches the wire


def test_c_v3_internal_flags_alone_seal_nothing_when_the_run_after_is_unverifiable(tmp_path, monkeypatch):
    """The v2 rule sealed the stretch here through its 7 internal flags; v3 needs a confirmed partner, and the run
    after the stretch is unverifiable -- so the stretch stays (the rule's stated cost, RULE.md (c'))."""
    monkeypatch.setenv(ENV, "on")
    out = Q.publish_bed(tmp_path, confirm_after=False)
    av = out["record"]["anchor_verify"]
    assert av["sealed_kf"] == 0 and av["motion_seal"]["groups"] == 0 and av["motion_seal"]["speed_flags"] == 8
    assert av["motion_seal_rule"]["rule"].startswith("c' v3")


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
    # documented (review V18, LOW-4; RVPQ LOW): the label reads `placed`, with no supported camera; nothing publishes it
    (comp,) = out["components"]
    assert (comp["state"], comp["supported"], comp["cameras"]) == ("placed", 0, 3)


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


# ---------------------------------------------------------------------------------------------------------------
# manager 098: each part on its own changes only its own output; the others stay today's, value for value


GOVERNS = {"path": "camera_path_w4", "riders": "publish_riders", "seal": "publish_bed_av_on"}


def _outputs(tmp_path):
    return {"camera_path_w4": Q.camera_path_of(tmp_path / "c", Q.walk_like_solution(
                Q.W4_ROOM, Q.W4_MEDIAN_RADIUS, {**Q.W4_BED, **Q.W4_RIDERS})),
            "publish_riders": Q.publish_riders(tmp_path / "r"),
            "publish_bed_av_on": Q.publish_bed(tmp_path / "b")}


# The Tower-internal audit keys each part adds wherever it runs, even when it changes nothing (additive, §2.5-like).
AUDIT_KEYS = {"path": (), "riders": ("pose_quarantine",), "seal": ("motion_seal", "motion_seal_rule")}


def _without(obj, keys):
    if isinstance(obj, dict):
        return {k: _without(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_without(v, keys) for v in obj]
    return obj


@pytest.mark.parametrize("part", ["path", "riders", "seal"])
def test_each_part_alone_changes_only_its_own_output(tmp_path, monkeypatch, part):
    """Every output a part does not govern is today's, value for value, but for the part's own audit keys; the output
    it governs is not. The camera path here is of a solution NO GATE PUBLISHED: `riders` and `seal` change the
    published solution, and a path read FROM it follows it (review V18, LOW-3) --
    `test_riders_and_seal_reach_the_path_through_the_published_solution` pins that coupling."""
    monkeypatch.delenv("TOWER_WORLD_ANCHOR_VERIFY", raising=False)
    monkeypatch.setenv(ENV, part)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    exact = (golden["cv2"], golden["numpy"]) == (cv2.__version__, np.__version__)
    now = json.loads(json.dumps(_outputs(tmp_path), sort_keys=True, default=str))
    for key, value in now.items():
        want = golden["outputs"][key]
        got = _without(value, AUDIT_KEYS[part])
        same = got == want if exact else _round_floats(got) == _round_floats(want)
        assert same == (key != GOVERNS[part]), (part, key)


def _published_path(root):
    from tower.world_builder.surface_render import _camera_path

    return [tuple(c[:3]) for c in _camera_path(F.Store(root), "w1", F.SID)]


@pytest.mark.parametrize("part", ["path", "riders", "seal"])
def test_riders_and_seal_reach_the_path_through_the_published_solution(tmp_path, monkeypatch, part):
    """Review V18, LOW-3: the parts are NOT disjoint. `path` is render-only; `riders` and `seal` change the PUBLISHED
    solution -- an unattached rider or a sealed camera leaves component 0 -- and every reader of component 0 follows
    it, the camera path included. Pinned exactly, per part, against the same publish with the switch off:
      path:   both published solutions are today's; their paths lose the riders (floor) and the 8 bed poses (radius);
      riders: the riders' path loses exactly the 3 unattached riders; the bed's is today's;
      seal:   the bed's path loses exactly the 8 sealed poses; the riders' is today's."""
    monkeypatch.delenv("TOWER_WORLD_ANCHOR_VERIFY", raising=False)
    monkeypatch.delenv(ENV, raising=False)
    off_r, off_b = Q.publish_riders(tmp_path / "off_r"), Q.publish_bed(tmp_path / "off_b")
    off_rp, off_bp = _published_path(tmp_path / "off_r"), _published_path(tmp_path / "off_b")
    monkeypatch.setenv(ENV, part)
    on_r, on_b = Q.publish_riders(tmp_path / "on_r"), Q.publish_bed(tmp_path / "on_b")
    on_rp, on_bp = _published_path(tmp_path / "on_r"), _published_path(tmp_path / "on_b")
    riders = [f"{F.SID}:{Q.RIDER_N + k:08d}" for k in range(len(Q.RIDERS))]
    unattached = {k for k in riders if on_r["poses"][k] != 0} if part == "riders" else set()
    bed = {f"{F.SID}:{i:08d}" for i in range(*Q.BED)}
    assert len(off_rp) == Q.RIDER_N + len(Q.RIDERS) and len(off_bp) == 96          # today: every component-0 pose

    def centres(sol_poses, kids):
        return {tuple(round(float(v), 5) for v in -np.asarray(sol_poses[k]["rotation"]).reshape(3, 3).T
                      @ np.asarray(sol_poses[k]["translation"])) for k in kids}

    sol_r = Q.rider_solution(Q.RIDER_N, Q.RIDERS).poses
    sol_b = Q.bed_solution().poses
    if part == "path":
        assert on_r["poses"] == off_r["poses"] and on_b["poses"] == off_b["poses"]
        assert set(off_rp) - set(on_rp) == centres(sol_r, riders) and len(on_rp) == Q.RIDER_N
        assert set(off_bp) - set(on_bp) == centres(sol_b, bed) and len(on_bp) == 88
    elif part == "riders":
        assert len(unattached) == 3 and on_b["poses"] == off_b["poses"] and on_bp == off_bp
        assert set(off_rp) - set(on_rp) == centres(sol_r, unattached) and len(on_rp) == len(off_rp) - 3
    else:
        assert on_r["poses"] == off_r["poses"] and on_rp == off_rp
        assert {k for k in bed if on_b["poses"][k] != 0} == bed
        assert set(off_bp) - set(on_bp) == centres(sol_b, bed) and len(on_bp) == 88


def test_part_path_alone_filters_the_list(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "path")
    path = _outputs(tmp_path)["camera_path_w4"]
    assert len(path) == 177 and not any(np.linalg.norm(c[:3]) > 100 for c in path)


def test_part_riders_alone_unplaces_the_unattached_riders(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "riders")
    out = Q.publish_riders(tmp_path)
    riders = [f"{F.SID}:{Q.RIDER_N + k:08d}" for k in range(len(Q.RIDERS))]
    assert [out["poses"][k] == 0 for k in riders] == [False, False, True, False]
    assert out["record"]["pose_quarantine"] == {"rider_min_shared": 3, "riders_unplaced": 3}


def test_part_seal_alone_seals_the_bed_stretch(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "seal")
    av = Q.publish_bed(tmp_path)["record"]["anchor_verify"]
    assert av["sealed_kf"] == 8 and av["motion_seal"]["groups"] == 1


# ---------------------------------------------------------------------------------------------------------------
# review V17, LOW-2: with `riders` on, the room must not flip on a tie. The tie-break in supported cameras counts
# TOTAL cameras, riders included; unplacing riders used to change that count.


def _tie_model(sizes, riders, seed=0):
    """Solver components of `sizes` supported cameras each (a chain: gaps 1 and 2 share 12 points; 40 points of
    their own), then RIDERS [(component, points shared with that component's first camera)], each with 1 own point.
    Every link honoured, every level 0. Returns (model, links, rotations, levels)."""
    obs_i, obs_p, comp_of, p = [], [], [], 0
    firsts, idx = [], 0
    for c, n in enumerate(sizes):
        firsts.append(idx)
        for j in range(n):
            comp_of.append(c)
            for _ in range(40):
                obs_i.append(idx + j)
                obs_p.append(p)
                p += 1
            for d in (1, 2):
                if j + d < n:
                    for _ in range(12):
                        obs_i += [idx + j, idx + j + d]
                        obs_p += [p, p]
                        p += 1
        idx += n
    shared_pool = {c: [pt for i, pt in zip(obs_i, obs_p) if i == firsts[c]] for c in range(len(sizes))}
    for c, k in riders:
        r = len(comp_of)
        comp_of.append(c)
        for pt in shared_pool[c][:k]:
            obs_i.append(r)
            obs_p.append(pt)
        obs_i.append(r)
        obs_p.append(p)
        p += 1
    n = len(comp_of)
    names = [F.name(i) for i in range(n)]
    n_obs = np.bincount(np.asarray(obs_i), minlength=n)
    model = CG.SolveModel(names=names, component=np.asarray(comp_of, np.int64),
                          R_cw=np.stack([F.rz(3.0 * i) for i in range(n)]), n_obs=n_obs,
                          obs_image=np.asarray(obs_i, np.int64), obs_point=np.asarray(obs_p, np.int64), n_points=p)
    links = {}
    idx = 0
    for n_c in sizes:
        for j in range(n_c):
            for d in (1, 2):
                if j + d < n_c:
                    links[(F.name(idx + j), F.name(idx + j + d))] = 100
        idx += n_c
    rots = {(a, b): F.rz(3.0 * int(b[:8])) @ F.rz(3.0 * int(a[:8])).T for a, b in links}
    levels = {F.name(i): 0.0 for i in range(sum(sizes))}
    return model, links, rots, levels


def _room(res, model):
    return sorted(nm for i, nm in enumerate(model.names) if model.n_obs[i] >= 30 and res["labels"][nm] == 0)


def test_b_v17_low2_a_tie_in_supported_cameras_does_not_flip_the_room():
    """V17's scenario: labels A and B hold 12 supported cameras each; A has 3 unattached riders, B 2 attached ones.
    Today A is the room (15 cameras against 14); with the rider hook it still is."""
    model, links, rots, levels = _tie_model((12, 12), [(0, 0), (0, 0), (0, 1), (1, 5), (1, 8)])
    assert model.n_obs[24:].tolist() == [1, 1, 2, 6, 9]                   # riders, all under 30
    off = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True)
    on = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True,
                       rider_min_shared=CP.RIDER_MIN_SHARED)
    assert _room(off, model) == [F.name(i) for i in range(12)]            # A
    assert _room(on, model) == _room(off, model)
    assert [on["labels"][F.name(i)] for i in (24, 25, 26)] != [0, 0, 0]  # A's riders are unplaced ...
    assert all(on["labels"][F.name(i)] == off["labels"][F.name(i)] for i in (27, 28))  # ... B's stay attached


def _rand_model(rng):
    """A random small solve for the LOW-2 differential (review V18, MED-2, after its `rand_low2.py`): 1-4 solver
    components of 0-10 supported cameras on a chain whose links are each DROPPED with p = 0.2 (so a component can hold
    several groups and several rounds), a level step now and then (scale splits: more labels), riders sharing 0-20
    points with 1-2 RANDOM cameras of their component (so today's label is often NOT the component's first, and equal
    shares make argmax ties), riders with own points only, and sometimes a rider-only component."""
    n_comp = int(rng.integers(1, 5))
    sizes = [int(rng.integers(0 if c > 0 and rng.random() < 0.15 else 2, 11)) for c in range(n_comp)]
    obs_i, obs_p, comp_of, p, idx = [], [], [], 0, 0
    for c, n in enumerate(sizes):
        for j in range(n):
            comp_of.append(c)
            for _ in range(40):
                obs_i.append(idx + j)
                obs_p.append(p)
                p += 1
            for d in (1, 2):
                if j + d < n:
                    for _ in range(12):
                        obs_i += [idx + j, idx + j + d]
                        obs_p += [p, p]
                        p += 1
        idx += n
    n_sup = idx
    own = {i: [pt for ii, pt in zip(obs_i, obs_p) if ii == i][:40] for i in range(n_sup)}
    for _ in range(int(rng.integers(0, 10))):
        c = int(rng.integers(0, n_comp))
        r = len(comp_of)
        comp_of.append(c)
        cams = [i for i in range(n_sup) if comp_of[i] == c]
        if cams:
            share = int(rng.choice([0, 0, 1, 2, 3, 3, 4, 8, 20]))       # equal shares with 2 cameras: an argmax tie
            for cam in rng.choice(cams, size=min(int(rng.integers(1, 3)), len(cams)), replace=False):
                for pt in own[int(cam)][:share]:
                    obs_i.append(r)
                    obs_p.append(pt)
        for _ in range(int(rng.integers(0, 4))):
            obs_i.append(r)
            obs_p.append(p)
            p += 1
    if rng.random() < 0.2:                                              # a rider-only component
        for _ in range(int(rng.integers(1, 4))):
            r = len(comp_of)
            comp_of.append(n_comp)
            obs_i.append(r)
            obs_p.append(p)
            p += 1
    n = len(comp_of)
    model = CG.SolveModel(names=[F.name(i) for i in range(n)], component=np.asarray(comp_of, np.int64),
                          R_cw=np.stack([F.rz(3.0 * i) for i in range(n)]) if n else np.zeros((0, 3, 3)),
                          n_obs=np.bincount(np.asarray(obs_i, np.int64), minlength=n),
                          obs_image=np.asarray(obs_i, np.int64), obs_point=np.asarray(obs_p, np.int64), n_points=p)
    links, idx = {}, 0
    for n_c in sizes:
        for j in range(n_c):
            for d in (1, 2):
                if j + d < n_c and rng.random() < 0.8:
                    links[(F.name(idx + j), F.name(idx + j + d))] = 100
        idx += n_c
    rots = {(a, b): F.rz(3.0 * int(b[:8])) @ F.rz(3.0 * int(a[:8])).T for a, b in links}
    levels = {F.name(i): float(rng.choice([0.0, 0.0, 0.0, 0.0, 0.7])) for i in range(n_sup)}
    return model, links, rots, levels


def test_b_v17_low2_every_supported_label_is_todays_on_random_small_solves():
    """V17 saw the room flip in 25 of 120 random small cases. With the fix, every supported camera keeps TODAY'S
    label -- the room and the label numbers of every other piece -- whatever the riders do: 400 random solves with
    several rounds per component, riders on random cameras and argmax ties (review V18, MED-2: the earlier generator
    gave each component one group and its riders only its first camera, so a mutant `todays_label = first`
    survived it)."""
    rng = np.random.default_rng(1817)
    flips = differing = with_unattached = not_first = 0
    for _ in range(400):
        model, links, rots, levels = _rand_model(rng)
        if model.n == 0:
            continue
        off = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True)
        on = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True,
                           rider_min_shared=CP.RIDER_MIN_SHARED)
        sup = [nm for i, nm in enumerate(model.names) if model.n_obs[i] >= 30]
        flips += _room(on, model) != _room(off, model)
        differing += any(on["labels"][nm] != off["labels"][nm] for nm in sup)
        with_unattached += any(rd.get("quarantined") for rd in on["rounds"])
        # the cases the mutant needs: an unplaced rider whose TODAY label is not its component's first round's
        firsts = {}
        for rd in off["rounds"]:
            firsts.setdefault(rd["source_component"], rd["final_label"])
        not_first += any(off["labels"][nm] != firsts.get(int(model.component[i]))
                         for i, nm in enumerate(model.names)
                         if model.n_obs[i] < 30 and on["labels"][nm] != off["labels"][nm])
    assert (flips, differing) == (0, 0)
    assert with_unattached > 100 and not_first > 10


@pytest.mark.parametrize("value, listed", [(None, 12), ("path", 0)])
def test_a_v4_an_all_rider_solve_has_an_empty_path_with_path_on(tmp_path, monkeypatch, value, listed):
    """RVPQ LOW (manager 102 §4): a solve of riders only keeps one label, 0 (the gate's all-unsupported case), and
    publishes no components record; with `path` on its camera path is empty too."""
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    entries = [(f"{F.SID}:{i:08d}", c, 12, 0) for i, c in enumerate(Q.ring_centres(12))]
    assert len(Q.camera_path_of(tmp_path, Q.pose_solution(entries))) == listed
