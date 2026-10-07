"""The final-solve scale and pose publication guard (`world_builder/final_scale_guard.py`): the setting
and the PURE assessment.

Design: RUN review/codex/cx-SCALE-GUARD-DESIGN-20261006.md. The OFF identity is pinned in
`test_world_builder_final_scale_guard_off_identity.py` (golden recorded on 324f4a6); the publish seam,
end to end, in `test_world_builder_final_scale_guard_seam.py`.

Pinned here:
  * the setting: off by default, `shadow`, `on`; a typo is off;
  * thresholds [0.8, 1.25] (inclusive) and the clear boundary 0.75 / 1.333, the 10-ratio rule on both
    sides, borderline is `final-scale-uncertified` never `final-scale-outlier`, a short stretch inherits
    only inside the band and between certified plateaus, plateaus never cross gate groups, the reference is
    the dominant core (a contaminated median does not pick the room), the pose screen (hard steps,
    warnings, rejoin, severe and non-finite poses, metres at the reference scale), and the withhold;
  * THE WALK-3 x2.56 BATHROOM on a synthetic fixture shaped like it (820 keyframes; the published bathroom
    729-819 at x3.19 / x2.40 with an unmeasured doorway, the desk stretch 74-83 at x3.13);
  * the contract's reasons and the summary (no figure).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tower import config
from tower.world_builder import coherence_gate as CG
from tower.world_builder import final_scale_guard as FSG

SWITCH = config.WORLD_FINAL_SCALE_GUARD_ENV
P = FSG.GuardParams()
L = math.log


# ---------------------------------------------------------------------------
# the setting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    (None, "off"), ("", "off"), ("  ", "off"), ("0", "off"), ("false", "off"), ("no", "off"), ("off", "off"),
    ("garbage", "off"), ("enforce", "off"), ("shadow", "shadow"), (" SHADOW ", "shadow"),
    ("on", "on"), ("1", "on"), ("true", "on"), ("yes", "on"), ("ON", "on"),
])
def test_the_setting_is_off_unless_asked(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    assert config.world_final_scale_guard_setting() == expected
    assert FSG.mode() == expected
    assert SWITCH == "TOWER_WORLD_FINAL_SCALE_GUARD"


def test_the_params_are_the_designs_and_are_digested():
    assert (P.screen_low, P.screen_high, P.clear_low, P.clear_high, P.min_ratios) == (0.8, 1.25, 0.75, 1.333, 10)
    assert (P.speed_mps, P.speed_margin_m, P.step_max_dt_s, P.step_max_index_gap) == (2.0, 0.3, 2.0, 3)
    assert (P.height_jump_m, P.height_jump_dt_s, P.height_from_room_m, P.flip_deg, P.room_radius_m) == \
        (0.75, 1.0, 1.5, 120.0, 15.0)
    assert len(P.digest()) == 16
    assert FSG.GuardParams(screen_high=1.3).digest() != P.digest()


# ---------------------------------------------------------------------------
# the pure assessment
# ---------------------------------------------------------------------------


def _cams(n, *, centers=None, rotations=None, times=None, group=None):
    out = []
    for i in range(n):
        c = np.asarray(centers[i] if centers is not None else [0.02 * i, 0.0, 0.0], np.float64)
        R = np.asarray(rotations[i] if rotations is not None else np.eye(3), np.float64)
        out.append(FSG.RoomCamera(kid=f"s:{i:08d}", name=f"{i:08d}.jpg", index=i,
                                  t=(times[i] if times is not None else 0.5 * i), R=R, tvec=-R @ c,
                                  group=group(i) if group else "room"))
    return out


def _levels(n, spec, *, noise=0.0, seed=0, missing=()):
    """{name: log ratio}: `spec` is [(start, end_exclusive, ratio)], ratio 1 elsewhere."""
    rng = np.random.default_rng(seed)
    v = np.zeros(n)
    for a, b, r in spec:
        v[a:b] = math.log(r)
    if noise:
        v = v + rng.normal(0.0, noise, n)
    return {f"{i:08d}.jpg": float(v[i]) for i in range(n) if i not in set(missing)}


def _isolated(out):
    return {kid: p["reason"] for p in out["pieces"] for kid in p["keyframe_ids"]}


def _ids(a, b):
    return {f"s:{i:08d}" for i in range(a, b)}


def test_a_coherent_room_is_published_whole():
    out = FSG.assess(_cams(200), _levels(200, [], noise=0.05))
    assert out["decision"] == "publish"
    assert out["pieces"] == []
    assert len(out["room_keyframes"]) == 200


@pytest.mark.parametrize("ratio, reason", [
    (1.25, None), (0.8, None), (1.2, None), (0.85, None),          # inside the inclusive screen band
    (1.26, FSG.REASON_SCALE_UNCERTIFIED), (0.79, FSG.REASON_SCALE_UNCERTIFIED),   # borderline
    (1.333, FSG.REASON_SCALE_UNCERTIFIED), (0.75, FSG.REASON_SCALE_UNCERTIFIED),  # the clear boundary itself
    (1.34, FSG.REASON_SCALE_OUTLIER), (0.74, FSG.REASON_SCALE_OUTLIER),          # clear failures
    (2.56, FSG.REASON_SCALE_OUTLIER), (0.0833, FSG.REASON_SCALE_OUTLIER),
])
def test_the_screen_and_clear_boundaries(ratio, reason):
    out = FSG.assess(_cams(150), _levels(150, [(60, 90, ratio)]))
    got = _isolated(out)
    if reason is None:
        assert got == {}
    else:
        assert got == {k: reason for k in _ids(60, 90)}
        (piece,) = out["pieces"]
        assert piece["basis"] == "scale" and piece["measured"] == 30


def test_ratio_reason_is_decided_in_log():
    # The plateau's own log ratio decides, so a boundary value is decided once.
    assert FSG.ratio_reason(L(1.25), P) is None and FSG.ratio_reason(L(0.8), P) is None
    assert FSG.ratio_reason(L(0.75), P) == FSG.REASON_SCALE_UNCERTIFIED
    assert FSG.ratio_reason(L(1.333), P) == FSG.REASON_SCALE_UNCERTIFIED
    assert FSG.ratio_reason(L(1.3331), P) == FSG.REASON_SCALE_OUTLIER
    assert FSG.ratio_reason(L(0.7499), P) == FSG.REASON_SCALE_OUTLIER


def test_borderline_is_never_called_an_outlier_whatever_its_interval():
    # x1.2525: the walk-6 simple bathroom (84 measured), 0.0025 beyond the screen.
    out = FSG.assess(_cams(400), _levels(400, [(300, 384, 1.2525)]))
    (piece,) = out["pieces"]
    assert piece["reason"] == FSG.REASON_SCALE_UNCERTIFIED
    assert piece["ci"][0] > 1.0 and piece["ci_excludes_one"] is True      # recorded, not decisive


@pytest.mark.parametrize("n_bad, reason", [(10, FSG.REASON_SCALE_OUTLIER), (9, FSG.REASON_SCALE_UNCERTIFIED)])
def test_ten_ratios_certify_a_plateau_and_nine_do_not(n_bad, reason):
    out = FSG.assess(_cams(120), _levels(120, [(60, 60 + n_bad, 3.0)]))
    assert _isolated(out) == {k: reason for k in _ids(60, 60 + n_bad)}


def test_a_short_stretch_inside_the_band_inherits_only_between_certified_plateaus():
    # Unmeasured cameras inside a certified plateau are kept with it.
    out = FSG.assess(_cams(100), _levels(100, [], missing=range(40, 47)))
    assert out["pieces"] == []
    # An unmeasured stretch at the end of the room has one certified neighbour: it is not inherited...
    # ...but it joins the plateau before it (the nearest measured camera), so it is kept.
    out = FSG.assess(_cams(100), _levels(100, [], missing=range(93, 100)))
    assert out["pieces"] == []


def test_a_short_off_level_stretch_is_not_inherited():
    # 7 cameras at x3 between two certified plateaus: a measured deviation with too few ratios.
    out = FSG.assess(_cams(100), _levels(100, [(50, 57, 3.0)]))
    assert _isolated(out) == {k: FSG.REASON_SCALE_UNCERTIFIED for k in _ids(50, 57)}
    row = [r for r in out["plateaus"] if r["first_index"] == 50][0]
    assert row["class"] == FSG.HIGH and row["measured"] == 7 and row["decision"] == "isolated"


def test_a_whole_gate_group_without_ten_ratios_is_an_uncertified_attachment():
    group = lambda i: "anchor:a" if i < 60 else "group:b"  # noqa: E731
    out = FSG.assess(_cams(68, group=group), _levels(68, []), )
    assert _isolated(out) == {k: FSG.REASON_SCALE_UNCERTIFIED for k in _ids(60, 68)}
    # With ten, at the room's level, it stays.
    out = FSG.assess(_cams(70, group=group), _levels(70, []))
    assert out["pieces"] == []


def test_plateaus_do_not_cross_gate_groups():
    group = lambda i: "anchor:a" if i < 50 else "group:b"  # noqa: E731
    out = FSG.assess(_cams(80, group=group), _levels(80, [(50, 80, 2.0)]))
    assert _isolated(out) == {k: FSG.REASON_SCALE_OUTLIER for k in _ids(50, 80)}
    assert {r["group"] for r in out["plateaus"]} == {"anchor:a", "group:b"}


def test_the_reference_is_the_dominant_core_not_a_contaminated_median():
    # 100 cameras at the room's level, 60 at x1.5 and 60 at x2.2: the median of all 220 sits on the x1.5
    # block (which would make it the room); the guard's reference is the dominant level.
    levels = _levels(220, [(100, 160, 1.5), (160, 220, 2.2)])
    assert np.median(list(levels.values())) == pytest.approx(L(1.5))
    out = FSG.assess(_cams(220), levels)
    assert out["reference"]["log"] == pytest.approx(0.0)
    assert set(_isolated(out)) == _ids(100, 220)
    assert set(out["room_keyframes"]) == _ids(0, 100)


def test_the_reference_is_refined_from_the_certified_core():
    # The densest window spans the room and a x1.2 block; the refined reference is the room's own median.
    out = FSG.assess(_cams(300), _levels(300, [(200, 230, 1.2), (250, 300, 3.0)], noise=0.02))
    assert out["reference"]["rounds"] >= 1
    assert set(_isolated(out)) == _ids(250, 300)
    assert abs(out["reference"]["log"] - float(np.median([0.0] * 200 + [L(1.2)] * 30))) < 0.05


def test_fewer_than_ten_room_ratios_withhold():
    out = FSG.assess(_cams(40), _levels(40, [], missing=range(9, 40)))
    assert out["decision"] == "withhold" and "9 cameras" in out["why"]
    out = FSG.assess(_cams(40), _levels(40, [], missing=range(10, 40)))
    assert out["decision"] == "publish"


def test_a_room_whose_every_plateau_is_off_withholds():
    # Two blocks too far apart and too small to make a core: nothing certified (each < 10 ratios).
    out = FSG.assess(_cams(16), _levels(16, [(8, 16, 3.0)]))
    assert out["decision"] == "withhold"


# poses ---------------------------------------------------------------------


def _walk_centers(n, step=0.05):
    return [[step * i, 0.0, 0.0] for i in range(n)]


def test_a_normal_walk_has_no_pose_flag():
    out = FSG.assess(_cams(100, centers=_walk_centers(100)), _levels(100, []))
    assert out["pose"]["flags"] == [] and out["pose"]["warnings"] == [] and out["pieces"] == []


def test_a_one_camera_spike_is_its_own_pose_piece_and_the_walk_rejoins():
    centers = _walk_centers(100)
    centers[50] = [80.0, 0.0, 0.0]          # 81 m away for one frame, then back (walk-6 534->535)
    out = FSG.assess(_cams(100, centers=centers), _levels(100, []))
    assert _isolated(out) == {"s:00000050": FSG.REASON_POSE_OUTLIER}
    assert len(out["room_keyframes"]) == 99
    assert out["pose"]["flag_counts"]["hard-step"] == 2


def test_a_small_speed_exceedance_is_a_warning_only():
    centers = _walk_centers(100)
    for i in range(50, 100):
        centers[i] = [centers[i][0] + 1.5, 0.0, 0.0]    # 1.55 m in 0.5 s: allowance 1.3, hard limit 2.6
    out = FSG.assess(_cams(100, centers=centers), _levels(100, []))
    assert out["pieces"] == []
    assert out["pose"]["warning_counts"] == {"speed": 1}


def test_a_hard_one_sided_jump_isolates_the_smaller_side_as_its_own_piece():
    centers = _walk_centers(100)
    for i in range(70, 100):
        centers[i] = [centers[i][0] + 6.0, 0.0, 0.0]    # 6 m in 0.5 s, and the walk continues there
    out = FSG.assess(_cams(100, centers=centers), _levels(100, []))
    assert _isolated(out) == {k: FSG.REASON_POSE_OUTLIER for k in _ids(70, 100)}
    assert len([p for p in out["pieces"] if p["basis"] == "pose"]) == 1


def test_a_speed_exceedance_corroborated_by_a_vertical_step_is_hard():
    centers = _walk_centers(100)
    for i in range(80, 100):
        centers[i] = [centers[i][0] + 0.5, 1.2, 0.0]    # 1.3 m in 0.5 s, 1.2 m of it vertical
    # camera "up" is -Y of the camera; the room's median up is the world -Y, so Y is vertical.
    out = FSG.assess(_cams(100, centers=centers), _levels(100, []))
    assert set(_isolated(out)) == _ids(80, 100)


def test_a_multi_defect_camera_is_quarantined_and_a_single_defect_is_not():
    centers = _walk_centers(100)
    rots = [np.eye(3) for _ in range(100)]
    rots[30] = np.diag([1.0, -1.0, -1.0])               # flipped (one defect): kept
    rots[60] = np.diag([1.0, -1.0, -1.0])
    centers[60] = [centers[60][0], 3.0, 0.0]            # flipped AND 3 m off the room's height
    times = [0.5 * i for i in range(100)]
    times[60] = times[59] + 3.0                         # not a testable step either side
    times[61:] = [t + 6.0 for t in times[61:]]
    out = FSG.assess(_cams(100, centers=centers, rotations=rots, times=times), _levels(100, []))
    assert _isolated(out) == {"s:00000060": FSG.REASON_POSE_OUTLIER}
    assert out["pose"]["defects"]["s:00000030"] == ["flip"]


def test_a_non_finite_pose_is_quarantined():
    centers = _walk_centers(50)
    cams = _cams(50, centers=centers)
    cams[20].tvec = np.array([np.nan, 0.0, 0.0])
    out = FSG.assess(cams, _levels(50, []))
    assert _isolated(out) == {"s:00000020": FSG.REASON_POSE_OUTLIER}


def test_the_pose_screen_reads_metres_at_the_reference_scale():
    # The same 1.0-unit jump is 1 m at the room's scale (a warning) but 4 m at 4 units per metre... inverted:
    # log(z_sfm/z_metric) = log 0.25 means 0.25 units per metre, so 1 unit is 4 m: hard.
    centers = _walk_centers(100, step=0.01)
    for i in range(50, 100):
        centers[i] = [centers[i][0] + 1.0, 0.0, 0.0]
    out = FSG.assess(_cams(100, centers=centers), _levels(100, [(0, 100, 0.25)]))
    assert set(_isolated(out)) == _ids(50, 100)
    out = FSG.assess(_cams(100, centers=centers), _levels(100, [(0, 100, 1.0)]))
    assert out["pieces"] == []


# ---------------------------------------------------------------------------
# the walk-3 x2.56 bathroom, on a fixture shaped like it
# ---------------------------------------------------------------------------

W3_N = 820
# The historical published solve's testable bad stretches (design, "missed unsafe attachments"): the
# bathroom 729-791 at x3.19 and 796-819 at x2.40 (an unmeasured doorway 792-795 between them), and the
# desk stretch 74-83 at x3.13. The bathroom REGION is x2.56. Everything else is the room, at its level
# with per-camera noise and a sprinkling of cameras with no ratio.
W3_BAD = [(74, 84, 3.1316), (729, 792, 3.1922), (796, 820, 2.3978)]
W3_BATHROOM = range(729, 820)


def _w3_levels(seed=3):
    rng = np.random.default_rng(seed)
    missing = set(range(792, 796)) | set(int(i) for i in rng.choice(np.arange(0, 729), 60, replace=False))
    return _levels(W3_N, W3_BAD, noise=0.08, seed=seed, missing=missing)


def test_the_walk3_bathroom_leaves_the_room_pure():
    out = FSG.assess(_cams(W3_N, centers=_walk_centers(W3_N, step=0.01)), _w3_levels())
    assert out["decision"] == "publish"
    room = set(out["room_keyframes"])
    iso = _isolated(out)
    bathroom = {f"s:{i:08d}" for i in W3_BATHROOM}
    assert not (room & bathroom), "a bathroom keyframe stayed in the room"
    assert bathroom <= set(iso)
    assert {iso[k] for k in bathroom} == {FSG.REASON_SCALE_OUTLIER}
    assert _ids(74, 84) <= set(iso)
    # ...and nothing else left the room.
    assert set(iso) == bathroom | _ids(74, 84)


# contract ------------------------------------------------------------------


def test_the_new_reasons_are_the_contracts_and_not_the_gates():
    assert FSG.REASONS == ("final-scale-outlier", "final-scale-uncertified", "final-pose-outlier")
    assert not set(FSG.REASONS) & set(CG.REASONS)


def test_the_withheld_sentence_is_one_line_without_a_path():
    s = FSG.withheld_reason({"why": "C:\\x\\y line one\nline two"})
    assert "\n" not in s
    s = FSG.withheld_reason({})
    assert s == FSG.WHY_WITHHELD.format("no certifiable room")


def test_the_summary_carries_no_figure_and_no_wall_clock_stamp():
    out = FSG.summary({"pieces": [{"reason": FSG.REASON_SCALE_OUTLIER, "keyframe_ids": ["a", "b"],
                                   "ratio": 2.56}], "params_digest": P.digest()},
                      mode_="on", decision="published", seconds=1.23456)
    assert out == {"id": FSG.GUARD_ID, "mode": "on", "decision": "published", "params_digest": P.digest(),
                   "pieces_isolated": 1, "keyframes_isolated": 2, "reasons": {FSG.REASON_SCALE_OUTLIER: 1},
                   "audit": FSG.AUDIT_FILENAME, "seconds": 1.235}
