"""Fix round 1 of the final scale guard (Codex cross-review RUN review/codex/cx-SCALE-GUARD-XREVIEW-20261007.md).

Each test here fails on 45b6404 (the reviewed head) and passes with its fix:

  F2  a short, grossly wrong excursion the running median swallows (5 cameras at x2.56 inside a normal
      run) is recorded `uncertified`, never published as certified room;
  F3  the pose rejoin keeps the step bounds: a distant island across an untestable gap never rejoins
      (the pose screen the audit records);
  F4  a supported room pose with a missing rotation or translation enters as a non-finite camera, which the
      pose screen records, never left unassessed;
  F6  a withheld final solve leaves the row explicitly NOT CERTIFIED (`finalization.notice`), and so does a
      withheld re-gate;
  F7  a publication the guard changed writes its audit and components record BEFORE the solution: if
      either write fails nothing is published (a withhold), and a withhold says whether its audit landed.
"""

from __future__ import annotations

import json

import pytest

from tests.test_world_builder_final_scale_guard import _cams, _ids, _levels, _walk_centers
from tests.test_world_builder_final_scale_guard_seam import (  # noqa: F401 -- fixtures and helpers
    G_A,
    G_B,
    W3_N,
    _Engines,
    _gated,
    _gated_levels,
    _gated_links,
    _published,
    _room_ids,
    _simple,
    _switches,
    _w3_levels_named,
    _walk,
    colmap,
)
from tower.world_builder import coherence_publish as CP
from tower.world_builder import final_scale_guard as FSG


# F2 ------------------------------------------------------------------------


def _attached_ids(out):
    return {k for u in out["uncertified"] for k in u["keyframe_ids"]}


def test_f2_a_short_gross_excursion_inside_a_normal_run_stays_attached_and_uncertified():
    # The reviewer's probe: 30 normal, 5 at x2.56, 30 normal. The running median never moves.
    out = FSG.assess_v2(_cams(65), _levels(65, [(30, 35, 2.56)]))
    assert out["pieces"] == []
    assert _attached_ids(out) == _ids(30, 35)
    assert out["scale_status"] == FSG.SCALE_UNCERTIFIED


@pytest.mark.parametrize("ratio", [0.3, 1.5, 2.56, 4.27])
def test_f2_excursions_on_either_side(ratio):
    out = FSG.assess_v2(_cams(80), _levels(80, [(40, 44, ratio)]))
    assert out["pieces"] == [] and _attached_ids(out) == _ids(40, 44)


def test_f2_three_cameras_make_an_excursion_two_do_not_and_unmeasured_ones_do_not_break_it():
    out = FSG.assess_v2(_cams(80), _levels(80, [(40, 43, 2.56)]))
    assert _attached_ids(out) == _ids(40, 43)
    out = FSG.assess_v2(_cams(80), _levels(80, [(40, 42, 2.56)]))
    assert _attached_ids(out) == set()
    # 40, 41, (42 unmeasured), 43: one excursion of three measured cameras, the gap inside it
    out = FSG.assess_v2(_cams(80), _levels(80, [(40, 44, 2.56)], missing={42}))
    assert _attached_ids(out) == _ids(40, 44)


def test_f2_a_borderline_wobble_is_not_an_excursion():
    # 5 cameras at x1.3: beyond the screen, inside the clear boundary -- noise-sized, not gross.
    out = FSG.assess_v2(_cams(80), _levels(80, [(40, 45, 1.3)]))
    assert out["pieces"] == [] and _attached_ids(out) == set()


def test_f2_noise_alone_makes_no_excursion():
    out = FSG.assess_v2(_cams(2000), _levels(2000, [], noise=0.08, seed=11))
    assert out["pieces"] == [] and _attached_ids(out) == set()


# F3 ------------------------------------------------------------------------


def test_f3_a_distant_island_never_rejoins_across_an_untestable_gap():
    # 0..9 at 0 m; a hard 81 m step; 10..19 at 81 m for a long stay (10 s apart: not testable inside); a
    # hard 81 m step; 20..29 at 162 m. 9 -> 20 is 91 s and 11 journal indices apart: not a verified link.
    centers = [[0.01 * i, 0.0, 0.0] for i in range(10)] + [[81.0 + 0.01 * i, 0.0, 0.0] for i in range(10)]         + [[162.0 + 0.01 * i, 0.0, 0.0] for i in range(10)]
    times = [0.5 * i for i in range(10)] + [5.0 + 10.0 * i for i in range(10)] + [95.5 + 0.5 * i for i in range(10)]
    out = FSG.assess_v2(_cams(30, centers=centers, times=times), _levels(30, []))
    assert out["pose"]["would_isolate_keyframes"] == 20


def test_f3_a_spike_still_rejoins_within_the_bounds():
    centers = _walk_centers(100)
    centers[50] = [80.0, 0.0, 0.0]
    out = FSG.assess_v2(_cams(100, centers=centers), _levels(100, []))
    assert out["pose"]["would_isolate_keyframes"] == 1


# F4 ------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["rotation", "translation"])
def test_f4_a_supported_room_pose_with_missing_data_is_recorded_non_finite(field):
    from tests.test_world_builder_final_scale_guard_seam import _candidate
    from tower.world_builder.records import CameraIntrinsics, Keyframe

    n = 40
    kfs = [Keyframe(keyframe_id=f"s:{i:08d}", session_id="s", source_seq=i, received_at=1000.0 + 0.5 * i,
                    image_relpath=f"images/{i:08d}.jpg", width=64, height=48, byte_count=1) for i in range(n)]
    cam = CameraIntrinsics(source="self_calibrated", model="pinhole", fx=50.0, fy=50.0, cx=32.0, cy=24.0,
                           calibrated_width=64, calibrated_height=48)
    sol = _candidate(kfs, cam, centers=_walk_centers(n), islands=[(0, n)])
    sol.poses["s:00000020"][field] = None
    name_of = {k.keyframe_id: f"{i:08d}.jpg" for i, k in enumerate(kfs)}
    cams = FSG.room_cameras(sol, kfs, name_of)
    assert "s:00000020" in {c.kid for c in cams}
    out = FSG.assess_v2(cams, _levels(n, []))
    assert out["pose"]["nonfinite"] == 1 and out["pose"]["would_isolate_keyframes"] == 1


def test_f4_a_supported_room_pose_the_keyframe_list_does_not_name_is_recorded_non_finite():
    from tests.test_world_builder_final_scale_guard_seam import _candidate
    from tower.world_builder.records import CameraIntrinsics, Keyframe

    n = 30
    kfs = [Keyframe(keyframe_id=f"s:{i:08d}", session_id="s", source_seq=i, received_at=1000.0 + 0.5 * i,
                    image_relpath=f"images/{i:08d}.jpg", width=64, height=48, byte_count=1) for i in range(n)]
    cam = CameraIntrinsics(source="self_calibrated", model="pinhole", fx=50.0, fy=50.0, cx=32.0, cy=24.0,
                           calibrated_width=64, calibrated_height=48)
    sol = _candidate(kfs, cam, centers=_walk_centers(n), islands=[(0, n)])
    sol.poses["s:stray"] = dict(sol.poses["s:00000005"])
    cams = FSG.room_cameras(sol, kfs, {k.keyframe_id: f"{i:08d}.jpg" for i, k in enumerate(kfs)})
    assert "s:stray" in {c.kid for c in cams}
    out = FSG.assess_v2(cams, _levels(n, []))
    assert out["pose"]["nonfinite"] == 1 and out["pose"]["would_isolate_keyframes"] == 1


# F6 ------------------------------------------------------------------------


def test_f6_the_withheld_notice_is_one_closed_client_safe_line():
    n = FSG.NOTICE_WITHHELD
    assert "\n" not in n and "/" not in n and "\\" not in n and len(n) < 700
    assert CP.client_safe_detail(n, max_chars=CP.FINALIZATION_TEXT_MAX_CHARS) == n


WITHHELD_STUB = r'''
import json, sys
print(json.dumps({"solved": False, "withheld": True,
                  "reason": "the final scale guard withheld the final solve (the room has 3 cameras with a "
                            "metric ratio; 10 are needed to certify it); nothing was published, and the "
                            "solution published before stands",
                  "final_scale_guard": {"decision": "withheld"}}))
'''


def test_f6_the_builder_marks_a_withheld_final_solve_not_certified_on_the_row(tmp_path, monkeypatch):
    from tests.test_world_builder_finish_stages_identity import _run_builder
    import scripts.world_build_session as builder_script

    frames, intrinsics = builder_script.synthetic_frames(12, 480, 360)
    rendered = ([f.payload for f in frames], intrinsics)
    out = _run_builder("soft", rendered, tmp_path, monkeypatch, stub_source=WITHHELD_STUB)
    assert out["exit"] == 0
    (session_json,) = list((tmp_path / "worlds").glob("worlds/*/sessions/*/session.json"))
    fin = json.loads(session_json.read_text(encoding="utf-8"))["finalization"]
    assert fin["final_solve"] == "unavailable"
    assert fin["notice"] == FSG.NOTICE_WITHHELD
    assert fin["detail"].startswith("the final scale guard withheld the final solve")


def test_f6_a_withheld_regate_says_not_certified(tmp_path, colmap, monkeypatch):
    _Engines(monkeypatch, n=G_A + G_B, levels=_gated_levels(), islands=[(0, G_A), (G_A, G_A + G_B)],
             links=_gated_links())
    s = _walk(tmp_path / "w", G_A + G_B, colmap, monkeypatch)
    _gated(s)

    def no_room(*a, **k):                # v2 withholds only on a failure (no scale reason withholds)
        raise ValueError("the relabelled candidate has no published room")

    monkeypatch.setattr(FSG, "apply", no_room)
    out = CP.regate_published(s.store, s.world_id, s.session_id)
    assert out["withheld"] is True and out["notice"] == FSG.NOTICE_WITHHELD


# F7 ------------------------------------------------------------------------


def test_f7_a_components_write_failure_publishes_nothing(tmp_path, colmap, monkeypatch):
    _Engines(monkeypatch, n=W3_N, levels=_w3_levels_named())
    s = _walk(tmp_path / "w", W3_N, colmap, monkeypatch)
    before = s.workspace.solution_path.read_bytes()

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(CP, "write_components", broken)
    out = _simple(s)
    assert out["solved"] is False and out["withheld"] is True
    assert "could not be written" in out["reason"]
    assert s.workspace.solution_path.read_bytes() == before            # the relabelled room never landed
    _, comp, audit = _published(s)
    assert comp is None and audit["decision"] == "withheld"


def test_f7_records_land_before_the_solution(tmp_path, colmap, monkeypatch):
    """The components record and the audit are on disk when the solution is written."""
    from tower.world_builder import global_solve as GS

    _Engines(monkeypatch, n=W3_N, levels=_w3_levels_named())
    s = _walk(tmp_path / "w", W3_N, colmap, monkeypatch)
    seen = {}
    real = GS.write_solution

    def spy(workspace, solution):
        seen["components"] = (workspace.root / CP.COMPONENTS_FILENAME).exists()
        seen["audit"] = (workspace.root / FSG.AUDIT_FILENAME).exists()
        return real(workspace, solution)

    monkeypatch.setattr(GS, "write_solution", spy)
    assert _simple(s)["solved"] is True
    assert seen == {"components": True, "audit": True}


def test_f7_a_withhold_says_whether_its_audit_landed(tmp_path, colmap, monkeypatch):
    _Engines(monkeypatch, n=W3_N, levels=_w3_levels_named())
    s = _walk(tmp_path / "w", W3_N, colmap, monkeypatch)

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(FSG, "write_audit", broken)
    out = _simple(s)
    assert out["solved"] is False and out["final_scale_guard"].get("audit_written") is False
