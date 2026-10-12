"""Fix round 1 of the final scale guard (Codex cross-review RUN review/codex/cx-SCALE-GUARD-XREVIEW-20261007.md).

Each test here fails on 45b6404 (the reviewed head) and passes with its fix:

  F1  off imports nothing new: the guard's module is not even imported on the off path (fresh process);
  F2  a short, grossly wrong excursion the running median swallows (5 cameras at x2.56 inside a normal
      run) is isolated, never published as certified room;
  F3  the pose rejoin keeps the step bounds: a distant island across an untestable gap never rejoins;
  F4  a supported room pose with a missing rotation or translation is quarantined, never left placed;
  F6  a withheld final solve leaves the row explicitly NOT CERTIFIED (`finalization.notice`), and so does a
      withheld re-gate;
  F7  a publication the guard changed writes its audit and components record BEFORE the solution: if
      either write fails nothing is published (a withhold), and a withhold says whether its audit landed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.test_world_builder_final_scale_guard import _cams, _ids, _isolated, _levels, _walk_centers
from tests.test_world_builder_final_scale_guard_seam import (  # noqa: F401 -- fixtures and helpers
    G_A,
    G_B,
    SWITCH,
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

TOWER = Path(__file__).resolve().parents[1]


# F1 ------------------------------------------------------------------------

_PROBE = textwrap.dedent('''
    import os, sys, types
    from pathlib import Path
    root = Path(sys.argv[1])
    from tower.world_builder import coherence_publish as CP
    from tower.world_builder import global_solve  # noqa: F401
    written = []
    ws = types.SimpleNamespace(root=root)
    try:
        out = CP.gate_and_publish(None, "w", "s", ws, types.SimpleNamespace(), final=True, gate=False,
                                  database_path=root / "database.db", keyframes=[],
                                  write=lambda w, s: written.append(1))
        print("OFF_PUBLISH_OK=%s" % (written == [1] and out[1] is None))
    except Exception as exc:  # on/shadow on this empty solution: what matters is the import
        print("RAISED=%s" % type(exc).__name__)
    print("GUARD_IMPORTED=%s" % ("tower.world_builder.final_scale_guard" in sys.modules))
''')


@pytest.mark.parametrize("value, imported", [(None, False), ("", False), ("off", False), ("garbage", False),
                                             ("shadow", True), ("on", True)])
def test_f1_off_never_imports_the_guard(tmp_path, value, imported):
    import os

    env = {k: v for k, v in os.environ.items() if k != SWITCH}
    env.update(PYTHONPATH=str(TOWER), CUDA_VISIBLE_DEVICES="-1")
    if value is not None:
        env[SWITCH] = value
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")
    # Off, the publish is today's write and the guard's module is never imported; on or shadow it is.
    proc = subprocess.run([sys.executable, str(script), str(tmp_path)], cwd=str(TOWER), env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert f"GUARD_IMPORTED={imported}" in proc.stdout
    if not imported:
        assert "OFF_PUBLISH_OK=True" in proc.stdout


def test_f1_off_source_has_no_module_level_guard_import():
    src = (TOWER / "tower" / "world_builder" / "coherence_publish.py").read_text(encoding="utf-8")
    top = [line for line in src.splitlines() if line.startswith(("import ", "from "))]
    assert not any("final_scale_guard" in line for line in top)


# F2 ------------------------------------------------------------------------


def test_f2_a_short_gross_excursion_inside_a_normal_run_is_isolated():
    # The reviewer's probe: 30 normal, 5 at x2.56, 30 normal. The running median never moves.
    out = FSG.assess(_cams(65), _levels(65, [(30, 35, 2.56)]))
    assert _isolated(out) == {k: FSG.REASON_SCALE_UNCERTIFIED for k in _ids(30, 35)}
    (piece,) = out["pieces"]
    assert piece["basis"] == "excursion" and piece["measured"] == 5
    assert set(out["room_keyframes"]) == _ids(0, 30) | _ids(35, 65)


@pytest.mark.parametrize("ratio", [0.3, 1.5, 2.56, 4.27])
def test_f2_excursions_on_either_side(ratio):
    out = FSG.assess(_cams(80), _levels(80, [(40, 44, ratio)]))
    assert set(_isolated(out)) == _ids(40, 44)


def test_f2_three_cameras_make_an_excursion_two_do_not_and_unmeasured_ones_do_not_break_it():
    out = FSG.assess(_cams(80), _levels(80, [(40, 43, 2.56)]))
    assert set(_isolated(out)) == _ids(40, 43)
    out = FSG.assess(_cams(80), _levels(80, [(40, 42, 2.56)]))
    assert _isolated(out) == {}
    # 40, 41, (42 unmeasured), 43: one excursion of three measured cameras, the gap inside it
    out = FSG.assess(_cams(80), _levels(80, [(40, 44, 2.56)], missing={42}))
    assert set(_isolated(out)) == _ids(40, 44)


def test_f2_a_borderline_wobble_is_not_an_excursion():
    # 5 cameras at x1.3: beyond the screen, inside the clear boundary -- noise-sized, not gross.
    out = FSG.assess(_cams(80), _levels(80, [(40, 45, 1.3)]))
    assert out["pieces"] == []


def test_f2_noise_alone_makes_no_excursion():
    out = FSG.assess(_cams(2000), _levels(2000, [], noise=0.08, seed=11))
    assert out["pieces"] == []


# F3 ------------------------------------------------------------------------


def test_f3_a_distant_island_never_rejoins_across_an_untestable_gap():
    # 0..9 at 0 m; a hard 81 m step; 10..19 at 81 m for a long stay (10 s apart: not testable inside); a
    # hard 81 m step; 20..29 at 162 m. 9 -> 20 is 91 s and 11 journal indices apart: not a verified link.
    centers = [[0.01 * i, 0.0, 0.0] for i in range(10)] + [[81.0 + 0.01 * i, 0.0, 0.0] for i in range(10)] \
        + [[162.0 + 0.01 * i, 0.0, 0.0] for i in range(10)]
    times = [0.5 * i for i in range(10)] + [5.0 + 10.0 * i for i in range(10)] + [95.5 + 0.5 * i for i in range(10)]
    out = FSG.assess(_cams(30, centers=centers, times=times), _levels(30, []))
    room = set(out["room_keyframes"])
    assert room == _ids(0, 10)
    assert not (room & _ids(20, 30))
    assert set(_isolated(out)) == _ids(10, 30)


def test_f3_a_spike_still_rejoins_within_the_bounds():
    centers = _walk_centers(100)
    centers[50] = [80.0, 0.0, 0.0]
    out = FSG.assess(_cams(100, centers=centers), _levels(100, []))
    assert set(_isolated(out)) == {"s:00000050"}


# F4 ------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["rotation", "translation"])
def test_f4_a_supported_room_pose_with_missing_data_is_quarantined_not_left_placed(field):
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
    out = FSG.assess(cams, _levels(n, []))
    assert _isolated(out) == {"s:00000020": FSG.REASON_POSE_OUTLIER}
    published, doc = FSG.apply(sol, out, session_id="s", keyframes=kfs, started_at=1000.0, gate_block=None)
    assert published.poses["s:00000020"]["component"] != 0
    room = [e for e in doc["components"] if e["state"] == "placed"][0]
    assert "s:00000020" not in room["keyframe_ids"]


def test_f4_a_supported_room_pose_the_keyframe_list_does_not_name_is_quarantined():
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
    out = FSG.assess(cams, _levels(n, []))
    assert _isolated(out) == {"s:stray": FSG.REASON_POSE_OUTLIER}


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
    monkeypatch.setenv(SWITCH, "on")
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
    monkeypatch.setenv(SWITCH, "on")
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
    monkeypatch.setenv(SWITCH, "on")
    assert _simple(s)["solved"] is True
    assert seen == {"components": True, "audit": True}


def test_f7_a_withhold_says_whether_its_audit_landed(tmp_path, colmap, monkeypatch):
    _Engines(monkeypatch, n=W3_N, levels=_w3_levels_named())
    s = _walk(tmp_path / "w", W3_N, colmap, monkeypatch)

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(FSG, "write_audit", broken)
    monkeypatch.setenv(SWITCH, "on")
    out = _simple(s)
    assert out["solved"] is False and out["final_scale_guard"].get("audit_written") is False
