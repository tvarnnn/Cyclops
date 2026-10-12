"""The final scale guard v2 (Exp1 slice 4; design RUN review/codex/cx-SCALE-GUARD-V2-DESIGN-20261007.md),
the rule `TOWER_WORLD_FINAL_SCALE_GUARD` (on / shadow) now runs at the publish seam.

Pinned here:
  * the rule: a plateau is isolated only with >= 10 finite ratios AND a median outside the inclusive
    [0.8, 1.25] AND a deterministic 95 % bootstrap interval of its median excluding 1; it is bounded by its
    first and last MEASURED camera (no class carried across an unmeasured tail); everything else stays
    attached and is recorded `uncertified`; no evidence is `uncertified`, never PASS, nothing isolated;
  * the walk-7 shapes, synthetic: x0.18 and x0.17 spans with >= 10 ratios are isolated; a 9-ratio span is
    attached and flagged;
  * the evidence is READ, never made: the live depth cache is read in place under its full identity
    (`live_depth.compatible_live_records`), nothing is written, and an identity mismatch is no evidence;
  * at the seam: the uncertified flag reaches BOTH `final_scale_guard.json` and the placed component in
    `components.json`, and no depth stage runs at publish.

The OFF identity, extended to a world holding a live depth cache, is in
`test_world_builder_final_scale_guard_off_identity.py`.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tests.test_world_builder_final_scale_guard import _cams, _ids, _levels
from tests.test_world_builder_final_scale_guard_seam import (  # noqa: F401 -- fixtures and helpers
    SWITCH,
    _Engines,
    _live_evidence,
    _placed,
    _published,
    _room_ids,
    _simple,
    _switches,
    _walk,
    colmap,
)
from tests.test_world_builder_live_depth import LABEL, _predict_one, walk  # noqa: F401 -- fixture
from tests.test_world_builder_surface_pipeline import SESSION, WORLD
from tower.world_builder import coherence_scale as CS
from tower.world_builder import final_scale_guard as FSG
from tower.world_builder.global_solve import load_solution

P = FSG.GuardParams()

# The walk-7 shapes (design / prereg): two gross spans with >= 10 ratios and one with 9.
W7_N = 400
W7_SPANS = [(100, 115, 0.18), (200, 212, 0.17), (300, 309, 0.18)]


def _w7_levels(seed=7):
    return _levels(W7_N, W7_SPANS, noise=0.05, seed=seed)


def _uncert_ids(out, kind=None):
    return {k for u in out["uncertified"] if kind is None or u["uncertified"] == kind for k in u["keyframe_ids"]}


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


def test_walk7_shapes_isolate_the_ten_ratio_spans_and_flag_the_nine():
    out = FSG.assess_v2(_cams(W7_N), _w7_levels())
    assert out["decision"] == "publish" and out["rule"] == "v2"
    pieces = {(p["first_index"], p["last_index"]): p for p in out["pieces"]}
    assert set(pieces) == {(100, 114), (200, 211)}
    for p in pieces.values():
        assert p["reason"] == FSG.REASON_SCALE_OUTLIER and p["measured"] >= 10 and p["ci_excludes_one"]
    assert pieces[(100, 114)]["ratio"] == pytest.approx(0.18, abs=0.01)
    assert pieces[(200, 211)]["ratio"] == pytest.approx(0.17, abs=0.01)
    # the 9-ratio x0.18 span stays in the room, flagged `short`
    assert _ids(300, 309) <= set(out["room_keyframes"])
    assert _uncert_ids(out, FSG.UNCERTIFIED_SHORT) == _ids(300, 309)
    assert out["scale_status"] == FSG.SCALE_UNCERTIFIED and out["uncertified_keyframes"] == 9


def test_ten_ratios_are_the_floor():
    for n_bad, isolated in ((10, True), (9, False)):
        out = FSG.assess_v2(_cams(120), _levels(120, [(60, 60 + n_bad, 0.18)]))
        assert bool(out["pieces"]) is isolated, n_bad
        assert (_uncert_ids(out) == _ids(60, 60 + n_bad)) is not isolated


def _wide_spread_levels():
    """A 20-camera stretch alternating x0.6 and x1.0 (2:1): its plateau's median is x0.6, outside the band,
    but its bootstrap interval reaches 1 -- out of band and NOT certifiably different."""
    v = {f"{i:08d}.jpg": 0.0 for i in range(200)}
    pattern = [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    for j, low in enumerate(pattern):
        v[f"{100 + j:08d}.jpg"] = math.log(0.6) if low else 0.0
    return v


def test_an_out_of_band_plateau_whose_interval_includes_one_stays_attached():
    out = FSG.assess_v2(_cams(200), _wide_spread_levels())
    assert out["pieces"] == []
    (row,) = [r for r in out["plateaus"] if r["uncertified"] == FSG.UNCERTIFIED_CI]
    assert row["measured"] >= 10 and not (0.8 <= row["ratio"] <= 1.25)
    assert row["ci"][0] <= 1.0 <= row["ci"][1] and row["ci_excludes_one"] is False
    assert len(out["room_keyframes"]) == 200


def test_a_long_unmeasured_tail_is_never_isolated():
    """v1 carried the last class forward over every unmeasured camera after it (walk-5: 365-996 called one
    outlier from 23 ratios). v2 bounds the x0.18 plateau by its last measured camera."""
    n = 320
    out = FSG.assess_v2(_cams(n), _levels(n, [(100, 120, 0.18)], missing=range(120, n)))
    (piece,) = out["pieces"]
    assert set(piece["keyframe_ids"]) == _ids(100, 120)
    assert _ids(120, n) <= set(out["room_keyframes"])
    assert _uncert_ids(out, FSG.UNCERTIFIED_UNMEASURED) == _ids(120, n)


def test_unmeasured_cameras_inside_a_plateau_go_with_it():
    lv = _levels(200, [(100, 130, 0.18)], missing={110, 111, 112})
    (piece,) = FSG.assess_v2(_cams(200), lv)["pieces"]
    assert set(piece["keyframe_ids"]) == _ids(100, 130) and piece["measured"] == 27


def test_no_evidence_is_uncertified_never_pass_and_isolates_nothing():
    out = FSG.assess_v2(_cams(50), {})
    assert out["decision"] == "publish" and out["scale_status"] == FSG.SCALE_UNCERTIFIED
    assert out["pieces"] == [] and len(out["room_keyframes"]) == 50
    assert out["uncertified_keyframes"] == 50
    assert out["uncertified"][0]["uncertified"] == FSG.UNCERTIFIED_NO_EVIDENCE
    assert "PASS" not in json.dumps(FSG._jsonable(out))


def test_a_fully_measured_coherent_room_is_certified():
    out = FSG.assess_v2(_cams(200), _levels(200, [], noise=0.05))
    assert out["scale_status"] == FSG.SCALE_CERTIFIED and out["uncertified"] == [] and out["pieces"] == []


def test_the_pose_screen_is_recorded_and_isolates_nothing():
    cams = _cams(100, centers=[[0.05 * i, 0.0, 0.0] for i in range(50)] + [[80.0, 0.0, 0.0]]
                 + [[0.05 * i, 0.0, 0.0] for i in range(51, 100)])
    out = FSG.assess_v2(cams, _levels(100, []))
    assert out["pieces"] == [] and len(out["room_keyframes"]) == 100
    assert out["pose"]["applied"] is False and out["pose"]["would_isolate_keyframes"] == 1


def test_the_rest_is_retested_after_a_removal():
    out = FSG.assess_v2(_cams(W7_N), _w7_levels())
    assert [r["isolated"] for r in out["retest"]] == [2, 0]


# ---------------------------------------------------------------------------
# the live depth cache, read in place
# ---------------------------------------------------------------------------


def _read_live(walk_, **kw):
    store, _session, keyframe, _ = walk_
    solution = load_solution(store, WORLD, SESSION)
    name_of = {keyframe.keyframe_id: "00000000.jpg"}
    return FSG._live_cache_evidence(store, WORLD, SESSION, solution, database_path="unused.db",
                                    name_of=name_of, **kw)


def test_the_live_cache_is_read_in_place_under_its_full_identity(walk, monkeypatch):  # noqa: F811
    monkeypatch.setenv("TOWER_WORLD_LIVE_DEPTH", "on")
    store = walk[0]
    worker = _predict_one(walk)
    seen = {}

    def metric(solution, name_of, database_path, live_work, ki_of_name):
        # The real sampler over the live files: the live prediction (the fake network's 2.0) is what it reads.
        sampler = CS.DepthStageSampler(live_work, ki_of_name)
        seen["depth"] = float(np.nanmedian(sampler("00000000.jpg", np.array([[1.0, 1.0]]))))
        seen["work"] = live_work
        return {"metric_log": {"00000000.jpg": 0.25}, "cameras_measured": 1}

    dense = store.world_dir(WORLD) / "dense" / SESSION
    log, record = _read_live(walk, live_metric_fn=metric)
    assert log == {"00000000.jpg": 0.25}
    assert record["state"] == "compatible" and record["records_matched"] == 1
    assert record["trust"] == f"trusted:{LABEL}" and len(record["align_sha1"]) == 16
    assert seen == {"depth": 2.0, "work": worker.root / "work"}
    assert not (dense / "work").exists(), "the guard staged files: evidence must be read in place"


def test_a_live_cache_of_another_fov_is_no_evidence(walk, monkeypatch):  # noqa: F811
    monkeypatch.setenv("TOWER_WORLD_LIVE_DEPTH", "on")
    worker = _predict_one(walk)
    doc = json.loads((worker.root / "align.json").read_text(encoding="utf-8"))
    doc["known_fov"] += 0.01
    (worker.root / "align.json").write_text(json.dumps(doc), encoding="utf-8")
    called = []
    log, record = _read_live(walk, live_metric_fn=lambda *a: called.append(a))
    assert log == {} and record["state"] == "incompatible" and called == []


def test_no_live_cache_is_absent(walk, monkeypatch):  # noqa: F811
    monkeypatch.setenv("TOWER_WORLD_LIVE_DEPTH", "on")
    assert _read_live(walk) == ({}, {"state": "absent"})


@pytest.mark.parametrize("mode", [None, "off", "shadow", "On"])
def test_a_live_cache_is_no_evidence_unless_live_depth_is_on(walk, monkeypatch, mode):  # noqa: F811
    """Arm L (shadow) measures the worker only: its cache must never reach the guard (lead gate)."""
    _predict_one(walk)
    if mode is None:
        monkeypatch.delenv("TOWER_WORLD_LIVE_DEPTH", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_LIVE_DEPTH", mode)
    called = []
    log, record = _read_live(walk, live_metric_fn=lambda *a: called.append(a))
    assert log == {} and record["state"] == "live-depth-not-on" and called == []


# ---------------------------------------------------------------------------
# at the seam: the walk-7 shapes through the live product's final solve
# ---------------------------------------------------------------------------


def test_walk7_shapes_at_the_seam_flag_the_placed_component_and_predict_no_depth(tmp_path, colmap, monkeypatch):
    eng = _Engines(monkeypatch, n=W7_N, levels={})
    s = _walk(tmp_path / "w", W7_N, colmap, monkeypatch)
    _live_evidence(monkeypatch, _w7_levels())
    monkeypatch.setenv(SWITCH, "on")
    out = _simple(s)
    assert out["solved"] is True
    meta, comp, audit = _published(s)
    sid = s.session_id
    ids = lambda a, b: {f"{sid}:{i:08d}" for i in range(a, b)}  # noqa: E731
    room = _room_ids(meta)
    assert not (room & (ids(100, 115) | ids(200, 212)))
    assert ids(300, 309) <= room
    status = _placed(comp)["final_scale_guard"]
    assert status["scale_status"] == FSG.SCALE_UNCERTIFIED and status["uncertified_keyframes"] == 9
    assert status["uncertified_spans"] == [{"first_keyframe_id": f"{sid}:00000300",
                                            "last_keyframe_id": f"{sid}:00000308", "keyframes": 9,
                                            "kind": FSG.UNCERTIFIED_SHORT}]
    assert "ratio" not in json.dumps(comp)
    assert audit["assessment"]["scale_status"] == FSG.SCALE_UNCERTIFIED
    assert {k for u in audit["assessment"]["uncertified"] for k in u["keyframe_ids"]} == ids(300, 309)
    assert {e["reason"] for e in comp["components"] if e["state"] == "unplaced"} == {FSG.REASON_SCALE_OUTLIER}
    assert eng.depth_calls == 0 and eng.metric_calls == 0
