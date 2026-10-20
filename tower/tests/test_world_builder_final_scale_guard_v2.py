"""The final scale guard v2 (Exp1 slice 4; design RUN review/codex/cx-SCALE-GUARD-V2-DESIGN-20261007.md),
the rule the publish seam runs.

Pinned here:
  * the rule: a plateau is isolated only with >= 10 finite ratios AND a median outside the inclusive
    [0.8, 1.25] AND a deterministic 95 % bootstrap interval of its median excluding 1; it is bounded by its
    first and last MEASURED camera (no class carried across an unmeasured tail); a SHORT plateau (< 10
    ratios) is isolated only when its median is out of band AND every ratio is beyond the certified core's
    own range on that side (lane D, `_decide_short`); everything else stays attached and is recorded
    `uncertified`; no evidence is `uncertified`, never PASS, nothing isolated;
  * the walk-7 shapes, synthetic: x0.18 and x0.17 spans with >= 10 ratios are isolated; a 9-ratio x0.18 span
    is isolated too against a room whose cameras never scatter that far, and attached and flagged when the
    room itself shows such ratios;
  * the evidence is READ, never made: the live depth cache is read in place under its full identity
    (`live_depth.compatible_live_records`), nothing is written, and an identity mismatch is no evidence;
  * at the seam: the uncertified flag reaches BOTH `final_scale_guard.json` and the placed component in
    `components.json`, and no depth stage runs at publish.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tests.test_world_builder_final_scale_guard import _cams, _ids, _levels, _walk_centers
from tests.test_world_builder_final_scale_guard_seam import (  # noqa: F401 -- fixtures and helpers
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
from tower.world_builder.components import read_components_record
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


def _w7_levels_room_spikes():
    """The walk-7 shapes in a room that itself shows x0.15: four lone cameras (never three in a row, so no
    excursion) inside its certified plateaus. The 9-ratio x0.18 span then lies inside the room's own range."""
    lv = _w7_levels()
    for i in (50, 150, 250, 350):
        lv[f"{i:08d}.jpg"] = math.log(0.15)
    return lv


def test_walk7_shapes_isolate_the_ten_ratio_spans_and_the_nine_beyond_the_rooms_range():
    out = FSG.assess_v2(_cams(W7_N), _w7_levels())
    assert out["decision"] == "publish" and out["rule"] == "v2"
    pieces = {(p["first_index"], p["last_index"]): p for p in out["pieces"]}
    assert set(pieces) == {(100, 114), (200, 211), (300, 308)}
    for key in ((100, 114), (200, 211)):
        p = pieces[key]
        assert p["reason"] == FSG.REASON_SCALE_OUTLIER and p["measured"] >= 10 and p["ci_excludes_one"]
        assert "beyond_core" not in p
    assert pieces[(100, 114)]["ratio"] == pytest.approx(0.18, abs=0.01)
    assert pieces[(200, 211)]["ratio"] == pytest.approx(0.17, abs=0.01)
    # the 9-ratio x0.18 span: every ratio below the room's lowest camera -> isolated, with no interval
    short = pieces[(300, 308)]
    assert short["reason"] == FSG.REASON_SCALE_OUTLIER and short["measured"] == 9 and "ci" not in short
    assert short["beyond_core"] is True and short["ratio_range"][1] < short["core_range"][0]
    assert short["ratio"] == pytest.approx(0.18, abs=0.01)
    assert not (_ids(300, 309) & set(out["room_keyframes"]))
    assert out["uncertified"] == [] and out["scale_status"] == FSG.SCALE_CERTIFIED


def test_walk7_shapes_flag_the_nine_when_the_room_itself_shows_such_ratios():
    out = FSG.assess_v2(_cams(W7_N), _w7_levels_room_spikes())
    assert {(p["first_index"], p["last_index"]) for p in out["pieces"]} == {(100, 114), (200, 211)}
    # the 9-ratio x0.18 span stays in the room, flagged `short`: the room's own x0.15 cameras are lower
    assert _ids(300, 309) <= set(out["room_keyframes"])
    assert _uncert_ids(out, FSG.UNCERTIFIED_SHORT) == _ids(300, 309)
    assert out["scale_status"] == FSG.SCALE_UNCERTIFIED and out["uncertified_keyframes"] == 9
    (row,) = [r for r in out["plateaus"] if r["uncertified"] == FSG.UNCERTIFIED_SHORT]
    assert row["beyond_core"] is False and row["core_range"][0] < row["ratio_range"][0]


def test_ten_ratios_are_the_floor():
    # The interval's floor. The room is noise-free, so its core range is degenerate (one value) and the
    # short-plateau rule cannot isolate the nine either (see the short-plateau tests below).
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
    assert [r["isolated"] for r in out["retest"]] == [3, 0]


# ---------------------------------------------------------------------------
# short plateaus (lane D, 2026-10-20): a plateau with < 10 ratios is isolated only when its median is out of
# band and every one of its ratios lies beyond the certified core's own range on that side, at the guard's own
# level; anything else about it stays as before (attached, `short`)
# ---------------------------------------------------------------------------

SHORT = (120, 126)  # six cameras: the shape of walk 10.2's x5,165 plateau


def _short(ratio, *, n=200, noise=0.05, seed=11, values=None):
    """{name: log ratio}: a room with per-camera `noise` and the span SHORT at `ratio`; `values` then sets the
    span's ratios one by one (None leaves one as it is; NaN and inf are kept as given)."""
    lv = _levels(n, [(SHORT[0], SHORT[1], ratio)], noise=noise, seed=seed)
    for i, r in zip(range(*SHORT), values or ()):
        if r is not None:
            lv[f"{i:08d}.jpg"] = math.log(r) if math.isfinite(r) and r > 0 else r
    return lv


def _own_group(i):
    """The span as a gate group of its own: its plateau is the whole span, whatever its values."""
    return "span" if SHORT[0] <= i < SHORT[1] else "room"


def _summary(out):
    return FSG.summary(out, mode_=FSG.MODE_ON, decision=FSG.DECISION_PUBLISHED, seconds=0.0)


@pytest.mark.parametrize("ratio", [5165.0, 68.8, 20.3, 0.05])
def test_a_short_plateau_every_ratio_beyond_the_rooms_range_is_isolated(ratio):
    out = FSG.assess_v2(_cams(200), _short(ratio))
    (piece,) = out["pieces"]
    assert set(piece["keyframe_ids"]) == _ids(*SHORT) and piece["measured"] == 6
    assert piece["decision"] == "isolated" and piece["reason"] == FSG.REASON_SCALE_OUTLIER
    assert piece["uncertified"] is None and "ci" not in piece and piece["beyond_core"] is True
    lo, hi = piece["core_range"]
    assert lo < 1.0 < hi
    assert piece["ratio_range"][0] > hi if ratio > 1 else piece["ratio_range"][1] < lo
    # out of the room, out of the uncertified list, into the summary's counts; nothing else moves
    assert not (_ids(*SHORT) & set(out["room_keyframes"])) and len(out["room_keyframes"]) == 194
    assert out["uncertified"] == [] and out["scale_status"] == FSG.SCALE_CERTIFIED
    summ = _summary(out)
    assert (summ["pieces_isolated"], summ["keyframes_isolated"], summ["uncertified_keyframes"]) == (1, 6, 0)
    assert summ["reasons"] == {FSG.REASON_SCALE_OUTLIER: 1}
    assert [r["isolated"] for r in out["retest"]] == [1, 0]


def test_one_ratio_inside_the_rooms_range_keeps_a_short_plateau_attached():
    out = FSG.assess_v2(_cams(200, group=_own_group), _short(5165.0, values=[None] * 5 + [1.0]))
    assert out["pieces"] == []
    (row,) = [u for u in out["uncertified"] if u["group"] == "span"]
    assert set(row["keyframe_ids"]) == _ids(*SHORT) and row["uncertified"] == FSG.UNCERTIFIED_SHORT
    assert row["measured"] == 6 and row["class"] == FSG.HIGH and row["ratio"] > 1000
    assert row["beyond_core"] is False
    assert row["core_range"][0] <= row["ratio_range"][0] <= row["core_range"][1]
    assert out["scale_status"] == FSG.SCALE_UNCERTIFIED and out["uncertified_keyframes"] == 6
    # the same span with all six beyond the range is isolated: that one camera is what keeps it
    out = FSG.assess_v2(_cams(200, group=_own_group), _short(5165.0))
    assert {k for p in out["pieces"] for k in p["keyframe_ids"]} == _ids(*SHORT)


def test_a_short_plateau_on_both_sides_of_the_rooms_range_stays_attached():
    # median x20.3 (high), two of its six cameras below the room's lowest: no one common scale error
    out = FSG.assess_v2(_cams(200, group=_own_group), _short(20.3, values=[None] * 4 + [0.05, 0.05]))
    assert out["pieces"] == []
    (row,) = [u for u in out["uncertified"] if u["group"] == "span"]
    assert row["uncertified"] == FSG.UNCERTIFIED_SHORT and row["beyond_core"] is False
    assert row["ratio_range"][0] < row["core_range"][0] and row["ratio_range"][1] > row["core_range"][1]


def test_a_short_plateau_inside_the_band_is_never_isolated():
    """Every ratio beyond a tight room's range, but the median inside the screen band: the long rule would keep
    and certify such a plateau with 10 ratios, so the short rule must not isolate it with fewer."""
    lv = _short(1.2, noise=0.01)
    out = FSG.assess_v2(_cams(200, group=_own_group), lv)
    assert out["pieces"] == []
    (row,) = [u for u in out["uncertified"] if u["group"] == "span"]
    assert row["ratio_range"][0] > row["core_range"][1] and row["ratio"] < P.screen_high
    assert row["uncertified"] == FSG.UNCERTIFIED_SHORT and row["beyond_core"] is False
    long_group = lambda i: "span" if 120 <= i < 132 else "room"  # noqa: E731
    out = FSG.assess_v2(_cams(200, group=long_group), _levels(200, [(120, 132, 1.2)], noise=0.01, seed=11))
    assert out["pieces"] == [] and out["scale_status"] == FSG.SCALE_CERTIFIED


def test_without_a_certified_core_a_short_plateau_stays_short():
    # 15 cameras: nothing has 10 ratios, so there is no core to compare with (as before: attached, `short`)
    out = FSG.assess_v2(_cams(15), _levels(15, [(9, 15, 5165.0)], noise=0.05, seed=11))
    assert out["pieces"] == [] and len(out["room_keyframes"]) == 15
    assert out["reference"]["core_measured"] == 0
    assert {u["uncertified"] for u in out["uncertified"]} == {FSG.UNCERTIFIED_SHORT}
    assert not any("beyond_core" in u for u in out["uncertified"])


def test_a_degenerate_core_range_isolates_nothing():
    # a noise-free room: every core camera the same value, a range with one end; no scatter was measured
    out = FSG.assess_v2(_cams(200), _short(5165.0, noise=0.0))
    assert out["pieces"] == []
    (row,) = out["uncertified"]
    assert set(row["keyframe_ids"]) == _ids(*SHORT) and row["uncertified"] == FSG.UNCERTIFIED_SHORT
    assert row["core_range"] == [1.0, 1.0] and row["beyond_core"] is False


def test_non_finite_ratios_are_unmeasured_and_go_with_the_plateau():
    nan, inf = float("nan"), float("inf")
    span = list(range(120, 129))
    lv = _levels(200, [(120, 129, 5165.0)], noise=0.05, seed=11)
    lv["00000122.jpg"], lv["00000125.jpg"] = nan, inf
    out = FSG.assess_v2(_cams(200), lv)
    (piece,) = out["pieces"]
    assert set(piece["keyframe_ids"]) == {f"s:{i:08d}" for i in span} and piece["measured"] == 7
    assert piece["beyond_core"] is True and out["room_measured"] == 198
    # a gross stretch whose every ratio is non-finite has no ratio at all: never a piece
    lv = _levels(200, [], noise=0.05, seed=11)
    for i in range(120, 126):
        lv[f"{i:08d}.jpg"] = nan if i % 2 else -inf
    out = FSG.assess_v2(_cams(200), lv)
    assert out["pieces"] == [] and _ids(120, 126) <= set(out["room_keyframes"])


def _row(*logs):
    return {"values": np.asarray(logs, np.float64), "decision": "attached", "reason": None,
            "uncertified": FSG.UNCERTIFIED_SHORT}


def test_the_short_rule_decides_at_the_guards_own_level_and_refuses_what_it_cannot_read():
    # one camera far out: 1 / C(N + 1, 1) <= (1 - 0.95) / 2 needs a core of 39
    for n_core, isolated in ((38, False), (39, True)):
        row = _row(math.log(50.0))
        FSG._decide_short(row, np.linspace(-0.1, 0.1, n_core), 0.0, P)
        assert row["beyond_core"] is isolated and (row["decision"] == "isolated") is isolated, n_core
    # two cameras pass from the guard's floor of 10 core ratios (1 / C(12, 2) = 1 / 66)
    row = _row(math.log(50.0), math.log(60.0))
    FSG._decide_short(row, np.linspace(-0.1, 0.1, 10), 0.0, P)
    assert row["decision"] == "isolated" and row["reason"] == FSG.REASON_SCALE_OUTLIER and row["uncertified"] is None
    # a ratio on the end of the range is inside it
    for v, isolated in ((2.0, False), (2.0 + 1e-9, True)):
        row = _row(v, 3.0)
        FSG._decide_short(row, np.linspace(-0.1, 2.0, 50), 0.0, P)
        assert row["beyond_core"] is isolated, v
    # nothing to read: a non-finite value, a core under the floor, no value -> no decision, no record
    for row, core in ((_row(math.log(50.0), float("nan")), np.linspace(-0.1, 0.1, 50)),
                      (_row(math.log(50.0)), np.linspace(-0.1, 0.1, 9)),
                      (_row(), np.linspace(-0.1, 0.1, 50)),
                      (_row(math.log(50.0)), np.append(np.linspace(-0.1, 0.1, 50), np.inf))):
        FSG._decide_short(row, core, 0.0, P)
        assert row["decision"] == "attached" and "beyond_core" not in row


def test_the_short_rule_is_gauge_free():
    """A global rescale of the solve moves every log ratio by one constant: no decision changes."""
    def key(out):
        return ([(p["keyframe_ids"], p["reason"], p.get("beyond_core")) for p in out["pieces"]],
                [(u["keyframe_ids"], u["uncertified"], u.get("beyond_core")) for u in out["uncertified"]])

    for n, lv in ((W7_N, _w7_levels()), (W7_N, _w7_levels_room_spikes()), (200, _short(20.3))):
        for shift in (math.log(10.0), -math.log(3.7)):
            a = FSG.assess_v2(_cams(n), lv)
            b = FSG.assess_v2(_cams(n), {k: v + shift for k, v in lv.items()})
            assert key(a) == key(b)


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
    worker = _predict_one(walk)
    doc = json.loads((worker.root / "align.json").read_text(encoding="utf-8"))
    doc["known_fov"] += 0.01
    (worker.root / "align.json").write_text(json.dumps(doc), encoding="utf-8")
    called = []
    log, record = _read_live(walk, live_metric_fn=lambda *a: called.append(a))
    assert log == {} and record["state"] == "incompatible" and called == []


def test_no_live_cache_is_absent(walk, monkeypatch):  # noqa: F811
    assert _read_live(walk) == ({}, {"state": "absent"})


# ---------------------------------------------------------------------------
# at the seam: the walk-7 shapes through the live product's final solve
# ---------------------------------------------------------------------------


def test_walk7_shapes_at_the_seam_flag_the_placed_component_and_predict_no_depth(tmp_path, colmap, monkeypatch):
    # The room's own x0.15 cameras keep the 9-ratio span inside its range, so it stays attached and flagged.
    eng = _Engines(monkeypatch, n=W7_N, levels={})
    s = _walk(tmp_path / "w", W7_N, colmap, monkeypatch)
    _live_evidence(monkeypatch, _w7_levels_room_spikes())
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


def test_walk7_shapes_at_the_seam_isolate_the_short_span_and_every_record_agrees(tmp_path, colmap, monkeypatch):
    """Lane D at the seam: the 9-ratio span beyond the room's range is a piece like the long ones -- its own
    unplaced component, coordinates untouched, every frame kept in the solution -- and the components record,
    the audit and the summary all say the same."""
    _Engines(monkeypatch, n=W7_N, levels={})
    s = _walk(tmp_path / "w", W7_N, colmap, monkeypatch)
    _live_evidence(monkeypatch, _w7_levels())
    assert _simple(s)["solved"] is True
    meta, comp, audit = _published(s)
    sid = s.session_id
    ids = lambda a, b: {f"{sid}:{i:08d}" for i in range(a, b)}  # noqa: E731
    spans = [ids(100, 115), ids(200, 212), ids(300, 309)]
    assert not (_room_ids(meta) & set().union(*spans)) and len(_room_ids(meta)) == W7_N - 36
    # frames kept, coordinates untouched: every keyframe still posed where the solve put it
    assert set(meta["poses"]) == ids(0, W7_N)
    centers = _walk_centers(W7_N, step=0.01)
    for kid in ids(300, 309):
        i = int(kid.rsplit(":", 1)[1])
        assert meta["poses"][kid]["component"] != 0
        assert meta["poses"][kid]["translation"] == pytest.approx([-c for c in centers[i]])
    # components.json: one unplaced `final-scale-outlier` component per piece, nothing flagged on the room
    unplaced = [e for e in comp["components"] if e["state"] == "unplaced"]
    assert sorted(map(frozenset, (e["keyframe_ids"] for e in unplaced)), key=min) == \
        sorted(map(frozenset, spans), key=min)
    assert all(e["reasons"] == [FSG.REASON_SCALE_OUTLIER] for e in unplaced)
    status = _placed(comp)["final_scale_guard"]
    assert status["scale_status"] == FSG.SCALE_CERTIFIED
    assert status["uncertified_keyframes"] == 0 and status["uncertified_spans"] == []
    assert comp["final_scale_guard"]["pieces_isolated"] == 3
    assert "ratio" not in json.dumps(comp) and "core_range" not in json.dumps(comp)
    assert read_components_record(s.store, s.world_id, s.session_id) is not None
    # the audit: the same three pieces, the short one with its range test and no interval
    pieces = audit["assessment"]["pieces"]
    assert [set(p["keyframe_ids"]) for p in pieces] == spans
    short = pieces[2]
    assert short["measured"] == 9 and short["beyond_core"] is True and "ci" not in short
    assert audit["assessment"]["uncertified"] == [] and audit["assessment"]["uncertified_keyframes"] == 0
    # the summary on the solution counts it
    summ = meta["solve"]["final_scale_guard"]
    assert (summ["pieces_isolated"], summ["keyframes_isolated"]) == (3, 36)
    assert summ["reasons"] == {FSG.REASON_SCALE_OUTLIER: 3} and summ["scale_status"] == FSG.SCALE_CERTIFIED
