"""The direct-certificate retrieval admission (`world_builder/retrieval_admission.py`; RUN experiments/C23-CERT,
BRIEF-IMPL.md; manager 144 section 3, 145 section 1) -- on synthetic solves, always.

Pinned here (BRIEF-IMPL section 5):
  * the switch: unset, blank, `off` and garbage are OFF, and OFF is today's publish -- the golden record through
    `gate_and_publish` (N = 1 and the consensus) and `regate_published` byte for byte against today's code path;
    `admit=None` is today's gate; the null projection is the input, with P4 seals and without;
  * the certificate's clauses, one by one (a fan, an exact tie, an unreadable link, a consistent alternative, the
    min-view hull and the H-dominant ambiguity);
  * the candidates (another solver component, a scale mismatch, withheld / sealed / P4 collateral never are);
  * the projection (a follower stays out; the post-checks refuse the anchor lost, a closure break, a changed piece);
  * P4 again (the seal re-gate carries `admit`; an image-contradicted admission is sealed), both entry points, the
    consensus's withhold path, failure containment, the binding invariant (no retrieval link reaches a gate call),
    the per-pair seed and a schedule-free verifier, and the stage-timing hook.

The frozen corpus's decisions (W4 27/27, W5 refused, the control, c10's bed) are in
`test_world_builder_retrieval_admission_corpus.py`, behind `TOWER_C23_RUN_DIR`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tests import wb_anchor_verify_fixtures as F
from tower import config
from tower.world_builder import anchor_verify as AV
from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP
from tower.world_builder import retrieval_admission as RA

ENV = config.WORLD_RETRIEVAL_ADMISSION_ENV
P = RA.AdmissionParams()
GP = CG.GateParams()
GOLDEN = Path(__file__).parent / "golden" / "world_builder_gate_d649f9f.json"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (ENV, config.WORLD_ANCHOR_VERIFY_ENV, config.WORLD_POSE_QUARANTINE_ENV,
                config.WORLD_STAGE_TIMING_ENV):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------------------------------------------
# synthetic evidence


def _rel(R, a, b, err_deg=0.0):
    """R_b_from_a as the solve has it, turned `err_deg` about z."""
    return F.rz(err_deg) @ R[b] @ R[a].T


def new_link(R, a, b, *, err=0.0, hull=0.3, amb=0.0, inl=40):
    """A verified retrieval link a -> b (keyframe indices), `err` degrees off the solve."""
    return {"a": F.name(a), "b": F.name(b), "R": _rel(R, a, b, err), "inl": inl, "hull_a": hull, "hull_b": hull,
            "amb": amb, "noncompact": RA.noncompact(hull, hull, amb, P)}


def fake_retrieval(monkeypatch, links):
    """`retrieve` returns `links` (those between a candidate camera and the room), and records its calls."""
    calls = []

    def retrieve(database_path, workspace_root, cands, room, K, area, params, T):
        cams = {n for c in cands for n in c["members"]}
        calls.append(sorted(cams))
        out = [dict(x) for x in links
               if (x["a"] in cams and x["b"] in room) or (x["b"] in cams and x["a"] in room)]
        T.update(features=0.0, bow=0.0, queries=0.0, verify=0.0)
        return out, {"pairs_queried": len(out), "pairs_verified": len(out), "workers": 1}

    monkeypatch.setattr(RA, "retrieve", retrieve)
    return calls


def island_links(sol, n, pairs, **kw):
    R, _ = F.poses_of(sol, n)
    return [new_link(R, a, b, **kw) for a, b in pairs]


# island 1 (30..49) <-> room (0..29), endpoint-disjoint and on the solve
ISLAND1 = [(30 + k, k) for k in range(6)]


def _gate(tmp_path, monkeypatch, sizes, cross=(), levels=None, offsets=None, components=None):
    n = sum(sizes)
    sol = F.multi_solution(sizes, offsets)
    if components is not None:
        for i, kid in enumerate(sol.keyframe_ids):
            sol.poses[kid]["component"] = int(components[i])
    links, rots = F.multi_links(sizes, cross)
    F.patch_gate_inputs(monkeypatch, links, rots, list(levels if levels is not None else [0.0] * n))
    result = CP.gate_final_solution(F.Store(tmp_path), "w1", F.SID, sol, database_path="db",
                                    keyframes=F.keyframes(n))
    return sol, result


def _admit(tmp_path, result, n):
    return RA.admitted(F.Store(tmp_path), "w1", F.SID, result, database_path="db", keyframes=F.keyframes(n),
                       workspace_root=tmp_path)


def _room(result, n):
    return sorted(int(k[-8:]) for k in CP._room_kids(result.solution, 30))


def _pieces(result):
    return {int(e["keyframe_ids"][0][-8:]): (e["keyframes"], e["reasons"]) for e in result.components["components"]}


# ---------------------------------------------------------------------------------------------------------------
# the switch


@pytest.mark.parametrize("value", [None, "", "   ", "off", "OFF", " Off "])
def test_unset_blank_and_off_are_off(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert config.world_retrieval_admission_setting() is False


@pytest.mark.parametrize("value", ["on", "ON", " On "])
def test_on_is_on(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert config.world_retrieval_admission_setting() is True


@pytest.mark.parametrize("value", ["1", "true", "yes", "onn", "on,off", "enabled"])
def test_garbage_is_off_and_logged(monkeypatch, caplog, value):
    monkeypatch.setenv(ENV, value)
    with caplog.at_level(logging.WARNING, logger="tower.config"):
        assert config.world_retrieval_admission_setting() is False
    assert ENV in caplog.text


@pytest.mark.parametrize("value", [None, "", "off", "garbage"])
def test_off_returns_the_very_object_and_runs_nothing(tmp_path, monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    _, result = _gate(tmp_path, monkeypatch, (30, 20))
    monkeypatch.setattr(RA, "_admit", lambda *a, **k: pytest.fail("the admission ran with the switch off"))
    assert _admit(tmp_path, result, 50) is result


def test_the_params_digest_and_the_gates_bound():
    assert P.honoured_deg == GP.max_link_disagreement_deg                # the gate's tau, one value
    assert (P.words, P.sample, P.seed, P.iterations, P.k_room, P.ratio) == (1024, 200_000, 0, 15, 15, 0.8)
    assert (P.e_prob, P.e_px, P.min_inliers, P.e_px_stored, P.h_px, P.h_dominant) == (0.9999, 1.0, 15, 1.5, 4.0,
                                                                                         0.9)
    assert (P.hull_min, P.amb_max_deg, P.min_group_cameras, P.m_h_min, P.m_a_min) == (0.10, 10.0, 3, 2, 2)
    assert P.digest() == RA.AdmissionParams().digest() and len(P.digest()) == 16
    assert P.digest() != RA.AdmissionParams(k_room=16).digest()
    assert CG.GateParams().digest() == "6ce602286efb999f"                 # the gate's digest, unchanged


# ---------------------------------------------------------------------------------------------------------------
# OFF is today's publish, byte for byte, through both entry points


def _round_floats(obj, nd=9):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, nd) for v in obj]
    return obj


@pytest.mark.parametrize("value", [None, "", "off", " OFF ", "garbage"])
def test_golden_with_the_admission_off_every_output_is_todays(tmp_path, monkeypatch, value):
    """The golden record (the gate's and the publish step's outputs before P4 existed; N = 1 and the consensus):
    unchanged with the admission's switch unset, blank, off or garbage."""
    if value is not None:
        monkeypatch.setenv(ENV, value)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    now = json.loads(json.dumps(F.golden_outputs(tmp_path), sort_keys=True))
    expected = golden["outputs"]
    assert sorted(now) == sorted(expected)
    for key in expected:
        if (golden["cv2"], golden["numpy"]) == (cv2.__version__, np.__version__):
            assert now[key] == expected[key], key
        else:
            assert _round_floats(now[key]) == _round_floats(expected[key]), key
    assert "retrieval_admission" not in json.dumps(now)


def _publish_bytes(tmp_path, monkeypatch, *, consensus: bool):
    """gate_and_publish on a solve the admission WOULD change (island 1 is certifiable), retrieval faked. Returns the
    written solution.json and components.json bytes, and the record."""
    from tower.world_builder.global_solve import SolveWorkspace, write_solution

    sizes = (30, 20) if not consensus else (30, 20, 20, 6, 10)
    n = sum(sizes)
    links, rots = F.multi_links(sizes, () if not consensus else F.CONSENSUS_CROSS)
    F.patch_gate_inputs(monkeypatch, links, rots, [0.0] * n)
    offsets = ({},) if not consensus else F.CONSENSUS_OFFSETS
    cands = [F.multi_solution(sizes, o) for o in offsets]
    pairs = ISLAND1 if not consensus else [(76 + k, k) for k in range(6)]
    fake_retrieval(monkeypatch, island_links(cands[0], n, pairs))
    plan = CP.ConsensusPlan(draws=3, seed=7, map_draw=lambda seed: cands[seed - 7]) if consensus else None
    ws = SolveWorkspace(tmp_path / "w1" / "solve" / F.SID)
    _, record = CP.gate_and_publish(F.Store(tmp_path), "w1", F.SID, ws, cands[0], final=True, gate=True,
                                    database_path="db", keyframes=F.keyframes(n), write=write_solution,
                                    consensus=plan)
    return {"solution.json": F.normalise(json.loads(ws.solution_path.read_text(encoding="utf-8"))),
            "components.json": json.loads((ws.root / CP.COMPONENTS_FILENAME).read_text(encoding="utf-8")),
            "record": F.normalise(json.loads(json.dumps(record, default=str)))}


@pytest.mark.parametrize("consensus", [False, True])
@pytest.mark.parametrize("value", [None, "off", "garbage"])
def test_gate_and_publish_off_is_todays_code_path_on_a_solve_it_would_change(tmp_path, monkeypatch, value,
                                                                              consensus):
    today = RA.admitted
    monkeypatch.setattr(RA, "admitted", lambda store, w, s, result, **kw: result)   # the call site removed
    before = _publish_bytes(tmp_path / "today", monkeypatch, consensus=consensus)
    monkeypatch.setattr(RA, "admitted", today)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    off = _publish_bytes(tmp_path / "off", monkeypatch, consensus=consensus)
    assert off == before
    monkeypatch.setenv(ENV, "on")
    on = _publish_bytes(tmp_path / "on", monkeypatch, consensus=consensus)
    assert on["components.json"] != before["components.json"]           # it would change: the test has power
    assert on["record"]["retrieval_admission"]["state"] == RA.STATE_APPLIED


def _fixed_clock(monkeypatch):
    """A deterministic clock for a literal byte comparison: `time.perf_counter` advances 1 ms per call from the same
    start on every `reset()`, and `time.time` is one constant. Returns `reset`."""
    import time as _time

    state = {"t": 0.0}

    def perf_counter():
        state["t"] += 0.001
        return state["t"]

    monkeypatch.setattr(_time, "perf_counter", perf_counter)
    monkeypatch.setattr(_time, "time", lambda: 1_790_000_000.0)

    def reset():
        state["t"] = 1000.0

    return reset


def _raw(root: Path, *files) -> list:
    """The files' bytes with the run's own root folder spelled out as <ROOT> (each run has its own tmp folder)."""
    out = []
    for f in files:
        b = f.read_bytes()
        for spelling in (str(root), json.dumps(str(root))[1:-1], root.as_posix()):
            b = b.replace(spelling.encode("utf-8"), b"<ROOT>")
        out.append(b)
    return out


@pytest.mark.parametrize("consensus", [False, True])
def test_gate_and_publish_off_writes_the_same_bytes_as_todays_code_path(tmp_path, monkeypatch, consensus):
    """The review's evidence limit (LOW): the OFF comparisons above normalise away timings and paths. Here, on a clock
    that runs the same in both runs, the written solution.json and components.json and the record are compared as
    RAW BYTES (only the run's own tmp folder is spelled out), the switch set to garbage (off, and logged)."""
    from tower.world_builder.global_solve import SolveWorkspace, write_solution

    reset = _fixed_clock(monkeypatch)
    sizes = (30, 20) if not consensus else (30, 20, 20, 6, 10)
    n = sum(sizes)
    links, rots = F.multi_links(sizes, () if not consensus else F.CONSENSUS_CROSS)
    F.patch_gate_inputs(monkeypatch, links, rots, [0.0] * n)
    cands = [F.multi_solution(sizes, o) for o in (({},) if not consensus else F.CONSENSUS_OFFSETS)]
    fake_retrieval(monkeypatch, island_links(cands[0], n, ISLAND1 if not consensus else
                                             [(76 + k, k) for k in range(6)]))

    def publish(root):
        reset()
        plan = CP.ConsensusPlan(draws=3, seed=7, map_draw=lambda seed: cands[seed - 7]) if consensus else None
        ws = SolveWorkspace(root / "w1" / "solve" / F.SID)
        _, record = CP.gate_and_publish(F.Store(root), "w1", F.SID, ws, cands[0], final=True, gate=True,
                                        database_path="db", keyframes=F.keyframes(n), write=write_solution,
                                        consensus=plan)
        return _raw(root, ws.solution_path, ws.root / CP.COMPONENTS_FILENAME) + \
            [json.dumps(record, default=str).replace(json.dumps(str(root))[1:-1], "<ROOT>").encode()]

    today = RA.admitted
    monkeypatch.setattr(RA, "admitted", lambda store, w, s, result, **kw: result)   # the call site removed
    before = publish(tmp_path / "a")
    monkeypatch.setattr(RA, "admitted", today)
    monkeypatch.setenv(ENV, "garbage")
    assert publish(tmp_path / "b") == before
    monkeypatch.setenv(ENV, "on")
    assert publish(tmp_path / "c")[1] != before[1]                           # the test has power


def test_regate_published_off_writes_the_same_bytes_as_todays_code_path(tmp_path, monkeypatch):
    reset = _fixed_clock(monkeypatch)

    def regate(root):
        reset()
        store, ws = _saved_world(root, monkeypatch)
        reset()
        out = CP.regate_published(store, "w1", F.SID)
        return _raw(root, ws.solution_path, ws.root / CP.COMPONENTS_FILENAME) + \
            [json.dumps(out, default=str).replace(json.dumps(str(root))[1:-1], "<ROOT>").encode()]

    today = RA.admitted
    monkeypatch.setattr(RA, "admitted", lambda store, w, s, result, **kw: result)
    before = regate(tmp_path / "a")
    monkeypatch.setattr(RA, "admitted", today)
    monkeypatch.setenv(ENV, "off")
    assert regate(tmp_path / "b") == before
    monkeypatch.setenv(ENV, "on")
    assert regate(tmp_path / "c")[1] != before[1]


def _saved_world(tmp_path, monkeypatch, sizes=(30, 20)):
    from tower.world_builder import global_solve as GS
    from tower.world_builder.store import WorldStore

    n = sum(sizes)
    store = WorldStore(tmp_path)
    sol = F.multi_solution(sizes)
    sol.gate = {"state": CP.GATE_STATE_APPLIED, "masks_applied": True, "metric_available": False,
                "retryable": True, "cause": CP.CAUSE_DEPTH_UNAVAILABLE}
    ws = GS.workspace_for(store, "w1", F.SID)
    GS.write_solution(ws, sol)
    ws.database_path.write_bytes(b"not read: the links are faked")
    links, rots = F.multi_links(sizes, ())
    F.patch_gate_inputs(monkeypatch, links, rots, [0.0] * n)
    kfs = F.keyframes(n)
    monkeypatch.setattr(store, "read_keyframes", lambda w, s: kfs, raising=False)
    monkeypatch.setattr(store, "read_session", lambda w, s: SimpleNamespace(started_at=1000.0, intrinsics=None),
                        raising=False)
    fake_retrieval(monkeypatch, island_links(sol, n, ISLAND1))
    return store, ws


def _regate_bytes(tmp_path, monkeypatch):
    store, ws = _saved_world(tmp_path, monkeypatch)
    out = CP.regate_published(store, "w1", F.SID)
    return {"solution.json": F.normalise(json.loads(ws.solution_path.read_text(encoding="utf-8"))),
            "components.json": json.loads((ws.root / CP.COMPONENTS_FILENAME).read_text(encoding="utf-8")),
            "out": F.normalise(json.loads(json.dumps(out, default=str)))}


@pytest.mark.parametrize("value", [None, "", "off", "garbage"])
def test_regate_published_off_is_todays_code_path(tmp_path, monkeypatch, value):
    today = RA.admitted
    monkeypatch.setattr(RA, "admitted", lambda store, w, s, result, **kw: result)
    before = _regate_bytes(tmp_path / "today", monkeypatch)
    monkeypatch.setattr(RA, "admitted", today)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert _regate_bytes(tmp_path / "off", monkeypatch) == before
    monkeypatch.setenv(ENV, "on")
    on = _regate_bytes(tmp_path / "on", monkeypatch)
    assert on["components.json"] != before["components.json"]
    assert on["solution.json"]["gate"]["retrieval_admission"]["state"] == RA.STATE_APPLIED


def test_a_world_saved_with_the_admission_loads_and_parses_with_it_off(tmp_path, monkeypatch):
    from tower.world_builder import components as COMP
    from tower.world_builder.global_solve import load_solution

    monkeypatch.setenv(ENV, "on")
    store, ws = _saved_world(tmp_path, monkeypatch)
    CP.regate_published(store, "w1", F.SID)
    monkeypatch.setenv(ENV, "off")
    again = load_solution(store, "w1", F.SID)
    assert again is not None and again.gate["retrieval_admission"]["state"] == RA.STATE_APPLIED
    doc = COMP.parse_components(json.loads((ws.root / CP.COMPONENTS_FILENAME).read_text(encoding="utf-8")))
    assert doc is not None
    # and a world saved by today's Tower (no admission record) loads exactly as before
    store2, ws2 = _saved_world(tmp_path / "today", monkeypatch)
    CP.regate_published(store2, "w1", F.SID)
    assert "retrieval_admission" not in load_solution(store2, "w1", F.SID).gate


# ---------------------------------------------------------------------------------------------------------------
# the gate's hook: admit=None is today's gate; the null projection is the input


def _dump(out):
    return json.dumps(out, sort_keys=True, default=str)


def test_admit_none_is_todays_gate():
    model, links, rots, levels = F.mid_stretch_model()
    base = dict(link_rotations=rots, masks_applied=True)
    for hooks in ({}, {"withhold": [F.name(40)], "room": [F.name(0)]},
                  {"seal": {F.name(i): CG.REASON_SCALE_MISMATCH for i in range(25, 35)},
                   "room": [F.name(i) for i in range(60)]},
                  {"rider_min_shared": 3}):
        today = CG.apply_gate(model, links, levels, **base, **hooks)
        assert _dump(CG.apply_gate(model, links, levels, **base, **hooks, admit=None)) == _dump(today)
        assert "admitted" not in _dump(today)


def _published_view(out, model):
    """Per supported camera: room or not; the outside pieces and their reasons."""
    sup = model.n_obs >= 30
    lab = out["labels"]
    reasons = {c["label"]: c["reasons"] for c in out["components"]}
    room = {n for i, n in enumerate(model.names) if sup[i] and lab[n] == 0}
    pieces = {}
    for i, n in enumerate(model.names):
        if sup[i] and lab[n] != 0:
            pieces.setdefault(lab[n], set()).add(n)
    return room, {frozenset(v): reasons[k] for k, v in pieces.items()}


@pytest.mark.parametrize("sealed", [False, True])
def test_the_null_projection_is_the_input(sealed):
    """Allow-list = the input's room (P4's pre-seal room when it sealed), nothing admitted, the riders forced: the
    input's room, pieces and reasons (PREREG C.3's machinery check)."""
    sizes = (30, 20, 8)
    sol = F.multi_solution(sizes)
    links, rots = F.multi_links(sizes, ((49, 50),))
    model = CP.solve_model(sol, {k.keyframe_id: F.name(i) for i, k in enumerate(F.keyframes(58))})
    lv = {F.name(i): 0.0 for i in range(58)}
    hooks = {"rider_min_shared": 3}
    free = CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True, **hooks)
    room0 = sorted(n for n, v in free["labels"].items() if v == 0)
    if sealed:
        hooks.update(seal={F.name(0): CG.REASON_SCALE_MISMATCH}, room=room0)
    base = CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True, **hooks)
    allow = room0 if sealed else sorted(n for n, v in base["labels"].items() if v == 0)
    null = CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True,
                         **dict(hooks, room=allow), admit=set())
    assert _published_view(null, model) == _published_view(base, model)


def test_the_camera_allow_list_bars_an_admitted_groups_follower():
    """F (50..57) has REDUNDANT honoured links to the admitted group A (30..49) -- a closed triangle through A's image
    49, so F is its own block: the camera allow-list (the room plus A) keeps it out; allow-listed too, it would join
    -- the allow-list is what bars it (W4's g283)."""
    sizes = (30, 20, 8)
    sol = F.multi_solution(sizes)
    links, rots = F.multi_links(sizes, ((49, 50), (49, 51)))
    model = CP.solve_model(sol, {k.keyframe_id: F.name(i) for i, k in enumerate(F.keyframes(58))})
    lv = {F.name(i): 0.0 for i in range(58)}
    adm = {F.name(i) for i in range(30, 50)}
    room = [F.name(i) for i in range(30)]
    out = CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True, room=room + sorted(adm),
                        admit=adm, rider_min_shared=3)
    lab = out["labels"]
    assert all(lab[F.name(i)] == 0 for i in range(50))
    assert all(lab[F.name(i)] != 0 for i in range(50, 58))
    admitted = [g for g in out["groups"] if g["first_camera"] == F.name(30)]
    assert admitted and admitted[0]["label"] == 0
    everyone = CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True,
                             room=[F.name(i) for i in range(58)], admit=adm, rider_min_shared=3)
    assert all(v == 0 for v in everyone["labels"].values())
    # and without `admit`, A has no link to the room at all: it stays out whatever the allow-list
    no_admit = CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True, rider_min_shared=3)
    assert all(no_admit["labels"][F.name(i)] != 0 for i in range(30, 58))


def test_admit_acts_only_in_the_round_of_the_allow_listed_room():
    """The hook's contract (C23-CERT's patch): `admit` counts only where the round's reference lies in the camera
    allow-list -- so `admit` without a `room` admits nothing, and an admitted camera is never a way into another
    piece."""
    sizes = (30, 20)
    sol = F.multi_solution(sizes)
    links, rots = F.multi_links(sizes, ())
    model = CP.solve_model(sol, {k.keyframe_id: F.name(i) for i, k in enumerate(F.keyframes(50))})
    lv = {F.name(i): 0.0 for i in range(50)}
    adm = {F.name(i) for i in range(30, 50)}
    out = CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True, admit=adm)
    assert all(out["labels"][F.name(i)] != 0 for i in range(30, 50))
    assert not any(d.get("admitted") for rd in out["rounds"] for d in rd["decisions"])
    out = CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True, admit=adm,
                        room=[F.name(i) for i in range(50)])
    assert all(v == 0 for v in out["labels"].values())


# ---------------------------------------------------------------------------------------------------------------
# the certificate, clause by clause (PREREG A.4)


def _L(*links):
    """Direct links: (g, r, honoured, noncompact, correction deg or None = no readout)."""
    out = []
    for g, r, hon, nc, deg in links:
        D = None if deg is None else F.rz(deg)
        out.append({"src": "new", "a": g, "b": r, "g": g, "r": r, "inl": 30, "has_readout": deg is not None,
                    "res": (float("nan") if deg is None else abs(deg)), "hon": bool(hon), "D": D,
                    "noncompact": bool(nc), "hull_min": 0.3 if nc else 0.05, "hull_max": 0.3, "amb": 0.0})
    return out


def test_two_disjoint_honoured_noncompact_links_certify():
    ok, st = RA.certificate(_L(("g1", "r1", True, True, 0.0), ("g2", "r2", True, True, 1.0)), P)
    assert ok and (st["M_H"], st["mA"], st["mB"], st["C1"], st["C2"], st["C3"]) == (2, 2, 0, True, True, True)


def test_a_fan_into_one_room_image_is_refused():
    ok, st = RA.certificate(_L(("g1", "r1", True, True, 0.0), ("g2", "r1", True, True, 0.0),
                               ("g3", "r1", True, True, 0.0)), P)
    assert not ok and st["M_H"] == 1 and not st["C1"]


def test_an_exact_tie_is_refused():
    # two honoured noncompact links against two contradicted compact ones, all disjoint: H = C = 2
    L = _L(("g1", "r1", True, True, 0.0), ("g2", "r2", True, True, 0.0),
           ("g3", "r3", False, False, 40.0), ("g4", "r4", False, False, 40.0))
    ok, st = RA.certificate(L, P)
    assert not ok and st["H"] == st["C"] == 2.0 and not st["C2"] and st["C1"] and st["C3"]
    # the same with one contradicted link less: H > C
    ok, st = RA.certificate(L[:3], P)
    assert ok and st["H"] > st["C"]


def test_a_link_without_a_readout_counts_as_contradicted():
    L = _L(("g1", "r1", True, True, 0.0), ("g2", "r2", True, True, 0.0),
           ("g3", "r3", False, True, None), ("g4", "r4", False, True, None))
    ok, st = RA.certificate(L, P)
    assert not ok and st["C"] == 2.0 and not st["C2"]
    ok, st = RA.certificate(L[:3], P)
    assert ok and st["C"] == 1.0


def test_c2_weights_by_degree_over_all_links():
    # both honoured links share their group image with two contradicted compact links each: every weight is 1/3
    L = _L(("g1", "r1", True, True, 0.0), ("g2", "r2", True, True, 0.0),
           ("g1", "r3", False, False, 40.0), ("g1", "r4", False, False, 40.0),
           ("g2", "r5", False, False, 40.0), ("g2", "r6", False, False, 40.0))
    ok, st = RA.certificate(L, P)
    assert not ok and st["H"] == pytest.approx(2 / 3, abs=1e-4) and st["C"] == pytest.approx(4 / 3, abs=1e-4)


def test_a_consistent_alternative_correction_as_strong_is_refused():
    # A*: two honoured noncompact links at the solve; B*: two noncompact links agreeing on a 40 deg correction
    L = _L(("g1", "r1", True, True, 0.0), ("g2", "r2", True, True, 0.0),
           ("g3", "r3", False, True, 40.0), ("g4", "r4", False, True, 40.0),
           ("g5", "r5", True, False, 0.0))                                  # compact honoured: weight in H only
    ok, st = RA.certificate(L, P)
    assert not ok and (st["mA"], st["mB"]) == (2, 2) and st["G_B"] == pytest.approx(40.0, abs=0.01)
    assert st["C1"] and st["C2"] and not st["C3"]
    ok, st = RA.certificate(L[:4] + _L(("g6", "r6", True, True, 0.0)) + L[4:], P)
    assert ok and (st["mA"], st["mB"]) == (3, 2)


def test_c3_classifies_the_seed_mean_not_a_recomputed_one():
    """Codex C23x HIGH-1, PREREG A.4 as frozen. Four endpoint-disjoint noncompact links whose corrections turn 30,
    13.3, 13.3 and 2.5 deg about one axis. Seeded at the 30 deg link: S = {30, 13.3, 13.3} (the 2.5 is 27.5 deg
    away), its seed mean G turns 18.85 deg -- a RIVAL correction beyond tau -- and S' around it takes all four links
    (m 4). Seeded at a 13.3 deg link: S = all four, G 14.75 deg, S' all four (m 4). So m(B*) = 4 = m(A*): refused.
    The mean RECOMPUTED over S' (all four, 14.75 deg) would move the rival across 16.8 deg and admit the group."""
    L = _L(("g1", "r1", False, True, 30.0), ("g2", "r2", True, True, 13.3), ("g3", "r3", True, True, 13.3),
           ("g4", "r4", True, True, 2.5))
    ok, st = RA.certificate(L, P)
    assert (st["M_H"], st["H"], st["C"], st["C1"], st["C2"]) == (3, 3.0, 1.0, True, True)       # only C3 decides
    assert (st["mA"], st["nA"], st["mB"]) == (4, 4, 4)
    assert st["G_A"] == pytest.approx(14.75, abs=0.01) and st["G_B"] == pytest.approx(18.85, abs=0.01)
    assert not st["C3"] and not ok
    # the power of the case: the seed mean is past tau, the recomputed mean over the same S' is not
    Ds = np.asarray([x["D"] for x in L])
    assert RA.rot_deg(RA.chordal_mean(Ds[:3])) > P.honoured_deg >= RA.rot_deg(RA.chordal_mean(Ds))
    # without the rival's seed (the 30 deg link alone removed), the same evidence certifies
    ok, st = RA.certificate(L[1:], P)
    assert ok and (st["mA"], st["mB"]) == (3, 0)


def test_compact_honoured_links_never_satisfy_c1():
    # A* is two noncompact links (one honoured at 5 deg, one contradicted at 20 deg, one correction of 12.5 deg), so
    # C3 holds; the honoured NONCOMPACT evidence is one link: C1 fails. A compact honoured link must not rescue it.
    L = _L(("g1", "r1", True, True, 5.0), ("g2", "r2", False, True, 20.0), ("g3", "r3", True, False, 0.0))
    ok, st = RA.certificate(L, P)
    assert st["C3"] and st["mA"] == 2 and st["C2"]
    assert not ok and st["M_H"] == 1 and not st["C1"]


def test_compactness_is_the_min_view_hull_and_ambiguity_counts_only_when_h_dominant():
    assert RA.noncompact(0.5, 0.2, 0.0, P) and not RA.noncompact(0.5, 0.05, 0.0, P)    # min view, not max
    assert RA.ambiguity(0.85, 30.0, P) == 0.0 and RA.ambiguity(0.95, 30.0, P) == 30.0
    assert RA.noncompact(0.5, 0.5, RA.ambiguity(0.85, 30.0, P), P)
    assert not RA.noncompact(0.5, 0.5, RA.ambiguity(0.95, 30.0, P), P)


def test_an_empty_link_set_certifies_nothing():
    ok, st = RA.certificate([], P)
    assert not ok and st["links"] == 0 and st["M_H"] == 0


def test_matching_is_a_maximum_matching():
    rng = np.random.default_rng(3)
    for _ in range(200):
        pairs = {(f"g{rng.integers(6)}", f"r{rng.integers(6)}") for _ in range(rng.integers(0, 14))}
        gs = sorted({g for g, _ in pairs})
        best = 0
        rs_of = {g: sorted(r for gg, r in pairs if gg == g) for g in gs}
        # brute force over assignments
        def rec(i, used):
            if i == len(gs):
                return 0
            out = rec(i + 1, used)
            for r in rs_of[gs[i]]:
                if r not in used:
                    out = max(out, 1 + rec(i + 1, used | {r}))
            return out
        best = rec(0, frozenset())
        assert RA.matching(pairs) == best


# ---------------------------------------------------------------------------------------------------------------
# the candidates (PREREG A.1)


def _cands(result, n, *, sealed=(), collateral=(), withheld=()):
    name_of = {k.keyframe_id: F.name(i) for i, k in enumerate(F.keyframes(n))}
    model = CP.solve_model(result.candidate, name_of)
    idx = model.index()
    r = np.full(model.n, np.nan)
    for nm, v in result.scale["metric_log"].items():
        r[idx[nm]] = v
    room = {nm for nm, v in result.gated["labels"].items() if v == 0}
    return RA.candidates(model, result.gated, r, room, sealed=set(sealed), collateral=set(collateral),
                         withheld=set(withheld), params=P, gp=GP)


def test_a_group_outside_the_room_in_its_component_is_a_candidate(tmp_path, monkeypatch):
    _, result = _gate(tmp_path, monkeypatch, (30, 20))
    cands, diag = _cands(result, 50)
    assert [(c["first_camera"], c["n"]) for c in cands] == [(F.name(30), 20)] and diag == []


def test_another_solver_component_is_never_a_candidate(tmp_path, monkeypatch):
    sol, result = _gate(tmp_path, monkeypatch, (30, 20), components=[0] * 30 + [1] * 20)
    assert _pieces(result)[30][1] == [CG.REASON_SOLVED_SEPARATELY]
    assert _cands(result, 50) == ([], [])
    monkeypatch.setenv(ENV, "on")
    fake_retrieval(monkeypatch, island_links(sol, 50, ISLAND1))
    out = _admit(tmp_path, result, 50)
    assert out.record["retrieval_admission"]["state"] == RA.STATE_NOT_RUN and _room(out, 50) == list(range(30))


def test_a_scale_mismatched_group_is_never_a_candidate(tmp_path, monkeypatch):
    levels = [0.0] * 30 + [math.log(2.0)] * 20
    sol, result = _gate(tmp_path, monkeypatch, (30, 20), levels=levels)
    cands, diag = _cands(result, 50)
    assert cands == [] and [(d["first_camera"], d["scale_ok"]) for d in diag] == [(F.name(30), False)]
    monkeypatch.setenv(ENV, "on")
    calls = fake_retrieval(monkeypatch, island_links(sol, 50, ISLAND1))       # perfect rotation evidence
    out = _admit(tmp_path, result, 50)
    ra = out.record["retrieval_admission"]
    assert ra["state"] == RA.STATE_NOT_RUN and ra["candidates"] == [] and calls == []
    assert ra["scale_refused"][0]["scale_factor"] == pytest.approx(2.0, rel=1e-3)
    assert _room(out, 50) == list(range(30))


def test_a_small_group_without_three_supported_cameras_is_never_a_candidate(tmp_path, monkeypatch):
    _, result = _gate(tmp_path, monkeypatch, (30, 2))
    assert _cands(result, 32) == ([], [])


def test_withheld_sealed_and_collateral_groups_are_never_candidates(tmp_path, monkeypatch):
    _, result = _gate(tmp_path, monkeypatch, (30, 20))
    assert _cands(result, 50, withheld={F.name(30)})[0] == []
    assert _cands(result, 50, sealed={F.name(41)})[0] == []
    assert _cands(result, 50, collateral={F.name(33)})[0] == []
    assert len(_cands(result, 50, sealed={F.name(3)})[0]) == 1             # a seal elsewhere changes nothing


def test_a_fail_safe_is_never_overridden(tmp_path, monkeypatch):
    sol, result = _gate(tmp_path, monkeypatch, (30, 20), levels=[None] * 50)   # no metric scale: attach false
    assert result.record["attach"] is False
    monkeypatch.setenv(ENV, "on")
    fake_retrieval(monkeypatch, island_links(sol, 50, ISLAND1))
    out = _admit(tmp_path, result, 50)
    assert out.record["retrieval_admission"]["state"] == RA.STATE_NOT_RUN
    assert out.components == result.components


# ---------------------------------------------------------------------------------------------------------------
# the whole step, N = 1


def test_a_certified_group_joins_the_room(tmp_path, monkeypatch):
    sol, result = _gate(tmp_path, monkeypatch, (30, 20))
    assert _pieces(result) == {0: (30, []), 30: (20, [CG.REASON_NO_VERIFIED_LINK])}
    monkeypatch.setenv(ENV, "on")
    fake_retrieval(monkeypatch, island_links(sol, 50, ISLAND1))
    out = _admit(tmp_path, result, 50)
    ra = out.record["retrieval_admission"]
    assert ra["state"] == RA.STATE_APPLIED and ra["admitted_keyframes"] == 20
    assert ra["admitted"] == [{"first_keyframe": f"{F.SID}:{30:08d}", "keyframes": 20}]
    (row,) = ra["candidates"]
    assert (row["M_H"], row["H"], row["C"], row["mA"], row["mB"], row["admitted"]) == (6, 6.0, 0.0, 6, 0, True)
    assert ra["room_before"] == 30 and ra["room_after"] == 50 and ra["rider_min_shared"] == CP.RIDER_MIN_SHARED
    assert ra["params_digest"] == P.digest() and set(ra["seconds"]) >= {"candidates", "certificate", "projection",
                                                                         "total"}
    assert _room(out, 50) == list(range(50)) and _pieces(out) == {0: (50, [])}
    assert out.record["components"] == {"placed": 1, "area": 0, "none": 0}
    assert out.record["params_digest"] == GP.digest() and out.record["gate"] == CG.GATE_ID
    assert out.depth is result.depth and out.scale is result.scale and out.candidate is result.candidate
    # privacy: ids and numbers only -- no path, no pixel, no descriptor
    blob = json.dumps(ra)
    assert "\\" not in blob and ".jpg" not in blob and "tmp" not in blob.lower()


def test_an_uncertified_group_is_published_as_it_was(tmp_path, monkeypatch):
    sol, result = _gate(tmp_path, monkeypatch, (30, 20))
    monkeypatch.setenv(ENV, "on")
    fake_retrieval(monkeypatch, island_links(sol, 50, [(30, 0), (31, 0), (32, 0)]))   # a fan: M_H = 1
    out = _admit(tmp_path, result, 50)
    ra = out.record["retrieval_admission"]
    assert ra["state"] == RA.STATE_APPLIED and ra["admitted"] == [] and ra["candidates"][0]["refused_by"] == "C1"
    assert out.components == result.components and out.gated is result.gated and _room(out, 50) == list(range(30))


def test_a_follower_stays_out_and_its_piece_keeps_the_published_reason(tmp_path, monkeypatch):
    """W4's g283: F (50..57) hangs on the admitted group by ONE link. It stays out; the re-gate would now say
    `single-unconfirmed-link`, the input published `no-verified-link` -- the input's is kept (decision (c)), and
    the difference is recorded."""
    sol, result = _gate(tmp_path, monkeypatch, (30, 20, 8), cross=((49, 50),))
    assert _pieces(result) == {0: (30, []), 30: (20, [CG.REASON_NO_VERIFIED_LINK]),
                               50: (8, [CG.REASON_NO_VERIFIED_LINK])}
    monkeypatch.setenv(ENV, "on")
    fake_retrieval(monkeypatch, island_links(sol, 58, ISLAND1))
    out = _admit(tmp_path, result, 58)
    ra = out.record["retrieval_admission"]
    assert ra["state"] == RA.STATE_APPLIED and _room(out, 58) == list(range(50))
    assert _pieces(out) == {0: (50, []), 50: (8, [CG.REASON_NO_VERIFIED_LINK])}
    (kept,) = ra["projection"]["reasons_kept"]
    assert (kept["published"], kept["regate"]) == ([CG.REASON_NO_VERIFIED_LINK],
                                                   [CG.REASON_SINGLE_UNCONFIRMED_LINK])
    follower = [c for c in (out.solution.components or []) if c["gate_reasons"] == [CG.REASON_NO_VERIFIED_LINK]]
    assert follower, "the solution's own components carry the kept reason"


def test_a_piece_that_would_lose_admitted_cameras_is_not_applied(tmp_path, monkeypatch):
    """Decision (c), Codex C23x MED-6: F is redundantly linked to A, so they are ONE input piece; A alone is
    certified. PREREG B.2(c) would publish F with the re-gate's reasons, which here contradict the contract (next
    test): conservative until the reason-contract ruling -- `not-applied`, nothing new is published."""
    sol, result = _gate(tmp_path, monkeypatch, (30, 20, 8), cross=((49, 50), (49, 51)))
    assert _pieces(result) == {0: (30, []), 30: (28, [CG.REASON_NO_VERIFIED_LINK])}
    monkeypatch.setenv(ENV, "on")
    fake_retrieval(monkeypatch, island_links(sol, 58, ISLAND1))
    out = _admit(tmp_path, result, 58)
    ra = out.record["retrieval_admission"]
    assert ra["state"] == RA.STATE_NOT_APPLIED and "OPEN: needs a reason-contract ruling" in ra["why"]
    assert ra["candidates"][0]["admitted"] is True                         # the certificate held; the piece did not
    assert out.components == result.components and _room(out, 58) == list(range(30))


def test_the_re_gates_reason_for_such_a_remainder_contradicts_the_contract():
    """Why MED-6 needs a ruling: in the projection re-gate, F (50..57) has TWO verified, redundant links to the
    admitted cameras now in the room; the camera allow-list, not the evidence, keeps it out; and the gate's reason
    for it falls back to `no-verified-link` -- which COMPONENTS section 2.2 defines as "no verified image pair links
    it to the room"."""
    sizes = (30, 20, 8)
    sol = F.multi_solution(sizes)
    links, rots = F.multi_links(sizes, ((49, 50), (49, 51)))
    model = CP.solve_model(sol, {k.keyframe_id: F.name(i) for i, k in enumerate(F.keyframes(58))})
    lv = {F.name(i): 0.0 for i in range(58)}
    adm = {F.name(i) for i in range(30, 50)}
    out = CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True,
                        room=[F.name(i) for i in range(30)] + sorted(adm), admit=adm, rider_min_shared=3)
    lab = out["labels"][F.name(50)]
    assert lab != 0 and {out["labels"][F.name(i)] for i in range(50, 58)} == {lab}
    (reasons,) = [c["reasons"] for c in out["components"] if c["label"] == lab]
    assert reasons == [CG.REASON_NO_VERIFIED_LINK]
    (d,) = [d for rd in out["rounds"] for d in rd["decisions"] if d.get("first_camera") == F.name(50)]
    assert d["cross_links"] == 2 and d["redundant"] and d["barred"] == "not a group of the room"


# ---------------------------------------------------------------------------------------------------------------
# the post-checks (PREREG B.2)


def _post(base_labels, proj_labels, admitted, anchor):
    names = sorted(base_labels)
    model = CG.SolveModel(names=names, component=np.zeros(len(names), np.int64),
                          R_cw=np.stack([np.eye(3)] * len(names)), n_obs=np.full(len(names), 40),
                          obs_image=np.zeros(0, np.int64), obs_point=np.zeros(0, np.int64), n_points=0)
    base = {"labels": base_labels, "groups": [{"label": 0, "reference": True, "members": anchor}]}
    return RA.post_checks(model, base, {"labels": proj_labels}, set(admitted), GP)[0]


def test_the_post_checks_refuse_in_turn():
    base = {"a1": 0, "a2": 0, "r3": 0, "g1": 1, "g2": 1, "p1": 2, "p2": 2}
    good = dict(base, g1=0, g2=0)
    assert _post(base, good, {"g1", "g2"}, ["a1", "a2"]) is None
    # the anchor is not kept
    assert "anchor" in _post(base, dict(good, a2=3), {"g1", "g2"}, ["a1", "a2"])
    # a closure break: an admitted camera the re-gate did not attach (its own scale check refused it at attach)
    assert "closure" in _post(base, dict(good, g2=1), {"g1", "g2"}, ["a1", "a2"])
    # ... or a camera nobody admitted joined
    assert "closure" in _post(base, dict(good, p1=0), {"g1", "g2"}, ["a1", "a2"])
    # an outside piece that changes
    assert "piece" in _post(base, dict(good, p2=4), {"g1", "g2"}, ["a1", "a2"])


def test_a_closure_break_in_the_re_gate_is_not_applied(tmp_path, monkeypatch):
    """The re-gate's own scale check refuses the admitted group at attach: the projection is `not-applied`."""
    sol, result = _gate(tmp_path, monkeypatch, (30, 20))
    monkeypatch.setenv(ENV, "on")
    fake_retrieval(monkeypatch, island_links(sol, 50, ISLAND1))
    real = CG.apply_gate

    def refusing_scale(*a, **kw):
        out = real(*a, **kw)
        if kw.get("admit"):
            for n in [x for x, v in out["labels"].items() if int(x[:8]) >= 30]:
                out["labels"][n] = 1
        return out

    monkeypatch.setattr(CG, "apply_gate", refusing_scale)
    out = _admit(tmp_path, result, 50)
    ra = out.record["retrieval_admission"]
    assert ra["state"] == RA.STATE_NOT_APPLIED and "evidence closure" in ra["why"]
    assert out.components == result.components


# ---------------------------------------------------------------------------------------------------------------
# the binding invariant: retrieval links never reach a gate call


def test_no_retrieval_link_reaches_any_gate_call(tmp_path, monkeypatch):
    sol, result = _gate(tmp_path, monkeypatch, (30, 20, 8), cross=((49, 50),))
    monkeypatch.setenv(ENV, "on")
    monkeypatch.setenv(config.WORLD_ANCHOR_VERIFY_ENV, "images")
    R, C = F.poses_of(sol, 58)
    arrays = F.synthetic_pairs(R, C, F.chain_pairs(58, breaks=(30, 50)))
    monkeypatch.setattr(AV, "_default_pair_builder", lambda root, names, camera: (arrays, {"source": "built"}))
    new = island_links(sol, 58, ISLAND1)
    fake_retrieval(monkeypatch, new)
    retrieval = {RA.sorted_pair(x["a"], x["b"]) for x in new}
    seen = []
    real = CG.apply_gate

    def spy(model, links, metric_log, **kw):
        seen.append((links, kw["link_rotations"]))
        return real(model, links, metric_log, **kw)

    real_verify = AV.verify_published

    def verify_with_a_seal(res, *, regate, **kw):
        # the P4 re-run's seal re-gate is a gate call too
        regate(seal={F.name(10): CG.REASON_LINK_CONTRADICTED}, room=[F.name(i) for i in range(50)])
        return real_verify(res, regate=regate, **kw)

    monkeypatch.setattr(CG, "apply_gate", spy)
    monkeypatch.setattr(AV, "verify_published", verify_with_a_seal)
    out = _admit(tmp_path, result, 58)
    assert out.record["retrieval_admission"]["state"] == RA.STATE_APPLIED and len(seen) >= 2
    db_links, db_rots = F.multi_links((30, 20, 8), ((49, 50),))
    for links, rots in seen:
        assert set(links) == set(db_links) and set(rots) == set(db_rots)
        assert not retrieval & {RA.sorted_pair(*k) for k in links}
        assert not retrieval & {RA.sorted_pair(*k) for k in rots}
    # every call of the admission got the SAME read-only objects: the database reader's
    assert len({id(links) for links, _ in seen}) == 1
    assert all(type(links).__name__ == "mappingproxy" for links, _ in seen)
    assert out.record["retrieval_admission"]["invariant"]["gate_calls"] == len(seen)


# ---------------------------------------------------------------------------------------------------------------
# P4 again


def _p4_sealed_input(tmp_path, monkeypatch):
    """The input as P4 leaves it: room camera 29 sealed (its seal re-gate: the seal, the pre-seal room as the
    allow-list) and the record saying so."""
    sol, _ = _gate(tmp_path, monkeypatch, (30, 20))
    seal = {F.name(29): CG.REASON_LINK_CONTRADICTED}
    result = CP.gate_final_solution(F.Store(tmp_path), "w1", F.SID, sol, database_path="db",
                                    keyframes=F.keyframes(50), seal=seal, room=[F.name(i) for i in range(30)])
    result.record = dict(result.record, anchor_verify={
        "state": "applied", "sealed": {CG.REASON_LINK_CONTRADICTED: {"keyframes": 1,
                                                                     "keyframe_ids": [f"{F.SID}:{29:08d}"]}},
        "collateral": {"keyframes": 0, "keyframe_ids": []}, "room_before": 30, "sealed_kf": 1})
    assert _room(result, 50) == list(range(29))
    return sol, result


def test_the_p4_re_run_carries_admit_and_the_seals(tmp_path, monkeypatch):
    sol, result = _p4_sealed_input(tmp_path, monkeypatch)
    monkeypatch.setenv(ENV, "on")
    monkeypatch.setenv(config.WORLD_ANCHOR_VERIFY_ENV, "images")
    fake_retrieval(monkeypatch, island_links(sol, 50, ISLAND1))
    gates = []
    real_gfs = CP.gate_final_solution

    def spy_gate(*a, **kw):
        gates.append(kw)
        return real_gfs(*a, **kw)

    reruns = []

    def spy_verify(res, *, regate, **kw):
        reruns.append(res)
        regate(seal={F.name(10): CG.REASON_LINK_CONTRADICTED}, room=[F.name(i) for i in range(50)])
        return res

    monkeypatch.setattr(CP, "gate_final_solution", spy_gate)
    monkeypatch.setattr(AV, "verify_published", spy_verify)
    out = _admit(tmp_path, result, 50)
    assert out.record["retrieval_admission"]["state"] == RA.STATE_APPLIED
    assert len(reruns) == 1 and _room(reruns[0], 50) == [i for i in range(50) if i != 29]
    projection, seal_regate = gates
    adm = [F.name(i) for i in range(30, 50)]
    assert projection["admit"] == adm and projection["seal"] == {F.name(29): CG.REASON_LINK_CONTRADICTED}
    assert sorted(projection["room"]) == sorted(F.name(i) for i in range(50))
    assert seal_regate["admit"] == adm and seal_regate["rider_min_shared"] == CP.RIDER_MIN_SHARED
    assert seal_regate["seal"] == {F.name(29): CG.REASON_LINK_CONTRADICTED, F.name(10): CG.REASON_LINK_CONTRADICTED}
    assert out.record["anchor_verify"] == result.record["anchor_verify"]     # P4's own record is the input's


def _contradicted_admission(tmp_path, monkeypatch):
    """The masked image tier contradicts island A (70..89) by 10 deg against the room and confirms the room's own
    groups; the first P4 run (on the gate's room, A outside) seals nothing. Retrieval certifies A. Returns (the
    input as the first P4 run publishes it, its keyframes, n)."""
    sizes = (70, 20)
    n = sum(sizes)
    segments = [0] * 35 + [2] * 35 + [1] * 20
    sol = F.multi_solution(sizes, centres=F.line_centres(n))
    links, rots = F.multi_links(sizes, ())
    F.patch_gate_inputs(monkeypatch, links, rots, [0.0] * n)
    monkeypatch.setenv(config.WORLD_ANCHOR_VERIFY_ENV, "images")
    R, C = F.poses_of(sol, n)
    good = [(0, 40), (3, 45), (6, 50), (9, 55), (12, 60)]                  # the room's runs agree with each other
    bad = [(5, 72), (8, 75), (11, 78), (14, 81)]                           # A against the room: 10 deg off
    arrays = F.synthetic_pairs(R, C, F.chain_pairs(n, breaks=(35, 70)) + good + bad,
                               error_deg={e: 10.0 for e in bad})
    monkeypatch.setattr(AV, "_default_pair_builder", lambda root, names, camera: (arrays, {"source": "built"}))
    kfs = F.keyframes(n, segments=segments)
    result = CP.gate_final_solution(F.Store(tmp_path), "w1", F.SID, sol, database_path="db", keyframes=kfs)
    result = CP.anchor_verified(F.Store(tmp_path), "w1", F.SID, result, database_path="db", keyframes=kfs,
                                workspace_root=tmp_path)
    assert result.record["anchor_verify"]["sealed_kf"] == 0 and _room(result, n) == list(range(70))
    assert _pieces(result)[70] == (20, [CG.REASON_NO_VERIFIED_LINK])
    monkeypatch.setenv(ENV, "on")
    fake_retrieval(monkeypatch, island_links(sol, n, [(70 + k, k) for k in range(6)]))
    return result, kfs, n


def test_an_image_contradicted_admission_is_sealed_by_the_p4_re_run(tmp_path, monkeypatch):
    """The P4 re-run seals A like any attached group (without it A would stay in the room), and the record says which
    run sealed what (Codex C23x MED-5): the published `anchor_verify` is the FIRST run's (0 sealed), and the
    admission's record carries both runs, the second with the keyframe ids and the reason of all 20 seals."""
    result, kfs, n = _contradicted_admission(tmp_path, monkeypatch)
    out = RA.admitted(F.Store(tmp_path), "w1", F.SID, result, database_path="db", keyframes=kfs,
                      workspace_root=tmp_path)
    ra = out.record["retrieval_admission"]
    assert ra["state"] == RA.STATE_APPLIED and ra["admitted_keyframes"] == 20
    assert _room(out, n) == list(range(70)) and ra["room_after"] == 70
    assert _pieces(out)[70] == (20, [AV.IMAGE_SEAL_REASON])
    ids = [f"{F.SID}:{i:08d}" for i in range(70, 90)]
    first, second = ra["p4_first_run"], ra["p4_second_run"]
    assert (first["state"], first["sealed_kf"], first["sealed"]) == (AV.STATE_APPLIED, 0, {})
    assert (second["state"], second["sealed_kf"], second["room_before"], second["room_after"]) == \
        (AV.STATE_APPLIED, 20, 90, 70)
    assert second["sealed"] == {AV.IMAGE_SEAL_REASON: {"keyframes": 20, "keyframe_ids": ids}}
    assert out.record["anchor_verify"] == result.record["anchor_verify"]     # the published P4 record: the first run
    # every sealed keyframe published is accounted for by exactly one run
    sealed_out = {k for e in out.components["components"] if e["reasons"] == [AV.IMAGE_SEAL_REASON]
                  for k in e["keyframe_ids"]}
    assert sealed_out == set(ids) == {k for b in second["sealed"].values() for k in b["keyframe_ids"]}


def test_the_first_runs_carried_seals_are_recorded_with_their_ids(tmp_path, monkeypatch):
    sol, result = _p4_sealed_input(tmp_path, monkeypatch)
    monkeypatch.setenv(ENV, "on")
    monkeypatch.setenv(config.WORLD_ANCHOR_VERIFY_ENV, "images")
    fake_retrieval(monkeypatch, island_links(sol, 50, ISLAND1))
    monkeypatch.setattr(AV, "verify_published", lambda res, **kw: dataclasses.replace(
        res, record=dict(res.record, anchor_verify={"state": AV.STATE_APPLIED, "sealed_kf": 0, "sealed": {}})))
    ra = _admit(tmp_path, result, 50).record["retrieval_admission"]
    assert ra["p4_first_run"]["sealed"] == {CG.REASON_LINK_CONTRADICTED: {"keyframes": 1,
                                                                          "keyframe_ids": [f"{F.SID}:{29:08d}"]}}
    assert ra["p4_second_run"] == {"state": AV.STATE_APPLIED, "sealed_kf": 0, "sealed": {}}


def _second_run_is(state):
    """`anchor_verify.verify_published` as a second run ending in `state`: the projected room returned as it came
    in (as verify_published does for anything but `applied`), its record saying so."""
    def verify(res, **kw):
        return dataclasses.replace(res, record=dict(res.record, anchor_verify={"state": state, "why": "stub"}))
    return verify


@pytest.mark.parametrize("second", ["not-applied", "failed", "not-run", "no-such-state"])
def test_every_p4_re_run_that_is_not_applied_publishes_the_input(tmp_path, monkeypatch, second):
    """Codex C23x HIGH-2. `not-applied`, for real: the second run's seal re-gate fails its post-checks, so the 20
    contradicted admitted keyframes it found are NOT sealed -- publishing the projection would leave them in the
    room. `failed`, for real: its pair builder breaks. `not-run` and an unknown state: stubbed. Every one publishes
    the input, never the projection."""
    result, kfs, n = _contradicted_admission(tmp_path, monkeypatch)
    if second == "not-applied":
        monkeypatch.setattr(AV, "seal_refusal", lambda *a, **k: "a piece outside the chosen room changed its keyframes")
    elif second == "failed":
        def broken_builder(root, names, camera):
            raise OSError("no pairs")
        monkeypatch.setattr(AV, "_default_pair_builder", broken_builder)
    else:
        monkeypatch.setattr(AV, "verify_published", _second_run_is(second))
    out = RA.admitted(F.Store(tmp_path), "w1", F.SID, result, database_path="db", keyframes=kfs,
                      workspace_root=tmp_path)
    ra = out.record["retrieval_admission"]
    assert ra["p4_second_run"]["state"] == second
    assert ra["state"] == (RA.STATE_FAILED if second == "failed" else RA.STATE_NOT_APPLIED)
    assert out.components == result.components and _room(out, n) == list(range(70))
    assert _pieces(out)[70] == (20, [CG.REASON_NO_VERIFIED_LINK])            # A is published as the input had it
    assert out.solution.poses == result.solution.poses
    assert out.record["anchor_verify"] == result.record["anchor_verify"]
    if second == "not-applied":
        # the seals the second run found and did NOT publish are on the record, by id
        assert ra["p4_second_run"]["sealed_kf"] == 0
        assert ra["p4_second_run"]["sealed"][AV.IMAGE_SEAL_REASON]["keyframes"] == 20


def test_a_failed_p4_re_run_publishes_the_input(tmp_path, monkeypatch):
    sol, result = _gate(tmp_path, monkeypatch, (30, 20))
    monkeypatch.setenv(ENV, "on")
    monkeypatch.setenv(config.WORLD_ANCHOR_VERIFY_ENV, "images")
    fake_retrieval(monkeypatch, island_links(sol, 50, ISLAND1))

    def broken_builder(root, names, camera):
        raise OSError("no pairs")

    monkeypatch.setattr(AV, "_default_pair_builder", broken_builder)
    out = _admit(tmp_path, result, 50)
    ra = out.record["retrieval_admission"]
    assert ra["state"] == RA.STATE_FAILED and ra["p4_second_run"]["state"] == AV.STATE_FAILED
    assert out.components == result.components and _room(out, 50) == list(range(30))


# ---------------------------------------------------------------------------------------------------------------
# both entry points reach the same function


def _spy_admitted(monkeypatch):
    calls = []
    real = RA.admitted

    def spy(store, world_id, session_id, result, **kw):
        calls.append((result, kw))
        return real(store, world_id, session_id, result, **kw)

    monkeypatch.setattr(RA, "admitted", spy)
    return calls


def test_the_finish_n1_reaches_the_admission(tmp_path, monkeypatch):
    calls = _spy_admitted(monkeypatch)
    monkeypatch.setenv(ENV, "on")
    out = _publish_bytes(tmp_path, monkeypatch, consensus=False)
    assert len(calls) == 1 and calls[0][1]["workspace_root"] == tmp_path / "w1" / "solve" / F.SID
    assert [e["keyframes"] for e in out["components.json"]["components"]] == [50]
    assert out["solution.json"]["gate"]["retrieval_admission"]["state"] == RA.STATE_APPLIED


def test_the_finish_with_the_consensus_reaches_the_admission_and_keeps_the_withheld_piece(tmp_path, monkeypatch):
    """The consensus's withhold path (decision (b)): D (70..75) is withheld (`seed-unstable`); E (76..85) has no
    database link and is certified by retrieval. D is never a candidate -- even with the same evidence -- and stays
    withheld with its one reason; E joins."""
    calls = _spy_admitted(monkeypatch)
    monkeypatch.setenv(ENV, "on")
    sizes = (30, 20, 20, 6, 10)
    n = sum(sizes)
    from tower.world_builder.global_solve import SolveWorkspace, write_solution

    links, rots = F.multi_links(sizes, F.CONSENSUS_CROSS)
    F.patch_gate_inputs(monkeypatch, links, rots, [0.0] * n)
    cands = [F.multi_solution(sizes, o) for o in F.CONSENSUS_OFFSETS]
    evidence = [(76 + k, k) for k in range(6)] + [(70 + k, 10 + k) for k in range(6)]
    fake_retrieval(monkeypatch, island_links(cands[0], n, evidence))
    plan = CP.ConsensusPlan(draws=3, seed=7, map_draw=lambda seed: cands[seed - 7])
    ws = SolveWorkspace(tmp_path / "w1" / "solve" / F.SID)
    _, record = CP.gate_and_publish(F.Store(tmp_path), "w1", F.SID, ws, cands[0], final=True, gate=True,
                                    database_path="db", keyframes=F.keyframes(n), write=write_solution,
                                    consensus=plan)
    assert len(calls) == 1 and calls[0][0].withheld == [F.name(70)]
    assert record["consensus"]["state"] == CP.CONSENSUS_APPLIED and record["consensus"]["detached"]
    ra = record["retrieval_admission"]
    assert ra["state"] == RA.STATE_APPLIED and ra["carried"]["withheld_groups"] == 1
    assert [c["first_keyframe"][-8:] for c in ra["candidates"]] == [f"{76:08d}"]
    assert ra["admitted"] == [{"first_keyframe": f"{F.SID}:{76:08d}", "keyframes": 10}]
    comps = {int(e["keyframe_ids"][0][-8:]): (e["keyframes"], e["reasons"])
             for e in json.loads((ws.root / CP.COMPONENTS_FILENAME).read_text())["components"]}
    assert comps == {0: (80, []), 70: (6, [CG.REASON_SEED_UNSTABLE])}


def test_the_re_gate_in_place_reaches_the_admission(tmp_path, monkeypatch):
    from tower.world_builder.global_solve import load_solution

    calls = _spy_admitted(monkeypatch)
    monkeypatch.setenv(ENV, "on")
    store, ws = _saved_world(tmp_path, monkeypatch)
    CP.regate_published(store, "w1", F.SID)
    assert len(calls) == 1 and calls[0][1]["workspace_root"] == ws.root
    again = load_solution(store, "w1", F.SID)
    assert again.gate["retrieval_admission"]["state"] == RA.STATE_APPLIED and "regate" in again.gate
    comps = json.loads((ws.root / CP.COMPONENTS_FILENAME).read_text())["components"]
    assert [e["keyframes"] for e in comps] == [50]


def test_a_stopped_draw_0_never_reaches_the_admission(tmp_path, monkeypatch):
    calls = _spy_admitted(monkeypatch)
    monkeypatch.setenv(ENV, "on")
    stopped = SimpleNamespace(record={"state": CP.GATE_STATE_APPLIED, "depth": {"state": CP.DEPTH_STOPPED}})
    assert CP.draw_0_stopped(stopped)
    from tower.world_builder.global_solve import SolveWorkspace

    monkeypatch.setattr(CP, "gate_final_solution", lambda *a, **k: stopped)
    CP.gate_and_publish(F.Store(tmp_path), "w1", F.SID, SolveWorkspace(tmp_path / "ws"), object(), final=True,
                        gate=True, database_path="db", keyframes=[], write=lambda *a: None, keep_on_stop=True)
    assert calls == []


# ---------------------------------------------------------------------------------------------------------------
# failure containment


def test_any_failure_publishes_the_input_and_says_so(tmp_path, monkeypatch, caplog):
    sol, result = _gate(tmp_path, monkeypatch, (30, 20))
    monkeypatch.setenv(ENV, "on")

    def broken(*a, **k):
        raise ValueError("retrieval broke")

    monkeypatch.setattr(RA, "retrieve", broken)
    with caplog.at_level(logging.ERROR, logger="tower.world_builder.retrieval_admission"):
        out = _admit(tmp_path, result, 50)
    ra = out.record["retrieval_admission"]
    assert ra["state"] == RA.STATE_FAILED and ra["detail"] == "ValueError: retrieval broke"
    assert out.solution is result.solution and out.components is result.components and out.gated is result.gated
    assert "retrieval admission failed" in caplog.text


def test_a_switch_that_cannot_be_read_is_off_and_returns_the_very_object(tmp_path, monkeypatch, caplog):
    """Codex C23x MED-4, the first boundary: the switch's read raises -- the input, the very object, as when off."""
    _, result = _gate(tmp_path, monkeypatch, (30, 20))
    monkeypatch.setenv(ENV, "on")

    def broken():
        raise RuntimeError("the environment is unreadable")

    monkeypatch.setattr(config, "world_retrieval_admission_setting", broken)
    monkeypatch.setattr(RA, "_admit", lambda *a, **k: pytest.fail("the admission ran on an unreadable switch"))
    with caplog.at_level(logging.ERROR, logger="tower.world_builder.retrieval_admission"):
        assert _admit(tmp_path, result, 50) is result
    assert "switch could not be read" in caplog.text


@pytest.mark.parametrize("where", ["AdmissionParams", "digest", "GateParams"])
def test_a_failure_building_the_parameters_publishes_the_input(tmp_path, monkeypatch, where):
    """Codex C23x MED-4, the second boundary: the parameters (or their digest, or the gate's) cannot be built -- the
    input is published as it was, its record saying `failed`."""
    sol, result = _gate(tmp_path, monkeypatch, (30, 20))
    monkeypatch.setenv(ENV, "on")
    fake_retrieval(monkeypatch, island_links(sol, 50, ISLAND1))

    def broken(*a, **k):
        raise ValueError(f"{where} broke")

    if where == "AdmissionParams":
        monkeypatch.setattr(RA, "AdmissionParams", broken)
    elif where == "digest":
        monkeypatch.setattr(RA.AdmissionParams, "digest", broken)
    else:
        monkeypatch.setattr(CG, "GateParams", broken)
    out = RA.admitted(F.Store(tmp_path), "w1", F.SID, result, database_path="db", keyframes=F.keyframes(50),
                      workspace_root=tmp_path)
    ra = out.record["retrieval_admission"]
    assert ra["state"] == RA.STATE_FAILED and ra["detail"] == f"ValueError: {where} broke"
    assert out.solution is result.solution and out.components is result.components and out.gated is result.gated


def test_a_failure_outside_the_admissions_own_containment_returns_the_very_object(tmp_path, monkeypatch):
    """The stage-timing hook (or anything else around `_admit`) raises: the input, the very object."""
    _, result = _gate(tmp_path, monkeypatch, (30, 20))
    monkeypatch.setenv(ENV, "on")

    def broken(*a, **k):
        raise RuntimeError("the timing hook broke")

    monkeypatch.setattr(RA, "_admit", broken)
    assert _admit(tmp_path, result, 50) is result


def test_not_even_the_failure_record_can_be_written_returns_the_very_object(tmp_path, monkeypatch):
    sol, result = _gate(tmp_path, monkeypatch, (30, 20))
    monkeypatch.setenv(ENV, "on")

    def broken(*a, **k):
        raise ValueError("broke")

    monkeypatch.setattr(RA, "_run", broken)
    monkeypatch.setattr(RA, "_done", broken)
    assert _admit(tmp_path, result, 50) is result


def test_the_real_retrieval_without_the_solves_masks_fails_safely(tmp_path, monkeypatch):
    sol, result = _gate(tmp_path, monkeypatch, (30, 20))
    monkeypatch.setenv(ENV, "on")
    out = _admit(tmp_path, result, 50)                                     # no masks directory under tmp_path
    assert out.record["retrieval_admission"]["state"] == RA.STATE_FAILED
    assert out.components is result.components


# ---------------------------------------------------------------------------------------------------------------
# the verifier: one seed per pair, the same links on any schedule (manager 145 section 1(a))


def test_pair_seed_is_stable_order_free_and_per_pair():
    assert RA.pair_seed("00000001.jpg", "00000002.jpg") == RA.pair_seed("00000002.jpg", "00000001.jpg")
    assert RA.pair_seed("00000001.jpg", "00000002.jpg") != RA.pair_seed("00000001.jpg", "00000003.jpg")
    assert RA.pair_seed("00001715.jpg", "00003153.jpg") == 1204541796         # pinned: sha1, not hash()
    assert 0 <= RA.pair_seed("a", "b") < 2 ** 31


def _scene(seed, n_cams=6, n_pts=400):
    """Synthetic masked features: 3-D points seen by `n_cams` pinhole cameras, SIFT-like uint8 descriptors per point
    (+ noise per view), and outliers."""
    rng = np.random.default_rng(seed)
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    X = rng.uniform([-2, -2, 4], [2, 2, 9], (n_pts, 3))
    desc = rng.integers(0, 256, (n_pts, 128))
    feats, names = {}, []
    for c in range(n_cams):
        Rc = cv2.Rodrigues(rng.normal(0, 0.08, 3))[0]
        t = np.array([0.25 * c, 0.03 * c, 0.0])
        x = (K @ ((Rc @ X.T).T + t).T).T
        xy = x[:, :2] / x[:, 2:]
        vis = (xy[:, 0] > 0) & (xy[:, 0] < 320) & (xy[:, 1] > 0) & (xy[:, 1] < 240)
        d = np.clip(desc[vis] + rng.integers(-6, 7, (int(vis.sum()), 128)), 0, 255).astype(np.uint8)
        o = rng.integers(0, 256, (80, 128)).astype(np.uint8)
        pts = np.vstack([xy[vis] + rng.normal(0, 0.5, (int(vis.sum()), 2)), rng.uniform([0, 0], [320, 240], (80, 2))])
        name = F.name(c)
        feats[name] = (pts.astype(np.float32), np.ascontiguousarray(np.vstack([d, o])))
        names.append(name)
    return feats, names, K


def test_the_verifier_gives_the_same_links_serially_and_on_eight_threads():
    feats, names, K = _scene(11)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    serial = RA.verify_pairs(pairs, feats, K, 320 * 240, P, workers=1)
    threaded = RA.verify_pairs(list(reversed(pairs)), feats, K, 320 * 240, P, workers=8)
    again = RA.verify_pairs(pairs, feats, K, 320 * 240, P, workers=8)
    key = lambda links: [(x["a"], x["b"], x["inl"], np.round(x["R"], 12).tolist(), x["noncompact"]) for x in links]  # noqa: E731
    assert serial and key(serial) == key(threaded) == key(again)
    assert [(x["a"], x["b"]) for x in serial] == sorted((x["a"], x["b"]) for x in serial)


def test_every_pair_is_verified_under_its_own_seed(monkeypatch):
    feats, names, K = _scene(12, n_cams=4)
    pairs = [(names[0], names[1]), (names[2], names[3]), (names[0], names[3])]
    seeds = []
    real = cv2.setRNGSeed
    monkeypatch.setattr(cv2, "setRNGSeed", lambda s: (seeds.append(s), real(s))[1])
    RA.verify_pairs(pairs, feats, K, 320 * 240, P, workers=1)
    assert seeds == [RA.pair_seed(a, b) for a, b in sorted(pairs)]


def test_the_database_links_re_fit_is_seeded_per_pair(monkeypatch):
    feats, names, K = _scene(13, n_cams=2)
    ia, ib = RA.mnn(feats[names[0]][1], feats[names[1]][1], P.ratio)
    xa, xb = feats[names[0]][0][ia].astype(np.float64), feats[names[1]][0][ib].astype(np.float64)
    seeds = []
    real = cv2.setRNGSeed
    monkeypatch.setattr(cv2, "setRNGSeed", lambda s: (seeds.append(s), real(s))[1])
    monkeypatch.setattr(RA, "read_pair_inliers", lambda db, keys: {k: (xa, xb) for k in keys})
    out = RA.db_geometry("db", [(names[0], names[1])], K, 320 * 240, P)
    assert seeds == [RA.pair_seed(names[0], names[1])] and out[(names[0], names[1])]["hull_a"] > 0


def test_mnn_is_exact_on_integer_descriptors():
    rng = np.random.default_rng(5)
    a = rng.integers(0, 256, (300, 128)).astype(np.uint8)
    b = np.vstack([a[:150], rng.integers(0, 256, (200, 128)).astype(np.uint8)])
    ia, ib = RA.mnn(a, b, P.ratio)
    assert list(ia) == list(range(150)) and list(ib) == list(range(150))


@pytest.fixture
def blas_threads():
    """numpy's OpenBLAS thread count: (get, set), restored after the test."""
    api = RA._openblas_threads_api()
    assert api is not None, "numpy's OpenBLAS is not reachable in this venv"
    get, put = api
    before = get()
    yield get, put
    put(before)


def test_one_blas_thread_pins_one_and_restores_the_processs_count(blas_threads):
    get, put = blas_threads
    put(4)
    with RA.one_blas_thread():
        assert get() == 1
        with RA.one_blas_thread():                                           # nested: one pin
            assert get() == 1
        assert get() == 1
    assert get() == 4
    with pytest.raises(ZeroDivisionError):
        with RA.one_blas_thread():
            1 / 0                                                            # noqa: B018
    assert get() == 4 and RA._BLAS["depth"] == 0


def test_without_an_openblas_handle_the_bag_of_words_refuses(monkeypatch):
    """No way to pin the thread count: the bag of words is not computed (the admission fails closed)."""
    monkeypatch.setitem(RA._BLAS, "api", False)
    with pytest.raises(RuntimeError, match="cannot be pinned"):
        with RA.one_blas_thread():
            pass
    feats, _names, _K = _scene(15, n_cams=4, n_pts=200)
    with pytest.raises(RuntimeError, match="cannot be pinned"):
        RA.build_bow(feats, RA.AdmissionParams(words=16, sample=500, iterations=2))


def test_the_bag_of_words_products_run_under_one_blas_thread(monkeypatch, blas_threads):
    """Codex C23x MED-3: the nearest-word products (k-means and the final words) and the query similarities run with
    ONE BLAS thread whatever the process's count, so the bag of words and the query pairs are the same at 1 and at
    8 threads (bitwise). The cross-process proof on W4 is the corpus test."""
    get, put = blas_threads
    feats, names, _K = _scene(16, n_cams=8, n_pts=300)
    small = RA.AdmissionParams(words=32, sample=2000, iterations=3)
    cand = [{"members": [names[6], names[7]]}]
    room = set(names[:6])
    out = {}
    for n in (1, 8):
        put(n)
        seen = []
        real = RA.one_blas_thread

        @RA.contextlib.contextmanager
        def spy():
            with real():
                seen.append(get())
                yield

        monkeypatch.setattr(RA, "one_blas_thread", spy)
        bn, V = RA.build_bow(feats, small)
        pairs = RA.query_pairs(cand, room, bn, V, set(), RA.AdmissionParams(k_room=3))
        monkeypatch.setattr(RA, "one_blas_thread", real)
        assert seen and set(seen) == {1} and get() == n
        # k-means: one pin per iteration and the final words; the queries: one per candidate camera
        assert len(seen) == small.iterations + 1 + 2
        out[n] = (bn, V.tobytes(), pairs)
    assert out[1] == out[8]


def test_the_bag_of_words_and_queries_are_seeded_and_candidate_only():
    feats, names, K = _scene(14, n_cams=8, n_pts=300)
    small = RA.AdmissionParams(words=32, sample=2000, iterations=3)
    n1, V1 = RA.build_bow(feats, small)
    n2, V2 = RA.build_bow(feats, small)
    assert n1 == n2 == sorted(feats) and np.array_equal(V1, V2)
    cand = [{"members": [names[6], names[7]]}]
    room = set(names[:5])
    tried = {RA.sorted_pair(names[6], names[0])}
    pairs = RA.query_pairs(cand, room, n1, V1, tried, RA.AdmissionParams(k_room=3))
    assert pairs and all(a in (names[6], names[7]) and b in room for a, b in pairs)
    assert (names[6], names[0]) not in pairs and len(pairs) == len(set(pairs)) <= 6


# ---------------------------------------------------------------------------------------------------------------
# the cost hook (manager 145 section 1(d))


class _TimingStore(F.Store):
    def session_dir(self, world_id, session_id):
        return Path(self.root) / world_id / "sessions" / session_id


def test_the_admission_is_a_stage_timing_stage(tmp_path, monkeypatch):
    from tower.world_builder import stage_timing as timing

    store = _TimingStore(tmp_path)
    store.session_dir("w1", F.SID).mkdir(parents=True)
    sol, result = _gate(tmp_path, monkeypatch, (30, 20))
    fake_retrieval(monkeypatch, island_links(sol, 50, ISLAND1))
    path = store.session_dir("w1", F.SID) / timing.FILENAME
    monkeypatch.setenv(config.WORLD_STAGE_TIMING_ENV, "on")
    RA.admitted(store, "w1", F.SID, result, database_path="db", keyframes=F.keyframes(50), workspace_root=tmp_path)
    assert not path.exists(), "the switch off records no admission stage"
    monkeypatch.setenv(ENV, "on")
    out = RA.admitted(store, "w1", F.SID, result, database_path="db", keyframes=F.keyframes(50),
                      workspace_root=tmp_path)
    assert out.record["retrieval_admission"]["state"] == RA.STATE_APPLIED
    doc = json.loads(path.read_text(encoding="utf-8"))
    stages = doc["finishes"][-1]["stages"]
    (rec,) = stages["admission"]
    assert rec["status"] == "ok" and rec["wall_ms"] >= 0
    assert "gate" in stages                                                  # the projection re-gate, nested
