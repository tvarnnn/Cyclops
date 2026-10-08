"""The final scale guard AT THE PUBLISH SEAM, end to end through `global_solve.solve` and
`coherence_publish.regate_published`, with the guard ON (and shadow).

Only the external engines are faked (pycolmap, the mapper, the gate's links, the depth stage, the metric
scale; on an ungated solve the live depth cache's metric log, `FSG._live_cache_evidence`). Pinned (v2, Exp1
slice 4: evidence is READ, never made at publish):
  * THE VETO CASE: the walk-3-shaped fixture through the live product's final solve (unmasked, ungated,
    one draw) WITH live-cache evidence: zero bathroom keyframes in the published room; the pieces are
    explicit components the product's own reader takes; the audit names them; the short desk stretch stays
    attached and is flagged uncertified on the placed component; no depth stage runs; the same walk OFF
    keeps the bathroom in the room;
  * shadow publishes exactly what off publishes, and writes the audit;
  * no evidence (or too little) publishes the room `uncertified`, isolates nothing and never runs depth; a
    failed gate is no evidence; an exception -> nothing written, the solution published before stands,
    `solved: false` with a one-line reason;
  * gated: the guard reuses the gate's scale, isolates a plateau the gate kept, and leaves the gate's own
    refusals untouched; both passes of a consensus that publishes draw 0 first are guarded; the re-gate in
    place is guarded, and a withheld re-gate writes nothing and is not a stop.
"""

from __future__ import annotations

import io
import json
import math
import sqlite3
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.test_world_builder_final_scale_guard import (
    W3_BATHROOM,
    W3_N,
    _w3_levels,
    _walk_centers,
)
from tests.test_world_builder_solve_masks import (  # noqa: F401 -- fixtures and helpers
    HEIGHT,
    WIDTH,
    StubDetector,
    _KEYPOINTS,
    colmap,
)
from tower import config
from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP
from tower.world_builder import final_scale_guard as FSG
from tower.world_builder import global_solve as GS
from tower.world_builder import relocalizer
from tower.world_builder.components import parse_components, read_components_record
from tower.world_builder.records import Keyframe

SWITCH = config.WORLD_FINAL_SCALE_GUARD_ENV
P = FSG.GuardParams()


# ---------------------------------------------------------------------------
# end to end: the final solve
# ---------------------------------------------------------------------------

CLOCK0 = 1_790_000_000.0


def _candidate(keyframes, camera, *, centers, islands):
    """One solver component; `islands` [(start, end)] of cameras that share tracks with their next two
    neighbours (30 points each), and 60 points coupling consecutive islands."""
    kids = [k.keyframe_id for k in keyframes]
    n = len(kids)
    obs, p = [], 0
    for a, b in islands:
        idx = list(range(a, b))
        for j in range(len(idx)):
            for _ in range(30):
                for c in idx[j:j + 3]:
                    obs.append([c, len(obs), p])
                p += 1
    for (a, b), (c, d) in zip(islands, islands[1:]):
        for k in range(60):
            obs.append([a + k % (b - a), len(obs), p])
            obs.append([c + k % (d - c), len(obs), p])
            p += 1
    obs = np.asarray(obs, np.int32)
    counts = np.bincount(obs[:, 0], minlength=n)
    poses = {kid: {"component": 0, "rotation": np.eye(3).ravel().tolist(),
                   "translation": (-np.asarray(centers[i], float)).tolist(), "observations": int(counts[i])}
             for i, kid in enumerate(kids)}
    first = np.zeros(p, np.int32)
    for row in obs[::-1]:
        first[row[2]] = row[0]
    rng = np.random.default_rng(1)
    return GS.Solution(
        solver="glomap", solved_at=CLOCK0 + 1000.0, input_digest=None, keyframe_ids=kids, poses=poses,
        components=[{"index": 0, "images": n, "points": p}],
        xyz=rng.normal(size=(p, 3)).astype(np.float32), rgb=np.zeros((p, 3), np.uint8),
        component=np.zeros(p, np.int32), first_keyframe=first, track_length=np.full(p, 3, np.int32),
        error=np.full(p, 0.5, np.float32), observations=obs,
        observation_xy=rng.uniform(0, 40, size=(len(obs), 2)).astype(np.float32),
        camera=camera.to_json_dict())


class _Engines:
    """The mapper, the gate's links, the depth stage and the metric scale, faked; counts the calls."""

    def __init__(self, monkeypatch, *, n, levels, centers=None, islands=None, links=None, depth_fails=False):
        self.depth_calls = 0
        self.metric_calls = 0
        centers = centers or _walk_centers(n, step=0.01)
        islands = islands or [(0, n)]

        def map_candidate(pycolmap, database_path, workspace, sparse_dir, keyframes, *, seed, threads,
                          input_digest, min_image_observations, camera):
            sol = _candidate(keyframes, camera, centers=centers, islands=islands)
            sol.input_digest = input_digest
            return sol

        monkeypatch.setattr(GS, "_map_candidate", map_candidate)
        name = "{:08d}.jpg".format
        if links is None:
            links = {}
            for a, b in islands:
                for j in range(a, b):
                    for d in (1, 2):
                        if j + d < b:
                            links[(name(j), name(j + d))] = 100
        rots = {k: np.eye(3) for k in links}
        monkeypatch.setattr(CG, "read_verified_links", lambda db, min_inliers=15: dict(links))
        monkeypatch.setattr(CG, "read_link_rotations", lambda db, cam, min_inliers=15: dict(rots))

        def depth(store, world_id, session_id, solution, intrinsics, should_stop=None, hold=None):
            self.depth_calls += 1
            if depth_fails:
                raise CP.DepthUnavailable("no network on this test Tower")
            align = {"backend": "moge2-vitl", "known_fov": 42.0, "targets": len(solution.poses), "records": [],
                     "image_origins": {"stored-redacted": len(solution.poses)},
                     "prediction_cache": {"token": "t", "hits": 0, "predicted": len(solution.poses)}}
            return align, store.world_dir(world_id) / "dense" / session_id / "work", CP._depth_params()

        def metric(solution, name_of, db, work):
            self.metric_calls += 1
            return {"metric_log": dict(levels), "cameras_published": n, "cameras_measured": len(levels)}

        monkeypatch.setattr(CP, "run_gate_depth", depth)
        monkeypatch.setattr(CP, "measure_metric_scale", metric)


def _walk(root: Path, n: int, colmap, monkeypatch):
    """A stopped n-keyframe walk whose background solve left undistorted images and a walk database."""
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    monkeypatch.setattr(relocalizer, "revisit_pairs", lambda session_dir: [])
    store = WorldStore(root)
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world()
    session_id = engine.start_session(
        world_id, frame_source="synthetic",
        intrinsics=CameraIntrinsics(source="self_calibrated", model="pinhole_radtan",
                                    fx=50.0, fy=50.0, cx=WIDTH / 2, cy=HEIGHT / 2,
                                    dist_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
                                    calibrated_width=WIDTH, calibrated_height=HEIGHT))
    started = store.read_session(world_id, session_id).started_at
    ok, buf = cv2.imencode(".jpg", np.random.default_rng(0).integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8))
    for i in range(n):
        store.write_keyframe_image(world_id, session_id, f"{i:08d}.jpg", buf.tobytes())
        store.append_keyframe(world_id, Keyframe(
            keyframe_id=f"{session_id}:{i:08d}", session_id=session_id, source_seq=i,
            received_at=started + 0.5 * i, image_relpath=f"images/{i:08d}.jpg",
            width=WIDTH, height=HEIGHT, byte_count=1, segment_index=0))
    engine.stop_session("stopped")
    s = types.SimpleNamespace(store=store, world_id=world_id, session_id=session_id, root=root,
                              workspace=GS.workspace_for(store, world_id, session_id))
    GS.solve(store, world_id, session_id, final=False)
    db = s.workspace.database_path
    db.unlink()
    con = sqlite3.connect(str(db))
    con.executescript(
        "create table images (image_id integer primary key, name text, camera_id integer);"
        "create table keypoints (image_id integer primary key, rows integer, cols integer, data blob);"
        "create table matches (pair_id integer primary key, rows integer, cols integer, data blob);"
        "create table two_view_geometries (pair_id integer primary key, rows integer, cols integer,"
        " data blob, config integer);")
    for i in range(n):
        con.execute("insert into images values (?, ?, 1)", (i + 1, f"{i:08d}.jpg"))
        con.execute("insert into keypoints values (?, 4, 6, ?)", (i + 1, _KEYPOINTS.tobytes()))
    con.commit()
    con.close()
    colmap.log.clear()
    return s


@pytest.fixture(autouse=True)
def _switches(monkeypatch):
    for name in ("TOWER_WORLD_SOLVE_GATE", "TOWER_WORLD_SOLVE_MASKS", "TOWER_WORLD_SOLVE_SEED",
                 "TOWER_WORLD_SOLVE_CONSENSUS", "TOWER_WORLD_ANCHOR_VERIFY", "TOWER_WORLD_POSE_QUARANTINE",
                 SWITCH):
        monkeypatch.delenv(name, raising=False)
    from scripts import world_refinish as wr

    monkeypatch.setattr(wr, "_restart_attempts", lambda *a, **k: {})


def _simple(s, **kw):
    return GS.solve(s.store, s.world_id, s.session_id, final=True, masks=False, gate=False, seed=0,
                    loop_detection=True, input_digest="walk-digest", **kw)


def _gated(s, **kw):
    return GS.solve(s.store, s.world_id, s.session_id, final=True, masks=True, seed=0, gate=True,
                    loop_detection=True, transient_backend_factory=StubDetector(),
                    mask_device_probe=lambda: None, input_digest="walk-digest", **kw)


def _published(s):
    meta = json.loads(s.workspace.solution_path.read_text(encoding="utf-8"))
    comp = s.workspace.root / CP.COMPONENTS_FILENAME
    audit = s.workspace.root / FSG.AUDIT_FILENAME
    return (meta, json.loads(comp.read_text(encoding="utf-8")) if comp.exists() else None,
            json.loads(audit.read_text(encoding="utf-8")) if audit.exists() else None)


def _room_ids(meta):
    return {k for k, p in meta["poses"].items() if p["component"] == 0 and p["observations"] >= 30}


def _w3_levels_named():
    return {k: v for k, v in _w3_levels().items()}


def _live_evidence(monkeypatch, levels):
    """The compatible live depth cache's metric log (its reader is pinned on real files in
    `test_world_builder_final_scale_guard_v2.py`)."""
    calls = []

    def live(store, world_id, session_id, solution, *, database_path, name_of, live_metric_fn=None):
        calls.append(session_id)
        return dict(levels), {"state": "compatible", "records_matched": len(levels)}

    monkeypatch.setattr(FSG, "_live_cache_evidence", live)
    return calls


def _placed(comp):
    (placed,) = [e for e in comp["components"] if e["state"] == "placed"]
    return placed


def test_walk3_bathroom_is_isolated_by_the_live_products_final_solve(tmp_path, colmap, monkeypatch):
    """THE VETO CASE, end to end: the simple solve (unmasked, ungated, one draw) with the guard ON and the
    walk's live depth cache as its evidence (v2 predicts no depth at publish)."""
    eng = _Engines(monkeypatch, n=W3_N, levels=_w3_levels_named())
    s = _walk(tmp_path / "w", W3_N, colmap, monkeypatch)
    _live_evidence(monkeypatch, _w3_levels_named())
    monkeypatch.setenv(SWITCH, "on")
    out = _simple(s)
    assert out["solved"] is True
    meta, comp, audit = _published(s)
    sid = s.session_id
    bathroom = {f"{sid}:{i:08d}" for i in W3_BATHROOM}
    desk_bad = {f"{sid}:{i:08d}" for i in range(74, 84)}
    room = _room_ids(meta)
    assert not (room & bathroom), "a bathroom keyframe stayed in the room"
    # v2: the desk stretch has fewer than 10 ratios (the fixture's missing ones): attached, flagged.
    assert desk_bad <= room
    assert len(room) == W3_N - len(bathroom)
    # the pieces are explicit components, never dropped: every keyframe is still posed
    assert set(meta["poses"]) == {f"{sid}:{i:08d}" for i in range(W3_N)}
    assert "gate" not in meta and meta["solve"]["final_scale_guard"]["decision"] == "published"
    assert meta["solve_identity"]
    # the components record: the room, the bathroom pieces as areas, the desk stretch counted only
    rec = parse_components(comp)
    assert rec is not None
    entries = comp["components"]
    assert entries[0]["state"] == "placed" and entries[0]["shown_as"] == "room"
    unplaced = [e for e in entries if e["state"] == "unplaced"]
    assert {e["reason"] for e in unplaced} == {FSG.REASON_SCALE_OUTLIER}
    assert set().union(*(set(e["keyframe_ids"]) for e in unplaced)) == bathroom
    assert any(e["shown_as"] == "area" for e in unplaced)
    # the placed component says the room is uncertified, and where: the desk stretch among its spans
    status = _placed(comp)["final_scale_guard"]
    assert status["id"] == FSG.GUARD_ID and status["scale_status"] == FSG.SCALE_UNCERTIFIED
    assert status["evidence"] == FSG.EVIDENCE_LIVE
    flagged = {k for sp in status["uncertified_spans"] if sp["kind"] == FSG.UNCERTIFIED_SHORT
               for k in (sp["first_keyframe_id"], sp["last_keyframe_id"])}
    assert flagged & desk_bad
    # the product's own reader takes it (solve_identity and input_digest agree)
    assert read_components_record(s.store, s.world_id, s.session_id) is not None
    # no ratio or figure in the record
    assert "ratio" not in json.dumps(comp) and "metres" not in json.dumps(comp)
    # the audit says which segments, and why
    assert audit["mode"] == "on" and audit["decision"] == "published"
    assert audit["solve_identity"] == meta["solve_identity"]
    isolated = {k for p in audit["assessment"]["pieces"] for k in p["keyframe_ids"]}
    assert isolated == bathroom
    assert all(p["ci_excludes_one"] for p in audit["assessment"]["pieces"])
    assert audit["assessment"]["scale_status"] == FSG.SCALE_UNCERTIFIED
    assert desk_bad <= {k for u in audit["assessment"]["uncertified"] for k in u["keyframe_ids"]}
    assert audit["inputs"]["source"] == FSG.EVIDENCE_LIVE
    assert eng.depth_calls == 0 and eng.metric_calls == 0


def test_off_the_same_walk_publishes_the_bathroom_in_the_room(tmp_path, colmap, monkeypatch):
    _Engines(monkeypatch, n=W3_N, levels=_w3_levels_named())
    s = _walk(tmp_path / "w", W3_N, colmap, monkeypatch)
    _simple(s)
    meta, comp, audit = _published(s)
    assert len(_room_ids(meta)) == W3_N and comp is None and audit is None


def test_shadow_publishes_what_off_publishes_and_writes_the_audit(tmp_path, colmap, monkeypatch):
    _Engines(monkeypatch, n=W3_N, levels=_w3_levels_named())
    s_off = _walk(tmp_path / "a", W3_N, colmap, monkeypatch)
    _simple(s_off)
    s_sh = _walk(tmp_path / "b", W3_N, colmap, monkeypatch)
    _live_evidence(monkeypatch, _w3_levels_named())
    monkeypatch.setenv(SWITCH, "shadow")
    out = _simple(s_sh)
    off, _, _ = _published(s_off)
    sh, comp, audit = _published(s_sh)
    for doc in (off, sh):
        doc.pop("solved_at"), doc.pop("timing")
    assert json.dumps(off, sort_keys=True).replace(s_off.session_id, "S") == \
        json.dumps(sh, sort_keys=True).replace(s_sh.session_id, "S")
    assert comp is None and out["solved"] is True and "final_scale_guard" not in out
    assert audit["mode"] == "shadow" and audit["decision"] == "shadow"
    # what ON would have isolated: the bathroom (x3.19, the doorway, x2.40: one plateau); the short desk
    # stretch would have stayed attached, uncertified (v2)
    sid = s_sh.session_id
    assert {k for p in audit["assessment"]["pieces"] for k in p["keyframe_ids"]} == \
        {f"{sid}:{i:08d}" for i in W3_BATHROOM}
    arrays_off = np.load(io.BytesIO(s_off.workspace.arrays_path.read_bytes()))
    arrays_sh = np.load(io.BytesIO(s_sh.workspace.arrays_path.read_bytes()))
    assert all(np.array_equal(arrays_off[k], arrays_sh[k]) for k in arrays_off.files)


def test_too_little_evidence_publishes_the_room_uncertified_and_never_withholds(tmp_path, colmap, monkeypatch):
    """v2: 5 ratios certify nothing and isolate nothing; the room is published, flagged (was: withheld)."""
    eng = _Engines(monkeypatch, n=60, levels={})
    s = _walk(tmp_path / "w", 60, colmap, monkeypatch)
    _live_evidence(monkeypatch, {f"{i:08d}.jpg": 0.0 for i in range(5)})
    monkeypatch.setenv(SWITCH, "on")
    out = _simple(s)
    meta, comp, audit = _published(s)
    assert out["solved"] is True and len(_room_ids(meta)) == 60
    assert [e["state"] for e in comp["components"]] == ["placed"]
    status = _placed(comp)["final_scale_guard"]
    assert status["scale_status"] == FSG.SCALE_UNCERTIFIED and status["uncertified_keyframes"] == 60
    assert audit["decision"] == "published" and audit["assessment"]["pieces"] == []
    assert meta["solve"]["final_scale_guard"]["scale_status"] == FSG.SCALE_UNCERTIFIED
    assert eng.depth_calls == 0


def test_on_without_evidence_runs_no_depth_and_publishes_uncertified(tmp_path, colmap, monkeypatch):
    """v2 (was: fail-closed on no depth): an ungated solve with no live cache has NO evidence. No depth stage
    runs at publish -- even one that would fail -- and the room is published `uncertified`, never PASS."""
    eng = _Engines(monkeypatch, n=60, levels={f"{i:08d}.jpg": 0.0 for i in range(60)}, depth_fails=True)
    s = _walk(tmp_path / "w", 60, colmap, monkeypatch)
    monkeypatch.setenv(SWITCH, "on")
    out = _simple(s)
    meta, comp, audit = _published(s)
    assert out["solved"] is True and len(_room_ids(meta)) == 60
    assert eng.depth_calls == 0 and eng.metric_calls == 0
    # TOWER_WORLD_LIVE_DEPTH is unset here, so a live cache is not even looked for (lead gate: only `on` is evidence).
    assert audit["inputs"]["source"] == FSG.EVIDENCE_NONE and audit["inputs"]["live"]["state"] == "live-depth-not-on"
    assert audit["assessment"]["scale_status"] == FSG.SCALE_UNCERTIFIED
    status = _placed(comp)["final_scale_guard"]
    assert status["scale_status"] == FSG.SCALE_UNCERTIFIED and status["evidence"] == FSG.EVIDENCE_NONE
    assert status["uncertified_spans"][0]["kind"] == FSG.UNCERTIFIED_NO_EVIDENCE


def test_shadow_without_evidence_publishes_unchanged(tmp_path, colmap, monkeypatch):
    eng = _Engines(monkeypatch, n=60, levels={}, depth_fails=True)
    s = _walk(tmp_path / "w", 60, colmap, monkeypatch)
    monkeypatch.setenv(SWITCH, "shadow")
    out = _simple(s)
    meta, comp, audit = _published(s)
    assert out["solved"] is True and len(_room_ids(meta)) == 60 and comp is None
    assert audit["decision"] == "shadow" and audit["error"] is None
    assert audit["assessment"]["scale_status"] == FSG.SCALE_UNCERTIFIED and eng.depth_calls == 0


def test_an_exception_inside_the_guard_withholds(tmp_path, colmap, monkeypatch):
    _Engines(monkeypatch, n=60, levels={f"{i:08d}.jpg": 0.0 for i in range(60)})
    s = _walk(tmp_path / "w", 60, colmap, monkeypatch)

    def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(FSG, "apply", broken)
    monkeypatch.setenv(SWITCH, "on")
    out = _simple(s)
    assert out["solved"] is False and "RuntimeError" in out["reason"]


# gated ---------------------------------------------------------------------

G_A, G_B = 50, 20


def _gated_levels(plateau=(20, 32, 2.0)):
    v = {f"{i:08d}.jpg": 0.0 for i in range(G_A + G_B)}
    a, b, r = plateau
    for i in range(a, b):
        v[f"{i:08d}.jpg"] = math.log(r)
    return v


def _gated_links():
    name = "{:08d}.jpg".format
    links = {}
    for a, b in ((0, G_A), (G_A, G_A + G_B)):
        for j in range(a, b):
            for d in (1, 2):
                if j + d < b:
                    links[(name(j), name(j + d))] = 100
    links[(name(G_A - 1), name(G_A))] = 40
    return links


def test_the_gated_publish_isolates_a_plateau_the_gate_kept(tmp_path, colmap, monkeypatch):
    """A x2 plateau in the MIDDLE of the anchor block: the gate's binary segmentation keeps it (neither side's
    median moves), the guard does not."""
    eng = _Engines(monkeypatch, n=G_A + G_B, levels=_gated_levels(),
                   islands=[(0, G_A), (G_A, G_A + G_B)], links=_gated_links())
    s = _walk(tmp_path / "off", G_A + G_B, colmap, monkeypatch)
    _gated(s)
    off_meta, off_comp, _ = _published(s)
    sid = s.session_id
    plateau = {f"{sid}:{i:08d}" for i in range(20, 32)}
    assert plateau <= _room_ids(off_meta), "the gate itself refused the plateau: the fixture tests nothing"
    gate_unplaced = [e for e in off_comp["components"] if e["state"] == "unplaced"]

    s = _walk(tmp_path / "on", G_A + G_B, colmap, monkeypatch)
    monkeypatch.setenv(SWITCH, "on")
    eng.metric_calls = 0
    _gated(s)
    meta, comp, audit = _published(s)
    sid = s.session_id
    plateau = {f"{sid}:{i:08d}" for i in range(20, 32)}
    assert not (plateau & _room_ids(meta))
    assert eng.metric_calls == 1, "the guard measured again instead of reusing the gate's scale"
    reasons = {e["reason"] for e in comp["components"] if e["state"] == "unplaced"}
    assert FSG.REASON_SCALE_OUTLIER in reasons
    # the gate's own refusals are untouched
    kept = [e for e in comp["components"] if e["state"] == "unplaced" and e["reason"] != FSG.REASON_SCALE_OUTLIER]
    assert [(e["reason"], e["keyframes"]) for e in kept] == [(e["reason"], e["keyframes"]) for e in gate_unplaced]
    assert meta["gate"]["final_scale_guard"]["pieces_isolated"] == 1
    assert meta["solve_identity"] == comp["solve_identity"] == audit["solve_identity"]
    assert audit["inputs"]["scale"] == "the gate's"


def test_a_gate_that_failed_is_no_evidence_and_publishes_uncertified_when_on(tmp_path, colmap, monkeypatch):
    """v2 (was: withheld): a failed gate measured nothing the guard may read; the room is published as the gate
    publishes it, flagged `uncertified`, nothing isolated."""
    eng = _Engines(monkeypatch, n=G_A + G_B, levels=_gated_levels(), islands=[(0, G_A), (G_A, G_A + G_B)],
                   links=_gated_links())
    s = _walk(tmp_path / "w", G_A + G_B, colmap, monkeypatch)

    def broken(*a, **k):
        raise RuntimeError("gate broke")

    monkeypatch.setattr(CP, "_gate", broken)
    monkeypatch.setenv(SWITCH, "on")
    eng.depth_calls = 0
    out = _gated(s)
    meta, comp, audit = _published(s)
    assert out["solved"] is True and eng.depth_calls == 0
    assert audit["inputs"]["source"] == FSG.EVIDENCE_NONE and audit["assessment"]["pieces"] == []
    assert _placed(comp)["final_scale_guard"]["scale_status"] == FSG.SCALE_UNCERTIFIED


def test_the_consensus_guards_its_early_draw_0_and_its_final_publish(tmp_path, colmap, monkeypatch):
    _Engines(monkeypatch, n=G_A + G_B, levels=_gated_levels(), islands=[(0, G_A), (G_A, G_A + G_B)],
             links=_gated_links())
    s = _walk(tmp_path / "w", G_A + G_B, colmap, monkeypatch)
    calls = []
    real = FSG.guard_publication

    def spy(*a, **k):
        out = real(*a, **k)
        calls.append(out.decision)
        return out

    monkeypatch.setattr(FSG, "guard_publication", spy)
    monkeypatch.setenv(SWITCH, "on")
    out = _gated(s, consensus=3)
    assert calls == ["published", "published"]          # draw 0 first, then the vote
    meta, comp, _ = _published(s)
    plateau = {f"{s.session_id}:{i:08d}" for i in range(20, 32)}
    assert not (plateau & _room_ids(meta)) and out["solved"] is True
    assert meta["gate"]["consensus"]["state"] in (CP.CONSENSUS_APPLIED, CP.CONSENSUS_PARTIAL)


def test_the_regate_in_place_is_guarded(tmp_path, colmap, monkeypatch):
    _Engines(monkeypatch, n=G_A + G_B, levels=_gated_levels(), islands=[(0, G_A), (G_A, G_A + G_B)],
             links=_gated_links())
    s = _walk(tmp_path / "w", G_A + G_B, colmap, monkeypatch)
    _gated(s)                                           # published OFF: the plateau is in the room
    plateau = {f"{s.session_id}:{i:08d}" for i in range(20, 32)}
    assert plateau <= _room_ids(_published(s)[0])
    monkeypatch.setenv(SWITCH, "on")
    out = CP.regate_published(s.store, s.world_id, s.session_id)
    meta, comp, audit = _published(s)
    assert not (plateau & _room_ids(meta))
    assert meta["gate"]["final_scale_guard"]["decision"] == "published" and "regate" in meta["gate"]
    assert out["publish"].get("components_written") is True and audit["decision"] == "published"


def test_a_withheld_regate_writes_nothing_and_is_not_a_stop(tmp_path, colmap, monkeypatch):
    eng = _Engines(monkeypatch, n=G_A + G_B, levels=_gated_levels(), islands=[(0, G_A), (G_A, G_A + G_B)],
                   links=_gated_links())
    s = _walk(tmp_path / "w", G_A + G_B, colmap, monkeypatch)
    _gated(s)
    before = s.workspace.solution_path.read_bytes()

    def no_room(*a, **k):                # v2 withholds only on a failure (no scale reason withholds)
        raise ValueError("the relabelled candidate has no published room")

    monkeypatch.setattr(FSG, "apply", no_room)
    monkeypatch.setenv(SWITCH, "on")
    out = CP.regate_published(s.store, s.world_id, s.session_id)
    assert out["withheld"] is True and out["publish"]["written"] is False and not out.get("stopped")
    assert s.workspace.solution_path.read_bytes() == before
    assert out["detail"].startswith("the final scale guard withheld")
    del eng


def test_a_background_solve_is_never_guarded(tmp_path, colmap, monkeypatch):
    """Only a FINAL solve's publication is guarded: the walk's background solves publish as today, even
    with the guard on (and they run no depth stage for it)."""
    eng = _Engines(monkeypatch, n=W3_N, levels=_w3_levels_named())
    s = _walk(tmp_path / "w", W3_N, colmap, monkeypatch)
    monkeypatch.setenv(SWITCH, "on")
    eng.depth_calls = eng.metric_calls = 0
    out = GS.solve(s.store, s.world_id, s.session_id, final=False, input_digest="walk-digest")
    meta, comp, audit = _published(s)
    assert out["solved"] is True and comp is None and audit is None
    assert len(_room_ids(meta)) == W3_N and "final_scale_guard" not in (meta.get("solve") or {})
    assert eng.depth_calls == 0 and eng.metric_calls == 0
