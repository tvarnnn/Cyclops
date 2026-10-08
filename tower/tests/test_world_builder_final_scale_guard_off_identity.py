"""NO BIG BANG for the final-solve scale guard: with `TOWER_WORLD_FINAL_SCALE_GUARD` off,
every final-solve output and every status byte is what 324f4a6 produced.

The guard (`world_builder/final_scale_guard.py`; design RUN
review/codex/cx-SCALE-GUARD-DESIGN-20261006.md) adds one default-off Tower setting. Off --
unset, blank, `0`, `off`, `false` or garbage -- it performs no new read, write,
serialization, timestamp or record change. This file proves it against
`golden/world_builder_final_scale_guard_324f4a6.json`, recorded by running this very file on
324f4a6 (the base, which has no guard) twice, identical.

WHAT IS COMPARED, per scenario and per off spelling, on a fresh walk each time (the
recording-pycolmap walk of `test_world_builder_reproducible_finish.py`, with deterministic
ids and a fixed engine clock):

  simple      the live product: a final solve, unmasked, ungated, one draw;
  gated       masks on, the evidence gate on, one draw;
  consensus   masks on, the gate on, three mapper-seed draws (draw 0 published first);
  regate      the gated world, then `coherence_publish.regate_published` in place.

1. FILES: every file under the world root, as the SHA-256 of its bytes -- JSON with its
   volatile values (wall-clock stamps, durations, `solve_identity`, per-solve database
   names) replaced by placeholders, `.npz` as the SHA-256 of each array, everything else raw.
2. STATUS: the status producer's payload and revision (polled twice), the session's row in
   the saved-worlds listing, and the render revision body, normalised the same way.
3. RETURN: the dict `global_solve.solve` (or `regate_published`) returned.

Within one run every off spelling must give the unset run's observation exactly.

TO RECORD: `WB_SCALE_GUARD_RECORD=<path>` writes the observations to <path> instead of
comparing. Record only on 324f4a6, twice, and keep the golden only if the two are identical.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
import re
import sqlite3
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.test_world_builder_reproducible_finish import N_A, N_KF, _candidate, _rz
from tests.test_world_builder_solve_masks import (  # noqa: F401 -- fixtures and helpers
    HEIGHT,
    WIDTH,
    StubDetector,
    _B,
    _KEYPOINTS,
    _blob,
    colmap,
)
from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP
from tower.world_builder import global_solve as GS
from tower.world_builder import relocalizer
from tower.world_builder.records import Keyframe

GOLDEN = Path(__file__).parent / "golden" / "world_builder_final_scale_guard_324f4a6.json"
RECORD_ENV = "WB_SCALE_GUARD_RECORD"
SWITCH = "TOWER_WORLD_FINAL_SCALE_GUARD"
OFF_SPELLINGS = (None, "", "0", "off", "false", "garbage")
SCENARIOS = ("simple", "gated", "consensus", "regate")
CLOCK0 = 1_790_000_000.0
OTHER_SWITCHES = ("TOWER_WORLD_SOLVE_GATE", "TOWER_WORLD_SOLVE_MASKS", "TOWER_WORLD_SOLVE_SEED",
                  "TOWER_WORLD_SOLVE_CONSENSUS", "TOWER_WORLD_ANCHOR_VERIFY",
                  "TOWER_WORLD_POSE_QUARANTINE", "TOWER_WORLD_FINISH_STAGES",
                  "TOWER_WORLD_PICTURE_BASIS", "TOWER_WORLD_LIVE_DEPTH")


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------


def _install_engines(monkeypatch):
    """The mapper, the gate's links and the depth network, faked
    (`test_world_builder_reproducible_finish.engines`), fresh for each walk: the depth fake
    counts its own cached predictions, and a walk must not inherit another's."""
    def map_candidate(pycolmap, database_path, workspace, sparse_dir, keyframes, *, seed, threads,
                      input_digest, min_image_observations, camera):
        sol = _candidate(keyframes, camera)
        sol.input_digest = input_digest
        sol.solved_at = CLOCK0 + 1000.0
        return sol

    monkeypatch.setattr(GS, "_map_candidate", map_candidate)
    name = "{:08d}.jpg".format
    links = {}
    for island in (range(0, N_A), range(N_A, N_KF)):
        idx = list(island)
        for j in range(len(idx)):
            for d in (1, 2):
                if j + d < len(idx):
                    links[(name(idx[j]), name(idx[j + d]))] = 100
    links[(name(29), name(30))] = 40
    rots = {(a, b): _rz(3.0 * int(b[:8])) @ _rz(3.0 * int(a[:8])).T for a, b in links}
    monkeypatch.setattr(CG, "read_verified_links", lambda db, min_inliers=15: dict(links))
    monkeypatch.setattr(CG, "read_link_rotations", lambda db, cam, min_inliers=15: dict(rots))
    made = set()

    def depth(store, world_id, session_id, solution, intrinsics, should_stop=None):
        posed = sorted(solution.poses)
        new = [k for k in posed if k not in made]
        made.update(posed)
        align = {"backend": "moge2-vitl", "known_fov": 42.0, "targets": len(posed), "records": [],
                 "image_origins": {"stored-redacted": len(posed)},
                 "prediction_cache": {"token": "t", "hits": len(posed) - len(new), "predicted": len(new)}}
        return align, store.world_dir(world_id) / "dense" / session_id / "work", CP._depth_params()

    monkeypatch.setattr(CP, "run_gate_depth", depth)
    monkeypatch.setattr(CP, "measure_metric_scale", lambda solution, name_of, db, work: {
        "metric_log": {name(i): 0.0 for i in range(N_KF)}, "cameras_published": N_KF,
        "cameras_measured": N_KF})


def _make_walk(root: Path, colmap, monkeypatch):
    """A stopped 50-keyframe walk whose background solve left undistorted images and a walk
    database (`test_world_builder_reproducible_finish.walk`, with fixed ids and clock)."""
    import tower.world_builder.engine as engine_module
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    _install_engines(monkeypatch)
    ids = itertools.count(1)
    ticks = itertools.count()
    monkeypatch.setattr(engine_module, "new_id", lambda: f"{next(ids):032x}")
    monkeypatch.setattr(relocalizer, "revisit_pairs", lambda session_dir: [])
    store = WorldStore(root)
    engine = WorldBuilderEngine(store, clock=lambda: CLOCK0 + next(ticks))
    world_id = engine.create_world()
    session_id = engine.start_session(
        world_id, frame_source="synthetic",
        intrinsics=CameraIntrinsics(source="self_calibrated", model="pinhole_radtan",
                                    fx=50.0, fy=50.0, cx=WIDTH / 2, cy=HEIGHT / 2,
                                    dist_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
                                    calibrated_width=WIDTH, calibrated_height=HEIGHT))
    rng = np.random.default_rng(0)
    started = store.read_session(world_id, session_id).started_at
    for i in range(N_KF):
        ok, buf = cv2.imencode(".jpg", rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8))
        assert ok
        store.write_keyframe_image(world_id, session_id, f"{i:08d}.jpg", buf.tobytes())
        store.append_keyframe(world_id, Keyframe(
            keyframe_id=f"{session_id}:{i:08d}", session_id=session_id, source_seq=i,
            received_at=started + 0.5 * i, image_relpath=f"images/{i:08d}.jpg",
            width=WIDTH, height=HEIGHT, byte_count=1, segment_index=0))
    engine.stop_session("stopped")
    s = types.SimpleNamespace(store=store, world_id=world_id, session_id=session_id, root=root,
                              workspace=GS.workspace_for(store, world_id, session_id))
    GS.solve(store, world_id, session_id, final=False)          # the walk's own solve
    db = s.workspace.database_path
    db.unlink()
    con = sqlite3.connect(str(db))
    con.executescript(
        "create table images (image_id integer primary key, name text, camera_id integer);"
        "create table keypoints (image_id integer primary key, rows integer, cols integer, data blob);"
        "create table matches (pair_id integer primary key, rows integer, cols integer, data blob);"
        "create table two_view_geometries (pair_id integer primary key, rows integer, cols integer,"
        " data blob, config integer);")
    for i in range(N_KF):
        con.execute("insert into images values (?, ?, 1)", (i + 1, f"{i:08d}.jpg"))
        con.execute("insert into keypoints values (?, 4, 6, ?)", (i + 1, _KEYPOINTS.tobytes()))
        if i:
            con.execute("insert into matches values (?, ?, ?, ?)",
                        (i * _B + i + 1, *_blob([[0, 0], [2, 2], [3, 3]])))
            con.execute("insert into two_view_geometries values (?, ?, ?, ?, 2)",
                        (i * _B + i + 1, *_blob([[0, 0], [2, 2], [3, 3]])))
    con.commit()
    con.close()
    colmap.log.clear()
    return s


def _final(s, scenario):
    """The scenario's final solve (and, for `regate`, the re-gate in place after it)."""
    common = dict(final=True, loop_detection=True, input_digest="walk-digest")
    if scenario == "simple":
        return GS.solve(s.store, s.world_id, s.session_id, masks=False, gate=False, seed=0, **common)
    gated = dict(masks=True, seed=0, gate=True, transient_backend_factory=StubDetector(),
                 mask_device_probe=lambda: None, **common)
    if scenario == "consensus":
        return GS.solve(s.store, s.world_id, s.session_id, consensus=3, **gated)
    out = GS.solve(s.store, s.world_id, s.session_id, **gated)
    if scenario == "regate":
        out = CP.regate_published(s.store, s.world_id, s.session_id)
    return out


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

_VOLATILE_KEYS = ("solve_identity", "revision")
_DB_NAME = re.compile(r"database\.masked\.p\d+\.[0-9a-f]{8}")


class _Norm:
    """Placeholders for volatile values, in order of first sight, so equal volatile values
    stay equal and different ones stay different."""

    def __init__(self, root: Path):
        self.tokens: dict = {}
        self.paths = {str(root), str(root.resolve()), str(root).replace("\\", "/"),
                      str(root.resolve()).replace("\\", "/")}

    def token(self, kind, value):
        key = (kind, json.dumps(value, sort_keys=True, default=str))
        if key not in self.tokens:
            self.tokens[key] = f"<{kind}{len(self.tokens)}>"
        return self.tokens[key]

    def value(self, v, key=None):
        if isinstance(key, str):
            timed = (key == "at" or key.endswith("_at") or key == "seconds" or key.endswith("_seconds")
                     or key.endswith("_s") or key == "timing")
            if timed and (isinstance(v, (int, float, dict)) and not isinstance(v, bool)):
                return "<time>"
            if key in _VOLATILE_KEYS and isinstance(v, str):
                return self.token(key, v)
        if isinstance(v, dict):
            return {k: self.value(x, k) for k, x in sorted(v.items())}
        if isinstance(v, list):
            return [self.value(x) for x in v]
        if isinstance(v, str):
            for p in self.paths:
                v = v.replace(p, "<ROOT>")
            return _DB_NAME.sub("<masked-db>", v)
        return v


def _file_digest(path: Path, norm: _Norm) -> str:
    data = path.read_bytes()
    if path.suffix == ".json":
        try:
            doc = json.loads(data.decode("utf-8"))
        except ValueError:
            return hashlib.sha256(data).hexdigest()
        return hashlib.sha256(json.dumps(norm.value(doc), sort_keys=True).encode("utf-8")).hexdigest()
    if path.suffix == ".jsonl":
        rows = [norm.value(json.loads(line)) for line in data.decode("utf-8").splitlines() if line.strip()]
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
    if path.suffix == ".npz":
        arrays = np.load(io.BytesIO(data))
        h = hashlib.sha256()
        for k in sorted(arrays.files):
            a = arrays[k]
            raw = a.tobytes()
            if a.dtype == np.uint8 and a.ndim == 1 and raw[:1] == b"{":
                # A JSON cache key stored as bytes (the transient caches): its stamps are volatile.
                try:
                    raw = json.dumps(norm.value(json.loads(raw.decode("utf-8"))), sort_keys=True).encode()
                except ValueError:
                    pass
            h.update(f"{k}|{a.dtype.str}|{a.ndim}|".encode("utf-8"))
            h.update(raw)
        return h.hexdigest()
    return hashlib.sha256(data).hexdigest()


def _status(s, norm: _Norm) -> dict:
    from tower.results.world_builder import WorldBuilderStatusProducer
    from tower.results.world_builder_library import build_world_listing
    from tower.results.world_builder_render import build_render_revision

    producer = WorldBuilderStatusProducer(s.root, lambda: CLOCK0 + 500.0)
    snap = producer.snapshot(world_id=s.world_id, session_id=s.session_id)
    again = producer.snapshot(world_id=s.world_id, session_id=s.session_id)
    listing = build_world_listing(s.store)
    row = [x for w in listing["worlds"] if w["world_id"] == s.world_id
           for x in w["sessions"] if x["session_id"] == s.session_id]
    out = {"status": {"payload": snap.payload, "revision": snap.revision},
           "status_again": {"payload": again.payload, "revision": again.revision},
           "row": row}
    for viewer in (None, "appearance-1"):
        try:
            out[f"revision:{viewer}"] = build_render_revision(s.store, s.world_id, s.session_id,
                                                              viewer=viewer)
        except Exception as exc:  # noqa: BLE001 -- the refusal is part of the payload
            out[f"revision:{viewer}"] = {"raised": type(exc).__name__, "message": str(exc)}
    return norm.value(json.loads(json.dumps(out, default=str)))


def _observe(s, returned) -> dict:
    norm = _Norm(s.root)
    files = {p.relative_to(s.root).as_posix(): _file_digest(p, norm)
             for p in sorted(s.root.rglob("*")) if p.is_file()}
    files = {_DB_NAME.sub("<masked-db>", k): v for k, v in files.items()}
    status = _status(s, norm)
    ret = norm.value(json.loads(json.dumps(returned, default=str)))
    return {
        "files": files,
        "status_sha256": hashlib.sha256(json.dumps(status, sort_keys=True).encode("utf-8")).hexdigest(),
        "return_sha256": hashlib.sha256(json.dumps(ret, sort_keys=True).encode("utf-8")).hexdigest(),
        # In the clear, so a failure says what moved.
        "row_keys": sorted({k for r in status["row"] for k in r}),
        "components": [r.get("components") for r in status["row"]],
        "gate_state": (returned or {}).get("gate", {}).get("state")
        if isinstance((returned or {}).get("gate"), dict) else None,
    }


# ---------------------------------------------------------------------------
# the test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_final_solve_outputs_and_status_are_324f4a6s_with_the_guard_off(
        scenario, tmp_path, colmap, monkeypatch):  # noqa: F811 -- fixture
    from scripts import world_refinish as wr

    monkeypatch.setattr(wr, "_restart_attempts", lambda *a, **k: {})
    for name in OTHER_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    runs = {}
    for n, spelling in enumerate(OFF_SPELLINGS):
        if spelling is None:
            monkeypatch.delenv(SWITCH, raising=False)
        else:
            monkeypatch.setenv(SWITCH, spelling)
        # A short directory per run: Windows MAX_PATH (the world tree adds ~150 characters).
        s = _make_walk(tmp_path / f"r{n}", colmap, monkeypatch)
        runs[spelling] = _observe(s, _final(s, scenario))
    for spelling in OFF_SPELLINGS[1:]:
        a, b = runs[spelling], runs[None]
        moved = sorted(k for k in set(a["files"]) | set(b["files"])
                       if a["files"].get(k) != b["files"].get(k))
        assert a == b, (scenario, spelling, moved, {k: (a[k], b[k]) for k in a if k != "files" and a[k] != b[k]})
    observed = runs[None]
    assert any(k.endswith("solution.json") for k in observed["files"]), "nothing was published"
    _check(scenario, observed)


def _check(name, observed):
    record_to = os.environ.get(RECORD_ENV)
    if record_to:
        path = Path(record_to)
        golden = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        golden[name] = observed
        path.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
        pytest.skip(f"recorded {name} into {path}")
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))[name]
    if isinstance(expected, dict) and set(expected) == set(observed):
        for key in sorted(expected):
            assert observed[key] == expected[key], f"{name}: {key} differs from 324f4a6"
    assert observed == expected


def test_the_golden_is_sensitive_to_one_byte(tmp_path, colmap, monkeypatch):  # noqa: F811
    """A 1-byte mutant of a published output fails the comparison: the golden is not vacuous."""
    if os.environ.get(RECORD_ENV):
        pytest.skip("recording")
    for name in OTHER_SWITCHES + (SWITCH,):
        monkeypatch.delenv(name, raising=False)
    s = _make_walk(tmp_path / "m", colmap, monkeypatch)
    returned = _final(s, "simple")
    path = s.workspace.solution_path
    data = bytearray(path.read_bytes())
    # One byte of a pose: the first digit after `"translation": [`.
    at = data.index(b'"translation"') + len(b'"translation": [')
    while not chr(data[at]).isdigit():
        at += 1
    data[at] = ord("7") if data[at] != ord("7") else ord("3")
    path.write_bytes(bytes(data))
    with pytest.raises(AssertionError):
        _check("simple", _observe(s, returned))


def test_live_depth_off_spellings_keep_the_three_golden_outputs(
        tmp_path, colmap, monkeypatch):  # noqa: F811
    """OFF's three spellings preserve the locked, normalized byte digests."""
    from scripts import world_refinish as wr

    monkeypatch.setattr(wr, "_restart_attempts", lambda *a, **k: {})
    for name in OTHER_SWITCHES + (SWITCH,):
        monkeypatch.delenv(name, raising=False)
    baseline_bytes = None
    for n, value in enumerate((None, "off", "misspelled")):
        if value is not None:
            monkeypatch.setenv("TOWER_WORLD_LIVE_DEPTH", value)
        else:
            monkeypatch.delenv("TOWER_WORLD_LIVE_DEPTH", raising=False)
        s = _make_walk(tmp_path / f"l{n}", colmap, monkeypatch)
        observed = _observe(s, _final(s, "gated"))
        for name in ("solution.json", "align.json", "components.json"):
            assert any(path.endswith(name) for path in observed["files"])
        _check("gated", observed)
        raw = {
            "align": (s.store.world_dir(s.world_id) / "dense" / s.session_id /
                      "align.json").read_bytes(),
            "components": (s.workspace.root / "components.json").read_bytes(),
        }
        if baseline_bytes is None:
            baseline_bytes = raw
        else:
            assert raw == baseline_bytes


def test_each_off_output_golden_kills_one_byte_mutant(
        tmp_path, colmap, monkeypatch):  # noqa: F811
    if os.environ.get(RECORD_ENV):
        pytest.skip("recording")
    for name in OTHER_SWITCHES + (SWITCH,):
        monkeypatch.delenv(name, raising=False)
    s = _make_walk(tmp_path / "b", colmap, monkeypatch)
    returned = _final(s, "gated")
    paths = [s.workspace.solution_path,
             s.store.world_dir(s.world_id) / "dense" / s.session_id / "align.json",
             s.workspace.root / "components.json"]
    for path in paths:
        before = path.read_bytes()
        mutant = bytearray(before)
        at = next(i for i, byte in enumerate(mutant) if 48 <= byte <= 57)
        mutant[at] = ord("7") if mutant[at] != ord("7") else ord("3")
        path.write_bytes(mutant)
        with pytest.raises(AssertionError):
            _check("gated", _observe(s, returned))
        path.write_bytes(before)
