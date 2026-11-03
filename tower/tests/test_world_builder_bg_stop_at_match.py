"""`TOWER_WORLD_BG_STOP_AT_MATCH` (opt run 2026-11-03, lever L2): at Stop the builder ends the
background solve at its matching commit instead of waiting (up to 120 s) through a map nobody uses.

Modules: `tower/world_builder/matching_marker.py` (the marker and its probes),
`global_solve.solve(on_matched=...)` (where the commit is), `scripts/world_solve.py` (the child
that writes the marker) and `BackgroundSolver.wait` (the builder that reads it).

What is tested:
  * the marker is bound to the launch that wrote it (token), atomic, and never raises;
  * `solve` calls the hook once, after every database write and before mapping, and with no hook
    its pycolmap calls and its published solution are what they were;
  * `world_solve.py` with no token and no probe calls `solve` exactly as before;
  * the wait ends a child at ITS marker (and leaves the database the final opens equal, digest for
    digest and byte for byte, to what the child committed), ignores another launch's marker, and
    is today's wait whenever there is no marker -- a child that dies, one killed between its
    commit and its marker, a wait entered after the child ended, and the switch off.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.world_build_session import BackgroundSolver, child_environment  # noqa: E402
from tower.world_builder import global_solve  # noqa: E402
from tower.world_builder import matching_marker as MM  # noqa: E402

TOWER_DIR = Path(__file__).resolve().parents[1]
SWITCH = "TOWER_WORLD_BG_STOP_AT_MATCH"
PROBE = "TOWER_WORLD_BG_STOP_AT_MATCH_PROBE"


@pytest.fixture(autouse=True)
def _switches_unset(monkeypatch):
    for name in (SWITCH, PROBE, MM.TOKEN_ENV):
        monkeypatch.delenv(name, raising=False)


class _Store:
    def __init__(self, root: Path):
        self._root = root

    def world_dir(self, world_id: str) -> Path:
        return self._root / "worlds" / world_id


def _workspace(tmp_path) -> Path:
    return tmp_path / "worlds" / "w" / "solve" / "s"


def _colmap_like_database(path: Path) -> None:
    """The two tables `database_digest` needs, in COLMAP's journal mode (WAL)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute("pragma journal_mode=wal")
        con.execute("create table images (image_id integer primary key, name text)")
        con.execute("create table two_view_geometries "
                    "(pair_id integer primary key, rows integer, config integer, data blob)")
        con.execute("insert into images values (1, 'a.jpg'), (2, 'b.jpg')")
        con.commit()
    finally:
        con.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


IDS = ["s:00000000", "s:00000001", "s:00000002"]


def _ctx(ids=IDS) -> dict:
    """What `solve` hands `on_matched`: the horizon's keyframes and the solve's timings so far."""
    return {"keyframe_ids": list(ids), "prepare_s": 0.1, "extract_s": 0.2, "match_s": 0.3,
            "images_undistorted": len(ids), "threads": 18, "database_existed": True}


# ---------------------------------------------------------------------------
# the setting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,on", [(None, False), ("", False), ("off", False), ("0", False),
                                      ("garbage", False), ("on", True), ("true", True)])
def test_the_switch_is_off_unless_said(monkeypatch, value, on):
    from tower.config import (world_bg_stop_at_match_probe_setting,
                              world_bg_stop_at_match_setting)

    for name in (SWITCH, PROBE):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert world_bg_stop_at_match_setting() is on
    assert world_bg_stop_at_match_probe_setting() is on


# ---------------------------------------------------------------------------
# the marker
# ---------------------------------------------------------------------------


def test_the_marker_belongs_to_the_launch_that_wrote_it(tmp_path):
    root = _workspace(tmp_path)
    _colmap_like_database(root / "database.db")
    MM.ChildHook(root, token="t1", probe=False, final=False).matched(root / "database.db", _ctx())
    record = MM.read(root, "t1")
    assert record["record"] == MM.MARKER_RECORD and record["token"] == "t1"
    assert record["pid"] == os.getpid() and record["writer_free"] is True
    assert record["keyframe_ids"] == IDS and record["solve"]["match_s"] == 0.3
    # The probe off: no digest anywhere (R2 #3 / R1 N1), so no file hashes either.
    assert "digest" not in record and "files" not in record
    assert MM.read(root, "t2") is None               # another launch's marker
    assert MM.read(root, None) is None and MM.read(root, "") is None
    MM.clear(root)
    assert not MM.marker_path(root).exists()
    assert MM.read(root, "t1") is None
    MM.clear(root)                                   # clearing nothing is fine


_WHOLE = {"record": MM.MARKER_RECORD, "token": "t1", "pid": 7, "committed_at": 1.0,
          "writer_free": True, "keyframe_ids": IDS}


@pytest.mark.parametrize("body", [
    "{\"record\": \"wb-bg-matching-committed/2\", \"tok", "[]",
    json.dumps({"record": "other/1", "token": "t1"}), json.dumps({"token": "t1"}),
    json.dumps(dict(_WHOLE, record="wb-bg-matching-committed/1")),     # the round-1 marker
    json.dumps({k: v for k, v in _WHOLE.items() if k != "writer_free"}),
    json.dumps(dict(_WHOLE, writer_free=False)),
    json.dumps({k: v for k, v in _WHOLE.items() if k != "keyframe_ids"}),
    json.dumps(dict(_WHOLE, keyframe_ids=["s:1"])),
    json.dumps(dict(_WHOLE, pid="7")),
    json.dumps(dict(_WHOLE, digest=None)),                                # a digest, but no valid one
    json.dumps(dict(_WHOLE, digest={"error": "boom"})),
])
def test_a_torn_foreign_or_incomplete_marker_is_no_marker(tmp_path, body):
    root = _workspace(tmp_path)
    root.mkdir(parents=True)
    MM.marker_path(root).write_text(body, encoding="utf-8")
    assert MM.read(root, "t1") is None


def test_a_whole_marker_is_read(tmp_path):
    root = _workspace(tmp_path)
    root.mkdir(parents=True)
    MM.marker_path(root).write_text(json.dumps(_WHOLE), encoding="utf-8")
    assert MM.read(root, "t1") == _WHOLE


def test_read_never_raises_even_when_the_file_system_does(tmp_path, monkeypatch):
    """R1 N2: `exists()` itself can raise (a delete-pending file on Windows): no marker, no error."""
    root = _workspace(tmp_path)
    root.mkdir(parents=True)
    MM.marker_path(root).write_text(json.dumps(_WHOLE), encoding="utf-8")

    def refuse(self, *a, **k):
        raise PermissionError("delete pending")

    monkeypatch.setattr(Path, "exists", refuse)
    assert MM.read(root, "t1") is None


def test_a_staging_file_is_not_a_marker(tmp_path):
    """A child killed inside the atomic write leaves its staging file, never the marker."""
    from tower.storage import staging_path

    root = _workspace(tmp_path)
    root.mkdir(parents=True)
    staging_path(MM.marker_path(root)).write_text(
        json.dumps({"record": MM.MARKER_RECORD, "token": "t1"}), encoding="utf-8")
    assert MM.read(root, "t1") is None


def test_a_marker_that_cannot_be_written_is_no_marker_and_no_error(tmp_path, monkeypatch):
    import tower.storage as storage

    root = _workspace(tmp_path)
    _colmap_like_database(root / "database.db")

    def refuse(*_a, **_k):
        raise PermissionError("held")

    monkeypatch.setattr(storage, "write_json_atomic", refuse)
    MM.ChildHook(root, token="t1", probe=False, final=False).matched(root / "database.db", _ctx())
    assert MM.read(root, "t1") is None


def test_a_database_still_held_by_a_writer_is_no_marker(tmp_path):
    """R2 #3: the marker means the matching call returned AND no connection holds the database
    for writing. A writer still in a transaction: fail closed."""
    root = _workspace(tmp_path)
    _colmap_like_database(root / "database.db")
    holder = sqlite3.connect(root / "database.db")
    try:
        holder.execute("begin immediate")
        MM.ChildHook(root, token="t1", probe=False, final=False).matched(root / "database.db",
                                                                          _ctx())
        assert not MM.marker_path(root).exists()
    finally:
        holder.rollback()
        holder.close()
    MM.ChildHook(root, token="t1", probe=False, final=False).matched(root / "database.db", _ctx())
    assert MM.read(root, "t1") is not None


@pytest.mark.parametrize("bad", [None, {"error": "OperationalError: locked"}, {"content": ""}])
def test_in_probe_mode_a_digest_that_cannot_be_taken_is_no_marker(tmp_path, monkeypatch, bad):
    root = _workspace(tmp_path)
    _colmap_like_database(root / "database.db")
    monkeypatch.setattr(MM, "_digest", lambda _db: bad)
    MM.ChildHook(root, token="t1", probe=True, final=False).matched(root / "database.db", _ctx())
    assert not MM.marker_path(root).exists()
    # The probe still says what happened.
    (record,) = [json.loads(l) for l in (root / MM.PROBE_FILENAME).read_text().splitlines()]
    assert record["stage"] == "matched" and record["digest"] == bad


def test_no_token_and_no_probe_is_no_hook(tmp_path, monkeypatch):
    store = _Store(tmp_path)
    assert MM.for_child(store, "w", "s", final=False) is None
    assert MM.for_child(store, "w", "s", final=True) is None
    monkeypatch.setenv(MM.TOKEN_ENV, "t1")
    hook = MM.for_child(store, "w", "s", final=False)
    assert hook is not None and hook.token == "t1" and not hook.probe_on
    assert hook.root == _workspace(tmp_path)
    # A final solve never takes a token: the builder never ends a final child at a marker.
    assert MM.for_child(store, "w", "s", final=True) is None
    monkeypatch.setenv(PROBE, "on")
    final = MM.for_child(store, "w", "s", final=True)
    assert final is not None and final.token is None and final.probe_on


def test_the_probes_record_the_database_byte_for_byte(tmp_path):
    root = _workspace(tmp_path)
    database = root / "database.db"
    _colmap_like_database(database)
    hook = MM.ChildHook(root, token="t1", probe=True, final=False)
    hook.opened()
    hook.matched(database, _ctx())
    hook.finished()
    records = [json.loads(line) for line in
               (root / MM.PROBE_FILENAME).read_text(encoding="utf-8").splitlines()]
    assert [r["stage"] for r in records] == ["opened", "matched", "finished"]
    for record in records:
        assert record["probe"] == MM.PROBE_RECORD and record["final"] is False
        assert record["files"]["db"]["sha256"] == _sha256(database)
        assert record["digest"] == global_solve.database_digest(database)
    marker = MM.read(root, "t1")
    assert marker["files"] == records[1]["files"]     # the bytes at the commit, in the marker
    assert marker["digest"] == records[1]["digest"]


# ---------------------------------------------------------------------------
# where the commit is: global_solve.solve(on_matched=...)
# ---------------------------------------------------------------------------


class _Opts:
    def __init__(self):
        object.__setattr__(self, "_values", {})

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        values = object.__getattribute__(self, "_values")
        return values.setdefault(name, _Opts())

    def __setattr__(self, name, value):
        self._values[name] = value


class _FakeColmap(types.ModuleType):
    """Records the order of the calls that write the database and the ones that map it."""

    def __init__(self):
        super().__init__("pycolmap")
        self.calls = []
        self.mapped = []
        self.__version__ = "fake"
        self.logging = types.SimpleNamespace(minloglevel=0)
        self.CameraMode = types.SimpleNamespace(SINGLE="SINGLE")
        for kind in ("ImageReaderOptions", "FeatureExtractionOptions", "FeatureMatchingOptions",
                     "SequentialPairingOptions", "TwoViewGeometryOptions",
                     "ImportedPairingOptions", "GlobalPipelineOptions",
                     "IncrementalPipelineOptions"):
            setattr(self, kind, _Opts)

    def extract_features(self, database_path, *_a, **_k):
        self.calls.append("extract_features")
        Path(database_path).touch()

    def match_sequential(self, database_path, **_k):
        self.calls.append("match_sequential")

    def global_mapping(self, database_path, image_path, output_path, **_k):
        self.calls.append("global_mapping")
        self.mapped.append((Path(database_path), Path(output_path)))
        return {}

    def incremental_mapping(self, database_path, image_path, output_path, **_k):
        self.calls.append("incremental_mapping")
        return {}

    def set_random_seed(self, seed):
        self.calls.append(("set_random_seed", seed))


@pytest.fixture
def session(tmp_path):
    import cv2
    import numpy as np

    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics, Keyframe
    from tower.world_builder.store import WorldStore

    width, height = 64, 48
    store = WorldStore(tmp_path / "worlds")
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world()
    session_id = engine.start_session(
        world_id, frame_source="synthetic",
        intrinsics=CameraIntrinsics(source="self_calibrated", model="pinhole_radtan",
                                    fx=50.0, fy=50.0, cx=width / 2, cy=height / 2,
                                    dist_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
                                    calibrated_width=width, calibrated_height=height))
    rng = np.random.default_rng(0)
    for i in range(4):
        ok, buf = cv2.imencode(".jpg", rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
        assert ok
        store.write_keyframe_image(world_id, session_id, f"{i:08d}.jpg", buf.tobytes())
        store.append_keyframe(world_id, Keyframe(
            keyframe_id=f"{session_id}:{i:08d}", session_id=session_id, source_seq=i,
            received_at=float(i), image_relpath=f"images/{i:08d}.jpg",
            width=width, height=height, byte_count=1, segment_index=0))
    engine.stop_session("stopped")
    return types.SimpleNamespace(store=store, world_id=world_id, session_id=session_id,
                                 workspace=global_solve.workspace_for(store, world_id, session_id))


def _bg_solve(s, colmap, monkeypatch, **kw):
    monkeypatch.setitem(sys.modules, "pycolmap", colmap)
    return global_solve.solve(s.store, s.world_id, s.session_id, final=False, **kw)


def test_the_hook_runs_once_after_every_database_write_and_before_mapping(session, monkeypatch):
    colmap = _FakeColmap()
    seen = []

    def on_matched(database_path, context):
        seen.append((Path(database_path), list(colmap.calls), context))
        time.sleep(0.3)

    summary = _bg_solve(session, colmap, monkeypatch, on_matched=on_matched)
    ((database, calls, context),) = seen
    assert (database, calls) == (session.workspace.database_path,
                                 ["extract_features", "match_sequential"])
    # What a deferred map of this horizon needs: the keyframes this solve read, in its order.
    keyframes = session.store.read_keyframes(session.world_id, session.session_id)
    assert context["keyframe_ids"] == [k.keyframe_id for k in keyframes]
    assert set(context) == {"keyframe_ids", "prepare_s", "extract_s", "match_s",
                            "images_undistorted", "threads", "database_existed"}
    assert colmap.calls == ["extract_features", "match_sequential", "global_mapping",
                            "incremental_mapping"]
    # The hook's own time counts in neither match_s nor map_s.
    assert summary["timing"]["map_s"] < 0.3 and summary["timing"]["match_s"] < 0.3


def test_without_a_hook_the_solve_is_what_it_was(session, monkeypatch, tmp_path):
    """Same pycolmap calls, same published solution (wall-clock stamps and seconds aside). The
    first solve warms the workspace (images, database) so the two compared find the same one."""
    def published():
        meta = json.loads(session.workspace.solution_path.read_text(encoding="utf-8"))
        meta.pop("solved_at")
        meta["timing"] = {k: v for k, v in meta["timing"].items() if not k.endswith("_s")}
        return meta

    _bg_solve(session, _FakeColmap(), monkeypatch)
    plain = _FakeColmap()
    _bg_solve(session, plain, monkeypatch)
    without = published()
    hooked = _FakeColmap()
    _bg_solve(session, hooked, monkeypatch, on_matched=lambda _p, _c: None)
    assert hooked.calls == plain.calls
    assert published() == without


# ---------------------------------------------------------------------------
# the child: scripts/world_solve.py
# ---------------------------------------------------------------------------


def _world_solve_kwargs(monkeypatch, tmp_path, argv):
    import scripts.world_solve as world_solve
    from tower.world_builder.store import WorldStore

    captured = {}

    def fake_solve(store, world, session, **kwargs):
        captured.update(kwargs)
        if "on_matched" in kwargs:
            kwargs["on_matched"](global_solve.workspace_for(store, world, session).database_path,
                                 _ctx())
        return {"solved": False, "reason": "fake"}

    monkeypatch.setattr(world_solve.global_solve, "solve", fake_solve)
    monkeypatch.setattr(WorldStore, "read_keyframes", lambda self, w, s: [])
    _colmap_like_database(_workspace(tmp_path) / "database.db")
    assert world_solve.main(["--root", str(tmp_path), "--world", "w", "--session", "s",
                             *argv]) == 0
    return captured


@pytest.mark.parametrize("final", [False, True])
def test_the_child_calls_solve_as_before_with_no_token_and_no_probe(monkeypatch, tmp_path, final):
    kwargs = _world_solve_kwargs(monkeypatch, tmp_path, ["--final"] if final else [])
    assert "on_matched" not in kwargs
    assert not MM.marker_path(_workspace(tmp_path)).exists()
    assert not (_workspace(tmp_path) / MM.PROBE_FILENAME).exists()


def test_a_background_child_with_a_token_marks_its_commit(monkeypatch, tmp_path):
    monkeypatch.setenv(MM.TOKEN_ENV, "t1")
    kwargs = _world_solve_kwargs(monkeypatch, tmp_path, ["--loop-detection"])
    assert "on_matched" in kwargs
    assert MM.read(_workspace(tmp_path), "t1") is not None
    assert not (_workspace(tmp_path) / MM.PROBE_FILENAME).exists()


def test_a_final_child_never_marks_and_only_probes(monkeypatch, tmp_path):
    monkeypatch.setenv(MM.TOKEN_ENV, "t1")    # inherited by accident: still no marker
    monkeypatch.setenv(PROBE, "on")
    kwargs = _world_solve_kwargs(monkeypatch, tmp_path, ["--final"])
    assert "on_matched" in kwargs
    assert not MM.marker_path(_workspace(tmp_path)).exists()
    stages = [json.loads(line)["stage"] for line in
              (_workspace(tmp_path) / MM.PROBE_FILENAME).read_text().splitlines()]
    assert stages == ["opened", "matched", "finished"]


# ---------------------------------------------------------------------------
# the builder: BackgroundSolver.wait
# ---------------------------------------------------------------------------

# A stand-in background child. It commits a row the way pycolmap's matcher does (WAL, one
# transaction, connection closed), then behaves as MODE says, and records its environment.
STUB = textwrap.dedent(r'''
    import json, os, sqlite3, sys, time
    sys.path.insert(0, os.environ["WB_L2_TOWER"])
    from pathlib import Path
    from tower.world_builder import matching_marker as MM
    root = Path(os.environ["WB_L2_WORKSPACE"])
    (root / "child_env.json").write_text(json.dumps(dict(os.environ)))
    mode = os.environ["WB_L2_MODE"]
    if mode == "exit-at-once":
        sys.exit(0)
    con = sqlite3.connect(root / "database.db")
    con.execute("insert into two_view_geometries values (12, 40, 2, x'00ff')")
    con.commit()
    con.close()
    if mode == "commit-then-exit":
        sys.exit(0)
    if mode == "killed-before-marker":
        time.sleep(30)                 # the digest, say: the kill lands before the marker
    token = os.environ.get(MM.TOKEN_ENV)
    if mode == "foreign-marker":
        token = "not-this-launch"
    if mode == "grandchild":
        # A descendant that holds the database open (a reader), as a mapper helper might.
        import subprocess
        gc = subprocess.Popen([sys.executable, "-c", "import sqlite3,sys,time; c=sqlite3.connect(sys.argv[1]); "
                               "c.execute('select count(*) from images').fetchone(); time.sleep(60)",
                               str(root / "database.db")])
        (root / "grandchild.pid").write_text(str(gc.pid))
    if mode != "no-marker":
        probe = os.environ.get("TOWER_WORLD_BG_STOP_AT_MATCH_PROBE") == "on"
        ids = ["s:00000000", "s:00000001", "s:00000002"]
        MM.ChildHook(root, token=token, probe=probe, final=False).matched(
            root / "database.db", {"keyframe_ids": ids, "match_s": 1.5, "threads": 18})
    # "Mapping": a reader holds the database, and nothing more is written.
    reader = sqlite3.connect(root / "database.db")
    reader.execute("select count(*) from two_view_geometries").fetchone()
    time.sleep(30)
''')


def _launch(tmp_path, monkeypatch, mode, *, stop_at_match=True):
    root = _workspace(tmp_path)
    _colmap_like_database(root / "database.db")
    stub = tmp_path / "stub_solve.py"
    stub.write_text(STUB, encoding="utf-8")
    monkeypatch.setenv("WB_L2_TOWER", str(TOWER_DIR))
    monkeypatch.setenv("WB_L2_WORKSPACE", str(root))
    monkeypatch.setenv("WB_L2_MODE", mode)
    spawned = []

    def spawn(argv, **kwargs):
        spawned.append(kwargs.get("env"))
        return subprocess.Popen(argv, **kwargs)

    kw = {"stop_at_match": True} if stop_at_match else {}
    solver = BackgroundSolver(root=tmp_path, world_id="w", session_id="s", every=1,
                              capture_dirs=[], script=stub, spawn=spawn, **kw)
    assert solver.maybe_launch(_Store(tmp_path), accepted=5, sources={}) is True
    return solver, root, spawned


def _wait_for(path: Path, seconds=20.0):
    deadline = time.monotonic() + seconds
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert path.exists(), path


def test_stop_ends_the_child_at_its_matching_commit(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="tower.world_build_session")
    solver, root, spawned = _launch(tmp_path, monkeypatch, "marker")
    try:
        _wait_for(MM.marker_path(root))
        child = solver._child
        started = time.monotonic()
        assert solver.wait(60.0) is False                 # ended, not run to its end
        assert time.monotonic() - started < 15.0          # nowhere near the cap
        assert child.poll() is not None and not solver.running
        assert "ended at its matching commit" in caplog.text
        marker = MM.read(root, spawned[0][MM.TOKEN_ENV])
        database = root / "database.db"
        assert global_solve.database_digest(database)["verified_pairs"] == 1
        with sqlite3.connect(database) as con:
            assert con.execute("pragma quick_check").fetchone() == ("ok",)
        # The committed horizon is kept for a deferred map: a snapshot equal to what the final
        # opens, the child's keyframes, and the time the default path would have given it.
        snapshot = MM.snapshot_path(root)
        assert global_solve.database_digest(snapshot) == global_solve.database_digest(database)
        deferred = solver._deferred
        assert deferred["snapshot"] == str(snapshot) and deferred["keyframe_ids"] == IDS
        assert deferred["token"] == marker["token"] and deferred["standing_solved_at"] is None
        # R1-G: the Stop path's own cost of ending the child and of the copy, on its own line.
        assert "snapshotted in" in caplog.text
        # From its commit to the cap: the 60-s cap plus the time it committed before the wait began.
        assert 59.0 < deferred["budget_s"] < 80.0
    finally:
        solver.close()


def test_the_bytes_the_final_opens_are_the_bytes_committed(tmp_path, monkeypatch):
    """With the probe on the marker carries the file hashes at the commit; after the kill the
    database file is byte for byte that file."""
    monkeypatch.setenv(PROBE, "on")
    solver, root, spawned = _launch(tmp_path, monkeypatch, "marker")
    try:
        _wait_for(MM.marker_path(root))
        assert solver.wait(60.0) is False
        marker = MM.read(root, spawned[0][MM.TOKEN_ENV])
        now = MM.file_identity(root / "database.db")
        assert now["db"] == marker["files"]["db"]
        assert now["-journal"] is None and (now["-wal"] or {}).get("size", 0) == 0
    finally:
        solver.close()


def test_the_token_is_this_launchs_and_only_with_the_switch(tmp_path, monkeypatch):
    solver, root, spawned = _launch(tmp_path, monkeypatch, "exit-at-once")
    try:
        assert solver.wait(20.0) is True
        token = spawned[0][MM.TOKEN_ENV]
        assert token == solver._match_token and len(token) == 32
        env = json.loads((root / "child_env.json").read_text())
        assert env[MM.TOKEN_ENV] == token
        assert {k: v for k, v in spawned[0].items() if k != MM.TOKEN_ENV} == child_environment()
    finally:
        solver.close()


def test_switch_off_passes_no_token_and_ignores_any_marker(tmp_path, monkeypatch, caplog):
    """Off is today's wait to the cap, whatever lies in the workspace."""
    caplog.set_level(logging.INFO, logger="tower.world_build_session")
    root = _workspace(tmp_path)
    root.mkdir(parents=True)
    MM.marker_path(root).write_text(json.dumps({"record": MM.MARKER_RECORD, "token": "x"}))
    solver, root, spawned = _launch(tmp_path, monkeypatch, "no-marker", stop_at_match=False)
    try:
        assert spawned[0] == child_environment() and MM.TOKEN_ENV not in spawned[0]
        assert solver._match_token is None
        assert MM.marker_path(root).exists()             # not even cleared
        started = time.monotonic()
        assert solver.wait(1.0) is False
        assert time.monotonic() - started >= 1.0
        assert "still running after" in caplog.text
        assert "matching commit" not in caplog.text
    finally:
        solver.close()


def test_another_launchs_marker_is_ignored(tmp_path, monkeypatch, caplog):
    """A marker this launch did not write -- an earlier launch's whose removal failed, or one
    under another token -- never ends the wait."""
    caplog.set_level(logging.INFO, logger="tower.world_build_session")
    root = _workspace(tmp_path)
    root.mkdir(parents=True)
    MM.marker_path(root).write_text(json.dumps({"record": MM.MARKER_RECORD, "token": "old"}))
    monkeypatch.setattr(MM, "clear", lambda _root: None)   # the removal failed
    solver, root, _ = _launch(tmp_path, monkeypatch, "foreign-marker")
    try:
        time.sleep(1.0)
        started = time.monotonic()
        assert solver.wait(1.0) is False
        assert time.monotonic() - started >= 1.0
        assert "still running after" in caplog.text and "matching commit" not in caplog.text
    finally:
        solver.close()


def test_a_kill_between_the_commit_and_the_marker_is_todays_wait(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="tower.world_build_session")
    solver, root, _ = _launch(tmp_path, monkeypatch, "killed-before-marker")
    try:
        _wait_for(root / "child_env.json")
        started = time.monotonic()
        assert solver.wait(1.5) is False
        assert time.monotonic() - started >= 1.5
        assert "still running after" in caplog.text and "matching commit" not in caplog.text
        assert not MM.marker_path(root).exists()
        with sqlite3.connect(root / "database.db") as con:
            assert con.execute("pragma quick_check").fetchone() == ("ok",)
            assert con.execute("select count(*) from two_view_geometries").fetchone() == (1,)
    finally:
        solver.close()


@pytest.mark.parametrize("mode", ["commit-then-exit", "exit-at-once"])
def test_a_child_that_ends_first_is_todays_wait(tmp_path, monkeypatch, mode):
    solver, root, _ = _launch(tmp_path, monkeypatch, mode)
    try:
        assert solver.wait(20.0) is True
        assert not MM.marker_path(root).exists()
    finally:
        solver.close()


def test_a_wait_after_the_child_ended_returns_at_once(tmp_path, monkeypatch):
    solver, root, _ = _launch(tmp_path, monkeypatch, "marker")
    try:
        _wait_for(MM.marker_path(root))
        solver._child.kill()
        solver._child.wait(timeout=20)
        started = time.monotonic()
        assert solver.wait(60.0) is True
        assert time.monotonic() - started < 1.0
    finally:
        solver.close()


def test_the_final_child_is_never_ended_at_a_marker(tmp_path, monkeypatch):
    """`run_final` forgets the background token: the final solve runs to its own end even with
    the last background child's marker in the workspace."""
    solver, root, _ = _launch(tmp_path, monkeypatch, "marker")
    try:
        _wait_for(MM.marker_path(root))
        assert solver.wait(60.0) is False
        final = tmp_path / "final_stub.py"
        final.write_text("import json, time\ntime.sleep(1.0)\n"
                         "print(json.dumps({'solved': True, 'reason': 'stub'}))\n")
        solver.script = final
        started = time.monotonic()
        report = solver.run_final(_Store(tmp_path), {})
        assert report["reason"] == "stub" and time.monotonic() - started >= 1.0
        assert solver._match_token is None
    finally:
        solver.close()


def test_the_cap_is_checked_before_a_marker(tmp_path, monkeypatch, caplog):
    """R2 #2: at or past the cap a marker is not an early commit; it is the cap's kill, as today,
    and owes no deferred map (the default path's cap kills the child too)."""
    caplog.set_level(logging.INFO, logger="tower.world_build_session")
    solver, root, _ = _launch(tmp_path, monkeypatch, "marker")
    try:
        _wait_for(MM.marker_path(root))
        assert solver.wait(0.0) is False
        assert "still running after" in caplog.text and "matching commit" not in caplog.text
        assert solver._deferred is None and not MM.snapshot_path(root).exists()
    finally:
        solver.close()


def _grandchild(root: Path):
    import psutil

    _wait_for(root / "grandchild.pid")
    return psutil.Process(int((root / "grandchild.pid").read_text()))


def test_every_descendant_is_gone_before_the_wait_returns_at_a_marker(tmp_path, monkeypatch):
    """R2 #4: a child that spawned a grandchild holding the database: when the wait returns, the
    grandchild has exited too, before the snapshot and before any final solve."""
    solver, root, _ = _launch(tmp_path, monkeypatch, "grandchild")
    try:
        grandchild = _grandchild(root)
        _wait_for(MM.marker_path(root))
        assert grandchild.is_running()
        assert solver.wait(60.0) is False
        assert not grandchild.is_running()
        assert solver._deferred is not None and MM.snapshot_path(root).exists()
    finally:
        solver.close()
        if grandchild.is_running():
            grandchild.kill()


def test_every_descendant_is_gone_before_the_final_after_a_cap_kill(tmp_path, monkeypatch):
    """R2 #4, the cap's kill: the wait remembers the family, and `run_final` waits (bounded)
    for it before the final solve starts -- a final that records whether it is still alive."""
    solver, root, _ = _launch(tmp_path, monkeypatch, "grandchild")
    try:
        grandchild = _grandchild(root)
        _wait_for(MM.marker_path(root))
        assert solver.wait(0.0) is False                    # the cap, first
        assert solver._ended_family and solver._deferred is None
        final = tmp_path / "final_stub.py"
        final.write_text(textwrap.dedent(f'''
            import json, psutil
            print(json.dumps({{"solved": True, "grandchild_alive": psutil.pid_exists({grandchild.pid})
                              and psutil.Process({grandchild.pid}).is_running()}}))
        '''))
        solver.script = final
        report = solver.run_final(_Store(tmp_path), {})
        assert report["solved"] is True and report["grandchild_alive"] is False
        assert solver._ended_family == []
    finally:
        solver.close()
        if grandchild.is_running():
            grandchild.kill()


def test_the_switch_off_never_waits_for_descendants(tmp_path, monkeypatch):
    solver, root, _ = _launch(tmp_path, monkeypatch, "grandchild", stop_at_match=False)
    try:
        grandchild = _grandchild(root)
        assert solver.wait(0.5) is False
        assert solver._ended_family == [] and solver._deferred is None
    finally:
        solver.close()
        if grandchild.is_running():
            grandchild.kill()


# ---------------------------------------------------------------------------
# never an older world: the deferred map (manager's decision 1, round 2)
# ---------------------------------------------------------------------------

# A final solve that ends as WB_L2_FINAL says, and the deferred map (`--deferred-map <order>`),
# which records its order and publishes (prints solved) after WB_L2_DEFERRED_SLEEP seconds.
FINAL_STUB = textwrap.dedent(r'''
    import json, os, sys, time
    from pathlib import Path
    argv = sys.argv[1:]
    root = Path(os.environ["WB_L2_WORKSPACE"])
    if "--deferred-map" in argv:
        order = json.loads(Path(argv[argv.index("--deferred-map") + 1]).read_text())
        with (root / "deferred_calls.jsonl").open("a") as fh:
            fh.write(json.dumps({"order": order, "argv": argv,
                                 "snapshot_exists": Path(order["database"]).is_file()}) + "\n")
        dmode = os.environ.get("WB_L2_DEFERRED_MODE", "")
        if dmode == "claim-gap-exit":
            # Claims, then takes a while to write, then exits: a publication in flight at the bound.
            sys.path.insert(0, os.environ["WB_L2_TOWER"])
            from tower.world_builder import matching_marker as MM
            assert MM.claim_publish(order["claim"], "deferred-map")
            time.sleep(3)
            (root / "solution.json").write_text(json.dumps({"solved_at": time.time(), "who": "deferred"}))
            print(json.dumps({"solved": True}))
            sys.exit(0)
        if dmode in ("survive", "claim-early"):
            # Publish only through the real claim, as map_committed_horizon does.
            sys.path.insert(0, os.environ["WB_L2_TOWER"])
            from tower.world_builder import matching_marker as MM
            if dmode == "claim-early":
                # Claimed, and then a gap before the write (Codex round 4, #2): the window in which
                # a builder that reported at once would report before the publication lands.
                assert MM.claim_publish(order["claim"], "deferred-map")
                (root / "deferred_started").write_text("1")
                time.sleep(3)
                (root / "solution.json").write_text(json.dumps({"solved_at": time.time(), "who": "deferred"}))
                (root / "deferred_done").write_text("1")
                time.sleep(60)
            (root / "deferred_started").write_text("1")
            deadline = time.time() + 30
            while not (root / "go").exists() and time.time() < deadline:
                time.sleep(0.05)
            if MM.claim_publish(order["claim"], "deferred-map"):
                (root / "solution.json").write_text(json.dumps({"solved_at": time.time(), "who": "deferred"}))
            (root / "deferred_done").write_text("1")
            sys.exit(0)
        time.sleep(float(os.environ.get("WB_L2_DEFERRED_SLEEP", "0")))
        print(json.dumps({"solved": True, "solver": "glomap", "keyframes": len(order["keyframe_ids"]),
                          "keyframes_posed": 2, "components": [{"id": 0}]}))
        sys.exit(0)
    mode = os.environ["WB_L2_FINAL"]
    if mode.startswith("published-then"):
        # The final PUBLISHES (write_solution's last act), then fails before a summary reaches the
        # builder -- or is hard-stopped right after (R1 round 2, probe A).
        (root / "solution.json").write_text(json.dumps({"solved_at": time.time(), "who": "final"}))
        if mode == "published-then-crash":
            sys.exit(3)
        time.sleep(60)
    if mode == "solved":
        print(json.dumps({"solved": True, "solver": "glomap", "components": []}))
    elif mode == "failed":
        sys.exit(3)
    elif mode == "withheld":
        print(json.dumps({"solved": False, "withheld": True, "reason": "withheld by the guard"}))
    elif mode == "nothing":
        print(json.dumps({"solved": False, "reason": "fewer than two readable frames"}))
    elif mode == "hang":
        time.sleep(60)
''')


def _ended_at_marker(tmp_path, monkeypatch, final_mode):
    solver, root, spawned = _launch(tmp_path, monkeypatch, "marker")
    _wait_for(MM.marker_path(root))
    assert solver.wait(60.0) is False and solver._deferred is not None
    final = tmp_path / "final_stub.py"
    final.write_text(FINAL_STUB, encoding="utf-8")
    solver.script = final
    monkeypatch.setenv("WB_L2_FINAL", final_mode)
    return solver, root


def _deferred_calls(root: Path) -> list:
    path = root / "deferred_calls.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def test_a_solved_final_owes_nothing(tmp_path, monkeypatch):
    solver, root = _ended_at_marker(tmp_path, monkeypatch, "solved")
    try:
        report = solver.run_final(_Store(tmp_path), {})
        assert report["solved"] is True and "deferred_background_map" not in report
        assert _deferred_calls(root) == []
        assert not MM.snapshot_path(root).exists() and solver._deferred is None
    finally:
        solver.close()


@pytest.mark.parametrize("final_mode", ["failed", "withheld", "nothing"])
def test_a_final_that_publishes_nothing_gets_the_ended_childs_map(tmp_path, monkeypatch,
                                                                   final_mode):
    """The default path could have left the ended child's solution standing; so the deferred map
    runs, on the committed snapshot, with exactly the child's keyframes, and publishes."""
    solver, root = _ended_at_marker(tmp_path, monkeypatch, final_mode)
    try:
        report = solver.run_final(_Store(tmp_path), {})
        assert not report.get("solved")
        assert report.get("withheld") is (True if final_mode == "withheld" else None)
        ((call,),) = [_deferred_calls(root)]
        assert call["snapshot_exists"] is True
        assert call["order"]["record"] == MM.DEFERRED_ORDER_RECORD
        assert call["order"]["database"] == str(MM.snapshot_path(root))
        assert call["order"]["keyframe_ids"] == IDS and call["order"]["solve"]["match_s"] == 1.5
        assert call["order"]["standing_solved_at"] is None   # nothing stood at the snapshot
        import re
        assert re.fullmatch(r"bg_deferred_map\.[0-9a-f]{32}\.claim", Path(call["order"]["claim"]).name)
        assert "--deferred-map" in call["argv"] and "--final" not in call["argv"]
        outcome = report["deferred_background_map"]
        # Published per its summary; reported as such only once confirmed gone -- its Job Object on
        # Windows; on POSIX (no job) the status is unknown (None), as an unconfirmed outcome must be.
        assert outcome["ran"] is True and outcome["summary"]["solved"] is True
        assert outcome["published"] is (True if os.name == "nt" else None)
        assert outcome["summary"]["keyframes"] == 3
        # Afterwards nothing is left of it but what it published.
        assert not MM.snapshot_path(root).exists()
        assert not (root / MM.DEFERRED_ORDER_FILENAME).exists()
        assert solver._deferred is None and not solver.finished()
    finally:
        solver.close()


def test_a_hard_stop_maps_nothing_and_discards_the_snapshot(tmp_path, monkeypatch):
    """A hard stop (Tower shutdown) during the final: the final is interrupted as today, nothing is
    mapped after it, the builder goes on at once, and the snapshot is discarded (round 3: the
    accepted residual -- the standing world may be one background solve older than off)."""
    solver, root = _ended_at_marker(tmp_path, monkeypatch, "hang")
    started = time.monotonic()
    try:
        report = solver.run_final(_Store(tmp_path), {},
                                  should_stop=lambda: time.monotonic() - started > 1.0)
        assert report["interrupted"] is True and report["solved"] is False
        assert report["deferred_background_map"] == {"ran": False, "published": False,
                                                     "why": "hard stop"}
        assert _deferred_calls(root) == [] and time.monotonic() - started < 15.0
        assert not MM.snapshot_path(root).exists()
        assert not (root / MM.DEFERRED_ORDER_FILENAME).exists()
    finally:
        solver.close()


@pytest.mark.parametrize("final_mode", ["published-then-crash", "published-then-hang"])
def test_a_final_that_published_is_never_overwritten(tmp_path, monkeypatch, final_mode):
    """R1 round 2, A: a final that published and then exited non-zero (or was hard-stopped after its
    publish) reports `not solved`; the deferred map is bound to the solution standing at the snapshot,
    so it is moot: never spawned, and the final's solution stands."""
    solver, root = _ended_at_marker(tmp_path, monkeypatch, final_mode)
    started = time.monotonic()

    def stop():
        return (final_mode == "published-then-hang" and time.monotonic() - started > 1.5
                and json.loads((root / "solution.json").read_text())["who"] == "final")

    try:
        report = solver.run_final(_Store(tmp_path), {}, should_stop=stop)
        assert not report.get("solved")
        assert json.loads((root / "solution.json").read_text())["who"] == "final"
        assert _deferred_calls(root) == []
        assert report["deferred_background_map"]["published"] is False
        assert not MM.snapshot_path(root).exists()
    finally:
        solver.close()


def test_a_run_final_that_never_runs_discards_the_snapshot(tmp_path, monkeypatch, caplog):
    """R1 round 2, D: a hard stop between the marker kill and the Stop branch's `run_final` (the
    builder skips the final): `close()` discards the snapshot and says so."""
    caplog.set_level(logging.INFO, logger="tower.world_build_session")
    solver, root = _ended_at_marker(tmp_path, monkeypatch, "failed")
    assert MM.snapshot_path(root).exists()
    solver.close()
    assert not MM.snapshot_path(root).exists() and solver._deferred is None
    assert "its snapshot is discarded" in caplog.text


def test_a_surviving_descendant_fails_closed_to_todays_wait(tmp_path, monkeypatch, caplog):
    """Round 3, decision 2: a process of the ended child still alive after the bounded wait: no
    snapshot, no deferred map, and the final does not start early -- the wait goes on to the cap."""
    caplog.set_level(logging.INFO, logger="tower.world_build_session")
    solver, root, _ = _launch(tmp_path, monkeypatch, "marker")
    monkeypatch.setattr(solver, "_await_gone", lambda child, family, **kw: False)
    try:
        _wait_for(MM.marker_path(root))
        started = time.monotonic()
        assert solver.wait(2.0) is False
        assert time.monotonic() - started >= 2.0             # the cap, not the marker
        assert solver._deferred is None and not MM.snapshot_path(root).exists()
        assert "failing closed to today's path" in caplog.text
    finally:
        solver.close()


def test_a_surviving_descendant_still_honours_a_stop(tmp_path, monkeypatch):
    solver, root, _ = _launch(tmp_path, monkeypatch, "marker")
    monkeypatch.setattr(solver, "_await_gone", lambda child, family, **kw: False)
    try:
        _wait_for(MM.marker_path(root))
        started = time.monotonic()
        assert solver.wait(60.0, should_stop=lambda: time.monotonic() - started > 0.5) is False
        assert time.monotonic() - started < 10.0 and solver._deferred is None
    finally:
        solver.close()


def test_a_hard_stop_during_the_deferred_map_ends_it(tmp_path, monkeypatch):
    solver, root = _ended_at_marker(tmp_path, monkeypatch, "failed")
    monkeypatch.setenv("WB_L2_DEFERRED_SLEEP", "60")
    try:
        report = solver.run_final(
            _Store(tmp_path), {}, should_stop=lambda: len(_deferred_calls(root)) > 0)
        outcome = report["deferred_background_map"]
        assert outcome["ran"] is True and outcome["published"] is False
        assert outcome["why"] == "hard stop" and not solver.running
        assert not MM.snapshot_path(root).exists()          # the accepted residual: discarded
    finally:
        solver.close()


def test_a_deferred_map_past_the_default_paths_time_is_ended(tmp_path, monkeypatch):
    """Bounded by the time the default path gave the child (its commit to the cap) plus the start
    allowance: past that the default path's cap would have killed it too."""
    import scripts.world_build_session as builder

    solver, root = _ended_at_marker(tmp_path, monkeypatch, "failed")
    monkeypatch.setenv("WB_L2_DEFERRED_SLEEP", "60")
    monkeypatch.setattr(builder, "DEFERRED_MAP_START_ALLOWANCE_S", 0.0)
    solver._deferred["budget_s"] = 1.5
    started = time.monotonic()
    try:
        report = solver.run_final(_Store(tmp_path), {})
        outcome = report["deferred_background_map"]
        assert outcome["published"] is False and "not mapped within 1.5 s" in outcome["why"]
        assert time.monotonic() - started < 20.0 and not solver.running
    finally:
        solver.close()


def test_a_deferred_child_that_survives_the_stop_cannot_publish(tmp_path, monkeypatch):
    """R2 round 3, #2: the builder ends the deferred map (a stop) and cannot confirm the child gone.
    It revokes the publication by taking the claim first; the surviving child, carrying on, cannot
    claim it and publishes nothing. The revocation stays on disk while the child may live."""
    solver, root = _ended_at_marker(tmp_path, monkeypatch, "failed")
    monkeypatch.setenv("WB_L2_DEFERRED_MODE", "survive")
    monkeypatch.setattr(solver, "_terminate_child_verified", lambda: False)   # it survives
    survivor = []
    real_spawn = solver._spawn

    def spawn(argv, **kwargs):
        child = real_spawn(argv, **kwargs)
        if "--deferred-map" in argv:
            survivor.append(child)
        return child

    solver._spawn = spawn
    try:
        report = solver.run_final(_Store(tmp_path), {},
                                  should_stop=lambda: (root / "deferred_started").exists())
        outcome = report["deferred_background_map"]
        assert outcome["published"] is False and outcome["revoked"] is True
        assert outcome["confirmed_gone"] is False
        (claim,) = list(root.glob("bg_deferred_map.*.claim"))
        assert "builder-revoked" in claim.read_text()
        (root / "go").write_text("1")                                   # the survivor carries on
        survivor[0].wait(timeout=30)
        assert (root / "deferred_done").exists()
        assert not (root / "solution.json").exists()                    # and published nothing
        assert "builder-revoked" in claim.read_text()
    finally:
        for child in survivor:
            if child.poll() is None:
                child.kill()
        solver.close()


def test_a_deferred_child_that_claimed_first_is_not_reported_until_it_is_gone(tmp_path,
                                                                               monkeypatch):
    """Codex round 4, #2: a child that claimed the publication and is stopped BEFORE it writes. The
    builder reports nothing until the child is confirmed gone: on Windows its Job Object confirms it
    (the kill lands before the write: `published` False, and the write never comes); without a job
    (POSIX, the dev platform) it cannot be confirmed: `published` is None (unknown), the claim stays,
    and whatever lands on disk is what stands."""
    solver, root = _ended_at_marker(tmp_path, monkeypatch, "failed")
    monkeypatch.setenv("WB_L2_DEFERRED_MODE", "claim-early")
    real_verified = solver._terminate_child_verified
    monkeypatch.setattr(solver, "_terminate_child_verified", lambda: False)   # not confirmed
    deferred_child = []
    real_spawn = solver._spawn

    def spawn(argv, **kwargs):
        child = real_spawn(argv, **kwargs)
        if "--deferred-map" in argv:
            deferred_child.append(child)
        return child

    solver._spawn = spawn
    try:
        report = solver.run_final(_Store(tmp_path), {},
                                  should_stop=lambda: (root / "deferred_started").exists())
        outcome = report["deferred_background_map"]
        assert "already claimed" in outcome["why"] and "revoked" not in outcome
        (claim,) = list(root.glob("bg_deferred_map.*.claim"))
        assert "deferred-map" in claim.read_text()
        if os.name == "nt":
            assert outcome["published"] is False and outcome["confirmed_gone"] is True
            time.sleep(5)
            assert not (root / "solution.json").exists()
        else:
            assert outcome["published"] is None and outcome["confirmed_gone"] is False
            assert claim.exists()
    finally:
        for child in deferred_child:
            if child.poll() is None:
                child.kill()
        solver._terminate_child_verified = real_verified
        solver.close()


def test_a_publication_in_flight_at_the_bound_gets_its_grace(tmp_path, monkeypatch):
    """Round 5 nit: at the bound (not a hard stop) the builder tries to revoke first; a child that
    already holds the claim is publishing and gets `DEFERRED_PUBLISH_GRACE_S` to finish before it is
    ended, so the kill does not land inside its write."""
    import scripts.world_build_session as builder

    solver, root = _ended_at_marker(tmp_path, monkeypatch, "failed")
    monkeypatch.setenv("WB_L2_DEFERRED_MODE", "claim-gap-exit")
    monkeypatch.setattr(builder, "DEFERRED_MAP_START_ALLOWANCE_S", 0.0)
    monkeypatch.setattr(builder, "DEFERRED_PUBLISH_GRACE_S", 15.0)
    solver._deferred["budget_s"] = 2.0          # the bound falls between its claim and its write
    try:
        report = solver.run_final(_Store(tmp_path), {})
        outcome = report["deferred_background_map"]
        assert "not mapped within 2.0 s" in outcome["why"] and "already claimed" in outcome["why"]
        assert json.loads((root / "solution.json").read_text())["who"] == "deferred"
        assert outcome["published"] is (True if os.name == "nt" else None)
    finally:
        solver.close()


def test_the_claim_is_taken_once(tmp_path):
    claim = MM.claim_path(tmp_path, "r1")
    assert claim.name == "bg_deferred_map.r1.claim"
    assert MM.claim_publish(claim, "deferred-map") is True
    assert MM.claim_publish(claim, "builder-revoked") is False
    assert "deferred-map" in claim.read_text()
    assert MM.claim_publish(tmp_path / "no-such-dir" / "c", "x") is False   # cannot: no claim


def test_a_revoked_deferred_map_publishes_nothing(session, monkeypatch):
    """`map_committed_horizon(publish_claim=...)`: the builder's revocation already there -> no write."""
    _bg_solve(session, _FakeColmap(), monkeypatch)
    ws = session.workspace
    standing = json.loads(ws.solution_path.read_text(encoding="utf-8"))["solved_at"]
    before = ws.solution_path.read_bytes()
    claim = MM.claim_path(ws.root, "r1")
    assert MM.claim_publish(claim, "builder-revoked")
    monkeypatch.setitem(sys.modules, "pycolmap", _FakeColmap())
    ids = [k.keyframe_id for k in session.store.read_keyframes(session.world_id,
                                                                session.session_id)[:3]]
    summary = global_solve.map_committed_horizon(
        session.store, session.world_id, session.session_id, database_path=ws.database_path,
        keyframe_ids=ids, expect_standing_solved_at=standing, publish_claim=claim)
    assert summary["solved"] is False and summary["revoked"] is True
    assert ws.solution_path.read_bytes() == before
    claim.unlink()
    summary = global_solve.map_committed_horizon(
        session.store, session.world_id, session.session_id, database_path=ws.database_path,
        keyframe_ids=ids, expect_standing_solved_at=standing, publish_claim=claim)
    assert summary["solved"] is True and "deferred-map" in claim.read_text()


def test_an_untrusted_enumeration_is_never_gone(tmp_path, monkeypatch):
    """R2 round 3, #1: a family that could not be enumerated (psutil failing, the parent exiting
    first or mid-way) is never "confirmed gone"."""
    import psutil

    from scripts.world_build_session import BackgroundSolver as BS

    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait(timeout=20)
    assert BS._family(done) == ([], False)                       # exited: its orphans are unknown
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert BS._family(live) == ([], True)
        monkeypatch.setattr(psutil, "Process",
                            lambda pid: (_ for _ in ()).throw(psutil.AccessDenied(pid)))
        assert BS._family(live) == ([], False)
    finally:
        live.kill()
        live.wait(timeout=20)
    solver = BS(root=tmp_path, world_id="w", session_id="s", every=1, capture_dirs=[],
                stop_at_match=True)
    assert solver._await_gone(live, [], enumerated=False) is False
    assert solver._await_gone(live, [], enumerated=True) is True


def test_an_untrusted_enumeration_fails_closed_to_todays_wait(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="tower.world_build_session")
    import scripts.world_build_session as builder

    solver, root, _ = _launch(tmp_path, monkeypatch, "marker")
    monkeypatch.setattr(builder.BackgroundSolver, "_family", staticmethod(lambda child: ([], False)))
    try:
        _wait_for(MM.marker_path(root))
        started = time.monotonic()
        assert solver.wait(2.0) is False
        assert time.monotonic() - started >= 2.0
        assert solver._deferred is None and not MM.snapshot_path(root).exists()
        assert "could not be enumerated" in caplog.text
    finally:
        solver.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")
def test_the_job_confirms_a_grandchild_nobody_enumerated_is_gone(tmp_path):
    """Windows: the kernel's `ActiveProcesses` answers for the whole tree, so a grandchild that was
    never enumerated is still waited for -- and an untrusted enumeration does not matter."""
    from scripts.world_build_session import BackgroundSolver as BS
    from tower.process_ownership import assign_to_job

    child = subprocess.Popen([sys.executable, "-c",
                              "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', "
                              "'import time; time.sleep(60)']); time.sleep(60)"])
    job = assign_to_job(child)
    assert job is not None
    try:
        deadline = time.monotonic() + 20
        while (job.active_processes() or 0) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert job.active_processes() == 2
        job.terminate()
        solver = BS(root=tmp_path, world_id="w", session_id="s", every=1, capture_dirs=[],
                    stop_at_match=True)
        child.wait(timeout=20)
        assert solver._await_gone(child, [], enumerated=False, job=job) is True
        assert job.active_processes() == 0
    finally:
        job.close()
        if child.poll() is None:
            child.kill()


def test_a_child_that_cannot_be_contained_runs_as_today(tmp_path, monkeypatch):
    """Codex round 4, #1, the fallback: a switch-on child started suspended that cannot be put in a
    Job Object (here: no job on this platform) is killed before it ran a line and launched again as
    today -- without a token, so it is never ended early."""
    import scripts.world_build_session as builder

    monkeypatch.setattr(builder, "_suspended_launch", lambda: True)
    root = _workspace(tmp_path)
    _colmap_like_database(root / "database.db")
    stub = tmp_path / "stub_solve.py"
    stub.write_text(STUB, encoding="utf-8")
    monkeypatch.setenv("WB_L2_TOWER", str(TOWER_DIR))
    monkeypatch.setenv("WB_L2_WORKSPACE", str(root))
    monkeypatch.setenv("WB_L2_MODE", "marker")
    spawned = []

    def spawn(argv, **kwargs):
        flags = kwargs.pop("creationflags", 0)    # recorded; the stand-in child starts normally
        child = subprocess.Popen(argv, **kwargs)
        spawned.append((flags, kwargs.get("env"), child))
        return child

    monkeypatch.setattr("tower.process_ownership.assign_to_job", lambda child: None)
    solver = BackgroundSolver(root=tmp_path, world_id="w", session_id="s", every=1,
                              capture_dirs=[], script=stub, spawn=spawn, stop_at_match=True)
    try:
        assert solver.maybe_launch(_Store(tmp_path), accepted=5, sources={}) is True
        assert len(spawned) == 2
        (flags0, env0, first), (flags1, env1, second) = spawned
        assert flags0 == builder.CREATE_SUSPENDED and MM.TOKEN_ENV in env0
        first.wait(timeout=10)                                   # killed
        assert flags1 == 0 and MM.TOKEN_ENV not in env1 and env1 == child_environment()
        assert solver._match_token is None and solver._match_job is None
        assert solver._child is second
        _wait_for(root / "child_env.json")
        assert solver.wait(1.0) is False                         # today's: the cap, no marker
        assert not MM.marker_path(root).exists() or MM.read(root, None) is None
    finally:
        solver.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows: CREATE_SUSPENDED + Job Objects")
def test_a_suspended_launch_leaves_no_process_outside_the_job(tmp_path, monkeypatch):
    """Codex round 4, #1: the child starts suspended, joins its job, and only then runs -- so a
    grandchild it spawns on its very first line is in the job too (no launch race)."""
    root = _workspace(tmp_path)
    root.mkdir(parents=True)
    stub = tmp_path / "stub_solve.py"
    stub.write_text("import subprocess, sys, time\n"
                    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                    "time.sleep(60)\n", encoding="utf-8")
    solver = BackgroundSolver(root=tmp_path, world_id="w", session_id="s", every=1,
                              capture_dirs=[], script=stub, stop_at_match=True)
    try:
        assert solver.maybe_launch(_Store(tmp_path), accepted=5, sources={}) is True
        job = solver._match_job
        assert job is not None and solver._match_token is not None
        deadline = time.monotonic() + 20
        while (job.active_processes() or 0) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert job.active_processes() == 2                       # it ran, and its grandchild is in
        assert solver._terminate_child_verified() is True
        assert job.active_processes() == 0
    finally:
        solver.close()


def test_the_switch_off_never_owes_a_deferred_map(tmp_path, monkeypatch):
    solver, root, _ = _launch(tmp_path, monkeypatch, "marker", stop_at_match=False)
    try:
        _wait_for(root / "child_env.json")
        assert solver.wait(0.5) is False and solver._deferred is None
        final = tmp_path / "final_stub.py"
        final.write_text(FINAL_STUB, encoding="utf-8")
        solver.script = final
        monkeypatch.setenv("WB_L2_FINAL", "failed")
        report = solver.run_final(_Store(tmp_path), {})
        assert "deferred_background_map" not in report and _deferred_calls(root) == []
    finally:
        solver.close()


def test_the_deferred_map_maps_the_snapshot_with_the_childs_keyframes(session, monkeypatch):
    """`global_solve.map_committed_horizon`: the snapshot, not the walk database; exactly the
    horizon's keyframes (not the ones the final added); its own scratch, `sparse/` untouched;
    published as a background solve would be."""
    _bg_solve(session, _FakeColmap(), monkeypatch)            # camera.json, images, a database
    colmap = _FakeColmap()
    monkeypatch.setitem(sys.modules, "pycolmap", colmap)
    ws = session.workspace
    keyframes = session.store.read_keyframes(session.world_id, session.session_id)
    horizon = [k.keyframe_id for k in keyframes[:3]]
    snapshot = ws.root / MM.SNAPSHOT_FILENAME
    snapshot.write_bytes(ws.database_path.read_bytes())
    (ws.sparse_dir / "final-model").mkdir(parents=True, exist_ok=True)
    summary = global_solve.map_committed_horizon(
        session.store, session.world_id, session.session_id, database_path=snapshot,
        keyframe_ids=horizon, num_threads=18, context={"match_s": 1.5, "threads": 18})
    assert colmap.mapped and colmap.mapped[0][0] == snapshot
    assert colmap.mapped[0][1] == ws.root / global_solve.DEFERRED_SPARSE_DIRNAME
    assert not (ws.root / global_solve.DEFERRED_SPARSE_DIRNAME).exists()
    assert (ws.sparse_dir / "final-model").is_dir()
    assert summary["solved"] is True and summary["keyframes"] == 3
    published = json.loads(ws.solution_path.read_text(encoding="utf-8"))
    assert published["keyframe_ids"] == horizon
    assert published["solve"]["deferred_map"] is True and published["solve"]["final"] is False
    assert published["solve"]["seed"] is None and published["timing"]["match_s"] == 1.5
    from tower.world_builder.store import compute_input_digest
    assert published["input_digest"] == compute_input_digest(keyframes[:3])


def test_the_deferred_map_publishes_only_over_the_solution_standing_at_the_snapshot(session,
                                                                                  monkeypatch):
    _bg_solve(session, _FakeColmap(), monkeypatch)            # publishes a solution
    ws = session.workspace
    standing = json.loads(ws.solution_path.read_text(encoding="utf-8"))["solved_at"]
    before = ws.solution_path.read_bytes()
    ids = [k.keyframe_id for k in session.store.read_keyframes(session.world_id,
                                                                session.session_id)[:3]]
    monkeypatch.setitem(sys.modules, "pycolmap", _FakeColmap())
    for expected in (None, standing + 1.0):                   # nothing stood / another stood
        summary = global_solve.map_committed_horizon(
            session.store, session.world_id, session.session_id, database_path=ws.database_path,
            keyframe_ids=ids, expect_standing_solved_at=expected)
        assert summary["solved"] is False and summary["moot"] is True
        assert ws.solution_path.read_bytes() == before
    ws.solution_path.write_text("{torn", encoding="utf-8")   # unreadable: never overwritten
    summary = global_solve.map_committed_horizon(
        session.store, session.world_id, session.session_id, database_path=ws.database_path,
        keyframe_ids=ids, expect_standing_solved_at=None)
    assert summary.get("moot") is True and ws.solution_path.read_text() == "{torn"
    ws.solution_path.write_bytes(before)
    summary = global_solve.map_committed_horizon(
        session.store, session.world_id, session.session_id, database_path=ws.database_path,
        keyframe_ids=ids, expect_standing_solved_at=standing)
    assert summary["solved"] is True
    assert json.loads(ws.solution_path.read_text())["keyframe_ids"] == ids


def test_the_deferred_map_refuses_keyframes_no_longer_in_the_session(session, monkeypatch):
    monkeypatch.setitem(sys.modules, "pycolmap", _FakeColmap())
    summary = global_solve.map_committed_horizon(
        session.store, session.world_id, session.session_id,
        database_path=session.workspace.database_path, keyframe_ids=["nope:1", "nope:2"])
    assert summary["solved"] is False and "not in the session" in summary["reason"]


def test_world_solve_deferred_map_hands_the_order_over(monkeypatch, tmp_path):
    import scripts.world_solve as world_solve

    captured = {}

    def fake(store, world, session, **kwargs):
        captured.update(kwargs, world=world, session=session)
        return {"solved": True}

    monkeypatch.setattr(world_solve.global_solve, "map_committed_horizon", fake)
    monkeypatch.setattr(world_solve.global_solve, "solve",
                        lambda *a, **k: pytest.fail("a deferred map is not a solve"))
    order = tmp_path / "order.json"
    order.write_text(json.dumps({"record": MM.DEFERRED_ORDER_RECORD, "token": "t",
                                 "database": str(tmp_path / "snap.db"), "keyframe_ids": IDS,
                                 "solve": {"match_s": 1.5}, "standing_solved_at": 7.5}))
    assert world_solve.main(["--root", str(tmp_path), "--world", "w", "--session", "s",
                             "--deferred-map", str(order)]) == 0
    assert captured["database_path"] == tmp_path / "snap.db"
    assert captured["keyframe_ids"] == IDS and captured["context"] == {"match_s": 1.5}
    assert captured["num_threads"] == world_solve.background_threads()
    assert captured["expect_standing_solved_at"] == 7.5
    # An order that does not say what stood at the snapshot is not an order.
    captured.clear()
    order.write_text(json.dumps({"record": MM.DEFERRED_ORDER_RECORD, "database": "x",
                                 "keyframe_ids": IDS}))
    assert world_solve.main(["--root", str(tmp_path), "--world", "w", "--session", "s",
                             "--deferred-map", str(order)]) == 0
    assert captured == {}


# ---------------------------------------------------------------------------
# the whole builder, switch on against off
# ---------------------------------------------------------------------------


def test_on_changes_nothing_but_the_background_childs_environment(tmp_path, monkeypatch):
    """`world_build_session.main` on synthetic frames with background solves (a stub child that
    records its argv and environment and writes no marker): with the switch on, every file under
    the root, every event, the report and every argv are the switch-off run's, and the only
    difference is the token in each BACKGROUND child's environment -- never the final's. Off is
    pinned to the 5c0dc12 golden in `test_world_builder_finish_stages_identity.py`. A soft Stop
    only: a hard one kills the stub wherever it happens to be, which no switch decides."""
    import scripts.world_build_session as builder_script
    from tests.test_world_builder_finish_stages_identity import FRAMES, _run_builder

    frames, intrinsics = builder_script.synthetic_frames(FRAMES, 480, 360)
    rendered = ([frame.payload for frame in frames], intrinsics)
    runs = {}
    for value in (None, "on"):
        if value is None:
            monkeypatch.delenv(SWITCH, raising=False)
        else:
            monkeypatch.setenv(SWITCH, value)
        runs[value] = _run_builder("soft", rendered, tmp_path / f"run-{value}", monkeypatch,
                                   solve_every=2)
    off, on = runs[None], runs["on"]
    background = [c for c in off["solve_calls"] if "--final" not in c["argv"]]
    assert background, "no background solve ran"
    for key in ("exit", "report", "events", "files"):
        assert on[key] == off[key], key
    assert len(on["solve_calls"]) == len(off["solve_calls"])
    for a, b in zip(off["solve_calls"], on["solve_calls"]):
        assert a["argv"] == b["argv"]
        assert a["env_changed"] == b["env_changed"] and a["env_removed"] == b["env_removed"]
        expected = sorted(a["env_added"] + ([MM.TOKEN_ENV] if "--final" not in a["argv"] else []))
        assert b["env_added"] == expected
