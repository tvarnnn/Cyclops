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
    MM.ChildHook(root, token="t1", probe=False, final=False).matched(root / "database.db")
    record = MM.read(root, "t1")
    assert record["record"] == MM.MARKER_RECORD and record["token"] == "t1"
    assert record["pid"] == os.getpid()
    assert record["digest"] == global_solve.database_digest(root / "database.db")
    assert "files" not in record                     # the probe is off
    assert MM.read(root, "t2") is None               # another launch's marker
    assert MM.read(root, None) is None and MM.read(root, "") is None
    MM.clear(root)
    assert not MM.marker_path(root).exists()
    assert MM.read(root, "t1") is None
    MM.clear(root)                                   # clearing nothing is fine


@pytest.mark.parametrize("body", ["{\"record\": \"wb-bg-matching-committed/1\", \"tok",
                                  "[]", json.dumps({"record": "other/1", "token": "t1"}),
                                  json.dumps({"token": "t1"})])
def test_a_torn_or_foreign_marker_is_no_marker(tmp_path, body):
    root = _workspace(tmp_path)
    root.mkdir(parents=True)
    MM.marker_path(root).write_text(body, encoding="utf-8")
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
    MM.ChildHook(root, token="t1", probe=False, final=False).matched(root / "database.db")
    assert MM.read(root, "t1") is None


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
    hook.matched(database)
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

    def global_mapping(self, *_a, **_k):
        self.calls.append("global_mapping")
        return {}

    def incremental_mapping(self, *_a, **_k):
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

    def on_matched(database_path):
        seen.append((Path(database_path), list(colmap.calls)))
        time.sleep(0.3)

    summary = _bg_solve(session, colmap, monkeypatch, on_matched=on_matched)
    assert seen == [(session.workspace.database_path, ["extract_features", "match_sequential"])]
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
    _bg_solve(session, hooked, monkeypatch, on_matched=lambda _p: None)
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
            kwargs["on_matched"](global_solve.workspace_for(store, world, session).database_path)
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
    if mode != "no-marker":
        probe = os.environ.get("TOWER_WORLD_BG_STOP_AT_MATCH_PROBE") == "on"
        MM.ChildHook(root, token=token, probe=probe, final=False).matched(root / "database.db")
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
        # The database the final solve opens is the one the child committed: digest and bytes.
        marker = MM.read(root, spawned[0][MM.TOKEN_ENV])
        database = root / "database.db"
        assert global_solve.database_digest(database) == marker["digest"]
        assert marker["digest"]["verified_pairs"] == 1
        with sqlite3.connect(database) as con:
            assert con.execute("pragma quick_check").fetchone() == ("ok",)
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
                         "print(json.dumps({'solved': False, 'reason': 'stub'}))\n")
        solver.script = final
        started = time.monotonic()
        report = solver.run_final(_Store(tmp_path), {})
        assert report["reason"] == "stub" and time.monotonic() - started >= 1.0
        assert solver._match_token is None
    finally:
        solver.close()


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
