"""Stop closes every live-solve launch and hands committed state to final."""

import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.world_build_session import BackgroundSolver  # noqa: E402


class Store:
    def __init__(self, root):
        self.root = root

    def world_dir(self, world_id):
        return self.root / "worlds" / world_id


def solver_at(tmp_path, **kwargs):
    return BackgroundSolver(root=tmp_path, world_id="w", session_id="s", every=10,
                            capture_dirs=[], final_immediate=True, **kwargs)


def test_latch_blocks_threshold_landing_and_retry_launches(tmp_path, monkeypatch):
    store = Store(tmp_path)
    stopped = False
    solver = solver_at(tmp_path, capture_stopped=lambda: stopped)
    spawned = []
    monkeypatch.setattr(solver, "_spawn", lambda *a, **k: spawned.append((a, k)))

    # All three rebuild paths call maybe_launch, including a landing below
    # the normal threshold. The guard is inside the solver, not the loop.
    stopped = True
    for accepted in (10, 2, 100):
        assert solver.maybe_launch(store, accepted, {}) is False
    assert solver.launches == 0
    assert spawned == []
    solver.stop_background(store)  # repeated Stop
    assert solver.maybe_launch(store, 1000, {}) is False


def test_stop_during_spawn_kills_new_child(tmp_path):
    store = Store(tmp_path)
    stopped = False
    solver = solver_at(tmp_path, capture_stopped=lambda: stopped)
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n")
    children = []

    def spawn(*args, **kwargs):
        nonlocal stopped
        child = subprocess.Popen([sys.executable, str(script)])
        children.append(child)
        stopped = True
        return child

    solver._spawn = spawn
    try:
        assert solver.maybe_launch(store, 10, {}) is False
        assert children[0].poll() is not None
        assert solver.launches == 0
        assert solver.maybe_launch(store, 20, {}) is False
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()


def test_off_keeps_existing_launch_path(tmp_path):
    store = Store(tmp_path)
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n")
    solver = BackgroundSolver(root=tmp_path, world_id="w", session_id="s", every=10,
                              capture_dirs=[], capture_stopped=lambda: True)
    solver._spawn = lambda *a, **k: subprocess.Popen([sys.executable, str(script)])
    try:
        assert solver.maybe_launch(store, 10, {}) is True
        assert solver.launches == 1
        solver.stop_background(store)  # OFF is an exact no-op
        assert solver.running
    finally:
        solver.close()


def test_stop_kills_writer_mid_transaction_and_final_sees_committed_rows(tmp_path):
    store = Store(tmp_path)
    database = store.world_dir("w") / "solve" / "s" / "database.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as con:
        con.execute("create table features (id integer primary key)")
        con.execute("insert into features values (1)")
    marker = tmp_path / "writer-ready"
    writer = tmp_path / "writer.py"
    writer.write_text(
        "import sqlite3,sys,time\n"
        "con=sqlite3.connect(sys.argv[1])\n"
        "con.execute('begin immediate')\n"
        "con.execute('insert into features values (2)')\n"
        "open(sys.argv[2],'w').close()\n"
        "time.sleep(30)\n"
    )
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import subprocess,sys,time\n"
        "subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2],sys.argv[3]])\n"
        "time.sleep(30)\n"
    )
    child = subprocess.Popen([sys.executable, str(launcher), str(writer),
                              str(database), str(marker)])
    solver = solver_at(tmp_path)
    solver._child = child
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        solver.stop_background(store)
        assert child.poll() is not None
        with sqlite3.connect(database, timeout=1) as con:
            assert con.execute("pragma quick_check").fetchone() == ("ok",)
            assert con.execute("select id from features").fetchall() == [(1,)]
            con.execute("insert into features values (3)")  # final writer can proceed
        solver.stop_background(store)
        assert solver.maybe_launch(store, 100, {}) is False
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_unusable_database_is_removed_before_final(tmp_path):
    store = Store(tmp_path)
    database = store.world_dir("w") / "solve" / "s" / "database.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"not a SQLite database")
    (database.parent / "database.matching.json").write_text("stale")
    solver = solver_at(tmp_path)
    solver.stop_background(store)
    assert not database.exists()
    assert not (database.parent / "database.matching.json").exists()
    with sqlite3.connect(database) as con:
        con.execute("create table fresh (id integer)")


def test_late_stop_does_not_kill_final_child(tmp_path):
    store = Store(tmp_path)
    marker = tmp_path / "final-ready"
    script = tmp_path / "final.py"
    script.write_text(
        "import json,sys,time\n"
        "open(sys.argv[1],'w').close()\n"
        "time.sleep(0.5)\n"
        "print(json.dumps({'solved':True}))\n"
    )
    solver = solver_at(tmp_path)
    solver.stop_background(store)
    solver._argv = lambda *, final: [sys.executable, str(script), str(marker)]
    result = {}
    thread = threading.Thread(target=lambda: result.update(solver.run_final(store, {})))
    thread.start()
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert marker.exists()
        solver.stop_background(store)
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert result["solved"] is True
    finally:
        solver.close()
