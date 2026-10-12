"""Stop closes every live-solve launch and hands committed state to final."""

import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.world_build_session import (  # noqa: E402
    BackgroundSolver, child_environment, python_executable,
)


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
        "import os,sqlite3,sys,time\n"
        "con=sqlite3.connect(sys.argv[1])\n"
        "con.execute('begin immediate')\n"
        "con.execute('insert into features values (2)')\n"
        "open(sys.argv[2],'w').write(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import subprocess,sys,time\n"
        "subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2],sys.argv[3]])\n"
        "time.sleep(0.2)\n"
    )
    child = subprocess.Popen([sys.executable, str(launcher), str(writer),
                              str(database), str(marker)])
    solver = solver_at(tmp_path)
    solver._child = child
    solver._track_background_descendants(child)
    writer_process = None
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        writer_process = psutil.Process(int(marker.read_text()))
        child.wait(timeout=5)  # the parent exits while its writer still owns SQLite
        assert writer_process.is_running()
        solver.stop_background(store)
        assert child.poll() is not None
        assert not writer_process.is_running()
        final = tmp_path / "final.py"
        final.write_text(
            "import json,sqlite3,sys\n"
            "with sqlite3.connect(sys.argv[1],timeout=1) as con:\n"
            "    rows=[row[0] for row in con.execute('select id from features')]\n"
            "    con.execute('insert into features values (3)')\n"
            "print(json.dumps({'solved':True,'rows':rows}))\n"
        )
        solver._argv = lambda *, final: [python_executable(), str(final_script), str(database)]
        final_script = final
        assert solver.run_final(store, {})["rows"] == [1]
        with sqlite3.connect(database, timeout=1) as con:
            assert con.execute("pragma quick_check").fetchone() == ("ok",)
            assert con.execute("select id from features").fetchall() == [(1,), (3,)]
        solver.stop_background(store)
        assert solver.maybe_launch(store, 100, {}) is False
    finally:
        solver.close()
        if writer_process is not None and writer_process.is_running():
            writer_process.kill()
            writer_process.wait(timeout=5)
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.parametrize("failure", ["no_op", "raises"])
def test_failed_immediate_kill_waits_and_still_runs_final(tmp_path, monkeypatch, failure):
    store = Store(tmp_path)
    started = tmp_path / "background-started"
    done = tmp_path / "background-done"
    background = tmp_path / "background.py"
    background.write_text(
        "import sys,time\nopen(sys.argv[1],'w').close()\n"
        "time.sleep(0.2)\nopen(sys.argv[2],'w').close()\n"
    )
    final = tmp_path / "final.py"
    final.write_text(
        "import json,os,sys\n"
        "print(json.dumps({'solved': True, 'background_done': os.path.exists(sys.argv[1])}))\n"
    )
    solver = solver_at(tmp_path, fallback_wait_seconds=5.0)
    solver._child = subprocess.Popen(
        [python_executable(), str(background), str(started), str(done)],
        env=child_environment(),
    )
    def fail_termination():
        if failure == "raises":
            raise RuntimeError("simulated kill failure")

    monkeypatch.setattr(solver, "_terminate_child", fail_termination)
    solver._argv = lambda *, final: [python_executable(), str(final_script), str(done)]
    final_script = final
    try:
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started.exists()
        solver.stop_background(store)
        assert solver.final_immediate_fallback is True
        assert done.exists(), f"background exit: {solver._child and solver._child.poll()}"
        assert solver.run_final(store, {})["background_done"] is True
    finally:
        solver.close()


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
