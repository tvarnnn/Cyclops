"""Fast W0-1 selection and private-child failure tests; no mapper or GPU runs."""

import hashlib
import ctypes
import dataclasses
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tower.config import world_solve_consensus_concurrent_setting
from tower.world_builder import global_solve as GS
from tower.world_builder import coherence_publish as CP
from tower.world_builder import stage_timing
from tower.world_builder.store import WorldStore
from tests.test_world_builder_reproducible_finish import engines, walk  # noqa: F401
from tests.test_world_builder_solve_masks import colmap  # noqa: F401
from tests.test_world_builder_solve_consensus import (  # noqa: F401
    PIECES, SID, _candidate, _Store, world)


def _base():
    return GS.Solution(
        solver="glomap", solved_at=1.0, input_digest="input", keyframe_ids=[],
        poses={}, components=[], xyz=np.zeros((0, 3), np.float32),
        rgb=np.zeros((0, 3), np.uint8), component=np.zeros(0, np.int32),
        first_keyframe=np.zeros(0, np.int32), track_length=np.zeros(0, np.int32),
        error=np.zeros(0, np.float32), observations=np.zeros((0, 3), np.int32),
        camera=dict(fx=1, fy=1, cx=0, cy=0, width=1, height=1),
        solve={"seed": 3, "threads": 1}, timing={"map_s": 99},
        transients={"state": "applied"})


class _Process:
    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0


class _Job:
    def close(self):
        pass


def _rig(tmp_path, monkeypatch, *, bad_copy=False, bad_child=False):
    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w", "s")
    ws.root.mkdir(parents=True)
    ws.database_path.write_bytes(b"db")
    launched = []
    serial = []
    events = []

    def digest(path):
        data = Path(path).read_bytes()
        return {"content": hashlib.sha1(data).hexdigest()}

    def copy(source, destination):
        Path(destination).write_bytes(b"bad" if bad_copy else Path(source).read_bytes())

    def launch(context, database, seed, sparse, output, expected_digest):
        launched.append((seed, Path(database)))
        content = digest(database)["content"]
        with Path(output).open("wb") as handle:
            pickle.dump({"before": content, "after": "bad" if bad_child else content,
                         "map_ms": 1250, "candidate": _base()}, handle)
        return _Process(), _Job()

    def frozen(*args, **kwargs):
        def draw(seed):
            serial.append(seed)
            return _base()
        return draw

    monkeypatch.setattr(GS, "database_digest", digest)
    monkeypatch.setattr(GS, "_private_draw_database", copy)
    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 32 * 1024**3)
    monkeypatch.setattr(GS, "frozen_draw_mapper", frozen)
    monkeypatch.setattr(stage_timing, "concurrent_draw_event", events.append)
    return store, ws, launched, serial, events


def test_config_defaults_and_rejects_later_value(monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT", raising=False)
    assert world_solve_consensus_concurrent_setting() == "off"
    monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT", "after-draw-0")
    assert world_solve_consensus_concurrent_setting() == "after-draw-0"
    monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT", "at-freeze")
    assert world_solve_consensus_concurrent_setting() == "off"


def test_final_solve_selects_only_explicit_concurrent_mode(walk, engines, colmap, monkeypatch):
    from tests.test_world_builder_solve_masks import StubDetector

    chosen = []

    def mapper(label):
        def make(store, world_id, session_id, database_path, base, **kwargs):
            chosen.append(label)
            return lambda seed: base
        return make

    monkeypatch.setattr(GS, "frozen_draw_mapper", mapper("serial"))
    monkeypatch.setattr(GS, "concurrent_draw_mapper", mapper("concurrent"))
    for setting in ("off", "after-draw-0"):
        monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT", setting)
        GS.solve(walk.store, walk.world_id, walk.session_id, final=True, masks=True,
                 seed=0, gate=True, consensus=3, transient_backend_factory=StubDetector(),
                 mask_device_probe=lambda: None)
    assert chosen == ["serial", "concurrent"]


def test_lazy_private_launch_and_seed_order(tmp_path, monkeypatch):
    store, ws, launched, serial, _ = _rig(tmp_path, monkeypatch)
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                       _base(), seeds=(4, 5), keyframes=[])
    assert launched == []  # draw 0 did not attach or consensus did not ask for draw 1
    assert mapper(4).solve["seed"] == 4
    assert [seed for seed, _ in launched] == [4, 5]
    assert len({path for _, path in launched}) == 2
    assert all(path != ws.database_path for _, path in launched)
    assert mapper(5).solve["seed"] == 5
    assert serial == []
    mapper.close()


@pytest.mark.parametrize("cause", ["ram", "copy", "child"])
def test_resource_or_digest_refusal_uses_serial(tmp_path, monkeypatch, cause):
    store, ws, launched, serial, events = _rig(
        tmp_path, monkeypatch, bad_copy=cause == "copy", bad_child=cause == "child")
    if cause == "ram":
        monkeypatch.setattr(GS, "_draw_free_ram", lambda: 0)
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                       _base(), seeds=(4, 5), keyframes=[])
    assert mapper(4).solve["seed"] == 3  # sentinel from the unchanged serial mapper
    assert serial == [4]
    assert events and "mismatch" in events[-1] if cause != "ram" else events == ["ram-refusal:seed-4"]
    assert launched == [] if cause != "child" else len(launched) == 2
    mapper.close()


def test_child_timeout_uses_serial_even_with_soft_stop(tmp_path, monkeypatch):
    store, ws, launched, serial, events = _rig(tmp_path, monkeypatch)
    stopped = []

    class Blocking(_Process):
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("fake draw", timeout)

        def poll(self):
            return None

    def launch(context, database, seed, sparse, output, expected_digest):
        launched.append(seed)
        return Blocking(), _Job()

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    monkeypatch.setattr(GS, "_stop_draw_children", lambda children: stopped.append(tuple(children)) or True)
    calls = []

    def should_stop():
        calls.append(1)
        return len(calls) > 1

    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path, _base(),
                                       seeds=(4, 5), keyframes=[], should_stop=should_stop)
    assert mapper(4).solve["seed"] == 3
    assert serial == [4] and launched == [4, 5]
    assert stopped == [(4, 5)] and events == ["after-draw-0", "child-timeout:seed-4"]
    assert calls == []  # ordinary soft stop does not kill a running draw
    mapper.close()


def test_child_crash_remaps_same_seed_serially(tmp_path, monkeypatch):
    store, ws, launched, serial, events = _rig(tmp_path, monkeypatch)

    class Crashed(_Process):
        def wait(self, timeout=None):
            return 7

    monkeypatch.setattr(GS, "_launch_draw_child",
                        lambda *args: (Crashed(), _Job()))
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                       _base(), seeds=(4, 5), keyframes=[])
    assert mapper(4).solve["seed"] == 3
    assert serial == [4]
    assert events[-1] == "child-exit-7:seed-4"
    assert not (ws.root / "sparse-draws").exists()
    journal = json.loads((ws.root / "consensus_concurrent.jsonl").read_text())
    assert journal["seed"] == 4 and journal["reason"] == "child-exit-7"
    mapper.close()


def test_parent_thread_counts_are_untouched(tmp_path, monkeypatch):
    import torch

    # This pinned NumPy bundles OpenBLAS under numpy.libs; query its actual pool,
    # since threadpoolctl is not installed in the Tower venv.
    blas = next((Path(np.__path__[0]).parent / "numpy.libs").glob("*openblas*.dll"))
    get_threads = ctypes.CDLL(str(blas)).scipy_openblas_get_num_threads64_
    get_threads.restype = ctypes.c_int

    store, ws, _, _, _ = _rig(tmp_path, monkeypatch)
    before = (torch.get_num_threads(), torch.get_num_interop_threads())
    blas_before = get_threads()
    numpy_pool = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    numpy_before = {name: os.environ.get(name) for name in numpy_pool}
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                       _base(), seeds=(4, 5), keyframes=[])
    mapper(4)
    mapper.close()
    assert (torch.get_num_threads(), torch.get_num_interop_threads()) == before
    assert get_threads() == blas_before
    assert {name: os.environ.get(name) for name in numpy_pool} == numpy_before


def test_child_timing_joins_in_seed_order_and_fallback_stays_in_i0(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("TOWER_WORLD_STAGE_TIMING", "on")
    store = WorldStore(tmp_path)
    store.session_dir("w", "s").mkdir(parents=True)

    @stage_timing.timed("gate")
    def gate(store, world_id, session_id):
        stage_timing.child_draw_timing(4, 1200)
        stage_timing.child_draw_timing(5, 900)
        stage_timing.concurrent_draw_event("ram-refusal")

    @stage_timing.timed("solve")
    def solve(store, world_id, session_id):
        gate(store, world_id, session_id)

    solve(store, "w", "s")
    doc = json.loads((store.session_dir("w", "s") / stage_timing.FILENAME).read_text())
    stages = doc["finishes"][0]["stages"]
    assert [(d["seed"], d["draw_index"], d["wall_ms"]) for d in stages["solve_draw"]] == [
        (4, 0, 1200), (5, 1, 900)]
    assert stages["solve"][0]["consensus_concurrent"] == "ram-refusal"


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ownership")
def test_job_object_kills_draw_child_when_parent_exits(tmp_path):
    import psutil

    pid_file = tmp_path / "child-pid.txt"
    parent_code = (
        "import os, subprocess, sys\n"
        "from tower.process_ownership import assign_to_job\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)'])\n"
        "job = assign_to_job(child)\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid) + ' ' + str(job is not None))\n"
        "os._exit(0)\n"
    )
    subprocess.run([sys.executable, "-c", parent_code], check=True, timeout=5)
    pid, owned = pid_file.read_text().split()
    assert owned == "True"
    deadline = time.monotonic() + 3
    while psutil.pid_exists(int(pid)) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not psutil.pid_exists(int(pid))


@pytest.mark.slow
def test_real_child_maps_full_schema_database_like_serial(tmp_path, monkeypatch):
    import pycolmap

    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w", "s")
    ws.root.mkdir(parents=True)
    ws.images_dir.mkdir(parents=True)
    # pycolmap creates its complete current schema, so its mapper cannot
    # silently add tables and trip the after-digest check.
    with pycolmap.Database.open(str(ws.database_path)):
        pass
    events = []
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 32 * 1024**3)
    monkeypatch.setattr(stage_timing, "concurrent_draw_event", events.append)
    order = []
    from tower import process_ownership
    assign = process_ownership.assign_to_job
    resume = GS._resume_draw_child
    monkeypatch.setattr(process_ownership, "assign_to_job",
                        lambda proc: order.append("assigned") or assign(proc))
    monkeypatch.setattr(GS, "_resume_draw_child",
                        lambda proc: order.append("resumed") or resume(proc))
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                       _base(), seeds=(4,), keyframes=[])
    try:
        concurrent = mapper(4)
    finally:
        mapper.close()
    assert events == ["after-draw-0"]
    assert order == ["assigned", "resumed"]
    assert not (ws.root / "sparse-draws").exists()
    serial = GS.frozen_draw_mapper(store, "w", "s", ws.database_path,
                                   _base(), keyframes=[])(4)
    for field in dataclasses.fields(serial):
        if field.name in ("solved_at", "timing"):
            continue
        left, right = getattr(serial, field.name), getattr(concurrent, field.name)
        if isinstance(left, np.ndarray):
            assert left.dtype == right.dtype and left.shape == right.shape
            assert left.tobytes() == right.tobytes(), field.name
        else:
            assert left == right, field.name


def test_child_exit_preserves_three_voter_published_room(world, tmp_path, monkeypatch):
    draws = [(), ("A",), ()]

    def candidate(seed):
        return _candidate(tuple(PIECES), rotated=draws[seed - 7])

    def publish(plan):
        result = CP.gate_by_consensus(
            _Store(), "w1", SID, candidate(7), plan=plan,
            database_path="db", keyframes=world.keyframes)
        return (result.record["consensus"]["state"],
                result.record["consensus"]["votes"]["draws"],
                sorted(k for k, p in result.solution.poses.items()
                       if p["component"] == 0))

    serial = publish(CP.ConsensusPlan(draws=3, seed=7, map_draw=candidate))
    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w1", SID)
    ws.root.mkdir(parents=True)
    ws.database_path.write_bytes(b"db")
    digest = lambda p: {"content": hashlib.sha1(Path(p).read_bytes()).hexdigest()}
    monkeypatch.setattr(GS, "database_digest", digest)
    monkeypatch.setattr(GS, "_private_draw_database",
                        lambda s, d: Path(d).write_bytes(Path(s).read_bytes()))
    monkeypatch.setattr(GS, "_draw_free_ram", lambda: 64 * 1024**3)
    monkeypatch.setattr(GS, "frozen_draw_mapper",
                        lambda *a, **kw: candidate)

    def launch(context, database, seed, sparse, output, expected):
        with Path(output).open("wb") as handle:
            pickle.dump({"before": expected, "after": expected,
                         "map_ms": 1.0, "candidate": candidate(seed)}, handle)
        return _ExitProcess(1 if seed == 9 else 0), _Job()

    monkeypatch.setattr(GS, "_launch_draw_child", launch)
    base = candidate(7)
    base.solve = {"seed": 7}
    mapper = GS.concurrent_draw_mapper(store, "w1", SID, ws.database_path,
                                       base, seeds=(8, 9), keyframes=[])
    try:
        concurrent = publish(CP.ConsensusPlan(draws=3, seed=7, map_draw=mapper))
    finally:
        mapper.close()
    assert concurrent == serial
    assert concurrent[0] == CP.CONSENSUS_APPLIED
    assert concurrent[1] == 3 and len(concurrent[2]) == 90
    assert not (ws.root / "sparse-draws").exists()
    journal = (ws.root / "consensus_concurrent.jsonl").read_text()
    assert "child-exit-1" in journal and '"seed": 9' in journal


class _ExitProcess(_Process):
    def __init__(self, code):
        self.code = code

    def wait(self, timeout=None):
        return self.code

    def poll(self):
        return self.code


@pytest.mark.slow
def test_child_script_uses_requested_seed_one_thread_and_after_digest(tmp_path):
    import pycolmap

    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w", "s")
    ws.root.mkdir(parents=True)
    ws.images_dir.mkdir(parents=True)
    with pycolmap.Database.open(str(ws.database_path)):
        pass
    context = tmp_path / "context.pkl"
    with context.open("wb") as handle:
        pickle.dump((ws, [], GS.PinholeCamera.from_json_dict(_base().camera),
                     "input", GS.MIN_IMAGE_OBSERVATIONS), handle)
    expected = GS.database_digest(ws.database_path)["content"]
    shim = tmp_path / "shim"
    shim.mkdir()
    observed = tmp_path / "observed.json"
    (shim / "sitecustomize.py").write_text(
        "import json, os, sqlite3\n"
        "from pathlib import Path\n"
        "from tower.world_builder import global_solve as GS\n"
        "def mapped(*args, **kwargs):\n"
        "    Path(os.environ['W01_OBSERVED']).write_text(json.dumps("
        "{'seed': kwargs['seed'], 'threads': kwargs['threads']}))\n"
        "    db = sqlite3.connect(str(args[1]))\n"
        "    db.execute('create table after_probe (id integer)')\n"
        "    db.commit(); db.close()\n"
        "    return None\n"
        "GS._map_candidate = mapped\n", encoding="utf-8")
    env = dict(os.environ, W01_OBSERVED=str(observed), CUDA_VISIBLE_DEVICES="-1",
               PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(shim) + os.pathsep + str(Path(GS.__file__).parents[2]))
    output = tmp_path / "candidate.pkl"
    script = Path(GS.__file__).parents[2] / "scripts" / "world_solve_draw.py"
    result = subprocess.run([
        sys.executable, str(script), "--context", str(context),
        "--database", str(ws.database_path), "--seed", "17",
        "--sparse", str(tmp_path / "sparse"), "--output", str(output),
        "--expected-digest", expected], env=env, capture_output=True, text=True,
        timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(observed.read_text()) == {"seed": 17, "threads": 1}
    with output.open("rb") as handle:
        child = pickle.load(handle)
    assert child["before"] == expected
    assert child["after"] != expected


def test_switch_off_does_not_write_concurrent_i0_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_STAGE_TIMING", "on")
    monkeypatch.setenv("TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT", "off")
    store = WorldStore(tmp_path)
    store.session_dir("w", "s").mkdir(parents=True)

    @stage_timing.timed("solve")
    def solve(store, world_id, session_id):
        return None

    solve(store, "w", "s")
    record = json.loads((store.session_dir("w", "s") /
                         stage_timing.FILENAME).read_text())["finishes"][0]["stages"]["solve"][0]
    assert "consensus_concurrent" not in record


def test_soft_stop_during_successful_child_wait_keeps_child(tmp_path, monkeypatch):
    store, ws, launched, serial, events = _rig(tmp_path, monkeypatch)
    stopped = []
    monkeypatch.setattr(GS, "_stop_draw_children",
                        lambda children: stopped.append(tuple(children)) or True)
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                       _base(), seeds=(4, 5), keyframes=[],
                                       should_stop=lambda: True)
    assert mapper(4).solve["seed"] == 4
    assert serial == [] and events == ["after-draw-0"]
    assert stopped == []
    mapper.close()
    assert stopped == [(5,)]


def test_stale_writer_swept_but_live_writer_preserved(tmp_path, monkeypatch):
    import psutil

    store, ws, _, _, _ = _rig(tmp_path, monkeypatch)
    root = ws.root / "sparse-draws"
    root.mkdir()
    (root / "writer.json").write_text(json.dumps(
        {"pid": 99999999, "created_at": 1.0}))
    (root / "stale.bin").write_bytes(b"stale")
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                       _base(), seeds=(4,), keyframes=[])
    mapper(4)
    mapper.close()
    assert not root.exists()

    root.mkdir()
    (root / "writer.json").write_text(json.dumps(
        {"pid": os.getpid(), "created_at": psutil.Process().create_time()}))
    (root / "live.bin").write_bytes(b"live")
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                       _base(), seeds=(4,), keyframes=[])
    with pytest.raises(RuntimeError, match="another consensus writer"):
        mapper(4)
    mapper.close()
    assert (root / "live.bin").read_bytes() == b"live"


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ownership")
def test_product_launcher_job_and_stop_end_real_child(tmp_path, monkeypatch):
    import psutil
    from tower import process_ownership

    monkeypatch.setattr(process_ownership, "interpreter_argv",
                        lambda *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    output = tmp_path / "seed" / "candidate.pkl"
    output.parent.mkdir()
    process, job = GS._launch_draw_child(tmp_path / "context", tmp_path / "db", 4,
                                         tmp_path / "sparse", output, "digest")
    try:
        assert job is not None and psutil.pid_exists(process.pid)
    finally:
        assert GS._stop_draw_children({4: (process, job)})
    assert process.poll() is not None
