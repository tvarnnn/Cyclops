"""Fast W0-1 selection and private-child failure tests; no mapper or GPU runs."""

import hashlib
import ctypes
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
from tower.world_builder import stage_timing
from tower.world_builder.store import WorldStore
from tests.test_world_builder_reproducible_finish import engines, walk  # noqa: F401
from tests.test_world_builder_solve_masks import colmap  # noqa: F401


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
    def wait(self):
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
    assert events and "mismatch" in events[-1] if cause != "ram" else events == ["ram-refusal"]
    assert launched == [] if cause != "child" else len(launched) == 2
    mapper.close()


def test_soft_stop_terminates_children_before_serial_record(tmp_path, monkeypatch):
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
    assert stopped == [(4, 5)] and events == ["after-draw-0", "stop-during-map"]
    mapper.close()


def test_child_crash_raises_mapping_failure(tmp_path, monkeypatch):
    store, ws, launched, serial, _ = _rig(tmp_path, monkeypatch)

    class Crashed(_Process):
        def wait(self):
            return 7

    monkeypatch.setattr(GS, "_launch_draw_child",
                        lambda *args: (Crashed(), _Job()))
    mapper = GS.concurrent_draw_mapper(store, "w", "s", ws.database_path,
                                       _base(), seeds=(4, 5), keyframes=[])
    with pytest.raises(RuntimeError, match="exited 7"):
        mapper(4)
    assert serial == []
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
