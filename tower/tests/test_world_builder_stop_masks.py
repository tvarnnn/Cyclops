"""Stop mask child ownership, isolation and cache handoff."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import world_build_session as B
from scripts import world_solve_masks as M
from tower.world_builder import global_solve, solve_masks as SM
from tower.world_builder.engine import WorldBuilderEngine
from tower.world_builder.records import CameraIntrinsics, Keyframe
from tower.world_builder.store import WorldStore
from tests.test_world_builder_solve_masks import StubDetector


class Job:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class Process:
    def __init__(self, rc=None):
        self.pid = os.getpid()
        self.returncode = rc

    def poll(self):
        return self.returncode


def solver(root):
    class Solver:
        running = True
        world_id = "w"
        session_id = "s"

        def __init__(self):
            self.root = root

        def wait(self, timeout, *, should_stop):
            self.timeout = timeout

    return Solver()


def test_stop_mask_switch_defaults_off(monkeypatch):
    from tower.config import world_solve_masks_at_stop_setting
    monkeypatch.delenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", raising=False)
    assert world_solve_masks_at_stop_setting() is False
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", "on")
    assert world_solve_masks_at_stop_setting() is True


def test_launch_contract_and_job_assignment(tmp_path, monkeypatch):
    process, job, seen = Process(rc=1), Job(), {}
    monkeypatch.setattr(B, "assign_to_job", lambda p: job if p is process else None)
    monkeypatch.setattr(B, "python_executable", lambda: "python")
    monkeypatch.setattr(B, "child_environment", lambda: {"sentinel": "yes"})

    def spawn(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return process

    child = B.StopSolverMasks(solver(tmp_path), script=Path("stub.py"), spawn=spawn)
    assert child.launch()
    assert seen["argv"] == ["python", "stub.py", "--root", str(tmp_path),
                             "--world", "w", "--session", "s"]
    assert seen["kwargs"]["stdin"] == B.subprocess.DEVNULL
    assert seen["kwargs"]["stderr"] == B.subprocess.STDOUT
    assert seen["kwargs"]["env"]["sentinel"] == "yes"
    assert seen["kwargs"]["env"]["OMP_NUM_THREADS"] == "2"
    if os.name == "nt":
        assert seen["kwargs"]["creationflags"] == B.subprocess.BELOW_NORMAL_PRIORITY_CLASS
    assert child.join(should_stop=lambda: False)
    assert job.closed
    assert (tmp_path / "worlds" / "w" / "solve" / "s" /
            "solve_masks_at_stop.log").exists()


def test_job_assignment_failure_disables_prefill(tmp_path, monkeypatch):
    process = Process()
    monkeypatch.setattr(B, "assign_to_job", lambda p: None)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: setattr(process, "returncode", 1) or True)
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process)
    assert not child.launch()
    assert process.poll() == 1 and child._child is None


def test_launch_exception_leaves_background_wait_available(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "python_executable", lambda: (_ for _ in ()).throw(OSError("bad python")))
    s = solver(tmp_path)
    assert B.wait_for_background_solve_at_stop(s, 0.2,
                                               should_stop=lambda: False)
    assert s.timeout == 0.2


def test_child_startup_consumes_original_stop_deadline(tmp_path, monkeypatch):
    class SlowLaunch:
        _child = None

        def __init__(self, *args, **kwargs):
            pass

        def launch(self):
            time.sleep(0.08)
            return True

        def join(self, **kwargs):
            return True

    monkeypatch.setattr(B, "StopSolverMasks", SlowLaunch)
    s = solver(tmp_path)
    stop_at = time.monotonic()
    assert B.wait_for_background_solve_at_stop(
        s, 0.05, should_stop=lambda: False, stop_at=stop_at)
    assert s.timeout == 0.0


@pytest.mark.parametrize("reason", ["timeout", "soft", "hard"])
def test_join_terminates_hung_child_before_return(tmp_path, monkeypatch, reason):
    process, job = Process(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    calls = []

    def terminate(p, **kw):
        calls.append(kw)
        p.returncode = 1
        return True

    monkeypatch.setattr(B, "terminate_tree", terminate)
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process,
                              join_timeout=-1 if reason == "timeout" else 1800)
    assert child.launch()
    assert child.join(should_stop=lambda: reason == "hard",
                      should_soft_stop=lambda: reason == "soft")
    assert calls and calls[0]["hard"]
    assert process.poll() == 1 and job.closed


def test_failed_termination_retains_owned_child_and_blocks_final(tmp_path, monkeypatch):
    process, job = Process(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: False)
    child = B.StopSolverMasks(solver(tmp_path), spawn=lambda *a, **k: process,
                              join_timeout=-1)
    assert child.launch()
    try:
        assert child.join(should_stop=lambda: False) is False
        assert child in B._owned_stalled_mask_children
        assert not job.closed and child._child is process
    finally:
        process.returncode = 1
        child.close()
        B._owned_stalled_mask_children.remove(child)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_job_kills_mask_child_when_parent_dies(tmp_path):
    import psutil

    pidfile = tmp_path / "child.pid"
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import os, pathlib, subprocess, sys\n"
        "from tower.process_ownership import assign_to_job\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],\n"
        "                     stdin=subprocess.DEVNULL)\n"
        "job = assign_to_job(p)\n"
        "if job is None: p.kill(); sys.exit(3)\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid))\n"
        "os._exit(0)\n", encoding="utf-8")
    completed = subprocess.run([sys.executable, str(parent)], timeout=20,
                               check=False, capture_output=True)
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 10
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not psutil.pid_exists(pid), "Job close left a mask child alive"


def test_wait_exception_kills_child(tmp_path, monkeypatch):
    process, job = Process(), Job()
    monkeypatch.setattr(B, "assign_to_job", lambda p: job)
    monkeypatch.setattr(B, "terminate_tree", lambda p, **k: setattr(p, "returncode", 1) or True)
    real_child = B.StopSolverMasks
    monkeypatch.setattr(B, "StopSolverMasks", lambda *a, **k:
                        real_child(*a, spawn=lambda *x, **y: process, **k))
    s = solver(tmp_path)
    s.wait = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        B.wait_for_background_solve_at_stop(s, 1, should_stop=lambda: False)
    assert process.poll() == 1 and job.closed


@pytest.fixture
def images(tmp_path, monkeypatch):
    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    world = engine.create_world("test")
    intrinsics = CameraIntrinsics(source="self_calibrated", model="pinhole_radtan",
                                  fx=50, fy=50, cx=32, cy=24,
                                  dist_coeffs=(0, 0, 0, 0, 0),
                                  calibrated_width=64, calibrated_height=48)
    session = engine.start_session(world, frame_source="synthetic", intrinsics=intrinsics)
    ws = global_solve.workspace_for(store, world, session)
    ws.images_dir.mkdir(parents=True)
    ws.camera_path.write_text(json.dumps(global_solve.PinholeCamera(
        50, 50, 32, 24, 64, 48).to_json_dict()), encoding="utf-8")
    for i in range(3):
        name = f"{i:08d}.jpg"
        if i < 2:
            rgb = np.full((48, 64, 3), i * 10 + 100, np.uint8)
            ok, encoded = cv2.imencode(".jpg", rgb)
            assert ok
            (ws.images_dir / name).write_bytes(encoded.tobytes())
        store.append_keyframe(world, Keyframe(
            keyframe_id=f"{session}:{i}", session_id=session, source_seq=i,
            received_at=float(i), image_relpath=f"images/{name}", width=64,
            height=48, byte_count=1, segment_index=0))
    ws.database_path.write_bytes(b"walk database sentinel")
    monkeypatch.setattr(SM.T, "BACKEND_FACTORY", StubDetector())
    monkeypatch.setattr(SM, "cuda_unavailable_reason", lambda: None)
    monkeypatch.setattr(global_solve, "prepare_images", lambda *a, **k: pytest.fail(
        "Stop child must not prepare solver images"))
    return tmp_path, store, world, session, ws


def _promote(images):
    root, _store, world, session, _ws = images
    child = B.StopSolverMasks(solver(root))
    child.solver.world_id, child.solver.session_id = world, session
    child._child = Process(rc=0)
    child._promote()


def test_child_uses_staging_and_never_touches_database(images):
    root, store, world, session, ws = images
    before = hashlib.sha256(ws.database_path.read_bytes()).hexdigest()
    assert M.prefill(root, world, session) == 0
    assert hashlib.sha256(ws.database_path.read_bytes()).hexdigest() == before
    assert not list((ws.root / "masks").glob("*.png"))
    _promote(images)
    ids = {global_solve.keyframe_image_name(k): k.keyframe_id
           for k in store.read_keyframes(world, session)}
    ok, encoded = cv2.imencode(".jpg", np.full((48, 64, 3), 130, np.uint8))
    assert ok
    (ws.images_dir / "00000002.jpg").write_bytes(encoded.tobytes())
    final = SM.ensure_solver_masks(ws, ["00000000.jpg", "00000001.jpg", "00000002.jpg"],
                                   keyframe_ids=ids, shape=(48, 64),
                                   backend_factory=StubDetector(), device_probe=lambda: None)
    assert final.cache_hits == 2 and final.computed == 1
    assert (ws.root / "masks" / "00000002.jpg.png").exists()


def test_replaced_image_is_never_promoted_under_old_hash(images, monkeypatch):
    root, _store, world, session, ws = images
    name = "00000000.jpg"
    original_sha = hashlib.sha1((ws.images_dir / name).read_bytes()).hexdigest()
    decode = SM._read_solver_image
    changed = [False]

    def replace_at_decode(path):
        if not changed[0] and Path(path).name == name:
            changed[0] = True
            ok, encoded = cv2.imencode(".jpg", np.full((48, 64, 3), 222, np.uint8))
            assert ok
            (ws.images_dir / name).write_bytes(encoded.tobytes())
        return decode(path)

    monkeypatch.setattr(SM, "_read_solver_image", replace_at_decode)
    assert M.prefill(root, world, session) == 0
    stage = ws.root / "transients" / "s"
    manifest = json.loads((stage / "result.json").read_text())
    assert manifest["images"][name] == original_sha
    _promote(images)
    assert not (ws.root / "masks" / f"{name}.png").exists()
    final = SM.ensure_solver_masks(ws, [name, "00000001.jpg"], shape=(48, 64),
                                   backend_factory=StubDetector(), device_probe=lambda: None)
    assert final.cache_hits == 1 and final.computed == 1


@pytest.mark.parametrize("failure", ["late failure", "CUDA out of memory"])
def test_partial_detector_failure_is_quarantined(images, monkeypatch, failure):
    root, _store, world, session, ws = images
    good = StubDetector()

    def factory(component):
        backend = good(component)
        if component == SM.T.COMPONENT_ONEFORMER:
            backend.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(failure))
        return backend

    monkeypatch.setattr(SM.T, "BACKEND_FACTORY", factory)
    assert M.prefill(root, world, session) == 1
    stage = ws.root / "transients" / "s"
    assert list((stage / "transients").glob("*.gdsam.npz"))
    assert not list((ws.root / "transients").glob("*.npz"))
    monkeypatch.setattr(SM.T, "BACKEND_FACTORY", StubDetector())
    final = SM.ensure_solver_masks(ws, ["00000000.jpg", "00000001.jpg"],
                                   shape=(48, 64), backend_factory=StubDetector(),
                                   device_probe=lambda: None)
    assert final.cache_hits == 0 and final.computed == 2
    assert final.record()["state"] == "applied"
