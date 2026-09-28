"""The optional Stop mask child and its ordering before the final solve."""

import hashlib
import json
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


@pytest.mark.parametrize("masks,at_stop,running,expected", [
    (False, False, True, False),
    (True, False, True, False),
    (True, True, False, False),
    (True, True, True, True),
])
def test_wait_launches_only_for_running_masked_solve(monkeypatch, masks, at_stop,
                                                      running, expected):
    calls = []
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS", "on" if masks else "off")
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", "on" if at_stop else "off")

    class Solver:
        def __init__(self):
            self.running = running

        def wait(self, timeout, *, should_stop):
            calls.append(("wait", timeout, should_stop()))

    class Child:
        def __init__(self, solver):
            calls.append(("child", solver.running))

        def launch(self):
            calls.append(("launch",))

        def join(self, *, should_stop):
            calls.append(("join", should_stop()))

    monkeypatch.setattr(B, "StopSolverMasks", Child)
    solver = Solver()
    B.wait_for_background_solve_at_stop(solver, 120.0, should_stop=lambda: False)
    calls.append(("final",))
    assert calls == ([('child', True), ('launch',), ('wait', 120.0, False),
                      ('join', False), ('final',)] if expected else
                     [('wait', 120.0, False), ('final',)])


def test_hard_stop_terminates_child_before_final_solve(monkeypatch):
    events = []
    stopped = [False]
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS", "on")
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", "on")

    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return self.returncode

    process = Process()

    class Solver:
        running = True
        root = Path("unused")
        world_id = "w"
        session_id = "s"

        def wait(self, timeout, *, should_stop):
            events.append("wait")
            stopped[0] = True

        def run_final(self):
            events.append("final")

    def terminate(child, **kwargs):
        assert child is process and kwargs["hard"]
        events.append("terminated")
        process.returncode = 1
        return True

    monkeypatch.setattr(B, "terminate_tree", terminate)
    monkeypatch.setattr(B, "assign_to_job", lambda child: None)
    monkeypatch.setattr(B, "python_executable", lambda: "python")
    monkeypatch.setattr(B, "child_environment", lambda: {})
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: process)
    solver = Solver()
    B.wait_for_background_solve_at_stop(solver, 120, should_stop=lambda: stopped[0])
    if not stopped[0]:
        solver.run_final()
    assert events == ["wait", "terminated"]


def test_failed_child_does_not_block_final_solve(monkeypatch):
    events = []
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS", "on")
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", "on")

    class Child:
        def __init__(self, solver):
            pass

        def launch(self):
            events.append("launch")
            return False

        def join(self, *, should_stop):
            events.append("join")

    class Solver:
        running = True

        def wait(self, timeout, *, should_stop):
            events.append("wait")

        def run_final(self):
            events.append("final")

    monkeypatch.setattr(B, "StopSolverMasks", Child)
    solver = Solver()
    B.wait_for_background_solve_at_stop(solver, 120, should_stop=lambda: False)
    solver.run_final()
    assert events == ["launch", "wait", "join", "final"]


def test_child_exit_failure_is_joined_and_final_solve_still_runs(monkeypatch):
    events = []
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS", "on")
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", "on")

    class Process:
        pid = 123
        returncode = 7

        def poll(self):
            events.append("poll")
            return self.returncode

    monkeypatch.setattr(B, "assign_to_job", lambda process: None)
    monkeypatch.setattr(B, "python_executable", lambda: "python")
    monkeypatch.setattr(B, "child_environment", lambda: {})
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: Process())

    class Solver:
        running = True
        root = Path("unused")
        world_id = "w"
        session_id = "s"

        def wait(self, timeout, *, should_stop):
            events.append("wait")

        def run_final(self):
            events.append("final")

    solver = Solver()
    B.wait_for_background_solve_at_stop(solver, 120, should_stop=lambda: False)
    solver.run_final()
    assert events.index("wait") < events.index("poll") < events.index("final")


def test_running_child_finishes_before_final_solve(monkeypatch):
    events = []
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS", "on")
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", "on")

    class Process:
        pid = 123
        returncode = None

        def poll(self):
            events.append("poll")
            return self.returncode

    process = Process()

    def finish_child(delay):
        events.append("mask exit")
        process.returncode = 0

    monkeypatch.setattr(B, "assign_to_job", lambda child: None)
    monkeypatch.setattr(B, "python_executable", lambda: "python")
    monkeypatch.setattr(B, "child_environment", lambda: {})
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(B.time, "sleep", finish_child)

    class Solver:
        running = True
        root = Path("unused")
        world_id = "w"
        session_id = "s"

        def wait(self, timeout, *, should_stop):
            events.append("wait")

        def run_final(self):
            events.append("final")

    solver = Solver()
    B.wait_for_background_solve_at_stop(solver, 120, should_stop=lambda: False)
    solver.run_final()
    assert events.index("wait") < events.index("mask exit") < events.index("final")


def test_child_masks_only_prepared_images_and_does_not_touch_database(tmp_path, monkeypatch):
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
    for i in range(2):
        name = f"{i:08d}.jpg"
        rgb = np.full((48, 64, 3), i * 10 + 100, np.uint8)
        ok, encoded = cv2.imencode(".jpg", rgb)
        assert ok
        ws.images_dir.joinpath(name).write_bytes(encoded.tobytes())
        store.append_keyframe(world, Keyframe(
            keyframe_id=f"{session}:{i}", session_id=session, source_seq=i,
            received_at=float(i), image_relpath=f"images/{name}", width=64,
            height=48, byte_count=1, segment_index=0))
    # A third keyframe has no prepared solver image yet.
    store.append_keyframe(world, Keyframe(
        keyframe_id=f"{session}:2", session_id=session, source_seq=2,
        received_at=2.0, image_relpath="images/00000002.jpg", width=64,
        height=48, byte_count=1, segment_index=0))
    ws.database_path.write_bytes(b"walk database sentinel")
    before = hashlib.sha256(ws.database_path.read_bytes()).hexdigest()
    monkeypatch.setattr(SM.T, "BACKEND_FACTORY", StubDetector())
    monkeypatch.setattr(SM, "cuda_unavailable_reason", lambda: None)
    monkeypatch.setattr(global_solve, "prepare_images", lambda *a, **k: pytest.fail(
        "Stop mask child must not prepare or rewrite solver images"))
    assert M.prefill(tmp_path, world, session) == 0
    assert hashlib.sha256(ws.database_path.read_bytes()).hexdigest() == before
    written = {p.relative_to(ws.root).as_posix() for p in ws.root.rglob("*") if p.is_file()}
    assert {"database.db", "camera.json", "images/00000000.jpg",
            "images/00000001.jpg"}.issubset(written)
    additions = written - {"database.db", "camera.json", "images/00000000.jpg",
                           "images/00000001.jpg"}
    assert additions and all(p.startswith("transients/") or p.startswith("masks/")
                             for p in additions)
    assert "masks/00000002.jpg.png" not in additions
    png_before = {p.name: p.read_bytes() for p in (ws.root / "masks").glob("*.png")}
    ids = {global_solve.keyframe_image_name(k): k.keyframe_id
           for k in store.read_keyframes(world, session)}
    final = SM.ensure_solver_masks(ws, ["00000000.jpg", "00000001.jpg"], keyframe_ids=ids,
                                   shape=(48, 64), backend_factory=StubDetector(),
                                   device_probe=lambda: None)
    assert final.cache_hits == 2 and final.computed == 0
    assert {p.name: p.read_bytes() for p in (ws.root / "masks").glob("*.png")} == png_before


def test_stop_mask_switch_defaults_off(monkeypatch):
    from tower.config import world_solve_masks_at_stop_setting
    monkeypatch.delenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", raising=False)
    assert world_solve_masks_at_stop_setting() is False
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_AT_STOP", "on")
    assert world_solve_masks_at_stop_setting() is True
