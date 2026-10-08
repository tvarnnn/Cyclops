"""Exp1 worker, cache identity, OFF isolation and GPU back-off."""

import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tower.world_builder import dense as D
from tower.world_builder import global_solve as GS
from tower.world_builder import live_depth as LD
from tower.world_builder import surface_pipeline as SP
from tower.world_builder.dense_pipeline import FILL_RULE, reusable_predictions
from tower.world_builder.global_solve import load_solution
from tower.world_builder.records import CameraIntrinsics, Keyframe
from tests.test_world_builder_surface_pipeline import _synthetic_world, WORLD, SESSION


LABEL = "faces-detected-and-filled/yunet-2023mar@0.30+plausibility1"


class FakeMoGe:
    name = LD.MODEL
    kind = "depth"
    licence = "MIT"
    accepts_fov = True
    calls = 0

    def predict(self, rgb, fov_x=None):
        type(self).calls += 1
        return np.full(rgb.shape[:2], 2.0, np.float32)


def _identity_maps(intrinsics, width, height):
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32),
                         np.arange(height, dtype=np.float32))
    camera = SimpleNamespace(fx=intrinsics.fx, width=width, height=height)
    return xs, ys, (0, 0, width, height), camera


@pytest.fixture
def walk(tmp_path, monkeypatch):
    store = _synthetic_world(tmp_path, n_frames=1, with_dense=False)
    session = store.read_session(WORLD, SESSION)
    camera = load_solution(store, WORLD, SESSION).camera
    image = np.random.default_rng(7).integers(
        0, 256, (camera["height"], camera["width"], 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    pixels = encoded.tobytes()
    store.write_keyframe_image(WORLD, SESSION, "00000000.jpg", pixels)
    keyframe = Keyframe(keyframe_id=f"{SESSION}:00000000", session_id=SESSION,
                        source_seq=0, received_at=1.0,
                        image_relpath="images/00000000.jpg",
                        width=camera["width"], height=camera["height"],
                        byte_count=len(pixels), segment_index=0)
    store.append_keyframe(WORLD, keyframe)
    monkeypatch.setattr(GS, "_undistort_maps", _identity_maps)
    monkeypatch.setitem(D._BACKENDS, LD.MODEL, FakeMoGe)
    FakeMoGe.calls = 0
    return store, session, keyframe, pixels


def _worker(walk):
    store, session, _, _ = walk
    return LD.LiveDepthWorker(store, WORLD, SESSION, session.intrinsics,
                              backend_factory=FakeMoGe, vram_probe=lambda: None)


def _predict_one(walk):
    worker = _worker(walk)
    _, _, keyframe, pixels = walk
    worker.submit(keyframe, pixels, pixels, LABEL)
    worker.jobs.join()
    worker.close()
    return worker


def test_worker_writes_reader_shape_and_numeric_fov(walk):
    worker = _predict_one(walk)
    document = json.loads((worker.root / "align.json").read_text())
    record = document["records"][0]
    assert record["kid"] == walk[2].keyframe_id
    assert record["image_sha1"] == hashlib.sha1(walk[3]).hexdigest()
    assert record["fill_rule"] == FILL_RULE
    assert record["known_fov"] == document["known_fov"]
    assert (worker.root / record["pred"]).is_file()
    assert reusable_predictions(worker.root / "align.json", LD.MODEL,
                                known_fov=True) == {0: (record["kid"], record["image_sha1"])}
    assert document["live"]["predicted"] == 1


@pytest.mark.parametrize("setting", ["shadow", "on"])
def test_shadow_ignores_live_prediction_on_reuses_it(walk, monkeypatch, setting):
    store, session, _, _ = walk
    _predict_one(walk)
    monkeypatch.setenv("TOWER_WORLD_LIVE_DEPTH", setting)
    align, _work = SP.ensure_depth_stage(
        store, WORLD, SESSION, load_solution(store, WORLD, SESSION),
        session.intrinsics, gate_rel=0.08, backend=LD.MODEL)
    assert FakeMoGe.calls == (2 if setting == "shadow" else 1)
    assert align["records"][0]["kid"] == f"{SESSION}:00000000"
    assert align["image_origins"].get("reused-prediction", 0) == (setting == "on")
    if setting == "shadow":
        assert (store.world_dir(WORLD) / "dense" / SESSION /
                "live_depth" / "align.json").is_file()
        assert align.get("known_fov") is None


def test_numeric_fov_is_part_of_live_identity(walk, monkeypatch):
    store, session, _, _ = walk
    worker = _predict_one(walk)
    document = json.loads((worker.root / "align.json").read_text())
    document["known_fov"] += 0.01
    (worker.root / "align.json").write_text(json.dumps(document))
    monkeypatch.setenv("TOWER_WORLD_LIVE_DEPTH", "on")
    SP.ensure_depth_stage(store, WORLD, SESSION, load_solution(store, WORLD, SESSION),
                          session.intrinsics, gate_rel=0.08, backend=LD.MODEL)
    assert FakeMoGe.calls == 2


def test_changed_image_is_not_offered(walk, monkeypatch):
    store, session, _, pixels = walk
    _predict_one(walk)
    image = cv2.imdecode(np.frombuffer(pixels, np.uint8), cv2.IMREAD_COLOR)
    image[0, 0] = 255 - image[0, 0]
    ok, changed = cv2.imencode(".jpg", image)
    assert ok
    (store.images_dir(WORLD, SESSION) / "00000000.jpg").write_bytes(changed.tobytes())
    monkeypatch.setenv("TOWER_WORLD_LIVE_DEPTH", "on")
    SP.ensure_depth_stage(store, WORLD, SESSION, load_solution(store, WORLD, SESSION),
                          session.intrinsics, gate_rel=0.08, backend=LD.MODEL)
    assert FakeMoGe.calls == 2


def test_final_solve_index_is_mapped_from_kid(walk):
    store, session, keyframe, _ = walk
    worker = _predict_one(walk)
    solution = load_solution(store, WORLD, SESSION)
    selected = SimpleNamespace(camera=solution.camera,
                               keyframe_ids=("another:00000000", keyframe.keyframe_id))
    root = worker.root.parent
    offers = LD.offer_live_predictions(
        store, WORLD, SESSION, selected, session.intrinsics, root,
        imagery_source="redacted", trust=f"trusted:{LABEL}")
    assert offers == {1: (keyframe.keyframe_id, hashlib.sha1(walk[3]).hexdigest())}
    assert (root / "work" / "depth" / "00001_pred.npy").is_file()


def test_off_never_imports_worker_or_installs_hook(walk, monkeypatch):
    import builtins
    from tower.world_builder import engine as E
    from tower.world_builder.redaction import RedactionResult

    class Redactor:
        available = True
        unavailable_reason = None

        def redact(self, image):
            return RedactionResult(image, LABEL, 0)

    class FakeWorker:
        starts = 0
        submissions = 0

        def __init__(self, *args, **kwargs):
            type(self).starts += 1

        def submit(self, *args):
            type(self).submissions += 1

        def close(self):
            pass

    monkeypatch.setattr(LD, "LiveDepthWorker", FakeWorker)
    store, session, _, pixels = walk
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tower.world_builder.live_depth":
            raise AssertionError("OFF imported the depth worker")
        return original_import(name, *args, **kwargs)

    for value in (None, "off", "misspelled"):
        if value is None:
            monkeypatch.delenv("TOWER_WORLD_LIVE_DEPTH", raising=False)
        else:
            monkeypatch.setenv("TOWER_WORLD_LIVE_DEPTH", value)
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        engine = E.WorldBuilderEngine(store, redactor_factory=Redactor)
        world = engine.create_world()
        engine.start_session(world, intrinsics=session.intrinsics)
        engine.observe(pixels, source_seq=0)
        engine.stop_session()
        assert engine._live_depth is None
        assert LD.mode() == "off"
    assert FakeWorker.starts == FakeWorker.submissions == 0


def test_active_hook_submits_confirmed_keyframe(walk, monkeypatch):
    from tower.world_builder import engine as E
    from tower.world_builder.redaction import RedactionResult

    calls = []

    class Redactor:
        available = True
        unavailable_reason = None

        def redact(self, image):
            return RedactionResult(image, LABEL, 0)

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, keyframe, image, raw, label):
            calls.append((keyframe.keyframe_id, image, raw, label))

        def close(self):
            pass

    monkeypatch.setenv("TOWER_WORLD_LIVE_DEPTH", "shadow")
    monkeypatch.setattr(LD, "LiveDepthWorker", FakeWorker)
    store, session, _, pixels = walk
    engine = E.WorldBuilderEngine(store, redactor_factory=Redactor)
    world = engine.create_world()
    engine.start_session(world, intrinsics=session.intrinsics)
    accepted = engine.observe(pixels, source_seq=0)
    engine.stop_session()
    assert accepted.keyframe_id == calls[0][0]
    assert calls == [(accepted.keyframe_id, pixels, pixels, LABEL)]


def test_backoff_drains_and_resumes_after_three_good_samples(walk, monkeypatch):
    # A non-running worker makes the queue and the injected clock deterministic.
    monkeypatch.setattr(LD.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(LD, "LATENCY_WINDOW", 3)
    durations = iter([0.001, 0.020] + [0.001] * 6)
    now = [0.0]

    def clock():
        if not hasattr(clock, "opening") or not clock.opening:
            clock.opening = True
            return now[0]
        clock.opening = False
        now[0] += next(durations)
        return now[0]

    probes = iter([{"free_mib": 8000, "total_mib": 12000,
                    "tower_reserved_mib": 1000}] * 20)
    store, session, keyframe, pixels = walk
    worker = LD.LiveDepthWorker(store, WORLD, SESSION, session.intrinsics,
                                monotonic=clock, vram_probe=lambda: next(probes),
                                backend_factory=FakeMoGe)
    worker.submit(keyframe, pixels, pixels, LABEL)
    assert worker.counts["submitted"] == 1
    worker.submit(keyframe, pixels, pixels, LABEL)
    assert worker.backed_off and worker.counts["back_off_engagements"] == 1
    assert worker.counts["dropped"] >= 1
    # Two samples first evict the breaching latency from the p95 window;
    # three subsequent in-budget windows earn resumption.
    for _ in range(6):
        worker.submit(keyframe, pixels, pixels, LABEL)
    assert not worker.backed_off
    assert worker.counts["submitted"] == 3


def test_vram_floor_triggers_backoff_even_with_low_latency(walk, monkeypatch):
    monkeypatch.setattr(LD.threading.Thread, "start", lambda self: None)
    values = iter([{"free_mib": 1000, "total_mib": 12000,
                    "tower_reserved_mib": 9000}] +
                  [{"free_mib": 8000, "total_mib": 12000,
                    "tower_reserved_mib": 1000}] * 4)
    tick = iter(np.arange(0, 1, .001))
    store, session, keyframe, pixels = walk
    worker = LD.LiveDepthWorker(store, WORLD, SESSION, session.intrinsics,
                                monotonic=lambda: next(tick),
                                vram_probe=lambda: next(values),
                                backend_factory=FakeMoGe)
    # The worker samples VRAM off the frame path; inject its first low sample.
    worker._sample(force=True)
    worker.submit(keyframe, pixels, pixels, LABEL)
    assert worker.backed_off and worker.counts["back_off_engagements"] == 1
    for _ in range(LD.RESUME_IN_BUDGET_SAMPLES):
        worker._sample(force=True)
        worker.submit(keyframe, pixels, pixels, LABEL)
    assert not worker.backed_off


def test_backoff_counters_flush_without_another_prediction(walk):
    store, session, keyframe, pixels = walk
    worker = _worker(walk)
    worker.submit(keyframe, pixels, pixels, LABEL)
    worker.jobs.join()
    worker.vram_probe = lambda: {"free_mib": 1000, "total_mib": 12000,
                                 "tower_reserved_mib": 9000}
    worker._sample(force=True)
    worker.submit(keyframe, pixels, pixels, LABEL)
    worker.close()
    worker.thread.join(timeout=2)
    assert not worker.thread.is_alive()
    document = json.loads((worker.root / "align.json").read_text())
    assert document["live"]["back_off_engagements"] == 1
    assert document["live"]["p95_confirm_ms"] >= 0
    assert document["live"]["peak_tower_reserved_mib"] == 9000


def test_untrusted_keyframe_never_loads_model(walk):
    store, session, keyframe, pixels = walk
    worker = LD.LiveDepthWorker(
        store, WORLD, SESSION, session.intrinsics,
        backend_factory=lambda: pytest.fail("untrusted pixels reached MoGe"),
        vram_probe=lambda: None)
    worker.submit(keyframe, pixels, pixels, "none")
    worker.jobs.join()
    worker.close()
    worker.thread.join(timeout=2)
    document = json.loads((worker.root / "align.json").read_text())
    assert document["records"] == []
    assert document["live"]["skipped"] == 1


def test_unknown_calibration_never_loads_model(walk):
    store, _, keyframe, pixels = walk
    worker = LD.LiveDepthWorker(
        store, WORLD, SESSION, CameraIntrinsics.unknown(),
        backend_factory=lambda: pytest.fail("uncalibrated frame reached MoGe"),
        vram_probe=lambda: None)
    worker.submit(keyframe, pixels, pixels, LABEL)
    worker.jobs.join()
    worker.close()
    worker.thread.join(timeout=2)
    assert json.loads((worker.root / "align.json").read_text())["live"]["skipped"] == 1


def test_model_load_failure_disables_the_worker(walk):
    store, session, keyframe, pixels = walk
    loads = []

    def unavailable():
        loads.append(1)
        raise RuntimeError("model unavailable")

    worker = LD.LiveDepthWorker(store, WORLD, SESSION, session.intrinsics,
                                backend_factory=unavailable, vram_probe=lambda: None)
    worker.submit(keyframe, pixels, pixels, LABEL)
    worker.jobs.join()
    worker.submit(keyframe, pixels, pixels, LABEL)
    worker.close()
    worker.thread.join(timeout=2)
    assert loads == [1]
    counters = json.loads((worker.root / "align.json").read_text())["live"]
    assert counters["failed"] == counters["skipped"] == 1
