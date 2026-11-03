"""Transient masks at Stop (`TOWER_WORLD_MASKS_AT_STOP`, `world_builder/masks_at_stop.py`).

The worker stages masks with the transient stage itself into a staging directory; the final
surface stage promotes a staged mask into the cache only when its inputs are the stage's own.
What must hold: the switch off changes nothing; a worker that fails leaves exactly today's
result; staged masks that match become cache hits with the same masks and the same
`frames_digest`; a staged mask whose inputs differ is refused and computed as today.

No model is loaded: the backends are the transient tests' stub. Runs on the appearance tests'
synthetic world (eight keyframes, a trusted label, the depth stage's records and fill masks).
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest

from tests.test_world_builder_appearance import SESSION, SH, SW, WORLD, World, _never_redact
from tests.test_world_builder_transients import HAND_KI, Stub

from tower.world_builder import appearance as A
from tower.world_builder import masks_at_stop as M
from tower.world_builder import transients as T

ENV = "TOWER_WORLD_MASKS_AT_STOP"


def _with_keyframes(world):
    """The session journal the worker reads its keyframes (and their final indices) from."""
    from tower.world_builder.records import Keyframe

    for i, kid in enumerate(world.kids):
        seq = kid.rsplit(":", 1)[-1]
        data = (world.images / f"{seq}.jpg").read_bytes()
        world.store.append_keyframe(WORLD, Keyframe(
            keyframe_id=kid, session_id=SESSION, source_seq=i + 1, received_at=float(i),
            image_relpath=f"images/{seq}.jpg", width=SW, height=SH, byte_count=len(data)))
    return world


CUDA = {"device": "cuda", "fp16": True, "torch": "2.x", "transformers": "4.x"}


@pytest.fixture(autouse=True)
def _cuda(monkeypatch):
    """The worker runs only on a CUDA device and promote compares execution modes: this suite
    has no GPU, so both halves see the same CUDA mode unless a test says otherwise."""
    monkeypatch.setattr(M, "execution_mode", lambda: dict(CUDA))


@pytest.fixture
def world(tmp_path):
    return _with_keyframes(World(tmp_path / "a"))


def _solution(world):
    from tower.world_builder.global_solve import load_solution

    return load_solution(world.store, WORLD, SESSION)


def _final(world, stub, *, promote: bool):
    """The final surface stage's transient step on this world: promote (when on), then
    `ensure_transient_masks` over every keyframe, as `surface_pipeline._ensure_transients`."""
    solution = _solution(world)
    records = {r["ki"]: r for r in world.records}
    params = T.TransientParams()
    policy = A.resolve_label_policy(world.store, WORLD, SESSION, redactor_factory=_never_redact)
    depth = world.dense / "work" / "depth"
    promoted = None
    if promote:
        promoted = M.promote(world.store, WORLD, SESSION, depth_dir=depth, align_records=records,
                             solution=solution, intrinsics=world.intrinsics, params=params,
                             policy=policy)
    report = T.ensure_transient_masks(
        world.store, WORLD, SESSION, list(enumerate(world.kids)), intrinsics=world.intrinsics,
        camera=solution.camera, align_records=records, depth_dir=depth, params=params,
        policy=policy, backend_factory=stub)
    return report, promoted


def _masks(report):
    return {ki: report.mask(ki) for ki in sorted(report.keys)}


def _same_masks(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def _promote(world, *, records=None, solution=None, params=None, policy=None, mode=None):
    return M.promote(world.store, WORLD, SESSION, depth_dir=world.dense / "work" / "depth",
                     align_records=records if records is not None else {r["ki"]: r for r in world.records},
                     solution=solution or _solution(world), intrinsics=world.intrinsics,
                     params=params or T.TransientParams(),
                     policy=policy or A.resolve_label_policy(world.store, WORLD, SESSION,
                                                             redactor_factory=_never_redact),
                     mode=mode)


class FillStub(Stub):
    """A FILL-DEPENDENT detector: the hand rectangle minus the unobserved (fill) pixels, as
    Grounding DINO + SAM's `filter_instance_masks` makes a mask depend on the fill."""

    def __call__(self, component):
        stub = self

        class Backend(T.ComponentBackend):
            def probe(self):
                return None

            def run(self, items, params, emit, should_stop=None):
                for ki, rgb, unobserved in items:
                    stub.calls.append((component, ki))
                    hand = np.zeros(rgb.shape[:2], bool)
                    if ki in stub.hands:
                        y0, y1, x0, x1 = stub.hands[ki]
                        hand[y0:y1, x0:x1] = True
                    emit(ki, hand & ~unobserved, np.zeros_like(hand), 0.001)
                return {}

        Backend.component = component
        return Backend()


def _builder_module(name):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "world_build_session.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------


def test_the_switch_is_off_by_default(monkeypatch):
    from tower.config import world_masks_at_stop_setting

    monkeypatch.delenv(ENV, raising=False)
    assert world_masks_at_stop_setting() is False
    monkeypatch.setenv(ENV, "on")
    assert world_masks_at_stop_setting() is True


def test_off_the_builder_starts_no_child(monkeypatch, world):
    wbs = _builder_module("wbs_masks_test")
    spawned = []
    monkeypatch.delenv(ENV, raising=False)
    got = wbs.MasksAtStop.start_if_on(world.store, root=world.root, world_id=WORLD, session_id=SESSION,
                                      sources={}, wanted=True, spawn=lambda *a, **k: spawned.append(a))
    assert got is None and spawned == []
    assert not M.staging_dir(world.store, WORLD, SESSION).exists()
    # On but nothing will be built after the final solve: no child either.
    monkeypatch.setenv(ENV, "on")
    got = wbs.MasksAtStop.start_if_on(world.store, root=world.root, world_id=WORLD, session_id=SESSION,
                                      sources={}, wanted=False, spawn=lambda *a, **k: spawned.append(a))
    assert got is None and spawned == []


def test_no_child_when_the_final_solve_uses_the_gpu_itself(monkeypatch, world):
    """Solve masks or the evidence gate put GPU work inside the final solve, on the critical
    path: the worker would compete with it, so it is not started."""
    wbs = _builder_module("wbs_masks_test3")
    monkeypatch.setenv(ENV, "on")
    for switch in ("TOWER_WORLD_SOLVE_MASKS", "TOWER_WORLD_SOLVE_GATE"):
        monkeypatch.setenv(switch, "on")
        spawned = []
        got = wbs.MasksAtStop.start_if_on(world.store, root=world.root, world_id=WORLD,
                                          session_id=SESSION, sources={}, wanted=True,
                                          spawn=lambda *a, **k: spawned.append(a))
        assert got is None and spawned == []
        monkeypatch.delenv(switch)


def test_off_the_surface_stage_never_promotes(monkeypatch, world):
    """Staged masks on disk, the switch off: the stage computes every mask, as today."""
    from tower.world_builder import surface_pipeline as SP

    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(M, "promote_quietly",
                        lambda *a, **k: pytest.fail("promote called with the switch off"))
    frames = types.SimpleNamespace(items=[(ki,) for ki in range(len(world.kids))],
                                   kids=dict(enumerate(world.kids)))
    align = {"records": world.records}
    stub = Stub()
    report = SP._ensure_transients(world.store, WORLD, SESSION, _solution(world), world.intrinsics,
                                   align, world.dense / "work", frames, T.TransientParams(), None,
                                   None, stub)
    assert report.computed == len(world.kids) and report.cached == 0
    assert sorted(stub.computed(T.COMPONENT_ONEFORMER)) == list(range(len(world.kids)))


def test_on_the_surface_stage_promotes_then_finds_hits(monkeypatch, world):
    from tower.world_builder import surface_pipeline as SP

    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    monkeypatch.setenv(ENV, "on")
    frames = types.SimpleNamespace(items=[(ki,) for ki in range(len(world.kids))],
                                   kids=dict(enumerate(world.kids)))
    stub = Stub()
    report = SP._ensure_transients(world.store, WORLD, SESSION, _solution(world), world.intrinsics,
                                   {"records": world.records}, world.dense / "work", frames,
                                   T.TransientParams(), None, None, stub)
    assert stub.calls == [] and report.cached == len(world.kids) and report.computed == 0


# ---------------------------------------------------------------------------
# cache hits, and what a hit must equal
# ---------------------------------------------------------------------------


def test_staged_masks_become_hits_equal_to_a_cold_build(tmp_path, world):
    cold_world = _with_keyframes(World(tmp_path / "cold"))
    cold, _ = _final(cold_world, Stub(), promote=False)

    summary = M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    assert summary["state"] == "done" and summary["staged"] == len(world.kids)
    # The real cache is untouched by the worker.
    assert not list((world.dense / "work" / "depth").glob("*_transient.*.npz"))
    stub = Stub()
    warm, promoted = _final(world, stub, promote=True)

    assert stub.calls == []                                   # nothing computed after Stop
    assert promoted["promoted"] == 2 * len(world.kids) and promoted["refused"] == {}
    assert warm.cached == len(world.kids) and warm.computed == 0
    assert warm.frames_digest() == cold.frames_digest()       # the manifests' digest is unchanged
    assert _same_masks(_masks(warm), _masks(cold))
    assert warm.mask(HAND_KI).any()


def test_the_worker_stages_the_fill_the_final_depth_stage_will_reuse(world):
    """The depth stage reuses the live depth worker's fill (copied over, it wins) else the last
    live depth stage's; it computes a fill only for the rest. The worker must stage THOSE files,
    not a fresh computation, or every reused frame would be refused at promotion."""
    import hashlib
    import json
    import math

    from tower.world_builder.dense_pipeline import FILL_RULE, depth_trust_now
    from tower.world_builder.global_solve import _undistort_maps

    depth, undist = world.dense / "work" / "depth", world.dense / "work" / "undist"
    undist.mkdir(parents=True)
    for i in range(len(world.kids)):
        (undist / f"{i:05d}.jpg").write_bytes(b"jpg")         # the last live depth stage's frames
    prior = np.zeros(np.load(depth / "00001_fill.npy").shape, bool)
    prior[5:20, 5:20] = True                                  # no fresh computation draws this
    world.set_fill(1, prior)
    world.set_fill(3, prior)
    live = prior.copy()
    live[40:60, 40:60] = True
    _m1, _m2, _roi, cam = _undistort_maps(world.intrinsics, SW, SH)
    fov = round(math.degrees(2.0 * math.atan(cam.width / (2.0 * cam.fx))), 6)
    root = world.dense / "live_depth"
    (root / "work" / "depth").mkdir(parents=True)
    (root / "work" / "undist").mkdir(parents=True)
    old_ki = 77
    np.save(root / "work" / "depth" / f"{old_ki:05d}_pred.npy", np.ones((cam.height, cam.width), np.float16))
    np.save(root / "work" / "depth" / f"{old_ki:05d}_fill.npy", live)
    (root / "work" / "undist" / f"{old_ki:05d}.jpg").write_bytes(b"jpg")
    kid = world.kids[3]
    sha = hashlib.sha1((world.images / f"{kid.rsplit(':', 1)[-1]}.jpg").read_bytes()).hexdigest()
    (root / "align.json").write_text(json.dumps({
        "backend": "moge2-vitl", "fill_rule": FILL_RULE, "imagery_source": "redacted",
        "redaction_trust": depth_trust_now(world.store, WORLD, SESSION, imagery_source="redacted"),
        "keyframe_image_set": None, "known_fov": fov,
        "records": [{"ki": old_ki, "kid": kid, "image_sha1": sha, "fill_rule": FILL_RULE,
                     "known_fov": fov}]}))

    summary = M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    assert summary["fills"] == {"reused": len(world.kids), "fresh": 0, "none": 0}
    staging = M.staging_dir(world.store, WORLD, SESSION)
    assert np.array_equal(np.load(staging / "00001_fill.npy"), prior)
    assert np.array_equal(np.load(staging / "00003_fill.npy"), live)
    world.set_fill(3, live)                                   # what `offer_live_predictions` copies in
    stub = Stub()
    report, promoted = _final(world, stub, promote=True)
    assert promoted["promoted"] == 2 * len(world.kids) and stub.calls == []


def test_a_staged_mask_whose_fill_differs_is_computed_as_today(world):
    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    fill = np.zeros_like(np.load(world.dense / "work" / "depth" / f"{HAND_KI:05d}_fill.npy"))
    fill[:10, :10] = True                                     # the final depth stage's fill moved
    world.set_fill(HAND_KI, fill)
    stub = Stub()
    report, promoted = _final(world, stub, promote=True)
    assert promoted["refused"] == {"fill": 2}
    assert sorted(set(stub.computed())) == [HAND_KI]
    assert report.computed == 1 and report.cached == len(world.kids) - 1


def test_a_staged_mask_of_other_image_bytes_is_refused(world):
    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    world.set_image(2, world.render(2) // 2)                  # the stored keyframe changed since
    world.write_align()
    stub = Stub()
    report, promoted = _final(world, stub, promote=True)
    assert promoted["refused"] == {"key": 2}
    assert sorted(set(stub.computed())) == [2]


def test_another_calibration_refuses_every_staged_mask(world):
    import dataclasses

    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    solution = _solution(world)
    other = dataclasses.replace(world.intrinsics, fx=world.intrinsics.fx + 1.0)
    out = M.promote(world.store, WORLD, SESSION, depth_dir=world.dense / "work" / "depth",
                    align_records={r["ki"]: r for r in world.records}, solution=solution,
                    intrinsics=other, params=T.TransientParams(),
                    policy=A.resolve_label_policy(world.store, WORLD, SESSION, redactor_factory=_never_redact))
    assert out["promoted"] == 0 and out["refused"] == {"calibration": 2 * len(world.kids)}


def test_a_hit_already_in_the_cache_is_never_replaced(world):
    _final(world, Stub(), promote=False)                      # the cache already holds every mask
    before = {p.name: p.read_bytes() for p in (world.dense / "work" / "depth").glob("*_transient.*.npz")}
    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    _report, promoted = _final(world, Stub(), promote=True)
    assert promoted["promoted"] == 0 and promoted["already"] == 2 * len(world.kids)
    after = {p.name: p.read_bytes() for p in (world.dense / "work" / "depth").glob("*_transient.*.npz")}
    assert after == before


# ---------------------------------------------------------------------------
# one run's files, never two (review L1 R1 F1, R2 1)
# ---------------------------------------------------------------------------


def test_a_second_run_never_promotes_a_mask_made_from_the_first_runs_fill(tmp_path, world):
    """R1's reproduction: run 1 computes its fills fresh; the final depth stage's fill for
    HAND_KI then differs and run 2 reuses it. Run 1's mask must not survive into run 2."""
    M.prefill(world.store, WORLD, SESSION, backend_factory=FillStub())
    fill = np.zeros_like(np.load(world.dense / "work" / "depth" / f"{HAND_KI:05d}_fill.npy"))
    y0, y1, x0, x1 = Stub().hands[HAND_KI]
    fill[y0:y1, x0:(x0 + x1) // 2] = True
    world.set_fill(HAND_KI, fill)
    undist = world.dense / "work" / "undist"
    undist.mkdir(parents=True, exist_ok=True)
    for i in range(len(world.kids)):
        (undist / f"{i:05d}.jpg").write_bytes(b"jpg")        # run 2 now reuses the prior fills
    M.prefill(world.store, WORLD, SESSION, backend_factory=FillStub())
    warm, promoted = _final(world, FillStub(), promote=True)

    cold_world = _with_keyframes(World(tmp_path / "cold"))
    cold_world.set_fill(HAND_KI, fill)
    cold, _ = _final(cold_world, FillStub(), promote=False)
    assert promoted["promoted"] == 2 * len(world.kids)
    assert np.array_equal(warm.mask(HAND_KI), cold.mask(HAND_KI))
    assert _same_masks(_masks(warm), _masks(cold))


def test_every_run_starts_from_an_empty_staging_directory(world):
    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    staging = M.staging_dir(world.store, WORLD, SESSION)
    M.write_sources(staging, {"k": "p"})
    (staging / "00003_transient.gdsam.npz.p99.tmp").write_bytes(b"debris")
    summary = M.prefill(world.store, WORLD, SESSION, backend_factory=Stub(), should_stop=lambda: True)
    assert summary["state"] == "stopped"
    assert sorted(p.name for p in staging.iterdir()) == [M.INDEX_FILENAME, M.SOURCES_FILENAME]
    stub = Stub()
    report, promoted = _final(world, stub, promote=True)
    assert promoted["staged"] == 0 and report.computed == len(world.kids)


def test_a_staging_directory_that_cannot_be_emptied_leaves_no_index(monkeypatch, world):
    import pathlib

    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    staging = M.staging_dir(world.store, WORLD, SESSION)
    real_unlink = pathlib.Path.unlink

    def unlink(self, *a, **k):
        if self.name.endswith("_transient.gdsam.npz"):
            raise PermissionError("held by another process")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    summary = M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    monkeypatch.setattr(pathlib.Path, "unlink", real_unlink)
    assert summary["state"] == "declined"
    assert not (staging / M.INDEX_FILENAME).exists()
    assert list(staging.glob("*_transient.gdsam.npz"))       # the old masks are still there ...
    assert _promote(world) is None                            # ... and nothing takes them


# ---------------------------------------------------------------------------
# the execution mode (review L1 R1 F2, R2 2)
# ---------------------------------------------------------------------------


def test_without_cuda_the_worker_declines(monkeypatch, world):
    monkeypatch.setattr(M, "execution_mode", lambda: dict(CUDA, device="cpu", fp16=False))
    stub = Stub()
    summary = M.prefill(world.store, WORLD, SESSION, backend_factory=stub)
    assert summary["state"] == "declined" and stub.calls == []
    assert not list(M.staging_dir(world.store, WORLD, SESSION).glob("*_transient.*.npz"))


@pytest.mark.parametrize("change", [{"device": "cpu", "fp16": False}, {"fp16": False},
                                    {"torch": "2.y"}, {"transformers": "5.0"}])
def test_a_stage_in_another_execution_mode_takes_nothing(world, change):
    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    out = _promote(world, mode=dict(CUDA, **change))
    assert out["promoted"] == 0 and out["refused"] == {"execution-mode": 2 * len(world.kids)}
    assert not list((world.dense / "work" / "depth").glob("*_transient.*.npz"))


def test_the_recorded_execution_mode_is_the_backends_choice(monkeypatch):
    """`execution_mode` mirrors the backends: CUDA when available (fp16 forward calls), else
    CPU (no autocast)."""
    import torch

    monkeypatch.undo()                                         # the real function
    for available, want in ((True, ("cuda", True)), (False, ("cpu", False))):
        monkeypatch.setattr(torch.cuda, "is_available", lambda a=available: a)
        got = M.execution_mode()
        assert (got["device"], got["fp16"]) == want and got["torch"] == torch.__version__


# ---------------------------------------------------------------------------
# each promotion gate, alone (review L1 R1: these four survived mutation)
# ---------------------------------------------------------------------------


def test_no_final_depth_record_for_the_same_image_no_promotion(world):
    """The final depth stage did not record keyframe 4 at all, and recorded keyframe 6 for
    other image bytes: both are refused, although their fill files are equal to the staged
    ones (the stage would refuse those frames, or read other bytes)."""
    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    records = {r["ki"]: dict(r) for r in world.records}
    del records[4]
    records[6]["image_sha1"] = "0" * 40
    out = _promote(world, records=records)
    assert out["refused"] == {"image": 4} and out["promoted"] == 2 * (len(world.kids) - 2)
    depth = world.dense / "work" / "depth"
    assert not list(depth.glob("00004_transient.*.npz")) and not list(depth.glob("00006_transient.*.npz"))


def test_another_camera_size_refuses_every_staged_mask(world):
    import dataclasses

    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    solution = _solution(world)
    wider = dataclasses.replace(solution, camera=dict(solution.camera, width=solution.camera["width"] + 1))
    out = _promote(world, solution=wider)
    assert out["promoted"] == 0 and out["refused"] == {"camera": 2 * len(world.kids)}


def test_a_research_build_takes_no_staged_mask(world):
    from tower.world_builder.raw_imagery import IMAGERY_RAW

    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    trusted = A.resolve_label_policy(world.store, WORLD, SESSION, redactor_factory=_never_redact)
    raw = A.LabelPolicy(session_redaction=trusted.session_redaction, trusted=False,
                        image_set=trusted.image_set, imagery_source=IMAGERY_RAW)
    out = _promote(world, policy=raw)
    assert out["promoted"] == 0 and out["refused"] == {"imagery": 2 * len(world.kids)}


def test_a_component_this_build_does_not_run_is_not_promoted(world):
    M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    out = _promote(world, params=T.TransientParams(mode=T.MODE_ONEFORMER))
    assert out["refused"] == {"not-this-build": len(world.kids)} and out["promoted"] == len(world.kids)
    assert not list((world.dense / "work" / "depth").glob("*_transient.gdsam.npz"))


# ---------------------------------------------------------------------------
# the worker failing, stopping, declining
# ---------------------------------------------------------------------------


def test_a_failing_worker_leaves_todays_result(tmp_path, world):
    cold_world = _with_keyframes(World(tmp_path / "cold"))
    cold, _ = _final(cold_world, Stub(), promote=False)

    summary = M.prefill(world.store, WORLD, SESSION, backend_factory=Stub(run_error=RuntimeError("boom")))
    assert summary["state"] == T.STATE_FAILED
    stub = Stub()
    report, promoted = _final(world, stub, promote=True)
    assert promoted["promoted"] == 0
    assert report.computed == len(world.kids)
    assert report.frames_digest() == cold.frames_digest()
    assert _same_masks(_masks(report), _masks(cold))


def test_a_worker_stopped_part_way_leaves_the_rest_to_the_stage(world):
    worker = Stub()

    def should_stop():
        return len(worker.calls) >= 2 * 3                     # once the first chunk of 3 is done

    summary = M.prefill(world.store, WORLD, SESSION, backend_factory=worker, should_stop=should_stop,
                        chunk=3)
    assert summary["state"] == "stopped"
    stub = Stub()
    report, promoted = _final(world, stub, promote=True)
    staged = promoted["promoted"] // 2
    assert staged == 3
    assert report.cached == staged and report.computed == len(world.kids) - staged
    assert len(set(stub.computed())) == len(world.kids) - staged


def test_an_untrusted_label_is_declined(world):
    world.set_label("none")
    summary = M.prefill(world.store, WORLD, SESSION, backend_factory=Stub())
    assert summary["state"] == "declined"
    assert not list(M.staging_dir(world.store, WORLD, SESSION).glob("*_transient.*.npz"))


def test_nothing_staged_promotes_nothing(world):
    stub = Stub()
    report, promoted = _final(world, stub, promote=True)
    assert promoted is None and report.computed == len(world.kids)


def test_the_builder_survives_a_child_that_cannot_start(monkeypatch, world):
    wbs = _builder_module("wbs_masks_test2")
    monkeypatch.setenv(ENV, "on")

    def broken(*a, **k):
        raise OSError("no interpreter")

    assert wbs.MasksAtStop.start_if_on(world.store, root=world.root, world_id=WORLD, session_id=SESSION,
                                       sources={"k": "p"}, wanted=True, spawn=broken) is None

    class Child:
        pid, returncode = 4242, None

        def poll(self):
            return self.returncode

    child = Child()
    ended = []
    monkeypatch.setattr(wbs, "_terminate_process_tree", lambda p: ended.append(p))
    worker = wbs.MasksAtStop.start_if_on(world.store, root=world.root, world_id=WORLD, session_id=SESSION,
                                         sources={"k": "p"}, wanted=True, spawn=lambda *a, **k: child)
    assert worker is not None
    staging = M.staging_dir(world.store, WORLD, SESSION)
    assert (staging / M.SOURCES_FILENAME).is_file()
    worker.close()
    worker.close()                                            # idempotent
    assert ended == [child]


# ---------------------------------------------------------------------------
# order
# ---------------------------------------------------------------------------


def test_the_order_puts_the_likely_frames_first():
    ids = [f"s:{i:08d}" for i in range(8)]
    posed = {"rotation": [1], "translation": [0]}
    solution = types.SimpleNamespace(
        keyframe_ids=ids[:6],
        poses={ids[0]: dict(posed, component=1), ids[1]: dict(posed, component=0),
               ids[3]: dict(posed, component=0), ids[4]: {"component": 0, "rotation": None},
               ids[5]: dict(posed, component=2)})
    # main component, then newer than the solve, then other components, then unposed
    assert M.keyframe_order(ids, solution) == [1, 3, 6, 7, 0, 5, 2, 4]
    assert M.keyframe_order(ids, None) == list(range(8))


def test_the_worker_stops_when_its_builder_is_gone():
    """Outside the supervisor's Job Object (a builder run by hand or by a replay harness) a
    killed builder must not leave the worker on the GPU: it watches its parent."""
    import importlib.util
    import os
    import pathlib

    import psutil

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "world_masks_at_stop.py"
    spec = importlib.util.spec_from_file_location("wmas_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.parent_gone(None)() is False
    assert cli.parent_gone(os.getpid())() is False
    unused = max(psutil.pids()) + 100_000
    assert cli.parent_gone(unused)() is True
