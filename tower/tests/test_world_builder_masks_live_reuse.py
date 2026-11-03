"""The live surface's OneFormer masks, reused at Stop (`TOWER_WORLD_MASKS_LIVE_REUSE`,
`world_builder/masks_at_stop.py` "LIVE MASKS").

During a walk the session records redaction `none`, so the live surface child reads every keyframe
through the re-redaction path and keys its OneFormer masks under `none&<redactor>`; the final build,
keyed under the label `stop_session` writes, never finds them. With the switch on, a live build
journals how each mask was computed and the worker at Stop stages, under the final key, every live
mask whose inputs are provably the final's. What must hold: off changes nothing; a reused mask
becomes a cache hit EQUAL to a cold final build's (the detector here depends on the pixels, so a
mask made from other pixels would differ); and each condition of reuse, broken alone, makes the
worker compute that mask itself.

No model is loaded; the synthetic world is the appearance tests' (eight keyframes).
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from tests.test_world_builder_appearance import SESSION, TRUSTED, WORLD, World
from tests.test_world_builder_masks_at_stop import (  # noqa: F401 -- `_cuda` is an autouse fixture
    CUDA,
    _cuda,
    _final,
    _masks,
    _same_masks,
    _solution,
    _with_keyframes,
)
from tests.test_world_builder_transients import HAND_KI, Stub

from tower.world_builder import masks_at_stop as M
from tower.world_builder import transients as T

AT_STOP = "TOWER_WORLD_MASKS_AT_STOP"
ENV = "TOWER_WORLD_MASKS_LIVE_REUSE"
ONE = T.COMPONENT_ONEFORMER
GDSAM = T.COMPONENT_GDSAM


class PixelStub(Stub):
    """A detector whose masks are a function of the pixels it is shown, so a mask made from
    other bytes is a different mask."""

    def __call__(self, component):
        stub = self

        class Backend(T.ComponentBackend):
            def probe(self):
                return None

            def run(self, items, params, emit, should_stop=None):
                for ki, rgb, _unobserved in items:
                    stub.calls.append((component, ki))
                    v = rgb.astype(np.int64).sum(-1)
                    hand = v > np.percentile(v, 85 if component == ONE else 70)
                    emit(ki, hand, (v % 7) == 0, 0.001)
                return {}

        Backend.component = component
        return Backend()


def _keeps_bytes(fill_these=frozenset()):
    """The FaceRedactor at walk time: available, under the configured label; it hands the stored
    bytes back untouched (nothing to fill) except for images in `fill_these`, where it paints."""
    import cv2

    from tower.world_builder.redaction import RedactionResult

    class Redactor:
        available = True
        unavailable_reason = None
        label = TRUSTED

        def redact(self, data):
            if data not in fill_these:
                return RedactionResult(image_bytes=data, label=TRUSTED, regions=0)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            img[30:70, 40:90] = 0
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 97])
            return RedactionResult(image_bytes=enc.tobytes(), label=TRUSTED, regions=1)

    return Redactor


def _image(world, ki):
    return (world.images / f"{world.kids[ki].rsplit(':', 1)[-1]}.jpg").read_bytes()


@pytest.fixture
def world(tmp_path):
    return _with_keyframes(World(tmp_path / "a"))


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(AT_STOP, "on")
    monkeypatch.setenv(ENV, "on")


def _live_build(monkeypatch, world, *, redactor=None, device="cuda", params=None, quality_live=True,
                kis=None):
    """The live surface child's transient step DURING the walk (session label `none`, OneFormer
    only, the stage's own backends on the device it decides), then Stop: the real label written.
    Returns (report, the detector stub)."""
    from tower.world_builder import redaction as R
    from tower.world_builder import surface_pipeline as SP

    stub = PixelStub()
    world.set_label("none")
    monkeypatch.setattr(R, "FaceRedactor", redactor or _keeps_bytes())
    monkeypatch.setattr(M, "stage_device", lambda: device)
    monkeypatch.setattr(T, "default_backend_factory", lambda component, device=None: stub(component))
    monkeypatch.setattr(T, "BACKEND_FACTORY", lambda component: stub(component))   # the switch-off path
    kis = list(range(len(world.kids))) if kis is None else list(kis)
    frames = types.SimpleNamespace(items=[(ki,) for ki in kis], kids={ki: world.kids[ki] for ki in kis})
    report = SP._ensure_transients(world.store, WORLD, SESSION, _solution(world), world.intrinsics,
                                   {"records": world.records}, world.dense / "work", frames,
                                   params or T.TransientParams(mode=T.MODE_ONEFORMER), None, None, None,
                                   live=quality_live)
    world.set_label(TRUSTED)                  # Stop: `stop_session` writes the real label
    return report, stub


def _journal(world):
    return world.dense / "work" / "depth" / M.LIVE_JOURNAL


def _lines(world):
    path = _journal(world)
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _worker(world):
    stub = PixelStub()
    return M.prefill(world.store, WORLD, SESSION, backend_factory=stub), stub


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------


def test_the_switch_is_off_by_default(monkeypatch):
    from tower.config import world_masks_live_reuse_setting

    monkeypatch.delenv(ENV, raising=False)
    assert world_masks_live_reuse_setting() is False
    monkeypatch.setenv(ENV, "on")
    assert world_masks_live_reuse_setting() is True


def test_off_a_live_build_journals_nothing_and_the_worker_reuses_nothing(monkeypatch, world):
    monkeypatch.setenv(AT_STOP, "on")
    monkeypatch.delenv(ENV, raising=False)
    report, _ = _live_build(monkeypatch, world)
    assert report.computed == len(world.kids)
    assert not _journal(world).exists()
    summary, stub = _worker(world)
    assert "rekeyed" not in summary
    assert sorted(set(stub.computed(ONE))) == list(range(len(world.kids)))


def test_a_final_build_journals_nothing(monkeypatch, world, on):
    _live_build(monkeypatch, world, quality_live=False)
    assert not _journal(world).exists()


def test_a_live_build_journals_every_mask_it_writes(monkeypatch, world, on):
    report, _ = _live_build(monkeypatch, world)
    lines = _lines(world)
    assert sorted(x["f"] for x in lines) == sorted(T.cache_path(world.dense, ki, ONE).name
                                                   for ki in range(len(world.kids)))
    for x in lines:
        assert {k: x[k] for k in CUDA} == CUDA
        assert x["intr"] == M.intrinsics_digest(world.intrinsics)
        import hashlib

        assert x["sha1"] == hashlib.sha1((world.dense / "work" / "depth" / x["f"]).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# reuse, and what a reused mask must equal
# ---------------------------------------------------------------------------


def test_reused_live_masks_become_hits_equal_to_a_cold_build(tmp_path, monkeypatch, world, on):
    cold_world = _with_keyframes(World(tmp_path / "cold"))
    cold, _ = _final(cold_world, PixelStub(), promote=False)

    _live_build(monkeypatch, world)
    live_keys = {json.loads(bytes(np.load(p)["key"]).decode())["redaction_effective"]
                 for p in (world.dense / "work" / "depth").glob("*_transient.oneformer.npz")}
    assert live_keys == {f"none&{TRUSTED}"}                    # the walk's key, which the final misses

    summary, worker = _worker(world)
    assert summary["state"] == "done" and summary["rekeyed"] == len(world.kids)
    assert worker.computed(ONE) == []                          # OneFormer not run again at Stop
    assert sorted(worker.computed(GDSAM)) == list(range(len(world.kids)))

    stage = PixelStub()
    warm, promoted = _final(world, stage, promote=True)
    assert stage.calls == []
    assert promoted["promoted"] == 2 * len(world.kids) and promoted["refused"] == {}
    assert warm.cached == len(world.kids) and warm.computed == 0
    assert warm.frames_digest() == cold.frames_digest()
    assert _same_masks(_masks(warm), _masks(cold))


def test_a_frame_the_re_redaction_painted_is_computed_again(monkeypatch, world, on):
    """The live detector saw re-redacted bytes for this frame, not the stored ones: its mask is
    of other pixels (the stub's output changes), so it must not become the final's."""
    painted = _image(world, HAND_KI)
    _live_build(monkeypatch, world, redactor=_keeps_bytes({painted}))
    summary, worker = _worker(world)
    assert summary["rekeyed"] == len(world.kids) - 1
    assert worker.computed(ONE) == [HAND_KI]


def test_without_a_journal_nothing_is_reused(monkeypatch, world, on):
    _live_build(monkeypatch, world)
    _journal(world).unlink()
    summary, worker = _worker(world)
    assert summary.get("rekeyed", 0) == 0
    assert len(worker.computed(ONE)) == len(world.kids)


def test_a_journal_line_without_its_newline_is_not_a_line(monkeypatch, world, on):
    _live_build(monkeypatch, world)
    text = _journal(world).read_text()
    _journal(world).write_text(text[:-1])                       # a kill mid-line
    last = json.loads(text.splitlines()[-1])["f"]
    summary, worker = _worker(world)
    assert summary["rekeyed"] == len(world.kids) - 1
    assert worker.computed(ONE) == [int(last.split("_", 1)[0])]


def test_a_mask_made_in_another_execution_mode_is_computed_again(monkeypatch, world, on):
    """The live build ran its detectors on the CPU (fp32): not the worker's or the stage's mode."""
    _live_build(monkeypatch, world, device="cpu")
    assert {x["device"] for x in _lines(world)} == {"cpu"}
    summary, worker = _worker(world)
    assert summary.get("rekeyed", 0) == 0
    assert len(worker.computed(ONE)) == len(world.kids)


def test_a_file_rewritten_after_its_line_is_computed_again(monkeypatch, world, on):
    _live_build(monkeypatch, world)
    path = T.cache_path(world.dense / "work" / "depth", HAND_KI, ONE)
    hand, phone, rec = T.read_component(path)
    key = {k: v for k, v in rec.items() if k not in ("image_sha1", "computed_at", "seconds")}
    T.write_component(path, key, hand, phone, image_sha1=rec["image_sha1"])   # same masks, another write
    summary, worker = _worker(world)
    assert summary["rekeyed"] == len(world.kids) - 1
    assert worker.computed(ONE) == [HAND_KI]


def test_another_calibration_at_stop_reuses_nothing(monkeypatch, world, on):
    from dataclasses import replace

    from tower.world_builder.records import Session

    _live_build(monkeypatch, world)
    world.intrinsics = replace(world.intrinsics, cx=world.intrinsics.cx + 0.5)
    world.store.write_session(Session(session_id=SESSION, world_id=WORLD, started_at=1.0,
                                      redaction=TRUSTED, intrinsics=world.intrinsics))
    summary, worker = _worker(world)
    assert summary.get("rekeyed", 0) == 0
    assert len(worker.computed(ONE)) == len(world.kids)


def test_a_mask_under_another_rule_is_computed_again(monkeypatch, world, on):
    _live_build(monkeypatch, world, params=T.TransientParams(mode=T.MODE_ONEFORMER, oneformer_threshold=0.6))
    summary, worker = _worker(world)
    assert summary.get("rekeyed", 0) == 0
    assert len(worker.computed(ONE)) == len(world.kids)


def test_a_keyframe_whose_stored_bytes_changed_is_computed_again(monkeypatch, world, on):
    _live_build(monkeypatch, world)
    world.set_image(HAND_KI, world.render(HAND_KI)[::-1].copy())
    summary, worker = _worker(world)
    assert summary["rekeyed"] == len(world.kids) - 1
    assert worker.computed(ONE) == [HAND_KI]


def test_a_mask_of_another_camera_size_is_not_reusable(monkeypatch, world, on):
    import hashlib

    _live_build(monkeypatch, world)
    depth = world.dense / "work" / "depth"
    path = T.cache_path(depth, HAND_KI, ONE)
    hand, phone, rec = T.read_component(path)
    key = {k: v for k, v in rec.items() if k not in ("image_sha1", "computed_at", "seconds")}
    T.write_component(path, key, hand[:-1], phone[:-1], image_sha1=rec["image_sha1"])
    with open(_journal(world), "a", encoding="utf-8") as fh:   # a line naming exactly this file
        fh.write(json.dumps(dict(CUDA, intr=M.intrinsics_digest(world.intrinsics), f=path.name,
                                 sha1=hashlib.sha1(path.read_bytes()).hexdigest())) + "\n")
    summary, worker = _worker(world)
    assert summary["rekeyed"] == len(world.kids) - 1
    assert worker.computed(ONE) == [HAND_KI]


def test_a_live_mask_already_under_the_final_label_is_left_to_the_stage(monkeypatch, world, on):
    """Same label at Stop as during the walk (no flip): the stage finds the live mask itself."""
    _live_build(monkeypatch, world)
    cam = _solution(world).camera
    policy = types.SimpleNamespace(effective=f"none&{TRUSTED}")
    got = M.live_reusable(world.store, WORLD, SESSION, keyframe_ids=list(world.kids),
                          image_set=world.store.keyframe_image_set(WORLD, SESSION),
                          params=T.TransientParams(), policy=policy, mode=dict(CUDA),
                          intrinsics=world.intrinsics, shape=(cam["height"], cam["width"]))
    assert got == {}


def test_a_worker_stopped_while_reusing_stages_no_gpu_work(monkeypatch, world, on):
    _live_build(monkeypatch, world)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3

    stub = PixelStub()
    summary = M.prefill(world.store, WORLD, SESSION, backend_factory=stub, should_stop=stop)
    assert summary["state"] == "stopped" and summary["rekeyed"] == 3 and stub.calls == []


def test_reusable_frames_lead_their_group():
    sol = types.SimpleNamespace(keyframe_ids=["a", "b", "c", "d", "e"],
                                poses={"a": {"rotation": [1], "component": 0},
                                       "b": {"rotation": [1], "component": 0},
                                       "c": {"rotation": [1], "component": 1},
                                       "d": {"rotation": [1], "component": 0},
                                       "e": {"rotation": None}})
    kids = ["a", "b", "c", "d", "e", "f", "g"]
    assert M.keyframe_order(kids, sol) == [0, 1, 3, 5, 6, 2, 4]
    assert M.keyframe_order(kids, sol, first={3, 6, 2}) == [3, 0, 1, 6, 5, 2, 4]
    assert M.keyframe_order(kids, None, first={4}) == [4, 0, 1, 2, 3, 5, 6]


def test_a_live_write_that_failed_journals_nothing(monkeypatch, world, on):
    """The stage could not write this mask (a held file, a full disk): the file on disk is an
    older one, which this computation says nothing about, so it gets no journal line."""
    depth = world.dense / "work" / "depth"
    path = T.cache_path(depth, HAND_KI, ONE)
    sha = __import__("hashlib").sha1(_image(world, HAND_KI)).hexdigest()
    old = T.component_key(ONE, T.TransientParams(mode=T.MODE_ONEFORMER), keyframe_id=world.kids[HAND_KI],
                          source_sha1=sha, redaction_effective="older&label")
    full = np.ones((119, 159), bool)
    T.write_component(path, old, full, full, image_sha1=sha)    # an older mask, other masks
    real = T.write_component

    def failing(p, *a, **k):
        if p.name == path.name and p.parent == depth:
            raise OSError("held by another process")
        return real(p, *a, **k)

    monkeypatch.setattr(T, "write_component", failing)
    _live_build(monkeypatch, world)
    monkeypatch.setattr(T, "write_component", real)
    assert path.name not in {x["f"] for x in _lines(world)}
    summary, worker = _worker(world)
    assert summary["rekeyed"] == len(world.kids) - 1
    assert worker.computed(ONE) == [HAND_KI]


def test_off_at_stop_the_worker_reuses_nothing_even_with_a_journal(monkeypatch, world, on):
    _live_build(monkeypatch, world)
    assert _lines(world)
    monkeypatch.delenv(ENV)
    summary, worker = _worker(world)
    assert "rekeyed" not in summary
    assert len(worker.computed(ONE)) == len(world.kids)


def test_a_live_mask_of_other_bytes_than_the_frame_staged_is_not_reused(monkeypatch, world, on):
    """The frame's stored bytes as the worker stages it are the ones the reused mask was read
    from; a mismatch (the keyframe rewritten in between) stages nothing for it."""
    _live_build(monkeypatch, world)
    real = M.live_reusable

    def stale(*a, **k):
        got = real(*a, **k)
        hand, phone, _sha = got[HAND_KI]
        got[HAND_KI] = (hand, phone, "0" * 40)
        return got

    monkeypatch.setattr(M, "live_reusable", stale)
    summary, worker = _worker(world)
    assert summary["rekeyed"] == len(world.kids) - 1
    assert worker.computed(ONE) == [HAND_KI]


def test_a_reused_mask_that_cannot_be_staged_is_computed_by_the_worker(monkeypatch, world, on):
    """A write that fails while staging a live mask (a held file, MAX_PATH) costs that mask, not
    the worker: it computes the frame's OneFormer itself and finishes."""
    _live_build(monkeypatch, world)
    real = T.write_component

    def failing(p, *a, **k):
        if p.name == T.cache_path(p.parent, HAND_KI, ONE).name and p.parent.parent.name == M.STAGING_DIRNAME:
            raise OSError("held")
        return real(p, *a, **k)

    monkeypatch.setattr(T, "write_component", failing)
    stub = PixelStub()
    summary = M.prefill(world.store, WORLD, SESSION, backend_factory=stub)
    monkeypatch.setattr(T, "write_component", real)
    assert summary["state"] == "done" and summary["rekeyed"] == len(world.kids) - 1
    assert stub.computed(ONE) == [HAND_KI]


# ---------------------------------------------------------------------------
# the worker's CPU threads (TOWER_WORLD_MASKS_AT_STOP_THREADS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expect", [(None, None), ("", None), ("0", None), ("x", None), ("-2", None),
                                           ("4", "4")])
def test_the_worker_thread_cap(monkeypatch, world, value, expect):
    """Unset (or 0, or garbage) leaves the worker's environment exactly the builder's child
    environment; N starts it with OMP_NUM_THREADS = MKL_NUM_THREADS = N."""
    from tests.test_world_builder_masks_at_stop import _builder_module

    wbs = _builder_module("wbs_masks_threads_test")
    monkeypatch.setenv(AT_STOP, "on")
    if value is None:
        monkeypatch.delenv("TOWER_WORLD_MASKS_AT_STOP_THREADS", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_MASKS_AT_STOP_THREADS", value)
    seen = {}

    class Child:
        pid, returncode = 4242, None

        def poll(self):
            return self.returncode

    def spawn(*a, **k):
        seen.update(k)
        return Child()

    monkeypatch.setattr(wbs, "_terminate_process_tree", lambda p: None)
    worker = wbs.MasksAtStop.start_if_on(world.store, root=world.root, world_id=WORLD, session_id=SESSION,
                                         sources={}, wanted=True, spawn=spawn)
    assert worker is not None
    worker.close()
    base = wbs.child_environment()
    if expect is None:
        assert seen["env"] == base
    else:
        assert seen["env"] == {**base, "OMP_NUM_THREADS": expect, "MKL_NUM_THREADS": expect}


# ---------------------------------------------------------------------------
# round 2 (reviews MASKCOV.R1 / R2)
# ---------------------------------------------------------------------------


def test_without_masks_at_stop_a_live_build_journals_nothing(monkeypatch, world):
    """LIVE_REUSE feeds the masks-at-Stop worker; with that switch off nothing reads a journal, so
    none is written (and the live detectors are the stage's default, untouched)."""
    monkeypatch.delenv(AT_STOP, raising=False)
    monkeypatch.setenv(ENV, "on")
    report, _ = _live_build(monkeypatch, world)
    assert report.computed == len(world.kids)
    assert not _journal(world).exists()


def test_a_mask_made_with_other_library_versions_is_computed_again(monkeypatch, world, on):
    """Same device and fp16, another torch: the versions are part of the mode a line must match."""
    monkeypatch.setattr(M, "execution_mode", lambda device=None: dict(CUDA, torch="2.other"))
    _live_build(monkeypatch, world)
    assert {x["torch"] for x in _lines(world)} == {"2.other"}
    monkeypatch.setattr(M, "execution_mode", lambda device=None: dict(CUDA))
    summary, worker = _worker(world)
    assert summary.get("rekeyed", 0) == 0
    assert len(worker.computed(ONE)) == len(world.kids)


def test_a_mask_of_another_model_revision_is_computed_again(monkeypatch, world, on):
    (model_id, _rev), = T.COMPONENT_MODELS[ONE]
    with monkeypatch.context() as m:
        m.setitem(T.COMPONENT_MODELS, ONE, ((model_id, "another-revision"),))
        _live_build(monkeypatch, world)
    summary, worker = _worker(world)
    assert summary.get("rekeyed", 0) == 0
    assert len(worker.computed(ONE)) == len(world.kids)


def _older_file_then_failed_write(monkeypatch, world, arrays):
    """An older mask file at HAND_KI's path (another label in its key, so a live build recomputes it),
    holding `arrays` ("same": this computation's own masks; else other ones), then a live build
    whose write of that mask fails."""
    depth = world.dense / "work" / "depth"
    path = T.cache_path(depth, HAND_KI, ONE)
    _live_build(monkeypatch, world, kis=[HAND_KI])            # this computation's masks, for "same"
    hand, phone, rec = T.read_component(path)
    _journal(world).unlink()
    key = {k: v for k, v in rec.items() if k not in ("image_sha1", "computed_at", "seconds")}
    if arrays != "same":
        hand, phone = ~hand, ~phone
    T.write_component(path, dict(key, redaction_effective="older&label"), hand, phone,
                      image_sha1=rec["image_sha1"])
    real = T.write_component

    def failing(p, *a, **k):
        if p == path:
            raise OSError("held by another process")
        return real(p, *a, **k)

    monkeypatch.setattr(T, "write_component", failing)
    _live_build(monkeypatch, world)
    monkeypatch.setattr(T, "write_component", real)
    return path


def test_an_older_file_holding_the_same_masks_gets_no_line(monkeypatch, world, on):
    """Only `computed_at` tells this file is not this computation's write: the masks are equal."""
    path = _older_file_then_failed_write(monkeypatch, world, "same")
    assert path.name not in {x["f"] for x in _lines(world)}
    assert len(_lines(world)) == len(world.kids) - 1


def test_a_fresh_file_holding_other_masks_gets_no_line(monkeypatch, world, on):
    """Written after the call (fresh `computed_at`), but not these masks: another writer's."""
    depth = world.dense / "work" / "depth"
    path = T.cache_path(depth, HAND_KI, ONE)
    real = T.write_component

    def other(p, key, hand, phone, **k):
        if p == path:
            return real(p, key, ~np.asarray(hand, bool), ~np.asarray(phone, bool), **k)
        return real(p, key, hand, phone, **k)

    monkeypatch.setattr(T, "write_component", other)
    _live_build(monkeypatch, world)
    monkeypatch.setattr(T, "write_component", real)
    assert path.name not in {x["f"] for x in _lines(world)}
    assert len(_lines(world)) == len(world.kids) - 1


def test_the_worker_runs_the_reusable_frames_first(monkeypatch, world, on):
    """`prefill` hands `keyframe_order` the reusable frames: they lead the GPU loop."""
    _live_build(monkeypatch, world, kis=[5, 6, 7])
    summary, worker = _worker(world)
    assert summary["rekeyed"] == 3
    assert worker.computed(GDSAM) == [5, 6, 7, 0, 1, 2, 3, 4]
    assert worker.computed(ONE) == [0, 1, 2, 3, 4]


def test_a_recorder_that_fails_costs_the_journal_not_the_live_build(monkeypatch, world, on):
    def broken(data):
        raise ValueError("an unforeseen failure while checking the write")

    monkeypatch.setattr(M, "_decode_component", broken)
    report, _ = _live_build(monkeypatch, world)
    assert report.state == T.STATE_OK and report.computed == len(world.kids)
    assert _lines(world) == []


# ---------------------------------------------------------------------------
# evidence for verification (TOWER_WORLD_MASKS_AT_STOP_EVIDENCE)
# ---------------------------------------------------------------------------

EVIDENCE = "TOWER_WORLD_MASKS_AT_STOP_EVIDENCE"


def test_without_the_evidence_switch_nothing_is_kept(monkeypatch, world, on):
    monkeypatch.delenv(EVIDENCE, raising=False)
    _live_build(monkeypatch, world)
    _worker(world)
    _final(world, PixelStub(), promote=True)
    assert not M.evidence_dir(world.store, WORLD, SESSION).exists()


def test_evidence_keeps_every_promoted_mask_and_its_fill_past_the_prune(monkeypatch, world, on):
    import shutil

    monkeypatch.setenv(EVIDENCE, "on")
    _live_build(monkeypatch, world)
    _worker(world)
    _, promoted = _final(world, PixelStub(), promote=True)
    root = M.evidence_dir(world.store, WORLD, SESSION)
    lines = [json.loads(x) for x in (root / M.EVIDENCE_LISTING).read_text().splitlines()]
    assert promoted["promoted"] == len(lines) == 2 * len(world.kids)
    assert sorted((x["component"], x["from"]) for x in lines) == sorted(
        [(ONE, M.LIVE_ORIGIN)] * len(world.kids) + [(GDSAM, "worker")] * len(world.kids))
    assert all(x["mode"] == CUDA for x in lines)
    depth = world.dense / "work" / "depth"
    kept = {x["f"]: (root / x["f"]).read_bytes() for x in lines}
    assert all(kept[n] == (depth / n).read_bytes() for n in kept)
    fills = {n: np.load(depth / n) for n in (f"{ki:05d}_fill.npy" for ki in range(len(world.kids)))}
    shutil.rmtree(world.dense / "work")                        # the final's prune
    assert all((root / n).read_bytes() == b for n, b in kept.items())
    assert all(np.array_equal(np.load(root / n), f) for n, f in fills.items())
    assert T.read_component(root / T.cache_path(root, HAND_KI, ONE).name)[2]["staged_from"] == M.LIVE_ORIGIN


def test_evidence_that_cannot_be_kept_costs_no_promotion(monkeypatch, world, on):
    monkeypatch.setenv(EVIDENCE, "on")
    _live_build(monkeypatch, world)
    _worker(world)

    def broken(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(M, "_keep", broken)
    stage = PixelStub()
    warm, promoted = _final(world, stage, promote=True)
    assert promoted["promoted"] == 2 * len(world.kids) and stage.calls == []


def test_a_link_that_fails_is_a_recorded_skip_never_a_copy(monkeypatch, world, on):
    """Evidence must not double the walk's masks on disk: no copy fallback. The skip is recorded,
    so a verification counts it as a shortfall; the promotion is unaffected."""
    monkeypatch.setenv(EVIDENCE, "on")
    _live_build(monkeypatch, world)
    _worker(world)

    def no_links(src, dst):
        raise OSError("hard links are not supported here")

    monkeypatch.setattr(M.os, "link", no_links)
    _, promoted = _final(world, PixelStub(), promote=True)
    root = M.evidence_dir(world.store, WORLD, SESSION)
    lines = [json.loads(x) for x in (root / M.EVIDENCE_LISTING).read_text().splitlines()]
    assert promoted["promoted"] == len(lines) == 2 * len(world.kids)
    assert all(x["kept"] is False and "hard links" in x["why"] for x in lines)
    assert not list(root.glob("*.npz")) and not list(root.glob("*_fill.npy"))


def test_the_promotion_summary_matches_its_listing(monkeypatch, world, on):
    monkeypatch.setenv(EVIDENCE, "on")
    _live_build(monkeypatch, world)
    _worker(world)
    _, promoted = _final(world, PixelStub(), promote=True)
    root = M.evidence_dir(world.store, WORLD, SESSION)
    summary = json.loads((root / "promote.json").read_text())
    lines = (root / M.EVIDENCE_LISTING).read_text().splitlines()
    assert summary["promoted"] == promoted["promoted"] == len(lines)
    assert all(json.loads(x)["kept"] is True for x in lines)


def test_old_evidence_is_swept_and_fresh_evidence_kept(monkeypatch, world, on):
    """`discard_session` (run by every promotion and by the builder) sweeps evidence of any session
    untouched for `EVIDENCE_MAX_AGE_S`; this promotion's own evidence survives its own discard."""
    import os
    import time

    monkeypatch.setenv(EVIDENCE, "on")
    old = M.evidence_dir(world.store, "another-world", "old-session")
    young = M.evidence_dir(world.store, "another-world", "young-session")
    for d, age in ((old, M.EVIDENCE_MAX_AGE_S + 3600), (young, 3600)):
        d.mkdir(parents=True)
        (d / M.EVIDENCE_LISTING).write_text("{}\n")
        t = time.time() - age
        os.utime(d / M.EVIDENCE_LISTING, (t, t))
        os.utime(d, (t, t))
    _live_build(monkeypatch, world)
    _worker(world)
    _final(world, PixelStub(), promote=True)                  # promote ends with discard_session
    assert not old.exists() and young.exists()
    assert (M.evidence_dir(world.store, WORLD, SESSION) / "promote.json").is_file()


def test_evidence_whose_directory_cannot_be_made_costs_no_promotion(monkeypatch, world, on):
    """Anything else going wrong while keeping evidence (here its directory cannot be created) is
    logged once; the promotion is complete."""
    monkeypatch.setenv(EVIDENCE, "on")
    _live_build(monkeypatch, world)
    _worker(world)
    root = M.evidence_dir(world.store, WORLD, SESSION)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text("a file where the evidence directory would go")
    stage = PixelStub()
    _, promoted = _final(world, stage, promote=True)
    assert promoted["promoted"] == 2 * len(world.kids) and stage.calls == []
