"""fp16 autocast for the transient detectors' model forward calls.

Module: `tower/world_builder/transients.py`. Autocast is used on a CUDA
device only (`T._fp16_enabled`).

No real model is loaded here. Every test that exercises a backend's `run()`
replaces the four `transformers` classes it imports with tiny fakes (via
`sys.modules["transformers"]`, so nothing downloads and nothing needs a real
CUDA device), and replaces `torch.autocast` with a fake that records whether
it was entered around the model call. `T._fp16_enabled` is monkeypatched
directly in those tests so they do not also need a CUDA device.
"""

from __future__ import annotations

import sys
import types

import numpy as np

from tower.world_builder import transients as T


# ---------------------------------------------------------------------------
# the device decision
# ---------------------------------------------------------------------------


def test_fp16_enabled_is_false_on_cpu():
    assert T._fp16_enabled("cpu") is False


def test_fp16_enabled_is_true_on_cuda():
    assert T._fp16_enabled("cuda") is True


def test_autocast_disabled_is_a_noop_context():
    import contextlib

    ctx = T._autocast(False)
    assert isinstance(ctx, type(contextlib.nullcontext()))


def test_autocast_enabled_builds_a_cuda_fp16_autocast():
    import torch

    ctx = T._autocast(True)
    assert isinstance(ctx, torch.autocast)


# ---------------------------------------------------------------------------
# fakes shared by the backend-level tests
# ---------------------------------------------------------------------------


class _FT:
    """A minimal stand-in for a torch tensor. `.float()` appends `tag` to
    `log` (when given) so a test can see whether OUR new cast ran; `.cpu()`,
    `.numpy()` and indexing behave just enough like a tensor for the code
    under test."""

    def __init__(self, arr=None, *, tag=None, log=None):
        self.arr = None if arr is None else np.asarray(arr)
        self.tag = tag
        self.log = log

    def float(self):
        if self.log is not None:
            self.log.append(self.tag)
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr

    def __getitem__(self, idx):
        return _FT(None if self.arr is None else self.arr[idx], tag=self.tag, log=self.log)


class _Batch(dict):
    """A fake processor output: supports `.to(dev)` and `.input_ids`, and is
    unpackable with `**` into a model call."""

    def to(self, dev):
        return self

    @property
    def input_ids(self):
        return self.get("input_ids")


class _Proc:
    """A fake processor: callable, plus whatever `post_process_*` method a
    test attaches."""

    def __init__(self, call_fn, **methods):
        self._call_fn = call_fn
        for name, fn in methods.items():
            setattr(self, name, fn)

    def __call__(self, **kwargs):
        return self._call_fn(**kwargs)


class _Model:
    """A fake model: `.to(dev).eval()` return itself, and calling it runs
    `call_fn`."""

    def __init__(self, call_fn, config=None):
        self._call_fn = call_fn
        self.config = config if config is not None else types.SimpleNamespace(id2label={})

    def to(self, dev):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        return self._call_fn(**kwargs)


class _FakeAutocast:
    """Replaces `torch.autocast`: records entry/exit into `state["active"]`
    so a model-call fake can see whether it ran while autocast was active."""

    def __init__(self, state):
        self.state = state

    def __call__(self, device_type, dtype=None):
        return self

    def __enter__(self):
        self.state["active"] = True
        return self

    def __exit__(self, *exc):
        self.state["active"] = False
        return False


def _patch_autocast(monkeypatch):
    import torch

    state = {"active": False}
    monkeypatch.setattr(torch, "autocast", _FakeAutocast(state))
    return state


def _run_gdsam(monkeypatch, fp16: bool):
    """Runs `GroundingDinoSamBackend.run()` on one synthetic frame with every
    transformers class and the model fetch faked. Returns
    (gdino_call_log, sam_call_log, cast_log, results) where the call logs
    hold the autocast-active flag observed at each model call, and cast_log
    holds the tags `.float()` was called with (our new conditional cast)."""
    autocast_state = _patch_autocast(monkeypatch)
    monkeypatch.setattr(T, "_fp16_enabled", lambda dev: fp16)
    monkeypatch.setattr(T, "_fetch", lambda *a, **kw: "dummy-dir")

    gdino_calls, sam_calls, cast_log = [], [], []

    def gdino_call(**kwargs):
        gdino_calls.append(autocast_state["active"])
        return types.SimpleNamespace(
            logits=_FT(tag="logits", log=cast_log),
            pred_boxes=_FT(tag="pred_boxes", log=cast_log),
        )

    def sam_call(**kwargs):
        sam_calls.append(autocast_state["active"])
        return types.SimpleNamespace(
            pred_masks=_FT(np.zeros((1, 1, 8, 8), dtype=bool), tag="pred_masks", log=cast_log),
        )

    def fake_post_process_gdino(outputs, input_ids, threshold, text_threshold, target_sizes):
        return [{
            "boxes": _FT(np.array([[0.0, 0.0, 6.0, 6.0]])),
            "scores": _FT(np.array([0.9])),
            "text_labels": ["hand"],
        }]

    def fake_post_process_masks(masks, original_sizes):
        return [_FT(np.zeros((1, 1, 8, 8), dtype=bool))]

    dproc = _Proc(lambda **kw: _Batch(input_ids="ids"),
                  post_process_grounded_object_detection=fake_post_process_gdino)
    sproc = _Proc(lambda **kw: _Batch(original_sizes=_FT(np.array([[8, 8]]))),
                  post_process_masks=fake_post_process_masks)
    dmodel = _Model(gdino_call)
    smodel = _Model(sam_call)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **kw: dproc)
    fake_transformers.GroundingDinoForObjectDetection = types.SimpleNamespace(
        from_pretrained=lambda *a, **kw: dmodel)
    fake_transformers.Sam2Model = types.SimpleNamespace(from_pretrained=lambda *a, **kw: smodel)
    fake_transformers.Sam2Processor = types.SimpleNamespace(from_pretrained=lambda *a, **kw: sproc)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend = T.GroundingDinoSamBackend()
    items = [(1, np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=bool))]
    results = []
    backend.run(items, T.TransientParams(), lambda ki, hand, phone, seconds: results.append((ki, hand, phone)))
    return gdino_calls, sam_calls, cast_log, results


def _run_oneformer(monkeypatch, fp16: bool):
    """Same shape as `_run_gdsam` for `OneFormerBackend.run()`."""
    autocast_state = _patch_autocast(monkeypatch)
    monkeypatch.setattr(T, "_fp16_enabled", lambda dev: fp16)
    monkeypatch.setattr(T, "_fetch", lambda *a, **kw: "dummy-dir")

    model_calls, cast_log = [], []

    def model_call(**kwargs):
        model_calls.append(autocast_state["active"])
        return types.SimpleNamespace(
            class_queries_logits=_FT(tag="class_queries_logits", log=cast_log),
            masks_queries_logits=_FT(tag="masks_queries_logits", log=cast_log),
        )

    def fake_post_process_panoptic(outputs, target_sizes, threshold):
        return [{"segmentation": _FT(np.full((8, 8), -1, dtype=np.int64)), "segments_info": []}]

    proc = _Proc(lambda **kw: _Batch(), post_process_panoptic_segmentation=fake_post_process_panoptic)
    model = _Model(model_call)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.OneFormerForUniversalSegmentation = types.SimpleNamespace(
        from_pretrained=lambda *a, **kw: model)
    fake_transformers.OneFormerProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **kw: proc)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend = T.OneFormerBackend()
    items = [(1, np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=bool))]
    results = []
    backend.run(items, T.TransientParams(), lambda ki, hand, phone, seconds: results.append((ki, hand, phone)))
    return model_calls, cast_log, results


# ---------------------------------------------------------------------------
# cpu (`_fp16_enabled` is False): no autocast, no cast
# ---------------------------------------------------------------------------


def test_gdino_and_sam_run_without_autocast_on_cpu(monkeypatch):
    gdino_calls, sam_calls, cast_log, results = _run_gdsam(monkeypatch, fp16=False)
    assert gdino_calls == [False]
    assert sam_calls == [False]
    assert cast_log == []
    assert len(results) == 1


def test_oneformer_runs_without_autocast_on_cpu(monkeypatch):
    model_calls, cast_log, results = _run_oneformer(monkeypatch, fp16=False)
    assert model_calls == [False]
    assert cast_log == []
    assert len(results) == 1


# ---------------------------------------------------------------------------
# cuda: autocast wraps the forward call, outputs are cast back for post-process
# ---------------------------------------------------------------------------


def test_gdino_and_sam_forward_calls_run_inside_autocast_when_on(monkeypatch):
    gdino_calls, sam_calls, cast_log, results = _run_gdsam(monkeypatch, fp16=True)
    assert gdino_calls == [True]
    assert sam_calls == [True]
    assert len(results) == 1


def test_gdino_and_sam_outputs_are_cast_to_float_before_post_process(monkeypatch):
    _, _, cast_log, _ = _run_gdsam(monkeypatch, fp16=True)
    # logits/pred_boxes cast before `post_process_grounded_object_detection`,
    # then pred_masks cast before `post_process_masks`.
    assert cast_log == ["logits", "pred_boxes", "pred_masks"]


def test_oneformer_forward_call_runs_inside_autocast_when_on(monkeypatch):
    model_calls, cast_log, results = _run_oneformer(monkeypatch, fp16=True)
    assert model_calls == [True]
    assert len(results) == 1


def test_oneformer_outputs_are_cast_to_float_before_post_process(monkeypatch):
    _, cast_log, _ = _run_oneformer(monkeypatch, fp16=True)
    assert cast_log == ["class_queries_logits", "masks_queries_logits"]
