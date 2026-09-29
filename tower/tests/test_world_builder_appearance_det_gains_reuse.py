"""DET-GAINS-F (the Codex C27x review of 2026-09-28): what a saved appearance is REUSED as.

1. THE EFFECTIVE PATH IS IN THE KEY. With `TOWER_WORLD_APPEARANCE_DETERMINISTIC_GAINS` on, the exposure solve sums
   with `index_put_(accumulate=True)` on CUDA and with `index_add_` on the CPU fallback. The reuse key (the params
   digest `appearance_pipeline._already_built` matches) used to say only "on", so an ON build made on the CPU
   fallback was served as ALREADY_BUILT to a later ON build on CUDA -- the CUDA op never ran -- and the reverse.
   Now, ON only, the key names the op the build EFFECTIVELY takes (`A.exposure_accumulate_path`, digest input
   `exposure_accumulate`), the manifest's `exposure.accumulate` says which op ran, and a build whose solve ran
   another op than its key names is refused. OFF: no key and no record, the device never entered the OFF key and
   still does not, and every OFF digest is the one before the switch.

   THE DEVICE DECISION IS SIMULATED AT ITS SOURCE: `build_appearance(device=None)`, the product's call, with the
   device the product itself RESOLVES (`A._torch_device`) answered as if it were CUDA (the device test's `_watch`).
   The pipeline's key and the solve's 15 sums then see one answer. On this CPU host the values are the CPU's; which
   op each sum runs, and what the key says, is the point.

2. A PRE-PATCH SAVED WORLD. `golden/world_builder_appearance_prepatch_44fbd13.zip` is a whole saved world -- the
   appearance suite's synthetic box room, solved, depth-staged, surfaced and shaded with the FINAL preset
   `AppearanceParams()` -- written by the UNEDITED 44fbd13 code (appearance.py blob a4eaf7b7; recorder
   RUN/experiments/DET-GAINS/prepatch_record.py, run from `git archive 44fbd13 tower`, CPU). Under this code it is
   LOADED (manifest, currency, routes), RENDERED (the appearance page and every published texture decoded) and
   FINISHED (`world_build_session.final_surface_stages`, the finisher's own appearance call with its own params;
   the surface stage stands in for a surface that did not change):
     - switch OFF: the finish is served ALREADY_BUILT under the pre-patch params digest, and not one byte of the
       saved appearance, nor of the page, changes;
     - switch ON (the CPU fallback, and the device decision answered as CUDA): the finish REBUILDS, never reusing
       the pre-patch build; it records the switch and the op; it renders; OFF afterwards returns to the pre-patch
       digest. On the CPU fallback ON runs the OFF op, so its gains are the pre-patch gains.
   The zip is a zip, not a directory, because git's autocrlf would rewrite a JSON file's bytes. Re-recording it
   means running the recorder on a 44fbd13 archive again: the fixture is only a fixture for "saved before the
   switch existed" while 44fbd13's code wrote it.

CPU only: CUDA is hidden from every build here, so no test takes a GPU because one happens to be visible.
"""

from __future__ import annotations

import hashlib
import json
import platform
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tests.test_world_builder_appearance import SESSION, WORLD, World, _never_redact
from tests.test_world_builder_appearance_det_gains import (
    ENV,
    GOLDEN_PARAMS,
    KEY,
    _as_if_cuda,
    _count_index_ops,
    _solve,
    index_sums,
)
from tests.test_world_builder_appearance_det_gains_device import CUDA_DEVICE, _hide_cuda
from tower.world_builder import appearance as A
from tower.world_builder import appearance_pipeline as AP

ATOMIC, DETERMINISTIC = A.EXPOSURE_ACCUMULATE_ATOMIC, A.EXPOSURE_ACCUMULATE_DETERMINISTIC
FIXTURE = Path(__file__).parent / "golden" / "world_builder_appearance_prepatch_44fbd13.zip"
FIXTURE_SHA256 = "772e5d0bf540fdd7a719444421fb9e4ce303c198b435a04a61bcfedb02803654"
PREPATCH_BLOB = "a4eaf7b7baf285209f329fecf6ba486d07af235c"      # 44fbd13's appearance.py
# What two builds of one world always differ in (measured: a forced OFF rebuild of the fixture under this code).
VOLATILE = ("build_id", "built_at", "seconds")


def _as_if_cuda_host(monkeypatch) -> list:
    """The one thing a CUDA host changes: `device=None` resolves to CUDA. Here it resolves to a CPU device object
    (the tensors stay on the CPU) that every decision answers as CUDA; an EXPLICIT device is resolved and answered
    exactly as the product does. So a key or a solve that stopped following the product's own `device=None`
    resolution answers differently from the one that follows it. Returns the objects resolved from None."""
    real_device, real_decide = A._torch_device, A._deterministic_accumulate
    from_none: list = []

    def torch_device(device=None):
        dev = real_device(device)
        if device is None:
            from_none.append(dev)
        return dev

    def decide(params, dev):
        return real_decide(params, CUDA_DEVICE if any(dev is d for d in from_none) else dev)

    monkeypatch.setattr(A, "_torch_device", torch_device)
    monkeypatch.setattr(A, "_deterministic_accumulate", decide)
    return from_none


# -- 1. the effective path in the key ------------------------------------------------------------------------------


def test_the_key_names_the_op_the_solve_will_take_and_nothing_when_off(monkeypatch):
    on = A.AppearanceParams(exposure_deterministic_gains=True)
    off = A.AppearanceParams(exposure_deterministic_gains=False)
    for dev in (None, "cpu", "cuda", torch.device("cuda", 0), torch.device("meta")):
        assert A.exposure_accumulate_path(off, dev) is None
    assert A.exposure_accumulate_path(on, "cpu") == ATOMIC
    assert A.exposure_accumulate_path(on, torch.device("meta")) == ATOMIC
    assert A.exposure_accumulate_path(on, "cuda") == DETERMINISTIC
    assert A.exposure_accumulate_path(on, torch.device("cuda", 0)) == DETERMINISTIC
    # device=None is resolved exactly as the solve resolves it (a device object only: no GPU is touched)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert A.exposure_accumulate_path(on, None) == ATOMIC
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert A.exposure_accumulate_path(on, None) == DETERMINISTIC
    # a live build never takes the switch, so its key never names an op either
    monkeypatch.setenv(ENV, "on")
    assert A.exposure_accumulate_path(A.AppearanceParams.live(), "cuda") is None
    assert A.exposure_accumulate_path(A.AppearanceParams(), "cuda") == DETERMINISTIC


def test_the_solve_records_the_op_its_sums_took_only_when_on(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    _g, _o, r_off = _solve(A.AppearanceParams(**GOLDEN_PARAMS))
    assert "accumulate" not in r_off
    monkeypatch.setenv(ENV, "on")
    _g, _o, r_cpu = _solve(A.AppearanceParams(**GOLDEN_PARAMS))
    assert r_cpu["accumulate"] == ATOMIC
    _as_if_cuda(monkeypatch)
    counts = _count_index_ops(monkeypatch)
    final = A.AppearanceParams(**GOLDEN_PARAMS)
    _g, _o, r_cuda = _solve(final)
    assert r_cuda["accumulate"] == DETERMINISTIC
    assert counts["index_add_"] == 0 and counts["index_put_"] == index_sums(final)
    # the one key the switch adds to the record, and only when on
    assert set(r_cpu) - {"accumulate"} == set(r_off) == set(r_cuda) - {"accumulate"}


def _build(w, **kw):
    """The product's call: no device, so the build resolves it (to the CPU here: CUDA is hidden)."""
    return w.build(params=A.AppearanceParams(selection_samples=4000), redactor_factory=_never_redact,
                   device=None, **kw)


def test_an_on_build_on_the_cpu_fallback_is_never_served_on_cuda_nor_the_reverse(monkeypatch, tmp_path):
    _hide_cuda(monkeypatch)
    w = World(tmp_path)
    counts = _count_index_ops(monkeypatch)

    monkeypatch.delenv(ENV, raising=False)
    assert _build(w).state == AP.STATE_OK
    m_off = w.manifest()
    d_off = m_off["params_digest"]
    assert KEY not in m_off["params"] and "accumulate" not in m_off["exposure"]

    # ON on the CPU fallback: a new key, and the manifest says the OFF op ran
    monkeypatch.setenv(ENV, "on")
    before = Counter(counts)
    r_cpu = _build(w)
    assert r_cpu.state == AP.STATE_OK and r_cpu.detail != AP.ALREADY_BUILT
    m_cpu = w.manifest()
    d_cpu = m_cpu["params_digest"]
    assert m_cpu["params"][KEY] is True and m_cpu["exposure"]["accumulate"] == ATOMIC
    assert d_cpu != d_off
    assert counts["index_put_"] == before["index_put_"] and counts["index_add_"] > before["index_add_"]
    assert _build(w).detail == AP.ALREADY_BUILT

    # ON on CUDA: the CPU-fallback ON build is NOT served; the CUDA op runs; its own key serves it afterwards
    with monkeypatch.context() as mp:
        from_none = _as_if_cuda_host(mp)
        before = Counter(counts)
        r_cuda = _build(w)
        assert r_cuda.state == AP.STATE_OK, r_cuda.detail
        assert r_cuda.detail != AP.ALREADY_BUILT
        assert from_none                                    # the product resolved device=None
        m_cuda = w.manifest()
        d_cuda = m_cuda["params_digest"]
        assert m_cuda["params"][KEY] is True and m_cuda["exposure"]["accumulate"] == DETERMINISTIC
        assert d_cuda not in (d_off, d_cpu)
        assert counts["index_add_"] == before["index_add_"] and counts["index_put_"] > before["index_put_"]
        assert _build(w).detail == AP.ALREADY_BUILT
        assert w.manifest()["params_digest"] == d_cuda

    # back on the CPU fallback: the CUDA ON build is NOT served either; the CPU key comes back exactly
    r_back = _build(w)
    assert r_back.state == AP.STATE_OK and r_back.detail != AP.ALREADY_BUILT
    assert w.manifest()["params_digest"] == d_cpu and w.manifest()["exposure"]["accumulate"] == ATOMIC

    # OFF: rebuilt from the ON build, back to the OFF key; and the device never entered the OFF key, as before
    # the switch existed (a CPU OFF build is served to an OFF build on CUDA, exactly as it always was)
    monkeypatch.delenv(ENV)
    r_off = _build(w)
    assert r_off.state == AP.STATE_OK and r_off.detail != AP.ALREADY_BUILT
    assert w.manifest()["params_digest"] == d_off
    with monkeypatch.context() as mp:
        _as_if_cuda_host(mp)
        assert _build(w).detail == AP.ALREADY_BUILT
        assert w.manifest()["params_digest"] == d_off and "accumulate" not in w.manifest()["exposure"]


def test_a_build_whose_solve_ran_another_op_than_its_key_names_is_refused_and_publishes_nothing(
        monkeypatch, tmp_path):
    _hide_cuda(monkeypatch)
    monkeypatch.setenv(ENV, "on")
    w = World(tmp_path)
    assert _build(w).state == AP.STATE_OK
    root = AP.appearance_dir(w.store, WORLD, SESSION)
    published = (root / "manifest.json").read_bytes()
    real = A.solve_gains

    def lying_solve(*a, **k):                  # the CPU ran index_add_, the record claims the CUDA op
        gains, per_frame, record = real(*a, **k)
        assert record["accumulate"] == ATOMIC
        return gains, per_frame, {**record, "accumulate": DETERMINISTIC}

    monkeypatch.setattr(A, "solve_gains", lying_solve)
    r = _build(w, force=True)
    assert r.state == AP.STATE_UNAVAILABLE and r.retryable is True
    assert DETERMINISTIC in r.detail and ATOMIC in r.detail
    assert (root / "manifest.json").read_bytes() == published


# -- 2. a saved world written before the switch existed -------------------------------------------------------------


def _saved_world(tmp_path):
    """The 44fbd13 world, unpacked; returns (store, fixture.json)."""
    from tower.world_builder.store import WorldStore

    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    root = tmp_path / "saved"
    with zipfile.ZipFile(FIXTURE) as z:
        info = json.loads(z.read("fixture.json"))
        for name in z.namelist():
            if name != "fixture.json":
                z.extract(name, root)
    assert info["commit"] == "44fbd13" and info["appearance_py_blob"] == PREPATCH_BLOB
    assert (info["world"], info["session"]) == (WORLD, SESSION)
    return WorldStore(root), info


def _host_encoders_are_the_recordings(info) -> bool:
    """The params digest names the texture encoders and their versions, so a host with other encoders rebuilds a
    saved world whatever the switch says; the OFF 'served unchanged' claim is checked on a host like the recording."""
    here = {k: [v["encoder"], v["version"], v["quality"], v["available"]]
            for k, v in A.encoder_versions(A.AppearanceParams(exposure_deterministic_gains=False)).items()}
    return here == info["encoders"]


def _host_is_the_recordings(info) -> bool:
    return (info["torch"], info["numpy"], info["cv2"], info["machine"], info["system"]) == (
        torch.__version__, np.__version__, cv2.__version__, platform.machine(), platform.system())


def _appearance_files(store) -> dict:
    """Every published appearance file's bytes (sha256), by name. `status.json` is the stage's live status, which
    every call rewrites, and the lock is transient: neither is the saved world."""
    root = AP.appearance_dir(store, WORLD, SESSION)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())
            if p.is_file() and p.name != "status.json" and not p.name.endswith(".lock")}


def _render(store) -> tuple:
    """What a viewer gets: the page, and every texture it can fetch, decoded."""
    from tower.world_builder.appearance_render import build_appearance_page

    page = build_appearance_page(store, WORLD, SESSION)
    man = AP.read_appearance_manifest(store, WORLD, SESSION)
    assert AP.read_appearance_file(store, WORLD, SESSION, "proxy", man["proxy"]["digest"], man) is not None
    astc_ok, _why = A.astc_available()
    textures = {}
    for k in man["keyframes"]:
        for enc, ref in (k.get("chunks") or {}).items():
            data = AP.read_appearance_file(store, WORLD, SESSION, "chunk", ref["digest"], man)
            assert data is not None, (k["ki"], enc)
            chunk = A.read_chunk(data)
            blob = chunk["slots"][ref["slot"]]
            if enc == A.ENC_ASTC and not astc_ok:      # no decoder on this host: the blocks themselves
                textures[(k["ki"], enc)] = hashlib.sha256(blob).hexdigest()
                continue
            img = (A.decode_astc(blob, chunk["width"], chunk["height"]) if enc == A.ENC_ASTC
                   else A.decode_webp(blob))
            textures[(k["ki"], enc)] = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
    assert textures, "nothing to draw"
    return page, textures


def _finish(store, monkeypatch) -> dict:
    """The finish's final stages -- `world_build_session.final_surface_stages`, which builds the appearance with
    its own `AppearanceParams()` and `device=None` -- on a surface that did not change (the surface stage returns
    `ok` and leaves the saved surface as it is)."""
    from scripts.world_build_session import final_surface_stages
    from tower.world_builder import surface_pipeline
    from tower.world_builder.surface import SurfaceResult

    monkeypatch.setattr(surface_pipeline, "surfacify", lambda *a, **k: SurfaceResult(state="ok"))
    return final_surface_stages(store, WORLD, SESSION, solved=True, appearance=True, prune_depth_work=False,
                                should_stop=lambda: False)


def _gains(man) -> dict:
    return {k["id"]: k["gain"] for k in man["keyframes"]}


def test_a_pre_patch_saved_world_with_the_switch_off_loads_renders_and_finishes_unchanged(monkeypatch, tmp_path):
    _hide_cuda(monkeypatch)
    monkeypatch.delenv(ENV, raising=False)
    store, info = _saved_world(tmp_path)
    if not _host_encoders_are_the_recordings(info):
        pytest.skip(f"this host's texture encoders are not the recording's {info['encoders']}: the params digest "
                    "names them, so this saved world would be rebuilt here whatever the switch says")
    # load
    man0 = AP.read_appearance_manifest(store, WORLD, SESSION)
    assert man0["params_digest"] == info["params_digest"] and man0["build_id"] == info["build_id"]
    assert KEY not in man0["params"] and "accumulate" not in man0["exposure"]
    assert AP.appearance_currency(store, WORLD, SESSION, man0)["current"] is True
    files0 = _appearance_files(store)
    # render
    page0, tex0 = _render(store)
    # finish: served as already built under the PRE-PATCH digest -- the new code's OFF key is 44fbd13's
    report = _finish(store, monkeypatch)
    assert report["appearance"]["state"] == AP.STATE_OK
    assert report["appearance"]["detail"] == AP.ALREADY_BUILT
    status = json.loads((AP.appearance_dir(store, WORLD, SESSION) / "status.json").read_text(encoding="utf-8"))
    assert status["params_digest"] == info["params_digest"]
    # not one byte of the saved appearance, nor of what a viewer is given, changed
    assert _appearance_files(store) == files0
    assert hashlib.sha256((AP.appearance_dir(store, WORLD, SESSION) / "manifest.json").read_bytes()
                          ).hexdigest() == info["manifest_sha256"]
    page1, tex1 = _render(store)
    assert page1 == page0 and tex1 == tex0


def _without(man, keys=VOLATILE) -> dict:
    return {k: v for k, v in man.items() if k not in keys}


def test_a_pre_patch_saved_world_rebuilt_with_the_switch_off_is_written_as_44fbd13_wrote_it(monkeypatch, tmp_path):
    """The downstream half of "OFF is unchanged": not only is the key the pre-patch key, a FORCED rebuild with the
    switch off (`world_appearance.py --force`, the finish's params) writes the pre-patch world again -- the whole
    manifest (params, exposure record, gains, slopes, tiers, ranks, chunk digests, provenance) and every published
    file -- except when and under which id it was built. Byte for byte needs the recording's torch/numpy/cv2."""
    _hide_cuda(monkeypatch)
    monkeypatch.delenv(ENV, raising=False)
    store, info = _saved_world(tmp_path)
    if not (_host_is_the_recordings(info) and _host_encoders_are_the_recordings(info)):
        pytest.skip("byte identity with the pre-patch world is pinned on the recording's torch/numpy/cv2/platform "
                    f"({info['torch']}, {info['numpy']}, {info['cv2']}, {info['machine']}, {info['system']})")
    man0 = AP.read_appearance_manifest(store, WORLD, SESSION)
    files0 = _appearance_files(store)
    _page0, tex0 = _render(store)
    r = AP.build_appearance(store, WORLD, SESSION, params=A.AppearanceParams(), force=True)
    assert r.state == AP.STATE_OK and r.detail != AP.ALREADY_BUILT
    man1 = AP.read_appearance_manifest(store, WORLD, SESSION)
    assert man1["build_id"] != man0["build_id"]
    assert _without(man1) == _without(man0)
    assert man1["seconds"].keys() == man0["seconds"].keys()
    # every file the manifest publishes (chunks, proxy) is the same file, byte for byte
    named0, named1 = AP.named_files(man0), AP.named_files(man1)
    assert named1 == named0
    files1 = _appearance_files(store)
    assert {n: files1[n] for n in named1} == {n: files0[n] for n in named0}
    _page1, tex1 = _render(store)
    assert tex1 == tex0


@pytest.mark.parametrize("cuda_decision", [False, True], ids=["cpu-fallback", "as-if-cuda"])
def test_a_pre_patch_saved_world_with_the_switch_on_is_rebuilt_by_the_finish_never_reused(
        monkeypatch, tmp_path, cuda_decision):
    _hide_cuda(monkeypatch)
    store, info = _saved_world(tmp_path)
    man0 = AP.read_appearance_manifest(store, WORLD, SESSION)
    _page0, tex0 = _render(store)
    op = DETERMINISTIC if cuda_decision else ATOMIC

    monkeypatch.setenv(ENV, "on")
    with monkeypatch.context() as mp:
        if cuda_decision:
            _as_if_cuda_host(mp)
        report = _finish(store, mp)
        assert report["appearance"]["state"] == AP.STATE_OK, report["appearance"]
        assert report["appearance"]["detail"] != AP.ALREADY_BUILT          # never the pre-patch build, reused
        man1 = AP.read_appearance_manifest(store, WORLD, SESSION)
        assert man1["build_id"] != man0["build_id"]
        assert man1["params_digest"] not in (man0["params_digest"], info["params_digest"])
        assert man1["params"][KEY] is True and man1["exposure"]["accumulate"] == op
        # the switch is the only params change: every other value is the pre-patch world's
        assert {k: v for k, v in man1["params"].items() if k != KEY} == man0["params"]
        # it renders
        page1, tex1 = _render(store)
        assert man1["build_id"] in page1 and set(tex1) == set(tex0)
        # a second ON finish on the same device is served from the ON build
        again = _finish(store, mp)
        assert again["appearance"]["detail"] == AP.ALREADY_BUILT
        assert AP.read_appearance_manifest(store, WORLD, SESSION)["build_id"] == man1["build_id"]
    if not cuda_decision:
        # On the CPU fallback ON runs the OFF op: the gains are the pre-patch gains (at manifest resolution;
        # every published byte too, on the recording's host)
        assert _gains(man1) == _gains(man0)
        if _host_is_the_recordings(info):
            assert [c["digest"] for c in man1["chunks"]] == [c["digest"] for c in man0["chunks"]]
            assert tex1 == tex0

    # OFF again: the ON build is not served as the OFF one; the key returns to the pre-patch digest
    monkeypatch.delenv(ENV)
    back = _finish(store, monkeypatch)
    assert back["appearance"]["state"] == AP.STATE_OK and back["appearance"]["detail"] != AP.ALREADY_BUILT
    man2 = AP.read_appearance_manifest(store, WORLD, SESSION)
    assert KEY not in man2["params"] and "accumulate" not in man2["exposure"]
    if _host_encoders_are_the_recordings(info):
        assert man2["params_digest"] == info["params_digest"]
    assert man2["params"] == man0["params"]
