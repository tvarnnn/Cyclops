import functools
import os
import tempfile

import pytest

from tower.world_builder.records import Keyframe, Session, World
from tower.world_builder.store import WorldStore, compute_input_digest

# Module-scope clear, in addition to the autouse fixture below: pytest
# imports conftest.py before any test module, but an ambient
# TOWER_CV_EXPERIMENT=depth in the operator's shell (e.g. left over from
# manually testing the depth experiment per the README) would otherwise
# still be set when tower.main is first imported during test collection.
# tower/main.py's last line is a module-level `app = create_app()` that
# runs at import time -- before any fixture, even an autouse one, gets a
# chance to run, since fixtures only wrap test *execution*, not import.
# Left uncleared, that import-time create_app() would build a real
# DepthEstimationModule and attempt a real torch import / MiDaS network
# fetch during collection, regardless of which tests are selected.
os.environ.pop("TOWER_CV_EXPERIMENT", None)
os.environ.pop("TOWER_CV_DEVICE", None)


@pytest.fixture(autouse=True)
def _no_transient_detector_models(monkeypatch):
    """The World Builder's transient detector (tower/world_builder/transients.py)
    would load -- and on a machine without them, DOWNLOAD -- three checkpoints
    (~2 GB) the first time a surface or appearance build asks. No test may do
    that. Off by environment, so a CLI subprocess inherits it too; tests of the
    detector's behaviour pass a stub backend explicitly."""
    monkeypatch.setenv("TOWER_WORLD_TRANSIENTS", "off")


@pytest.fixture(autouse=True, scope="session")
def _live_depth_never_loads_a_model():
    """Every builder session starts a `LiveDepthWorker`, and its first trusted,
    calibrated keyframe would load MoGe on the GPU. No test may do that. Unless
    a test passes its own, the worker gets a backend factory that reports the
    model unavailable (the worker then disables itself, as on a machine without
    the weights) and a CPU host's VRAM probe. `close()` also waits for the
    thread, so the worker's last `align.json` write lands before a test reads
    the world rather than racing it. A builder SUBPROCESS is out of reach of
    this fixture: such a test passes `no_depth_model_env` to it. Session scope,
    so a module-scoped fixture that builds a world is covered too."""
    import threading

    from tower.world_builder import live_depth
    from tower.world_builder.dense import DepthModelUnavailable

    def unavailable():
        raise DepthModelUnavailable("tests never load a depth model")

    class Worker(live_depth.LiveDepthWorker):
        def __init__(self, *args, backend_factory=None, vram_probe=None, **kwargs):
            super().__init__(*args, backend_factory=backend_factory or unavailable,
                             vram_probe=vram_probe or (lambda: None), **kwargs)

        def close(self):
            super().close()
            if self.thread.is_alive() and self.thread is not threading.current_thread():
                self.thread.join(timeout=10)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(live_depth, "LiveDepthWorker", Worker)
        yield


@pytest.fixture
def live_depth_on_its_own_clock(monkeypatch):
    """The engine hands the live depth worker its monotonic clock, and the
    worker reads it twice per keyframe. With the real clocks that costs nothing;
    with a test's injected counting clock every read moves each stamp that
    follows. A test pinning those stamps, or counting the reads, uses this:
    the worker still runs, timed on `time.monotonic` as in the product."""
    import time

    from tower.world_builder import live_depth

    worker = live_depth.LiveDepthWorker

    def on_its_own_clock(*args, monotonic=None, **kwargs):
        return worker(*args, monotonic=time.monotonic, **kwargs)

    monkeypatch.setattr(live_depth, "LiveDepthWorker", on_its_own_clock)


@pytest.fixture
def final_scale_guard_bypassed(monkeypatch):
    """Every final publication runs the final scale guard, which withholds a
    candidate with no supported room (`final_scale_guard.apply`) and reads the
    session record and the world's live depth cache. A test of ANOTHER stage
    whose fakes cannot carry that -- a fake mapper that maps nothing, a store
    with no session or `world_dir` -- uses this: the publish step then skips the
    guard, as it does for a background solve. What the guard does with such a
    candidate (a withhold) is pinned in
    `test_world_builder_final_scale_guard_seam.py`.

    A test that uses this must not assert what a final publication writes (the
    components record, the guard's audit, their order with the solution): that
    output is the guard's, and only a test that runs the guard can pin it."""
    from tower.world_builder import coherence_publish

    monkeypatch.setattr(coherence_publish, "_final_scale_guard", lambda *a, **k: None)


def without_live_depth(files: dict) -> dict:
    """`files` (keyed by world-relative posix path) without the live depth
    worker's sidecar, for a golden recorded before the worker existed."""
    return {k: v for k, v in files.items() if "/live_depth/" not in k}


def no_depth_model_env(env=None) -> dict:
    """`env` (default: this process's) for a builder subprocess that must not
    load a depth model: the Hugging Face hub offline over an empty cache, so
    the live depth worker's MoGe reports itself unavailable, and no CUDA
    device, so neither it nor the worker's VRAM probe can touch the GPU."""
    out = dict(os.environ if env is None else env)
    out.update(HF_HUB_OFFLINE="1", HF_HUB_CACHE=_empty_hub_cache(), CUDA_VISIBLE_DEVICES="-1")
    return out


@functools.cache
def _empty_hub_cache() -> str:
    return tempfile.mkdtemp(prefix="tower-tests-empty-hub-")


@pytest.fixture(autouse=True)
def _clear_cv_experiment_env(monkeypatch):
    monkeypatch.delenv("TOWER_CV_EXPERIMENT", raising=False)
    monkeypatch.delenv("TOWER_CV_DEVICE", raising=False)


@pytest.fixture(autouse=True)
def _isolate_the_default_observation_root(monkeypatch, tmp_path):
    """No test reads the object memory of whoever owns this checkout.

    `config.DEFAULT_OBSERVATION_ROOT` is `TOWER_ROOT / "data" /
    "object_memory"`, resolved from `config.py`'s own file location, and
    that is CORRECT FOR THE PRODUCT. It is the default that was
    deliberately reversed on 2026-08-26, after a real walk wrote 64
    observations that every HTTP request answered 404 about: "a default
    that hides data from its owner while still storing it protects
    nobody." Nothing here undoes that. An unconfigured Tower still
    serves; this only changes WHICH directory an unconfigured Tower
    under test serves out of.

    Because for a TEST the same default means something else entirely.
    A test that unsets TOWER_OBSERVATION_ROOT does not get a Tower that
    has observed nothing -- it gets the developer's own accumulated
    store, and asserts against it. That is a privacy defect first (the
    suite reads a real wearer's history off disk) and a flake second.

    It is invisible in CI, which is why it survived: `data/` is
    gitignored, so a fresh clone has an empty default and every
    "observation_count == 0" passes for the wrong reason. On a machine
    that has actually been walked around, the same assertion reads
    `assert 64 == 0` and looks like a mystery.

    A temp directory, not an empty string and not None: "unconfigured"
    must keep meaning "the default root, whatever it is", so the routes
    still answer 200 with an empty listing and the 404 path stays
    reachable only by switching the cartridge off. The directory is not
    created -- a Tower that has never observed anything has no store
    directory either, and the read routes have to cope with that.

    Enforced by two tests in `test_config.py`, which fail if this
    fixture is removed.
    """
    from tower import config

    monkeypatch.setattr(
        config, "DEFAULT_OBSERVATION_ROOT", str(tmp_path / "default_object_memory")
    )


@pytest.fixture(autouse=True)
def _isolate_the_default_document_root(monkeypatch, tmp_path):
    """No test reads the documents of whoever owns this checkout.

    Every word of `_isolate_the_default_observation_root` above applies
    here unchanged, and this fixture exists because the reasoning was
    not carried across when Document Memory gained a default root.

    Before 2026-09-07 `config.document_root` had no default: a stock
    Tower declared the cartridge unavailable and told the wearer to set
    `TOWER_DOCUMENT_ROOT`. Document Memory V1 fixed that with
    `DEFAULT_DOCUMENT_ROOT = TOWER_ROOT / "data" / "document_memory"`,
    which is the same correct product decision Object Memory made a
    fortnight earlier -- and it inherits the same test hazard, which
    Object Memory's fixture had already been written to close.

    The hazard is worse for this cartridge, not better. What Document
    Memory persists is the TEXT OF PAGES a person held in front of their
    face: an unisolated suite would read a real wearer's correspondence
    off disk and assert against it. `data/` is gitignored, so on a fresh
    clone the default is empty and every "no documents recorded" passes
    for the wrong reason; on a machine that has actually read a page the
    same assertion fails and looks like a mystery.

    A temp directory, for the same reason as above: "unconfigured" must
    keep meaning "the default root, whatever it is", so the routes still
    answer with an empty library and the 404 path stays reachable only by
    switching the cartridge off with `TOWER_DOCUMENT_ENABLED`. The
    directory is not created -- a Tower that has never read a page has no
    store either, and the read routes have to cope with that.
    """
    from tower import config

    monkeypatch.setattr(
        config, "DEFAULT_DOCUMENT_ROOT", str(tmp_path / "default_document_memory")
    )


def _keyframe(session_id: str, seq: int, segment_index: int) -> Keyframe:
    return Keyframe(
        keyframe_id=f"{session_id}:{seq:08d}",
        session_id=session_id,
        source_seq=seq,
        received_at=1000.0 + seq,
        image_relpath=f"images/{seq:08d}.jpg",
        width=360,
        height=640,
        byte_count=1234,
        segment_index=segment_index,
    )


@pytest.fixture
def keyframe_factory():
    """`_keyframe` as a fixture, because conftest is not an importable module."""
    return _keyframe


@pytest.fixture
def derived_world(tmp_path):
    """A two-segment world with a derived tree that verifies.

    Segment 0 resolves (an anchor plus a solved pose, with points).
    Segment 1 does not (an anchor plus a refused pose, no points) -- the
    32-of-51 case on the real walk.
    """
    store = WorldStore(tmp_path)
    world_id = "w0"
    session_id = "s0"
    store.write_world(World(world_id=world_id, created_at=1.0, updated_at=2.0,
                            session_ids=(session_id,)))
    store.write_session(Session(session_id=session_id, world_id=world_id,
                                started_at=1.0, ended_at=2.0))

    layout = [(1, 0), (2, 0), (3, 1), (4, 1)]
    for seq, segment_index in layout:
        store.append_keyframe(world_id, _keyframe(session_id, seq, segment_index))

    poses = [
        {"keyframe_id": f"{session_id}:00000001", "segment_index": 0,
         "status": "anchor", "degeneracy": "",
         "rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
        {"keyframe_id": f"{session_id}:00000002", "segment_index": 0,
         "status": "solved", "degeneracy": "",
         "rotation": [0.0, 1.0, 0.0, 0.0], "translation": [1.0, 2.0, 3.0]},
        {"keyframe_id": f"{session_id}:00000003", "segment_index": 1,
         "status": "anchor", "degeneracy": "",
         "rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
        {"keyframe_id": f"{session_id}:00000004", "segment_index": 1,
         "status": "unavailable", "degeneracy": "low_parallax",
         "rotation": None, "translation": None},
    ]
    points = [
        {"segment_index": 0, "xyz": [1.0, 2.0, 3.0]},
        {"segment_index": 0, "xyz": [-1.0, 0.0, 5.0]},
    ]
    digest = compute_input_digest(store.read_keyframes(world_id, session_id))
    manifest = {
        "schema_version": 1, "input_digest": digest, "built_at": 3.0,
        "backend_id": "classical-sfm", "session_id": session_id,
        "keyframes": 4, "poses_solved": 1, "poses_refused": 1,
        "poses_anchor": 2, "poses_positioned": 2, "points": 2,
        "segments": 2, "scale_state": "unknown", "global_solve": None,
    }
    store.write_derived(world_id, session_id, poses=poses, points=points,
                        manifest=manifest)
    return store, world_id, session_id
