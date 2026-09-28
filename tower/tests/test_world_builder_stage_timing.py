"""The finish timer is a bounded sibling of the authoritative journal."""

import json
import logging
import shutil
import sys
import time
from types import SimpleNamespace

import pytest

from tower import config
from tower.config import world_stage_timing_setting
from tower.world_builder import stage_timing as timing
from tests.test_world_builder_solve_masks import (  # noqa: F401
    N, FakeColmap, StubDetector, _masked, colmap, session,
)

ENV = "TOWER_WORLD_STAGE_TIMING"


class Store:
    def __init__(self, root):
        self.root = root

    def session_dir(self, world_id, session_id):
        return self.root / world_id / session_id


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("  ", False),
    ("1", True), ("true", True), ("yes", True), ("on", True), (" ON ", True),
    ("0", False), ("false", False), ("no", False), ("off", False), (" OFF ", False),
])
def test_parser(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert world_stage_timing_setting() is expected


@pytest.mark.parametrize("value", ["maybe", "2", "enabled", "onn"])
def test_bad_flag_is_off_and_logged(monkeypatch, caplog, value):
    monkeypatch.setattr(config, "_world_stage_timing_warned_values", set())
    monkeypatch.setenv(ENV, value)
    with caplog.at_level(logging.WARNING, logger="tower.config"):
        assert world_stage_timing_setting() is False
    assert len([r for r in caplog.records if ENV in r.getMessage()]) == 1


def test_bad_flag_warns_once_per_value(monkeypatch, caplog):
    monkeypatch.setattr(config, "_world_stage_timing_warned_values", set())
    monkeypatch.setenv(ENV, "a-new-invalid-value")
    with caplog.at_level(logging.WARNING, logger="tower.config"):
        for _ in range(5):
            assert world_stage_timing_setting() is False
    assert len([r for r in caplog.records if ENV in r.getMessage()]) == 1


def _run(root, monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, flag)
    store = Store(root)
    session = store.session_dir("w", "s")
    session.mkdir(parents=True)
    journal = session / "events.jsonl"
    journal.write_bytes(b'{"event_id":1,"kind":"keyframe"}\n')

    def finish():
        timing.cache("mask", True)
        timing.cache("mask", False)
        timing.cache("depth", False)
        timing.cache("pair", True)
        for name in ("solution.json", "components.json", "surface.json",
                     "appearance.json", "areas.json"):
            (session / name).write_bytes(b'{"payload":[1,2,3]}\n')
        return {"payload": [1, 2, 3]}

    result = timing.measure("surface", store, "w", "s", finish)
    files = {p.name: p.read_bytes() for p in session.iterdir()}
    return result, files


def test_off_is_byte_identical_to_uninstrumented_and_on_changes_only_timing(tmp_path, monkeypatch):
    baseline, original = _run(tmp_path / "base", monkeypatch, None)
    off, off_files = _run(tmp_path / "off", monkeypatch, "off")
    on, on_files = _run(tmp_path / "on", monkeypatch, "on")
    assert baseline == off == on
    assert original == off_files
    assert set(on_files) - set(original) == {timing.FILENAME}
    assert {name: data for name, data in on_files.items() if name != timing.FILENAME} == original
    assert on_files["events.jsonl"] == original["events.jsonl"]
    doc = json.loads(on_files[timing.FILENAME])
    assert doc["schema"] == timing.SCHEMA
    assert len(doc["finishes"]) == 1
    assert set(doc["finishes"][0]["stages"]) == {"surface"}
    record, = doc["finishes"][0]["stages"]["surface"]
    assert set(record) == {"wall_ms", "gpu_wait_ms", "gpu_wait_source", "cache", "status"}
    assert record["wall_ms"] >= 0
    assert record["gpu_wait_ms"] == 0
    assert record["gpu_wait_source"] == "no_blocking_gpu_lock"
    assert record["cache"] == {"mask": {"hits": 1, "misses": 1},
                               "depth": {"hits": 0, "misses": 1},
                               "pair": {"hits": 1, "misses": 0}}


def test_records_are_bounded_independent_of_input_count(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    store = Store(tmp_path)
    # `admission` (the retrieval admission, `retrieval_admission.py`; C23-IMPL): 2, as `p4` -- the early publish of
    # draw 0 and the full consensus. It raised the bound from 90 to 96 records.
    expected = {"solve": 1, "solve_draw": 7, "gate": 16, "p4": 2, "admission": 2,
                "regate": 1, "surface": 1, "appearance": 1, "areas": 1}
    assert timing.FINISH_LIMIT == 3
    assert timing.LIMITS == expected
    assert sum(expected.values()) * timing.FINISH_LIMIT == 96  # the doc's and handoff's bound (was 90)
    for stage, limit in expected.items():
        store.session_dir("w", stage).mkdir(parents=True)
        for _ in range(20):
            def work():
                for _ in range(100):
                    timing.cache("mask", False)
                return 7
            assert timing.measure(stage, store, "w", stage, work) == 7
        doc = json.loads((store.session_dir("w", stage) / timing.FILENAME).read_bytes())
        records = doc["finishes"][-1]["stages"][stage]
        assert len(records) == (1 if stage in ("solve", "regate", "areas") else limit)
        assert all(r["cache"]["mask"]["misses"] == 100 for r in records)
        assert len(doc["finishes"]) <= 3
        assert all("finish_id" in finish for finish in doc["finishes"])


def test_nested_area_surface_is_counted_in_area_only(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    store = Store(tmp_path)
    store.session_dir("w", "s").mkdir(parents=True)

    def area():
        return timing.measure("surface", store, "w", "s", lambda: timing.cache("depth", True))

    timing.measure("areas", store, "w", "s", area)
    doc = json.loads((store.session_dir("w", "s") / timing.FILENAME).read_bytes())
    assert set(doc["finishes"][0]["stages"]) == {"areas"}
    assert doc["finishes"][0]["stages"]["areas"][0]["cache"]["depth"]["hits"] == 1


def test_real_final_stage_calls_keep_result_and_stage_records_identical(tmp_path, monkeypatch):
    from scripts import world_build_session as build
    from tower.world_builder import appearance_pipeline, surface_pipeline

    monkeypatch.setattr(build, "_solution_gated", lambda *a: False)
    seen = []

    def surface(*args, **kwargs):
        seen.append(("surface", kwargs["force"]))
        return SimpleNamespace(state="ok", as_dict=lambda: {"mesh": "same"})

    def appearance(*args, **kwargs):
        seen.append(("appearance", True))
        return SimpleNamespace(state="ok", as_dict=lambda: {"texture": "same"})

    monkeypatch.setattr(surface_pipeline, "surfacify", surface)
    monkeypatch.setattr(appearance_pipeline, "build_appearance", appearance)

    def run(flag, root):
        monkeypatch.setenv(ENV, flag)
        store = Store(root)
        session = store.session_dir("w", "s")
        session.mkdir(parents=True)
        (session / "events.jsonl").write_bytes(b'{"event_id":1}\n')
        records = []
        result = build.final_surface_stages(
            store, "w", "s", solved=True, appearance=True, prune_depth_work=False,
            should_stop=lambda: False,
            record=lambda *args, **kwargs: records.append((args, kwargs)))
        files = {p.name: p.read_bytes() for p in session.iterdir()}
        return result, records, files

    off_result, off_records, off_files = run("off", tmp_path / "off")
    on_result, on_records, on_files = run("on", tmp_path / "on")
    assert off_result == on_result
    assert off_records == on_records
    assert seen == [("surface", True), ("appearance", True)] * 2
    assert {k: v for k, v in on_files.items() if k != timing.FILENAME} == off_files
    doc = json.loads(on_files[timing.FILENAME])
    assert set(doc["finishes"][0]["stages"]) == {"surface", "appearance"}


def test_nested_draw_write_waits_until_enclosing_solve_returns(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    store = Store(tmp_path)
    store.session_dir("w", "s").mkdir(parents=True)
    writes = []
    real_write = timing._write

    def spy(*args):
        writes.append(args[3].stage)
        real_write(*args)

    monkeypatch.setattr(timing, "_write", spy)

    @timing.timed("solve_draw", store_arg=False)
    def draw():
        return 3

    @timing.timed("solve")
    def solve(store, world_id, session_id):
        assert draw() == 3
        assert writes == []  # solver's own timing/payload is captured before diagnostic I/O
        return {"solution": 3}

    assert solve(store, "w", "s") == {"solution": 3}
    assert writes == ["solve_draw", "solve"]


def test_stage_exception_and_failed_diagnostic_write_preserve_outcome(tmp_path, monkeypatch):
    from tower import storage

    monkeypatch.setenv(ENV, "on")
    store = Store(tmp_path)
    store.session_dir("w", "s").mkdir(parents=True)

    @timing.timed("gate")
    def raising(store, world_id, session_id):
        raise KeyError("stage failure")

    with pytest.raises(KeyError, match="stage failure"):
        raising(store, "w", "s")
    doc = json.loads((store.session_dir("w", "s") / timing.FILENAME).read_bytes())
    assert doc["finishes"][0]["stages"]["gate"][0]["status"] == "raised"

    def failed_write(*args, **kwargs):
        raise PermissionError("held destination")

    monkeypatch.setattr(storage, "replace_with_retry", failed_write)
    assert timing.measure("surface", store, "w", "s", lambda: "stage result") == "stage result"
    with pytest.raises(KeyError, match="stage failure"):
        raising(store, "w", "s")
    assert not list(store.session_dir("w", "s").glob("*.tmp"))


def test_measure_of_a_raising_stage_reraises_and_records_raised(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    store = Store(tmp_path)
    store.session_dir("w", "s").mkdir(parents=True)

    def raiser():
        raise KeyError("surface failure")

    with pytest.raises(KeyError, match="surface failure"):
        timing.measure("surface", store, "w", "s", raiser)
    doc = json.loads((store.session_dir("w", "s") / timing.FILENAME).read_bytes())
    record, = doc["finishes"][-1]["stages"]["surface"]
    assert record["status"] == "raised"


def test_deferred_function_that_raises_passes_it_through_and_flushes_children(
        tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    store = Store(tmp_path)
    store.session_dir("w", "s").mkdir(parents=True)
    path = store.session_dir("w", "s") / timing.FILENAME

    def raiser():
        raise KeyError("appearance failure")

    @timing.defer
    def room(store, world_id, session_id):  # no @timed of its own, like final_surface_stages
        assert timing.measure("surface", store, world_id, session_id, lambda: "mesh") == "mesh"
        assert not path.exists()  # buffered until the room returns
        timing.measure("appearance", store, world_id, session_id, raiser)

    with pytest.raises(KeyError, match="appearance failure"):
        room(store, "w", "s")
    doc = json.loads(path.read_bytes())
    assert len(doc["finishes"]) == 1
    stages = doc["finishes"][0]["stages"]
    assert [r["status"] for r in stages["surface"]] == ["ok"]
    assert [r["status"] for r in stages["appearance"]] == ["raised"]


def test_keyword_call_and_store_without_session_dir_keep_stage_result(monkeypatch):
    monkeypatch.setenv(ENV, "on")
    calls = []

    @timing.timed("gate")
    def gate(store, world_id, session_id):
        calls.append(1)
        return "ok"

    assert gate(store=object(), world_id="w", session_id="s") == "ok"
    assert gate(object(), "w", "s") == "ok"
    assert calls == [1, 1]


def test_final_only_skips_background_solve(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    store = Store(tmp_path)
    store.session_dir("w", "s").mkdir(parents=True)

    @timing.timed("solve", final_only=True)
    def solve(store, world_id, session_id, *, final):
        return final

    assert solve(store, "w", "s", final=False) is False
    assert not (store.session_dir("w", "s") / timing.FILENAME).exists()
    assert solve(store, "w", "s", final=True) is True
    assert (store.session_dir("w", "s") / timing.FILENAME).exists()


def test_finish_identity_and_regate_draw_indices(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    store = Store(tmp_path)
    store.session_dir("w", "s").mkdir(parents=True)

    @timing.timed("solve_draw", store_arg=False)
    def draw(*, seed):
        return seed

    @timing.timed("solve")
    def solve(store, world_id, session_id):
        return [draw(seed=i + 10) for i in range(3)]

    @timing.timed("regate")
    def regate(store, world_id, session_id):
        return [draw(seed=i + 20) for i in range(2)]

    assert solve(store, "w", "s") == [10, 11, 12]
    assert solve(store, "w", "s") == [10, 11, 12]
    assert regate(store, "w", "s") == [20, 21]
    doc = json.loads((store.session_dir("w", "s") / timing.FILENAME).read_bytes())
    assert len(doc["finishes"]) == 3
    assert len({f["finish_id"] for f in doc["finishes"]}) == 3
    assert [r["draw_index"] for r in doc["finishes"][0]["stages"]["solve_draw"]] == [0, 1, 2]
    assert [r["seed"] for r in doc["finishes"][-1]["stages"]["solve_draw"]] == [20, 21]


def test_runless_stage_never_joins_a_finish_begun_before_this_process(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    store = Store(tmp_path)
    store.session_dir("w", "s").mkdir(parents=True)
    path = store.session_dir("w", "s") / timing.FILENAME
    old = {"finish_id": "1-0-1", "started_at": 0, "pid": 1, "context": "solve",
           "stages": {"surface": [{"wall_ms": 5.0, "gpu_wait_ms": 0.0,
                                   "gpu_wait_source": "no_blocking_gpu_lock",
                                   "cache": {name: {"hits": 0, "misses": 0}
                                             for name in timing.CACHES},
                                   "status": "raised"}]}}
    path.write_text(json.dumps({"schema": timing.SCHEMA, "finishes": [old]}), encoding="utf-8")
    old_bytes = json.dumps(old, sort_keys=True)

    assert timing.measure("surface", store, "w", "s", lambda: "mesh") == "mesh"
    doc = json.loads(path.read_bytes())
    assert len(doc["finishes"]) == 2
    assert json.dumps(doc["finishes"][0], sort_keys=True) == old_bytes
    new = doc["finishes"][1]
    assert new["finish_id"] != old["finish_id"] and new["context"] == "surface"
    assert new["started_at"] >= timing._PROCESS_STARTED
    assert [r["status"] for r in new["stages"]["surface"]] == ["ok"]

    # A finish this process began is still joined: appearance lands beside that surface.
    assert timing.measure("appearance", store, "w", "s", lambda: "texture") == "texture"
    doc = json.loads(path.read_bytes())
    assert len(doc["finishes"]) == 2
    assert json.dumps(doc["finishes"][0], sort_keys=True) == old_bytes
    assert set(doc["finishes"][1]["stages"]) == {"surface", "appearance"}


def test_cold_and_warm_mask_counts_are_initial_lookups_only(session, colmap, monkeypatch):
    from tower.world_builder.transients import TransientParams

    monkeypatch.setenv(ENV, "on")
    stub = StubDetector()
    path = session.store.session_dir(session.world_id, session.session_id) / timing.FILENAME
    count = N * len(TransientParams().components)

    cold = _masked(session, stub)
    cold_doc = json.loads(path.read_bytes())
    cold_cache = cold_doc["finishes"][-1]["stages"]["solve"][0]["cache"]["mask"]
    assert cold["transients"]["computed"] == N
    assert cold_cache == {"hits": 0, "misses": count}

    warm = _masked(session, stub)
    warm_doc = json.loads(path.read_bytes())
    warm_cache = warm_doc["finishes"][-1]["stages"]["solve"][0]["cache"]["mask"]
    assert warm["transients"]["computed"] == 0
    assert warm_cache == {"hits": count, "misses": 0}
    assert warm_doc["finishes"][0]["finish_id"] != warm_doc["finishes"][1]["finish_id"]


def test_final_solve_product_bytes_are_identical_on_and_off(
        tmp_path, monkeypatch, session, colmap):
    from tower.world_builder import global_solve as GS
    from tower.world_builder.store import WorldStore
    from tests.test_world_builder_coherence_publish import _stub_gate

    calls = []
    _stub_gate(monkeypatch, calls)  # deterministic CPU gate; product writers remain real
    monkeypatch.setattr(time, "time", lambda: 1234.5)
    monkeypatch.setattr(time, "perf_counter", lambda: 20.0)

    def run(flag):
        root = tmp_path / f"copy-{flag}"
        shutil.copytree(session.store.root, root)
        store = WorldStore(root)
        monkeypatch.setitem(sys.modules, "pycolmap", FakeColmap())
        monkeypatch.setenv(ENV, flag)
        result = GS.solve(store, session.world_id, session.session_id,
                          final=True, gate=True, consensus=1, masks=False,
                          seed=None, loop_detection=False)
        workspace = GS.workspace_for(store, session.world_id, session.session_id)
        payload = {p.name: p.read_bytes() for p in workspace.root.iterdir()
                   if p.is_file() and p.name in {
                       "solution.json", "components.json", "points.bin", "observations.bin"}}
        return result, payload, store.session_dir(session.world_id, session.session_id)

    off_result, off_payload, off_session = run("off")
    on_result, on_payload, on_session = run("on")
    assert off_result["solved"] and on_result["solved"]
    assert off_payload and "solution.json" in off_payload and "components.json" in off_payload
    assert on_payload == off_payload
    assert (off_session / "events.jsonl").read_bytes() == (on_session / "events.jsonl").read_bytes()
    assert not (off_session / timing.FILENAME).exists()
    assert (on_session / timing.FILENAME).exists()
