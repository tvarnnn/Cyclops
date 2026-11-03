"""The live counter sampler is optional and cannot change build outcomes."""

import collections
import csv
import functools
import ctypes
import logging
import os
import struct
import subprocess
import sys
import time

import pytest

from tower.world_builder import live_counters as counters

ENV = "TOWER_WORLD_CPU_SAMPLER"
MODULE = "tower.world_builder.live_counters"
windows_only = pytest.mark.skipif(os.name != "nt", reason="Win32 readback")
CoreTimes = collections.namedtuple("CoreTimes", "user system idle")


class Store:
    def __init__(self, root):
        self.root = root

    def world_dir(self, world_id):
        return self.root / world_id


@functools.lru_cache(maxsize=None)
def _builder_module(name="lctr_builder_under_test"):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "world_build_session.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_rows(path, deadline_s=3.0, want=1):
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if path.exists() and len(path.read_text(encoding="utf-8").splitlines()) > want:
            break
        time.sleep(0.01)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# -- the switch and the builder's entry point --------------------------------------------------


def test_switch_off_does_not_start_or_write(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    store = Store(tmp_path)
    assert counters.start_if_on(store, "world", "session") is None
    assert not (tmp_path / "world").exists()


def test_builder_entry_point_off_imports_nothing(monkeypatch, tmp_path):
    builder = _builder_module()
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    assert builder._start_live_counters(Store(tmp_path), "world", "session") is None
    assert MODULE not in sys.modules
    assert not (tmp_path / "world").exists()


def test_builder_entry_point_survives_an_import_failure(monkeypatch, tmp_path, caplog):
    builder = _builder_module()
    monkeypatch.setenv(ENV, "on")
    monkeypatch.setitem(sys.modules, MODULE, None)  # `import` raises ImportError
    with caplog.at_level(logging.ERROR):
        assert builder._start_live_counters(Store(tmp_path), "world", "session") is None
    assert "live counters unavailable; the build goes on" in caplog.text


def test_builder_entry_point_on_writes_beside_solve_log(monkeypatch, tmp_path):
    builder = _builder_module()
    monkeypatch.setenv(ENV, "on")
    monkeypatch.setattr(counters, "PERIOD_SECONDS", 60.0)
    monkeypatch.setattr(counters, "_windows_api", lambda: None)
    monkeypatch.setattr(counters, "_psutil_processes", lambda: {os.getpid(): (1, "builder", 1.0, 0.0)})
    monkeypatch.setattr(counters.psutil, "cpu_times", lambda percpu=False: [CoreTimes(0, 0, 0)])
    sampler = builder._start_live_counters(Store(tmp_path), "world", "session")
    try:
        path = tmp_path / "world" / "solve" / "session" / counters.FILENAME
        deadline = time.monotonic() + 3
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert path.read_text(encoding="utf-8").startswith("t,clock,")
    finally:
        assert sampler.stop() is True
    assert sampler.closed


# -- the sampler ---------------------------------------------------------------------------------


class FakeMachine:
    """Four logical CPUs half busy; a builder, a child, a grandchild, an unrelated busy process,
    and a process that names the builder as parent but predates it (a reused pid)."""

    def __init__(self, builder):
        self.builder = builder
        self.calls = 0
        self.cores = 0

    def processes(self):
        self.calls += 1
        n = self.calls
        return {
            self.builder: (1, "builder", 100.0, 0.01 * n),
            22222: (self.builder, "solve", 101.0, 0.02 * n),
            22223: (22222, "torch", 102.0, 0.03 * n),
            33333: (1, "other", 50.0, 1.0 * n),
            44444: (self.builder, "reused", 99.0, 0.5 * n),
        }

    def cpu_times(self, percpu=False):
        self.cores += 1
        return [CoreTimes(self.cores, 0.0, self.cores)] * 4


def _fake(monkeypatch, machine):
    monkeypatch.setattr(counters, "_windows_api", lambda: None)
    monkeypatch.setattr(counters, "_psutil_processes", machine.processes)
    monkeypatch.setattr(counters.psutil, "cpu_times", machine.cpu_times)
    monkeypatch.setattr(counters.psutil, "cpu_count", lambda: 4)
    monkeypatch.setattr(counters.psutil, "virtual_memory", lambda: type("Mem", (), {"available": 2**30})())
    monkeypatch.setattr(counters, "_Gpu", lambda: (lambda: ("12", "1750")))


def test_sampler_writes_builder_tree_rows_and_stops(monkeypatch, tmp_path):
    machine = FakeMachine(builder=11111)
    _fake(monkeypatch, machine)
    path = tmp_path / counters.FILENAME
    sampler = counters.LiveCounters(path, pid=11111, period=0.01).start()
    try:
        rows = _read_rows(path, want=3)
    finally:
        assert sampler.stop() is True
    assert sampler.closed and not sampler._thread.is_alive()
    assert rows
    assert {row["name"] for row in rows} == {"builder", "solve", "torch"}  # not "reused"
    own = next(row for row in rows if row["pid"] == "11111")
    assert own["tree_procs"] == "3"
    assert own["gpu_util_pct"] == "12" and own["gpu_sm_mhz"] == "1750"
    assert own["logical_cpu_pct"] == "0:50.0;1:50.0;2:50.0;3:50.0"
    assert own["cpu_pct"] == "50.0"
    assert float(own["tree_cpu_pct"]) > 0
    assert "other:33333:" in own["top_other"] and "reused:44444:" in own["top_other"]
    assert own["mem_avail_gb"] == "1.0"
    assert float(own["sample_ms"]) >= 0 and float(own["sampler_cpu_s"]) >= 0
    # Off Windows the Win32 fields say "not observed", never a guess.
    assert own["throttle_control"] == own["priority_class"] == own["perf_pct"] == ""


def test_rows_per_sample_are_capped_busiest_first(monkeypatch, tmp_path):
    machine = FakeMachine(builder=11111)
    _fake(monkeypatch, machine)
    monkeypatch.setattr(counters, "MAX_TREE_ROWS", 2)
    path = tmp_path / counters.FILENAME
    sampler = counters.LiveCounters(path, pid=11111, period=0.01).start()
    try:
        rows = _read_rows(path, want=4)
    finally:
        sampler.stop()
    first = [row for row in rows if row["t"] == rows[0]["t"]]
    assert [row["name"] for row in first] == ["torch", "solve"]
    assert first[0]["tree_procs"] == "3"


def test_sampling_error_is_logged_swallowed_and_closes_the_csv(monkeypatch, tmp_path, caplog):
    machine = FakeMachine(builder=11111)
    _fake(monkeypatch, machine)
    calls = []

    def fails_on_second_call():
        calls.append(1)
        if len(calls) > 1:
            raise OSError("counter unavailable")
        return machine.processes()

    monkeypatch.setattr(counters, "_psutil_processes", fails_on_second_call)
    path = tmp_path / counters.FILENAME
    with caplog.at_level(logging.ERROR):
        sampler = counters.LiveCounters(path, pid=11111, period=0.01).start()
        sampler._thread.join(timeout=3)
    assert not sampler._thread.is_alive() and sampler.closed
    assert "live counters stopped after a sampling error" in caplog.text
    assert path.read_text(encoding="utf-8").splitlines() == [",".join(counters.FIELDS)]
    assert sampler.stop() is True  # stopping a dead sampler is harmless


def test_stop_is_bounded_and_prompt_with_a_long_period(monkeypatch, tmp_path):
    _fake(monkeypatch, FakeMachine(builder=11111))
    sampler = counters.LiveCounters(tmp_path / "x.csv", pid=11111, period=3600).start()
    started = time.monotonic()
    assert sampler.stop() is True
    assert time.monotonic() - started < 2
    assert sampler.closed


def test_a_failed_gpu_query_is_not_repeated(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired("nvidia-smi", kwargs.get("timeout"))

    monkeypatch.setattr(counters.subprocess, "run", run)
    gpu = counters._Gpu()
    assert gpu() == ("", "") and gpu() == ("", "")
    assert calls == [counters.GPU_TIMEOUT_SECONDS]


def test_core_classes_use_cpu_set_efficiency_not_model_numbers():
    class Kernel:
        def GetSystemCpuSetInformation(self, buf, length, size, process, flags):
            entries = b"".join(
                struct.pack("<IIIHBBBBBB", 20, 0, logical + 1, 0,
                            logical, logical, 0, 0, efficiency, 0)
                for logical, efficiency in ((3, 0), (7, 1)))
            size._obj.value = len(entries)
            if buf is not None:
                buf.raw = entries
                return True
            return False

    assert counters._core_classes((Kernel(), None)) == {(0, 3): "E", (0, 7): "P"}


def test_class_means_weight_the_clock_by_where_work_ran():
    classes = {(0, 0): "P", (0, 1): "P", (0, 2): "E", (0, 3): "E"}
    busy = {0: 100.0, 1: 0.0, 2: 50.0, 3: 50.0}
    perf = {0: 210.0, 1: 90.0, 2: 160.0, 3: 120.0}
    assert counters._class_means(busy, perf, classes) == {
        "p_busy_pct": 50.0, "p_perf_pct": 210.0, "e_busy_pct": 50.0, "e_perf_pct": 140.0}
    assert counters._logical({"0,1": 5.0, "0,_Total": 7.0, "_Total": 7.0, "1,0": 9.0}) == {1: 5.0}


# -- Windows readback (the Tower) ----------------------------------------------------------------


@windows_only
def test_windows_readback_of_throttling_priority_and_process_list():
    api = counters._windows_api()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel.SetProcessInformation.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                             ctypes.c_ulong)
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    # Explicit classes: a BelowNormal parent (a scheduled task) would hand its class down.
    eco = subprocess.Popen(sleeper, creationflags=subprocess.NORMAL_PRIORITY_CLASS)
    low = subprocess.Popen(sleeper, creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    try:
        handle = kernel.OpenProcess(0x0200, False, eco.pid)  # PROCESS_SET_INFORMATION
        state = counters._Throttle(1, 1, 1)  # EXECUTION_SPEED on: EcoQoS
        assert kernel.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state))
        kernel.CloseHandle(handle)
        assert counters._windows_process(api, eco.pid) == (1, 1, "normal")
        assert counters._windows_process(api, low.pid) == (0, 0, "below_normal")
        started = time.perf_counter()
        procs = counters._NtProcesses()()
        assert time.perf_counter() - started < 1.0
        assert procs[eco.pid][0] == os.getpid() and procs[eco.pid][1].lower() == "python.exe"
        assert counters._tree(procs, os.getpid()) >= {os.getpid(), eco.pid, low.pid}
        classes = counters._core_classes(api)
        assert not classes or set(classes.values()) == {"P", "E"}
        pdh = counters._Pdh()
        time.sleep(0.3)
        perf = counters._logical(pdh.read().get("perf"))
        pdh.close()
        assert sorted(perf) == list(range(os.cpu_count()))
    finally:
        for child in (eco, low):
            child.kill()
            child.wait()
