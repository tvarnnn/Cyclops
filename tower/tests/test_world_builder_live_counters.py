"""The live counter sampler is optional and cannot change build outcomes."""

import csv
import logging
import os
import struct
import time

from tower.world_builder import live_counters as counters


class Store:
    def __init__(self, root):
        self.root = root

    def world_dir(self, world_id):
        return self.root / world_id


def test_switch_off_does_not_start_or_write(monkeypatch, tmp_path):
    monkeypatch.delenv("TOWER_WORLD_CPU_SAMPLER", raising=False)
    store = Store(tmp_path)
    assert counters.start_if_on(store, "world", "session") is None
    assert not (tmp_path / "world").exists()


def test_sampler_writes_builder_rows_and_stops(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_CPU_SAMPLER", "on")
    monkeypatch.setattr(counters, "PERIOD_SECONDS", 0.01)
    monkeypatch.setattr(counters, "_gpu", lambda: ("12", "1750"))
    # The restricted Mac test runner cannot enumerate host PIDs. Exercise the
    # same delta accounting with a builder, descendant, and outside process.
    class Process:
        def __init__(self, pid, name):
            self.pid = pid
            self.info = {"name": name}
            self.total = 0.0

        def create_time(self):
            return 100.0

        def cpu_times(self):
            self.total += 0.02
            return self.total, 0.0

        def children(self, recursive=False):
            return [child] if self is builder else []

        def cpu_num(self):
            return 0

    builder = Process(os.getpid(), "builder")
    child = Process(22222, "solve")
    other = Process(33333, "other")
    procs = (builder, child, other)
    monkeypatch.setattr(counters.psutil, "process_iter", lambda *args: iter(procs))
    monkeypatch.setattr(counters.psutil, "Process", lambda pid: builder)
    monkeypatch.setattr(counters.psutil, "cpu_count", lambda: 4)
    monkeypatch.setattr(counters.psutil, "cpu_percent", lambda _, percpu=False: [25.0] * 4 if percpu else 25.0)
    monkeypatch.setattr(counters.psutil, "virtual_memory", lambda: type("Mem", (), {"available": 2**30})())
    sampler = counters.start_if_on(Store(tmp_path), "world", "session")
    path = tmp_path / "world" / "solve" / "session" / counters.FILENAME
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if path.exists() and len(path.read_text().splitlines()) > 1:
                break
            time.sleep(0.01)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        own = next(row for row in rows if int(row["pid"]) == os.getpid())
        assert own["gpu_util_pct"] == "12"
        assert own["gpu_sm_mhz"] == "1750"
        assert float(own["cpu_pct"]) >= 0
        assert float(own["process_cpu_pct"]) >= 0
        assert own["tree_procs"] == "2"
        assert own["logical_cpu_pct"] == "0:25.0;1:25.0;2:25.0;3:25.0"
        assert own["mem_avail_gb"] == "1.0"
        assert "other:33333:" in own["top_other"]
        assert {row["name"] for row in rows} == {"builder", "solve"}
    finally:
        sampler.stop()
        sampler._thread.join(timeout=2)
    assert not sampler._thread.is_alive()


def test_sampling_error_is_logged_and_swallowed(monkeypatch, tmp_path, caplog):
    def fail(self):
        raise OSError("counter unavailable")

    monkeypatch.setattr(counters.LiveCounters, "_sample_loop", fail)
    with caplog.at_level(logging.ERROR):
        sampler = counters.LiveCounters(tmp_path / "never.csv").start()
        sampler._thread.join(timeout=2)
    assert not sampler._thread.is_alive()
    assert "live counters stopped after a sampling error" in caplog.text
    assert not (tmp_path / "never.csv").exists()


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
