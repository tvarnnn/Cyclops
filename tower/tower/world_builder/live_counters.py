"""Best-effort post-Stop counters for the builder process tree.

CPU percentages use BENCH's machine-wide scale: 100% means all logical CPUs.
An empty Windows field means the host/API did not provide that observation.
"""

import csv
import ctypes
import ctypes.wintypes as wt
import logging
import os
import struct
import subprocess
import threading
import time

import psutil

from tower.config import world_cpu_sampler_setting

logger = logging.getLogger(__name__)
FILENAME = "live_counters.csv"
PERIOD_SECONDS = 10.0
FIELDS = ("t", "clock", "cpu_pct", "tree_cpu_pct", "tree_procs", "top_other",
          "logical_cpu_pct", "logical_core_classes", "mem_avail_gb",
          "gpu_util_pct", "gpu_sm_mhz",
          "console_foreground", "logonui_present",
          "pid", "name", "process_cpu_pct", "logical_cpu", "core_class",
          "throttle_control", "throttle_state", "priority_class")


def _windows_api():
    if os.name != "nt":
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    user = ctypes.WinDLL("user32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wt.DWORD, wt.BOOL, wt.DWORD)
    kernel.OpenProcess.restype = wt.HANDLE
    kernel.CloseHandle.argtypes = (wt.HANDLE,)
    kernel.GetProcessInformation.argtypes = (wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD)
    kernel.GetPriorityClass.argtypes = (wt.HANDLE,)
    kernel.GetPriorityClass.restype = wt.DWORD
    kernel.GetConsoleWindow.restype = wt.HWND
    user.GetForegroundWindow.restype = wt.HWND
    kernel.GetSystemCpuSetInformation.argtypes = (
        ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), wt.HANDLE, wt.DWORD)
    return kernel, user


def _core_classes(api):
    """Use the host's CPU-set EfficiencyClass, never a model-specific core list."""
    if api is None:
        return {}
    kernel, _ = api
    size = wt.DWORD()
    kernel.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    if not size.value or size.value > 1024 * 1024:
        return {}
    buf = ctypes.create_string_buffer(size.value)
    if not kernel.GetSystemCpuSetInformation(buf, size.value, ctypes.byref(size), None, 0):
        return {}
    raw = buf.raw
    classes = {}
    offset = 0
    while offset + 20 <= size.value:
        entry_size, kind = struct.unpack_from("<II", raw, offset)
        if entry_size < 20 or offset + entry_size > size.value:
            break
        if kind == 0:  # CpuSetInformation
            _, group, logical, _, _, _, efficiency, _ = struct.unpack_from(
                "<IHBBBBBB", raw, offset + 8)
            classes[(group, logical)] = efficiency
        offset += entry_size
    levels = set(classes.values())
    if len(levels) < 2:
        return {}
    return {key: "P" if value == max(levels) else "E" for key, value in classes.items()}


def _windows_process(api, pid):
    if api is None:
        return "", "", ""
    kernel, _ = api
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return "", "", ""
    class Throttle(ctypes.Structure):
        _fields_ = (("version", wt.DWORD), ("control", wt.DWORD), ("state", wt.DWORD))
    try:
        throttle = Throttle(1, 0, 0)
        ok = kernel.GetProcessInformation(handle, 4, ctypes.byref(throttle),
                                          ctypes.sizeof(throttle))
        priority = kernel.GetPriorityClass(handle)
        return (throttle.control if ok else "", throttle.state if ok else "",
                priority or "")
    finally:
        kernel.CloseHandle(handle)


def _console_state(api, names):
    if api is None:
        return "", ""
    kernel, user = api
    console = kernel.GetConsoleWindow()
    foreground = user.GetForegroundWindow()
    return (int(bool(console and foreground and console == foreground)) if console else "",
            int("logonui.exe" in names))


def _gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=2, check=False)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.splitlines()[0].split(",")
            if len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "", ""


class LiveCounters:
    def __init__(self, path, *, pid=None, period=None):
        self.path = path
        self.pid = pid or os.getpid()
        self.period = PERIOD_SECONDS if period is None else period
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="world-live-counters", daemon=True)

    def start(self):
        try:
            self._thread.start()
        except Exception:  # noqa: BLE001 -- diagnostics must not affect the build
            logger.exception("live counters could not start")
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        try:
            self._sample_loop()
        except Exception:  # noqa: BLE001 -- one failed sample ends diagnostics only
            logger.exception("live counters stopped after a sampling error")

    def _sample_loop(self):
        api = _windows_api()
        classes = _core_classes(api)
        ncpu = psutil.cpu_count() or 1
        previous = {}
        for proc in psutil.process_iter():
            try:
                previous[(proc.pid, proc.create_time())] = sum(proc.cpu_times()[:2])
            except (psutil.Error, OSError):
                continue
        psutil.cpu_percent(None)
        psutil.cpu_percent(None, percpu=True)
        last = time.monotonic()
        with open(self.path, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            output.flush()
            while not self._stop.wait(self.period):
                now = time.monotonic()
                elapsed = max(now - last, 1e-6)
                try:
                    root = psutil.Process(self.pid)
                    tree = {root.pid, *(child.pid for child in root.children(recursive=True))}
                except psutil.NoSuchProcess:
                    return
                current, build, others, names = {}, [], [], set()
                for proc in psutil.process_iter(["name"]):
                    try:
                        key = (proc.pid, proc.create_time())
                        total = sum(proc.cpu_times()[:2])
                        name = proc.info.get("name") or "?"
                    except (psutil.Error, OSError):
                        continue
                    names.add(name.lower())
                    current[key] = total
                    used = total - previous[key] if key in previous else (
                        total if key[1] >= time.time() - elapsed else 0.0)
                    share = round(max(0.0, used) / elapsed / ncpu * 100, 2)
                    if proc.pid in tree:
                        build.append((proc, name, share))
                    elif proc.pid != 0 and share >= 0.5:
                        others.append((share, name, proc.pid))
                previous, last = current, now
                gpu_util, gpu_clock = _gpu()
                foreground, logonui = _console_state(api, names)
                common = {"t": round(time.time(), 3), "clock": time.strftime("%H:%M:%S"),
                          "cpu_pct": psutil.cpu_percent(None),
                          "logical_cpu_pct": ";".join(
                              f"{i}:{pct:.1f}" for i, pct in enumerate(
                                  psutil.cpu_percent(None, percpu=True))),
                          "logical_core_classes": ";".join(
                              f"{logical}:{kind}" for (group, logical), kind in sorted(classes.items())
                              if group == 0),
                          "mem_avail_gb": round(psutil.virtual_memory().available / 2**30, 2),
                          "tree_cpu_pct": round(sum(item[2] for item in build), 2),
                          "tree_procs": len(build),
                          "top_other": ";".join(f"{name}:{pid}:{share:.1f}"
                                                for share, name, pid in sorted(others, reverse=True)[:3]),
                          "gpu_util_pct": gpu_util, "gpu_sm_mhz": gpu_clock,
                          "console_foreground": foreground, "logonui_present": logonui}
                if self._stop.is_set():
                    return
                for proc, name, share in build:
                    try:
                        logical = proc.cpu_num() if hasattr(proc, "cpu_num") else ""
                    except (psutil.Error, OSError):
                        logical = ""
                    control, state, priority = _windows_process(api, proc.pid)
                    row = {**common, "pid": proc.pid, "name": name,
                           "process_cpu_pct": share, "logical_cpu": logical,
                           "core_class": classes.get((0, logical), ""),
                           "throttle_control": control, "throttle_state": state,
                           "priority_class": priority}
                    writer.writerow(row)
                output.flush()


def start_if_on(store, world_id, session_id):
    """Start only for an enabled live builder; never make sampling a build dependency."""
    if not world_cpu_sampler_setting():
        return None
    try:
        path = store.world_dir(world_id) / "solve" / session_id / FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        return LiveCounters(path).start()
    except Exception:  # noqa: BLE001 -- diagnostics must not affect the build
        logger.exception("live counters could not be prepared")
        return None
