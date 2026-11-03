"""Best-effort post-Stop counters for the builder process tree (`TOWER_WORLD_CPU_SAMPLER`).

One daemon thread writes one row per build process every `PERIOD_SECONDS` to
`solve/<session>/live_counters.csv`. Nothing the build reads; a failure is logged and ends the
sampling, never the build. CPU percentages use BENCH's machine-wide scale: 100% means every
logical CPU busy. An empty field means the host or API did not provide that observation.

What Windows gives, and what it does not:
- The CPU clock: `\\Processor Information(*)\\% Processor Performance` (PDH), averaged over each
  period, 100 = nominal clock, above 100 = turbo; `% Performance Limit` (_Total) is the share of
  the nominal clock the firmware, power or thermal limits allow. psutil's `cpu_freq` only ever
  reports the nominal clock on Windows, so it is not used.
- P/E placement: there is no cheap per-process or per-thread "which core did it run on" (psutil
  has no `cpu_num` on Windows; only ETW sees it). The evidence is per-core busy % and clock with
  the host's own P/E map (`GetSystemCpuSetInformation` EfficiencyClass), summed by class.
- Power throttling: `GetProcessInformation(ProcessPowerThrottling)` reports only an EXPLICIT
  policy: (1, 1) EcoQoS, (1, 0) HighQoS opt-out, (0, 0) none, i.e. the system's heuristics decide.
  What the heuristics decided is not readable; a heuristically throttled build shows as busy
  E-cores and a low clock while its explicit state reads (0, 0).
"""

import atexit
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
# Bounds on one sample's work. The machine-wide process list is ONE system call on Windows;
# the per-process Win32 queries cover at most `MAX_TREE_ROWS` build processes (busiest first).
MAX_TREE_ROWS = 16
GPU_TIMEOUT_SECONDS = 2.0
STOP_TIMEOUT_SECONDS = 5.0
FIELDS = ("t", "clock", "sample_ms", "sampler_cpu_s",
          "cpu_pct", "tree_cpu_pct", "other_cpu_pct", "tree_procs", "top_other",
          "perf_pct", "perf_limit_pct", "p_busy_pct", "e_busy_pct", "p_perf_pct", "e_perf_pct",
          "logical_cpu_pct", "logical_perf_pct", "logical_core_classes", "mem_avail_gb",
          "gpu_util_pct", "gpu_sm_mhz",
          "foreground_proc", "console_foreground", "logonui_present",
          "pid", "name", "process_cpu_pct", "throttle_control", "throttle_state",
          "priority_class")
PRIORITY_CLASSES = {0x40: "idle", 0x4000: "below_normal", 0x20: "normal",
                    0x8000: "above_normal", 0x80: "high", 0x100: "realtime"}
_EPOCH_AS_FILETIME = 116444736000000000  # 1970-01-01 in 100 ns ticks since 1601-01-01


class _Throttle(ctypes.Structure):
    """PROCESS_POWER_THROTTLING_STATE."""
    _fields_ = (("version", wt.DWORD), ("control", wt.DWORD), ("state", wt.DWORD))


class _PdhValue(ctypes.Structure):
    """PDH_FMT_COUNTERVALUE read as a double (ctypes pads `status` to the union's alignment)."""
    _fields_ = (("status", wt.DWORD), ("value", ctypes.c_double))


class _PdhItem(ctypes.Structure):
    """PDH_FMT_COUNTERVALUE_ITEM_W."""
    _fields_ = (("name", wt.LPWSTR), ("value", _PdhValue))


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
    kernel.GetSystemCpuSetInformation.argtypes = (
        ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), wt.HANDLE, wt.DWORD)
    user.GetForegroundWindow.restype = wt.HWND
    user.GetWindowThreadProcessId.argtypes = (wt.HWND, ctypes.POINTER(wt.DWORD))
    user.GetWindowThreadProcessId.restype = wt.DWORD
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


class _NtProcesses:
    """Every process's (ppid, name, create time, CPU seconds) from ONE NtQuerySystemInformation.

    psutil opens each of the machine's ~400 processes in turn; this is one system call into a
    reused buffer, whatever the process count, and needs no access to any other process."""

    MAX_BYTES = 32 * 1024 * 1024

    def __init__(self):
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise OSError("the SYSTEM_PROCESS_INFORMATION offsets below are 64-bit")
        self._query = ctypes.WinDLL("ntdll").NtQuerySystemInformation
        self._query.argtypes = (ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
                                ctypes.POINTER(ctypes.c_ulong))
        self._query.restype = ctypes.c_ulong
        self._buf = ctypes.create_string_buffer(1024 * 1024)

    def __call__(self):
        length = ctypes.c_ulong()
        for _ in range(3):
            status = self._query(5, self._buf, len(self._buf), ctypes.byref(length))
            if status == 0:  # SystemProcessInformation = 5
                break
            if status != 0xC0000004 or length.value > self.MAX_BYTES:  # INFO_LENGTH_MISMATCH
                raise OSError(f"NtQuerySystemInformation status 0x{status:08x}")
            self._buf = ctypes.create_string_buffer(min(length.value + (256 << 10), self.MAX_BYTES))
        else:
            raise OSError("the process list kept outgrowing its buffer")
        buf, end, base = self._buf, length.value, ctypes.addressof(self._buf)
        procs = {}
        offset = 0
        # SYSTEM_PROCESS_INFORMATION (x64): NextEntryOffset @0, CreateTime/UserTime/KernelTime
        # @32/40/48, ImageName (Length @56, Buffer @64), UniqueProcessId @80, parent @88.
        while offset + 104 <= end:
            step, = struct.unpack_from("<I", buf, offset)
            created, user, kernel = struct.unpack_from("<qqq", buf, offset + 32)
            name_bytes, = struct.unpack_from("<H", buf, offset + 56)
            name_ptr, = struct.unpack_from("<Q", buf, offset + 64)
            pid, ppid = struct.unpack_from("<QQ", buf, offset + 80)
            start = name_ptr - base
            name = (bytes(buf[start:start + name_bytes]).decode("utf-16-le", "replace")
                    if name_ptr and 0 <= start and start + name_bytes <= end else "")
            procs[pid] = (ppid, name or "?", (created - _EPOCH_AS_FILETIME) / 1e7,
                          (user + kernel) / 1e7)
            if not step:
                break
            offset += step
        return procs


def _psutil_processes():
    procs = {}
    for proc in psutil.process_iter(["ppid", "name", "create_time", "cpu_times"]):
        info = proc.info
        times, created = info.get("cpu_times"), info.get("create_time")
        if times is None or created is None:
            continue
        procs[proc.pid] = (info.get("ppid") or 0, info.get("name") or "?", created,
                           times.user + times.system)
    return procs


class _Pdh:
    """Per-logical-CPU `% Processor Performance` and the _Total `% Performance Limit`."""

    COUNTERS = {"perf": r"\Processor Information(*)\% Processor Performance",
                "limit": r"\Processor Information(*)\% Performance Limit"}

    def __init__(self):
        pdh = ctypes.WinDLL("pdh")
        pdh.PdhOpenQueryW.argtypes = (wt.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wt.HANDLE))
        pdh.PdhAddEnglishCounterW.argtypes = (wt.HANDLE, wt.LPCWSTR, ctypes.c_size_t,
                                              ctypes.POINTER(wt.HANDLE))
        pdh.PdhCollectQueryData.argtypes = (wt.HANDLE,)
        pdh.PdhGetFormattedCounterArrayW.argtypes = (
            wt.HANDLE, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.POINTER(wt.DWORD), ctypes.c_void_p)
        pdh.PdhCloseQuery.argtypes = (wt.HANDLE,)
        for function in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData",
                         "PdhGetFormattedCounterArrayW", "PdhCloseQuery"):
            getattr(pdh, function).restype = wt.DWORD
        self._pdh, self._query, self._counters = pdh, wt.HANDLE(), {}
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)):
            raise OSError("PdhOpenQueryW failed")
        for key, path in self.COUNTERS.items():
            counter = wt.HANDLE()
            if not pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(counter)):
                self._counters[key] = counter
        pdh.PdhCollectQueryData(self._query)  # a rate needs two collections

    def read(self):
        """{"perf"|"limit": {instance: value}} over the time since the previous read."""
        if self._pdh.PdhCollectQueryData(self._query):
            return {}
        out = {}
        fmt = 0x200 | 0x8000  # PDH_FMT_DOUBLE | PDH_FMT_NOCAP100: turbo reads above 100
        for key, counter in self._counters.items():
            size, count = wt.DWORD(0), wt.DWORD(0)
            if self._pdh.PdhGetFormattedCounterArrayW(counter, fmt, ctypes.byref(size),
                                                      ctypes.byref(count), None) != 0x800007D2:
                continue  # not PDH_MORE_DATA: no data this period
            buf = ctypes.create_string_buffer(size.value)
            if self._pdh.PdhGetFormattedCounterArrayW(counter, fmt, ctypes.byref(size),
                                                      ctypes.byref(count), buf):
                continue
            items = (_PdhItem * count.value).from_buffer(buf)
            out[key] = {item.name: item.value.value for item in items
                        if item.value.status in (0, 1)}  # VALID_DATA, NEW_DATA
        return out

    def close(self):
        self._pdh.PdhCloseQuery(self._query)


def _logical(instances):
    """{logical index: value} for group 0's `"0,<n>"` PDH instances."""
    values = {}
    for name, value in (instances or {}).items():
        group, _, index = name.partition(",")
        if group == "0" and index.isdigit():
            values[int(index)] = value
    return values


def _windows_process(api, pid):
    if api is None:
        return "", "", ""
    kernel, _ = api
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return "", "", ""
    try:
        throttle = _Throttle(1, 0, 0)  # PROCESS_POWER_THROTTLING_CURRENT_VERSION
        ok = kernel.GetProcessInformation(handle, 4, ctypes.byref(throttle),  # PowerThrottling
                                          ctypes.sizeof(throttle))
        priority = kernel.GetPriorityClass(handle)
        return (throttle.control if ok else "", throttle.state if ok else "",
                PRIORITY_CLASSES.get(priority, priority or ""))
    finally:
        kernel.CloseHandle(handle)


def _desktop_state(api, procs):
    """(foreground window's process name, our console in front?, LogonUI running?)."""
    logonui = int(any(name.lower() == "logonui.exe" for _, name, _, _ in procs.values()))
    if api is None:
        return "", "", logonui
    kernel, user = api
    foreground = user.GetForegroundWindow()
    owner = wt.DWORD(0)
    if foreground:
        user.GetWindowThreadProcessId(foreground, ctypes.byref(owner))
    name = procs[owner.value][1] if owner.value and owner.value in procs else ""
    console = kernel.GetConsoleWindow()
    return name, (int(bool(foreground) and console == foreground) if console else ""), logonui


class _Gpu:
    """nvidia-smi, bounded by a timeout; one failure or timeout turns it off for the run."""

    def __init__(self):
        self.on = True

    def __call__(self):
        if not self.on:
            return "", ""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=GPU_TIMEOUT_SECONDS, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            parts = result.stdout.splitlines()[0].split(",") if result.stdout.strip() else []
            if result.returncode == 0 and len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()
        except (OSError, subprocess.SubprocessError):
            pass
        self.on = False
        logger.warning("live counters: nvidia-smi failed or timed out; GPU columns stay empty")
        return "", ""


def _tree(procs, root):
    """`root` and its descendants by parent pid; a child must not predate its parent (pid reuse)."""
    children = {}
    for pid, (ppid, _, created, _) in procs.items():
        if pid != ppid:
            children.setdefault(ppid, []).append((pid, created))
    tree, todo = set(), [root] if root in procs else []
    while todo:
        pid = todo.pop()
        tree.add(pid)
        todo.extend(child for child, created in children.get(pid, ())
                    if child not in tree and created >= procs[pid][2])
    return tree


def _class_means(busy, perf, classes):
    """{P,E}: mean busy %, and the clock weighted by each core's busy share (where work ran)."""
    out = {}
    for kind in ("P", "E"):
        cores = [index for (group, index), value in classes.items() if group == 0 and value == kind]
        load = [busy[i] for i in cores if i in busy]
        out[f"{kind.lower()}_busy_pct"] = round(sum(load) / len(load), 1) if load else ""
        weight = sum(busy[i] for i in cores if i in busy and i in perf)
        out[f"{kind.lower()}_perf_pct"] = (
            round(sum(busy[i] * perf[i] for i in cores if i in busy and i in perf) / weight, 1)
            if weight > 0 else "")
    return out


class LiveCounters:
    def __init__(self, path, *, pid=None, period=None):
        self.path = path
        self.pid = pid or os.getpid()
        self.period = PERIOD_SECONDS if period is None else period
        self.closed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="world-live-counters", daemon=True)

    def start(self):
        try:
            self._thread.start()
            atexit.register(self.stop)  # an exception leaving the builder still closes the CSV
        except Exception:  # noqa: BLE001 -- diagnostics must not affect the build
            logger.exception("live counters could not start")
        return self

    def stop(self, timeout=STOP_TIMEOUT_SECONDS):
        """Ask the thread to end and wait (bounded) for it to close the CSV. Never raises."""
        try:
            self._stop.set()
            atexit.unregister(self.stop)
            if self._thread.is_alive() and self._thread is not threading.current_thread():
                self._thread.join(timeout)
        except Exception:  # noqa: BLE001 -- diagnostics must not affect the build
            logger.exception("live counters did not stop cleanly")
        return self.closed or not self._thread.is_alive()

    def _run(self):
        try:
            self._sample_loop()
        except Exception:  # noqa: BLE001 -- one failed sample ends diagnostics only
            logger.exception("live counters stopped after a sampling error")
        finally:
            self.closed = True

    def _sample_loop(self):
        api = _windows_api()
        classes = _core_classes(api)
        snapshot = _NtProcesses() if api is not None else _psutil_processes
        pdh = None
        if api is not None:
            try:
                pdh = _Pdh()
            except Exception:  # noqa: BLE001 -- the clock columns stay empty, the rest goes on
                logger.warning("live counters: no processor performance counters", exc_info=True)
        try:
            self._write_samples(api, classes, snapshot, pdh, _Gpu())
        finally:
            if pdh is not None:
                pdh.close()

    def _write_samples(self, api, classes, snapshot, pdh, gpu):
        ncpu = psutil.cpu_count() or 1
        procs = snapshot()
        previous = {(pid, item[2]): item[3] for pid, item in procs.items()}
        cores = psutil.cpu_times(percpu=True)
        last = time.monotonic()
        with open(self.path, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            output.flush()
            while not self._stop.wait(self.period):
                begun = time.perf_counter()
                now = time.monotonic()
                elapsed = max(now - last, 1e-6)
                procs = snapshot()
                if self.pid not in procs:
                    return
                tree = _tree(procs, self.pid)
                current, build, others = {}, [], []
                for pid, (_, name, created, total) in procs.items():
                    key = (pid, created)
                    current[key] = total
                    used = total - previous[key] if key in previous else (
                        total if created >= time.time() - elapsed else 0.0)
                    share = round(max(0.0, used) / elapsed / ncpu * 100, 2)
                    if pid in tree:
                        build.append((share, pid, name))
                    elif pid != 0 and share >= 0.5:  # pid 0 is the idle process
                        others.append((share, name, pid))
                previous, last = current, now
                # Per-core busy % exactly as psutil.cpu_percent computes it, from our own
                # previous times (psutil's module-global baseline is shared with any caller).
                cores_now = psutil.cpu_times(percpu=True)
                busy = {}
                for index, (before, after) in enumerate(zip(cores, cores_now)):
                    spent = sum(after) - sum(before)
                    busy[index] = (round(100.0 * (1 - (after.idle - before.idle) / spent), 1)
                                   if spent > 0 else 0.0)
                cores = cores_now
                counters = pdh.read() if pdh is not None else {}
                perf = _logical(counters.get("perf"))
                total_perf = (counters.get("perf") or {}).get("_Total")
                limit = (counters.get("limit") or {}).get("_Total")
                cpu = round(sum(busy.values()) / len(busy), 1) if busy else ""
                tree_cpu = round(sum(item[0] for item in build), 2)
                gpu_util, gpu_clock = gpu()
                foreground_proc, console_front, logonui = _desktop_state(api, procs)
                common = {"t": round(time.time(), 3), "clock": time.strftime("%H:%M:%S"),
                          "cpu_pct": cpu, "tree_cpu_pct": tree_cpu,
                          "other_cpu_pct": round(max(0.0, cpu - tree_cpu), 1) if busy else "",
                          "tree_procs": len(build),
                          "top_other": ";".join(f"{name}:{pid}:{share:.1f}"
                                                for share, name, pid in sorted(others, reverse=True)[:3]),
                          "perf_pct": "" if total_perf is None else round(total_perf, 1),
                          "perf_limit_pct": "" if limit is None else round(limit, 1),
                          **_class_means(busy, perf, classes),
                          "logical_cpu_pct": ";".join(f"{i}:{pct:.1f}" for i, pct in sorted(busy.items())),
                          "logical_perf_pct": ";".join(f"{i}:{pct:.0f}" for i, pct in sorted(perf.items())),
                          "logical_core_classes": ";".join(
                              f"{logical}:{kind}" for (group, logical), kind in sorted(classes.items())
                              if group == 0),
                          "mem_avail_gb": round(psutil.virtual_memory().available / 2**30, 2),
                          "gpu_util_pct": gpu_util, "gpu_sm_mhz": gpu_clock,
                          "foreground_proc": foreground_proc, "console_foreground": console_front,
                          "logonui_present": logonui}
                rows = []
                for share, pid, name in sorted(build, reverse=True)[:MAX_TREE_ROWS]:
                    control, state, priority = _windows_process(api, pid)
                    rows.append({**common, "pid": pid, "name": name, "process_cpu_pct": share,
                                 "throttle_control": control, "throttle_state": state,
                                 "priority_class": priority})
                cost = {"sample_ms": round(1000 * (time.perf_counter() - begun), 1),
                        "sampler_cpu_s": round(time.thread_time(), 3)}
                for row in rows:
                    writer.writerow({**row, **cost})
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
