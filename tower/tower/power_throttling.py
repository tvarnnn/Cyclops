"""Opt a World Builder process out of Windows power throttling (EcoQoS). Default off.

Switch: `TOWER_WORLD_NO_POWER_THROTTLING` (`tower.config.world_no_power_throttling_setting`).

WHY THIS EXISTS

The live build of walk 10.2 ran its CPU-bound steps about twice as slowly as a
quiet rerun of the same world (pack 179.6 s against 84.4 s, GLOMAP map 361
against 187 s, encode 56.5 against 28.6 s) while the GPU-bound steps ran at
about the same speed. Windows' power throttling is ONE mechanism that produces
that pattern on the Tower's hybrid CPU (8 performance cores, 12 efficiency
cores): a process marked EcoQoS is moved toward the efficient cores and a lower
clock. It is a reproduced mechanism, not the proven cause of the 10.2 gap
(ECOQOS lane, 2026-11-03):

* EcoQoS set on a process slowed the pack's L0 -> L1 decimation 1.51x and the
  10.2 appearance stage 1.26x (282 / 289 s against 226 s), less than the live
  gap;
* a Tower-style launch (Normal priority, its own minimized console) was NOT
  throttled by Windows in that test, so the live trigger was not reproduced;
* with this switch on, the same throttled appearance stage ran in 220 / 229 s,
  the unthrottled speed.

So the switch removes a measured slowdown WHEN throttling occurs, and costs
nothing when it does not.

WHAT IT DOES

`SetProcessInformation(ProcessPowerThrottling)` on the CURRENT process with
`ControlMask = EXECUTION_SPEED` and `StateMask = 0`: the documented "HighQoS"
opt-out, "never throttle this process's execution speed". Microsoft documents
it as overriding the system's own background heuristics; the ECOQOS proof
measured it overriding an explicit EcoQoS mark. Each process calls it for
itself in its `main()`, right after logging is set up and before any work,
because the setting is not inherited: a child of an EcoQoS parent starts with
no explicit state (measured on the Tower). With the switch on, the walk-time
processes (the builder's follow loop, the background solves, the BelowNormal
live surface child) are opted out during the walk too; the Tower server
process is not.

What it deliberately leaves alone:

* Priority. The builder inherits the Tower's class (Normal, from an
  interactive console) and its solve children inherit the builder's; the live
  surface child and the masks-at-Stop child are deliberately BelowNormal, and
  stay so. Priority only matters under contention; throttling is about the
  clock and the core class.
* `IGNORE_TIMER_RESOLUTION`. That mechanism lets Windows ignore a hidden
  process's timer-resolution requests. No World Builder throughput depends on
  timer resolution: the builder's polls are 0.25 s sleeps, and CPython 3.11+
  sleeps on a high-resolution waitable timer that ignores `timeBeginPeriod`.

OUTPUTS

It computes nothing differently: the one call changes where and how fast the
CPU runs the same work. Single-thread stages give the same bytes at any speed
(the decimation was identical across every condition tested). The appearance
stage is not bit-reproducible at main, and its outputs with the switch on
differed from a main run no more than two main runs differ from each other.
In a live walk, choices that depend on timing (when a background solve
launches, how far the live surface or the masks-at-Stop child gets) shift with
speed, as they do with any speedup. It is Windows-only and a no-op elsewhere;
a failure is reported and the process carries on exactly as it would have
with the switch off.
"""

from __future__ import annotations

import logging
import sys

from tower.config import world_no_power_throttling_setting

logger = logging.getLogger(__name__)

# PROCESS_INFORMATION_CLASS.ProcessPowerThrottling (processthreadsapi.h).
PROCESS_POWER_THROTTLING = 4
PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1


def _report(level: int, message: str) -> None:
    """One line in THIS process's own log.

    Through the logger when logging is configured (the builder, the
    masks-at-Stop child). The solve, surface, appearance and finisher scripts
    configure none, so a logger.info would be dropped there; their stderr is
    already their log (`solve.log`, `surface.log`, the Tower console), so the
    line goes there directly. Never stdout: a final solve's stdout is the
    summary its builder parses.
    """
    if logging.getLogger().handlers:
        logger.log(level, message)
    else:
        print(message, file=sys.stderr, flush=True)


def _kernel32():
    """A private kernel32 handle, so the argtypes set here never leak into
    `ctypes.windll.kernel32`, which other modules share."""
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    kernel32.SetProcessInformation.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    return kernel32


def _state_structure():
    """`PROCESS_POWER_THROTTLING_STATE`: three 32-bit ULONGs (12 bytes). Spelled `c_uint32`, not
    `wintypes.ULONG`, so the layout is the Win32 one on every platform the tests run on."""
    import ctypes  # noqa: PLC0415

    class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):  # noqa: N801 -- the Win32 name
        _fields_ = [
            ("Version", ctypes.c_uint32),
            ("ControlMask", ctypes.c_uint32),
            ("StateMask", ctypes.c_uint32),
        ]

    return PROCESS_POWER_THROTTLING_STATE


def opt_out_of_power_throttling(role: str, *, kernel32=None) -> bool:
    """HighQoS for THIS process: never throttle its execution speed.

    Returns True when Windows accepted it. Never raises: a refusal or an
    exception is logged and returns False, and the caller carries on.
    `kernel32` is injectable for tests.
    """
    try:
        import ctypes  # noqa: PLC0415

        if kernel32 is None:
            kernel32 = _kernel32()
        state = _state_structure()(
            PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED,  # ControlMask: we decide execution speed
            0,                                          # StateMask: and the decision is "never throttle"
        )
        ok = kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(), PROCESS_POWER_THROTTLING,
            ctypes.byref(state), ctypes.sizeof(state),
        )
        if not ok:
            error_code = ctypes.get_last_error()
            _report(logging.WARNING,
                    f"[Tower][WorldBuilder] {role}: could not opt out of power throttling "
                    f"(SetProcessInformation error {error_code}); running as before")
            return False
    except Exception as error:  # noqa: BLE001 -- a speed hint must never stop a build
        try:
            _report(logging.WARNING,
                    f"[Tower][WorldBuilder] {role}: could not opt out of power throttling "
                    f"({type(error).__name__}: {error}); running as before")
        except Exception:  # noqa: BLE001 -- not even the report may raise
            pass
        return False
    try:
        _report(logging.INFO, f"[Tower][WorldBuilder] {role}: opted out of power throttling (EcoQoS)")
    except Exception:  # noqa: BLE001 -- see above
        pass
    return True


def apply_world_build_power_policy(role: str, *, setting=None, platform: str | None = None,
                                   opt_out=None) -> bool:
    """Call in a World Builder process's `main()`: after its logging is set up, before any work.

    Off (the default): returns False and makes no call at all. On, on
    Windows: `opt_out_of_power_throttling(role)`. On, elsewhere: False, no
    call. Returns whether the process is now opted out. Never raises.
    """
    try:
        enabled = world_no_power_throttling_setting() if setting is None else bool(setting)
    except Exception:  # noqa: BLE001 -- see opt_out_of_power_throttling
        return False
    if not enabled:
        return False
    if (sys.platform if platform is None else platform) != "win32":
        return False
    return (opt_out or opt_out_of_power_throttling)(role)
