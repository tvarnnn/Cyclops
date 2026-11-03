"""Opt a World Builder process out of Windows power throttling (EcoQoS). Default off.

Switch: `TOWER_WORLD_NO_POWER_THROTTLING` (`tower.config.world_no_power_throttling_setting`).

WHY THIS EXISTS

The live build of walk 10.2 ran every CPU-bound step about twice as slowly as
a quiet rerun of the same world (pack 179.6 s against 84.4 s, GLOMAP map 361
against 187 s, encode 56.5 against 28.6 s) while the GPU-bound steps ran at
about the same speed. On the Tower's hybrid CPU (8 performance cores and 12
efficiency cores) that fingerprint is Windows' power throttling: a process the
system considers background work, or one marked EcoQoS, is given the
efficient cores and the most efficient clock. The world build is exactly the
work that runs while nobody is looking at the Tower -- its console is behind
another window, or the screen is off, because the owner is walking with the
glasses.

WHAT IT DOES

`SetProcessInformation(ProcessPowerThrottling)` on the CURRENT process with
`ControlMask = EXECUTION_SPEED` and `StateMask = 0`: the documented "HighQoS"
opt-out, "never throttle this process's execution speed", which takes
precedence over the system's own background heuristics. Each process calls it
for itself at the top of its `main()`, because the setting is not inherited:
a child of an opted-out (or EcoQoS) parent starts with no explicit state
(measured on the Tower, ECOQOS lane 2026-11-03).

What it deliberately leaves alone:

* Priority. The builder inherits the Tower's class (Normal, from an
  interactive console) and its solve children inherit the builder's; the live
  surface child is deliberately BelowNormal so the walk's solve and frame
  ingestion win the CPU, and stays so. Priority only matters under
  contention; throttling is about the clock and the core class.
* `IGNORE_TIMER_RESOLUTION`. That mechanism lets Windows ignore a hidden
  process's timer-resolution requests. No World Builder throughput depends on
  timer resolution: the builder's polls are 0.25 s sleeps, and CPython 3.11+
  sleeps on a high-resolution waitable timer that ignores `timeBeginPeriod`.

It never changes what is computed, so every output is byte-identical with the
switch on or off. It is Windows-only and a no-op elsewhere; a failure is
logged and the process carries on exactly as it would have with the switch
off.
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
            logger.warning(
                "[Tower][WorldBuilder] %s: could not opt out of power throttling "
                "(SetProcessInformation error %s); running as before",
                role, ctypes.get_last_error(),
            )
            return False
    except Exception as error:  # noqa: BLE001 -- a speed hint must never stop a build
        logger.warning(
            "[Tower][WorldBuilder] %s: could not opt out of power throttling (%s: %s); "
            "running as before", role, type(error).__name__, error,
        )
        return False
    logger.info("[Tower][WorldBuilder] %s: opted out of power throttling (EcoQoS)", role)
    return True


def apply_world_build_power_policy(role: str, *, setting=None, platform: str | None = None,
                                   opt_out=None) -> bool:
    """Call at the top of a World Builder process's `main()`.

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
