"""`TOWER_WORLD_NO_POWER_THROTTLING`: World Builder processes opt out of Windows EcoQoS. Default off.

The call happens when the switch is on (on Windows), never when it is off, and a
failure is logged, never raised. Every World Builder entry point makes the call
right after parsing its arguments, before any work.
"""

import ctypes
import importlib.util
import logging
import pathlib
import subprocess
import sys

import pytest

from tower import power_throttling as pt
from tower.config import WORLD_NO_POWER_THROTTLING_ENV, world_no_power_throttling_setting

TOWER_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Spy:
    def __init__(self, result=True):
        self.calls = []
        self.result = result

    def __call__(self, role):
        self.calls.append(role)
        return self.result


# -- the switch ---------------------------------------------------------------

def test_the_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv(WORLD_NO_POWER_THROTTLING_ENV, raising=False)
    assert world_no_power_throttling_setting() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "ON"])
def test_the_switch_reads_true(monkeypatch, value):
    monkeypatch.setenv(WORLD_NO_POWER_THROTTLING_ENV, value)
    assert world_no_power_throttling_setting() is True


@pytest.mark.parametrize("value", ["", " ", "0", "off", "false", "no", "garbage"])
def test_the_switch_reads_false(monkeypatch, value):
    monkeypatch.setenv(WORLD_NO_POWER_THROTTLING_ENV, value)
    assert world_no_power_throttling_setting() is False


# -- the policy: on calls, off never calls ----------------------------------------

def test_off_makes_no_call(monkeypatch):
    monkeypatch.delenv(WORLD_NO_POWER_THROTTLING_ENV, raising=False)
    spy = _Spy()
    assert pt.apply_world_build_power_policy("builder", platform="win32", opt_out=spy) is False
    assert spy.calls == []


@pytest.mark.parametrize("value", ["off", "0", ""])
def test_an_explicit_off_makes_no_call(monkeypatch, value):
    monkeypatch.setenv(WORLD_NO_POWER_THROTTLING_ENV, value)
    spy = _Spy()
    assert pt.apply_world_build_power_policy("solve", platform="win32", opt_out=spy) is False
    assert spy.calls == []


def test_on_windows_calls_once_with_the_role(monkeypatch):
    monkeypatch.setenv(WORLD_NO_POWER_THROTTLING_ENV, "on")
    spy = _Spy()
    assert pt.apply_world_build_power_policy("surface", platform="win32", opt_out=spy) is True
    assert spy.calls == ["surface"]


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_on_elsewhere_is_a_no_op(monkeypatch, platform):
    monkeypatch.setenv(WORLD_NO_POWER_THROTTLING_ENV, "on")
    spy = _Spy()
    assert pt.apply_world_build_power_policy("appearance", platform=platform, opt_out=spy) is False
    assert spy.calls == []


def test_the_reported_result_is_the_opt_outs(monkeypatch):
    monkeypatch.setenv(WORLD_NO_POWER_THROTTLING_ENV, "on")
    spy = _Spy(result=False)
    assert pt.apply_world_build_power_policy("finisher", platform="win32", opt_out=spy) is False
    assert spy.calls == ["finisher"]


def test_an_explicit_setting_overrides_the_environment(monkeypatch):
    monkeypatch.delenv(WORLD_NO_POWER_THROTTLING_ENV, raising=False)
    spy = _Spy()
    assert pt.apply_world_build_power_policy("builder", setting=True, platform="win32", opt_out=spy)
    assert spy.calls == ["builder"]


# -- the opt-out itself: the right call, and failures logged not raised -------------

class _FakeKernel32:
    """Records what SetProcessInformation was asked for."""

    def __init__(self, ok=1, raises=None):
        self.ok = ok
        self.raises = raises
        self.calls = []

    def GetCurrentProcess(self):  # noqa: N802 -- the Win32 name
        return -1

    def SetProcessInformation(self, handle, info_class, ref, size):  # noqa: N802
        if self.raises is not None:
            raise self.raises
        state = ref._obj
        self.calls.append({
            "handle": handle, "class": info_class, "size": size,
            "version": state.Version, "control": state.ControlMask, "state": state.StateMask,
        })
        return self.ok


def test_the_opt_out_asks_for_never_throttle_execution_speed():
    kernel32 = _FakeKernel32()
    assert pt.opt_out_of_power_throttling("builder", kernel32=kernel32) is True
    assert kernel32.calls == [{
        "handle": -1, "class": 4, "size": 12,
        "version": 1, "control": 0x1, "state": 0,
    }]


def test_a_refused_call_is_logged_and_returns_false(caplog):
    kernel32 = _FakeKernel32(ok=0)
    with caplog.at_level(logging.WARNING, logger="tower.power_throttling"):
        assert pt.opt_out_of_power_throttling("solve", kernel32=kernel32) is False
    assert "could not opt out of power throttling" in caplog.text
    assert "solve" in caplog.text


def test_an_exception_is_logged_never_raised(caplog):
    kernel32 = _FakeKernel32(raises=OSError("no such function"))
    with caplog.at_level(logging.WARNING, logger="tower.power_throttling"):
        assert pt.opt_out_of_power_throttling("surface", kernel32=kernel32) is False
    assert "OSError" in caplog.text


@pytest.mark.skipif(sys.platform == "win32", reason="the off-Windows path")
def test_off_windows_the_real_opt_out_fails_soft(caplog):
    # No kernel32 here: ctypes.WinDLL does not exist. Logged, False, no raise.
    with caplog.at_level(logging.WARNING, logger="tower.power_throttling"):
        assert pt.opt_out_of_power_throttling("builder") is False
    assert "could not opt out of power throttling" in caplog.text


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only: the real SetProcessInformation")
def test_on_windows_the_process_reads_back_as_opted_out():
    # In a child, so the test runner's own state is not touched. EcoQoS first, then the
    # opt-out: the later explicit state wins, and GetProcessInformation reads it back.
    code = (
        "import ctypes, ctypes.wintypes as w, json\n"
        "from tower import power_throttling as pt\n"
        "k = pt._kernel32()\n"
        "k.GetProcessInformation.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]\n"
        "S = pt._state_structure()\n"
        "on = S(1, 1, 1)\n"
        "k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(on), ctypes.sizeof(on))\n"
        "ok = pt.apply_world_build_power_policy('builder', setting=True)\n"
        "s = S(1, 0, 0)\n"
        "k.GetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s))\n"
        "print(json.dumps([ok, s.ControlMask, s.StateMask]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=TOWER_ROOT, capture_output=True,
                         text=True, check=True, timeout=60)
    assert out.stdout.strip().splitlines()[-1] == "[true, 1, 0]"


# -- every World Builder entry point calls it, right after its arguments, before any work ----

class _Called(Exception):
    pass


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"_pt_{name}", TOWER_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script, role, argv", [
    ("world_build_session", "builder", ["--synthetic"]),
    ("world_solve", "solve", ["--world", "w", "--session", "s"]),
    ("world_surface", "surface", ["--world", "w"]),
    ("world_appearance", "appearance", ["--world", "w"]),
    ("world_finish_pending", "finisher", []),
])
def test_every_entry_point_applies_the_policy_before_any_work(monkeypatch, tmp_path, script, role, argv):
    module = _load_script(script)
    seen = []

    def policy(name):
        seen.append(name)
        raise _Called

    monkeypatch.setattr(module, "apply_world_build_power_policy", policy)
    argv = ["--root", str(tmp_path / "root"), *argv]
    with pytest.raises(_Called):
        if script == "world_surface":  # its main() reads sys.argv
            monkeypatch.setattr(sys, "argv", [script, *argv])
            module.main()
        else:
            module.main(argv)
    assert seen == [role]
    # Nothing was created before the policy ran.
    assert not (tmp_path / "root").exists()


def test_the_structure_is_the_win32_size():
    # PROCESS_POWER_THROTTLING_STATE is three ULONGs.
    assert ctypes.sizeof(pt._state_structure()) == 12
