"""Did a patch of the appearance suite's world id outlive its undo? (lane T1, 2026-11-03)

The raw-allowlist tests patch `tests.test_world_builder_appearance.WORLD` ("w1") to a
32-hex id, `HEX`, so that their allowlist can name the world. A module imported for the
FIRST time inside that patch copies `HEX` for good, through `from ... import WORLD` or a
default `world_id=WORLD`, and the patch's undo restores only what it set. That once
happened to `tests.test_world_builder_components_areas`. As a result, the photos
identity file's seven area scenarios wrote to world `HEX` whenever a raw-allowlist file
ran first.

`copies_left_by` checks this regardless of test order. In a FRESH interpreter it:
1. imports the test module, as collection does;
2. applies the scenario's `WORLD` patch;
3. walks the module's area import path (its `_enter_the_area_import_path`, which stops
   at the first use of the world, so nothing is built);
4. undoes the patch;
5. reports what still holds `HEX`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TOWER = Path(__file__).resolve().parents[1]
APPEARANCE = "tests.test_world_builder_appearance"
COMPONENTS_AREAS = "tests.test_world_builder_components_areas"
# components_areas' copies of `WORLD`: the global, and the defaults made from it
COPIES = ("WORLD", "write_components", "_lifecycle", "_area_url")


class Halt(Exception):
    """Raised at the scenario's first use of its world."""


class Halting:
    """A world, or a world factory, that halts the scenario on first use."""

    def __call__(self, *args, **kwargs):
        raise Halt

    def __getattr__(self, name):
        raise Halt


def copies_left_by(module_name: str, targets: tuple[str, ...]) -> dict:
    """`{"original": <the unpatched id>, "components_areas": {copy: value},
    "copies_of_hex": [<module.attr still holding HEX>]}`, from a fresh interpreter
    that ran `module_name`'s area import path under its patch of `targets`."""
    code = (f"from tests.wb_world_id_leak import _child; "
            f"_child({module_name!r}, {list(targets)!r})")
    run = subprocess.run([sys.executable, "-c", code], cwd=TOWER, capture_output=True,
                         text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-4000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


def _child(module_name: str, targets: list[str]) -> None:
    import importlib
    import inspect

    import pytest

    module = importlib.import_module(module_name)          # what collection does
    original = sys.modules[APPEARANCE].WORLD
    with pytest.MonkeyPatch.context() as patch:            # what the scenario does
        for target in targets:
            patch.setattr(importlib.import_module(target), "WORLD", module.HEX)
        try:
            module._enter_the_area_import_path(Halting())
        except Halt:
            pass
    areas = sys.modules[COMPONENTS_AREAS]
    held = {"WORLD": areas.WORLD}
    for name in COPIES[1:]:
        held[name] = inspect.signature(getattr(areas, name)).parameters["world_id"].default
    print(json.dumps({"original": original, "components_areas": held,
                      "copies_of_hex": _copies_of(module.HEX)}))


def _copies_of(value: str) -> list[str]:
    """Every global, and every function default, of a loaded `tests.*` module that
    holds `value`, outside the modules that spell it themselves (`HEX`)."""
    def holds(v):
        return isinstance(v, str) and v == value

    out = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("tests.") or module is None or holds(getattr(module, "HEX", None)):
            continue
        for attr, v in list(vars(module).items()):
            defaults = [*(getattr(v, "__defaults__", None) or ()),
                        *(getattr(v, "__kwdefaults__", None) or {}).values()]
            if holds(v) or any(holds(d) for d in defaults):
                out.append(f"{name}.{attr}")
    return out
