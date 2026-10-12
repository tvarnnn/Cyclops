"""ADAPTIVE SELECT OFF -- and ON BUT UNTRIGGERED -- IS 41cee79, BYTE FOR BYTE.

`TOWER_WORLD_SOLVE_SELECT` (`world_builder/solve_select.py`) is default off. Off, and on with draw 1's
guard isolating fewer keyframes than the trigger, the final solve must publish exactly what 41cee79 (the base,
which has no selector) published. This file proves it two ways:

1. AGAINST THE BASE: `golden/world_builder_solve_select_41cee79.json`, recorded by running THIS FILE on an
   export of 41cee79 (`git archive 41cee79 tower`), twice, identical. Every observation is the
   final-scale-guard off-identity harness's (`test_world_builder_final_scale_guard_off_identity._observe`):
   every file under the world root as the SHA-256 of its bytes (JSON with wall-clock stamps, durations
   and `solve_identity` replaced by placeholders), the status producer's payload, the saved-worlds row,
   the render revision, and the dict the solve returned. Three scenarios of the live product's final
   solve (unmasked, ungated, one draw):
     guard-off             the guard unset: the selector can never run;
     guard-on              the guard on, no metric evidence: room published uncertified, 0 isolated;
     guard-on-isolating    the guard on with live-cache evidence that isolates the 20-keyframe piece
                           (relabelled solution, components record, audit): isolated 20 < 150.
   Each under six spellings of the switch: unset, `off`, garbage, `adaptive:7` (invalid), `adaptive:1`
   and `adaptive:6` (valid, untriggered). On the base the switch means nothing, so every spelling there
   gives the base's bytes; here every spelling must give them too.
2. RAW BYTES, HERE: with the clocks frozen, solution.json, solution.npz, components.json and the guard's
   audit are the same raw bytes unset and `adaptive:6`.

TO RECORD (on 41cee79 only): `WB_SOLVE_SELECT_RECORD=<path>` writes the observations to <path>.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.test_world_builder_final_scale_guard_off_identity import _final, _make_walk, _observe
from tests.test_world_builder_solve_masks import colmap  # noqa: F401 -- fixture
from tower.world_builder import final_scale_guard as FSG

GOLDEN = Path(__file__).parent / "golden" / "world_builder_solve_select_41cee79.json"
RECORD_ENV = "WB_SOLVE_SELECT_RECORD"
SELECT = "TOWER_WORLD_SOLVE_SELECT"
GUARD = "TOWER_WORLD_FINAL_SCALE_GUARD"
SPELLINGS = (None, "off", "garbage", "adaptive:7", "adaptive:1", "adaptive:6")
SCENARIOS = ("guard-off", "guard-on", "guard-on-isolating")
SWITCHES = ("TOWER_WORLD_SOLVE_GATE", "TOWER_WORLD_SOLVE_MASKS", "TOWER_WORLD_SOLVE_SEED",
            "TOWER_WORLD_SOLVE_CONSENSUS", "TOWER_WORLD_ANCHOR_VERIFY", "TOWER_WORLD_POSE_QUARANTINE",
            "TOWER_WORLD_FINISH_STAGES", "TOWER_WORLD_PICTURE_BASIS", "TOWER_WORLD_LIVE_DEPTH",
            "TOWER_WORLD_SOLVE_SELECT_BUDGET_S", SELECT, GUARD)
# The walk (`test_world_builder_reproducible_finish`): a 30-keyframe room and a 20-keyframe piece.
PIECE = range(30, 50)


def _isolating_evidence(monkeypatch):
    """The live depth cache's metric log: the room at its level, the piece at 3x."""
    levels = {f"{i:08d}.jpg": (1.0986 if i in PIECE else 0.0) + 0.01 * ((i * 7) % 5 - 2) for i in range(50)}

    def live(store, world_id, session_id, solution, *, database_path, name_of, live_metric_fn=None):
        return dict(levels), {"state": "compatible", "records_matched": len(levels)}

    monkeypatch.setattr(FSG, "_live_cache_evidence", live)


def _setup(scenario, monkeypatch):
    from scripts import world_refinish as wr

    monkeypatch.setattr(wr, "_restart_attempts", lambda *a, **k: {})
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    if scenario != "guard-off":
        monkeypatch.setenv(GUARD, "on")
    if scenario == "guard-on-isolating":
        _isolating_evidence(monkeypatch)


def _spell(monkeypatch, spelling):
    if spelling is None:
        monkeypatch.delenv(SELECT, raising=False)
    else:
        monkeypatch.setenv(SELECT, spelling)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_off_and_untriggered_select_publish_41cee79s_bytes(scenario, tmp_path, colmap, monkeypatch):  # noqa: F811
    _setup(scenario, monkeypatch)
    runs = {}
    for n, spelling in enumerate(SPELLINGS):
        _spell(monkeypatch, spelling)
        s = _make_walk(tmp_path / f"r{n}", colmap, monkeypatch)
        out = _final(s, "simple")
        assert out["solved"] is True
        assert "select" not in (out.get("solve") or {})
        runs[spelling] = _observe(s, out)
    for spelling in SPELLINGS[1:]:
        a, b = runs[spelling], runs[None]
        moved = sorted(k for k in set(a["files"]) | set(b["files"]) if a["files"].get(k) != b["files"].get(k))
        assert a == b, (scenario, spelling, moved)
    observed = runs[None]
    if scenario == "guard-on-isolating":
        # Not vacuous: the guard did isolate the piece (relabelled solution, components record).
        assert any(k.endswith("components.json") for k in observed["files"])
        assert any(k.endswith(FSG.AUDIT_FILENAME) for k in observed["files"])
    _check(scenario, observed)


def _check(name, observed):
    record_to = os.environ.get(RECORD_ENV)
    if record_to:
        path = Path(record_to)
        golden = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        golden[name] = observed
        path.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        pytest.skip(f"recorded {name} into {path}")
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))[name]
    if isinstance(expected, dict) and set(expected) == set(observed):
        for key in sorted(expected):
            assert observed[key] == expected[key], f"{name}: {key} differs from 41cee79"
    assert observed == expected


def test_the_isolating_scenario_isolates_fewer_than_the_trigger(tmp_path, colmap, monkeypatch):  # noqa: F811
    """The untriggered case is a real one: the guard isolates keyframes, just fewer than the trigger's
    60-keyframe floor (so fewer than any threshold, fraction or absolute)."""
    if os.environ.get(RECORD_ENV):
        pytest.skip("recording")
    from tower import config

    _setup("guard-on-isolating", monkeypatch)
    s = _make_walk(tmp_path / "i", colmap, monkeypatch)
    _final(s, "simple")
    audit = json.loads((s.workspace.root / FSG.AUDIT_FILENAME).read_text(encoding="utf-8"))
    isolated = {k for p in audit["assessment"]["pieces"] for k in p["keyframe_ids"]}
    assert len(isolated) == len(PIECE) and 0 < len(isolated) < config.WORLD_SOLVE_SELECT_TRIGGER_MIN_KF


def test_raw_bytes_unset_and_untriggered_are_identical(tmp_path, colmap, monkeypatch):  # noqa: F811
    if os.environ.get(RECORD_ENV):
        pytest.skip("recording")
    import time

    from tower.world_builder import solve_select as SS

    _setup("guard-on-isolating", monkeypatch)
    monkeypatch.setattr(time, "perf_counter", lambda: 100.0)
    real, results = SS.select, []

    def spy(*a, **k):
        results.append(real(*a, **k))
        return results[-1]

    monkeypatch.setattr(SS, "select", spy)
    snapshots = []
    for n, spelling in enumerate((None, "adaptive:6")):
        _spell(monkeypatch, spelling)
        s = _make_walk(tmp_path / f"b{n}", colmap, monkeypatch)
        _final(s, "simple")
        root = s.workspace.root
        snapshots.append({name: (root / name).read_bytes()
                          for name in ("solution.json", "solution.npz", "components.json", FSG.AUDIT_FILENAME)})
        assert not (root / "select").exists()
    # The selector ran exactly once (adaptive:6), saw 20 isolated, and did not trigger.
    assert len(results) == 1 and results[0][1] is None and SS.isolated_keyframes(results[0][0]) == len(PIECE)
    assert snapshots[0] == snapshots[1]


def test_the_golden_is_sensitive_to_one_byte(tmp_path, colmap, monkeypatch):  # noqa: F811
    if os.environ.get(RECORD_ENV):
        pytest.skip("recording")
    _setup("guard-on-isolating", monkeypatch)
    s = _make_walk(tmp_path / "m", colmap, monkeypatch)
    out = _final(s, "simple")
    path = s.workspace.solution_path
    data = bytearray(path.read_bytes())
    at = data.index(b'"translation"') + len(b'"translation": [')
    while not chr(data[at]).isdigit():
        at += 1
    data[at] = ord("7") if data[at] != ord("7") else ord("3")
    path.write_bytes(bytes(data))
    with pytest.raises(AssertionError):
        _check("guard-on-isolating", _observe(s, out))
