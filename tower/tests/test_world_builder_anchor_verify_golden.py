"""THE GOLDEN (P4-IV RULE.md section 9.3, item 1): with `TOWER_WORLD_ANCHOR_VERIFY` unset, empty or `off`, the gate's
and the publish step's outputs -- `solution.json`, `components.json`, `consensus.json`, the gate record and
`apply_gate`'s own result, with and without the consensus's hooks -- are what the product wrote BEFORE the anchor
verification existed, recorded from the d649f9f working tree by RUN/experiments/P4-PROD/golden_record.py
(`wb_anchor_verify_fixtures.golden_outputs`). The only exclusions are the comparison's ALWAYS set (timestamps,
timings, paths).

THE RETRIEVAL ADMISSION'S ON PATH (C23-IMPL-F; `wb_retrieval_admission_on`): with `TOWER_WORLD_RETRIEVAL_ADMISSION`
on, each publish scenario is the golden's plus exactly one additive key, `retrieval_admission`, in its record and in
`solution.json`'s gate -- `ADMISSION_ON`, value for value; `apply_gate`'s own outputs are the golden's. Unset or off,
the golden as it was."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests import wb_anchor_verify_fixtures as F
from tests import wb_retrieval_admission_on as RA_ON
from tower.world_builder import coherence_gate as CG

GOLDEN = Path(__file__).parent / "golden" / "world_builder_gate_d649f9f.json"
# The ON golden's one addition per publish scenario: no gate group is a candidate in any of them -- the triangle's
# island is attached, the other-level island is refused on scale, and the consensus's D is withheld.
ADMISSION_ON = {
    "single_triangle": RA_ON.not_run(room_before=50),
    "single_link_other_level": RA_ON.not_run(room_before=30, scale_refused=[
        {"first_keyframe": "s1:00000030", "keyframes": 20, "ratios": 20, "scale_factor": 2.0}]),
    "consensus_withhold": RA_ON.not_run(room_before=70, withheld_groups=1),
}


def _expected(golden: dict) -> dict:
    """The golden's outputs as the switches in the environment write them: the OFF golden, and with the retrieval
    admission on, the OFF golden plus `ADMISSION_ON`."""
    expected = golden["outputs"]
    if not RA_ON.on():
        return expected
    return {k: RA_ON.with_record(v, ADMISSION_ON[k]) if k in ADMISSION_ON else v for k, v in expected.items()}


def _round_floats(obj, nd=9):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, nd) for v in obj]
    return obj


@pytest.mark.parametrize("value", [None, "", "off", "  OFF "])
def test_golden_with_the_switch_unset_or_off_every_output_is_todays(tmp_path, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TOWER_WORLD_ANCHOR_VERIFY", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_ANCHOR_VERIFY", value)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    now = json.loads(json.dumps(F.golden_outputs(tmp_path), sort_keys=True))
    expected = _expected(golden)
    assert sorted(now) == sorted(expected)
    for key in expected:
        if (golden["cv2"], golden["numpy"]) == (cv2.__version__, np.__version__):
            assert now[key] == expected[key], key            # value for value
        else:  # another OpenCV/NumPy build: last digits may move, nothing else may
            assert _round_floats(now[key]) == _round_floats(expected[key]), key


def test_the_gate_params_digest_is_unchanged():
    assert CG.GateParams().digest() == "6ce602286efb999f"
    assert "anchor" not in json.dumps(CG.GateParams().to_json())


def test_the_golden_exercises_what_it_claims():
    outputs = json.loads(GOLDEN.read_text(encoding="utf-8"))["outputs"]
    c = outputs["consensus_withhold"]
    assert c["record"]["consensus"]["state"] == "applied" and c["record"]["consensus"]["detached"]
    assert [e["reasons"] for e in c["components.json"]["components"]] == [[], ["seed-unstable"]]
    assert c["consensus.json"] is not None
    # the anchor's mid stretch at x2 is NOT split today (RULE.md 3.1): the case part (a) exists for
    assert [x["label"] for x in outputs["apply_gate_mid_stretch"]["components"]] == [0]
    assert "anchor_verify" not in json.dumps(outputs)
