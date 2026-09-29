"""The retrieval admission's ON PATH in the suite's goldens and run-against-run comparisons (C23-IMPL-F; manager 149
section 3). Not a test module: no `test_` prefix.

With `TOWER_WORLD_RETRIEVAL_ADMISSION` on (`config.world_retrieval_admission_setting`; OFF by default, and OFF for walk
6), a publish through `coherence_publish.gate_and_publish` or `regate_published` whose draw 0 was not stopped carries
ONE additive, Tower-internal key, `retrieval_admission` (`retrieval_admission.py` step 7), in the gate record and so
in `solution.json`'s `gate`; and the record's `seconds` also counts the admission's own seconds
(`retrieval_admission._carried`). On the suite's synthetic solves nothing else changes: either no gate group is a
candidate (`not-run`, `no candidate group`), or -- the recording fake's database holds no SIFT descriptors -- the
retrieval fails closed (`failed`) and the room is published as it was gated.

The golden tests that meet it are SWITCH-AWARE: with the switch unset, blank, `off` or garbage they compare against
their OFF golden exactly as before; with it on, against the OFF golden plus exactly this key, spelled out here value
for value (`record`, `not_run`, `with_record`). Never a skip.
"""

from __future__ import annotations

import copy

KEY = "retrieval_admission"
ID = "retrieval-admission:c23-cert-direct-certificate/1"
# `AdmissionParams()` as the record writes it: PREREG section A's and P5-RETR's frozen values, and their digest.
PARAMS = {"amb_max_deg": 10.0, "e_prob": 0.9999, "e_px": 1.0, "e_px_stored": 1.5, "h_dominant": 0.9, "h_px": 4.0,
          "honoured_deg": 16.8, "hull_min": 0.1, "iterations": 15, "k_room": 15, "m_a_min": 2, "m_h_min": 2,
          "min_group_cameras": 3, "min_inliers": 15, "ratio": 0.8, "sample": 200000, "seed": 0, "words": 1024}
PARAMS_DIGEST = "9190d27dac2bb6e8"
STATE_NOT_RUN = "not-run"
STATE_FAILED = "failed"
# The retrieval's failure on the recording fake's database (`test_world_builder_solve_masks.colmap`): it writes no
# `descriptors` table, so `retrieval_admission.load_masked_features` cannot read the SIFT.
NO_DESCRIPTORS = "OperationalError: no such table: descriptors"


def on() -> bool:
    """The switch, read as the product reads it (garbage is off, and logged)."""
    from tower.config import world_retrieval_admission_setting

    return world_retrieval_admission_setting()


def record(state: str, *, room_before: int, withheld_groups: int = 0, sealed_keyframes: int = 0,
           collateral_keyframes: int = 0, scale_refused=(), **fields) -> dict:
    """The record `gate.retrieval_admission` as a golden comparison sees it: without its `seconds` (a timing; the
    goldens' `ALWAYS` exclusions drop every `seconds` key)."""
    return {"id": ID, "params": dict(PARAMS), "params_digest": PARAMS_DIGEST, "rider_min_shared": None,
            "state": state, "room_before": room_before,
            "carried": {"collateral_keyframes": collateral_keyframes, "sealed_keyframes": sealed_keyframes,
                        "withheld_groups": withheld_groups},
            "scale_refused": [dict(x) for x in scale_refused], **fields}


def not_run(**kw) -> dict:
    """No gate group is a candidate (retrieval_admission.py step 1): the input is published, and the record says so."""
    return record(STATE_NOT_RUN, why="no candidate group", candidates=[], **kw)


def with_record(output: dict, rec: dict) -> dict:
    """A publish output of an OFF golden (`record`, `solution.json`, ...) as the ON path writes it: `rec` in the gate
    record and in `solution.json`'s `gate`, and every other value the OFF golden's."""
    out = copy.deepcopy(output)
    out["record"][KEY] = copy.deepcopy(rec)
    out["solution.json"]["gate"][KEY] = copy.deepcopy(rec)
    return out
