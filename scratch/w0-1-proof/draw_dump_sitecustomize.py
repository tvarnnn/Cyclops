"""Installed as sitecustomize.py only by prove_identity.py's copied proof runs."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tower.world_builder import global_solve as GS


def digest_candidate(candidate):
    result = {}
    for field in dataclasses.fields(candidate):
        if field.name in ("solved_at", "timing"):
            continue
        value = getattr(candidate, field.name)
        if isinstance(value, np.ndarray):
            data = str(value.dtype).encode() + repr(value.shape).encode() + value.tobytes()
        else:
            data = json.dumps(value, sort_keys=True, allow_nan=True).encode()
        result[field.name] = hashlib.sha256(data).hexdigest()
    return result


_original = GS._map_candidate


def _recorded_map(*args, **kwargs):
    candidate = _original(*args, **kwargs)
    seed = kwargs.get("seed")
    if seed is not None:
        root = Path(os.environ["TOWER_W01_DRAW_DUMP"])
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"seed-{int(seed)}-pid-{os.getpid()}.json"
        path.write_text(json.dumps(digest_candidate(candidate), sort_keys=True), encoding="utf-8")
    return candidate


if os.environ.get("TOWER_W01_DRAW_DUMP"):
    GS._map_candidate = _recorded_map
