"""`scripts/world_scale_guard_shadow.py`: the guard's pure assessment over an ALREADY PUBLISHED final solve,
read-only and CPU-only, for the shadow replays of frozen walks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.test_world_builder_final_scale_guard import W3_BATHROOM, W3_N
from tests.test_world_builder_final_scale_guard_seam import (  # noqa: F401 -- fixtures and helpers
    P,
    _Engines,
    _simple,
    _switches,
    _walk,
    _w3_levels_named,
    colmap,
)


def _tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_shadow_cli_scores_a_published_solve_and_writes_nothing_under_its_root(tmp_path, colmap,
                                                                                    monkeypatch):
    from scripts import world_scale_guard_shadow as cli

    _Engines(monkeypatch, n=W3_N, levels=_w3_levels_named())
    s = _walk(tmp_path / "w", W3_N, colmap, monkeypatch)
    _simple(s)                                          # published OFF: the bathroom is in the room
    work = s.store.world_dir(s.world_id) / "dense" / s.session_id / "work"
    (work / "depth").mkdir(parents=True)
    before = _tree(s.root)
    out = cli.shadow_score(s.root, s.world_id, s.session_id)
    assert _tree(s.root) == before
    sid = s.session_id
    assert {k for p in out["assessment"]["pieces"] for k in p["keyframe_ids"]} == \
        {f"{sid}:{i:08d}" for i in list(range(74, 84)) + list(W3_BATHROOM)}
    assert out["params_digest"] == P.digest() and out["room_source"] == "solver component 0"
    # --out is refused under --root, and taken elsewhere
    with pytest.raises(SystemExit):
        cli.main(["--root", str(s.root), "--world", s.world_id, "--session", s.session_id,
                  "--out", str(s.root / "x.json")])
    target = tmp_path / "shadow.json"
    assert cli.main(["--root", str(s.root), "--world", s.world_id, "--session", s.session_id,
                     "--out", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["assessment"]["decision"] == "publish"
    assert _tree(s.root) == before
