"""Byte identity of every payload the read-path raw allowlist must NOT move.

`TOWER_WORLD_RAW_IMAGERY_WORLDS` (manager 201 §1, option C) widens the
appearance serving gate by exactly one case: a raw-local-research artifact of a
world on the list, on a Tower serving the product. Everything else must be byte
for byte what `ba5f138` (the base: the photos-after-Stop fix) serves:

- with the list UNSET, EMPTY, or malformed in any way (fail closed) -- every
  scenario below, raw or redacted;
- with the list naming another world -- every scenario;
- with the list naming THIS world -- every scenario whose artifact is not raw,
  and every scenario on a research Tower (`TOWER_WORLD_RAW_IMAGERY` on), which
  does not consult the list.

What is observed is `test_world_builder_photos_after_stop_identity._observe`
unchanged: the build's redaction boundary, the manifest and every file on disk,
the revision (with and without the appearance viewer), the appearance manifest,
every chunk, the proxy, gzip and deflate on the wire, the render page, the
`/worlds` listing, the viewer config, the label / imagery / carry-over rules --
and, for an area, the same over its twin routes and
`components.area_appearance_servable`. The world id is a real (32-hex) one, so
that the list can name it.

TO RECORD: `WB_RAW_ALLOWLIST_RECORD=<path>` writes the observations (list
unset) to <path> instead of comparing. Record only from `ba5f138`, twice, and
keep the golden only if the two recordings are identical. The env var name is
spelled out here rather than imported, so this file runs on the base unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

import tests.test_world_builder_appearance as TWA
import tests.test_world_builder_photos_after_stop_identity as ID
from tests.test_world_builder_appearance import SESSION, TRUSTED, UNGATED, World, _never_redact
from tower.world_builder import appearance as A
from tower.world_builder import appearance_pipeline as AP
from tower.world_builder import raw_imagery as RAWIMG

GOLDEN = Path(__file__).parent / "golden" / "world_builder_raw_allowlist_ba5f138.json"
RECORD_ENV = "WB_RAW_ALLOWLIST_RECORD"
LIST_ENV = "TOWER_WORLD_RAW_IMAGERY_WORLDS"
HEX = "9835c5180000400080000000000000aa"
OTHER = "b2a75ab40000400080000000000000bb"
AREA = "a1a1a1a1a1a1a1a1"

UNSET = None
# The list must change NOTHING under any of these, for any scenario.
NEVER_WIDENS = {
    "unset": UNSET, "empty": "", "space": " ", "tab": "\t", "newline": "\n",
    "not_hex": "zz", "partial": HEX[:8], "short_by_one": HEX[:-1], "upper": HEX.upper(),
    "duplicate": f"{HEX},{HEX}", "trailing_comma": f"{HEX},", "leading_space": f" {HEX}",
    "spaced_list": f"{OTHER}, {HEX}", "one_bad_entry": f"{HEX},zz",
    "semicolon": f"{HEX};{OTHER}", "other_world": OTHER,
}
# The list names this world: nothing changes unless the artifact is raw on a
# product Tower.
NAMES_THIS_WORLD = {"listed": HEX, "listed_second": f"{OTHER},{HEX}"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


def _raw_params(**kw):
    kw.setdefault("selection_samples", 4000)
    return A.AppearanceParams(imagery_source=RAWIMG.IMAGERY_RAW, **kw)


def _trusted(tmp_path, log):
    w = World(tmp_path)
    assert w.build(redactor_factory=_never_redact).state == AP.STATE_OK
    return w


def _raw(tmp_path, log):
    w = World(tmp_path)
    TWA._paint_forbidden_sources(w)
    assert w.build(params=_raw_params(), redactor_factory=_never_redact).state == AP.STATE_OK
    return w


def _raw_then_redacted(tmp_path, log):
    w = _raw(tmp_path, log)
    assert w.build(redactor_factory=_never_redact, force=True).state == AP.STATE_OK
    return w


def _redacted_then_raw(tmp_path, log):
    w = _trusted(tmp_path, log)
    TWA._paint_forbidden_sources(w)
    assert w.build(params=_raw_params(), redactor_factory=_never_redact,
                   force=True).state == AP.STATE_OK
    return w


def _relabel(label):
    return lambda w: w.set_label(label)


def _purge(w):
    record = w.store.read_world(HEX)
    w.store.write_world(type(record)(**{**record.__dict__, "images_purged": True}))


def _then(make, *steps):
    def run(tmp_path, log):
        w = make(tmp_path, log)
        for step in steps:
            step(w)
        return w
    return run


def _with_area(make, raw: bool):
    """`make`, then an area built by the real pipeline through `AreaStore`."""
    def run(tmp_path, log):
        from tests.test_world_builder_components_areas import _entries, write_components
        from tower.world_builder import components as C

        w = make(tmp_path, log)
        if raw:
            TWA._paint_forbidden_sources(w)
        record = write_components(w.store, _entries(w.kids), world_id=HEX)
        view = C.AreaStore(w.store, HEX, SESSION, AREA)
        for stage in ("solve", "dense", "surface"):
            shutil.copytree(w.store.world_dir(HEX) / stage / SESSION,
                            view.area_dir / stage / SESSION,
                            ignore=shutil.ignore_patterns(C.COMPONENTS_FILENAME))
        C.write_area_record(w.store, HEX, SESSION, AREA, components_sha1=record.sha1,
                            stage="surface", state="ok", levelled=True)
        params = _raw_params() if raw else A.AppearanceParams(selection_samples=4000)
        result = AP.build_appearance(view, HEX, SESSION, params=params, device="cpu",
                                     redactor_factory=_never_redact)
        assert result.state == AP.STATE_OK, result.detail
        C.write_area_record(w.store, HEX, SESSION, AREA, components_sha1=record.sha1,
                            stage="appearance", state="ok")
        w.area = AREA
        return w
    return run


SCENARIOS = {
    # name: (factory, research Tower (`TOWER_WORLD_RAW_IMAGERY` on)?, raw artifact?)
    "redacted_trusted": (_trusted, False, False),
    "redacted_trusted_research_tower": (_trusted, True, False),
    "redacted_relabelled_ungated": (_then(_trusted, _relabel(UNGATED)), False, False),
    "redacted_purged": (_then(_trusted, _purge), False, False),
    "redacted_walk_during_walk": (ID._walk, False, False),
    "redacted_walk_after_stop": (_then(ID._walk, _relabel(TRUSTED)), False, False),
    "redacted_after_a_raw_build": (_raw_then_redacted, False, False),
    "raw": (_raw, False, True),
    "raw_research_tower": (_raw, True, True),
    "raw_relabelled_ungated": (_then(_raw, _relabel(UNGATED)), False, True),
    "raw_purged": (_then(_raw, _purge), False, True),
    "raw_after_a_redacted_build": (_redacted_then_raw, False, True),
    "raw_walk_manifest_after_stop": (ID._raw_manifest, False, True),
    "area_redacted": (_with_area(_trusted, raw=False), False, False),
    "area_raw": (_with_area(_trusted, raw=True), False, True),
    "area_raw_research_tower": (_with_area(_trusted, raw=True), True, True),
}


def _variants(raw_env: bool, raw_artifact: bool) -> dict:
    out = dict(NEVER_WIDENS)
    if raw_env or not raw_artifact:
        out.update(NAMES_THIS_WORLD)
    return out


def _observe(w, log, monkeypatch, value):
    if value is None:
        monkeypatch.delenv(LIST_ENV, raising=False)
    else:
        monkeypatch.setenv(LIST_ENV, value)
    observed = ID._observe(w, ID._Norm(), log)
    return json.dumps(observed, indent=1, sort_keys=True)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_other_payload_is_byte_identical_to_ba5f138(name, tmp_path, monkeypatch):
    make, raw_env, raw_artifact = SCENARIOS[name]
    monkeypatch.setattr(TWA, "WORLD", HEX)
    monkeypatch.setattr(ID, "WORLD", HEX)
    monkeypatch.delenv(RAWIMG.RAW_IMAGERY_ENV, raising=False)
    monkeypatch.delenv(LIST_ENV, raising=False)
    log: list = []
    w = make(tmp_path, log)
    if raw_env:
        monkeypatch.setenv(RAWIMG.RAW_IMAGERY_ENV, "1")
    record = os.environ.get(RECORD_ENV)
    if record:
        text = _observe(w, log, monkeypatch, UNSET)
        path = Path(record)
        doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"scenarios": {}}
        doc["scenarios"][name] = json.loads(text)
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8",
                        newline="\n")
        return
    expected = json.dumps(json.loads(GOLDEN.read_text(encoding="utf-8"))["scenarios"][name],
                          indent=1, sort_keys=True)
    moved = [variant for variant, value in sorted(_variants(raw_env, raw_artifact).items())
             if _observe(w, log, monkeypatch, value) != expected]
    assert moved == [], f"{name}: the payload moved under {moved}"


def test_the_scenarios_cover_both_sides_of_the_list():
    """At least one raw artifact on a product Tower (where only NEVER_WIDENS is
    compared) and every other kind (where the listed variants are compared too)."""
    kinds = {(raw_env, raw) for _make, raw_env, raw in SCENARIOS.values()}
    assert kinds == {(False, False), (True, False), (False, True), (True, True)}
    assert any(name.startswith("area_") for name in SCENARIOS)


def test_the_golden_file_itself_is_the_recorded_bytes():
    """A one-byte edit to the golden is caught even where it would decode to the
    same JSON: the file is pinned by its digest (line endings folded first)."""
    data = GOLDEN.read_bytes().replace(b"\r\n", b"\n")
    assert _sha(data) == GOLDEN_SHA256, _sha(data)


GOLDEN_SHA256 = "c9e57a1a754a64150e085e023b0b28b307d8948f679158cd8cbb8763782d4751"
