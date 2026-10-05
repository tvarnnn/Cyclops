"""Byte identity of every payload the photos-after-Stop fix must NOT move.

The fix (`appearance_pipeline.may_serve`, manager 185 A) widens the appearance
serving gate by exactly one case: a manifest whose label no longer matches but
whose textures carry over to the keyframe set as it is now (the ordinary Stop
over a walk build that re-redacted every frame). It is landed without a switch
on the condition that every OTHER payload is byte for byte as it was at
`dc35d51` (the deployed build), and that the privacy redaction path itself is
provably unchanged. This file is that proof.

WHAT IS COMPARED, per scenario, against
`golden/world_builder_photos_after_stop_dc35d51.json`:

- the build itself, i.e. the redaction path: the bytes each keyframe handed the
  redactor and the bytes it got back (SHA-256), the manifest the build wrote,
  and the SHA-256 of every chunk and proxy file it wrote;
- every route a client reaches: `GET /worlds/{w}/render/revision` (with and
  without the appearance viewer), the appearance manifest, every chunk, the
  proxy (status, privacy headers, `X-World-Redaction`, `X-World-Imagery`, body
  digest), a gzip chunk, the render page (representation and body digest), the
  `/worlds` listing, and the appearance viewer config (or its refusal);
- the label rule, the imagery-source checks and the carry-over rule
  (`label_matches`, `imagery_source_of`, `imagery_matches`,
  `textures_carry_over`, `withdrawal_state`, `label_is_trusted`,
  `pixel_trust_token`) over a truth table, and under `TOWER_WORLD_RAW_IMAGERY`.

Scenarios cover served, withdrawn, absent, purged, raw-on, raw-manifest,
unlisted-redactor, unknown-redaction and re-redaction-switch cases. The single
intended change -- a walk build after Stop, served -- is excluded here and
pinned by `test_world_builder_photos_after_stop.py`.

Volatile values (build ids, epochs, wall-clock times, the surface file's
time-based name) are replaced by stable placeholders in order of first
appearance; nothing else is normalised.

TO RECORD: `WB_PHOTOS_AFTER_STOP_RECORD=<path>` writes the observed payloads to
<path> instead of comparing. Record only from `dc35d51`, twice, and keep the
golden only if the two recordings are identical.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from tests.test_world_builder_appearance import (
    SESSION,
    TRUSTED,
    UNGATED,
    WORLD,
    World,
    _app,
    _FakeRedactor,
    _never_redact,
    _records_carry,
)
from tower.world_builder import appearance as A
from tower.world_builder import appearance_pipeline as AP
from tower.world_builder import raw_imagery as RAWIMG

GOLDEN = Path(__file__).parent / "golden" / "world_builder_photos_after_stop_dc35d51.json"
RECORD_ENV = "WB_PHOTOS_AFTER_STOP_RECORD"
VIEWER = "appearance-1"
HEADERS = ("cache-control", "pragma", "x-content-type-options", "content-type",
           "content-encoding", "vary", "x-world-redaction", "x-world-imagery",
           "x-world-imagery-warning", "etag", "last-modified")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# a redactor that records what crossed the redaction boundary
# ---------------------------------------------------------------------------


def _recording(base, log):
    class _Recording(base):
        def redact(self, data):
            result = super().redact(data)
            log.append({"in": _sha(data), "out": _sha(result.image_bytes),
                        "label": result.label, "regions": result.regions})
            return result

    _Recording.__name__ = base.__name__
    return _Recording


class _Unlisted(_FakeRedactor):
    label = "faces-blurred/some-other-detector"


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


class _Norm:
    """Replaces volatile values with placeholders, in order of first sight."""

    VOLATILE_KEYS = {"built_at", "build_id", "epoch", "solved_at", "at", "updated_at",
                     "surface_built_at", "seconds", "created_at", "started_at",
                     "stopped_at", "changed_at", "requested_at", "mtime",
                     # a SHA-256 over the build's inputs, which include the
                     # surface's wall-clock `built_at`; what it digests is
                     # compared field by field in the manifest itself
                     "params_digest"}

    def __init__(self):
        self.tokens: dict[str, str] = {}

    def token(self, value) -> str:
        key = json.dumps(value, sort_keys=True)
        if key not in self.tokens:
            self.tokens[key] = f"<v{len(self.tokens)}>"
        return self.tokens[key]

    def learn(self, manifest):
        if isinstance(manifest, dict):
            for k in ("build_id", "epoch"):
                if isinstance(manifest.get(k), str):
                    self.token(manifest[k])

    def text(self, s: str) -> str:
        for key, tok in sorted(self.tokens.items(), key=lambda kv: -len(kv[0])):
            raw = json.loads(key)
            if isinstance(raw, str) and len(raw) >= 8:
                s = s.replace(raw, tok)
        s = re.sub(r"mesh_l0\.[0-9a-f]+\.bin", "mesh_l0.<t>.bin", s)
        s = re.sub(r"conf_l0\.[0-9a-f]+\.bin", "conf_l0.<t>.bin", s)
        s = re.sub(r"surface:\d+(\.\d+)?", "surface:<t>", s)
        return s

    def value(self, v, key=None):
        if key in self.VOLATILE_KEYS and v is not None:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return "<time>"
            if isinstance(v, str):
                return self.token(v)
            if isinstance(v, dict):
                return {k: "<time>" for k in sorted(v)}
        if isinstance(v, dict):
            return {k: self.value(x, k) for k, x in sorted(v.items())}
        if isinstance(v, list):
            return [self.value(x) for x in v]
        if isinstance(v, str):
            return self.text(v)
        return v


# ---------------------------------------------------------------------------
# observing a world
# ---------------------------------------------------------------------------


def _doc(v):
    """A manifest, compactly but completely: the SHA-256 of its normalised
    JSON (any byte of any field moves it), plus its provenance and counts in
    the clear so a failure says what moved."""
    if not isinstance(v, dict) or "appearance_provenance" not in v:
        return v
    return {"sha256": _sha(json.dumps(v, sort_keys=True).encode("utf-8")),
            "appearance_provenance": v.get("appearance_provenance"),
            "currency": v.get("currency"),
            "imagery_source": v.get("imagery_source"),
            "privacy_safe": v.get("privacy_safe"),
            "keyframes": len(v.get("keyframes") or []),
            "chunks": len(v.get("chunks") or []),
            "keys": sorted(v)}


def _client(root):
    from fastapi.testclient import TestClient

    return TestClient(_app(root))


def _resp(r, norm, body="json"):
    out = {"status": r.status_code,
           "headers": {h: norm.text(r.headers[h]) for h in HEADERS if h in r.headers}}
    if body == "json":
        try:
            out["body"] = _doc(norm.value(r.json()))
        except ValueError:
            out["body_sha256"] = _sha(r.content)
    else:
        out["body_sha256"] = _sha(r.content)
        out["bytes"] = len(r.content)
    return out


def _wire(client, url, headers, norm):
    """The bytes as sent, NOT as decoded: the test client undoes
    `Content-Encoding`, so `.content` would hide the compressed stream."""
    with client.stream("GET", url, headers=headers) as r:
        raw = b"".join(r.iter_raw())
        return {"status": r.status_code,
                "headers": {h: norm.text(r.headers[h]) for h in HEADERS if h in r.headers},
                "wire_sha256": _sha(raw), "wire_bytes": len(raw)}


def _files(w, norm):
    root = AP.appearance_dir(w.store, WORLD, SESSION)
    if not root.is_dir():
        return {}
    return {norm.text(p.name): _sha(p.read_bytes())
            for p in sorted(root.iterdir()) if p.is_file() and p.name.endswith(".bin")}


def _observe(w, norm, redactions):
    from tower.world_builder.appearance_render import (
        AppearanceViewerUnavailable,
        build_appearance_config,
    )

    man = w.manifest()
    norm.learn(man)
    client = _client(w.root)
    out = {"redactions": redactions, "manifest_on_disk": _doc(norm.value(man)),
           "files": _files(w, norm)}
    base = f"/worlds/{WORLD}/appearance/{SESSION}"
    out["revision_viewer"] = _resp(client.get(
        f"/worlds/{WORLD}/render/revision", params={"session_id": SESSION, "viewer": VIEWER}),
        norm)
    out["revision_plain"] = _resp(client.get(
        f"/worlds/{WORLD}/render/revision", params={"session_id": SESSION}), norm)
    out["manifest"] = _resp(client.get(f"{base}/manifest"), norm)
    if man:
        out["chunks"] = [_resp(client.get(f"{base}/chunk/{c['digest']}"), norm, body="bytes")
                         for c in man.get("chunks") or []]
        if man.get("chunks"):
            out["chunk_gzip"] = _wire(client, f"{base}/chunk/{man['chunks'][0]['digest']}",
                                      {"Accept-Encoding": "gzip"}, norm)
            out["chunk_deflate"] = _wire(client, f"{base}/chunk/{man['chunks'][0]['digest']}",
                                         {"Accept-Encoding": "deflate"}, norm)
        out["proxy"] = _resp(client.get(f"{base}/proxy/{(man.get('proxy') or {}).get('digest')}"),
                             norm, body="bytes")
    page = client.get(f"/worlds/{WORLD}/render", params={"session_id": SESSION, "viewer": VIEWER})
    m = re.search(r'<meta name="wb-representation" content="([a-z]+)">', page.text)
    out["page"] = {"status": page.status_code, "representation": m.group(1) if m else None,
                   "body_sha256": _sha(norm.text(page.text).encode("utf-8"))}
    listing = client.get("/worlds").json()
    rows = [x for x in listing.get("worlds", []) if x.get("world_id") == WORLD]
    out["listing_appearance"] = norm.value(
        [s.get("appearance") for r in rows for s in r.get("sessions", [])])
    try:
        cfg = build_appearance_config(w.store, WORLD, SESSION)
        out["viewer_config"] = norm.value(cfg)
    except AppearanceViewerUnavailable as exc:
        out["viewer_config"] = {"refused": str(exc)}
    out["rules"] = _rules(w, man)
    if getattr(w, "area", None):
        out["area"] = _observe_area(w, w.area, norm)
    return out


def _observe_area(w, area, norm):
    """The area's routes (COMPONENTS §5.3): the same gate, through
    `components.area_appearance_servable`."""
    from tower.world_builder import components as C

    view = C.AreaStore(w.store, WORLD, SESSION, area)
    man = AP.read_appearance_manifest(view, WORLD, SESSION)
    norm.learn(man)
    client = _client(w.root)
    base = f"/worlds/{WORLD}/areas/{SESSION}/{area}"
    root = AP.appearance_dir(view, WORLD, SESSION)
    out = {"manifest_on_disk": _doc(norm.value(man)),
           "files": ({norm.text(p.name): _sha(p.read_bytes()) for p in sorted(root.iterdir())
                      if p.is_file() and p.name.endswith(".bin")} if root.is_dir() else {}),
           "revision": _resp(client.get(f"{base}/render/revision"), norm),
           "manifest": _resp(client.get(f"{base}/appearance/manifest"), norm)}
    if man:
        out["chunks"] = [_resp(client.get(f"{base}/appearance/chunk/{c['digest']}"), norm,
                               body="bytes") for c in man.get("chunks") or []]
        out["proxy"] = _resp(client.get(
            f"{base}/appearance/proxy/{(man.get('proxy') or {}).get('digest')}"), norm,
            body="bytes")
    page = client.get(f"{base}/render")
    m = re.search(r'<meta name="wb-representation" content="([a-z]+)">', page.text)
    out["page"] = {"status": page.status_code, "representation": m.group(1) if m else None,
                   "body_sha256": _sha(norm.text(page.text).encode("utf-8"))}
    servable = C.area_appearance_servable(w.store, WORLD, SESSION, area)
    out["servable"] = [servable[0] is not None, servable[1]]
    out["withdrawal_state"] = AP.withdrawal_state(view, WORLD, SESSION, man)
    return out


def _rules(w, man):
    """The label rule, the imagery checks and the carry-over rule, as functions
    of this world's manifest and keyframe set."""
    if not isinstance(man, dict):
        return None
    prov = man.get("appearance_provenance") or {}
    out = {}
    for src in (RAWIMG.IMAGERY_REDACTED, RAWIMG.IMAGERY_RAW):
        out[f"label_matches[{src}]"] = AP.label_matches(w.store, WORLD, SESSION, man,
                                                        imagery_source=src)
        out[f"imagery_matches[{src}]"] = AP.imagery_matches(man, src)
    out["label_matches[env]"] = AP.label_matches(w.store, WORLD, SESSION, man)
    out["imagery_source_of"] = AP.imagery_source_of(man)
    out["withdrawal_state"] = AP.withdrawal_state(w.store, WORLD, SESSION, man)
    label, image_set = A.keyframe_set_identity(w.store, WORLD, SESSION)
    out["keyframe_set_identity"] = [label, image_set]
    out["label_is_trusted"] = A.label_is_trusted(label)
    out["pixel_trust_token"] = A.pixel_trust_token(label, prov.get("redactor_applied_here"))
    out["consensus_rule_of"] = AP.consensus_rule_of(prov)
    return out


def _truth_table():
    """`textures_carry_over`, `imagery_source_of`, `label_is_trusted` and
    `pixel_trust_token` over every combination that decides them."""
    labels = [None, "none", "", TRUSTED, UNGATED, "x", RAWIMG.RAW_LABEL]
    rows = []
    for imagery_prev in (None, RAWIMG.IMAGERY_REDACTED, RAWIMG.IMAGERY_RAW, "bogus"):
        for imagery_now in (None, RAWIMG.IMAGERY_REDACTED, RAWIMG.IMAGERY_RAW):
            for trusted_prev in (False, True):
                for redactor in (None, TRUSTED, "x", "none"):
                    for label_prev, label_now in ((None, TRUSTED), (TRUSTED, TRUSTED),
                                                  (TRUSTED, UNGATED), ("none", "none")):
                        for set_prev, set_now in ((None, None), (None, "t1"), ("t1", "t1")):
                            for rule_prev, rule_now in ((None, None), ("a", None),
                                                        ("a", "a"), ("a", "b")):
                                prev = {"imagery_source": imagery_prev,
                                        "label_trusted": trusted_prev,
                                        "redactor_applied_here": redactor,
                                        "session_redaction": label_prev,
                                        "keyframe_image_set": set_prev,
                                        "redaction_consensus": {"rule": rule_prev}
                                        if rule_prev else None}
                                now = {"imagery_source": imagery_now,
                                       "label_trusted": A.label_is_trusted(label_now),
                                       "session_redaction": label_now,
                                       "keyframe_image_set": set_now,
                                       "redaction_consensus": {"rule": rule_now}
                                       if rule_now else None}
                                rows.append(AP.textures_carry_over(prev, now))
    return {
        "carry_over_bits": "".join("1" if r else "0" for r in rows),
        "imagery_source_of": [AP.imagery_source_of(x) for x in (
            None, {}, {"imagery_source": RAWIMG.IMAGERY_RAW},
            {"imagery_source": "bogus"},
            {"appearance_provenance": {"imagery_source": RAWIMG.IMAGERY_RAW}},
            {"imagery_source": "", "appearance_provenance": {"imagery_source": "x"}})],
        "label_is_trusted": [A.label_is_trusted(x) for x in labels],
        "pixel_trust_token": [A.pixel_trust_token(x, r) for x in labels for r in (None, TRUSTED)],
        "raw_env": [RAWIMG.imagery_source_from_env({RAWIMG.RAW_IMAGERY_ENV: v})
                    for v in ("", "0", "false", "off", "no", "1", "true", "on", "yes")],
    }


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


def _edit_manifest(w, edit):
    root = AP.appearance_dir(w.store, WORLD, SESSION)
    doc = json.loads((root / "manifest.json").read_text())
    edit(doc)
    (root / "manifest.json").write_text(json.dumps(doc))


def _purge(w):
    record = w.store.read_world(WORLD)
    w.store.write_world(type(record)(**{**record.__dict__, "images_purged": True}))


def _walk(tmp_path, log, redactor=_FakeRedactor):
    w = World(tmp_path, label="none")
    _records_carry(w, redactor())
    r = w.build(params=A.AppearanceParams.live(selection_samples=3000, transient_detector="off"),
                redactor_factory=_recording(redactor, log))
    assert r.state == AP.STATE_OK, r.detail
    return w


def _walk_with_a_fill_mask(tmp_path, log):
    """A walk whose depth stage recorded a NON-EMPTY fill mask on every frame
    (review rv-pas LOW-4, mutant R12): the re-run path ORs the stored fill into
    the re-redaction's own, and with all-zero masks nothing could see it."""
    import numpy as np

    w = World(tmp_path, label="none")
    for i in range(len(w.kids)):
        mask = np.zeros((119, 159), bool)
        mask[30:60, 40 + 5 * i:90 + 5 * i] = True
        w.set_fill(i, mask)
    _records_carry(w, _FakeRedactor())
    r = w.build(params=A.AppearanceParams.live(selection_samples=3000, transient_detector="off"),
                redactor_factory=_recording(_FakeRedactor, log))
    assert r.state == AP.STATE_OK, r.detail
    return w


def _trusted(tmp_path):
    w = World(tmp_path)
    assert w.build(redactor_factory=_never_redact).state == AP.STATE_OK
    return w


UNTRUSTED = "faces-detected-and-filled/yunet-2023mar@0.50"


def _untrusted_from_the_start(tmp_path, log):
    """A label off the allowlist from the start: every frame re-redacted, and
    the label never changes, so `label_matches` serves it (no carry-over)."""
    w = World(tmp_path, label=UNTRUSTED)
    r = w.build(redactor_factory=_recording(_FakeRedactor, log))
    assert r.state == AP.STATE_OK, r.detail
    return w


def _final_after_stop(tmp_path, log):
    w = _walk(tmp_path, log)
    w.set_label(TRUSTED)
    _records_carry(w)
    r = w.build(params=A.AppearanceParams(selection_samples=4000, transient_detector="off"),
                redactor_factory=_never_redact)
    assert r.state == AP.STATE_OK, r.detail
    return w


def _unknown_redaction(tmp_path, log):
    w = _walk(tmp_path, log)
    _edit_manifest(w, lambda d: d["appearance_provenance"].pop("redactor_applied_here"))
    w.set_label(TRUSTED)
    return w


def _reredaction_switch(tmp_path, log):
    w = _walk(tmp_path, log)
    _edit_manifest(w, lambda d: d["appearance_provenance"].update(
        keyframe_image_set="reredacted:other"))
    w.set_label(TRUSTED)
    return w


def _raw_manifest(tmp_path, log):
    w = _walk(tmp_path, log)

    def edit(d):
        d["imagery_source"] = RAWIMG.IMAGERY_RAW
        d["appearance_provenance"]["imagery_source"] = RAWIMG.IMAGERY_RAW
    _edit_manifest(w, edit)
    w.set_label(TRUSTED)
    return w


def _then(make, *steps):
    def run(tmp_path, log):
        w = make(tmp_path, log)
        for step in steps:
            step(w)
        return w
    return run


def _stop(label=TRUSTED):
    return lambda w: w.set_label(label)


AREA = "a1a1a1a1a1a1a1a1"


def _with_area(make, redactor=None, finalize_label=None):
    """`make`, then an area built through `AreaStore` by the real pipeline, under
    the session's label at that moment (`redactor` when it is not trusted)."""
    def run(tmp_path, log):
        import shutil

        from tests.test_world_builder_components_areas import _entries, write_components
        from tower.world_builder import components as C

        w = make(tmp_path, log)
        record = write_components(w.store, _entries(w.kids))
        view = C.AreaStore(w.store, WORLD, SESSION, AREA)
        for stage in ("solve", "dense", "surface"):
            shutil.copytree(w.store.world_dir(WORLD) / stage / SESSION,
                            view.area_dir / stage / SESSION,
                            ignore=shutil.ignore_patterns(C.COMPONENTS_FILENAME))
        C.write_area_record(w.store, WORLD, SESSION, AREA, components_sha1=record.sha1,
                            stage="surface", state="ok", levelled=True)
        factory = _recording(redactor, log) if redactor is not None else _never_redact
        params = (A.AppearanceParams.live(selection_samples=3000, transient_detector="off")
                  if redactor is not None
                  else A.AppearanceParams(selection_samples=4000, transient_detector="off"))
        result = AP.build_appearance(view, WORLD, SESSION, params=params, device="cpu",
                                     redactor_factory=factory)
        assert result.state == AP.STATE_OK, result.detail
        C.write_area_record(w.store, WORLD, SESSION, AREA, components_sha1=record.sha1,
                            stage="appearance", state="ok")
        if finalize_label is not None:
            from tests.test_world_builder_components_areas import _finalize

            w.set_label(finalize_label)
            _finalize(w)
        w.area = AREA
        return w
    return run


SCENARIOS = {
    # (factory, raw imagery env on?)
    "absent": (lambda tmp_path, log: World(tmp_path), False),
    "trusted_served": (lambda tmp_path, log: _trusted(tmp_path), False),
    "trusted_relabelled_ungated": (lambda tmp_path, log: _then(
        lambda t, lg: _trusted(t), _stop(UNGATED))(tmp_path, log), False),
    "trusted_relabelled_none": (lambda tmp_path, log: _then(
        lambda t, lg: _trusted(t), _stop("none"))(tmp_path, log), False),
    "trusted_purged": (lambda tmp_path, log: _then(
        lambda t, lg: _trusted(t), _purge)(tmp_path, log), False),
    "trusted_raw_env": (lambda tmp_path, log: _trusted(tmp_path), True),
    "untrusted_label_served": (_untrusted_from_the_start, False),
    "walk_during_walk": (_walk, False),
    "walk_during_walk_raw_env": (_walk, True),
    "walk_after_stop_raw_env": (_then(_walk, _stop()), True),
    "walk_after_stop_purged": (_then(_walk, _stop(), _purge), False),
    "walk_unlisted_redactor_after_stop": (
        lambda tmp_path, log: _then(lambda t, lg: _walk(t, lg, _Unlisted), _stop())(
            tmp_path, log), False),
    "walk_unlisted_redactor_during_walk": (
        lambda tmp_path, log: _walk(tmp_path, log, _Unlisted), False),
    "walk_unknown_redaction_after_stop": (_unknown_redaction, False),
    "walk_reredaction_switch_after_stop": (_reredaction_switch, False),
    "walk_raw_manifest_after_stop": (_raw_manifest, False),
    "walk_then_final_build": (_final_after_stop, False),
    "walk_with_a_fill_mask_during_walk": (_walk_with_a_fill_mask, False),
    # areas (the intended change, a walk-time area after Stop, is excluded)
    "area_trusted_served": (_with_area(lambda t, lg: _trusted(t)), False),
    "area_trusted_relabelled_none": (
        _with_area(lambda t, lg: _trusted(t), finalize_label="none"), False),
    "area_trusted_purged": (
        _then(_with_area(lambda t, lg: _trusted(t)), _purge), False),
    "area_walk_during_walk": (_with_area(_walk, redactor=_FakeRedactor), False),
    "area_walk_after_stop_raw_env": (
        _with_area(_walk, redactor=_FakeRedactor, finalize_label=TRUSTED), True),
    "area_walk_after_stop_purged": (
        _then(_with_area(_walk, redactor=_FakeRedactor, finalize_label=TRUSTED), _purge),
        False),
    "area_walk_unlisted_after_stop": (
        _with_area(lambda t, lg: _walk(t, lg, _Unlisted), redactor=_Unlisted,
                   finalize_label=TRUSTED), False),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_other_payload_is_byte_identical_to_dc35d51(name, tmp_path, monkeypatch):
    make, raw_env = SCENARIOS[name]
    monkeypatch.delenv(RAWIMG.RAW_IMAGERY_ENV, raising=False)
    log: list = []
    w = make(tmp_path, log)
    if raw_env:
        monkeypatch.setenv(RAWIMG.RAW_IMAGERY_ENV, "1")
    observed = _observe(w, _Norm(), log)
    text = json.dumps(observed, indent=1, sort_keys=True)
    record = os.environ.get(RECORD_ENV)
    if record:
        path = Path(record)
        doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"scenarios": {}}
        doc["scenarios"][name] = json.loads(text)
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8",
                        newline="\n")
        return
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))["scenarios"][name]
    assert json.dumps(expected, indent=1, sort_keys=True) == text


def test_the_rules_truth_table_is_byte_identical_to_dc35d51():
    observed = json.dumps(_truth_table(), indent=1, sort_keys=True)
    record = os.environ.get(RECORD_ENV)
    if record:
        path = Path(record)
        doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"scenarios": {}}
        doc["truth_table"] = json.loads(observed)
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8",
                        newline="\n")
        return
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))["truth_table"]
    assert json.dumps(expected, indent=1, sort_keys=True) == observed


def test_the_golden_file_itself_is_the_recorded_bytes():
    """A one-byte edit to the golden is caught even where it would decode to
    the same JSON (whitespace, key order): the file is pinned by its digest.
    Line endings are folded first, so a `core.autocrlf` checkout still matches."""
    data = GOLDEN.read_bytes().replace(b"\r\n", b"\n")
    assert _sha(data) == GOLDEN_SHA256, _sha(data)


GOLDEN_SHA256 = "8a6312756520dad72099e53749b4bc7241cfb045c6206ffaaec5b426c71cb6ae"
