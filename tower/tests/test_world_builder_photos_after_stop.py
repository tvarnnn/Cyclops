"""Walk photos stay available to a viewer opened AFTER Stop (owner bug, manager 185 A).

Tristan: "photos captured during the walk should remain available to the
post-Stop saved-world viewer, subject to the existing face/privacy redaction
rules. Stop should end capture, not make already-captured walk imagery
unavailable."

The mechanism (U06-TUX1 spec §2.4, reasoned there, reproduced here): a walk's
appearance builds run under the label `none` and re-redact every keyframe with
the Tower's own redactor. At Stop the session's label becomes the real one, so
`label_matches` refused the walk-time manifest; the revision route answered
`state: rebuilding, revision: null`, and a viewer opened in that gap (up to
~39 minutes on walk 5) got the surface ladder rung, never the photos. An open
page kept drawing the very same textures (`textures_carry_over`, the ordinary
Stop), so the privacy decision for those pixels was already made: the fix
serves exactly what an open page was already allowed to keep, and nothing else.

Every refusal below is a privacy guard that must keep holding after the fix.
"""

from __future__ import annotations

import json

import pytest

from tests.test_world_builder_appearance import (
    SESSION,
    TRUSTED,
    UNGATED,
    WORLD,
    World,
    _app,
    _assert_private,
    _FakeRedactor,
    _never_redact,
    _records_carry,
)
from tower.world_builder import appearance as A
from tower.world_builder import appearance_pipeline as AP
from tower.world_builder import raw_imagery as RAWIMG

VIEWER = "appearance-1"


def _client(root):
    from fastapi.testclient import TestClient

    return TestClient(_app(root))


def _revision(client):
    return client.get(f"/worlds/{WORLD}/render/revision",
                      params={"session_id": SESSION, "viewer": VIEWER}).json()


def _walk(tmp_path, redactor=_FakeRedactor):
    """A walk: label `none`, a live appearance that re-redacted every frame."""
    w = World(tmp_path, label="none")
    _records_carry(w, redactor())
    live = w.build(params=A.AppearanceParams.live(selection_samples=3000,
                                                   transient_detector="off"),
                   redactor_factory=redactor)
    assert live.state == AP.STATE_OK, live.detail
    return w


def _stop(w, label=TRUSTED):
    w.set_label(label)


def _summary(client):
    body = client.get("/worlds").json()
    (row,) = [x for x in body["worlds"] if x["world_id"] == WORLD]
    (s,) = row["sessions"]
    return s["appearance"]


def _served_everything(client, man):
    """Every route a viewer opened now would use answers 200, privately."""
    m = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest")
    assert m.status_code == 200, m.text
    _assert_private(m)
    for chunk in man["chunks"]:
        c = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{chunk['digest']}")
        assert c.status_code == 200 and len(c.content) == chunk["bytes"]
        _assert_private(c)
    p = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/proxy/{man['proxy']['digest']}")
    assert p.status_code == 200 and p.content[:8] == b"WBSURF01"
    _assert_private(p)
    return m


def _refused_everything(client, man, state):
    m = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest")
    assert m.status_code == 404
    _assert_private(m)
    for chunk in man["chunks"]:
        assert client.get(
            f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{chunk['digest']}").status_code == 404
    assert client.get(
        f"/worlds/{WORLD}/appearance/{SESSION}/proxy/{man['proxy']['digest']}").status_code == 404
    rev = _revision(client)
    assert rev["appearance"]["revision"] is None
    assert rev["appearance"]["state"] == state
    assert rev["representation"] != "appearance"
    page = client.get(f"/worlds/{WORLD}/render", params={"session_id": SESSION, "viewer": VIEWER})
    assert 'name="wb-representation" content="appearance"' not in page.text
    return m


# ---------------------------------------------------------------------------
# the reproduction
# ---------------------------------------------------------------------------


class TestAViewerOpenedAfterStop:

    def test_gets_the_walk_photos(self, tmp_path):
        """THE BUG. Before the fix: revision null, rung `surface`, every route 404."""
        w = _walk(tmp_path)
        walking = _revision(_client(w.root))
        assert walking["representation"] == "appearance"
        man = w.manifest()

        _stop(w)
        client = _client(w.root)                       # a viewer opened after Stop
        after = _revision(client)
        assert after["representation"] == "appearance", after
        assert after["appearance"]["state"] == AP.SERVED
        assert after["appearance"]["revision"] == walking["appearance"]["revision"]
        assert after["appearance"]["epoch"] == walking["appearance"]["epoch"]
        assert after["revision"] == walking["revision"], "an open page is not reloaded"
        m = _served_everything(client, man)
        # The header says what was applied to THESE pixels, not the new label:
        # the walk build re-redacted every frame under `none`.
        assert m.headers["x-world-redaction"] == f"none&{TRUSTED}"
        assert m.headers["x-world-redaction"] != TRUSTED
        assert m.json()["appearance_provenance"]["label_trusted"] is False
        assert m.json()["appearance_provenance"]["redactor_applied_here"] == TRUSTED
        page = client.get(f"/worlds/{WORLD}/render",
                          params={"session_id": SESSION, "viewer": VIEWER})
        assert page.status_code == 200 and 'name="wb-representation" content="appearance"' in page.text
        assert _summary(client)["state"] == AP.SERVED
        assert _summary(client)["redaction"] is None or _summary(client)["redaction"] == "none"

    def test_the_final_build_still_replaces_it_in_place(self, tmp_path):
        w = _walk(tmp_path)
        walking = _revision(_client(w.root))
        _stop(w)
        gap = _revision(_client(w.root))
        _records_carry(w)
        final = w.build(params=A.AppearanceParams(selection_samples=4000,
                                                  transient_detector="off"),
                        redactor_factory=_never_redact)
        assert final.state == AP.STATE_OK, final.detail
        done = _revision(_client(w.root))
        assert done["appearance"]["state"] == AP.SERVED
        assert done["appearance"]["revision"] != gap["appearance"]["revision"]
        assert done["appearance"]["epoch"] == walking["appearance"]["epoch"]
        assert done["revision"] == walking["revision"] == gap["revision"]
        assert w.manifest()["appearance_provenance"]["label_trusted"] is True

    def test_the_viewer_config_is_built_for_it(self, tmp_path):
        from tower.world_builder.appearance_render import build_appearance_config

        w = _walk(tmp_path)
        _stop(w)
        config = build_appearance_config(w.store, WORLD, SESSION)
        assert config["build_id"] == w.manifest()["build_id"]
        assert config["imagery_source"] == RAWIMG.IMAGERY_REDACTED


# ---------------------------------------------------------------------------
# the privacy rules that must not move
# ---------------------------------------------------------------------------


class TestStillRefusedAfterStop:

    def test_a_walk_build_by_a_redactor_that_is_not_allowlisted(self, tmp_path):
        class _Unlisted(_FakeRedactor):
            label = "faces-blurred/some-other-detector"

        w = _walk(tmp_path, redactor=_Unlisted)
        man = w.manifest()
        assert man["appearance_provenance"]["redactor_applied_here"] == _Unlisted.label
        _stop(w)
        _refused_everything(_client(w.root), man, AP.WITHDRAWN)
        assert _summary(_client(w.root))["state"] == AP.WITHDRAWN

    @pytest.mark.parametrize("value", [None, "", "none", "unknown", 3])
    def test_a_walk_build_whose_redaction_is_unknown(self, tmp_path, value):
        """Never serve walk-time imagery whose redaction state is unknown."""
        w = _walk(tmp_path)
        man = w.manifest()
        root = AP.appearance_dir(w.store, WORLD, SESSION)
        doc = json.loads((root / "manifest.json").read_text())
        if value is None:
            del doc["appearance_provenance"]["redactor_applied_here"]
        else:
            doc["appearance_provenance"]["redactor_applied_here"] = value
        (root / "manifest.json").write_text(json.dumps(doc))
        _stop(w)
        _refused_everything(_client(w.root), man, AP.WITHDRAWN)

    def test_a_trusted_stored_bytes_build_after_a_relabel(self, tmp_path):
        w = World(tmp_path)
        assert w.build(redactor_factory=_never_redact).state == AP.STATE_OK
        man = w.manifest()
        _stop(w, UNGATED)
        _refused_everything(_client(w.root), man, AP.WITHDRAWN)

    def test_a_different_keyframe_set(self, tmp_path):
        """A re-redaction switch changes the pixels themselves (§6.5)."""
        w = _walk(tmp_path)
        man = w.manifest()
        root = AP.appearance_dir(w.store, WORLD, SESSION)
        doc = json.loads((root / "manifest.json").read_text())
        doc["appearance_provenance"]["keyframe_image_set"] = "reredacted:other"
        (root / "manifest.json").write_text(json.dumps(doc))
        _stop(w)
        _refused_everything(_client(w.root), man, AP.WITHDRAWN)

    def test_a_purged_world(self, tmp_path):
        w = _walk(tmp_path)
        man = w.manifest()
        _stop(w)
        record = w.store.read_world(WORLD)
        w.store.write_world(type(record)(**{**record.__dict__, "images_purged": True}))
        client = _client(w.root)
        assert client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest").status_code == 404
        assert _revision(client)["appearance"]["state"] == AP.WITHDRAWN
        assert client.get(
            f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{man['chunks'][0]['digest']}"
        ).status_code == 404
        # `build_appearance_config` has no purge check of its own (it never
        # had): the carry-over must not be what lets a purged world through.
        from tower.world_builder.appearance_render import (
            AppearanceViewerUnavailable,
            build_appearance_config,
        )
        with pytest.raises(AppearanceViewerUnavailable):
            build_appearance_config(w.store, WORLD, SESSION)
        assert not AP.may_serve(w.store, WORLD, SESSION, man)

    def test_a_tower_running_the_raw_research_bypass(self, tmp_path, monkeypatch):
        """TOWER_WORLD_RAW_IMAGERY on: a redacted walk build is still refused
        (imagery mismatch, checked first), after Stop as before it."""
        w = _walk(tmp_path)
        man = w.manifest()
        _stop(w)
        monkeypatch.setenv("TOWER_WORLD_RAW_IMAGERY", "1")
        assert RAWIMG.imagery_source_from_env() == RAWIMG.IMAGERY_RAW
        client = _client(w.root)
        r = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest")
        assert r.status_code == 404
        assert r.json()["detail"] == AP.imagery_mismatch_detail(man, RAWIMG.IMAGERY_RAW)
        assert _revision(client)["appearance"]["revision"] is None

    def test_a_raw_walk_build_under_the_product(self, tmp_path):
        """A manifest that says it is raw imagery never rides the carry-over."""
        w = _walk(tmp_path)
        man = w.manifest()
        root = AP.appearance_dir(w.store, WORLD, SESSION)
        doc = json.loads((root / "manifest.json").read_text())
        doc["imagery_source"] = RAWIMG.IMAGERY_RAW
        doc["appearance_provenance"]["imagery_source"] = RAWIMG.IMAGERY_RAW
        (root / "manifest.json").write_text(json.dumps(doc))
        _stop(w)
        client = _client(w.root)
        r = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest")
        assert r.status_code == 404
        assert r.json()["detail"] == AP.imagery_mismatch_detail(doc, RAWIMG.IMAGERY_REDACTED)
        assert client.get(
            f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{man['chunks'][0]['digest']}"
        ).status_code == 404

    def test_the_rule_is_the_open_pages_rule_and_nothing_wider(self, tmp_path):
        """`may_serve` is `label_matches` OR exactly the carry-over an open page
        is already given (`withdrawal_state == rebuilding`), under the imagery
        this Tower serves."""
        w = _walk(tmp_path)
        man = w.manifest()
        assert AP.may_serve(w.store, WORLD, SESSION, man)
        assert AP.label_matches(w.store, WORLD, SESSION, man)
        _stop(w)
        assert not AP.label_matches(w.store, WORLD, SESSION, man), "the label rule is unchanged"
        assert AP.withdrawal_state(w.store, WORLD, SESSION, man) == AP.REBUILDING
        assert AP.may_serve(w.store, WORLD, SESSION, man)
        assert not AP.may_serve(w.store, WORLD, SESSION, man,
                                imagery_source=RAWIMG.IMAGERY_RAW)


# ---------------------------------------------------------------------------
# an area (COMPONENTS §5.3): the same gate, through `area_appearance_servable`
# ---------------------------------------------------------------------------


def _area_walk(tmp_path, redactor=_FakeRedactor):
    """A walk whose area appearance was built under `none` (re-redacted)."""
    import shutil

    from tests.test_world_builder_components_areas import AREA1, _entries, write_components
    from tower.world_builder import components as C

    w = _walk(tmp_path, redactor)
    record = write_components(w.store, _entries(w.kids))
    view = C.AreaStore(w.store, WORLD, SESSION, AREA1)
    for stage in ("solve", "dense", "surface"):
        shutil.copytree(w.store.world_dir(WORLD) / stage / SESSION,
                        view.area_dir / stage / SESSION,
                        ignore=shutil.ignore_patterns(C.COMPONENTS_FILENAME))
    C.write_area_record(w.store, WORLD, SESSION, AREA1, components_sha1=record.sha1,
                        stage="surface", state="ok", levelled=True)
    result = AP.build_appearance(view, WORLD, SESSION,
                                 params=A.AppearanceParams.live(selection_samples=3000,
                                                                transient_detector="off"),
                                 device="cpu", redactor_factory=redactor)
    assert result.state == AP.STATE_OK, result.detail
    C.write_area_record(w.store, WORLD, SESSION, AREA1, components_sha1=record.sha1,
                        stage="appearance", state="ok")
    return w, AREA1


def _area_url(area_id, tail):
    return f"/worlds/{WORLD}/areas/{SESSION}/{area_id}/{tail}"


class TestAnArea:

    def test_a_walk_time_area_appearance_is_served_after_stop(self, tmp_path):
        from tests.test_world_builder_components_areas import _finalize

        w, area = _area_walk(tmp_path)
        _stop(w)
        _finalize(w)
        client = _client(w.root)
        r = client.get(_area_url(area, "appearance/manifest"))
        assert r.status_code == 200, r.text
        assert r.headers["x-world-redaction"] == f"none&{TRUSTED}"
        rev = client.get(_area_url(area, "render/revision")).json()
        assert rev["appearance"]["state"] == AP.SERVED
        assert rev["appearance"]["revision"] is not None

    def test_not_when_its_redactor_is_not_allowlisted(self, tmp_path):
        from tests.test_world_builder_components_areas import _finalize

        class _Unlisted(_FakeRedactor):
            label = "faces-blurred/some-other-detector"

        w, area = _area_walk(tmp_path, _Unlisted)
        _stop(w)
        _finalize(w)
        client = _client(w.root)
        r = client.get(_area_url(area, "appearance/manifest"))
        assert r.status_code == 404 and r.json()["detail"] == AP.STALE_LABEL_DETAIL
        rev = client.get(_area_url(area, "render/revision")).json()
        assert rev["appearance"] == {"revision": None, "current": False,
                                     "state": AP.WITHDRAWN, "epoch": None}


# ---------------------------------------------------------------------------
# fix round 1 (review rv-pas): fail closed on unreadable or partial metadata
# (MED-1), no new serving for a Stop to a label off the allowlist (LOW-2), and
# no superseded-file lending under the carry-over (LOW-3). Each refusal must be
# exactly dc35d51's answer: 404 on every route, `state: rebuilding`.
# ---------------------------------------------------------------------------


def _as_dc35d51(w, man):
    client = _client(w.root)
    m = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest")
    assert m.status_code == 404 and m.json()["detail"] == AP.STALE_LABEL_DETAIL
    for chunk in man["chunks"]:
        assert client.get(
            f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{chunk['digest']}").status_code == 404
    rev = _revision(client)
    assert rev["appearance"] == {"revision": None, "current": False,
                                 "state": AP.REBUILDING, "epoch": None}
    assert rev["representation"] == "surface"
    if hasattr(AP, "may_serve"):          # absent on dc35d51, where the routes run alone
        assert not AP.may_serve(w.store, WORLD, SESSION, man)


class TestFailClosed:

    def test_an_unreadable_session_record(self, tmp_path):
        w = _walk(tmp_path)
        man = w.manifest()
        _stop(w)
        w.store.session_path(WORLD, SESSION).write_text("{not json", encoding="utf-8")
        _as_dc35d51(w, man)

    def test_a_session_record_of_another_schema(self, tmp_path):
        w = _walk(tmp_path)
        man = w.manifest()
        _stop(w)
        p = w.store.session_path(WORLD, SESSION)
        doc = json.loads(p.read_text(encoding="utf-8"))
        doc["schema_version"] = 999
        p.write_text(json.dumps(doc), encoding="utf-8")
        _as_dc35d51(w, man)

    @pytest.mark.parametrize("drop", ["session_redaction", "keyframe_image_set",
                                      "imagery_source", "label_trusted", "all-but-redactor"])
    def test_a_partial_provenance(self, tmp_path, drop):
        w = _walk(tmp_path)
        root = AP.appearance_dir(w.store, WORLD, SESSION)
        doc = json.loads((root / "manifest.json").read_text())
        prov = doc["appearance_provenance"]
        keys = ([k for k in list(prov) if k != "redactor_applied_here"]
                if drop == "all-but-redactor" else [drop])
        for k in keys:
            prov.pop(k, None)
        doc.pop("imagery_source", None) if drop in ("imagery_source", "all-but-redactor") else None
        (root / "manifest.json").write_text(json.dumps(doc))
        _stop(w)
        _as_dc35d51(w, w.manifest())

    @pytest.mark.parametrize("value", [None, 0, "false"])
    def test_a_provenance_whose_trust_flag_is_not_exactly_false(self, tmp_path, value):
        w = _walk(tmp_path)
        root = AP.appearance_dir(w.store, WORLD, SESSION)
        doc = json.loads((root / "manifest.json").read_text())
        doc["appearance_provenance"]["label_trusted"] = value
        (root / "manifest.json").write_text(json.dumps(doc))
        _stop(w)
        assert not AP.may_serve(w.store, WORLD, SESSION, w.manifest())

    def test_an_unreadable_world_record(self, tmp_path):
        """Review mutant R1: the world record must read, or nothing rides the
        carry-over -- including `build_appearance_config`, which has no purge
        check of its own."""
        from tower.world_builder.appearance_render import (
            AppearanceViewerUnavailable,
            build_appearance_config,
        )

        w = _walk(tmp_path)
        man = w.manifest()
        _stop(w)
        assert AP.may_serve(w.store, WORLD, SESSION, man)
        w.store.world_path(WORLD).write_text("{torn", encoding="utf-8")
        assert not AP.may_serve(w.store, WORLD, SESSION, man)
        with pytest.raises(AppearanceViewerUnavailable):
            build_appearance_config(w.store, WORLD, SESSION)

    def test_an_unreadable_reredaction_pointer(self, tmp_path):
        w = _walk(tmp_path)
        man = w.manifest()
        _stop(w)
        w.store.redaction_set_path(WORLD, SESSION).write_text("{torn", encoding="utf-8")
        assert w.store.keyframe_image_set(WORLD, SESSION).cache_token is None, (
            "the store reads a torn pointer as no switch -- which is why the gate must not")
        _as_dc35d51(w, man)


class TestAStopToALabelOffTheAllowlist:

    @pytest.mark.parametrize("label", [
        "faces-detected-and-filled/yunet-2023mar@0.50",
        "faces-detected-and-filled/yunet-2023mar@0.30+plausibility2",   # RERUN, not trusted
        "x", "", "none"])
    def test_is_not_newly_served(self, tmp_path, label):
        w = _walk(tmp_path)
        man = w.manifest()
        _stop(w, label)
        if label == "none":
            # Not a label change at all, so `label_matches` still serves it, as
            # during the walk and on dc35d51.
            assert AP.label_matches(w.store, WORLD, SESSION, man)
            return
        _as_dc35d51(w, man)

    @pytest.mark.parametrize("label", [TRUSTED, UNGATED,
                                       "faces-detected-and-filled/yunet-2023mar@0.30+plausibility1"])
    def test_every_allowlisted_label_is(self, tmp_path, label):
        w = _walk(tmp_path)
        _stop(w, label)
        _served_everything(_client(w.root), w.manifest())


class TestNoLendingUnderTheCarryOver:

    def test_an_unlisted_redactors_superseded_chunks_are_not_served_after_stop(self, tmp_path):
        class _Unlisted(_FakeRedactor):
            label = "faces-blurred/some-other-detector"

        w = World(tmp_path, label="none")
        _records_carry(w, _Unlisted())
        assert w.build(params=A.AppearanceParams.live(selection_samples=3000,
                                                       transient_detector="off"),
                       redactor_factory=_Unlisted).state == AP.STATE_OK
        m1 = w.manifest()
        _records_carry(w, _FakeRedactor())
        assert w.build(params=A.AppearanceParams.live(selection_samples=3001,
                                                       transient_detector="off"),
                       redactor_factory=_FakeRedactor).state == AP.STATE_OK
        m2 = w.manifest()
        old = [c["digest"] for c in m1["chunks"]
               if c["digest"] not in {x["digest"] for x in m2["chunks"]}]
        assert old, "the two builds share every chunk; the test would prove nothing"
        client = _client(w.root)
        # during the walk the lend works, as on dc35d51 (same label, same set)
        assert all(client.get(f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{d}").status_code
                   == 200 for d in old)
        _stop(w)
        client = _client(w.root)
        assert client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest").status_code == 200
        assert all(client.get(f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{d}").status_code
                   == 404 for d in old)
        for c in m2["chunks"]:
            assert client.get(
                f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{c['digest']}").status_code == 200


def test_a_provenance_that_says_raw_under_a_redacted_top_level_is_refused(tmp_path):
    """The top-level `imagery_source` is what `imagery_matches` reads; the
    provenance's own must say `redacted` too, or nothing rides the carry-over.
    Two guards hold this (`_carry_over_metadata_whole` and the carry-over's own
    imagery comparison); this test is what notices if both go."""
    w = _walk(tmp_path)
    root = AP.appearance_dir(w.store, WORLD, SESSION)
    doc = json.loads((root / "manifest.json").read_text())
    assert doc["imagery_source"] == RAWIMG.IMAGERY_REDACTED
    doc["appearance_provenance"]["imagery_source"] = RAWIMG.IMAGERY_RAW
    (root / "manifest.json").write_text(json.dumps(doc))
    _stop(w)
    client = _client(w.root)
    assert client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest").status_code == 404
    assert _revision(client)["appearance"]["revision"] is None
    assert not AP.may_serve(w.store, WORLD, SESSION, w.manifest())


# ---------------------------------------------------------------------------
# fix round 2 (Codex cross-review MED-1/MED-2/LOW-3; re-check LOW-A/B): ANY
# re-redaction pointer on disk keeps the walk build off the carry-over -- a
# non-whole one, a revert, a switch -- and a pointer or label change between the
# gate and the read is caught before the bytes leave.
# ---------------------------------------------------------------------------


def _pointer(w, doc):
    w.store.redaction_set_path(WORLD, SESSION).write_text(
        doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")


def _a_whole_switch(w, name="images.redacted-x"):
    import shutil

    sd = w.store.session_dir(WORLD, SESSION)
    shutil.copytree(sd / "images", sd / name)
    _pointer(w, {"active": name, "redaction": TRUSTED, "set_digest": "d1",
                 "stored_redaction": TRUSTED})
    assert w.store.keyframe_image_set(WORLD, SESSION).cache_token == f"{name}@d1"


class TestAnyReredactionPointerAfterStop:

    @pytest.mark.parametrize("doc", [
        {},                                                                # empty object
        {"active": None},                                                  # explicit revert
        {"active": "images.redacted-gone", "redaction": TRUSTED,           # directory missing
         "set_digest": "d1", "stored_redaction": TRUSTED},
        {"active": "images.redacted-x", "set_digest": "d1",                # label missing
         "stored_redaction": TRUSTED},
        {"active": "../../../images", "redaction": TRUSTED,                # path traversal
         "set_digest": "d1", "stored_redaction": TRUSTED},
        {"active": "images.redacted-x", "redaction": TRUSTED,              # stale stored label
         "set_digest": "d1", "stored_redaction": "none"},
    ], ids=["empty", "revert", "dir-missing", "label-missing", "traversal", "stale-stored"])
    def test_a_pointer_the_store_does_not_honour(self, tmp_path, doc):
        import shutil

        w = _walk(tmp_path)
        man = w.manifest()
        _stop(w)
        sd = w.store.session_dir(WORLD, SESSION)
        shutil.copytree(sd / "images", sd / "images.redacted-x")
        _pointer(w, doc)
        assert w.store.keyframe_image_set(WORLD, SESSION).cache_token is None, (
            "the store reads every one of these as no switch -- the gate must not")
        _as_dc35d51(w, man)

    def test_a_switch_after_stop_then_its_revert(self, tmp_path):
        """Codex MED-2: the revert returns the set token to None, the walk build's
        own; dc35d51 still refuses it (its label differs), and so must we."""
        w = _walk(tmp_path)
        man = w.manifest()
        _stop(w)
        _a_whole_switch(w)
        client = _client(w.root)
        assert client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest").status_code == 404
        assert _revision(client)["appearance"]["state"] == AP.WITHDRAWN
        _pointer(w, {"active": None})                                      # the revert
        assert w.store.keyframe_image_set(WORLD, SESSION).cache_token is None
        _as_dc35d51(w, man)


class TestTheGateIsCheckedAgainAfterTheRead:
    """Codex LOW-3: a set-pointer or label change after the gate authorised a
    request but before the bytes were read. The file routes re-run the gate after
    the read and answer 404 when it no longer holds (or names another build)."""

    def _racing(self, monkeypatch, change):
        real = AP.read_appearance_file
        fired = []

        def read_then_change(*a, **k):
            data = real(*a, **k)
            if not fired:
                fired.append(True)
                change()
            return data

        monkeypatch.setattr(AP, "read_appearance_file", read_then_change)
        return fired

    def test_a_switch_between_the_gate_and_the_read_of_a_walk_build(self, tmp_path, monkeypatch):
        w = _walk(tmp_path)
        man = w.manifest()
        _stop(w)
        fired = self._racing(monkeypatch, lambda: _a_whole_switch(w))
        r = _client(w.root).get(
            f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{man['chunks'][0]['digest']}")
        assert fired and r.status_code == 404
        _assert_private(r)

    def test_a_relabel_between_the_gate_and_the_read_of_a_trusted_build(self, tmp_path,
                                                                       monkeypatch):
        w = World(tmp_path)
        assert w.build(redactor_factory=_never_redact).state == AP.STATE_OK
        man = w.manifest()
        fired = self._racing(monkeypatch, lambda: _stop(w, UNGATED))
        r = _client(w.root).get(
            f"/worlds/{WORLD}/appearance/{SESSION}/proxy/{man['proxy']['digest']}")
        assert fired and r.status_code == 404

    def test_a_final_build_publishing_between_the_gate_and_the_read(self, tmp_path,
                                                                    monkeypatch):
        """The walk build's chunk was authorised; the final build (trusted
        label, stored bytes) publishes before the bytes leave. The final build
        never lends the walk build's files (another label), so: 404."""
        w = _walk(tmp_path)
        man = w.manifest()
        _stop(w)

        def final_build():
            _records_carry(w)
            assert w.build(params=A.AppearanceParams(selection_samples=4000,
                                                     transient_detector="off"),
                           redactor_factory=_never_redact).state == AP.STATE_OK

        fired = self._racing(monkeypatch, final_build)
        digest = man["chunks"][0]["digest"]
        r = _client(w.root).get(f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{digest}")
        assert fired and w.manifest()["build_id"] != man["build_id"]
        assert digest not in {c["digest"] for c in w.manifest()["chunks"]}
        assert r.status_code == 404

    def test_a_live_publish_between_the_gate_and_the_read_still_lends(self, tmp_path,
                                                                      monkeypatch):
        """During the walk the same race is a live build replacing a live build
        under the same label: the old chunk is lent through the grace, exactly
        as on dc35d51."""
        w = World(tmp_path, label="none")
        _records_carry(w, _FakeRedactor())
        params = A.AppearanceParams.live(selection_samples=3000, transient_detector="off")
        assert w.build(params=params, redactor_factory=_FakeRedactor).state == AP.STATE_OK
        man = w.manifest()

        def next_live_build():
            img = w.render(3)
            img[10:30, 10:30] = (200, 40, 40)
            w.set_image(3, img)
            _records_carry(w, _FakeRedactor())
            assert w.build(params=params, redactor_factory=_FakeRedactor).state == AP.STATE_OK

        fired = self._racing(monkeypatch, next_live_build)
        client = _client(w.root)
        codes = {c["digest"]: client.get(
            f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{c['digest']}").status_code
            for c in man["chunks"]}
        new = {c["digest"] for c in w.manifest()["chunks"]}
        gone = [d for d in codes if d not in new]
        assert fired and w.manifest()["build_id"] != man["build_id"]
        assert gone, "the rebuild replaced no chunk; the test would prove nothing"
        # The first request raced the publish; whichever chunk it was, every
        # old chunk -- including the ones only the grace lends -- is served.
        assert set(codes.values()) == {200}, codes

    def test_no_change_still_serves(self, tmp_path, monkeypatch):
        w = _walk(tmp_path)
        man = w.manifest()
        _stop(w)
        fired = self._racing(monkeypatch, lambda: None)
        r = _client(w.root).get(
            f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{man['chunks'][0]['digest']}")
        assert fired and r.status_code == 200

    def test_an_area_switch_between_the_gate_and_the_read(self, tmp_path, monkeypatch):
        from tests.test_world_builder_components_areas import _finalize
        from tower.world_builder import components as C

        w, area = _area_walk(tmp_path)
        _stop(w)
        _finalize(w)
        man = AP.read_appearance_manifest(C.AreaStore(w.store, WORLD, SESSION, area),
                                          WORLD, SESSION)
        fired = self._racing(monkeypatch, lambda: _a_whole_switch(w))
        r = _client(w.root).get(_area_url(area, f"appearance/chunk/{man['chunks'][0]['digest']}"))
        assert fired and r.status_code == 404


@pytest.mark.parametrize("value", [None, "", "garbage", "Redacted"])
def test_a_provenance_imagery_source_that_is_not_exactly_redacted_is_refused(tmp_path, value):
    """Re-check LOW-B: `imagery_source_of` reads None, "" or an unknown string as
    `redacted`, so the carry-over's own comparison passes them; only the
    provenance guard refuses them (mutant F6)."""
    w = _walk(tmp_path)
    root = AP.appearance_dir(w.store, WORLD, SESSION)
    doc = json.loads((root / "manifest.json").read_text())
    doc["appearance_provenance"]["imagery_source"] = value
    (root / "manifest.json").write_text(json.dumps(doc))
    _stop(w)
    _as_dc35d51(w, w.manifest())
