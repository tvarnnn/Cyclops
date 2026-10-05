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
