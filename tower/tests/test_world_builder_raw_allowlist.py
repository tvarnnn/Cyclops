"""The read-path raw allowlist: `TOWER_WORLD_RAW_IMAGERY_WORLDS` (manager 201 §1, option C).

Home worlds shown raw, the live walk redacted. A Tower serving the product
(`TOWER_WORLD_RAW_IMAGERY` off) serves a raw-local-research artifact for exactly
the world ids on the list, with its research marker, and for no other world. A
listed world's REDACTED artifact is still served. Builds never read the list, so
a build with it set is still redacted. A list that is not whole fails closed.

Contract: `docs/contracts/WORLD-BUILDER-APPEARANCE.md` §6.6 (*The read-path
allowlist*). Byte identity of everything the list must NOT move:
`test_world_builder_raw_allowlist_identity.py`.
"""

from __future__ import annotations

import logging
import shutil

import pytest

import tests.test_world_builder_appearance as TWA
from tests.test_world_builder_appearance import SESSION, World, _app, _never_redact
from tower.world_builder import appearance as A
from tower.world_builder import appearance_pipeline as AP
from tower.world_builder import raw_imagery as RAWIMG

RAW = RAWIMG.IMAGERY_RAW
RED = RAWIMG.IMAGERY_REDACTED
ENV = RAWIMG.RAW_IMAGERY_WORLDS_ENV
HEX = "9835c5180000400080000000000000aa"      # a whole world id: 32 lowercase hex
OTHER = "b2a75ab40000400080000000000000bb"
AREA = "a1a1a1a1a1a1a1a1"
VIEWER = "appearance-1"

# Every spelling that must fail CLOSED: nothing is served raw.
MALFORMED = [
    " ", "\t", "  \n",                       # whitespace only
    "zz", "not-a-world-id",                  # not hex
    HEX[:8], HEX[:-1], HEX + "0",            # partial / overlong
    HEX.upper(),                             # not the exact id
    f" {HEX}", f"{HEX} ", f"{OTHER}, {HEX}",  # spaces
    f"{HEX},", f",{HEX}", f"{OTHER},,{HEX}",  # empty entries
    f"{HEX},{HEX}",                          # duplicate
    f"{HEX};{OTHER}",                        # wrong separator
    f"{HEX},zz",                             # one bad entry refuses the good one
    f"{OTHER},{HEX}g",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(RAWIMG.RAW_IMAGERY_ENV, raising=False)
    monkeypatch.delenv(ENV, raising=False)
    RAWIMG._LOGGED.clear()
    yield
    RAWIMG._LOGGED.clear()


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The appearance suite's synthetic world, under a real (hex) world id."""
    monkeypatch.setattr(TWA, "WORLD", HEX)
    return World(tmp_path)


def _raw_params(**kw):
    kw.setdefault("selection_samples", 4000)
    return A.AppearanceParams(imagery_source=RAW, **kw)


def _raw_world(world):
    TWA._paint_forbidden_sources(world)
    assert world.build(params=_raw_params(), redactor_factory=_never_redact).state == AP.STATE_OK
    assert AP.imagery_source_of(world.manifest()) == RAW
    return world


def _client(world):
    from fastapi.testclient import TestClient

    return TestClient(_app(world.root))


def _base(world_id=HEX):
    return f"/worlds/{world_id}/appearance/{SESSION}"


def _served_raw(client, world):
    """Every room route serves the raw artifact, and says what it is."""
    man = world.manifest()
    r = client.get(f"{_base()}/manifest")
    assert r.status_code == 200, r.text
    assert r.headers["x-world-imagery"] == RAW
    assert r.headers["x-world-imagery-warning"] == RAWIMG.RAW_NOTE
    assert r.headers["x-world-redaction"] == RAWIMG.RAW_LABEL
    assert r.headers["cache-control"] == "no-store"
    body = r.json()
    assert body["imagery_source"] == RAW and body["privacy_safe"] is False
    for c in man["chunks"]:
        rc = client.get(f"{_base()}/chunk/{c['digest']}")
        assert rc.status_code == 200
        assert rc.headers["x-world-imagery"] == RAW
        assert rc.headers["x-world-imagery-warning"] == RAWIMG.RAW_NOTE
    rp = client.get(f"{_base()}/proxy/{man['proxy']['digest']}")
    assert rp.status_code == 200 and rp.headers["x-world-imagery"] == RAW
    rev = client.get(f"/worlds/{HEX}/render/revision",
                     params={"session_id": SESSION, "viewer": VIEWER}).json()
    assert rev["appearance"]["state"] == AP.SERVED


def _refused_raw(client, world):
    man = world.manifest()
    r = client.get(f"{_base()}/manifest")
    assert r.status_code == 404
    assert r.json()["detail"] == AP.imagery_mismatch_detail(man, RED)
    for c in man["chunks"][:2]:
        assert client.get(f"{_base()}/chunk/{c['digest']}").status_code == 404
    assert client.get(f"{_base()}/proxy/{man['proxy']['digest']}").status_code == 404
    rev = client.get(f"/worlds/{HEX}/render/revision",
                     params={"session_id": SESSION, "viewer": VIEWER}).json()
    assert rev["appearance"]["state"] != AP.SERVED


# ---------------------------------------------------------------------------
# 1. the list itself
# ---------------------------------------------------------------------------


def test_unset_and_empty_are_the_default_and_list_nothing():
    assert RAWIMG.raw_imagery_worlds_from_env({}) == frozenset()
    assert RAWIMG.raw_imagery_worlds_from_env({ENV: ""}) == frozenset()


def test_a_whole_list_is_read_exactly():
    assert RAWIMG.raw_imagery_worlds_from_env({ENV: HEX}) == {HEX}
    assert RAWIMG.raw_imagery_worlds_from_env({ENV: f"{HEX},{OTHER}"}) == {HEX, OTHER}


@pytest.mark.parametrize("value", MALFORMED)
def test_a_list_that_is_not_whole_lists_nothing_and_is_logged_once(value, caplog):
    with caplog.at_level(logging.WARNING, logger=RAWIMG.__name__):
        for _ in range(3):
            assert RAWIMG.raw_imagery_worlds_from_env({ENV: value}) == frozenset()
            assert RAWIMG.reader_imagery_source(HEX, RAW, {ENV: value}) == RED
    refusals = [r for r in caplog.records if ENV in r.getMessage()
                and "NO world is served raw" in r.getMessage()]
    assert len(refusals) == 1


def test_the_reader_answer_over_every_case():
    listed = {ENV: HEX}
    research = {RAWIMG.RAW_IMAGERY_ENV: "1"}
    # product Tower: raw only for a listed world's raw artifact
    assert RAWIMG.reader_imagery_source(HEX, RAW, listed) == RAW
    assert RAWIMG.reader_imagery_source(HEX, RED, listed) == RED
    assert RAWIMG.reader_imagery_source(OTHER, RAW, listed) == RED
    assert RAWIMG.reader_imagery_source(HEX, RAW, {}) == RED
    assert RAWIMG.reader_imagery_source(None, RAW, listed) == RED
    # a world id that is not a string is refused, never raised on
    assert RAWIMG.reader_imagery_source([HEX], RAW, listed) == RED
    assert RAWIMG.reader_imagery_source(HEX[:8], RAW, listed) == RED
    # research Tower: unchanged, the list is not consulted
    for env in (research, dict(research, **listed), dict(research, **{ENV: "zz"})):
        for wid in (HEX, OTHER):
            for art in (RAW, RED):
                assert RAWIMG.reader_imagery_source(wid, art, env) == RAW


def test_builds_never_read_the_list():
    """What a build reads is `imagery_source_from_env`; the list never moves it."""
    for value in (HEX, f"{HEX},{OTHER}", "zz"):
        assert RAWIMG.imagery_source_from_env({ENV: value}) == RED
    assert RAWIMG.imagery_source_from_env({ENV: HEX, RAWIMG.RAW_IMAGERY_ENV: "1"}) == RAW


# ---------------------------------------------------------------------------
# 2. a listed raw world is served, with its marker
# ---------------------------------------------------------------------------


def test_a_listed_raw_world_is_served_with_its_marker(world, monkeypatch):
    from tower.results.world_builder_library import _appearance_summary
    from tower.world_builder import appearance_render as AR

    _raw_world(world)
    client = _client(world)
    _refused_raw(client, world)                     # the default: refused
    monkeypatch.setenv(ENV, f"{OTHER},{HEX}")
    _served_raw(client, world)
    # the page is told, and draws the marker it always draws for raw
    config = AR.build_appearance_config(world.store, HEX, SESSION)
    assert config["imagery_source"] == RAW
    assert config["privacy_safe"] is False
    assert config["redaction_effective"] == RAWIMG.RAW_LABEL
    page = client.get(f"/worlds/{HEX}/render", params={"session_id": SESSION, "viewer": VIEWER})
    assert page.status_code == 200
    assert '<meta name="wb-representation" content="appearance">' in page.text
    # the listing never presents it as the product
    summary = _appearance_summary(world.store, HEX, SESSION, world.store.read_world(HEX))
    assert summary["state"] == AP.SERVED
    assert summary["imagery_source"] == RAW and summary["privacy_safe"] is False
    assert summary["imagery"] == RAWIMG.RAW_NOTE


def test_serving_a_listed_world_raw_is_logged_once(world, monkeypatch, caplog):
    _raw_world(world)
    monkeypatch.setenv(ENV, HEX)
    with caplog.at_level(logging.WARNING, logger=RAWIMG.__name__):
        for _ in range(3):
            assert AP.may_serve(world.store, HEX, SESSION, world.manifest())
    served = [r for r in caplog.records if "listed world" in r.getMessage()]
    assert len(served) == 1 and HEX in served[0].getMessage()


# ---------------------------------------------------------------------------
# 3. an unlisted raw world is still refused; a malformed list fails closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", OTHER, f"{OTHER},{OTHER[:-1]}c"])
def test_an_unlisted_raw_world_is_refused(world, monkeypatch, value):
    _raw_world(world)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    _refused_raw(_client(world), world)
    assert not AP.may_serve(world.store, HEX, SESSION, world.manifest())
    assert not AP.label_matches(world.store, HEX, SESSION, world.manifest())


@pytest.mark.parametrize("value", MALFORMED)
def test_a_malformed_list_fails_closed(world, monkeypatch, value):
    _raw_world(world)
    monkeypatch.setenv(ENV, value)
    _refused_raw(_client(world), world)


def test_the_list_widens_nothing_but_the_imagery_check(world, monkeypatch):
    """A listed raw build is still held to the label and keyframe set: relabel the
    session and it is refused, exactly as a raw build on a research Tower is."""
    _raw_world(world)
    monkeypatch.setenv(ENV, HEX)
    assert AP.may_serve(world.store, HEX, SESSION, world.manifest())
    world.set_label(TWA.UNGATED)
    assert not AP.may_serve(world.store, HEX, SESSION, world.manifest())
    assert _client(world).get(f"{_base()}/manifest").json()["detail"] == AP.STALE_LABEL_DETAIL


def test_a_listed_purged_world_is_not_served(world, monkeypatch):
    _raw_world(world)
    monkeypatch.setenv(ENV, HEX)
    record = world.store.read_world(HEX)
    world.store.write_world(type(record)(**{**record.__dict__, "images_purged": True}))
    r = _client(world).get(f"{_base()}/manifest")
    assert (r.status_code, r.json()["detail"]) == (404, "appearance imagery was purged")


# ---------------------------------------------------------------------------
# 4. redacted worlds are unchanged, listed or not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, HEX, OTHER, "zz"])
def test_a_redacted_world_is_served_listed_or_not(world, monkeypatch, value):
    assert world.build(redactor_factory=_never_redact).state == AP.STATE_OK
    if value is not None:
        monkeypatch.setenv(ENV, value)
    r = _client(world).get(f"{_base()}/manifest")
    assert r.status_code == 200
    assert r.headers["x-world-imagery"] == RED
    assert "x-world-imagery-warning" not in r.headers
    assert AP.may_serve(world.store, HEX, SESSION, world.manifest())


def test_a_listed_world_serves_whichever_artifact_its_manifest_holds(world, monkeypatch):
    """Raw, then redacted, then raw again over one session: the list never makes
    the redacted one unservable, and each is served under its own marker."""
    monkeypatch.setenv(ENV, HEX)
    client = _client(world)
    _raw_world(world)
    _served_raw(client, world)
    assert world.build(redactor_factory=_never_redact, force=True).state == AP.STATE_OK
    r = client.get(f"{_base()}/manifest")
    assert r.status_code == 200 and r.headers["x-world-imagery"] == RED
    assert world.build(params=_raw_params(), redactor_factory=_never_redact,
                       force=True).state == AP.STATE_OK
    _served_raw(client, world)


def test_a_research_tower_is_unchanged_by_the_list(world, monkeypatch):
    """`TOWER_WORLD_RAW_IMAGERY` on: raw served for every world, redacted refused,
    whatever the list says."""
    assert world.build(redactor_factory=_never_redact).state == AP.STATE_OK
    monkeypatch.setenv(RAWIMG.RAW_IMAGERY_ENV, "1")
    for value in (HEX, OTHER, "zz"):
        monkeypatch.setenv(ENV, value)
        assert _client(world).get(f"{_base()}/manifest").status_code == 404
    TWA._paint_forbidden_sources(world)
    assert world.build(params=_raw_params(), redactor_factory=_never_redact,
                       force=True).state == AP.STATE_OK
    for value in (HEX, OTHER, "zz"):
        monkeypatch.setenv(ENV, value)
        assert _client(world).get(f"{_base()}/manifest").status_code == 200


# ---------------------------------------------------------------------------
# 5. the area twins: the same gate
# ---------------------------------------------------------------------------


def _raw_area(world):
    from tests.test_world_builder_components_areas import _entries, write_components
    from tower.world_builder import components as C

    TWA._paint_forbidden_sources(world)
    record = write_components(world.store, _entries(world.kids), world_id=HEX)
    view = C.AreaStore(world.store, HEX, SESSION, AREA)
    for stage in ("solve", "dense", "surface"):
        shutil.copytree(world.store.world_dir(HEX) / stage / SESSION,
                        view.area_dir / stage / SESSION,
                        ignore=shutil.ignore_patterns(C.COMPONENTS_FILENAME))
    C.write_area_record(world.store, HEX, SESSION, AREA, components_sha1=record.sha1,
                        stage="surface", state="ok", levelled=True)
    result = AP.build_appearance(view, HEX, SESSION, params=_raw_params(), device="cpu",
                                 redactor_factory=_never_redact)
    assert result.state == AP.STATE_OK, result.detail
    C.write_area_record(world.store, HEX, SESSION, AREA, components_sha1=record.sha1,
                        stage="appearance", state="ok")
    return view


@pytest.mark.parametrize("value,served", [(None, False), (OTHER, False), ("zz", False),
                                          (f"{HEX} ", False), (HEX, True),
                                          (f"{OTHER},{HEX}", True)])
def test_an_area_is_served_raw_only_for_a_listed_world(world, monkeypatch, value, served):
    from tower.world_builder import components as C

    view = _raw_area(world)
    man = AP.read_appearance_manifest(view, HEX, SESSION)
    assert AP.imagery_source_of(man) == RAW
    if value is not None:
        monkeypatch.setenv(ENV, value)
    client = _client(world)
    base = f"/worlds/{HEX}/areas/{SESSION}/{AREA}/appearance"
    r = client.get(f"{base}/manifest")
    chunk = client.get(f"{base}/chunk/{man['chunks'][0]['digest']}")
    proxy = client.get(f"{base}/proxy/{man['proxy']['digest']}")
    servable, reason = C.area_appearance_servable(world.store, HEX, SESSION, AREA)
    if served:
        for resp in (r, chunk, proxy):
            assert resp.status_code == 200
            assert resp.headers["x-world-imagery"] == RAW
            assert resp.headers["x-world-imagery-warning"] == RAWIMG.RAW_NOTE
        assert servable is not None and reason is None
    else:
        for resp in (r, chunk, proxy):
            assert resp.status_code == 404
        assert servable is None and reason == AP.imagery_mismatch_detail(man, RED)


# ---------------------------------------------------------------------------
# 6. a build with the list set is still a redacted build
# ---------------------------------------------------------------------------


def test_a_build_with_the_list_set_is_still_redacted(world, monkeypatch):
    import builtins
    import pathlib

    TWA._paint_forbidden_sources(world)
    monkeypatch.setenv(ENV, HEX)
    opened = []
    real_open, real_path_open = builtins.open, pathlib.Path.open

    def spy_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    def spy_path_open(self, *a, **k):
        opened.append(str(self))
        return real_path_open(self, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(pathlib.Path, "open", spy_path_open)
    result = world.build(redactor_factory=_never_redact)
    monkeypatch.setattr(builtins, "open", real_open)
    monkeypatch.setattr(pathlib.Path, "open", real_path_open)
    assert result.state == AP.STATE_OK, result.detail
    man = world.manifest()
    assert man["imagery_source"] == RED and man["privacy_safe"] is True
    assert man["appearance_provenance"]["source"] != RAWIMG.SOURCE_RAW_LOCAL
    norm = [p.replace("\\", "/") for p in opened]
    assert [p for p in norm if "/captures/" in p or p.endswith("sources.json")] == []
    # and no magenta (only the original capture frames carry it) reached a texel
    for k in man["keyframes"]:
        rgba = world.decoded(k["ki"])
        assert not (TWA._near_colour(rgba[..., :3], TWA.MAGENTA) & TWA._opaque(rgba)).any()


def test_the_build_clis_resolve_redacted_with_the_list_set(monkeypatch):
    import argparse

    import scripts.world_surface as SURF

    monkeypatch.setenv(ENV, HEX)
    assert SURF._imagery_from_args(argparse.Namespace(imagery_source=None)) == RED
    assert RAWIMG.imagery_source_from_env() == RED
