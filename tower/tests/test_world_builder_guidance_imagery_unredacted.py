"""TOWER_WORLD_GUIDANCE_IMAGERY_UNREDACTED: the demo's deferred-privacy switch.

Off (anything but exactly `on`): today's imagery, byte for byte -- a session
whose keyframe set is labelled `none` is 404 and produces nothing. On: such a
session is served by both routes and both producers (B1 live, B2 landed incl.
the S3 dial) from its stored keyframe JPEGs, every entry saying
`"redaction": "unredacted"`. Any other untrusted label stays refused, and a
trusted session still says `redacted`.
"""

import hashlib

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tower.config import world_guidance_imagery_unredacted_setting
from tower.results import world_builder_guidance_imagery as route
from tower.world_builder import guidance_imagery as imagery
from tower.world_builder.appearance import RERUN_REDACTION_LABELS, TRUSTED_REDACTION_LABELS

ENV = "TOWER_WORLD_GUIDANCE_IMAGERY_UNREDACTED"
TRUSTED = sorted(TRUSTED_REDACTION_LABELS)[0]
RERUN = sorted(RERUN_REDACTION_LABELS)[0]
UNTRUSTED = ["faces-blurred", RERUN, "None", "NONE", "none ", "", None]
KIDS = ["s9:00004417", "s9:00004418", "s9:00004419"]
RECEIPT = {"solved_at": 1791240313.0, "geometry_revision": "g-1", "frame_revision": 1}
PATH = "/worlds/home/guidance/s9/imagery"


def _jpeg(seed, width=320, height=180):
    image = np.random.default_rng(seed).integers(0, 255, (height, width, 3), dtype=np.uint8)
    ok, data = cv2.imencode(".jpg", image)
    assert ok
    return data.tobytes()


class _Row:
    def __init__(self, kid, segment):
        self.keyframe_id, self.segment_index = kid, segment


class _ImageSet:
    def __init__(self, directory, redaction):
        self.directory, self.redaction = directory, redaction


class _Store:
    directory = None
    redaction = "none"

    def __init__(self, root):
        pass

    def read_session(self, world, session):
        return type("Session", (), {"world_id": world})()

    def read_keyframes(self, world, session):
        return [_Row(kid, 812) for kid in KIDS]

    def read_derived(self, world, session, verify=False):
        return {"poses": [{"keyframe_id": kid, "segment_index": 812, "status": "solved"}
                          for kid in KIDS]}

    def keyframe_image_set(self, world, session):
        return _ImageSet(self.directory, self.redaction)


@pytest.fixture
def state(monkeypatch, tmp_path):
    """Real trust (`label_is_trusted` is NOT patched), real tile reads from
    stored JPEGs on disk, a fake store around them."""
    for i, kid in enumerate(KIDS):
        (tmp_path / f"{kid.rsplit(':', 1)[-1]}.jpg").write_bytes(_jpeg(i))
    monkeypatch.setattr(_Store, "directory", tmp_path)
    monkeypatch.setattr(_Store, "redaction", "none")
    monkeypatch.setattr(imagery, "WorldStore", _Store)
    monkeypatch.setattr(imagery, "contained_world_id", lambda store, world: world)
    monkeypatch.setattr(imagery, "contained_session_id", lambda store, world, session: session)
    monkeypatch.setattr(route, "store_from_root", _Store)
    monkeypatch.setattr(route, "contained_world_id", lambda store, world: world)
    monkeypatch.setattr(route, "contained_session_id", lambda store, world, session: session)
    monkeypatch.delenv(ENV, raising=False)
    fresh = imagery.ImageryState()
    monkeypatch.setattr(route, "state_for", lambda root: fresh)
    # Warm cv2 so the first real landing is not charged its import-time cost
    # against the 80 ms budget.
    imagery._tile(None, "home", "s9", KIDS[0], _ImageSet(tmp_path, TRUSTED))
    return fresh


def _client(monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    app = FastAPI()
    app.state.world_root = "synthetic-root"
    app.include_router(route.router)
    return TestClient(app)


def _dial():
    return {"reference_segment": 17,
            "sectors": ((2, (KIDS[2], KIDS[1])), (7, (KIDS[0],))),
            "latest": {"component_reference_segment": 17, "sector": 2, "keyframe_id": KIDS[2]},
            "motion_sector": 2}


def _land(state, dial=True):
    if dial:
        state.publish_landed("root", "home", "s9", RECEIPT, [], dial=_dial())
    else:
        state.publish_landed("root", "home", "s9", RECEIPT, [(kid, 17) for kid in KIDS])


# -- the setting ---------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("off", False), ("1", False), ("true", False),
    ("ON", False), (" on", False), ("on", True)])
def test_setting_is_exactly_on(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert world_guidance_imagery_unredacted_setting() is expected


# -- off: today's Tower ----------------------------------------------------------


@pytest.mark.parametrize("value", [None, "off", "1", "true", "ON"])
def test_off_none_session_is_404_and_produces_nothing(state, monkeypatch, value):
    client = _client(monkeypatch)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert client.get(PATH + "/manifest").status_code == 404
    _land(state)
    assert state.landed("home", "s9") is None
    assert state.live("root", "home", "s9")["entries"] == []
    for kid in KIDS:
        assert imagery._tile(None, "home", "s9", kid, _ImageSet(_Store.directory, "none")) is None


def test_trusted_manifest_is_identical_with_switch_on_or_off(state, monkeypatch):
    monkeypatch.setattr(_Store, "redaction", TRUSTED)
    monkeypatch.setattr(imagery.time, "monotonic", lambda: 100.0)  # no second tile
    client = _client(monkeypatch)
    off_live = client.get(PATH + "/manifest")
    _land(state)
    off = client.get(PATH + "/manifest")
    monkeypatch.setenv(ENV, "on")
    on = client.get(PATH + "/manifest")
    assert off_live.status_code == off.status_code == on.status_code == 200
    assert off.content == on.content
    payload = on.json()
    assert payload["live"]["entries"] and payload["landed"]["entries"]
    assert {e["redaction"] for e in payload["live"]["entries"] +
            payload["landed"]["entries"]} == {"redacted"}
    _land(state)  # a fresh landing under the switch is also "redacted"
    assert {e["redaction"] for e in state.landed("home", "s9")["entries"]} == {"redacted"}


# -- on + `none`: served, labelled unredacted ------------------------------------


def test_on_none_session_live_manifest_and_tile(state, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    monkeypatch.setattr(imagery, "keyframe_image_bytes",
                        lambda *a, **kw: pytest.fail("the redaction boundary re-redacts `none`"))
    client = _client(monkeypatch)
    reply = client.get(PATH + "/manifest")
    assert reply.status_code == 200
    live = reply.json()["live"]
    assert live["segment_index"] == 812
    assert [e["keyframe_id"] for e in live["entries"]] == [KIDS[-1]]  # one new tile per second
    entry = live["entries"][0]
    assert entry["redaction"] == "unredacted"
    assert entry["pose_source"] == "segment_local"
    tile = client.get(PATH + "/tile/" + entry["digest"])
    assert tile.status_code == 200
    assert tile.headers["content-type"] == "image/jpeg"
    assert tile.headers["cache-control"] == "no-store"
    assert hashlib.sha256(tile.content).hexdigest() == entry["digest"]
    assert len(tile.content) <= 16 * 1024
    decoded = cv2.imdecode(np.frombuffer(tile.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[0] <= 90 and decoded.shape[1] <= 160
    # Switch off again at runtime: the same session and tile are refused.
    monkeypatch.setenv(ENV, "off")
    assert client.get(PATH + "/manifest").status_code == 404
    assert client.get(PATH + "/tile/" + entry["digest"]).status_code == 404


def test_on_none_tile_is_the_stored_keyframe_thumbnail(state, monkeypatch, tmp_path):
    """The bytes read are `<set directory>/<seq>.jpg` exactly as persisted."""
    monkeypatch.setenv(ENV, "on")
    unredacted = imagery._tile(None, "home", "s9", KIDS[1], _ImageSet(tmp_path, "none"))
    redacted = imagery._tile(None, "home", "s9", KIDS[1], _ImageSet(tmp_path, TRUSTED))
    # Same stored file, same thumbnail pipeline: identical bytes, only the label differs.
    assert unredacted is not None and unredacted == redacted
    assert imagery._tile(None, "home", "s9", "s9:missing", _ImageSet(tmp_path, "none")) is None
    assert imagery._tile(None, "home", "s9", "s9:..\\x", _ImageSet(tmp_path, "none")) is None


@pytest.mark.parametrize("dial", [True, False])
def test_on_none_session_landed_manifest_and_tiles(state, monkeypatch, dial):
    monkeypatch.setenv(ENV, "on")
    client = _client(monkeypatch)
    _land(state, dial=dial)
    reply = client.get(PATH + "/manifest")
    assert reply.status_code == 200
    landed = reply.json()["landed"]
    assert landed["entries"]
    assert all(e["redaction"] == "unredacted" and e["pose_source"] == "landed_global_solve"
               and e["component_reference_segment"] == 17 and e["segment_index"] is None
               for e in landed["entries"])
    if dial:
        assert [(e["sector"], e["keyframe_id"]) for e in landed["entries"]] == [
            (2, KIDS[2]), (7, KIDS[0])]
        assert landed["motion_sector"] == 2
        assert landed["latest"]["keyframe_id"] == KIDS[2]
    for entry in landed["entries"]:
        tile = client.get(PATH + "/tile/" + entry["digest"])
        assert tile.status_code == 200
        assert hashlib.sha256(tile.content).hexdigest() == entry["digest"]


def test_live_bytes_are_never_relabelled_across_a_label_change(state, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    clock = [100.0]
    monkeypatch.setattr(imagery.time, "monotonic", lambda: clock[0])
    first = state.live("root", "home", "s9")["entries"]
    assert [e["redaction"] for e in first] == ["unredacted"]
    monkeypatch.setattr(_Store, "redaction", TRUSTED)
    clock[0] += .5  # inside the one-new-tile-per-second budget: no re-read yet
    assert state.live("root", "home", "s9")["entries"] == []
    clock[0] += 1.0
    assert [e["redaction"] for e in state.live("root", "home", "s9")["entries"]] == ["redacted"]


# -- on + any other untrusted label: refused exactly as today ------------------------


@pytest.mark.parametrize("label", UNTRUSTED)
def test_on_other_untrusted_labels_still_refused(state, monkeypatch, label):
    monkeypatch.setenv(ENV, "on")
    monkeypatch.setattr(_Store, "redaction", label)
    client = _client(monkeypatch)
    assert client.get(PATH + "/manifest").status_code == 404
    _land(state)
    assert state.landed("home", "s9") is None
    assert state.live("root", "home", "s9")["entries"] == []
    assert imagery.entry_redaction(label) is None
    assert imagery._tile(None, "home", "s9", KIDS[0], _ImageSet(_Store.directory, label)) is None


def test_on_tile_route_refuses_untrusted_label_even_for_a_cached_digest(state, monkeypatch):
    monkeypatch.setenv(ENV, "on")
    client = _client(monkeypatch)
    _land(state)
    digest = client.get(PATH + "/manifest").json()["landed"]["entries"][0]["digest"]
    assert client.get(PATH + "/tile/" + digest).status_code == 200
    monkeypatch.setattr(_Store, "redaction", RERUN)
    assert client.get(PATH + "/tile/" + digest).status_code == 404


# -- the real store: what `none` looks like ---------------------------------------


def test_a_fresh_real_session_is_labelled_the_literal_none(tmp_path, monkeypatch):
    import json

    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path / "worlds")
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world()
    session_id = engine.start_session(world_id, frame_source="synthetic")
    on_disk = json.loads((store.session_dir(world_id, session_id) / "session.json")
                         .read_text(encoding="utf-8"))
    assert on_disk["redaction"] == "none"
    image_set = store.keyframe_image_set(world_id, session_id)
    assert image_set.redaction == "none"
    assert image_set.directory == store.images_dir(world_id, session_id)
    store.write_keyframe_image(world_id, session_id, "00000007.jpg", _jpeg(7))
    kid = f"{session_id}:00000007"
    monkeypatch.delenv(ENV, raising=False)
    assert imagery._tile(store, world_id, session_id, kid, image_set) is None
    monkeypatch.setenv(ENV, "on")
    assert imagery._tile(store, world_id, session_id, kid, image_set) is not None
    engine.stop_session("stopped")
