"""Synthetic FOW B1/B2 wire, redaction, dating and cap fixtures."""

import hashlib
import os
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tower.results import world_builder_guidance_imagery as route
from tower.world_builder import guidance_imagery as imagery
from tower.world_builder import guidance_coverage as coverage


class _Row:
    def __init__(self, kid, segment):
        self.keyframe_id, self.segment_index = kid, segment


class _Store:
    rows = [_Row("s9:4417", 812)]
    poses = [{"keyframe_id": "s9:4417", "segment_index": 812, "status": "solved"}]
    redaction = "faces-blurred"

    def __init__(self, root):
        pass

    def read_session(self, world, session):
        return type("Session", (), {"world_id": world})()

    def read_keyframes(self, world, session):
        return self.rows

    def read_derived(self, world, session, verify=False):
        return {"poses": self.poses}

    def keyframe_image_set(self, world, session):
        return type("ImageSet", (), {"redaction": self.redaction})()


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(imagery, "WorldStore", _Store)
    monkeypatch.setattr(imagery, "contained_world_id", lambda store, world: world)
    monkeypatch.setattr(imagery, "contained_session_id", lambda store, world, session: session)
    monkeypatch.setattr(imagery, "label_is_trusted", lambda label: label == "faces-blurred")
    monkeypatch.setattr(route, "store_from_root", _Store)
    monkeypatch.setattr(route, "contained_world_id", lambda store, world: world)
    monkeypatch.setattr(route, "contained_session_id", lambda store, world, session: session)
    monkeypatch.setattr(route, "label_is_trusted", lambda label: label == "faces-blurred")
    _Store.rows = [_Row("s9:4417", 812)]
    _Store.poses = [{"keyframe_id": "s9:4417", "segment_index": 812, "status": "solved"}]
    _Store.redaction = "faces-blurred"
    state = imagery.ImageryState()
    monkeypatch.setattr(route, "state_for", lambda root: state)
    return state


def _client(monkeypatch, *, live=False, landed=False):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_IMAGERY_LIVE", "on" if live else "off")
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_IMAGERY_LANDED", "on" if landed else "off")
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    app = FastAPI()
    app.state.world_root = "synthetic-root"
    if live or landed:
        app.include_router(route.router)
    return TestClient(app)


def _jpeg(width=320, height=180):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 50
    ok, data = cv2.imencode(".jpg", image)
    assert ok
    return data.tobytes()


def test_b1_live_fixture_redacted_digest_caps_and_break(fake, monkeypatch):
    calls = []

    def boundary(*args, **kwargs):
        calls.append(kwargs["imagery_source"])
        return _jpeg(), "world-keyframe", None

    monkeypatch.setattr(imagery, "keyframe_image_bytes", boundary)
    monkeypatch.setenv("TOWER_WORLD_RAW_IMAGERY_WORLDS", "home")
    client = _client(monkeypatch, live=True)
    class GuardedEnvironment(dict):
        def get(self, key, default=None):
            assert key != "TOWER_WORLD_RAW_IMAGERY_WORLDS"
            return super().get(key, default)

        def __getitem__(self, key):
            assert key != "TOWER_WORLD_RAW_IMAGERY_WORLDS"
            return super().__getitem__(key)

    monkeypatch.setattr(os, "environ", GuardedEnvironment(os.environ))
    path = "/worlds/home/guidance/s9/imagery"
    first = client.get(path + "/manifest")
    assert first.status_code == 200
    live = first.json()["live"]
    assert first.json()["version"] == 1
    assert live["segment_index"] == 812
    assert len(live["entries"]) == 1
    entry = live["entries"][0]
    assert entry["keyframe_id"] == "s9:4417"
    assert entry["pose_source"] == "segment_local"
    assert entry["component_reference_segment"] is None
    assert entry["segment_index"] == 812
    assert entry["redaction"] == "redacted"
    assert "landed" not in first.json()
    assert calls == ["redacted"]  # raw allowlist was present but not consulted
    tile = client.get(path + "/tile/" + entry["digest"])
    assert tile.status_code == 200
    assert tile.headers["content-type"] == "image/jpeg"
    assert tile.headers["cache-control"] == "no-store"
    assert len(tile.content) <= 16 * 1024
    assert hashlib.sha256(tile.content).hexdigest() == entry["digest"]
    decoded = cv2.imdecode(np.frombuffer(tile.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[0] <= 90 and decoded.shape[1] <= 160
    _Store.rows = [_Row("s9:4417", 812), _Row("s9:4418", 813)]
    assert client.get(path + "/manifest").json()["live"] == {"segment_index": 813, "entries": []}
    assert client.get(path + "/tile/" + entry["digest"]).status_code == 404


def test_off_route_absence_and_404s(fake, monkeypatch):
    client = _client(monkeypatch)
    path = "/worlds/home/guidance/s9/imagery"
    assert path + "/manifest" not in client.get("/openapi.json").json()["paths"]
    assert client.get(path + "/manifest").status_code == 404
    assert client.get(path + "/tile/" + "a" * 64).status_code == 404
    client = _client(monkeypatch, landed=True)
    assert client.get(path + "/manifest").json() == {"version": 1, "landed": None}
    assert client.get(path + "/tile/" + "a" * 64).status_code == 404
    assert client.get("/worlds/no/guidance/s9/imagery/tile/" + "x" * 64).status_code == 404
    client = _client(monkeypatch, live=True)
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_IMAGERY_LIVE", "off")
    assert client.get(path + "/manifest").status_code == 404
    assert client.get(path + "/tile/" + "a" * 64).status_code == 404


def test_missing_store_is_404_not_5xx(fake, monkeypatch):
    from tower.world_builder.store import WorldStoreError
    monkeypatch.setattr(_Store, "read_session",
                        lambda self, world, session: (_ for _ in ()).throw(WorldStoreError("absent")))
    client = _client(monkeypatch, live=True)
    path = "/worlds/home/guidance/s9/imagery"
    assert client.get(path + "/manifest").status_code == 404
    assert client.get(path + "/tile/" + "a" * 64).status_code == 404


def test_session_path_is_contained(tmp_path):
    from tower.world_builder.store import WorldStore
    store = WorldStore(tmp_path)
    assert imagery.contained_session_id(store, "world", "session") == "session"
    assert imagery.contained_session_id(store, "world", "..\\other") is None
    assert imagery.contained_session_id(store, "world", "C:\\outside") is None


def test_real_app_does_not_register_imagery_when_off(monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_GUIDANCE_IMAGERY_LIVE", raising=False)
    monkeypatch.delenv("TOWER_WORLD_GUIDANCE_IMAGERY_LANDED", raising=False)
    from tower.main import create_app
    app = create_app()
    paths = app.openapi()["paths"]
    assert not any("/guidance/{session_id}/imagery/" in path for path in paths)


@pytest.mark.parametrize("switch", ["LIVE", "LANDED"])
def test_real_app_registers_imagery_when_one_stage_is_on(monkeypatch, switch):
    monkeypatch.delenv("TOWER_WORLD_GUIDANCE_IMAGERY_LIVE", raising=False)
    monkeypatch.delenv("TOWER_WORLD_GUIDANCE_IMAGERY_LANDED", raising=False)
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_IMAGERY_" + switch, "on")
    from tower.main import create_app
    paths = create_app().openapi()["paths"]
    prefix = "/worlds/{world_id}/guidance/{session_id}/imagery/"
    assert prefix + "manifest" in paths
    assert prefix + "tile/{digest}" in paths


def test_b1_no_untrusted_or_failed_redaction(fake, monkeypatch):
    monkeypatch.setattr(imagery, "keyframe_image_bytes", lambda *a, **kw: (None, "failed", None))
    client = _client(monkeypatch, live=True)
    path = "/worlds/home/guidance/s9/imagery/manifest"
    assert client.get(path).json()["live"]["entries"] == []
    _Store.redaction = "none"
    assert client.get(path).status_code == 404


def test_real_redaction_boundary_does_not_read_raw_allowlist(tmp_path, monkeypatch):
    from tower.world_builder.appearance import TRUSTED_REDACTION_LABELS
    label = next(iter(TRUSTED_REDACTION_LABELS))
    (tmp_path / "4417.jpg").write_bytes(_jpeg())
    image_set = SimpleNamespace(directory=tmp_path, redaction=label)
    monkeypatch.setenv("TOWER_WORLD_RAW_IMAGERY_WORLDS", "a" * 32)
    class GuardedEnvironment(dict):
        def get(self, key, default=None):
            assert key != "TOWER_WORLD_RAW_IMAGERY_WORLDS"
            return super().get(key, default)

        def __getitem__(self, key):
            assert key != "TOWER_WORLD_RAW_IMAGERY_WORLDS"
            return super().__getitem__(key)

    monkeypatch.setattr(os, "environ", GuardedEnvironment(os.environ))
    blob = imagery._tile(object(), "a" * 32, "s9", "s9:4417", image_set)
    assert blob is not None and len(blob) <= 16 * 1024


def test_b1_count_cadence_and_jpeg_floor(fake, monkeypatch):
    _Store.rows = [_Row(f"s9:{i}", 812) for i in range(20)]
    _Store.poses = [{"keyframe_id": r.keyframe_id, "segment_index": 812, "status": "solved"}
                    for r in _Store.rows]
    monkeypatch.setattr(imagery, "keyframe_image_bytes", lambda *a, **kw: (_jpeg(), "safe", None))
    clock = [10.0]
    monkeypatch.setattr(imagery.time, "monotonic", lambda: clock[0])
    first = fake.live("root", "home", "s9")
    clock[0] = 10.1
    second = fake.live("root", "home", "s9")
    clock[0] = 11.2
    third = fake.live("root", "home", "s9")
    assert len(first["entries"]) == 1
    assert len(second["entries"]) == 1
    assert len(third["entries"]) == 2
    assert len(third["entries"]) <= 16
    assert [e["keyframe_id"] for e in third["entries"]] == ["s9:19", "s9:18"]
    _Store.rows.append(_Row("s9:20", 813))
    _Store.poses.append({"keyframe_id": "s9:20", "segment_index": 813, "status": "solved"})
    assert fake.live("root", "home", "s9") == {"segment_index": 813, "entries": []}
    clock[0] = 11.3
    assert fake.live("root", "home", "s9")["entries"] == []
    clock[0] = 12.3
    assert [e["keyframe_id"] for e in fake.live("root", "home", "s9")["entries"]] == ["s9:20"]


def test_tile_omitted_when_quality_floor_cannot_fit(fake, monkeypatch):
    source = _jpeg()
    monkeypatch.setattr(imagery, "keyframe_image_bytes",
                        lambda *a, **kw: (source, "safe", None))
    monkeypatch.setattr(cv2, "imencode",
                        lambda *a, **kw: (True, np.zeros(16 * 1024 + 1, dtype=np.uint8)))
    assert imagery._tile(_Store("root"), "home", "s9", "s9:4417",
                         _Store("root").keyframe_image_set("home", "s9")) is None


def test_b1_never_exceeds_sixteen_entries(fake, monkeypatch):
    _Store.rows = [_Row(f"s9:{i}", 812) for i in range(20)]
    _Store.poses = [{"keyframe_id": r.keyframe_id, "segment_index": 812, "status": "solved"}
                    for r in _Store.rows]
    monkeypatch.setattr(imagery, "_tile", lambda store, world, session, kid, image_set: kid.encode())
    clock = [100.0]
    monkeypatch.setattr(imagery.time, "monotonic", lambda: clock[0])
    for _ in range(20):
        block = fake.live("root", "home", "s9")
        clock[0] += 1.0
    assert len(block["entries"]) == 16
    assert [e["keyframe_id"] for e in block["entries"]] == [f"s9:{i}" for i in range(19, 3, -1)]


def test_b2_landing_fixture_replacement_and_cap(fake, monkeypatch):
    monkeypatch.setattr(imagery, "_tile", lambda *args: _jpeg())
    block = {"solved_at": 1791240313.0, "geometry_revision": "g-stop-984", "frame_revision": 1}
    eligible = [(f"s9:{i}", 0) for i in range(40)]
    fake.publish_landed("root", "home", "s9", block, eligible)
    landed = fake.landed("home", "s9")
    assert {k: landed[k] for k in block} == block
    assert 1 <= len(landed["entries"]) <= 16
    assert all(e["pose_source"] == "landed_global_solve" and
               e["component_reference_segment"] == 0 and e["segment_index"] is None
               for e in landed["entries"])
    assert all(hashlib.sha256(fake.tile("home", "s9", e["digest"], False, True)).hexdigest() == e["digest"]
               for e in landed["entries"])
    newer = {**block, "solved_at": block["solved_at"] + 1, "geometry_revision": "g-new"}
    fake.publish_landed("root", "home", "s9", newer, [("s9:new", 17)])
    assert fake.landed("home", "s9")["entries"][0]["keyframe_id"] == "s9:new"
    assert fake.landed("home", "s9")["solved_at"] == newer["solved_at"]


def test_b2_wire_tile_and_404(fake, monkeypatch):
    monkeypatch.setattr(imagery, "_tile", lambda *args: _jpeg())
    fake.publish_landed("root", "home", "s9",
                        {"solved_at": 1791240313.0, "geometry_revision": "g-stop-984",
                         "frame_revision": 1}, [("s9:120", 0)])
    client = _client(monkeypatch, landed=True)
    path = "/worlds/home/guidance/s9/imagery"
    response = client.get(path + "/manifest")
    assert response.status_code == 200
    assert "live" not in response.json()
    entry = response.json()["landed"]["entries"][0]
    tile = client.get(path + "/tile/" + entry["digest"])
    assert tile.status_code == 200
    assert tile.headers["cache-control"] == "no-store"
    assert hashlib.sha256(tile.content).hexdigest() == entry["digest"]
    assert client.get(path + "/tile/" + "a" * 64).status_code == 404
    _Store.redaction = "none"
    assert client.get(path + "/manifest").status_code == 404
    assert client.get(path + "/tile/" + entry["digest"]).status_code == 404


def test_b2_budget_failure_keeps_previous(fake, monkeypatch):
    monkeypatch.setattr(imagery, "_tile", lambda *args: _jpeg())
    block = {"solved_at": 1.0, "geometry_revision": "g1", "frame_revision": 1}
    fake.publish_landed("root", "home", "s9", block, [("a", 0)])
    previous = fake.landed("home", "s9")
    ticks = iter([0.0, .081])
    monkeypatch.setattr(imagery.time, "monotonic", lambda: next(ticks, .081))
    fake.publish_landed("root", "home", "s9", {**block, "solved_at": 2.0}, [("b", 0)])
    assert fake.landed("home", "s9") == previous


def test_b2_shutdown_cannot_publish_late(fake, monkeypatch):
    monkeypatch.setattr(imagery, "_tile", lambda *args: _jpeg())
    stopped = threading.Event()
    stopped.set()
    block = {"solved_at": 2.0, "geometry_revision": "g2", "frame_revision": 1}
    fake.publish_landed("root", "home", "s9", block, [("b", 0)], cancelled=stopped)
    assert fake.landed("home", "s9") is None


def test_b2_manifest_waits_for_pending_landing_by_next_push(fake, monkeypatch):
    monkeypatch.setattr(imagery, "_tile", lambda *args: _jpeg())
    fake.begin_landed("home", "s9")
    block = {"solved_at": 2.0, "geometry_revision": "g2", "frame_revision": 1}
    requested = threading.Event()
    original_landed = fake.landed

    def observed_landed(*args, **kwargs):
        requested.set()
        return original_landed(*args, **kwargs)

    monkeypatch.setattr(fake, "landed", observed_landed)

    def finish():
        assert requested.wait(1)
        threading.Event().wait(.02)
        fake.publish_landed("root", "home", "s9", block, [("s9:120", 0)])
        fake.finish_landed("home", "s9")

    client = _client(monkeypatch, landed=True)
    thread = threading.Thread(target=finish)
    thread.start()
    response = client.get("/worlds/home/guidance/s9/imagery/manifest")
    thread.join(1)
    assert not thread.is_alive()
    assert response.status_code == 200
    assert response.json()["landed"]["solved_at"] == 2.0


def test_b2_candidates_share_coverages_accepted_solve_placement():
    from tests.test_world_builder_guidance_coverage import _inputs
    data = _inputs(((0, 4), (17, 4)), unposed_in_horizon=1, pending=1)
    data["poses"][1]["status"] = "unavailable"
    candidates = []
    coverage.compute_coverage(**data, eligible_out=candidates)
    ids = {kid for kid, _ in candidates}
    assert "k0-1" not in ids
    assert "unposed-0" not in ids
    assert "pending-0" not in ids
    assert ("k17-0", 17) in candidates
    many = _inputs(tuple((n, 4) for n in range(17)))
    candidates = []
    block = coverage.compute_coverage(**many, eligible_out=candidates)
    visible = {c["reference_segment"] for c in block["components"]}
    assert visible
    assert {ref for _, ref in candidates} == visible


def test_b2_worker_does_not_redate_on_rebuild(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_IMAGERY_LANDED", "on")
    calls = []
    published = []

    def compute(root, world, session, expected, revision, clock, tag, budget, *, eligible_out=None):
        calls.append(expected[0])
        if eligible_out is not None:
            eligible_out.append(("s9:120", 0))
        return {"solved_at": expected[0], "geometry_revision": revision, "frame_revision": 1}

    monkeypatch.setattr(coverage, "compute_from_tree", compute)
    class Sink:
        def begin_landed(self, *args):
            pass

        def finish_landed(self, *args):
            pass

        def publish_landed(self, *args, **kwargs):
            published.append(args)

    monkeypatch.setattr(imagery, "state_for", lambda root: Sink())
    worker = coverage.CoverageWorker(tmp_path)
    try:
        worker.offer("home", "s9", 1.0, 1, (1, 1), "g1", (1, "digest"))
        # Synthetic next status push after 100 ms: imagery follows the
        # coverage publication in this same worker, without a new poller.
        threading.Event().wait(.1)
        for _ in range(100):
            if published:
                break
            threading.Event().wait(.01)
        assert len(published) == 1
        worker.offer("home", "s9", 1.0, 1, (1, 1), "g2", (2, "digest"))
        threading.Event().wait(.1)
        assert calls == [1.0]
        assert len(published) == 1
        worker.offer("home", "s9", 2.0, 1, (1, 2), "g3", (3, "digest"))
        for _ in range(100):
            if len(published) == 2:
                break
            threading.Event().wait(.01)
        assert calls == [1.0, 2.0]
        assert len(published) == 2
    finally:
        worker.shutdown()
        worker._thread.join(1)
