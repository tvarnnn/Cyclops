"""Photo floor plan KEPT notice (TOWER_WORLD_PHOTO_FLOORPLAN_KEPT=on; manager mail 354).

When the newest landing for a session publishes nothing while an earlier plan is published,
the manifest carries an optional top-level `kept` block -- the skipped landing, the reason,
and the landing whose plan is served -- so the phone can keep drawing the held plan past its
45 s hold. The served plan is never re-dated; the block goes with the next published landing.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from tower import config
from tower.world_builder import photo_floorplan_worker as pfw
from tests.test_world_builder_photo_floorplan_regress_v2 import (
    FULL, NEW_GAUGE, ROOM, _landing, _room, _v2_build)
from tests.test_world_builder_photo_floorplan_worker import (
    SESSION, WORLD, _client, _fake_build_factory, _inputs, _landed_tree, _offer, _on,
    _opaque_build, _receipt, _wait)

BASE = f"/worlds/{WORLD}/guidance/{SESSION}/photo-floorplan"


def _ident(solved_at, revision, frame_revision=1):
    return {"solved_at": solved_at, "geometry_revision": revision, "frame_revision": frame_revision}


def _done(worker, solved_at, revision, frame_revision=1, timeout=5.0):
    return _wait(lambda: worker.outcomes.get((WORLD, SESSION), {}).get("identity")
                 == (solved_at, revision, frame_revision), timeout=timeout)


def _kept_worker(build, *, kept=True, state=None, **kwargs):
    latest = {}
    state = state if state is not None else pfw.PhotoFloorplanState()
    worker = pfw.PhotoFloorplanWorker("synthetic", latest=lambda w, s: latest.get((w, s)),
                                      state=state, method="homography", build=build,
                                      kept=kept, **kwargs)
    return worker, state, latest


def _kept_of(state):
    return state.served(WORLD, SESSION)[1]


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [(None, False), ("", False), ("off", False),
                                             ("1", False), ("true", False), ("ON", False),
                                             (" on", False), ("on ", False), ("on", True)])
def test_kept_switch_is_exactly_on(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(config.WORLD_PHOTO_FLOORPLAN_KEPT_ENV, raising=False)
    else:
        monkeypatch.setenv(config.WORLD_PHOTO_FLOORPLAN_KEPT_ENV, value)
    assert config.WORLD_PHOTO_FLOORPLAN_KEPT_ENV == "TOWER_WORLD_PHOTO_FLOORPLAN_KEPT"
    assert config.world_photo_floorplan_kept_setting() is expected


@pytest.mark.parametrize("value, expected", [(None, False), ("1", False), ("on", True)])
def test_the_worker_reads_the_switch_when_not_told(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(config.WORLD_PHOTO_FLOORPLAN_KEPT_ENV, raising=False)
    else:
        monkeypatch.setenv(config.WORLD_PHOTO_FLOORPLAN_KEPT_ENV, value)
    worker = pfw.PhotoFloorplanWorker("synthetic", state=pfw.PhotoFloorplanState(),
                                      method="homography", build=_fake_build_factory())
    try:
        assert worker.kept is expected
    finally:
        worker.shutdown()


# ---------------------------------------------------------------------------
# OFF identity
# ---------------------------------------------------------------------------


REGRESSING = {1000.0: {0: 0.55}, 2000.0: {0: 0.004}}   # e2e4: the second landing is skipped


def _today_bytes(landed):
    # The v1.2 manifest exactly as it was serialised before this switch existed.
    return json.dumps({"version": 1, "landed": landed}, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def test_manifest_bytes_without_kept_are_todays():
    block = _fake_build_factory()("r", WORLD, SESSION, _receipt(1000.0), [], {}, None,
                                  method="homography", check=lambda: None)[0]
    assert pfw.manifest_bytes(block) == _today_bytes(block)
    assert pfw.manifest_bytes(None) == _today_bytes(None) == b'{"version":1,"landed":null}'
    assert pfw.manifest_bytes(block, None) == _today_bytes(block)


def test_off_records_nothing_and_serves_todays_bytes(monkeypatch, tmp_path):
    _on(monkeypatch)
    monkeypatch.delenv(config.WORLD_PHOTO_FLOORPLAN_KEPT_ENV, raising=False)
    root = tmp_path / "worlds"
    _landed_tree(root, _inputs())
    client = _client(root)
    worker, state, latest = _kept_worker(_opaque_build(REGRESSING), kept=None,
                                         state=pfw.state_for(root))
    try:
        assert worker.kept is False
        _offer(worker, latest, _receipt(1000.0))
        assert _done(worker, 1000.0, "g1")
        _offer(worker, latest, _receipt(2000.0, revision="g2"))
        assert _done(worker, 2000.0, "g2")
        assert worker.outcomes[(WORLD, SESSION)]["disposition"].startswith("skipped: regressed")
        assert "kept" not in worker.outcomes[(WORLD, SESSION)]
        assert state._kept == {}
        landed = state.landed(WORLD, SESSION)
        reply = client.get(BASE + "/manifest")
        assert reply.content == _today_bytes(landed)
        assert "kept" not in reply.json()
    finally:
        worker.shutdown()


def test_off_route_never_serves_a_notice_even_if_one_is_held(monkeypatch, tmp_path):
    _on(monkeypatch)
    monkeypatch.delenv(config.WORLD_PHOTO_FLOORPLAN_KEPT_ENV, raising=False)
    root = tmp_path / "worlds"
    _landed_tree(root, _inputs())
    state = pfw.state_for(root)
    block, tiles, _ = _fake_build_factory()("r", WORLD, SESSION, _receipt(1000.0), [], {}, None,
                                            method="homography", check=lambda: None)
    assert state.publish(WORLD, SESSION, block, tiles)
    assert state.keep(WORLD, SESSION, _receipt(2000.0, revision="g2"), pfw.KEPT_REGRESSED)
    reply = _client(root).get(BASE + "/manifest")
    assert reply.content == _today_bytes(block)


# ---------------------------------------------------------------------------
# ON: a regressed skip keeps the plan, labelled; the plan bytes never change
# ---------------------------------------------------------------------------


@pytest.fixture
def kept_on(monkeypatch, tmp_path):
    _on(monkeypatch)
    monkeypatch.setenv(config.WORLD_PHOTO_FLOORPLAN_KEPT_ENV, "on")
    root = tmp_path / "worlds"
    _landed_tree(root, _inputs())
    return root, _client(root), pfw.state_for(root)


def test_on_a_regressed_skip_serves_kept_beside_the_unchanged_plan(kept_on):
    root, client, state = kept_on
    worker, _, latest = _kept_worker(_opaque_build({**REGRESSING, 3000.0: {0: 0.52}}),
                                     kept=None, state=state)
    try:
        assert worker.kept is True
        _offer(worker, latest, _receipt(1000.0))
        assert _done(worker, 1000.0, "g1")
        before = client.get(BASE + "/manifest")
        assert set(before.json()) == {"version", "landed"}          # nothing skipped yet
        plan_digest = before.json()["landed"]["components"][0]["digest"]
        tile_before = client.get(BASE + "/tile/" + plan_digest).content

        _offer(worker, latest, _receipt(2000.0, revision="g2", frame_revision=2))
        assert _done(worker, 2000.0, "g2", 2)
        assert worker.outcomes[(WORLD, SESSION)]["kept"] == "regressed"
        reply = client.get(BASE + "/manifest")
        doc = reply.json()
        assert len(reply.content) <= pfw.MANIFEST_MAX_BYTES
        assert doc["version"] == 1
        assert doc["kept"] == {"for_landing": _ident(2000.0, "g2", 2), "reason": "regressed",
                               "plan_landing": _ident(1000.0, "g1", 1)}
        # The plan is never re-dated: `landed` serialises to exactly the bytes served before.
        assert json.dumps(doc["landed"], separators=(",", ":")) == json.dumps(
            before.json()["landed"], separators=(",", ":"))
        assert reply.content == pfw.manifest_bytes(before.json()["landed"], doc["kept"])
        assert client.get(BASE + "/tile/" + plan_digest).content == tile_before

        # A later PUBLISHED landing removes the notice in the same step.
        _offer(worker, latest, _receipt(3000.0, revision="g3", frame_revision=3))
        assert _done(worker, 3000.0, "g3", 3)
        after = client.get(BASE + "/manifest")
        assert after.json()["landed"]["solved_at"] == 3000.0
        assert "kept" not in after.json()
        assert after.content == _today_bytes(state.landed(WORLD, SESSION))
        assert worker.outcomes[(WORLD, SESSION)]["kept"] is None
    finally:
        worker.shutdown()


def test_each_skip_replaces_the_notice_with_the_newest_landing():
    worker, state, latest = _kept_worker(_opaque_build(
        {1000.0: {0: 0.55}, 2000.0: {0: 0.004}, 3000.0: {0: 0.003}}))
    try:
        for solved_at, rev in ((1000.0, "g1"), (2000.0, "g2"), (3000.0, "g3")):
            _offer(worker, latest, _receipt(solved_at, revision=rev))
            assert _done(worker, solved_at, rev)
        assert _kept_of(state) == {"for_landing": _ident(3000.0, "g3"), "reason": "regressed",
                                   "plan_landing": _ident(1000.0, "g1")}
        assert state.landed(WORLD, SESSION)["solved_at"] == 1000.0
    finally:
        worker.shutdown()


# ---------------------------------------------------------------------------
# each reason
# ---------------------------------------------------------------------------


def _two_landings(build, **kwargs):
    """Publish 1000/g1, then offer 2000/g2; returns (outcome of 2000, state)."""
    worker, state, latest = _kept_worker(build, **kwargs)
    try:
        _offer(worker, latest, _receipt(1000.0))
        assert _done(worker, 1000.0, "g1")
        assert state.landed(WORLD, SESSION)["solved_at"] == 1000.0
        _offer(worker, latest, _receipt(2000.0, revision="g2"))
        assert _done(worker, 2000.0, "g2")
        return worker.outcomes[(WORLD, SESSION)], state
    finally:
        worker.shutdown()


def _assert_kept(state, reason, for_landing=None):
    assert _kept_of(state) == {"for_landing": for_landing or _ident(2000.0, "g2"),
                               "reason": reason, "plan_landing": _ident(1000.0, "g1")}
    assert state.landed(WORLD, SESSION)["solved_at"] == 1000.0


def test_reason_regressed_legacy_guard():
    outcome, state = _two_landings(_opaque_build(REGRESSING))
    assert outcome["disposition"].startswith("skipped: regressed")
    _assert_kept(state, "regressed")


def test_reason_wall_cap():
    fast, slow = _fake_build_factory(), _fake_build_factory(delay=5.0)

    def build(root, world, session, receipt, *rest, **kwargs):
        return (slow if receipt["solved_at"] == 2000.0 else fast)(
            root, world, session, receipt, *rest, **kwargs)

    outcome, state = _two_landings(build, wall_cap_s=0.25)
    assert "wall cap" in outcome["disposition"]
    _assert_kept(state, "wall_cap")


@pytest.mark.parametrize("error", [pfw.LandingSkipped("solve arrays do not match"),
                                   RuntimeError("boom")])
def test_reason_failed(error):
    good = _fake_build_factory()

    def build(root, world, session, receipt, *rest, **kwargs):
        if receipt["solved_at"] == 2000.0:
            raise error
        return good(root, world, session, receipt, *rest, **kwargs)

    outcome, state = _two_landings(build)
    assert outcome["disposition"].startswith(("skipped: solve arrays", "failed: RuntimeError"))
    _assert_kept(state, "failed")


def test_reason_superseded_in_coverage():
    good, slow = _fake_build_factory(), _fake_build_factory(delay=0.1)
    worker, state, latest = _kept_worker(
        lambda root, world, session, receipt, *rest, **kwargs:
        (slow if receipt["solved_at"] == 2000.0 else good)(root, world, session, receipt,
                                                           *rest, **kwargs))
    try:
        _offer(worker, latest, _receipt(1000.0))
        assert _done(worker, 1000.0, "g1")
        _offer(worker, latest, _receipt(2000.0, revision="g2"))
        latest[(WORLD, SESSION)] = _receipt(2500.0, revision="newer")   # coverage moved on
        assert _done(worker, 2000.0, "g2")
        assert "superseded" in worker.outcomes[(WORLD, SESSION)]["disposition"]
        _assert_kept(state, "superseded")
    finally:
        worker.shutdown()


def test_reason_superseded_by_cancellation_then_the_newer_landing_publishes():
    calls = []
    fast, slow = _fake_build_factory(calls), _fake_build_factory(calls, delay=2.0)

    def build(root, world, session, receipt, *rest, **kwargs):
        return (slow if receipt["solved_at"] == 2000.0 else fast)(
            root, world, session, receipt, *rest, **kwargs)

    seen = []
    worker, state, latest = _kept_worker(build, wall_cap_s=5.0)
    original_keep = state.keep

    def spy(*args):
        result = original_keep(*args)
        seen.append((args[3], result, dict(_kept_of(state) or {})))
        return result

    state.keep = spy
    try:
        _offer(worker, latest, _receipt(1000.0))
        assert _done(worker, 1000.0, "g1")
        _offer(worker, latest, _receipt(2000.0, revision="g2"))
        assert _wait(lambda: (2000.0, "g2", 1) in calls)
        _offer(worker, latest, _receipt(3000.0, revision="g3"))   # cancels 2000 in flight
        assert _done(worker, 3000.0, "g3")
        assert seen and seen[0][0] == "superseded" and seen[0][1] is True
        assert seen[0][2]["for_landing"] == _ident(2000.0, "g2")
        assert state.landed(WORLD, SESSION)["solved_at"] == 3000.0
        assert _kept_of(state) is None                       # the publish removed it
    finally:
        worker.shutdown()


def test_reason_not_newer_for_a_local_rebuild_of_the_same_solve():
    worker, state, latest = _kept_worker(_fake_build_factory())
    try:
        _offer(worker, latest, _receipt(1000.0, revision="g1"))
        assert _done(worker, 1000.0, "g1")
        before = state.landed(WORLD, SESSION)
        _offer(worker, latest, _receipt(1000.0, revision="g1-rebuilt"))
        assert _done(worker, 1000.0, "g1-rebuilt")
        assert "never re-dated" in worker.outcomes[(WORLD, SESSION)]["disposition"]
        assert state.landed(WORLD, SESSION) is before
        _assert_kept(state, "not_newer", _ident(1000.0, "g1-rebuilt"))
    finally:
        worker.shutdown()


def _run_v2(plan):
    worker, state, latest = _kept_worker(_v2_build(plan), regress_v2=True,
                                         clock=lambda: 0.0)
    outcomes = []
    try:
        for i, solved_at in enumerate(sorted(plan)):
            _offer(worker, latest, _receipt(solved_at, revision=f"g{i}"))
            assert _done(worker, solved_at, f"g{i}")
            outcomes.append((worker.outcomes[(WORLD, SESSION)], _kept_of(state)))
    finally:
        worker.shutdown()
    return outcomes, state


def test_reasons_regressed_then_stalled_under_v2():
    degraded = _landing(ROOM, _room(0.0, 1.0, 0.0, 4.0), source="fallback_camera_drop",
                        **NEW_GAUGE)
    plan = {1000.0: _landing(ROOM, FULL),
            **{t: degraded for t in (2000.0, 3000.0, 4000.0, 5000.0)}}
    outcomes, state = _run_v2(plan)
    assert [o["regress_v2"]["decision"] for o, _ in outcomes] == [
        "publish", "skip", "skip", "skip", "stalled"]
    assert [o["kept"] for o, _ in outcomes] == [
        None, "regressed", "regressed", "regressed", "stalled"]
    assert [k and k["for_landing"]["geometry_revision"] for _, k in outcomes] == [
        None, "g1", "g2", "g3", "g4"]
    assert all(k["plan_landing"] == _ident(1000.0, "g0") for _, k in outcomes[1:])
    assert state.landed(WORLD, SESSION)["solved_at"] == 1000.0


def test_unknown_reason_is_refused():
    state = pfw.PhotoFloorplanState()
    with pytest.raises(ValueError):
        state.keep(WORLD, SESSION, _receipt(2000.0), "because")


# ---------------------------------------------------------------------------
# nothing to keep
# ---------------------------------------------------------------------------


def test_no_published_plan_means_no_kept_block(kept_on):
    root, client, state = kept_on

    def build(root_, world, session, receipt, *rest, **kwargs):
        raise pfw.LandingSkipped("solve arrays unreadable")

    worker, _, latest = _kept_worker(build, state=state)
    try:
        _offer(worker, latest, _receipt(1000.0))
        assert _done(worker, 1000.0, "g1")
        assert worker.outcomes[(WORLD, SESSION)]["kept"] is None
        assert state._kept == {}
        reply = client.get(BASE + "/manifest")
        assert reply.content == b'{"version":1,"landed":null}'
    finally:
        worker.shutdown()


def test_a_skip_of_the_plans_own_identity_records_nothing():
    state = pfw.PhotoFloorplanState()
    block, tiles, _ = _fake_build_factory()("r", WORLD, SESSION, _receipt(1000.0), [], {}, None,
                                            method="homography", check=lambda: None)
    assert state.publish(WORLD, SESSION, block, tiles)
    assert not state.keep(WORLD, SESSION, _receipt(1000.0), pfw.KEPT_NOT_NEWER)
    assert state.served(WORLD, SESSION) == (block, None)


# ---------------------------------------------------------------------------
# caps and atomicity
# ---------------------------------------------------------------------------


def _sixteen(revision):
    comp = {"reference_segment": 2**31 - 1,
            "origin_xyz": [-1234.567890123456] * 3, "up_xyz": [-0.5773502691896258] * 3,
            "forward_xyz": [-0.7071067811865476, 0.7071067811865476, -1e-17],
            "raster_bounds_xy": [round(-1234.567890123456, 6)] * 4, "width": 512, "height": 512,
            "digest": "f" * 64, "bytes": 184320, "projected_keyframes": 65535,
            "imagery_source": "unredacted"}
    return {"solved_at": 1791782871.515068, "geometry_revision": revision,
            "frame_revision": 2**31 - 1, "horizon_keyframes": 65535, "components": [comp] * 16}


def test_the_worst_case_sixteen_component_manifest_with_kept_stays_under_8_kib():
    # The same worst case as test_sixteen_components_fit_the_8_kib_manifest, plus a notice.
    state = pfw.PhotoFloorplanState()
    block = _sixteen("c352e76c00d0ba9a" * 2)
    assert state.publish(WORLD, SESSION, block, {})
    newer = dict(block, solved_at=1791782999.123456, geometry_revision="d" * 32)
    assert state.keep(WORLD, SESSION, newer, pfw.KEPT_STALLED)
    landed, kept = state.served(WORLD, SESSION)
    assert len(pfw.manifest_bytes(landed, kept)) <= pfw.MANIFEST_MAX_BYTES


def test_a_notice_that_would_break_the_cap_is_dropped_never_the_plan(kept_on, monkeypatch):
    root, client, state = kept_on
    block = _sixteen("c352e76c00d0ba9a" * 2)
    assert state.publish(WORLD, SESSION, block, {})
    plain = pfw.manifest_bytes(block)
    monkeypatch.setattr(pfw, "MANIFEST_MAX_BYTES", len(plain) + 10)
    assert not state.keep(WORLD, SESSION, dict(block, solved_at=1791782999.0,
                                               geometry_revision="d" * 128),
                          pfw.KEPT_REGRESSED)
    assert state.served(WORLD, SESSION) == (block, None)


def test_the_route_drops_an_oversize_notice_and_serves_the_plan(kept_on, monkeypatch):
    root, client, state = kept_on
    block = _sixteen("c352e76c00d0ba9a" * 2)
    assert state.publish(WORLD, SESSION, block, {})
    assert state.keep(WORLD, SESSION, dict(block, solved_at=1791782999.0), pfw.KEPT_REGRESSED)
    from tower.results import world_builder_photo_floorplan as route
    monkeypatch.setattr(route, "MANIFEST_MAX_BYTES", len(pfw.manifest_bytes(block)) + 10)
    reply = client.get(BASE + "/manifest")
    assert reply.content == _today_bytes(block)


def test_the_notice_is_always_served_beside_the_plan_it_describes():
    state = pfw.PhotoFloorplanState()
    stop = threading.Event()
    build = _fake_build_factory()

    def writer():
        t = 1000.0
        while not stop.is_set():
            block, tiles, _ = build("r", WORLD, SESSION, _receipt(t, f"g{t}"), [], {}, None,
                                    method="homography", check=lambda: None)
            state.publish(WORLD, SESSION, block, tiles)
            state.keep(WORLD, SESSION, _receipt(t + 1.0, f"g{t + 1.0}"), pfw.KEPT_REGRESSED)
            t += 2.0

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 0.5
        checked = 0
        while time.monotonic() < deadline:
            landed, kept = state.served(WORLD, SESSION)
            if landed is None:
                continue
            if kept is not None:
                assert kept["plan_landing"] == pfw._kept_identity(landed)
                assert kept["for_landing"]["solved_at"] == landed["solved_at"] + 1.0
                checked += 1
        assert checked > 0
    finally:
        stop.set()
        thread.join(timeout=2)
