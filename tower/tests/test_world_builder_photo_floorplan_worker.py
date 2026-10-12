"""FOW v1.2 photo floor plan, Phase B: switch, side worker, identity/dating, routes, caps.

Wire: mail 303 / tower/docs/contracts/WORLD-BUILDER-GUIDANCE-IMAGERY.md
"Photo floor plan v1.2".
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from tower import config
from tower.results import world_builder_photo_floorplan as route
from tower.results.world_builder import WorldBuilderStatusProducer
from tower.world_builder import guidance_coverage as coverage_module
from tower.world_builder import photo_floorplan_worker as pfw
from tower.world_builder.photo_floorplan import encode_tile, padded_square_bounds
from tests.test_world_builder_guidance_coverage import _inputs, _landed_tree
from tests.test_world_builder_photo_floorplan import _pitched_camera, _render

WORLD, SESSION = "fow-world", "fow-session"


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _on(monkeypatch, method=None):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    monkeypatch.setenv("TOWER_WORLD_PHOTO_FLOORPLAN", "on")
    if method is None:
        monkeypatch.delenv("TOWER_WORLD_PHOTO_FLOORPLAN_METHOD", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_PHOTO_FLOORPLAN_METHOD", method)


# ---------------------------------------------------------------------------
# switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "off", "1", "true", "yes", "ON", " on"])
def test_only_exactly_on_enables_the_photo_floorplan(monkeypatch, value):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    if value is None:
        monkeypatch.delenv("TOWER_WORLD_PHOTO_FLOORPLAN", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_PHOTO_FLOORPLAN", value)
    assert config.world_photo_floorplan_setting() is False


def test_the_photo_floorplan_needs_coverage_on(monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_PHOTO_FLOORPLAN", "on")
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "off")
    assert config.world_photo_floorplan_setting() is False
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    assert config.world_photo_floorplan_setting() is True


@pytest.mark.parametrize("value,expected", [(None, "homography"), ("homography", "homography"),
                                            ("depth", "depth"), ("DEPTH", "depth"),
                                            ("moge", "homography")])
def test_method_defaults_to_homography(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("TOWER_WORLD_PHOTO_FLOORPLAN_METHOD", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_PHOTO_FLOORPLAN_METHOD", value)
    assert config.world_photo_floorplan_method_setting() == expected


def test_off_starts_no_photo_worker_and_reads_no_extra_tree(monkeypatch, tmp_path):
    """Identical behaviour OFF: coverage computes exactly as before (no `tree_out`),
    no photo worker or thread exists, and no photo state is touched."""
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    monkeypatch.setenv("TOWER_WORLD_PHOTO_FLOORPLAN", "off")
    monkeypatch.delenv("TOWER_WORLD_GUIDANCE_IMAGERY_LANDED", raising=False)
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, _inputs())
    seen = []
    original = coverage_module.compute_from_tree

    def spy(*args, **kwargs):
        seen.append(sorted(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(coverage_module, "compute_from_tree", spy)
    photo_threads = lambda: sum(1 for t in threading.enumerate()  # noqa: E731
                                if t.name == "world-photo-floorplan")
    before = photo_threads()
    producer = WorldBuilderStatusProducer(root, lambda: 1000.12)
    try:
        producer.snapshot(world, session)
        worker = producer._coverage_worker
        assert worker._photo_worker is None
        assert _wait(lambda: producer.snapshot(world, session).payload["guidance"]["coverage"])
        assert seen and all("tree_out" not in kw and "eligible_out" not in kw for kw in seen)
        assert photo_threads() <= before
        assert pfw.state_for(root).landed(world, session) is None
    finally:
        producer.shutdown_guidance()


def test_off_routes_answer_404(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", "on")
    monkeypatch.setenv("TOWER_WORLD_PHOTO_FLOORPLAN", "off")
    root = tmp_path / "worlds"
    _landed_tree(root, _inputs())
    client = _client(root)
    base = f"/worlds/{WORLD}/guidance/{SESSION}/photo-floorplan"
    assert client.get(base + "/manifest").status_code == 404
    assert client.get(base + "/tile/" + "a" * 64).status_code == 404


@pytest.mark.parametrize("photo,coverage", [(None, "on"), ("off", "on"), ("1", "on"), ("on", "off")])
def test_real_app_does_not_register_the_routes_when_off(monkeypatch, photo, coverage):
    monkeypatch.setenv("TOWER_WORLD_GUIDANCE_COVERAGE", coverage)
    if photo is None:
        monkeypatch.delenv("TOWER_WORLD_PHOTO_FLOORPLAN", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_PHOTO_FLOORPLAN", photo)
    from tower.main import create_app
    paths = create_app().openapi()["paths"]
    assert not any("/photo-floorplan/" in path for path in paths)


def test_real_app_registers_the_routes_when_on(monkeypatch):
    _on(monkeypatch)
    from tower.main import create_app
    paths = create_app().openapi()["paths"]
    prefix = "/worlds/{world_id}/guidance/{session_id}/photo-floorplan/"
    assert prefix + "manifest" in paths
    assert prefix + "tile/{digest}" in paths


def _client(root):
    app = FastAPI()
    app.state.world_root = root
    app.include_router(route.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# coverage -> photo worker integration (identity is the receipt's)
# ---------------------------------------------------------------------------


def test_on_coverage_hands_its_published_receipt_and_tree_to_the_photo_worker(monkeypatch, tmp_path):
    _on(monkeypatch)
    root = tmp_path / "worlds"
    world, session, _ = _landed_tree(root, _inputs())
    offers = []

    def fake_build(root_, world_, session_, receipt, eligible, tree, stat, method, check):
        offers.append((receipt, list(eligible), sorted(tree), stat, method))
        tile = encode_tile(np.zeros((8, 8, 4), np.uint8))
        digest = hashlib.sha256(tile["data"]).hexdigest()
        block = {"solved_at": receipt["solved_at"], "geometry_revision": receipt["geometry_revision"],
                 "frame_revision": receipt["frame_revision"],
                 "horizon_keyframes": receipt["horizon_keyframes"], "components": []}
        return block, {digest: (tile["data"], tile["mime"])}, []

    monkeypatch.setattr(pfw, "build_landing", fake_build)
    producer = WorldBuilderStatusProducer(root, lambda: 1000.12)
    try:
        producer.snapshot(world, session)
        worker = producer._coverage_worker
        assert worker._photo_worker is not None
        assert _wait(lambda: pfw.state_for(root).landed(world, session) is not None)
        receipt = producer.snapshot(world, session).payload["guidance"]["coverage"]
        landed = pfw.state_for(root).landed(world, session)
        for key in ("solved_at", "geometry_revision", "frame_revision", "horizon_keyframes"):
            assert landed[key] == receipt[key]
        (offered, eligible, tree_keys, stat, method), = offers
        assert pfw.identity_of(offered) == pfw.identity_of(receipt)
        assert eligible and {ref for _, ref in eligible} == {0}
        assert {"solution", "manifest", "placements", "poses"} <= set(tree_keys)
        assert method == "homography"
        assert isinstance(stat, tuple) and len(stat) == 2
    finally:
        producer.shutdown_guidance()
    assert _wait(lambda: not worker._photo_worker._thread.is_alive(), timeout=3)


# ---------------------------------------------------------------------------
# side worker: atomic replacement, dating, superseded, cap, cancellation
# ---------------------------------------------------------------------------


def _receipt(solved_at, revision="g1", frame_revision=1):
    return {"solved_at": solved_at, "geometry_revision": revision,
            "frame_revision": frame_revision, "horizon_keyframes": 4,
            "components": [{"reference_segment": 0, "origin_xyz": [0, 0, 0],
                            "up_xyz": [0, 0, 1], "forward_xyz": [1, 0, 0],
                            "bounds_xy": [-1, -1, 1, 1]}]}


def _tile_for(tag):
    rgba = np.zeros((16, 16, 4), np.uint8)
    rgba[2:10, 2:10] = (int(tag) % 256, 40, 90, 255)
    return encode_tile(rgba)


def _fake_build_factory(calls=None, delay=0.0):
    def build(root, world, session, receipt, eligible, tree, stat, method, check):
        if calls is not None:
            calls.append(pfw.identity_of(receipt))
        end = time.monotonic() + delay
        while time.monotonic() < end:
            check()
            time.sleep(0.005)
        tile = _tile_for(receipt["solved_at"])
        digest = hashlib.sha256(tile["data"]).hexdigest()
        comp = {"reference_segment": 0, "origin_xyz": [0.0, 0.0, 0.0], "up_xyz": [0.0, 0.0, 1.0],
                "forward_xyz": [1.0, 0.0, 0.0],
                "raster_bounds_xy": list(padded_square_bounds((-1, -1, 1, 1))),
                "width": tile["width"], "height": tile["height"], "digest": digest,
                "bytes": tile["bytes"], "projected_keyframes": 1, "imagery_source": "unredacted"}
        block = {"solved_at": receipt["solved_at"], "geometry_revision": receipt["geometry_revision"],
                 "frame_revision": receipt["frame_revision"],
                 "horizon_keyframes": receipt["horizon_keyframes"], "components": [comp]}
        return block, {digest: (tile["data"], tile["mime"])}, []
    return build


def _worker(latest, **kwargs):
    state = pfw.PhotoFloorplanState()
    kwargs.setdefault("build", _fake_build_factory())
    worker = pfw.PhotoFloorplanWorker("synthetic", latest=lambda w, s: latest.get((w, s)),
                                      state=state, method="homography", **kwargs)
    return worker, state


def _offer(worker, latest, receipt):
    latest[(WORLD, SESSION)] = receipt
    worker.offer(WORLD, SESSION, receipt, [("k", 0)], {}, (1, 1))


def test_new_landing_replaces_the_whole_plan_atomically(monkeypatch):
    latest = {}
    worker, state = _worker(latest)
    try:
        _offer(worker, latest, _receipt(1000.0))
        assert _wait(lambda: state.landed(WORLD, SESSION) is not None)
        first = state.landed(WORLD, SESSION)
        old_digest = first["components"][0]["digest"]
        assert state.tile(WORLD, SESSION, old_digest) is not None
        _offer(worker, latest, _receipt(2000.0, revision="g2", frame_revision=2))
        assert _wait(lambda: state.landed(WORLD, SESSION)["solved_at"] == 2000.0)
        second = state.landed(WORLD, SESSION)
        assert (second["geometry_revision"], second["frame_revision"]) == ("g2", 2)
        new_digest = second["components"][0]["digest"]
        assert new_digest != old_digest
        assert state.tile(WORLD, SESSION, old_digest) is None      # superseded
        assert state.tile(WORLD, SESSION, new_digest) is not None
    finally:
        worker.shutdown()


def test_a_local_rebuild_never_re_dates_or_re_places_the_plan():
    latest = {}
    worker, state = _worker(latest)
    try:
        _offer(worker, latest, _receipt(1000.0, revision="g1"))
        assert _wait(lambda: state.landed(WORLD, SESSION) is not None)
        before = state.landed(WORLD, SESSION)
        # The same solve with a rebuilt tree (new geometry revision) must not move it.
        _offer(worker, latest, _receipt(1000.0, revision="g1-rebuilt"))
        assert _wait(worker.idle)
        assert state.landed(WORLD, SESSION) is before
        assert "never re-dated" in worker.outcomes[(WORLD, SESSION)]["disposition"]
        # Nor may an OLDER solve roll it back.
        _offer(worker, latest, _receipt(900.0, revision="g0"))
        assert _wait(worker.idle)
        assert state.landed(WORLD, SESSION) is before
    finally:
        worker.shutdown()


def test_a_failed_landing_keeps_the_prior_plan_and_its_date():
    latest = {}
    calls = []
    good = _fake_build_factory(calls)

    def flaky(root, world, session, receipt, *rest, **kwargs):
        if receipt["solved_at"] == 2000.0:
            raise pfw.LandingSkipped("solve arrays do not match the landing's solution.json")
        return good(root, world, session, receipt, *rest, **kwargs)

    worker, state = _worker(latest, build=flaky)
    try:
        _offer(worker, latest, _receipt(1000.0))
        assert _wait(lambda: state.landed(WORLD, SESSION) is not None)
        _offer(worker, latest, _receipt(2000.0, revision="g2"))
        assert _wait(lambda: worker.outcomes.get((WORLD, SESSION), {}).get("identity")
                     == (2000.0, "g2", 1))
        assert state.landed(WORLD, SESSION)["solved_at"] == 1000.0
        assert worker.outcomes[(WORLD, SESSION)]["disposition"].startswith("skipped: solve arrays")
    finally:
        worker.shutdown()


def test_a_landing_superseded_in_coverage_publishes_nothing():
    latest = {}
    worker, state = _worker(latest, build=_fake_build_factory(delay=0.1))
    try:
        receipt = _receipt(1000.0)
        _offer(worker, latest, receipt)
        latest[(WORLD, SESSION)] = _receipt(1500.0, revision="newer")  # coverage moved on
        assert _wait(worker.idle)
        assert state.landed(WORLD, SESSION) is None
        assert "superseded" in worker.outcomes[(WORLD, SESSION)]["disposition"]
    finally:
        worker.shutdown()


def test_the_wall_cap_skips_the_landing_and_keeps_the_prior_plan():
    latest = {}
    calls = []
    fast = _fake_build_factory(calls)
    slow = _fake_build_factory(calls, delay=5.0)

    def build(root, world, session, receipt, *rest, **kwargs):
        return (slow if receipt["solved_at"] == 2000.0 else fast)(
            root, world, session, receipt, *rest, **kwargs)

    worker, state = _worker(latest, build=build, wall_cap_s=0.25)
    try:
        _offer(worker, latest, _receipt(1000.0))
        assert _wait(lambda: state.landed(WORLD, SESSION) is not None)
        started = time.monotonic()
        _offer(worker, latest, _receipt(2000.0, revision="g2"))
        assert _wait(lambda: worker.outcomes.get((WORLD, SESSION), {}).get("identity")
                     == (2000.0, "g2", 1), timeout=3)
        assert time.monotonic() - started < 1.5     # stopped at the cap, not after 5 s
        assert "wall cap" in worker.outcomes[(WORLD, SESSION)]["disposition"]
        assert state.landed(WORLD, SESSION)["solved_at"] == 1000.0
    finally:
        worker.shutdown()


def test_a_newer_offer_cancels_the_in_flight_job_and_shutdown_never_waits():
    latest = {}
    calls = []
    worker, state = _worker(latest, build=_fake_build_factory(calls, delay=0.5),
                            wall_cap_s=5.0)
    _offer(worker, latest, _receipt(1000.0))
    assert _wait(lambda: calls)
    _offer(worker, latest, _receipt(2000.0, revision="g2"))
    assert _wait(lambda: state.landed(WORLD, SESSION) is not None, timeout=3)
    assert state.landed(WORLD, SESSION)["solved_at"] == 2000.0
    assert calls[0] == (1000.0, "g1", 1)
    _offer(worker, latest, _receipt(3000.0, revision="g3"))
    started = time.monotonic()
    worker.shutdown()
    assert time.monotonic() - started < 0.1
    assert _wait(lambda: not worker._thread.is_alive(), timeout=2)
    assert state.landed(WORLD, SESSION)["solved_at"] == 2000.0   # cancelled job published nothing


def test_offer_is_a_constant_time_enqueue():
    latest = {}
    worker, _ = _worker(latest, build=_fake_build_factory(delay=1.0), wall_cap_s=5.0)
    try:
        started = time.monotonic()
        for i in range(50):
            _offer(worker, latest, _receipt(1000.0 + i))
        assert time.monotonic() - started < 0.1
    finally:
        worker.shutdown()


# ---------------------------------------------------------------------------
# build_landing on a real synthetic solve (homography): identity, caps, digest
# ---------------------------------------------------------------------------


def _quat_wxyz(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    return [w, x, y, z]


def _synthetic_solve(root, *, solved_at=1000.0, n_cams=12):
    """A world whose solve/<session> holds arrays, metadata and undistorted images of a
    checkerboard floor at Z=0 seen by a ring of cameras pitched 50 degrees down."""
    solve_dir = Path(root) / "worlds" / WORLD / "solve" / SESSION
    (solve_dir / "images").mkdir(parents=True, exist_ok=True)
    kids, poses_rows, solution_poses = [], [], {}
    for i in range(n_cams):
        ang = 2 * math.pi * i / n_cams
        pos = np.array([1.5 * math.cos(ang), 1.5 * math.sin(ang), 1.5])
        yaw = math.degrees(ang + math.pi)            # look toward the centre
        pos, fwd, cam_up = _pitched_camera(pos=pos, yaw_deg=yaw, pitch_deg=50)
        right = np.cross(fwd, cam_up)
        R_wc = np.stack([right, -cam_up, fwd], axis=1)
        kid = f"{SESSION}:{i:08d}"
        kids.append(kid)
        img, _ = _render(pos, fwd, cam_up)
        cv2.imwrite(str(solve_dir / "images" / f"{i:08d}.jpg"), img[:, :, ::-1])
        poses_rows.append({"keyframe_id": kid, "segment_index": 0, "status": "solved",
                           "rotation": _quat_wxyz(R_wc), "translation": list(pos)})
        R_cw = R_wc.T
        solution_poses[kid] = {"component": 0, "observations": 50,
                               "rotation": list(R_cw.reshape(-1)),
                               "translation": list(-R_cw @ pos)}
    gx, gy = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
    floor = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    xyz = floor.astype(np.float32)
    arrays = dict(xyz=xyz, rgb=np.zeros((len(xyz), 3), np.uint8),
                  component=np.zeros(len(xyz), np.int32),
                  first_keyframe=np.zeros(len(xyz), np.int32),
                  track_length=np.full(len(xyz), 5, np.int32),
                  error=np.full(len(xyz), 0.5, np.float32),
                  observations=np.zeros((0, 3), np.int32),
                  observation_xy=np.zeros((0, 2), np.float32))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    (solve_dir / "solution.npz").write_bytes(buf.getvalue())
    camera = {"fx": 90.0, "fy": 90.0, "cx": 35.5, "cy": 63.5, "width": 72, "height": 128}
    solution = {"solved_at": solved_at, "keyframe_ids": kids, "poses": solution_poses,
                "camera": camera, "components": [{"index": 0, "points": int(len(xyz))}]}
    time.sleep(0.02)  # the metadata is written after the arrays, as write_solution does
    (solve_dir / "solution.json").write_text(json.dumps(solution), encoding="utf-8")
    st = (solve_dir / "solution.json").stat()
    tree = {"solution": solution, "poses": poses_rows,
            "placements": [{"segment_index": 0, "state": "registered", "reference_segment": 0,
                            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0], "translation": [0.0] * 3,
                            "scale": 1.0}]}
    receipt = {"solved_at": solved_at, "geometry_revision": "g-synthetic", "frame_revision": 3,
               "horizon_keyframes": n_cams,
               "components": [{"reference_segment": 0, "origin_xyz": [0.0, 0.0, 1.5],
                               "up_xyz": [0.0, 0.0, 1.0], "forward_xyz": [1.0, 0.0, 0.0],
                               "bounds_xy": [-1.5, -1.5, 1.5, 1.5]}]}
    eligible = [(kid, 0) for kid in kids]
    return solve_dir, tree, receipt, eligible, (st.st_size, st.st_mtime_ns)


def test_build_landing_matches_the_receipt_and_the_wire(tmp_path):
    solve_dir, tree, receipt, eligible, stat = _synthetic_solve(tmp_path)
    block, tiles, stats = pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree, stat)
    assert pfw.identity_of(block) == pfw.identity_of(receipt)
    assert block["horizon_keyframes"] == receipt["horizon_keyframes"]
    (comp,) = block["components"]
    rc = receipt["components"][0]
    for key in ("origin_xyz", "up_xyz", "forward_xyz"):
        assert comp[key] == rc[key]
    assert comp["raster_bounds_xy"] == pytest.approx(padded_square_bounds(rc["bounds_xy"]), abs=1e-6)
    assert comp["imagery_source"] == "unredacted"
    assert comp["projected_keyframes"] > 0
    data, mime = tiles[comp["digest"]]
    assert hashlib.sha256(data).hexdigest() == comp["digest"]
    assert len(data) == comp["bytes"] <= 180 * 1024
    assert mime in ("image/png", "image/webp")
    decoded = Image.open(io.BytesIO(data))
    assert decoded.mode == "RGBA"
    assert decoded.size == (comp["width"], comp["height"])
    assert max(decoded.size) <= 512
    alpha = np.asarray(decoded)[:, :, 3]
    assert 0 < int((alpha == 255).sum()) < alpha.size
    assert len(pfw.manifest_bytes(block)) <= 8 * 1024
    assert stats[0]["plane_ok"] is True


def test_build_landing_refuses_arrays_from_another_solve(tmp_path):
    solve_dir, tree, receipt, eligible, stat = _synthetic_solve(tmp_path)
    npz = solve_dir / "solution.npz"
    # Arrays newer than the landing's metadata: a newer solve is being written.
    later = stat[1] + 5_000_000_000
    os.utime(npz, ns=(later, later))
    with pytest.raises(pfw.LandingSkipped, match="newer"):
        pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree, stat)


def test_build_landing_refuses_a_changed_solution_json(tmp_path):
    solve_dir, tree, receipt, eligible, stat = _synthetic_solve(tmp_path)
    with pytest.raises(pfw.LandingSkipped, match="no longer"):
        pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree,
                          (stat[0] + 1, stat[1]))


def test_build_landing_refuses_arrays_whose_point_counts_disagree(tmp_path):
    solve_dir, tree, receipt, eligible, stat = _synthetic_solve(tmp_path)
    tree["solution"]["components"] = [{"index": 0, "points": 17}]
    with pytest.raises(pfw.LandingSkipped, match="do not match"):
        pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree, stat)


def test_build_landing_refuses_a_manifest_over_8_kib(tmp_path, monkeypatch):
    solve_dir, tree, receipt, eligible, stat = _synthetic_solve(tmp_path)
    monkeypatch.setattr(pfw, "MANIFEST_MAX_BYTES", 200)
    with pytest.raises(pfw.LandingSkipped, match="8 KiB"):
        pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree, stat)


def test_sixteen_components_fit_the_8_kib_manifest():
    comp = {"reference_segment": 2**31 - 1,
            "origin_xyz": [-1234.567890123456] * 3, "up_xyz": [-0.5773502691896258] * 3,
            "forward_xyz": [-0.7071067811865476, 0.7071067811865476, -1e-17],
            "raster_bounds_xy": [round(-1234.567890123456, 6)] * 4, "width": 512, "height": 512,
            "digest": "f" * 64, "bytes": 184320, "projected_keyframes": 65535,
            "imagery_source": "unredacted"}
    block = {"solved_at": 1791782871.515068, "geometry_revision": "c352e76c00d0ba9a" * 2,
             "frame_revision": 2**31 - 1, "horizon_keyframes": 65535, "components": [comp] * 16}
    assert len(pfw.manifest_bytes(block)) <= pfw.MANIFEST_MAX_BYTES


def test_the_overrun_check_stops_a_real_build_between_frames(tmp_path):
    solve_dir, tree, receipt, eligible, stat = _synthetic_solve(tmp_path)
    calls = []

    def check():
        calls.append(1)
        if len(calls) > 3:
            raise pfw.LandingOverrun("exceeded the 1.5 s wall cap")

    with pytest.raises(pfw.LandingOverrun):
        pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible, tree, stat, check=check)


# ---------------------------------------------------------------------------
# routes: wire shape, digest, superseded 404, containment, caps
# ---------------------------------------------------------------------------


@pytest.fixture
def served(monkeypatch, tmp_path):
    _on(monkeypatch)
    root = tmp_path / "worlds"
    _landed_tree(root, _inputs())
    return root, _client(root), pfw.state_for(root)


def _publish(state, solved_at, revision="g1"):
    block, tiles, _ = _fake_build_factory()("r", WORLD, SESSION, _receipt(solved_at, revision),
                                            [], {}, None, method="homography", check=lambda: None)
    assert state.publish(WORLD, SESSION, block, tiles)
    return block


def test_manifest_is_null_before_a_landing_and_no_store(served):
    root, client, state = served
    reply = client.get(f"/worlds/{WORLD}/guidance/{SESSION}/photo-floorplan/manifest")
    assert reply.status_code == 200
    assert reply.headers["content-type"].startswith("application/json")
    assert reply.headers["cache-control"] == "no-store"
    assert reply.json() == {"version": 1, "landed": None}


def test_manifest_and_tile_round_trip_then_superseded_tile_is_404(served):
    root, client, state = served
    base = f"/worlds/{WORLD}/guidance/{SESSION}/photo-floorplan"
    first = _publish(state, 1000.0)
    reply = client.get(base + "/manifest")
    assert reply.status_code == 200 and len(reply.content) <= 8 * 1024
    landed = reply.json()["landed"]
    assert pfw.identity_of(landed) == (1000.0, "g1", 1)
    comp = landed["components"][0]
    assert set(comp) == {"reference_segment", "origin_xyz", "up_xyz", "forward_xyz",
                         "raster_bounds_xy", "width", "height", "digest", "bytes",
                         "projected_keyframes", "imagery_source"}
    tile = client.get(base + "/tile/" + comp["digest"])
    assert tile.status_code == 200
    assert tile.headers["content-type"] in ("image/png", "image/webp")
    assert tile.headers["cache-control"] == "no-store"
    assert hashlib.sha256(tile.content).hexdigest() == comp["digest"]
    assert len(tile.content) == comp["bytes"] <= 180 * 1024
    assert Image.open(io.BytesIO(tile.content)).mode == "RGBA"
    _publish(state, 2000.0, "g2")
    assert client.get(base + "/tile/" + first["components"][0]["digest"]).status_code == 404
    new = client.get(base + "/manifest").json()["landed"]["components"][0]["digest"]
    assert client.get(base + "/tile/" + new).status_code == 200


@pytest.mark.parametrize("world,session", [("..", SESSION), (WORLD, ".."), ("no-such", SESSION),
                                           (WORLD, "no-such"), ("fow-world.", SESSION),
                                           ("x%5C..%5Cfow-world", SESSION)])
def test_uncontained_or_unknown_ids_are_404(served, world, session):
    root, client, state = served
    _publish(state, 1000.0)
    base = f"/worlds/{world}/guidance/{session}/photo-floorplan"
    assert client.get(base + "/manifest").status_code == 404
    digest = state.landed(WORLD, SESSION)["components"][0]["digest"]
    assert client.get(base + "/tile/" + digest).status_code == 404


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, "a" * 65, "..%2F" + "a" * 59])
def test_malformed_digests_are_404(served, digest):
    root, client, state = served
    _publish(state, 1000.0)
    reply = client.get(f"/worlds/{WORLD}/guidance/{SESSION}/photo-floorplan/tile/{digest}")
    assert reply.status_code == 404


def test_a_tile_over_the_hard_cap_is_never_served(served):
    root, client, state = served
    blob = os.urandom(180 * 1024 + 1)
    digest = hashlib.sha256(blob).hexdigest()
    block = _receipt(1000.0)
    block["components"] = []
    assert state.publish(WORLD, SESSION, block, {digest: (blob, "image/png")})
    reply = client.get(f"/worlds/{WORLD}/guidance/{SESSION}/photo-floorplan/tile/{digest}")
    assert reply.status_code == 404


def test_manifest_waits_for_an_in_flight_landing_but_never_past_the_cap(served):
    root, client, state = served
    state.begin(WORLD, SESSION)

    def land():
        time.sleep(0.2)
        _publish(state, 1000.0)
        state.finish(WORLD, SESSION)

    threading.Thread(target=land, daemon=True).start()
    started = time.monotonic()
    reply = client.get(f"/worlds/{WORLD}/guidance/{SESSION}/photo-floorplan/manifest")
    assert reply.json()["landed"]["solved_at"] == 1000.0
    assert time.monotonic() - started < 1.7


def test_landing_cap_is_8_s_and_manifest_wait_is_decoupled_at_1_6_s():
    # Manager 323 (walk-8 e2e): 1.5 s skipped live landings under load. The job cap is 8 s; the
    # manifest route's wait on an in-flight job stays 1.6 s, its own constant, never the job cap.
    from tower.results import world_builder_photo_floorplan as routes
    assert pfw.LANDING_WALL_CAP_S == 8.0
    assert routes._MANIFEST_WAIT_S == 1.6
    assert not hasattr(routes, "LANDING_WALL_CAP_S")


def test_default_worker_uses_the_8_s_cap():
    import inspect
    default = inspect.signature(pfw.PhotoFloorplanWorker.__init__).parameters["wall_cap_s"].default
    assert default == pfw.LANDING_WALL_CAP_S == 8.0


# ---------------------------------------------------------------------------
# e2e2 regression: REAL live-landing fixtures (tests/golden/fow_v12_e2e2)
#
# e2e2 published three `ok` landings with projected_keyframes 0 (the empty tile).
# On a partial live trajectory the camera-centre plane is unusable, so the floor
# estimate fell back to the receipt `up` -- the mean camera up axis, 10-13 deg off
# vertical -- and (a) the global "below the 5th-percentile camera height" cut along
# that tilted axis left almost no candidate points, (b) the 10 deg tilt window could
# exclude the true floor. Landing 0 was simply too sparse for any plane.
# ---------------------------------------------------------------------------

GOLDEN = Path(__file__).parent / "golden" / "fow_v12_e2e2"


def _landing2():
    doc = json.loads((GOLDEN / "landing2.json").read_text(encoding="utf-8"))
    with np.load(GOLDEN / "landing2_plane.npz") as z:
        return doc["receipt_component"], z["points"].astype(float), z["cameras"], z["rights"]


def test_e2e2_kf353_landing_was_on_the_failure_margin_under_the_old_rule():
    rc, pts, cams, _ = _landing2()
    up = np.asarray(rc["up_xyz"])
    fwd = np.asarray(rc["forward_xyz"])
    right = np.cross(fwd, up)
    b = padded_square_bounds(rc["bounds_xy"])
    x, y = pts @ fwd, pts @ right
    inside = (x >= b[0]) & (x <= b[2]) & (y >= b[1]) & (y <= b[3])
    old_candidates = int((inside & (pts @ up < np.percentile(cams @ up, 5))).sum())
    assert old_candidates < 100          # 61 of 10,987 points (vs a 40-inlier minimum)
    from tower.world_builder.photo_floorplan import _camera_plane_reference
    _, info = _camera_plane_reference(cams, up)
    assert info["reference_normal"] == "receipt_up"   # the partial trajectory is not a plane


def test_e2e2_kf353_landing_gets_a_well_supported_floor_now():
    from tower.world_builder.photo_floorplan import estimate_floor_plane
    rc, pts, cams, rights = _landing2()
    plane = estimate_floor_plane(points_xyz=pts, camera_positions=cams, up_xyz=rc["up_xyz"],
                                 forward_xyz=rc["forward_xyz"], bounds_xy=rc["bounds_xy"],
                                 camera_rights=rights)
    assert plane.ok, plane.reason
    assert plane.stats["reference_normal"] == "roll_free"
    assert plane.stats["candidate_points"] >= 1000
    assert plane.stats["inliers"] >= 100
    assert plane.stats["camera_height_cv"] <= 0.15


def _landing0():
    doc = json.loads((GOLDEN / "landing0.json").read_text(encoding="utf-8"))
    with np.load(GOLDEN / "landing0_arrays.npz") as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    tree = {"solution": doc["solution"], "poses": doc["poses"], "placements": doc["placements"]}
    eligible = [tuple(e) for e in doc["eligible"]]
    return doc["receipt_component"], tree, eligible, arrays


def test_e2e2_sparse_first_landing_is_never_blank():
    rc, tree, eligible, arrays = _landing0()
    entries = pfw.placed_entries(tree, eligible)[rc["reference_segment"]]
    keyframe_index = {k: i for i, k in enumerate(tree["solution"]["keyframe_ids"])}
    result, plane, frames, stats = pfw.homography_component(
        receipt_component=rc, entries=entries, tree=tree, arrays=arrays,
        keyframe_index=keyframe_index, camera=tree["solution"]["camera"],
        images_dir=GOLDEN / "images")
    # Too sparse for an estimated plane (the old code stopped here: an empty tile)...
    assert plane.stats["fallback_from"].startswith("inadequate support")
    # ...so the camera-trajectory fallback carries it.
    assert plane.ok and plane.stats["source"] == "fallback_camera_drop"
    assert plane.stats["camera_height_median"] > 0
    assert len(frames) >= 5
    assert len(result.contributing_keyframes) >= 5
    assert 0.01 < result.opaque_fraction < 1.0
    assert set(stats["ms"]) >= {"points", "plane", "gate", "load", "warp"}


def test_fallback_needs_enough_downward_frames_with_band_points():
    from tower.world_builder.photo_floorplan import FloorPlane, fallback_floor_plane
    failed = FloorPlane(ok=False, reason="inadequate support: 14 inlier points (< 40)",
                        stats={"reference_normal_xyz": [0.0, 0.0, 1.0]})
    cams = [[0, 0, 1.5], [1, 0, 1.5], [0, 1, 1.5], [1, 1, 1.5]]
    few = fallback_floor_plane(failed=failed, camera_positions=cams,
                               band_drops_by_frame=[np.full(10, 1.4)] * 2)
    assert not few.ok and "no fallback" in few.reason
    drops = [np.linspace(0.5, 1.5, 20)] * 3 + [np.array([1.0, 1.1])]   # last frame: too few
    ok = fallback_floor_plane(failed=failed, camera_positions=cams, band_drops_by_frame=drops)
    assert ok.ok and ok.stats["source"] == "fallback_camera_drop"
    assert ok.stats["fallback_frames"] == 3
    expected = float(np.percentile(np.linspace(0.5, 1.5, 20), 80))
    assert ok.stats["camera_height_median"] == pytest.approx(expected)
    assert ok.offset == pytest.approx(-(1.5 - expected))


def test_the_floor_fit_has_a_fixed_point_budget():
    from tower.world_builder.photo_floorplan import FLOOR_POINTS_BUDGET, estimate_floor_plane
    rng = np.random.default_rng(0)
    n = FLOOR_POINTS_BUDGET + 15000
    pts = np.c_[rng.uniform(-3, 3, (n, 2)), rng.normal(0, 0.01, n)]
    ang = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    cams = np.c_[2 * np.cos(ang), 2 * np.sin(ang), np.full(60, 1.5)]
    plane = estimate_floor_plane(points_xyz=pts, camera_positions=cams, up_xyz=(0, 0, 1),
                                 forward_xyz=(1, 0, 0), bounds_xy=(-2, -2, 2, 2))
    assert plane.ok
    assert plane.stats["points_used"] == FLOOR_POINTS_BUDGET
    assert abs(plane.offset) < 0.05


def test_the_previous_landings_plane_is_reused_then_refit(tmp_path):
    solve_dir, tree, receipt, eligible, stat = _synthetic_solve(tmp_path)
    memory = {}
    seen = []
    for _ in range(4):
        block, tiles, stats = pfw.build_landing(tmp_path, WORLD, SESSION, receipt, eligible,
                                                tree, stat, plane_memory=memory)
        seen.append(stats[0]["plane_reuse"])
        assert stats[0]["projected_keyframes"] > 0
    assert seen == ["no previous plane", "ok", "ok", "refit due"]


def test_reuse_carries_the_plane_through_a_gauge_change():
    from tower.world_builder.photo_floorplan import FloorPlane
    rng = np.random.default_rng(1)
    cams = np.c_[rng.uniform(-2, 2, (30, 2)), 1.5 + rng.normal(0, 0.02, 30)]
    kids = [f"k{i}" for i in range(30)]
    prior = FloorPlane(ok=True, reason="ok", normal=(0.0, 0.0, 1.0), offset=0.0,
                       stats={"camera_height_median": 1.5})
    memory = {"plane": prior, "positions": dict(zip(kids, cams)), "reuses": 0}
    # The next solve's gauge: rotated 30 deg about x, scaled 2x, shifted.
    a = math.radians(30)
    R = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    new = (2.0 * (R @ cams.T)).T + np.array([5.0, -1.0, 2.0])
    entries = [(k, p, None, None, 1.0) for k, p in zip(kids, new)]
    plane, why = pfw.reuse_plane(memory, entries)
    assert why == "ok"
    assert np.asarray(plane.normal) == pytest.approx(R @ [0, 0, 1], abs=1e-6)
    assert plane.stats["camera_height_median"] == pytest.approx(3.0, rel=0.05)
    # The floor point under the first camera maps onto the carried plane.
    foot = 2.0 * (R @ np.array([cams[0][0], cams[0][1], 0.0])) + np.array([5.0, -1.0, 2.0])
    assert float(np.asarray(plane.normal) @ foot + plane.offset) == pytest.approx(0.0, abs=1e-6)


def test_the_disposition_log_line_carries_per_component_stats(caplog):
    latest = {}

    def build(root, world, session, receipt, eligible, tree, stat, method, check):
        block, tiles, _ = _fake_build_factory()(root, world, session, receipt, eligible, tree,
                                                stat, method=method, check=check)
        stats = [{"reference_segment": 0, "eligible": 29, "points_quality": 2344,
                  "plane_ok": False, "plane_reason": "inadequate support: 14 inlier points (< 40)",
                  "plane": {"reference_normal": "roll_free", "candidate_points": 263},
                  "frame_gate_counts": {}, "projected_keyframes": 0, "opaque_fraction": 0.0,
                  "ms": {"points": 1.3, "plane": 14.1}}]
        return block, tiles, stats

    worker, state = _worker(latest, build=build)
    try:
        with caplog.at_level("INFO", logger="tower.world_builder.photo_floorplan_worker"):
            _offer(worker, latest, _receipt(1000.0))
            assert _wait(lambda: state.landed(WORLD, SESSION) is not None)
            assert _wait(lambda: any("components=" in r.getMessage() for r in caplog.records))
    finally:
        worker.shutdown()
    line = next(r.getMessage() for r in caplog.records if "components=" in r.getMessage())
    summary = json.loads(line.split("components=", 1)[1])
    assert summary[0]["plane_reason"].startswith("inadequate support")
    assert summary[0]["eligible"] == 29 and summary[0]["points"] == 2344
    assert summary[0]["projected"] == 0 and summary[0]["ms"]["plane"] == 14.1


# ---------------------------------------------------------------------------
# e2e4 regression: 10:37:47 published projected 53 / opaque 0.004 over a 0.55 plan.
# The estimated plane failed (its inliers covered 1 support cell), and the camera-
# trajectory FALLBACK was accepted although this solve's cameras are not at one
# height above it (CV 0.52, 11% below it) -- so almost no frame saw admissible floor.
# ---------------------------------------------------------------------------

GOLDEN_E2E4 = Path(__file__).parent / "golden" / "fow_v12_e2e4"


def test_e2e4_fallback_is_refused_when_the_cameras_contradict_it():
    from tower.world_builder.photo_floorplan import FloorPlane, fallback_floor_plane
    doc = json.loads((GOLDEN_E2E4 / "landing4.json").read_text(encoding="utf-8"))
    with np.load(GOLDEN_E2E4 / "landing4_fallback.npz") as z:
        cams, normal = z["cameras"], z["normal"]
        drops = np.split(z["drops"].astype(float), np.cumsum(z["lengths"])[:-1])
    failed = FloorPlane(ok=False, reason=doc["fallback_from"],
                        stats={"reference_normal_xyz": [float(v) for v in normal]})
    plane = fallback_floor_plane(failed=failed, camera_positions=cams,
                                 band_drops_by_frame=drops, frame_ids=doc["frame_ids"])
    assert not plane.ok
    assert "fallback refused: cameras inconsistent" in plane.reason
    assert plane.stats["camera_height_cv"] > 0.25


def test_e2e2_sparse_fallback_still_passes_the_camera_checks():
    rc, tree, eligible, arrays = _landing0()
    entries = pfw.placed_entries(tree, eligible)[rc["reference_segment"]]
    keyframe_index = {k: i for i, k in enumerate(tree["solution"]["keyframe_ids"])}
    _, plane, _, _ = pfw.homography_component(
        receipt_component=rc, entries=entries, tree=tree, arrays=arrays,
        keyframe_index=keyframe_index, camera=tree["solution"]["camera"],
        images_dir=GOLDEN / "images")
    assert plane.ok and plane.stats["source"] == "fallback_camera_drop"
    assert plane.stats["camera_height_cv"] <= 0.25
    assert plane.stats["camera_above_fraction"] >= 0.95


def _opaque_build(opaque_by_solve):
    def build(root, world, session, receipt, eligible, tree, stat, method, check):
        block, tiles, _ = _fake_build_factory()(root, world, session, receipt, eligible, tree,
                                                stat, method=method, check=check)
        stats = [{"reference_segment": ref, "opaque_fraction": value, "projected_keyframes": 1}
                 for ref, value in opaque_by_solve[receipt["solved_at"]].items()]
        return block, tiles, stats
    return build


def test_a_landing_that_regresses_a_component_is_skipped_and_the_prior_plan_kept():
    latest = {}
    build = _opaque_build({1000.0: {0: 0.55}, 2000.0: {0: 0.004}, 3000.0: {0: 0.52},
                           4000.0: {0: 0.10, 26: 0.0}})
    worker, state = _worker(latest, build=build)
    try:
        _offer(worker, latest, _receipt(1000.0))
        assert _wait(lambda: state.landed(WORLD, SESSION) is not None)
        _offer(worker, latest, _receipt(2000.0, revision="g2"))
        assert _wait(lambda: worker.outcomes.get((WORLD, SESSION), {}).get("identity")
                     == (2000.0, "g2", 1))
        assert worker.outcomes[(WORLD, SESSION)]["disposition"].startswith("skipped: regressed")
        assert state.landed(WORLD, SESSION)["solved_at"] == 1000.0      # prior, prior date
        _offer(worker, latest, _receipt(3000.0, revision="g3"))
        assert _wait(lambda: state.landed(WORLD, SESSION)["solved_at"] == 3000.0)
        # 0.10 >= 0.25 x 0.52 is false -> regressed; a NEW component (26) never blocks.
        _offer(worker, latest, _receipt(4000.0, revision="g4"))
        assert _wait(lambda: worker.outcomes[(WORLD, SESSION)]["identity"] == (4000.0, "g4", 1))
        assert "ref 0" in worker.outcomes[(WORLD, SESSION)]["disposition"]
        assert state.landed(WORLD, SESSION)["solved_at"] == 3000.0
    finally:
        worker.shutdown()


def test_the_regress_guard_compares_with_the_last_PUBLISHED_landing():
    latest = {}
    build = _opaque_build({1000.0: {0: 0.40}, 2000.0: {0: 0.05}, 3000.0: {0: 0.12}})
    worker, state = _worker(latest, build=build)
    try:
        for solved_at, rev in ((1000.0, "g1"), (2000.0, "g2"), (3000.0, "g3")):
            _offer(worker, latest, _receipt(solved_at, revision=rev))
            assert _wait(lambda: worker.outcomes.get((WORLD, SESSION), {}).get("identity")
                         == (solved_at, rev, 1))
        # 0.05 regressed against 0.40 and was skipped; 0.12 >= 0.25 x 0.40 publishes.
        assert state.landed(WORLD, SESSION)["solved_at"] == 3000.0
    finally:
        worker.shutdown()
