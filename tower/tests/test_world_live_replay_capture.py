"""Synthetic recorder checks: no Tower, replay, or GPU."""

import asyncio
import base64
import hashlib
import json
import threading
import time

import pytest

from scripts import world_live_replay as replay
from scripts.world_live_replay_capture import ReplayCapture, supported_by_landing


def _coverage(revision="g1"):
    return {"source": "landed_global_solve", "solved_at": 10.0,
            "horizon_keyframes": 2, "geometry_revision": revision,
            "components": [{"reference_segment": 3, "stations": [
                {"x": 1, "y": 2, "supported_mask": 0b100000000011}]}]}


def _status(revision, *, coverage=False):
    payload = {"world_snapshot": {"world_id": "w"}, "session": {"session_id": "s"},
               "geometry": {"revision": revision}}
    if coverage:
        payload["guidance"] = {"coverage": _coverage(revision)}
    envelope = {"type": "cartridge_result", "cartridge": "world_builder", "payload": payload}
    return json.dumps(envelope), envelope


def _fixture(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recorder = ReplayCapture(out)
    solution_dir = out / "solution-snapshots"
    solution_dir.mkdir()
    (solution_dir / "000.json").write_text(json.dumps({
        "solved_at": 10.0, "keyframe_ids": ["a", "b"]}), encoding="utf-8")
    manifest = {"segments": [{"segment_index": 2, "content_hash": "c", "placement_hash": "p"}]}
    segment = b'{"points":[1,2,3]}'
    recorder.geometry(("w", "s", "g1"), json.dumps(manifest).encode(),
                      [(manifest["segments"][0], segment)])
    envelope = {"type": "cartridge_result", "cartridge": "world_builder", "seq": 7,
                "payload": {"guidance": {"coverage": _coverage()}}}
    raw = json.dumps(envelope, separators=(", ", ": "), ensure_ascii=False)
    recorder.status(raw, envelope)
    recorder.status(raw, envelope)
    recorder.finish(1.0, [{"file": "000.json"}])
    return out, recorder, raw, segment


def test_exact_status_bytes_and_supported_sectors(tmp_path):
    out, _, raw, segment = _fixture(tmp_path)
    rows = [json.loads(line) for line in (out / "status-pushes.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert base64.b64decode(rows[0]["envelope_b64"]) == raw.encode()
    assert rows[0]["envelope_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert rows[0]["received_monotonic"] <= rows[1]["received_monotonic"]
    assert (out / "geometry-snapshots" / "g1" / "segment-2.json").read_bytes() == segment
    score = supported_by_landing(out)
    assert len(score) == 1  # repeat pushes do not become repeat landings
    assert score[0]["stations"] == [{"reference_segment": 3, "x": 1, "y": 2,
                                      "supported_sectors": [0, 1, 11]}]


def test_revision_reuses_saved_body_and_refuses_missing_sources(tmp_path):
    out, recorder, _, segment = _fixture(tmp_path)
    manifest = {"segments": [{"segment_index": 2, "content_hash": "c", "placement_hash": "p"}]}
    recorder.geometry(("w", "s", "g2"), json.dumps(manifest).encode(), [])
    assert (out / "geometry-snapshots" / "g2" / "segment-2.json").read_bytes() == segment
    (out / "geometry-snapshots" / "g1" / "segment-2.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="segment body"):
        supported_by_landing(out)


def test_missing_solution_and_walk_clock_refuse_score(tmp_path):
    out, recorder, _, _ = _fixture(tmp_path)
    (out / "solution-snapshots" / "000.json").write_text('{"solved_at":10}', encoding="utf-8")
    with pytest.raises(ValueError, match="accepted-keyframe"):
        supported_by_landing(out)
    recorder.finish(None, [{"file": "000.json"}])
    with pytest.raises(ValueError, match="walk clock"):
        supported_by_landing(out)


def test_flag_off_uses_original_fetch_functions_and_writes_nothing(monkeypatch, tmp_path):
    manifest = {"segments": [{"segment_index": 0, "content_hash": "a", "placement_hash": "b"}]}
    monkeypatch.setattr(replay, "http_json", lambda *args: (200, manifest))
    monkeypatch.setattr(replay, "http_bytes", lambda *args: (200, 9))
    monkeypatch.setattr(replay, "http_raw", lambda *args: pytest.fail("raw fetch with flag off"))
    mirror = replay.GeometryMirror("http://127.0.0.1:9")
    asyncio.run(mirror._fetch(("w", "s", "g1")))
    assert mirror.summary()["bytes"] == 9
    assert list(tmp_path.iterdir()) == []


def test_flag_on_fetches_and_preserves_raw_bodies(monkeypatch, tmp_path):
    manifest = b'{ "segments" : [{"segment_index":4,"content_hash":"x","placement_hash":"y"}] }'
    body = b'{ "segment_index":4, "points":[] }'
    def raw(url, timeout=30.0):
        return 200, body if "/segment/" in url else manifest
    monkeypatch.setattr(replay, "http_raw", raw)
    out = tmp_path / "run"
    out.mkdir()
    mirror = replay.GeometryMirror("http://127.0.0.1:9", capture=ReplayCapture(out))
    asyncio.run(mirror._fetch(("w", "s", "g1")))
    assert (out / "geometry-snapshots" / "g1" / "manifest.json").read_bytes() == manifest
    assert (out / "geometry-snapshots" / "g1" / "segment-4.json").read_bytes() == body


def test_socket_records_each_raw_status_before_phone_projection(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recorder = ReplayCapture(out)
    phone = replay.PhoneView()
    mirror = replay.GeometryMirror("http://127.0.0.1:9", capture=recorder)
    socket = replay.TowerSocket("ws://127.0.0.1:9/ws", replay.StreamStats(), phone, mirror,
                               capture=recorder)
    message = {"type": "cartridge_result", "cartridge": "world_builder", "seq": 1,
               "payload": {"guidance": {"coverage": _coverage()}, "model_state": "receiving"}}
    raw = json.dumps(message, separators=(", ", ": "))
    socket._dispatch(message, raw)
    socket._dispatch(message, raw)
    assert phone.pushes == 2
    rows = [json.loads(line) for line in (out / "status-pushes.jsonl").read_text().splitlines()]
    assert [base64.b64decode(row["envelope_b64"]) for row in rows] == [raw.encode()] * 2


def test_status_and_wire_revisions_can_differ_at_each_landing(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recorder = ReplayCapture(out)
    solutions = out / "solution-snapshots"
    solutions.mkdir()
    for n, (status_revision, wire_revision) in enumerate((("status-a", "wire-a"),
                                                          ("status-b", "wire-b"))):
        solved_at = 10.0 + n
        (solutions / f"{n:03}.json").write_text(json.dumps({
            "solved_at": solved_at, "keyframe_ids": ["a", "b"][:n + 1]
        }), encoding="utf-8")
        manifest = json.dumps({
            "world_id": "w", "session_id": "s", "geometry_revision": wire_revision,
            "segments": [{"segment_index": 0, "content_hash": f"c{n}",
                          "placement_hash": f"p{n}"}],
        }).encode()
        recorder.geometry(("w", "s", status_revision), manifest,
                          [({"segment_index": 0, "content_hash": f"c{n}",
                             "placement_hash": f"p{n}"},
                            json.dumps({"segment_index": 0, "content_hash": f"c{n}",
                                        "placement_hash": f"p{n}"}).encode())])
        coverage = _coverage(status_revision)
        coverage["solved_at"] = solved_at
        coverage["horizon_keyframes"] = n + 1
        envelope = {"payload": {"guidance": {"coverage": coverage}}}
        recorder.status(json.dumps(envelope), envelope)
    recorder.finish(1.0, [{"file": "000.json"}, {"file": "001.json"}])

    landings = supported_by_landing(out)
    assert [landing["geometry_revision"] for landing in landings] == ["status-a", "status-b"]
    assert [landing["horizon_keyframes"] for landing in landings] == [1, 2]
    assert (out / "geometry-snapshots" / "status-a" / "manifest.json").is_file()
    assert (out / "geometry-snapshots" / "status-b" / "segment-0.json").is_file()


def test_pinned_revision_that_vanishes_is_timed_and_incomplete(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recorder = ReplayCapture(out)
    raw, envelope = _status("landing", coverage=True)
    recorder.status(raw, envelope)
    raw, envelope = _status("next")
    recorder.status(raw, envelope)
    recorder.finish(None, [])

    summary = recorder.capture_summary()
    assert summary["pinned_missed"] == 1
    assert [row["revision"] for row in summary["pinned_misses"]] == ["landing"]
    miss = summary["pinned_misses"][0]
    assert miss["first_received_monotonic"] <= miss["missed_monotonic"]
    assert miss["reason"] == "revision superseded before complete receipt"
    assert recorder.incomplete


def test_unpinned_misses_alone_leave_capture_complete(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recorder = ReplayCapture(out)
    for revision in ("transient-a", "transient-b"):
        raw, envelope = _status(revision)
        recorder.status(raw, envelope)
    recorder.finish(None, [])

    summary = recorder.capture_summary()
    assert summary["unpinned_missed"] == 2
    assert {row["revision"] for row in summary["unpinned_misses"]} == {
        "transient-a", "transient-b"}
    assert summary["pinned_missed"] == 0
    assert not recorder.incomplete


def test_status_dispatch_pins_only_matching_coverage_revision(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recorder = ReplayCapture(out)
    mirror = replay.GeometryMirror("http://127.0.0.1:9", capture=recorder)
    socket = replay.TowerSocket("ws://127.0.0.1:9/ws", replay.StreamStats(),
                               replay.PhoneView(), mirror, capture=recorder)
    raw, envelope = _status("landing", coverage=True)
    socket._dispatch(envelope, raw)
    assert list(mirror._pinned) == [("w", "s", "landing")]

    raw, envelope = _status("later", coverage=True)
    envelope["payload"]["guidance"]["coverage"]["geometry_revision"] = "landing"
    socket._dispatch(envelope, json.dumps(envelope))
    assert list(mirror._pinned) == [("w", "s", "landing")]


def test_short_lived_pinned_revision_is_fetched_while_unpinned_fetch_is_busy(
        monkeypatch, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recorder = ReplayCapture(out)
    mirror = replay.GeometryMirror("http://127.0.0.1:9", capture=recorder)
    socket = replay.TowerSocket("ws://127.0.0.1:9/ws", replay.StreamStats(),
                               replay.PhoneView(), mirror, capture=recorder)
    first_fetch_started = threading.Event()
    release_first_fetch = threading.Event()
    current = {"revision": "busy"}
    calls = 0

    def raw(url, timeout=30.0):
        nonlocal calls
        if "/manifest?" in url:
            calls += 1
            if calls == 1:
                first_fetch_started.set()
                assert release_first_fetch.wait(2)
                revision = "busy"
            else:
                revision = current["revision"]
            return 200, json.dumps({"world_id": "w", "session_id": "s", "segments": [
                {"segment_index": 0, "content_hash": revision, "placement_hash": "p"}]}).encode()
        revision = current["revision"]
        return 200, json.dumps({"segment_index": 0, "content_hash": revision,
                                "placement_hash": "p"}).encode()

    monkeypatch.setattr(replay, "http_raw", raw)

    async def exercise():
        stop = asyncio.Event()
        worker = asyncio.create_task(mirror.run(stop))

        def push(revision, *, coverage=False):
            raw_status, envelope = _status(revision, coverage=coverage)
            current["revision"] = revision
            socket._dispatch(envelope, raw_status)

        try:
            push("busy")
            assert await asyncio.to_thread(first_fetch_started.wait, 1)
            push("landing", coverage=True)
            deadline = time.perf_counter() + 0.5
            while time.perf_counter() < deadline:
                if any(row["revision"] == "landing" and row["complete"]
                       for row in recorder.geometry_rows):
                    break
                await asyncio.sleep(0.005)
            captured_in_window = any(row["revision"] == "landing" and row["complete"]
                                     for row in recorder.geometry_rows)
            push("later")
        finally:
            release_first_fetch.set()
            await asyncio.sleep(0.02)
            stop.set()
            await asyncio.wait_for(worker, 2)
        recorder.finish(None, [])
        return captured_in_window

    assert asyncio.run(exercise())
    assert recorder.capture_summary()["pinned_missed"] == 0


def test_pinned_revision_retries_changed_segment_while_current(monkeypatch, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recorder = ReplayCapture(out)
    mirror = replay.GeometryMirror("http://127.0.0.1:9", capture=recorder)
    attempts = 0

    def raw(url, timeout=30.0):
        nonlocal attempts
        if "/manifest?" in url:
            attempts += 1
            return 200, json.dumps({"world_id": "w", "session_id": "s", "segments": [
                {"segment_index": 0, "content_hash": "new", "placement_hash": "p"}]}).encode()
        return 200, json.dumps({"segment_index": 0,
                                "content_hash": "old" if attempts == 1 else "new",
                                "placement_hash": "p"}).encode()

    monkeypatch.setattr(replay, "http_raw", raw)

    async def exercise():
        stop = asyncio.Event()
        worker = asyncio.create_task(mirror.run(stop))
        mirror.request(("w", "s", "landing"), pinned=True)
        try:
            for _ in range(100):
                if recorder.geometry_rows and recorder.geometry_rows[-1]["complete"]:
                    break
                await asyncio.sleep(0.005)
        finally:
            stop.set()
            await asyncio.wait_for(worker, 2)

    asyncio.run(exercise())
    assert attempts >= 2
    assert recorder.geometry_rows[-1]["complete"]
    assert len(mirror.capture_errors) == 1


def test_changed_segment_does_not_publish_partial_revision_before_retry(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recorder = ReplayCapture(out)
    item = {"segment_index": 0, "content_hash": "new", "placement_hash": "p"}
    manifest = json.dumps({"segments": [item]}).encode()
    stale = b'{"segment_index":0,"content_hash":"old","placement_hash":"p"}'
    current = b'{"segment_index":0,"content_hash":"new","placement_hash":"p"}'

    with pytest.raises(ValueError, match="changed before revision"):
        recorder.geometry(("w", "s", "landing"), manifest, [(item, stale)])
    assert not (out / "geometry-snapshots" / "landing" / "manifest.json").exists()
    recorder.geometry(("w", "s", "landing"), manifest, [(item, current)])
    assert recorder.geometry_rows[-1]["complete"]
    assert (out / "geometry-snapshots" / "landing" / "segment-0.json").read_bytes() == current


def test_partial_fetch_does_not_lock_retry_to_old_manifest(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recorder = ReplayCapture(out)
    old = json.dumps({"segments": [{"segment_index": 0, "content_hash": "old",
                                     "placement_hash": "p"}]}).encode()
    item = {"segment_index": 0, "content_hash": "new", "placement_hash": "p"}
    new = json.dumps({"segments": [item]}).encode()
    body = b'{"segment_index":0,"content_hash":"new","placement_hash":"p"}'

    assert recorder.geometry(("w", "s", "landing"), old, []) is False
    assert not (out / "geometry-snapshots" / "landing" / "manifest.json").exists()
    assert recorder.geometry(("w", "s", "landing"), new, [(item, body)]) is True
    assert (out / "geometry-snapshots" / "landing" / "manifest.json").read_bytes() == new
