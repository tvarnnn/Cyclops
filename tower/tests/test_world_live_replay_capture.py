"""Synthetic recorder checks: no Tower, replay, or GPU."""

import asyncio
import base64
import hashlib
import json

import pytest

from scripts import world_live_replay as replay
from scripts.world_live_replay_capture import ReplayCapture, supported_by_landing


def _coverage(revision="g1"):
    return {"source": "landed_global_solve", "solved_at": 10.0,
            "horizon_keyframes": 2, "geometry_revision": revision,
            "components": [{"reference_segment": 3, "stations": [
                {"x": 1, "y": 2, "supported_mask": 0b100000000011}]}]}


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


def test_revision_mismatch_is_refused(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recorder = ReplayCapture(out)
    with pytest.raises(ValueError, match="manifest changed"):
        recorder.geometry(("w", "s", "g1"),
                          b'{"geometry_revision":"g2","segments":[]}', [])
