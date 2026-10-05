"""Pins for three false-`stop` doors the e1cb8ec re-review found unpinned (LOW-2: mutants A8, A29, A2).

Each test fails if the guard counts a frame it could not read as observed. Synthetic frames only."""
import json, time
import pytest
import tower.capture as capture_module
from scripts.world_build_session import StopRequest
from tests.test_world_builder_closed_capture_drain import (
    _hook, _run_session, _write_manifest, _image_unreadable_for, _labelled_finished,
)


def _stamped(directory, count, start=0, end_reason="stop", **extra):
    directory.mkdir(parents=True, exist_ok=True)
    mono0 = time.monotonic() - 100.0
    rows = []
    for index in range(start, start + count):
        name = f"frame{index:04d}.jpg"
        (directory / name).write_bytes(b"synthetic-frame")
        rows.append(json.dumps({"source_seq": index + 1, "received_at": float(index),
                                "received_monotonic": mono0 + index * 0.08, "relpath": name}) + "\n")
    (directory / "frames.jsonl").write_text("".join(rows), encoding="utf-8")
    _write_manifest(directory, end_reason, **extra)
    return directory


def test_k8_unparseable_line_in_the_drain_after_close_then_soft(tmp_path, monkeypatch):
    """iOS order (close, then soft stop) with a late builder; the drain reads a
    corrupt journal line. Must not be stop/complete. Kills A8."""
    capture = _stamped(tmp_path / "captures" / ("a" * 32), 48, end_reason=None)
    journal = capture / "frames.jsonl"
    lines = journal.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[29] = lines[29][:25] + "\n"
    journal.write_text("".join(lines), encoding="utf-8")

    def on_observe(n, stop):
        if n == 10:
            _write_manifest(capture, "stop")
            stop.request(StopRequest.SOFT, "stdin-closed")

    _hook(monkeypatch, on_observe)
    exit_code, session = _run_session(capture, tmp_path / "worlds")
    assert session.frames_observed == 47
    assert session.end_reason == "interrupted"
    assert not _labelled_finished(session)


def test_k29_unreadable_image_in_a_predecessor_is_counted(tmp_path, monkeypatch):
    """Two-capture walk (disconnect -> successor -> stop), no stop request; one
    image in the PREDECESSOR stays locked past the budget. Kills A29."""
    captures = tmp_path / "captures"
    first = _stamped(captures / ("a" * 32), 24, end_reason="disconnect")
    _stamped(captures / ("b" * 32), 24, start=24, end_reason="stop", continues_capture=first.name)
    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", 0.25)
    _image_unreadable_for(monkeypatch, {"frame0009.jpg"}, 60.0)
    _hook(monkeypatch)
    exit_code, session = _run_session(first, tmp_path / "worlds", max_idle_polls="5")
    assert session.frames_observed == 47
    assert session.end_reason == "interrupted"


def test_k2_absent_journal_and_a_manifest_unreadable_at_the_guard(tmp_path, monkeypatch):
    """Closed capture says frames_written=10 but has no journal; after the close
    is verified, the manifest becomes unreadable. Must not answer 0. Kills A2."""
    capture = tmp_path / "captures" / ("a" * 32)
    capture.mkdir(parents=True)
    _write_manifest(capture, "stop", frames_written=10)
    manifest = capture / "capture.json"
    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", 0.2)
    original = capture_module.read_json_closed
    state = {"seen_closed": False}

    def read(path):
        if path == manifest and state["seen_closed"]:
            raise PermissionError(13, "held", str(path))
        payload = original(path)
        if path == manifest and payload.get("ended_at") is not None:
            state["seen_closed"] = True
        return payload

    monkeypatch.setattr(capture_module, "read_json_closed", read)
    _hook(monkeypatch)
    exit_code, session = _run_session(capture, tmp_path / "worlds", max_idle_polls="5")
    assert state["seen_closed"]
    assert session.frames_observed == 0
    assert session.end_reason == "interrupted"
