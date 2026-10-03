"""A late World Builder follower drains a normally closed capture on soft stop."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

import tower.capture as capture_module
from scripts.world_build_session import StopRequest, first_observed_frame, follow_capture
from tower.world_builder.store import WorldStore


@pytest.fixture
def recorded_capture(tmp_path):
    """The field failure's 3,197 journal entries, without private imagery."""
    directory = tmp_path / "capture"
    directory.mkdir()
    (directory / "frame.jpg").write_bytes(b"synthetic-frame")
    records = (
        {
            "source_seq": 1 + (index * 6476 // 3196),
            "received_at": float(index),
            "relpath": "frame.jpg",
        }
        for index in range(3197)
    )
    (directory / "frames.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return directory


@pytest.mark.parametrize(
    ("end_reason", "stop_level", "expected_count"),
    [
        ("stop", StopRequest.SOFT, 3197),
        (None, StopRequest.SOFT, 805),
        ("stop", StopRequest.HARD, 805),
        ("disconnect", StopRequest.SOFT, 805),
    ],
)
def test_recorded_backlog_stop_policy(
    recorded_capture, end_reason, stop_level, expected_count
):
    # The worker attaches late while the capture is still open. Its first
    # journal read sees a backlog; the recorder closes while it processes it.
    manifest = recorded_capture / "capture.json"
    manifest.write_text(
        json.dumps({"ended_at": None, "end_reason": None}), encoding="utf-8"
    )
    stop = StopRequest()
    handle = {}
    should_stop = lambda: stop.asked_for_capture(handle)
    frames = follow_capture(
        recorded_capture,
        poll_seconds=0,
        max_idle_polls=1,
        should_stop=should_stop,
        handle=handle,
    )
    first, frames = first_observed_frame(frames)
    assert first.source_seq == 1

    observed = []
    for frame in stop.bounded(frames, should_stop=should_stop):
        observed.append(frame.source_seq)
        if len(observed) == 805:
            if end_reason is not None:
                manifest.write_text(
                    json.dumps({"ended_at": 3197.0, "end_reason": end_reason}),
                    encoding="utf-8",
                )
            stop.request(stop_level, "test-stdin-close")

    assert len(observed) == expected_count
    assert observed[-1] == (6477 if expected_count == 3197 else 1630)
    if expected_count == 3197:
        assert observed[805] > observed[804]
        assert handle["follower"].end_reason() == "stop"


def test_soft_stop_before_first_frame_still_drains_closed_capture(recorded_capture):
    (recorded_capture / "capture.json").write_text(
        json.dumps({"ended_at": 3197.0, "end_reason": "stop"}), encoding="utf-8"
    )
    stop = StopRequest()
    stop.request(StopRequest.SOFT, "test-stdin-close")
    handle = {}
    should_stop = lambda: stop.asked_for_capture(handle)
    frames = follow_capture(
        recorded_capture,
        poll_seconds=0,
        max_idle_polls=1,
        should_stop=should_stop,
        handle=handle,
    )
    first, frames = first_observed_frame(frames)
    assert first.source_seq == 1
    observed = [frame.source_seq for frame in stop.bounded(frames, should_stop=should_stop)]
    assert len(observed) == 3197
    assert observed[-1] == 6477


def test_transient_manifest_read_at_soft_stop_does_not_truncate(
    recorded_capture, monkeypatch
):
    manifest = recorded_capture / "capture.json"
    manifest.write_text(
        json.dumps({"ended_at": None, "end_reason": None}), encoding="utf-8"
    )
    original_read = capture_module.read_json_closed
    fault = {"armed": False, "raised": False}

    def read_with_one_replace_failure(path):
        if path == manifest and fault["armed"] and not fault["raised"]:
            fault["raised"] = True
            raise ValueError("manifest caught during atomic replacement")
        return original_read(path)

    monkeypatch.setattr(capture_module, "read_json_closed", read_with_one_replace_failure)
    stop = StopRequest()
    handle = {}
    should_stop = lambda: stop.asked_for_capture(handle)
    frames = follow_capture(
        recorded_capture,
        poll_seconds=0,
        max_idle_polls=1,
        should_stop=should_stop,
        handle=handle,
    )
    first, frames = first_observed_frame(frames)
    assert first.source_seq == 1

    observed = []
    for frame in stop.bounded(frames, should_stop=should_stop):
        observed.append(frame.source_seq)
        if len(observed) == 805:
            manifest.write_text(
                json.dumps({"ended_at": 3197.0, "end_reason": "stop"}),
                encoding="utf-8",
            )
            fault["armed"] = True
            stop.request(StopRequest.SOFT, "test-stdin-close")

    assert fault["raised"]
    assert len(observed) == 3197
    assert observed[-1] == 6477

    # Once normal close is verified, another transient read cannot undo it.
    def unreadable(_path):
        raise OSError("manifest temporarily unavailable")

    monkeypatch.setattr(capture_module, "read_json_closed", unreadable)
    assert not should_stop()
    stop.request(StopRequest.HARD, "test-hard-stop")
    assert should_stop()


def test_builder_process_records_every_frame_after_normal_close_and_soft_stop(
    recorded_capture, tmp_path
):
    """Carry a late close through the CLI observe loop and persisted session.

    The frame bytes are deliberately malformed. The engine still counts each
    observation, while decode and reconstruction do no expensive image work.
    """
    journal = recorded_capture / "frames.jsonl"
    lines = journal.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(lines) == 3197
    journal.write_text("".join(lines[:805]), encoding="utf-8")
    manifest = recorded_capture / "capture.json"
    manifest.write_text(
        json.dumps({"ended_at": None, "end_reason": None}), encoding="utf-8"
    )
    root = tmp_path / "worlds"
    process = subprocess.Popen(
        [
            sys.executable,
            "scripts/world_build_session.py",
            "--follow-capture", str(recorded_capture),
            "--root", str(root),
            "--poll-seconds", "0.01",
            "--max-idle-polls", "10000",
            "--stop-on-stdin-close",
            "--format", "json",
        ],
        cwd=Path(__file__).parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    store = WorldStore(root)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            worlds = store.list_world_ids()
            if len(worlds) == 1:
                sessions = store.list_session_ids(worlds[0])
                if len(sessions) == 1:
                    events_path = store.events_path(worlds[0], sessions[0])
                    if events_path.exists() and events_path.read_text(
                        encoding="utf-8"
                    ).count('"frame_rejected"') >= 805:
                        break
            if process.poll() is not None:
                raise AssertionError("builder exited before the capture closed")
            time.sleep(0.1)
        else:
            raise AssertionError("builder did not observe the first 805 frames")

        assert process.poll() is None
        with journal.open("a", encoding="utf-8") as out:
            out.writelines(lines[805:])
        manifest.write_text(
            json.dumps({"ended_at": 3197.0, "end_reason": "stop"}),
            encoding="utf-8",
        )
        process.stdin.close()
        process.stdin = None
        stdout, stderr = process.communicate(timeout=90)
        assert process.returncode == 0, stderr[-2000:]
        assert "stop requested (soft, stdin-closed) after the capture closed" in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    report = json.loads(stdout)
    assert report["frames_observed"] == 3197
    assert report["rejected_by_reason"]["malformed_frame"] == 3197
    assert report["end_reason"] == "stop"
    assert report["finalization"] == "complete"
    session = store.read_session(report["world_id"], report["session_id"])
    assert session.frames_observed == 3197
    assert session.end_reason == "stop"
    assert session.finalization["state"] == "complete"
    events = store.events_path(report["world_id"], report["session_id"]).read_text(
        encoding="utf-8"
    )
    assert events.count('"frame_rejected"') == 3197
    assert json.loads(events.splitlines()[-1])["kind"] == "session_stopped"
