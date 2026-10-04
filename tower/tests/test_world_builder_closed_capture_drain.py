"""A late World Builder follower drains a normally closed capture on soft stop."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

import tower.capture as capture_module
import scripts.world_build_session as builder_script
from scripts.world_build_session import StopRequest, first_observed_frame, follow_capture
from tower.capture import CaptureRecorder
from tower.capture_workers import CaptureWorkerSupervisor, WorkerSpec
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


@pytest.mark.parametrize("read_error", [OSError, ValueError])
def test_persistent_manifest_read_failure_cannot_publish_partial_complete(
    recorded_capture, tmp_path, monkeypatch, read_error
):
    """A healed post-loop read must not turn a truncated run into Saved."""
    manifest = recorded_capture / "capture.json"
    manifest.write_text(
        json.dumps({"ended_at": None, "end_reason": None}), encoding="utf-8"
    )
    original_read = capture_module.read_json_closed
    fault = {"armed": False, "remaining": 3}

    def read_with_three_failures(path):
        if path == manifest and fault["armed"] and fault["remaining"]:
            fault["remaining"] -= 1
            raise read_error("closed manifest temporarily unreadable")
        return original_read(path)

    monkeypatch.setattr(capture_module, "read_json_closed", read_with_three_failures)
    stop_holder = {}
    monkeypatch.setattr(
        builder_script.StopRequest,
        "install",
        lambda self, **_kwargs: stop_holder.setdefault("request", self),
    )
    original_observe = builder_script.WorldBuilderEngine.observe
    observed = 0

    def observe_and_close(self, *args, **kwargs):
        nonlocal observed
        outcome = original_observe(self, *args, **kwargs)
        observed += 1
        if observed == 805:
            manifest.write_text(
                json.dumps({"ended_at": 3197.0, "end_reason": "stop"}),
                encoding="utf-8",
            )
            fault["armed"] = True
            stop_holder["request"].request(StopRequest.SOFT, "test-soft-stop")
        return outcome

    monkeypatch.setattr(builder_script.WorldBuilderEngine, "observe", observe_and_close)
    root = tmp_path / "worlds"
    exit_code = builder_script.main(
        [
            "--follow-capture", str(recorded_capture),
            "--root", str(root),
            "--poll-seconds", "0.01",
            "--max-idle-polls", "10000",
            "--format", "json",
        ]
    )

    assert observed == 805
    assert fault["remaining"] == 0
    # The manifest is healthy again. The old post-loop read used this to
    # relabel 805/3197 observations as an ordinary, complete capture stop.
    assert capture_module.read_json_closed(manifest)["end_reason"] == "stop"
    assert exit_code == 1
    store = WorldStore(root)
    world_id = store.list_world_ids()[0]
    session_id = store.list_session_ids(world_id)[0]
    session = store.read_session(world_id, session_id)
    assert session.frames_observed == 805
    assert session.end_reason == "error"
    assert session.finalization["state"] == "interrupted"


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


def test_recorder_and_supervisor_drain_late_builder_after_normal_stop(tmp_path):
    """Exercise the real journal writer, close manifest, worker, and soft stop.

    Malformed image bytes avoid decoding, mapping, and GPU work. The recorder
    still fsyncs each frame and atomically closes the same manifest used by
    the WebSocket stream path. The supervisor owns the subprocess and closes
    its stdin exactly as a World Builder session Stop does.
    """
    recorder = CaptureRecorder(tmp_path / "recordings")
    capture_id = recorder.start()
    payload = b"synthetic-frame"
    source_seq = lambda index: 1 + (index * 6476 // 3196)
    # A late attach sees an existing backlog. The writer is complete, but
    # the manifest remains open until the user ends the capture below.
    for index in range(3197):
        assert recorder.write_frame(payload, source_seq=source_seq(index))

    capture_dir = recorder.capture_dir(capture_id)
    root = tmp_path / "worlds"
    tower_dir = Path(__file__).parents[1]
    processes = []
    worker_log = tmp_path / "worker.log"

    with worker_log.open("wb") as output:
        def spawn(argv, **kwargs):
            kwargs["stdout"] = output
            kwargs["stderr"] = output
            process = subprocess.Popen(argv, **kwargs)
            processes.append(process)
            return process

        supervisor = CaptureWorkerSupervisor(
            WorkerSpec(
                argv=(
                    sys.executable,
                    str(tower_dir / "scripts" / "world_build_session.py"),
                    "--follow-capture", "{capture_dir}",
                    "--root", str(root),
                    "--poll-seconds", "0.01",
                    "--max-idle-polls", "10000",
                    "--stop-on-stdin-close",
                ),
                cwd=str(tower_dir),
                name="world-build-session",
                stop_via_stdin=True,
            ),
            spawn=spawn,
        )
        assert supervisor.attach("world-build-session", capture_id, capture_dir)
        assert len(processes) == 1
        process = processes[0]
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
                time.sleep(0.05)
            else:
                raise AssertionError("builder did not observe the first 805 frames")

            assert recorder.status.frames_written == 3197
            assert process.poll() is None
            observed_at_stop = events_path.read_text(
                encoding="utf-8"
            ).count('"frame_rejected"')
            # Keep the reproducer close to the field's 805/3,197 split:
            # at least 2,000 journal entries must still be unobserved when
            # the normal close and stdin soft stop happen.
            assert 805 <= observed_at_stop <= 1197, observed_at_stop
            recorder.stop()
            supervisor.capture_closed(capture_id)
            assert supervisor.request_stop("world-build-session") == 1
            process.wait(timeout=90)
            assert process.returncode == 0, worker_log.read_text(
                encoding="utf-8", errors="replace"
            )[-3000:]
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            supervisor.reap()

    manifest = recorder.manifest(capture_id)
    assert manifest["end_reason"] == "stop"
    assert manifest["frames_written"] == 3197
    assert [row["source_seq"] for row in recorder.read_frames(capture_id)][-1] == 6477
    worlds = store.list_world_ids()
    assert len(worlds) == 1
    sessions = store.list_session_ids(worlds[0])
    assert len(sessions) == 1
    session = store.read_session(worlds[0], sessions[0])
    assert session.frames_observed == 3197
    assert session.end_reason == "stop"
    assert session.finalization["state"] == "complete"
    events = store.events_path(worlds[0], sessions[0]).read_text(encoding="utf-8")
    assert events.count('"frame_rejected"') == 3197
    assert json.loads(events.splitlines()[-1])["kind"] == "session_stopped"
    assert "stop requested (soft, stdin-closed) after the capture closed" in (
        worker_log.read_text(encoding="utf-8", errors="replace")
    )
