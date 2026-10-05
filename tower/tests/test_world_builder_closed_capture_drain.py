"""A late World Builder follower drains a normally closed capture on soft stop."""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import tower.capture as capture_module
import scripts.world_build_session as builder_script
from scripts.world_build_session import StopRequest, first_observed_frame, follow_capture
from tower.capture import CaptureFollower, CaptureRecorder
from tower.capture_workers import CaptureWorkerSupervisor, WorkerSpec
from tower.world_builder.store import WorldStore

# A short budget for the fault-injection tests, so a fault that outlasts it
# costs a fraction of a second. The tests with a REAL lock use the shipped
# `MANIFEST_READ_BUDGET_S`.
SHORT_BUDGET_S = 0.25


def _unreadable_for(monkeypatch, path, seconds, error=OSError):
    """Make `path` unreadable for `seconds` of wall time once armed.

    A lock is held for a DURATION, so the fault is one: it says nothing
    about how many reads a reader makes in that time, and a reader that
    retries more or less often meets the same fault (adversarial review
    LOW-3: the old "fail the first 3 reads" broke on a more tolerant policy
    and let a reader that never backed off pass).
    """
    original_read = capture_module.read_json_closed
    fault = {"until": None, "failures": 0}

    def arm():
        fault["until"] = time.monotonic() + seconds

    def read(candidate):
        if (
            candidate == path
            and fault["until"] is not None
            and time.monotonic() < fault["until"]
        ):
            fault["failures"] += 1
            raise error(f"{candidate.name} is held by another process")
        return original_read(candidate)

    monkeypatch.setattr(capture_module, "read_json_closed", read)
    fault["arm"] = arm
    return fault


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
    # Follower level: how many frames each order reads. What each one is
    # LABELLED is asserted on the stored session, through `main()`, in
    # `test_stop_policy_labels_the_stored_session` below.
    #
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
    # Unreadable for 100 ms from the soft stop: well inside the budget.
    fault = _unreadable_for(monkeypatch, manifest, 0.1, error=ValueError)
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
            fault["arm"]()
            stop.request(StopRequest.SOFT, "test-stdin-close")

    assert fault["failures"] >= 1
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
    """A healed post-loop read must not turn a truncated run into Saved.

    The manifest stays unreadable for longer than the read budget and then
    heals, so the post-loop read sees a clean `stop`.
    """
    manifest = recorded_capture / "capture.json"
    manifest.write_text(
        json.dumps({"ended_at": None, "end_reason": None}), encoding="utf-8"
    )
    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", SHORT_BUDGET_S)
    fault = _unreadable_for(
        monkeypatch, manifest, SHORT_BUDGET_S + 0.5, error=read_error
    )
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
            fault["arm"]()
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
    assert fault["failures"] >= 2
    time.sleep(max(0.0, fault["until"] - time.monotonic()))
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


@pytest.mark.parametrize("read_error", [OSError, ValueError])
def test_manifest_failure_before_first_frame_records_error_session(
    recorded_capture, tmp_path, monkeypatch, caplog, read_error
):
    """Both read errors, because both are caught (standard review M15).

    A `ValueError` is a manifest parsed as truncated JSON; narrowing the
    catch to `OSError` let it escape `main()` before any session existed --
    the very bug 8ede341 fixed for `OSError`.
    """
    manifest = recorded_capture / "capture.json"
    manifest.write_text(
        json.dumps({"ended_at": 3197.0, "end_reason": "stop"}), encoding="utf-8"
    )
    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", SHORT_BUDGET_S)
    fault = _unreadable_for(
        monkeypatch, manifest, SHORT_BUDGET_S + 0.5, error=read_error
    )
    fault["arm"]()

    def install_and_stop(self, **_kwargs):
        self.request(StopRequest.SOFT, "test-pre-first-frame-stop")

    monkeypatch.setattr(builder_script.StopRequest, "install", install_and_stop)
    root = tmp_path / "worlds"
    with caplog.at_level(logging.INFO, logger="tower.world_build_session"):
        exit_code = builder_script.main(
            [
                "--follow-capture", str(recorded_capture),
                "--root", str(root),
                "--poll-seconds", "0.01",
                "--max-idle-polls", "2",
                "--format", "json",
            ]
        )

    assert fault["failures"] >= 2
    assert exit_code == 1
    store = WorldStore(root)
    world_id = store.list_world_ids()[0]
    session_id = store.list_session_ids(world_id)[0]
    session = store.read_session(world_id, session_id)
    assert session.frames_observed == 0
    assert session.end_reason == "error"
    assert session.finalization["state"] == "interrupted"
    # The error is what the log says, not a frame-size miss (standard review
    # LOW-4): no lookup was made, because no frame was ever going to come.
    assert "failed before its first frame" in caplog.text
    assert "could not observe the frame resolution" not in caplog.text


def test_verified_normal_close_finishes_when_later_manifest_reads_fail(
    recorded_capture, tmp_path, monkeypatch
):
    manifest = recorded_capture / "capture.json"
    manifest.write_text(
        json.dumps({"ended_at": None, "end_reason": None}), encoding="utf-8"
    )
    original_read = capture_module.read_json_closed
    fault = {"armed": False, "verified": False, "later_failures": 0}

    def read_once_after_close(path):
        if path == manifest and fault["armed"]:
            if not fault["verified"]:
                fault["verified"] = True
            else:
                fault["later_failures"] += 1
                raise OSError("manifest unavailable after verified normal close")
        return original_read(path)

    monkeypatch.setattr(capture_module, "read_json_closed", read_once_after_close)
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
            "--max-idle-polls", "2",
            "--format", "json",
        ]
    )

    assert fault["verified"]
    assert fault["later_failures"] > 0
    assert exit_code == 0
    store = WorldStore(root)
    world_id = store.list_world_ids()[0]
    session_id = store.list_session_ids(world_id)[0]
    session = store.read_session(world_id, session_id)
    assert session.frames_observed == 3197
    assert session.end_reason == "stop"
    assert session.finalization["state"] == "complete"


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


# The builder at a pace the field's late builder kept on Oct 2 or slower:
# every observe padded, so it is provably still behind 11 s after the close.
_PACED_BUILDER = """
import sys, time
import scripts.world_build_session as builder
_observe = builder.WorldBuilderEngine.observe
def _paced(self, *args, **kwargs):
    outcome = _observe(self, *args, **kwargs)
    time.sleep(%r)
    return outcome
builder.WorldBuilderEngine.observe = _paced
sys.exit(builder.main(sys.argv[1:]))
"""
OCT2_PACE_S = 0.006


def test_the_oct2_order_close_then_a_disconnect_stop_11s_later_then_a_reconnect(
    tmp_path,
):
    """The Oct 2 incident's exact order (manager 164), end to end.

    A late builder is attached to a 3,197-frame capture and is still BEHIND
    when the wearer's Stop closes it (`stop`) at 805 observed. 11 s later the
    last client disconnects: `ws.py` stops the cartridge session, the World
    Builder's stop policy is `request`, and the supervisor closes the
    builder's stdin -- the soft stop. 0.9 s after that the phone reconnects
    and the workspace re-sends `session/start`; nothing is recording, so
    nothing attaches. 44fbd13 stopped at the soft request and stored about
    1,392 of 3,197 as `stop` / `complete`. Now the builder reads every frame
    the capture recorded -- all of them were recorded before the Stop.

    Real recorder, real supervisor, real `CartridgeSession`, a builder
    SUBPROCESS with its real stdin watcher; malformed frames, so no image
    work. The builder is padded to `OCT2_PACE_S` per frame, so at the soft
    stop it has read at most 805 + 11 / OCT2_PACE_S < 3,197.
    """
    from tower.cartridge_session import CartridgeSession

    recorder = CaptureRecorder(tmp_path / "recordings")
    capture_id = recorder.start()
    for index in range(TOTAL):
        assert recorder.write_frame(
            b"synthetic-frame", source_seq=1 + (index * 6476 // 3196)
        )
    root = tmp_path / "worlds"
    tower_dir = Path(__file__).parents[1]
    worker_log = tmp_path / "worker.log"
    processes = []
    timeline = {}

    def open_capture():
        status = recorder.status
        if status is None or not status.is_open:
            return None
        return status.capture_id, recorder.capture_dir(status.capture_id)

    def observed(store):
        worlds = store.list_world_ids()
        if len(worlds) != 1 or len(store.list_session_ids(worlds[0])) != 1:
            return 0
        events = store.events_path(worlds[0], store.list_session_ids(worlds[0])[0])
        if not events.exists():
            return 0
        return events.read_text(encoding="utf-8").count('"frame_rejected"')

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
                    sys.executable, "-c", _PACED_BUILDER % OCT2_PACE_S,
                    "--follow-capture", "{capture_dir}",
                    "--root", str(root),
                    "--max-idle-polls", "3600",
                    "--stop-on-stdin-close",
                ),
                cwd=str(tower_dir),
                name="world-build-session",
                stop_via_stdin=True,
                stop_grace_seconds=60.0,
            ),
            spawn=spawn,
        )
        session = CartridgeSession(
            cartridge="world_builder",
            worker="world-build-session",
            supervisor=supervisor,
            open_capture=open_capture,
            clock=time.time,
            stop_policy="request",
        )
        assert session.start()["attached_capture_id"] == capture_id
        assert len(processes) == 1
        process = processes[0]
        store = WorldStore(root)
        try:
            deadline = time.monotonic() + 120
            while observed(store) < 805:
                assert process.poll() is None, worker_log.read_text(errors="replace")[-3000:]
                assert time.monotonic() < deadline, "the builder never reached 805"
                time.sleep(0.02)
            timeline["at_close"] = observed(store)
            recorder.stop()  # the wearer's Stop: the capture closes `stop`
            supervisor.capture_closed(capture_id)
            time.sleep(11.0)  # +11 s: the last client disconnects
            timeline["at_soft_stop"] = observed(store)
            session.stop()  # `_stop_cartridge_sessions` -> stdin closed
            time.sleep(0.9)  # +0.9 s: the phone reconnects
            timeline["reattached"] = session.start().get("attached_capture_id")
            process.wait(timeout=180)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            supervisor.reap()

    log = worker_log.read_text(encoding="utf-8", errors="replace")
    assert process.returncode == 0, log[-3000:]
    assert 805 <= timeline["at_close"] < 1000, timeline
    assert timeline["at_soft_stop"] < TOTAL, timeline  # still behind
    assert timeline["reattached"] is None, timeline
    assert len(processes) == 1
    assert recorder.manifest(capture_id)["frames_written"] == TOTAL
    worlds = store.list_world_ids()
    record = store.read_session(worlds[0], store.list_session_ids(worlds[0])[0])
    outcome = (record.frames_observed, record.end_reason, record.finalization["state"])
    assert outcome == (TOTAL, "stop", "complete"), (outcome, timeline)
    assert "stop requested (soft, stdin-closed) after the capture closed (stop)" in log
    assert "were never observed" not in log


# ---------------------------------------------------------------------------
# A PARTIAL WORLD IS NEVER `stop` / `complete`.
#
# The drain above fixes one stop order. These are the others both reviews of
# 8ede341 reproduced (STD HIGH-1/MED-1, ADV HIGH-1 P1-P4): every one of them
# left recorded frames unread and still published the session as an ordinary
# finished walk, because the label asked the manifest how the capture ENDED
# and never asked whether the builder had READ it. Each drives `main()`, so
# the assertion is on the stored session, not on a follower count.
# ---------------------------------------------------------------------------

TOTAL = 3197


def _write_manifest(directory, end_reason, **extra):
    (directory / "capture.json").write_text(
        json.dumps(
            {
                "ended_at": None if end_reason is None else 3197.0,
                "end_reason": end_reason,
                **extra,
            }
        ),
        encoding="utf-8",
    )


def _journal_rows(count, start=0):
    return "".join(
        json.dumps(
            {
                "source_seq": 1 + (index * 6476 // 3196),
                "received_at": float(index),
                "relpath": "frame.jpg",
            }
        )
        + "\n"
        for index in range(start, count)
    )


def _hook(monkeypatch, on_observe=None, *, install_level=None):
    """Capture the session's StopRequest and call `on_observe(n, stop)`."""
    holder = {"observed": 0}

    def install(self, **_kwargs):
        holder["stop"] = self
        if install_level is not None:
            self.request(install_level, "test-install")

    monkeypatch.setattr(builder_script.StopRequest, "install", install)
    original_observe = builder_script.WorldBuilderEngine.observe

    def observe(self, *args, **kwargs):
        outcome = original_observe(self, *args, **kwargs)
        holder["observed"] += 1
        if on_observe is not None:
            on_observe(holder["observed"], holder["stop"])
        return outcome

    monkeypatch.setattr(builder_script.WorldBuilderEngine, "observe", observe)
    return holder


def _run_session(capture_dir, root, *, max_idle_polls="10000"):
    exit_code = builder_script.main(
        [
            "--follow-capture", str(capture_dir),
            "--root", str(root),
            "--poll-seconds", "0.01",
            "--max-idle-polls", max_idle_polls,
            "--format", "json",
        ]
    )
    store = WorldStore(root)
    world_id = store.list_world_ids()[0]
    session = store.read_session(world_id, store.list_session_ids(world_id)[0])
    return exit_code, session


def _labelled_finished(session):
    return session.end_reason == "stop" and session.finalization["state"] == "complete"


@pytest.mark.parametrize(
    ("close_reason", "stops", "expected_observed", "expected_end"),
    [
        # The fix 8ede341 made: a soft stop on a normal close drains.
        pytest.param("stop", {805: "soft"}, TOTAL, "stop", id="normal-close-soft-drains"),
        # Frames were still coming: unchanged since 44fbd13.
        pytest.param(None, {805: "soft"}, 805, "interrupted", id="open-soft"),
        # Tower shutdown at the moment of the close.
        pytest.param("stop", {805: "hard"}, 805, "interrupted", id="normal-close-hard"),
        # STD MED-1 / ADV P1: the Tower shuts down WHILE the drain runs.
        pytest.param(
            "stop", {805: "soft", 1500: "hard"}, 1500, "interrupted",
            id="hard-stop-during-drain-1500",
        ),
        # STD HIGH-1 R1 / ADV P3: the link died with a backlog, then a soft
        # stop (ws.py's deferred stop after the resume grace). The backlog is
        # NOT drained -- M16 showed that waits out the successor grace -- but
        # it is not labelled a finished walk either.
        pytest.param(
            "disconnect", {805: "soft"}, 805, "interrupted",
            id="disconnect-backlog-soft-805",
        ),
    ],
)
def test_stop_policy_labels_the_stored_session(
    recorded_capture, tmp_path, monkeypatch,
    close_reason, stops, expected_observed, expected_end,
):
    _write_manifest(recorded_capture, None)

    def on_observe(n, stop):
        level = stops.get(n)
        if level is None:
            return
        if n == 805 and close_reason is not None:
            _write_manifest(recorded_capture, close_reason)
        stop.request(StopRequest.SOFT if level == "soft" else StopRequest.HARD, "test")

    _hook(monkeypatch, on_observe)
    exit_code, session = _run_session(recorded_capture, tmp_path / "worlds")

    assert exit_code == 0
    assert session.frames_observed == expected_observed
    assert session.end_reason == expected_end
    assert session.finalization["state"] == "complete"
    if expected_observed < TOTAL:
        assert not _labelled_finished(session)


def test_soft_stop_decided_just_before_the_close_is_interrupted(
    recorded_capture, tmp_path, monkeypatch
):
    """ADV P2: `session/stop` (HTTP) can beat `stream_stop` (WS).

    The builder asks while the manifest is still open, decides to stop, and
    the recorder's normal close lands before the post-loop re-read. 8ede341
    then read `stop` and published 805 of 3,197 as a finished walk.
    """
    _write_manifest(recorded_capture, None)
    _hook(
        monkeypatch,
        lambda n, stop: stop.request(StopRequest.SOFT, "test") if n == 805 else None,
    )
    original_bounded = builder_script.StopRequest.bounded

    def bounded_then_recorder_closes(self, frames, **kwargs):
        yield from original_bounded(self, frames, **kwargs)
        if self.asked:
            _write_manifest(recorded_capture, "stop")

    monkeypatch.setattr(
        builder_script.StopRequest, "bounded", bounded_then_recorder_closes
    )
    exit_code, session = _run_session(recorded_capture, tmp_path / "worlds")

    assert capture_module.read_json_closed(recorded_capture / "capture.json")[
        "end_reason"
    ] == "stop"
    assert exit_code == 0
    assert session.frames_observed == 805
    assert session.end_reason == "interrupted"
    assert not _labelled_finished(session)


def test_soft_stop_in_a_disconnected_backlog_across_a_reconnect_is_interrupted(
    tmp_path, monkeypatch
):
    """STD HIGH-1 R3: 805 of 3,697 recorded frames, across a reconnect.

    Capture A ends `disconnect` with a backlog; successor B continues it and
    closes normally; the wearer's Stop lands while the builder is still in
    A's backlog. Nothing of A's tail or of B was observed.
    """
    captures = tmp_path / "captures"
    captures.mkdir()
    first = captures / ("a" * 32)
    first.mkdir()
    (first / "frame.jpg").write_bytes(b"synthetic-frame")
    (first / "frames.jsonl").write_text(_journal_rows(TOTAL), encoding="utf-8")
    _write_manifest(first, None)

    def on_observe(n, stop):
        if n != 805:
            return
        _write_manifest(first, "disconnect")
        successor = captures / ("b" * 32)
        successor.mkdir()
        (successor / "frame.jpg").write_bytes(b"synthetic-frame")
        (successor / "frames.jsonl").write_text(
            _journal_rows(TOTAL + 500, start=TOTAL), encoding="utf-8"
        )
        _write_manifest(successor, "stop", continues_capture=first.name)
        stop.request(StopRequest.SOFT, "test")

    _hook(monkeypatch, on_observe)
    exit_code, session = _run_session(first, tmp_path / "worlds")

    assert exit_code == 0
    assert session.frames_observed == 805
    assert session.end_reason == "interrupted"
    assert not _labelled_finished(session)


def test_hard_stop_before_the_first_frame_of_a_closed_capture_is_interrupted(
    recorded_capture, tmp_path, monkeypatch
):
    """ADV P4: 0 of 3,197 observed, and 8ede341 stored it `stop`/`complete`."""
    _write_manifest(recorded_capture, "stop")
    _hook(monkeypatch, install_level=StopRequest.HARD)
    exit_code, session = _run_session(recorded_capture, tmp_path / "worlds")

    assert exit_code == 0
    assert session.frames_observed == 0
    assert session.end_reason == "interrupted"
    assert not _labelled_finished(session)


def test_a_closed_capture_whose_last_journal_read_failed_is_not_finished(
    recorded_capture, tmp_path, monkeypatch
):
    """No stop at all, and still part of a walk.

    The follower's one journal read after the close comes back empty --
    which is what `_JournalTail.read_new` returns on an `OSError`, e.g. a
    sharing violation on `frames.jsonl` -- so the follower returns with the
    last 2,392 recorded frames unread. The label must not call that a
    finished walk; 8ede341 did.
    """
    journal = recorded_capture / "frames.jsonl"
    lines = journal.read_text(encoding="utf-8").splitlines(keepends=True)
    journal.write_text("".join(lines[:805]), encoding="utf-8")
    _write_manifest(recorded_capture, None)
    fail_next = {"armed": False, "failed": 0}
    original_read_new = capture_module._JournalTail.read_new

    def read_new(self):
        if fail_next["armed"]:
            fail_next["armed"] = False
            fail_next["failed"] += 1
            return []
        return original_read_new(self)

    monkeypatch.setattr(capture_module._JournalTail, "read_new", read_new)

    def on_observe(n, _stop):
        if n == 805:
            with journal.open("a", encoding="utf-8") as out:
                out.writelines(lines[805:])
            _write_manifest(recorded_capture, "stop")
            fail_next["armed"] = True

    _hook(monkeypatch, on_observe)
    exit_code, session = _run_session(recorded_capture, tmp_path / "worlds")

    assert fail_next["failed"] == 1
    assert exit_code == 0
    assert session.frames_observed == 805
    assert session.end_reason == "interrupted"


# ---------------------------------------------------------------------------
# EVERY CAPTURE OF THE WALK IS COUNTED (re-review of 40a4883, HIGH-1).
#
# The same failed post-close read, on a capture a reconnect then LEFT. The
# follower rebinds to the successor anyway, reads all of it, and a guard that
# asked only the capture it ended on saw nothing unread: 1,305 of 3,697
# recorded frames stored `stop` / `complete`, with no stop at all. The
# reviewer reproduced it with a real Windows sharing violation, and so does
# the last test of this block.
# ---------------------------------------------------------------------------


def _capture_dir(directory, rows, end_reason=None, **extra):
    directory.mkdir(parents=True)
    (directory / "frame.jpg").write_bytes(b"synthetic-frame")
    (directory / "frames.jsonl").write_text(rows, encoding="utf-8")
    _write_manifest(directory, end_reason, **extra)
    return directory


def _lineage_whose_closing_read_fails(
    tmp_path, monkeypatch, *, real_lock, soft_after_close=False
):
    """A (805 read of 3,197) ends `disconnect`; the one read of A after its
    close fails; B (500) continues A and closes `stop`."""
    captures = tmp_path / "captures"
    first = _capture_dir(captures / ("a" * 32), _journal_rows(805))
    state = {"armed": False, "failed": 0, "timer": None}
    if not real_lock:
        original_read_new = capture_module._JournalTail.read_new

        def read_new(self):
            if state["armed"]:
                state["armed"] = False
                state["failed"] += 1
                return []
            return original_read_new(self)

        monkeypatch.setattr(capture_module._JournalTail, "read_new", read_new)

    def on_observe(n, stop):
        if n == 805:
            with (first / "frames.jsonl").open("a", encoding="utf-8") as out:
                out.write(_journal_rows(TOTAL, start=805))
            _write_manifest(first, "disconnect")
            state["second"] = _capture_dir(
                captures / ("b" * 32),
                _journal_rows(TOTAL + 500, start=TOTAL),
                continues_capture=first.name,
            )
            if real_lock:
                # Share mode 0 on A's journal from A's close: the follower's
                # one read after the close meets a sharing violation.
                state["timer"] = _hold(first / "frames.jsonl", 0.5, "exclusive-open")
                with pytest.raises(PermissionError):
                    (first / "frames.jsonl").open("rb")
            else:
                state["armed"] = True
        if n == 805 + 500:
            _write_manifest(state["second"], "stop", continues_capture=first.name)
            if soft_after_close:
                stop.request(StopRequest.SOFT, "test")

    _hook(monkeypatch, on_observe)
    try:
        exit_code, session = _run_session(first, tmp_path / "worlds")
    finally:
        if state["timer"] is not None:
            state["timer"].join()
    return exit_code, session, state


@pytest.mark.parametrize(
    "soft_after_close", [False, True], ids=["no-stop", "soft-after-successor-close"]
)
def test_a_predecessor_whose_closing_read_failed_is_not_finished(
    tmp_path, monkeypatch, caplog, soft_after_close
):
    with caplog.at_level(logging.WARNING, logger="tower.world_build_session"):
        exit_code, session, state = _lineage_whose_closing_read_fails(
            tmp_path, monkeypatch, real_lock=False, soft_after_close=soft_after_close
        )

    assert state["failed"] == 1
    assert exit_code == 0
    assert session.frames_observed == 805 + 500
    assert session.end_reason == "interrupted"
    assert not _labelled_finished(session)
    # A's 2,392 unread rows, counted from the successor B the follower ended on.
    assert "2392 of the frames recorded in this walk" in caplog.text


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_a_real_sharing_violation_on_a_predecessor_journal_is_not_finished(
    tmp_path, monkeypatch
):
    exit_code, session, _state = _lineage_whose_closing_read_fails(
        tmp_path, monkeypatch, real_lock=True
    )

    assert exit_code == 0
    assert session.frames_observed == 805 + 500
    assert session.end_reason == "interrupted"
    assert not _labelled_finished(session)


def test_a_successor_whose_closing_read_failed_is_not_finished(tmp_path, monkeypatch):
    """The re-review's Q10, which kills its RV-M4: a rebind that left the
    follower measuring the PREDECESSOR's journal made the successor's unread
    tail invisible -- HIGH-1's twin on the other side of the reconnect."""
    captures = tmp_path / "captures"
    first = _capture_dir(captures / ("a" * 32), _journal_rows(24))
    state = {"armed": False, "failed": 0}
    original_read_new = capture_module._JournalTail.read_new

    def read_new(self):
        if state["armed"]:
            state["armed"] = False
            state["failed"] += 1
            return []
        return original_read_new(self)

    monkeypatch.setattr(capture_module._JournalTail, "read_new", read_new)

    def on_observe(n, _stop):
        if n == 24:
            _write_manifest(first, "disconnect")
            state["second"] = _capture_dir(
                captures / ("b" * 32),
                _journal_rows(48, start=24),
                continues_capture=first.name,
            )
        if n == 48:
            with (state["second"] / "frames.jsonl").open("a", encoding="utf-8") as out:
                out.write(_journal_rows(548, start=48))
            _write_manifest(state["second"], "stop", continues_capture=first.name)
            state["armed"] = True

    _hook(monkeypatch, on_observe)
    exit_code, session = _run_session(first, tmp_path / "worlds")

    assert state["failed"] == 1
    assert exit_code == 0
    assert session.frames_observed == 48
    assert session.end_reason == "interrupted"
    assert not _labelled_finished(session)


@pytest.mark.parametrize("stop_level", [None, StopRequest.SOFT], ids=["no-stop", "soft"])
def test_a_capture_that_recorded_no_frame_has_no_journal_and_is_still_stop(
    tmp_path, monkeypatch, stop_level
):
    """The re-review's Q9, which kills its RV-M1. The recorder writes no
    `frames.jsonl` until its first frame, so a phone that connected and
    dropped leaves a closed capture with NO journal -- and a manifest that
    says `frames_written: 0`, which is how that is told apart from a journal
    that went missing (Codex H2). Nothing recorded is unread: a finished,
    empty walk, on 44fbd13 and here."""
    capture = tmp_path / "captures" / ("e" * 32)
    capture.mkdir(parents=True)
    _write_manifest(capture, "stop", frames_written=0)
    _hook(monkeypatch, install_level=stop_level)
    exit_code, session = _run_session(capture, tmp_path / "worlds", max_idle_polls="5")

    assert not (capture / "frames.jsonl").exists()
    assert exit_code == 0
    assert session.frames_observed == 0
    assert session.end_reason == "stop"


def test_a_real_recording_with_no_frame_is_still_stop(tmp_path, monkeypatch):
    """The same, written by the real recorder: start, then Stop, no frame."""
    recorder = CaptureRecorder(tmp_path / "recordings")
    capture_id = recorder.start()
    recorder.stop()
    _hook(monkeypatch, install_level=StopRequest.SOFT)
    exit_code, session = _run_session(
        recorder.capture_dir(capture_id), tmp_path / "worlds", max_idle_polls="5"
    )

    assert not (recorder.capture_dir(capture_id) / "frames.jsonl").exists()
    assert recorder.manifest(capture_id)["frames_written"] == 0
    assert exit_code == 0
    assert session.end_reason == "stop"


@pytest.mark.parametrize(
    "manifest_says",
    [pytest.param({"frames_written": 10}, id="frames_written-10"), pytest.param({}, id="silent")],
)
def test_a_capture_whose_journal_is_missing_before_the_first_read_is_not_finished(
    tmp_path, monkeypatch, caplog, manifest_says
):
    """Codex H2's second form (the re-review's h2c): a closed capture whose
    manifest says frames were written -- or does not say -- and whose journal
    is gone before the builder read anything. 6124de8 answered "0 unread" and
    stored 0 of 10 as `stop` / `complete`."""
    capture = tmp_path / "captures" / ("e" * 32)
    capture.mkdir(parents=True)
    _write_manifest(capture, "stop", **manifest_says)
    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", SHORT_BUDGET_S)
    _hook(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="tower.world_build_session"):
        exit_code, session = _run_session(capture, tmp_path / "worlds", max_idle_polls="5")

    assert exit_code == 0
    assert session.frames_observed == 0
    assert session.end_reason == "interrupted"
    assert "an unknown number of the frames" in caplog.text


def test_a_journal_that_vanished_after_a_read_cannot_be_measured(tmp_path, monkeypatch):
    """Re-review LOW-2: a missing journal is "nothing recorded" only while
    nothing has been read from it. One that disappears after the follower
    read part of it cannot say what lay past the read position."""
    directory = _small_closed_capture(tmp_path)
    _write_manifest(directory, None)
    follower = CaptureFollower(directory, poll_seconds=0)
    frames = follower.follow(max_idle_polls=1)
    for _ in range(4):
        next(frames)
    assert follower.unobserved_records() == 7
    os.replace(directory / "frames.jsonl", tmp_path / "moved.jsonl")

    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", 0.05)
    assert follower.unobserved_records() is None
    # A follower that never read anything answers 0 for no journal only if
    # the capture says it wrote no frame. This one wrote ten (Codex H2).
    _write_manifest(directory, "stop", frames_written=10)
    assert CaptureFollower(directory).unobserved_records() is None
    _write_manifest(directory, "stop", frames_written=0)
    assert CaptureFollower(directory).unobserved_records() == 0


# ---------------------------------------------------------------------------
# What the follower counts as unread.
# ---------------------------------------------------------------------------


def _small_closed_capture(tmp_path, count=10):
    directory = tmp_path / "small"
    directory.mkdir()
    (directory / "frame.jpg").write_bytes(b"synthetic-frame")
    (directory / "frames.jsonl").write_text(_journal_rows(count), encoding="utf-8")
    _write_manifest(directory, "stop")
    return directory


def test_a_follower_read_to_the_end_has_nothing_unobserved(tmp_path):
    follower = CaptureFollower(_small_closed_capture(tmp_path), poll_seconds=0)
    assert len(list(follower.follow(max_idle_polls=1))) == 10
    assert follower.unobserved_records() == 0


def test_a_frame_is_taken_only_when_the_consumer_asks_for_the_next(tmp_path):
    follower = CaptureFollower(_small_closed_capture(tmp_path), poll_seconds=0)
    frames = follower.follow(max_idle_polls=1)
    for _ in range(4):
        next(frames)
    # Six never handed on, plus the fourth: received, but a consumer that
    # stops here may have dropped it, as `StopRequest.bounded` does.
    assert follower.unobserved_records() == 7
    next(frames)
    assert follower.unobserved_records() == 6


def test_records_appended_past_the_read_position_are_unobserved(tmp_path):
    directory = _small_closed_capture(tmp_path)
    follower = CaptureFollower(directory, poll_seconds=0)
    assert len(list(follower.follow(max_idle_polls=1))) == 10
    with (directory / "frames.jsonl").open("a", encoding="utf-8") as out:
        out.write(_journal_rows(13, start=10))
        out.write('{"torn": ')
    # Three complete records; the torn trailing piece is not a frame.
    assert follower.unobserved_records() == 3


def _land_with_the_close(directory, start, stop):
    """Frames the recorder appends just before its close manifest: the
    follower meets them in its one journal read AFTER seeing the close."""
    with (directory / "frames.jsonl").open("a", encoding="utf-8") as out:
        out.write(_journal_rows(stop, start=start))
    _write_manifest(directory, "stop")


def test_frames_that_land_with_the_close_are_accounted(tmp_path):
    directory = _small_closed_capture(tmp_path)
    _write_manifest(directory, None)
    follower = CaptureFollower(directory, poll_seconds=0)
    frames = follower.follow(max_idle_polls=5)
    for _ in range(10):
        next(frames)
    _land_with_the_close(directory, 10, 13)

    next(frames)
    # The eleventh, received but not yet released, and the two behind it.
    assert follower.unobserved_records() == 3
    assert len(list(frames)) == 2
    assert follower.unobserved_records() == 0


def test_a_drain_whose_frames_land_with_the_close_is_still_stop(
    recorded_capture, tmp_path, monkeypatch
):
    """The 2,392-frame tail arrives in the follower's post-close read."""
    journal = recorded_capture / "frames.jsonl"
    journal.write_text(_journal_rows(805), encoding="utf-8")
    _write_manifest(recorded_capture, None)

    def on_observe(n, stop):
        if n == 805:
            _land_with_the_close(recorded_capture, 805, TOTAL)
            stop.request(StopRequest.SOFT, "test")

    _hook(monkeypatch, on_observe)
    exit_code, session = _run_session(recorded_capture, tmp_path / "worlds")

    assert exit_code == 0
    assert session.frames_observed == TOTAL
    assert session.end_reason == "stop"
    assert session.finalization["state"] == "complete"


def test_a_journal_that_cannot_be_measured_is_not_a_finished_walk(
    recorded_capture, tmp_path, monkeypatch, caplog
):
    """None from the journal is "not shown", never zero -- also after the
    measurement has been retried for its whole budget."""
    _write_manifest(recorded_capture, "stop")
    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", SHORT_BUDGET_S)
    monkeypatch.setattr(
        capture_module._JournalTail, "records_not_yet_read", lambda self, **_: None
    )
    _hook(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="tower.world_build_session"):
        exit_code, session = _run_session(recorded_capture, tmp_path / "worlds")

    assert exit_code == 0
    assert session.frames_observed == TOTAL
    assert session.end_reason == "interrupted"
    assert "an unknown number of the frames" in caplog.text


@pytest.mark.parametrize(("start_at_end", "expected"), [(False, 10), (True, 0)])
def test_a_follower_that_never_started_measures_from_where_it_would_have(
    tmp_path, start_at_end, expected
):
    follower = CaptureFollower(
        _small_closed_capture(tmp_path), start_at_end=start_at_end
    )
    assert follower.unobserved_records() == expected


# ---------------------------------------------------------------------------
# NO BIG BANG: caught up at Stop, the path is exactly 44fbd13's.
#
# The identity test the manager's no-switch exemption rests on lives in
# `test_world_builder_stop_orders_identity.py`: every caught-up stop order,
# on valid frames, file bytes and event payloads against a golden recorded on
# 44fbd13 (manager 163 §1, 166 §1). It replaced the malformed-frame A/B that
# stood here (Codex L5).
# ---------------------------------------------------------------------------

CAUGHT_UP_FRAMES = 48


# ---------------------------------------------------------------------------
# ONLY WHAT WAS RECORDED BY THE STOP COUNTS (re-review MED-1, lead decision),
# AND A DRAIN BUILDS NOTHING RECORDED AFTER IT (Codex H1), ON A CLOCK NOBODY
# CAN STEP (Codex H3).
#
# The guard compares each unread journal record's `received_monotonic` with
# the builder's `StopRequest.soft_requested_monotonic`. Both are
# `time.monotonic()` on the Tower host: the recorder's when `write_frame`
# began, and this process's when the first soft request was recorded. A frame
# received at or before that moment and never observed still makes the walk
# `interrupted`; one received after it is not part of the walk, and a drain
# ends at the first of them. A record that cannot say when it arrived is
# counted, and drained. A hard stop alone counts everything.
#
# Rows here carry both stamps, as the recorder writes them. Where the order
# depends on what the recorder writes, the real `CaptureRecorder` writes it.
# ---------------------------------------------------------------------------


def _row(source_seq, received_at, received_monotonic=None, **extra):
    record = {"source_seq": source_seq, "relpath": "frame.jpg", **extra}
    if received_at is not None:
        record["received_at"] = received_at
    if received_monotonic is not None:
        record["received_monotonic"] = received_monotonic
    return json.dumps(record) + "\n"


def _now():
    return {"at": time.time(), "mono": time.monotonic()}


def _after_the_stop(stopped, count, first_seq):
    """`count` rows received after the Stop, at 12 fps, on both clocks."""
    return "".join(
        _row(first_seq + i, stopped["at"] + (i + 1) / 12, stopped["mono"] + (i + 1) / 12)
        for i in range(count)
    )


def _caught_up_stop_while_open(
    tmp_path, monkeypatch, *, level, before=lambda now: "", after=lambda stopped: ""
):
    """Caught up at frame 48 of an open capture, `level` stop. `before(now)`
    is appended just BEFORE the request, so it is recorded before the stop
    and never read; `after(stopped)` once the observe loop has ended, then
    the recorder's normal close lands before the post-loop read."""
    capture = tmp_path / "captures" / ("c" * 32)
    capture.mkdir(parents=True)
    (capture / "frame.jpg").write_bytes(b"synthetic-frame")
    journal = capture / "frames.jsonl"
    journal.write_text(_journal_rows(CAUGHT_UP_FRAMES), encoding="utf-8")
    _write_manifest(capture, None)
    stopped = {}

    def on_observe(n, stop):
        if n != CAUGHT_UP_FRAMES:
            return
        with journal.open("a", encoding="utf-8") as out:
            out.write(before(_now()))
        stop.request(level, "test")
        stopped.update(_now())

    _hook(monkeypatch, on_observe)
    original_bounded = builder_script.StopRequest.bounded

    def bounded_then_record_then_close(self, frames, **kwargs):
        yield from original_bounded(self, frames, **kwargs)
        with journal.open("a", encoding="utf-8") as out:
            out.write(after(stopped))
        _write_manifest(capture, "stop")

    monkeypatch.setattr(
        builder_script.StopRequest, "bounded", bounded_then_record_then_close
    )
    return _run_session(capture, tmp_path / "worlds", max_idle_polls="50")


def _three_after(stopped):
    return _after_the_stop(stopped, 3, 49)


def test_a_frame_recorded_before_the_soft_stop_and_never_read_still_counts(
    tmp_path, monkeypatch, caplog
):
    with caplog.at_level(logging.WARNING, logger="tower.world_build_session"):
        exit_code, session = _caught_up_stop_while_open(
            tmp_path, monkeypatch, level=StopRequest.SOFT,
            before=lambda now: _row(49, now["at"] - 0.05, now["mono"] - 0.05),
            after=lambda stopped: _after_the_stop(stopped, 3, 50),
        )

    assert exit_code == 0
    assert session.frames_observed == CAUGHT_UP_FRAMES
    assert session.end_reason == "interrupted"
    # One of the four unread: the three received after the Stop do not count.
    assert "1 of the frames recorded in this walk before the stop" in caplog.text


_NOT_FINITE_ROW = (
    '{"source_seq": 52, "relpath": "frame.jpg", "received_monotonic": Infinity}\n'
)
_UNPARSEABLE_ROW = '{"source_seq": 52, "received_monotonic": \n'


@pytest.mark.parametrize(
    "odd_row",
    [
        # A journal written before the field existed, or by hand. Its WALL
        # stamp says "after the Stop", and the wall clock decides nothing.
        pytest.param(
            lambda stopped: _row(52, stopped["at"] + 5.0), id="no-received_monotonic"
        ),
        pytest.param(
            lambda stopped: _row(52, stopped["at"] + 5.0, "soon"),
            id="received_monotonic-not-a-number",
        ),
        pytest.param(lambda stopped: _NOT_FINITE_ROW, id="received_monotonic-not-finite"),
        pytest.param(lambda stopped: _UNPARSEABLE_ROW, id="unparseable"),
    ],
)
def test_an_unread_record_that_cannot_say_when_it_arrived_counts(
    tmp_path, monkeypatch, caplog, odd_row
):
    with caplog.at_level(logging.WARNING, logger="tower.world_build_session"):
        exit_code, session = _caught_up_stop_while_open(
            tmp_path, monkeypatch, level=StopRequest.SOFT,
            after=lambda stopped: _three_after(stopped) + odd_row(stopped),
        )

    assert exit_code == 0
    assert session.frames_observed == CAUGHT_UP_FRAMES
    assert session.end_reason == "interrupted"
    assert "1 of the frames recorded in this walk before the stop" in caplog.text


def test_frames_recorded_after_a_hard_stop_still_count(tmp_path, monkeypatch, caplog):
    """A hard stop is the Tower shutting down mid-walk, not the wearer's
    Stop: what the camera went on recording is still the walk, so nothing
    is excluded (the re-review's design; the lead's decision is for SOFT)."""
    with caplog.at_level(logging.WARNING, logger="tower.world_build_session"):
        exit_code, session = _caught_up_stop_while_open(
            tmp_path, monkeypatch, level=StopRequest.HARD, after=_three_after,
        )

    assert exit_code == 0
    assert session.frames_observed == CAUGHT_UP_FRAMES
    assert session.end_reason == "interrupted"
    assert "3 of the frames recorded in this walk were never observed" in caplog.text


def _past_the_stop(stop):
    """Wait out the clock tick the soft request was stamped in.

    `time.monotonic()` ticks every 15.625 ms on Windows; a frame received in
    the Stop's own tick is stamped AT the Stop and counts as before it (the
    conservative tie). A frame meant to be AFTER the Stop is received after,
    on both clocks -- so this also runs on a tree that compared wall stamps.
    """
    mono = getattr(stop, "soft_requested_monotonic", None) or time.monotonic()
    wall = getattr(stop, "soft_requested_at", None) or time.time()
    while time.monotonic() <= mono or time.time() <= wall:
        time.sleep(0.001)


@pytest.mark.parametrize("before_stop", [0, 1], ids=["all-after-the-stop", "one-before-it"])
def test_the_recorders_own_stamps_decide_what_was_after_the_stop(
    tmp_path, monkeypatch, caplog, before_stop
):
    """The real `CaptureRecorder` stamps `received_monotonic` with its own
    clock, so this pins the clock the cutoff is compared on."""
    recorder = CaptureRecorder(tmp_path / "recordings")
    capture_id = recorder.start()
    for index in range(45):
        assert recorder.write_frame(b"synthetic-frame", source_seq=index + 1)
    stop_holder = {}

    def on_observe(n, stop):
        if n != 45:
            return
        for index in range(before_stop):
            assert recorder.write_frame(b"synthetic-frame", source_seq=46 + index)
        stop.request(StopRequest.SOFT, "test")
        stop_holder["stop"] = stop

    _hook(monkeypatch, on_observe)
    original_bounded = builder_script.StopRequest.bounded

    def bounded_then_record_then_close(self, frames, **kwargs):
        yield from original_bounded(self, frames, **kwargs)
        _past_the_stop(stop_holder["stop"])
        for index in range(3):
            assert recorder.write_frame(b"synthetic-frame", source_seq=50 + index)
        recorder.stop()

    monkeypatch.setattr(
        builder_script.StopRequest, "bounded", bounded_then_record_then_close
    )
    with caplog.at_level(logging.INFO, logger="tower.world_build_session"):
        exit_code, session = _run_session(
            recorder.capture_dir(capture_id), tmp_path / "worlds", max_idle_polls="50"
        )

    rows = recorder.read_frames(capture_id)
    assert all(isinstance(row["received_monotonic"], float) for row in rows)
    assert recorder.manifest(capture_id)["frames_written"] == 45 + before_stop + 3
    assert exit_code == 0
    assert session.frames_observed == 45
    if before_stop:
        assert session.end_reason == "interrupted"
        assert "1 of the frames recorded in this walk before the stop" in caplog.text
    else:
        assert session.end_reason == "stop"
        assert "after the capture closed (stop)" in caplog.text
        assert "were never observed" not in caplog.text


@pytest.mark.parametrize("observed_at_stop", [48, 20], ids=["caught-up", "behind"])
def test_a_drain_ends_at_the_first_frame_recorded_after_the_stop(
    tmp_path, monkeypatch, caplog, observed_at_stop
):
    """Codex H1 (the re-review's `pd`, MED-1), through the real recorder.

    48 frames recorded; the soft stop lands with the builder at
    `observed_at_stop`; the camera records 3 more frames AFTER the Stop and
    the recorder's normal close lands, all before the builder's next stop
    check. That check then reads a normal close and drains. 6124de8 drained
    the 3 post-Stop frames into the world (51); 44fbd13 stopped where it was
    (48 caught up -- byte for byte, see the identity test's
    `soft-while-open-3-after-stop-close-before-next-check` -- and 20 behind,
    stored as a finished walk). Now the drain builds every frame recorded up
    to the Stop, in order, and none after it."""
    recorder = CaptureRecorder(tmp_path / "recordings")
    capture_id = recorder.start()
    for index in range(48):
        assert recorder.write_frame(b"synthetic-frame", source_seq=index + 1)
    observed = []

    def on_observe(n, stop):
        if n != observed_at_stop:
            return
        stop.request(StopRequest.SOFT, "stdin-closed")
        _past_the_stop(stop)
        for index in range(3):
            assert recorder.write_frame(b"synthetic-frame", source_seq=49 + index)
        recorder.stop()

    _hook(monkeypatch, on_observe)
    original_observe = builder_script.WorldBuilderEngine.observe

    def recording_observe(self, *args, **kwargs):
        observed.append(kwargs.get("source_seq"))
        return original_observe(self, *args, **kwargs)

    monkeypatch.setattr(builder_script.WorldBuilderEngine, "observe", recording_observe)
    with caplog.at_level(logging.INFO, logger="tower.world_build_session"):
        exit_code, session = _run_session(
            recorder.capture_dir(capture_id), tmp_path / "worlds", max_idle_polls="50"
        )

    assert recorder.manifest(capture_id)["frames_written"] == 51
    assert exit_code == 0
    assert session.frames_observed == 48
    assert observed == list(range(1, 49))
    assert session.end_reason == "stop"
    assert "after the capture closed (stop)" in caplog.text
    assert "were never observed" not in caplog.text


def test_a_drain_ends_only_at_a_frame_stamped_after_the_stop():
    """A tie with the Stop's own clock tick is before it (the guard counts it
    the same way), a frame with no stamp is drained, and the check belongs
    to a drain: before one is latched the stop policy alone decides."""

    class ClosedNormally:
        def end_reason(self, **_kwargs):
            return "stop"

    stop = StopRequest()
    stop.request(StopRequest.SOFT, "stdin-closed")
    at = stop.soft_requested_monotonic

    def frame(received_monotonic):
        return builder_script.ObservedFrame(
            payload=b"", source_seq=1, received_monotonic=received_monotonic
        )

    assert not stop.recorded_after_the_stop(frame(at + 1.0))
    assert stop.asked_for_capture({"follower": ClosedNormally()}) is False  # drains
    assert not stop.recorded_after_the_stop(frame(at - 1.0))
    assert not stop.recorded_after_the_stop(frame(at))
    assert not stop.recorded_after_the_stop(frame(None))
    assert stop.recorded_after_the_stop(frame(at + 0.001))


@pytest.mark.parametrize("step_back_s", [0.0, 3.0], ids=["no-step", "wall-clock-3s-back"])
def test_a_backward_wall_clock_step_cannot_hide_a_frame_from_before_the_stop(
    tmp_path, monkeypatch, caplog, step_back_s
):
    """Codex H3 (the re-review's `c2`, MED-2), through the real recorder.

    Caught up at 48, then the link drops with a backlog: the recorder takes
    10 more frames and closes `disconnect`. The host's wall clock is then
    stepped BACK 3 s -- an NTP correction -- and the wearer's soft stop
    lands. The 10 were received before the Stop and never observed. On the
    wall clock their `received_at` read as after the request, the guard
    counted none, and 6124de8 stored 48 of 58 as `stop` / `complete`."""
    recorder = CaptureRecorder(tmp_path / "recordings")
    capture_id = recorder.start()
    for index in range(48):
        assert recorder.write_frame(b"synthetic-frame", source_seq=index + 1)
    real_time = time.time
    holder = {}

    def on_observe(n, stop):
        if n != 48:
            return
        for index in range(10):
            assert recorder.write_frame(b"synthetic-frame", source_seq=49 + index)
        recorder.stop(capture_module.END_REASON_DISCONNECT)
        if step_back_s:
            monkeypatch.setattr(time, "time", lambda: real_time() - step_back_s)
        stop.request(StopRequest.SOFT, "stdin-closed")
        holder["stop"] = stop

    _hook(monkeypatch, on_observe)
    with caplog.at_level(logging.WARNING, logger="tower.world_build_session"):
        exit_code, session = _run_session(
            recorder.capture_dir(capture_id), tmp_path / "worlds", max_idle_polls="50"
        )
    monkeypatch.setattr(time, "time", real_time)

    last_received_at = recorder.read_frames(capture_id)[-1]["received_at"]
    assert exit_code == 0
    assert session.frames_observed == 48
    assert session.end_reason == "interrupted"
    assert not _labelled_finished(session)
    assert "10 of the frames recorded in this walk before the stop" in caplog.text
    if step_back_s:
        # The step really put the Stop "before" the backlog on the wall clock.
        assert holder["stop"].soft_requested_at < last_received_at


def _rows_on_both_clocks(directory, first, stop):
    """Rows first..stop-1: received_at = index, received_monotonic = 100 + index."""
    with (directory / "frames.jsonl").open("a", encoding="utf-8") as out:
        out.write("".join(_row(1 + i, float(i), 100.0 + i) for i in range(first, stop)))


def test_unobserved_records_counts_only_what_was_received_by_the_cutoff(tmp_path):
    """In hand and past the read position alike; AT the cutoff counts."""
    directory = tmp_path / "small"
    directory.mkdir()
    (directory / "frame.jpg").write_bytes(b"synthetic-frame")
    _rows_on_both_clocks(directory, 0, 10)  # monotonic 100.0 .. 109.0
    _write_manifest(directory, None)
    follower = CaptureFollower(directory, poll_seconds=0)
    frames = follower.follow(max_idle_polls=1)
    for _ in range(4):
        next(frames)
    # In hand: the fourth (103.0) and the six behind it (104.0 .. 109.0).
    assert follower.unobserved_records() == 7
    assert follower.unobserved_records(received_by_monotonic=105.0) == 3
    assert follower.unobserved_records(received_by_monotonic=102.0) == 0
    _rows_on_both_clocks(directory, 10, 13)  # 110.0, 111.0, 112.0
    assert follower.unobserved_records() == 10
    assert follower.unobserved_records(received_by_monotonic=111.0) == 7 + 2
    assert follower.unobserved_records(received_by_monotonic=109.5) == 7
    # The wall stamps (0.0 .. 12.0) decide nothing.
    assert follower.unobserved_records(received_by_monotonic=10_000.0) == 10


def test_only_the_first_soft_request_sets_the_cutoff():
    stop = StopRequest()
    assert stop.soft_requested_monotonic is None and stop.soft_requested_at is None
    stop.request(StopRequest.SOFT, "first")
    first, first_at = stop.soft_requested_monotonic, stop.soft_requested_at
    assert first is not None and abs(first - time.monotonic()) < 5
    assert first_at is not None and abs(first_at - time.time()) < 5
    time.sleep(0.02)
    stop.request(StopRequest.SOFT, "again")
    stop.request(StopRequest.HARD, "SIGBREAK")
    assert stop.soft_requested_monotonic == first
    assert stop.soft_requested_at == first_at

    hard_first = StopRequest()
    hard_first.request(StopRequest.HARD, "SIGBREAK")
    hard_first.request(StopRequest.SOFT, "ignored")
    assert hard_first.soft_requested_monotonic is None
    assert hard_first.soft_requested_at is None


def test_the_monotonic_clock_is_one_clock_across_processes():
    """The cutoff compares the recorder's `time.monotonic()` (the Tower) with
    the builder's (its child process). That is only meaningful if the clock
    is system-wide, which Python does not promise and every platform the
    Tower runs on provides (GetTickCount64, CLOCK_MONOTONIC). Pinned here so
    a platform where it is not fails loudly instead of mislabelling walks."""
    before = time.monotonic()
    child = subprocess.run(
        [sys.executable, "-c", "import time; print(repr(time.monotonic()))"],
        capture_output=True, text=True, check=True, timeout=60,
    )
    after = time.monotonic()
    assert before <= float(child.stdout) <= after


# ---------------------------------------------------------------------------
# The read budget is TIME, not attempts (adversarial review MED-1, LOW-3).
# ---------------------------------------------------------------------------


def _closed_directory(tmp_path, end_reason="stop", name="unit"):
    directory = tmp_path / name
    directory.mkdir()
    _write_manifest(directory, end_reason)
    return directory


def test_strict_read_waits_out_a_short_lock_with_backoff(tmp_path, monkeypatch):
    directory = _closed_directory(tmp_path)
    fault = _unreadable_for(monkeypatch, directory / "capture.json", 0.15)
    follower = CaptureFollower(directory)
    fault["arm"]()
    started = time.monotonic()

    assert follower.end_reason(strict=True) == "stop"
    assert time.monotonic() - started >= 0.15
    # Backed off, not spinning: 5 ms doubling to 50 ms is ~7 reads in
    # 150 ms. A reader with no sleep makes thousands against the same lock.
    assert 2 <= fault["failures"] <= 20, fault["failures"]


def test_strict_read_gives_up_after_its_time_budget(tmp_path, monkeypatch):
    directory = _closed_directory(tmp_path)
    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", 0.3)
    fault = _unreadable_for(monkeypatch, directory / "capture.json", 60.0)
    follower = CaptureFollower(directory)
    fault["arm"]()
    started = time.monotonic()

    with pytest.raises(OSError):
        follower.end_reason(strict=True)
    elapsed = time.monotonic() - started
    assert 0.3 <= elapsed < 0.8, elapsed
    assert fault["failures"] <= 30, fault["failures"]


def test_default_read_never_waits(tmp_path, monkeypatch):
    directory = _closed_directory(tmp_path)
    fault = _unreadable_for(monkeypatch, directory / "capture.json", 60.0)
    follower = CaptureFollower(directory)
    fault["arm"]()
    started = time.monotonic()

    assert follower.end_reason() is None
    assert time.monotonic() - started < 0.05
    assert fault["failures"] == 1


def test_a_missing_manifest_reads_as_open_without_waiting(tmp_path):
    """Standard LOW-2 / adversarial LOW-1: no `capture.json` is not a
    capture caught mid-replace (the recorder writes one first and replaces
    it atomically), so it is "still open", as before 8ede341 -- no wait, no
    raise."""
    directory = tmp_path / "bare"
    directory.mkdir()
    started = time.monotonic()

    assert CaptureFollower(directory).end_reason(strict=True) is None
    assert time.monotonic() - started < 0.05


def test_soft_stop_on_a_directory_without_a_manifest_is_interrupted_not_error(
    recorded_capture, tmp_path, monkeypatch
):
    assert not (recorded_capture / "capture.json").exists()
    _hook(
        monkeypatch,
        lambda n, stop: stop.request(StopRequest.SOFT, "test") if n == 805 else None,
    )
    exit_code, session = _run_session(
        recorded_capture, tmp_path / "worlds", max_idle_polls="5"
    )

    assert exit_code == 0
    assert session.frames_observed == 805
    assert session.end_reason == "interrupted"


def test_a_hard_stop_during_the_strict_wait_is_honoured_at_once(
    recorded_capture, tmp_path, monkeypatch
):
    """Re-review LOW-3 (its Q6): a soft stop on a normal close meets a
    manifest unreadable for 10 s, and a hard stop lands 0.2 s into the wait.
    It used to be held for the whole 2 s budget and the session then ended
    `error`, exit 1. Now the wait ends with the hard stop."""
    manifest = recorded_capture / "capture.json"
    _write_manifest(recorded_capture, None)
    fault = _unreadable_for(monkeypatch, manifest, 10.0)
    marks = {}

    def on_observe(n, stop):
        if n != 805:
            return
        _write_manifest(recorded_capture, "stop")
        fault["arm"]()
        stop.request(StopRequest.SOFT, "test")

        def hard():
            time.sleep(0.2)
            marks["hard"] = time.monotonic()
            stop.request(StopRequest.HARD, "SIGBREAK")

        threading.Thread(target=hard, daemon=True).start()

    _hook(monkeypatch, on_observe)
    exit_code, session = _run_session(recorded_capture, tmp_path / "worlds")
    done = time.monotonic()

    assert fault["failures"] >= 2
    assert done - marks["hard"] < capture_module.MANIFEST_READ_BUDGET_S / 2
    assert exit_code == 0
    assert session.frames_observed == 805
    assert session.end_reason == "interrupted"
    assert not _labelled_finished(session)


def test_an_unmeasurable_journal_is_retried_for_the_budget_unless_cancelled(
    tmp_path, monkeypatch
):
    """The re-measurement waits out the shipped budget, and a hard stop
    (`cancel`) ends it at once -- the guard runs during a Tower shutdown
    too, and an unmeasurable journal reads `interrupted` either way."""
    follower = CaptureFollower(_small_closed_capture(tmp_path), poll_seconds=0)
    monkeypatch.setattr(
        capture_module._JournalTail, "records_not_yet_read", lambda self, **_: None
    )
    started = time.monotonic()
    assert follower.unobserved_records(cancel=lambda: True) is None
    assert time.monotonic() - started < capture_module.MANIFEST_READ_BUDGET_S / 4

    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", 0.2)
    started = time.monotonic()
    assert follower.unobserved_records(cancel=lambda: False) is None
    assert time.monotonic() - started >= 0.2


def test_a_hard_stop_does_not_wait_out_an_unmeasurable_journal(
    recorded_capture, tmp_path, monkeypatch
):
    """The builder passes its hard stop to the guard's re-measurement: a
    Tower shutting down measures once and moves on, labelled honestly."""
    _write_manifest(recorded_capture, "stop")
    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", SHORT_BUDGET_S)
    calls = {"n": 0}

    def unmeasurable(self, **_kwargs):
        calls["n"] += 1
        return None

    monkeypatch.setattr(capture_module._JournalTail, "records_not_yet_read", unmeasurable)
    _hook(monkeypatch, install_level=StopRequest.HARD)
    exit_code, session = _run_session(recorded_capture, tmp_path / "worlds")

    assert calls["n"] == 1
    assert exit_code == 0
    assert session.end_reason == "interrupted"


def test_one_failed_journal_stat_after_the_loop_is_measured_again(tmp_path, monkeypatch):
    """Re-review LOW-1 (its Q7): caught up, 48 of 48, and the guard's one
    stat of the journal fails once. An unmeasurable journal is measured
    again within the budget, so a transient fault does not relabel a
    finished walk `interrupted`."""
    capture = tmp_path / "captures" / ("h" * 32)
    capture.mkdir(parents=True)
    (capture / "frame.jpg").write_bytes(b"synthetic-frame")
    journal = capture / "frames.jsonl"
    journal.write_text(_journal_rows(CAUGHT_UP_FRAMES), encoding="utf-8")
    _write_manifest(capture, None)
    armed = {"on": False, "hits": 0}
    path_type = type(journal)
    original_stat = path_type.stat

    def stat(self, *args, **kwargs):
        if armed["on"] and self == journal:
            armed["on"] = False
            armed["hits"] += 1
            raise PermissionError("transient")
        return original_stat(self, *args, **kwargs)

    def on_observe(n, stop):
        if n == CAUGHT_UP_FRAMES:
            _write_manifest(capture, "stop")
            stop.request(StopRequest.SOFT, "test")

    _hook(monkeypatch, on_observe)
    original_bounded = builder_script.StopRequest.bounded

    def bounded_then_arm(self, frames, **kwargs):
        yield from original_bounded(self, frames, **kwargs)
        armed["on"] = True

    monkeypatch.setattr(builder_script.StopRequest, "bounded", bounded_then_arm)
    monkeypatch.setattr(path_type, "stat", stat)
    exit_code, session = _run_session(capture, tmp_path / "worlds", max_idle_polls="50")
    monkeypatch.setattr(path_type, "stat", original_stat)

    assert armed["hits"] == 1
    assert exit_code == 0
    assert session.frames_observed == CAUGHT_UP_FRAMES
    assert session.end_reason == "stop"


# ---------------------------------------------------------------------------
# A REAL lock, not a monkeypatch: what an AV scanner or the indexer does to
# a freshly replaced file. The adversarial review's P6 held one for 150 ms
# and 8ede341 ended in `error` with 2,392 recorded frames never built.
# ---------------------------------------------------------------------------

GENERIC_READ = 0x80000000
OPEN_EXISTING = 3


def _hold(path, seconds, kind):
    """Hold `path` for real for `seconds`; returns the releasing timer."""
    if kind == "exclusive-open":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        ]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        # Share mode 0: nobody else may open it at all.
        handle = kernel32.CreateFileW(
            str(path), GENERIC_READ, 0, None, OPEN_EXISTING, 0, None
        )
        if handle in (None, wintypes.HANDLE(-1).value):
            raise OSError(ctypes.get_last_error(), "could not take the lock")

        def release():
            kernel32.CloseHandle(handle)
    else:
        import msvcrt

        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        length = max(1, os.fstat(fd).st_size)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, length)

        def release():
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, length)
            os.close(fd)

    timer = threading.Timer(seconds, release)
    timer.start()
    return timer


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing and lock semantics")
@pytest.mark.parametrize("kind", ["exclusive-open", "byte-range-lock"])
def test_a_real_300ms_lock_at_soft_stop_still_drains(
    recorded_capture, tmp_path, monkeypatch, kind
):
    manifest = recorded_capture / "capture.json"
    _write_manifest(recorded_capture, None)
    original_read = capture_module.read_json_closed
    attempts = {"during_hold": 0, "errors": []}
    held = {"timer": None}

    def counting_read(path):
        timer = held["timer"]
        if path == manifest and timer is not None and timer.is_alive():
            attempts["during_hold"] += 1
        try:
            return original_read(path)
        except OSError as exc:
            attempts["errors"].append(type(exc).__name__)
            raise

    monkeypatch.setattr(capture_module, "read_json_closed", counting_read)

    def on_observe(n, stop):
        if n == 805:
            _write_manifest(recorded_capture, "stop")
            held["timer"] = _hold(manifest, 0.3, kind)
            with pytest.raises(PermissionError):
                original_read(manifest)
            stop.request(StopRequest.SOFT, "test")

    _hook(monkeypatch, on_observe)
    try:
        exit_code, session = _run_session(recorded_capture, tmp_path / "worlds")
    finally:
        held["timer"].join()

    assert "PermissionError" in attempts["errors"]
    assert attempts["during_hold"] <= 20, attempts
    assert exit_code == 0
    assert session.frames_observed == TOTAL
    assert session.end_reason == "stop"
    assert session.finalization["state"] == "complete"


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_a_real_lock_longer_than_the_budget_ends_honestly(
    recorded_capture, tmp_path, monkeypatch
):
    """Held past the shipped `MANIFEST_READ_BUDGET_S`: fail closed."""
    manifest = recorded_capture / "capture.json"
    _write_manifest(recorded_capture, None)
    held = {"timer": None}

    def on_observe(n, stop):
        if n == 805:
            _write_manifest(recorded_capture, "stop")
            held["timer"] = _hold(
                manifest, capture_module.MANIFEST_READ_BUDGET_S + 0.5, "exclusive-open"
            )
            stop.request(StopRequest.SOFT, "test")

    _hook(monkeypatch, on_observe)
    started = time.monotonic()
    try:
        exit_code, session = _run_session(recorded_capture, tmp_path / "worlds")
    finally:
        held["timer"].join()

    assert time.monotonic() - started >= capture_module.MANIFEST_READ_BUDGET_S
    assert exit_code == 1
    assert session.frames_observed == 805
    assert session.end_reason == "error"
    assert session.finalization["state"] == "interrupted"
    assert not _labelled_finished(session)


# ---------------------------------------------------------------------------
# A FRAME THAT WAS RECORDED AND COULD NOT BE READ IS NEVER SKIPPED INTO A
# FINISHED WALK (Codex H2, the re-review's HIGH-1).
#
# `_load` answered None on any OSError reading a JPEG, the read position moved
# on, and nothing counted the frame: 47 of 48 stored `stop` / `complete` with
# one injected PermissionError, 3,196 of 3,197 with a real Windows lock on one
# JPEG during the drain. A complete journal line that did not parse was
# skipped the same way. Now an image that cannot be opened is waited for,
# within the manifest's budget, and built if it comes back; if it does not --
# or the line cannot be parsed -- the frame is counted as recorded and not
# observed, and the walk is `interrupted`.
# ---------------------------------------------------------------------------


def _distinct_frames(directory, count, end_reason="stop"):
    """`count` frames, each in its own JPEG file, journalled; closed."""
    directory.mkdir(parents=True)
    rows = []
    for index in range(count):
        name = f"frame{index:04d}.jpg"
        (directory / name).write_bytes(b"synthetic-frame")
        rows.append(
            json.dumps({"source_seq": index + 1, "received_at": float(index), "relpath": name})
            + "\n"
        )
    (directory / "frames.jsonl").write_text("".join(rows), encoding="utf-8")
    _write_manifest(directory, end_reason)
    return directory


def _image_unreadable_for(monkeypatch, names, seconds):
    """The JPEGs called `names` raise PermissionError for `seconds` of time
    from the first attempt to read each: a lock held for a DURATION."""
    original = Path.read_bytes
    fault = {"until": {}, "failures": 0}

    def read_bytes(self):
        if self.name in names:
            now = time.monotonic()
            until = fault["until"].setdefault(self.name, now + seconds)
            if now < until:
                fault["failures"] += 1
                raise PermissionError(13, "held by another process", str(self))
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return fault


def test_an_image_locked_for_a_moment_is_waited_for_and_built(tmp_path, monkeypatch):
    directory = _distinct_frames(tmp_path / "cap", 10)
    fault = _image_unreadable_for(monkeypatch, {"frame0004.jpg"}, 0.15)
    follower = CaptureFollower(directory, poll_seconds=0)

    seqs = [frame.source_seq for frame in follower.follow(max_idle_polls=1)]

    assert seqs == list(range(1, 11))
    assert fault["failures"] >= 2
    assert follower.unobserved_records() == 0


def test_an_image_still_locked_after_the_budget_is_counted_not_skipped(
    tmp_path, monkeypatch
):
    """And the budget is spent ONCE per follower: a second locked image is
    counted at once rather than waited on for another whole budget."""
    directory = _distinct_frames(tmp_path / "cap", 10)
    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", 0.3)
    _image_unreadable_for(monkeypatch, {"frame0003.jpg", "frame0007.jpg"}, 60.0)
    follower = CaptureFollower(directory, poll_seconds=0)
    started = time.monotonic()

    seqs = [frame.source_seq for frame in follower.follow(max_idle_polls=1)]

    elapsed = time.monotonic() - started
    assert seqs == [1, 2, 3, 5, 6, 7, 9, 10]
    assert 0.3 <= elapsed < 0.55, elapsed
    assert follower.unobserved_records() == 2
    # Cut off before both were received (received_at 3.0 and 7.0; these rows
    # carry no monotonic stamp, so nothing can show they came after): counted.
    assert follower.unobserved_records(received_by_monotonic=0.0) == 2


def test_an_unreadable_image_that_lands_with_the_close_is_counted(tmp_path, monkeypatch):
    """The follower's one journal read AFTER it sees the close has its own
    loop; a frame it cannot read there is counted too."""
    directory = _distinct_frames(tmp_path / "cap", 10, end_reason=None)
    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", 0.2)
    follower = CaptureFollower(directory, poll_seconds=0)
    frames = follower.follow(max_idle_polls=5)
    for _ in range(10):
        next(frames)
    with (directory / "frames.jsonl").open("a", encoding="utf-8") as out:
        for index in range(10, 13):
            name = f"frame{index:04d}.jpg"
            (directory / name).write_bytes(b"synthetic-frame")
            out.write(
                json.dumps({"source_seq": index + 1, "received_at": float(index), "relpath": name})
                + "\n"
            )
    _write_manifest(directory, "stop")
    _image_unreadable_for(monkeypatch, {"frame0011.jpg"}, 60.0)

    assert [frame.source_seq for frame in frames] == [11, 13]
    assert follower.unobserved_records() == 1


def test_a_missing_image_and_a_record_without_one_are_counted_without_waiting(tmp_path):
    directory = _distinct_frames(tmp_path / "cap", 10)
    (directory / "frame0004.jpg").unlink()
    with (directory / "frames.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps({"source_seq": 11, "received_at": 10.0}) + "\n")
    follower = CaptureFollower(directory, poll_seconds=0)
    started = time.monotonic()

    seqs = [frame.source_seq for frame in follower.follow(max_idle_polls=1)]

    assert time.monotonic() - started < capture_module.MANIFEST_READ_BUDGET_S / 4
    assert seqs == [1, 2, 3, 4, 6, 7, 8, 9, 10]
    assert follower.unobserved_records() == 2


@pytest.mark.parametrize(
    ("locked_s", "expected_observed", "expected_end"),
    [
        pytest.param(0.1, 48, "stop", id="locked-within-the-budget-built"),
        pytest.param(60.0, 47, "interrupted", id="locked-past-the-budget-counted"),
    ],
)
def test_one_unreadable_image_is_never_a_finished_walk(
    tmp_path, monkeypatch, caplog, locked_s, expected_observed, expected_end
):
    """The re-review's h2a, end to end: a capture closed normally before the
    builder started, no stop at all; the 20th image cannot be read for
    `locked_s`. 6124de8 stored 47 of 48 as `stop` / `complete`."""
    capture = _distinct_frames(tmp_path / "captures" / ("a" * 32), 48)
    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", SHORT_BUDGET_S)
    _image_unreadable_for(monkeypatch, {"frame0019.jpg"}, locked_s)
    _hook(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="tower"):
        exit_code, session = _run_session(capture, tmp_path / "worlds", max_idle_polls="5")

    assert exit_code == 0
    assert session.frames_observed == expected_observed
    assert session.end_reason == expected_end
    if expected_end == "interrupted":
        assert "frame0019.jpg" in caplog.text
        assert "1 of the frames recorded in this walk were never observed" in caplog.text
    else:
        assert "were never observed" not in caplog.text


def test_a_journal_line_that_cannot_be_parsed_is_never_a_finished_walk(
    tmp_path, monkeypatch, caplog
):
    """The same door through the journal: a complete line the follower read
    and could not parse. Skipped, the read position moved past it, and
    6124de8 stored 47 of 48 recorded frames as `stop` / `complete`."""
    capture = _distinct_frames(tmp_path / "captures" / ("a" * 32), 48)
    journal = capture / "frames.jsonl"
    lines = journal.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[19] = lines[19][:25] + "\n"
    journal.write_text("".join(lines), encoding="utf-8")
    _hook(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="tower"):
        exit_code, session = _run_session(capture, tmp_path / "worlds", max_idle_polls="5")

    assert exit_code == 0
    assert session.frames_observed == 47
    assert session.end_reason == "interrupted"
    assert "skipping an unreadable journal line" in caplog.text
    assert "1 of the frames recorded in this walk were never observed" in caplog.text


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
@pytest.mark.parametrize(
    ("held_s", "expected_observed", "expected_end"),
    [
        pytest.param(0.3, 400, "stop", id="300ms-lock-waited-out-and-built"),
        pytest.param(None, 399, "interrupted", id="lock-past-the-budget-counted"),
    ],
)
def test_a_real_lock_on_one_image_during_the_drain(
    tmp_path, monkeypatch, held_s, expected_observed, expected_end
):
    """The re-review's h2b at test scale, with a REAL Windows lock and no
    product code patched: a late builder behind, the normal close, the soft
    stop, and the drain meets one JPEG held with share mode 0 -- the AV
    scanner or indexer shape. 6124de8: 3,196 of 3,197 `stop` / `complete`."""
    capture = _distinct_frames(tmp_path / "captures" / ("a" * 32), 400, end_reason=None)
    # The very next frame the drain reads, so the wait is always exercised.
    locked = capture / "frame0100.jpg"
    held = {"timer": None}

    def on_observe(n, stop):
        if n == 100:
            _write_manifest(capture, "stop")
            seconds = held_s or capture_module.MANIFEST_READ_BUDGET_S + 1.5
            held["timer"] = _hold(locked, seconds, "exclusive-open")
            with pytest.raises(PermissionError):
                locked.open("rb")
            stop.request(StopRequest.SOFT, "stdin-closed")

    _hook(monkeypatch, on_observe)
    try:
        exit_code, session = _run_session(capture, tmp_path / "worlds")
    finally:
        if held["timer"] is not None:
            held["timer"].join()

    assert exit_code == 0
    assert session.frames_observed == expected_observed
    assert session.end_reason == expected_end
    if expected_end != "stop":
        assert not _labelled_finished(session)


def test_a_hard_stop_while_an_image_is_waited_for_is_honoured_at_once(
    tmp_path, monkeypatch
):
    """The wait for a locked image asks the stop check between tries, as the
    strict manifest read does: a Tower shutting down is not held for the
    budget by one frame."""
    capture = _distinct_frames(tmp_path / "captures" / ("a" * 32), 48, end_reason=None)
    _image_unreadable_for(monkeypatch, {"frame0029.jpg"}, 60.0)
    marks = {}

    def on_observe(n, stop):
        if n != 20:
            return
        _write_manifest(capture, "stop")
        stop.request(StopRequest.SOFT, "stdin-closed")

        def hard():
            time.sleep(0.2)
            marks["hard"] = time.monotonic()
            stop.request(StopRequest.HARD, "SIGBREAK")

        threading.Thread(target=hard, daemon=True).start()

    _hook(monkeypatch, on_observe)
    exit_code, session = _run_session(capture, tmp_path / "worlds")
    done = time.monotonic()

    assert done - marks["hard"] < capture_module.MANIFEST_READ_BUDGET_S / 2
    assert exit_code == 0
    assert session.frames_observed == 29
    assert session.end_reason == "interrupted"


# ---------------------------------------------------------------------------
# The surviving mutants of both reviews of 8ede341, killed: standard M06 /
# adversarial M3 (a `bounded_limit` or `disconnect` close remembered as
# `stop`), adversarial M17 (a remembered `stop` overriding a readable
# manifest) and standard M10 (the drain re-reading the manifest per frame).
# Standard/adversarial M15 is killed by the parametrized pre-first-frame
# test above, and adversarial M2 by the backoff bound in
# `test_strict_read_waits_out_a_short_lock_with_backoff`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["bounded_limit", "disconnect"])
def test_only_a_normal_close_answers_a_later_unreadable_read(
    tmp_path, monkeypatch, reason
):
    """Standard review M06 / adversarial M3: a `bounded_limit` or
    `disconnect` close read once must not come back as `stop` when the
    manifest later cannot be read. That turns an honest `interrupted`
    (bounded) into an ordinary finish."""
    directory = _closed_directory(tmp_path, reason)
    follower = CaptureFollower(directory)
    assert follower.end_reason() == reason

    monkeypatch.setattr(capture_module, "MANIFEST_READ_BUDGET_S", 0.05)
    fault = _unreadable_for(monkeypatch, directory / "capture.json", 60.0)
    fault["arm"]()
    assert follower.end_reason() is None
    with pytest.raises(OSError):
        follower.end_reason(strict=True)


def test_a_remembered_normal_close_never_overrides_a_readable_manifest(tmp_path):
    """Adversarial M17: the remembered `stop` answers an UNREADABLE read
    only. A readable manifest is always what the follower reports."""
    directory = _closed_directory(tmp_path, "stop")
    follower = CaptureFollower(directory)
    assert follower.end_reason() == "stop"

    _write_manifest(directory, "disconnect")
    assert follower.end_reason() == "disconnect"
    assert follower.end_reason(strict=True) == "disconnect"


def test_a_bounded_close_read_once_then_unreadable_is_still_interrupted(
    recorded_capture, tmp_path, monkeypatch
):
    """Standard M06 / adversarial M3, end to end: the bound is read once at
    the close, every later read fails, and a soft stop came after the
    builder caught up. The session must stay `interrupted`."""
    manifest = recorded_capture / "capture.json"
    _write_manifest(recorded_capture, None)
    original_read = capture_module.read_json_closed
    state = {"closed_read": False}

    def read_once_closed(path):
        if path == manifest and state["closed_read"]:
            raise OSError("manifest unavailable after the bound was read")
        payload = original_read(path)
        if path == manifest and payload.get("ended_at") is not None:
            state["closed_read"] = True
        return payload

    monkeypatch.setattr(capture_module, "read_json_closed", read_once_closed)

    def on_observe(n, stop):
        if n == TOTAL:
            _write_manifest(recorded_capture, "bounded_limit")
            stop.request(StopRequest.SOFT, "test")

    _hook(monkeypatch, on_observe)
    exit_code, session = _run_session(recorded_capture, tmp_path / "worlds")

    assert state["closed_read"]
    assert exit_code == 0
    assert session.frames_observed == TOTAL
    assert session.end_reason == "interrupted"


def test_a_drain_does_not_reread_the_manifest_per_frame(
    recorded_capture, tmp_path, monkeypatch
):
    """Standard M10: once a normal close is verified, the drain stops asking.

    Without the `StopRequest` latch every remaining frame re-reads
    `capture.json` (the suite measured 76 s instead of 22 s). The answer
    cannot change -- a closed capture stays closed -- so the reads are pure
    cost on the path that is already the slow one.
    """
    manifest = recorded_capture / "capture.json"
    _write_manifest(recorded_capture, None)
    original_read = capture_module.read_json_closed
    reads = {"armed": False, "count": 0}

    def counting_read(path):
        if reads["armed"] and path == manifest:
            reads["count"] += 1
        return original_read(path)

    monkeypatch.setattr(capture_module, "read_json_closed", counting_read)

    def on_observe(n, stop):
        if n == 805:
            _write_manifest(recorded_capture, "stop")
            reads["armed"] = True
            stop.request(StopRequest.SOFT, "test")

    _hook(monkeypatch, on_observe)
    exit_code, session = _run_session(recorded_capture, tmp_path / "worlds")

    assert exit_code == 0
    assert session.frames_observed == TOTAL
    assert session.end_reason == "stop"
    # 2,392 frames drained; the manifest is read a handful of times.
    assert reads["count"] <= 10, reads["count"]
