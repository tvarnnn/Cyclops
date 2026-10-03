"""A late World Builder follower drains a normally closed capture on soft stop."""

import json

import pytest

from scripts.world_build_session import StopRequest, first_observed_frame, follow_capture


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
