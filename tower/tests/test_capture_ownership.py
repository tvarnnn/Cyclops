"""A dead connection must not disarm a live connection's recording.

The race is not hypothetical and the timing is not close. uvicorn does not
learn a WebSocket is dead for 20-40 seconds (`ws_ping_interval` and
`ws_ping_timeout` are both 20 s), while iOS reconnects in 0.5 s and
re-sends `stream_start` -- `docs/contracts/CARTRIDGE-RESULTS.md` §19.1
records that backoff schedule, and §19.2 that a repeat `stream_start` on a
fresh connection is the EXPECTED case on this link, not an edge case.

So on any WiFi hiccup the ordering is:

    new connection arms a recording
    ...then the old connection's `finally` block finally runs
    ...and stops it

The recorder is one process-global object shared by every connection, so
before this guard it could not tell the two apart. Measured consequence:
the phone streams on, the Tower answers every frame_result, `/health`
reports `recording: false`, and zero frames reach disk for the rest of the
walk. Silently -- which is worse than losing the capture, because it looks
like success.
"""

import base64

import numpy as np
import pytest

from tower.capture import (
    END_REASON_DISCONNECT,
    END_REASON_STOP,
    CaptureRecorder,
)


def _jpeg() -> str:
    import cv2

    ok, buffer = cv2.imencode(".jpg", np.full((360, 640, 3), 128, dtype=np.uint8))
    assert ok
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _frame(seq: int, data: str) -> dict:
    return {
        "type": "frame",
        "seq": seq,
        "width": 640,
        "height": 360,
        "format": "jpeg",
        "data": data,
    }


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from tower.main import create_app

    monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(tmp_path / "capture"))
    return TestClient(create_app())


def test_a_superseded_connection_cannot_stop_the_live_recording(client):
    """The exact ordering a WiFi hiccup produces."""
    data = _jpeg()
    recorder = client.app.state.frame_observers[0]

    # Connection 1 arms a recording, then its socket dies WITHOUT the
    # server noticing yet -- so its teardown has not run.
    first = client.websocket_connect("/ws")
    ws1 = first.__enter__()
    ws1.send_json({"type": "stream_start"})
    ws1.send_json(_frame(1, data))
    ws1.receive_json()
    capture_a = recorder.status.capture_id

    # Connection 2 arrives and arms its own recording.
    with client.websocket_connect("/ws") as ws2:
        ws2.send_json({"type": "stream_start"})
        ws2.send_json(_frame(3, data))
        ws2.receive_json()
        capture_b = recorder.status.capture_id
        assert capture_b != capture_a

        # NOW connection 1's teardown runs, late, as it does in production.
        first.__exit__(None, None, None)

        assert recorder.is_recording, (
            "a dead connection disarmed the live recording"
        )
        assert recorder.status.capture_id == capture_b

        # And the live connection keeps recording.
        ws2.send_json(_frame(5, data))
        ws2.receive_json()

    frames = recorder.read_frames(capture_b)
    assert [row["wire_seq"] for row in frames] == [3, 5], (
        "frames sent after the zombie teardown were not recorded"
    )


def test_a_connection_still_stops_its_own_recording(client):
    """The privacy invariant is unchanged.

    Recording must stop on ANY exit of the owning connection, polite or
    not -- otherwise the next connection's frames land in the previous
    connection's capture with no stream_start and no consent.
    """
    data = _jpeg()
    recorder = client.app.state.frame_observers[0]

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        ws.send_json(_frame(1, data))
        ws.receive_json()
        assert recorder.is_recording

    assert not recorder.is_recording
    assert recorder.status.end_reason == END_REASON_DISCONNECT


def test_an_unowned_stop_is_unconditional(tmp_path):
    """An operator or a test stopping the recorder must always win."""
    recorder = CaptureRecorder(tmp_path)
    token = object()
    recorder.start(owner=token)
    assert recorder.is_recording

    status = recorder.stop(END_REASON_STOP)
    assert status is not None and not status.is_open


def test_a_foreign_owner_cannot_stop_a_recording(tmp_path):
    recorder = CaptureRecorder(tmp_path)
    mine, theirs = object(), object()
    recorder.start(owner=mine)

    recorder.stop(END_REASON_DISCONNECT, owner=theirs)
    assert recorder.is_recording, "a foreign owner stopped the recording"

    recorder.stop(END_REASON_DISCONNECT, owner=mine)
    assert not recorder.is_recording


def test_ownership_is_released_when_the_recording_closes(tmp_path):
    """A closed recorder must not stay bound to a connection that is gone."""
    recorder = CaptureRecorder(tmp_path)
    token = object()
    recorder.start(owner=token)
    recorder.stop(END_REASON_STOP, owner=token)

    assert recorder.owner is None
    # A different connection can now arm it.
    other = object()
    recorder.start(owner=other)
    assert recorder.owner is other


def test_the_connection_tracker_survives_a_late_teardown(client):
    """`/health` must not report "no client" while one is streaming.

    Same race as the recorder: a superseded connection's teardown runs
    after the new connection is already live, and a boolean flag cleared
    by that teardown lies about the current state.
    """
    tracker = client.app.state.session

    first = client.websocket_connect("/ws")
    first.__enter__()
    assert tracker.is_client_connected()

    with client.websocket_connect("/ws") as ws2:
        ws2.send_json({"type": "ping"})
        ws2.receive_json()

        first.__exit__(None, None, None)

        assert tracker.is_client_connected(), (
            "a late teardown reported no client while one was connected"
        )
        assert tracker.live_connections == 1

    assert not tracker.is_client_connected()
    assert tracker.live_connections == 0


def test_a_repeated_teardown_cannot_make_the_tracker_owe_a_connection(client):
    tracker = client.app.state.session
    tracker.client_connected()
    tracker.client_disconnected()
    tracker.client_disconnected()
    assert tracker.live_connections == 0
    assert not tracker.is_client_connected()
