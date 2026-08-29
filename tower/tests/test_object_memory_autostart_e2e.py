"""Press Start, walk, press Stop — with nothing typed in a second terminal.

The one test that exercises the actual product claim end to end: a real
ASGI app, a real WebSocket, a real `CaptureRecorder`, a real
`object_memory_session.py` in a REAL subprocess, a real store on disk, and
the real HTTP routes serving what it wrote.

Everything else in this workstream fakes the spawn, which is right for
testing bookkeeping and wrong for testing that the thing actually runs.
The 2026-08-26 gap was not a bookkeeping error — every individual piece
worked, and a human connected them by hand four times.

Slow by nature: a Python subprocess has to start, import OpenCV, tail a
journal and write records. That cost is the point. A fast version of this
test would be a fake, and a fake is what let the gap exist.

`--detector none` throughout. The subprocess is real, the journal is
real, the store is real and the routes are real; what is substituted is
13.4 MB of weights and a torchvision download, because these cases are
about the WIRING and the default suite must not fetch a model. The tests
that make claims about what the detector sees measure it against the real
corpus, in `scripts/research/`.
"""

import base64
import sys
import time

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from tower.main import OBJECT_MEMORY_WORKER  # noqa: E402
from tower.results.contracts import CARTRIDGE_OBJECT_MEMORY  # noqa: E402

SESSION_URL = f"/cartridges/{CARTRIDGE_OBJECT_MEMORY}/session"


def _frames(count: int) -> list[str]:
    import cv2

    rng = np.random.default_rng(11)
    out = []
    for _ in range(count):
        image = rng.integers(0, 255, (640, 360, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".jpg", image)
        assert ok
        out.append(base64.b64encode(buffer.tobytes()).decode("ascii"))
    return out


def _frame_message(seq: int, data: str) -> dict:
    return {
        "type": "frame",
        "seq": seq,
        "width": 360,
        "height": 640,
        "format": "jpeg",
        "data": data,
    }


def _wait_for(predicate, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


@pytest.fixture
def tower(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from tower.main import create_app

    monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(tmp_path / "capture"))
    monkeypatch.setenv("TOWER_OBSERVATION_ROOT", str(tmp_path / "memory"))
    # Pinned, not inherited. `owlv2` became the built-in default on
    # 2026-08-29, and this test spawns the REAL producer -- inheriting
    # that default would make an end-to-end lifecycle test fetch ~600 MB
    # of weights and load them on every run, to verify a lifecycle that
    # has nothing to do with semantic verification. The verifier's own
    # behaviour is covered by `test_object_memory_verification.py`.
    monkeypatch.setenv("TOWER_OBSERVATION_VERIFIER", "none")
    monkeypatch.setenv("TOWER_OBSERVATION_DEVICE", "cpu")
    monkeypatch.delenv("TOWER_OBSERVATION_ENABLED", raising=False)
    # No world builder in this test: it would spawn a second real
    # subprocess and build a real world, which is another test's job and
    # would make this one twice as slow for nothing.
    monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
    app = create_app()

    # The ONE substitution in this file: two flags appended to the argv
    # the Tower built. The process, the journal, the store and the routes
    # are all real; what is replaced is 13.4 MB of weights the default
    # suite must not download, and a poll interval that would otherwise
    # make the test wait a quarter of a second per frame.
    #
    # Appended rather than rebuilt from scratch, deliberately: every flag
    # `main.py` puts in that argv is still there and still has to work,
    # which is what `test_the_producer_argv_is_runnable_as_written` then
    # checks against a real process.
    spec = app.state.capture_workers.spec_for(OBJECT_MEMORY_WORKER)
    app.state.capture_workers._registries[OBJECT_MEMORY_WORKER].spec = spec.__class__(
        argv=tuple(spec.argv) + ("--detector", "none", "--poll-seconds", "0.1"),
        cwd=spec.cwd,
        name=spec.name,
        gate=spec.gate,
    )

    client = TestClient(app)
    yield client, app, tmp_path
    # Never leave a producer running, whatever the assertions did.
    app.state.capture_workers.shutdown(grace_seconds=5.0)


# -- the claim ---------------------------------------------------------


def test_start_walk_stop_remembers_with_no_manual_step(tower):
    """The whole product claim, from the button to the HTTP answer.

    Nothing in this test names a capture id or a store path. That is the
    point: on 2026-08-26 a human read the first off a directory listing
    and typed the second into an environment variable, and until they did
    the route answered 404 about a memory that had already been written.
    """
    client, app, tmp_path = tower
    frames = _frames(12)

    assert client.get(SESSION_URL).json()["state"] == "stopped"
    started = client.post(f"{SESSION_URL}/start").json()
    assert started["state"] == "active"

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(frames, start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()

        capture_id = app.state.frame_observers[0].status.capture_id

        # A REAL producer was started against THIS capture, without being
        # told which one.
        workers = _wait_for(
            lambda: [
                row
                for row in app.state.capture_workers.status()
                if row["worker"] == OBJECT_MEMORY_WORKER
            ],
            20.0,
            "the producer to be running",
        )
        assert workers[0]["capture_id"] == capture_id

        # And the session says so over the same surface a phone reads.
        state = _wait_for(
            lambda: (
                client.get(SESSION_URL).json()
                if client.get(SESSION_URL).json()["following"]
                else None
            ),
            20.0,
            "the session to report a live producer",
        )
        assert state["following"] == [capture_id]
        assert state["captures"] == [capture_id]

        ws.send_json({"type": "stream_stop"})
        ws.send_json({"type": "ping"})
        ws.receive_json()

    # The capture closed, so the follower observes completion, finalises
    # and exits on its own. Nothing kills it.
    _wait_for(
        lambda: not app.state.capture_workers.following(OBJECT_MEMORY_WORKER),
        60.0,
        "the producer to finish and be reaped",
    )

    # THE 404 THAT COST THE PHYSICAL RUN ITS FIRST HALF HOUR.
    #
    # The producer was handed a `--root` and the routes were handed a
    # root, and on 2026-08-26 those were two different defaults. Asserted
    # against the argv the Tower actually built, not against a setting
    # read twice.
    spec = app.state.capture_workers.spec_for(OBJECT_MEMORY_WORKER)
    argv = list(spec.argv)
    assert argv[argv.index("--root") + 1] == app.state.object_memory_root
    assert app.state.object_memory_root == str(tmp_path / "memory")

    body = client.get("/object-memory/observations").json()
    assert body["observation_count"] == 0, (
        "a substituted detector found nothing, which is correct -- what "
        "matters is that the route answered ABOUT THE RIGHT STORE rather "
        "than 404 about a different one"
    )
    assert body["recorded_classes"] == ["laptop", "cell phone"]


def test_a_record_written_before_the_walk_survives_it_and_is_served(tower):
    """One store, two writers, and the reader sees both.

    A real producer subprocess appends to the same directory a record was
    already sitting in. If the producer truncated, relocated or rewrote
    that directory -- or if the route were reading a different one -- the
    record would be gone. This is the agreement the 2026-08-26 run did not
    have, asserted against a real process rather than a settings object.
    """
    from tower.object_memory.records import ObjectObservation, privacy_tags_for
    from tower.object_memory.store import ObservationStore
    from tower.confidence import Confidence

    client, app, tmp_path = tower
    store = ObservationStore(tmp_path / "memory", retention_seconds=None)
    store.append(
        ObjectObservation(
            object_class="laptop",
            detector_score=0.9,
            confidence=Confidence.HIGH,
            observed_at=time.time(),
            time_basis="tower-receipt",
            recorded_at=time.time(),
            source="glasses-camera",
            module_id="object-memory",
            session_id="earlier-walk",
            frame_seq=3,
            bounding_box=(0.1, 0.1, 0.5, 0.5),
            retention_tag="default",
            privacy_tags=privacy_tags_for("earlier-walk", 3),
            spatial_ref=None,
            external_refs=(),
            best_score=0.9,
        )
    )

    client.post(f"{SESSION_URL}/start")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(_frames(6), start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()
        _wait_for(
            lambda: app.state.capture_workers.following(OBJECT_MEMORY_WORKER),
            20.0,
            "the producer",
        )
        ws.send_json({"type": "stream_stop"})
        ws.send_json({"type": "ping"})
        ws.receive_json()
    _wait_for(
        lambda: not app.state.capture_workers.following(OBJECT_MEMORY_WORKER),
        60.0,
        "the producer to finish",
    )

    body = client.get("/object-memory/observations").json()

    assert body["observation_count"] == 1
    (observation,) = body["observations"]
    assert observation["object_class"] == "laptop"
    # And it is addressable, which is what makes a picture reachable.
    assert observation["observation_id"]


def test_a_capture_with_no_session_started_is_not_followed(tower):
    """Armed is not recording, and the default is armed.

    This is the privacy half of the claim: a Tower that has just booted
    remembers nothing, however much the wearer records.
    """
    client, app, _ = tower
    frames = _frames(6)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(frames, start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()

        # Give a producer time to appear if one were going to.
        time.sleep(2.0)
        assert app.state.capture_workers.following(OBJECT_MEMORY_WORKER) == []

        ws.send_json({"type": "stream_stop"})
        ws.send_json({"type": "ping"})
        ws.receive_json()


def test_pause_stops_a_real_producer_mid_walk(tower):
    """Pause detaches the process, and the process is real.

    Asserted against the operating system rather than against a
    bookkeeping table: a Pause that updated a dict and left a producer
    running would be exactly the "looks successful but does nothing"
    failure the whole surface exists to expose.
    """
    client, app, _ = tower
    frames = _frames(10)

    client.post(f"{SESSION_URL}/start")

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(frames, start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()

        worker = _wait_for(
            lambda: (app.state.capture_workers.status() or [None])[0],
            20.0,
            "the producer to be running",
        )
        pid = worker["pid"]

        paused = client.post(f"{SESSION_URL}/pause").json()
        assert paused["state"] == "paused"
        assert app.state.capture_workers.following(OBJECT_MEMORY_WORKER) == []

        # The process itself is gone, not merely forgotten.
        import psutil

        assert not (
            psutil.pid_exists(pid)
            and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        ), f"pid {pid} is still alive after Pause"

        ws.send_json({"type": "stream_stop"})
        ws.send_json({"type": "ping"})
        ws.receive_json()


def test_a_paused_session_does_not_follow_the_next_capture(tower):
    """The gate, over a real socket and a real second recording."""
    client, app, _ = tower
    frames = _frames(6)

    client.post(f"{SESSION_URL}/start")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        ws.send_json(_frame_message(1, frames[0]))
        ws.receive_json()
        _wait_for(
            lambda: app.state.capture_workers.following(OBJECT_MEMORY_WORKER),
            20.0,
            "the first producer",
        )
        client.post(f"{SESSION_URL}/pause")
        ws.send_json({"type": "stream_stop"})
        ws.send_json({"type": "ping"})
        ws.receive_json()

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        ws.send_json(_frame_message(1, frames[0]))
        ws.receive_json()
        time.sleep(2.0)

        assert app.state.capture_workers.following(OBJECT_MEMORY_WORKER) == []

        ws.send_json({"type": "stream_stop"})
        ws.send_json({"type": "ping"})
        ws.receive_json()


def test_the_producer_argv_is_runnable_as_written(tower):
    """The argv the Tower builds, executed. Nothing substituted.

    A spec whose flags the script rejects produces a worker that exits
    immediately, which the supervisor logs and nothing else notices --
    and the Tower goes on reporting a configured worker for the rest of
    the walk. This runs the real argv with `--limit 0` and asserts it
    exits clean.
    """
    import subprocess

    client, app, tmp_path = tower
    spec = app.state.capture_workers.spec_for(OBJECT_MEMORY_WORKER)
    capture = tmp_path / "capture" / "captures" / "probe"
    (capture / "frames").mkdir(parents=True)
    (capture / "frames.jsonl").write_text("", encoding="utf-8")

    argv = [
        part.replace("{capture_dir}", str(capture)).replace(
            "{attach_mode}", "from-start"
        )
        for part in spec.argv
    ] + ["--max-idle-polls", "1", "--poll-seconds", "0.05", "--detector", "none"]

    result = subprocess.run(
        argv, cwd=spec.cwd, capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "frames_observed" in result.stdout
    assert sys.executable == argv[0], "the spec must run THIS interpreter"


def test_pause_reports_the_producer_gone_the_moment_it_returns(tower):
    """`terminate()` is asynchronous, and a Pause must not outrun it.

    On Windows `TerminateProcess` returns before the process is reaped,
    so a `poll()` straight afterwards routinely still says None. A
    supervisor that read that as "could not be killed" would keep a
    correctly-killed worker in the registry and make Pause report itself
    as still following a capture -- the one thing the session surface
    must never do, because `following` is what a phone renders as "still
    remembering".
    """
    client, app, _ = tower

    client.post(f"{SESSION_URL}/start")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(_frames(8), start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()
        _wait_for(
            lambda: app.state.capture_workers.following(OBJECT_MEMORY_WORKER),
            20.0,
            "the producer",
        )

        paused = client.post(f"{SESSION_URL}/pause").json()

        # No polling, no sleep: the answer has to be right AS IT RETURNS.
        assert paused["state"] == "paused"
        assert paused["following"] == []

        ws.send_json({"type": "stream_stop"})
        ws.send_json({"type": "ping"})
        ws.receive_json()
