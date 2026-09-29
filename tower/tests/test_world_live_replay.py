"""C22: the real-time replay harness -- pacing, the /ws framing, and the report.

The pacing tests pin that a replay keeps the RECORDED pace (absolute offsets,
never accumulated lateness) and the walk's shape (Stop vs disconnect, a
reconnected walk's second socket, a truncated walk ending with Stop). The
framing tests run the client against a fake /ws server and check every
message the phone would have sent, including a reconnect's re-sent
`session/start` and an abort mid-stream. The report tests parse a log
written in the Tower's own line formats (copied from walk 5's log) and a
small world root, and check the waterfall, the milestones, the 10-minute
verdict, the live-safety gate (C19 F8) and the N-run comparison. The runner
tests drive its whole lifecycle with fakes for the process, the job and the
network.

No test here starts a Tower, touches a GPU or reads the live store. None
reaches :8000 or a port in 8031-8040: an autouse fixture turns any such
HTTP request, socket open or port probe into a test failure.
"""

import asyncio
import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from scripts import world_live_replay as replay
from scripts import world_live_replay_report as report
from scripts import world_live_replay_run as runner

# -- helpers -----------------------------------------------------------------------


def _forbidden(port) -> bool:
    return port == 8000 or (port is not None and 8031 <= port <= 8040)


@pytest.fixture(autouse=True)
def _never_the_live_tower_or_a_test_port(monkeypatch):
    """Every test: no HTTP to, no socket to, no probe of :8000 or 8031-8040.
    A full OLD baseline replay may be running on 8031 while these run."""
    real_urlopen = urllib.request.urlopen

    def guarded_urlopen(url, *args, **kwargs):
        target = getattr(url, "full_url", None) or str(url)
        if _forbidden(urllib.parse.urlsplit(target).port):
            raise AssertionError(f"a test tried to reach {target}")
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)
    real_open = replay.TowerSocket.open

    async def guarded_open(self):
        if _forbidden(urllib.parse.urlsplit(self.uri).port):
            raise AssertionError(f"a test tried to open {self.uri}")
        return await real_open(self)

    monkeypatch.setattr(replay.TowerSocket, "open", guarded_open)

    def no_probe(port):
        raise AssertionError(f"a test probed a real port {port}; monkeypatch runner.port_in_use")

    monkeypatch.setattr(runner, "port_in_use", no_probe)


def _capture(tmp_path, capture_id, *, started, frames, ended, end_reason="stop", continues=None,
             jpeg=b"\xff\xd8fake-jpeg\xff\xd9"):
    """A capture directory as the Tower's recorder writes it."""
    directory = tmp_path / "captures" / capture_id
    (directory / "frames").mkdir(parents=True)
    rows = []
    for index, (seq, received_at) in enumerate(frames):
        relpath = f"frames/{seq:08d}.jpg"
        (directory / relpath).write_bytes(jpeg + bytes([index % 256]))
        rows.append(json.dumps({"schema_version": 1, "source_seq": seq, "wire_seq": seq, "tx_seq": index,
                                "received_at": received_at, "time_basis": "tower-receipt",
                                "relpath": relpath, "byte_count": len(jpeg) + 1,
                                "width": 360, "height": 640}))
    (directory / "frames.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (directory / "capture.json").write_text(json.dumps({
        "schema_version": 1, "capture_id": capture_id, "started_at": started, "ended_at": ended,
        "end_reason": end_reason, "continues_capture": continues, "frames_written": len(frames)}),
        encoding="utf-8")
    return directory


A = "a" * 32
B = "b" * 32
C1 = "1" * 32
C2 = "2" * 32

IDLE_HEALTH = {"capture": {"recording": False, "capture_id": None}, "capture_workers": {"workers": []},
               "background_chore": {"state": "idle", "runs": 1}}


# -- the walk and its schedule --------------------------------------------------------


def test_the_schedule_keeps_the_recorded_offsets(tmp_path):
    _capture(tmp_path, A, started=1000.0, frames=[(1, 1001.5), (3, 1001.6), (5, 1001.75)], ended=1002.0)
    walk = replay.load_walk(tmp_path / "captures", [A])
    steps = replay.build_schedule(walk)
    assert [(s.kind, round(s.at, 3)) for s in steps] == [
        ("stream_start", 0.0), ("frame", 1.5), ("frame", 1.6), ("frame", 1.75), ("stream_stop", 2.0)]
    assert [s.frame.wire_seq for s in steps if s.kind == "frame"] == [1, 3, 5]


def test_speed_divides_every_offset(tmp_path):
    _capture(tmp_path, A, started=0.0, frames=[(1, 1.0), (2, 2.0)], ended=4.0)
    steps = replay.build_schedule(replay.load_walk(tmp_path / "captures", [A]), speed=2.0)
    assert [round(s.at, 3) for s in steps] == [0.0, 0.5, 1.0, 2.0]
    with pytest.raises(ValueError):
        replay.build_schedule(replay.load_walk(tmp_path / "captures", [A]), speed=0)


def test_frames_are_never_scheduled_out_of_journal_order(tmp_path):
    _capture(tmp_path, A, started=0.0, frames=[(1, 1.0), (2, 0.9), (3, 1.2)], ended=2.0)
    steps = replay.build_schedule(replay.load_walk(tmp_path / "captures", [A]))
    frames = [s for s in steps if s.kind == "frame"]
    assert [f.frame.wire_seq for f in frames] == [1, 2, 3]
    assert [f.at for f in frames] == [1.0, 1.0, 1.2]


def test_first_seconds_ends_the_walk_with_stop_there(tmp_path):
    _capture(tmp_path, A, started=0.0, frames=[(i, float(i)) for i in range(1, 11)], ended=11.0)
    steps = replay.build_schedule(replay.load_walk(tmp_path / "captures", [A]), first_seconds=4.5)
    assert [s.kind for s in steps][-1] == "stream_stop"
    assert steps[-1].at == 4.5
    assert [s.frame.wire_seq for s in steps if s.kind == "frame"] == [1, 2, 3, 4]


def test_a_reconnected_walk_is_two_sockets_and_one_stop(tmp_path):
    _capture(tmp_path, A, started=100.0, frames=[(1, 101.0), (2, 102.0)], ended=103.0, end_reason="disconnect")
    _capture(tmp_path, B, started=110.0, frames=[(3, 111.0)], ended=112.0, end_reason="stop", continues=A)
    walk = replay.load_walk(tmp_path / "captures", [B])  # followed back from the successor
    assert [c.capture_id for c in walk] == [A, B]
    steps = replay.build_schedule(walk)
    assert [(s.kind, s.at) for s in steps] == [
        ("stream_start", 0.0), ("frame", 1.0), ("frame", 2.0), ("disconnect", 3.0),
        ("connect", 9.5), ("stream_start", 10.0), ("frame", 11.0), ("stream_stop", 12.0)]
    # And forwards from the first capture.
    assert [c.capture_id for c in replay.load_walk(tmp_path / "captures", [A])] == [A, B]


def test_a_walk_that_ended_by_disconnect_is_replayed_as_one_unless_asked(tmp_path):
    _capture(tmp_path, A, started=0.0, frames=[(1, 1.0)], ended=2.0, end_reason="disconnect")
    walk = replay.load_walk(tmp_path / "captures", [A])
    assert replay.build_schedule(walk)[-1].kind == "disconnect"
    assert replay.build_schedule(walk, end_with_stop=True)[-1].kind == "stream_stop"


def test_a_cut_inside_a_reconnect_gap_sends_no_stop(tmp_path):
    _capture(tmp_path, A, started=0.0, frames=[(1, 1.0)], ended=2.0, end_reason="disconnect")
    _capture(tmp_path, B, started=10.0, frames=[(2, 11.0)], ended=12.0, continues=A)
    steps = replay.build_schedule(replay.load_walk(tmp_path / "captures", [A]), first_seconds=9.8)
    assert [s.kind for s in steps] == ["stream_start", "frame", "disconnect"]


def test_a_torn_last_journal_line_is_dropped_but_a_bad_middle_line_is_not(tmp_path):
    directory = _capture(tmp_path, A, started=0.0, frames=[(1, 1.0), (2, 2.0)], ended=3.0)
    journal = directory / "frames.jsonl"
    journal.write_text(journal.read_text(encoding="utf-8") + '{"wire_seq": 3, "rece', encoding="utf-8")
    assert len(replay.read_capture(directory).frames) == 2
    lines = journal.read_text(encoding="utf-8").splitlines()
    journal.write_text("\n".join([lines[0], "not json", lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        replay.read_capture(directory)


class _FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        # Wake a little early, as asyncio does within one clock tick.
        self.now += max(0.001, seconds - 0.004)


def test_the_pacer_waits_for_absolute_offsets_and_reports_lateness():
    clock = _FakeClock()
    pacer = replay.Pacer(clock=clock, sleep=clock.sleep)
    pacer.start()

    async def go():
        late = [await pacer.wait_until(at) for at in (0.1, 0.2, 0.3)]
        clock.now += 0.5  # one send stalls for half a second
        late.append(await pacer.wait_until(0.4))
        late.append(await pacer.wait_until(1.2))
        return late

    late = asyncio.run(go())
    assert all(0 <= value < 0.002 for value in late[:3])
    assert late[3] == pytest.approx(0.4, abs=0.002)  # 0.4 s behind right after the stall
    # The stall does not shift the schedule: 1.2 s is still met at 1.2 s.
    assert clock.now == pytest.approx(1.2, abs=0.002)
    assert late[4] < 0.002


def test_distribution_has_p99_by_nearest_rank():
    assert replay.distribution([]) == {"count": 0}
    stats = replay.distribution([1, 2, 3, 4, 100])
    assert stats["count"] == 5 and stats["p50"] == 3 and stats["max"] == 100
    hundred = list(range(1, 202))  # 201 values: index round(q * 200)
    assert replay.distribution(hundred)["p99"] == 199
    assert report.distribution(hundred)["p99"] == 199
    assert report.distribution(hundred)["p95"] == 191


def test_the_live_tower_port_and_low_ports_are_refused():
    with pytest.raises(SystemExit):
        replay.check_target_port(8000)
    with pytest.raises(SystemExit):
        replay.check_target_port(8030)
    replay.check_target_port(8031)


def test_the_frame_message_is_the_phones():
    frame = replay.FrameRecord(wire_seq=12, source_seq=12, tx_seq=3, received_at=1.0,
                               relpath="frames/00000012.jpg", width=360, height=640)
    message = replay.frame_message(frame, b"\xff\xd8abc")
    assert set(message) == {"type", "seq", "tx_seq", "width", "height", "format", "data"}
    assert message["type"] == "frame" and message["format"] == "jpeg"
    assert (message["seq"], message["tx_seq"], message["width"], message["height"]) == (12, 3, 360, 640)
    assert base64.b64decode(message["data"]) == b"\xff\xd8abc"
    other = replay.FrameRecord(wire_seq=12, source_seq=40, tx_seq=None, received_at=1.0,
                               relpath="x", width=1, height=1)
    assert replay.frame_message(other, b"")["source_seq"] == 40
    assert "tx_seq" not in replay.frame_message(other, b"")


# -- the socket, against a fake /ws server ----------------------------------------------


class FakeTower:
    """A /ws server and a /health + session/start that answer as a Tower does.

    `order` is every message the Tower saw, the socket's and the HTTP POSTs,
    in the order it saw them. /health names the capture of the latest
    `stream_start`; `stale_after_reconnect` keeps naming the PREVIOUS capture
    for that many polls after a reconnect's stream_start, as a Tower can.
    """

    def __init__(self, capture_ids, *, stale_after_reconnect=0, on_frame=None):
        self.capture_ids = capture_ids
        self.order = []
        self.received = []
        self.connections = 0
        self.stream_starts = 0
        self.frames = 0
        self.settling = False
        self.stale_after_reconnect = stale_after_reconnect
        self._stale_served = 0
        self.on_frame = on_frame
        self.settle_states = iter([
            {"capture_workers": {"workers": [{"pid": 1}]}, "background_chore": {"state": "idle", "runs": 1}},
            {"capture_workers": {"workers": []}, "background_chore": {"state": "running", "runs": 2}},
        ])

    async def handler(self, connection):
        import websockets

        self.connections += 1
        number = self.connections
        try:
            async for raw in connection:
                message = json.loads(raw)
                kind = message["type"]
                self.received.append((number, message))
                self.order.append(("ws", kind))
                if kind == "ping":
                    await connection.send(json.dumps({"type": "pong"}))
                elif kind == "cartridges":
                    await connection.send(json.dumps({"type": "cartridges", "cartridges": [
                        {"cartridge": "world_builder", "result_type": "status",
                         "contract": "world_builder.status/test", "available": True}]}))
                elif kind == "result_subscribe":
                    assert message["contract"] == "world_builder.status/test"
                    await connection.send(json.dumps({"type": "result_subscribed", "subscription_id": "s1"}))
                    await connection.send(json.dumps({
                        "type": "cartridge_result", "cartridge": "world_builder", "seq": 1,
                        "revision_changed": True,
                        "payload": {"model_state": "receiving", "lifecycle": {"state": "receiving"},
                                    "world_snapshot": {"world_id": "w"}, "session": {"session_id": "s"},
                                    "geometry": {"revision": "r1"}}}))
                elif kind == "stream_start":
                    self.stream_starts += 1
                elif kind == "frame":
                    self.frames += 1
                    await connection.send(json.dumps({"type": "frame_result", "seq": message["seq"]}))
                    if self.on_frame is not None:
                        self.on_frame(self)
        except websockets.ConnectionClosed:
            pass

    def http_json(self, method, url, timeout=10.0):
        if method == "POST":
            assert url.endswith("/cartridges/world_builder/session/start")
            self.order.append(("http", "session/start"))
            return 200, {"state": "active", "session_id": "sess"}
        if self.settling:
            return 200, next(self.settle_states, {"capture_workers": {"workers": []}, "background_chore": {
                "state": "idle", "runs": 2, "last": {"outcome": "finished"}}})
        if self.stream_starts == 0:
            return 200, IDLE_HEALTH
        index = self.stream_starts - 1
        if index > 0 and self._stale_served < self.stale_after_reconnect:
            self._stale_served += 1
            index -= 1
        return 200, {"capture": {"recording": True, "capture_id": self.capture_ids[index]},
                     "capture_workers": {"workers": [{}]}, "background_chore": {"state": "idle", "runs": 1}}


def _serve_and_replay(monkeypatch, tower, options_for):
    """Run the replay against `tower` on an ephemeral port."""
    import websockets

    monkeypatch.setattr(replay, "http_json", tower.http_json)
    original = replay._follow_settle

    async def follow(*args, **kwargs):
        tower.settling = True
        return await original(*args, **kwargs)

    monkeypatch.setattr(replay, "_follow_settle", follow)

    async def go():
        async with websockets.serve(tower.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            return await replay.run_replay(options_for(port))

    return asyncio.run(go())


def test_a_replay_against_a_fake_tower_speaks_the_phones_protocol(tmp_path, monkeypatch):
    """The whole client run: handshake, session start, stream, Stop, settle."""
    jpeg = b"\xff\xd8replay\xff\xd9"
    _capture(tmp_path, A, started=500.0, frames=[(1, 500.05), (3, 500.10), (5, 500.15)],
             ended=500.2, jpeg=jpeg)
    tower = FakeTower([C1])
    record = _serve_and_replay(monkeypatch, tower, lambda port: replay.ReplayOptions(
        port=port, captures=[A], capture_root=tmp_path / "captures", out=tmp_path / "out",
        settle_timeout_min=0.5, poll_seconds=0.05, sample_seconds=60.0, session_lead=0.0,
        live_guard=False, phone_fetches=False))

    kinds = [kind for _, kind in tower.order]
    assert kinds[:3] == ["ping", "cartridges", "result_subscribe"]
    assert tower.order.index(("http", "session/start")) < tower.order.index(("ws", "stream_start"))
    assert kinds[-1] == "stream_stop"
    frames = [m for _, m in tower.received if m["type"] == "frame"]
    assert [m["seq"] for m in frames] == [1, 3, 5]
    assert [m["tx_seq"] for m in frames] == [0, 1, 2]
    assert all(base64.b64decode(m["data"]).startswith(jpeg) for m in frames)
    assert record["outcome"] == "settled"
    assert record["stream"]["frames_sent"] == 3
    assert record["stream"]["frame_results"] == 3
    assert record["stream"]["unanswered"] == 0
    assert record["tower_captures"] == [C1]
    assert record["phone_view"]["transitions"][0]["changed"]["model_state"] == "receiving"
    assert record["target_listener"] == {"checked": False, "why": "no tower pid given"}
    assert record["harness"]["files_sha1"]["world_live_replay.py"]
    saved = json.loads((tmp_path / "out" / "client.json").read_text(encoding="utf-8"))
    assert saved["outcome"] == "settled"


def test_a_reconnect_re_sends_the_handshake_and_session_start_before_stream_start(tmp_path, monkeypatch):
    """T1 / H1: the phone re-POSTs session/start on every reconnect; without it
    the Tower's deferred stop ends World Builder 105 s after the disconnect.
    Also L1: /health still naming the old capture does not lose the new one."""
    _capture(tmp_path, A, started=0.0, frames=[(1, 0.05), (2, 0.10)], ended=0.15, end_reason="disconnect")
    # B streams long enough (to 2.1 s) for the capture lookup's stale poll
    # (1.4 s) and fresh poll (1.9 s) to land before Stop.
    _capture(tmp_path, B, started=0.9, frames=[(3, 0.95), (4, 1.5), (5, 2.0)], ended=2.1, end_reason="stop",
             continues=A)
    tower = FakeTower([C1, C2], stale_after_reconnect=1)
    record = _serve_and_replay(monkeypatch, tower, lambda port: replay.ReplayOptions(
        port=port, captures=[A], capture_root=tmp_path / "captures", out=tmp_path / "out",
        settle_timeout_min=0.5, poll_seconds=0.05, sample_seconds=60.0, session_lead=0.0,
        live_guard=False, phone_fetches=False))

    assert tower.connections == 2
    pings = [index for index, item in enumerate(tower.order) if item == ("ws", "ping")]
    assert len(pings) == 2
    expected = [("ws", "ping"), ("ws", "cartridges"), ("ws", "result_subscribe"), ("http", "session/start"),
                ("ws", "stream_start")]
    assert tower.order[pings[0]:pings[0] + 5] == expected
    assert tower.order[pings[1]:pings[1] + 5] == expected  # the reconnect, as the phone does it
    second = [m["type"] for number, m in tower.received if number == 2]
    assert second == ["ping", "cartridges", "result_subscribe", "stream_start", "frame", "frame", "frame",
                      "stream_stop"]
    reconnected = next(e for e in record["events"] if e["kind"] == "reconnected")
    assert reconnected["session_start"]["status"] == 200
    assert record["tower_captures"] == [C1, C2]
    assert record["outcome"] == "settled"
    assert record["stream"]["frames_sent"] == 5 and record["stream"]["connections"] == 2


def test_an_abort_mid_stream_kills_the_test_tower_before_the_teardown(tmp_path, monkeypatch):
    """T2 / M3: the guard sees a live walk, streaming stops, `on_abort` (the
    runner's kill) runs BEFORE the client closes its socket, exit 3."""
    _capture(tmp_path, A, started=0.0, frames=[(i, 0.05 * i) for i in range(1, 41)], ended=2.1)
    calls = []
    tower = FakeTower([C1])
    monkeypatch.setattr(replay, "live_tower_state",
                        lambda url=None, timeout=None: "recording" if tower.frames >= 5 else "idle")
    real_close = replay.TowerSocket.close

    async def close(self):
        calls.append("socket-close")
        await real_close(self)

    monkeypatch.setattr(replay.TowerSocket, "close", close)

    def on_abort():
        time.sleep(0.2)  # a kill takes a moment; the teardown must still wait for it
        calls.append("on_abort")

    record = _serve_and_replay(monkeypatch, tower, lambda port: replay.ReplayOptions(
        port=port, captures=[A], capture_root=tmp_path / "captures", out=tmp_path / "out",
        settle_timeout_min=0.5, poll_seconds=0.05, sample_seconds=60.0, session_lead=0.0,
        live_guard=True, live_guard_every=0.02, phone_fetches=False, on_abort=on_abort))

    assert record["outcome"] == "aborted" and replay.exit_code(record) == replay.EXIT_ABORTED
    assert "live walk" in record["aborted"]["reason"]
    assert record["aborted"]["on_abort_done"] >= record["aborted"]["t"]
    assert 5 <= record["stream"]["frames_sent"] < 40
    assert calls[0] == "on_abort" and "socket-close" in calls
    assert "settle" not in record
    assert record["live_tower_watch"]["states_seen"] == ["idle", "recording"]


def _refusal_options(tmp_path, **extra):
    _capture(tmp_path, A, started=0.0, frames=[(1, 0.1)], ended=0.2)
    return replay.ReplayOptions(port=8031, captures=[A], capture_root=tmp_path / "captures",
                                out=tmp_path / "out", **extra)


@pytest.mark.parametrize("state, outcome", [("recording", "refused-live-walk"), ("busy", "refused-live-busy"),
                                            ("unknown", "refused-live-unknown")])
def test_the_replay_does_not_start_unless_8000_is_idle_or_down(tmp_path, monkeypatch, state, outcome):
    """M1 / M2: recording, finishing a world, and no usable answer all refuse."""
    monkeypatch.setattr(replay, "live_tower_state", lambda url=None, timeout=None: state)
    monkeypatch.setattr(replay, "http_json", lambda *a, **k: pytest.fail("streamed toward a target"))
    record = asyncio.run(replay.run_replay(_refusal_options(tmp_path)))
    assert record["outcome"] == outcome
    assert replay.exit_code(record) == replay.EXIT_ABORTED
    assert record["live_tower_at_start"] == state


def test_a_down_live_tower_lets_the_replay_go_on_to_its_target(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "live_tower_state", lambda url=None, timeout=None: "down")
    monkeypatch.setattr(replay, "http_json", lambda *a, **k: (None, None))
    record = asyncio.run(replay.run_replay(_refusal_options(tmp_path)))
    assert record["outcome"] == "no-test-tower"  # past the guard; no target answered


@pytest.mark.parametrize("health, why", [
    ({**IDLE_HEALTH, "capture": {"recording": True, "capture_id": C1}}, "recording"),
    ({**IDLE_HEALTH, "capture_workers": {"workers": [{"pid": 7}]}}, "capture worker"),
])
def test_the_replay_refuses_a_target_that_is_recording_or_has_workers(tmp_path, monkeypatch, health, why):
    """M4: a second stream_start into somebody's run supersedes its recording."""
    monkeypatch.setattr(replay, "live_tower_state", lambda url=None, timeout=None: "idle")
    monkeypatch.setattr(replay, "http_json", lambda *a, **k: (200, health))
    record = asyncio.run(replay.run_replay(_refusal_options(tmp_path)))
    assert record["outcome"] == "refused-target-busy"
    assert replay.exit_code(record) == replay.EXIT_ABORTED
    assert why in (tmp_path / "out" / "client.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("owners", [{9999}, {4242, 9999}, None])
def test_the_replay_refuses_a_target_served_by_another_process(tmp_path, monkeypatch, owners):
    """M4: given the test Tower's pid, the listener on the port must be it."""
    monkeypatch.setattr(replay, "live_tower_state", lambda url=None, timeout=None: "idle")
    monkeypatch.setattr(replay, "http_json", lambda *a, **k: (200, IDLE_HEALTH))
    monkeypatch.setattr(replay, "listener_pids", lambda port: owners)
    record = asyncio.run(replay.run_replay(_refusal_options(tmp_path, tower_pid=4242)))
    assert record["outcome"] == "refused-target-foreign"
    assert record["target_listener"]["checked"] is True


def test_a_live_walk_starting_mid_replay_aborts_it_and_calls_on_abort(tmp_path, monkeypatch):
    answers = iter(["idle", "idle", "recording"])
    monkeypatch.setattr(replay, "live_tower_state", lambda url=None, timeout=None: next(answers))
    killed = []
    options = replay.ReplayOptions(port=8031, captures=[A], capture_root=tmp_path, out=tmp_path / "out",
                                   live_guard_every=0.01, on_abort=lambda: killed.append(True))
    (tmp_path / "out").mkdir()

    async def go():
        abort, stop, record = asyncio.Event(), asyncio.Event(), {}
        await asyncio.wait_for(replay._guard_live(options, abort, record, stop, replay.LiveGuard("idle")), 5)
        return abort.is_set(), record

    aborted, record = asyncio.run(go())
    assert aborted and "live walk" in record["aborted"]["reason"]
    assert killed == [True] and "on_abort_done" in record["aborted"]
    assert replay.exit_code({"outcome": "aborted"}) == replay.EXIT_ABORTED


def test_the_live_guard_fails_closed_on_two_unusable_answers_and_records_busy():
    """M1: one late answer is a loaded machine; two in a row abort. M2: busy
    is recorded, not aborted on."""
    clock = iter(range(100, 200))
    guard = replay.LiveGuard("idle", clock=lambda: float(next(clock)))
    assert guard.observe("unknown") is None
    assert guard.observe("idle") is None  # the streak is broken
    assert guard.observe("busy") is None
    assert guard.busy_since is not None
    assert guard.observe("unknown") is None
    assert "failing closed" in guard.observe("unknown")
    assert "live walk" in replay.LiveGuard("down").observe("recording")
    assert guard.summary()["states_seen"] == ["busy", "idle", "unknown"]


@pytest.mark.parametrize("probe, state", [
    (("refused", None), "down"),
    (("timeout", None), "unknown"),
    (("error", {"status": 503}), "unknown"),
    (("error", None), "unknown"),
    (("ok", {"capture": {"recording": True}}), "recording"),
    (("ok", {"capture": {"recording": False}, "capture_workers": {"workers": [{"pid": 1}]}}), "busy"),
    (("ok", {"capture": {"recording": False}, "background_chore": {"state": "running"}}), "busy"),
    (("ok", IDLE_HEALTH), "idle"),
])
def test_the_live_tower_state_is_read_from_health(monkeypatch, probe, state):
    monkeypatch.setattr(replay, "health_probe", lambda url, timeout=None: probe)
    assert replay.live_tower_state() == state


class _Response:
    def __init__(self, body, status=200):
        self._body, self.status = body, status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("behaviour, kind", [
    (urllib.error.URLError(ConnectionRefusedError(10061, "refused")), "refused"),
    (ConnectionRefusedError(10061, "refused"), "refused"),
    (urllib.error.URLError(TimeoutError("timed out")), "timeout"),
    (TimeoutError("timed out"), "timeout"),
    (urllib.error.HTTPError("http://x/health", 503, "busy", {}, None), "error"),
    (ConnectionResetError(10054, "reset"), "error"),
    (_Response(b"not json"), "error"),
    (_Response(b"[1, 2]"), "error"),
    (_Response(b'{"capture": {}}'), "ok"),
])
def test_the_health_probe_tells_refused_from_slow(monkeypatch, behaviour, kind):
    """M1: only a refused connection is `down`; a timeout is never "not recording"."""
    def fake_urlopen(url, timeout=None):
        if isinstance(behaviour, BaseException):
            raise behaviour
        return behaviour

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert replay.health_probe("http://127.0.0.1:9/health", 5.0)[0] == kind


def test_geometry_coordinates_are_read_as_the_phone_reads_them():
    payload = {"world_snapshot": {"world_id": "w"}, "session": {"session_id": "s"},
               "geometry": {"revision": "g"}, "selection": {"world_id": "other"}}
    assert replay.geometry_coordinates(payload) == ("w", "s", "g")
    assert replay.geometry_coordinates({**payload, "geometry": {}}) is None
    assert replay.geometry_coordinates(None) is None


def test_the_mirror_is_asked_again_only_when_the_geometry_revision_moves():
    mirror = replay.GeometryMirror("http://127.0.0.1:8031")
    mirror.request(("w", "s", "r1"))
    assert mirror._wanted.is_set()
    mirror._wanted.clear()
    mirror.request(("w", "s", "r1"))  # a heartbeat, or a keyframe count moving
    assert not mirror._wanted.is_set()
    mirror.request(("w", "s", "r2"))
    assert mirror._wanted.is_set()
    off = replay.GeometryMirror("http://127.0.0.1:8031", enabled=False)
    off.request(("w", "s", "r1"))
    assert not off._wanted.is_set()


def test_precise_sleep_wakes_on_time():
    async def measure():
        started = time.perf_counter()
        await replay.precise_sleep(0.05)
        return time.perf_counter() - started

    elapsed = asyncio.run(measure())
    assert 0.049 <= elapsed < 0.2


def test_the_geometry_mirror_fetches_only_what_changed(monkeypatch):
    manifests = iter([
        {"segments": [{"segment_index": 0, "content_hash": "a", "placement_hash": None},
                      {"segment_index": 1, "content_hash": "b", "placement_hash": None}]},
        {"segments": [{"segment_index": 0, "content_hash": "a", "placement_hash": None},
                      {"segment_index": 1, "content_hash": "b2", "placement_hash": None}]},
    ])
    fetched = []
    monkeypatch.setattr(replay, "http_json", lambda method, url, timeout=10.0: (200, next(manifests)))
    monkeypatch.setattr(replay, "http_bytes", lambda url, timeout=30.0: (fetched.append(url) or 200, 10))
    mirror = replay.GeometryMirror("http://127.0.0.1:8031")

    async def go():
        await mirror._fetch(("w", "s", "r1"))
        await mirror._fetch(("w", "s", "r2"))

    asyncio.run(go())
    assert [url.split("/segment/")[1].split("?")[0] for url in fetched] == ["0", "1", "1"]
    assert mirror.summary()["manifests"] == 2 and mirror.summary()["segments"] == 3


def test_the_phone_view_records_only_transitions():
    view = replay.PhoneView(clock=lambda: 5.0)
    push = {"seq": 1, "revision_changed": True,
            "payload": {"model_state": "receiving", "lifecycle": {"state": "receiving", "reason": "x"},
                        "points": 10, "geometry": {"current": True, "revision": "r1"},
                        "world_snapshot": {"world_id": "w"}, "session": {"session_id": "s"}}}
    assert view.update(push) == ("w", "s", "r1")
    # `geometry.current` flips on every rebuild; it is not a state the phone shows.
    view.update({**push, "seq": 2, "payload": {**push["payload"], "geometry": {"current": False,
                                                                              "revision": "r1"}}})
    push3 = {"seq": 3, "revision_changed": False,
             "payload": {"model_state": "finalizing", "lifecycle": {"state": "finalizing"}}}
    assert view.update(push3) is None
    assert len(view.transitions) == 2
    assert view.transitions[1]["changed"] == {"model_state": "finalizing", "lifecycle.state": "finalizing"}
    assert "points" not in view.last


def test_the_settle_judge_waits_for_the_worker_and_a_chore_run_after_stop():
    judge = replay.SettleJudge(chore_runs_at_stop=1)
    assert not judge.settled({"health": True, "workers": 1, "chore_state": "idle", "chore_runs": 1})
    assert not judge.settled({"health": True, "workers": 0, "chore_state": "owed", "chore_runs": 1})
    assert not judge.settled({"health": True, "workers": 0, "chore_state": "running", "chore_runs": 2})
    assert judge.settled({"health": True, "workers": 0, "chore_state": "idle", "chore_runs": 2})
    assert replay.SettleJudge(None).settled({"health": True, "workers": 0, "chore_state": None})
    assert not replay.SettleJudge(0).settled({"health": False})


def test_settle_snapshot_flattens_health_and_session():
    snap = replay.settle_snapshot(
        {"capture_workers": {"workers": []}, "background_chore": {"state": "idle", "runs": 3,
                                                                  "last": {"outcome": "finished"}},
         "capture": {"recording": False}},
        {"finalization": {"state": "complete", "final_solve": "solved"},
         "stages": {"surface": {"state": "ok"}, "appearance": {"state": "running"}},
         "frames_observed": 10, "keyframes_accepted": 4})
    assert snap["workers"] == 0 and snap["chore_runs"] == 3 and snap["chore_last"] == "finished"
    assert snap["finalization"] == "complete" and snap["stage.appearance"] == "running"


def test_solution_snapshots_keep_each_version_once(tmp_path):
    """M6: draw 0's early publish survives the consensus overwriting it."""
    solve = tmp_path / "root" / "worlds" / "w" / "solve" / "s"
    solve.mkdir(parents=True)
    session = tmp_path / "root" / "worlds" / "w" / "sessions" / "s" / "session.json"
    path = replay.solution_path_for(session)
    assert path == solve / "solution.json"
    snaps = replay.SolutionSnapshots(tmp_path / "out")
    assert snaps.poll(path) is False  # not written yet
    path.write_text(json.dumps({"solved_at": 1.0, "timing": {"map_s": 30.5, "gate_s": 90.0},
                                "gate": {"consensus": {"state": "deferred"}}}), encoding="utf-8")
    assert snaps.poll(path) is True
    assert snaps.poll(path) is False  # unchanged
    path.write_text("{torn", encoding="utf-8")
    assert snaps.poll(path) is False  # skipped, retried next time
    os.utime(path, None)
    path.write_text(json.dumps({"solved_at": 2.0, "timing": {"map_s": 40.0},
                                "gate": {"consensus": {"state": "applied"}}}), encoding="utf-8")
    assert snaps.poll(path) is True
    index = json.loads((tmp_path / "out" / "solution-snapshots" / "index.json").read_text(encoding="utf-8"))
    assert [(i["file"], i["consensus_state"], i["map_s"]) for i in index] == [
        ("000.json", "deferred", 30.5), ("001.json", "applied", 40.0)]
    first = json.loads((tmp_path / "out" / "solution-snapshots" / "000.json").read_text(encoding="utf-8"))
    assert first["timing"]["map_s"] == 30.5


def test_the_surface_watch_keeps_transitions_and_their_kind(tmp_path):
    path = tmp_path / "status.json"
    watch = replay.SurfaceWatch()
    assert watch.poll(path) is False
    for doc in ({"pid": 5, "state": "running", "stage": "depth", "updated_at": 10.0, "params_digest": "x|live|y"},
                {"pid": 5, "state": "running", "stage": "depth", "updated_at": 10.0, "params_digest": "x|live|y"},
                {"pid": 5, "state": "ok", "updated_at": 40.0, "params_digest": "x|live|y"},
                {"pid": 9, "state": "ok", "updated_at": 90.0, "params_digest": "x|final|y"}):
        path.write_text(json.dumps(doc), encoding="utf-8")
        watch.poll(path)
    assert [(t["state"], t["kind"], t["updated_at"]) for t in watch.transitions] == [
        ("running", "live", 10.0), ("ok", "live", 40.0), ("ok", "final", 90.0)]


def test_the_sampler_never_appends_to_an_old_samples_file(tmp_path, monkeypatch):
    """L4: one samples.csv is one run's samples."""
    monkeypatch.setattr(replay, "gpu_sample", lambda: (None, None))
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("old,run\n1,2\n", encoding="utf-8")
    sampler = replay.ResourceSampler(None, 0.01, csv_path)
    sampler.start()
    time.sleep(0.1)
    sampler.stop()
    sampler.join(5)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("t,sys_cpu_pct") and "old,run" not in lines


def test_the_harness_pins_its_own_scripts():
    """L5: run.json and client.json carry the harness's own identity."""
    identity = replay.harness_identity()
    assert set(identity["files_sha1"]) == set(replay.HARNESS_FILES)
    assert all(len(v) == 40 for v in identity["files_sha1"].values())
    assert len(identity["sha1"]) == 40
    assert len(identity.get("git_head", "")) == 40  # this worktree is a checkout
    assert isinstance(identity.get("git_dirty"), list)


def test_a_reused_out_is_refused(tmp_path):
    replay.refuse_non_empty_out(tmp_path / "new")
    (tmp_path / "used").mkdir()
    replay.refuse_non_empty_out(tmp_path / "used")  # empty is fine
    (tmp_path / "used" / "client.log").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="not empty"):
        replay.refuse_non_empty_out(tmp_path / "used")


def test_the_standalone_client_needs_the_tower_pid_and_a_fresh_out(tmp_path):
    with pytest.raises(SystemExit) as missing:
        replay.main(["--capture", A, "--port", "8031", "--out", str(tmp_path / "out")])
    assert missing.value.code == 2
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "samples.csv").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="not empty"):
        replay.main(["--capture", A, "--port", "8031", "--tower-pid", "1", "--out", str(tmp_path / "used")])


# -- the report ----------------------------------------------------------------------------

W = "d" * 32
S = "e" * 32
CAP = "b5750fa3271d4e80b959c628e3e82c94"
CAP2 = "c" * 32


def _ts(clock):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(clock))) + f",{int(round((clock % 1) * 1000)):03d}"


def _line(clock, logger, message, level="INFO"):
    return f"{_ts(clock)} {level} {logger} {message}"


def _summary(frames, end_reason, **extra):
    fields = {"session_duration_s": 100.0, "frames_received": frames, "effective_fps": 12.0,
              "tx_seq_gap_total": 0, "backpressure_drops": 0, "frame_processing_errors": 0,
              "frames_rejected": 0, "receive_to_result_ms_max": 30.0, **extra, "end_reason": end_reason}
    return "[Tower][Session] final summary: " + repr(fields)


def _walk_log(base, *, reconnect=False, processing_errors=0):
    """A Tower log in the real line formats (walk 5's), 100 s walk, then the settle.
    `reconnect`: the walk is two captures, the first ended by disconnect."""
    wb = "tower.world_build_session"
    ws = "tower.routes.ws"
    lines = [
        "INFO:     Started server process [1]",
        _line(base - 2, "tower.cartridge_session", "[Tower][Session] world_builder start -> state=active"),
        _line(base, ws, "[Tower][Session] stream_start: measurement window opened"),
        _line(base + 0.006, ws, f"[Tower][Capture] recording started: {CAP}"),
        _line(base + 0.013, "tower.capture_workers",
              f"[Tower][Worker] started world-build-session pid 21204 for capture {CAP}: python x"),
        _line(base + 1.7, wb, f"[Tower][WorldBuilder] session {S} in world {W}: source=live-capture "
                              f"capture={CAP} root=x observed=360x640"),
        _line(base + 5.8, wb, "[Tower][WorldBuilder] rebuild 1: 4 keyframes -> 2 positioned poses, "
                              "586 points, 2 segments in 0.02s"),
        _line(base + 20.0, wb, "[Tower][WorldBuilder] background solve 1 launched at 52 keyframes (pid 28752)"),
        _line(base + 30.0, wb, "[Tower][WorldBuilder] rebuild 2: 8 keyframes -> 6 positioned poses, "
                               "1255 points, 2 segments in 0.50s"),
        _line(base + 40.0, wb, "[Tower][WorldBuilder] live surface 1 launched (pid 2656)"),
    ]
    if reconnect:
        lines += [
            _line(base + 50.0, ws, _summary(500, "disconnect", tx_seq_gap_total=2, receive_to_result_ms_max=90.5)),
            _line(base + 50.002, ws, "[Tower][Capture] recording stopped (disconnect): 500 frames, 400 bytes"),
            _line(base + 51.0, ws, "[Tower][Session] stream_start: measurement window opened"),
            _line(base + 51.006, ws, f"[Tower][Capture] recording started: {CAP2}"),
        ]
    lines += [
        _line(base + 99.0, wb, "[Tower][WorldBuilder] rebuild 3: 12 keyframes -> 10 positioned poses, "
                               "1823 points, 2 segments in 1.08s"),
        _line(base + 100.0, ws, _summary(700 if reconnect else 1200, "stream_stop",
                                         frame_processing_errors=processing_errors)),
        _line(base + 100.002, ws, f"[Tower][Capture] recording stopped (stop): {700 if reconnect else 1200} "
                                  "frames, 900 bytes"),
        _line(base + 101.5, wb, "[Tower][WorldBuilder] rebuild 4: 16 keyframes -> 12 positioned poses, "
                                "2000 points, 3 segments in 1.21s"),
        _line(base + 224.0, wb, "[Tower][WorldBuilder] background solve pid 31120 still running after "
                                "120.0s; terminating it so the final solve owns the workspace", "WARNING"),
        _line(base + 224.1, wb, "[Tower][WorldBuilder] final global solve launched (pid 34464)"),
        _line(base + 400.0, wb, "[Tower][WorldBuilder] final global solve: solved=True solver=glomap "
                                "posed=15/16 components=2 in 175.90s"),
        _line(base + 401.0, wb, f"[Tower][WorldBuilder] session {S} finished: 1200 frames, 16 keyframes, "
                                "3 segments, backend=classical-sfm (downgraded_from=None), 15 solved poses, "
                                "2000 points, scale=unknown, final build 1.00s"),
        _line(base + 460.0, "tower.world_builder.transients",
              f"[Tower][WorldBuilder][transients] {W}/{S}: 16 keyframes masked (16 computed, 0 cached) "
              "under union in 50.0 s"),
        _line(base + 520.0, "tower.world_builder.appearance_pipeline",
              f"[Tower][WorldBuilder][appearance] {W}/{S} built: 16 keyframes (8 phone), 2 chunks, 1.0 MB, "
              "{'detector': 10.0, 'encode': 5.0, 'total': 30.0}"),
        _line(base + 530.0, "tower.capture_workers",
              f"[Tower][Worker] world-build-session worker pid 21204 for capture {CAP} finished after "
              f"530.0s (lineage: {CAP})"),
        "[Tower][WorldBuilder][surface] not fusing 8 frames whose depth fit is inverted",
        _line(base + 534.0, "tower.main", "[Tower][Worker] started the world-finish-pending chore (the Tower "
                                          "is idle and work may be owed), pid 41924: python finish"),
        _line(base + 600.0, "tower.main", "[Tower][Worker] the world-finish-pending chore (pid 41924) exited 0 "
                                          "after 66.0 s: finished"),
        # A later walk on the same Tower must not leak into this one.
        _line(base + 900.0, ws, _summary(77, "stream_stop")),
        _line(base + 900.0, ws, f"[Tower][Capture] recording started: {'f' * 32}"),
        _line(base + 901.0, wb, "[Tower][WorldBuilder] rebuild 1: 4 keyframes -> 2 positioned poses, "
                                "5 points, 1 segments in 9.99s"),
    ]
    return "\n".join(lines) + "\n"


# The keyframes of the synthetic session: (source_seq, segment_index, received, accepted).
KEYFRAMES = [(1, 0, 10.0, 10.5), (10, 0, 11.0, 13.0), (54, 1, 12.0, 12.25), (80, 1, 20.0, 27.0)]


def _world(root, base, *, appearance_end, draw0_map=None):
    world = root / "worlds" / W
    (world / "sessions" / S).mkdir(parents=True)
    (world / "sessions" / S / "session.json").write_text(json.dumps({
        "session_id": S, "world_id": W, "capture_id": CAP, "ended_at": base + 104.0,
        "frames_observed": 1200, "keyframes_accepted": 4,
        "finalization": {"state": "complete", "final_solve": "solved", "started_at": base + 104.0,
                         "updated_at": base + 401.0},
        "stages": {"surface": {"state": "ok", "started_at": base + 401.0, "updated_at": base + 500.0},
                   "appearance": {"state": "ok", "started_at": base + 500.0, "updated_at": appearance_end},
                   "dense": {"state": "unavailable", "started_at": base + 525.0, "updated_at": base + 525.0}},
    }), encoding="utf-8")
    (world / "sessions" / S / "keyframes.jsonl").write_text("".join(
        json.dumps({"keyframe_id": f"{S}:{seq:08d}", "source_seq": seq, "segment_index": seg,
                    "received_at": base + got}) + "\n" for seq, seg, got, _ in KEYFRAMES), encoding="utf-8")
    (world / "sessions" / S / "events.jsonl").write_text("".join(
        [json.dumps({"kind": "session_started", "at": base}) + "\n"]
        + [json.dumps({"kind": "keyframe_accepted", "at": base + at,
                       "payload": {"keyframe_id": f"{S}:{seq:08d}"}}) + "\n" for seq, _, _, at in KEYFRAMES]),
        encoding="utf-8")
    (world / "solve" / S).mkdir(parents=True)
    (world / "solve" / S / "solution.json").write_text(json.dumps({
        "solved_at": base + 392.0,
        "timing": {"prepare_s": 1.0, "masks_s": 60.0, "freeze_s": 1.0, "extract_s": 1.0, "match_s": 5.0,
                   "map_s": 25.0, "gate_s": 20.0},
        "transients": {"computed": 16, "cache_hits": 0, "seconds": {"total": 60.0}},
        "gate": {"state": "applied", "consensus": {"requested": 2, "state": "applied",
                                                   "chosen": {"draw": 1, "seed": 1}, "draws": [
                                                       {"draw": 0, "seed": 0, "map_s": draw0_map, "gate_s": 30.0},
                                                       {"draw": 1, "seed": 1, "map_s": 25.0, "gate_s": 5.0}]}},
    }), encoding="utf-8")
    (world / "surface" / S).mkdir(parents=True)
    (world / "surface" / S / "status.json").write_text(json.dumps({
        "state": "ok", "result": {"seconds": {"transients": 50.0, "pack": 20.0}}}), encoding="utf-8")
    (world / "sessions" / S / "stage_timing.json").write_text(json.dumps({
        "schema": 2, "finishes": [{"finish_id": "1", "context": "solve", "started_at": base + 224.0,
                                   "stages": {"solve_draw": [{"wall_ms": 40000, "seed": 1, "draw_index": 0,
                                                              "status": "ok", "cache": {}}]}}]}),
        encoding="utf-8")
    (world / "areas" / "22a4b5d1fe879458").mkdir(parents=True)
    (world / "areas" / "22a4b5d1fe879458" / "record.json").write_text(json.dumps({
        "area_id": "22a4b5d1fe879458", "session_id": S, "updated_at": base + 599.0,
        "stages": {"surface": {"state": "ok", "started_at": base + 540.0, "updated_at": base + 590.0}}}),
        encoding="utf-8")
    return world


BASE = time.mktime(time.strptime("2026-09-28 04:43:42", "%Y-%m-%d %H:%M:%S"))


def test_the_timeline_is_read_from_the_towers_own_lines(tmp_path):
    base = BASE
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(base), encoding="utf-8")
    timeline = report.walk_timeline(report.scan_log(log), CAP)
    assert timeline["world_id"] == W and timeline["session_id"] == S
    assert timeline["stop"]["t"] == pytest.approx(base + 100.002, abs=0.002)
    assert [r["n"] for r in timeline["rebuilds"]] == [1, 2, 3, 4]  # the later walk's rebuild is excluded
    assert timeline["tower_summary"]["frames_received"] == 1200
    assert timeline["tower_totals"]["windows"] == 1  # the later walk's summary is excluded
    assert timeline["bg_solve_wait"]["pid"] == "31120"
    assert timeline["final_done"]["posed"] == "15" and timeline["final_done"]["s"] == "175.90"
    assert timeline["appearance_seconds"]["total"] == 30.0
    assert [c["outcome"] for c in timeline["chores"]] == ["finished"]
    assert [s["keyframes"] for s in timeline["background_solves"]] == [52]


def test_a_reconnected_walk_sums_every_measurement_window(tmp_path):
    """T5 / H2: the Tower's summary is per window; the last one alone undercounts."""
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE, reconnect=True, processing_errors=1), encoding="utf-8")
    timeline = report.walk_timeline(report.scan_log(log), CAP)
    assert timeline["captures"] == [CAP, CAP2]
    totals = timeline["tower_totals"]
    assert totals["windows"] == 2 and totals["end_reasons"] == ["disconnect", "stream_stop"]
    assert totals["frames_received"] == 1200 and timeline["tower_summary"]["frames_received"] == 700
    assert totals["tx_seq_gap_total"] == 2 and totals["frame_processing_errors"] == 1
    assert totals["receive_to_result_ms_max"] == 90.5
    assert timeline["recorded_frames"] == 1200
    _world(tmp_path / "root", BASE, appearance_end=BASE + 700)
    built = report.build_report(tower_log=log, world_root=tmp_path / "root", client={"tower_captures": [CAP]})
    rows = {r["check"]: r for r in built["live_safety"]["rows"]}
    assert rows["frames received (Tower, all windows) = frames recorded"]["result"] == "PASS"
    assert rows["frames observed (builder) = frames received"]["result"] == "PASS"
    assert rows["tx_seq gaps (all windows)"]["result"] == "FAIL"
    assert rows["frame processing errors (all windows)"]["value"] == 1
    assert built["live_safety"]["result"] == "FAIL"
    markdown = report.render_markdown(built)
    assert "frame processing errors 1" in markdown and "## 2. Live safety (C19 F8): **FAIL**" in markdown
    assert "ended by disconnect" in markdown


@pytest.mark.parametrize("appearance_after_stop_min, verdict", [(9.5, "PASS"), (10.5, "FAIL")])
def test_the_report_draws_the_waterfall_and_judges_the_hard_maximum(
        tmp_path, appearance_after_stop_min, verdict):
    base = BASE
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(base), encoding="utf-8")
    out_log = tmp_path / "tower.out.log"
    out_log.write_text("\n".join([
        'INFO:     127.0.0.1:1 - "GET /health HTTP/1.1" 200 OK',
        "{",
        '  "finished": [',
        'INFO:     127.0.0.1:2 - "GET /health HTTP/1.1" 200 OK',
        f'    {{"world_id": "{W}", "session_id": "{S}", "stage": "areas", "areas": [',
        '      {"area_id": "22a4b5d1fe879458", "prepare_s": 54.0,',
        '       "stages": {"surface": {"state": "ok", "seconds": {"pack": 81.9}},',
        '                  "appearance": {"state": "ok", "seconds": {"total": 68.9}}}}]}]',
        "}",
    ]) + "\n", encoding="utf-8")
    stop = base + 100.002
    _world(tmp_path / "root", base, appearance_end=stop + appearance_after_stop_min * 60)
    samples = tmp_path / "samples.csv"
    samples.write_text("t,sys_cpu_pct,tree_cores,tree_rss_mb,tree_procs,busy,gpu_util_pct,gpu_mem_mb\n"
                       + "".join(f"{base + 230 + 3 * i},5,1.0,100,4,world_solve.py:1.0,0,1000\n"
                                 for i in range(20)), encoding="utf-8")
    built = report.build_report(tower_log=log, tower_out_log=out_log, world_root=tmp_path / "root",
                                samples=samples, client={"tower_captures": [CAP]})
    assert built["verdict"]["result"] == verdict
    assert built["verdict"]["stop_to_room_with_photos_min"] == pytest.approx(appearance_after_stop_min, abs=0.01)
    assert built["verdict"]["stop_to_settled_min"] == pytest.approx((600.0 - 100.002) / 60, abs=0.01)
    numbers = [row["n"] for row in built["waterfall"] if row["n"]]
    assert numbers[:4] == ["1", "2", "3", "4"] and numbers[-1] == "Σ"
    solve_row = next(row for row in built["waterfall"] if row["n"] == "3")
    assert solve_row["minutes"] == pytest.approx((400.0 - 224.1) / 60, abs=0.01)
    assert solve_row["use"]["busy"] == ["world_solve.py"]
    assert built["areas_from_chore"][0]["appearance_total"] == 68.9
    assert built["tower_walk"]["rebuilds"]["after_stop"] == 1
    assert built["store"]["stage_timing"]["finishes"][0]["stages"]["solve_draw"][0]["wall_s"] == 40.0
    # M6: the solve sub-rows are the CHOSEN draw's, and say so.
    labels = [row["stage"] for row in built["waterfall"] if row.get("child")]
    assert "solve map (solution.json timing: the chosen draw's (draw 1, seed 1) map/gate)" in labels
    assert not any("last draw" in label for label in labels)
    assert "consensus draw 1 (seed 1) CHOSEN" in labels
    markdown = report.render_markdown(built)
    assert f"**{verdict}**" in markdown
    assert "| 3 | Final global solve, pid 34464 |" in markdown
    assert "solve_draw: 40.0 s (seed 1)" in markdown
    assert "## 2. Live safety (C19 F8): **PASS**" in markdown


def test_the_report_gives_the_keyframe_sequence_and_the_observe_lag(tmp_path):
    """T5 / H2: identity by (source_seq, segment_index), never keyframe_id, and
    the lag from `received_at` to `keyframe_accepted`."""
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _world(tmp_path / "root", BASE, appearance_end=BASE + 700)
    built = report.build_report(tower_log=log, world_root=tmp_path / "root", client={"tower_captures": [CAP]})
    keyframes = built["keyframes"]
    assert keyframes["sequence"] == [[1, 0], [10, 0], [54, 1], [80, 1]]
    assert keyframes["sha256"] == report.hashlib.sha256(
        b"[[1,0],[10,0],[54,1],[80,1]]").hexdigest()
    lag = keyframes["observe_lag_s"]  # 0.5, 2.0, 0.25, 7.0
    assert (lag["count"], lag["p50"], lag["p95"], lag["p99"], lag["max"]) == (4, 2.0, 7.0, 7.0, 7.0)
    assert keyframes["observe_lag_unmatched"] == 0
    report.write_report(tmp_path / "out", built)
    saved = json.loads((tmp_path / "out" / "keyframes.json").read_text(encoding="utf-8"))
    assert saved["sha256"] == keyframes["sha256"] and saved["sequence"][2] == [54, 1]
    # The same walk in another session (another id) is the same sequence.
    other = report.keyframe_facts(tmp_path / "root" / "worlds" / W / "sessions" / S)
    assert other["sha256"] == keyframes["sha256"]


def test_draw_zero_comes_from_the_snapshot_when_the_run_kept_one(tmp_path):
    """M6: the consensus overwrote draw 0's solution.json; the snapshot did not."""
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _world(tmp_path / "root", BASE, appearance_end=BASE + 700)
    run_dir = tmp_path / "run"
    snaps = run_dir / "solution-snapshots"
    snaps.mkdir(parents=True)
    (snaps / "000.json").write_text(json.dumps({"solved_at": BASE + 330.0, "timing": {"map_s": 37.7, "gate_s": 44.0},
                                                "gate": {"consensus": {"state": "deferred"}}}), encoding="utf-8")
    (snaps / "index.json").write_text(json.dumps([{"n": 0, "file": "000.json", "mtime": BASE + 336.0}]),
                                      encoding="utf-8")
    built = report.build_report(tower_log=log, world_root=tmp_path / "root", client={"tower_captures": [CAP]},
                                run_dir=run_dir)
    zero = built["solve_draw_0"]
    assert (zero["source"], zero["map_s"], zero["published_at"]) == ("snapshot", 37.7, BASE + 336.0)
    draw_row = next(r for r in built["waterfall"] if r["stage"].startswith("consensus draw 0"))
    assert draw_row["detail"].startswith("map 37.7 s (draw-0 snapshot)")
    assert built["milestones"]["draw0_published"]["t"] == BASE + 336.0


def _glog(clock):
    return time.strftime("%Y%m%d %H:%M:%S", time.localtime(int(clock))) + f".{int((clock % 1) * 1e6):06d}"


def test_draw_zero_is_estimated_from_solve_log_when_no_snapshot_exists(tmp_path):
    """M6 fallback for runs made before the snapshots: one end-of-map burst per
    draw in solve.log, calibrated on the chosen draw's solved_at."""
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    world = _world(tmp_path / "root", BASE, appearance_end=BASE + 700)
    (world / "solve" / S / "solve.log").write_text("\n".join([
        f"E{_glog(BASE + 90.0)}  7260 global_mapper.cc:26] a background solve, before the final launch",
        "Loading weights: 100%",
        f"E{_glog(BASE + 330.0)} 15500 global_mapper.cc:26] Cannot run bundle adjustment",
        f"E{_glog(BASE + 330.001)} 15500 global_pipeline.cc:190] Global mapping failed",
        f"E{_glog(BASE + 390.0)} 15500 global_mapper.cc:26] Cannot run bundle adjustment",
    ]) + "\n", encoding="utf-8")
    built = report.build_report(tower_log=log, world_root=tmp_path / "root", client={"tower_captures": [CAP]})
    zero = built["solve_draw_0"]
    # map starts at launch 224.1 + 68 s of pre-map stages = 292.1; draw 0 ends at
    # its marker 330.001 + the chosen draw's calibration (392 - 390 = 2 s).
    assert zero["source"] == "estimate"
    assert zero["map_s"] == pytest.approx(332.001 - 292.1, abs=0.05)
    assert zero["published_at"] == pytest.approx(BASE + 392.0 - 25.0, abs=0.01)
    assert "calibrated +2.00 s on chosen draw 1" in zero["how"]
    draw_row = next(r for r in built["waterfall"] if r["stage"].startswith("consensus draw 0"))
    assert "(estimated:" in draw_row["detail"]
    assert "draw0_published_estimated" in built["milestones"]


def test_live_surface_latency_needs_the_runs_surface_watch(tmp_path):
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _world(tmp_path / "root", BASE, appearance_end=BASE + 700)
    blind = report.build_report(tower_log=log, world_root=tmp_path / "root", client={"tower_captures": [CAP]})
    assert blind["live_surfaces_latency"]["computable"] is False
    watch = [{"t": BASE + 41.0, "state": "running", "kind": "live", "updated_at": BASE + 41.0},
             {"t": BASE + 71.0, "state": "ok", "kind": "live", "updated_at": BASE + 70.0},
             {"t": BASE + 500.0, "state": "ok", "kind": "final", "updated_at": BASE + 500.0}]
    seen = report.build_report(tower_log=log, world_root=tmp_path / "root",
                               client={"tower_captures": [CAP], "surface_watch": watch})
    surfaces = seen["live_surfaces_latency"]
    assert surfaces["computable"] and surfaces["surfaces"][0]["latency_s"] == 30.0
    assert surfaces["surfaces"][0]["killed_at_stop"] is False
    row = next(r for r in seen["live_safety"]["rows"] if r["check"].startswith("live surfaces"))
    assert row["result"] == "INFO" and "1 launched" in row["value"]


def test_the_live_safety_gate_checks_the_client_against_the_tower(tmp_path):
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _world(tmp_path / "root", BASE, appearance_end=BASE + 700)
    client = {"tower_captures": [CAP], "schedule": {"frames": 1200},
              "stream": {"frames_sent": 1200, "unanswered": 0, "frame_errors": {}},
              "live_tower_watch": {"states_seen": ["idle"]}}
    ok = report.build_report(tower_log=log, world_root=tmp_path / "root", client=client)
    assert ok["live_safety"]["result"] == "PASS"
    contended = report.build_report(tower_log=log, world_root=tmp_path / "root", client={
        **client, "stream": {"frames_sent": 1201, "unanswered": 1, "frame_errors": {"bad": 1}},
        "live_tower_watch": {"states_seen": ["busy", "idle"], "busy_since": BASE + 300}})
    rows = {r["check"]: r["result"] for r in contended["live_safety"]["rows"]}
    assert rows["frames sent (client) = frames scheduled"] == "FAIL"
    assert rows["frames received (Tower, all windows) = frames sent"] == "FAIL"
    assert rows["every frame answered (client): unanswered, frame_error replies"] == "FAIL"
    # A contended :8000 is the environment's FAIL, reported apart (C22-F2 item 3).
    environment = contended["live_safety"]["environment"]
    assert {r["check"]: r["result"] for r in environment["rows"]}[":8000 during the run"] == "FAIL"
    assert environment["result"] == "FAIL" and ":8000 during the run" not in rows


def test_the_report_says_not_reached_when_the_photos_never_came(tmp_path):
    base = BASE
    log = tmp_path / "tower.err.log"
    log.write_text("\n".join(_walk_log(base).splitlines()[:14]) + "\n", encoding="utf-8")
    built = report.build_report(tower_log=log, client={"tower_captures": [CAP]})
    assert built["verdict"]["result"] == "NOT REACHED"
    assert "not reached" in report.render_markdown(built)


def test_window_use_averages_the_samples_inside_a_row():
    rows = [{"t": 1.0, "tree_cores": 2.0, "gpu_util_pct": 50.0, "busy": "a.py:2.0"},
            {"t": 2.0, "tree_cores": 4.0, "gpu_util_pct": 100.0, "busy": "a.py:3.0;b.py:1.0"},
            {"t": 9.0, "tree_cores": 99.0, "gpu_util_pct": 0.0, "busy": ""}]
    use = report.window_use(rows, 0.5, 2.5)
    assert use["cores"] == {"mean": 3.0, "max": 4.0} and use["gpu"] == {"mean": 75.0, "max": 100.0}
    assert use["busy"] == ["a.py", "b.py"]
    assert report.window_use(rows, 3.0, 4.0) is None


def test_a_finished_run_is_re_reported_from_its_run_dir_into_another_out(tmp_path):
    """H2: everything is computable after the fact, from the run's --out, its
    data root and its log, without touching the run's own directory."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log = run_dir / "tower-8031-x.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    data_root = tmp_path / "data"
    _world(data_root / "world_builder", BASE, appearance_end=BASE + 700)
    (run_dir / "run.json").write_text(json.dumps({"err_log": str(log), "out_log": str(run_dir / "none.log"),
                                                  "data_root": str(data_root), "harness": {"sha1": "h"}}),
                                      encoding="utf-8")
    (run_dir / "client.json").write_text(json.dumps({"tower_captures": [CAP], "label": "old run 1"}),
                                         encoding="utf-8")
    before = sorted(p.name for p in run_dir.iterdir())
    assert report.main(["--run-dir", str(run_dir), "--out", str(tmp_path / "rerender")]) == 0
    assert sorted(p.name for p in run_dir.iterdir()) == before  # read only
    built = json.loads((tmp_path / "rerender" / "report.json").read_text(encoding="utf-8"))
    assert built["label"] == "old run 1" and built["run"]["harness"]["sha1"] == "h"
    assert built["keyframes"]["count"] == 4 and (tmp_path / "rerender" / "keyframes.json").exists()
    assert built["verdict"]["stop_to_settled_min"] == pytest.approx((600.0 - 100.002) / 60, abs=0.01)


# A proof run's comparability key, as `_fake_run` writes it (review C24 HIGH-2).
FAKE_SWITCHES = {"TOWER_WORLD_SOLVE": "true", "TOWER_WORLD_SOLVE_CONSENSUS": "3"}
FAKE_CODE = {"py_fingerprint": "bd4f37d4c9f16a21", "tower_dir": "C:\\code\\a4afea1\\tower"}
FAKE_HARNESS = {"sha1": "h" * 40, "files_sha1": {"world_live_replay.py": "a" * 40,
                                                 "world_live_replay_run.py": "b" * 40,
                                                 "world_live_replay_report.py": "c" * 40}}
FAKE_JOURNAL = {CAP: {"capture.json": "1" * 64, "frames.jsonl": "2" * 64}}


def _fake_run(directory, *, photos, lag_p95, sequence, horizons=(52,), fidelity="PASS", environment="PASS",
              version=None, run_started=None, client_started=None, switches=None, code=None, harness=None,
              journal=None, not_proof=None, store_photos=None, speed=1.0, tower_side_only=False):
    """`version` None: a verdict rendered before the bar was versioned (C22-F5). `run_started` /
    `client_started`: the run's recorded start, as the report keeps run.json and the client record.
    By default a proof-eligible run whose comparability key is `FAKE_*` (C24 HIGH-2/3); `photos` is
    the W0 timing, Stop to `phone_photos_at` (C24 HIGH-1)."""
    directory.mkdir(parents=True)
    sha = report.hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode()).hexdigest()
    verdict = {"result": fidelity, **({"version": version} if version is not None else {}),
               **({"tower_side_only": True} if tower_side_only else {})}
    run = {"switches": FAKE_SWITCHES if switches is None else switches, "code": code or FAKE_CODE,
           "harness": harness or FAKE_HARNESS, "intrinsics_copied": ["360x640.json"],
           **({"started_at": run_started} if run_started is not None else {})}
    client = {"walk": [{"capture_id": CAP}], "speed": speed, "first_seconds": None, "after_stop": "stay",
              "schedule": {"frames": 4005, "captures": 1, "stop_at_s": 335.861, "ends_with": "stream_stop",
                           "reconnects": 0},
              "stream": {"frames_sent": 4005}, "live_tower_watch": {"states_seen": ["idle"]},
              **({"started_at": client_started} if client_started is not None else {})}
    (directory / "report.json").write_text(json.dumps({
        **({"replay_fidelity": verdict} if fidelity is not None else {}),
        "run": run, "client": client,
        "proof": {"proof": not not_proof, "not_proof_reasons": [not_proof] if not_proof else []},
        "tower_side_pacing": {"computable": True, "source_sha256": journal or FAKE_JOURNAL},
        "label": directory.name,
        "verdict": {"basis": "phone", "stop_to_room_with_photos_min": photos, "stop_to_phone_photos_min": photos,
                    "stop_to_store_photos_min": round(photos - 0.25, 2) if store_photos is None else store_photos,
                    "stop_to_settled_min": photos + 7},
        "tower_walk": {"totals": {"frames_received": 4005, "tx_seq_gap_total": 0},
                       "frames_observed": 4005, "rebuilds": {"count": 201, "seconds": {"p95": 1.1, "max": 1.9}},
                       "background_solves": [{"keyframes": k} for k in horizons]},
        "keyframes": {"sha256": sha, "observe_lag_s": {"p50": 2.0, "p95": lag_p95, "p99": 7.0, "max": 7.2}},
        "waterfall": [{"n": "3", "minutes": photos - 13, "child": False}],
        "live_safety": {"result": "PASS", "environment": {"result": environment}}}), encoding="utf-8")
    (directory / "keyframes.json").write_text(json.dumps({"sha256": sha, "sequence": sequence}), encoding="utf-8")


def test_compare_gives_the_baseline_spread_and_flags_a_candidate_outside_it(tmp_path):
    same = [[1, 0], [10, 0], [54, 1]]
    for index, (photos, lag) in enumerate([(47.0, 6.7), (47.5, 6.9), (46.8, 6.8)]):
        _fake_run(tmp_path / f"old{index}", photos=photos, lag_p95=lag, sequence=same)
    _fake_run(tmp_path / "new0", photos=40.0, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "new1", photos=41.0, lag_p95=9.5, sequence=[[1, 0], [10, 0], [55, 1]], horizons=(60,))
    _fake_run(tmp_path / "new2", photos=40.5, lag_p95=6.75, sequence=same)
    out = tmp_path / "cmp"
    assert report.main(["--out", str(out), "--compare", *(str(tmp_path / f"old{i}") for i in range(3)),
                        "--candidate", *(str(tmp_path / f"new{i}") for i in range(3))]) == 0
    result = json.loads((out / "compare.json").read_text(encoding="utf-8"))
    assert result["enough_runs"] is True
    metrics = {m["metric"]: m for m in result["metrics"]}
    photos = metrics["stop_to_phone_photos_min"]
    assert (photos["min"], photos["max"], photos["spread"]) == (46.8, 47.5, 0.7)
    assert photos["outside"] == [40.0, 41.0, 40.5]
    assert metrics["keyframe_accept_lag_s.p95"]["outside"] == [9.5]
    assert metrics["row 3 min"]["baseline"] == pytest.approx([34.0, 34.5, 33.8])
    candidates = result["keyframes"]["candidate"]
    assert [c["identical"] for c in candidates] == [True, False, True]
    assert candidates[1]["first_difference"] == 2 and candidates[1]["horizons_identical"] is False
    assert all(item["identical"] for item in result["keyframes"]["baseline"])
    assert "| stop_to_phone_photos_min **(W0 timing)** |" in (out / "COMPARE.md").read_text(encoding="utf-8")


def test_compare_says_when_there_are_too_few_runs(tmp_path):
    _fake_run(tmp_path / "old0", photos=47.0, lag_p95=6.7, sequence=[[1, 0]])
    result = report.compare_runs([tmp_path / "old0"])
    assert result["enough_runs"] is False
    assert "not a noise estimate" in report.render_compare(result)
    with pytest.raises(SystemExit, match="no report.json"):
        report.compare_runs([tmp_path / "missing"])


# -- C22-F2: the review's non-blocking follow-ups ---------------------------------------------

# Item 1: the Tower-side pacing row (review C22 H2, "README fidelity").


def _pacing_fixture(tmp_path, *, speed=1.0):
    """A recorded capture A (the source) and the test Tower's re-recording of it
    as C1 in a data root: frame 1 +4 ms, 3 -2 ms, 5 +60 ms late, 7 +20 ms, and a
    frame 9 the source never had."""
    _capture(tmp_path / "src", A, started=1000.0,
             frames=[(1, 1001.0), (3, 1001.1), (5, 1001.25), (7, 1001.3)], ended=1002.0)
    scale = 1.0 / speed
    _capture(tmp_path / "data", C1, started=5000.0, frames=[
        (1, 5000.0 + 1.0 * scale + 0.004), (3, 5000.0 + 1.1 * scale - 0.002),
        (5, 5000.0 + 1.25 * scale + 0.060), (7, 5000.0 + 1.3 * scale + 0.020), (9, 5000.0 + 1.9 * scale)],
             ended=5002.0)
    return {"walk": [{"capture_id": A}], "tower_captures": [C1], "speed": speed}


@pytest.mark.parametrize("speed", [1.0, 2.0])
def test_the_tower_side_pacing_joins_the_re_recording_to_the_source_on_wire_seq(tmp_path, speed):
    client = {**_pacing_fixture(tmp_path, speed=speed), "first_seconds": 1.95}
    pacing = report.tower_side_pacing(client=client, data_root=tmp_path / "data",
                                      capture_root=tmp_path / "src" / "captures")
    assert pacing["computable"] is True
    assert (pacing["matched"], pacing["replay_only"], pacing["source_only"]) == (4, 1, 0)
    offset = pacing["offset_error_ms"]  # signed, late positive: -2, 4, 20, 60
    assert offset["count"] == 4
    assert offset["min"] == pytest.approx(-2.0, abs=0.01) and offset["max"] == pytest.approx(60.0, abs=0.01)
    assert offset["p50"] == pytest.approx(20.0, abs=0.01) and offset["p95"] == pytest.approx(60.0, abs=0.01)
    assert (pacing["late_over_50ms"], pacing["early_over_50ms"]) == (1, 0)
    gaps = pacing["inter_arrival_s"]
    assert gaps["recorded"]["count"] == 3 and gaps["recorded"]["max"] == pytest.approx(0.15 / speed, abs=1e-3)
    assert gaps["replayed"]["max"] == pytest.approx(0.15 / speed + 0.062, abs=1e-3)
    assert pacing["inter_arrival_error_ms"]["min"] == pytest.approx(-40.0, abs=0.01)
    row = next(r for r in report.live_safety(
        client=client, timeline={}, session={}, keyframes=None, surfaces={}, rebuild_seconds=[], cadence=[],
        pacing=pacing)["rows"] if r["check"].startswith("Tower-side pacing"))
    assert row["result"] == "INFO" and "4 joined (1 replay-only, 0 source-only, the walk cut at first_seconds 1.95)" \
        in row["value"] and "beyond 50 ms: 1 late, 0 early" in row["value"]


def test_the_pacing_row_is_rendered_from_a_kept_data_root_with_a_data_root_override(tmp_path):
    """Re-render mode: run.json names a data root that has since moved; the
    report's --data-root reads the kept one. The run dir is never written."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log = run_dir / "tower-8031-x.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _capture(tmp_path / "src", A, started=1000.0, frames=[(1, 1001.0), (3, 1001.1)], ended=1002.0)
    kept = tmp_path / "kept"
    _capture(kept, CAP, started=5000.0, frames=[(1, 5001.010), (3, 5001.105)], ended=5002.0)
    _world(kept / "world_builder", BASE, appearance_end=BASE + 700)
    (run_dir / "run.json").write_text(json.dumps({"err_log": str(log), "data_root": str(tmp_path / "moved")}),
                                      encoding="utf-8")
    (run_dir / "client.json").write_text(json.dumps({
        "tower_captures": [CAP], "speed": 1.0, "capture_root": str(tmp_path / "src" / "captures"),
        "walk": [{"capture_id": A, "frames": 2, "recorded_seconds": 2.0, "recorded_fps": 1.0,
                  "end_reason": "stop"}]}), encoding="utf-8")
    before = sorted(p.name for p in run_dir.iterdir())
    assert report.main(["--run-dir", str(run_dir), "--out", str(tmp_path / "gone")]) == 0
    stale = json.loads((tmp_path / "gone" / "report.json").read_text(encoding="utf-8"))
    row = next(r for r in stale["live_safety"]["rows"] if r["check"].startswith("Tower-side pacing"))
    assert row["result"] == "n/a" and "no journal" in row["value"]
    assert report.main(["--run-dir", str(run_dir), "--out", str(tmp_path / "kept-out"),
                        "--data-root", str(kept)]) == 0
    assert sorted(p.name for p in run_dir.iterdir()) == before  # read only
    built = json.loads((tmp_path / "kept-out" / "report.json").read_text(encoding="utf-8"))
    rows = {r["check"]: r for r in built["live_safety"]["rows"]}
    pacing = rows[report.PACING_ROW]
    assert pacing["result"] == "INFO" and pacing["value"].startswith("2 joined (0 replay-only, 0 source-only)")
    assert rows[report.INTERVAL_ROW]["result"] == "INFO"
    assert built["tower_side_pacing"]["offset_error_ms"]["max"] == pytest.approx(10.0, abs=0.01)
    assert built["keyframes"]["count"] == 4  # the world root followed --data-root
    metrics = report.comparable_metrics(built)
    assert metrics["recorder_stamp_offset_error_ms.p95"] == pytest.approx(10.0, abs=0.01)
    assert "Tower-side pacing (the recorder's post-reply stamps, not socket arrival): `" in \
        (tmp_path / "kept-out" / "REPORT.md").read_text(encoding="utf-8")


def test_the_client_records_where_it_read_the_source_captures(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "live_tower_state", lambda url=None, timeout=None: "down")
    monkeypatch.setattr(replay, "http_json", lambda *a, **k: (None, None))
    record = asyncio.run(replay.run_replay(_refusal_options(tmp_path)))
    assert record["capture_root"] == str(tmp_path / "captures")


# Item 2: phone_photos_at, the background solves' landing, terminated at Stop.


def _told(t, **word):
    payload = {"lifecycle": {"state": "finalizing"}}
    if word:
        payload["lifecycle"]["photographic"] = word
    return t, {"type": "cartridge_result", "cartridge": "world_builder", "payload": payload}


PUSHES_AFTER_STOP = [
    _told(BASE + 50.0),  # during the walk: no photographic block
    _told(BASE + 500.0, state="running", stage="surface"),
    _told(BASE + 600.0, state="owed", stage="appearance"),
    _told(BASE + 601.0, state="running", stage="appearance"),
    # The store wrote the room's appearance `ok` at BASE + 700; the phone
    # heard the word move on to an area at 705.
    _told(BASE + 705.0, state="owed", stage="surface", scope="area"),
    _told(BASE + 706.0, state="owed", stage="surface", scope="area"),
    _told(BASE + 800.0, state="complete", stage="appearance"),
]


def _phone_view(pushes):
    clock = {"t": 0.0}
    view = replay.PhoneView(clock=lambda: clock["t"])
    for t, message in pushes:
        clock["t"] = t
        view.update(message)
    return view


def test_the_phone_view_keeps_the_photographic_word_with_its_receive_time():
    view = _phone_view(PUSHES_AFTER_STOP)
    assert [(w["t"], w["state"], w["stage"], w["scope"]) for w in view.photographic] == [
        (BASE + 500.0, "running", "surface", None), (BASE + 600.0, "owed", "appearance", None),
        (BASE + 601.0, "running", "appearance", None), (BASE + 705.0, "owed", "surface", "area"),
        (BASE + 800.0, "complete", "appearance", None)]
    assert view.photographic[3]["scope_present"] is True and view.photographic[0]["scope_present"] is False


def test_phone_photos_at_is_the_clients_receive_time_not_the_stores(tmp_path):
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _world(tmp_path / "root", BASE, appearance_end=BASE + 700.0)
    view = _phone_view(PUSHES_AFTER_STOP)
    built = report.build_report(tower_log=log, world_root=tmp_path / "root", client={
        "tower_captures": [CAP], "phone_view": {"pushes": view.pushes, "transitions": view.transitions,
                                                "photographic": view.photographic}})
    photos = built["phone_photos"]
    assert photos["phone_photos_at"] == BASE + 705.0 and "scope: area" in photos["how"]
    assert photos["store_to_phone_s"] == pytest.approx(5.0)
    assert photos["photographic_complete_at"] == BASE + 800.0
    assert built["milestones"]["phone_photos_at"]["t"] == BASE + 705.0
    assert built["milestones"]["room_appearance_ok"]["t"] == BASE + 700.0
    assert built["verdict"]["stop_to_phone_photos_min"] == pytest.approx((705.0 - 100.002) / 60, abs=0.01)
    assert report.comparable_metrics(built)["stop_to_phone_photos_min"] == built["verdict"]["stop_to_phone_photos_min"]
    # C24 HIGH-1: the W0 verdict is the CLIENT's receipt. The store wrote the room's appearance
    # at +10.00 min (a PASS on the store's clock); the client was told at +10.08: FAIL.
    verdict = built["verdict"]
    assert verdict["basis"] == "phone" and verdict["measure"] == report.BASIS_PHONE
    assert verdict["stop_to_room_with_photos_min"] == verdict["stop_to_phone_photos_min"] == 10.08
    assert verdict["stop_to_store_photos_min"] == 10.0 and verdict["result"] == "FAIL"
    metrics = report.comparable_metrics(built)
    assert (metrics["stop_to_phone_photos_min"], metrics["stop_to_store_photos_min"]) == (10.08, 10.0)
    assert "stop_to_room_with_photos_min" not in metrics
    markdown = report.render_markdown(built)
    assert ("1. **Stop to room with photos, as the replay client received it (`phone_photos_at`): 10.08 min** "
            "against the 10-min hard maximum: **FAIL**.") in markdown
    assert "INFO: the store wrote the room appearance `ok` at +10.0 min (not the W0 measure)" in markdown


def test_phone_photos_at_is_inferred_for_a_client_that_kept_no_scope(tmp_path):
    """A run made before this change kept only the state-like leaves."""
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _world(tmp_path / "root", BASE, appearance_end=BASE + 700.0)
    view = _phone_view(PUSHES_AFTER_STOP)
    built = report.build_report(tower_log=log, world_root=tmp_path / "root", client={
        "tower_captures": [CAP], "phone_view": {"pushes": view.pushes, "transitions": view.transitions}})
    photos = built["phone_photos"]
    assert photos["scope_recorded"] is False
    assert photos["phone_photos_at"] == BASE + 705.0 and photos["how"].startswith("INFERRED")
    assert photos["photographic_complete_at"] == BASE + 800.0
    # C24 HIGH-1: the verdict rests on that inferred time, and its line says so.
    assert built["verdict"]["phone_photos_how"].startswith("INFERRED")
    assert "**The client's time is INFERRED**" in report.render_markdown(built)
    scoped = report.build_report(tower_log=log, world_root=tmp_path / "root", client={
        "tower_captures": [CAP], "phone_view": {"pushes": view.pushes, "transitions": view.transitions,
                                                "photographic": view.photographic}})
    assert "INFERRED" not in report.render_markdown(scoped).split("## 2.")[0]


def test_phone_photos_at_needs_the_rooms_appearance_ok_and_a_push_after_stop():
    moved_on = [{"t": 50.0, "state": "complete", "stage": "appearance", "scope": None, "scope_present": True},
                {"t": 150.0, "state": "owed", "stage": "surface", "scope": "area", "scope_present": True}]
    failed = report.phone_photos({"photographic": moved_on}, 100.0, {"state": "failed", "updated_at": 140.0})
    assert failed["phone_photos_at"] is None and "failed" in failed["why"]
    ok = report.phone_photos({"photographic": moved_on}, 100.0, {"state": "ok", "updated_at": 140.0})
    assert ok["phone_photos_at"] == 150.0  # the `complete` at 50 was before Stop: another world's
    assert report.phone_photos({}, 100.0, None)["phone_photos_at"] is None


def _solves_log(base, *, terminated):
    """Three background solves: 2 launched at 1's landing rebuild, a live surface
    right after it (one loop iteration: rebuild, solve launch, surface); 3 is
    still running at Stop and is terminated after the wait (True), terminated
    at once by a hard stop ("stop"), or waited for (False)."""
    wb, ws = "tower.world_build_session", "tower.routes.ws"
    lines = [
        _line(base, ws, "[Tower][Session] stream_start: measurement window opened"),
        _line(base + 0.006, ws, f"[Tower][Capture] recording started: {CAP}"),
        _line(base + 0.013, "tower.capture_workers",
              f"[Tower][Worker] started world-build-session pid 21204 for capture {CAP}: python x"),
        _line(base + 1.7, wb, f"[Tower][WorldBuilder] session {S} in world {W}: source=live-capture "
                              f"capture={CAP} root=x observed=360x640"),
        _line(base + 20.0, wb, "[Tower][WorldBuilder] background solve 1 launched at 52 keyframes (pid 100)"),
        _line(base + 30.0, wb, "[Tower][WorldBuilder] rebuild 4: 70 keyframes -> 60 positioned poses, "
                               "900 points, 2 segments in 0.30s"),
        _line(base + 35.0, wb, "[Tower][WorldBuilder] rebuild 5: 110 keyframes -> 100 positioned poses, "
                               "1500 points, 2 segments in 0.40s"),
        _line(base + 35.004, wb, "[Tower][WorldBuilder] background solve 2 launched at 110 keyframes (pid 200)"),
        _line(base + 35.010, wb, "[Tower][WorldBuilder] live surface 1 launched (pid 900)"),
        _line(base + 69.9, wb, "[Tower][WorldBuilder] rebuild 9: 180 keyframes -> 170 positioned poses, "
                               "2500 points, 3 segments in 0.50s"),
        _line(base + 70.0, wb, "[Tower][WorldBuilder] background solve 3 launched at 180 keyframes (pid 300)"),
        _line(base + 100.0, ws, _summary(1200, "stream_stop")),
        _line(base + 100.002, ws, "[Tower][Capture] recording stopped (stop): 1200 frames, 900 bytes"),
    ]
    if terminated == "stop":
        lines.append(_line(base + 101.0, wb, "[Tower][WorldBuilder] background solve pid 300 terminated: "
                                             "a stop was requested", "WARNING"))
    elif terminated:
        lines.append(_line(base + 224.0, wb, "[Tower][WorldBuilder] background solve pid 300 still running after "
                                             "120.0s; terminating it so the final solve owns the workspace",
                           "WARNING"))
    lines.append(_line(base + 224.1, wb, "[Tower][WorldBuilder] final global solve launched (pid 34464)"))
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("terminated", [True, False])
def test_each_background_solve_has_a_landing_and_a_terminated_at_stop_flag(tmp_path, terminated):
    log = tmp_path / "tower.err.log"
    log.write_text(_solves_log(BASE, terminated=terminated), encoding="utf-8")
    solves = report.walk_timeline(report.scan_log(log), CAP)["background_solves"]
    first, second, third = solves
    assert first["landed_by"]["t"] == pytest.approx(BASE + 35.004, abs=0.002)
    assert first["landed_by"]["via"] == "background solve 2 launch"
    assert (first["landed_by"]["rebuild_n"], first["landed_by"]["rebuild_line"]) == (5, 7)
    assert first["horizon_s"] == pytest.approx(15.0, abs=0.01) and first["terminated_at_stop"] is False
    # Round 3 L-a: surface 1 is logged after solve 2's launch line, but it was
    # launched by rebuild 5, solve 1's landing: it is solve 1's, by log line.
    assert first["live_surface_between"]["n"] == 1
    assert (first["live_surface_between"]["rebuild_n"], first["live_surface_between"]["rebuild_line"]) == (5, 7)
    assert second["live_surface_between"] is None and second["horizon_s"] == pytest.approx(35.0, abs=0.01)
    assert third["terminated_at_stop"] is terminated
    if terminated:
        assert third["landed_by"] is None and third["horizon_s"] is None and third["terminated_line"] == 14
    else:
        assert third["landed_by"]["via"].startswith("the final solve's launch")
        assert third["horizon_s"] == pytest.approx(154.1, abs=0.01)
    built = report.build_report(tower_log=log, client={"tower_captures": [CAP]})
    row = next(r for r in built["live_safety"]["rows"] if r["check"].startswith("background-solve horizons"))
    assert row["result"] == "INFO" and row["value"].startswith("[52, 110, 180]; landed by [15.0, 35.0, ")
    assert row["value"].endswith("terminated at Stop: solve 3" if terminated else "terminated at Stop: none")
    metrics = report.comparable_metrics(built)
    assert metrics["bg_solves_terminated_at_stop"] == (1 if terminated else 0)
    assert metrics["bg_solve_landed_by_s.max"] == pytest.approx(35.0 if terminated else 154.1, abs=0.01)
    markdown = report.render_markdown(built)
    assert ("TERMINATED at Stop" in markdown) is terminated
    solve_1 = next(line for line in markdown.splitlines() if line.startswith("- background solve 1 "))
    assert "live surface 1 launched at" in solve_1 and "by rebuild 5 (L7), attributed by log line" in solve_1
    solve_2 = next(line for line in markdown.splitlines() if line.startswith("- background solve 2 "))
    assert "live surface" not in solve_2


# Item 3: :8000 during the run is the environment's verdict, not the code's.


def _clean_client(**extra):
    return {"tower_captures": [CAP], "schedule": {"frames": 1200},
            "stream": {"frames_sent": 1200, "unanswered": 0, "frame_errors": {}}, **extra}


def test_an_isolated_unknown_from_8000_is_tolerated_and_shown(tmp_path):
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _world(tmp_path / "root", BASE, appearance_end=BASE + 700)
    watch = {"states_seen": ["idle", "unknown"], "history": [
        {"t": 1.0, "state": "idle"}, {"t": 2.0, "state": "unknown"}, {"t": 3.0, "state": "idle"},
        {"t": 4.0, "state": "unknown"}, {"t": 5.0, "state": "idle"}]}
    built = report.build_report(tower_log=log, world_root=tmp_path / "root",
                                client=_clean_client(live_tower_watch=watch))
    environment = built["live_safety"]["environment"]
    row = environment["rows"][0]
    assert row["check"] == ":8000 during the run" and row["result"] == "PASS"
    assert "2 unknown answer(s), each isolated: tolerated" in row["value"]
    assert environment["result"] == "PASS" and built["live_safety"]["result"] == "PASS"
    markdown = report.render_markdown(built)
    assert "**Environment (the `:8000` guard): PASS.**" in markdown and "each isolated: tolerated" in markdown


@pytest.mark.parametrize("watch, aborted, why", [
    ({"states_seen": ["busy", "idle"], "busy_since": BASE + 300,
      "history": [{"t": 1.0, "state": "idle"}, {"t": 2.0, "state": "busy"}]}, None, "busy from"),
    ({"states_seen": ["idle", "unknown"], "history": [{"t": 1.0, "state": "idle"}, {"t": 2.0, "state": "unknown"}]},
     {"reason": ":8000 gave no usable /health answer 2 times in a row (timeout or error); failing closed"},
     "two in a row aborted the run"),
])
def test_a_contended_or_failed_closed_8000_fails_the_environment_not_the_code(tmp_path, watch, aborted, why):
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _world(tmp_path / "root", BASE, appearance_end=BASE + 700)
    client = _clean_client(live_tower_watch=watch, **({"aborted": aborted} if aborted else {}))
    built = report.build_report(tower_log=log, world_root=tmp_path / "root", client=client)
    environment = built["live_safety"]["environment"]
    assert environment["result"] == "FAIL" and why in environment["rows"][0]["value"]
    assert built["live_safety"]["result"] == "PASS"  # the code under test is not blamed for the machine


def test_the_runners_startup_watch_is_an_environment_row(tmp_path):
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    run = {"live_tower_watch_startup": {"states_seen": ["idle", "unknown"], "history": [
        {"t": 1.0, "state": "idle"}, {"t": 2.0, "state": "unknown"}, {"t": 3.0, "state": "idle"}]}}
    watch = {"states_seen": ["idle"], "history": [{"t": 4.0, "state": "idle"}]}
    built = report.build_report(tower_log=log, client=_clean_client(live_tower_watch=watch), run=run)
    rows = {r["check"]: r for r in built["live_safety"]["environment"]["rows"]}
    assert rows[":8000 during the run"]["result"] == "PASS"
    assert rows[":8000 during the test Tower's startup"]["result"] == "PASS"
    assert built["live_safety"]["environment"]["result"] == "PASS"


# Item 4: --compare flags a candidate value that is missing.


def test_compare_flags_a_candidate_metric_missing_where_the_baseline_has_it(tmp_path):
    same = [[1, 0], [10, 0]]
    for index, photos in enumerate([47.0, 47.5, 46.8]):
        _fake_run(tmp_path / f"old{index}", photos=photos, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "new0", photos=40.0, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "new1", photos=40.0, lag_p95=6.8, sequence=same)
    # new1 never settled: no photos, and its report has no observe lag at all.
    path = tmp_path / "new1" / "report.json"
    never = json.loads(path.read_text(encoding="utf-8"))
    never["verdict"]["stop_to_phone_photos_min"] = None
    never["keyframes"].pop("observe_lag_s")
    path.write_text(json.dumps(never), encoding="utf-8")
    out = tmp_path / "cmp"
    assert report.main(["--out", str(out), "--compare", *(str(tmp_path / f"old{i}") for i in range(3)),
                        "--candidate", str(tmp_path / "new0"), str(tmp_path / "new1")]) == 0
    result = json.loads((out / "compare.json").read_text(encoding="utf-8"))
    metrics = {m["metric"]: m for m in result["metrics"]}
    new1 = str(tmp_path / "new1")
    assert metrics["stop_to_phone_photos_min"]["missing"] == [new1]
    assert metrics["stop_to_phone_photos_min"]["outside"] == [40.0]  # new0's, judged as before
    assert metrics["keyframe_accept_lag_s.p95"]["missing"] == [new1]
    assert "stop_to_phone_photos_min" in result["missing"][new1]
    assert str(tmp_path / "new0") not in result["missing"]
    assert result["candidates_complete"] is False
    markdown = (out / "COMPARE.md").read_text(encoding="utf-8")
    assert "**MISSING**" in markdown and "Not passing" in markdown


# Item 5: the abort edge -- a send caught in the drain when the Tower is killed.


def test_a_send_that_raises_after_the_abort_is_the_abort_not_an_error(tmp_path, monkeypatch):
    import threading

    import websockets

    _capture(tmp_path, A, started=0.0, frames=[(i, 0.05 * i) for i in range(1, 41)], ended=2.1)
    tower = FakeTower([C1])
    blocked, killed = threading.Event(), threading.Event()
    real_send = replay.TowerSocket.send_frame

    async def send_frame(self, frame, text, size):
        if tower.frames >= 5:
            blocked.set()  # a send waiting in the drain...
            while not killed.is_set():
                await asyncio.sleep(0.005)
            raise websockets.exceptions.ConnectionClosedError(None, None)  # ...when the Tower died
        await real_send(self, frame, text, size)

    monkeypatch.setattr(replay.TowerSocket, "send_frame", send_frame)
    monkeypatch.setattr(replay, "live_tower_state",
                        lambda url=None, timeout=None: "recording" if blocked.is_set() else "idle")

    def on_abort():
        time.sleep(0.1)
        killed.set()

    record = _serve_and_replay(monkeypatch, tower, lambda port: replay.ReplayOptions(
        port=port, captures=[A], capture_root=tmp_path / "captures", out=tmp_path / "out",
        settle_timeout_min=0.5, poll_seconds=0.05, sample_seconds=60.0, session_lead=0.0,
        live_guard=True, live_guard_every=0.02, phone_fetches=False, on_abort=on_abort))
    assert record["outcome"] == "aborted" and replay.exit_code(record) == replay.EXIT_ABORTED
    assert "ConnectionClosedError" in record["aborted"]["stream_error"]
    assert "live walk" in record["aborted"]["reason"]
    saved = json.loads((tmp_path / "out" / "client.json").read_text(encoding="utf-8"))
    assert saved["outcome"] == "aborted" and saved["stream"]["frames_sent"] == record["stream"]["frames_sent"]
    # Round 3 L-c: the swallowed exception keeps its traceback, in the record
    # (a string) and in client.log.
    trace = saved["aborted"]["stream_traceback"]
    assert isinstance(trace, str) and trace.startswith("Traceback (most recent call last)")
    assert "ConnectionClosedError" in trace and "in send_frame" in trace
    log = (tmp_path / "out" / "client.log").read_text(encoding="utf-8")
    assert "the stream ended under the abort" in log and "Traceback (most recent call last)" in log


def test_a_send_that_raises_without_an_abort_is_still_an_error(tmp_path, monkeypatch):
    _capture(tmp_path, A, started=0.0, frames=[(1, 0.05), (2, 0.1)], ended=0.2)
    tower = FakeTower([C1])

    async def send_frame(self, frame, text, size):
        raise RuntimeError("a harness fault")

    monkeypatch.setattr(replay.TowerSocket, "send_frame", send_frame)
    with pytest.raises(RuntimeError, match="a harness fault"):
        _serve_and_replay(monkeypatch, tower, lambda port: replay.ReplayOptions(
            port=port, captures=[A], capture_root=tmp_path / "captures", out=tmp_path / "out",
            settle_timeout_min=0.5, poll_seconds=0.05, sample_seconds=60.0, session_lead=0.0,
            live_guard=False, phone_fetches=False))


# -- the runner ------------------------------------------------------------------------------


def test_the_tower_environment_drops_the_shells_switches_and_owns_the_roots(tmp_path):
    env = runner.tower_environment(
        {"PATH": "p", "TOWER_WORLD_SOLVE": "false", "tower_dev_mode": "x", "PYTHONPATH": "elsewhere"},
        {"TOWER_WORLD_SOLVE": "true", "TOWER_WORLD_ROOT": "C:/live", "TOWER_WORLD_STAGE_TIMING": "on"},
        data_root=tmp_path / "root", tower_dir=tmp_path / "code" / "tower", port=8031)
    assert env["PATH"] == "p" and "tower_dev_mode" not in env
    assert env["TOWER_WORLD_SOLVE"] == "true" and env["TOWER_WORLD_STAGE_TIMING"] == "on"
    assert env["TOWER_WORLD_ROOT"] == str(tmp_path / "root" / "world_builder")
    assert env["TOWER_CAPTURE_ROOT"] == str(tmp_path / "root")
    assert env["TOWER_HOST"] == "127.0.0.1" and env["TOWER_PORT"] == "8031"
    assert env["PYTHONPATH"] == str(tmp_path / "code" / "tower")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_switch_files_take_comments_a_bom_and_the_last_value(tmp_path):
    path = tmp_path / "walk.env"
    path.write_bytes("\ufeff# a comment\nTOWER_A=1\n\nTOWER_B='x y'\nTOWER_A=2\nnot a pair\n".encode("utf-8"))
    assert runner.read_switch_file(path) == {"TOWER_A": "2", "TOWER_B": "x y"}


def test_the_code_tree_is_found_from_a_repo_root_or_its_tower_dir(tmp_path):
    (tmp_path / "repo" / "tower" / "tower").mkdir(parents=True)
    (tmp_path / "repo" / "tower" / "tower" / "main.py").write_text("", encoding="utf-8")
    assert runner.resolve_code_tree(tmp_path / "repo") == (tmp_path / "repo" / "tower").resolve()
    assert runner.resolve_code_tree(tmp_path / "repo" / "tower") == (tmp_path / "repo" / "tower").resolve()
    with pytest.raises(SystemExit):
        runner.resolve_code_tree(tmp_path)


def test_inside_is_a_path_test_not_a_string_prefix(tmp_path):
    assert runner._inside(tmp_path / "a" / "b", tmp_path / "a")
    assert not runner._inside(tmp_path / "ab", tmp_path / "a")


def test_code_identity_takes_no_optional_git_locks(tmp_path, monkeypatch):
    """L3: a plain `git status` can take another lane's index.lock."""
    (tmp_path / "tower" / "tower").mkdir(parents=True)
    calls = []

    class Done:
        returncode, stdout = 0, str(tmp_path / "tower")

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return Done()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    identity = runner.code_identity(tmp_path / "tower")
    assert [argv[1] for argv in calls] == ["--no-optional-locks"] * 3
    assert calls[-1][-3:] == ["status", "--porcelain", "--untracked-files=no"]
    assert "git_dirty" in identity


def test_the_runner_refuses_a_non_empty_data_root_before_starting_anything(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "leftover").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="not empty"):
        runner.main(["--capture", A, "--code", str(tmp_path), "--port", "8031",
                     "--data-root", str(root), "--out", str(tmp_path / "out")])


def test_the_runner_refuses_port_8000(tmp_path):
    with pytest.raises(SystemExit, match="live Tower"):
        runner.main(["--capture", A, "--code", str(tmp_path), "--port", "8000",
                     "--data-root", str(tmp_path / "root"), "--out", str(tmp_path / "out")])


@pytest.mark.parametrize("flag", ["--data-root", "--out"])
def test_the_runner_refuses_a_drive_root_for_what_it_writes(tmp_path, flag):
    """T6: the two flags that write go through artifact_root_arg."""
    drive = Path(tmp_path.anchor) / "c22-drive-root-test"
    paths = {"--data-root": str(tmp_path / "root"), "--out": str(tmp_path / "out"), flag: str(drive)}
    with pytest.raises(SystemExit) as refused:
        runner.main(["--capture", A, "--code", str(tmp_path), "--port", "8031",
                     "--data-root", paths["--data-root"], "--out", paths["--out"]])
    assert refused.value.code == 2
    assert not drive.exists()


def test_the_client_and_the_report_refuse_a_drive_root_out(tmp_path):
    drive = str(Path(tmp_path.anchor) / "c22-drive-root-test")
    with pytest.raises(SystemExit) as client:
        replay.main(["--capture", A, "--port", "8031", "--tower-pid", "1", "--out", drive])
    with pytest.raises(SystemExit) as rendered:
        report.main(["--out", drive, "--tower-log", str(tmp_path / "x.log")])
    assert client.value.code == 2 and rendered.value.code == 2


class _FakeProcess:
    pid = 4242
    returncode = None

    def poll(self):
        return None


class _FakeJob:
    closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    """The runner with every outside effect replaced: no process, no job, no
    network, no git. `state` steers it; `calls` records what it did."""
    code = tmp_path / "code" / "tower"
    (code / "tower").mkdir(parents=True)
    (code / "tower" / "main.py").write_text("", encoding="utf-8")
    state = {"live": ["idle"], "health": (200, IDLE_HEALTH), "owners": {4242}, "job": _FakeJob(),
             "replay": lambda options: {"outcome": "settled", "tower_captures": []}}
    calls = {"spawn": [], "terminate": [], "options": []}

    def live(url=None, timeout=None):
        return state["live"].pop(0) if len(state["live"]) > 1 else state["live"][0]

    def spawn(command, **kwargs):
        calls["spawn"].append(command)
        return _FakeProcess()

    async def fake_run_replay(options):
        calls["options"].append(options)
        return state["replay"](options)

    monkeypatch.setattr(runner, "live_tower_state", live)
    monkeypatch.setattr(runner, "port_in_use", lambda port: False)
    monkeypatch.setattr(runner, "probe_import", lambda tower_dir, env: str(tower_dir / "tower" / "__init__.py"))
    monkeypatch.setattr(runner, "spawn_tower", spawn)
    monkeypatch.setattr(runner, "assign_to_job", lambda process: state["job"])
    monkeypatch.setattr(runner, "terminate_tree",
                        lambda process, job=None, timeout=0, hard=False: calls["terminate"].append(process))
    monkeypatch.setattr(runner, "http_json", lambda *a, **k: state["health"])
    monkeypatch.setattr(runner, "listener_pids", lambda port: state["owners"])
    monkeypatch.setattr(runner, "run_replay", fake_run_replay)
    monkeypatch.setattr(runner, "_leftovers", lambda port, markers: [])
    monkeypatch.setattr(runner, "harness_identity", lambda: {"sha1": "harness"})
    monkeypatch.setattr(runner, "HEALTH_POLL_S", 0.01)
    monkeypatch.setattr(runner, "IDLE_POLL_S", 0.0)
    monkeypatch.setattr(runner, "AFTER_STOP_S", 0.0)
    argv = ["--capture", A, "--code", str(code), "--port", "8031", "--data-root", str(tmp_path / "data"),
            "--out", str(tmp_path / "out"), "--intrinsics-from", str(tmp_path / "no-intrinsics"),
            "--health-timeout", "0.2"]
    return {"state": state, "calls": calls, "argv": argv, "out": tmp_path / "out"}


def _run_json(lifecycle):
    return json.loads((lifecycle["out"] / "run.json").read_text(encoding="utf-8"))


def test_the_runner_settles_stops_the_tower_and_hands_the_client_a_kill_switch(lifecycle):
    assert runner.main(lifecycle["argv"]) == 0
    calls = lifecycle["calls"]
    assert len(calls["spawn"]) == 1 and len(calls["terminate"]) == 1
    options = calls["options"][0]
    assert options.tower_pid == 4242 and callable(options.on_abort)
    options.on_abort()  # M3: the client's abort kills the Tower at once
    assert len(calls["terminate"]) == 2
    run = _run_json(lifecycle)
    assert run["harness"] == {"sha1": "harness"} and run["listener_pids"] == [4242]
    assert set(run["timing_env"]) == set(runner.TIMING_ENV_KEYS)
    assert run["live_tower_at_start"] == "idle" and run["job"] is True
    assert (lifecycle["out"] / "report.json").exists()


@pytest.mark.parametrize("how", ["health-timeout", "replay-raises", "ctrl-c", "no-job", "foreign-listener",
                                 "live-walk-during-startup"])
def test_the_runner_stops_the_tower_on_every_exit_path(lifecycle, monkeypatch, how):
    """T3 / L2 / M3 / M4: whatever ends the run, the Tower started is stopped."""
    state, calls = lifecycle["state"], lifecycle["calls"]
    expect = None
    if how == "health-timeout":
        state["health"] = (None, None)
        expect = pytest.raises(SystemExit, match="did not answer")
    elif how == "replay-raises":
        state["replay"] = lambda options: (_ for _ in ()).throw(RuntimeError("boom"))
    elif how == "ctrl-c":
        state["replay"] = lambda options: (_ for _ in ()).throw(KeyboardInterrupt())
    elif how == "no-job":
        if os.name != "nt":
            pytest.skip("a Job Object is a Windows guarantee")
        state["job"] = None
        expect = pytest.raises(SystemExit, match="Job Object")
    elif how == "foreign-listener":
        state["owners"] = {4242, 777}
        expect = pytest.raises(SystemExit, match="served by")
    elif how == "live-walk-during-startup":
        # idle at the start check and at the pre-spawn re-read (C24 HIGH-3), a walk once the Tower is starting
        state["live"] = ["idle", "idle", "recording"]
        monkeypatch.setattr(runner, "LIVE_WATCH_EVERY_S", 0.0)
    if expect is None:
        code = runner.main(lifecycle["argv"])
    else:
        with expect:
            runner.main(lifecycle["argv"])
        code = None
    assert len(calls["spawn"]) == 1
    assert len(calls["terminate"]) == 1, "the finally must stop the Tower exactly once here"
    run = _run_json(lifecycle)
    assert "stopped_at" in run
    if how in ("replay-raises", "ctrl-c"):
        assert code == runner.EXIT_ERROR
    if how == "live-walk-during-startup":
        assert code == runner.EXIT_ABORTED and run["aborted"]["during"] == "startup"
        assert calls["options"] == []  # the replay never began
    if how == "no-job":
        assert run["job"] is False and calls["options"] == []


def test_the_runner_exits_3_and_reports_the_client_record_when_the_abort_raised(lifecycle):
    """C22-F2 item 5: the client's own record reached disk (its `finally`) but
    the exception reached the runner. The run was aborted, not broken."""
    def aborted_then_raised(options):
        (options.out / "client.json").write_text(json.dumps({
            "outcome": "aborted", "tower_captures": [], "stream": {"frames_sent": 7},
            "aborted": {"t": 1.0, "reason": "a live walk is being recorded on :8000", "on_abort_done": 1.2}}),
            encoding="utf-8")
        raise ConnectionResetError("the send was in the drain when the Tower died")

    lifecycle["state"]["replay"] = aborted_then_raised
    assert runner.main(lifecycle["argv"]) == runner.EXIT_ABORTED
    assert len(lifecycle["calls"]["terminate"]) == 1
    run = _run_json(lifecycle)
    assert run["aborted"]["during"] == "stream" and "ConnectionResetError" in run["aborted"]["raised"]
    built = json.loads((lifecycle["out"] / "report.json").read_text(encoding="utf-8"))
    assert built["client"]["aborted"]["reason"].startswith("a live walk")
    assert built["client"]["stream"]["frames_sent"] == 7


@pytest.mark.parametrize("how, match", [
    ("port-in-use", "already listens"),
    ("data-root-in-live-store", "inside the live store"),
    ("import-elsewhere", "import tower resolves"),
    ("out-not-empty", "not empty"),
])
def test_the_runner_refuses_before_starting_anything(lifecycle, monkeypatch, tmp_path, how, match):
    """T4: refusals, with the port probe and the live store monkeypatched --
    no port in 8031-8040 is bound or probed."""
    argv = list(lifecycle["argv"])
    if how == "port-in-use":
        monkeypatch.setattr(runner, "port_in_use", lambda port: True)
    elif how == "data-root-in-live-store":
        monkeypatch.setattr(runner, "LIVE_DATA", tmp_path / "live")
        argv[argv.index("--data-root") + 1] = str(tmp_path / "live" / "world_builder" / "x")
    elif how == "import-elsewhere":
        monkeypatch.setattr(runner, "probe_import", lambda tower_dir, env: r"C:\elsewhere\tower\__init__.py")
    elif how == "out-not-empty":
        lifecycle["out"].mkdir()
        (lifecycle["out"] / "runner.log").write_text("an earlier run", encoding="utf-8")
    with pytest.raises(SystemExit, match=match):
        runner.main(argv)
    assert lifecycle["calls"]["spawn"] == [] and lifecycle["calls"]["terminate"] == []


@pytest.mark.parametrize("live", ["recording", "busy", "unknown"])
def test_the_runner_starts_nothing_unless_8000_is_idle_or_down(lifecycle, live):
    """M1 / M2: the runner's own start check fails closed too."""
    lifecycle["state"]["live"] = [live]
    assert runner.main(lifecycle["argv"]) == runner.EXIT_ABORTED
    assert lifecycle["calls"]["spawn"] == []


def test_chore_reports_are_read_across_interleaved_access_lines(tmp_path):
    path = tmp_path / "out.log"
    path.write_text('INFO: a\n{\n  "owed": [],\nINFO:     127.0.0.1 - "GET /health"\n  "finished": []\n}\n'
                    "{\n broken\n}\n", encoding="utf-8")
    assert report.scan_chore_reports(path) == [{"owed": [], "finished": []}]


def test_the_client_script_path_is_where_the_brief_says(tmp_path):
    assert Path(replay.__file__).name == "world_live_replay.py"
    assert Path(runner.__file__).name == "world_live_replay_run.py"


# -- C22-F3: the replay-fidelity bar (manager 142) and the round-3 LOWs -------------------------


V1_NUMBERS = {"offset_p50_abs_ms": 5.0, "offset_p95_ms": 60.0, "offset_p99_ms": 250.0,
              "beyond_ms": 50.0, "beyond_max_fraction": 0.05, "early_max_ms": 50.0,
              "client_lateness_p95_ms": 5.0}


def test_the_fidelity_bar_is_the_one_manager_142_approved():
    v1 = dict(report.FIDELITY_BAR["v1"])
    assert "manager 142" in v1.pop("ruling") and v1.pop("applies_after") is None
    assert v1 == V1_NUMBERS
    assert "manager 142" in report.FIDELITY_RULING
    assert report.PACING_OVER_S == pytest.approx(0.05)
    assert "NOT a pass" in report.FIDELITY_NA_NOTE and "--capture-root" in report.FIDELITY_NA_NOTE


def _fidelity_inputs():
    """A pacing block and a client record ON every edge of the bar: 1000 of
    4000 source frames sent (a first_seconds cut), all joined."""
    pacing = {"computable": True, "matched": 1000, "replay_only": 0, "source_only": 3000, "source_frames": 4000,
              "source_duplicate_seq": 0, "late_over_50ms": 50, "early_over_50ms": 0,
              "offset_error_ms": {"count": 1000, "p50": -5.0, "p95": 60.0, "p99": 250.0, "min": -50.0}}
    client = {"first_seconds": 60.0, "schedule": {"frames": 1000},
              "stream": {"frames_sent": 1000, "lateness_ms": {"p95": 5.0}, "late_over_50ms": 2}}
    return pacing, client


def test_a_replay_on_every_edge_of_the_fidelity_bar_passes():
    pacing, client = _fidelity_inputs()
    fidelity = report.replay_fidelity(pacing=pacing, client=client)
    assert fidelity["result"] == "PASS" and len(fidelity["rows"]) == 7
    assert all(row["result"] == "PASS" for row in fidelity["rows"])
    assert fidelity["bar"] == report.FIDELITY_BAR["v1"] and "manager 142" in fidelity["ruling"]
    # no recorded start: the bar as written
    assert fidelity["version"] == "v1" and fidelity["started_at"] is None
    assert "no recorded start" in fidelity["version_why"]


FIDELITY_BREAKS = {
    "p50 late": (lambda p, c: p["offset_error_ms"].update(p50=5.01), "|p50|"),
    "p50 early": (lambda p, c: p["offset_error_ms"].update(p50=-5.01), "|p50|"),
    "p95": (lambda p, c: p["offset_error_ms"].update(p95=60.01), "error p95"),
    "p99": (lambda p, c: p["offset_error_ms"].update(p99=250.01), "error p99"),
    "beyond 5 %": (lambda p, c: p.update(late_over_50ms=51), "frames beyond"),
    "early": (lambda p, c: p["offset_error_ms"].update(min=-50.01), "early: min"),
    "client lateness": (lambda p, c: c["stream"]["lateness_ms"].update(p95=5.01), "client send lateness"),
    "replay-only": (lambda p, c: p.update(replay_only=1), "wire_seq join"),
    "source unmatched": (lambda p, c: p.update(source_only=3001), "wire_seq join"),
    "sent, not joined": (lambda p, c: c["stream"].update(frames_sent=1001), "wire_seq join"),
    "duplicate seq": (lambda p, c: p.update(source_duplicate_seq=1), "wire_seq join"),
}


@pytest.mark.parametrize("name", list(FIDELITY_BREAKS))
def test_each_fidelity_bar_fails_the_verdict_on_its_own(name):
    pacing, client = _fidelity_inputs()
    change, check = FIDELITY_BREAKS[name]
    change(pacing, client)
    fidelity = report.replay_fidelity(pacing=pacing, client=client)
    failed = [row["check"] for row in fidelity["rows"] if row["result"] == "FAIL"]
    assert fidelity["result"] == "FAIL" and len(failed) == 1 and check in failed[0], failed


def test_no_source_journal_is_n_a_and_says_n_a_is_not_a_pass(tmp_path):
    fidelity = report.replay_fidelity(pacing={"computable": False, "why": f"no journal for ['{A}']"}, client={})
    assert fidelity["result"] == "n/a" and fidelity["rows"] == [] and "no journal" in fidelity["why"]
    assert "NOT a pass" in fidelity["note"]
    pacing, client = _fidelity_inputs()
    client.pop("schedule")  # a check with no value is not a pass either
    partial = report.replay_fidelity(pacing=pacing, client=client)
    assert partial["result"] == "n/a" and "wire_seq join" in partial["why"]
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    built = report.build_report(tower_log=log, client=_clean_client(), capture_root=tmp_path / "nothing-here",
                                data_root=tmp_path / "no-data")
    assert built["replay_fidelity"]["result"] == "n/a"
    markdown = report.render_markdown(built)
    assert "**Replay fidelity: n/a.**" in markdown and "n/a is NOT a pass" in markdown
    assert "replay fidelity **n/a** (n/a is NOT a pass for a proof set)" in markdown


def _pinned_run(tmp_path, *, late_ms=(1.0, 2.0), capture_root=None):
    """A finished 10976c6-style run: its client.json names NO capture_root. The
    source journal is in a snapshot directory; the test Tower's re-recording is
    `late_ms` behind it."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log = run_dir / "tower-8031-x.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    snapshot = _capture(tmp_path / "snap", A, started=1000.0, frames=[(1, 1001.0), (3, 1001.1)],
                        ended=1002.0).parent
    kept = tmp_path / "data"
    _capture(kept, CAP, started=5000.0, frames=[(1, 5001.0 + late_ms[0] / 1000), (3, 5001.1 + late_ms[1] / 1000)],
             ended=5002.0)
    _world(kept / "world_builder", BASE, appearance_end=BASE + 700)
    (run_dir / "run.json").write_text(json.dumps({"err_log": str(log), "data_root": str(kept)}), encoding="utf-8")
    client = {**_clean_client(), "schedule": {"frames": 2},
              "stream": {"frames_sent": 2, "unanswered": 0, "frame_errors": {}, "lateness_ms": {"p95": 1.2},
                         "late_over_50ms": 0},
              "speed": 1.0, "walk": [{"capture_id": A, "frames": 2, "recorded_seconds": 2.0, "recorded_fps": 1.0,
                                      "end_reason": "stop"}]}
    if capture_root is not None:
        client["capture_root"] = str(capture_root)
    (run_dir / "client.json").write_text(json.dumps(client), encoding="utf-8")
    return run_dir, snapshot


def test_capture_root_pins_a_re_render_to_a_snapshot_and_the_report_says_which_journal(tmp_path, monkeypatch):
    """Round 3 L-g: a pinned run's pacing row must not depend on the live store."""
    run_dir, snapshot = _pinned_run(tmp_path)
    monkeypatch.setattr(replay, "DEFAULT_CAPTURE_ROOT", tmp_path / "live-store-today")  # no longer holds A
    assert report.main(["--run-dir", str(run_dir), "--out", str(tmp_path / "unpinned")]) == 0
    unpinned = json.loads((tmp_path / "unpinned" / "report.json").read_text(encoding="utf-8"))
    assert unpinned["tower_side_pacing"]["source_capture_root_from"] == report.CAPTURE_ROOT_DEFAULT
    assert unpinned["replay_fidelity"]["result"] == "n/a"
    assert report.main(["--run-dir", str(run_dir), "--out", str(tmp_path / "pinned"),
                        "--capture-root", str(snapshot)]) == 0
    pinned = json.loads((tmp_path / "pinned" / "report.json").read_text(encoding="utf-8"))
    pacing = pinned["tower_side_pacing"]
    assert pacing["source_capture_root"] == str(snapshot) and pacing["source_capture_root_from"] == "--capture-root"
    digest = report.hashlib.sha256((snapshot / A / "frames.jsonl").read_bytes()).hexdigest()
    assert pacing["source_sha256"][A]["frames.jsonl"] == digest
    assert pinned["replay_fidelity"]["result"] == "PASS"
    markdown = (tmp_path / "pinned" / "REPORT.md").read_text(encoding="utf-8")
    assert "**Replay fidelity: PASS.**" in markdown and f"frames.jsonl sha256 {digest[:16]}" in markdown
    assert "The source root is from --capture-root" in markdown


def test_capture_root_given_wins_over_the_client_record_which_wins_over_the_default(tmp_path):
    client = {"capture_root": str(tmp_path / "recorded")}
    assert report.resolve_capture_root(tmp_path / "snap", client) == (tmp_path / "snap", report.CAPTURE_ROOT_GIVEN)
    assert report.resolve_capture_root(None, client) == (tmp_path / "recorded", report.CAPTURE_ROOT_FROM_CLIENT)
    assert report.resolve_capture_root(None, {}) == (replay.DEFAULT_CAPTURE_ROOT, report.CAPTURE_ROOT_DEFAULT)
    assert "AT RENDER TIME" in report.CAPTURE_ROOT_DEFAULT


def _beyond_max_path(path: Path) -> Path:
    """`path` in Windows' extended-length form, which MAX_PATH does not limit.
    This fixture nests a world (two 32-character ids) one level below
    tmp_path and reached exactly 260 characters under a long --basetemp
    (review C22 round 4). Elsewhere the path is returned unchanged."""
    if os.name != "nt" or str(path).startswith("\\\\?\\"):
        return path
    return Path("\\\\?\\" + str(Path(path).resolve()))


def test_a_fidelity_fail_is_its_own_verdict_and_leaves_the_codes_alone(tmp_path):
    built = {}
    for name, late in (("faithful", (1.0, 2.0)), ("unfaithful", (1.0, 300.0))):
        root = _beyond_max_path(tmp_path) / name
        root.mkdir()
        run_dir, _ = _pinned_run(root, late_ms=late, capture_root=root / "snap" / "captures")
        assert report.main(["--run-dir", str(run_dir), "--out", str(root / "out")]) == 0
        built[name] = json.loads((root / "out" / "report.json").read_text(encoding="utf-8"))
    bad = built["unfaithful"]
    assert bad["tower_side_pacing"]["source_capture_root_from"] == report.CAPTURE_ROOT_FROM_CLIENT
    assert (built["faithful"]["replay_fidelity"]["result"], bad["replay_fidelity"]["result"]) == ("PASS", "FAIL")
    # The code's verdict and its rows do not move with the replay's fidelity.
    assert [(r["check"], r["result"]) for r in bad["live_safety"]["rows"]] == \
        [(r["check"], r["result"]) for r in built["faithful"]["live_safety"]["rows"]]
    assert bad["live_safety"]["result"] == built["faithful"]["live_safety"]["result"]
    assert "replay_fidelity" not in bad["live_safety"]
    markdown = (_beyond_max_path(tmp_path) / "unfaithful" / "out" / "REPORT.md").read_text(encoding="utf-8")
    assert "**Replay fidelity: FAIL.**" in markdown and "invalid as proof" in markdown


def test_a_safe_code_verdict_stays_pass_when_the_replay_was_not_faithful(tmp_path):
    """The code's PASS is not turned into a FAIL by the harness's pace, nor the reverse."""
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _world(tmp_path / "data" / "world_builder", BASE, appearance_end=BASE + 700)
    _capture(tmp_path / "src", A, started=1000.0, frames=[(1, 1001.0), (3, 1001.1)], ended=1002.0)
    _capture(tmp_path / "data", CAP, started=5000.0, frames=[(1, 5001.0), (3, 5001.4)], ended=5002.0)
    built = report.build_report(tower_log=log, world_root=tmp_path / "data" / "world_builder",
                                client={**_clean_client(), "walk": [{"capture_id": A}], "speed": 1.0},
                                data_root=tmp_path / "data", capture_root=tmp_path / "src" / "captures")
    assert built["replay_fidelity"]["result"] == "FAIL"
    assert built["live_safety"]["result"] == "PASS"
    # C24 HIGH-3: this client streamed and kept no :8000 watch -- FAIL, never n/a.
    assert built["live_safety"]["environment"]["result"] == "FAIL"


def test_compare_leaves_runs_that_fail_fidelity_or_the_environment_out_of_the_baseline_range(tmp_path):
    """Manager 142 and round 3 L-f: an invalid run is flagged, and never widens the old path's noise."""
    same = [[1, 0], [10, 0]]
    for index, photos in enumerate([47.0, 47.5, 46.8]):
        _fake_run(tmp_path / f"old{index}", photos=photos, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "old-unfaithful", photos=30.0, lag_p95=6.8, sequence=same, fidelity="FAIL")
    _fake_run(tmp_path / "old-contended", photos=60.0, lag_p95=6.8, sequence=same, environment="FAIL")
    _fake_run(tmp_path / "old-unjudged", photos=47.2, lag_p95=6.8, sequence=same, fidelity=None)
    _fake_run(tmp_path / "new0", photos=47.1, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "new-unfaithful", photos=47.1, lag_p95=6.8, sequence=same, fidelity="FAIL")
    olds = ["old0", "old1", "old2", "old-unfaithful", "old-contended", "old-unjudged"]
    out = tmp_path / "cmp"
    assert report.main(["--out", str(out), "--compare", *(str(tmp_path / n) for n in olds),
                        "--candidate", str(tmp_path / "new0"), str(tmp_path / "new-unfaithful")]) == 0
    result = json.loads((out / "compare.json").read_text(encoding="utf-8"))
    photos = {m["metric"]: m for m in result["metrics"]}["stop_to_phone_photos_min"]
    assert (photos["min"], photos["max"]) == (46.8, 47.5)  # not 30.0, not 60.0
    # Round 4 M-1: the unjudged run (47.2) is not counted either.
    assert photos["baseline_in_range"] == [True, True, True, False, False, False]
    assert photos["mean"] == pytest.approx((47.0 + 47.5 + 46.8) / 3, abs=1e-4)
    unjudged = "replay fidelity not in this render (re-render it): not judged, not counted"
    excluded = {item["dir"]: item["reasons"] for item in result["excluded_from_baseline"]}
    assert excluded == {str(tmp_path / "old-unfaithful"): ["replay fidelity FAIL"],
                        str(tmp_path / "old-contended"): ["Environment (:8000) FAIL"],
                        str(tmp_path / "old-unjudged"): [unjudged]}
    assert [item["dir"] for item in result["invalid_candidates"]] == [str(tmp_path / "new-unfaithful")]
    assert [item["dir"] for item in result["fidelity_not_judged"]] == [str(tmp_path / "old-unjudged")]
    assert result["baseline_counted"] == 3
    markdown = (out / "COMPARE.md").read_text(encoding="utf-8")
    assert "**EXCLUDED FROM THE BASELINE RANGE: 3 run(s).**" in markdown
    assert f"- `{tmp_path / 'old-unfaithful'}`: replay fidelity FAIL" in markdown
    assert f"- `{tmp_path / 'old-contended'}`: Environment (:8000) FAIL" in markdown
    assert f"- `{tmp_path / 'old-unjudged'}`: {unjudged}" in markdown
    assert ("| stop_to_phone_photos_min **(W0 timing)** | 47.1 | 46.8 | 47.5 | 0.7 | 30.0, 60.0, 47.2 | "
            "47.1, 47.1 (INVALID) |") in markdown
    assert "(excluded from the range)" in markdown and "**INVALID CANDIDATE(S): 1.**" in markdown


# -- C22-F6: bar v3 (manager 149 §2) replaces v2, for runs started after 16:55 EDT ---------------

V3_APPLIES_AFTER = 1790628900.0  # 2026-09-28 16:55:00 EDT = 20:55:00 UTC
BEFORE, AFTER = V3_APPLIES_AFTER - 1.0, V3_APPLIES_AFTER + 1.0
# The OLD runs' recorded starts (run.json started_at, RUN\experiments\C22-REPLAY\runs\walk5-old-a4afea1-<n>).
OLD_RUN_STARTED = {2: 1790620478.101, 3: 1790631854.085, 4: 1790623633.864, 5: 1790634922.735}
JITTER_CHECK = "recorder-stamp offset jitter: p95 |offset - median(offset)| (ms)"
BIAS_CHECK = "recorder-stamp offset constant bias: |median(offset)| (ms)"
P50_CHECK = "recorder-stamp offset error |p50| (ms)"


def test_v3_is_v1_without_its_p50_clause_plus_jitter_and_bias_for_runs_started_after_16_55_edt():
    v1, v3 = report.FIDELITY_BAR["v1"], report.FIDELITY_BAR["v3"]
    # ONE versioned bar: v1 and v3. v2 (manager 148) judged no run; manager 149 withdrew it.
    assert sorted(report.FIDELITY_BAR) == ["v1", "v3"]
    assert v3["applies_after"] == V3_APPLIES_AFTER
    assert report.FIDELITY_V3_APPLIES_AFTER_TEXT == "2026-09-28 16:55 EDT"
    assert all(f"manager {n}" in v3["ruling"] for n in (142, 148, 149)) and "17:25 EDT" in v3["ruling"]
    assert v3["offset_abs_deviation_p95_ms"] == 45.0 and v3["offset_median_abs_ms"] == 20.0
    assert "offset_p50_abs_ms" not in v3                      # NO |p50| clause
    # every other clause is v1's, unchanged
    unchanged = {k: v for k, v in V1_NUMBERS.items() if k != "offset_p50_abs_ms"}
    assert {k: v3[k] for k in unchanged} == unchanged == {k: v1[k] for k in unchanged}
    assert set(v3) == set(unchanged) | {"ruling", "applies_after", "offset_abs_deviation_p95_ms",
                                        "offset_median_abs_ms"}
    assert "v1: manager 142" in report.FIDELITY_RULING and "v3: manager 149" in report.FIDELITY_RULING
    assert "v2:" not in report.FIDELITY_RULING


@pytest.mark.parametrize("started, version", [
    (None, "v1"), (V3_APPLIES_AFTER - 3600, "v1"), (BEFORE, "v1"),
    (V3_APPLIES_AFTER, "v1"),                  # AT the cut-off is not after it
    (V3_APPLIES_AFTER + 0.001, "v3"), (AFTER, "v3"),
    (V3_APPLIES_AFTER + 1799, "v3"),           # 17:24:59: before v3 was DECLARED, still after its cut-off
    (V3_APPLIES_AFTER + 86400, "v3"),
    # the OLD runs: 2 and 4 started before 16:55, 3 and 5 after it
    (OLD_RUN_STARTED[2], "v1"), (OLD_RUN_STARTED[4], "v1"), (OLD_RUN_STARTED[3], "v3"), (OLD_RUN_STARTED[5], "v3"),
])
def test_the_bar_version_is_the_one_in_force_when_the_run_started(started, version):
    got, why = report.fidelity_version(started)
    assert got == version
    assert ("after v3's cut-off" in why) == (version == "v3")
    if started is None:
        assert "no recorded start" in why


def test_the_runs_start_is_run_json_s_else_the_client_record_s():
    assert report.run_started_at({"started_at": BEFORE}, {"started_at": AFTER}) == (BEFORE, "run.json started_at")
    assert report.run_started_at({"err_log": "x"}, {"started_at": AFTER}) == (AFTER, "client.json started_at")
    assert report.run_started_at(None, {"started_at": 7}) == (7.0, "client.json started_at")
    assert report.run_started_at({"started_at": True}, {"started_at": "soon"}) == (None, None)
    assert report.run_started_at(None, None) == (None, None)


def _v3_inputs(*, median=20.0):
    """`_fidelity_inputs`, on every edge of v3 too: p95 |offset - median| is 45, |median| is 20."""
    pacing, client = _fidelity_inputs()
    pacing.update(offset_median_ms=median,
                  offset_abs_deviation_ms={"count": 1000, "mean": 9.0, "p50": 3.0, "p95": 45.0, "p99": 200.0,
                                           "max": 250.0})
    client["started_at"] = AFTER
    return pacing, client


@pytest.mark.parametrize("median", [20.0, -20.0])
def test_a_v3_run_on_every_edge_of_the_v3_bar_passes_and_says_v3(median):
    pacing, client = _v3_inputs(median=median)
    fidelity = report.replay_fidelity(pacing=pacing, client=client)
    assert fidelity["result"] == "PASS" and len(fidelity["rows"]) == 8
    assert all(row["result"] == "PASS" for row in fidelity["rows"])
    assert (fidelity["version"], fidelity["started_at"], fidelity["started_at_from"]) == \
        ("v3", AFTER, "client.json started_at")
    assert fidelity["bar"] == report.FIDELITY_BAR["v3"] and "manager 149" in fidelity["ruling"]
    checks = [row["check"] for row in fidelity["rows"]]
    assert P50_CHECK not in checks
    assert checks[1:3] == [JITTER_CHECK, BIAS_CHECK]
    assert [(row["value"], row["required"]) for row in fidelity["rows"][1:3]] == [(45.0, "<= 45"), (median, "<= 20")]


V3_BREAKS = {
    "jitter": (lambda p, c: p["offset_abs_deviation_ms"].update(p95=45.01), "jitter"),
    "bias late": (lambda p, c: p.update(offset_median_ms=20.01), "constant bias"),
    "bias early": (lambda p, c: p.update(offset_median_ms=-20.01), "constant bias"),
    **{name: case for name, case in FIDELITY_BREAKS.items() if not name.startswith("p50")},
}


@pytest.mark.parametrize("name", list(V3_BREAKS))
def test_each_v3_clause_fails_the_verdict_on_its_own(name):
    pacing, client = _v3_inputs()
    change, check = V3_BREAKS[name]
    change(pacing, client)
    fidelity = report.replay_fidelity(pacing=pacing, client=client)
    failed = [row["check"] for row in fidelity["rows"] if row["result"] == "FAIL"]
    assert fidelity["version"] == "v3"
    assert fidelity["result"] == "FAIL" and len(failed) == 1 and check in failed[0], failed


@pytest.mark.parametrize("p50", [5.01, -5.01, 7.35, 19.9])
def test_v3_does_not_judge_the_raw_p50_and_v1_still_does(p50):
    pacing, client = _v3_inputs()
    pacing["offset_error_ms"].update(p50=p50)
    assert report.replay_fidelity(pacing=pacing, client=client)["result"] == "PASS"
    client["started_at"] = BEFORE
    fidelity = report.replay_fidelity(pacing=pacing, client=client)
    failed = [row["check"] for row in fidelity["rows"] if row["result"] == "FAIL"]
    assert fidelity["version"] == "v1" and fidelity["result"] == "FAIL" and failed == [P50_CHECK]


def test_a_run_before_16_55_stays_on_v1_exactly_v3_s_clauses_are_not_added_to_it():
    pacing, client = _v3_inputs(median=35.0)
    pacing["offset_abs_deviation_ms"].update(p95=59.0)
    client["started_at"] = BEFORE
    fidelity = report.replay_fidelity(pacing=pacing, client=client)
    assert (fidelity["version"], fidelity["result"], len(fidelity["rows"])) == ("v1", "PASS", 7)
    assert fidelity["bar"] == report.FIDELITY_BAR["v1"] and "manager 142" in fidelity["ruling"]
    client["started_at"] = AFTER
    fidelity = report.replay_fidelity(pacing=pacing, client=client)
    assert [row["check"] for row in fidelity["rows"] if row["result"] == "FAIL"] == [JITTER_CHECK, BIAS_CHECK]


def test_a_v3_run_whose_pacing_lacks_the_v3_values_is_n_a_not_a_pass():
    pacing, client = _v3_inputs()
    del pacing["offset_abs_deviation_ms"], pacing["offset_median_ms"]
    fidelity = report.replay_fidelity(pacing=pacing, client=client)
    assert fidelity["result"] == "n/a" and "jitter" in fidelity["why"] and "constant bias" in fidelity["why"]
    assert "NOT a pass" in fidelity["note"]


def test_the_pacing_block_carries_the_offset_median_and_the_absolute_deviation_from_it(tmp_path):
    client = {**_pacing_fixture(tmp_path), "first_seconds": 1.95}
    pacing = report.tower_side_pacing(client=client, data_root=tmp_path / "data",
                                      capture_root=tmp_path / "src" / "captures")
    # offsets -2, 4, 20, 60: the median is (4 + 20) / 2 = 12; |offset - 12| = 14, 8, 8, 48
    assert pacing["offset_median_ms"] == pytest.approx(12.0, abs=0.01)
    jitter = pacing["offset_abs_deviation_ms"]
    assert jitter["count"] == 4 and jitter["mean"] == pytest.approx(19.5, abs=0.01)
    assert jitter["p50"] == pytest.approx(14.0, abs=0.01)
    assert jitter["p95"] == pytest.approx(48.0, abs=0.01) and jitter["max"] == pytest.approx(48.0, abs=0.01)
    assert "offset_detrended_ms" not in pacing   # v2's, withdrawn


def test_the_jitter_is_absolute_so_an_early_tail_counts_as_much_as_a_late_one(tmp_path):
    _capture(tmp_path / "src", A, started=1000.0,
             frames=[(1, 1001.0), (3, 1001.1), (5, 1001.25), (7, 1001.3)], ended=1002.0)
    # offsets 0, +1, -40, +2 ms: the median is 0.5; |offset - 0.5| = 0.5, 0.5, 40.5, 1.5
    _capture(tmp_path / "data", C1, started=5000.0,
             frames=[(1, 5001.0), (3, 5001.101), (5, 5001.21), (7, 5001.302)], ended=5002.0)
    client = {"walk": [{"capture_id": A}], "tower_captures": [C1], "speed": 1.0, "first_seconds": 1.95}
    pacing = report.tower_side_pacing(client=client, data_root=tmp_path / "data",
                                      capture_root=tmp_path / "src" / "captures")
    assert pacing["offset_median_ms"] == pytest.approx(0.5, abs=0.01)
    assert pacing["offset_abs_deviation_ms"]["p95"] == pytest.approx(40.5, abs=0.01)
    # the signed offset's p95 sees none of the early tail
    assert pacing["offset_error_ms"]["p95"] == pytest.approx(2.0, abs=0.01)


def _started_run(root, *, started, late_ms):
    run_dir, _snapshot = _pinned_run(root, late_ms=late_ms, capture_root=root / "snap" / "captures")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run["started_at"] = started
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return run_dir


def _render_started(tmp_path, name, *, started, late_ms):
    """A pinned run with a recorded start, re-rendered through --run-dir: (report.json, REPORT.md)."""
    root = _beyond_max_path(tmp_path) / name
    root.mkdir()
    run_dir = _started_run(root, started=started, late_ms=late_ms)
    assert report.main(["--run-dir", str(run_dir), "--out", str(root / "out")]) == 0
    return (json.loads((root / "out" / "report.json").read_text(encoding="utf-8")),
            (root / "out" / "REPORT.md").read_text(encoding="utf-8"))


def test_a_run_started_before_16_55_stays_on_v1_run_4_fails_and_run_2_passes(tmp_path):
    """Managers 148 and 149: never loosen a bar after seeing the data it fails. Run 4's shape -- a
    constant 7 ms receipt lag, the whole distribution shifted late -- at run 4's recorded start is
    judged by v1 and FAILS |p50|; run 2's shape (1-2 ms) at run 2's start PASSES v1. The same 7 ms
    lag started after 16:55 passes v3 (no |p50| clause, |median| 7 <= 20, no jitter)."""
    run4, run4_md = _render_started(tmp_path, "run4", started=OLD_RUN_STARTED[4], late_ms=(7.0, 7.0))
    run2, run2_md = _render_started(tmp_path, "run2", started=OLD_RUN_STARTED[2], late_ms=(1.0, 2.0))
    late, late_md = _render_started(tmp_path, "late", started=AFTER, late_ms=(7.0, 7.0))
    assert (run4["replay_fidelity"]["version"], run4["replay_fidelity"]["result"]) == ("v1", "FAIL")
    assert [r["check"] for r in run4["replay_fidelity"]["rows"] if r["result"] == "FAIL"] == [P50_CHECK]
    assert run4["replay_fidelity"]["started_at_from"] == "run.json started_at"
    assert (run2["replay_fidelity"]["version"], run2["replay_fidelity"]["result"]) == ("v1", "PASS")
    assert len(run2["replay_fidelity"]["rows"]) == 7
    assert (late["replay_fidelity"]["version"], late["replay_fidelity"]["result"]) == ("v3", "PASS")
    assert late["tower_side_pacing"]["offset_median_ms"] == pytest.approx(7.0, abs=0.01)
    assert late["tower_side_pacing"]["offset_abs_deviation_ms"]["p95"] == pytest.approx(0.0, abs=0.01)
    # the report says which version judged the run, and why
    assert "replay fidelity **FAIL**, judged by fidelity bar v1." in run4_md
    assert "**fidelity bar v1**, manager 142" in run4_md and "at or before v3's cut-off" in run4_md
    assert "replay fidelity **PASS**, judged by fidelity bar v1." in run2_md
    assert "replay fidelity **PASS**, judged by fidelity bar v3." in late_md
    assert "**fidelity bar v3**, manager 149" in late_md and "after v3's cut-off" in late_md
    assert "(run.json started_at)" in late_md


def test_a_run_started_after_16_55_with_a_constant_bias_fails_v3_on_that_clause_alone(tmp_path):
    biased, biased_md = _render_started(tmp_path, "biased", started=AFTER, late_ms=(25.0, 25.0))
    fidelity = biased["replay_fidelity"]
    assert (fidelity["version"], fidelity["result"]) == ("v3", "FAIL")
    assert [r["check"] for r in fidelity["rows"] if r["result"] == "FAIL"] == [BIAS_CHECK]
    assert "replay fidelity **FAIL**, judged by fidelity bar v3." in biased_md


def test_a_render_keeps_the_client_s_start_so_compare_judges_it_by_its_own_version(tmp_path):
    """A run without the runner's run.json start: the client record's decides, in the render and in --compare."""
    root = _beyond_max_path(tmp_path)
    run_dir, _snapshot = _pinned_run(root, late_ms=(7.0, 7.0), capture_root=root / "snap" / "captures")
    client = json.loads((run_dir / "client.json").read_text(encoding="utf-8"))
    (run_dir / "client.json").write_text(json.dumps({**client, "started_at": AFTER}), encoding="utf-8")
    assert report.main(["--run-dir", str(run_dir), "--out", str(root / "out")]) == 0
    built = json.loads((root / "out" / "report.json").read_text(encoding="utf-8"))
    assert built["client"]["started_at"] == AFTER and "started_at" not in built["run"]
    fidelity = built["replay_fidelity"]
    assert (fidelity["version"], fidelity["started_at_from"], fidelity["result"]) == \
        ("v3", "client.json started_at", "PASS")
    (row,) = report.compare_runs([root / "out"])["keyframes"]["baseline"]
    assert (row["fidelity"], row["fidelity_version"], row["fidelity_judged_by"], row["not_a_pass"]) == \
        ("PASS", "v3", "v3", [])


def test_compare_judges_each_run_by_its_own_bar_version(tmp_path):
    same = [[1, 0], [10, 0]]
    # counted: a v1 run rendered before versioning, run 2 (v1), a v3 run (the start from the client record)
    _fake_run(tmp_path / "old-v1-unversioned", photos=47.0, lag_p95=6.8, sequence=same, run_started=BEFORE)
    _fake_run(tmp_path / "old-2", photos=47.5, lag_p95=6.8, sequence=same, version="v1",
              run_started=OLD_RUN_STARTED[2])
    _fake_run(tmp_path / "old-v3", photos=46.8, lag_p95=6.8, sequence=same, version="v3", client_started=AFTER)
    # run 4: before 16:55, its v1 FAIL stands
    _fake_run(tmp_path / "old-4", photos=47.2, lag_p95=6.8, sequence=same, fidelity="FAIL", version="v1",
              run_started=OLD_RUN_STARTED[4])
    # after 16:55 but rendered before versioning (v1's numbers): not judged, whichever way it went
    _fake_run(tmp_path / "old-late-render-fail", photos=30.0, lag_p95=6.8, sequence=same, fidelity="FAIL",
              run_started=AFTER)
    _fake_run(tmp_path / "old-late-render-pass", photos=60.0, lag_p95=6.8, sequence=same, run_started=AFTER)
    # after 16:55, rendered under the withdrawn v2 (C22-F5): not judged either, whichever way it went
    _fake_run(tmp_path / "old-v2-render-pass", photos=20.0, lag_p95=6.8, sequence=same, version="v2",
              run_started=OLD_RUN_STARTED[3])
    _fake_run(tmp_path / "old-v2-render-fail", photos=70.0, lag_p95=6.8, sequence=same, fidelity="FAIL",
              version="v2", run_started=OLD_RUN_STARTED[5])
    # candidates: one judged by the wrong version (v3 on a pre-16:55 run), and a v3 run
    _fake_run(tmp_path / "new-wrong-bar", photos=47.1, lag_p95=6.8, sequence=same, version="v3",
              run_started=BEFORE)
    _fake_run(tmp_path / "new-v3", photos=47.1, lag_p95=6.8, sequence=same, version="v3", run_started=AFTER)
    olds = ["old-v1-unversioned", "old-2", "old-v3", "old-4", "old-late-render-fail", "old-late-render-pass",
            "old-v2-render-pass", "old-v2-render-fail"]
    out = tmp_path / "cmp"
    assert report.main(["--out", str(out), "--compare", *(str(tmp_path / n) for n in olds),
                        "--candidate", str(tmp_path / "new-v3"), str(tmp_path / "new-wrong-bar")]) == 0
    result = json.loads((out / "compare.json").read_text(encoding="utf-8"))
    assert result["compare"] == "c22-live-replay-compare/6"
    assert "v1: manager 142" in result["fidelity_ruling"] and "v3: manager 149" in result["fidelity_ruling"]
    photos = {m["metric"]: m for m in result["metrics"]}["stop_to_phone_photos_min"]
    assert photos["baseline_in_range"] == [True, True, True, False, False, False, False, False]
    assert (photos["min"], photos["max"]) == (46.8, 47.5) and result["baseline_counted"] == 3
    excluded = {item["dir"]: item["reasons"] for item in result["excluded_from_baseline"]}
    assert excluded[str(tmp_path / "old-4")] == ["replay fidelity FAIL"]
    for name, verdict, bar in (("old-late-render-fail", "FAIL", "v1"), ("old-late-render-pass", "PASS", "v1"),
                               ("old-v2-render-pass", "PASS", "v2"), ("old-v2-render-fail", "FAIL", "v2")):
        (reason,) = excluded[str(tmp_path / name)]
        assert reason.startswith(f"replay fidelity {verdict} under bar {bar}, but this run's own bar is v3")
        assert reason.endswith("not judged, not counted (re-render it)")
    assert [item["dir"] for item in result["invalid_candidates"]] == [str(tmp_path / "new-wrong-bar")]
    (reason,) = result["invalid_candidates"][0]["reasons"]
    assert reason.startswith("replay fidelity PASS under bar v3, but this run's own bar is v1")
    rows = {item["dir"]: item for item in result["keyframes"]["baseline"] + result["keyframes"]["candidate"]}
    assert (rows[str(tmp_path / "old-v1-unversioned")]["fidelity_version"],
            rows[str(tmp_path / "old-v1-unversioned")]["fidelity_judged_by"]) == ("v1", "v1")
    assert [rows[str(tmp_path / n)]["fidelity_version"] for n in ("old-2", "old-4", "old-v3", "new-v3")] == \
        ["v1", "v1", "v3", "v3"]
    markdown = (out / "COMPARE.md").read_text(encoding="utf-8")
    assert "| Fidelity bar (own / render) |" in markdown and "| PASS | v3 / v3 | yes |" in markdown
    assert "| FAIL | v3 / v1 | **NO**: replay fidelity FAIL under bar v1" in markdown
    assert "| PASS | v3 / v2 | **NO**: replay fidelity PASS under bar v2" in markdown
    assert "Each run is judged by its own fidelity bar version" in markdown


def test_the_runner_exits_1_when_the_guard_aborted_only_after_a_genuine_fault(lifecycle):
    """Round 3 L-b: the guard can abort inside run_replay's finally, after a
    real fault was re-raised. The OUTCOME is not `aborted`: exit 1."""
    def fault_then_a_late_abort(options):
        (options.out / "client.json").write_text(json.dumps({
            "outcome": None, "tower_captures": [], "stream": {"frames_sent": 7},
            "aborted": {"t": 1.0, "reason": "a live walk is being recorded on :8000"}}), encoding="utf-8")
        raise RuntimeError("a harness fault")

    lifecycle["state"]["replay"] = fault_then_a_late_abort
    assert runner.main(lifecycle["argv"]) == runner.EXIT_ERROR
    run = _run_json(lifecycle)
    assert "aborted" not in run and "a harness fault" in run["error"]
    assert "a harness fault" in run["guard_aborted_after_the_fault"]["raised"]
    built = json.loads((lifecycle["out"] / "report.json").read_text(encoding="utf-8"))
    assert built["client"]["stream"]["frames_sent"] == 7  # still reported from the client's record


@pytest.mark.parametrize("states, expected", [
    (["unknown", "unknown"], 2),
    (["unknown", "idle", "unknown", "unknown"], 3),
])
def test_a_fail_closed_abort_counts_every_unknown_answer(states, expected):
    """Round 3 L-d: the history keeps changes of state, so the aborting run of
    unknowns is one entry."""
    guard = replay.LiveGuard("idle", clock=lambda: 1.0)
    reason = None
    for state in states:
        reason = guard.observe(state)
    assert reason and "failing closed" in reason
    row = report._guard_row(":8000 during the run", guard.summary(), {"reason": reason})
    assert row["result"] == "FAIL"
    assert f"{expected} unknown answer(s) -- two in a row aborted the run" in row["value"]


def test_a_hard_stops_termination_of_a_background_solve_is_recognised(tmp_path):
    """Round 3 L-e: `terminated: a stop was requested` is a termination too."""
    log = tmp_path / "tower.err.log"
    log.write_text(_solves_log(BASE, terminated="stop"), encoding="utf-8")
    third = report.walk_timeline(report.scan_log(log), CAP)["background_solves"][2]
    assert third["terminated_at_stop"] is True and third["terminated_how"] == "a stop was requested"
    assert third["landed_by"] is None and third["terminated_line"] == 14
    built = report.build_report(tower_log=log, client={"tower_captures": [CAP]})
    row = next(r for r in built["live_safety"]["rows"] if r["check"].startswith("background-solve horizons"))
    assert row["value"].endswith("terminated at Stop: solve 3")
    assert "TERMINATED at Stop (a stop was requested) (L14)" in report.render_markdown(built)


def test_the_stop_to_final_timeline_names_a_hard_stops_termination_as_its_step(tmp_path):
    """Round 3 L-e, the waterfall part (round 4: mutant Le3 survived). Between
    the catch-up and the final solve, a hard stop's termination is the step --
    not "the last background solve finishes (no terminate line)"."""
    log = tmp_path / "tower.err.log"
    log.write_text(_solves_log(BASE, terminated="stop"), encoding="utf-8")
    _world(tmp_path / "wb", BASE, appearance_end=BASE + 700)  # its finalization starts at +104 s
    built = report.build_report(tower_log=log, world_root=tmp_path / "wb", client={"tower_captures": [CAP]})
    step = next(row for row in built["waterfall"] if row["n"] == "2")
    assert step["stage"] == ("Background solve pid 300 terminated: a stop was requested (a hard stop ends the "
                             "wait at once)")
    assert step["evidence"] == "L14" and step["wait"] == "waits on another process"
    assert step["start"] == pytest.approx(BASE + 104.0) and step["end"] == pytest.approx(BASE + 224.1, abs=0.002)
    assert "Background solve pid 300 terminated: a stop was requested" in report.render_markdown(built)


def test_a_surface_before_the_next_launch_is_a_possibly_earlier_landing_and_open(tmp_path):
    """Round 3 L-a, 1b's solve 1: live surface 1 was launched a rebuild before
    the next solve (not yet due). Round 4 L-1: the log proves only the next
    launch's rebuild; the surface's is a possibly-earlier landing, OPEN (a
    stale surface's relaunch looks the same)."""
    wb, ws = "tower.world_build_session", "tower.routes.ws"
    lines = [
        _line(BASE, ws, "[Tower][Session] stream_start: measurement window opened"),
        _line(BASE + 0.006, ws, f"[Tower][Capture] recording started: {CAP}"),
        _line(BASE + 0.013, "tower.capture_workers",
              f"[Tower][Worker] started world-build-session pid 21204 for capture {CAP}: python x"),
        _line(BASE + 1.7, wb, f"[Tower][WorldBuilder] session {S} in world {W}: source=live-capture "
                              f"capture={CAP} root=x observed=360x640"),
        _line(BASE + 20.0, wb, "[Tower][WorldBuilder] background solve 1 launched at 52 keyframes (pid 100)"),
        _line(BASE + 30.0, wb, "[Tower][WorldBuilder] rebuild 4: 90 keyframes -> 80 positioned poses, "
                               "900 points, 2 segments in 0.30s"),
        _line(BASE + 30.01, wb, "[Tower][WorldBuilder] live surface 1 launched (pid 900)"),
        _line(BASE + 31.5, wb, "[Tower][WorldBuilder] rebuild 5: 102 keyframes -> 95 positioned poses, "
                               "1000 points, 2 segments in 0.30s"),
        _line(BASE + 31.504, wb, "[Tower][WorldBuilder] background solve 2 launched at 102 keyframes (pid 200)"),
        _line(BASE + 100.0, ws, _summary(1200, "stream_stop")),
        _line(BASE + 100.002, ws, "[Tower][Capture] recording stopped (stop): 1200 frames, 900 bytes"),
        _line(BASE + 101.0, wb, "[Tower][WorldBuilder] final global solve launched (pid 34464)"),
    ]
    log = tmp_path / "tower.err.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    first = report.walk_timeline(report.scan_log(log), CAP)["background_solves"][0]
    assert first["landed_by"]["t"] == pytest.approx(BASE + 31.504, abs=0.002)  # still "landed BY" the launch
    assert (first["landed_by"]["rebuild_n"], first["landed_by"]["rebuild_line"]) == (5, 8)
    assert first["landed_by"]["rebuild_why"] == "it launched background solve 2; landed by then at the latest"
    earlier = first["landed_by"]["possibly_earlier"]
    assert (earlier["rebuild_n"], earlier["rebuild_line"], earlier["surface"]) == (4, 6, 1)
    assert earlier["after_launch_s"] == pytest.approx(10.01, abs=0.002) and "stale" in earlier["open"]
    # Solve 1 has no earlier solve whose late landing it could be: not "unknown".
    assert "landing_rebuild_unknown" not in first["landed_by"]
    assert first["live_surface_between"]["n"] == 1
    built = report.build_report(tower_log=log, client={"tower_captures": [CAP]})
    solve_1 = next(line for line in report.render_markdown(built).splitlines()
                   if line.startswith("- background solve 1 "))
    assert ("landed-by rebuild 5 L8: it launched background solve 2; landed by then at the latest; "
            "possibly earlier, rebuild 4 L6 (+10.01 s): it launched live surface 1 -- OPEN: the landing, "
            "or a stale surface's relaunch") in solve_1


def _late_landing_log(base):
    """Smoke 1's shape (review C22 round 4 L-1 (a)): solve 1 is reaped INSIDE
    the launch of solve 2 (no surface at rebuild 35), so its landing surfaces
    one rebuild later, 0.3 s after solve 2's launch. Solve 2's own landing is
    at rebuild 40, which launches solve 3 and live surface 2."""
    wb, ws = "tower.world_build_session", "tower.routes.ws"

    def rebuild(n, kf):
        return f"[Tower][WorldBuilder] rebuild {n}: {kf} keyframes -> {kf - 5} positioned poses, " \
               "900 points, 2 segments in 0.30s"

    return "\n".join([
        _line(base, ws, "[Tower][Session] stream_start: measurement window opened"),
        _line(base + 0.006, ws, f"[Tower][Capture] recording started: {CAP}"),
        _line(base + 0.013, "tower.capture_workers",
              f"[Tower][Worker] started world-build-session pid 21204 for capture {CAP}: python x"),
        _line(base + 1.7, wb, f"[Tower][WorldBuilder] session {S} in world {W}: source=live-capture "
                              f"capture={CAP} root=x observed=360x640"),
        _line(base + 20.0, wb, "[Tower][WorldBuilder] background solve 1 launched at 52 keyframes (pid 100)"),
        _line(base + 60.0, wb, rebuild(34, 100)),
        _line(base + 61.5, wb, rebuild(35, 104)),
        _line(base + 61.504, wb, "[Tower][WorldBuilder] background solve 2 launched at 104 keyframes (pid 200)"),
        _line(base + 61.8, wb, rebuild(36, 105)),
        _line(base + 61.81, wb, "[Tower][WorldBuilder] live surface 1 launched (pid 900)"),
        _line(base + 90.0, wb, rebuild(40, 160)),
        _line(base + 90.004, wb, "[Tower][WorldBuilder] background solve 3 launched at 160 keyframes (pid 300)"),
        _line(base + 90.01, wb, "[Tower][WorldBuilder] live surface 2 launched (pid 901)"),
        _line(base + 100.0, ws, _summary(1200, "stream_stop")),
        _line(base + 100.002, ws, "[Tower][Capture] recording stopped (stop): 1200 frames, 900 bytes"),
        _line(base + 130.0, wb, "[Tower][WorldBuilder] final global solve launched (pid 34464)"),
    ]) + "\n"


def test_a_surface_0_3_s_after_a_launch_is_an_unknown_landing_not_this_solves(tmp_path):
    """Review C22 round 4 L-1: ee6c385 read rebuild 36 as solve 2's landing,
    0.3 s after solve 2 launched. It is shown as UNKNOWN, and OPEN."""
    log = tmp_path / "tower.err.log"
    log.write_text(_late_landing_log(BASE), encoding="utf-8")
    first, second, third = report.walk_timeline(report.scan_log(log), CAP)["background_solves"]
    assert (first["landed_by"]["rebuild_n"], first["landed_by"]["rebuild_line"]) == (35, 7)
    assert first["live_surface_between"] is None and "possibly_earlier" not in first["landed_by"]
    landed = second["landed_by"]
    assert (landed["rebuild_n"], landed["rebuild_line"]) == (40, 11)  # not 36
    assert landed["rebuild_why"] == "it launched background solve 3 and live surface 2"
    unknown = landed["landing_rebuild_unknown"]
    assert (unknown["rebuild_n"], unknown["rebuild_line"], unknown["surface"]) == (36, 9, 1)
    assert unknown["after_launch_s"] == pytest.approx(0.31, abs=0.002)
    assert "possibly_earlier" not in landed
    assert second["horizon_s"] == pytest.approx(28.5, abs=0.01)  # the metric does not move
    assert "landing_rebuild_unknown" not in (third["landed_by"] or {})
    markdown = report.render_markdown(report.build_report(tower_log=log, client={"tower_captures": [CAP]}))
    solve_2 = next(line for line in markdown.splitlines() if line.startswith("- background solve 2 "))
    assert "landed-by rebuild 40 L11: it launched background solve 3 and live surface 2" in solve_2
    assert ("landing rebuild UNKNOWN: live surface 1 at rebuild 36 L9, +0.31 s after this launch, is not read "
            "as its landing -- OPEN: the first rebuild after this solve's launch") in solve_2
    assert "rebuild 36 L9:" not in solve_2  # never shown as a landing rebuild


# -- C22-F4: review round 4 -- M-1 and the surviving mutants ----------------------------------

# Every way a run is not valid as proof; n/a and a pre-F3 render (no verdict) count no more than a FAIL.
NOT_VALID = {"fidelity n/a": {"fidelity": "n/a"}, "fidelity not rendered": {"fidelity": None},
             "fidelity FAIL": {"fidelity": "FAIL"}, "Environment FAIL": {"environment": "FAIL"}}


@pytest.mark.parametrize("why", list(NOT_VALID))
def test_three_baselines_one_not_valid_are_not_enough_valid_runs(tmp_path, why):
    """Review C22 round 4 M-1 (and mutant C04): exactly 3 baselines, one of them
    n/a -- or not rendered, FAIL, contended. Only 2 count, and 2 is not N >= 3."""
    same = [[1, 0], [10, 0]]
    _fake_run(tmp_path / "old0", photos=47.0, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "old1", photos=47.5, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "old2", photos=52.0, lag_p95=6.8, sequence=same, **NOT_VALID[why])
    result = report.compare_runs([tmp_path / f"old{i}" for i in range(3)])
    assert result["enough_runs"] is False and result["baseline_counted"] == 2
    photos = {m["metric"]: m for m in result["metrics"]}["stop_to_phone_photos_min"]
    assert (photos["mean"], photos["min"], photos["max"]) == (47.25, 47.0, 47.5)  # 52.0 is not counted
    assert photos["baseline_in_range"] == [True, True, False]
    assert [item["dir"] for item in result["excluded_from_baseline"]] == [str(tmp_path / "old2")]
    markdown = report.render_compare(result)
    assert "**Fewer than 3 valid runs on a side" in markdown and "2 of 3 baseline run(s) valid" in markdown
    assert "**EXCLUDED FROM THE BASELINE RANGE: 1 run(s).**" in markdown
    # A fourth, valid run makes it enough; the bad one still does not count.
    _fake_run(tmp_path / "old3", photos=46.8, lag_p95=6.8, sequence=same)
    four = report.compare_runs([tmp_path / f"old{i}" for i in range(4)])
    assert four["enough_runs"] is True and four["baseline_counted"] == 3


@pytest.mark.parametrize("why", list(NOT_VALID))
def test_a_candidate_that_is_not_valid_is_invalid_and_not_counted_toward_n_3(tmp_path, why):
    """Round 4 M-1 and C04, the candidate side: an n/a candidate is INVALID
    like a FAIL one, and 3 candidates with one of them not valid are too few."""
    same = [[1, 0], [10, 0]]
    for index, photos in enumerate([47.0, 47.5, 46.8]):
        _fake_run(tmp_path / f"old{index}", photos=photos, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "new0", photos=47.1, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "new1", photos=47.2, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "new2", photos=47.3, lag_p95=6.8, sequence=same, **NOT_VALID[why])
    result = report.compare_runs([tmp_path / f"old{i}" for i in range(3)], [tmp_path / f"new{i}" for i in range(3)])
    assert result["baseline_counted"] == 3 and result["candidates_counted"] == 2
    assert result["enough_runs"] is False
    assert [item["dir"] for item in result["invalid_candidates"]] == [str(tmp_path / "new2")]
    markdown = report.render_compare(result)
    assert "2 of 3 candidate(s)" in markdown and "**INVALID CANDIDATE(S): 1.**" in markdown
    assert "| 47.1, 47.2, 47.3 (INVALID) |" in markdown
    new2 = next(line for line in markdown.splitlines() if line.startswith(f"| candidate: `{tmp_path / 'new2'}`"))
    assert new2.endswith(" (INVALID) |") and "| **NO**: " in new2


@pytest.mark.parametrize("why", list(NOT_VALID))
def test_the_keyframe_identity_reference_is_a_counted_baseline_run(tmp_path, why):
    """Round 4, mutant C08: the first baseline run is not valid as proof (and
    its sequence differs), so the reference is the first COUNTED run."""
    same, other = [[1, 0], [10, 0], [54, 1]], [[1, 0], [10, 0], [55, 1]]
    _fake_run(tmp_path / "old-bad", photos=47.0, lag_p95=6.8, sequence=other, **NOT_VALID[why])
    _fake_run(tmp_path / "old0", photos=47.5, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "old1", photos=46.8, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "new0", photos=47.1, lag_p95=6.8, sequence=same)
    result = report.compare_runs([tmp_path / "old-bad", tmp_path / "old0", tmp_path / "old1"], [tmp_path / "new0"])
    keyframes = result["keyframes"]
    assert keyframes["reference"] == str(tmp_path / "old0") and keyframes["reference_counted"] is True
    identical = {item["dir"]: item["identical"] for item in keyframes["baseline"] + keyframes["candidate"]}
    assert identical == {str(tmp_path / "old-bad"): False, str(tmp_path / "old0"): True,
                         str(tmp_path / "old1"): True, str(tmp_path / "new0"): True}
    assert keyframes["baseline"][0]["first_difference"] == 2
    markdown = report.render_compare(result)
    assert f"Reference: `{tmp_path / 'old0'}`\n" in markdown
    # With no counted run at all, the fallback reference says it is not proof.
    alone = report.compare_runs([tmp_path / "old-bad"], [tmp_path / "new0"])
    assert alone["keyframes"]["reference"] == str(tmp_path / "old-bad")
    assert alone["keyframes"]["reference_counted"] is False
    assert "**NO baseline run is valid as proof: this reference is not one either.**" in report.render_compare(alone)


def test_the_runners_report_labels_its_capture_root_as_the_one_the_replay_streamed_from(lifecycle, tmp_path):
    """Round 3 L-g, the runner's part (round 4: mutant Lg6 survived). The
    runner's --capture-root is what the replay read, so the report says so --
    not the bare "--capture-root" a re-render's explicit root gets."""
    snapshot = tmp_path / "snapshot"
    assert runner.main([*lifecycle["argv"], "--capture-root", str(snapshot)]) == 0
    assert lifecycle["calls"]["options"][0].capture_root == snapshot
    built = json.loads((lifecycle["out"] / "report.json").read_text(encoding="utf-8"))
    pacing = built["tower_side_pacing"]
    assert pacing["source_capture_root"] == str(snapshot)
    assert pacing["source_capture_root_from"] == "the runner's --capture-root (the replay streamed from it)"
    assert pacing["source_capture_root_from"] != report.CAPTURE_ROOT_GIVEN


# -- C22-F7: the Codex xhigh review C24 (HIGH-1..4, MED-5..8) ---------------------------------------


# HIGH-1: the W0 verdict is the replay client's receipt (`phone_photos_at`); the store is INFO.


def test_a_client_that_was_never_told_is_not_reached_even_when_the_store_passes(tmp_path):
    """C24 HIGH-1: a run whose client heard no photographic push (no subscription) must not PASS on
    the store's clock alone. The store wrote the room's appearance at +8.3 min here."""
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _world(tmp_path / "root", BASE, appearance_end=BASE + 600.0)
    client = _clean_client(phone_view={"pushes": 0, "transitions": []})
    built = report.build_report(tower_log=log, world_root=tmp_path / "root", client=client)
    verdict = built["verdict"]
    assert verdict["basis"] == "phone" and verdict["result"] == "NOT REACHED"
    assert verdict["stop_to_room_with_photos_min"] is None and verdict["stop_to_phone_photos_min"] is None
    assert verdict["stop_to_store_photos_min"] == pytest.approx(8.33, abs=0.01)
    assert "no lifecycle.photographic word" in verdict["not_reached_why"]
    markdown = report.render_markdown(built)
    assert "(`phone_photos_at`): not reached (the client kept no lifecycle.photographic word" in markdown
    assert "**NOT REACHED**" in markdown
    # --compare's W0 metric is the client's: MISSING for this run, never the store's 8.33.
    assert report.comparable_metrics(built)[report.W0_TIMING_METRIC] is None


def test_a_real_walks_log_gives_the_store_time_labelled_not_the_client_measure(tmp_path):
    """C24 HIGH-1: no client record (a real walk's log): the store's time, and it says so."""
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _world(tmp_path / "root", BASE, appearance_end=BASE + 600.0)
    built = report.build_report(tower_log=log, world_root=tmp_path / "root", client={})
    verdict = built["verdict"]
    assert (verdict["basis"], verdict["measure"], verdict["result"]) == ("store", report.BASIS_STORE, "PASS")
    assert verdict["stop_to_room_with_photos_min"] == verdict["stop_to_store_photos_min"] == pytest.approx(8.33, abs=0.01)
    assert verdict["stop_to_phone_photos_min"] is None
    assert "NOT the client measure" in report.BASIS_STORE
    markdown = report.render_markdown(built)
    assert "-- the STORE's `updated_at`, NOT the client measure** (no client record: a real walk's log)" in markdown
    assert built["proof"] == {"proof": False, "not_proof_reasons": ["no replay client record (not a replay run)"]}


def test_compare_reads_a_render_made_before_f7_as_the_store_time_not_the_clients():
    """A pre-F7 render's `stop_to_room_with_photos_min` was the store's: it is never the W0 metric."""
    old = {"verdict": {"stop_to_room_with_photos_min": 47.03, "stop_to_phone_photos_min": 47.18}}
    metrics = report.comparable_metrics(old)
    assert (metrics["stop_to_phone_photos_min"], metrics["stop_to_store_photos_min"]) == (47.18, 47.03)
    new = {"verdict": {"basis": "phone", "stop_to_room_with_photos_min": 47.18, "stop_to_phone_photos_min": 47.18,
                       "stop_to_store_photos_min": 47.03}}
    assert report.comparable_metrics(new)["stop_to_store_photos_min"] == 47.03
    assert list(report.comparable_metrics(new))[0] == report.W0_TIMING_METRIC == "stop_to_phone_photos_min"


# HIGH-2: --compare counts only comparable runs; a candidate differs only in the declared switches.


def _keyed_render(root):
    """A real render whose comparability key is complete: a runner run.json and a client record."""
    run_dir, _snapshot = _pinned_run(root, capture_root=root / "snap" / "captures")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run.update(switches=dict(FAKE_SWITCHES), code=dict(FAKE_CODE), harness=FAKE_HARNESS,
               intrinsics_copied=["360x640.json"], started_at=BEFORE)
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    client = json.loads((run_dir / "client.json").read_text(encoding="utf-8"))
    client.update(after_stop="stay", live_tower_watch={"states_seen": ["idle"]}, live_guard=True,
                  not_a_proof_run=False)
    (run_dir / "client.json").write_text(json.dumps(client), encoding="utf-8")
    assert report.main(["--run-dir", str(run_dir), "--out", str(root / "out")]) == 0
    return json.loads((root / "out" / "report.json").read_text(encoding="utf-8"))


def test_the_comparability_key_is_complete_on_a_runners_render(tmp_path):
    built = _keyed_render(_beyond_max_path(tmp_path))
    key = built["comparability_key"]
    assert key == report.comparability_key(built)
    assert list(key) == list(report.KEY_FIELDS) and all(key[name] is not None for name in report.KEY_FIELDS)
    assert key["source_captures"] == [A]
    assert set(key["source_journal_sha256"][A]) == {"capture.json", "frames.jsonl"}
    assert key["switches"] == FAKE_SWITCHES and key["code"] == FAKE_CODE
    # the STREAMING harness only: a report-script change does not make runs incomparable
    assert key["harness"] == {"world_live_replay.py": "a" * 40, "world_live_replay_run.py": "b" * 40}
    assert key["replay"]["speed"] == 1.0 and key["replay"]["after_stop"] == "stay"
    assert key["replay"]["schedule"]["frames"] == 2
    assert key["calibration"] == ["360x640.json"] and key["fidelity_family"] == "recorder-stamp"


# Each way a baseline run can differ from the others' key.
KEY_BREAKS = {
    "switches": {"switches": {**FAKE_SWITCHES, "TOWER_WORLD_SOLVE_CONSENSUS": "1"}},
    "source_journal_sha256": {"journal": {CAP: {"capture.json": "1" * 64, "frames.jsonl": "9" * 64}}},
    "code": {"code": {**FAKE_CODE, "py_fingerprint": "0000000000000000"}},
    "code path": {"code": {**FAKE_CODE, "tower_dir": "C:\\\\elsewhere\\\\tower"}},
    "harness": {"harness": {"files_sha1": {**FAKE_HARNESS["files_sha1"], "world_live_replay.py": "d" * 40}}},
    "replay": {"speed": 2.0},
}


@pytest.mark.parametrize("name", list(KEY_BREAKS))
def test_a_baseline_run_that_differs_in_its_key_is_excluded_not_counted(tmp_path, name):
    same = [[1, 0], [10, 0]]
    for index, photos in enumerate([47.0, 47.5, 46.8]):
        _fake_run(tmp_path / f"old{index}", photos=photos, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "odd", photos=30.0, lag_p95=6.8, sequence=same, **KEY_BREAKS[name])
    # The odd run is FIRST: the reference is the majority key, not the first run's.
    result = report.compare_runs([tmp_path / "odd", *(tmp_path / f"old{i}" for i in range(3))])
    assert result["baseline_counted"] == 3 and result["enough_runs"] is True
    photos = {m["metric"]: m for m in result["metrics"]}["stop_to_phone_photos_min"]
    assert (photos["min"], photos["max"]) == (46.8, 47.5)  # 30.0 is not counted
    (reason,) = {item["dir"]: item["reasons"] for item in result["excluded_from_baseline"]}[str(tmp_path / "odd")]
    field = name.split()[0]
    assert reason.startswith("not comparable: " + ("switches differ" if field == "switches" else f"{field} differs"))
    assert result["comparability"]["reference_from"] == str(tmp_path / "old0")
    assert [item["dir"] for item in result["comparability"]["not_comparable"]] == [str(tmp_path / "odd")]
    markdown = report.render_compare(result)
    assert "**NOT COMPARABLE: 1 run(s), excluded.**" in markdown and "## Comparability (review C24 HIGH-2)" in markdown


def test_a_candidate_may_differ_only_in_the_declared_switches(tmp_path):
    same = [[1, 0], [10, 0]]
    for index, photos in enumerate([47.0, 47.5, 46.8]):
        _fake_run(tmp_path / f"old{index}", photos=photos, lag_p95=6.8, sequence=same)
    lever = {**FAKE_SWITCHES, "TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT": "on"}
    _fake_run(tmp_path / "new-lever", photos=40.0, lag_p95=6.8, sequence=same, switches=lever)
    _fake_run(tmp_path / "new-lever-2", photos=40.2, lag_p95=6.8, sequence=same, switches=lever)
    _fake_run(tmp_path / "new-extra", photos=40.1, lag_p95=6.8, sequence=same,
              switches={**lever, "TOWER_WORLD_DENSIFY": "true"})                 # an undeclared switch too
    _fake_run(tmp_path / "new-unapplied", photos=40.3, lag_p95=6.8, sequence=same)  # the lever never set
    _fake_run(tmp_path / "new-other-code", photos=40.4, lag_p95=6.8, sequence=same, switches=lever,
              code={**FAKE_CODE, "py_fingerprint": "1111111111111111"})          # the code differs too
    olds = [tmp_path / f"old{i}" for i in range(3)]
    news = [tmp_path / n for n in ("new-lever", "new-lever-2", "new-extra", "new-unapplied", "new-other-code")]
    out = tmp_path / "cmp"
    assert report.main(["--out", str(out), "--compare", *map(str, olds), "--candidate", *map(str, news),
                        "--candidate-switches", "TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT=on"]) == 0
    result = json.loads((out / "compare.json").read_text(encoding="utf-8"))
    assert result["comparability"]["candidate_switches"] == {"TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT": "on"}
    invalid = {item["dir"]: item["reasons"] for item in result["invalid_candidates"]}
    assert set(invalid) == {str(tmp_path / n) for n in ("new-extra", "new-unapplied", "new-other-code")}
    assert "TOWER_WORLD_DENSIFY: expected None, got 'true'" in invalid[str(tmp_path / "new-extra")][0]
    assert "TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT: expected 'on', got None" in \
        invalid[str(tmp_path / "new-unapplied")][0]
    assert invalid[str(tmp_path / "new-other-code")][0].startswith("not comparable: code differs")
    assert result["candidates_counted"] == 2
    # Without the declaration the lever itself is an undeclared difference.
    bare = report.compare_runs(olds, [tmp_path / "new-lever"])
    (reason,) = bare["invalid_candidates"][0]["reasons"]
    assert reason.startswith("not comparable: switches differ from the baseline's (")
    markdown = (out / "COMPARE.md").read_text(encoding="utf-8")
    assert "`TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT=on`" in markdown
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        report.parse_switches(["NOEQUALS"])


def test_a_run_whose_key_is_incomplete_is_never_comparable(tmp_path):
    """Fail closed: a run with no run.json (the standalone client) cannot show it is the same code."""
    same = [[1, 0], [10, 0]]
    for index, photos in enumerate([47.0, 47.5, 46.8]):
        _fake_run(tmp_path / f"old{index}", photos=photos, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "new-standalone", photos=40.0, lag_p95=6.8, sequence=same)
    path = tmp_path / "new-standalone" / "report.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc.pop("run")
    path.write_text(json.dumps(doc), encoding="utf-8")
    result = report.compare_runs([tmp_path / f"old{i}" for i in range(3)], [tmp_path / "new-standalone"])
    (reason,) = result["invalid_candidates"][0]["reasons"]
    assert reason.startswith("not comparable: its comparability key lacks switches, code, harness, calibration")
    # And with no complete baseline key there is nothing to be comparable WITH.
    alone = report.compare_runs([tmp_path / "new-standalone"])
    assert alone["baseline_counted"] == 0
    assert "no baseline run has a complete comparability key" in alone["excluded_from_baseline"][0]["reasons"][0]


def test_runs_judged_by_v1_and_by_v3_are_one_fidelity_family_and_compare(tmp_path):
    """Old runs 2 (started before 16:55, v1) and 3 and 5 (after, v3) are comparable: both bars judge
    the same recorder stamp. The family is the run's OWN bar's, from its recorded start."""
    same = [[1, 0], [10, 0]]
    for name, started in (("old2", OLD_RUN_STARTED[2]), ("old3", OLD_RUN_STARTED[3]), ("old5", OLD_RUN_STARTED[5])):
        _fake_run(tmp_path / name, photos=47.0, lag_p95=6.8, sequence=same, run_started=started,
                  version="v1" if name == "old2" else "v3")
    result = report.compare_runs([tmp_path / n for n in ("old2", "old3", "old5")])
    assert result["baseline_counted"] == 3 and result["comparability"]["not_comparable"] == []
    assert result["comparability"]["reference_key"]["fidelity_family"] == "recorder-stamp"
    assert report.FIDELITY_FAMILY == {"v1": "recorder-stamp", "v3": "recorder-stamp"}


# HIGH-3: --no-live-guard only in a declared NOT-proof run; no blind interval at the start.


def test_the_runner_refuses_no_live_guard_unless_the_run_is_declared_not_proof(lifecycle):
    lifecycle["state"]["replay"] = lambda options: {
        "tool": "world_live_replay", "outcome": "settled", "tower_captures": [],
        "live_guard": options.live_guard, "not_a_proof_run": options.not_a_proof_run}
    with pytest.raises(SystemExit, match="--not-a-proof-run"):
        runner.main([*lifecycle["argv"], "--no-live-guard"])
    assert lifecycle["calls"]["spawn"] == [] and not lifecycle["out"].exists()
    assert runner.main([*lifecycle["argv"], "--no-live-guard", "--not-a-proof-run"]) == 0
    run = _run_json(lifecycle)
    assert (run["live_guard"], run["not_a_proof_run"]) == (False, True)
    options = lifecycle["calls"]["options"][0]
    assert (options.live_guard, options.not_a_proof_run) == (False, True)
    built = json.loads((lifecycle["out"] / "report.json").read_text(encoding="utf-8"))
    assert built["proof"]["proof"] is False
    assert built["proof"]["not_proof_reasons"] == ["declared --not-a-proof-run",
                                                   "the :8000 guard was off (--no-live-guard)"]
    assert "**NOT-PROOF:**" in (lifecycle["out"] / "REPORT.md").read_text(encoding="utf-8")


def test_the_standalone_client_refuses_no_live_guard_unless_declared_not_proof(tmp_path):
    with pytest.raises(SystemExit, match="--not-a-proof-run"):
        replay.main(["--capture", A, "--port", "8031", "--tower-pid", "1", "--out", str(tmp_path / "out"),
                     "--no-live-guard"])
    assert not (tmp_path / "out").exists()


def test_an_unguarded_or_undeclared_stream_is_not_proof_and_its_environment_fails(tmp_path):
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    watched = {"states_seen": ["idle"], "history": [{"t": 1.0, "state": "idle"}]}
    run = {"live_tower_watch_startup": watched}   # the runner's startup watch PASSES...
    unguarded = report.build_report(tower_log=log, client=_clean_client(live_guard=False), run=run)
    rows = {r["check"]: r for r in unguarded["live_safety"]["environment"]["rows"]}
    assert rows[":8000 during the run"]["result"] == "FAIL"               # ...but the stream was not watched
    assert rows[":8000 during the run"]["value"] == "NOT WATCHED: the guard was off (--no-live-guard)"
    assert unguarded["live_safety"]["environment"]["result"] == "FAIL"
    assert unguarded["proof"]["not_proof_reasons"] == ["the :8000 guard was off (--no-live-guard)"]
    declared = report.build_report(tower_log=log, client=_clean_client(live_tower_watch=watched,
                                                                        not_a_proof_run=True), run=run)
    assert declared["live_safety"]["environment"]["result"] == "PASS"
    assert declared["proof"] == {"proof": False, "not_proof_reasons": ["declared --not-a-proof-run"]}
    proof = report.build_report(tower_log=log, client=_clean_client(live_tower_watch=watched), run=run)
    assert proof["proof"] == {"proof": True, "not_proof_reasons": []}
    assert "Proof: eligible." in report.render_markdown(proof)


def test_compare_never_counts_a_not_proof_run(tmp_path):
    same = [[1, 0], [10, 0]]
    for index, photos in enumerate([47.0, 47.5, 46.8]):
        _fake_run(tmp_path / f"old{index}", photos=photos, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "old-unguarded", photos=30.0, lag_p95=6.8, sequence=same,
              not_proof="the :8000 guard was off (--no-live-guard)")
    result = report.compare_runs([tmp_path / n for n in ("old0", "old1", "old2", "old-unguarded")])
    assert result["baseline_counted"] == 3
    (reason,) = {i["dir"]: i["reasons"] for i in result["excluded_from_baseline"]}[str(tmp_path / "old-unguarded")]
    assert reason == "NOT-PROOF: the :8000 guard was off (--no-live-guard)"


def test_a_render_made_before_f7_is_judged_not_proof_from_its_own_client_record():
    """A pre-F7 report has no `proof`: it is worked out from what the report kept."""
    streamed_unwatched = {"client": {"stream": {"frames_sent": 4005}}}
    assert report.proof_status(client=streamed_unwatched["client"], run={})["not_proof_reasons"] == [
        "it streamed with no :8000 watch on record"]
    watched = {"stream": {"frames_sent": 4005}, "live_tower_watch": {"states_seen": ["idle"]}}
    assert report.proof_status(client=watched, run={})["proof"] is True


def test_the_runner_re_reads_8000_right_before_the_spawn(lifecycle, monkeypatch):
    """C24 HIGH-3: the preflight (import probe, up to 120 s) was unwatched. A live walk that began
    in it now stops the Tower from being started at all."""
    order = []
    answers = iter(["idle", "recording"])
    monkeypatch.setattr(runner, "live_tower_state", lambda url=None, timeout=None: order.append("live")
                        or next(answers))
    monkeypatch.setattr(runner, "probe_import", lambda tower_dir, env: order.append("import")
                        or str(tower_dir / "tower" / "__init__.py"))
    assert runner.main(lifecycle["argv"]) == runner.EXIT_ABORTED
    assert order == ["live", "import", "live"]
    assert lifecycle["calls"]["spawn"] == [] and lifecycle["calls"]["terminate"] == []
    assert "became recording during the preflight" in (lifecycle["out"] / "runner.log").read_text(encoding="utf-8")


def test_the_runner_reads_8000_at_the_hand_off_even_when_its_watch_is_not_due(lifecycle, monkeypatch):
    """C24 HIGH-3: the startup watch reads every 10 s; the hand-off to the client reads NOW."""
    monkeypatch.setattr(runner, "LIVE_WATCH_EVERY_S", 1000.0)   # never due during this startup
    lifecycle["state"]["live"] = ["idle", "idle", "recording"]  # start, pre-spawn, hand-off
    assert runner.main(lifecycle["argv"]) == runner.EXIT_ABORTED
    run = _run_json(lifecycle)
    assert run["aborted"]["during"] == "startup" and run["live_tower_before_spawn"] == "idle"
    assert lifecycle["calls"]["options"] == [] and len(lifecycle["calls"]["terminate"]) == 1
    assert run["live_tower_watch_startup"]["states_seen"] == ["idle", "recording"]


def test_the_client_arms_its_guard_before_anything_else_it_does(tmp_path, monkeypatch):
    """C24 HIGH-3: the guard is read AND armed first -- its own first read happens before the
    harness pin's git calls (which used to run before even the start read)."""
    import threading

    reads = []
    guard_read = threading.Event()

    def live(url=None, timeout=None):
        reads.append("live")
        if len(reads) >= 2:
            guard_read.set()  # the armed guard's first read
        return "idle"

    calls = []

    def harness():
        # every call, in order: was the guard already reading :8000 when the harness pin ran?
        calls.append(guard_read.wait(5))
        return {"sha1": "h"}

    monkeypatch.setattr(replay, "live_tower_state", live)
    monkeypatch.setattr(replay, "harness_identity", harness)
    monkeypatch.setattr(replay, "http_json", lambda *a, **k: (None, None))
    record = asyncio.run(replay.run_replay(_refusal_options(tmp_path, live_guard_every=0.01)))
    assert calls == [True]  # one harness call, and only after the guard was armed and reading
    assert record["outcome"] == "no-test-tower" and record["harness"] == {"sha1": "h"}
    assert record["live_tower_watch"]["states_seen"] == ["idle"]  # the disarmed guard's record is kept
    assert (record["live_guard"], record["not_a_proof_run"]) == (True, False)


def test_a_live_walk_before_the_first_frame_aborts_the_client_and_kills_the_tower(tmp_path, monkeypatch):
    answers = iter(["idle"] + ["recording"] * 100)
    monkeypatch.setattr(replay, "live_tower_state", lambda url=None, timeout=None: next(answers))

    def slow_target(*a, **k):
        time.sleep(0.3)  # the guard reads every 10 ms meanwhile
        return None, None

    monkeypatch.setattr(replay, "http_json", slow_target)
    killed = []
    record = asyncio.run(replay.run_replay(_refusal_options(tmp_path, live_guard_every=0.01,
                                                            on_abort=lambda: killed.append(True))))
    assert record["outcome"] == "aborted" and replay.exit_code(record) == replay.EXIT_ABORTED
    assert killed == [True] and "live walk" in record["aborted"]["reason"]
    assert "stream" not in record


# HIGH-4: the pacing clock is the recorder's post-reply stamp, and every label says so.


def test_the_fidelity_rows_name_the_recorder_stamp_and_say_what_the_bar_measures():
    pacing, client = _fidelity_inputs()
    fidelity = report.replay_fidelity(pacing=pacing, client=client)
    checks = [row["check"] for row in fidelity["rows"]]
    assert not any("receipt" in check for check in checks)
    assert [c for c in checks if c.startswith("recorder-stamp offset")] == [
        P50_CHECK, "recorder-stamp offset error p95 (ms)", "recorder-stamp offset error p99 (ms)"]
    assert fidelity["family"] == "recorder-stamp" and "post-reply" in fidelity["measures"]
    assert "not socket arrival" in fidelity["measures"]
    assert "post-reply stamp, not socket arrival" in report.PACING_ROW
    assert "not arrivals" in report.INTERVAL_ROW
    # The numbers are the approved ones, unchanged (review C24 HIGH-4: relabel, never re-tune).
    assert {k: v for k, v in report.FIDELITY_BAR["v1"].items() if k not in ("ruling", "applies_after")} == V1_NUMBERS


def test_a_tower_side_fail_with_the_client_on_time_is_flagged_as_possibly_the_code_under_test(tmp_path):
    pacing, client = _fidelity_inputs()
    pacing["offset_error_ms"].update(p95=61.0)                    # a Tower-side clause fails...
    tower_side = report.replay_fidelity(pacing=pacing, client=client)
    assert tower_side["result"] == "FAIL" and tower_side["tower_side_only"] is True
    client["stream"]["lateness_ms"].update(p95=9.0)               # ...and now the client was late too
    both = report.replay_fidelity(pacing=pacing, client=client)
    assert both["result"] == "FAIL" and both["tower_side_only"] is False
    same = [[1, 0], [10, 0]]
    for index, photos in enumerate([47.0, 47.5, 46.8]):
        _fake_run(tmp_path / f"old{index}", photos=photos, lag_p95=6.8, sequence=same)
    _fake_run(tmp_path / "new-slow-frame-path", photos=40.0, lag_p95=6.8, sequence=same, fidelity="FAIL",
              tower_side_only=True)
    result = report.compare_runs([tmp_path / f"old{i}" for i in range(3)], [tmp_path / "new-slow-frame-path"])
    (reason,) = result["invalid_candidates"][0]["reasons"]
    assert reason.startswith("replay fidelity FAIL (Tower-side clauses only, the client on time")
    assert "code under test's own frame path" in reason


def test_the_report_says_the_pacing_clock_is_post_reply(tmp_path):
    run_dir, snapshot = _pinned_run(_beyond_max_path(tmp_path), late_ms=(1.0, 300.0),
                                    capture_root=_beyond_max_path(tmp_path) / "snap" / "captures")
    out = _beyond_max_path(tmp_path) / "out"
    assert report.main(["--run-dir", str(run_dir), "--out", str(out)]) == 0
    markdown = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "Did the test Tower handle the frames at the recorded pace?" in markdown
    assert "What the bar measures (review C24 HIGH-4): the recorder's post-reply stamp" in markdown
    assert "This FAIL is on the Tower-side clauses alone, with the client on time" in markdown
    assert "Did the test Tower receive the recorded pace?" not in markdown


# MED-5: an incomplete /health answer is `unknown`, never `idle`.


@pytest.mark.parametrize("doc, state", [
    ({}, "unknown"),
    ({"capture_workers": {"workers": []}, "background_chore": {"state": "idle"}}, "unknown"),
    ({"capture": {"recording": False}, "background_chore": {"state": "idle"}}, "unknown"),
    ({"capture": {"recording": False}, "capture_workers": {"workers": []}}, "unknown"),
    ({**IDLE_HEALTH, "capture": {"armed": True, "error": "unavailable"}}, "unknown"),
    ({**IDLE_HEALTH, "capture": {"recording": False, "error": "unavailable"}}, "unknown"),
    ({**IDLE_HEALTH, "capture_workers": {"error": "unavailable"}}, "unknown"),
    ({**IDLE_HEALTH, "background_chore": {"error": "unavailable"}}, "unknown"),
    ({**IDLE_HEALTH, "capture": {"recording": None}}, "unknown"),
    ({**IDLE_HEALTH, "capture_workers": {"workers": None}}, "unknown"),
    ({**IDLE_HEALTH, "background_chore": {"runs": 1}}, "unknown"),
    ({**IDLE_HEALTH, "capture": "armed"}, "unknown"),
    # null blocks are the Tower saying "not configured here": a definite answer
    ({"capture": None, "capture_workers": None, "background_chore": None}, "idle"),
    (IDLE_HEALTH, "idle"),
    # a positive answer is believed whatever else is missing
    ({"capture": {"recording": True}}, "recording"),
    ({"capture_workers": {"workers": [{"pid": 1}]}}, "busy"),
    ({"background_chore": {"state": "running"}}, "busy"),
])
def test_an_incomplete_health_answer_is_unknown_not_idle(monkeypatch, doc, state):
    monkeypatch.setattr(replay, "health_probe", lambda url, timeout=None: ("ok", doc))
    assert replay.live_tower_state() == state
    assert (replay.health_gaps(doc) == []) is (state == "idle")


def test_the_replay_does_not_start_when_8000_answers_an_empty_health(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "health_probe", lambda url, timeout=None: ("ok", {}))
    monkeypatch.setattr(replay, "http_json", lambda *a, **k: pytest.fail("streamed toward a target"))
    record = asyncio.run(replay.run_replay(_refusal_options(tmp_path)))
    assert record["outcome"] == "refused-live-unknown" and replay.exit_code(record) == replay.EXIT_ABORTED
    assert "missing capture / capture_workers / background_chore" in \
        (tmp_path / "out" / "client.log").read_text(encoding="utf-8")


def test_a_target_whose_health_lacks_its_capture_or_worker_block_is_refused():
    assert "lacks capture_workers (absent)" in replay.target_refusal({"capture": {"recording": False}})
    assert "lacks capture.recording" in replay.target_refusal(
        {"capture": {"armed": True, "error": "unavailable"}, "capture_workers": {"workers": []}})
    # the target's chore is its own: not needed to vouch that nothing records or works
    assert replay.target_refusal({"capture": {"recording": False}, "capture_workers": {"workers": []}}) is None
    assert replay.target_refusal(IDLE_HEALTH) is None


# MED-6: every writing entrypoint refuses the live store.


def test_the_client_and_the_report_refuse_an_out_inside_the_live_store(tmp_path, monkeypatch):
    live = tmp_path / "live"
    monkeypatch.setattr(replay, "LIVE_DATA", live)
    inside = live / "world_builder" / "c22-out"
    with pytest.raises(SystemExit, match="inside the live store"):
        replay.main(["--capture", A, "--port", "8031", "--tower-pid", "1", "--out", str(inside)])
    with pytest.raises(SystemExit, match="inside the live store"):
        report.main(["--out", str(inside), "--tower-log", str(tmp_path / "x.log")])
    _fake_run(tmp_path / "old0", photos=47.0, lag_p95=6.8, sequence=[[1, 0]])
    with pytest.raises(SystemExit, match="inside the live store"):
        report.main(["--out", str(inside), "--compare", str(tmp_path / "old0")])
    assert not inside.exists()
    assert replay.LIVE_DATA == live and runner.LIVE_DATA == replay.CANONICAL_CAPTURE_ROOT.parent


# MED-7 and MED-8: the limits are printed, and the lag and identity labels say what they are.


def test_the_report_prints_what_the_replay_cannot_exercise(tmp_path):
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    markdown = report.render_markdown(report.build_report(tower_log=log, client=_clean_client()))
    assert f"Limits (review C24 MED-7): {report.REPLAY_LIMITS}" in markdown
    for words in ("never drops a frame", "send window", "stalled-socket reconnect", "physical walk"):
        assert words in report.REPLAY_LIMITS


def test_the_keyframe_rows_are_a_selection_sequence_and_an_accepted_keyframe_lag(tmp_path):
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    _world(tmp_path / "root", BASE, appearance_end=BASE + 700)
    built = report.build_report(tower_log=log, world_root=tmp_path / "root", client=_clean_client())
    rows = {r["check"]: r for r in built["live_safety"]["rows"]}
    assert "not their image content, not capture-qualified" in report.KEYFRAME_SEQUENCE_ROW
    assert "ACCEPTED keyframes only (not a per-frame observe lag)" in report.KEYFRAME_LAG_ROW
    assert rows[report.KEYFRAME_SEQUENCE_ROW]["result"] == "INFO"
    assert rows[report.KEYFRAME_LAG_ROW]["value"].endswith("; 0 accepted keyframe(s) unmatched")
    metrics = report.comparable_metrics(built)
    assert metrics["keyframe_accept_lag_s.p95"] == 7.0 and metrics["keyframe_accept_lag_unmatched"] == 0
    assert not any(name.startswith("observe_lag") for name in metrics)
    assert not any("inter_arrival" in name or name.startswith("tower_pacing") for name in metrics)
    report.write_report(tmp_path / "out", built)
    saved = json.loads((tmp_path / "out" / "keyframes.json").read_text(encoding="utf-8"))
    assert "not their image content" in saved["identity"]
    markdown = report.render_compare(report.compare_runs([]))
    assert "## Keyframe selection sequence, (source_seq, segment_index)" in markdown


# Manager 154 §2: the harness pin and version are self-evident in run.json, client.json and the report.


def test_the_harness_pin_and_its_version_are_in_every_record_and_report(lifecycle, tmp_path):
    # a committed, clean pin, whether or not these scripts sit in a checkout right now
    identity = {**replay.harness_identity(), "git_head": "f" * 40, "git_dirty": []}
    assert identity["version"] == replay.HARNESS_VERSION and "F7" in replay.HARNESS_VERSION
    pin = replay.harness_pin(identity)
    assert pin["version"] == replay.HARNESS_VERSION and pin["git_head"] == identity.get("git_head")
    assert set(pin["streaming_sha1"]) == set(replay.STREAMING_FILES) == set(report.STREAMING_HARNESS_FILES)
    assert pin["streaming_sha1"]["world_live_replay.py"] == identity["files_sha1"]["world_live_replay.py"]
    assert replay.harness_pin({"git_head": "x" * 40, "git_dirty": []})["committed_clean"] is True
    assert replay.harness_pin({"git_head": "x" * 40, "git_dirty": [" M a.py"]})["committed_clean"] is False
    assert replay.harness_pin({"files_sha1": {}})["committed_clean"] is False
    # run.json (the fixture's harness is a stand-in with no version)
    lifecycle["state"]["replay"] = lambda options: {"tool": "world_live_replay", "outcome": "settled",
                                                    "tower_captures": []}
    assert runner.main(lifecycle["argv"]) == 0
    run = _run_json(lifecycle)
    assert run["harness_pin"] == replay.harness_pin({"sha1": "harness"})
    # the report: which harness streamed the run, and which rendered it
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(BASE), encoding="utf-8")
    built = report.build_report(tower_log=log, client=_clean_client(harness=identity))
    assert built["harness_pin"]["version"] == replay.HARNESS_VERSION
    assert built["rendered_by"]["version"] == replay.HARNESS_VERSION and len(built["rendered_by"]["report_sha1"]) == 40
    markdown = report.render_markdown(built)
    assert (f"Harness pin (the harness that STREAMED this run): version `{replay.HARNESS_VERSION}`, git HEAD "
            f"`{identity.get('git_head')}` (committed, clean); streaming scripts sha1 world_live_replay.py "
            f"`{identity['files_sha1']['world_live_replay.py']}`") in markdown
    assert f"Rendered by harness version `{replay.HARNESS_VERSION}`" in markdown
    old = report.render_markdown(report.build_report(tower_log=log, client=_clean_client(
        harness={"git_head": "10976c6933db6b2f6662e53b79f0ee776ffed868", "git_dirty": [], "sha1": "f8d9"})))
    assert "version `not recorded (made before C22-F7)`, git HEAD `10976c6933db6b2f6662e53b79f0ee776ffed868`" in old
