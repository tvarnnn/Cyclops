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
    assert rows[":8000 during the run"] == "FAIL"


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


def _fake_run(directory, *, photos, lag_p95, sequence, horizons=(52,)):
    directory.mkdir(parents=True)
    sha = report.hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode()).hexdigest()
    (directory / "report.json").write_text(json.dumps({
        "label": directory.name,
        "verdict": {"stop_to_room_with_photos_min": photos, "stop_to_settled_min": photos + 7},
        "tower_walk": {"totals": {"frames_received": 4005, "tx_seq_gap_total": 0},
                       "frames_observed": 4005, "rebuilds": {"count": 201, "seconds": {"p95": 1.1, "max": 1.9}},
                       "background_solves": [{"keyframes": k} for k in horizons]},
        "keyframes": {"sha256": sha, "observe_lag_s": {"p50": 2.0, "p95": lag_p95, "p99": 7.0, "max": 7.2}},
        "waterfall": [{"n": "3", "minutes": photos - 13, "child": False}],
        "live_safety": {"result": "PASS"}}), encoding="utf-8")
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
    photos = metrics["stop_to_room_with_photos_min"]
    assert (photos["min"], photos["max"], photos["spread"]) == (46.8, 47.5, 0.7)
    assert photos["outside"] == [40.0, 41.0, 40.5]
    assert metrics["observe_lag_s.p95"]["outside"] == [9.5]
    assert metrics["row 3 min"]["baseline"] == pytest.approx([34.0, 34.5, 33.8])
    candidates = result["keyframes"]["candidate"]
    assert [c["identical"] for c in candidates] == [True, False, True]
    assert candidates[1]["first_difference"] == 2 and candidates[1]["horizons_identical"] is False
    assert all(item["identical"] for item in result["keyframes"]["baseline"])
    assert "| stop_to_room_with_photos_min |" in (out / "COMPARE.md").read_text(encoding="utf-8")


def test_compare_says_when_there_are_too_few_runs(tmp_path):
    _fake_run(tmp_path / "old0", photos=47.0, lag_p95=6.7, sequence=[[1, 0]])
    result = report.compare_runs([tmp_path / "old0"])
    assert result["enough_runs"] is False
    assert "not a noise estimate" in report.render_compare(result)
    with pytest.raises(SystemExit, match="no report.json"):
        report.compare_runs([tmp_path / "missing"])


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
        state["live"] = ["idle", "recording"]  # idle at the start check, a walk once the Tower is starting
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
