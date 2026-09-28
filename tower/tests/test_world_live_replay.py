"""C22: the real-time replay harness -- pacing, the /ws framing, and the report.

The pacing tests pin that a replay keeps the RECORDED pace (absolute offsets,
never accumulated lateness) and the walk's shape (Stop vs disconnect, a
reconnected walk's second socket, a truncated walk ending with Stop). The
framing test runs the client against a fake /ws server and checks every
message the phone would have sent. The report tests parse a log written in
the Tower's own line formats (copied from walk 5's log) and a small world
root, and check the waterfall, the milestones and the 10-minute verdict.

No test here starts a Tower, touches a GPU or reads the live store.
"""

import asyncio
import base64
import json
import time
from pathlib import Path

import pytest

from scripts import world_live_replay as replay
from scripts import world_live_replay_report as report
from scripts import world_live_replay_run as runner

# -- helpers -----------------------------------------------------------------------


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


def test_distribution():
    assert replay.distribution([]) == {"count": 0}
    stats = replay.distribution([1, 2, 3, 4, 100])
    assert stats["count"] == 5 and stats["p50"] == 3 and stats["max"] == 100


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


def test_a_replay_against_a_fake_tower_speaks_the_phones_protocol(tmp_path, monkeypatch):
    """The whole client run: handshake, session start, stream, Stop, settle."""
    import websockets

    jpeg = b"\xff\xd8replay\xff\xd9"
    _capture(tmp_path, A, started=500.0, frames=[(1, 500.05), (3, 500.10), (5, 500.15)],
             ended=500.2, jpeg=jpeg)
    received = []
    order = []

    async def handler(connection):
        async for raw in connection:
            message = json.loads(raw)
            received.append(message)
            order.append(("ws", message["type"]))
            kind = message["type"]
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
            elif kind == "frame":
                await connection.send(json.dumps({"type": "frame_result", "seq": message["seq"]}))

    health_polls = iter([
        {"capture": {"recording": True, "capture_id": "c" * 32}, "capture_workers": {"workers": [{}]},
         "background_chore": {"state": "idle", "runs": 1}},
    ])
    settle_states = iter([
        {"capture_workers": {"workers": [{"pid": 1}]}, "background_chore": {"state": "idle", "runs": 1}},
        {"capture_workers": {"workers": []}, "background_chore": {"state": "running", "runs": 2}},
        {"capture_workers": {"workers": []}, "background_chore": {"state": "idle", "runs": 2,
                                                                  "last": {"outcome": "finished"}}},
    ])
    streamed = {"done": False}

    def fake_http_json(method, url, timeout=10.0):
        if method == "POST":
            order.append(("http", "session/start"))
            assert url.endswith("/cartridges/world_builder/session/start")
            return 200, {"state": "active", "session_id": "sess"}
        if streamed["done"]:
            return 200, next(settle_states)
        return 200, next(health_polls, {"capture": {"recording": True, "capture_id": "c" * 32},
                                        "capture_workers": {"workers": [{}]},
                                        "background_chore": {"state": "idle", "runs": 1}})

    monkeypatch.setattr(replay, "http_json", fake_http_json)

    async def go():
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            options = replay.ReplayOptions(
                port=port, captures=[A], capture_root=tmp_path / "captures", out=tmp_path / "out",
                settle_timeout_min=0.5, poll_seconds=0.05, sample_seconds=60.0, session_lead=0.0,
                live_guard=False, phone_fetches=False)
            original = replay._follow_settle

            async def follow(*args, **kwargs):
                streamed["done"] = True
                return await original(*args, **kwargs)

            monkeypatch.setattr(replay, "_follow_settle", follow)
            return await replay.run_replay(options)

    record = asyncio.run(go())

    kinds = [kind for _, kind in order]
    assert kinds[:3] == ["ping", "cartridges", "result_subscribe"]
    assert order.index(("http", "session/start")) < order.index(("ws", "stream_start"))
    assert kinds[-1] == "stream_stop"
    frames = [m for m in received if m["type"] == "frame"]
    assert [m["seq"] for m in frames] == [1, 3, 5]
    assert [m["tx_seq"] for m in frames] == [0, 1, 2]
    assert all(base64.b64decode(m["data"]).startswith(jpeg) for m in frames)
    assert record["outcome"] == "settled"
    assert record["stream"]["frames_sent"] == 3
    assert record["stream"]["frame_results"] == 3
    assert record["stream"]["unanswered"] == 0
    assert record["tower_captures"] == ["c" * 32]
    assert record["phone_view"]["transitions"][0]["changed"]["model_state"] == "receiving"
    saved = json.loads((tmp_path / "out" / "client.json").read_text(encoding="utf-8"))
    assert saved["outcome"] == "settled"


def test_the_replay_refuses_while_a_live_walk_is_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "http_json",
                        lambda method, url, timeout=10.0: (200, {"capture": {"recording": True}}))
    options = replay.ReplayOptions(port=8031, captures=[A], capture_root=tmp_path, out=tmp_path / "out")
    record = asyncio.run(replay.run_replay(options))
    assert record["outcome"] == "refused-live-walk"
    assert replay.exit_code(record) == replay.EXIT_ABORTED


def test_a_live_walk_starting_mid_replay_aborts_it(tmp_path, monkeypatch):
    answers = iter([False, False, True])
    monkeypatch.setattr(replay, "live_walk_recording", lambda url=None, timeout=3.0: next(answers))
    options = replay.ReplayOptions(port=8031, captures=[A], capture_root=tmp_path, out=tmp_path / "out",
                                   live_guard_every=0.01)
    (tmp_path / "out").mkdir()

    async def go():
        abort, stop, record = asyncio.Event(), asyncio.Event(), {}
        await asyncio.wait_for(replay._guard_live(options, abort, record, stop), 5)
        return abort.is_set(), record

    aborted, record = asyncio.run(go())
    assert aborted and "live walk" in record["aborted"]["reason"]
    assert replay.exit_code({"outcome": "aborted"}) == replay.EXIT_ABORTED


def test_the_live_guard_reads_recording_from_health(monkeypatch):
    monkeypatch.setattr(replay, "http_json", lambda *a, **k: (200, {"capture": {"recording": False}}))
    assert replay.live_walk_recording() is False
    monkeypatch.setattr(replay, "http_json", lambda *a, **k: (None, None))
    assert replay.live_walk_recording() is None


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


# -- the report ----------------------------------------------------------------------------

W = "d" * 32
S = "e" * 32
CAP = "b5750fa3271d4e80b959c628e3e82c94"


def _ts(clock):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(clock))) + f",{int(round((clock % 1) * 1000)):03d}"


def _line(clock, logger, message, level="INFO"):
    return f"{_ts(clock)} {level} {logger} {message}"


def _walk_log(base):
    """A Tower log in the real line formats (walk 5's), 100 s walk, then the settle."""
    wb = "tower.world_build_session"
    lines = [
        "INFO:     Started server process [1]",
        _line(base - 2, "tower.cartridge_session", "[Tower][Session] world_builder start -> state=active"),
        _line(base, "tower.routes.ws", "[Tower][Session] stream_start: measurement window opened"),
        _line(base + 0.006, "tower.routes.ws", f"[Tower][Capture] recording started: {CAP}"),
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
        _line(base + 99.0, wb, "[Tower][WorldBuilder] rebuild 3: 12 keyframes -> 10 positioned poses, "
                               "1823 points, 2 segments in 1.08s"),
        _line(base + 100.0, "tower.routes.ws",
              "[Tower][Session] final summary: {'session_duration_s': 100.0, 'frames_received': 1200, "
              "'effective_fps': 12.0, 'tx_seq_gap_total': 0, 'backpressure_drops': 0, "
              "'frames_rejected': 0, 'end_reason': 'stream_stop'}"),
        _line(base + 100.002, "tower.routes.ws", "[Tower][Capture] recording stopped (stop): 1200 frames, 900 bytes"),
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
        _line(base + 900.0, "tower.routes.ws", f"[Tower][Capture] recording started: {'f' * 32}"),
        _line(base + 901.0, wb, "[Tower][WorldBuilder] rebuild 1: 4 keyframes -> 2 positioned poses, "
                                "5 points, 1 segments in 9.99s"),
    ]
    return "\n".join(lines) + "\n"


def _world(root, base, *, appearance_end):
    world = root / "worlds" / W
    (world / "sessions" / S).mkdir(parents=True)
    (world / "sessions" / S / "session.json").write_text(json.dumps({
        "session_id": S, "world_id": W, "capture_id": CAP, "ended_at": base + 104.0,
        "frames_observed": 1200, "keyframes_accepted": 16,
        "finalization": {"state": "complete", "final_solve": "solved", "started_at": base + 104.0,
                         "updated_at": base + 401.0},
        "stages": {"surface": {"state": "ok", "started_at": base + 401.0, "updated_at": base + 500.0},
                   "appearance": {"state": "ok", "started_at": base + 500.0, "updated_at": appearance_end},
                   "dense": {"state": "unavailable", "started_at": base + 525.0, "updated_at": base + 525.0}},
    }), encoding="utf-8")
    (world / "solve" / S).mkdir(parents=True)
    (world / "solve" / S / "solution.json").write_text(json.dumps({
        "timing": {"masks_s": 60.0, "match_s": 5.0, "map_s": 40.0, "gate_s": 20.0},
        "transients": {"computed": 16, "cache_hits": 0, "seconds": {"total": 60.0}},
        "gate": {"state": "applied", "consensus": {"requested": 3, "state": "applied", "draws": [
            {"draw": 0, "seed": 0, "map_s": None, "gate_s": 30.0},
            {"draw": 1, "seed": 1, "map_s": 40.0, "gate_s": 5.0}]}},
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


def test_the_timeline_is_read_from_the_towers_own_lines(tmp_path):
    base = time.mktime(time.strptime("2026-09-28 04:43:42", "%Y-%m-%d %H:%M:%S"))
    log = tmp_path / "tower.err.log"
    log.write_text(_walk_log(base), encoding="utf-8")
    timeline = report.walk_timeline(report.scan_log(log), CAP)
    assert timeline["world_id"] == W and timeline["session_id"] == S
    assert timeline["stop"]["t"] == pytest.approx(base + 100.002, abs=0.002)
    assert [r["n"] for r in timeline["rebuilds"]] == [1, 2, 3, 4]  # the later walk's rebuild is excluded
    assert timeline["tower_summary"]["frames_received"] == 1200
    assert timeline["bg_solve_wait"]["pid"] == "31120"
    assert timeline["final_done"]["posed"] == "15" and timeline["final_done"]["s"] == "175.90"
    assert timeline["appearance_seconds"]["total"] == 30.0
    assert [c["outcome"] for c in timeline["chores"]] == ["finished"]
    assert [s["keyframes"] for s in timeline["background_solves"]] == [52]


@pytest.mark.parametrize("appearance_after_stop_min, verdict", [(9.5, "PASS"), (10.5, "FAIL")])
def test_the_report_draws_the_waterfall_and_judges_the_hard_maximum(
        tmp_path, appearance_after_stop_min, verdict):
    base = time.mktime(time.strptime("2026-09-28 04:43:42", "%Y-%m-%d %H:%M:%S"))
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
    markdown = report.render_markdown(built)
    assert f"**{verdict}**" in markdown
    assert "| 3 | Final global solve, pid 34464 |" in markdown
    assert "solve_draw: 40.0 s (seed 1)" in markdown


def test_the_report_says_not_reached_when_the_photos_never_came(tmp_path):
    base = time.mktime(time.strptime("2026-09-28 04:43:42", "%Y-%m-%d %H:%M:%S"))
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


# -- the runner's pure parts ----------------------------------------------------------------


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


def test_chore_reports_are_read_across_interleaved_access_lines(tmp_path):
    path = tmp_path / "out.log"
    path.write_text('INFO: a\n{\n  "owed": [],\nINFO:     127.0.0.1 - "GET /health"\n  "finished": []\n}\n'
                    "{\n broken\n}\n", encoding="utf-8")
    assert report.scan_chore_reports(path) == [{"owed": [], "finished": []}]


def test_the_client_script_path_is_where_the_brief_says(tmp_path):
    assert Path(replay.__file__).name == "world_live_replay.py"
    assert Path(runner.__file__).name == "world_live_replay_run.py"
