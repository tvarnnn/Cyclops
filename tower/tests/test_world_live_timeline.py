import asyncio
import json

from scripts.world_live_timeline import LiveTimeline, horizon_summary, percentile
from scripts.world_live_replay import (FrameRecord, GeometryMirror, PhoneView,
                                       StreamStats, TowerSocket)


def test_percentile_matches_walk6_index_rule():
    assert percentile([1, 2, 3, 4], 50) == 3
    assert percentile([1, 2, 3, 4], 95) == 4


def test_horizon_proxy_counts_only_landings_before_stop():
    result = horizon_summary([1, 2, 3, 4], [(5, 2), (10, 3), (15, 4)], 12)
    assert result["keyframe_to_horizon_s"]["p50"] == 8
    assert result["unposed_at_stop"] == {"count": 2, "of": 4, "fraction": 0.5}


def test_frame_keyframe_pose_and_stop_summary(tmp_path):
    timeline = LiveTimeline(tmp_path, tmp_path, poll_interval=0.1)
    timeline.sent("source", 1, 10.0)
    timeline.sent("source", 2, 11.0)
    timeline.received("source", {"source_seq": 1, "received_at": 10.2})
    timeline.received("source", {"source_seq": 2, "received_at": 11.2})
    timeline.event({"kind": "keyframe_accepted", "at": 12.0,
                    "payload": {"keyframe_id": "s:00000001"}}, "source")
    timeline.event({"kind": "keyframe_accepted", "at": 13.0,
                    "payload": {"keyframe_id": "s:00000002"}}, "source")
    timeline.pose({"poses": {"s:00000001": {"component": 0}}}, 15.0)
    timeline.stop(16.0)
    timeline.pose({"poses": {"s:00000002": {"component": 0}}}, 20.0)
    summary = timeline.finish()
    assert summary["capture_to_keyframe_s"]["p50"] == 1.8
    assert summary["keyframe_to_first_common_pose_s"]["p50"] == 3.0
    assert summary["unposed_at_stop"] == {"count": 1, "of": 2, "fraction": 0.5}
    rows = [json.loads(line) for line in (tmp_path / "live-timeline-frames.jsonl").read_text().splitlines()]
    assert rows[0]["first_common_pose_at"] == 15.0
    assert rows[1]["first_common_pose_at"] == 20.0


def test_rejected_frame_and_phone_events_are_preserved(tmp_path):
    timeline = LiveTimeline(tmp_path, tmp_path)
    timeline.sent("source", 7, 1.0)
    timeline.frame_reply(7, {"type": "frame_result", "seq": 7}, 1.1)
    timeline.quality({"source_seq": 7, "reason": "blurred", "at": 1.2})
    timeline.phone({"type": "cartridge_result", "cartridge": "world_builder",
                    "payload": {"geometry": {"revision": "r1"}}}, 1.3)
    timeline.stop(2.0)
    timeline.finish()
    frame = json.loads((tmp_path / "live-timeline-frames.jsonl").read_text().splitlines()[0])
    events = [json.loads(line) for line in (tmp_path / "live-timeline-events.jsonl").read_text().splitlines()]
    assert frame["verdict"] == "blurred"
    assert events[0]["message"]["payload"]["geometry"]["revision"] == "r1"


def test_poll_joins_test_tower_capture_and_published_solution(tmp_path):
    data = tmp_path / "data"
    world = data / "world_builder"
    capture = data / "captures" / "tower"
    session = world / "worlds" / "world" / "sessions" / "session"
    solve = world / "worlds" / "world" / "solve" / "session"
    for path in (capture, session, solve):
        path.mkdir(parents=True)
    (capture / "frames.jsonl").write_text(json.dumps({"source_seq": 4, "received_at": 10.5}) + "\n")
    (session / "session.json").write_text(json.dumps({"capture_id": "tower"}))
    (session / "events.jsonl").write_text(json.dumps({"kind": "keyframe_accepted", "at": 11.0,
                                                      "payload": {"keyframe_id": "session:00000004"}}) + "\n")
    (solve / "solution.json").write_text(json.dumps({"poses": {"session:00000004": {"component": 0}}}))
    timeline = LiveTimeline(tmp_path / "out", world)
    timeline.sent("source", 4, 10.0)
    timeline.poll(["tower"])
    timeline.stop(timeline.frames[("source", 4)]["first_common_pose_at"] + 1)
    result = timeline.finish()
    assert result["capture_to_keyframe_s"]["p50"] == 0.5
    assert result["unposed_at_stop"]["count"] == 0


def test_frame_reply_can_arrive_before_send_returns(tmp_path):
    timeline = LiveTimeline(tmp_path, tmp_path)
    socket = TowerSocket("ws://unused", StreamStats(), PhoneView(), GeometryMirror("http://unused"), timeline)
    socket.capture_id = "source"
    class ImmediateReply:
        async def send(self, _text):
            socket._dispatch({"type": "frame_result", "seq": 7})
    socket.ws = ImmediateReply()
    frame = FrameRecord(wire_seq=7, source_seq=7, tx_seq=0, received_at=0,
                        relpath="frames/7.jpg", width=360, height=640)
    asyncio.run(socket.send_frame(frame, "{}", 2))
    assert timeline.frames[("source", 7)]["frame_reply"]["type"] == "frame_result"
