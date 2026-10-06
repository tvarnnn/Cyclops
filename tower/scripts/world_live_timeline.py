"""Optional replay-side evidence recorder. Reads only the TEST Tower store."""

from __future__ import annotations

import json
import time
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    # Match the Walk-6 live-journal measurement's nearest indexed sample.
    return round(values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))], 3)


def _distribution(values):
    return {"n": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95)}


def horizon_summary(accepted_at, launches, stop_at):
    """Walk-6-compatible proxy: the next launch timestamps the prior solve landing.

    This counts input horizon, not actual published pose membership; callers
    must keep it separate from the pose rows observed in solution.json.
    """
    landings = [(launches[i + 1][0], launches[i][1]) for i in range(len(launches) - 1)]
    lags = []
    for index, accepted in enumerate(sorted(accepted_at), 1):
        landing = next((t for t, n in landings if n >= index and accepted <= t <= stop_at), None)
        if landing is not None:
            lags.append(landing - accepted)
    total = len(accepted_at)
    unposed = total - len(lags)
    return {"keyframe_to_horizon_s": _distribution(lags),
            "unposed_at_stop": {"count": unposed, "of": total,
                                 "fraction": round(unposed / total, 4) if total else None},
            "launches": launches}


class LiveTimeline:
    """Join source sends, test Tower journals, published solutions, and phone messages.

    Published `solution.json` poses are common-frame poses. The first time
    this observer sees a published pose is a conservative upper bound on its
    landing (bounded by poll_interval), not a claim that the phone rendered it.
    """

    def __init__(self, out, world_root, *, data_root=None, poll_interval=0.25):
        self.out = Path(out)
        self.world_root = Path(world_root)
        self.data_root = Path(data_root) if data_root else self.world_root.parent
        self.poll_interval = poll_interval
        self.frames = {}
        self._latest_by_seq = {}
        self.keyframes = {}
        self.events = []
        self.stop_at = None
        self.session_path = None
        self._session_source = None
        self._offsets = {}
        self._solution_stamp = None
        self._session = None
        self._tower_captures = []
        self._source_captures = []

    def sent(self, capture_id, seq, at):
        if capture_id not in self._source_captures:
            self._source_captures.append(capture_id)
        self.frames[(capture_id, int(seq))] = {"capture_id": capture_id, "source_seq": int(seq),
                                               "sent_at": at, "tower_received_at": None,
                                               "keyframe_accepted_at": None,
                                               "first_common_pose_at": None,
                                               "verdict": None}
        self._latest_by_seq[int(seq)] = self.frames[(capture_id, int(seq))]

    def _frame(self, capture_id, seq):
        return self.frames.get((capture_id, int(seq)))

    def received(self, capture_id, row):
        frame = self._frame(capture_id, row.get("source_seq", row.get("wire_seq")))
        if frame is not None:
            frame["tower_received_at"] = row.get("received_at")

    def frame_reply(self, seq, message, at):
        frame = self._latest_by_seq.get(seq)
        if frame is not None and "frame_reply_at" not in frame:
            frame["frame_reply_at"] = at
            frame["frame_reply"] = message

    def quality(self, row):
        seq = row.get("source_seq", row.get("wire_seq"))
        frame = self._latest_by_seq.get(seq)
        if frame is not None:
            frame["quality"] = row
            frame["verdict"] = row.get("reason", row.get("verdict"))

    def event(self, row, capture_id=None):
        if row.get("kind") != "keyframe_accepted":
            return
        key = (row.get("payload") or {}).get("keyframe_id")
        if not key:
            return
        try:
            seq = int(key.rsplit(":", 1)[1])
        except (ValueError, IndexError):
            return
        frame = self._frame(capture_id, seq) if capture_id else self._latest_by_seq.get(seq)
        if frame is not None:
            frame["keyframe_id"] = key
            frame["keyframe_accepted_at"] = row.get("at")
            frame["selection_reason"] = (row.get("payload") or {}).get("reason")
            self.keyframes[key] = frame

    def phone(self, message, at):
        self.events.append({"kind": "phone_message", "at": at, "message": message})
        geometry = (message.get("payload") or {}).get("geometry") or {}
        if geometry.get("revision") is not None:
            self.events.append({"kind": "phone_geometry_revision", "at": at,
                                "revision": geometry["revision"]})

    def pose(self, solution, at):
        for key, value in (solution.get("poses") or {}).items():
            frame = self.keyframes.get(key)
            if frame is not None and frame["first_common_pose_at"] is None and isinstance(value, dict):
                frame["first_common_pose_at"] = at
                frame["pose_component"] = value.get("component")
                self.events.append({"kind": "first_common_pose", "at": at, "keyframe_id": key})

    def stop(self, at):
        if self.stop_at is None:
            self.stop_at = at
            self.events.append({"kind": "stream_stop", "at": at})

    def _journal(self, path, callback):
        if not path.is_file():
            return
        offset = self._offsets.get(path, 0)
        with path.open("rb") as handle:
            handle.seek(offset)
            while line := handle.readline():
                if not line.endswith(b"\n"):
                    break
                offset = handle.tell()
                try:
                    callback(json.loads(line))
                except (ValueError, UnicodeError):
                    pass
        self._offsets[path] = offset

    def poll(self, tower_captures=()):
        self._tower_captures = list(tower_captures)
        for index, capture_id in enumerate(self._tower_captures):
            source_id = self._source_captures[index] if index < len(self._source_captures) else capture_id
            self._journal(self.data_root / "captures" / capture_id / "frames.jsonl",
                          lambda row, cid=source_id: self.received(cid, row))
        if self.session_path is None and self._tower_captures:
            for path in self.world_root.glob("worlds/*/sessions/*/session.json"):
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if doc.get("capture_id") in self._tower_captures:
                    self.session_path = path
                    index = self._tower_captures.index(doc["capture_id"])
                    self._session_source = (self._source_captures[index]
                                            if index < len(self._source_captures) else doc["capture_id"])
                    break
        if self.session_path is None:
            return
        directory = self.session_path.parent
        self._journal(directory / "events.jsonl",
                      lambda row: self.event(row, self._session_source))
        self._journal(directory / "frames_quality.jsonl", self.quality)
        try:
            self._session = json.loads(self.session_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        path = self.world_root / "worlds" / self.session_path.parents[2].name / "solve" / directory.name / "solution.json"
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp != self._solution_stamp:
                solution = json.loads(path.read_text(encoding="utf-8"))
                self._solution_stamp = stamp
                self.pose(solution, time.time())
                self.events.append({"kind": "solution_published", "at": time.time(),
                                    "solved_at": solution.get("solved_at"),
                                    "posed": len(solution.get("poses") or {})})
        except (OSError, ValueError):
            pass

    def finish(self, phone_photographic=(), tower_log=None):
        frames = list(self.frames.values())
        accepted = list(self.keyframes.values())
        for frame in frames:
            frame["keyframe_decision"] = ("accepted" if frame["keyframe_accepted_at"] is not None
                                          else "not_accepted")
        capture_to_keyframe = [f["keyframe_accepted_at"] - f["tower_received_at"] for f in accepted
                               if f["keyframe_accepted_at"] is not None and f["tower_received_at"] is not None]
        posed = [f for f in accepted if f["first_common_pose_at"] is not None
                 and (self.stop_at is None or f["first_common_pose_at"] <= self.stop_at)]
        lag = [f["first_common_pose_at"] - f["keyframe_accepted_at"] for f in posed]
        unposed = sum(f["first_common_pose_at"] is None or
                       (self.stop_at is not None and f["first_common_pose_at"] > self.stop_at)
                       for f in accepted)
        summary = {"frames_sent": len(frames), "keyframes_accepted": len(accepted),
                   "capture_to_keyframe_basis": "Tower frames.jsonl received_at (post-reply recorder stamp); physical capture time unavailable",
                   "capture_to_keyframe_s": _distribution(capture_to_keyframe),
                   "first_common_pose_basis": f"first {self.poll_interval} s poll observing keyframe in published solution.json poses",
                   "keyframe_to_first_common_pose_s": _distribution(lag),
                   "unposed_at_stop": {"count": unposed, "of": len(accepted),
                                        "fraction": round(unposed / len(accepted), 4) if accepted else None},
                   "stop_at": self.stop_at, "poll_interval_s": self.poll_interval,
                   "post_stop": {"session_finalization": (self._session or {}).get("finalization"),
                                 "stages": (self._session or {}).get("stages"),
                                 "phone_photographic": [row for row in phone_photographic
                                                         if self.stop_at is None or row["t"] >= self.stop_at],
                                 "events": [e for e in self.events if self.stop_at is not None
                                            and e["at"] > self.stop_at]}}
        if tower_log is not None and self.stop_at is not None:
            from scripts.world_live_replay_report import scan_log  # noqa: PLC0415
            launches = [(event.t, int(event.groups["kf"])) for event in scan_log(tower_log)
                        if event.kind == "bg_solve" and event.t <= self.stop_at]
            summary["walk6_compatible_horizon_proxy"] = horizon_summary(
                [f["keyframe_accepted_at"] for f in accepted], launches, self.stop_at)
        self.out.mkdir(parents=True, exist_ok=True)
        for name, rows in (("frames", frames), ("events", self.events)):
            with (self.out / f"live-timeline-{name}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, default=str) + "\n")
        (self.out / "live-timeline-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
