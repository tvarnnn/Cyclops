#!/usr/bin/env python
"""The C22 replay report: from Stop to settled, in the shape of PROFILE-walk5.

Reads, never writes, everything except the report itself:
  * the test Tower's stderr log (every `[Tower][...]` line the builder, the
    supervisor and the chore write, with timestamps);
  * its stdout log (the finisher chore's JSON report: which areas, how long);
  * the world root: `session.json` (finalization and stage records),
    `solution.json` (the final solve's own stage timings and consensus),
    the room surface's `status.json`, the areas' `record.json`, and
    `stage_timing.json` when the Tower ran with TOWER_WORLD_STAGE_TIMING=on;
  * the client's own record (`client.json`) and CPU/GPU samples
    (`samples.csv`) when a replay produced them.

Everything is anchored on the capture the walk began with, so the same code
reads a replay Tower's log and a long live log holding many walks (which is
how it was checked against walk 5's `PROFILE-walk5.md`).

It includes the C19 F8 live-safety gate ("Live safety"): the Tower's
counters summed over every measurement window, frames sent = received =
observed, the keyframe sequence (`keyframes.json`), the keyframe observe
lag, rebuild latency and cadence, live-surface latency where the run
watched it, and :8000 during the run. All of it is read AFTER the fact, so
a finished run is re-reported without re-running it:

    python scripts/world_live_replay_report.py --run-dir <run's --out> --out <dir>
    python scripts/world_live_replay_report.py --out <dir> --tower-log <err.log> \
        [--tower-out-log <out.log>] [--world-root <root>] [--capture-id <id>]

and N runs are compared (old x N against new x N, per-metric spread):

    python scripts/world_live_replay_report.py --out <dir> --compare <old1> <old2> <old3> \
        [--candidate <new1> <new2> <new3>]
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402

HARD_MAX_MINUTES = 10.0
CHORE_CHAIN_GAP_S = 90.0

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) (\w+) (\S+) (.*)$")
_ID = r"[0-9a-f]{32}"

PATTERNS = {
    "stream_start": re.compile(r"\[Tower\]\[Session\] stream_start: measurement window opened"),
    "recording_started": re.compile(rf"\[Tower\]\[Capture\] recording started: (?P<cid>{_ID})"),
    "recording_stopped": re.compile(
        r"\[Tower\]\[Capture\] recording stopped \((?P<reason>[\w-]+)\): (?P<frames>\d+) frames, "
        r"(?P<bytes>\d+) bytes"),
    "final_summary": re.compile(r"\[Tower\]\[Session\] final summary: (?P<dict>\{.*\})\s*$"),
    "worker_started": re.compile(
        rf"\[Tower\]\[Worker\] started world-build-session pid (?P<pid>\d+) for capture (?P<cid>{_ID})"),
    "worker_finished": re.compile(
        rf"\[Tower\]\[Worker\] world-build-session worker pid (?P<pid>\d+) for capture (?P<cid>{_ID}) "
        r"finished after (?P<s>[\d.]+)s"),
    "worker_asked_to_stop": re.compile(
        rf"\[Tower\]\[Worker\] asked world-build-session worker pid (?P<pid>\d+) for capture (?P<cid>{_ID}) "
        r"to stop"),
    "session_opened": re.compile(
        rf"\[Tower\]\[WorldBuilder\] session (?P<sid>{_ID}) in world (?P<wid>{_ID}): source=\S+ "
        rf"capture=(?P<cid>{_ID})"),
    "rebuild": re.compile(
        r"\[Tower\]\[WorldBuilder\] rebuild (?P<n>\d+): (?P<kf>\d+) keyframes -> (?P<poses>\d+) positioned "
        r"poses, (?P<points>\d+) points, (?P<segments>\d+) segments in (?P<s>[\d.]+)s"),
    "rebuild_failed": re.compile(r"\[Tower\]\[WorldBuilder\] rebuild (?P<n>\d+) failed"),
    "bg_solve": re.compile(
        r"\[Tower\]\[WorldBuilder\] background solve (?P<n>\d+) launched at (?P<kf>\d+) keyframes "
        r"\(pid (?P<pid>\d+)\)"),
    "bg_solve_wait": re.compile(
        r"\[Tower\]\[WorldBuilder\] background solve pid (?P<pid>\d+) still running after (?P<s>[\d.]+)s; "
        r"terminating"),
    "live_surface": re.compile(r"\[Tower\]\[WorldBuilder\] live surface (?P<n>\d+) launched \(pid (?P<pid>\d+)\)"),
    "final_launched": re.compile(r"\[Tower\]\[WorldBuilder\] final global solve launched \(pid (?P<pid>\d+)\)"),
    "final_done": re.compile(
        r"\[Tower\]\[WorldBuilder\] final global solve: solved=(?P<solved>\w+) solver=(?P<solver>\S+) "
        r"posed=(?P<posed>\d+)/(?P<of>\d+)(?: components=(?P<components>\d+))?.*? in (?P<s>[\d.]+)s"),
    "final_failed": re.compile(r"\[Tower\]\[WorldBuilder\] final global solve (?:failed|could not start)"),
    "session_finished": re.compile(
        rf"\[Tower\]\[WorldBuilder\] session (?P<sid>{_ID}) finished: (?P<frames>\d+) frames, "
        r"(?P<kf>\d+) keyframes, .*?final build (?P<s>[\d.]+)s"),
    "transients": re.compile(
        rf"\[transients\] (?P<wid>{_ID})/(?P<sid>{_ID}): (?P<n>\d+) keyframes masked "
        r"\((?P<computed>\d+) computed, (?P<cached>\d+) cached\).*? in (?P<s>[\d.]+) s"),
    "consistency": re.compile(r"\[consistency\] applied over (?P<n>\d+) frames in (?P<s>[\d.]+)s"),
    "appearance_built": re.compile(
        rf"\[appearance\] (?P<wid>{_ID})/(?P<sid>{_ID}) built: (?P<what>[^{{]*)(?P<dict>\{{.*\}})\s*$"),
    "chore_started": re.compile(
        r"\[Tower\]\[Worker\] started the (?P<name>[\w-]+) chore \((?P<why>.*)\), pid (?P<pid>\d+)"),
    "chore_exited": re.compile(
        r"\[Tower\]\[Worker\] the (?P<name>[\w-]+) chore \(pid (?P<pid>\d+)\) exited (?P<code>-?\d+) "
        r"after (?P<s>[\d.]+) s: (?P<outcome>[\w-]+)"),
    "client_connected": re.compile(r"tower\.routes\.ws client connected|^client connected$"),
    "client_disconnected": re.compile(r"^client disconnected$"),
}


@dataclass(frozen=True)
class LogEvent:
    line: int
    t: float
    kind: str
    groups: dict
    text: str


def _epoch(stamp: str, millis: str) -> float:
    return time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S")) + int(millis) / 1000.0


def scan_log(path) -> list:
    """Every recognised, timestamped event in a Tower stderr log, in order."""
    events = []
    if path is None or not Path(path).exists():
        return events
    # Split on LF only, as `grep -n` counts: tqdm progress bars in the log
    # carry bare CRs, and universal newlines would shift every later line
    # number the report cites.
    with open(path, encoding="utf-8", errors="replace", newline="\n") as handle:
        for number, raw in enumerate(handle, 1):
            if "[Tower]" not in raw and "client " not in raw:
                continue
            match = _TS.match(raw.rstrip("\r\n"))
            if not match:
                continue
            stamp, millis, _level, logger, message = match.groups()
            for kind, pattern in PATTERNS.items():
                found = pattern.search(message)
                if found:
                    if kind in ("client_connected", "client_disconnected") and logger != "tower.routes.ws":
                        continue
                    events.append(LogEvent(number, _epoch(stamp, millis), kind,
                                           found.groupdict(), message))
                    break
    return events


def _literal(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        cleaned = re.sub(r"\b(nan|inf|-inf)\b", "None", text)
        try:
            return ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            return None


def scan_chore_reports(path) -> list:
    """The finisher chore's JSON reports from the Tower's stdout log.

    A report is a JSON object printed from a line that is exactly `{` to one
    that is exactly `}`; uvicorn's access lines may interleave and are
    skipped.
    """
    reports = []
    if path is None or not Path(path).exists():
        return reports
    block = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if block is None:
                if line == "{":
                    block = [line]
                continue
            if line.startswith("INFO:") or line.startswith("WARNING:") or line.startswith("ERROR:"):
                continue
            block.append(line)
            if line == "}":
                try:
                    reports.append(json.loads("\n".join(block)))
                except ValueError:
                    pass
                block = None
    return reports


def _first(events, kind, *, after=None, before=None, where=None):
    for event in events:
        if event.kind != kind:
            continue
        if after is not None and event.t < after:
            continue
        if before is not None and event.t > before:
            continue
        if where is not None and not where(event):
            continue
        return event
    return None


def _all(events, kind, *, after=None, before=None, where=None):
    return [e for e in events if e.kind == kind
            and (after is None or e.t >= after) and (before is None or e.t <= before)
            and (where is None or where(e))]


def walk_timeline(events: list, capture_id: str | None = None) -> dict:
    """The walk that began with `capture_id` (or the first capture in the
    log), from its stream_start to the last chore run that followed its
    builder's exit."""
    if capture_id is None:
        first = _first(events, "recording_started")
        capture_id = first.groups["cid"] if first else None
    started = _first(events, "recording_started", where=lambda e: e.groups["cid"] == capture_id)
    if started is None:
        return {"capture_id": capture_id, "found": False}
    t_start = started.t
    # The walk's captures: this one, and every capture opened after it until
    # the stream is stopped politely (a reconnect ends a capture by
    # disconnect and begins the next).
    stop = _first(events, "recording_stopped", after=t_start, where=lambda e: e.groups["reason"] == "stop")
    captures = [e.groups["cid"] for e in
                _all(events, "recording_started", after=t_start, before=stop.t if stop else None)]
    worker = _first(events, "worker_started", after=t_start - 5, where=lambda e: e.groups["cid"] in captures)
    worker_pid = worker.groups["pid"] if worker else None
    finished = _first(events, "worker_finished", after=t_start,
                      where=lambda e: e.groups["pid"] == worker_pid) if worker_pid else None
    session = _first(events, "session_opened", after=t_start - 5, where=lambda e: e.groups["cid"] in captures)
    end_of_worker = finished.t if finished else None
    t0 = stop.t if stop else None
    summary = _first(events, "final_summary", after=t_start,
                     before=(t0 + 5) if t0 else None,
                     where=lambda e: "'stream_stop'" in e.text or t0 is None)
    # Every measurement window of the walk, whatever ended it (review C22
    # H2): a reconnected walk is one window per capture, and its last
    # window alone undercounts it.
    windows = [e for e in _all(events, "final_summary", after=t_start, before=(t0 + 5) if t0 else None)]
    stopped = [e for e in _all(events, "recording_stopped", after=t_start, before=(t0 + 5) if t0 else None)]
    walk_end = end_of_worker
    rebuilds = _all(events, "rebuild", after=t_start, before=walk_end)
    timeline = {
        "capture_id": capture_id,
        "found": True,
        "captures": captures,
        "stream_start": _event(started),
        "stop": _event(stop),
        "tower_summary": _literal(summary.groups["dict"]) if summary else None,
        "tower_summary_line": summary.line if summary else None,
        "tower_windows": [{"line": e.line, "t": e.t, **(_literal(e.groups["dict"]) or {})} for e in windows],
        "tower_totals": window_totals([_literal(e.groups["dict"]) or {} for e in windows]),
        "recorded_frames": sum(int(e.groups["frames"]) for e in stopped) if stopped else None,
        "worker": {"pid": worker_pid, "started": _event(worker), "finished": _event(finished),
                   "seconds": float(finished.groups["s"]) if finished else None},
        "world_id": session.groups["wid"] if session else None,
        "session_id": session.groups["sid"] if session else None,
        "rebuilds": [{"t": e.t, "line": e.line, "n": int(e.groups["n"]), "keyframes": int(e.groups["kf"]),
                      "poses": int(e.groups["poses"]), "points": int(e.groups["points"]),
                      "segments": int(e.groups["segments"]), "seconds": float(e.groups["s"])}
                     for e in rebuilds],
        "rebuilds_failed": len(_all(events, "rebuild_failed", after=t_start, before=walk_end)),
        "background_solves": [{"t": e.t, "line": e.line, "n": int(e.groups["n"]),
                               "keyframes": int(e.groups["kf"]), "pid": e.groups["pid"]}
                              for e in _all(events, "bg_solve", after=t_start, before=t0)],
        "live_surfaces": [{"t": e.t, "line": e.line, "n": int(e.groups["n"]), "pid": e.groups["pid"]}
                          for e in _all(events, "live_surface", after=t_start, before=t0)],
    }
    if t0 is None:
        return timeline
    horizon = walk_end
    timeline["bg_solve_wait"] = _event(_first(events, "bg_solve_wait", after=t0, before=horizon))
    launched = _first(events, "final_launched", after=t0, before=horizon)
    timeline["final_launched"] = _event(launched)
    done = _first(events, "final_done", after=t0, before=horizon)
    timeline["final_done"] = _event(done)
    timeline["final_failed"] = _event(_first(events, "final_failed", after=t0, before=horizon))
    timeline["session_finished"] = _event(_first(
        events, "session_finished", after=t0, before=horizon,
        where=lambda e: timeline["session_id"] is None or e.groups["sid"] == timeline["session_id"]))
    sid = timeline["session_id"]
    timeline["transients"] = [_event(e) for e in _all(
        events, "transients", after=t0, before=horizon, where=lambda e: sid is None or e.groups["sid"] == sid)]
    timeline["consistency"] = [_event(e) for e in _all(events, "consistency", after=t0, before=horizon)]
    built = _first(events, "appearance_built", after=t0, before=horizon,
                   where=lambda e: sid is None or e.groups["sid"] == sid)
    timeline["appearance_built"] = _event(built)
    if built is not None:
        timeline["appearance_seconds"] = _literal(built.groups["dict"])
    timeline["phone_left"] = _event(_first(events, "worker_asked_to_stop", after=t0, before=horizon,
                                           where=lambda e: e.groups["pid"] == worker_pid))
    # The chore runs that follow the builder's exit, one after another.
    chores = []
    cursor = end_of_worker
    while cursor is not None:
        started_chore = _first(events, "chore_started", after=cursor, before=cursor + CHORE_CHAIN_GAP_S)
        if started_chore is None:
            break
        interrupted = _first(events, "recording_started", after=cursor, before=started_chore.t)
        if interrupted is not None:
            break
        exited = _first(events, "chore_exited", after=started_chore.t,
                        where=lambda e: e.groups["pid"] == started_chore.groups["pid"])
        chores.append({"name": started_chore.groups["name"], "pid": started_chore.groups["pid"],
                       "why": started_chore.groups["why"], "started": _event(started_chore),
                       "exited": _event(exited),
                       "outcome": exited.groups["outcome"] if exited else None,
                       "seconds": float(exited.groups["s"]) if exited else None})
        if exited is None:
            break
        cursor = exited.t
    timeline["chores"] = chores
    return timeline


def _event(event):
    if event is None:
        return None
    out = {"t": event.t, "line": event.line}
    out.update({k: v for k, v in event.groups.items() if k != "dict"})
    return out


WINDOW_SUM_KEYS = ("frames_received", "frames_rejected", "tx_seq_gap_total", "backpressure_drops",
                   "frame_processing_errors", "bytes_received")


def window_totals(summaries: list) -> dict | None:
    """The walk's Tower-side totals over every measurement window: sums of
    the counters, the max of `receive_to_result_ms_max`, and which windows."""
    if not summaries:
        return None
    totals: dict = {"windows": len(summaries),
                    "end_reasons": [s.get("end_reason") for s in summaries]}
    for key in WINDOW_SUM_KEYS:
        values = [s.get(key) for s in summaries if isinstance(s.get(key), (int, float))]
        totals[key] = sum(values) if values else None
    peaks = [s.get("receive_to_result_ms_max") for s in summaries
             if isinstance(s.get("receive_to_result_ms_max"), (int, float))]
    totals["receive_to_result_ms_max"] = max(peaks) if peaks else None
    return totals


# -- the store -------------------------------------------------------------------


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _pick(mapping, *keys):
    if not isinstance(mapping, dict):
        return None
    return {k: mapping.get(k) for k in keys if k in mapping}


def store_facts(world_root, world_id, session_id) -> dict:
    """What the store says about the walk's session. Reads only."""
    if world_root is None or world_id is None or session_id is None:
        return {"available": False}
    world = Path(world_root) / "worlds" / world_id
    session = _read_json(world / "sessions" / session_id / "session.json")
    facts: dict = {"available": session is not None, "world_dir": str(world), "session_id": session_id}
    if isinstance(session, dict):
        facts["session"] = {
            **(_pick(session, "frames_observed", "keyframes_accepted", "rejected_by_reason",
                     "started_at", "ended_at", "end_reason", "redaction", "backend_id") or {}),
            "finalization": session.get("finalization"),
            "stages": session.get("stages"),
        }
    solution = _read_json(world / "solve" / session_id / "solution.json")
    if isinstance(solution, dict):
        gate = solution.get("gate") if isinstance(solution.get("gate"), dict) else {}
        consensus = gate.get("consensus") if isinstance(gate.get("consensus"), dict) else None
        anchor = gate.get("anchor_verify") if isinstance(gate.get("anchor_verify"), dict) else None
        transients = solution.get("transients") if isinstance(solution.get("transients"), dict) else {}
        facts["solution"] = {
            "solved_at": solution.get("solved_at"),
            "timing": solution.get("timing"),
            "solve": _pick(solution.get("solve"), "seed", "threads", "seeded", "masking", "final"),
            "transients": {
                **(_pick(transients, "state", "computed", "cache_hits", "gpu_peak_mb", "device") or {}),
                "seconds": _pick(transients.get("seconds"), "gdsam.gdino", "gdsam.sam",
                                 "oneformer.oneformer", "write", "total"),
            },
            "gate_state": gate.get("state"),
            "gate_seconds": gate.get("seconds"),
            "depth": _pick(gate.get("depth"), "state", "backend", "frames", "seconds", "predictions"),
            "metric_scale_seconds": (gate.get("metric_scale") or {}).get("seconds")
            if isinstance(gate.get("metric_scale"), dict) else None,
            "consensus": None if consensus is None else {
                **(_pick(consensus, "requested", "state", "seeds", "chosen", "votes") or {}),
                "draws": [_pick(d, "draw", "seed", "map_s", "gate_s", "gate_state", "votes",
                                "agreement", "predictions")
                          for d in consensus.get("draws") or [] if isinstance(d, dict)],
            },
            "anchor_verify_seconds": anchor.get("seconds") if anchor else None,
        }
    surface = _read_json(world / "surface" / session_id / "status.json")
    if isinstance(surface, dict):
        result = surface.get("result") if isinstance(surface.get("result"), dict) else {}
        facts["surface"] = {"state": surface.get("state"), "updated_at": surface.get("updated_at"),
                            "seconds": result.get("seconds"), "frames_used": result.get("frames_used"),
                            "frames_offered": result.get("frames_offered")}
    areas = []
    for record_path in sorted((world / "areas").glob("*/record.json")):
        record = _read_json(record_path)
        if not isinstance(record, dict) or record.get("session_id") != session_id:
            continue
        stages = record.get("stages") if isinstance(record.get("stages"), dict) else {}
        areas.append({"area_id": record.get("area_id"), "updated_at": record.get("updated_at"),
                      "stages": {name: _pick(stage, "state", "started_at", "updated_at")
                                 for name, stage in stages.items() if isinstance(stage, dict)}})
    facts["areas"] = areas
    timing = _read_json(world / "sessions" / session_id / "stage_timing.json")
    facts["stage_timing"] = summarize_stage_timing(timing)
    facts["keyframes"] = keyframe_facts(world / "sessions" / session_id)
    return facts


def read_jsonl(path) -> list:
    """The objects of a JSON-lines file; a torn LAST line is dropped."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            if number == len(lines):
                break
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def keyframe_facts(session_dir) -> dict | None:
    """The session's keyframe sequence and its observe lag (review C22 H2).

    IDENTITY is the ordered `(source_seq, segment_index)` of `keyframes.jsonl`,
    never `keyframe_id`: that embeds the session id, which differs between
    runs. Selection is content-driven, so two runs of one walk that differ
    here leaked timing into the builder.

    OBSERVE LAG is `events.jsonl` `keyframe_accepted.at` minus that
    keyframe's `received_at` (joined on `keyframe_id`): how long after the
    Tower received a frame the builder had observed and accepted it. It is
    keyframe-level; a true per-frame lag would need an `observed_at` in
    `frames_quality.jsonl`, which the Tower does not write.
    """
    session_dir = Path(session_dir)
    rows = read_jsonl(session_dir / "keyframes.jsonl")
    if not rows:
        return None
    sequence = [[row.get("source_seq"), row.get("segment_index")] for row in rows]
    received = {row.get("keyframe_id"): row.get("received_at") for row in rows}
    lags, unmatched = [], 0
    for event in read_jsonl(session_dir / "events.jsonl"):
        if event.get("kind") != "keyframe_accepted":
            continue
        keyframe_id = (event.get("payload") or {}).get("keyframe_id")
        at, got = event.get("at"), received.get(keyframe_id)
        if isinstance(at, (int, float)) and isinstance(got, (int, float)):
            lags.append(at - got)
        else:
            unmatched += 1
    return {
        "count": len(sequence),
        "sequence": sequence,
        "sha256": hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode()).hexdigest(),
        "observe_lag_s": distribution(lags),
        "observe_lag_unmatched": unmatched,
    }


_GLOG = re.compile(r"^[IWEF](\d{8}) (\d{2}:\d{2}:\d{2})\.(\d+)\s")
DRAW_MARKER_GAP_S = 30.0


def solve_log_markers(path, after=None) -> list:
    """GLOMAP's own (glog) lines in `solve.log` after `after`, clustered: the
    last line of each burst more than 30 s from the next. On walk 5 each
    consensus draw left exactly one burst as its map ended (PROFILE-walk5 §4)."""
    stamps = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                match = _GLOG.match(raw)
                if not match:
                    continue
                day, clock, frac = match.groups()
                t = time.mktime(time.strptime(f"{day} {clock}", "%Y%m%d %H:%M:%S")) + float(f"0.{frac}")
                if after is None or t >= after:
                    stamps.append(t)
    except OSError:
        return []
    ends = []
    for index, t in enumerate(stamps):
        if index + 1 == len(stamps) or stamps[index + 1] - t > DRAW_MARKER_GAP_S:
            ends.append(t)
    return ends


PRE_MAP_KEYS = ("prepare_s", "masks_s", "freeze_s", "extract_s", "match_s")


def draw_zero(snapshots: list, facts: dict, launched_t) -> dict | None:
    """Draw 0 of a consensus final solve: its map and gate times and when it
    was published first (review C22 M6).

    1. A SNAPSHOT (`<run>/solution-snapshots/`, taken by the client while it
       followed the settle): the first version whose consensus is `deferred`
       IS draw 0's early publish. Exact.
    2. Otherwise an ESTIMATE from the finished store, for runs made before
       the snapshots existed. Draw 0 starts mapping after the solve's
       prepare, masks, freeze, extract and match (`timing`, which every draw
       inherits from draw 0) from the final solve's launch. It ends at its
       end-of-map marker in `solve.log` when there is one burst per draw,
       calibrated on the chosen draw (whose `solved_at` is exact); else at
       `sparse/`'s mtime, which holds draw 0's model. Draw 0 was published
       just before draw 1 began mapping: draw 1's end minus its `map_s`.
    """
    for snap in snapshots or []:
        doc = snap.get("doc") or {}
        consensus = ((doc.get("gate") or {}).get("consensus") or {}) if isinstance(doc.get("gate"), dict) else {}
        if consensus.get("state") != "deferred":
            continue
        timing = doc.get("timing") or {}
        return {"source": "snapshot", "file": snap.get("file"), "map_s": timing.get("map_s"),
                "gate_total_s": timing.get("gate_s"), "solved_at": doc.get("solved_at"),
                "published_at": snap.get("mtime")}
    solution = (facts or {}).get("solution") or {}
    consensus = solution.get("consensus") or {}
    draws = consensus.get("draws") or []
    if not draws or launched_t is None:
        return None
    timing = solution.get("timing") or {}
    world = Path(facts["world_dir"])
    session_id = facts.get("session_id")
    estimate: dict = {"source": "estimate", "map_s": None, "published_at": None}
    pre = [timing.get(key) for key in PRE_MAP_KEYS]
    if any(not isinstance(value, (int, float)) for value in pre):
        return {**estimate, "how": "the solve's timing lacks a pre-map stage"}
    map_start = launched_t + sum(pre)
    estimate["map_start"] = round(map_start, 3)
    solve_dir = world / "solve" / session_id if session_id else None
    ends = solve_log_markers(solve_dir / "solve.log", launched_t) if solve_dir else []
    chosen = (consensus.get("chosen") or {}).get("draw") if isinstance(consensus.get("chosen"), dict) else None
    if len(ends) == len(draws):
        offset = 0.0
        if isinstance(chosen, int) and 0 <= chosen < len(ends) and isinstance(solution.get("solved_at"), (int, float)):
            offset = solution["solved_at"] - ends[chosen]
        estimate["how"] = (f"solve.log: {len(ends)} end-of-map bursts for {len(draws)} draws, "
                           f"calibrated {offset:+.2f} s on chosen draw {chosen}")
        estimate["markers"] = [round(t, 3) for t in ends]
        end_0 = ends[0] + offset
        if len(draws) > 1 and isinstance(draws[1].get("map_s"), (int, float)):
            estimate["published_at"] = round(ends[1] + offset - draws[1]["map_s"], 3)
    else:
        try:
            end_0 = (solve_dir / "sparse").stat().st_mtime
        except (OSError, TypeError):
            return {**estimate, "how": f"no end-of-map marker per draw ({len(ends)} for {len(draws)}) "
                                       "and no sparse/"}
        estimate["how"] = (f"sparse/ mtime (draw 0's model); solve.log had {len(ends)} end-of-map "
                           f"bursts for {len(draws)} draws")
    estimate["map_s"] = round(end_0 - map_start, 1)
    return estimate


def read_snapshots(run_dir) -> list:
    """The client's `solution-snapshots/` of a run, in order, with their content."""
    if run_dir is None:
        return []
    folder = Path(run_dir) / "solution-snapshots"
    index = _read_json(folder / "index.json")
    items = []
    for item in index if isinstance(index, list) else []:
        if not isinstance(item, dict):
            continue
        doc = _read_json(folder / str(item.get("file")))
        if isinstance(doc, dict):
            items.append({**item, "doc": doc})
    return items


def live_surface_latency(live_surfaces: list, watch: list, t0) -> dict:
    """Each live surface's launch (log) to `ok` (the client's SurfaceWatch).

    A live surface is launched when a background solve lands, so this is the
    solve-landed -> surface-ok latency. The store alone cannot give it: each
    live surface and then the final one overwrite `surface/<s>/status.json`,
    and the Tower logs a launch but not an end.
    """
    if not live_surfaces:
        return {"computable": True, "launches": 0, "surfaces": []}
    if not watch:
        return {"computable": False, "launches": len(live_surfaces), "surfaces": [],
                "why": "no surface watch in this run (client.json surface_watch): the store keeps only "
                       "the last surface status.json and the log has launch lines only"}
    rows = []
    for index, launch in enumerate(live_surfaces):
        after = launch["t"]
        before = live_surfaces[index + 1]["t"] if index + 1 < len(live_surfaces) else None
        done = None
        for item in watch:
            at = item.get("updated_at") or item.get("t")
            if item.get("state") != "ok" or item.get("kind") == "final" or at is None:
                continue
            if at >= after and (before is None or at <= before):
                done = at
                break
        rows.append({"n": launch.get("n"), "launched": after, "ok_at": done,
                     "latency_s": None if done is None else round(done - after, 2),
                     "killed_at_stop": done is None and before is None and t0 is not None})
    latencies = [row["latency_s"] for row in rows if row["latency_s"] is not None]
    return {"computable": True, "launches": len(rows), "surfaces": rows,
            "latency_s": distribution(latencies)}


def summarize_stage_timing(doc) -> dict | None:
    """`stage_timing.json` (TOWER_WORLD_STAGE_TIMING=on), per finish and stage."""
    if not isinstance(doc, dict) or not isinstance(doc.get("finishes"), list):
        return None
    finishes = []
    for finish in doc["finishes"]:
        if not isinstance(finish, dict):
            continue
        stages = {}
        for name, records in (finish.get("stages") or {}).items():
            if not isinstance(records, list):
                continue
            rows = []
            for record in records:
                if not isinstance(record, dict):
                    continue
                cache = record.get("cache") or {}
                rows.append({
                    "wall_s": round((record.get("wall_ms") or 0) / 1000.0, 2),
                    "status": record.get("status"),
                    "seed": record.get("seed"),
                    "draw_index": record.get("draw_index"),
                    "cache": {k: v for k, v in cache.items()
                              if isinstance(v, dict) and (v.get("hits") or v.get("misses"))},
                })
            stages[name] = rows
        finishes.append({"finish_id": finish.get("finish_id"), "context": finish.get("context"),
                         "started_at": finish.get("started_at"), "stages": stages})
    return {"schema": doc.get("schema"), "finishes": finishes}


# -- samples -----------------------------------------------------------------------


def read_samples(path) -> list:
    rows = []
    if path is None or not Path(path).exists():
        return rows
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if key == "busy":
                    parsed[key] = value or ""
                    continue
                try:
                    parsed[key] = float(value) if value not in (None, "") else None
                except ValueError:
                    parsed[key] = None
            rows.append(parsed)
    return rows


def window_use(samples: list, start, end) -> dict | None:
    """Mean/max of the test Tower's cores and the GPU between two times."""
    if not samples or start is None or end is None:
        return None
    inside = [s for s in samples if s.get("t") is not None and start <= s["t"] <= end]
    if not inside:
        return None

    def stat(key):
        values = [s[key] for s in inside if s.get(key) is not None]
        if not values:
            return None
        return {"mean": round(sum(values) / len(values), 1), "max": round(max(values), 1)}

    labels = {}
    for sample in inside:
        for item in (sample.get("busy") or "").split(";"):
            if ":" in item:
                name, cores = item.rsplit(":", 1)
                labels[name] = labels.get(name, 0.0) + float(cores)
    top = sorted(labels.items(), key=lambda kv: -kv[1])[:3]
    return {"samples": len(inside), "cores": stat("tree_cores"), "gpu": stat("gpu_util_pct"),
            "gpu_mem_mb": stat("gpu_mem_mb"), "sys_cpu": stat("sys_cpu_pct"),
            "busy": [name for name, _ in top]}


# -- the report ----------------------------------------------------------------------


def _minutes(a, b):
    if a is None or b is None:
        return None
    return round((b - a) / 60.0, 2)


def _t(event):
    return None if event is None else event.get("t")


def distribution(values) -> dict:
    """count / mean / p50 / p95 / p99 / max; nearest-rank on (n - 1)."""
    values = sorted(values)
    if not values:
        return {"count": 0}

    def pick(q):
        return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]

    return {"count": len(values), "mean": round(sum(values) / len(values), 3),
            "p50": round(pick(0.5), 3), "p95": round(pick(0.95), 3), "p99": round(pick(0.99), 3),
            "max": round(values[-1], 3)}


def live_safety(*, client: dict, timeline: dict, session: dict, keyframes: dict | None,
                surfaces: dict, rebuild_seconds: list, cadence: list) -> dict:
    """The C19 F8 live-safety gate, one row per check (review C22 H2).

    PASS / FAIL where the check has a required value; INFO where F8 asks for
    "no regression beyond old-path noise" (that is `--compare`'s job across
    N runs); n/a where this run holds no record of it (a report on a real
    walk's log has no client). Everything is read after the fact, from the
    Tower's log, the store and the run's `client.json`.
    """
    totals = timeline.get("tower_totals") or {}
    stream = client.get("stream") or {}
    schedule = client.get("schedule") or {}
    received = totals.get("frames_received")
    sent = stream.get("frames_sent") if stream else None
    rows = []

    def row(check, value, required, result):
        rows.append({"check": check, "value": value, "required": required, "result": result})

    def equal(a, b):
        if a is None or b is None:
            return "n/a"
        return "PASS" if a == b else "FAIL"

    def zero(value):
        if value is None:
            return "n/a"
        return "PASS" if value == 0 else "FAIL"

    windows = totals.get("windows")
    row("Tower measurement windows summed", f"{windows} ({', '.join(map(str, totals.get('end_reasons') or []))})"
        if windows else None, "every window of the walk", "INFO" if windows else "n/a")
    if stream:
        row("frames sent (client) = frames scheduled", f"{sent} / {schedule.get('frames')}", "equal",
            equal(sent, schedule.get("frames")))
        row("frames received (Tower, all windows) = frames sent", f"{received} / {sent}", "equal",
            equal(received, sent))
    else:
        row("frames received (Tower, all windows) = frames recorded", f"{received} / "
            f"{timeline.get('recorded_frames')}", "equal", equal(received, timeline.get("recorded_frames")))
    observed = session.get("frames_observed") if session else None
    row("frames observed (builder) = frames received", f"{observed} / {received}", "equal",
        equal(observed, received))
    for key, label in (("tx_seq_gap_total", "tx_seq gaps"), ("backpressure_drops", "backpressure drops"),
                       ("frames_rejected", "frames rejected"),
                       ("frame_processing_errors", "frame processing errors")):
        row(f"{label} (all windows)", totals.get(key), "0", zero(totals.get(key)))
    if stream:
        errors = stream.get("frame_errors") or {}
        unanswered = stream.get("unanswered")
        row("every frame answered (client): unanswered, frame_error replies",
            f"{unanswered}, {errors or '{}'}", "0, {}",
            "PASS" if unanswered == 0 and not errors else "FAIL")
    else:
        row("every frame answered (client)", None, "0, {}", "n/a")
    row("receive_to_result_ms_max (all windows)", totals.get("receive_to_result_ms_max"),
        "no regression (--compare)", "INFO" if totals.get("receive_to_result_ms_max") is not None else "n/a")
    accepted = session.get("keyframes_accepted") if session else None
    if keyframes:
        row("keyframes accepted; sequence (source_seq, segment_index)",
            f"{accepted} accepted, {keyframes['count']} in keyframes.jsonl, sha256 {keyframes['sha256'][:16]}",
            "identical to baseline (--compare)",
            "INFO" if accepted in (None, keyframes["count"]) else "FAIL")
        lag = keyframes.get("observe_lag_s") or {}
        row("observe lag, keyframe accepted - received (s)",
            _dist_text(lag) + (f"; {keyframes.get('observe_lag_unmatched')} unmatched"
                               if keyframes.get("observe_lag_unmatched") else ""),
            "no regression (--compare)", "INFO" if lag.get("count") else "n/a")
    else:
        row("keyframes accepted; sequence", accepted, "identical to baseline (--compare)", "n/a")
        row("observe lag (s)", None, "no regression (--compare)", "n/a")
    row("rebuild latency (s)", _dist_text(distribution(rebuild_seconds)), "no regression (--compare)",
        "INFO" if rebuild_seconds else "n/a")
    row("rebuild cadence (s)", _dist_text(distribution(cadence)), "no regression (--compare)",
        "INFO" if cadence else "n/a")
    solves = timeline.get("background_solves") or []
    row("background-solve horizons (keyframes at launch)", [s["keyframes"] for s in solves],
        "unchanged (--compare)", "INFO")
    if surfaces.get("computable"):
        latency = surfaces.get("latency_s") or {}
        killed = sum(1 for s in surfaces.get("surfaces") or [] if s.get("killed_at_stop"))
        row("live surfaces: launches; launch -> ok latency (s)",
            f"{surfaces.get('launches')} launched; " + _dist_text(latency)
            + (f"; {killed} killed at Stop" if killed else ""),
            "no regression (--compare)", "INFO")
    else:
        row("live surfaces: launches; launch -> ok latency", f"{surfaces.get('launches')} launched; latency "
            "not computable post hoc", "no regression (--compare)", "n/a")
    watch = client.get("live_tower_watch") or {}
    if watch:
        seen = watch.get("states_seen") or []
        contended = [s for s in seen if s not in ("idle", "down")]
        row(":8000 during the run", ", ".join(seen) + (f"; busy from {_clock(watch.get('busy_since'))}"
                                                     if watch.get("busy_since") else ""),
            "idle or down throughout", "FAIL" if contended else "PASS")
    else:
        row(":8000 during the run", None, "idle or down throughout", "n/a")
    judged = [r["result"] for r in rows if r["result"] in ("PASS", "FAIL")]
    result = "FAIL" if "FAIL" in judged else ("PASS" if judged else "n/a")
    return {"result": result, "rows": rows, "totals": totals}


def _dist_text(stats: dict) -> str:
    if not stats or not stats.get("count"):
        return "none"
    return (f"n {stats['count']}, p50 {stats.get('p50')}, p95 {stats.get('p95')}, "
            f"p99 {stats.get('p99')}, max {stats.get('max')}")


def build_report(*, tower_log, tower_out_log=None, world_root=None, capture_id=None,
                 client=None, samples=None, label=None, run_dir=None) -> dict:
    """Everything, as one JSON-able dict. `render_markdown` draws it.

    `run_dir` is the run's `--out` (read): its `solution-snapshots/`.
    """
    if client is None:
        client = {}
    if capture_id is None and client.get("tower_captures"):
        capture_id = client["tower_captures"][0]
    events = scan_log(tower_log)
    timeline = walk_timeline(events, capture_id)
    facts = store_facts(world_root, timeline.get("world_id"), timeline.get("session_id"))
    sample_rows = read_samples(samples) if samples is not None else []
    chore_reports = scan_chore_reports(tower_out_log)
    session = (facts.get("session") or {}) if facts.get("available") else {}
    finalization = session.get("finalization") or {}
    stages = session.get("stages") or {}
    snapshots = read_snapshots(run_dir)
    zero = draw_zero(snapshots, facts, _t(timeline.get("final_launched"))) if facts.get("solution") else None

    t0 = _t(timeline.get("stop"))
    rows = []
    last_sample = sample_rows[-1]["t"] if sample_rows else None

    def add(number, stage, start, end, *, wait="", detail="", evidence="", use=True, child=False):
        rows.append({
            "n": number, "stage": stage, "start": start, "end": end,
            "minutes": _minutes(start, end), "wait": wait, "detail": detail,
            "evidence": evidence, "child": child,
            # A row still open when the run ended is measured to the last sample.
            "use": window_use(sample_rows, start, end if end is not None else last_sample) if use else None,
        })

    milestones = {}
    if t0 is not None:
        catchup_after = [r for r in timeline.get("rebuilds", []) if r["t"] >= t0]
        finalization_started = finalization.get("started_at") or session.get("ended_at")
        catchup_end = finalization_started or (catchup_after[-1]["t"] if catchup_after else None)
        add("1", "Live catch-up: the builder drains the journal and closes the session", t0, catchup_end,
            detail=f"{len(catchup_after)} rebuild(s) after Stop",
            evidence=f"L{timeline['stop']['line']}")
        wait = timeline.get("bg_solve_wait")
        launched, done = timeline.get("final_launched"), timeline.get("final_done")
        cursor = catchup_end
        gap_end = _t(launched) or _t(wait)
        if catchup_end is not None and gap_end is not None and gap_end - catchup_end > 1.0:
            if wait is not None:
                stage = (f"Wait for background solve pid {wait['pid']} ({wait['s']} s), then terminate it "
                         "and launch the final solve")
                evidence = f"L{wait['line']}"
            else:
                stage = ("Before the final solve: the last background solve finishes (no terminate line; "
                         "it ended inside the wait)")
                evidence = ""
            add("2", stage, catchup_end, gap_end, wait="waits on another process", evidence=evidence)
            cursor = gap_end
        if launched is not None:
            what = (f"solved={done['solved']} solver={done['solver']} posed={done['posed']}/{done['of']}"
                    if done else ("failed" if timeline.get("final_failed") else "not finished"))
            solution = facts.get("solution") or {}
            if solution:
                consensus_state = (solution.get("consensus") or {}).get("state")
                what += (f"; solution.json solved_at {_clock(solution.get('solved_at'))}, "
                         f"consensus {consensus_state}")
            add("3", f"Final global solve, pid {launched['pid']}", launched["t"], _t(done), detail=what,
                evidence=f"L{launched['line']}" + (f"-L{done['line']}" if done else ""))
            timing = solution.get("timing") or {}
            consensus = solution.get("consensus") or {}
            chosen = consensus.get("chosen") if isinstance(consensus.get("chosen"), dict) else {}
            whose = (f"the chosen draw's (draw {chosen.get('draw')}, seed {chosen.get('seed')})"
                     if chosen else "the chosen draw's")
            for key in ("prepare_s", "masks_s", "freeze_s", "extract_s", "match_s", "map_s", "gate_s"):
                if timing.get(key) is not None:
                    add("", f"solve {key[:-2]} (solution.json timing: {whose} map/gate)", None, None,
                        detail=f"{timing[key]} s", child=True, use=False)
            for draw in consensus.get("draws") or []:
                map_s = draw.get("map_s")
                note = ""
                if draw.get("draw") == 0 and map_s is None and zero and zero.get("map_s") is not None:
                    map_s = zero["map_s"]
                    note = (" (draw-0 snapshot)" if zero["source"] == "snapshot"
                            else f" (estimated: {zero.get('how')})")
                add("", f"consensus draw {draw.get('draw')} (seed {draw.get('seed')})"
                        + (" CHOSEN" if chosen and draw.get("draw") == chosen.get("draw") else ""), None, None,
                    detail=f"map {map_s} s{note}, gate {draw.get('gate_s')} s", child=True, use=False)
            if zero and zero.get("published_at") is not None:
                milestones["draw0_published" if zero["source"] == "snapshot"
                           else "draw0_published_estimated"] = zero["published_at"]
            cursor = _t(done) or cursor
        finished = timeline.get("session_finished")
        terminal = finalization.get("state") not in (None, "pending", "running")
        fin_updated = finalization.get("updated_at") if terminal else None
        fin_end = _t(finished) or fin_updated
        if fin_end is not None:
            add("4", f"Final build; finalization: {finalization.get('state')} "
                     f"(final_solve {finalization.get('final_solve')})", cursor, fin_end,
                evidence=f"L{finished['line']}" if finished else "session.json")
        if finalization.get("state") == "complete":
            milestones["finalization_complete"] = fin_updated or fin_end
        for number, name in (("5", "surface"), ("6", "appearance"), ("7", "dense")):
            stage = stages.get(name)
            if not isinstance(stage, dict):
                continue
            label_text = {"surface": "Final room surface", "appearance": "Room appearance (photos)",
                          "dense": "Dense"}[name]
            open_stage = stage.get("state") in ("running", "pending")
            add(number, f"{label_text}: {stage.get('state')}", stage.get("started_at"),
                None if open_stage else stage.get("updated_at"), detail=stage.get("detail") or "",
                evidence="session.json stages")
            if name == "surface":
                for key, value in ((facts.get("surface") or {}).get("seconds") or {}).items():
                    add("", f"surface {key}", None, None, detail=f"{value} s", child=True, use=False)
            if name == "appearance" and timeline.get("appearance_seconds"):
                parts = timeline["appearance_seconds"]
                add("", "appearance breakdown", None, None,
                    detail=", ".join(f"{k} {v}" for k, v in parts.items()), child=True, use=False)
            if stage.get("state") == "ok":
                milestones[f"room_{name}_ok"] = stage.get("updated_at")
        worker = timeline.get("worker") or {}
        exit_t = _t(worker.get("finished"))
        last_stage_end = max([s.get("updated_at") for s in stages.values()
                              if isinstance(s, dict) and s.get("updated_at")] or [fin_end or t0])
        if last_stage_end is None:
            last_stage_end = t0
        if exit_t is not None:
            add("8", "Worker exit", last_stage_end, exit_t,
                evidence=f"L{worker['finished']['line']}")
            milestones["worker_exit"] = exit_t
        previous = exit_t
        for index, chore in enumerate(timeline.get("chores") or []):
            started = _t(chore["started"])
            if previous is not None and started is not None:
                add("9" if index == 0 else f"9.{index}", "Tower idle detection, then chore spawn",
                    previous, started, wait="waits for the worker to exit (by design)", use=False)
            add("10" if index == 0 else f"10.{index}",
                f"Chore {chore['name']} pid {chore['pid']}: {chore['outcome']}", started,
                _t(chore["exited"]), evidence=f"L{chore['started']['line']}")
            previous = _t(chore["exited"])
        settled = previous if (timeline.get("chores") or exit_t) else None
        if settled is not None:
            milestones["settled"] = settled
            add("Σ", "Stop to settled", t0, settled, use=False)

    photos = milestones.get("room_appearance_ok")
    photos_minutes = _minutes(t0, photos)
    verdict = ("PASS" if photos_minutes is not None and photos_minutes <= HARD_MAX_MINUTES
               else "FAIL" if photos_minutes is not None else "NOT REACHED")

    rebuild_seconds = [r["seconds"] for r in timeline.get("rebuilds", [])]
    rebuild_times = [r["t"] for r in timeline.get("rebuilds", [])]
    cadence = [b - a for a, b in zip(rebuild_times, rebuild_times[1:])]
    t_start = _t(timeline.get("stream_start"))
    during_walk = window_use(sample_rows, t_start, t0)
    after_stop = window_use(sample_rows, t0, milestones.get("settled") or
                            (sample_rows[-1]["t"] if sample_rows else None))
    busy_gpu = _minutes_where(sample_rows, t0, lambda s: (s.get("gpu_util_pct") or 0) >= 10)
    one_core = _minutes_where(sample_rows, t0, lambda s: s.get("tree_cores") is not None
                              and s["tree_cores"] <= 1.5 and (s.get("gpu_util_pct") or 0) < 10)

    areas_from_chore = []
    for report in chore_reports:
        for item in report.get("finished") or []:
            if item.get("session_id") != timeline.get("session_id"):
                continue
            for area in item.get("areas") or []:
                surface = ((area.get("stages") or {}).get("surface") or {})
                appearance = ((area.get("stages") or {}).get("appearance") or {})
                areas_from_chore.append({
                    "area_id": area.get("area_id"), "prepare_s": area.get("prepare_s"),
                    "surface_state": surface.get("state"), "surface_seconds": surface.get("seconds"),
                    "appearance_state": appearance.get("state"),
                    "appearance_total": (appearance.get("seconds") or {}).get("total")
                    if isinstance(appearance.get("seconds"), dict) else None,
                })

    keyframes = facts.get("keyframes") if facts.get("available") else None
    surfaces = live_surface_latency(timeline.get("live_surfaces") or [], client.get("surface_watch") or [], t0)
    safety = live_safety(client=client, timeline=timeline, session=session, keyframes=keyframes,
                         surfaces=surfaces, rebuild_seconds=rebuild_seconds, cadence=cadence)

    return {
        "report": "c22-live-replay/2",
        "label": label or client.get("label"),
        "generated_at": round(time.time(), 3),
        "inputs": {"tower_log": str(tower_log), "tower_out_log": None if tower_out_log is None
                   else str(tower_out_log), "world_root": None if world_root is None else str(world_root),
                   "samples": None if samples is None else str(samples),
                   "run_dir": None if run_dir is None else str(run_dir)},
        "harness": client.get("harness"),
        "hard_max_minutes": HARD_MAX_MINUTES,
        "verdict": {"stop_to_room_with_photos_min": photos_minutes, "result": verdict,
                    "stop_to_settled_min": _minutes(t0, milestones.get("settled")),
                    "stop_to_finalization_min": _minutes(t0, milestones.get("finalization_complete"))},
        "live_safety": safety,
        "keyframes": keyframes,
        "solve_draw_0": zero,
        "live_surfaces_latency": surfaces,
        "client": {k: client.get(k) for k in ("outcome", "speed", "first_seconds", "walk", "schedule",
                                              "stream", "phone_fetches", "handshake", "session_start",
                                              "live_tower_at_start", "live_tower_watch", "target_listener",
                                              "aborted", "t0", "stopped_at")
                   if k in client},
        "tower_walk": {
            "captures": timeline.get("captures"),
            "world_id": timeline.get("world_id"),
            "session_id": timeline.get("session_id"),
            "stream_start": timeline.get("stream_start"),
            "stop": timeline.get("stop"),
            "summary": timeline.get("tower_summary"),
            "summary_line": timeline.get("tower_summary_line"),
            "windows": timeline.get("tower_windows"),
            "totals": timeline.get("tower_totals"),
            "frames_observed": session.get("frames_observed"),
            "keyframes_accepted": session.get("keyframes_accepted"),
            "rejected_by_reason": session.get("rejected_by_reason"),
            "rebuilds": {"count": len(rebuild_seconds), "failed": timeline.get("rebuilds_failed"),
                         "seconds": distribution(rebuild_seconds),
                         "cadence_s": distribution(cadence),
                         "after_stop": len([r for r in timeline.get("rebuilds", [])
                                            if t0 is not None and r["t"] >= t0]),
                         "last_before_stop_keyframes": max(
                             [r["keyframes"] for r in timeline.get("rebuilds", [])
                              if t0 is None or r["t"] < t0] or [0])},
            "background_solves": timeline.get("background_solves"),
            "live_surfaces": timeline.get("live_surfaces"),
            "use_during_walk": during_walk,
        },
        "waterfall": rows,
        "milestones": {k: {"t": v, "minutes_after_stop": _minutes(t0, v)} for k, v in milestones.items()},
        "use_after_stop": after_stop,
        "gpu_busy_minutes_after_stop": busy_gpu,
        "low_cpu_gpu_idle_minutes_after_stop": one_core,
        "store": facts,
        "areas_from_chore": areas_from_chore,
        "phone_view": (client.get("phone_view") or {}),
        "settle": (client.get("settle") or {}),
        "timeline": timeline,
        "samples": len(sample_rows),
    }


def _minutes_where(samples, start, predicate):
    if not samples or start is None:
        return None
    inside = [s for s in samples if s.get("t") is not None and s["t"] >= start]
    total = 0.0
    for a, b in zip(inside, inside[1:]):
        if predicate(a):
            total += b["t"] - a["t"]
    return round(total / 60.0, 2)


def _clock(t):
    if t is None:
        return "—"
    return datetime.datetime.fromtimestamp(round(t, 1)).strftime("%H:%M:%S.%f")[:-5]


def _use_text(use):
    if not use:
        return "—"
    parts = []
    if use.get("cores"):
        parts.append(f"{use['cores']['mean']} cores (max {use['cores']['max']})")
    if use.get("gpu"):
        parts.append(f"GPU {use['gpu']['mean']}% (max {use['gpu']['max']}%)")
    if use.get("busy"):
        parts.append("busy: " + ", ".join(use["busy"]))
    return "; ".join(parts) or "—"


def render_markdown(report: dict) -> str:
    lines = []
    walk = report.get("tower_walk") or {}
    client = report.get("client") or {}
    verdict = report.get("verdict") or {}
    title = report.get("label") or (walk.get("captures") or ["?"])[0]
    lines.append(f"# C22 live replay: {title}")
    lines.append("")
    lines.append(f"Generated {datetime.datetime.fromtimestamp(report['generated_at']).isoformat(timespec='seconds')}"
                 f" by `world_live_replay_report.py`. Times are local. \"min\" is wall time.")
    lines.append("")
    run = report.get("run") or {}
    if run:
        code = run.get("code") or {}
        switches = run.get("switches") or {}
        lines.append(f"Code: `{code.get('git_head') or code.get('tower_dir')}` (py fingerprint "
                     f"{code.get('py_fingerprint')}, stage timing module {'present' if code.get('has_stage_timing') else 'ABSENT'}). "
                     f"Test Tower 127.0.0.1:{run.get('port')}, data root `{run.get('data_root')}`. Switches: "
                     + ", ".join(f"{k}={v}" for k, v in sorted(switches.items())) + ".")
        if code.get("stage_timing_note"):
            lines.append(f"Note: {code['stage_timing_note']}.")
        lines.append(f"Tower stopped at {_clock(run.get('stopped_at'))}; port free after: "
                     f"{run.get('port_free_after')}; leftover processes: {len(run.get('leftover_processes') or [])}.")
        lines.append("")
    harness = run.get("harness") or report.get("harness") or {}
    if harness:
        dirty = harness.get("git_dirty")
        lines.append(f"Harness: `{harness.get('git_head') or 'not a checkout'}`"
                     + (f" with uncommitted changes {dirty}" if dirty else (" (clean)" if dirty == [] else ""))
                     + f"; scripts sha1 `{harness.get('sha1')}`.")
        lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    photos = verdict.get("stop_to_room_with_photos_min")
    lines.append(f"1. **Stop to room with photos: {photos if photos is not None else 'not reached'} min** "
                 f"against the {report['hard_max_minutes']:.0f}-min hard maximum: **{verdict.get('result')}**.")
    lines.append(f"2. Stop to `finalization: complete`: {verdict.get('stop_to_finalization_min')} min. "
                 f"Stop to settled (worker exit and the chore's areas): {verdict.get('stop_to_settled_min')} min.")
    stream = client.get("stream") or {}
    summary = walk.get("summary") or {}
    totals = walk.get("totals") or {}
    if stream or summary or totals:
        sent = (f"{stream.get('frames_sent')} sent, {stream.get('frame_results')} answered, "
                f"{sum((stream.get('frame_errors') or {}).values())} refused, "
                f"at {stream.get('achieved_send_fps')} fps sent. " if stream else "")
        lines.append(f"3. Frames: {sent}The Tower received "
                     f"{totals.get('frames_received')} over {totals.get('windows')} measurement window(s) "
                     f"({summary.get('effective_fps')} fps in the last), rejected "
                     f"{totals.get('frames_rejected')}, tx_seq gaps {totals.get('tx_seq_gap_total')}, "
                     f"backpressure drops {totals.get('backpressure_drops')}, frame processing errors "
                     f"{totals.get('frame_processing_errors')}; the builder observed "
                     f"{walk.get('frames_observed')} and accepted {walk.get('keyframes_accepted')} keyframes.")
    rebuilds = walk.get("rebuilds") or {}
    seconds, cadence = rebuilds.get("seconds") or {}, rebuilds.get("cadence_s") or {}
    lines.append(f"4. Live builder: {rebuilds.get('count')} rebuilds ({rebuilds.get('failed')} failed), "
                 f"latency p50 {seconds.get('p50')} s / p95 {seconds.get('p95')} s / p99 {seconds.get('p99')} s "
                 f"/ max {seconds.get('max')} s, cadence p50 {cadence.get('p50')} s / p95 {cadence.get('p95')} s "
                 f"/ p99 {cadence.get('p99')} s / max {cadence.get('max')} s; "
                 f"{rebuilds.get('after_stop')} after Stop. "
                 f"{len(walk.get('background_solves') or [])} background solves and "
                 f"{len(walk.get('live_surfaces') or [])} live surfaces launched during the walk.")
    if client.get("aborted"):
        lines.append(f"5. **Aborted** at {_clock(client['aborted'].get('t'))}: {client['aborted'].get('reason')}.")
    lines.append("")
    safety = report.get("live_safety") or {}
    lines.append(f"## 2. Live safety (C19 F8): **{safety.get('result')}**")
    lines.append("")
    lines.append("PASS/FAIL where F8 fixes a value; INFO where it asks for \"no regression beyond old-path "
                 "noise\", which `--compare` judges across N >= 3 runs; n/a where this run holds no record "
                 "of it (a real walk's log has no client).")
    lines.append("")
    lines.append("| Check | Value | Required | Result |")
    lines.append("|---|---|---|---|")
    for item in safety.get("rows") or []:
        value = item.get("value")
        value = "—" if value is None else str(value).replace("|", "/")
        lines.append(f"| {item['check']} | {value} | {item['required']} | {item['result']} |")
    zero = report.get("solve_draw_0")
    if zero:
        lines.append("")
        lines.append(f"Draw 0 of the final solve ({zero.get('source')}"
                     + (f": {zero.get('how')}" if zero.get("how") else "") + f"): map {zero.get('map_s')} s"
                     + (f", published {_clock(zero.get('published_at'))}" if zero.get("published_at") else "")
                     + ".")
    lines.append("")
    lines.append("## 3. Waterfall")
    lines.append("")
    stop = walk.get("stop") or {}
    lines.append(f"Stop (T0) = **{_clock(stop.get('t'))}** (log L{stop.get('line')}). "
                 "Observed use is the test Tower's process tree (cores) and the whole GPU, "
                 "from the client's samples inside each row's window.")
    lines.append("")
    lines.append("| # | Stage | Start | End | min | Wait | Detail | Observed use | Evidence |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in report.get("waterfall") or []:
        stage = ("↳ " + row["stage"]) if row.get("child") else row["stage"]
        if row["n"] in ("1", "2", "3", "5", "6", "10", "Σ") and not row.get("child"):
            stage = f"**{stage}**" if row["n"] == "Σ" else stage
        lines.append(f"| {row['n']} | {stage} | {_clock(row['start'])} | {_clock(row['end'])} | "
                     f"{'' if row['minutes'] is None else row['minutes']} | {row.get('wait') or ''} | "
                     f"{str(row.get('detail') or '').replace('|', '/')[:160]} | {_use_text(row.get('use'))} | "
                     f"{row.get('evidence') or ''} |")
    lines.append("")
    lines.append("Milestones (minutes after Stop):")
    for name, value in (report.get("milestones") or {}).items():
        lines.append(f"- {name}: {_clock(value['t'])} (+{value['minutes_after_stop']})")
    lines.append("")
    areas = report.get("areas_from_chore") or []
    if areas:
        lines.append("Areas (from the chore's report):")
        for area in areas:
            lines.append(f"- {area['area_id']}: prepare {area['prepare_s']} s; surface {area['surface_state']} "
                         f"{area['surface_seconds']}; appearance {area['appearance_state']} "
                         f"total {area['appearance_total']} s")
        lines.append("")
    lines.append("## 4. During the walk")
    lines.append("")
    if client.get("walk"):
        for capture in client["walk"]:
            lines.append(f"- recorded capture {capture['capture_id']}: {capture['frames']} frames over "
                         f"{capture['recorded_seconds']} s ({capture['recorded_fps']} fps), ended by "
                         f"{capture['end_reason']}")
    if client.get("schedule"):
        lines.append(f"- schedule: {client['schedule']}; speed {client.get('speed')}; "
                     f"first_seconds {client.get('first_seconds')}")
    if stream:
        lines.append(f"- send lateness (ms): {stream.get('lateness_ms')}; >50 ms late: "
                     f"{stream.get('late_over_50ms')}, >250 ms: {stream.get('late_over_250ms')}")
        lines.append(f"- frame round trip (ms, send to frame_result): {stream.get('ack_ms')}")
        lines.append(f"- sockets: {stream.get('connections')} opened, {stream.get('unexpected_closes')} "
                     f"closed by the Tower; other messages {stream.get('other_messages')}")
    fetches = client.get("phone_fetches") or {}
    if fetches:
        lines.append(f"- phone-style geometry fetches: {fetches.get('manifests')} manifests, "
                     f"{fetches.get('segments')} segments, {fetches.get('bytes')} bytes, "
                     f"{fetches.get('errors')} errors; manifest ms {fetches.get('manifest_ms')}")
    for window in walk.get("windows") or []:
        lines.append(f"- Tower measurement window (L{window.get('line')}, ended by {window.get('end_reason')}): "
                     + ", ".join(f"{k} {window.get(k)}" for k in (
                         "frames_received", "effective_fps", "frames_rejected", "tx_seq_gap_total",
                         "backpressure_drops", "frame_processing_errors", "receive_to_result_ms_avg",
                         "receive_to_result_ms_max")))
    lines.append(f"- use during the walk: {_use_text(walk.get('use_during_walk'))}")
    for solve in walk.get("background_solves") or []:
        lines.append(f"- background solve {solve['n']} at {_clock(solve['t'])}, {solve['keyframes']} keyframes (L{solve['line']})")
    for surface in walk.get("live_surfaces") or []:
        lines.append(f"- live surface {surface['n']} at {_clock(surface['t'])} (L{surface['line']})")
    lines.append("")
    lines.append("## 5. After Stop: resources")
    lines.append("")
    lines.append(f"- use after Stop: {_use_text(report.get('use_after_stop'))}")
    lines.append(f"- GPU busy (>= 10%): {report.get('gpu_busy_minutes_after_stop')} min; "
                 f"<= 1.5 cores with the GPU idle: {report.get('low_cpu_gpu_idle_minutes_after_stop')} min "
                 f"({report.get('samples')} samples)")
    stage_timing = (report.get("store") or {}).get("stage_timing")
    lines.append("")
    lines.append("## 6. Stage timing (TOWER_WORLD_STAGE_TIMING)")
    lines.append("")
    if not stage_timing:
        lines.append("No `stage_timing.json`: the Tower ran without TOWER_WORLD_STAGE_TIMING=on, or its code "
                     "predates stage timing (a4afea1 does).")
    else:
        for finish in stage_timing.get("finishes") or []:
            lines.append(f"- finish {finish.get('context')} ({_clock(finish.get('started_at'))}):")
            for name, records in (finish.get("stages") or {}).items():
                walls = ", ".join(f"{r['wall_s']} s" + (f" (seed {r['seed']})" if r.get("seed") is not None else "")
                                  for r in records)
                lines.append(f"  - {name}: {walls}")
    view = report.get("phone_view") or {}
    lines.append("")
    lines.append("## 7. What the phone was told (World Builder status pushes)")
    lines.append("")
    t0 = stop.get("t")
    transitions = view.get("transitions") or []
    lines.append(f"{view.get('pushes', 0)} pushes, {len(transitions)} state transitions.")
    for item in transitions[-40:]:
        offset = "" if t0 is None else f" ({(item['t'] - t0) / 60:+.2f} min)"
        lines.append(f"- {_clock(item['t'])}{offset}: {json.dumps(item.get('changed'))[:300]}")
    lines.append("")
    lines.append("## 8. Inputs")
    lines.append("")
    for key, value in (report.get("inputs") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def write_report(out_dir: Path, report: dict) -> tuple:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "REPORT.md"
    tmp = out_dir / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, json_path)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    keyframes = report.get("keyframes")
    if keyframes:
        (out_dir / "keyframes.json").write_text(json.dumps({
            "identity": "(source_seq, segment_index), in keyframes.jsonl order",
            "count": keyframes.get("count"), "sha256": keyframes.get("sha256"),
            "sequence": keyframes.get("sequence")}), encoding="utf-8")
    return json_path, md_path


# -- comparing runs (N old vs N new) --------------------------------------------------


def comparable_metrics(report: dict) -> dict:
    """One report's numbers, flat, for `--compare`."""
    metrics: dict = {}
    verdict = report.get("verdict") or {}
    for key in ("stop_to_room_with_photos_min", "stop_to_finalization_min", "stop_to_settled_min"):
        metrics[key] = verdict.get(key)
    walk = report.get("tower_walk") or {}
    totals = walk.get("totals") or {}
    for key in ("frames_received", "frames_rejected", "tx_seq_gap_total", "backpressure_drops",
                "frame_processing_errors", "receive_to_result_ms_max"):
        metrics[key] = totals.get(key)
    metrics["frames_observed"] = walk.get("frames_observed")
    metrics["keyframes_accepted"] = walk.get("keyframes_accepted")
    stream = (report.get("client") or {}).get("stream") or {}
    metrics["frames_sent"] = stream.get("frames_sent")
    for name, stats in (("send_lateness_ms", stream.get("lateness_ms") or {}),
                        ("observe_lag_s", ((report.get("keyframes") or {}).get("observe_lag_s")) or {}),
                        ("rebuild_s", (walk.get("rebuilds") or {}).get("seconds") or {}),
                        ("rebuild_cadence_s", (walk.get("rebuilds") or {}).get("cadence_s") or {}),
                        ("live_surface_latency_s",
                         (report.get("live_surfaces_latency") or {}).get("latency_s") or {})):
        for q in ("p50", "p95", "p99", "max"):
            metrics[f"{name}.{q}"] = stats.get(q)
    metrics["rebuilds"] = (walk.get("rebuilds") or {}).get("count")
    metrics["background_solves"] = len(walk.get("background_solves") or [])
    metrics["live_surfaces"] = len(walk.get("live_surfaces") or [])
    zero = report.get("solve_draw_0") or {}
    metrics["draw0_map_s"] = zero.get("map_s")
    for row in report.get("waterfall") or []:
        if row.get("n") and not row.get("child") and row.get("minutes") is not None:
            metrics[f"row {row['n']} min"] = row["minutes"]
    return metrics


def _load_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    report = _read_json(run_dir / "report.json")
    if not isinstance(report, dict):
        raise SystemExit(f"{run_dir} has no report.json; render it first "
                         "(world_live_replay_report.py --run-dir <run> --out <run>)")
    keyframes = _read_json(run_dir / "keyframes.json") or {}
    return {"dir": str(run_dir), "label": report.get("label"), "report": report,
            "metrics": comparable_metrics(report), "sequence": keyframes.get("sequence"),
            "sha256": keyframes.get("sha256") or (report.get("keyframes") or {}).get("sha256"),
            "horizons": [s.get("keyframes") for s in (report.get("tower_walk") or {}).get("background_solves")
                         or []],
            "live_safety": (report.get("live_safety") or {}).get("result")}


def _first_difference(a, b):
    if a is None or b is None:
        return None
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_runs(baseline_dirs, candidate_dirs=()) -> dict:
    """Per-metric spread over the baseline runs, and each candidate run
    against that spread; keyframe-sequence identity to the first baseline."""
    baseline = [_load_run(d) for d in baseline_dirs]
    candidate = [_load_run(d) for d in candidate_dirs]
    names = []
    for run in baseline + candidate:
        for name in run["metrics"]:
            if name not in names:
                names.append(name)
    metrics = []
    for name in names:
        base = [run["metrics"].get(name) for run in baseline]
        cand = [run["metrics"].get(name) for run in candidate]
        known = [v for v in base if isinstance(v, (int, float))]
        item = {"metric": name, "baseline": base, "candidate": cand}
        if known:
            low, high = min(known), max(known)
            item.update({"mean": round(sum(known) / len(known), 4), "min": low, "max": high,
                         "spread": round(high - low, 4)})
            item["outside"] = [v for v in cand if isinstance(v, (int, float)) and not low <= v <= high]
        metrics.append(item)
    reference = baseline[0] if baseline else None

    def identity(run):
        if reference is None:
            return None
        same = run["sha256"] is not None and run["sha256"] == reference["sha256"]
        return {"dir": run["dir"], "sha256": run["sha256"], "identical": same,
                "first_difference": None if same else _first_difference(run["sequence"], reference["sequence"]),
                "horizons": run["horizons"], "horizons_identical": run["horizons"] == reference["horizons"],
                "live_safety": run["live_safety"]}

    return {
        "compare": "c22-live-replay-compare/1",
        "generated_at": round(time.time(), 3),
        "baseline": [run["dir"] for run in baseline],
        "candidate": [run["dir"] for run in candidate],
        "enough_runs": len(baseline) >= 3 and (not candidate or len(candidate) >= 3),
        "metrics": metrics,
        "keyframes": {"reference": None if reference is None else reference["dir"],
                      "baseline": [identity(run) for run in baseline],
                      "candidate": [identity(run) for run in candidate]},
    }


def render_compare(result: dict) -> str:
    lines = ["# C22 live replay: compare", ""]
    lines.append(f"Baseline ({len(result['baseline'])} runs): " + ", ".join(f"`{d}`" for d in result["baseline"]))
    if result["candidate"]:
        lines.append(f"Candidate ({len(result['candidate'])} runs): "
                     + ", ".join(f"`{d}`" for d in result["candidate"]))
    if not result["enough_runs"]:
        lines.append("")
        lines.append("**Fewer than 3 runs on a side: this is not a noise estimate (C19 F8 asks for N >= 3).**")
    lines.append("")
    lines.append("## Keyframe identity, (source_seq, segment_index), against the first baseline run")
    lines.append("")
    lines.append("| Run | sha256 | Identical | First differing index | Solve horizons identical | Live safety |")
    lines.append("|---|---|---|---|---|---|")
    for side in ("baseline", "candidate"):
        for item in result["keyframes"][side]:
            if item is None:
                continue
            lines.append(f"| {side}: `{item['dir']}` | {str(item['sha256'])[:16]} | {item['identical']} | "
                         f"{'' if item['first_difference'] is None else item['first_difference']} | "
                         f"{item['horizons_identical']} | {item['live_safety']} |")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("A candidate value outside the baseline's [min, max] is flagged. The spread is max - min "
                 "over the baseline runs: the old path's own noise.")
    lines.append("")
    lines.append("| Metric | Baseline mean | min | max | spread | Candidate | Outside |")
    lines.append("|---|---|---|---|---|---|---|")
    for item in result["metrics"]:
        flag = "**yes** " + str(item["outside"]) if item.get("outside") else ""
        lines.append(f"| {item['metric']} | {item.get('mean', '')} | {item.get('min', '')} | "
                     f"{item.get('max', '')} | {item.get('spread', '')} | "
                     f"{', '.join(str(v) for v in item['candidate'])} | {flag} |")
    lines.append("")
    return "\n".join(lines)


def _run_defaults(run_dir: Path) -> dict:
    """What a finished run's `run.json` says about where its inputs are."""
    run = _read_json(Path(run_dir) / "run.json")
    if not isinstance(run, dict):
        return {}
    defaults = {"run": run}
    if run.get("err_log"):
        defaults["tower_log"] = Path(run["err_log"])
    if run.get("out_log"):
        defaults["tower_out_log"] = Path(run["out_log"])
    if run.get("data_root"):
        defaults["world_root"] = Path(run["data_root"]) / "world_builder"
    return defaults


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=artifact_root_arg, required=True,
                        help="Where report.json, REPORT.md and keyframes.json (or compare.json and "
                             "COMPARE.md) are written.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="A finished run's --out, READ: client.json, samples.csv, run.json, "
                             "solution-snapshots/. Default: --out. Its run.json supplies the logs and "
                             "the world root unless they are given.")
    parser.add_argument("--tower-log", type=Path, default=None)
    parser.add_argument("--tower-out-log", type=Path, default=None)
    parser.add_argument("--world-root", type=Path, default=None, help="Read only.")
    parser.add_argument("--capture-id", default=None,
                        help="The walk's first capture (default: client.json's, else the log's first).")
    parser.add_argument("--label", default=None)
    parser.add_argument("--compare", type=Path, nargs="+", default=None, metavar="RUN",
                        help="Compare mode: the BASELINE run dirs (each with report.json), N >= 3.")
    parser.add_argument("--candidate", type=Path, nargs="+", default=(), metavar="RUN",
                        help="With --compare: the candidate run dirs, judged against the baseline's spread.")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if args.compare:
        result = compare_runs(args.compare, args.candidate)
        out.mkdir(parents=True, exist_ok=True)
        (out / "compare.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        (out / "COMPARE.md").write_text(render_compare(result), encoding="utf-8")
        print(json.dumps({"compare": str(out / "compare.json"), "enough_runs": result["enough_runs"]}, indent=2))
        return 0
    run_dir = Path(args.run_dir) if args.run_dir is not None else out
    defaults = _run_defaults(run_dir)
    tower_log = args.tower_log or defaults.get("tower_log")
    if tower_log is None:
        parser.error("--tower-log is required unless --run-dir holds a run.json naming it")
    client = _read_json(run_dir / "client.json") or {}
    samples = run_dir / "samples.csv"
    report = build_report(tower_log=tower_log, tower_out_log=args.tower_out_log or defaults.get("tower_out_log"),
                          world_root=args.world_root or defaults.get("world_root"), capture_id=args.capture_id,
                          client=client, samples=samples if samples.exists() else None, label=args.label,
                          run_dir=run_dir)
    if defaults.get("run"):
        report["run"] = defaults["run"]
    json_path, md_path = write_report(out, report)
    print(json.dumps({"report": str(json_path), "markdown": str(md_path), "verdict": report["verdict"],
                      "live_safety": report["live_safety"]["result"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
