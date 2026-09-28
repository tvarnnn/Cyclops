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

    python scripts/world_live_replay_report.py --out <dir> --tower-log <err.log> \
        [--tower-out-log <out.log>] [--world-root <root>] [--capture-id <id>]
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
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
    facts: dict = {"available": session is not None, "world_dir": str(world)}
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
    return facts


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
    values = sorted(values)
    if not values:
        return {"count": 0}

    def pick(q):
        return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]

    return {"count": len(values), "mean": round(sum(values) / len(values), 3),
            "p50": round(pick(0.5), 3), "p95": round(pick(0.95), 3), "max": round(values[-1], 3)}


def build_report(*, tower_log, tower_out_log=None, world_root=None, capture_id=None,
                 client=None, samples=None, label=None) -> dict:
    """Everything, as one JSON-able dict. `render_markdown` draws it."""
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
            for key in ("prepare_s", "masks_s", "freeze_s", "extract_s", "match_s", "map_s", "gate_s"):
                if timing.get(key) is not None:
                    add("", f"solve {key[:-2]} (solution.json timing; last draw's map/gate)", None, None,
                        detail=f"{timing[key]} s", child=True, use=False)
            consensus = solution.get("consensus") or {}
            for draw in consensus.get("draws") or []:
                add("", f"consensus draw {draw.get('draw')} (seed {draw.get('seed')})", None, None,
                    detail=f"map {draw.get('map_s')} s, gate {draw.get('gate_s')} s", child=True, use=False)
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

    return {
        "report": "c22-live-replay/1",
        "label": label or client.get("label"),
        "generated_at": round(time.time(), 3),
        "inputs": {"tower_log": str(tower_log), "tower_out_log": None if tower_out_log is None
                   else str(tower_out_log), "world_root": None if world_root is None else str(world_root),
                   "samples": None if samples is None else str(samples)},
        "hard_max_minutes": HARD_MAX_MINUTES,
        "verdict": {"stop_to_room_with_photos_min": photos_minutes, "result": verdict,
                    "stop_to_settled_min": _minutes(t0, milestones.get("settled")),
                    "stop_to_finalization_min": _minutes(t0, milestones.get("finalization_complete"))},
        "client": {k: client.get(k) for k in ("outcome", "speed", "first_seconds", "walk", "schedule",
                                              "stream", "phone_fetches", "handshake", "session_start",
                                              "live_tower_at_start", "aborted", "t0", "stopped_at")
                   if k in client},
        "tower_walk": {
            "captures": timeline.get("captures"),
            "world_id": timeline.get("world_id"),
            "session_id": timeline.get("session_id"),
            "stream_start": timeline.get("stream_start"),
            "stop": timeline.get("stop"),
            "summary": timeline.get("tower_summary"),
            "summary_line": timeline.get("tower_summary_line"),
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
    lines.append("## 1. Summary")
    lines.append("")
    photos = verdict.get("stop_to_room_with_photos_min")
    lines.append(f"1. **Stop to room with photos: {photos if photos is not None else 'not reached'} min** "
                 f"against the {report['hard_max_minutes']:.0f}-min hard maximum: **{verdict.get('result')}**.")
    lines.append(f"2. Stop to `finalization: complete`: {verdict.get('stop_to_finalization_min')} min. "
                 f"Stop to settled (worker exit and the chore's areas): {verdict.get('stop_to_settled_min')} min.")
    stream = client.get("stream") or {}
    summary = walk.get("summary") or {}
    if stream or summary:
        sent = (f"{stream.get('frames_sent')} sent, {stream.get('frame_results')} answered, "
                f"{sum((stream.get('frame_errors') or {}).values())} refused, "
                f"at {stream.get('achieved_send_fps')} fps sent. " if stream else "")
        lines.append(f"3. Frames: {sent}The Tower received "
                     f"{summary.get('frames_received')} ({summary.get('effective_fps')} fps), rejected "
                     f"{summary.get('frames_rejected')}, tx_seq gaps {summary.get('tx_seq_gap_total')}; "
                     f"the builder observed {walk.get('frames_observed')} and accepted "
                     f"{walk.get('keyframes_accepted')} keyframes.")
    rebuilds = walk.get("rebuilds") or {}
    lines.append(f"4. Live builder: {rebuilds.get('count')} rebuilds ({rebuilds.get('failed')} failed), "
                 f"latency p50 {rebuilds.get('seconds', {}).get('p50')} s / p95 "
                 f"{rebuilds.get('seconds', {}).get('p95')} s / max {rebuilds.get('seconds', {}).get('max')} s, "
                 f"cadence p50 {rebuilds.get('cadence_s', {}).get('p50')} s / max "
                 f"{rebuilds.get('cadence_s', {}).get('max')} s; {rebuilds.get('after_stop')} after Stop. "
                 f"{len(walk.get('background_solves') or [])} background solves and "
                 f"{len(walk.get('live_surfaces') or [])} live surfaces launched during the walk.")
    if client.get("aborted"):
        lines.append(f"5. **Aborted** at {_clock(client['aborted'].get('t'))}: {client['aborted'].get('reason')}.")
    lines.append("")
    lines.append("## 2. Waterfall")
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
    lines.append("## 3. During the walk")
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
    if summary:
        lines.append(f"- Tower final summary (L{walk.get('summary_line')}): "
                     + ", ".join(f"{k} {summary.get(k)}" for k in (
                         "frames_received", "effective_fps", "frames_rejected", "tx_seq_gap_total",
                         "backpressure_drops", "receive_to_result_ms_avg", "receive_to_result_ms_max")))
    lines.append(f"- use during the walk: {_use_text(walk.get('use_during_walk'))}")
    for solve in walk.get("background_solves") or []:
        lines.append(f"- background solve {solve['n']} at {_clock(solve['t'])}, {solve['keyframes']} keyframes (L{solve['line']})")
    for surface in walk.get("live_surfaces") or []:
        lines.append(f"- live surface {surface['n']} at {_clock(surface['t'])} (L{surface['line']})")
    lines.append("")
    lines.append("## 4. After Stop: resources")
    lines.append("")
    lines.append(f"- use after Stop: {_use_text(report.get('use_after_stop'))}")
    lines.append(f"- GPU busy (>= 10%): {report.get('gpu_busy_minutes_after_stop')} min; "
                 f"<= 1.5 cores with the GPU idle: {report.get('low_cpu_gpu_idle_minutes_after_stop')} min "
                 f"({report.get('samples')} samples)")
    stage_timing = (report.get("store") or {}).get("stage_timing")
    lines.append("")
    lines.append("## 5. Stage timing (TOWER_WORLD_STAGE_TIMING)")
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
    lines.append("## 6. What the phone was told (World Builder status pushes)")
    lines.append("")
    t0 = stop.get("t")
    transitions = view.get("transitions") or []
    lines.append(f"{view.get('pushes', 0)} pushes, {len(transitions)} state transitions.")
    for item in transitions[-40:]:
        offset = "" if t0 is None else f" ({(item['t'] - t0) / 60:+.2f} min)"
        lines.append(f"- {_clock(item['t'])}{offset}: {json.dumps(item.get('changed'))[:300]}")
    lines.append("")
    lines.append("## 7. Inputs")
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
    return json_path, md_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=artifact_root_arg, required=True,
                        help="Report directory (client.json and samples.csv are read from it if present).")
    parser.add_argument("--tower-log", type=Path, required=True)
    parser.add_argument("--tower-out-log", type=Path, default=None)
    parser.add_argument("--world-root", type=Path, default=None, help="Read only.")
    parser.add_argument("--capture-id", default=None,
                        help="The walk's first capture (default: client.json's, else the log's first).")
    parser.add_argument("--label", default=None)
    args = parser.parse_args(argv)
    out = Path(args.out)
    client = _read_json(out / "client.json") or {}
    samples = out / "samples.csv"
    report = build_report(tower_log=args.tower_log, tower_out_log=args.tower_out_log,
                          world_root=args.world_root, capture_id=args.capture_id, client=client,
                          samples=samples if samples.exists() else None, label=args.label)
    run = _read_json(out / "run.json")
    if isinstance(run, dict):
        report["run"] = run
    json_path, md_path = write_report(out, report)
    print(json.dumps({"report": str(json_path), "markdown": str(md_path), "verdict": report["verdict"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
