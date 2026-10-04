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
lag, rebuild latency and cadence, the background solves' launch, landing and
termination at Stop, the Tower-side pacing (the re-recorded capture joined
to the source journal on `wire_seq`), and live-surface latency where the
run watched it. What :8000 did during the run is judged apart, as the
environment's verdict, and so is "Replay fidelity": the Tower-side pacing
against the bar manager 142 approved (`FIDELITY_BAR` v1), or, for a run
started after 2026-09-28 16:55 EDT, v3 (manager 149 §2, which replaced
manager 148's v2) -- each run is judged by the version in force when it
started, and the report says which -- which every proof-set run must PASS
(n/a is not a pass). `--compare` flags a run that fails either
and leaves it out of the baseline's range.

THE W0 VERDICT (review C24 HIGH-1) is Stop -> `phone_photos_at`, when the
replay CLIENT received the push saying the room's photos were ready. The
store's `updated_at` for the room appearance is INFO. A report with no client
record (a real walk's log) can only give the store time, and says so: that is
NOT the client measure. Neither is a phone rendering the photos.

THE PACING CLOCK (review C24 HIGH-4). `frames.jsonl` `received_at` is the
capture recorder's stamp, taken when `ws.py` hands it the frame AFTER parsing,
decoding, the CV module and the `frame_result` send: a post-reply stamp, not
socket arrival. The pacing rows and the fidelity bar are labelled so.

`--compare` counts only COMPARABLE runs (review C24 HIGH-2,
`comparability_key`): the same source capture and journal, switch set, code,
streaming harness, replay shape, calibration and fidelity family. A candidate
may differ from the baseline only in the switches declared with
`--candidate-switches`. A run made without the :8000 guard, or declared
`--not-a-proof-run`, is NOT-PROOF and never counted (review C24 HIGH-3).

`--compare` counts DISTINCT streamed runs (review F11 MED-1): an argument
that shares a Tower-minted capture id, a run directory, a data root or a
client.json with an earlier one is the same run again (a repeat, a
re-render, a copy) and is not counted. And it reads every run's verdicts
against the run's own records SEALED AT RUN TIME (review F11 LOW-9,
`evidence_seal`): the runner's run-time render records the seal and prints
it for the RUN ledger; a run whose records no longer hash to it, whose
render embeds anything else, or that another report script rendered, is
not counted.

All of it is read AFTER the fact, so a finished run is re-reported without
re-running it:

    python scripts/world_live_replay_report.py --run-dir <run's --out> --out <dir> \
        [--data-root <the run's data root>] [--capture-root <the source captures>]
    python scripts/world_live_replay_report.py --out <dir> --tower-log <err.log> \
        [--tower-out-log <out.log>] [--world-root <root>] [--capture-id <id>]

and N runs are compared (old x N against new x N, per-metric spread):

    python scripts/world_live_replay_report.py --out <dir> --compare <old1> <old2> <old3> \
        [--candidate <new1> <new2> <new3> [--candidate-switches KEY=VALUE ...]]
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402

HARD_MAX_MINUTES = 10.0
CHORE_CHAIN_GAP_S = 90.0
# The report script's own version. The STREAMING harness's version
# (`world_live_replay.HARNESS_VERSION`) is unchanged by a report-only round:
# C22-F12 left both streaming scripts byte-identical to 807054d.
REPORT_VERSION = "c22-report/F12 (distinct sealed runs, timing-env key, store-backed photos, full walk, Walk 6 gate)"

# The W0 timing measure (review C24 HIGH-1) and what each basis is.
W0_TIMING_METRIC = "stop_to_phone_photos_min"
BASIS_PHONE = ("phone_photos_at: the replay client's receipt of the status push that said the room's photos "
               "were ready (not photos rendered on a phone)")
BASIS_STORE = ("the STORE's room-appearance updated_at -- NOT the client measure: this report has no client "
               "record (a real walk's log), so nothing says when a client was told")

# What the replay cannot exercise (review C24 MED-7), printed in every report.
REPLAY_LIMITS = ("The replay awaits every send, so it never drops a frame, and it replays only the frames the "
                 "original Tower recorded; it reconnects cleanly. It cannot exercise the phone's bounded send "
                 "window, its frame drops or its stalled-socket reconnect under a changed Tower, nor the "
                 "overlapping reconnect of a multi-capture walk: those need a phone trace and a physical walk. "
                 "It is a Tower benchmark.")

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
    # A HARD stop ends the wait at once (`BackgroundSolver.wait`, should_stop):
    # the solve is terminated just the same (review C22 round 3 L-e).
    "bg_solve_stop_requested": re.compile(
        r"\[Tower\]\[WorldBuilder\] background solve pid (?P<pid>\d+) terminated: a stop was requested"),
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
        timeline["background_solves"] = solve_landings(timeline["background_solves"], rebuilds, [], None,
                                                     timeline["live_surfaces"])
        return timeline
    horizon = walk_end
    timeline["bg_solve_wait"] = _event(_first(events, "bg_solve_wait", after=t0, before=horizon))
    timeline["bg_solve_stop_requested"] = _event(_first(events, "bg_solve_stop_requested", after=t0,
                                                        before=horizon))
    launched = _first(events, "final_launched", after=t0, before=horizon)
    timeline["final_launched"] = _event(launched)
    terminations = [e for e in events if e.kind in ("bg_solve_wait", "bg_solve_stop_requested")
                    and e.t >= t0 and (horizon is None or e.t <= horizon)]
    timeline["background_solves"] = solve_landings(
        timeline["background_solves"], rebuilds, terminations, launched, timeline["live_surfaces"])
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


def solve_landings(solves: list, rebuilds: list, waits: list, final_launched, live_surfaces=()) -> list:
    """Each background solve's landing, and whether Stop terminated it
    (review C22 H2, the background-solve horizons row).

    LANDED BY. The builder logs a solve's launch but not its end. A solve
    "lands" when the builder next sees it finished; that forces a rebuild,
    and the rebuild is where the NEXT background solve is launched
    (`world_build_session.py`, the observe loop: rebuild, then
    `solver.maybe_launch`, then the live surface). The next solve cannot
    launch while this one runs, so the next launch line is when this solve
    had landed BY -- exact when the next solve was due at the landing, a
    little late when the solve cadence held it back. The last solve of a
    walk has no next background solve: Stop waits for it and only then
    launches the final solve, so the final solve's launch is when it had
    landed by.

    BY LOG LINE, NOT BY TIME (review C22 round 3 L-a). One loop iteration
    logs the rebuild, then the next solve's launch, then the live surface.
    So a surface launched at this solve's landing rebuild is logged AFTER
    the next solve's launch line, and a time window would give it to the
    next solve. Each live surface belongs to the rebuild logged just before
    it, and that rebuild to the last solve launched before IT
    (`live_surface_between`, with its rebuild).

    THE REBUILDS (review C22 round 4 L-1). The log proves one: the rebuild
    that launched the next solve, which this solve had landed BY
    (`rebuild_*`; none for a walk's last solve). A surface's rebuild before
    it is at most a POSSIBLY EARLIER landing (`possibly_earlier`), and OPEN:
    a surface is also relaunched when an earlier, stale one finishes, and
    the log cannot tell that from a landing. A surface at the FIRST rebuild
    after this solve's launch, when an earlier solve exists, is not read as
    this solve's landing at all (`landing_rebuild_unknown`): the builder
    reads `finished()` before its rebuild, so an earlier solve reaped inside
    the launch of this one surfaces one rebuild late, right there (smoke 1:
    solve 2 at L5180, surface 1 at L5209, 0.3 s later). The log cannot tell
    which, so its landing rebuild is unknown.

    TERMINATED AT STOP. Stop gives a running background solve a bounded
    wait, then terminates it (`background solve pid N still running after
    Ns; terminating`); a HARD stop terminates it at once (`background solve
    pid N terminated: a stop was requested`, review C22 round 3 L-e). A
    terminated solve never landed.
    """
    rebuilds = sorted(rebuilds, key=lambda r: r.line)

    def rebuild_before(line):
        found = None
        for rebuild in rebuilds:
            if rebuild.line >= line:
                break
            found = rebuild
        return found

    def rebuild_after(line):
        return next((rebuild for rebuild in rebuilds if rebuild.line > line), None)

    surfaces = []
    for surface in live_surfaces:
        rebuild = rebuild_before(surface["line"])
        surfaces.append({**surface, "rebuild_line": None if rebuild is None else rebuild.line,
                         "rebuild_n": None if rebuild is None else int(rebuild.groups["n"])})
    out = []
    for index, solve in enumerate(solves):
        item = dict(solve)
        terminated = next((w for w in waits if w.groups.get("pid") == solve.get("pid")), None)
        item["terminated_at_stop"] = terminated is not None
        item["terminated_line"] = terminated.line if terminated is not None else None
        item["terminated_how"] = None if terminated is None else (
            "a stop was requested" if terminated.kind == "bg_solve_stop_requested"
            else f"still running after {terminated.groups.get('s')} s")
        after = solves[index + 1] if index + 1 < len(solves) else None
        landed = None
        if after is not None:
            landed = {"t": after["t"], "line": after["line"], "via": f"background solve {after['n']} launch"}
        elif terminated is None and final_launched is not None:
            landed = {"t": final_launched.t, "line": final_launched.line,
                      "via": "the final solve's launch (Stop waits for the last solve)"}
        # This solve's own log lines: from its launch to the next launch.
        mine = [s for s in surfaces if s["rebuild_line"] is not None and s["rebuild_line"] > solve["line"]
                and (after is None or s["rebuild_line"] < after["line"])]
        surface = mine[0] if mine and landed is not None else None
        if landed is not None:
            launcher = rebuild_before(after["line"]) if after is not None else None
            if launcher is not None and launcher.line <= solve["line"]:
                launcher = None
            # A surface at the first rebuild after this launch is not read as
            # this solve's landing when an earlier solve exists (L-1).
            first = rebuild_after(solve["line"])
            doubtful = [s for s in mine if index > 0 and first is not None and s["rebuild_line"] == first.line]
            plausible = next((s for s in mine if s not in doubtful), None)
            if launcher is not None:
                shared = plausible is not None and plausible["rebuild_line"] == launcher.line
                landed.update({"rebuild_line": launcher.line, "rebuild_n": int(launcher.groups["n"]),
                               "rebuild_why": f"it launched background solve {after['n']}"
                               + (f" and live surface {plausible['n']}" if shared else
                                  "; landed by then at the latest")})

            def shown(s, why):
                return {"rebuild_line": s["rebuild_line"], "rebuild_n": s["rebuild_n"], "surface": s["n"],
                        "after_launch_s": round(s["t"] - solve["t"], 2), "open": why}

            if plausible is not None and (launcher is None or plausible["rebuild_line"] < launcher.line):
                landed["possibly_earlier"] = shown(
                    plausible, "the landing, or a stale surface's relaunch; the log cannot tell")
            if doubtful:
                landed["landing_rebuild_unknown"] = shown(
                    doubtful[0], "the first rebuild after this solve's launch: the previous solve's landing "
                                 "seen one rebuild late, this solve's, or a stale surface's relaunch; the log "
                                 "cannot tell")
        item["landed_by"] = landed
        item["horizon_s"] = None if landed is None else round(landed["t"] - solve["t"], 2)
        item["live_surface_between"] = None if surface is None else dict(surface)
        out.append(item)
    return out


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
        # A missing counter in any window is unknown, not an implicit zero.
        totals[key] = sum(values) if len(values) == len(summaries) else None
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
    """The session's keyframe SELECTION SEQUENCE and its keyframe-accept lag
    (review C22 H2; labels qualified by review C24 MED-8).

    THE SELECTION SEQUENCE is the ordered `(source_seq, segment_index)` of
    `keyframes.jsonl`, never `keyframe_id`: that embeds the session id, which
    differs between runs. It says WHICH frames were selected, in which
    segments -- not their image content, and not capture-qualified (two
    captures of one walk can reuse a `source_seq`). Selection is
    content-driven, so two runs of one walk that differ here leaked timing
    into the builder; two that agree may still differ in pixels.

    THE KEYFRAME-ACCEPT LAG is `events.jsonl` `keyframe_accepted.at` minus
    that keyframe's `received_at` (joined on `keyframe_id`), for ACCEPTED
    keyframes only -- never every frame. `received_at` is the recorder's
    post-reply stamp (review C24 HIGH-4), not socket arrival. A per-frame
    observe lag would need an `observed_at` in `frames_quality.jsonl`, which
    the Tower does not write. `observe_lag_s` / `observe_lag_unmatched` keep
    their data names so older renders still compare.
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


def read_capture_journal(directory) -> dict | None:
    """A capture's `capture.json` `started_at` and its `frames.jsonl` rows
    (`wire_seq`, `received_at`), in journal order. Hash the bytes parsed."""
    directory = Path(directory)
    try:
        manifest_bytes = (directory / "capture.json").read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        manifest_bytes, manifest = None, None
    try:
        journal_bytes = (directory / "frames.jsonl").read_bytes()
        lines = journal_bytes.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        journal_bytes, lines = None, []
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
        if isinstance(row, dict) and isinstance(row.get("wire_seq"), int) and \
                isinstance(row.get("received_at"), (int, float)):
            rows.append(row)
    if not isinstance(manifest, dict) and not rows:
        return None
    started = manifest.get("started_at") if isinstance(manifest, dict) else None
    if not isinstance(started, (int, float)):
        started = rows[0]["received_at"] if rows else None
    return {"capture_id": directory.name, "started_at": started,
            "frames": [(row["wire_seq"], row["received_at"]) for row in rows],
            "sha256": {"capture.json": hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes else None,
                       "frames.jsonl": hashlib.sha256(journal_bytes).hexdigest() if journal_bytes else None}}


def journal_sha256(directory) -> dict:
    """sha256 of a capture's `capture.json` and `frames.jsonl`, so a re-render
    shows WHICH source journal it joined to (review C22 round 3 L-g: a pinned
    run's pacing row must not silently depend on whatever the live store holds
    at render time). Reads only."""
    digests = {}
    for name in ("capture.json", "frames.jsonl"):
        try:
            digests[name] = hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
        except OSError:
            digests[name] = None
    return digests


# -- the run's own records, sealed at run time (review F11 LOW-9) --------------------


def _sha256_of(path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def evidence_seal(run_dir, run: dict | None, timeline: dict | None) -> dict:
    """sha256 of every record a run's verdicts are read from (review F11 LOW-9).

    The runner renders its run's report at RUN TIME (`build_report` and
    `write_report` of this script, into the run's own --out), after the test
    Tower has stopped and every record below is final. That render keeps this
    seal, and `write_report` prints it to stderr, which the quiet launcher's
    log keeps OUTSIDE the run directory: copy it to the RUN ledger. `--compare`
    recomputes it and counts a run only when its records still hash to the
    seal its run-time render recorded, and when every record a render embeds
    equals the sealed file.

      client.json, run.json, samples.csv   the run directory's own records;
      err_log, out_log                     the test Tower's logs (run.json);
      session.json                         the walk's session in the test
                                           Tower's store (the room appearance);
      captures/<id>/capture.json, frames.jsonl   what the test Tower re-recorded.

    A missing file is None, which a later seal must match exactly."""
    run = run if isinstance(run, dict) else {}
    timeline = timeline if isinstance(timeline, dict) else {}
    run_dir = Path(run_dir)
    files = {name: _sha256_of(run_dir / name) for name in ("client.json", "run.json", "samples.csv")}
    for key in ("err_log", "out_log"):
        files[key] = _sha256_of(run[key]) if run.get(key) else None
    data_root = run.get("data_root")
    world_id, session_id = timeline.get("world_id"), timeline.get("session_id")
    files["session.json"] = (_sha256_of(Path(data_root) / "world_builder" / "worlds" / str(world_id) / "sessions"
                                        / str(session_id) / "session.json")
                             if data_root and world_id and session_id else None)
    for capture_id in timeline.get("captures") or []:
        for name in ("capture.json", "frames.jsonl"):
            files[f"captures/{capture_id}/{name}"] = (
                _sha256_of(Path(data_root) / "captures" / str(capture_id) / name) if data_root else None)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"files": files, "seal": digest, "at": None}


def _norm_path(path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def session_appearance(run: dict | None, timeline: dict | None) -> dict | None:
    """The walk's room-appearance stage in the test Tower's store
    (`<data root>/world_builder/worlds/<w>/sessions/<s>/session.json`), or None."""
    run = run if isinstance(run, dict) else {}
    timeline = timeline if isinstance(timeline, dict) else {}
    if not run.get("data_root") or not timeline.get("world_id") or not timeline.get("session_id"):
        return None
    session = _read_json(Path(run["data_root"]) / "world_builder" / "worlds" / str(timeline["world_id"])
                         / "sessions" / str(timeline["session_id"]) / "session.json")
    stages = session.get("stages") if isinstance(session, dict) and isinstance(session.get("stages"), dict) else {}
    stage = stages.get("appearance")
    return stage if isinstance(stage, dict) else None


# THE REPLAY-FIDELITY BAR, VERSIONED. These are the only copies of these
# numbers; `replay_fidelity` judges against them and the report prints them.
#   v1  Manager 142 (RUN\lead\W0-STAGES.md, "Manager 142 rulings", 2026-09-28)
#       APPROVED it as the C22 review's round 3 proposed it: "Every proof-set
#       run must pass it, or it is discarded and re-run."
#   (v2 Manager 148 §1, declared 2026-09-28 16:55 EDT for runs started after
#       it: a detrended |p50(offset - median(offset))| <= 5 ms plus
#       |median(offset)| <= 20 ms. It was degenerate -- the detrended p50 is
#       ~0 by construction -- it judged no run, and manager 149 withdrew it.
#       No copy of it is kept: a render that names v2 is re-rendered.)
#   v3  Manager 149 §2 (RUN\physical-test\WALK6-PLAN.md), DECLARED 2026-09-28
#       17:25 EDT for every run started after 2026-09-28 16:55 EDT (manager
#       148's cut-off; no run was made between 16:55 and 17:25) -- "never
#       loosen a bar after seeing the data it fails": v1 WITHOUT its |p50|
#       clause, PLUS a jitter clause, p95 |offset - median(offset)| <= 45 ms,
#       and a constant-bias clause, |median(offset)| <= 20 ms. Every other
#       clause is v1's, unchanged (so they are taken from v1 below, never
#       restated).
# A run is judged by the version in force when it STARTED (`fidelity_version`):
# v3 only when its recorded start is AFTER 16:55 EDT; a run started at or
# before it, or with no recorded start, keeps v1 -- the bar as written.
_EDT = datetime.timezone(datetime.timedelta(hours=-4), "EDT")
FIDELITY_BAR = {
    "v1": {
        "ruling": "manager 142 (W0-STAGES.md, 2026-09-28), as proposed by the C22 review round 3",
        # In force for runs started after this (epoch s); None: from the first.
        "applies_after": None,
        # Signed recorder-stamp offset error (`tower_side_pacing`: the post-reply
        # stamp, not socket arrival -- review C24 HIGH-4), ms. Positive is late.
        "offset_p50_abs_ms": 5.0,
        "offset_p95_ms": 60.0,
        "offset_p99_ms": 250.0,
        # At most this fraction of the joined frames beyond +/- `beyond_ms`...
        "beyond_ms": 50.0,
        "beyond_max_fraction": 0.05,
        # ...and no frame more than this early.
        "early_max_ms": 50.0,
        # The client's own send lateness (stops at the send call), ms.
        "client_lateness_p95_ms": 5.0,
    },
}
FIDELITY_BAR["v3"] = {
    **{key: value for key, value in FIDELITY_BAR["v1"].items() if key != "offset_p50_abs_ms"},
    "ruling": ("manager 149 §2 (WALK6-PLAN.md), declared 2026-09-28 17:25 EDT for every run started after "
               "2026-09-28 16:55 EDT; replaces manager 148's v2, which judged no run; amends manager 142's bar"),
    "applies_after": datetime.datetime(2026, 9, 28, 16, 55, tzinfo=_EDT).timestamp(),
    # p95 of |offset - median(offset)|, ms: the jitter clause.
    "offset_abs_deviation_p95_ms": 45.0,
    # |median(offset)|, ms: the constant-bias clause.
    "offset_median_abs_ms": 20.0,
}
FIDELITY_V3_APPLIES_AFTER_TEXT = "2026-09-28 16:55 EDT"
# THE FAMILY of a bar version: the clock its offsets are measured on (review
# C24 HIGH-2, one field of `comparability_key`). v1 and v3 both judge the
# recorder's post-reply stamp, so runs judged by either compare; a bar on a
# future pre-decode ingress stamp (C24 HIGH-4's Tower-side fix) would be a
# new family, and its runs would not compare with these.
FIDELITY_FAMILY = {"v1": "recorder-stamp", "v3": "recorder-stamp"}
# What the bar measures, printed with the verdict (review C24 HIGH-4).
FIDELITY_MEASURES = ("the recorder's post-reply stamp (`frames.jsonl` `received_at`, taken after parse, decode, "
                     "the CV module and the frame_result send), not socket arrival: a FAIL on the Tower-side "
                     "clauses with the client on time can be the code under test's own frame path")
# Both versions' rulings, for a record that spans runs (`--compare`).
FIDELITY_RULING = "; ".join(f"{version}: {bar['ruling']}" for version, bar in FIDELITY_BAR.items())
FIDELITY_NA_NOTE = ("n/a is NOT a pass. Every proof-set run must PASS replay fidelity (manager 142): "
                    "re-render it with the source journal (`--capture-root`, e.g. "
                    "RUN\\experiments\\C22-REPLAY\\source-captures, and `--data-root` if the test "
                    "Tower's root moved), or discard it and re-run.")

PACING_OVER_S = FIDELITY_BAR["v1"]["beyond_ms"] / 1000.0   # v3 keeps v1's
# The pacing rows stay INFO in the CODE's verdict: their bar is the separate
# Replay fidelity verdict, and their tails are `--compare`'s.
PACING_ROW_REQUIRED = "its bar is the Replay fidelity verdict (manager 142); the tails: no regression (--compare)"
# The pacing rows' names (review C24 HIGH-4: a post-reply stamp, not receipt).
# The keyframe rows' names (review C24 MED-8): a selection sequence, not an
# image identity; a lag over ACCEPTED keyframes only, from the post-reply stamp.
KEYFRAME_SEQUENCE_ROW = ("keyframe selection sequence (source_seq, segment_index) -- which frames were selected, "
                         "not their image content, not capture-qualified")
KEYFRAME_LAG_ROW = ("keyframe-accept lag (s): keyframe_accepted.at - the recorder's post-reply received_at, "
                    "ACCEPTED keyframes only (not a per-frame observe lag)")
PACING_ROW = ("Tower-side pacing: recorder-stamp offset - recorded offset (ms), joined on wire_seq "
              "(post-reply stamp, not socket arrival)")
INTERVAL_ROW = "Tower-side recorder-stamp intervals (s), replayed vs recorded (post-reply stamps, not arrivals)"


def tower_side_pacing(*, client: dict, data_root, capture_root, capture_root_from=None) -> dict:
    """How faithfully the test Tower HANDLED the frames at the recorded pace
    (review C22 H2, the "README fidelity" row), on the recorder's clock.

    The replay's send lateness is measured at the client and stops at the
    send call. This is the Tower's own side: the test Tower re-records every
    frame into `<data-root>/captures/<id>/frames.jsonl`, stamped exactly as
    the recorded walk's journal was. Each re-recorded frame is joined to the
    source journal on `wire_seq` (the phone's `seq`, which the replay sends
    as recorded), capture by capture in walk order.

    WHAT THE STAMP IS (review C24 HIGH-4). `received_at` is NOT socket
    arrival. `CaptureRecorder.write_frame` takes it (`time.time()`, a4afea1
    `capture.py:257`) when `ws.py` `_record_capture` calls it -- AFTER the
    frame was JSON-parsed off the socket, base64- and header-decoded, run
    through the CV module and answered with `frame_result`, and after the
    previous frame's fsync'd write, because the socket is read one message
    at a time (`ws.py` `_handle_frame_message`, `_fan_out_frame`). The
    recorded walk's stamps are the same kind, and the schedule SENDS each
    frame at its recorded stamp. So the offset below is, per frame:

        client send lateness + loopback transit
        + the TEST Tower's handling delay (queue + parse + decode + CV + reply)

    relative to the recorded walk, whose own handling delay is already in
    the schedule. A frame path made slower by the code under test moves it,
    exactly as a late client does; only the client-lateness clause of the
    fidelity bar is the harness's alone.

    RECORDER-STAMP OFFSET ERROR, per frame: (replayed stamp - the replay's
    first capture `started_at`) - (recorded stamp - the recorded walk's first
    capture `started_at`) / speed. Both origins are the recorder's start on
    the walk's first `stream_start`, the origin the schedule is built on.
    Positive is late. Signed p1/p5/p50/p95/p99, min/max, and the count beyond
    50 ms either way.

    STAMP INTERVALS (`inter_arrival_*`, a data name kept for older renders):
    consecutive stamps within a capture, replayed and recorded (scaled by
    speed), and their per-frame difference. Intervals between post-reply
    stamps, not between arrivals.

    Reads the run's data root and the source captures (both read only).
    """
    walk = client.get("walk") or []
    source_ids = [c.get("capture_id") for c in walk if isinstance(c, dict) and c.get("capture_id")] \
        or list(client.get("captures_replayed") or [])
    replay_ids = list(client.get("tower_captures") or [])
    base = {"computable": False, "data_root": None if data_root is None else str(data_root),
            "source_capture_root": None if capture_root is None else str(capture_root),
            "source_capture_root_from": capture_root_from,
            "source_captures": source_ids, "replay_captures": replay_ids}
    if data_root is None:
        return {**base, "why": "no data root (run.json names it; or give --data-root)"}
    if capture_root is None:
        return {**base, "why": "no source capture root (client.json names it; or give --capture-root)"}
    if not source_ids:
        return {**base, "why": "the client record names no source capture"}
    if not replay_ids:
        return {**base, "why": "the client record names no capture the test Tower recorded"}
    sources = [read_capture_journal(Path(capture_root) / cid) for cid in source_ids]
    replays = [read_capture_journal(Path(data_root) / "captures" / cid) for cid in replay_ids]
    missing = [cid for cid, j in zip(source_ids, sources) if not j or not j["frames"]] + \
              [cid for cid, j in zip(replay_ids, replays) if not j or not j["frames"]]
    if missing:
        return {**base, "why": f"no journal for {missing}"}
    speed = client.get("speed") if isinstance(client.get("speed"), (int, float)) and client.get("speed") > 0 else 1.0
    src_origin, rep_origin = sources[0]["started_at"], replays[0]["started_at"]
    errors, rec_gaps, rep_gaps, gap_errors = [], [], [], []
    matched = replay_only = source_only = duplicates = 0
    for source, replayed in zip(sources, replays):
        recorded = {}
        for seq, at in source["frames"]:
            if seq in recorded:
                duplicates += 1
                continue
            recorded[seq] = at
        seen = set()
        previous = None
        for seq, at in replayed["frames"]:
            if seq not in recorded or seq in seen:
                replay_only += 1
                previous = None
                continue
            seen.add(seq)
            matched += 1
            got = at - rep_origin
            want = (recorded[seq] - src_origin) / speed
            errors.append(got - want)
            if previous is not None:
                rep_gap, rec_gap = got - previous[0], want - previous[1]
                rep_gaps.append(rep_gap)
                rec_gaps.append(rec_gap)
                gap_errors.append(rep_gap - rec_gap)
            previous = (got, want)
        source_only += len(recorded) - len(seen)
    for extra in sources[len(replays):]:
        source_only += len(extra["frames"])
    for extra in replays[len(sources):]:
        replay_only += len(extra["frames"])
    if not matched:
        return {**base, "why": "no re-recorded frame joins the source journal on wire_seq"}
    offsets_ms = [e * 1000 for e in errors]
    median_ms = _median(offsets_ms)
    return {
        **base,
        "computable": True,
        "speed": speed,
        "first_seconds": client.get("first_seconds"),
        "origins": {"source_started_at": src_origin, "replay_started_at": rep_origin},
        "matched": matched,
        "replay_only": replay_only,
        "source_only": source_only,
        # Every distinct wire_seq of the source journals: those the schedule
        # did not send (the `first_seconds` cut) are source-only by design.
        "source_frames": sum(len({seq for seq, _ in source["frames"]}) for source in sources),
        "source_duplicate_seq": duplicates,
        "source_sha256": {cid: source["sha256"] for cid, source in zip(source_ids, sources)},
        "captures_paired": min(len(sources), len(replays)),
        "offset_error_ms": signed_distribution(offsets_ms),
        # Fidelity bar v3 (manager 149 §2): the offset's median (the constant
        # bias), and each offset's absolute deviation from it,
        # |offset - median(offset)| (the jitter), whose p95 v3 judges.
        "offset_median_ms": round(median_ms, 3),
        "offset_abs_deviation_ms": distribution([abs(v - median_ms) for v in offsets_ms]),
        "late_over_50ms": sum(1 for e in errors if e > PACING_OVER_S),
        "early_over_50ms": sum(1 for e in errors if e < -PACING_OVER_S),
        "inter_arrival_s": {"replayed": distribution(rep_gaps), "recorded": distribution(rec_gaps)},
        "inter_arrival_error_ms": signed_distribution([e * 1000 for e in gap_errors]),
    }


def signed_distribution(values) -> dict:
    """`distribution` for a signed quantity: its low tail (min, p1, p5) too."""
    stats = distribution(values)
    if not stats.get("count"):
        return stats
    ordered = sorted(values)

    def pick(q):
        return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]

    return {**stats, "min": round(ordered[0], 3), "p1": round(pick(0.01), 3), "p5": round(pick(0.05), 3)}


def _median(values) -> float:
    """The median: the middle value, or the mean of the two middle values."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def run_started_at(run: dict | None, client: dict | None) -> tuple:
    """(epoch seconds, where it was read) -- the run's RECORDED start, which
    decides its fidelity bar version: `run.json`'s `started_at` (the runner's,
    written before it launched the test Tower: the run's earliest record), else
    `client.json`'s (a replay run without the runner). (None, None) when
    neither records one."""
    for source, record in (("run.json started_at", run), ("client.json started_at", client)):
        value = record.get("started_at") if isinstance(record, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), source
    return None, None


def fidelity_version(started_at) -> tuple:
    """(version, why): the fidelity bar in force when a run STARTED. v3 only
    for a start AFTER 2026-09-28 16:55 EDT; at or before it, or no recorded
    start, v1, the bar as written (managers 148 and 149: never loosen a bar
    after seeing the data it fails)."""
    if started_at is None:
        return "v1", (f"no recorded start, so the bar as written (v3 applies only to a run started after "
                      f"{FIDELITY_V3_APPLIES_AFTER_TEXT})")
    when = datetime.datetime.fromtimestamp(started_at, _EDT).isoformat(timespec="seconds")
    if started_at > FIDELITY_BAR["v3"]["applies_after"]:
        return "v3", f"the run started {when}, after v3's cut-off ({FIDELITY_V3_APPLIES_AFTER_TEXT})"
    return "v1", f"the run started {when}, at or before v3's cut-off ({FIDELITY_V3_APPLIES_AFTER_TEXT})"


def replay_fidelity(*, pacing: dict | None, client: dict, run: dict | None = None) -> dict:
    """Did the replay reproduce the recorded pace well enough to be PROOF?
    Judged against `FIDELITY_BAR`, in the version in force when the run
    started (`fidelity_version`; the start from `run` -- `run.json` -- else
    `client`): v1 (manager 142), or v3 (manager 149 §2) for a run started
    after 2026-09-28 16:55 EDT. The result says which version judged it.

    Its own verdict, apart from the code's live-safety verdict and the
    Environment's: a FAIL says nothing about the code under test, and makes
    the run invalid as proof (discard it and re-run). `n/a` -- no source
    journal to join, or no client record -- is NOT a pass.

      1. the `wire_seq` join is exact: every frame sent was joined, the Tower
         re-recorded nothing the source lacks, and the only source frames not
         joined are those the schedule never sent (the `first_seconds` cut);
      2. signed recorder-stamp offset error p95 and p99 within the bar, and
         v1: its |p50|;
         v3: its jitter, p95 |offset - median(offset)|, AND its constant
         bias, |median(offset)| (no |p50| clause);
      3. at most 5 % of the frames beyond +/-50 ms, and none more than 50 ms
         early;
      4. the client's own send lateness p95 within the bar, so that a FAIL on
         the Tower's side is the Tower's and not the harness's.

    WHAT IT MEASURES (review C24 HIGH-4). Clauses 1-3 judge the recorder's
    post-reply stamp (`tower_side_pacing`), not socket arrival: they judge
    whether the test Tower HANDLED each frame at the recorded pace, which
    includes the code under test's own parse, decode, CV and reply time.
    The numbers are unchanged; only what they are called. So a Tower-side
    FAIL with the client on time (clause 4 PASS) is not necessarily the
    harness's: it can be the candidate's own frame path, and `--compare`
    says so rather than calling it a bad replay (`tower_side_only`).
    """
    started_at, started_from = run_started_at(run, client)
    version, version_why = fidelity_version(started_at)
    bar = FIDELITY_BAR[version]
    result: dict = {"result": "n/a", "version": version, "version_why": version_why,
                    "family": FIDELITY_FAMILY[version], "measures": FIDELITY_MEASURES,
                    "started_at": started_at, "started_at_from": started_from,
                    "ruling": bar["ruling"], "bar": dict(bar), "rows": []}
    if not pacing or not pacing.get("computable"):
        result["why"] = (pacing or {}).get("why") or "no Tower-side pacing"
        result["note"] = FIDELITY_NA_NOTE
        return result
    rows = result["rows"]

    def row(check, value, required, ok):
        rows.append({"check": check, "value": value, "required": required,
                     "result": "n/a" if ok is None else ("PASS" if ok else "FAIL")})

    stream = client.get("stream") or {}
    sent = stream.get("frames_sent")
    scheduled = (client.get("schedule") or {}).get("frames")
    matched, replay_only = pacing.get("matched"), pacing.get("replay_only")
    source_only, source_frames = pacing.get("source_only"), pacing.get("source_frames")
    duplicates = pacing.get("source_duplicate_seq") or 0
    known = all(isinstance(v, int) for v in (sent, scheduled, matched, replay_only, source_only, source_frames))
    cut = source_frames - scheduled if known else None
    unmatched_source = source_only - cut if known else None
    row("wire_seq join exact: joined = frames sent; nothing unmatched on either side",
        f"{matched} joined / {sent} sent; {replay_only} replay-only; {source_only} source-only = "
        f"{cut} never scheduled (first_seconds {client.get('first_seconds')}) + {unmatched_source} unmatched"
        + (f"; {duplicates} duplicate source seq" if duplicates else ""),
        "joined = sent, 0 replay-only, 0 unmatched source frames",
        None if not known else (matched == sent and replay_only == 0 and unmatched_source == 0
                                and duplicates == 0))
    offset = pacing.get("offset_error_ms") or {}

    def number(value):
        return value if isinstance(value, (int, float)) else None

    p50, p95, p99, low = (number(offset.get(k)) for k in ("p50", "p95", "p99", "min"))
    if "offset_p50_abs_ms" in bar:              # v1
        row("recorder-stamp offset error |p50| (ms)", p50, f"<= {bar['offset_p50_abs_ms']:g}",
            None if p50 is None else abs(p50) <= bar["offset_p50_abs_ms"])
    if "offset_abs_deviation_p95_ms" in bar:    # v3
        jitter = number((pacing.get("offset_abs_deviation_ms") or {}).get("p95"))
        row("recorder-stamp offset jitter: p95 |offset - median(offset)| (ms)", jitter,
            f"<= {bar['offset_abs_deviation_p95_ms']:g}",
            None if jitter is None else jitter <= bar["offset_abs_deviation_p95_ms"])
    if "offset_median_abs_ms" in bar:           # v3
        median = number(pacing.get("offset_median_ms"))
        row("recorder-stamp offset constant bias: |median(offset)| (ms)", median,
            f"<= {bar['offset_median_abs_ms']:g}",
            None if median is None else abs(median) <= bar["offset_median_abs_ms"])
    row("recorder-stamp offset error p95 (ms)", p95, f"<= {bar['offset_p95_ms']:g}",
        None if p95 is None else p95 <= bar["offset_p95_ms"])
    row("recorder-stamp offset error p99 (ms)", p99, f"<= {bar['offset_p99_ms']:g}",
        None if p99 is None else p99 <= bar["offset_p99_ms"])
    late, early = pacing.get("late_over_50ms"), pacing.get("early_over_50ms")
    beyond = (late or 0) + (early or 0)
    fraction = beyond / matched if matched else None
    row(f"frames beyond +/-{bar['beyond_ms']:g} ms",
        None if fraction is None else f"{beyond} of {matched} ({fraction * 100:.2f} %): {late} late, {early} early",
        f"<= {bar['beyond_max_fraction'] * 100:g} %",
        None if fraction is None else fraction <= bar["beyond_max_fraction"])
    row(f"no frame more than {bar['early_max_ms']:g} ms early: min (ms)", low,
        f">= -{bar['early_max_ms']:g}", None if low is None else low >= -bar["early_max_ms"])
    lateness = number((stream.get("lateness_ms") or {}).get("p95"))
    row("client send lateness p95 (ms)",
        None if lateness is None else f"{lateness} ({stream.get('late_over_50ms')} frame(s) sent > 50 ms late, "
                                      "INFO)",
        f"<= {bar['client_lateness_p95_ms']:g}",
        None if lateness is None else lateness <= bar["client_lateness_p95_ms"])
    verdicts = [r["result"] for r in rows]
    result["result"] = ("FAIL" if "FAIL" in verdicts else "PASS" if all(v == "PASS" for v in verdicts)
                        else "n/a")
    # A FAIL on the Tower-side clauses alone, with the client on time: the
    # stamp is post-reply, so it can be the code under test's frame path
    # (review C24 HIGH-4). Reported, never a change to the verdict.
    result["tower_side_only"] = (result["result"] == "FAIL" and rows[-1]["result"] == "PASS"
                                 and rows[-1]["check"].startswith("client send lateness"))
    if result["result"] == "n/a":
        result["why"] = "a check had no value: " + ", ".join(r["check"] for r in rows if r["result"] == "n/a")
        result["note"] = FIDELITY_NA_NOTE
    return result


_GLOG =re.compile(r"^[IWEF](\d{8}) (\d{2}:\d{2}:\d{2})\.(\d+)\s")
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


PHOTOGRAPHIC_PREFIX = "lifecycle.photographic."


def _photographic_changes(phone_view: dict) -> tuple:
    """The phone's `lifecycle.photographic` words in receive order, and
    whether `scope` was recorded.

    A client from this change on keeps them itself (`phone_view.photographic`,
    scope included). An older client kept only the state-like leaves of each
    push (`phone_view.transitions`), which hold `state` and `stage` but never
    `scope`; they are replayed into the same shape."""
    if isinstance(phone_view.get("photographic"), list):
        return [item for item in phone_view["photographic"] if isinstance(item, dict)], True
    words, current = [], {}
    for item in phone_view.get("transitions") or []:
        if not isinstance(item, dict):
            continue
        for key in item.get("gone") or []:
            current.pop(key, None)
        current.update(item.get("changed") or {})
        word = {"t": item.get("t"), "state": current.get(PHOTOGRAPHIC_PREFIX + "state"),
                "stage": current.get(PHOTOGRAPHIC_PREFIX + "stage"), "scope": None, "scope_present": False}
        if word["state"] is None:
            continue
        if not words or (words[-1]["state"], words[-1]["stage"]) != (word["state"], word["stage"]):
            words.append(word)
    return words, False


def phone_photos(phone_view: dict, t0, room_appearance: dict | None) -> dict:
    """When the PHONE was told the room's photos were ready (review C22 H2:
    `phone_photos_at`). The client's own receive time of a status push, not
    the store's `updated_at`, which is when the worker wrote the record.

    `lifecycle.photographic` is one word for the room AND its areas
    (`WORLD-BUILDER-COMPONENTS.md` §3.4): the room's word while the room is
    unsettled, then the first unsettled area's word with `scope: "area"`,
    then `complete` once everything is. So the room's photos reach the phone
    at the first push after Stop that says `complete`, or that has moved on
    to an area -- provided the store says the room's appearance is `ok`
    (a room that FAILED also moves the word on to its areas).

    An older client did not keep `scope`. For it the move to an area is
    INFERRED from the word's stage going back from `appearance` to `surface`
    (a room builds surface then appearance; an area starts again at surface),
    and labelled so. `photographic_complete_at` is the first `complete`: the
    room and every area done.
    """
    words, scoped = _photographic_changes(phone_view or {})
    result = {"phone_photos_at": None, "how": None, "photographic_complete_at": None,
              "scope_recorded": scoped, "words": len(words)}
    if not words:
        result["why"] = "the client kept no lifecycle.photographic word (no subscription, or no client)"
        return result
    room_state = (room_appearance or {}).get("state") if isinstance(room_appearance, dict) else None
    previous_stage = None
    for word in words:
        t = word.get("t")
        if not isinstance(t, (int, float)) or (t0 is not None and t < t0):
            previous_stage = word.get("stage")
            continue
        state, stage = word.get("state"), word.get("stage")
        if state == "complete" and result["photographic_complete_at"] is None:
            result["photographic_complete_at"] = t
        if result["phone_photos_at"] is None:
            how = None
            if state == "complete":
                how = "the word said complete"
            elif scoped and word.get("scope") == "area":
                how = "the word moved on to an area (scope: area)"
            elif not scoped and previous_stage == "appearance" and stage == "surface":
                how = ("INFERRED: the word's stage went from appearance back to surface (an area's first "
                       "stage); this client did not record scope")
            if how is not None and room_state not in (None, "ok"):
                result["why"] = f"the word moved on at {t}, but the store's room appearance is {room_state}"
            elif how is not None:
                result["phone_photos_at"], result["how"] = t, how
        previous_stage = stage
    if result["phone_photos_at"] is not None and room_state is None:
        result["how"] += "; the store's room appearance was not available to confirm it"
    stored = (room_appearance or {}).get("updated_at") if isinstance(room_appearance, dict) else None
    if result["phone_photos_at"] is not None and isinstance(stored, (int, float)):
        result["store_to_phone_s"] = round(result["phone_photos_at"] - stored, 3)
    return result


# The client's receive stamps are rounded to the millisecond (`PhoneView`):
# a store write within half of that before the receipt is "at or before".
PHONE_STAMP_TOLERANCE_S = 0.0005
# The test Tower's Stop line against the client's own stream_stop send. Seven
# OLD runs measured 0.001-0.082 s (review F11 adversarial LOW-6).
STOP_AGREEMENT_S = 1.0


def photos_store_backing(phone_photos_at, appearance) -> dict:
    """Is the client's photos-ready time backed by the STORE (review F11
    MED-3)? Only when the world has a room-appearance stage, the store says
    `ok`, and it wrote it at or before the client was told (to the client
    stamp's rounding). A push that runs ahead of the store -- or a world with
    no appearance stage at all -- is a time with no room photos behind it.
    `backed` is None when there is no client time to back."""
    if phone_photos_at is None:
        return {"backed": None, "why": "no client photos-ready time to back"}
    if not isinstance(appearance, dict):
        return {"backed": False,
                "why": "the world has no room-appearance stage (session.json stages.appearance)"}
    state, stored = appearance.get("state"), appearance.get("updated_at")
    if state != "ok":
        return {"backed": False, "why": f"the store's room appearance is {state!r}, not ok"}
    if not isinstance(stored, (int, float)) or isinstance(stored, bool):
        return {"backed": False, "why": "the store's room appearance has no updated_at"}
    if stored > phone_photos_at + PHONE_STAMP_TOLERANCE_S:
        return {"backed": False, "store_to_phone_s": round(phone_photos_at - stored, 3),
                "why": f"the client was told {stored - phone_photos_at:.3f} s BEFORE the store wrote the room "
                       "appearance ok"}
    return {"backed": True, "store_to_phone_s": round(phone_photos_at - stored, 3)}


def source_manifests(client: dict, pacing: dict | None) -> dict | None:
    """Each source capture's `capture.json`, read from the source root the
    run is joined to and checked against the sha256 the stream pinned
    (client.json `source_journals`): {capture id: manifest}, or None."""
    pinned = client.get("source_journals") if isinstance(client, dict) else None
    root_name = pacing.get("source_capture_root") if isinstance(pacing, dict) else None
    if not isinstance(pinned, dict) or not pinned or not root_name:
        return None
    root = Path(root_name).resolve()
    manifests = {}
    for capture_id, expected in pinned.items():
        if not isinstance(expected, dict):
            return None
        directory = (root / str(capture_id)).resolve()
        try:
            directory.relative_to(root)
            data = (directory / "capture.json").read_bytes()
            manifest = json.loads(data.decode("utf-8"))
        except (ValueError, OSError, UnicodeDecodeError):
            return None
        if hashlib.sha256(data).hexdigest() != expected.get("capture.json") or not isinstance(manifest, dict):
            return None
        manifests[str(capture_id)] = manifest
    return manifests


def full_walk_evidence(client: dict, pacing: dict | None) -> dict:
    """Did the run stream the WHOLE recorded walk at the recorded pace
    (review F11 MED-4)? "Full" is not relative to whatever journal the run
    was given: it is the source captures' OWN manifests (`capture.json`,
    hash-checked by `source_manifests`) that say how much was recorded.
      * speed exactly 1.0, and no `first_seconds` cut;
      * the walk is a closed chain: its first capture continues nothing,
        each next one continues the one before, and the last ended by the
        wearer's Stop (a successor exists only after a disconnect);
      * per capture, the frames streamed are `frames_written` of them and
        their bytes `bytes_written` (Walk 5: 4005 frames, 75241745 bytes);
        and in all, frames streamed = scheduled = the sent list = the walk's
        `frames_written`.
    Anything else is NOT-PROOF against the absolute 9.0-min bar."""
    client = client if isinstance(client, dict) else {}
    problems = []
    speed = client.get("speed")
    if not isinstance(speed, (int, float)) or isinstance(speed, bool) or speed != 1.0:
        problems.append(f"speed {speed!r} is not the recorded pace (exactly 1.0)")
    if client.get("first_seconds") is not None:
        problems.append("first_seconds cut is not a full-walk proof")
    walk = [capture for capture in client.get("walk") or [] if isinstance(capture, dict)]
    ids = [str(capture.get("capture_id")) for capture in walk]
    manifests = source_manifests(client, pacing)
    images = client.get("source_images") if isinstance(client.get("source_images"), dict) else {}
    sent = images.get("sent")
    frames_sent = (client.get("stream") or {}).get("frames_sent") if isinstance(client.get("stream"), dict) else None
    scheduled = (client.get("schedule") or {}).get("frames") if isinstance(client.get("schedule"), dict) else None
    written = None
    if not ids or manifests is None or set(manifests) != set(ids):
        problems.append("the source captures' capture.json cannot be read and matched to the sha256 the stream "
                        "pinned, so the recorded walk's frames_written is unknown")
    elif not isinstance(sent, list) or not all(isinstance(entry, dict) for entry in sent):
        problems.append("no list of the frames sent (client.json source_images.sent)")
    else:
        if manifests[ids[0]].get("continues_capture") is not None:
            problems.append(f"the walk begins mid-chain: capture {ids[0]} continues "
                            f"{manifests[ids[0]].get('continues_capture')}")
        for before, after in zip(ids, ids[1:]):
            if manifests[after].get("continues_capture") != before:
                problems.append(f"capture {after} does not continue {before}")
        if manifests[ids[-1]].get("end_reason") != "stop":
            problems.append(f"the walk's last capture ended by {manifests[ids[-1]].get('end_reason')!r}, not the "
                            "wearer's Stop: the walk goes on in a capture this run did not stream")
        written = 0
        for capture_id in ids:
            manifest = manifests[capture_id]
            frames_written, bytes_written = manifest.get("frames_written"), manifest.get("bytes_written")
            mine = [entry for entry in sent if str(entry.get("capture_id")) == capture_id]
            mine_bytes = sum(entry["bytes"] for entry in mine if type(entry.get("bytes")) is int)
            if type(frames_written) is not int or type(bytes_written) is not int:
                problems.append(f"capture {capture_id}'s capture.json records no frames_written / bytes_written")
                written = None
                continue
            if written is not None:
                written += frames_written
            if frames_written != len(mine) or bytes_written != mine_bytes:
                problems.append(f"capture {capture_id}: {len(mine)} frame(s) of {mine_bytes} bytes streamed, but its "
                                f"capture.json records frames_written {frames_written}, bytes_written {bytes_written}")
        if written is not None and not (written == frames_sent == scheduled == len(sent)):
            problems.append(f"{frames_sent} frame(s) streamed ({scheduled} scheduled, {len(sent)} in the sent list) "
                            f"is not the recorded walk's frames_written {written}")
    return {"full": not problems, "problems": problems, "frames_written": written, "speed": speed}


def client_stop_agreement(client: dict, tower_stop) -> dict:
    """Is the W0 origin the wearer's Stop (review F11 adversarial LOW-6)? The
    origin is the test Tower's own `recording stopped (stop)` line; it must be
    within `STOP_AGREEMENT_S` of the client's own stream_stop send, or work the
    code under test adds before its Stop line is off the clock."""
    events = client.get("events") if isinstance(client, dict) and isinstance(client.get("events"), list) else []
    sends = [event.get("t") for event in events if isinstance(event, dict) and event.get("kind") == "stream_stop"
             and isinstance(event.get("t"), (int, float))]
    sent = sends[-1] if sends else None
    if sent is None or not isinstance(tower_stop, (int, float)):
        return {"agrees": False, "client_stop": sent, "tower_stop": tower_stop,
                "why": "no client stream_stop send, or no Tower Stop line, to set the W0 origin against"}
    gap = round(tower_stop - sent, 3)
    if abs(gap) > STOP_AGREEMENT_S:
        return {"agrees": False, "client_stop": sent, "tower_stop": tower_stop, "tower_minus_client_s": gap,
                "why": f"the test Tower logged Stop {gap:+.3f} s from the client's stream_stop send (more than "
                       f"{STOP_AGREEMENT_S:g} s): the W0 origin is not the wearer's Stop"}
    return {"agrees": True, "client_stop": sent, "tower_stop": tower_stop, "tower_minus_client_s": gap}


def admission_problems(*, phone_photos_at, backing: dict, walk: dict | None, stop: dict) -> list:
    """Why a run's W0 time cannot be proof for the absolute bar (review F11
    MED-3, MED-4, adversarial LOW-6): the render's NOT-PROOF reasons, and
    --compare's, from the run's sealed records."""
    problems = list((walk or {}).get("problems") or [])
    if phone_photos_at is not None:
        if backing.get("backed") is False:
            problems.append("store backing: " + backing["why"])
        if not stop.get("agrees"):
            problems.append("Stop: " + stop["why"])
    return problems


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


def _seconds(a, b):
    """b - a in seconds, to the millisecond the stamps carry (review F11
    MED-5): a bar is judged on this, never on `_minutes`' 0.01-min rounding,
    which reads 540.25 s as 9.0 min."""
    if a is None or b is None:
        return None
    return round(b - a, 3)


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
                surfaces: dict, rebuild_seconds: list, cadence: list, pacing: dict | None = None,
                run: dict | None = None) -> dict:
    """The C19 F8 live-safety gate, one row per check (review C22 H2).

    PASS / FAIL where the check has a required value; INFO where F8 asks for
    "no regression beyond old-path noise" (that is `--compare`'s job across
    N runs); n/a where this run holds no record of it (a report on a real
    walk's log has no client). Everything is read after the fact, from the
    Tower's log, the store and the run's `client.json`.

    `result` is the CODE's verdict. What :8000 was doing is the machine's,
    not the code's, so it is judged apart, in `environment` (review C22
    round 2, "Still open" 3): a contended run is reported, and does not make
    the code under test look unsafe.
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
        row(KEYFRAME_SEQUENCE_ROW,
            f"{accepted} accepted, {keyframes['count']} in keyframes.jsonl, sha256 {keyframes['sha256'][:16]}",
            "identical to baseline (--compare)",
            "INFO" if accepted in (None, keyframes["count"]) else "FAIL")
        lag = keyframes.get("observe_lag_s") or {}
        row(KEYFRAME_LAG_ROW,
            _dist_text(lag) + f"; {keyframes.get('observe_lag_unmatched') or 0} accepted keyframe(s) unmatched",
            "no regression (--compare), every accepted keyframe joined", "INFO" if lag.get("count") else "n/a")
    else:
        row(KEYFRAME_SEQUENCE_ROW, accepted, "identical to baseline (--compare)", "n/a")
        row(KEYFRAME_LAG_ROW, None, "no regression (--compare)", "n/a")
    row("rebuild latency (s)", _dist_text(distribution(rebuild_seconds)), "no regression (--compare)",
        "INFO" if rebuild_seconds else "n/a")
    row("rebuild cadence (s)", _dist_text(distribution(cadence)), "no regression (--compare)",
        "INFO" if cadence else "n/a")
    solves = timeline.get("background_solves") or []
    terminated = [s["n"] for s in solves if s.get("terminated_at_stop")]
    row("background-solve horizons: keyframes at launch; landed by (s after launch); terminated at Stop",
        f"{[s['keyframes'] for s in solves]}; landed by "
        f"{[s.get('horizon_s') for s in solves]}; terminated at Stop: "
        + (", ".join(f"solve {n}" for n in terminated) if terminated else "none"),
        "unchanged (--compare)", "INFO")
    if pacing and pacing.get("computable"):
        offset, gaps = pacing.get("offset_error_ms") or {}, pacing.get("inter_arrival_s") or {}
        row(PACING_ROW,
            f"{pacing.get('matched')} joined ({pacing.get('replay_only')} replay-only, "
            f"{pacing.get('source_only')} source-only"
            + (f", the walk cut at first_seconds {pacing['first_seconds']}" if pacing.get("first_seconds") else "")
            + f"); signed p50 {offset.get('p50')}, p95 {offset.get('p95')}, "
            f"p99 {offset.get('p99')}, max {offset.get('max')} (min {offset.get('min')}, p5 {offset.get('p5')}); "
            f"beyond 50 ms: {pacing.get('late_over_50ms')} late, {pacing.get('early_over_50ms')} early",
            PACING_ROW_REQUIRED, "INFO")
        row(INTERVAL_ROW,
            f"replayed {_dist_text(gaps.get('replayed') or {})} / recorded "
            f"{_dist_text(gaps.get('recorded') or {})}; difference (ms) "
            f"{_dist_text(pacing.get('inter_arrival_error_ms') or {})}",
            PACING_ROW_REQUIRED, "INFO")
    else:
        row(PACING_ROW, (pacing or {}).get("why"), PACING_ROW_REQUIRED, "n/a")
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
    judged = [r["result"] for r in rows if r["result"] in ("PASS", "FAIL")]
    result = "FAIL" if "FAIL" in judged else ("PASS" if judged else "n/a")
    return {"result": result, "rows": rows, "totals": totals,
            "environment": live_environment(client=client, run=run)}


def _guard_row(check: str, watch: dict | None, aborted: dict | None) -> dict:
    """One `:8000` guard record (a `LiveGuard.summary()`) as an environment row.

    Idle or down throughout is PASS. An `unknown` answer is TOLERATED when it
    stood alone: the guard aborts on two in a row, so in a run that did not
    abort on them every `unknown` was isolated -- a late answer from a loaded
    machine, not a Tower that could not be vouched for. Each is still shown.
    `recording`, `busy` (the run was contended) and an abort on consecutive
    `unknown`s FAIL.
    """
    if not watch:
        return {"check": check, "value": None, "required": "idle or down throughout; an isolated unknown "
                "tolerated", "result": "n/a"}
    seen = watch.get("states_seen") or []
    history = watch.get("history") or []
    unknowns = sum(1 for item in history if isinstance(item, dict) and item.get("state") == "unknown")
    reason = (aborted or {}).get("reason") or ""
    failed_closed = "unknown" in seen and "failing closed" in reason
    if failed_closed and unknowns:
        # `history` keeps CHANGES of state (`LiveGuard.observe`), so the run
        # of `unknown`s that aborted is one entry however many answers it
        # held (review C22 round 3 L-d). The guard aborts at exactly the Nth
        # in a row, which its reason states; every earlier run was a single
        # answer, or it would have aborted then.
        streak = re.search(r"answer (\d+) times in a row", reason)
        unknowns += (int(streak.group(1)) if streak else _unknown_abort_after()) - 1
    contended = [s for s in seen if s in ("recording", "busy")]
    value = ", ".join(seen) + (f"; busy from {_clock(watch.get('busy_since'))}" if watch.get("busy_since") else "")
    if unknowns:
        value += (f"; {unknowns} unknown answer(s)"
                  + (" -- two in a row aborted the run" if failed_closed else ", each isolated: tolerated"))
    return {"check": check, "value": value,
            "required": "idle or down throughout; an isolated unknown tolerated",
            "result": "FAIL" if contended or failed_closed else "PASS"}


def _unknown_abort_after() -> int:
    from scripts.world_live_replay import LIVE_UNKNOWN_ABORT_AFTER  # noqa: PLC0415

    return LIVE_UNKNOWN_ABORT_AFTER


def client_recorded(client) -> bool:
    """Whether a replay client's own record is at hand (a real walk's log has
    none): the W0 verdict's basis (review C24 HIGH-1)."""
    return isinstance(client, dict) and any(key in client for key in ("tool", "stream", "phone_view", "handshake"))


def _streamed(client: dict) -> bool:
    sent = (client.get("stream") or {}).get("frames_sent")
    return isinstance(sent, int) and not isinstance(sent, bool) and sent > 0


def _startup_watch_complete(run: dict | None) -> bool:
    watch = (run or {}).get("live_tower_watch_startup")
    polls = watch.get("poll_count") if isinstance(watch, dict) else None
    background = watch.get("background_poll_count") if isinstance(watch, dict) else None
    gap = watch.get("background_max_gap_s") if isinstance(watch, dict) else None
    return (isinstance(polls, int) and not isinstance(polls, bool) and polls > 0
            and isinstance(background, int) and not isinstance(background, bool) and background > 0
            and isinstance(gap, (int, float)) and not isinstance(gap, bool) and 0 <= gap <= 10.0)


def proof_status(*, client: dict, run: dict | None) -> dict:
    """Can this run be proof at all? (review C24 HIGH-3.) NOT-PROOF when it
    was declared so (`--not-a-proof-run`), when the :8000 guard was off
    (`--no-live-guard`), when it streamed with no :8000 watch on record, or
    when there is no replay client record. `--compare` never counts it."""
    run = run or {}
    reasons = []
    if not client_recorded(client):
        reasons.append("no replay client record (not a replay run)")
    if client.get("not_a_proof_run") or run.get("not_a_proof_run"):
        reasons.append("declared --not-a-proof-run")
    outcome = client.get("outcome")
    if isinstance(outcome, str) and outcome != "settled":
        reasons.append(f"replay outcome was {outcome}, not settled")
    if client.get("live_guard") is False or run.get("live_guard") is False:
        reasons.append("the :8000 guard was off (--no-live-guard)")
    elif _streamed(client) and not client.get("live_tower_watch"):
        reasons.append("it streamed with no :8000 watch on record")
    if client.get("live_guard_error"):
        reasons.append("the :8000 guard failed: " + str(client["live_guard_error"]))
    if not reasons and not _startup_watch_complete(run):
        reasons.append("no completed :8000 startup/preflight watch on record")
    return {"proof": not reasons, "not_proof_reasons": reasons}


def live_environment(*, client: dict, run: dict | None) -> dict:
    """What `:8000` did while the test Tower ran: the machine, not the code.
    From the client's watch (the stream and the settle) and, when the runner
    wrote one, its own (the test Tower's startup and idle wait).

    A stream that nobody watched is a FAIL, never n/a (review C24 HIGH-3):
    the runner's startup watch alone must not make an unguarded stream PASS."""
    watch = client.get("live_tower_watch")
    if client.get("live_guard_error"):
        rows = [{"check": ":8000 during the run",
                 "value": "WATCH FAILED: " + str(client["live_guard_error"]),
                 "required": "idle or down throughout; an isolated unknown tolerated", "result": "FAIL"}]
    elif not watch and (client.get("live_guard") is False or _streamed(client)):
        why = ("the guard was off (--no-live-guard)" if client.get("live_guard") is False
               else "the client streamed and kept no :8000 watch")
        rows = [{"check": ":8000 during the run", "value": f"NOT WATCHED: {why}",
                 "required": "idle or down throughout; an isolated unknown tolerated", "result": "FAIL"}]
    else:
        rows = [_guard_row(":8000 during the run", watch, client.get("aborted"))]
    startup = (run or {}).get("live_tower_watch_startup")
    if _startup_watch_complete(run):
        aborted = (run or {}).get("aborted")
        rows.append(_guard_row(":8000 during the test Tower's startup", startup,
                               aborted if (aborted or {}).get("during") == "startup" else None))
    elif _streamed(client):
        rows.append({"check": ":8000 during the test Tower's startup",
                     "value": "NOT WATCHED: no completed background startup/preflight poll or bounded cadence on record",
                     "required": "background startup/preflight polls with <=10 s maximum gap", "result": "FAIL"})
    judged = [r["result"] for r in rows if r["result"] in ("PASS", "FAIL")]
    return {"result": "FAIL" if "FAIL" in judged else ("PASS" if judged else "n/a"), "rows": rows}


def _dist_text(stats: dict) -> str:
    if not stats or not stats.get("count"):
        return "none"
    return (f"n {stats['count']}, p50 {stats.get('p50')}, p95 {stats.get('p95')}, "
            f"p99 {stats.get('p99')}, max {stats.get('max')}")


CAPTURE_ROOT_GIVEN = "--capture-root"
CAPTURE_ROOT_FROM_CLIENT = "client.json capture_root (where the replay read the walk)"
CAPTURE_ROOT_DEFAULT = ("the replay's default capture root (the live store AT RENDER TIME: this client "
                        "recorded none; pass --capture-root <a snapshot> for a pinned run)")


def default_capture_root(client: dict):
    """Where a run's source captures were read: its client record says so
    from C22-F2 on; before that it was always the replay's default."""
    return resolve_capture_root(None, client)[0]


def resolve_capture_root(given, client: dict) -> tuple:
    """The source-capture root the pacing row joins to, and where that came
    from (review C22 round 3 L-g). An explicit `--capture-root` wins -- a
    proof set's re-render names the RUN's snapshot of the source journal --
    then the run's `client.json` value, then the replay's default, which is
    the live store as it is at render time and is labelled so."""
    if given is not None:
        return Path(given), CAPTURE_ROOT_GIVEN
    if client.get("capture_root"):
        return Path(client["capture_root"]), CAPTURE_ROOT_FROM_CLIENT
    from scripts.world_live_replay import DEFAULT_CAPTURE_ROOT  # noqa: PLC0415

    return DEFAULT_CAPTURE_ROOT, CAPTURE_ROOT_DEFAULT


def build_report(*, tower_log, tower_out_log=None, world_root=None, capture_id=None,
                 client=None, samples=None, label=None, run_dir=None, data_root=None,
                 capture_root=None, run=None, capture_root_from=None) -> dict:
    """Everything, as one JSON-able dict. `render_markdown` draws it.

    `run_dir` is the run's `--out` (read): its `solution-snapshots/`.
    `data_root` is the test Tower's (read): its re-recorded captures, for the
    Tower-side pacing row. `capture_root` holds the source captures (read);
    by default the one the client record names (`resolve_capture_root`);
    `capture_root_from` says where a given one came from. `run` is the
    runner's `run.json`: its own :8000 watch during the Tower's startup.
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
        stopped = timeline.get("bg_solve_stop_requested")
        launched, done = timeline.get("final_launched"), timeline.get("final_done")
        cursor = catchup_end
        gap_end = _t(launched) or _t(wait) or _t(stopped)
        if catchup_end is not None and gap_end is not None and gap_end - catchup_end > 1.0:
            if wait is not None:
                stage = (f"Wait for background solve pid {wait['pid']} ({wait['s']} s), then terminate it "
                         "and launch the final solve")
                evidence = f"L{wait['line']}"
            elif stopped is not None:
                stage = (f"Background solve pid {stopped['pid']} terminated: a stop was requested (a hard "
                         "stop ends the wait at once)")
                evidence = f"L{stopped['line']}"
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

    store_minutes = _minutes(t0, milestones.get("room_appearance_ok"))

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
    if client:
        capture_root, resolved_from = resolve_capture_root(capture_root, client)
        capture_root_from = capture_root_from or resolved_from
    pacing = tower_side_pacing(client=client, data_root=data_root, capture_root=capture_root,
                               capture_root_from=capture_root_from) if client else {
        "computable": False, "why": "no client record (a real walk's log has none)"}
    fidelity = replay_fidelity(pacing=pacing, client=client, run=run)
    appearance_stage = stages.get("appearance") if facts.get("available") else None
    photos_told = phone_photos(client.get("phone_view") or {}, t0, appearance_stage)
    # Review F11 MED-3: a photos-ready time counts only with the store's room
    # appearance `ok` written at or before it.
    photos_told["store_backing"] = photos_store_backing(photos_told.get("phone_photos_at"), appearance_stage)
    if photos_told.get("phone_photos_at") is not None:
        milestones["phone_photos_at"] = photos_told["phone_photos_at"]
    if photos_told.get("photographic_complete_at") is not None:
        milestones["phone_photographic_complete_at"] = photos_told["photographic_complete_at"]
    # THE W0 VERDICT (review C24 HIGH-1): Stop -> the replay CLIENT's receipt
    # of the room's photos. The store's `updated_at` is INFO -- it says when a
    # worker wrote the record, not that any client was told. Without a client
    # record (a real walk's log) only the store time exists, and the verdict
    # says it is NOT the client measure.
    phone_minutes = _minutes(t0, milestones.get("phone_photos_at"))
    has_client = client_recorded(client)
    photos_minutes = phone_minutes if has_client else store_minutes
    verdict = ("PASS" if photos_minutes is not None and photos_minutes <= HARD_MAX_MINUTES
               else "FAIL" if photos_minutes is not None else "NOT REACHED")
    not_reached_why = None
    if verdict == "NOT REACHED":
        not_reached_why = ((photos_told.get("why") or "no push after Stop told the client the room's photos were "
                            "ready") if has_client else "the store has no room appearance `ok`")
    safety = live_safety(client=client, timeline=timeline, session=session, keyframes=keyframes,
                         surfaces=surfaces, rebuild_seconds=rebuild_seconds, cadence=cadence,
                         pacing=pacing, run=run)

    built = {
        "report": "c22-live-replay/3",
        "label": label or client.get("label"),
        "generated_at": round(time.time(), 3),
        "inputs": {"tower_log": str(tower_log), "tower_out_log": None if tower_out_log is None
                   else str(tower_out_log), "world_root": None if world_root is None else str(world_root),
                   "samples": None if samples is None else str(samples),
                   "run_dir": None if run_dir is None else str(run_dir)},
        "harness": client.get("harness"),
        # Which harness STREAMED this run, and which rendered this report
        # (manager 154 §2: the pin and version are self-evident).
        "harness_pin": harness_pin_of((run or {}).get("harness") or client.get("harness")),
        "rendered_by": rendered_by(),
        "hard_max_minutes": HARD_MAX_MINUTES,
        "verdict": {"basis": "phone" if has_client else "store",
                    "measure": BASIS_PHONE if has_client else BASIS_STORE,
                    "stop_to_room_with_photos_min": photos_minutes, "result": verdict,
                    "not_reached_why": not_reached_why,
                    # How `phone_photos_at` was read: INFERRED for a client that
                    # recorded no `scope` (made before C22-F2).
                    "phone_photos_how": photos_told.get("how") if has_client else None,
                    "stop_to_phone_photos_min": phone_minutes,
                    # The same, in seconds to the ms (review F11 MED-5: bars
                    # are judged on this, never on the 0.01-min rounding).
                    "stop_to_phone_photos_s": _seconds(t0, milestones.get("phone_photos_at")),
                    # INFO: when the store wrote the room appearance `ok`.
                    "stop_to_store_photos_min": store_minutes,
                    "stop_to_settled_min": _minutes(t0, milestones.get("settled")),
                    "stop_to_finalization_min": _minutes(t0, milestones.get("finalization_complete"))},
        # Can this run be proof at all (review C24 HIGH-3)?
        "proof": proof_status(client=client, run=run),
        "live_safety": safety,
        # Apart from the code's verdict and the Environment's (manager 142).
        "replay_fidelity": fidelity,
        "tower_side_pacing": pacing,
        # Review F11 MED-4: the whole recorded walk, at the recorded pace, by
        # the source captures' own manifests; and LOW-6: the W0 origin.
        "full_walk": full_walk_evidence(client, pacing) if has_client else None,
        "stop_agreement": client_stop_agreement(client, t0) if has_client else None,
        "phone_photos": photos_told,
        "keyframes": keyframes,
        "solve_draw_0": zero,
        "live_surfaces_latency": surfaces,
        "client": {k: client.get(k) for k in ("outcome", "speed", "first_seconds", "after_stop", "options",
                                              "capture_root", "walk", "schedule", "source_images", "source_journals",
                                              "calibration_check",
                                              "stream", "phone_fetches", "handshake", "session_start",
                                              "live_tower_at_start", "live_tower_watch", "target_listener",
                                              "live_guard", "not_a_proof_run",
                                              "aborted", "started_at", "t0", "stopped_at")
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
    if run:
        built["run"] = run
    if run_dir is not None:
        # The records this render read, sealed (review F11 LOW-9). The runner's
        # own render is the RUN-TIME seal (`write_report` marks it).
        built["evidence_seal"] = evidence_seal(run_dir, run, timeline)
    # What a run must share with the runs --compare sets it against (review
    # C24 HIGH-2); recomputed there from the report, shown here.
    built["comparability_key"] = comparability_key(built)
    # A malformed or absent version cannot exempt a proof-labelled render
    # from the input evidence this harness promises. Comparison also rejects
    # incomplete keys, but a standalone report must fail closed on its own.
    missing_input = [name for name in ("source_journal_sha256", "source_jpegs_sha256", "calibration")
                     if built["comparability_key"].get(name) is None]
    if built["proof"]["proof"] and missing_input:
        built["proof"]["proof"] = False
        built["proof"]["not_proof_reasons"].append(
            "input evidence missing or changed: " + ", ".join(missing_input))
    # Review F11 MED-3, MED-4, adversarial LOW-6: a W0 time with no room
    # photos in the store behind it, a run that is not the whole recorded walk
    # at the recorded pace, or one whose Stop line is not the client's Stop, is
    # not proof against the absolute bar.
    admission = admission_problems(phone_photos_at=photos_told.get("phone_photos_at"),
                                   backing=photos_told["store_backing"], walk=built["full_walk"],
                                   stop=built["stop_agreement"] or {})
    if built["proof"]["proof"] and admission:
        built["proof"]["proof"] = False
        built["proof"]["not_proof_reasons"].extend(admission)
    return built


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
        pin = harness_pin_of(harness)
        lines.append(f"Harness pin (the harness that STREAMED this run): version "
                     f"`{pin.get('version') or 'not recorded (made before C22-F7)'}`, git HEAD "
                     f"`{harness.get('git_head') or 'not a checkout'}`"
                     + (f" with uncommitted changes {dirty}" if dirty else (" (committed, clean)" if dirty == [] else ""))
                     + "; streaming scripts sha1 "
                     + ", ".join(f"{name} `{sha}`" for name, sha in (pin.get("streaming_sha1") or {}).items())
                     + f"; all three scripts sha1 `{harness.get('sha1')}`.")
        lines.append("")
    rendered = report.get("rendered_by") or {}
    if rendered:
        lines.append(f"Rendered by harness version `{rendered.get('version')}`, world_live_replay_report.py sha1 "
                     f"`{rendered.get('report_sha1')}`.")
        lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    proof = report.get("proof") or {}
    if proof and not proof.get("proof"):
        lines.append(f"**NOT-PROOF:** {'; '.join(proof.get('not_proof_reasons') or [])}. `--compare` never counts "
                     "this run (review C24 HIGH-3).")
        lines.append("")
    photos = verdict.get("stop_to_room_with_photos_min")
    shown = f"{photos} min" if photos is not None else f"not reached ({verdict.get('not_reached_why')})"
    store = verdict.get("stop_to_store_photos_min")
    if verdict.get("basis") == "store":
        lines.append(f"1. **Stop to room with photos: {shown} -- the STORE's `updated_at`, NOT the client "
                     f"measure** (no client record: a real walk's log) against the "
                     f"{report['hard_max_minutes']:.0f}-min hard maximum: **{verdict.get('result')}**.")
    else:
        lines.append(f"1. **Stop to room with photos, as the replay client received it (`phone_photos_at`): "
                     f"{shown}** against the {report['hard_max_minutes']:.0f}-min hard maximum: "
                     f"**{verdict.get('result')}**. This is the replay client's receipt of the status push, not "
                     "photos rendered on a phone."
                     + (" **The client's time is INFERRED**: this client did not record the word's `scope` (a "
                        "record made before C22-F2), so the move to an area is read from its stage."
                        if str(verdict.get("phone_photos_how") or "").startswith("INFERRED") else "")
                     + (f" INFO: the store wrote the room appearance `ok` at +{store} min (not the W0 measure)."
                        if store is not None else ""))
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
    safety_head = report.get("live_safety") or {}
    fidelity_head = report.get("replay_fidelity") or {}
    lines.append(f"5. Verdicts, kept apart: code live safety **{safety_head.get('result')}**; Environment "
                 f"(:8000) **{(safety_head.get('environment') or {}).get('result')}**; replay fidelity "
                 f"**{fidelity_head.get('result')}**"
                 + (" (n/a is NOT a pass for a proof set)" if fidelity_head.get("result") == "n/a" else "")
                 + (f", judged by fidelity bar {fidelity_head['version']}" if fidelity_head.get("version") else "")
                 + ". Proof: " + ("**NOT-PROOF**" if proof and not proof.get("proof") else "eligible") + ".")
    if client.get("aborted"):
        lines.append(f"6. **Aborted** at {_clock(client['aborted'].get('t'))}: {client['aborted'].get('reason')}.")
    lines.append("")
    lines.append(f"Limits (review C24 MED-7): {REPLAY_LIMITS}")
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
    environment = safety.get("environment") or {}
    if environment:
        lines.append("")
        lines.append(f"**Environment (the `:8000` guard): {environment.get('result')}.** What the live Tower "
                     "was doing is the machine's state, not the code's, so it is judged apart from the verdict "
                     "above. An isolated `unknown` answer is tolerated and shown; two in a row abort the run.")
        lines.append("")
        lines.append("| Check | Value | Required | Result |")
        lines.append("|---|---|---|---|")
        for item in environment.get("rows") or []:
            value = item.get("value")
            value = "—" if value is None else str(value).replace("|", "/")
            lines.append(f"| {item['check']} | {value} | {item['required']} | {item['result']} |")
    fidelity = report.get("replay_fidelity") or {}
    if fidelity:
        lines.append("")
        judged_by = (f"**fidelity bar {fidelity['version']}**, {fidelity.get('ruling')} -- the version in force "
                     f"when the run started: {fidelity.get('version_why')}"
                     + (f" ({fidelity['started_at_from']})" if fidelity.get("started_at_from") else "")
                     if fidelity.get("version") else f"the bar approved by {fidelity.get('ruling')}")
        lines.append(f"**Replay fidelity: {fidelity.get('result')}.** Did the test Tower handle the frames at the "
                     f"recorded pace? Judged apart from the code's verdict and the Environment's, against {judged_by}. "
                     "A FAIL makes the run invalid as proof: discard it and re-run.")
        lines.append("")
        lines.append(f"What the bar measures (review C24 HIGH-4): {fidelity.get('measures') or FIDELITY_MEASURES}."
                     + (" **This FAIL is on the Tower-side clauses alone, with the client on time: inspect the "
                        "code under test's frame path before discarding the run.**"
                        if fidelity.get("tower_side_only") else ""))
        if fidelity.get("result") == "n/a":
            lines.append("")
            lines.append(f"n/a: {fidelity.get('why')}. **{fidelity.get('note') or FIDELITY_NA_NOTE}**")
        if fidelity.get("rows"):
            lines.append("")
            lines.append("| Check | Value | Required | Result |")
            lines.append("|---|---|---|---|")
            for item in fidelity["rows"]:
                value = item.get("value")
                value = "—" if value is None else str(value).replace("|", "/")
                lines.append(f"| {item['check']} | {value} | {item['required']} | {item['result']} |")
    pacing = report.get("tower_side_pacing") or {}
    if pacing.get("computable"):
        origins = pacing.get("origins") or {}
        digests = "; ".join(f"{cid[:8]}… frames.jsonl sha256 {str((d or {}).get('frames.jsonl'))[:16]}, "
                            f"capture.json {str((d or {}).get('capture.json'))[:16]}"
                            for cid, d in (pacing.get("source_sha256") or {}).items())
        lines.append("")
        lines.append(f"Tower-side pacing (the recorder's post-reply stamps, not socket arrival): "
                     f"`{pacing.get('data_root')}` captures {pacing.get('replay_captures')} "
                     f"joined on `wire_seq` to `{pacing.get('source_capture_root')}` captures "
                     f"{pacing.get('source_captures')}, speed {pacing.get('speed')}; origins are each walk's "
                     f"first `started_at` ({_clock(origins.get('replay_started_at'))} replayed, "
                     f"{_clock(origins.get('source_started_at'))} recorded). The source root is from "
                     f"{pacing.get('source_capture_root_from') or 'the caller'}"
                     + (f"; source journal {digests}" if digests else "") + ".")
    photos = report.get("phone_photos") or {}
    if photos.get("phone_photos_at") is not None or photos.get("why"):
        lines.append("")
        lines.append(f"Phone told the room's photos were ready: {_clock(photos.get('phone_photos_at'))} "
                     f"({photos.get('how') or photos.get('why')})"
                     + (f"; {photos['store_to_phone_s']} s after the store's room appearance `ok`"
                        if photos.get("store_to_phone_s") is not None else "")
                     + (f"; every photographic stage complete: {_clock(photos.get('photographic_complete_at'))}"
                        if photos.get("photographic_complete_at") else "") + ".")
    backing = photos.get("store_backing") or {}
    if backing.get("backed") is False:
        lines.append("")
        lines.append(f"**The client's photos-ready time is NOT backed by the store** (review F11 MED-3): "
                     f"{backing.get('why')}. NOT-PROOF.")
    walk_check = report.get("full_walk")
    if isinstance(walk_check, dict):
        lines.append("")
        lines.append("Full walk at the recorded pace, by the source captures' own capture.json (review F11 MED-4): "
                     + ("yes" if walk_check.get("full") else
                        "**NO** (NOT-PROOF for the absolute bar): " + "; ".join(walk_check.get("problems") or []))
                     + f" (frames_written {walk_check.get('frames_written')}, speed {walk_check.get('speed')}).")
    stop_check = report.get("stop_agreement")
    if isinstance(stop_check, dict) and stop_check.get("tower_minus_client_s") is not None:
        lines.append("")
        lines.append(f"W0 origin: the test Tower's Stop line is {stop_check['tower_minus_client_s']:+.3f} s from the "
                     f"client's stream_stop send"
                     + (" (within 1 s)." if stop_check.get("agrees") else f": **{stop_check.get('why')}**."))
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
        landed = solve.get("landed_by") or {}
        earlier = landed.get("possibly_earlier")
        unknown = landed.get("landing_rebuild_unknown")
        fate = ("; TERMINATED at Stop" + (f" ({solve['terminated_how']})" if solve.get("terminated_how") else "")
                + (f" (L{solve['terminated_line']})" if solve.get("terminated_line") else "")
                if solve.get("terminated_at_stop") else
                f"; landed by {_clock(landed.get('t'))} (+{solve.get('horizon_s')} s, {landed.get('via')}, "
                f"L{landed.get('line')}"
                + (f"; landed-by rebuild {landed['rebuild_n']} L{landed['rebuild_line']}: {landed.get('rebuild_why')}"
                   if landed.get("rebuild_line") else "")
                + (f"; possibly earlier, rebuild {earlier['rebuild_n']} L{earlier['rebuild_line']} "
                   f"(+{earlier['after_launch_s']} s): it launched live surface {earlier['surface']} -- "
                   f"OPEN: {earlier['open']}" if earlier else "")
                + (f"; landing rebuild UNKNOWN: live surface {unknown['surface']} at rebuild {unknown['rebuild_n']} "
                   f"L{unknown['rebuild_line']}, +{unknown['after_launch_s']} s after this launch, is not read as "
                   f"its landing -- OPEN: {unknown['open']}" if unknown else "")
                + ")"
                if landed else "; landing not seen")
        between = solve.get("live_surface_between")
        if between:
            fate += (f"; live surface {between.get('n')} launched at {_clock(between.get('t'))} by rebuild "
                     f"{between.get('rebuild_n')} (L{between.get('rebuild_line')}), attributed by log line "
                     "(OPEN whose landing it marks, or a stale surface's relaunch: the log cannot tell)")
        lines.append(f"- background solve {solve['n']} at {_clock(solve['t'])}, {solve['keyframes']} keyframes "
                     f"(L{solve['line']}){fate}")
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
    """report.json, REPORT.md and keyframes.json into `out_dir`.

    A render written INTO ITS OWN RUN DIRECTORY is the runner's (or the
    standalone client's) run-time render -- `main` refuses a non-empty --out,
    and a run directory always holds client.json -- so its evidence seal is
    marked `run time` and printed to stderr for the RUN ledger (review F11
    LOW-9). Any other render's seal is marked `render`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seal = report.get("evidence_seal")
    run_dir = (report.get("inputs") or {}).get("run_dir")
    if isinstance(seal, dict):
        seal["at"] = "run time" if run_dir and _norm_path(run_dir) == _norm_path(out_dir) else "render"
    json_path = out_dir / "report.json"
    md_path = out_dir / "REPORT.md"
    tmp = out_dir / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, json_path)
    if isinstance(seal, dict) and seal["at"] == "run time":
        print(f"[report] evidence seal {seal['seal']} for {run_dir}: copy this line to the RUN ledger "
              "(--compare counts the run only while its records still hash to it)", file=sys.stderr, flush=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    keyframes = report.get("keyframes")
    if keyframes:
        (out_dir / "keyframes.json").write_text(json.dumps({
            "identity": ("the keyframe SELECTION sequence (source_seq, segment_index), in keyframes.jsonl order: "
                         "which frames were selected, not their image content, not capture-qualified"),
            "count": keyframes.get("count"), "sha256": keyframes.get("sha256"),
            "sequence": keyframes.get("sequence")}), encoding="utf-8")
    return json_path, md_path


# -- comparing runs (N old vs N new) --------------------------------------------------


def comparable_metrics(report: dict) -> dict:
    """One report's numbers, flat, for `--compare`.

    THE W0 TIMING METRIC is `stop_to_phone_photos_min` (review C24 HIGH-1):
    Stop to the replay client's receipt of the room's photos. The store's
    time is `stop_to_store_photos_min`, INFO. A render made before C22-F7 has
    no verdict `basis`: its `stop_to_room_with_photos_min` was the store's.

    Names say what they measure (review C24 HIGH-4, MED-8): the keyframe lag
    is over ACCEPTED keyframes only, and the pacing metrics are the
    recorder's post-reply stamps, not arrivals."""
    metrics: dict = {}
    verdict = report.get("verdict") or {}
    metrics[W0_TIMING_METRIC] = verdict.get("stop_to_phone_photos_min")
    metrics["stop_to_store_photos_min"] = (verdict.get("stop_to_store_photos_min") if "basis" in verdict
                                           else verdict.get("stop_to_room_with_photos_min"))
    for key in ("stop_to_finalization_min", "stop_to_settled_min"):
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
    keyframes = report.get("keyframes") or {}
    for name, stats in (("send_lateness_ms", stream.get("lateness_ms") or {}),
                        ("keyframe_accept_lag_s", keyframes.get("observe_lag_s") or {}),
                        ("rebuild_s", (walk.get("rebuilds") or {}).get("seconds") or {}),
                        ("rebuild_cadence_s", (walk.get("rebuilds") or {}).get("cadence_s") or {}),
                        ("live_surface_latency_s",
                         (report.get("live_surfaces_latency") or {}).get("latency_s") or {})):
        for q in ("p50", "p95", "p99", "max"):
            metrics[f"{name}.{q}"] = stats.get(q)
    # An accepted keyframe the lag could not join (review C24 MED-8): the lag
    # distribution is incomplete by that many.
    metrics["keyframe_accept_lag_unmatched"] = keyframes.get("observe_lag_unmatched")
    pacing = report.get("tower_side_pacing") or {}
    if pacing.get("computable"):
        for name, stats in (("recorder_stamp_offset_error_ms", pacing.get("offset_error_ms") or {}),
                            ("recorder_stamp_interval_s", (pacing.get("inter_arrival_s") or {}).get("replayed") or {}),
                            ("recorder_stamp_interval_error_ms", pacing.get("inter_arrival_error_ms") or {})):
            for q in ("p50", "p95", "p99", "max"):
                metrics[f"{name}.{q}"] = stats.get(q)
        metrics["recorder_stamp_offset_error_ms.min"] = (pacing.get("offset_error_ms") or {}).get("min")
        metrics["recorder_stamp_beyond_50ms"] = (pacing.get("late_over_50ms") or 0) + (pacing.get("early_over_50ms") or 0)
    solves = walk.get("background_solves") or []
    landed = [s.get("horizon_s") for s in solves if isinstance(s.get("horizon_s"), (int, float))]
    if any("terminated_at_stop" in s for s in solves):
        stats = distribution(landed)
        for q in ("p50", "p95", "max"):
            metrics[f"bg_solve_landed_by_s.{q}"] = stats.get(q)
        metrics["bg_solves_terminated_at_stop"] = sum(1 for s in solves if s.get("terminated_at_stop"))
    metrics["rebuilds"] = (walk.get("rebuilds") or {}).get("count")
    metrics["background_solves"] = len(walk.get("background_solves") or [])
    metrics["live_surfaces"] = len(walk.get("live_surfaces") or [])
    zero = report.get("solve_draw_0") or {}
    metrics["draw0_map_s"] = zero.get("map_s")
    for row in report.get("waterfall") or []:
        if row.get("n") and not row.get("child") and row.get("minutes") is not None:
            metrics[f"row {row['n']} min"] = row["minutes"]
    return metrics


# THE COMPARABILITY KEY (review C24 HIGH-2), field by field. `timing_env` and
# `interpreter` are review F11 MED-2's.
KEY_FIELDS = ("source_captures", "source_journal_sha256", "source_jpegs_sha256", "switches", "code", "harness", "replay",
              "calibration", "fidelity_family", "timing_env", "interpreter")
# The timing variables the runner records in run.json `timing_env`
# (`world_live_replay_run.TIMING_ENV_KEYS` at 807054d): inherited from the
# launcher's shell, not scrubbed, and each changes the test Tower's speed.
# Every one must be on record (None = unset, which is not ""). A runner that
# records more has every one it records bound too.
TIMING_ENV_REQUIRED = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                       "PYTORCH_CUDA_ALLOC_CONF")
# The harness that STREAMED the run. The report script is not part of the
# pin: a re-render reads, it does not stream.
STREAMING_HARNESS_FILES = ("world_live_replay.py", "world_live_replay_run.py")


def _ordered_digest(entries: list[dict]) -> str:
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _verified_source_jpegs(client: dict, pacing: dict) -> str | None:
    """Recheck the stream's per-frame hashes against the pinned source at render/compare time."""
    evidence = client.get("source_images")
    if not isinstance(evidence, dict) or evidence.get("verified") is not True:
        return None
    planned, sent = evidence.get("planned"), evidence.get("sent")
    if not isinstance(planned, list) or not planned or planned != sent:
        return None
    digest = _ordered_digest(planned)
    if digest != evidence.get("planned_sha256") or digest != evidence.get("sent_sha256"):
        return None
    if len(sent) != (client.get("schedule") or {}).get("frames"):
        return None
    root_name = pacing.get("source_capture_root")
    if not root_name:
        return None
    root = Path(root_name).resolve()
    for item in sent:
        if not isinstance(item, dict) or not all(k in item for k in
                ("capture_id", "relpath", "wire_seq", "bytes", "sha256")):
            return None
        path = (root / str(item["capture_id"]) / str(item["relpath"])).resolve()
        try:
            path.relative_to(root)
            data = path.read_bytes()
        except (ValueError, OSError):
            return None
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            return None
    return digest


def _verified_source_journals(client: dict, pacing: dict) -> dict | None:
    """The journals joined at render/compare must be the bytes that built the stream schedule."""
    pinned = client.get("source_journals")
    joined = pacing.get("source_sha256")
    root_name = pacing.get("source_capture_root")
    if not isinstance(pinned, dict) or not pinned or pinned != joined or not root_name:
        return None
    root = Path(root_name).resolve()
    if set(pinned) != {c.get("capture_id") for c in client.get("walk") or [] if isinstance(c, dict)}:
        return None
    for capture_id, expected in pinned.items():
        if not isinstance(expected, dict) or set(expected) != {"capture.json", "frames.jsonl"}:
            return None
        directory = (root / str(capture_id)).resolve()
        try:
            directory.relative_to(root)
        except ValueError:
            return None
        if journal_sha256(directory) != expected:
            return None
    return dict(sorted(pinned.items()))


def _verified_calibration(run: dict, client: dict) -> dict | None:
    """Bind the copied files the test Tower actually used, not their names alone."""
    expected = run.get("intrinsics_sha256")
    check = client.get("calibration_check")
    if not isinstance(expected, dict) or not expected or not isinstance(check, dict) or \
            check.get("verified") is not True or check.get("expected") != expected or \
            check.get("before_stream") != expected or check.get("after_stream") != expected or \
            set(expected) != set(run.get("intrinsics_copied") or []):
        return None
    root_name = run.get("data_root")
    if not root_name:
        return None
    root = Path(root_name) / "world_builder" / "intrinsics"
    try:
        paths = sorted(root.glob("*.json"))
        actual = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    except OSError:
        return None
    return dict(sorted(expected.items())) if actual == expected else None


def timing_env_key(run: dict) -> dict | None:
    """run.json `timing_env`, every variable it records (review F11 MED-2),
    or None when it lacks one of `TIMING_ENV_REQUIRED` or a value is neither
    a string nor None (unset)."""
    env = run.get("timing_env") if isinstance(run, dict) else None
    if not isinstance(env, dict) or any(name not in env for name in TIMING_ENV_REQUIRED):
        return None
    if any(value is not None and not isinstance(value, str) for value in env.values()):
        return None
    return dict(sorted(env.items()))


def interpreter_key(run: dict) -> dict | None:
    """The interpreter and venv the test Tower ran under (review F11 MED-2),
    from run.json:
      executable      `command[0]`: on Windows the venv's BASE interpreter
                      (`process_ownership.interpreter_executable`), e.g.
                      ...\\Python312\\python.exe, whose directory names its
                      major.minor version;
      venv_launcher   `effective_tower_env.__PYVENV_LAUNCHER__`: the venv's
                      python, which is what makes it the venv's (None when the
                      runner ran outside a venv);
      recorded        run.json `interpreter`, when a runner records one (the
                      807054d runner does not; see the F12 report's OPEN items).
    None when run.json records no command or no effective environment."""
    if not isinstance(run, dict):
        return None
    command = run.get("command")
    effective = run.get("effective_tower_env")
    if not isinstance(command, list) or not command or not isinstance(command[0], str) or not command[0] \
            or not isinstance(effective, dict):
        return None
    key = {"executable": command[0], "venv_launcher": effective.get("__PYVENV_LAUNCHER__")}
    if isinstance(run.get("interpreter"), dict):
        key["recorded"] = dict(sorted(run["interpreter"].items()))
    return key


def comparability_key(report: dict) -> dict:
    """What a run must share with every run `--compare` sets it against
    (review C24 HIGH-2). Two walks can each pass fidelity against their OWN
    journal; only runs of the same input, code, switches and harness measure
    the same thing. A field is None when the report cannot say -- a
    standalone-client run has no run.json, a render made before C22-F7 lacks
    the replay's shape -- and such a run is never comparable (fail closed).

      source_captures        the walk's source capture ids, in walk order;
      source_journal_sha256  sha256 of each source `capture.json` and
                             `frames.jsonl` the fidelity join read;
      source_jpegs_sha256    ordered digest of the JPEG bytes read at send time,
                             checked against preflight and the source snapshot;
      switches               run.json's switch set (--env-file / --set);
      code                   the code tree's .py fingerprint AND its path;
      harness                the sha1 of the streaming harness the run was
                             made with (`STREAMING_HARNESS_FILES`);
      replay                 speed, first_seconds, after_stop, the schedule's
                             shape, and the client's options when recorded;
      calibration            content hashes of the intrinsics copied into the
                             fresh root and checked during streaming;
      fidelity_family        the family of the run's OWN fidelity bar version
                             (the one in force when it started): v1 and v3
                             are one family, `FIDELITY_FAMILY`;
      timing_env             run.json's inherited timing variables
                             (`timing_env_key`: CUDA_VISIBLE_DEVICES, OMP/MKL/
                             OPENBLAS_NUM_THREADS, PYTORCH_CUDA_ALLOC_CONF);
      interpreter            the base interpreter and the venv the test Tower
                             ran under (`interpreter_key`).
    """
    run = report.get("run") if isinstance(report.get("run"), dict) else {}
    client = report.get("client") if isinstance(report.get("client"), dict) else {}
    harness = run.get("harness") or report.get("harness") or {}
    files = harness.get("files_sha1") if isinstance(harness, dict) and isinstance(harness.get("files_sha1"), dict) \
        else {}
    pacing = report.get("tower_side_pacing") if isinstance(report.get("tower_side_pacing"), dict) else {}
    code = run.get("code") if isinstance(run.get("code"), dict) else {}
    walk = [capture for capture in client.get("walk") or [] if isinstance(capture, dict)]
    schedule = client.get("schedule") if isinstance(client.get("schedule"), dict) else {}
    journals = _verified_source_journals(client, pacing)
    speed = client.get("speed")
    replay = None
    if isinstance(speed, (int, float)) and client.get("after_stop") and schedule.get("frames") is not None:
        replay = {"speed": speed, "first_seconds": client.get("first_seconds"), "after_stop": client["after_stop"],
                  "schedule": {k: schedule.get(k) for k in ("frames", "captures", "stop_at_s", "ends_with",
                                                            "reconnects")},
                  "options": client.get("options")}
    version = fidelity_version(run_started_at(report.get("run"), report.get("client"))[0])[0]
    return {
        "source_captures": [capture.get("capture_id") for capture in walk] or None,
        "source_journal_sha256": journals,
        "source_jpegs_sha256": _verified_source_jpegs(client, pacing),
        "switches": dict(sorted(run["switches"].items())) if isinstance(run.get("switches"), dict) else None,
        "code": ({"py_fingerprint": code["py_fingerprint"], "tower_dir": code["tower_dir"]}
                 if code.get("py_fingerprint") and code.get("tower_dir") else None),
        "harness": ({name: files[name] for name in STREAMING_HARNESS_FILES}
                    if all(files.get(name) for name in STREAMING_HARNESS_FILES) else None),
        "replay": replay,
        "calibration": _verified_calibration(run, client),
        "fidelity_family": FIDELITY_FAMILY.get(version),
        "timing_env": timing_env_key(run),
        "interpreter": interpreter_key(run),
    }


def harness_pin_of(identity) -> dict:
    """`world_live_replay.harness_pin`: the version, git HEAD, clean or not,
    and the streaming scripts' sha1 (manager 154 §2)."""
    from scripts.world_live_replay import harness_pin  # noqa: PLC0415

    return harness_pin(identity)


def rendered_by() -> dict:
    """Which harness version and report script rendered a report (a
    re-render may be newer than the run's own harness)."""
    from scripts.world_live_replay import HARNESS_VERSION  # noqa: PLC0415

    try:
        report_sha1 = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        report_sha1 = None
    return {"version": HARNESS_VERSION, "report_version": REPORT_VERSION, "report_sha1": report_sha1}


def _short(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)[:200]


def key_differences(key: dict, reference: dict | None, declared: dict | None = None) -> list:
    """Why a run is NOT comparable with the baseline's key, or [] (review C24
    HIGH-2). A candidate (`declared` given) may differ in the switches it
    declares, and only there: its switches must be exactly the baseline's
    with the declared ones laid over them. A baseline run may differ in
    nothing."""
    if reference is None:
        return ["no baseline run has a complete comparability key to compare against"]
    missing = [name for name in KEY_FIELDS if key.get(name) is None]
    if missing:
        return [f"its comparability key lacks {', '.join(missing)} (a run without run.json or a client record, "
                "or a render made before C22-F7: re-render it)"]
    out = []
    for name in KEY_FIELDS:
        mine, theirs = key[name], reference[name]
        if name == "switches":
            expected = {**theirs, **(declared or {})}
            if mine != expected:
                names = sorted(k for k in set(expected) | set(mine) if expected.get(k) != mine.get(k))
                detail = ", ".join(f"{k}: expected {expected.get(k)!r}, got {mine.get(k)!r}" for k in names)
                out.append("switches differ from the baseline's"
                           + (" with the declared --candidate-switches" if declared else "")
                           + f" ({detail})")
        elif mine != theirs:
            out.append(f"{name} differs (this run {_short(mine)}; the baseline {_short(theirs)})")
    return out


def _reference_key(runs: list) -> tuple:
    """(key, run dir) the baseline is compared on: the most common COMPLETE
    key among the baseline runs otherwise valid as proof (ties: the earliest
    given), else among every baseline run; (None, None) when none is
    complete."""
    for pool in ([run for run in runs if run["validity"]["counted"]], runs):
        complete = [(json.dumps(run["key"], sort_keys=True), run) for run in pool
                    if all(run["key"].get(name) is not None for name in KEY_FIELDS)]
        if complete:
            counts = Counter(text for text, _ in complete)
            best = max(counts.values())
            text, run = next(item for item in complete if counts[item[0]] == best)
            return json.loads(text), run["dir"]
    return None, None


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare_evidence(report: dict) -> dict:
    """What `--compare` re-derives from the run's own SEALED records, never
    from the render alone (review F11 MED-1, LOW-9).

    The render names its run directory (`inputs.run_dir`). From there:
      1. the seal: the run's records must still hash to the evidence seal its
         run-time render recorded (`<run dir>/report.json`), and to the one
         this render recorded;
      2. every record the render embeds (`run`, each `client` key it kept,
         `phone_view`, the Tower captures and Stop it read from the log)
         must equal the sealed file / a fresh scan of the sealed log;
      3. the STREAM IDENTITY: the Tower-minted capture id(s) in the test
         Tower's own log (equal to client.json's), the run directory, the
         run's data root and client.json's sha256. Two arguments sharing any
         of these are one streamed run, which counts once;
      4. the W0 time, Stop (the sealed log) -> the client's photos-ready
         receipt (sealed client.json), in seconds; the render's must match.
    Every failure is a reason the run is not counted (`problems`)."""
    problems: list = []
    evidence: dict = {"problems": problems, "run_dir": None, "identity": None, "seal": None,
                      "w0_seconds": None, "phone_photos_at": None, "stop_t": None, "stream_start_t": None,
                      "appearance": None, "run": None, "client": None}
    inputs = report.get("inputs") if isinstance(report.get("inputs"), dict) else {}
    if not inputs.get("run_dir"):
        problems.append("evidence: the render names no run directory (inputs.run_dir), so its records cannot be "
                        "checked against the run's own")
        return evidence
    run_dir = Path(inputs["run_dir"])
    evidence["run_dir"] = str(run_dir)
    run, client = _read_json(run_dir / "run.json"), _read_json(run_dir / "client.json")
    if not isinstance(run, dict) or not isinstance(client, dict):
        problems.append(f"evidence: the run directory {run_dir} has no readable run.json and client.json")
        return evidence
    evidence["run"], evidence["client"] = run, client
    tower_captures = client.get("tower_captures")
    tower_captures = tower_captures if isinstance(tower_captures, list) and tower_captures else None
    events = scan_log(run["err_log"]) if run.get("err_log") else []
    timeline = walk_timeline(events, tower_captures[0]) if tower_captures else {"found": False}
    evidence["stop_t"] = _t(timeline.get("stop"))
    evidence["stream_start_t"] = _t(timeline.get("stream_start"))
    # 1. The seal.
    now = evidence_seal(run_dir, run, timeline)
    evidence["seal"] = now["seal"]
    run_time_doc = _read_json(run_dir / "report.json")
    run_time = run_time_doc.get("evidence_seal") if isinstance(run_time_doc, dict) else None
    if not isinstance(run_time, dict) or run_time.get("at") != "run time":
        problems.append("evidence: the run directory holds no evidence seal recorded at run time (its run-time "
                        "render predates C22-F12, or is gone), so its records cannot be bound")
    elif run_time.get("seal") != now["seal"]:
        recorded = run_time.get("files") if isinstance(run_time.get("files"), dict) else {}
        changed = sorted(set(recorded) | set(now["files"]))
        changed = [name for name in changed if recorded.get(name) != now["files"].get(name)]
        problems.append("evidence: the run's records changed since run time (" + ", ".join(changed)
                        + " no longer hash to the run-time evidence seal)")
    rendered = report.get("evidence_seal")
    if not isinstance(rendered, dict) or rendered.get("seal") != now["seal"]:
        problems.append("evidence: this render's evidence seal is missing or is not the run's records' now "
                        "(re-render it)")
    # 2. What the render embeds is what was sealed.
    mismatched = []
    if "run" in report and report.get("run") != run:
        mismatched.append("run")
    embedded = report.get("client") if isinstance(report.get("client"), dict) else {}
    mismatched += [f"client.{key}" for key, value in embedded.items() if client.get(key) != value]
    if "phone_view" in report and report.get("phone_view") != (client.get("phone_view") or {}):
        mismatched.append("phone_view")
    walk = report.get("tower_walk") if isinstance(report.get("tower_walk"), dict) else {}
    if walk.get("captures") != timeline.get("captures"):
        mismatched.append("tower_walk.captures")
    if _t(walk.get("stop")) != evidence["stop_t"]:
        mismatched.append("tower_walk.stop")
    if mismatched:
        problems.append("evidence: the render's embedded records differ from the run's sealed records ("
                        + ", ".join(mismatched) + ")")
    # 3. The stream's identity.
    captures = timeline.get("captures") if timeline.get("found") else None
    if not captures or captures != tower_captures:
        problems.append("stream identity: the test Tower's own log and client.json do not name the same "
                        f"Tower-minted capture(s) ({captures} vs {tower_captures})")
    elif not run.get("data_root"):
        problems.append("stream identity: run.json names no data root")
    else:
        evidence["identity"] = ([("Tower capture", str(capture_id)) for capture_id in captures]
                                + [("run directory", _norm_path(run_dir)),
                                   ("data root", _norm_path(run["data_root"]))]
                                + ([("client.json sha256", now["files"]["client.json"])]
                                   if now["files"].get("client.json") else []))
    # 4. The W0 time, from the sealed records.
    appearance = session_appearance(run, timeline)
    evidence["appearance"] = appearance
    told = phone_photos(client.get("phone_view") or {}, evidence["stop_t"], appearance)
    evidence["phone_photos_at"] = told.get("phone_photos_at")
    evidence["w0_seconds"] = _seconds(evidence["stop_t"], told.get("phone_photos_at"))
    verdict = report.get("verdict") if isinstance(report.get("verdict"), dict) else {}
    rendered_s, rendered_min = verdict.get("stop_to_phone_photos_s"), verdict.get("stop_to_phone_photos_min")
    expected_min = _minutes(evidence["stop_t"], told.get("phone_photos_at"))
    same_s = (rendered_s is None and evidence["w0_seconds"] is None) or (
        _number(rendered_s) and evidence["w0_seconds"] is not None and abs(rendered_s - evidence["w0_seconds"]) <= 0.001)
    if not same_s or rendered_min != expected_min:
        problems.append(f"evidence: the render's W0 time ({rendered_min} min, {rendered_s} s) is not the one the "
                        f"run's sealed records give ({expected_min} min, {evidence['w0_seconds']} s)")
    # 5. Admission against the absolute bar (review F11 MED-3, MED-4,
    #    adversarial LOW-6), from the sealed records and the pinned sources.
    evidence["store_backing"] = photos_store_backing(told.get("phone_photos_at"), appearance)
    evidence["full_walk"] = full_walk_evidence(client, report.get("tower_side_pacing"))
    evidence["stop_agreement"] = client_stop_agreement(client, evidence["stop_t"])
    problems.extend(admission_problems(phone_photos_at=told.get("phone_photos_at"),
                                       backing=evidence["store_backing"], walk=evidence["full_walk"],
                                       stop=evidence["stop_agreement"]))
    return evidence


def _load_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    report = _read_json(run_dir / "report.json")
    if not isinstance(report, dict):
        raise SystemExit(f"{run_dir} has no report.json; render it first "
                         "(world_live_replay_report.py --run-dir <run> --out <new-empty-dir>)")
    keyframes = _read_json(run_dir / "keyframes.json") or {}
    fidelity = report.get("replay_fidelity") or {}
    # EACH RUN'S OWN BAR (managers 148 and 149): the version in force when THIS
    # run started, from its own recorded start (`run.json`, else `client.json`,
    # as the report kept them). A verdict rendered before the bar was versioned
    # (no `version`) was judged by v1's numbers, the only ones there were; one
    # rendered under the withdrawn v2 is never a run's own bar (re-render it).
    own_version, own_why = fidelity_version(run_started_at(report.get("run"), report.get("client"))[0])
    judged_by = fidelity.get("version") or ("v1" if fidelity.get("result") is not None else None)
    # A render made before C22-F7 has no `proof`: it is worked out from the
    # record the report kept (review C24 HIGH-3).
    proof = report.get("proof") if isinstance(report.get("proof"), dict) else proof_status(
        client=report.get("client") or {}, run=report.get("run"))
    return {"dir": str(run_dir), "label": report.get("label"), "report": report,
             "metrics": comparable_metrics(report), "sequence": keyframes.get("sequence"),
            "sha256": keyframes.get("sha256") or (report.get("keyframes") or {}).get("sha256"),
            "horizons": [s.get("keyframes") for s in (report.get("tower_walk") or {}).get("background_solves")
                         or []],
             "live_safety": (report.get("live_safety") or {}).get("result"),
             "environment": ((report.get("live_safety") or {}).get("environment") or {}).get("result"),
             "tower_windows": (report.get("tower_walk") or {}).get("windows"),
             "client_first_seconds": (report.get("client") or {}).get("first_seconds"),
             "scheduled_frames": ((report.get("client") or {}).get("schedule") or {}).get("frames"),
             "source_frames": (report.get("tower_side_pacing") or {}).get("source_frames"),
             "pacing_matched": (report.get("tower_side_pacing") or {}).get("matched"),
             "pacing_source_only": (report.get("tower_side_pacing") or {}).get("source_only"),
             "pacing_replay_only": (report.get("tower_side_pacing") or {}).get("replay_only"),
             "pacing_duplicate_seq": (report.get("tower_side_pacing") or {}).get("source_duplicate_seq"),
             "client_unanswered": (((report.get("client") or {}).get("stream") or {}).get("unanswered")),
             "client_frame_errors": (((report.get("client") or {}).get("stream") or {}).get("frame_errors")),
             "phone_basis": (report.get("verdict") or {}).get("basis"),
             "phone_photos_minutes": (report.get("verdict") or {}).get("stop_to_phone_photos_min"),
            # None: a report rendered before the bar existed.
            "fidelity": fidelity.get("result"),
            "fidelity_tower_side_only": bool(fidelity.get("tower_side_only")),
            # The bar version this run is judged by, and the one its render used.
            "fidelity_version": own_version, "fidelity_version_why": own_why,
            "fidelity_judged_by": judged_by,
            "proof": proof,
            # Recomputed from the report, never trusted from it (review C24 HIGH-2).
            "key": comparability_key(report),
            # Re-derived from the run's sealed records (review F11 MED-1, LOW-9).
            "evidence": compare_evidence(report),
            "rendered_by": report.get("rendered_by")}


def run_validity(run: dict) -> dict:
    """Is a run valid as proof? Only a replay-fidelity PASS is (manager 142:
    "every proof-set run must pass it, or it is discarded and re-run").

    A replay-fidelity FAIL, live-safety failure, incomplete frame path, or an
    Environment FAIL (review C22 round 3 L-f: a contended run widens the old
    path's noise), makes a run INVALID. So does
    NOT-PROOF (review C24 HIGH-3: made without the :8000 guard, or declared
    `--not-a-proof-run`). A fidelity verdict that is n/a, or absent (a report
    rendered before the bar), is NOT a pass either (review C22 round 4 M-1):
    the run is "not judged". Neither kind is COUNTED: not in the baseline's
    mean, min, max or spread, and not toward N >= 3. Both are shown and
    flagged, and a candidate of either kind is marked INVALID.
    (`compare_runs` adds the last reason, review C24 HIGH-2: a run that is
    not comparable with the baseline.)

    Each run is judged by ITS OWN fidelity bar version (managers 148 and 149):
    a verdict its render reached under another version -- the withdrawn v2
    included -- is not judged either (re-render it), whichever way it went.

    A fidelity FAIL on the Tower-side clauses alone, with the client on time,
    says so (review C24 HIGH-4): the stamp is post-reply, so it can be the
    code under test's own frame path rather than a bad replay."""
    invalid, unjudged = [], []
    own, judged_by = run.get("fidelity_version"), run.get("fidelity_judged_by")
    if run.get("fidelity") is not None and own and judged_by and judged_by != own:
        unjudged.append(f"replay fidelity {run.get('fidelity')} under bar {judged_by}, but this run's own bar is "
                        f"{own} ({run.get('fidelity_version_why')}): not judged, not counted (re-render it)")
    elif run.get("fidelity") == "FAIL":
        invalid.append("replay fidelity FAIL" + (
            " (Tower-side clauses only, the client on time: the post-reply stamp includes the code under test's "
            "own frame path -- inspect it before discarding the run)" if run.get("fidelity_tower_side_only")
            else ""))
    elif run.get("fidelity") != "PASS":
        unjudged.append("replay fidelity " + ("not in this render (re-render it)" if run.get("fidelity") is None
                                              else str(run.get("fidelity"))) + ": not judged, not counted")
    if run.get("environment") != "PASS":
        invalid.append(f"Environment (:8000) {run.get('environment') or 'not judged'}")
    if run.get("live_safety") != "PASS":
        invalid.append(f"live safety {run.get('live_safety') or 'not judged'}")
    metrics = run.get("metrics") or {}
    sent = metrics.get("frames_sent")
    received = metrics.get("frames_received")
    observed = metrics.get("frames_observed")
    # The safety table can be PASS with an n/a row when a counter is absent.
    # A partial world can finish quickly, so proof requires positive, equal
    # client, recorder, and builder counts even if the timing bar is met.
    if any(type(value) is not int or value <= 0 for value in (sent, received, observed)) or not (
        sent == received == observed
    ):
        invalid.append(
            f"incomplete frame path: sent={sent}, received={received}, builder_observed={observed}"
        )
    zero_keys = ("tx_seq_gap_total", "backpressure_drops", "frames_rejected", "frame_processing_errors")
    for key in zero_keys:
        value = metrics.get(key)
        if type(value) is not int or value != 0:
            invalid.append(f"Tower {key} must be observed zero; got {value}")
    windows = run.get("tower_windows")
    if not isinstance(windows, list) or not windows:
        invalid.append("Tower measurement windows missing")
    else:
        for index, window in enumerate(windows):
            if not isinstance(window, dict):
                invalid.append(f"Tower window {index} missing")
                continue
            for key in zero_keys:
                value = window.get(key)
                if type(value) is not int or value != 0:
                    invalid.append(f"Tower window {index} {key} must be observed zero; got {value}")
        window_received = [window.get("frames_received") for window in windows if isinstance(window, dict)]
        if len(window_received) != len(windows) or any(type(value) is not int or value < 0
                                                     for value in window_received) or sum(window_received) != received:
            invalid.append("Tower window frame counts do not sum to frames received")
    source = run.get("source_frames")
    scheduled = run.get("scheduled_frames")
    matched = run.get("pacing_matched")
    # A `first_seconds` cut, another speed, or fewer frames than the source
    # captures' own frames_written: `full_walk_evidence`, from the sealed
    # client.json (review F11 MED-4), among the evidence problems below.
    if any(type(value) is not int or value <= 0 for value in (source, scheduled, matched)) or not (
        source == scheduled == sent == matched
    ):
        invalid.append(f"incomplete source schedule: source={source}, scheduled={scheduled}, "
                       f"sent={sent}, matched={matched}")
    for key in ("pacing_source_only", "pacing_replay_only", "pacing_duplicate_seq", "client_unanswered"):
        value = run.get(key)
        if type(value) is not int or value != 0:
            invalid.append(f"{key} must be observed zero; got {value}")
    if run.get("client_frame_errors") != {}:
        invalid.append("client frame errors must be observed empty")
    photos = run.get("phone_photos_minutes")
    if run.get("phone_basis") != "phone" or type(photos) not in (int, float) or not math.isfinite(photos) or photos < 0:
        invalid.append("client phone-photo receipt is missing")
    proof = run.get("proof") or {}
    if proof.get("proof") is not True:
        invalid.append("NOT-PROOF: " + "; ".join(proof.get("not_proof_reasons") or ["declared"]))
    # The run's own sealed records (review F11 MED-1, LOW-9). A loaded run
    # always carries them; a hand-built one without them is not proof.
    evidence = run.get("evidence") if isinstance(run.get("evidence"), dict) else {
        "problems": ["evidence: the run's sealed records were not read"]}
    invalid.extend(evidence.get("problems") or [])
    # ONE report script judges the whole set: the verdict fields compare
    # reads from a render (fidelity, safety, environment, proof) are this
    # script's, or the render is re-rendered first (review F11 LOW-9).
    mine = rendered_by().get("report_sha1")
    theirs = (run.get("rendered_by") or {}).get("report_sha1") if isinstance(run.get("rendered_by"), dict) else None
    if theirs != mine:
        invalid.append(f"rendered by another report script (report sha1 {theirs}), not this one ({mine}): "
                       "re-render it with this script into a new --out")
    return {"invalid": invalid, "not_a_pass": unjudged, "counted": not invalid and not unjudged}


def _first_difference(a, b):
    if a is None or b is None:
        return None
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def parse_switches(items) -> dict:
    """`KEY=VALUE` items (`--candidate-switches`) as a dict; later wins."""
    switches = {}
    for item in items or ():
        if "=" not in str(item):
            raise SystemExit(f"--candidate-switches wants KEY=VALUE, got {item!r}")
        key, value = str(item).split("=", 1)
        switches[key.strip()] = value.strip()
    return switches


def mark_duplicates(baseline: list, candidate: list) -> list:
    """Every argument that is a streamed run already given (review F11
    MED-1): it shares a stream-identity component (`compare_evidence`) with
    an EARLIER argument, the baseline's before the candidates'. The first
    stands for the run; each later one is not counted, whatever it is -- the
    same directory again, a re-render, a copy -- and says which it repeats.
    A run whose identity cannot be established is already not counted."""
    seen: dict = {}
    duplicates = []
    for side, runs in (("baseline", baseline), ("candidate", candidate)):
        for run in runs:
            components = (run.get("evidence") or {}).get("identity")
            if not components:
                continue
            shared = [component for component in components if component in seen]
            if shared:
                first = seen[shared[0]]
                run["duplicate_of"] = first["dir"]
                what = "; ".join(f"{name} {value}" for name, value in shared)
                run["validity"]["invalid"].append(
                    f"duplicate of {first['side']} `{first['dir']}`: the same streamed run ({what}) counts once")
                run["validity"]["counted"] = False
                duplicates.append({"dir": run["dir"], "side": side, "of": first["dir"], "of_side": first["side"],
                                   "shared": [list(component) for component in shared]})
                for component in components:
                    seen.setdefault(component, first)
            else:
                for component in components:
                    seen[component] = {"dir": run["dir"], "side": side}
    return duplicates


def compare_runs(baseline_dirs, candidate_dirs=(), candidate_switches=None) -> dict:
    """Per-metric spread over the COUNTED baseline runs, and each candidate
    run against that spread; keyframe-sequence identity to the first counted
    baseline run. Counted means valid as full-walk proof (`run_validity`:
    complete source and frame path, known-zero safety counters, client photo
    receipt, fidelity/environment PASS and explicit proof) AND comparable with
    the baseline's key (review C24 HIGH-2): a candidate may differ from it
    only in `candidate_switches`, the switches under test."""
    baseline = [_load_run(d) for d in baseline_dirs]
    candidate = [_load_run(d) for d in candidate_dirs]
    for run in baseline + candidate:
        run["validity"] = run_validity(run)
    # One streamed run counts ONCE (review F11 MED-1), before anything is
    # counted or a reference chosen: a duplicate never becomes the reference.
    duplicates = mark_duplicates(baseline, candidate)
    declared = dict(candidate_switches or {})
    reference_key, reference_from = _reference_key([run for run in baseline if not run.get("duplicate_of")])
    not_comparable = []
    for side, runs in (("baseline", baseline), ("candidate", candidate)):
        for run in runs:
            if run.get("duplicate_of"):
                continue  # judged once, as the run it repeats
            differences = key_differences(run["key"], reference_key, declared if side == "candidate" else None)
            run["comparable"] = not differences
            if differences:
                run["validity"]["invalid"].append("not comparable: " + "; ".join(differences))
                run["validity"]["counted"] = False
                not_comparable.append({"dir": run["dir"], "side": side, "differences": differences})
    # A baseline run that is invalid OR not judged is shown and never counted
    # (manager 142; review C22 round 3 L-f, round 4 M-1): the range is the
    # VALID old runs' noise.
    in_range = [run["validity"]["counted"] for run in baseline]
    names = []
    for run in baseline + candidate:
        for name in run["metrics"]:
            if name not in names:
                names.append(name)
    metrics = []
    for name in names:
        base = [run["metrics"].get(name) for run in baseline]
        cand = [run["metrics"].get(name) for run in candidate]
        known = [v for v, counted in zip(base, in_range) if counted and isinstance(v, (int, float))]
        item = {"metric": name, "baseline": base, "baseline_in_range": in_range, "candidate": cand}
        if known:
            low, high = min(known), max(known)
            item.update({"mean": round(sum(known) / len(known), 4), "min": low, "max": high,
                         "spread": round(high - low, 4)})
            item["outside"] = [v for v in cand if isinstance(v, (int, float)) and not low <= v <= high]
            # A candidate with NO value where the baseline has one -- one
            # that never settled, never told the phone, lost a row -- is not
            # "within the baseline's range" (review C22 round 2, "Still
            # open" 4). It is flagged MISSING and counted as not passing.
            item["missing"] = [candidate[index]["dir"] for index, v in enumerate(cand)
                               if not isinstance(v, (int, float))]
        metrics.append(item)
    missing = {}
    for item in metrics:
        for directory in item.get("missing") or []:
            missing.setdefault(directory, []).append(item["metric"])
    counted = [run for run, keep in zip(baseline, in_range) if keep]
    valid_candidates = [run for run in candidate if run["validity"]["counted"]]
    # The identity reference is a counted run; only when none is does it fall
    # back to the first baseline run, and then it says so.
    reference = (counted or baseline or [None])[0]

    def identity(run):
        if reference is None:
            return None
        same = run["sha256"] is not None and run["sha256"] == reference["sha256"]
        return {"dir": run["dir"], "sha256": run["sha256"], "identical": same,
                "first_difference": None if same else _first_difference(run["sequence"], reference["sequence"]),
                "horizons": run["horizons"], "horizons_identical": run["horizons"] == reference["horizons"],
                "live_safety": run["live_safety"], "environment": run.get("environment"),
                "fidelity": run.get("fidelity"), "fidelity_version": run.get("fidelity_version"),
                "fidelity_judged_by": run.get("fidelity_judged_by"), "comparable": run.get("comparable"),
                "proof": (run.get("proof") or {}).get("proof"), "invalid": run["validity"]["invalid"],
                "not_a_pass": run["validity"]["not_a_pass"]}

    def flagged(runs, key):
        return [{"dir": run["dir"], "reasons": run["validity"][key]} for run in runs if run["validity"][key]]

    def not_counted(runs):
        return [{"dir": run["dir"], "reasons": run["validity"]["invalid"] + run["validity"]["not_a_pass"]}
                for run in runs if not run["validity"]["counted"]]

    return {
        # /4: each run judged by its own fidelity bar version (manager 148 §1).
        # /5: the versions are v1 and v3 (manager 149 §2); v2 is withdrawn.
        # /6: the comparability key, NOT-PROOF, and the W0 timing metric on
        #     phone_photos_at (review C24 HIGH-1..3).
        # /7: distinct streamed runs, the sealed evidence, the timing-env and
        #     interpreter key fields and the WALK6-GATE block (review F11).
        "compare": "c22-live-replay-compare/7",
        "generated_at": round(time.time(), 3),
        "rendered_by": rendered_by(),
        "baseline": [run["dir"] for run in baseline],
        "candidate": [run["dir"] for run in candidate],
        "fidelity_ruling": FIDELITY_RULING,
        "w0_timing_metric": W0_TIMING_METRIC,
        # Arguments that repeat a streamed run already given: never counted
        # (review F11 MED-1).
        "duplicates": duplicates,
        # Each argument's evidence seal, recomputed now: match a counted run's
        # against the line its run-time render printed (the RUN ledger).
        "evidence_seals": [{"dir": run["dir"], "side": side, "seal": (run.get("evidence") or {}).get("seal"),
                            "counted": run["validity"]["counted"]}
                           for side, runs in (("baseline", baseline), ("candidate", candidate)) for run in runs],
        # What every counted run shares (review C24 HIGH-2).
        "comparability": {"fields": list(KEY_FIELDS), "reference_key": reference_key,
                          "reference_from": reference_from, "candidate_switches": declared,
                          "not_comparable": not_comparable},
        # Shown, never counted in the range nor toward N >= 3: replay fidelity
        # FAIL, n/a or not rendered, Environment FAIL, NOT-PROOF, or not
        # comparable.
        "excluded_from_baseline": not_counted(baseline),
        # Not proof (discard and re-run): the same reasons. Judged against the
        # range only for the record.
        "invalid_candidates": not_counted(candidate),
        # Replay fidelity n/a or not rendered: not a pass for a proof set.
        "fidelity_not_judged": flagged(baseline + candidate, "not_a_pass"),
        "baseline_counted": len(counted),
        "candidates_counted": len(valid_candidates),
        "enough_runs": len(counted) >= 3 and (not candidate or len(valid_candidates) >= 3),
        # Per candidate run, every metric the baseline has and it lacks.
        # Not passing: a candidate is never judged on the metrics it is
        # missing.
        "missing": missing,
        "candidates_complete": not missing,
        "metrics": metrics,
        "keyframes": {"reference": None if reference is None else reference["dir"],
                      "reference_counted": reference is not None and reference["validity"]["counted"],
                      "baseline": [identity(run) for run in baseline],
                      "candidate": [identity(run) for run in candidate]},
    }


def _key_text(key: dict | None) -> list:
    """The comparability key, one markdown bullet per field."""
    if not key:
        return ["- (none: no baseline run has a complete key)"]
    journals = "; ".join(f"{cid[:8]}… frames.jsonl {str(d.get('frames.jsonl'))[:16]}, capture.json "
                         f"{str(d.get('capture.json'))[:16]}" for cid, d in (key.get("source_journal_sha256") or {}).items())
    code = key.get("code") or {}
    harness = key.get("harness") or {}
    return [
        f"- source captures: {key.get('source_captures')}; journals: {journals}",
        f"- code: py fingerprint `{code.get('py_fingerprint')}` at `{code.get('tower_dir')}`",
        "- streaming harness: " + ", ".join(f"{name} `{str(sha)[:12]}`" for name, sha in harness.items()),
        f"- switches ({len(key.get('switches') or {})}): "
        + ", ".join(f"{k}={v}" for k, v in (key.get("switches") or {}).items()),
        f"- replay: {_short(key.get('replay'))}",
        f"- calibration: {key.get('calibration')}; fidelity family: {key.get('fidelity_family')}",
        "- timing environment (run.json timing_env; None = unset): "
        + ", ".join(f"{k}={v!r}" for k, v in (key.get("timing_env") or {}).items()),
        f"- interpreter: {_short(key.get('interpreter'))}",
    ]


def render_compare(result: dict) -> str:
    lines = ["# C22 live replay: compare", ""]
    lines.append(f"Baseline ({len(result['baseline'])} runs): " + ", ".join(f"`{d}`" for d in result["baseline"]))
    if result["candidate"]:
        lines.append(f"Candidate ({len(result['candidate'])} runs): "
                     + ", ".join(f"`{d}`" for d in result["candidate"]))
    lines.append("")
    lines.append(f"**The W0 timing metric is `{result.get('w0_timing_metric', W0_TIMING_METRIC)}`**: Stop to the "
                 "replay client's receipt of the room's photos (`phone_photos_at`, review C24 HIGH-1). "
                 "`stop_to_store_photos_min`, the store's `updated_at`, is INFO.")
    if not result["enough_runs"]:
        lines.append("")
        counted_text = f"{result.get('baseline_counted')} of {len(result['baseline'])} baseline run(s) valid"
        if result["candidate"]:
            counted_text += f", {result.get('candidates_counted')} of {len(result['candidate'])} candidate(s)"
        lines.append("**Fewer than 3 valid runs on a side: this is not a noise estimate (C19 F8 asks for N >= 3; "
                     "each counted run also needs full-walk, safety, client-photo, environment and proof evidence "
                     f"plus replay-fidelity PASS): {counted_text}.**")
    duplicates = result.get("duplicates") or []
    if duplicates:
        lines.append("")
        lines.append(f"**DUPLICATE RUNS: {len(duplicates)} argument(s) repeat a streamed run already given, and are "
                     "not counted (review F11 MED-1): one streamed run counts once, whatever its directory.**")
        for item in duplicates:
            lines.append(f"- {item['side']} `{item['dir']}` repeats {item['of_side']} `{item['of']}` (shares "
                         + "; ".join(f"{name} {value}" for name, value in item["shared"]) + ")")
    comparability = result.get("comparability") or {}
    if comparability:
        lines.append("")
        lines.append("## Comparability (review C24 HIGH-2)")
        lines.append("")
        lines.append("Only runs with the SAME comparability key are counted: " + ", ".join(
            f"`{name}`" for name in comparability.get("fields") or KEY_FIELDS)
            + ". A candidate may differ from the baseline only in the declared `--candidate-switches`"
            + (": " + ", ".join(f"`{k}={v}`" for k, v in comparability["candidate_switches"].items())
               if comparability.get("candidate_switches") else " (none declared: its switches must equal the "
                                                                "baseline's)")
            + f". The baseline's key (from `{comparability.get('reference_from')}`):")
        lines.extend(_key_text(comparability.get("reference_key")))
        not_comparable = comparability.get("not_comparable") or []
        if not_comparable:
            lines.append("")
            lines.append(f"**NOT COMPARABLE: {len(not_comparable)} run(s), excluded.**")
            for item in not_comparable:
                lines.append(f"- {item['side']} `{item['dir']}`: {'; '.join(item['differences'])}")
    excluded = result.get("excluded_from_baseline") or []
    if excluded:
        lines.append("")
        lines.append(f"**EXCLUDED FROM THE BASELINE RANGE: {len(excluded)} run(s).** A run that fails replay "
                     f"fidelity ({result.get('fidelity_ruling')}) or the Environment (:8000) verdict, that is "
                     "NOT-PROOF or not comparable, or whose replay fidelity is n/a or not in its render (not "
                     "judged: n/a is NOT a pass), is not valid as proof: it is shown below and never counted in the "
                     f"mean, min, max or spread, nor toward N >= 3 ({result.get('baseline_counted')} baseline "
                     "run(s) counted). Each run is judged by its own fidelity bar version: the one in force when it "
                     "started.")
        for item in excluded:
            lines.append(f"- `{item['dir']}`: {', '.join(item['reasons'])}")
    invalid = result.get("invalid_candidates") or []
    if invalid:
        lines.append("")
        lines.append(f"**INVALID CANDIDATE(S): {len(invalid)}.** Not proof (discard and re-run): replay fidelity "
                     "FAIL, n/a or not in its render, an Environment FAIL, NOT-PROOF, or not comparable; its values "
                     "are marked INVALID below.")
        for item in invalid:
            lines.append(f"- `{item['dir']}`: {', '.join(item['reasons'])}")
    unjudged = result.get("fidelity_not_judged") or []
    if unjudged:
        lines.append("")
        lines.append(f"**Replay fidelity not judged for {len(unjudged)} run(s): {FIDELITY_NA_NOTE}**")
        for item in unjudged:
            lines.append(f"- `{item['dir']}`: {', '.join(item['reasons'])}")
    missing = result.get("missing") or {}
    if missing:
        lines.append("")
        lines.append(f"**MISSING: {sum(len(v) for v in missing.values())} candidate value(s) absent where the "
                     "baseline has one. Not passing: a candidate is not judged on what it lacks.**")
        for directory, names in missing.items():
            lines.append(f"- `{directory}`: {', '.join(names)}")
    lines.append("")
    lines.append("## Keyframe selection sequence, (source_seq, segment_index), against the first counted baseline "
                 "run")
    lines.append("")
    lines.append("Which frames were selected, in which segments -- not their image content, and not "
                 "capture-qualified (review C24 MED-8).")
    lines.append("")
    reference = result["keyframes"].get("reference")
    if reference is not None:
        lines.append(f"Reference: `{reference}`" + (
            "" if result["keyframes"].get("reference_counted") else
            " -- **NO baseline run is valid as proof: this reference is not one either.**"))
        lines.append("")
    lines.append("| Run | sha256 | Identical | First differing index | Solve horizons identical | Live safety "
                 "| Environment (:8000) | Replay fidelity | Fidelity bar (own / render) | Valid as proof |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for side in ("baseline", "candidate"):
        for item in result["keyframes"][side]:
            if item is None:
                continue
            reasons = (item.get("invalid") or []) + (item.get("not_a_pass") or [])
            valid = ("**NO**: " + ", ".join(reasons) + (" (excluded from the range)" if side == "baseline"
                                                        else " (INVALID)")
                     if reasons else "yes")
            lines.append(f"| {side}: `{item['dir']}` | {str(item['sha256'])[:16]} | {item['identical']} | "
                         f"{'' if item['first_difference'] is None else item['first_difference']} | "
                         f"{item['horizons_identical']} | {item['live_safety']} | {item.get('environment')} | "
                         f"{item.get('fidelity')} | {item.get('fidelity_version')} / "
                         f"{item.get('fidelity_judged_by') or '—'} | {valid} |")
    seals = result.get("evidence_seals") or []
    if seals:
        lines.append("")
        lines.append("## Evidence seals (review F11 LOW-9)")
        lines.append("")
        lines.append("Each run's records, hashed now. A counted run's seal equals the one its run-time render "
                     "recorded; check it against the `[report] evidence seal` line the runner printed into the quiet "
                     "launcher's log, copied to the RUN ledger.")
        lines.append("")
        for item in seals:
            lines.append(f"- {item['side']} `{item['dir']}`: `{item['seal']}`"
                         + ("" if item.get("counted") else " (not counted)"))
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("A candidate value outside the baseline's [min, max] is flagged; a candidate with no value "
                 "where the baseline has one is flagged MISSING. The spread is max - min over the baseline "
                 "runs COUNTED (full-walk and safety evidence, replay fidelity and environment PASS, "
                 "client-photo proof, comparable): the old "
                 "path's own noise. A baseline run excluded above is not in the mean, min, max or spread; an "
                 "INVALID candidate's value is marked. `keyframe_accept_lag_s` is over ACCEPTED keyframes only; "
                 "`recorder_stamp_*` are the recorder's post-reply stamps, not arrivals.")
    lines.append("")
    lines.append("| Metric | Baseline mean | min | max | spread | Excluded baseline value(s) | Candidate | Flag |")
    lines.append("|---|---|---|---|---|---|---|---|")
    invalid_dirs = {item["dir"] for item in invalid}
    w0_metric = result.get("w0_timing_metric", W0_TIMING_METRIC)
    for item in result["metrics"]:
        flags = []
        if item.get("outside"):
            flags.append("**outside** " + str(item["outside"]))
        if item.get("missing"):
            flags.append(f"**MISSING** in {len(item['missing'])}")
        flag = "; ".join(flags)
        counted = item.get("baseline_in_range") or [True] * len(item["baseline"])
        dropped = ", ".join(str(v) for v, keep in zip(item["baseline"], counted) if not keep)
        values = ", ".join(str(v) + (" (INVALID)" if directory in invalid_dirs else "")
                           for v, directory in zip(item["candidate"], result["candidate"]))
        name = f"{item['metric']} **(W0 timing)**" if item["metric"] == w0_metric else item["metric"]
        lines.append(f"| {name} | {item.get('mean', '')} | {item.get('min', '')} | "
                     f"{item.get('max', '')} | {item.get('spread', '')} | {dropped} | {values} | {flag} |")
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
        defaults["data_root"] = Path(run["data_root"])
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
    parser.add_argument("--data-root", type=Path, default=None,
                        help="The test Tower's data root, READ: its re-recorded captures "
                             "(<data-root>/captures/<id>/frames.jsonl) for the Tower-side pacing row. "
                             "Default: the one run.json names. Also the world root (<data-root>/world_builder) "
                             "unless --world-root is given.")
    parser.add_argument("--capture-root", type=Path, default=None,
                        help="Where the SOURCE captures are, READ (<root>/<id>/capture.json + frames.jsonl): "
                             "the recorded pace the pacing row and the replay-fidelity verdict join to. "
                             "Default: the one the run's client.json names, else the replay's default "
                             "capture root, which is the live store AS IT IS AT RENDER TIME. Give a snapshot "
                             "directory to pin it: proof-set re-renders pass "
                             "RUN\\experiments\\C22-REPLAY\\source-captures. The report records which root "
                             "it read, where that came from, and the journal's sha256.")
    parser.add_argument("--capture-id", default=None,
                        help="The walk's first capture (default: client.json's, else the log's first).")
    parser.add_argument("--label", default=None)
    parser.add_argument("--compare", type=Path, nargs="+", default=None, metavar="RUN",
                        help="Compare mode: the BASELINE run dirs (each with report.json), N >= 3.")
    parser.add_argument("--candidate", type=Path, nargs="+", default=(), metavar="RUN",
                        help="With --compare: the candidate run dirs, judged against the baseline's spread.")
    parser.add_argument("--candidate-switches", nargs="+", default=(), metavar="KEY=VALUE",
                        help="With --candidate: the switches under test. A candidate is comparable only when "
                             "its switch set is the baseline's with exactly these laid over it, and its every "
                             "other key field equals the baseline's (review C24 HIGH-2).")
    args = parser.parse_args(argv)
    out = Path(args.out)
    # Nothing is written inside the live store (review C24 MED-6).
    from scripts.world_live_replay import refuse_inside_live_store, refuse_non_empty_out  # noqa: PLC0415

    refuse_inside_live_store(out, "--out")
    refuse_non_empty_out(out)
    if args.compare:
        result = compare_runs(args.compare, args.candidate, parse_switches(args.candidate_switches))
        out.mkdir(parents=True, exist_ok=True)
        (out / "compare.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        (out / "COMPARE.md").write_text(render_compare(result), encoding="utf-8")
        print(json.dumps({"compare": str(out / "compare.json"), "enough_runs": result["enough_runs"],
                          "not_comparable": result["comparability"]["not_comparable"],
                          "candidates_complete": result["candidates_complete"],
                          "excluded_from_baseline": result["excluded_from_baseline"],
                          "invalid_candidates": result["invalid_candidates"]}, indent=2))
        return 0
    run_dir = Path(args.run_dir) if args.run_dir is not None else out
    defaults = _run_defaults(run_dir)
    tower_log = args.tower_log or defaults.get("tower_log")
    if tower_log is None:
        parser.error("--tower-log is required unless --run-dir holds a run.json naming it")
    client = _read_json(run_dir / "client.json") or {}
    samples = run_dir / "samples.csv"
    data_root = args.data_root or defaults.get("data_root")
    world_root = args.world_root or (Path(args.data_root) / "world_builder" if args.data_root is not None
                                     else defaults.get("world_root"))
    report = build_report(tower_log=tower_log, tower_out_log=args.tower_out_log or defaults.get("tower_out_log"),
                          world_root=world_root, capture_id=args.capture_id,
                          client=client, samples=samples if samples.exists() else None, label=args.label,
                          run_dir=run_dir, data_root=data_root, capture_root=args.capture_root,
                          run=defaults.get("run"))
    if defaults.get("run"):
        report["run"] = defaults["run"]
    json_path, md_path = write_report(out, report)
    print(json.dumps({"report": str(json_path), "markdown": str(md_path), "verdict": report["verdict"],
                      "live_safety": report["live_safety"]["result"],
                      "environment": (report["live_safety"].get("environment") or {}).get("result"),
                      "replay_fidelity": report["replay_fidelity"]["result"],
                      "proof": report["proof"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
