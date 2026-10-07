#!/usr/bin/env python
"""Replay a stored capture into a running TEST Tower in real time, and time
everything from the first frame to the settled world (C22).

WHY THIS EXISTS

The owner's bar is Stop -> room with photos in at most ten minutes, a hard
maximum, with the world finished DURING the walk. Until now the only way to
measure that was to walk. `world_replay.py` rebuilds a walk offline and
deterministically, which is right for geometry and wrong for timing: it has
no Tower, no /ws, no live builder racing the frames, no background solves
competing with them, no Stop and no finisher chore. This script puts the
recorded walk back through the phone's own door instead:

  * the SAME /ws ingest (`tower/routes/ws.py`): `stream_start`, one `frame`
    message per recorded frame carrying the recorded JPEG bytes and the
    recorded `seq` / `tx_seq` / width / height, and `stream_stop`;
  * at the RECORDED pace: each frame leaves at its recorded offset from its
    capture's start (`frames.jsonl` `received_at`), divided by `--speed`
    (default 1.0), against an absolute clock, so lateness never accumulates.
    WHAT THAT CLOCK IS (review C24 HIGH-4): `received_at` is NOT socket
    arrival. The Tower's capture recorder stamps it (`CaptureRecorder.
    write_frame`, `time.time()`) when `ws.py` hands it the frame, which is
    AFTER the frame was JSON-parsed, base64- and header-decoded, run through
    the CV module and answered with `frame_result` -- and the socket is read
    one message at a time, so also after the previous frame's fsync'd write.
    The recorded pace is therefore the live Tower's post-reply recording
    pace, an approximation of when the phone's frames arrived;
  * with the World Builder cartridge started the way the phone starts it
    (`POST /cartridges/world_builder/session/start` before `stream_start`),
    subscribed to its status channel the way the phone is, and fetching the
    geometry manifest and the changed segments whenever a push moves
    `geometry.revision`, the way the phone's `WorldBuilderClient` does;
  * across a multi-capture walk as the phone produced it. A capture that
    ended by `disconnect` is ended by closing the socket, and the next one
    begins on a new socket at its recorded offset. That offset is inside the
    90 s resume grace, so the Tower keeps one session. On the new socket the
    phone re-sends its handshake AND `POST .../session/start`
    (`WorldBuilderSessionController.towerReachabilityChanged`), and so does
    this: without the POST the Tower's deferred stop ends World Builder
    105 s after the disconnect (`ws.py` `_arm_world_builder_follow_up`).

Then it follows the Tower from Stop until it settles: finalization, room
surface, room appearance, worker exit, and the finisher chore's areas. It
samples CPU and GPU every few seconds, keeps a copy of `solution.json` each
time it changes (draw 0's early publish is overwritten by the consensus),
watches the session's surface `status.json` (the store keeps only the last
one), and writes a report in the shape of `PROFILE-walk5.md`
(`world_live_replay_report.py`).

SAFETY
  * It never talks to :8000 except to read `/health`, and it refuses a
    target port below 8031.
  * THE :8000 GUARD FAILS CLOSED. Before streaming, and every few seconds
    while it runs, it reads :8000's `/health` into one of five states
    (`live_tower_state`): `recording`, `busy` (a world is being finished:
    capture workers alive or the finisher chore running), `idle`, `down`
    (the connection was refused: nothing listens), `unknown` (a timeout, a
    5xx, anything else). It does not start on `recording`, `busy` or
    `unknown`. Once running it aborts on `recording` or on two `unknown`
    answers in a row, and records when :8000 became `busy`.
  * On abort the runner's `on_abort` kills the test Tower FIRST, then the
    client tears down.
  * An incomplete `/health` answer -- `capture`, `capture_workers` or
    `background_chore` absent or unreadable -- is `unknown`, never `idle`
    (review C24 MED-5).
  * The guard is armed as soon as :8000 has been read at the start, before
    anything else the client does (review C24 HIGH-3). `--no-live-guard`
    is refused unless `--not-a-proof-run` is given too, and such a run is
    marked NOT-PROOF in its record and its report.
  * It refuses a target that is recording or has capture workers, and,
    given the test Tower's pid, a target whose listener is another process.
  * It reads the stored capture and writes nothing beside it; its `--out`
    must be outside the live store (review C24 MED-6).

WHAT IT CANNOT SEE is listed in RUN\\experiments\\C22-REPLAY\\README.md. In
short: the phone's own experience (rendering, battery, the radio link), the
phone's SEND WINDOW and its RECONNECT WINDOW (review C24 MED-7: the replay
awaits every send, so it never drops a frame, and it replays only the frames
the original Tower recorded; the phone's client has a bounded send window,
drops frames when it is full and replaces a stalled socket -- none of which
a replay against a changed Tower can exercise), and a phone's reconnect
OVERLAP: a phone's dropped socket is noticed 20-40 s late, so its new
socket's `stream_start` supersedes the old capture while the old connection
still counts; the replay closes cleanly and reconnects, so the Tower always
takes the last-client path. It is a Tower benchmark: send-window loss and a
reconnected walk need a phone trace and a physical walk.

Usually driven by `world_live_replay_run.py`, which starts a test Tower from
a named code tree, runs this, stops the Tower and writes the report.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from scripts.world_live_replay_capture import ReplayCapture  # noqa: E402

TOWER_ROOT = Path(__file__).resolve().parents[1]
# The stored captures live in the canonical checkout's store; a worktree has
# no `data` of its own. Read only, always.
CANONICAL_CAPTURE_ROOT = Path(r"C:\Users\tvllo\Projects\Glasses\tower\data\captures")
DEFAULT_CAPTURE_ROOT = (CANONICAL_CAPTURE_ROOT if CANONICAL_CAPTURE_ROOT.is_dir()
                        else TOWER_ROOT / "data" / "captures")
# Tristan's live store. Nothing any of the three scripts writes may land
# inside it (the runner's rule, now every writing entrypoint's: review C24
# MED-6).
LIVE_DATA = CANONICAL_CAPTURE_ROOT.parent

# Tristan's own Tower. Read-only (/health), and only to find out whether a
# real walk is being recorded.
LIVE_TOWER_PORT = 8000
LIVE_HEALTH_URL = f"http://127.0.0.1:{LIVE_TOWER_PORT}/health"
# A test Tower lives at or above this port (the C22 brief), so a typo can
# never point a replay at the live Tower or at another lane's.
MIN_TEST_PORT = 8031

# The live guard (review C22 M1/M2). A slow answer is not "down": a Tower
# that is recording or finishing a world while the test Tower loads every
# core answers late, and that is exactly the case the guard exists for.
LIVE_GUARD_TIMEOUT_S = 5.0
# States in which a replay does not start.
LIVE_REFUSE_AT_START = frozenset({"recording", "busy", "unknown"})
# Consecutive `unknown` answers that abort a running replay.
LIVE_UNKNOWN_ABORT_AFTER = 2

WORLD_BUILDER = "world_builder"
WORLD_BUILDER_RESULT_TYPE = "status"

EXIT_SETTLED = 0
EXIT_ERROR = 1
EXIT_NOT_SETTLED = 2
EXIT_ABORTED = 3

# The harness itself, pinned in every run record (review C22 L5).
HARNESS_FILES = ("world_live_replay.py", "world_live_replay_report.py", "world_live_replay_run.py",
                 "world_live_timeline.py", "world_live_replay_capture.py")
# The scripts that STREAM or record a run. Their sha1 is the pin `--compare` keys
# on (review C24 HIGH-2); the report script only reads a finished run.
STREAMING_FILES = ("world_live_replay.py", "world_live_replay_run.py", "world_live_replay_capture.py")
# The harness's own version, recorded beside the pin in run.json, client.json
# and every report, so which harness made a run is self-evident (manager 154
# §2: proof sets run only on a pin that passed the C24x2 re-verify). Change it
# with every change to the harness.
HARNESS_VERSION = "c22-harness/F15 (optional status and geometry receipts)"

# Leaf keys copied out of each World Builder status push to show what the
# phone was being told, and when. Generic on purpose: the payload is large
# and changes between contract revisions, and a replay must read an old
# Tower's pushes as well as a new one's.
PHONE_STATE_KEYS = frozenset({
    "model_state", "state", "final_solve", "build_in_progress", "representation", "stage",
})


# -- the walk on disk --------------------------------------------------------


@dataclass(frozen=True)
class FrameRecord:
    """One row of `frames.jsonl`: what the Tower received and when."""

    wire_seq: int
    source_seq: int
    tx_seq: int | None
    received_at: float
    relpath: str
    width: int
    height: int
    byte_count: int | None = None


@dataclass
class CaptureRecord:
    capture_id: str
    directory: Path
    started_at: float
    ended_at: float | None
    end_reason: str | None
    continues: str | None
    frames: list = field(default_factory=list)
    source_sha256: dict = field(default_factory=dict)

    @property
    def recorded_seconds(self) -> float:
        end = self.ended_at if self.ended_at is not None else (
            self.frames[-1].received_at if self.frames else self.started_at)
        return max(0.0, end - self.started_at)


def _frame_from_row(row: dict) -> FrameRecord:
    wire = int(row["wire_seq"])
    source = row.get("source_seq")
    tx = row.get("tx_seq")
    return FrameRecord(
        wire_seq=wire,
        source_seq=wire if source is None else int(source),
        tx_seq=None if tx is None else int(tx),
        received_at=float(row["received_at"]),
        relpath=str(row["relpath"]),
        width=int(row["width"]),
        height=int(row["height"]),
        byte_count=None if row.get("byte_count") is None else int(row["byte_count"]),
    )


def read_capture(directory: Path) -> CaptureRecord:
    """A capture's manifest and journal. Reads only; never writes.

    A torn LAST line (a Tower killed mid-write) is dropped; a bad line
    anywhere else is an error, because a replay that silently skips frames
    measures a different walk.
    """
    directory = Path(directory)
    manifest_bytes = (directory / "capture.json").read_bytes()
    journal_bytes = (directory / "frames.jsonl").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    lines = journal_bytes.decode("utf-8").splitlines()
    frames = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            frames.append(_frame_from_row(json.loads(line)))
        except (ValueError, KeyError, TypeError) as exc:
            if number == len(lines):
                break
            raise SystemExit(f"{directory / 'frames.jsonl'} line {number} is unreadable: {exc}")
    started = manifest.get("started_at")
    if started is None:
        started = frames[0].received_at if frames else 0.0
    return CaptureRecord(
        capture_id=str(manifest.get("capture_id") or directory.name),
        directory=directory,
        started_at=float(started),
        ended_at=None if manifest.get("ended_at") is None else float(manifest["ended_at"]),
        end_reason=manifest.get("end_reason"),
        continues=manifest.get("continues_capture"),
        frames=frames,
        source_sha256={"capture.json": hashlib.sha256(manifest_bytes).hexdigest(),
                       "frames.jsonl": hashlib.sha256(journal_bytes).hexdigest()},
    )


def _successors(capture_root: Path) -> dict:
    """predecessor id -> successor id, from every manifest under the root."""
    found = {}
    for manifest in capture_root.glob("*/capture.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        before = data.get("continues_capture")
        if before:
            found[str(before)] = str(data.get("capture_id") or manifest.parent.name)
    return found


def load_walk(capture_root: Path, capture_ids, *, follow_chain: bool = True) -> list:
    """The captures of one walk, in time order.

    Several ids are taken as given, in the order given. One id is followed
    both ways through `continues_capture` when `follow_chain`: a walk the
    phone reconnected during is several captures, and replaying one of them
    measures part of a walk.
    """
    capture_root = Path(capture_root)
    ids = list(capture_ids)
    if not ids:
        raise SystemExit("no capture id given")
    if len(ids) > 1 or not follow_chain:
        return [read_capture(capture_root / cid) for cid in ids]
    first = read_capture(capture_root / ids[0])
    walk = [first]
    seen = {first.capture_id}
    while walk[0].continues and walk[0].continues not in seen:
        before = read_capture(capture_root / walk[0].continues)
        seen.add(before.capture_id)
        walk.insert(0, before)
    successors = _successors(capture_root)
    while walk[-1].capture_id in successors and successors[walk[-1].capture_id] not in seen:
        after = read_capture(capture_root / successors[walk[-1].capture_id])
        seen.add(after.capture_id)
        walk.append(after)
    return walk


# -- the schedule ------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One thing the phone did, `at` seconds after the walk began (scaled)."""

    at: float
    kind: str  # connect | stream_start | frame | stream_stop | disconnect
    capture: int
    frame: FrameRecord | None = None


def build_schedule(walk, *, speed: float = 1.0, first_seconds: float | None = None,
                   end_with_stop: bool = False, reconnect_lead: float = 0.5) -> list:
    """Every step of the walk, at its recorded offset divided by `speed`.

    Offsets are from the FIRST capture's `started_at` (the Tower's receipt of
    `stream_start`), so the recorded gap before the first frame and every
    recorded stall are reproduced, not smoothed. Within a capture the offsets
    are forced non-decreasing: the journal is receipt order, and a replay
    must never send frames out of it.

    A capture that ended by `stop` ends with `stream_stop`; one that ended by
    `disconnect` ends with the socket closing, and the next capture begins on
    a new socket `reconnect_lead` seconds before its recorded `stream_start`.
    `end_with_stop` turns a walk whose LAST capture ended by disconnect into
    one that ends with Stop.

    `first_seconds` keeps only the first part of the walk (walk time, before
    scaling) and ends it there with Stop, as if the wearer pressed Stop then.
    """
    if speed <= 0:
        raise ValueError("speed must be positive")
    if not walk:
        return []
    t0 = walk[0].started_at
    steps: list = []
    previous_end = 0.0
    for index, capture in enumerate(walk):
        start = max(capture.started_at - t0, previous_end)
        if index > 0:
            steps.append(Step(max(previous_end, start - reconnect_lead), "connect", index))
        steps.append(Step(start, "stream_start", index))
        last = start
        for frame in capture.frames:
            at = max(frame.received_at - t0, last)
            steps.append(Step(at, "frame", index, frame))
            last = at
        ended = capture.ended_at if capture.ended_at is not None else last
        end = max(ended - t0, last)
        is_last = index == len(walk) - 1
        polite = capture.end_reason == "stop" or (is_last and end_with_stop)
        steps.append(Step(end, "stream_stop" if polite else "disconnect", index))
        previous_end = end

    if first_seconds is not None and steps and first_seconds < steps[-1].at:
        kept = [step for step in steps if step.at <= first_seconds]
        while kept and kept[-1].kind == "connect":
            kept.pop()  # a socket opened for a capture that never starts
        if kept and kept[-1].kind not in ("stream_stop", "disconnect"):
            # A stream is open at the cut: the wearer pressed Stop there.
            kept.append(Step(float(first_seconds), "stream_stop", kept[-1].capture))
        steps = kept

    return [Step(step.at / speed, step.kind, step.capture, step.frame) for step in steps]


def schedule_summary(schedule: list) -> dict:
    frames = [step for step in schedule if step.kind == "frame"]
    stops = [step.at for step in schedule if step.kind in ("stream_stop", "disconnect")]
    return {
        "steps": len(schedule),
        "frames": len(frames),
        "captures": len({step.capture for step in schedule}),
        "first_frame_at_s": round(frames[0].at, 3) if frames else None,
        "stop_at_s": round(stops[-1], 3) if stops else None,
        "ends_with": schedule[-1].kind if schedule else None,
        "reconnects": sum(1 for step in schedule if step.kind == "connect"),
    }


# -- the wire ----------------------------------------------------------------


def frame_message(frame: FrameRecord, jpeg: bytes) -> dict:
    """The `frame` message the phone sent for this recorded frame.

    The phone sends `seq`, `tx_seq`, `width`, `height`, `format`, `data`
    (`TowerClient.swift`) and no `source_seq`, which the Tower then takes to
    be `seq` (`tower/frames.py`). The recorded `source_seq` is sent only when
    it differs, so a phone capture is replayed byte-for-byte in shape.
    """
    message = {
        "type": "frame",
        "seq": frame.wire_seq,
        "width": frame.width,
        "height": frame.height,
        "format": "jpeg",
        "data": base64.b64encode(jpeg).decode("ascii"),
    }
    if frame.tx_seq is not None:
        message["tx_seq"] = frame.tx_seq
    if frame.source_seq != frame.wire_seq:
        message["source_seq"] = frame.source_seq
    return message


def encode(message: dict) -> str:
    return json.dumps(message, separators=(",", ":"))


def check_target_port(port: int) -> None:
    if port == LIVE_TOWER_PORT:
        raise SystemExit(f"refused: port {LIVE_TOWER_PORT} is Tristan's live Tower")
    if port < MIN_TEST_PORT:
        raise SystemExit(f"refused: a test Tower runs on port {MIN_TEST_PORT} or higher, not {port}")


# -- pacing ------------------------------------------------------------------


class Pacer:
    """Waits for absolute offsets from one origin, and records the lateness.

    Absolute, so one late step never delays the next: the recorded walk's
    pace is kept even when a single send stalls. The sleep is re-checked
    after waking because the event loop may wake early (asyncio treats a
    timer within one clock tick as due) or late (the Windows timer tick).
    """

    def __init__(self, clock=time.perf_counter, sleep=None):
        self._clock = clock
        self._sleep = sleep or precise_sleep
        self.origin: float | None = None

    def start(self) -> None:
        self.origin = self._clock()

    def elapsed(self) -> float:
        return self._clock() - self.origin

    async def wait_until(self, at: float, cancel: asyncio.Event | None = None) -> float:
        """Sleep until `at` seconds after the origin; return how late we are.

        Returns early (with a negative lateness) once `cancel` is set, so an
        abort is obeyed within half a second even across a reconnect gap.
        """
        while True:
            remaining = at - self.elapsed()
            if remaining <= 0:
                return -remaining
            if cancel is not None and cancel.is_set():
                return -remaining
            await self._sleep(min(remaining, 0.5))


async def precise_sleep(seconds: float) -> None:
    """A sleep that wakes on time on Windows.

    The event loop's own timer wakes on the ~15.6 ms system tick (a 10 ms
    median lateness in the first smoke run), but `time.sleep` on CPython
    3.11+ uses a high-resolution waitable timer. So the wait runs in a worker
    thread and its completion wakes the loop. Very short waits stay on the
    loop, where a thread hop would cost more than it saves.
    """
    if seconds <= 0.002:
        await asyncio.sleep(seconds)
        return
    await asyncio.to_thread(time.sleep, seconds)


@contextlib.contextmanager
def fine_timer():
    """1 ms timer resolution on Windows for the life of the replay.

    Without it, a sleep on Windows rounds to the ~15.6 ms system tick
    (`soak_test_stream.py`'s first caveat), which is a fifth of the gap
    between two frames at 12 fps.
    """
    winmm = None
    if os.name == "nt":
        try:
            import ctypes

            winmm = ctypes.WinDLL("winmm")
            winmm.timeBeginPeriod(1)
        except Exception:  # noqa: BLE001 -- coarser pacing, recorded as lateness
            winmm = None
    try:
        yield
    finally:
        if winmm is not None:
            with contextlib.suppress(Exception):
                winmm.timeEndPeriod(1)


def distribution(values) -> dict:
    """count / mean / p50 / p95 / p99 / max of a list of numbers, rounded.
    Percentiles are nearest-rank on (n - 1), as in the report."""
    values = sorted(float(v) for v in values)
    if not values:
        return {"count": 0}

    def pick(q):
        return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]

    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 4),
        "p50": round(pick(0.5), 4),
        "p95": round(pick(0.95), 4),
        "p99": round(pick(0.99), 4),
        "max": round(values[-1], 4),
    }


# -- HTTP (urllib, in a thread) ----------------------------------------------


def http_json(method: str, url: str, timeout: float = 10.0):
    """(status, parsed JSON or None). Never raises for an HTTP or socket fault."""
    request = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        body, status = exc.read(), exc.code
    except (OSError, ValueError):
        return None, None
    try:
        return status, json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return status, None


def http_bytes(url: str, timeout: float = 30.0):
    """(status, byte count). Never raises for an HTTP or socket fault."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, len(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, 0
    except (OSError, ValueError):
        return None, 0


def http_raw(url: str, timeout: float = 30.0):
    """The exact HTTP response body, used only by the optional recorder."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (OSError, ValueError):
        return None, b""


def health_probe(url: str, timeout: float = LIVE_GUARD_TIMEOUT_S):
    """(kind, doc) for one GET of a `/health`. `kind` is `ok` (200 and a JSON
    object), `refused` (the connection was refused: nothing listens), or
    `timeout` / `error` (anything else: a slow Tower, a 5xx, a reset, a body
    that is not JSON). Never raises for an HTTP or socket fault."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        return "error", {"status": exc.code}
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, ConnectionRefusedError):
            return "refused", None
        if isinstance(reason, TimeoutError):
            return "timeout", None
        return "error", None
    except ConnectionRefusedError:
        return "refused", None
    except TimeoutError:  # socket.timeout is TimeoutError on 3.10+
        return "timeout", None
    except (OSError, ValueError):
        return "error", None
    if status != 200:
        return "error", {"status": status}
    try:
        doc = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return "error", None
    return ("ok", doc) if isinstance(doc, dict) else ("error", None)


# The /health blocks an "idle" decision needs (review C24 MED-5). The Tower
# always sends all three keys (`routes/health.py`); each is `null` when that
# subsystem is not configured, which is a definite answer ("no recorder",
# "no supervisor", "no chore"). An ABSENT key, a block that reports
# `error`, or a block without the field that is read is not.
HEALTH_BLOCKS = ("capture", "capture_workers", "background_chore")


def health_gaps(doc) -> list:
    """What an `idle` decision needs that this `/health` answer lacks: the
    absent, unreadable or malformed blocks, by name. Empty: complete."""
    if not isinstance(doc, dict):
        return list(HEALTH_BLOCKS)
    gaps = []
    for name, field_name, kind in (("capture", "recording", bool), ("capture_workers", "workers", list),
                                   ("background_chore", "state", str)):
        if name not in doc:
            gaps.append(f"{name} (absent)")
            continue
        block = doc[name]
        if block is None:
            continue  # not configured on this Tower: a definite answer
        if not isinstance(block, dict) or "error" in block or not isinstance(block.get(field_name), kind):
            gaps.append(f"{name}.{field_name}")
    return gaps


def classify_health(doc: dict) -> str:
    """`recording`, `busy`, `idle` or `unknown` from a `/health` answer.

    `busy` is a Tower finishing a world: after a real walk's Stop, :8000
    runs 35-55 min of GPU-heavy finishing with `recording: false`, and a
    replay started then distorts both runs (review C22 M2).

    FAILS CLOSED (review C24 MED-5): a positive answer (`recording`, then
    `busy`) is believed whatever else the answer lacks, but `idle` needs
    every block (`health_gaps`). `{}`, or an answer missing its worker or
    chore block, is `unknown`, which the guard refuses.
    """
    if not isinstance(doc, dict):
        return "unknown"
    capture = doc.get("capture") if isinstance(doc.get("capture"), dict) else {}
    if capture.get("recording"):
        return "recording"
    workers = (doc.get("capture_workers") or {}).get("workers") if isinstance(
        doc.get("capture_workers"), dict) else None
    chore = doc.get("background_chore") if isinstance(doc.get("background_chore"), dict) else {}
    if (isinstance(workers, list) and workers) or chore.get("state") == "running":
        return "busy"
    if health_gaps(doc):
        return "unknown"
    return "idle"


def live_tower_state(url: str = LIVE_HEALTH_URL, timeout: float = LIVE_GUARD_TIMEOUT_S) -> str:
    """:8000 as one of `recording`, `busy`, `idle`, `down`, `unknown`.

    FAILS CLOSED (review C22 M1). Only a refused connection is `down`: on
    loopback that is a definite "nothing listens", and a Tower that is not
    running records nothing. A timeout, a 5xx or a garbled answer is
    `unknown`, never "not recording": a live Tower that is recording, or
    finishing a world while the test Tower loads every core, answers late.
    So is an answer that parses but lacks a block the decision needs
    (`classify_health`, review C24 MED-5).
    """
    kind, doc = health_probe(url, timeout)
    if kind == "refused":
        return "down"
    if kind != "ok":
        return "unknown"
    return classify_health(doc)


class LiveGuard:
    """What :8000's state means once a replay (or its runner) is running.

    Aborts on `recording`, and on `LIVE_UNKNOWN_ABORT_AFTER` `unknown`
    answers in a row: one late answer is a loaded machine, two is a Tower
    that cannot be vouched for. `busy` does not abort -- a finisher chore
    can start on :8000 at any idle moment -- but when it began is kept, so
    the report can say the run was contended. Every change of state is kept.
    """

    def __init__(self, at_start: str | None = None, clock=time.time):
        self.at_start = at_start
        self._clock = clock
        self.unknown_streak = 0
        self.history: list = []
        self.busy_since = None
        self.last = at_start
        self.poll_count = 0
        self.last_probe_at = None
        if at_start is not None:
            self.history.append({"t": round(clock(), 3), "state": at_start})

    def observe(self, state: str) -> str | None:
        """An abort reason, or None to carry on."""
        now = self._clock()
        self.poll_count += 1
        self.last_probe_at = round(now, 3)
        if state != self.last:
            self.history.append({"t": round(now, 3), "state": state})
            self.last = state
        if state == "busy" and self.busy_since is None:
            self.busy_since = round(now, 3)
        if state == "recording":
            return "a live walk is being recorded on :8000"
        if state == "unknown":
            self.unknown_streak += 1
            if self.unknown_streak >= LIVE_UNKNOWN_ABORT_AFTER:
                return (f":8000 gave no usable /health answer {self.unknown_streak} times in a row "
                        "(timeout or error); failing closed")
        else:
            self.unknown_streak = 0
        return None

    def summary(self) -> dict:
        states = [item["state"] for item in self.history]
        return {"at_start": self.at_start, "history": self.history, "busy_since": self.busy_since,
                "states_seen": sorted(set(states)), "poll_count": self.poll_count,
                "last_probe_at": self.last_probe_at}


def listener_pids(port: int):
    """The pids listening on TCP `port`, or None when that cannot be read."""
    try:
        import psutil

        return {c.pid for c in psutil.net_connections("tcp")
                if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port}
    except Exception:  # noqa: BLE001 -- unknown is refused by the caller
        return None


def target_refusal(health: dict | None) -> str | None:
    """Why a test Tower must not be streamed into, or None (review C22 M4).
    A target that is recording or has capture workers is somebody's run."""
    if not isinstance(health, dict):
        return "the target gave no /health answer"
    state = classify_health(health)
    if state == "recording":
        return "the target is recording a capture already"
    workers = (health.get("capture_workers") or {}).get("workers") if isinstance(
        health.get("capture_workers"), dict) else None
    if isinstance(workers, list) and workers:
        return f"the target has {len(workers)} capture worker(s) alive"
    # What this decision reads must be there to be believed (review C24
    # MED-5). The chore is not read here: a target's chore is its own.
    gaps = [gap for gap in health_gaps(health) if not gap.startswith("background_chore")]
    if gaps:
        return f"the target's /health lacks {', '.join(gaps)}; it cannot be vouched for as idle"
    return None


def harness_identity() -> dict:
    """Which harness this is: the sha1 of its scripts, and git HEAD and
    the scripts' own uncommitted changes when they sit in a checkout
    (review C22 L5: "pin its SHA before any proof"). Read-only git, without
    optional locks."""
    scripts = Path(__file__).resolve().parent
    digest = hashlib.sha1()
    files = {}
    for name in HARNESS_FILES:
        try:
            data = (scripts / name).read_bytes()
        except OSError:
            files[name] = None
            continue
        files[name] = hashlib.sha1(data).hexdigest()
        digest.update(name.encode())
        digest.update(data)
    identity = {"version": HARNESS_VERSION, "scripts_dir": str(scripts), "sha1": digest.hexdigest(),
                "files_sha1": files}
    try:
        head = subprocess.run(["git", "--no-optional-locks", "-C", str(scripts), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20)
        if head.returncode == 0:
            identity["git_head"] = head.stdout.strip()
            dirty = subprocess.run(["git", "--no-optional-locks", "-C", str(scripts), "status",
                                    "--porcelain", "--", *HARNESS_FILES],
                                   capture_output=True, text=True, timeout=60)
            identity["git_dirty"] = [line for line in dirty.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return identity


def harness_pin(identity) -> dict:
    """The pin, in one place (manager 154 §2): the harness version, its git
    HEAD, whether the scripts were committed and clean there, and the
    sha1 of the STREAMING scripts (what `--compare` keys on). A record
    made before C22-F7 has no version: `None`, shown as not recorded."""
    identity = identity if isinstance(identity, dict) else {}
    files = identity.get("files_sha1") if isinstance(identity.get("files_sha1"), dict) else {}
    return {"version": identity.get("version"), "git_head": identity.get("git_head"),
            "committed_clean": bool(identity.get("git_head")) and identity.get("git_dirty") == [],
            "streaming_sha1": {name: files.get(name) for name in STREAMING_FILES}}


# -- what the phone would have been told ---------------------------------------


def summarize_payload(payload, prefix: str = "", depth: int = 0, out=None) -> dict:
    """The state-like scalar leaves of a status payload, by dotted path."""
    out = {} if out is None else out
    if not isinstance(payload, dict) or depth > 4 or len(out) >= 60:
        return out
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            summarize_payload(value, path + ".", depth + 1, out)
        elif key in PHONE_STATE_KEYS and (value is None or isinstance(value, (str, bool, int, float))):
            out[path] = value
    return out


def geometry_coordinates(payload):
    """(world_id, session_id, geometry revision), as the phone reads them
    (`TowerWorldBuilderClient.geometryCoordinates`). The fetch is keyed on
    `geometry.revision`, not on the snapshot's revision, so a keyframe count
    or a tracking state moving does not pull the geometry again."""
    if not isinstance(payload, dict):
        return None
    snapshot = payload.get("world_snapshot") if isinstance(payload.get("world_snapshot"), dict) else {}
    session = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    geometry = payload.get("geometry") if isinstance(payload.get("geometry"), dict) else {}
    world_id, session_id, revision = (snapshot.get("world_id"), session.get("session_id"),
                                      geometry.get("revision"))
    if all(isinstance(value, str) and value for value in (world_id, session_id, revision)):
        return world_id, session_id, revision
    return None


def photographic_word(payload):
    """`lifecycle.photographic` of a status push as (state, stage, scope,
    scope_present), or None when the push carries no such block.

    `scope` is `"area"` exactly when the word is an area's; absent means the
    room's (`WORLD-BUILDER-COMPONENTS.md` §3.4). The combined word moves on to
    an area only once the room's own word is settled, so it is how a phone
    learns that the room is done while its areas are still being made."""
    if not isinstance(payload, dict):
        return None
    lifecycle = payload.get("lifecycle") if isinstance(payload.get("lifecycle"), dict) else {}
    block = lifecycle.get("photographic")
    if not isinstance(block, dict):
        return None
    return block.get("state"), block.get("stage"), block.get("scope"), "scope" in block


class PhoneView:
    """The World Builder status pushes, reduced to their state transitions.

    `photographic` keeps every change of `lifecycle.photographic` (state,
    stage, scope) with the CLIENT's receive time: when the phone was told the
    room's photos were ready, which the store's `updated_at` cannot say
    (review C22 H2, `phone_photos_at`)."""

    def __init__(self, clock=time.time):
        self._clock = clock
        self.pushes = 0
        self.last: dict = {}
        self.transitions: list = []
        self.photographic: list = []
        self._photographic_last = None
        self.target = None

    def update(self, message: dict):
        self.pushes += 1
        payload = message.get("payload") or {}
        summary = summarize_payload(payload)
        changed = {k: v for k, v in summary.items() if self.last.get(k, object()) != v}
        gone = [k for k in self.last if k not in summary]
        if changed or gone:
            self.transitions.append({
                "t": round(self._clock(), 3),
                "seq": message.get("seq"),
                "changed": changed,
                "gone": gone,
            })
        self.last = summary
        word = photographic_word(payload)
        if word is not None and word != self._photographic_last:
            self._photographic_last = word
            self.photographic.append({"t": round(self._clock(), 3), "seq": message.get("seq"),
                                      "state": word[0], "stage": word[1], "scope": word[2],
                                      "scope_present": word[3]})
        coordinates = geometry_coordinates(payload)
        if coordinates is not None:
            self.target = coordinates[:2]
        return coordinates


class GeometryMirror:
    """Fetches what the phone fetches when the geometry revision moves: the
    manifest, then every segment whose (content, placement) key it lacks
    (`WorldBuilderClient.swift`, `geometryDidChange`). Coalesced, one fetch at
    a time for ordinary revisions. Coverage revisions have a separate priority
    worker so a slow ordinary fetch cannot consume their entire live window."""

    def __init__(self, base_url: str, enabled: bool = True, capture: ReplayCapture | None = None):
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.capture = capture
        self._wanted = asyncio.Event()
        self._pinned_wanted = asyncio.Event()
        self._pinned = deque()
        self._pinned_seen = set()
        self._active_pin = None
        self._target = None
        self._last_revision = None
        self._cache: set = set()
        self._cache_target = None
        self.manifests = 0
        self.segments = 0
        self.bytes = 0
        self.errors = 0
        self.capture_errors: list[str] = []
        self.manifest_ms: list = []
        self.segment_ms: list = []

    def request(self, coordinates, *, pinned: bool = False) -> None:
        """A status push named (world, session, geometry revision)."""
        if not self.enabled or coordinates is None:
            return
        if pinned and coordinates not in self._pinned_seen:
            self._pinned_seen.add(coordinates)
            self._pinned.append(coordinates)
            self._pinned_wanted.set()
        if coordinates == self._last_revision:
            return
        self._last_revision = coordinates
        self._target = coordinates
        self._wanted.set()

    async def run(self, stop: asyncio.Event) -> None:
        pinned = asyncio.create_task(self._run_pinned(stop))
        try:
            await self._run_ordinary(stop)
        finally:
            pinned.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pinned

    async def _run_ordinary(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            waiter = asyncio.create_task(self._wanted.wait())
            stopper = asyncio.create_task(stop.wait())
            await asyncio.wait({waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            stopper.cancel()
            if stop.is_set():
                return
            self._wanted.clear()
            target = self._target
            if target in self._pinned_seen:
                continue
            try:
                await self._fetch(target)
            except Exception as exc:  # noqa: BLE001 -- retain failure in the run record
                if self.capture is not None:
                    self.capture_errors.append(f"{type(exc).__name__}: {exc}")

    async def _run_pinned(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            waiter = asyncio.create_task(self._pinned_wanted.wait())
            stopper = asyncio.create_task(stop.wait())
            await asyncio.wait({waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            stopper.cancel()
            if stop.is_set():
                return
            self._pinned_wanted.clear()
            while self._pinned and not stop.is_set():
                target = self._pinned.popleft()
                self._active_pin = target
                try:
                    while target == self._target and not stop.is_set():
                        try:
                            if await self._fetch(target, pinned=True):
                                break
                        except Exception as exc:  # noqa: BLE001 -- retry a live pin
                            if self.capture is not None:
                                self.capture_errors.append(f"{type(exc).__name__}: {exc}")
                        if target == self._target:
                            await asyncio.sleep(0.1)
                finally:
                    self._active_pin = None

    async def _fetch(self, target, *, pinned: bool = False) -> bool:
        world_id, session_id = target[0], target[1]
        if (world_id, session_id) != self._cache_target:
            self._cache, self._cache_target = set(), (world_id, session_id)
        started = time.perf_counter()
        manifest_url = f"{self.base_url}/worlds/{world_id}/geometry/manifest?session_id={session_id}"
        if self.capture is None:
            status, manifest = await asyncio.to_thread(http_json, "GET", manifest_url, 30.0)
            manifest_body = None
        else:
            status, manifest_body = await asyncio.to_thread(http_raw, manifest_url, 30.0)
            try:
                manifest = json.loads(manifest_body) if status == 200 else None
            except ValueError:
                manifest = None
        self.manifest_ms.append((time.perf_counter() - started) * 1000)
        if status != 200 or not isinstance(manifest, dict):
            self.errors += status != 404
            return False
        self.manifests += 1
        fetched = []
        complete = True
        keys = {}
        for segment in manifest.get("segments") or []:
            key = (segment.get("content_hash"), segment.get("placement_hash"))
            keys[key] = segment.get("segment_index")
        for key, index in keys.items():
            if (key in self._cache and not pinned) or index is None:
                continue
            if not pinned and (self._wanted.is_set() or self._pinned_wanted.is_set()
                               or self._active_pin is not None):
                complete = False
                break  # superseded: the phone stops fetching a dead manifest
            started = time.perf_counter()
            segment_url = f"{self.base_url}/worlds/{world_id}/geometry/segment/{index}?session_id={session_id}"
            if self.capture is None:
                status, size = await asyncio.to_thread(http_bytes, segment_url)
            else:
                status, body = await asyncio.to_thread(http_raw, segment_url)
                size = len(body)
            self.segment_ms.append((time.perf_counter() - started) * 1000)
            if status == 200:
                self.segments += 1
                self.bytes += size
                self._cache.add(key)
                if self.capture is not None:
                    fetched.append((next(s for s in manifest["segments"] if s["segment_index"] == index), body))
            else:
                self.errors += 1
                complete = False
        self._cache &= set(keys)
        if self.capture is not None:
            return self.capture.geometry(target, manifest_body, fetched)
        return complete

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "manifests": self.manifests,
            "segments": self.segments,
            "bytes": self.bytes,
            "errors": self.errors,
            **({"capture_errors": self.capture_errors} if self.capture is not None else {}),
            "manifest_ms": distribution(self.manifest_ms),
            "segment_ms": distribution(self.segment_ms),
        }


# -- the socket --------------------------------------------------------------


@dataclass
class StreamStats:
    frames_sent: int = 0
    json_bytes_sent: int = 0
    jpeg_bytes_sent: int = 0
    frame_results: int = 0
    frame_errors: Counter = field(default_factory=Counter)
    other_messages: Counter = field(default_factory=Counter)
    ack_ms: list = field(default_factory=list)
    lateness_s: list = field(default_factory=list)
    first_send: float | None = None
    last_send: float | None = None
    connections: int = 0
    unexpected_closes: int = 0

    def summary(self) -> dict:
        span = (self.last_send - self.first_send) if (
            self.first_send is not None and self.last_send is not None) else 0.0
        late = self.lateness_s
        return {
            "frames_sent": self.frames_sent,
            "frame_results": self.frame_results,
            "frame_errors": dict(self.frame_errors),
            "unanswered": self.frames_sent - self.frame_results - sum(self.frame_errors.values()),
            "json_bytes_sent": self.json_bytes_sent,
            "jpeg_bytes_sent": self.jpeg_bytes_sent,
            "send_span_s": round(span, 3),
            "achieved_send_fps": round((self.frames_sent - 1) / span, 3) if span > 0 else None,
            "ack_ms": distribution(self.ack_ms),
            "lateness_ms": distribution([x * 1000 for x in late]),
            "late_over_50ms": sum(1 for x in late if x > 0.05),
            "late_over_250ms": sum(1 for x in late if x > 0.25),
            "other_messages": dict(self.other_messages),
            "connections": self.connections,
            "unexpected_closes": self.unexpected_closes,
        }


class TowerSocket:
    """One /ws connection, as the phone holds it: a reader that never blocks
    the sender, and replies matched to what was sent."""

    def __init__(self, uri: str, stats: StreamStats, phone: PhoneView, mirror: GeometryMirror,
                 timeline=None, capture: ReplayCapture | None = None):
        self.uri = uri
        self.stats = stats
        self.phone = phone
        self.mirror = mirror
        self.timeline = timeline
        self.capture = capture
        self.capture_id = None
        self.ws = None
        self._reader = None
        self._waiters: dict = {}
        self._sent_at: dict = {}
        self.subscription_id = None
        self.closed_by_us = False
        self.capture_error = None

    async def open(self) -> None:
        import websockets

        # No protocol pings: over loopback there is nothing to keep alive,
        # and a ping the loaded Tower answers late would close the socket
        # and look exactly like a phone dropping. No compression: the
        # phone's URLSession socket does not negotiate it.
        self.ws = await websockets.connect(
            self.uri, max_size=None, compression=None, open_timeout=20,
            ping_interval=None, close_timeout=5)
        self.stats.connections += 1
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            async for raw in self.ws:
                try:
                    message = json.loads(raw)
                except (ValueError, TypeError):
                    self.stats.other_messages["unparseable"] += 1
                    continue
                if isinstance(message, dict):
                    self._dispatch(message, raw)
        except Exception as exc:  # noqa: BLE001 -- a closed socket ends the reader
            if self.capture is not None:
                self.capture_error = f"{type(exc).__name__}: {exc}"
        finally:
            if not self.closed_by_us:
                self.stats.unexpected_closes += 1
            for futures in self._waiters.values():
                for future in futures:
                    if not future.done():
                        future.set_result(None)

    def _dispatch(self, message: dict, raw: str | bytes | None = None) -> None:
        kind = message.get("type")
        now = time.perf_counter()
        if self.timeline is not None:
            if kind in ("frame_result", "frame_error"):
                self.timeline.frame_reply(message.get("seq"), message, time.time())
            else:
                self.timeline.phone(message, time.time())
        if kind in ("frame_result", "frame_error"):
            sent = self._sent_at.pop(message.get("seq"), None)
            if sent is not None:
                self.stats.ack_ms.append((now - sent) * 1000)
            if kind == "frame_result":
                self.stats.frame_results += 1
            else:
                self.stats.frame_errors[str(message.get("reason"))] += 1
        elif kind == "cartridge_result":
            if message.get("cartridge") == WORLD_BUILDER:
                if self.capture is not None and raw is not None:
                    self.capture.status(raw, message)
                coordinates = self.phone.update(message)
                payload = message.get("payload") or {}
                guidance = payload.get("guidance") or {} if isinstance(payload, dict) else {}
                coverage = guidance.get("coverage") or {} if isinstance(guidance, dict) else {}
                pinned = (coordinates is not None and isinstance(coverage, dict)
                          and coverage.get("geometry_revision") == coordinates[2])
                self.mirror.request(coordinates, pinned=pinned)
        else:
            self.stats.other_messages[str(kind)] += 1
        for future in self._waiters.pop(kind, []):
            if not future.done():
                future.set_result(message)

    async def request(self, message: dict, reply: str, timeout: float = 15.0):
        future = asyncio.get_running_loop().create_future()
        self._waiters.setdefault(reply, []).append(future)
        await self.ws.send(encode(message))
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return None

    async def send_frame(self, frame: FrameRecord, text: str, jpeg_size: int) -> None:
        now = time.perf_counter()
        self._sent_at[frame.wire_seq] = now
        if self.timeline is not None:
            self.timeline.sent(self.capture_id, frame.source_seq, time.time())
        await self.ws.send(text)
        self.stats.frames_sent += 1
        self.stats.json_bytes_sent += len(text)
        self.stats.jpeg_bytes_sent += jpeg_size
        if self.stats.first_send is None:
            self.stats.first_send = now
        self.stats.last_send = now

    async def send(self, message: dict) -> None:
        await self.ws.send(encode(message))

    async def handshake(self, subscribe: bool) -> dict:
        """What the phone does on a new socket before it streams: a ping,
        the capability declaration, and the World Builder subscription."""
        info = {"pong": False, "offer": None, "subscribed": False}
        pong = await self.request({"type": "ping"}, "pong")
        info["pong"] = pong is not None
        if not subscribe:
            return info
        declaration = await self.request({"type": "cartridges"}, "cartridges")
        offers = (declaration or {}).get("cartridges") or []
        offer = next((o for o in offers if o.get("cartridge") == WORLD_BUILDER
                      and o.get("result_type") == WORLD_BUILDER_RESULT_TYPE), None)
        info["offer"] = offer
        if offer and offer.get("available", True):
            ack = await self.request({
                "type": "result_subscribe", "cartridge": WORLD_BUILDER,
                "result_type": WORLD_BUILDER_RESULT_TYPE, "contract": offer.get("contract"),
            }, "result_subscribed")
            info["subscribed"] = bool(ack and ack.get("type") == "result_subscribed")
            self.subscription_id = (ack or {}).get("subscription_id")
        return info

    async def close(self) -> None:
        self.closed_by_us = True
        if self.ws is not None:
            with contextlib.suppress(Exception):
                await self.ws.close()
        if self._reader is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._reader, 10)


# -- resources ---------------------------------------------------------------


SAMPLE_FIELDS = ("t", "sys_cpu_pct", "tree_cores", "tree_rss_mb", "tree_procs",
                 "busy", "gpu_util_pct", "gpu_mem_mb")


def _process_label(proc) -> str:
    try:
        argv = proc.cmdline()
    except Exception:  # noqa: BLE001
        return "?"
    for index, arg in enumerate(argv):
        if arg.endswith(".py"):
            return Path(arg).name
        if arg == "-m" and index + 1 < len(argv):
            return argv[index + 1]
    try:
        return proc.name()
    except Exception:  # noqa: BLE001
        return "?"


def gpu_sample():
    """(utilization %, memory used MiB) of GPU 0, or (None, None)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        util, mem = (part.strip() for part in out[0].split(","))
        return float(util), float(mem)
    except Exception:  # noqa: BLE001
        return None, None


class ResourceSampler(threading.Thread):
    """CPU (system, and the test Tower's process tree) and GPU, every few
    seconds, to a CSV. The tree's cores are CPU-seconds used between two
    samples divided by the wall time between them; `busy` names the
    processes using at least a third of a core, by script."""

    def __init__(self, root_pid: int | None, interval: float, csv_path: Path | None):
        super().__init__(name="c22-sampler", daemon=True)
        self.root_pid = root_pid
        self.interval = interval
        self.csv_path = csv_path
        self.rows: list = []
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            import psutil
        except ImportError:
            psutil = None
        previous: dict = {}
        last_t = time.time()
        if psutil is not None:
            psutil.cpu_percent(None)
        writer = handle = None
        if self.csv_path is not None:
            # Never appended to (review C22 L4): one file is one run's samples.
            handle = open(self.csv_path, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
        try:
            while not self._stop_event.wait(self.interval):
                row = self._sample(psutil, previous, last_t)
                last_t = row["t"]
                self.rows.append(row)
                if writer is not None:
                    writer.writerow(row)
                    handle.flush()
        finally:
            if handle is not None:
                handle.close()

    def _sample(self, psutil, previous: dict, last_t: float) -> dict:
        now = time.time()
        dt = max(1e-6, now - last_t)
        row = {"t": round(now, 3), "sys_cpu_pct": None, "tree_cores": None,
               "tree_rss_mb": None, "tree_procs": None, "busy": ""}
        if psutil is not None:
            row["sys_cpu_pct"] = psutil.cpu_percent(None)
            if self.root_pid is not None:
                try:
                    root = psutil.Process(self.root_pid)
                    tree = [root, *root.children(recursive=True)]
                except Exception:  # noqa: BLE001 -- the Tower has gone
                    tree = []
                cores = 0.0
                rss = 0
                busy = []
                seen = {}
                for proc in tree:
                    try:
                        with proc.oneshot():
                            times = proc.cpu_times()
                            total = times.user + times.system
                            key = (proc.pid, proc.create_time())
                            rss += proc.memory_info().rss
                    except Exception:  # noqa: BLE001 -- exited between the two calls
                        continue
                    seen[key] = total
                    if key in previous:
                        used = total - previous[key]
                    elif key[1] >= last_t:
                        used = total  # born since the last sample
                    else:
                        used = 0.0
                    share = max(0.0, used) / dt
                    cores += share
                    if share >= 0.33:
                        busy.append((share, _process_label(proc)))
                previous.clear()
                previous.update(seen)
                row["tree_cores"] = round(cores, 2)
                row["tree_rss_mb"] = round(rss / 2**20, 1)
                row["tree_procs"] = len(tree)
                row["busy"] = ";".join(f"{label}:{share:.1f}" for share, label in sorted(busy, reverse=True))
        util, mem = gpu_sample()
        row["gpu_util_pct"] = util
        row["gpu_mem_mb"] = mem
        return row


# -- the Tower's own view ------------------------------------------------------


def find_session_file(world_root: Path, capture_ids) -> Path | None:
    """The session.json whose capture is one of ours (a fresh root holds one)."""
    wanted = set(capture_ids)
    newest = None
    for path in Path(world_root).glob("worlds/*/sessions/*/session.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("capture_id") in wanted:
            if newest is None or path.stat().st_mtime > newest.stat().st_mtime:
                newest = path
    return newest


def _read_json(path: Path | None):
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def solution_path_for(session_path: Path | None) -> Path | None:
    """`worlds/<w>/solve/<s>/solution.json` beside `worlds/<w>/sessions/<s>/session.json`."""
    if session_path is None:
        return None
    session_path = Path(session_path)
    return session_path.parents[2] / "solve" / session_path.parent.name / "solution.json"


class SolutionSnapshots:
    """A copy of `solution.json` each time it changes, into `<out>/solution-snapshots/`.

    WHY (review C22 M6). A consensus solve publishes draw 0 first, its
    consensus `deferred`, and overwrites it minutes later with the chosen
    draw's. Draw 0's map and gate times, and when it was published, exist
    only in that first version. The file is written atomically by the
    Tower, so a copy is never torn; a copy that does not parse is skipped
    and retried on the next poll. Reads the data root; writes only `out`.
    """

    def __init__(self, out: Path):
        self.dir = Path(out) / "solution-snapshots"
        self.items: list = []
        self._last = None

    def poll(self, path: Path | None) -> bool:
        if path is None:
            return False
        try:
            st = Path(path).stat()
            data = Path(path).read_bytes()
        except OSError:
            return False
        stamp = (st.st_mtime_ns, st.st_size)
        if stamp == self._last:
            return False
        try:
            doc = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return False
        self._last = stamp
        n = len(self.items)
        name = f"{n:03d}.json"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / name).write_bytes(data)
        gate = doc.get("gate") if isinstance(doc.get("gate"), dict) else {}
        consensus = gate.get("consensus") if isinstance(gate.get("consensus"), dict) else {}
        timing = doc.get("timing") if isinstance(doc.get("timing"), dict) else {}
        self.items.append({
            "n": n, "file": name, "mtime": round(st.st_mtime, 3), "seen_at": round(time.time(), 3),
            "solved_at": doc.get("solved_at"), "consensus_state": consensus.get("state"),
            "map_s": timing.get("map_s"), "gate_s": timing.get("gate_s"),
        })
        index = self.dir / "index.json.tmp"
        index.write_text(json.dumps(self.items, indent=2), encoding="utf-8")
        os.replace(index, self.dir / "index.json")
        return True


class SurfaceWatch:
    """The session's `surface/<s>/status.json`, read while the run lasts.

    WHY. Each live surface and then the final one overwrite that one file,
    and the Tower logs a live surface's launch but not its end, so a
    finished run's store cannot say when a live surface became `ok`. Each
    change of (pid, state, stage, updated_at) is kept, with the kind -- live
    or final -- read from the params digest. Read only.
    """

    def __init__(self):
        self.transitions: list = []
        self._last = None

    def poll(self, path: Path) -> bool:
        doc = _read_json(path)
        if not isinstance(doc, dict):
            return False
        key = (doc.get("pid"), doc.get("state"), doc.get("stage"), doc.get("updated_at"))
        if key == self._last:
            return False
        self._last = key
        digest = str(doc.get("params_digest") or "")
        kind = "live" if "|live|" in digest else "final" if "|final|" in digest else None
        self.transitions.append({"t": round(time.time(), 3), "pid": key[0], "state": key[1],
                                 "stage": key[2], "updated_at": key[3], "kind": kind})
        return True


def settle_snapshot(health: dict | None, session: dict | None) -> dict:
    """The facts the settle follower watches, flattened."""
    snap = {"health": health is not None}
    if isinstance(health, dict):
        workers = ((health.get("capture_workers") or {}).get("workers")) or []
        snap["workers"] = len(workers) if isinstance(workers, list) else None
        chore = health.get("background_chore")
        if isinstance(chore, dict):
            snap["chore_state"] = chore.get("state")
            snap["chore_runs"] = chore.get("runs")
            snap["chore_last"] = (chore.get("last") or {}).get("outcome")
        else:
            snap["chore_state"] = None
        capture = health.get("capture") or {}
        snap["recording"] = bool(isinstance(capture, dict) and capture.get("recording"))
    if isinstance(session, dict):
        final = session.get("finalization") or {}
        snap["finalization"] = final.get("state")
        snap["final_solve"] = final.get("final_solve")
        for stage, record in (session.get("stages") or {}).items():
            if isinstance(record, dict):
                snap[f"stage.{stage}"] = record.get("state")
        snap["frames_observed"] = session.get("frames_observed")
        snap["keyframes_accepted"] = session.get("keyframes_accepted")
    return snap


class SettleJudge:
    """Decides when the Tower has settled after Stop.

    Settled: no capture worker is alive, and the finisher chore (if this
    Tower runs one) is idle having run at least once since Stop. The walk
    held the chore off (`ws.py` `_yield_background_work`), so it is owed and
    will run once the builder exits; until it has, the areas are not done.
    """

    def __init__(self, chore_runs_at_stop):
        self.runs_at_stop = chore_runs_at_stop
        self.worker_seen_gone = False

    def settled(self, snap: dict) -> bool:
        if not snap.get("health"):
            return False
        if snap.get("workers") != 0:
            return False
        self.worker_seen_gone = True
        state = snap.get("chore_state")
        if state is None:
            return True  # no chore on this Tower
        runs = snap.get("chore_runs") or 0
        base = self.runs_at_stop or 0
        return state == "idle" and runs > base


# -- the replay ----------------------------------------------------------------


@dataclass
class ReplayOptions:
    port: int
    captures: list
    capture_root: Path
    out: Path
    world_root: Path | None = None
    tower_pid: int | None = None
    follow_chain: bool = True
    speed: float = 1.0
    first_seconds: float | None = None
    end_with_stop: bool = False
    settle_timeout_min: float = 90.0
    poll_seconds: float = 5.0
    sample_seconds: float = 3.0
    after_stop: str = "stay"  # stay | disconnect
    subscribe: bool = True
    phone_fetches: bool = True
    start_session: bool = True
    session_lead: float = 2.0
    live_guard: bool = True
    # Declared NOT proof (a smoke, a test). The CLIs refuse `--no-live-guard`
    # without it (review C24 HIGH-3); either one marks the run NOT-PROOF.
    not_a_proof_run: bool = False
    live_guard_url: str = LIVE_HEALTH_URL
    live_guard_every: float = 10.0
    live_guard_timeout: float = LIVE_GUARD_TIMEOUT_S
    # Called (in a thread) the moment the live guard aborts, BEFORE the
    # client tears down: the runner passes "kill the test Tower", so the GPU
    # is free at once rather than after the teardown's waits (review C22 M3).
    on_abort: object = None
    on_guard_armed: object = None
    surface_watch_seconds: float = 1.0
    label: str | None = None
    calibration_root: Path | None = None
    calibration_expected: dict[str, str] | None = None
    live_timeline: bool = False
    capture_status_geometry: bool = False
    tower_log: Path | None = None


def _log(out: Path, text: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
    print(line, flush=True)
    with open(out / "client.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


async def _guard_live(options: ReplayOptions, abort: asyncio.Event, record: dict, stop: asyncio.Event,
                      guard: LiveGuard | None = None, first_observation: asyncio.Event | None = None):
    """Watch :8000 for the whole run, streaming and settle alike. On an abort
    reason: set `abort`, then run `on_abort` (kill the test Tower) and only
    then return, so the caller's teardown comes after the kill."""
    guard = guard if guard is not None else LiveGuard(record.get("live_tower_at_start"))
    busy_logged = False
    async def refuse(reason: str) -> None:
        record["aborted"] = {"t": round(time.time(), 3), "reason": reason}
        abort.set()
        if first_observation is not None:
            first_observation.set()
        _log(options.out, f"ABORT: {reason}; stopping the replay now")
        if options.on_abort is not None:
            try:
                await asyncio.to_thread(options.on_abort)
                record["aborted"]["on_abort_done"] = round(time.time(), 3)
            except Exception as exc:  # noqa: BLE001 -- the runner's finally stops it again
                record["aborted"]["on_abort_error"] = repr(exc)

    try:
        while not stop.is_set() and not abort.is_set():
            state = await asyncio.to_thread(live_tower_state, options.live_guard_url, options.live_guard_timeout)
            reason = guard.observe(state)
            record["live_tower_watch"] = guard.summary()
            if guard.busy_since is not None and not busy_logged:
                busy_logged = True
                _log(options.out, ":8000 is busy finishing a world; this run is contended from here "
                                  "(recorded in client.json live_tower_watch)")
            if reason:
                await refuse(reason)
                return
            if first_observation is not None:
                if state not in ("idle", "down"):
                    await refuse(f":8000 first client guard observation was {state}; refusing handoff")
                    return
                first_observation.set()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), options.live_guard_every)
    except Exception as exc:  # noqa: BLE001 -- any failed observer must stop proof traffic
        record["live_guard_error"] = f"{type(exc).__name__}: {exc}"
        record["live_tower_watch"] = guard.summary()
        if not abort.is_set():
            await refuse(f":8000 live guard failed unexpectedly ({record['live_guard_error']})")
    finally:
        if first_observation is not None:
            first_observation.set()


async def _watch_surface(options: ReplayOptions, phone: "PhoneView", watch: SurfaceWatch,
                         stop: asyncio.Event) -> None:
    """Poll the session's surface status.json once the pushes name the session."""
    while not stop.is_set():
        if options.world_root is not None and phone.target:
            world_id, session_id = phone.target
            path = Path(options.world_root) / "worlds" / world_id / "surface" / session_id / "status.json"
            with contextlib.suppress(Exception):
                await asyncio.to_thread(watch.poll, path)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), options.surface_watch_seconds)


async def _watch_solutions(options: ReplayOptions, phone: "PhoneView", snapshots: SolutionSnapshots,
                           stop: asyncio.Event) -> None:
    """Catch live solve publishes before the next solve overwrites solution.json."""
    while not stop.is_set():
        if options.world_root is not None and phone.target:
            world_id, session_id = phone.target
            path = Path(options.world_root) / "worlds" / world_id / "solve" / session_id / "solution.json"
            await asyncio.to_thread(snapshots.poll, path)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), 0.25)


async def _wait_or_abort(abort: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(abort.wait(), seconds)


async def run_replay(options: ReplayOptions) -> dict:
    """Stream the walk, follow the settle, and return everything the client saw.

    Also writes `client.json` (and `samples.csv`) into `options.out`.

    `phone_view.photographic` holds when THIS CLIENT received each status
    push; the report's W0 verdict is judged on it (`phone_photos_at`, review
    C24 HIGH-1). It is the replay client's receipt, not photos rendered on a
    phone.
    """
    check_target_port(options.port)
    if options.capture_status_geometry and (options.world_root is None or not options.subscribe
                                            or not options.phone_fetches):
        raise SystemExit("--capture-status-geometry requires --world-root, status subscription, "
                         "and phone geometry fetches")
    out = Path(options.out)
    out.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{options.port}"
    record: dict = {
        "tool": "world_live_replay",
        "label": options.label,
        "port": options.port,
        "captures_replayed": list(options.captures),
        # Where the source journals were read (read only): the report's
        # Tower-side pacing row joins the re-recorded capture to them.
        "capture_root": str(options.capture_root),
        "speed": options.speed,
        "first_seconds": options.first_seconds,
        "after_stop": options.after_stop,
        # The rest of the replay's shape, for `--compare`'s comparability key
        # (review C24 HIGH-2).
        "options": {"end_with_stop": options.end_with_stop, "follow_chain": options.follow_chain,
                    "subscribe": options.subscribe, "phone_fetches": options.phone_fetches,
                    "start_session": options.start_session, "session_lead": options.session_lead},
        # Whether this run can be proof at all (review C24 HIGH-3): an
        # unguarded or declared-not-proof run is NOT-PROOF in its report.
        "live_guard": bool(options.live_guard),
        "not_a_proof_run": bool(options.not_a_proof_run),
        "started_at": round(time.time(), 3),
        "outcome": None,
    }
    if options.capture_status_geometry:
        record["options"]["capture_status_geometry"] = True

    # The :8000 guard is read and ARMED FIRST (review C24 HIGH-3), before the
    # harness pin's git calls, the walk's journals and the target checks:
    # from here to the teardown nothing the client does is unwatched.
    guard = guard_task = None
    abort = asyncio.Event()
    stop_background = asyncio.Event()
    if options.live_guard:
        live = await asyncio.to_thread(live_tower_state, options.live_guard_url, options.live_guard_timeout)
        record["live_tower_at_start"] = live
        if live in LIVE_REFUSE_AT_START:
            record["outcome"] = {"recording": "refused-live-walk", "busy": "refused-live-busy",
                                 "unknown": "refused-live-unknown"}[live]
            _log(out, {"recording": "REFUSED: :8000 is recording a live walk",
                       "busy": "REFUSED: :8000 is finishing a world (capture workers alive or the chore running)",
                       "unknown": "REFUSED: :8000 gave no usable /health answer (a timeout, an error, or an "
                                  "answer missing capture / capture_workers / background_chore); the guard "
                                  "fails closed",
                       }[live] + "; nothing was streamed")
            record["harness"] = await asyncio.to_thread(harness_identity)
            record["harness_pin"] = harness_pin(record["harness"])
            _write_client(out, record)
            return record
        guard = LiveGuard(live)
        first_observation = asyncio.Event()
        guard_task = asyncio.create_task(_guard_live(
            options, abort, record, stop_background, guard, first_observation))
        # Keep the runner's faster preflight watcher until this independent
        # client observer has actually completed its first /health read.
        await first_observation.wait()
        if abort.is_set():
            stop_background.set()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(asyncio.shield(guard_task), 45)
            record["live_tower_watch"] = guard.summary()
            record["outcome"] = "aborted"
            _write_client(out, record)
            return record
    # The runner's background preflight observer hands off only after this
    # client's live guard is armed. It then stops its faster preflight polls.
    if options.on_guard_armed is not None:
        await asyncio.to_thread(options.on_guard_armed)
    record["harness"] = await asyncio.to_thread(harness_identity)
    record["harness_pin"] = harness_pin(record["harness"])

    async def finish_before_streaming(outcome: str, text: str) -> dict:
        """End the run before a frame was sent: disarm the guard (after its
        kill, when it aborted), keep what it saw, write the record."""
        stop_background.set()
        if guard_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(asyncio.shield(guard_task), 45)
            record["live_tower_watch"] = guard.summary()
        record["outcome"] = "aborted" if abort.is_set() else outcome
        _log(out, text)
        _write_client(out, record)
        return record

    walk = load_walk(options.capture_root, options.captures, follow_chain=options.follow_chain)
    record["source_journals"] = {c.capture_id: c.source_sha256 for c in walk}
    schedule = build_schedule(walk, speed=options.speed, first_seconds=options.first_seconds,
                              end_with_stop=options.end_with_stop)
    record["walk"] = [{
        "capture_id": c.capture_id, "started_at": c.started_at, "ended_at": c.ended_at,
        "end_reason": c.end_reason, "continues": c.continues, "frames": len(c.frames),
        "recorded_seconds": round(c.recorded_seconds, 3),
        "recorded_fps": round(len(c.frames) / c.recorded_seconds, 3) if c.recorded_seconds else None,
    } for c in walk]
    record["schedule"] = schedule_summary(schedule)
    _log(out, f"walk: {len(walk)} capture(s) {[c.capture_id for c in walk]}; schedule "
              f"{record['schedule']}")

    try:
        planned_inputs = pin_frame_inputs(walk, schedule)
    except OSError as exc:
        record["input_error"] = f"source JPEG unavailable before streaming: {exc}"
        return await finish_before_streaming("invalid-input", record["input_error"])
    record["source_images"] = {"planned": planned_inputs,
                               "planned_sha256": input_list_sha256(planned_inputs),
                               "sent": [], "sent_sha256": None, "verified": False}
    if options.calibration_expected is not None:
        try:
            actual = calibration_digests(options.calibration_root)
        except (OSError, TypeError) as exc:
            actual = {"error": str(exc)}
        record["calibration_check"] = {"expected": options.calibration_expected,
                                       "before_stream": actual, "after_stream": None,
                                       "verified": False}
        if not actual or actual != options.calibration_expected:
            return await finish_before_streaming("invalid-input", "copied calibration is missing or changed before streaming")

    status, health = await asyncio.to_thread(http_json, "GET", f"{base}/health", 10.0)
    if status != 200:
        return await finish_before_streaming("no-test-tower", f"no test Tower answers {base}/health")
    record["tower_health_at_start"] = settle_snapshot(health, None)
    # Is the target idle, and is it ours? (review C22 M4) A second
    # `stream_start` into somebody else's run supersedes its recording.
    busy = target_refusal(health)
    if busy is not None:
        return await finish_before_streaming(
            "refused-target-busy", f"REFUSED: {busy} on :{options.port}; nothing was streamed")
    if options.tower_pid is None:
        record["target_listener"] = {"checked": False, "why": "no tower pid given"}
    else:
        owners = await asyncio.to_thread(listener_pids, options.port)
        record["target_listener"] = {"checked": True, "pids": None if owners is None else sorted(owners),
                                     "expected": options.tower_pid}
        if owners != {options.tower_pid}:
            return await finish_before_streaming(
                "refused-target-foreign", f"REFUSED: :{options.port} is served by {owners}, not the test "
                                          f"Tower {options.tower_pid}; nothing was streamed")
    if abort.is_set():
        return await finish_before_streaming("aborted", "the :8000 guard aborted the run before streaming; "
                                                        "nothing was streamed")

    stats = StreamStats()
    phone = PhoneView()
    capture = ReplayCapture(out) if options.capture_status_geometry else None
    live_solutions = SolutionSnapshots(out) if capture is not None else None
    solution_watch_stop = asyncio.Event()
    mirror = GeometryMirror(base, enabled=options.phone_fetches, capture=capture)
    timeline = None
    if options.live_timeline:
        if options.world_root is None:
            raise SystemExit("--live-timeline requires --world-root")
        from scripts.world_live_timeline import LiveTimeline  # noqa: PLC0415
        timeline = LiveTimeline(out, options.world_root)
    surfaces = SurfaceWatch()
    background = [asyncio.create_task(mirror.run(stop_background)),
                  asyncio.create_task(_watch_surface(options, phone, surfaces, stop_background))]
    solution_watch = None
    if live_solutions is not None:
        solution_watch = asyncio.create_task(
            _watch_solutions(options, phone, live_solutions, solution_watch_stop))
    if guard_task is not None:
        background.append(guard_task)
    sampler = ResourceSampler(options.tower_pid, options.sample_seconds, out / "samples.csv")
    sampler.start()
    uri = f"ws://127.0.0.1:{options.port}/ws"
    events: list = []
    capture_socket_errors: list[str] = []
    tower_captures: list = []
    if timeline is not None:
        async def watch_timeline():
            while not stop_background.is_set():
                await asyncio.to_thread(timeline.poll, tower_captures)
                try:
                    await asyncio.wait_for(stop_background.wait(), timeline.poll_interval)
                except asyncio.TimeoutError:
                    pass
            await asyncio.to_thread(timeline.poll, tower_captures)
        background.append(asyncio.create_task(watch_timeline()))
    sent_inputs: list[dict] = record["source_images"]["sent"]
    frame_input_index = 0

    def note(kind: str, **extra) -> None:
        events.append({"t": round(time.time(), 3), "kind": kind, **extra})

    async def start_session() -> dict:
        status, body = await asyncio.to_thread(
            http_json, "POST", f"{base}/cartridges/{WORLD_BUILDER}/session/start", 15.0)
        return {"status": status, "state": (body or {}).get("state"),
                "session_id": (body or {}).get("session_id")}

    socket = None
    try:
        with fine_timer():
            socket = TowerSocket(uri, stats, phone, mirror, timeline, capture)
            await socket.open()
            record["handshake"] = await socket.handshake(options.subscribe)
            note("connected", handshake=record["handshake"])
            if options.start_session:
                record["session_start"] = await start_session()
                note("session_start", status=record["session_start"]["status"])
                _log(out, f"POST session/start -> {record['session_start']['status']} {record['session_start']}")
                await _wait_or_abort(abort, options.session_lead)

            pacer = Pacer()
            pacer.start()
            record["t0"] = round(time.time(), 3)
            _log(out, "streaming")
            progress_every = 30.0
            next_progress = progress_every
            for step in schedule:
                if abort.is_set():
                    break
                text = size = None
                if step.kind == "frame":
                    if options.calibration_expected is not None and \
                            calibration_digests(options.calibration_root) != options.calibration_expected:
                        raise RuntimeError("copied calibration changed during the stream")
                    jpeg = (walk[step.capture].directory / step.frame.relpath).read_bytes()
                    entry = frame_input_entry(walk[step.capture], step.frame, jpeg)
                    expected = planned_inputs[frame_input_index]
                    if entry != expected:
                        raise RuntimeError(f"source JPEG changed during the stream: {entry['capture_id']}/"
                                           f"{entry['relpath']}")
                    text, size = encode(frame_message(step.frame, jpeg)), len(jpeg)
                late = await pacer.wait_until(step.at, abort)
                if abort.is_set():
                    break
                if step.kind == "frame":
                    # The pacing wait can outlast a source-file mutation. The
                    # encoded payload above is pinned, but the source must
                    # still be the same at the send boundary for a proof run.
                    if options.calibration_expected is not None and \
                            calibration_digests(options.calibration_root) != options.calibration_expected:
                        raise RuntimeError("copied calibration changed during the stream")
                    try:
                        on_disk = sha256_file(walk[step.capture].directory / step.frame.relpath)
                    except OSError as exc:
                        raise RuntimeError(f"source JPEG disappeared during the stream: {entry['relpath']}") from exc
                    if on_disk != entry["sha256"]:
                        raise RuntimeError(f"source JPEG changed during the stream: {entry['capture_id']}/"
                                           f"{entry['relpath']}")
                    # Integrity reads happen after the pacing wait. Measure at
                    # the actual call boundary so their I/O cannot hide a late
                    # frame from the client-lateness fidelity bar.
                    stats.lateness_s.append(pacer.elapsed() - step.at)
                    if timeline is not None:
                        socket.capture_id = walk[step.capture].capture_id
                    await socket.send_frame(step.frame, text, size)
                    sent_inputs.append(entry)
                    frame_input_index += 1
                    if step.at >= next_progress:
                        next_progress += progress_every
                        _log(out, f"t={step.at:6.1f}s sent {stats.frames_sent} frames, "
                                  f"{stats.frame_results} results, {sum(stats.frame_errors.values())} errors")
                elif step.kind == "connect":
                    socket = TowerSocket(uri, stats, phone, mirror, timeline, capture)
                    await socket.open()
                    handshake = await socket.handshake(options.subscribe)
                    # The phone re-sends `start` on every reconnect
                    # (`towerReachabilityChanged`). It is the only thing that
                    # moves `requested_at`, so without it the Tower's deferred
                    # stop ends World Builder 105 s after the disconnect
                    # (review C22 H1).
                    restarted = await start_session() if options.start_session else None
                    note("reconnected", handshake=handshake, session_start=restarted)
                elif step.kind == "stream_start":
                    await socket.send({"type": "stream_start"})
                    note("stream_start", capture=step.capture, late_s=round(late, 4))
                    # Off the send path: the Tower is spawning the builder
                    # right now, and an inline /health held the first frames
                    # back by ~0.5 s in the first smoke run.
                    background.append(asyncio.create_task(_learn_capture(base, tower_captures)))
                elif step.kind == "stream_stop":
                    await socket.send({"type": "stream_stop"})
                    if timeline is not None:
                        timeline.stop(time.time())
                    note("stream_stop", capture=step.capture, late_s=round(late, 4))
                elif step.kind == "disconnect":
                    await socket.close()
                    if socket.capture_error is not None:
                        capture_socket_errors.append(socket.capture_error)
                    note("disconnect", capture=step.capture, late_s=round(late, 4))
            record["stopped_at"] = round(time.time(), 3)
            record["tower_captures"] = tower_captures
            if abort.is_set():
                record["outcome"] = "aborted"
            else:
                _log(out, f"Stop sent; {stats.frames_sent} frames sent, {stats.frame_results} "
                          f"results; following the settle (timeout {options.settle_timeout_min} min)")

        if not abort.is_set():
            await asyncio.sleep(1.0)  # the last results arrive
            if options.after_stop == "disconnect" and socket is not None:
                await socket.close()
                note("disconnect_after_stop")
            if solution_watch is not None:
                solution_watch_stop.set()
                await solution_watch
            settle = await _follow_settle(options, base, tower_captures, abort, out, live_solutions)
            record["settle"] = settle
            if abort.is_set():
                record["outcome"] = "aborted"
            else:
                record["outcome"] = "settled" if settle.get("settled") else "not-settled"
    except Exception as exc:
        # THE ABORT EDGE (review C22 round 2, "Still open" 5). The guard sets
        # `abort` BEFORE `on_abort` kills the test Tower, so a send that was
        # already waiting in the socket's drain when the Tower died raises
        # here with `abort` set. That is the abort, not a harness fault: the
        # record says `aborted` (exit 3) and is written by the `finally`,
        # rather than the exception reaching the runner as an error (exit 1)
        # with no client record. Anything raised WITHOUT an abort is still
        # an error and still propagates.
        if not abort.is_set():
            raise
        # Swallowed, so its traceback is kept, in the record (a string) and
        # in client.log (review C22 round 3 L-c): if it was not the kill's
        # consequence after all, this is the only trace of where it came from.
        trace = traceback.format_exc()
        record["outcome"] = "aborted"
        record.setdefault("aborted", {})["stream_error"] = f"{type(exc).__name__}: {exc}"
        record["aborted"]["stream_traceback"] = trace
        record.setdefault("stopped_at", round(time.time(), 3))
        _log(out, f"the stream ended under the abort: {exc!r}\n{trace.rstrip()}")
    finally:
        stop_background.set()
        solution_watch_stop.set()
        if solution_watch is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await solution_watch
        aborted = abort.is_set()
        if aborted and guard_task is not None:
            # The guard kills the test Tower (`on_abort`) before it returns:
            # let it finish that first, so the teardown comes after the kill.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(asyncio.shield(guard_task), 45)
        for task in background:
            if aborted and not task.done():
                task.cancel()  # nothing they would still learn matters after an abort
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, 5 if aborted else 35)
        if socket is not None:
            await socket.close()
            if socket.capture_error is not None:
                capture_socket_errors.append(socket.capture_error)
        sampler.stop()
        sampler.join(timeout=5 if aborted else 20)
        record["events"] = events
        record["stream"] = stats.summary()
        record["phone_view"] = {"pushes": phone.pushes, "target": phone.target,
                                "transitions": phone.transitions,
                                "photographic": phone.photographic}
        record["phone_fetches"] = mirror.summary()
        if capture is not None:
            solutions = (record.get("settle") or {}).get("solution_snapshots", [])
            capture.finish(record.get("t0"), solutions)
            capture_summary = capture.capture_summary()
            record["status_geometry_capture"] = {"status_pushes": capture.status_count,
                                                    "geometry_revisions": len(capture.geometry_rows),
                                                    "socket_errors": capture_socket_errors,
                                                    "geometry_errors": mirror.capture_errors,
                                                    **capture_summary}
            # A broken status stream is always incomplete; geometry is incomplete only when a
            # coverage-pinned revision was missed (unpinned misses are reported, not fatal).
            if capture_socket_errors or capture_summary["pinned_missed"]:
                record["outcome"] = "capture-incomplete"
        record["surface_watch"] = surfaces.transitions
        if timeline is not None:
            record["live_timeline"] = timeline.finish(phone.photographic, options.tower_log)
        if guard is not None:
            record["live_tower_watch"] = guard.summary()
        record["tower_captures"] = tower_captures
        record["ended_at"] = round(time.time(), 3)
        source = record["source_images"]
        source["sent_sha256"] = input_list_sha256(sent_inputs)
        source["verified"] = (len(sent_inputs) == len(planned_inputs)
                              and source["sent_sha256"] == source["planned_sha256"]
                              and record.get("outcome") == "settled")
        if options.calibration_expected is not None:
            check = record["calibration_check"]
            try:
                check["after_stream"] = calibration_digests(options.calibration_root)
            except (OSError, TypeError) as exc:
                check["after_stream"] = {"error": str(exc)}
            check["verified"] = (check["before_stream"] == check["after_stream"]
                                 == options.calibration_expected)
        _write_client(out, record)
    return record


async def _learn_capture(base: str, captures: list, polls: int = 40) -> None:
    """The capture id the Tower minted for the stream just opened (/health).

    Keeps polling until an id it has not seen appears (review C22 L1): right
    after a reconnect /health can still name the previous capture.
    """
    for _ in range(polls):
        await asyncio.sleep(0.5)
        status, health = await asyncio.to_thread(http_json, "GET", f"{base}/health", 10.0)
        cid = ((health or {}).get("capture") or {}).get("capture_id") if status == 200 else None
        if cid and cid not in captures:
            captures.append(cid)
            return


async def _follow_settle(options: ReplayOptions, base: str, captures: list,
                         abort: asyncio.Event, out: Path,
                         snapshots: SolutionSnapshots | None = None) -> dict:
    started = time.time()
    deadline = started + options.settle_timeout_min * 60.0
    transitions: list = []
    last: dict = {}
    judge = None
    session_path = None
    settled_at = None
    snapshots = snapshots if snapshots is not None else SolutionSnapshots(out)
    while time.time() < deadline and not abort.is_set():
        status, health = await asyncio.to_thread(http_json, "GET", f"{base}/health", 10.0)
        if options.world_root is not None and captures and session_path is None:
            session_path = await asyncio.to_thread(find_session_file, options.world_root, captures)
        with contextlib.suppress(Exception):
            if await asyncio.to_thread(snapshots.poll, solution_path_for(session_path)):
                item = snapshots.items[-1]
                _log(out, f"solution.json changed: snapshot {item['file']} (consensus "
                          f"{item['consensus_state']}, map_s {item['map_s']})")
        session = await asyncio.to_thread(_read_json, session_path)
        snap = settle_snapshot(health if status == 200 else None, session)
        if judge is None and snap.get("health"):
            judge = SettleJudge(snap.get("chore_runs"))
        changed = {k: v for k, v in snap.items() if last.get(k, object()) != v}
        if changed:
            transitions.append({"t": round(time.time(), 3), "changed": changed})
            _log(out, f"+{(time.time() - started) / 60:5.1f} min {changed}")
        last = snap
        if judge is not None and judge.settled(snap):
            settled_at = time.time()
            break
        await _wait_or_abort(abort, options.poll_seconds)
    return {
        "settled": settled_at is not None,
        "settled_at": None if settled_at is None else round(settled_at, 3),
        "timed_out": settled_at is None and not abort.is_set(),
        "followed_from": round(started, 3),
        "session_file": None if session_path is None else str(session_path),
        "transitions": transitions,
        "last": last,
        "solution_snapshots": snapshots.items,
    }


def _write_client(out: Path, record: dict) -> None:
    tmp = out / "client.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, out / "client.json")


def sha256_file(path: Path) -> str:
    """Hash bytes from the file the replay will use, without trusting journal sizes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_digests(root: Path) -> dict[str, str]:
    """The actual copied JSONs visible to the test Tower."""
    return {path.name: sha256_file(path) for path in sorted(Path(root).glob("*.json"))}


def input_list_sha256(entries: list[dict]) -> str:
    """Stable, ordered manifest digest; order and capture identity matter."""
    wire = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def frame_input_entry(capture, frame, jpeg: bytes) -> dict:
    return {"capture_id": capture.capture_id, "relpath": frame.relpath,
            "wire_seq": frame.wire_seq, "bytes": len(jpeg),
            "sha256": hashlib.sha256(jpeg).hexdigest()}


def pin_frame_inputs(walk, schedule) -> list[dict]:
    """Read every scheduled JPEG before proof traffic. Missing files fail now."""
    return [frame_input_entry(walk[step.capture], step.frame,
                              (walk[step.capture].directory / step.frame.relpath).read_bytes())
            for step in schedule if step.kind == "frame"]


# -- CLI -------------------------------------------------------------------------


def add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    """The replay's own options; the runner shares them."""
    parser.add_argument("--capture", nargs="+", required=True,
                        help="Capture id(s). One id is followed through continues_capture "
                             "both ways (a reconnected walk); several are replayed in the order given.")
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT,
                        help="Where the stored captures live (<root>/<id>/frames.jsonl). Read only.")
    parser.add_argument("--no-follow-chain", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Pace factor; 1.0 is the recorded pace.")
    parser.add_argument("--first-seconds", type=float, default=None,
                        help="Replay only the first N seconds of the walk, then Stop.")
    parser.add_argument("--end-with-stop", action="store_true",
                        help="End a walk whose last capture ended by disconnect with Stop instead.")
    parser.add_argument("--settle-timeout-min", type=float, default=90.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--sample-seconds", type=float, default=3.0)
    parser.add_argument("--after-stop", choices=("stay", "disconnect"), default="stay",
                        help="stay: the socket stays open after Stop, as a phone on the World "
                             "Builder screen; disconnect: close it, as a phone that leaves.")
    parser.add_argument("--no-subscribe", action="store_true",
                        help="Do not subscribe to the World Builder status channel.")
    parser.add_argument("--no-phone-fetches", action="store_true",
                        help="Do not fetch the geometry manifest/segments after status changes.")
    parser.add_argument("--no-session-start", action="store_true",
                        help="Do not POST /cartridges/world_builder/session/start first.")
    parser.add_argument("--no-live-guard", action="store_true",
                        help="Do not watch :8000 for a live walk. For tests only: refused unless "
                             "--not-a-proof-run is given too, and the run is marked NOT-PROOF.")
    parser.add_argument("--not-a-proof-run", action="store_true",
                        help="Declare this run NOT proof (a smoke, a test). Its record and report say "
                             "NOT-PROOF and --compare never counts it. Required with --no-live-guard.")
    parser.add_argument("--label", default=None)
    parser.add_argument("--live-timeline", action="store_true",
                        help="Record per-frame/keyframe live timing, phone messages, and published poses.")
    parser.add_argument("--capture-status-geometry", action="store_true",
                        help="Preserve every received World Builder status envelope and fetched geometry body.")


def refuse_unguarded_proof(args) -> None:
    """`--no-live-guard` only in a run declared NOT proof (review C24 HIGH-3):
    a run nobody watched :8000 for must never become a proof run."""
    if getattr(args, "no_live_guard", False) and not getattr(args, "not_a_proof_run", False):
        raise SystemExit("refused: --no-live-guard turns off the :8000 guard, so this run could never be "
                         "proof; give --not-a-proof-run as well to run it anyway (it is marked NOT-PROOF)")


def path_inside(path, parent) -> bool:
    """Whether `path` resolves inside `parent` (junctions resolved; a path
    test, not a string prefix)."""
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def refuse_inside_live_store(path, what: str) -> None:
    """No writing entrypoint writes inside the live store (review C24 MED-6:
    the runner refused it; the client and the report did not)."""
    if path is not None and path_inside(path, LIVE_DATA):
        raise SystemExit(f"refused: {what} {path} is inside the live store {LIVE_DATA}")


def options_from_args(args, *, port, out, world_root=None, tower_pid=None, on_abort=None,
                      calibration_root=None, calibration_expected=None, on_guard_armed=None) -> ReplayOptions:
    return ReplayOptions(
        port=port, captures=list(args.capture), capture_root=Path(args.capture_root),
        out=Path(out), world_root=None if world_root is None else Path(world_root),
        tower_pid=tower_pid, follow_chain=not args.no_follow_chain, speed=args.speed,
        first_seconds=args.first_seconds, end_with_stop=args.end_with_stop,
        settle_timeout_min=args.settle_timeout_min, poll_seconds=args.poll_seconds,
        sample_seconds=args.sample_seconds, after_stop=args.after_stop,
        subscribe=not args.no_subscribe, phone_fetches=not args.no_phone_fetches,
        start_session=not args.no_session_start, live_guard=not args.no_live_guard,
        not_a_proof_run=bool(getattr(args, "not_a_proof_run", False)),
        on_abort=on_abort, on_guard_armed=on_guard_armed, label=args.label,
        calibration_root=calibration_root, calibration_expected=calibration_expected,
        live_timeline=bool(getattr(args, "live_timeline", False)),
        capture_status_geometry=bool(getattr(args, "capture_status_geometry", False)),
        tower_log=getattr(args, "tower_log", None),
    )


def refuse_non_empty_out(out: Path) -> None:
    """An `--out` must be new or empty (review C22 L4): the logs are appended
    to, and a reused directory mixes two runs' records."""
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"refused: --out {out} is not empty; every run needs a fresh --out")


def exit_code(record: dict) -> int:
    outcome = record.get("outcome")
    if isinstance(outcome, str) and outcome.startswith("refused-"):
        return EXIT_ABORTED
    return {
        "settled": EXIT_SETTLED,
        "not-settled": EXIT_NOT_SETTLED,
        "aborted": EXIT_ABORTED,
    }.get(outcome, EXIT_ERROR)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a stored capture into a RUNNING test Tower's /ws at the recorded "
                    "pace, follow it to settled, and write client.json + the report. "
                    "world_live_replay_run.py starts and stops the Tower for you.")
    add_replay_arguments(parser)
    parser.add_argument("--port", type=int, required=True, help=f"The test Tower's port (>= {MIN_TEST_PORT}).")
    parser.add_argument("--out", type=artifact_root_arg, required=True,
                        help="A NEW or empty directory for client.json, samples.csv and the report.")
    parser.add_argument("--world-root", type=Path, default=None,
                        help="The test Tower's TOWER_WORLD_ROOT (read), for session.json and the report.")
    parser.add_argument("--tower-pid", type=int, required=True,
                        help="The test Tower's pid: the replay refuses a port another process serves, "
                             "and samples this pid's process tree.")
    parser.add_argument("--tower-log", type=Path, default=None, help="The test Tower's stderr log.")
    parser.add_argument("--tower-out-log", type=Path, default=None, help="The test Tower's stdout log.")
    args = parser.parse_args(argv)
    refuse_unguarded_proof(args)
    check_target_port(args.port)
    refuse_inside_live_store(args.out, "--out")
    refuse_non_empty_out(args.out)
    options = options_from_args(args, port=args.port, out=args.out, world_root=args.world_root,
                                tower_pid=args.tower_pid)
    record = asyncio.run(run_replay(options))
    if args.tower_log is not None and record.get("tower_captures"):
        from scripts.world_live_replay_report import build_report, write_report

        report = build_report(tower_log=args.tower_log, tower_out_log=args.tower_out_log,
                              world_root=args.world_root, client=record,
                              samples=Path(args.out) / "samples.csv", label=args.label,
                              run_dir=Path(args.out))
        write_report(Path(args.out), report)
    return exit_code(record)


if __name__ == "__main__":
    raise SystemExit(main())
