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
  * at the RECORDED pace: each frame leaves at its recorded tower-receipt
    offset from its capture's start (`frames.jsonl` `received_at`), divided
    by `--speed` (default 1.0), against an absolute clock, so lateness never
    accumulates;
  * with the World Builder cartridge started the way the phone starts it
    (`POST /cartridges/world_builder/session/start` before `stream_start`),
    subscribed to its status channel the way the phone is, and fetching the
    geometry manifest and the changed segments whenever a push moves
    `geometry.revision`, the way the phone's `WorldBuilderClient` does;
  * across a multi-capture walk as the phone produced it. A capture that
    ended by `disconnect` is ended by closing the socket, and the next one
    begins on a new socket at its recorded offset. That offset is inside the
    90 s resume grace, so the Tower keeps one session.

Then it follows the Tower from Stop until it settles: finalization, room
surface, room appearance, worker exit, and the finisher chore's areas. It
samples CPU and GPU every few seconds and writes a report in the shape of
`PROFILE-walk5.md` (`world_live_replay_report.py`).

SAFETY
  * It never talks to :8000 except to read `/health`, and it refuses a
    target port below 8031.
  * Before streaming, and every few seconds while it runs, it asks :8000's
    `/health` whether a real walk is being recorded. If one is, it does not
    start; or it stops streaming at once and returns `aborted`, so its runner
    can stop the test Tower and free the GPU.
  * It reads the stored capture and writes nothing beside it.

WHAT IT CANNOT SEE is listed in RUN\\experiments\\C22-REPLAY\\README.md. In
short: the phone's own experience (rendering, battery, the radio link, the
phone's send window dropping frames under backpressure), and anything that
depends on the store holding earlier worlds (a fresh root has none).

Usually driven by `world_live_replay_run.py`, which starts a test Tower from
a named code tree, runs this, stops the Tower and writes the report.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import csv
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402

TOWER_ROOT = Path(__file__).resolve().parents[1]
# The stored captures live in the canonical checkout's store; a worktree has
# no `data` of its own. Read only, always.
CANONICAL_CAPTURE_ROOT = Path(r"C:\Users\tvllo\Projects\Glasses\tower\data\captures")
DEFAULT_CAPTURE_ROOT = (CANONICAL_CAPTURE_ROOT if CANONICAL_CAPTURE_ROOT.is_dir()
                        else TOWER_ROOT / "data" / "captures")

# Tristan's own Tower. Read-only (/health), and only to find out whether a
# real walk is being recorded.
LIVE_TOWER_PORT = 8000
LIVE_HEALTH_URL = f"http://127.0.0.1:{LIVE_TOWER_PORT}/health"
# A test Tower lives at or above this port (the C22 brief), so a typo can
# never point a replay at the live Tower or at another lane's.
MIN_TEST_PORT = 8031

WORLD_BUILDER = "world_builder"
WORLD_BUILDER_RESULT_TYPE = "status"

EXIT_SETTLED = 0
EXIT_ERROR = 1
EXIT_NOT_SETTLED = 2
EXIT_ABORTED = 3

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
    manifest = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    lines = (directory / "frames.jsonl").read_text(encoding="utf-8").splitlines()
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
    """count / mean / p50 / p95 / max of a list of numbers, rounded."""
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


def live_walk_recording(url: str = LIVE_HEALTH_URL, timeout: float = 3.0):
    """True when the live Tower is recording a walk, False when it is not,
    None when it does not answer (a Tower that is down records nothing)."""
    status, doc = http_json("GET", url, timeout=timeout)
    if status != 200 or not isinstance(doc, dict):
        return None
    capture = doc.get("capture") or {}
    return bool(isinstance(capture, dict) and capture.get("recording"))


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


class PhoneView:
    """The World Builder status pushes, reduced to their state transitions."""

    def __init__(self, clock=time.time):
        self._clock = clock
        self.pushes = 0
        self.last: dict = {}
        self.transitions: list = []
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
        coordinates = geometry_coordinates(payload)
        if coordinates is not None:
            self.target = coordinates[:2]
        return coordinates


class GeometryMirror:
    """Fetches what the phone fetches when the geometry revision moves: the
    manifest, then every segment whose (content, placement) key it lacks
    (`WorldBuilderClient.swift`, `geometryDidChange`). Coalesced, one fetch at
    a time, and a newer revision supersedes a fetch in flight, as the phone's
    does."""

    def __init__(self, base_url: str, enabled: bool = True):
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self._wanted = asyncio.Event()
        self._target = None
        self._last_revision = None
        self._cache: set = set()
        self._cache_target = None
        self.manifests = 0
        self.segments = 0
        self.bytes = 0
        self.errors = 0
        self.manifest_ms: list = []
        self.segment_ms: list = []

    def request(self, coordinates) -> None:
        """A status push named (world, session, geometry revision)."""
        if not self.enabled or coordinates is None or coordinates == self._last_revision:
            return
        self._last_revision = coordinates
        self._target = coordinates
        self._wanted.set()

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            waiter = asyncio.create_task(self._wanted.wait())
            stopper = asyncio.create_task(stop.wait())
            await asyncio.wait({waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            stopper.cancel()
            if stop.is_set():
                return
            self._wanted.clear()
            with contextlib.suppress(Exception):
                await self._fetch(self._target)

    async def _fetch(self, target) -> None:
        world_id, session_id = target[0], target[1]
        if (world_id, session_id) != self._cache_target:
            self._cache, self._cache_target = set(), (world_id, session_id)
        started = time.perf_counter()
        status, manifest = await asyncio.to_thread(
            http_json, "GET",
            f"{self.base_url}/worlds/{world_id}/geometry/manifest?session_id={session_id}", 30.0)
        self.manifest_ms.append((time.perf_counter() - started) * 1000)
        if status != 200 or not isinstance(manifest, dict):
            self.errors += status != 404
            return
        self.manifests += 1
        keys = {}
        for segment in manifest.get("segments") or []:
            key = (segment.get("content_hash"), segment.get("placement_hash"))
            keys[key] = segment.get("segment_index")
        for key, index in keys.items():
            if key in self._cache or index is None:
                continue
            if self._wanted.is_set():
                break  # superseded: the phone stops fetching a dead manifest
            started = time.perf_counter()
            status, size = await asyncio.to_thread(
                http_bytes,
                f"{self.base_url}/worlds/{world_id}/geometry/segment/{index}?session_id={session_id}")
            self.segment_ms.append((time.perf_counter() - started) * 1000)
            if status == 200:
                self.segments += 1
                self.bytes += size
                self._cache.add(key)
            else:
                self.errors += 1
        self._cache &= set(keys)

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "manifests": self.manifests,
            "segments": self.segments,
            "bytes": self.bytes,
            "errors": self.errors,
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

    def __init__(self, uri: str, stats: StreamStats, phone: PhoneView, mirror: GeometryMirror):
        self.uri = uri
        self.stats = stats
        self.phone = phone
        self.mirror = mirror
        self.ws = None
        self._reader = None
        self._waiters: dict = {}
        self._sent_at: dict = {}
        self.subscription_id = None
        self.closed_by_us = False

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
                    self._dispatch(message)
        except Exception:  # noqa: BLE001 -- a closed socket ends the reader
            pass
        finally:
            if not self.closed_by_us:
                self.stats.unexpected_closes += 1
            for futures in self._waiters.values():
                for future in futures:
                    if not future.done():
                        future.set_result(None)

    def _dispatch(self, message: dict) -> None:
        kind = message.get("type")
        now = time.perf_counter()
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
                self.mirror.request(self.phone.update(message))
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
            handle = open(self.csv_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
            if handle.tell() == 0:
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
    live_guard_url: str = LIVE_HEALTH_URL
    live_guard_every: float = 10.0
    label: str | None = None


def _log(out: Path, text: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
    print(line, flush=True)
    with open(out / "client.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


async def _guard_live(options: ReplayOptions, abort: asyncio.Event, record: dict, stop: asyncio.Event):
    while not stop.is_set() and not abort.is_set():
        recording = await asyncio.to_thread(live_walk_recording, options.live_guard_url)
        if recording:
            record["aborted"] = {"t": round(time.time(), 3),
                                 "reason": "a live walk is being recorded on :8000"}
            _log(options.out, "ABORT: :8000 is recording a live walk; stopping the replay now")
            abort.set()
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), options.live_guard_every)


async def _wait_or_abort(abort: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(abort.wait(), seconds)


async def run_replay(options: ReplayOptions) -> dict:
    """Stream the walk, follow the settle, and return everything the client saw.

    Also writes `client.json` (and `samples.csv`) into `options.out`.
    """
    check_target_port(options.port)
    out = Path(options.out)
    out.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{options.port}"
    record: dict = {
        "tool": "world_live_replay",
        "label": options.label,
        "port": options.port,
        "captures_replayed": list(options.captures),
        "speed": options.speed,
        "first_seconds": options.first_seconds,
        "after_stop": options.after_stop,
        "started_at": round(time.time(), 3),
        "outcome": None,
    }

    if options.live_guard:
        live = live_walk_recording(options.live_guard_url)
        record["live_tower_at_start"] = {True: "recording", False: "idle", None: "no answer"}[live]
        if live:
            record["outcome"] = "refused-live-walk"
            _log(out, "REFUSED: :8000 is recording a live walk; nothing was streamed")
            _write_client(out, record)
            return record

    walk = load_walk(options.capture_root, options.captures, follow_chain=options.follow_chain)
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

    status, health = await asyncio.to_thread(http_json, "GET", f"{base}/health", 10.0)
    if status != 200:
        record["outcome"] = "no-test-tower"
        _log(out, f"no test Tower answers {base}/health")
        _write_client(out, record)
        return record
    record["tower_health_at_start"] = settle_snapshot(health, None)

    stats = StreamStats()
    phone = PhoneView()
    mirror = GeometryMirror(base, enabled=options.phone_fetches)
    abort = asyncio.Event()
    stop_background = asyncio.Event()
    background = [asyncio.create_task(mirror.run(stop_background))]
    if options.live_guard:
        background.append(asyncio.create_task(_guard_live(options, abort, record, stop_background)))
    sampler = ResourceSampler(options.tower_pid, options.sample_seconds, out / "samples.csv")
    sampler.start()
    uri = f"ws://127.0.0.1:{options.port}/ws"
    events: list = []
    tower_captures: list = []

    def note(kind: str, **extra) -> None:
        events.append({"t": round(time.time(), 3), "kind": kind, **extra})

    socket = None
    try:
        with fine_timer():
            socket = TowerSocket(uri, stats, phone, mirror)
            await socket.open()
            record["handshake"] = await socket.handshake(options.subscribe)
            note("connected", handshake=record["handshake"])
            if options.start_session:
                status, body = await asyncio.to_thread(
                    http_json, "POST", f"{base}/cartridges/{WORLD_BUILDER}/session/start", 15.0)
                record["session_start"] = {"status": status, "state": (body or {}).get("state"),
                                           "session_id": (body or {}).get("session_id")}
                note("session_start", status=status)
                _log(out, f"POST session/start -> {status} {record['session_start']}")
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
                    jpeg = (walk[step.capture].directory / step.frame.relpath).read_bytes()
                    text, size = encode(frame_message(step.frame, jpeg)), len(jpeg)
                late = await pacer.wait_until(step.at, abort)
                if abort.is_set():
                    break
                if step.kind == "frame":
                    stats.lateness_s.append(late)
                    await socket.send_frame(step.frame, text, size)
                    if step.at >= next_progress:
                        next_progress += progress_every
                        _log(out, f"t={step.at:6.1f}s sent {stats.frames_sent} frames, "
                                  f"{stats.frame_results} results, {sum(stats.frame_errors.values())} errors")
                elif step.kind == "connect":
                    socket = TowerSocket(uri, stats, phone, mirror)
                    await socket.open()
                    note("reconnected", handshake=await socket.handshake(options.subscribe))
                elif step.kind == "stream_start":
                    await socket.send({"type": "stream_start"})
                    note("stream_start", capture=step.capture, late_s=round(late, 4))
                    # Off the send path: the Tower is spawning the builder
                    # right now, and an inline /health held the first frames
                    # back by ~0.5 s in the first smoke run.
                    background.append(asyncio.create_task(
                        _learn_capture(base, tower_captures, len(tower_captures))))
                elif step.kind == "stream_stop":
                    await socket.send({"type": "stream_stop"})
                    note("stream_stop", capture=step.capture, late_s=round(late, 4))
                elif step.kind == "disconnect":
                    await socket.close()
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
            settle = await _follow_settle(options, base, tower_captures, abort, out)
            record["settle"] = settle
            if abort.is_set():
                record["outcome"] = "aborted"
            else:
                record["outcome"] = "settled" if settle.get("settled") else "not-settled"
    finally:
        stop_background.set()
        for task in background:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, 35)
        if socket is not None:
            await socket.close()
        sampler.stop()
        sampler.join(timeout=20)
        record["events"] = events
        record["stream"] = stats.summary()
        record["phone_view"] = {"pushes": phone.pushes, "target": phone.target,
                                "transitions": phone.transitions}
        record["phone_fetches"] = mirror.summary()
        record["tower_captures"] = tower_captures
        record["ended_at"] = round(time.time(), 3)
        _write_client(out, record)
    return record


async def _learn_capture(base: str, captures: list, index: int) -> None:
    """The capture id the Tower minted for the stream just opened (/health)."""
    for _ in range(20):
        await asyncio.sleep(0.5)
        status, health = await asyncio.to_thread(http_json, "GET", f"{base}/health", 10.0)
        cid = ((health or {}).get("capture") or {}).get("capture_id") if status == 200 else None
        if cid and cid not in captures and len(captures) == index:
            captures.append(cid)
            return
        if cid in captures:
            return


async def _follow_settle(options: ReplayOptions, base: str, captures: list,
                         abort: asyncio.Event, out: Path) -> dict:
    started = time.time()
    deadline = started + options.settle_timeout_min * 60.0
    transitions: list = []
    last: dict = {}
    judge = None
    session_path = None
    settled_at = None
    while time.time() < deadline and not abort.is_set():
        status, health = await asyncio.to_thread(http_json, "GET", f"{base}/health", 10.0)
        if options.world_root is not None and captures and session_path is None:
            session_path = await asyncio.to_thread(find_session_file, options.world_root, captures)
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
    }


def _write_client(out: Path, record: dict) -> None:
    tmp = out / "client.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, out / "client.json")


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
                        help="Do not watch :8000 for a live walk. For tests only.")
    parser.add_argument("--label", default=None)


def options_from_args(args, *, port, out, world_root=None, tower_pid=None) -> ReplayOptions:
    return ReplayOptions(
        port=port, captures=list(args.capture), capture_root=Path(args.capture_root),
        out=Path(out), world_root=None if world_root is None else Path(world_root),
        tower_pid=tower_pid, follow_chain=not args.no_follow_chain, speed=args.speed,
        first_seconds=args.first_seconds, end_with_stop=args.end_with_stop,
        settle_timeout_min=args.settle_timeout_min, poll_seconds=args.poll_seconds,
        sample_seconds=args.sample_seconds, after_stop=args.after_stop,
        subscribe=not args.no_subscribe, phone_fetches=not args.no_phone_fetches,
        start_session=not args.no_session_start, live_guard=not args.no_live_guard,
        label=args.label,
    )


def exit_code(record: dict) -> int:
    return {
        "settled": EXIT_SETTLED,
        "not-settled": EXIT_NOT_SETTLED,
        "aborted": EXIT_ABORTED,
        "refused-live-walk": EXIT_ABORTED,
    }.get(record.get("outcome"), EXIT_ERROR)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a stored capture into a RUNNING test Tower's /ws at the recorded "
                    "pace, follow it to settled, and write client.json + the report. "
                    "world_live_replay_run.py starts and stops the Tower for you.")
    add_replay_arguments(parser)
    parser.add_argument("--port", type=int, required=True, help=f"The test Tower's port (>= {MIN_TEST_PORT}).")
    parser.add_argument("--out", type=artifact_root_arg, required=True,
                        help="Where client.json, samples.csv and the report go.")
    parser.add_argument("--world-root", type=Path, default=None,
                        help="The test Tower's TOWER_WORLD_ROOT (read), for session.json and the report.")
    parser.add_argument("--tower-pid", type=int, default=None,
                        help="The test Tower's pid, for CPU sampling of its process tree.")
    parser.add_argument("--tower-log", type=Path, default=None, help="The test Tower's stderr log.")
    parser.add_argument("--tower-out-log", type=Path, default=None, help="The test Tower's stdout log.")
    args = parser.parse_args(argv)
    check_target_port(args.port)
    options = options_from_args(args, port=args.port, out=args.out, world_root=args.world_root,
                                tower_pid=args.tower_pid)
    record = asyncio.run(run_replay(options))
    if args.tower_log is not None and record.get("tower_captures"):
        from scripts.world_live_replay_report import build_report, write_report

        report = build_report(tower_log=args.tower_log, tower_out_log=args.tower_out_log,
                              world_root=args.world_root, client=record,
                              samples=Path(args.out) / "samples.csv", label=args.label)
        write_report(Path(args.out), report)
    return exit_code(record)


if __name__ == "__main__":
    raise SystemExit(main())
