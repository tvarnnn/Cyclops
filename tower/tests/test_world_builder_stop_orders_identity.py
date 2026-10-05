"""NO BIG BANG for the closed-capture drain: a builder caught up at the Stop
builds byte for byte what 44fbd13 built.

WHY THIS FILE EXISTS. The drain (`claude/drain-guard`) changes how a World
Builder session ends after a soft stop, and it ships with no switch: the
manager exempted it as a correctness fix (manager 163 §1) on condition that
this identity test stays in the suite PERMANENTLY, for every stop order in
which the builder was caught up, against 44fbd13 -- the base before the
drain -- and that it compares FILE BYTES and EVENT PAYLOADS on VALID frames,
with only timestamps masked (manager 166 §1, answering the Codex review's L5:
the first version compared keys, counts and kinds on frames that were all
rejected).

WHAT IS COMPARED, per order, against `golden/world_builder_stop_orders_
44fbd13.json` (recorded by running this very file on 44fbd13):

- every file the session wrote under `--root` -- `world.json`,
  `session.json`, `events.jsonl`, `keyframes.jsonl`, `edges.jsonl`, the
  derived `manifest.json` / `poses.json` / `points.json` / `support.json`,
  every keyframe image, the intrinsics -- as the SHA-256 of its bytes with
  each wall-clock stamp (`"at"` and every `"*_at"` key except `received_at`,
  which these journals fix) replaced by a constant;
- every event, kind and payload, in order, with only `"at"` removed;
- the session's JSON report (`--format json`) with only its two durations
  masked;
- every line the builder logs at INFO or above, with durations masked.

World and session ids are made deterministic (`engine.new_id`), so nothing
else is masked. The frames are the synthetic renderer's 24 decodable JPEGs:
the session accepts keyframes, rebuilds and triangulates points, so a change
to what is BUILT shows up in the bytes.

TO RECORD THE GOLDEN AGAIN -- only if the world output legitimately changes
on the base branch -- run this file on that base with
`WB_STOP_ORDERS_RECORD=<new golden path>`, review the diff, and rename the
golden after the base SHA. Never record it from a tree whose stop handling
is under review.
"""

import contextlib
import io
import itertools
import json
import logging
import os
import re
import time
from pathlib import Path

import pytest

import scripts.world_build_session as builder_script
import tower.world_builder.engine as engine_module
from scripts.world_build_session import StopRequest
from tower.world_builder.intrinsics_store import IntrinsicsStore

FRAMES = 24
OPEN_SOFT_AT = 15
GOLDEN = Path(__file__).parent / "golden" / "world_builder_stop_orders_44fbd13.json"
RECORD_ENV = "WB_STOP_ORDERS_RECORD"
CAPTURE_NAME = "c" * 32

# Every caught-up stop order, by name. In each the builder has observed every
# frame recorded by the Stop, so the guard has nothing to count and the drain
# nothing to read, and the session must be exactly 44fbd13's.
CAUGHT_UP_ORDERS = [
    # The capture closed normally before the builder attached; no stop at all.
    "closed-before-start-no-stop",
    # The wearer's Stop closes the capture, then the soft stop (iOS today).
    "close-then-soft",
    # Soft stop decided while the capture is open; the close lands after the loop.
    "soft-while-open-then-close",
    # Soft stop while open; 1 frame recorded after the Stop; then the close.
    "soft-while-open-1-after-stop-then-close",
    # Soft stop while open; 3 frames recorded after the Stop; then the close.
    "soft-while-open-3-after-stop-then-close",
    # Soft stop while open; 36 frames recorded after the Stop; then the close.
    "soft-while-open-36-after-stop-then-close",
    # The capture closes normally, then the Tower shuts down (hard stop).
    "close-then-hard",
    # The link drops (capture closes `disconnect`), then the deferred soft stop.
    "disconnect-then-soft",
    # Oct 2 order, caught up: close, +11 s disconnect soft stop, +0.9 s reconnect.
    "oct2-close-then-disconnect-soft-then-reconnect",
    # Control, NOT caught up: soft stop on an open capture -> `interrupted`.
    "open-soft",
]

# Codex H1 / re-review MED-1, kept apart because 8ede341 and 6124de8 fail
# them by design -- they are the defect. Frames recorded after the Stop AND
# the close land before the builder's next stop check, and the drain read the
# post-Stop frames into the world; 44fbd13 did not.
POST_STOP_ORDERS = [
    # Caught up at the soft stop on an open capture; 3 after it, then the close.
    "soft-while-open-3-after-stop-close-before-next-check",
    # Soft stop before the first frame; 3 frames after it, then the close.
    "soft-before-the-first-frame-then-3-after-stop-and-close",
]
POST_STOP_ORDER, BEFORE_FIRST_FRAME_ORDER = POST_STOP_ORDERS

_STAMP = re.compile(rb'"(at|(?!received_at")[A-Za-z0-9_]*_at)"\s*:\s*-?[0-9][0-9.eE+-]*')
_DURATION = re.compile(r"\d+(\.\d+)?(e-?\d+)? ?(ms|s)\b")


@pytest.fixture(scope="module")
def rendered():
    frames, intrinsics = builder_script.synthetic_frames(FRAMES, 480, 360)
    return [frame.payload for frame in frames], intrinsics


def _row(index, received_at, received_monotonic, payload):
    seq = index + 1
    return json.dumps(
        {
            "schema_version": 1,
            "source_seq": seq,
            "wire_seq": seq,
            "tx_seq": seq,
            "received_at": received_at,
            # What the recorder writes since the H3 fix; 44fbd13 ignores it.
            "received_monotonic": received_monotonic,
            "time_basis": "tower-receipt",
            "relpath": f"frames/{seq:08d}.jpg",
            "byte_count": len(payload),
            "width": 480,
            "height": 360,
        }
    ) + "\n"


def _write_manifest(capture, end_reason, frames_written):
    (capture / "capture.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "capture_id": capture.name,
                "started_at": 1000.0,
                "ended_at": None if end_reason is None else 2000.0,
                "end_reason": end_reason,
                "time_basis": "tower-receipt",
                "frames_written": frames_written,
                "continues_capture": None,
            }
        ),
        encoding="utf-8",
    )


def _run_order(order, rendered, workdir, monkeypatch, caplog):
    payloads, intrinsics = rendered
    capture = workdir / "captures" / CAPTURE_NAME
    (capture / "frames").mkdir(parents=True)
    journal = capture / "frames.jsonl"
    # Every frame of the walk was received before the Stop, in true time and
    # on both clocks: wall stamps fixed (they reach `keyframes.jsonl`),
    # monotonic stamps a few seconds before now.
    mono0 = time.monotonic() - FRAMES / 12 - 2.0
    walk = [] if order == BEFORE_FIRST_FRAME_ORDER else payloads
    for index, payload in enumerate(walk):
        (capture / "frames" / f"{index + 1:08d}.jpg").write_bytes(payload)
    if walk:
        # The recorder writes no journal before its first frame.
        journal.write_text(
            "".join(
                _row(index, 1000.0 + index / 12, mono0 + index / 12, payload)
                for index, payload in enumerate(walk)
            ),
            encoding="utf-8",
        )
    written = {"n": len(walk)}
    _write_manifest(
        capture, "stop" if order == "closed-before-start-no-stop" else None, len(walk)
    )

    def record_after_the_stop(stop, count):
        """`count` frames the camera records after the wearer's Stop.

        Stamped after the request on both clocks, as the recorder would. The
        base has neither attribute; it never reads these stamps either.
        """
        at = getattr(stop, "soft_requested_at", None) or time.time()
        mono = getattr(stop, "soft_requested_monotonic", None) or time.monotonic()
        rows = []
        for k in range(count):
            index = written["n"]
            payload = payloads[index % FRAMES]
            (capture / "frames" / f"{index + 1:08d}.jpg").write_bytes(payload)
            rows.append(_row(index, at + (k + 1) / 12, mono + (k + 1) / 12, payload))
            written["n"] += 1
        with journal.open("a", encoding="utf-8") as out:
            out.write("".join(rows))

    holder = {}

    def install(self, **_kwargs):
        holder["stop"] = self
        if order == BEFORE_FIRST_FRAME_ORDER:
            # The wearer leaves before the first frame; the camera records 3
            # and the capture closes before the follower first looks.
            self.request(StopRequest.SOFT, "stdin-closed")
            record_after_the_stop(self, 3)
            _write_manifest(capture, "stop", written["n"])

    monkeypatch.setattr(builder_script.StopRequest, "install", install)
    original_observe = builder_script.WorldBuilderEngine.observe
    count = {"n": 0}

    def observe(self, *args, **kwargs):
        outcome = original_observe(self, *args, **kwargs)
        count["n"] += 1
        n, stop = count["n"], holder["stop"]
        if order == "open-soft" and n == OPEN_SOFT_AT:
            stop.request(StopRequest.SOFT, "stdin-closed")
        if n != FRAMES:
            return outcome
        if order == "close-then-soft":
            _write_manifest(capture, "stop", written["n"])
            stop.request(StopRequest.SOFT, "stdin-closed")
        elif order.startswith("soft-while-open-"):
            stop.request(StopRequest.SOFT, "stdin-closed")
            if order == POST_STOP_ORDER:
                # Recorded after the Stop, and the close, before the
                # follower's next look at the journal.
                record_after_the_stop(stop, 3)
                _write_manifest(capture, "stop", written["n"])
        elif order == "close-then-hard":
            _write_manifest(capture, "stop", written["n"])
            stop.request(StopRequest.HARD, "SIGBREAK")
        elif order == "disconnect-then-soft":
            _write_manifest(capture, "disconnect", written["n"])
            stop.request(StopRequest.SOFT, "stdin-closed")
        elif order.startswith("oct2-"):
            # The wearer's Stop closes the capture. The builder is caught up,
            # so it reads to the end and leaves the observe loop on its own.
            _write_manifest(capture, "stop", written["n"])
        return outcome

    monkeypatch.setattr(builder_script.WorldBuilderEngine, "observe", observe)
    original_bounded = builder_script.StopRequest.bounded

    def bounded(self, frames, **kwargs):
        yield from original_bounded(self, frames, **kwargs)
        if order.startswith("soft-while-open-") and order != POST_STOP_ORDER:
            # The camera went on recording; the close lands before the
            # post-loop read.
            after = 0 if order == "soft-while-open-then-close" else int(order.split("-")[3])
            record_after_the_stop(self, after)
            _write_manifest(capture, "stop", written["n"])
        elif order.startswith("oct2-"):
            # +11 s the last client disconnects: `ws.py` stops the cartridge
            # session and the supervisor closes this builder's stdin -- after
            # it finished reading. +0.9 s the phone reconnects and re-sends
            # `session/start`; nothing is recording, so nothing reaches this
            # builder.
            self.request(StopRequest.SOFT, "stdin-closed")

    monkeypatch.setattr(builder_script.StopRequest, "bounded", bounded)
    ids = itertools.count(1)
    monkeypatch.setattr(engine_module, "new_id", lambda: f"{next(ids):032x}")

    root = workdir / "worlds"
    IntrinsicsStore(root).save(intrinsics)
    stdout = io.StringIO()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="tower"), contextlib.redirect_stdout(stdout):
        exit_code = builder_script.main(
            [
                "--follow-capture", str(capture),
                "--root", str(root),
                "--poll-seconds", "0.01",
                "--max-idle-polls", "50",
                "--rebuild-every", "4",
                "--format", "json",
            ]
        )
    return exit_code, json.loads(stdout.getvalue()), root, workdir


def _mask_bytes(data, workdir):
    """Wall-clock stamps, and the two paths that name the machine and tree."""
    data = _STAMP.sub(lambda m: b'"' + m.group(1) + b'": "<T>"', data)
    for path, label in (
        (str(workdir), b"<WD>"),
        (str(workdir.resolve()), b"<WD>"),
        (str(Path.cwd()), b"<CWD>"),
    ):
        for form in {path, path.replace("\\", "/"), json.dumps(path)[1:-1]}:
            data = data.replace(form.encode("utf-8"), label)
    return data


def _observed(exit_code, report, root, workdir, caplog):
    import hashlib

    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            masked = _mask_bytes(path.read_bytes(), workdir)
            files[path.relative_to(root).as_posix()] = hashlib.sha256(masked).hexdigest()
    (events_path,) = list(root.glob("worlds/*/sessions/*/events.jsonl"))
    events = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        event.pop("at", None)
        events.append(event)
    masked_report = dict(report)
    for key in ("observe_ms_per_frame", "build_seconds"):
        if key in masked_report:
            masked_report[key] = "<T>"
    log = []
    for record in caplog.records:
        if not record.name.startswith(("tower.world_build_session", "tower.capture")):
            continue
        line = _mask_bytes(record.getMessage().encode("utf-8"), workdir).decode("utf-8")
        log.append(f"{record.levelname} {record.name} {_DURATION.sub('<T>', line)}")
    return {
        "exit": exit_code,
        "report": masked_report,
        "events": events,
        "files": files,
        "log": log,
    }


def _check(order, observed):
    record_to = os.environ.get(RECORD_ENV)
    if record_to:
        path = Path(record_to)
        golden = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"frames": FRAMES, "orders": {}}
        )
        golden["orders"][order] = observed
        path.write_text(
            json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        pytest.skip(f"recorded {order} into {path}")
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))["orders"][order]
    # Most specific first, so a failure names what differs.
    assert observed["files"] == expected["files"]
    assert observed["events"] == expected["events"]
    assert observed["report"] == expected["report"]
    assert observed["log"] == expected["log"]
    assert observed["exit"] == expected["exit"]


def _assert_valid_walk(observed, frames_observed, end_reason):
    """The frames really were built: keyframes, rebuilds, points, none malformed."""
    report = observed["report"]
    assert report["frames_observed"] == frames_observed
    assert report["end_reason"] == end_reason
    assert report["finalization"] == "complete"
    assert report["keyframes_accepted"] >= 4
    assert report["rebuilds"] >= 1
    assert report["points"] > 0
    assert "malformed_frame" not in report["rejected_by_reason"]
    assert any(event["kind"] == "keyframe_accepted" for event in observed["events"])
    assert any(name.endswith(".jpg") for name in observed["files"])


@pytest.mark.parametrize("order", CAUGHT_UP_ORDERS)
def test_caught_up_stop_orders_build_what_44fbd13_built(
    order, rendered, tmp_path, monkeypatch, caplog
):
    observed = _observed(*_run_order(order, rendered, tmp_path, monkeypatch, caplog), caplog)

    if order == "open-soft":
        _assert_valid_walk(observed, OPEN_SOFT_AT, "interrupted")
    else:
        _assert_valid_walk(observed, FRAMES, "stop")
    _check(order, observed)


@pytest.mark.parametrize("order", POST_STOP_ORDERS)
def test_frames_recorded_after_the_stop_are_not_built(
    order, rendered, tmp_path, monkeypatch, caplog
):
    observed = _observed(*_run_order(order, rendered, tmp_path, monkeypatch, caplog), caplog)

    if order == BEFORE_FIRST_FRAME_ORDER:
        assert observed["report"]["frames_observed"] == 0
        assert observed["report"]["end_reason"] == "stop"
    else:
        _assert_valid_walk(observed, FRAMES, "stop")
    _check(order, observed)
