#!/usr/bin/env python
"""Drive a REAL running Tower through the CV Lab workflow, over the wire.

The test suite exercises this with `TestClient`, which is an in-process
ASGI shim: no socket, no uvicorn, no event loop of its own. That is the
right tool for a contract test and the wrong one for the question "does
this work when a Tower is actually running", which is the question a
person asks before walking around a room with glasses on.

So this is the pre-flight for the physical test. It connects to a Tower
you started yourself, and walks the workflow the product promises:

    browse experiments -> select one -> start -> live results
        -> pause -> resume -> stop -> read the run summary

It sends its own frames rather than waiting for glasses, so a FAILURE
here is a Tower problem and never a phone problem. If this passes and the
phone still shows nothing, the fault is between the phone and the Tower.

    # in one terminal
    .venv\\Scripts\\python.exe -m uvicorn tower.main:app --host 0.0.0.0 --port 8000

    # in another
    .venv\\Scripts\\python.exe scripts/cv_lab_smoke.py
    .venv\\Scripts\\python.exe scripts/cv_lab_smoke.py --host 100.110.156.55
    .venv\\Scripts\\python.exe scripts/cv_lab_smoke.py --experiment depth --frames 5

Exit code 0 means every step did what the contract says it does.
"""

import argparse
import asyncio
import base64
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WIDTH, HEIGHT = 640, 360


def _jpeg(seed: int) -> str:
    """A textured frame. Flat grey gives every experiment nothing to find,
    which makes a working Tower look broken."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=60)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def that(self, condition, description: str) -> None:
        mark = "ok  " if condition else "FAIL"
        print(f"  [{mark}] {description}")
        if not condition:
            self.failures.append(description)


async def _drain(ws, expect, limit: int = 40, timeout: float = 20.0) -> dict:
    """The next message of an acceptable type, skipping the rest.

    `expect` is a type or a tuple of them, and passing a tuple is usually
    right. The first version of this waited for ONE type, and the first
    time the Tower answered a command with `cv_lab_error` instead of
    `cv_lab_status` -- which is exactly what the contract says it does
    when a command does not apply -- this function skipped the reply it
    was given and blocked until the socket died.

    That is a bug worth naming rather than quietly fixing, because it is
    the same bug an iOS client will write: wherever you expect
    `cv_lab_status`, a `cv_lab_error` may arrive instead, and it carries
    the status anyway.
    """
    wanted = (expect,) if isinstance(expect, str) else tuple(expect)
    for _ in range(limit):
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if message.get("type") in wanted:
            return message
    raise AssertionError(f"never saw any of {wanted} in {limit} messages")


async def _command(ws, payload: dict, checks, description: str) -> dict:
    """Send a command and report which of the two replies came back."""
    await ws.send(json.dumps(payload))
    reply = await _drain(ws, ("cv_lab_status", "cv_lab_error"))
    accepted = reply["type"] == "cv_lab_status"
    checks.that(
        accepted,
        f"{description}"
        + ("" if accepted else f" -- refused: {reply.get('reason')}"),
    )
    return reply


async def _await_running(ws, checks, timeout_s: float) -> dict:
    """Poll status until the Lab stops arming.

    A model-backed experiment takes seconds to arm -- `depth` measured
    3.46 s on a warm CUDA cache and can take two minutes on a cold one --
    and every frame sent in that window is refused with
    `cv_lab_starting`. A client that starts an experiment and immediately
    expects results is measuring its own impatience.
    """
    deadline = asyncio.get_running_loop().time() + timeout_s
    started_at = asyncio.get_running_loop().time()
    while True:
        await ws.send(json.dumps({"type": "cv_lab_status"}))
        status = (await _drain(ws, ("cv_lab_status", "cv_lab_error")))["status"]
        state = status["lifecycle"]["state"]
        if state != "starting":
            elapsed = asyncio.get_running_loop().time() - started_at
            print(f"       armed in {elapsed:.2f} s, state {state}")
            checks.that(state == "running", f"the Lab reached running (got {state})")
            if state == "failed":
                print(f"       reason: {status['lifecycle']['reason']}")
            return status
        if asyncio.get_running_loop().time() > deadline:
            checks.that(False, f"the Lab was still arming after {timeout_s} s")
            return status
        await asyncio.sleep(0.25)


async def run(args) -> int:
    import websockets

    checks = Checks()
    http_base = f"http://{args.host}:{args.port}"
    ws_url = f"ws://{args.host}:{args.port}/ws"

    print(f"-- HTTP discovery at {http_base}")
    with urllib.request.urlopen(f"{http_base}/cartridges", timeout=10) as response:
        declaration = json.load(response)
    offers = {entry["cartridge"]: entry for entry in declaration["cartridges"]}
    checks.that("experimental_cv" in offers, "GET /cartridges offers experimental_cv")
    cv_offer = offers.get("experimental_cv", {})
    checks.that(
        cv_offer.get("available") is True,
        f"the offer is available (reason: {cv_offer.get('unavailable_reason')})",
    )
    print(f"       contract {cv_offer.get('contract')}")

    with urllib.request.urlopen(f"{http_base}/cv-lab", timeout=10) as response:
        over_http = json.load(response)
    catalog = over_http["status"]["available"]
    checks.that(len(catalog) >= 1, f"GET /cv-lab lists {len(catalog)} experiments")
    for entry in catalog:
        flag = "" if entry["available"] else "  UNAVAILABLE"
        print(f"       {entry['id']:20s} {entry['headline_label']:26s}{flag}")
    checks.that(
        any(entry["id"] == args.experiment for entry in catalog),
        f"{args.experiment!r} is in the catalog",
    )

    print(f"-- socket at {ws_url}")
    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "ping"}))
        checks.that((await _drain(ws, "pong")) is not None, "ping/pong")

        await ws.send(json.dumps({"type": "cv_lab_status", "request_id": "smoke-0"}))
        reply = await _drain(ws, "cv_lab_status")
        checks.that(reply.get("request_id") == "smoke-0", "request_id is echoed")
        checks.that(
            json.dumps(reply["status"]["available"], sort_keys=True)
            == json.dumps(catalog, sort_keys=True),
            "the socket and GET /cv-lab agree about the catalog",
        )

        await ws.send(json.dumps({"type": "stream_start"}))

        print(f"-- start {args.experiment}")
        started = await _command(
            ws,
            {
                "type": "cv_lab_start",
                "experiment_id": args.experiment,
                "request_id": "smoke-1",
            },
            checks,
            "the start was accepted",
        )
        if started["type"] != "cv_lab_status":
            print("cannot continue without an accepted start")
            return 1
        checks.that(
            started.get("accepted_command") == "cv_lab_start",
            "the reply names the command it answers",
        )
        checks.that(
            started["status"]["selected"] == args.experiment,
            f"selected is now {args.experiment!r}",
        )
        run_id = started["status"]["lifecycle"]["run_id"]
        print(f"       run {run_id}, state {started['status']['lifecycle']['state']}")

        # Wait for the arm before sending a frame. See _await_running.
        armed = await _await_running(ws, checks, args.arm_timeout)
        checks.that(
            armed["lifecycle"]["run_id"] == run_id,
            "arming did not change the run out from under us",
        )

        print(f"-- {args.frames} frames")
        labels = set()
        refusals = set()
        run_ids = set()
        for seq in range(1, args.frames + 1):
            await ws.send(
                json.dumps(
                    {
                        "type": "frame",
                        "seq": seq * 30,
                        "width": WIDTH,
                        "height": HEIGHT,
                        "format": "jpeg",
                        "data": _jpeg(seq),
                    }
                )
            )
            reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            if reply["type"] == "frame_error":
                refusals.add(reply["reason"])
                print(f"       #{seq} refused: {reply['reason']}")
                continue
            provenance = reply.get("cv_lab", {})
            labels.add(reply["result_label"])
            run_ids.add(provenance.get("run_id"))
            print(
                f"       #{seq} {reply['result_label']}={reply['result_value']:.5f}"
                f"  {reply['processing_ms']:.2f} ms"
                f"  run {provenance.get('run_id')}"
                f"  seq {provenance.get('result_seq')}"
            )

        checks.that(bool(labels), "at least one frame produced a result")
        checks.that(
            not refusals,
            f"no frame was refused after arming (saw {refusals or 'none'})",
        )
        checks.that(
            run_ids == {run_id},
            f"every result carried the current run id (saw {run_ids})",
        )

        await ws.send(json.dumps({"type": "cv_lab_status"}))
        running = (await _drain(ws, "cv_lab_status"))["status"]
        checks.that(
            running["source"]["receiving_frames"] is True,
            "the Tower says it is receiving frames",
        )
        checks.that(
            running["run"]["frames_processed"] >= 1,
            f"frames_processed = {running['run']['frames_processed']}",
        )
        for metric in running["run"]["metrics"]:
            print(
                f"       {metric['label']:28s} {metric['value']}"
                f" {metric['unit'] or ''}"
                f"  [{metric['aggregation']}, {metric['provenance']},"
                f" {metric['frames']} frames]"
            )

        print("-- pause")
        paused = await _command(
            ws,
            {"type": "cv_lab_pause", "run_id": run_id},
            checks,
            "pause was accepted",
        )
        checks.that(
            paused.get("status", {}).get("lifecycle", {}).get("state") == "paused",
            "the Lab is paused",
        )
        await ws.send(
            json.dumps(
                {
                    "type": "frame",
                    "seq": 9000,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "format": "jpeg",
                    "data": _jpeg(99),
                }
            )
        )
        refused = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        checks.that(
            refused["type"] == "frame_error" and refused["reason"] == "cv_lab_paused",
            "a frame while paused is refused with cv_lab_paused",
        )
        print(f"       {refused.get('message')}")

        print("-- resume, then stop")
        resumed = await _command(
            ws,
            {"type": "cv_lab_resume", "run_id": run_id},
            checks,
            "resume was accepted",
        )
        checks.that(
            resumed.get("status", {}).get("lifecycle", {}).get("state") == "running",
            "the Lab resumed",
        )

        stopped = await _command(
            ws,
            {"type": "cv_lab_stop", "run_id": run_id},
            checks,
            "stop was accepted",
        )
        summary = stopped.get("status", {})
        checks.that(
            summary["lifecycle"]["state"] == "stopped", "the Lab stopped"
        )
        checks.that(
            summary["run"]["ended_at"] is not None, "the run has an end time"
        )
        checks.that(
            summary["run"]["frames_processed"] >= 1,
            "the run summary survives the stop",
        )
        print(
            f"       {summary['run']['frames_processed']} processed,"
            f" {summary['run']['frames_refused']} refused,"
            f" {summary['run']['frames_failed']} failed,"
            f" {summary['run']['timings']['processing_ms']} ms/frame"
        )

        print("-- refusals are legible")
        await ws.send(
            json.dumps({"type": "cv_lab_start", "experiment_id": "not_an_experiment"})
        )
        error = await _drain(ws, "cv_lab_error")
        checks.that(
            error["reason"] == "unknown_experiment",
            f"an unknown experiment is refused ({error['reason']})",
        )
        await ws.send(json.dumps({"type": "cv_lab_stop", "run_id": "not-a-run"}))
        error = await _drain(ws, "cv_lab_error")
        checks.that(
            error["reason"] == "stale_run", f"a stale run id is refused ({error['reason']})"
        )

        print("-- the result channel")
        await ws.send(
            json.dumps(
                {
                    "type": "result_subscribe",
                    "cartridge": "experimental_cv",
                    "result_type": "status",
                }
            )
        )
        ack = await _drain(ws, "result_subscribed")
        checks.that(ack["cartridge"] == "experimental_cv", "subscribed")
        envelope = await _drain(ws, "cartridge_result")
        checks.that(
            envelope["payload"]["lifecycle"]["run_id"] == run_id,
            "the pushed snapshot names the same run",
        )
        print(
            f"       envelope seq {envelope['seq']}, revision"
            f" {envelope['revision']}, {len(json.dumps(envelope['payload']))} B"
        )

        await ws.send(json.dumps({"type": "stream_stop"}))

    print()
    if checks.failures:
        print(f"{len(checks.failures)} CHECK(S) FAILED:")
        for failure in checks.failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Drive a running Tower through the CV Lab workflow."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--experiment", default="edge_detection")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument(
        "--arm-timeout",
        type=float,
        default=150.0,
        help=(
            "seconds to wait for an experiment to arm. A model-backed one "
            "downloads weights on a cold cache; the Tower bounds it at 120 s"
        ),
    )
    args = parser.parse_args(argv)

    try:
        return asyncio.run(run(args))
    except (urllib.error.URLError, OSError) as exc:
        print(f"could not reach a Tower at {args.host}:{args.port} -- {exc}")
        print("start one with:")
        print(
            "  .venv\\Scripts\\python.exe -m uvicorn tower.main:app "
            "--host 0.0.0.0 --port 8000"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
