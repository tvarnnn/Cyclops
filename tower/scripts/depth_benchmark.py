#!/usr/bin/env python3
"""CPU vs GPU depth-experiment benchmark client for V0.9.1 measurement.

Sends synthetic JPEG frames to a running Tower /ws endpoint running the
`depth` experiment (TOWER_CV_EXPERIMENT=depth), and reports client-
observed round-trip latency per frame.

This script cannot see the Tower's own per-stage timing (stage_ms_avg,
GPU memory) directly -- copy the Tower process's own
[Tower][Session] final summary log line into the printed report's
"Tower-side measured" section by hand, same as scripts/soak_test_stream.py.

Usage: start the Tower twice, once per device, and run this script once
against each running instance:

    TOWER_CV_EXPERIMENT=depth TOWER_CV_DEVICE=cpu  python -m uvicorn tower.main:app
    .venv\\Scripts\\python.exe scripts/depth_benchmark.py --label cpu

    TOWER_CV_EXPERIMENT=depth TOWER_CV_DEVICE=cuda python -m uvicorn tower.main:app
    .venv\\Scripts\\python.exe scripts/depth_benchmark.py --label cuda

Requires the `dev` extra: pip install -e ".[dev,ml]"
"""
import argparse
import asyncio
import base64
import io
import json
import time

import websockets
from PIL import Image

VALID_LABELS = ("cpu", "cuda")


def _make_jpeg_base64(width: int, height: int) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(90, 90, 90)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def run_benchmark(uri: str, width: int, height: int, frame_count: int) -> dict:
    frame_data_b64 = _make_jpeg_base64(width, height)
    round_trip_ms: list[float] = []
    frame_errors: list[dict] = []

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "stream_start"}))
        for seq in range(1, frame_count + 1):
            payload = json.dumps(
                {
                    "type": "frame",
                    "seq": seq,
                    "width": width,
                    "height": height,
                    "format": "jpeg",
                    "data": frame_data_b64,
                }
            )
            frame_start = time.perf_counter()
            await ws.send(payload)
            response = json.loads(await ws.recv())
            elapsed_ms = (time.perf_counter() - frame_start) * 1000
            # frame_error is a legitimate reply (added V0.9.2): a skipped
            # frame or an unavailable module answers with it instead of
            # frame_result. Count it rather than crashing, and keep it out
            # of the latency stats -- it did not complete the CV path.
            if response.get("type") == "frame_error":
                frame_errors.append(response)
                continue
            round_trip_ms.append(elapsed_ms)
            if response.get("type") != "frame_result":
                raise RuntimeError(f"expected frame_result, got: {response}")
        await ws.send(json.dumps({"type": "stream_stop"}))

    # First frame excluded from the average: CUDA context first-init is a
    # known one-time warmup cost, not representative steady-state latency.
    if not round_trip_ms:
        raise RuntimeError(
            f"no frame_result received; {len(frame_errors)} frame_error replies, "
            f"first: {frame_errors[0] if frame_errors else 'none'}"
        )

    steady_state = round_trip_ms[1:] if len(round_trip_ms) > 1 else round_trip_ms
    return {
        "frame_count": frame_count,
        "frames_answered_with_result": len(round_trip_ms),
        # Non-zero invalidates the run as a clean latency baseline -- some
        # frames never completed the CV path.
        "frames_answered_with_error": len(frame_errors),
        "first_frame_round_trip_ms": round(round_trip_ms[0], 2),
        "round_trip_ms_avg": round(sum(steady_state) / len(steady_state), 2),
        "round_trip_ms_max": round(max(steady_state), 2),
    }


def _print_report(label: str, result: dict) -> None:
    print("## V0.9.1 Depth CPU/GPU Benchmark -- Client Run Report\n")
    print(f"**Device label (server-side TOWER_CV_DEVICE):** {label}\n")
    print("### Measured -- client (round-trip) side, this script")
    for key, value in result.items():
        print(f"- {key}: {value}")
    print(
        "\n### Measured -- Tower (stage_ms_avg / GPU memory) side\n"
        "Paste the Tower process's own log line here "
        "([Tower][Session] final summary: ...):\n\n"
        "```\n<paste here>\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="ws://127.0.0.1:8000/ws")
    parser.add_argument("--width", type=int, default=504)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--frame-count", type=int, default=60)
    parser.add_argument(
        "--label",
        required=True,
        choices=VALID_LABELS,
        help="Which TOWER_CV_DEVICE the currently-running Tower was started with, so the report cannot be mistaken for the other run.",
    )
    args = parser.parse_args()

    result = asyncio.run(run_benchmark(args.uri, args.width, args.height, args.frame_count))
    _print_report(args.label, result)


if __name__ == "__main__":
    main()
