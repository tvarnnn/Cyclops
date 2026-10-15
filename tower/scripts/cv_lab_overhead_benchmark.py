#!/usr/bin/env python
"""What the CV Lab productization COSTS, on top of the experiment itself.

A different question from `cv_lab_benchmark.py`, which measures what each
experiment costs to run. This measures what was added around it: the
per-frame bookkeeping and attribution, the status document, and the
result channel that publishes it. The point is to establish that the Lab
can coexist with a Tower that is also answering frames, recording a
capture and supervising a world build -- not to celebrate a small number.

SYNTHETIC, NOT PHYSICAL. The imagery is rendered and the client is a
`TestClient` in this process, so nothing here says anything about the
Ray-Ban camera or about a Tailscale link. What IS real is the code path:
the same `CVLab`, the same status builder, the same envelope.

Read the four sections as:

    frame            what one frame pays for bookkeeping + attribution
    status           what building the document costs, and how it scales
                     with subscribers
    channel          what a subscription costs per published snapshot
    memory           whether a long run grows

    .venv\\Scripts\\python.exe scripts/cv_lab_overhead_benchmark.py
    .venv\\Scripts\\python.exe scripts/cv_lab_overhead_benchmark.py --format json
    .venv\\Scripts\\python.exe scripts/cv_lab_overhead_benchmark.py --frames 5000
"""

import argparse
import asyncio
import gc
import io
import tracemalloc
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.cv_lab import CVLab  # noqa: E402
from tower.experiments import EXPERIMENTS, ExperimentSettings  # noqa: E402
from tower.results.envelope import compute_revision  # noqa: E402

# The resolution the glasses actually deliver today.
WIDTH, HEIGHT = 640, 360

# The experiment used wherever the point is the OVERHEAD rather than the
# experiment. `baseline` is the cheapest registered one (~1 ms), so it is
# the harshest denominator: overhead that disappears against `depth` at
# 26 ms is still visible here.
CHEAPEST = "baseline"

# A stateless experiment with a real metrics bag, for the case where
# bookkeeping actually has something to do.
BUSIEST = "frame_quality"


def _frame() -> bytes:
    rng = np.random.default_rng(11)
    array = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=50)
    return buffer.getvalue()


def _timed(call, repeat: int) -> dict:
    """Mean, median and p95 over `repeat` calls, after one warm-up.

    p95 rather than max: a single sample on a machine running three other
    lanes says more about the machine than the code, and the Tower this
    was measured on had two world-builder benchmarks running beside it.
    """
    call()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return {
        "mean_ms": round(statistics.fmean(samples), 5),
        "median_ms": round(statistics.median(samples), 5),
        "p95_ms": round(samples[int(len(samples) * 0.95) - 1], 5),
        "samples": len(samples),
    }


def measure_frame_overhead(repeat: int) -> dict:
    """What the Lab's per-frame bookkeeping costs, measured DIRECTLY.

    Not by subtracting two end-to-end timings. That was the first version
    of this function and it was worthless: the effect is tens of
    microseconds and the two measurements it differenced were 2 ms and
    12 ms with several percent of run-to-run spread, so it reported a
    NEGATIVE overhead for `baseline` on a loaded machine. A difference
    smaller than the noise in either term is not a measurement.

    So the bookkeeping is timed on its own, against a pre-computed
    result: `record_result` folds one frame into the run's accumulators,
    and `_provenance` builds the attribution block that travels on
    `frame_result`. Those two ARE the per-frame cost of productization --
    everything else on the frame path was there before.

    The end-to-end comparison is kept beside it, honestly labelled, so
    the direct number can be sanity-checked against a total rather than
    trusted alone.
    """
    payload = _frame()
    rows = {}
    for experiment_id in (CHEAPEST, BUSIEST):
        lab = asyncio.run(_armed(experiment_id))
        result = lab.process(payload)
        lab.frame_provenance()
        run = lab._run

        bookkeeping = _timed(lambda: run.record_result(result, 1.0), repeat * 20)
        attribution = _timed(
            lambda: lab._provenance(run, result, 1, 1.0), repeat * 20
        )

        bare = EXPERIMENTS[experiment_id]()
        bare.load(ExperimentSettings())
        direct = _timed(lambda: bare.run(payload), repeat)
        bare.release()

        def through_lab():
            lab.process(payload)
            lab.frame_provenance()

        wrapped = _timed(through_lab, repeat)
        lab.release()

        added = bookkeeping["mean_ms"] + attribution["mean_ms"]
        rows[experiment_id] = {
            "record_result_ms": bookkeeping,
            "provenance_ms": attribution,
            "added_per_frame_ms": round(added, 5),
            "added_percent_of_experiment": round(
                100 * added / direct["mean_ms"], 3
            ),
            # Kept for scale, NOT for the overhead figure. On a machine
            # running other work these two differ by more than the
            # quantity above.
            "experiment_ms": direct,
            "through_lab_ms": wrapped,
        }
    return rows


async def _armed(experiment_id: str) -> CVLab:
    lab = CVLab(experiment_id, connection_count=lambda: 1)
    await lab.load_initial()
    return lab


def measure_status_cost(repeat: int) -> dict:
    """What building the one document costs, and what hashing it costs.

    Both matter because the result channel does both on every poll: the
    hub builds a snapshot and `compute_revision` canonicalises it to
    decide whether anything changed.
    """
    rows = {}
    for experiment_id in (CHEAPEST, BUSIEST):
        lab = asyncio.run(_armed(experiment_id))
        payload = _frame()
        for _ in range(20):
            lab.process(payload)
            lab.frame_provenance()

        build = _timed(lab.status, repeat)
        document = lab.status()
        revision = _timed(lambda: compute_revision(document), repeat)
        serialise = _timed(lambda: json.dumps(document), repeat)
        rows[experiment_id] = {
            "build_ms": build,
            "revision_ms": revision,
            "serialise_ms": serialise,
            "bytes": len(json.dumps(document)),
            "metrics": len(document["run"]["metrics"]),
        }
        lab.release()
    return rows


def measure_channel_cost(repeat: int) -> dict:
    """A full poll pass with N subscribers, through the real hub.

    The hub computes each distinct target ONCE and offers the result to
    every channel, so this is where that claim is checked rather than
    asserted: ten subscribers must not cost ten status builds.
    """
    from fastapi.testclient import TestClient

    from tower.main import create_app

    logging.disable(logging.CRITICAL)
    client = TestClient(create_app())
    client.__enter__()
    try:
        hub = client.app.state.result_hub
        hub._poll_seconds = 3600.0
        hub._heartbeat_seconds = 0.0
        # Count the snapshot BUILDS, not just the wall clock. The hub's
        # whole claim is that N subscribers watching one thing cost one
        # computation, and a timing that grows with N cannot tell "we
        # built it N times" from "we sent it N times". Only one of those
        # is a defect.
        builds = {"count": 0}
        real_snapshot_for = hub._snapshot_for

        def counting_snapshot_for(*args, **kwargs):
            builds["count"] += 1
            return real_snapshot_for(*args, **kwargs)

        hub._snapshot_for = counting_snapshot_for

        rows = {}
        sockets = []
        try:
            for count in (1, 4, 8):
                while len(sockets) < count:
                    ws = client.websocket_connect("/ws").__enter__()
                    ws.send_json(
                        {
                            "type": "result_subscribe",
                            "cartridge": "experimental_cv",
                            "result_type": "status",
                        }
                    )
                    sockets.append(ws)

                async def poll():
                    await hub.poll_once()

                builds["count"] = 0
                measured = _timed(lambda: client.portal.call(poll), repeat)
                # `_timed` runs one warm-up call plus `repeat` timed ones.
                measured["snapshot_builds_per_poll"] = round(
                    builds["count"] / (repeat + 1), 3
                )
                measured["subscribers"] = count
                rows[f"subscribers_{count}"] = measured
        finally:
            for ws in sockets:
                try:
                    ws.__exit__(None, None, None)
                except Exception:
                    pass
        return rows
    finally:
        client.__exit__(None, None, None)
        logging.disable(logging.NOTSET)


def measure_memory(frames: int) -> dict:
    """Does a long run grow. Measured with TWO instruments, on purpose.

    `handoff.md` 9.3 says a `stream_stop` MAY NEVER ARRIVE, so a run is
    open for as long as the Tower is up. A run that grew per frame would
    be the unbounded store this whole design avoids.

    RSS alone cannot answer it. RSS is what the process has asked the OS
    for, and in a process running OpenCV and numpy that includes their own
    pools -- which grow, shrink and hold memory on their own schedule.
    Two runs of an earlier version of this benchmark, over identical
    work, reported -524 KB and +2.8 MB. A quantity that changes sign
    between runs is not measuring the thing.

    So `tracemalloc` runs alongside it and attributes PYTHON allocations
    to the lines that made them. That one answers "did the Lab keep
    anything", and RSS stays as context for "what did the process do".
    """
    process = psutil.Process()
    payload = _frame()
    lab = asyncio.run(_armed(BUSIEST))

    # A long warm-up before the first reading. Without it the measurement
    # is dominated by one-off allocation -- OpenCV's own buffers, the
    # logging machinery, the arena the interpreter grows once -- and
    # publishes it as per-frame growth.
    for _ in range(500):
        lab.process(payload)
        lab.frame_provenance()
    gc.collect()

    # Several checkpoints, not two. Two readings cannot tell steady
    # allocator noise from linear growth, and linear growth is the only
    # thing worth reporting: a run is open for as long as the Tower is
    # up.
    checkpoints = []
    baseline_rss = process.memory_info().rss
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    step = max(1, frames // 5)
    processed = 0
    start = time.perf_counter()
    for _ in range(5):
        for _ in range(step):
            lab.process(payload)
            lab.frame_provenance()
        processed += step
        gc.collect()
        checkpoints.append(
            {
                "frames": processed,
                "rss_bytes": process.memory_info().rss,
                "growth_bytes": process.memory_info().rss - baseline_rss,
            }
        )
    elapsed = time.perf_counter() - start

    after = tracemalloc.take_snapshot()
    stats = after.compare_to(before, "lineno")
    tracked = sum(stat.size_diff for stat in stats)
    # The lines the Lab itself owns, separated from the interpreter's own
    # bookkeeping and from tracemalloc's.
    lab_lines = [
        {
            "line": str(stat.traceback[0]),
            "bytes": stat.size_diff,
            "objects": stat.count_diff,
        }
        for stat in stats[:20]
        if "tower" in str(stat.traceback[0]) and stat.size_diff > 0
    ][:5]
    tracemalloc.stop()

    document = lab.status()
    lab.release()
    growth = checkpoints[-1]["growth_bytes"]
    # Growth between the FIRST and LAST checkpoint, both taken after the
    # warm-up. If the Lab leaked per frame this would track the frame
    # count; if it is allocator noise it stays flat and may go negative.
    steady = checkpoints[-1]["rss_bytes"] - checkpoints[0]["rss_bytes"]
    steady_frames = checkpoints[-1]["frames"] - checkpoints[0]["frames"]
    return {
        "frames": processed,
        "tracked_bytes": tracked,
        "tracked_bytes_per_frame": round(tracked / max(processed, 1), 4),
        "tracked_lab_lines": lab_lines,
        "checkpoints": checkpoints,
        "rss_growth_bytes": growth,
        "steady_growth_bytes": steady,
        "steady_growth_bytes_per_frame": round(steady / max(steady_frames, 1), 4),
        "wall_s": round(elapsed, 3),
        "frames_per_second": round(processed / elapsed, 1),
        "document_bytes": len(json.dumps(document)),
        "tracked_metrics": len(document["run"]["metrics"]),
    }


def _render(report: dict) -> str:
    lines = [
        "CV Lab productization overhead -- SYNTHETIC, NOT PHYSICAL",
        f"frame {WIDTH}x{HEIGHT} JPEG q50, {report['host']['cpu_count']} CPUs",
        "",
        "-- per frame ------------------------------------------------",
    ]
    for name, row in report["frame"].items():
        lines.append(
            f"  {name:16s} record {row['record_result_ms']['mean_ms']:7.5f} ms"
            f"  + attribute {row['provenance_ms']['mean_ms']:7.5f} ms"
            f"  = {row['added_per_frame_ms']:7.5f} ms"
            f"  ({row['added_percent_of_experiment']:.3f}% of the experiment"
            f" at {row['experiment_ms']['mean_ms']:.2f} ms)"
        )
    lines.append(
        "  (end-to-end, for scale only -- on a loaded machine these differ"
        " by more than the figure above)"
    )
    for name, row in report["frame"].items():
        lines.append(
            f"  {name:16s} experiment {row['experiment_ms']['mean_ms']:8.4f} ms"
            f"   through Lab {row['through_lab_ms']['mean_ms']:8.4f} ms"
        )
    lines += ["", "-- status document ------------------------------------------"]
    for name, row in report["status"].items():
        lines.append(
            f"  {name:16s} build {row['build_ms']['mean_ms']:7.4f} ms"
            f"   revision {row['revision_ms']['mean_ms']:7.4f} ms"
            f"   json {row['serialise_ms']['mean_ms']:7.4f} ms"
            f"   {row['bytes']:5d} B  {row['metrics']} metrics"
        )
    lines += ["", "-- one poll pass, by subscriber count ------------------------"]
    for name, row in report["channel"].items():
        lines.append(
            f"  {name:16s} {row['mean_ms']:8.4f} ms  p95 {row['p95_ms']:8.4f} ms"
            f"   {row['snapshot_builds_per_poll']:.2f} snapshot builds per poll"
        )
    memory = report["memory"]
    lines += [
        "",
        "-- memory over a long run -----------------------------------",
        f"  {memory['frames']} frames in {memory['wall_s']} s"
        f" ({memory['frames_per_second']} fps), after a 500-frame warm-up",
    ]
    for checkpoint in memory["checkpoints"]:
        lines.append(
            f"    {checkpoint['frames']:6d} frames  RSS"
            f" {checkpoint['rss_bytes'] / 1048576:8.2f} MB"
            f"  ({checkpoint['growth_bytes']:+d} B since warm-up)"
        )
    lines += [
        f"  RSS first -> last checkpoint: {memory['steady_growth_bytes']:+d} B"
        f" over {memory['frames'] - memory['checkpoints'][0]['frames']} frames"
        f" ({memory['steady_growth_bytes_per_frame']:+.4f} B/frame)",
        "  -- RSS includes OpenCV/numpy pools and changes sign between"
        " runs. The next line is the one that answers the question --",
        f"  tracemalloc net: {memory['tracked_bytes']:+d} B over"
        f" {memory['frames']} frames"
        f" ({memory['tracked_bytes_per_frame']:+.4f} B/frame)",
    ]
    for entry in memory["tracked_lab_lines"]:
        lines.append(
            f"    {entry['bytes']:+8d} B  {entry['objects']:+5d} obj"
            f"  {entry['line']}"
        )
    lines += [
        f"  document still {memory['document_bytes']} B with"
        f" {memory['tracked_metrics']} metrics",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="What the CV Lab costs on top of its experiments."
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument(
        "--frames",
        type=int,
        default=2000,
        help="frames for the memory-growth measurement",
    )
    args = parser.parse_args(argv)

    report = {
        "host": {"cpu_count": psutil.cpu_count(logical=True)},
        "resolution": [WIDTH, HEIGHT],
        "frame": measure_frame_overhead(args.repeat),
        "status": measure_status_cost(args.repeat),
        "channel": measure_channel_cost(max(4, args.repeat // 4)),
        "memory": measure_memory(args.frames),
        "caveat": (
            "SYNTHETIC, NOT PHYSICAL: rendered imagery, in-process client, "
            "no network. Real for the code, not a statement about a room "
            "or a link."
        ),
    }

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(_render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
