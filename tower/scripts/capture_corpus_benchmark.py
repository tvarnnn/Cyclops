"""Run a CV Lab experiment over the frames the glasses actually produced.

`cv_lab_benchmark.py` measures against synthetic renders, which is right
for comparing experiments to each other and wrong for predicting anything.
Synthetic frames carry no motion blur, no rolling shutter, no auto-exposure
hunting, and no JPEG artefacts at the quality the phone actually sends. A
number measured on them describes the renderer.

This walks `data/captures/` instead -- real Ray-Ban frames at 360x640
(18 captures and ~9,199 frames in the corpus when this was written).

    python scripts/capture_corpus_benchmark.py object_detection
    python scripts/capture_corpus_benchmark.py depth --per-capture-limit 50
    python scripts/capture_corpus_benchmark.py baseline --format json

It reads. It never writes to the corpus, and it persists nothing: the
Experimental CV Lab declares `persists_data=False` and a boundary test
enforces it, so a benchmark that wrote results would break the cartridge's
own guarantee.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# `scripts/` is a package inside the tower project; running this file
# directly still needs the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.experiments import (  # noqa: E402
    EXPERIMENTS,
    ExperimentSettings,
    MetricKind,
    classify_metric,
)

# How to combine a metric across frames is NOT decided here. A name is not
# evidence of what a number means; the experiment that produced it is, and
# it says so in its own METRIC_KINDS. A metric that says nothing raises here
# rather than defaulting to a plausible, meaningless total. (The superseded
# name allowlist, with everything else summed, had eight of its eleven names
# dead and summed fifteen rate-like metrics, so `tracked_fraction` -- a
# quantity that cannot exceed 1 -- came out as 768 over the real corpus.)

# A CONSTANT with more distinct values than this is not a constant, and
# whatever it is, accumulating one entry per frame across the corpus (~9,199
# frames when this was written) is not a report. The corpus holds a handful of resolutions, so the ceiling
# is loose enough never to fire on a correct declaration.
_MAX_CONSTANT_VALUES = 16


class MisclassifiedConstantError(ValueError):
    """A metric declared CONSTANT that keeps changing.

    Raised rather than smoothed over: a constant that varies is either a
    misdeclaration by the experiment or a genuinely surprising corpus,
    and both are things a benchmark should stop and say out loud.
    """


@dataclass
class CaptureReport:
    """One capture's contribution."""

    capture_id: str
    frames: int = 0
    failed: int = 0
    timings_ms: list[float] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.timings_ms) if self.timings_ms else 0.0

    def to_json_dict(self) -> dict:
        return {
            "capture_id": self.capture_id,
            "frames": self.frames,
            "failed": self.failed,
            "median_ms": round(self.median_ms, 3),
        }


@dataclass
class CorpusReport:
    """What the whole corpus said.

    `failed` is carried separately from `frames` rather than folded into
    it, because "9,198 frames at 40 ms" and "9,198 frames at 40 ms plus one
    that would not decode" are different claims, and the second one is the
    true one.
    """

    experiment: str
    device: str
    frames: int = 0
    failed: int = 0
    captures: int = 0
    timings_ms: list[float] = field(default_factory=list)
    per_capture: dict[str, CaptureReport] = field(default_factory=dict)
    # One field per MetricKind, so a reader never has to ask which
    # aggregation produced a number.
    summed_metrics: dict[str, float] = field(default_factory=dict)
    averaged_metrics: dict[str, float] = field(default_factory=dict)
    # metric -> {value: frames that reported it}. Usually one entry.
    constant_metrics: dict[str, dict[float, int]] = field(default_factory=dict)
    # metric -> frames that reported it, and nothing else. A circular
    # quantity has no mean and no total; saying how often it was measured
    # is the most a corpus summary can honestly say about it.
    unaggregated_metrics: dict[str, int] = field(default_factory=dict)
    mean_intensity: float = 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.timings_ms) if self.timings_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.timings_ms:
            return 0.0
        ordered = sorted(self.timings_ms)
        return ordered[max(0, int(len(ordered) * 0.95) - 1)]

    def to_json_dict(self) -> dict:
        return {
            "experiment": self.experiment,
            "device": self.device,
            "captures": self.captures,
            "frames": self.frames,
            "failed": self.failed,
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "fps_at_median": round(1000.0 / self.median_ms, 2) if self.median_ms else 0.0,
            "mean_intensity": round(self.mean_intensity, 3),
            "summed_metrics": {k: round(v, 3) for k, v in sorted(self.summed_metrics.items())},
            "averaged_metrics": {
                k: round(v, 4) for k, v in sorted(self.averaged_metrics.items())
            },
            "constant_metrics": {
                name: [
                    {"value": value, "frames": frames}
                    for value, frames in sorted(
                        values.items(), key=lambda kv: (-kv[1], kv[0])
                    )
                ]
                for name, values in sorted(self.constant_metrics.items())
            },
            "unaggregated_metrics": dict(sorted(self.unaggregated_metrics.items())),
            "per_capture": [
                r.to_json_dict() for r in sorted(
                    self.per_capture.values(), key=lambda r: r.capture_id
                )
            ],
        }


def iter_capture_frames(
    root: Path | str, *, per_capture_limit: int | None = None
) -> Iterator[tuple[str, int, bytes]]:
    """Yield `(capture_id, source_seq, jpeg_bytes)` across every capture.

    The limit is PER CAPTURE, not per corpus: a corpus-wide cap would spend
    its whole budget inside the first long capture and never reach the
    others, which is the opposite of what a corpus benchmark is for.

    A capture directory with no frames is skipped rather than reported as
    an error -- `capture.py` creates the directory when recording starts,
    so a start-then-stop leaves exactly that shape behind.
    """
    root = Path(root)
    if not root.is_dir():
        return

    for capture_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        frames_dir = capture_dir / "frames"
        if not frames_dir.is_dir():
            continue

        paths = sorted(frames_dir.glob("*.jpg"))
        if per_capture_limit is not None:
            paths = paths[:per_capture_limit]

        for path in paths:
            try:
                seq = int(path.stem)
            except ValueError:
                # Not one of ours; the recorder writes zero-padded seqs.
                continue
            yield capture_dir.name, seq, path.read_bytes()


def benchmark_corpus(
    root: Path | str,
    experiment_name: str,
    *,
    per_capture_limit: int | None = None,
    device: str = "auto",
) -> CorpusReport:
    """Run one experiment over the corpus and summarise honestly."""
    factory = EXPERIMENTS[experiment_name]  # KeyError names the bad choice
    report = CorpusReport(experiment=experiment_name, device=device)

    experiment = factory()
    experiment.load(ExperimentSettings(device=device))

    summed: dict[str, float] = {}
    averaged: dict[str, list[float]] = {}
    constants: dict[str, dict[float, int]] = {}
    unaggregated: dict[str, int] = {}
    intensities: list[float] = []

    try:
        for capture_id, _seq, raw in iter_capture_frames(
            root, per_capture_limit=per_capture_limit
        ):
            per = report.per_capture.setdefault(capture_id, CaptureReport(capture_id))

            started = time.perf_counter()
            try:
                result = experiment.run(raw)
            except Exception:
                # One truncated JPEG must not cost the other 9,198 frames.
                # Counted, never silently dropped.
                per.failed += 1
                report.failed += 1
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            per.frames += 1
            per.timings_ms.append(elapsed_ms)
            report.frames += 1
            report.timings_ms.append(elapsed_ms)

            if result.mean_intensity is not None:
                intensities.append(result.mean_intensity)

            for name, value in result.metrics.items():
                # Raises UnclassifiedMetricError if the experiment never
                # said what this number is. That is the point.
                kind = classify_metric(experiment_name, name)
                if kind is MetricKind.RATE:
                    averaged.setdefault(name, []).append(value)
                elif kind is MetricKind.COUNT:
                    summed[name] = summed.get(name, 0.0) + value
                elif kind is MetricKind.CONSTANT:
                    seen = constants.setdefault(name, {})
                    seen[value] = seen.get(value, 0) + 1
                    if len(seen) > _MAX_CONSTANT_VALUES:
                        raise MisclassifiedConstantError(
                            f"{experiment_name}.{name} is declared CONSTANT "
                            f"but has taken more than {_MAX_CONSTANT_VALUES} "
                            f"distinct values across {report.frames} frames"
                        )
                else:
                    unaggregated[name] = unaggregated.get(name, 0) + 1
    finally:
        experiment.release()

    report.captures = len(report.per_capture)
    report.summed_metrics = summed
    report.averaged_metrics = {k: statistics.fmean(v) for k, v in averaged.items()}
    report.constant_metrics = constants
    report.unaggregated_metrics = unaggregated
    report.mean_intensity = statistics.fmean(intensities) if intensities else 0.0
    return report


def _render_text(report: CorpusReport) -> str:
    lines = [
        f"experiment   {report.experiment}  (device={report.device})",
        f"captures     {report.captures}",
        f"frames       {report.frames}"
        + (f"   FAILED {report.failed}" if report.failed else ""),
    ]
    if report.frames:
        lines += [
            f"median       {report.median_ms:.1f} ms"
            f"   ({1000.0 / report.median_ms:.1f} fps)",
            f"p95          {report.p95_ms:.1f} ms",
        ]
        if report.mean_intensity:
            lines.append(f"mean_intensity {report.mean_intensity:.2f}")
        if report.summed_metrics:
            lines.append("summed:")
            for name, value in sorted(report.summed_metrics.items()):
                lines.append(f"  {name:<28} {value:g}")
        if report.averaged_metrics:
            lines.append("averaged:")
            for name, value in sorted(report.averaged_metrics.items()):
                lines.append(f"  {name:<28} {value:.4f}")
        if report.constant_metrics:
            lines.append("constant (value x frames):")
            for name, values in sorted(report.constant_metrics.items()):
                rendered = ", ".join(
                    f"{value:g} x{frames}"
                    for value, frames in sorted(
                        values.items(), key=lambda kv: (-kv[1], kv[0])
                    )
                )
                lines.append(f"  {name:<28} {rendered}")
        if report.unaggregated_metrics:
            lines.append("not aggregated (no meaningful corpus summary):")
            for name, frames in sorted(report.unaggregated_metrics.items()):
                lines.append(f"  {name:<28} measured on {frames} frames")
    else:
        lines.append("no frames found -- is TOWER_CAPTURE_ROOT right?")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a CV Lab experiment over real captured frames."
    )
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument(
        "--root",
        type=artifact_root_arg,
        default="data/captures",
        help="capture root (default: data/captures)",
    )
    parser.add_argument(
        "--per-capture-limit",
        type=int,
        default=None,
        help="cap frames taken from EACH capture, so one long capture "
        "cannot starve the rest",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--format", default="text", choices=["text", "json"])
    args = parser.parse_args(argv)

    report = benchmark_corpus(
        args.root,
        args.experiment,
        per_capture_limit=args.per_capture_limit,
        device=args.device,
    )

    if args.format == "json":
        print(json.dumps(report.to_json_dict(), indent=2))
    else:
        print(_render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
