"""Offline, pure pacing measurements from v1 frame-quality journal rows.

All defaults below are OPEN choices for validation, not calibrated wearer cues:
gap_percentile=0.75, gap_multiplier=4.0, blur_tolerance=1,
min_overlap_ratio=0.75, sharpness_floor=25.0, and top_k=5. The gap
threshold is derived from this session's positive adjacent receipt intervals;
there is no fixed frame-period cutoff. A held candidate measures pacing only:
it does not predict whether a look-back will link, identify a doorway, or
establish why frames were blurred. Times are Tower receipt times, not exposure
times. Inputs must be the rows of one session in journal order.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import median


_BURST_REASONS = frozenset({"blurred", "tracking_held", "tracking_degraded", "tracking_lost"})


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _percentile(values, percentile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _duration(first, last):
    return last - first if first is not None and last is not None else None


def _run_eligible(row, min_overlap_ratio):
    if row.get("sharpness") is None or row.get("tracker") != "reference":
        return False
    return row.get("outcome") == "accept" or (
        _number(row.get("overlap_ratio")) and row["overlap_ratio"] >= min_overlap_ratio
    )


def summarize(
    rows,
    *,
    gap_percentile=0.75,
    gap_multiplier=4.0,
    blur_tolerance=1,
    min_overlap_ratio=0.75,
    sharpness_floor=25.0,
    top_k=5,
) -> dict:
    """Summarize one journal in order; all thresholds/defaults are OPEN.

    ``held_runs`` are candidate sharp, reference-tracked spans with up to
    ``blur_tolerance`` internal blurred rows. They measure pacing only; a held
    look-back may still fail to link due to scene content, geometry, or scale.
    ``pause_after_burst_seconds`` instead measures the immediate uninterrupted
    sharp tracked dwell after each reported blur/loss burst, with no blur
    tolerance. Missing or unusable timestamps leave duration/rate unknown.
    Loss rate uses the sum of positive receipt intervals below the session gap
    threshold; reconnect downtime is excluded. No row, filesystem, global
    state, or Tower output is changed.
    """
    if (not _number(gap_percentile) or not 0 <= gap_percentile <= 1 or
            not _number(gap_multiplier) or gap_multiplier <= 0):
        raise ValueError("gap percentile must be in [0, 1] and multiplier positive")
    if not isinstance(blur_tolerance, int) or blur_tolerance < 0:
        raise ValueError("blur_tolerance must be a nonnegative integer")
    if (not _number(min_overlap_ratio) or not 0 <= min_overlap_ratio <= 1 or
            not _number(sharpness_floor) or sharpness_floor < 0):
        raise ValueError("overlap ratio and sharpness floor are out of range")
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")

    rows = list(rows)
    times = [r.get("received_at") if _number(r.get("received_at")) else None for r in rows]
    intervals = [b - a for a, b in zip(times, times[1:]) if a is not None and b is not None and b > a]
    gap_seconds = _percentile(intervals, gap_percentile) * gap_multiplier if intervals else None
    boundaries = [True]
    for a, b in zip(times, times[1:]):
        dt = b - a if a is not None and b is not None else None
        boundaries.append(dt is None or dt <= 0 or (gap_seconds is not None and dt > gap_seconds))

    scored = sum(r.get("sharpness") is not None for r in rows)
    blurred = sum(r.get("reason") == "blurred" for r in rows)
    lost = sum(r.get("reason") == "tracking_lost" for r in rows)
    # A rate across a missing or reversed clock interval would imply coverage
    # the journal cannot establish. Large forward gaps are excluded too.
    valid_span = sum(
        times[i] - times[i - 1] for i in range(1, len(rows)) if not boundaries[i]
    )

    def burst(first, last):
        section = rows[first:last + 1]
        counts = Counter(r.get("reason") for r in section)
        if not (counts["blurred"] or counts["tracking_lost"]):
            return None  # Degraded-only is never a pacing cue.
        sharp = [r["sharpness"] for r in section if _number(r.get("sharpness"))]
        parallax = [r["median_parallax_px"] for r in section if _number(r.get("median_parallax_px"))]
        return {
            "start_received_at": times[first], "end_received_at": times[last],
            "duration_seconds": _duration(times[first], times[last]),
            "frame_count": len(section), "reason_counts": dict(counts),
            "kind": "blur_and_loss" if counts["blurred"] and counts["tracking_lost"] else
                    "blur" if counts["blurred"] else "loss",
            "below_sharpness_floor_count": sum(value < sharpness_floor for value in sharp),
            "below_sharpness_floor_share": (
                sum(value < sharpness_floor for value in sharp) / len(sharp) if sharp else None
            ),
            "scored_count": len(sharp),
            "median_sharpness": median(sharp) if sharp else None,
            "median_parallax_px": median(parallax) if parallax else None,
            "_last_index": last,
        }

    bursts = []
    start = None
    for i, row in enumerate(rows):
        if start is not None and (boundaries[i] or row.get("reason") not in _BURST_REASONS):
            found = burst(start, i - 1)
            if found is not None:
                bursts.append(found)
            start = None
        if start is None and row.get("reason") in _BURST_REASONS:
            start = i
    if start is not None:
        found = burst(start, len(rows) - 1)
        if found is not None:
            bursts.append(found)

    def pause_after(last):
        first = end = None
        for i in range(last + 1, len(rows)):
            row = rows[i]
            if boundaries[i] or row.get("reason") in _BURST_REASONS:
                break
            if _run_eligible(row, min_overlap_ratio) and row.get("reason") != "blurred":
                if first is None:
                    first = i
                end = i
            elif first is not None:
                break
        return _duration(times[first], times[end]) if first is not None else None

    for item in bursts:
        item["pause_after_burst_seconds"] = pause_after(item.pop("_last_index"))
    bursts.sort(key=lambda item: (
        -(item["duration_seconds"] or 0), -item["frame_count"],
        item["start_received_at"] if item["start_received_at"] is not None else math.inf,
    ))

    held_runs = []
    first = last_sharp = None
    sharp_count = blur_count = overlap_count = 0

    def finish_run():
        if first is not None and last_sharp is not None and sharp_count >= 2 and overlap_count:
            held_runs.append({
                "start_received_at": times[first], "end_received_at": times[last_sharp],
                "duration_seconds": _duration(times[first], times[last_sharp]),
                "sharp_frame_count": sharp_count, "tolerated_blur_count": blur_count,
                "overlap_backed_count": overlap_count,
            })

    for i, row in enumerate(rows):
        hard_break = (boundaries[i] or row.get("reason") == "tracking_lost"
                      or row.get("tracker") == "no_reference" or times[i] is None)
        if hard_break:
            finish_run()
            first = last_sharp = None
            sharp_count = blur_count = overlap_count = 0
        if row.get("reason") == "tracking_lost" or row.get("tracker") == "no_reference" or times[i] is None:
            continue
        if _run_eligible(row, min_overlap_ratio) and row.get("reason") != "blurred":
            if first is None:
                first = i
            last_sharp = i
            sharp_count += 1
            overlap_count += bool(_number(row.get("overlap_ratio")) and row["overlap_ratio"] >= min_overlap_ratio)
        elif row.get("reason") == "blurred" and first is not None and blur_count < blur_tolerance:
            blur_count += 1
        else:
            finish_run()
            first = last_sharp = None
            sharp_count = blur_count = overlap_count = 0
    finish_run()
    held_runs.sort(key=lambda item: (-item["duration_seconds"], item["start_received_at"]))

    return {
        "row_count": len(rows), "reason_counts": dict(Counter(r.get("reason") for r in rows)),
        "gap_threshold_seconds": gap_seconds,
        "clock_gap_count": sum(boundaries[1:]),
        "observed_span_seconds": valid_span if intervals else None,
        "blur": {"refused_count": blurred, "scored_count": scored,
                 "share": blurred / scored if scored else None},
        "loss": {"count": lost, "per_minute": lost * 60 / valid_span if valid_span else None},
        "burst_count": len(bursts), "bursts": bursts[:top_k],
        "held_run_count": len(held_runs), "held_runs": held_runs[:top_k],
    }
