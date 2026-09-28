"""Offline frame-quality summary, with two declarations kept side by side.

v1 was fixed by Codex C4r at 00:48-00:52 EDT on 2026-09-28, using walk-4
data only, before walk 5's 04:43:42 recording start. v1.1 adds the MED-A
seed diagnostic and MED-B room-first mapping approved by manager 126 at
05:25 EDT, before anyone read walk 5's quality journal. Neither version is
selected by the CLI or calibrated from walk 5.
"""

from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median


SCHEMA = "glasses.world_builder.pacing_summary/1"
DECLARED_AT = {
    "v1": "2026-09-28 00:48-00:52 EDT",
    "v1_1": "2026-09-28 05:25 EDT",
}
BURST_REASONS = frozenset(("blurred", "tracking_degraded", "tracking_held", "tracking_lost"))


@dataclass(frozen=True)
class Parameters:
    """Frozen on walk-4 only before walk 5; manager review may revise for walk 6.

    gap_percentile and gap_multiplier: codex-pace 00:42, C4r 00:52, OPEN.
    blur_tolerance: codex-pace 00:42, C4r 00:52, OPEN.
    min_overlap_ratio: KeyframePolicy.min_overlap_ratio, selector.
    sharpness_floor: KeyframePolicy.min_sharpness, selector; diagnostic only.
    min_run_sharp_rows: codex-pace 00:42, OPEN.
    display_top_k and region_span_cap_s: C4r 00:52, OPEN.
    confident: C4r 00:52 plus review LOW-E, OPEN.
    """

    gap_percentile: float = .75
    gap_multiplier: float = 4.0
    blur_tolerance: int = 1
    min_overlap_ratio: float = .75
    sharpness_floor: float = 25.0
    min_run_sharp_rows: int = 2
    display_top_k: int = 6
    region_span_cap_s: float = 5.0
    confident: tuple[str, ...] = ("high", "med", "medium")


PARAMETERS = Parameters()


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _span(times, first, last):
    if first is None or last is None or times[first] is None or times[last] is None:
        return None
    return times[last] - times[first]


def _relative(times, index, origin):
    return times[index] - origin if times[index] is not None and origin is not None else None


def load_journal(path: str | Path):
    """Read one JSONL journal; skip torn and non-object lines, counting each."""
    rows = []
    skipped = 0
    with Path(path).open("rb") as handle:
        for raw_line in handle:
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeError):
                skipped += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                skipped += 1
    return rows, skipped


def load_regions(path: str | Path):
    """Read only the three permitted columns, discarding paths and notes."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"keyframe_id", "region", "confidence"} <= set(reader.fieldnames):
            raise ValueError("regions CSV lacks required columns")
        return [{key: item.get(key) for key in ("keyframe_id", "region", "confidence")}
                for item in reader]


def load_frames_observed(path: str | Path | None):
    if path is None or not Path(path).is_file():
        return None
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(value, dict):
        return None
    result = value.get("frames_observed")
    return result if isinstance(result, int) and not isinstance(result, bool) and result >= 0 else None


def _timing(rows):
    times = [r.get("received_at") if _finite(r.get("received_at")) else None for r in rows]
    intervals = [b - a for a, b in zip(times, times[1:])
                 if a is not None and b is not None and b > a]
    p75 = _percentile(intervals, PARAMETERS.gap_percentile)
    threshold = p75 * PARAMETERS.gap_multiplier if p75 is not None else None
    breaks = [True] * len(rows)
    for i in range(1, len(rows)):
        before, after = times[i - 1], times[i]
        breaks[i] = (before is None or after is None or after <= before or
                     (threshold is not None and after - before > threshold))
    valid = [t for t in times if t is not None]
    origin = valid[0] if valid else None
    clock = {
        "interval_median_s": median(intervals) if intervals else None,
        "interval_p75_s": p75,
        "gap_threshold_s": threshold,
        "max_interval_s": max(intervals) if intervals else None,
        "clock_break_count": sum(breaks[1:]),
        "observed_span_s": valid[-1] - valid[0] if len(valid) > 1 else (0.0 if valid else None),
    }
    return times, breaks, origin, clock


def _bursts(rows, times, breaks, origin):
    retained = []
    non_blur = Counter({"loss_only": 0, "degraded_only": 0, "mixed_no_blur": 0})

    def finish(first, last):
        if first is None:
            return
        section = rows[first:last + 1]
        reasons = Counter(r.get("reason") for r in section)
        if not reasons["blurred"]:
            if reasons["tracking_lost"] and (reasons["tracking_degraded"] or reasons["tracking_held"]):
                non_blur["mixed_no_blur"] += 1
            elif reasons["tracking_lost"]:
                non_blur["loss_only"] += 1
            else:
                non_blur["degraded_only"] += 1
            return
        sharpness = [r["sharpness"] for r in section if _finite(r.get("sharpness"))]
        parallax = [r["median_parallax_px"] for r in section if _finite(r.get("median_parallax_px"))]
        retained.append({
            "_first": first, "_last": last,
            "start_s": _relative(times, first, origin),
            "end_s": _relative(times, last, origin),
            "duration_s": _span(times, first, last),
            "rows": len(section), "reason_counts": dict(reasons),
            "below_floor_count": sum(value < PARAMETERS.sharpness_floor for value in sharpness),
            "median_sharpness": median(sharpness) if sharpness else None,
            "median_parallax_px": median(parallax) if parallax else None,
        })

    first = None
    for i, row in enumerate(rows):
        if first is not None and (breaks[i] or row.get("reason") not in BURST_REASONS):
            finish(first, i - 1)
            first = None
        if first is None and row.get("reason") in BURST_REASONS:
            first = i
    finish(first, len(rows) - 1)
    retained.sort(key=lambda b: (-(b["duration_s"] if b["duration_s"] is not None else -1),
                                 -b["rows"], b["start_s"] if b["start_s"] is not None else math.inf))
    for rank, item in enumerate(retained, 1):
        item["rank"] = rank
    return retained, dict(non_blur)


def _sharp_runs(rows, times, breaks, origin):
    runs = []
    first = last_sharp = None
    sharp_count = tolerated = pending = overlap_count = 0

    def finish():
        if (first is not None and last_sharp is not None and
                sharp_count >= PARAMETERS.min_run_sharp_rows and overlap_count):
            runs.append({"_first": first, "_last": last_sharp,
                         "start_s": _relative(times, first, origin),
                         "end_s": _relative(times, last_sharp, origin),
                         "duration_s": _span(times, first, last_sharp),
                         "sharp_rows": sharp_count, "tolerated_blur_rows": tolerated})

    for i, row in enumerate(rows):
        if breaks[i] and i:
            finish()
            first = last_sharp = None
            sharp_count = tolerated = pending = overlap_count = 0
        reason = row.get("reason")
        hard_break = reason == "tracking_lost" or row.get("tracker") == "no_reference" or times[i] is None
        overlap = row.get("overlap_ratio")
        overlap_backed = _finite(overlap) and overlap >= PARAMETERS.min_overlap_ratio
        eligible = (row.get("tracker") == "reference" and row.get("sharpness") is not None and
                    reason != "blurred" and (row.get("outcome") == "accept" or overlap_backed))
        if not hard_break and eligible:
            if first is None:
                first = i
            tolerated += pending
            pending = 0
            last_sharp = i
            sharp_count += 1
            overlap_count += int(overlap_backed)
        elif not hard_break and reason == "blurred" and first is not None and tolerated + pending < PARAMETERS.blur_tolerance:
            pending += 1
        else:
            finish()
            first = last_sharp = None
            sharp_count = tolerated = pending = overlap_count = 0
    finish()
    runs.sort(key=lambda r: (-(r["duration_s"] if r["duration_s"] is not None else -1),
                             r["start_s"] if r["start_s"] is not None else math.inf))
    return runs


def _public(item):
    return {key: value for key, value in item.items() if not key.startswith("_")}


def _following(item, bursts, runs, breaks):
    later = [b["_first"] for b in bursts if b["_first"] > item["_last"]]
    next_burst = min(later) if later else len(breaks)
    for run in sorted(runs, key=lambda r: r["_first"]):
        if run["_first"] <= item["_last"] or run["_first"] >= next_burst:
            continue
        if not any(breaks[item["_last"] + 1:run["_first"] + 1]):
            return _public(run)
    return None


def _seed_groups(rows, times, breaks, origin, bursts, runs):
    ordered = sorted(bursts, key=lambda b: b["_first"])
    chains = []
    chain = []
    for item in ordered:
        if chain and not (item["_first"] == chain[-1]["_last"] + 2 and
                          rows[item["_first"] - 1].get("reason") == "session_seed" and
                          not any(breaks[chain[-1]["_last"] + 1:item["_first"] + 1])):
            if len(chain) > 1:
                chains.append(chain)
            chain = []
        chain.append(item)
    if len(chain) > 1:
        chains.append(chain)
    groups = []
    for chain in chains:
        first, last = chain[0]["_first"], chain[-1]["_last"]
        item = {"_first": first, "_last": last,
                "start_s": _relative(times, first, origin), "end_s": _relative(times, last, origin),
                "duration_s": _span(times, first, last), "rows": last - first + 1,
                "reason_counts": dict(Counter(r.get("reason") for r in rows[first:last + 1])),
                "member_burst_ranks": [b["rank"] for b in chain]}
        item["following_run"] = _following(item, bursts, runs, breaks)
        groups.append(_public(item))
    return groups


def _session_prefix(keyframe_id):
    if not isinstance(keyframe_id, str) or ":" not in keyframe_id:
        return None
    return keyframe_id.split(":", 1)[0]


def _region_assignments(rows, times, regions, room_map, room_first):
    empty = {"status": "not_supplied", "room_map": room_map or {}, "buckets": {}}
    if regions is None:
        return [None] * len(rows), None, empty, []
    labels = {r.get("keyframe_id"): r for r in regions if isinstance(r, dict)}
    csv_prefixes = {_session_prefix(key) for key in labels}
    journal_prefixes = {_session_prefix(r.get("keyframe_id")) for r in rows if r.get("keyframe_id")}
    if (len(csv_prefixes) != 1 or len(journal_prefixes) != 1 or
            None in csv_prefixes or csv_prefixes != journal_prefixes):
        return [None] * len(rows), None, {**empty, "status": "session_mismatch"}, []
    labelled_indices = [i for i, r in enumerate(rows) if r.get("keyframe_id") in labels]
    labelled_set = set(labelled_indices)

    def entry(index):
        value = labels[rows[index]["keyframe_id"]]
        return {"label": value.get("region"), "confidence": value.get("confidence")}

    def room(label):
        return room_map.get(label, "other_labelled") if room_map else "other_labelled"

    def bracket(first, last):
        before_pos = bisect_left(labelled_indices, first) - 1
        after_pos = bisect_right(labelled_indices, last)
        if before_pos < 0 or after_pos >= len(labelled_indices):
            return "unknown", None
        before, after = labelled_indices[before_pos], labelled_indices[after_pos]
        left, right = entry(before), entry(after)
        span = times[after] - times[before] if times[before] is not None and times[after] is not None else None
        evidence = {"before": left, "after": right, "span_s": span}
        confident = (left["confidence"] in PARAMETERS.confident and
                     right["confidence"] in PARAMETERS.confident)
        if not confident or span is None or span < 0 or span > PARAMETERS.region_span_cap_s:
            return "unknown", evidence
        if room_first:
            same = room(left["label"]) == room(right["label"])
        else:
            same = left["label"] == right["label"]
        return (room(left["label"]) if same else "between"), evidence

    assignments = []
    for i, row in enumerate(rows):
        if i in labelled_set:
            label = entry(i)
            assignments.append(room(label["label"]) if label["confidence"] in PARAMETERS.confident else "unknown")
        elif row.get("keyframe_id"):
            assignments.append("unknown")
        else:
            assignments.append(bracket(i, i)[0])
    counters = {name: {"refused": 0, "scored": 0, "share": None}
                for name in dict.fromkeys([*room_map.values(), "other_labelled", "between", "unknown"])}
    for row, bucket in zip(rows, assignments):
        counters[bucket]["refused"] += int(row.get("reason") == "blurred")
        counters[bucket]["scored"] += int(row.get("sharpness") is not None)
    for value in counters.values():
        if value["scored"]:
            value["share"] = value["refused"] / value["scored"]
    return assignments, bracket, {"status": "joined", "room_map": room_map or {}, "buckets": counters}, labelled_indices


def summarize(rows, *, skipped=0, frames_observed=None, regions=None, room_map=None):
    """Summarize a single journal in order; never choose a version at runtime."""
    rows = list(rows)
    room_map = dict(room_map or {})
    if not all(isinstance(r, dict) for r in rows):
        raise ValueError("journal rows must be objects")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in room_map.items()):
        raise ValueError("room map must contain string labels and rooms")
    times, breaks, origin, clock = _timing(rows)
    bursts, non_blur = _bursts(rows, times, breaks, origin)
    runs = _sharp_runs(rows, times, breaks, origin)
    for item in bursts:
        item["following_run"] = _following(item, bursts, runs, breaks)
    reason_counts = dict(Counter(r.get("reason") for r in rows))
    scored = sum(r.get("sharpness") is not None for r in rows)
    blurred = reason_counts.get("blurred", 0)
    input_info = {"rows_parsed": len(rows), "rows_skipped": skipped,
                  "completeness": "verified" if frames_observed == len(rows) else "unverified",
                  "frames_observed": frames_observed}
    result = {"schema": SCHEMA, "status": "available" if scored else "unavailable",
              "declared_at": DECLARED_AT,
              "parameters": {**asdict(PARAMETERS), "confident": list(PARAMETERS.confident)}}
    for name, room_first in (("v1", False), ("v1_1", True)):
        assignments, bracket, region_info, labelled = _region_assignments(
            rows, times, regions, room_map, room_first)
        displayed = []
        for item in bursts:
            public = _public(item)
            if bracket is not None:
                assert not any(i in labelled for i in range(item["_first"], item["_last"] + 1)), "burst holds keyframe"
                public["bucket"], public["bracket"] = bracket(item["_first"], item["_last"])
            else:
                public["bucket"], public["bracket"] = None, None
            displayed.append(public)
        result[name] = {
            "status": "available" if scored else "unavailable",
            "input": input_info.copy(), "clock": clock.copy(), "reason_counts": reason_counts.copy(),
            "blur": {"refused_count": blurred, "scored_count": scored,
                     "share": blurred / scored if scored else None},
            "bursts": {"retained_count": len(bursts), "non_blur_stretches": non_blur.copy(),
                       "top": displayed[:PARAMETERS.display_top_k], "all": displayed[:200]},
            "sharp_runs": {"count": len(runs), "longest": [_public(r) for r in runs[:20]]},
            "regions": region_info,
        }
        if name == "v1_1":
            result[name]["seed_joined_groups"] = _seed_groups(rows, times, breaks, origin, bursts, runs)
    return result


def _minute_second(value):
    if value is None:
        return "?:??"
    total = int(value + .5)
    return f"{total // 60}:{total % 60:02d}"


def _sentences(summary, text_runs):
    if summary["status"] == "unavailable":
        return ["Blur summary unavailable for this walk."]
    count = summary["blur"]
    if summary["input"]["completeness"] == "verified":
        lines = [f"During this walk, the Tower checked {count['scored_count']} frames and refused {count['refused_count']} of them as blurry."]
    else:
        lines = [f"In this walk's frame log, {count['refused_count']} of {count['scored_count']} checked frames were refused as blurry."]
    for burst in summary["bursts"]["top"]:
        start, end = _minute_second(burst["start_s"]), _minute_second(burst["end_s"])
        duration = f"{burst['duration_s']:.1f}" if burst["duration_s"] is not None else "?"
        bucket = burst["bucket"]
        if bucket and bucket not in ("between", "unknown", "other_labelled"):
            lines.append(f"Blurry frames near {bucket}, {start} to {end} into the walk ({duration} seconds).")
        elif bucket == "between":
            lines.append(f"Blurry frames, {start} to {end} into the walk ({duration} seconds). The frames just before and after show different rooms.")
        else:
            lines.append(f"Blurry frames, {start} to {end} into the walk ({duration} seconds). The room is unknown.")
        following = burst["following_run"]
        if text_runs and following is not None:
            lines.append(f"{following['sharp_rows']} sharp frames followed, over {following['duration_s']:.1f} seconds.")
    if summary["regions"]["status"] == "joined":
        for room, values in summary["regions"]["buckets"].items():
            if room not in ("between", "unknown", "other_labelled") and values["scored"]:
                lines.append(f"In frames assigned to {room}, {values['refused']} of {values['scored']} checked frames were refused as blurry.")
        buckets = summary["regions"]["buckets"]
        lines.append(f"{buckets['between']['refused']} blurry frames came between frames showing different rooms; no room was assigned.")
        unassigned = buckets["unknown"]["refused"] + buckets["other_labelled"]["refused"]
        lines.append(f"{unassigned} blurry frames could not be assigned a room.")
    return lines


def render_text(result, *, text_runs=False):
    """Parallel v1/v1.1 columns; each nonempty cell is a closed template."""
    left = _sentences(result["v1"], text_runs)
    right = _sentences(result["v1_1"], text_runs)
    return "\n".join(["v1\tv1.1", *(f"{a}\t{b}" for a, b in zip(left + [""] * (max(len(left), len(right)) - len(left)),
                                                              right + [""] * (max(len(left), len(right)) - len(right))))]) + "\n"
