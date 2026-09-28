"""Frozen offline pacing rules; the optional walk-4 replay is never live input."""

from __future__ import annotations

import copy
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tower.world_builder.pacing_summary import (
    _region_assignments, _sharp_runs, _timing, load_frames_observed, load_journal, load_regions,
    render_text, summarize,
)


RUN = Path(r"C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23")
WALK4_REASONS = RUN / "lead/diag4/reasons_walk4.json"
WALK4_REGIONS = RUN / "physical-test/regions-w4/c81766a3bd3a4eeb9739aa4271fdbb75_regions.csv"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/world_pacing_summary.py"
ROOM_MAP = {"desk": "bedroom", "bed": "bedroom", "closet": "closet", "bathroom": "bathroom"}


def row(t, reason="insufficient_motion", *, tracker="reference", sharp=100.0,
        overlap=.9, outcome="skip", keyframe_id=None, segment=0, parallax=2.0):
    return {"received_at": t, "reason": reason, "tracker": tracker,
            "sharpness": sharp, "overlap_ratio": overlap, "outcome": outcome,
            "median_parallax_px": parallax, "keyframe_id": keyframe_id,
            "segment_index": segment}


def version(rows, name="v1_1", **kwargs):
    return summarize(rows, **kwargs)[name]


def test_jitter_gap_is_four_times_linear_p75_and_segment_does_not_split():
    rows = [row(t, segment=i % 2) for i, t in enumerate([10, 10.08, 10.18, 10.30, 10.42])]
    result = version(rows)
    assert result["clock"]["interval_p75_s"] == pytest.approx(.12)
    assert result["clock"]["gap_threshold_s"] == pytest.approx(.48)
    assert result["clock"]["clock_break_count"] == 0
    assert result["sharp_runs"]["longest"][0]["duration_s"] == pytest.approx(.42)
    assert result["sharp_runs"]["longest"][0]["start_s"] == 0


def test_reconnect_breaks_bursts_and_runs():
    rows = [row(0), row(.1, "blurred"), row(.2, "blurred"), row(.3), row(.4),
            row(21.4, "blurred"), row(21.5, "blurred"), row(21.6), row(21.7)]
    result = version(rows)
    assert result["clock"]["clock_break_count"] == 1
    assert result["bursts"]["retained_count"] == 2
    assert all(b["duration_s"] <= .11 for b in result["bursts"]["all"])
    assert all(r["duration_s"] < 1 for r in result["sharp_runs"]["longest"])


# Review C4i MED-1: in each journal below the 21 s reconnect gap falls where nothing
# else would split, so only the clock-break rule can.  The 0.1 s rows around it keep
# p75 at 0.1 s, so the gap threshold is 0.4 s.
def _by_start(items):
    return sorted(items, key=lambda item: item["start_s"])


def test_reconnect_gap_inside_a_burst_splits_it():
    rows = [row(0), row(.1), row(.2), row(.3, "blurred"), row(.4, "blurred"),
            row(21.4, "blurred"), row(21.5, "blurred"), row(21.6), row(21.7), row(21.8)]
    result = version(rows)
    assert result["clock"]["clock_break_count"] == 1
    assert result["bursts"]["retained_count"] == 2
    bursts = _by_start(result["bursts"]["all"])
    assert [(b["start_s"], b["end_s"], b["rows"]) for b in bursts] == [
        pytest.approx((.3, .4, 2)), pytest.approx((21.4, 21.5, 2))]


def test_reconnect_gap_inside_a_run_splits_it_and_the_next_row_starts_fresh():
    rows = [row(0), row(.1), row(.2), row(.3), row(21.3), row(21.4), row(21.5), row(21.6)]
    runs = _by_start(version(rows)["sharp_runs"]["longest"])
    assert [(r["start_s"], r["end_s"], r["sharp_rows"], r["tolerated_blur_rows"]) for r in runs] == [
        pytest.approx((0, .3, 4, 0)), pytest.approx((21.3, 21.6, 4, 0))]
    # A pending tolerated blur is not carried across the gap either.
    rows = [row(0), row(.1), row(.2), row(.3, "blurred"), row(21.3), row(21.4), row(21.5)]
    runs = _by_start(version(rows)["sharp_runs"]["longest"])
    assert [(r["start_s"], r["sharp_rows"], r["tolerated_blur_rows"]) for r in runs] == [
        pytest.approx((0, 3, 0)), pytest.approx((21.3, 3, 0))]


def test_following_run_stops_at_a_reconnect_gap():
    def journal(resume):
        return [row(0), row(.1), row(.2, "blurred"), row(.3, "blurred"),
                row(resume), row(resume + .1), row(resume + .2)]
    control = version(journal(.4))["bursts"]["all"][0]["following_run"]
    assert control["start_s"] == pytest.approx(.4) and control["sharp_rows"] == 3
    result = version(journal(21.3))
    assert result["clock"]["clock_break_count"] == 1
    assert [r["start_s"] for r in _by_start(result["sharp_runs"]["longest"])] == pytest.approx([0, 21.3])
    assert result["bursts"]["all"][0]["following_run"] is None


@pytest.mark.parametrize("seed_at, blur_at", [(21.3, 21.4), (.4, 21.4)], ids=["gap-before-seed", "gap-after-seed"])
def test_no_seed_group_forms_across_a_reconnect_gap(seed_at, blur_at):
    def journal(seed_t, blur_t):
        return [row(0), row(.1), row(.2, "blurred"), row(.3, "tracking_lost", tracker="no_reference"),
                row(seed_t, "session_seed", tracker="no_reference"), row(blur_t, "blurred"),
                row(blur_t + .1), row(blur_t + .2)]
    assert len(summarize(journal(.4, .5))["v1_1"]["seed_joined_groups"]) == 1
    summary = summarize(journal(seed_at, blur_at))
    assert summary["v1_1"]["clock"]["clock_break_count"] == 1
    assert summary["v1_1"]["bursts"]["retained_count"] == 2
    assert summary["v1_1"]["seed_joined_groups"] == []


def test_loss_stays_inside_burst_and_seed_creates_diagnostic_group():
    rows = [row(0, "blurred"), row(.1, "tracking_lost", tracker="no_reference"),
            row(.2, "blurred", segment=9), row(.3, "session_seed", tracker="no_reference"),
            row(.4, "blurred"), row(.5, "tracking_lost", tracker="no_reference"),
            row(.6, "blurred"), row(.7), row(.8)]
    summary = summarize(rows)
    for name in ("v1", "v1_1"):
        bursts = sorted(summary[name]["bursts"]["all"], key=lambda b: b["start_s"])
        assert len(bursts) == 2
        assert bursts[0]["reason_counts"] == {"blurred": 2, "tracking_lost": 1}
        assert bursts[0]["rows"] == 3
    assert "seed_joined_groups" not in summary["v1"]
    groups = summary["v1_1"]["seed_joined_groups"]
    assert len(groups) == 1
    assert groups[0]["rows"] == 7
    assert groups[0]["reason_counts"] == {"blurred": 4, "tracking_lost": 2, "session_seed": 1}
    assert len(groups[0]["member_burst_ranks"]) == 2


def test_non_blur_stretches_are_counted_and_unscored_rows_bound_bursts():
    rows = [row(0, "tracking_lost"), row(.1), row(.2, "tracking_degraded"),
            row(.3, "tracking_held"), row(.4), row(.5, "tracking_lost"),
            row(.6, "tracking_degraded"), row(.7), row(.8, "blurred"),
            row(.9, "malformed_frame", sharp=None), row(1, "blurred"),
            row(1.1, "frame_size_changed", sharp=None)]
    result = version(rows)
    assert result["bursts"]["non_blur_stretches"] == {
        "loss_only": 1, "degraded_only": 1, "mixed_no_blur": 1}
    assert result["bursts"]["retained_count"] == 2
    assert result["blur"] == {"refused_count": 2, "scored_count": 10, "share": .2}
    assert "tracking_degraded" not in render_text(summarize(rows))


def test_sharp_run_absorbs_one_blur_then_second_or_no_reference_ends_it():
    rows = [row(0), row(.1, "blurred"), row(.2), row(.3, "blurred"),
            row(.4, "blurred"), row(.5), row(.6), row(.7, tracker="no_reference"),
            row(.8), row(.9)]
    runs = sorted(version(rows)["sharp_runs"]["longest"], key=lambda r: r["start_s"])
    assert [(r["sharp_rows"], r["tolerated_blur_rows"]) for r in runs] == [(2, 1), (2, 0), (2, 0)]
    assert runs[0]["duration_s"] == pytest.approx(.2)


def test_trailing_blur_is_not_reported_as_absorbed_inside_a_run():
    runs = version([row(0), row(.1), row(.2, "blurred")])["sharp_runs"]["longest"]
    assert runs[0]["duration_s"] == pytest.approx(.1)
    assert runs[0]["tolerated_blur_rows"] == 0


def test_following_run_is_null_when_next_burst_arrives_first():
    rows = [row(0, "blurred"), row(.1, tracker="no_reference"),
            row(.2, "blurred"), row(.3), row(.4)]
    bursts = sorted(version(rows)["bursts"]["all"], key=lambda b: b["start_s"])
    assert bursts[0]["following_run"] is None
    assert bursts[1]["following_run"]["sharp_rows"] == 2


def test_loader_skips_torn_tail_and_marks_missing_session_unverified(tmp_path):
    journal = tmp_path / "frames_quality.jsonl"
    journal.write_text(json.dumps(row(1790000000.0, "blurred")) + "\n{" + "\n", encoding="utf-8")
    rows, skipped = load_journal(journal)
    summary = summarize(rows, skipped=skipped)
    assert summary["v1"]["input"] == {
        "rows_parsed": 1, "rows_skipped": 1, "completeness": "unverified", "frames_observed": None}
    assert "In this walk's frame log" in render_text(summary)
    assert "1790000000" not in json.dumps(summary) + render_text(summary)


def test_loader_counts_a_torn_utf8_tail(tmp_path):
    journal = tmp_path / "frames_quality.jsonl"
    journal.write_bytes((json.dumps(row(0)) + "\n").encode("utf-8") + b'{"bad":"\xff')
    rows, skipped = load_journal(journal)
    assert len(rows) == 1 and skipped == 1


def test_malformed_session_json_cannot_verify_completeness(tmp_path):
    session = tmp_path / "session.json"
    session.write_text("{", encoding="utf-8")
    assert load_frames_observed(session) is None


def test_no_scored_rows_has_unavailable_json_and_text():
    result = summarize([row(0, "malformed_frame", sharp=None)])
    assert result["status"] == "unavailable"
    assert result["v1"]["status"] == result["v1_1"]["status"] == "unavailable"
    assert render_text(result) == "v1\tv1.1\nBlur summary unavailable for this walk.\tBlur summary unavailable for this walk.\n"


def test_regions_require_session_and_compare_rooms_after_mapping():
    a, b, c = "same:00000001", "same:00000002", "same:00000003"
    rows = [row(0, "session_seed", keyframe_id=a), row(.1, "blurred"),
            row(.2, "parallax", keyframe_id=b), row(.3, "blurred"),
            row(5.4, "parallax", keyframe_id=c)]
    labels = [{"keyframe_id": a, "region": "desk", "confidence": "high"},
              {"keyframe_id": b, "region": "bed", "confidence": "med"},
              {"keyframe_id": c, "region": "closet", "confidence": "high"}]
    summary = summarize(rows, regions=labels, room_map=ROOM_MAP)
    assert summary["v1"]["regions"]["buckets"]["between"]["refused"] == 1
    assert summary["v1_1"]["regions"]["buckets"]["bedroom"]["refused"] == 1
    assert summary["v1_1"]["regions"]["buckets"]["unknown"]["refused"] == 1
    first = sorted(summary["v1_1"]["bursts"]["all"], key=lambda x: x["start_s"])[0]
    assert first["bucket"] == "bedroom"
    assert first["bracket"]["before"]["label"] == "desk"
    assert first["bracket"]["after"]["confidence"] == "med"
    assert summary["v1_1"]["regions"]["status"] == "joined"
    labels[2]["keyframe_id"] = "other:00000003"
    mismatch = version(rows, regions=labels, room_map=ROOM_MAP)
    assert mismatch["regions"]["status"] == "session_mismatch"
    assert mismatch["regions"]["buckets"] == {}
    assert all(b["bucket"] is None for b in mismatch["bursts"]["all"])


def _columns(rendered):
    cells = [line.split("\t") for line in rendered.splitlines()]
    assert cells[0] == ["v1", "v1.1"] and all(len(c) == 2 for c in cells)
    return [c[0] for c in cells[1:]], [c[1] for c in cells[1:]]


def test_v1_text_never_claims_different_rooms():
    # Review C4i MED-2 fix (a): v1 compares raw labels, so desk/bed is "between" in v1
    # although both are the bedroom.  Its text uses B3 and R3 only; the JSON is unchanged.
    a, b, c = "same:00000001", "same:00000002", "same:00000003"
    rows = [row(0, "parallax", keyframe_id=a), row(.1, "blurred"), row(.2, "blurred"),
            row(.3, "parallax", keyframe_id=b), row(.4, "blurred"),
            row(.5, "parallax", keyframe_id=c)]
    labels = [{"keyframe_id": a, "region": "desk", "confidence": "high"},
              {"keyframe_id": b, "region": "bed", "confidence": "high"},
              {"keyframe_id": c, "region": "closet", "confidence": "high"}]
    summary = summarize(rows, regions=labels, room_map=ROOM_MAP, frames_observed=len(rows))
    assert [x["bucket"] for x in summary["v1"]["bursts"]["top"]] == ["between", "between"]
    assert [x["bucket"] for x in summary["v1_1"]["bursts"]["top"]] == ["bedroom", "between"]
    assert summary["v1"]["regions"]["buckets"]["between"]["refused"] == 3
    v1, v11 = _columns(render_text(summary))
    assert not any("different rooms" in cell for cell in v1)
    assert v1[1:3] == ["Blurry frames, 0:00 to 0:00 into the walk (0.1 seconds). The room is unknown.",
                       "Blurry frames, 0:00 to 0:00 into the walk (0.0 seconds). The room is unknown."]
    assert v11[1:3] == ["Blurry frames near bedroom, 0:00 to 0:00 into the walk (0.1 seconds).",
                        "Blurry frames, 0:00 to 0:00 into the walk (0.0 seconds). The frames just before and after show different rooms."]
    assert v1[-2:] == ["", "3 blurry frames could not be assigned a room."]
    assert v11[-2:] == ["1 blurry frames came between frames showing different rooms; no room was assigned.",
                        "0 blurry frames could not be assigned a room."]


def test_v1_1_text_calls_only_room_to_room_brackets_different_rooms():
    # OPEN (g), C4i-G: in v1.1 a "between" bracket with exactly one label outside the room
    # map (closet / other, other / bathroom) is not two rooms.  It renders B3 and counts in
    # R3, with existing sentences only; the room/room bracket (desk / closet) keeps B2 and R2.
    a, b, c, d = (f"same:{i:08d}" for i in range(1, 5))
    rows = [row(0, "parallax", keyframe_id=a), row(.1, "blurred"), row(.2, "blurred"),
            row(.3, "parallax", keyframe_id=b), row(.4, "blurred"), row(.5, "blurred"), row(.6, "blurred"),
            row(.7, "parallax", keyframe_id=c), row(.8, "blurred"), row(.9, "parallax", keyframe_id=d)]
    labels = [{"keyframe_id": a, "region": "desk", "confidence": "high"},
              {"keyframe_id": b, "region": "closet", "confidence": "high"},
              {"keyframe_id": c, "region": "other", "confidence": "high"},
              {"keyframe_id": d, "region": "bathroom", "confidence": "high"}]
    summary = summarize(rows, regions=labels, room_map=ROOM_MAP, frames_observed=len(rows))
    # The JSON is unchanged: all three bursts are still "between" in both versions.
    for name in ("v1", "v1_1"):
        assert [x["bucket"] for x in summary[name]["bursts"]["top"]] == ["between"] * 3
        assert summary[name]["regions"]["buckets"]["between"]["refused"] == 6
    v1, v11 = _columns(render_text(summary))
    assert v11[1:4] == [
        "Blurry frames, 0:00 to 0:01 into the walk (0.2 seconds). The room is unknown.",
        "Blurry frames, 0:00 to 0:00 into the walk (0.1 seconds). The frames just before and after show different rooms.",
        "Blurry frames, 0:01 to 0:01 into the walk (0.0 seconds). The room is unknown."]
    assert v11[-2:] == ["2 blurry frames came between frames showing different rooms; no room was assigned.",
                        "4 blurry frames could not be assigned a room."]
    assert all(cell.endswith("The room is unknown.") for cell in v1[1:4])
    assert v1[-2:] == ["", "6 blurry frames could not be assigned a room."]


def test_v1_1_between_folds_into_r3_when_the_bursts_cannot_account_for_it():
    # OPEN (g) fallback, part 1: a blurred row carrying its own keyframe id that the CSV
    # lacks is "unknown" while its burst is "between", so the bursts cannot split the
    # bucket exactly; all of v1.1's "between" goes to R3 and R2 is left empty.
    a, missing, c = "same:00000001", "same:00000002", "same:00000003"
    rows = [row(0, "parallax", keyframe_id=a), row(.1, "blurred"),
            row(.2, "blurred", keyframe_id=missing), row(.3, "parallax", keyframe_id=c)]
    labels = [{"keyframe_id": a, "region": "desk", "confidence": "high"},
              {"keyframe_id": c, "region": "other", "confidence": "high"}]
    summary = summarize(rows, regions=labels, room_map=ROOM_MAP)
    buckets = summary["v1_1"]["regions"]["buckets"]
    assert (buckets["between"]["refused"], buckets["unknown"]["refused"]) == (1, 1)
    assert summary["v1_1"]["bursts"]["top"][0]["reason_counts"]["blurred"] == 2
    _, v11 = _columns(render_text(summary))
    assert v11[-2:] == ["", "2 blurry frames could not be assigned a room."]


def test_v1_1_between_folds_into_r3_past_the_200_burst_cap():
    # OPEN (g) fallback, part 2: bursts.all is capped at 200, so past it the split is not
    # known; all of v1.1's "between" goes to R3, while a room/room burst in the top six
    # still renders B2.
    a, b = "same:00000001", "same:00000002"
    rows = [row(0, "parallax", keyframe_id=a), row(.1, "blurred"), row(.2, "blurred"),
            row(.3, "parallax", keyframe_id=b)]
    for i in range(205):
        t = .4 + i * .4
        rows += [row(t, "blurred"), row(t + .1, tracker="no_reference"), row(t + .2), row(t + .3)]
    labels = [{"keyframe_id": a, "region": "desk", "confidence": "high"},
              {"keyframe_id": b, "region": "closet", "confidence": "high"}]
    summary = summarize(rows, regions=labels, room_map=ROOM_MAP)
    v11 = summary["v1_1"]
    assert v11["bursts"]["retained_count"] == 206 and len(v11["bursts"]["all"]) == 200
    assert v11["regions"]["buckets"]["between"]["refused"] == 2
    _, right = _columns(render_text(summary))
    assert right[1].endswith("The frames just before and after show different rooms.")
    assert right[-2:] == ["", "207 blurry frames could not be assigned a room."]


def test_a_burst_without_a_time_gets_no_line():
    # Review LOW-5: never a B line with "?:??"; the engine always sets received_at.
    rows = [row(0), row(.1, "blurred"), row(.2), row(None, "blurred"), row(.3), row(.4)]
    summary = summarize(rows)
    assert summary["v1_1"]["bursts"]["retained_count"] == 2
    rendered = render_text(summary)
    assert "?" not in rendered
    v1, v11 = _columns(rendered)
    assert [cell for cell in v1 + v11 if cell.startswith("Blurry frames")] == [
        "Blurry frames, 0:00 to 0:00 into the walk (0.0 seconds). The room is unknown."] * 2


def test_keyframe_check_is_a_set_lookup():
    # Review LOW-6: "a burst holds no keyframe" looks keyframes up in a set, not a list.
    a = "same:00000001"
    rows = [row(0, "parallax", keyframe_id=a), row(.1, "blurred")]
    labels = [{"keyframe_id": a, "region": "desk", "confidence": "high"}]
    times = _timing(rows)[0]
    assert isinstance(_region_assignments(rows, times, labels, ROOM_MAP, True)[3], (set, frozenset))


def test_cli_refuses_a_malformed_regions_csv_in_one_line(tmp_path):
    # Review LOW-6: csv.Error (here a field over the csv module's 131072-char limit) is one
    # line on stderr and exit 1, like ValueError, not a traceback.
    journal = tmp_path / "input" / "frames_quality.jsonl"
    journal.parent.mkdir()
    journal.write_text(json.dumps(row(0, "parallax", keyframe_id="same:00000001")) + "\n", encoding="utf-8")
    regions = tmp_path / "regions.csv"
    regions.write_text("keyframe_id,region,confidence\nsame:00000001," + "x" * 140000 + ",high\n",
                       encoding="utf-8")
    room_map = tmp_path / "room_map.json"
    room_map.write_text(json.dumps(ROOM_MAP), encoding="utf-8")
    output = tmp_path / "output"
    refused = _run_cli(journal, output, "--regions", regions, "--room-map", room_map)
    assert refused.returncode == 1
    assert len(refused.stderr.splitlines()) == 1 and "field larger than field limit" in refused.stderr
    assert not output.exists()


@pytest.mark.parametrize("reserved", ["unknown", "between", "other_labelled"])
def test_reserved_bucket_names_are_refused_as_rooms(reserved):
    a = "same:00000001"
    rows = [row(0, "parallax", keyframe_id=a), row(.1, "blurred")]
    labels = [{"keyframe_id": a, "region": "desk", "confidence": "high"}]
    with pytest.raises(ValueError, match="reserved"):
        summarize(rows, regions=labels, room_map={**ROOM_MAP, "desk": reserved})


def test_cli_refuses_a_reserved_room_in_one_line(tmp_path):
    journal = tmp_path / "input" / "frames_quality.jsonl"
    journal.parent.mkdir()
    journal.write_text(json.dumps(row(0, "parallax", keyframe_id="same:00000001")) + "\n", encoding="utf-8")
    regions = tmp_path / "regions.csv"
    regions.write_text("keyframe_id,region,confidence\nsame:00000001,desk,high\n", encoding="utf-8")
    room_map = tmp_path / "room_map.json"
    room_map.write_text(json.dumps({"desk": "between"}), encoding="utf-8")
    output = tmp_path / "output"
    refused = _run_cli(journal, output, "--regions", regions, "--room-map", room_map)
    assert refused.returncode != 0
    assert len(refused.stderr.splitlines()) == 1 and "reserved" in refused.stderr
    assert not output.exists()


# Review C4i LOW-2: one assertion per rule that a surviving mutant showed unguarded.
def test_gap_threshold_is_strict_at_equality():
    # Binary-exact times: p75 = 0.125 s, so the threshold is exactly 0.5 s.
    base = [0, .125, .25, .375, .5]
    equal = version([row(t) for t in base + [1.0, 1.125, 1.25, 1.375]])
    assert equal["clock"]["gap_threshold_s"] == .5 and equal["clock"]["clock_break_count"] == 0
    above = version([row(t) for t in base + [1.03125, 1.15625, 1.28125, 1.40625]])
    assert above["clock"]["gap_threshold_s"] == .5 and above["clock"]["clock_break_count"] == 1


def test_run_needs_an_overlap_backed_row():
    accepted = [row(t, "parallax", outcome="accept", overlap=.5) for t in (0, .1, .2)]
    assert version(accepted)["sharp_runs"]["count"] == 0
    accepted[1]["overlap_ratio"] = .75
    assert version(accepted)["sharp_runs"]["count"] == 1


def test_tracking_lost_ends_a_run_even_with_a_reference_tracker():
    rows = [row(0), row(.1), row(.2, "tracking_lost", tracker="reference"), row(.3), row(.4)]
    runs = _by_start(version(rows)["sharp_runs"]["longest"])
    assert [(r["start_s"], r["sharp_rows"]) for r in runs] == [pytest.approx((0, 2)), pytest.approx((.3, 2))]


def test_burst_rank_ties_break_on_rows_then_earlier_start():
    # Binary-exact times, so the three durations are exactly equal (0.125 s).
    rows = [row(0), row(.125, "blurred"), row(.25, "blurred"), row(.375), row(.5),
            row(.625, "blurred"), row(.75, "blurred"), row(.875), row(1.0),
            row(1.125, "blurred"), row(1.1875, "blurred"), row(1.25, "blurred"), row(1.375)]
    bursts = version(rows)["bursts"]["all"]
    assert [(b["rank"], b["start_s"], b["rows"], b["duration_s"]) for b in bursts] == [
        (1, 1.125, 3, .125), (2, .125, 2, .125), (3, .625, 2, .125)]


def test_completeness_needs_frames_observed_to_equal_the_row_count():
    rows = [row(0), row(.1)]
    for observed, status in [(2, "verified"), (3, "unverified"), (1, "unverified"), (None, "unverified")]:
        assert version(rows, frames_observed=observed)["input"]["completeness"] == status


def test_regions_from_another_session_are_a_mismatch():
    rows = [row(0, "parallax", keyframe_id="same:00000001"), row(.1, "blurred"),
            row(.2, "parallax", keyframe_id="same:00000002")]
    labels = [{"keyframe_id": "other:00000001", "region": "desk", "confidence": "high"},
              {"keyframe_id": "other:00000002", "region": "desk", "confidence": "high"}]
    for name in ("v1", "v1_1"):
        assert version(rows, name, regions=labels, room_map=ROOM_MAP)["regions"]["status"] == "session_mismatch"


def test_a_keyframe_missing_from_the_csv_is_unknown():
    a, missing, b = "same:00000001", "same:00000002", "same:00000003"
    rows = [row(0, "parallax", keyframe_id=a), row(.1, "parallax", keyframe_id=missing),
            row(.2, "parallax", keyframe_id=b)]
    labels = [{"keyframe_id": a, "region": "desk", "confidence": "high"},
              {"keyframe_id": b, "region": "desk", "confidence": "high"}]
    buckets = version(rows, regions=labels, room_map=ROOM_MAP)["regions"]["buckets"]
    assert (buckets["bedroom"]["scored"], buckets["unknown"]["scored"]) == (2, 1)


def test_region_span_cap_is_inclusive():
    a, b = "same:00000001", "same:00000002"
    rows = [row(0, "parallax", keyframe_id=a), row(2.5, "blurred"), row(5.0, "parallax", keyframe_id=b)]
    labels = [{"keyframe_id": a, "region": "desk", "confidence": "high"},
              {"keyframe_id": b, "region": "desk", "confidence": "high"}]
    burst = version(rows, regions=labels, room_map=ROOM_MAP)["bursts"]["top"][0]
    assert burst["bracket"]["span_s"] == 5.0 and burst["bucket"] == "bedroom"


def test_burst_diagnostics_floor_sharpness_and_parallax():
    rows = [row(0), row(.1, "blurred", sharp=10.0, parallax=1.0),
            row(.2, "blurred", sharp=25.0, parallax=3.0),
            row(.3, "blurred", sharp=30.0, parallax=8.0), row(.4)]
    burst = version(rows)["bursts"]["top"][0]
    assert (burst["below_floor_count"], burst["median_sharpness"], burst["median_parallax_px"]) == (1, 25.0, 3.0)


def test_output_caps_all_bursts_at_200_and_runs_at_20():
    rows = []
    for i in range(210):
        t = i * .4
        rows += [row(t, "blurred"), row(t + .1, tracker="no_reference"), row(t + .2), row(t + .3)]
    result = version(rows)
    assert result["bursts"]["retained_count"] == 210
    assert [b["rank"] for b in result["bursts"]["all"]] == list(range(1, 201))
    assert result["sharp_runs"]["count"] == 210 and len(result["sharp_runs"]["longest"]) == 20


def test_text_is_closed_and_outputs_contain_no_private_data():
    ids = [f"same:{i:08d}" for i in range(3)]
    rows = [row(1790000000, "parallax", keyframe_id=ids[0]),
            row(1790000000.1, "blurred"), row(1790000000.2, "parallax", keyframe_id=ids[1]),
            row(1790000000.3), row(1790000000.4)]
    labels = [{"keyframe_id": ids[0], "region": "desk", "confidence": "high"},
              {"keyframe_id": ids[1], "region": "bed", "confidence": "med"}]
    summary = summarize(rows, regions=labels, room_map=ROOM_MAP, frames_observed=5)
    rendered = render_text(summary, text_runs=True)
    assert "v1\tv1.1" in rendered
    banned = r"worked|held|hold|link|join|success|fail|fast|slow|turn|pace|pacing|pause|photographed|captured|never looked"
    assert not re.search(banned, rendered, re.I)
    for version_name in ("v1", "v1_1"):
        assert summary[version_name]["input"]["completeness"] == "verified"
    patterns = [
        r"During this walk, the Tower checked \d+ frames and refused \d+ of them as blurry\.",
        r"In this walk's frame log, \d+ of \d+ checked frames were refused as blurry\.",
        r"Blurry frames near [^\t\n]+, \d+:\d\d to \d+:\d\d into the walk \(\d+\.\d seconds\)\.",
        r"Blurry frames, \d+:\d\d to \d+:\d\d into the walk \(\d+\.\d seconds\)\. The frames just before and after show different rooms\.",
        r"Blurry frames, \d+:\d\d to \d+:\d\d into the walk \(\d+\.\d seconds\)\. The room is unknown\.",
        r"In frames assigned to [^\t\n]+, \d+ of \d+ checked frames were refused as blurry\.",
        r"\d+ blurry frames came between frames showing different rooms; no room was assigned\.",
        r"\d+ blurry frames could not be assigned a room\.",
        r"\d+ sharp frames followed, over \d+\.\d seconds\.",
        r"Blur summary unavailable for this walk\.",
    ]
    for line in rendered.splitlines()[1:]:
        for cell in line.split("\t"):
            assert not cell or any(re.fullmatch(pattern, cell) for pattern in patterns), cell
    for forbidden in ("1790000000", "image_path", "frame_ref", "note", "same:"):
        assert forbidden not in rendered + json.dumps(summary)


def test_default_off_static_scan():
    root = Path(__file__).resolve().parents[1] / "tower"
    mentions = [p for p in root.rglob("*.py") if "pacing_summary" in p.read_text(encoding="utf-8")]
    assert [p.relative_to(root).as_posix() for p in mentions] == ["world_builder/pacing_summary.py"]


def _run_cli(journal, output, *extra):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(SCRIPT.parents[1])}
    return subprocess.run([sys.executable, str(SCRIPT), "--journal", str(journal),
                           "--out-dir", str(output), *map(str, extra)],
                          capture_output=True, text=True, env=env, check=False)


def test_cli_guard_and_exact_outputs(tmp_path):
    journal_dir = tmp_path / "input"
    journal_dir.mkdir()
    journal = journal_dir / "frames_quality.jsonl"
    journal.write_text("\n".join(json.dumps(row(i*.1)) for i in range(3)), encoding="utf-8")
    refused = _run_cli(journal, journal_dir)
    assert refused.returncode != 0
    assert not list(journal_dir.glob("pacing_summary.*"))
    output = tmp_path / "output"
    completed = _run_cli(journal, output)
    assert completed.returncode == 0, completed.stderr
    assert sorted(p.name for p in output.iterdir()) == ["pacing_summary.json", "pacing_summary.txt"]
    assert json.loads((output / "pacing_summary.json").read_text(encoding="utf-8"))["schema"] == "glasses.world_builder.pacing_summary/1"
    assert "artifact_root_arg" in SCRIPT.read_text(encoding="utf-8")


def test_cli_unreadable_input_is_one_line_and_tower_data_output_is_refused(tmp_path):
    journal = tmp_path / "missing.jsonl"
    output = tmp_path / "output"
    unreadable = _run_cli(journal, output)
    assert unreadable.returncode != 0
    assert len(unreadable.stderr.splitlines()) == 1
    assert not output.exists()
    journal.write_text(json.dumps(row(0)), encoding="utf-8")
    forbidden = tmp_path / "tower" / "data" / "output"
    refused = _run_cli(journal, forbidden)
    assert refused.returncode != 0
    assert not forbidden.exists()


def test_run_sentence_is_opt_in():
    rows = [row(0, "blurred"), row(.1), row(.2)]
    summary = summarize(rows)
    assert "sharp frames followed" not in render_text(summary)
    assert "2 sharp frames followed, over 0.1 seconds." in render_text(summary, text_runs=True)


def _walk4_rows(path):
    source = json.loads(path.read_text(encoding="utf-8"))
    assert source["replay_matches_journal"] is True
    return [{"v": 1, "received_at": r["at"], "source_seq": r["seq"],
             "reason": r["reason"], "outcome": r["outcome"], "sharpness": r["sharp"],
             "overlap_ratio": r["ref_overlap"], "median_parallax_px": r["ref_disp"],
             "tracker": "reference" if r["ref_surv"] is not None else "no_reference",
             "keyframe_id": "9b03ac81387b471bb90645f6c74ddc03:%08d" % r["seq"] if r["outcome"] == "accept" else None}
            for r in source["rows"]]


def test_optional_walk4_frozen_pins_and_cli(tmp_path):
    reasons = Path(os.environ.get("PACE_WALK4_REASONS", WALK4_REASONS))
    regions = Path(os.environ.get("PACE_WALK4_REGIONS", WALK4_REGIONS))
    if not reasons.is_file() or not regions.is_file():
        pytest.skip("optional frozen walk-4 inputs absent")
    rows = _walk4_rows(reasons)
    labels = load_regions(regions)
    summary = summarize(rows, regions=labels, room_map=ROOM_MAP, frames_observed=len(rows))
    v1, v11 = summary["v1"], summary["v1_1"]
    assert v1["input"]["rows_parsed"] == 2196
    assert v1["blur"]["refused_count"] == 887
    assert v1["clock"]["interval_median_s"] == pytest.approx(.0851, abs=.0001)
    assert v1["clock"]["interval_p75_s"] == pytest.approx(.1067, abs=.0001)
    assert v1["clock"]["gap_threshold_s"] == pytest.approx(.4267, abs=.0001)
    assert v1["clock"]["max_interval_s"] == pytest.approx(.4095, abs=.0001)
    assert v1["clock"]["clock_break_count"] == 0
    assert v1["bursts"]["retained_count"] == 57
    assert v1["bursts"]["non_blur_stretches"] == {"loss_only": 3, "mixed_no_blur": 4, "degraded_only": 0}
    top = v1["bursts"]["top"]
    expected = [(59.702, 65.405, 68, 69), (118.624, 124.029, 65, 66),
                (78.827, 83.884, 61, 62), (48.997, 53.644, 55, 56),
                (55.644, 59.519, 47, 48), (145.621, 149.268, 46, 47)]
    for burst, (start, end, blurry, count) in zip(top, expected):
        assert burst["start_s"] == pytest.approx(start + .314, abs=.005)
        assert burst["end_s"] == pytest.approx(end + .314, abs=.005)
        assert burst["reason_counts"]["blurred"] == blurry
        assert burst["rows"] == count
        assert burst["reason_counts"]["tracking_lost"] == 1
    closet = v1["bursts"]["all"][6]
    assert closet["rank"] == 7
    assert closet["start_s"] == pytest.approx(108.987 + .314, abs=.005)
    assert closet["end_s"] == pytest.approx(112.308 + .314, abs=.005)
    assert closet["rows"] == 41 and closet["reason_counts"] == {"blurred": 40, "tracking_lost": 1}
    assert closet["bucket"] == "between" and closet["following_run"] is None
    assert len(v11["seed_joined_groups"]) == 5
    groups = sorted(v11["seed_joined_groups"], key=lambda g: g["start_s"])
    expected_groups = [(28.390, 29.692), (46.788, 48.049), (55.644, 65.405),
                       (108.987, 113.034), (157.197, 158.394)]
    for group, (start, end) in zip(groups, expected_groups):
        assert group["start_s"] == pytest.approx(start + .314, abs=.005)
        assert group["end_s"] == pytest.approx(end + .314, abs=.005)
    for group, rows_count, reasons, (start, end), duration, sharp, tolerated in [
        (groups[2], 118, {"blurred": 115, "tracking_lost": 2, "session_seed": 1}, (65.549, 67.794), 2.245, 27, 1),
        (groups[3], 51, {"blurred": 48, "tracking_lost": 2, "session_seed": 1}, (113.252, 114.792), 1.540, 20, 0),
    ]:
        assert group["rows"] == rows_count and group["reason_counts"] == reasons
        assert group["following_run"]["start_s"] == pytest.approx(start + .314, abs=.005)
        assert group["following_run"]["end_s"] == pytest.approx(end + .314, abs=.005)
        assert group["following_run"]["duration_s"] == pytest.approx(duration, abs=.005)
        assert group["following_run"]["sharp_rows"] == sharp
        assert group["following_run"]["tolerated_blur_rows"] == tolerated
    # These measured runs are regression pins, not evidence of a connected view.
    times, breaks, origin, _ = _timing(rows)
    all_runs = _sharp_runs(rows, times, breaks, origin)
    for window, duration in [((108.7, 115.0), 1.540), ((116.1, 118.2), 2.341),
                             ((150.0, 152.5), 1.262), ((176.5, 180.0), 6.602),
                             ((177.9, 178.6), .714)]:
        candidates = [r for r in all_runs
                      if r["start_s"] <= window[1] + .314 and r["end_s"] >= window[0] + .314]
        longest = max(candidates, key=lambda r: r["duration_s"])
        assert longest["duration_s"] == pytest.approx(duration, abs=.005)
    # v1 keeps the C4r table's label-first order; v1.1 maps labels to rooms first.
    assert list(v1["regions"]["buckets"]) == list(v11["regions"]["buckets"]) == [
        "bedroom", "closet", "bathroom", "other_labelled", "between", "unknown"]
    assert [(b["refused"], b["scored"]) for b in v1["regions"]["buckets"].values()] == [
        (127, 849), (29, 233), (123, 320), (26, 63), (214, 241), (368, 490)]
    assert [(b["refused"], b["scored"]) for b in v11["regions"]["buckets"].values()] == [
        (195, 922), (29, 233), (123, 320), (26, 63), (146, 168), (368, 490)]
    for summary_version in (v1, v11):
        assert [b["bucket"] for b in summary_version["bursts"]["top"]] == [
            "unknown", "unknown", "unknown", "bedroom", "unknown", "bathroom"]
    journal = tmp_path / "frames_quality.jsonl"
    journal.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    room_map = tmp_path / "room_map.json"
    room_map.write_text(json.dumps(ROOM_MAP), encoding="utf-8")
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"frames_observed": len(rows)}), encoding="utf-8")
    output = tmp_path / "result"
    cli = _run_cli(journal, output, "--regions", regions, "--room-map", room_map)
    assert cli.returncode == 0, cli.stderr
    emitted = json.loads((output / "pacing_summary.json").read_text(encoding="utf-8"))
    assert emitted["v1_1"]["regions"]["buckets"] == v11["regions"]["buckets"]
    # MED-2: v1's 214 label-first "between" frames (68 of them desk/bed) are never called
    # "different rooms"; they are folded into R3 (608 = 368 + 26 + 214).
    left, right = _columns((output / "pacing_summary.txt").read_text(encoding="utf-8"))
    assert not any("different rooms" in cell for cell in left)
    assert left[-2:] == ["", "608 blurry frames could not be assigned a room."]
    # OPEN (g), C4i-G: of v1.1's 146 "between" frames, 119 have an unmapped label ("other")
    # on one side; only 27 are room/room.  The 119 move to R3 (513 = 368 + 26 + 119).
    area_ranks = [b["rank"] for b in v11["bursts"]["all"] if b["bucket"] == "between" and
                  (b["bracket"]["before"]["label"] in ROOM_MAP) != (b["bracket"]["after"]["label"] in ROOM_MAP)]
    assert area_ranks == [7, 9, 17, 25, 32, 38, 40, 54]
    assert sum(b["reason_counts"]["blurred"] for b in v11["bursts"]["all"] if b["rank"] in area_ranks) == 119
    assert right[-2:] == ["27 blurry frames came between frames showing different rooms; no room was assigned.",
                          "513 blurry frames could not be assigned a room."]
    # Rank 7, the closet exit (closet / other), is B3 in both columns if it is ever displayed.
    probe = copy.deepcopy(emitted)
    for name in ("v1", "v1_1"):
        probe[name]["bursts"]["top"] = probe[name]["bursts"]["all"][6:7]
    left7, right7 = _columns(render_text(probe))
    assert left7[1] == right7[1] == "Blurry frames, 1:49 to 1:53 into the walk (3.3 seconds). The room is unknown."
    print("WALK4_JSON=" + json.dumps(emitted, separators=(",", ":")))
    print("WALK4_TEXT=" + (output / "pacing_summary.txt").read_text(encoding="utf-8"))
