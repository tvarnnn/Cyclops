"""Frozen offline pacing rules; the optional walk-4 replay is never live input."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tower.world_builder.pacing_summary import (
    _sharp_runs, _timing, load_frames_observed, load_journal, load_regions, render_text, summarize,
)


RUN = Path(r"C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23")
WALK4_REASONS = RUN / "lead/diag4/reasons_walk4.json"
WALK4_REGIONS = RUN / "physical-test/regions-w4/c81766a3bd3a4eeb9739aa4271fdbb75_regions.csv"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/world_pacing_summary.py"
ROOM_MAP = {"desk": "bedroom", "bed": "bedroom", "closet": "closet", "bathroom": "bathroom"}


def row(t, reason="insufficient_motion", *, tracker="reference", sharp=100.0,
        overlap=.9, outcome="skip", keyframe_id=None, segment=0):
    return {"received_at": t, "reason": reason, "tracker": tracker,
            "sharpness": sharp, "overlap_ratio": overlap, "outcome": outcome,
            "median_parallax_px": 2.0, "keyframe_id": keyframe_id,
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
    assert closet["rows"] == 41 and closet["reason_counts"] == {"blurred": 40, "tracking_lost": 1}
    assert closet["bucket"] == "between" and closet["following_run"] is None
    assert len(v11["seed_joined_groups"]) == 5
    groups = sorted(v11["seed_joined_groups"], key=lambda g: g["start_s"])
    expected_groups = [(28.390, 29.692), (46.788, 48.049), (55.644, 65.405),
                       (108.987, 113.034), (157.197, 158.394)]
    for group, (start, end) in zip(groups, expected_groups):
        assert group["start_s"] == pytest.approx(start + .314, abs=.005)
        assert group["end_s"] == pytest.approx(end + .314, abs=.005)
    for group, rows_count, reasons, duration, sharp, tolerated in [
        (groups[2], 118, {"blurred": 115, "tracking_lost": 2, "session_seed": 1}, 2.245, 27, 1),
        (groups[3], 51, {"blurred": 48, "tracking_lost": 2, "session_seed": 1}, 1.540, 20, 0),
    ]:
        assert group["rows"] == rows_count and group["reason_counts"] == reasons
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
        assert any(r["duration_s"] == pytest.approx(duration, abs=.005) for r in candidates)
    assert [(b["refused"], b["scored"]) for b in v11["regions"]["buckets"].values()] == [
        (195, 922), (29, 233), (123, 320), (26, 63), (146, 168), (368, 490)]
    assert [b["bucket"] for b in v11["bursts"]["top"]] == [
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
    print("WALK4_JSON=" + json.dumps(emitted, separators=(",", ":")))
    print("WALK4_TEXT=" + (output / "pacing_summary.txt").read_text(encoding="utf-8"))
