"""Pacing rules and optional, read-only facts from two local walks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tower.world_builder.frame_quality import read_frames_quality
from tower.world_builder.pacing_summary import summarize


PREFLIGHT = Path(
    r"C:\Users\tvllo\Projects\Glasses\tower\data\world_builder\worlds"
    r"\488ed62141404069ad9c6b4452b90ead\sessions"
    r"\5b9467f0a8844628819e3694ffafcc9a\frames_quality.jsonl"
)
WALK4 = Path(
    r"C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23"
    r"\lead\diag4\reasons_walk4.json"
)


def row(t, reason="insufficient_motion", *, tracker="reference", segment=0,
        sharpness=100, overlap=0.9, outcome="skip", parallax=2):
    return {"v": 1, "received_at": t, "reason": reason, "tracker": tracker,
            "segment_index": segment, "sharpness": sharpness,
            "overlap_ratio": overlap, "outcome": outcome,
            "median_parallax_px": parallax}


def test_jittered_intervals_and_blur_tolerance():
    rows = [row(0), row(.08), row(.18), row(.30, "blurred"), row(.39),
            row(.51, segment=3), row(.70)]
    result = summarize(rows, gap_percentile=.75, gap_multiplier=3)
    assert result["gap_threshold_seconds"] > .19
    assert result["clock_gap_count"] == 0
    assert result["held_runs"][0] == {
        "start_received_at": 0, "end_received_at": .70,
        "duration_seconds": .70, "sharp_frame_count": 6,
        "tolerated_blur_count": 1, "overlap_backed_count": 6,
    }
    assert result["blur"] == {"refused_count": 1, "scored_count": 7, "share": 1 / 7}


def test_blur_burst_keeps_loss_and_gives_following_pause():
    rows = [row(0), row(.1, "blurred", sharpness=8),
            row(.2, "tracking_lost", segment=0),
            row(.3, "blurred", segment=1, sharpness=10),
            row(.4, tracker="no_reference", segment=1), row(.5, segment=1),
            row(.6, segment=1), row(.7, segment=1)]
    result = summarize(rows)
    assert result["burst_count"] == 1
    assert result["bursts"][0]["kind"] == "blur_and_loss"
    assert result["bursts"][0]["reason_counts"] == {"blurred": 2, "tracking_lost": 1}
    assert result["bursts"][0]["frame_count"] == 3
    assert result["bursts"][0]["duration_seconds"] == pytest.approx(.2)
    assert result["bursts"][0]["below_sharpness_floor_share"] == pytest.approx(2 / 3)
    assert result["bursts"][0]["pause_after_burst_seconds"] == pytest.approx(.2)
    assert result["loss"]["per_minute"] == pytest.approx(60 / .7)
    assert all(run["start_received_at"] >= .5 for run in result["held_runs"])


def test_segment_change_does_not_split_sharp_run_or_burst():
    rows = [row(0), row(.1, segment=1), row(.2, segment=1),
            row(.3, "blurred", segment=1), row(.4, "tracking_lost", segment=2),
            row(.5, "blurred", segment=3)]
    result = summarize(rows)
    assert result["held_runs"][0]["duration_seconds"] == .2
    assert result["burst_count"] == 1
    assert result["bursts"][0]["frame_count"] == 3


def test_degraded_only_is_not_a_cue():
    result = summarize([row(0), row(.1, "tracking_degraded"),
                        row(.2, "tracking_held"), row(.3)])
    assert result["burst_count"] == 0
    assert result["bursts"] == []
    assert result["reason_counts"]["tracking_degraded"] == 1


def test_reconnect_gap_breaks_bursts_and_runs():
    rows = [row(0), row(.1), row(.2, "blurred"), row(.3, "blurred"),
            row(.4), row(.5), row(21.7), row(21.8),
            row(21.9, "blurred"), row(22, "blurred")]
    result = summarize(rows, gap_percentile=.75, gap_multiplier=3)
    assert result["gap_threshold_seconds"] < 21
    assert result["clock_gap_count"] == 1
    assert result["burst_count"] == 2
    assert all(b["duration_seconds"] < 1 for b in result["bursts"])
    assert all(run["duration_seconds"] < 1 for run in result["held_runs"])


def test_no_clock_basis_does_not_invent_duration_or_rate():
    result = summarize([row(1), row(1, "tracking_lost"), row(1, "blurred")])
    assert result["loss"]["per_minute"] is None
    assert result["observed_span_seconds"] is None
    assert result["clock_gap_count"] == 2
    assert result["burst_count"] == 2


def _local_path(name, default):
    path = Path(os.environ.get(name, default))
    if not path.is_file():
        pytest.skip(f"optional local evidence absent: {path}")
    return path


def test_preflight_journal_facts_and_reconnect():
    path = _local_path("PACE_PREFLIGHT_JOURNAL", PREFLIGHT)
    rows = read_frames_quality(path)
    result = summarize(rows, top_k=1995)
    assert result["row_count"] == 1995
    assert result["reason_counts"]["blurred"] == 20
    assert result["reason_counts"]["tracking_degraded"] == 92
    assert result["reason_counts"]["tracking_lost"] == 1
    assert result["blur"]["refused_count"] == 20
    assert result["loss"]["count"] == 1
    session_path = path.with_name("session.json")
    if session_path.is_file():
        session = json.loads(session_path.read_text(encoding="utf-8"))
        assert session["frames_observed"] == len(rows)
        for reason, count in session["rejected_by_reason"].items():
            assert result["reason_counts"][reason] == count
        assert result["reason_counts"]["session_seed"] + result["reason_counts"].get(
            "parallax", 0) + result["reason_counts"].get("overlap_floor", 0) == session["keyframes_accepted"]
    gaps = [(i, rows[i]["received_at"] - rows[i-1]["received_at"])
            for i in range(1, len(rows))]
    reconnect, gap = max(gaps, key=lambda item: item[1])
    assert gap == pytest.approx(21.159, abs=.02)
    assert gap > result["gap_threshold_seconds"]
    assert all(not (b["start_received_at"] < rows[reconnect]["received_at"]
                    <= b["end_received_at"]) for b in result["bursts"])
    assert all(not (run["start_received_at"] < rows[reconnect]["received_at"]
                    <= run["end_received_at"]) for run in result["held_runs"])
    for i in range(1, len(rows)):
        if rows[i]["segment_index"] != rows[i-1]["segment_index"] and rows[i-1]["reason"] != "tracking_lost":
            assert rows[i]["received_at"] - rows[i-1]["received_at"] < result["gap_threshold_seconds"]


def _walk4_rows(path):
    """DIAG4 replay -> the relevant v1 journal fields, preserving capture time."""
    replay = json.loads(path.read_text(encoding="utf-8"))
    assert replay["replay_matches_journal"] is True
    result = []
    segment = 0
    for source in replay["rows"]:
        result.append({
            "v": 1, "received_at": source["at"], "source_seq": source["seq"],
            "reason": source["reason"], "outcome": source["outcome"],
            "sharpness": source["sharp"], "overlap_ratio": source["ref_overlap"],
            "median_parallax_px": source["ref_disp"],
            "tracker": "reference" if source["ref_surv"] is not None else "no_reference",
            "segment_index": segment,
            "keyframe_id": "accepted" if source["outcome"] == "accept" else None,
        })
        if source["reason"] == "tracking_lost":
            segment += 1
    return replay, result


def test_walk4_closet_exit_is_one_burst():
    path = _local_path("PACE_WALK4_REASONS", WALK4)
    replay, rows = _walk4_rows(path)
    result = summarize(rows, top_k=len(rows))
    assert result["row_count"] == 2196
    assert result["reason_counts"]["blurred"] == 887
    assert result["reason_counts"]["tracking_lost"] == replay["journal_lost"] == 47
    # DIAG4's labelled closet-exit interval has exactly 40 blur rows and one loss.
    closet = [b for b in result["bursts"] if b["frame_count"] == 41 and
              108.5 <= b["start_received_at"] - rows[0]["received_at"] <= 109.5]
    assert len(closet) == 1
    assert closet[0]["frame_count"] == 41
    assert closet[0]["reason_counts"] == {"blurred": 40, "tracking_lost": 1}
    origin = rows[0]["received_at"] - replay["rows"][0]["t"]
    assert closet[0]["start_received_at"] - origin == pytest.approx(109.0, abs=.1)
    assert closet[0]["end_received_at"] - origin == pytest.approx(112.3, abs=.1)
    assert closet[0]["duration_seconds"] == pytest.approx(3.32, abs=.15)
