"""Scorer truth separation and fail-closed external checker evidence."""

import hashlib
import json

import pytest

from scripts import exp2_score as S


def test_score_requires_independent_pair_truth_and_observation_floor():
    labels = [
        {"keyframe_id": "s:0", "region": "closet", "room": "closet", "confidence": "high"},
        {"keyframe_id": "s:1", "region": "desk", "room": "bedroom", "confidence": "medium"},
    ]
    truth = {("000.jpg", "001.jpg"): "true_doorway_return",
             ("000.jpg", "002.jpg"): "false_cross_room"}
    candidates = [{"query": "000.jpg", "targets": ["001.jpg", "002.jpg"]}]
    solution = {"poses": {"s:0": {"observations": 30},
                          "s:1": {"observations": 29}}, "timing": {"solve_s": 5}}
    components = {"components": [{"state": "placed", "shown_as": "room",
                                  "keyframe_ids": ["s:0", "s:1"]}]}
    result = S.score_arm(labels, truth, candidates,
                         {("000.jpg", "001.jpg"): 22, ("000.jpg", "002.jpg"): 16},
                         solution, components)
    assert result["bridges"]["verified_true_doorway_return"] == 1
    assert result["bridges"]["verified_false_cross_room"] == 1
    assert result["frames"]["closet/high"]["posed"] == 1
    assert result["frames"]["desk/medium"]["posed"] == 0
    assert result["frames"]["closet/high"]["published"] == 1
    assert result["closet_scale"] is None
    assert result["fow_wrong_room_support"] is None
    assert result["frames"]["closet/high"]["placed_correct_room"] is None


def test_score_treats_ambiguous_pair_as_excluded():
    result = S.score_arm([], {("a", "b"): "ambiguous"},
                         [{"query": "a", "targets": ["b"]}], {("a", "b"): 60},
                         {"poses": {}}, None)
    assert result["bridges"]["verified_true_doorway_return"] == 0
    assert result["bridges"]["verified_false_cross_room"] == 0
    assert result["bridges"]["ambiguous_verified_excluded"] == 1


def test_low_confidence_frames_remain_reported_but_not_certified():
    labels = [{"keyframe_id": "s:0", "region": "other", "room": "ambiguous",
               "confidence": "low"}]
    out = S.score_arm(labels, {}, [], {}, {"poses": {"s:0": {"observations": 30}}}, None)
    assert out["frames"]["other/low"]["captured"] == 1
    assert out["frames"]["other/low"]["posed"] == 1
    assert out["frames"]["other/low"]["placed_correct_room"] is None


def test_verified_bridge_use_and_wrong_room_need_separate_evidence():
    labels = [{"keyframe_id": "s:0", "region": "closet", "room": "closet",
               "confidence": "high"}]
    pair = ("a.jpg", "b.jpg")
    result = S.score_arm(labels, {pair: "true_doorway_return"},
                         [{"query": "a.jpg", "query_index": 0, "targets": ["b.jpg"],
                           "target_indices": [50]}], {pair: 40},
                         {"poses": {"s:0": {"observations": 30}}}, None,
                         used_pairs={pair}, published_ids={"s:0"},
                         placement_review={"s:0": "bedroom"})
    assert result["bridges"]["verified_true_used"] == 1
    assert result["frames"]["closet/high"]["published_wrong_room"] == 1
    assert result["candidate_nonsequential_unique"] == 1


def test_frozen_scale_checker_outer_arms_and_pose_flags_are_read():
    checker = {"arms": [{"arm": "R", "scale": {"status": "PASS",
                "regions": [{"region": "closet", "ratio": 1.02, "measured": 12}]},
                "pose": {"status": "FAIL", "flag_counts": {"speed": 1}}}]}
    out = S.score_arm([], {}, [], {}, {"poses": {}}, None,
                      scale_checker=checker, scale_arm="R")
    assert out["closet_scale"]["in_band"] is True
    assert out["closet_scale"]["pose_flag_counts"]["speed"] == 1


def test_score_refuses_a_one_byte_change_to_frozen_labels(tmp_path):
    labels = tmp_path / "labels.csv"
    sheet = tmp_path / "adjudication.csv"
    labels.write_bytes(b"labels")
    sheet.write_bytes(b"pairs")
    frozen = tmp_path / "exp2-freeze.json"
    frozen.write_text(json.dumps({
        "labels_sha256": hashlib.sha256(b"labels").hexdigest(),
        "adjudication_sha256": hashlib.sha256(b"pairs").hexdigest()}))
    pin = hashlib.sha256(frozen.read_bytes()).hexdigest()
    S.verify_frozen_labels(frozen, pin, labels, sheet)
    sheet.write_bytes(b"pAirs")
    with pytest.raises(ValueError, match="adjudication"):
        S.verify_frozen_labels(frozen, pin, labels, sheet)


def test_walk3_6_room_mapping_requires_inspected_other_and_preserves_walk7_room():
    labels = [
        {"keyframe_id": "s:0", "region": "desk", "confidence": "high"},
        {"keyframe_id": "s:1", "region": "other", "confidence": "medium"},
        {"keyframe_id": "s:2", "region": "other", "confidence": "low"},
        {"keyframe_id": "s:3", "region": "closet", "room": "closet", "confidence": "high"},
    ]
    sheet = [{"record_type": "frame", "keyframe_id": "s:1", "room": "bedroom"},
             {"record_type": "frame", "keyframe_id": "s:3", "room": "bedroom"}]
    out = S.resolve_label_rooms(labels, sheet)
    assert [r["room"] for r in out] == ["bedroom", "bedroom", "ambiguous", "closet"]


def test_an_unplaced_area_is_published_but_not_a_certified_room():
    label = {"keyframe_id": "s:0", "region": "closet", "room": "closet", "confidence": "high"}
    components = {"components": [{"state": "unplaced", "shown_as": "area",
                                  "keyframe_ids": ["s:0"]}]}
    out = S.score_arm([label], {}, [], {}, {"poses": {"s:0": {"observations": 30}}},
                      components, placement_review={"s:0": "closet"})
    assert out["frames"]["closet/high"]["published"] == 1
    assert out["frames"]["closet/high"]["placed_correct_room"] == 0
