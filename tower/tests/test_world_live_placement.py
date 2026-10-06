"""Synthetic checks for causal selection, 3-D matching, vetoes, and alignment."""

import numpy as np
import cv2
import sqlite3

from scripts import world_live_placement as placement


def test_causal_selector_excludes_unpublished_and_input_horizon():
    rows = [
        {"seen_at": 9, "keyframe_ids": ["old"]},
        {"seen_at": 12, "keyframe_ids": ["old", "query"]},
        {"seen_at": 20, "keyframe_ids": ["old", "query", "later"]},
    ]
    assert placement.causal_snapshot(rows, "query", 11) is rows[0]
    assert placement.causal_snapshot(rows, "query", 13) is rows[0]
    assert placement.causal_snapshot(rows, "query", 9) is None


def test_unique_landmark_matches_use_ratio_and_deduplicate():
    # Three descriptors, with two references to the same 3-D point.
    query = np.zeros((3, 128), np.uint8)
    query[1, :] = 100
    query[2, :] = 200
    refs = np.stack((query[0], query[1], query[2], np.full(128, 240, np.uint8)))
    got = placement.unique_matches(query, [(refs, np.array([5, 5, 7, 8]), "r")])
    assert {(m[0], m[1]) for m in got} == {(0, 5), (2, 7)}


def test_matching_keeps_competing_component_hypotheses():
    query = np.zeros((1, 128), np.uint8)
    refs = np.stack((query[0], np.full(128, 200, np.uint8)))
    got = placement.unique_matches(query, [(refs, np.array([5, 6]), "room-a", 0),
                                           (refs, np.array([50, 60]), "room-b", 1)])
    assert {(m[1], m[2]) for m in got} == {(5, "room-a"), (50, "room-b")}


def test_gate_rejects_wrong_room_single_link_and_competing_pose():
    strong = {"component": 0, "inliers": 45, "references": {"a", "b", "c"},
              "cells": 7, "positive_fraction": 1.0, "p95_px": 2.0,
              "centre": np.array([0., 0., 0.]), "rotation": np.eye(3)}
    assert placement.pose_veto([strong], tracking_break=True) is None
    weak = {**strong, "references": {"a"}}
    assert placement.pose_veto([weak], tracking_break=True) == "insufficient-independent-references"
    rival = {**strong, "component": 1, "centre": np.array([10., 0., 0.]),
             "references": {"d", "e", "f"}, "inliers": 40}
    assert placement.pose_veto([strong, rival], tracking_break=True) == "ambiguous-pose"
    assert placement.pose_veto([strong, {**rival, "component": 0}], tracking_break=True) == "ambiguous-pose"
    motion = {"centre": np.array([20., 0., 0.]), "max_distance": 3.0}
    assert placement.pose_veto([strong], tracking_break=False, motion=motion) == "motion-disagreement"


def test_sim3_alignment_reports_position_and_rotation_without_using_query():
    old = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    final = old * 2 + np.array([3., -4., 5.])
    transform = placement.fit_sim3(old, final)
    assert np.allclose(placement.transform_centres(transform, old), final, atol=1e-8)
    assert placement.alignment_residuals(transform, old, final)["p95"] < 1e-7
    result = placement.pose_error(transform, np.array([0.5, 0.5, 0.5]), np.eye(3),
                                  np.array([4., -3., 6.]), np.eye(3))
    assert result["position"] < 1e-7
    assert result["rotation_deg"] < 1e-5
    assert result["forward_deg"] < 1e-5


def test_component_pnp_recovers_synthetic_camera_and_distributed_support():
    points = np.array([[x, y, z] for x in np.linspace(-2, 2, 6)
                       for y, z in ((-1., 5.), (1., 6.))], np.float32)
    camera = {"fx": 400., "fy": 400., "cx": 320., "cy": 240., "width": 640, "height": 480}
    intrinsic = np.array([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
    image, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), intrinsic, None)
    keypoints = [cv2.KeyPoint(float(x), float(y), 1) for x, y in image.reshape(-1, 2)]
    matches = [(i, i, ("a", "b", "c")[i % 3], 0.) for i in range(len(points))]
    candidates = placement.pnp_candidates(keypoints, matches, points,
                                          np.zeros(len(points), np.int32), camera)
    assert len(candidates) == 1
    assert candidates[0]["inliers"] == len(points)
    assert np.linalg.norm(candidates[0]["centre"]) < 1e-3


def test_reference_features_excludes_database_descriptors_mismatched_to_published_xy(tmp_path):
    solve = {"keyframe_ids": ["s:00000001"],
             "poses": {"s:00000001": {"component": 0}}}
    xyz = np.array([[0., 0., 5.], [1., 0., 5.], [2., 0., 5.]], np.float32)
    np.savez_compressed(tmp_path / "000.npz", xyz=xyz, component=np.zeros(3, np.int32),
                        observations=np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2]], np.int32),
                        observation_xy=np.array([[10., 20.], [30., 40.], [50., 60.]], np.float32))
    db = sqlite3.connect(tmp_path / "000.db")
    db.execute("create table images(image_id integer primary key, name text)")
    db.execute("create table descriptors(image_id integer primary key, rows integer, cols integer, data blob)")
    db.execute("create table keypoints(image_id integer primary key, rows integer, cols integer, data blob)")
    db.execute("insert into images values(1,'00000001.jpg')")
    db.execute("insert into descriptors values(1,3,128,?)", (np.arange(384, dtype=np.uint8).tobytes(),))
    db.execute("insert into keypoints values(1,3,2,?)", (
        np.array([[10., 20.], [30., 40.], [500., 600.]], np.float32).tobytes(),))
    db.commit()
    db.close()
    _, _, refs = placement.reference_features({"doc": solve, "array_file": "000.npz",
                                               "descriptor_file": "000.db"}, tmp_path)
    assert len(refs) == 1
    assert refs[0][2].tolist() == [0, 1]
