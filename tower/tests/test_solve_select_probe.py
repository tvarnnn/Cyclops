"""Pure-helper checks for the offline best-of-N solve selection probe (SEL)."""

import math

import numpy as np

from scripts.solve_select_probe import (
    hull_inside_fraction,
    label,
    score_s1,
    score_s2,
    selection_stats,
    separation,
    similarity,
    walk_second,
)


def _square(n=40, half=1.0):
    t = np.linspace(-half, half, n)
    return np.array([[x, y, 0.01 * x] for x in t for y in (-half, half)]
                    + [[x, y, 0.0] for y in t for x in (-half, half)])


def test_walk_second_reads_source_sequence_at_24_fps():
    assert walk_second("abc:00002400") == 100.0


def test_hull_fraction_counts_inside_and_outside_points():
    bedroom = _square()
    query = np.array([[0.0, 0.0, 0.0], [0.5, -0.5, 0.0], [3.0, 0.0, 0.0], [0.0, -4.0, 0.0]])
    assert hull_inside_fraction(bedroom, query) == 0.5
    assert hull_inside_fraction(bedroom, np.zeros((0, 3))) is None


def test_label_needs_room_outside_hull_and_unit_scale():
    assert label(1.0, 0.1, 1.05) == "GOOD"
    assert label(1.0, 0.1, 2.0) == "WELDED"       # outside the hull but at twice the scale (G2)
    assert label(1.0, 0.8, 1.0) == "WELDED"       # inside the bedroom hull
    assert label(0.2, 0.0, 1.0) == "SPLIT"        # closet not in the main component


def _row(tag, lab, isolated, uncertified, room, worst=0.0):
    return {"tag": tag, "label": lab, "isolated_kf": isolated, "uncertified_kf": uncertified,
            "room_kf": room, "worst_attached_abslog": worst, "closet_ratio": 1.0}


def test_scores_prefer_fewer_isolated_and_more_certified():
    good = _row("g", "GOOD", 60, 40, 690)
    welded = _row("w", "WELDED", 300, 46, 463)
    assert score_s1(good) < score_s1(welded)
    assert score_s2(good) < score_s2(welded)


def test_selection_stats_and_separation():
    rows = [_row("a", "GOOD", 10, 0, 700), _row("b", "WELDED", 300, 0, 460), _row("c", "WELDED", 250, 0, 500)]
    s = selection_stats(rows, score_s1, 2)
    assert s["subsets"] == 3 and math.isclose(s["p_any_good"], 2 / 3) and math.isclose(s["p_pick_good"], 2 / 3)
    assert separation(rows, score_s1)["auc"] == 1.0
    flipped = [dict(rows[0], isolated_kf=400)] + rows[1:]
    assert separation(flipped, score_s1)["auc"] == 0.0


def test_similarity_recovers_scale_rotation_shift():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(50, 3))
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    dst = 2.0 * src @ R.T + np.array([1.0, 2.0, 3.0])
    scale, R_fit, t = similarity(src, dst)
    assert abs(scale - 2.0) < 1e-9 and np.allclose(R_fit, R) and np.allclose(t, [1.0, 2.0, 3.0])
