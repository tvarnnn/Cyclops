"""CLI-contract tests for the World Builder Experiment 1/2 analysis scripts.

These cover argument handling and the pure-numeric helpers only -- they do
not run MiDaS or decode a real video, so they stay in the default (fast,
no-model) suite, and they assert nothing about the experiments' measured
results.
"""
import subprocess
import sys

import cv2
import numpy as np

from scripts.depth_temporal_consistency import (
    _deviation_from_raw,
    _ema,
    _flicker_metrics,
    _normalize_median_mad,
    _normalize_minmax,
    _normalize_percentile,
    _renormalize,
    _resize_to_platform_budget,
    _step_response_lag_frames,
    _temporal_median,
)
from scripts.feature_trackability import _match_pair, _summarize


def test_depth_temporal_consistency_requires_video_argument():
    result = subprocess.run(
        [sys.executable, "scripts/depth_temporal_consistency.py"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--video" in result.stderr


def test_feature_trackability_requires_video_argument():
    result = subprocess.run(
        [sys.executable, "scripts/feature_trackability.py"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--video" in result.stderr


def test_normalize_maps_to_unit_range():
    depth = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)

    normalized = _normalize_minmax(depth)

    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_normalize_handles_flat_map_without_dividing_by_zero():
    flat = np.full((4, 4), 7.0, dtype=np.float32)

    for normalizer in (_normalize_minmax, _normalize_percentile, _normalize_median_mad):
        assert np.all(normalizer(flat) == 0.0), normalizer.__name__


def test_percentile_normalizer_is_robust_to_a_single_outlier_where_minmax_is_not():
    # The reason minmax is not the default: one pixel sets the scale for
    # the entire frame, so an outlier moves every other pixel.
    base = np.random.RandomState(0).rand(64, 64).astype(np.float32)
    spiked = base.copy()
    spiked[0, 0] = 1000.0

    minmax_shift = np.abs(_normalize_minmax(spiked) - _normalize_minmax(base)).mean()
    robust_shift = np.abs(
        _normalize_percentile(spiked) - _normalize_percentile(base)
    ).mean()

    assert robust_shift < minmax_shift / 10


def test_renormalize_restores_full_dynamic_range():
    compressed = np.linspace(0.4, 0.6, 16, dtype=np.float32).reshape(4, 4)

    restored = _renormalize(compressed)

    assert restored.min() == 0.0
    assert restored.max() == 1.0


def test_resize_preserves_orientation_at_platform_budget():
    landscape = np.zeros((1080, 1920, 3), dtype=np.uint8)
    portrait = np.zeros((1920, 1080, 3), dtype=np.uint8)

    assert _resize_to_platform_budget(landscape).shape[:2] == (504, 896)
    assert _resize_to_platform_budget(portrait).shape[:2] == (896, 504)


def test_flicker_metrics_are_zero_for_a_perfectly_static_sequence():
    static = [np.full((8, 8), 0.5, dtype=np.float32) for _ in range(5)]

    metrics = _flicker_metrics(static)

    assert metrics["mad_mean"] == 0.0
    assert metrics["local_temporal_std_mean"] == 0.0


def test_ema_reduces_flicker_of_an_alternating_sequence():
    alternating = [
        np.full((8, 8), 1.0 if index % 2 else 0.0, dtype=np.float32)
        for index in range(10)
    ]

    raw = _flicker_metrics(alternating)["mad_mean"]
    smoothed = _flicker_metrics(_ema(alternating, 0.3))["mad_mean"]

    assert smoothed < raw


def test_temporal_median_window_of_one_is_a_passthrough():
    sequence = [np.random.rand(4, 4).astype(np.float32) for _ in range(5)]

    smoothed = _temporal_median(sequence, window=1)

    for original, result in zip(sequence, smoothed):
        assert np.allclose(original, result)


def _first_affected_index(smoother, length=20, perturb_at=5):
    """Index of the first output changed by a step applied at perturb_at.

    A step (not a single spike) is used so that a median filter actually
    responds -- one outlier in a 5-window median is rejected by design.
    Any index < perturb_at means the smoother is reading future frames.
    """
    base = [np.zeros((4, 4), dtype=np.float32) for _ in range(length)]
    perturbed = [
        np.ones((4, 4), dtype=np.float32) if i >= perturb_at else f.copy()
        for i, f in enumerate(base)
    ]

    before, after = smoother(base), smoother(perturbed)
    for index, (b, a) in enumerate(zip(before, after)):
        if not np.allclose(b, a):
            return index
    return None


def test_ema_is_causal():
    # The load-bearing property: a live stream cannot see future frames.
    # A window-of-1 passthrough test cannot detect a violation, so apply a
    # step and confirm no EARLIER output moves.
    assert _first_affected_index(lambda s: _ema(s, 0.3)) == 5


def test_temporal_median_is_causal():
    # A trailing median responds later than an EMA (it needs the majority
    # of its window to flip), but it must still never respond EARLY.
    first_affected = _first_affected_index(lambda s: _temporal_median(s, 5))

    assert first_affected is not None
    assert first_affected >= 5


def test_step_response_lag_is_larger_for_heavier_smoothing():
    light = _step_response_lag_frames(lambda s: _ema(s, 0.5))
    heavy = _step_response_lag_frames(lambda s: _ema(s, 0.3))

    assert heavy > light
    assert light >= 1.0


def test_deviation_from_raw_is_zero_when_smoothed_equals_raw():
    sequence = [np.random.rand(4, 4).astype(np.float32) for _ in range(4)]

    assert _deviation_from_raw(sequence, sequence) == 0.0


def test_deviation_from_raw_is_large_for_perfect_zero_lag_denoising():
    """Pins why this metric must not be called "lag".

    Construct a smoother that returns the true constant signal with zero
    delay. Lag is zero by construction, yet the metric is large -- so a
    large value cannot be read as evidence of lag.
    """
    truth = np.full((8, 8), 0.5, dtype=np.float32)
    rng = np.random.RandomState(0)
    noisy = [truth + rng.normal(0, 0.1, (8, 8)).astype(np.float32) for _ in range(20)]
    perfectly_denoised = [truth.copy() for _ in noisy]

    assert _deviation_from_raw(noisy, perfectly_denoised) > 0.05


def test_summarize_reports_expected_statistics():
    summary = _summarize([1.0, 2.0, 3.0, 10.0])

    assert summary == {"mean": 4.0, "median": 2.5, "min": 1.0, "max": 10.0}


def test_match_pair_returns_empty_stats_when_descriptors_are_missing():
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    stats = _match_pair(matcher, None, None, [], [])

    assert stats == {
        "matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "homography_inliers": 0,
        "r_h": None,
    }


def test_match_pair_finds_verified_inliers_between_a_frame_and_its_shift():
    # A real image translated by a few pixels must produce a large number
    # of geometrically-verified matches; this is the sanity floor for the
    # whole experiment.
    rng = np.random.RandomState(0)
    scene = rng.randint(0, 255, (240, 320), dtype=np.uint8)
    scene = cv2.GaussianBlur(scene, (3, 3), 0)
    shifted = np.roll(scene, 4, axis=1)

    detector = cv2.ORB_create(nfeatures=500)
    kp_a, des_a = detector.detectAndCompute(scene, None)
    kp_b, des_b = detector.detectAndCompute(shifted, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    stats = _match_pair(matcher, des_a, des_b, kp_a, kp_b)

    assert stats["matches"] >= 8
    assert stats["inliers"] > 0
    assert 0.0 <= stats["inlier_ratio"] <= 1.0
    assert stats["r_h"] is None or 0.0 <= stats["r_h"] <= 1.0


def test_match_pair_reports_no_inliers_when_too_few_matches():
    detector = cv2.ORB_create(nfeatures=500)
    blank = np.zeros((64, 64), dtype=np.uint8)
    _, des = detector.detectAndCompute(blank, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    stats = _match_pair(matcher, des, des, [], [])

    assert stats["inliers"] == 0
    assert stats["r_h"] is None
