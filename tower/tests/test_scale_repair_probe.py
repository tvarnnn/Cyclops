"""Numerical checks for the offline, fail-closed scale repair probe."""

import numpy as np

from scripts.scale_repair_probe import robust_similarity, triangulate


def test_robust_similarity_recovers_known_scale_with_outliers():
    rng = np.random.default_rng(17)
    src = rng.normal(size=(80, 3))
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    dst = 0.5 * src @ rotation.T + np.array([3., -2., 1.])
    dst += rng.normal(scale=0.001, size=dst.shape)
    dst[::7] += np.array([2., 3., -1.])

    fit = robust_similarity(src, dst, threshold=0.02)

    assert fit is not None
    scale, fitted_rotation, shift, residuals, inliers = fit
    assert abs(scale - 0.5) < 0.01
    assert np.linalg.norm(fitted_rotation - rotation) < 0.01
    assert np.linalg.norm(shift - [3., -2., 1.]) < 0.02
    assert inliers.sum() >= 65
    assert np.median(residuals[inliers]) < 0.01


def test_triangulate_rejects_camera_behind_point():
    centers = {0: np.array([0., 0., 0.]), 1: np.array([1., 0., 0.])}
    rotations = {0: np.eye(3), 1: np.eye(3)}
    intrinsics = (100., 100., 0., 0.)
    assert triangulate([(0, np.array([0., 0.])), (1, np.array([-20., 0.]))],
                       centers, rotations, intrinsics) is not None
    assert triangulate([(0, np.array([0., 0.])), (1, np.array([20., 0.]))],
                       centers, rotations, intrinsics) is None
