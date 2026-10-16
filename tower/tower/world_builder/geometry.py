"""Two-view geometry primitives, and the degeneracy criterion.

Separated from the backends so the criterion is testable on its own and so
both backends measure identically -- the uncalibrated one still reports
what it can, it just cannot turn those measurements into a pose.

The thresholds here were measured on synthetic scenes with exact ground
truth (tests/synthetic_scene.py). They are documented starting points, not
tuned constants, and the measurements that produced them are recorded in
tests/test_world_builder_geometry.py so a successor can see why these
numbers and not others.
"""

import cv2
import math

import numpy as np

# Minimum median triangulation angle for translation to be trustworthy.
#
# This is the textbook conditioning criterion -- the angle each 3-D point
# subtends between the two viewing rays -- and it is used here in
# preference to pixel displacement, which is actively misleading: a pure
# rotation produces LARGE pixel motion (measured 18.4 px) with zero
# baseline, so a displacement-based gate passes exactly the case it exists
# to catch.
#
# Measured separation on synthetic ground truth: pairs whose recovered
# translation direction was wrong by 62-106 degrees had median
# triangulation angles of 0.11-0.42 deg, while every pair that recovered
# direction to within ~1 deg measured 0.54 deg or more.
MIN_TRIANGULATION_ANGLE_DEG = 0.5

# Secondary guard. An essential-matrix fit that retains almost no
# correspondences has not found a consistent geometry, whatever angle the
# few survivors happen to subtend. Measured: degenerate pairs kept
# 0.1-2.5% of matches, usable pairs kept 7.7-95.8%.
MIN_INLIER_RATIO = 0.05

# Below this there is not enough evidence to fit anything at all.
MIN_INLIERS = 15

# The per-landmark reprojection bar. This is the same 3.0 px budget
# solvePnPRansac already uses to call a POSE an inlier
# (classical.PNP_REPROJECTION_ERROR_PX), applied to the landmark that
# comes out.
MAX_LANDMARK_REPROJECTION_PX = 3.0

ORB_FEATURES = 1500
LOWE_RATIO = 0.75
RANSAC_THRESHOLD_PX = 1.0
RANSAC_CONFIDENCE = 0.999


def detect_and_describe(gray: np.ndarray, nfeatures: int = ORB_FEATURES):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    return orb.detectAndCompute(gray, None)


def match_descriptors(
    keypoints_a, descriptors_a, keypoints_b, descriptors_b, ratio: float = LOWE_RATIO
) -> tuple[np.ndarray, np.ndarray]:
    """Lowe ratio-test correspondences.

    Ratio test rather than crossCheck: on repetitive indoor texture
    crossCheck was measured to collapse the essential-matrix inlier ratio
    to 0.256 where the ratio test reached 0.941 on the same pair. Same
    0.75 threshold scripts/feature_trackability.py already uses.
    """
    empty = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    if descriptors_a is None or descriptors_b is None:
        return empty
    if len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return empty

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    good = [
        pair[0]
        for pair in matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]
    if not good:
        return empty
    points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good]).reshape(-1, 2)
    points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return points_a, points_b


def match_indices(
    descriptors_a, descriptors_b, ratio: float = LOWE_RATIO
) -> list[tuple[int, int]]:
    """Ratio-test matches as INDEX pairs rather than coordinates.

    Incremental SfM needs feature identity, not just position: a landmark
    triangulated from frames (i-1, i) can only be reused to solve frame
    i+1 by PnP if we can say which feature in frame i it belongs to.
    Returning coordinates throws that identity away.
    """
    if descriptors_a is None or descriptors_b is None:
        return []
    if len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return [
        (pair[0].queryIdx, pair[0].trainIdx)
        for pair in matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]


def homography_ratio(points_a: np.ndarray, points_b: np.ndarray) -> float | None:
    """ORB-SLAM's r_H = H_inliers / (H_inliers + F_inliers).

    Recorded, and NOT used as a gate. Measured on synthetic indoor scenes
    it saturates at 0.471-0.499 across the full range from total
    degeneracy to healthy parallax, so the conventional 0.45 threshold
    classifies every pair as rotation-dominant and separates nothing.

    That is not a defect in r_H; it is r_H answering its actual question.
    A homography suffices whenever the scene is EITHER purely rotating OR
    dominated by a plane -- and a room is nothing but planes. In a
    plane-dominated environment r_H cannot isolate rotation, which is
    precisely the thing we need isolated.
    """
    if len(points_a) < 8:
        return None

    # THE MODEL IS CHECKED BEFORE THE MASK.
    #
    # When RANSAC cannot fit at all it returns model=None and, on OpenCV
    # 5, leaves the mask buffer UNWRITTEN rather than zeroing it. Reading
    # it then yields whatever was in that memory: measured on a real pair
    # (capture 22e9d428, keyframes 00000345 x 00001824), 242 Lowe matches
    # landing on 3 distinct locations in a keyframe holding only 5 ORB
    # features, so neither F (needs 8 in general position) nor H (needs 4)
    # can fit. The mask came back uint8 with 46 distinct values summing to
    # 9552 -- on 242 elements, where a binary mask cannot exceed 242.
    #
    # It is NON-DETERMINISTIC and self-heals within a warm process: the
    # buffer is dirty on first use and zeroed after, so 50 identical
    # calls in one process show it once. Measured one call per FRESH
    # process: 30/40 corrupt in one run, 37/40 and 34/40 in others, with
    # sums to 32,880. `mask.sum()` on garbage is a plausible-looking
    # number, so nothing raises.
    h_model, h_mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 3.0)
    f_model, f_mask = cv2.findFundamentalMat(
        points_a, points_b, cv2.FM_RANSAC, 3.0, 0.99
    )
    h_inliers = (
        int(h_mask.sum()) if h_model is not None and h_mask is not None else 0
    )
    f_inliers = (
        int(f_mask.sum()) if f_model is not None and f_mask is not None else 0
    )
    total = h_inliers + f_inliers
    if total == 0:
        return None
    return h_inliers / total


def median_triangulation_angle_deg(
    points_a: np.ndarray,
    points_b: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
) -> float | None:
    """Median angle subtended at each triangulated point by the two rays.

    Small angle means the two views are nearly the same viewpoint, so
    depth is poorly constrained no matter how many correspondences agree.
    """
    if len(points_a) < 2:
        return None
    projection_a = camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = camera_matrix @ np.hstack(
        [rotation, translation.reshape(3, 1)]
    )
    homogeneous = cv2.triangulatePoints(
        projection_a, projection_b, points_a.T, points_b.T
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        xyz = (homogeneous[:3] / homogeneous[3]).T
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if len(xyz) < 2:
        return None

    centre_b = (-rotation.T @ translation.reshape(3)).ravel()
    ray_a = xyz
    ray_b = xyz - centre_b
    norm_a = np.linalg.norm(ray_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(ray_b, axis=1, keepdims=True)
    usable = (norm_a[:, 0] > 1e-9) & (norm_b[:, 0] > 1e-9)
    if usable.sum() < 2:
        return None
    cosines = ((ray_a[usable] / norm_a[usable]) * (ray_b[usable] / norm_b[usable])).sum(
        axis=1
    )
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return float(np.median(angles))


def motion_direction(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Camera motion direction in the FIRST camera's frame.

    cv2.recoverPose returns [R|t] mapping points from camera 1 into camera
    2, so `t` is camera 1's position in camera 2's frame -- not the
    direction the camera travelled. The motion, in camera 1's frame, is
    -R.T @ t.

    Isolated in one function because getting it wrong is silent: a
    179 deg translation error can be entirely this conversion while the
    underlying geometry is correct to 0.075 deg.
    """
    direction = -rotation.T @ np.asarray(translation, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-12 else direction


def min_parallax_deg(camera_matrix, pixel_noise_px: float = RANSAC_THRESHOLD_PX):
    """The angle below which a two-view landmark carries no depth at all.

    For a two-view triangulation the relative depth uncertainty is

        sigma_d / d  =  sigma_px / (f * theta)

    with theta the angle the two camera centres subtend AT the landmark.
    Setting that to 1.0 -- an error bar as wide as the measurement, i.e.
    an error bar that reaches infinity -- gives

        theta_min = sigma_px / f

    At this platform's calibration (f ~ 438, sigma_px = RANSAC_THRESHOLD_PX
    = 1.0) that is 0.131 deg. Above it a landmark is imprecise; below it a
    landmark is not a measurement, and its distance is set by pixel noise.

    Deliberately NOT MIN_TRIANGULATION_ANGLE_DEG (0.5 deg). That constant
    answers a different question -- "is this PAIR good enough to trust a
    pose from" -- and 0.5 deg corresponds to a 26% depth error, which is
    imprecise but real geometry. Measured on the real corpus, gating
    landmarks at 0.5 deg discards 37-44% of all points while a gate at
    this bound discards 8-9%, and both recover essentially the same
    fragment legibility. It is the same trade as a loss grace of 3 frames,
    which destroys a third of the reconstruction
    (`keyframes.KeyframePolicy.loss_grace_frames`).

    Scales with focal length, so it stays correct if the delivered
    resolution ever changes -- unlike a hardcoded pixel or degree bar.
    """
    focal = 0.5 * (
        float(camera_matrix[0][0]) + float(camera_matrix[1][1])
    )
    if not math.isfinite(focal) or focal <= 0:
        # No usable calibration: fall back to the pair-level constant
        # rather than admitting everything.
        return MIN_TRIANGULATION_ANGLE_DEG
    return math.degrees(pixel_noise_px / focal)


def _camera_centre(pose):
    """World-frame position of a camera, given a world->camera pose."""
    rotation, translation = pose
    return -np.asarray(rotation, dtype=np.float64).T @ np.asarray(
        translation, dtype=np.float64
    ).reshape(3)


def landmark_gate(
    xyz,
    points_a,
    points_b,
    pose_a,
    pose_b,
    camera_matrix,
    min_angle_deg: float | None = None,
    max_reprojection_px: float = MAX_LANDMARK_REPROJECTION_PX,
):
    """Which triangulated landmarks are geometry, and why the rest are not.

    Two gates, both enforcing invariants this module already declares:

    1. The angle subtended AT the landmark by the two camera centres must
       reach `min_angle_deg`, defaulting to `min_parallax_deg(camera_matrix)`
       -- the angle at which depth uncertainty reaches 100%. Below it the
       rays are parallel to within pixel noise and where they "meet" is
       set by that noise, not by the scene.
    2. The landmark must reproject into BOTH source views within
       `max_reprojection_px`.

    Why it matters, measured on real captures: without these, landmarks
    survive at up to 33,363 baselines from the camera, 12.2% of a real
    world sits beyond the 50-baseline cull horizon, and the resulting
    bounding box is 329x larger than the geometry inside it. The phone
    fits each fragment card to that box, so the room collapses to a dot.

    `xyz` is (N,3) in the frame the poses map FROM. `points_a`/`points_b`
    are (N,2) observed pixels. Each pose is (rotation (3,3),
    translation (3,)) mapping world->camera.

    Returns (keep, counts). Every rejected landmark is counted under
    exactly ONE reason: gate 1 is evaluated first, so a point failing both
    is a low_parallax point. That keeps
    `kept + low_parallax + high_reprojection == len(xyz)` exact.
    """
    # Shape is validated, not coerced. A (3, N) array reshapes to (N, 3)
    # perfectly cleanly and yields a mask of the CORRECT LENGTH, so a
    # transposed argument would sail through, reject every landmark, and
    # leave counts that look self-consistent. `_triangulate_new` builds its
    # observation arrays transposed (backends/classical.py), so this is one
    # careless argument away, and it fails toward silent deletion.
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz must be (N, 3), got {xyz.shape}. If this is (3, N), "
            f"transpose it -- do not let it reshape."
        )
    for name, observed in (("points_a", points_a), ("points_b", points_b)):
        shape = np.asarray(observed).shape
        if shape != (len(xyz), 2):
            raise ValueError(
                f"{name} must be ({len(xyz)}, 2) to match xyz, got {shape}."
            )
    if min_angle_deg is None:
        min_angle_deg = min_parallax_deg(camera_matrix)
    counts = {"low_parallax": 0, "high_reprojection": 0}
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool), counts

    ray_a = _camera_centre(pose_a) - xyz
    ray_b = _camera_centre(pose_b) - xyz
    with np.errstate(divide="ignore", invalid="ignore"):
        cosines = np.einsum("ij,ij->i", ray_a, ray_b) / (
            np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
        )
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    # A zero baseline, or a camera centre coincident with the landmark,
    # makes the cosine non-finite. Treat that as zero parallax rather than
    # letting a NaN compare False by accident.
    angles = np.where(np.isfinite(angles), angles, 0.0)
    parallax_ok = angles >= min_angle_deg

    def _reprojection_error(pose, observed):
        rotation, translation = pose
        cam = (np.asarray(rotation, dtype=np.float64) @ xyz.T).T + np.asarray(
            translation, dtype=np.float64
        ).reshape(3)
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (camera_matrix @ cam.T).T
            uv = uv[:, :2] / uv[:, 2:3]
        error = np.linalg.norm(
            uv - np.asarray(observed, dtype=np.float64), axis=1
        )
        # Behind the camera or on the principal plane: not a small error,
        # no error at all. Refuse rather than admit on a NaN comparison.
        return np.where(np.isfinite(error), error, np.inf)

    reprojection_ok = (
        _reprojection_error(pose_a, points_a) <= max_reprojection_px
    ) & (_reprojection_error(pose_b, points_b) <= max_reprojection_px)

    keep = parallax_ok & reprojection_ok
    counts["low_parallax"] = int((~parallax_ok).sum())
    counts["high_reprojection"] = int((parallax_ok & ~reprojection_ok).sum())
    return keep, counts


def triangulate_points(
    points_a: np.ndarray,
    points_b: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    return_mask: bool = False,
    return_quality: bool = False,
):
    """Triangulate into the FIRST camera's frame, dropping bad points.

    Points behind either camera, or non-finite, are removed rather than
    returned with a flag: a point the geometry says is behind the camera
    is not a low-confidence point, it is not a point.

    `return_quality` additionally yields a per-surviving-point boolean --
    True where `landmark_gate` says the landmark's position is defensible
    -- and the refusal counts. Quality is REPORTED, not applied: see the
    comment in the body for why a landmark that fails the gate still
    belongs in the solver's map.
    """
    projection_a = camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = camera_matrix @ np.hstack([rotation, translation.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(
        projection_a, projection_b, points_a.T, points_b.T
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        xyz = (homogeneous[:3] / homogeneous[3]).T

    in_front_a = xyz[:, 2] > 0
    xyz_b = (rotation @ xyz.T).T + translation.reshape(3)
    in_front_b = xyz_b[:, 2] > 0
    keep = np.isfinite(xyz).all(axis=1) & in_front_a & in_front_b

    # The gate JUDGES; it does not filter here.
    #
    # Dropping failing landmarks here, measured over the pinned eight,
    # costs 26 solved poses (346 -> 320) across four captures: chain
    # extension needs >= MIN_PNP_CORRESPONDENCES 3-D to 2-D
    # correspondences against EXISTING landmarks, so removing them from
    # the map starves PnP and breaks chains early.
    #
    # A landmark whose depth is unconstrained is still a perfectly good
    # correspondence anchor -- its bearing is fine, only its distance is
    # not. So it stays in the map for solving, and the caller declines to
    # PUBLISH a coordinate it cannot defend. Refusing to publish a number
    # is honest; refusing to solve with a usable bearing is just lossy.
    quality = np.zeros(int(keep.sum()), dtype=bool)
    counts = {"low_parallax": 0, "high_reprojection": 0}
    if keep.any():
        pa = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
        pb = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
        quality, counts = landmark_gate(
            xyz[keep],
            pa[keep],
            pb[keep],
            (np.eye(3), np.zeros(3)),
            (rotation, translation),
            camera_matrix,
        )

    result = (xyz[keep],)
    if return_mask:
        result += (keep,)
    if return_quality:
        result += (quality, counts)
    return result[0] if len(result) == 1 else result
