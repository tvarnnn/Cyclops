"""Backend for the uncalibrated case: measures, but never poses.

This is the honest answer to the platform's actual situation. Meta DAT 0.9
exposes no intrinsics -- VideoFrame carries a sample buffer and nothing
else -- and the published Ray-Ban field of view describes a 3:4 still,
while the stream is 9:16 through an undocumented crop, so no legitimate
conversion exists. Without intrinsics, an essential matrix cannot be
formed, and a guessed focal length does not degrade the result gracefully:
it yields a bent trajectory that looks entirely plausible.

So this backend returns no poses at all. What it does return is every
measurement that genuinely does not need calibration -- correspondences,
fundamental-matrix inliers, pixel displacement, r_H -- because those are
real evidence about the footage, and because a world that records why it
has no geometry is far more useful than one that is merely empty.

The moment a ChArUco calibration exists, select_backend() picks the
classical backend instead and the same pipeline starts producing poses.
Nothing above this seam changes.
"""

from collections.abc import Sequence

import cv2
import numpy as np

from tower.world_builder.backend import (
    BackendCapabilities,
    Extension,
    GeometryBackend,
    GeometryEstimate,
    KeyframeInput,
    PoseEstimate,
)
from tower.world_builder.geometry import (
    RANSAC_THRESHOLD_PX,
    detect_and_describe,
    homography_ratio,
    match_descriptors,
)
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.schema import (
    DEGENERACY_NO_INTRINSICS,
    POSE_STATUS_ANCHOR,
    POSE_STATUS_UNAVAILABLE,
)

CAPABILITIES = BackendCapabilities(
    backend_id="unposed",
    version="1",
    requires_intrinsics=False,
    estimates_intrinsics=False,
    produces_dense_geometry=False,
    produces_metric_scale=False,
    preferred_window=2,
    device="cpu",
)


# One constant, because both the from-scratch and the incremental path
# report it and a consumer reading two different strings from the same
# backend would have to guess which one means what.
WITHHELD = "no intrinsics; poses withheld"


class UnposedBackend(GeometryBackend):
    capabilities = CAPABILITIES

    def __init__(self) -> None:
        self._previous_features = None
        self._poses: list[PoseEstimate] = []

    def prepare(self, intrinsics: CameraIntrinsics) -> None:
        # Accepts anything, including unknown intrinsics -- that is the
        # entire point of this backend.
        return None

    def estimate_window(
        self, window: Sequence[KeyframeInput]
    ) -> GeometryEstimate:
        if not window:
            return GeometryEstimate(poses=())

        # The first keyframe anchors the window frame. It gets ANCHOR
        # rather than SOLVED: its pose is definitional, not estimated, and
        # conflating the two would let a downstream consumer count it as
        # evidence.
        poses = [PoseEstimate(keyframe_id=window[0].keyframe_id, status=POSE_STATUS_ANCHOR)]

        previous_features = detect_and_describe(window[0].image_gray)

        for current in window[1:]:
            current_features = detect_and_describe(current.image_gray)
            poses.append(
                self._measure(
                    previous_features, current_features, current.keyframe_id
                )
            )
            previous_features = current_features

        return GeometryEstimate(
            poses=tuple(poses), points=None, diagnostics={"reason": WITHHELD}
        )

    # -- the incremental seam --------------------------------------------
    #
    # The engine drives every backend through begin/extend/snapshot, and the
    # GeometryBackend default re-runs the whole window per keyframe. Here a
    # measurement depends only on the previous frame, so this override makes
    # extend O(1); every pose stays withheld and no points are returned.

    def begin(self, intrinsics: CameraIntrinsics) -> None:
        self.prepare(intrinsics)
        self.reset()

    def reset(self) -> None:
        self._previous_features = None
        self._poses = []

    def extend(self, frame: KeyframeInput) -> Extension:
        features = detect_and_describe(frame.image_gray)
        if self._previous_features is None:
            pose = PoseEstimate(
                keyframe_id=frame.keyframe_id, status=POSE_STATUS_ANCHOR
            )
        else:
            pose = self._measure(self._previous_features, features, frame.keyframe_id)
        self._previous_features = features
        self._poses.append(pose)
        # Never any points, by construction.
        return Extension(pose=pose)

    def snapshot(self) -> GeometryEstimate:
        if not self._poses:
            return GeometryEstimate(poses=())
        return GeometryEstimate(
            poses=tuple(self._poses), points=None, diagnostics={"reason": WITHHELD}
        )

    # -- helpers ----------------------------------------------------------

    def _measure(self, previous_features, current_features, keyframe_id):
        """Everything that genuinely does not need calibration."""
        points_a, points_b = match_descriptors(*previous_features, *current_features)

        inliers = 0
        displacement = None
        if len(points_a) >= 8:
            _, mask = cv2.findFundamentalMat(
                points_a, points_b, cv2.FM_RANSAC, RANSAC_THRESHOLD_PX, 0.99
            )
            if mask is not None:
                kept = mask.ravel() > 0
                inliers = int(kept.sum())
                if inliers:
                    displacement = float(
                        np.median(
                            np.linalg.norm(points_a[kept] - points_b[kept], axis=1)
                        )
                    )

        return PoseEstimate(
            keyframe_id=keyframe_id,
            status=POSE_STATUS_UNAVAILABLE,
            degeneracy=DEGENERACY_NO_INTRINSICS,
            matches=len(points_a),
            inliers=inliers,
            inlier_ratio=(inliers / len(points_a) if len(points_a) else None),
            median_displacement_px=displacement,
            r_h=homography_ratio(points_a, points_b),
        )
