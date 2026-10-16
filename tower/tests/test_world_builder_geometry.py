"""Geometry correctness against exact synthetic ground truth.

SYNTHETIC, NOT PHYSICAL. These assert that the geometry CODE is correct --
that a known rotation comes back as that rotation, that a degenerate pair
is refused. They assert nothing about the Ray-Ban camera, whose optics,
distortion, rolling shutter and exposure behaviour are all untested and
must remain so until physical footage exists.

Tolerances are motion-dependent on purpose. Lateral motion recovers
translation direction to a fraction of a degree; forward motion -- which
is what a walking person actually does -- puts the epipole inside the
image and was measured at up to 9.7 deg on clean synthetic features. A
single global tolerance would be either vacuous or wrong.
"""

import cv2
import numpy as np
import pytest

from tests import synthetic_scene as ss
from tower.world_builder.backend import KeyframeInput
from tower.world_builder.backends import (
    BACKEND_AUTO,
    BACKEND_CLASSICAL,
    ClassicalTwoViewBackend,
    UnposedBackend,
    select_backend,
)
from tower.world_builder.geometry import (
    MIN_TRIANGULATION_ANGLE_DEG,
    detect_and_describe,
    homography_ratio,
    match_descriptors,
    motion_direction,
)
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.schema import (
    INTRINSICS_SOURCE_SELF_CALIBRATED,
    POSE_STATUS_ANCHOR,
    POSE_STATUS_ROTATION_ONLY,
    POSE_STATUS_SOLVED,
    POSE_STATUS_UNAVAILABLE,
)

WIDTH, HEIGHT = 480, 360


@pytest.fixture(scope="module")
def scene():
    return ss.furnished_room()


@pytest.fixture(scope="module")
def camera_matrix():
    return ss.camera_matrix(WIDTH, HEIGHT)


@pytest.fixture(scope="module")
def intrinsics(camera_matrix):
    """Ground-truth intrinsics -- this is the calibrated path."""
    return CameraIntrinsics(
        source=INTRINSICS_SOURCE_SELF_CALIBRATED,
        model="pinhole",
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]),
        cy=float(camera_matrix[1, 2]),
        calibrated_width=WIDTH,
        calibrated_height=HEIGHT,
    )


def _window(scene, camera_matrix, poses):
    images = ss.render_sequence(scene, poses, camera_matrix, WIDTH, HEIGHT)
    return [
        KeyframeInput(
            keyframe_id=f"s:{index:08d}",
            image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        )
        for index, image in enumerate(images)
    ]


def _solve(scene, camera_matrix, intrinsics, poses):
    backend = ClassicalTwoViewBackend()
    backend.prepare(intrinsics)
    return backend.estimate_window(_window(scene, camera_matrix, poses))


class TestCalibratedRecovery:
    def test_lateral_motion_recovers_rotation_and_direction(
        self, scene, camera_matrix, intrinsics
    ):
        poses = ss.strafe(2, step=0.30)
        estimate = _solve(scene, camera_matrix, intrinsics, poses)
        solved = estimate.poses[1]
        assert solved.status == POSE_STATUS_SOLVED

        expected_rotation, expected_direction = ss.relative_pose(poses[0], poses[1])
        assert ss.rotation_error_deg(solved.rotation, expected_rotation) < 1.0
        assert (
            ss.direction_error_deg(
                motion_direction(solved.rotation, solved.translation),
                expected_direction,
            )
            < 3.0
        )

    def test_forward_motion_recovers_pose_with_a_looser_bound(
        self, scene, camera_matrix, intrinsics
    ):
        """Walking forward is the realistic AND the hard case.

        The epipole sits inside the image, so translation direction is
        far less constrained than for lateral motion. Measured up to
        9.7 deg on clean synthetic features -- this bound records that
        reality rather than pretending the two motions are equivalent.
        """
        poses = ss.forward_walk(2, step=0.30)
        estimate = _solve(scene, camera_matrix, intrinsics, poses)
        solved = estimate.poses[1]
        assert solved.status == POSE_STATUS_SOLVED

        expected_rotation, expected_direction = ss.relative_pose(poses[0], poses[1])
        assert ss.rotation_error_deg(solved.rotation, expected_rotation) < 1.0
        assert (
            ss.direction_error_deg(
                motion_direction(solved.rotation, solved.translation),
                expected_direction,
            )
            < 15.0
        )

    def test_the_first_keyframe_is_an_anchor_not_a_solved_pose(
        self, scene, camera_matrix, intrinsics
    ):
        """Its pose is definitional. Counting it as evidence would inflate
        every downstream statistic about how much was actually solved."""
        estimate = _solve(scene, camera_matrix, intrinsics, ss.strafe(2, step=0.30))

        assert estimate.poses[0].status == POSE_STATUS_ANCHOR
        assert estimate.poses[0].rotation is None

    def test_triangulated_points_are_produced_and_finite(
        self, scene, camera_matrix, intrinsics
    ):
        estimate = _solve(scene, camera_matrix, intrinsics, ss.strafe(2, step=0.30))

        assert estimate.points is not None
        assert len(estimate.points) > 50
        assert np.isfinite(estimate.points.xyz).all()

    def test_triangulated_structure_matches_truth_up_to_a_similarity(
        self, scene, camera_matrix, intrinsics
    ):
        """Monocular reconstruction is only ever correct up to a similarity.

        Comparing raw coordinates would fail a perfectly correct result,
        so the comparison aligns first and then measures residual shape
        error -- which is the actual claim being made.
        """
        poses = ss.strafe(3, step=0.30)
        window = _window(scene, camera_matrix, poses)
        backend = ClassicalTwoViewBackend()
        backend.prepare(intrinsics)
        estimate = backend.estimate_window(window[:2])
        assert estimate.points is not None

        _, depth = ss.render(scene, poses[0], camera_matrix, WIDTH, HEIGHT)
        reconstructed = estimate.points.xyz.astype(np.float64)

        # Project reconstructed points back and compare their depth with
        # the analytic depth buffer at the same pixel.
        projected = reconstructed @ camera_matrix.T
        uv = projected[:, :2] / projected[:, 2:3]
        inside = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < WIDTH - 1)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < HEIGHT - 1)
        )
        uv, reconstructed = uv[inside], reconstructed[inside]
        truth = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
        usable = np.isfinite(truth)
        assert usable.sum() > 50

        # One global scale, since the reconstruction is unit-free.
        scale = np.median(truth[usable] / reconstructed[usable, 2])
        relative_error = np.abs(
            reconstructed[usable, 2] * scale - truth[usable]
        ) / truth[usable]
        assert float(np.median(relative_error)) < 0.10


class TestDegeneracyIsRefused:
    def test_pure_rotation_yields_rotation_only_and_no_translation(
        self, scene, camera_matrix, intrinsics
    ):
        """The failure this prevents is a confident, mirrored, wrong pose.

        Under zero baseline recoverPose still returns a translation. Its
        direction is meaningless -- measured 62 and 106 degrees of error
        on pairs that reported no other complaint.
        """
        poses = ss.pure_rotation(2, degrees_per_step=2.0)
        estimate = _solve(scene, camera_matrix, intrinsics, poses)
        refused = estimate.poses[1]

        assert refused.status == POSE_STATUS_ROTATION_ONLY
        assert refused.translation is None
        assert refused.degeneracy != ""

    def test_rotation_survives_even_when_translation_is_refused(
        self, scene, camera_matrix, intrinsics
    ):
        """Rotation is well-conditioned at zero baseline; translation is not.

        Discarding both would throw away real information.
        """
        poses = ss.pure_rotation(2, degrees_per_step=2.0)
        estimate = _solve(scene, camera_matrix, intrinsics, poses)
        refused = estimate.poses[1]

        assert refused.rotation is not None
        expected_rotation, _ = ss.relative_pose(poses[0], poses[1])
        assert ss.rotation_error_deg(refused.rotation, expected_rotation) < 2.0

    def test_no_points_are_triangulated_from_a_degenerate_pair(
        self, scene, camera_matrix, intrinsics
    ):
        estimate = _solve(
            scene, camera_matrix, intrinsics, ss.pure_rotation(2, 2.0)
        )

        assert estimate.points is None

    def test_a_tiny_baseline_is_refused_as_low_parallax(
        self, scene, camera_matrix, intrinsics
    ):
        """5 mm of travel is not enough to triangulate a room."""
        angle = np.deg2rad(2.0)
        start = ss.look_at((0.0, -1.6, 1.2), (0.0, -1.6, 2.2))
        nudged = ss.look_at(
            (0.005, -1.6, 1.2),
            (0.005 + np.sin(angle), -1.6, 1.2 + np.cos(angle)),
        )
        estimate = _solve(scene, camera_matrix, intrinsics, [start, nudged])

        assert estimate.poses[1].status == POSE_STATUS_ROTATION_ONLY

    def test_blank_frames_report_no_correspondence(
        self, camera_matrix, intrinsics
    ):
        blank = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        backend = ClassicalTwoViewBackend()
        backend.prepare(intrinsics)

        estimate = backend.estimate_window(
            [
                KeyframeInput("s:00000000", blank),
                KeyframeInput("s:00000001", blank.copy()),
            ]
        )

        assert estimate.poses[1].status == POSE_STATUS_UNAVAILABLE
        assert estimate.poses[1].degeneracy == "no_correspondence"


class TestDegeneracyCriterionEvidence:
    """The measurements that justify the criterion, kept executable.

    A threshold whose supporting evidence lives only in a commit message
    gets changed by someone who never sees the evidence.
    """

    def test_triangulation_angle_separates_degenerate_from_usable(
        self, scene, camera_matrix, intrinsics
    ):
        angle = np.deg2rad(2.0)

        def pair_for(baseline):
            start = ss.look_at((0.0, -1.6, 1.2), (0.0, -1.6, 2.2))
            moved = ss.look_at(
                (baseline, -1.6, 1.2),
                (baseline + np.sin(angle), -1.6, 1.2 + np.cos(angle)),
            )
            return _solve(scene, camera_matrix, intrinsics, [start, moved]).poses[1]

        degenerate = pair_for(0.005)
        usable = pair_for(0.20)

        assert degenerate.status == POSE_STATUS_ROTATION_ONLY
        assert usable.status == POSE_STATUS_SOLVED
        assert usable.median_triangulation_deg > MIN_TRIANGULATION_ANGLE_DEG

    def test_pixel_displacement_would_have_passed_the_degenerate_pair(
        self, scene, camera_matrix
    ):
        """Why the gate is not median pixel displacement.

        A pure rotation moves every pixel a long way while contributing
        zero baseline, so a displacement threshold passes precisely the
        case it exists to catch. Measured here on a genuinely degenerate
        pair.
        """
        poses = ss.pure_rotation(2, degrees_per_step=2.0)
        images = ss.render_sequence(scene, poses, camera_matrix, WIDTH, HEIGHT)
        grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
        points_a, points_b = match_descriptors(
            *detect_and_describe(grays[0]), *detect_and_describe(grays[1])
        )

        displacement = float(
            np.median(np.linalg.norm(points_a - points_b, axis=1))
        )

        assert displacement > 5.0

    def test_r_h_does_not_separate_and_is_recorded_not_gated(
        self, scene, camera_matrix
    ):
        """Why ORB-SLAM's r_H threshold is not used as the gate here.

        r_H asks whether a homography suffices, which is true both for
        pure rotation AND for a plane-dominated scene. A room is nothing
        but planes, so r_H saturates: measured 0.471-0.499 across the full
        range from total degeneracy to healthy parallax, leaving every
        pair on the same side of the conventional 0.45 threshold.
        """
        def ratio_for(poses):
            images = ss.render_sequence(scene, poses, camera_matrix, WIDTH, HEIGHT)
            grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
            points_a, points_b = match_descriptors(
                *detect_and_describe(grays[0]), *detect_and_describe(grays[1])
            )
            return homography_ratio(points_a, points_b)

        degenerate = ratio_for(ss.pure_rotation(2, 2.0))
        healthy = ratio_for(ss.strafe(2, step=0.30))

        assert degenerate is not None and healthy is not None
        # Both land on the same side of 0.45: r_H cannot separate them.
        assert abs(degenerate - healthy) < 0.15


class TestBackendSelection:
    def test_auto_without_intrinsics_picks_the_unposed_backend_loudly(self):
        """AUTO landing on unposed is reported, like every other route to it.

        This used to assert `not was_downgraded`, on the reasoning that
        AUTO choosing the only backend it CAN choose is a selection
        rather than a downgrade. That reasoning is defensible and it cost
        a night: on the 2026-08-24 physical walk AUTO landed here, the
        session recorded `downgraded_from: null`, and the world came back
        with 155 keyframes, zero poses and zero points with nothing
        anywhere saying why.

        The sibling test below already says a silent downgrade "would
        leave an operator with an empty map and no way to learn that
        calibration was the reason". That was true of this path too, and
        the two tests now agree.
        """
        selection = select_backend(BACKEND_AUTO, CameraIntrinsics.unknown())

        assert isinstance(selection.backend, UnposedBackend)
        assert selection.was_downgraded
        assert selection.downgraded_from == BACKEND_CLASSICAL
        assert "intrinsics" in selection.downgrade_reason

    def test_auto_with_intrinsics_picks_the_classical_backend(self, intrinsics):
        selection = select_backend(BACKEND_AUTO, intrinsics)

        assert isinstance(selection.backend, ClassicalTwoViewBackend)

    def test_requesting_classical_without_intrinsics_downgrades_loudly(self):
        """A silent downgrade would leave an operator with an empty map and
        no way to learn that calibration was the reason."""
        selection = select_backend(BACKEND_CLASSICAL, CameraIntrinsics.unknown())

        assert isinstance(selection.backend, UnposedBackend)
        assert selection.was_downgraded
        assert selection.downgraded_from == BACKEND_CLASSICAL
        assert "intrinsics" in selection.downgrade_reason

    def test_classical_backend_refuses_to_prepare_without_intrinsics(self):
        with pytest.raises(ValueError, match="requires known intrinsics"):
            ClassicalTwoViewBackend().prepare(CameraIntrinsics.unknown())

    def test_unknown_backend_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown backend"):
            select_backend("magic", CameraIntrinsics.unknown())


class TestUnposedBackend:
    def test_returns_no_poses_but_still_reports_measurements(
        self, scene, camera_matrix
    ):
        """An empty map that explains itself beats an empty map that does not."""
        backend = UnposedBackend()
        backend.prepare(CameraIntrinsics.unknown())

        estimate = backend.estimate_window(
            _window(scene, camera_matrix, ss.strafe(2, step=0.30))
        )
        measured = estimate.poses[1]

        assert measured.status == POSE_STATUS_UNAVAILABLE
        assert measured.rotation is None
        assert measured.translation is None
        assert measured.degeneracy == "no_intrinsics"
        assert measured.matches > 100
        assert measured.inliers > 50
        assert measured.median_displacement_px is not None

    def test_produces_no_geometry(self, scene, camera_matrix):
        backend = UnposedBackend()
        backend.prepare(CameraIntrinsics.unknown())

        estimate = backend.estimate_window(
            _window(scene, camera_matrix, ss.strafe(2, step=0.30))
        )

        assert estimate.points is None


class TestBackendIsolation:
    def test_backends_do_not_import_storage_or_engine(self):
        """The seam only works if geometry never learns about disk.

        If a backend imports the store, swapping in a learned backend
        stops being a one-file change.
        """
        import ast
        import pathlib

        root = pathlib.Path("tower/world_builder")
        forbidden = {"store", "engine", "paths", "capture"}
        offenders = []

        for path in [root / "backend.py", *(root / "backends").glob("*.py")]:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                for name in names:
                    tail = name.split(".")[-1]
                    if name.startswith("tower.world_builder") and tail in forbidden:
                        offenders.append(f"{path.name} -> {name}")

        assert offenders == []


class TestMultiFrameTrajectory:
    """Regression cover for a fabricated-trajectory bug.

    An earlier backend chained independent two-view solutions. Every
    recoverPose translation is unit-length, so the resulting path had a
    constant step length regardless of how far the camera actually
    travelled -- a confident, smooth, entirely invented trajectory. No
    test caught it because every test used exactly two keyframes.

    These use FOUR keyframes with deliberately UNEQUAL spacing, which is
    the only shape that can distinguish real scale propagation from
    normalisation.
    """

    def _unequal_walk(self):
        return [
            ss.look_at((0.00, -1.6, 0.6), (0.00, -1.6, 1.6)),
            ss.look_at((0.30, -1.6, 0.6), (0.30, -1.6, 1.6)),
            ss.look_at((0.90, -1.6, 0.6), (0.90, -1.6, 1.6)),
            ss.look_at((1.20, -1.6, 0.6), (1.20, -1.6, 1.6)),
        ]

    def _centres(self, estimate):
        """Camera centres in the anchor frame: C = -R.T @ t."""
        centres = [np.zeros(3)]
        for pose in estimate.poses[1:]:
            if pose.rotation is None or pose.translation is None:
                return None
            centres.append((-pose.rotation.T @ pose.translation).ravel())
        return np.asarray(centres)

    def test_unequal_steps_are_not_normalised_to_equal_length(
        self, scene, camera_matrix, intrinsics
    ):
        """The bug this exists to catch, stated directly.

        True spacing is 0.30 / 0.60 / 0.30, so the middle step is twice
        the others. A chain of normalised two-view translations reports
        1 / 1 / 1.
        """
        poses = self._unequal_walk()
        estimate = _solve(scene, camera_matrix, intrinsics, poses)
        centres = self._centres(estimate)
        assert centres is not None

        steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
        ratios = steps / steps[0]

        assert ratios[1] > 1.4, f"middle step not recovered as longer: {ratios}"
        assert ratios[2] < 1.4, f"final step not recovered as shorter: {ratios}"

    def test_every_pose_is_expressed_in_the_anchor_frame(
        self, scene, camera_matrix, intrinsics
    ):
        """Poses must be anchor-relative, not predecessor-relative.

        Predecessor-relative poses are each individually correct and
        collectively meaningless, because their translations carry no
        common scale.
        """
        poses = self._unequal_walk()
        estimate = _solve(scene, camera_matrix, intrinsics, poses)
        centres = self._centres(estimate)
        assert centres is not None

        # Monotonic travel away from the anchor, since the walk is one way.
        distances = np.linalg.norm(centres, axis=1)
        assert np.all(np.diff(distances) > 0)

    def test_trajectory_matches_truth_after_similarity_alignment(
        self, scene, camera_matrix, intrinsics
    ):
        """Drift measured against ground truth -- the BA trigger metric.

        Monocular reconstruction is correct only up to a similarity, so
        the comparison aligns first. Measured residual at the time of
        writing: 1.32% of path length over a 1.2 m walk, with the 2x step
        recovered as 1.79x rather than 2.00x. That error is real and is
        exactly what bundle adjustment would reduce; the bound here is
        deliberately loose so it documents current behaviour rather than
        freezing it.
        """
        poses = self._unequal_walk()
        estimate = _solve(scene, camera_matrix, intrinsics, poses)
        centres = self._centres(estimate)
        assert centres is not None

        truth = np.asarray([pose.position for pose in poses])
        scale, rotation, translation = ss.umeyama_similarity(centres, truth)
        aligned = (scale * (rotation @ centres.T).T) + translation

        residual = np.linalg.norm(aligned - truth, axis=1)
        path_length = float(
            np.linalg.norm(np.diff(truth, axis=0), axis=1).sum()
        )

        assert residual.max() / path_length < 0.10

    def test_a_long_window_produces_points_in_one_frame(
        self, scene, camera_matrix, intrinsics
    ):
        """Points from every pair must share the anchor frame.

        Concatenating per-pair triangulations, each in its own camera's
        frame at its own arbitrary unit, produces a cloud that looks
        populated and means nothing.
        """
        estimate = _solve(
            scene, camera_matrix, intrinsics, self._unequal_walk()
        )

        assert estimate.points is not None
        assert len(estimate.points) > 200
        # A single coherent frame keeps the cloud in front of the anchor
        # camera; mixed frames scatter points behind it.
        in_front = (estimate.points.xyz[:, 2] > 0).mean()
        assert in_front > 0.9

    def test_a_window_that_cannot_initialise_reports_all_unavailable(
        self, scene, camera_matrix, intrinsics
    ):
        """A chain that never starts must not silently report later poses."""
        poses = ss.pure_rotation(4, degrees_per_step=2.0)
        estimate = _solve(scene, camera_matrix, intrinsics, poses)

        assert estimate.poses[0].status == POSE_STATUS_ANCHOR
        for pose in estimate.poses[1:]:
            assert pose.status != POSE_STATUS_SOLVED
            assert pose.translation is None


class TestHomographyRatioRefusesRatherThanReadingGarbage:
    """r_H must never return a number RANSAC did not produce.

    When neither model can be fitted, OpenCV 5 returns model=None and
    leaves the mask buffer UNWRITTEN rather than zeroing it. Summing it
    then yields whatever was in that memory, which looks like a perfectly
    ordinary integer and so never raised.

    The trigger is real and was measured on this corpus: capture
    22e9d428, keyframes 00000345 x 00001824. Keyframe B holds only FIVE
    ORB features, the Lowe test still returns 242 matches because each of
    A's 1,321 features picks a nearest neighbour among those five, and
    those 242 matches land on 3 distinct locations. F needs 8 points in
    general position and H needs 4; three support neither.

    Observed before the fix: uint8 mask, 46 distinct values, sum 9552 on
    242 elements -- a binary mask over 242 elements cannot exceed 242.
    Non-deterministic across FRESH processes (30/40, 34/40 and 37/40
    corrupt in three separate runs) and self-healing within a warm one,
    which is why it survived so long and why it would have been miserable
    to debug later.
    """

    def _degenerate_pair(self):
        """242 correspondences on 3 distinct locations -- the real shape."""
        anchors = np.array(
            [[100.0, 100.0], [200.0, 140.0], [150.0, 260.0]], dtype=np.float32
        )
        points_a = np.repeat(anchors, 81, axis=0)[:242]
        points_b = np.repeat(anchors[::-1], 81, axis=0)[:242]
        return points_a, points_b

    def test_a_pair_that_fits_no_model_returns_none(self):
        value = homography_ratio(*self._degenerate_pair())
        assert value is None, (
            f"expected a refusal, got {value!r}. A ratio here means the mask "
            "was read without checking whether RANSAC produced a model."
        )

    def test_it_is_a_ratio_or_nothing_never_an_arbitrary_number(self):
        """The invariant, stated so any future edit has to keep it."""
        for _ in range(20):
            value = homography_ratio(*self._degenerate_pair())
            assert value is None or 0.0 <= value <= 1.0, (
                f"r_H escaped [0, 1]: {value!r}"
            )

    def test_a_healthy_pair_still_produces_a_ratio(self, scene, camera_matrix):
        """The fix must refuse the degenerate case WITHOUT refusing the
        working one, or it has just disabled the field."""
        window = _window(scene, camera_matrix, ss.strafe(2, step=0.12))
        first, second = (
            detect_and_describe(frame.image_gray) for frame in window[:2]
        )
        points_a, points_b = match_descriptors(
            first[0], first[1], second[0], second[1]
        )
        value = homography_ratio(points_a, points_b)
        assert value is not None and 0.0 <= value <= 1.0
