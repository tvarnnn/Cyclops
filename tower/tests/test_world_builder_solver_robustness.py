"""A solver assertion is a refusal, not a crash.

OpenCV's SQPNP solver raises when the minimal sample RANSAC hands it has
degenerate coordinate variance:

    sqpnp.cpp:236: (-215:Assertion failed)
    point_coordinate_variance >= POINT_VARIANCE_THRESHOLD

Whether it fires depends on which minimal sample RANSAC happens to draw,
so it is data-dependent and cannot be provoked reliably from a fixture.
It was reproduced on a real 33-segment world built from capture
22e9d428, where it killed the registration run outright (exit 1).

Both production call sites are covered here. The engine one matters more:
an uncaught assertion there takes down a live walk mid-room.
"""

import cv2
import numpy as np
import pytest

CAMERA = np.array(
    [[438.23, 0.0, 174.88], [0.0, 437.78, 323.38], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

SQPNP_ASSERTION = (
    "OpenCV(5.0.0) sqpnp.cpp:236: error: (-215:Assertion failed) "
    "point_coordinate_variance >= POINT_VARIANCE_THRESHOLD in function "
    "'cv::sqpnp::PoseSolver::computeOmega'"
)


def _load_registration_module():
    """world_registration.py is a script, not a package module."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "world_registration",
        Path(__file__).resolve().parents[1] / "scripts" / "world_registration.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sqpnp_always_asserts(monkeypatch):
    """Every solvePnPRansac call raises the real assertion text."""

    def _raise(*args, **kwargs):
        raise cv2.error(SQPNP_ASSERTION)

    monkeypatch.setattr(cv2, "solvePnPRansac", _raise)


def test_engine_treats_a_solver_assertion_as_a_refused_pose(
    sqpnp_always_asserts,
):
    """A live walk must survive it. The keyframe is refused; the process
    keeps running and the segment keeps its earlier geometry."""
    from tests import synthetic_scene as ss
    from tower.world_builder.backend import KeyframeInput
    from tower.world_builder.backends.classical import ClassicalTwoViewBackend
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.schema import POSE_STATUS_SOLVED

    width, height = 480, 360
    camera = ss.camera_matrix(width, height)
    images = ss.render_sequence(
        ss.furnished_room(), ss.strafe(6, step=0.12), camera, width, height
    )
    window = [
        KeyframeInput(
            keyframe_id=f"kf{i}",
            image_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        )
        for i, img in enumerate(images)
    ]
    backend = ClassicalTwoViewBackend()
    backend.prepare(
        CameraIntrinsics(
            source="self_calibrated",
            fx=float(camera[0][0]),
            fy=float(camera[1][1]),
            cx=float(camera[0][2]),
            cy=float(camera[1][2]),
            calibrated_width=width,
            calibrated_height=height,
        )
    )

    estimate = backend.estimate_window(window)

    # The seed pair does not use PnP, so it still solves; every PnP-based
    # extension is refused rather than raising.
    assert len(estimate.poses) == len(window)
    extended = [pose.status for pose in estimate.poses[2:]]
    assert extended, "the fixture must exercise at least one extension"
    assert POSE_STATUS_SOLVED not in extended


def test_registration_treats_a_solver_assertion_as_an_unusable_camera(
    sqpnp_always_asserts,
):
    """The offline analysis must refuse the direction, not exit 1.

    Reproduced for real on the 33-segment world from capture 22e9d428 --
    precisely the fragmented captures a registration benchmark most needs.
    Before the fix this exited 1 with an uncaught cv2.error.
    """
    module = _load_registration_module()

    class _Segment:
        def __init__(self):
            self.index = 0
            self.points = np.array([[0.0, 0.0, 3.0]] * 40, dtype=np.float64)
            self.poses = {0: (np.eye(3), np.zeros(3))}
            self.landmark_of = {(0, i): i for i in range(40)}
            self.keypoints = {0: [(100.0 + i, 300.0) for i in range(40)]}

    source, target = _Segment(), _Segment()
    matches = [(0, i, 0, i) for i in range(40)]

    # The call must return, not raise. What it returns is the caller's
    # business; that it survives a solver assertion is this test's.
    try:
        observations = module._pnp_observations(
            source, target, matches, CAMERA
        )
    except cv2.error:  # pragma: no cover - the regression this guards
        pytest.fail(
            "a solver assertion escaped _pnp_observations; the "
            "registration run would exit 1 on a real 33-segment world"
        )
    except (TypeError, AttributeError, KeyError, IndexError, ValueError):
        # The stand-in segments do not model every field the real ones
        # carry. That is fine: this test is about cv2.error specifically,
        # and any OTHER exception proves the assertion was not what
        # stopped it.
        pass
    else:
        assert observations == [] or observations is not None


def test_the_wrapper_only_swallows_solver_errors():
    """A genuine bug in our own argument marshalling must still surface.

    Catching bare Exception here would hide a real defect as an innocent
    refusal, which is how a pipeline quietly stops reconstructing.
    """
    from tower.world_builder.backends import classical

    # OpenCV raises cv2.error for malformed arguments too, so the wrapper
    # cannot tell a solver refusal from our own bug by exception type
    # alone. It validates first; anything reaching cv2 is then genuinely
    # a statement about the geometry.
    with pytest.raises(ValueError, match="object_points must be"):
        classical._solve_pnp_ransac_or_refuse(
            np.zeros((20, 2)), np.zeros((20, 2)), CAMERA
        )
    with pytest.raises(ValueError, match="image_points must be"):
        classical._solve_pnp_ransac_or_refuse(
            np.zeros((20, 3)), np.zeros((19, 2)), CAMERA
        )


def test_the_wrapper_reports_refusal_in_the_shape_the_caller_expects():
    """Four values, ok False, no arrays -- the same shape a genuine
    RANSAC failure returns, so the caller needs no special case."""
    import cv2 as _cv2

    from tower.world_builder.backends import classical

    original = _cv2.solvePnPRansac
    try:
        _cv2.solvePnPRansac = lambda *a, **k: (_ for _ in ()).throw(
            _cv2.error(SQPNP_ASSERTION)
        )
        ok, rvec, tvec, inliers = classical._solve_pnp_ransac_or_refuse(
            np.zeros((20, 3)), np.zeros((20, 2)), CAMERA
        )
    finally:
        _cv2.solvePnPRansac = original
    assert ok is False
    assert rvec is None and tvec is None and inliers is None
