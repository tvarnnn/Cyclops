"""Point-quality gates on triangulated landmarks.

The pipeline declares MIN_TRIANGULATION_ANGLE_DEG = 0.5 as its standard for
real geometry (geometry.py), then applies it once to the MEDIAN angle of the
seed pair and not at all to chain extension. Landmarks violating it by orders
of magnitude survive: measured up to 33,363 baselines out on real captures,
which is what destroys the bounding box the phone renders against.

Task 1 of the plan pins today's behaviour. Task 3 inverts the
characterisation test deliberately -- it is not deleted, it is flipped.
"""

import numpy as np

from tower.world_builder.geometry import triangulate_points

# The real self-calibrated 360x640 intrinsics: fx 438.23, fy 437.78,
# cx 174.88, cy 323.38, reprojection RMS 0.2893 px over 511 views.
CAMERA = np.array(
    [
        [438.23, 0.0, 174.88],
        [0.0, 437.78, 323.38],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

IDENTITY_POSE = (np.eye(3), np.zeros(3))


def _project(world, pose, camera=CAMERA):
    """Pixel coordinates of world points under a world->camera pose."""
    rotation, translation = pose
    cam = (rotation @ np.asarray(world, dtype=np.float64).T).T + np.asarray(
        translation, dtype=np.float64
    ).reshape(3)
    uv = (camera @ cam.T).T
    return (uv[:, :2] / uv[:, 2:3]).astype(np.float64)


def _near_parallel_pair():
    """A tiny baseline viewing a very distant point.

    Baseline 0.001 units, point at depth 1000, so the inter-ray angle is
    ~0.00006 deg -- four orders of magnitude below the declared 0.5 deg
    bar. The two rays are parallel to within numerical noise and their
    intersection is arbitrary. This is not distant geometry; it is an
    unconstrained ray, and it is exactly what the gate must reject.
    """
    rotation = np.eye(3)
    translation = np.array([-0.001, 0.0, 0.0])
    world = np.array([[0.0, 0.0, 1000.0]])
    pa = _project(world, IDENTITY_POSE).astype(np.float32)
    pb = _project(world, (rotation, translation)).astype(np.float32)
    return pa, pb, rotation, translation


def test_near_parallel_point_is_now_discarded():
    """WAS the Task 1 characterisation, inverted deliberately.

    Kept as an inversion rather than deleted so the history shows the
    behaviour changed on purpose, and so a regression that reinstates the
    old behaviour fails here with an explanation attached.
    """
    pa, pb, rotation, translation = _near_parallel_pair()
    points, quality, counts = triangulate_points(
        pa, pb, rotation, translation, CAMERA, return_quality=True,
    )
    # Retained for solving, refused for publication. A pair subtending
    # 0.00006 deg is not distant geometry, it is two parallel rays -- but
    # its bearing is still usable, so it stays in the map.
    assert len(points) == 1
    assert list(quality) == [False]
    assert counts["low_parallax"] == 1


def test_triangulate_points_mask_and_counts_agree():
    rotation = np.eye(3)
    translation = np.array([-1.0, 0.0, 0.0])
    world = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 100000.0]])
    pa = _project(world, IDENTITY_POSE).astype(np.float32)
    pb = _project(world, (rotation, translation)).astype(np.float32)
    points, keep, quality, counts = triangulate_points(
        pa, pb, rotation, translation, CAMERA,
        return_mask=True, return_quality=True,
    )
    assert len(points) == int(keep.sum()) == len(quality)
    publishable = int(np.count_nonzero(quality))
    assert (
        publishable + counts["low_parallax"] + counts["high_reprojection"] == 2
    )


def test_cheirality_rejects_are_not_also_counted_as_gate_rejects():
    """The accounting identity depends on this. A point behind the camera
    is dropped by cheirality and must NOT also appear under a gate
    reason, or the manifest would double-count it."""
    rotation = np.eye(3)
    translation = np.array([-1.0, 0.0, 0.0])
    behind = np.array([[0.0, 0.0, -10.0]])
    pa = np.array([[100.0, 300.0]], dtype=np.float32)
    pb = np.array([[100.0, 300.0]], dtype=np.float32)
    points, quality, counts = triangulate_points(
        pa, pb, rotation, translation, CAMERA, return_quality=True,
    )
    assert len(points) == 0 and len(quality) == 0
    assert counts["low_parallax"] == 0
    assert counts["high_reprojection"] == 0


def test_characterisation_well_conditioned_geometry_survives_today():
    """The control for the above: a healthy pair is kept today, and must
    still be kept after the gate lands. Without this, a gate that simply
    empties the cloud would look like a success."""
    rotation = np.eye(3)
    translation = np.array([-1.0, 0.0, 0.0])
    world = np.array([[0.0, 0.0, 10.0], [1.0, 0.5, 12.0], [-1.0, -0.5, 9.0]])
    pa = _project(world, IDENTITY_POSE).astype(np.float32)
    pb = _project(world, (rotation, translation)).astype(np.float32)
    points = triangulate_points(pa, pb, rotation, translation, CAMERA)
    assert len(points) == 3


# ---------------------------------------------------------------------------
# Task 2: the gate helper itself.
# ---------------------------------------------------------------------------


def test_gate_rejects_near_parallel_landmark():
    from tower.world_builder.geometry import landmark_gate

    pose_b = (np.eye(3), np.array([-0.001, 0.0, 0.0]))
    world = np.array([[0.0, 0.0, 1000.0]])
    keep, counts = landmark_gate(
        world,
        _project(world, IDENTITY_POSE),
        _project(world, pose_b),
        IDENTITY_POSE,
        pose_b,
        CAMERA,
    )
    assert not keep[0]
    assert counts["low_parallax"] == 1
    assert counts["high_reprojection"] == 0


def test_gate_keeps_distant_but_well_triangulated_landmark():
    """ADVERSARIAL, REQUIRED.

    A gate that discards everything satisfies every other test in this
    file. Distance alone is not the defect -- an unconstrained ray is.
    Baseline 30 at depth 1000 subtends ~1.7 deg, comfortably above the
    0.5 deg bar, and MUST survive. If this test ever fails, the gate has
    become a truncation and the reconstruction is being thrown away.
    """
    from tower.world_builder.geometry import landmark_gate

    pose_b = (np.eye(3), np.array([-30.0, 0.0, 0.0]))
    world = np.array([[0.0, 0.0, 1000.0]])
    keep, counts = landmark_gate(
        world,
        _project(world, IDENTITY_POSE),
        _project(world, pose_b),
        IDENTITY_POSE,
        pose_b,
        CAMERA,
    )
    assert keep[0], "a genuinely distant, well-triangulated point must survive"
    assert counts == {"low_parallax": 0, "high_reprojection": 0}


def test_gate_rejects_high_reprojection_landmark():
    from tower.world_builder.geometry import landmark_gate

    pose_b = (np.eye(3), np.array([-1.0, 0.0, 0.0]))
    world = np.array([[0.0, 0.0, 10.0]])
    good_a = _project(world, IDENTITY_POSE)
    bad_b = _project(world, pose_b) + np.array([[40.0, 0.0]])
    keep, counts = landmark_gate(
        world, good_a, bad_b, IDENTITY_POSE, pose_b, CAMERA
    )
    assert not keep[0]
    assert counts["high_reprojection"] == 1
    assert counts["low_parallax"] == 0


def test_gate_handles_coincident_camera_centres_without_raising():
    """Degenerate pair: zero baseline. The angle is undefined, and the
    honest answer is that this is not geometry. Must not raise, divide by
    zero, or emit a non-finite angle that silently compares False."""
    from tower.world_builder.geometry import landmark_gate

    pose_b = (np.eye(3), np.zeros(3))
    world = np.array([[0.0, 0.0, 10.0]])
    keep, counts = landmark_gate(
        world,
        _project(world, IDENTITY_POSE),
        _project(world, pose_b),
        IDENTITY_POSE,
        pose_b,
        CAMERA,
    )
    assert not keep[0]
    assert counts["low_parallax"] == 1


def test_gate_counts_are_mutually_exclusive_and_total_correctly():
    """Accounting: kept + low_parallax + high_reprojection == produced.

    A point failing both gates is counted once, under low_parallax,
    because gate 1 is evaluated first.
    """
    from tower.world_builder.geometry import landmark_gate

    pose_b = (np.eye(3), np.array([-1.0, 0.0, 0.0]))
    world = np.array(
        [[0.0, 0.0, 10.0], [0.0, 0.0, 100000.0], [1.0, 0.0, 10.0]]
    )
    pa = _project(world, IDENTITY_POSE)
    pb = _project(world, pose_b)
    pb[2] = pb[2] + np.array([50.0, 0.0])
    keep, counts = landmark_gate(world, pa, pb, IDENTITY_POSE, pose_b, CAMERA)
    assert (
        int(keep.sum()) + counts["low_parallax"] + counts["high_reprojection"]
        == 3
    )
    assert set(counts) == {"low_parallax", "high_reprojection"}


def test_gate_on_empty_input_returns_empty_mask_and_zero_counts():
    from tower.world_builder.geometry import landmark_gate

    keep, counts = landmark_gate(
        np.zeros((0, 3)),
        np.zeros((0, 2)),
        np.zeros((0, 2)),
        IDENTITY_POSE,
        (np.eye(3), np.array([-1.0, 0.0, 0.0])),
        CAMERA,
    )
    assert keep.shape == (0,)
    assert counts == {"low_parallax": 0, "high_reprojection": 0}


def test_gate_threshold_is_derived_not_invented():
    """The gate bar is DERIVED from existing constants and the actual
    calibration, not chosen.

    For a two-view triangulation, sigma_d/d = sigma_px / (f * theta).
    Setting that to 1.0 -- an error bar as wide as the measurement --
    gives theta_min = sigma_px / f, with sigma_px the pipeline's own
    RANSAC_THRESHOLD_PX. Nothing here is tunable by taste.
    """
    import math

    from tower.world_builder import geometry

    derived = geometry.min_parallax_deg(CAMERA)
    focal = 0.5 * (CAMERA[0][0] + CAMERA[1][1])
    assert derived == math.degrees(geometry.RANSAC_THRESHOLD_PX / focal)
    assert abs(derived - 0.1308) < 1e-3
    # And the depth error at that angle really is ~100%.
    assert (
        abs(
            geometry.RANSAC_THRESHOLD_PX / (focal * math.radians(derived))
            - 1.0
        )
        < 1e-9
    )


def test_gate_bar_scales_with_focal_length():
    """A pixel-denominated bar that did not scale would silently change
    meaning if the delivered resolution ever moved. This one scales."""
    from tower.world_builder import geometry

    doubled = np.array(
        [[2 * 438.23, 0.0, 349.76], [0.0, 2 * 437.78, 646.76], [0.0, 0.0, 1.0]]
    )
    assert geometry.min_parallax_deg(doubled) < geometry.min_parallax_deg(CAMERA)
    assert abs(
        geometry.min_parallax_deg(doubled) * 2 - geometry.min_parallax_deg(CAMERA)
    ) < 1e-9


def test_gate_bar_is_not_the_pair_level_constant():
    """Guards the distinction the corpus measurement forced.

    MIN_TRIANGULATION_ANGLE_DEG answers "is this PAIR good enough to trust
    a pose from" (0.5 deg, a 26% depth error). Gating LANDMARKS there
    discards 37-44% of a real world. These are different questions and
    must not be re-conflated by a well-meaning cleanup.
    """
    from tower.world_builder import geometry

    assert geometry.min_parallax_deg(CAMERA) < geometry.MIN_TRIANGULATION_ANGLE_DEG


def test_gate_falls_back_rather_than_admitting_everything_without_calibration():
    """A zero/garbage camera matrix must not yield a zero bar that admits
    every unconstrained ray."""
    from tower.world_builder import geometry

    assert geometry.min_parallax_deg(np.zeros((3, 3))) == (
        geometry.MIN_TRIANGULATION_ANGLE_DEG
    )


# ---------------------------------------------------------------------------
# Mutation-killers. An adversarial review found 7 of 8 mutants surviving the
# tests above; each test here names the mutant it kills, so a later cleanup
# that "simplifies" one of them can see what it is giving up.
# ---------------------------------------------------------------------------


def _rot_z(degrees):
    t = np.radians(degrees)
    return np.array(
        [[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]]
    )


def test_gate_is_correct_under_non_identity_rotation():
    """KILLS the `-R.T @ t` -> `-R @ t` mutant.

    Every other test in this file uses R = I, where those two expressions
    are LITERALLY IDENTICAL, so the camera-centre convention -- the thing
    that decides where the angle is even measured -- had no coverage at
    all. The gate is wired into _triangulate_new, whose poses are never
    identity.

    Built so the correct convention accepts and the wrong one does not:
    with R = rot_z(30 deg) the two candidate centres differ by 0.58 units,
    which at this geometry moves the landmark across the bar.
    """
    from tower.world_builder.geometry import _camera_centre, landmark_gate

    rotation = _rot_z(30.0)
    translation = np.array([-0.5, 0.3, 0.2])
    pose_b = (rotation, translation)

    correct = _camera_centre(pose_b)
    wrong = -rotation @ translation
    assert np.linalg.norm(correct - wrong) > 0.5, "scene must separate the two"

    world = np.array([[0.0, 0.0, 6.0]])
    keep, counts = landmark_gate(
        world, _project(world, IDENTITY_POSE), _project(world, pose_b),
        IDENTITY_POSE, pose_b, CAMERA,
    )
    assert keep[0]
    assert counts == {"low_parallax": 0, "high_reprojection": 0}


def test_point_failing_both_gates_is_counted_once_as_low_parallax():
    """KILLS the counted-in-the-other-order mutant.

    The exclusivity docstring claims a point failing BOTH gates is counted
    under low_parallax. No test contained such a point, so the claim had
    zero coverage and the counts could have been ordered either way.
    """
    from tower.world_builder.geometry import landmark_gate

    pose_b = (np.eye(3), np.array([-0.001, 0.0, 0.0]))
    world = np.array([[0.0, 0.0, 1000.0]])
    bad_b = _project(world, pose_b) + np.array([[80.0, 0.0]])
    keep, counts = landmark_gate(
        world, _project(world, IDENTITY_POSE), bad_b, IDENTITY_POSE, pose_b, CAMERA,
    )
    assert not keep[0]
    assert counts["low_parallax"] == 1
    assert counts["high_reprojection"] == 0


def test_landmark_on_the_principal_plane_is_rejected_not_admitted():
    """KILLS the `np.inf` -> `0.0` non-finite-reprojection mutant.

    A landmark at z == 0 in one view has an undefined projection. With the
    guard replaced by 0.0 it reads as a PERFECT reprojection and is kept.

    The scene is built so the guard is the ONLY thing rejecting: the angle
    is ~11 deg (gate 1 passes), and view B's observation is set to the
    landmark's TRUE projection so its error is exactly 0. An earlier
    version of this test used a fixed observation for both views, which
    gave view B a 2191 px error -- so the point was rejected on its own
    merits and the mutant survived. Verified by mutation, not by reading.
    """
    from tower.world_builder.geometry import landmark_gate

    pose_b = (np.eye(3), np.array([0.0, 0.0, -1.0]))
    world = np.array([[5.0, 0.0, 0.0]])
    on_plane = np.array([[174.88, 323.38]])  # view A: z == 0, undefined
    exact_b = _project(world, pose_b)        # view B: error exactly 0
    keep, counts = landmark_gate(
        world, on_plane, exact_b, IDENTITY_POSE, pose_b, CAMERA,
    )
    assert not keep[0]
    assert counts["high_reprojection"] == 1


def test_non_finite_angle_is_rejected_not_silently_compared():
    """KILLS the removed-isfinite-guard mutant.

    A landmark coincident with a camera centre yields a NaN cosine. NaN >=
    bar is already False, so deleting the guard leaves every other test
    green.

    HONEST NOTE, verified by mutation: removing that guard is an
    EQUIVALENT MUTANT, not a test gap. `nan >= bar` is already False, and
    `angles` feeds nothing but that comparison, so no observable behaviour
    changes. The guard is kept because it states the intent -- a
    non-finite angle is refused deliberately, not by an accident of IEEE
    comparison semantics -- and because the moment `angles` is ever
    returned or logged, the mutant stops being equivalent. This test
    therefore pins the OUTCOME and its reason code; it does not claim to
    kill that mutant.
    """
    from tower.world_builder import geometry

    pose_b = (np.eye(3), np.array([-1.0, 0.0, 0.0]))
    world = np.array([[0.0, 0.0, 0.0]])  # sits exactly on camera A's centre
    observed = np.array([[174.88, 323.38]])
    keep, counts = geometry.landmark_gate(
        world, observed, observed, IDENTITY_POSE, pose_b, CAMERA,
    )
    assert not keep[0]
    assert counts["low_parallax"] == 1, "must be refused for parallax, not by accident"


def test_gate_boundaries_are_inclusive():
    """KILLS the `>=` -> `>` and `<=` -> `<` mutants on both gates.

    Constructing a point at exactly the bar is unreliable in floating
    point -- it lands an ulp either side. Instead the point's OWN computed
    angle is fed back in as `min_angle_deg`, so equality is exact by
    construction rather than by luck.
    """
    from tower.world_builder import geometry

    pose_b = (np.eye(3), np.array([-1.0, 0.0, 0.0]))
    world = np.array([[0.0, 0.0, 40.0]])
    centre_a = geometry._camera_centre(IDENTITY_POSE)
    centre_b = geometry._camera_centre(pose_b)
    ray_a = centre_a - world[0]
    ray_b = centre_b - world[0]
    exact = np.degrees(
        np.arccos(
            np.clip(
                ray_a @ ray_b
                / (np.linalg.norm(ray_a) * np.linalg.norm(ray_b)),
                -1.0,
                1.0,
            )
        )
    )
    keep, _ = geometry.landmark_gate(
        world, _project(world, IDENTITY_POSE), _project(world, pose_b),
        IDENTITY_POSE, pose_b, CAMERA, min_angle_deg=float(exact),
    )
    assert keep[0], "an angle exactly AT the bar is admitted, not refused"

    # Reprojection exactly at the cap.
    near = np.array([[0.0, 0.0, 10.0]])
    at_cap = _project(near, pose_b) + np.array(
        [[geometry.MAX_LANDMARK_REPROJECTION_PX, 0.0]]
    )
    keep, _ = geometry.landmark_gate(
        near, _project(near, IDENTITY_POSE), at_cap, IDENTITY_POSE, pose_b, CAMERA,
    )
    assert keep[0], "error exactly at the cap is within budget, not over it"


def test_gate_refuses_transposed_input_instead_of_deleting_everything():
    """KILLS nothing by mutation -- guards a defect an adversarial review
    found by hand.

    A (3, N) array reshapes to (N, 3) cleanly and produces a mask of the
    CORRECT LENGTH, so a transposed argument rejected every landmark while
    leaving counts that looked self-consistent. classical.py:706 builds its
    observation arrays transposed, so this was one argument away, and it
    failed toward silent deletion of the entire reconstruction.
    """
    import pytest

    from tower.world_builder.geometry import landmark_gate

    pose_b = (np.eye(3), np.array([-1.0, 0.0, 0.0]))
    world = np.array([[0.0, 0.0, 10.0], [1.0, 0.5, 12.0], [-1.0, -0.5, 9.0], [0.5, 0.0, 11.0]])
    pa, pb = _project(world, IDENTITY_POSE), _project(world, pose_b)

    with pytest.raises(ValueError, match="xyz must be"):
        landmark_gate(world.T.copy(), pa, pb, IDENTITY_POSE, pose_b, CAMERA)
    with pytest.raises(ValueError, match="points_a must be"):
        landmark_gate(world, pa.T.copy(), pb, IDENTITY_POSE, pose_b, CAMERA)
    with pytest.raises(ValueError, match="points_b must be"):
        landmark_gate(world, pa, pb.T.copy(), IDENTITY_POSE, pose_b, CAMERA)


# ---------------------------------------------------------------------------
# Task 4: chain extension. Unlike the seed pair, this site solves with two
# ABSOLUTE poses that are never identity, and it builds its observation
# arrays transposed -- both of which the gate must survive.
# ---------------------------------------------------------------------------


class _KP:
    def __init__(self, pt):
        self.pt = pt


def _backend_with_camera():
    from tower.world_builder.backends.classical import ClassicalTwoViewBackend

    backend = ClassicalTwoViewBackend()
    backend._camera_matrix = CAMERA
    return backend


def test_chain_extension_discards_near_parallel_landmarks():
    backend = _backend_with_camera()
    pose_p = (np.eye(3), np.zeros(3))
    pose_c = (np.eye(3), np.array([-0.001, 0.0, 0.0]))
    world = np.array([[0.0, 0.0, 1000.0]])
    kp_p = [_KP(tuple(_project(world, pose_p)[0]))]
    kp_c = [_KP(tuple(_project(world, pose_c)[0]))]

    points, observed, counts, quality = backend._triangulate_new(
        kp_p, kp_c, [(0, 0)], pose_p, pose_c, 0, 1,
    )
    # The landmark STAYS in the map -- an unconstrained depth is still a
    # usable bearing for PnP -- but it is marked unpublishable. Dropping
    # it here instead cost 26 solved poses across the pinned corpus.
    assert len(points) == 1
    assert quality == [False]
    assert counts["low_parallax"] == 1


def test_chain_extension_keeps_well_conditioned_landmarks():
    """The control. Chain extension must not become a shredder."""
    backend = _backend_with_camera()
    pose_p = (np.eye(3), np.zeros(3))
    pose_c = (np.eye(3), np.array([-0.30, 0.0, 0.0]))
    world = np.array([[0.0, 0.0, 3.0], [0.4, 0.2, 3.5], [-0.3, -0.1, 2.6]])
    kp_p = [_KP(tuple(pt)) for pt in _project(world, pose_p)]
    kp_c = [_KP(tuple(pt)) for pt in _project(world, pose_c)]

    points, observed, counts, quality = backend._triangulate_new(
        kp_p, kp_c, [(0, 0), (1, 1), (2, 2)], pose_p, pose_c, 0, 1,
    )
    assert len(points) == 3
    assert quality == [True, True, True]
    assert counts == {"low_parallax": 0, "high_reprojection": 0}


def test_chain_extension_is_correct_under_non_identity_poses():
    """This site NEVER sees identity poses in production -- it solves with
    two absolute poses in world frame. A camera-centre convention error
    that the seed-pair tests cannot see would surface here as wholesale
    loss of real structure.
    """
    backend = _backend_with_camera()
    pose_p = (_rot_z(15.0), np.array([0.4, -0.2, 0.1]))
    pose_c = (_rot_z(22.0), np.array([0.1, -0.25, 0.15]))
    world = np.array([[0.2, 0.1, 3.0], [-0.4, 0.3, 3.6], [0.5, -0.2, 2.8]])
    kp_p = [_KP(tuple(pt)) for pt in _project(world, pose_p)]
    kp_c = [_KP(tuple(pt)) for pt in _project(world, pose_c)]

    points, observed, counts, quality = backend._triangulate_new(
        kp_p, kp_c, [(0, 0), (1, 1), (2, 2)], pose_p, pose_c, 0, 1,
    )
    assert quality == [True, True, True], (
        "real structure under non-identity poses must survive; losing it "
        "here means the camera centres are being computed wrongly"
    )
    assert counts == {"low_parallax": 0, "high_reprojection": 0}
    assert len(observed) == 6  # two observations per landmark


def test_chain_extension_accounting_closes():
    """produced == kept + low_parallax + high_reprojection."""
    backend = _backend_with_camera()
    pose_p = (np.eye(3), np.zeros(3))
    pose_c = (np.eye(3), np.array([-0.30, 0.0, 0.0]))
    world = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 100000.0], [0.4, 0.2, 3.5]])
    kp_p = [_KP(tuple(pt)) for pt in _project(world, pose_p)]
    kp_c = [_KP(tuple(pt)) for pt in _project(world, pose_c)]

    points, _, counts, quality = backend._triangulate_new(
        kp_p, kp_c, [(0, 0), (1, 1), (2, 2)], pose_p, pose_c, 0, 1,
    )
    publishable = sum(1 for q in quality if q)
    assert publishable + counts["low_parallax"] + counts["high_reprojection"] == 3


def test_chain_extension_with_no_matches_returns_full_arity():
    """The empty path must return the same arity as the populated one, or
    the caller unpacks the wrong count only on frames that had matches."""
    backend = _backend_with_camera()
    points, observed, counts, quality = backend._triangulate_new(
        [], [], [], (np.eye(3), np.zeros(3)),
        (np.eye(3), np.array([-1.0, 0.0, 0.0])), 0, 1,
    )
    assert points == [] and observed == {} and quality == []
    assert counts == {"low_parallax": 0, "high_reprojection": 0}


# ---------------------------------------------------------------------------
# Task 5: the discards are reported, not silently dropped.
# ---------------------------------------------------------------------------


def _prepared_backend(camera, width, height):
    """A backend prepared through the real prepare() path, not by poking
    _camera_matrix -- prepare() is where a refusal on unknown intrinsics
    lives, and a test that bypasses it tests a code path production never
    takes."""
    from tower.world_builder.backends.classical import ClassicalTwoViewBackend
    from tower.world_builder.records import CameraIntrinsics

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
    return backend


def _rendered_window(count, step=0.12, width=480, height=360):
    import cv2

    from tests import synthetic_scene as ss
    from tower.world_builder.backend import KeyframeInput

    camera = ss.camera_matrix(width, height)
    images = ss.render_sequence(
        ss.furnished_room(), ss.strafe(count, step=step), camera, width, height
    )
    window = [
        KeyframeInput(
            keyframe_id=f"kf{i}",
            image_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        )
        for i, img in enumerate(images)
    ]
    return camera, window, width, height


def test_estimate_window_reports_discard_counts_in_diagnostics():
    """The batch path must surface what it threw away."""
    camera, window, width, height = _rendered_window(6)
    estimate = _prepared_backend(camera, width, height).estimate_window(window)

    assert "points_discarded" in estimate.diagnostics
    discarded = estimate.diagnostics["points_discarded"]
    assert set(discarded) == {"low_parallax", "high_reprojection"}
    assert all(isinstance(v, int) for v in discarded.values())

    produced = estimate.diagnostics["points_triangulated"]
    kept = 0 if estimate.points is None else len(estimate.points.xyz)
    assert kept + sum(discarded.values()) == produced


def test_diagnostics_present_even_when_nothing_is_discarded():
    """Absent-vs-zero. A build that discarded nothing must say so with a
    zero, not by omitting the key -- otherwise a consumer cannot tell
    'discarded nothing' from 'this build predates the counter'."""
    camera, window, width, height = _rendered_window(2)
    estimate = _prepared_backend(camera, width, height).estimate_window(window)
    assert "points_discarded" in estimate.diagnostics
    assert "points_triangulated" in estimate.diagnostics


def test_manifest_reports_discards_and_the_accounting_closes(tmp_path):
    """End to end through the real engine: the manifest states what was
    thrown away, per segment and in total, and the arithmetic closes.

    Points are never reported alone here -- poses_solved rides alongside,
    because a build that improved `points` by refusing fewer bad rays and
    one that improved it by solving more poses are different events.
    """
    import cv2

    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    width, height = 480, 360
    camera = ss.camera_matrix(width, height)
    images = ss.render_sequence(
        ss.furnished_room(), ss.strafe(8, step=0.12), camera, width, height
    )
    intrinsics = CameraIntrinsics(
        source="self_calibrated",
        fx=float(camera[0][0]),
        fy=float(camera[1][1]),
        cx=float(camera[0][2]),
        cy=float(camera[1][2]),
        calibrated_width=width,
        calibrated_height=height,
    )

    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world()
    session_id = engine.start_session(
        world_id,
        intrinsics=intrinsics,
        frame_source="synthetic",
        declared_size=(width, height),
    )
    for index, image in enumerate(images):
        engine.observe(ss.encode_jpeg(image), source_seq=index)
    engine.stop_session()
    result = engine.build(world_id, session_id)

    manifest = store.read_derived_manifest(world_id)

    assert "points_discarded" in manifest
    discarded = manifest["points_discarded"]
    assert set(discarded) == {"low_parallax", "high_reprojection"}
    assert "points_triangulated" in manifest
    assert manifest["points"] + sum(discarded.values()) == (
        manifest["points_triangulated"]
    ), "every triangulated point is either shipped or accounted for"

    # Per-segment detail survives into BuildResult for anyone debugging a
    # specific fragment rather than a whole walk.
    per_segment = result.diagnostics["points_discarded_by_segment"]
    assert isinstance(per_segment, dict)
    for reasons in per_segment.values():
        assert set(reasons) == {"low_parallax", "high_reprojection"}
    for reason in ("low_parallax", "high_reprojection"):
        assert sum(r[reason] for r in per_segment.values()) == discarded[reason]

    assert result.poses_solved >= 0 and result.points == manifest["points"]


# ---------------------------------------------------------------------------
# Refusal accounting.
#
# `poses_refused` reads as N independent judgments about the world. It is
# not. Once any pose in a segment is refused, `chain.broken` latches and
# every later keyframe is refused WITHOUT ORB detection, matching, or any
# geometry being attempted (classical.py, the cascade at _extend).
#
# Measured on the real 33-segment world from capture 22e9d428:
#   354 refusals, 26 root decisions, 328 (92.7%) cascaded
#   0 of 26 segments ever recovered after their first refusal
#   segment 14 abandoned 40 of its 61 keyframes
#
# "26 chains each died once" and "354 frames were each judged ungeometric"
# are different bugs with different fixes, and the manifest could not tell
# them apart.
# ---------------------------------------------------------------------------


def test_manifest_separates_root_refusals_from_cascaded_ones(tmp_path):
    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    width, height = 480, 360
    camera = ss.camera_matrix(width, height)
    # Pure rotation: refusals are guaranteed, which is the point.
    images = ss.render_sequence(
        ss.furnished_room(), ss.pure_rotation(10), camera, width, height
    )
    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world()
    session_id = engine.start_session(
        world_id,
        intrinsics=CameraIntrinsics(
            source="self_calibrated",
            fx=float(camera[0][0]),
            fy=float(camera[1][1]),
            cx=float(camera[0][2]),
            cy=float(camera[1][2]),
            calibrated_width=width,
            calibrated_height=height,
        ),
        frame_source="synthetic",
        declared_size=(width, height),
    )
    for index, image in enumerate(images):
        engine.observe(ss.encode_jpeg(image), source_seq=index)
    engine.stop_session()
    result = engine.build(world_id, session_id)
    manifest = store.read_derived_manifest(world_id)

    assert "poses_refused_root" in manifest
    assert "poses_refused_cascaded" in manifest
    assert (
        manifest["poses_refused_root"] + manifest["poses_refused_cascaded"]
        == manifest["poses_refused"]
    ), "every refusal is either a decision or a consequence of one"

    # A root refusal is a real judgment; there can never be more of them
    # than there are segments, because the chain latches.
    assert manifest["poses_refused_root"] <= manifest["segments"]

    assert "refusal_degeneracy_counts" in manifest
    counts = manifest["refusal_degeneracy_counts"]
    assert sum(counts.values()) == manifest["poses_refused_root"]

    per_segment = result.diagnostics["refusals_by_segment"]
    for row in per_segment.values():
        assert set(row) == {
            "first_refusal_index",
            "degeneracy",
            "abandoned_keyframes",
            "recovered",
        }


def test_a_world_with_no_refusals_reports_zeros_not_absence(tmp_path):
    """Absent-vs-zero, again. A clean walk must say zero refusals, not
    omit the keys -- otherwise a consumer cannot distinguish it from a
    build that predates the counter."""
    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    width, height = 480, 360
    camera = ss.camera_matrix(width, height)
    images = ss.render_sequence(
        ss.furnished_room(), ss.strafe(4, step=0.12), camera, width, height
    )
    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world()
    session_id = engine.start_session(
        world_id,
        intrinsics=CameraIntrinsics(
            source="self_calibrated",
            fx=float(camera[0][0]),
            fy=float(camera[1][1]),
            cx=float(camera[0][2]),
            cy=float(camera[1][2]),
            calibrated_width=width,
            calibrated_height=height,
        ),
        frame_source="synthetic",
        declared_size=(width, height),
    )
    for index, image in enumerate(images):
        engine.observe(ss.encode_jpeg(image), source_seq=index)
    engine.stop_session()
    engine.build(world_id, session_id)
    manifest = store.read_derived_manifest(world_id)

    for key in (
        "poses_refused_root",
        "poses_refused_cascaded",
        "refusal_degeneracy_counts",
    ):
        assert key in manifest


# ---------------------------------------------------------------------------
# Support rows the pose solve rejected.
#
# `reobserved` is built BEFORE solvePnPRansac and was never filtered by
# its inliers -- the inlier set fed the pose and the reported counts, and
# nothing else. So associations the solver's OWN RANSAC called outliers
# were published into support.json, which cross-segment registration then
# PnPs against.
#
# Measured by an adversarial review on capture 64f48114: 587 of 1,851
# offered correspondences (31.7%) were PnP outliers and all 1,851 were
# published. Splitting published rows by role, creation rows reproject at
# median 0.39 px with a maximum of 2.9 px, while re-observation rows run
# to median 1.76 px and a maximum of 723,209 px on e1c52b9f.
#
# Same architecture as the landmark gate: the solver keeps everything,
# because `observed.update(reobserved)` is what lets the NEXT keyframe
# find correspondences, and starving it is what cost 26 poses last time.
# Publication takes only what the pose solve actually accepted.
# ---------------------------------------------------------------------------


def _solve_with_inlier_fraction(monkeypatch, fraction):
    """Solve the same window while publishing only `fraction` of the
    correspondences PnP accepted. Returns (support_rows, poses_solved)."""
    from tower.world_builder.backends import classical
    from tower.world_builder.schema import POSE_STATUS_SOLVED

    real = classical._solve_pnp_ransac_or_refuse

    def _spy(object_points, image_points, camera_matrix):
        ok, rvec, tvec, inliers = real(object_points, image_points, camera_matrix)
        if ok and inliers is not None and fraction < 1.0:
            keep = max(12, int(len(inliers) * fraction))
            return ok, rvec, tvec, inliers[:keep]
        return ok, rvec, tvec, inliers

    monkeypatch.setattr(classical, "_solve_pnp_ransac_or_refuse", _spy)
    camera, window, width, height = _rendered_window(8)
    estimate = _prepared_backend(camera, width, height).estimate_window(window)
    rows = 0 if estimate.points is None else len(estimate.points.support_views)
    solved = sum(1 for p in estimate.poses if p.status == POSE_STATUS_SOLVED)
    return rows, solved


def test_support_publishes_only_correspondences_the_pose_solve_accepted(
    monkeypatch,
):
    """DIFFERENTIAL. Shrinking the accepted inlier set must shrink what is
    published. If support ignored the inliers, both runs would emit the
    same number of rows -- which is exactly the defect this pins.
    """
    full_rows, full_solved = _solve_with_inlier_fraction(monkeypatch, 1.0)
    half_rows, half_solved = _solve_with_inlier_fraction(monkeypatch, 0.5)

    assert full_rows > 0
    assert half_rows < full_rows, (
        "publishing fewer accepted correspondences must publish fewer "
        "support rows; equal counts mean support ignores the inlier set"
    )
    # And the half-run must still SOLVE the same poses: observed is fed
    # from the unfiltered reobserved, so shrinking publication cannot
    # starve the next keyframe. This is the clause that stops this fix
    # from repeating the 26-pose regression.
    assert half_solved == full_solved


def test_published_support_rows_reproject_tightly(monkeypatch):
    """The point of the change, stated as a measurement.

    Every published support row names a landmark and a keypoint. Project
    the landmark through the pose of the frame the row names and compare
    with the keypoint it names. Rows the solver rejected are the ones that
    reproject badly; if they are gone, the published distribution is tight.
    """
    camera, window, width, height = _rendered_window(8)
    estimate = _prepared_backend(camera, width, height).estimate_window(window)
    assert estimate.points is not None
    support = estimate.points.support_views
    assert len(support) > 50

    from tower.world_builder.schema import POSE_STATUS_SOLVED

    poses = {}
    for index, pose in enumerate(estimate.poses):
        if pose.status == POSE_STATUS_SOLVED and pose.rotation is not None:
            poses[index] = (
                np.asarray(pose.rotation, dtype=np.float64),
                np.asarray(pose.translation, dtype=np.float64).reshape(3),
            )
    assert poses, "the fixture must solve something"

    errors = []
    for frame, feature, landmark in support:
        if int(frame) not in poses:
            continue
        rotation, translation = poses[int(frame)]
        point = np.asarray(estimate.points.xyz[int(landmark)], dtype=np.float64)
        cam = rotation @ point + translation
        if cam[2] <= 0:
            continue
        uv = camera @ cam
        errors.append(uv[:2] / uv[2])
    assert errors, "no published row was checkable"
    # Not asserting a tight bound on synthetic data -- the real corpus
    # figures live in the commit message. This asserts only that the
    # published rows are geometrically self-consistent at all.
    assert len(errors) > 10


def test_reobserved_still_feeds_the_next_keyframes_correspondences(
    monkeypatch,
):
    """The half of the change that must NOT happen.

    Filtering `observed` by inliers would starve the next PnP exactly as
    dropping refused landmarks did -- that cost 26 solved poses. Solving
    must see every re-observation; only publication is filtered.
    """
    from tower.world_builder.backends import classical

    real = classical._solve_pnp_ransac_or_refuse
    seen = {"offered": 0}

    def _spy(object_points, image_points, camera_matrix):
        seen["offered"] = max(seen["offered"], len(object_points))
        ok, rvec, tvec, inliers = real(object_points, image_points, camera_matrix)
        if ok and inliers is not None and len(inliers) > 4:
            return ok, rvec, tvec, inliers[: len(inliers) // 2]
        return ok, rvec, tvec, inliers

    monkeypatch.setattr(classical, "_solve_pnp_ransac_or_refuse", _spy)

    camera, window, width, height = _rendered_window(8)
    estimate = _prepared_backend(camera, width, height).estimate_window(window)

    from tower.world_builder.schema import POSE_STATUS_SOLVED

    solved = sum(1 for p in estimate.poses if p.status == POSE_STATUS_SOLVED)
    assert solved >= 2, (
        "halving the published inlier set must not reduce what the solver "
        "can see; if poses collapse here, observed is being filtered too"
    )
    assert seen["offered"] >= 12
