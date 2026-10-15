"""Which 2D feature in which keyframe produced which 3D landmark.

SYNTHETIC, NOT PHYSICAL. The room is rendered by a perfect pinhole; no
number here says anything about the Ray-Ban camera.

The association exists at solve time -- `classical.py`'s `observed` dict
-- and used to die with the stack frame. Cross-segment registration
needs it, and re-deriving it costs a full re-solve, so it is persisted.

Two things this file is really checking.

1. THE ASSOCIATION IS TRUE, not merely present. Every row is verified by
   reprojecting the landmark it names into the frame it names and
   comparing against the feature it names. A table of plausible integers
   that points at the wrong features would pass a shape assertion and
   fabricate a registration later.

2. THE LIVE PATH RECORDS THE WHOLE HISTORY. `_Chain.forget_before` prunes
   `observed` down to one frame deliberately, so a table read off the
   chain at the end would hold one frame's worth on the live path and
   everything on the rebuild path -- different data under one field name.
   `test_a_live_solve_records_the_same_association_as_a_cold_solve` is
   the trap, stated as a test.
"""

import json

import numpy as np
import pytest

from tests import synthetic_scene as ss
from tower.world_builder.backend import KeyframeInput
from tower.world_builder.backends.classical import (
    EXTEND_REFERENCE_DEPTH,
    ClassicalTwoViewBackend,
)
from tower.world_builder.engine import WorldBuilderEngine
from tower.world_builder.geometry import detect_and_describe
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.schema import (
    INTRINSICS_SOURCE_SELF_CALIBRATED,
    POSE_STATUS_SOLVED,
)
from tower.world_builder.store import WorldStore

WIDTH, HEIGHT = 480, 360

# Generous on purpose. The claim under test is "this row names THAT
# feature", not "triangulation is accurate to a tenth of a pixel". The
# measured distribution on this walk is a median of 0.62 px with a tail
# out to 422 px -- forward-only SfM with no bundle adjustment, so late
# landmarks carry the chain's drift. A MIS-association sits in the
# hundreds of pixels as a matter of course, which is the gap these
# numbers live in.
REPROJECTION_TOLERANCE_PX = 3.0
REPROJECTION_TAIL_PX = 10.0


@pytest.fixture(scope="module")
def scene():
    return ss.furnished_room()


@pytest.fixture(scope="module")
def camera_matrix():
    return ss.camera_matrix(WIDTH, HEIGHT)


@pytest.fixture(scope="module")
def intrinsics(camera_matrix):
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


def _grays(scene, camera_matrix, poses):
    import cv2

    return [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for image in ss.render_sequence(scene, poses, camera_matrix, WIDTH, HEIGHT)
    ]


@pytest.fixture(scope="module")
def window(scene, camera_matrix):
    grays = _grays(scene, camera_matrix, ss.strafe(10, step=0.09))
    return [
        KeyframeInput(keyframe_id=f"kf{index:04d}", image_gray=gray)
        for index, gray in enumerate(grays)
    ]


@pytest.fixture(scope="module")
def cold(window, intrinsics):
    backend = ClassicalTwoViewBackend()
    backend.prepare(intrinsics)
    return backend.estimate_window(window)


def _solved_window(estimate):
    """How many leading frames actually joined the chain."""
    count = 0
    for pose in estimate.poses:
        if pose.status in (POSE_STATUS_SOLVED, "anchor"):
            count += 1
        else:
            break
    return count


# -- the association itself --------------------------------------------


def test_the_window_path_records_the_association(cold):
    support = cold.points.support_views

    assert support is not None, (
        "PointBlock.support_views was declared for exactly this and is the "
        "only record of which feature made which landmark"
    )
    assert support.ndim == 2 and support.shape[1] == 3, (
        "flat (M, 3) table of [frame_index, feature_index, landmark_index]"
    )
    assert np.issubdtype(support.dtype, np.integer)
    assert len(support) > len(cold.points), (
        "a landmark is seen by at least two frames or it could not have "
        "been triangulated"
    )


def test_every_landmark_is_named_by_at_least_two_frames(cold):
    """Triangulation needs two views. A one-view row is a bug, not data."""
    support = cold.points.support_views
    views = np.bincount(support[:, 2], minlength=len(cold.points))

    assert views.min() >= 2
    assert set(np.unique(support[:, 2])) == set(range(len(cold.points))), (
        "every landmark in the block is accounted for"
    )


def test_the_frame_index_is_the_position_within_the_window(cold, window):
    """The convention, pinned: window-relative, 0 == the segment anchor.

    Not session-relative. The backend is handed one window and has no
    idea where it sits in a session; the engine tags the rows with the
    segment on the way to disk.
    """
    support = cold.points.support_views
    frames = np.unique(support[:, 0])

    assert frames.min() == 0, "the anchor keyframe observes structure too"
    assert frames.max() < len(window)
    assert frames.max() == _solved_window(cold) - 1, (
        "the last frame that joined the chain is the last one that can "
        "have observed anything"
    )


def test_each_row_reprojects_onto_the_feature_it_names(cold, window):
    """The association is TRUE, not just well-shaped.

    Project the landmark a row names through the pose of the frame it
    names and land on the keypoint it names. This is what distinguishes
    a real table from a plausible one.
    """
    camera = ss.camera_matrix(WIDTH, HEIGHT)
    keypoints = [detect_and_describe(frame.image_gray)[0] for frame in window]
    xyz = cold.points.xyz.astype(np.float64)
    support = cold.points.support_views

    errors = []
    for frame_index, feature_index, landmark_index in support:
        pose = cold.poses[frame_index]
        rotation = (
            np.eye(3) if pose.rotation is None else np.asarray(pose.rotation)
        )
        translation = (
            np.zeros(3)
            if pose.translation is None
            else np.asarray(pose.translation)
        )
        in_camera = rotation @ xyz[landmark_index] + translation
        projected = camera @ in_camera
        projected = projected[:2] / projected[2]
        observed = np.asarray(keypoints[frame_index][feature_index].pt)
        errors.append(float(np.linalg.norm(projected - observed)))

    errors = np.asarray(errors)
    assert np.median(errors) < REPROJECTION_TOLERANCE_PX, (
        f"median reprojection {np.median(errors):.2f} px -- the rows do not "
        "name the features that produced these landmarks"
    )
    # The tail, bounded rather than ignored. A SHUFFLED table fails this
    # outright, which is what the next test shows.
    assert (errors < REPROJECTION_TAIL_PX).mean() > 0.95


def test_a_shuffled_association_would_fail_that_check(cold, window):
    """The previous test's teeth, shown rather than asserted."""
    camera = ss.camera_matrix(WIDTH, HEIGHT)
    keypoints = [detect_and_describe(frame.image_gray)[0] for frame in window]
    xyz = cold.points.xyz.astype(np.float64)
    support = cold.points.support_views.copy()
    rng = np.random.default_rng(11)
    support[:, 2] = rng.permutation(support[:, 2])

    errors = []
    for frame_index, feature_index, landmark_index in support:
        pose = cold.poses[frame_index]
        rotation = (
            np.eye(3) if pose.rotation is None else np.asarray(pose.rotation)
        )
        translation = (
            np.zeros(3)
            if pose.translation is None
            else np.asarray(pose.translation)
        )
        in_camera = rotation @ xyz[landmark_index] + translation
        projected = camera @ in_camera
        projected = projected[:2] / projected[2]
        errors.append(
            float(
                np.linalg.norm(
                    projected
                    - np.asarray(keypoints[frame_index][feature_index].pt)
                )
            )
        )

    errors = np.asarray(errors)
    assert np.median(errors) > 10 * REPROJECTION_TAIL_PX
    assert (errors < REPROJECTION_TAIL_PX).mean() < 0.2


# -- the pruning trap ---------------------------------------------------


def test_a_live_solve_records_the_same_association_as_a_cold_solve(
    window, intrinsics, cold
):
    """`_Chain.forget_before` prunes; the table must not be read off it.

    Bit for bit, and no tolerance, for the same reason
    test_world_builder_incremental.py has none: a difference here is
    state that was not accumulated, not a rounding question.
    """
    backend = ClassicalTwoViewBackend()
    backend.begin(intrinsics)
    for frame in window:
        backend.extend(frame)
    live = backend.snapshot()

    assert live.points.support_views is not None
    assert live.points.support_views.dtype == cold.points.support_views.dtype
    assert live.points.support_views.tobytes() == (
        cold.points.support_views.tobytes()
    ), (
        "the live table differs from the rebuild's -- most likely it was "
        "read off _Chain.observed, which keeps one frame"
    )


def test_the_live_table_outlives_the_pruning_it_survives(window, intrinsics):
    """Stated directly against the prune, so the trap cannot creep back."""
    backend = ClassicalTwoViewBackend()
    backend.begin(intrinsics)
    for frame in window:
        backend.extend(frame)

    retained = {key[0] for key in backend._chain.observed}
    frames = set(
        np.unique(backend.snapshot().points.support_views[:, 0]).tolist()
    )

    # Bounded by the reference window, not fixed at one. `_extend` now
    # looks back EXTEND_REFERENCE_DEPTH keyframes for further sightings
    # of landmarks it already holds, so the prune must retain that many
    # frames rather than one. The property being defended is unchanged
    # and is the reason this test exists: what survives is a CONSTANT
    # number of frames, not a number that grows with the walk.
    assert 1 <= len(retained) <= EXTEND_REFERENCE_DEPTH, (
        f"the prune is still doing its job: retained {len(retained)} frames "
        f"against a reference depth of {EXTEND_REFERENCE_DEPTH}"
    )
    assert len(frames) > len(retained), (
        "and the association is not derived from it"
    )


def test_the_live_delta_names_only_the_landmarks_it_carries(window, intrinsics):
    """Extension.new_points is a delta, so its indices are delta-local."""
    backend = ClassicalTwoViewBackend()
    backend.begin(intrinsics)
    seen_any = False
    for frame in window:
        extension = backend.extend(frame)
        if extension.new_points is None:
            continue
        support = extension.new_points.support_views
        assert support is not None
        assert support[:, 2].min() >= 0
        assert support[:, 2].max() < len(extension.new_points), (
            "landmark_index indexes this block's own xyz, never the "
            "accumulated map"
        )
        seen_any = True
    assert seen_any, "this walk added no structure to compare"


# -- persistence --------------------------------------------------------


def _map_session(store, payloads, intrinsics):
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world("Support")
    session_id = engine.start_session(
        world_id,
        intrinsics=intrinsics,
        frame_source="synthetic",
        declared_size=(WIDTH, HEIGHT),
    )
    for index, payload in enumerate(payloads):
        engine.observe(payload, source_seq=index)
    engine.stop_session()
    return engine, world_id, session_id


@pytest.fixture(scope="module")
def walk_jpegs(scene, camera_matrix):
    return [
        ss.encode_jpeg(image)
        for image in ss.render_sequence(
            scene, ss.strafe(14, step=0.09), camera_matrix, WIDTH, HEIGHT
        )
    ]


@pytest.fixture
def built(tmp_path, walk_jpegs, intrinsics):
    store = WorldStore(tmp_path)
    engine, world_id, session_id = _map_session(store, walk_jpegs, intrinsics)
    engine.build(world_id, session_id)
    return store, world_id, session_id


def test_support_json_sits_beside_points_json(built):
    store, world_id, session_id = built
    path = store.derived_dir(world_id) / session_id / "support.json"
    data = json.loads(path.read_text())

    assert set(data) == {"support"}
    assert data["support"], "the walk solved but recorded no association"
    for row in data["support"]:
        assert len(row) == 4, (
            "[segment_index, frame_index, feature_index, point_index]"
        )
        assert all(isinstance(value, int) for value in row)


def test_the_persisted_rows_index_the_persisted_points(built):
    """point_index is segment-local, matching points.json's own ordering."""
    store, world_id, session_id = built
    derived = store.read_derived(world_id, session_id)
    support = np.asarray(derived["support"], dtype=np.int64)

    per_segment: dict[int, int] = {}
    for row in derived["points"]:
        per_segment[row["segment_index"]] = (
            per_segment.get(row["segment_index"], 0) + 1
        )
    for segment, frame, feature, point in support.tolist():
        assert segment in per_segment
        assert 0 <= point < per_segment[segment]
        assert feature >= 0
        assert frame >= 0


def test_the_persisted_frame_index_is_segment_local(built):
    """Same convention as in memory, one level out: within the segment."""
    store, world_id, session_id = built
    derived = store.read_derived(world_id, session_id)
    support = np.asarray(derived["support"], dtype=np.int64)

    per_segment: dict[int, int] = {}
    for row in derived["poses"]:
        per_segment[row["segment_index"]] = (
            per_segment.get(row["segment_index"], 0) + 1
        )
    for segment, frame, _feature, _point in support.tolist():
        assert 0 <= frame < per_segment[segment], (
            "a frame index that is not a position within its segment's "
            "keyframes cannot be joined against poses.json"
        )


def test_a_world_without_support_json_still_reads(derived_world):
    """~29 worlds on disk predate this file. Absent is not an error."""
    store, world_id, session_id = derived_world
    assert not (
        store.derived_dir(world_id) / session_id / "support.json"
    ).exists()

    derived = store.read_derived(world_id, session_id)

    assert derived is not None
    assert derived["poses"] and derived["points"]
    assert derived["support"] is None, "absent, and said so rather than []"


def test_deleting_support_json_leaves_the_world_readable(built):
    store, world_id, session_id = built
    (store.derived_dir(world_id) / session_id / "support.json").unlink()

    derived = store.read_derived(world_id, session_id)

    assert derived is not None
    assert derived["support"] is None


def test_writing_the_association_changes_no_geometry(
    tmp_path, walk_jpegs, intrinsics
):
    """The whole change's cost, stated: nothing about the reconstruction.

    Two engines over the same journal -- one that flushes a live solve,
    one cold -- must still produce identical poses and points. The
    equivalence test in test_world_builder_incremental.py is the oracle;
    this is the same claim narrowed to the files this change touches.
    """
    store = WorldStore(tmp_path)
    engine, world_id, session_id = _map_session(store, walk_jpegs, intrinsics)
    engine.build(world_id, session_id)
    derived = store.derived_dir(world_id) / session_id
    live_poses = (derived / "poses.json").read_bytes()
    live_points = (derived / "points.json").read_bytes()

    WorldBuilderEngine(store).build(world_id, session_id)

    assert (derived / "poses.json").read_bytes() == live_poses
    assert (derived / "points.json").read_bytes() == live_points
