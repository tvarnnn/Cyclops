"""The geometry backend seam.

This is the most important boundary in World Builder. Above it sits the
world model -- keyframes, journals, segments, persistence -- which is
GPU-free, dependency-free, and most of the code. Below it sits whatever
actually estimates geometry.

The point of the seam is that a feed-forward pointmap model (DA3 and its
kin) is a *backend*, not an architecture. Adopting one later should be a
new file under backends/ plus a line in select_backend(), touching nothing
that owns data.

Two invariants make that work, and both are pinned by tests:

1. Nothing under this module or backends/ imports store, engine, or paths.
   Geometry estimation never learns about disk, worlds, or revisions.
2. A backend that cannot justify a pose returns None with a degeneracy
   reason. That is a first-class answer, not an error -- it is exactly
   what lets the whole engine run honestly on uncalibrated footage
   instead of fabricating an intrinsic to keep the pipeline moving.

The estimation call takes a WINDOW of keyframes rather than a pair. A
pairwise interface would cripple a pointmap model, which reasons over a
whole submap; a per-frame interface cannot express relative geometry at
all. Poses come back in the window's own local frame; placing that frame
into the world is the caller's job, so the backend never sees a world
coordinate.

There is a second, narrower entry point: begin/extend/snapshot/reset.
It exists for cost, not for semantics -- see the comment on
GeometryBackend.begin.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.schema import (
    DEGENERACY_NONE,
    POSE_STATUS_UNAVAILABLE,
)


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can and cannot do, declared rather than inferred."""

    backend_id: str
    version: str
    requires_intrinsics: bool
    estimates_intrinsics: bool
    produces_dense_geometry: bool
    produces_metric_scale: bool
    preferred_window: int
    device: str = "cpu"


@dataclass(frozen=True)
class KeyframeInput:
    """One keyframe as the backend sees it: pixels and nothing else."""

    keyframe_id: str
    image_gray: np.ndarray


@dataclass(frozen=True)
class PoseEstimate:
    """A pose, or an honest refusal to give one.

    rotation/translation are in the WINDOW's local frame, with the window
    anchor at identity. translation is a direction of unit length when the
    status is SOLVED -- monocular two-view geometry cannot recover its
    magnitude, and pretending otherwise by leaving an arbitrary scale in
    place is how a plausible-looking wrong trajectory gets built.
    """

    keyframe_id: str
    status: str = POSE_STATUS_UNAVAILABLE
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None
    degeneracy: str = DEGENERACY_NONE
    # Every measured signal is carried even when the pose is refused --
    # a refusal that cannot be explained is indistinguishable from a bug.
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float | None = None
    median_triangulation_deg: float | None = None
    median_displacement_px: float | None = None
    cheirality_fraction: float | None = None
    r_h: float | None = None


@dataclass(frozen=True)
class PointBlock:
    """Triangulated structure in the window's local frame."""

    xyz: np.ndarray
    # Which 2-D feature in which keyframe produced which landmark: a flat
    # (M, 3) int32 table of rows [frame_index, feature_index,
    # landmark_index]. `None` means a backend does not report it, which is
    # different from "this block was observed by nothing".
    #
    # Flat and numpy-native rather than ragged (a dict, or a list per
    # landmark): registration filters it by frame and joins it against
    # matched feature indices, a boolean mask on a column; it costs 12 bytes
    # a row against ~200 for a dict entry, and it is the one piece of solve
    # state that CANNOT be pruned (_Chain.forget_before); and it round-trips
    # to JSON as integers with no dtype contract to get wrong.
    #
    # CONVENTIONS, which a consumer has to know to join anything:
    #
    #   frame_index    position within the window this block was solved
    #                  from, 0 == the anchor. NOT session-relative -- a
    #                  backend is handed one window and does not know
    #                  where it sits in a session. The engine tags the
    #                  rows with a segment on the way to disk, where the
    #                  same index reads as "position within the segment".
    #   feature_index  index into detect_and_describe()'s keypoints for
    #                  that frame. Reproducible, because detection is
    #                  deterministic; it is not a stored measurement.
    #   landmark_index row of THIS block's own `xyz`, never of some
    #                  accumulated map. Extension.new_points is a delta,
    #                  so its table names only the landmarks that delta
    #                  carries -- re-observations of older landmarks are
    #                  not expressible there and appear in snapshot().
    support_views: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


@dataclass(frozen=True)
class Extension:
    """What one keyframe added to a live solve.

    `pose` is that keyframe's own estimate. `new_points` is only the
    structure this keyframe created, never the whole map, so a live
    viewer can append instead of re-reading. Both are conveniences:
    snapshot() stays authoritative, because a backend that re-solves
    cannot promise the earlier poses it already returned are still
    current, and this type deliberately does not claim they are.
    """

    pose: PoseEstimate
    new_points: PointBlock | None = None
    # True ONLY on the keyframe whose failure broke the solve chain -- an
    # edge, not a level. If it stayed true afterwards the engine would
    # start a new segment per frame, turning one failure into dozens of
    # one-frame segments, which is worse than the cascade it replaces.
    chain_broken: bool = False


@dataclass(frozen=True)
class GeometryEstimate:
    """Everything a backend produces for one window."""

    poses: tuple[PoseEstimate, ...]
    points: PointBlock | None = None
    scale_is_metric: bool = False
    diagnostics: dict = field(default_factory=dict)


class GeometryBackend(ABC):
    """Estimates relative geometry for a window of keyframes."""

    capabilities: BackendCapabilities

    @abstractmethod
    def prepare(self, intrinsics: CameraIntrinsics) -> None:
        """Bind intrinsics. Must raise if required intrinsics are absent."""

    @abstractmethod
    def estimate_window(
        self, window: Sequence[KeyframeInput]
    ) -> GeometryEstimate: ...

    # -- the incremental seam --------------------------------------------
    #
    # estimate_window() is a from-scratch solve. Measured on this host
    # (classical backend only, images already in RAM, 480x360 synthetic
    # strafe), one solve costs 6.89 ms/kf at 4 keyframes rising to
    # 10.02 ms/kf at 64 (27.6 -> 641.2 ms): roughly O(N^1.2). So a walk
    # rebuilt every k keyframes by re-solving pays O(N^2/k), and asking for
    # more live updates costs more. Total backend work over a walk at
    # --rebuild-every 4, measured the same way, re-solving vs extending:
    # 360 vs 152 ms at 16 keyframes, 1325 vs 316 ms at 32, 5920 vs 777 ms at 64.
    #
    # These four methods carry one solve across many calls, so a rebuild
    # is a flush rather than a re-solve. They are an OPTIMISATION and
    # nothing else: for the same keyframes in the same order, extending
    # then snapshotting must equal estimate_window() over the whole
    # sequence, and tests/test_world_builder_incremental.py pins that
    # bit-for-bit against estimate_window as the oracle.
    #
    # The default implementation below buffers and re-solves. It is
    # correct for every backend and quadratic for every backend; it
    # exists so a future pointmap backend, which genuinely reasons over a
    # whole submap and cannot be extended one frame at a time, can be
    # dropped in without the engine learning anything about it. A backend
    # whose solve is already forward-only should override all four.

    def begin(self, intrinsics: CameraIntrinsics) -> None:
        """Bind intrinsics and start a fresh incremental solve.

        Must raise exactly where prepare() raises: an incremental path is
        not a way to get geometry out of a camera nobody calibrated.
        """
        self.prepare(intrinsics)
        self.reset()

    def reset(self) -> None:
        """Discard the incremental solve; a new segment starts a new one.

        Segments do not share a coordinate frame or a unit, so carrying
        state across one would be worse than useless -- it would look
        like continuity.
        """
        self._incremental_window: list[KeyframeInput] = []
        self._incremental_estimate: GeometryEstimate | None = None

    def extend(self, frame: KeyframeInput) -> Extension:
        window = self._incremental_buffer()
        window.append(frame)
        estimate = self.estimate_window(tuple(window))
        self._incremental_estimate = estimate
        pose = (
            estimate.poses[-1]
            if estimate.poses
            else PoseEstimate(keyframe_id=frame.keyframe_id)
        )
        # No new_points from the default path on purpose. A re-solve can
        # rebuild the entire cloud, so "the points this frame added" is
        # not a question it can answer; claiming a suffix of the new
        # cloud is new would be a guess dressed as a delta.
        return Extension(pose=pose)

    def snapshot(self) -> GeometryEstimate:
        """Everything solved since begin()/reset(). Must not mutate state."""
        estimate = getattr(self, "_incremental_estimate", None)
        return estimate if estimate is not None else GeometryEstimate(poses=())

    def _incremental_buffer(self) -> list[KeyframeInput]:
        window = getattr(self, "_incremental_window", None)
        if window is None:
            window = []
            self._incremental_window = window
        return window

    def release(self) -> None:
        """Free any held resource. Default no-op; must not raise."""
        return None
