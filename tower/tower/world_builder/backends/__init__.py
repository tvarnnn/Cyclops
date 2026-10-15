"""Geometry backends, and the rule for choosing one.

A backend is swapped, never edited-around: adding a learned pointmap
backend later should be a new module here plus a branch in
select_backend(), touching nothing that owns data.
"""

import logging

from tower.world_builder.backend import GeometryBackend
from tower.world_builder.backends.classical import ClassicalTwoViewBackend
from tower.world_builder.backends.unposed import UnposedBackend
from tower.world_builder.records import CameraIntrinsics

logger = logging.getLogger(__name__)

BACKEND_AUTO = "auto"
BACKEND_CLASSICAL = "classical"
BACKEND_UNPOSED = "unposed"

BACKEND_NAMES = (BACKEND_AUTO, BACKEND_CLASSICAL, BACKEND_UNPOSED)


class BackendSelection:
    """The chosen backend plus, crucially, whether it was a downgrade.

    A downgrade is recorded rather than silently applied. "The map has no
    poses" and "the map has no poses because you asked for a calibrated
    backend without calibration" are very different operator experiences,
    and only the second is actionable.
    """

    def __init__(self, backend, downgraded_from=None, downgrade_reason=None):
        self.backend = backend
        self.downgraded_from = downgraded_from
        self.downgrade_reason = downgrade_reason

    @property
    def was_downgraded(self) -> bool:
        return self.downgraded_from is not None


def select_backend(
    name: str, intrinsics: CameraIntrinsics, *, announce: bool = True
) -> BackendSelection:
    """Choose a backend, and say so the first time.

    `announce=False` is for rebuild-time re-selection: `build()` re-selects
    on every rebuild, so the warning is logged once per session rather than
    once per rebuild.

    The RETURNED selection is identical either way -- `downgraded_from`
    and `downgrade_reason` are always populated, so the session record is
    unaffected and nothing downstream has to care which call announced.
    """
    if name not in BACKEND_NAMES:
        raise ValueError(f"unknown backend {name!r}; expected one of {BACKEND_NAMES}")

    if name == BACKEND_UNPOSED:
        return BackendSelection(UnposedBackend())

    # AUTO and an explicit classical request agree whenever intrinsics are known.
    if intrinsics.is_known:
        return BackendSelection(ClassicalTwoViewBackend())

    if name == BACKEND_AUTO:
        # Recorded and announced exactly like the explicit request below,
        # even though AUTO had no other choice: the operator's question is
        # "why is there no geometry?", and the answer (nothing has
        # calibrated this camera) must reach the session record.
        reason = (
            "no backend that solves poses can run without intrinsics, and "
            f"this session's intrinsics are source={intrinsics.source!r}; "
            "the unposed backend will record keyframes and tracking but "
            "will produce no poses and no points"
        )
        if announce:
            logger.warning("world builder: backend selected unposed -- %s", reason)
    else:
        # Explicitly requested classical, without intrinsics.
        reason = (
            "classical backend requires intrinsics, but this session's "
            f"intrinsics are source={intrinsics.source!r}; no intrinsics were "
            "invented"
        )
        if announce:
            logger.warning("world builder: backend downgraded to unposed -- %s", reason)
    return BackendSelection(
        UnposedBackend(),
        downgraded_from=BACKEND_CLASSICAL,
        downgrade_reason=reason,
    )


__all__ = [
    "BACKEND_AUTO",
    "BACKEND_CLASSICAL",
    "BACKEND_NAMES",
    "BACKEND_UNPOSED",
    "BackendSelection",
    "ClassicalTwoViewBackend",
    "GeometryBackend",
    "UnposedBackend",
    "select_backend",
]
