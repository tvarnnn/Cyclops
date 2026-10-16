"""A resolution-keyed store of camera calibrations on disk.

It closes the loop between `calibrate_charuco.py` and
`world_build_session.py`: the calibrator files a calibration where the
builder looks for one, so neither end needs an operator to remember a
flag. Measured on a walk with no calibration on file: 155 keyframes
accepted, 0 camera poses solved and 0 map points, with nothing saying
why -- so every miss now says where it looked (`lookup`).

WHERE CALIBRATIONS LIVE, AND WHY

    <world_root>/intrinsics/<width>x<height>.json

Beside `worlds/`, not inside one. A calibration describes the CAMERA, not
a world and not a session: the same glasses build every world on this
Tower, and filing the calibration under the world that happened to be
recording when it was measured would mean the next world starts
uncalibrated again. Worlds are purgeable for privacy (`purge_world`
deletes raw imagery); a calibration contains no imagery and must survive
that.

The resolution is in the FILENAME rather than only inside the record, so
"is there a calibration for this frame size?" is a file-existence
question. A single `intrinsics.json` would have to be opened, parsed, and
then rejected for the wrong resolution -- and the whole failure mode this
guards against is a calibration that loads fine and is silently wrong by
a crop factor. DAT's adaptive ladder changes resolution mid-stream, so
several calibrations coexisting is the expected steady state, not an edge
case.

WHAT THIS STORE REFUSES TO DO

It never rescales. `records.CameraIntrinsics.scaled_to` already refuses
unless `scales_linearly_across_resolutions` is explicitly True, and
whether DAT resizes or crops between its three modes has never been
established. A 480x360 calibration therefore does NOT satisfy a 360x640
lookup, even though a session can declare one while its keyframes measure
the other.

A MISS IS NOT AN ERROR

`lookup` returns `CameraIntrinsics.unknown()` for a missing, corrupt,
truncated, mismatched or physically-impossible record. An uncalibrated
Tower must keep working exactly as it does today: honest keyframes, the
unposed backend, no poses. Following `store.WorldStore`'s posture toward
unreadable derived output -- warn and treat as absent -- rather than
raising, because a calibration file is an optional input and a Tower that
refuses to map because one is malformed is worse than one that maps
without poses and says so.
"""

import json
import logging
from pathlib import Path

from tower.storage import read_json_closed, write_json_atomic
from tower.world_builder.records import (
    CameraIntrinsics,
    camera_intrinsics_from_json_dict,
)

logger = logging.getLogger(__name__)

INTRINSICS_DIRNAME = "intrinsics"


def intrinsics_filename(width: int, height: int) -> str:
    return f"{int(width)}x{int(height)}.json"


class IntrinsicsStore:
    """Filesystem storage for camera calibrations, keyed by resolution."""

    def __init__(self, root: Path) -> None:
        # The WORLD root (e.g. data/world_builder), not the intrinsics
        # directory itself, so callers pass the same --root they already
        # pass to WorldStore and cannot point the two at different trees.
        self._root = Path(root)

    @property
    def directory(self) -> Path:
        return self._root / INTRINSICS_DIRNAME

    def path_for(self, width: int, height: int) -> Path:
        """Where a calibration for this resolution would live.

        Public and total: it answers for resolutions that have no
        calibration, because the log line an operator needs on a MISS is
        "nothing at <this path>", and only this store knows that path.
        """
        return self.directory / intrinsics_filename(width, height)

    def lookup(self, width: int, height: int) -> CameraIntrinsics:
        """The calibration for exactly this resolution, or `unknown()`.

        Never raises and never rescales. Every non-hit is logged, so a
        session with no geometry can say which calibration it looked for.
        """
        path = self.path_for(width, height)
        if not path.exists():
            logger.info(
                "world builder: no calibration for %sx%s (looked for %s); "
                "intrinsics stay unknown and no poses will be solved",
                width,
                height,
                path,
            )
            return CameraIntrinsics.unknown()

        try:
            data = read_json_closed(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "world builder: calibration at %s is unreadable (%s); "
                "treating as absent",
                path,
                exc,
            )
            return CameraIntrinsics.unknown()

        try:
            intrinsics = camera_intrinsics_from_json_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            # A partial record -- hand-edited, truncated, or written by a
            # different schema. `camera_intrinsics_from_json_dict` indexes
            # required keys and raises on a self_calibrated record with no fx.
            logger.warning(
                "world builder: calibration at %s is not a usable "
                "CameraIntrinsics record (%s); treating as absent",
                path,
                exc,
            )
            return CameraIntrinsics.unknown()

        if not intrinsics.is_known:
            # Parsed, but source=unknown or a non-finite / non-positive
            # focal length. `is_known` is the physical-possibility gate;
            # honouring it here stops an impossible camera reaching the
            # classical backend, which would build a confident
            # reconstruction from it.
            logger.warning(
                "world builder: calibration at %s does not describe a "
                "usable camera (source=%r fx=%r fy=%r); treating as absent",
                path,
                intrinsics.source,
                intrinsics.fx,
                intrinsics.fy,
            )
            return CameraIntrinsics.unknown()

        if (
            intrinsics.calibrated_width != width
            or intrinsics.calibrated_height != height
        ):
            # The filename says one resolution and the record says
            # another: a file copied or renamed by hand. Refuse rather
            # than trust either. Rescaling is not on the table -- see the
            # module docstring.
            logger.warning(
                "world builder: calibration at %s was measured at %sx%s but "
                "was filed under %sx%s; refusing to use it. Intrinsics are "
                "not rescaled across resolutions -- calibrate at %sx%s",
                path,
                intrinsics.calibrated_width,
                intrinsics.calibrated_height,
                width,
                height,
                width,
                height,
            )
            return CameraIntrinsics.unknown()

        logger.info(
            "world builder: using calibration %s -- source=%s %sx%s "
            "fx=%.2f fy=%.2f reprojection_rms=%s px views=%s",
            path,
            intrinsics.source,
            intrinsics.calibrated_width,
            intrinsics.calibrated_height,
            intrinsics.fx,
            intrinsics.fy,
            (
                f"{intrinsics.reprojection_rms_px:.4f}"
                if intrinsics.reprojection_rms_px is not None
                else "unrecorded"
            ),
            intrinsics.view_count,
        )
        return intrinsics

    def save(self, intrinsics: CameraIntrinsics) -> Path:
        """File a calibration under its OWN resolution. Returns the path.

        The key comes from the record, never from a caller-supplied
        resolution: a calibration filed under a resolution it was not
        measured at is the one thing `lookup` cannot recover from
        honestly, so the writer is not given the chance to get it wrong.

        Raises rather than degrading. A miss on read is a normal state of
        the world; a save that silently does not happen is a bug that
        surfaces hours later as "why is there still no geometry?".
        """
        if not intrinsics.is_known:
            raise ValueError(
                "refusing to store intrinsics that do not describe a usable "
                f"camera (source={intrinsics.source!r}, fx={intrinsics.fx!r}, "
                f"fy={intrinsics.fy!r}). A stored unknown is indistinguishable "
                "from a calibration that was never run."
            )
        if intrinsics.calibrated_width is None or intrinsics.calibrated_height is None:
            raise ValueError(
                "refusing to store intrinsics with no recorded resolution: "
                "this store is keyed by the resolution the calibration was "
                "measured at, and intrinsics are never rescaled."
            )

        path = self.path_for(
            intrinsics.calibrated_width, intrinsics.calibrated_height
        )
        write_json_atomic(path, intrinsics.to_json_dict())
        logger.info(
            "world builder: stored %sx%s calibration at %s",
            intrinsics.calibrated_width,
            intrinsics.calibrated_height,
            path,
        )
        return path

    def list_resolutions(self) -> tuple[tuple[int, int], ...]:
        """Every resolution that has a calibration file, sorted.

        For operator-facing output: "you have 480x360, you are streaming
        360x640" is a far more actionable message than "no calibration".
        Derived from filenames only -- it does not open or validate the
        records, so a name that is not `<w>x<h>.json` is skipped.
        """
        found: list[tuple[int, int]] = []
        if not self.directory.exists():
            return ()
        for path in self.directory.glob("*.json"):
            width, _, height = path.stem.partition("x")
            try:
                found.append((int(width), int(height)))
            except ValueError:
                continue
        return tuple(sorted(found))
