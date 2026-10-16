#!/usr/bin/env python
"""Recover camera intrinsics from ChArUco board views.

ChArUco rather than a plain checkerboard because the wearer cannot see the
frame: ChArUco identifies each corner individually, so a partially visible
or partly occluded board still contributes, where a plain checkerboard
needs the whole grid.

    # print this, mount it flat and rigid
    .venv\\Scripts\\python.exe scripts/calibrate_charuco.py --generate-board board.png

    # then calibrate from frames captured THROUGH THE REAL PIPELINE
    .venv\\Scripts\\python.exe scripts/calibrate_charuco.py \\
        --frames data/captures/<capture_id>/frames

With no --out, the result lands in the intrinsics store under the world
root -- `data/world_builder/intrinsics/<width>x<height>.json` -- which is
exactly where `world_build_session.py` looks it up by the OBSERVED frame
size. That default is the whole point: no flag is needed at either end,
and a calibration written elsewhere is not found. See
`docs/CALIBRATION.md` for the physical procedure.

WHAT THIS HAS AND HAS NOT BEEN VALIDATED AGAINST
------------------------------------------------
The code path is unit-tested against synthetically rendered board views
and recovers a known focal length to within ~0.3%. That validates the
wiring -- which genuinely matters, because `calibrateCameraCharuco` does
NOT exist in OpenCV 5 and the working sequence is CharucoDetector ->
matchImagePoints -> calibrateCamera.

It has NOT been validated against:

- any real lens. The synthetic input is a perfect pinhole with zero
  distortion, so the distortion-model choice -- the parameter most likely
  to be wrong on a ~100-degree wearable lens -- is completely unexercised.
- real JPEG compression, motion blur, rolling shutter, or auto-exposure.
- DAT's adaptive resolution ladder, which switches mid-stream.

Even noise-free, the principal point came back ~2.5 px off truth. It is
the weakly-constrained parameter before any real-world effect is added.
Treat a first real calibration as provisional and check its reprojection
RMS and view count before trusting it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder.intrinsics_store import IntrinsicsStore  # noqa: E402
from tower.world_builder.records import CameraIntrinsics  # noqa: E402
from tower.world_builder.schema import (  # noqa: E402
    INTRINSICS_SOURCE_SELF_CALIBRATED,
)

# Must equal `world_build_session.DEFAULT_ROOT` (the `--root` default there),
# so the calibration is looked up where it was written.
DEFAULT_WORLD_ROOT = Path("data/world_builder")

# Board geometry. Square/marker lengths are in metres and must match the
# printed board, because they set the scale of everything downstream.
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH_M = 0.040
MARKER_LENGTH_M = 0.030
DICTIONARY = cv2.aruco.DICT_4X4_50

# Refuse below this many usable views rather than emitting a shaky
# calibration. A calibration nobody can tell is bad is worse than none:
# every downstream pose inherits its error and nothing reports it.
MIN_VIEWS = 8
# Corners per view below which the view carries too little constraint.
MIN_CORNERS_PER_VIEW = 8

# View COUNT alone is not sufficient. Ten byte-identical board views recover
# fx with 287% error; ten fronto-parallel views at a single distance recover
# it with 3787% error.
# Both come back source="self_calibrated", is_known=True.
#
# They also score a BETTER reprojection RMS (0.158 and 0.217 px) than a
# correct calibration (0.253 px) -- because RMS measures fit to the views
# you supplied, and degenerate views are trivially easy to fit. So a low
# RMS here is not reassurance; on this failure mode it is actively
# misleading, and it cannot be the quality gate.
#
# Viewpoint DIVERSITY is the gate. Focal length is constrained by
# perspective foreshortening, so a board held parallel to the sensor
# contributes almost nothing toward pinning it down no matter how many
# times it is photographed.
#
# Measured tilt of the board-to-image homography's perspective row,
# normalised by board extent, across board orientations:
#
#     fronto-parallel (0 deg)   median 0.0067   max 0.0092
#     barely tilted   (~3 deg)  median 0.0152
#     mildly tilted   (~9 deg)  median 0.0511
#     well tilted     (~26 deg) median 0.1417
#
# A fronto-parallel view does NOT measure zero -- corner-detection noise
# puts a floor around 0.009. The threshold sits above that floor and above
# a 3-degree tilt, which is still too little to separate focal length from
# distance. Roughly: the board must be visibly angled, not merely held
# imperfectly.
MIN_TILTED_VIEWS = 4
MIN_VIEW_TILT = 0.03
MIN_VIEW_SEPARATION_PX = 8.0


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary
    )


def generate_board_image(path: Path, pixels_per_square: int = 120) -> None:
    board = make_board()
    size = (SQUARES_X * pixels_per_square, SQUARES_Y * pixels_per_square)
    cv2.imwrite(str(path), board.generateImage(size))


def detect_views(images, board):
    """Detect ChArUco corners in each image.

    Returns (object_points, image_points, size, per_view_corner_counts).
    """
    detector = cv2.aruco.CharucoDetector(board)
    object_points, image_points, corner_counts = [], [], []
    size = None

    for image in images:
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        )
        size = (gray.shape[1], gray.shape[0])
        corners, ids, _, _ = detector.detectBoard(gray)
        if corners is None or ids is None or len(ids) < MIN_CORNERS_PER_VIEW:
            corner_counts.append(0 if ids is None else int(len(ids)))
            continue
        object_view, image_view = board.matchImagePoints(corners, ids)
        if object_view is None or len(object_view) < MIN_CORNERS_PER_VIEW:
            corner_counts.append(0)
            continue
        object_points.append(object_view)
        image_points.append(image_view)
        corner_counts.append(int(len(object_view)))

    return object_points, image_points, size, corner_counts


def _view_tilt(object_points, image_points) -> float:
    """How much perspective a single view carries.

    The board is planar, so model-to-image is a homography. Its bottom row
    is precisely what makes that map projective rather than affine: near
    zero means the board faced the sensor squarely, and an affine view
    cannot separate focal length from distance.

    Normalised by board extent so the measure does not depend on the
    board's size in metres or on how large it appears in frame.
    """
    model = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)[:, :2]
    image = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(model) < 4:
        return 0.0
    homography, _ = cv2.findHomography(model, image, 0)
    if homography is None or abs(homography[2, 2]) < 1e-12:
        return 0.0
    homography = homography / homography[2, 2]
    extent = float(np.linalg.norm(model.max(axis=0) - model.min(axis=0)))
    return float(
        np.hypot(homography[2, 0], homography[2, 1]) * max(extent, 1e-9)
    )


def _assess_diversity(object_points, image_points) -> str | None:
    """Why these views cannot constrain a calibration, or None if they can.

    Guards the two failure modes that view count and reprojection RMS both
    miss and that yield confidently wrong intrinsics.
    """
    centres = [
        np.asarray(points, dtype=np.float64).reshape(-1, 2).mean(axis=0)
        for points in image_points
    ]
    distinct: list[np.ndarray] = []
    for centre in centres:
        if all(
            float(np.linalg.norm(centre - kept)) >= MIN_VIEW_SEPARATION_PX
            for kept in distinct
        ):
            distinct.append(centre)
    if len(distinct) < MIN_VIEWS:
        return (
            f"only {len(distinct)} sufficiently distinct viewpoints among "
            f"{len(image_points)} views (need {MIN_VIEWS}) -- the board "
            "barely moved between shots, so the extra views add no "
            "constraint"
        )

    tilts = [
        _view_tilt(model, image)
        for model, image in zip(object_points, image_points)
    ]
    tilted = sum(1 for tilt in tilts if tilt >= MIN_VIEW_TILT)
    if tilted < MIN_TILTED_VIEWS:
        return (
            f"only {tilted} of {len(tilts)} views carry meaningful "
            f"perspective (need {MIN_TILTED_VIEWS}) -- the board was held "
            "nearly parallel to the sensor throughout, which leaves focal "
            "length almost unidentifiable. Tilt the board between shots"
        )
    return None


def calibrate(images) -> CameraIntrinsics:
    """Solve intrinsics, or raise if the evidence is too thin.

    Raises ValueError rather than returning a low-confidence result: an
    under-constrained calibration is silently wrong everywhere it is
    later used, and there is no downstream check that would catch it.
    """
    board = make_board()
    object_points, image_points, size, corner_counts = detect_views(images, board)

    if len(object_points) < MIN_VIEWS:
        raise ValueError(
            f"only {len(object_points)} usable views of the board "
            f"(need {MIN_VIEWS}); per-view corner counts {corner_counts}. "
            "Refusing to emit intrinsics from insufficient evidence."
        )

    problem = _assess_diversity(object_points, image_points)
    if problem is not None:
        raise ValueError(
            f"refusing to emit intrinsics: {problem}. Note that reprojection "
            "error cannot catch this -- degenerate views fit BETTER than "
            "good ones, so a low RMS here would be actively misleading."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, size, None, None
    )
    return CameraIntrinsics(
        source=INTRINSICS_SOURCE_SELF_CALIBRATED,
        model="pinhole_radtan",
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]),
        cy=float(camera_matrix[1, 2]),
        dist_coeffs=tuple(float(v) for v in np.asarray(dist_coeffs).ravel()),
        calibrated_width=size[0],
        calibrated_height=size[1],
        reprojection_rms_px=float(rms),
        view_count=len(object_points),
        calibrated_at=time.time(),
        # Left None deliberately. Whether DAT resizes or crops between its
        # three resolutions has never been established, and asserting
        # linearity would let intrinsics be silently rescaled by a wrong
        # factor. Establish it by calibrating at two resolutions.
        scales_linearly_across_resolutions=None,
    )


def load_images(directory: Path):
    paths = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png"))
    if not paths:
        raise SystemExit(f"no .jpg/.png frames found under {directory}")
    return [cv2.imread(str(path)) for path in paths]


def split_half_report(images) -> tuple[CameraIntrinsics, CameraIntrinsics, float]:
    """Calibrate two disjoint halves and measure how far apart they land.

    The quality check that reprojection RMS cannot be. RMS says the solver
    fitted the views it was given; agreement between two independent
    halves says those views actually CONSTRAIN the camera. Measured on
    synthetic sets: 0.31% and 1.94% spread on well-varied captures, 4.55%
    on a deliberately marginal one whose full-set fx error was only 0.13%
    and whose RMS was indistinguishable from the good ones.

    Honest limit: this detects under-constraint. It cannot detect a
    systematically wrong board -- an anisotropically printed one, say --
    because both halves photograph the same board.
    """
    first = calibrate(images[0::2])
    second = calibrate(images[1::2])
    spread = abs(first.fx - second.fx) / ((first.fx + second.fx) / 2.0)
    return first, second, spread


# Above this, the two halves disagree enough that the views did not pin
# the camera down and the capture should be repeated. Not a hard refusal:
# --split-half is a diagnostic that writes nothing, so there is no unsafe
# artifact to guard against, and the operator is better served by the
# number than by an exit code.
SPLIT_HALF_WARN_SPREAD = 0.05


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Recover camera intrinsics from ChArUco board views."
    )
    parser.add_argument(
        "--generate-board",
        type=Path,
        help="Write a printable board image and exit.",
    )
    parser.add_argument("--frames", type=Path, help="Directory of board views.")
    parser.add_argument(
        "--root",
        type=artifact_root_arg,
        default=str(DEFAULT_WORLD_ROOT),
        help=(
            "World root holding the intrinsics store "
            f"(default: {DEFAULT_WORLD_ROOT}). Must match the --root the "
            "builder runs with."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Write intrinsics JSON here instead of into the store. The "
            "default is <root>/intrinsics/<width>x<height>.json, which is "
            "where world_build_session.py discovers it -- and it can only "
            "be computed AFTER calibrating, because the resolution is a "
            "result. A file written elsewhere is not discovered."
        ),
    )
    parser.add_argument(
        "--split-half",
        action="store_true",
        help=(
            "Diagnostic: calibrate two disjoint halves of the views and "
            "report how far apart they land. Writes nothing. This is the "
            "quality check reprojection RMS cannot be -- degenerate views "
            "score a BETTER RMS than good ones."
        ),
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    if args.generate_board:
        generate_board_image(args.generate_board)
        print(f"board written to {args.generate_board}")
        print(
            f"squares {SQUARES_X}x{SQUARES_Y}  square {SQUARE_LENGTH_M} m  "
            f"marker {MARKER_LENGTH_M} m"
        )
        print(
            "Print at a known scale, mount flat and rigid, and MEASURE the "
            "printed square size -- the metres above must match reality."
        )
        return 0

    if not args.frames:
        parser.error("--frames is required unless --generate-board is given")

    images = load_images(args.frames)

    if args.split_half:
        try:
            first, second, spread = split_half_report(images)
        except ValueError as exc:
            print(
                f"split-half refused: {exc}\n\nA half that cannot be "
                "calibrated on its own is itself the answer: this capture "
                "does not carry enough independent evidence.",
                file=sys.stderr,
            )
            return 1
        print(
            f"half A  fx={first.fx:8.2f}  fy={first.fy:8.2f}  "
            f"rms={first.reprojection_rms_px:.4f}  views={first.view_count}"
        )
        print(
            f"half B  fx={second.fx:8.2f}  fy={second.fy:8.2f}  "
            f"rms={second.reprojection_rms_px:.4f}  views={second.view_count}"
        )
        print(f"\nfx disagreement between halves: {100 * spread:.2f}%")
        if spread > SPLIT_HALF_WARN_SPREAD:
            print(
                f"ABOVE {100 * SPLIT_HALF_WARN_SPREAD:.0f}%: the views do not "
                "pin this camera down. Recapture with more tilt, more "
                "distance variation, and the board in more parts of the frame."
            )
        else:
            print(
                "Within tolerance -- the two halves agree, so the views "
                "constrain the camera. This does NOT rule out a "
                "systematically wrong board: both halves photographed the "
                "same one. Check the print (docs/CALIBRATION.md section 3)."
            )
        print("\nNothing was written; --split-half is a diagnostic.")
        return 0

    try:
        intrinsics = calibrate(images)
    except ValueError as exc:
        print(f"calibration refused: {exc}", file=sys.stderr)
        return 1

    payload = intrinsics.to_json_dict()
    width = intrinsics.calibrated_width
    height = intrinsics.calibrated_height

    # No --out means the store, not "no file". A calibration nobody can
    # find is the same as no calibration, and the store is the only place
    # the builder looks.
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        out_path = args.out
        in_store = False
    else:
        out_path = IntrinsicsStore(args.root).save(intrinsics)
        in_store = True

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"source            {intrinsics.source}")
        print(f"resolution        {width}x{height}")
        print(f"fx, fy            {intrinsics.fx:.2f}, {intrinsics.fy:.2f}")
        print(f"cx, cy            {intrinsics.cx:.2f}, {intrinsics.cy:.2f}")
        print(f"reprojection RMS  {intrinsics.reprojection_rms_px:.4f} px")
        print(f"views used        {intrinsics.view_count}")
        print(f"written to        {out_path}")

        print(f"\nVALID ONLY FOR {width}x{height} FRAMES.")
        print(
            "scales_linearly_across_resolutions is null, so nothing will "
            "rescale these to another size -- a build at any other "
            "resolution will behave as if this calibration did not exist. "
            "Calibrate again at another DAT resolution to establish "
            "linearity."
        )

        print("\nNEXT:")
        if in_store:
            print(
                f"  Nothing else to wire up. A session whose frames measure "
                f"{width}x{height} will find this automatically."
            )
        else:
            print(
                "  This file is OUTSIDE the store, so it will NOT be "
                "discovered. Either re-run without --out, or pass "
                f"--intrinsics {out_path} to every build."
            )
        print(
            "  .venv\\Scripts\\python.exe scripts/world_build_session.py "
            "--follow-capture data/captures/<capture_id>"
        )
        print(
            "  Confirm it was picked up: the run logs "
            f"'using calibration ... {width}x{height}' at start, and the "
            "session report says backend_id=classical, not unposed."
        )
        print(
            "\n  A LOW RMS IS NOT PROOF. Degenerate views fit better than "
            "good ones. See docs/CALIBRATION.md for how to check this "
            "result is real."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
