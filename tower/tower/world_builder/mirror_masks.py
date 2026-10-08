"""Oracle mirror-interior masks for Exp3 (`PREREG-EXP3-MIRROR-ORACLE-20261008.md`,
addendum 1; design `review/codex/cx-EXP3-DESIGN-20261008.md`).

WHAT THIS IS. `TOWER_WORLD_MIRROR_MASKS` (`tower.config.world_mirror_masks_setting`),
default OFF, independent of `TOWER_WORLD_SOLVE_MASKS`: either, neither or both may be
on. It reads a hand-drawn, blind-annotated polygon JSONL -- one record per annotated
keyframe, `{"walk", "keyframe", "image_sha256", "width", "height", "uncertain",
"polygons"}`, coordinates on the ORIGINAL raw frame -- and unions the visible mirror
INTERIOR into:

  * the final solve's feature-extraction mask (`global_solve.py`'s `_ensure_mirror_masks`),
    so no match touching the mirror reaches the model, exactly as the existing transient
    (hands/arms/phone) mask does, but kept as a SEPARATE record (`solve.mirror`): the
    evidence gate's hard dependency is `transients.state` (TOWER_WORLD_SOLVE_MASKS), which
    this module must never touch or imply;
  * the dense/fusion validity mask (`dense_pipeline.py`'s depth and fuse stages), excluding
    the mirror from sparse fit anchors and from `VALID[ki]`, unioned with but never replacing
    the redaction-fill exclusion (`_fill.npy`) -- the two keep separate cache keys and files
    (`_mirror.npy`), so an OFF solve/dense run is untouched byte for byte.

A keyframe with NO record in the polygon file is simply outside the annotated range (the
closed intervals the prereg locks): nothing is excluded for it, exactly as if the switch
were off for that one frame. A record with `polygons: []` (mirror screened `no`, or none
visible) is a frame that IS in range and explicitly has nothing to exclude -- also nothing
excluded, but deliberately rather than by omission.

NEVER FATAL. A missing/unreadable polygon file, a bad record, a frame whose width/height
disagree with the record, or an undistortion failure: the caller's mask step logs it and
carries on without the oracle, with its own record saying so. The transient (hands/phone)
mask step, if any, is never affected.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = 1

STATE_OFF = "off"
STATE_OK = "ok"
STATE_FAILED = "failed"

_HEX64 = frozenset("0123456789abcdef")


class MirrorPolygonError(ValueError):
    """A polygon JSONL record (or the file itself) is malformed. Raised by
    `load_polygons`; every caller catches it and runs without the oracle."""


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(cond: bool, line_no: int, why: str) -> None:
    if not cond:
        raise MirrorPolygonError(f"mirror polygon record at line {line_no}: {why}")


def _validate_polygon(poly, line_no: int, width: int, height: int) -> list:
    _require(isinstance(poly, list) and len(poly) >= 3, line_no,
              "a polygon needs at least 3 vertices")
    out = []
    for vertex in poly:
        _require(isinstance(vertex, (list, tuple)) and len(vertex) == 2, line_no,
                  f"vertex {vertex!r} is not an [x, y] pair")
        x, y = vertex
        _require(isinstance(x, (int, float)) and isinstance(y, (int, float)), line_no,
                  f"vertex {vertex!r} is not numeric")
        _require(float(x).is_integer() and float(y).is_integer(), line_no,
                  f"vertex {vertex!r} is not an integer pixel corner")
        x, y = int(x), int(y)
        _require(0 <= x <= width and 0 <= y <= height, line_no,
                  f"vertex ({x}, {y}) is outside the {width}x{height} frame")
        out.append((x, y))
    return out


def load_polygons(path) -> dict:
    """`{(walk, keyframe): record}` from the canonical JSONL (prereg §Lock): one JSON
    object per line, sorted order is NOT required to read it (only to hash it). Raises
    `MirrorPolygonError` on any malformed or duplicate record -- no interpolation, no
    silent skip: a replay must know its polygon set is exactly what it asked for."""
    text = Path(path).read_text(encoding="utf-8")
    out: dict = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MirrorPolygonError(f"line {line_no} is not valid JSON: {exc}") from exc
        _require(isinstance(rec, dict), line_no, "not a JSON object")
        for key in ("walk", "keyframe", "image_sha256", "width", "height", "uncertain", "polygons"):
            _require(key in rec, line_no, f"missing {key!r}")
        walk, keyframe = rec["walk"], rec["keyframe"]
        _require(isinstance(walk, int) and isinstance(keyframe, int), line_no,
                  "walk/keyframe are integers")
        width, height = rec["width"], rec["height"]
        _require(isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0,
                  line_no, "width/height are positive integers")
        sha = rec["image_sha256"]
        _require(isinstance(sha, str) and len(sha) == 64 and set(sha.lower()) <= _HEX64,
                  line_no, "image_sha256 is not a 64-hex digest")
        _require(isinstance(rec["uncertain"], bool), line_no, "uncertain is a boolean")
        polys = rec["polygons"]
        _require(isinstance(polys, list), line_no, "polygons is a list")
        validated = [_validate_polygon(p, line_no, width, height) for p in polys]
        key = (walk, keyframe)
        _require(key not in out, line_no, f"duplicate record for walk {walk} keyframe {keyframe}")
        out[key] = {**rec, "polygons": validated}
    return out


def rasterize_polygons(polygons: list, width: int, height: int) -> np.ndarray:
    """The UNION of every polygon's visible interior, on the raw pixel grid: a boolean
    (height, width) array, True = inside at least one polygon. Empty `polygons` is an
    all-False mask -- exactly "nothing to exclude". No dilation (prereg §Geometry)."""
    mask = np.zeros((height, width), bool)
    if not polygons:
        return mask
    import cv2  # noqa: PLC0415

    canvas = np.zeros((height, width), np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, np.int32).reshape(-1, 1, 2)
        # One polygon per call, OR'd by `fillPoly` writing 1s into the same canvas:
        # the union of disconnected visible regions (prereg §Geometry), not an
        # even-odd parity ACROSS polygons -- only a self-intersecting single polygon
        # needs that, and none of the annotation tooling produces one.
        cv2.fillPoly(canvas, [pts], 1)
    return canvas.astype(bool)


def oracle_mask_in_solver_space(record: dict, *, m1, m2, roi, out_shape) -> np.ndarray:
    """`record`'s polygons, rasterized on its own raw grid and remapped through the
    same undistortion maps (`global_solve._undistort_maps`) the solver/dense image was
    made with -- nearest-neighbor, exactly as the redaction-fill mask is (prereg
    §Geometry: "same documented undistortion/ROI map ... nearest-neighbor
    rasterization"). Returns a boolean array of `out_shape` (the undistorted ROI)."""
    import cv2  # noqa: PLC0415

    raw = rasterize_polygons(record["polygons"], record["width"], record["height"])
    x, y, rw, rh = roi
    remapped = cv2.remap(raw.astype(np.uint8) * 255, m1, m2, cv2.INTER_NEAREST)
    cropped = remapped[y:y + rh, x:x + rw] > 0
    if cropped.shape != tuple(out_shape):
        raise MirrorPolygonError(
            f"remapped mirror mask is {cropped.shape}, solver/dense image is {tuple(out_shape)}")
    return cropped


def interior_mask_for_frame(polygons_by_key: dict, walk: int | None, source_seq: int | None, *,
                            m1, m2, roi, out_shape) -> np.ndarray | None:
    """The oracle interior for one keyframe, in solver/dense pixel space, or None when
    the frame has no record at all (outside the annotated closed interval -- untouched,
    not an exclusion of anything)."""
    if walk is None or source_seq is None:
        return None
    record = polygons_by_key.get((int(walk), int(source_seq)))
    if record is None:
        return None
    return oracle_mask_in_solver_space(record, m1=m1, m2=m2, roi=roi, out_shape=out_shape)


# -- records (kept separate from `solve_masks`'/`transients`' gate-critical record) --------

def off_record(detail: str | None = None) -> dict:
    return {"schema": SCHEMA, "requested": False, "state": STATE_OFF,
            "detail": detail or "TOWER_WORLD_MIRROR_MASKS is off"}


def failed_record(detail: str) -> dict:
    return {"schema": SCHEMA, "requested": True, "state": STATE_FAILED, "detail": detail}


def applied_record(*, polygons_path: str, polygons_sha256: str, walk: int, images: int,
                   images_with_record: int, images_touched: int, oracle_pixels_excluded: int,
                   oracle_pixel_fraction: float) -> dict:
    return {"schema": SCHEMA, "requested": True, "state": STATE_OK, "detail": None,
            "polygons_path": str(polygons_path), "polygons_sha256": polygons_sha256,
            "walk": walk, "images": images, "images_with_record": images_with_record,
            "images_touched": images_touched, "oracle_pixels_excluded": oracle_pixels_excluded,
            "oracle_pixel_fraction": round(float(oracle_pixel_fraction), 6)}


# -- the solve's feature-extraction consumers (`solve_masks.filter_walk_database`, --------
# -- `solve_masks.masked_database`, `solve_masks._keypoints_in_mask`) only read `.masked` --
# -- (name -> {image_sha1, mask_sha1, masked_frac}) and `.params.rule_id()`. This is a ------
# -- duck-typed stand-in for `solve_masks.SolverMasks` so a mirror-only solve (transient ---
# -- masks off) can drive the same walk-database-filtered / re-extracted machinery without -
# -- borrowing `SolverMasks`' hands/phone-specific `TransientParams` or its gate record. ----

class _MirrorRule:
    def rule_id(self) -> str:
        return "mirror-oracle"


@dataclass
class MirrorExtractionMasks:
    masked: dict = field(default_factory=dict)
    params: _MirrorRule = field(default_factory=_MirrorRule)
    # Read by `solve_masks.masked_database` alongside `.masked` (an image the re-extract
    # path excluded outright). The mirror mask never excludes a whole image -- only an
    # interior region -- so this is always empty; the attribute still has to exist.
    excluded: dict = field(default_factory=dict)
