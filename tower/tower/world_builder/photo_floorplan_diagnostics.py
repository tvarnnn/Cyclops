"""Photo floor plan NEW-SCENE DIAGNOSTICS (manager 349 lane A; generalization audit item 5).

BEHAVIOUR-NEUTRAL. Nothing here feeds a raster, a tile, the manifest, the regression guard or
any disposition. The worker computes a landing's diagnostics only AFTER that landing's
disposition is final (so even the time spent here cannot move a landing across the wall
cap), from data the landing already produced, and any exception here is swallowed and
logged. Pure: no I/O except `write_files`, which the worker calls only when
TOWER_WORLD_PHOTO_FLOORPLAN_DIAGNOSTICS is exactly `on`.

WHAT IS RECORDED

Per landing (`landing_record`, one bounded JSON object, <= MAX_RECORD_BYTES): identity,
disposition, published / escaped (a regressed landing published by the escape valve),
landing lag (solve -> landing, wall clock), coverage lag (solve -> coverage receipt), the
coverage receipt's omitted stations and components, and per component (`component_record`):

  - plane: ok, source (estimated / reused_sim3 / fallback_camera_drop / rejected), the
    reference normal used (camera_plane / roll_free / receipt_up, and its vector), tilt,
    inliers, candidates, inlier spread (`support_cells`: 8x8 cells holding >= 3 inliers),
    the cells the cameras themselves occupy on the same grid, the second-plane ambiguity
    (runner-up support ratio and its height relative to the chosen plane, in H), the median
    camera height H, the camera-height CV and the camera-plane spread / flatness;
  - cams: the camera-height split (`camera_height_split`);
  - band: the bottom-band floor fraction of each WARPED frame (`band_floor_fractions`):
    the share of the SfM points it observed in its own bottom band that lie on the plane,
    median and p10, plus the median shares above and below it;
  - area: opaque fraction and area (also in H^2), the raster side and cell in H, projected /
    selected / admissible / warped frames.

All distances are in units of H, the median camera-to-plane height: the solve is unitless, and
components of one walk have different placement scales (walk 8 e2e6: H = 2.9, 12.1, 10.1).

THE NEW-SCENE FLAGS (`component_flags`; rules and why)

  multi_level_suspected -- two floor heights inside one component. Either
    (a) the camera heights split into two levels (`camera_height_split`): an Otsu split
        whose two clusters each hold >= 15% of >= 20 cameras, are separated by >= 4 pooled
        within-cluster SDs, and (when H is known) lie >= 0.2 H apart. A ONE-level walk cannot
        reach 4: an Otsu cut of a normal distribution gives 2.65 SDs, of a uniform one 3.46,
        so 4 needs a real empty band between the clusters. 0.2 H (~27 cm at a 1.35 m camera)
        is more than one ~17 cm stair riser and well above hand-held bobbing (walk 8 height
        CV 0.06-0.10). 15% excludes a crouch or a short detour;
        Then the FLOOR is checked (`floor_level_check`): if the frames' own bottom-band
        floor heights did not move with the cameras (less than half the camera gap), the
        walker changed carrying height on one floor, and the flag does not fire; with < 3
        downward frames on a side the camera evidence alone fires it, marked `unavailable`;
    (b) or the floor fit found a valid second plane BELOW the chosen floor with >= 0.5x its
        support (`ambiguous`, FLOOR_AMBIGUITY_RATIO). Nothing visible lies below a floor, so
        a supported layer there is a second level or a misplaced piece (walk 8 R0: the
        closet floor 0.6 H below the room). A runner-up ABOVE is not used: furniture tops
        (a bed, a desk) produce exactly that.
  low_floor_support -- the floor estimate is gray, fallback-only, or within a factor 2 of its
    rejection floor: the plane was rejected, or it is the camera-trajectory fallback, or its
    inliers < 2 x FLOOR_MIN_INLIERS (80) or its inlier spread < 2 x FLOOR_MIN_SUPPORT_CELLS
    (6). The factor 2 marks "one more sparse landing away from gray" (the audit: 40 was a
    guess; low-texture floors sit at that edge).
  furniture_dominated_bands -- the median warped frame's bottom band is mostly NOT floor:
    with >= 3 frames carrying >= 5 band points each, the median on-plane share is < 0.5 and
    the off-plane mass lies above the plane (median above > median below), i.e. furniture
    and walls, not pose error. 0.5 = "most of a typical frame's band evidence is not floor";
    the probe measured 0.03 (R0, smeared), 0.40 (G2 c0, patchwork) and 0.58 (G2 c45, the one
    readable plan). SfM under-samples carpet, so the share is a LOWER bound on floor.
  large_extent -- the fixed 512-cell raster and 64-view budget are spread too thin:
    the raster cell exceeds 0.02 H (side > 10.24 H; at H = 1.35 m that is a 2.7 cm cell,
    about the floor footprint of one 256 px sample pixel two camera heights away), or the
    64-view cap bound (more admissible frames than 64) while the padded square holds more
    than 1 H^2 per selected view (side > 8 H) -- more than one bottom band covers (about
    0.7 H^2 at 30 deg pitch on walk 8's 72 deg vertical field of view).
  collinear_trajectory -- the camera-centre plane is unusable because the path is a line
    (camera-plane spread < CAMERA_PLANE_MIN_SPREAD = 0.05): the floor's tilt about the path
    axis is then unconstrained by the walk, the reference falls back to roll-free / receipt
    up, and a long hallway or a partial live trajectory is the usual cause.

  Reflections are NOT flagged: nothing cheap in sparse SfM + RGB separates a mirror's or a
  glossy floor's features from real ones, so the flag is omitted rather than guessed.

The per-walk summary (`walk_summary`) evaluates the flags on the LAST landing that built
components (after the final solve's landing, the final world), lists the firing components
with their keyframe share, and counts how many built landings fired each flag.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from tower.world_builder.photo_floorplan import (
    CAMERA_PLANE_MIN_SPREAD,
    EPS,
    FLOOR_MIN_INLIERS,
    FLOOR_MIN_SUPPORT_CELLS,
    FLOOR_SUPPORT_GRID,
    FRAME_BELOW_PLANE_TOL_FRAC,
    FRAME_MIN_CHECK_POINTS,
    MAX_DIM,
    MIN_FRAME_PITCH_DEG,
    padded_square_bounds,
)

VERSION = 1
SUMMARY_FILENAME = "photo_floorplan_diagnostics.json"
RECORDS_FILENAME = "photo_floorplan_diagnostics.jsonl"
MAX_RECORD_BYTES = 8 * 1024
MAX_SUMMARY_BYTES = 32 * 1024
MAX_RECORDS = 512
"""Size bounds: one landing record <= 8 KiB (components beyond it are dropped, largest
first kept, and counted); at most 512 records per walk are kept or appended (a 5-minute walk
lands ~10-40 times); the summary <= 32 KiB."""

BAND_ON_TOL_FRAC = FRAME_BELOW_PLANE_TOL_FRAC
BAND_MIN_POINTS = FRAME_MIN_CHECK_POINTS
BAND_MIN_FRAMES = 3
FURNITURE_ON_FLOOR_MAX = 0.5
"""Band floor fraction: a band point is ON the plane within 0.12 H -- the per-frame pose
gate's own tolerance, so "on" here means what the pose gate means by it. A frame counts with
>= 5 band points (the pose gate's minimum); the flag needs >= 3 such frames."""

MULTILEVEL_MIN_CAMERAS = 20
MULTILEVEL_MIN_SHARE = 0.15
MULTILEVEL_MIN_SEPARATION = 4.0
MULTILEVEL_MIN_GAP_FRAC = 0.2
FLOOR_CHECK_MIN_FRAMES = 3
FLOOR_CHECK_PERCENTILE = 20.0
FLOOR_CHECK_MOVED_MIN = 0.5
LOW_SUPPORT_FACTOR = 2
LARGE_EXTENT_CELL_FRAC = 0.02
LARGE_EXTENT_VIEW_AREA_H2 = 1.0
MAX_SELECTED_VIEWS = 64
"""See the module docstring, "THE NEW-SCENE FLAGS". MAX_SELECTED_VIEWS mirrors the worker's
MAX_SELECTED (not imported: the worker imports this module)."""

FLAGS = ("multi_level_suspected", "low_floor_support", "furniture_dominated_bands",
         "large_extent", "collinear_trajectory")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _num(value, digits=3):
    """A finite float rounded for JSON, else None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, digits)


def _vec(value, digits=4):
    if value is None:
        return None
    try:
        out = [_num(v, digits) for v in value]
    except TypeError:
        return None
    return out if all(v is not None for v in out) else None


def _short(text, limit=96):
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= limit else text[:limit - 3] + "..."


# ---------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------


def camera_height_split(heights, scale=None, *, min_cameras=MULTILEVEL_MIN_CAMERAS,
                        min_share=MULTILEVEL_MIN_SHARE,
                        min_separation=MULTILEVEL_MIN_SEPARATION,
                        min_gap_frac=MULTILEVEL_MIN_GAP_FRAC):
    """The best two-level split of camera heights (Otsu: maximum between-cluster variance,
    each side >= `min_share` of the cameras). `scale` is H when known.

    Returns {n, cut, low_share, gap, gap_h, separation, bimodal}; `bimodal` is None when
    there are too few cameras, or no H, to judge."""
    h = np.asarray(heights, dtype=np.float64).reshape(-1)
    h = np.sort(h[np.isfinite(h)])
    n = len(h)
    out = {"n": n, "bimodal": None}
    if n < max(min_cameras, 2):
        return out
    k_min = max(1, int(math.ceil(min_share * n)))
    k_max = n - k_min
    if k_max < k_min:
        return out
    csum = np.cumsum(h)
    k = np.arange(k_min, k_max + 1)
    mu1 = csum[k - 1] / k
    mu2 = (csum[-1] - csum[k - 1]) / (n - k)
    between = k * (n - k) * (mu2 - mu1) ** 2
    best = int(k[int(np.argmax(between))])
    lo, hi = h[:best], h[best:]
    gap = float(hi.mean() - lo.mean())
    pooled = math.sqrt((float(((lo - lo.mean()) ** 2).sum()) + float(((hi - hi.mean()) ** 2).sum()))
                       / n)
    separation = gap / pooled if pooled > EPS * max(1.0, abs(gap)) else math.inf
    gap_h = gap / scale if scale is not None and scale > EPS else None
    # Without H the gap cannot be judged against a riser, and a noise-free path would pass
    # the separation test on any tiny step: not evaluable rather than guessed.
    bimodal = (None if gap_h is None else
               bool(separation >= min_separation and gap_h >= min_gap_frac))
    out.update(cut=_num(0.5 * (lo[-1] + hi[0]), 4), low_share=_num(best / n),
               gap=_num(gap, 4), gap_h=_num(gap_h),
               separation=(_num(separation, 2) if math.isfinite(separation) else 999.0),
               bimodal=bimodal)
    return out


def _downward_band_points(entries, inp, axis):
    """[(kid, own bottom-band SfM points)] of every DOWNWARD eligible keyframe (pitch >= the
    warp's MIN_FRAME_PITCH_DEG along `axis`), from the solve's observations when the build
    left its reader; else of the frames it loaded."""
    reader = inp.get("band_points")
    if reader is None:
        return list(inp.get("frames") or [])
    axis = np.asarray(axis, np.float64)
    out = []
    for e in entries:
        if e[3] is None:
            continue
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(np.asarray(e[3]) @ axis)))))
        if pitch < MIN_FRAME_PITCH_DEG:
            continue
        _, pts = reader(e[0])
        out.append((e[0], pts))
    return out


def floor_level_check(entries, frames, axis, offset, split, *,
                      min_points=BAND_MIN_POINTS, min_frames=FLOOR_CHECK_MIN_FRAMES,
                      floor_percentile=FLOOR_CHECK_PERCENTILE,
                      moved_min=FLOOR_CHECK_MOVED_MIN):
    """Did the FLOOR move with the cameras? For each downward frame with >= `min_points` own
    bottom-band points, its floor height is the low (20th-percentile) tail of those points
    (furniture lies above a floor; the fallback plane reads the same tail). Per camera
    cluster (split at `split['cut']`) the median floor height; `moved` = (upper floor -
    lower floor) / camera gap. A walk on two levels at a constant carrying height moves the
    floor by the whole gap (1); a walker who only changed carrying height does not move it
    (0). `confirmed` at >= 0.5, `contradicted` below, `unavailable` with < 3 frames on a
    side."""
    pos = {e[0]: np.asarray(e[1], np.float64) for e in entries}
    axis = np.asarray(axis, np.float64)
    lower, upper = [], []
    for kid, pts in frames:
        if pts is None or kid not in pos:
            continue
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        p = p[np.all(np.isfinite(p), axis=1)]
        if len(p) < min_points:
            continue
        floor_h = float(np.percentile(p @ axis + offset, floor_percentile))
        (upper if float(pos[kid] @ axis + offset) >= split["cut"] else lower).append(floor_h)
    out = {"frames_lower": len(lower), "frames_upper": len(upper)}
    if len(lower) < min_frames or len(upper) < min_frames or not split.get("gap"):
        return dict(out, result="unavailable")
    moved = (float(np.median(upper)) - float(np.median(lower))) / float(split["gap"])
    return dict(out, moved=_num(moved),
                result="confirmed" if moved >= moved_min else "contradicted")


def band_floor_fractions(frames, normal, offset, h_med, *, tol_frac=BAND_ON_TOL_FRAC,
                         min_points=BAND_MIN_POINTS):
    """Per frame with >= `min_points` band points: (on, above, below) shares of its own
    bottom-band SfM points relative to the plane (within / above / below `tol_frac` H)."""
    n = np.asarray(normal, dtype=np.float64)
    tol = tol_frac * float(h_med)
    rows = []
    for _kid, pts in frames:
        if pts is None:
            continue
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        p = p[np.all(np.isfinite(p), axis=1)]
        if len(p) < min_points:
            continue
        r = p @ n + float(offset)
        rows.append((float(np.mean(np.abs(r) <= tol)), float(np.mean(r > tol)),
                     float(np.mean(r < -tol))))
    return rows


def occupied_cells(positions, forward, up, bounds_xy, grid=FLOOR_SUPPORT_GRID):
    """Cells of the floor fit's 8x8 support grid (padded square bounds) holding a camera."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    pos = pos[np.all(np.isfinite(pos), axis=1)]
    if not len(pos):
        return 0
    fwd = np.asarray(forward, dtype=np.float64)
    fwd = fwd / max(float(np.linalg.norm(fwd)), EPS)
    upv = np.asarray(up, dtype=np.float64)
    upv = upv / max(float(np.linalg.norm(upv)), EPS)
    right = np.cross(fwd, upv)
    right = right / max(float(np.linalg.norm(right)), EPS)
    min_x, min_y, max_x, max_y = padded_square_bounds(bounds_xy)
    span = max(max_x - min_x, EPS)
    gx = np.clip(((pos @ fwd - min_x) / span * grid).astype(int), 0, grid - 1)
    gy = np.clip(((pos @ right - min_y) / span * grid).astype(int), 0, grid - 1)
    return int(len(np.unique(gy * grid + gx)))


def _roll_free_axis(entries, up):
    from tower.world_builder.photo_floorplan import _roll_free_reference

    rights = [np.cross(np.asarray(e[3], np.float64), np.asarray(e[2], np.float64))
              for e in entries if e[2] is not None and e[3] is not None]
    if len(rights) < 3:
        return None
    axis, _ = _roll_free_reference(np.stack(rights), np.asarray(up, np.float64))
    return axis


# ---------------------------------------------------------------------------
# one component, one landing
# ---------------------------------------------------------------------------


def component_record(cstats):
    """The diagnostics of one built component, from the stats the landing already made
    (`cstats`) and, when the homography build left them, its `_diag_inputs`."""
    inp = cstats.get("_diag_inputs") or {}
    plane = inp.get("plane")
    pstats = dict((plane.stats if plane is not None else cstats.get("plane")) or {})
    ok = bool(plane.ok) if plane is not None else bool(cstats.get("plane_ok"))
    source = (pstats.get("source") or "estimated") if ok else "rejected"
    h_med = _num(pstats.get("camera_height_median"), 6)
    if h_med is not None and h_med <= 0:
        h_med = None
    normal = (plane.normal if (plane is not None and plane.ok) else
              pstats.get("reference_normal_xyz"))
    rec = {"ref": cstats.get("reference_segment"), "eligible": cstats.get("eligible"),
           "points": cstats.get("points_quality")}
    rec["plane"] = {
        "ok": ok, "source": source,
        "reason": _short(cstats.get("plane_reason") or (plane.reason if plane else None)),
        "reuse": _short(cstats.get("plane_reuse"), 40),
        "reference": pstats.get("reference_normal"),
        "normal": _vec(normal),
        "tilt_ref_deg": _num(pstats.get("tilt_from_reference_deg"), 2),
        "tilt_up_deg": _num(pstats.get("tilt_from_up_deg"), 2),
        "inliers": pstats.get("inliers"), "candidates": pstats.get("candidate_points"),
        "support_cells": pstats.get("support_cells"),
        "camera_cells": None,
        "alt_ratio": _num(pstats.get("alt_plane_support_ratio")),
        "alt_h": _num(pstats.get("alt_plane_height_above_chosen_frac")),
        "ambiguous": pstats.get("ambiguous"),
        "H": _num(h_med, 4), "cam_cv": _num(pstats.get("camera_height_cv")),
        "cam_spread": _num(pstats.get("camera_plane_spread"), 4),
        "cam_flat": _num(pstats.get("camera_plane_flatness"), 4),
    }

    entries = inp.get("entries") or []
    rc = inp.get("receipt_component")
    cams = None
    if entries:
        cams = np.stack([np.asarray(e[1], np.float64) for e in entries])
        if rc is not None:
            rec["plane"]["camera_cells"] = occupied_cells(cams, rc["forward_xyz"], rc["up_xyz"],
                                                          rc["bounds_xy"])
    # -- camera heights: above the plane when there is one; else along the roll-free
    # vertical when usable, else along the floor fit's reference normal.
    split = {"n": 0, "bimodal": None}
    axis_name = None
    if cams is not None:
        axis, offset0 = None, 0.0
        if ok and plane is not None and plane.normal is not None:
            axis, offset0 = np.asarray(plane.normal, np.float64), float(plane.offset)
            axis_name = "plane"
        else:
            axis = _roll_free_axis(entries, rc["up_xyz"]) if rc is not None else None
            axis_name = "roll_free"
            if axis is None and pstats.get("reference_normal_xyz") is not None:
                axis = np.asarray(pstats["reference_normal_xyz"], np.float64)
                axis_name = pstats.get("reference_normal") or "reference"
        if axis is not None:
            split = camera_height_split(cams @ axis + offset0, h_med)
            if split.get("bimodal"):
                split["floor_check"] = floor_level_check(
                    entries, _downward_band_points(entries, inp, axis), axis, offset0, split)
    rec["cams"] = dict(split, axis=axis_name)

    # -- bottom-band floor fraction of the WARPED frames.
    band = {"frames": 0}
    frames = inp.get("frames") or []
    warped = inp.get("warped_kids")
    if warped is not None:
        warped = set(warped)
        frames = [f for f in frames if f[0] in warped]
    if ok and plane is not None and h_med is not None and frames:
        rows = band_floor_fractions(frames, plane.normal, plane.offset, h_med)
        band["frames"] = len(rows)
        band["of"] = len(frames)
        if rows:
            on = np.array([r[0] for r in rows])
            band.update(on_median=_num(np.median(on)), on_p10=_num(np.percentile(on, 10)),
                        above_median=_num(np.median([r[1] for r in rows])),
                        below_median=_num(np.median([r[2] for r in rows])))
    rec["band"] = band

    # -- area and resolution.
    area = {"opaque": _num(cstats.get("opaque_fraction")),
            "opaque_area": _num(cstats.get("opaque_area"), 3),
            "projected": cstats.get("projected_keyframes"),
            "selected": cstats.get("selected_frames"),
            "admissible": cstats.get("admissible_frames"),
            "warped": cstats.get("frames_warped")}
    bounds = inp.get("raster_bounds_xy")
    width = inp.get("raster_width") or MAX_DIM
    if bounds is not None:
        side = float(bounds[2]) - float(bounds[0])
        area["side"] = _num(side, 4)
        if h_med is not None:
            area["side_H"] = _num(side / h_med, 3)
            area["cell_H"] = _num(side / float(width) / h_med, 5)
            if area["opaque_area"] is not None:
                area["opaque_area_H2"] = _num(float(cstats["opaque_area"]) / h_med ** 2, 3)
    rec["area"] = area
    rec["flags"] = component_flags(rec)
    return rec


def component_flags(rec):
    """{flag: evidence} for the flags this component fires (module docstring)."""
    out = {}
    plane, cams, band, area = rec.get("plane", {}), rec.get("cams", {}), rec.get("band", {}), \
        rec.get("area", {})
    # multi_level_suspected
    floor_check = (cams.get("floor_check") or {}).get("result")
    if cams.get("bimodal") and floor_check != "contradicted":
        out["multi_level_suspected"] = (
            f"camera heights split in two: gap {cams.get('gap_h')} H, separation "
            f"{cams.get('separation')} SD, lower share {cams.get('low_share')}; floor "
            f"{floor_check or 'unavailable'}")
    elif (plane.get("ambiguous") and plane.get("alt_h") is not None
          and plane["alt_h"] < 0):
        out["multi_level_suspected"] = (
            f"a second valid floor {abs(plane['alt_h'])} H BELOW the chosen one with "
            f"{plane.get('alt_ratio')}x its support")
    # low_floor_support
    why = []
    if not plane.get("ok"):
        why.append(f"plane rejected ({plane.get('reason')})")
    elif plane.get("source") == "fallback_camera_drop":
        why.append("camera-trajectory fallback plane (no estimated floor)")
    inl, cells = plane.get("inliers"), plane.get("support_cells")
    if plane.get("ok") and plane.get("source") != "fallback_camera_drop":
        if inl is not None and inl < LOW_SUPPORT_FACTOR * FLOOR_MIN_INLIERS:
            why.append(f"{inl} inliers < {LOW_SUPPORT_FACTOR * FLOOR_MIN_INLIERS}")
        if cells is not None and cells < LOW_SUPPORT_FACTOR * FLOOR_MIN_SUPPORT_CELLS:
            why.append(f"inliers in {cells} cells < {LOW_SUPPORT_FACTOR * FLOOR_MIN_SUPPORT_CELLS}")
    if why:
        out["low_floor_support"] = "; ".join(why)
    # furniture_dominated_bands
    if (band.get("frames", 0) >= BAND_MIN_FRAMES and band.get("on_median") is not None
            and band["on_median"] < FURNITURE_ON_FLOOR_MAX
            and (band.get("above_median") or 0.0) > (band.get("below_median") or 0.0)):
        out["furniture_dominated_bands"] = (
            f"median band on-floor {band['on_median']} (p10 {band.get('on_p10')}), above "
            f"{band.get('above_median')}, over {band['frames']} warped frames")
    # large_extent
    side_h = area.get("side_H")
    if side_h is not None:
        resolution = side_h > LARGE_EXTENT_CELL_FRAC * MAX_DIM
        selected, admissible = area.get("selected") or 0, area.get("admissible") or 0
        budget = (selected >= MAX_SELECTED_VIEWS and admissible > MAX_SELECTED_VIEWS
                  and side_h ** 2 / MAX_SELECTED_VIEWS > LARGE_EXTENT_VIEW_AREA_H2)
        if resolution or budget:
            out["large_extent"] = (
                f"raster side {side_h} H, cell {area.get('cell_H')} H"
                + (f"; view cap bound ({admissible} admissible > {MAX_SELECTED_VIEWS}), "
                   f"{round(side_h ** 2 / MAX_SELECTED_VIEWS, 2)} H^2 per view" if budget else ""))
    # collinear_trajectory
    spread = plane.get("cam_spread")
    if spread is not None and spread < CAMERA_PLANE_MIN_SPREAD:
        out["collinear_trajectory"] = (
            f"camera-plane spread {spread} < {CAMERA_PLANE_MIN_SPREAD}: reference "
            f"{plane.get('reference')}")
    return out


def _dumps(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False, default=str)


def _disposition_class(disposition):
    d = str(disposition or "")
    if d == "ok":
        return "ok"
    if d.startswith("failed"):
        return "failed"
    for key, label in (("regressed", "regressed"), ("wall cap", "wall_cap"),
                       ("superseded", "superseded"), ("cancelled", "cancelled"),
                       ("not newer", "not_newer")):
        if key in d:
            return label
    return "skipped_other"


def landing_record(*, world_id, session_id, receipt, disposition, published, escaped,
                   elapsed_ms, stats, now):
    """One bounded record for one landing (any disposition)."""
    receipt = receipt or {}
    solved_at = _num(receipt.get("solved_at"), 3)
    computed_at = _num(receipt.get("computed_at"), 3)
    rec = {"v": VERSION, "world": world_id, "session": session_id, "solved_at": solved_at,
           "geometry_revision": receipt.get("geometry_revision"),
           "frame_revision": receipt.get("frame_revision"),
           "horizon_keyframes": receipt.get("horizon_keyframes"),
           "keyframes_pending": receipt.get("keyframes_pending"),
           "disposition": _short(disposition, 160), "class": _disposition_class(disposition),
           "published": bool(published), "escaped": bool(escaped),
           "elapsed_ms": _num(elapsed_ms, 1),
           "landing_lag_s": _num(now - solved_at, 2) if solved_at is not None else None,
           "coverage_lag_s": (_num(computed_at - solved_at, 2)
                              if solved_at is not None and computed_at is not None else None),
           "stations_omitted": receipt.get("stations_omitted"),
           "components_omitted": receipt.get("components_omitted"),
           "components_total": receipt.get("components_total"),
           "components": []}
    comps = []
    for c in stats or ():
        if not isinstance(c, dict):
            continue
        try:
            comps.append(component_record(c))
        except Exception as exc:  # noqa: BLE001 -- a diagnostic never fails a landing
            comps.append({"ref": c.get("reference_segment"), "error": _short(repr(exc))})
    comps.sort(key=lambda r: -(r.get("eligible") or 0))
    total = sum((r.get("eligible") or 0) for r in comps) or 1
    for r in comps:
        r["share"] = _num((r.get("eligible") or 0) / total)
    rec["components"] = comps
    dropped = 0
    while len(_dumps(rec).encode("utf-8")) > MAX_RECORD_BYTES and rec["components"]:
        rec["components"] = rec["components"][:-1]
        dropped += 1
    if dropped:
        rec["components_dropped"] = dropped
    return rec


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------


def _stats(values):
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    return {"median": _num(np.median(v), 2), "max": _num(max(v), 2), "last": _num(v[-1], 2)}


def walk_summary(records, *, world_id=None, session_id=None, generated_at=None):
    """The per-walk aggregates and new-scene flags from this walk's landing records."""
    recs = sorted((r for r in records if isinstance(r, dict)),
                  key=lambda r: (r.get("solved_at") is None, r.get("solved_at") or 0.0))
    classes = {}
    for r in recs:
        classes[r.get("class")] = classes.get(r.get("class"), 0) + 1
    built = [r for r in recs if r.get("components")]
    solved = [r["solved_at"] for r in recs if r.get("solved_at") is not None]
    gaps = [b - a for a, b in zip(solved, solved[1:])]
    out = {
        "version": VERSION, "kind": "photo_floorplan_diagnostics",
        "world": world_id or (recs[-1].get("world") if recs else None),
        "session": session_id or (recs[-1].get("session") if recs else None),
        "generated_at": _num(generated_at, 3),
        "landings": {"total": len(recs), "built": len(built),
                     "published": sum(1 for r in recs if r.get("published")),
                     "escaped": sum(1 for r in recs if r.get("escaped")),
                     "by_class": classes},
        "landing_lag_s": _stats([r.get("landing_lag_s") for r in recs]),
        "coverage_lag_s": _stats([r.get("coverage_lag_s") for r in recs]),
        "landing_gap_s": _stats(gaps),
        "elapsed_ms": _stats([r.get("elapsed_ms") for r in recs]),
        "stations_omitted": _stats([r.get("stations_omitted") for r in recs]),
        "components_omitted": _stats([r.get("components_omitted") for r in recs]),
        "plane_sources": {},
        "final_landing": None,
        "flags": {},
        "flags_fired": [],
        "not_evaluated": {"reflections_suspected": "omitted: not cheaply detectable from sparse "
                                                   "SfM points and RGB"},
    }
    for r in built:
        for c in r["components"]:
            src = (c.get("plane") or {}).get("source")
            per = out["plane_sources"].setdefault(str(c.get("ref")), {})
            per[src] = per.get(src, 0) + 1
    if built:
        last = built[-1]
        out["final_landing"] = {
            "solved_at": last.get("solved_at"), "disposition": last.get("disposition"),
            "published": last.get("published"), "keyframes_pending": last.get("keyframes_pending"),
            "covers_all_accepted": last.get("keyframes_pending") == 0,
            "components": last["components"]}
        for key in ("source", "live_components"):     # offline rebuilds say what they are
            if key in last:
                out["final_landing"][key] = last[key]
        for flag in FLAGS:
            firing = [{"ref": c.get("ref"), "share": c.get("share"),
                       "evidence": (c.get("flags") or {}).get(flag)}
                      for c in last["components"] if flag in (c.get("flags") or {})]
            out["flags"][flag] = {
                "fired": bool(firing),
                "main_component": bool(firing) and firing[0]["ref"] == last["components"][0].get("ref"),
                "components": firing,
                "landings_fired": sum(1 for r in built
                                      if any(flag in (c.get("flags") or {}) for c in r["components"])),
                "landings_built": len(built)}
            if firing:
                out["flags_fired"].append(flag)
    while len(_dumps(out).encode("utf-8")) > MAX_SUMMARY_BYTES and out["final_landing"] \
            and out["final_landing"]["components"]:
        out["final_landing"]["components"] = out["final_landing"]["components"][:-1]
        out["final_landing"]["components_dropped"] = \
            out["final_landing"].get("components_dropped", 0) + 1
    return out


# ---------------------------------------------------------------------------
# files (only with TOWER_WORLD_PHOTO_FLOORPLAN_DIAGNOSTICS=on)
# ---------------------------------------------------------------------------


def session_dir(root, world_id, session_id):
    return Path(root) / "worlds" / str(world_id) / "sessions" / str(session_id)


def write_files(root, world_id, session_id, record, summary):
    """Append `record` to the JSONL and replace the summary, inside an EXISTING session
    directory only (never creates one: a purged or foreign world gets nothing). Returns
    True if written."""
    directory = session_dir(root, world_id, session_id)
    if not directory.is_dir():
        return False
    with open(directory / RECORDS_FILENAME, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(_dumps(record) + "\n")
    target = directory / SUMMARY_FILENAME
    tmp = directory / (SUMMARY_FILENAME + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(summary, indent=1, allow_nan=False, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return True


def read_records(path):
    """The records of a diagnostics JSONL (offline use); malformed lines are skipped."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out
