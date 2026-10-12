"""FOW v1.2 slice 1 replay PROBE (Tower side), run OFFLINE on a finished replay.

Design: FOW-V12-PHOTO-FLOORPLAN-DESIGN-20261012.md, "Raster and evidence".
Wire:   303-v12-wire-contract.md.
Core:   tower.world_builder.photo_floorplan (pure; this script is the only
        place that touches disk, the depth-fit, and the coverage internals).

For each distinct `guidance.coverage` landing recorded in a replay run's
`status-pushes.jsonl`, and for each of that landing's displayed registered
components, this script:

  1. takes the landing's own receipt basis and (unpadded) bounds verbatim;
  2. reconstructs that landing's eligible, posed keyframes by re-running
     `guidance_coverage.compute_coverage` against the replay's
     `tree-snapshots/<geometry_revision>/` copy of the five files it reads,
     and recovers each keyframe's placed world position/up/forward by
     calling `guidance_coverage._placed` on the SAME (row, placement) pairs
     coverage used -- i.e. REUSING coverage's own geometry, not re-deriving
     it;
  3. intersects that set with the session's live-depth cache (`align.json`)
     by keyframe id;
  4. selects <= 64 spatially distributed cache-hit keyframes;
  5. fits each selected keyframe's cached MoGe prediction to THIS solve's
     own SfM observations (`dense.align_frame`), accepting only a positive,
     finite fit with enough clean (non-fill) anchors and an acceptable
     held-out residual;
  6. hands the fitted, placed, cache-image samples to
     `photo_floorplan.build_raster` and writes the RGBA PNG plus a gray-plate
     composite preview.

`--method homography` (fallback (3), no depth cache, no GPU) instead, for the
FINAL landing only (it needs the solve's 3-D points, kept only for the final
solve):

  a. the same eligible, placed keyframes (step 2 above);
  b. this component's solve points, mapped into the output frame by the same
     Sim3 algebra as `guidance_coverage._solution_relative`, quality-filtered
     (track length >= 3, reprojection error <= 2 px);
  c. `photo_floorplan.estimate_floor_plane` (lower envelope + camera check);
  d. downward-looking frames, <= 64 spread over where their bottom band meets
     the floor, read from the solve's own undistorted `solve/<s>/images`
     with `camera.json` intrinsics, area-resampled to <= 256x144;
  e. `photo_floorplan.build_homography_raster`, then `encode_tile`
     (PNG, else WebP with lossless alpha, else 384 px; <= 180 KiB or omitted).

KNOWN DATA-AVAILABILITY GAP (see PROBE-REPORT.md "open problems" for the
full discussion): step 5 needs this solve's own sparse 3-D observations
(`solution.npz`), which the replay harness only preserved for the FINAL
solve -- the five intermediate `solution-snapshots/*.json` are metadata
only. Earlier landings therefore get real cache-hit/selection numbers but
no fit/raster; this is reported, not hidden or faked.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np  # noqa: E402

from tower.world_builder import guidance_coverage as gc  # noqa: E402
from tower.world_builder.dense import depth_from_prediction  # noqa: E402
from tower.world_builder.photo_floorplan import (  # noqa: E402
    KeyframeSample,
    build_homography_raster,
    build_raster,
    encode_tile,
)
# The landing inputs live with the live side worker; the probe REUSES them (one copy).
from tower.world_builder.photo_floorplan_worker import (  # noqa: E402,F401
    MAX_HELD_OUT_REL,
    MAX_SELECTED,
    MIN_FIT_POINTS,
    TARGET_PIXEL_BUDGET,
    DepthCache,
    SolveArrays,
    fit_keyframe,
    homography_component,
    load_frame_image,
    placed_entries,
    resize_sample,
    select_spatial,
    solve_arrays_from,
    solve_to_output,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("fow_v12_probe")

GRAY_PLATE = (0x70, 0x75, 0x7B)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# landings
# ---------------------------------------------------------------------------


def extract_landings(run_dir: Path):
    """Distinct (solved_at, geometry_revision, frame_revision) coverage
    receipts, in the order they were first pushed, each with its own
    `received_at` (the walk-clock "landing time" for the report/caption)."""
    landings = []
    seen_keys = set()
    path = run_dir / "status-pushes.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        push = json.loads(line)
        payload = push.get("envelope", {}).get("payload", {})
        guidance = payload.get("guidance")
        coverage = guidance.get("coverage") if guidance else None
        if not coverage:
            continue
        key = (coverage["solved_at"], coverage["geometry_revision"], coverage["frame_revision"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        landings.append({"key": key, "receipt": coverage,
                         "received_at": push.get("received_at")})
    return landings


def tree_snapshot_dir(run_dir: Path, geometry_revision: str) -> Path:
    return run_dir / "tree-snapshots" / geometry_revision


def load_tree_snapshot(run_dir: Path, geometry_revision: str):
    """The five files `compute_coverage` reads, as they were at this
    landing's geometry revision. Returns None if the snapshot is missing or
    incomplete (a failed/unverified capture)."""
    d = tree_snapshot_dir(run_dir, geometry_revision)
    names = ("solution.json", "manifest.json", "placements.json", "poses.json", "keyframes.jsonl")
    if not all((d / n).is_file() for n in names):
        return None
    try:
        solution = load_json(d / "solution.json")
        manifest = load_json(d / "manifest.json")
        placements = load_json(d / "placements.json")["placements"]
        poses = load_json(d / "poses.json")["poses"]
        keyframes = load_jsonl(d / "keyframes.jsonl")
    except (OSError, ValueError, KeyError):
        return None
    return dict(solution=solution, manifest=manifest, placements=placements,
               poses=poses, keyframes=keyframes)


def eligible_placed_keyframes(tree):
    """Re-run coverage's own eligibility+placement math (reused, not
    re-derived) and return {reference_segment: [(kid, position, up_world,
    forward_world, scale), ...]}."""
    captured = []
    block = gc.compute_coverage(
        solution=tree["solution"], manifest=tree["manifest"],
        placements=tree["placements"], poses=tree["poses"],
        keyframes=tree["keyframes"],
        geometry_revision=_revision_of(tree),
        computed_at=float(tree["manifest"]["global_solve"]["solved_at"]) + 0.01,
        eligible_out=captured)
    return block, placed_entries(tree, captured)


def _revision_of(tree):
    from tower.results.world_builder import geometry_revision_from_manifest
    return geometry_revision_from_manifest(tree["manifest"])


# ---------------------------------------------------------------------------
# SfM arrays (the live-depth cache, spatial selection and the SfM depth fit are
# imported from tower.world_builder.photo_floorplan_worker above)
# ---------------------------------------------------------------------------


def load_solve_arrays(solve_session_dir: Path) -> "SolveArrays | None":
    meta_path = solve_session_dir / "solution.json"
    npz_path = solve_session_dir / "solution.npz"
    if not meta_path.is_file() or not npz_path.is_file():
        return None
    meta = load_json(meta_path)
    with np.load(npz_path) as arrays:
        return solve_arrays_from(meta, {k: np.asarray(arrays[k]) for k in
                                        ("xyz", "observations", "observation_xy")})


# ---------------------------------------------------------------------------
# main per-landing/per-component pipeline
# ---------------------------------------------------------------------------


def process_component(*, component_receipt, cache: DepthCache, placed_entries,
                      solve_arrays, order_base, png_path, preview_path,
                      landing_label, is_final):
    t0 = time.perf_counter()
    ref = component_receipt["reference_segment"]
    forward = np.asarray(component_receipt["forward_xyz"], np.float64)
    up = np.asarray(component_receipt["up_xyz"], np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    bounds_xy = component_receipt["bounds_xy"]
    origin = component_receipt["origin_xyz"]

    eligible = placed_entries
    cache_hit_entries = [e for e in eligible if cache.has(e[0])]
    cache_hit_fraction = (len(cache_hit_entries) / len(eligible)) if eligible else 0.0
    selected = select_spatial(cache_hit_entries, forward, right)

    stats = dict(reference_segment=ref, eligible=len(eligible),
                cache_hits=len(cache_hit_entries), cache_hit_fraction=cache_hit_fraction,
                selected=len(selected), landing=landing_label)

    if solve_arrays is None:
        stats.update(fit_attempted=0, fit_success=0, fit_success_fraction=None,
                    opaque_fraction=None, cpu_ms=None, png_bytes=None,
                    reason="no SfM arrays for this landing (see open problems)")
        return stats, None

    samples = []
    fit_reasons = {}
    for i, (kid, position, up_world, forward_world, scale) in enumerate(selected):
        record = cache.by_kid[kid]
        pred, fill, rgb_full = cache.load(kid)
        ok, a, b, n_clean, ho, reason = fit_keyframe(kid, solve_arrays, pred, fill)
        fit_reasons[kid] = reason
        if not ok:
            continue
        full_depth = depth_from_prediction(pred, a, b, kind="depth")
        full_valid = np.isfinite(full_depth) & (full_depth > 1e-6) & ~fill
        cam = solve_arrays.camera
        depth_r, valid_r, color_r, fx, fy, cx, cy = resize_sample(
            full_depth, full_valid, rgb_full, cam["fx"], cam["fy"], cam["cx"], cam["cy"])
        samples.append(KeyframeSample(
            kid=kid, position_xyz=tuple(position), forward_xyz=tuple(forward_world),
            up_xyz=tuple(up_world), scale=scale, fx=fx, fy=fy, cx=cx, cy=cy,
            depth=depth_r, valid=valid_r, color=color_r, order=order_base + i))

    fit_attempted = len(selected)
    fit_success = len(samples)
    result = build_raster(origin_xyz=origin, up_xyz=up, forward_xyz=forward,
                          bounds_xy=bounds_xy, keyframes=samples)
    cpu_ms = (time.perf_counter() - t0) * 1000.0

    from PIL import Image
    img = Image.fromarray(result.rgba, mode="RGBA")
    img.save(png_path, format="PNG")
    png_bytes = png_path.stat().st_size

    _write_composite_preview(result, preview_path, component_ref=ref, label=landing_label)

    stats.update(fit_attempted=fit_attempted, fit_success=fit_success,
                fit_success_fraction=(fit_success / fit_attempted) if fit_attempted else None,
                opaque_fraction=result.opaque_fraction, cpu_ms=round(cpu_ms, 1),
                png_bytes=png_bytes, multi_view_cells=result.multi_view_cells,
                single_view_cells=result.single_view_cells,
                raster_bounds_xy=result.raster_bounds_xy,
                fit_failure_reasons={r: sum(1 for v in fit_reasons.values() if v == r)
                                     for r in set(fit_reasons.values()) if r != "ok"})
    return stats, result


def _write_composite_preview(result, path, *, component_ref, label):
    from PIL import Image, ImageDraw

    plate = Image.new("RGB", (result.width, result.height), GRAY_PLATE)
    photo = Image.fromarray(result.rgba, mode="RGBA")
    plate.paste(photo, (0, 0), photo)
    canvas = Image.new("RGB", (result.width, result.height + 24), GRAY_PLATE)
    canvas.paste(plate, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, result.height + 4), f"component {component_ref} - {label}", fill=(255, 255, 255))
    canvas.save(path, format="PNG")


# ---------------------------------------------------------------------------
# --method homography (fallback (3))
# ---------------------------------------------------------------------------

def process_component_homography(*, component_receipt, placed_entries, tree, solve_dir,
                                  npz, keyframe_index, camera, out_prefix, landing_label,
                                  run_label, furniture_mask=False):
    t_wall = time.perf_counter()
    t_cpu = time.process_time()
    ref = component_receipt["reference_segment"]
    up = np.asarray(component_receipt["up_xyz"], np.float64)
    forward = np.asarray(component_receipt["forward_xyz"], np.float64)
    stats = dict(reference_segment=ref, landing=landing_label, method="homography")
    if not placed_entries:
        stats["reason"] = "no eligible keyframes"
        return stats, None
    # (b)-(e): the SAME code the live side worker runs.
    result, plane, frames, core = homography_component(
        receipt_component=component_receipt, entries=placed_entries, tree=tree, arrays=npz,
        keyframe_index=keyframe_index, camera=camera, images_dir=solve_dir / "images",
        furniture_mask=furniture_mask)
    stats.update(core)
    cams = np.stack([e[1] for e in placed_entries])
    t_enc = time.perf_counter()
    tile = encode_tile(result.rgba)
    encode_ms = (time.perf_counter() - t_enc) * 1000.0
    cpu_ms = (time.process_time() - t_cpu) * 1000.0
    wall_ms = (time.perf_counter() - t_wall) * 1000.0

    from PIL import Image
    Image.fromarray(result.rgba, mode="RGBA").save(f"{out_prefix}_raster.png", format="PNG")
    tile_path = None
    if tile is not None:
        ext = "png" if tile["mime"] == "image/png" else "webp"
        tile_path = Path(f"{out_prefix}.{ext}")
        tile_path.write_bytes(tile["data"])
    _write_tile_preview(tile, result, Path(f"{out_prefix}_preview.png"),
                        caption=f"{run_label} component {ref} - homography"
                                f"{'' if furniture_mask else ' (no furniture mask)'} - {landing_label}")

    stats.update(
        frames_warped=result.stats.get("frames_warped", 0),
        frames_rejected=result.stats.get("frames_rejected", {}),
        cells_rejected=result.stats.get("cells_rejected", {}),
        furniture_mask=result.stats.get("furniture_mask"),
        furniture_masked_blocks=result.stats.get("furniture_masked_blocks"),
        raster_reason=result.stats.get("reason"),
        projected_keyframes=len(result.contributing_keyframes),
        opaque_fraction=result.opaque_fraction,
        multi_view_cells=result.multi_view_cells, single_view_cells=result.single_view_cells,
        raster_bounds_xy=result.raster_bounds_xy,
        band_pixels_hitting_plane=result.stats.get("band_pixels_hitting_plane"),
        band_pixels_outside_bounds=result.stats.get("band_pixels_outside_bounds"),
        tile_path=str(tile_path) if tile_path else None,
        tile_bytes=tile["bytes"] if tile else None,
        tile_encoding=tile["encoding"] if tile else "omitted (> 180 KiB)",
        tile_size=[tile["width"], tile["height"]] if tile else None,
        encode_ms=round(encode_ms, 1),
        cpu_ms=round(cpu_ms, 1), wall_ms=round(wall_ms, 1),
        imagery_source="unredacted")

    # Diagnostic only: the runner-up plane, if the estimate was ambiguous.
    if plane.ok and plane.stats.get("ambiguous"):
        from tower.world_builder.photo_floorplan import FloorPlane
        alt_stats = dict(plane.stats)
        alt_n = np.asarray(plane.stats["alt_plane_normal"])
        alt_d = float(plane.stats["alt_plane_offset"])
        alt_stats["camera_height_median"] = float(np.median(cams @ alt_n + alt_d))
        alt = FloorPlane(ok=True, reason="ok", normal=tuple(alt_n), offset=alt_d, stats=alt_stats)
        alt_result = build_homography_raster(origin_xyz=component_receipt["origin_xyz"],
                                             up_xyz=up, forward_xyz=forward,
                                             bounds_xy=component_receipt["bounds_xy"],
                                             plane=alt, frames=frames,
                                             furniture_mask=furniture_mask)
        _write_tile_preview(None, alt_result, Path(f"{out_prefix}_ALTPLANE-diagnostic_preview.png"),
                            caption=f"DIAGNOSTIC runner-up plane, NOT the rule's choice - {run_label} c{ref}")
        stats["alt_plane_opaque_fraction"] = alt_result.opaque_fraction
    return stats, result


def _write_tile_preview(tile, result, path, *, caption):
    """The DECODED tile (what a client would show) over the #70757B plate."""
    import io

    from PIL import Image, ImageDraw

    if tile is not None:
        photo = Image.open(io.BytesIO(tile["data"])).convert("RGBA")
    else:
        photo = Image.fromarray(result.rgba, mode="RGBA")
    w, h = photo.size
    plate = Image.new("RGB", (w, h), GRAY_PLATE)
    plate.paste(photo, (0, 0), photo)
    canvas = Image.new("RGB", (w, h + 24), GRAY_PLATE)
    canvas.paste(plate, (0, 0))
    ImageDraw.Draw(canvas).text((4, h + 4), caption, fill=(255, 255, 255))
    canvas.save(path, format="PNG")


def run_homography(*, run_dir, world_dir, session_id, out_dir, run_label, furniture_mask=False):
    solve_dir = world_dir / "solve" / session_id
    camera = load_json(solve_dir / "camera.json")
    meta = load_json(solve_dir / "solution.json")
    keyframe_index = {kid: i for i, kid in enumerate(meta["keyframe_ids"])}
    with np.load(solve_dir / "solution.npz") as arrays:
        npz = {k: np.asarray(arrays[k]) for k in
               ("xyz", "component", "track_length", "error", "observations", "observation_xy")}
    npz["observations"] = npz["observations"].reshape(-1, 3)
    npz["observation_xy"] = npz["observation_xy"].reshape(-1, 2)
    # Warm the PIL/WebP/cv2 imports once, as a resident side worker would
    # have them, so the first component's timing is not an import benchmark.
    import cv2  # noqa: F401
    encode_tile(np.random.default_rng(0).integers(0, 256, (64, 64, 4), dtype=np.uint8),
                target_bytes=1, cap_bytes=1 << 20)
    landings = extract_landings(run_dir)
    rows = []
    for idx, landing in enumerate(landings):
        label = f"landing{idx}" + ("-final" if idx == len(landings) - 1 else "")
        solved_at, geometry_revision, _ = landing["key"]
        if idx != len(landings) - 1:
            rows.append(dict(landing=label, reference_segment=None, solved_at=solved_at,
                             reason="non-final landing: no solution.npz (3-D points) kept"))
            continue
        if float(meta["solved_at"]) != float(solved_at):
            raise SystemExit(f"final landing solved_at {solved_at} != solve dir {meta['solved_at']}")
        tree = load_tree_snapshot(run_dir, geometry_revision)
        if tree is None:
            rows.append(dict(landing=label, reference_segment=None,
                             reason="tree snapshot missing for the final landing"))
            continue
        _, by_ref = eligible_placed_keyframes(tree)
        for component_receipt in landing["receipt"]["components"]:
            ref = component_receipt["reference_segment"]
            prefix = out_dir / f"{run_label}_{label}_component{ref}_homography"
            stats, _ = process_component_homography(
                component_receipt=component_receipt, placed_entries=by_ref.get(ref, []),
                tree=tree, solve_dir=solve_dir, npz=npz, keyframe_index=keyframe_index,
                camera=camera, out_prefix=prefix, landing_label=label, run_label=run_label,
                furniture_mask=furniture_mask)
            stats["solved_at"] = solved_at
            stats["geometry_revision"] = geometry_revision
            rows.append(stats)
            log.info("%s %s component %s: plane=%s opaque=%.3f frames=%s cpu=%.0fms bytes=%s",
                     run_label, label, ref, stats.get("plane_reason"),
                     stats.get("opaque_fraction") or 0.0, stats.get("frames_warped"),
                     stats.get("cpu_ms") or 0.0, stats.get("tile_bytes"))
    (out_dir / f"{run_label}-homography-stats.json").write_text(
        json.dumps(rows, indent=1, default=float), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("depth", "homography"), default="depth")
    parser.add_argument("--furniture-mask", choices=("on", "off"), default="off",
                        help="homography: the opt-in (default off) bottom-band furniture mask")
    parser.add_argument("--run-label", default=None,
                        help="homography: output file prefix (default: the run dir's name)")
    parser.add_argument("--run-dir", default=str(Path(
        r"C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\experiments\W8LOOP\runs\g2")))
    parser.add_argument("--data-root", default=str(Path(
        r"C:\Users\tvllo\Projects\Glasses-scratch\wbcpt\w8-g2-d\world_builder")))
    parser.add_argument("--out-dir", default=str(Path(
        r"C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\experiments\W8LOOP\p2probe")))
    parser.add_argument("--world-id", default="467e6db5e9664cd1b3d9ed0a4f071911")
    parser.add_argument("--session-id", default="856b0df35403423a999fd64082aaa1e2")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    world_dir = data_root / "worlds" / args.world_id
    if args.method == "homography":
        run_homography(run_dir=run_dir, world_dir=world_dir, session_id=args.session_id,
                       out_dir=out_dir, run_label=args.run_label or run_dir.name,
                       furniture_mask=args.furniture_mask == "on")
        return
    dense_root = world_dir / "dense" / args.session_id
    solve_session_dir = world_dir / "solve" / args.session_id

    cache = DepthCache(dense_root)
    landings = extract_landings(run_dir)
    log.info("found %d distinct coverage landings", len(landings))

    final_key = landings[-1]["key"] if landings else None
    solve_arrays_final = load_solve_arrays(solve_session_dir)

    report_rows = []
    order_base = 0
    for idx, landing in enumerate(landings):
        solved_at, geometry_revision, frame_revision = landing["key"]
        is_final = landing["key"] == final_key
        label = f"landing{idx}" + ("-final" if is_final else "")
        tree = load_tree_snapshot(run_dir, geometry_revision)
        if tree is None:
            report_rows.append(dict(landing=label, reference_segment=None,
                                    reason="tree snapshot missing/incomplete for this geometry_revision"))
            log.warning("%s: tree snapshot missing for revision %s", label, geometry_revision)
            continue
        try:
            block, by_ref = eligible_placed_keyframes(tree)
        except Exception as exc:  # noqa: BLE001 -- a probe, never crash the whole run
            report_rows.append(dict(landing=label, reference_segment=None,
                                    reason=f"recompute_coverage failed: {exc}"))
            log.exception("%s: recompute_coverage failed", label)
            continue
        receipt = landing["receipt"]
        solve_arrays = solve_arrays_final if is_final else None
        for component_receipt in receipt["components"]:
            ref = component_receipt["reference_segment"]
            entries = by_ref.get(ref, [])
            png_path = out_dir / f"{label}_component{ref}.png"
            preview_path = out_dir / f"{label}_component{ref}_preview.png"
            stats, _ = process_component(
                component_receipt=component_receipt, cache=cache, placed_entries=entries,
                solve_arrays=solve_arrays, order_base=order_base,
                png_path=png_path, preview_path=preview_path,
                landing_label=label, is_final=is_final)
            order_base += 1
            stats["solved_at"] = solved_at
            stats["geometry_revision"] = geometry_revision
            stats["received_at"] = landing["received_at"]
            report_rows.append(stats)
            log.info("%s component %s: %s", label, ref, stats)

    write_report(out_dir, report_rows)


def write_report(out_dir: Path, rows):
    lines = ["# PROBE-REPORT: FOW v1.2 slice 1, walk 8 / G2 replay\n",
            "One row per (landing, displayed component). `fit_success_fraction` and "
            "`opaque_fraction` are None where no SfM arrays were available for that "
            "landing (see PROBE's module docstring / the open problems note below).\n",
            "| landing | solved_at | ref | eligible | cache_hit_frac | selected | "
            "fit_success_frac | opaque_frac | cpu_ms | png_bytes | multi/single cells |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("reference_segment") is None and "reason" in r and "eligible" not in r:
            lines.append(f"| {r['landing']} | - | - | - | - | - | - | - | - | - | {r['reason']} |")
            continue
        chf = r.get("cache_hit_fraction")
        fsf = r.get("fit_success_fraction")
        of = r.get("opaque_fraction")
        mv, sv = r.get("multi_view_cells"), r.get("single_view_cells")
        lines.append(
            f"| {r['landing']} | {r.get('solved_at')} | {r['reference_segment']} | "
            f"{r.get('eligible')} | {chf:.2f} | {r.get('selected')} | "
            f"{'%.2f' % fsf if fsf is not None else 'n/a'} | "
            f"{'%.3f' % of if of is not None else 'n/a'} | "
            f"{r.get('cpu_ms') if r.get('cpu_ms') is not None else 'n/a'} | "
            f"{r.get('png_bytes') if r.get('png_bytes') is not None else 'n/a'} | "
            f"{mv if mv is not None else '-'}/{sv if sv is not None else '-'} |")
    (out_dir / "PROBE-REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", out_dir / "PROBE-REPORT.md")


if __name__ == "__main__":
    main()
