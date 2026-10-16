"""FOW v1.2 slice 1 replay PROBE (Tower side), run OFFLINE on a finished replay.

Design: FOW-V12-PHOTO-FLOORPLAN-DESIGN-20261012.md, "Raster and evidence".
Wire:   303-v12-wire-contract.md.
Core:   tower.world_builder.photo_floorplan (pure; this script is the only
        place that touches disk and the coverage internals).

For the FINAL `guidance.coverage` landing recorded in a replay run's
`status-pushes.jsonl` (it needs the solve's 3-D points, kept only for the final
solve), and for each of that landing's displayed registered components, this
script runs fallback (3), the lower-image floor homography:

  a. the landing's own receipt basis and (unpadded) bounds, verbatim, and its
     eligible, posed keyframes, recovered by re-running
     `guidance_coverage.compute_coverage` against the replay's
     `tree-snapshots/<geometry_revision>/` copy of the five files it reads and
     placing each with coverage's own `_placed` (REUSED, not re-derived);
  b. this component's solve points, mapped into the output frame by the same
     Sim3 algebra as `guidance_coverage._solution_relative`, quality-filtered
     (track length >= 3, reprojection error <= 2 px);
  c. `photo_floorplan.estimate_floor_plane` (lower envelope + camera check);
  d. downward-looking frames, <= 64 spread over where their bottom band meets
     the floor, read from the solve's own undistorted `solve/<s>/images`
     with `camera.json` intrinsics, area-resampled to <= 256x144;
  e. `photo_floorplan.build_homography_raster`, then `encode_tile`
     (PNG, else WebP with lossless alpha, else 384 px; <= 180 KiB or omitted).
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
from tower.world_builder.photo_floorplan import (  # noqa: E402
    build_homography_raster,
    encode_tile,
)
# The landing inputs live with the live side worker; the probe REUSES them (one copy).
from tower.world_builder.photo_floorplan_worker import (  # noqa: E402,F401
    MAX_SELECTED,
    TARGET_PIXEL_BUDGET,
    homography_component,
    load_frame_image,
    placed_entries,
    select_spatial,
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
    # Only `homography` remains; the flag stays so existing command lines still parse.
    parser.add_argument("--method", choices=("homography",), default="homography")
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
    run_homography(run_dir=run_dir, world_dir=world_dir, session_id=args.session_id,
                   out_dir=out_dir, run_label=args.run_label or run_dir.name,
                   furniture_mask=args.furniture_mask == "on")


if __name__ == "__main__":
    main()
