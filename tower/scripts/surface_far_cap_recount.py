#!/usr/bin/env python
"""Recount a world's surface block allocation under the far-depth cap, per k.

WHY THIS EXISTS

`TOWER_WORLD_SURFACE_FAR_DEPTH_CAP` (`surface_pipeline.FarDepthCap`) stops
depth deeper than k x the scene scale from allocating blocks, so far "depth
spray" cannot push the walk over `SurfaceParams.max_blocks` and coarsen the
whole room's voxel. Choosing k is a question about real worlds, and a full
surface build per k is minutes of GPU each. This answers it on the CPU from
what the surface stage reads -- the depth stage's maps and fits, the solve's
poses and sparse points -- with the SAME arithmetic: `_Frames`,
`_scene_scale`, `_fusion_bound` and `_Frames.prepared` prepare the frames
exactly as a build does, and `_allocate_blocks` is the build's own budget loop.
No fusion, no meshing, and nothing is written.

    python scripts/surface_far_cap_recount.py --root data/world_builder --world <id>
    python scripts/surface_far_cap_recount.py --root data/world_builder --world <id> \\
        --session <sid> --k 2,3,4,5,6,8,off --json

Per k (and `off`): the blocks the frames allocate at the stage's own voxel
under that cap, the voxel coarsening a build's block budget would then apply
and the blocks it would reserve, and the fraction of valid depth samples
beyond the cap. A build engages the cap only when the walk is over the budget
without it (`engaged`), and a k whose cap would drop more than `--max-drop`
of the samples stands down (`FarDepthCap`); either way that build allocates
as `off`, and the row says so.

TWO APPROXIMATIONS, both the same for every row: the transient (hand and
phone) masks are not applied, and the depth-consistency field is applied only
when the session's cached field is the one its last surface build recorded
(otherwise the plain affine fit is used; the header says which). The `off`
row's coarsening is therefore a check against the manifest's
`outliers.voxel_coarsened_for_block_budget`, not a copy of it.

NEEDS THE DEPTH WORK. A finished build prunes `<world>/dense/<sid>/work/`
(`prune_intermediates`); such a session is reported and skipped. Rebuilding
the depth stage (`world_surface.py --world <id> --session <sid> --force`, on
the GPU) brings it back.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402

DEFAULT_ROOT = Path("data") / "world_builder"
DEFAULT_KS = "2,3,4,5,6,8,off"


def _parse_ks(text: str) -> list:
    out = []
    for word in (w.strip().lower() for w in text.split(",")):
        if not word:
            continue
        if word == "off":
            out.append(None)
            continue
        k = float(word)
        if not (k > 0 and k == k and k != float("inf")):
            raise argparse.ArgumentTypeError(f"k must be a positive number or 'off': {word!r}")
        out.append(k)
    if not out:
        raise argparse.ArgumentTypeError("no k given")
    return out


def _sessions_with_solves(store, world_id):
    from tower.world_builder.global_solve import load_solution  # noqa: PLC0415

    world = store.read_world(world_id)
    return [sid for sid in world.session_ids
            if load_solution(store, world_id, sid) is not None]


def _cached_consistency(dense_root: Path):
    """The field the session's last surface build applied, read-only, or None."""
    from tower.world_builder import depth_consistency as DC  # noqa: PLC0415

    try:
        record = json.loads((dense_root / DC.RECORD_FILE).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("state") != DC.STATE_APPLIED:
        return None
    field, key = DC.ConsistencyField.load(dense_root / DC.FIELD_FILE)
    if field is None or key != record.get("key"):
        return None
    return field


def recount(store, world_id: str, session_id: str, ks: list, *, params, max_drop: float,
            device: str = "cpu") -> dict:
    """The recount of one session, as a dict (see the module docstring)."""
    import torch

    from tower.world_builder import surface_pipeline as SP  # noqa: PLC0415
    from tower.world_builder.dense_pipeline import dense_dir  # noqa: PLC0415
    from tower.world_builder.global_solve import load_solution  # noqa: PLC0415
    from tower.world_builder.surface import truncation_rel  # noqa: PLC0415

    out = {"world": world_id, "session": session_id}
    solution = load_solution(store, world_id, session_id)
    if solution is None:
        return {**out, "skipped": "no global solution for this session"}
    root = dense_dir(store, world_id, session_id)
    try:
        align = json.loads((root / "align.json").read_text())
    except (OSError, ValueError):
        return {**out, "skipped": f"no readable depth stage at {root / 'align.json'}"}
    work = root / "work"
    if not (work / "depth").is_dir() or not (work / "undist").is_dir():
        return {**out, "skipped": f"the depth work under {work} is gone (pruned after the "
                                  "build); rebuild the depth stage to recount"}

    dev = torch.device(device)
    frames = SP._Frames(align, work, solution, params)
    if not len(frames):
        return {**out, "skipped": "no keyframe passed the alignment gate"}
    if params.depth_consistency:
        frames.correction = _cached_consistency(root)
    median_depth, scale_source = SP._scene_scale(frames, solution)
    if not (median_depth > 0):
        return {**out, "skipped": "the solve has no usable scene depth"}
    voxel = params.voxel_frac * median_depth
    frames.bound = SP._fusion_bound(frames, solution, params, median_depth)
    # The compaction `_build` applies (float16 depth on the host); the images
    # and weights are not needed to allocate.
    cached = [(z.to(torch.float16).cpu(), ok.cpu(), None, R, tt, None)
              for z, ok, _img, R, tt, _w in frames.prepared(params, median_depth, dev)]
    if not cached:
        return {**out, "skipped": "no frame produced usable depth"}
    trel = truncation_rel(params, frames.median_held_out)
    unbudgeted = dataclasses.replace(params, max_blocks=0)

    def allocate(p, cap, always=False):
        vol = SP._new_volume(p, trel, voxel, median_depth, frames.median_held_out, dev)
        return SP._allocate_blocks(vol, cached, frames.K, p, voxel, trel, median_depth,
                                   frames.median_held_out, dev, far_cap=cap,
                                   far_cap_always=always)

    out.update({
        "frames": len(cached), "scene_scale": median_depth, "scene_scale_source": scale_source,
        "preset": params.quality, "voxel": voxel, "max_blocks": params.max_blocks,
        "max_drop": max_drop,
        "consistency_field": "cached" if frames.correction is not None else "plain affine",
        "transient_masks": "not applied", "rows": [],
    })
    for k in ks:
        t = time.time()
        row = {"k": "off" if k is None else k}
        cap = None
        if k is not None:
            far = SP.FarDepthCap(k=float(k), max_drop=float(max_drop))
            cap = far.measure(cached, median_depth)
            measured = far.record(1.0)
            row.update({name: measured[name] for name in (
                "cap", "fraction_beyond_cap", "frames_wholly_beyond_cap")})
            row["stood_down"] = far.stood_down
        # What the cap allocates at the stage's voxel, applied whether or not
        # a build would need it; then what a build does, which engages it only
        # over the budget.
        free = allocate(unbudgeted, cap, always=True)
        row["blocks_at_voxel"] = None if free.keys is None else int(free.keys.numel())
        # Uncapped and within the budget, the build's loop stops at its first
        # pass, which is exactly `free`.
        within = cap is None and free.keys is not None and free.key_coarsened == 1.0 and (
            params.max_blocks <= 0 or free.keys.numel() <= params.max_blocks)
        budget = free if within else allocate(params, cap)
        if k is not None:
            row["engaged"] = budget.far_cap_engaged
        row.update({
            "voxel_coarsened_by": round(budget.coarsened, 4),
            "voxel_coarsened_for_block_budget": round(budget.coarsened / budget.key_coarsened, 4),
            "blocks_reserved": None if budget.keys is None else int(budget.keys.numel()),
            "refusal": budget.refusal or free.refusal,
            "seconds": round(time.time() - t, 2),
        })
        out["rows"].append(row)
    return out


def _print(result: dict) -> None:
    print(f"=== {result['world']} / {result['session']} ===")
    if "skipped" in result:
        print(f"  skipped: {result['skipped']}")
        return
    print(f"  {result['frames']} frames, scene scale {result['scene_scale']:.4g} "
          f"({result['scene_scale_source']}), voxel {result['voxel']:.5g}, budget "
          f"{result['max_blocks']} blocks ({result['preset']} preset); consistency "
          f"{result['consistency_field']}, "
          f"transient masks {result['transient_masks']}")
    print(f"  {'k':>5} {'cap':>9} {'beyond':>8} {'wholly':>6} {'blocks@voxel':>13} "
          f"{'coarsened':>9} {'reserved':>9}  note")
    for r in result["rows"]:
        cap = f"{r['cap']:.4g}" if "cap" in r else "-"
        beyond = f"{100 * r['fraction_beyond_cap']:.2f}%" if "cap" in r else "-"
        wholly = str(r.get("frames_wholly_beyond_cap", "-"))
        note = ("stood down (allocates as off)" if r.get("stood_down") else
                "not needed: within budget, a build allocates as off"
                if "engaged" in r and not r["engaged"] else "")
        if r.get("refusal"):
            note = (note + "; " if note else "") + "refused: " + r["refusal"]
        print(f"  {str(r['k']):>5} {cap:>9} {beyond:>8} {wholly:>6} "
              f"{str(r['blocks_at_voxel']):>13} {r['voxel_coarsened_by']:>8.3f}x "
              f"{str(r['blocks_reserved']):>9}  {note}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=artifact_root_arg, default=str(DEFAULT_ROOT),
                    help="world root (routed through the artifact-path guard)")
    ap.add_argument("--world", required=True)
    ap.add_argument("--session", help="one session; default every session with a solve")
    ap.add_argument("--k", dest="ks", type=_parse_ks, default=_parse_ks(DEFAULT_KS),
                    help=f"comma list of multiples of the scene scale, and/or 'off' "
                         f"(default {DEFAULT_KS})")
    ap.add_argument("--max-drop", dest="max_drop", type=float, default=None,
                    help="the stand-down fraction (default: the setting's default)")
    ap.add_argument("--live", action="store_true", help="the live preset, not the final one")
    ap.add_argument("--device", default="cpu", help="torch device (default cpu)")
    ap.add_argument("--json", action="store_true", help="print one JSON object per session")
    args = ap.parse_args(argv)

    from tower.config import WORLD_SURFACE_FAR_DEPTH_CAP_MAX_DROP_DEFAULT  # noqa: PLC0415
    from tower.world_builder.store import WorldStore  # noqa: PLC0415
    from tower.world_builder.surface import SurfaceParams  # noqa: PLC0415

    max_drop = WORLD_SURFACE_FAR_DEPTH_CAP_MAX_DROP_DEFAULT if args.max_drop is None \
        else args.max_drop
    if not (0.0 < max_drop <= 1.0):
        ap.error("--max-drop must be in (0, 1]")
    store = WorldStore(Path(args.root))
    params = SurfaceParams.live() if args.live else SurfaceParams()
    sessions = [args.session] if args.session else _sessions_with_solves(store, args.world)
    if not sessions:
        print(f"no solved session in world {args.world}", file=sys.stderr)
        return 2
    skipped = 0
    for sid in sessions:
        result = recount(store, args.world, sid, args.ks, params=params, max_drop=max_drop,
                         device=args.device)
        skipped += "skipped" in result
        if args.json:
            print(json.dumps(result), flush=True)
        else:
            _print(result)
    return 1 if skipped == len(sessions) else 0


if __name__ == "__main__":
    raise SystemExit(main())
