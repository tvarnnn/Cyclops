"""Shadow-score a PUBLISHED final solve with the final scale guard, read-only and CPU-only.

The guard itself (`tower/world_builder/final_scale_guard.py`) runs at the publish seam of a final solve.
This command runs its pure assessment, with the same frozen parameters, over a solve that is already
published -- a frozen replay arm, a pinned walk -- so its decisions can be scored without re-solving and
without a GPU. Nothing under --root is written: the solve database is opened read-only and immutable, the
depth predictions are only read, and the result goes to stdout (or to --out).

WHAT IS MEASURED
  * the room: the published solution's component-0 cameras with >= 30 observations (the components
    record's `room` when one exists -- a gated solve -- with its keyframe ids);
  * the metric scale: `coherence_publish.measure_metric_scale` over the solve's own database and the depth
    stage's raw predictions in --depth-work (default `<world>/dense/<session>/work`). An ungated walk has
    no complete depth cache of its own; pass the SAME CAPTURE's gated depth cache and --depth-solution, its
    `solution.json`, whose keyframe ids must equal this solve's (the predictions are indexed by position);
  * groups: one (the gate's in-memory groups are not on disk). A gated solve's published room is assessed
    as one group, which is stricter than the seam, never looser.

    python scripts/world_scale_guard_shadow.py --root <world root> --world <id> --session <id> \\
        [--depth-work <dir>] [--depth-solution <solution.json>] [--out <file.json>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tower.artifact_paths import artifact_root_arg  # noqa: E402

DEFAULT_ROOT = Path("data/world_builder")


def shadow_score(root: Path, world_id: str, session_id: str, *, depth_work: Path | None = None,
                 depth_solution: Path | None = None) -> dict:
    from tower.world_builder import coherence_publish as CP
    from tower.world_builder import final_scale_guard as FSG
    from tower.world_builder import global_solve as GS
    from tower.world_builder.store import WorldStore

    started = time.perf_counter()
    store = WorldStore(root)
    solution = GS.load_solution(store, world_id, session_id)
    if solution is None:
        raise SystemExit(f"no published solution for {world_id}/{session_id}")
    workspace = GS.workspace_for(store, world_id, session_id)
    keyframes = store.read_keyframes(world_id, session_id)
    database = workspace.root / ((solution.solve or {}).get("database") or "database.db")
    if not database.is_file():
        raise SystemExit(f"the solve's database is missing: {database}")
    work = Path(depth_work) if depth_work else store.world_dir(world_id) / "dense" / session_id / "work"
    if not (work / "depth").is_dir():
        raise SystemExit(f"no depth predictions under {work}")
    if depth_solution is not None:
        other = json.loads(Path(depth_solution).read_text(encoding="utf-8"))
        if other.get("keyframe_ids") != solution.keyframe_ids or other.get("input_digest") != solution.input_digest:
            raise SystemExit("--depth-solution is not the same frozen keyframe set as this solve")
    name_of = CP._image_names(keyframes)
    scale = CP.measure_metric_scale(solution, name_of, database, work)
    params = FSG.GuardParams()
    room_solution = solution
    comp_path = workspace.root / CP.COMPONENTS_FILENAME
    room_source = "solver component 0"
    if comp_path.is_file():
        doc = json.loads(comp_path.read_text(encoding="utf-8"))
        room = {k for e in doc.get("components") or [] if e.get("shown_as") == "room" for k in e.get("keyframe_ids") or []}
        if room:
            import dataclasses

            room_solution = dataclasses.replace(solution, poses={
                k: dict(p, component=0 if k in room else 1) for k, p in solution.poses.items()})
            room_source = "the components record's room"
    cams = FSG.room_cameras(room_solution, keyframes, name_of, min_obs=params.min_obs)
    assessment = FSG.assess(cams, scale["metric_log"], params)
    return {
        "record": "wb-final-scale-guard-shadow/1", "guard": FSG.GUARD_ID_V1, "params_digest": params.digest(),
        "world_id": world_id, "session_id": session_id, "input_digest": solution.input_digest,
        "solve_identity": GS.solve_identity(solution), "room_source": room_source,
        "database": str(database), "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "depth_work": str(work), "frames_without_prediction": scale.get("frames_without_prediction"),
        "metric": {k: scale.get(k) for k in ("cameras_published", "cameras_measured", "pairs_used",
                                             "inliers_total", "inliers_gated")},
        "seconds": round(time.perf_counter() - started, 3),
        "assessment": FSG._jsonable(assessment),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=artifact_root_arg, default=str(DEFAULT_ROOT))
    parser.add_argument("--world", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--depth-work", type=Path, default=None)
    parser.add_argument("--depth-solution", type=Path, default=None)
    parser.add_argument("--out", type=artifact_root_arg, default=None,
                        help="write the JSON here instead of stdout (never under --root)")
    args = parser.parse_args(argv)
    result = shadow_score(Path(args.root), args.world, args.session, depth_work=args.depth_work,
                          depth_solution=args.depth_solution)
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        out = Path(args.out)
        if Path(args.root).resolve() in out.resolve().parents:
            raise SystemExit("--out must not be under --root: this command writes nothing there")
        out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
