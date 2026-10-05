"""CPU-only walk-size child commit probe; output must be inside this worktree.

Copies one frozen world, launches one real world_solve_draw.py child on a private
database, and reports its peak Windows pagefile/commit. Never deletes its copy.
"""

import argparse
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import psutil

from prove_identity import GS, WorldStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-world", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        parser.error("set CUDA_VISIBLE_DEVICES=-1")
    root = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    if root not in out.parents or ".w01f-scratch" not in out.parts or out.exists():
        parser.error("out must be a fresh directory under this worktree/.w01f-scratch")
    world = out / "worlds" / args.source_world.name
    world.parent.mkdir(parents=True)
    shutil.copytree(args.source_world, world)
    store = WorldStore(out)
    workspace = GS.workspace_for(store, world.name, args.session)
    base = GS.load_solution(store, world.name, args.session)
    if base is None or not base.solve or "database" not in base.solve:
        raise RuntimeError("source lacks a solved database")
    source = workspace.root / base.solve["database"]
    digest = GS.database_digest(source)["content"]
    private_root = workspace.root / "commit-probe"
    private_root.mkdir()
    database = private_root / "database.db"
    GS._private_draw_database(source, database)
    if GS.database_digest(database)["content"] != digest:
        raise RuntimeError("private database digest differs")
    camera = GS.PinholeCamera.from_json_dict(base.camera or
                                              GS.read_json_closed(workspace.camera_path))
    context = private_root / "context.pkl"
    with context.open("wb") as handle:
        pickle.dump((workspace, store.read_keyframes(world.name, args.session), camera,
                     base.input_digest, GS.MIN_IMAGE_OBSERVATIONS), handle)
    free_at_launch = GS._draw_free_ram()
    process, job = GS._launch_draw_child(context, database, args.seed,
                                         private_root / "sparse",
                                         private_root / "candidate.pkl", digest)
    peak_commit = 0
    try:
        child = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                info = child.memory_info()
                peak_commit = max(peak_commit, getattr(info, "pagefile", info.vms))
            except psutil.NoSuchProcess:
                break
            time.sleep(0.05)
        code = process.wait(timeout=5)
    finally:
        if job is not None:
            job.close()
    result = {"world": world.name, "session": args.session, "seed": args.seed,
              "keyframes": len(store.read_keyframes(world.name, args.session)),
              "images": GS._draw_image_count(source), "free_at_launch": free_at_launch,
              "peak_commit_bytes": peak_commit, "exit_code": code,
              "source_digest_after": GS.database_digest(source)["content"],
              "private_digest_after": GS.database_digest(database)["content"]}
    (out / "commit_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if code or result["source_digest_after"] != digest or result["private_digest_after"] != digest:
        raise RuntimeError("child failed or a database changed")


if __name__ == "__main__":
    main()
