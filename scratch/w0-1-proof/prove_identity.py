"""C18 W0-1 proof runner. Run only under the lead's gpulock; never reads live data.

The input is one already frozen, finished world directory. Six independent copies
are made beneath --out. This script refuses an existing output directory and never
deletes input or output. The direct phase maps draws 1 and 2 three times per arm.
With --full-refinish it also runs six full finishes and compares published artifacts.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tower"))

import numpy as np  # noqa: E402
from tower.world_builder import global_solve as GS  # noqa: E402
from tower.world_builder.store import WorldStore  # noqa: E402


def _digest_candidate(candidate):
    out = {}
    for field in dataclasses.fields(candidate):
        if field.name in ("solved_at", "timing"):
            continue
        value = getattr(candidate, field.name)
        if isinstance(value, np.ndarray):
            data = str(value.dtype).encode() + repr(value.shape).encode() + value.tobytes()
        else:
            data = json.dumps(value, sort_keys=True, allow_nan=True).encode()
        out[field.name] = hashlib.sha256(data).hexdigest()
    return out


def _copy_world(source, root, *, omit_pair_cache=False):
    world_id = source.name
    destination = root / "worlds" / world_id
    destination.parent.mkdir(parents=True)

    def ignore(directory, names):
        if omit_pair_cache and Path(directory).parent.name == "solve":
            return {"verify_pairs"} & set(names)
        return set()

    shutil.copytree(source, destination, ignore=ignore)
    return destination


def _load(root, world_id, session, expected_seed):
    store = WorldStore(root)
    workspace = GS.workspace_for(store, world_id, session)
    base = GS.load_solution(store, world_id, session)
    if base is None or (base.solve or {}).get("matching") != "frozen":
        raise RuntimeError("source solve is absent or its matching is not frozen")
    if base.solve.get("seed") != expected_seed:
        raise RuntimeError(f"frozen solve seed is {base.solve.get('seed')}, expected {expected_seed}")
    database = workspace.root / base.solve["database"]
    digest = (GS.database_digest(database) or {}).get("content")
    if not digest:
        raise RuntimeError(f"frozen database digest unavailable: {database}")
    return store, workspace, base, database, digest


def _direct(source, out, session, seed):
    hashes = {}
    world_id = source.name
    for arm in ("old", "new"):
        for repeat in range(1, 4):
            root = out / "direct" / f"{arm}-{repeat}"
            _copy_world(source, root)
            store, workspace, base, database, expected = _load(root, world_id, session, seed)
            keyframes = store.read_keyframes(world_id, session)
            if arm == "old":
                mapper = GS.frozen_draw_mapper(store, world_id, session, database,
                                               base, keyframes=keyframes)
            else:
                mapper = GS.concurrent_draw_mapper(store, world_id, session, database, base,
                                                   seeds=(seed + 1, seed + 2), keyframes=keyframes)
            try:
                pair = {str(s): _digest_candidate(mapper(s)) for s in (seed + 1, seed + 2)}
            finally:
                close = getattr(mapper, "close", None)
                if close:
                    close()
            if (GS.database_digest(database) or {}).get("content") != expected:
                raise RuntimeError(f"mapped source database changed: {arm}-{repeat}")
            if arm == "new":
                for s in (seed + 1, seed + 2):
                    private = workspace.root / "sparse-draws" / f"seed-{s}" / "database.db"
                    if (GS.database_digest(private) or {}).get("content") != expected:
                        raise RuntimeError(f"private database missing or changed: {arm}-{repeat}-{s}")
                    with private.with_name("candidate.pkl").open("rb") as handle:
                        child = pickle.load(handle)
                    if (child.get("before") != expected or child.get("after") != expected or
                            child.get("candidate") is None):
                        raise RuntimeError(f"child map did not complete on the private database: {arm}-{repeat}-{s}")
            hashes[f"{arm}-{repeat}"] = pair
            print(f"direct {arm}-{repeat}: mapped and digest checked", flush=True)
    reference = hashes["old-1"]
    for name, pair in hashes.items():
        if pair != reference:
            raise RuntimeError(f"per-draw candidate differs: old-1 vs {name}")
    return hashes


def _env_file(path):
    env = dict(os.environ)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"invalid env line: {line}")
        env[key.strip()] = value.strip()
    env["PYTHONPATH"] = str(REPO / "tower")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TOWER_WORLD_SOLVE_CONSENSUS"] = "3"
    env["TOWER_WORLD_APPEARANCE_DETERMINISTIC_GAINS"] = "on"
    env["TOWER_WORLD_STAGE_TIMING"] = "on"
    return env


def _pairs(world, session):
    caches = sorted((world / "solve" / session / "verify_pairs").glob("*/pairs.npz"))
    if not caches:
        raise RuntimeError(f"P4 pair set did not run: {world}")
    result = {}
    for path in caches:
        with np.load(path, allow_pickle=False) as arrays:
            result[path.parent.name] = {
                key: hashlib.sha256(str(arrays[key].dtype).encode() +
                                    repr(arrays[key].shape).encode() +
                                    arrays[key].tobytes()).hexdigest()
                for key in arrays.files
            }
            if "similarity" not in arrays.files:
                raise RuntimeError(f"P4 similarity missing: {path}")
    return result


def _compare_published(compare_path, world_a, world_b, session, report):
    """C18 spec with one I0-only provenance key excluded in memory."""
    spec = importlib.util.spec_from_file_location("w0_compare_identity", compare_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VOLATILE_KEYS.add("consensus_concurrent")
    result = module.compare(world_a, world_b, session=session)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [f"{name}: {record['status']}" for name, record in result["artifacts"].items()]
    for name in result["differs"]:
        lines.extend(result["artifacts"][name].get("diffs", []))
    lines.append(f"VERDICT {result['verdict']}")
    report.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _full(source, out, session, seed, env_file, compare_w0):
    env = _env_file(env_file)
    worlds = {}
    pair_hashes = {}
    for arm in ("old", "new"):
        for repeat in range(1, 4):
            root = out / "full" / f"{arm}-{repeat}"
            world = _copy_world(source, root, omit_pair_cache=True)
            local_env = dict(env, TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT=(
                "off" if arm == "old" else "after-draw-0"))
            command = [sys.executable, str(REPO / "tower" / "scripts" / "world_refinish.py"),
                       "--root", str(root), "--world", source.name,
                       "--session", session, "--seed", str(seed), "--format", "json"]
            with (root / "refinish.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, env=local_env, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"refinish {arm}-{repeat} exited {result.returncode}; see {root / 'refinish.log'}")
            _, _, solution, _, _ = _load(root, source.name, session, seed)
            consensus = (solution.gate or {}).get("consensus") or {}
            if consensus.get("requested") != 3:
                raise RuntimeError(f"consensus did not run: {arm}-{repeat}")
            i0_path = world / "sessions" / session / "stage_timing.json"
            i0 = json.loads(i0_path.read_text(encoding="utf-8"))
            solves = [finish["stages"]["solve"][0]
                      for finish in i0.get("finishes", [])
                      if (finish.get("stages") or {}).get("solve")]
            if not solves:
                raise RuntimeError(f"I0 solve stage missing: {arm}-{repeat}")
            mode = solves[-1].get("consensus_concurrent")
            if mode != (None if arm == "old" else "after-draw-0"):
                raise RuntimeError(f"concurrent draw stage did not run as intended: {arm}-{repeat}, {mode}")
            worlds[f"{arm}-{repeat}"] = world
            pair_hashes[f"{arm}-{repeat}"] = _pairs(world, session)
            print(f"full {arm}-{repeat}: refinish and P4 checked", flush=True)
    reference = pair_hashes["old-1"]
    for name, pair in pair_hashes.items():
        if pair != reference:
            raise RuntimeError(f"P4 pair arrays or similarity differ: old-1 vs {name}")
    for name, world in worlds.items():
        if name == "old-1":
            continue
        report = out / f"compare-old-1-vs-{name}.json"
        result = _compare_published(compare_w0, worlds["old-1"], world, session, report)
        if result["verdict"] != "IDENTICAL":
            raise RuntimeError(f"published world differs: old-1 vs {name}; see {report}")
    return pair_hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-world", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full-refinish", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--compare-w0", type=Path)
    args = parser.parse_args(argv)
    source = args.source_world.resolve()
    out = args.out.resolve()
    if not source.is_dir() or out == source or source in out.parents:
        parser.error("source world must exist and output must be outside it")
    if out.exists():
        parser.error("output already exists; choose a fresh directory")
    if args.full_refinish and (args.env_file is None or args.compare_w0 is None):
        parser.error("--full-refinish requires --env-file and --compare-w0")
    if args.full_refinish and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        parser.error("full refinish needs the lead's GPU-locked environment")
    out.mkdir(parents=True)
    report = {"source": str(source), "session": args.session, "seed": args.seed}
    report["direct"] = _direct(source, out, args.session, args.seed)
    if args.full_refinish:
        report["pairs"] = _full(source, out, args.session, args.seed,
                                args.env_file, args.compare_w0)
    (out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("VERDICT IDENTICAL; see result.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
