#!/usr/bin/env python
"""Prepare and replay Exp2 S/R on independent copied roots; never target a live store.

Fill exp2_adjudication_template.csv from raw views BEFORE opening candidate files.
Pair truth: true_doorway_return, false_cross_room, ambiguous. Record the copied
label sheet SHA in freeze.json before either arm's output is opened.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from tower.world_builder import exp2_retrieval as E
from tower.world_builder import global_solve as GS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    copy = sub.add_parser("copy")
    copy.add_argument("source", type=Path)
    copy.add_argument("destination", type=Path)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("root", type=Path)
    freeze.add_argument("--images-dir", type=Path, required=True)
    freeze.add_argument("--tree", type=Path, required=True)
    freeze.add_argument("--database", type=Path, required=True)
    freeze.add_argument("--calibration", type=Path, required=True)
    freeze.add_argument("--labels", type=Path, required=True)
    freeze.add_argument("--adjudication", type=Path, required=True)
    freeze.add_argument("--settings", type=Path, required=True,
                        help="JSON pin of calibration, seed, masks, gate, guard, mapper and checker")
    tree = sub.add_parser("tree-candidates")
    tree.add_argument("root", type=Path)
    tree.add_argument("--database", type=Path, required=True)
    tree.add_argument("--images-dir", type=Path, required=True)
    tree.add_argument("--tree", type=Path, required=True)
    tree.add_argument("--freeze-sha256", required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("root", type=Path)
    replay.add_argument("--arm", choices=("S", "R"), required=True)
    replay.add_argument("--world", required=True)
    replay.add_argument("--session", required=True)
    replay.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args(argv)

    if args.action == "copy":
        print(json.dumps(E.copy_inputs(args.source, args.destination), sort_keys=True))
        return 0
    E.verify_copied_inputs(args.root)
    if args.action == "freeze":
        import pycolmap
        target = args.root / "exp2-freeze.json"
        if target.exists():
            raise FileExistsError(target)
        options = pycolmap.SequentialPairingOptions()
        options.overlap = E.OVERLAP
        options.quadratic_overlap = False
        options.loop_detection = True
        images = sorted(args.images_dir.glob("*.jpg"))
        if not images:
            raise ValueError("freeze requires prepared solver images")
        if (args.database.resolve() != (args.images_dir.parent / "database.db").resolve()
                or args.calibration.resolve() != (args.images_dir.parent / "camera.json").resolve()):
            raise ValueError("database and calibration must be beside the prepared images")
        settings = json.loads(args.settings.read_text(encoding="utf-8"))
        repo = Path(__file__).resolve().parents[2]
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                           text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain", "--", "tower"],
                                   cwd=repo, text=True).strip():
            raise ValueError("freeze requires committed, clean tower code")
        manifest = E.freeze_manifest(images, args.tree, E.checkpoint_file(),
                                     pycolmap.__version__, options.todict(),
                                     labels_sha256=E.sha256(args.labels),
                                     adjudication_sha256=E.sha256(args.adjudication),
                                     settings=settings)
        manifest["label_path"] = str(args.labels.resolve())
        manifest["adjudication_path"] = str(args.adjudication.resolve())
        manifest["database_sha256"] = E.sha256(args.database)
        manifest["calibration_sha256"] = E.sha256(args.calibration)
        manifest["settings_file_sha256"] = E.sha256(args.settings)
        manifest["code_sha"] = code_sha
        E._json(target, manifest)
        print(json.dumps({"manifest": str(target), "sha256": E.sha256(target)}))
        return 0
    frozen_path = args.root / "exp2-freeze.json"
    if not frozen_path.is_file() or E.sha256(frozen_path) != args.freeze_sha256:
        raise ValueError("freeze manifest SHA-256 differs from pinned value")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    for path_key, hash_key in (("label_path", "labels_sha256"),
                               ("adjudication_path", "adjudication_sha256")):
        if E.sha256(Path(frozen[path_key])) != frozen[hash_key]:
            raise ValueError(f"frozen {path_key} digest changed")
    if args.action == "tree-candidates":
        if E.sha256(args.tree) != frozen["tree_sha256"]:
            raise ValueError("frozen tree digest changed")
        if E.sha256(args.database) != frozen["database_sha256"]:
            raise ValueError("frozen database digest changed")
        names = [p.name for p in sorted(args.images_dir.glob("*.jpg"))]
        output = args.root / "exp2-tree-candidates.json"
        if output.exists():
            raise FileExistsError(output)
        print(json.dumps(E.log_tree_candidates(args.database, args.tree, names, output)))
        return 0
    if E.sha256(E.checkpoint_file()) != frozen["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint digest changed")
    images_dir = args.root / "worlds" / args.world / "solve" / args.session / "images"
    if E.sha256(images_dir.parent / "database.db") != frozen["database_sha256"]:
        raise ValueError("frozen database digest changed before replay")
    if E.sha256(images_dir.parent / "camera.json") != frozen["calibration_sha256"]:
        raise ValueError("frozen calibration digest changed")
    observed = [{"name": p.name, "sha256": E.sha256(p)}
                for p in sorted(images_dir.glob("*.jpg"))]
    if observed != frozen["images"]:
        raise ValueError("frozen replay image order or digest changed")
    from scripts import world_finalize
    previous = os.environ.get(E.SWITCH)
    try:
        os.environ[E.SWITCH] = "dinov2_gem" if args.arm == "R" else "off"
        return world_finalize.main(["--root", str(args.root), "--world", args.world,
                                    "--session", args.session, "--format", "json"])
    finally:
        if previous is None:
            os.environ.pop(E.SWITCH, None)
        else:
            os.environ[E.SWITCH] = previous


if __name__ == "__main__":
    raise SystemExit(main())
