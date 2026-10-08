"""Offline Exp2 candidate ranking. The product path imports this only for the exact opt-in."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np

from tower.world_builder import global_solve as GS

SWITCH = "TOWER_WORLD_EXP2_RETRIEVAL"
MODEL_ID = "facebook/dinov2-small"
HEIGHT, WIDTH, GEM_P = 224, 392, 3
CADENCE, K, MIN_DISTANCE = 10, 50, 0
OVERLAP = 20
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
COPY_RECORD = "exp2-copied-inputs.json"


def enabled() -> bool:
    return os.environ.get(SWITCH) == "dinov2_gem"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n",
                          encoding="utf-8", newline="\n")


def _json_round_trip(value: dict) -> dict:
    """Normalize a pycolmap `.todict()` the same way the frozen manifest was written.

    `.todict()` can hold enum members (e.g. `FeatureMatcherType.SIFT_BRUTEFORCE`); `_json`
    stringifies them (`default=str`) on the way to disk. Comparing a freshly-read `.todict()`
    straight against the re-loaded (already-stringified) frozen value would spuriously
    mismatch on every enum field, so both sides must go through the same round trip.
    """
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def copy_inputs(source: Path, destination: Path) -> dict:
    """Copy a closed input tree; never hard-link evidence or overwrite a replay."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir() or source == destination or source in destination.parents:
        raise ValueError("source must be a separate closed directory")
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)
    files = {p.relative_to(destination).as_posix(): sha256(p)
             for p in sorted(destination.rglob("*")) if p.is_file()}
    for relative, digest in files.items():
        if sha256(source / relative) != digest:
            raise ValueError(f"source changed during copy: {relative}")
    record = {"source": str(source), "destination": str(destination), "files": files}
    _json(destination / COPY_RECORD, record)
    return record


def verify_copied_inputs(destination: Path, record: dict | None = None) -> bool:
    destination = Path(destination).resolve()
    if record is None:
        record = json.loads((destination / COPY_RECORD).read_text(encoding="utf-8"))
    if record.get("destination") != str(destination):
        raise ValueError("copy provenance destination differs")
    for relative, expected in record["files"].items():
        file = destination / relative
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"copied input digest differs: {relative}")
    return True


def checkpoint_file() -> Path:
    """Resolve the already cached standard LVD-142M weights without network access."""
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(MODEL_ID, local_files_only=True))
    path = root / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def freeze_manifest(images, tree: Path, checkpoint: Path, pycolmap_version: str,
                    sequential_pairing_options: dict, *, labels_sha256: str,
                    feature_matching_options: dict, two_view_geometry_options: dict,
                    adjudication_sha256: str | None = None,
                    settings: dict | None = None) -> dict:
    images = [Path(p) for p in images]
    return {
        "images": [{"name": p.name, "sha256": sha256(p)} for p in images],
        "tree_sha256": sha256(tree), "checkpoint_sha256": sha256(checkpoint),
        "model": MODEL_ID, "pycolmap_version": pycolmap_version,
        "sequential_pairing_options": sequential_pairing_options,
        # PREREG-EXP2-RETRIEVAL-20261008.md's freeze-manifest clause: "PyCOLMAP installed
        # defaults including RANSAC seed/threshold, matching threads". These are the
        # *installed library's* defaults (a fresh `FeatureMatchingOptions()` /
        # `TwoViewGeometryOptions()`), not the per-solve overrides global_solve.solve makes --
        # a pycolmap upgrade that silently moves them must refuse replay, not drift unnoticed.
        "feature_matching_options": feature_matching_options,
        "two_view_geometry_options": two_view_geometry_options,
        "labels_sha256": labels_sha256,
        "adjudication_sha256": adjudication_sha256,
        "retrieval": {"height": HEIGHT, "width": WIDTH, "mean": MEAN, "std": STD,
                      "gem_p": GEM_P, "gem_clamp": 1e-6,
                      "normalization": "L2", "similarity": "cosine",
                      "tie_break": "keyframe_index_ascending", "cadence": CADENCE,
                      "k": K, "min_index_distance": MIN_DISTANCE,
                      "sequential_overlap": OVERLAP, "sift_features": GS.MAX_FEATURES},
        "settings": settings or {},
        "conditional_v_candidate_sha256": None,
        "conditional_v_status": "not triggered in slices 1-3",
    }


def verify_installed_pycolmap_defaults(frozen: dict, pycolmap_module) -> None:
    """Refuse a replay whose installed PyCOLMAP differs from the one the freeze pinned.

    Prereg freeze-manifest clause: PyCOLMAP version, matching threads and RANSAC/two-view
    defaults are pinned at freeze and must be re-checked at replay, not just the pairing
    options -- an upgraded pycolmap can change candidate-count accounting silently otherwise.
    """
    live_version = getattr(pycolmap_module, "__version__", None)
    if live_version != frozen["pycolmap_version"]:
        raise ValueError(
            f"installed pycolmap version {live_version!r} differs from frozen "
            f"{frozen['pycolmap_version']!r}")
    live_matching = _json_round_trip(pycolmap_module.FeatureMatchingOptions().todict())
    if live_matching != frozen["feature_matching_options"]:
        raise ValueError("installed pycolmap matching defaults differ from the frozen value")
    live_verification = _json_round_trip(pycolmap_module.TwoViewGeometryOptions().todict())
    if live_verification != frozen["two_view_geometry_options"]:
        raise ValueError("installed pycolmap verification defaults differ from the frozen value")


def gem_patch_tokens(patches: np.ndarray) -> np.ndarray:
    """Fixed p=3 GeM on final-layer patch tokens, then per-image L2."""
    patches = np.asarray(patches, dtype=np.float32)
    pooled = np.power(np.mean(np.power(np.maximum(patches, 1e-6), GEM_P), axis=1),
                      1.0 / GEM_P)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)


def dinov2_descriptors(paths) -> np.ndarray:
    """CPU-only local checkpoint; fixed RGB bicubic resize, ImageNet normalization."""
    import torch
    from PIL import Image
    from transformers import Dinov2Model

    model = Dinov2Model.from_pretrained(MODEL_ID, local_files_only=True).cpu().eval()
    out = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
            rgb = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0)
        with torch.inference_mode():
            patches = model(pixel_values=tensor, interpolate_pos_encoding=True).last_hidden_state[:, 1:, :].cpu().numpy()
        out.append(gem_patch_tokens(patches)[0])
    return np.stack(out) if out else np.empty((0, 384), np.float32)


def descriptor_cache(images, path: Path, checkpoint_sha256: str, model=dinov2_descriptors) -> np.ndarray:
    images, path = [Path(p) for p in images], Path(path)
    identity = {"checkpoint_sha256": checkpoint_sha256,
                "images": [{"name": p.name, "sha256": sha256(p)} for p in images],
                "height": HEIGHT, "width": WIDTH, "gem_p": GEM_P, "gem_clamp": 1e-6,
                "mean": list(MEAN), "std": list(STD)}
    if path.exists():
        held = json.loads(path.read_text(encoding="utf-8"))
        if held.get("identity") == identity:
            return np.asarray(held["vectors"], dtype=np.float32)
    vectors = np.asarray(model(images), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(images):
        raise ValueError("descriptor count differs from image count")
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    _json(path, {"identity": identity, "vectors": vectors.tolist()})
    return vectors


def rank_candidates(names, vectors: np.ndarray) -> list[dict]:
    names = list(names)
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(names) != len(vectors) or len(set(names)) != len(names):
        raise ValueError("ordered names and descriptors must be unique and aligned")
    scores = vectors @ vectors.T
    rows = []
    for q in range(0, len(names), CADENCE):
        candidates = [i for i in range(len(names)) if i != q and abs(i - q) > MIN_DISTANCE]
        candidates.sort(key=lambda i: (-float(scores[q, i]), i))
        targets = candidates[:K]
        rows.append({"query_index": q, "query": names[q], "target_indices": targets,
                     "targets": [names[i] for i in targets],
                     "cosine": [float(scores[q, i]) for i in targets]})
    return rows


def write_candidate_manifest(rows, path: Path, *, overlap: int = OVERLAP) -> dict:
    """An imported-pair file with stable order; duplicate query/target pairs count once."""
    path = Path(path)
    all_names = list(dict.fromkeys(n for row in rows for n in [row["query"], *row["targets"]]))
    index = {name: i for i, name in enumerate(all_names)}
    seen, pairs, nonsequential = set(), [], 0
    for row in rows:
        for j, target in enumerate(row["targets"]):
            key = tuple(sorted((row["query"], target)))
            if key[0] == key[1] or key in seen:
                continue
            seen.add(key)
            pairs.append((row["query"], target))
            target_index = row.get("target_indices", [])[j] if "target_indices" in row else index[target]
            if abs(row["query_index"] - target_index) > overlap:
                nonsequential += 1
    path.write_bytes("".join(f"{a} {b}\n" for a, b in pairs).encode("utf-8"))
    return {"path": str(path), "sha256": sha256(path), "pairs": len(pairs),
            "nonsequential_unique": nonsequential, "queries": len(rows),
            "p50_candidates": float(np.median([len(r["targets"]) for r in rows])) if rows else 0,
            "p95_candidates": float(np.percentile([len(r["targets"]) for r in rows], 95)) if rows else 0}


def log_tree_candidates(database_path: Path, tree_path: Path, names, output: Path,
                        *, pycolmap_module=None) -> dict:
    """Read unverified proposals directly from the installed vocabulary-tree generator."""
    names = list(names)
    if pycolmap_module is None:
        import pycolmap as pycolmap_module
    options = pycolmap_module.VocabTreePairingOptions()
    options.num_images = K
    options.num_nearest_neighbors = 1
    options.vocab_tree_path = Path(tree_path)
    with pycolmap_module.Database.open(database_path) as db:
        ids = {row.name: row.image_id for row in db.read_all_images()}
        by_id = {value: key for key, value in ids.items()}
        query_names = names[::CADENCE]
        generator = pycolmap_module.VocabTreePairGenerator(
            options, db, [ids[n] for n in query_names])
        proposed = generator.all_pairs()
        rows = [{"query": q, "query_index": i * CADENCE, "targets": [],
                 "target_indices": []}
            for i, q in enumerate(query_names)]
        lookup = {row["query"]: row for row in rows}
        name_index = {name: i for i, name in enumerate(names)}
        for a, b in proposed:
            an, bn = by_id[a], by_id[b]
            if an in lookup:
                lookup[an]["targets"].append(bn)
                lookup[an]["target_indices"].append(name_index[bn])
            elif bn in lookup:
                lookup[bn]["targets"].append(an)
                lookup[bn]["target_indices"].append(name_index[an])
    _json(output, {"generator": "pycolmap.VocabTreePairGenerator",
                   "pycolmap_version": pycolmap_module.__version__, "options": options.todict(),
                   "tree_sha256": sha256(tree_path), "rows": rows,
                   "note": "generator order is proposal order; no SIFT/two-view verification run"})
    return {"queries": len(rows), "proposals": sum(len(row["targets"]) for row in rows)}


def remove_nonsequential_pairs(database_path: Path, names, overlap: int = OVERLAP) -> int:
    """On the copied R database, remove prior tree links while retaining ±overlap pairs."""
    order = {name: i for i, name in enumerate(names)}
    con = sqlite3.connect(str(database_path))
    try:
        ids = {iid: order[name] for iid, name in con.execute("select image_id,name from images")
               if name in order}
        removed = 0
        for table in ("matches", "two_view_geometries"):
            for (pair_id,) in con.execute(f"select pair_id from {table}").fetchall():
                a, b = divmod(pair_id, 2147483647)
                if a in ids and b in ids and abs(ids[a] - ids[b]) > overlap:
                    con.execute(f"delete from {table} where pair_id=?", (pair_id,))
                    removed += 1
        con.commit()
        return removed
    finally:
        con.close()


def match_retrieval(pycolmap, database_path, images_dir: Path, names, workspace_root: Path,
                    matching, pairing, verification, match_sequential) -> dict:
    """R: same sequential geometry, with tree proposals replaced by frozen DINO proposals."""
    names = list(names)
    workspace_root = Path(workspace_root)
    checkpoint = checkpoint_file()
    vectors = descriptor_cache([Path(images_dir) / name for name in names],
                               workspace_root / "exp2-descriptors.json", sha256(checkpoint))
    rows = rank_candidates(names, vectors)
    manifest = write_candidate_manifest(rows, workspace_root / "exp2-candidates.txt",
                                        overlap=pairing.overlap)
    _json(workspace_root / "exp2-candidates.json", {"candidate_file": manifest,
          "checkpoint_sha256": sha256(checkpoint), "rows": rows})
    removed = remove_nonsequential_pairs(database_path, names, overlap=pairing.overlap)
    # Preserve every sequential setting except the S vocabulary-tree proposal step.
    previous_loop_detection = pairing.loop_detection
    pairing.loop_detection = False
    try:
        match_sequential(pycolmap, database_path, matching, pairing, verification)
    finally:
        pairing.loop_detection = previous_loop_detection
    imported = pycolmap.ImportedPairingOptions()
    imported.match_list_path = str(workspace_root / "exp2-candidates.txt")
    kwargs = {"matching_options": matching, "pairing_options": imported}
    if verification is not None:
        kwargs["verification_options"] = verification
    pycolmap.match_image_pairs(database_path, **kwargs)
    return dict(manifest, prior_nonsequential_rows_removed=removed)
