"""Offline, CPU-only causal placement evaluation for flagged live replay snapshots.

The final solution is read solely for comparison after a candidate has been
decided. This script never writes to the replay store or a Tower data root.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def causal_snapshot(rows: list[dict], keyframe_id: str, accepted_at: float) -> dict | None:
    return next((row for row in reversed(rows)
                 if row["seen_at"] < accepted_at and keyframe_id not in row["keyframe_ids"]), None)


def unique_matches(query: np.ndarray, references: list[tuple]) -> list[tuple]:
    """Ratio-test each reference, then keep one query feature and one landmark."""
    if query is None or len(query) == 0:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    offers = []
    for reference in references:
        descriptors, point_ids, ref_id = reference[:3]
        component = reference[3] if len(reference) > 3 else None
        if len(descriptors) < 2:
            continue
        for pair in matcher.knnMatch(query, descriptors, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                hit = pair[0]
                offers.append((hit.distance, hit.queryIdx, int(point_ids[hit.trainIdx]), ref_id, component))
    result, queries, points = [], set(), set()
    for distance, qi, point, ref, component in sorted(offers):
        if (component, qi) not in queries and point not in points:
            result.append((qi, point, ref, distance))
            queries.add((component, qi))
            points.add(point)
    return result


def pose_veto(candidates: list[dict], *, tracking_break: bool,
              motion: dict | None = None) -> str | None:
    if not candidates:
        return "no-overlap"
    winner = max(candidates, key=lambda c: c["inliers"])
    if winner["inliers"] < 30:
        return "insufficient-3d-inliers"
    if len(winner["references"]) < (3 if tracking_break else 2):
        return "insufficient-independent-references"
    if winner["cells"] < 4:
        return "clustered-inliers"
    if winner["positive_fraction"] < 0.95:
        return "negative-depth"
    if winner["p95_px"] > 4.0:
        return "reprojection-tail"
    for rival in candidates:
        if rival is winner or rival["inliers"] < 0.75 * winner["inliers"]:
            continue
        separation = float(np.linalg.norm(rival["centre"] - winner["centre"]))
        angle = rotation_angle(rival["rotation"] @ winner["rotation"].T)
        if rival["component"] != winner["component"] or separation > 1.0 or angle > 10.0:
            return "ambiguous-pose"
    if motion is not None and np.linalg.norm(winner["centre"] - motion["centre"]) > motion["max_distance"]:
        return "motion-disagreement"
    return None


def rotation_angle(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def fit_sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Robust Sim(3) from earlier shared camera centres; source -> target."""
    if len(source) < 4:
        raise ValueError("at least four earlier shared centres required")
    keep = np.ones(len(source), bool)
    for _ in range(4):
        a, b = source[keep], target[keep]
        am, bm = a.mean(axis=0), b.mean(axis=0)
        centred = a - am
        if np.linalg.matrix_rank(centred, tol=1e-7) < 2:
            raise ValueError("degenerate alignment centres")
        u, singular, vt = np.linalg.svd((b - bm).T @ centred / len(a))
        reflection = np.diag([1., 1., np.linalg.det(u @ vt)])
        rotation = u @ reflection @ vt
        scale = float(np.sum(singular * np.diag(reflection)) / np.mean(np.sum(centred ** 2, axis=1)))
        translation = bm - scale * rotation @ am
        residual = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)
        cutoff = max(0.1, 3 * float(np.median(residual)))
        new_keep = residual <= cutoff
        if new_keep.sum() < 4 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return scale, rotation, translation


def transform_centres(transform, centres):
    scale, rotation, translation = transform
    return scale * (np.asarray(centres) @ rotation.T) + translation


def percentile(values, fraction):
    return float(np.percentile(values, fraction)) if len(values) else None


def distribution(values):
    return {"n": len(values), **{f"p{p}": percentile(values, p) for p in (50, 90, 95, 99)},
            "max": max(values) if values else None}


def alignment_residuals(transform, source, target):
    return distribution(np.linalg.norm(transform_centres(transform, source) - target, axis=1).tolist())


def pose_error(transform, centre, rotation_wc, final_centre, final_rotation_wc):
    aligned_centre = transform_centres(transform, centre)
    aligned_rotation = transform[1] @ rotation_wc
    return {"position": float(np.linalg.norm(aligned_centre - final_centre)),
            "rotation_deg": rotation_angle(aligned_rotation @ final_rotation_wc.T),
            "forward_deg": math.degrees(math.acos(float(np.clip(
                np.dot(aligned_rotation[:, 2], final_rotation_wc[:, 2]), -1, 1))))}


def entry_pose(entry):
    rotation_cw = np.asarray(entry["rotation"], np.float64).reshape(3, 3)
    translation_cw = np.asarray(entry["translation"], np.float64)
    rotation_wc = rotation_cw.T
    return -rotation_wc @ translation_cw, rotation_wc


def reference_features(snapshot: dict, directory: Path, *, max_refs=200):
    """Mapped feature indices only; no unobserved or later database images."""
    doc = snapshot["doc"]
    with np.load(directory / snapshot["array_file"], allow_pickle=False) as arrays:
        xyz = arrays["xyz"].copy()
        components = arrays["component"].copy()
        observations = arrays["observations"].copy()
        observation_xy = arrays["observation_xy"].copy()
    by_image = defaultdict(dict)
    for (image_idx, feature_idx, point_idx), xy in zip(observations, observation_xy):
        key = doc["keyframe_ids"][int(image_idx)]
        if key in doc["poses"]:
            by_image[key][int(feature_idx)] = (int(point_idx), xy)
    refs = sorted(by_image, key=lambda k: (-len(by_image[k]), k))
    # Bound descriptor read and coarse matching cost. Prefer well observed
    # views, then spread the budget over the full mapped horizon.
    if len(refs) > max_refs:
        refs = [refs[i] for i in np.linspace(0, len(refs) - 1, max_refs, dtype=int)]
    db = sqlite3.connect((directory / snapshot["descriptor_file"]).as_uri() + "?mode=ro", uri=True)
    try:
        names = {name: image_id for image_id, name in db.execute("SELECT image_id,name FROM images")}
        found = []
        for key in refs:
            image_id = names.get(key.rsplit(":", 1)[-1] + ".jpg")
            if image_id is None:
                continue
            row = db.execute("SELECT rows,cols,data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
            if row is None or row[1] != 128 or len(row[2]) != row[0] * row[1]:
                continue
            keypoint_row = db.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
            if (keypoint_row is None or keypoint_row[0] != row[0] or keypoint_row[1] < 2
                    or len(keypoint_row[2]) != keypoint_row[0] * keypoint_row[1] * 4):
                continue
            all_desc = np.frombuffer(row[2], np.uint8).reshape(row[0], 128)
            all_xy = np.frombuffer(keypoint_row[2], np.float32).reshape(keypoint_row[0], keypoint_row[1])[:, :2]
            indices = np.array([i for i, (_, xy) in by_image[key].items()
                                if i < len(all_desc) and np.linalg.norm(all_xy[i] - xy) <= 2.0], np.int32)
            if len(indices) < 2:
                continue
            point_ids = np.array([by_image[key][int(i)][0] for i in indices], np.int32)
            found.append((key, all_desc[indices].copy(), point_ids,
                          int(doc["poses"][key]["component"])))
    finally:
        db.close()
    return xyz, components, found


def retrieve(query_desc, refs, *, max_candidates=8):
    """Sample at most 32 mapped descriptors per reference; return top views."""
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    scores = []
    for key, descriptors, points, component in refs:
        sample = descriptors[np.linspace(0, len(descriptors) - 1, min(32, len(descriptors)), dtype=int)]
        hits = matcher.knnMatch(query_desc, sample, k=2) if len(sample) >= 2 else []
        score = sum(len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance for pair in hits)
        if score:
            scores.append((score, key, descriptors, points, component))
    scores.sort(key=lambda x: (-x[0], x[1]))
    # Reserve one slot for every supported component before filling by score.
    # Otherwise a repetitive dominant room can hide the competing pose.
    chosen, seen_components, seen_keys = [], set(), set()
    for item in scores:
        if item[4] not in seen_components and len(chosen) < max_candidates:
            chosen.append(item)
            seen_components.add(item[4])
            seen_keys.add(item[1])
    for item in scores:
        if len(chosen) >= max_candidates:
            break
        if item[1] not in seen_keys:
            chosen.append(item)
            seen_keys.add(item[1])
    return [(d, p, key, component) for _, key, d, p, component in chosen]


def pnp_candidates(keypoints, matches, xyz, components, camera):
    intrinsic = np.array([[camera["fx"], 0, camera["cx"]],
                          [0, camera["fy"], camera["cy"]], [0, 0, 1]], np.float64)
    output = []
    for component in sorted(set(int(components[m[1]]) for m in matches)):
        group = [m for m in matches if components[m[1]] == component]
        for _ in range(2):  # a second supported pose in the same component vetoes the first
            if len(group) < 6:
                break
            obj = np.asarray([xyz[m[1]] for m in group], np.float64)
            img = np.asarray([keypoints[m[0]].pt for m in group], np.float64)
            try:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj, img, intrinsic, None, iterationsCount=2000, reprojectionError=4.0,
                    confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
            except cv2.error:
                break
            if not ok or inliers is None or len(inliers) < 6:
                break
            chosen = inliers[:, 0]
            rvec, tvec = cv2.solvePnPRefineLM(obj[chosen], img[chosen], intrinsic, None, rvec, tvec)
            rotation, _ = cv2.Rodrigues(rvec)
            projected, _ = cv2.projectPoints(obj[chosen], rvec, tvec, intrinsic, None)
            residual = np.linalg.norm(projected.reshape(-1, 2) - img[chosen], axis=1)
            depth = (obj[chosen] @ rotation[2, :] + tvec[2, 0])
            cells = {(min(3, int(x / camera["width"] * 4)),
                      min(3, int(y / camera["height"] * 4))) for x, y in img[chosen]}
            output.append({"component": component, "inliers": len(chosen),
                           "references": {group[int(i)][2] for i in chosen}, "cells": len(cells),
                           "positive_fraction": float(np.mean(depth > 0)), "p95_px": percentile(residual, 95),
                           "centre": -rotation.T @ tvec[:, 0], "rotation": rotation.T})
            inlier_set = set(int(i) for i in chosen)
            group = [m for i, m in enumerate(group) if i not in inlier_set]
    return output


def alignment_for(doc, final, component, final_component, accepted_by_id, accepted_at, query_id):
    shared = [key for key, pose in doc["poses"].items()
              if key != query_id and pose["component"] == component
              and key in final["poses"] and final["poses"][key]["component"] == final_component
              and accepted_by_id.get(key, float("inf")) < accepted_at]
    old = np.asarray([entry_pose(doc["poses"][key])[0] for key in shared])
    new = np.asarray([entry_pose(final["poses"][key])[0] for key in shared])
    return fit_sim3(old, new), alignment_residuals(fit_sim3(old, new), old, new), len(shared)


def evaluate(run: Path, *, max_refs=200, max_candidates=8):
    run = Path(run)
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    world_root = Path(metadata["data_root"]) / "world_builder"
    session_path = Path(json.loads((run / "client.json").read_text(encoding="utf-8"))["settle"]["session_file"])
    solve_dir = world_root / "worlds" / session_path.parents[2].name / "solve" / session_path.parent.name
    final = json.loads((solve_dir / "solution.json").read_text(encoding="utf-8"))
    snapshot_dir = run / "solution-snapshots"
    snapshots = json.loads((snapshot_dir / "index.json").read_text(encoding="utf-8"))
    for item in snapshots:
        if "array_file" not in item or "descriptor_file" not in item:
            raise ValueError("unpaired snapshot; rerun with --capture-placement-snapshots")
        item["doc"] = json.loads((snapshot_dir / item["file"]).read_text(encoding="utf-8"))
        item["keyframe_ids"] = set(item["doc"]["keyframe_ids"])
    frames = [json.loads(line) for line in (run / "live-timeline-frames.jsonl").read_text(encoding="utf-8").splitlines()]
    accepted = {row["keyframe_id"]: row for row in frames if row.get("keyframe_id") and row.get("keyframe_accepted_at")}
    accepted_at = {key: row["keyframe_accepted_at"] for key, row in accepted.items()}
    journal = [json.loads(line) for line in (session_path.parent / "keyframes.jsonl").read_text(encoding="utf-8").splitlines()]
    segments = {item["keyframe_id"]: item["segment_index"] for item in journal}
    first_in_segment = {}
    for key, at in accepted_at.items():
        segment = segments.get(key)
        if segment is not None and (segment not in first_in_segment or at < accepted_at[first_in_segment[segment]]):
            first_in_segment[segment] = key
    cache = {}
    outcomes, errors, alignment_errors, cpu_ms = [], defaultdict(list), [], []
    for key in final["poses"]:
        row = {"keyframe_id": key, "accepted_at": accepted_at.get(key)}
        if key not in accepted:
            row["outcome"] = "missing-acceptance-clock"
            outcomes.append(row)
            continue
        snap = causal_snapshot(snapshots, key, row["accepted_at"])
        if snap is None:
            row["outcome"] = "no-map"
            outcomes.append(row)
            continue
        row["snapshot"] = snap["file"]
        start = time.perf_counter()
        if snap["file"] not in cache:
            cache[snap["file"]] = reference_features(snap, snapshot_dir, max_refs=max_refs)
        xyz, components, refs = cache[snap["file"]]
        image_path = solve_dir / "images" / (key.rsplit(":", 1)[-1] + ".jpg")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            row["outcome"] = "missing-query-image"
            outcomes.append(row)
            continue
        keypoints, descriptors = cv2.SIFT_create().detectAndCompute(image, None)
        selected = retrieve(descriptors, refs, max_candidates=max_candidates) if descriptors is not None else []
        matches = unique_matches(descriptors, selected)
        candidates = pnp_candidates(keypoints, matches, xyz, components, snap["doc"]["camera"])
        # The segment's first accepted view is an independent anchor. The
        # local journal supplies no common-frame motion prior for later views.
        segment = segments.get(key)
        tracking_break = segment is None or first_in_segment.get(segment) == key
        reason = pose_veto(candidates, tracking_break=tracking_break)
        row.update(cpu_ms=round((time.perf_counter() - start) * 1000, 3), matches=len(matches),
                   retrieved_refs=len(selected), tracking_break=tracking_break)
        cpu_ms.append(row["cpu_ms"])
        if reason:
            row["outcome"] = "no-overlap" if reason == "no-overlap" or not matches else reason
            outcomes.append(row)
            continue
        chosen = max(candidates, key=lambda c: c["inliers"])
        row.update(outcome="accepted", component=chosen["component"], inliers=chosen["inliers"],
                   references=len(chosen["references"]), reprojection_p95_px=chosen["p95_px"])
        try:
            transform, residual, n = alignment_for(snap["doc"], final, chosen["component"],
                                                    final["poses"][key]["component"], accepted_at,
                                                    row["accepted_at"], key)
            comparison = pose_error(transform, chosen["centre"], chosen["rotation"],
                                    *entry_pose(final["poses"][key]))
            row.update(error=comparison, alignment_shared=n, alignment_residual=residual)
            for metric, value in comparison.items():
                errors[metric].append(value)
            alignment_errors.append(residual["p95"])
        except ValueError:
            row["comparison"] = "insufficient-earlier-alignment"
        outcomes.append(row)
    counts = Counter(row["outcome"] for row in outcomes)
    return {"schema": "offline-live-placement/1", "run": str(run), "final_posed": len(final["poses"]),
            "counts": dict(counts), "accepted_fraction": counts["accepted"] / len(final["poses"]),
            "cpu_ms": distribution(cpu_ms), "errors": {k: distribution(v) for k, v in errors.items()},
            "alignment_p95_solve_units": distribution(alignment_errors),
            "units": "solve units; final solve is comparison, not physical ground truth",
            "limitations": ["offline CPU time excludes queue, status delivery and client display",
                            "no independent physical position truth", "gates are sweep candidates, not validated"],
            "rows": outcomes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-refs", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args(argv)
    if args.max_refs < 3 or args.max_candidates < 3:
        parser.error("retrieval bounds must be at least three")
    report = evaluate(args.run, max_refs=args.max_refs, max_candidates=args.max_candidates)
    args.out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
