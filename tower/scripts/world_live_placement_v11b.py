"""CPU-only walk replay for bounded local chaining and an early ORB seed.

No final pose enters either candidate decision. Final poses are read only by
the retrospective error scorer. The seed frame is local until a global solve
can establish a common frame; seed coverage is reported separately.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from scripts import world_live_placement as base
from tower.world_builder.backend import KeyframeInput
from tower.world_builder.backends.classical import ClassicalTwoViewBackend
from tower.world_builder.geometry import detect_and_describe
from tower.world_builder.records import CameraIntrinsics


def local_pose(pose):
    if pose.status == "anchor":
        return np.zeros(3), np.eye(3)
    if pose.rotation is None or pose.translation is None:
        return None
    rotation = np.asarray(pose.rotation, np.float64).T
    return -rotation @ np.asarray(pose.translation, np.float64), rotation


def transform_from_two(older, newer):
    """Use the latest PnP orientation and two causal baselines for scale."""
    a, b = older, newer
    baseline_local = b[1][0] - a[1][0]
    baseline_global = b[2][0] - a[2][0]
    local_length = float(np.linalg.norm(baseline_local))
    global_length = float(np.linalg.norm(baseline_global))
    if local_length < 0.08 or global_length < 0.08:
        return None, "unobservable-scale"
    rotation = b[2][1] @ b[1][1].T
    scale = global_length / local_length
    if not 0.05 <= scale <= 30.0:
        return None, "scale-out-of-bounds"
    bearing = rotation @ baseline_local / local_length
    agreement = math.degrees(math.acos(float(np.clip(np.dot(bearing, baseline_global / global_length), -1, 1))))
    if agreement > 15.0:
        return None, "anchor-disagreement"
    return (scale, rotation), None


def chain_pose(anchors, local, accepted_at, index, *, max_age_s=2.0, max_travel=0.5, max_steps=6):
    if len(anchors) < 2:
        return None, "fewer-than-two-anchors"
    recent = anchors[-1]
    if accepted_at - recent[3] > max_age_s or index - recent[4] > max_steps:
        return None, "drift-budget-expired"
    fit, reason = transform_from_two(anchors[-2], recent)
    if reason:
        return None, reason
    scale, rotation = fit
    travel = scale * float(np.linalg.norm(local[0] - recent[1][0]))
    if travel > max_travel:
        return None, "drift-budget-expired"
    return (recent[2][0] + scale * rotation @ (local[0] - recent[1][0]),
            rotation @ local[1]), None


def seed_pnp(estimate, earlier_features, query_features, camera, *, max_refs=8, max_points=3000):
    """Small causal ORB map from earlier same-segment frames only."""
    points = estimate.points
    if points is None or points.support_views is None or len(points.xyz) < 30:
        return None, "seed-too-small", 0
    support = points.support_views
    reference_indices = sorted(set(int(v) for v in support[:, 0]))[-max_refs:]
    keep_points = set(range(min(max_points, len(points.xyz))))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    query_kp, query_desc = query_features
    if query_desc is None or len(query_desc) < 2:
        return None, "seed-no-overlap", 0
    offers = []
    for ref_index in reference_indices:
        if ref_index not in earlier_features:
            continue
        ref_kp, ref_desc = earlier_features[ref_index]
        if ref_desc is None:
            continue
        rows = [(int(feature), int(point)) for frame, feature, point in support
                if int(frame) == ref_index and int(point) in keep_points and int(feature) < len(ref_desc)]
        if len(rows) < 2:
            continue
        desc = np.ascontiguousarray(ref_desc[[feature for feature, _ in rows]])
        for pair in matcher.knnMatch(query_desc, desc, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                hit = pair[0]
                offers.append((hit.distance, hit.queryIdx, rows[hit.trainIdx][1], str(ref_index)))
    matches, seen_q, seen_p = [], set(), set()
    for distance, qi, point, ref in sorted(offers):
        if qi not in seen_q and point not in seen_p:
            matches.append((qi, point, ref, distance))
            seen_q.add(qi)
            seen_p.add(point)
    candidates = base.pnp_candidates(query_kp, matches, points.xyz,
                                     np.zeros(len(points.xyz), np.int32), camera)
    reason = base.pose_veto(candidates, tracking_break=False)
    if reason:
        return None, reason if matches else "seed-no-overlap", len(matches)
    winner = max(candidates, key=lambda c: c["inliers"])
    return (winner["centre"], winner["rotation"]), None, len(matches)


def orientation_alignment(local_poses, final_poses):
    """Diagnostic Sim(3) constrained by earlier shared camera orientations."""
    cross = sum((final_rotation @ local_rotation.T
                 for (_, local_rotation), (_, final_rotation) in zip(local_poses, final_poses)),
                np.zeros((3, 3)))
    u, _, vt = np.linalg.svd(cross)
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    old = np.asarray([centre for centre, _ in local_poses])
    new = np.asarray([centre for centre, _ in final_poses])
    old_centered = old - old.mean(axis=0)
    new_centered = new - new.mean(axis=0)
    scale = float(np.sum(new_centered * (old_centered @ rotation.T)) /
                  np.sum(old_centered ** 2))
    if scale <= 0:
        raise ValueError("orientation alignment has nonpositive scale")
    return scale, rotation, new.mean(axis=0) - scale * rotation @ old.mean(axis=0)


def run(replay: Path, baseline: Path, *, first_map_only=False):
    meta = json.loads((replay / "run.json").read_text())
    session_file = Path(json.loads((replay / "client.json").read_text())["settle"]["session_file"])
    session_dir = session_file.parent
    solve_dir = Path(meta["data_root"]) / "world_builder" / "worlds" / session_dir.parents[1].name / "solve" / session_dir.name
    final = json.loads((solve_dir / "solution.json").read_text())
    session = json.loads(session_file.read_text())
    intrinsics_row = dict(session["intrinsics"])
    intrinsics_row["dist_coeffs"] = tuple(intrinsics_row["dist_coeffs"])
    intrinsics = CameraIntrinsics(**intrinsics_row)
    seed_camera = {"fx": intrinsics.fx, "fy": intrinsics.fy,
                   "cx": intrinsics.cx, "cy": intrinsics.cy,
                   "width": intrinsics.calibrated_width,
                   "height": intrinsics.calibrated_height}
    journal = [json.loads(line) for line in (session_dir / "keyframes.jsonl").read_text().splitlines()]
    frames = [json.loads(line) for line in (replay / "live-timeline-frames.jsonl").read_text().splitlines()]
    clocks = {f["keyframe_id"]: f["keyframe_accepted_at"] for f in frames
              if f.get("keyframe_id") and f.get("keyframe_accepted_at")}
    baseline_rows = {r["keyframe_id"]: r for r in json.loads(baseline.read_text())["rows"]}
    snapshot_dir = replay / "solution-snapshots"
    snapshots = json.loads((snapshot_dir / "index.json").read_text())
    for snapshot in snapshots:
        snapshot["doc"] = json.loads((snapshot_dir / snapshot["file"]).read_text())
        snapshot["keyframe_ids"] = set(snapshot["doc"]["keyframe_ids"])
    reference_cache = {}
    backend = ClassicalTwoViewBackend()
    backend.begin(intrinsics)
    current_segment = None
    local_history = {}
    local_features = {}
    segment_keys = []
    anchors = []
    segment_index = 0
    rows = []
    first_map_at = snapshots[0]["seen_at"]
    for frame in journal:
        key = frame["keyframe_id"]
        if key not in clocks:
            continue
        at = clocks[key]
        if first_map_only and at >= first_map_at:
            break
        segment = frame["segment_index"]
        if segment != current_segment:
            backend.reset()
            current_segment = segment
            segment_index = 0
            local_history = {}
            local_features = {}
            segment_keys = []
            anchors = []
        image = cv2.imread(str(session_dir / frame["image_relpath"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(frame["image_relpath"])
        local_start = time.perf_counter()
        # A seed query uses the map at t-1, then enters the map at t.
        query_features = detect_and_describe(image) if at < first_map_at else None
        seed = None
        seed_reason = None
        seed_matches = 0
        seed_ms = None
        if at < first_map_at:
            seed_reference_history = dict(local_history)
            seed_start = time.perf_counter()
            seed, seed_reason, seed_matches = seed_pnp(
                backend.snapshot(), local_features, query_features,
                seed_camera)
            seed_ms = (time.perf_counter() - seed_start) * 1000
        step = backend.extend(KeyframeInput(keyframe_id=key, image_gray=image))
        segment_keys.append(key)
        estimate = backend.snapshot()
        local_history = {prior_key: local_pose(pose)
                         for prior_key, pose in zip(segment_keys, estimate.poses)}
        anchors = [(anchor_key, local_history[anchor_key], global_pose, anchor_at, anchor_index, component)
                   for anchor_key, _, global_pose, anchor_at, anchor_index, component in anchors
                   if local_history.get(anchor_key) is not None]
        local = local_history[key]
        local_ms = (time.perf_counter() - local_start) * 1000
        if query_features is not None:
            local_features[segment_index] = query_features
        row = {"keyframe_id": key, "accepted_at": at, "segment": segment,
               "local_status": step.pose.status, "local_ms": round(local_ms, 3),
               "baseline": baseline_rows.get(key, {}).get("outcome"),
               "seed_reason": seed_reason, "seed_matches": seed_matches,
               "seed_ms": round(seed_ms, 3) if seed_ms is not None else None}
        if key in final["poses"]:
            snap = base.causal_snapshot(snapshots, key, at)
            pnp = None
            if baseline_rows.get(key, {}).get("outcome") == "accepted":
                if snap is None:
                    raise AssertionError("accepted baseline lacks causal map")
                if snap["file"] not in reference_cache:
                    xyz, components, refs = base.reference_features(snap, snapshot_dir)
                    reference_cache[snap["file"]] = (xyz, components, [
                        (name, desc.astype(np.float32), points, component)
                        for name, desc, points, component in refs])
                xyz, components, refs = reference_cache[snap["file"]]
                pnp_start = time.perf_counter()
                pnp_kp, pnp_desc = cv2.SIFT_create().detectAndCompute(
                    cv2.imread(str(solve_dir / "images" / (key.rsplit(":", 1)[-1] + ".jpg")), cv2.IMREAD_GRAYSCALE), None)
                selected = base.retrieve(pnp_desc, refs)
                matches = base.unique_matches(pnp_desc, selected)
                candidates = base.pnp_candidates(pnp_kp, matches, xyz, components, snap["doc"]["camera"])
                if base.pose_veto(candidates, tracking_break=False) is None:
                    winner = max(candidates, key=lambda c: c["inliers"])
                    pnp = (winner["centre"], winner["rotation"])
                    row["pnp_component"] = winner["component"]
                row["pnp_recompute_ms"] = round((time.perf_counter() - pnp_start) * 1000, 3)
                if pnp is None:
                    row["pnp_recompute_mismatch"] = True
                else:
                    row["pnp_pose"] = [pnp[0].tolist(), pnp[1].tolist()]
            if pnp is not None and local is not None:
                anchors.append((key, local, pnp, at, segment_index, row["pnp_component"]))
            elif pnp is None and local is not None and snap is not None:
                chained, reason = chain_pose(anchors, local, at, segment_index)
                row["chain_reason"] = reason
                if chained is not None:
                    row["chain_pose"] = [chained[0].tolist(), chained[1].tolist()]
                    try:
                        transform, residual, shared = base.alignment_for(
                            snap["doc"], final, anchors[-1][5],
                            final["poses"][key]["component"], clocks, at, key)
                        row["chain_error"] = base.pose_error(transform, *chained,
                                                              *base.entry_pose(final["poses"][key]))
                    except (ValueError, KeyError):
                        row["chain_comparison"] = "unalignable"
            elif pnp is None and snap is not None:
                row["chain_reason"] = "local-pose-refused"
            if seed is not None:
                row["seed_pose"] = [seed[0].tolist(), seed[1].tolist()]
                if local is not None:
                    row["seed_local_rotation_delta_deg"] = base.rotation_angle(
                        seed[1] @ local[1].T)
                # A retrospective alignment of the early local seed frame.
                # It is comparison-only and never used to accept a pose.
                shared = [k for k, pose in seed_reference_history.items()
                          if k != key and pose is not None and k in final["poses"]]
                if len(shared) >= 4:
                    old = np.asarray([seed_reference_history[k][0] for k in shared])
                    new = np.asarray([base.entry_pose(final["poses"][k])[0] for k in shared])
                    try:
                        transform = base.fit_sim3(old, new)
                        row["seed_error"] = base.pose_error(transform, *seed,
                                                             *base.entry_pose(final["poses"][key]))
                        row["seed_alignment_p95"] = base.alignment_residuals(transform, old, new)["p95"]
                        reference_rotation_errors = [base.rotation_angle(
                            transform[1] @ seed_reference_history[k][1]
                            @ base.entry_pose(final["poses"][k])[1].T) for k in shared]
                        row["seed_reference_rotation_p50_deg"] = base.percentile(reference_rotation_errors, 50)
                        row["seed_reference_rotation_p95_deg"] = base.percentile(reference_rotation_errors, 95)
                        orientation_transform = orientation_alignment(
                            [seed_reference_history[k] for k in shared],
                            [base.entry_pose(final["poses"][k]) for k in shared])
                        row["seed_orientation_aligned_error"] = base.pose_error(
                            orientation_transform, *seed,
                            *base.entry_pose(final["poses"][key]))
                        row["seed_orientation_aligned_ref_p95"] = base.alignment_residuals(
                            orientation_transform, old, new)["p95"]
                    except ValueError:
                        row["seed_comparison"] = "degenerate-alignment"
        rows.append(row)
        segment_index += 1
    return {"schema": "offline-live-placement-v11b/1", "rows": rows,
            "parameters": {"chain_max_age_s": 2, "chain_max_steps": 6,
                           "chain_max_travel_solve_units": 0.5,
                           "seed_max_refs": 8, "seed_max_points": 3000}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-map-only", action="store_true")
    args = parser.parse_args()
    result = run(args.run, args.baseline, first_map_only=args.first_map_only)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(result["rows"]), "seed": sum("seed_pose" in r for r in result["rows"]),
                      "chain": sum("chain_pose" in r for r in result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
