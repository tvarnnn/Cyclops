"""Offline, fail-closed scale-repair feasibility probe for the walk-8 copies.

Reads the published solve and final guard audit. Never opens a COLMAP database
through pycolmap. Refused pieces stay untouched in the candidate solution.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np


def triangulate(observations, centers, rotations, K):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for image_idx, xy in observations:
        ray = rotations[image_idx].T @ np.array([(xy[0] - K[2]) / K[0],
                                                   (xy[1] - K[3]) / K[1], 1.0])
        ray /= np.linalg.norm(ray)
        projector = np.eye(3) - np.outer(ray, ray)
        A += projector
        b += projector @ centers[image_idx]
    if np.linalg.cond(A) > 1e7:
        return None
    xyz = np.linalg.solve(A, b)
    for image_idx, xy in observations:
        camera = rotations[image_idx] @ (xyz - centers[image_idx])
        if camera[2] <= 0:
            return None
        projected = np.array([K[0] * camera[0] / camera[2] + K[2],
                              K[1] * camera[1] / camera[2] + K[3]])
        if np.linalg.norm(projected - xy) > 3.0:
            return None
    return xyz


def similarity(src, dst):
    src0, dst0 = src.mean(0), dst.mean(0)
    u, _, vh = np.linalg.svd((dst - dst0).T @ (src - src0))
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vh)])
    rotation = u @ correction @ vh
    scale = np.sum((dst - dst0) * ((src - src0) @ rotation.T)) / np.sum((src - src0)**2)
    shift = dst0 - scale * rotation @ src0
    return float(scale), rotation, shift


def robust_similarity(src, dst, *, threshold, trials=300):
    if len(src) < 6:
        return None
    rng = np.random.default_rng(20261012)
    best = np.zeros(len(src), dtype=bool)
    for _ in range(trials):
        sample = rng.choice(len(src), 3, replace=False)
        try:
            scale, rotation, shift = similarity(src[sample], dst[sample])
        except np.linalg.LinAlgError:
            continue
        if not 0.01 < scale < 100:
            continue
        residuals = np.linalg.norm(scale * src @ rotation.T + shift - dst, axis=1)
        inliers = residuals < threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 6:
        return None
    scale, rotation, shift = similarity(src[best], dst[best])
    residuals = np.linalg.norm(scale * src @ rotation.T + shift - dst, axis=1)
    return scale, rotation, shift, residuals, residuals < threshold


def probe(run: str, out: Path):
    source = Path.cwd() / run
    solve_path = next(source.rglob("solution.json"))
    solve = json.loads(solve_path.read_text())
    audit = json.loads((solve_path.parent / "final_scale_guard.json").read_text())["assessment"]
    data = np.load(solve_path.parent / "solution.npz")
    obs = data["observations"]
    xy = data["observation_xy"]
    ids = solve["keyframe_ids"]
    poses = solve["poses"]
    label = np.array([poses.get(k, {}).get("component", -1) for k in ids])
    R, C = {}, {}
    for idx, kid in enumerate(ids):
        if kid in poses:
            R[idx] = np.asarray(poses[kid]["rotation"]).reshape(3, 3)
            C[idx] = -R[idx].T @ np.asarray(poses[kid]["translation"])
    camera = solve["camera"]
    K = (camera["fx"], camera["fy"], camera["cx"], camera["cy"])
    metres_per_unit = float(audit["pose"]["metres_per_unit"])
    room = np.flatnonzero(label == 0)
    room_set = set(room)
    db = solve_path.parent / "database.db"
    con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    image_ids = {name: image_id for image_id, name in con.execute("select image_id,name from images")}
    room_image_ids = {image_ids[f"{int(ids[i].split(':')[-1]):08d}.jpg"] for i in room
                      if f"{int(ids[i].split(':')[-1]):08d}.jpg" in image_ids}
    verified = list(con.execute("select pair_id,rows from two_view_geometries "
                                "where rows>0 and config not in (0,1)"))
    con.close()
    per_point = defaultdict(list)
    for (image_idx, _feature_idx, point_idx), uv in zip(obs, xy):
        if int(image_idx) in C:
            per_point[int(point_idx)].append((int(image_idx), uv))
    pieces = []
    for number, piece in enumerate(audit["pieces"]):
        piece_indices = {ids.index(k) for k in piece["keyframe_ids"]}
        piece_image_ids = {image_ids[f"{int(ids[i].split(':')[-1]):08d}.jpg"] for i in piece_indices
                           if f"{int(ids[i].split(':')[-1]):08d}.jpg" in image_ids}
        boundary_pairs = []
        for pair_id, rows in verified:
            b = pair_id % 2147483647
            a = (pair_id - b) // 2147483647
            if (a in room_image_ids and b in piece_image_ids) or (b in room_image_ids and a in piece_image_ids):
                boundary_pairs.append(rows)
        local, fixed = [], []
        shared = 0
        for track in per_point.values():
            a = [x for x in track if x[0] in piece_indices]
            b = [x for x in track if x[0] in room_set]
            if a and b:
                shared += 1
            if len(a) < 2 or len(b) < 2:
                continue
            p = triangulate(a, C, R, K)
            q = triangulate(b, C, R, K)
            if p is not None and q is not None:
                local.append(p)
                fixed.append(q)
        src = np.asarray(local).reshape(-1, 3)
        dst = np.asarray(fixed).reshape(-1, 3)
        fit = robust_similarity(src, dst, threshold=0.15 / metres_per_unit)
        ratio = piece["ratio"]
        desired_scale = 1.0 / ratio if ratio else None
        row = {"piece": number, "keyframes": len(piece["keyframe_ids"]), "ratio": ratio,
               "shared_tracks": shared, "verified_boundary_pairs": len(boundary_pairs),
               "verified_boundary_inliers": sum(boundary_pairs), "triangulated_matches": len(src),
               "desired_scale": desired_scale, "fitted_scale": None,
               "median_residual_m": None, "forced_scale_residual_m": None,
               "inliers": 0, "post_ratio": None,
               "accepted": False, "why": "insufficient valid independent correspondences"}
        if fit is not None:
            scale, rotation, shift, residuals, inliers = fit
            post_ratio = ratio * scale
            median_m = float(np.median(residuals) * metres_per_unit)
            anchor_src = src[inliers]
            anchor_dst = dst[inliers]
            forced_shift = anchor_dst.mean(0) - desired_scale * rotation @ anchor_src.mean(0)
            forced_residual_m = float(np.median(np.linalg.norm(
                desired_scale * anchor_src @ rotation.T + forced_shift - anchor_dst, axis=1)) * metres_per_unit)
            row.update(fitted_scale=scale, median_residual_m=median_m,
                       forced_scale_residual_m=forced_residual_m,
                       inliers=int(inliers.sum()), post_ratio=post_ratio)
            if not 0.8 <= scale / desired_scale <= 1.2:
                row["why"] = "shared-solve alignment cannot validate depth scale"
            elif not 0.75 <= post_ratio <= 1.333:
                row["why"] = "alignment scale fails depth certification band"
            elif inliers.sum() < 30 or inliers.mean() < 0.6 or median_m > 0.15:
                row["why"] = "alignment residual or support fails"
            else:
                # A partial transform requires splitting every crossing 3D track,
                # rebuilding observations, and bundle adjustment. Do not publish a
                # camera-only correction from this feasibility probe.
                row["why"] = "track-splitting and bundle adjustment not validated"
        pieces.append(row)
    out.mkdir(parents=True, exist_ok=True)
    result_dir = out / run.lstrip("_")
    result_dir.mkdir(exist_ok=True)
    candidate = result_dir / "solution.json"
    shutil.copyfile(solve_path, candidate)
    shutil.copyfile(solve_path.parent / "solution.npz", result_dir / "solution.npz")
    shutil.copyfile(solve_path.parent / "components.json", result_dir / "components.json")
    # Both candidate files are byte-equivalent geometry, since every unsafe
    # correction is refused. This is intentionally a diagnostic candidate.
    report = {"run": run, "status": "unchanged-refused", "pieces": pieces,
              "components": [c["images_supported"] for c in solve["components"]],
              "posed_fraction": len(poses) / len(ids),
              "supported_fraction": sum(p["observations"] >= 30 for p in poses.values()) / len(ids),
              "scale_status": audit["scale_status"], "rejoined": 0}
    (result_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    scatter(solve, audit, result_dir / "camera_centres_before_after.png")
    return report


def scatter(solve, audit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = solve["keyframe_ids"]
    stretches = ((73, 91), (104, 135), (141, 148), (160, 167), (174, 188), (201, 209))
    centers = []
    groups = []
    for kid in ids:
        p = solve["poses"].get(kid)
        if not p or p["observations"] < 30:
            continue
        R = np.asarray(p["rotation"]).reshape(3, 3)
        centers.append(-R.T @ np.asarray(p["translation"]))
        walk_second = int(kid.split(":")[-1]) / 24
        groups.append(next((i+1 for i, (start, end) in enumerate(stretches)
                            if start <= walk_second <= end), 0 if p["component"] == 0 else -1))
    centers = np.array(centers)
    groups = np.array(groups)
    core = centers[groups == 0]
    _, _, vh = np.linalg.svd(core - core.mean(0), full_matrices=False)
    horizontal = centers @ vh[:2].T
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, title in zip(axes, ("Before", "After refused correction")):
        for group in sorted(set(groups)):
            mask = groups == group
            color = "#999999" if group < 0 else "#1f77b4" if group == 0 else plt.cm.tab10((group - 1) % 10)
            ax.scatter(horizontal[mask, 0], horizontal[mask, 1], s=6, alpha=.7,
                       color=color, label="other" if group < 0 else "room" if group == 0 else
                       f"stretch {group}: {stretches[group-1][0]}-{stretches[group-1][1]}s")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("walk horizontal axis 1 (SfM units)")
        ax.set_ylabel("walk horizontal axis 2 (SfM units)")
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path.with_name("camera_centres_overview.png"), dpi=160)
    corep = horizontal[groups == 0]
    low = np.quantile(corep, .05, axis=0) - 3
    high = np.quantile(corep, .95, axis=0) + 3
    for ax in axes:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("_w8g2", "_w8g2b"), required=True)
    parser.add_argument("--output", type=Path, default=Path("_out"))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((Path.cwd() / "_out").resolve()):
        parser.error("output must stay under cwd/_out")
    print(json.dumps(probe(args.run, args.output), indent=2))
