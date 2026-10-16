#!/usr/bin/env python3
"""World Builder Experiment 2 -- feature_trackability (offline analysis).

Answers one narrow, load-bearing question: does ordinary, undirected
head-mounted motion produce frame pairs with enough shared, trackable
structure to make any multi-view geometry worth attempting at all?

Measures, for frame n vs n+k at several small k: ORB keypoint counts,
raw match counts, and RANSAC-geometrically-verified inlier counts/ratios,
plus per-pair cost.

Deliberately intrinsics-free. Geometric verification uses the fundamental
matrix (and a homography cross-check), neither of which needs a calibrated
camera. Essential-matrix `recoverPose` -- which WOULD need intrinsics -- is
Experiment 3 and is explicitly out of scope and blocked.

IMPORTANT -- results from a public dataset clip are *feasibility evidence*,
NOT physical-glasses/DAT validation. A public clip differs from the
platform in optics/FOV, encoding (DAT uses an adaptive bitrate ladder),
resolution and frame rate (resampling mitigates, not eliminates), and
scene regime. Acceptance requires a re-run on real DAT footage.

Usage:
    .venv\\Scripts\\python.exe scripts/feature_trackability.py \\
        --video path/to/clip.MP4 --frames 150 --target-fps 15

Uses only OpenCV -- no model download, no torch, no GPU.
"""
import argparse
import json
import time

import cv2
import numpy as np

# Same platform-budget resampling as the depth experiment, for the same
# reason -- see scripts/depth_temporal_consistency.py.
TARGET_LONG_EDGE = 896
TARGET_SHORT_EDGE = 504

ORB_FEATURES = 1000
LOWE_RATIO = 0.75
RANSAC_REPROJ_THRESHOLD = 3.0

# ORB-SLAM's homography-vs-fundamental model-selection threshold. Above
# this, the homography explains the pair well enough to indicate near-pure
# rotation or a dominant plane -- geometrically degenerate for
# triangulation even when match counts look healthy.
R_H_THRESHOLD = 0.45

# Frame gaps to test. k=1 is "consecutive frames as the Tower sees them";
# larger k probes how fast shared structure decays -- the thing that
# actually determines whether multi-view geometry has anything to work with.
FRAME_GAPS = (1, 2, 5, 10)


def _resize_to_platform_budget(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    target = (
        (TARGET_LONG_EDGE, TARGET_SHORT_EDGE)
        if width >= height
        else (TARGET_SHORT_EDGE, TARGET_LONG_EDGE)
    )
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


_EMPTY_PAIR = {
    "matches": 0,
    "inliers": 0,
    "inlier_ratio": 0.0,
    "homography_inliers": 0,
    "r_h": None,
}


def _match_pair(matcher, descriptors_a, descriptors_b, keypoints_a, keypoints_b) -> dict:
    """Ratio-test match + RANSAC fundamental-matrix verification."""
    if descriptors_a is None or descriptors_b is None:
        return dict(_EMPTY_PAIR)

    knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        nearest, second = pair
        if nearest.distance < LOWE_RATIO * second.distance:
            good.append(nearest)

    if len(good) < 8:  # 8-point algorithm minimum for the fundamental matrix
        return {**_EMPTY_PAIR, "matches": len(good)}

    points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good])
    points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good])

    _, mask = cv2.findFundamentalMat(
        points_a, points_b, cv2.FM_RANSAC, RANSAC_REPROJ_THRESHOLD, 0.99
    )
    inliers = int(mask.sum()) if mask is not None else 0

    _, h_mask = cv2.findHomography(
        points_a, points_b, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD
    )
    homography_inliers = int(h_mask.sum()) if h_mask is not None else 0

    # ORB-SLAM's model-selection heuristic: R_H = S_H / (S_H + S_F).
    # Raw H-vs-F inlier counts are NOT directly comparable -- the
    # fundamental-matrix residual is point-to-epipolar-LINE (a 1-D
    # constraint) while the homography residual is point-to-POINT (2-D),
    # so F is strictly the weaker test and H <= F almost always,
    # regardless of scene geometry. R_H with an explicit threshold
    # (ORB-SLAM uses ~0.45, above which it selects the homography model,
    # indicating near-pure rotation or a dominant plane) is the
    # established comparison; a bare H/F ratio is not.
    denominator = homography_inliers + inliers
    r_h = round(homography_inliers / denominator, 4) if denominator else None

    return {
        "matches": len(good),
        "inliers": inliers,
        "inlier_ratio": round(inliers / len(good), 4),
        "homography_inliers": homography_inliers,
        "r_h": r_h,
    }


def _summarize(values: list[float], precision: int = 2) -> dict:
    array = np.array(values, dtype=float)
    return {
        "mean": round(float(array.mean()), precision),
        "median": round(float(np.median(array)), precision),
        "min": round(float(array.min()), precision),
        "max": round(float(array.max()), precision),
    }


def run(video_path: str, frame_count: int, target_fps: float) -> dict:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    stride = max(1, int(round(source_fps / target_fps))) if source_fps > 0 else 1

    detector = cv2.ORB_create(nfeatures=ORB_FEATURES)

    grays: list[np.ndarray] = []
    keypoints_all = []
    descriptors_all = []
    detect_ms: list[float] = []
    source_index = 0

    try:
        while len(grays) < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index % stride != 0:
                source_index += 1
                continue
            source_index += 1

            # Timer spans resize + grayscale + detectAndCompute, i.e. all
            # per-frame CV work this experiment performs. Timing only
            # detectAndCompute would not be comparable to the depth
            # experiment's figure, which includes its own decode/preprocess.
            started = time.perf_counter()
            gray = cv2.cvtColor(_resize_to_platform_budget(frame), cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            detect_ms.append((time.perf_counter() - started) * 1000)
            grays.append(gray)
            keypoints_all.append(keypoints)
            descriptors_all.append(descriptors)
    finally:
        capture.release()

    if len(grays) < max(FRAME_GAPS) + 1:
        raise RuntimeError(f"only decoded {len(grays)} frames; need more")

    keypoint_counts = [len(k) for k in keypoints_all]

    # Constructed once, outside the timed region: rebuilding it per pair
    # inflated the reported match cost.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    by_gap = {}
    for gap in FRAME_GAPS:
        matches, inliers, ratios, homography_inliers, pair_ms = [], [], [], [], []
        r_h_values = []
        for index in range(len(grays) - gap):
            started = time.perf_counter()
            stats = _match_pair(
                matcher,
                descriptors_all[index],
                descriptors_all[index + gap],
                keypoints_all[index],
                keypoints_all[index + gap],
            )
            pair_ms.append((time.perf_counter() - started) * 1000)
            matches.append(stats["matches"])
            inliers.append(stats["inliers"])
            ratios.append(stats["inlier_ratio"])
            homography_inliers.append(stats["homography_inliers"])
            if stats["r_h"] is not None:
                r_h_values.append(stats["r_h"])

        # A pair needs enough verified inliers to constrain geometry at all.
        # 30 is a commonly used practical floor for a usable two-view
        # relationship; reported as a fraction of pairs clearing it rather
        # than as a pass/fail claim about the platform.
        usable = sum(1 for value in inliers if value >= 30)
        rotation_dominant = sum(1 for value in r_h_values if value >= R_H_THRESHOLD)
        by_gap[f"k={gap}"] = {
            "pairs": len(inliers),
            "matches": _summarize(matches),
            "verified_inliers": _summarize(inliers),
            "inlier_ratio": _summarize(ratios, precision=4),
            "homography_inliers": _summarize(homography_inliers),
            "r_h": _summarize(r_h_values, precision=4) if r_h_values else None,
            "pairs_rotation_dominant_pct": (
                round(100.0 * rotation_dominant / len(r_h_values), 2)
                if r_h_values
                else None
            ),
            "pairs_with_ge_30_inliers_pct": round(100.0 * usable / len(inliers), 2),
            "match_ms": _summarize(pair_ms),
        }

    return {
        "video": video_path,
        "source_fps": round(source_fps, 2),
        "sample_stride": stride,
        "effective_sampled_fps": round(source_fps / stride, 2) if source_fps else None,
        "frames_analyzed": len(grays),
        "frame_shape": list(grays[0].shape),
        "orb_nfeatures": ORB_FEATURES,
        "keypoints_per_frame": _summarize(keypoint_counts),
        "detect_ms_per_frame": _summarize(detect_ms),
        # Excludes the first frame, matching the depth harness's warm-up
        # convention so the two experiments' cost figures are comparable.
        "detect_ms_per_frame_excluding_warmup": _summarize(detect_ms[1:]),
        "detect_ms_scope": "resize + grayscale + ORB detectAndCompute",
        "by_frame_gap": by_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--out", help="write JSON here instead of stdout")
    args = parser.parse_args()

    payload = json.dumps(run(args.video, args.frames, args.target_fps), indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        print(f"wrote {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
