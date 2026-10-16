#!/usr/bin/env python3
"""World Builder Experiment 1 -- depth_temporal_consistency (offline analysis).

Measures how much the shipped MiDaS-small `depth` experiment's output
flickers frame-to-frame on real continuous motion, and how much a cheap
post-hoc smoothing filter (EMA / temporal median) reduces that flicker.

This is an OFFLINE dataset analysis, not a live Tower/WS run: the wire
protocol only carries the scalar mean_relative_depth, but this experiment
needs the full per-frame depth array. It drives `DepthEstimation` directly
with its opt-in `capture_depth_array` hook, reusing the shipped inference
path unmodified.

IMPORTANT -- what the numbers this prints do and do not mean:
results from a public dataset clip are *feasibility evidence*, NOT
physical-glasses/DAT validation. A public clip differs from the platform
in optics/FOV, encoding (DAT uses an adaptive bitrate ladder), resolution
and frame rate (resampling mitigates, not eliminates), and scene regime.
Acceptance requires a re-run on real DAT footage.

Usage:
    .venv\\Scripts\\python.exe scripts/depth_temporal_consistency.py \\
        --video path/to/clip.MP4 --frames 150 --target-fps 15

Requires the `ml` extra: pip install -e ".[dev,ml]"
"""
import argparse
import json
import time

import cv2
import numpy as np

from tower.experiments.depth import DepthEstimation

# The platform's target stream geometry (03-ROADMAP.md V0.7). Source
# footage is resampled toward this so the measurement reflects roughly the
# pixel budget the Tower would really see, rather than the dataset's native
# resolution. Aspect ratio is preserved (no distortion), so a landscape
# source becomes landscape at an equivalent pixel count.
TARGET_LONG_EDGE = 896
TARGET_SHORT_EDGE = 504

# Window for the local temporal-std metric, in sampled frames (~0.3s at
# the 16.67 fps effective rate) -- short enough that scene content is
# roughly stationary, so the statistic reflects estimator jitter.
LOCAL_STD_WINDOW = 5

# JPEG quality for the re-encode this harness does before handing frames
# to the experiment. Pinned rather than left at OpenCV's default 95 so the
# harness-introduced recompression is an explicit, reproducible parameter.
# NOTE: this is a real confound -- it stacks a second lossy encode on top
# of the dataset's existing H.264, and it is NOT the platform's adaptive
# bitrate ladder, which is the regime most likely to threaten this result.
JPEG_QUALITY = 95


def _resize_to_platform_budget(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if width >= height:
        target = (TARGET_LONG_EDGE, TARGET_SHORT_EDGE)
    else:
        target = (TARGET_SHORT_EDGE, TARGET_LONG_EDGE)
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


def _normalize_minmax(depth: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1].

    WARNING -- kept as the comparison baseline for the robust variants: the
    scale is set by two single pixels, so one flickering outlier shifts every
    pixel in the frame. Measured on a 128x256 grid, changing one pixel out of
    32768 moves the whole-frame mean absolute difference by ~0.33, several
    times the raw flicker signal being measured.
    """
    lo = float(depth.min())
    hi = float(depth.max())
    if hi - lo < 1e-9:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - lo) / (hi - lo)).astype(np.float32)


def _normalize_percentile(depth: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Percentile-clipped normalization -- robust to single-pixel outliers."""
    lo = float(np.percentile(depth, low))
    hi = float(np.percentile(depth, high))
    if hi - lo < 1e-9:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip((depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _normalize_median_mad(depth: np.ndarray) -> np.ndarray:
    """Median / median-absolute-deviation normalization.

    This is the alignment family MiDaS's own scale-and-shift-invariant
    literature uses (Ranftl et al.), rather than min-max. Rescaled into a
    roughly [0,1]-comparable band so its numbers sit on the same axis as
    the other two variants.
    """
    median = float(np.median(depth))
    mad = float(np.median(np.abs(depth - median)))
    if mad < 1e-9:
        return np.zeros_like(depth, dtype=np.float32)
    # 6 MADs ~ the central bulk of the distribution; clip the tails.
    return np.clip((depth - median) / (6.0 * mad) + 0.5, 0.0, 1.0).astype(np.float32)


NORMALIZERS = {
    "minmax": _normalize_minmax,
    "percentile_1_99": _normalize_percentile,
    "median_mad": _normalize_median_mad,
}


def _renormalize(frame: np.ndarray) -> np.ndarray:
    """Rescale a smoothed frame back to full [0,1] range.

    Without this, smoothing shrinks a frame's dynamic range, and a
    lower-amplitude signal trivially has smaller absolute frame-to-frame
    differences -- which would show up as "flicker reduction" that is
    really just amplitude compression. Re-normalizing removes that
    confound so the reported reduction reflects stabilization only.
    """
    lo = float(frame.min())
    hi = float(frame.max())
    if hi - lo < 1e-9:
        return np.zeros_like(frame, dtype=np.float32)
    return ((frame - lo) / (hi - lo)).astype(np.float32)


def _flicker_metrics(sequence: list[np.ndarray]) -> dict:
    """Frame-to-frame instability of a sequence of normalized depth maps.

    - mad: mean absolute difference between consecutive frames, in
      normalized depth units (0-1 scale). Lower = more temporally stable.
    - p95: 95th-percentile absolute per-pixel change, i.e. how bad the
      worst-flickering regions are, not just the average.
    - local_temporal_std: per-pixel standard deviation over a SHORT
      sliding window (5 frames), averaged. Deliberately not computed over
      the whole sequence: across ~9s of continuous head motion the scene
      content changes completely, so a whole-window std measures camera
      movement, not estimator stability, and no smoother could reduce it.
    """
    consecutive_mad = []
    consecutive_p95 = []
    for previous, current in zip(sequence, sequence[1:]):
        delta = np.abs(current - previous)
        consecutive_mad.append(float(delta.mean()))
        consecutive_p95.append(float(np.percentile(delta, 95)))

    local_stds = []
    for index in range(len(sequence) - LOCAL_STD_WINDOW + 1):
        window = np.stack(sequence[index : index + LOCAL_STD_WINDOW], axis=0)
        local_stds.append(float(window.std(axis=0).mean()))

    return {
        "mad_mean": round(float(np.mean(consecutive_mad)), 5),
        "mad_max": round(float(np.max(consecutive_mad)), 5),
        "p95_mean": round(float(np.mean(consecutive_p95)), 5),
        "local_temporal_std_mean": round(float(np.mean(local_stds)), 5),
    }


def _ema(sequence: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    """Causal exponential moving average -- usable in a live stream."""
    smoothed = [sequence[0]]
    for frame in sequence[1:]:
        smoothed.append(alpha * frame + (1.0 - alpha) * smoothed[-1])
    return smoothed


def _temporal_median(sequence: list[np.ndarray], window: int) -> list[np.ndarray]:
    """Causal trailing-window median -- also usable in a live stream."""
    smoothed = []
    for index in range(len(sequence)):
        start = max(0, index - window + 1)
        smoothed.append(np.median(np.stack(sequence[start : index + 1]), axis=0))
    return smoothed


def _deviation_from_raw(raw: list[np.ndarray], smoothed: list[np.ndarray]) -> float:
    """Mean |smoothed - raw| at the same index.

    NOT a lag measurement, despite the intuition. If raw = signal + noise
    and the smoother removes the noise perfectly with zero delay, this
    quantity equals the noise magnitude -- large -- even though lag is
    zero. It is also not comparable to `mad_mean`: that is a statistic of
    first differences, this is an instantaneous level difference.

    Reported only as "how much the smoother changed the output". For an
    actual lag figure see `_step_response_lag_frames`, which measures a
    delay in frames against a known injected step.
    """
    return round(
        float(np.mean([np.abs(s - r).mean() for r, s in zip(raw, smoothed)])), 5
    )


def _step_response_lag_frames(smoother, threshold: float = 0.632) -> float:
    """Frames for the smoother to reach `threshold` of a unit step.

    Lag is only measurable against a known input, so this uses a synthetic
    step (0 -> 1 at frame 10) rather than the depth data, which has no
    ground truth. 0.632 is the standard time-constant crossing (1 - 1/e).
    Returns the number of frames after the step to first reach it.
    """
    step = [np.zeros((4, 4), dtype=np.float32)] * 10 + [
        np.ones((4, 4), dtype=np.float32)
    ] * 40
    response = smoother(step)
    for index in range(10, len(response)):
        if float(response[index].mean()) >= threshold:
            return float(index - 10 + 1)
    return float("inf")


def run(video_path: str, frame_count: int, target_fps: float, device: str) -> dict:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    # Subsample toward the platform's ~15 FPS target so inter-frame motion
    # magnitude resembles what the Tower would actually receive. Sampling a
    # 50 FPS clip at every frame would understate flicker by making
    # consecutive frames unrealistically similar.
    stride = max(1, int(round(source_fps / target_fps))) if source_fps > 0 else 1

    experiment = DepthEstimation(capture_depth_array=True)
    experiment.load(device)

    raw_depths: list[np.ndarray] = []
    inference_ms: list[float] = []
    source_index = 0
    try:
        while len(raw_depths) < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index % stride != 0:
                source_index += 1
                continue
            source_index += 1

            resized = _resize_to_platform_budget(frame)
            ok_encode, encoded = cv2.imencode(
                ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok_encode:
                raise RuntimeError("failed to JPEG-encode a frame")

            started = time.perf_counter()
            result = experiment.run(encoded.tobytes())
            inference_ms.append((time.perf_counter() - started) * 1000)
            # Keep the un-normalized array: normalization strategy is a
            # variable this harness sweeps, not a fixed preprocessing step.
            raw_depths.append(np.asarray(experiment.last_depth_array, dtype=np.float32))
            del result
    finally:
        capture.release()
        experiment.release()

    if len(raw_depths) < 3:
        raise RuntimeError(f"only decoded {len(raw_depths)} frames; need at least 3")

    smoothers = {
        "ema_alpha_0.5": lambda seq: _ema(seq, 0.5),
        "ema_alpha_0.3": lambda seq: _ema(seq, 0.3),
        "median_window_3": lambda seq: _temporal_median(seq, 3),
        "median_window_5": lambda seq: _temporal_median(seq, 5),
    }

    # Lag depends only on the smoother, not on the depth data, so it is
    # measured once against a known synthetic step rather than per
    # normalizer.
    lag_frames = {
        name: _step_response_lag_frames(smoother)
        for name, smoother in smoothers.items()
    }

    by_normalizer = {}
    for norm_name, normalizer in NORMALIZERS.items():
        normalized = [normalizer(depth) for depth in raw_depths]
        baseline = _flicker_metrics(normalized)
        variants = {
            "raw_baseline": {"flicker": baseline, "deviation_from_raw": 0.0}
        }
        for name, smoother in smoothers.items():
            # Re-normalize so the comparison measures stabilization, not
            # the amplitude compression smoothing inevitably causes.
            smoothed = [_renormalize(f) for f in smoother(normalized)]
            variants[name] = {
                "flicker": _flicker_metrics(smoothed),
                "deviation_from_raw": _deviation_from_raw(normalized, smoothed),
            }

        for name, data in variants.items():
            if name == "raw_baseline" or baseline["mad_mean"] <= 0.0:
                data["mad_reduction_vs_raw_pct"] = 0.0
                continue
            data["mad_reduction_vs_raw_pct"] = round(
                100.0
                * (baseline["mad_mean"] - data["flicker"]["mad_mean"])
                / baseline["mad_mean"],
                2,
            )
        by_normalizer[norm_name] = variants

    steady_state = inference_ms[1:] if len(inference_ms) > 1 else inference_ms
    return {
        "video": video_path,
        "device": device,
        "source_fps": round(source_fps, 2),
        "sample_stride": stride,
        "effective_sampled_fps": round(source_fps / stride, 2) if source_fps else None,
        "frames_analyzed": len(raw_depths),
        # NOTE: this is the grid MiDaS-small's own transform produces, NOT
        # the platform's 504x896 budget. small_transform emits a fixed
        # 128x256 for any 16:9 input, so the input resize barely affects
        # this experiment (it does matter for feature_trackability, where
        # ORB sees the resized frame directly).
        "depth_map_shape": list(raw_depths[0].shape),
        "jpeg_quality": JPEG_QUALITY,
        "local_std_window": LOCAL_STD_WINDOW,
        "per_frame_experiment_ms_avg": round(sum(steady_state) / len(steady_state), 2),
        "per_frame_experiment_ms_note": "first frame excluded (warm-up)",
        "smoother_step_response_lag_frames": lag_frames,
        "by_normalizer": by_normalizer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="path to a real-motion video clip")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    # torch.hub prints "Loading weights: ..." to stdout during model load,
    # which corrupts stdout as a JSON stream. --out writes clean JSON.
    parser.add_argument("--out", help="write JSON here instead of stdout")
    args = parser.parse_args()

    report = run(args.video, args.frames, args.target_fps, args.device)
    payload = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        print(f"wrote {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
