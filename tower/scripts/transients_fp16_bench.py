"""fp32 vs fp16 bench for the transient detectors (OneFormer, Grounding DINO + SAM).

Reuses `tower.world_builder.transients.GroundingDinoSamBackend` /
`OneFormerBackend` and their `run()` -- the exact per-frame pipeline
`ensure_transient_masks` drives -- over a frozen set of walk-8 keyframes read
straight off disk. No store, no session, no Tower: this script only needs
the two backends and the images.

Each invocation runs ONE (component, precision) pair over the frame set and
caches its per-frame masks and aggregate timings under `--out-dir`
(`_raw_<component>_<precision>.npz` + a `.json` sidecar). `--finalize` reads
whichever of the four pairs are present, computes per-frame IoU / detection-
count deltas / lost detections between fp32 and fp16, and writes `BENCH.md`
+ `bench.json`. Splitting the run this way keeps one invocation comfortably
under the ~15 minute window a timed replay needs to be able to preempt
between (see the campaign's `gpulock.py`, which this script is always run
behind -- it does not take the lock itself).

Usage:

    python transients_fp16_bench.py --component oneformer --precision fp32
    python transients_fp16_bench.py --component oneformer --precision fp16
    python transients_fp16_bench.py --component gdsam --precision fp32
    python transients_fp16_bench.py --component gdsam --precision fp16
    python transients_fp16_bench.py --finalize
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

TOWER_ROOT = Path(__file__).resolve().parents[1]
if str(TOWER_ROOT) not in sys.path:
    sys.path.insert(0, str(TOWER_ROOT))

from tower import config  # noqa: E402
from tower.world_builder import transients as T  # noqa: E402

DEFAULT_KEYFRAMES_DIR = Path(
    r"C:\Users\tvllo\Projects\Glasses-scratch\wbcpt\w8-r0q-d\world_builder\worlds"
    r"\4fc0b91286d54e9598d18045f1698661\solve\539ac7d413cd465f9819348df7ea0328\images"
)
DEFAULT_OUT_DIR = Path(
    r"C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\experiments"
    r"\W8LOOP\fp16bench"
)

COMPONENTS = (T.COMPONENT_GDSAM, T.COMPONENT_ONEFORMER)
PRECISIONS = ("fp32", "fp16")
BACKEND_CLASSES = {
    T.COMPONENT_GDSAM: T.GroundingDinoSamBackend,
    T.COMPONENT_ONEFORMER: T.OneFormerBackend,
}


# ---------------------------------------------------------------------------
# frame selection + loading
# ---------------------------------------------------------------------------


def select_frames(keyframes_dir: Path, n: int) -> list[tuple[int, Path]]:
    """`n` frames spread evenly across every jpg in `keyframes_dir`, in walk
    order. `ki` is the frame's own id (its filename, as the pipeline names
    keyframes), not the position in this subsample."""
    paths = sorted(keyframes_dir.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no jpgs under {keyframes_dir}")
    if n >= len(paths):
        chosen = paths
    else:
        idx = np.linspace(0, len(paths) - 1, n)
        idx = sorted({int(round(i)) for i in idx})
        chosen = [paths[i] for i in idx]
    return [(int(p.stem), p) for p in chosen]


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


# ---------------------------------------------------------------------------
# one (component, precision) pass
# ---------------------------------------------------------------------------


def _raw_path(out_dir: Path, component: str, precision: str) -> Path:
    return out_dir / f"_raw_{component}_{precision}.npz"


def _meta_path(out_dir: Path, component: str, precision: str) -> Path:
    return out_dir / f"_raw_{component}_{precision}.json"


def run_pass(component: str, precision: str, frames: list[tuple[int, Path]], out_dir: Path,
            time_budget_sec: float | None) -> None:
    os.environ[config.WORLD_TRANSIENTS_FP16_ENV] = "on" if precision == "fp16" else "off"

    rgbs = [load_rgb(path) for _, path in frames]
    items = [(ki, rgb, np.zeros(rgb.shape[:2], bool)) for (ki, _path), rgb in zip(frames, rgbs)]
    backend = BACKEND_CLASSES[component]()
    params = T.TransientParams()

    per_ki_hand: dict[int, np.ndarray] = {}
    per_ki_phone: dict[int, np.ndarray] = {}
    per_ki_seconds: dict[int, float] = {}

    def emit(ki, hand, phone, seconds):
        per_ki_hand[ki] = np.packbits(np.asarray(hand, bool), axis=-1)
        per_ki_phone[ki] = np.packbits(np.asarray(phone, bool), axis=-1)
        per_ki_seconds[ki] = float(seconds)

    t0 = time.time()
    should_stop = None
    if time_budget_sec is not None:
        deadline = t0 + time_budget_sec

        def should_stop():
            return time.time() > deadline

    wall0 = time.time()
    timings = backend.run(items, params, emit, should_stop=should_stop)
    wall = time.time() - wall0

    shape = rgbs[0].shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        _raw_path(out_dir, component, precision),
        ki=np.array(sorted(per_ki_hand)),
        **{f"hand_{ki}": per_ki_hand[ki] for ki in per_ki_hand},
        **{f"phone_{ki}": per_ki_phone[ki] for ki in per_ki_phone},
        shape=np.array(shape),
    )
    _meta_path(out_dir, component, precision).write_text(json.dumps({
        "component": component,
        "precision": precision,
        "requested_frames": len(frames),
        "completed_frames": len(per_ki_hand),
        "stopped_early": bool((timings or {}).get("stopped")),
        "wall_seconds": round(wall, 3),
        "per_ki_seconds": per_ki_seconds,
        "backend_timings": {k: v for k, v in (timings or {}).items() if k != "stopped"},
    }, indent=2))
    print(f"[{component}/{precision}] {len(per_ki_hand)}/{len(frames)} frames in {wall:.1f}s "
          f"(stopped_early={bool((timings or {}).get('stopped'))})")


# ---------------------------------------------------------------------------
# finalize: compare fp32 vs fp16
# ---------------------------------------------------------------------------


def _load_raw(out_dir: Path, component: str, precision: str):
    p = _raw_path(out_dir, component, precision)
    m = _meta_path(out_dir, component, precision)
    if not p.is_file() or not m.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    meta = json.loads(m.read_text())
    shape = tuple(int(v) for v in z["shape"])
    hand, phone = {}, {}
    for ki in z["ki"]:
        ki = int(ki)
        hand[ki] = np.unpackbits(z[f"hand_{ki}"], axis=-1)[:, :shape[1]].astype(bool)
        phone[ki] = np.unpackbits(z[f"phone_{ki}"], axis=-1)[:, :shape[1]].astype(bool)
    return {"hand": hand, "phone": phone, "meta": meta, "shape": shape}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks; both empty counts as a perfect match (the
    bench's own pass rule, not a general convention)."""
    inter = int((a & b).sum())
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return inter / union


def _n_components(mask: np.ndarray) -> int:
    import cv2  # noqa: PLC0415

    if not mask.any():
        return 0
    n, _ = cv2.connectedComponents(mask.astype(np.uint8))
    return n - 1  # cv2 counts background as label 0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def finalize(out_dir: Path) -> None:
    raws = {(c, p): _load_raw(out_dir, c, p) for c in COMPONENTS for p in PRECISIONS}
    present = {k: v for k, v in raws.items() if v is not None}
    if not present:
        raise SystemExit(f"no _raw_*.npz found under {out_dir}; run the four passes first")

    report: dict = {"components": {}}
    md_lines = ["# fp16 autocast bench -- transient detectors", "",
                f"Keyframes: `{DEFAULT_KEYFRAMES_DIR}`", ""]

    overall_pass = True
    for component in COMPONENTS:
        fp32 = raws.get((component, "fp32"))
        fp16 = raws.get((component, "fp16"))
        if fp32 is None or fp16 is None:
            md_lines.append(f"## {component}: INCOMPLETE (missing "
                            f"{'fp32' if fp32 is None else 'fp16'} pass)")
            md_lines.append("")
            continue

        kis = sorted(set(fp32["hand"]) & set(fp16["hand"]))
        hand_ious, phone_ious = [], []
        hand_deltas, phone_deltas = [], []
        lost_hand, lost_phone = [], []
        for ki in kis:
            h32, h16 = fp32["hand"][ki], fp16["hand"][ki]
            p32, p16 = fp32["phone"][ki], fp16["phone"][ki]
            hand_ious.append(_iou(h32, h16))
            phone_ious.append(_iou(p32, p16))
            c32h, c16h = _n_components(h32), _n_components(h16)
            c32p, c16p = _n_components(p32), _n_components(p16)
            hand_deltas.append(c16h - c32h)
            phone_deltas.append(c16p - c32p)
            if h32.any() and not h16.any():
                lost_hand.append(ki)
            if p32.any() and not p16.any():
                lost_phone.append(ki)

        meta32, meta16 = fp32["meta"], fp16["meta"]
        gdino_s = None
        sam_s = None
        oneformer_s = None
        if component == T.COMPONENT_GDSAM:
            t32, t16 = meta32["backend_timings"], meta16["backend_timings"]
            gdino_s = (t32.get("gdino", 0) / max(1, t32.get("frames", 1)),
                      t16.get("gdino", 0) / max(1, t16.get("frames", 1)))
            sam_s = (t32.get("sam", 0) / max(1, t32.get("frames_with_boxes", 1)),
                    t16.get("sam", 0) / max(1, t16.get("frames_with_boxes", 1)))
        else:
            t32, t16 = meta32["backend_timings"], meta16["backend_timings"]
            oneformer_s = (t32.get("oneformer", 0) / max(1, t32.get("frames", 1)),
                          t16.get("oneformer", 0) / max(1, t16.get("frames", 1)))

        hand_med, hand_p5 = statistics.median(hand_ious), _percentile(hand_ious, 5)
        phone_med, phone_p5 = statistics.median(phone_ious), _percentile(phone_ious, 5)
        comp_pass = (hand_med >= 0.95 and hand_p5 >= 0.85 and phone_med >= 0.95
                    and phone_p5 >= 0.85 and not lost_hand and not lost_phone)
        overall_pass = overall_pass and comp_pass

        report["components"][component] = {
            "n_frames": len(kis),
            "hand_iou_median": hand_med, "hand_iou_p5": hand_p5,
            "phone_iou_median": phone_med, "phone_iou_p5": phone_p5,
            "hand_component_delta_mean": statistics.fmean(hand_deltas) if hand_deltas else None,
            "phone_component_delta_mean": statistics.fmean(phone_deltas) if phone_deltas else None,
            "lost_hand_frames": lost_hand,
            "lost_phone_frames": lost_phone,
            "fp32_seconds_per_frame": {"gdino": gdino_s and gdino_s[0], "sam": sam_s and sam_s[0],
                                       "oneformer": oneformer_s and oneformer_s[0]},
            "fp16_seconds_per_frame": {"gdino": gdino_s and gdino_s[1], "sam": sam_s and sam_s[1],
                                       "oneformer": oneformer_s and oneformer_s[1]},
            "pass": comp_pass,
        }

        md_lines.append(f"## {component}")
        md_lines.append("")
        md_lines.append(f"- frames compared: {len(kis)}")
        if gdino_s is not None:
            md_lines.append(f"- gdino s/frame: fp32 {gdino_s[0]:.3f}  fp16 {gdino_s[1]:.3f}  "
                            f"(x{(gdino_s[0] / gdino_s[1]) if gdino_s[1] else float('nan'):.2f})")
        if sam_s is not None:
            md_lines.append(f"- sam s/frame (frames with boxes): fp32 {sam_s[0]:.3f}  "
                            f"fp16 {sam_s[1]:.3f}  "
                            f"(x{(sam_s[0] / sam_s[1]) if sam_s[1] else float('nan'):.2f})")
        if oneformer_s is not None:
            md_lines.append(f"- oneformer s/frame: fp32 {oneformer_s[0]:.3f}  "
                            f"fp16 {oneformer_s[1]:.3f}  "
                            f"(x{(oneformer_s[0] / oneformer_s[1]) if oneformer_s[1] else float('nan'):.2f})")
        md_lines.append(f"- hand IoU: median {hand_med:.4f}  p5 {hand_p5:.4f}")
        md_lines.append(f"- phone IoU: median {phone_med:.4f}  p5 {phone_p5:.4f}")
        md_lines.append(f"- hand detection-count delta (fp16 - fp32), mean: "
                        f"{report['components'][component]['hand_component_delta_mean']:.3f}")
        md_lines.append(f"- phone detection-count delta (fp16 - fp32), mean: "
                        f"{report['components'][component]['phone_component_delta_mean']:.3f}")
        md_lines.append(f"- lost hand detections: {lost_hand or 'none'}")
        md_lines.append(f"- lost phone detections: {lost_phone or 'none'}")
        md_lines.append(f"- PASS: {comp_pass}")
        md_lines.append("")

    report["pass"] = overall_pass and bool(report["components"])
    md_lines.insert(2, f"**Overall: {'PASS' if report['pass'] else 'FAIL'}**")
    md_lines.insert(3, "")

    (out_dir / "BENCH.md").write_text("\n".join(md_lines), encoding="utf-8")
    (out_dir / "bench.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'BENCH.md'} and {out_dir / 'bench.json'}; PASS={report['pass']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keyframes-dir", type=Path, default=DEFAULT_KEYFRAMES_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-frames", type=int, default=150)
    ap.add_argument("--component", choices=COMPONENTS)
    ap.add_argument("--precision", choices=PRECISIONS)
    ap.add_argument("--time-budget-sec", type=float, default=None,
                    help="stop early (should_stop) once this many seconds have elapsed")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()

    if args.finalize:
        finalize(args.out_dir)
        return

    if not args.component or not args.precision:
        raise SystemExit("pass --component and --precision, or --finalize")

    frames = select_frames(args.keyframes_dir, args.n_frames)
    print(f"{len(frames)} frames selected from {args.keyframes_dir}")
    run_pass(args.component, args.precision, frames, args.out_dir, args.time_budget_sec)


if __name__ == "__main__":
    main()
