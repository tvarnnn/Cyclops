#!/usr/bin/env python
"""Render the best look at each sighting, so a human can check the label.

Every figure this cartridge has ever measured describes what the DETECTOR
said. "cell phone at a 0.844 median score" is a statement about
SSDLite320's confidence, not about whether a phone was there, and no
label exists anywhere in `data/captures/` to settle the difference. A
relevance policy built only on scores is a policy built on the detector's
opinion of itself.

This closes that gap the only way available without an annotation
budget: it crops the strongest frame of each sighting, lays the crops out
on a labelled sheet, and lets a person look. A hundred crops read in a
few minutes is a real precision estimate for the classes that matter, and
it is the only evidence in this repository about whether a high score
means a correct one.

    python scripts/research/sighting_contact_sheet.py \\
        --detections analysis/corpus-detections.jsonl \\
        --captures ../../Glasses/tower/data/captures \\
        --out analysis/sheets --classes laptop,remote,backpack

THE OUTPUT IS SENSITIVE IMAGERY. Crops come from raw first-person frames
recorded with `redaction: "none"`, and a crop is not safer than the frame
it came from -- `ios/Glasses/Project_Overview_Steps/docs/modules/OBJECT-MEMORY.md`
says exactly this. The output directory is under `analysis/`, which
`.gitignore` excludes, and these sheets must not be committed, published, or attached anywhere.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# A sighting survives this long a dropout before it is treated as a new
# one. Three seconds is roughly 36 delivered frames at the measured
# ~12 fps, and it is long enough to cover a head turn away and back.
GAP_SECONDS = 3.0


def sightings_of(rows, min_score: float, min_frames: int):
    """Group detections into temporally contiguous runs, per capture, per class."""
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        if row["score"] < min_score:
            continue
        grouped[row["capture_id"]][row["label"]].append(row)

    found = collections.defaultdict(list)
    for capture_id, labels in grouped.items():
        for label, detections in labels.items():
            detections.sort(
                key=lambda r: (
                    r["received_at"]
                    if r["received_at"] is not None
                    else r["source_seq"]
                )
            )
            run = [detections[0]]
            for previous, current in zip(detections, detections[1:]):
                before, after = previous["received_at"], current["received_at"]
                gap = (after - before) if (before and after) else 0.0
                if gap > GAP_SECONDS:
                    if len(run) >= min_frames:
                        found[label].append(run)
                    run = []
                run.append(current)
            if len(run) >= min_frames:
                found[label].append(run)
    return found


def crop_for(capture_root: Path, detection: dict, pad: float, size: int):
    """The detection's box, padded, letterboxed into a square tile.

    Padded because a tight crop of a small object is unreadable, and
    context is most of what tells a person whether the label is right --
    a 5%-of-frame box called `remote` is only checkable if the sofa
    around it is visible too.
    """
    path = capture_root / detection["capture_id"] / detection["relpath"]
    frame = cv2.imread(str(path))
    if frame is None:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection["box"]
    box_w, box_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1 = int(max(0, x1 - pad * box_w))
    y1 = int(max(0, y1 - pad * box_h))
    x2 = int(min(width, x2 + pad * box_w))
    y2 = int(min(height, y2 + pad * box_h))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = frame[y1:y2, x1:x2]

    scale = size / max(patch.shape[0], patch.shape[1])
    resized = cv2.resize(
        patch,
        (max(1, int(patch.shape[1] * scale)), max(1, int(patch.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    tile = np.full((size, size, 3), 30, dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    tile[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return tile


def sheet(tiles, captions, columns: int, size: int, header: int = 22):
    rows = (len(tiles) + columns - 1) // columns
    canvas = np.full(
        (rows * (size + header), columns * size, 3), 15, dtype=np.uint8
    )
    for index, (tile, caption) in enumerate(zip(tiles, captions)):
        r, c = divmod(index, columns)
        top = r * (size + header)
        left = c * size
        canvas[top + header : top + header + size, left : left + size] = tile
        cv2.putText(
            canvas,
            caption,
            (left + 3, top + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--classes", default="", help="Comma-separated; all if empty.")
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--pad", type=float, default=0.35)
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.detections.open(encoding="utf-8")]
    found = sightings_of(rows, args.min_score, args.min_frames)
    wanted = [c for c in args.classes.split(",") if c] or sorted(found)
    args.out.mkdir(parents=True, exist_ok=True)

    written = []
    for label in wanted:
        runs = found.get(label, [])
        if not runs:
            continue
        # Strongest-first, so a sheet of N shows the cases the policy
        # would MOST confidently remember. If the top of the list is
        # wrong, nothing below it is worth arguing about.
        runs = sorted(runs, key=lambda r: -max(d["score"] for d in r))
        tiles, captions = [], []
        for run in runs[: args.per_class]:
            best = max(run, key=lambda d: d["score"])
            tile = crop_for(args.captures, best, args.pad, args.size)
            if tile is None:
                continue
            tiles.append(tile)
            captions.append(
                f"{best['score']:.2f} {best['area_fraction'] * 100:.1f}% n={len(run)}"
            )
        if not tiles:
            continue
        path = args.out / f"{label.replace(' ', '_')}.png"
        cv2.imwrite(str(path), sheet(tiles, captions, args.columns, args.size))
        written.append({"label": label, "sightings": len(runs), "tiles": len(tiles),
                        "path": str(path)})

    print(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
