#!/usr/bin/env python
"""Can an open-vocabulary model tell a remote from a laptop keyboard?

THE QUESTION, AND WHY IT IS THIS ONE.

Reading the crops the shipped detector produced over the real corpus
found that its labels are unreliable for exactly the classes this
cartridge is for. A ceiling fan is `airplane` at 0.99 and `scissors` at
0.93. A white door is `refrigerator` at 0.95. The three highest-scoring
`remote` sightings in 18,821 frames are all laptop keyboards.

So widening the class list requires a SECOND OPINION, and the only
honest way to choose one is to measure it against crops whose true label
is known. `data/captures/` carries no annotation, so the labels here come
from a human reading contact sheets
(`scripts/research/sighting_contact_sheet.py`) -- 82 crops from classes
where every inspected tile was right or every one was wrong, plus the
two mixed classes that matter most, labelled tile by tile.

That is a small set and it is stated as one. It is also the only ground
truth about this corpus that exists anywhere in this repository.

WHAT IS ASKED OF EACH MODEL.

Not "detect everything". The funnel has already produced a crop and a
proposed label; the question is whether the proposed label survives. So
each model is given the crop and a fixed vocabulary of object names, and
is judged on whether the proposed name comes FIRST. That is exactly the
shape `VerificationQueue` needs, and it is a much easier question than
open-set detection -- which is the point: the expensive stage should be
asked the narrowest question that answers the need.

    python scripts/research/open_vocab_verifier_bench.py \\
        --detections analysis/corpus-detections.jsonl \\
        --captures ../../Glasses/tower/data/captures \\
        --out analysis/verifier-bench.json

It reads the corpus and writes only its report and its contact sheets.
The sheets contain crops of raw first-person imagery and must not be
committed; `analysis/` is gitignored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scripts.research.sighting_contact_sheet import (  # noqa: E402
    crop_for,
    sheet,
    sightings_of,
)
from tower.object_memory.classes import prompt_for  # noqa: E402

# THE VOCABULARY IS THE SHIPPED ONE, imported rather than restated.
#
# A restated copy drifts from the shipped list (a restated one ran 34 words
# where `verifier_vocabulary()` returned 31), and the accuracy figures
# would then describe a configuration that does not ship. Importing removes
# that class of error.
#
# Two prompts are ADDED for the benchmark and only for it: the labelled
# set contains classes the shipped policy ignores entirely (`airplane`,
# `tie`), and a verifier that cannot even CONSIDER a proposed label
# rejects it for free. A rejection rate measured that way would be a
# measurement of the vocabulary. They are listed in the report so the
# difference is visible.
def bench_vocabulary():
    from tower.object_memory.classes import prompt_for, verifier_vocabulary

    shipped = list(verifier_vocabulary())
    added = [
        prompt_for(name)
        for name in GOLDEN
        if prompt_for(name) not in shipped
    ]
    return tuple(shipped + sorted(set(added))), tuple(sorted(set(added)))


# The labelled set, by class and by rank within class -- strongest
# sighting first, which is the order `sighting_contact_sheet.py` lays
# them out in and the order they were read in.
#
# True means the detector's label was RIGHT for that crop. Classes where
# every inspected tile agreed are given as a count; the two mixed classes
# that matter are given tile by tile.
GOLDEN = {
    "laptop": [True] * 24,
    "cell phone": [True] * 24,
    "cup": [True] * 3,
    "bottle": [True] * 2,
    # Every one of these was wrong. The score in brackets is the
    # strongest tile, to make the point that they are not marginal.
    "airplane": [False] * 7,      # a ceiling fan (0.99)
    "scissors": [False] * 4,      # the same ceiling fan (0.93)
    "refrigerator": [False] * 6,  # a white interior door (0.95)
    "microwave": [False] * 2,     # a monitor showing a logo (0.71)
    "tie": [False] * 2,           # a door frame and window blinds (0.84)
    "book": [False],              # a laptop screen (0.66)
    "backpack": [False],          # a closet of hanging clothes (0.66)
    "toothbrush": [False],        # a boxed tube of toothpaste (0.61)
    "suitcase": [False] * 5,      # a backpack being carried -- right object, wrong name
    # The class this cartridge most needs, labelled tile by tile. The
    # first three are the HIGHEST-scoring sightings in the corpus.
    "remote": [False, False, False, False, True, False, True, True],
    "mouse": [True, True, True, False],  # the fourth is an AirPods case
}

MODELS = {
    "owlv2-base": "google/owlv2-base-patch16-ensemble",
    "llmdet-tiny": "iSEE-Laboratory/llmdet_tiny",
}


def golden_crops(rows, captures: Path, size: int, pad: float):
    """Every labelled crop, with the label the detector proposed and the truth."""
    found = sightings_of(rows, min_score=0.5, min_frames=3)
    items = []
    for object_class, truths in GOLDEN.items():
        runs = sorted(
            found.get(object_class, []),
            key=lambda run: -max(d["score"] for d in run),
        )
        for rank, truth in enumerate(truths):
            if rank >= len(runs):
                break
            best = max(runs[rank], key=lambda d: d["score"])
            tile = crop_for(captures, best, pad, size)
            if tile is None:
                continue
            items.append(
                {
                    "proposed": object_class,
                    "prompt": prompt_for(object_class),
                    "truth": truth,
                    "score": best["score"],
                    "area_fraction": best["area_fraction"],
                    "capture_id": best["capture_id"],
                    "relpath": best["relpath"],
                    "tile": tile,
                }
            )
    return items


class OwlV2:
    """Per-query detection: one score per prompt, and nothing else.

    The shape a verifier wants. Every prompt is embedded separately and
    matched against the image's object queries, so "a wallet" scores
    against the picture rather than against a tokenised sentence.
    """

    def __init__(self, repo, device):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.repo, self.device = repo, device
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(repo)
            .eval()
            .to(device)
        )

    def scores(self, image_rgb, vocabulary):
        import torch
        from PIL import Image

        image = Image.fromarray(image_rgb)
        inputs = self.processor(
            text=[list(vocabulary)], images=image, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=0.0, target_sizes=[(image.height, image.width)]
        )[0]
        best = {name: 0.0 for name in vocabulary}
        for index, score in zip(
            results["labels"].tolist(), results["scores"].tolist()
        ):
            name = vocabulary[int(index)]
            best[name] = max(best[name], float(score))
        return best


class GroundingStyle:
    """Phrase grounding: one sentence in, spans out.

    The output is a list of TEXT SPANS the model matched, which is not
    the same thing as a score per class -- a prompt list joined with
    full stops comes back split into "a set", "a pair" and "a". Mapping
    the spans back onto the vocabulary is the only interface this family
    offers, and how lossy that mapping is is part of what is measured
    here rather than a flaw in the harness.
    """

    def __init__(self, repo, device):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.repo, self.device = repo, device
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(repo)
            .eval()
            .to(device)
        )

    def scores(self, image_rgb, vocabulary):
        import torch
        from PIL import Image

        image = Image.fromarray(image_rgb)
        text = ". ".join(list(vocabulary)) + "."
        inputs = self.processor(
            images=image, text=text, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.0,
            text_threshold=0.0,
            target_sizes=[(image.height, image.width)],
        )[0]
        spans = results.get("text_labels") or results.get("labels") or []
        best = {name: 0.0 for name in vocabulary}
        for span, score in zip(spans, results["scores"].tolist()):
            span = str(span).strip().lower()
            if not span:
                continue
            for name in vocabulary:
                # A span counts for a name when one contains the other.
                # Generous on purpose: being stricter would measure the
                # harness rather than the model.
                if span in name or name in span:
                    best[name] = max(best[name], float(score))
        return best


def build(name, repo, device):
    return (OwlV2 if name.startswith("owlv2") else GroundingStyle)(repo, device)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sheets", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=336)
    parser.add_argument("--pad", type=float, default=0.35)
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args(argv)

    import torch

    vocabulary, added_for_bench = bench_vocabulary()
    rows = [json.loads(line) for line in args.detections.open(encoding="utf-8")]
    items = golden_crops(rows, args.captures, args.size, args.pad)
    print(
        f"{len(items)} labelled crops, {len(vocabulary)}-word vocabulary "
        f"({len(added_for_bench)} added for the benchmark: "
        f"{', '.join(added_for_bench)})",
        flush=True,
    )

    report = {
        "device": args.device,
        "crops": len(items),
        "positives": sum(1 for item in items if item["truth"]),
        "negatives": sum(1 for item in items if not item["truth"]),
        "vocabulary": list(vocabulary),
        # The shipped list plus whatever the labelled set needed. Both
        # reported, so a reader can tell an accuracy figure about the
        # shipped configuration from one about this experiment.
        "added_for_bench": list(added_for_bench),
        "crop_size": args.size,
        "crop_padding": args.pad,
        "models": {},
    }

    for name in args.models.split(","):
        repo = MODELS[name]
        print("=" * 20, name, flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        model = build(name, repo, args.device)
        load_seconds = time.perf_counter() - began
        resident = (
            torch.cuda.memory_allocated() / 1e6
            if args.device.startswith("cuda")
            else None
        )

        latencies, results = [], []
        for index, item in enumerate(items):
            rgb = cv2.cvtColor(item["tile"], cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            scores = model.scores(rgb, vocabulary)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            results.append(
                {
                    "proposed": item["proposed"],
                    "prompt": item["prompt"],
                    "truth": item["truth"],
                    "detector_score": item["score"],
                    "area_fraction": item["area_fraction"],
                    "top": ranked[:3],
                    "proposed_score": scores[item["prompt"]],
                    "proposed_rank": [n for n, _ in ranked].index(item["prompt"]),
                }
            )
            if (index + 1) % 20 == 0:
                print(f"  {index + 1}/{len(items)}", flush=True)

        report["models"][name] = {
            "repo": repo,
            "load_seconds": round(load_seconds, 2),
            "resident_vram_mb": None if resident is None else round(resident, 1),
            "peak_vram_mb": (
                round(torch.cuda.max_memory_allocated() / 1e6, 1)
                if args.device.startswith("cuda")
                else None
            ),
            "median_ms": round(float(np.median(latencies)), 1),
            "p95_ms": round(float(np.percentile(latencies, 95)), 1),
            "results": results,
        }
        if args.sheets is not None:
            _write_sheet(args.sheets / f"{name}.png", items, results, args.size)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    _summarise(report)
    return 0


def _write_sheet(path: Path, items, results, size: int) -> None:
    """Every crop with what the model called it, so a person can check.

    The numbers below are only as good as the labels, and the labels came
    from a person reading a sheet. This is how the next person checks
    them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    captions = []
    for item, result in zip(items, results):
        top, score = result["top"][0]
        mark = "OK" if (top == item["prompt"]) == item["truth"] else "XX"
        captions.append(f"{mark} {item['proposed']}->{top} {score:.2f}")
    cv2.imwrite(str(path), sheet([i["tile"] for i in items], captions, 6, size))


def _summarise(report: dict) -> None:
    print()
    print(
        f"{'model':14s} {'accept+':>8s} {'reject-':>8s} {'balanced':>9s} "
        f"{'ms':>6s} {'vramMB':>7s}"
    )
    for name, model in report["models"].items():
        results = model["results"]
        positives = [r for r in results if r["truth"]]
        negatives = [r for r in results if not r["truth"]]
        agreed = lambda r: r["proposed_rank"] == 0  # noqa: E731
        accept = sum(1 for r in positives if agreed(r)) / max(len(positives), 1)
        reject = sum(1 for r in negatives if not agreed(r)) / max(len(negatives), 1)
        print(
            f"{name:14s} {accept:8.3f} {reject:8.3f} {(accept + reject) / 2:9.3f} "
            f"{model['median_ms']:6.0f} {model['peak_vram_mb'] or 0:7.0f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
