#!/usr/bin/env python
r"""What Document Memory costs, and how well it reads.

SYNTHETIC, NOT PHYSICAL. Timings are real measurements of this code on
this machine; the pages are rendered, so a COST here is guidance and a
RECALL number is a statement about clean printed text under ideal
lighting -- an upper bound on what real footage will do, never a
prediction of it.

Two things are measured, and the second is the one that matters:

  1. The cheap path -- detection per frame, which runs on every frame.
  2. Read quality against KNOWN text, swept over frame size and tilt.

The sweep shows that tilt barely matters once the page is warped, and
RESOLUTION dominates. The glasses deliver 640x360, and that is not enough
to read a page.

    .venv\Scripts\python.exe scripts/document_memory_benchmark.py
    .venv\Scripts\python.exe scripts/document_memory_benchmark.py --format json
    .venv\Scripts\python.exe scripts/document_memory_benchmark.py --no-ocr
"""

import argparse
import contextlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import document_fixtures as fx  # noqa: E402
from tower.document_memory.detect import detect_page, warp_page  # noqa: E402

# The delivered resolution first.
FRAME_SIZES = [(640, 360), (640, 480), (896, 504), (1280, 720)]
TILTS = [0.0, 0.5, 1.0]
DOCUMENTS = {
    "paper": fx.TRANSFORMER_PAPER,
    "notes": fx.DEPTH_NOTES,
    "receipt": fx.RECEIPT,
}

_TOKEN = re.compile(r"[a-z0-9]+")


def word_recall(expected: str, actual: str) -> float:
    """Fraction of the rendered words OCR captured.

    The metric retrieval depends on. Sequence similarity would score a
    receipt's collapsed column whitespace as a failure; a lexical search
    would not notice it at all.
    """
    expected_words = _TOKEN.findall(expected.lower())
    actual_words = set(_TOKEN.findall(actual.lower()))
    if not expected_words:
        return 1.0
    return sum(1 for word in expected_words if word in actual_words) / len(
        expected_words
    )


def _timed(fn, repeat: int) -> dict:
    fn()  # untimed warm-up
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": round(statistics.mean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "max_ms": round(max(samples), 3),
    }


def bench_detection(repeat: int) -> list[dict]:
    """The cheap gate, which runs on EVERY frame."""
    rows = []
    for width, height in FRAME_SIZES:
        page_frame, _ = fx.place_page(
            fx.render_page(fx.TRANSFORMER_PAPER), frame_size=(width, height)
        )
        page_gray = cv2.cvtColor(page_frame, cv2.COLOR_BGR2GRAY)
        empty_gray = cv2.cvtColor(
            fx.no_page_frame(frame_size=(width, height)), cv2.COLOR_BGR2GRAY
        )
        rows.append(
            {
                "resolution": f"{width}x{height}",
                "with_page": _timed(lambda: detect_page(page_gray), repeat),
                # The common case in a real session is NO page in view, so
                # its cost is the one that dominates a day of wearing.
                "without_page": _timed(lambda: detect_page(empty_gray), repeat),
                "detected": detect_page(page_gray) is not None,
            }
        )
    return rows


def bench_read_quality(recogniser) -> list[dict]:
    rows = []
    for name, lines in DOCUMENTS.items():
        truth = fx.page_text(lines)
        for width, height in FRAME_SIZES:
            for tilt in TILTS:
                frame, _ = fx.place_page(
                    fx.render_page(lines), frame_size=(width, height), tilt=tilt
                )
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                candidate = detect_page(gray)
                if candidate is None:
                    rows.append(
                        {
                            "document": name,
                            "resolution": f"{width}x{height}",
                            "tilt": tilt,
                            "detected": False,
                        }
                    )
                    continue
                warped = warp_page(gray, candidate.corners)
                start = time.perf_counter()
                result = recogniser.read(warped)
                ocr_ms = (time.perf_counter() - start) * 1000
                rows.append(
                    {
                        "document": name,
                        "resolution": f"{width}x{height}",
                        "tilt": tilt,
                        "detected": True,
                        "warped": f"{warped.shape[1]}x{warped.shape[0]}",
                        "word_recall": round(word_recall(truth, result.text), 4),
                        "regions": result.region_count,
                        "mean_confidence": (
                            round(result.mean_confidence, 4)
                            if result.mean_confidence
                            else None
                        ),
                        "ocr_ms": round(ocr_ms, 1),
                    }
                )
    return rows


def bench_retrieval(repeat: int) -> dict:
    """Query latency over a corpus, so growth is visible rather than assumed."""
    import tempfile

    from tower.confidence import Confidence
    from tower.document_memory.records import DocumentObservation, PageObservation
    from tower.document_memory.retrieval import DocumentMemory
    from tower.document_memory.store import DocumentStore

    results = {}
    for count in (10, 100, 1000):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(directory)
            for index in range(count):
                lines = list(DOCUMENTS.values())[index % len(DOCUMENTS)]
                store.append(
                    DocumentObservation(
                        document_id=f"d{index}",
                        observed_at=1000.0 + index,
                        recorded_at=1000.0 + index,
                        observed_seconds=10.0,
                        pages=(
                            PageObservation(
                                page_index=0,
                                text=fx.page_text(lines),
                                region_count=8,
                                confidence=Confidence.HIGH,
                            ),
                        ),
                    )
                )
            memory = DocumentMemory(store)
            results[str(count)] = {
                "search": _timed(
                    lambda: memory.search_text("transformer attention"), repeat
                ),
                "recent": _timed(lambda: memory.recent(10), repeat),
                "bytes": store.bytes_used(),
                "bytes_per_document": round(
                    store.bytes_used()["total"] / max(count, 1), 1
                ),
            }
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Document Memory's cheap path and its read quality."
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip the read-quality sweep, which needs the [ocr] extra.",
    )
    args = parser.parse_args(argv)

    report = {
        "note": (
            "SYNTHETIC, NOT PHYSICAL. Recall numbers are clean printed text "
            "under ideal lighting -- an upper bound on real footage."
        ),
        "repeat": args.repeat,
        "detection": bench_detection(args.repeat),
        "retrieval": bench_retrieval(args.repeat),
    }

    if not args.no_ocr:
        from tower.document_memory.ocr import EasyOcrRecogniser

        recogniser = EasyOcrRecogniser()
        # The reader's loader prints to stdout, which would corrupt
        # --format json. Anything it says goes to stderr instead.
        start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            recogniser.load()
        report["ocr_load_seconds"] = round(time.perf_counter() - start, 2)
        try:
            report["read_quality"] = bench_read_quality(recogniser)
        finally:
            recogniser.release()

    if args.format == "json":
        print(json.dumps(report, indent=2))
        return 0

    print(report["note"])
    print("\n=== Detection (runs on EVERY frame) ===")
    print(f"{'resolution':14s}{'with page':>12s}{'no page':>12s}  detected")
    for row in report["detection"]:
        print(
            f"{row['resolution']:14s}{row['with_page']['mean_ms']:>10.2f}ms"
            f"{row['without_page']['mean_ms']:>10.2f}ms  {row['detected']}"
        )

    print("\n=== Retrieval ===")
    print(f"{'documents':12s}{'search':>12s}{'recent':>12s}{'bytes/doc':>12s}")
    for count, row in report["retrieval"].items():
        print(
            f"{count:12s}{row['search']['mean_ms']:>10.2f}ms"
            f"{row['recent']['mean_ms']:>10.2f}ms{row['bytes_per_document']:>12.0f}"
        )

    if "read_quality" in report:
        print(f"\n=== Read quality (OCR reader load {report['ocr_load_seconds']}s) ===")
        print(
            f"{'document':10s}{'resolution':12s}{'tilt':6s}{'warped':12s}"
            f"{'recall':>8s}{'ocr ms':>9s}"
        )
        for row in report["read_quality"]:
            if not row["detected"]:
                print(
                    f"{row['document']:10s}{row['resolution']:12s}"
                    f"{row['tilt']:<6.1f}NOT DETECTED"
                )
                continue
            print(
                f"{row['document']:10s}{row['resolution']:12s}{row['tilt']:<6.1f}"
                f"{row['warped']:12s}{row['word_recall']:>8.3f}{row['ocr_ms']:>9.0f}"
            )
        print(
            "\nRESOLUTION, NOT TILT, IS THE CONSTRAINT. The glasses deliver "
            "640x360 today and it is not enough to read a page."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
