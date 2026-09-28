"""Write the two frozen offline blur summaries for one frame-quality journal."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

from tower.artifact_paths import artifact_root_arg
from tower.world_builder.pacing_summary import (
    load_frames_observed,
    load_journal,
    load_regions,
    render_text,
    summarize,
)


def _inside_tower_data(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    return any(a == "tower" and b == "data" for a, b in zip(parts, parts[1:]))


def _atomic_text(path: Path, content: str):
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            name = Path(handle.name)
            handle.write(content)
        os.replace(name, path)
    finally:
        if name is not None and name.exists():
            name.unlink()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--regions", type=Path)
    parser.add_argument("--room-map", type=Path)
    parser.add_argument("--session-json", type=Path)
    parser.add_argument("--out-dir", required=True, type=artifact_root_arg)
    parser.add_argument("--text-runs", action="store_true")
    args = parser.parse_args(argv)
    try:
        journal = args.journal.resolve()
        output = args.out_dir.resolve()
        if output == journal.parent:
            raise ValueError("output directory must differ from the journal directory")
        if _inside_tower_data(output):
            raise ValueError("output directory cannot be under tower/data")
        if args.regions is None and args.room_map is not None or args.regions is not None and args.room_map is None:
            raise ValueError("--regions and --room-map must be supplied together")
        rows, skipped = load_journal(journal)
        labels = load_regions(args.regions) if args.regions else None
        if args.room_map:
            with args.room_map.open("r", encoding="utf-8") as handle:
                room_map = json.load(handle)
            if not isinstance(room_map, dict):
                raise ValueError("room map JSON must be an object")
        else:
            room_map = None
        session_path = args.session_json or journal.with_name("session.json")
        observed = load_frames_observed(session_path)
        result = summarize(rows, skipped=skipped, frames_observed=observed,
                           regions=labels, room_map=room_map)
        output.mkdir(parents=True, exist_ok=True)
        _atomic_text(output / "pacing_summary.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
        _atomic_text(output / "pacing_summary.txt", render_text(result, text_runs=args.text_runs))
    except (OSError, UnicodeError, ValueError, AssertionError, csv.Error) as exc:  # csv.Error: review LOW-6
        print(f"pacing summary: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
