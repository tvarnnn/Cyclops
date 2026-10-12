"""Exp1 slice 3: what the live depth cache saved on one finished C22 replay.

    python scripts/exp1_live_depth_replay_check.py <C22 run root>

Reads, never writes, and loads no model: CPU-only arithmetic over the run's
existing artifacts. It reports

  coverage   c = final-depth keyframes whose (kid, image_sha1) the live cache
             (`dense/<session>/live_depth/align.json`) predicted, over the
             Stop-time depth stage's targets (its `align.json` `targets`);
  depth      the Stop-time depth stage's network calls made vs skipped
             (its `image_origins` ledger: `reused-prediction` is a skip), and
             how many of the skipped were live-covered -- a skip can also come
             from an earlier live surface's prediction cache;
  surface    the surface stage's per-step `seconds` and their sum, from the
             session summary the build child prints to the Tower's out log.

The world and session come from `report.json` (`store.world_dir`,
`store.session_id`), else the single world under `run.json`'s `data_root`.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

_SURFACE_LINE = re.compile(r"^surface\s+(\{.*\})\s*$")
_REUSED_LINE = re.compile(r"\[dense\] depth: (\d+) of (\d+) predictions reused")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _session_dir(run: Path) -> tuple[Path, str]:
    report = _read_json(run / "report.json") or {}
    store = report.get("store") if isinstance(report.get("store"), dict) else {}
    if store.get("world_dir") and store.get("session_id"):
        return Path(store["world_dir"]), str(store["session_id"])
    meta = _read_json(run / "run.json") or {}
    worlds = sorted((Path(meta.get("data_root", "")) / "world_builder" / "worlds").glob("*/dense/*"))
    if len(worlds) != 1:
        raise SystemExit(f"cannot find exactly one world session under {run} ({len(worlds)} found)")
    return worlds[0].parent.parent, worlds[0].name


def _surface_seconds(run: Path):
    meta = _read_json(run / "run.json") or {}
    logs = [Path(meta["out_log"])] if meta.get("out_log") else sorted(run.glob("tower-*.out.log"))
    found = None
    for log in logs:
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            match = _SURFACE_LINE.match(line)
            if match:
                try:
                    summary = ast.literal_eval(match.group(1))
                except (ValueError, SyntaxError):
                    continue
                if isinstance(summary, dict) and isinstance(summary.get("seconds"), dict):
                    found = {"log": str(log), "state": summary.get("state"),
                             "steps": summary["seconds"],
                             "total": round(sum(float(v) for v in summary["seconds"].values()), 3)}
    return found


def _reused_log_line(run: Path):
    meta = _read_json(run / "run.json") or {}
    logs = [Path(meta["err_log"])] if meta.get("err_log") else sorted(run.glob("tower-*.err.log"))
    last = None
    for log in logs:
        try:
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                match = _REUSED_LINE.search(line)
                if match:
                    last = {"reused": int(match.group(1)), "of": int(match.group(2))}
        except OSError:
            continue
    return last


def check(run: Path) -> dict:
    world_dir, session_id = _session_dir(run)
    dense = world_dir / "dense" / session_id
    final = _read_json(dense / "align.json")
    if not isinstance(final, dict):
        raise SystemExit(f"no Stop-time depth stage at {dense / 'align.json'}")
    live = _read_json(dense / "live_depth" / "align.json")
    live_pairs = set()
    if isinstance(live, dict):
        live_pairs = {(r.get("kid"), r.get("image_sha1")) for r in live.get("records", [])
                      if isinstance(r, dict)}
    records = [r for r in final.get("records", []) if isinstance(r, dict)]
    targets = int(final.get("targets") or len(records))
    covered = [r for r in records if (r.get("kid"), r.get("image_sha1")) in live_pairs]
    # The stage's own `image_origins` counter is the call ledger: a failed fit
    # record drops `image_origin`, so per-record origins undercount. A frame
    # counted under its origin but refused before the network ran ("image
    # <origin>", "image undecodable", "frame is WxH") made no call.
    origins = final.get("image_origins") or {}
    skipped = int(origins.get("reused-prediction", 0))
    resumed = int(origins.get("resumed", 0))
    refused = sum(str(r.get("why", "")).startswith(("image ", "frame is "))
                  for r in records)
    made = sum(int(v) for k, v in origins.items()
               if k not in ("reused-prediction", "resumed")) - refused
    return {
        "run_root": str(run), "world_dir": str(world_dir), "session_id": session_id,
        "coverage": {"live_cached": len(covered), "final_keyframes": targets,
                     "c": round(len(covered) / targets, 4) if targets else None,
                     "live_records": len(live_pairs), "live_sidecar": live is not None,
                     "live_counters": (live or {}).get("live")},
        "depth": {"made": made, "skipped": skipped, "resumed": resumed,
                  # A lower bound: a reused frame whose fit failed has no origin.
                  "skipped_live_covered_at_least": sum(
                      r.get("image_origin") == "reused-prediction" for r in covered),
                  "refused_before_predict": refused,
                  "image_origins": origins,
                  "stage_seconds": final.get("seconds"),
                  "tower_log_reused": _reused_log_line(run)},
        "surface_seconds": _surface_seconds(run),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_root", type=Path, help="a finished C22 replay run root (read only)")
    args = parser.parse_args(argv)
    print(json.dumps(check(args.run_root), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
