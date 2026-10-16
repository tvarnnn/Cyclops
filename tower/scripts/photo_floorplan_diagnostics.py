"""Offline per-walk photo floor plan diagnostics (`tower.world_builder.photo_floorplan_diagnostics`).

Builds the walk summary (`photo_floorplan_diagnostics.json`'s content) from either or both of:

  --log tower.err.log   the live Tower's log. `world photo floorplan diagnostics:` lines (written
                        by every Tower carrying the diagnostics) are read as is; for an older log,
                        the `world photo floorplan:` DISPOSITION lines are converted to partial
                        records (plane source/reason, inliers, opaque, frames; no band, camera or
                        area-in-H measures -- those need the solve).
  --root DATA_ROOT      a world builder data root (the directory holding `worlds/`). The FINAL
                        landing is rebuilt offline from the session's final solve, exactly as the
                        live worker builds a landing (fresh plane, no reuse memory), and its full
                        diagnostics replace that landing's log record (keeping the log's
                        disposition, elapsed time and lag).

Reads only; writes nothing but `--out` (or stdout). CPU only. Example:

  set CUDA_VISIBLE_DEVICES=-1 & set PYTHONPATH=%CD%
  python scripts\\photo_floorplan_diagnostics.py --log ...\\tower.err.log ^
      --root <data-root> --world <id> --session <id> --out summary.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from tower.artifact_paths import artifact_root_arg
from tower.world_builder import guidance_coverage as gc
from tower.world_builder import photo_floorplan_diagnostics as diag
from tower.world_builder import photo_floorplan_worker as pfw

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) ")
_DIAG = re.compile(r"world photo floorplan diagnostics: world=(\S+) session=(\S+) landing=(.*?) "
                   r"record=(\{.*\})\s*$")
_DISP = re.compile(r"world photo floorplan: world=(\S+) session=(\S+) landing=(\(.*?\)) "
                   r"method=(\S+) elapsed_ms=([\d.]+) disposition=(.*?) components=(.*)$")


def _epoch(line):
    m = _TS.match(line)
    if not m:
        return None
    return time.mktime(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timetuple()) + \
        int(m.group(2)) / 1000.0


def _legacy_component(c):
    ok = bool(c.get("plane_ok"))
    gates = c.get("gates") or {}
    rec = {"ref": c.get("ref"), "eligible": c.get("eligible"), "points": c.get("points"),
           "plane": {"ok": ok, "source": (c.get("plane_source") or "estimated") if ok else "rejected",
                     "reason": diag._short(c.get("plane_reason")), "reuse": c.get("plane_reuse"),
                     "reference": c.get("reference"), "inliers": c.get("inliers"),
                     "candidates": c.get("candidates")},
           "cams": {"n": None, "bimodal": None}, "band": {"frames": 0},
           "area": {"opaque": c.get("opaque"), "opaque_area": c.get("opaque_area"),
                    "projected": c.get("projected"), "selected": c.get("selected"),
                    "admissible": gates.get("ok"), "warped": c.get("warped")},
           "legacy": True}
    rec["flags"] = diag.component_flags(rec)
    return rec


def records_from_log(path, world=None, session=None):
    """(diagnostics records, legacy records) from one Tower log."""
    new, legacy = [], []
    escape_pending = False
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if "world photo floorplan" not in line:
                continue
            if "publishing a regressed landing after" in line:
                # The escape valve's WARNING precedes its own disposition line.
                escape_pending = True
                continue
            m = _DIAG.search(line)
            if m:
                if (world and m.group(1) != world) or (session and m.group(2) != session):
                    continue
                try:
                    new.append(json.loads(m.group(4)))
                except ValueError:
                    pass
                continue
            m = _DISP.search(line)
            if not m or (world and m.group(1) != world) or (session and m.group(2) != session):
                continue
            try:
                solved_at, revision, frame_revision = ast.literal_eval(m.group(3))
                comps = json.loads(m.group(7))
            except (ValueError, SyntaxError):
                continue
            logged = _epoch(line)
            disposition = m.group(6)
            escaped, escape_pending = escape_pending, False
            comps = [_legacy_component(c) for c in comps or ()]
            comps.sort(key=lambda r: -(r.get("eligible") or 0))
            total = sum((r.get("eligible") or 0) for r in comps) or 1
            for r in comps:
                r["share"] = diag._num((r.get("eligible") or 0) / total)
            legacy.append({
                "v": diag.VERSION, "source": "legacy_log", "world": m.group(1),
                "session": m.group(2), "solved_at": float(solved_at),
                "geometry_revision": revision, "frame_revision": frame_revision,
                "disposition": disposition, "class": diag._disposition_class(disposition),
                "published": disposition == "ok",
                "escaped": disposition == "ok" and escaped,
                "elapsed_ms": float(m.group(5)),
                "landing_lag_s": (diag._num(logged - float(solved_at), 2)
                                  if logged is not None else None),
                "components": comps})
    return new, legacy


def offline_final(root, world, session):
    """The final landing's full diagnostics record, rebuilt from the data root."""
    base = Path(root) / "worlds" / world
    manifest = json.loads((base / "derived" / session / "manifest.json").read_text("utf-8"))
    summary = manifest["global_solve"]
    st = (base / "solve" / session / "solution.json").stat()
    stat = (st.st_size, st.st_mtime_ns)
    from tower.results.world_builder import geometry_revision_from_manifest
    budget = gc._Budget()
    budget.deadline = math.inf   # offline: no 250 ms coverage budget
    eligible, tree = [], {}
    receipt = gc.compute_from_tree(root, world, session,
                                   (summary["solved_at"], summary["horizon_keyframes"], stat),
                                   geometry_revision_from_manifest(manifest),
                                   tree_tag=(manifest.get("built_at"), manifest.get("input_digest")),
                                   budget=budget, eligible_out=eligible, tree_out=tree)
    t0 = time.monotonic()
    _, _, stats = pfw.build_landing(root, world, session, receipt, eligible, tree, stat,
                                    plane_memory={})
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    rec = diag.landing_record(world_id=world, session_id=session, receipt=receipt,
                              disposition="ok", published=True, escaped=False,
                              elapsed_ms=elapsed_ms, stats=stats, now=float("nan"))
    rec.update(source="offline_rebuild", landing_lag_s=None, coverage_lag_s=None)
    return rec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default=None, help="a Tower log (tower.err.log)")
    parser.add_argument("--root", type=artifact_root_arg, default=None,
                        help="a world builder data root (holds worlds/); read only")
    parser.add_argument("--world", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--out", default=None, help="write the summary here (default: stdout)")
    args = parser.parse_args(argv)
    if args.log is None and args.root is None:
        parser.error("give --log, --root, or both")
    if args.root is not None and not (args.world and args.session):
        parser.error("--root needs --world and --session")

    records, notes = [], {}
    if args.log:
        new, legacy = records_from_log(args.log, args.world, args.session)
        records = new or legacy
        notes["log_records"] = {"diagnostics": len(new), "legacy": len(legacy)}
    if args.root is not None:
        final = offline_final(args.root, args.world, args.session)
        same = [r for r in records if r.get("solved_at") is not None
                and abs(float(r["solved_at"]) - float(final["solved_at"])) < 1e-3]
        if same:
            live = same[-1]
            final.update({k: live.get(k) for k in ("disposition", "class", "published", "escaped",
                                                   "elapsed_ms", "landing_lag_s", "coverage_lag_s")})
            final["live_components"] = [{"ref": c.get("ref"),
                                         "plane": (c.get("plane") or {}).get("source"),
                                         "reuse": (c.get("plane") or {}).get("reuse"),
                                         "opaque": (c.get("area") or {}).get("opaque")}
                                        for c in live.get("components", [])]
            records = [r for r in records if r is not live]
        records.append(final)
        notes["final_rebuilt_offline"] = True
    summary = diag.walk_summary(records, world_id=args.world, session_id=args.session,
                                generated_at=time.time())
    summary["offline"] = notes
    text = json.dumps(summary, indent=1, allow_nan=False, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
