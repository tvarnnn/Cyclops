"""SEL: offline best-of-N final solve, selected by the final scale guard (walk 8, research only).

Maps the FINAL global solve of a frozen world copy N times as independent mapper draws, each on its own
COPY of the matched database (pycolmap opening a database mutates it), applies final-scale-guard v2 with
the world's live depth cache to every draw, and evaluates a guard-only selection score against a
ground-truth label (camera-centre hull test + closet scale) that production could not compute.

Subcommands
  draw     one draw: `_map_candidate` exactly as the final solve calls it, then `guard_publication`.
  replay   the published G2 candidate rebuilt from its own `sparse/`, guarded: checks the harness.
  analyze  per-draw table, labels, scores, best-of-k selection, PNGs, `sel.json`.

Production code is imported, never modified. Nothing is written outside `--out`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402

FPS = 24.0                     # walk second = source sequence / 24 (FINDINGS.md)
STRETCHES = ((73, 91), (104, 135), (141, 148), (160, 167), (174, 188), (201, 209))
CENTRAL = STRETCHES[1:5]       # the closet proper, 104-188 s
BEDROOM = ((0, 70), (92, 103))  # the early-bedroom hull of geometry_analysis.py
MIN_OBS = 30
SCALE_BAND = (0.8, 1.25)       # the guard's screen band: "scale ~ 1"
HULL_MOSTLY_OUTSIDE = 0.5      # GOOD needs < 50 % of closet centres inside the bedroom hull

# ---------------------------------------------------------------------------------------------------
# PRE-REGISTERED SCORES (written before any draw was labelled; see SEL-REPORT.md "Pre-registration").
#   S1, the lead's: fewest isolated keyframes, then the guard-visible scale closest to 1. Guard-only, the
#       scale is the worst |log ratio| over the ATTACHED measured plateaus left in the room (the guard does
#       not know where the closet is). S1c is the literal variant with the closet's own median ratio
#       (needs walk-time knowledge, evaluation only).
#   S2, proposed: most CERTIFIED room keyframes (placed room minus uncertified), then fewest isolated.
#       Rationale: a welded draw that the guard catches loses keyframes to isolation; a welded draw the
#       guard misses keeps them only as uncertified spans; a split draw loses them to another component.
# Lower key = better.
# ---------------------------------------------------------------------------------------------------


def score_s1(row: dict) -> tuple:
    return (row["isolated_kf"], row["worst_attached_abslog"])


def score_s1c(row: dict) -> tuple:
    r = row["closet_ratio"]
    return (row["isolated_kf"], abs(math.log(r)) if r else math.inf)


def score_s2(row: dict) -> tuple:
    return (-(row["room_kf"] - row["uncertified_kf"]), row["isolated_kf"])


SCORES = {"S1": score_s1, "S1c": score_s1c, "S2": score_s2}


# --------------------------------------------------------------------------- pure helpers


def walk_second(kid: str) -> float:
    return int(kid.rsplit(":", 1)[-1]) / FPS


def in_spans(s: float, spans) -> bool:
    return any(a <= s <= b for a, b in spans)


def centre(pose: dict) -> np.ndarray:
    R = np.asarray(pose["rotation"], np.float64).reshape(3, 3)
    return -R.T @ np.asarray(pose["translation"], np.float64)


def bedroom_basis(points: np.ndarray) -> np.ndarray:
    """(3, 2) horizontal basis: the two leading PCA directions of the bedroom centres."""
    _, _, vt = np.linalg.svd(points - np.median(points, axis=0), full_matrices=False)
    return vt[:2].T


def hull_inside_fraction(bedroom: np.ndarray, query: np.ndarray) -> float | None:
    """Fraction of `query` centres inside the 2-D convex hull of `bedroom` centres (top view)."""
    from scipy.spatial import ConvexHull, Delaunay

    if len(query) == 0 or len(bedroom) < 4:
        return None
    basis = bedroom_basis(bedroom)
    b2 = bedroom @ basis
    hull = Delaunay(b2[ConvexHull(b2).vertices])
    return float(np.mean(hull.find_simplex(query @ basis) >= 0))


def label(closet_in_room_frac: float, inside: float | None, ratio: float | None) -> str:
    """GOOD: the closet is in the main component, mostly outside the bedroom hull, at scale ~1.
    SPLIT: the closet is mostly not in the main component (not placed; counted not-GOOD).
    WELDED: everything else."""
    if closet_in_room_frac < 0.5:
        return "SPLIT"
    if inside is not None and inside < HULL_MOSTLY_OUTSIDE and ratio is not None \
            and SCALE_BAND[0] <= ratio <= SCALE_BAND[1]:
        return "GOOD"
    return "WELDED"


def similarity(src: np.ndarray, dst: np.ndarray):
    """Umeyama: (scale, R, t) with dst ~ scale * R @ src + t."""
    s0, d0 = src.mean(0), dst.mean(0)
    u, _, vh = np.linalg.svd((dst - d0).T @ (src - s0))
    R = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vh)]) @ vh
    scale = np.sum((dst - d0) * ((src - s0) @ R.T)) / np.sum((src - s0) ** 2)
    return float(scale), R, d0 - scale * R @ s0


def reference_deviation(sol: dict, ref: dict) -> dict:
    """Corroboration only (FINDINGS.md): fit a similarity on the early-bedroom cameras (seq < 1700) the
    draw's and the reference solve's (R0q, a GOOD no-guard draw) component 0 share, then the median
    distance of the draw's closet centres from R0q's, in R0q SfM units. G2b ~1.2, G2 2.7-4.5."""
    def by_seq(s):
        return {int(k.rsplit(":", 1)[-1]): p for k, p in s["poses"].items() if int(p["observations"]) >= MIN_OBS}
    a, b = by_seq(sol), by_seq(ref)
    common = sorted(q for q in a.keys() & b.keys() if q < 1700
                    and int(a[q]["component"]) == 0 and int(b[q]["component"]) == 0)
    if len(common) < 20:
        return {"anchors": len(common), "closet_median": None}
    scale, R, t = similarity(np.array([centre(a[q]) for q in common]), np.array([centre(b[q]) for q in common]))
    closet = [q for q in a.keys() & b.keys() if in_spans(q / FPS, CENTRAL)
              and int(a[q]["component"]) == 0 and int(b[q]["component"]) == 0]
    if not closet:
        return {"anchors": len(common), "closet_median": None}
    err = np.linalg.norm(scale * np.array([centre(a[q]) for q in closet]) @ R.T + t
                         - np.array([centre(b[q]) for q in closet]), axis=1)
    return {"anchors": len(common), "closet_n": len(closet), "closet_median": float(np.median(err))}


def selection_stats(rows: list[dict], key, k: int) -> dict:
    """Over every k-subset of draws: P(the key's pick is GOOD), P(some draw is GOOD), P(a random one is)."""
    subsets = list(itertools.combinations(range(len(rows)), k))
    if not subsets:
        return {"k": k, "subsets": 0}
    picked = any_good = 0
    for sub in subsets:
        best = min(sub, key=lambda i: (key(rows[i]), i))
        picked += rows[best]["label"] == "GOOD"
        any_good += any(rows[i]["label"] == "GOOD" for i in sub)
    p = sum(r["label"] == "GOOD" for r in rows) / len(rows)
    return {"k": k, "subsets": len(subsets), "p_pick_good": picked / len(subsets),
            "p_any_good": any_good / len(subsets), "p_single_good": p}


def separation(rows: list[dict], key) -> dict:
    """Does `key` rank every GOOD draw above every non-GOOD one? AUC over (GOOD, other) pairs; ties 0.5."""
    good = [r for r in rows if r["label"] == "GOOD"]
    bad = [r for r in rows if r["label"] != "GOOD"]
    if not good or not bad:
        return {"auc": None, "pairs": 0}
    wins = sum(1.0 if key(g) < key(b) else 0.5 if key(g) == key(b) else 0.0 for g in good for b in bad)
    return {"auc": wins / (len(good) * len(bad)), "pairs": len(good) * len(bad)}


# --------------------------------------------------------------------------- world access


def locate(world: Path):
    """(store, world_id, session_id, solve dir) of a single-world copy."""
    from tower.world_builder.store import WorldStore

    store = WorldStore(world / "world_builder")
    (world_dir,) = [p for p in (world / "world_builder" / "worlds").iterdir() if p.is_dir()]
    (solve_dir,) = [p for p in (world_dir / "solve").iterdir() if p.is_dir()]
    return store, world_dir.name, solve_dir.name, solve_dir


def guard(store, world_id, session_id, solution, database_path, keyframes):
    """final-scale-guard v2 exactly as `guard_publication` runs it at the publish seam (mode on,
    ungated: `result=None`), on the live depth cache. Writes nothing."""
    from tower.world_builder import final_scale_guard as FSG

    os.environ["TOWER_WORLD_LIVE_DEPTH"] = "on"     # the live cache is evidence only when on
    return FSG.guard_publication(store, world_id, session_id, solution, None, mode_=FSG.MODE_ON,
                                 database_path=database_path, keyframes=keyframes)


def _copy_db(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "database.db"
    shutil.copyfile(src, dst)
    return dst


def _write_draw(draw_dir: Path, candidate, outcome, meta: dict) -> None:
    from tower.world_builder import final_scale_guard as FSG
    from tower.world_builder.global_solve import SolveWorkspace, write_solution

    write_solution(SolveWorkspace(draw_dir), candidate)         # the PRE-guard candidate
    FSG.write_audit(draw_dir, outcome.audit)
    meta = dict(meta, guard_decision=outcome.decision, guard_summary=outcome.summary)
    (draw_dir / "draw.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_draw(world: Path, out: Path, *, tag: str, seed: int | None, threads: int) -> dict:
    """One final-solve draw, as `global_solve.solve` maps it (`_map_candidate`, GLOMAP then the
    incremental fallback) on a copy of the frozen matched database, then guarded."""
    import pycolmap

    from tower.world_builder import global_solve as GS

    GS._quiet_pycolmap()
    store, wid, sid, solve_dir = locate(world)
    keyframes = store.read_keyframes(wid, sid)
    published = GS.load_solution(store, wid, sid)
    camera = GS.PinholeCamera.from_json_dict(published.camera)
    draw_dir = out / "draws" / tag
    if draw_dir.exists():
        raise SystemExit(f"{draw_dir} exists; pick another tag")
    started_wall = time.time()
    c0 = time.perf_counter()
    db = _copy_db(solve_dir / "database.db", draw_dir)
    copy_s = time.perf_counter() - c0
    t0 = time.perf_counter()
    candidate = GS._map_candidate(
        pycolmap, db, GS.SolveWorkspace(solve_dir), draw_dir / "sparse", keyframes,
        seed=seed, threads=(1 if seed is not None else threads), input_digest=published.input_digest,
        min_image_observations=GS.MIN_IMAGE_OBSERVATIONS, camera=camera)
    map_s = time.perf_counter() - t0
    candidate.timing = {"map_s": round(map_s, 3), "final": True}
    candidate.transients = published.transients
    candidate.solve = dict(published.solve or {}, seed=seed, seeded=seed is not None,
                           threads=1 if seed is not None else threads, final_scale_guard=None)
    candidate.solve.pop("final_scale_guard")
    t1 = time.perf_counter()
    outcome = guard(store, wid, sid, candidate, db, keyframes)
    guard_s = time.perf_counter() - t1
    meta = {"tag": tag, "seed": seed, "threads": 1 if seed is not None else threads, "pid": os.getpid(),
            "solver": candidate.solver, "started_wall": started_wall, "ended_wall": time.time(),
            "copy_s": round(copy_s, 3), "map_s": round(map_s, 3), "guard_s": round(guard_s, 3),
            "cpu_count": os.cpu_count(), "pycolmap": getattr(pycolmap, "__version__", None),
            "database_source": str(solve_dir / "database.db")}
    _write_draw(draw_dir, candidate, outcome, meta)
    return meta


def run_replay(world: Path, out: Path) -> dict:
    """The published candidate rebuilt from the world's own `sparse/` and guarded on a DB copy: the
    harness must reproduce the published audit's pieces (G2: 6 pieces, 303 keyframes)."""
    import pycolmap

    from tower.world_builder import global_solve as GS

    store, wid, sid, solve_dir = locate(world)
    keyframes = store.read_keyframes(wid, sid)
    published = GS.load_solution(store, wid, sid)
    recs = {int(p.name): pycolmap.Reconstruction(str(p))
            for p in (solve_dir / "sparse").iterdir() if p.is_dir() and p.name.isdigit()}
    candidate = GS._solution_from_reconstructions(
        recs, keyframes, solver=published.solver, input_digest=published.input_digest,
        min_image_observations=GS.MIN_IMAGE_OBSERVATIONS,
        camera=GS.PinholeCamera.from_json_dict(published.camera))
    candidate.solve = {k: v for k, v in (published.solve or {}).items() if k != "final_scale_guard"}
    candidate.transients = published.transients
    draw_dir = out / "draws" / ("replay-" + world.name.lstrip("_"))
    db = _copy_db(solve_dir / "database.db", draw_dir)
    t1 = time.perf_counter()
    outcome = guard(store, wid, sid, candidate, db, keyframes)
    old = json.loads((solve_dir / "final_scale_guard.json").read_text())["assessment"]
    new = outcome.audit["assessment"]
    same_pose = all(np.allclose(candidate.poses[k]["translation"], p["translation"], atol=1e-6)
                    for k, p in published.poses.items() if k in candidate.poses)
    meta = {"tag": draw_dir.name, "seed": None, "threads": None, "solver": candidate.solver,
            "map_s": None, "guard_s": round(time.perf_counter() - t1, 3),
            "poses_equal_published": bool(same_pose),
            "published_pieces": [len(p["keyframe_ids"]) for p in old["pieces"]],
            "replayed_pieces": [len(p["keyframe_ids"]) for p in new["pieces"]],
            "published_reference_log": old["reference"]["log"], "replayed_reference_log": new["reference"]["log"],
            "published_uncertified": old["uncertified_keyframes"],
            "replayed_uncertified": new["uncertified_keyframes"]}
    _write_draw(draw_dir, candidate, outcome, meta)
    return meta


# --------------------------------------------------------------------------- analysis


def _load_draw(draw_dir: Path):
    sol = json.loads((draw_dir / "solution.json").read_text())
    audit = json.loads((draw_dir / "final_scale_guard.json").read_text())
    meta = json.loads((draw_dir / "draw.json").read_text())
    return sol, audit, meta


def metric_log_of(store, wid, sid, draw_dir: Path, keyframes):
    """The guard's per-camera metric log for this draw (recomputed: the audit stores only plateaus)."""
    from tower.world_builder import coherence_publish as CP
    from tower.world_builder import final_scale_guard as FSG

    sol = _load_solution_dir(draw_dir)
    os.environ["TOWER_WORLD_LIVE_DEPTH"] = "on"
    log, _inputs = FSG.read_evidence(store, wid, sid, sol, None, database_path=draw_dir / "database.db",
                                     name_of=CP._image_names(keyframes))
    return log


def _load_solution_dir(draw_dir: Path):
    import io

    from tower.world_builder.global_solve import Solution

    meta = json.loads((draw_dir / "solution.json").read_text())
    with np.load(io.BytesIO((draw_dir / "solution.npz").read_bytes())) as a:
        return Solution(solver=meta["solver"], solved_at=float(meta["solved_at"]),
                        input_digest=meta.get("input_digest"), keyframe_ids=list(meta["keyframe_ids"]),
                        poses=dict(meta["poses"]), components=list(meta["components"]), xyz=a["xyz"],
                        rgb=a["rgb"], component=a["component"], first_keyframe=a["first_keyframe"],
                        track_length=a["track_length"], error=a["error"],
                        observations=a["observations"].reshape(-1, 3),
                        observation_xy=a["observation_xy"].reshape(-1, 2), camera=meta.get("camera"),
                        timing=dict(meta.get("timing") or {}), transients=meta.get("transients"),
                        solve=meta.get("solve"))


def measure(sol: dict, assessment: dict, metric_log: dict, name_of: dict) -> dict:
    """Per-draw guard numbers and the evaluation-only ground truth."""
    poses = sol["poses"]
    sup = {k: p for k, p in poses.items() if int(p["observations"]) >= MIN_OBS}
    comp0 = {k: p for k, p in sup.items() if int(p["component"]) == 0}
    isolated = {k for p in assessment.get("pieces") or [] for k in p["keyframe_ids"]}
    ref = (assessment.get("reference") or {}).get("log")
    closet_all = [k for k in sol["keyframe_ids"] if in_spans(walk_second(k), CENTRAL)]
    closet_sup = [k for k in closet_all if k in sup]
    closet_room = [k for k in closet_sup if k in comp0]
    bedroom = np.array([centre(p) for k, p in comp0.items() if in_spans(walk_second(k), BEDROOM)])
    inside = hull_inside_fraction(bedroom, np.array([centre(comp0[k]) for k in closet_room]).reshape(-1, 3))
    per_stretch = []
    for a, b in CENTRAL:
        ks = [k for k in closet_room if a <= walk_second(k) <= b]
        f = hull_inside_fraction(bedroom, np.array([centre(comp0[k]) for k in ks]).reshape(-1, 3))
        vals = [metric_log[name_of[k]] for k in ks if name_of[k] in metric_log]
        per_stretch.append({"span": [a, b], "n": len(ks), "inside": f,
                            "ratio": math.exp(float(np.median(vals)) - ref) if vals and ref is not None
                            else None})
    vals = [metric_log[name_of[k]] for k in closet_room if name_of[k] in metric_log]
    closet_ratio = math.exp(float(np.median(vals)) - ref) if vals and ref is not None else None
    attached = [p for p in assessment.get("plateaus") or []
                if p.get("decision") in ("kept", "attached") and p.get("ratio") and p.get("measured", 0) >= 10]
    worst = max((abs(math.log(p["ratio"])) for p in attached), default=0.0)
    room_kf = len(assessment.get("room_keyframes") or [])
    frac_room = len(closet_room) / max(1, len(closet_sup))
    return {
        "solver_comp0_kf": len(comp0), "room_kf": room_kf, "isolated_kf": len(isolated),
        "pieces": [len(p["keyframe_ids"]) for p in assessment.get("pieces") or []],
        "uncertified_kf": int(assessment.get("uncertified_keyframes") or 0),
        "scale_status": assessment.get("scale_status"), "reference_log": ref,
        "worst_attached_abslog": worst, "closet_kf": len(closet_all), "closet_supported": len(closet_sup),
        "closet_in_comp0": len(closet_room), "closet_isolated": sum(k in isolated for k in closet_room),
        "closet_ratio": closet_ratio, "closet_inside": inside, "per_stretch": per_stretch,
        "components": [c["images_supported"] for c in sol["components"]],
        "posed": len(poses), "supported": len(sup),
        "label": label(frac_room, inside, closet_ratio),
    }


def plot(sol: dict, assessment: dict, row: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comp0 = {k: p for k, p in sol["poses"].items()
             if int(p["observations"]) >= MIN_OBS and int(p["component"]) == 0}
    isolated = {k for p in assessment.get("pieces") or [] for k in p["keyframe_ids"]}
    bedroom = np.array([centre(p) for k, p in comp0.items() if in_spans(walk_second(k), BEDROOM)])
    basis = bedroom_basis(bedroom)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    groups = {"bedroom (0-70, 92-103 s)": [], "other room": []}
    groups.update({f"closet {a}-{b} s": [] for a, b in CENTRAL})
    groups.update({f"{a}-{b} s": [] for a, b in (STRETCHES[0], STRETCHES[5])})
    for k, p in comp0.items():
        s = walk_second(k)
        g = next((f"closet {a}-{b} s" for a, b in CENTRAL if a <= s <= b), None) \
            or next((f"{a}-{b} s" for a, b in (STRETCHES[0], STRETCHES[5]) if a <= s <= b), None) \
            or ("bedroom (0-70, 92-103 s)" if in_spans(s, BEDROOM) else "other room")
        groups[g].append(k)
    colours = {"bedroom (0-70, 92-103 s)": "#1f77b4", "other room": "#b0b0b0",
               "closet 104-135 s": "#d62728", "closet 141-148 s": "#ff7f0e", "closet 160-167 s": "#9467bd",
               "closet 174-188 s": "#8c564b", "73-91 s": "#2ca02c", "201-209 s": "#17becf"}
    order = ["other room", "bedroom (0-70, 92-103 s)", "73-91 s", "201-209 s"] + \
            [f"closet {a}-{b} s" for a, b in CENTRAL]
    for g in order:
        ks = groups[g]
        if not ks:
            continue
        xy = np.array([centre(comp0[k]) for k in ks]) @ basis
        iso = np.array([k in isolated for k in ks])
        ax.scatter(xy[~iso, 0], xy[~iso, 1], s=7, color=colours[g], label=f"{g} ({len(ks)})", alpha=.8)
        if iso.any():
            ax.scatter(xy[iso, 0], xy[iso, 1], s=16, color=colours[g], marker="x", alpha=.9)
    b2 = bedroom @ basis
    lo, hi = np.quantile(b2, .02, axis=0), np.quantile(b2, .98, axis=0)
    pad = 1.5 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    r = row
    ax.set_title(f"{r['tag']}: {r['label']} | closet ratio {_f(r['closet_ratio'])}, inside "
                 f"{_f(r['closet_inside'])}\nisolated {r['isolated_kf']} kf (x), uncertified "
                 f"{r['uncertified_kf']}, room {r['room_kf']} | solver comp 0 only", fontsize=9)
    ax.set_xlabel("bedroom PCA axis 1 (SfM units)")
    ax.set_ylabel("bedroom PCA axis 2 (SfM units)")
    ax.legend(fontsize=7, loc="best", markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _f(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def analyze(world: Path, out: Path) -> dict:
    from tower.world_builder import coherence_publish as CP

    store, wid, sid, _solve_dir = locate(world)
    keyframes = store.read_keyframes(wid, sid)
    name_of = CP._image_names(keyframes)
    ref_path = next((world.parent / "_w8r0q").rglob("solution.json"), None)
    ref_sol = json.loads(ref_path.read_text()) if ref_path else None
    rows = []
    for draw_dir in sorted(p for p in (out / "draws").iterdir() if (p / "draw.json").is_file()):
        sol, audit, meta = _load_draw(draw_dir)
        log = metric_log_of(store, wid, sid, draw_dir, keyframes)
        row = dict(tag=draw_dir.name, seed=meta.get("seed"), threads=meta.get("threads"),
                   map_s=meta.get("map_s"), guard_s=meta.get("guard_s"), started=meta.get("started_wall"),
                   ended=meta.get("ended_wall"), solver=meta.get("solver"),
                   **measure(sol, audit["assessment"], log, name_of))
        row["r0q"] = reference_deviation(sol, ref_sol) if ref_sol else None
        plot(sol, audit["assessment"], row, out / f"centres_{draw_dir.name}.png")
        rows.append(row)
    draws = [r for r in rows if not r["tag"].startswith("replay")]
    result = {"rows": rows, "selection": {}}
    for family, members in (("all", draws),
                            ("unseeded", [r for r in draws if r["seed"] is None]),
                            ("seeded", [r for r in draws if r["seed"] is not None])):
        if not members:
            continue
        result["selection"][family] = {
            "n": len(members), "good": sum(r["label"] == "GOOD" for r in members),
            "scores": {name: {"order": [r["tag"] for r in sorted(members, key=lambda r: (key(r), r["tag"]))],
                              "separation": separation(members, key),
                              "best_of_3": selection_stats(members, key, 3),
                              "best_of_2": selection_stats(members, key, 2)}
                       for name, key in SCORES.items()}}
    (out / "sel.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("draw", "replay", "analyze"))
    parser.add_argument("--world", type=artifact_root_arg, required=True,
                        help="a world copy, e.g. <worktree>/_w8g2")
    parser.add_argument("--out", type=artifact_root_arg, required=True, help="e.g. <worktree>/_out/sel")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=None, help="seeded one-thread draw (consensus style)")
    parser.add_argument("--threads", type=int, default=-1, help="unseeded mapper threads (production: -1)")
    args = parser.parse_args(argv)
    if "_out" not in Path(args.out).parts:
        parser.error("--out must be under an _out directory")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "draw":
        print(json.dumps(run_draw(args.world, args.out, tag=args.tag, seed=args.seed, threads=args.threads),
                         indent=2))
    elif args.command == "replay":
        print(json.dumps(run_replay(args.world, args.out), indent=2))
    else:
        res = analyze(args.world, args.out)
        print(json.dumps(res["selection"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
