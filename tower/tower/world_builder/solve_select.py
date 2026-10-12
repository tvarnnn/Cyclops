"""ADAPTIVE SELECT: a best-of-N final solve, chosen by the final scale guard
(`TOWER_WORLD_SOLVE_SELECT`, `tower/config.py`; brief CX-SOLVE-SELECT, evidence SEL-REPORT 2026-10-12).

WHY. On walk 8 the final mapper draw welds a closet onto the bedroom at 1.5-6.4x scale far more
often than not; the guard catches every weld by cutting the whole closet out (>= 265 keyframes
isolated), and a GOOD draw isolates 43-99. The guard's own numbers rank the good draw first (AUC
1.0 over the probe's 15 draws), so the supply of draws -- not the ranking -- is the limit. This
module buys more draws, only when draw 1 looks welded, and publishes the best.

WHAT HAPPENS, in `global_solve.solve` (ungated, single-draw final solve, guard `on`, switch
`adaptive:N`):

  1. DRAW 1 is today's final solve: today's mapping, then today's guard call
     (`final_scale_guard.guard_publication`, mode on, ungated). Its outcome is handed to the
     existing publish step, which then does not guard again.
  2. Fewer than `WORLD_SOLVE_SELECT_TRIGGER_KF` (150) keyframes isolated: NOTHING ELSE. The
     publish is byte for byte the switch-off publish (pinned by
     `tests/test_world_builder_solve_select_off_identity.py` against 41cee79).
  3. Otherwise N further draws map the ALREADY-MATCHED database -- no extraction, no matching --
     each in its own process on its OWN COPY of it (opening a COLMAP database through pycolmap
     mutates the file), at most `WORLD_SOLVE_SELECT_PARALLEL` at a time, each on every core
     (the final solve's own mapper threads; a seeded solve maps draw k with seed + k - 1 on one
     thread, as the consensus does). Each worker guards its own candidate exactly as draw 1 was.
  4. The best by the PRE-FIXED score (`score`) wins; ties go to draw 1. Only the winner is
     published, through the existing publish step (`coherence_publish.gate_and_publish`), as if
     it were the single draw: its `sparse/` replaces draw 1's, its guard outcome is the one
     published, and every later stage sees one solution.
  5. A HARD WALL BUDGET (`TOWER_WORLD_SOLVE_SELECT_BUDGET_S`, 240 s) covers the extra draws: past
     it the draws still running are killed -- the whole process tree, through a Job Object
     (`tower.process_ownership`) -- and the best finished draw so far wins. A caller's stop does
     the same (a stop never loses the finish: what is published is what a budget cut would
     publish). The selection never waits on a draw without a bound.

THE RECORD (`solve.select` in solution.json and the solve's summary, and `select` in the guard's
audit `final_scale_guard.json`), only when the selection triggered:
`{mode, n, triggered, draws: [{draw, isolated_kf, certified_kf, ratio, seconds, outcome}], winner,
added_seconds, budget_hit}`.

LOCKS AND SCRATCH. The caller holds the world's writer lock for the whole solve, selection
included. The workers write only below `<workspace>/select/` (a database copy, a `sparse/`, a
log and a result each) and publish nothing; the parent removes the directory when it is done,
and the next selection clears whatever a killed one left.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from tower.world_builder import finish_phase

logger = logging.getLogger(__name__)

SELECT_DIRNAME = "select"            # <workspace>/select/: this module's scratch, and only its
JOB_FILENAME = "job.pkl"             # what every worker needs, written once by the parent
RESULT_FILENAME = "result.pkl"       # one worker's guarded candidate
LOG_FILENAME = "worker.log"
POLL_S = 0.1
KILL_WAIT_S = 5.0
# The pre-registered score (`solve_select_probe.py`, SEL-REPORT "Pre-registration"): a plateau's
# ratio counts toward the scale term only when it is attached and measured by >= 10 cameras.
RATIO_MIN_MEASURED = 10
ATTACHED_DECISIONS = ("kept", "attached")

OUTCOME_BUDGET = "budget"
OUTCOME_STOPPED = "stopped"
OUTCOME_FAILED = "failed"


# ---------------------------------------------------------------------------
# the score (guard-only, fixed before any draw was labelled)
# ---------------------------------------------------------------------------


def _assessment(outcome) -> dict | None:
    a = (getattr(outcome, "audit", None) or {}).get("assessment")
    return a if isinstance(a, dict) else None


def isolated_keyframes(outcome) -> int | None:
    """The keyframes the guard isolated from the room; None when it made no assessment."""
    a = _assessment(outcome)
    if a is None:
        return None
    return len({k for p in a.get("pieces") or () for k in p.get("keyframe_ids") or ()})


def certified_keyframes(outcome) -> int | None:
    """The room keyframes the guard left attached AND certified (room minus uncertified)."""
    a = _assessment(outcome)
    if a is None:
        return None
    return max(0, len(a.get("room_keyframes") or ()) - int(a.get("uncertified_keyframes") or 0))


def worst_attached_ratio(outcome) -> float | None:
    """The attached, measured (>= 10) plateau ratio farthest from 1 in log terms; None without one."""
    a = _assessment(outcome) or {}
    ratios = [float(p["ratio"]) for p in a.get("plateaus") or ()
              if p.get("decision") in ATTACHED_DECISIONS and (p.get("measured") or 0) >= RATIO_MIN_MEASURED
              and isinstance(p.get("ratio"), (int, float)) and math.isfinite(p["ratio"]) and p["ratio"] > 0]
    return max(ratios, key=lambda r: abs(math.log(r)), default=None)


def publishable(outcome) -> bool:
    from tower.world_builder import final_scale_guard as FSG  # noqa: PLC0415

    return getattr(outcome, "solution", None) is not None and outcome.decision == FSG.DECISION_PUBLISHED


def score(outcome) -> tuple:
    """Lower is better: a publishable candidate first; then the FEWEST isolated keyframes; then the
    MOST certified room keyframes; then the attached scale ratio CLOSEST TO 1. Every unpublishable
    candidate scores the same, so none of them beats draw 1."""
    if not publishable(outcome):
        return (1, 0, 0, 0.0)
    ratio = worst_attached_ratio(outcome)
    return (0, isolated_keyframes(outcome) or 0, -(certified_keyframes(outcome) or 0),
            0.0 if ratio is None else abs(math.log(ratio)))


def row(draw: int, outcome, seconds: float) -> dict:
    """One draw's line of the record."""
    return {"draw": int(draw), "isolated_kf": isolated_keyframes(outcome),
            "certified_kf": certified_keyframes(outcome), "ratio": worst_attached_ratio(outcome),
            "seconds": round(float(seconds), 3), "outcome": getattr(outcome, "decision", None)}


def _cut_row(draw: int, seconds: float, outcome: str) -> dict:
    return {"draw": int(draw), "isolated_kf": None, "certified_kf": None, "ratio": None,
            "seconds": round(float(seconds), 3), "outcome": outcome}


# ---------------------------------------------------------------------------
# the selection (the parent: the final solve's own process)
# ---------------------------------------------------------------------------


def select(store, world_id: str, session_id: str, workspace, database_path, base, keyframes, *,
           n: int, budget_s: float, threads: int, seed, min_image_observations: int,
           should_stop=None):
    """`(the guard outcome to publish, the record or None)`.

    Draw 1 is `base` (the final solve's candidate, every record on it already) guarded exactly as
    the publish step guards it. Untriggered, that outcome is returned with no record and nothing
    else is read, written or run. Triggered, the extra draws run (module docstring) and the
    winner's outcome is returned, the record on its solution (`solve.select`) and its audit.

    Never raises for a failed draw, a failed launch or a failed install of the winner's model: each
    is recorded and draw 1 stands. The caller holds the world's writer lock."""
    from tower.config import WORLD_SOLVE_SELECT_TRIGGER_KF  # noqa: PLC0415
    from tower.world_builder import final_scale_guard as FSG  # noqa: PLC0415

    root = Path(workspace.root) / SELECT_DIRNAME
    if root.exists():
        # What a killed selection left (a database copy per draw): never this run's.
        shutil.rmtree(root, ignore_errors=True)
    began = time.perf_counter()
    first = FSG.guard_publication(store, world_id, session_id, base, None, mode_=FSG.MODE_ON,
                                  database_path=database_path, keyframes=keyframes)
    first_s = time.perf_counter() - began
    isolated = isolated_keyframes(first)
    if isolated is None or isolated < WORLD_SOLVE_SELECT_TRIGGER_KF:
        logger.info("[Tower][WorldBuilder] adaptive select: draw 1 isolated %s keyframes (< %s); "
                    "draw 1 is published as today", isolated, WORLD_SOLVE_SELECT_TRIGGER_KF)
        return first, None
    logger.warning("[Tower][WorldBuilder] adaptive select: draw 1 isolated %s keyframes (>= %s); "
                   "mapping %s further draws (budget %g s)", isolated, WORLD_SOLVE_SELECT_TRIGGER_KF,
                   n, budget_s)
    started = time.perf_counter()
    rows = {1: row(1, first, first_s)}
    outcomes = {1: first}
    budget_hit = False
    try:
        _run_draws(store, world_id, session_id, workspace, database_path, base, keyframes, root,
                   rows=rows, outcomes=outcomes, n=n, budget_s=budget_s, threads=threads, seed=seed,
                   min_image_observations=min_image_observations, should_stop=should_stop)
        budget_hit = any(r["outcome"] == OUTCOME_BUDGET for r in rows.values())
        winner = min(outcomes, key=lambda k: (score(outcomes[k]), k))
        if winner != 1 and not _install_sparse(workspace, root / f"d{winner}" / "sparse"):
            rows[winner] = dict(rows[winner], outcome=f"{OUTCOME_FAILED}: its model could not be installed")
            outcomes.pop(winner)
            winner = min(outcomes, key=lambda k: (score(outcomes[k]), k))
            if winner != 1 and not _install_sparse(workspace, root / f"d{winner}" / "sparse"):
                winner = 1
    except Exception as exc:  # noqa: BLE001 -- the selection never costs the finish: draw 1 stands
        logger.exception("[Tower][WorldBuilder] adaptive select failed; draw 1 is published")
        for k in range(2, n + 2):
            rows.setdefault(k, _cut_row(k, 0.0, f"{OUTCOME_FAILED}: {type(exc).__name__}: {exc}"))
        winner = 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
    for k in range(2, n + 2):
        rows.setdefault(k, _cut_row(k, 0.0, f"{OUTCOME_FAILED}: never reported"))
    record = {"mode": "adaptive", "n": int(n), "triggered": True,
              "draws": [rows[k] for k in sorted(rows)], "winner": int(winner),
              "added_seconds": round(time.perf_counter() - started, 3), "budget_hit": bool(budget_hit)}
    chosen = outcomes[winner]
    if chosen.solution is not None:
        # `chosen.result.solution` is this same object (`guard_publication`): one assignment covers both.
        chosen.solution.solve = dict(chosen.solution.solve or {}, select=record)
    chosen.audit = dict(chosen.audit or {}, select=record)
    logger.warning("[Tower][WorldBuilder] adaptive select: draw %s of %s wins %s (+%.1f s%s)", winner,
                   n + 1, [(r["draw"], r["isolated_kf"], r["outcome"]) for r in record["draws"]],
                   record["added_seconds"], ", budget hit" if budget_hit else "")
    return chosen, record


def _run_draws(store, world_id, session_id, workspace, database_path, base, keyframes, root: Path, *,
               rows: dict, outcomes: dict, n: int, budget_s: float, threads: int, seed,
               min_image_observations: int, should_stop) -> None:
    """Map and guard draws 2..n+1 into `rows` / `outcomes`, at most `WORLD_SOLVE_SELECT_PARALLEL` at
    once, until they are done, the budget is spent or `should_stop()`. Every draw still running when
    this returns -- or raises -- has been killed, process tree and all."""
    from tower.config import WORLD_SOLVE_SELECT_PARALLEL  # noqa: PLC0415

    root.mkdir(parents=True, exist_ok=True)
    job_path = root / JOB_FILENAME
    _write_pickle(job_path, {
        "store_root": str(store.root), "world_id": world_id, "session_id": session_id,
        "workspace": str(workspace.root), "database": str(database_path), "keyframes": list(keyframes),
        "camera": base.camera, "input_digest": base.input_digest, "solve": base.solve,
        "transients": base.transients, "timing": base.timing, "threads": threads,
        "seed": seed, "min_image_observations": min_image_observations,
    })
    deadline = time.perf_counter() + float(budget_s)
    parallel = max(1, min(int(WORLD_SOLVE_SELECT_PARALLEL), int(n), os.cpu_count() or 1))
    pending = deque(range(2, n + 2))
    running: dict = {}
    cut = OUTCOME_STOPPED
    finish_phase.mark(finish_phase.CHECKING, (1, n + 1))
    try:
        while pending or running:
            if should_stop is not None and should_stop():
                logger.warning("[Tower][WorldBuilder] adaptive select: a stop was asked for; the draws "
                               "still running are killed")
                break
            if time.perf_counter() >= deadline:
                cut = OUTCOME_BUDGET
                logger.warning("[Tower][WorldBuilder] adaptive select: the %g s budget is spent; the "
                               "draws still running are killed", budget_s)
                break
            while pending and len(running) < parallel:
                k = pending.popleft()
                try:
                    running[k] = (_start(k, root / f"d{k}", job_path), time.perf_counter())
                except Exception as exc:  # noqa: BLE001 -- a draw that cannot start is a failed draw
                    logger.warning("[Tower][WorldBuilder] adaptive select: draw %s could not start: %s", k, exc)
                    rows[k] = _cut_row(k, 0.0, f"{OUTCOME_FAILED}: {type(exc).__name__}: {exc}")
            for k, (handle, at) in list(running.items()):
                code = handle.poll()
                if code is None:
                    continue
                del running[k]
                handle.release()
                rows[k], outcome = _collect(k, root / f"d{k}", code, time.perf_counter() - at)
                if outcome is not None:
                    outcomes[k] = outcome
                finish_phase.mark(finish_phase.CHECKING, (min(len(rows), n + 1), n + 1))
            if running:
                time.sleep(max(0.0, min(POLL_S, deadline - time.perf_counter())))
    finally:
        for k, (handle, at) in running.items():
            code = handle.poll()
            if code is not None:
                # It finished in the last poll interval: it counts, as it would have a moment earlier.
                handle.release()
                rows[k], outcome = _collect(k, root / f"d{k}", code, time.perf_counter() - at)
                if outcome is not None:
                    outcomes[k] = outcome
                continue
            handle.kill()
            rows[k] = _cut_row(k, time.perf_counter() - at, cut)
        for k in pending:
            rows[k] = _cut_row(k, 0.0, cut)


def _collect(draw: int, draw_dir: Path, code: int, seconds: float):
    """(the draw's row, its guard outcome or None) from a finished worker."""
    try:
        if code != 0:
            raise RuntimeError(f"the worker exited {code} (see {LOG_FILENAME})")
        with (draw_dir / RESULT_FILENAME).open("rb") as handle:
            result = pickle.load(handle)  # noqa: S301 -- written by this solve's own worker, below its workspace
        outcome = result["outcome"]
        if int(result["draw"]) != draw:
            raise RuntimeError(f"the result names draw {result['draw']}")
        return row(draw, outcome, seconds), outcome
    except Exception as exc:  # noqa: BLE001 -- one failed draw does not lose the finish
        logger.warning("[Tower][WorldBuilder] adaptive select: draw %s failed: %s", draw, exc)
        return _cut_row(draw, seconds, f"{OUTCOME_FAILED}: {type(exc).__name__}: {exc}"), None


def _install_sparse(workspace, sparse: Path) -> bool:
    """The winner's model becomes the workspace's `sparse/` (draw 1's, until now). Restores draw 1's
    on failure; True when the winner's is in place."""
    target = Path(workspace.sparse_dir)
    aside = target.with_name(target.name + ".select-prev")
    try:
        if not sparse.is_dir():
            raise FileNotFoundError(sparse)
        shutil.rmtree(aside, ignore_errors=True)
        if target.exists():
            os.replace(target, aside)
        try:
            shutil.move(str(sparse), str(target))
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            if aside.exists():
                os.replace(aside, target)
            raise
        shutil.rmtree(aside, ignore_errors=True)
        return True
    except Exception:  # noqa: BLE001 -- recorded by the caller; draw 1 stands
        logger.exception("[Tower][WorldBuilder] adaptive select: the winning model could not replace %s",
                         target)
        return False


# ---------------------------------------------------------------------------
# one draw's process
# ---------------------------------------------------------------------------


class _Draw:
    """A launched worker and the Job Object that owns its tree."""

    def __init__(self, process, job) -> None:
        self.process = process
        self.job = job

    def poll(self):
        return self.process.poll()

    def kill(self) -> None:
        """Kill the worker's whole tree, bounded; never raises."""
        from tower.process_ownership import terminate_tree  # noqa: PLC0415

        try:
            terminate_tree(self.process, job=self.job, timeout=KILL_WAIT_S, hard=True)
        except Exception:  # noqa: BLE001 -- never let a kill hang or raise the selection
            logger.warning("[Tower][WorldBuilder] adaptive select: could not kill pid %s",
                           getattr(self.process, "pid", "?"), exc_info=True)
        self.release()

    def release(self) -> None:
        if self.job is not None:
            try:
                self.job.close()
            except Exception:  # noqa: BLE001
                pass
            self.job = None


def worker_environment() -> dict:
    """The worker's environment: this process's (the guard's switches with it), venv-aware, no GPU,
    and no finish-phase token -- a worker never marks the phone's finish phase."""
    from tower.process_ownership import interpreter_environment  # noqa: PLC0415

    env = interpreter_environment()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env.pop(finish_phase.TOKEN_ENV, None)
    return env


def worker_argv(job_path: Path, draw: int) -> tuple:
    from tower.process_ownership import interpreter_command  # noqa: PLC0415

    return interpreter_command("-m", "tower.world_builder.solve_select", str(job_path), str(draw))


def _start(draw: int, draw_dir: Path, job_path: Path) -> _Draw:
    """Launch draw `draw` as ONE process (no venv launcher pair), in a Job Object of its own."""
    from tower.config import TOWER_ROOT  # noqa: PLC0415
    from tower.process_ownership import assign_to_job  # noqa: PLC0415

    draw_dir.mkdir(parents=True, exist_ok=True)
    with (draw_dir / LOG_FILENAME).open("ab") as log:
        process = subprocess.Popen(worker_argv(job_path, draw), cwd=str(TOWER_ROOT), env=worker_environment(),
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    return _Draw(process, assign_to_job(process))


def _write_pickle(path: Path, value) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def run_draw(job_path: Path, draw: int) -> None:
    """ONE extra draw, in its own process: copy the matched database, map the copy exactly as the
    final solve maps (`global_solve._map_candidate`), carry draw 1's records, guard the candidate
    exactly as draw 1 was guarded, and leave the outcome in `d<draw>/result.pkl`. Publishes
    nothing; writes only below `d<draw>/`."""
    import pycolmap  # noqa: PLC0415

    from tower.world_builder import final_scale_guard as FSG  # noqa: PLC0415
    from tower.world_builder import global_solve as GS  # noqa: PLC0415
    from tower.world_builder.store import WorldStore  # noqa: PLC0415

    GS._quiet_pycolmap()
    with Path(job_path).open("rb") as handle:
        job = pickle.load(handle)  # noqa: S301 -- written by the parent solve, below its workspace
    started = time.perf_counter()
    draw_dir = Path(job_path).parent / f"d{draw}"
    draw_dir.mkdir(parents=True, exist_ok=True)
    source = Path(job["database"])
    database = draw_dir / source.name           # the same name: the guard's audit records it
    shutil.copyfile(source, database)
    for suffix in ("-wal", "-shm"):
        side = source.with_name(source.name + suffix)
        if side.exists():
            shutil.copyfile(side, database.with_name(database.name + suffix))
    seed = job["seed"]
    draw_seed = None if seed is None else int(seed) + int(draw) - 1
    threads = 1 if draw_seed is not None else job["threads"]
    mapped_at = time.perf_counter()
    candidate = GS._map_candidate(
        pycolmap, database, GS.SolveWorkspace(Path(job["workspace"])), draw_dir / "sparse", job["keyframes"],
        seed=draw_seed, threads=threads, input_digest=job["input_digest"],
        min_image_observations=job["min_image_observations"],
        camera=GS.PinholeCamera.from_json_dict(job["camera"]))
    map_s = round(time.perf_counter() - mapped_at, 3)
    # Draw 1's records: the same database, masks and matching; this draw's own seed and mapping time.
    candidate.transients = job["transients"]
    candidate.solve = (None if job["solve"] is None else
                       dict(job["solve"], **({} if draw_seed is None else {"seed": draw_seed})))
    candidate.timing = dict(job["timing"] or {}, map_s=map_s)
    outcome = FSG.guard_publication(WorldStore(Path(job["store_root"])), job["world_id"], job["session_id"],
                                    candidate, None, mode_=FSG.MODE_ON, database_path=database,
                                    keyframes=job["keyframes"])
    _write_pickle(draw_dir / RESULT_FILENAME, {"draw": int(draw), "outcome": outcome, "map_s": map_s,
                                               "seconds": round(time.perf_counter() - started, 3)})
    logger.info("adaptive select draw %s: %s, %s keyframes isolated, map %.1f s", draw, outcome.decision,
                isolated_keyframes(outcome), map_s)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or not args[1].isdigit():
        print("usage: python -m tower.world_builder.solve_select JOB_PKL DRAW", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_draw(Path(args[0]), int(args[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
