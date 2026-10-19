"""BEST-OF-N FINAL SOLVE (`world_builder/solve_select.py`, `TOWER_WORLD_SOLVE_SELECT`): the final solve's
mapper draw, chosen by the final scale guard. No GPU, no Tower; real pycolmap is never run.
Pinned here (restored from 41727b73^ by lane P, patchy run 2026-10-19; the guard is always on now):
  * the setting: unset or blank is the default `adaptive:3`; exact `adaptive:N`, N 1..6; a spelling
    of false is off, anything else is off; the budget's bounds;
  * the trigger: the 60-keyframe floor whenever the room size is known (max(60, ceil(0 x room)):
    walk-7/walk-8-like draws, walk-10.1-like draws with the low-texture stretches cut or intact, a
    tiny walk); without a room size, the absolute rule (149 isolated runs nothing, 150 runs the
    extra draws);
  * the pre-fixed score (fewest isolated, most certified, ratio closest to 1) and ties to draw 1;
  * the record's shape, on the solution and the audit;
  * the hard budget kills the draws still running and keeps the best finished one;
  * a stop during selection kills the draws, leaves no scratch and publishes the best finished one;
  * a failed draw, a failed launch or a failed selection never costs draw 1;
  * END TO END through `global_solve.solve` with REAL worker processes (a test shim stands in for
    the mapper and the depth evidence inside them): the winner is published through the existing
    publish step with its own `sparse/`, the walk database is never mapped by an extra draw, and
    below the trigger nothing is launched;
  * the Job Object: the budget kills a real worker's whole tree, and a parent that dies takes its
    draws with it.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_world_builder_solve_masks import colmap  # noqa: F401 -- fixture
from tower import config
from tower.world_builder import final_scale_guard as FSG
from tower.world_builder import finish_phase
from tower.world_builder import solve_select as SS

SELECT = config.WORLD_SOLVE_SELECT_ENV
BUDGET = config.WORLD_SOLVE_SELECT_BUDGET_ENV
SHIM_ENV = "WB_SOLVE_SELECT_TEST_SHIM"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def outcome(isolated, certified=100, ratio=1.0, decision="published", tag=None, judged=None):
    """A guard outcome as `guard_publication` shapes its audit (`judged`: the assessment's
    `room_candidate`; None leaves it out, as a pre-fraction fake did)."""
    pieces = [{"keyframe_ids": [f"k{i}" for i in range(isolated)]}] if isolated else []
    extra = {} if judged is None else {"room_candidate": judged}
    return SimpleNamespace(
        audit={"assessment": {
            **extra,
            "pieces": pieces, "room_keyframes": [f"r{i}" for i in range(certified)],
            "uncertified_keyframes": 0,
            "plateaus": [{"decision": "kept", "measured": 20, "ratio": ratio}],
        }},
        solution=None if decision != "published" else SimpleNamespace(solve={"seed": None}, solver=tag or "glomap"),
        result=None, decision=decision, summary={})


class Done:
    def __init__(self, code=0):
        self.code = code
        self.killed = False

    def poll(self):
        return self.code

    def kill(self):
        raise AssertionError("a finished draw is never killed")

    def release(self):
        pass


class Hung:
    def __init__(self, log):
        self.log = log

    def poll(self):
        return None

    def kill(self):
        self.log.append(self)

    def release(self):
        pass


def finished(draw_dir, draw, result):
    """A worker that already wrote its result."""
    draw_dir.mkdir(parents=True, exist_ok=True)
    (draw_dir / "sparse").mkdir(exist_ok=True)
    (draw_dir / "sparse" / "draw.txt").write_text(str(draw))
    with (draw_dir / SS.RESULT_FILENAME).open("wb") as handle:
        pickle.dump({"draw": draw, "outcome": result, "map_s": 0.01, "seconds": 0.01}, handle)
    return Done()


@pytest.fixture
def scene(tmp_path, monkeypatch):
    base = SimpleNamespace(camera={}, input_digest="digest", timing={}, solve={"seed": None}, transients=None)
    workspace = SimpleNamespace(root=tmp_path / "solve", sparse_dir=tmp_path / "solve" / "sparse")
    workspace.sparse_dir.mkdir(parents=True)
    (workspace.sparse_dir / "draw.txt").write_text("1")
    db = workspace.root / "database.db"
    db.write_bytes(b"matched")
    store = SimpleNamespace(root=tmp_path / "store")
    first = outcome(200, certified=50)
    monkeypatch.setattr(FSG, "guard_publication", lambda *a, **k: first)
    for name in (SELECT, BUDGET):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(store=store, workspace=workspace, db=db, base=base, first=first)


def run(scene, **kw):
    return SS.select(scene.store, "world", "session", scene.workspace, scene.db, scene.base, [],
                     n=kw.pop("n", 2), budget_s=kw.pop("budget_s", 5.0), threads=-1, seed=None,
                     min_image_observations=30, **kw)


# ---------------------------------------------------------------------------
# the setting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (None, ("adaptive", 3)), ("", ("adaptive", 3)), ("  ", ("adaptive", 3)),
    ("off", ("off", 0)), ("false", ("off", 0)), ("0", ("off", 0)),
    ("adaptive:1", ("adaptive", 1)), ("adaptive:3", ("adaptive", 3)), (" Adaptive:6 ", ("adaptive", 6)),
    ("adaptive:0", ("off", 0)), ("adaptive:7", ("off", 0)), ("adaptive:01", ("off", 0)),
    ("adaptive", ("off", 0)), ("adaptive:", ("off", 0)), ("on", ("off", 0)), ("3", ("off", 0)),
])
def test_the_setting_is_exact(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(SELECT, raising=False)
    else:
        monkeypatch.setenv(SELECT, value)
    assert config.world_solve_select_setting() == expected


@pytest.mark.parametrize("value,expected", [
    (None, 240.0), ("", 240.0), ("90", 90.0), ("0.5", 0.5), ("3600", 3600.0),
    ("0", 240.0), ("-5", 240.0), ("nan", 240.0), ("inf", 240.0), ("3601", 240.0), ("soon", 240.0),
])
def test_the_budget_setting(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(BUDGET, raising=False)
    else:
        monkeypatch.setenv(BUDGET, value)
    assert config.world_solve_select_budget_setting() == expected


def test_the_trigger_and_the_wave_are_config_constants():
    assert config.WORLD_SOLVE_SELECT_TRIGGER_KF == 150
    assert config.WORLD_SOLVE_SELECT_TRIGGER_FRAC == 0.0
    assert config.WORLD_SOLVE_SELECT_DEFAULT_DRAWS == config.WORLD_SOLVE_SELECT_PARALLEL
    assert config.WORLD_SOLVE_SELECT_TRIGGER_MIN_KF == 60
    assert config.WORLD_SOLVE_SELECT_PARALLEL == 3


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------


def test_the_pre_fixed_score_order():
    # fewest isolated first, whatever else
    assert SS.score(outcome(40, 5, 2.0)) < SS.score(outcome(41, 1000, 1.0))
    # then the most certified
    assert SS.score(outcome(40, 10, 2.0)) < SS.score(outcome(40, 9, 1.0))
    # then the attached ratio closest to 1, in log terms either side
    assert SS.score(outcome(40, 10, 1.1)) < SS.score(outcome(40, 10, 0.8))
    assert SS.score(outcome(40, 10, 1 / 1.1)) == pytest.approx(SS.score(outcome(40, 10, 1.1)))
    # a withheld candidate never beats a published one, and all withheld ones tie
    assert SS.score(outcome(10_000, 0, 9.0)) < SS.score(outcome(0, 1000, 1.0, decision="withheld"))
    assert SS.score(outcome(1, decision="withheld")) == SS.score(outcome(500, decision="withheld"))


def test_only_attached_well_measured_plateaus_count_toward_the_ratio():
    o = outcome(0)
    o.audit["assessment"]["plateaus"] = [
        {"decision": "kept", "measured": 20, "ratio": 1.2},
        {"decision": "attached", "measured": 12, "ratio": 0.7},       # the worst attached: counts
        {"decision": "attached", "measured": 9, "ratio": 5.0},        # too few ratios
        {"decision": "isolated", "measured": 50, "ratio": 3.0},       # not in the room
        {"decision": "attached", "measured": 0, "ratio": None},       # unmeasured
    ]
    assert SS.worst_attached_ratio(o) == 0.7
    assert SS.row(4, o, 1.23456) == {"draw": 4, "isolated_kf": 0, "certified_kf": 100, "ratio": 0.7,
                                     "seconds": 1.235, "outcome": "published"}


# ---------------------------------------------------------------------------
# the selection, with fake workers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("isolated,triggered", [(149, False), (150, True)])
def test_without_a_room_size_the_trigger_is_150_isolated_keyframes(scene, monkeypatch, isolated, triggered):
    first = outcome(isolated)
    assert SS.trigger_threshold(first) == {"rule": "absolute", "base_kf": None, "threshold_kf": 150}
    monkeypatch.setattr(FSG, "guard_publication", lambda *a, **k: first)
    launches = []
    monkeypatch.setattr(SS, "_start", lambda draw, d, job: (launches.append(draw), Done(1))[1])
    chosen, record = run(scene, n=2)
    assert chosen is first
    assert launches == ([2, 3] if triggered else [])
    assert (record is not None) is triggered
    if not triggered:
        assert first.solution.solve == {"seed": None} and "select" not in first.audit
    assert not (scene.workspace.root / SS.SELECT_DIRNAME).exists()


@pytest.mark.parametrize("judged,threshold", [(972, 60), (937, 60), (1400, 60), (200, 60), (1000, 60),
                                              (1232, 60), (0, 150), (True, 150), (-5, 150), (12.5, 150)])
def test_the_trigger_threshold(judged, threshold):
    t = SS.trigger_threshold(outcome(0, judged=judged))
    assert t["threshold_kf"] == threshold
    assert t["rule"] == ("fraction" if threshold != 150 else "absolute")
    assert t["base_kf"] == (judged if t["rule"] == "fraction" else None)
    assert SS.judged_keyframes(SimpleNamespace(audit={})) is None


@pytest.mark.parametrize("isolated,judged,triggered", [
    (83, 972, True), (429, 972, True),         # walk 7: a good draw and the n1 weld: both buy draws now
    (99, 937, True), (265, 937, True),         # walk 8: the worst good draw, the smallest weld
    (26, 1222, False), (51, 1229, False),      # walk 10.1: low-texture stretches intact
    (100, 1222, True), (259, 1232, True),      # walk 10.1: low-texture stretches cut (8 % and 21 %)
    (40, 200, False), (59, 200, False), (60, 200, True),   # a tiny walk: the 60-keyframe floor
])
def test_the_trigger_is_the_60_keyframe_floor(scene, monkeypatch, isolated, judged, triggered):
    first = outcome(isolated, judged=judged)
    monkeypatch.setattr(FSG, "guard_publication", lambda *a, **k: first)
    launches = []
    monkeypatch.setattr(SS, "_start", lambda draw, d, job: (launches.append(draw), Done(1))[1])
    chosen, record = run(scene, n=2)
    assert chosen is first
    assert launches == ([2, 3] if triggered else [])
    assert (record is not None) is triggered
    if triggered:
        assert record["trigger"] == SS.trigger_threshold(first)
        assert record["trigger"]["rule"] == "fraction" and record["trigger"]["base_kf"] == judged
    else:
        assert first.solution.solve == {"seed": None} and "select" not in first.audit


def test_a_tie_goes_to_draw_1_and_the_record_has_its_shape(scene, monkeypatch):
    same = outcome(200, certified=50)
    monkeypatch.setattr(SS, "_start", lambda draw, d, job: finished(d, draw, outcome(200, certified=50)))
    assert SS.score(same) == SS.score(scene.first)
    chosen, record = run(scene, n=2)
    assert chosen is scene.first
    assert set(record) == {"mode", "n", "triggered", "trigger", "draws", "winner", "added_seconds", "budget_hit"}
    assert record["trigger"] == {"rule": "absolute", "base_kf": None, "threshold_kf": 150}
    assert record["mode"] == "adaptive" and record["n"] == 2 and record["triggered"] is True
    assert record["winner"] == 1 and record["budget_hit"] is False
    assert isinstance(record["added_seconds"], float)
    assert [r["draw"] for r in record["draws"]] == [1, 2, 3]
    for r in record["draws"]:
        assert set(r) == {"draw", "isolated_kf", "certified_kf", "ratio", "seconds", "outcome"}
        assert (r["isolated_kf"], r["certified_kf"], r["ratio"], r["outcome"]) == (200, 50, 1.0, "published")
    # the record travels on the published solution and in the audit; JSON as written
    assert chosen.solution.solve["select"] == record and chosen.audit["select"] == record
    assert json.loads(json.dumps(record)) == record
    # draw 1's model stays; the scratch is gone
    assert (scene.workspace.sparse_dir / "draw.txt").read_text() == "1"
    assert not (scene.workspace.root / SS.SELECT_DIRNAME).exists()


def test_the_best_draw_wins_and_its_model_replaces_draw_1s(scene, monkeypatch):
    results = {2: outcome(280), 3: outcome(43, certified=600), 4: outcome(43, certified=500)}
    monkeypatch.setattr(SS, "_start", lambda draw, d, job: finished(d, draw, results[draw]))
    chosen, record = run(scene, n=3)
    assert record["winner"] == 3 and (SS.isolated_keyframes(chosen), SS.certified_keyframes(chosen)) == (43, 600)
    assert chosen.solution.solve["select"] == record
    assert (scene.workspace.sparse_dir / "draw.txt").read_text() == "3"
    assert not scene.workspace.sparse_dir.with_name("sparse.select-prev").exists()
    assert not (scene.workspace.root / SS.SELECT_DIRNAME).exists()
    assert [r["isolated_kf"] for r in record["draws"]] == [200, 280, 43, 43]


def test_the_job_carries_draw_1s_records_and_the_matched_database(scene, monkeypatch):
    jobs = []

    def start(draw, d, job_path):
        with Path(job_path).open("rb") as handle:
            jobs.append(pickle.load(handle))
        return Done(1)

    monkeypatch.setattr(SS, "_start", start)
    scene.base.solve = {"seed": 7, "threads": 1, "seeded": True}
    scene.base.transients = {"state": "off"}
    scene.base.timing = {"map_s": 3.0}
    SS.select(scene.store, "world", "session", scene.workspace, scene.db, scene.base, ["kf"], n=1,
              budget_s=5.0, threads=1, seed=7, min_image_observations=30)
    (job,) = jobs
    assert job["database"] == str(scene.db) and job["keyframes"] == ["kf"] and job["seed"] == 7
    assert job["solve"] == scene.base.solve and job["transients"] == {"state": "off"}
    assert job["timing"] == {"map_s": 3.0} and job["threads"] == 1


def test_the_budget_kills_the_draws_still_running_and_keeps_the_best_so_far(scene, monkeypatch):
    killed = []
    good = outcome(60, certified=300)

    def start(draw, d, job):
        return finished(d, draw, good) if draw == 2 else Hung(killed)

    monkeypatch.setattr(SS, "_start", start)
    started = time.perf_counter()
    chosen, record = run(scene, n=5, budget_s=0.3)
    assert time.perf_counter() - started < 3.0
    assert record["winner"] == 2 and SS.isolated_keyframes(chosen) == 60
    assert record["budget_hit"] is True
    assert len(killed) == 3                                   # 3 and 4, and 5 in the slot 2 freed
    outcomes = {r["draw"]: r["outcome"] for r in record["draws"]}
    assert outcomes == {1: "published", 2: "published", 3: "budget", 4: "budget", 5: "budget", 6: "budget"}
    assert {r["draw"]: r["seconds"] for r in record["draws"]}[6] == 0.0      # never started
    assert record["added_seconds"] >= 0.3
    assert not (scene.workspace.root / SS.SELECT_DIRNAME).exists()


def test_the_waves_are_three_wide(scene, monkeypatch):
    live, peak = set(), []

    class Slow:
        def __init__(self, draw, d):
            self.draw, self.d, self.polls = draw, d, 0
            live.add(draw)
            peak.append(len(live))

        def poll(self):
            self.polls += 1
            if self.polls < 3:
                return None
            live.discard(self.draw)
            finished(self.d, self.draw, outcome(300))
            return 0

        def kill(self):
            raise AssertionError

        def release(self):
            pass

    monkeypatch.setattr(SS, "POLL_S", 0.0)
    monkeypatch.setattr(SS, "_start", lambda draw, d, job: Slow(draw, d))
    _, record = run(scene, n=6)
    assert max(peak) == min(3, os.cpu_count() or 1)
    assert [r["outcome"] for r in record["draws"]] == ["published"] * 7 and record["winner"] == 1


def test_a_stop_during_selection_kills_the_draws_and_publishes_the_best_so_far(scene, monkeypatch):
    killed = []
    asked = iter([False, False, True])
    monkeypatch.setattr(SS, "_start", lambda draw, d, job: Hung(killed))
    chosen, record = run(scene, n=2, should_stop=lambda: next(asked, True))
    assert chosen is scene.first and record["winner"] == 1
    assert len(killed) == 2 and record["budget_hit"] is False
    assert [r["outcome"] for r in record["draws"]] == ["published", "stopped", "stopped"]
    assert not (scene.workspace.root / SS.SELECT_DIRNAME).exists()
    assert (scene.workspace.sparse_dir / "draw.txt").read_text() == "1"


def test_failed_draws_and_launches_never_cost_draw_1(scene, monkeypatch):
    def start(draw, d, job):
        if draw == 2:
            raise OSError("no process for you")
        if draw == 3:
            return Done(3)                                    # e.g. a COLMAP abort
        d.mkdir(parents=True, exist_ok=True)
        (d / SS.RESULT_FILENAME).write_bytes(b"not a pickle")
        return Done(0)

    monkeypatch.setattr(SS, "_start", start)
    chosen, record = run(scene, n=3)
    assert chosen is scene.first and record["winner"] == 1
    outcomes = [r["outcome"] for r in record["draws"]]
    assert outcomes[0] == "published" and all(o.startswith("failed: ") for o in outcomes[1:])
    assert "exited 3" in outcomes[2]


def test_a_selection_that_breaks_publishes_draw_1(scene, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(SS, "_write_pickle", broken)
    chosen, record = run(scene, n=2)
    assert chosen is scene.first and record["winner"] == 1
    assert [r["outcome"] for r in record["draws"]][1:] == ["failed: RuntimeError: disk full"] * 2


def test_a_winner_whose_model_cannot_be_installed_yields_to_the_next(scene, monkeypatch):
    results = {2: outcome(40, certified=600), 3: outcome(50)}

    def start(draw, d, job):
        done = finished(d, draw, results[draw])
        if draw == 2:
            import shutil
            shutil.rmtree(d / "sparse")
        return done

    monkeypatch.setattr(SS, "_start", start)
    chosen, record = run(scene, n=2)
    assert record["winner"] == 3 and SS.isolated_keyframes(chosen) == 50
    assert record["draws"][1]["outcome"].startswith("failed: ")
    assert (scene.workspace.sparse_dir / "draw.txt").read_text() == "3"


def test_a_withheld_draw_1_still_triggers_and_a_published_draw_wins(scene, monkeypatch):
    withheld = outcome(400, decision="withheld")
    monkeypatch.setattr(FSG, "guard_publication", lambda *a, **k: withheld)
    good = outcome(400)
    monkeypatch.setattr(SS, "_start", lambda draw, d, job: finished(d, draw, good))
    chosen, record = run(scene, n=1)
    assert record["winner"] == 2 and chosen.decision == "published" and chosen.solution.solve["select"] == record


def test_the_selection_marks_the_finish_phase_as_checking(scene, monkeypatch):
    marks = []
    monkeypatch.setattr(SS, "_start", lambda draw, d, job: finished(d, draw, outcome(300)))
    token = finish_phase.install(SimpleNamespace(mark=lambda stage, step=None: marks.append((stage, step))))
    try:
        run(scene, n=2)
    finally:
        finish_phase.uninstall(token)
    assert marks[0] == ("checking", (1, 3)) and marks[-1] == ("checking", (3, 3))
    assert all(stage == "checking" for stage, _ in marks)


def test_the_worker_maps_its_own_copy_and_writes_only_below_its_draw(tmp_path, monkeypatch):
    from tower.world_builder import global_solve as GS
    from tower.world_builder.store import WorldStore

    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace())
    ws = tmp_path / "solve"
    root = ws / SS.SELECT_DIRNAME
    root.mkdir(parents=True)
    db = ws / "database.db"
    db.write_bytes(b"matched")
    job = root / SS.JOB_FILENAME
    SS._write_pickle(job, {
        "store_root": str(tmp_path / "store"), "world_id": "w", "session_id": "s", "workspace": str(ws),
        "database": str(db), "keyframes": ["kf"], "input_digest": "digest", "min_image_observations": 30,
        "camera": {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "width": 2, "height": 2},
        "solve": {"seed": 4, "threads": 1}, "transients": {"state": "off"}, "timing": {"map_s": 9.0},
        "threads": 1, "seed": 4})
    mapped, guarded = [], []

    def map_copy(pycolmap, database, workspace, sparse, keyframes, *, seed, threads, **kw):
        mapped.append((database, workspace.root, sparse, keyframes, seed, threads))
        return SimpleNamespace(solver="glomap", camera=None)

    def guard(store, world_id, session_id, candidate, result, *, database_path, keyframes):
        guarded.append((candidate, result, database_path, keyframes))
        return SimpleNamespace(decision="published", solution=candidate, audit={})

    monkeypatch.setattr(GS, "_map_candidate", map_copy)
    monkeypatch.setattr(FSG, "guard_publication", guard)
    monkeypatch.setattr(WorldStore, "__init__", lambda self, root: None)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert SS.main([str(job), "3"]) == 0
    d3 = root / "d3"
    ((database, wsroot, sparse, keyframes, seed, threads),) = mapped
    assert database == d3 / "database.db" and database.read_bytes() == b"matched"
    assert wsroot == ws and sparse == d3 / "sparse" and keyframes == ["kf"]
    assert (seed, threads) == (6, 1)                          # seeded: seed + draw - 1, one thread
    ((candidate, result, guard_db, _),) = guarded
    assert result is None and guard_db == d3 / "database.db"
    assert candidate.solve == {"seed": 6, "threads": 1} and candidate.transients == {"state": "off"}
    assert candidate.timing["map_s"] >= 0 and set(candidate.timing) == {"map_s"}
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert all(p in before or d3.relative_to(tmp_path) in p.parents or p == d3.relative_to(tmp_path)
               for p in after)
    with (d3 / SS.RESULT_FILENAME).open("rb") as handle:
        assert pickle.load(handle)["draw"] == 3
    assert db.read_bytes() == b"matched"


def test_the_worker_environment_has_no_gpu_and_no_phase_token(monkeypatch):
    monkeypatch.setenv(finish_phase.TOKEN_ENV, "{}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = SS.worker_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == "-1" and finish_phase.TOKEN_ENV not in env
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"          # a copy


# ---------------------------------------------------------------------------
# end to end, through global_solve.solve, with real worker processes
# ---------------------------------------------------------------------------

N_E2E = 360
# Draw k's `solved_at` (the shim's): a wall clock later than any launch, as a real solve's is.
SHIM_CLOCK = 4_000_000_000.0
WELD = (200, N_E2E)        # 160 keyframes at 3x: draw 1 isolates them (>= max(60, 0 x 360))
CLEAN_DRAW = 2


def _draw_of_sparse(sparse_dir) -> int:
    sparse_dir = Path(sparse_dir)
    parent = sparse_dir.parent
    if parent.parent.name == SS.SELECT_DIRNAME and parent.name.startswith("d"):
        return int(parent.name[1:])
    return 1


def install_shim(n: int, weld, clean_draws, log_path=None, monkeypatch=None):
    """The mapper and the live depth evidence, faked the same way in the test and in each worker: draw k's
    candidate has `solved_at = SHIM_CLOCK + k` (after any launch) and writes `sparse/draw.txt`; its evidence welds
    `weld` at 3x unless k is clean."""
    from tests.test_world_builder_final_scale_guard import _levels, _walk_centers
    from tests.test_world_builder_final_scale_guard_seam import _candidate
    from tower.world_builder import global_solve as GS

    centers = _walk_centers(n, step=0.01)
    welded = _levels(n, [(weld[0], weld[1], 3.0)], noise=0.05, seed=1)
    clean = _levels(n, [], noise=0.05, seed=1)

    def note(row):
        if log_path:
            # One file per process: Windows appends are seek-then-write, and parallel draws would race.
            with open(f"{log_path}.{os.getpid()}", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

    def map_candidate(pycolmap, database_path, workspace, sparse_dir, keyframes, *, seed, threads,
                      input_digest, min_image_observations, camera):
        draw = _draw_of_sparse(sparse_dir)
        Path(sparse_dir).mkdir(parents=True, exist_ok=True)
        (Path(sparse_dir) / "draw.txt").write_text(str(draw))
        note({"map": draw, "database": str(database_path), "seed": seed, "threads": threads, "pid": os.getpid()})
        if draw != 1 and os.environ.get("WB_SOLVE_SELECT_TEST_SLOW"):
            time.sleep(float(os.environ["WB_SOLVE_SELECT_TEST_SLOW"]))
        sol = _candidate(keyframes, camera, centers=centers, islands=[(0, n)])
        sol.input_digest = input_digest
        sol.solved_at = SHIM_CLOCK + draw
        return sol

    def live(store, world_id, session_id, solution, *, database_path, name_of, live_metric_fn=None):
        draw = int(round(solution.solved_at - SHIM_CLOCK))
        note({"guard": draw, "database": str(database_path)})
        levels = clean if draw in clean_draws else welded
        return dict(levels), {"state": "compatible", "records_matched": len(levels)}

    if monkeypatch is not None:
        monkeypatch.setattr(GS, "_map_candidate", map_candidate)
        monkeypatch.setattr(FSG, "_live_cache_evidence", live)
    else:
        GS._map_candidate = map_candidate
        FSG._live_cache_evidence = live


_SHIM_WORKER = textwrap.dedent("""
    import json, os, sys, types
    sys.modules["pycolmap"] = types.SimpleNamespace(logging=types.SimpleNamespace())
    spec = json.loads(os.environ["{env}"])
    from tests.test_world_builder_solve_select import install_shim
    install_shim(spec["n"], spec["weld"], spec["clean"], spec["log"])
    from tower.world_builder import solve_select
    raise SystemExit(solve_select.main(sys.argv[1:]))
""").format(env=SHIM_ENV)


@pytest.fixture
def e2e(tmp_path, colmap, monkeypatch):  # noqa: F811 -- fixture
    from tests.test_world_builder_final_scale_guard_seam import _walk
    from tower.process_ownership import interpreter_command

    for name in ("TOWER_WORLD_SOLVE_GATE", "TOWER_WORLD_SOLVE_MASKS", "TOWER_WORLD_SOLVE_SEED",
                 "TOWER_WORLD_SOLVE_CONSENSUS", "TOWER_WORLD_ANCHOR_VERIFY", "TOWER_WORLD_POSE_QUARANTINE",
                 "TOWER_WORLD_FINISH_STAGES", "TOWER_WORLD_LIVE_DEPTH", SELECT, BUDGET):
        monkeypatch.delenv(name, raising=False)
    from scripts import world_refinish as wr

    monkeypatch.setattr(wr, "_restart_attempts", lambda *a, **k: {})
    log = tmp_path / "shim.jsonl"

    def configure(n, weld, clean):
        install_shim(n, weld, clean, str(log), monkeypatch)
        monkeypatch.setenv(SHIM_ENV, json.dumps({"n": n, "weld": list(weld), "clean": list(clean),
                                                 "log": str(log)}))
        monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]) + os.pathsep
                           + os.environ.get("PYTHONPATH", ""))
        monkeypatch.setattr(SS, "worker_argv",
                            lambda job, draw: interpreter_command("-c", _SHIM_WORKER, str(job), str(draw)))
        return _walk(tmp_path / "w", n, colmap, monkeypatch)

    def rows():
        return [json.loads(line) for part in sorted(tmp_path.glob(log.name + ".*"))
                for line in part.read_text(encoding="utf-8").splitlines()]

    return SimpleNamespace(configure=configure, rows=rows)


def _solve(s):
    from tower.world_builder import global_solve as GS

    return GS.solve(s.store, s.world_id, s.session_id, final=True, masks=False, gate=False, seed=None,
                    loop_detection=True, input_digest="walk-digest")


def test_end_to_end_the_winner_is_published_through_the_existing_step(e2e, monkeypatch):
    from tests.test_world_builder_final_scale_guard_seam import _published
    from tower.world_builder.components import read_components_record

    s = e2e.configure(N_E2E, WELD, [CLEAN_DRAW])
    monkeypatch.setenv(SELECT, "adaptive:2")
    monkeypatch.setenv(BUDGET, "120")
    walk_db = s.workspace.database_path.read_bytes()
    out = _solve(s)
    assert out["solved"] is True
    record = out["solve"]["select"]
    assert record["winner"] == CLEAN_DRAW and record["budget_hit"] is False and record["n"] == 2
    by_draw = {r["draw"]: r for r in record["draws"]}
    assert record["trigger"] == {"rule": "fraction", "base_kf": N_E2E, "threshold_kf": 60}
    assert by_draw[1]["isolated_kf"] >= record["trigger"]["threshold_kf"]
    assert by_draw[2]["isolated_kf"] == 0 and by_draw[3]["isolated_kf"] == by_draw[1]["isolated_kf"]
    assert all(r["outcome"] == "published" for r in record["draws"])
    meta, comp, audit = _published(s)
    # ONE solution, the winner's: its poses, its model, the guard's own publication of it
    assert meta["solved_at"] == SHIM_CLOCK + CLEAN_DRAW
    assert meta["solve"]["select"] == record and audit["select"] == record
    assert meta["solve"]["final_scale_guard"]["decision"] == "published"
    assert meta["solve"]["final_scale_guard"]["keyframes_isolated"] == 0
    assert audit["solve_identity"] == meta["solve_identity"]
    assert read_components_record(s.store, s.world_id, s.session_id) is not None
    assert {k for k, p in meta["poses"].items() if p["component"] == 0} == set(meta["keyframe_ids"])
    assert (s.workspace.sparse_dir / "draw.txt").read_text() == str(CLEAN_DRAW)
    assert not (s.workspace.root / SS.SELECT_DIRNAME).exists()
    # draw 1 mapped and was guarded on the walk database; every extra draw, on its own copy, in its own
    # process -- and nothing guarded twice
    rows = e2e.rows()
    maps = {r["map"]: r for r in rows if "map" in r}
    guards = [r["guard"] for r in rows if "guard" in r]
    assert sorted(guards) == [1, 2, 3]
    assert Path(maps[1]["database"]) == s.workspace.database_path and maps[1]["pid"] == os.getpid()
    for k in (2, 3):
        assert Path(maps[k]["database"]) == s.workspace.root / SS.SELECT_DIRNAME / f"d{k}" / "database.db"
        assert maps[k]["pid"] != os.getpid() and maps[k]["threads"] == maps[1]["threads"]
        assert maps[k]["seed"] is None
    assert s.workspace.database_path.read_bytes() == walk_db


def test_end_to_end_below_the_trigger_nothing_is_launched(e2e, monkeypatch):
    from tests.test_world_builder_final_scale_guard_seam import _published

    s = e2e.configure(N_E2E, (320, N_E2E), [])                 # 40 keyframes welded: < 60
    monkeypatch.setenv(SELECT, "adaptive:3")
    launched = []
    monkeypatch.setattr(SS, "_start", lambda *a: launched.append(a))
    out = _solve(s)
    meta, _, audit = _published(s)
    assert out["solved"] is True and launched == []
    assert "select" not in meta["solve"] and "select" not in audit
    assert 0 < meta["solve"]["final_scale_guard"]["keyframes_isolated"] < config.WORLD_SOLVE_SELECT_TRIGGER_MIN_KF
    assert sorted(r["guard"] for r in e2e.rows() if "guard" in r) == [1]          # guarded once


def test_end_to_end_off_is_one_draw_guarded_once(e2e, monkeypatch):
    s = e2e.configure(N_E2E, WELD, [CLEAN_DRAW])
    monkeypatch.setenv(SELECT, "off")
    launched = []
    monkeypatch.setattr(SS, "_start", lambda *a: launched.append(a))
    out = _solve(s)
    assert out["solved"] is True and launched == [] and "select" not in out["solve"]
    assert sorted(r["guard"] for r in e2e.rows() if "guard" in r) == [1]          # guarded once, in the publish step


# ---------------------------------------------------------------------------
# the Job Object: real processes
# ---------------------------------------------------------------------------

_TREE = textwrap.dedent("""
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(sys.argv[1], "w") as handle:
        handle.write("%d %d" % (__import__("os").getpid(), child.pid))
    time.sleep(120)
""")


def _wait_pids(path: Path, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip().count(" ") == 1:
            return [int(v) for v in path.read_text().split()]
        time.sleep(0.05)
    raise AssertionError("the worker never wrote its pids")


def _gone(pids, timeout=15.0) -> bool:
    import psutil

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = []
        for pid in pids:
            try:
                alive.append(psutil.Process(pid).status() != psutil.STATUS_ZOMBIE)
            except psutil.NoSuchProcess:
                alive.append(False)
        if not any(alive):
            return True
        time.sleep(0.1)
    return False


def _reap(pids):
    import psutil

    for pid in pids:
        try:
            psutil.Process(pid).kill()
        except Exception:  # noqa: BLE001
            pass


def test_the_budget_kills_a_real_worker_and_its_child(scene, monkeypatch, tmp_path):
    from tower.process_ownership import interpreter_command

    pid_file = tmp_path / "pids.txt"
    monkeypatch.setattr(SS, "worker_argv", lambda job, draw: interpreter_command("-c", _TREE, str(pid_file)))
    pids = []
    real_start = SS._start

    def start(draw, d, job):
        handle = real_start(draw, d, job)
        pids.extend(_wait_pids(pid_file))
        return handle

    monkeypatch.setattr(SS, "_start", start)
    try:
        started = time.perf_counter()
        chosen, record = run(scene, n=1, budget_s=1.0)
        assert time.perf_counter() - started < 1.0 + 3 * SS.KILL_WAIT_S + 30
        assert chosen is scene.first and record["budget_hit"] is True
        assert record["draws"][1]["outcome"] == "budget"
        assert len(pids) == 2 and _gone(pids), "a draw's process tree outlived the budget"
    finally:
        _reap(pids)


@pytest.mark.skipif(os.name != "nt", reason="the Job Object is Windows'")
def test_a_parent_that_dies_takes_its_draws_with_it(tmp_path):
    pid_file = tmp_path / "pids.txt"
    parent = textwrap.dedent("""
        import os, sys
        from pathlib import Path
        from tower.process_ownership import interpreter_command
        from tower.world_builder import solve_select as SS
        SS.worker_argv = lambda job, draw: interpreter_command("-c", sys.argv[1], sys.argv[2])
        handle = SS._start(2, Path(sys.argv[3]) / "d2", Path(sys.argv[3]) / "job.pkl")
        assert handle.job is not None, "no job object"
        import time
        deadline = time.monotonic() + 30
        while not Path(sys.argv[2]).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        os._exit(0)          # no cleanup at all: TerminateProcess-like
    """)
    from tower.process_ownership import interpreter_command, interpreter_environment

    env = interpreter_environment()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    pids = []
    try:
        done = subprocess.run(interpreter_command("-c", parent, _TREE, str(pid_file), str(tmp_path / "select")),
                              cwd=str(Path(__file__).resolve().parents[1]), env=env, timeout=90,
                              stdin=subprocess.DEVNULL, capture_output=True)
        assert done.returncode == 0, done.stderr.decode(errors="replace")[-2000:]
        pids = _wait_pids(pid_file, timeout=5)
        assert _gone(pids), "a draw outlived the solve that launched it"
    finally:
        _reap(pids)


# ---------------------------------------------------------------------------
# THE BUILDER'S CHILD PROTOCOL: the real `scripts/world_solve.py --final`, spawned and parsed by the
# builder's own `BackgroundSolver.run_final` (SA1, 2026-10-12: selection worked and the builder
# recorded `final_solve: failed`, "unreadable summary"). Every process in it is real; a shim stands in
# for pycolmap, the mapper and the live depth evidence inside the solve child and its draws.
# ---------------------------------------------------------------------------

_SHIM_SOLVE_CHILD = textwrap.dedent("""
    import json, os, runpy, sys
    if os.environ.get("WB_SOLVE_SELECT_TEST_GARBAGE"):
        os.write(1, os.environ["WB_SOLVE_SELECT_TEST_GARBAGE"].encode() + os.linesep.encode())   # a polluted stdout
    tower_root = os.environ["WB_SOLVE_SELECT_TEST_ROOT"]
    sys.path.insert(0, tower_root)
    from tests.test_world_builder_solve_masks import FakeColmap
    sys.modules["pycolmap"] = FakeColmap()
    spec = json.loads(os.environ["{env}"])
    from tests import test_world_builder_solve_select as T
    T.install_shim(spec["n"], spec["weld"], spec["clean"], spec["log"])
    from tower.process_ownership import interpreter_command
    from tower.world_builder import relocalizer, solve_select
    relocalizer.revisit_pairs = lambda session_dir: []
    solve_select.worker_argv = lambda job, draw: interpreter_command("-c", T._SHIM_WORKER, str(job), str(draw))
    script = os.path.join(tower_root, "scripts", "world_solve.py")
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
""").format(env=SHIM_ENV)


@pytest.fixture
def builder(e2e, tmp_path, monkeypatch):
    from scripts import world_build_session as wbs

    tower_root = Path(__file__).resolve().parents[1]
    wrapper = tmp_path / "world_solve_shim.py"
    wrapper.write_text(_SHIM_SOLVE_CHILD, encoding="utf-8")
    monkeypatch.setenv("WB_SOLVE_SELECT_TEST_ROOT", str(tower_root))

    def run_final(s, **kw):
        solver = wbs.BackgroundSolver(root=s.store.root, world_id=s.world_id, session_id=s.session_id,
                                      every=10_000, capture_dirs=[], script=wrapper)
        try:
            return solver.run_final(s.store, {}, **kw)
        finally:
            solver.close()

    return SimpleNamespace(e2e=e2e, run_final=run_final)


def test_the_builders_final_solve_child_reports_a_selected_solve(builder, monkeypatch):
    from tests.test_world_builder_final_scale_guard_seam import _published

    s = builder.e2e.configure(N_E2E, WELD, [CLEAN_DRAW])
    monkeypatch.setenv(SELECT, "adaptive:2")
    summary = builder.run_final(s)
    assert summary.get("solved") is True, summary.get("error") or summary
    assert "error" not in summary
    assert summary["solve"]["select"]["winner"] == CLEAN_DRAW
    assert summary["keyframes_posed"] == N_E2E and summary["solver"] == "glomap"
    meta, _, _ = _published(s)
    assert meta["solved_at"] == SHIM_CLOCK + CLEAN_DRAW and meta["solve"]["select"] == summary["solve"]["select"]
    draws = {r["map"] for r in builder.e2e.rows() if "map" in r}
    assert {2, 3} <= draws


def test_a_hard_stop_during_selection_through_the_builder_leaves_no_draw_running(builder, monkeypatch):
    s = builder.e2e.configure(N_E2E, WELD, [CLEAN_DRAW])
    monkeypatch.setenv(SELECT, "adaptive:3")
    monkeypatch.setenv("WB_SOLVE_SELECT_TEST_SLOW", "120")           # every extra draw hangs
    before = s.workspace.solution_path.read_bytes()
    select_dir = s.workspace.root / SS.SELECT_DIRNAME

    def stop():
        pids = [r["pid"] for r in builder.e2e.rows() if r.get("map", 1) != 1]
        return len(pids) >= 3

    started = time.perf_counter()
    summary = builder.run_final(s, should_stop=stop)
    assert summary.get("interrupted") is True and summary.get("solved") is False
    assert time.perf_counter() - started < 120
    pids = [r["pid"] for r in builder.e2e.rows() if r.get("map", 1) != 1]
    try:
        assert len(pids) == 3 and _gone(pids), "a draw outlived the stopped final solve"
    finally:
        _reap(pids)
    # nothing was published by the stopped final solve: the walk's own solution stands
    assert s.workspace.solution_path.read_bytes() == before


def test_an_unusable_summary_of_a_selected_solve_is_read_back_from_its_publication(builder, monkeypatch):
    """SA1: the selected solve was published and the builder recorded `final_solve: failed`
    ("unreadable summary"). The builder now reads the finished selected solve back from what it
    published, and keeps the raw stdout for the record."""
    from scripts import world_build_session as wbs

    s = builder.e2e.configure(N_E2E, WELD, [CLEAN_DRAW])
    monkeypatch.setenv(SELECT, "adaptive:2")
    monkeypatch.setenv("WB_SOLVE_SELECT_TEST_GARBAGE", "not json")
    summary = builder.run_final(s)
    assert summary.get("solved") is True and summary.get("summary_from_publication") is True
    assert "error" not in summary and "interrupted" not in summary
    assert summary["solve"]["select"]["winner"] == CLEAN_DRAW
    assert summary["solved_at"] == SHIM_CLOCK + CLEAN_DRAW
    assert summary["keyframes_posed"] == N_E2E and summary["solver"] == "glomap"
    kept = s.workspace.root / wbs.FINAL_SOLVE_STDOUT_FILENAME
    assert kept.read_bytes().startswith(b"not json")


def test_an_unusable_summary_without_selection_still_fails_as_before(builder, monkeypatch):
    s = builder.e2e.configure(N_E2E, WELD, [CLEAN_DRAW])
    monkeypatch.setenv(SELECT, "off")
    monkeypatch.setenv("WB_SOLVE_SELECT_TEST_GARBAGE", "not json")
    summary = builder.run_final(s)
    assert summary.get("solved") is False and summary["error"].startswith("unreadable summary: ")
    assert "summary_from_publication" not in summary


def test_a_selected_publication_from_before_this_launch_is_not_this_finalizations(builder, monkeypatch):
    from scripts import world_build_session as wbs

    s = builder.e2e.configure(N_E2E, WELD, [CLEAN_DRAW])
    monkeypatch.setenv(SELECT, "adaptive:2")
    assert builder.run_final(s)["solved"] is True
    solved_at = json.loads(s.workspace.solution_path.read_text(encoding="utf-8"))["solved_at"]
    assert wbs.published_selected_solve(s.store, s.world_id, s.session_id, since=solved_at) is not None
    assert wbs.published_selected_solve(s.store, s.world_id, s.session_id, since=solved_at + 1) is None
