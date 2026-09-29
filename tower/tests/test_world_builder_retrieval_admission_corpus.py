"""The retrieval admission reproduces the frozen C23-CERT study's decisions on its saved inputs (RUN
experiments/C23-CERT: RESULTS.md, `out\\cases\\*.json`; BRIEF-IMPL section 5, "Fixtures").

OPTIONAL: set TOWER_C23_RUN_DIR to the run folder (the lead's copy is
`C:\\Users\\<you>\\Projects\\Glasses-scratch\\wb-coherence-run-2026-09-23`). The study's inputs are hundreds of MB and
stay there; they are read in place, in a subprocess (`tests/wb_retrieval_admission_corpus.py`), and never copied.
Unset: skipped. The synthetic tests in `test_world_builder_retrieval_admission.py` always run.

Pinned, per case and in BOTH P4 settings, over all 44 solves: the product's candidates, every certificate statistic
and verdict, the admitted groups, the projection's state and published room, the null projection, and the product
gate equal to the study's patched gate on every gate call. By name: W4 group 249 admits 27/27 with no closet or
bathroom camera; W5's Area 1 stays out; the control never has a candidate; c10's bed is never admitted. And, fresh on
W4: the product's own retrieval reproduces the study's links and certificate, serially and on 8 threads alike, and
in two processes with different BLAS thread counts alike (Codex C23x MED-3).

C3 (C23-IMPL-G; Codex C23x HIGH-1). The study's `c23c_lib.certificate` classified each consensus set by a mean
RECOMPUTED over it; PREREG A.4, as frozen, classifies it by its SEED mean. The product implements the frozen text.
A corrected copy of the study's library replayed all 44 solves x 2 settings (`wbcpt\\c23g-replay`): no admission and no
C1-C3 verdict changed; only the reported angles G_A / G_B moved on 4 solves. So the frozen outputs pin every
decision and every statistic but those two angles, and the product's C3 statistics (m(A*), |A*|, G_A, m(B*), G_B,
C3) are pinned against `prereg_c3`, a brute-force reading of the frozen text computed in the driver.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOWER = Path(__file__).resolve().parents[1]
RUN_ENV = "TOWER_C23_RUN_DIR"
# Every certificate statistic the frozen study pins: all but the two reported angles, which the study computed from a
# recomputed mean (see the docstring); those, and the whole of C3, are pinned against the driver's `prereg_c3`.
CERT_KEYS = ("links", "db", "new", "honoured", "noncompact", "M_H", "H", "C", "mA", "nA", "mB", "C1", "C2", "C3")
C3_KEYS = ("mA", "nA", "G_A", "mB", "G_B", "C3")


def _run_dir() -> Path:
    value = os.environ.get(RUN_ENV)
    if not value:
        pytest.skip(f"{RUN_ENV} is not set")
    run = Path(value)
    if not (run / "experiments" / "C23-CERT" / "out" / "cases").is_dir():
        pytest.skip(f"{RUN_ENV} has no experiments/C23-CERT/out/cases")
    return run


def _frozen(run: Path, tag: str) -> dict:
    return json.loads((run / "experiments" / "C23-CERT" / "out" / "cases" / f"{tag.replace(':', '_')}.json")
                      .read_text(encoding="utf-8"))


def _all_tags(run: Path) -> list:
    return sorted(p.stem.replace("_", ":", 2) for p in (run / "experiments" / "C23-CERT" / "out" / "cases").glob(
        "*.json"))


def _driver(mode: str, run: Path, out: Path, tags: list, **extra_env) -> subprocess.Popen:
    # CPU only: -1, never an empty value (Windows drops an empty variable and the GPU stays visible; manager 146 s6)
    env = dict(os.environ, PYTHONPATH=str(TOWER), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", **extra_env)
    return subprocess.Popen([sys.executable, "-m", "tests.wb_retrieval_admission_corpus", mode, str(run), str(out),
                             *tags], cwd=str(TOWER), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    run = _run_dir()
    tags = _all_tags(run)
    tmp = tmp_path_factory.mktemp("c23replay")
    chunks = [tags[i::4] for i in range(4)]                                  # 4 processes (the run's cap is 8)
    procs = []
    for k, chunk in enumerate(chunks):
        out = tmp / f"replay{k}.json"
        procs.append((out, _driver("replay", run, out, chunk)))
    res = {}
    for out, p in procs:
        log, _ = p.communicate(timeout=1800)
        assert p.returncode == 0, log.decode("utf-8", "replace")[-4000:]
        res.update(json.loads(out.read_text(encoding="utf-8")))
    assert len(res) == 44, sorted(res)
    return run, res


def test_every_case_reproduces_the_frozen_decisions(replayed):
    run, res = replayed
    bad = []
    for tag, got in sorted(res.items()):
        want = _frozen(run, tag)
        if not got["checks"]["product_gate_admit_none_equals_study"] or not got[
                "product_gate_equals_study_on_every_call"]:
            bad.append((tag, "gate"))
        for setting in ("P4off", "P4on"):
            g, w = got["settings"][setting], want["settings"][setting]
            if not (g["candidates_equal"] and g["diag_equal"] and g.get("allow_rebuilt_equals_pre_seal_room", True)):
                bad.append((tag, setting, "candidates"))
            if g["db_geometry_vs_saved"]["differ"]:
                bad.append((tag, setting, "db geometry", g["db_geometry_vs_saved"]))
            if [(c["first"], c["n"], c["scale_factor"]) for c in g["candidates"]] != \
                    [(c["first"], c["n"], c["scale_factor"]) for c in w["candidates"]]:
                bad.append((tag, setting, "candidate list"))
                continue
            for gc, wc in zip(g["candidates"], w["candidates"]):
                keys = [k for k in CERT_KEYS if k in wc["cert"]]
                if {k: gc["cert"][k] for k in keys} != {k: wc["cert"][k] for k in keys} or \
                        gc["admitted"] != wc["admitted"]:
                    bad.append((tag, setting, gc["first"], {k: (gc["cert"][k], wc["cert"][k]) for k in keys
                                                           if gc["cert"][k] != wc["cert"][k]}))
                # C3 exactly as frozen (the seed mean), against the driver's independent reading of the text
                if {k: gc["cert"][k] for k in C3_KEYS} != gc["prereg_c3"]:
                    bad.append((tag, setting, gc["first"], "C3 vs PREREG", gc["cert"], gc["prereg_c3"]))
            if g["admitted"] != w["admitted"]:
                bad.append((tag, setting, "admitted", g["admitted"], w["admitted"]))
            if g["state"] != w["projection"]["state"]:
                bad.append((tag, setting, "state", g["state"], w["projection"]["state"]))
            if (g["room_base"], g["room_published"]) != (w["room_base"], w["room_published"]):
                bad.append((tag, setting, "room", g["room_base"], g["room_published"], w["room_base"],
                            w["room_published"]))
            if g["null_projection_equals_base"] is not True or w["check_null_projection_equals_base"] is not True:
                bad.append((tag, setting, "null projection"))
            for k in ("w4_g249_in_room", "w5_area1_in_room", "bed_base", "bed_pub", "bed_admitted", "gained_regions"):
                if k in w and g.get(k) != w[k]:
                    bad.append((tag, setting, k, g.get(k), w[k]))
    assert bad == []


def test_the_named_cases(replayed):
    _run, res = replayed
    for setting in ("P4off", "P4on"):
        w4 = res["walk:c81766a3"]["settings"][setting]
        assert w4["state"] == "applied" and w4["admitted"] == [{"first": "00001715.jpg", "n": 27}]
        assert (w4["w4_g249_in_room"], w4["w4_g249_n"]) == (27, 27)
        assert (w4["room_base"], w4["room_published"]) == (354, 381)
        assert "closet" not in w4["gained_regions"] and "bathroom" not in w4["gained_regions"]
        w5 = res["walk:da4ac2d3"]["settings"][setting]
        assert w5["w5_area1_in_room"] == 0 and w5["admitted"] == [] and w5["lost"] == 0
        assert all(c["first"] != "00000798.jpg" for c in w5["candidates"])              # never a candidate
        for tag in ("acc2:b2a75ab4:c10", "acc2:b2a75ab4:c20", "acc2:b2a75ab4:r0a", "acc2:b2a75ab4:r0b",
                    "acc2:b2a75ab4:w20", "gt:A0:b2a75ab4", "gt:A0h:b2a75ab4", "gt:A1:b2a75ab4", "gt:A1h:b2a75ab4"):
            c = res[tag]["settings"][setting]
            assert c["candidates"] == [] and c["room_published"] == c["room_base"] and c["lost"] == 0, tag
        c10 = res["acc2:6839fb8f:c10"]["settings"][setting]
        assert c10["bed_admitted"] == 0 and c10["bed_pub"] <= c10["bed_base"] and c10["admitted"] == []
        w3 = res["walk:4f5d0b15"]["settings"][setting]
        assert w3["admitted"] == []
    admitted = sorted((tag, s, a["first"], a["n"]) for tag, r in res.items() for s, S in r["settings"].items()
                      for a in S["admitted"])
    assert admitted == sorted([(t, s, f, n) for t, f, n in (("walk:c81766a3", "00001715.jpg", 27),
                                                            ("gt:A1:6839fb8f", "000620_k.jpg", 21),
                                                            ("acc2:6839fb8f:w20", "00000156.jpg", 7))
                               for s in ("P4off", "P4on")])


def test_fresh_retrieval_on_w4_reproduces_the_study_and_is_schedule_free(tmp_path):
    run = _run_dir()
    out = tmp_path / "fresh.json"
    p = _driver("fresh", run, out, ["walk:c81766a3"])
    log, _ = p.communicate(timeout=1800)
    assert p.returncode == 0, log.decode("utf-8", "replace")[-4000:]
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["same_features_as_study"] and res["same_bow_as_study"]
    assert res["candidates"] == [["00001715.jpg", 27], ["00001887.jpg", 7]]
    assert res["serial_equals_8_threads"]
    # C23-CERT's serial cost run: 499 new pairs, 71 verified, all in the saved pool (RESULTS section 5)
    assert res["pairs_queried"] == 499 and res["verified_serial"] == 71
    assert res["pool_agreement"] == {"saved": 71, "fresh": 71, "both": 71}
    g249 = res["certificates"]["00001715.jpg"]
    assert g249["admitted"] and (g249["links"], g249["M_H"], g249["H"], g249["C"], g249["mA"], g249["mB"]) == \
        (76, 11, 8.406, 2.0214, 11, 2)
    assert not res["certificates"]["00001887.jpg"]["admitted"] and res["admitted_cameras"] == 27


def test_the_links_and_verdicts_do_not_depend_on_the_blas_thread_count(tmp_path):
    """Codex C23x MED-3: two processes whose OpenBLAS starts with 1 and with 8 threads (numpy imported before anything
    can pin it) give the same bag of words, query pairs, verified links, database-link geometry and certificates on
    W4 -- the study's 71 links and g249's admission -- and each process's own thread count is restored after the bag
    of words."""
    run = _run_dir()
    res = {}
    for n in ("1", "8"):
        out = tmp_path / f"threads{n}.json"
        p = _driver("threads", run, out, ["walk:c81766a3"], OPENBLAS_NUM_THREADS=n, OMP_NUM_THREADS=n)
        log, _ = p.communicate(timeout=1800)
        assert p.returncode == 0, log.decode("utf-8", "replace")[-4000:]
        res[n] = json.loads(out.read_text(encoding="utf-8"))
    one, eight = res["1"], res["8"]
    assert (one["openblas_threads"], eight["openblas_threads"]) == (1, 8)          # really two configurations
    assert (one["openblas_threads_after_bow"], eight["openblas_threads_after_bow"]) == (1, 8)   # restored
    for key in ("V_sha1", "pairs", "links", "db_noncompact", "certificates"):
        assert one[key] == eight[key], key
    assert len(one["pairs"]) == 499 and len(one["links"]) == 71
    g249 = one["certificates"]["00001715.jpg"]
    assert g249["admitted"] and (g249["links"], g249["M_H"], g249["mA"], g249["mB"]) == (76, 11, 11, 2)
    assert not one["certificates"]["00001887.jpg"]["admitted"]
