"""The retrieval admission against the frozen C23-CERT study, on its saved inputs. NOT a test module: the optional
tests in `test_world_builder_retrieval_admission_corpus.py` run it in a subprocess, because the study's loaders pin
BLAS threads at import and load the reference gate by file path.

    python -m tests.wb_retrieval_admission_corpus replay  <RUN_DIR> <out.json> TAG [TAG ...]
    python -m tests.wb_retrieval_admission_corpus fresh   <RUN_DIR> <out.json> walk:c81766a3
    python -m tests.wb_retrieval_admission_corpus threads <RUN_DIR> <out.json> walk:c81766a3

`replay`: per case and per P4 setting, BASE is built by the study's own machinery (`c23c_lib`: the case's gate, then
P4 with its seals); then the PRODUCT's functions decide -- `retrieval_admission.candidates`, `direct_links` and
`certificate` on the study's saved link pool (`P5-RETR` / `C23-CERT out\\retr`), the PRODUCT gate's projection re-gate
(`coherence_gate.apply_gate(admit=...)`) and `post_checks` -- and the P4 re-run goes through the product gate too.
Every gate call is also made with the study's in-memory patched gate and compared. The output is compared with
`C23-CERT\\out\\cases\\*.json`. Each candidate also carries `prereg_c3`: (C3) recomputed here from PREREG A.4's frozen
text alone (`_prereg_c3`), because the study's own `c23c_lib.certificate` classified a RECOMPUTED mean (Codex C23x
HIGH-1; its corrected replay, C23-IMPL-G, changed no decision): the product's C3 statistics are checked against this.

`fresh`: the product's own retrieval on W4's frozen database and masks (features, bag of words, queries, verification
serially and on 8 threads), then the certificates.

`threads`: numpy is imported FIRST, so the process's OpenBLAS runs with ITS OWN environment's thread count (the
study's loader cannot pin it); then the product's candidate path end to end -- features, the bag of words, the
queries, the verification, the database links' geometry, the certificates -- on W4. Two processes with different
`OPENBLAS_NUM_THREADS` must give the same bag of words, pairs, links and verdicts (Codex C23x MED-3).

READ-ONLY on RUN: nothing is written but `<out.json>`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path


def _setup(run_dir: Path):
    """The study's machinery and the PRODUCT's modules side by side. The study's case loaders import the coherence
    LANE's own `tower` package (`coherence_eval`, P4-IV p2_acc2.py) on first use, so the product's modules are imported
    first and then taken out of `sys.modules`: they keep their own references, and the lane's package loads as it did
    in C23-CERT."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(run_dir / "experiments" / "C23-CERT" / "scripts"))
    import c23c_lib as X  # noqa: PLC0415  (FIRST: pins BLAS threads, loads the study's machinery)

    assert not any(k == "tower" or k.startswith("tower.") for k in sys.modules), "a tower package is loaded already"
    from tower.world_builder import coherence_gate as PCG  # noqa: PLC0415
    from tower.world_builder import retrieval_admission as RA  # noqa: PLC0415

    product = Path(RA.__file__).resolve().parents[2]
    for k in [k for k in sys.modules if k == "tower" or k.startswith("tower.")]:
        del sys.modules[k]
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != product]
    return X, PCG, RA


def _prereg_c3(X, L, tau: float = 16.8) -> dict:
    """PREREG A.4 (C3), written from the frozen text alone (independent of the product's `certificate` and of the
    study's): over N = the noncompact links of L with a readout, for each l: S_l = {k : angle(D_k^T D_l) <= tau};
    G_l = the chordal mean of D over S_l; S'_l = {k : angle(D_k^T G_l) <= tau}; A* = the S'_l with angle(G_l) <= tau
    maximising m (ties: larger |S|, smaller angle(G), lower l); B* = the S'_l with angle(G_l) > tau maximising m
    (0 if none; its reported angle is the first l's); pass iff m(A*) >= 2 and m(A*) > m(B*). Brute force."""
    import numpy as np  # noqa: PLC0415

    RL = X.RL
    N = [x for x in L if x["noncompact"] and x["D"] is not None]
    best_a, best_b = None, None
    for l_, x in enumerate(N):
        S = [k for k, y in enumerate(N) if RL.rot_deg(np.asarray(y["D"]).T @ np.asarray(x["D"])) <= tau]
        G = RL.chordal_mean(np.asarray([N[k]["D"] for k in S], np.float64))
        S2 = [k for k, y in enumerate(N) if RL.rot_deg(np.asarray(y["D"]).T @ G) <= tau]
        if not S2:
            continue
        m = X.matching([(N[k]["g"], N[k]["r"]) for k in S2])
        ang = RL.rot_deg(G)
        if ang <= tau:
            key = (m, len(S2), -ang, -l_)
            if best_a is None or key > best_a[0]:
                best_a = (key, m, len(S2), ang)
        elif best_b is None or m > best_b[0]:
            best_b = (m, ang)
    m_a = best_a[1] if best_a else 0
    m_b = best_b[0] if best_b else 0
    return {"mA": m_a, "nA": best_a[2] if best_a else 0, "G_A": round(best_a[3], 2) if best_a else None,
            "mB": m_b, "G_B": round(best_b[1], 2) if best_b else None, "C3": bool(m_a >= 2 and m_a > m_b)}


def _gate_equal(a, b):
    return (a["labels"] == b["labels"] and a["components"] == b["components"]
            and json.dumps(a["rounds"], sort_keys=True, default=str) == json.dumps(b["rounds"], sort_keys=True,
                                                                                    default=str))


def _same_published(X, cs, a, b):
    ra, rb = X.room_of(a["labels"], cs), X.room_of(b["labels"], cs)
    pa, pb = X.pieces_of(a["labels"], cs), X.pieces_of(b["labels"], cs)
    if ra != rb or set(pa.values()) != set(pb.values()):
        return False
    rra, rrb = X.reasons_by_label(a), X.reasons_by_label(b)
    la = {v: k for k, v in pa.items()}
    return all(rra[la[v]] == rrb[k] for k, v in pb.items())


def replay(run_dir: Path, tags: list) -> dict:
    X, PCG, RA = _setup(run_dir)
    import numpy as np  # noqa: PLC0415

    RL = X.RL
    P, GP = RA.AdmissionParams(), PCG.GateParams()
    out = {}
    for tag in tags:
        t0 = time.time()
        cs = RL.load_case(tag)
        if cs is None:
            out[tag] = {"skipped": "not re-gatable"}
            continue
        ctx = X.P4Ctx(cs, tag)
        new, dba = X.load_pool(cs, tag)
        names = list(cs.model.names)
        idx = cs.model.index()
        sup = X.sup_mask(cs)
        hooks0 = X.base_hooks(cs)
        withheld = set(hooks0.get("withhold") or [])
        cam = RL.camera_of(cs)
        area = float(cam["width"]) * float(cam["height"])
        checks = {}
        gate_calls = {"n": 0, "equal": True}

        def prod_gate(**hooks):
            """The PRODUCT gate, and the study's patched gate on the same call: they must agree."""
            gate_calls["n"] += 1
            p = PCG.apply_gate(cs.model, cs.links, cs.metric_log, link_rotations=cs.rots,
                               masks_applied=cs.masks_applied, **hooks)
            ref = X.gate_db(cs, **hooks)
            if not _gate_equal(p, ref):
                gate_calls["equal"] = False
            return p

        base_off = prod_gate(**hooks0)
        checks["product_gate_admit_none_equals_study"] = _gate_equal(
            base_off, RL.CG.apply_gate(cs.model, cs.links, cs.metric_log, link_rotations=cs.rots,
                                       masks_applied=cs.masks_applied, **cs.hooks))
        settings = {}
        for setting in ("P4off", "P4on"):
            S = {}
            if setting == "P4off":
                BASE, seals_B, coll_B, allow_B = base_off, {}, set(), None
            else:
                before = sorted(X.room_of(base_off["labels"], cs))
                BASE, audit_b, seals_B = X.p4_run(ctx, base_off,
                                                  lambda seal, before=before: X.gate_db(cs, **dict(hooks0, seal=seal,
                                                                                                   room=before)), {})
                coll_B = set(audit_b.get("collateral") or [])
                allow_B = set(before) if seals_B else None
            ROOM_B = X.room_of(BASE["labels"], cs)
            if seals_B:
                # the product rebuilds P4's pre-seal room from its record: the room, the sealed, the collateral
                S["allow_rebuilt_equals_pre_seal_room"] = (ROOM_B | set(seals_B) | coll_B) == allow_B
            allow = allow_B if allow_B is not None else set(ROOM_B)
            r = np.full(cs.model.n, np.nan)
            if BASE["metric_available"]:
                for nm, v in (cs.metric_log or {}).items():
                    if nm in idx and v is not None and np.isfinite(v):
                        r[idx[nm]] = float(v)
            pc, pdiag = RA.candidates(cs.model, BASE, r, ROOM_B, sealed=set(seals_B), collateral=coll_B,
                                      withheld=withheld, params=P, gp=GP)
            rc, rdiag = X.candidates(cs, BASE, ROOM_B, set(seals_B), coll_B, withheld)
            S["candidates_equal"] = ([(c["first_camera"], c["n"], c["scale_factor"]) for c in pc]
                                     == [(c["first"], c["n"], c["scale_factor"]) for c in rc])
            S["diag_equal"] = ([(c["first_camera"], c["n"]) for c in pdiag] == [(c["first"], c["n"]) for c in rdiag])
            # the database links' geometry: the study's saved table, else the PRODUCT's own reader and re-fit
            cams = {n for c in pc for n in c["members"]}
            keys = [k for k in cs.links if (k[0] in cams and k[1] in ROOM_B) or (k[1] in cams and k[0] in ROOM_B)]
            db_attrs = {k: (dba.get(k) or dba.get((k[1], k[0]))) for k in keys
                        if (dba.get(k) or dba.get((k[1], k[0]))) is not None}
            missing = [k for k in keys if k not in db_attrs]
            if missing:
                db_attrs.update(RA.db_geometry(RL.db_of(cs), missing, cs.K, area, P))
            S["db_geometry_fresh"] = len(missing)
            # the product's own reader and re-fit against the study's saved table, link by link
            mine = RA.db_geometry(RL.db_of(cs), keys, cs.K, area, P)
            agree, differ = 0, []
            for k in keys:
                saved = dba.get(k) or dba.get((k[1], k[0]))
                got = mine.get(k) or mine.get((k[1], k[0]))
                if saved is None:
                    continue
                if got is not None and bool(got["noncompact"]) == bool(saved["noncompact"]):
                    agree += 1
                else:
                    differ.append([k[0], k[1], None if got is None else bool(got["noncompact"]),
                                   bool(saved["noncompact"])])
            S["db_geometry_vs_saved"] = {"agree": agree, "differ": differ}
            S["candidates"] = []
            admitted = []
            for c in pc:
                L = RA.direct_links(cs.model, set(c["members"]), ROOM_B, cs.links, cs.rots, new, db_attrs, P)
                ok, st = RA.certificate(L, P)
                S["candidates"].append({"first": c["first_camera"], "n": c["n"], "scale_factor": c["scale_factor"],
                                        "cert": {k: st.get(k) for k in ("links", "db", "new", "honoured",
                                                                        "noncompact", "M_H", "H", "C", "mA", "G_A",
                                                                        "nA", "mB", "G_B", "C1", "C2", "C3")},
                                        "prereg_c3": _prereg_c3(X, L, P.honoured_deg),
                                        "admitted": ok})
                if ok:
                    admitted.append(c)
            adm = {n for c in admitted for n in c["members"]}
            S["admitted"] = [{"first": c["first_camera"], "n": c["n"]} for c in admitted]
            hooks_p = dict(hooks0, rider_min_shared=RA_RIDER)
            if seals_B:
                hooks_p["seal"] = dict(seals_B)
            null = prod_gate(**dict(hooks_p, room=sorted(allow), admit=set()))
            S["null_projection_equals_base"] = _same_published(X, cs, null, BASE)
            published, state = BASE, "no-admission"
            if adm:
                proj = prod_gate(**dict(hooks_p, room=sorted(allow | adm), admit=set(adm)))
                why, _info = RA.post_checks(cs.model, BASE, proj, adm, GP)
                if why is None:
                    why = RA._lost_admitted(cs.model, BASE, adm, GP)
                state = "applied" if why is None else f"not-applied: {why}"
                if why is None:
                    published = proj
                    if setting == "P4on":
                        pr = sorted(X.room_of(published["labels"], cs))
                        published, p4c, _s = X.p4_run(
                            ctx, published, lambda seal, pr=pr: prod_gate(**dict(hooks_p, seal=seal, room=pr,
                                                                                admit=set(adm))), dict(seals_B))
                        S["p4_rerun_sealed"] = p4c.get("sealed", 0)
                        S["p4_rerun_state"] = p4c.get("state")
                        if p4c.get("state") != "applied":
                            # the product (Codex C23x HIGH-2): a P4 re-run that is not applied publishes BASE
                            published, state = BASE, f"not-applied: the P4 re-run is {p4c.get('state')}"
            S["state"] = state
            RP = X.room_of(published["labels"], cs)
            S["room_base"], S["room_published"] = len(ROOM_B), len(RP)
            gained = sorted(RP - ROOM_B)
            S["lost"] = len(ROOM_B - RP)
            if cs.kind == "walk":
                region = {n: cs.region.get(cs.kid_of_name.get(n), "?") for n in names}
                S["gained_regions"] = {k: sum(1 for n in gained if region[n] == k) for k in sorted({region[n]
                                                                                                    for n in gained})}
                g249 = next((g for g in BASE["groups"] if g["first_camera"] == X.W4_G249), None)
                a1 = next((g for g in BASE["groups"] if g["first_camera"] == X.W5_AREA1), None)
                if cs.w8 == "c81766a3" and g249 is not None:
                    mem = [n for n in g249["members"] if sup[idx[n]]]
                    S["w4_g249_in_room"], S["w4_g249_n"] = sum(1 for n in mem if n in RP), len(mem)
                if cs.w8 == "da4ac2d3" and a1 is not None:
                    mem = [n for n in a1["members"] if sup[idx[n]]]
                    S["w5_area1_in_room"], S["w5_area1_n"] = sum(1 for n in mem if n in RP), len(mem)
            if cs.w8 == "6839fb8f":
                bed = lambda s: sum(1 for n in s if n[:8].isdigit() and int(n[:8]) >= X.BED_FIRST)  # noqa: E731
                S["bed_base"], S["bed_pub"], S["bed_admitted"] = bed(ROOM_B), bed(RP), bed(adm)
            settings[setting] = S
        out[tag] = {"checks": checks, "settings": settings, "gate_calls": gate_calls["n"],
                    "product_gate_equals_study_on_every_call": gate_calls["equal"],
                    "seconds": round(time.time() - t0, 1)}
    return out


RA_RIDER = 3   # coherence_publish.RIDER_MIN_SHARED (the driver does not import the publish step)


def fresh(run_dir: Path, tag: str) -> dict:
    X, PCG, RA = _setup(run_dir)
    import numpy as np  # noqa: PLC0415

    RL = X.RL
    P, GP = RA.AdmissionParams(), PCG.GateParams()
    cs = RL.load_case(tag)
    idx = cs.model.index()
    base = X.gate_db(cs, **X.base_hooks(cs))
    ROOM_B = X.room_of(base["labels"], cs)
    r = np.full(cs.model.n, np.nan)
    for nm, v in (cs.metric_log or {}).items():
        if nm in idx and v is not None and np.isfinite(v):
            r[idx[nm]] = float(v)
    cands, _ = RA.candidates(cs.model, base, r, ROOM_B, sealed=set(), collateral=set(),
                             withheld=set(cs.hooks.get("withhold") or []), params=P, gp=GP)
    cam = RL.camera_of(cs)
    area = float(cam["width"]) * float(cam["height"])
    T = {}
    t = time.perf_counter()
    feats, tried, fstats = RA.load_masked_features(RL.db_of(cs), RL.mask_dir_of(cs))
    T["features"] = time.perf_counter() - t
    rfeats, rtried, rstats = RL.load_features(RL.db_of(cs), RL.mask_dir_of(cs))
    same_feats = (sorted(feats) == sorted(rfeats) and tried == rtried and fstats == rstats
                  and all(np.array_equal(feats[n][0], rfeats[n][0]) and np.array_equal(feats[n][1], rfeats[n][1])
                          for n in feats))
    t = time.perf_counter()
    bnames, V = RA.build_bow(feats, P)
    T["bow"] = time.perf_counter() - t
    rnames, rV = RL.build_bow(rfeats)
    same_bow = bnames == rnames and np.array_equal(V, rV)
    pairs = RA.query_pairs(cands, ROOM_B, bnames, V, tried, P)
    t = time.perf_counter()
    serial = RA.verify_pairs(pairs, feats, cs.K, area, P, workers=1)
    T["verify_serial"] = time.perf_counter() - t
    t = time.perf_counter()
    threaded = RA.verify_pairs(list(reversed(pairs)), feats, cs.K, area, P, workers=8)
    T["verify_8_threads"] = time.perf_counter() - t

    def key(links):
        return [(x["a"], x["b"], x["inl"], np.asarray(x["R"]).round(12).tolist(), x["noncompact"], round(x["amb"], 9),
                 round(x["hull_a"], 12), round(x["hull_b"], 12)) for x in links]

    new_saved, _dba = X.load_pool(cs, tag)
    cams = {n for c in cands for n in c["members"]}
    saved = {RL.sorted_pair(x["a"], x["b"]) for x in new_saved
             if (x["a"] in cams and x["b"] in ROOM_B) or (x["b"] in cams and x["a"] in ROOM_B)}
    fr = {RL.sorted_pair(x["a"], x["b"]) for x in serial}
    keys = sorted(k for k in cs.links if (k[0] in cams and k[1] in ROOM_B) or (k[1] in cams and k[0] in ROOM_B))
    db_attrs = RA.db_geometry(RL.db_of(cs), keys, cs.K, area, P)
    certs = {}
    admitted = set()
    for c in cands:
        L = RA.direct_links(cs.model, set(c["members"]), ROOM_B, cs.links, cs.rots, serial, db_attrs, P)
        ok, st = RA.certificate(L, P)
        certs[c["first_camera"]] = {"n": c["n"], "admitted": ok,
                                    **{k: st.get(k) for k in ("links", "db", "new", "M_H", "H", "C", "mA", "mB")}}
        if ok:
            admitted |= set(c["members"])
    return {"candidates": [(c["first_camera"], c["n"]) for c in cands], "pairs_queried": len(pairs),
            "same_features_as_study": bool(same_feats), "same_bow_as_study": bool(same_bow),
            "verified_serial": len(serial), "serial_equals_8_threads": key(serial) == key(threaded),
            "pool_agreement": {"saved": len(saved), "fresh": len(fr), "both": len(saved & fr)},
            "db_links_with_geometry": len(db_attrs), "certificates": certs, "admitted_cameras": len(admitted),
            "seconds": {k: round(v, 2) for k, v in T.items()}}


def threads(run_dir: Path, tag: str) -> dict:
    """The product's candidate path end to end in a process whose OpenBLAS runs with the environment's own thread
    count (`main` imports numpy before anything else). Every decision-relevant output is returned, plus the bag of
    words computed WITHOUT the pin in the same process (to show the pin is what makes it thread-free)."""
    import contextlib  # noqa: PLC0415
    import hashlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    X, PCG, RA = _setup(run_dir)
    RL = X.RL
    P, GP = RA.AdmissionParams(), PCG.GateParams()
    get, _put = RA._openblas_threads_api()
    threads_before = int(get())
    cs = RL.load_case(tag)
    idx = cs.model.index()
    base = X.gate_db(cs, **X.base_hooks(cs))
    room = X.room_of(base["labels"], cs)
    r = np.full(cs.model.n, np.nan)
    for nm, v in (cs.metric_log or {}).items():
        if nm in idx and v is not None and np.isfinite(v):
            r[idx[nm]] = float(v)
    cands, _ = RA.candidates(cs.model, base, r, room, sealed=set(), collateral=set(),
                             withheld=set(cs.hooks.get("withhold") or []), params=P, gp=GP)
    cam = RL.camera_of(cs)
    area = float(cam["width"]) * float(cam["height"])
    feats, tried, _ = RA.load_masked_features(RL.db_of(cs), RL.mask_dir_of(cs))
    names, V = RA.build_bow(feats, P)
    threads_after = int(get())
    pairs = RA.query_pairs(cands, room, names, V, tried, P)
    new = RA.verify_pairs(pairs, feats, cs.K, area, P, workers=1)
    cams = {n for c in cands for n in c["members"]}
    keys = sorted(k for k in cs.links if (k[0] in cams and k[1] in room) or (k[1] in cams and k[0] in room))
    db_attrs = RA.db_geometry(RL.db_of(cs), keys, cs.K, area, P)
    certs = {}
    for c in cands:
        L = RA.direct_links(cs.model, set(c["members"]), room, cs.links, cs.rots, new, db_attrs, P)
        ok, st = RA.certificate(L, P)
        certs[c["first_camera"]] = {"admitted": ok, **{k: st.get(k) for k in (
            "links", "db", "new", "honoured", "noncompact", "M_H", "H", "C", "mA", "nA", "G_A", "mB", "G_B", "C1",
            "C2", "C3")}}
    real = RA.one_blas_thread
    RA.one_blas_thread = contextlib.nullcontext                          # the same bag of words, NOT pinned
    try:
        _n2, V_unpinned = RA.build_bow(feats, P)
    finally:
        RA.one_blas_thread = real
    return {"openblas_threads": threads_before, "openblas_threads_after_bow": threads_after,
            "V_sha1": hashlib.sha1(np.ascontiguousarray(V).tobytes()).hexdigest(),
            "V_unpinned_sha1": hashlib.sha1(np.ascontiguousarray(V_unpinned).tobytes()).hexdigest(),
            "pairs": [list(p) for p in pairs],
            "links": [[x["a"], x["b"], x["inl"], np.asarray(x["R"]).round(12).tolist(), bool(x["noncompact"]),
                       round(float(x["amb"]), 9), round(float(x["hull_a"]), 12), round(float(x["hull_b"]), 12)]
                      for x in new],
            "db_noncompact": {f"{a}|{b}": bool(v["noncompact"]) for (a, b), v in sorted(db_attrs.items())},
            "certificates": certs}


def main():
    mode, run_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    tags = sys.argv[4:]
    if mode == "threads":
        import numpy  # noqa: F401,PLC0415  (FIRST: OpenBLAS starts with this process's own thread count)

        res = threads(run_dir, tags[0])
    else:
        res = replay(run_dir, tags) if mode == "replay" else fresh(run_dir, tags[0])
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
