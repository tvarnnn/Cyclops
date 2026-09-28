"""The direct-certificate retrieval admission: a piece the gate left outside the room joins it only on direct,
image-verified evidence (RUN experiments/C23-CERT: PREREG sections A-B, RESULTS, BRIEF-IMPL; manager 144 section 3,
145 section 1).

WHAT. With `TOWER_WORLD_RETRIEVAL_ADMISSION` on (`config.world_retrieval_admission_setting`; OFF by default, and off
is today's publish byte for byte), once per publish -- after the gate, the consensus and the anchor verification have
produced the chosen draw's published result, in both publish entry points (`coherence_publish.gate_and_publish` and
`regate_published`) -- `admitted` does, in this order:

  1. CANDIDATES (PREREG A.1). A gate group of the published result is a candidate iff: its label is not the room's;
     it is in the room's solver component; it holds >= 3 supported cameras; it is not withheld by the consensus and
     holds no camera the anchor verification sealed or took out as collateral (an admission never undoes a P4
     decision); the gate's `attach` is true (masks applied, metric levels available: it never overrides a
     fail-safe); and the gate's own scale check against the room (`coherence_gate.scale_levels_differ`, unchanged
     `GateParams`) is not True.
  2. RETRIEVAL, candidates only (A.2). The solve database's SIFT minus every keypoint on a masked pixel of the solve's
     final mask (`solve_masks._keypoints_in_mask`'s rule; an image with no mask png keeps nothing), a seeded
     bag of words (1,024 words), and each candidate camera's top 15 room cameras by cosine; pairs already in the
     database's `matches` are not tried again. Each new pair is verified: mutual nearest neighbours at ratio 0.8,
     the calibrated essential matrix (USAC_MAGSAC, 0.9999, 1.0 px, >= 15 inliers) and `recoverPose`, with the
     inlier hull, the homography share and the planar ambiguity on the E-inliers. EVERY PAIR IS VERIFIED UNDER ITS
     OWN SEED (`pair_seed`, from its two image names) and the pairs are taken in sorted order, so a parallel and a
     serial verification give the same links (manager 145 section 1(a); RESULTS section 9.6).
  3. THE CERTIFICATE (A.3-A.4) on L(G) = the database's direct links between G and the room (today's pycolmap
     readout) plus the verified new links (the calibrated readout):
       (C1) M_H >= 2, a maximum endpoint-disjoint matching over the honoured NONCOMPACT links;
       (C2) H > C, strictly: weights 1 / max(deg_G, deg_R) over ALL of L; a link without a readout counts in C;
       (C3) one consistent correction: m(A*) >= 2 and m(A*) > m(B*);
       (C4) the scale check, at certification (step 1) and again inside the re-gate at attachment.
  4. THE PROJECTION RE-GATE (B.1): `coherence_publish.gate_final_solution` on the same candidate with the same depth
     and scale, the consensus's withhold, the P4 seals carried, the CAMERA allow-list (the room's cameras -- P4's
     pre-seal room when it sealed -- plus the admitted ones), `admit` (the admitted cameras pass the redundancy and
     coupling tests on their certificate, and still need `attach`, `not barred` and the gate's own `scale_ok`), and
     the rider rule forced (`RIDER_MIN_SHARED`), so an admitted group's riders never take the room label by default.
  5. POST-CHECKS (B.2), else `not-applied` and the input is published: the anchor is kept; the room is exactly the
     input room plus the admitted cameras; every outside piece minus the admitted cameras is unchanged. Unchanged
     pieces keep the input's published reasons (`coherence_publish.keep_outside_pieces`).
  6. P4 AGAIN, when the anchor verification's parts are on: `anchor_verify.verify_published` on the projected room,
     its seal re-gate carrying `admit`, the carried seals and the projected room as the allow-list. An admitted group
     is image-verified like any attached group.
  7. THE RECORD `gate.retrieval_admission` (Tower-internal, additive, absent when off): ids and numbers only.

THE BINDING INVARIANT (manager 144 section 3; PREREG A.5). Retrieval links reach ONLY the certificate. Every gate
call of the admission receives the database reader's own `links` and `link_rotations` objects, read-only
(`MappingProxyType`), and the reader asserts it on every call: no retrieval link reaches block formation, the peel,
`_seal_pieces` or the P4 seal re-gate.

THE CHOICES THE STUDY LEFT OPEN (manager 145 section 1), made here:
  (a) the verifier's seed: per pair, from its image names (`pair_seed`); the same for the database links' 1.5 px
      re-fit; pairs in sorted order. A serial and a threaded verification give the same links. BUT (C23-IMPL probe,
      W4): OpenCV 5.0.0's USAC_MAGSAC is deterministic here without any seed, and the 70-vs-71 is the BAG OF WORDS:
      its float32 GEMM is not bitwise stable across BLAS thread counts (1 vs 8 threads: 499 query pairs each, about
      10 different, 71 vs 70 verified). Within one Tower (one BLAS configuration) the links are reproducible; across
      thread configurations they are not -- OPEN (pin the BoW's BLAS threads, or a tie-refined nearest word).
  (b) the consensus's withhold: a withheld group is never a candidate; the projection re-gate carries the withhold, so
      the withheld piece stays sealed with its one reason `seed-unstable`, and the post-checks require it unchanged.
  (c) the published reason: an outside piece whose keyframes the admission did not change keeps the reasons the
      input published (the frozen rule; the consensus's and P4's precedent: a piece's reason is the database gate's
      decision that produced it). A piece that LOSES admitted cameras has no reason in the closed set that fits its
      remainder (redundantly linked to cameras admitted on retrieval evidence) -- OPEN: needs a contract amendment --
      so the projection is `not-applied` and nothing new is published.
  (d) the cost: `stage_timing` stage `admission` (with `TOWER_WORLD_STAGE_TIMING` on), and per-stage seconds in the
      record.

THRESHOLDS: none new. Every value in `AdmissionParams` is PREREG section A's or P5-RETR's (PREREG B.3-B.6, C.1-C.2),
frozen and passed in C23-CERT; `honoured_deg` is the gate's `max_link_disagreement_deg`. `VERIFY_WORKERS` is not a
decision parameter (any value gives the same links) and is OPEN (the thread budget, BRIEF-IMPL section 6.3).

Never raises: any failure publishes the input result as it was, with `state: failed` in the record.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import logging
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from tower.world_builder import coherence_gate as CG
from tower.world_builder import stage_timing

logger = logging.getLogger(__name__)

ADMISSION_ID = "retrieval-admission:c23-cert-direct-certificate/1"

STATE_APPLIED = "applied"            # it ran; `admitted` may be empty (then the input is published unchanged)
STATE_NOT_APPLIED = "not-applied"    # a group was admitted but the projection failed its checks (`why`)
STATE_NOT_RUN = "not-run"            # nothing to admit to or nothing to query (`why`)
STATE_FAILED = "failed"              # an error (`detail`); the input is published as it was

# Not a decision parameter: every value gives the same links (per-pair seeds, sorted order). OPEN (BRIEF-IMPL 6.3):
# the thread budget beside the consensus's concurrent draws; 1 until the W0 owner sets it.
VERIFY_WORKERS = 1

_PAIR_BASE = 2147483647

# The certificate's clauses, in the order a refusal names the first that failed.
CLAUSES = ("C1", "C2", "C3")


@dataclass(frozen=True)
class AdmissionParams:
    """PREREG section A and P5-RETR's declared parameters (C23-CERT froze and passed them). Its own digest; the
    gate's `GateParams` and digest are unchanged."""

    # the bag of words (P5-RETR PREREG B.3): k-means on a seeded descriptor sample, tf-idf, L2, cosine
    words: int = 1024
    sample: int = 200_000
    seed: int = 0
    iterations: int = 15
    # the queries: each candidate camera's top-k room cameras
    k_room: int = 15
    # matching: mutual nearest neighbours with Lowe's ratio on masked descriptors
    ratio: float = 0.8
    # verification: USAC_MAGSAC essential matrix, then recoverPose on its inliers
    e_prob: float = 0.9999
    e_px: float = 1.0
    min_inliers: int = 15
    # the database links' calibrated re-fit on their stored inliers
    e_px_stored: float = 1.5
    # the homography: RANSAC at h_px; H-dominant at h_dominant (P5-RETR C.2, declared on the control)
    h_px: float = 4.0
    h_dominant: float = 0.9
    # noncompact (P5-RETR C.1-C.2): min-view inlier hull >= hull_min AND planar ambiguity < amb_max_deg
    hull_min: float = 0.10
    amb_max_deg: float = 10.0
    # honoured: the gate's `max_link_disagreement_deg` (a test pins them equal)
    honoured_deg: float = 16.8
    # the certificate
    min_group_cameras: int = 3
    m_h_min: int = 2
    m_a_min: int = 2

    def to_json(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        doc = {"admission": ADMISSION_ID, **self.to_json()}
        return hashlib.sha1(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------
# small helpers


def sorted_pair(a, b):
    return (a, b) if a <= b else (b, a)


def pair_seed(a: str, b: str) -> int:
    """The verifier's seed for one pair (manager 145 section 1(a)): from the two image names, order-free, the same in
    every process and on every schedule."""
    lo, hi = sorted_pair(str(a), str(b))
    return int.from_bytes(hashlib.sha1(f"{lo}|{hi}".encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF


def rot_deg(R) -> float:
    c = (float(np.trace(R)) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def chordal_mean(Rs):
    M = np.sum(np.asarray(Rs, np.float64), axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def matching(pairs) -> int:
    """The size of a maximum bipartite matching of (group camera, room camera) links: endpoint-disjoint links.
    Kuhn's augmenting paths, iterative (no recursion limit); the size is unique whatever the algorithm."""
    nbrs: dict = {}
    for a, b in pairs:
        nbrs.setdefault(a, set()).add(b)
    nbrs = {a: sorted(v) for a, v in nbrs.items()}
    match_b: dict = {}
    size = 0
    for root in sorted(nbrs):
        seen: set = set()
        stack = [(root, iter(nbrs[root]))]
        via: list = []
        while stack:
            a, it = stack[-1]
            nxt = None
            for b in it:
                if b not in seen:
                    nxt = b
                    break
            if nxt is None:
                stack.pop()
                if via:
                    via.pop()
                continue
            seen.add(nxt)
            if nxt not in match_b:
                match_b[nxt] = a
                for k in range(len(via) - 1, -1, -1):
                    match_b[via[k]] = stack[k][0]
                size += 1
                break
            via.append(nxt)
            stack.append((match_b[nxt], iter(nbrs[match_b[nxt]])))
    return size


# ---------------------------------------------------------------------------------------------------------------
# masked features, bag of words, queries (P5-RETR retr_lib, as C23-CERT ran it)


def _ro(database_path):
    import sqlite3  # noqa: PLC0415

    return sqlite3.connect(Path(database_path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def load_masked_features(database_path, mask_dir) -> tuple[dict, set, dict]:
    """({name: (xy float32 (n, 2), descriptors uint8 (n, 128))}, the pairs already in `matches` (sorted names), stats):
    the solve database's SIFT minus every keypoint on a masked pixel of the solve's final mask (`<name>.png`, 0 =
    masked; the pixel is floor(x), floor(y), clipped: `solve_masks._keypoints_in_mask`). An image without a mask png
    keeps nothing. `mask_dir` None keeps every keypoint (only for a solve that ran without masks, which the gate's
    `attach` already refuses). Read-only."""
    import cv2  # noqa: PLC0415

    con = _ro(database_path)
    try:
        names = dict(con.execute("select image_id, name from images").fetchall())
        keep, xy = {}, {}
        stats = {"images": 0, "kp_total": 0, "kp_kept": 0, "images_without_mask": 0}
        for iid, rows, cols, data in con.execute("select image_id, rows, cols, data from keypoints"):
            nm = names.get(iid)
            if nm is None:
                continue
            k = (np.frombuffer(data, np.float32).reshape(int(rows), int(cols))[:, :2].copy() if rows
                 else np.zeros((0, 2), np.float32))
            m = np.ones(len(k), bool)
            if mask_dir is not None:
                png = cv2.imread(str(Path(mask_dir) / f"{nm}.png"), cv2.IMREAD_GRAYSCALE)
                if png is None:
                    m[:] = False
                    stats["images_without_mask"] += 1
                elif len(k):
                    x = np.clip(np.floor(k[:, 0]).astype(np.int64), 0, png.shape[1] - 1)
                    y = np.clip(np.floor(k[:, 1]).astype(np.int64), 0, png.shape[0] - 1)
                    m = png[y, x] != 0
            keep[iid], xy[iid] = m, k
            stats["images"] += 1
            stats["kp_total"] += len(k)
            stats["kp_kept"] += int(m.sum())
        feats = {}
        for iid, rows, cols, data in con.execute("select image_id, rows, cols, data from descriptors"):
            if iid not in keep:
                continue
            d = np.frombuffer(data, np.uint8).reshape(int(rows), int(cols)) if rows else np.zeros((0, 128), np.uint8)
            m = keep[iid]
            feats[names[iid]] = (xy[iid][m], np.ascontiguousarray(d[m]))
        tried = set()
        for (pid,) in con.execute("select pair_id from matches"):
            b = int(pid) % _PAIR_BASE
            a = (int(pid) - b) // _PAIR_BASE
            if a in names and b in names:
                tried.add(sorted_pair(names[a], names[b]))
        return feats, tried, stats
    finally:
        con.close()


def _nearest(X, C, cc, chunk=8192):
    out = np.empty(len(X), np.int32)
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        d = cc[None, :] - 2.0 * (x @ C.T)
        out[s:s + chunk] = np.argmin(d, axis=1)
    return out


def build_bow(feats: dict, params: AdmissionParams) -> tuple[list, np.ndarray]:
    """k-means (`words` words, a `sample`-descriptor sample, seed `seed`, `iterations` Lloyd iterations from a seeded
    random start), tf-idf, L2-normalised. Returns (names in sorted order, V (n, words) float32)."""
    from scipy.sparse import csr_matrix  # noqa: PLC0415

    names = sorted(feats)
    rng = np.random.default_rng(params.seed)
    sizes = np.array([len(feats[n][1]) for n in names])
    alld = np.concatenate([feats[n][1] for n in names]).astype(np.float32)
    samp = alld[np.sort(rng.choice(len(alld), min(params.sample, len(alld)), replace=False))]
    C = samp[rng.choice(len(samp), params.words, replace=False)].copy()
    for _ in range(params.iterations):
        cc = (C * C).sum(1)
        a = _nearest(samp, C, cc)
        cnt = np.bincount(a, minlength=params.words)
        sums = csr_matrix((np.ones(len(a), np.float32), (a, np.arange(len(a)))),
                          shape=(params.words, len(a))) @ samp
        ok = cnt > 0
        C[ok] = sums[ok] / cnt[ok, None]
        dead = np.flatnonzero(~ok)
        if len(dead):
            C[dead] = samp[rng.choice(len(samp), len(dead), replace=False)]
    cc = (C * C).sum(1)
    words = _nearest(alld, C, cc)
    Hm = np.zeros((len(names), params.words), np.float32)
    img = np.repeat(np.arange(len(names)), sizes)
    np.add.at(Hm, (img, words), 1.0)
    df = (Hm > 0).sum(0)
    idf = np.where(df > 0, np.log(len(names) / np.maximum(df, 1)), 0.0).astype(np.float32)
    tf = Hm / np.maximum(Hm.sum(1, keepdims=True), 1.0)
    V = tf * idf[None, :]
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return names, V


def query_pairs(candidates: list, room: set, bow_names: list, V: np.ndarray, tried: set,
                params: AdmissionParams) -> list:
    """Each supported camera of each candidate group queries its top `k_room` ROOM cameras (cosine); a pair already
    in the database's `matches`, or already listed, is skipped. Returns [(candidate camera, room camera)]."""
    vi = {n: i for i, n in enumerate(bow_names)}
    room_list = sorted(n for n in room if n in vi)
    if not room_list:
        return []
    room_V = V[[vi[n] for n in room_list]]
    pairs, seen = [], set()
    for g in candidates:
        for q in g["members"]:
            if q not in vi:
                continue
            s = room_V @ V[vi[q]]
            for j in np.argsort(-s, kind="stable")[:params.k_room]:
                p = sorted_pair(q, room_list[j])
                if p in tried or p in seen:
                    continue
                seen.add(p)
                pairs.append((q, room_list[j]))
    return pairs


# ---------------------------------------------------------------------------------------------------------------
# verification (P5-RETR PREREG B.4), one seed per pair


def mnn(da, db, ratio: float):
    """Mutual nearest neighbours with Lowe's ratio (on distances) between two uint8 descriptor sets. The squared
    distances of integer-valued float32 descriptors are exact, so this does not depend on the BLAS schedule."""
    if len(da) < 2 or len(db) < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a = da.astype(np.float32)
    b = db.astype(np.float32)
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    np.maximum(d2, 0, out=d2)
    part = np.argpartition(d2, 1, axis=1)[:, :2]
    r = np.arange(len(a))
    v0, v1 = d2[r, part[:, 0]], d2[r, part[:, 1]]
    j = np.where(v0 <= v1, part[:, 0], part[:, 1])
    best, second = np.minimum(v0, v1), np.maximum(v0, v1)
    ratio_ok = np.sqrt(best) < ratio * np.sqrt(second)
    back = np.argmin(d2, axis=0)
    ok = ratio_ok & (back[j] == r)
    return r[ok], j[ok]


def hull_fraction(x, area: float) -> float:
    import cv2  # noqa: PLC0415

    if len(x) < 3:
        return 0.0
    h = cv2.convexHull(np.asarray(x, np.float32))
    return float(cv2.contourArea(h)) / area


def h_measures(xa, xb, K, params: AdmissionParams) -> tuple[float, int, float]:
    """(H-share, visible decompositions, their largest rotation difference): the homography at `h_px` on the given
    inliers (cv2's RANSAC is seeded internally)."""
    import cv2  # noqa: PLC0415

    if len(xa) < 4:
        return 0.0, 0, 0.0
    Hm, hm = cv2.findHomography(xa, xb, cv2.RANSAC, params.h_px)
    if Hm is None or hm is None:
        return 0.0, 0, 0.0
    hin = hm.ravel().astype(bool)
    share = float(hin.mean())
    try:
        _n, Rs, _ts, ns = cv2.decomposeHomographyMat(Hm, K)
        Ki = np.linalg.inv(K)
        na = (np.c_[xa[hin], np.ones(hin.sum())] @ Ki.T)[:, :2]
        nb = (np.c_[xb[hin], np.ones(hin.sum())] @ Ki.T)[:, :2]
        keep = cv2.filterHomographyDecompByVisibleRefpoints(
            Rs, ns, na.reshape(-1, 1, 2).astype(np.float32), nb.reshape(-1, 1, 2).astype(np.float32))
    except cv2.error:
        return share, 0, 0.0
    idx = [] if keep is None else [int(v) for v in np.asarray(keep).ravel()]
    amb = 0.0
    for i, j in itertools.combinations(idx, 2):
        amb = max(amb, rot_deg(np.asarray(Rs[i]).T @ np.asarray(Rs[j])))
    return share, len(idx), amb


def calibrated(xa, xb, K, px: float, params: AdmissionParams, seed: int):
    """cv2's 5-point E (USAC_MAGSAC, `e_prob`, threshold `px`) under `seed`, then recoverPose on its inliers.
    Returns (E-inliers, R_b_from_a or None, the inlier mask, points in front)."""
    import cv2  # noqa: PLC0415

    if len(xa) < 5:
        return 0, None, None, 0
    cv2.setRNGSeed(int(seed))
    try:
        Em, m = cv2.findEssentialMat(xa, xb, K, method=cv2.USAC_MAGSAC, prob=params.e_prob, threshold=px)
    except cv2.error:
        return 0, None, None, 0
    if Em is None or m is None or Em.shape[0] < 3:
        return 0, None, None, 0
    Em = Em[:3]
    inl = m.ravel().astype(bool)
    n = int(inl.sum())
    if n < 5:
        return n, None, inl, 0
    nf, R, _t, _pm = cv2.recoverPose(Em, xa[inl], xb[inl], K)
    return n, np.asarray(R, np.float64), inl, int(nf)


def ambiguity(h_share: float, amb_raw: float, params: AdmissionParams) -> float:
    """The planar ambiguity counts only when the pair is H-dominant."""
    return float(amb_raw) if h_share >= params.h_dominant else 0.0


def noncompact(hull_a: float, hull_b: float, amb: float, params: AdmissionParams) -> bool:
    """P5-RETR's declared definition: the MIN-view inlier hull >= `hull_min` and the ambiguity < `amb_max_deg`."""
    return (min(hull_a, hull_b) >= params.hull_min) and (amb < params.amb_max_deg)


def verify_pair(fa, fb, K, area: float, params: AdmissionParams, seed: int) -> dict:
    ka, da = fa
    kb, db = fb
    ia, ib = mnn(da, db, params.ratio)
    rec = {"putative": int(len(ia)), "inl": 0}
    if len(ia) < params.min_inliers:
        return rec
    xa = ka[ia].astype(np.float64)
    xb = kb[ib].astype(np.float64)
    n, R, inl, nf = calibrated(xa, xb, K, params.e_px, params, seed)
    rec["inl"] = n
    if n < params.min_inliers or R is None:
        return rec
    xa, xb = xa[inl], xb[inl]
    share, nvis, amb = h_measures(xa, xb, K, params)
    rec.update(R=R, n_front=nf, hull_a=hull_fraction(xa, area), hull_b=hull_fraction(xb, area), h_share=share,
               n_vis=nvis, amb_raw=amb)
    return rec


def verify_pairs(pairs: list, feats: dict, K, area: float, params: AdmissionParams, *, workers: int = 1) -> list:
    """The verified new links: [{a, b, R (b from a), inl, hull_a, hull_b, amb, noncompact}] for every pair with
    >= `min_inliers` E-inliers and a pose. Pairs are taken in sorted order and each is verified under its own seed
    (`pair_seed`), so the result is the same for any `workers` (a thread pool; cv2's RNG is per thread)."""
    order = sorted(pairs)

    def one(p):
        a, b = p
        return verify_pair(feats[a], feats[b], K, area, params, pair_seed(a, b))

    if workers > 1 and len(order) > 1:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            recs = list(ex.map(one, order))
    else:
        recs = [one(p) for p in order]
    out = []
    for (a, b), rec in zip(order, recs):
        if rec["inl"] >= params.min_inliers and "R" in rec:
            amb = ambiguity(rec["h_share"], rec["amb_raw"], params)
            out.append({"a": a, "b": b, "R": rec["R"], "inl": int(rec["inl"]), "hull_a": rec["hull_a"],
                        "hull_b": rec["hull_b"], "amb": amb,
                        "noncompact": noncompact(rec["hull_a"], rec["hull_b"], amb, params)})
    return out


# ---------------------------------------------------------------------------------------------------------------
# the database's direct links: stored inliers and their geometry


def read_pair_inliers(database_path, pairs) -> dict:
    """{(name_a, name_b): (xa, xb)} -- the stored verified inliers of each pair (named in either order), keyed and
    oriented in the DATABASE's own order (the lower image id first, as COLMAP's pair id and
    `coherence_gate.read_link_rotations`' keys are). Read-only; pairs it cannot find are absent."""
    con = _ro(database_path)
    try:
        ids = {name: iid for iid, name in con.execute("select image_id, name from images").fetchall()}
        kp: dict = {}

        def keypoints(i):
            if i not in kp:
                r, c, d = con.execute("select rows, cols, data from keypoints where image_id=?", (i,)).fetchone()
                kp[i] = np.frombuffer(d, np.float32).reshape(r, c)[:, :2].astype(np.float64)
            return kp[i]

        out = {}
        for a, b in pairs:
            ia, ib = ids.get(a), ids.get(b)
            if ia is None or ib is None or ia == ib:
                continue
            if ia > ib:
                a, b, ia, ib = b, a, ib, ia
            row = con.execute("select rows, data from two_view_geometries where pair_id=?",
                              (int(ia) * _PAIR_BASE + int(ib),)).fetchone()
            if row is None or row[1] is None or not row[0]:
                continue
            m = np.frombuffer(row[1], np.uint32).reshape(int(row[0]), 2)
            out[(a, b)] = (keypoints(ia)[m[:, 0]], keypoints(ib)[m[:, 1]])
        return out
    finally:
        con.close()


def db_link_attrs(xa, xb, K, area: float, params: AdmissionParams, seed: int) -> dict:
    """A database link's compactness (PREREG A.3): the hull on its stored inliers; the H-share and ambiguity on the
    calibrated `e_px_stored` re-fit's E-inliers (ambiguity 0 when the re-fit has < `min_inliers`)."""
    xa, xb = np.asarray(xa, np.float64), np.asarray(xb, np.float64)
    n, R, inl, _nf = calibrated(xa, xb, K, params.e_px_stored, params, seed)
    amb = 0.0
    if n >= params.min_inliers and R is not None:
        hs, _nv, ambr = h_measures(xa[inl], xb[inl], K, params)
        amb = ambiguity(hs, ambr, params)
    ha, hb = hull_fraction(xa, area), hull_fraction(xb, area)
    return {"hull_a": ha, "hull_b": hb, "amb": amb, "noncompact": noncompact(ha, hb, amb, params)}


# ---------------------------------------------------------------------------------------------------------------
# candidates, the direct link set and the certificate (PREREG A.1, A.3, A.4)


def _anchor(gated) -> dict | None:
    return next((g for g in gated.get("groups") or [] if g.get("label") == 0 and g.get("reference")), None)


def candidates(model: CG.SolveModel, gated: dict, r: np.ndarray, room: set, *, sealed: set, collateral: set,
               withheld: set, params: AdmissionParams, gp: CG.GateParams) -> tuple[list, list]:
    """PREREG A.1 on the published gate result `gated`. Returns (candidates, the same-component groups that fail
    only the scale check -- diagnostics, never admitted). `withheld`: the consensus's withheld groups' first
    cameras; `sealed`, `collateral`: camera names the anchor verification sealed or took out."""
    idx = model.index()
    sup = model.n_obs >= gp.min_obs
    an = _anchor(gated)
    room_comp = int(an["source_component"]) if an else None
    room_idx = np.asarray(sorted(idx[n] for n in room if n in idx), np.int64)
    held = {nm for g in gated.get("groups") or [] if g.get("first_camera") in withheld for nm in g["members"]}
    attach = bool((gated.get("evidence") or {}).get("attach"))
    out, diag = [], []
    for g in gated.get("groups") or []:
        if g.get("label") == 0 or g.get("sealed"):
            continue
        mem = [n for n in g["members"] if n in idx and sup[idx[n]]]
        if len(mem) < params.min_group_cameras or room_comp is None or int(g["source_component"]) != room_comp:
            continue
        gi = np.asarray(sorted(idx[n] for n in mem), np.int64)
        differ = CG.scale_levels_differ(r, room_idx, gi, gp)
        lg, ng = CG._level(r, gi)
        lk, _ = CG._level(r, room_idx)
        rec = {"first_camera": g["first_camera"], "label": g["label"], "n": len(mem), "members": mem,
               "scale_ok": differ is not True, "ratios": int(ng),
               "scale_factor": (round(math.exp(lg - lk), 4) if lg is not None and lk is not None else None)}
        excl = []
        if g["first_camera"] in withheld or any(n in held for n in mem):
            excl.append("withheld")
        if any(n in sealed for n in mem):
            excl.append("sealed")
        if any(n in collateral for n in mem):
            excl.append("p4-collateral")
        if not attach:
            excl.append("no-attach")
        rec["excluded"] = excl
        if excl:
            continue
        (out if rec["scale_ok"] else diag).append(rec)
    return out, diag


def direct_links(model: CG.SolveModel, G: set, room: set, links, rots, new_links: list, db_attrs: dict,
                 params: AdmissionParams) -> list:
    """L(G) (PREREG A.3): every verified database link and every verified new link between G's cameras and the
    room's. Each: src, a, b, g, r, inl, has_readout, res, hon, D, noncompact, hull_min, hull_max, amb."""
    idx = model.index()
    tau = params.honoured_deg
    L = []

    def geom(a, b, R):
        if R is None:
            return float("nan"), None
        Ra, Rb = model.R_cw[idx[a]], model.R_cw[idx[b]]
        res = rot_deg(np.asarray(R).T @ (Rb @ Ra.T))
        Q = Rb.T @ np.asarray(R) @ Ra
        return res, (Q if b in G else Q.T)

    for (a, b), inl in links.items():
        if not ((a in G and b in room) or (b in G and a in room)):
            continue
        key = (a, b) if (a, b) in rots else (b, a)
        R = rots.get(key)
        at = db_attrs.get(key) or db_attrs.get((key[1], key[0]))
        if at is None:
            # No stored inliers to measure. Only a link without a readout can matter here, and such a link is never
            # honoured and has no correction: only C2's contradicted weight counts it, where compactness is not used
            # (PREREG F, plumbing fact 1).
            at = {"hull_a": 0.0, "hull_b": 0.0, "amb": 0.0, "noncompact": False}
        ka, kb = key
        res, D = geom(ka, kb, R)
        g_, r_ = (ka, kb) if ka in G else (kb, ka)
        L.append({"src": "db", "a": ka, "b": kb, "g": g_, "r": r_, "inl": int(inl), "has_readout": R is not None,
                  "res": res, "hon": bool(R is not None and res <= tau), "D": D,
                  "noncompact": bool(at["noncompact"]), "hull_min": min(at["hull_a"], at["hull_b"]),
                  "hull_max": max(at["hull_a"], at["hull_b"]), "amb": at["amb"]})
    for x in new_links:
        a, b = x["a"], x["b"]
        if not ((a in G and b in room) or (b in G and a in room)):
            continue
        res, D = geom(a, b, x["R"])
        g_, r_ = (a, b) if a in G else (b, a)
        L.append({"src": "new", "a": a, "b": b, "g": g_, "r": r_, "inl": int(x["inl"]), "has_readout": True,
                  "res": res, "hon": bool(res <= tau), "D": D, "noncompact": bool(x["noncompact"]),
                  "hull_min": min(x["hull_a"], x["hull_b"]), "hull_max": max(x["hull_a"], x["hull_b"]),
                  "amb": x["amb"]})
    return L


def certificate(L: list, params: AdmissionParams) -> tuple[bool, dict]:
    """PREREG A.4 (C1)-(C3) on a direct link set. Returns (all three pass, the statistics)."""
    tau = params.honoured_deg
    st = {"links": len(L), "db": sum(1 for x in L if x["src"] == "db"),
          "new": sum(1 for x in L if x["src"] == "new"), "honoured": sum(1 for x in L if x["hon"])}
    if not L:
        st.update(noncompact=0, M_H=0, H=0.0, C=0.0, mA=0, G_A=None, nA=0, mB=0, G_B=None,
                  C1=False, C2=False, C3=False)
        return False, st
    nc = [x for x in L if x["noncompact"]]
    st["noncompact"] = len(nc)
    # (C1) endpoint-disjoint, noncompact honoured evidence
    MH = matching([(x["g"], x["r"]) for x in nc if x["hon"]])
    # (C2) honoured outweighs contradicted, over ALL of L
    dg = Counter(x["g"] for x in L)
    dk = Counter(x["r"] for x in L)
    H = C = 0.0
    for x in L:
        w = 1.0 / max(dg[x["g"]], dk[x["r"]])
        if x["hon"]:
            H += w
        else:
            C += w
    # (C3) one consistent correction over the noncompact links with a readout
    N = [x for x in nc if x["D"] is not None]
    best_A, A_star = None, None
    B_star_m, G_B = 0, None
    if N:
        Ds = np.asarray([x["D"] for x in N], np.float64)
        cos_tau = math.cos(math.radians(tau))
        cosm = (np.einsum("nij,mij->nm", Ds, Ds) - 1.0) / 2.0
        seen: dict = {}
        for i in range(len(N)):
            S = np.flatnonzero(cosm[i] >= cos_tau)
            Gi = chordal_mean(Ds[S])
            c2 = (np.einsum("nij,ij->n", Ds, Gi) - 1.0) / 2.0
            S2 = tuple(int(k) for k in np.flatnonzero(c2 >= cos_tau))
            if not S2:
                continue
            if S2 not in seen:
                seen[S2] = (matching([(N[k]["g"], N[k]["r"]) for k in S2]), rot_deg(chordal_mean(Ds[list(S2)])))
            m_, ang = seen[S2]
            if ang <= tau:
                key = (m_, len(S2), -ang, -i)
                if A_star is None or key > best_A:
                    best_A, A_star = key, (S2, m_, ang)
            elif m_ > B_star_m or (m_ == B_star_m and G_B is None):
                B_star_m, G_B = m_, ang
    mA = A_star[1] if A_star else 0
    st.update(M_H=MH, H=round(H, 4), C=round(C, 4), mA=mA, G_A=(round(A_star[2], 2) if A_star else None),
              nA=(len(A_star[0]) if A_star else 0), mB=B_star_m, G_B=(round(G_B, 2) if G_B is not None else None))
    st["C1"] = MH >= params.m_h_min
    st["C2"] = H > C
    st["C3"] = mA >= params.m_a_min and mA > B_star_m
    return bool(st["C1"] and st["C2"] and st["C3"]), st


# ---------------------------------------------------------------------------------------------------------------
# the projection's post-checks (PREREG B.2)


def _room_of(labels: dict, names, sup) -> set:
    return {n for i, n in enumerate(names) if sup[i] and labels.get(n) == 0}


def _pieces_of(labels: dict, names, sup) -> dict:
    out: dict = {}
    for i, n in enumerate(names):
        if sup[i] and labels.get(n) != 0:
            out.setdefault(labels.get(n), set()).add(n)
    return {k: frozenset(v) for k, v in out.items()}


def post_checks(model: CG.SolveModel, base: dict, proj: dict, admitted: set, gp: CG.GateParams):
    """(a) the input's anchor is kept; (b) evidence closure: the projected room is exactly the input room plus the
    admitted cameras; (c) every input piece outside the room, minus the admitted cameras, is one projected piece (or
    empty), and every projected outside piece is such a piece. Per supported camera. Returns (why or None, info)."""
    names = model.names
    idx = model.index()
    sup = model.n_obs >= gp.min_obs
    rb, rp = _room_of(base["labels"], names, sup), _room_of(proj["labels"], names, sup)
    an = _anchor(base)
    anc = {n for n in (an["members"] if an else []) if n in idx and sup[idx[n]]}
    info = {"room_before": len(rb), "room_after": len(rp), "admitted_cameras": len(admitted)}
    if not anc <= rp:
        return "the input's anchor is not all in the projected room", info
    if rp != rb | admitted:
        info["closure_missing"] = len((rb | admitted) - rp)
        info["closure_extra"] = len(rp - (rb | admitted))
        return "evidence closure: the projected room is not the input room plus the admitted cameras", info
    pb, pp = _pieces_of(base["labels"], names, sup), _pieces_of(proj["labels"], names, sup)
    expect = {frozenset(v - admitted) for v in pb.values() if v - admitted}
    if expect != set(pp.values()):
        return "an outside piece changed", info
    return None, info


# ---------------------------------------------------------------------------------------------------------------
# the whole step


def admitted(store, world_id: str, session_id: str, result, *, database_path, keyframes, workspace_root,
             params: AdmissionParams | None = None, gp: CG.GateParams | None = None):
    """The retrieval admission on a published gate result (`coherence_publish.GateResult`; the chosen draw after the
    consensus and the anchor verification). Off (`TOWER_WORLD_RETRIEVAL_ADMISSION` unset, blank, `off` or garbage):
    `result`, the very object, untouched. On: the result to publish, its record carrying `retrieval_admission`.
    Never raises."""
    from tower.config import world_retrieval_admission_setting  # noqa: PLC0415

    if not world_retrieval_admission_setting():
        return result
    return _admit(store, world_id, session_id, result, database_path=database_path, keyframes=keyframes,
                  workspace_root=workspace_root, params=params or AdmissionParams(), gp=gp or CG.GateParams())


@stage_timing.timed("admission")
def _admit(store, world_id, session_id, result, *, database_path, keyframes, workspace_root, params, gp):
    started = time.perf_counter()
    audit: dict = {"id": ADMISSION_ID, "params": params.to_json(), "params_digest": params.digest(),
                   "rider_min_shared": None, "seconds": {}}
    try:
        return _run(store, world_id, session_id, result, database_path=database_path, keyframes=keyframes,
                    workspace_root=workspace_root, params=params, gp=gp, audit=audit, started=started)
    except Exception as exc:  # noqa: BLE001 -- the input is published as it was, and the record says why
        logger.exception("[Tower][WorldBuilder] the retrieval admission failed on %s/%s; the room is published as "
                         "it was gated", world_id, session_id)
        try:
            return _done(result, result, audit, STATE_FAILED, started, detail=f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001 -- not even the record could be written: the input, exactly
            return result


def _carried(result, published, seconds: float = 0.0):
    """`published` (the input, or a re-gate of it) as the input publishes it: the input's record with the gate keys of
    what is published (the anchor verification's own record stays the input's), and the input's consensus detail,
    draw 0, withhold, depth and scale. Like `anchor_verify.verify_published`'s `done`."""
    from tower.world_builder import anchor_verify as AV  # noqa: PLC0415

    record = dict(result.record or {})
    if published is not result:
        record.update({k: published.record.get(k) for k in AV._GATE_KEYS if k in published.record})
    record["seconds"] = round(float((result.record or {}).get("seconds") or 0.0) + float(seconds), 3)
    return dataclasses.replace(published, record=record, consensus_detail=result.consensus_detail,
                               draw_0=result.draw_0, withheld=result.withheld, depth=result.depth, scale=result.scale)


def _done(result, published, audit: dict, state: str, started: float, **kw):
    """The result to publish, its record carrying the audit."""
    audit.update(state=state, **kw)
    audit["seconds"]["total"] = round(time.perf_counter() - started, 3)
    out = _carried(result, published, audit["seconds"]["total"])
    out.record["retrieval_admission"] = CG._json_clean(audit)
    return out


def _names_of_kids(kids, name_of: dict) -> set:
    return {name_of[k] for k in kids or [] if k in name_of}


def _run(store, world_id, session_id, result, *, database_path, keyframes, workspace_root, params, gp, audit,
         started):
    from tower.world_builder import coherence_publish as CP  # noqa: PLC0415

    T = audit["seconds"]

    def lap(key, t0):
        T[key] = round(time.perf_counter() - t0, 3)

    rec = result.record or {}
    if (rec.get("state") != CP.GATE_STATE_APPLIED or not rec.get("attach") or result.gated is None
            or result.components is None or result.candidate is None):
        return _done(result, result, audit, STATE_NOT_RUN,
                     started, why="the gate attached nothing (a fail-safe) or published no components record: there "
                                  "is no room to admit to")
    t = time.perf_counter()
    name_of = CP._image_names(keyframes)
    kid_of_name = {v: k for k, v in name_of.items()}
    candidate = result.candidate
    model = CP.solve_model(candidate, name_of)
    idx = model.index()
    sup = model.n_obs >= gp.min_obs
    gated = result.gated
    room = _room_of(gated["labels"], model.names, sup)
    # The anchor verification's decisions, which an admission never undoes: its seals and its collateral (only when
    # it was applied: a `not-applied` record lists the seals it did NOT publish).
    av = rec.get("anchor_verify") or {}
    sealed: dict = {}
    collateral: set = set()
    if av.get("state") == "applied":
        for why, block in (av.get("sealed") or {}).items():
            for kid in (block or {}).get("keyframe_ids") or []:
                if kid in name_of:
                    sealed[name_of[kid]] = why
        collateral = _names_of_kids((av.get("collateral") or {}).get("keyframe_ids"), name_of)
    allow_base = set(room)
    if sealed:
        # The seal re-gate's allow-list was P4's pre-seal room: the room now, plus what it sealed and took out.
        allow_base = room | {n for n in sealed if n in idx} | {n for n in collateral if n in idx}
        if av.get("room_before") is not None and len(allow_base) != int(av["room_before"]):
            return _done(result, result, audit, STATE_NOT_RUN, started,
                         why="the anchor verification's pre-seal room could not be rebuilt from its record")
    withheld = set(result.withheld or [])
    r = np.full(model.n, np.nan)
    if gated.get("metric_available"):
        for nm, v in ((result.scale or {}).get("metric_log") or {}).items():
            if nm in idx and v is not None and np.isfinite(v):
                r[idx[nm]] = float(v)
    cands, diag = candidates(model, gated, r, room, sealed=set(sealed), collateral=collateral, withheld=withheld,
                             params=params, gp=gp)
    audit["room_before"] = len(room)
    audit["carried"] = {"withheld_groups": len(withheld), "sealed_keyframes": len(sealed),
                        "collateral_keyframes": len(collateral)}
    audit["scale_refused"] = [{"first_keyframe": kid_of_name.get(c["first_camera"]), "keyframes": c["n"],
                               "scale_factor": c["scale_factor"], "ratios": c["ratios"]} for c in diag]
    lap("candidates", t)
    if not cands:
        audit["candidates"] = []
        return _done(result, result, audit, STATE_NOT_RUN, started, why="no candidate group")

    # the database's links and readouts: read ONCE, and the only links any gate call of the admission sees
    t = time.perf_counter()
    links_raw, rots_raw = CP.read_links(database_path, candidate.camera, gp.min_link_inliers)
    links, rots = MappingProxyType(links_raw), MappingProxyType(rots_raw)
    n_links, n_rots = len(links), len(rots)
    gate_calls = {"n": 0}

    def db_reader(database, camera, min_inliers):
        # THE BINDING INVARIANT (PREREG A.5): every gate call gets the database reader's own objects, read-only and
        # unchanged -- never a retrieval link.
        if len(links) != n_links or len(rots) != n_rots:
            raise RuntimeError("the database links changed under the admission")
        gate_calls["n"] += 1
        return links, rots

    lap("db_links", t)

    # retrieval, candidate cameras only
    cam = candidate.camera or {}
    K = np.array([[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]], np.float64)
    area = float(cam["width"]) * float(cam["height"])
    new, rstats = retrieve(database_path, workspace_root, cands, room, K, area, params, T)
    audit["retrieval"] = rstats
    if any(sorted_pair(x["a"], x["b"]) in links for x in new):
        # A retrieval link is a pair the matcher never tried, so it is never one of the database's links.
        raise RuntimeError("a retrieval link is among the database links")

    # the direct database links' geometry (their stored inliers), candidates <-> room: every one, as PREREG A.3
    # defines compactness for every database link (for a link without a readout it is only ever reported)
    t = time.perf_counter()
    cand_cams = {n for c in cands for n in c["members"]}
    keys = sorted(k for k in links if (k[0] in cand_cams and k[1] in room) or (k[1] in cand_cams and k[0] in room))
    db_attrs = db_geometry(database_path, keys, K, area, params)
    lap("db_link_geometry", t)

    # the certificates
    t = time.perf_counter()
    rows, admitted_groups = [], []
    for c in cands:
        L = direct_links(model, set(c["members"]), room, links, rots, new, db_attrs, params)
        ok, st = certificate(L, params)
        row = {"first_keyframe": kid_of_name.get(c["first_camera"]), "keyframes": c["n"], "label": c["label"],
               "scale_factor": c["scale_factor"], "ratios": c["ratios"], **st, "C4": True, "admitted": ok}
        if not ok:
            row["refused_by"] = next(k for k in CLAUSES if not st[k])
        rows.append(row)
        if ok:
            admitted_groups.append(c)
    audit["candidates"] = rows
    lap("certificate", t)
    adm = {n for c in admitted_groups for n in c["members"]}
    audit["admitted"] = [{"first_keyframe": kid_of_name.get(c["first_camera"]), "keyframes": c["n"]}
                         for c in admitted_groups]
    audit["admitted_keyframes"] = len(adm)
    if not adm:
        audit["room_after"] = len(room)
        return _done(result, result, audit, STATE_APPLIED, started)

    # the projection re-gate: database links only, the camera allow-list, admit, the riders forced
    t = time.perf_counter()
    depth = result.depth or {}
    audit["rider_min_shared"] = CP.RIDER_MIN_SHARED

    def regate(*, seal, room):
        return CP.gate_final_solution(
            store, world_id, session_id, candidate, database_path=database_path, keyframes=keyframes, params=gp,
            depth_runner=lambda *a, **kw: (depth.get("align"), depth.get("work"), depth.get("dparams")),
            metric_fn=lambda *a, **kw: result.scale, withhold=result.withheld or None, room=room,
            seal=seal or None, link_reader=db_reader, admit=sorted(adm), rider_min_shared=CP.RIDER_MIN_SHARED)

    proj = regate(seal=dict(sealed), room=sorted(allow_base | adm))
    lap("projection", t)
    audit["invariant"] = {"database_links": n_links, "database_rotations": n_rots}
    if proj.record.get("state") != CP.GATE_STATE_APPLIED or proj.components is None or proj.gated is None:
        audit["invariant"]["gate_calls"] = gate_calls["n"]
        return _done(result, result, audit, STATE_NOT_APPLIED, started,
                     why="the projection re-gate did not publish a components record")
    why, info = post_checks(model, gated, proj.gated, adm, gp)
    audit["projection"] = info
    if why is None:
        why = _lost_admitted(model, gated, adm, gp)
    if why is None:
        audit["projection"]["reasons_kept"] = _reasons_kept(result, proj)
        why = CP.keep_outside_pieces(result, proj, set(), min_obs=gp.min_obs)
    if why is not None:
        audit["invariant"]["gate_calls"] = gate_calls["n"]
        logger.warning("[Tower][WorldBuilder] retrieval admission on %s/%s: the projection failed its check (%s); "
                       "the room is published as it was gated", world_id, session_id, why)
        return _done(result, result, audit, STATE_NOT_APPLIED, started,
                     why=f"the projection failed its check ({why}); the room is published as it was gated")
    projected = _carried(result, proj)

    # the anchor verification again, on the projected room, when its parts are on
    final = projected
    parts = CP._anchor_verify_parts()
    from tower.world_builder import anchor_verify as AV  # noqa: PLC0415

    if {AV.PART_SCALE, AV.PART_IMAGES, AV.PART_MOTION} & parts:
        t = time.perf_counter()
        carried = dict(sealed)
        extra = {"motion_seal": True} if CP._pose_quarantine_on(CP.PQ_SEAL) else {}
        rerun = AV.verify_published(projected, keyframes=keyframes, parts=parts,
                                    regate=lambda *, seal, room: regate(seal={**carried, **seal}, room=room),
                                    workspace_root=workspace_root, min_obs=gp.min_obs, gp=gp, **extra)
        a2 = (rerun.record or {}).get("anchor_verify") or {}
        audit["p4_rerun"] = {k: a2.get(k) for k in ("state", "why", "detail", "sealed_kf", "room_before",
                                                   "room_after", "cap_hit") if k in a2}
        audit["p4_rerun"]["sealed"] = {w: (b or {}).get("keyframes") for w, b in (a2.get("sealed") or {}).items()}
        lap("p4_rerun", t)
        if a2.get("state") == AV.STATE_FAILED:
            audit["invariant"]["gate_calls"] = gate_calls["n"]
            return _done(result, result, audit, STATE_FAILED, started,
                         detail="the anchor verification of the admitted room failed; the room is published as it "
                                "was gated")
        final = rerun
    audit["invariant"]["gate_calls"] = gate_calls["n"]
    audit["room_after"] = len(CP._room_kids(final.solution, gp.min_obs))
    return _done(result, final, audit, STATE_APPLIED, started)


def retrieve(database_path, workspace_root, cands: list, room: set, K, area: float, params: AdmissionParams,
             T: dict) -> tuple[list, dict]:
    """Step 2: the masked features, the bag of words, the candidates' queries and their verification. Returns (the
    verified new links, the counts). `T` is given each stage's seconds."""
    from tower.world_builder import solve_masks  # noqa: PLC0415

    t = time.perf_counter()
    mask_dir = Path(workspace_root) / solve_masks.MASKS_DIRNAME
    if not mask_dir.is_dir():
        raise FileNotFoundError("the solve's masks directory is missing")
    feats, tried, fstats = load_masked_features(database_path, mask_dir)
    T["features"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    bow_names, V = build_bow(feats, params)
    T["bow"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    pairs = query_pairs(cands, room, bow_names, V, tried, params)
    T["queries"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    new = verify_pairs(pairs, feats, K, area, params, workers=VERIFY_WORKERS)
    T["verify"] = round(time.perf_counter() - t, 3)
    return new, {"images": fstats["images"], "images_without_mask": fstats["images_without_mask"],
                 "keypoints": fstats["kp_total"], "keypoints_kept": fstats["kp_kept"],
                 "pairs_queried": len(pairs), "pairs_verified": len(new), "workers": int(VERIFY_WORKERS)}


def db_geometry(database_path, keys: list, K, area: float, params: AdmissionParams) -> dict:
    """{key in the database's order: db_link_attrs} for the database links `keys` (named in either order), each
    re-fitted under its own seed."""
    if not keys:
        return {}
    inliers = read_pair_inliers(database_path, keys)
    return {k: db_link_attrs(xa, xb, K, area, params, pair_seed(*k)) for k, (xa, xb) in sorted(inliers.items())}


def _lost_admitted(model: CG.SolveModel, base: dict, adm: set, gp: CG.GateParams) -> str | None:
    """Decision (c): an outside piece that LOSES admitted cameras keeps a remainder that no reason of the closed set
    describes (it is linked to the room only through cameras admitted on retrieval evidence). OPEN: needs a contract
    amendment -- until then nothing new is published."""
    sup = model.n_obs >= gp.min_obs
    for v in _pieces_of(base["labels"], model.names, sup).values():
        if v & adm and not v <= adm:
            return ("an outside piece would lose admitted cameras, and no reason of the contract's closed set fits "
                    "its remainder (OPEN: needs a contract amendment)")
    return None


def _reasons_kept(result, proj) -> list:
    """The outside pieces whose reasons the projection re-gate gave differently from the input's published ones
    (the input's are kept: decision (c)). Ids and reasons only."""
    src = {e["id"]: list(e.get("reasons") or []) for e in (result.components or {}).get("components") or []
           if e.get("state") != "placed"}
    out = []
    for e in (proj.components or {}).get("components") or []:
        if e.get("state") == "placed" or e.get("id") not in src:
            continue
        if list(e.get("reasons") or []) != src[e["id"]]:
            out.append({"id": e["id"], "keyframes": e.get("keyframes"), "published": src[e["id"]],
                        "regate": list(e.get("reasons") or [])})
    return out
