"""The final-solve scale and pose PUBLICATION guard (`TOWER_WORLD_FINAL_SCALE_GUARD`, off by default).

V2 (Exp1 slice 4; design RUN review/codex/cx-SCALE-GUARD-V2-DESIGN-20261007.md) IS THE RULE THE SEAM RUNS
(`guard_publication`, `assess_v2`); the description below is v1's and still holds for v1's `assess` (the
offline shadow CLI) except where v2 says otherwise:
  * EVIDENCE IS READ, NEVER MADE (`read_evidence`): the applied gate's `metric_log` first, else the session's
    live depth cache under a full identity match (`live_depth.compatible_live_records`), else none. No depth
    stage runs at publish.
  * ISOLATE only a plateau with >= 10 finite ratios, its median outside the inclusive [0.8, 1.25] AND its
    deterministic 95 % bootstrap interval (400 draws, seed 20261006) excluding 1. A plateau is bounded by
    its first and last MEASURED camera: no class is carried across an unmeasured tail. Re-segment after
    each removal.
  * EVERYTHING ELSE STAYS ATTACHED and is recorded `uncertified`, in `final_scale_guard.json` and on the
    placed component of `components.json` (`placed_scale_status`). No pose-only quarantine (pose screen
    recorded only). No evidence: the room is `uncertified` (never PASS), nothing is isolated, nothing is
    withheld. ON still withholds on a failure (an exception, a relabel that leaves no room).

WHY. The evidence gate decides which pieces of a final solve to attach to the room, but it does not check
the result it publishes: the historical walk-3 bathroom was published inside the room at x2.56 relative to
it despite the gate, and the live product (the "simple" solve: unmasked, ungated, one draw) publishes the
solver's component 0 with no check at all. The guard checks THE ACTUAL OUTCOME THAT WOULD BE PUBLISHED --
the chosen draw, its anchor block, every attached group -- immediately before it is written, and can only
REMOVE room membership: it never attaches anything and never overrides a gate refusal.

Design: RUN review/codex/cx-SCALE-GUARD-DESIGN-20261006.md (binding, manager 234). Method: a port of the
read-only checker RUN experiments/SIMPLE-SOLVE/tools/scale_outlier_check.py (per-camera log(z_sfm/z_metric),
the [0.8, 1.25] screen, >= 10 ratios per side, the P4 motion screens), with its LABEL-DEFINED segments
replaced by label-free ones (below). Region labels are evaluation only; nothing here reads one.

THE SEAM (`coherence_publish.gate_and_publish` and `coherence_publish.regate_published`): after the gate,
the consensus and the anchor verification, before `write`. So it covers every publication pass -- the
consensus's early draw-0 publish, its final vote, the plain gate, the simple (ungated) solve, a re-finish
(a final solve) and the re-gate in place.

WHAT IT MEASURES, on the candidate's ROOM (component 0, supported cameras: >= 30 observations):

  1. SCALE. Per camera log(z_sfm/z_metric) (`coherence_scale.metric_scale`: the solve database's verified
     inlier pairs, the candidate's poses, the depth stage's metric predictions). The gate's own measurement
     is reused when it ran; on an ungated solve the guard runs the same depth stage (whose predictions are
     cached and handed to the surface, which would otherwise make them after publish).
  2. CANDIDATE PIECES, label-free: the gate's groups of the room (one group on an ungated solve), each cut
     in capture order into PLATEAUS -- runs whose centred running median (11 measured cameras) sits on the
     same side of the screen band around the reference. A camera without a ratio joins the plateau of the
     nearest measured camera before it.
  3. THE REFERENCE is the dominant room core: first the densest x1.25-wide window of camera ratios, then
     the median of the cameras in certified in-band plateaus, re-segmented until the decisions stop
     changing (a contaminated first median cannot hide a second mismatch).
  4. DECISIONS, per plateau with >= 10 ratios: its median ratio to the reference inside [0.8, 1.25] is
     kept; below 0.75 or above 1.333 (the CLEAR boundary) is isolated `final-scale-outlier`; in the
     borderline bands between is isolated `final-scale-uncertified` -- borderline is never described as a
     measured gross error, whatever its deterministic bootstrap interval says (recorded). A stretch of
     plateaus with < 10 ratios each is kept only when it sits inside the band and a certified plateau of the
     same group bounds it on BOTH sides (it inherits their certificate, recorded as `inherited`); otherwise
     it is isolated `final-scale-uncertified`: an unmeasured independent piece never gets a room claim.
     (Cameras with no ratio inside a certified plateau are kept with it, the same inheritance; the plateau
     rows record `keyframes` and `measured`.)
  5. POSES, after the reference is fixed (metres per unit = exp(-reference)), over the kept cameras in
     capture order: a consecutive pair (<= 3 journal indices, <= 2 s) is a HARD step when it moves more than
     max(allowance + 0.5 m, 2 x allowance), allowance = 2.0 m/s x dt + 0.3 m (P4), or more than the
     allowance AND a corroborating screen fires (> 0.75 m vertical step in <= 1 s; an endpoint > 1.5 m off
     the room's median height, > 120 deg from its median up, or > 15 m from its median centre). A smaller
     exceedance is a recorded WARNING only. A camera with two or more of the per-camera defects, or a
     non-finite pose, is quarantined outright. The room is cut at hard steps; a run rejoins an earlier one
     when the step between them would not have been hard; the run cluster with the most cameras is the
     room, and every other cluster is isolated `final-pose-outlier` -- each its own piece, so a step never
     moves two otherwise coherent sides together.
  6. If the remaining room has fewer than 10 ratios, nothing is certified: the candidate is WITHHELD.

WHAT IT DOES (`on`). Each isolated piece becomes its own unplaced component (its own label; coordinates
untouched; its points go with the majority of their observers, `coherence_publish.relabel_solution`), with
the new reason, in the components record (`components.json`, contract §2) -- written on an ungated solve
too, where every other solver component is `solved-separately`. Captured imagery is never dropped. A
withheld candidate is not written at all (`publish.written: false`): the last published solution stands,
and the solve reports `solved: false` with the reason, which the builder records as the finalization's
detail. ON IS FAIL-CLOSED: no depth, too few ratios, a failed gate, a relabel that leaves no room, or any
exception withholds. `shadow` measures and writes the audit but publishes exactly what off publishes.

THE AUDIT (`solve/<session>/final_scale_guard.json`, Tower-internal, keyed to the published or withheld
solve's identity): the params and digest, the reference, every plateau and piece with its keyframe ids,
journal indices, ratio count, median ratio, bootstrap interval, class and decision, the pose flags and
warnings, and the timing -- so a scorer can see which segments were isolated and why. No ratio or metric
figure ever reaches the components record or the phone (contract §2.4 rule 6).

OFF performs no new read, write, serialization, timestamp or record change: `mode()` reads only the
environment, and every caller does nothing else when it says `off`
(`tests/test_world_builder_final_scale_guard_off_identity.py`, golden recorded on 324f4a6).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# v2 (Exp1 slice 4) is the rule the publish seam runs; v1's `assess` stays only for the offline shadow CLI.
GUARD_ID = "final-scale-guard/2"
GUARD_ID_V1 = "final-scale-guard/1"
RECORD = "wb-final-scale-guard/2"
AUDIT_FILENAME = "final_scale_guard.json"

# v2's room scale status: `certified` only with evidence, a >= 10-ratio in-band core and no attached
# uncertified stretch. Never PASS; no evidence is `uncertified`.
SCALE_CERTIFIED = "certified"
SCALE_UNCERTIFIED = "uncertified"

# v2's evidence, in the order it is looked for. Read only: no depth is predicted at publish.
EVIDENCE_GATE = "gate-metric-log"
EVIDENCE_LIVE = "live-depth-cache"
EVIDENCE_NONE = "none"

# Why a stretch stays attached but uncertified (v2).
UNCERTIFIED_NO_EVIDENCE = "no-evidence"
UNCERTIFIED_UNMEASURED = "unmeasured"
UNCERTIFIED_SHORT = "short"
UNCERTIFIED_CI = "ci-includes-one"
UNCERTIFIED_NO_CORE = "no-certified-core"
UNCERTIFIED_EXCURSION = "excursion"
UNCERTIFIED_RETEST_LIMIT = "retest-limit"

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ON = "on"

# Contract COMPONENTS §2.2 (vNEXT): after every existing reason in precedence.
REASON_SCALE_OUTLIER = "final-scale-outlier"
REASON_SCALE_UNCERTIFIED = "final-scale-uncertified"
REASON_POSE_OUTLIER = "final-pose-outlier"
REASONS = (REASON_SCALE_OUTLIER, REASON_SCALE_UNCERTIFIED, REASON_POSE_OUTLIER)

# Plateau classes.
IN, LOW, HIGH = "in", "low", "high"

DECISION_PUBLISHED = "published"
DECISION_WITHHELD = "withheld"
DECISION_SHADOW = "shadow"

# F6 (review XR): the row's `finalization.notice` after a withhold. The solution published before -- the
# walk's background solve, never checked by this guard -- still stands, so the phone is told, in a closed
# sentence, that the room it shows is NOT CERTIFIED. One line, no slash, no figure (contract COMPONENTS 3.1).
NOTICE_WITHHELD = ("the final check could not certify this walk's room, so the room shown is the walk's "
                   "unchecked preview and its photos are kept; an owner can re-finish this walk")

WHY_WITHHELD = ("the final scale guard withheld the final solve ({}); nothing was published, and the "
                "solution published before stands")


@dataclass(frozen=True)
class GuardParams:
    """Every threshold, frozen before any replay (design §Decision thresholds). Recorded with its digest."""

    # global_solve.MIN_IMAGE_OBSERVATIONS: only published (supported) cameras are judged.
    min_obs: int = 30
    # The checker's and the contract's screen boundary, inclusive.
    screen_low: float = 0.8
    screen_high: float = 1.25
    # The clear-failure boundary: outside it a measured gross error; between it and the screen, borderline.
    clear_low: float = 0.75
    clear_high: float = 1.333
    # Ratios needed on each side of a scale comparison (the checker's MIN_SEGMENT_RATIOS).
    min_ratios: int = 10
    # Plateau detection: a centred running median over 2h+1 measured cameras.
    smooth_half_window: int = 5
    # The EXCURSION screen (review XR F2): inside a kept plateau, this many consecutive measured cameras each
    # beyond the CLEAR boundary on the same side are a short, grossly wrong stretch the running median
    # swallows. They are isolated `final-scale-uncertified` (too few ratios to certify either way). Three:
    # at the clear boundary a single camera's own ratio is out in a few per cent of cameras, three in a row
    # on one side in about one per 40 000 (independent cameras); a run of 6+ already moves the median.
    excursion_min_cameras: int = 3
    # Reference refinement rounds (it stops earlier when the decisions stop changing).
    max_reference_rounds: int = 6
    # v2: re-segment the rest after each removal, at most this many times.
    max_retest_rounds: int = 8
    # The deterministic bootstrap interval of a piece's log ratio to the core.
    bootstrap_draws: int = 400
    bootstrap_seed: int = 20261006
    ci_level: float = 0.95
    # The P4 motion rule and the checker's corroborating screens.
    speed_mps: float = 2.0
    speed_margin_m: float = 0.3
    hard_extra_m: float = 0.5
    hard_factor: float = 2.0
    step_max_dt_s: float = 2.0
    step_max_index_gap: int = 3
    height_jump_m: float = 0.75
    height_jump_dt_s: float = 1.0
    height_from_room_m: float = 1.5
    flip_deg: float = 120.0
    room_radius_m: float = 15.0

    def to_json(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        doc = {"guard": GUARD_ID, **self.to_json()}
        return hashlib.sha1(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def mode() -> str:
    """`TOWER_WORLD_FINAL_SCALE_GUARD` (`config.world_final_scale_guard_setting`): off, shadow or on.
    Reads the environment only."""
    from tower.config import world_final_scale_guard_setting  # noqa: PLC0415

    return world_final_scale_guard_setting()


# ---------------------------------------------------------------------------
# the pure assessment
# ---------------------------------------------------------------------------


@dataclass
class RoomCamera:
    """One supported room camera of the candidate, in capture order."""

    kid: str
    name: str            # the solve database's image name (the metric_log key)
    index: int           # position in the solution's keyframe ids (journal order)
    t: float | None      # the keyframe's receipt time, seconds
    R: np.ndarray        # world -> camera rotation (3, 3)
    tvec: np.ndarray     # world -> camera translation (3,)
    group: str           # the gate group it belongs to ("room" on an ungated solve)


def _finite_pose(c: RoomCamera) -> bool:
    return bool(np.isfinite(c.R).all() and np.isfinite(c.tvec).all())


def _center_up(c: RoomCamera) -> tuple[np.ndarray, np.ndarray]:
    center = -c.R.T @ c.tvec
    # The image's +Y is downward; minus the camera's +Y in the world is its "up" (the checker's proxy).
    up = -c.R.T @ np.array([0.0, 1.0, 0.0])
    n = float(np.linalg.norm(up))
    return center, (up / n if n > 0 else up)


def dominant_level(values: np.ndarray, params: GuardParams) -> float | None:
    """The densest window of log ratios exactly as wide as the screen band (ties: the lowest window), and
    the median of the values in it. None without values."""
    v = np.sort(values[np.isfinite(values)])
    if not len(v):
        return None
    width = math.log(params.screen_high) - math.log(params.screen_low)
    hi = np.searchsorted(v, v + width, side="right")
    counts = hi - np.arange(len(v))
    i = int(np.argmax(counts))
    return float(np.median(v[i:hi[i]]))


def classify(d: float, params: GuardParams) -> str:
    """The class of a log ratio to the reference: IN the inclusive screen band, LOW or HIGH outside it."""
    if d < math.log(params.screen_low):
        return LOW
    if d > math.log(params.screen_high):
        return HIGH
    return IN


def ratio_reason(d: float, params: GuardParams) -> str | None:
    """For a LOG ratio `d` to the reference: None inside the inclusive screen band; `final-scale-outlier`
    beyond the clear boundary; otherwise borderline, `final-scale-uncertified`. Compared in log, exactly as
    `classify` compares, so a boundary value is decided once."""
    if classify(d, params) == IN:
        return None
    if d < math.log(params.clear_low) or d > math.log(params.clear_high):
        return REASON_SCALE_OUTLIER
    return REASON_SCALE_UNCERTIFIED


def plateaus(values: np.ndarray, ref: float, params: GuardParams) -> list[tuple[int, int, str]]:
    """[(start, end_exclusive, class)] over one group's cameras in capture order: maximal runs of one class
    of the centred running median of the measured values. An unmeasured camera takes the class of the
    nearest measured camera before it (after it, at the start); a group with none is one `None` run."""
    n = len(values)
    measured = np.flatnonzero(np.isfinite(values))
    if not len(measured):
        return [(0, n, None)] if n else []
    y = values[measured]
    h = params.smooth_half_window
    cls_m = [classify(float(np.median(y[max(0, k - h):k + h + 1])) - ref, params) for k in range(len(y))]
    cls = [None] * n
    for k, i in enumerate(measured):
        cls[int(i)] = cls_m[k]
    first = cls_m[0]
    last = None
    for i in range(n):
        if cls[i] is None:
            cls[i] = last if last is not None else first
        last = cls[i]
    runs = []
    start = 0
    for i in range(1, n + 1):
        if i == n or cls[i] != cls[start]:
            runs.append((start, i, cls[start]))
            start = i
    return runs


def bootstrap_interval(piece: np.ndarray, core: np.ndarray, params: GuardParams, salt: int) -> list[float]:
    """The deterministic percentile interval of median(piece) - median(core) in log, as RATIOS."""
    rng = np.random.default_rng([params.bootstrap_seed, int(salt), len(piece), len(core)])
    b = params.bootstrap_draws
    ps = rng.choice(piece, size=(b, len(piece)), replace=True)
    cs = rng.choice(core, size=(b, len(core)), replace=True)
    d = np.median(ps, axis=1) - np.median(cs, axis=1)
    a = (1.0 - params.ci_level) / 2.0
    lo, hi = np.quantile(d, [a, 1.0 - a])
    return [round(math.exp(float(lo)), 4), round(math.exp(float(hi)), 4)]


def _scale_round(groups: dict, values: dict, ref: float, params: GuardParams) -> list[dict]:
    """One segmentation of every group against `ref`: the plateaus with their decisions."""
    out = []
    for gname, cams in groups.items():
        vals = np.asarray([values.get(c.name, np.nan) for c in cams], np.float64)
        runs = plateaus(vals, ref, params)
        decided = []
        for start, end, cls in runs:
            seg = vals[start:end]
            m = seg[np.isfinite(seg)]
            d = float(np.median(m)) - ref if len(m) else None
            entry = {"group": gname, "start": start, "end": end, "class": cls, "measured": int(len(m)),
                     "ratio": None if d is None else math.exp(d), "cameras": cams[start:end], "values": m}
            if len(m) >= params.min_ratios:
                reason = ratio_reason(d, params)
                entry["decision"] = "kept" if reason is None else "isolated"
                entry["reason"] = reason
                entry["certified"] = reason is None
            else:
                entry["decision"] = "short"
                entry["reason"] = None
                entry["certified"] = False
            decided.append(entry)
        # A stretch of short plateaus inherits only between two certified plateaus of this group, and only
        # when none of it sits off the band: a short run with a measured deviation is not certified either
        # way, and an unmeasured independent piece never gets a room claim.
        i = 0
        while i < len(decided):
            if decided[i]["decision"] != "short":
                i += 1
                continue
            j = i
            while j < len(decided) and decided[j]["decision"] == "short":
                j += 1
            bounded = (i > 0 and decided[i - 1]["certified"] and j < len(decided) and decided[j]["certified"]
                       and all(decided[k]["class"] in (IN, None) for k in range(i, j)))
            for k in range(i, j):
                decided[k]["decision"] = "inherited" if bounded else "isolated"
                decided[k]["reason"] = None if bounded else REASON_SCALE_UNCERTIFIED
            i = j
        for row in decided:
            out.extend(_split_excursions(row, values, ref, params))
    return out


def _excursions(cams: list, values: dict, ref: float, params: GuardParams) -> list[tuple[int, int]]:
    """[(start, end_exclusive)] positions in `cams` of the runs of >= `excursion_min_cameras` consecutive
    MEASURED cameras each beyond the clear boundary on one side (unmeasured cameras between them belong to
    the run; they do not break it)."""
    lo, hi = math.log(params.clear_low), math.log(params.clear_high)
    out = []
    run: list[int] = []
    side = 0

    def close():
        if len(run) >= params.excursion_min_cameras:
            out.append((run[0], run[-1] + 1))

    for pos, c in enumerate(cams):
        v = values.get(c.name)
        if v is None:
            continue
        d = v - ref
        sd = 1 if d > hi else (-1 if d < lo else 0)
        if sd != 0 and sd == side:
            run.append(pos)
            continue
        close()
        run, side = ([pos], sd) if sd != 0 else ([], 0)
    close()
    return out


def _split_excursions(row: dict, values: dict, ref: float, params: GuardParams) -> list[dict]:
    """A kept or inherited plateau with its excursions taken out as isolated rows of their own (the rest
    keeps the plateau's decision and certificate); any other row unchanged."""
    if row["decision"] not in ("kept", "inherited"):
        return [dict(row, kind="plateau")]
    cams = row["cameras"]
    spans = _excursions(cams, values, ref, params)
    if not spans:
        return [dict(row, kind="plateau")]
    parts = []
    cut = 0
    for a, b in spans:
        parts.append((cut, a, False))
        parts.append((a, b, True))
        cut = b
    parts.append((cut, len(cams), False))
    out = []
    for a, b, exc in parts:
        if a >= b:
            continue
        sub = cams[a:b]
        m = np.asarray([values[c.name] for c in sub if c.name in values], np.float64)
        d = float(np.median(m)) - ref if len(m) else None
        part = dict(row, start=row["start"] + a, end=row["start"] + b, cameras=sub, values=m,
                    measured=int(len(m)), ratio=None if d is None else math.exp(d),
                    kind="excursion" if exc else "plateau")
        if exc:
            reason = ratio_reason(d, params) if len(m) >= params.min_ratios else REASON_SCALE_UNCERTIFIED
            part.update({"decision": "isolated", "reason": reason, "certified": False,
                         "class": HIGH if d is not None and d > 0 else LOW})
        out.append(part)
    return out


def _signature(rows: list[dict]) -> tuple:
    return tuple((r["group"], r["start"], r["end"], r["decision"], r["reason"]) for r in rows)


def _pose_screen(cams: list[RoomCamera], mpu: float, params: GuardParams) -> dict:
    """The pose screen over the kept cameras (capture order). Returns the clusters, flags and warnings."""
    finite = [c for c in cams if _finite_pose(c)]
    nonfinite = [c for c in cams if not _finite_pose(c)]
    flags: list[dict] = []
    warnings: list[dict] = []
    if not finite:
        return {"room": [], "isolated": [[c] for c in nonfinite] if nonfinite else [], "flags": flags,
                "warnings": warnings, "nonfinite": len(nonfinite)}
    cu = {c.kid: _center_up(c) for c in finite}
    centers = np.stack([cu[c.kid][0] for c in finite])
    ups = np.stack([cu[c.kid][1] for c in finite])
    room_center = np.median(centers, axis=0)
    room_up = np.median(ups, axis=0)
    nu = float(np.linalg.norm(room_up))
    room_up = room_up / nu if nu > 0 else np.array([0.0, 1.0, 0.0])
    defects: dict[str, list[str]] = {}
    for c in finite:
        center, up = cu[c.kid]
        disp = (center - room_center) * mpu
        kinds = []
        vertical = float(disp @ room_up)
        if abs(vertical) > params.height_from_room_m:
            kinds.append("height-outlier")
        if float(np.linalg.norm(disp)) > params.room_radius_m:
            kinds.append("outside-room")
        if math.degrees(math.acos(float(np.clip(up @ room_up, -1.0, 1.0)))) > params.flip_deg:
            kinds.append("flip")
        if kinds:
            defects[c.kid] = kinds
    severe = {k for k, v in defects.items() if len(v) >= 2}
    for k in sorted(severe):
        flags.append({"kind": "multi-defect", "id": k, "defects": defects[k]})

    def allowance(dt):
        return params.speed_mps * dt + params.speed_margin_m

    def step(a: RoomCamera, b: RoomCamera) -> dict | None:
        """The step a -> b, or None when it is not testable: more than `step_max_index_gap` journal indices or
        `step_max_dt_s` apart (or no time). The same bounds for a cut and for a rejoin (review XR F3): elapsed
        time alone never certifies where an island sits relative to the room."""
        if a.t is None or b.t is None:
            return None
        dt = float(b.t - a.t)
        if b.index - a.index > params.step_max_index_gap or not 0 < dt <= params.step_max_dt_s:
            return None
        delta = (cu[b.kid][0] - cu[a.kid][0]) * mpu
        dist = float(np.linalg.norm(delta))
        height = abs(float(delta @ room_up))
        allow = allowance(dt)
        hard_limit = max(allow + params.hard_extra_m, params.hard_factor * allow)
        speed = dist > allow
        jump = dt <= params.height_jump_dt_s and height > params.height_jump_m
        corroborated = jump or a.kid in defects or b.kid in defects
        hard = dist > hard_limit or (speed and corroborated)
        return {"a": a, "b": b, "dt": dt, "dist": dist, "height": height, "allow": allow,
                "hard_limit": hard_limit, "speed": speed, "jump": jump, "hard": hard}

    seq = [c for c in finite if c.kid not in severe]
    runs: list[list[RoomCamera]] = []
    for c in seq:
        if not runs:
            runs.append([c])
            continue
        s = step(runs[-1][-1], c)
        if s is not None and s["hard"]:
            flags.append({"kind": "hard-step", "indices": [s["a"].index, s["b"].index], "dt_s": round(s["dt"], 3),
                          "metres": round(s["dist"], 3), "limit_m": round(s["hard_limit"], 3),
                          "height_jump": s["jump"]})
            runs.append([c])
            continue
        if s is not None and (s["speed"] or s["jump"]):
            warnings.append({"kind": "speed" if s["speed"] else "height-jump",
                             "indices": [s["a"].index, s["b"].index], "dt_s": round(s["dt"], 3),
                             "metres": round(s["dist"], 3), "allowance_m": round(s["allow"], 3)})
        runs[-1].append(c)
    # Rejoin: a run joins the earliest-formed cluster whose latest camera it follows within the step bounds and
    # without a hard step. An untestable gap never rejoins: the run stays its own cluster.
    clusters: list[list[list[RoomCamera]]] = []
    for run in runs:
        joined = False
        for cl in clusters:
            s = step(cl[-1][-1], run[0])
            if s is not None and not s["hard"]:
                cl.append(run)
                joined = True
                break
        if not joined:
            clusters.append([run])
    sizes = [sum(len(r) for r in cl) for cl in clusters]
    core = int(np.argmax(sizes)) if clusters else None
    room = [c for r in clusters[core] for c in r] if core is not None else []
    isolated = [[c for r in cl for c in r] for i, cl in enumerate(clusters) if i != core]
    # Severe and non-finite cameras: contiguous (in capture order) runs form one piece each.
    quarantined = sorted([c for c in finite if c.kid in severe] + nonfinite, key=lambda c: c.index)
    order = {c.kid: i for i, c in enumerate(cams)}
    piece: list[RoomCamera] = []
    for c in quarantined:
        if piece and order[c.kid] != order[piece[-1].kid] + 1:
            isolated.append(piece)
            piece = []
        piece.append(c)
    if piece:
        isolated.append(piece)
    return {"room": room, "isolated": isolated, "flags": flags, "warnings": warnings,
            "nonfinite": len(nonfinite), "defects": {k: v for k, v in sorted(defects.items())}}


def assess(cameras: list[RoomCamera], metric_log: dict, params: GuardParams | None = None) -> dict:
    """The guard's decision on one candidate room. Pure: no IO.

    `cameras`: the room's supported cameras in capture order; `metric_log`: {image name: log ratio}.
    Returns {"decision": "publish" | "withhold", "why", "reference", "rounds", "plateaus", "pieces",
    "pose", "room_keyframes", ...}; `pieces` are the isolated pieces, each {"keyframe_ids", "reason",
    "basis", ...}, in capture order of their first camera."""
    params = params or GuardParams()
    values = {c.name: float(v) for c in cameras for v in [metric_log.get(c.name)]
              if v is not None and math.isfinite(v)}
    measured = np.asarray([values[c.name] for c in cameras if c.name in values], np.float64)
    base = {"guard": GUARD_ID_V1, "params": params.to_json(), "params_digest": params.digest(),
            "room_candidate": len(cameras), "room_measured": int(len(measured))}
    if len(measured) < params.min_ratios:
        return dict(base, decision="withhold", pieces=[], plateaus=[], rounds=[], reference=None, pose=None,
                    room_keyframes=[],
                    why=f"the room has {len(measured)} cameras with a metric ratio; {params.min_ratios} "
                        "are needed to certify it")
    groups: dict[str, list[RoomCamera]] = {}
    for c in cameras:
        groups.setdefault(c.group, []).append(c)
    ref = dominant_level(measured, params)
    rounds = [{"reference_log": ref, "basis": "densest-window"}]
    rows = _scale_round(groups, values, ref, params)
    seen = {_signature(rows)}
    for _ in range(params.max_reference_rounds):
        core_vals = np.concatenate([r["values"] for r in rows if r["certified"]] or [np.zeros(0)])
        if len(core_vals) < params.min_ratios:
            break
        new_ref = float(np.median(core_vals))
        if abs(new_ref - ref) < 1e-12:
            break
        new_rows = _scale_round(groups, values, new_ref, params)
        sig = _signature(new_rows)
        rounds.append({"reference_log": new_ref, "basis": "certified-core-median"})
        ref, rows = new_ref, new_rows
        if sig in seen:
            break
        seen.add(sig)
    core_vals = np.concatenate([r["values"] for r in rows if r["certified"]] or [np.zeros(0)])
    reference = {"log": ref, "core_measured": int(len(core_vals)), "rounds": len(rounds)}
    plateau_rows = []
    pieces = []
    for r in rows:
        ids = [c.kid for c in r["cameras"]]
        row = {"group": r["group"], "first_index": r["cameras"][0].index, "last_index": r["cameras"][-1].index,
               "keyframes": len(ids), "measured": r["measured"], "class": r["class"], "kind": r.get("kind"),
               "ratio": None if r["ratio"] is None else round(r["ratio"], 4),
               "decision": r["decision"], "reason": r["reason"]}
        if r["measured"] >= params.min_ratios and len(core_vals) and r["decision"] == "isolated":
            row["ci"] = bootstrap_interval(r["values"], core_vals, params, salt=row["first_index"])
            row["ci_excludes_one"] = bool(row["ci"][0] > 1.0 or row["ci"][1] < 1.0)
        plateau_rows.append(row)
        if r["decision"] == "isolated":
            pieces.append(dict(row, basis="excursion" if r.get("kind") == "excursion" else "scale",
                               keyframe_ids=ids))
    if len(core_vals) < params.min_ratios:
        return dict(base, decision="withhold", pieces=pieces, plateaus=plateau_rows, rounds=rounds,
                    reference=reference, pose=None, room_keyframes=[],
                    why=f"the certified room core has {len(core_vals)} cameras with a metric ratio; "
                        f"{params.min_ratios} are needed")
    kept = [c for r in rows if r["decision"] in ("kept", "inherited") for c in r["cameras"]]
    kept.sort(key=lambda c: c.index)
    mpu = math.exp(-ref)
    pose = _pose_screen(kept, mpu, params)
    for cams in pose["isolated"]:
        m = [values[c.name] for c in cams if c.name in values]
        pieces.append({"group": cams[0].group, "first_index": cams[0].index, "last_index": cams[-1].index,
                       "keyframes": len(cams), "measured": len(m),
                       "ratio": round(math.exp(float(np.median(m)) - ref), 4) if m else None,
                       "decision": "isolated", "reason": REASON_POSE_OUTLIER, "basis": "pose",
                       "keyframe_ids": [c.kid for c in cams]})
    pieces.sort(key=lambda p: (p["first_index"], p["basis"]))
    room = pose["room"]
    room_measured = sum(1 for c in room if c.name in values)
    pose_record = {"metres_per_unit": mpu, "room_poses": len(room), "flags": pose["flags"],
                   "warnings": pose["warnings"], "nonfinite": pose["nonfinite"],
                   "defects": pose.get("defects", {}),
                   "flag_counts": _counts(pose["flags"]), "warning_counts": _counts(pose["warnings"])}
    out = dict(base, reference=reference, rounds=rounds, plateaus=plateau_rows, pieces=pieces, pose=pose_record,
               room_keyframes=[c.kid for c in room], room_certified_measured=room_measured)
    if room_measured < params.min_ratios:
        return dict(out, decision="withhold",
                    why=f"after the pose screen the room has {room_measured} cameras with a metric ratio; "
                        f"{params.min_ratios} are needed")
    return dict(out, decision="publish", why=None)


def _counts(items: list[dict]) -> dict:
    out: dict = {}
    for f in items:
        out[f["kind"]] = out.get(f["kind"], 0) + 1
    return out


# ---------------------------------------------------------------------------
# v2: the rule the publish seam runs (design RUN review/codex/cx-SCALE-GUARD-V2-DESIGN-20261007.md)
# ---------------------------------------------------------------------------
#
# Isolate a plateau ONLY when all three hold: >= `min_ratios` finite ratios; the median ratio to the
# reference outside the inclusive [screen_low, screen_high]; and the deterministic 95 % bootstrap interval
# of its median against the certified core (`bootstrap_draws`, `bootstrap_seed`) excludes 1. A plateau is
# bounded by its FIRST and LAST MEASURED camera: unmeasured cameras inside it go with it, an unmeasured
# head, tail or gap never does (v1 carried a class across an arbitrarily long unmeasured tail). Everything
# else -- short plateaus, out-of-band plateaus whose interval includes 1, excursions, unmeasured stretches --
# STAYS ATTACHED and is recorded `uncertified`, in the audit and on the placed component. No evidence: the
# whole room is `uncertified`, nothing is isolated, nothing is withheld. No pose-only quarantine: the pose
# screen is recorded for audit and applied to nothing. After each removal the rest is re-segmented.


def measured_plateaus(values: np.ndarray, ref: float, params: GuardParams) -> list[tuple[int, int, str]]:
    """[(start, end_exclusive, class)]: maximal runs of one class of the centred running median of the
    MEASURED values, each from its first to its last measured camera (positions in `values`). Positions
    before the first run, after the last, or between two runs belong to none (the caller's unmeasured
    stretches)."""
    measured = np.flatnonzero(np.isfinite(values))
    if not len(measured):
        return []
    y = values[measured]
    h = params.smooth_half_window
    cls_m = [classify(float(np.median(y[max(0, k - h):k + h + 1])) - ref, params) for k in range(len(y))]
    runs = []
    k0 = 0
    for k in range(1, len(measured) + 1):
        if k == len(measured) or cls_m[k] != cls_m[k0]:
            runs.append((int(measured[k0]), int(measured[k - 1]) + 1, cls_m[k0]))
            k0 = k
    return runs


def _gap_row(group: str, cams: list, start: int, end: int) -> dict:
    return {"group": group, "start": start, "end": end, "class": None, "measured": 0, "ratio": None,
            "cameras": cams[start:end], "values": np.zeros(0), "kind": "unmeasured", "decision": "attached",
            "reason": None, "certified": False, "uncertified": UNCERTIFIED_UNMEASURED}


def _v2_rows(groups: dict, values: dict, ref: float, params: GuardParams) -> list[dict]:
    """One segmentation of every group against `ref`: measured-bounded plateaus and unmeasured stretches,
    each with its provisional v2 decision (`kept`, `candidate` or `attached`)."""
    out = []
    for gname, cams in groups.items():
        vals = np.asarray([values.get(c.name, np.nan) for c in cams], np.float64)
        rows = []
        covered = 0
        for start, end, cls in measured_plateaus(vals, ref, params):
            if start > covered:
                rows.append(_gap_row(gname, cams, covered, start))
            seg = vals[start:end]
            m = seg[np.isfinite(seg)]
            d = float(np.median(m)) - ref
            row = {"group": gname, "start": start, "end": end, "class": cls, "measured": int(len(m)),
                   "ratio": math.exp(d), "cameras": cams[start:end], "values": m, "kind": "plateau"}
            if len(m) < params.min_ratios:
                row.update(decision="attached", reason=None, certified=False, uncertified=UNCERTIFIED_SHORT)
            elif classify(d, params) == IN:
                row.update(decision="kept", reason=None, certified=True, uncertified=None)
            else:
                row.update(decision="candidate", reason=None, certified=False, uncertified=None)
            rows.append(row)
            covered = end
        if covered < len(cams):
            rows.append(_gap_row(gname, cams, covered, len(cams)))
        for row in rows:
            if row["decision"] != "kept":
                out.append(row)
                continue
            # A short gross excursion the running median swallows stays attached, flagged (never isolated).
            for part in _split_excursions(row, values, ref, params):
                if part["kind"] == "excursion":
                    part.update(decision="attached", reason=None, certified=False,
                                uncertified=UNCERTIFIED_EXCURSION)
                out.append(part)
    return out


def _v2_segment(groups: dict, values: dict, params: GuardParams):
    """(reference log or None, rows, reference rounds, certified core values) for `groups`, the reference
    refined from the certified core exactly as v1 refines it; every out-of-band >= 10-ratio plateau then
    decided by its bootstrap interval against that core."""
    measured = np.asarray([values[c.name] for cams in groups.values() for c in cams if c.name in values],
                          np.float64)
    ref = dominant_level(measured, params)
    if ref is None:
        return None, _v2_rows(groups, values, 0.0, params), [], np.zeros(0)
    rounds = [{"reference_log": ref, "basis": "densest-window"}]
    rows = _v2_rows(groups, values, ref, params)
    seen = {_signature(rows)}
    for _ in range(params.max_reference_rounds):
        core = np.concatenate([r["values"] for r in rows if r["certified"]] or [np.zeros(0)])
        if len(core) < params.min_ratios:
            break
        new_ref = float(np.median(core))
        if abs(new_ref - ref) < 1e-12:
            break
        new_rows = _v2_rows(groups, values, new_ref, params)
        sig = _signature(new_rows)
        rounds.append({"reference_log": new_ref, "basis": "certified-core-median"})
        ref, rows = new_ref, new_rows
        if sig in seen:
            break
        seen.add(sig)
    core = np.concatenate([r["values"] for r in rows if r["certified"]] or [np.zeros(0)])
    for r in rows:
        if r["decision"] != "candidate":
            continue
        if len(core) < params.min_ratios:
            r.update(decision="attached", uncertified=UNCERTIFIED_NO_CORE)
            continue
        r["ci"] = bootstrap_interval(r["values"], core, params, salt=r["cameras"][0].index)
        r["ci_excludes_one"] = bool(r["ci"][0] > 1.0 or r["ci"][1] < 1.0)
        if r["ci_excludes_one"]:
            r.update(decision="isolated", reason=REASON_SCALE_OUTLIER)
        else:
            r.update(decision="attached", uncertified=UNCERTIFIED_CI)
    return ref, rows, rounds, core


def _v2_row_record(r: dict) -> dict:
    """A row as the audit records it: frozen keyframe ids, counts, ratio and interval (Tower-internal)."""
    cams = r["cameras"]
    out = {"group": r["group"], "first_index": cams[0].index, "last_index": cams[-1].index,
           "first_keyframe_id": cams[0].kid, "last_keyframe_id": cams[-1].kid, "keyframes": len(cams),
           "measured": r["measured"], "class": r["class"], "kind": r.get("kind"),
           "ratio": None if r["ratio"] is None else round(r["ratio"], 4),
           "decision": r["decision"], "reason": r["reason"], "uncertified": r.get("uncertified")}
    if "ci" in r:
        out["ci"] = r["ci"]
        out["ci_excludes_one"] = r["ci_excludes_one"]
    return out


def assess_v2(cameras: list[RoomCamera], metric_log: dict, params: GuardParams | None = None) -> dict:
    """The v2 decision on one candidate room. Pure: no IO. Always `decision: publish` (v2 never withholds
    on scale); `pieces` are only the qualifying isolated plateaus (`final-scale-outlier`), `uncertified`
    every attached stretch the evidence does not certify, `scale_status` certified or uncertified."""
    params = params or GuardParams()
    values = {c.name: float(v) for c in cameras for v in [metric_log.get(c.name)]
              if v is not None and math.isfinite(v)}
    base = {"guard": GUARD_ID, "rule": "v2", "params": params.to_json(), "params_digest": params.digest(),
            "room_candidate": len(cameras), "room_measured": len(values), "decision": "publish", "why": None}
    if not values:
        span = []
        if cameras:
            span = [dict(_v2_row_record(dict(_gap_row("room", cameras, 0, len(cameras)),
                                             uncertified=UNCERTIFIED_NO_EVIDENCE)),
                         keyframe_ids=[c.kid for c in cameras])]
        return dict(base, scale_status=SCALE_UNCERTIFIED, pieces=[], plateaus=[], rounds=[], retest=[],
                    reference=None, pose=None, room_keyframes=[c.kid for c in cameras], uncertified=span,
                    uncertified_keyframes=len(cameras),
                    why="no metric evidence (no applied gate scale, no compatible live depth cache): the room "
                        "is published uncertified and nothing is isolated")
    groups: dict[str, list[RoomCamera]] = {}
    for c in cameras:
        groups.setdefault(c.group, []).append(c)
    pieces: list[dict] = []
    retest: list[dict] = []
    for rnd in range(params.max_retest_rounds + 1):
        ref, rows, rounds, core = _v2_segment(groups, values, params)
        isolated = [r for r in rows if r["decision"] == "isolated"]
        if isolated and rnd == params.max_retest_rounds:
            for r in isolated:
                r.update(decision="attached", reason=None, uncertified=UNCERTIFIED_RETEST_LIMIT)
            isolated = []
        retest.append({"round": rnd, "reference_log": ref, "reference_rounds": len(rounds),
                       "core_measured": int(len(core)), "isolated": len(isolated)})
        if not isolated:
            break
        for r in isolated:
            pieces.append(dict(_v2_row_record(r), basis="scale", round=rnd,
                               keyframe_ids=[c.kid for c in r["cameras"]]))
        gone = {c.kid for r in isolated for c in r["cameras"]}
        groups = {g: [c for c in cams if c.kid not in gone] for g, cams in groups.items()}
        groups = {g: cams for g, cams in groups.items() if cams}
    pieces.sort(key=lambda p: p["first_index"])
    attached = [r for r in rows if r["decision"] == "attached"]
    uncertified = [dict(_v2_row_record(r), keyframe_ids=[c.kid for c in r["cameras"]]) for r in attached]
    room = sorted((c for r in rows for c in r["cameras"]), key=lambda c: c.index)
    status = SCALE_CERTIFIED if (not uncertified and len(core) >= params.min_ratios) else SCALE_UNCERTIFIED
    pose = None
    if ref is not None and room:
        # Recorded for audit only: v2 isolates nothing on pose (design §3).
        screen = _pose_screen(room, math.exp(-ref), params)
        pose = {"applied": False, "metres_per_unit": math.exp(-ref), "flags": screen["flags"],
                "warnings": screen["warnings"], "nonfinite": screen["nonfinite"],
                "defects": screen.get("defects", {}), "flag_counts": _counts(screen["flags"]),
                "warning_counts": _counts(screen["warnings"]),
                "would_isolate_keyframes": sum(len(p) for p in screen["isolated"])}
    why = None
    if status == SCALE_UNCERTIFIED:
        why = (f"{sum(len(u['keyframe_ids']) for u in uncertified)} attached keyframes are not certified"
               if uncertified else f"the certified room core has {len(core)} cameras with a metric ratio; "
                                   f"{params.min_ratios} are needed")
    return dict(base, scale_status=status, why=why,
                reference=None if ref is None else {"log": ref, "core_measured": int(len(core)),
                                                    "rounds": len(rounds)},
                rounds=rounds, retest=retest, plateaus=[_v2_row_record(r) for r in rows], pieces=pieces,
                pose=pose, room_keyframes=[c.kid for c in room], uncertified=uncertified,
                uncertified_keyframes=sum(len(u["keyframe_ids"]) for u in uncertified))


# ---------------------------------------------------------------------------
# the candidate, as the guard sees it
# ---------------------------------------------------------------------------


def room_cameras(solution, keyframes, name_of: dict, *, groups_of: dict | None = None,
                 min_obs: int = 30) -> list[RoomCamera]:
    """The candidate's supported component-0 cameras, in the solution's keyframe order. `groups_of`
    {image name: group} names the gate's groups; None = one group."""
    received = {k.keyframe_id: float(k.received_at) for k in keyframes}
    out = []
    for i, kid in enumerate(solution.keyframe_ids):
        p = (solution.poses or {}).get(kid)
        if not p or int(p.get("component", 0)) != 0 or int(p.get("observations", 0)) < min_obs:
            continue
        # A supported room pose with a missing or malformed rotation or translation, or no image name, is
        # NOT skipped (review XR F4): skipped, it stayed in the published room unassessed. It enters as a
        # non-finite camera, which the pose screen quarantines (`final-pose-outlier`).
        R = _array_or_nan(p.get("rotation"), (3, 3))
        tvec = _array_or_nan(p.get("translation"), (3,))
        name = name_of.get(kid)
        if name is None:
            R = np.full((3, 3), np.nan)
            name = kid
        out.append(RoomCamera(
            kid=kid, name=name, index=i, t=received.get(kid), R=R, tvec=tvec,
            group="room" if groups_of is None else groups_of.get(name, "room")))
    # A supported room pose the keyframe list does not name: no capture position, so never certifiable.
    listed = set(solution.keyframe_ids)
    for kid, p in sorted((solution.poses or {}).items()):
        if kid in listed or int(p.get("component", 0)) != 0 or int(p.get("observations", 0)) < min_obs:
            continue
        out.append(RoomCamera(kid=kid, name=name_of.get(kid, kid), index=len(listed) + len(out),
                              t=received.get(kid), R=np.full((3, 3), np.nan), tvec=np.full(3, np.nan),
                              group="room"))
    return out


def _array_or_nan(value, shape) -> np.ndarray:
    try:
        return np.asarray(value, np.float64).reshape(shape)
    except (TypeError, ValueError):
        return np.full(shape, np.nan)


def gate_groups(gated: dict | None) -> dict | None:
    """{image name: group id} for the room's gate groups (label 0), or None without a gate output."""
    if not gated:
        return None
    out = {}
    for g in gated.get("groups") or []:
        if g.get("label") != 0:
            continue
        gid = ("anchor:" if g.get("reference") else "group:") + str(g.get("first_camera"))
        for nm in g.get("members") or []:
            out[nm] = gid
    return out


# ---------------------------------------------------------------------------
# applying a decision
# ---------------------------------------------------------------------------


def _existing_labels(solution, min_obs: int) -> tuple[dict, list[dict]]:
    """{kid: label} for every posed keyframe and the gate-style component list of the solution."""
    labels = {kid: int(p.get("component", 0)) for kid, p in (solution.poses or {}).items()}
    comps = []
    for c in solution.components or []:
        lab = int(c.get("index", 0))
        state = c.get("gate_state") or ("placed" if lab == 0 else "unplaced")
        reasons = list(c.get("gate_reasons") or ([] if lab == 0 else ["solved-separately"]))
        comps.append({"label": lab, "state": state, "reasons": [] if state == "placed" else reasons,
                      "source_components": list(c.get("solver_components") or [lab])})
    known = {c["label"] for c in comps}
    for lab in sorted(set(labels.values()) - known):
        comps.append({"label": lab, "state": "placed" if lab == 0 else "unplaced",
                      "reasons": [] if lab == 0 else ["solved-separately"], "source_components": [lab]})
    return labels, comps


def apply(solution, assessment: dict, *, session_id: str, keyframes, started_at: float,
          gate_block: dict | None, min_obs: int = 30):
    """(the relabelled solution, its components record) with every isolated piece its own unplaced
    component. Raises ValueError when the record cannot hold the contract's shape (no room left)."""
    from tower.world_builder import coherence_publish as CP  # noqa: PLC0415

    labels, comps = _existing_labels(solution, min_obs)
    nxt = max(labels.values(), default=0) + 1
    for piece in assessment["pieces"]:
        for kid in piece["keyframe_ids"]:
            labels[kid] = nxt
        comps.append({"label": nxt, "state": "unplaced", "reasons": [piece["reason"]], "source_components": [0]})
        nxt += 1
    relabelled = CP.relabel_solution(solution, labels, comps, min_obs=min_obs)
    gate = dict(gate_block or {}, components=comps)
    gate.setdefault("gate", None)
    gate.setdefault("params", None)
    gate.setdefault("params_digest", None)
    gate.setdefault("masks_applied", (solution.transients or {}).get("state") == "applied")
    gate.setdefault("metric_available", True)
    doc = CP.components_document(session_id, relabelled, labels, gate, keyframes, started_at, min_obs=min_obs)
    if doc is None:
        raise ValueError("the relabelled candidate has no published room")
    if assessment.get("rule") == "v2":
        placed = next(e for e in doc["components"] if e["state"] == CP.CG_PLACED)
        placed["final_scale_guard"] = placed_scale_status(assessment, placed["keyframe_ids"])
    return relabelled, doc


def placed_scale_status(assessment: dict, placed_ids) -> dict:
    """v2's versioned, Tower-internal scale status of the PLACED component (components.json): the status and
    every attached uncertified stretch as frozen keyframe-id spans with counts. No ratio, no metric factor
    (the phone's reader drops unknown keys; contract §2.4 rule 6)."""
    placed = set(placed_ids)
    spans = []
    for u in assessment.get("uncertified") or []:
        ids = [k for k in u["keyframe_ids"] if k in placed]
        if ids:
            spans.append({"first_keyframe_id": ids[0], "last_keyframe_id": ids[-1], "keyframes": len(ids),
                          "kind": u["uncertified"]})
    return {"id": GUARD_ID, "scale_status": assessment.get("scale_status") or SCALE_UNCERTIFIED,
            "evidence": assessment.get("evidence"),
            "uncertified_keyframes": sum(s["keyframes"] for s in spans), "uncertified_spans": spans}


def summary(assessment: dict | None, *, mode_: str, decision: str, seconds: float, error: str | None = None
            ) -> dict:
    """What the solution (`solve.final_scale_guard`) and the gate record say: no ratio, no figure."""
    pieces = (assessment or {}).get("pieces") or []
    reasons: dict = {}
    for p in pieces:
        reasons[p["reason"]] = reasons.get(p["reason"], 0) + 1
    out = {"id": GUARD_ID, "mode": mode_, "decision": decision,
           "params_digest": (assessment or {}).get("params_digest") or GuardParams().digest(),
           "pieces_isolated": len(pieces), "keyframes_isolated": sum(len(p["keyframe_ids"]) for p in pieces),
           "reasons": reasons, "audit": AUDIT_FILENAME, "seconds": round(seconds, 3)}
    if (assessment or {}).get("rule") == "v2":
        out["scale_status"] = assessment["scale_status"]
        out["uncertified_keyframes"] = assessment["uncertified_keyframes"]
        out["evidence"] = assessment.get("evidence")
    if decision == DECISION_WITHHELD:
        out["why"] = (assessment or {}).get("why") or error
    if error:
        out["error"] = error
    return out


def audit_document(assessment: dict | None, *, mode_: str, decision: str, solution, inputs: dict,
                   seconds: float, error: str | None = None) -> dict:
    from tower.world_builder.global_solve import solve_identity  # noqa: PLC0415

    doc = {"record": RECORD, "guard": GUARD_ID, "mode": mode_, "decision": decision,
           "input_digest": getattr(solution, "input_digest", None),
           "solve_identity": solve_identity(solution) if solution is not None else None,
           "solved_at": getattr(solution, "solved_at", None),
           "inputs": inputs, "seconds": round(seconds, 3), "error": error}
    doc["assessment"] = _jsonable(assessment)
    return doc


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, RoomCamera):
        return v.kid
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def write_audit(workspace_root, document: dict) -> Path:
    from tower.storage import write_json_atomic  # noqa: PLC0415

    path = Path(workspace_root) / AUDIT_FILENAME
    write_json_atomic(path, document)
    return path


# ---------------------------------------------------------------------------
# the publication step
# ---------------------------------------------------------------------------


@dataclass
class GuardOutcome:
    mode: str
    decision: str                  # published | withheld | shadow
    solution: object               # what to publish (None when withheld)
    result: object                 # the GateResult to hand `after_publish` (None: today's ungated publish)
    summary: dict
    audit: dict
    depth: dict | None = None      # the guard's own depth stage (an ungated solve), for the surface


def read_evidence(store, world_id, session_id, solution, result, *, database_path, name_of,
                  live_metric_fn=None) -> tuple[dict, dict]:
    """(metric_log, inputs record) from EXISTING evidence only (v2; no depth is predicted at publish): the
    applied gate's `scale.metric_log` first; else the session's live depth cache, every record of it whose
    FULL identity matches this solve (`live_depth.compatible_live_records`), read in place; else `{}`
    (the room is then published `uncertified`)."""
    from tower.world_builder import coherence_publish as CP  # noqa: PLC0415

    db = Path(str(database_path)).name
    gate_state = None
    if result is not None:
        record = result.record or {}
        gate_state = record.get("state")
        scale = result.scale or {}
        metric_log = dict(scale.get("metric_log") or {})
        if gate_state == CP.GATE_STATE_APPLIED and metric_log:
            return metric_log, {"source": EVIDENCE_GATE, "scale": "the gate's", "database": db,
                                "depth": (record.get("depth") or {}).get("predictions"),
                                "cameras_measured": scale.get("cameras_measured")}
    metric_log, live = _live_cache_evidence(store, world_id, session_id, solution, database_path=database_path,
                                            name_of=name_of, live_metric_fn=live_metric_fn)
    if metric_log:
        return metric_log, {"source": EVIDENCE_LIVE, "scale": "the live depth cache", "database": db,
                            "gate_state": gate_state, "live": live}
    return {}, {"source": EVIDENCE_NONE, "scale": None, "database": db, "gate_state": gate_state, "live": live}


def _live_cache_evidence(store, world_id, session_id, solution, *, database_path, name_of,
                         live_metric_fn=None) -> tuple[dict, dict]:
    """(metric_log, record) over the compatible live depth cache, read in place; ({}, why) without one."""
    from tower.world_builder import live_depth as LD  # noqa: PLC0415
    from tower.world_builder.coherence_publish import _depth_params  # noqa: PLC0415
    from tower.world_builder.dense_pipeline import dense_dir, depth_trust_now  # noqa: PLC0415

    if LD.mode() != "on":
        # Exp1 design §5 / prereg arm L: a shadow (or off) live cache measures the worker's cost only;
        # it is never evidence for the guard (lead gate, 2026-10-08).
        return {}, {"state": "live-depth-not-on", "mode": LD.mode()}
    root = dense_dir(store, world_id, session_id)
    align = root / "live_depth" / "align.json"
    if not align.is_file():
        return {}, {"state": "absent"}
    imagery = _depth_params().imagery_source
    trust = depth_trust_now(store, world_id, session_id, imagery_source=imagery)
    session = store.read_session(world_id, session_id)
    fingerprint = hashlib.sha1(align.read_bytes()).hexdigest()[:16]
    matches = LD.compatible_live_records(store, world_id, session_id, solution, session.intrinsics, root,
                                         imagery_source=imagery, trust=trust)
    ki_of_name = {name_of[kid]: old_ki for _ki, (kid, _sha1, old_ki) in sorted(matches.items())
                  if kid in name_of}
    record = {"state": "compatible" if ki_of_name else "incompatible", "align_sha1": fingerprint,
              "backend": LD.MODEL, "imagery_source": imagery, "trust": trust, "records_matched": len(ki_of_name)}
    if not ki_of_name:
        return {}, record
    scale = (live_metric_fn or _live_metric_scale)(solution, name_of, database_path, root / "live_depth" / "work",
                                                   ki_of_name)
    record["cameras_measured"] = scale.get("cameras_measured")
    return dict(scale.get("metric_log") or {}), record


def _live_metric_scale(solution, name_of: dict, database_path, live_work, ki_of_name: dict) -> dict:
    """`coherence_scale.metric_scale` with the live predictions read in place (`live_work/depth/<live ki>`):
    CPU only, the final solve's own verified pairs and poses."""
    from tower.world_builder import coherence_scale as CS  # noqa: PLC0415

    cameras = CS.cameras_from_solution(solution, name_of.__getitem__)
    sampler = CS.DepthStageSampler(live_work, ki_of_name)
    out = CS.metric_scale(cameras, CS.read_inlier_pairs(database_path), solution.camera, sampler)
    out["frames_without_prediction"] = len(sampler.missing)
    return out


def guard_publication(store, world_id: str, session_id: str, solution, result, *, mode_: str, database_path,
                      keyframes, params: GuardParams | None = None, live_metric_fn=None) -> GuardOutcome:
    """The guard (v2) at the publish seam. `solution`: what would be published; `result`: its GateResult,
    or None on an ungated solve. Reads existing evidence only (`read_evidence`): it predicts no depth. Never
    withholds on scale: without evidence the room is published `uncertified`. Never raises: in `on`, a
    failure (an exception, a relabel that leaves no room) withholds; in `shadow`, it is recorded and the
    candidate is published unchanged."""
    from tower.world_builder import coherence_publish as CP  # noqa: PLC0415
    from tower.world_builder.coherence_publish import GateResult  # noqa: PLC0415

    params = params or GuardParams()
    started = time.perf_counter()
    assessment = None
    inputs: dict = {}
    depth = None
    try:
        name_of = CP._image_names(keyframes)
        metric_log, inputs = read_evidence(store, world_id, session_id, solution, result,
                                           database_path=database_path, name_of=name_of,
                                           live_metric_fn=live_metric_fn)
        groups_of = gate_groups(result.gated) if result is not None else None
        cams = room_cameras(solution, keyframes, name_of, groups_of=groups_of, min_obs=params.min_obs)
        assessment = dict(assess_v2(cams, metric_log, params), evidence=inputs["source"])
        if mode_ == MODE_SHADOW:
            secs = time.perf_counter() - started
            return GuardOutcome(mode_, DECISION_SHADOW, solution, result,
                                summary(assessment, mode_=mode_, decision=DECISION_SHADOW, seconds=secs),
                                audit_document(assessment, mode_=mode_, decision=DECISION_SHADOW,
                                               solution=solution, inputs=inputs, seconds=secs), depth)
        if assessment["decision"] != "publish":
            raise _Withhold(assessment["why"])
        session = store.read_session(world_id, session_id)
        gate_block = None
        if result is not None and result.components is not None:
            g = result.components.get("gate") or {}
            gate_block = {"gate": g.get("id"), "params": g.get("params"), "params_digest": g.get("params_digest"),
                          "masks_applied": g.get("masks_applied"), "metric_available": g.get("metric_available")}
        published, doc = apply(solution, assessment, session_id=session_id, keyframes=keyframes,
                               started_at=session.started_at, gate_block=gate_block, min_obs=params.min_obs)
        secs = time.perf_counter() - started
        summ = summary(assessment, mode_=mode_, decision=DECISION_PUBLISHED, seconds=secs)
        doc["final_scale_guard"] = {"id": GUARD_ID, "params_digest": params.digest(),
                                    "pieces_isolated": summ["pieces_isolated"], "audit": AUDIT_FILENAME,
                                    "scale_status": summ["scale_status"]}
        published = dataclasses.replace(published, solve=dict(solution.solve or {}, final_scale_guard=summ))
        if result is not None:
            new_result = dataclasses.replace(result, solution=published, components=doc)
        else:
            new_result = GateResult(solution=published, record={}, components=doc, depth=depth)
        return GuardOutcome(mode_, DECISION_PUBLISHED, published, new_result, summ,
                            audit_document(assessment, mode_=mode_, decision=DECISION_PUBLISHED,
                                           solution=published, inputs=inputs, seconds=secs), depth)
    except Exception as exc:  # noqa: BLE001 -- ON is fail-closed; shadow publishes unchanged
        secs = time.perf_counter() - started
        why = str(exc) if isinstance(exc, _Withhold) else f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, _Withhold):
            logger.exception("[Tower][WorldBuilder] the final scale guard failed on %s/%s (%s)",
                             world_id, session_id, mode_)
        if mode_ == MODE_SHADOW:
            return GuardOutcome(mode_, DECISION_SHADOW, solution, result,
                                summary(assessment, mode_=mode_, decision=DECISION_SHADOW, seconds=secs, error=why),
                                audit_document(assessment, mode_=mode_, decision=DECISION_SHADOW, solution=solution,
                                               inputs=inputs, seconds=secs, error=why), depth)
        logger.warning("[Tower][WorldBuilder] the final scale guard WITHHELD %s/%s: %s", world_id, session_id, why)
        summ = summary(assessment, mode_=mode_, decision=DECISION_WITHHELD, seconds=secs,
                       error=None if isinstance(exc, _Withhold) else why)
        summ["why"] = why
        return GuardOutcome(mode_, DECISION_WITHHELD, None, None, summ,
                            audit_document(assessment, mode_=mode_, decision=DECISION_WITHHELD, solution=solution,
                                           inputs=inputs, seconds=secs, error=why), depth)


def withhold(outcome: "GuardOutcome", why: str) -> "GuardOutcome":
    """`outcome` turned into a withhold (on): nothing to publish, the summary and the audit say why."""
    summ = dict(outcome.summary, decision=DECISION_WITHHELD, why=why)
    audit = dict(outcome.audit, decision=DECISION_WITHHELD, error=why)
    return GuardOutcome(outcome.mode, DECISION_WITHHELD, None, None, summ, audit, outcome.depth)


class _Withhold(RuntimeError):
    """The assessment refused the candidate (not an error)."""


def withheld_reason(summ: dict) -> str:
    """The solve's `reason` for a withheld candidate: one line, no path."""
    why = str((summ or {}).get("why") or "no certifiable room").splitlines()[0][:200]
    return WHY_WITHHELD.format(why)

