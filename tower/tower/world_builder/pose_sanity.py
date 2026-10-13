"""Quarantine implausible room poses before publishing final geometry.

Two rules share one switch and one cap.

Radius: the room centre is the coordinate-wise median of its posed camera centres.
A camera farther than 3 times the 90th percentile radius is a candidate. The p90 is
taken over the cameras within 10 times the median radius, so a far cluster of up to
the cap cannot inflate its own threshold. Settled clean draws have max/median radius
of at most 5.8 and p90/median of at most 3.5: the trim drops none of their cameras
and leaves their p90 unchanged.

Temporal jump: among the remaining room cameras in capture order, a camera is a
candidate when its step to BOTH temporal neighbours is implausibly fast and the
neighbours agree with each other far better than with it (an isolated out-and-back
spike, not a genuine fast walk, whose steps run straight through). Speeds are
room-scale-free: a step's length over the room's median step length, times the
room's median keyframe interval over the step's interval (the interval floored at
that median, so bunched receipts cannot inflate a speed). Settled draws put every
straight-through step at or below 8 median speeds and every out-and-back step of
the clean walk-7/walk-8 draws at or below 7.3; the w9b turn-bridge spike 00003461
is 46.6 (ratio 0.22). JUMP_SPEED_MULTIPLE = 20 and JUMP_RETURN_RATIO = 0.5 sit
between them; see the pose-jump report for per-draw hits.

The union of both candidate sets is capped at 10% of posed room cameras; above it
the room publishes unchanged with a warning. Quarantined cameras leave component 0
along with their sparse points. Dense, surface, plan, appearance and the viewer path
then read the same smaller published room. The switch is exact `on` and defaults off.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from tower.config import world_pose_sanity_setting

logger = logging.getLogger(__name__)

RADIUS_MULTIPLE = 3.0
TRIM_MEDIAN_MULTIPLE = 10.0
JUMP_SPEED_MULTIPLE = 20.0
JUMP_RETURN_RATIO = 0.5
MAX_ROOM_FRACTION = 0.10
REASON = "pose-sanity-outlier"


def enabled() -> bool:
    return world_pose_sanity_setting()


def assess(solution, keyframes, *, started_at: float) -> dict:
    """Pure far-camera decision on the published component 0, including riders."""
    from tower.world_builder.coherence_publish import capture_spans  # noqa: PLC0415

    order = {kid: i for i, kid in enumerate(solution.keyframe_ids or [])}
    receipts = {k.keyframe_id: float(k.received_at) for k in keyframes}
    room = []
    finite = []
    for kid, pose in (solution.poses or {}).items():
        if int(pose.get("component", 0)) != 0:
            continue
        room.append(kid)
        try:
            R = np.asarray(pose["rotation"], np.float64).reshape(3, 3)
            t = np.asarray(pose["translation"], np.float64).reshape(3)
            centre = -R.T @ t
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(centre).all():
            finite.append((kid, centre))
    audit = {"rule": "pose-sanity/2", "radius_multiple": RADIUS_MULTIPLE,
             "trim_median_multiple": TRIM_MEDIAN_MULTIPLE,
             "jump_speed_multiple": JUMP_SPEED_MULTIPLE, "jump_return_ratio": JUMP_RETURN_RATIO,
             "max_room_fraction": MAX_ROOM_FRACTION, "room_posed": len(room),
             "finite_centres": len(finite), "candidate_count": 0, "quarantined": 0,
             "keyframe_ids": [], "capture_spans_s": [], "p90_radius": None,
             "max_distance": None, "median_radius": None, "radius_trimmed": 0,
             "radius_candidates": [], "jump_candidates": [],
             "jump": None, "state": "unchanged", "warning": None}
    if len(finite) < 3:
        audit["warning"] = "too few finite room camera centres"
        return audit
    centres = np.asarray([centre for _, centre in finite])
    median = np.median(centres, axis=0)
    distances = np.linalg.norm(centres - median, axis=1)
    p50 = float(np.median(distances))
    inner = distances[distances <= TRIM_MEDIAN_MULTIPLE * p50]
    p90 = float(np.percentile(inner, 90)) if len(inner) else float("nan")
    audit["median_radius"] = round(p50, 6)
    audit["radius_trimmed"] = int(len(distances) - len(inner))
    audit["p90_radius"] = round(p90, 6)
    audit["max_distance"] = round(float(np.max(distances)), 6)
    if not np.isfinite(p90) or p90 <= 0:
        audit["warning"] = "degenerate room radius"
        return audit
    far = {kid for (kid, _), d in zip(finite, distances) if d > RADIUS_MULTIPLE * p90}
    near = [(kid, centre) for kid, centre in finite if kid not in far]
    jumps, audit["jump"] = temporal_jumps(near, keyframes)
    rank = lambda kid: order.get(kid, len(order))  # noqa: E731
    audit["radius_candidates"] = sorted(far, key=rank)
    audit["jump_candidates"] = sorted(jumps, key=rank)
    candidates = sorted(far | set(jumps), key=rank)
    audit["candidate_count"] = len(candidates)
    if not candidates:
        return audit
    if len(candidates) > MAX_ROOM_FRACTION * len(room):
        audit["state"] = "cap-exceeded"
        audit["warning"] = "pose-sanity candidates exceed 10% of the posed room; published unchanged"
        return audit
    times = [max(0.0, receipts[kid] - started_at) for kid in candidates if kid in receipts]
    spans, _duration = capture_spans(times)
    audit.update(state="quarantined", quarantined=len(candidates),
                 room_fraction=round(len(candidates) / len(room), 6),
                 keyframe_ids=candidates, capture_spans_s=spans)
    return audit


def temporal_jumps(cameras, keyframes) -> tuple[list[str], dict]:
    """Isolated out-and-back spikes among (keyframe_id, centre) room cameras.

    Neighbours are consecutive by capture order: the keyframe's source sequence
    when every camera has one, else its receipt time. Cameras without a journal
    record have no capture time and are left out of the sequence.
    """
    records = {k.keyframe_id: k for k in keyframes}
    timed = [(kid, centre, records[kid]) for kid, centre in cameras if kid in records]
    by_seq = all(isinstance(getattr(rec, "source_seq", None), int) for _, _, rec in timed)
    timed.sort(key=lambda row: ((row[2].source_seq, float(row[2].received_at)) if by_seq
                                else (float(row[2].received_at),)))
    summary = {"sequenced": len(timed), "order": "source_seq" if by_seq else "received_at",
               "median_step": None, "median_interval_s": None, "max_spike_speed": None}
    if len(timed) < 3:
        return [], summary
    centres = np.asarray([centre for _, centre, _ in timed], np.float64)
    times = np.asarray([float(rec.received_at) for _, _, rec in timed], np.float64)
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    intervals = np.diff(times)
    unit = float(np.median(steps))
    tau = float(np.median(intervals))
    summary.update(median_step=round(unit, 6), median_interval_s=round(tau, 6))
    if not (np.isfinite(unit) and unit > 0 and np.isfinite(tau) and tau > 0):
        summary["warning"] = "degenerate room step or interval; temporal rule skipped"
        return [], summary
    # Step speed in room median speeds; a bunched receipt interval is floored at tau.
    speed = steps / unit * tau / np.maximum(intervals, tau)
    to_both = np.minimum(speed[:-1], speed[1:])
    through = np.linalg.norm(centres[2:] - centres[:-2], axis=1)
    out_and_back = through <= JUMP_RETURN_RATIO * (steps[:-1] + steps[1:])
    spikes = out_and_back & (to_both >= JUMP_SPEED_MULTIPLE)
    if out_and_back.any():
        summary["max_spike_speed"] = round(float(to_both[out_and_back].max()), 3)
    return [timed[i + 1][0] for i in np.flatnonzero(spikes)], summary


def enforce_sparse_quarantine(solution, keyframe_ids):
    """Keep points seen by quarantined cameras out of component 0 after relabeling."""
    if not keyframe_ids:
        return solution
    # Each camera keeps its own current label: a later relabel (the scale guard's) may
    # have split the quarantined cameras across several unplaced components. A point
    # still in the room moves to the label of the first quarantined camera, in solver
    # order, that observes it.
    far_ids = set(keyframe_ids)
    labels = {}
    for i, kid in enumerate(solution.keyframe_ids):
        if kid in far_ids and kid in (solution.poses or {}):
            label = int(solution.poses[kid].get("component", 0))
            if label != 0:
                labels[i] = label
    observations = np.asarray(solution.observations)
    if not len(observations) or not labels:
        return solution
    point_label = np.asarray(solution.component).copy()
    cams = observations[:, 0].astype(np.int64)
    points = observations[:, 2].astype(np.int64)
    for index in sorted(labels):
        point_ids = np.unique(points[cams == index])
        point_label[point_ids[point_label[point_ids] == 0]] = labels[index]
    errors = np.asarray(solution.error)
    refreshed = []
    for entry in solution.components:
        entry = dict(entry)
        mask = point_label == entry["index"]
        entry["points"] = int(mask.sum())
        entry["mean_error_px"] = (float(errors[mask].mean()) if mask.any() and len(errors) == len(mask)
                                  else None)
        refreshed.append(entry)
    return dataclasses.replace(solution, component=point_label.astype(np.int32), components=refreshed)


def apply(solution, result, *, keyframes, session_id: str, started_at: float):
    """Return (published solution, gate result, audit), preserving the input if none are far."""
    from tower.world_builder import coherence_publish as CP  # noqa: PLC0415

    prior = (solution.solve or {}).get("pose_sanity") or {}
    prior_ids = prior.get("keyframe_ids") or []
    if result is not None and prior.get("state") == "quarantined" and prior_ids and all(
            kid in (solution.poses or {}) and int(solution.poses[kid].get("component", 0)) != 0
            for kid in prior_ids):
        # A published solution may be handed to this seam again. A re-gate first
        # restores solver labels, so it does not take this path.
        return solution, result, prior
    audit = assess(solution, keyframes, started_at=started_at)
    if audit["state"] == "unchanged":
        return solution, result, audit
    if audit["state"] == "cap-exceeded":
        logger.warning("[Tower][WorldBuilder] %s: %s", session_id, audit["warning"])
        return dataclasses.replace(solution, solve=dict(solution.solve or {}, pose_sanity=audit)), result, audit

    labels = {kid: int(p.get("component", 0)) for kid, p in (solution.poses or {}).items()}
    components = []
    for entry in solution.components or []:
        label = int(entry.get("index", 0))
        state = entry.get("gate_state") or ("placed" if label == 0 else "unplaced")
        components.append({"label": label, "state": state,
                           "reasons": list(entry.get("gate_reasons") or ([] if label == 0 else ["solved-separately"])),
                           "source_components": list(entry.get("solver_components") or [label])})
    known = {entry["label"] for entry in components}
    for label in sorted(set(labels.values()) - known):
        components.append({"label": label, "state": "placed" if label == 0 else "unplaced",
                           "reasons": [] if label == 0 else ["solved-separately"],
                           "source_components": [label]})
    new_label = max([*labels.values(), *(entry["label"] for entry in components)], default=0) + 1
    for kid in audit["keyframe_ids"]:
        labels[kid] = new_label
    components.append({"label": new_label, "state": "unplaced", "reasons": [REASON],
                       "source_components": [0]})
    published = CP.relabel_solution(solution, labels, components)
    # Majority voting can retain a far camera's sparse points when tracks also
    # touch nearby cameras. The scale guard later relabels again, so it calls
    # this same function after its own relabel.
    published = enforce_sparse_quarantine(published, audit["keyframe_ids"])
    published = dataclasses.replace(published, solve=dict(solution.solve or {}, pose_sanity=audit))
    gate = dict(((result.components or {}).get("gate") if result is not None else None) or {})
    gate_block = {"gate": gate.get("id"), "params": gate.get("params"),
                  "params_digest": gate.get("params_digest"),
                  "masks_applied": gate.get("masks_applied", (solution.transients or {}).get("state") == "applied"),
                  "metric_available": gate.get("metric_available", True), "components": components}
    document = CP.components_document(session_id, published, labels, gate_block, keyframes, started_at)
    if document is None:
        raise ValueError("the pose sanity quarantine left no published room")
    if result is None:
        result = CP.GateResult(solution=published, record={}, components=document)
    else:
        result = dataclasses.replace(result, solution=published, components=document)
    return published, result, audit
