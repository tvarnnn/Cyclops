"""Quarantine implausibly distant room poses before publishing final geometry.

The room centre is the coordinate-wise median of its posed camera centres. A camera
farther than 3 times the 90th percentile radius leaves component 0, along with its
sparse points. Dense, surface, plan, appearance and the viewer path then read the
same smaller published room. The switch is exact `on` and defaults off.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from tower.config import world_pose_sanity_setting

logger = logging.getLogger(__name__)

RADIUS_MULTIPLE = 3.0
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
    audit = {"rule": "pose-sanity/1", "radius_multiple": RADIUS_MULTIPLE,
             "max_room_fraction": MAX_ROOM_FRACTION, "room_posed": len(room),
             "finite_centres": len(finite), "candidate_count": 0, "quarantined": 0,
             "keyframe_ids": [], "capture_spans_s": [], "p90_radius": None,
             "max_distance": None, "state": "unchanged", "warning": None}
    if len(finite) < 3:
        audit["warning"] = "too few finite room camera centres"
        return audit
    centres = np.asarray([centre for _, centre in finite])
    median = np.median(centres, axis=0)
    distances = np.linalg.norm(centres - median, axis=1)
    p90 = float(np.percentile(distances, 90))
    audit["p90_radius"] = round(p90, 6)
    audit["max_distance"] = round(float(np.max(distances)), 6)
    if not np.isfinite(p90) or p90 <= 0:
        audit["warning"] = "degenerate room radius"
        return audit
    candidates = sorted((kid for (kid, _), d in zip(finite, distances)
                         if d > RADIUS_MULTIPLE * p90), key=lambda kid: order.get(kid, len(order)))
    audit["candidate_count"] = len(candidates)
    if not candidates:
        return audit
    if len(candidates) > MAX_ROOM_FRACTION * len(room):
        audit["state"] = "cap-exceeded"
        audit["warning"] = "far-pose candidates exceed 10% of the posed room; published unchanged"
        return audit
    times = [max(0.0, receipts[kid] - started_at) for kid in candidates if kid in receipts]
    spans, _duration = capture_spans(times)
    audit.update(state="quarantined", quarantined=len(candidates),
                 room_fraction=round(len(candidates) / len(room), 6),
                 keyframe_ids=candidates, capture_spans_s=spans)
    return audit


def enforce_sparse_quarantine(solution, keyframe_ids):
    """Keep points seen by quarantined cameras out of component 0 after relabeling."""
    if not keyframe_ids:
        return solution
    far_ids = set(keyframe_ids)
    far_indices = {i for i, kid in enumerate(solution.keyframe_ids) if kid in far_ids}
    observations = np.asarray(solution.observations)
    if not len(observations) or not far_indices:
        return solution
    label = int(solution.poses[keyframe_ids[0]]["component"])
    hit = np.isin(observations[:, 0], list(far_indices))
    point_ids = np.unique(observations[hit, 2].astype(np.int64))
    point_label = np.asarray(solution.component).copy()
    point_label[point_ids[point_label[point_ids] == 0]] = label
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
