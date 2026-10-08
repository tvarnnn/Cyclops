#!/usr/bin/env python
"""Read-only Exp2 arm score. Missing physical certification/checker evidence stays null."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tower.world_builder import exp2_retrieval as E


def _pair(a, b):
    return tuple(sorted((a, b)))


def _scale_result(checker, arm_name=None):
    if checker is None:
        return None
    if "arms" in checker:
        arms = checker["arms"]
        matches = ([arm for arm in arms if arm.get("arm") == arm_name]
                   if arm_name else arms)
        if len(matches) != 1:
            raise ValueError("select exactly one frozen scale-checker arm")
        checker = matches[0]
    regions = (checker.get("scale") or {}).get("regions") or []
    closet = next((r for r in regions if r.get("region") == "closet"), None)
    if closet is None:
        return {"status": "uncertified", "reason": "frozen checker has no closet region"}
    ratio, measured = closet.get("ratio"), closet.get("measured", 0)
    return {"ratio": ratio, "measured": measured,
            "in_band": bool(measured >= 10 and ratio is not None and 0.8 <= ratio <= 1.25),
            "checker_status": closet.get("status"), "walk7_baseline_ratio": 0.1696,
            "wall_door_reference_about": 0.18,
            "pose_checker_status": (checker.get("pose") or {}).get("status"),
            "pose_flag_counts": (checker.get("pose") or {}).get("flag_counts")}


def score_arm(labels, truth, candidates, verified, solution, components,
              *, scale_checker=None, fow_checker=None, placement_review=None,
              surface_review=None, published_ids=None, used_pairs=None,
              scale_arm=None) -> dict:
    """No room is inferred from geometry; certification must be supplied independently."""
    proposals = {_pair(row["query"], target)
                 for row in candidates for target in row["targets"]}
    true = {pair for pair, label in truth.items() if label == "true_doorway_return"}
    false = {pair for pair, label in truth.items() if label == "false_cross_room"}
    ambiguous = {pair for pair, label in truth.items() if label == "ambiguous"}
    verified_pairs = set(verified)
    queried = {row["query"] for row in candidates}
    eligible = {pair for pair in true if pair[0] in queried or pair[1] in queried}
    bridges = {
        "eligible_true_doorway_return": len(eligible),
        "eligible_recall_at_50": len(eligible & proposals) / len(eligible) if eligible else None,
        "true_candidates": len(proposals & true),
        "false_candidates": len(proposals & false),
        "verified_true_doorway_return": len(verified_pairs & true),
        "verified_false_cross_room": len(verified_pairs & false),
        "verified_true_used": len(verified_pairs & true & used_pairs) if used_pairs is not None else None,
        "verified_false_used": len(verified_pairs & false & used_pairs) if used_pairs is not None else None,
        "ambiguous_verified_excluded": len(verified_pairs & ambiguous),
        "verified_true_pairs": [list(p) for p in sorted(verified_pairs & true)],
        "verified_false_pairs": [list(p) for p in sorted(verified_pairs & false)],
    }
    poses = solution.get("poses") or {}
    membership = {}
    if components is not None:
        for item in components.get("components", []):
            for kid in item.get("keyframe_ids") or []:
                membership[kid] = item
    has_publication = components is not None or published_ids is not None
    frames = defaultdict(lambda: {"captured": 0, "posed": 0,
                                  "published": 0 if has_publication else None,
                                  "placed_correct_room": 0 if placement_review is not None else None,
                                  "published_wrong_room": 0 if placement_review is not None else None,
                                  "surfaced": 0 if surface_review is not None else None,
                                  "textured": 0 if surface_review is not None else None,
                                  "viewer_visible": 0 if surface_review is not None else None})
    for label in labels:
        kid = label["keyframe_id"]
        row = frames[f'{label["region"]}/{label["confidence"]}']
        row["captured"] += 1
        pose = poses.get(kid) or {}
        posed = int(pose.get("observations") or 0) >= 30
        row["posed"] += int(posed)
        member = membership.get(kid) or {}
        published = bool(posed and ((published_ids is not None and kid in published_ids)
                         or (member.get("shown_as") in ("room", "area"))))
        if has_publication:
            row["published"] += int(published)
        if placement_review is not None:
            certified = placement_review.get(kid)
            physical = label.get("room")
            eligible_room = physical not in (None, "", "ambiguous", "unclear")
            placed = member.get("state") == "placed" if components is not None else True
            row["placed_correct_room"] += int(published and placed and eligible_room and
                certified == physical)
            row["published_wrong_room"] += int(published and eligible_room and
                certified not in (None, "", "ambiguous", physical))
        if surface_review is not None:
            evidence = surface_review.get(kid) or {}
            for key in ("surfaced", "textured", "viewer_visible"):
                row[key] += int(bool(evidence.get(key)))
    fow = None if fow_checker is None else {
        "verdict": fow_checker.get("verdict"),
        "wrong_room_support": len(fow_checker.get("violations") or []),
        "coverage_receipts": fow_checker.get("coverage_receipts"),
    }
    nonsequential = set()
    for row in candidates:
        for j, target in enumerate(row["targets"]):
            indices = row.get("target_indices")
            if indices is not None and abs(row["query_index"] - indices[j]) > E.OVERLAP:
                nonsequential.add(_pair(row["query"], target))
    times = solution.get("timing") or {}
    solve_seconds = (sum(float(times[k]) for k in ("prepare_s", "extract_s", "match_s", "map_s"))
                     if all(k in times for k in ("prepare_s", "extract_s", "match_s", "map_s")) else None)
    return {"bridges": bridges, "frames": dict(sorted(frames.items())),
            "closet_scale": _scale_result(scale_checker, scale_arm),
            "fow_wrong_room_support": fow,
            "solve_time": {"seconds": solve_seconds, "stages": times},
            "candidate_queries": len(candidates),
            "candidate_p50": float(np.median([len(r["targets"]) for r in candidates])) if candidates else 0,
            "candidate_p95": float(np.percentile([len(r["targets"]) for r in candidates], 95)) if candidates else 0,
            "candidate_nonsequential_unique": len(nonsequential) if all(
                "target_indices" in r for r in candidates) else None,
            "sift_verified_pairs": len(verified_pairs),
            "candidate_sift_pass": len(proposals & verified_pairs),
            "candidate_sift_fail": len(proposals - verified_pairs),
            "verification": "SIFT/COLMAP; LightGlue not triggered in slices 1-3"}


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def resolve_label_rooms(labels, adjudication):
    """Walk 7's blind room wins; older walks need a frozen inspected-other row."""
    inspected = {row["keyframe_id"]: row.get("room") for row in adjudication
                 if row.get("record_type") == "frame" and row.get("keyframe_id")}
    default = {"desk": "bedroom", "bed": "bedroom", "closet": "closet",
               "bathroom": "bathroom"}
    resolved = []
    for row in labels:
        item = dict(row)
        if not item.get("room"):
            item["room"] = (inspected.get(item["keyframe_id"])
                            if item.get("region") == "other" else default.get(item.get("region")))
            item["room"] = item["room"] or "ambiguous"
        resolved.append(item)
    return resolved


def verify_frozen_labels(freeze_path, expected_sha256, labels_path, adjudication_path):
    if E.sha256(freeze_path) != expected_sha256:
        raise ValueError("freeze manifest SHA-256 differs from pinned value")
    frozen = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    if E.sha256(labels_path) != frozen["labels_sha256"]:
        raise ValueError("frozen labels digest changed")
    if E.sha256(adjudication_path) != frozen["adjudication_sha256"]:
        raise ValueError("frozen adjudication digest changed")
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--components", type=Path)
    parser.add_argument("--scale-result", type=Path, help="JSON stdout of frozen scale_outlier_check.py")
    parser.add_argument("--scale-arm", help="arm name when the frozen checker output has multiple arms")
    parser.add_argument("--fow-result", type=Path, help="JSON stdout of frozen fow_door_wall_check.py")
    parser.add_argument("--placement-review", type=Path, help="CSV keyframe_id,certified_room")
    parser.add_argument("--bridge-review", type=Path,
                        help="CSV query_id,target_id,used,rotation_deg,metric_ratio,position_jump_m,pose_jump")
    parser.add_argument("--surface-review", type=Path, help="JSON keyed by keyframe_id")
    parser.add_argument("--published-ids", type=Path,
                        help="JSON list of published keyframe IDs from the settled derived tree")
    args = parser.parse_args(argv)
    verify_frozen_labels(args.freeze, args.freeze_sha256, args.labels, args.adjudication)
    from tower.world_builder.coherence_gate import read_verified_links
    adjudication = _read_csv(args.adjudication)
    labels = resolve_label_rooms(_read_csv(args.labels), adjudication)
    truth = {}
    for row in adjudication:
        if row.get("record_type") != "pair":
            continue
        pair = _pair(row["query_id"], row["target_id"])
        if pair in truth or row["pair_truth"] not in (
                "true_doorway_return", "false_cross_room", "ambiguous"):
            raise ValueError(f"duplicate or invalid adjudication: {pair}")
        truth[pair] = row["pair_truth"]
    candidate_data = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    candidates = candidate_data["rows"] if isinstance(candidate_data, dict) else candidate_data
    solution = json.loads(args.solution.read_text(encoding="utf-8"))
    components = json.loads(args.components.read_text(encoding="utf-8")) if args.components else None
    if components and components.get("input_digest") != solution.get("input_digest"):
        raise ValueError("components and solution name different input digests")
    scale = json.loads(args.scale_result.read_text(encoding="utf-8")) if args.scale_result else None
    fow = json.loads(args.fow_result.read_text(encoding="utf-8")) if args.fow_result else None
    placement = ({r["keyframe_id"]: r["certified_room"] for r in _read_csv(args.placement_review)}
                 if args.placement_review else None)
    bridge_review = _read_csv(args.bridge_review) if args.bridge_review else None
    used = {_pair(r["query_id"], r["target_id"]) for r in bridge_review
            if r.get("used") == "yes"} if bridge_review is not None else None
    surface = json.loads(args.surface_review.read_text(encoding="utf-8")) if args.surface_review else None
    published_ids = set(json.loads(args.published_ids.read_text(encoding="utf-8"))) if args.published_ids else None
    result = score_arm(labels, truth, candidates, read_verified_links(args.database),
                       solution, components, scale_checker=scale, fow_checker=fow,
                       placement_review=placement, surface_review=surface,
                       published_ids=published_ids, used_pairs=used, scale_arm=args.scale_arm)
    result["bridge_review"] = bridge_review
    result["input_sha256"] = {key: E.sha256(getattr(args, key)) for key in (
        "labels", "adjudication", "freeze", "candidate_manifest", "database", "solution")}
    for key in ("components", "scale_result", "fow_result", "placement_review",
                "bridge_review", "surface_review", "published_ids"):
        if getattr(args, key):
            result["input_sha256"][key] = E.sha256(getattr(args, key))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
