"""Offline walk-6 reachability audit; no solve or GPU work is launched."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import world_live_placement as placement


def load_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def audit(run: Path):
    snaps = json.loads((run / "solution-snapshots" / "index.json").read_text())
    for snap in snaps:
        snap["doc"] = json.loads((run / "solution-snapshots" / snap["file"]).read_text())
        snap["posed"] = set(snap["doc"]["poses"])
    baseline = json.loads((run / "placement-eval" / "evaluation.json").read_text())
    earlier = json.loads((run / "placement-eval" / "v11b-evaluation-final.json").read_text())
    b_rows = {row["keyframe_id"]: row for row in baseline["rows"]}
    e_rows = {row["keyframe_id"]: row for row in earlier["rows"]}
    client = json.loads((run / "client.json").read_text())
    session = Path(client["settle"]["session_file"]).parent
    journal = load_lines(session / "keyframes.jsonl")
    events = load_lines(session / "events.jsonl")
    accepted = {row["keyframe_id"]: row for row in load_lines(run / "live-timeline-frames.jsonl") if row.get("keyframe_id") and row.get("keyframe_accepted_at")}
    segments = defaultdict(list)
    for frame in journal:
        segments[frame["segment_index"]].append(frame["keyframe_id"])
    rows = []
    for segment, keys in segments.items():
        for index, key in enumerate(keys):
            if key not in b_rows:
                continue
            base = b_rows[key]
            at = base["accepted_at"]
            if at is None:
                continue
            landed = [s for s in snaps if s["seen_at"] < at and key not in s["doc"]["keyframe_ids"]]
            snap = landed[-1] if landed else None
            prior = keys[:index]
            shared = [k for k in prior if snap and k in snap["posed"]]
            next_direct = next((s["seen_at"]-at for s in snaps if s["seen_at"] >= at and key in s["posed"]), None)
            rows.append({"keyframe_id":key,"accepted_at":at,"segment":segment,"outcome":base["outcome"],
                         "local_status":e_rows.get(key,{}).get("local_status"),"shared_segment_poses":len(shared),
                         "shared_keys":shared,"map":snap["file"] if snap else None,"next_direct_pose_s":next_direct,
                         "local_ms":e_rows.get(key,{}).get("local_ms")})
    recovery = [e for e in events if e.get("kind", "").startswith("recovery_") or
                e.get("kind") in ("relocalizer_started", "relocalizer_stopped")]
    # A single valid PnP anchor can at most carry a component hint. It cannot
    # supply the scale required for a new camera centre after local motion.
    hints = []
    hint_abstentions = Counter()
    anchors = {}
    steps = Counter()
    for row in sorted(rows, key=lambda item: item["accepted_at"]):
        segment = row["segment"]
        if row["outcome"] == "accepted":
            anchors[segment] = (row["keyframe_id"], row["accepted_at"], steps[segment],
                                b_rows[row["keyframe_id"]]["component"])
            continue
        steps[segment] += 1
        if row["map"] is None:
            hint_abstentions["no-causal-map"] += 1
        elif row["local_status"] not in ("solved", "anchor"):
            hint_abstentions["no-local-pose"] += 1
        elif segment not in anchors:
            hint_abstentions["no-same-segment-pnp-anchor"] += 1
        else:
            anchor, anchor_at, anchor_step, component = anchors[segment]
            age = row["accepted_at"] - anchor_at
            count = steps[segment] - anchor_step
            if age > 2 or count > 6:
                hint_abstentions["anchor-expired"] += 1
            else:
                hints.append({"keyframe_id":row["keyframe_id"], "anchor":anchor,
                              "component":component,"age_s":age,"steps":count,
                              "baseline":row["outcome"],
                              "final_component":None})
    final_meta = json.loads((run / "run.json").read_text())
    final_path = Path(final_meta["data_root"]) / "world_builder" / "worlds" / session.parents[1].name / "solve" / session.name / "solution.json"
    final = json.loads(final_path.read_text())
    for hint in hints:
        hint["final_component"] = final["poses"][hint["keyframe_id"]]["component"]
        anchor_pose = placement.entry_pose(final["poses"][hint["anchor"]])
        query_pose = placement.entry_pose(final["poses"][hint["keyframe_id"]])
        hint["final_distance_from_anchor"] = float(np.linalg.norm(query_pose[0] - anchor_pose[0]))
        hint["final_rotation_from_anchor_deg"] = placement.rotation_angle(query_pose[1] @ anchor_pose[1].T)
    direct_delays = [r["next_direct_pose_s"] for r in rows if r["next_direct_pose_s"] is not None]
    abstentions = [r for r in rows if r["outcome"] != "accepted"]
    match_counts = {name: Counter(min(10, b_rows[r["keyframe_id"]].get("matches", 0)) for r in rows
                                  if r["outcome"] == name)
                    for name in ("no-overlap", "insufficient-3d-inliers")}
    return {"snapshots":[{"file":s["file"],"seen_at":s["seen_at"],"posed":len(s["posed"]),
                           "keyframes":len(s["doc"]["keyframe_ids"])} for s in snaps],
            "rows":rows,"recovery":recovery,"hints":hints,
            "summary":{"baseline":dict(Counter(r["outcome"] for r in rows)),
                       "segments":len(segments),"recovery":dict(Counter(e["kind"] for e in recovery)),
                       "shared_prior_segment_pose":sum(r["shared_segment_poses"] > 0 for r in abstentions),
                       "hint_abstentions":dict(hint_abstentions),"hints":len(hints),
                       "hint_final_component_agreement":sum(h["component"] == h["final_component"] for h in hints),
                       "hint_age_p50_p95_s":np.percentile([h["age_s"] for h in hints], [50,95]).tolist() if hints else None,
                       "hint_final_motion_p50_p95":{
                           "distance_solve_units":np.percentile([h["final_distance_from_anchor"] for h in hints],[50,95]).tolist(),
                           "rotation_deg":np.percentile([h["final_rotation_from_anchor_deg"] for h in hints],[50,95]).tolist()} if hints else None,
                       "next_direct_pose_s_p50_p95_min":np.percentile(direct_delays,[50,95,0]).tolist(),
                       "next_direct_within_s":{str(n):sum(r["next_direct_pose_s"] is not None and r["next_direct_pose_s"] <= n for r in abstentions)
                                               for n in (1,2,5,10,30,60,120)},
                       "matches_capped_at_10":{k:dict(v) for k,v in match_counts.items()}}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
