"""Synthetic solves for the pose-quarantine tests (`test_world_builder_pose_quarantine*.py`, RUN P5-PQ RULE.md) and the
golden record of today's outputs with `TOWER_WORLD_POSE_QUARANTINE` unset.

KEEP IN STEP with RUN/experiments/P5-PQ/golden_record.py, which ran `golden_outputs` against the product tree BEFORE
P5-PQ changed it (6db75f3, exported read-only) and wrote `golden/world_builder_pose_quarantine_off.json`.

Not a test module: no `test_` prefix. Everything here is deterministic.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from tests import wb_anchor_verify_fixtures as F

SID = F.SID


def _pose(i: int, centre) -> dict:
    R = F.rz(3.0 * i)
    C = np.asarray(centre, np.float64)
    return {"rotation": R.ravel().tolist(), "translation": (-R @ C).tolist()}


def ring_centres(n: int, radius: float = 1.0, wobble: float = 0.25):
    """A room: centres on a wobbly ring, so the median radius is ~`radius` and the p95 a little more."""
    return [np.array([radius * (1 + wobble * math.sin(3.1 * i)) * math.cos(2 * math.pi * i / n),
                      0.05 * (i % 5),
                      radius * (1 + wobble * math.sin(3.1 * i)) * math.sin(2 * math.pi * i / n)]) for i in range(n)]


def rider_solution(n: int, riders, *, centres=None):
    """`F.multi_solution((n,))` -- n supported room cameras in one solver component -- followed by RIDERS, cameras
    under 30 observations. `riders`: [(room camera it shares points with, shared points, own points, centre)]. A
    rider's own points are seen by it alone. Every camera is component 0, as the solver returns it."""
    from tower.world_builder.global_solve import Solution

    base = F.multi_solution((n,), centres=centres if centres is not None else ring_centres(n))
    obs = [list(map(int, r)) for r in np.asarray(base.observations)]
    by_cam: dict = {}
    for c, _f, p in obs:
        by_cam.setdefault(c, []).append(p)
    p_next = len(base.xyz)
    poses = dict(base.poses)
    kids = list(base.keyframe_ids)
    for k, (partner, shared, own, centre) in enumerate(riders):
        c = n + k
        mine = sorted(set(by_cam[partner]))[:shared]
        for p in mine:
            obs.append([c, len(obs), p])
        for _ in range(own):
            obs.append([c, len(obs), p_next])
            p_next += 1
        kid = f"{SID}:{c:08d}"
        kids.append(kid)
        poses[kid] = {"component": 0, **_pose(c, centre), "observations": int(shared + own)}
    obs = np.asarray(obs, np.int32)
    first = np.zeros(p_next, np.int32)
    for row in obs[::-1]:
        first[row[2]] = row[0]
    return dataclasses.replace(
        base, keyframe_ids=kids, poses=poses, xyz=np.zeros((p_next, 3), np.float32),
        rgb=np.zeros((p_next, 3), np.uint8), component=np.zeros(p_next, np.int32), first_keyframe=first,
        track_length=np.bincount(obs[:, 2], minlength=p_next).astype(np.int32),
        error=np.full(p_next, 0.5, np.float32), observations=obs,
        observation_xy=np.zeros((len(obs), 2), np.float32),
        components=[{"index": 0, "images": len(kids), "points": p_next}])


# The rider scenario of the golden and the tests: 40 room cameras on a ring of radius 1; riders sharing 0 (the walk-4
# leak: 0 observations, ~2,000 radii out), 2 (under m = 3) and 5 points (attached) with room camera 10; one far
# rider sharing 0 points but with 12 of its own.
RIDERS = ((10, 0, 0, (2000.0, 0.0, 0.0)), (10, 2, 3, (0.5, 0.0, 0.2)), (10, 5, 4, (0.8, 0.0, 0.1)),
          (20, 0, 12, (0.0, 0.0, 2200.0)))
RIDER_N = 40


def rider_model(n=RIDER_N, riders=RIDERS):
    """(`coherence_gate.SolveModel`, links, rotations, levels) of `rider_solution`."""
    from tower.world_builder import coherence_gate as CG

    sol = rider_solution(n, riders)
    kids = sol.keyframe_ids
    model = CG.SolveModel(
        names=[F.name(i) for i in range(len(kids))],
        component=np.zeros(len(kids), np.int64),
        R_cw=np.asarray([np.asarray(sol.poses[k]["rotation"], np.float64).reshape(3, 3) for k in kids]),
        n_obs=np.asarray([int(sol.poses[k]["observations"]) for k in kids], np.int64),
        obs_image=np.asarray(sol.observations[:, 0], np.int64), obs_point=np.asarray(sol.observations[:, 2], np.int64),
        n_points=int(len(sol.xyz)))
    links, rots = F.multi_links((n,), ())
    levels = {F.name(i): 0.0 for i in range(n)}
    return model, links, rots, levels


def publish_riders(tmp: Path, n=RIDER_N, riders=RIDERS) -> dict:
    """`gate_and_publish` on `rider_solution`, the gate's inputs faked (`F.patch_gate_inputs`)."""
    import pytest

    from tower.world_builder import coherence_publish as CP
    from tower.world_builder.global_solve import SolveWorkspace, write_solution

    links, rots = F.multi_links((n,), ())
    total = n + len(riders)
    tmp.mkdir(parents=True, exist_ok=True)
    ws = SolveWorkspace(tmp / "w1" / "solve" / SID)
    with pytest.MonkeyPatch.context() as mp:
        F.patch_gate_inputs(mp, links, rots, [0.0] * n + [None] * len(riders))
        out, record = CP.gate_and_publish(F.Store(tmp), "w1", SID, ws, rider_solution(n, riders), final=True,
                                          gate=True, database_path="db", keyframes=F.keyframes(total),
                                          write=write_solution)
    return {"record": F.normalise(json.loads(json.dumps(record, default=str))),
            "solution.json": F._read(ws.solution_path),
            "components.json": F._read(ws.root / CP.COMPONENTS_FILENAME),
            "poses": {k: int(p["component"]) for k, p in out.poses.items()}}


# ---------------------------------------------------------------------------------------------------------------
# the walk-4 bed stretch, for the anchor verification's motion seal (RULE.md (c))

BED_N = 100
BED = (70, 78)           # 8 cameras whose poses fly off, each farther than the last
GAP = (66, 70)           # 4 cameras of ANOTHER solver component between the room and the bed stretch (walk 4: 251-291)
SEGMENTS = [0] * 50 + [1] * 50


def bed_centres(n=BED_N, bed=BED, far=1.0e5):
    cs = F.line_centres(n)
    for k, i in enumerate(range(*bed)):
        cs[i] = cs[i] + np.array([far * (k + 1), 0.0, 0.0])
    return cs


def bed_solution(n=BED_N, bed=BED, gap=GAP, far=1.0e5):
    """One solver component of n cameras along a line, except cameras `gap` (solver component 1), and the `bed`
    stretch flown `far` units out, farther at each camera (walk 4: 0.6-10 M units, consecutive jumps of 52 k-1.3 M m
    in 0.07-0.46 s)."""
    sol = F.multi_solution((n,), centres=bed_centres(n, bed, far))
    poses = {k: dict(p) for k, p in sol.poses.items()}
    for i in range(*gap):
        poses[f"{SID}:{i:08d}"]["component"] = 1
    return dataclasses.replace(sol, poses=poses)


BED_CROSS = ((5, 80), (5, 81), (6, 81))   # the run after the stretch is attached to the room by a closed triangle


def _side(i, bed=BED, gap=GAP):
    return "gap" if gap[0] <= i < gap[1] else "bed" if bed[0] <= i < bed[1] else "room"


def bed_pairs(n=BED_N, *, confirm_after=True):
    """The masked tier: an image chain within the room and within the bed stretch, never across the gap or out of the
    stretch; revisit pairs confirm the room before the gap and (`confirm_after`) the run after the stretch; the stretch
    has none (unverifiable)."""
    chain = [(i, j) for i, j in F.chain_pairs(n) if _side(i) == _side(j) != "gap"]
    revisit = [(5, 55), (8, 58), (12, 61), (15, 64), (20, 52), (25, 60), (30, 63)]
    if confirm_after:
        revisit += [(2, 80), (6, 84), (10, 88), (14, 92), (18, 95), (22, 81)]
    return chain + revisit


def publish_bed(tmp: Path, *, anchor_verify="on", confirm_after=True, n=BED_N) -> dict:
    """`gate_and_publish` on `bed_solution` with the anchor verification's parts `anchor_verify` (the gate's inputs and
    the masked tier faked)."""
    import pytest

    from tower.world_builder import anchor_verify as AV
    from tower.world_builder import coherence_publish as CP
    from tower.world_builder.global_solve import SolveWorkspace, write_solution

    sol = bed_solution(n)
    links, rots = F.multi_links((n,), BED_CROSS)
    links = {k: v for k, v in links.items() if not any(GAP[0] <= int(x[:8]) < GAP[1] for x in k)}
    rots = {k: v for k, v in rots.items() if k in links}
    R, C = F.poses_of(sol, n)
    arrays = F.synthetic_pairs(R, C, bed_pairs(n, confirm_after=confirm_after))
    tmp.mkdir(parents=True, exist_ok=True)
    ws = SolveWorkspace(tmp / "w1" / "solve" / SID)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TOWER_WORLD_ANCHOR_VERIFY", anchor_verify)
        F.patch_gate_inputs(mp, links, rots, [0.0] * n)
        mp.setattr(AV, "_default_pair_builder", lambda root, names, camera: (
            arrays, {"key": "synthetic", "digest": AV.pair_digest(arrays), "pairs": len(arrays["i"]),
                     "source": "built", "seconds": 0.0}))
        out, record = CP.gate_and_publish(F.Store(tmp), "w1", SID, ws, sol, final=True, gate=True,
                                          database_path="db", keyframes=F.keyframes(n, segments=SEGMENTS),
                                          write=write_solution)
    return {"record": F.normalise(json.loads(json.dumps(record, default=str))),
            "solution.json": F._read(ws.solution_path),
            "components.json": F._read(ws.root / CP.COMPONENTS_FILENAME),
            "poses": {k: int(p["component"]) for k, p in out.poses.items()}}


# ---------------------------------------------------------------------------------------------------------------
# the viewer's camera list: walk 4's and walk 3's impossible poses, at their recorded distances (RUN lead/diag4
# DIAGNOSIS.md Q3), in a synthetic room of the recorded size. No walk geometry is shipped: only these numbers.

# walk 4: 354 published room cameras, median radius 1.724 units
W4_ROOM, W4_MEDIAN_RADIUS = 354, 1.724
# source seq: (distance from the room's median centre in units, observations)
W4_BED = {2026: (10_171_442, 100), 2031: (9_439_401, 99), 2037: (7_969_445, 118), 2047: (3_306_913, 142),
          2050: (2_117_677, 138), 2053: (1_259_308, 113), 2055: (832_315, 93), 2056: (643_707, 62)}
W4_RIDERS = {3090: (1_841, 2), 3092: (2_232, 2), 3095: (2_150, 0), 3097: (1_907, 0), 3098: (2_140, 2)}
# walk 3: s2400, 0 observations at 2,936 x the median radius of 3.03 units (748 room poses)
W3_ROOM, W3_MEDIAN_RADIUS = 748, 3.03
W3_FAR = {2400: (2_936 * 3.03, 0)}


def pose_solution(entries):
    """A `global_solve.Solution` of posed keyframes only: `entries` [(keyframe id, centre, observations,
    component)], in capture order, with no points."""
    from tower.world_builder.global_solve import Solution

    poses = {kid: {"component": int(comp), **_pose(i, c), "observations": int(o)}
             for i, (kid, c, o, comp) in enumerate(entries)}
    return Solution(
        solver="glomap", solved_at=1234.5, input_digest="digest-final", keyframe_ids=[e[0] for e in entries],
        poses=poses, components=[{"index": 0, "images": len(entries), "points": 0}],
        xyz=np.zeros((0, 3), np.float32), rgb=np.zeros((0, 3), np.uint8), component=np.zeros(0, np.int32),
        first_keyframe=np.zeros(0, np.int32), track_length=np.zeros(0, np.int32), error=np.zeros(0, np.float32),
        observations=np.zeros((0, 3), np.int32), observation_xy=np.zeros((0, 2), np.float32),
        camera={"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240})


def walk_like_solution(room: int, median_radius: float, far: dict, *, riders_obs: int = 12):
    """`room` published cameras on a ring whose median radius is ~`median_radius`, a few riders inside the room
    (`riders_obs` observations: unsupported but plausible), and the `far` poses {source seq: (distance, observations)}
    placed at their recorded distance, in capture order among them."""
    ring = ring_centres(room, radius=median_radius)
    entries = [(f"{SID}:{i * 10 + 4:08d}", ring[i], 60 + (i % 40), 0) for i in range(room)]
    entries += [(f"{SID}:{i * 10 + 8:08d}", ring[i] * 0.9, riders_obs, 0) for i in range(0, room, 50)]
    for k, (seq, (dist, obs)) in enumerate(sorted(far.items())):
        u = np.array([math.cos(0.7 * k), 0.3, math.sin(0.7 * k)])
        entries.append((f"{SID}:{seq:08d}", u / np.linalg.norm(u) * float(dist), int(obs), 0))
    entries.sort(key=lambda e: e[0])
    return pose_solution(entries)


def camera_path_of(tmp: Path, solution) -> list:
    """`surface_render._camera_path` on `solution`, written where it reads it."""
    from tower.world_builder.global_solve import SolveWorkspace, write_solution
    from tower.world_builder.surface_render import _camera_path

    write_solution(SolveWorkspace(tmp / "w1" / "solve" / SID), solution)
    return _camera_path(F.Store(tmp), "w1", SID)


def golden_outputs(tmp: Path) -> dict:
    """Every output the golden pins, for the code as it is imported now (the switch unset)."""
    from tower.world_builder import coherence_gate as CG

    model, links, rots, levels = rider_model()
    out = {"apply_gate_riders": F.normalise(json.loads(json.dumps(
        CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True))))}
    out["publish_riders"] = publish_riders(tmp / "r")
    out["publish_bed_av_on"] = publish_bed(tmp / "b")
    out["publish_bed_av_on_unconfirmed"] = publish_bed(tmp / "u", confirm_after=False)
    out["camera_path_riders"] = camera_path_of(tmp / "cr", rider_solution(RIDER_N, RIDERS))
    out["camera_path_bed"] = camera_path_of(tmp / "cb", bed_solution())
    out["camera_path_w4"] = camera_path_of(tmp / "c4", walk_like_solution(W4_ROOM, W4_MEDIAN_RADIUS,
                                                                           {**W4_BED, **W4_RIDERS}))
    out["camera_path_w3"] = camera_path_of(tmp / "c3", walk_like_solution(W3_ROOM, W3_MEDIAN_RADIUS, W3_FAR))
    return out
