#!/usr/bin/env python
"""A review pack for one saved world: a plan, turnarounds, orientation checks.

WHY THIS EXISTS

The manager needs to LOOK at a world before a demo and answer three questions
without a phone: where did every part of the walk land, is there anything
behind me when I stand where I started, and is the room upright and at one
scale. `world_render.py` draws the derived segments; this draws what the
PRODUCT shows -- the appearance viewer -- plus every solve component.

    plan_topdown.png          every component's sparse points, the fused
                              surface, the walk coloured by time, the stations
    turnaround_sN.png         8 yaw views (45 deg steps) from station N
    turnaround_sN_yawDDD.png  the single views
    turnarounds_all.png       all stations on one sheet
    viewer_opening.png        the viewer page as it opens (browser method)
    pack.json                 method, stations, coverage per view,
                              orientation checks, per-component scale

HOW THE TURNAROUNDS ARE MADE

`browser` (preferred): a throwaway Tower app holding ONLY the geometry router
(`tower.routes.geometry`) is served on a test port, pointed at a staged COPY
of the world. Headless Chrome/Edge with SwiftShader opens the real appearance
page (`/worlds/{w}/render?representation=appearance&transport=tower`) and the
page's own verification hooks (`window.__wbAppearance`: `walk`, `setPose`,
`snapshot`, `coverage`) set each pose and read back the canvas. One change to
the served page: the RGBA8 layer cap is raised to the phone's (ASTC) cap, so
a renderer WITHOUT ASTC would still load the phone's keyframe set instead of
48. (Measured 2026-10-12: Chrome's SwiftShader exposes ASTC, chose
`astc-6x6-rgba` and loaded all 128 phone layers, so the patch was inert.)
pack.json records the encoding, the layer count and the patch.

`splat` (fallback): a CPU point splat of the fused surface's vertex colours
from the same stations. It is geometry, not the product picture.

No GPU is used by either method. Nothing is written to the world store: the
plan and the checks read solution/npz/json files directly, and every Tower
code path that might touch a store runs against the staged copy. The COLMAP
database is never copied and never opened.

    python scripts/world_render_pack.py --world-root <data/world_builder> \\
        --world 420f147d --session bd3d5829 --out-root <renders> \\
        --stage-root <scratch>/stage --label "walk 9"
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import io
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tower.artifact_paths import artifact_root_arg  # noqa: E402

PACK_FORMAT = "wb-render-pack/1"
LIVE_TOWER_PORT = 8000
DEFAULT_PORT = 8240
CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)
# What the stage copy leaves behind: the COLMAP database (opening it mutates
# it, and nothing here needs it), the solver's image copies and models, and
# the live-depth work tree (read in place, read-only, for the scale estimate).
STAGE_IGNORE = ("database.db", "database.db-*", "*.db-journal", "*.db-wal", "*.db-shm")
STAGE_SKIP_DIRS = (("solve", "*", "images"), ("solve", "*", "sparse"), ("dense", "*", "live_depth", "work"))


# ---------------------------------------------------------------------------
# small geometry
# ---------------------------------------------------------------------------


def unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def plane_basis(up) -> tuple[np.ndarray, np.ndarray]:
    """Two unit axes spanning the plane perpendicular to `up` (right-handed)."""
    u = unit(up)
    seed = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = unit(seed - u * (seed @ u))
    e2 = np.cross(u, e1)
    return e1, e2


def angle_deg(a, b) -> float:
    a, b = unit(a), unit(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def pose_arrays(pose: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """(R world->camera, t) of a solution pose, or None when refused."""
    rot, tr = pose.get("rotation"), pose.get("translation")
    if rot is None or tr is None:
        return None
    R = np.asarray(rot, np.float64).reshape(3, 3)
    t = np.asarray(tr, np.float64).reshape(3)
    if not (np.isfinite(R).all() and np.isfinite(t).all()):
        return None
    return R, t


def camera_centre(R, t) -> np.ndarray:
    return -R.T @ t


def camera_forward(R) -> np.ndarray:
    return R.T @ np.array([0.0, 0.0, 1.0])


def camera_up(R) -> np.ndarray:
    return R.T @ np.array([0.0, -1.0, 0.0])


# ---------------------------------------------------------------------------
# stations
# ---------------------------------------------------------------------------


def revisit_counts(centres: np.ndarray, times: np.ndarray, radius: float,
                   gap_s: float = 8.0) -> np.ndarray:
    """For each position: how many separate PASSES of the walk come within
    `radius` of it. Neighbours in time closer than `gap_s` are one pass."""
    C = np.asarray(centres, np.float64)
    T = np.asarray(times, np.float64)
    out = np.zeros(len(C), np.int64)
    for i in range(len(C)):
        near = np.linalg.norm(C - C[i], axis=1) <= radius
        ts = np.sort(T[near & np.isfinite(T)])
        out[i] = 0 if len(ts) == 0 else 1 + int(np.sum(np.diff(ts) > gap_s))
    return out


def choose_stations(centres, times, radius: float, n: int = 4) -> list[dict]:
    """Up to `n` stations on the viewer's path: the START, the MOST-REVISITED
    place, then the rest spread through the walk in time (each at least
    `radius` from the ones already chosen when possible)."""
    C = np.asarray(centres, np.float64)
    T = np.asarray(times, np.float64)
    if len(C) == 0:
        return []
    T = np.where(np.isfinite(T), T, np.arange(len(C), dtype=np.float64))
    chosen: list[dict] = [{"index": 0, "role": "start"}]
    rc = revisit_counts(C, T, radius)
    near_counts = np.array([(np.linalg.norm(C - c, axis=1) <= radius).sum() for c in C])
    order = np.lexsort((-near_counts, -rc))     # most passes, then most cameras
    for i in order:
        if np.linalg.norm(C[i] - C[0]) > radius or len(C) == 1:
            chosen.append({"index": int(i), "role": "most-revisited", "passes": int(rc[i])})
            break
    else:
        chosen[0]["role"] = "start+most-revisited"
    chosen[0]["passes"] = int(rc[0])
    t0, t1 = float(T.min()), float(T.max())
    k = 1
    while len(chosen) < n and k < 4 * n:
        frac = k / (n - 1) if n > 2 else 0.5
        k += 1
        target = t0 + min(frac, 1.0) * (t1 - t0)
        taken = [c["index"] for c in chosen]
        best = None
        for i in np.argsort(np.abs(T - target)):
            if int(i) in taken:
                continue
            d = min(np.linalg.norm(C[i] - C[j]) for j in taken)
            if d > radius:
                best = int(i)
                break
        if best is None:
            continue
        chosen.append({"index": best, "role": f"walk-{int(round(100 * (T[best] - t0) / max(t1 - t0, 1e-9)))}pct",
                       "passes": int(rc[best])})
    while len(chosen) < min(n, len(C)):           # far-apart fill when time spread failed
        taken = [c["index"] for c in chosen]
        d = np.min(np.linalg.norm(C[:, None, :] - C[taken][None], axis=2), axis=1)
        d[taken] = -1
        i = int(np.argmax(d))
        chosen.append({"index": i, "role": "far", "passes": int(rc[i])})
    for c in chosen:
        c["t_s"] = round(float(T[c["index"]]), 2)      # the caller's clock (walk time)
    return chosen


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------


def face_normals_areas(V, F) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    cross = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area2 = np.linalg.norm(cross, axis=1)
    n = cross / np.maximum(area2, 1e-30)[:, None]
    centroid = V[F].mean(axis=1)
    return n, 0.5 * area2, centroid


def orientation_checks(V, F, up, cam_centres=None, metres_per_unit: float | None = None,
                       planes: list | None = None) -> dict:
    """Is the surface upright under `up`? Floor tilt, wall verticality and the
    dominant planes' angles to the vertical. Pure; no store."""
    u = unit(up)
    out: dict = {"up": [round(float(x), 5) for x in u]}
    if V is None or F is None or len(F) == 0:
        out["surface"] = "absent"
        return out
    n, area, cen = face_normals_areas(V, F)
    keep = area > 0
    n, area, cen = n[keep], area[keep], cen[keep]
    total = float(area.sum())
    d = n @ u
    horiz = np.abs(d) > math.cos(math.radians(30))
    wall = np.abs(d) < math.cos(math.radians(60))
    out["area_total"] = round(total, 4)
    out["horizontal_area_fraction"] = round(float(area[horiz].sum() / total), 4)
    out["wall_area_fraction"] = round(float(area[wall].sum() / total), 4)
    if wall.any():
        tilt = np.degrees(np.arcsin(np.clip(np.abs(d[wall]), 0, 1)))
        w = area[wall]
        o = np.argsort(tilt)
        cw = np.cumsum(w[o]) / w.sum()
        out["wall_tilt_deg"] = {
            "area_weighted_mean": round(float((tilt * w).sum() / w.sum()), 2),
            "area_weighted_median": round(float(tilt[o][np.searchsorted(cw, 0.5)]), 2),
            "note": "angle between wall-like faces (|n.up| < cos 60) and the vertical; 0 = plumb"}
    h = cen @ u
    floor_info: dict = {"found": False}
    if cam_centres is not None and len(cam_centres):
        hc = np.asarray(cam_centres, np.float64) @ u
        cam_h = float(np.median(hc))
        p5, p95 = np.percentile(hc, [5, 95])
        spread = {"p5_p95_units": round(float(p95 - p5), 4)}
        if metres_per_unit:
            spread["p5_p95_m"] = round(float(p95 - p5) * metres_per_unit, 3)
        spread["note"] = ("the walk's camera heights along up; a level walk stays within ~0.5 m, more is a "
                          "bent/drifting solve, a wrong up, or a wrong scale")
        out["camera_height_spread"] = spread
        hs = (h - cam_h) * (metres_per_unit or 1.0)
        out["surface_height_vs_cameras"] = {
            "p1_p50_p99": [round(float(x), 3) for x in np.percentile(hs, [1, 50, 99])],
            "unit": "m" if metres_per_unit else "units"}
        floor_info = floor_under_cameras(n, area, cen, d, horiz, u, np.asarray(cam_centres, np.float64),
                                         metres_per_unit, total)
    out["floor"] = floor_info
    if planes:
        rows = []
        for p in planes:
            pn = unit(p.get("normal") or [0, 0, 0])
            a = angle_deg(pn, u)
            a = min(a, 180 - a)
            kind = "horizontal" if a < 20 else "vertical" if a > 70 else "oblique"
            rows.append({"area": p.get("area"), "angle_to_up_deg": round(a, 2), "kind": kind,
                         "off_ideal_deg": round(a if kind == "horizontal" else 90 - a if kind == "vertical"
                                                else min(a, 90 - a), 2)})
        rows.sort(key=lambda r: -(r["area"] or 0))
        out["dominant_planes"] = rows
        big = [r for r in rows if r["kind"] == "oblique"]
        out["oblique_plane_area_fraction"] = round(
            sum(r["area"] or 0 for r in big) / max(sum(r["area"] or 0 for r in rows), 1e-9), 3)
    return out


# The floor is looked for UNDER THE WALK, not anywhere in the fuse box: a fused
# surface can hold stray far geometry (the kitchen walk 9.1: 5.9% of the area ~28
# units below the cameras in a ~75 m box), and a global "lowest dominant
# horizontal" reported the camera 18.3 m above the floor (W9-DIAG, 2026-10-12).
FLOOR_RADIUS_M = 1.5          # horizontal reach from the nearest camera
FLOOR_BAND_M = (0.5, 2.5)     # how far below that camera a floor may be
FLOOR_MODE_TOL_M = 0.15       # the floor is the support peak +- this


def floor_under_cameras(n, area, cen, d, horiz, u, cams, metres_per_unit, total) -> dict:
    """The floor as the plane BELOW THE CAMERAS with the most support near them.

    Each horizontal face is measured against the camera horizontally nearest to it
    (so a drifting or sloped walk is judged locally): it is a floor candidate only
    within FLOOR_RADIUS_M of that camera in plan and FLOOR_BAND_M below it. The
    floor is the area-weighted mode of those relative heights (+-0.15 m); the
    camera height is minus that mode. Without a metric scale the lengths are
    fractions of the walk's own plan extent (radius 25%, mode +-2%) and the band
    has no lower limit: nearness in plan is what excludes stray far geometry."""
    from scipy.spatial import cKDTree

    e1, e2 = plane_basis(u)
    cam_xy = np.c_[cams @ e1, cams @ e2]
    cam_h = cams @ u
    extent = float(max(np.ptp(cam_xy[:, 0]), np.ptp(cam_xy[:, 1]), 1e-9))
    if metres_per_unit:
        radius = FLOOR_RADIUS_M / metres_per_unit
        lo, hi = (b / metres_per_unit for b in FLOOR_BAND_M)
        tol = FLOOR_MODE_TOL_M / metres_per_unit
    else:
        # no scale: no metric band either; "near the walk in plan" is what excludes stray far
        # geometry, and the best-supported plane below the cameras wins
        radius, lo, hi, tol = 0.25 * extent, 0.02 * extent, np.inf, 0.02 * extent
    rule = (f"horizontal faces (|n.up| > cos 30) within {FLOOR_RADIUS_M} m in plan of the nearest camera and "
            f"{FLOOR_BAND_M[0]}-{FLOOR_BAND_M[1]} m below it; floor = area-weighted mode of the height below "
            f"that camera +-{FLOOR_MODE_TOL_M} m" if metres_per_unit else
            "no scale: horizontal faces within 25% of the walk's plan extent of the nearest camera and "
            "any distance (>= 2% of the extent) below it; floor = area-weighted mode +-2% of the extent")
    out: dict = {"found": False, "rule": rule}
    idx = np.flatnonzero(horiz)
    if len(idx) == 0:
        return out
    dist, near = cKDTree(cam_xy).query(np.c_[cen[idx] @ e1, cen[idx] @ e2])
    below = cam_h[near] - cen[idx] @ u                      # how far under its nearest camera
    ok = (dist <= radius) & (below >= lo) & (below <= hi)
    out["candidate_area_fraction"] = round(float(area[idx[ok]].sum() / total), 4)
    if ok.sum() < 20:
        return out
    b, w = below[ok], area[idx[ok]]
    nb = max(1, int(math.ceil((float(b.max() - b.min()) + 1e-12) / (tol / 3))))
    hist, edges = np.histogram(b, bins=nb, weights=w)
    k = int(np.argmax(hist))
    peak = 0.5 * (edges[k] + edges[k + 1])
    sel = np.abs(b - peak) <= tol
    if sel.sum() < 20:
        return out
    fi, fw = idx[ok][sel], w[sel]
    m = (n[fi] * np.sign(d[fi])[:, None] * fw[:, None]).sum(axis=0)
    P = cen[fi]
    mu = (P * fw[:, None]).sum(axis=0) / fw.sum()
    _, _, vt = np.linalg.svd((P - mu) * np.sqrt(fw)[:, None], full_matrices=False)
    pn = vt[-1] if vt[-1] @ u >= 0 else -vt[-1]
    cam_above = float(np.average(b[sel], weights=fw))
    out.update({
        "found": True,
        "area_fraction": round(float(fw.sum() / total), 4),
        "share_of_candidate_area": round(float(fw.sum() / w.sum()), 3),
        "cameras_over_floor": int(len(np.unique(near[ok][sel]))),
        "normal_tilt_deg": round(angle_deg(m, u), 2),
        "plane_fit_tilt_deg": round(angle_deg(pn, u), 2),
        "camera_height_above_floor_units": round(cam_above, 4),
        "tilt_note": ("normal_tilt is the floor faces' own tilt from up; plane_fit_tilt fits ONE plane through "
                      "them, and since each face is chosen relative to its nearest camera, a walk whose camera "
                      "heights drift (camera_height_spread) shows up there as tilt too"),
    })
    if metres_per_unit:
        m_h = cam_above * metres_per_unit
        out["camera_height_above_floor_m"] = round(m_h, 3)
        out["camera_height_plausible"] = bool(0.8 <= m_h <= 2.2)
    return out


# ---------------------------------------------------------------------------
# reading a world (plain file reads, never a store write)
# ---------------------------------------------------------------------------


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_ids(world_root: Path, world: str, session: str | None) -> tuple[str, str]:
    worlds = world_root / "worlds"
    hits = sorted(p.name for p in worlds.iterdir() if p.is_dir() and p.name.startswith(world))
    if len(hits) != 1:
        raise SystemExit(f"world prefix {world!r} matches {len(hits)} worlds")
    wid = hits[0]
    sessions = sorted(p.name for p in (worlds / wid / "sessions").iterdir() if p.is_dir())
    if session:
        sessions = [s for s in sessions if s.startswith(session)]
    if len(sessions) != 1:
        raise SystemExit(f"session {session!r} matches {len(sessions)} sessions of {wid}")
    return wid, sessions[0]


def keyframe_times(session_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    path = session_dir / "keyframes.jsonl"
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            t = row.get("received_at")
            if row.get("keyframe_id") and isinstance(t, (int, float)):
                out[row["keyframe_id"]] = float(t)
    return out


def load_world(world_root: Path, wid: str, sid: str) -> dict:
    wdir = world_root / "worlds" / wid
    solve = wdir / "solve" / sid
    sol = read_json(solve / "solution.json")
    with open(solve / "solution.npz", "rb") as fh:
        raw = fh.read()
    with np.load(io.BytesIO(raw)) as z:
        arrays = {k: z[k] for k in z.files}
    w = {"wid": wid, "sid": sid, "wdir": wdir, "solution": sol, "arrays": arrays,
         "times": keyframe_times(wdir / "sessions" / sid),
         "world_json": read_json(wdir / "world.json")}
    for name in ("components.json", "final_scale_guard.json", "camera.json"):
        p = solve / name
        w[name.split(".")[0]] = read_json(p) if p.exists() else None
    for kind in ("surface", "appearance"):
        p = wdir / kind / sid / "manifest.json"
        w[kind] = read_json(p) if p.exists() else None
    return w


def read_mesh_file(path: Path):
    from tower.world_builder.surface import read_mesh_bytes
    with open(path, "rb") as fh:
        return read_mesh_bytes(fh.read())


def proxy_mesh(w: dict):
    """The mesh the viewer draws (the appearance proxy), else the surface's
    phone level. (V, F, C, N, source)."""
    man = w.get("appearance")
    if man and (man.get("proxy") or {}).get("digest"):
        p = w["wdir"] / "appearance" / w["sid"] / f"p.{man['proxy']['digest']}.bin"
        if p.exists():
            return (*read_mesh_file(p), "appearance-proxy")
    return surface_level(w, prefer=2)


def surface_level(w: dict, prefer: int = 1):
    man = w.get("surface")
    if not man:
        return None, None, None, None, "absent"
    levels = {int(L["level"]): L for L in man.get("levels") or [] if L.get("file")}
    if not levels:
        return None, None, None, None, "absent"
    lv = prefer if prefer in levels else max(levels)
    p = w["wdir"] / "surface" / w["sid"] / levels[lv]["file"]
    if not p.exists():
        return None, None, None, None, "absent"
    return (*read_mesh_file(p), f"surface-l{lv}")


# ---------------------------------------------------------------------------
# components, frames, scale
# ---------------------------------------------------------------------------


def walk_t0(w: dict) -> float:
    ts = list(w["times"].values())
    return min(ts) if ts else 0.0


def spans_from_times(ts: list[float], gap_s: float = 2.5) -> list[list[float]]:
    ts = sorted(ts)
    if not ts:
        return []
    out = [[ts[0], ts[0]]]
    for t in ts[1:]:
        if t - out[-1][1] > gap_s:
            out.append([t, t])
        else:
            out[-1][1] = t
    return [[round(a, 1), round(b, 1)] for a, b in out]


def component_table(w: dict) -> list[dict]:
    """One row per gate component: id, gate index, solver frame, state, kf,
    time spans, whether the viewer shows it."""
    sol = w["solution"]
    poses = sol.get("poses") or {}
    t0 = walk_t0(w)
    solver_of = {}
    for c in sol.get("components") or []:
        sc = c.get("solver_components") or [c["index"]]
        solver_of[int(c["index"])] = int(sc[0])
    by_gate: dict[int, list[str]] = {}
    for kid, p in poses.items():
        try:
            by_gate.setdefault(int(p.get("component")), []).append(kid)
        except (TypeError, ValueError):
            continue
    rows = []
    comps = (w.get("components") or {}).get("components")
    if comps:
        for c in comps:
            votes: dict[int, int] = {}
            for kid in c.get("keyframe_ids") or []:
                p = poses.get(kid)
                if p is not None and p.get("component") is not None:
                    votes[int(p["component"])] = votes.get(int(p["component"]), 0) + 1
            gate = max(votes, key=votes.get) if votes else None
            rows.append({"id": c["id"], "gate_index": gate, "state": c.get("state"),
                         "reason": c.get("reason"), "shown_as": c.get("shown_as"),
                         "keyframes": c.get("keyframes"),
                         "capture_spans_s": c.get("capture_spans_s") or [],
                         "final_scale_guard": c.get("final_scale_guard")})
    else:
        for c in sol.get("components") or []:
            g = int(c["index"])
            ts = [w["times"][k] - t0 for k in by_gate.get(g, []) if k in w["times"]]
            rows.append({"id": f"c{g}", "gate_index": g,
                         "state": c.get("gate_state") or ("solve-only" if g == 0 else "unknown"),
                         "reason": (c.get("gate_reasons") or [None])[0],
                         "shown_as": "room" if g == 0 else "area",
                         "keyframes": c.get("images_supported", c.get("images")),
                         "capture_spans_s": spans_from_times(ts), "final_scale_guard": None})
    for r in rows:
        g = r["gate_index"]
        r["solver_frame"] = solver_of.get(g, g)
        r["viewer_shows"] = (g == 0)
        r["kids"] = by_gate.get(g, [])
        sp = r["capture_spans_s"]
        r["time_span_s"] = [sp[0][0], sp[-1][1]] if sp else None
    return rows


def guard_scale(w: dict) -> dict:
    g = w.get("final_scale_guard")
    if not g:
        return {"present": False}
    a = g.get("assessment") or {}
    mpu = (a.get("pose") or {}).get("metres_per_unit")
    ref = (a.get("reference") or {}).get("log")
    if mpu is None and isinstance(ref, (int, float)):
        mpu = math.exp(-ref)
    pieces = []
    for p in a.get("pieces") or []:
        pieces.append({k: p.get(k) for k in ("first_keyframe_id", "last_keyframe_id", "keyframes",
                                            "ratio", "ci", "decision", "reason", "class")}
                      | {"keyframe_ids": p.get("keyframe_ids") or []})
    return {"present": True, "decision": g.get("decision"), "scale_status": a.get("scale_status"),
            "why": a.get("why"), "evidence": (g.get("inputs") or {}).get("source"),
            "metres_per_unit_room": mpu, "reference_log": ref,
            "uncertified_keyframes": a.get("uncertified_keyframes"), "pieces": pieces}


def depth_scale_estimates(w: dict, max_obs_per_kf: int = 400) -> dict:
    """Per keyframe log(z_sfm / z_metric) from the live-depth cache (read in
    place, read-only) at the solve's own observations; the guard's own
    quantity, measured for EVERY component -- the guard only measures the room.
    Returns {kid: log ratio}."""
    align = w["wdir"] / "dense" / w["sid"] / "live_depth" / "align.json"
    if not align.exists():
        return {}
    try:
        recs = read_json(align).get("records") or []
    except (OSError, ValueError):
        return {}
    pred_of = {r["kid"]: align.parent / r["pred"] for r in recs if r.get("kid") and r.get("pred")}
    sol, A = w["solution"], w["arrays"]
    kids = sol.get("keyframe_ids") or []
    obs, oxy, xyz = A.get("observations"), A.get("observation_xy"), A.get("xyz")
    if obs is None or oxy is None or len(obs) == 0:
        return {}
    order = np.argsort(obs[:, 0], kind="stable")
    obs, oxy = obs[order], oxy[order]
    starts = np.searchsorted(obs[:, 0], np.arange(len(kids) + 1))
    out = {}
    for ki, kid in enumerate(kids):
        pose = (sol.get("poses") or {}).get(kid)
        path = pred_of.get(kid)
        if pose is None or path is None or not path.exists():
            continue
        Rt = pose_arrays(pose)
        if Rt is None:
            continue
        a, b = starts[ki], starts[ki + 1]
        if b <= a:
            continue
        sel = np.arange(a, b)
        if len(sel) > max_obs_per_kf:
            sel = sel[np.linspace(0, len(sel) - 1, max_obs_per_kf).astype(int)]
        try:
            pred = np.load(path).astype(np.float32)
        except (OSError, ValueError):
            continue
        X = xyz[obs[sel, 2]].astype(np.float64)
        z = (X @ Rt[0].T + Rt[1])[:, 2]
        uv = np.rint(oxy[sel]).astype(int)
        hh, ww = pred.shape[:2]
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < ww) & (uv[:, 1] >= 0) & (uv[:, 1] < hh) & (z > 0)
        if ok.sum() < 20:
            continue
        zm = pred[uv[ok, 1], uv[ok, 0]]
        good = np.isfinite(zm) & (zm > 0)
        if good.sum() < 20:
            continue
        out[kid] = float(np.median(np.log(z[ok][good] / zm[good])))
    return out


def scale_by_component(rows: list[dict], kf_logs: dict, guard: dict) -> list[dict]:
    room_logs = [kf_logs[k] for r in rows if r["gate_index"] == 0 for k in r["kids"] if k in kf_logs]
    room_med = float(np.median(room_logs)) if room_logs else None
    out = []
    for r in rows:
        logs = np.array([kf_logs[k] for k in r["kids"] if k in kf_logs])
        row = {"id": r["id"], "gate_index": r["gate_index"], "solver_frame": r["solver_frame"],
               "state": r["state"], "reason": r["reason"], "keyframes_measured": int(len(logs))}
        if len(logs) >= 3:
            med = float(np.median(logs))
            row["metres_per_unit_est"] = round(math.exp(-med), 5)
            row["log_iqr"] = round(float(np.subtract(*np.percentile(logs, [75, 25]))), 4)
            if room_med is not None:
                ratio = math.exp(med - room_med)       # >1: this part is bigger in units than the room
                row["ratio_to_room"] = round(ratio, 4)
                row["consistent_with_room"] = bool(0.8 <= ratio <= 1.25)
        # the guard's own verdict
        if not guard.get("present"):
            row["guard"] = "no final scale guard record for this solve"
        elif r["gate_index"] == 0:
            row["guard"] = {"scale_status": guard.get("scale_status"),
                            "metres_per_unit": guard.get("metres_per_unit_room"),
                            "uncertified_keyframes": guard.get("uncertified_keyframes")}
        else:
            kids = set(r["kids"])
            hit = [p for p in guard.get("pieces") or [] if kids & set(p["keyframe_ids"])]
            row["guard"] = ({"decision": hit[0]["decision"], "reason": hit[0]["reason"],
                             "ratio_to_room": hit[0]["ratio"], "ci": hit[0]["ci"]} if hit else
                            "not assessed (the guard measures the room solve only)")
        if isinstance(row.get("guard"), dict) and row["guard"].get("ratio_to_room") and row.get("ratio_to_room"):
            row["agrees_with_guard"] = bool(abs(math.log(row["ratio_to_room"] / row["guard"]["ratio_to_room"]))
                                            < math.log(1.25))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# the viewer path
# ---------------------------------------------------------------------------


def frame_cameras(w: dict, frame: int, rows: list[dict]) -> dict:
    """kid -> (centre, forward, up, t) for every posed keyframe in a solver frame."""
    gates = {r["gate_index"] for r in rows if r["solver_frame"] == frame}
    t0 = walk_t0(w)
    out = {}
    for kid in w["solution"].get("keyframe_ids") or []:
        p = (w["solution"].get("poses") or {}).get(kid)
        if p is None or p.get("component") is None or int(p["component"]) not in gates:
            continue
        Rt = pose_arrays(p)
        if Rt is None:
            continue
        t = w["times"].get(kid)
        out[kid] = (camera_centre(*Rt), camera_forward(Rt[0]), camera_up(Rt[0]),
                    None if t is None else t - t0, int(p["component"]))
    return out


def match_path_times(path: list, cams: dict) -> list[tuple[str | None, float | None]]:
    """For each viewer-path entry [cx, cy, cz, fx, fy, fz]: the keyframe whose
    centre it is, and its walk time."""
    if not cams:
        return [(None, None)] * len(path)
    kids = list(cams)
    C = np.array([cams[k][0] for k in kids])
    out = []
    for e in path:
        d = np.linalg.norm(C - np.asarray(e[:3], np.float64), axis=1)
        i = int(np.argmin(d))
        out.append((kids[i], cams[kids[i]][3]) if d[i] < 1e-3 else (None, None))
    return out


# ---------------------------------------------------------------------------
# the stage copy and the throwaway server
# ---------------------------------------------------------------------------


def stage_world(world_root: Path, stage_root: Path, wid: str, sid: str) -> Path:
    """Copy one world (without its COLMAP database, solver images or the
    live-depth work tree) to `stage_root/world_builder/worlds/<wid>`. Reused
    when the stage already holds this build (same solution.json bytes)."""
    src = world_root / "worlds" / wid
    dst_root = stage_root / "world_builder"
    dst = dst_root / "worlds" / wid
    marker = dst / "solve" / sid / "solution.json"
    if marker.exists() and marker.read_bytes() == (src / "solve" / sid / "solution.json").read_bytes() \
            and (dst / "appearance").exists() == (src / "appearance").exists():
        return dst_root
    if dst.exists():
        shutil.rmtree(dst)                       # our own stale stage copy, never the store

    def ignore(directory, names):
        rel = Path(directory).relative_to(src).parts
        skip = set(shutil.ignore_patterns(*STAGE_IGNORE)(directory, names))
        for name in names:
            full = (*rel, name)
            if any(len(full) == len(pat) and all(fnmatch.fnmatch(a, b) for a, b in zip(full, pat))
                   for pat in STAGE_SKIP_DIRS):
                skip.add(name)
        return skip

    shutil.copytree(src, dst, ignore=ignore)
    return dst_root


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


class PackServer:
    """The geometry router (and nothing else) of a Tower, on 127.0.0.1:<port>,
    over the STAGED world root, plus one route serving the appearance page
    with the phone's layer cap."""

    def __init__(self, stage_world_root: Path, port: int, rgba8_cap: int | None):
        if port == LIVE_TOWER_PORT:
            raise SystemExit("refusing port 8000: that is the live Tower")
        if not port_free(port):
            raise SystemExit(f"port {port} is in use; pass --port")
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse

        from tower.results.world_builder_geometry import store_from_root
        from tower.results.world_builder_render import (
            WorldRenderUnavailable,
            build_world_render,
            render_content_security_policy,
        )
        from tower.routes import geometry

        app = FastAPI()
        app.state.world_root = str(stage_world_root)
        app.include_router(geometry.router)
        self.patched = None

        @app.get("/pack/{world_id}/{session_id}/appearance.html", response_class=HTMLResponse)
        def pack_page(world_id: str, session_id: str):
            try:
                html = build_world_render(store_from_root(str(stage_world_root)), world_id, session_id,
                                          representation="appearance", transport="tower",
                                          viewer="appearance-1")
            except WorldRenderUnavailable as exc:
                raise HTTPException(status_code=404, detail=exc.reason) from None
            if rgba8_cap:
                html, n = re.subn(r'("rgba8_layer_cap":\s*)\d+', r"\g<1>" + str(int(rgba8_cap)), html)
                self.patched = n
            return HTMLResponse(html, headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": render_content_security_policy(html, "tower")})

        self.port = port
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                                    log_level="warning", lifespan="off"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 30
        while not self.server.started and time.time() < deadline:
            time.sleep(0.1)
        if not self.server.started:
            raise RuntimeError("the pack server did not start")
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)


# ---------------------------------------------------------------------------
# headless Chromium over the DevTools protocol
# ---------------------------------------------------------------------------


class Browser:
    def __init__(self, exe: str, profile: Path, width: int, height: int):
        import warnings

        from websockets.sync.client import connect
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"connect\(\) must be used")

        profile.mkdir(parents=True, exist_ok=True)
        port_file = profile / "DevToolsActivePort"
        if port_file.exists():
            port_file.unlink()
        self.proc = subprocess.Popen(
            [exe, "--headless=new", "--disable-gpu", "--enable-unsafe-swiftshader", "--use-gl=angle",
             "--use-angle=swiftshader", "--hide-scrollbars", "--no-first-run",
             "--no-default-browser-check", "--disable-extensions", "--mute-audio",
             f"--window-size={width},{height}", "--remote-debugging-port=0",
             "--remote-allow-origins=*", f"--user-data-dir={profile}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 30
        while time.time() < deadline and not (port_file.exists() and port_file.read_text().strip()):
            time.sleep(0.2)
        if not port_file.exists():
            self.close()
            raise RuntimeError("the browser did not open a DevTools port")
        port = int(port_file.read_text().split()[0])
        targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10).read())
        page = next(t for t in targets if t.get("type") == "page")
        self.ws = connect(page["webSocketDebuggerUrl"], max_size=None, open_timeout=30)
        self.mid = 0
        self.call("Emulation.setDeviceMetricsOverride",
                  {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})

    def call(self, method: str, params: dict | None = None, timeout: float = 120.0):
        self.mid += 1
        mid = self.mid
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError(method)
            msg = json.loads(self.ws.recv(timeout=left))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result") or {}

    def eval(self, expr: str, timeout: float = 120.0):
        r = self.call("Runtime.evaluate", {"expression": expr, "returnByValue": True,
                                           "awaitPromise": True}, timeout)
        if "exceptionDetails" in r:
            raise RuntimeError(f"page error: {json.dumps(r['exceptionDetails'])[:600]}")
        return (r.get("result") or {}).get("value")

    def close(self):
        try:
            self.call("Browser.close", timeout=5)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def find_browser(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    return next((c for c in CHROME_CANDIDATES if Path(c).exists()), None)


def data_url_png(u: str) -> bytes:
    return base64.b64decode(u.split(",", 1)[1])


# ---------------------------------------------------------------------------
# fallback: a CPU splat of the surface's own vertex colours
# ---------------------------------------------------------------------------


def splat_view(V, C, eye, forward, up, width: int, height: int, hfov_deg: float = 57.0) -> np.ndarray:
    u = unit(up)
    f = unit(forward - u * (forward @ u))
    r = np.cross(f, u)
    P = np.asarray(V, np.float64) - np.asarray(eye, np.float64)
    x, y, z = P @ r, P @ u, P @ f
    near = 0.05 * max(float(np.median(np.abs(z))), 1e-6)
    keep = z > near
    x, y, z, col = x[keep], y[keep], z[keep], np.asarray(C)[keep]
    focal = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    px = np.rint(focal * x / z + width / 2).astype(int)
    py = np.rint(-focal * y / z + height / 2).astype(int)
    img = np.full((height, width, 3), 24, np.uint8)
    order = np.argsort(-z)                  # far first, near overwrites
    px, py, col = px[order], py[order], col[order]
    for dx in (0, 1):
        for dy in (0, 1):
            qx, qy = px + dx, py + dy
            ok = (qx >= 0) & (qx < width) & (qy >= 0) & (qy < height)
            img[qy[ok], qx[ok]] = col[ok]
    return img


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


def fmt_t(s: float | None) -> str:
    if s is None:
        return "?"
    s = max(0.0, float(s))
    return f"{int(s // 60)}:{int(s % 60):02d}"


def nice_bar(span_units: float, mpu: float | None) -> tuple[float, str]:
    if mpu:
        target_m = span_units * mpu / 5
        for m in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 50):
            if m >= target_m:
                break
        return m / mpu, f"{m:g} m"
    target = span_units / 5
    mag = 10 ** math.floor(math.log10(max(target, 1e-9)))
    step = next(k * mag for k in (1, 2, 5, 10) if k * mag >= target)
    return step, f"{step:g} units (scale unknown)"


COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
           "#9a6324", "#808000", "#000075", "#469990", "#bfef45"]


def draw_plan(path: Path, w: dict, rows: list[dict], scales: list[dict], up_by_frame: dict,
              surface_xyz, stations: list[dict], path_cams: list, label: str, mpu_room: float | None,
              mpu_note: str) -> None:
    import logging

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    from matplotlib.collections import LineCollection

    frames = sorted({r["solver_frame"] for r in rows if r["solver_frame"] is not None},
                    key=lambda f: (f != 0, f))
    others = [f for f in frames if f != 0]
    ncol = 1 if not others else 2
    fig = plt.figure(figsize=(16 if others else 12, 11), dpi=110)
    gs = fig.add_gridspec(max(1, len(others)), ncol, width_ratios=[3, 1] if others else [1],
                          hspace=0.35, wspace=0.12)
    A = w["arrays"]
    xyz, comp = A["xyz"].astype(np.float64), A["component"]
    scale_of = {s["gate_index"]: s for s in scales}
    colour_of = {r["gate_index"]: COLOURS[i % len(COLOURS)] for i, r in enumerate(rows)}
    t_max = max([c[3] for f in frames for c in frame_cameras(w, f, rows).values() if c[3] is not None] or [1])

    def panel(ax, frame, main):
        up = up_by_frame.get(frame)
        e1, e2 = plane_basis(up)
        gates = [r for r in rows if r["solver_frame"] == frame]
        handles = []
        allpts = []
        must = []
        if main and surface_xyz is not None and len(surface_xyz):
            S = surface_xyz[:: max(1, len(surface_xyz) // 120000)]
            ax.scatter(S @ e1, S @ e2, s=0.15, c="#9a9a9a", alpha=0.25, linewidths=0, rasterized=True)
            handles.append(plt.Line2D([], [], ls="", marker="s", color="#9a9a9a",
                                      label=f"fused surface ({len(surface_xyz):,} verts)"))
        for r in gates:
            m = (comp == r["gate_index"]) & np.isfinite(xyz).all(axis=1)
            P = xyz[m]
            allpts.append(P)
            ax.scatter(P @ e1, P @ e2, s=0.6 if main else 1.0, c=colour_of[r["gate_index"]],
                       alpha=0.55, linewidths=0, rasterized=True)
            sc = scale_of.get(r["gate_index"], {})
            gtxt = sc.get("guard")
            gtxt = (f"guard {gtxt.get('scale_status') or gtxt.get('decision')}"
                    + (f" x{gtxt['ratio_to_room']:.2f}" if gtxt.get("ratio_to_room") else "")
                    if isinstance(gtxt, dict) else "guard n/a")
            est = (f", depth-est x{sc['ratio_to_room']:.2f}" if sc.get("ratio_to_room") is not None else "")
            spans = r["capture_spans_s"]
            span_txt = f"{fmt_t(spans[0][0])}-{fmt_t(spans[-1][1])} ({len(spans)} span{'s' if len(spans) != 1 else ''})" \
                if spans else "no times"
            tag = "  << VIEWER SHOWS" if r["viewer_shows"] else ""
            handles.append(plt.Line2D([], [], ls="", marker="o", color=colour_of[r["gate_index"]],
                                      label=f"{r['id'][:8]} {r['state']}/{r['reason'] or r['shown_as']}: "
                                            f"{r['keyframes']} kf, {span_txt}; {gtxt}{est}{tag}"))
        cams = frame_cameras(w, frame, rows)
        if cams:
            C = np.array([c[0] for c in cams.values()])
            T = np.array([c[3] if c[3] is not None else np.nan for c in cams.values()])
            G = np.array([c[4] for c in cams.values()])
            xy = np.c_[C @ e1, C @ e2]
            # the walk is drawn only between consecutive cameras of ONE component: a
            # line from the room into an isolated piece is a jump between scales,
            # not a path anybody walked
            same = G[:-1] == G[1:]
            seg = np.stack([xy[:-1], xy[1:]], axis=1)[same]
            lc = LineCollection(seg, cmap="viridis", linewidths=1.4 if main else 1.0, alpha=0.9)
            lc.set_array(np.nan_to_num(T[:-1][same], nan=0.0))
            lc.set_clim(0, t_max)
            ax.add_collection(lc)
            ax.plot(*xy[0], marker="^", ms=12, color="lime", mec="k", zorder=5)
            ax.plot(*xy[-1], marker="s", ms=10, color="red", mec="k", zorder=5)
            handles.append(plt.Line2D([], [], ls="", marker="^", color="lime", mec="k",
                                      label=f"start of this part ({fmt_t(T[0])})"))
            handles.append(plt.Line2D([], [], ls="", marker="s", color="red", mec="k",
                                      label=f"end of this part ({fmt_t(T[-1])})"))
            allpts.append(C)
            if main:
                cb = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.01)
                cb.set_label("walk time (s)")
        if main:
            for k, s in enumerate(stations):
                e = path_cams[s["index"]]
                p = np.asarray(e[:3])
                must.append(p)                     # a station is never off the plot
                f = np.asarray(e[3:6])
                fx, fy = f @ e1, f @ e2
                nrm = math.hypot(fx, fy) or 1
                ax.plot(p @ e1, p @ e2, marker="o", ms=16, mfc="white", mec="black", zorder=6)
                ax.text(p @ e1, p @ e2, str(k), ha="center", va="center", fontsize=9, weight="bold", zorder=7)
                L = 0.06 * span
                ax.annotate("", xy=(p @ e1 + L * fx / nrm, p @ e2 + L * fy / nrm), xytext=(p @ e1, p @ e2),
                            arrowprops=dict(arrowstyle="->", lw=2, color="black"), zorder=6)
            handles.append(plt.Line2D([], [], ls="", marker="o", mfc="white", mec="black", ms=10,
                                      label="turnaround stations (number, heading = yaw 0)"))
        if allpts:
            P = np.vstack(allpts)
            X, Y = P @ e1, P @ e2
            lo = np.percentile(np.c_[X, Y], 1, axis=0)
            hi = np.percentile(np.c_[X, Y], 99, axis=0)
            if must:
                M = np.array(must)
                lo = np.minimum(lo, np.c_[M @ e1, M @ e2].min(axis=0))
                hi = np.maximum(hi, np.c_[M @ e1, M @ e2].max(axis=0))
            pad = 0.08 * float(max(hi - lo))
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        return handles

    ax = fig.add_subplot(gs[:, 0])
    cams0 = frame_cameras(w, 0, rows)
    C0 = np.array([c[0] for c in cams0.values()]) if cams0 else xyz[comp == 0]
    e1, e2 = plane_basis(up_by_frame.get(0))
    span = float(max(np.ptp(C0 @ e1), np.ptp(C0 @ e2), 1e-6)) * 1.5 if len(C0) else 1.0
    handles = panel(ax, 0, True)
    L, txt = nice_bar(span, mpu_room)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx, by = x0 + 0.04 * (x1 - x0), y0 + 0.04 * (y1 - y0)
    ax.plot([bx, bx + L], [by, by], color="black", lw=4)
    ax.text(bx, by + 0.015 * (y1 - y0), f"{txt}{' ' + mpu_note if mpu_note and mpu_room else ''}", fontsize=9)
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.85)
    ax.set_title(f"{label}  world {w['wid'][:8]} / session {w['sid'][:8]} -- top-down, ROOM frame "
                 f"(viewer up), all components solved in it", fontsize=10)
    for i, f in enumerate(others):
        axo = fig.add_subplot(gs[i, 1])
        h = panel(axo, f, False)
        rr = [r for r in rows if r["solver_frame"] == f]
        sc = [s for s in scales if s["solver_frame"] == f and s.get("metres_per_unit_est")]
        if sc:
            pts = np.vstack([xyz[comp == r["gate_index"]] for r in rr])
            ee1, ee2 = plane_basis(up_by_frame.get(f))
            sp = float(max(np.ptp(pts @ ee1), np.ptp(pts @ ee2), 1e-6))
            Lb, tb = nice_bar(sp, sc[0]["metres_per_unit_est"])
            xa, xb = axo.get_xlim()
            ya, yb = axo.get_ylim()
            axo.plot([xa + 0.05 * (xb - xa), xa + 0.05 * (xb - xa) + Lb], [ya + 0.06 * (yb - ya)] * 2,
                     color="black", lw=3)
            axo.text(xa + 0.05 * (xb - xa), ya + 0.09 * (yb - ya), tb + " (depth est.)", fontsize=7)
        axo.set_title("OWN FRAME (not placed in the room): " + "; ".join(
            f"{r['id'][:8]} {r['keyframes']}kf" for r in rr), fontsize=7.5)
        axo.legend(handles=h[:len(rr)], loc="upper left", fontsize=5.5, framealpha=0.8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def contact_sheet(images: list[tuple[np.ndarray, str]], cols: int, title: str, path: Path,
                  cell_w: int | None = None) -> None:
    from PIL import Image, ImageDraw

    if not images:
        return
    h0, w0 = images[0][0].shape[:2]
    if cell_w:
        scale = cell_w / w0
        w0, h0 = cell_w, int(round(h0 * scale))
    rows = (len(images) + cols - 1) // cols
    top = 26
    sheet = Image.new("RGB", (cols * w0, top + rows * (h0 + 18)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), title, fill=(0, 0, 0))
    for i, (img, cap) in enumerate(images):
        im = Image.fromarray(img[..., :3])
        if im.size != (w0, h0):
            im = im.resize((w0, h0))
        x, y = (i % cols) * w0, top + (i // cols) * (h0 + 18)
        sheet.paste(im, (x, y + 18))
        d.text((x + 4, y + 3), cap, fill=(0, 0, 0))
    sheet.save(path)


def png_array(data: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


# ---------------------------------------------------------------------------
# the pack
# ---------------------------------------------------------------------------


def build_pack(args) -> dict:
    t_start = time.time()
    wid, sid = resolve_ids(args.world_root, args.world, args.session)
    out = args.out_root / f"{wid}-{sid}"
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[render-pack {time.time() - t_start:6.1f}s] {m}", flush=True)  # noqa: E731
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        os.environ[k] = v
    w = load_world(args.world_root, wid, sid)
    rows = component_table(w)
    guard = guard_scale(w)
    log(f"world {wid} session {sid}: {len(rows)} components, {len(w['arrays']['xyz'])} sparse points")

    stage = stage_world(args.world_root, args.stage_root, wid, sid)
    log(f"staged copy at {stage}")
    from tower.world_builder.store import WorldStore
    from tower.world_builder.surface_render import _camera_path, _camera_up, _world_up
    sstore = WorldStore(stage)

    V, F, _C, _N, proxy_src = proxy_mesh(w)
    up_viewer = _world_up(sstore, wid, sid, V, F) if V is not None else _camera_up(sstore, wid, sid)
    up_cams = _camera_up(sstore, wid, sid)
    path_py = _camera_path(sstore, wid, sid)
    up_by_frame = {0: up_viewer}
    for f in {r["solver_frame"] for r in rows} - {0}:
        ups = [c[2] for c in frame_cameras(w, f, rows).values()]
        up_by_frame[f] = unit(np.mean(ups, axis=0)).tolist() if ups else [0, -1, 0]

    kf_logs = depth_scale_estimates(w) if not args.no_depth_scale else {}
    scales = scale_by_component(rows, kf_logs, guard)
    mpu_room = guard.get("metres_per_unit_room")
    mpu_note = f"(guard: {guard.get('scale_status')})" if mpu_room else ""
    if not mpu_room:
        est = next((s.get("metres_per_unit_est") for s in scales if s["gate_index"] == 0), None)
        if est:
            mpu_room, mpu_note = est, "(live-depth estimate, no guard)"
    log(f"up (viewer) {np.round(up_viewer, 3).tolist() if up_viewer is not None else None}; "
        f"room m/unit {mpu_room} {mpu_note}; depth-measured keyframes {len(kf_logs)}")

    cams0 = frame_cameras(w, 0, rows)

    def stations_for(path):
        pt = match_path_times(path, cams0)
        times = np.array([t if t is not None else np.nan for _, t in pt], np.float64)
        centres = np.array([e[:3] for e in path], np.float64)
        radius = (0.6 / mpu_room) if mpu_room else 0.08 * float(np.ptp(centres, axis=0).max() + 1e-9)
        st = choose_stations(centres, times, radius, args.stations)
        for s in st:
            s["keyframe_id"], _ = pt[s["index"]]
            s["centre"] = [round(float(x), 4) for x in path[s["index"]][:3]]
        return st, radius

    stations, radius = stations_for(path_py)
    surface_xyz = surface_level(w, prefer=1)[0]
    if surface_xyz is None:
        surface_xyz = V
    plan = out / "plan_topdown.png"
    draw_plan(plan, w, rows, scales, up_by_frame, surface_xyz, stations, path_py, args.label, mpu_room, mpu_note)
    log(f"plan -> {plan}")

    pack = {
        "format": PACK_FORMAT, "label": args.label, "world_id": wid, "session_id": sid,
        "world_root": str(args.world_root), "stage_root": str(stage), "out_dir": str(out),
        "files": {"plan": plan.name}, "components": [
            {k: r[k] for k in ("id", "gate_index", "solver_frame", "state", "reason", "shown_as",
                               "keyframes", "capture_spans_s", "time_span_s", "viewer_shows")} for r in rows],
        "scale": {"guard": {k: v for k, v in guard.items() if k != "pieces"}
                  | {"pieces": [{k: v for k, v in p.items() if k != "keyframe_ids"} for p in guard.get("pieces") or []]},
                  "metres_per_unit_used": mpu_room, "metres_per_unit_note": mpu_note,
                  "per_component": scales,
                  "depth_method": "per keyframe median log(z_sfm / z_live_depth) at the solve's observations "
                                  "(the guard's quantity), median per component; ratio_to_room > 1 means the "
                                  "part is LARGER in solve units per metre than the room (it would be drawn "
                                  "too big if placed at the room's scale)"},
        "env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith("TOWER_WORLD")},
    }
    planes = (((w.get("surface") or {}).get("detail") or {}).get("plane_snap") or {}).get("planes")
    pack["orientation"] = orientation_checks(V, F, up_viewer, [c[0] for c in cams0.values()],
                                             mpu_room, planes)
    pack["orientation"]["mesh"] = proxy_src
    pack["orientation"]["up_cameras"] = up_cams
    if up_cams is not None and up_viewer is not None:
        pack["orientation"]["up_viewer_vs_cameras_deg"] = round(angle_deg(up_viewer, up_cams), 2)
    ups = np.array([c[2] for c in cams0.values()])
    if len(ups) and up_viewer is not None:
        tilt = np.degrees(np.arccos(np.clip(ups @ unit(up_viewer), -1, 1)))
        pack["orientation"]["camera_roll_pitch_vs_up_deg"] = {
            "median": round(float(np.median(tilt)), 2), "p90": round(float(np.percentile(tilt, 90)), 2),
            "note": "angle between each room camera's up (-y) and the viewer's up: head tilt + any frame tilt"}
    write_pack(out, pack)

    method = args.method
    exe = find_browser(args.browser) if method in ("auto", "browser") else None
    if method == "browser" and exe is None:
        raise SystemExit("no Chrome/Edge found for --method browser")
    used = None
    if exe and not args.plan_only:
        try:
            pack["turnarounds"] = turnarounds_browser(args, exe, w, stage, out, stations, path_py, pack, log)
            used = "browser"
        except Exception as exc:  # noqa: BLE001 -- the fallback is the point
            log(f"browser method failed: {exc!r}")
            pack.setdefault("warnings", []).append(f"browser method failed: {exc!r}")
            if method == "browser":
                raise
    if used is None and not args.plan_only:
        pack["turnarounds"] = turnarounds_splat(args, w, out, stations, path_py, up_viewer, log)
        used = "splat"
    pack["method"] = used or "plan-only"
    if used == "browser" and pack["turnarounds"].get("path_matches_python") is False:
        # the page walked a different path than this process computed: redraw the
        # plan with the page's own stations
        draw_plan(plan, w, rows, scales, up_by_frame, surface_xyz, pack["turnarounds"]["stations"],
                  pack["turnarounds"]["path"], args.label, mpu_room, mpu_note)
    pack.get("turnarounds", {}).pop("path", None)
    pack["stations_radius_units"] = round(radius, 4)
    pack["seconds"] = round(time.time() - t_start, 1)
    write_pack(out, pack)
    log(f"pack -> {out / 'pack.json'} ({pack['method']})")
    return pack


def write_pack(out: Path, pack: dict) -> None:
    tmp = out / "pack.json.tmp"
    tmp.write_text(json.dumps(pack, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, out / "pack.json")


def yaw_label(k: int, step: float) -> str:
    deg = int(round(k * step)) % 360
    named = {0: "ahead", 90: "left?", 180: "BEHIND", 270: "right?"}
    return f"yaw+{deg:03d}" + (f" {named[deg]}" if deg in (0, 180) else "")


def turnarounds_browser(args, exe, w, stage, out, stations, path_py, pack, log) -> dict:
    wid, sid = w["wid"], w["sid"]
    man = w.get("appearance") or {}
    phone = sum(1 for k in man.get("keyframes") or [] if k.get("tier") == "phone")
    cap = max(192, phone) if args.phone_layers else None
    with PackServer(stage, args.port, cap) as srv:
        url = f"http://127.0.0.1:{args.port}/pack/{wid}/{sid}/appearance.html"
        b = Browser(exe, args.stage_root / "chrome-profile", args.width, args.height)
        try:
            b.call("Page.enable")
            b.call("Page.navigate", {"url": url})
            t0 = time.time()
            state = None
            while time.time() - t0 < args.load_timeout:
                state = b.eval("(()=>{const S=window.__wbAppearance; if(!S) return null;"
                               "return {phase:S.phase, nav:S.navReady?S.navReady():null, layers:S.layers,"
                               "capacity:S.capacity, encoding:S.encoding, errors:S.errors}})()")
                if state and state.get("phase") == "failed":
                    raise RuntimeError(f"viewer failed: {state.get('errors')}")
                if state and state.get("phase") == "ready" and state.get("nav") is not False:
                    break
                time.sleep(1.0)
            else:
                raise TimeoutError(f"viewer not ready after {args.load_timeout}s: {state}")
            log(f"viewer ready in {time.time() - t0:.0f}s: {state}")
            cfg = b.eval("({cameras: CONFIG.cameras, up: CONFIG.up, rgba8: CONFIG.rgba8_layer_cap,"
                         " kp: CONFIG.keyframes_phone, kt: CONFIG.keyframes_total, scale: CONFIG.scale_state})")
            shot = b.call("Page.captureScreenshot", {"format": "png"})
            (out / "viewer_opening.png").write_bytes(base64.b64decode(shot["data"]))
            opening = b.eval("window.__wbAppearance.pose()")
            path = cfg["cameras"] or path_py
            same = len(path) == len(path_py) and bool(np.allclose(np.asarray(path, float),
                                                                    np.asarray(path_py, float), atol=1e-4))
            if not same:
                stations = choose_from_page(path, w, stations, pack)
            step = 360.0 / args.views
            sheets, allcells = [], []
            for k, s in enumerate(stations):
                base = b.eval(f"(()=>{{const S=window.__wbAppearance; S.walk({s['index']},0,0);"
                              f" return S.pose();}})()")
                views = []
                cells = []
                for j in range(args.views):
                    yaw = base["yaw"] + math.radians(j * step)
                    r = b.eval(
                        "(()=>{const S=window.__wbAppearance;"
                        f"S.setPose({{p:{json.dumps(base['p'])}, yaw:{yaw}, pitch:{math.radians(args.pitch_deg)}}});"
                        "const u=S.snapshot(); return {u, pose:S.pose(), cov:S.coverage(), dark:S.dark||null,"
                        " choice:S.choice};})()", timeout=180)
                    png = data_url_png(r["u"])
                    name = f"turnaround_s{k}_yaw{int(round(j * step)):03d}.png"
                    (out / name).write_bytes(png)
                    cov = r.get("cov") or {}
                    moved = float(np.linalg.norm(np.asarray(r["pose"]["p"]) - np.asarray(base["p"])))
                    views.append({"file": name, "yaw_offset_deg": round(j * step, 1), "coverage": cov,
                                  "eye_moved_units": round(moved, 4), "sources": r.get("choice")})
                    img = png_array(png)
                    caption = f"{yaw_label(j, step)}  drawn {100 * cov.get('drawn', 0):.0f}%"
                    cells.append((img, caption))
                    allcells.append((img, f"s{k} {caption}"))
                drawn = [v["coverage"].get("drawn", 0) for v in views]
                s.update({"base_pose": base, "views": views,
                          "drawn_mean": round(float(np.mean(drawn)), 3),
                          "drawn_min": round(float(np.min(drawn)), 3),
                          "drawn_behind": round(float(views[len(views) // 2]["coverage"].get("drawn", 0)), 3)})
                sheet = out / f"turnaround_s{k}.png"
                contact_sheet(cells, 4, f"station {k} ({s['role']}, t={fmt_t(s.get('t_s'))}, path index "
                                        f"{s['index']}) -- appearance viewer, yaw steps {step:g} deg, pitch "
                                        f"{args.pitch_deg:g}", sheet, cell_w=480)
                sheets.append(sheet.name)
                log(f"station {k} ({s['role']}): drawn mean {s['drawn_mean']:.2f}, behind {s['drawn_behind']:.2f}")
            contact_sheet(allcells, args.views, f"{args.label}: all stations (rows) x yaw (columns)",
                          out / "turnarounds_all.png", cell_w=240)
        finally:
            b.close()
    return {"method": "browser", "page_url": url, "browser": exe, "viewer_state": state,
            "config": {k: cfg[k] for k in ("up", "rgba8", "kp", "kt", "scale")},
            "rgba8_cap_patched_to": cap, "patched_count": srv.patched,
            "canvas": [args.width, args.height], "pitch_deg": args.pitch_deg,
            "opening_pose": opening, "path_len": len(path), "path_matches_python": same,
            "stations": stations, "sheets": sheets, "path": path,
            "files": ["viewer_opening.png", "turnarounds_all.png"]}


def choose_from_page(path, w, stations, pack):
    """Re-pick stations on the page's own path (only when it differs)."""
    rows = component_table(w)
    cams0 = frame_cameras(w, 0, rows)
    pt = match_path_times(path, cams0)
    times = np.array([t if t is not None else np.nan for _, t in pt], np.float64)
    centres = np.array([e[:3] for e in path], np.float64)
    mpu = pack["scale"].get("metres_per_unit_used")
    radius = (0.6 / mpu) if mpu else 0.08 * float(np.ptp(centres, axis=0).max() + 1e-9)
    st = choose_stations(centres, times, radius, len(stations))
    for s in st:
        s["keyframe_id"], _ = pt[s["index"]]
        s["centre"] = [round(float(x), 4) for x in path[s["index"]][:3]]
    return st


def turnarounds_splat(args, w, out, stations, path, up, log) -> dict:
    V, _F, C, _N, src = surface_level(w, prefer=1)
    if V is None:
        return {"method": "splat", "error": "no surface"}
    step = 360.0 / args.views
    e1, e2 = plane_basis(up)
    u = unit(up)
    allcells = []
    for k, s in enumerate(stations):
        e = path[s["index"]]
        eye, f = np.asarray(e[:3]), np.asarray(e[3:6])
        f = unit(f - u * (f @ u))
        yaw0 = math.atan2(f @ e2, f @ e1)
        cells, views = [], []
        for j in range(args.views):
            yaw = yaw0 + math.radians(j * step)
            d = math.cos(yaw) * e1 + math.sin(yaw) * e2
            img = splat_view(V, C, eye, d, u, args.width, args.height)
            name = f"turnaround_s{k}_yaw{int(round(j * step)):03d}.png"
            from PIL import Image
            Image.fromarray(img).save(out / name)
            filled = float((img.sum(axis=2) != 72).mean())
            views.append({"file": name, "yaw_offset_deg": round(j * step, 1), "filled": round(filled, 3)})
            cells.append((img, f"{yaw_label(j, step)} filled {100 * filled:.0f}%"))
            allcells.append((img, f"s{k} {yaw_label(j, step)}"))
        s["views"] = views
        contact_sheet(cells, 4, f"station {k} ({s['role']}) -- SPLAT fallback of {src} vertex colours", out /
                      f"turnaround_s{k}.png", cell_w=480)
        log(f"station {k} splat done")
    contact_sheet(allcells, args.views, "all stations (splat fallback)", out / "turnarounds_all.png", cell_w=240)
    return {"method": "splat", "mesh": src, "stations": stations}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--world-root", type=artifact_root_arg, required=True,
                    help="the world store root (the directory holding worlds/); read only")
    ap.add_argument("--world", required=True, help="world id or unique prefix")
    ap.add_argument("--session", default=None, help="session id or unique prefix (default: the only one)")
    ap.add_argument("--out-root", type=artifact_root_arg, required=True,
                    help="packs are written to <out-root>/<world>-<session>/")
    ap.add_argument("--stage-root", type=artifact_root_arg, required=True,
                    help="scratch directory for the staged world copy and the browser profile")
    ap.add_argument("--label", default="", help="a name for the titles, e.g. 'walk 9'")
    ap.add_argument("--method", choices=("auto", "browser", "splat"), default="auto")
    ap.add_argument("--browser", default=None, help="path to chrome.exe / msedge.exe")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="test port for the throwaway server")
    ap.add_argument("--stations", type=int, default=4)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--pitch-deg", type=float, default=0.0)
    ap.add_argument("--load-timeout", type=float, default=600.0)
    ap.add_argument("--no-phone-layers", dest="phone_layers", action="store_false",
                    help="keep the desktop RGBA8 layer cap (48) instead of the phone's")
    ap.add_argument("--no-depth-scale", action="store_true", help="skip the live-depth scale estimate")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--env", action="append", metavar="KEY=VALUE",
                    help="set an environment variable before building the page (e.g. TOWER_WORLD_POSE_QUARANTINE=on)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    build_pack(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
