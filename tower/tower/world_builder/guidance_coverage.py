"""Dated, solve-placed station coverage for the optional status guidance block.

All disk reads and geometry work belong to the guidance worker.  In particular,
neither a status poll nor a builder call parses the pose or keyframe journals.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)
_EPS = 1e-6
_MAX_BYTES = 4096
_DISCOVERY_INTERVAL_S = 0.5


class _FrozenDict(dict):
    def _immutable(self, *args, **kwargs):
        raise TypeError("published coverage is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable

    def __deepcopy__(self, memo):
        return self


def _freeze(value):
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def _unit(a):
    if not isinstance(a, (list, tuple)) or len(a) != 3:
        raise ValueError("invalid vector")
    a = [float(v) for v in a]
    if not all(math.isfinite(v) for v in a):
        raise ValueError("nonfinite vector")
    n = math.sqrt(_dot(a, a))
    if n < _EPS:
        raise ValueError("degenerate vector")
    return [v/n for v in a]


def _rotate(q, v):
    if not isinstance(q, (list, tuple)) or len(q) != 4:
        raise ValueError("invalid quaternion")
    q = [float(x) for x in q]
    if not all(math.isfinite(x) for x in q) or abs(sum(x*x for x in q)-1) > 0.01:
        raise ValueError("nonunit quaternion")
    w, x, y, z = q
    u = [x, y, z]
    t = [2*a for a in _cross(u, v)]
    return [v[i] + w*t[i] + _cross(u, t)[i] for i in range(3)]


def _placed(row, placement):
    centre = _rotate(placement["rotation_wxyz"], _unit_or_zero(row["translation"]))
    scale = float(placement["scale"])
    offset = _unit_or_zero(placement["translation"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid placement scale")
    position = [scale*centre[i]+offset[i] for i in range(3)]
    # T_world_camera in the local segment, followed by the registered Sim3.
    def axis(local):
        return _rotate(placement["rotation_wxyz"], _rotate(row["rotation"], local))
    return position, axis([0.0, -1.0, 0.0]), axis([0.0, 0.0, 1.0])


def _unit_or_zero(a):
    if not isinstance(a, (list, tuple)) or len(a) != 3:
        raise ValueError("invalid translation")
    values = [float(v) for v in a]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite translation")
    return values


def _source_pose(entry):
    """Solution's R_cw/t_cw, expressed as camera centre/up/forward in its component."""
    matrix = entry["rotation"]
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 9:
        raise ValueError("missing solution rotation")
    matrix = [float(v) for v in matrix]
    translation = _unit_or_zero(entry["translation"])
    if not all(math.isfinite(v) for v in matrix):
        raise ValueError("nonfinite solution rotation")
    rows = [matrix[i*3:i*3+3] for i in range(3)]
    if any(not math.isclose(_dot(rows[i], rows[j]), float(i == j), abs_tol=1e-3)
           for i in range(3) for j in range(3)):
        raise ValueError("nonorthogonal solution rotation")
    def camera_to_world(v):
        return [sum(matrix[i*3+j]*v[i] for i in range(3)) for j in range(3)]
    centre = camera_to_world([-v for v in translation])
    return matrix, centre, camera_to_world([0.0, -1.0, 0.0]), camera_to_world([0.0, 0.0, 1.0])


def _solution_relative(entry, anchor):
    _, centre, up, forward = _source_pose(entry)
    (a_matrix, a_centre, _, _), anchor_row, anchor_placement = anchor
    def world_to_output(v):
        camera = [sum(a_matrix[i*3+j]*v[j] for j in range(3)) for i in range(3)]
        return _rotate(anchor_placement["rotation_wxyz"],
                       _rotate(anchor_row["rotation"], camera))
    anchor_position, _, _ = _placed(anchor_row, anchor_placement)
    delta = world_to_output([centre[i]-a_centre[i] for i in range(3)])
    scale = float(anchor_placement["scale"])
    return ([anchor_position[i]+scale*delta[i] for i in range(3)],
            world_to_output(up), world_to_output(forward))


def _agrees(a, b):
    return all(math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-5) for x, y in zip(a, b))


def _integer(value, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError("out-of-bound integer")
    return value


def _component(reference, entries):
    entries.sort(key=lambda item: item[0])
    up = _unit([sum(e[2][i] for e in entries) for i in range(3)])
    forward = None
    for _, _, _, ray in entries:
        projected = [ray[i] - _dot(ray, up)*up[i] for i in range(3)]
        try:
            forward = _unit(projected)
            break
        except ValueError:
            pass
    if forward is None:
        raise ValueError("no horizontal forward")
    right = _unit(_cross(forward, up))
    xy = [(_dot(position, forward), _dot(position, right)) for _, position, _, _ in entries]
    min_x, max_x = min(x for x, _ in xy), max(x for x, _ in xy)
    min_y, max_y = min(y for _, y in xy), max(y for _, y in xy)
    side = max(max_x-min_x, max_y-min_y)
    if side < _EPS:
        raise ValueError("single-point component")
    mid_x, mid_y = (min_x+max_x)/2, (min_y+max_y)/2
    z = sum(_dot(position, up) for _, position, _, _ in entries)/len(entries)
    origin = [mid_x*forward[i]+mid_y*right[i]+z*up[i] for i in range(3)]
    cell_size = side/8
    if not all(math.isfinite(v) for v in (
            min_x, min_y, max_x, max_y, *origin, *up, *forward, cell_size)) or cell_size <= 0:
        raise ValueError("nonfinite component geometry")
    cells = {}
    for ((_, _, _, ray), (x, y)) in zip(entries, xy):
        cx = min(7, max(0, math.floor((x-(mid_x-side/2))/cell_size)))
        cy = min(7, max(0, math.floor((y-(mid_y-side/2))/cell_size)))
        cell = cells.setdefault((cy, cx), [0]*13)
        cell[12] += 1
        projected = [ray[i]-_dot(ray, up)*up[i] for i in range(3)]
        if math.sqrt(_dot(projected, projected)) < _EPS:
            continue
        angle = math.degrees(math.atan2(_dot(projected, right), _dot(projected, forward))) % 360
        # Trig results for an exact boundary can land a few ulps below it.
        # The contract assigns that boundary to the clockwise sector.
        cell[int((angle + 1e-12)//30) % 12] += 1
    stations = []
    for (y, x), counts in cells.items():
        weak = sum(1 << i for i, n in enumerate(counts[:12]) if n == 1)
        supported = sum(1 << i for i, n in enumerate(counts[:12]) if n >= 2)
        stations.append({"x": x, "y": y, "keyframes": _integer(counts[12], 65535),
                         "weak_mask": weak, "supported_mask": supported})
    return ({"reference_segment": _integer(reference, 2**31-1),
             "posed_keyframes": _integer(len(entries), 65535),
             "bounds_xy": [min_x, min_y, max_x, max_y], "origin_xyz": origin,
             "up_xyz": up, "forward_xyz": forward, "cell_size": cell_size,
             "stations": []}, stations)


def _finite_component(component):
    return (all(math.isfinite(v) for name in ("bounds_xy", "origin_xyz", "up_xyz", "forward_xyz")
                for v in component[name]) and
            math.isfinite(component["cell_size"]) and component["cell_size"] > 0)


def compute_coverage(*, solution, manifest, placements, poses, keyframes,
                     geometry_revision, computed_at):
    """Pure computation from one coherent snapshot; invalid global identity refuses all."""
    summary = manifest["global_solve"]
    solved_at = float(summary["solved_at"])
    if (not math.isfinite(solved_at) or solved_at < 0 or
            isinstance(computed_at, bool) or
            not isinstance(computed_at, (int, float)) or
            not math.isfinite(computed_at) or computed_at < solved_at or
            float(solution["solved_at"]) != solved_at):
        raise ValueError("invalid solve clock")
    horizon = _integer(summary["horizon_keyframes"], 65535)
    ids = solution["keyframe_ids"]
    if len(ids) != horizon or len(set(ids)) != horizon:
        raise ValueError("invalid solve horizon")
    if not isinstance(geometry_revision, str) or not 1 <= len(geometry_revision) <= 128 or not geometry_revision.isascii():
        raise ValueError("invalid geometry revision")
    accepted = []
    seen = set()
    for row in keyframes:
        kid = row["keyframe_id"]
        if kid not in seen:
            accepted.append(kid)
            seen.add(kid)
    if accepted[:horizon] != ids or len(accepted) > 65535 or manifest["keyframes"] != len(accepted):
        raise ValueError("accepted journal mismatch")
    digest = hashlib.sha256(b"".join(k.encode("utf-8")+b"\0" for k in accepted)).hexdigest()
    if digest != manifest["input_digest"]:
        raise ValueError("manifest input digest mismatch")
    if solution.get("input_digest") != hashlib.sha256(
            b"".join(k.encode("utf-8")+b"\0" for k in ids)).hexdigest():
        raise ValueError("solution input digest mismatch")
    if solution.get("solver") != summary["solver"] or solution.get("gate") != summary.get("gate"):
        raise ValueError("solution identity mismatch")
    references = {_integer(c["reference_segment"], 2**31-1) for c in summary["components"]}
    if len(references) > 65535:
        raise ValueError("too many components")
    placement_by_segment = {}
    revisions = set()
    for p in placements:
        _integer(p["frame_revision"], 2**31-1)
        if p["state"] != "registered":
            continue
        seg = _integer(p["segment_index"], 2**31-1)
        if (p["input_digest"] != digest or p["reference_segment"] not in references or
                p.get("evidence", {}).get("solver") != summary["solver"] or
                not summary["segments"].get(str(seg), {}).get("replaced")):
            raise ValueError("placement source mismatch")
        revisions.add(_integer(p["frame_revision"], 2**31-1))
        if seg in placement_by_segment:
            raise ValueError("duplicate placement")
        placement_by_segment[seg] = p
    if len(revisions) != 1 or not 1 <= len(placement_by_segment):
        raise ValueError("no consistent registered placements")
    revision = revisions.pop()
    poses_by_id = {r["keyframe_id"]: r for r in poses}
    if len(poses_by_id) != len(poses):
        raise ValueError("duplicate pose")
    eligible = {ref: [] for ref in references}
    source_poses = solution["poses"]
    anchor_candidates = {ref: [] for ref in references}
    for row in poses:
        seg = row["segment_index"]
        if seg in anchor_candidates and row["status"] == "anchor" and row["keyframe_id"] in source_poses:
            anchor_candidates[seg].append(row)
    anchors = {}
    for ref in references:
        if ref not in placement_by_segment:
            continue
        evidence = placement_by_segment[ref].get("evidence", {})
        if (evidence.get("placement_reliable") is False or
                evidence.get("scale_reliable") is False):
            continue
        candidates = anchor_candidates[ref]
        if len(candidates) != 1:
            continue
        try:
            _placed(candidates[0], placement_by_segment[ref])
            anchors[ref] = (_source_pose(source_poses[candidates[0]["keyframe_id"]]),
                            candidates[0], placement_by_segment[ref])
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
    for kid in ids:
        if kid not in seen or kid not in source_poses or kid not in poses_by_id:
            continue
        row = poses_by_id[kid]
        if row["status"] not in ("solved", "anchor"):
            continue
        # Local pose rows do not carry this solve's observation count. Merge
        # writes it only for keyframes actually posed by the named solution.
        observations = source_poses[kid].get("observations")
        if (not isinstance(observations, int) or observations < 30 or
                row.get("observations") != observations):
            continue
        p = placement_by_segment.get(row["segment_index"])
        if p is None or p.get("evidence", {}).get("coverage") not in ("confident", "partial"):
            continue
        if (p.get("evidence", {}).get("placement_reliable") is False or
                p.get("evidence", {}).get("scale_reliable") is False):
            continue
        ref = p["reference_segment"]
        if ref not in anchors:
            continue
        if source_poses[kid]["component"] != p["evidence"]["component"]:
            continue
        try:
            position, up, ray = _placed(row, p)
        except (ValueError, TypeError, KeyError, OverflowError):
            # One malformed pose contributes no evidence, never a painted bit.
            continue
        try:
            expected_position, expected_up, expected_ray = _solution_relative(source_poses[kid], anchors[ref])
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
        if not (_agrees(position, expected_position) and _agrees(up, expected_up) and
                _agrees(ray, expected_ray)):
            raise ValueError("solution pose and merged placement disagree")
        eligible[ref].append((kid, position, up, ray))
    rows, all_stations = {}, []
    for ref in sorted(references):
        if not eligible[ref]:
            continue
        try:
            component, stations = _component(ref, eligible[ref])
        except (ValueError, TypeError, OverflowError):
            continue
        if not _finite_component(component):
            continue
        rows[ref] = component
        all_stations.extend((ref, station) for station in stations)
    all_stations.sort(key=lambda pair: (-pair[1]["keyframes"], pair[0], pair[1]["y"], pair[1]["x"]))
    selected = all_stations[:16]
    for ref, station in selected:
        rows[ref]["stations"].append(station)
    components = []
    for ref in sorted(rows):
        if rows[ref]["stations"]:
            rows[ref]["stations"].sort(key=lambda s: (s["y"], s["x"]))
            components.append(rows[ref])
    block = {"version": 1, "source": "landed_global_solve", "solved_at": solved_at,
             "computed_at": computed_at, "horizon_keyframes": horizon,
             "keyframes_now": len(accepted), "keyframes_pending": max(0, len(accepted)-horizon),
             "geometry_revision": geometry_revision, "frame_revision": revision,
             "station_grid": "component_square_8x8_v1",
             "sector_frame": "first_qualified_forward_cw_from_up_v1",
             "components_total": len(references), "components_omitted": len(references)-len(components),
             "stations_omitted": len(all_stations)-len(selected), "components": components}
    if len(json.dumps(block, separators=(",", ":"), allow_nan=False).encode("utf-8")) > _MAX_BYTES:
        raise ValueError("coverage exceeds 4096 bytes")
    return block


def _signature(paths):
    return tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compute_from_tree(root, world_id, session_id, expected, geometry_revision,
                      clock=time.time, tree_tag=None):
    """Read without a builder lock; reject torn trees and never read image bytes."""
    base = Path(root) / "worlds" / world_id
    session = base / "sessions" / session_id
    derived = base / "derived" / session_id
    paths = (base / "solve" / session_id / "solution.json", derived / "manifest.json",
             derived / "placements.json", derived / "poses.json", session / "keyframes.jsonl")
    before = _signature(paths)
    solution, manifest, placed, posed = (_json(p) for p in paths[:4])
    keyframes = [json.loads(line) for line in paths[4].read_text(encoding="utf-8").splitlines() if line]
    after = _signature(paths)
    if before != after or before[:1] != (expected[2],):
        raise ValueError("derived tree changed during guidance read")
    summary = manifest.get("global_solve") or {}
    if (summary.get("solved_at"), summary.get("horizon_keyframes")) != expected[:2]:
        raise ValueError("build has not merged the named solution")
    if tree_tag is not None and (manifest.get("built_at"), manifest.get("input_digest")) != tree_tag:
        raise ValueError("derived tree changed before guidance read")
    from tower.results.world_builder import geometry_revision_from_manifest
    actual_revision = geometry_revision_from_manifest(manifest)
    if geometry_revision is not None and actual_revision != geometry_revision:
        raise ValueError("geometry revision changed before guidance read")
    return compute_coverage(solution=solution, manifest=manifest,
                            placements=placed["placements"], poses=posed["poses"],
                            keyframes=keyframes, geometry_revision=actual_revision,
                            computed_at=clock())


class CoverageWorker:
    """One startup daemon discovers landings and publishes immutable receipts."""

    def __init__(self, root, clock=time.time):
        self.root, self.clock = root, clock
        self._condition = threading.Condition()
        self._pending = None
        self._candidate = None
        self._completed = None
        self._successful_candidate = None
        self._block = None
        self._block_session = None
        self._attempt_tag = None
        # A few pinned subscriptions may inspect different saved sessions.
        # Keep this bounded so they cannot resubmit the same solve every poll.
        self._receipts = OrderedDict()
        self._attempts = OrderedDict()
        self._published = {}
        self._thread = threading.Thread(target=self._run, name="world-guidance", daemon=True)
        self._thread.start()

    def latest(self, world_id, session_id):
        """A constant-time, lock-free read; the worker replaces the mapping atomically."""
        return self._published.get((world_id, session_id))

    def _discover(self):
        """All file discovery stays in the worker, independently of status polls."""
        candidates = []
        for path in (Path(self.root) / "worlds").glob("*/solve/*/solution.json"):
            world_id, session_id = path.parents[2].name, path.parent.name
            try:
                stat = path.stat()
                manifest = _json(Path(self.root) / "worlds" / world_id / "derived" /
                                 session_id / "manifest.json")
                summary = manifest["global_solve"]
                from tower.results.world_builder import geometry_revision_from_manifest
                revision = geometry_revision_from_manifest(manifest)
                candidates.append((summary["solved_at"], world_id, session_id,
                                   summary["horizon_keyframes"],
                                   (stat.st_size, stat.st_mtime_ns), revision,
                                   (manifest.get("built_at"), manifest.get("input_digest"))))
            except (OSError, KeyError, TypeError, ValueError, OverflowError):
                continue
        for solved_at, world, session, horizon, stat, revision, tag in sorted(candidates):
            self.offer(world, session, solved_at, horizon, stat, revision, tag)

    def offer(self, world_id, session_id, solved_at, horizon, solution_stat,
              geometry_revision, tree_tag):
        candidate = (world_id, session_id, solved_at, horizon, solution_stat)
        with self._condition:
            if candidate in self._receipts:
                self._receipts.move_to_end(candidate)
                return self._receipts[candidate]
            previous = next((block for key, block in reversed(self._receipts.items())
                             if key[:2] == (world_id, session_id)), None)
            if self._attempts.get(candidate) != tree_tag:
                if self._pending is not None:
                    self._attempts.pop(self._pending[0], None)
                self._candidate = candidate
                self._attempt_tag = tree_tag
                self._attempts[candidate] = tree_tag
                if len(self._attempts) > 8:
                    self._attempts.popitem(last=False)
                self._pending = (candidate, geometry_revision, tree_tag)
                self._condition.notify()
            return previous

    def _run(self):
        if os.name == "nt":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetThreadPriority(kernel32.GetCurrentThread(), -1)
            except (AttributeError, OSError):
                logger.warning("world builder guidance: could not lower worker priority")
        while True:
            try:
                self._discover()
            except Exception:
                logger.exception("world builder guidance: discovery failed")
            with self._condition:
                if self._pending is None:
                    self._condition.wait(timeout=_DISCOVERY_INTERVAL_S)
                    if self._pending is None:
                        continue
                candidate, revision, tree_tag = self._pending
                self._pending = None
            start = time.monotonic()
            try:
                world, session, solved_at, horizon, stat = candidate
                block = compute_from_tree(self.root, world, session,
                                          (solved_at, horizon, stat), revision,
                                          self.clock, tree_tag)
                if time.monotonic()-start > 0.250:
                    raise ValueError("coverage worker exceeded 250 ms")
            except Exception as exc:
                logger.warning("world builder guidance: coverage unavailable: %s", exc)
                block = None
            with self._condition:
                if candidate == self._candidate:
                    self._completed = candidate
                    if block is not None:
                        self._block = _freeze(block)
                        self._block_session = (world, session)
                        self._successful_candidate = candidate
                        self._receipts[candidate] = block
                        self._published = {**self._published,
                                           (world, session): self._block}
                        if len(self._receipts) > 8:
                            self._receipts.popitem(last=False)
                else:
                    self._attempts.pop(candidate, None)
