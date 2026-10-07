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
_MEMO_LIMIT = 256
_LANDING_BUDGET_S = 0.250
_RETRY_DELAYS_S = (5.0, 20.0)
_READ_CHUNK_BYTES = 64 * 1024
# Per-file read caps, about 5-10x the walk-6 files (solution 438 KB,
# manifest 25 KB, placements 68 KB, poses 332 KB, keyframes 778 KB for
# 1,090 keyframes). The cap is also the bound on the one step the budget
# cannot interrupt: a single json.loads of a capped 4 MiB file measured
# 25-29 ms on the Tower host, holding the GIL for that long.
_FILE_LIMITS = {"solution.json": 4 * 1024 * 1024,
                "manifest.json": 1024 * 1024,
                "placements.json": 1024 * 1024,
                "poses.json": 4 * 1024 * 1024,
                "keyframes.jsonl": 4 * 1024 * 1024}


class _TransientCoverageError(ValueError):
    """A coherent snapshot may become available on a later attempt."""


class _Budget:
    """The hard per-landing budget: wall time since the job began, plus cancellation.

    `check()` raises once either is exceeded. It is called before and after
    every bounded chunk read, every parse, and every loop iteration of the
    computation, so the job stops at the next check, publishes nothing, and
    the worker keeps the last published block (and logs a warning).
    """

    def __init__(self, cancelled=None):
        self.deadline = time.monotonic() + _LANDING_BUDGET_S
        self.cancelled = cancelled

    def check(self):
        if self.cancelled is not None and self.cancelled.is_set():
            raise ValueError("coverage worker cancelled")
        if time.monotonic() >= self.deadline:
            raise _TransientCoverageError("coverage worker exceeded 250 ms")


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


def _component(reference, entries, check=lambda: None):
    entries.sort(key=lambda item: item[0])
    check()
    up = _unit([sum(e[2][i] for e in entries) for i in range(3)])
    forward = None
    for _, _, _, ray in entries:
        check()
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
        check()
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
                     geometry_revision, computed_at, check=lambda: None):
    """Pure computation from one coherent snapshot; invalid global identity refuses all."""
    summary = manifest["global_solve"]
    solved_at = float(summary["solved_at"])
    if (not math.isfinite(solved_at) or solved_at < 0 or
            isinstance(computed_at, bool) or
            not isinstance(computed_at, (int, float)) or
            not math.isfinite(computed_at) or
            float(solution["solved_at"]) != solved_at):
        raise ValueError("invalid solve clock")
    if computed_at < solved_at:
        raise _TransientCoverageError("solve clock not yet settled")
    horizon = _integer(summary["horizon_keyframes"], 65535)
    ids = solution["keyframe_ids"]
    if len(ids) != horizon or len(set(ids)) != horizon:
        raise ValueError("invalid solve horizon")
    if not isinstance(geometry_revision, str) or not 1 <= len(geometry_revision) <= 128 or not geometry_revision.isascii():
        raise ValueError("invalid geometry revision")
    accepted = []
    seen = set()
    for row in keyframes:
        check()
        kid = row["keyframe_id"]
        if kid not in seen:
            accepted.append(kid)
            seen.add(kid)
    built_count = _integer(manifest["keyframes"], 65535)
    common = min(horizon, len(accepted))
    if (accepted[:common] != ids[:common] or len(accepted) > 65535 or
            built_count < horizon):
        raise ValueError("accepted journal mismatch")
    if len(accepted) < built_count:
        raise _TransientCoverageError("accepted journal mismatch: build ahead of journal")
    # The derived build consumed a frozen prefix; the append-only journal may
    # already have more accepted keyframes by the time guidance reads it.
    digest = hashlib.sha256(b"".join(k.encode("utf-8")+b"\0"
                                     for k in accepted[:built_count])).hexdigest()
    if digest != manifest["input_digest"]:
        raise ValueError("manifest input digest mismatch")
    if solution.get("input_digest") != hashlib.sha256(
            b"".join(k.encode("utf-8")+b"\0" for k in ids)).hexdigest():
        raise ValueError("solution input digest mismatch")
    if (solution.get("solver") != summary["solver"] or
            solution.get("gate") != summary.get("gate") or
            solution.get("solve") != summary.get("solve")):
        raise ValueError("solution identity mismatch")
    references = {_integer(c["reference_segment"], 2**31-1) for c in summary["components"]}
    if len(references) > 65535:
        raise ValueError("too many components")
    placement_by_segment = {}
    revisions = set()
    for p in placements:
        check()
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
        check()
        seg = row["segment_index"]
        if seg in anchor_candidates and row["status"] == "anchor" and row["keyframe_id"] in source_poses:
            anchor_candidates[seg].append(row)
    anchors = {}
    for ref in references:
        check()
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
        check()
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
        check()
        if not eligible[ref]:
            continue
        try:
            component, stations = _component(ref, eligible[ref], check)
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
    check()
    return block


def _signature(paths):
    return tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


def _read_bounded(path, budget):
    limit = _FILE_LIMITS[path.name]
    budget.check()
    if path.stat().st_size > limit:
        raise ValueError(f"guidance file exceeds size limit: {path.name}")
    chunks, total = [], 0
    with path.open("rb") as stream:
        while True:
            budget.check()
            chunk = stream.read(min(_READ_CHUNK_BYTES, limit + 1 - total))
            budget.check()
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ValueError(f"guidance file exceeds size limit: {path.name}")
            chunks.append(chunk)
    budget.check()
    return b"".join(chunks).decode("utf-8")


def _json(path, budget=None):
    budget = budget or _Budget()
    value = json.loads(_read_bounded(path, budget))
    budget.check()
    return value


def compute_from_tree(root, world_id, session_id, expected, geometry_revision,
                      clock=time.time, tree_tag=None, budget=None):
    """Read without a builder lock; reject torn trees and never read image bytes."""
    base = Path(root) / "worlds" / world_id
    session = base / "sessions" / session_id
    derived = base / "derived" / session_id
    paths = (base / "solve" / session_id / "solution.json", derived / "manifest.json",
             derived / "placements.json", derived / "poses.json", session / "keyframes.jsonl")
    budget = budget or _Budget()
    budget.check()
    before = _signature(paths)
    if any(size > _FILE_LIMITS[path.name] for path, (size, _) in zip(paths, before)):
        raise ValueError("guidance file exceeds size limit")
    solution, manifest, placed, posed = (_json(p, budget) for p in paths[:4])
    keyframes = []
    for line in _read_bounded(paths[4], budget).splitlines():
        budget.check()
        if line:
            keyframes.append(json.loads(line))
    budget.check()
    after = _signature(paths)
    if before != after or before[:1] != (expected[2],):
        raise _TransientCoverageError("derived tree changed during guidance read")
    summary = manifest.get("global_solve") or {}
    if (manifest.get("session_id") != session_id or
            any(row.get("session_id") != session_id for row in keyframes)):
        raise ValueError("accepted journal session mismatch")
    if (summary.get("solved_at"), summary.get("horizon_keyframes")) != expected[:2]:
        raise _TransientCoverageError("build has not merged the named solution")
    if tree_tag is not None and (manifest.get("built_at"), manifest.get("input_digest")) != tree_tag:
        raise _TransientCoverageError("derived tree changed before guidance read")
    from tower.results.world_builder import geometry_revision_from_manifest
    actual_revision = geometry_revision_from_manifest(manifest)
    budget.check()
    if geometry_revision is not None and actual_revision != geometry_revision:
        raise _TransientCoverageError("geometry revision changed before guidance read")
    return compute_coverage(solution=solution, manifest=manifest,
                            placements=placed["placements"], poses=posed["poses"],
                            keyframes=keyframes, geometry_revision=actual_revision,
                            computed_at=clock(), check=budget.check)


class CoverageWorker:
    """One daemon discovers requested landings and publishes immutable receipts.

    A Python thread cannot be killed, so this is exactly what is guaranteed:

    - A status poll never waits on this thread. `latest()` is one dict
      lookup on a mapping the worker replaces atomically; the poll does no
      I/O and takes no lock this thread holds during a job.
    - No builder lock is taken. Stop and the final solve run in the capture
      worker process and never join, wait on, or signal this thread.
    - A job stops at its next budget check once 250 ms have elapsed or
      `shutdown()` was called. Checks sit between bounded 64 KiB reads,
      after each parse, and inside every computation loop. Each derived file
      is closed when its read is abandoned, so the builder's
      `replace_with_retry` (2 s budget) is not refused for longer than the
      budget plus one chunk read.
    - A failed job publishes nothing; the last published block stays.
      A transient failure (deadline miss, tree changed mid-read, solve clock
      not yet settled) retries the same candidate and tree in this thread
      after 5 s, then 20 s: at most two timed retries per candidate. Distinct
      requested worlds wait in a queue; a newer offer supersedes a cooling-down
      retry, and `shutdown()` cancels queued work. After a failure a rebuilt
      tree is re-attempted at once, without resetting the retry count.
      Validation failures are not retried. Every attempt logs candidate,
      elapsed, reason, disposition.
    - A solve candidate that published is never recomputed: a later tree
      tag of the same solve returns that receipt (WORLDS v8: a local rebuild
      advances geometry.revision without re-dating coverage).
    - `shutdown()` returns without waiting. After it, nothing is published
      and no new job starts.

    Not guaranteed: an OS read that itself never returns (a stalled disk)
    holds that one handle and this daemon thread until it returns. One
    `json.loads` of a capped file (<= 4 MiB, about 25-29 ms measured) cannot
    be interrupted and holds the GIL for that long. Low thread priority
    does not remove disk or CPU contention with a final solve; the budget
    only bounds how long a job contends.
    """

    def __init__(self, root, clock=time.time, retry_clock=time.monotonic):
        self.root, self.clock = root, clock
        self._retry_clock = retry_clock
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._pending = None
        self._queued = OrderedDict()
        self._candidate = None
        self._completed = None
        self._successful_candidate = None
        self._block = None
        self._block_session = None
        self._attempt_tag = None
        self._wanted = {}
        self._targets = OrderedDict()
        # A world list can contain more than eight historical sessions, so
        # both memos are large and bounded. A published receipt is keyed by
        # the solve candidate alone: a live rebuild (new tree tag) of the same
        # solve neither recomputes nor re-dates it (WORLDS v8). An attempt is
        # keyed by (candidate, tree tag), so a failure is retried on a new tree.
        self._receipts = OrderedDict()
        self._attempts = OrderedDict()
        # Timed retries used per attempted candidate (evicted with it).
        self._retries = {}
        self._published = {}
        self._thread = threading.Thread(target=self._run, name="world-guidance", daemon=True)
        self._thread.start()

    def latest(self, world_id, session_id):
        """A constant-time, lock-free read; the worker replaces the mapping atomically."""
        return self._published.get((world_id, session_id))

    def request(self, world_id, session_id):
        """Schedule discovery for one status target; never inspect other worlds."""
        with self._condition:
            if not self._stop.is_set():
                self._targets[(world_id, session_id)] = None
                self._condition.notify()
        return self.latest(world_id, session_id)

    def shutdown(self):
        """Cancel future work and publication without waiting on an OS read."""
        self._stop.set()
        with self._condition:
            self._pending = None
            self._queued.clear()
            self._condition.notify_all()

    def _discover(self):
        """Inspect only status targets, off the status polling thread."""
        with self._condition:
            targets = tuple(self._targets)
            self._targets.clear()
        candidates = []
        for world_id, session_id in targets:
            if self._stop.is_set():
                return
            path = Path(self.root) / "worlds" / world_id / "solve" / session_id / "solution.json"
            try:
                stat = path.stat()
                manifest = _json(Path(self.root) / "worlds" / world_id / "derived" /
                                 session_id / "manifest.json", _Budget(self._stop))
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
            if self._stop.is_set():
                return
            self.offer(world, session, solved_at, horizon, stat, revision, tag)

    def offer(self, world_id, session_id, solved_at, horizon, solution_stat,
              geometry_revision, tree_tag):
        candidate = (world_id, session_id, solved_at, horizon, solution_stat)
        identity = (candidate, tree_tag)
        with self._condition:
            if self._stop.is_set():
                return self._published.get((world_id, session_id))
            pair = (world_id, session_id)
            self._wanted[pair] = identity
            if candidate in self._receipts:
                # This solve already published: any tree tag reuses it as is.
                self._receipts.move_to_end(candidate)
                block, frozen = self._receipts[candidate]
                if self._published.get(pair) is not frozen:
                    self._published = {**self._published, pair: frozen}
                return block
            previous = next((block for key, (block, _) in reversed(self._receipts.items())
                             if key[:2] == pair), None)
            if identity not in self._attempts:
                # A newer tree of the same session makes queued older work
                # obsolete; a different world keeps its place in the queue.
                for queued_identity in tuple(self._queued):
                    if queued_identity[0][:2] == pair:
                        self._queued.pop(queued_identity)
                        self._attempts.pop(queued_identity, None)
                if self._pending is not None:
                    dropped, retry = self._pending[0], self._pending[3]
                    if retry:
                        # A cooling-down retry has already run; at most one
                        # job is pending, so the newer offer supersedes it.
                        logger.warning("world builder guidance: candidate=%r retry %d "
                                       "cancelled: superseded", dropped, retry)
                        self._pending = None
                    else:
                        queued = self._pending
                        queued_identity = (queued[0], queued[2])
                        if dropped[:2] != pair:
                            # Different requested worlds still need work.
                            self._queued[queued_identity] = queued
                        else:
                            self._attempts.pop(queued_identity, None)
                        self._pending = None
                self._attempts[identity] = None
                if len(self._attempts) > _MEMO_LIMIT:
                    evicted, _ = self._attempts.popitem(last=False)
                    self._retries.pop(evicted[0], None)
                self._pending = (candidate, geometry_revision, tree_tag, 0, None)
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
        while not self._stop.is_set():
            try:
                self._discover()
            except Exception:
                logger.exception("world builder guidance: discovery failed")
            with self._condition:
                if self._stop.is_set():
                    break
                if self._pending is None and self._queued:
                    _, self._pending = self._queued.popitem(last=True)
                if self._pending is None:
                    self._condition.wait(timeout=_DISCOVERY_INTERVAL_S)
                    if self._stop.is_set():
                        break
                    if self._pending is None:
                        continue
                candidate, revision, tree_tag, retry, due = self._pending
                if due is not None:
                    remaining = due - self._retry_clock()
                    if remaining > 0:
                        # Cooling down: keep discovering, never sleep past shutdown.
                        self._condition.wait(timeout=min(_DISCOVERY_INTERVAL_S, remaining))
                        continue
                self._pending = None
                if candidate in self._receipts:
                    # Published while this job waited; never recompute a solve.
                    continue
                self._candidate = candidate
                self._attempt_tag = tree_tag
            budget = _Budget(self._stop)
            started = time.monotonic()
            reason = "ok"
            failure = None
            try:
                world, session, solved_at, horizon, stat = candidate
                block = compute_from_tree(self.root, world, session,
                                          (solved_at, horizon, stat), revision,
                                          self.clock, tree_tag, budget)
                budget.check()
            except Exception as exc:
                failure = exc
                reason = str(exc) or type(exc).__name__
                block = None
            elapsed_ms = (time.monotonic() - started) * 1000
            with self._condition:
                if self._stop.is_set():
                    logger.warning("world builder guidance: candidate=%r elapsed_ms=%.1f "
                                   "reason=%s disposition=gave up (shutdown)",
                                   candidate, elapsed_ms, reason)
                    break
                if self._wanted.get((world, session)) == (candidate, tree_tag):
                    self._completed = candidate
                    if block is not None:
                        self._block = _freeze(block)
                        self._block_session = (world, session)
                        self._successful_candidate = candidate
                        self._receipts[candidate] = (block, self._block)
                        self._published = {**self._published,
                                           (world, session): self._block}
                        if len(self._receipts) > _MEMO_LIMIT:
                            self._receipts.popitem(last=False)
                        self._retries.pop(candidate, None)
                        disposition = "ok"
                    elif (isinstance(failure, _TransientCoverageError) and
                          self._pending is None and
                          self._retries.get(candidate, 0) < len(_RETRY_DELAYS_S)):
                        # Same candidate and tree, after 5 s then 20 s; the
                        # previous dated block stays published meanwhile.
                        used = self._retries.get(candidate, 0)
                        self._retries[candidate] = used + 1
                        self._pending = (candidate, revision, tree_tag, used + 1,
                                         self._retry_clock() + _RETRY_DELAYS_S[used])
                        self._condition.notify()
                        disposition = f"retry {used + 1}"
                    else:
                        disposition = "gave up"
                else:
                    # Superseded: the requested identity changed in flight.
                    # Only a published receipt is memoized, so this one is not.
                    disposition = "gave up (superseded)"
            log = logger.info if block is not None else logger.warning
            log("world builder guidance: candidate=%r elapsed_ms=%.1f reason=%s disposition=%s",
                candidate, elapsed_ms, reason, disposition)
