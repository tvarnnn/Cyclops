"""Optional replay receipts and a deliberately narrow FOW data stub.

Coverage names a pinned status geometry revision. A missing pinned receipt
means ``capture-incomplete``; coalesced ordinary revisions are reported
separately and do not make the capture incomplete.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import time
from pathlib import Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def revision_dir(revision: str) -> str:
    """Keep ordinary opaque revisions readable without trusting them as paths."""
    if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", revision):
        return revision
    return _sha(revision.encode("utf-8"))


class ReplayCapture:
    def __init__(self, out: Path):
        self.out = Path(out)
        self.started_monotonic = time.perf_counter()
        self.started_at = time.time()
        self.status_count = 0
        self.geometry_rows = []
        self._bodies = {}  # (world, session, content hash, placement hash) -> saved file
        self._revisions = {}
        self._current_revision = None
        self._finished_at = None

    def status(self, raw: str | bytes, envelope: dict) -> None:
        data = raw.encode("utf-8") if isinstance(raw, str) else raw
        row = {"n": self.status_count, "received_at": time.time(),
               "received_monotonic": time.perf_counter(),
               "clock_basis": "client wall clock on test Tower host",
               "envelope_sha256": _sha(data), "envelope_b64": base64.b64encode(data).decode("ascii"),
               "envelope": envelope}
        with (self.out / "status-pushes.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        self.status_count += 1
        payload = envelope.get("payload") or {}
        if not isinstance(payload, dict):
            return
        snapshot = payload.get("world_snapshot") or {}
        session = payload.get("session") or {}
        geometry = payload.get("geometry") or {}
        if not all(isinstance(block, dict) for block in (snapshot, session, geometry)):
            return
        target = (snapshot.get("world_id"), session.get("session_id"), geometry.get("revision"))
        if not all(isinstance(value, str) and value for value in target):
            return
        if self._current_revision is not None and self._current_revision != target:
            previous = self._revisions[self._current_revision]
            previous["superseded_at"] = row["received_at"]
            previous["superseded_monotonic"] = row["received_monotonic"]
        self._current_revision = target
        tracked = self._revisions.setdefault(target, {
            "world_id": target[0], "session_id": target[1], "revision": target[2],
            "first_received_at": row["received_at"],
            "first_received_monotonic": row["received_monotonic"],
            "pinned_at": None, "pinned_monotonic": None,
            "superseded_at": None, "superseded_monotonic": None,
        })
        coverage = ((payload.get("guidance") or {}).get("coverage") or {})
        if (isinstance(coverage, dict) and coverage.get("geometry_revision") == target[2]
                and tracked["pinned_at"] is None):
            tracked["pinned_at"] = row["received_at"]
            tracked["pinned_monotonic"] = row["received_monotonic"]

    def geometry(self, target: tuple[str, str, str], manifest: bytes,
                 fetched: list[tuple[dict, bytes]]) -> bool:
        world, session, revision = target
        doc = json.loads(manifest)
        # Status geometry.revision names the captured landing. The HTTP
        # manifest's geometry_revision is a separate segment-hash rollup.
        if (doc.get("world_id", world) != world or doc.get("session_id", session) != session):
            raise ValueError(f"geometry manifest changed before revision {revision} was fetched")
        by_index = {item["segment_index"]: item for item in doc.get("segments", [])}
        for item, body in fetched:
            index = item["segment_index"]
            chunk = json.loads(body)
            if (chunk.get("segment_index", index) != index or
                    chunk.get("content_hash", item.get("content_hash")) != item.get("content_hash") or
                    chunk.get("placement_hash", item.get("placement_hash")) != item.get("placement_hash")):
                raise ValueError(f"segment {index} changed before revision {revision} was fetched")
        fetched_indices = {item["segment_index"] for item, _ in fetched}
        if any(index not in fetched_indices and
               (world, session, item.get("content_hash"), item.get("placement_hash"))
               not in self._bodies for index, item in by_index.items()):
            return False  # a retry may see a newer manifest; publish no partial receipt
        directory = self.out / "geometry-snapshots" / revision_dir(revision)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists() and manifest_path.read_bytes() != manifest:
            raise ValueError(f"geometry revision {revision} changed manifest bytes")
        manifest_path.write_bytes(manifest)
        for item, body in fetched:
            index = item["segment_index"]
            key = (world, session, item.get("content_hash"), item.get("placement_hash"))
            path = directory / f"segment-{index}.json"
            path.write_bytes(body)
            self._bodies[key] = path
        segments = []
        for index, item in by_index.items():
            key = (world, session, item.get("content_hash"), item.get("placement_hash"))
            path = directory / f"segment-{index}.json"
            if not path.exists() and key in self._bodies:
                path.write_bytes(self._bodies[key].read_bytes())
            segments.append({"segment_index": index, "content_hash": item.get("content_hash"),
                             "placement_hash": item.get("placement_hash"),
                             "file": path.name if path.exists() else None,
                             "sha256": _sha(path.read_bytes()) if path.exists() else None})
        row = {"world_id": world, "session_id": session, "revision": revision,
               "directory": directory.relative_to(self.out).as_posix(),
               "received_at": time.time(), "received_monotonic": time.perf_counter(),
               "manifest_sha256": _sha(manifest), "segments": segments,
               "complete": all(s["file"] is not None for s in segments)}
        self.geometry_rows.append(row)
        with (self.out / "geometry-snapshots.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        return row["complete"]

    def finish(self, walk_t0: float | None, solutions: list[dict]) -> None:
        self._finished_at = (time.time(), time.perf_counter())
        index = {"clock_basis": "client wall clock on test Tower host",
                 "recording_started_at": self.started_at,
                 "recording_started_monotonic": self.started_monotonic,
                 "walk_t0": walk_t0, "status_pushes": self.status_count,
                 "geometry": self.geometry_rows, "solutions": solutions,
                 "capture_summary": self.capture_summary()}
        (self.out / "walk-clock-index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    def capture_summary(self) -> dict:
        """Classify distinct status revisions by whether a complete receipt exists."""
        complete = {(row["world_id"], row["session_id"], row["revision"])
                    for row in self.geometry_rows if row["complete"]}
        ended_at, ended_monotonic = self._finished_at or (time.time(), time.perf_counter())
        pinned_misses = []
        unpinned_misses = []
        for target, tracked in self._revisions.items():
            if target in complete:
                continue
            superseded = tracked["superseded_at"] is not None
            missed_at = tracked["superseded_at"] if superseded else ended_at
            missed_monotonic = (tracked["superseded_monotonic"] if superseded
                                else ended_monotonic)
            miss = {**tracked, "missed_at": missed_at,
                    "missed_monotonic": missed_monotonic,
                    "elapsed_ms": round((missed_monotonic - tracked["first_received_monotonic"]) * 1000, 3),
                    "reason": ("revision superseded before complete receipt" if superseded
                               else "recording ended without complete receipt")}
            (pinned_misses if tracked["pinned_at"] is not None else unpinned_misses).append(miss)
        return {"pinned_missed": len(pinned_misses), "pinned_misses": pinned_misses,
                "unpinned_missed": len(unpinned_misses), "unpinned_misses": unpinned_misses}

    @property
    def incomplete(self) -> bool:
        return self.capture_summary()["pinned_missed"] > 0


def supported_by_landing(out: Path) -> list[dict]:
    """Report supported station sectors only with complete source receipts.

    This does not compare labels or claim semantic correctness. Missing joins
    raise ValueError so a partial replay cannot be mistaken for a score.
    """
    out = Path(out)
    statuses = [json.loads(line) for line in (out / "status-pushes.jsonl").read_text(
        encoding="utf-8").splitlines()]
    geometries = {row["revision"]: row for row in (json.loads(line) for line in
                  (out / "geometry-snapshots.jsonl").read_text(encoding="utf-8").splitlines())}
    index = json.loads((out / "walk-clock-index.json").read_text(encoding="utf-8"))
    if index.get("walk_t0") is None:
        raise ValueError("missing walk clock")
    solutions = []
    for item in index.get("solutions", []):
        path = out / "solution-snapshots" / item["file"]
        if not path.is_file():
            raise ValueError(f"missing solution {path}")
        doc = json.loads(path.read_bytes())
        if not isinstance(doc.get("keyframe_ids"), list):
            raise ValueError("missing accepted-keyframe list")
        solutions.append(doc)
    result = []
    seen = set()
    for row in statuses:
        raw = base64.b64decode(row["envelope_b64"], validate=True)
        if _sha(raw) != row["envelope_sha256"] or json.loads(raw) != row["envelope"]:
            raise ValueError("status envelope bytes do not match receipt")
        if not isinstance(row.get("received_at"), (int, float)):
            raise ValueError("missing status time join")
        coverage = (((row["envelope"].get("payload") or {}).get("guidance") or {})
                    .get("coverage"))
        if not coverage:
            continue
        if coverage.get("source") != "landed_global_solve":
            raise ValueError("coverage is not a landed solve")
        if row["received_at"] < coverage["solved_at"]:
            raise ValueError("status predates named landing")
        revision = coverage["geometry_revision"]
        geometry = geometries.get(revision)
        if geometry is None or not geometry["complete"]:
            raise ValueError(f"missing complete geometry revision {revision}")
        directory = out / geometry["directory"]
        if not (directory / "manifest.json").is_file():
            raise ValueError(f"missing manifest for {revision}")
        if _sha((directory / "manifest.json").read_bytes()) != geometry["manifest_sha256"]:
            raise ValueError(f"manifest digest mismatch for {revision}")
        for segment in geometry["segments"]:
            path = directory / segment["file"]
            if not path.is_file() or _sha(path.read_bytes()) != segment["sha256"]:
                raise ValueError(f"missing or changed segment body for {revision}")
        horizon = coverage["horizon_keyframes"]
        if not any(solution.get("solved_at") == coverage["solved_at"] and
                   len(solution["keyframe_ids"]) == horizon for solution in solutions):
            raise ValueError("missing solution / accepted-keyframe horizon join")
        identity = (coverage["solved_at"], revision, horizon)
        if identity in seen:
            continue
        seen.add(identity)
        stations = [{"reference_segment": component["reference_segment"],
                     "x": station["x"], "y": station["y"],
                     "supported_sectors": [sector for sector in range(12)
                                           if station["supported_mask"] & (1 << sector)]}
                    for component in coverage["components"] for station in component["stations"]]
        result.append({"solved_at": coverage["solved_at"], "geometry_revision": revision,
                       "horizon_keyframes": horizon, "first_received_at": row["received_at"],
                       "walk_seconds": row["received_at"] - index["walk_t0"],
                       "stations": stations})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List supported sectors per recorded landed solve")
    parser.add_argument("out", type=Path, help="the finished replay output directory")
    args = parser.parse_args()
    print(json.dumps(supported_by_landing(args.out), indent=2))
