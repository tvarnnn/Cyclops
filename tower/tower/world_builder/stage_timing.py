"""Bounded, best-effort timing for World Builder finish stages."""

from __future__ import annotations

import contextvars
import functools
import itertools
import json
import logging
import os
import time

from tower.config import world_stage_timing_setting
from tower.storage import write_json_atomic

logger = logging.getLogger(__name__)
FILENAME = "stage_timing.json"
SCHEMA = 2
FINISH_LIMIT = 3
LIMITS = {"solve": 1, "solve_draw": 7, "gate": 16, "p4": 2,
          "regate": 1, "surface": 1, "appearance": 1, "areas": 1}
CACHES = ("mask", "depth", "pair")
_active = contextvars.ContextVar("world_stage_timing_active", default=None)
_run_serial = itertools.count(1)
# A run-less stage joins only a finish begun by this process or a solve child it launched.
_PROCESS_STARTED = time.time()


def _new_run(context):
    return {"finish_id": f"{os.getpid()}-{time.time_ns()}-{next(_run_serial)}",
            "started_at": time.time(), "pid": os.getpid(), "context": context}


class _Sample:
    def __init__(self, stage: str, identity, parent=None, *, seed=None):
        self.stage = stage
        self.identity = identity
        self.started = time.perf_counter()
        self.caches = {name: {"hits": 0, "misses": 0} for name in CACHES}
        self.pending = []
        self.parent = parent
        self.run = parent.run if parent is not None else None
        self.seed = seed
        self.concurrent = None
        if self.run is None and stage in ("solve", "regate", "areas"):
            self.run = _new_run(stage)


def cache(name: str, hit: bool) -> None:
    """Count only an explicitly marked cache lookup in the active stage."""
    try:
        sample = _active.get()
        if sample is not None and name in sample.caches:
            sample.caches[name]["hits" if hit else "misses"] += 1
    except Exception:  # noqa: BLE001 -- diagnostics cannot change a lookup
        pass


def concurrent_draw_event(event: str) -> None:
    """I0-only reason for taking the serial path; never enters a world record."""
    sample = _active.get()
    while sample is not None and sample.stage != "solve":
        sample = sample.parent
    if sample is not None:
        sample.concurrent = event


def child_draw_timing(seed: int, wall_ms: float, status: str = "ok") -> None:
    """Join a child's map time into this solve, in the parent's seed order."""
    parent = _active.get()
    while parent is not None and parent.stage != "solve":
        parent = parent.parent
    if parent is None:
        return
    sample = _Sample("solve_draw", parent.identity, parent, seed=seed)
    parent.pending.append((sample, {
        "wall_ms": max(0.0, round(float(wall_ms), 3)), "gpu_wait_ms": 0.0,
        "gpu_wait_source": "no_blocking_gpu_lock", "cache": sample.caches,
        "status": status,
    }))


def _write(store, world_id: str, session_id: str, sample: _Sample, record: dict) -> None:
    try:
        path = store.session_dir(world_id, session_id) / FILENAME
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            doc = {}
        if doc.get("schema") != SCHEMA or not isinstance(doc.get("finishes"), list):
            doc = {"schema": SCHEMA, "finishes": []}
        finishes = doc["finishes"]
        run = sample.run
        if run is None:
            # Later room stages can run in the finisher parent after the solve child exits,
            # but never join a finish begun before this process started (another run's).
            last = finishes[-1] if finishes else None
            started = last.get("started_at") if isinstance(last, dict) else None
            ours = isinstance(started, (int, float)) and started >= _PROCESS_STARTED
            run = last if ours else _new_run(sample.stage)
        finish_id = run.get("finish_id") if isinstance(run, dict) else None
        finish = next((item for item in finishes if item.get("finish_id") == finish_id), None)
        if finish is None:
            run = run or _new_run(sample.stage)
            finish = {**{k: run[k] for k in ("finish_id", "started_at", "pid", "context")},
                      "stages": {}}
            finishes.append(finish)
        stages = finish["stages"]
        records = stages.get(sample.stage, [])
        if not isinstance(records, list):
            records = []
        if sample.stage == "solve_draw":
            record["draw_index"] = len(records)
            record["seed"] = sample.seed
        stages[sample.stage] = (records + [record])[-LIMITS[sample.stage]:]
        doc["finishes"] = finishes[-FINISH_LIMIT:]
        write_json_atomic(path, doc)
    except Exception:  # noqa: BLE001 -- diagnostics cannot fail a world build
        try:
            logger.warning("world stage timing could not write a record", exc_info=True)
        except Exception:  # noqa: BLE001
            pass


def _finish(sample, parent, status):
    try:
        record = {"wall_ms": max(0, round((time.perf_counter() - sample.started) * 1000, 3)),
                  "gpu_wait_ms": 0.0, "gpu_wait_source": "no_blocking_gpu_lock",
                  "cache": sample.caches, "status": status}
        if sample.concurrent is not None:
            record["consensus_concurrent"] = sample.concurrent
        sample.pending.append((sample, record))
        if parent is not None:
            parent.pending.extend(sample.pending)
        else:
            store, world_id, session_id = sample.identity
            for child, entry in sample.pending:
                _write(store, world_id, session_id, child, entry)
    except Exception:  # noqa: BLE001 -- including malformed store and clock failures
        pass


def _flush(sample):
    try:
        store, world_id, session_id = sample.identity
        for child, entry in sample.pending:
            _write(store, world_id, session_id, child, entry)
    except Exception:  # noqa: BLE001 -- diagnostics cannot replace the stage result
        pass


def defer(fn):
    """Flush nested room records after the room report is complete."""
    @functools.wraps(fn)
    def wrapper(store, world_id, session_id, *args, **kwargs):
        if not world_stage_timing_setting() or _active.get() is not None:
            return fn(store, world_id, session_id, *args, **kwargs)
        try:
            buffer = _Sample("buffer", (store, world_id, session_id))
            token = _active.set(buffer)
        except Exception:  # noqa: BLE001
            return fn(store, world_id, session_id, *args, **kwargs)
        try:
            return fn(store, world_id, session_id, *args, **kwargs)
        finally:
            try:
                _active.reset(token)
            except Exception:  # noqa: BLE001
                pass
            _flush(buffer)
    return wrapper


def timed(stage: str, *, store_arg: bool = True, final_only: bool = False):
    """Decorate a finish stage; unavailable identity falls back to the plain call."""
    if stage not in LIMITS:
        raise ValueError(stage)

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not world_stage_timing_setting() or (final_only and not kwargs.get("final", False)):
                return fn(*args, **kwargs)
            try:
                parent = _active.get()
                skip = parent is not None and parent.stage == "areas" and stage in ("surface", "appearance")
                if store_arg:
                    skip = skip or len(args) < 3
                    identity = args[:3] if not skip else None
                elif parent is not None:
                    identity = parent.identity
                else:
                    skip = True
                    identity = None
                if not skip:
                    sample = _Sample(stage, identity, parent, seed=kwargs.get("seed"))
                    token = _active.set(sample)
            except Exception:  # noqa: BLE001 -- setup cannot prevent the stage call
                return fn(*args, **kwargs)
            if skip:
                return fn(*args, **kwargs)
            status = "ok"
            try:
                return fn(*args, **kwargs)
            except BaseException:
                status = "raised"
                raise
            finally:
                try:
                    _active.reset(token)
                except Exception:  # noqa: BLE001
                    pass
                _finish(sample, parent, status)
        return wrapper
    return decorate


def measure(stage: str, store, world_id: str, session_id: str, fn, *args, **kwargs):
    """Measure a call site whose public function also serves live builds."""
    if not world_stage_timing_setting():
        return fn(*args, **kwargs)
    try:
        parent = _active.get()
        skip = parent is not None and parent.stage == "areas" and stage in ("surface", "appearance")
        if not skip:
            sample = _Sample(stage, (store, world_id, session_id), parent)
            token = _active.set(sample)
    except Exception:  # noqa: BLE001
        return fn(*args, **kwargs)
    if skip:
        return fn(*args, **kwargs)
    status = "ok"
    try:
        return fn(*args, **kwargs)
    except BaseException:
        status = "raised"
        raise
    finally:
        try:
            _active.reset(token)
        except Exception:  # noqa: BLE001
            pass
        _finish(sample, parent, status)


def cached(name: str):
    """Count truthy cached reads without changing their value or exceptions."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            cache(name, result is not None)
            return result
        return wrapper
    return decorate
