"""Best-effort depth predictions made while keyframes arrive.

The sidecar is deliberately separate from the solved depth stage. A final solve
may omit or reorder keyframes; ``offer_live_predictions`` validates and moves
only matching predictions into that stage's index space.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import queue
import shutil
import threading
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

# EXP1 frozen parameters (prereg d6995545..., design sections 1, 2, 5).
MODEL = "moge2-vitl"
FOV_SOURCE = "session calibration through global_solve._undistort_maps; cropped pinhole fx/width"
SAMPLING = "every confirmed keyframe, no fixed stride; bounded drop-oldest queue"
MASK = "MoGe out['mask'] via MoGeBackend.predict; existing FILL_RULE"
CACHE_IDENTITY = "kid, image_sha1, backend, imagery_source, redaction_trust, fill_rule, numeric calibrated FoV"
QUEUE_SIZE = 64
LATENCY_WINDOW = 64
P95_BUDGET_MS = 15.0
VRAM_SAMPLE_EVERY = 10
VRAM_MIN_FREE_FRACTION = 0.10
RESUME_IN_BUDGET_SAMPLES = 3


def mode() -> str:
    value = os.environ.get("TOWER_WORLD_LIVE_DEPTH", "off")
    return value if value in ("shadow", "on") else "off"


def _vram_sample():
    """Driver free/total and Tower reserved bytes; None on a CPU host."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {"free_mib": free / 1048576, "total_mib": total / 1048576,
                "tower_reserved_mib": torch.cuda.memory_reserved() / 1048576}
    except Exception:  # monitor failure must not cost a keyframe
        logger.exception("[Tower][WorldBuilder] live depth VRAM probe failed")
        return None


def _p95(values) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


class LiveDepthWorker:
    def __init__(self, store, world_id, session_id, intrinsics, *,
                 monotonic=time.monotonic, vram_probe=_vram_sample,
                 backend_factory=None):
        self.store, self.world_id, self.session_id = store, world_id, session_id
        self.intrinsics = intrinsics
        self.clock, self.vram_probe = monotonic, vram_probe
        self.backend_factory = backend_factory
        self.root = store.world_dir(world_id) / "dense" / session_id / "live_depth"
        self.jobs = queue.Queue(maxsize=QUEUE_SIZE)
        self.lock = threading.Lock()
        self.latencies = deque(maxlen=LATENCY_WINDOW)
        self.backed_off = False
        self.good_samples = 0
        self.closed = False
        self.disabled = False
        self.counts = {"submitted": 0, "predicted": 0, "skipped": 0,
                       "dropped": 0, "failed": 0, "back_off_engagements": 0}
        self.vram = None
        self.label = None
        self.known_fov = None
        self.dirty = False
        self.peak_tower_reserved_mib = 0.0
        self.peak_driver_used_mib = 0.0
        self.records = []
        self.thread = threading.Thread(target=self._run, name="wb-live-depth", daemon=True)
        self.thread.start()

    def _sample(self, *, force=False):
        with self.lock:
            due = force or self.counts["predicted"] % VRAM_SAMPLE_EVERY == 0
        if due:
            # The driver probe can be slow; never hold the submitter's lock.
            sample = self.vram_probe()
            with self.lock:
                self.vram = sample
                if self.vram is not None:
                    self.peak_tower_reserved_mib = max(
                        self.peak_tower_reserved_mib, self.vram["tower_reserved_mib"])
                    self.peak_driver_used_mib = max(
                        self.peak_driver_used_mib,
                        self.vram["total_mib"] - self.vram["free_mib"])
                self.dirty = True
        return self.vram

    def _over_budget(self):
        vram = self.vram
        return (_p95(self.latencies) > P95_BUDGET_MS or
                (vram is not None and vram["free_mib"] <
                 VRAM_MIN_FREE_FRACTION * vram["total_mib"]))

    def _drain(self):
        while True:
            try:
                self.jobs.get_nowait()
                self.jobs.task_done()
                self.counts["dropped"] += 1
            except queue.Empty:
                return

    def _drop_oldest(self):
        try:
            self.jobs.get_nowait()
            self.jobs.task_done()
            self.counts["dropped"] += 1
        except queue.Empty:
            pass

    def submit(self, keyframe, image_bytes, raw_bytes, redaction_label):
        start = self.clock()
        with self.lock:
            if self.closed:
                return
            self.label = redaction_label
            if self.disabled:
                self.counts["skipped"] += 1
                self.dirty = True
                return
            if self.backed_off:
                if self._over_budget():
                    self.good_samples = 0
                else:
                    self.good_samples += 1
                    if self.good_samples >= RESUME_IN_BUDGET_SAMPLES:
                        self.backed_off = False
                        self.good_samples = 0
            if self.backed_off:
                self.counts["skipped"] += 1
            else:
                job = (keyframe, image_bytes, raw_bytes, redaction_label)
                try:
                    self.jobs.put_nowait(job)
                    self.counts["submitted"] += 1
                except queue.Full:
                    self._drop_oldest()
                    self.jobs.put_nowait(job)
                    self.counts["submitted"] += 1
            self.latencies.append((self.clock() - start) * 1000.0)
            if not self.backed_off and self._over_budget():
                self.backed_off = True
                self.good_samples = 0
                self.counts["back_off_engagements"] += 1
                self._drain()
            self.dirty = True

    def close(self):
        with self.lock:
            self.closed = True
            self._drain()
            self.dirty = True

    def _run(self):
        from tower.world_builder.appearance import label_is_trusted

        backend = None
        while True:
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                self._flush_record()
                if self.closed:
                    return
                if self.backed_off:
                    self._sample(force=True)
                continue
            try:
                if not label_is_trusted(job[3]) or not self.intrinsics.is_known:
                    with self.lock:
                        self.counts["skipped"] += 1
                        self.dirty = True
                    continue
                if backend is None:
                    try:
                        if self.backend_factory is None:
                            from tower.world_builder.dense import make_backend
                            backend = make_backend(MODEL)
                        else:
                            backend = self.backend_factory()
                    except Exception:
                        logger.exception("[Tower][WorldBuilder] live depth model unavailable")
                        with self.lock:
                            self.disabled = True
                            self.counts["failed"] += 1
                            self.dirty = True
                            self._drain()
                        continue
                self._predict(backend, *job)
            except Exception:  # a depth failure must never lose a keyframe
                logger.exception("[Tower][WorldBuilder] live depth prediction failed")
                with self.lock:
                    self.counts["failed"] += 1
                    self.dirty = True
            finally:
                self.jobs.task_done()

    def _predict(self, backend, keyframe, image_bytes, raw_bytes, label):
        import cv2
        import numpy as np

        from tower.world_builder.appearance import label_is_trusted
        from tower.world_builder.dense_pipeline import FILL_RULE, _fill_mask_for, redaction_fill_mask
        from tower.world_builder.global_solve import _undistort_maps

        # No raw or ambiguously redacted image may enter the live cache.
        if not label_is_trusted(label):
            return
        raw = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if raw is None or raw.shape[:2] != (keyframe.height, keyframe.width):
            return
        m1, m2, (x, y, width, height), camera = _undistort_maps(
            self.intrinsics, keyframe.width, keyframe.height)
        fov_x = math.degrees(2.0 * math.atan(width / (2.0 * camera.fx)))
        img = cv2.remap(raw, m1, m2, cv2.INTER_LINEAR)[y:y + height, x:x + width]
        exact_fill = _fill_mask_for(image_bytes, raw_bytes)
        fill = (cv2.dilate(exact_fill.astype(np.uint8), np.ones((3, 3), np.uint8),
                           iterations=3).astype(bool) if exact_fill is not None
                else redaction_fill_mask(raw, None))
        fill_u = cv2.remap(fill.astype(np.uint8) * 255, m1, m2,
                           cv2.INTER_NEAREST)[y:y + height, x:x + width] > 0
        if fill_u.any():
            img = cv2.inpaint(img, fill_u.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = backend.predict(rgb, fov_x=fov_x)
        ki = int(keyframe.source_seq)
        work = self.root / "work"
        (work / "depth").mkdir(parents=True, exist_ok=True)
        (work / "undist").mkdir(parents=True, exist_ok=True)
        np.save(work / "depth" / f"{ki:05d}_pred.npy", np.asarray(pred).astype(np.float16))
        np.save(work / "depth" / f"{ki:05d}_fill.npy", fill_u)
        cv2.imwrite(str(work / "undist" / f"{ki:05d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        record = {"ki": ki, "kid": keyframe.keyframe_id,
                  "image_sha1": hashlib.sha1(image_bytes).hexdigest(),
                  "fill_rule": FILL_RULE, "pred": f"work/depth/{ki:05d}_pred.npy",
                  "known_fov": round(fov_x, 6)}
        with self.lock:
            self.records.append(record)
            self.counts["predicted"] += 1
            self.known_fov = round(fov_x, 6)
            self.dirty = True
        self._sample()
        self._flush_record()

    def _flush_record(self):
        from tower.world_builder.appearance import label_is_trusted, pixel_trust_token
        from tower.world_builder.dense_pipeline import FILL_RULE

        with self.lock:
            if not self.dirty:
                return
            payload = {"backend": MODEL, "imagery_source": "redacted",
                       "redaction_trust": (pixel_trust_token(self.label)
                                           if label_is_trusted(self.label) else None),
                       "keyframe_image_set": None, "known_fov": self.known_fov,
                       "fill_rule": FILL_RULE, "records": list(self.records),
                       "live": {**self.counts, "p95_confirm_ms": _p95(self.latencies),
                                "vram": self.vram,
                                "peak_tower_reserved_mib": self.peak_tower_reserved_mib,
                                "peak_driver_used_mib": self.peak_driver_used_mib,
                                "backed_off": self.backed_off}}
            self.dirty = False
        temp = self.root / "align.json.tmp"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            temp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            temp.replace(self.root / "align.json")
        except OSError:
            logger.exception("[Tower][WorldBuilder] live depth sidecar write failed")
            with self.lock:
                self.dirty = True


def offer_live_predictions(store, world_id, session_id, solution, intrinsics,
                           root: Path, *, imagery_source: str, trust: str) -> dict:
    """Stage only exact live matches at final solve indices; return reader-shaped offers."""
    live = root / "live_depth"
    offers = {}
    work = root / "work"
    for ki, (kid, image_sha1, old_ki) in compatible_live_records(
            store, world_id, session_id, solution, intrinsics, root,
            imagery_source=imagery_source, trust=trust).items():
        try:
            for directory, src, dst in _live_files(old_ki, ki):
                target = work / directory
                target.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(live / "work" / directory / src, target / dst)
            offers[ki] = (kid, image_sha1)
        except OSError:
            continue
    return offers


def _live_files(old_ki: int, ki: int) -> list:
    return [("depth", f"{old_ki:05d}_pred.npy", f"{ki:05d}_pred.npy"),
            ("depth", f"{old_ki:05d}_fill.npy", f"{ki:05d}_fill.npy"),
            ("undist", f"{old_ki:05d}.jpg", f"{ki:05d}.jpg")]


def compatible_live_records(store, world_id, session_id, solution, intrinsics,
                            root: Path, *, imagery_source: str, trust: str) -> dict:
    """`{final ki: (kid, image_sha1, live ki)}` for every live prediction whose FULL
    identity matches this solve (backend, fill rule, imagery, trust, keyframe image
    set, numeric calibrated FoV == the solve's FoV, camera size, the keyframe's
    current image bytes, the prediction's shape). READ-ONLY: nothing is copied or
    written; `offer_live_predictions` stages these for the depth stage, and the
    final scale guard (v2) reads them in place as publication evidence."""
    import numpy as np

    from tower.world_builder.dense_pipeline import FILL_RULE
    from tower.world_builder.global_solve import _undistort_maps

    live = root / "live_depth"
    try:
        document = json.loads((live / "align.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    camera = solution.camera or {}
    if (document.get("backend") != MODEL or document.get("fill_rule") != FILL_RULE
            or document.get("imagery_source") != imagery_source
            or document.get("redaction_trust") != trust
            or document.get("keyframe_image_set") !=
            store.keyframe_image_set(world_id, session_id).cache_token):
        return {}
    try:
        width, height = int(camera["width"]), int(camera["height"])
        final_fov = round(math.degrees(2.0 * math.atan(width /
                                (2.0 * float(camera["fx"])))), 6)
        frame = next(iter(store.read_keyframes(world_id, session_id)))
        _m1, _m2, _roi, calibrated = _undistort_maps(
            intrinsics, frame.width, frame.height)
        calibrated_fov = round(math.degrees(2.0 * math.atan(
            calibrated.width / (2.0 * calibrated.fx))), 6)
    except (KeyError, ValueError, StopIteration, AttributeError, TypeError):
        return {}
    if (document.get("known_fov") != calibrated_fov or
            final_fov != calibrated_fov or
            (width, height) != (calibrated.width, calibrated.height)):
        return {}
    by_kid = {r.get("kid"): r for r in document.get("records", [])
              if isinstance(r, dict) and r.get("known_fov") == final_fov
              and r.get("fill_rule") == FILL_RULE}
    matches = {}
    for ki, kid in enumerate(solution.keyframe_ids):
        record = by_kid.get(kid)
        if record is None:
            continue
        try:
            source = (store.keyframe_image_set(world_id, session_id).directory /
                      f"{kid.rsplit(':', 1)[-1]}.jpg")
            image_sha1 = hashlib.sha1(source.read_bytes()).hexdigest()
            old_ki = int(record["ki"])
            if image_sha1 != record.get("image_sha1") or any(
                    not (live / "work" / d / src).is_file()
                    for d, src, _ in _live_files(old_ki, ki)):
                continue
            prediction = np.load(live / "work" / "depth" /
                                 f"{old_ki:05d}_pred.npy", mmap_mode="r")
            if prediction.shape != (height, width):
                continue
            matches[ki] = (kid, image_sha1, old_ki)
        except (OSError, ValueError, KeyError):
            continue
    return matches
