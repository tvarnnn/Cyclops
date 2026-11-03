"""Transient masks at Stop: the final build's masks, computed while the final solve runs.

Switch: `TOWER_WORLD_MASKS_AT_STOP` (`config.world_masks_at_stop_setting`), default off. Off,
nothing in this module runs and the builder and the stages are byte for byte as before.

WHY. After Stop the builder waits for the background solve (up to 120 s) and then runs the
final global solve, which on this Tower is CPU work (pycolmap has no CUDA build on Windows).
On walk 10.2 that was 556 s with the GPU idle. Then, after the solve, the surface stage
computes the transient masks of every fused keyframe and the appearance stage those of its
extra candidates (`transients.ensure_transient_masks`): 397 + 42 s of GPU on that walk, on
the critical path. Nearly everything a mask is made from is fixed at Stop -- the stored
keyframe JPEG, the redaction label `stop_session` just wrote, the session calibration, the
models -- so those masks can be computed in the idle window and only looked up afterwards.

TWO HALVES, AND THE CACHE IS NEVER WRITTEN BY THE FIRST.

1. `prefill` runs in a child the builder starts right after `stop_session` and stops when
   the final solve returns (`scripts/world_masks_at_stop.py`). It calls the transient stage
   UNCHANGED -- `ensure_transient_masks`, the same backends, batch 1, the same fp16 -- on
   chunks of keyframes, most-likely-needed first, with `depth_dir` = a STAGING directory of
   its own, `<store root>/masks_at_stop/<run id>/`. The masks it writes there are the cache's own files
   under the cache's own key (`transients.component_key`), so the manifest's
   `frames_digest` of a build that uses them is the one it would have had anyway.
2. `promote` runs in the final surface stage after its depth stage, just before its
   transient step. A staged mask moves into the real cache ONLY when the final's own inputs
   are the staged ones: the same key (the keyframe at this index of the FINAL solve, the
   SHA-1 of its stored JPEG, the label, the rule, the models); the same calibration and
   camera size; the same execution mode (CUDA, fp16 forward calls, torch and transformers
   versions: the key has none of these); the final depth record naming the same image bytes
   under `FILL_RULE`; and the final `_fill.npy` equal to the staged one. Anything else stays
   staged and the stage computes it exactly as it does today. Then `ensure_transient_masks`
   finds hits.

ONE RUN'S FILES, NEVER TWO. The index describes ONE prefill run (its world, session,
calibration, camera, execution mode), and a staged mask's key does not carry its fill or its
calibration. So every run stages into a directory of its own, an 8-hex run id created
exclusively (`os.mkdir`, which fails if the name exists: two runs, even at the same moment,
never share one), and its index names that id. `promote` reads exactly one run: the one of
this session whose complete index (written whole, by replace) is newest. A mask of another
run -- earlier, or interleaved with this one -- is never checked against this run's index or
fills. After promoting, the session's runs are removed (and by the builder when no surface
stage follows).

SHORT PATHS (Windows MAX_PATH, 260). The staging lives at the STORE ROOT, not under
`worlds/<wid>/dense/<sid>/work/` (two 32-hex ids and four directories: 99 characters before
any file, which put a staged mask's temp file at 278 characters on a bench root). The longest
staging path is `<store root>/masks_at_stop/<8 hex>/<ki>_transient.oneformer.npz.p<pid>.tmp`:
the store root plus 66 characters (`test_..._paths_stay_short`).

THE EXECUTION MODE OF EVERY MASK. The key has no device or dtype, and the stage's backends
choose CUDA (fp16 forward calls) or the CPU (fp32) when they run. The worker's backends are
held to CUDA (`device="cuda"`: a forward call that cannot run there fails, it never moves to
the CPU), the worker declines outright without a CUDA device, and every mask it stages gets a
`<name>.mode.json` beside it naming the device and dtype it was computed with. `promote`
requires that record, and the index's (device, fp16, library versions), to equal the stage's.
The records are one line each in the run's `modes.jsonl`, appended after the mask is written;
a torn last line is a missing record.

THE ONE INPUT NOT FIXED AT STOP IS THE FILL MASK. Grounding DINO + SAM drops a mask that
lies mostly inside redaction fill (`filter_instance_masks`), so the cached gdsam masks depend
on `_fill.npy`, which the final depth stage writes after the solve. The fill is a function of
the keyframe bytes and its raw frame only (no pose), and the depth stage takes it from the
live depth worker's sidecar, else from the last live depth stage, else computes it
(`surface_pipeline.ensure_depth_stage`, `dense_pipeline.run_depth_stage`). `prefill` stages
the fill the final is expected to use by the same three rules, and `promote` checks the
expectation against the file the final actually wrote: a wrong guess costs a recomputation,
never a different mask.

WHAT IS LEFT TO THE STAGE, AS TODAY: every keyframe the worker did not reach, every staged
mask the checks refuse, and every case the worker declines -- a label that is not trusted at
Stop (the re-redaction path's fill is not the stored one), no CUDA device (the worker exists
for the idle GPU, and a CPU run would be fp32 and load the cores the final solve needs), a
detector that is unavailable; mode `off` stages nothing. The worker stages for the imagery
the builder's final stages use, `SurfaceParams().imagery_source`, which is always `redacted`:
the research bypass (`raw-local-research`) is only reachable from the standalone stage CLIs,
which start no worker, and `promote` refuses every staged mask for such a build.

LIVE MASKS (`TOWER_WORLD_MASKS_LIVE_REUSE`, default off; only with this module's switch on). During
the walk `session.json` says redaction `none` (`stop_session` writes the real label), so the live
surface child reads every keyframe through the re-redaction path and keys its OneFormer masks under
`none&<redactor>`: the final build, keyed under the real label, recomputes every one of them. Yet a
OneFormer mask is a function of the undistorted pixels alone (it never reads the fill), the
model, gamma, threshold and execution mode; and when the re-redaction finds nothing to fill (the
stored bytes are already redacted) the live detector saw EXACTLY the stored bytes, which the final
reads under a trusted label. With the switch on:
1. a live surface build appends, after each mask it writes, one line to `LIVE_JOURNAL` beside its
   masks: the file name, the SHA-1 of the file it just wrote, the execution mode it was computed
   in (device, fp16, torch, transformers) and the session calibration's digest;
2. the worker, before its GPU loop, stages under the FINAL key every live OneFormer mask whose
   journal line names the file's exact bytes, whose mode and calibration are the worker's own,
   whose key is the final key in every field but `redaction_effective`, and whose recorded
   `image_sha1` is the stored keyframe's SHA-1 now (the re-redaction changed nothing). The worker
   then computes only Grounding DINO + SAM for those frames. `promote` checks a re-keyed mask
   exactly as it checks one the worker computed.

GPU. One child, models loaded one at a time inside `ensure_transient_masks` as in the stage
(at most ~2.6 GB on the card). Chunks of `CHUNK_FRAMES`: each chunk reloads the three models
(about 5 s on 10.2: Grounding DINO 2.9, SAM 0.6, OneFormer 1.6), and a stop in a chunk's
Grounding DINO pass loses that pass (its boxes are emitted only after SAM). 256 frames is
about 85 s of GPU: reload overhead ~6% and an expected loss at the stop of ~9 s.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from tower.world_builder import transients as T

logger = logging.getLogger(__name__)

SCHEMA = 3
STAGING_DIRNAME = "masks_at_stop"
RUN_ID_HEX = 8
INDEX_FILENAME = "index.json"
MODES_FILENAME = "modes.jsonl"
LIVE_JOURNAL = "transient_live.jsonl"     # beside the live build's masks (see LIVE MASKS)
LIVE_ORIGIN = "live-surface"              # `staged_from` of a re-keyed live mask's record
EVIDENCE_LISTING = "promoted.jsonl"
CHUNK_FRAMES = 256

LOG = "[Tower][WorldBuilder][masks-at-stop]"


def staging_root(store) -> Path:
    """`<store root>/masks_at_stop`: one directory per prefill run (any world, any session; each
    run's index says whose), and the builder's raw-frame paths per session. At the store root,
    beside the depth `predictions` cache, to keep every path short (see SHORT PATHS)."""
    return Path(store.root) / STAGING_DIRNAME


def sources_path(store, world_id: str, session_id: str) -> Path:
    """The builder's `keyframe_id -> raw frame path` for one session's worker, named by a short
    digest of the world AND the session (two worlds' sessions never share or delete one)."""
    key = hashlib.sha1(f"{world_id}/{session_id}".encode()).hexdigest()[:16]
    return staging_root(store) / f"sources-{key}.json"


def evidence_dir(store, world_id: str, session_id: str) -> Path:
    """Where `promote` keeps what it promoted when `TOWER_WORLD_MASKS_AT_STOP_EVIDENCE` is on: each
    mask moved into the cache and its frame's final fill, hard-linked (they outlive the prune of
    `work/`), and `promoted.jsonl`. Never an 8-hex name, so no discard ever removes it."""
    key = hashlib.sha1(f"{world_id}/{session_id}".encode()).hexdigest()[:16]
    return staging_root(store) / f"evidence-{key}"


def _keep(src: Path, dst: Path) -> None:
    """`dst` as a hard link to `src` (same volume), else a copy; an existing `dst` is replaced."""
    import shutil  # noqa: PLC0415

    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _is_run_name(name: str) -> bool:
    return len(name) == RUN_ID_HEX and all(c in "0123456789abcdef" for c in name)


def new_run_dir(root: Path) -> Path:
    """A run directory no other run has or will have: a random 8-hex id, created with
    `os.mkdir`, which fails rather than reuse an existing name (a taken id is drawn again)."""
    import uuid  # noqa: PLC0415

    root.mkdir(parents=True, exist_ok=True)
    while True:
        path = root / uuid.uuid4().hex[:RUN_ID_HEX]
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            continue


def session_runs(root: Path, world_id: str, session_id: str) -> list:
    """`[(run dir, its index)]` for every run of this session with a complete index. An index is
    whole or absent (written by replace), names its own directory and its world and session;
    anything else is not this session's run and is never read."""
    out = []
    try:
        candidates = [p for p in root.iterdir() if p.is_dir() and _is_run_name(p.name)]
    except OSError:
        return out
    for run in candidates:
        try:
            index = json.loads((run / INDEX_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(index, dict) and index.get("run") == run.name
                and index.get("world_id") == world_id and index.get("session_id") == session_id):
            out.append((run, index))
    return out


def latest_run(root: Path, world_id: str, session_id: str):
    """`(run dir, its index)` of this session's run with the newest complete index, or None."""
    runs = session_runs(root, world_id, session_id)
    if not runs:
        return None
    return max(runs, key=lambda r: (float(r[1].get("started_at") or 0.0), r[0].name))


def worker_gone(index: dict) -> bool:
    """Whether the worker that wrote `index` is known to be finished: its run ended (any state
    but `running`), or the process the index names (pid + start time) no longer exists. Unknown
    is NOT gone."""
    if index.get("state") not in (None, "running"):
        return True
    pid, started = index.get("pid"), index.get("pid_started")
    if pid is None or started is None:
        return False
    import psutil  # noqa: PLC0415

    try:
        return psutil.Process(int(pid)).create_time() != started
    except psutil.NoSuchProcess:
        return True
    except (psutil.Error, ValueError, TypeError):
        return False


def discard_session(store, world_id: str, session_id: str) -> None:
    """Remove this session's FINISHED runs (`worker_gone`) and its raw-frame paths; never raises.
    A run whose worker may still be writing -- a termination that failed -- is left for a later
    discard, as is a run of another session or one too young to have an index."""
    import shutil  # noqa: PLC0415

    root = staging_root(store)
    for run, index in session_runs(root, world_id, session_id):
        if worker_gone(index):
            shutil.rmtree(run, ignore_errors=True)
    try:
        sources_path(store, world_id, session_id).unlink(missing_ok=True)
    except OSError:
        pass


def log_path(store, world_id: str, session_id: str) -> Path:
    """The worker's own log, `dense/<sid>/masks_at_stop.log`: beside the work, not in it, so
    it outlives the prune and says afterwards what the worker staged."""
    from tower.world_builder.dense_pipeline import dense_dir  # noqa: PLC0415

    return dense_dir(store, world_id, session_id) / "masks_at_stop.log"


def intrinsics_record(intrinsics) -> dict:
    """The calibration the undistortion maps are made from, as plain JSON."""
    out = {}
    for k in ("fx", "fy", "cx", "cy", "dist_coeffs", "calibrated_width", "calibrated_height"):
        v = getattr(intrinsics, k, None)
        out[k] = list(v) if isinstance(v, (tuple, list)) else v
    return out


def intrinsics_digest(intrinsics) -> str:
    """A short digest of `intrinsics_record`: the calibration a live mask was undistorted with."""
    return hashlib.sha1(json.dumps(intrinsics_record(intrinsics), sort_keys=True).encode()).hexdigest()[:16]


def keyframe_order(keyframe_ids: list, solution, first=frozenset()) -> list:
    """Indices into `keyframe_ids` (the final solve's `ki`), most likely needed first.

    The final needs the keyframes it poses in its main component (the surface fuses the
    depth-gated ones, the appearance offers a few more). At Stop the best evidence is the
    latest background solution: (1) posed in its largest component, (2) newer than it,
    (3) posed in another of its components, (4) in it but unposed. Index order within each,
    except that the keyframes in `first` lead their group: those with a reusable live mask
    (LIVE MASKS), which a live surface already gated and which need only one model more.
    No solution: index order (`first` leading)."""
    if solution is None:
        groups: tuple = (list(range(len(keyframe_ids))),)
    else:
        known = set(solution.keyframe_ids)
        groups = ([], [], [], [])
        for ki, kid in enumerate(keyframe_ids):
            pose = solution.poses.get(kid)
            if kid not in known:
                groups[1].append(ki)
            elif not pose or pose.get("rotation") is None:
                groups[3].append(ki)
            elif int(pose.get("component", 0)) == 0:
                groups[0].append(ki)
            else:
                groups[2].append(ki)
    return [ki for g in groups for lead in (True, False) for ki in g if (ki in first) == lead]


@dataclasses.dataclass
class _Camera:
    """What `compatible_live_records` reads of a solution: the final's index space and camera."""
    keyframe_ids: list
    camera: dict


def _expected_fills(store, world_id, session_id, keyframe_ids, camera, intrinsics, *,
                    imagery_source) -> dict:
    """`{ki: (fill file, image_sha1)}`: the `_fill.npy` the final depth stage is expected to
    REUSE for keyframe `ki` -- the live depth worker's when its record matches
    (`live_depth.compatible_live_records`, which wins: `offer_live_predictions` copies it over),
    else the last live depth stage's (`dense_pipeline.reusable_predictions`, with its
    prediction, fill and undistorted frame on disk). Read only."""
    import math  # noqa: PLC0415

    from tower.world_builder.dense import DenseParams  # noqa: PLC0415
    from tower.world_builder.dense_pipeline import (  # noqa: PLC0415
        dense_dir,
        depth_trust_now,
        reusable_predictions,
    )
    from tower.world_builder.live_depth import compatible_live_records  # noqa: PLC0415

    root = dense_dir(store, world_id, session_id)
    work = root / "work"
    out: dict = {}
    try:
        fov = round(math.degrees(2.0 * math.atan(int(camera["width"]) / (2.0 * float(camera["fx"])))), 6)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        fov = None
    prior = reusable_predictions(root / "align.json", DenseParams().backend, imagery_source, known_fov=fov)
    for ki, (kid, sha) in prior.items():
        if (0 <= ki < len(keyframe_ids) and keyframe_ids[ki] == kid
                and (work / "depth" / f"{ki:05d}_pred.npy").is_file()
                and (work / "depth" / f"{ki:05d}_fill.npy").is_file()
                and (work / "undist" / f"{ki:05d}.jpg").is_file()):
            out[ki] = (work / "depth" / f"{ki:05d}_fill.npy", sha)
    trust = depth_trust_now(store, world_id, session_id, imagery_source=imagery_source)
    live = compatible_live_records(store, world_id, session_id, _Camera(list(keyframe_ids), camera),
                                   intrinsics, root, imagery_source=imagery_source, trust=trust)
    for ki, (_kid, sha, old_ki) in live.items():
        out[ki] = (root / "live_depth" / "work" / "depth" / f"{int(old_ki):05d}_fill.npy", sha)
    return out


def _fresh_fill(store, world_id, session_id, kid, source_path, image_set, intrinsics, imagery_source):
    """`(fill_u, image_sha1)` as `run_depth_stage` computes a frame it does not reuse, for a
    trusted label (the only one the worker runs under): the stored bytes, the fill against the
    raw frame (`keyframe_image_bytes`), dilated 3x3 three times, else the shape-gated guess,
    remapped nearest-neighbour into the solve camera. (None, None) when the stage would
    refuse the frame. `promote` compares the result with the final's own file."""
    import cv2  # noqa: PLC0415

    from tower.world_builder.dense_pipeline import (  # noqa: PLC0415
        keyframe_image_bytes,
        redaction_fill_mask,
    )
    from tower.world_builder.global_solve import _undistort_maps  # noqa: PLC0415

    data, _origin, exact_fill = keyframe_image_bytes(
        store, world_id, session_id, kid, source_path, None, keyframes_are_redacted=True,
        image_set=image_set, imagery_source=imagery_source)
    if data is None:
        return None, None
    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        return None, None
    m1, m2, (x0, y0, rw, rh), _cam = _undistort_maps(intrinsics, raw.shape[1], raw.shape[0])
    fill = (cv2.dilate(exact_fill.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3).astype(bool)
            if exact_fill is not None else redaction_fill_mask(raw, None))
    fill_u = cv2.remap(fill.astype(np.uint8) * 255, m1, m2, cv2.INTER_NEAREST)[y0:y0 + rh, x0:x0 + rw] > 0
    return fill_u, hashlib.sha1(data).hexdigest()


def stage_device() -> str | None:
    """The device the `transients` backends would pick now: CUDA when available, else CPU
    (None without torch). The surface stage asks ONCE, then builds its detectors on exactly this
    device and promotes against it, so the two can never disagree."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    return "cuda" if torch.cuda.is_available() else "cpu"


def execution_mode(device: str | None = None) -> dict:
    """How detector backends run on `device` (None: the device they would pick now,
    `stage_device`): the device, whether their forward calls are fp16
    (`transients._fp16_enabled`), and the library versions. Not part of the cache key, so the
    worker records it and `promote` requires the stage's own to be equal."""
    import importlib.metadata  # noqa: PLC0415

    dev = device if device is not None else stage_device()
    try:
        import torch  # noqa: PLC0415

        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None
    return {"device": dev, "fp16": bool(dev and T._fp16_enabled(dev)), "torch": torch_version,
            "transformers": transformers_version}


def _this_process() -> dict:
    """The pid and start time `worker_gone` checks, so a cleanup never removes a live run."""
    try:
        import psutil  # noqa: PLC0415

        return {"pid": os.getpid(), "pid_started": psutil.Process(os.getpid()).create_time()}
    except Exception:  # noqa: BLE001 -- without them the run is never taken for finished
        return {"pid": os.getpid(), "pid_started": None}


def mask_mode(device: str) -> dict:
    """What a backend run on `device` computes with: the device, and whether its forward calls
    are fp16 (`transients._fp16_enabled`)."""
    return {"device": device, "fp16": bool(T._fp16_enabled(device))}


class _HeldToCuda(T.ComponentBackend):
    """The stage's backend, run on CUDA only, appending each emitted mask's execution mode to
    the run's `modes.jsonl`. The record is written after the mask (`emit` writes it), so a mask
    without one -- a kill in between -- is never promoted."""

    def __init__(self, inner, run_dir: Path):
        self.inner = inner
        self.component = inner.component
        self.run_dir = run_dir
        self.mode = mask_mode(getattr(inner, "device", None) or "cuda")

    def probe(self):
        return self.inner.probe()

    def run(self, items, params, emit, should_stop=None):
        def noted(ki, hand, phone, seconds):
            emit(ki, hand, phone, seconds)
            line = json.dumps(dict(self.mode, f=T.cache_path(self.run_dir, ki, self.component).name))
            with open(self.run_dir / MODES_FILENAME, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

        return self.inner.run(items, params, noted, should_stop)


class _LiveRecorder(T.ComponentBackend):
    """A LIVE surface build's backend (LIVE MASKS): after the stage's `emit` has written a mask,
    one line is appended to the journal naming that file, the SHA-1 of its bytes as just
    written, and how it was computed. Only when the file on disk IS this write (its masks are
    the ones just computed, its `computed_at` later than the call): a write that failed leaves
    whatever was there before, which this computation says nothing about. A line cut by a kill
    is no line. Without a line a mask is simply computed again at Stop."""

    def __init__(self, inner, journal: Path, record: dict):
        self.inner = inner
        self.component = inner.component
        self.journal = journal
        self.record = record

    def probe(self):
        return self.inner.probe()

    def run(self, items, params, emit, should_stop=None):
        def noted(ki, hand, phone, seconds):
            before = time.time()
            emit(ki, hand, phone, seconds)
            path = T.cache_path(self.journal.parent, ki, self.component)
            try:
                data = path.read_bytes()
                got = _decode_component(data)
                if (got is None or float(got[2].get("computed_at") or 0.0) < before
                        or not np.array_equal(got[0], np.asarray(hand, bool))
                        or not np.array_equal(got[1], np.asarray(phone, bool))):
                    return
                line = json.dumps(dict(self.record, f=path.name, sha1=hashlib.sha1(data).hexdigest()),
                                  sort_keys=True)
                with open(self.journal, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:  # noqa: BLE001 -- an optimisation: no line, the mask is computed at Stop
                pass

        return self.inner.run(items, params, noted, should_stop)


def _live_reuse_on() -> bool:
    from tower.config import world_masks_live_reuse_setting  # noqa: PLC0415

    return world_masks_live_reuse_setting()


def live_recorder(factory, *, depth_dir, mode: dict, intrinsics):
    """`factory` with every backend journaling its masks (LIVE MASKS). `mode`: the execution mode
    of the detectors `factory` builds (`execution_mode(device)` of the device the stage decided)."""
    journal = Path(depth_dir) / LIVE_JOURNAL
    record = dict(mode, intr=intrinsics_digest(intrinsics))
    return lambda component: _LiveRecorder(factory(component), journal, record)


def _read_live_journal(depth_dir: Path) -> dict:
    """`{mask file name: its newest COMPLETE journal line}` (newline-terminated, as `_read_modes`)."""
    out = {}
    try:
        text = (Path(depth_dir) / LIVE_JOURNAL).read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.split("\n")[:-1]:
        try:
            rec = json.loads(line)
            out[rec.pop("f")] = rec
        except (ValueError, KeyError, AttributeError, TypeError):
            continue
    return out


def _decode_component(data: bytes):
    """(hand, phone, key record) of a mask file's bytes, or None, as `transients.read_component`."""
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            record = json.loads(bytes(z["key"]).decode())
            hs, ws = (int(v) for v in z["shape"])
            hand = np.unpackbits(z["hand"], axis=-1)[..., :ws].astype(bool)
            phone = np.unpackbits(z["phone"], axis=-1)[..., :ws].astype(bool)
    except Exception:  # noqa: BLE001 -- unreadable is not reusable
        return None
    if not isinstance(record, dict) or hand.shape != (hs, ws) or phone.shape != (hs, ws):
        return None
    return hand, phone, record


def live_reusable(store, world_id: str, session_id: str, *, keyframe_ids: list, image_set,
                  params: T.TransientParams, policy, mode: dict, intrinsics, shape) -> dict:
    """`{ki: (hand, phone, image_sha1)}`: the live surface's OneFormer masks that are the final
    build's own but for the label in their key (LIVE MASKS). Each must pass, in this order:
    a complete journal line for its file; the line's execution mode and calibration equal to
    `mode` (the worker's) and `intrinsics`; the file's bytes now hashing to the line's SHA-1
    (the very file that computation wrote); shape `shape` (the final camera); every field of the
    final key -- the keyframe id at this `ki`, the stored keyframe's SHA-1 now, the rule, the
    models -- but `redaction_effective` equal; a label other than the final's (an equal one is a
    plain cache hit for the stage); and `image_sha1`, the bytes the live detector actually read,
    equal to the stored keyframe's SHA-1 (the re-redaction filled nothing). Read only."""
    from tower.world_builder.dense_pipeline import dense_dir  # noqa: PLC0415

    component = T.COMPONENT_ONEFORMER
    if component not in params.components:
        return {}
    depth = dense_dir(store, world_id, session_id) / "work" / "depth"
    want = dict(mode, intr=intrinsics_digest(intrinsics))
    out: dict = {}
    for name, rec in _read_live_journal(depth).items():
        try:
            ki = int(name.split("_", 1)[0])
            if name != T.cache_path(depth, ki, component).name or not 0 <= ki < len(keyframe_ids):
                continue
        except ValueError:
            continue
        if any(rec.get(k) != v for k, v in want.items()):
            continue
        try:
            data = (depth / name).read_bytes()
        except OSError:
            continue
        if hashlib.sha1(data).hexdigest() != rec.get("sha1"):
            continue
        got = _decode_component(data)
        if got is None or got[0].shape != tuple(shape):
            continue
        hand, phone, live = got
        kid = keyframe_ids[ki]
        try:
            sha = hashlib.sha1((image_set.directory / f"{kid.rsplit(':', 1)[-1]}.jpg").read_bytes()).hexdigest()
        except OSError:
            continue
        final = T.component_key(component, params, keyframe_id=kid, source_sha1=sha,
                                redaction_effective=policy.effective)
        if any(live.get(k) != v for k, v in final.items() if k != "redaction_effective"):
            continue
        if live.get("redaction_effective") == policy.effective or live.get("image_sha1") != sha:
            continue
        out[ki] = (hand, phone, sha)
    return out


def _save_atomic(path: Path, array) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.t")
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def _write_json(path: Path, doc: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.t")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _write_index(staging: Path, doc: dict) -> None:
    _write_json(staging / INDEX_FILENAME, doc)


def write_sources(store, world_id: str, session_id: str, sources: dict) -> None:
    """The builder's `keyframe_id -> raw frame path`, as `run_final` hands it to the final
    solve's `sources.json`: the worker reads raw frames exactly where the depth stage will."""
    path = sources_path(store, world_id, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, {"sources": {k: str(v) for k, v in sources.items()}})


def read_sources(store, world_id: str, session_id: str) -> dict:
    try:
        doc = json.loads(sources_path(store, world_id, session_id).read_text(encoding="utf-8"))
        return dict(doc.get("sources") or {})
    except (OSError, ValueError, AttributeError):
        return {}


def prefill(store, world_id: str, session_id: str, *, sources: dict | None = None,
            should_stop=None, backend_factory=None, chunk: int = CHUNK_FRAMES) -> dict:
    """Stage the final build's transient masks in a new run directory under `staging_root`,
    most-likely-needed first, until done or `should_stop()`. The detector backends are held to
    CUDA. Never writes the real cache. Returns a summary dict (also the run's index).
    `backend_factory` (tests): the backends to hold, instead of the stage's own on CUDA."""
    from tower.world_builder import appearance as A  # noqa: PLC0415
    from tower.world_builder.dense_pipeline import FILL_RULE  # noqa: PLC0415
    from tower.world_builder.global_solve import _undistort_maps, load_solution  # noqa: PLC0415
    from tower.world_builder.surface import SurfaceParams  # noqa: PLC0415

    t0 = time.time()
    run_dir = new_run_dir(staging_root(store))
    surface_params = SurfaceParams()
    params = T.TransientParams(mode=surface_params.transient_detector)
    imagery_source = surface_params.imagery_source
    mode = execution_mode()
    summary = {"schema": SCHEMA, "run": run_dir.name, "world_id": world_id, "session_id": session_id,
               "rule": params.rule_id(), "execution": mode, "started_at": t0, "state": "running",
               **_this_process()}
    _write_index(run_dir, summary)          # at once: the run is this session's from its start

    def done(state, detail=None, **more):
        summary.update(state=state, detail=detail, seconds=round(time.time() - t0, 3), **more)
        _write_index(run_dir, summary)
        logger.info("%s %s/%s: %s%s", LOG, world_id, session_id, state,
                    f" ({detail})" if detail else "")
        return summary

    if params.mode == T.MODE_OFF:
        return done("declined", "transient detection is off for the final build")
    if mode["device"] != "cuda":
        return done("declined", f"no CUDA device ({mode['device']}): the worker is for the idle GPU")
    session = store.read_session(world_id, session_id)
    intrinsics = session.intrinsics
    keyframes = store.read_keyframes(world_id, session_id)
    if not keyframes or intrinsics is None:
        return done("declined", "no keyframes or no calibration")
    policy = A.resolve_label_policy(store, world_id, session_id, imagery_source=imagery_source)
    if policy.raw or not policy.trusted:
        # `policy.raw` is unreachable from the builder (its final imagery is always redacted);
        # kept so a future caller with another imagery source declines rather than stages.
        return done("declined", f"label {policy.effective!r}: only a trusted, redacted build is staged")
    first = keyframes[0]
    _m1, _m2, (_x, _y, rw, rh), cam = _undistort_maps(intrinsics, first.width, first.height)
    camera = {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy, "width": rw, "height": rh}
    keyframe_ids = [k.keyframe_id for k in keyframes]
    image_set = policy.image_set
    live = {}
    if _live_reuse_on():
        try:
            live = live_reusable(store, world_id, session_id, keyframe_ids=keyframe_ids,
                                 image_set=image_set, params=params, policy=policy, mode=mode,
                                 intrinsics=intrinsics, shape=(rh, rw))
        except Exception:  # noqa: BLE001 -- an optimisation: the worker computes them instead
            logger.exception("%s %s/%s: reading the live masks failed", LOG, world_id, session_id)
            live = {}
    order = keyframe_order(keyframe_ids, load_solution(store, world_id, session_id), first=set(live))
    expected = _expected_fills(store, world_id, session_id, keyframe_ids, camera, intrinsics,
                               imagery_source=imagery_source)
    summary.update(intrinsics=intrinsics_record(intrinsics), camera=camera,
                   redaction_effective=policy.effective, keyframes=len(keyframe_ids),
                   expected_fills=len(expected), chunk=chunk, staged=0, chunks=0,
                   fills={"reused": 0, "fresh": 0, "none": 0})
    _write_index(run_dir, summary)
    inner_factory = backend_factory or (lambda component: T.default_backend_factory(component, device="cuda"))

    def held(component):
        return _HeldToCuda(inner_factory(component), run_dir)

    sources = sources or {}
    staged_frames: dict = {}

    def stage_frame(ki):
        """Keyframe `ki`'s align record with its fill staged in the run, or None; once per `ki`."""
        if ki in staged_frames:
            return staged_frames[ki]
        staged_frames[ki] = None
        kid = keyframe_ids[ki]
        stored = image_set.directory / f"{kid.rsplit(':', 1)[-1]}.jpg"
        try:
            sha = hashlib.sha1(stored.read_bytes()).hexdigest()
        except OSError:
            return None
        fill = None
        got = expected.get(ki)
        if got is not None and got[1] == sha:
            try:
                fill = np.load(got[0])
                summary["fills"]["reused"] += 1
            except (OSError, ValueError):
                fill = None
        if fill is None:
            fill, data_sha = _fresh_fill(store, world_id, session_id, kid, sources.get(kid),
                                         image_set, intrinsics, imagery_source)
            if fill is None or data_sha != sha:
                summary["fills"]["none"] += 1
                return None
            summary["fills"]["fresh"] += 1
        _save_atomic(run_dir / f"{ki:05d}_fill.npy", np.asarray(fill, bool))
        staged_frames[ki] = {"ki": ki, "kid": kid, "ok": True, "fill_rule": FILL_RULE, "image_sha1": sha}
        return staged_frames[ki]

    if live:
        # LIVE MASKS: no GPU. Each reusable live OneFormer mask is staged under the final key,
        # with its frame's fill and a mode record, exactly as if this worker had computed it;
        # the loop below then finds it cached and runs Grounding DINO + SAM only.
        rekeyed = 0
        for ki in order:
            if ki not in live:
                continue
            if should_stop is not None and should_stop():
                return done("stopped", staged=0, rekeyed=rekeyed)
            rec = stage_frame(ki)
            hand, phone, sha = live[ki]
            if rec is None or rec["image_sha1"] != sha:
                continue
            path = T.cache_path(run_dir, ki, T.COMPONENT_ONEFORMER)
            try:
                key = T.component_key(T.COMPONENT_ONEFORMER, params, keyframe_id=rec["kid"],
                                      source_sha1=sha, redaction_effective=policy.effective)
                # `staged_from` is provenance, not key: readers match only the key's fields.
                T.write_component(path, dict(key, staged_from=LIVE_ORIGIN), hand, phone, image_sha1=sha)
                with open(run_dir / MODES_FILENAME, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(dict(mask_mode(mode["device"]), f=path.name)) + "\n")
            except OSError:
                # Not staged after all (a mask without its mode record would be cached for the
                # worker yet never promoted): the GPU loop computes this frame's OneFormer.
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                continue
            rekeyed += 1
        summary["rekeyed"] = rekeyed
        _write_index(run_dir, summary)
        logger.info("%s %s/%s: %d live OneFormer masks staged under the final key (%d reusable) in "
                    "%.1f s since Stop", LOG, world_id, session_id, rekeyed, len(live), time.time() - t0)

    computed = 0
    for start in range(0, len(order), chunk):
        if should_stop is not None and should_stop():
            return done("stopped", staged=computed)
        frames, records = [], {}
        for ki in order[start:start + chunk]:
            rec = stage_frame(ki)
            if rec is None:
                continue
            records[ki] = rec
            frames.append((ki, rec["kid"]))
        report = T.ensure_transient_masks(
            store, world_id, session_id, frames, intrinsics=intrinsics, camera=camera,
            align_records=records, depth_dir=run_dir, params=params, policy=policy,
            backend_factory=held, should_stop=should_stop, donor=False)
        computed += report.computed
        summary["chunks"] += 1
        summary["staged"] = computed
        summary["gpu_peak_mb"] = max(summary.get("gpu_peak_mb") or 0, report.gpu_peak_mb or 0) or None
        _write_index(run_dir, summary)
        if report.state == T.STATE_STOPPED:
            return done("stopped", staged=computed)
        if report.state != T.STATE_OK:
            return done(report.state, report.detail, staged=computed)
        logger.info("%s %s/%s: chunk %d: %d keyframes staged (%d computed) in %.1f s since Stop",
                    LOG, world_id, session_id, summary["chunks"], len(report.keys), report.computed,
                    time.time() - t0)
    return done("done", staged=computed)


def _same_array(a: Path, b: Path) -> bool:
    try:
        x, y = np.load(a), np.load(b)
    except (OSError, ValueError):
        return False
    return x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)


def promote(store, world_id: str, session_id: str, *, depth_dir, align_records: dict, solution,
            intrinsics, params: T.TransientParams, policy, mode: dict | None = None) -> dict | None:
    """Move every staged mask of this session's newest run whose inputs are the final's own into
    the real cache, then remove the session's runs (what was not promoted is never used).

    None when nothing was staged. Otherwise a record of what happened, also logged:
    `promoted`, `already` (the cache held it under the final key), `refused` by reason. A
    refused mask is not promoted and the transient stage computes it as it always has.
    `mode`: the `execution_mode` of the detectors the calling stage will build -- the surface
    stage passes the device it decided once and builds its detectors on; None reads it now."""
    latest = latest_run(staging_root(store), world_id, session_id)
    try:
        if latest is None:
            return None
        return _promote_run(store, world_id, session_id, *latest, depth_dir=Path(depth_dir),
                            align_records=align_records, solution=solution, intrinsics=intrinsics,
                            params=params, policy=policy, mode=mode)
    finally:
        discard_session(store, world_id, session_id)


def _read_modes(run: Path) -> dict:
    """`{mask file name: {device, fp16}}` from the COMPLETE records of the run's `modes.jsonl`:
    each ends with a newline, so whatever follows the last newline -- even valid JSON, a record
    cut before its terminator -- is not a record, and its mask is not promoted."""
    out = {}
    try:
        text = (run / MODES_FILENAME).read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.split("\n")[:-1]:
        try:
            rec = json.loads(line)
            out[rec.pop("f")] = rec
        except (ValueError, KeyError, AttributeError, TypeError):
            continue
    return out


def _promote_run(store, world_id, session_id, staging: Path, index: dict, *, depth_dir: Path,
                 align_records, solution, intrinsics, params, policy, mode) -> dict:
    from tower.world_builder import appearance as A  # noqa: PLC0415
    from tower.world_builder.dense_pipeline import FILL_RULE  # noqa: PLC0415

    out = {"run": staging.name, "staged": 0, "promoted": 0, "already": 0, "refused": {}}

    def refuse(why):
        out["refused"][why] = out["refused"].get(why, 0) + 1

    files = sorted(staging.glob("*_transient.*.npz"))
    out["staged"] = len(files)
    camera = solution.camera or {}
    H, W = int(camera.get("height", 0)), int(camera.get("width", 0))
    staged_camera = index.get("camera") or {}
    mode = mode if mode is not None else execution_mode()
    # Whole-run gates: each refuses every staged file, under its own reason.
    if policy.raw:
        why = "imagery"                       # a research build never takes a redacted build's mask
    elif index.get("execution") != mode:
        why = "execution-mode"                # device / fp16 / versions are not in the key
    elif index.get("intrinsics") != intrinsics_record(intrinsics):
        why = "calibration"
    elif (staged_camera.get("width"), staged_camera.get("height")) != (W, H):
        why = "camera"
    else:
        why = None
    if why is not None:
        for _ in files:
            refuse(why)
        _log_promote(world_id, session_id, out)
        return out
    undistorter = A.Undistorter(intrinsics, camera)
    evidence = _evidence_keeper(store, world_id, session_id, depth_dir, staging.name, mode)
    modes = _read_modes(staging)
    source_sha1: dict = {}
    for path in files:
        try:
            ki = int(path.name.split("_", 1)[0])
            component = path.name.split(".")[1]
        except (ValueError, IndexError):
            refuse("name")
            continue
        if component not in params.components or not 0 <= ki < len(solution.keyframe_ids):
            refuse("not-this-build")
            continue
        kid = solution.keyframe_ids[ki]
        if ki not in source_sha1:
            src = A.keyframe_source(store, world_id, session_id, kid, ki, policy=policy, align_record=None,
                                    depth_dir=None, undistorter=undistorter, hash_only=True)
            source_sha1[ki] = None if src.refused else src.source_sha1
        if source_sha1[ki] is None:
            refuse("source")
            continue
        key = T.component_key(component, params, keyframe_id=kid, source_sha1=source_sha1[ki],
                              redaction_effective=policy.effective)
        target = T.cache_path(depth_dir, ki, component)
        if T.read_component(target, key, (H, W)) is not None:
            out["already"] += 1
            continue
        got = T.read_component(path, key, (H, W))
        if got is None:
            refuse("key")
            continue
        rec = align_records.get(ki)
        if not (isinstance(rec, dict) and rec.get("fill_rule") == FILL_RULE
                and rec.get("image_sha1") == got[2].get("image_sha1")):
            refuse("image")
            continue
        if not _same_array(staging / f"{ki:05d}_fill.npy", depth_dir / f"{ki:05d}_fill.npy"):
            refuse("fill")
            continue
        if modes.get(path.name) != mask_mode(mode.get("device")):
            refuse("mask-mode")
            continue
        try:
            os.replace(path, target)
        except OSError:
            refuse("write")
            continue
        out["promoted"] += 1
        if evidence is not None:
            evidence(target, ki, kid, component, got[2].get("staged_from") or "worker")
    _log_promote(world_id, session_id, out)
    return out


def _evidence_keeper(store, world_id, session_id, depth_dir: Path, run: str, mode: dict):
    """With `TOWER_WORLD_MASKS_AT_STOP_EVIDENCE` on, a function keeping one promoted mask (and its
    frame's final fill) in `evidence_dir` with a `promoted.jsonl` line; else None. Never raises:
    evidence is for verification and must not cost the build anything but the copies."""
    from tower.config import world_masks_at_stop_evidence_setting  # noqa: PLC0415

    if not world_masks_at_stop_evidence_setting():
        return None
    root = evidence_dir(store, world_id, session_id)
    failed = []

    def keep(target: Path, ki: int, kid: str, component: str, origin: str) -> None:
        try:
            root.mkdir(parents=True, exist_ok=True)
            _keep(target, root / target.name)
            fill = root / f"{ki:05d}_fill.npy"
            if not fill.exists():
                _keep(Path(depth_dir) / fill.name, fill)
            line = json.dumps({"f": target.name, "ki": ki, "kid": kid, "component": component,
                               "from": origin, "run": run, "mode": mode}, sort_keys=True)
            with open(root / EVIDENCE_LISTING, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:  # noqa: BLE001 -- verification aid only
            if not failed:
                failed.append(True)
                logger.exception("%s %s/%s: keeping promotion evidence failed", LOG, world_id, session_id)

    return keep


def _log_promote(world_id, session_id, out) -> None:
    logger.info("%s %s/%s: %d staged masks, %d promoted into the cache, %d already there, refused %s",
                LOG, world_id, session_id, out["staged"], out["promoted"], out["already"],
                json.dumps(out["refused"], sort_keys=True))


def promote_quietly(*args, **kwargs) -> dict | None:
    """`promote`, never raising: a failure here is a stage that computes its masks as today."""
    try:
        return promote(*args, **kwargs)
    except Exception:  # noqa: BLE001 -- an optimisation, never a failure of the stage
        logger.exception("%s promoting the staged masks failed; the stage computes them", LOG)
        return None
