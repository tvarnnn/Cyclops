"""NO BIG BANG for T-UX1: with both switches off, every payload and every builder file is
byte for byte what 5c0dc12 produced.

T-UX1 (`RUN\\lead\\specs\\U06-TUX1-PROGRESS-SPEC-20261004.md` §3, §7.1 T2) adds two
default-off Tower settings:

- `TOWER_WORLD_FINISH_STAGES`: the builder records its finish phase in
  `<world>/solve/<session>/finish_phase.json`, and the status channel and the Saved Worlds
  row project it as `processing`;
- `TOWER_WORLD_PICTURE_BASIS`: the render revision body gains `basis`.

Off -- unset, blank, `0`, `off` or garbage -- nothing is written, nothing is read, and this
file proves it against `golden/world_builder_finish_stages_5c0dc12.json`, recorded by running
this very file on 5c0dc12 (the base, which has neither setting) twice, identical.

WHAT IS COMPARED

1. PAYLOADS. One real world (built by the real engine on synthetic frames), copied into a
   matrix of 6 records (open; finalization `pending`; `complete`/`solved`;
   `complete`/`skipped`; `interrupted`; closed with no finalization) x 3 locks (none; a live
   process; a dead one) x 4 phase files (absent; valid for the live lock; a stale epoch; a
   foreign owner). Per cell: the status payload and its revision, the session's listing row,
   and `GET /worlds/{w}/render/revision`'s body (with and without the appearance viewer).
   Volatile values (wall-clock `*_at` stamps, durations, the live holder's pid and start time,
   revision strings) are replaced by placeholders in order of first sight; nothing else is
   normalised. Within one run, every off spelling must give the RAW, unnormalised bytes of the
   unset run -- so the revision strings are pinned too.
2. BUILDER FILES. `world_build_session.main` with `--solve` and a stub `--solve-script` that
   records its argv and the environment it was given, for a soft Stop (the final solve runs)
   and a hard Stop (it is skipped): every file under `--root` as the SHA-256 of its bytes with
   wall-clock stamps masked, every event, the JSON report, the stub's argv and the keys the
   final spawn's environment adds to the builder's.

TO RECORD: `WB_FINISH_STAGES_RECORD=<path>` writes the observations to <path> instead of
comparing. Record only on 5c0dc12, twice, and keep the golden only if the two are identical.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden" / "world_builder_finish_stages_5c0dc12.json"
RECORD_ENV = "WB_FINISH_STAGES_RECORD"
SWITCHES = ("TOWER_WORLD_FINISH_STAGES", "TOWER_WORLD_PICTURE_BASIS")
OFF_SPELLINGS = (None, "", "0", "off", "garbage")
EPOCH = 1790620836.43
PHASE = "finish_phase.json"

RECORDS = ("open", "pending", "complete-solved", "complete-skipped", "interrupted",
           "closed-no-finalization")
LOCKS = ("none", "live", "dead")
PHASES = ("absent", "valid", "stale-epoch", "foreign-owner")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _set_switches(monkeypatch, value):
    for name in SWITCHES:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def _identity(pid):
    import psutil

    return {"pid": pid, "created_at": float(psutil.Process(pid).create_time())}


@pytest.fixture(scope="module")
def processes():
    """A live process to hold the lock, and the identity of one that has ended."""
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(900)"],
                            stdin=subprocess.DEVNULL)
    dead = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                            stdin=subprocess.DEVNULL)
    try:
        dead_identity = _identity(dead.pid)
        dead.kill()
        dead.wait(timeout=30)
        yield {"live": _identity(live.pid), "dead": dead_identity}
    finally:
        live.kill()
        live.wait(timeout=30)


@pytest.fixture(scope="module")
def base_world(tmp_path_factory):
    """One real world, built once with deterministic ids and a fixed engine clock."""
    import tower.world_builder.engine as engine_module
    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    root = tmp_path_factory.mktemp("finish-stages-base") / "worlds"
    ids = itertools.count(1)
    ticks = itertools.count()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(engine_module, "new_id", lambda: f"{next(ids):032x}")
        width, height = 480, 360
        camera_matrix = ss.camera_matrix(width, height)
        images = ss.render_sequence(ss.furnished_room(), ss.strafe(8, step=0.09),
                                    camera_matrix, width, height)
        intrinsics = CameraIntrinsics(
            source="self_calibrated", model="pinhole",
            fx=float(camera_matrix[0, 0]), fy=float(camera_matrix[1, 1]),
            cx=float(camera_matrix[0, 2]), cy=float(camera_matrix[1, 2]),
            calibrated_width=width, calibrated_height=height)
        engine = WorldBuilderEngine(WorldStore(root), clock=lambda: 1_790_000_000.0 + next(ticks))
        world_id = engine.create_world("Finish Stages Room")
        session_id = engine.start_session(world_id, intrinsics=intrinsics,
                                          frame_source="synthetic", declared_size=(width, height))
        for index, image in enumerate(images):
            engine.observe(ss.encode_jpeg(image), source_seq=index, wire_seq=index)
        engine.stop_session()
        engine.build(world_id, session_id)
    return root, world_id, session_id


def _make_cell(base_world, processes, workdir, record, lock, phase):
    from tower.world_builder.store import WorldStore

    src, world_id, session_id = base_world
    root = workdir / "worlds"
    shutil.copytree(src, root)
    store = WorldStore(root)
    session = store.read_session(world_id, session_id)
    finalization = {
        "open": None, "closed-no-finalization": None,
        "pending": {"state": "pending", "final_solve": "pending"},
        "complete-solved": {"state": "complete", "final_solve": "solved"},
        "complete-skipped": {"state": "complete", "final_solve": "skipped"},
        "interrupted": {"state": "interrupted", "final_solve": "pending"},
    }[record]
    if finalization is not None:
        finalization = dict(finalization, started_at=EPOCH, updated_at=EPOCH + 5.0, detail=None)
    session = dataclasses.replace(
        session, finalization=finalization,
        ended_at=None if record == "open" else session.ended_at)
    store.write_session(session)
    if lock != "none":
        path = store.lock_path(world_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(processes[lock]), encoding="utf-8")
    if phase != "absent":
        owner = processes["live"] if phase != "foreign-owner" else processes["dead"]
        doc = {"format": "wb-finish-phase/1", "session_id": session_id,
               "epoch": EPOCH if phase != "stale-epoch" else EPOCH - 60.0,
               "owner": owner, "writer": owner, "stage": "checking",
               "step": {"n": 2, "of": 3}, "written_at": EPOCH + 100.0}
        target = store.world_dir(world_id) / "solve" / session_id / PHASE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(doc), encoding="utf-8")
    return root, world_id, session_id


def _observe_payloads(root, world_id, session_id):
    """Raw (unnormalised) canonical JSON of everything a client reads about the session."""
    from tower.results.world_builder import WorldBuilderStatusProducer
    from tower.results.world_builder_library import build_world_listing
    from tower.results.world_builder_render import build_render_revision
    from tower.world_builder.store import WorldStore

    store = WorldStore(root)
    producer = WorldBuilderStatusProducer(root, lambda: 1_790_000_500.0)
    snapshot = producer.snapshot(world_id=world_id, session_id=session_id)
    # Twice: the second poll goes through the producer's file cache.
    again = producer.snapshot(world_id=world_id, session_id=session_id)
    listing = build_world_listing(store)
    row = [s for w in listing["worlds"] if w["world_id"] == world_id
           for s in w["sessions"] if s["session_id"] == session_id]
    out = {
        "status": {"payload": snapshot.payload, "revision": snapshot.revision,
                   "volatile": list(snapshot.volatile_fields)},
        "status_again": {"payload": again.payload, "revision": again.revision},
        "row": row,
        "listing_keys": sorted(listing),
    }
    for viewer in (None, "appearance-1"):
        try:
            out[f"revision:{viewer}"] = build_render_revision(store, world_id, session_id,
                                                              viewer=viewer)
        except Exception as exc:  # noqa: BLE001 -- the refusal is part of the payload
            out[f"revision:{viewer}"] = {"raised": type(exc).__name__, "message": str(exc)}
    return json.dumps(out, sort_keys=True, default=str)


class _Norm:
    """Placeholders for volatile values, in order of first sight."""

    def __init__(self, processes):
        self.tokens: dict = {}
        self.pids = {str(p["pid"]) for p in processes.values()}
        self.created = {repr(p["created_at"]) for p in processes.values()}

    def token(self, kind, value):
        key = (kind, json.dumps(value, sort_keys=True))
        if key not in self.tokens:
            self.tokens[key] = f"<{kind}{len(self.tokens)}>"
        return self.tokens[key]

    def value(self, v, key=None):
        if isinstance(key, str) and (key == "at" or key.endswith("_at") or key.endswith("_seconds")
                                     or key in ("seconds", "mapping_seconds")):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return "<time>"
        if key == "revision" and isinstance(v, str):
            return self.token("rev", v)
        if key == "pid" and isinstance(v, int) and str(v) in self.pids:
            return "<pid>"
        if isinstance(v, dict):
            return {k: self.value(x, k) for k, x in sorted(v.items())}
        if isinstance(v, list):
            return [self.value(x) for x in v]
        if isinstance(v, str):
            for pid in self.pids:
                v = re.sub(rf"\b{pid}\b", "<pid>", v)
            return v
        return v


# ---------------------------------------------------------------------------
# 1. payloads
# ---------------------------------------------------------------------------


def test_every_payload_is_5c0dc12s_with_both_switches_off(base_world, processes, tmp_path,
                                                         monkeypatch):
    observed = {}
    norm = _Norm(processes)
    for record, lock, phase in itertools.product(RECORDS, LOCKS, PHASES):
        cell = f"{record}|{lock}|{phase}"
        # A short directory per cell: Windows MAX_PATH (the world tree adds ~150 characters).
        root, world_id, session_id = _make_cell(base_world, processes, tmp_path / f"c{len(observed)}",
                                                record, lock, phase)
        raw = {}
        for spelling in OFF_SPELLINGS:
            _set_switches(monkeypatch, spelling)
            raw[spelling] = _observe_payloads(root, world_id, session_id)
        # Every off spelling is the unset run, byte for byte (revision strings included).
        for spelling in OFF_SPELLINGS[1:]:
            assert raw[spelling] == raw[None], (cell, spelling)
        # Nothing was written by reading.
        assert not any(p.name.startswith("finish_phase") and phase == "absent"
                       for p in root.rglob("finish_phase*")), cell
        full = norm.value(json.loads(raw[None]))
        # Compact, but complete: the SHA-256 of the whole normalised observation (any byte of
        # any field moves it), with the parts T-UX1 is near kept in the clear so a failure
        # says what moved.
        observed[cell] = {
            "sha256": hashlib.sha256(json.dumps(full, sort_keys=True).encode("utf-8")).hexdigest(),
            "lifecycle": full["status"]["payload"].get("lifecycle"),
            "status_revision": full["status"]["revision"],
            "row": [{"state": r.get("state"), "keys": sorted(r)} for r in full["row"]],
            "revision": {k: v for k, v in sorted(full.items()) if k.startswith("revision:")},
        }
    _check("payloads", observed)


# ---------------------------------------------------------------------------
# 2. builder files
# ---------------------------------------------------------------------------

FRAMES = 12
CAPTURE_NAME = "d" * 32
_STAMP = re.compile(rb'"(at|(?!received_at")[A-Za-z0-9_]*_at)"\s*:\s*-?[0-9][0-9.eE+-]*')
_SECONDS = re.compile(rb'"([A-Za-z0-9_]*seconds|[A-Za-z0-9_]*_s|observe_ms_per_frame)"\s*:\s*-?[0-9][0-9.eE+-]*')

STUB = r'''
import json, os, sys
from pathlib import Path
out = Path(os.environ["WB_FS_STUB_OUT"])
with out.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ)}) + "\n")
print(json.dumps({"solved": False, "reason": "stub"}))
'''


@pytest.fixture(scope="module")
def rendered():
    import scripts.world_build_session as builder_script

    frames, intrinsics = builder_script.synthetic_frames(FRAMES, 480, 360)
    return [frame.payload for frame in frames], intrinsics


def _mask(data: bytes, workdir: Path) -> bytes:
    data = _STAMP.sub(lambda m: b'"' + m.group(1) + b'": "<T>"', data)
    data = _SECONDS.sub(lambda m: b'"' + m.group(1) + b'": "<T>"', data)
    for path, label in ((str(workdir), b"<WD>"), (str(workdir.resolve()), b"<WD>"),
                        (str(Path.cwd()), b"<CWD>"), (sys.executable, b"<PY>")):
        for form in {path, path.replace("\\", "/"), json.dumps(path)[1:-1]}:
            data = data.replace(form.encode("utf-8"), label)
    return data


def _run_builder(order, rendered, workdir, monkeypatch, *, solve_every=0, stub_source=STUB,
                 extra_argv=()):
    import scripts.world_build_session as builder_script
    import tower.world_builder.engine as engine_module
    from scripts.world_build_session import StopRequest
    from tower.world_builder.intrinsics_store import IntrinsicsStore

    payloads, intrinsics = rendered
    capture = workdir / "captures" / CAPTURE_NAME
    (capture / "frames").mkdir(parents=True)
    mono0 = time.monotonic() - FRAMES / 12 - 2.0
    rows = []
    for index, payload in enumerate(payloads):
        seq = index + 1
        (capture / "frames" / f"{seq:08d}.jpg").write_bytes(payload)
        rows.append(json.dumps({
            "schema_version": 1, "source_seq": seq, "wire_seq": seq, "tx_seq": seq,
            "received_at": 1000.0 + index / 12, "received_monotonic": mono0 + index / 12,
            "time_basis": "tower-receipt", "relpath": f"frames/{seq:08d}.jpg",
            "byte_count": len(payload), "width": 480, "height": 360}) + "\n")
    (capture / "frames.jsonl").write_text("".join(rows), encoding="utf-8")

    def manifest(end_reason):
        (capture / "capture.json").write_text(json.dumps({
            "schema_version": 1, "capture_id": capture.name, "started_at": 1000.0,
            "ended_at": None if end_reason is None else 2000.0, "end_reason": end_reason,
            "time_basis": "tower-receipt", "frames_written": FRAMES,
            "continues_capture": None}), encoding="utf-8")

    manifest(None)
    holder = {}

    def install(self, **_kwargs):
        holder["stop"] = self

    monkeypatch.setattr(builder_script.StopRequest, "install", install)
    original_observe = builder_script.WorldBuilderEngine.observe
    count = {"n": 0}

    def observe(self, *args, **kwargs):
        outcome = original_observe(self, *args, **kwargs)
        count["n"] += 1
        if count["n"] == FRAMES:
            manifest("stop")
            holder["stop"].request(StopRequest.HARD if order == "hard" else StopRequest.SOFT,
                                   "SIGBREAK" if order == "hard" else "stdin-closed")
        return outcome

    monkeypatch.setattr(builder_script.WorldBuilderEngine, "observe", observe)
    ids = itertools.count(1)
    monkeypatch.setattr(engine_module, "new_id", lambda: f"{next(ids):032x}")
    stub = workdir / "stub_solve.py"
    stub.write_text(stub_source, encoding="utf-8")
    stub_out = workdir / "stub_calls.jsonl"
    monkeypatch.setenv("WB_FS_STUB_OUT", str(stub_out))
    root = workdir / "worlds"
    IntrinsicsStore(root).save(intrinsics)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = builder_script.main([
            "--follow-capture", str(capture), "--root", str(root),
            "--poll-seconds", "0.01", "--max-idle-polls", "50", "--rebuild-every", "4",
            "--solve", "--solve-every", str(solve_every), "--solve-script", str(stub),
            *extra_argv, "--format", "json"])
    parent_env = dict(os.environ)
    calls = []
    if stub_out.exists():
        for line in stub_out.read_text(encoding="utf-8").splitlines():
            call = json.loads(line)
            env = call["env"]
            added = sorted(k for k in env if k not in parent_env)
            changed = sorted(k for k in env if k in parent_env and env[k] != parent_env[k])
            removed = sorted(k for k in parent_env if k not in env)
            calls.append({
                "argv": [_mask(a.encode("utf-8"), workdir).decode("utf-8") for a in call["argv"]],
                "env_added": added, "env_changed": changed, "env_removed": removed})
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = hashlib.sha256(
                _mask(path.read_bytes(), workdir)).hexdigest()
    (events_path,) = list(root.glob("worlds/*/sessions/*/events.jsonl"))
    events = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        event.pop("at", None)
        events.append(event)
    report = json.loads(_mask(stdout.getvalue().encode("utf-8"), workdir).decode("utf-8"))
    return {"exit": exit_code, "report": report, "events": events, "files": files,
            "solve_calls": calls}


@pytest.mark.parametrize("order", ["soft", "hard"])
def test_the_builder_writes_what_5c0dc12_wrote_with_the_switch_off(order, rendered, tmp_path,
                                                                    monkeypatch):
    runs = {}
    for spelling in OFF_SPELLINGS:
        _set_switches(monkeypatch, spelling)
        runs[spelling] = _run_builder(order, rendered, tmp_path / f"run-{spelling!r}"
                                      .replace("'", "").replace('"', "q"), monkeypatch)
    # The workdir name differs per run and is masked; everything else must agree.
    for spelling in OFF_SPELLINGS[1:]:
        assert runs[spelling] == runs[None], spelling
    observed = runs[None]
    assert observed["exit"] == 0
    assert not any(name.endswith(PHASE) for name in observed["files"])
    if order == "soft":
        assert len(observed["solve_calls"]) == 1, "the final solve did not run"
    else:
        assert observed["solve_calls"] == []
    _check(f"builder:{order}", observed)


# ---------------------------------------------------------------------------
# the golden
# ---------------------------------------------------------------------------


def _check(name, observed):
    record_to = os.environ.get(RECORD_ENV)
    if record_to:
        path = Path(record_to)
        golden = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        golden[name] = observed
        path.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
        pytest.skip(f"recorded {name} into {path}")
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))[name]
    if isinstance(expected, dict) and set(expected) == set(observed):
        for key in sorted(expected):
            assert observed[key] == expected[key], f"{name}: {key} differs from 5c0dc12"
    assert observed == expected
