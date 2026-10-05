"""NO BIG BANG for W0-1F (concurrent sparse draws): with the switch OFF, a final solve builds
byte for byte what 44fbd13 built.

WHY THIS FILE EXISTS. W0-1F (`claude/w01f-fix`) adds the concurrent consensus-draw mapper
behind `TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT` (default `off`) and a per-session writer lock
that every final solve and re-gate takes whatever the switch says. The manager let the lock ship
without a switch on condition that OFF identity with 44fbd13 -- the base before W0-1F -- is
PINNED by a test that stays in the suite (ruling 179, condition 4; review
claude-W01F-4bdbeac-REVIEW-ADV-20261005 MED-1). This is that test.

WHAT IS RUN. The deterministic consensus-3 final solve of the reproducible-finish `walk` (the
mapper, the gate's links and the depth network faked as pure functions; a stub transient
detector; no GPU, no pycolmap), three times over, with the switch unset:

- `off`: nothing stops it;
- `off-stop-after-1`: a stop is asked once draw 1 has been mapped;
- `off-after-a-failed-on-run[...]`: first, in this same interpreter, an ON run of a COPY of the
  store fails (a draw child that could not be confirmed stopped, alive or since dead; or a
  journal that cannot be written); then the OFF run of the original store must still be `off`'s
  golden (review W01F-1d9931a MED-1 / Codex M4). At 44fbd13 the switch does not exist, so there
  the "ON" run is a plain run of the copy, and the OFF run is `off` all the same.

WHAT IS COMPARED, against `golden/world_builder_concurrent_off_44fbd13.json` (recorded by
running this very file on 44fbd13): every file under the store root -- the world, its solve
workspace, solution, components, consensus, arrays, stage timing, journals -- as the SHA-256 of
its bytes; for `.json`/`.jsonl` files after masking epoch timestamps, the I0 `finish_id` and
this test's own temporary directory. The one file allowed beyond the golden is the session
writer lock's own, `.session-locks/<32 hex>.lock`, which must be EMPTY (ruling 179; the
review's LOW-3, accepted as is).

TO RECORD THE GOLDEN AGAIN -- only if a plain final solve's output legitimately changes on the
base branch -- run this file on that base with `WB_CONCURRENT_OFF_RECORD=<new golden path>`,
review the diff, and rename the golden after the base SHA. Never record it from a tree whose
concurrent draws or session lock are under review.
"""

import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from tests.test_world_builder_reproducible_finish import engines, walk  # noqa: F401
from tests.test_world_builder_solve_masks import StubDetector, colmap  # noqa: F401
from tower.world_builder import global_solve as GS
from tower.world_builder import solve_masks as SM

GOLDEN = Path(__file__).parent / "golden" / "world_builder_concurrent_off_44fbd13.json"
RECORD_ENV = "WB_CONCURRENT_OFF_RECORD"
SWITCH = "TOWER_WORLD_SOLVE_CONSENSUS_CONCURRENT"
LOCKS_DIRNAME = ".session-locks"
_LOCK_FILE = re.compile(r"\.session-locks/[0-9a-f]{32}\.lock")
_EPOCH = re.compile(r"1[67]\d{8}(\.\d+)?")
_FINISH = re.compile(r'"finish_id": "[^"]*"')
GIB = 1024 ** 3
_REAL_GETPID = os.getpid


class _Ids:
    """`uuid.uuid4` and `os.getpid`, deterministic. The counter is swapped out for the failed ON
    run and back for the OFF run, so the OFF run draws exactly the ids `off`'s run draws."""

    def __init__(self):
        self.counter = itertools.count(1)

    def uuid4(self):
        return uuid.UUID(int=next(self.counter))


@pytest.fixture(autouse=True)
def ids(monkeypatch):
    made = _Ids()
    monkeypatch.setattr(uuid, "uuid4", made.uuid4)
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    monkeypatch.delenv(SWITCH, raising=False)
    return made


def _seeded(monkeypatch):
    """Draw 1 posts fewer observations than draw 0, so the consensus has something to decide."""
    base_map = GS._map_candidate

    def seeded_map(pycolmap, database_path, workspace, sparse_dir, keyframes, *, seed, **kw):
        sol = base_map(pycolmap, database_path, workspace, sparse_dir, keyframes, seed=seed, **kw)
        if seed == 1:
            for kid in sorted(sol.poses)[5:30]:
                sol.poses[kid] = dict(sol.poses[kid], observations=0)
        return sol

    monkeypatch.setattr(GS, "_map_candidate", seeded_map)


def _off_runs(walk, monkeypatch, stop_after=None):
    """The OFF final solve, three times over, as the NO-BIG-BANG harness ran it."""
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.setenv("TOWER_WORLD_STAGE_TIMING", "on")
    monkeypatch.setattr(time, "perf_counter", lambda: 100.0)
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)
    monkeypatch.setattr(SM, "new_masked_database_path",
                        lambda workspace: workspace.root / "database.masked.pX.fixed.db")
    _seeded(monkeypatch)
    mapped = []
    stop = None
    if stop_after is not None:
        real = GS._map_candidate

        def counting(*args, **kwargs):
            mapped.append(kwargs.get("seed"))
            return real(*args, **kwargs)

        monkeypatch.setattr(GS, "_map_candidate", counting)
        stop = lambda: len(mapped) >= 1 + stop_after  # noqa: E731
    for _ in range(3):
        mapped.clear()
        GS.solve(walk.store, walk.world_id, walk.session_id, final=True, masks=True, seed=0,
                 gate=True, consensus=3, transient_backend_factory=StubDetector(),
                 mask_device_probe=lambda: None, input_digest="walk-digest", should_stop=stop)


def _mask(text, tmp_path):
    for path in {str(tmp_path), str(Path(tmp_path).resolve())}:
        for form in {path, path.replace("\\", "/"), json.dumps(path)[1:-1]}:
            text = text.replace(form, "<TMP>")
    return _FINISH.sub('"finish_id": "F"', _EPOCH.sub("T", text))


def _snapshot(store_root, tmp_path):
    """Every file under the store root, hashed (JSON masked), and the lock files set apart."""
    files, locks = {}, {}
    for path in sorted(Path(store_root).rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(store_root).as_posix()
        if name.startswith(LOCKS_DIRNAME + "/"):
            locks[name] = path.stat().st_size
            continue
        if path.suffix in (".json", ".jsonl"):
            data = _mask(path.read_text(encoding="utf-8"), tmp_path).encode("utf-8")
        else:
            data = path.read_bytes()
        files[name] = hashlib.sha256(data).hexdigest()
    return files, locks


def _check(run, store_root, tmp_path, *, record=True):
    files, locks = _snapshot(store_root, tmp_path)
    # The only file beside the golden's: the session writer lock's own, empty, at most one (one
    # session was solved in this store).
    assert all(_LOCK_FILE.fullmatch(name) for name in locks), locks
    assert all(size == 0 for size in locks.values()), locks
    assert len(locks) <= 1, locks
    assert files, "the run wrote nothing"
    record_to = os.environ.get(RECORD_ENV)
    if record_to:
        if not record:
            pytest.skip(f"{run}: compared with `off`, nothing to record")
        path = Path(record_to)
        golden = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
                  else {"base": "44fbd13", "runs": {}})
        golden["runs"][run] = files
        path.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8",
                        newline="\n")
        pytest.skip(f"recorded {run} into {path}")
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))["runs"][run]
    # Most specific first, so a failure names what differs.
    assert sorted(files) == sorted(expected)
    assert {name: digest for name, digest in files.items() if expected[name] != digest} == {}


@pytest.mark.parametrize("stop_after", [None, 1], ids=["off", "off-stop-after-1"])
def test_a_final_solve_with_the_switch_off_builds_what_44fbd13_built(
        walk, engines, colmap, monkeypatch, tmp_path, stop_after):
    _off_runs(walk, monkeypatch, stop_after)
    _check("off" if stop_after is None else "off-stop-after-1", walk.store.root, tmp_path)


def _failing_on_run(store, walk, mode, monkeypatch, sleepers):
    """An ON final solve of `store` (a copy of the walk's) that fails. Returns what it raised."""
    from tower import process_ownership

    monkeypatch.setenv(SWITCH, "after-draw-0")
    _seeded(monkeypatch)
    for name, value in (("_draw_free_ram", lambda: 64 * GIB), ("_draw_free_commit", lambda: 64 * GIB),
                        ("_DRAW_WAIT_FLOOR_S", 1.0), ("_DRAW_WAIT_PER_IMAGE_S", 1e-6)):
        monkeypatch.setattr(GS, name, value, raising=False)

    class Exited:
        pid = 0

        def wait(self, timeout=None):
            return 1

        def poll(self):
            return 1

    class Job:
        def close(self):
            pass

    def launch(context, database, seed, sparse, output, expected):
        if mode.startswith("unconfirmed"):
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                                       stdin=subprocess.DEVNULL)
            sleepers.append(process)
            return process, process_ownership.assign_to_job(process)
        return Exited(), Job()          # a child that exits 1: re-mapped, not a failure

    monkeypatch.setattr(GS, "_launch_draw_child", launch, raising=False)
    if mode.startswith("unconfirmed"):
        monkeypatch.setattr(process_ownership, "terminate_tree", lambda *a, **k: False)
    if mode == "journal":
        def full(*args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(GS, "append_jsonl", full, raising=False)
    try:
        GS.solve(store, walk.world_id, walk.session_id, final=True, masks=True, seed=0,
                 gate=True, consensus=3, transient_backend_factory=StubDetector(),
                 mask_device_probe=lambda: None, input_digest="walk-digest")
    except Exception as exc:  # noqa: BLE001
        return exc
    return None


@pytest.mark.parametrize("mode", ["unconfirmed-alive", "unconfirmed-dead", "journal"])
def test_off_after_a_failed_on_run_still_builds_what_44fbd13_built(
        walk, engines, colmap, monkeypatch, tmp_path, ids, mode):
    from tower.world_builder.store import WorldStore

    concurrent = hasattr(GS, "concurrent_draw_mapper")      # False at 44fbd13: no switch there
    unconfirmed = getattr(GS, "_UNCONFIRMED_DRAW_CHILDREN", [])
    before = len(unconfirmed)
    sleepers = []
    copy_root = tmp_path / "on-copy"
    shutil.copytree(walk.store.root, copy_root)
    try:
        with monkeypatch.context() as on:
            on.setattr(uuid, "uuid4", _Ids().uuid4)          # the OFF run's ids are untouched
            on.setattr(os, "getpid", _REAL_GETPID)            # its draw children are real processes
            raised = _failing_on_run(WorldStore(copy_root), walk, mode, on, sleepers)
        if concurrent:
            # The ON run really failed: it raised, or left a draw child it could not confirm.
            if mode.startswith("unconfirmed"):
                assert len(unconfirmed) > before and sleepers, (raised, unconfirmed)
            else:
                assert raised is not None
        if mode == "unconfirmed-dead":
            for process in sleepers:
                process.kill()
                process.wait(10)
        _off_runs(walk, monkeypatch)
        # Compared with `off` itself: after a failed ON run, OFF is still exactly the base.
        _check("off", walk.store.root, tmp_path, record=False)
    finally:
        for process in sleepers:
            if process.poll() is None:
                process.kill()
            process.wait(10)
        del unconfirmed[before:]
