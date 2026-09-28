#!/usr/bin/env python
"""Start a TEST Tower from a named code tree, replay a stored walk into it at
the recorded pace, stop the Tower, and write the report (C22).

"Old vs new" is two runs of this with a different `--code` and switch set:

    python scripts/world_live_replay_run.py --code <tree A> --env-file walk5.env ...
    python scripts/world_live_replay_run.py --code <tree B> --env-file walk5.env --set X=on ...

Run it with the Tower venv's python (the test Tower is started as ONE
interpreter process of that venv, `tower.process_ownership`), and under the
coherence run's GPU lock:

    <venv python> RUN\\lead\\gpulock.py run --who C22 --what "..." -- \\
        <venv python> <this> --code ... --port 8031 --data-root ... --out ...

WHAT IT GUARANTEES
  * Never port 8000, never a port below 8031, never a port that is already
    listening.
  * A FRESH data root: it must not exist or must be empty, and it must not
    be inside the live store. Every root the Tower could write is pointed
    into it (captures, worlds, object memory, document memory, sources), and
    every inherited TOWER_* variable is dropped first, so nothing from the
    shell that runs this reaches the test Tower.
  * The Tower binds 127.0.0.1 only: no phone can reach it by accident.
  * `import tower` in the test Tower resolves to the code tree under test
    (checked before the start, as start-test-tower.ps1 checks it).
  * A NEW or empty --out: a reused one would mix two runs' logs and samples.
  * The Tower and everything it spawns sit in a Job Object this process
    owns with KILL_ON_JOB_CLOSE; when the kernel refuses one, the Tower is
    stopped and nothing is replayed. It is stopped at the end on every path
    (settled, timeout, abort, error, Ctrl+C), and if this process dies the
    kernel kills the whole tree. Afterwards it checks that the port is free
    and that no process of the tree is left.
  * Once the Tower answers, the process listening on --port must be the
    Tower this runner started (psutil); anything else is refused.
  * :8000 is only ever asked for /health, and the guard FAILS CLOSED
    (`world_live_replay.live_tower_state`). Nothing starts while :8000 is
    recording, finishing a world, or not answering usably. From the start
    of the test Tower to its stop -- its startup, the idle wait, the stream
    and the settle -- a live walk (or two unusable answers in a row) kills
    the test Tower FIRST, so the GPU is free for the real walk.
  * run.json pins the harness itself (git HEAD, its uncommitted changes, and
    a sha1 of the three scripts) and the thread-count variables inherited
    from the shell.

The calibration (`intrinsics/*.json`) is copied into the fresh world root from
`--intrinsics-from` (read only): without a calibration at the observed
resolution the builder downgrades to unposed and reconstructs nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.world_live_replay import (  # noqa: E402
    EXIT_ABORTED,
    EXIT_ERROR,
    LIVE_HEALTH_URL,
    LIVE_REFUSE_AT_START,
    LiveGuard,
    add_replay_arguments,
    check_target_port,
    exit_code,
    harness_identity,
    http_json,
    listener_pids,
    live_tower_state,
    options_from_args,
    refuse_non_empty_out,
    run_replay,
)
from scripts.world_live_replay_report import build_report, write_report  # noqa: E402
from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.process_ownership import (  # noqa: E402
    assign_to_job,
    interpreter_command,
    interpreter_environment,
    terminate_tree,
)

LIVE_DATA = Path(r"C:\Users\tvllo\Projects\Glasses\tower\data")
DEFAULT_INTRINSICS = LIVE_DATA / "world_builder" / "intrinsics"

# Roots the runner owns. Anything a switch file says about them is replaced.
FORCED_KEYS = ("TOWER_CAPTURE_ROOT", "TOWER_WORLD_ROOT", "TOWER_OBSERVATION_ROOT",
               "TOWER_DOCUMENT_ROOT", "TOWER_SOURCES_ROOT", "TOWER_HOST", "TOWER_PORT")

# Inherited from the shell and not scrubbed, but they change timing, so
# run.json records them (review C22 L6).
TIMING_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")

HEALTH_POLL_S = 2.0
IDLE_POLL_S = 3.0
IDLE_WAIT_S = 120.0
# :8000 during the runner's own waits (startup and idle), as often as the
# client watches it during the stream.
LIVE_WATCH_EVERY_S = 10.0
AFTER_STOP_S = 2.0


class LiveTowerAbort(Exception):
    """:8000 said stop while the runner itself was waiting (startup, idle)."""


def resolve_code_tree(code: Path) -> Path:
    """The `tower/` directory of a repository root, a worktree, a `git
    archive` directory, or that `tower/` directory itself."""
    code = Path(code).resolve()
    for candidate in (code / "tower", code):
        if (candidate / "tower" / "main.py").is_file():
            return candidate
    raise SystemExit(f"no tower/tower/main.py under {code}")


def code_identity(tower_dir: Path) -> dict:
    """Which code this is: git HEAD when it is a checkout (read-only git), and
    always a fingerprint of the .py sources, so two archives can be told apart."""
    identity = {"tower_dir": str(tower_dir)}
    # `--no-optional-locks` on every call (review C22 L3): a plain `git
    # status` may refresh another lane's index and take its index.lock.
    git = ["git", "--no-optional-locks", "-C", str(tower_dir)]
    try:
        top = subprocess.run([*git, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=20)
        # Only this tree's own checkout: an archive directory sitting inside
        # some other repository must not report that repository's HEAD.
        own = top.returncode == 0 and Path(top.stdout.strip()).resolve() in (
            tower_dir.resolve(), tower_dir.parent.resolve())
        head = subprocess.run([*git, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20) if own else None
        if head is not None and head.returncode == 0:
            identity["git_head"] = head.stdout.strip()
            dirty = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, timeout=60)
            identity["git_dirty"] = [line for line in dirty.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    digest = hashlib.sha1()
    count = 0
    for sub in ("tower", "scripts"):
        for path in sorted((tower_dir / sub).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(path.relative_to(tower_dir).as_posix().encode())
            digest.update(path.read_bytes())
            count += 1
    identity["py_fingerprint"] = digest.hexdigest()[:16]
    identity["py_files"] = count
    identity["has_stage_timing"] = (tower_dir / "tower" / "world_builder" / "stage_timing.py").is_file()
    identity["has_serve_loop"] = (tower_dir / "tower" / "serve_loop.py").is_file()
    return identity


def read_switch_file(path: Path) -> dict:
    """KEY=VALUE lines, `#` comments, later keys win (python-dotenv's rule)."""
    values = {}
    text = Path(path).read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _inside(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def tower_environment(base: dict, switches: dict, *, data_root: Path, tower_dir: Path, port: int) -> dict:
    """The test Tower's environment: the parent's, minus every TOWER_*, plus
    the switches, plus the roots this runner owns."""
    env = {k: v for k, v in base.items() if not k.upper().startswith("TOWER_")
           and k.upper() not in ("PYTHONPATH", "PYTHONHOME")}
    for key, value in switches.items():
        if key in FORCED_KEYS:
            continue
        env[key] = value
    env.update({
        "TOWER_CAPTURE_ROOT": str(data_root),
        "TOWER_WORLD_ROOT": str(data_root / "world_builder"),
        "TOWER_OBSERVATION_ROOT": str(data_root / "object_memory"),
        "TOWER_DOCUMENT_ROOT": str(data_root / "document_memory"),
        "TOWER_SOURCES_ROOT": str(data_root),
        "TOWER_HOST": "127.0.0.1",
        "TOWER_PORT": str(port),
        "PYTHONPATH": str(tower_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _log(out: Path, text: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [runner] {text}"
    print(line, flush=True)
    with open(out / "runner.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _leftovers(port: int, markers) -> list:
    """Processes that still look like this run's Tower or its children."""
    try:
        import psutil
    except ImportError:
        return []
    found = []
    mine = {os.getpid()}
    try:
        mine.update(parent.pid for parent in psutil.Process().parents())
    except Exception:  # noqa: BLE001
        pass
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            argv = " ".join(proc.info.get("cmdline") or [])
        except Exception:  # noqa: BLE001
            continue
        if not argv:
            continue
        if (f"--port {port}" in argv and "tower.main:app" in argv) or any(m in argv for m in markers):
            if proc.pid not in mine:
                found.append({"pid": proc.pid, "cmdline": argv[:300]})
    return found




def spawn_tower(command, **kwargs):
    """`subprocess.Popen`, behind a name the tests can replace."""
    return subprocess.Popen(command, **kwargs)


def read_client_record(out: Path) -> dict:
    """The client's `client.json` in `out`, or {} (a fresh --out holds only
    this run's)."""
    try:
        record = json.loads((Path(out) / "client.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def probe_import(tower_dir: Path, env: dict) -> str:
    """Which `tower` the server will import: the same env, cwd and interpreter."""
    probe = subprocess.run(interpreter_command("-c", "import tower, sys; print(tower.__file__)"),
                           cwd=str(tower_dir), env=env, capture_output=True, text=True, timeout=120)
    return (probe.stdout.strip().splitlines() or [""])[-1]


class StartupLiveWatch:
    """:8000 while the runner waits for its own Tower (review C22 M3): the
    startup can take 240 s and the idle wait 120 s, and a live walk that
    begins then must stop the test Tower just as it would mid-stream."""

    def __init__(self, guard: LiveGuard, every: float | None = None, clock=time.monotonic):
        self.guard = guard
        self.every = LIVE_WATCH_EVERY_S if every is None else every
        self._clock = clock
        self._next = clock() + self.every

    def check(self) -> None:
        now = self._clock()
        if now < self._next:
            return
        self._next = now + self.every
        reason = self.guard.observe(live_tower_state(LIVE_HEALTH_URL))
        if reason:
            raise LiveTowerAbort(reason)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_replay_arguments(parser)
    parser.add_argument("--code", type=Path, required=True,
                        help="The code tree to run the test Tower from (repo root, worktree, "
                             "git-archive directory, or its tower/).")
    parser.add_argument("--env-file", type=Path, action="append", default=[],
                        help="Switch file(s), KEY=VALUE; later files and --set win. Roots are ignored.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--data-root", type=artifact_root_arg, required=True,
                        help="A FRESH root for the test Tower's captures and worlds (keep it short: "
                             "Windows MAX_PATH).")
    parser.add_argument("--out", type=artifact_root_arg, required=True,
                        help="A NEW or empty directory: logs, client.json, samples.csv, report.json "
                             "and REPORT.md.")
    parser.add_argument("--intrinsics-from", type=Path, default=DEFAULT_INTRINSICS,
                        help="Calibrations copied into the fresh world root (read only).")
    parser.add_argument("--health-timeout", type=float, default=240.0)
    args = parser.parse_args(argv)

    check_target_port(args.port)
    out = Path(args.out)
    data_root = Path(args.data_root)
    for path, what in ((data_root, "--data-root"), (out, "--out")):
        if _inside(path, LIVE_DATA):
            raise SystemExit(f"refused: {what} {path} is inside the live store {LIVE_DATA}")
    if data_root.exists() and any(data_root.iterdir()):
        raise SystemExit(f"refused: --data-root {data_root} is not empty; a replay needs a fresh root")
    refuse_non_empty_out(out)
    if _inside(out, data_root) or _inside(data_root, out):
        raise SystemExit("refused: --out and --data-root must be separate directories")

    tower_dir = resolve_code_tree(args.code)
    identity = code_identity(tower_dir)
    switches: dict = {}
    for path in args.env_file:
        switches.update(read_switch_file(path))
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set wants KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        switches[key.strip()] = value.strip()
    ignored = sorted(k for k in switches if k in FORCED_KEYS)
    if switches.get("TOWER_WORLD_STAGE_TIMING", "").lower() in ("on", "true", "1", "yes") \
            and not identity["has_stage_timing"]:
        identity["stage_timing_note"] = ("TOWER_WORLD_STAGE_TIMING is set but this code has no "
                                         "tower/world_builder/stage_timing.py: no stage_timing.json will exist")

    # The guard fails closed (review C22 M1, M2): recording, finishing a
    # world, or no usable answer all mean "do not start".
    live = live_tower_state(LIVE_HEALTH_URL)
    if live in LIVE_REFUSE_AT_START:
        print(f"REFUSED: :8000 is {live} (the guard refuses recording, busy and unknown). "
              "Nothing was started.", flush=True)
        return EXIT_ABORTED
    if port_in_use(args.port):
        raise SystemExit(f"refused: something already listens on 127.0.0.1:{args.port}")

    if not (Path(sys.prefix) / "pyvenv.cfg").is_file():
        print("warning: this runner is not running from a venv; the test Tower gets this interpreter",
              flush=True)

    out.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    world_root = data_root / "world_builder"
    intrinsics = world_root / "intrinsics"
    intrinsics.mkdir(parents=True, exist_ok=True)
    copied = []
    if Path(args.intrinsics_from).is_dir():
        for calibration in sorted(Path(args.intrinsics_from).glob("*.json")):
            shutil.copyfile(calibration, intrinsics / calibration.name)
            copied.append(calibration.name)

    env = interpreter_environment(tower_environment(os.environ, switches, data_root=data_root,
                                                    tower_dir=tower_dir, port=args.port))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    err_log = out / f"tower-{args.port}-{stamp}.err.log"
    out_log = out / f"tower-{args.port}-{stamp}.out.log"
    uvicorn = ["-m", "uvicorn", "tower.main:app", "--host", "127.0.0.1", "--port", str(args.port),
               "--timeout-graceful-shutdown", "10"]
    if identity["has_serve_loop"]:
        uvicorn += ["--loop", "tower.serve_loop:resilient_loop_factory"]
    command = list(interpreter_command(*uvicorn))

    resolved = probe_import(tower_dir, env)
    if not resolved or not _inside(Path(resolved), tower_dir):
        raise SystemExit(f"refused: import tower resolves to {resolved!r}, not under {tower_dir}")

    run = {
        "tool": "world_live_replay_run",
        "label": args.label,
        "started_at": round(time.time(), 3),
        "harness": harness_identity(),
        "code": identity,
        "import_tower": resolved,
        "switches": switches,
        "switches_ignored_roots": ignored,
        "effective_tower_env": {k: v for k, v in sorted(env.items()) if k.startswith("TOWER_")
                                or k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "__PYVENV_LAUNCHER__")},
        "timing_env": {k: env.get(k) for k in TIMING_ENV_KEYS},
        "port": args.port,
        "data_root": str(data_root),
        "intrinsics_copied": copied,
        "intrinsics_from": str(args.intrinsics_from),
        "command": command,
        "cwd": str(tower_dir),
        "err_log": str(err_log),
        "out_log": str(out_log),
        "live_tower_at_start": live,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    _log(out, f"harness {run['harness'].get('git_head')} sha1 {run['harness'].get('sha1')} "
              f"dirty {run['harness'].get('git_dirty')}")
    _log(out, f"code {identity}; switches {switches}")
    _log(out, f"starting the test Tower: {' '.join(command)} (cwd {tower_dir})")

    record: dict = {}
    code = EXIT_ERROR
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    process = job = None
    watch = StartupLiveWatch(LiveGuard(live))
    health_url = f"http://127.0.0.1:{args.port}/health"
    try:
        # Spawn and job assignment INSIDE the try (review C22 L2): whatever
        # happens from here on, the finally stops what was started.
        with open(err_log, "ab") as err_handle, open(out_log, "ab") as out_handle:
            process = spawn_tower(command, cwd=str(tower_dir), env=env, stdin=subprocess.DEVNULL,
                                  stdout=out_handle, stderr=err_handle, creationflags=creationflags)
        job = assign_to_job(process)
        run["tower_pid"] = process.pid
        run["job"] = job is not None
        if job is None and os.name == "nt":
            run["error"] = "no Job Object"
            raise SystemExit("refused: the test Tower could not be put in a Job Object with "
                             "KILL_ON_JOB_CLOSE, so a dead runner could leave it running; it was "
                             "stopped and nothing was replayed")
        deadline = time.time() + args.health_timeout
        health = None
        while time.time() < deadline:
            if process.poll() is not None:
                raise SystemExit(f"the test Tower exited with {process.returncode} before answering; "
                                 f"read {err_log}")
            watch.check()
            status, health = http_json("GET", health_url, 5.0)
            if status == 200:
                break
            time.sleep(HEALTH_POLL_S)
        else:
            raise SystemExit(f"the test Tower did not answer /health within {args.health_timeout} s")
        # Is the process answering on the port the one just started?
        # (review C22 M4) Another process could have won a bind race.
        owners = listener_pids(args.port)
        run["listener_pids"] = None if owners is None else sorted(owners)
        if owners != {process.pid}:
            run["error"] = "foreign listener"
            raise SystemExit(f"refused: :{args.port} is served by {owners}, not the test Tower {process.pid}")
        _log(out, f"test Tower up: pid {process.pid}, job {job is not None}, listener {sorted(owners)}")
        # The startup chore on an empty root finds nothing; let it end so
        # the replay starts from an idle Tower.
        idle_deadline = time.time() + IDLE_WAIT_S
        while time.time() < idle_deadline:
            chore = (health or {}).get("background_chore") or {}
            if chore.get("state") not in ("running",):
                break
            watch.check()
            time.sleep(IDLE_POLL_S)
            _, health = http_json("GET", health_url, 5.0)

        def kill_tower_now() -> None:
            # M3: the guard calls this the moment it aborts, before the
            # client's teardown. The finally below stops it again; that is
            # idempotent.
            _log(out, f"ABORT: stopping the test Tower pid {process.pid} now")
            terminate_tree(process, job=job, timeout=30.0, hard=True)

        options = options_from_args(args, port=args.port, out=out, world_root=world_root,
                                    tower_pid=process.pid, on_abort=kill_tower_now)
        record = asyncio.run(run_replay(options))
        code = exit_code(record)
    except LiveTowerAbort as exc:
        _log(out, f"ABORT before the replay: {exc}; stopping the test Tower")
        run["aborted"] = {"t": round(time.time(), 3), "reason": str(exc), "during": "startup"}
        code = EXIT_ABORTED
    except KeyboardInterrupt:
        _log(out, "interrupted; stopping the test Tower")
        code = EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 -- the Tower is stopped and a report still written
        import traceback

        _log(out, f"the replay failed: {exc!r}\n{traceback.format_exc()}")
        run["error"] = repr(exc)
        code = EXIT_ERROR
        # THE ABORT EDGE (review C22 round 2, "Still open" 5). If the client
        # raised anyway, its own record is on disk (`run_replay` writes it in
        # its `finally`): report from it, and when its OUTCOME is `aborted`,
        # the exception was the kill's consequence, so exit 3 as every other
        # abort does.
        #
        # The outcome, not the `aborted` block (review C22 round 3 L-b): the
        # guard can still abort DURING `run_replay`'s `finally`, after a
        # genuine fault was re-raised. That record has an `aborted` block and
        # no `aborted` outcome, and the fault is what ended the run: exit 1,
        # with the late abort kept beside it.
        saved = read_client_record(out)
        if saved:
            record = saved
            if saved.get("outcome") == "aborted":
                run["aborted"] = {**(saved.get("aborted") or {}), "during": "stream",
                                  "raised": repr(exc)}
                code = EXIT_ABORTED
            elif saved.get("aborted"):
                run["guard_aborted_after_the_fault"] = {**saved["aborted"], "raised": repr(exc)}
    finally:
        run["live_tower_watch_startup"] = watch.guard.summary()
        if process is not None:
            _log(out, f"stopping the test Tower pid {process.pid}")
            terminate_tree(process, job=job, timeout=30.0, hard=True)
        if job is not None:
            job.close()
        time.sleep(AFTER_STOP_S)
        left = _leftovers(args.port, [str(world_root)])
        run["stopped_at"] = round(time.time(), 3)
        run["port_free_after"] = not port_in_use(args.port)
        run["leftover_processes"] = left
        if left:
            _log(out, f"WARNING: processes left after the stop: {left}")
        _log(out, f"test Tower stopped; port {args.port} free: {run['port_free_after']}")
        (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    captures = record.get("tower_captures") or []
    report = build_report(tower_log=err_log, tower_out_log=out_log, world_root=world_root,
                          capture_id=captures[0] if captures else None, client=record,
                          samples=out / "samples.csv" if (out / "samples.csv").exists() else None,
                          label=args.label, run_dir=out, data_root=data_root,
                          capture_root=args.capture_root, run=run,
                          capture_root_from="the runner's --capture-root (the replay streamed from it)")
    report["run"] = run
    json_path, md_path = write_report(out, report)
    _log(out, f"report: {md_path}; verdict {report['verdict']}; outcome {record.get('outcome')}; "
              f"replay fidelity {report['replay_fidelity']['result']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
