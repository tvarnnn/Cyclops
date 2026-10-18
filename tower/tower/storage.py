"""Filesystem primitives shared by every module that persists to disk.

Object memory, World Builder and the capture recorder (shared transport
infrastructure, so it must not import from a cartridge) all need the same
operations, so they live in one place.

The atomic-write pattern is deliberate: the try/finally makes a failed write
leave no temp file behind.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import BinaryIO, Callable

logger = logging.getLogger(__name__)

TEMP_SUFFIX = ".tmp"


def staging_path(path: Path) -> Path:
    """A staging name no other writer can be using.

    Public, because two modules need the one convention: this file's two
    atomic writers and `world_builder/store.py`'s keyframe images.

    A shared `path.name + ".tmp"` lets two writers of one destination publish
    each other's half-written temp: with two writers it measured 656 torn
    reads at the published path over 12 seconds, plus a writer killed by
    `FileNotFoundError` out of `replace` because its peer's `finally` had
    already unlinked the shared temp. The torn read is the `BadZipFile` the
    atomic write exists to prevent.

    Nothing in this Tower serialises writers of `solution.npz`:
    `acquire_writer_lock` is per-world and taken only by the engine, and
    `scripts/world_solve.py` takes no lock at all, so a hand-run solve
    against a world with a live builder is two writers.

    pid plus uuid4: the pid makes a stray temp attributable, and the uuid
    makes it unique even within one process.
    """
    # `p` before the pid. The sweeper has to find the pid again, and "the
    # all-digit component" is not a safe rule: `uuid4().hex[:8]` is all
    # decimal digits 2.33% of the time (measured over 200k samples), and the
    # capture recorder names its own staging files `<source_seq>.jpg.tmp`, so
    # a frame number would be read as a pid. A `p`-prefixed component cannot
    # be produced by either.
    return path.with_name(
        f"{path.name}.p{os.getpid()}.{uuid.uuid4().hex[:8]}{TEMP_SUFFIX}"
    )


def new_id() -> str:
    """Mint an opaque identifier.

    uuid4 hex, never derived from a display name (users rename things) and
    never a content hash (refinement changes content). Anything that
    references an id must survive both.
    """
    return uuid.uuid4().hex


# How long a writer keeps trying to replace a destination a reader has open,
# and how it paces the attempts. Bounded: this rides out a reader, not a
# process parked on the file, and a writer that cannot win in the budget
# raises.
#
# The budget is 2 s because a reader can lose the CPU mid-read under heavy
# background load: with the global solver's background child taking 18 of 20
# cores, the web thread reading a 2.2 MB points.json held the file past
# 60 ms (the budget was 12 x 5 ms = 60 ms before it was raised). Two seconds
# is longer than any reader this Tower has that is still making progress,
# and short next to a rebuild cadence of one per second.
REPLACE_BUDGET_S = 2.0
REPLACE_BACKOFF_S = 0.005
REPLACE_BACKOFF_MAX_S = 0.05


def sweep_abandoned_staging(directory: Path) -> int:
    """Remove staging files whose writer is gone. Returns how many.

    Unique staging names (see `staging_path`) mean a killed writer leaves a
    stray temp every time, because the `finally` does not run under
    `TerminateProcess` (six hard kills measured 21 strays and 11.4 MB). The
    builder kills a solve child on every stop that outstays its budget.

    The pid in the name is what makes this safe: a staging file belonging
    to a LIVE process is someone's write in flight and is left alone. Only
    a dead writer's leavings are swept, so this can run at any time.
    """
    swept = 0
    for candidate in directory.iterdir():
        if not candidate.is_file():
            continue
        parts = candidate.name.split(".")
        # `.tmp` as a component, not as a suffix. `staging_path` puts it
        # last for a JSON or npz write, but `prepare_images` stages an
        # undistorted frame as `<stem>.<pid>.<uuid>.tmp.jpg`, which a suffix
        # match would miss.
        if TEMP_SUFFIX.lstrip(".") not in parts:
            continue
        # Exactly the `p<digits>` component `staging_path` writes, and nothing
        # else. Reading "the last all-digit component" as a pid deletes a
        # live writer's file whenever its uuid is all digits, a user's
        # `2024.tmp`, and the capture recorder's `<source_seq>.jpg.tmp` (the
        # frame number read as a process). A deletion primitive does not
        # guess.
        pid = None
        for part in parts:
            if len(part) > 1 and part[0] == "p" and part[1:].isdigit():
                pid = int(part[1:])
                break
        if pid is None:
            # Not a name this module wrote. Not ours to judge, and deleting
            # an unattributable file is not a sweep.
            continue
        try:
            import psutil

            if psutil.pid_exists(pid):
                continue
        except Exception:  # noqa: BLE001 -- absence of psutil is not a reason to delete
            continue
        try:
            candidate.unlink()
        except OSError:
            continue
        swept += 1
    if swept:
        logger.info(
            "storage: swept %s abandoned staging file(s) from %s", swept, directory
        )
    return swept


def write_json_atomic(path: Path, payload: dict) -> None:
    """Replace `path` atomically, leaving no temp file behind either way.

    Inputs: the destination and a JSON-serialisable `payload`. Steps: create
    the parent directory, write `json.dumps(payload)` to a unique staging
    file beside the destination, flush and fsync it, then `replace_with_retry`
    onto `path`; the `finally` unlinks the staging file whether or not the
    replace happened. Output: none.

    Why: a reader sees the previous file or the next one, never a torn one,
    and the staging name is unique per writer (see `staging_path`). The
    retry rides out a reader that holds the destination open on Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = staging_path(path)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            # `json.dumps(...)` then one write, not `json.dump(payload,
            # handle)`: the streaming form calls handle.write() once per
            # token, and building the string once and writing it once
            # measured 3.9x faster (23.97 ms -> 6.18 ms on a representative
            # payload) for byte-identical output (same encoder, same
            # defaults).
            #
            # The cost is that the whole document is materialised first:
            # measured at 750k points, peak went 71.5 MB -> 167 MB, roughly
            # 1.4x the document on top of it, because the string and its
            # UTF-8 encoding coexist. That is negligible at the sizes this
            # Tower writes (the largest JSON it has persisted is a 1.71 MB
            # points.json against ~184 MB RSS), but it scales with payload,
            # so a caller writing something an order of magnitude larger
            # should revisit it.
            #
            # Not orjson, which was measured and refused because its bytes
            # differ: separators, and `1e-07` against `1e-7`.
            handle.write(json.dumps(payload))
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def replace_with_retry(temp_path: Path, path: Path) -> None:
    """os.replace, retried while a concurrent READER holds the destination.

    Public, because every writer of a file a reader can hold needs it, not
    only the JSON ones (`prepare_images` uses it for its undistorted
    keyframes).

    Windows refuses `replace()` onto a destination any handle has open,
    and `FILE_SHARE_DELETE` does not lift that: `MOVEFILE_REPLACE_EXISTING`
    fails with WinError 5 even against a share-delete handle. So a reader
    cannot make itself harmless, and the tolerance has to live here. The
    Tower->iOS result channel is such a reader: the web process polls world
    state while a build session writes it, and without this retry the
    failure lands on the writer.

    Measured on this host, 400 atomic writes against a reader looping as
    fast as it can (a far harsher case than the channel's 2 Hz poll):

        no reader                        0 / 400 failed
        reader, no retry               223 / 400 failed   (55.8%)
        reader, this retry               0 / 400 failed

    The retried run was the harsher of the two: it completed 68,455
    reader opens against the 8,648 of the failing run, because a writer
    that is not erroring out early leaves the reader more time to run.

    Retrying a rename is safe in a way that retrying most IO is not: the
    operation is atomic, so it either happened or it did not. There is no
    partial state to reconcile and no possibility of writing twice.
    """
    deadline = time.monotonic() + REPLACE_BUDGET_S
    backoff = REPLACE_BACKOFF_S
    while True:
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, REPLACE_BACKOFF_MAX_S)


def write_bytes_atomic(path: Path, write: "Callable[[BinaryIO], None]") -> None:
    """Publish a binary artifact atomically: `write` fills a temp file, and
    the destination is replaced only once the bytes are whole and on disk.

    The binary twin of `write_json_atomic`. `world_builder/global_solve.py`
    writes `solution.npz` through it: a `.npz` is a zip, a zip is only a zip
    once its central directory is written last, and the reader lives in a
    different process (the builder rebuilding the derived tree while the
    solver child it launched is still writing). A reader that opened the
    file mid-write got `BadZipFile: File is not a zip file` on a session
    holding 795 keyframes and 26,634 points. The same window is worse when
    the writer is terminated rather than merely slow (`BackgroundSolver.wait`
    kills a child that outstays a stop): a torn file would then persist and
    every later read of that world would fail the same way.

    Replacing a whole temp file closes both cases: a reader sees the
    previous solution or the next one, never half of either.

    It does not clean up after a kill. The `finally` below runs on an
    exception; it does not run on `TerminateProcess`, which is how
    `BackgroundSolver` ends a solve child that outstays a stop. A real kill
    mid-write leaves the published archive valid and a stray temp beside it,
    which `sweep_abandoned_staging` removes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = staging_path(path)
    try:
        with temp_path.open("wb") as handle:
            write(handle)
            handle.flush()
            # fsync before the replace, not after. The replace is what
            # publishes; bytes still sitting in the OS cache at that
            # moment are bytes a crash can take with the rename already
            # visible, which is the one ordering that produces a file
            # that is published and torn.
            os.fsync(handle.fileno())
        replace_with_retry(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_bytes_closed(path: Path) -> bytes:
    """Read a binary artifact whole, with the handle closed before use.

    The binary twin of `read_json_closed`, and for the same Windows
    reason: a handle held across parsing blocks a writer's `os.replace`
    (WinError 5).

    `np.load` on a path is a lazy reader (`NpzFile` keeps the zip open until
    it is closed), so callers hand these bytes to `np.load(io.BytesIO(...))`
    instead. The arrays this Tower persists are single-digit MB; the field
    session's `solution.npz` is 1.7 MB, and the extra peak is one copy of
    the compressed file.

    This is a margin, not a rescue: measured on that artifact, the lazy form
    holds the handle 23.6-31.7 ms and the eager form 0.9-1.2 ms, against a
    `REPLACE_BUDGET_S` of 2000 ms, so the retry already absorbs the lazy
    reader.
    """
    with path.open("rb") as handle:
        return handle.read()


def read_json_closed(path: Path) -> dict:
    """Read and parse with the handle closed before parsing.

    Windows cannot os.replace onto an open destination (verified:
    WinError 5), so a reader holding a handle across other work can block
    a writer. Closing first makes that structurally impossible.
    """
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    return json.loads(text)


def append_jsonl(path: Path, payload: dict) -> None:
    """Append one record, and heal a torn line rather than compounding it.

    The newline check is not defensive noise. An interrupted write leaves a
    partial line with NO trailing newline, and a plain append then glues
    the next record onto the end of it -- so ONE crash destroys TWO
    records: the torn one, which is expected, and the next good one, which
    is not. `read_raw_jsonl` drops the fused line as corrupt, and the
    caller never learns the second record existed.

    Starting the new record on its own line confines the damage to the
    write that was actually interrupted. Two opens instead of one; an
    append is never the hot loop in this codebase, since even the capture
    recorder does one per delivered frame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(json.dumps(payload) + "\n")


def read_raw_jsonl(path: Path) -> tuple[list[dict], int]:
    """Parse a journal, counting genuinely-corrupt lines separately.

    A line that is not valid JSON is corruption. A well-formed line whose
    fields the caller's schema cannot interpret is NOT corruption, and
    callers treat the two differently. A torn final line from an
    interrupted write lands in the first category and is skipped without
    touching the file.
    """
    if not path.exists():
        return [], 0
    raw_records: list[dict] = []
    corrupt = 0
    # errors="replace" rather than the default strict: a write interrupted
    # mid-codepoint leaves an invalid byte sequence, and a UnicodeDecodeError
    # raised from the file iterator would take out the WHOLE journal rather
    # than the one torn line -- exactly the failure this function promises
    # not to have. Not reachable through this module's own writes today
    # (json.dumps defaults to ensure_ascii, so every byte written is ASCII
    # and a tear can only fall on a character boundary), but the promise
    # should not depend on that staying true.
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping corrupt line at %s:%s", path, line_number)
                corrupt += 1
    return raw_records, corrupt
