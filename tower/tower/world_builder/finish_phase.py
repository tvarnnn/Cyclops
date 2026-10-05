"""What the live builder is doing between Stop and the end of its final solve (T-UX1).

WORLD-BUILDER-WORLDS.md §2b; CARTRIDGE-RESULTS.md §10.1 `lifecycle.processing`. Behind
`TOWER_WORLD_FINISH_STAGES`, off by default: off, nothing here writes a byte, and the
status producer and the listing never call `read_phase` or `project`.

WHY A FILE OF ITS OWN. After Stop the builder keeps the world's writer lock through the
final solve (a child process) and the final build -- about 30 of the 45 minutes a long walk
waits -- and nothing on disk said which part it was in. The stage lives in one small
atomic file in the solve workspace, `<world>/solve/<session>/finish_phase.json`, so the
session record, every product file and every solve output stay exactly what they were.

WHY IT IS STAMPED (review H-1). A stage is only worth saying while the process that wrote
it is still the one doing the work. The file names three things:

- the OWNER: the builder that holds the world lock (pid + process start time);
- the WRITER: the builder itself, or the final-solve child it handed a token to;
- the EPOCH: the session record's `finalization.started_at`, written by
  `stop_session(hold_lock=True)` for this finalization and no other.

`project` says the stage only when the record is still `pending` under that epoch, the
lock's live holder IS the owner, and the writer is still running. Anything else -- a child
that died, a finisher or a re-finish holding the lock later, a file left behind -- and the
answer is absent, and the phone says what it said before this existed.

Only two processes can write: the builder parent (`PhaseWriter.for_builder`) and the
final-solve child it spawned with `child_environment()` (`writer_from_environment`).
Background solves, the finisher's re-gate, a re-finish, a hand-run `world_finalize.py` and
every test that does not install a writer get no token, so `mark` is a no-op in them.

This module imports nothing from the solver, so the status path can import it cheaply.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import os
import time

from tower.config import world_finish_stages_setting
from tower.storage import write_json_atomic

logger = logging.getLogger(__name__)

FILENAME = "finish_phase.json"
FORMAT = "wb-finish-phase/1"
TOKEN_ENV = "TOWER_WORLD_FINISH_PHASE"

WAITING, PREPARING, MATCHING, PLACING, CHECKING, ASSEMBLING = (
    "waiting", "preparing", "matching", "placing", "checking", "assembling")
STAGES = (WAITING, PREPARING, MATCHING, PLACING, CHECKING, ASSEMBLING)
# `config.WORLD_SOLVE_CONSENSUS_VALUES[-1]`: the most draws a consensus maps.
MAX_STEPS = 7


def phase_path(store, world_id: str, session_id: str):
    """`<world>/solve/<session>/finish_phase.json`: the directory `global_solve.workspace_for`
    names (pinned equal by a test), spelled here so the status path never imports the solver."""
    return store.world_dir(world_id) / "solve" / session_id / FILENAME


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _identity_ok(identity) -> bool:
    return (isinstance(identity, dict) and _is_int(identity.get("pid"))
            and _is_finite(identity.get("created_at")))


def _valid_step(stage, n, of):
    """`(n, of)` when it may be said: only on `checking`, ints (not bools), 1 <= n <= of <= 7."""
    if stage != CHECKING or not (_is_int(n) and _is_int(of)):
        return None
    if not 1 <= n <= of <= MAX_STEPS:
        return None
    return (n, of)


def _pending_epoch(session):
    """The record's finalization epoch (`started_at`) when it is `pending`, else None."""
    from tower.world_builder.records import FINALIZATION_PENDING  # noqa: PLC0415

    finalization = getattr(session, "finalization", None)
    if not isinstance(finalization, dict) or finalization.get("state") != FINALIZATION_PENDING:
        return None
    started_at = finalization.get("started_at")
    return started_at if _is_finite(started_at) else None


class PhaseWriter:
    """One process's writer of the session's finish phase. Every method never raises."""

    def __init__(self, path, *, world_id: str, session_id: str, epoch: float, owner: dict,
                 writer: dict) -> None:
        self.path = path
        self.world_id = world_id
        self.session_id = session_id
        self.epoch = epoch
        self.owner = dict(owner)
        self.writer = dict(writer)
        self._last = None
        self._warned = False

    @classmethod
    def for_builder(cls, store, world_id: str, session_id: str) -> "PhaseWriter | None":
        """The builder parent's writer, or None.

        None unless `TOWER_WORLD_FINISH_STAGES` is on, the session record's `finalization`
        is `pending` with a finite `started_at` (what `stop_session(hold_lock=True)` writes),
        and this process's identity carries a start time. owner = writer = this process."""
        try:
            if not world_finish_stages_setting():
                return None
            epoch = _pending_epoch(store.read_session(world_id, session_id))
            if epoch is None:
                return None
            from tower.world_builder.store import process_identity  # noqa: PLC0415

            me = process_identity(os.getpid())
            if not _identity_ok(me):
                return None
            return cls(phase_path(store, world_id, session_id), world_id=world_id,
                       session_id=session_id, epoch=epoch, owner=me, writer=me)
        except Exception:  # noqa: BLE001 -- a phase is not worth a walk
            logger.warning("[Tower][WorldBuilder] the finish phase cannot be recorded for %s",
                           session_id, exc_info=True)
            return None

    def _warn(self, what: str) -> None:
        if not self._warned:
            self._warned = True
            logger.warning("[Tower][WorldBuilder] could not %s the finish phase %s",
                           what, self.path, exc_info=True)

    def mark(self, stage: str, step=None) -> None:
        """Replace the file with this stage.

        Ignores a word not in STAGES; keeps `step` (`(n, of)`) only for CHECKING and only
        when 1 <= n <= of <= MAX_STEPS; does nothing when (stage, step) equals the last mark
        this writer made. A failure is logged at WARNING once per writer and swallowed."""
        try:
            if stage not in STAGES:
                return
            kept = None
            if step is not None:
                n, of = step
                kept = _valid_step(stage, n, of)
            key = (stage, kept)
            if key == self._last:
                return
            doc = {"format": FORMAT, "session_id": self.session_id, "epoch": self.epoch,
                   "owner": self.owner, "writer": self.writer, "stage": stage}
            if kept is not None:
                doc["step"] = {"n": kept[0], "of": kept[1]}
            doc["written_at"] = time.time()
            write_json_atomic(self.path, doc)
            self._last = key
        except Exception:  # noqa: BLE001 -- a phase is not worth a walk
            self._warn("write")

    def clear(self) -> None:
        """Unlink the file (missing is fine). Failure: WARNING once, swallowed."""
        try:
            self._last = None
            self.path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            self._warn("clear")

    def child_environment(self) -> dict:
        """The token the final-solve child needs to write as this builder's child."""
        token = {"format": FORMAT, "world_id": self.world_id, "session_id": self.session_id,
                 "epoch": self.epoch, "owner": self.owner}
        return {TOKEN_ENV: json.dumps(token, sort_keys=True)}


def writer_from_environment(store, world_id: str, session_id: str,
                            environ=None) -> "PhaseWriter | None":
    """The final-solve child's writer, or None.

    None unless TOKEN_ENV is present, parses, has this FORMAT, names this world and session,
    and its epoch equals the record's `finalization.started_at` with the record `pending`
    (read now, from the record the parent wrote before the spawn). owner = the token's owner;
    writer = this process. It does NOT read the switch: the token is the switch's carrier.
    Never raises."""
    try:
        env = os.environ if environ is None else environ
        raw = env.get(TOKEN_ENV)
        if not raw:
            return None
        token = json.loads(raw)
        if not isinstance(token, dict) or token.get("format") != FORMAT:
            return None
        if token.get("world_id") != world_id or token.get("session_id") != session_id:
            return None
        epoch = token.get("epoch")
        owner = token.get("owner")
        if not _is_finite(epoch) or not _identity_ok(owner):
            return None
        if _pending_epoch(store.read_session(world_id, session_id)) != epoch:
            return None
        from tower.world_builder.store import process_identity  # noqa: PLC0415

        me = process_identity(os.getpid())
        if not _identity_ok(me):
            return None
        return PhaseWriter(phase_path(store, world_id, session_id), world_id=world_id,
                           session_id=session_id, epoch=epoch, owner=owner, writer=me)
    except Exception:  # noqa: BLE001
        logger.warning("[Tower][WorldBuilder] the finish-phase token was refused", exc_info=True)
        return None


_current: contextvars.ContextVar = contextvars.ContextVar("world_finish_phase", default=None)


def install(writer):
    """Make `writer` (or None) the one `mark` writes through. Returns the reset token."""
    return _current.set(writer)


def uninstall(token) -> None:
    try:
        _current.reset(token)
    except Exception:  # noqa: BLE001
        pass


def mark(stage: str, step=None) -> None:
    """The installed writer's mark, or nothing. A no-op in every process the builder did not
    hand a token: background solves, the finisher's re-gate, a re-finish, world_finalize.py,
    tests."""
    writer = _current.get()
    if writer is not None:
        writer.mark(stage, step)


def _document_ok(doc) -> bool:
    return (isinstance(doc, dict) and doc.get("format") == FORMAT
            and isinstance(doc.get("session_id"), str) and _is_finite(doc.get("epoch"))
            and _identity_ok(doc.get("owner")) and _identity_ok(doc.get("writer"))
            and isinstance(doc.get("stage"), str))


def read_phase(path) -> dict | None:
    """The phase document, or None unless it is a dict with FORMAT, a string session_id, a
    finite epoch, owner/writer identity dicts (int pid, not bool; finite created_at) and a
    string stage. Never raises."""
    try:
        if not path.is_file():
            return None
        from tower.world_builder.store import _read_json_past_a_replace  # noqa: PLC0415

        doc = _read_json_past_a_replace(path, absent_on_failure=True)
    except Exception:  # noqa: BLE001 -- truncated, malformed, unreadable: no phase
        return None
    return doc if _document_ok(doc) else None


def project(doc, *, session_id: str, session, holder, is_running=None) -> dict | None:
    """The wire's `processing` (WORLDS §2b), or None. Never raises.

    `{"stage": s}`, or `{"stage": "checking", "step": {"n": n, "of": N}}`, only when every one
    holds:
      1. `doc` is a phase document for this session;
      2. the record is closed, its finalization `pending`, and its `started_at` IS the
         document's epoch;
      3. `holder` (the world's lock holder as the status path passes it -- None where the lock
         does not speak for this session) is alive and its pid is the document's owner's;
      4. the owner is still running as the same process (pid AND start time);
      5. the writer is the owner, or is still running;
      6. the stage is one of STAGES (a bad `step` is dropped, the stage kept).
    """
    try:
        if is_running is None:
            from tower.world_builder.store import process_is_running as is_running  # noqa: PLC0415,N813
        if not _document_ok(doc) or doc["session_id"] != session_id:
            return None
        if getattr(session, "ended_at", None) is None:
            return None
        epoch = _pending_epoch(session)
        if epoch is None or doc["epoch"] != epoch:
            return None
        if not isinstance(holder, dict) or holder.get("alive") is not True:
            return None
        owner, writer = doc["owner"], doc["writer"]
        if not _is_int(holder.get("pid")) or holder["pid"] != owner["pid"]:
            return None
        stage = doc["stage"]
        if stage not in STAGES:
            return None
        if not is_running(owner):
            return None
        if writer != owner and not is_running(writer):
            return None
        out = {"stage": stage}
        step = doc.get("step")
        if isinstance(step, dict):
            kept = _valid_step(stage, step.get("n"), step.get("of"))
            if kept is not None:
                out["step"] = {"n": kept[0], "of": kept[1]}
        return out
    except Exception:  # noqa: BLE001
        return None
