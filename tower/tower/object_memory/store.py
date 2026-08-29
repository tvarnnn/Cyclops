import json
import logging
import os
import threading
import time
from pathlib import Path  # noqa: F401  (used by _replace_locked's annotation)

from tower.confidence import Confidence
from tower.object_memory.keyframes import KeyframeStore
from tower.object_memory.records import (
    ObjectObservation,
    object_observation_from_json_dict,
)
from tower.object_memory.relevance import PERSISTED_CLASSES

logger = logging.getLogger(__name__)

OBSERVATIONS_FILENAME = "observations.jsonl"
MANIFEST_FILENAME = "manifest.json"
TEMP_SUFFIX = ".jsonl.tmp"
MANIFEST_SCHEMA_VERSION = 1

# The window every producer run to date was actually invoked with, and
# both CLIs' default. It is also what a store with no manifest is read
# under -- see _persisted_retention_seconds_locked.
DEFAULT_RETENTION_DAYS = 30.0
DEFAULT_RETENTION_SECONDS = DEFAULT_RETENTION_DAYS * 86400.0

# How hard to try to replace a file that a reader has open.
#
# Windows refuses `os.replace` while any handle is open, and the web
# process holds this file for the length of a read. The reads are short,
# so five attempts over ~150 ms clears essentially all of them; more
# would be a busy-wait on a store that is genuinely contended, which is
# a different problem with a different fix.
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF_SECONDS = 0.01


class ObservationStore:
    """JSONL store for one module's observations.

    Mostly append-only: new observations are appended, and prune, purge's
    cleanup and an in-window best_score upgrade rewrite the file whole.

    JSONL, not SQLite: at V1 scale a single module's observation history is
    small, and the canonical-memory research explicitly sequences SQLite +
    sqlite-vec behind a measured need. Rewriting this file wholesale during
    prune/purge/upgrade is acceptable precisely because the file is expected
    to stay small; that assumption is the trigger to revisit, and Task 8
    measures it.

    RETENTION IS THE PRODUCER'S PROMISE, NOT THE READER'S CHOICE.

    `retention_seconds` here used to be the only word on the subject, so
    whoever opened the store last decided how far back it could see: a
    reader passing `--retention-days 3650` at a store written under the
    30-day default was served a 40-day-old record in full, and retention
    under 06-PRIVACY-DATA.md stopped being a promise at all. The window
    the store was WRITTEN under is now recorded in a small manifest
    beside the data, and every read and prune clamps to
    min(persisted, requested). A caller may narrow the window; nothing a
    caller passes can widen it.

    RETENTION NOW GOVERNS PICTURES AS WELL AS RECORDS.

    Since `keyframes.py`, a record may have a small filtered crop of its
    own at `<root>/keyframes/<observation_id>.jpg`. That file is only
    honestly "governed by Object Memory's retention" if the two things
    that enforce retention -- `prune_expired` and `purge` -- reach it, so
    both do.

    The store BUILDS ITS OWN `KeyframeStore` rather than being handed
    one, and that is the answer to "what if the store cannot reach a
    keyframe store": it always can. The keyframe directory is derived
    from the same root this store was constructed with, so there is no
    second path to configure, no caller who can forget to pass it, and no
    configuration in which records are pruned while their pictures are
    not. A store constructed against a root with no `keyframes/`
    directory -- every store written before this existed, and every test
    that never writes one -- finds nothing to prune and does nothing,
    which is why the constructor parameter exists only for a test that
    wants to observe the calls.

    Rewrites (prune, purge's cleanup) operate on raw JSON dicts, not on
    parsed ObjectObservation instances. That is deliberate: round-tripping
    through the current dataclass schema would silently drop any key the
    schema does not interpret (spatial_ref/external_refs today, whatever
    is reserved next tomorrow), breaking records.py's promise that a
    future cross-module need does not require rewriting already-persisted
    records. Rewriting raw dicts keeps that promise true.
    """

    def __init__(
        self,
        directory: Path,
        retention_seconds: float | None,
        *,
        clock=time.time,
        allowed_classes: tuple[str, ...] = PERSISTED_CLASSES,
        keyframes=None,
    ) -> None:
        if retention_seconds is not None and retention_seconds < 0:
            raise ValueError(
                "retention_seconds must be non-negative or None, got "
                f"{retention_seconds!r}"
            )
        self._directory = Path(directory)
        self._retention_seconds = retention_seconds
        # The whitelist is enforced HERE as well as in RelevanceFilter,
        # and this is the copy that matters: the filter guards the
        # engine's path, while this guards the disk. An `append()` from
        # anywhere else -- a script, a future consumer, a careless
        # refactor -- used to write `person` straight through a filter it
        # never passed. Defaults CLOSED to PERSISTED_CLASSES; a caller
        # running an in-process RelevancePolicy(allowed_classes=...) has
        # to hand the store the same list, which keeps that widening in
        # one visible place instead of implicit in what nobody checked.
        self._allowed_classes = frozenset(allowed_classes)
        # Injected so a READ can apply the retention cutoff without every
        # caller having to pass the time in. prune_expired keeps its
        # explicit `now`, so the deterministic tests it was written for
        # read exactly as before; this only supplies a default.
        self._clock = clock
        # Derived from this store's own root rather than injected. See
        # the class docstring: the one way to leave orphaned pictures
        # behind is to make the link something a caller can forget, so
        # there is no caller. The parameter exists for tests that want to
        # watch the calls, and for nothing else.
        self._keyframes = (
            KeyframeStore(self._directory) if keyframes is None else keyframes
        )
        # What the last `purge()` did to the owned keyframes, as
        # `(removed, retained)`. Kept on the instance because `purge()`
        # returns an observation count and a shipped CLI reads it, while
        # "the pictures could not all be deleted" is a thing a human
        # asking for erasure has to be told -- `scripts/object_query.py`
        # prints this beside the count. Never None: a purge that could
        # not run at all still reports zero of each rather than silence.
        self.last_keyframe_purge: tuple[int, tuple[str, ...]] = (0, ())
        # The store's OWN artifacts that a purge could not delete. Same
        # reason as `last_keyframe_purge`: a purge that prints a count
        # while `observations.jsonl` is still on disk is the false claim
        # of deletion `CARTRIDGE-GROUNDWORK.md` calls worse than an
        # honest failure. Never None; reset at the top of every purge.
        self.last_purge_retained: tuple[str, ...] = ()
        self._path = self._directory / OBSERVATIONS_FILENAME
        self._temp_path = self._path.with_suffix(TEMP_SUFFIX)
        self._manifest_path = self._directory / MANIFEST_FILENAME
        # Guards append (called from the live frame path) against purge
        # and prune_expired (called from FastAPI's threadpool per a later
        # task): both of the latter unlink/replace the backing file out
        # from under a concurrent append, which raises PermissionError on
        # Windows and can silently lose the appended line on POSIX.
        # Non-reentrant by design: locked public methods never call each
        # other -- they call the `_locked`-suffixed helpers below, which
        # assume the lock is already held.
        self._lock = threading.Lock()

    def append(self, observation: ObjectObservation) -> None:
        if observation.object_class not in self._allowed_classes:
            raise ValueError(
                f"object_class {observation.object_class!r} is not one this "
                "store may persist. See relevance.PERSISTED_CLASSES: "
                "widening the list is a decision about what the system is "
                "allowed to remember, not a threshold to tune."
            )
        with self._lock:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._write_manifest_locked()
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(observation.to_json_dict()) + "\n")

    def update_sighting(
        self,
        object_class: str,
        observed_at: float,
        *,
        best_score: float | None = None,
        last_seen_at: float | None = None,
        frame_count: int | None = None,
        best_frame_seq: int | None = None,
        best_relpath: str | None = None,
        best_bounding_box=None,
        verification: dict | None = None,
    ) -> bool:
        """Fold what the sighting has since become into the record on disk.

        The producer writes a record the moment a sighting matures, so a
        killed session loses nothing and `observed_at` keeps meaning "when
        it came into view". Everything a sighting only learns LATER --
        that it lasted 4.4 seconds, that the strongest look was frame
        3,410 rather than frame 3,398, that something agreed with the
        label -- is folded back into that record here rather than
        becoming a second one.

        `confidence` moves with `best_score`, in the same rewrite. That
        field is the INTERPRETATION a consumer reads, and the claim a
        record makes is "this category was in view" -- the strength of
        the evidence for that claim is the best look during the sighting,
        not the first one. A record left saying "medium" about a laptop
        the detector went on to see at 0.97 under-reports what the system
        knows. Both raw scores stay exactly as written, so the record
        remains auditable back to the sighting that created it.

        This is not the tautology the resample review warned about: that
        was raising `min_score` to MEDIUM_CONFIDENCE_MAX, which would
        make every record HIGH by construction. Here the label follows
        evidence actually observed, so a sighting the detector never saw
        clearly keeps its honest label.

        MONOTONIC WHERE MONOTONICITY IS MEANINGFUL. `best_score` is never
        revised downwards; `frame_count` and `last_seen_at` only grow. A
        second producer against the same store cannot shrink a sighting
        somebody else observed more of.

        Returns whether anything changed; an update that would change
        nothing is not written. This makes the store no longer purely
        append-only: an update is an O(n) rewrite, negligible at the size
        this file is expected to stay and already named in the class
        docstring as the trigger to move to SQLite. The producer keeps
        the rate down by updating on a better score, on a slow tick, and
        at the sighting's end -- never per frame.
        """
        with self._lock:
            raw_records, _ = self._read_raw_records()
            changed = False
            for raw in raw_records:
                if raw.get("object_class") != object_class:
                    continue
                if raw.get("observed_at") != observed_at:
                    continue
                changed |= self._apply_update(
                    raw,
                    best_score=best_score,
                    last_seen_at=last_seen_at,
                    frame_count=frame_count,
                    best_frame_seq=best_frame_seq,
                    best_relpath=best_relpath,
                    best_bounding_box=best_bounding_box,
                    verification=verification,
                )
            if changed:
                # Raw dicts, like every other rewrite here, so reserved
                # and future keys survive. Truly corrupt lines do not:
                # they were never observations, and _read_raw_records has
                # already dropped them.
                self._rewrite_locked(raw_records)
            return changed

    def _apply_update(
        self,
        raw: dict,
        *,
        best_score,
        last_seen_at,
        frame_count,
        best_frame_seq,
        best_relpath,
        best_bounding_box,
        verification,
    ) -> bool:
        changed = False
        if best_score is not None:
            current = raw.get("best_score")
            if not (self._is_number(current) and current >= best_score):
                raw["best_score"] = best_score
                # The second of the two places confidence is derived.
                # The first is the engine's initial write, where the best
                # look IS the first look; both are pinned by tests.
                raw["confidence"] = Confidence.from_score(best_score).value
                changed = True
                # The representative frame belongs to the best look, so
                # it moves with it and only with it. Passing a new frame
                # alongside a score that did not improve would leave the
                # record pointing at a weaker view than the one its
                # numbers describe.
                if best_frame_seq is not None:
                    raw["best_frame_seq"] = best_frame_seq
                if best_relpath is not None:
                    raw["best_relpath"] = best_relpath
                if best_bounding_box is not None:
                    raw["best_bounding_box"] = list(best_bounding_box)
        if last_seen_at is not None:
            current = raw.get("last_seen_at")
            if not (self._is_number(current) and current >= last_seen_at):
                raw["last_seen_at"] = last_seen_at
                changed = True
        if frame_count is not None:
            current = raw.get("frame_count")
            if not (self._is_number(current) and current >= frame_count):
                raw["frame_count"] = frame_count
                changed = True
        if verification is not None and raw.get("verification") != verification:
            raw["verification"] = verification
            changed = True
        return changed

    def update_best_score(
        self, object_class: str, observed_at: float, best_score: float
    ) -> bool:
        """The narrow form of `update_sighting`, kept because it is used.

        Not a deprecation shim: "a stronger look at the same sighting"
        is a real thing to say on its own, and every caller that only has
        that to say should not have to pass six Nones to say it.
        """
        return self.update_sighting(
            object_class, observed_at, best_score=best_score
        )

    @staticmethod
    def _is_number(value) -> bool:
        # bool is excluded: it's an int subclass in Python.
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _read_manifest_locked(self) -> dict | None:
        if not self._manifest_path.exists():
            return None
        try:
            with self._manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "object memory: unreadable manifest at %s", self._manifest_path
            )
            return None
        return manifest if isinstance(manifest, dict) else None

    def _persisted_retention_seconds_locked(self) -> float | None:
        """The window this store was WRITTEN under. None means unbounded.

        A store with no manifest is the interesting case: the records
        written before the manifest existed carry no record of what was
        promised about them. Reading them as unbounded would leave the
        hole the manifest exists to close -- any reader could widen the
        window by asking. Refusing to read them would destroy real data
        to make a point. So a manifest-less store is read as though it
        had been written under DEFAULT_RETENTION_SECONDS, which is the
        window every producer run to date was actually invoked with and
        the default both CLIs still carry. An unreadable manifest is
        treated the same way rather than trusted.

        An empty directory has nothing to protect and no promise to
        infer, so it is unbounded until the first append writes one.
        """
        manifest = self._read_manifest_locked()
        if manifest is None:
            return DEFAULT_RETENTION_SECONDS if self._path.exists() else None
        if "retention_seconds" not in manifest:
            return DEFAULT_RETENTION_SECONDS
        value = manifest["retention_seconds"]
        if value is None:
            return None
        if not self._is_number(value) or value < 0:
            logger.warning(
                "object memory: manifest at %s has an unusable "
                "retention_seconds; reading under the default",
                self._manifest_path,
            )
            return DEFAULT_RETENTION_SECONDS
        return float(value)

    @staticmethod
    def _narrower(a: float | None, b: float | None) -> float | None:
        """The tighter of two windows. None is unbounded, so it never wins."""
        if a is None:
            return b
        if b is None:
            return a
        return min(a, b)

    def _effective_retention_seconds_locked(self) -> float | None:
        """What this store will actually honour: min(persisted, requested)."""
        return self._narrower(
            self._persisted_retention_seconds_locked(), self._retention_seconds
        )

    def effective_retention_seconds(self) -> float | None:
        """The window this store will actually honour. None means unbounded.

        A read-only view of the clamp, and nothing else: it computes what
        `_effective_retention_seconds_locked` already computes for every
        read and prune, and changes no state. It exists so a transport can
        SHOW the clamp rather than merely be subject to it -- a client that
        asked for 3650 days and silently received 30 has no way to learn
        that its question was refused, and a promise nobody can observe
        being kept is the kind that quietly stops being kept.
        """
        with self._lock:
            return self._effective_retention_seconds_locked()

    def _write_manifest_locked(self) -> None:
        """Record the window at first append. Tighten it, never widen it.

        A later producer writing under a NARROWER window has made the
        stricter promise, and the manifest is where that promise lives,
        so it moves down. A later producer asking for a wider one changes
        nothing: it does not get to relax what earlier records were
        written under, and the read-time clamp would refuse it anyway.
        The result is monotonic -- the manifest always holds the tightest
        window anything has written under.
        """
        existing = self._read_manifest_locked()
        retention = self._effective_retention_seconds_locked()
        if existing is not None and (
            retention == self._persisted_retention_seconds_locked()
        ):
            return
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "retention_seconds": retention,
            # Carried over rather than restamped: the store was created
            # when it was created, whatever the window has since become.
            "created_at": (existing or {}).get("created_at", self._clock()),
        }
        # ATOMIC, and the reason is the one direction retention must never
        # move. This was a bare `open("w")`, so a producer killed between
        # the truncate and the write left a zero-byte or partial manifest
        # -- which `_persisted_retention_seconds_locked` reads as
        # unreadable and falls back to the 30-day default. A store written
        # under a 3-day promise would silently become a 30-day one, and
        # the next append would persist that. Pause and Stop terminate the
        # producer promptly, so this is not a remote possibility.
        temp = self._manifest_path.with_suffix(".json.tmp")
        try:
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
                handle.flush()
                os.fsync(handle.fileno())
            self._replace_locked(temp, self._manifest_path)
        finally:
            temp.unlink(missing_ok=True)

    def _read_raw_records(self) -> tuple[list[dict], int]:
        """Read the backing file as JSON objects, without schema validation.

        Returns the parsed dicts plus a count of lines that were not valid
        JSON at all. That is true corruption, distinct from a well-formed
        record whose fields this version's schema cannot interpret --
        callers decide separately, and differently, how to treat each.
        Must be called with the lock held.
        """
        if not self._path.exists():
            return [], 0
        raw_records = []
        corrupt = 0
        # `errors="replace"`, not strict. A single invalid byte anywhere
        # in the file used to raise `UnicodeDecodeError` out of every read
        # path -- including `purge()`, so the one operation that could
        # have cleaned it up was the one that could not run, and the HTTP
        # routes answered 500. Nothing this cartridge writes can produce
        # such a byte; a truncated write, a filesystem fault or a restored
        # backup can. Replacing it turns a bricked store into one corrupt
        # line, which the loop below already knows how to skip and
        # `prune_expired` already knows how to rewrite away.
        with self._path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(
                        "object memory: skipping corrupt line at %s:%s",
                        self._path,
                        line_number,
                    )
                    corrupt += 1
        return raw_records, corrupt

    def _parse_observations(self, raw_records: list[dict]) -> list[ObjectObservation]:
        """Every read path turns raw records into observations HERE.

        Which is why the class guard is here as well as on `append`.
        `append` refuses to persist a class this store may not hold, and
        its message names the threat model exactly: "an `append()` from
        anywhere else -- a script, a future consumer, a careless
        refactor". That guard protects the file. It does not protect the
        WIRE, and until 2026-08-27 nothing did.

        A record that reached the file by any route other than `append`
        -- a build from before the guard existed, an out-of-tree script, a
        restored backup -- was parsed and served in full: `object_class:
        "person"`, a normalised bounding box, and a `session_id` plus
        `frame_seq` that resolve to the original first-person JPEG through
        `/frame`. The payload said `recordable: false` while handing all of
        it over.

        `person` is not a tier and not a threshold. It is a separate
        constant, checked first, that no model can reach past -- and a
        refusal that only covers the writer is not that. Symmetric now:
        what this store may not persist, it also may not serve.

        Dropped silently rather than raised, for the same reason a schema
        mismatch is: one bad record must not make the whole store
        unreadable. Logged, because a record that had to be filtered here
        means something wrote where it should not have.
        """
        observations = []
        for raw in raw_records:
            object_class = raw.get("object_class")
            if object_class is not None and object_class not in self._allowed_classes:
                logger.warning(
                    "object memory: refusing to SERVE a record of class %r "
                    "from %s -- this store may not persist that class, so "
                    "something wrote it by a path that bypassed append()",
                    object_class,
                    self._path,
                )
                continue
            try:
                observations.append(object_observation_from_json_dict(raw))
            except (KeyError, ValueError):
                # Valid JSON, schema mismatch (missing field added since,
                # unknown enum label, ...). Not corruption: skip it from
                # the parsed view without touching the file or counting
                # it as a removed/unparseable line.
                logger.warning(
                    "object memory: skipping record with an unrecognised "
                    "schema in %s",
                    self._path,
                )
        return observations

    def _retention_cutoff(self) -> float | None:
        """The oldest recorded_at a read may still serve; None means no bound.

        The clamped window, not the requested one -- see the class
        docstring. Lock held: the manifest can be deleted by purge.
        """
        retention = self._effective_retention_seconds_locked()
        if retention is None:
            return None
        return self._clock() - retention

    @staticmethod
    def _is_within_retention(raw: dict, cutoff: float | None) -> bool:
        """Shared by reads and prune so the two can never disagree.

        recorded_at, not observed_at: retention is about how long WE have
        held the data, which is the privacy-relevant clock. They are
        equal today, but diverge the moment a real capture timestamp is
        threaded through. A missing or non-numeric recorded_at can't be
        shown to be within retention, so it is treated as expired.
        (bool is excluded: it's an int subclass in Python.)
        """
        if cutoff is None:
            return True
        recorded_at = raw.get("recorded_at")
        is_numeric = isinstance(recorded_at, (int, float)) and not isinstance(
            recorded_at, bool
        )
        return is_numeric and recorded_at >= cutoff

    def _all_observations_locked(
        self, cutoff: float | None
    ) -> list[ObjectObservation]:
        raw_records, _ = self._read_raw_records()
        if cutoff is not None:
            raw_records = [
                raw for raw in raw_records if self._is_within_retention(raw, cutoff)
            ]
        return self._parse_observations(raw_records)

    def _read_cutoff(self, include_expired: bool) -> float | None:
        return None if include_expired else self._retention_cutoff()

    def all_observations(
        self, *, include_expired: bool = False
    ) -> list[ObjectObservation]:
        """Every observation still within retention.

        Filtering is the DEFAULT and the opt-out has to be asked for by
        name. Retention under 06-PRIVACY-DATA.md is a promise about how
        long data stays AVAILABLE, not merely about how long it sits on
        disk; a read that ignored the cutoff made that promise true only
        for whoever remembered to call prune_expired(), which on a tower
        that stays up for days is nobody.

        `include_expired=True` exists for maintenance paths that must see
        what is physically on disk -- purge counting what it deletes, an
        operator auditing the file. It is never the right answer for
        anything a wearer will be shown.
        """
        with self._lock:
            return self._all_observations_locked(self._read_cutoff(include_expired))

    def last_seen(
        self, object_class: str, *, include_expired: bool = False
    ) -> ObjectObservation | None:
        with self._lock:
            matching = [
                o
                for o in self._all_observations_locked(
                    self._read_cutoff(include_expired)
                )
                if o.object_class == object_class
            ]
        if not matching:
            return None
        return max(matching, key=lambda o: o.observed_at)

    def purge(self) -> int:
        """Delete all observations and every file artifact the store owns.

        Returns the count of parseable observations removed. This may be
        fewer than the number of lines the file contained if there are
        unparseable lines. Both the main file and a stale rewrite temp
        file (left behind by a crash mid-_rewrite) are removed regardless.

        THE PICTURES GO TOO, and so does everything else, and what could
        not go is reported rather than swallowed. `last_purge_retained`
        holds the store's own artifacts that survived and
        `last_keyframe_purge` holds `(removed, retained)` for the crops.
        Either is non-empty only when the filesystem refused, which on
        Windows means a reader had the file open. A caller that prints
        only the observation count while a directory of crops survives is
        making the false claim of deletion `CARTRIDGE-GROUNDWORK.md`
        names as worse than an honest failure, so
        `scripts/object_query.py` prints both and exits non-zero when
        anything is left.
        """
        with self._lock:
            # Counted WITHOUT the retention cutoff: purge deletes the
            # files outright, so it must report what it actually removed
            # rather than only the part a read was still willing to serve.
            count = len(self._all_observations_locked(None))
            self.last_purge_retained = ()
            self.last_keyframe_purge = self._keyframes.purge()
            # The manifest goes too: it describes observations that no
            # longer exist, and a store that is asked to keep forever
            # after a purge must not still be bound by a window the
            # deleted records were written under.
            # THE RECORDS' HALF REPORTS ITS FAILURES TOO, and it did not.
            #
            # The keyframe half returned `(removed, retained)` while this
            # one was a bare `unlink(missing_ok=True)`. On Windows a
            # reader holding `observations.jsonl` open makes that raise
            # `PermissionError` straight out of `purge()` -- so the one
            # command a wearer's erasure request actually runs answered a
            # traceback rather than the structured "here is what I could
            # not delete" this docstring promises. A reviewer found it.
            #
            # Each artifact is attempted independently: one locked file
            # must not stop the others being removed, because a partial
            # deletion that continues is strictly better than a partial
            # deletion that stops.
            for artifact in (
                self._path,
                self._temp_path,
                self._manifest_path,
                self._manifest_path.with_suffix(".json.tmp"),
            ):
                try:
                    artifact.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "[Tower][ObjectMemory] could not delete %s during a "
                        "purge; it is still on disk",
                        artifact.name,
                    )
                    self.last_purge_retained += (artifact.name,)
            return count

    def prune_expired(self, now: float | None = None) -> int:
        """Delete expired records from disk. Reads already refuse to serve them.

        Still required, and not merely tidiness: read-time filtering stops
        expired data being SERVED, while 06-PRIVACY-DATA.md wants it gone.
        `now` stays explicit so the deterministic tests keep working, and
        defaults to the store's clock for a caller with nothing to say.

        Prunes on the CLAMPED window, exactly as reads filter on it, so
        the two can never disagree about what retention means.

        AND IT PRUNES THE PICTURES. A record's owned keyframe is deleted
        with the record, which is what makes `imagery_retention:
        "object-memory"` a promise rather than a label. `retention is
        None` still short-circuits, keyframes included: an unbounded
        store expires nothing, so there is nothing for either half to do,
        and the byte-for-byte "leaves everything untouched" behaviour
        this method has always had is unchanged.
        """
        if now is None:
            now = self._clock()
        with self._lock:
            retention = self._effective_retention_seconds_locked()
            if retention is None:
                return 0
            raw_records, corrupt = self._read_raw_records()
            cutoff = now - retention
            kept = []
            removed = 0
            for raw in raw_records:
                if self._is_within_retention(raw, cutoff):
                    kept.append(raw)
                else:
                    removed += 1
            # A KEEP list, computed from the records that SURVIVED,
            # rather than a delete list computed from the ones that did
            # not. Everything on disk that is not one of these is either
            # expired, orphaned by a failed write, or left by something
            # this cartridge does not model -- and all three should go.
            #
            # The ids are derived by the same `_parse_observations` the
            # read path uses, not read out of the raw dicts, so a
            # keyframe survives exactly when a reader could still be
            # served the record it belongs to. A record this version's
            # schema cannot parse cannot be served, so its picture is not
            # kept for a reader that will never see it.
            self._keyframes.prune(
                {
                    observation.observation_id
                    for observation in self._parse_observations(kept)
                }
            )
            # Rewrite if valid records expired OR if corrupt lines exist.
            # A corrupt line cannot be shown to be within retention, so
            # letting it survive would mean retention silently fails to
            # cover data still at rest.
            if removed or corrupt:
                self._rewrite_locked(kept)
            # Return count of removed observations only, not corrupt
            # lines: corrupt lines were never observations, just data to
            # clean up.
            return removed

    def _replace_locked(self, source: Path, destination: Path) -> None:
        """`os.replace`, retried past a Windows sharing violation.

        The lock in this class is IN-PROCESS. It serialises the producer
        against itself and does nothing about the web process, which
        holds this file open for the length of a read -- and on Windows
        `os.replace` raises `PermissionError` while any handle is open.
        Measured under a reader loop, 87-92% of rewrites failed.

        The reads are short, so a bounded retry clears essentially all of
        them. This is the same "tolerate a transient sharing violation"
        the repository already accepts elsewhere, and it is a mitigation
        rather than a fix: two processes writing one store would need a
        real lock file, and the honest place for that is the SQLite move
        the class docstring already names.
        """
        last = None
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                source.replace(destination)
                return
            except PermissionError as exc:  # noqa: PERF203
                last = exc
                time.sleep(_REPLACE_BACKOFF_SECONDS * (attempt + 1))
        raise last

    def _rewrite_locked(self, raw_records: list[dict]) -> None:
        # try/finally so a failure anywhere in the write leaves nothing
        # behind: no observations.jsonl.tmp with a live copy of data that
        # nothing else ever reads, prunes, or deletes. If replace()
        # succeeds the temp file is already gone, so the unlink is a
        # harmless no-op; if anything raises before that, it cleans up.
        try:
            with self._temp_path.open("w", encoding="utf-8") as handle:
                for raw in raw_records:
                    handle.write(json.dumps(raw) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._replace_locked(self._temp_path, self._path)
        finally:
            self._temp_path.unlink(missing_ok=True)
