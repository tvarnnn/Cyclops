"""On-disk world storage.

Deliberately the same shape as tower/object_memory/store.py: atomic JSON
via temp+fsync+replace inside try/finally, append-only JSONL journals,
raw-dict rewrites, corrupt-line tolerance, and a real purge that removes
every artifact including temp files.

That shape, not a numbered-checkpoint/HEAD/mmap design, because of three
Windows behaviours verified on the development host:

    directory fsync           -> PermissionError [Errno 13]
    replace onto an OPEN dest -> PermissionError [WinError 5]
    unlink an mmap'd file     -> PermissionError [WinError 32]

A checkpoint design needs a validating-recovery path, a fallback policy,
and a retrying garbage collector purely to survive those. Two of the three
do not arise here at all: no directory is renamed and nothing is mmap'd.
The third is solved by the lock this store inherits, and for a reader in
another process by retrying the read across the writer's replace
(`_read_json_past_a_replace`).

Derived output is a single rebuildable directory rather than numbered
checkpoints. Staleness is detected by comparing a digest of the inputs
that produced it, so a stale derived tree is detected rather than trusted.
"""

import hashlib
import json
import math
import os
import re
import time
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from tower.storage import (
    staging_path,
    append_jsonl,
    read_json_closed,
    read_raw_jsonl,
    write_json_atomic,
)

from tower.world_builder.records import (
    SegmentPlacement,
    Keyframe,
    KeyframeEdge,
    Session,
    World,
    keyframe_edge_from_json_dict,
    keyframe_from_json_dict,
    session_from_json_dict,
    world_from_json_dict,
)
from tower.world_builder.schema import (
    POSE_CONVENTION,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

WORLD_FILENAME = "world.json"
SESSION_FILENAME = "session.json"
KEYFRAMES_FILENAME = "keyframes.jsonl"
EDGES_FILENAME = "edges.jsonl"
EVENTS_FILENAME = "events.jsonl"
LOCK_FILENAME = "LOCK"
DERIVED_DIRNAME = "derived"
DERIVED_MANIFEST = "manifest.json"
IMAGES_DIRNAME = "images"
# A re-redacted keyframe set sits beside `images/` and never replaces it
# (`world_builder/reredaction.py`); the pointer names which one builds read.
REREDACTED_IMAGES_PREFIX = "images.redacted-"
REDACTION_SET_FILENAME = "redaction_set.json"


@dataclass(frozen=True)
class KeyframeImageSet:
    """Which keyframe images a build reads, and the label describing them."""

    directory: Path
    name: str
    redaction: str | None          # the label a reader applies its trust rule to
    stored_redaction: str | None   # what `session.json` records for `images/`
    digest: str | None             # the set's per-frame digest; None for `images/`

    @property
    def reredacted(self) -> bool:
        return self.name != IMAGES_DIRNAME

    @property
    def cache_token(self) -> str | None:
        """None for the stored set, so every cache key written before
        re-redaction existed stays valid; otherwise names the set exactly."""
        return f"{self.name}@{self.digest}" if self.reredacted else None


# How many times `acquire_writer_lock` will lose the exclusive create
# before it gives up. Contention here is two processes reclaiming one
# dead lock, which resolves in one round; this is a bound, not a wait.
_LOCK_ACQUIRE_ATTEMPTS = 8

# How long an unreadable lock is given to become readable before it is
# treated as a writer that died between its create and its write. The write
# is one `json.dumps` and an `fsync`; a tenth of a second is four orders of
# magnitude more than that and still imperceptible to a wearer.
_LOCK_UNREADABLE_GRACE_S = 0.1
# A pause between attempts, so eight of them span long enough to outlast a
# peer's reclaim rather than burning through in a millisecond. Without it
# every loser of a natural race exhausted all eight attempts in 1.2 ms and
# reported "contending" instead of naming the live holder -- 104 of 104
# times.
_LOCK_RETRY_SLEEP_S = 0.02


class WorldStoreError(Exception):
    """Base for storage-layer failures."""


class UnsupportedSchemaError(WorldStoreError):
    """A persisted artifact declares a schema version we cannot read."""


class UnknownPoseConventionError(WorldStoreError):
    """A world declares a pose convention this build does not recognise.

    Raised rather than defaulted. A bare [x, y, z] read under the wrong
    convention does not fail loudly -- it yields a plausible, mirrored,
    wrong answer, and every downstream reference inherits the error.
    """


class WorldLockedError(WorldStoreError):
    """Another live process holds this world's writer lock."""


@dataclass(frozen=True)
class PurgeReport:
    """What a purge actually managed to delete.

    Carries `retained` because a purge that could not remove everything
    must never report success: 06-PRIVACY-DATA requires real deletion, and
    a false claim of deletion is worse than an honest failure.
    """

    removed: tuple[str, ...]
    retained: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.retained


def require_schema(data: dict, what: str) -> None:
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"{what} declares schema_version {version!r}, this build reads "
            f"{SCHEMA_VERSION}. Refusing rather than guessing at the meaning "
            "of fields written by a different version."
        )


def require_pose_convention(convention: dict) -> None:
    if convention != POSE_CONVENTION:
        differing = sorted(
            key
            for key in set(convention) | set(POSE_CONVENTION)
            if convention.get(key) != POSE_CONVENTION.get(key)
        )
        raise UnknownPoseConventionError(
            "world declares a pose convention this build does not recognise; "
            f"differing keys: {differing}. Refusing to interpret coordinates "
            "rather than silently reading them under the wrong convention."
        )


# Five attempts over ~14 ms of backoff. `write_json_atomic`'s own
# `replace_with_retry` carries a 2,000 ms budget on the writer side, so
# the two ladders overlap rather than race each other to a failure.
_REPLACE_READ_ATTEMPTS = 5
_REPLACE_READ_BACKOFF = 0.001


def _read_json_past_a_replace(path, *, absent_on_failure=True):
    """`read_json_closed`, surviving a writer replacing the file underneath.

    **ONE BUILD REPLACES SIX FILES MICROSECONDS APART** -- `world.json`
    from `write_world`, then poses, points, support and both manifests
    from `write_derived`, which `engine.build` calls on the very next
    statement -- and on Windows `os.replace` onto
    a path a reader holds open fails with WinError 5, symmetrically with
    the reader's own `open()` during the writer's window. Unretried, that
    `PermissionError` walked out through `build_manifest` as an HTTP
    **500** on the geometry route on **2.4% of requests** beside a live
    writer -- a route whose 404 means ABSENT only, and for which the phone
    has no branch -- and on the status channel, caught by `snapshot()`, it
    blinks the whole world out of existence for a poll.

    Retrying works because a replace is over in microseconds. Measured with
    two immediate retries (before the backoff in the loop below was added):
    a reader went from 2.77% failures to **0%** on a 1.3 MB file, and 2.60%
    to 0.39% at 2,360 writes in six seconds, with **zero** additional
    writer failures -- `replace_with_retry` already carries a 2,000 ms
    budget and the worst observed write used 6% of it.

    A `ValueError` is NOT retried and is re-raised for the caller's own
    handler: a file that parsed and was wrong will parse and be wrong
    again. An `OSError` that survives the retries is a real fault. By
    default (`absent_on_failure=True`) it is logged and returned as None,
    "no manifest", because those callers treat an absent manifest as
    "nothing here can judge it" -- honest and degraded, where the raise is
    a 500 the phone cannot read. With `absent_on_failure=False` it is
    re-raised.
    """
    last = None
    for attempt in range(_REPLACE_READ_ATTEMPTS):
        try:
            return read_json_closed(path)
        except OSError as exc:
            last = exc
            if attempt + 1 < _REPLACE_READ_ATTEMPTS:
                # A SMALL BACKOFF, AND ONLY ON FAILURE.
                #
                # Three back-to-back attempts are not enough against a
                # writer that starts its next replace immediately: an
                # unthrottled writer (~80 write_derived/s, far hotter than
                # any real build) put all three inside one collision window
                # and the PermissionError still escaped. The successful path
                # never sleeps, and the whole ladder is 14 ms against a
                # 500 ms poll.
                time.sleep(_REPLACE_READ_BACKOFF * (2 ** attempt))
    if not absent_on_failure:
        # RETRY IS NOT SWALLOW, and this half keeps that true.
        #
        # `read_derived` deliberately leaves `OSError` out of its except
        # tuple so that a genuine disk fault (an injected EIO) is a 500
        # rather than a silent 404. A transient Windows replace collision
        # is not a disk fault, though, and telling them apart is exactly
        # what the retries do: what survives them is reported as itself.
        raise last
    logger.warning(
        "world builder: %s stayed unreadable (%r); treating it as absent",
        path, last,
    )
    return None


class WorldStore:
    """Filesystem storage for one root directory of worlds."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        # Guards journal appends (live path) against rewrite/purge, both
        # of which replace or unlink files out from under a concurrent
        # append -- PermissionError on Windows, silent line loss on POSIX.
        # Non-reentrant: locked public methods call `_locked` helpers, not
        # each other.
        self._lock = threading.Lock()

    # -- paths ---------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    def world_dir(self, world_id: str) -> Path:
        return self._root / "worlds" / world_id

    def world_path(self, world_id: str) -> Path:
        return self.world_dir(world_id) / WORLD_FILENAME

    def session_dir(self, world_id: str, session_id: str) -> Path:
        return self.world_dir(world_id) / "sessions" / session_id

    def session_path(self, world_id: str, session_id: str) -> Path:
        return self.session_dir(world_id, session_id) / SESSION_FILENAME

    def keyframes_path(self, world_id: str, session_id: str) -> Path:
        return self.session_dir(world_id, session_id) / KEYFRAMES_FILENAME

    def edges_path(self, world_id: str, session_id: str) -> Path:
        return self.session_dir(world_id, session_id) / EDGES_FILENAME

    def events_path(self, world_id: str, session_id: str) -> Path:
        return self.session_dir(world_id, session_id) / EVENTS_FILENAME

    def images_dir(self, world_id: str, session_id: str) -> Path:
        """The keyframes the capture wrote: AUTHORITATIVE, never rewritten.

        Writers (the engine) and the re-redaction step's source read this.
        A reader that wants the pixels a build may use asks
        `keyframe_image_set` instead, which honours a re-redaction switch.
        """
        return self.session_dir(world_id, session_id) / IMAGES_DIRNAME

    def redaction_set_path(self, world_id: str, session_id: str) -> Path:
        return self.session_dir(world_id, session_id) / REDACTION_SET_FILENAME

    def keyframe_image_set(self, world_id: str, session_id: str) -> KeyframeImageSet:
        """THE ONE ACCESSOR for which keyframe images a build reads, and the
        redaction label that describes them.

        Normally that is `images/` under the label `session.json` records. After
        `scripts/world_reredact.py --apply` it is the re-redacted set beside it
        (`images.redacted-<slug>/`), named by `redaction_set.json`, under the
        label that set was written with. The pointer is honoured only while it
        is whole: it names a directory of that exact shape that exists, and the
        stored label it was made from is still the one `session.json` records.
        Anything else reads as "no switch": the authoritative keyframes, whose
        fill on every label the step accepts contains the set's.

        Contract: WORLD-BUILDER-APPEARANCE.md section 6.5.
        """
        try:
            stored = self.read_session(world_id, session_id).redaction
        except Exception:  # noqa: BLE001 -- unreadable is untrusted, never an error
            stored = None
        stored = stored if isinstance(stored, str) and stored else None
        default = KeyframeImageSet(
            directory=self.images_dir(world_id, session_id), name=IMAGES_DIRNAME,
            redaction=stored, stored_redaction=stored, digest=None)
        path = self.redaction_set_path(world_id, session_id)
        if not path.exists():
            return default
        try:
            pointer = _read_json_past_a_replace(path)
        except ValueError:
            pointer = None
        if not isinstance(pointer, dict):
            logger.warning("world builder: redaction set pointer unreadable at %s; "
                           "reading the stored keyframes", path)
            return default
        active = pointer.get("active")
        if active is None:
            return default
        label = pointer.get("redaction")
        digest = pointer.get("set_digest")
        whole = (isinstance(active, str)
                 and active.startswith(REREDACTED_IMAGES_PREFIX)
                 and len(active) > len(REREDACTED_IMAGES_PREFIX)
                 and "/" not in active and "\\" not in active and ".." not in active
                 and isinstance(label, str) and bool(label)
                 and isinstance(digest, str) and bool(digest)
                 and pointer.get("stored_redaction") == stored)
        directory = self.session_dir(world_id, session_id) / str(active)
        if not whole or not directory.is_dir():
            logger.warning(
                "world builder: redaction set pointer at %s is not usable (active=%r, "
                "stored label now %r, recorded %r); reading the stored keyframes",
                path, active, stored, pointer.get("stored_redaction"))
            return default
        return KeyframeImageSet(directory=directory, name=active, redaction=label,
                                stored_redaction=stored, digest=digest)

    def read_redaction_set_pointer(self, world_id: str, session_id: str) -> dict | None:
        path = self.redaction_set_path(world_id, session_id)
        if not path.exists():
            return None
        try:
            data = _read_json_past_a_replace(path)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def write_redaction_set_pointer(self, world_id: str, session_id: str,
                                    pointer: dict) -> None:
        """Switch, or switch back, a session's keyframe set: one atomic replace."""
        with self._lock:
            write_json_atomic(self.redaction_set_path(world_id, session_id), pointer)

    def derived_dir(self, world_id: str) -> Path:
        return self.world_dir(world_id) / DERIVED_DIRNAME

    def lock_path(self, world_id: str) -> Path:
        return self.world_dir(world_id) / LOCK_FILENAME

    # -- worlds --------------------------------------------------------

    def list_world_ids(self) -> list[str]:
        worlds_root = self._root / "worlds"
        if not worlds_root.exists():
            return []
        return sorted(
            entry.name
            for entry in worlds_root.iterdir()
            if entry.is_dir() and (entry / WORLD_FILENAME).exists()
        )

    def write_world(self, world: World) -> None:
        with self._lock:
            write_json_atomic(self.world_path(world.world_id), world.to_json_dict())

    def read_world(self, world_id: str) -> World:
        path = self.world_path(world_id)
        if not path.exists():
            raise WorldStoreError(f"no world at {path}")
        # THE SIXTH FILE. `engine.build` calls `write_world` and then, on
        # the very next statement, `write_derived` -- so `world.json` is
        # replaced microseconds before the five derived files. With those
        # five read through the retry and this line on a bare read, escapes
        # out of `build_manifest` fell from 9.35% to 0.00% against a writer
        # running only `write_derived`, and stopped at **1.10%** against
        # one running `write_world` too -- every residual one a
        # `PermissionError` from this line.
        #
        # `absent_on_failure=False`, because this method's contract is to
        # RAISE for a world it cannot produce (`WorldStoreError` above),
        # and callers distinguish that from an absent world. A persistent
        # fault must keep reaching them.
        try:
            data = _read_json_past_a_replace(path, absent_on_failure=False)
        except ValueError as exc:
            # A CORRUPT WORLD IS NOT A SERVER FAULT. `read_json_closed`
            # raises `JSONDecodeError` (a `ValueError`) on a truncated or
            # non-UTF-8 `world.json`; let out, it passes
            # `world_builder_geometry._read`, which catches only
            # `WorldStoreError`, and `routes/geometry.py`, which has no
            # handler at all: an HTTP **500 on an unauthenticated route**
            # for a world whose file is merely damaged.
            #
            # `WorldStoreError` is the word every caller here already
            # understands, and it is the honest one: this world cannot be
            # produced. A genuine `OSError` still escapes, which keeps
            # "the disk is broken" a 500 rather than a silent 404.
            raise WorldStoreError(f"world {world_id} is unreadable: {exc}") from exc
        if not isinstance(data, dict):
            # A top-level list or string parses fine and then raises
            # `AttributeError` out of `require_schema`'s `.get` -- an HTTP
            # 500 on `GET /worlds` and the status producer raising on
            # every poll, picker and panel blind while the file exists.
            # Not shown reachable from the Tower's own writers, which is
            # why it is a `WorldStoreError` rather than a wider net.
            raise WorldStoreError(
                f"world {world_id} is unreadable: top level is "
                f"{type(data).__name__}, not an object"
            )
        require_schema(data, f"world {world_id}")
        require_pose_convention(data["pose_convention"])
        return world_from_json_dict(data)

    # -- sessions ------------------------------------------------------

    def write_session(self, session: Session) -> None:
        with self._lock:
            write_json_atomic(
                self.session_path(session.world_id, session.session_id),
                session.to_json_dict(),
            )

    def read_session(self, world_id: str, session_id: str) -> Session:
        data = read_json_closed(self.session_path(world_id, session_id))
        require_schema(data, f"session {session_id}")
        return session_from_json_dict(data)

    def list_session_ids(self, world_id: str) -> list[str]:
        sessions_root = self.world_dir(world_id) / "sessions"
        if not sessions_root.exists():
            return []
        return sorted(
            entry.name
            for entry in sessions_root.iterdir()
            if entry.is_dir() and (entry / SESSION_FILENAME).exists()
        )

    # -- keyframes and edges -------------------------------------------

    def append_keyframe(self, world_id: str, keyframe: Keyframe) -> None:
        with self._lock:
            append_jsonl(
                self.keyframes_path(world_id, keyframe.session_id),
                keyframe.to_json_dict(),
            )

    def read_keyframes(self, world_id: str, session_id: str) -> list[Keyframe]:
        raw_records, _ = read_raw_jsonl(self.keyframes_path(world_id, session_id))
        return _parse_all(raw_records, keyframe_from_json_dict, "keyframe")

    def clear_edges(self, world_id: str, session_id: str) -> None:
        """Drop the edge journal before a rebuild recomputes it.

        Edges between keyframes are recomputed from the keyframes on every
        build, so despite living in an append-only journal they are
        derived. Appending without clearing duplicates the whole set per
        rebuild.
        """
        with self._lock:
            self.edges_path(world_id, session_id).unlink(missing_ok=True)

    def append_edge(
        self, world_id: str, session_id: str, edge: KeyframeEdge
    ) -> None:
        with self._lock:
            append_jsonl(self.edges_path(world_id, session_id), edge.to_json_dict())

    def read_edges(self, world_id: str, session_id: str) -> list[KeyframeEdge]:
        raw_records, _ = read_raw_jsonl(self.edges_path(world_id, session_id))
        return _parse_all(raw_records, keyframe_edge_from_json_dict, "edge")

    def append_event(self, world_id: str, session_id: str, event) -> None:
        with self._lock:
            append_jsonl(
                self.events_path(world_id, session_id), event.to_json_dict()
            )

    def read_events(
        self, world_id: str, session_id: str, after_event_id: int | None = None
    ) -> list[dict]:
        """Every event, or only those strictly newer than a cursor.

        The cursor is what turns this journal from an archive into a live
        stream: a reader keeps the last `event_id` it saw and asks for
        what came after. Because `event_id` is dense within a session, a
        gap in what comes back means an event was genuinely dropped -- so
        a viewer can tell it missed something rather than quietly showing
        an incomplete world.

        A record with no `event_id` is skipped when a cursor is given.
        There is no way to advance past it, so returning it would hand it
        back on every poll forever.
        """
        raw_records, _ = read_raw_jsonl(self.events_path(world_id, session_id))
        if after_event_id is None:
            return raw_records
        return [
            record
            for record in raw_records
            if isinstance(record.get("event_id"), int)
            and record["event_id"] > after_event_id
        ]

    # -- images --------------------------------------------------------

    def write_keyframe_image(
        self, world_id: str, session_id: str, filename: str, jpeg_bytes: bytes
    ) -> Path:
        """Persist one keyframe image, durably, BEFORE its journal line.

        The ordering is deliberate and asymmetric: a journal line pointing
        at a missing image is corruption, while an image with no journal
        line is a harmless orphan that purge and verification both sweep.
        Never invert this.
        """
        images = self.images_dir(world_id, session_id)
        images.mkdir(parents=True, exist_ok=True)
        path = images / filename
        # `staging_path`, not the destination name plus a fixed suffix: a
        # staging name derived only from the destination is shared by every
        # writer of it. Keyframe filenames are unique per session so a
        # collision is unlikely here, but the same pattern in
        # `write_bytes_atomic` was measured producing 656 torn reads. One
        # convention, one place.
        temp_path = staging_path(path)
        try:
            with temp_path.open("wb") as handle:
                handle.write(jpeg_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
        return path

    # -- derived -------------------------------------------------------

    def derived_manifest_path(self, world_id: str) -> Path:
        return self.derived_dir(world_id) / DERIVED_MANIFEST

    def clear_derived(self, world_id: str) -> None:
        """Delete the whole derived tree.

        Safe by construction: every byte under derived/ is rebuildable
        from the authoritative journals plus the keyframe images. This is
        also the operation the "derived is genuinely derived" test drives.
        """
        derived = self.derived_dir(world_id)
        if derived.exists():
            shutil.rmtree(derived, ignore_errors=True)

    def write_derived_manifest(self, world_id: str, manifest: dict) -> None:
        write_json_atomic(self.derived_manifest_path(world_id), manifest)

    def read_derived_manifest(self, world_id: str) -> dict | None:
        path = self.derived_manifest_path(world_id)
        if not path.exists():
            return None
        try:
            data = _read_json_past_a_replace(path)
        except ValueError:
            # `ValueError`, NOT `json.JSONDecodeError`. The latter is a
            # subclass, and `UnicodeDecodeError` -- which is what invalid
            # UTF-8 raises -- is a sibling. Catching only `JSONDecodeError`,
            # the same three bad bytes in each copy of one manifest gave
            # opposite answers: the session copy (catching `ValueError`)
            # refused cleanly, the world copy raised out of every reader
            # that touches it, including out of `read_derived`'s verify
            # gate, which sits ABOVE its own `try`. Two files meant to be
            # identical have to fail identically.
            logger.warning("world builder: derived manifest unreadable at %s", path)
            return None
        return data if isinstance(data, dict) else None

    def session_manifest_path(self, world_id: str, session_id: str) -> Path:
        """The manifest beside a session's own poses and points.

        Absent for anything built before `write_derived` started writing it.
        `read_derived_manifest` above is the WORLD's, which names whichever
        session built last; this one always describes the session it sits
        in.
        """
        return self.derived_dir(world_id) / session_id / DERIVED_MANIFEST

    def read_session_manifest(self, world_id: str, session_id: str) -> dict | None:
        path = self.session_manifest_path(world_id, session_id)
        if not path.exists():
            return None
        try:
            data = _read_json_past_a_replace(path)
        except ValueError:
            logger.warning("world builder: session manifest unreadable at %s", path)
            return None
        return data if isinstance(data, dict) else None

    def write_derived(
        self,
        world_id: str,
        session_id: str,
        *,
        poses,
        points,
        manifest,
        support=None,
    ) -> None:
        """Write the rebuildable reconstruction outputs.

        Poses and points are JSON rather than .npy: at V1 scale a session
        is hundreds of poses and low tens of thousands of points, JSON
        round-trips exactly without a dtype contract to get wrong, and it
        stays inspectable by eye. The trigger to switch to .npy is a world
        large enough that a viewer needs partial or memory-mapped reads --
        the same "small enough that rewriting wholesale is fine" reasoning
        object memory's store already documents for itself.

        `support` is the 2-D/3-D association -- which feature in which
        keyframe produced which point -- and goes in its OWN file rather
        than into a points.json row. points.json is the source for a live
        cross-platform wire contract (docs/contracts/
        WORLD-BUILDER-GEOMETRY.md) whose row shape is pinned by test; a
        second file costs a reader one open and costs that contract
        nothing. `None` writes no file at all, which is what every world
        built before this existed looks like on disk.
        """
        # Under the same lock as purge_world. Without it an in-flight
        # build can recreate a world directory that purge has just
        # reported as completely deleted -- and the resurrected tree is
        # then invisible to list_world_ids(), so no later purge targets it.
        with self._lock:
            derived = self.derived_dir(world_id) / session_id
            write_json_atomic(derived / "poses.json", {"poses": poses})
            write_json_atomic(derived / "points.json", {"points": points})
            if support is not None:
                write_json_atomic(derived / "support.json", {"support": support})
            # THE SAME MANIFEST, BESIDE THE FILES IT DESCRIBES.
            #
            # A world has ONE `derived/manifest.json` and it names whichever
            # session built last. The status producer discards the manifest
            # for any other session (correctly -- attributing one session's
            # figures to another is worse), and then has no figures at all:
            # without this copy an older session of a world walked twice
            # reports no geometry, no poses, no currency, and the phone
            # renders a red "Needs retry" over a reconstruction sitting on
            # disk.
            #
            # A session that describes itself avoids all of that. Cheap
            # (one small JSON per build, beside megabytes of points),
            # atomic like everything else here, and additive: a world built
            # before this has no per-session copy and reads exactly as it
            # did.
            write_json_atomic(derived / DERIVED_MANIFEST, manifest)
            write_json_atomic(self.derived_manifest_path(world_id), manifest)

    def read_derived(
        self, world_id: str, session_id: str, *, verify: bool = True
    ) -> dict | None:
        """Read the derived reconstruction, refusing stale output.

        The digest is checked by default rather than on request. Serving a
        derived tree that no longer matches the journal that produced it
        is silent corruption of exactly the kind this store is built to
        prevent: the numbers look fine, they are just answers to an older
        question. Returning None makes a stale tree indistinguishable from
        an absent one, which is the honest outcome -- both mean "rebuild".

        `support` is OPTIONAL and reads as None when the file is not
        there. It arrived after ~29 worlds were already on disk, and a
        reconstruction is complete without it -- it is an index into the
        reconstruction, not part of it. A missing support.json is
        therefore absent, never an error, and never a reason to refuse
        poses and points that are perfectly good.
        """
        if verify:
            digest = compute_input_digest(
                self.read_keyframes(world_id, session_id)
            )
            # `is False`, NOT `not ...`. `None` is "no manifest here can
            # judge this" -- a legacy world walked twice, where the only
            # manifest describes another session. Refusing that is a
            # guaranteed 404 for a reconstruction that is sitting on disk
            # and perfectly good; serving it with the wire contract's
            # `current` flag OFF is the honest compromise, and the status
            # channel says in words why it cannot be judged. Only a
            # manifest that actually disagrees is stale.
            if self.derived_currency(world_id, digest, session_id) is False:
                logger.warning(
                    "world builder: derived output for %s/%s is stale; "
                    "treating as absent",
                    world_id, session_id,
                )
                return None
        derived = self.derived_dir(world_id) / session_id
        poses_path = derived / "poses.json"
        points_path = derived / "points.json"
        if not poses_path.exists() or not points_path.exists():
            return None
        try:
            # THROUGH THE RETRY, for the reason `_read_json_past_a_replace`
            # gives at length -- and with `absent_on_failure=False`, so a
            # real fault still raises out of here.
            #
            # These two are the LARGEST files `write_derived` replaces, so
            # they hold the collision window open longest. With a live
            # writer beside the geometry route, a bare read here escaped as
            # an HTTP 500, one layer above the manifest readers.
            return {
                "poses": _read_json_past_a_replace(
                    poses_path, absent_on_failure=False
                )["poses"],
                "points": _read_json_past_a_replace(
                    points_path, absent_on_failure=False
                )["points"],
                "support": self._read_support(derived),
            }
        except (KeyError, TypeError, ValueError) as exc:
            # `TypeError` BELONGS HERE, as in `_read_support` and
            # `read_placements` below: a top-level list (`[]`) hits the same
            # subscript, and without it the TypeError comes out through
            # `build_manifest` as an HTTP 500 where every other corrupt
            # derived tree gives a 404. A corrupt world and an absent one
            # are both "nothing to serve"; neither is a server fault.
            # (`json.JSONDecodeError` is a `ValueError`, so it is in here
            # by inheritance rather than by being listed twice.)
            #
            # `OSError` IS NOT IN THIS TUPLE. With it, an injected EIO
            # became "no geometry for this session" -- a 404 on a route
            # whose 404 means ABSENT only, and an empty reconstruction in
            # `world_inspect`. A server that cannot read its own storage
            # should say so, loudly, and 500 is how.
            logger.warning(
                "world builder: derived output unreadable for %s/%s: %s: %s",
                world_id, session_id, type(exc).__name__, exc,
            )
            return None

    def write_placements(self, world_id: str, session_id: str, placements) -> None:
        """Persist where each segment sits, or why it does not sit anywhere.

        Its own file, for the same reason `support` has one: geometry is
        immutable once solved and PLACEMENT is not, so a later registration
        pass must be able to change where a segment sits without rewriting
        the points that decide its content hash. Folding placement into
        points.json would couple the two and make every re-placement look
        like new geometry.
        """
        with self._lock:
            derived = self.derived_dir(world_id) / session_id
            payload = {"placements": [p.to_json_dict() for p in placements]}
            # allow_nan=False: Python writes NaN and Infinity as bare
            # tokens that JSON.parse, Swift's JSONSerialization and Go's
            # encoding/json all reject. A file only this runtime can read
            # is not a wire artifact. Raising here leaves no file, because
            # the write is atomic.
            json.dumps(payload, allow_nan=False)
            write_json_atomic(derived / "placements.json", payload)

    def read_placements(self, world_id: str, session_id: str):
        """The placements, or None. Never raises, never refuses a read.

        Absent and unreadable are the same answer, exactly as for
        `support`: every world built before placements existed has no such
        file, and a reconstruction is complete without one. Refusing a
        world over a truncated index beside it would turn an optional file
        into a hard dependency by the back door.

        A row that fails its own invariants is dropped rather than
        poisoning the rest -- a placement that cannot be represented is
        one that must not be drawn, and the others are still good.
        """
        path = self.derived_dir(world_id) / session_id / "placements.json"
        if not path.exists():
            return None
        # Broad on purpose. JSONDecodeError and KeyError alone cover a
        # truncated file and nothing else -- `{"placements": null}`, a
        # top-level list, a string, or a null row are all VALID JSON and
        # each would raise straight through build_manifest into an HTTP
        # 500, taking poses and points that are perfectly good down with an
        # optional index beside them. That is exactly what this method's
        # contract exists to prevent, so the contract is enforced rather
        # than described.
        try:
            document = read_json_closed(path)
            rows = document["placements"]
            if not isinstance(rows, list):
                raise TypeError(f"placements must be a list, got {type(rows)}")
        except Exception:
            logger.warning(
                "world builder: placements unreadable at %s; treating as "
                "absent",
                path,
            )
            return None
        kept = []
        for row in rows:
            try:
                kept.append(SegmentPlacement.from_json_dict(row))
            except Exception:
                logger.warning(
                    "world builder: dropping unrepresentable placement in "
                    "%s: %r",
                    path,
                    row,
                )
        return kept

    def _read_support(self, derived: Path):
        """The association, or None. Never raises, never refuses a read.

        Unreadable is treated the same as missing on purpose. Poses and
        points do not become wrong because an index beside them is
        truncated, and refusing the whole derived tree over it would turn
        an optional file into a hard dependency by the back door -- which
        is precisely what keeping it out of points.json was meant to
        avoid. Logged, because a corrupt file is still worth knowing about.
        """
        path = derived / "support.json"
        if not path.exists():
            return None
        try:
            # Through the retry as well: a replace collision here would
            # otherwise be absorbed by the `except Exception` below and
            # silently drop an index that is perfectly good, one poll
            # after the build that wrote it.
            support = _read_json_past_a_replace(path)["support"]
            # Shape-checked, not just parsed. A top-level list would raise
            # TypeError straight out of a method whose docstring promises
            # it never raises, and a string would be returned AS the support
            # table -- which cross-segment registration then consumes.
            if not isinstance(support, list):
                raise TypeError(
                    f"support must be a list, got {type(support)}"
                )
            return support
        except Exception:
            logger.warning(
                "world builder: support association unreadable at %s; "
                "treating as absent",
                path,
            )
            return None

    def derived_currency(
        self, world_id: str, input_digest: str, session_id: str | None = None
    ):
        """True, False, or **None for "nothing here can judge it"**.

        THE THIRD ANSWER IS THE POINT; folding it into either boolean is
        wrong. Two different questions are asked of it:

          * *is this geometry current?* -- what the wire contract's
            `current` flag means, "reflects every keyframe accepted so
            far", and what the status channel reports.
          * *may this geometry be served at all?* -- what
            `read_derived`'s verify gate decides.

        For a world built before per-session manifests existed, walked
        twice, the honest answer to the first is "unknown" and to the
        second is "yes, with the flag off". Returning `True` makes the
        ROUTE assert `current: true` over geometry that one appended
        keyframe makes genuinely stale -- while the status channel beside
        it says `current: false`. Returning `False` refuses to serve a good
        reconstruction. Neither is the answer; there are three.
        """
        manifest = None
        if session_id is not None:
            manifest = self.read_session_manifest(world_id, session_id)
            if manifest is not None and manifest.get("session_id") != session_id:
                # A session's own copy naming somebody else is corruption,
                # not a world walked twice.
                manifest = None
        if manifest is None:
            world_manifest = self.read_derived_manifest(world_id)
            if not isinstance(world_manifest, dict):
                world_manifest = None
            if (
                session_id is not None
                and world_manifest is not None
                and world_manifest.get("session_id") != session_id
            ):
                # The world's manifest is about another session and this
                # one has no copy of its own: a legacy world walked twice.
                return None
            manifest = world_manifest
        if manifest is None:
            # NO MANIFEST ANYWHERE IS THE THIRD ANSWER, NOT `False`.
            #
            # `False` means "a manifest exists and disagrees", and
            # `read_derived`'s gate refuses on exactly that. So a world
            # built before per-session manifests existed, or one whose
            # manifests were lost, would 404 for a reconstruction sitting on
            # disk -- while the status channel, which recounts such a
            # session's poses and points, reports it `ready`: a promise the
            # wearer's next tap would break.
            #
            # Nothing is lost by serving it. The wire contract carries
            # `current`, the route sets it from this same three-valued
            # answer, and the status channel says in words that currency
            # cannot be judged.
            return None
        return (
            manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("input_digest") == input_digest
        )

    def world_bytes(self, world_id: str) -> dict:
        """On-disk size of one world, split by tier.

        Reported rather than capped. A hard cap that stopped mapping
        mid-session would trade a storage problem for a silently truncated
        world, which is worse: the operator would get an incomplete map
        with no obvious sign it was cut short. Making growth visible lets
        retention be decided deliberately -- and the derived figure is the
        reclaimable half, since every byte under derived/ is rebuildable.
        """
        directory = self.world_dir(world_id)
        if not directory.exists():
            return {"total": 0, "images": 0, "derived": 0, "journals": 0}

        totals = {"total": 0, "images": 0, "derived": 0, "journals": 0}
        derived_root = self.derived_dir(world_id)
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            size = path.stat().st_size
            totals["total"] += size
            if path.suffix == ".jpg":
                totals["images"] += size
            elif derived_root in path.parents:
                totals["derived"] += size
            elif path.suffix in (".jsonl", ".json"):
                totals["journals"] += size
        return totals

    # -- locking -------------------------------------------------------

    def acquire_writer_lock(self, world_id: str) -> None:
        """Take the single-writer lock, reclaiming it from a dead process.

        Liveness by pid AND process start time rather than by timeout. A
        stale-lock timer has to guess how long a legitimate writer might
        pause; asking the OS whether the pid is still running does not
        guess, and psutil is already a dependency. The start time is
        there because Windows recycles pids aggressively: a lock left by a
        builder that died could otherwise name a pid that now belongs to
        an unrelated process, and the next session would be refused for
        as long as that stranger lived (see `lock_holder`).
        """
        path = self.lock_path(world_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # CREATE-EXCLUSIVE, NOT CHECK-THEN-WRITE.
        #
        # Reading the holder and then writing the lock, with nothing atomic
        # in between, lets two processes arriving in that window both see
        # "no lock" and both proceed. The second write overwrites the
        # first's record, so the file names only one of them -- and when the
        # first finishes, `release_writer_lock` unlinks a lock the OTHER
        # writer still believes it holds. Check-then-write admitted TWO
        # SIMULTANEOUS WRITERS IN 8 OF 8 TRIALS, and drove two concurrent
        # `world_finalize.py` runs to `finalized: True` on one world.
        # `world_finalize.py` calls this "the whole safety story".
        #
        # `O_CREAT | O_EXCL` is one atomic operation on Windows and POSIX
        # alike: exactly one caller creates the file. Everyone else falls
        # through to the liveness check, and a caller that decides the
        # holder is gone RECLAIMS by unlinking and trying the exclusive
        # create again -- so two processes that both find a dead lock still
        # cannot both win, and the loser re-reads and sees the winner.
        unreadable_since: float | None = None
        for _attempt in range(_LOCK_ACQUIRE_ATTEMPTS):
            try:
                handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                pass
            else:
                with os.fdopen(handle, "w", encoding="utf-8") as file:
                    file.write(json.dumps(_lock_record(os.getpid())))
                    file.flush()
                    os.fsync(file.fileno())
                # WON THE CREATE -- NOW CONFIRM WE STILL HOLD IT.
                #
                # The create is atomic; the reclaim below is not. A peer
                # that decided this lock was dead unlinks whatever is at
                # the path -- not the file it read -- so it can delete a
                # lock created since, and then create its own. With a stall
                # injected in the reclaim window that is BOTH PROCESSES
                # ACQUIRING, 5 of 5 (0 of 120 naturally, so it is narrow,
                # not imaginary).
                #
                # Reading our own record back closes the outcome that
                # matters: whoever's record is on disk owns the world, and
                # a process whose record was replaced goes round rather
                # than returning to write underneath the winner.
                mine = self.lock_holder(world_id)
                if mine is not None and mine["pid"] == os.getpid():
                    return
                continue
            holder = self.lock_holder(world_id)
            if holder is None:
                # Unreadable. Two very different things look like this and
                # only time tells them apart.
                #
                # TRANSIENT: a peer is between its `O_CREAT|O_EXCL` and its
                # write, so the file exists and is empty for microseconds.
                # Reclaiming here would delete a live writer's lock.
                #
                # ORPHANED: that peer was killed in the same window, and
                # the zero-byte file it left names nobody (the exclusive
                # create can leave a partial lock; `write_json_atomic`
                # never does). Refusing it forever bricks the world:
                # `acquire_writer_lock` raised in 1.3 ms and on every retry,
                # and `world_finalize.py` was refused identically.
                # Permanently.
                #
                # So: wait out the transient case, then reclaim. The write
                # window is measured in microseconds; anything unreadable
                # for a tenth of a second is not a writer in progress.
                now = time.monotonic()
                if unreadable_since is None:
                    unreadable_since = now
                elif now - unreadable_since >= _LOCK_UNREADABLE_GRACE_S:
                    logger.warning(
                        "world builder: reclaiming an unreadable lock on %s; it "
                        "names no process and has not become readable in %ss, so "
                        "it is a writer that died mid-write",
                        world_id, _LOCK_UNREADABLE_GRACE_S,
                    )
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    unreadable_since = None
                time.sleep(_LOCK_RETRY_SLEEP_S)
                continue
            unreadable_since = None
            if holder["alive"] and holder["pid"] != os.getpid():
                raise WorldLockedError(
                    f"world {world_id} is locked by live pid {holder['pid']}; "
                    "refusing a second writer"
                )
            logger.warning(
                "world builder: reclaiming lock on %s held by pid %r",
                world_id,
                holder["pid"],
            )
            # Unlink and go round: the create is what decides, not this.
            try:
                path.unlink()
            except OSError:
                pass
            time.sleep(_LOCK_RETRY_SLEEP_S)
        raise WorldLockedError(
            f"world {world_id}: could not take the writer lock in "
            f"{_LOCK_ACQUIRE_ATTEMPTS} attempts; another writer is contending "
            "for it"
        )

    def lock_holder(self, world_id: str) -> dict | None:
        """Who holds the writer lock, and whether that process is alive.

        None when no readable lock exists. Otherwise
        `{"pid": int|None, "alive": bool, "unreadable": bool}`:
        `unreadable` is a lock file that names no usable pid, which is NOT
        "no lock" -- reporting a healthy idle world over a file that is
        right there would downgrade a crashed builder to nothing at all.

        One answer for the store, the status producer and the listing, so
        the three cannot disagree about whether a world is live.
        """
        path = self.lock_path(world_id)
        try:
            if not path.exists():
                return None
            # THROUGH THE RETRY. Swallowing a transient `OSError` here
            # returns "no lock", which every caller reads as "this world
            # is idle" -- so a collision on the LOCK file reports a live
            # walk as dead: `live: false` in the picker, and the status
            # channel dropping out of `receiving` for a poll. On a bare
            # read, the one test that asserts a live session reads
            # `receiving` failed once under full-suite load and passed 3/3
            # otherwise.
            holder = _read_json_past_a_replace(path)
            # `None` after the retries falls through to the `isinstance`
            # check below, which already answers `unreadable: True` -- NOT
            # "no lock", which is the downgrade this docstring warns
            # about. No branch is needed here.
            lock_written_at = path.stat().st_mtime
        except (OSError, ValueError):
            return None
        if not isinstance(holder, dict):
            return {"pid": None, "alive": False, "unreadable": True}
        pid = holder.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool):
            return {"pid": None, "alive": False, "unreadable": True}
        return {
            "pid": pid,
            "alive": _holder_is_running(
                pid, holder.get("created_at"), lock_written_at
            ),
            "unreadable": False,
        }

    def release_writer_lock(self, world_id: str) -> None:
        self.lock_path(world_id).unlink(missing_ok=True)

    # -- purge ---------------------------------------------------------

    def purge_world(self, world_id: str) -> PurgeReport:
        """Really delete a world: images, journals, derived tree, temps.

        Reports what could not be removed rather than claiming success.
        06-PRIVACY-DATA requires real deletion, and the failure mode this
        guards against is a purge that returns a success count while
        leaving a temp file holding live data behind.
        """
        with self._lock:
            world_dir = self.world_dir(world_id)
            removed: list[str] = []
            retained: list[str] = []
            # THE WORLD'S AREA BUILD DIRECTORIES FIRST (`area_build.AREA_BUILD_DIRNAME`,
            # `<root>/.ab/<area>`): an area is built outside the world directory for
            # MAX_PATH's sake, and a killed build leaves keyframe imagery there. Found
            # by the owner marker each one carries, so even an area whose record is
            # gone is purged with its world. Before the world itself, while its areas
            # can still be recognised.
            from tower.world_builder.area_build import purge_area_builds  # noqa: PLC0415

            try:
                staged_removed, staged_retained = purge_area_builds(self, world_id)
                removed.extend(staged_removed)
                retained.extend(staged_retained)
            except Exception as exc:  # noqa: BLE001 -- reported, never claimed as purged
                logger.warning("world builder: could not purge the area builds of %s: %s",
                               world_id, exc)
                retained.append(str(self.root / ".ab"))
            if not world_dir.exists():
                return PurgeReport(removed=tuple(removed), retained=tuple(retained))

            for path in sorted(
                world_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True
            ):
                try:
                    if path.is_dir():
                        path.rmdir()
                    else:
                        path.unlink()
                    removed.append(str(path))
                except OSError as exc:
                    logger.warning("world builder: could not remove %s: %s", path, exc)
                    retained.append(str(path))

            try:
                world_dir.rmdir()
                removed.append(str(world_dir))
            except OSError as exc:
                logger.warning(
                    "world builder: could not remove %s: %s", world_dir, exc
                )
                retained.append(str(world_dir))

            return PurgeReport(removed=tuple(removed), retained=tuple(retained))


REQUIRED_MANIFEST_KEYS = (
    "input_digest",
    "session_id",
    "keyframes",
    "points",
    "poses_solved",
    "poses_refused",
    "segments",
)


def session_has_drawable_geometry(store, world_id, session_id, manifest=None) -> bool:
    """Would opening this session SHOW the wearer anything?

    Sparse geometry, OR a reconstruction the viewer ladder would actually
    serve. The second half exists because the render route asks this
    question BEFORE walking the ladder: without it a session with a
    complete surface and an empty sparse tree is refused as "no geometry
    yet", and the listing puts `has_geometry: false` on it, so the picker
    shows a reconstructed world as empty.

    Both halves apply the same standard the ladder does to the same files,
    because two surfaces disagreeing about one session is the failure the
    sparse half's own docstring describes.
    """
    if _sparse_drawable(store, world_id, session_id, manifest):
        return True
    return reconstruction_drawable(store, world_id, session_id)


def reconstruction_drawable(store, world_id, session_id) -> bool:
    """True when a surface or dense artifact would render for this session.

    Cheap enough for a listing over a whole world root: the manifest is read
    and each level file is checked by size, and a surface level by its
    header, without decoding any geometry. A torn level fails the size check;
    that is the case that matters, since every artifact write is atomic and
    a torn file only arises from a copy or a disk fault.
    """
    world_dir = store.world_dir(world_id)
    return (_surface_drawable(world_dir / "surface" / session_id)
            or _dense_drawable(world_dir / "dense" / session_id))


def surface_artifact_drawable(store, world_id, session_id) -> bool:
    """Whether this session's surface artifact would render. See
    `reconstruction_drawable`. Not one built from a re-redacted keyframe set
    the session no longer reads (`built_from_an_inactive_keyframe_set`)."""
    return (_surface_drawable(store.world_dir(world_id) / "surface" / session_id)
            and not built_from_an_inactive_keyframe_set(store, world_id, session_id, "surface"))


def dense_artifact_drawable(store, world_id, session_id) -> bool:
    """Whether this session's dense point artifact would render (and was not
    built from a keyframe set the session no longer reads)."""
    return (_dense_drawable(store.world_dir(world_id) / "dense" / session_id)
            and not built_from_an_inactive_keyframe_set(store, world_id, session_id, "dense"))


_SET_IN_PARAMS_DIGEST = re.compile(r"\|set:([^|]+)")


def built_from_an_inactive_keyframe_set(store, world_id, session_id, kind: str) -> bool:
    """Whether the `kind` (`surface` or `dense`) artifact's colours came from a
    re-redacted keyframe set that is not the one the session reads now.

    After `world_reredact.py --revert` the appearance route stops serving the
    set's imagery at once (its label check names the set), but the surface and
    dense pages -- vertex colours and point colours from the same pixels --
    would still be served until rebuilt. Such an artifact is reported as not
    drawable, so the ladder, the revision and the listing all
    step past it until it is rebuilt from the active set.

    Only that direction: an artifact built from the capture's own `images/`
    (no set recorded) is never stale by this test, because every label a switch
    accepts fills at least what the set does. An unreadable pointer is treated
    as "not the same set", the safe answer.
    """
    man = _read_manifest_quietly(store.world_dir(world_id) / kind / session_id / "manifest.json")
    if man is None:
        return False
    built = man.get("keyframe_image_set")
    if not built and kind == "surface":
        match = _SET_IN_PARAMS_DIGEST.search(str(man.get("params_digest") or ""))
        built = match.group(1) if match else None
    if not built:
        return False
    try:
        active = store.keyframe_image_set(world_id, session_id).cache_token
    except Exception:  # noqa: BLE001 -- unreadable is "not this set"
        return True
    return built != active


def _read_manifest_quietly(path):
    try:
        man = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return man if isinstance(man, dict) else None


def _level_files_whole(root, levels, name) -> bool:
    if not isinstance(levels, list) or not levels:
        return False
    for lv in levels:
        if not isinstance(lv, dict):
            return False
        level, size = lv.get("level"), lv.get("bytes")
        if not isinstance(level, int) or not isinstance(size, int):
            return False
        # A level may name its own file (surface builds published as a unit);
        # it must be a bare name in this directory. Otherwise the legacy name.
        fname = lv.get("file", name.format(level))
        if (not isinstance(fname, str) or "/" in fname or "\\" in fname
                or fname.startswith(".")):
            return False
        try:
            if (root / fname).stat().st_size != size:
                return False
        except OSError:
            return False
    return True


def _surface_drawable(root) -> bool:
    from tower.world_builder.surface import (  # noqa: PLC0415
        _HEADER,
        MESH_MAGIC,
        SURFACE_FORMAT,
        SURFACE_SCHEMA_VERSION,
    )

    man = _read_manifest_quietly(root / "manifest.json")
    if (man is None or man.get("format") != SURFACE_FORMAT
            or man.get("schema_version") != SURFACE_SCHEMA_VERSION):
        return False
    faces = man.get("faces")
    if not isinstance(faces, int) or faces <= 0:
        return False
    levels = man.get("levels")
    if not _level_files_whole(root, levels, "mesh_l{}.bin"):
        return False
    try:
        last = levels[-1]
        with open(root / last.get("file", f"mesh_l{last['level']}.bin"), "rb") as handle:
            head = handle.read(_HEADER.size)
        magic, _nv, n_i, _flags, version, *_box = _HEADER.unpack(head)
    except Exception:  # noqa: BLE001 -- short or foreign header
        return False
    return magic == MESH_MAGIC and version == SURFACE_SCHEMA_VERSION and n_i > 0


def _dense_drawable(root) -> bool:
    from tower.world_builder.dense import DENSE_FORMAT  # noqa: PLC0415

    man = _read_manifest_quietly(root / "manifest.json")
    if man is None or man.get("format") != DENSE_FORMAT:
        return False
    levels = man.get("levels")
    if not isinstance(levels, list) or not any(
            isinstance(lv, dict) and isinstance(lv.get("points"), int)
            and lv["points"] > 0 for lv in levels):
        return False
    return _level_files_whole(root, levels, "points_l{}.bin")


def _sparse_drawable(store, world_id, session_id, manifest=None) -> bool:
    """Would opening this session SHOW the wearer anything?

    **Not "do the files exist".** `engine.build` calls `write_derived`
    unconditionally, so a walk that solved nothing still leaves `poses.json`
    and a `points.json` holding `{"points": []}` -- 14 bytes. Eleven
    sessions on the real 163-world root are exactly that, and asked "do the
    files exist" they list as `complete, has_geometry: true` while the
    status channel for the same session projects `needsRetry`.

    Two surfaces need this answer and they must not compute it apart:

      * `results/world_builder_library` puts it on the wire as
        `has_geometry`, which `WorldPickerView` branches on;
      * `results/world_builder_render.resolve_session` uses it to choose
        WHICH session to draw, so an empty newer walk would otherwise be
        picked over an older one that has geometry, and the page drawn
        blank.

    A third predicate, `results/world_builder._has_session_geometry`,
    deliberately still answers EXISTENCE -- it decides which lifecycle
    state a session is in ("was there a build"), which is a different
    question, and its own test says so.

    `manifest` is passed in when the caller already has it, which the
    listing does; otherwise it is read here. The count of `points.json` is
    the fallback for a session no manifest describes -- 0 of the 49 real
    sessions with a tree, and every one of those manifests carries both
    figures.
    """
    derived = store.derived_dir(world_id) / session_id
    if not ((derived / "poses.json").exists() and (derived / "points.json").exists()):
        return False
    if manifest is None:
        # `manifest_describing`, NOT A HAND-ROLLED READ.
        #
        # A loose local rule -- `isinstance(dict)` and `session_id` --
        # re-admits exactly the manifests `validate_manifest` refused: the
        # caller passes the strict answer, `None`, and a looser one would
        # be substituted. A session whose manifest claims `points: 500`
        # under a schema this build does not know, over `poses.json` and
        # `points.json` that are both empty, then gives **three answers
        # from one payload** -- picker "complete", panel "Needs retry",
        # render page blank. Five manifest readers were counted, and they
        # must not compute this answer apart.
        manifest = manifest_describing(
            store, world_id, session_id, purpose="figures"
        )
    if isinstance(manifest, dict):
        points = manifest.get("points")
        positioned = manifest.get("poses_positioned")
        if isinstance(points, int) or isinstance(positioned, int):
            return (points or 0) > 0 or (positioned or 0) > 0
    return _points_on_disk(store, world_id, session_id) > 0


def _points_on_disk(store, world_id, session_id) -> int:
    """`len(points.json)`, for a session no manifest summarises.

    The rows are dropped as soon as they are counted; only the length is
    kept. Unreadable is not empty -- but the only honest answer inside a
    bool is the one that does not promise a wearer something to look at,
    and the status channel says the difference in words.
    """
    path = store.derived_dir(world_id) / session_id / "points.json"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))["points"]
    except (KeyError, TypeError, ValueError, OSError):
        return 0
    return len(rows) if isinstance(rows, list) else 0


def manifest_describing(store, world_id, session_id, *, purpose):
    """The manifest that describes THIS session, judged for one PURPOSE.

    **ONE READER, ONE RULE -- and the rule depends on the question.** Two
    questions are asked of a manifest and they need different standards:

    ``"figures"`` -- how many points, how many poses. The status channel
    reports these and the picker draws from them, so a manifest that
    cannot be trusted field-by-field must not supply them: schema checked,
    required keys checked.

    ``"identity"`` -- WHICH BUILD produced this tree, i.e. `input_digest`.
    `usable_placements` and `_is_current` compare that digest and nothing
    else. Holding them to the figures refuses manifests that answer their
    question perfectly, and holding them to the SCHEMA is worse:
    `SCHEMA_VERSION` versions the whole record family (`World`, `Session`,
    `Keyframe`, `KeyframeEdge`), the manifest inherits
    `world.schema_version` rather than the module constant, and
    `poses.json`/`points.json` rows carry no version at all -- their shape
    is pinned by `world_builder.geometry/2026-08-25`, which has never
    moved under it. Bumping the constant against the real root under the
    schema check drops **408 placements across 10 sessions to 0**, every
    segment becoming a disconnected island, on a change that has nothing
    to do with a Sim3.

    The session's own copy first, then the world's but only if it names
    this session -- a manifest naming somebody else is not evidence about
    this one.
    """
    figures = purpose == "figures"
    try:
        manifest = validate_manifest(
            store.read_session_manifest(world_id, session_id),
            world_id,
            source="session manifest",
            require_figures=figures,
            require_schema=figures,
        )
        if manifest is not None and manifest.get("session_id") == session_id:
            return manifest
        world = validate_manifest(
            store.read_derived_manifest(world_id),
            world_id,
            require_figures=figures,
            require_schema=figures,
        )
        if world is not None and world.get("session_id") == session_id:
            return world
    except (WorldStoreError, OSError, ValueError, KeyError, TypeError):
        # OSError BELONGS HERE. `write_derived` replaces five files
        # microseconds apart, and on Windows a replace onto a path a
        # reader has open -- and a reader's open during the writer's
        # window -- fails with WinError 5. The manifest readers retry it
        # (`_read_json_past_a_replace`); whatever still escapes them must
        # read as "cannot judge" here. Unguarded, the escape reached
        # `GET /worlds/...` as an HTTP **500** on 2.4% of requests beside a
        # live writer, on a route whose 404 means ABSENT only and for which
        # the phone has no branch.
        return None
    return None


def validate_manifest(
    manifest,
    world_id,
    source="derived manifest",
    *,
    require_figures=True,
    require_schema=True,
):
    """Schema-check a manifest, wherever it was read from, or None.

    **HERE, RATHER THAN IN ONE READER, BECAUSE THERE ARE FOUR READERS**:
    the status producer, and `results/world_builder_geometry._session_manifest`,
    which the geometry route, `usable_placements` and the saved-worlds
    listing all reach. With the check in the status producer only, a
    reader checking just `isinstance(dict)` and `session_id` disagrees
    with it about any manifest that is readable and wrong: for a session
    whose manifest declares a schema this build does not know, and whose
    derived tree is gone, the picker said `interrupted` -- "a build ran and
    its output is gone" -- while the status channel behind that same row
    said `stopped_unbuilt`, "this walk produced no geometry".

    A no-op on real data: all 49 sessions with a derived tree on the
    163-world root pass both the loose and the strict rule.
    """
    if not isinstance(manifest, dict):
        # `isinstance`, not `is None`. A world manifest holding a
        # top-level list or string reached `.get()` and came out as an
        # AttributeError -- a 500 on the status channel, where the
        # identically corrupt per-session copy gave a clean refusal.
        return None
    if require_schema and manifest.get("schema_version") != SCHEMA_VERSION:
        # A manifest from another schema describes fields whose meaning
        # this build does not know. It is refused as a SUMMARY; the poses
        # and points beside it are a separate question, and
        # `_figures_from_the_tree` answers it separately.
        return None
    if not require_figures:
        # IDENTITY AND PROVENANCE ONLY, which is all some readers need.
        #
        # The figure check below exists because the STATUS CHANNEL reports
        # those figures, and "geometry: available with every count null"
        # is a claim with nothing behind it. A reader asking "which build
        # produced this" needs only `session_id` and `input_digest`, and
        # holding it to the figures refuses manifests that answer its
        # question perfectly well.
        #
        # Without this split `usable_placements` refuses every placement
        # of a session whose manifest carries a digest and no counts, so a
        # registered segment stops serving its transform -- five tests in
        # `test_world_builder_placements.py` fail at once.
        return manifest
    missing = [key for key in REQUIRED_MANIFEST_KEYS if manifest.get(key) is None]
    if missing:
        logger.warning(
            "world builder: %s for %s is missing %s; treating it as absent "
            "rather than reporting geometry with no figures",
            source, world_id, missing,
        )
        return None
    return manifest


def compute_input_digest(keyframes: list[Keyframe]) -> str:
    """Digest the authoritative inputs a derived build consumed.

    Lets a later read detect that derived output no longer matches the
    journal that produced it, instead of trusting whatever is on disk.
    """
    hasher = hashlib.sha256()
    for keyframe in keyframes:
        hasher.update(keyframe.keyframe_id.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _parse_all(raw_records: list[dict], parser, what: str) -> list:
    parsed = []
    for raw in raw_records:
        try:
            require_schema(raw, what)
            parsed.append(parser(raw))
        except (KeyError, ValueError, UnsupportedSchemaError):
            # Valid JSON, schema this build cannot interpret. Not
            # corruption: skipped from the parsed view without touching
            # the file, matching object memory's two-stage discipline.
            logger.warning(
                "world builder: skipping %s record with an unrecognised schema",
                what,
            )
    return parsed


# How far apart two readings of one process's start time may be and still
# name the same process. psutil reports the value at millisecond precision
# on Windows and the two readings here are taken by different processes.
_CREATE_TIME_TOLERANCE_S = 1.0


def _lock_record(pid: int) -> dict:
    """What a writer puts in its lock: the pid, and when that pid started.

    The start time is what makes the pid mean one process rather than
    whichever process the OS next hands that number to.
    """
    record = {"pid": pid}
    try:
        import psutil

        record["created_at"] = float(psutil.Process(pid).create_time())
    except Exception:  # pragma: no cover - a lock without a start time still works
        pass
    return record


# How much later than the lock file a process may have started and still
# be believed to be its writer. A builder writes its lock within
# milliseconds of starting; this only has to absorb clock skew and
# filesystem timestamp granularity, and erring generous here is safe --
# the failure it prevents (calling a LIVE builder dead) is far worse than
# the one it allows (believing a pid recycled within five seconds).
_RECYCLED_PID_GRACE_S = 5.0


def _holder_is_running(pid: int, created_at, lock_written_at=None) -> bool:
    """Whether the process a lock names is the process that WROTE it.

    With a start time on the lock, a running pid whose start time differs
    is a DIFFERENT process -- the builder that wrote the lock is dead and
    its number was recycled.

    **AND WITHOUT ONE, THE LOCK FILE'S OWN MTIME ANSWERS THE SAME
    QUESTION.** A process that started AFTER the lock was written cannot
    be the process that wrote it. That is not a heuristic; it is the same
    argument the `created_at` check makes, from a timestamp the filesystem
    keeps for free.

    This is not hypothetical and it is not rare: on the retest machine
    **29 of 163 worlds hold a lock file and not one of them carries
    `created_at`** -- they were all written before the field existed.
    Trusting a bare running pid there, reproduced end to end through a
    real Tower: a pid from a fortnight-old lock was recycled onto a live
    `bash.exe`, the status producer prefers a world with a live lock over
    every saved world, and the phone was told a 16-day-old empty world was
    `receiving` with 0 keyframes and `mapping_seconds: 1,407,085`. The
    wearer's real walk built correctly and then, **at the moment
    finalization completed and released its own lock**, the live screen
    reverted to the ghost and said "Mapping" forever.

    The tolerance runs one way on purpose. A real builder writes its lock
    within milliseconds of starting, so its `create_time` is at or before
    the lock's mtime; a recycled pid on an old lock starts hours or days
    after. Only a process that started **clearly** later is called dead,
    so a live builder can never be judged dead by a slow clock or a coarse
    filesystem timestamp.

    Nothing is deleted to make this work: the 29 legacy locks stay exactly
    where they are and simply read dead, which is what they are.
    """
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        try:
            actual = psutil.Process(pid).create_time()
        except psutil.AccessDenied:
            # CANNOT JUDGE IS NOT DEAD. The process exists and its start
            # time is hidden -- an elevation mismatch between the Tower
            # and a `world_finalize.py` run is enough on Windows. Calling
            # it dead would let a second writer onto one store, which is
            # the one failure this whole lock exists to prevent; assuming
            # it alive costs at worst a refused acquisition. It is the safe
            # direction.
            return True
        except psutil.Error:
            return False
        if created_at is not None:
            return abs(float(actual) - float(created_at)) <= _CREATE_TIME_TOLERANCE_S
        if lock_written_at is not None:
            return float(actual) <= float(lock_written_at) + _RECYCLED_PID_GRACE_S
        return True
    except Exception:  # pragma: no cover - psutil is a hard dependency
        return False


def process_identity(pid: int) -> dict:
    """What a lock or a finish phase names a process by: `_lock_record(pid)`.

    `tower.world_builder.finish_phase`: the builder stamps its finish
    phase with this, so a reader can tell the process that wrote it from any
    process the OS later hands the same pid to."""
    return _lock_record(pid)


def process_is_running(identity) -> bool:
    """Whether `identity` (as `process_identity` wrote it) names a process still running.

    False for anything malformed, and -- stricter than the lock, which falls
    back to the lock file's mtime -- for an identity with no finite numeric
    `created_at`: a finish phase is only believed when the process that wrote
    it can be named exactly. Never raises."""
    try:
        if not isinstance(identity, dict):
            return False
        pid = identity.get("pid")
        created_at = identity.get("created_at")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return False
        if (not isinstance(created_at, (int, float)) or isinstance(created_at, bool)
                or not math.isfinite(created_at)):
            return False
        return _holder_is_running(pid, float(created_at))
    except Exception:  # noqa: BLE001 -- a probe that cannot tell answers "not running"
        return False
