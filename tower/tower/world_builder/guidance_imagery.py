"""Optional FOW thumbnail manifests and volatile tile bytes.

Redacted-only, except under TOWER_WORLD_GUIDANCE_IMAGERY_UNREDACTED=on (the
demo's deferred-privacy switch, `tower.config`): then a session whose keyframe
set is labelled exactly `none` is served from its stored keyframe JPEGs, and
every entry from it says `"redaction": "unredacted"`."""

from __future__ import annotations

import hashlib
import logging
import threading
import time

import cv2
import numpy as np

from tower.config import world_guidance_imagery_unredacted_setting
from tower.results.world_builder_geometry import contained_world_id
from tower.world_builder.appearance import label_is_trusted
from tower.world_builder.dense_pipeline import keyframe_image_bytes
from tower.world_builder.raw_imagery import IMAGERY_REDACTED
from tower.world_builder.redaction import REDACTION_NONE
from tower.world_builder.store import WorldStore, WorldStoreError

logger = logging.getLogger(__name__)
MAX_TILES = 16
MAX_BYTES = 16 * 1024
MAX_WIDTH = 160
MAX_HEIGHT = 90
# FOW capture dial: one tile per set sector, so at most 12. A sector whose
# newest candidate has no redacted tile falls back to the next newest, at most
# this many reads per sector, all inside the 80 ms budget.
DIAL_MAX_TILES = 12
DIAL_CANDIDATES_PER_SECTOR = 3
# The 80 ms landed-tile budget is the tile job's own CPU time (this thread's
# `time.thread_time()`), not wall time. The coverage worker runs at below-normal
# priority, so under a live walk's load a ~2 ms job could miss an 80 ms
# wall-clock budget by waiting for the CPU alone. A wall-clock cap still bounds
# the job, so a stalled read can never hang the worker.
LANDED_CPU_BUDGET_S = .080
LANDED_WALL_CAP_S = 1.0
# publish_landed outcomes. Only LANDED_OVER_BUDGET is worth one retry.
LANDED_PUBLISHED = "published"
LANDED_OVER_BUDGET = "over_budget"
LANDED_SKIPPED = "skipped"


def contained_session_id(store, world_id, session_id):
    """Both URL identifiers are untrusted; session must name one child."""
    try:
        root = store.session_dir(world_id, "probe").parent
        path = store.session_dir(world_id, session_id)
        return session_id if path.parent.resolve() == root.resolve() else None
    except (OSError, ValueError, TypeError):
        return None


def serves_unredacted(label):
    """True only for a set labelled exactly `none` (no redaction ran) while
    TOWER_WORLD_GUIDANCE_IMAGERY_UNREDACTED is exactly `on`. Every other
    untrusted label -- unknown strings, a `+plausibility2` rerun label, an
    unreadable session (None) -- is refused whatever the switch says."""
    return label == REDACTION_NONE and world_guidance_imagery_unredacted_setting()


def entry_redaction(label):
    """The `redaction` word an entry from a set with this label carries, or
    None when the set may not be served at all."""
    if label_is_trusted(label):
        return "redacted"
    if serves_unredacted(label):
        return "unredacted"
    return None


def _stored_keyframe_bytes(image_set, keyframe_id):
    """The stored keyframe JPEG, as persisted, with no redaction applied.
    Same file `keyframe_image_bytes` reads (`<set directory>/<seq>.jpg`)."""
    seq = keyframe_id.rsplit(":", 1)[-1]
    if not seq.isdigit():
        return None
    try:
        return (image_set.directory / f"{seq}.jpg").read_bytes()
    except (OSError, AttributeError, TypeError):
        return None


def _tile(store, world_id, session_id, keyframe_id, image_set):
    """The sole imagery read is the established redaction boundary, or, for a
    `none` set under the unredacted switch, the stored keyframe as persisted."""
    redaction = entry_redaction(image_set.redaction)
    if redaction is None:
        return None
    if redaction == "unredacted":
        data = _stored_keyframe_bytes(image_set, keyframe_id)
    else:
        data, _, _ = keyframe_image_bytes(
            store, world_id, session_id, keyframe_id, None, None,
            keyframes_are_redacted=True, image_set=image_set,
            imagery_source=IMAGERY_REDACTED,
        )
    if data is None:
        return None
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
    if image is None:
        return None
    height, width = image.shape[:2]
    scale = min(1.0, MAX_WIDTH / width, MAX_HEIGHT / height)
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(width * scale)),
                                    max(1, int(height * scale))),
                           interpolation=cv2.INTER_AREA)
    for quality in (60, 50, 40, 30, 20):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(encoded) <= MAX_BYTES:
            return encoded.tobytes()
    logger.warning("world guidance imagery: thumbnail cannot fit 16 KiB for %s", keyframe_id)
    return None


class ImageryState:
    """Per-root memory only. A manifest replacement also replaces its tile index."""

    def __init__(self):
        self._lock = threading.Lock()
        self._live_build_lock = threading.Lock()
        self._live = {}
        self._landed = {}
        self._landed_pending = {}

    def live(self, root, world_id, session_id):
        store = WorldStore(root)
        if (contained_world_id(store, world_id) != world_id or
                contained_session_id(store, world_id, session_id) != session_id):
            return None
        try:
            session = store.read_session(world_id, session_id)
            if session.world_id != world_id:
                return None
            keyframes = store.read_keyframes(world_id, session_id)
            segment = keyframes[-1].segment_index if keyframes else 0
            image_set = store.keyframe_image_set(world_id, session_id)
            derived = store.read_derived(world_id, session_id, verify=False)
            posed = {p["keyframe_id"] for p in (derived or {}).get("poses", [])
                     if p.get("segment_index") == segment and
                     p.get("status") in ("solved", "anchor")}
        except (OSError, ValueError, KeyError, TypeError, WorldStoreError):
            return None
        with self._live_build_lock:
            return self._live_from_snapshot(store, world_id, session_id, keyframes,
                                            segment, image_set, posed)

    def _live_from_snapshot(self, store, world_id, session_id, keyframes,
                            segment, image_set, posed):
        pair = (world_id, session_id)
        # The journal declares a break before the next local rebuild lands.
        with self._lock:
            previous = self._live.get(pair)
            if previous and previous[0] > segment:
                return {"segment_index": previous[0], "entries": previous[1]}
            if previous and previous[0] != segment:
                # A break clears pictures, not the one-new-tile-per-second budget.
                self._live[pair] = (segment, [], {}, previous[3])
                return {"segment_index": segment, "entries": []}
            old_tiles = previous[2] if previous else {}
            last_new = previous[3] if previous else float("-inf")
        redaction = entry_redaction(image_set.redaction)
        if redaction is None:
            # Untrusted (and not a served `none` set): no tiles, because each
            # `_tile` read refuses; the entries keep the "redacted" label.
            redaction = "redacted"
        entries, tiles = [], {}
        now = time.monotonic()
        for row in reversed(keyframes):
            if row.segment_index != segment or len(entries) >= MAX_TILES:
                break
            kid = row.keyframe_id
            if kid not in posed:
                continue
            # At most one newly encoded tile each second; reuse existing bytes.
            # Bytes are reused only under the same redaction word, so a label
            # change never relabels pixels read under the other one.
            prior_digest = next((e["digest"] for e in previous[1]
                                 if e["keyframe_id"] == kid and
                                 e.get("redaction") == redaction), None) if previous else None
            old = old_tiles.get(prior_digest)
            if old is None:
                if now - last_new < 1.0:
                    continue
                started = time.monotonic()
                try:
                    old = _tile(store, world_id, session_id, kid, image_set)
                except Exception:
                    logger.exception("world guidance imagery: live tile omitted after read failure")
                    continue
                if old is None:
                    continue
                last_new = now
                if (time.monotonic() - started) > .005:
                    logger.warning("world guidance imagery: live tile exceeded 5 ms")
            digest = hashlib.sha256(old).hexdigest()
            entries.append({"keyframe_id": kid, "digest": digest,
                            "redaction": redaction, "pose_source": "segment_local",
                            "component_reference_segment": None, "segment_index": segment})
            tiles[digest] = old
        with self._lock:
            self._live[pair] = (segment, entries, tiles, last_new)
        return {"segment_index": segment, "entries": entries}

    def publish_landed(self, root, world_id, session_id, coverage, eligible,
                       cancelled=None, dial=None):
        """Called after coverage publication, within an 80 ms imagery budget.

        `dial` is coverage's `dial_out` for this receipt; the coverage worker
        always supplies it. With it, the landed block carries one tile per set
        sector of the hero component plus `latest` and `motion_sector`.
        Without it (a direct caller), tiles are spread across the accepted
        solve horizon by stride.

        Returns LANDED_PUBLISHED, LANDED_OVER_BUDGET (the CPU budget or the
        wall cap was exceeded; nothing published, the previous block stays) or
        LANDED_SKIPPED (no tile, untrusted set, or cancelled)."""
        started = time.monotonic()
        cpu_started = time.thread_time()

        def over_budget():
            return (time.thread_time() - cpu_started >= LANDED_CPU_BUDGET_S or
                    time.monotonic() - started >= LANDED_WALL_CAP_S)

        store = WorldStore(root)
        image_set = store.keyframe_image_set(world_id, session_id)
        redaction = entry_redaction(image_set.redaction)
        if redaction is None:
            return LANDED_SKIPPED
        if dial is None:
            # Spread thumbnails across the accepted solve horizon, deterministically.
            stride = max(1, (len(eligible) + MAX_TILES - 1) // MAX_TILES)
            slots = [(ref, None, (kid,)) for kid, ref in eligible[::stride]]
        else:
            # One slot per set sector, candidates newest first.
            slots = [(dial["reference_segment"], sector, kids[:DIAL_CANDIDATES_PER_SECTOR])
                     for sector, kids in dial.get("sectors", ())][:DIAL_MAX_TILES]
        entries, tiles = [], {}
        exceeded = False
        for ref, sector, kids in slots:
            if len(entries) >= MAX_TILES:
                break
            if over_budget():
                exceeded = True
                break
            for kid in kids:
                if over_budget():
                    exceeded = True
                    break
                blob = _tile(store, world_id, session_id, kid, image_set)
                if blob is None:
                    continue
                digest = hashlib.sha256(blob).hexdigest()
                entry = {"keyframe_id": kid, "digest": digest,
                         "redaction": redaction, "pose_source": "landed_global_solve",
                         "component_reference_segment": ref, "segment_index": None}
                if sector is not None:
                    entry["sector"] = sector
                entries.append(entry)
                tiles[digest] = blob
                break
        if exceeded or over_budget():
            logger.warning("world guidance imagery: landing omitted after budget "
                           "(cpu_ms=%.1f wall_ms=%.1f)",
                           (time.thread_time() - cpu_started) * 1000,
                           (time.monotonic() - started) * 1000)
            return LANDED_OVER_BUDGET
        if not entries:
            logger.warning("world guidance imagery: landing omitted, no tile")
            return LANDED_SKIPPED
        block = {"solved_at": coverage["solved_at"],
                 "geometry_revision": coverage["geometry_revision"],
                 "frame_revision": coverage["frame_revision"], "entries": entries}
        if dial is not None:
            block["latest"] = dict(dial["latest"]) if dial.get("latest") else None
            block["motion_sector"] = dial.get("motion_sector")
        with self._lock:
            if cancelled is not None and cancelled.is_set():
                return LANDED_SKIPPED
            if over_budget():
                logger.warning("world guidance imagery: landing publication exceeded its budget")
                return LANDED_OVER_BUDGET
            self._landed[(world_id, session_id)] = (block, tiles)
        return LANDED_PUBLISHED

    def begin_landed(self, world_id, session_id):
        """A coverage receipt is about to publish; expose its pending tile job."""
        with self._lock:
            self._landed_pending[(world_id, session_id)] = threading.Event()

    def finish_landed(self, world_id, session_id):
        with self._lock:
            event = self._landed_pending.pop((world_id, session_id), None)
        if event is not None:
            event.set()

    def landed(self, world_id, session_id, *, wait=False):
        # A manifest fetched immediately after the next status push may race
        # the 80 ms tile job. Only this imagery request waits, never status.
        with self._lock:
            event = self._landed_pending.get((world_id, session_id))
        if wait and event is not None:
            event.wait(.080)
        with self._lock:
            row = self._landed.get((world_id, session_id))
            return row[0] if row else None

    def tile(self, world_id, session_id, digest, live, landed):
        with self._lock:
            pair = (world_id, session_id)
            sources = []
            if live and pair in self._live:
                sources.append(self._live[pair][2])
            if landed and pair in self._landed:
                sources.append(self._landed[pair][1])
            for source in sources:
                blob = source.get(digest)
                if blob is not None and hashlib.sha256(blob).hexdigest() == digest:
                    return blob
        return None


_states = {}
_states_lock = threading.Lock()


def state_for(root):
    key = str(root)
    with _states_lock:
        return _states.setdefault(key, ImageryState())
