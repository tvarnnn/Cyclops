"""Geometry transport adapter for World Builder.

One of the few files permitted to import `world_builder`, and named after
its cartridge for the same reason `tower/results/world_builder.py` is: an
adapter named after one cartridge cannot leak that cartridge's
assumptions into the next, because the next one gets its own file.

Why geometry does not travel on the result channel: `tower/routes/ws.py`
gives the result sender and the frame path a single `asyncio.Lock`, and
one real session's `points.json` is 1.07 MB against a 3,884-byte status
snapshot. Bulk data there would starve `frame_result`.

Why the segment is the unit: the engine (`WorldBuilderEngine`) freezes a
segment when tracking is lost, so a closed segment is fetched once and
cached for the life of the world. Only the open segment churns.
"""

import hashlib
import json
import logging
from collections import Counter

from tower.world_builder.records import PLACEMENT_REGISTERED, World
from tower.world_builder.schema import POSE_CONVENTION
logger = logging.getLogger(__name__)

from tower.world_builder.store import (
    WorldStore,
    WorldStoreError,
    compute_input_digest,
    manifest_describing,
)

GEOMETRY_CONTRACT = "world_builder.geometry/2026-08-25"

# Statuses that are neither measured evidence nor a segment origin. Their
# degeneracy is the only place the reason for a refusal survives.
_REFUSED = ("unavailable", "rotation_only")


def segment_content_hash(poses: list[dict], points: list[dict]) -> str:
    """A stable, opaque identity for one segment's geometry.

    Truncated to 16 hex characters, matching `compute_revision`. This is a
    change detector, not a security primitive: the client compares it for
    equality and nothing else.
    """
    canonical = json.dumps(
        {"poses": poses, "points": points}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _bounds(points: list[dict]) -> dict | None:
    if not points:
        return None
    xyz = [p["xyz"] for p in points]
    return {
        "min": [min(v[i] for v in xyz) for i in range(3)],
        "max": [max(v[i] for v in xyz) for i in range(3)],
    }


def _dominant_degeneracy(poses: list[dict]) -> str | None:
    reasons = Counter(
        p["degeneracy"] for p in poses
        if p.get("status") in _REFUSED and p.get("degeneracy")
    )
    if not reasons:
        return None
    return reasons.most_common(1)[0][0]


def _grouped(derived: dict) -> dict[int, dict]:
    """Split the session's derived rows into per-segment buckets.

    Both files already carry `segment_index`, because segments share
    neither a coordinate frame nor a unit and an untagged row could not be
    placed.
    """
    segments: dict[int, dict] = {}
    for pose in derived["poses"]:
        segments.setdefault(
            pose["segment_index"], {"poses": [], "points": []}
        )["poses"].append(pose)
    for point in derived["points"]:
        segments.setdefault(
            point["segment_index"], {"poses": [], "points": []}
        )["points"].append(point)
    return segments


def _revision_over(hashes: list[str]) -> str:
    """A rollup identity for the whole session's geometry.

    Separate from `segment_content_hash` on purpose: that function's input
    IS a segment, and reusing it here would hash fabricated rows and claim
    they were geometry.
    """
    canonical = json.dumps(hashes, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def store_from_root(root) -> WorldStore:
    """Construct a `WorldStore` for a configured world root.

    Lives here, not in `tower/routes/geometry.py`, so the route imports
    only this adapter and never reaches `tower.world_builder` directly --
    the same rule that keeps `tower/results/world_builder.py` as the sole
    place outside this file that knows a `WorldStore` exists.
    """
    return WorldStore(root)


def _is_current(store, world_id: str, session_id: str) -> bool:
    """Whether the derived tree reflects every keyframe accepted so far.

    False is NORMAL during a walk: the digest moves with every keyframe, so
    a build finishes and the next keyframe puts it behind.
    """
    try:
        digest = compute_input_digest(store.read_keyframes(world_id, session_id))
    except (WorldStoreError, KeyError, ValueError, OSError):
        return False
    # WITH the session id: without it the store asks the world's manifest,
    # which names whichever session built last, and an older session of a
    # world walked twice would be judged against another session's digest.
    #
    # `derived_currency` can also answer None, "nothing here can judge it".
    # The wire contract defines `current` as "reflects every keyframe
    # accepted so far", which is a claim, so an unjudgeable tree reports
    # FALSE -- the same answer the status channel gives, and the
    # conservative one. `read_derived` treats None differently, because it
    # is deciding whether to serve rather than what to claim.
    return store.derived_currency(world_id, digest, session_id) is True


def contained_world_id(store, world_id: str) -> str | None:
    """This world's id as the store spells it, or None if it escapes.

    Both geometry routes take `world_id` from an unauthenticated request,
    and both reach the store through a path join. `world_id` is a PATH
    parameter, so Starlette's `[^/]+` excludes a forward slash -- and on
    Windows, the only platform this Tower ships on, a BACKSLASH is equally
    a separator and is not excluded. `%5C` is decoded into the path before
    routing, so it matches the route too.

    CONTAINMENT FIRST, THEN CANONICAL, and the order is the whole
    correctness of this function. Canonicalising first would turn
    `..\\..\\elsewhere\\worlds\\secret` into `secret` and then happily
    serve a LOCAL world of that name -- contained, but not the world that
    was asked for, and silently so.

    Canonical second, because `junk\\..\\<real>` genuinely names a world
    inside the root and is admitted -- but the payload must answer under
    the world's own id, or two requests for one world reply with two
    different identities and a client caching on `world_id` holds both.
    It also stops the caller choosing how long that identity is.

    This is O(1). A whitelist over `list_world_ids()` would give the same
    two properties by construction but is a directory scan per request,
    growing with a directory that only grows: measured at 4.16 ms against
    120 real worlds (69% of `_read`), 17.4 ms at 500 and 167.7 ms at
    2,000.
    """
    worlds_root = store.world_dir("probe").parent
    try:
        if store.world_dir(world_id).parent.resolve() != worlds_root.resolve():
            return None
    except (OSError, ValueError):
        # A path the OS will not parse -- a NUL byte, an over-long name.
        return None
    return store.world_dir(world_id).name


def _read(store, world_id: str, session_id: str):
    """Return `(world, derived, grouped, current)` or `None` if absent.

    The digest is deliberately NOT verified on the read, and currency is
    computed alongside instead. `read_derived`'s default treats a tree that
    no longer matches the journal as absent, which is the wrong trade on
    this channel: during a live walk the digest moves with every keyframe,
    so every manifest request would answer 404 and the fragment gallery
    would stay empty until the session ended, with real geometry on disk
    the whole time.

    The status channel (`tower/results/world_builder.py`, which asks
    `store.derived_currency`) makes the same decision for the same reason:
    a build over the first N keyframes is not wrong, it is a correct answer
    to an older question. Hiding it discards true information; serving it
    unflagged would let a viewer mistake it for the finished world. The
    `current` flag is the whole difference, and it rides on the manifest
    AND on every chunk, so a client holding only a chunk still knows.

    Absent stays distinguishable from behind: a tree with no poses.json is
    still `None`, and the routes still answer 404 for it.
    """
    # BOTH identifiers are unauthenticated and BOTH escape, and neither
    # check below can stand in for the other. The session check is anchored
    # on `derived_dir(world_id)`, so an escaped `world_id` moves the anchor
    # and the session check still passes relative to the escaped base,
    # serving a complete world from outside the configured root.
    #
    # `world_id` is a PATH parameter, so Starlette's `[^/]+` excludes a
    # forward slash -- and on Windows, the only platform this Tower ships
    # on, a BACKSLASH is equally a separator and is not excluded. `%5C` is
    # decoded into the path before routing, so it matches too. (`%2F` does
    # not match the route at all and is not a vector.) An absolute
    # `world_id` (`C:\...`) escapes the same way.
    #
    # The world gets the SAME direct-child rule as the session check below,
    # against the worlds directory. `world_dir()` is
    # `<root>/worlds/<world_id>`, so `worlds/` is fixed and the caller does
    # not control it; what moves with `world_id` is the base for the
    # SESSION check, which is why that one cannot catch this and both are
    # needed. The rule is O(1); `contained_world_id` gives the cost of the
    # whitelist alternative.
    #
    # `read_world` still requires a real `world.json` immediately after.
    #
    # The base is asked of the store with a separator-free probe rather
    # than by writing "worlds" here, so this stays ignorant of the layout
    # `WorldStore` owns.
    if contained_world_id(store, world_id) is None:
        return None
    try:
        world: World = store.read_world(world_id)
    except WorldStoreError:
        return None
    # `session_id` arrives from an unauthenticated QUERY parameter on both
    # geometry routes -- declared as a bare `str`, so unlike a path
    # parameter it is not restricted to `[^/]+` -- and it reaches
    # `read_derived` as `derived_dir(world_id) / session_id`, an unguarded
    # join. Unchecked, a relative traversal (`../../../../elsewhere`) or an
    # absolute path (`C:/elsewhere`) answers 200 with poses and points read
    # from OUTSIDE the world root, under this world's id.
    #
    # NOT a `..` scan: an absolute path does not traverse out of the base,
    # it REPLACES it (`Path("/a/b") / "C:/elsewhere"` is `C:/elsewhere`),
    # so a scan for `..` misses the more dangerous half entirely.
    #
    # A DIRECT-CHILD check rather than a whitelist against
    # `list_session_ids`: `_tiny_world` (in
    # `test_world_builder_placements.py`) writes a world and a derived tree
    # without a session record, so a legitimate derived tree can exist for
    # a session `list_session_ids` does not return (a whitelist failed 11
    # tests there). Requiring the session record would make a containment
    # fix into a behaviour change.
    #
    # `parent == base` is also the TIGHTER rule, not merely the cheaper
    # one. A session id names one directory, so anything with a separator
    # in it is already wrong; and basing containment on THIS world's
    # derived directory means a sibling world's tree fails the same check
    # -- `../../w_other/derived/s0` leaves `w0/derived` exactly as an
    # absolute path does. One rule covers traversal, absolute replacement
    # and cross-world reads together.
    derived_root = store.derived_dir(world_id)
    try:
        # `.parent.resolve()`, NOT `.resolve().parent`. Resolving the whole
        # path follows the LEAF, so a session directory that is a junction
        # or symlink (`mklink /J`) resolves to its target, compares unequal
        # against the derived root, and refuses real geometry. Resolving
        # the parent is exactly as tight, because a value that escapes
        # moves the parent too: `..\..\x` has parent `<derived>\..\..`, and
        # an absolute value has its own root.
        if (derived_root / session_id).parent.resolve() != derived_root.resolve():
            return None
    except (OSError, ValueError):
        # A path the OS will not even parse -- a NUL byte, an over-long
        # name. Absent, like any other unreadable tree.
        return None
    derived = store.read_derived(world_id, session_id, verify=False)
    if derived is None:
        return None
    current = _is_current(store, world_id, session_id)
    return world, derived, _grouped(derived), current


def usable_placements(store, world_id: str, session_id: str) -> dict:
    """Placements that may be served, keyed by segment index.

    Two ways a stored placement is refused here rather than shipped.

    **It was solved against different geometry.** `write_derived` rewrites
    poses and points wholesale and never touches placements.json, so a
    Sim3 outlives the reconstruction it was fitted to. Without this check
    it is served against points that no longer exist, still flagged
    registered -- and the cache guarantee does not help, because
    content_hash moves, the client refetches exactly as it should, and is
    handed the stale transform. A placement whose `input_digest` does not
    match the build being served is not current evidence, and a placement
    with no digest cannot be checked at all.

    **Its reference is not itself placed.** The contract says segments
    sharing a `reference_segment` are in one space and may be drawn
    together. A registered row pointing at a segment that is absent,
    dropped as unrepresentable, or itself refused invites a client to
    composite geometry into a frame nothing defines. A cluster missing its
    origin is worse than no cluster.
    """
    stored = store.read_placements(world_id, session_id) or []
    if not stored:
        return {}

    # THE SESSION'S OWN MANIFEST. The world's names whichever session built
    # last, so every placement of an OLDER session would be compared
    # against another session's digest, fail, and be served as unplaced --
    # an earlier walk could not be composited at all.
    manifest = _session_manifest(store, world_id, session_id) or {}
    digest = manifest.get("input_digest")
    fresh = {}
    for placement in stored:
        if digest is None or placement.input_digest != digest:
            logger.warning(
                "world builder: placement for segment %s was solved against "
                "a different build (%r, current %r); serving it as unplaced",
                placement.segment_index,
                placement.input_digest,
                digest,
            )
            continue
        fresh[placement.segment_index] = placement

    registered = {
        index for index, p in fresh.items() if p.state == PLACEMENT_REGISTERED
    }
    usable = {}
    for index, placement in fresh.items():
        if (
            placement.state == PLACEMENT_REGISTERED
            and placement.reference_segment not in registered
        ):
            logger.warning(
                "world builder: segment %s is placed against reference %s, "
                "which is not itself placed; refusing the placement rather "
                "than inviting a composite into an undefined frame",
                index,
                placement.reference_segment,
            )
            continue
        usable[index] = placement
    return usable


def placement_hash(placement) -> str:
    """A hash over WHERE a segment sits, separate from what it contains.

    `content_hash` covers poses and points only, deliberately, so a
    segment that gains a placement keeps its content hash. That is a
    feature only if something else carries the placement -- otherwise a
    client holding a cached chunk never refetches and draws an unplaced
    version of a segment the world now knows how to place, and no test
    catches it because the fields it would check are constants.

    This is that something else. A client keys its cache on
    (content_hash, placement_hash).
    """
    if placement is None:
        payload = {"state": "unplaced"}
    else:
        payload = {
            "state": placement.state,
            "rotation_wxyz": (
                list(placement.rotation_wxyz)
                if placement.rotation_wxyz
                else None
            ),
            "translation": (
                list(placement.translation) if placement.translation else None
            ),
            "scale": placement.scale,
            "reference_segment": placement.reference_segment,
            "frame_revision": placement.frame_revision,
            # Served, therefore hashed. Left out, a re-registration that
            # changed only the refusal reasons would move no hash a client
            # is told to key on, and a conforming client would show stale
            # reason text forever; on the real corpus 26 of 29 segments are
            # refused and would share ONE placement_hash.
            "refusal_reason": placement.refusal_reason,
        }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _placement_fields(placement) -> dict:
    """The wire view of a placement.

    `registered` stays a plain bool so an older decoder keeps working.
    `registration_state` is the field that can actually tell a refusal
    from an untried segment, which `registered: false` alone could not --
    and on the real corpus the refusal is usually "the wearer stood
    still", a message about how to walk rather than about the software.
    """
    if placement is None:
        return {
            "registered": False,
            "registration_state": "unplaced",
            "transform_to_world": None,
            "registration_refusal_reason": None,
        }
    if placement.state == "registered":
        return {
            "registered": True,
            "registration_state": "registered",
            "transform_to_world": {
                "rotation_wxyz": list(placement.rotation_wxyz),
                "translation": list(placement.translation),
                "scale": placement.scale,
                "reference_segment": placement.reference_segment,
                "frame_revision": placement.frame_revision,
            },
            "registration_refusal_reason": None,
        }
    return {
        "registered": False,
        "registration_state": "refused",
        "transform_to_world": None,
        "registration_refusal_reason": placement.refusal_reason,
    }


def _session_manifest(store, world_id: str, session_id: str) -> dict | None:
    """The manifest that describes THIS session, from either copy.

    The session's own first -- `write_derived` writes one beside the poses
    and points it describes -- then the world's, but only if it names this
    session. A world manifest naming another session is not evidence about
    this one, and treating it as such would serve an older walk another
    session's coverage classes and refuse every one of its placements.
    """
    # `purpose="identity"`: this module's callers ask WHICH BUILD produced
    # a tree -- `usable_placements` compares `input_digest` -- and a
    # manifest carrying a session id and a digest answers that completely.
    # See `manifest_describing` for why neither the figures nor the SCHEMA
    # belongs on this path.
    return manifest_describing(store, world_id, session_id, purpose="identity")


def manifest_for(store, world_id: str, session_id: str | None = None) -> dict | None:
    """The manifest to judge a session by, or the world's when none is named.

    `session_id` is defaulted only so callers written before per-session
    manifests existed keep working; every caller inside this module passes
    it. Without one this answers about whichever session built last, which
    for `global_solve.segments` means serving one walk's coverage classes
    as another's.
    """
    if session_id is not None:
        return _session_manifest(store, world_id, session_id)
    try:
        return store.read_derived_manifest(world_id)
    except Exception:  # noqa: BLE001 -- an unreadable manifest is "no judgement"
        return None


def _solve_segments(store, world_id: str, session_id: str) -> dict:
    """The global solve's per-segment verdicts, or `{}`.

    **A `.get` CHAIN IS NOT A GUARD.** `(manifest or {}).get("global_solve")
    or {}).get("segments") or {}` raises `AttributeError` the moment
    `global_solve` is a string or a list rather than a dict, and again if
    `segments` is -- straight out of `build_manifest` and into an HTTP
    **500** on a route whose 404 means ABSENT only, and where the phone has
    no branch for a 500. So the lookup is type-checked: a malformed
    manifest degrades to `{}`. `validate_manifest` checks that required
    keys are present, not what shape `global_solve` or `segments` has.

    Four malformed shapes of `global_solve`/`segments` measured 200 -> 500
    on a manifest read without those checks. A manifest from an unknown
    schema (whose fields `validate_manifest` says "this build does not
    know") is refused by the figures read; the checks cover a manifest of
    the current schema whose `global_solve` is malformed.

    The manifest is read with `purpose="figures"`, which holds it to the
    schema and the required keys, because coverage verdicts are figures.
    The identity path (`usable_placements`) deliberately does not: it
    needs only the digest, and refusing 408 real placements across 10
    sessions over a record schema bump is the defect that split prevents
    (see `manifest_describing`). A FIGURES-shaped field must not be read
    off the identity path without checking its shape.
    """
    manifest = manifest_describing(store, world_id, session_id, purpose="figures")
    if not isinstance(manifest, dict):
        return {}
    solve = manifest.get("global_solve")
    if not isinstance(solve, dict):
        return {}
    segments = solve.get("segments")
    return segments if isinstance(segments, dict) else {}


def coverage_for_segment(index: int, poses: list, points: list, solve_segments: dict) -> str | None:
    """The global solve's coverage class for a segment, or a truthful
    fallback: `unresolved` for a segment with keyframes and no points, and
    null when nothing has judged a segment that does have points."""
    judged = solve_segments.get(str(index)) or solve_segments.get(index)
    # `isinstance`, not truthiness: a `segments` map whose VALUES are
    # strings would pass an `if judged` and raise `AttributeError` on the
    # `.get` below. Same class as `_solve_segments`, one level deeper.
    if isinstance(judged, dict) and judged.get("coverage") in (
        "confident", "partial", "unresolved"
    ):
        return judged["coverage"]
    if poses and not points:
        return "unresolved"
    return None


def build_manifest(store, world_id: str, session_id: str) -> dict | None:
    """The geometry manifest for one session: what segments exist and how
    to fetch and trust each (`GET /worlds/{id}/geometry/manifest`).

    Inputs are a `WorldStore`, the requested `world_id` and a `session_id`;
    both ids come from an unauthenticated request. The world id is first
    contained and canonicalised (`contained_world_id`), then `_read` loads
    the session's derived poses and points grouped by `segment_index`
    together with the `current` flag; `usable_placements` and the global
    solve's per-segment verdicts (`_solve_segments`) are read beside them.
    Each segment, in index order, yields a row: `content_hash` (poses and
    points only), `frame_id`, the placement fields and `placement_hash`,
    `resolution_state`, `dominant_degeneracy`, keyframe, solved and point
    counts, `bounds`, and `coverage`. The result adds `contract`, the
    canonical `world_id`, `session_id`, `current`, `geometry_revision`,
    `pose_convention`, `scale` and `segment_count`. Returns `None` when
    the world or the session's derived tree is absent; the route answers
    404 for that and for nothing else.

    Why it is shaped this way: a segment's placement is hashed apart from
    its content, so `geometry_revision` rolls up BOTH hash lists -- a
    placement-only change moves no content hash and would otherwise give
    a polling client no signal. `current` is False for geometry that is
    real but behind the newest keyframes (the normal state during a walk),
    where hiding it would starve the fragment gallery. The payload carries
    the canonical world id, never the caller's spelling. Wire shapes and
    hash definitions are pinned by `docs/contracts/WORLD-BUILDER-GEOMETRY.md`.
    """
    # Rebound BEFORE anything reads it, because this function builds the
    # payload from its own copy: canonicalising inside `_read` would change
    # only a local and leave `world_id` in the reply exactly as the caller
    # spelled it.
    world_id = contained_world_id(store, world_id)
    if world_id is None:
        return None
    read = _read(store, world_id, session_id)
    if read is None:
        return None
    world, _, grouped, current = read
    placements = usable_placements(store, world_id, session_id)

    segments = []
    solve_segments = _solve_segments(store, world_id, session_id)
    for index in sorted(grouped):
        poses = grouped[index]["poses"]
        points = grouped[index]["points"]
        segments.append({
            "segment_index": index,
            "content_hash": segment_content_hash(poses, points),
            "frame_id": f"segment:{index}",
            # Placement is separate from content on purpose: a segment's
            # own geometry does not move when it is placed, so every
            # cached content_hash stays valid -- which is safe only
            # because placement_hash exists to change instead.
            **_placement_fields(placements.get(index)),
            "placement_hash": placement_hash(placements.get(index)),
            "resolution_state": "resolved" if points else "unresolved",
            "dominant_degeneracy": _dominant_degeneracy(poses),
            "keyframe_count": len(poses),
            "solved_count": sum(1 for p in poses if p.get("status") == "solved"),
            "point_count": len(points),
            "bounds": _bounds(points),
            # How much to trust this segment's geometry, from the global
            # solve when it ran: confident | partial | unresolved, or null
            # when no solve has judged it. Unresolved is "keyframes exist,
            # no geometry"; unseen space has no segment at all and is
            # never drawn.
            "coverage": coverage_for_segment(index, poses, points, solve_segments),
        })

    return {
        "contract": GEOMETRY_CONTRACT,
        "world_id": world_id,
        "session_id": session_id,
        # Additive, deliberately not a contract bump: CARTRIDGE-RESULTS
        # section 12 says a field an older decoder ignores is not grounds
        # for one. False means "this geometry is real, and behind the
        # newest keyframes" -- the normal state during a walk.
        "current": current,
        # Over BOTH hashes. A placement-only change moves no content
        # hash, so a rollup over content alone would leave a client with
        # no signal that anything had moved.
        "geometry_revision": _revision_over(
            [s["content_hash"] for s in segments]
            + [s["placement_hash"] for s in segments]
        ),
        "pose_convention": dict(world.pose_convention or POSE_CONVENTION),
        "scale": {
            "state": world.scale.state,
            "meters_per_unit": world.scale.meters_per_unit,
        },
        "segment_count": len(segments),
        "segments": segments,
    }


def build_segment(
    store, world_id: str, session_id: str, segment_index: int,
    max_points: int | None = None,
) -> dict | None:
    """One segment's geometry, in its own frame
    (`GET /worlds/{id}/geometry/segment/{n}`).

    Inputs are a `WorldStore`, the requested `world_id`, a `session_id`, a
    `segment_index` and an optional `max_points`. The world id is
    contained and canonicalised, `_read` loads the session's derived tree
    grouped by segment together with the `current` flag, and
    `usable_placements` supplies this segment's placement. The result
    carries `contract`, `current`, `segment_index`, the segment's
    `content_hash`, `frame_id`, the placement fields and `placement_hash`,
    the poses (`keyframe_id`, `status`, `degeneracy`, `rotation`,
    `translation`) and the points as bare `xyz` lists, plus
    `points_sent`, `points_total` and `point_sampling` (`"none"` or
    `"stride"`). Returns `None` when the world, the session's tree or the
    segment is absent; `max_points < 1` raises `ValueError`.

    `max_points` exists so a budget lever is available without a contract
    change. It defaults to unlimited: the largest segment on the real walk
    is 3,033 points, and because a closed segment is fetched exactly once
    that is a one-time cost rather than a per-revision one. When it caps
    the cloud, points are taken evenly across the whole cloud, never as a
    prefix, and `content_hash` is taken over the whole segment before
    sampling, so it names the segment and not this transfer. `current` is
    repeated here from the manifest so a client holding only a cached
    chunk can still tell the geometry is behind the newest keyframes.
    """
    world_id = contained_world_id(store, world_id)
    if world_id is None:
        return None
    read = _read(store, world_id, session_id)
    if read is None:
        return None
    _, _, grouped, current = read
    placement = usable_placements(store, world_id, session_id).get(
        segment_index
    )
    if segment_index not in grouped:
        return None

    poses = grouped[segment_index]["poses"]
    points = grouped[segment_index]["points"]
    # Hash the WHOLE segment before sampling: the hash identifies the
    # segment, not this transfer, so a client that sampled once and
    # refetches in full must not see a changed identity.
    content_hash = segment_content_hash(poses, points)

    total = len(points)
    sampling = "none"
    sent = points
    if max_points is not None:
        if max_points < 1:
            raise ValueError("max_points must be at least 1")
        if total > max_points:
            # Evenly spaced across the WHOLE cloud, never a prefix: a prefix
            # is one corner of the room and would read as a smaller world
            # rather than a coarser one.
            #
            # An INTEGER stride cannot do this. `total // max_points`
            # collapses to 1 whenever max_points > total/2 -- so capping
            # 3,033 points at 2,000 would return points[0:2000], a prefix.
            # Hence the float step.
            step = total / max_points
            sent = [points[int(i * step)] for i in range(max_points)]
            sampling = "stride"

    return {
        "contract": GEOMETRY_CONTRACT,
        # Repeated from the manifest on purpose: a client that fetched one
        # chunk from cache and never re-read the manifest would otherwise
        # have no way to know the geometry in its hand is behind.
        "current": current,
        "segment_index": segment_index,
        "content_hash": content_hash,
        "frame_id": f"segment:{segment_index}",
        **_placement_fields(placement),
        "placement_hash": placement_hash(placement),
        "poses": [
            {
                "keyframe_id": p["keyframe_id"],
                "status": p["status"],
                "degeneracy": p["degeneracy"],
                "rotation": p["rotation"],
                "translation": p["translation"],
            }
            for p in poses
        ],
        "points": [p["xyz"] for p in sent],
        "points_sent": len(sent),
        "points_total": total,
        "point_sampling": sampling,
    }
