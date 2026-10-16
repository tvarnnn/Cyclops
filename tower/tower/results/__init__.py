"""The Tower -> iOS structured cartridge result channel.

A READ-ONLY reporting surface over state that has already been produced
elsewhere. Nothing in this package runs a cartridge, drives a module,
touches a frame, or writes anything. It answers one question -- "what does
the Tower currently know about cartridge X?" -- from files on disk, or
from a live session object it is handed and may only read.

That restraint is the architecture, not an accident of scope. The obvious
alternative was to register World Builder as a live production module and
publish its results as they were computed; four blockers stop that, and
every one of them is real. But treating them as blocking any World
Builder result at all conflates two things that turn out to be separable:

    a World Builder TRANSPORT        <- achievable now, this package
    World Builder as a LIVE MODULE   <- blocked at V1.0/V1.1

The blockers are all properties of the second. `process()` being
synchronous and `bytes`-only, `ExperimentResult` being scalar-shaped,
`_build_cv_module` being a registry of one, `LIFECYCLE_TIMEOUT_S` being
too short for a build -- none of them bear on a reader that never joins
the frame path at all.

**Scene Understanding does not fit the file-reading model above.** It
persists nothing by design -- enforced, not intended -- so there is no
file for a reader to read, and the journal-follower pattern that gives
Object Memory a route is unavailable to it. It is served from a handed-in
session object instead, and the separation that matters holds: the LIVE
part is `tower/scene/live.py`, which owns a worker
thread and a model and is constructed by `tower/cartridge_runtime.py`;
this package is handed the resulting session object and may only call
`status()` and `latest()` on it. `test_the_result_channel_never_writes`
forbids a call named `observe` or `build` anywhere under `tower/results/`
and so mechanically prevents that handle from becoming a second frame
path.
"""

import time

from tower.results.contracts import (
    CARTRIDGE_DOCUMENT_MEMORY,
    CARTRIDGE_EXPERIMENTAL_CV,
    CARTRIDGE_SCENE_UNDERSTANDING,
    CARTRIDGE_WORLD_BUILDER,
    EXPERIMENTAL_CV_STATUS_CONTRACT,
    RESULT_TYPE_LIVE,
    RESULT_TYPE_STATUS,
    WORLD_BUILDER_STATUS_CONTRACT,
)
from tower.results.envelope import Snapshot, compute_revision
from tower.results.publisher import ResultHub


def make_snapshot_for(
    world_root,
    clock=time.time,
    *,
    document_root=None,
    scene_source=None,
    document_source=None,
    cv_lab=None,
    document_unavailable_reason=None,
):
    """Build the callable that turns (cartridge, result_type, world, session)
    into a Snapshot by choosing the cartridge's producer.

    HOW: the returned `snapshot_for(cartridge, result_type, world_id,
    session_id)` has one arm per wired (cartridge, result_type) pair --
    CV Lab status, World Builder status, Scene Understanding live,
    Document Memory status. Each arm returns a `Snapshot` (payload plus
    revision, plus `volatile_fields` for the live and document arms), or
    an explicit `unavailable` snapshot when its required input is missing
    (`world_root`, `scene_source`, `document_root`; a missing `cv_lab` gives
    the CV Lab's own status payload saying it is unavailable); an unwired
    pair also yields `unavailable`. The World Builder producer is built
    here when `world_root` is given, because it holds a small per-target
    cache that a per-poll producer would discard; the CV Lab and Document
    Memory producers are built on first poll and reused; Scene
    Understanding builds its payload directly from `scene_source` on each
    poll. Document Memory needs only `document_root`: without
    `document_source` it still reports the library's status. The
    callable also carries a `shutdown_guidance` attribute that forwards to
    the World Builder producer's own `shutdown_guidance`, if one exists.

    WHY: it is built once and handed to the hub, so the hub itself imports
    no cartridge and can be tested against a stub.

    `scene_source`, `document_source` and `cv_lab` are LIVE SESSION
    OBJECTS rather than roots: those cartridges report on work happening
    in this process, not on a file another process wrote. They are passed
    in rather than constructed here so this module imports nothing that
    owns a thread or a model, and a test can drive the whole channel with
    a stub that has never seen torch. They are read-only handles; see
    `tower/results/experimental_cv.py` for the CV Lab concurrency.

    `world_root` may be None, and `document_root` and the live sources
    default to None; a missing required input produces an explicit
    `unavailable` answer rather than an exception (a missing
    `document_source` alone still reports the library status) -- the
    same under-promise-on-omission rule `registry.declare` follows.
    """
    producers: dict = {}
    if world_root is not None:
        # Built here, not on first poll: its constructor starts the coverage
        # worker thread, and a status poll must never start a thread.
        from tower.results.world_builder import WorldBuilderStatusProducer
        producers[CARTRIDGE_WORLD_BUILDER] = WorldBuilderStatusProducer(
            world_root, clock)

    def snapshot_for(cartridge, result_type, world_id, session_id) -> Snapshot:
        if (
            cartridge == CARTRIDGE_EXPERIMENTAL_CV
            and result_type == RESULT_TYPE_STATUS
        ):
            if cv_lab is None:
                # Function-local import, the same pattern the World
                # Builder producer uses above: `tower/results/__init__.py`
                # is in the cartridge-blind core list, and a module-level
                # import would put a cartridge in the shared surface.
                from tower.results.experimental_cv import unavailable_payload

                payload = unavailable_payload(
                    "this Tower is running without a CV Lab module"
                )
                return Snapshot(payload=payload, revision=compute_revision(payload))
            producer = producers.get(cartridge)
            if producer is None:
                from tower.results.experimental_cv import (
                    ExperimentalCVStatusProducer,
                )

                producer = ExperimentalCVStatusProducer(cv_lab)
                producers[cartridge] = producer
            return producer.snapshot()
        if cartridge == CARTRIDGE_WORLD_BUILDER and result_type == RESULT_TYPE_STATUS:
            if world_root is None:
                return _unavailable_snapshot(
                    "no world root is configured on this Tower"
                )
            return producers[cartridge].snapshot(world_id, session_id)

        if (
            cartridge == CARTRIDGE_SCENE_UNDERSTANDING
            and result_type == RESULT_TYPE_LIVE
        ):
            if scene_source is None:
                return _unavailable_snapshot(
                    "Scene Understanding is not enabled on this Tower"
                )
            from tower.results.scene_understanding import VOLATILE_PATHS, live_payload

            state, _observed_at, _computed_at = scene_source.latest()
            payload = live_payload(scene_source.status(), state)
            return Snapshot(
                payload=payload,
                revision=compute_revision(payload, VOLATILE_PATHS),
                volatile_fields=VOLATILE_PATHS,
            )

        if (
            cartridge == CARTRIDGE_DOCUMENT_MEMORY
            and result_type == RESULT_TYPE_STATUS
        ):
            if document_root is None:
                return _unavailable_snapshot(
                    "no document root is configured on this Tower"
                )
            from tower.results.document_memory import (
                STATUS_VOLATILE_PATHS,
                DocumentStatusProducer,
            )

            producer = producers.get(cartridge)
            if producer is None:
                producer = DocumentStatusProducer(
                    document_root,
                    document_source,
                    clock=clock,
                    unavailable_reason=document_unavailable_reason,
                )
                producers[cartridge] = producer
            payload = producer.payload()
            return Snapshot(
                payload=payload,
                revision=compute_revision(payload, STATUS_VOLATILE_PATHS),
                volatile_fields=STATUS_VOLATILE_PATHS,
            )

        # Unreachable through the wire: registry.find_offer refuses an
        # unknown pair before a subscription is ever created. Present so
        # that a future producer added to the registry and forgotten here
        # fails as an explicit unavailable rather than a KeyError inside
        # the poll loop.
        return _unavailable_snapshot(
            f"no producer is wired for {cartridge}/{result_type}"
        )

    def shutdown_guidance():
        producer = producers.get(CARTRIDGE_WORLD_BUILDER)
        if producer is not None:
            producer.shutdown_guidance()

    snapshot_for.shutdown_guidance = shutdown_guidance
    return snapshot_for


def _unavailable_snapshot(reason: str) -> Snapshot:
    """The Tower cannot serve this cartridge at all.

    Distinct from "there is nothing to show yet": this is a Tower
    limitation, so it projects to iOS's `.unsupported`, which tells a
    person the Tower cannot do this rather than inviting them to wait.
    """
    payload = {
        "lifecycle": {
            "state": "unavailable",
            "evidence": "nothing to read",
            "reason": reason,
            "build_in_progress": None,
            "build_in_progress_unavailable_reason": reason,
        },
        "model_state": "unsupported",
        "model_state_reason": reason,
        "world_snapshot": None,
    }
    return Snapshot(payload=payload, revision=compute_revision(payload))


def build_hub(
    world_root,
    clock=time.time,
    *,
    document_root=None,
    scene_source=None,
    document_source=None,
    cv_lab=None,
    document_unavailable_reason=None,
) -> ResultHub:
    return ResultHub(
        make_snapshot_for(
            world_root,
            clock,
            document_root=document_root,
            scene_source=scene_source,
            document_source=document_source,
            cv_lab=cv_lab,
            document_unavailable_reason=document_unavailable_reason,
        ),
        clock=clock,
    )


__all__ = [
    "CARTRIDGE_EXPERIMENTAL_CV",
    "EXPERIMENTAL_CV_STATUS_CONTRACT",
    "CARTRIDGE_WORLD_BUILDER",
    "RESULT_TYPE_STATUS",
    "WORLD_BUILDER_STATUS_CONTRACT",
    "ResultHub",
    "build_hub",
    "make_snapshot_for",
]
