# Module Concept — Environmental Memory / Physical-World Search

## Status

**PLANNED.** No code exists under `tower/`. Longer-term module.

**Its neighbour now exists, and the boundary is sharp.** Scene
Understanding (`docs/modules/SCENE-UNDERSTANDING.md`, built 2026-08-22)
answers *"what is around me now"*; this module answers *"what did I
encounter, and when"*. One is a live state, the other a history.

Scene Understanding deliberately **persists nothing** -- no store, no
journal, no imagery, enforced by test -- precisely so that the decision
to keep a durable record of the physical world lands here, where the
retention, purge and privacy policy this document already demands can be
applied to it. Do not add a store to Scene Understanding; the day one is
wanted is the day this module starts.

What it leaves ready: anonymous tracking with counts that survive
detector dropout, camera-relative relationships, and a documented set of
relationships it REFUSES to assert with the evidence each would need. Broader than Object Memory and likely dependent on several mature perception services. This module has the highest privacy exposure of the current module set (see Privacy below) — it must not begin real data collection until the retention/deletion policy in `06-PRIVACY-DATA.md` is actually implemented for this module, not merely documented.

## Goal

Create a searchable memory of meaningful things the user has encountered in the physical world.

Example questions:

- What was written on the sign I passed earlier?
- Did I already walk past the library?
- What changed in this room since this morning?
- When did I last see this storefront?
- What color was the car parked outside?
- Did I see a fire extinguisher in this hallway?
- What did the whiteboard say before it was erased?

This is not intended to be a raw surveillance archive. The emphasis is structured, relevance-filtered memory.

## Intended Inputs

Potential:
- selected camera observations;
- OCR text;
- object detections;
- scene embeddings;
- timestamps;
- session context;
- optional spatial context when explicitly available.

## Core Architecture

```text
sensor stream
    |
shared/basic perception
    |
relevance / novelty detection
    |
event extraction
    |
environmental memory store
    |
retrieval / search
```

Instead of storing "video," the system should prefer storing meaningful observations.

## Observation Types

Potential normalized events:

```text
TEXT_SEEN
OBJECT_SEEN
OBJECT_MOVED
SCENE_CHANGED
LANDMARK_SEEN
PLACE_REVISITED
USER_SAVED_EVENT
```

Example event:

```text
type: TEXT_SEEN
timestamp: 2026-08-18T17:12:03
text: "Woodward Hall"
confidence: 0.96
context: exterior_sign
```

## Relevance / Novelty

Environmental Memory should aggressively reduce redundant data.

Potential signals:
- scene embedding novelty;
- OCR text not recently seen;
- object/state changes;
- user attention or explicit request;
- high-confidence landmarks;
- large visual changes;
- module-defined retention rules.

## Search

Potential retrieval interfaces:

```text
search_text("Woodward")
search_events(time_range)
last_seen("fire extinguisher")
what_changed(location_or_context)
recent_landmarks()
```

Natural-language search may sit above structured retrieval.

## Relationship to Object Memory

Object Memory focuses on tracked objects and their latest/history states.

Environmental Memory is broader:
- text;
- places;
- scenes;
- events;
- landmarks;
- environmental changes.

Do not merge the two modules prematurely. If both later require the same low-level observation service, that service can be promoted explicitly.

## Relationship to World Build

Environmental Memory does not require a geometric 3D map for its first versions.

If World Build later exposes a stable shared spatial service, environmental observations may optionally reference spatial locations. This must be an explicit architecture evolution, not an assumed dependency.

## Persistence

This module owns its event/index store.

Potential artifacts:
- structured events;
- text index;
- image/scene embeddings;
- selected keyframes;
- confidence metadata;
- session timestamps.

Raw continuous video should not be retained by default.

## Privacy

This module has significant privacy implications because long-lived environmental memory can contain bystander, location, document, and private-space information.

Design principles:
- store structured events instead of raw footage where possible;
- establish retention/deletion controls;
- avoid biometric identity features unless a future use case explicitly justifies them and is handled appropriately;
- provide a way to clear module memory.

These principles are governed by the platform-level policy in `06-PRIVACY-DATA.md`; this module's descriptor must declare its data behavior per `04-MODULE-SYSTEM.md` before implementation begins.

## Failure Behavior

If retrieval evidence is weak:
- return "not found" or uncertainty;
- never create a memory event retroactively to satisfy a query.

Observation gaps cannot establish absence: if this module never observed something, that means unknown, not "confirmed not present" or "confirmed unchanged." See `07-PLATFORM-CONSTRAINTS.md` Limitation 7 and Core Principle 3.

## First-Version Success Criteria

A first version should choose one constrained memory type, such as:
- searchable OCR history; or
- scene/event novelty memory.

Do not attempt universal physical-world memory in the first implementation.
