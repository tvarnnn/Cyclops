# Module Concept — Object Memory

## Status

> ## CORRECTED AGAIN 2026-08-27 — it has a Start button, a policy and a picture
>
> Three of the gaps the 2026-08-26 note left open are closed, and one of
> its own claims is corrected.
>
> - **The producer attaches itself.** `POST
>   /cartridges/object_memory/session/{start,pause,resume,stop}` and a
>   gated worker spec, so nobody copies a capture id into a second
>   terminal. The observation root now has ONE default, handed to the
>   producer and the read routes from the same settings object.
> - **The two-class whitelist became a measured policy.** Every detection
>   over all 18,821 real frames was dumped and the strongest crop of each
>   sighting was READ BY EYE. A ceiling fan is `airplane` at 0.99 and
>   `scissors` at 0.93; a white door is `refrigerator` at 0.95; the three
>   highest-scoring `remote` sightings are all laptop keyboards. Score
>   does not order correctness across classes, so the class list became
>   tiers: written on the detector's word, written only if a second
>   opinion agrees, context, or ignored. `person` is still excluded and
>   still unreachable by any model.
> - **The unit of memory is a SIGHTING**, not a 30-second timer.
> - **The frame reference became a picture.**
>   `/object-memory/observations/{id}/{imagery,frame,crop}`, face-filtered
>   on read, refusing rather than degrading, and answering "the memory is
>   kept and the picture is not" when capture-side retention has moved on.
> - **Persistent identity is NOT "forbidden outright"** — a claim the
>   contract document made and this brief does not. Limitation 6 of
>   `07-PLATFORM-CONSTRAINTS.md` lists embeddings and
>   confidence-scored association as MITIGATIONS. What this brief actually
>   says is *"Do not claim unique-object identity unless the
>   implementation actually supports it"*, which is a condition. It is
>   still not claimed, now for a measured reason: best frozen embeddings
>   get 26.4% Recall@1 on small mass-produced objects and tracking IDF1
>   collapses to ~40% from identical distractors alone.
> - **Still genuinely blocked:** the live in-process `Module` subclass and
>   any WebSocket surface, on the lifecycle ruling below. The cartridge
>   lifecycle added this run is a SESSION over an out-of-process producer;
>   it does not touch that ruling and does not resolve it.

> ## CORRECTED 2026-08-26 — the producer and the wire surface now exist
>
> This section said "Nothing currently produces an observation" and
> marked both the producer and any HTTP surface **BLOCKED**. Two of those
> three claims are now false; one is still true, and the distinction is
> the whole point.
>
> **The blocker gates a live in-process `Module`, not the cartridge.**
> Every other cartridge produces *out of process* by tailing a capture
> journal. Object Memory now does the same via
> `scripts/object_memory_session.py`, so the lifecycle ruling below is
> **untouched and still pending** — it was never what gated producing
> observations.
>
> - **55 real observations** from 9,199 real frames (29 `laptop`,
>   26 `cell phone`), written and read back.
> - **An HTTP read surface exists**: `tower/routes/observations.py`,
>   registered as a fifth router. Read-only by construction — `purge`
>   and `prune_expired` are unreachable from the wire, AST-enforced — and
>   retention cannot be widened over HTTP.
> - **Zero `person` records.** The `person` ruling is *sidestepped*, not
>   resolved: a closed whitelist is enforced at the **store**, not merely
>   at the filter, because a review found `append()` accepting a `person`
>   record directly.
> - **Still genuinely blocked:** the live in-process `Module` subclass and
>   any WebSocket surface, on the lifecycle ruling described below.
> - **Still missing:** an iOS surface.
>
> Detail: `docs/contracts/OBJECT-MEMORY.md`.

**PARTIALLY IMPLEMENTED.** Bounded enough to build incrementally while
exercising detection, tracking, temporal reasoning, relevance filtering
and persistence. It produces and serves observations today; what remains
blocked is running *live in process*.

| Part | Status |
|---|---|
| Observation record schema, relevance filtering, observation store (append, read, `last_seen`, retention pruning, real `purge`) | **CURRENTLY IMPLEMENTED** — `tower/object_memory/{records,relevance,store}.py`, tested |
| A detector producing observations | **IMPLEMENTED**, out of process — `scripts/object_memory_session.py`, 55 real observations |
| An HTTP read surface | **IMPLEMENTED** — `tower/routes/observations.py`, read-only, retention non-wideable |
| A `Module` subclass, wiring, any **WS** surface | **BLOCKED** on the lifecycle ruling |
| Spatial anchors against a World Builder world | **PLANNED** — the contract is written down but no anchor exists yet |

**The blocker, precisely.** Task 4 requires the module's `_do_load()` to
load a detector synchronously, which reproduces the unbounded-blocking
lifecycle gap `DepthEstimationModule` already has (see
`01-SYSTEM-ARCHITECTURE.md` — Reliability Policies, Known exception).
The costed options are: (A) `asyncio.to_thread` in this module only;
(B) the same fix made centrally, for every module's `_do_load()`;
(C) pre-provisioned weights, accepting the gap; (D) deferring Object
Memory until V1.1 lifecycle hardening; and (E) a separate, longer load
timeout, which composes with A or B. The Experimental CV Lab load path
uses the longer timeout (`LOAD_TIMEOUT_S = 120.0` in
`tower/modules/container.py`) and its own dedicated off-thread loader
(`ExperimentLoader` in `tower/cv_lab/loader.py`), not option A as
defined here; B, the contract change, remains V1.1 work. The Master Guide classifies choosing between them as
needing user judgement, so an agent must not resolve it autonomously.

That gap is real for the *live* path: `Module.process()` receives raw
JPEG bytes and no timestamp, sequence number or session id, so even a
working in-process detector could not populate the fields the schema
already has. The journal-follower sidesteps it by reading those fields
from the capture journal, where they exist.

**Before the first anchor is ever written**, add `anchor_keyframe_id` and
`position_in_anchor_frame` to the anchor schema. Without them the first
loop closure permanently and undetectably invalidates every earlier
anchor, because a submap re-anchor is not a global similarity and cannot
be composed forward. See `CARTRIDGE-GROUNDWORK.md` §4.

## Goal

Maintain a searchable history of where objects were observed over time so the user can ask questions such as:

- Where did I last see my keys?
- Have I seen my backpack today?
- When was my charger last visible?
- What room did I leave my water bottle in?

The module should reason from observations rather than continuously storing every raw frame.

## Intended Inputs

Primary:
- camera frames;
- frame timestamps;
- module/session metadata.

Possible future inputs:
- user voice queries;
- spatial/map context from another explicitly shared service if the architecture later supports it.

Do not assume cross-module data sharing exists in V1.

## Initial Sensor Profile Hypothesis

Early experiments may target:

- camera enabled;
- approximately 5–15 FPS;
- microphone optional and only enabled for voice-query interaction;
- audio output enabled when spoken responses are desired;
- moderate resolution sufficient for object detection/tracking.

Exact values must be verified against current DAT support and measured performance.

## Core CV Pipeline

Possible first implementation:

```text
camera frame
    |
object detector
    |
multi-object tracker
    |
observation builder
    |
relevance / state-change filter
    |
persistent object history
```

An observation may eventually contain fields equivalent to:

```text
objectClass
trackId
timestamp
confidence
boundingBox
sessionId
locationContext
visualEmbedding
eventType
```

The exact schema should be defined when implementation begins.

## Relevance

The module should not save every detection from every frame.

High-value events may include:

- newly observed object;
- object picked up or moved;
- object disappears after sustained visibility;
- object reappears in a new context;
- meaningful change in location;
- high-confidence observation useful for later retrieval.

Low-value repeated detections should be sampled, merged, summarized, or discarded.

## Identity vs. Category

Distinguishing "a backpack" from "my backpack" is harder than generic object detection.

Potential future techniques:
- visual embeddings;
- instance re-identification;
- user confirmation;
- object-specific enrollment/reference images.

Do not claim unique-object identity unless the implementation actually supports it.

## Persistence

Object Memory owns its own observation store.

Potential stored artifacts:
- object events;
- timestamps;
- selected keyframes/crops;
- embeddings;
- session metadata;
- confidence scores.

Avoid storing complete continuous video by default.

## Query Layer

The module may eventually support queries such as:

```text
last_seen("keys")
seen_today("backpack")
history("charger")
recent_objects()
```

A language model may translate natural-language questions into structured queries, but the underlying object-history data should remain queryable independently of an LLM.

## Output

Possible responses:

- "Your keys were last seen on the kitchen counter at 4:18 PM."
- "I last saw your backpack during the previous session."
- "I do not have a confident recent observation of your charger."

The system must expose uncertainty rather than inventing locations. Report "last observed at X," never "is currently at X" — an unobserved object is unknown, not confirmed absent or confirmed stationary. See `07-PLATFORM-CONSTRAINTS.md` Limitation 7 (Environmental Memory / Observational Gaps, which explicitly also covers Object Memory) and Core Principle 3 (Absence of Observation ≠ Observation of Absence).

## Failure Behavior

If the tracker/detector loses confidence:
- record uncertainty;
- avoid overwriting a strong previous observation with a weak one;
- never present a guessed location as fact.

If the tower is unavailable:
- module is unavailable;
- no false memory updates occur.

## Privacy

Object Memory may inadvertently capture sensitive surroundings. Prefer structured metadata/embeddings over stored imagery when they satisfy the feature. Selected object crops reduce data volume but are not inherently safe — a crop can still contain a bystander's face, a private room, a screen, or a document in the background. Treat any stored crop as potentially sensitive imagery, subject to the same retention/deletion requirements as full frames. See `06-PRIVACY-DATA.md`.

## First-Version Success Criteria

A bounded V1 should demonstrate:

1. stable first-person object detection;
2. basic temporal tracking;
3. meaningful observation persistence;
4. retrieval of the most recent reliable observation;
5. measured precision/recall or task-specific retrieval accuracy;
6. measured resource usage and latency.

Do not require full spatial mapping for the first version.
