# Module Concept — World Build

## Status

**PARTIALLY IMPLEMENTED** as of 2026-08-22. This document began as a
concept seed and much of it still is one; the sections below are no longer
uniformly aspirational, so read the split carefully.

| Part | Status |
|---|---|
| Mapping engine — world/session/keyframe lifecycle, frame evaluation, keyframe selection, relative-scale geometry, persistence, cold reload, inspection | **CURRENTLY IMPLEMENTED** (`tower/world_builder/`, V1) |
| ChArUco camera calibration | **CURRENTLY IMPLEMENTED** (`scripts/calibrate_charuco.py`) |
| Incremental update stream and Tower-side live following | **CURRENTLY IMPLEMENTED** (`events.jsonl` + cursor, `world_inspect.py --follow`) |
| Registration as a production module on the live frame path | **BLOCKED** — the module contract is a registry of one with a scalar result type. A test pins non-registration deliberately (`tests/test_architecture_boundaries.py`) |
| Geometry transport to a phone | **IMPLEMENTED** on the Tower side — manifest + per-segment content-hashed chunks over HTTP, `world_builder.geometry/2026-08-25`, contract in `docs/contracts/` |
| PC/phone viewer | **WRITTEN, NEVER COMPILED** — the iOS decoder, client, cache and fragments renderer exist; no Swift toolchain on the Tower host has ever built them |
| Metric scale, loop closure, relocalisation, multi-session refinement | **PLANNED** |
| Validation against real Ray-Ban footage | **PARTLY DONE — this line was wrong, corrected 2026-08-26.** See below |

> **Correction, 2026-08-26.** "Every measurement to date is synthetic" is
> no longer true, and World Builder is now the *most* real-data-validated
> cartridge in the program. It has been run over **9,199 real frames**
> across real captures:
>
> - Tracking was diagnosed and fixed on real footage — the tracker was
>   losing *reach*, not the image (47 of 50 declared losses still had
>   survival above the floor against the previous frame). Across five real
>   captures: **151 -> 114 segments, poses 211 -> 265, points
>   27,406 -> 42,100**.
> - Calibration is solved, not blocked: `intrinsics/360x640.json`,
>   self-calibrated over **511 views at 0.289 px RMS**.
> - Cross-segment registration produces its first merged geometry on the
>   real world: 51 segments, 19 with geometry, **3 registered carrying
>   31.1% of all points** — and it refuses more than it admits.
>
> **What is still genuinely blocked on hardware** is narrower: eleven
> physical checks, P1–P11. The three that matter: whether fragments appear *during* a walk (P3),
> whether a deliberate sideways walk raises the registrable fraction
> (P11 — 16 of 19 segments are refused because the wearer stood still, so
> scale is unobservable), and whether a registered pair is actually
> correct (P9 — nothing automated can catch a wrong Sim3; pair (30,50)
> fits at 1.62 px while being **3.2x wrong on scale**).
>
> Separately: **no Swift in this repo has ever been compiled.** That gates
> the viewer regardless of hardware.

## Goal

Use first-person wearable camera observations to incrementally construct a spatial representation of environments the user traverses.

## Session Model — superseded product ruling (2026-08-22)

**For V1, a deliberate Start → Walk → Stop mapping session IS World Build.**

The V1 product experience is explicitly: start World Builder, walk
naturally around a room, watch the world build incrementally, stop, and
have the world persist on the Tower for later reopening and inspection.

This is a deliberate product decision recorded on 2026-08-22, and it
supersedes the final paragraph of the Passive Operation Requirement below
— the sentence ruling that a deliberate capture action "is not World
Build". That earlier ruling is **preserved verbatim below, not erased**,
because it records real design reasoning that still applies to everything
except the session boundary.

What the earlier requirement still governs, unchanged and binding:

- The wearer is **not** directed *during* a session. No instruction to
  scan a specific wall, rotate slowly, walk around an object, point the
  camera somewhere specific, or complete a predefined capture sequence.
- Within a session the wearer walks and looks **naturally**, and the
  system — not the user — decides which ordinary frames are worth the
  expensive path (see Relevance, below).
- A session that degenerates into scanner-operator behaviour has failed
  the premise, and that failure is still a product-integrity bug.

What changed is only the **boundary**: an explicit start and stop are now
permitted, and are how V1 scopes a mapping session to one room. Explicit
start/stop also aligns the module with `06-PRIVACY-DATA.md`'s Explicit
Dataset-Recording Session rules, which is a privacy improvement over
implicit always-on accumulation: capture becomes visible and bounded
rather than incidental.

Passive, undirected accumulation across sessions remains the long-term
direction; V1 does not foreclose it.

## Passive Operation Requirement

*(Historical. Superseded on its session-boundary clause only — see the
Session Model ruling above. Retained in full.)*

World Build must not behave like a traditional room-scanning application. Do not instruct the wearer to scan a specific wall, rotate slowly, walk around an object, point the camera somewhere specific, or complete any predefined capture sequence. The wearer uses the glasses normally; World Build incrementally accumulates spatial information from whatever the wearer naturally looks at.

Illustrative behavior:

```text
enter room                                -> rough structure begins appearing
look around naturally                     -> additional geometry becomes available
observe an object again from a new angle  -> existing geometry becomes more
                                              confident/refined
return to the room later                  -> the persistent model can continue
                                              improving
```

This is a product-integrity requirement, not only a UX preference: a system that requires scanner-operator behavior from the wearer has failed the platform's premise that the glasses are a normal-use device, not a capture tool. Any future capability that depends on the wearer performing a deliberate scan action is not World Build — it is a different feature and must be designed and labeled as one.

## Intended Inputs

Primary:
- camera frames.

Possible future inputs, only if DAT/hardware exposes useful supported data:
- motion/inertial information;
- timestamps;
- additional sensor metadata.

Do not assume unavailable sensors.

## Initial Sensor Profile Hypothesis

For early experiments:
- camera enabled;
- approximately 15 FPS target;
- moderate supported resolution such as approximately 504x896 if current DAT supports it;
- microphone off unless a later feature requires it;
- minimal audio feedback.

These are experimental targets, not guaranteed device capabilities. Verify through current DAT documentation and benchmarks.

## Hardware Reality: Monocular RGB Only

The current target Meta glasses provide a monocular RGB camera. They do not provide direct metric depth sensing such as LiDAR or stereo depth. World Build's design must treat this as a hard constraint, not an implementation detail to work around later.

World Build must therefore distinguish, in its data model and in anything it ever surfaces, between three distinct kinds of spatial information:
- **relative spatial geometry** — structure, layout, and relationships derived from multi-view geometry, without a claimed absolute physical scale;
- **inferred metric depth** — a distance estimate produced by ML monocular depth estimation or geometric inference, carrying model/method uncertainty;
- **directly measured metric depth** — depth from dedicated depth-sensing hardware (LiDAR, stereo). Not available on the current target hardware.

**World Build must never represent monocularly inferred depth as ground-truth physical distance.** Any distance figure derived from monocular inference must be identifiable as an estimate wherever it is stored, displayed, or consumed by another module.

This module is the primary case behind `07-PLATFORM-CONSTRAINTS.md` Limitation 1 (Monocular Depth / Scale) and Core Principle 2 (Inference ≠ Measurement); see that document for the full mitigation classification and confirmed DAT camera-configuration constraints this pipeline must work within.

## Candidate Pipeline (Hybrid, Not a Single Technique)

World Build should use a hybrid spatial-reconstruction pipeline rather than relying on one depth-estimation technique:

```text
RGB camera frames
    |
keyframe selection
    |
visual SLAM / Structure-from-Motion
    |
ML monocular depth estimation
    |
semantic/object understanding
    |
spatial fusion
    |
persistent world representation
```

Potential sources of spatial constraints to fuse, none authoritative alone:
- multi-view geometry across camera motion;
- ML monocular depth estimates;
- recognized objects with approximately known dimensions;
- floor/wall/ceiling plane estimation;
- estimated camera/head height where appropriate;
- temporal consistency across observations;
- future IMU/motion data, if DAT exposes suitable sensor access.

Do not adopt every technique by default. Select the smallest pipeline that satisfies the project's actual course/research objective. Do not choose or implement a specific ML depth model as part of this documentation decision — that remains a later, separate decision.

## ML Depth Estimation

Machine-learning monocular depth estimation is intended to partially compensate for the lack of dedicated depth hardware, not to stand in as an equivalent of it.

- Prefer metric-depth-capable models where practical over relative-depth-only models, since metric output is more useful to fuse with other constraints.
- ML-derived distance remains an estimate. It must not be treated as equivalent to LiDAR/depth-sensor measurement anywhere in the system, including in later modules that consume World Build's output.
- The Tower GPU should perform this computationally expensive depth/reconstruction work rather than the glasses or iPhone, whenever practical — consistent with the platform's existing lightweight-phone / compute-on-Tower principle.

## Confidence / Uncertainty

Spatial observations should eventually carry confidence/uncertainty information rather than being presented as uniformly reliable. Illustrative levels, extended here to include the unmeasured case explicitly rather than treating it as merely "low confidence":
- **unknown**: never observed; no geometry should be asserted here at all;
- **low confidence**: single-frame inference, reflective surfaces, low light, motion blur, textureless regions, or conflicting estimates;
- **medium confidence**: ML depth supported by some geometric constraints;
- **high confidence**: repeatedly observed geometry supported by multiple viewpoints.

**Unknown space must remain unknown.** World Build must not use generative AI, or any other technique, to fabricate geometry for a region the camera never observed — even when that geometry is statistically likely (for example, inferring the exact shape of the back of a couch that was never seen). A region with no observation is represented as unknown/unmapped, not filled in with a plausible guess. This is both a technical requirement — an unobserved region is architecturally the same case as `07-PLATFORM-CONSTRAINTS.md` Core Principle 3 (absence of observation ≠ observation of absence), extended here to mean absence of observation is also not license to invent one — and a product-integrity requirement: World Build's value depends on the wearer being able to trust that what it shows was actually seen, not synthesized.

Repeated observations from multiple independent viewpoints raise a region's confidence toward high; a single viewpoint keeps it at low confidence; a region never observed stays unknown until it is actually observed. World Build should prefer repeated observations and multi-view consistency over a single-frame depth prediction. The exact confidence representation/schema is not decided here — this section records the requirement that one must exist, not its implementation.

## Real-Time vs. Asynchronous Processing

World Build does not need to perform full high-quality reconstruction synchronously. A future architecture may split live, lightweight work from heavier asynchronous Tower work:

```text
Live (glasses/iPhone):
camera -> lightweight tracking / pose estimation -> keyframe selection

Asynchronous (Tower):
keyframes -> depth estimation -> SfM/SLAM refinement -> map optimization -> persistent reconstruction
```

This keeps expensive reconstruction workloads on Tower hardware rather than making the glasses/iPhone responsible for heavy computation, consistent with the platform's existing architecture.

## Candidate CV Techniques

Techniques the pipeline above may draw on, to evaluate as the CV course progresses:
- feature detection/description;
- feature matching;
- optical flow;
- visual odometry;
- keyframe selection;
- monocular depth;
- Structure from Motion;
- visual SLAM;
- point-cloud reconstruction;
- later reconstruction/rendering techniques.

## Persistence

World Build owns its mapping artifacts, which may eventually include:
- keyframes;
- features/descriptors;
- camera pose estimates;
- reconstruction data;
- map metadata;
- experiment metrics;
- per-observation confidence/uncertainty metadata (see Confidence / Uncertainty above).

Storage format is intentionally undecided until the implemented pipeline requires one.

## Relevance

Continuous video is highly redundant. World Build may eventually use keyframe/relevance selection so only observations that improve the map receive expensive processing or long-term storage.

This is also the mechanism that reconciles the Passive Operation Requirement (above) with computational cost. The wearer is not asked to slow down, hold still, or scan deliberately — so the system, not the user, is responsible for deciding which ordinary frames are worth the expensive path (SLAM refinement, semantic segmentation, depth estimation) versus lightweight live tracking only. This is a genuine tension worth naming explicitly: "passive" and "always run the full heavy pipeline" are in conflict, and keyframe/relevance selection plus the Real-Time vs. Asynchronous split above is the intended resolution, not a fully general SLAM system running on every frame.

## Output

Possible outputs:
- map/reconstruction status;
- mapping quality metrics;
- reconstructed environment artifacts;
- minimal audio/status feedback.

Any output that includes a distance/measurement must be labeled according to the relative / inferred-metric / measured-metric distinction above — never presented as measured when it is inferred.

## Live Visualization (Future)

World Build should eventually support a live PC and/or phone visualization: while the wearer naturally moves through an environment, the reconstructed world progressively appears in a viewer.

```text
Glasses -> iPhone -> Tower reconstruction -> world-state updates -> PC/phone viewer
```

Conceptual viewer content, once built:
- reconstructed geometry;
- color/texture where available;
- mapped vs. unmapped areas (see Confidence / Uncertainty, above — an unknown region should render as unknown, never as blank-as-if-absent and never as fabricated);
- current camera/glasses pose;
- semantic objects, once semantic understanding is part of the pipeline;
- confidence/quality information.

Prefer incremental world-state updates/deltas over repeatedly transmitting the complete world representation, consistent with the platform's general preference for bounded, freshness-oriented data flow (`01-SYSTEM-ARCHITECTURE.md` — Reliability Policies).

**Status update, 2026-08-22.** The premise of the sentence this
paragraph used to carry — "it depends on a working reconstruction pipeline
that does not yet exist" — is no longer true. The pipeline exists and runs
end to end offline, and the Tower half of live viewing is built: the
append-only event journal is an incremental update stream, `read_events`
takes a cursor, `world_build_session.py --follow-capture` builds a world
while frames are still arriving, and `world_inspect.py --follow` watches
it happen from a separate process.

**The conclusion still holds, for a different reason.** Do not implement
the PC/phone viewer yet — not because there is nothing to show, but
because there is **no transport to show it over**. `frame_result` carries
five scalars and World Builder is not a registered module, so no world
data can reach a phone at all.

The preference for incremental deltas over whole-world transmission,
stated above, is already how the persisted representation works — a viewer
built later inherits it rather than needing it retrofitted.

## Relationship to Object Memory / Environmental Memory (Future)

`docs/modules/ENVIRONMENTAL-MEMORY.md` already reserves this relationship as an explicit future architecture evolution, not an assumed dependency ("If World Build later exposes a stable shared spatial service, environmental observations may optionally reference spatial locations."); this section records World Build's side of the same boundary.

Today, Object Memory and Environmental Memory record observations independent of any spatial map (e.g., "keys detected at timestamp X"). If World Build later exposes a stable shared spatial service, an observation could instead be associated with spatial context:

```text
Object: keys
Location: spatial coordinate / mapped surface
Semantic location: kitchen counter
Last observed: timestamp
Confidence: value
```

This could eventually support queries like "where did I leave my keys?" answered from structured spatial-memory association rather than exhaustive replay/search across raw video.

This is future architecture, not a current implementation requirement. It depends on World Build and Object/Environmental Memory each independently reaching a stable, bounded first version before any explicit shared-service design is proposed — do not couple them prematurely, for the same reason `docs/modules/ENVIRONMENTAL-MEMORY.md` gives for not merging Object Memory and Environmental Memory prematurely.

## Hardware Suitability

Given the monocular-RGB-only constraint, the current glasses should be considered suitable for:
- relative spatial mapping;
- place recognition;
- rough environmental reconstruction;
- semantic mapping;
- approximate depth.

They should **not** be considered authoritative for:
- centimeter-accurate measurements;
- safety-critical distance estimation;
- precise obstacle dimensions;
- other applications requiring measured metric depth.

Any future feature (in World Build or elsewhere, e.g. Accessibility) that would need one of the disallowed capabilities above must not rely on this module's monocular inference to provide it.

## Future Hardware / Sensors

Recorded here as a future architectural direction only — not authorization to implement, and not a reason to move World Build earlier in the roadmap.

- If future DAT versions expose useful synchronized IMU/motion data, evaluate visual-inertial odometry (VIO) as an enhancement.
- If future glasses hardware exposes stereo/depth/LiDAR sensing, World Build should be designed so those measurements can augment or replace inferred monocular depth **without rewriting the higher-level world representation** — i.e., the relative/inferred/measured distinction and the fusion/confidence model above should already accommodate a future measured-depth input as just another, higher-confidence source.

## Success Criteria for First Version

Define these when the CV course requirements are known. The first version should prove one bounded spatial-CV capability rather than attempting city-scale reconstruction.

## Safety / Privacy

First-person mapping may capture bystanders, private spaces, screens, documents, and location-revealing imagery. Data collection/storage policies must be deliberate before real-world large-scale capture. See `06-PRIVACY-DATA.md` for the platform-level policy this module's data behavior must satisfy.
