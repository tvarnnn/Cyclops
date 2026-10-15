# Module Concept — Accessibility

## Status

**PLANNED**, and additionally **BLOCKED** on a capability that does not
exist. No code exists under `tower/`. Beyond that, this module's defining
requirement is minimum latency, and `Module.process()` is synchronous on
the event loop with no worker, queue or executor anywhere in `tower/`.
A genuine low-latency execution path is V1.0/V1.1 work and is the same
blocker World Builder stops behind.

Not a validated safety or navigation device, and nothing here may be
represented as one (Rule 13).

## Goal

Use wearable perception to provide useful spoken environmental information to people who benefit from visual assistance.

## Potential Capabilities

Candidates include:
- object identification;
- OCR/text reading;
- door/stair/obstacle awareness;
- requested-object localization;
- scene description;
- contextual audio feedback;
- later use of known mapped environments.

Each capability should be implemented and evaluated independently before being treated as reliable.

## Intended Inputs

Potential:
- camera;
- microphone for user requests;
- supported device interaction signals.

## Sensor Profile Hypothesis

Likely priorities:
- camera enabled;
- moderate/high supported frame rate when active;
- microphone enabled when voice interaction is required;
- audio output enabled;
- low end-to-end latency.

Exact settings must be negotiated against current DAT support and measured device behavior.

## Processing

Possible tower-side services:
- object detection;
- OCR;
- segmentation;
- depth estimation;
- tracking;
- speech recognition;
- language generation/reasoning.

Do not require all services for the first feature.

## Persistence

Accessibility owns its own preferences and any feature-specific history. Do not automatically record or retain continuous first-person video.

## Feedback

Audio should prioritize concise, timely, actionable information. Avoid flooding the user with every detection.

Future settings may include:
- verbosity;
- categories of alerts;
- OCR behavior;
- feedback frequency;
- confidence thresholds.

These may justify optional module-specific iOS controls later.

## Failure Behavior

If tower/module processing is unavailable:
- surface unavailable state;
- stop presenting AI-derived environmental guidance;
- do not replay stale warnings;
- do not imply the system is still protecting or navigating the user.

## Safety

This module is experimental assistive information. It must not be marketed or represented as a replacement for mobility aids, trained assistance, or validated safety/navigation systems without appropriate testing and validation.

This is the module where `07-PLATFORM-CONSTRAINTS.md` matters most. In particular:
- **Depth/distance** (candidate capabilities like door/stair/obstacle awareness) relies on monocular inference under current hardware (`07-PLATFORM-CONSTRAINTS.md` Limitation 1). Inferred spatial information must never be presented as safety-critical ground truth — the current sensor assumptions are explicitly documented as unsuitable for safety-critical distance estimation or precise obstacle dimensions.
- **Camera visibility ≠ user attention** (Limitation 8): the system observing something is not evidence the user perceived it, and must not be phrased as if it were.
- **Absence of observation ≠ confirmed safety** (Core Principle 3): not detecting a hazard must never be communicated as "the path is clear."

## Success Criteria

Each accessibility feature must receive its own measurable criteria, including latency and accuracy/error analysis relevant to that feature, before claims are made about usefulness.
