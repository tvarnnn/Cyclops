# Module Concept — Visual Q&A / Reading

## Status

**PLANNED.** No code exists anywhere under `tower/` for any stage of this
pipeline. Everything below describes intent, not behaviour. Comparatively heavy relative to other candidates — its pipeline includes speech-to-text, OCR/CV, multimodal reasoning, and text-to-speech, with a low end-to-end latency requirement. Not positioned as an early starter module; see `03-ROADMAP.md` Phase 3.

## Goal

Allow the user to look at something in the physical world, ask a question, and receive a concise spoken response through the glasses.

Examples:

- What does this sign say?
- Read this label.
- What object am I looking at?
- Summarize this page.
- Explain the diagram in front of me.
- What color is this item?
- Help me understand this homework problem.

The module is intended for legitimate assistance, learning, accessibility, and environmental understanding. It must not be designed around covert cheating or evading academic/testing rules.

## Intended Inputs

Potential:
- current camera frame or short burst of frames;
- microphone input for spoken queries;
- optional recent visual context;
- optional OCR output.

## Initial Sensor Profile Hypothesis

Likely:
- camera enabled on demand or at a low idle rate;
- microphone enabled when voice interaction is active;
- audio output enabled;
- prioritize response latency over continuous high-FPS streaming.

This module may not need continuous 15–30 FPS operation.

## Processing Pipeline

Conceptual flow:

```text
user question
    |
speech-to-text
    |
capture/select relevant visual frame(s)
    |
OCR / visual perception
    |
multimodal reasoning
    |
response generation
    |
text-to-speech
    |
glasses audio
```

A simpler reading-only path may be:

```text
frame -> text detection -> OCR -> cleanup -> speech
```

## Task Routing

The module may eventually classify requests into specialized paths:

```text
READ_TEXT
IDENTIFY_OBJECT
DESCRIBE_SCENE
ANSWER_VISUAL_QUESTION
EXPLAIN_DOCUMENT
```

Use deterministic/specialized CV/OCR components when they are sufficient instead of sending every request through an expensive multimodal model.

## Context

Do not automatically send a long continuous video history to the reasoning model.

Prefer:
- current frame;
- a small number of relevant recent frames;
- OCR text;
- detected objects;
- explicit user query.

## Local AI

Per the platform's local-first data policy (`06-PRIVACY-DATA.md`), this module's default processing path is local: tower-hosted models process camera/audio/document content without sending it to third-party AI/cloud services. This module is a primary reason that policy exists — it routinely handles documents, screens, IDs, financial information, and private communications in frame. Any exception requires the explicit documented process defined in `06-PRIVACY-DATA.md` (justification, user opt-in, disclosure, minimization, retention review) — it is not a default implementation path.

The architecture must not require Meta AI to perform application reasoning.

Model selection remains an implementation decision based on GPU limits, latency, accuracy, and licensing.

## Persistence

Default persistence should be minimal.

Possible saved data:
- optional query history;
- user settings;
- explicitly saved readings/notes.

Do not retain every viewed document/image by default.

## Output Style

Responses should be concise enough for audio delivery.

Examples:

```text
"The sign says: Computer Science Building."
```

or:

```text
"This is a quadratic equation. The next step is to move the constant term to the other side."
```

If confidence is low, say so.

## Failure Behavior

The module must permit and clearly surface "insufficient visual evidence" as a first-class response, not an edge case to be avoided. If no readable text or relevant visual evidence exists:
- state that clearly;
- avoid hallucinating an answer.

See `07-PLATFORM-CONSTRAINTS.md` Limitation 5 (Probabilistic ML Output) — this module must not be required to always produce a confident answer.

If the tower is unavailable:
- report module unavailable;
- do not fall back silently to an unrelated AI provider.

## Privacy

Documents, screens, IDs, financial information, and private communications may enter the camera view. Avoid persistent storage unless explicitly required by a feature. See `06-PRIVACY-DATA.md` for the platform-level retention/transmission policy this module must follow.

## First-Version Success Criteria

A bounded V1 could demonstrate:
- reliable frame capture;
- OCR on real first-person imagery;
- voice/text question routing;
- concise spoken answers;
- measured OCR/question-answer latency;
- evaluation against a small labeled test set.
