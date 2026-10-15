# Module Concept — Translator

## Status

**PLANNED — do not implement.** No code exists under `tower/`, and the
platform has **no audio path of any kind**: no microphone transport, no
audio recorder, no streaming primitive, no output routing. This module is
the first planned cartridge whose primary input is not a camera frame,
which is why its first prototype is deliberately specified to run on
Tower-local microphone and speakers, entirely outside the glasses path.

**Measured host reality, so nobody starts down a dead end.** Every audio
library is absent: no `sounddevice`, `pyaudio`, `soundfile`, `torchaudio`,
`webrtcvad`, `silero_vad`, `faster_whisper`, `transformers`,
`ctranslate2`, `piper` or `pyttsx3`, and no `ffmpeg` on PATH. Only
`winsound` (Windows built-in, WAV playback).

**And a likely hardware blocker.** Enumerating audio devices on this host
found sound *devices* (Realtek, NVIDIA) but the only PnP AudioEndpoint is
a monitor — an output. **No capture endpoint is listed, so there is very
likely no microphone attached.** This is the same shape as the finding
that no webcam exists here, which shaped the entire World Builder run.
Verify with a plugged-in microphone before writing code: the plan's own
first stage is otherwise untestable.

Future module concept. Not scheduled on the current roadmap (see `03-ROADMAP.md` Phase 3 and Future Research) and not authorized for implementation. This is a specification seed, recorded so the module-system and roadmap documentation have a concrete place to point to — see `02-DEVELOPMENT-RULES.md` Rule 10 (No Premature Scope Expansion).

## Goal

Provide low-latency conversational translation using the glasses' microphone and (eventually) audio output, so a wearer can converse with someone speaking a different language without operating a phone.

## Conceptual Pipeline

```text
microphone
    |
streaming speech recognition
    |
language detection / translation
    |
optional contextual language model
    |
speech synthesis
    |
audio output
```

The eventual goal is low-latency conversational translation. Together with CV, this module is one of the two motivating cases for the platform-wide latency-instrumentation requirement in `01-SYSTEM-ARCHITECTURE.md` — Latency Instrumentation: a translation pipeline with an unmeasured, unattributed latency budget cannot be tuned for a conversational use case.

## Visual Context (Future, Speculative)

Visual context from the glasses camera could later help resolve ambiguous language or references to visible objects — for example, disambiguating a spoken word that names something currently in view. This is speculative and explicitly not part of a first version: it introduces a second sensor stream and a multimodal fusion problem on top of an already latency-sensitive audio pipeline, and should not be attempted before a audio-only version is working and measured.

## Local Inference

Per the platform's local-first data policy (`06-PRIVACY-DATA.md`), local/Tower-hosted inference should be investigated where practical, for both privacy (conversational audio is sensitive) and latency (round-tripping to a third-party cloud service is unlikely to meet a conversational-latency bar). No specific ASR/MT/TTS model or vendor is selected here — that remains a later, separate, measurement-driven decision, consistent with how `docs/modules/WORLD-BUILD.md` and `docs/modules/VISUAL-QA.md` defer model selection.

## Relationship to Other Modules

Shares infrastructure with every other module per `04-MODULE-SYSTEM.md` (streaming/session infrastructure, tower compute), but is audio-first where the current module set is camera-first — its sensor profile and real-time requirements should be expected to look different from CV-oriented modules (e.g., microphone and audio-output enabled, camera likely off by default).

## Do Not Implement

This module is documented for future planning purposes only. Do not begin implementation, model selection, or protocol design as part of current work unless a future roadmap milestone explicitly schedules it.
