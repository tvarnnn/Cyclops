# Cartridge Groundwork — what World Builder leaves behind

Status: **architecture groundwork, not authorization.** Nothing here
authorizes implementing any cartridge named below. It records what the
first major cartridge built, what is genuinely reusable, and — more
importantly — which of its assumptions must **not** leak into the next one.

Written 2026-08-22.

---

## 1. The governing rule

World Builder is one consumer of the camera. It is **not** the definition
of how the camera behaves.

The transport layer (`tower/frames.py`, `tower/routes/ws.py`,
`tower/metrics.py`) is shared sensor infrastructure. The current iOS
~12 fps selection policy is a **current transport default**, not a
cartridge-independent law. Each cartridge should consume the available
frame stream and make its own decisions about which frames matter.

---

## 2. What is genuinely reusable today

| Asset | Where | Why it generalises |
|---|---|---|
| Append-only JSONL + atomic-rewrite store shape | `tower/object_memory/store.py`, `tower/world_builder/store.py` | Two independent modules have now converged on it. It survives torn lines, purges completely, and needs no engine |
| `Confidence` label vocabulary | `tower/object_memory/records.py` | Stored as a **label**, never a recomputed score, so a later threshold change cannot silently relabel history (Rule 16) |
| Real `purge()` that reports what it could NOT delete | both stores | `06-PRIVACY-DATA.md` requires real deletion; a false claim of deletion is worse than an honest failure |
| Single choke point for persisted pixels | `world_builder/store.write_keyframe_image`, `capture.write_frame` | Any future redaction or filtering policy becomes a change to one function |
| `time_basis` on every timestamp | both modules | There is no capture timestamp on the wire. Every cartridge inherits that and must say which clock it means |
| Explicit Dataset-Recording Session pattern | `tower/capture.py` | Off by default, bounded in seconds and bytes, purgeable, manifest declares posture |
| Synthetic ground-truth harness | `tests/synthetic_scene.py` | Exact poses, exact intrinsics, deterministic. Any cartridge doing geometry can assert real answers instead of eyeballing plausible ones |
| Refusal as a first-class result | `document_memory.retrieval.QueryResult` | `sufficient_evidence: False` plus a reason that says the refusal is about what was CAPTURED, not about the world. Every memory-shaped cartridge needs this and none should re-derive it |
| An OCR seam with a fast fake | `document_memory.ocr` | `TextRecogniser` protocol, `FixedTextRecogniser` for tests. A default suite must not download a model or pay 1.2 s per assertion |
| Tailing a capture as it is written | `capture.CaptureFollower` | A cartridge can process a dataset session live, in its own process, reading the journal so `source_seq`/`tx_seq`/receipt time survive. Bounded, so a crashed recorder ends the follow rather than hanging it |
| Calibration record + harness | `records.CameraIntrinsics`, `scripts/calibrate_charuco.py` | Intrinsics are a **platform** property, not a World Builder property. Every geometric cartridge needs them and none should re-derive them |

---

## 3. Assumptions that must NOT leak

These are World Builder's, and they are wrong for other cartridges.

| Assumption | Why it is World-Builder-only |
|---|---|
| **Keyframes over freshness.** Most frames are discarded; the useful ones are chosen for parallax | Visual Q&A and Accessibility want the **freshest** frame. A shared "frame selection" service built from World Builder's policy would actively harm them |
| **Latency does not matter.** `build()` takes tens of ms to seconds and runs offline | Accessibility is a latency-first consumer. Nothing in a shared path may assume deferred processing is acceptable |
| **Retaining raw imagery is justified.** World Builder persists keyframe JPEGs | It has a specific justification (rebuild, relocalisation). Most cartridges should persist derived data only, per `06-PRIVACY-DATA.md`. `retains_raw_imagery=True` must stay an exception that is argued for, not a default |
| **Motion is the signal.** Frames without parallax are worthless | Object Memory cares about object *change*; a static scene is not worthless to it. Text/Document wants deliberate still-like frames — exactly the frames World Builder rejects as `insufficient_motion` |
| **Geometry needs intrinsics.** No intrinsics → no poses | True for geometry, irrelevant for OCR, detection or QA. Nothing should gate a non-geometric cartridge on calibration |
| **A session is a mapping window.** Bounded by `stream_start`/`stream_stop` | Accessibility is continuous and has no natural session boundary |

---

## 4. Per-cartridge notes

### Object Memory (nearest neighbour, data layer already shipped)

- **Reuses:** the store shape it originated, `Confidence`, real purge.
- **Camera pattern:** occasional high-value frames on object change. Not
  parallax-driven, not continuous.
- **Extension point World Builder created:** the **spatial anchor
  contract**. `world_id` is an opaque `uuid4` that survives rename and
  refinement; keyframes are keyed `(session_id, source_seq)`; every record
  carries `frame_revision` and a declared `pose_convention`.
- **Frozen for it now:** an anchor must carry `world_id`,
  `frame_revision`, a position in **world units** (never metres),
  `observed_at` + `time_basis`, and a confidence **label**.
- **Additionally required, and NOT yet in Object Memory's schema:**
  `anchor_keyframe_id` plus `position_in_anchor_frame`. Without them the
  first loop closure permanently and undetectably invalidates every
  earlier anchor, because a submap re-anchor is not a global similarity
  and cannot be composed forward. Add them **before** any anchor exists.
- **Still missing:** relocalisation, and the loop closure that makes
  `frame_revision` ever advance.
- **Do not wire it yet.** Both modules must reach a stable bounded first
  version first.

### Visual Q&A

- **Reuses:** store shape, capture recorder, calibration record.
- **Camera pattern:** freshest observation, possibly a higher-quality
  still on demand.
- **Must not inherit:** keyframe selection, parallax gating, offline build.
- **Missing:** a "give me the current frame" primitive. The frame path
  currently pushes to one module; nothing exposes latest-frame semantics.

### Accessibility

- **Reuses:** almost nothing from World Builder. Possibly the calibration
  record if it ever does geometry.
- **Camera pattern:** continuous, minimum latency, immediate warnings.
- **Must not inherit:** anything deferred or batched, and certainly not a
  synchronous multi-millisecond stage on the frame path.
- **Missing:** a genuine low-latency execution path. `process()` being
  synchronous on the event loop is a blocker for this cartridge
  specifically, and it is the same V1.0/V1.1 work World Builder is
  stopped behind.

### Document Memory (BUILT, 2026-08-22 — see `DOCUMENT-MEMORY.md`)

This entry was written as a prediction. It is now a report, and the
prediction held on every point.

- **Reuses:** the capture recorder AND its follower, `Confidence`, the
  shared JSONL store shape, `time_basis` discipline. It processes a
  capture in a separate process exactly as World Builder does.
- **Camera pattern:** high resolution, deliberate still-like frames.
  Confirmed, and quantified below.
- **Did not inherit** World Builder's blur and motion gates, as predicted
  — and the reasoning turned out to be exactly right. A held-still page
  has near-zero parallax and is `insufficient_motion` to a mapper. This
  cartridge INVERTS World Builder's signal, and a test now forbids the
  import in both directions.
- **The predicted missing piece is now the measured blocker.** Word recall
  against known rendered text: 0.957–1.000 at 1280×720, 0.905–1.000 at
  640×480, and **0.429–0.810 at the 640×360 the glasses deliver**. Tilt
  barely matters once the page is warped; resolution dominates. Page
  *detection* still works at the delivered resolution — only *recognition*
  is starved. That is a requirement on the iOS/DAT side.
- **What it adds for the next cartridge:** a working, measured OCR path
  behind a substitutable seam, and a lexical retrieval layer that refuses
  rather than guesses. Visual Q&A's `READ_TEXT` route can reuse both.

### Scene Understanding (BUILT, 2026-08-22 -- see `SCENE-UNDERSTANDING.md`)

- **Reuses:** the capture recorder and its follower, `Confidence`, and
  the DETECTOR MEASUREMENT the Lab produced -- not the Lab's code. The
  Lab's `ExperimentResult` cannot carry a box, so the two share a model
  and nothing else. That is the promotion path working as intended:
  measure in the sandbox, build in the cartridge.
- **Camera pattern:** continuous, every frame, cheap. ~33 ms detection
  plus 0.016 ms for tracking, relations and the query layer together.
- **Persists NOTHING**, enforced by test, and that is the cleanest
  privacy posture any cartridge has managed. It is affordable only
  because the cartridge's purpose is a live answer.
- **Adds, for anyone building a perception cartridge:** counts that come
  from tracks rather than detections and survive 20% detector dropout
  unchanged; anonymous IoU-only association, which is how you track
  without acquiring identity; and a REFUSAL REGISTRY -- relationships the
  evidence cannot support, each recorded with what would settle it.
- **The pattern worth copying:** an answer type with `answered: False`
  and a reason. Asking "how many people are facing me" with orientation
  disabled refuses instead of returning 0, because zero looks like data.

### Environmental Memory

- **Reuses:** store shape, `Confidence`, retention/purge discipline.
- **Camera pattern:** sparse, event-driven.
- **Its boundary with Scene Understanding is now real, not hypothetical.**
  That cartridge holds the present and persists nothing; this one holds
  the past. The day a durable record of the physical world is wanted is
  the day this module starts -- not the day a store is added there.
- **Missing:** the shared spatial service, which depends on the Object
  Memory contract above.

### Experimental CV Lab (the one cartridge that already exists)

- **Reuses:** `Confidence`, the capture recorder and its follower, the
  synthetic ground-truth harness, `StageTimer`.
- **Camera pattern:** whatever the experiment under test needs. This is
  the one cartridge with no single camera posture, and that is its point.
- **Must not inherit:** any World Builder gate. Its job is to *measure*
  gates like blur rejection and parallax, so a Lab that silently applied
  them could not evaluate them.
- **Owns the constraint everyone else is waiting on:** `ExperimentResult`
  is five scalars, and it is the Lab's type. Every other cartridge's
  "there is no non-scalar result channel" problem is a change to a file
  the Lab owns.
- **Must not mutate authoritative state.** The Lab may read a world; it
  must never write one.

### Translator

- **Reuses:** almost nothing shipped so far. Audio is a different sensor
  path entirely (`07-PLATFORM-CONSTRAINTS.md` Limitation 13), and none of
  the camera, keyframe, geometry or calibration work applies.
- **Sensor pattern:** continuous microphone, not camera. It is the first
  planned cartridge whose primary input is not a frame.
- **Must not inherit:** the frame-shaped assumptions in every row above.
  A "sensor observation" in this codebase means a JPEG today; Translator
  is the cartridge that proves that is a transport detail, not a law.
- **Missing:** an audio path of any kind. There is no audio transport, no
  audio recorder, no streaming primitive and no output routing. Its first
  prototype is deliberately specified to run on Tower-local microphone and
  speakers, entirely outside the glasses path, precisely because none of
  that exists yet.

---

## 5. Infrastructure still missing

Named so nobody assumes it exists.

1. **Multi-consumer frame distribution.** `ModuleContainer` is a
   registry of one, so two cartridges cannot both be the active module.
   `app.state.frame_observers` *is* a real list and is now populated in
   production (`TOWER_CAPTURE_ROOT`), but it is a side-errand channel: an
   observer returns no result to the client. It is not a second module
   slot, and using it as one would spend the architecture decision the
   roadmap is protecting.
2. **Frame metadata into modules.** `Module.process()` takes only `bytes`.
   `received_at`, `source_seq` and `tx_seq` are unavailable to a module,
   which is why World Builder's engine is driven offline.
3. **A non-scalar result channel.** PARTIALLY CLOSED at V0.9.5:
   `ExperimentResult.metrics` is a `name -> number` bag, additive on the
   wire. That is a MEASUREMENT channel. Structured results — a detection
   list with boxes, a geometry delta, a world event — still cannot cross,
   and widening `metrics` beyond numbers would be a quiet way of
   pretending otherwise.
4. **An asynchronous execution path.** There is no worker, queue or
   executor anywhere in `tower/`.
5. **Cartridge-declared sensor requirements.** Rule 4 forbids designing a
   generalised negotiation protocol before the real DAT configuration
   model is known.
6. **A shared privacy filter.** Deliberately absent: one consumer is not
   an abstraction. The choke points exist; the filter does not.

Items 1–4 are the same V1.0 / V1.1 work that World Builder stops behind.
**A second cartridge with real requirements is exactly the trigger
`03-ROADMAP.md` names for V1.0** — and World Builder is now that second
set of requirements, written down.

---

## 6. Extraction rule

Before moving anything out of a cartridge into shared infrastructure:

1. Is it actually useful to another **known** cartridge?
2. Is the abstraction understood from at least **two** plausible consumers?
3. Does extracting it reduce real duplication?
4. Does it preserve cartridge isolation?
5. Does it add complexity today?

If unclear, **keep it local**. `Confidence` passes (two consumers, frozen,
20 lines). A privacy filter does not — yet.
