# Module Concept — Experimental CV Lab

## Status

**CURRENTLY IMPLEMENTED.** The only cartridge that actually runs on the
live frame path today (`tower/modules/experimental_cv.py`, wired in
`tower/main.py`). See "Actual structure" below for how the shipped layout
differs from the one this document originally proposed.

Module #1 — the first module to be implemented (see `03-ROADMAP.md` V0.8–V0.9). Its role in this phase is to prove the complete glasses -> iPhone -> tower -> CV pipeline, serve as the sandbox for course experiments, and validate the module lifecycle/descriptor contract against real implementation experience before any dynamic registry is built.

## Goal

Provide a controlled place to test computer-vision techniques on first-person glasses data without contaminating production modules or forcing every experiment into the long-term architecture.

This module is especially useful while completing a computer-vision course.

## Philosophy

The experimental module may be messy internally, but its boundaries with the platform must remain clean.

It receives observations through the normal module interface and returns results through the normal module/output path.

It must not bypass:
- the DAT abstraction;
- the tower module manager;
- lifecycle rules;
- resource cleanup;
- privacy/security rules.

## Registered Experiments (V1, 2026-08-22)

What actually exists. Each names one **headline** measurement and may
report others alongside it.

| Experiment | Headline | Cost at 640x360 | State |
|---|---|---|---|
| `baseline` | `mean_intensity` | 1.0 ms | none |
| `edge_detection` | `edge_density` | 1.5 ms | none |
| `frame_quality` | `sharpness_laplacian_var` | 5.6 ms | none |
| `feature_detection` | `keypoint_count` | 4.2 ms | none |
| `redaction_impact` | `region_keypoint_retention` | 4.9 ms | none |
| `optical_flow` | `median_flow_px` | 4.6 ms | previous frame |
| `object_detection` | `detections` | 35.3 ms | model (`[ml]` extra) |
| `depth` | `mean_relative_depth` | 26.0 ms | model (`[ml]` extra) |

Costs are measured on this host with synthetic imagery
(`scripts/cv_lab_benchmark.py`) — real for the code, not a statement
about a real room.

**Selected at runtime**, since 2026-08-27: a client sends `cv_lab_start`
and the Lab arms the named experiment with no Tower restart. The full
wire contract — enumeration, lifecycle, provenance, metrics — is
`tower/docs/contracts/EXPERIMENTAL-CV-LAB.md`. `TOWER_CV_EXPERIMENT` survives as
the **startup default**: what this Tower arms at boot, so that a client
which knows nothing about the CV Lab still receives a `frame_result` for
every frame exactly as before.

**Every experiment must expose useful measurements.** An experiment whose
result nobody can act on is not an experiment, and the headline exists to
force the question: an experiment that cannot name its single most
important number has not decided what it is measuring.

## Result contract

```python
ExperimentResult(
    result_value: float,        # the headline, mandatory
    result_label: str,          # its name, mandatory
    processing_ms: float,
    stage_ms: dict[str, float],
    mean_intensity: float | None = None,   # historical, V0.7-compatible
    metrics: dict[str, float] = {},        # everything else
)
```

`metrics` reaches a client on `frame_result`, and is **omitted entirely
when empty** so a client that has never heard of it is unaffected. It is
deliberately `name -> number`: a **measurement** channel, not a general
result channel. Structured results (a detection list, a geometry delta)
need the module-contract work that is blocked at V1.0/V1.1.

## Experiment protocol

An experiment is anything with `name`, `load(settings)`, `run(bytes)` and
`release()`. The registry holds **factories**, not instances — building a
detector at import time would load model weights in any process that
imported the module.

A stateless experiment is still just a function; `StatelessExperiment`
adapts it. Before this existed, a stateful experiment cost a whole
`Module` subclass, and there were two of them sharing one descriptor id.

`release()` must free whatever `load()` allocated, must be safe to call
twice, and must be safe after a partial load — it runs on the FAILED
transition, which is reachable from anywhere.

## Candidate Experiments (not yet built)

Examples:

- edge detection;
- feature detection;
- SIFT/ORB-style descriptors where appropriate;
- feature matching;
- optical flow;
- background/foreground segmentation;
- object detection;
- instance/semantic segmentation;
- tracking;
- monocular depth estimation;
- camera-motion estimation;
- visual odometry;
- image retrieval;
- clustering;
- novelty detection;
- lightweight relevance classification;
- a GPU-accelerated variant of an existing experiment, benchmarked against its CPU baseline (see GPU / Acceleration Benchmarking, below).

The list is intentionally broad. Only one experiment needs to be active at a time.

**Do not turn this into a random collection of models.** Each addition
must answer a question another part of the platform actually has. Several
of the items above were considered for V1 and deliberately left out:
semantic and instance segmentation, tracking-by-detection, image
retrieval and novelty detection all lack a consumer today. Face detection
is not merely unbuilt but **BLOCKED**: this OpenCV 5 build has no
`CascadeClassifier`, `FaceDetectorYN` ships no model, no cascade or ONNX
file exists anywhere on disk, and there is no face imagery to validate
against. `redaction_impact` measures the *consequence* of redaction
without pretending to detect a face.

## Sensor Profile

The module should expose experiment-specific settings.

Examples:
- FPS;
- resolution;
- whether audio is required;
- frame sampling rate;
- model/algorithm selection;
- visualization/debug-output settings.

## Data Capture

The module may optionally support explicit dataset-recording sessions for course/research purposes.

Dataset capture is **not** this module's own mechanism. `tower/capture.py`
is shared infrastructure — the transport arms it, and Object Memory or a
Document cartridge would use the same recorder. Versioning a recording
made by shared transport with one cartridge's schema would let an
unrelated schema bump invalidate it.

Dataset capture must:
- be manually started/stopped;
- clearly indicate recording state (`GET /health` reports it);
- store data under the recorder's own root, not inside a cartridge;
- avoid indefinite background capture;
- include useful metadata such as timestamp/configuration;
- follow the platform data policy in `06-PRIVACY-DATA.md`.

## Experiment Structure

### Proposed structure (historical)

This was the shape proposed before implementation:

```text
experimental_cv/
    experiments/
        optical_flow/
        object_detection/
        depth/
        ...
    data/
    results/
```

### Actual structure

The Tower did not adopt it, and the divergence is deliberate rather than
accidental — a per-experiment package directory buys nothing when an
experiment is one module:

```text
tower/experiments/          one module per experiment
tower/modules/experimental_cv.py   the Module that hosts whichever one is selected
scripts/                    benchmark and analysis drivers
```

There is no `data/` or `results/` directory. Recorded datasets live under
the **shared** recorder's root (`tower/capture.py`, armed by
`TOWER_CAPTURE_ROOT`), and measured results are written up in the
documentation, never stored by the Lab.

## Adding a new experiment

Six steps, in order. Four of them are enforced — you cannot skip them and
get a working Tower, because each is a missing positional argument or a
failing gate rather than a convention.

### 1. Write the experiment

One module under `tower/experiments/`. A **stateless** experiment is a
function; `StatelessExperiment` adapts it:

```python
# tower/experiments/my_experiment.py
import cv2
import numpy as np

from tower.experiments import ExperimentResult, MetricKind, decode_gray
from tower.instrumentation import StageTimer

METRIC_KINDS: dict[str, MetricKind] = {
    "things_found": MetricKind.COUNT,
    "coverage": MetricKind.RATE,
}


def run(raw_bytes: bytes) -> ExperimentResult:
    timer = StageTimer()
    with timer.stage("decode"):
        image = decode_gray(raw_bytes)
    with timer.stage("measure"):
        ...
    return ExperimentResult(
        result_value=score,          # the headline. Mandatory.
        result_label="my_score",     # its name. Mandatory.
        processing_ms=timer.total_ms,
        stage_ms=timer.snapshot(),
        metrics={"things_found": count, "coverage": coverage},
    )
```

A **stateful** experiment is a class with `name`, `load(settings)`,
`run(bytes)` and `release()`. `release()` must free whatever `load()`
allocated, be safe to call twice, and be safe after a partial load — it
runs on the FAILED transition, which is reachable from anywhere. If it
holds a model, guard the handover with `LoadInvalidation` (see
`tower/loading.py`; a load can be ABANDONED and a late install is a leak).

**Decode through `decode_color` / `decode_gray`.** They turn three
distinct wire-reachable failures — a truncated JPEG, an empty buffer, a
1-pixel dimension — into a `FrameProcessingError`, which drops one frame.
A bare `cv2.error` is a MODULE failure, and `mark_failed()` is terminal.

**Persist nothing.** `test_no_experiment_persists_anything` forbids
`open()`, `write_json_atomic`, `append_jsonl`, `WorldStore` and
`ObservationStore` anywhere under `tower/experiments/`. **Import no
cartridge**: `test_the_experimental_cv_lab_does_not_import_a_cartridge`
forbids `world_builder` and `object_memory`. **Import torch inside a
function, never at module level**:
`test_importing_the_lab_does_not_import_torch` runs a subprocess to check.

### 2. Classify every metric

`METRIC_KINDS` above, beside the code that produces the numbers, because
only that code knows how they combine:

| Kind | Corpus answer | Use for |
|---|---|---|
| `RATE` | mean | a fraction, a score, a mean, a magnitude |
| `COUNT` | sum | a tally, including a 0/1 flag whose sum is "how many frames" |
| `CONSTANT` | the value observed | an image dimension, a configured threshold |
| `UNAGGREGATED` | nothing — a frame count only | a circular or otherwise unsummarisable quantity |

There is **no default**. A metric you emit and do not classify raises
`UnclassifiedMetricError`, and
`test_every_metric_the_experiment_emits_is_classified` will catch it
before a user does. That loudness is deliberate: the predecessor
classified by allowlist and SUMMED anything it did not recognise, which
hid eight dead names and fifteen mis-summed rates.

### 3. Register it, with its metadata

In `tower/experiments/__init__.py`, one entry in `_REGISTRY`:

```python
"my_experiment": ExperimentRegistration(
    lambda: StatelessExperiment("my_experiment", my_experiment.run),
    my_experiment.METRIC_KINDS,
    ExperimentMetadata(
        name="My experiment",          # what a person reads
        summary="One line saying what it measures and why.",
        provenance=PROVENANCE_MEASURED,  # or PROVENANCE_INFERRED
        stateful=False,
        requires_model=False,
        headline_label="my_score",     # must equal your result_label
        backend="opencv",              # or "torch"
        headline_unit=_FRACTION,       # or None if it genuinely has none
        metric_units={"coverage": _FRACTION},
        annotation_metric=None,        # or the metric that counts findings
    ),
),
```

A **FACTORY**, not an instance: constructing a detector at import time
loads model weights in any process that imports this module, including
every unrelated test.

All three arguments are positional. An experiment registered without
metric kinds or without metadata is a `TypeError` at import, not a blank
row on somebody's phone.

`provenance` is the one to think hardest about.
**Model output is inference, not measured fact** unless the experiment
validates against a ground-truth reference. iOS makes provenance a
REQUIRED field on every metric so that whoever decodes a reply has to
answer it — this is where the answer comes from.

`headline_unit=None` is a real answer meaning "this quantity has no
unit", and iOS renders it bare. Do not invent one: `depth` emits relative
inverse depth on an arbitrary scale, and "metric is not metres".

### 4. Add the tests

Copy the shape of the ones that exist:

| What | Where |
|---|---|
| It measures what it claims | `tests/test_experiments_measure_truth.py` |
| Hostile frames do not take the module down | `tests/test_experiments_hostile_input.py` |
| Every metric is classified, none is dead | `tests/test_experiment_metric_classification.py` |
| It is in the registry and is a factory | `tests/test_experiments_registry.py` |
| Its metadata is true | `tests/test_cv_lab_catalog.py` |

Three of those iterate the registry and will fail on your new entry
until you extend them. That is the intended signal.

If it is model-backed, add an **opt-in** integration test gated on
`TOWER_RUN_MODEL_TESTS=1` — see
`tests/test_object_detection_integration.py`.

### 5. Say what it costs

`scripts/cv_lab_benchmark.py` runs every cheap experiment at three
resolutions and prints a per-frame budget. Add yours to `CHEAP` (or
`MODEL_BACKED`) and put the number in the table at the top of this
document. An experiment with no measured cost cannot be chosen against
another one, which is most of what the Lab is for.

### 6. Check the wire

Nothing else is needed: `GET /cv-lab` and `cv_lab_status` derive the
catalog from `_REGISTRY`, so a registered experiment is a selectable one.
Confirm it:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_cv_lab_catalog.py -q
curl http://localhost:8000/cv-lab
```

If `test_the_document_names_every_registered_experiment` fails, add your
id to `tower/docs/contracts/EXPERIMENTAL-CV-LAB.md` §3.2. The contract document
is what a phone engineer implements from; a catalog that grows without it
is a list that becomes a surprise.

## Promotion Path

An experiment that proves useful should not remain permanently inside the sandbox.

Promotion process:

```text
experiment
   |
measured success
   |
write dedicated module/shared-service spec
   |
implement through normal architecture
```

For example, if relevance classification becomes essential across modules, it may later become a shared service after a separate architecture decision.

**Graduating out of the Lab, concretely.** The Lab is where an idea is
measured; a cartridge is where it is depended on. The move is not a
rename, and four things change:

1. **It stops being one of eight in one slot.** A cartridge gets its own
   module, its own descriptor, and its own `data_behavior` declaration —
   which is the point at which "persists nothing" stops being free and
   becomes a decision somebody signs.
2. **It stops sharing a run.** A Lab run is process-wide and shared by
   every connection. A cartridge that needs per-session state needs the
   session-boundary hook the module contract does not have yet (see
   `optical_flow`'s own note about the residual it cannot close).
3. **It gets its own contract identifier.** The Lab's status document
   describes A LAB — what can run, what is running, what it found. A
   cartridge's payload describes its own domain. Reusing
   `experimental_cv.status/...` for a shipped cartridge would tell a
   phone the wrong thing about what it is looking at.
4. **Its metrics stop being generic.** `MetricKind` and the Lab's
   aggregation exist because a bench compares experiments. A cartridge
   reports what its domain means, and `tower/results/<cartridge>.py` is
   where that shape is decided.

What DOES carry over: the experiment code itself, its measured cost, its
metric classification, and the evidence that made the case. That is the
whole reason to measure in the Lab first.

## Persistence

**The Lab persists nothing.** Its descriptor declares
`persists_data=False` and `retains_raw_imagery=False`, and the descriptor
is what `06-PRIVACY-DATA.md` is enforced against — an experiment that
quietly wrote to disk would make that declaration a lie. A test asserts
that no experiment calls a write primitive.

Measured results are written up in the documentation. Recorded datasets
live under the shared recorder's root. Neither belongs to an experiment.

Do not read/write another module's private data without an explicit shared-data design.

## Isolation from the cartridges

The Lab must not import `tower.world_builder` or `tower.object_memory`,
and a test enforces it in both directions.

The reason is specific rather than tidiness: the Lab's job is to
**measure** properties — blur rejection, feature yield, apparent motion —
that World Builder also has private opinions about. Importing that
opinion would mean restating a cartridge's answer instead of measuring
the underlying property, and a threshold change on the cartridge side
would silently move a measurement. It would also put a sandbox that may
be thrown away upstream of a persistent world.

The Lab may read a world for visualisation one day. Today it does not,
and it must never write one.

## Failure Behavior

An experiment failure must:
- mark the module/experiment failed;
- release GPU/CPU resources;
- leave the persistent tower runtime alive;
- not break module discovery or switching.

## GPU / Acceleration Benchmarking

Experimental CV Lab is the appropriate place to evaluate whether a GPU-acceleration technology is justified for a given workload, following the correctness -> instrument -> profile -> identify bottlenecks -> accelerate philosophy in `01-SYSTEM-ARCHITECTURE.md` — GPU / Acceleration Strategy. Benchmarking a candidate technology (PyTorch CUDA execution, TensorRT, CV-CUDA, or another) against a measured CPU baseline is itself a valid experiment here and should follow the same Success Criteria (hypothesis, baseline, metric, result) as any other experiment in this module. Do not adopt a GPU-acceleration technology platform-wide based on an unmeasured assumption that it will help — that decision is made from this module's measured results, not in the abstract.

## Model Output vs. Measured Fact

Experiment output (detections, depth estimates, classifications, tracked poses, etc.) is model inference, not a measured sensor fact, unless the experiment specifically validates against a ground-truth reference. Results/logs must distinguish the two. See `07-PLATFORM-CONSTRAINTS.md`, particularly Core Principle 2 (Inference ≠ Measurement) and Limitation 5 (Probabilistic ML Output).

## Success Criteria

Every experiment should define:
- hypothesis/question;
- dataset/session;
- metric(s);
- baseline;
- result;
- resource usage where relevant.

Avoid declaring an approach "better" without a measurement.
