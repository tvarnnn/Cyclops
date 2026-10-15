# System Architecture

## Top-Level Architecture

```text
[Ray-Ban Meta]
      |
     DAT
      |
[iOS Control Plane]
      |
 authenticated network transport
      |
[Persistent Tower Runtime]
      |
[Module Manager]
      |
[Exactly One Active Module]
```

## iOS Control Plane

The Swift app owns:
- DAT integration through a `GlassesConnection` abstraction.
- Stream/session coordination.
- Tower connectivity.
- Module discovery (a single fixed module initially; dynamic discovery once the registry is generalized — see `03-ROADMAP.md`).
- Module activation requests.
- Status and telemetry presentation.
- Generic module controls by default.
- Optional module-specific UI only when genuinely required.

It does not own heavy CV/AI processing.

## Tower Runtime

One server/runtime remains running while modules are swapped. Shared runtime responsibilities may include:
- transport/session endpoint;
- module container (a single hardcoded module slot initially; a dynamic module registry once a second module justifies it — see `03-ROADMAP.md`);
- module lifecycle manager;
- logging and telemetry;
- model/resource manager when sharing is justified.

Module-specific state should not automatically become global state.

### Initial Tower Stack

Pinned for the first implementation: Python, FastAPI, WebSockets for
real-time frame transport, OpenCV, and CUDA when GPU acceleration is
required. PyTorch is an optional `ml` extra (`pip install -e ".[ml]"`),
required only by a module that actually selects a model-backed CV
experiment — the core transport/health/WS runtime has no dependency on
it. Do not introduce gRPC, C++, distributed infrastructure, message
brokers, or container orchestration speculatively — revisit only if
measurements expose a real limitation.

## Module Switching

V1 permits one active major module.

Required lifecycle:

```text
User selects module
        |
Pause forwarding frames to processing
        |
Stop current module
        |
Persist module state if needed
        |
Release module-specific CPU/GPU resources
        |
Load selected module
        |
Initialize models/config/storage
        |
Apply requested sensor profile where supported
        |
Module reports READY
        |
Resume forwarding
```

The glasses connection should remain alive during a normal switch when DAT permits it. If current DAT behavior requires a session restart for a configuration change, the transport layer may perform that restart. Modules must not contain DAT-specific logic.

No sensor observations may be processed by a module while it is STARTING, STOPPING, FAILED, or otherwise not READY.

## Reliability Policies

These are architectural requirements; exact timing constants are an implementation/measurement decision, not fixed here.

**Lifecycle timeouts.** No lifecycle operation (LOADING, STOPPING, or equivalent) may block indefinitely. Each requires a bounded timeout. On timeout, the module transitions to a defined FAILED/unavailable state rather than hanging. A module timeout must not crash the persistent tower runtime.

> **Known exception, stated rather than glossed (2026-08-22).** This
> guarantee is **not currently true for a synchronous `_do_load()`**.
> `ModuleContainer` wraps each lifecycle call in `asyncio.wait_for`, but
> `asyncio.wait_for` cannot cancel synchronous CPU-bound work already
> running on the event-loop thread — the timeout fires only *after* the
> blocking call returns. `DepthEstimationModule._do_load()` calls
> `torch.hub.load()` synchronously inside an `async def`, so a slow or
> hanging model download is genuinely unbounded today. This is a **known,
> named, deliberately unfixed gap**: it is the V1.1 lifecycle work, and
> resolving it is a standing decision gate that also blocks Object Memory
> Tasks 4–8. Do not treat the bullet above as describing current
> behaviour for that path.

**Reconnection.** Automatic reconnection (glasses session or tower connection) must use bounded/exponential backoff rather than a tight retry loop. Exact retry timing is an implementation decision.

**Backpressure.** The real-time perception pipeline must not grow an unbounded frame queue. Freshness generally matters more than processing every frame; when the tower/module cannot keep up, stale frames should normally be dropped rather than accumulating latency. Exact queue/drop strategy should be set from the measurements taken in `03-ROADMAP.md` V0.7, not decided in the abstract.

**Adaptive streaming (future).** Conceptually allow future operating states such as IDLE / TRACKING / HIGH_RATE frame-rate adjustment (see the original brainstorm doc for the shape). This is a future optimization to be designed from V0.7 measurements — do not implement it before those measurements exist.

## Tower Failure

If the tower becomes unreachable:
- iOS reports `Tower: Unavailable`.
- Active module is presented as unavailable.
- Processing stream is paused.
- Do not fabricate results.
- Do not buffer large amounts of video by default.
- The glasses connection may remain alive.
- Reconnection may occur automatically, using bounded/exponential backoff (see Reliability Policies above).
- After reconnection, refresh module state/registry before resuming.

## Remote Tower Access (Future Milestone)

Remote (non-LAN) tower access is a real project goal but is explicitly out of scope until the local pipeline (through Roadmap V0.7) works reliably. See `03-ROADMAP.md` Phase 1.5.

Target shape:

```text
Glasses -> iPhone -> cellular/remote network -> private secure connection -> home tower
```

Preferred approach: a private overlay/tunnel (e.g., Tailscale or WireGuard) rather than exposing the tower directly to the public internet. Do not design the full remote-access implementation before this milestone is reached.

## Dynamic Discovery

The iPhone must eventually obtain the available module catalog from the tower. The production module list must not be permanently hardcoded into Swift long-term.

Sequencing: the first tower implementation hosts exactly one module (a "registry of one," see `03-ROADMAP.md` V0.8–V0.9). The iPhone may treat the single module as effectively fixed during this phase. Full dynamic discovery is built only once a second production module creates real requirements to generalize against — do not build the general discovery protocol speculatively.

Conceptual interaction (target shape once generalized):

```text
iPhone -> request module registry
Tower  -> module descriptors
iPhone -> activate module ID
Tower  -> lifecycle transition
Tower  -> READY / FAILED
```

Exact protocol/API choices belong to that later milestone.

## GPU / Acceleration Strategy

Tower hardware currently includes an RTX 5070, expected to perform heavy CV/AI inference when available. NVIDIA acceleration is an **optional Tower capability, not a hard platform dependency** — the glasses platform (glasses, iPhone, module contracts, and CPU-only Tower code paths) must remain functional, testable, and deployable without a CUDA-capable GPU present.

Philosophy governing every acceleration decision, not only NVIDIA's:

```text
build for correctness
    |
instrument
    |
profile
    |
identify bottlenecks
    |
accelerate only where measurements justify it
```

Do not adopt a GPU-acceleration technology because it is available or because the tower hardware supports it. Adopt it because profiling data (see Latency Instrumentation, below) identified a specific bottleneck that the technology specifically addresses.

**Candidate technologies** — evaluate each on its own merits; this is not a commitment to use all of them:
- **CUDA** — baseline GPU compute; already pinned as available "when GPU acceleration is required" (see Initial Tower Stack, above).
- **PyTorch CUDA execution** — the natural first step once a model-based bottleneck is identified, since PyTorch is already the ML framework used by the optional `ml` extra (see Initial Tower Stack, above); lowest integration cost of this list.
- **TensorRT** — inference-graph optimization/compilation; justified once a specific PyTorch model is profiled as the bottleneck and its inference time, not data movement or preprocessing, dominates.
- **CV-CUDA** — GPU-accelerated classical CV/preprocessing operations; justified only if profiling shows CPU-side OpenCV preprocessing, not model inference, is the bottleneck.
- **DeepStream** — a multi-stream video-analytics pipeline framework built for many concurrent camera feeds (e.g., NVR/edge-analytics deployments on Jetson-class hardware). The current architecture processes one glasses stream through one active module on one desktop Tower. DeepStream's pipeline/plugin model, GStreamer dependency, and multi-stream orchestration are unlikely to earn their integration and operational cost at this scale. **This is the weakest candidate on this list** — do not adopt it without a specific, measured multi-stream requirement that CUDA/PyTorch CUDA/TensorRT cannot satisfy.

Evaluation of these candidates belongs in Experimental CV Lab (`docs/modules/EXPERIMENTAL-CV.md`), where bounded experiments can measure whether a given technology's integration cost is justified by an actual latency/throughput improvement. See also `02-DEVELOPMENT-RULES.md` Rule 17 — a documented technology candidate is not a mandate, and should be challenged if a later measurement shows it isn't justified.

## Latency Instrumentation (Future Requirement)

End-to-end latency is a first-class platform metric, not an incidental log line. This applies beyond CV: any latency-sensitive future module (e.g., a live Translator — see `03-ROADMAP.md` Phase 3 and `docs/modules/TRANSLATOR.md`) depends on the same measurement infrastructure.

The platform should eventually be able to attribute latency to each stage of the pipeline:

```text
capture
    |
transport
    |
decode
    |
preprocessing
    |
inference
    |
postprocessing
    |
application/module processing
    |
response/output
```

Each stage should eventually carry enough timestamp/metric data to determine where latency actually originates, rather than only measuring a single aggregate figure. This mirrors the existing timestamp-provenance discipline in `07-PLATFORM-CONSTRAINTS.md` Limitation 9 (capture time, network arrival time, and processing time are conceptually distinct) and Limitation 15 (Sensor Authority / Provenance) — per-stage latency attribution is the same discipline applied to timing instead of identity/confidence.

This is a documented architectural requirement, not an implemented capability. Do not build full per-stage instrumentation before it is the current milestone. V0.7 already reports two coarse figures (`cv_processing_ms`, `receive_to_result_ms`) as a starting point; per-stage breakdown is future work, informed by whichever stage those coarser measurements identify as the actual bottleneck.

## Heterogeneous Compute & Graceful Degradation (Future Direction)

The platform's compute hierarchy is intentionally uneven:

```text
Glasses — sensors, capture, minimal necessary device-side work
iPhone  — DAT integration, connectivity/relay, UI/control, and
          potentially lightweight inference where appropriate
Tower   — heavy CV, GPU inference, local LLMs, reconstruction,
          memory processing, and other computationally expensive
          workloads
```

This mirrors `00-PROJECT-VISION.md`'s Responsibilities section; this section adds one point that document does not yet make explicit: **the architecture must not assume a Tower always exists.** A future module should eventually be able to detect available compute (Tower present/absent, GPU present/absent) and degrade gracefully — e.g., a lighter on-device experience, or a clear "heavy features unavailable" state — rather than failing opaquely or silently pretending full capability.

This is a future direction, not a current requirement. Do not design a distributed compute scheduler, capability-negotiation protocol, or automatic workload placement system now. The current architecture — Tower assumed present, with `02-DEVELOPMENT-RULES.md` Rule 3's truthful-unavailable-state requirement already covering the "Tower absent" case — is sufficient until a real second compute target or a real Tower-optional use case exists.

## Privacy & Data Handling

Raw sensor data (camera/audio) is local-first by default: it stays within Glasses -> iPhone -> tower and is not sent to third-party AI/cloud services unless an explicit, documented exception is made. See `06-PRIVACY-DATA.md` for the full policy and `02-DEVELOPMENT-RULES.md` Rule 12.

## Platform Constraints & Epistemic Rules

The platform's sensors and models produce evidence, not ground truth: monocular depth is inferred, ML outputs are probabilistic, sessions can be interrupted, the Tower is not always reachable, and camera visibility does not imply user attention. `07-PLATFORM-CONSTRAINTS.md` is the canonical, cross-module record of these limitations and their engineering mitigations, and defines the platform's epistemic rules (observation ≠ fact, inference ≠ measurement, absence of observation ≠ observation of absence, confidence must survive the pipeline, timestamps represent observation time). Every module must respect it; see `02-DEVELOPMENT-RULES.md` Rule 16.
