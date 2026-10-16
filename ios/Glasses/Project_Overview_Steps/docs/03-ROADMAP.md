# Roadmap

The roadmap is intentionally incremental. Complete and verify each milestone before expanding scope.

## Phase 0 — iOS Foundation

### V0.1 — Control Plane Skeleton (Complete)
- SwiftUI dashboard.
- `GlassesConnection` abstraction.
- `StreamManager`.
- `TowerClient`.
- `ProjectManager`.
- Real disconnected/offline/stopped states.
- Clean Xcode build.

Exit criterion: app builds and runs without fake functionality. **Met** — verified with an `xcodebuild` run against the iOS Simulator.

### V0.2 — DAT Development Environment
- Meta Wearables MCP verified.
- Current DAT iOS setup retrieved from official docs.
- Official DAT coding guidance installed if appropriate.
- Mock Device Kit setup understood.
- Required Meta project configuration documented.

Known blocker: the current Xcode bundle identifier (`tv.lloyd-icloud.com.Glasses`) contains a dash. Current DAT documentation (verified via `search_dat_docs`) states the iOS Bundle ID does not support the `-` character. This must be corrected before DAT app registration. See `05-DAT-INTEGRATION.md` — Known Setup Blockers.

Exit criterion: repository is ready to integrate DAT using current official APIs.

### V0.3 — Mock/Device Connection
- Connect through supported DAT flow.
- Expose real connection state through `GlassesConnection`.
- Test with Mock Device Kit when physical glasses are unavailable.

Exit criterion: app can establish a supported wearable session.

### V0.4 — Camera Into Swift
- Receive camera data through DAT.
- Display/inspect real frames.
- Start/stop correctly.

Exit criterion: real or simulated wearable camera frames reach the iOS application.

## Phase 1 — Tower Transport

### V0.5 — Minimal Tower Receiver
- Establish a small server on the Windows tower.
- Initial stack (pinned): Python, FastAPI, WebSockets for real-time frame transport, and OpenCV; PyTorch is an optional `ml` extra required only by a model-backed CV experiment, and CUDA is used only when GPU acceleration is required. Do not introduce gRPC, C++, distributed infrastructure, message brokers, or container orchestration speculatively — revisit only if measurements expose a real limitation.
- LAN-only. No remote/internet exposure at this milestone — see Phase 1.5 for remote access.
- Secure development transport appropriate for a trusted local network (exact mechanism decided during implementation).
- iPhone discovers/connects to configured tower.
- Real online/unavailable state.

### V0.6 — First Frame to Tower
- Send one frame.
- Receive it.
- Process with OpenCV.
- Return a minimal acknowledgement/result.

### V0.7 — Sustained Streaming
Target experiment:
- approximately 504x896;
- approximately 15 FPS;
- 20–30 minute target session.

Measure actual FPS, dropped frames, latency, bandwidth, disconnects, battery behavior, thermal behavior, and tower utilization.

Do not claim targets as achieved until measured.

**2026-08-21 status — the ~15 FPS target is blocked in iOS, not in the Tower.** Measured received rate is 0.8 fps over LAN (2026-08-19) and 0.8 fps over a remote Tailscale path (2026-08-21), because the iOS sender forwards only ~1-in-30 DAT capture frames. The Tower sustains 33–40 fps on synthetic load with the heavier depth module and was >99.8% idle during the physical run, so no Tower-side work (queue, drop policy, adaptive streaming) is warranted by current measurements.

Backpressure policy: the pipeline must not accumulate an unbounded frame queue. Freshness generally matters more than processing every frame — when the tower/module cannot keep up, stale frames should normally be dropped rather than accumulating latency. Exact queue/drop mechanics are an implementation decision informed by these measurements, not a fixed constant.

Adaptive streaming (IDLE / TRACKING / HIGH_RATE style frame-rate adjustment, as originally sketched in the project brainstorm doc) is a candidate future optimization once V0.7 measurements exist. Do not implement adaptive streaming before this milestone produces real data to design it from.

## Phase 1.5 — Secure Remote Tower Access (Future Milestone)

Remote (non-LAN) tower access is a real project goal, not merely hypothetical — but it is explicitly deferred until the local pipeline (through V0.7) works reliably. This is a placeholder milestone; do not design the full implementation yet.

Target shape:

```text
Glasses -> iPhone -> cellular/remote network -> private secure connection -> home tower
```

Preferred approach: a private overlay/tunnel (e.g., Tailscale or WireGuard) rather than exposing the tower directly to the public internet.

Exit criterion (future): tower reachable from outside the LAN through an authenticated, encrypted, non-public-facing connection, with the same truthful online/unavailable state contract as local access.

**2026-08-21 status — partially exercised, exit criterion NOT met.** The first physical-glasses run reached the Tower over Tailscale from roughly two hours away, proving the transport shape above works end to end on real hardware. Two of the three exit conditions are still outstanding: the connection was **plaintext `ws://` with no authentication**, permitted only by a deliberately narrow iOS ATS development exception. That exception must stay narrow, and authentication/encryption remain required before this milestone can be claimed. What the run did establish is that remote operation does **not** degrade the observation rate — the same 0.8 fps was measured over LAN and over the remote path, because the limit is a sender-side sampling stride rather than the link (`07-PLATFORM-CONSTRAINTS.md` Limitation 9).

## Phase 2 — Module Runtime

Sequencing note: build only the minimal module infrastructure required to run one real module. Generalize into a dynamic registry/discovery system only once a second production module gives real requirements to generalize against. Do not build the full dynamic plugin system speculatively.

### V0.8 — Minimal Module Container ("Registry of One")
- Tower runtime hosts exactly one hardcoded module slot.
- Preserve the module lifecycle abstraction (load/start/process/stop/unload; UNLOADED/LOADING/READY/ACTIVE/STOPPING/FAILED) so a real module can be swapped later without rewriting the runtime.
- No dynamic discovery, no descriptor negotiation protocol, no module registry yet.

Exit criterion: the tower runtime can load, run, and cleanly tear down one module through the full defined lifecycle.

### V0.9 — Experimental CV Lab (Module #1)
- Implement Experimental CV Lab as the module running in the V0.8 container.
- Prove the complete pipeline: DAT connection -> camera frames in Swift -> tower transport -> Experimental CV Lab -> one bounded CV experiment -> measurements.
- Use it as the sandbox for course CV experiments.
- Validate the module lifecycle/descriptor contract against real implementation experience before generalizing it.
- Where relevant, use these experiments to gather the profiling data that would justify any future GPU-acceleration technology adoption (see `01-SYSTEM-ARCHITECTURE.md` — GPU / Acceleration Strategy, and `docs/modules/EXPERIMENTAL-CV.md` — GPU / Acceleration Benchmarking). Do not adopt TensorRT, CV-CUDA, DeepStream, or similar before this milestone produces the measurements to justify one.

Exit criterion: at least one bounded CV experiment runs end-to-end from glasses/mock device input through to measured results.

### V0.9.1 — Depth CV Baseline (complete)
- MiDaS-small monocular depth as the second Experimental CV Lab experiment, and the first module holding a resource across frames (`tower/modules/depth_cv.py`).
- CPU-versus-CUDA baseline measured on the RTX 5070 rather than assumed (`scripts/depth_benchmark.py`, 60 synthetic 504x896 frames per device). CUDA beat CPU on every steady-state latency metric, but only modestly (~31% on the client-side average round trip), because MiDaS-small at that size is already light on CPU.
- Established that depth output is **relative inverse depth, not metric distance**, and must be labelled as inference rather than measurement.

This milestone was executed and reported but never given a heading here; recorded retrospectively on 2026-08-22.

Exit criterion: a measured GPU-versus-CPU comparison exists for a real workload, so any future acceleration decision is made from data.

### V0.9.2 — Backend Truthfulness & Reliability Hardening (complete)
- Client-facing `frame_error` WebSocket message on frame-skip and module-unavailable, so a connected client no longer has to poll `GET /health` to learn why `frame_result` messages stopped (closes a standing Rule 3 truthful-state gap).
- Malformed non-JSON / non-dict WebSocket messages are logged and ignored instead of abruptly ending the connection.
- `httpx2` replaces `httpx` as the dev/test-client dependency, clearing the `StarletteDeprecationWarning` (note: `httpx` itself must remain installed — `huggingface_hub` requires it).

Two findings from the same audit were deliberately **not** fixed here because both need a product/architecture decision rather than a patch: the lifecycle timeout's inability to bound a *synchronous* blocking `_do_load()` (real today for `torch.hub.load()`), and the no-auth / `0.0.0.0` bind default (already scheduled for Phase 1.5).

Exit criterion: the Tower reports frame-level and module-level failures to the connected client, and malformed input cannot silently drop a session.

### V0.9.3 — World Builder Foundations, Experiments 1–2 (complete, dataset-based)
- `depth_temporal_consistency` and `feature_trackability` run as bounded offline experiments per `docs/modules/EXPERIMENTAL-CV.md`'s Success Criteria discipline.
- **Measured on a public head-mounted dataset clip, not on Ray-Ban Meta glasses.** Results are feasibility evidence only, under a hard acceptance gate: both experiments must be re-run on real DAT footage before any conclusion is used as validation.
- Baselines a later experiment must beat or select against (same public clip): an EMA with alpha 0.3 cuts depth flicker by 50.35% at about 180 ms lag (`mad_mean` 0.0355, the number to beat); 53.7% of consecutive frame pairs exceed ORB-SLAM's `R_H` 0.45 threshold (near-pure rotation or a dominant plane, degenerate for triangulation), falling to 41.2% at a stride of k=2 and 12.4% at k=5, while only 72% of k=5 pairs clear the inlier floor (the trade-off a pose experiment must choose its frame baseline from).
- Experiments 3 (`monocular_pose_feasibility`) and 4 (`depth_scale_fusion`) remain **blocked on DAT camera intrinsics**. **2026-08-22 update:** the *tooling* to obtain them now exists — `scripts/calibrate_charuco.py` produces a versioned, resolution-keyed `CameraIntrinsics` from board views, and the exact physical procedure is written down. What is still missing is the *values*, which require a real device and a printed board. The blocker's substance is unchanged; its shape is now "run the procedure" rather than "invent an approach".

Exit criterion: both experiments produce measured numbers with their validity scope explicitly bounded.

### V0.9.4 — World Builder V1 engine (complete, synthetic-only)
- A monocular mapping engine: create a world, run a session, evaluate and reject frames, select keyframes on measured information, reconstruct relative-scale geometry where the data supports it, persist everything, survive process restart, reload and inspect cold (`tower/world_builder/`).
- ChArUco camera calibration, and calibration-gated geometry: unknown intrinsics produce **no poses** rather than guessed ones.
- Incremental update stream (append-only event journal with a cursor) plus Tower-side live following, so a world can be watched as it is built from a separate process.
- A production-armable dataset recorder (`TOWER_CAPTURE_ROOT`), which is what makes the physical validation procedure executable.

**Deliberately NOT part of this milestone:** registration as a production module. That crosses the V1.0 registry boundary and the V1.1 lifecycle boundary, both of which remain untriggered/blocked, so a test pins non-registration rather than leaving it to memory.

Numbered V0.9.4 rather than V1.x deliberately: it is Experimental-CV-era work that stops at the same registry boundary as everything before it, and calling it V1.x would imply the registry generalisation below had happened.

Exit criterion: **met** for the engine — a world survives a process restart and can be inspected cold, every claim is labelled synthetic, and the integration boundary is documented and tested.

### V0.9.5 — Experimental CV Lab V1 (complete, synthetic-only)
- A real measurement channel: `ExperimentResult.metrics` (`name -> number`), additive on the wire and omitted when empty. This is the type four other cartridges' "there is no non-scalar result channel" complaint traced back to; it is the Lab's own type, so the Lab fixed it.
- One `Experiment` protocol (`load`/`run`/`release`) and a registry of factories, so a stateful experiment no longer costs a `Module` subclass. `tower/modules/depth_cv.py` was **deleted**: the refactor removed a class rather than adding one.
- Five new experiments, each with a named headline and measured cost: `frame_quality`, `feature_detection`, `optical_flow`, `redaction_impact`, `object_detection`.
- `scripts/cv_lab_benchmark.py` — every experiment at three resolutions, plus the sparse-versus-dense optical-flow A/B kept as the evidence that chose sparse.
- Isolation asserted by test: no experiment may import a cartridge, and none may persist anything.

Still **not** a second module: the Lab is the one registry slot it has always been. Nothing here crosses the V1.0 boundary.

Exit criterion: **met** — every experiment exposes a measurement checked against independent truth, and every cost is measured rather than assumed.

### V0.9.6 — Document Memory V1 (complete, synthetic-only)
- Observe a document being read without asking the wearer to photograph it: page-quad detection plus text-likeness (~2.6 ms/frame), dwell tracking, best-frame selection, perspective correction, then OCR on one or two frames per document (~1.2 s each).
- Persistent, purgeable, retention-bounded memory of the TEXT (not the pixels), with lexical BM25 retrieval by time, content and recency, and an explicit "insufficient evidence" refusal.
- OCR via EasyOCR behind a substitutable seam, as an optional `[ocr]` extra. `rapidocr_onnxruntime` was rejected because a dry-run showed it installs `opencv-python` alongside this project's `opencv-python-headless`.
- Named honestly throughout: the camera cannot establish attention (`07-PLATFORM-CONSTRAINTS.md` Limitation 8), so the record is an *observation*, never a *reading*.

**The milestone's most valuable output is a measurement that blocks it.** Word recall against known rendered text is 0.957–1.000 at 1280×720 and **0.429–0.810 at the 640×360 the glasses deliver**. Tilt barely matters; resolution dominates. Page *detection* still works at the delivered resolution — only *recognition* is starved. That is an iOS/DAT requirement no Tower work can satisfy.

Exit criterion: **met for the Tower half** — a rendered reading session becomes a searchable memory that refuses to answer questions it has no record of. Not met, and not meetable here, for reading at the delivered resolution.

### V0.9.7 — Scene Understanding V1 (complete, synthetic-only)
- A live structured view of what is around the wearer: detection, anonymous tracking, counts from CONFIRMED TRACKS, camera-relative positions and relationships, and a query layer that refuses rather than guesses.
- **Counting uses tracking**, measured rather than asserted: the count of 2 holds exactly through 0%, 10% and 20% detector dropout, and is correct 93.9% of the time at 40%.
- Coarse head orientation ("appears to be facing your direction") implemented but **OFF by default**: `keypointrcnn_resnet50_fpn` costs 43.4 ms per call on CUDA and 956.4 ms on CPU, and CPU is the default device. Against the measured 83.5 ms delivered frame interval that is 0.52x on CUDA and 11.5x on CPU. It runs at a cadence of 3 frames (~250 ms) with every estimate carrying its age, and it is never called gaze. The earlier "798 ms, 24x the detector, 2.5x the interval" figures were measured on CPU with synthetic input, named no device, and are withdrawn (2026-08-26).
- **Persists nothing**, enforced by test. This cartridge is the present; Environmental Memory is the past, and that boundary is now real.
- Relationships the evidence cannot support are refused with the measurement that would settle each one, rather than being silently absent.

Exit criterion: **met** — every asserted relationship is checked against geometry a test chose, every refused one explains itself, and the count survives detector dropout.

### V1.0 — Generalize Module Registry (When Justified)
- Triggered only once a second production module (a promoted Experimental CV Lab result, or another module such as Object Memory) creates real, concrete requirements.
- Build dynamic module discovery, the module registry, and descriptor/sensor-profile negotiation generalized from actual usage rather than speculative design.

Exit criterion: two or more modules coexist through the registry; adding a normal tower-only module does not require a new iOS release.

### V1.1 — Module Lifecycle Hardening
- activate / pause / stop / persist / unload / load / initialize / READY-FAILED / resume, exercised across real module switches;
- bounded timeouts on LOADING/STOPPING with a defined FAILED/unavailable transition on timeout (see `01-SYSTEM-ARCHITECTURE.md` — Reliability Policies).

### V1.2 — First Promoted Production Module
- Promote a successful Experimental CV Lab result (or implement Object Memory directly) into a dedicated module per the promotion path documented in `docs/modules/EXPERIMENTAL-CV.md`.

## Phase 3 — Advanced Modules

Candidates, each with its own specification under `docs/modules/`:
- Object Memory — data layer **built and tested**; detector, `Module` subclass and routes **blocked** on the synchronous-`_do_load()` decision gate. See `docs/modules/OBJECT-MEMORY.md`.
- World Build — engine **built** at V0.9.4 above; production registration blocked at the V1.0/V1.1 boundary. See `docs/modules/WORLD-BUILD.md`.
- Accessibility.
- Visual Q&A / Reading — comparatively heavy (STT + OCR/CV + multimodal reasoning + TTS); not an early starter module. Its OCR half now has a working, measured implementation to reuse in Document Memory (V0.9.6), which removes one of its four unknowns.
- Document Memory — **built** at V0.9.6 above; blocked on delivered camera resolution rather than on Tower work. See `docs/modules/DOCUMENT-MEMORY.md`.
- Environmental / Physical-World Search — highest privacy exposure of the current module set; requires the retention/deletion policy in `06-PRIVACY-DATA.md` to be actually implemented, not just documented, before real data collection begins. **Its live-state neighbour now exists** (Scene Understanding, V0.9.7), and deliberately persists nothing so that this decision lands here.
- Scene Understanding — **built** at V0.9.7 above. See `docs/modules/SCENE-UNDERSTANDING.md`.
- Translator — future low-latency conversational translation module; see `docs/modules/TRANSLATOR.md`. Not an early candidate — depends on a streaming ASR/MT/TTS pipeline and the latency-instrumentation work in `01-SYSTEM-ARCHITECTURE.md` that have not been built or measured yet.

## Future Research

Possible but explicitly outside V1:
- direct glasses-to-tower transport (bypassing the iPhone entirely);
- alternative wearable hardware;
- deeper sensor fusion;
- persistent shared world models;
- custom firmware/reverse engineering;
- Tower-optional / heterogeneous compute degradation (see `01-SYSTEM-ARCHITECTURE.md` — Heterogeneous Compute & Graceful Degradation);
- live world-state visualization for World Build on a PC or phone (see `docs/modules/WORLD-BUILD.md` — Live Visualization). **Partially delivered 2026-08-22:** the Tower half exists — a world can be built and followed live from a separate process. The viewer half is blocked on there being no transport for world data at all, not on the reconstruction.

These must not block the supported DAT-based platform.
