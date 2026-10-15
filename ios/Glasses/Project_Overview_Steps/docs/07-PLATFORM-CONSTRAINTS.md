# Platform Constraints & Epistemic Rules

## Purpose

This is the canonical, cross-platform record of what the Glasses platform can actually know, what it can only infer, what it cannot reliably know, and what engineering techniques mitigate each gap. It exists so that no module silently treats an inference as a measurement, a gap as a fact, or a workaround as a complete fix.

This document does not duplicate `06-PRIVACY-DATA.md` (data retention/deletion/transmission policy) or `05-DAT-INTEGRATION.md` (DAT API boundary/history) — it references them where relevant. Module docs should reference this document for cross-cutting constraints rather than restating them.

DAT-specific claims below were verified against `search_dat_docs` and the official `facebook/meta-wearables-dat-ios` 0.9.0 source/sample during this pass. Where something isn't confirmed by an actual source, it's marked as such rather than guessed.

## How to Use This Document

Every module doc should, where relevant, state its module-specific *consequence* of a limitation below (e.g., "Object Memory: report last-seen, not current location") and link back here rather than re-explaining the general rule.

---

## Core Platform Principles

These are architecture-wide epistemic rules. They apply to every module, not just the ones enumerated in the limitations below.

**1. Observation ≠ Fact.** A sensor/model observation is evidence about the world, not automatically ground truth.

**2. Inference ≠ Measurement.** ML-derived depth, object identity, semantic interpretation, and similar outputs must not be represented as equivalent to directly measured sensor values.

**3. Absence of Observation ≠ Observation of Absence.** If the glasses did not observe something, the system cannot conclude that it was absent or that an event did not occur.

**4. Confidence Must Survive the Pipeline.** Where an observation originates from probabilistic inference, uncertainty/confidence must not disappear simply because the observation is persisted, transmitted, fused, or consumed by another module.

**5. Timestamps Represent Observation Time.** Network arrival time, processing time, and observation/capture time must remain conceptually distinct.

---

## Workaround Classification

Every mitigation below is tagged with one of these, so a workaround is never mistaken for eliminating the underlying limitation:

- **MITIGATES** — reduces the limitation but cannot remove it.
- **COMPENSATES** — provides another source of information that partially fills the missing capability.
- **RECOVERS** — handles temporary failures/interruption; does not prevent them.
- **VALIDATES** — provides evidence/confidence through repeated or independent observation.
- **REQUIRES FUTURE HARDWARE/API** — cannot be meaningfully solved with the current platform.

---

## 1. Monocular Depth / Scale

**Limitation:** The current target glasses provide monocular RGB vision only. They must not be treated as providing direct metric depth such as LiDAR or stereo depth.

**Why it exists:** Hardware choice (current-generation Ray-Ban Meta glasses have no dedicated depth sensor).

**Affected modules:** World Builder (primary), Object Memory, Environmental Memory, Visual Q&A, Experimental CV.

**Mitigation** — hybrid pipeline, detailed in `docs/modules/WORLD-BUILD.md`:
```text
RGB frames -> keyframe selection -> visual SLAM/SfM -> ML monocular depth
  -> semantic/object understanding -> spatial fusion -> persistent world representation
```
Constraint sources: multi-view geometry, ML monocular depth, known-size object priors, floor/wall/ceiling plane estimation, approximate camera/head height, temporal consistency, future IMU data if DAT exposes it. Prefer metric-depth-capable ML models where practical — no specific model is selected here.

**Classification:** COMPENSATES (multi-view geometry, priors, fusion) + MITIGATES (ML monocular depth alone, which stays an estimate no matter how it's tuned).

**What it improves:** Usable relative geometry and approximate metric estimates for non-safety-critical purposes.

**What it does not solve:** ML depth is inferred depth, not LiDAR-equivalent measurement, regardless of pipeline sophistication. World Builder must eventually represent uncertainty/confidence on every depth-derived value. Do not use World Builder for centimeter-accurate or safety-critical distance measurement under current sensor assumptions.

**Confidence implications:** Every depth value carries model/method uncertainty; must be represented, not discarded (Core Principle 4).

**Mock Device Kit testable:** Yes, for pipeline/API integration (mock camera feed exercises the same `Camera`/`Stream` path).

**Physical-glasses validation required:** Yes, for real optical/depth-estimation accuracy — mock video/live-phone-camera feeds don't reproduce the real glasses' lens/sensor characteristics.

---

## 2. Camera Quality / Bandwidth

**Limitation:** DAT camera delivery must not be assumed to provide perfectly stable image quality, resolution, bandwidth, or frame timing.

**Why it exists (confirmed DAT 0.9.0 behavior):**
- Supported resolutions (`StreamingResolution`): `high` (720×1280), `medium` (504×896), `low` (360×640), all 9:16.
- Supported frame rates (`StreamConfiguration.frameRate`): one of `2, 7, 15, 24, 30` — a fixed discrete set.
- Supported codecs (`VideoCodec`): `.raw` (uncompressed; streaming pauses when the app backgrounds) or `.hvc1` (compressed HEVC; continues in background, per the 0.5.0 changelog).
- Link limitation: resolution and frame rate are constrained by the Bluetooth Classic connection between phone and glasses. DAT applies its own automatic adaptive ladder: it lowers resolution one step first (e.g. `high` → `medium`), then lowers frame rate if bandwidth remains constrained (never below 15 fps). Per-frame compression also adapts to available bandwidth independent of the reported resolution/frame-rate state — delivered image quality can look lower than the reported configuration suggests.
- No app-exposed bitrate/quality knob exists beyond codec/resolution/frameRate choice; the adaptive behavior is internal to DAT.

**Affected modules:** All camera-consuming modules (World Builder, Object Memory, Environmental Memory, Visual Q&A, Experimental CV).

**Mitigation:**
- Design downstream systems to tolerate variable quality; do not assume every frame carries equal information value.
- Keyframe selection (pick informative frames rather than processing everything uniformly).
- Quality-aware processing where useful (e.g., skip expensive inference on frames the adaptive ladder has visibly degraded).
- Future Tower processing may adapt to measured conditions once V0.7 sustained-streaming measurements exist (`03-ROADMAP.md`).

**Classification:** MITIGATES. Do not invent controls DAT does not expose — there is no app-side override of the adaptive ladder.

**What it improves:** Robustness of downstream processing under real-world variability.

**What it does not solve:** The underlying BLE-Classic bandwidth ceiling and DAT's internal adaptive behavior are not controllable from the app.

**Confidence implications:** Degraded-quality frames should lower downstream confidence, not be treated identically to full-quality frames.

**Mock Device Kit testable:** Partially — mock feeds don't reproduce real Bluetooth Classic bandwidth constraints or the adaptive ladder's real-world triggering conditions.

**Physical-glasses validation required:** Yes, for real link-quality behavior; this is explicitly a V0.7 measurement milestone, not yet performed.

---

## 3. Frame Drops / Backpressure

**Limitation:** The system must not require processing every camera frame.

**Why it exists:** No app-facing frame-queue-depth, drop-count, or backpressure API was found anywhere in DAT docs or the real sample (verified this session). DAT delivers frames via a push listener (`Stream.videoFramePublisher`) at whatever rate its internal adaptive layer settles on; there is no documented guarantee of complete delivery under adverse conditions.

**Affected modules:** All camera-consuming modules; most acute for real-time paths (Experimental CV live testing, future Accessibility) vs. history-building paths (World Builder, Environmental Memory).

**Mitigation:**
- Freshest-useful-frame semantics for real-time workloads (already the pattern used in `GlassesConnection`'s camera-proof code and in the official sample's `currentVideoFrame`, which is overwritten each callback — "latest wins", never queued).
- Bounded queues, stale-frame dropping — already stated platform-wide in `01-SYSTEM-ARCHITECTURE.md` Reliability Policies and `02-DEVELOPMENT-RULES.md` Rule 15.
- Keyframe selection for reconstruction/history workloads (a *different* mechanism — deliberately retaining selected frames, not dropping to latest-only).
- Separate real-time and asynchronous processing paths, per `docs/modules/WORLD-BUILD.md`'s Real-Time vs. Asynchronous section.

**Important distinction — do not conflate these two workload types:**
- Real-time CV (e.g. live Experimental CV testing) may prefer latest-frame-wins.
- World reconstruction/history workloads (World Builder, Environmental Memory) may deliberately retain selected keyframes rather than discarding stale ones — "stale for real-time display" is not the same as "irrelevant for history."

**Classification:** MITIGATES.

**What it improves:** Bounded latency/memory under sustained streaming; freshness for interactive use.

**What it does not solve:** Does not guarantee any specific frame reaches the app — DAT's own adaptive/link behavior can already reduce what's delivered before our backpressure logic ever runs.

**Confidence implications:** N/A directly, but dropped/undelivered frames mean gaps in observation history (see Limitation 7).

**Mock Device Kit testable:** Yes, for the app-side drop/queue logic itself.

**Physical-glasses validation required:** Yes, to observe real link-induced drops (distinct from app-induced drops).

---

## 4. Session / Sensor Interruptions

**Limitation:** DAT sessions can be interrupted by physical/device/network state, and the app must not assume session continuity.

**Confirmed DAT 0.9.0 behavior:**
- **Disconnect/fold:** Closing the hinges disconnects Bluetooth, stops active streams, and forces `DeviceSessionState` to `.stopped`.
- **Reconnect:** Opening the hinges restores Bluetooth when the glasses are nearby, but does **not** restart the device session — the app must explicitly start a new session once the device is available again.
- **Pause:** `DeviceSessionState.paused` keeps the connection alive; streams stop delivering data; the device resumes to `.started` automatically. The app should not try to restart a session while paused.
- **Competing sessions:** Confirmed — "Only one session can run on a device at a time." Another app or system feature starting a session, or a system gesture opening another experience, are documented causes of a `DeviceSessionState` transition (mostly to `.stopped`; some gestures pause-then-resume).
- **Doff:** Not separately documented beyond the fold/hinge behavior above; Mock Device Kit models `doff()`/`don()` as distinct simulated states, but the real-hardware doff-specific session effect (as opposed to fold) is not confirmed from available sources.
- **App backgrounding:** DAT does **not** handle this automatically. The official sample explicitly listens for `UIApplication.didEnterBackgroundNotification` and proactively ends the session, with a bounded (5s) error-suppression window afterward. `.hvc1` streaming *can* continue through backgrounding (0.5.0 changelog); `.raw` cannot. This is an app-level policy choice DAT permits either way, not a DAT requirement.
- **Bluetooth loss:** Falls under the general connectivity-drop causes above; leads to `.stopped`.

**Affected modules:** All DAT-dependent modules; `GlassesConnection` (the DAT adapter) is where this must be handled once, per `05-DAT-INTEGRATION.md`'s boundary rule.

**Mitigation:**
- Explicit lifecycle state machine mirroring `DeviceSessionState`/`StreamState` directly (already the pattern in `GlassesConnection`'s camera-proof code).
- Bounded reconnect/backoff (per `02-DEVELOPMENT-RULES.md` Rule 15) — not yet implemented for the camera/session path specifically as of this writing.
- Modules must tolerate sensor gaps rather than assuming continuous observation.
- Session restart must not imply continuity of observation — a new session is a new observation stream, not a resumption of the old one.
- Preserve gap information in persistent histories where relevant (ties to Limitation 7).

**Classification:** RECOVERS (reconnect/backoff, lifecycle state machine) — none of this prevents interruptions, only handles them.

**What it improves:** Predictable app behavior across interruptions; no silent hangs.

**What it does not solve:** Cannot prevent interruptions (physical/BLE/OS-level), cannot guarantee reconnection succeeds, cannot recover frames lost during a gap.

**Confidence implications:** A session gap should be recorded as a gap, not smoothed over (Core Principle 3).

**Mock Device Kit testable:** Yes — `fold()`/`unfold()`/`don()`/`doff()`/`powerOn()`/`powerOff()` simulate these states directly, already exercised in our mock device flow.

**Physical-glasses validation required:** Yes, for real Bluetooth-loss timing/frequency and any doff-specific behavior beyond what mock simulates.

---

## 5. Probabilistic ML Output

**Limitation:** Object detection, OCR, scene understanding, depth estimation, tracking, classification, and other ML outputs are probabilistic, not certain.

**Why it exists:** Inherent to ML inference; not DAT-specific.

**Affected modules:** Object Memory, Environmental Memory, Visual Q&A, World Builder, Experimental CV, future Accessibility.

**Mitigation:**
- Confidence scores attached to outputs.
- Temporal confirmation / multi-frame consensus (prefer repeated observation over single-frame inference — already required in `docs/modules/WORLD-BUILD.md`).
- Cross-model validation where justified.
- Geometric consistency checks.
- Thresholding appropriate to module risk (a higher bar for Accessibility than for Experimental CV).
- Ability to return UNKNOWN / INSUFFICIENT EVIDENCE rather than a forced answer.

**Classification:** VALIDATES (temporal/cross-model/geometric confirmation).

**What it improves:** Reduces false positives/confident wrong answers when consumers respect the confidence signal.

**What it does not solve:** Does not make any single inference certain. Do not create a requirement that every module must always produce an answer — UNKNOWN is a legitimate, required output.

**Confidence implications:** This entire limitation *is* the confidence-implication; see Core Principle 4.

**Mock Device Kit testable:** Yes, for pipeline plumbing; not for real-world model accuracy (mock feeds are either synthetic/prerecorded or the phone's own camera, not the target glasses optics).

**Physical-glasses validation required:** Yes, for real accuracy figures.

---

## 6. Object Identity vs. Object Class

**Limitation:** Detecting "a backpack" does not establish that it is the same backpack observed previously.

**Affected modules:** Object Memory (primary), Environmental Memory.

**Mitigation:**
- Visual embeddings / re-identification.
- Appearance features, spatial continuity, temporal continuity, contextual cues.
- Repeated observations.
- Confidence-scored identity association rather than binary identity claims.

**Classification:** COMPENSATES (embeddings/context provide identity evidence) + VALIDATES (repeated observation raises confidence). No current technique fully solves general re-identification.

**Conceptual distinction to preserve exactly:**
> "black backpack detected" vs. "likely the same black backpack previously observed"

**What it improves:** Usefully probable identity association for retrieval-style queries ("where did I last see my backpack").

**What it does not solve:** Does not establish certain identity. Do not select a specific re-identification model as part of this documentation pass — that remains a later, separate decision. Persistent identity should be represented probabilistically unless strongly established.

**Confidence implications:** Identity claims must carry confidence exactly like any other inference (Core Principle 4).

**Mock Device Kit testable:** Yes, for pipeline integration; not for real re-identification accuracy.

**Physical-glasses validation required:** Yes, for real-world accuracy under varying lighting/angle/occlusion.

---

## 7. Environmental Memory / Observational Gaps

**Limitation:** The system must distinguish "last observed at X" from "is currently at X," and "not observed" from "confirmed absent."

**Affected modules:** Environmental Memory (primary), Object Memory.

**Mitigation:**
- Observation timestamps; last-seen semantics (never "current location" from a stale observation).
- Confidence decay where appropriate (older observations become less trustworthy as current-state evidence over time).
- Observation history retained, not collapsed into a single "latest fact."
- Explicit unknown state, distinct from a negative/absence state.
- Repeated confirmation before treating a state as stable.

**Classification:** MITIGATES (decay/history reduce the practical impact of staleness, but cannot make a stale observation current).

**What it improves:** Prevents the module from confidently asserting something false about present state.

**What it does not solve:** Cannot convert an old observation into current knowledge. Cannot conclude absence from lack of observation (Core Principle 3) — the user may have simply not walked past the object again.

**Confidence implications:** This limitation is fundamentally about confidence decay over time — must never silently convert stale observations into current facts.

**Mock Device Kit testable:** Yes, for the state-machine/data-model logic.

**Physical-glasses validation required:** No — this is an app/data-model concern, not a hardware behavior; mock sessions exercise it fully.

---

## 8. Camera FOV vs. Human Attention

**Limitation:** Something appearing in the glasses camera does not prove the user looked directly at it, noticed it, read it, understood it, or interacted with it.

**Affected modules:** Object Memory, Environmental Memory, Visual Q&A, future personal-memory features.

**Mitigation:**
- Describe camera-derived events as "observed by the system," never "seen by the user," without additional evidence.
- Do not assume eye tracking exists on the current target hardware — none is confirmed available.
- Future attention/gaze signals may improve this only if actual hardware/API support exists; this is aspirational, not planned.

**Classification:** REQUIRES FUTURE HARDWARE/API for genuine attention detection. Current mitigation is purely linguistic/labeling discipline, not a technical fix.

**What it improves:** Prevents the platform from overclaiming what the user actually perceived.

**What it does not solve:** Nothing about camera FOV can currently establish attention; this is not mitigated by better CV, only by different sensors that don't exist on current hardware.

**Confidence implications:** "Observed by system" carries no attention-confidence at all — it is a distinct axis from detection confidence, not a lower version of it.

**Mock Device Kit testable:** N/A — this is a labeling/representation discipline, not a testable behavior.

**Physical-glasses validation required:** N/A for current mitigation; would apply to any future gaze-hardware validation.

---

## 9. Timestamp / Network Latency

**Limitation:** Tower receipt time must not be treated as camera capture time.

**RESOLVED 2026-08-26 by direct measurement — the PTS is a capture-side timestamp.** DAT's documentation still says nothing about it, so this was measured rather than read: 1,084 frames from the real Ray-Bans over 45 s, timestamps sampled on DAT's callback thread *before* the main-actor hop, compared against `mach_absolute_time`.

| | mean | **sd** | min | max |
|---|---|---|---|---|
| `d_pts` (PTS deltas) | 0.041666 s | **0.00238** | 0.03332 | 0.05006 |
| `d_host` (arrival deltas) | 0.041598 s | **0.01684** | 0.00246 | 0.12006 |
| residual (`d_pts − d_host`) | 0.000068 s | **0.01689** | −0.07839 | 0.03927 |

The argument is the jitter, not the offset. A stable offset proves nothing — a phone-side stamp applied on arrival produces one too. What separates the two is that an arrival stamp *inherits* transport delay, so `d_pts` would track `d_host` and the residual would collapse to ~0. Instead **residual_sd / d_host_sd = 1.003**: the residual is entirely arrival jitter and the PTS carries none of it. **`d_pts_sd / d_host_sd = 0.141`** — PTS deltas sit on a tight grid at exactly 1/24 s (24.000 fps) while arrivals scatter from 2.5 ms bursts to 120 ms stalls. A clock that stays regular while delivery is irregular is upstream of the delivery.

The epoch says the same thing independently: `pts_timescale = 1000000` (microseconds), `pts_epoch = 0`, and the first frame read **424.72 s** against a host uptime of **519,597 s**. Not the phone's clock, and already ~7 minutes into its own epoch when the stream opened — so not stream-relative-from-zero either.

**What this does and does not license.**

- **Do** use the PTS for *relative* ordering and inter-frame intervals within one stream session. It is the only regular time base available and it is clean.
- **Do not** treat it as comparable to any phone or Tower clock without an offset estimate. The offset is ~519,172.68 s and is a property of this boot pair, not a constant.
- **Do not** assume it survives a reconnect. Whether the epoch is device-persistent or per-session is **not** established (see Unresolved Questions). World Builder chains captures across reconnects, so this matters before any use there.
- **Drift is not established.** The least-squares slope wandered −4772 → −662 → +701 → +167 ppm as the window grew — noise, not convergence, since the offset spread (132 ms) is dominated by arrival jitter. The offset *mean* held within ~8 ms across nine windows, so the clocks are closely rate-matched, but no ppm figure should be quoted from a 45 s run.

Measured with `ios/Glasses/FramePTSProbe.swift` (DEBUG-only, pure observation). Delete that file to remove the experiment.

### 9.1 The capture clock stops when the camera does — measured 2026-08-26

**Do not use the PTS to measure elapsed time across a stream stop.** It is
monotonic across one, which is exactly what makes this dangerous: nothing
looks broken.

| Event | wall gap | clock advanced | ratio |
|---|---|---|---|
| pause → resume | 17.86 s | 17.80 s | 99.7 % |
| pause → resume | 39.165 s | 39.165 s | 100.0 % |
| **stop → start** | **25.95 s** | **7.19 s** | **28 %** |
| **stop → start** | **192.70 s** | **~10.95 s** | **5.7 %** |

A *pause* keeps the camera subsystem alive and the clock runs through it,
matching wall time to ~5 ppm. A *stop* tears the subsystem down and the clock
freezes. Note the two stop rows: the gaps differ by 7.4x and the advance
barely moves (7.19 s vs ~10.95 s). What survives is a fixed teardown-and-
startup tail at each end — **not elapsed time, and not proportional to it.**

The epoch itself is durable: it does not reset on a stop, and it does not
reset when the *app process* restarts (observed continuing across a rebuild
and relaunch). It belongs to the glasses or to the DAT daemon, not to the app
or to the stream session.

**Consequence for World Builder.** The Tower chains captures across a
reconnect into one lineage. If any consumer ever derives a duration, a rate,
or a gap from PTS across that boundary, it will silently understate the
dropout by an unbounded amount — a five-minute WiFi outage would read as about
ten seconds. Elapsed time across a reconnect must come from the Tower's own
receipt clock, which is what `time_basis: "tower-receipt"` already records.

**Safe uses:** ordering and inter-frame intervals *within* one continuous
streaming session. **Unsafe:** anything spanning a stop.

**Affected modules:** World Builder, Object Memory (temporal memory), Environmental Memory, any tracking/multi-frame reasoning.

**Mitigation (future transport protocol, not designed here):**
- Preserve a capture/observation timestamp where available.
- Preserve a sequence identifier.
- Preserve Tower receive timestamp separately, if useful.

This is a requirement statement for the future frame-transport protocol (explicitly out of scope for the current camera-proof milestone), not an implemented mechanism.

**2026-08-19 V0.7 finding — the single `seq` field is already insufficient.** The current wire protocol's `seq` field is, in the deployed iOS sender, the DAT/source capture-frame index (`frameCount` of incoming VideoFrames) — but the sender only forwards roughly 1-in-30 of them (a throttled capture -> transmit branch), so the Tower normally receives `seq` like 1, 30, 60, 90, ... by design. Under the current single-field protocol this is indistinguishable from network/transit loss or a sender-side drop — all three look identical as a gap in `seq`. A future frame-transport protocol revision should carry two distinct fields instead of one:
- `source_seq`: the original DAT/capture frame index (what `seq` currently is).
- `tx_seq`: a dense counter incremented once per frame message actually transmitted by the iPhone, independent of `source_seq`.

With that split, gaps in `source_seq` would describe intentional source sampling/selection (expected, not an error), while gaps in `tx_seq` would identify frame messages the phone attempted to send but the Tower never received (genuine transit loss). Not implemented as of V0.7 — V0.7 deliberately made no wire-protocol change; see the Tower's `seq_gap_total` metric, which explicitly cannot attribute cause under the current single-field protocol, and `03-ROADMAP.md` V0.7.

**2026-08-20/21 status update — Tower half done, iOS half outstanding.** The Tower now *accepts* optional `source_seq`/`tx_seq` fields and reports `tx_seq_gap_total` (commit `0346c05`), but the iOS sender does not yet send them, so `tx_seq_gap_total` still reads `null` and **genuine transit loss remains formally unmeasured on this platform**.

Separately, the Tower can now estimate the two quantities it *can* derive without any protocol change: `sampling_stride_avg` (capture frames elapsed per frame received) and `source_fps_estimate` (the upstream capture rate being sampled), both computed from `source_seq` with its fallback to `seq`. These do not distinguish sampling from loss — only the `tx_seq` split can do that — but they do remove the need to infer the *sender's throttle* by hand from `seq_gap_total`. Added 2026-08-21 after the first physical-glasses remote run required exactly that arithmetic.

**2026-08-21 finding — the 1-in-30 stride is the platform's observation-rate bottleneck, and it is rate-independent of the transport.** The first physical-glasses run delivered 0.8 fps over a remote Tailscale path; the 2026-08-19 LAN soak delivered 0.81 fps. A fixed decimation stride bounds delivered rate to capture-rate ÷ stride regardless of available bandwidth, so improving the link cannot improve the observation rate. Note also that the "1-in-30" mechanism itself has still never been read off the Swift source (Rule 4) — the wire arithmetic confirms the *effective* stride is ~30 but not whether it is implemented as a frame-count stride or a timer.

**Classification:** MITIGATES (once implemented) — cannot fully eliminate latency, only make its effect measurable/correctable.

**What it improves (once implemented):** Enables correct temporal ordering and latency-aware reasoning.

**What it does not solve:** Does not eliminate latency; does not resolve the open question of what DAT's own timestamp actually represents.

**Confidence implications:** Temporal reasoning that conflates arrival time with capture time can silently corrupt ordering/duration inferences — must be kept distinct (Core Principle 5).

**Mock Device Kit testable:** Partially — mock frame delivery timing doesn't reproduce real BLE-Classic-to-WiFi-to-Tower latency chains.

**Physical-glasses validation required:** Yes, and additionally requires the (not-yet-built) Tower transport protocol to carry the right fields.

---

## 10. Tower Availability

**Limitation:** Tower-dependent capabilities must not be assumed continuously available.

**Possible causes:** LAN loss, future Tailscale/WireGuard loss (Phase 1.5), Tower offline, WebSocket failure, Tower process failure, GPU workload failure.

**Affected modules:** Every Tower-dependent module (i.e., all CV/AI modules); already partially reflected in `TowerClient`'s truthful `.offline/.connecting/.online/.failed` state.

**Mitigation:**
- Explicit Tower state (already implemented in `TowerClient`).
- Bounded reconnect/backoff (per `02-DEVELOPMENT-RULES.md` Rule 15 — not yet implemented for `TowerClient` specifically; it currently does a single bounded-timeout connection attempt, not automatic reconnection).
- Module capability state reflecting Tower dependency.
- Graceful degradation; do not advertise Tower-backed functionality as available when the Tower is unavailable.
- Keep iPhone responsibilities minimal but sufficient for safe lifecycle/transport handling — do not implement offline AI fallback unless a module specifically requires it later.

**Classification:** RECOVERS (reconnect/backoff) + MITIGATES (graceful degradation). Neither prevents Tower unavailability.

**What it improves:** Predictable, truthful behavior when the Tower is unreachable; no fabricated results (already a platform rule — `02-DEVELOPMENT-RULES.md` Rule 3).

**What it does not solve:** Cannot make the Tower available; cannot substitute for it locally unless a module has an explicit, justified reason to.

**Confidence implications:** N/A directly — this is an availability concern, not an inference-confidence one, though a module that silently falls back to stale cached results would violate Core Principle 1.

**Mock Device Kit testable:** N/A — Tower availability is independent of Mock Device Kit; already tested via `TowerClient`'s ping/pong milestone.

**Physical-glasses validation required:** No — this is a networking concern, not a glasses-hardware one.

---

## 11. Network Security / Privacy

**Limitation:** The current V0.5 LAN transport (`ws://172.16.60.232:8000/ws`) is development infrastructure, not a production security posture.

**Must be documented clearly:**
- Plaintext HTTP/`ws://` development transport is **not** the final security posture.
- Current port exposure should not be treated as production-ready.
- No real sensitive camera data should be considered adequately protected merely because the system is on a LAN.

**Affected modules:** All Tower-transport-dependent modules; all camera/CV modules once real frame data flows to the Tower.

**Planned mitigation (not designed in detail here):**
- Secure private networking — the already-planned Tailscale/WireGuard-style Phase 1.5 path (`03-ROADMAP.md`).
- Restricted firewall scope.
- Authentication before sensitive/production usage.
- Encrypted transport where appropriate.
- Data minimization (`06-PRIVACY-DATA.md`).
- Existing `06-PRIVACY-DATA.md` policies govern data handling regardless of transport security state.

**Classification:** REQUIRES FUTURE HARDWARE/API is not quite right here — this is better described as **planned-but-not-yet-implemented MITIGATES**. The final authentication protocol is explicitly not designed as part of this pass.

**What it improves (once implemented):** Meaningful protection against unauthorized access on untrusted networks.

**What it does not solve today:** Nothing — today's transport has no authentication or encryption. Treat all current development traffic as unprotected.

**Confidence implications:** N/A — this is a security concern, not an inference-confidence one.

**Mock Device Kit testable:** Yes — transport security is independent of Mock Device Kit vs. real glasses.

**Physical-glasses validation required:** No.

---

## 12. Mock Device Kit vs. Real Hardware

**Limitation:** Mock Device Kit validates API integration and application behavior, but does not prove real-world hardware performance.

**Mock Device Kit can validate:**
- DAT API integration.
- Session state logic (`DeviceSessionState`/`StreamState` transitions).
- Camera pipeline structure (`addCamera`/`Stream`/`videoFramePublisher`).
- Mock device lifecycle (pair/power/don/doff/fold/unfold).
- Application UI/state.
- Tower transport using generated/live-phone-camera input.

**Physical glasses are still required to validate:**
- Real optics/image characteristics.
- Motion blur.
- Real RF behavior.
- Bluetooth/Wi-Fi reliability.
- Real latency/throughput.
- Thermal behavior.
- Battery impact.
- Firmware compatibility.
- Real fold/doff behavior (beyond what Mock Device Kit's simulated states model).
- Real environmental conditions.

**Affected modules:** All of them — this is a testing-methodology constraint, not a module-specific one.

**Mitigation:** Maintain an explicit distinction between **"Mock-validated"** and **"Hardware-validated"** in future milestone/testing documentation. A capability that has only been Mock-validated must not be reported as working on real hardware.

**Classification:** COMPENSATES (Mock Device Kit provides a real, useful substitute for API/logic testing) — it does not and cannot substitute for hardware validation.

**What it improves:** Fast, hardware-independent iteration on everything except real device physics.

**What it does not solve:** Cannot prove real-world performance, reliability, or accuracy claims.

**Confidence implications:** N/A directly, but conflating Mock-validated with Hardware-validated is itself a Core-Principle-1 violation (treating a simulated observation as equivalent to a real one).

**Mock Device Kit testable:** This limitation is *about* Mock Device Kit's own boundary.

**Physical-glasses validation required:** Yes, for everything in the second list above — by definition.

---

## 13. Audio Is a Separate Sensor Path

**Limitation:** Camera access through `MWDATCamera` does not automatically provide microphone audio.

**Confirmed DAT 0.9.0 architecture:**
- Microphone capture uses the Bluetooth **HFP** (Hands-Free Profile) via standard iOS `AVFoundation` (`AVAudioSession`, `AVAudioEngine`) — **not** a DAT SDK call.
- **A2DP** (output-only, high quality) and **HFP** (bidirectional, 8kHz mono, beamformed) are mutually exclusive — activating HFP switches the glasses away from A2DP for the session.
- **Documented ordering constraint:** add the DAT camera stream to the session first, then configure/start HFP and wait for the route to settle, *then* start the DAT stream. Starting the DAT stream before HFP is ready can cause the audio route to fail silently.

**Affected modules:** Any module assuming synchronized audio (Visual Q&A's voice-query path, any future voice-interaction feature).

**Mitigation:** N/A — this isn't a limitation to mitigate so much as an architecture fact to respect: treat audio as an independent capability with its own permission flow, its own AVFoundation-based implementation, and its own ordering constraint relative to camera streaming.

**Classification:** N/A (informational/architectural, not a gap being worked around).

**What it improves:** N/A.

**What it does not solve:** N/A. Do not implement audio now. Do not let future modules assume camera streaming implies synchronized microphone availability — the two are independent capabilities that must both be explicitly established.

**Confidence implications:** N/A.

**Mock Device Kit testable:** Not yet exercised in this codebase; Mock Device Kit does model mock audio-adjacent device state but our implementation doesn't touch audio yet.

**Physical-glasses validation required:** Yes, once audio is implemented — HFP route behavior is real-hardware/real-Bluetooth-stack dependent.

---

## 14. World Builder: Real-Time vs. Async

Fully documented in `docs/modules/WORLD-BUILD.md` ("Real-Time vs. Asynchronous Processing"). Summary for cross-reference:

```text
LIVE:  camera -> lightweight tracking/pose estimation -> keyframe selection
ASYNC (Tower): keyframes -> depth estimation -> SfM/SLAM refinement -> map optimization -> persistent reconstruction
```

**Classification:** MITIGATES (keeps the iPhone/glasses lightweight; doesn't reduce reconstruction cost, just relocates and defers it).

No specific SLAM, SfM, Gaussian Splatting, or depth-model implementation is selected here or in the module doc.

---

## 15. Sensor Authority / Provenance

**General architectural requirement:** Future observations should retain enough provenance to answer:
- Which sensor produced this?
- When was it observed?
- Was it measured or inferred?
- Which module/model produced an inference?
- What confidence existed at creation time?
- Was it later fused with other evidence?

**Affected modules:** All modules with persistent observation data (World Builder, Object Memory, Environmental Memory; Experimental CV for its own datasets).

**Mitigation:** This is a requirement for future data models, not a schema. No database schema is designed here.

**Classification:** MITIGATES future auditability/debuggability gaps; does not itself solve any of Limitations 1–14 — it's what makes it possible to *tell* whether they were handled correctly after the fact.

**What it improves:** Debuggability, auditability, and the ability to later re-derive confidence if fusion logic changes.

**What it does not solve:** Nothing on its own — it's a record-keeping requirement, not a capability.

**Confidence implications:** This requirement exists specifically to serve Core Principle 4 (confidence must survive the pipeline) and Core Principle 1 (observation ≠ fact) — provenance is how a consumer can tell which principle applies to a given stored value.

**Mock Device Kit testable:** Yes, for schema/plumbing once implemented.

**Physical-glasses validation required:** No — this is a data-model concern.

---

## Unresolved Questions Requiring Future Investigation

- **~~`VideoFrame`/`CMSampleBuffer` timestamp semantics~~ — ANSWERED 2026-08-26.** It is a capture-side clock; the measurement and its limits are in Limitation 9. **Two narrower questions remain open:**
  - **~~Is the PTS epoch device-persistent or per-stream-session?~~ — ANSWERED 2026-08-26. Neither.** See Limitation 9.1: the clock free-runs through a *pause* but freezes for most of a *stop*, so it stays monotonic across a reconnect while understating the gap. **This is the one to know about before using PTS for anything spanning a dropout.**
  - **~~What is the actual clock drift?~~ — BOUNDED 2026-08-26 at ~5 ppm.** Not from the 45 s stream, where arrival jitter swamped it, but from a clean bracket across a 39.165 s pause: the two clocks disagreed by 0.19 ms. The earlier three-figure ppm values were jitter, not drift.
  - **Still open: does the clock survive the glasses powering off?** Every measurement so far kept them awake. A cold boot would distinguish "camera-subsystem uptime" from "device uptime" and would say whether the epoch is stable for a whole wearing session.
- **Doff-specific session behavior** (Limitation 4): DAT's documented session-interruption causes cover fold/hinge-close explicitly; whether doff (without folding) has a distinct, separately-documented effect on `DeviceSessionState` versus what Mock Device Kit's `doff()` simulates is not confirmed.
- **Real BLE-Classic bandwidth figures** (Limitation 2): the adaptive ladder's behavior is documented qualitatively; actual achieved FPS/resolution/latency under real conditions is explicitly a V0.7 measurement milestone, not yet performed.

## What Cannot Be Meaningfully Mitigated With Current Hardware

- **Camera FOV vs. human attention** (Limitation 8) — no current or near-term hardware/API path exists; this is a labeling discipline, not an engineering fix.
- **True metric depth accuracy for safety-critical use** (Limitation 1) — monocular inference, however fused, is not a substitute for depth-sensing hardware the current glasses don't have.
- **Real Bluetooth Classic bandwidth ceiling** (Limitation 2) — app-side logic cannot exceed the physical link's throughput.
