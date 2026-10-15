# iOS current architecture — handoff for Tower Claude

**Audience:** Tower Claude, working from a Tower-only checkout during the World
Builder backend perfection pass.

**Purpose:** so the Tower→iOS boundary is not designed blind. This describes the
**Swift that exists right now**, read directly from source — not the roadmap,
not `ios/docs/agent-handoffs/IOS-TO-TOWER.md` (which is a *requirements* document written from the iOS
side and is already stale about what Tower can do).

**Scope:** only what is implemented and relevant to Tower / World Builder
integration. Nothing here is a redesign, a proposal, or a schema.

Everything is labelled with one of three tags:

- **[NOW]** — implemented in Swift today.
- **[EXPECTED]** — not implemented, but the existing architecture is shaped for
  it; a Tower contract that contradicts it forces a rebuild or a compat layer.
- **[NOT]** — not implemented, do not assume, do not design against.

---

## 1. Source state

| Fact | Value |
|---|---|
| Monorepo HEAD | `a0ad693` ("updated agent rules"), 2026-08-23 |
| `ios/` subtree imported at | `22c5783` (`aa484af Add 'ios/' from commit …`) |
| Commits touching `ios/` since import | none — `ios/` is unchanged from the subtree import |
| Swift source size | ~12.9k lines across `ios/Glasses/**` |
| Test target | `ios/GlassesTests/` — `TowerClientTests` (38 tests), `SenderPipelineTests`, `ProductShellTests`, `MockTowerServer` |

Read that second-to-last row literally: **no iOS work has happened in this
monorepo.** The iOS side is exactly what it was at import.

---

## 2. Directory / file structure (only what matters here)

```
ios/Glasses/
  GlassesApp.swift            @main; Wearables.configure(); glasses:// URL callback
  ContentView.swift           root view, owns ProjectManager, cartridge switch
  ProjectManager.swift        THE integration point: owns every runtime object
  GlassesConnection.swift     the only type that talks to Meta DAT
  TowerClient.swift           the only type that talks to the Tower WebSocket
  SendWindow.swift            bounded outstanding-send set (rate limiter)
  FrameRateGate.swift         MonotonicClock + 12 fps selection gate
  SenderMetrics.swift         shared capture→transmit instrumentation
  StreamManager.swift         placeholder, 33 lines, does nothing (see §7)
  TowerConfiguration.swift    hardcoded ws:// endpoint
  DeviceHealth.swift          iPhone thermal/battery
  Presentation/StateDisplay.swift   enum → human strings
  Cartridges/
    Cartridge.swift                       static catalog (8 entries)
    Integration/CartridgeClient.swift     CartridgeClient + TowerCapabilities
    Integration/CartridgeAvailability.swift  CartridgeContract, availability, CartridgeFailure
    Integration/CartridgePhase.swift      shared coarse phase vocabulary
    Integration/Observation.swift         WorldScaleSemantics, provenance, ObservationTime
    Integration/VisualArtifact.swift
  Workspaces/
    CartridgeWorkspace.swift              closed enum of 4 workspaces
    WorldBuilder/
      WorldBuilderClient.swift            client protocol + Unavailable impl + view model
      WorldModel.swift                    ALL World Builder data types
      WorldBuilderWorkspaceView.swift     the screen (capture half is #if DEBUG)
      WorldCanvasView.swift               renders WorldModelState; WorldSummaryView
    SceneUnderstanding/ DocumentMemory/ ExperimentalCV/   same 3-file shape
  Views/                      ConnectionSheet, ShellStatusBar, DeveloperToolsView, Components/
```

---

## 3. Runtime object graph and ownership **[NOW]**

```
ContentView (@StateObject project: ProjectManager)   <- created exactly once per launch
└── ProjectManager
    ├── glassesConnection : GlassesConnection   (DAT boundary)
    ├── towerClient       : TowerClient         (WebSocket boundary)
    ├── senderMetrics     : SenderMetrics       (shared by both halves)
    ├── streamManager     : StreamManager       (inert placeholder)
    ├── deviceHealth      : DeviceHealth
    └── cartridgeClients  : CartridgeClients
          ├── worldBuilder        : any WorldBuilderClient        = UnavailableWorldBuilderClient()
          ├── experimentalCV      : any ExperimentalCVClient      = Unavailable…
          ├── documentMemory      : any DocumentMemoryClient      = Unavailable…
          └── sceneUnderstanding  : any SceneUnderstandingClient  = Unavailable…
```

Hard rules that are enforced in code and by tests:

- `GlassesConnection` and `TowerClient` **never reference each other**.
  `ProjectManager` bridges them with three Combine sinks (§5).
- Cartridge clients are owned by `ProjectManager`, **above** the workspace,
  because a workspace's `@StateObject` view model is destroyed on every
  cartridge switch. A Tower-backed client holding a subscription and an
  accumulated world must outlive that.
- View models (`WorldBuilderViewModel` etc.) hold **no** runtime references —
  no socket, no DAT object, no `TowerClient`. They hold a subscription to their
  client and nothing else. `TowerClientTests.testCartridgeViewModelsSendNothingToTheTower`
  and `testDiscardingCartridgeViewModelsDoesNotDisturbALiveStream` pin this.

---

## 4. `GlassesConnection` — responsibilities and lifecycle **[NOW]**

The only DAT-facing type. `@MainActor`, `ObservableObject`.

Always compiled: `registrationState`, `devices`, `cameraPermissionStatus`,
`errorMessage`, link-state observation.

`#if DEBUG` only: everything camera — `deviceSessionState`, `cameraStreamState`,
`frameCount`, `latestCapturedFrame`, `hasActiveDevice`, `glassesThermalLevel`,
`cameraStreamDidStart`/`cameraStreamDidStop` subjects, MockDeviceKit.

Lifecycle, in order:

1. `init` — seeds registration + device list synchronously, creates
   `AutoDeviceSelector` (must exist before any session), and starts three
   long-lived tasks: `registrationStateStream()`, `devicesStream()`,
   `activeDeviceStream()`. Logs `[Glasses][Init] GlassesConnection created`
   exactly once per launch; a second line means the object graph was torn down.
2. Camera permission is read opportunistically — on device availability and on
   link state reaching `.connected` — and the read is a **pure query**, never a
   prompt.
3. `startCameraSession()` — the only entry point to capture, reachable from
   exactly two buttons (Home and World Builder). Requires `hasActiveDevice`.
   Resets `frameCount = 0`, clears `latestCapturedFrame`, resets the frame-rate
   gate, calls `metrics.begin()`, then `createSession` + `session.start()`.
4. When the session reaches `.started`, `beginCameraStream` adds a camera with
   `StreamConfiguration(videoCodec: .raw, resolution: .low, frameRate: 24)` and
   starts the stream. Camera permission must already be `.granted`; if not, the
   session is torn down synchronously (`abandonSessionAfterFailedStart`).
5. Stream `.streaming` → `cameraStreamDidStart` fires (once per camera session).
6. Every DAT video frame increments `frameCount` → that ordinal is the frame's
   `seq`. The 12 fps `FrameRateGate` then decides whether the frame is
   forwarded; only selected frames are decoded (`makeUIImage`) and published as
   `latestCapturedFrame`.
7. `stopCameraSession()` → `camera.stop()`, `cameraStreamDidStop` fires,
   `metrics.finish()`, `deviceSession.stop()`.
8. A **DAT-initiated** stop (glasses folded/doffed, BT lost, stream error)
   reaches `cleanupCameraSession()` without passing through
   `stopCameraSession()`; that path also fires `cameraStreamDidStop`.

### Camera start/stop, stated for Tower

- Capture is **never** automatic. `ProjectManager.startAutomaticConnections()`
  opens the Tower socket and reads permission; it cannot start the camera, and
  even an open socket transmits nothing without a `stream_start`.
- Source rate is configured at 24 fps; **12 fps** is selected for transmission.
- `seq` is the **DAT callback ordinal**, incremented for every delivered frame
  whether or not it is transmitted. Therefore **`seq` gaps are normal and
  expected** — roughly every other value is missing at a 24 fps source.
- `seq` restarts at 1 on every `startCameraSession()`, and only there.

---

## 5. The bridges in `ProjectManager` **[NOW]**

Four Combine subscriptions, all `#if DEBUG`:

| Source | Action |
|---|---|
| `glassesConnection.$latestCapturedFrame` | `towerClient.sendFrame(image, width, height, sequence:)` |
| `glassesConnection.cameraStreamDidStart` | `towerClient.sendStreamStart()` + `deviceHealth.refresh()` |
| `glassesConnection.cameraStreamDidStop` | `towerClient.sendStreamStop()` |
| `towerClient.$status == .online` **and** `cameraStreamState == .streaming` | `towerClient.sendStreamStart()` — reopens the stream bracket after a reconnect |

That fourth row is the single most important lifecycle fact on this page. See
§9.3.

`ProjectManager` deliberately does **not** fan child `objectWillChange` into its
own publisher: the root view must not re-render at capture rate, because the
main actor is where send-window completions release their slots.

---

## 6. `TowerClient` — responsibilities **[NOW]**

`@MainActor`, `ObservableObject`, `URLSessionWebSocketDelegate`. Owns the app's
**only** network connection. 968 lines; the whole frame-send half is `#if DEBUG`.

Published state:

- `status: TowerStatus` = `.offline | .connecting | .online | .failed(String)`
- DEBUG only: `frameResultCount`, `latestFrameResult: TowerFrameResult?`,
  `isStreamingToTower`

Endpoint: `TowerConfiguration.webSocketURL` = `ws://100.110.156.55:8000/ws`,
hardcoded, plaintext, over Tailscale. There is no discovery, no config UI, no
persistence of the endpoint.

### 6.1 Wire vocabulary, exactly as implemented

Outbound (iOS → Tower), all **JSON text frames**:

| Message | Exact payload |
|---|---|
| ping | `{"type":"ping"}` |
| stream_start | `{"type":"stream_start"}` — exactly one key; pinned by `testStreamStartSendsExactPayloadOnce` |
| frame | `{"type":"frame","seq":Int,"width":Int,"height":Int,"format":"jpeg","data":"<base64 JPEG, quality 0.5>"}` |
| stream_stop | `{"type":"stream_stop"}` — exactly one key; pinned by `testStreamStopSendsExactPayloadOnce` |

Inbound (Tower → iOS), everything iOS understands:

| Message | Decoded fields |
|---|---|
| pong | handshake only — see 6.2 |
| `frame_result` | `seq` as `Int?`, `mean_intensity` as `Double?`, `processing_ms` as `Double?` — all optional, none fabricated |

That is the complete implemented vocabulary. There is no capability message, no
module selection, no session id, no world message, no ack, no error message
type, no binary channel.

### 6.2 Handshake — the sharpest landmine on this page

`validateConnection` does, in order:

1. `send({"type":"ping"})`
2. `await task.receive()` with a **6 second** timeout
3. requires the result to be `.string`, decodable as JSON, **castable to
   `[String: String]`**, with `json["type"] == "pong"`

Consequences Tower must respect:

- **The very first message the Tower sends must be the pong.** A banner, a
  `hello`, a capability advertisement, or a queued world update delivered before
  the pong fails validation → `.failed("Unexpected/malformed response from Tower")`
  → teardown → backoff reconnect → the same failure forever.
- **Every value in the pong object must be a JSON string.** `{"type":"pong",
  "server_time":1724371200}` or `{"type":"pong","modules":[…]}` fails the
  `[String: String]` cast. Extra *string* keys are tolerated and ignored.
- The pong must arrive within 6 s of the socket opening.

### 6.3 Receive loop and message routing

- One `receiveLoop` per connection, running for the connection's whole life. A
  `receive()` throw is the definitive "connection is gone" signal.
- **Non-text frames are dropped** with a log line. Binary/msgpack/protobuf
  payloads are invisible to iOS.
- The payload must be a **JSON object at the root** with a **`type` String**.
  A JSON array root (batching), NDJSON, or a missing `type` is dropped as
  "undecodable payload".
- `switch type { case "frame_result": … ; default: log("unknown message type") }`.
  **Unknown types are ignored, non-fatally** — a new Tower message type will not
  break or disconnect an existing iOS build; it will simply be discarded. Nothing
  routes inbound messages to cartridge clients today.
- Stale-connection guard: every inbound/failure path checks `isCurrent(task)`, so
  callbacks from a superseded socket cannot mutate current state.

### 6.4 Connection and reconnect behaviour

- `connect(to:)` — user-initiated. Refills the reconnect budget, tears down any
  existing connection first.
- `connectIfIdle(to:)` — automation-only (app launch). Guards on `.offline`, does
  **not** refill the budget. A Tower that has given up stays visibly failed.
- `disconnect()` — clears the reconnect intent before teardown.
- `autoReconnect` is `false` by default and **`true` in the production graph**.
- Backoff schedule: `[0.5, 1, 2, 4, 8]` seconds, then it gives up and stays
  `.failed` until a person taps Connect. Against a dead endpoint, giving up takes
  ~45 s (each attempt can burn the 6 s pong timeout).
- The budget is refilled only if the previous connection stayed `.online` for
  ≥ 30 s, measured *at the moment of failure*. A Tower that accepts a socket and
  immediately wedges will exhaust the schedule and stop — by design.
- Two independent drop detectors: the receive-loop error, and the
  `didCloseWith` delegate callback (which only fires for a real close frame).
- Teardown calls `session.invalidateAndCancel()`, resets the send window, and
  clears `isStreamingToTower` **without sending a `stream_stop`** — there is no
  socket left to send on.

### 6.5 Send path gates

`sendFrame` refuses, in this order, before doing any work:

1. `status != .online` → `sessionGateDrop`
2. `!isStreamingToTower` → `sessionGateDrop` (no `stream_start` on this socket)
3. send window stalled → **tear down and replace the connection**
4. send window full → `sendWindowDrop`, frame is dropped, never queued
5. JPEG/JSON encode failure → counted

`stream_start` / `stream_stop` deliberately **bypass the send window** — they are
two-byte payloads defining session boundaries, and dropping one corrupts frame
accounting on both sides.

---

## 7. `StreamManager` and `SendWindow` **[NOW]**

**`StreamManager` is inert.** 33 lines. `state` is permanently `.stopped`,
`metrics` permanently `nil`, `init()` empty. It is a placeholder for a future DAT
streaming abstraction and is **not** part of the Tower path. Ignore it entirely;
do not design anything that assumes it reports something.

**`SendWindow` is the pipeline's actual rate limiter**, a pure value type:

- `capacity = round(targetFPS × latencyBudget)` = `round(12 × 1/3)` = **4**.
- Steady-state achievable rate is `capacity / averageSlotLifetime`. A slot is held
  from `send()` until the completion handler's hop **back onto the main actor**.
- `isStalled` requires the window to be **full** *and* the oldest send to be older
  than `stallTimeout` = **2.0 s** → the connection is replaced, since
  `URLSessionWebSocketTask` cannot cancel one outstanding send.
- A stall verdict is skipped if the previous `sendFrame` was more than
  `mainActorGapAllowance` = 1.0 s ago, so a main-actor hitch is not misread as a
  wedged socket.

**What this means for Tower:** iOS applies backpressure by **dropping frames, never
queueing them**. If the Tower stops draining the socket — a slow handler, a
blocking write, a synchronous world-builder step on the receive path — iOS will
tear down and reconnect after 2 s. **A Tower-side per-frame handler that
occasionally blocks for >2 s will cause connection churn on the phone.** Keep
world-building work off the socket's read path.

---

## 8. World Builder on iOS **[NOW]**

### 8.1 The seam

```swift
@MainActor protocol WorldBuilderClient: CartridgeClient {
    var state: WorldModelState { get }
    var stateUpdates: AnyPublisher<WorldModelState, Never> { get }   // default: never emits
}
```

The only conformer is `UnavailableWorldBuilderClient`, `cartridgeID =
"world-build"`, whose `state` is a constant `.unsupported(reason:)`.

`WorldBuilderViewModel` seeds `@Published state` from `client.state` and
republishes `stateUpdates` on the main queue. It exposes
`availability(isTowerReachable:)`, `phase(isTowerReachable:)`, and
`unavailableExplanation(isTowerReachable:)`.

**A Tower-backed World Builder is an injection, not a redesign:** construct a new
`WorldBuilderClient` inside `CartridgeClients` (which is owned by
`ProjectManager`, which already owns `TowerClient`), and emit `WorldModelState`
values on `stateUpdates`. No view changes. That is the whole intended
integration, and it is the thing a bad backend contract would break.

### 8.2 `WorldModelState` — the state machine iOS already renders

```
.unsupported(reason: String)     // only reachable state today
.idle                            // builder exists, session not feeding it
.awaitingFirstUpdate             // frames going out, Tower has said nothing yet
.receiving(WorldSnapshot)        // Tower actively reporting
.finalizing(WorldSnapshot)       // capture ended, Tower still working; figures may change
.finalized(WorldSnapshot)        // final and inspectable
.failed(CartridgeFailure)
```

Projected to the shared `CartridgePhase`: `unsupported / idle / waiting /
live(receiving+finalizing) / settled(finalized) / failed`. The invariant
`!phase.mayCarryData ⇒ no payload` is asserted across all cartridges in
`CartridgeIntegrationTests` (`GlassesTests/ProductShellTests.swift`).

`.finalizing` vs `.finalized` is load-bearing: during finalisation the camera may
already be off, so the UI must not say "live", but the numbers on screen are not
yet the numbers that will be stored.

### 8.3 `WorldSnapshot` — the exact fields iOS can hold

Every field optional; absent means absent and is **not drawn** (not drawn as "—").

| Field | Type | Notes |
|---|---|---|
| `name` | `String?` | human-readable, optional |
| `worldID` | `String?` | opaque stable id |
| `keyframeCount` | `Int?` | "keyframes accepted into the reconstruction" |
| `revision` | `String?` | **opaque marker, compared for equality only** — never ordered, never parsed. Deliberately not an `Int`. |
| `tracking` | `WorldTrackingQuality` | `good / limited / lost / unavailable` — coarse on purpose, no percentage |
| `scale` | `WorldScaleSemantics` | `relative / inferredMetric / measuredMetric / unknown` |
| `mappingSeconds` | `TimeInterval?` | **the Tower's mapping clock**; iOS refuses to derive it from an iPhone timer |
| `calibration` | `WorldCalibrationState` | `unknown / uncalibrated / calibrating / calibrated` — no progress percentage exists |
| `geometry` | `WorldGeometryReport` | `representation: String?` (opaque label, displayed verbatim, **never branched on**), `elementCount: Int?`, `isIncremental: Bool?` |
| `trajectory` | `WorldTrajectoryReport` | `poseCount: Int?`, `pathLength: Double?`, `pathLengthUnit: String?`, `scale: WorldScaleSemantics` |
| `persistence` | `WorldPersistenceState` | `unknown / session / saved(revision: String?) / reloading` |

Display gates already implemented:

- `permitsMetricDisplay = calibration == .calibrated && (scale == .inferredMetric || .measuredMetric)`.
- `WorldTrajectoryReport.distanceDisplayable` is false for `.relative` and
  `.unknown` — a relative path length is **never** rendered as a distance.
- `ReportedFigure.format(value, unit:)` renders the number **bare** when the Tower
  named no unit. `inferredMetric` means "metric in kind", **not metres**.
- `WorldScaleSemantics.isEstimate` forces an "estimated, not measured" caption.

### 8.4 What the World Builder screen does today

`WorldBuilderWorkspaceView` has two halves: a **real** capture half (viewfinder +
Start/Stop capture, `#if DEBUG`) and a **world half** that currently renders the
unsupported panel. There is deliberately no "Start Mapping" button, no
placeholder metrics, no fabricated geometry, and no 3D framework linked
(no SceneKit/RealityKit/Metal).

`WorldCanvasView` consults **availability before state**: if
`CartridgeAvailability.forcedPhase` is non-nil, the shared `CartridgeStatePanel`
is drawn and `WorldModelState` is not consulted at all.

---

## 9. What the existing architecture expects of Tower

### 9.1 Capability declaration **[EXPECTED]**

`TowerCapabilities.declared: [String: CartridgeContract]` is an **empty local
table**, not a fetch. `supported: Set<String>` is empty.
`CartridgeIntegrationTests.testTheTowerDeclaresNoCartridgeContracts` (in `GlassesTests/ProductShellTests.swift`) asserts both.

`CartridgeContract` is `{ cartridgeID: String, identifier: String }` where
`identifier` is **opaque and compared for equality only** — iOS does not assume
integer versions, ordering, or backward compatibility. `resolve()` precedence is
fixed: *what the Tower can do outranks whether we can reach it*
(`noContract` → `unsupportedContract` → `towerUnreachable` → `available`).

Cartridge ids in the catalog: `experimental-cv`, `object-memory`, `visual-qa`,
`world-build`, `accessibility`, `environmental-memory`, `document-memory`,
`scene-understanding`. **World Builder's id is `world-build`.**

### 9.2 World / session identity **[NOW + EXPECTED]**

**[NOW]** iOS sends **no identity of any kind**. No session id, no world id, no
device id, no client id, no cartridge selection, no configuration. The only
session delimiter on the wire is the `stream_start` … `stream_stop` bracket, and
the only per-frame correlator is `seq`.

**[EXPECTED]** If a world needs an identity, **the Tower must mint it and report
it back** in `worldID`. iOS will hold it opaquely and echo nothing.

**[NOT]** Do not design a contract that requires iOS to supply a session id,
world id, or any field in `stream_start`. That payload is exactly `{"type":
"stream_start"}` and is pinned by a test; adding a required field means editing
the lifecycle-marker path that deliberately bypasses the send window.

### 9.3 Stream bracket semantics — invariants Tower must preserve

1. `stream_start` is sent **once per open bracket**, and is suppressed if a
   bracket is already open (`isStreamingToTower`).
2. `stream_start` is only marked as sent if it actually reached a socket. If the
   Tower is offline at capture start, no bracket opens and **every frame of that
   session is suppressed** until the socket comes back.
3. On a Tower reconnect while the camera is still streaming, `ProjectManager`
   **re-sends `stream_start` on the new socket**. So the Tower will see:
   `stream_start` → frames → *socket dies* → new socket → ping/pong →
   **`stream_start` again** → frames **continuing from the previous `seq`**.
   - **`seq` does NOT restart at 1 after every `stream_start`.**
   - A `stream_stop` **may never arrive** — a dropped socket produces no stop.
4. `seq` restarts at 1 only when a *camera* session starts.
5. Multiple start/stop cycles on one connection are supported and tested.

**Therefore:** Tower must not key a world's identity or lifetime on
"`stream_start` ⇒ new world, `seq == 1`". Treat a repeat `stream_start` on a new
connection as *resumption of an in-progress capture*, not necessarily a new
world; and treat an absent `stream_stop` (socket drop) as an ordinary,
recoverable end condition rather than an error state that poisons the world.

### 9.4 Mapping / build state **[EXPECTED]**

iOS needs a signal that is **distinct from "frames are arriving"**. Today it
cannot tell "the Tower received a frame" from "the Tower did something spatial
with it", and it must not conflate them. Concretely, iOS needs to be able to
reach `.awaitingFirstUpdate`, `.receiving`, `.finalizing`, `.finalized`, and
`.failed` from Tower-reported facts alone.

### 9.5 Keyframes / trajectory / geometry **[EXPECTED]**

- iOS holds **summary figures**, not arrays: `keyframeCount`, `poseCount`,
  `pathLength` + `pathLengthUnit`, `geometry.representation` +
  `geometry.elementCount`.
- **There is no pose schema on iOS and no renderer.** Pose arrays, point clouds,
  meshes, and landmark lists cannot be displayed and would be dropped. Sending
  them costs bandwidth on a link that already runs ~2 Mbit/s of JPEG.
- `geometry.representation` is displayed **verbatim** and never matched against a
  known set — pick a human-readable label.
- `geometry.isIncremental` exists precisely so iOS knows whether it is looking at
  a whole world or a delta. **iOS has no accumulation or merge layer**: a
  `WorldSnapshot` replaces the previous one wholesale.
- **Prefer self-contained snapshots** with a changed `revision`. Delta-only
  updates would require building accumulation state on the phone, which is
  exactly the "phone stays lightweight" rule the codebase is built on.

### 9.6 Calibration and scale semantics **[NOW, as display rules]**

The target glasses are **monocular RGB — no LiDAR, no stereo**. The rule is
absolute in `docs/modules/WORLD-BUILD.md` and enforced *at the type level* in
`Observation.swift`:

- Any monocularly inferred distance must be identifiable as an estimate wherever
  it is stored, displayed, or consumed.
- `.relative` is the honest default. `.inferredMetric` renders with a mandatory
  "estimated from a single camera" caption. `.measuredMetric` is **unreachable on
  current hardware**.
- `.unknown` scale ⇒ the figure is not shown as a distance at all.
- Calibration is four coarse states with **no progress percentage** — there is no
  field for one, deliberately, because no denominator has been defined.

**Do not send an unlabelled distance.** iOS will either render it bare (no unit)
or refuse to render it as a distance. Send `scale` and `pathLengthUnit` together
with any spatial figure, per-figure — `WorldTrajectoryReport` carries its own
`scale` separately from the snapshot's.

### 9.7 Persistence / reload **[EXPECTED]**

- **iOS stores no world data at all** and must not imply it could. The Tower owns
  persistence entirely (each module owns its storage namespace).
- `WorldPersistenceState` distinguishes *silence* (`.unknown`) from *"this session
  only"* (`.session`) — silence is **not** a promise that the world was discarded.
- `.saved(revision: String?)` — `revision` is opaque, equality only.
- `WorldInspectionMode` (`.live` / `.inspecting(worldID:)`) exists in the view
  model but is `private(set)` and **nothing can change it today**: there is no UI
  and no client method to open a stored world.

### 9.8 Error / stale / reconnect behaviour expected by iOS **[EXPECTED]**

`CartridgeFailure.Kind` is the vocabulary iOS already has:

| Kind | Meaning |
|---|---|
| `notSupported` | iOS refused locally — **never** attributed to the Tower |
| `towerReportedFailure` | the Tower said its module failed |
| `transport` | the connection dropped or the request never completed |
| `undecodableResponse` | the answer did not match the contract this build implements |
| `timedOut` | a bounded operation ran out of time |

`CartridgeFailure.message` is required and must be prose a person can read. On
failure the Tower must not "return stale or fabricated results" — iOS has a
`.failed` state precisely so it never has to render a failure as emptiness.

Staleness: iOS's only anti-stale mechanism is `revision` inequality plus the
`.receiving`/`.finalizing`/`.finalized` distinction. Nothing on the phone ages a
snapshot out. If a world becomes stale, **the Tower must say so** with a new
state, not by going silent.

---

## 10. How UI state is derived **[NOW]**

One rule, applied identically in all four cartridges:

```
phase = availability(isTowerReachable:).forcedPhase ?? state.phase
```

Availability outranks domain state, so an unreachable or uncontracted Tower can
never render as `.idle` (which would invite a user to press a button that cannot
work). `forcedPhase` maps `noContract`/`unsupportedContract` → `.unsupported` and
`towerUnreachable` → `.disconnected`; `.available` returns `nil` and lets the
cartridge's own state decide.

Explanation strings are composed once, in the shared layer
(`CartridgeAvailability.explanation(cartridgeName:clientReason:)`), never in a
view.

`isTowerReachable` is `tower.status == .online` — `.connecting` is **not**
reachable.

---

## 11. What transport logic belongs outside views **[NOW]**

All of it. Enforced by construction and by tests:

- Views receive **facts, not objects**. The three non-capture workspaces get a
  `Bool` from `TowerReachabilityReader`, which is the smallest thing that
  observes `TowerClient`. They are never handed `tower` or `glasses` — if they
  were, they would re-render at the Tower's ~12 Hz reply rate on the main actor,
  which is the actor the send window depends on.
- `WorldBuilderWorkspaceView` *does* observe both, because it draws the viewfinder
  and owns a capture button. Its body therefore runs at capture rate — anything
  expensive placed on that path costs sender throughput.
- `WorldBuilderViewModel.init(client:)` has **no default argument**, deliberately,
  so "construct a Tower-backed client here in the view" is not the path of least
  resistance.
- Nothing below `ContentView` constructs anything that talks to DAT or the socket.

**Implication for Tower:** any client that holds a subscription or accumulated
world state must be constructed in `CartridgeClients`. A contract that pushes
per-frame world updates into SwiftUI without coalescing will put main-actor work
at frame rate and directly reduce the frame rate reaching the Tower. **Coalesce
world updates to a few Hz at most.**

---

## 12. Privacy constraints **[NOW]**

- Full camera frames leave the phone: JPEG quality 0.5, base64, inside a JSON
  text frame, over **plaintext `ws://`** on a Tailscale path. Unauthenticated and
  unencrypted at the application layer.
- The phone performs **no** OCR, no detection, no recognition, and stores no
  world, document, or scene data.
- Documents/faces/screens appearing in frame are treated as *a standing risk of
  the input modality, not an edge case*.
- iOS deliberately makes **no privacy assurance about what the Tower stores** —
  it has no channel through which it could know. Every "unavailable" reason string
  is confined to what iOS can observe: what it sends, and the one brightness
  figure that comes back. If the Tower wants iOS to state a retention or
  redaction property, the Tower has to **report** it; iOS will not assume it.
- iOS will not display persisted imagery whose redaction treatment was not stated.
- "Observed by the system", never "seen by the user" — there is no gaze signal on
  this hardware. `ObservedDuration` is named for the camera, not the wearer.
- Three clocks stay distinct: observation time, arrival time, processing time.
  `ObservationTime.observedAt` is **never** filled in from an iOS `Date()`; a
  report with no Tower-supplied observation time renders as "time unknown".

---

## 13. Invariants Tower must preserve

1. First inbound message on a new socket is `{"type":"pong"}` with **string-only
   values**, within 6 s.
2. **Text frames only.** JSON **object** root. A `type` **String** key on every
   message.
3. `frame_result` keeps `seq` / `mean_intensity` / `processing_ms` (or iOS's
   round-trip evidence, `SenderMetrics.frameResults` and the Home tile, goes
   blank). New keys are ignored harmlessly; renaming or removing these breaks
   the only visible proof the pipeline works.
4. Never assume `seq` starts at 1 after a `stream_start`, nor that it is
   contiguous, nor that every `seq` gets a result.
5. Treat a repeated `stream_start` on a fresh connection as resumption.
6. Tolerate a missing `stream_stop`.
7. Do not block the socket read path for >2 s — iOS treats a stalled window as a
   wedged socket and replaces the connection.
8. Do not close the socket as a way of signalling anything: iOS turns a close into
   `.failed` and a bounded reconnect, and gives up after 5 attempts.
9. Every derived value carries provenance (measured vs. inferred, plus confidence
   where one exists); every figure carries its unit; every spatial figure carries
   its scale.
10. Contract identifiers are opaque tokens. Do not assume iOS will compare them
    with `>=`.
11. Report state; do not require iOS to infer it from frame traffic.

---

## 14. What iOS does **NOT** support — do not assume **[NOT]**

- **No inbound routing to cartridges.** `TowerClient` decodes exactly one inbound
  type and publishes to nothing outside itself. Any world message needs a new
  routing seam on iOS (small, but it does not exist).
- **No second transport.** One WebSocket, one owner. No HTTP client, no REST
  calls, no second port, no polling loop, no long-poll, no SSE. A world contract
  that needs an HTTP fetch means new ownership on the phone.
- **No binary frames, no compression beyond fixed-quality JPEG, no batching, no
  msgpack/protobuf/CBOR.**
- **No request/response correlation.** iOS sends fire-and-forget and awaits no
  reply for anything except the initial pong. There is no request id, no ack, no
  retry-on-nack.
- **No iOS-initiated World Builder request.** `WorldBuilderClient` has `state` and
  `stateUpdates` only — no method. (Document Memory's client has a `search`
  method; World Builder deliberately does not.) Capture start is the only trigger
  that exists.
- **No 3D rendering.** No SceneKit, RealityKit, or Metal is linked.
- **No pose/coordinate/handedness convention** is decided or assumed anywhere.
- **No world storage, reload UI, or world picker** on the phone.
- **No calibration progress percentage, no tracking confidence percentage.**
- **No module/cartridge selection message.** Opening a workspace is *local
  navigation*; it sends nothing and selects nothing on the Tower.
- **No dynamic module discovery.** `TowerCapabilities` is a local table, and a
  test asserts it is empty.
- **No endpoint discovery or configuration.** The Tower address is a hardcoded
  constant.
- **No frame sending in Release builds.** The whole capture/send path is
  `#if DEBUG`, so a Release build never sends a frame and never receives a
  `frame_result`.
- **`StreamManager` reports nothing** and is not part of any path.

---

## 15. Areas where Tower must not invent behaviour

- **Geometry representation.** iOS stores a label and a count and refuses to
  branch on either. Choose whatever representation is right; just name it in
  words a person can read, and do not expect iOS to interpret it.
- **Coordinate frame, units, handedness, pose conventions.** iOS has taken no
  position and has no renderer to be wrong.
- **Scale.** Do not report a metric figure unless it genuinely is metric in kind,
  and never omit the unit string. `inferredMetric` obliges iOS to caption the
  figure as an estimate; sending `measuredMetric` from a monocular pipeline would
  make the app lie.
- **Calibration procedure.** iOS models only four coarse outcomes; it does not
  presume an initialisation motion, a reference object, or a plane fit.
- **Confidence.** If the Tower reports an inference without a confidence, iOS
  renders "the Tower did not report a confidence" — which is worse for everyone
  than a number. Do not synthesise a confidence to avoid that; do report one if
  you genuinely have it.
- **Privacy claims.** Do not expect iOS to assert anything about Tower-side
  storage. If it matters, report it as data.
- **Session identity.** Do not require the phone to supply one.

---

## 16. Contract shapes that would force ugly compatibility layers

Ranked by how expensive they would be on the iOS side.

| Shape | Cost on iOS |
|---|---|
| Anything sent before the pong, or a pong with non-string values | **Total failure.** Connection never validates; reconnect loop; app looks dead. |
| Binary or array-root or NDJSON messages | Silently dropped. Looks like the Tower is mute. |
| Requiring fields in `stream_start` / `frame` | Breaks two pinned tests and the deliberately window-bypassing marker path. |
| Assuming `seq` restarts per `stream_start` | Silent, wrong world segmentation after every mid-session reconnect — the *expected* case on this link. |
| Requiring a `stream_stop` to finalise a world | Worlds never finalise after a socket drop; needs a phone-side "did we stop?" tracker that does not exist. |
| A second transport (HTTP/REST/second socket) for world data | New ownership, new lifecycle, new reconnect logic, duplicated status — the exact "duplicated state / lifecycle hack" to avoid. |
| Delta-only geometry with no snapshot semantics | Forces accumulation + merge state on the phone. `WorldSnapshot` is whole-value replace. |
| Pose/point arrays as the primary world payload | Nothing can consume them; bandwidth competes with frames. |
| World updates at frame rate, uncoalesced | Main-actor churn → longer send-window slot lifetimes → **lower delivered frame rate to the Tower**. Self-defeating. |
| Integer/ordered contract versions | iOS compares identifiers by equality only; ordering logic would have to be added and tested. |
| Numeric progress (calibration %, tracking confidence %) | No field exists; iOS deliberately refuses to invent a denominator. |
| Blocking the socket read path >2 s | Stall detection replaces the connection; frames drop for the whole reconnect. |
| Reusing `frame_result` to carry world state | Conflates "a frame arrived" with "something spatial happened" — the one conflation the World Builder state machine exists to prevent. |

**The shape that costs nothing:** one new inbound message `type` on the existing
socket, carrying a self-contained, coarsely-updated world report whose fields map
1:1 onto `WorldSnapshot` (§8.3) plus an explicit lifecycle state mapping onto
`WorldModelState` (§8.2), with a changed opaque `revision` whenever the world
changes. That is decoded in one place and injected as a `WorldBuilderClient`; no
view, no lifecycle, and no existing test changes.

---

## 17. Exact Swift files to reconcile against

Highest value first.

| File | Why |
|---|---|
| `ios/Glasses/Workspaces/WorldBuilder/WorldModel.swift` | **The single most important file.** Every World Builder data type and display gate. |
| `ios/Glasses/TowerClient.swift` | Wire vocabulary, handshake strictness, decoder, reconnect, send gates. |
| `ios/Glasses/Workspaces/WorldBuilder/WorldBuilderClient.swift` | The seam a Tower-backed client must fit. |
| `ios/Glasses/Cartridges/Integration/Observation.swift` | `WorldScaleSemantics`, `ObservationProvenance`, `ObservationTime`, `ReportedFigure`. |
| `ios/Glasses/Cartridges/Integration/CartridgeAvailability.swift` | `CartridgeContract`, availability precedence, `CartridgeFailure.Kind`. |
| `ios/Glasses/Cartridges/Integration/CartridgeClient.swift` | `TowerCapabilities` — where a capability declaration would be cached. |
| `ios/Glasses/ProjectManager.swift` | The four bridges, including stream-bracket reopening on reconnect. |
| `ios/Glasses/GlassesConnection.swift` | Camera lifecycle, `seq` semantics, DAT-initiated stops. |
| `ios/Glasses/SendWindow.swift` + `ios/Glasses/FrameRateGate.swift` | Backpressure and rate arithmetic. |
| `ios/Glasses/Cartridges/Integration/CartridgePhase.swift` | Shared phase vocabulary and the `mayCarryData` invariant. |
| `ios/Glasses/Workspaces/WorldBuilder/WorldCanvasView.swift` | What is actually rendered from a snapshot, field by field. |
| `ios/GlassesTests/TowerClientTests.swift` | The pinned wire behaviour — payload shapes, bracket rules, reconnect, stall. |
| `ios/docs/agent-handoffs/IOS-TO-TOWER.md` | The iOS *requirements* list. Useful, but it is a wish list, and its description of Tower is stale. |

---

## 18. One-paragraph summary

The iOS app today is a **frame pump with an honest empty UI**. One WebSocket, one
camera, four cartridge screens that all say "the Tower cannot do this yet"
because `TowerCapabilities` is an empty table. The World Builder presentation
layer is already fully written against a snapshot model
(`WorldSnapshot`/`WorldModelState`) that was deliberately built with opaque holes
where Tower decisions belong — representation label, revision marker, contract
identifier, unit string. A Tower contract that reports **coarse, self-contained,
unit-and-provenance-labelled world snapshots plus an explicit lifecycle state, as
a new message type on the existing socket, without requiring anything new from
the phone**, drops into that seam as a single injected client. Anything that
requires identity from the phone, a second transport, per-frame world pushes,
accumulation on the device, or a stricter handshake, does not.
