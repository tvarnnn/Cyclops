# Meta DAT Integration Boundary

## Rule

This file documents architectural expectations and verified DAT constraints. It is not a substitute for current Meta documentation.

Before writing DAT code, query the Meta Wearables MCP using `search_dat_docs`.

## Supported Architecture

DAT is the V1 interface between the custom iOS app and supported Meta wearable hardware.

```text
Module requirements
      |
Tower / iOS control logic
      |
GlassesConnection abstraction
      |
DAT adapter
      |
Meta wearable
```

Only the DAT adapter should understand Meta-specific APIs.

## Current Development Assumptions

Based on the project's August 2026 setup research:
- iOS is the chosen mobile platform.
- Development uses Xcode/Swift.
- Developer Mode permits unpublished development integrations.
- Physical Ray-Ban Meta Gen 2 hardware will be used later.
- Meta provides a Mock Device Kit for development/testing without physical hardware.
- The Meta Wearables MCP is configured for this repository.

Verify all version/API-specific details before implementation.

## Configuration Responsibilities

Keep distinct:
- Apple/iPhone Developer Mode and signing;
- Meta AI Developer Mode;
- Meta Wearables developer project configuration;
- DAT SDK/application integration;
- tower networking.

## Sensor Configuration

Modules may request sensor characteristics, but DAT configuration is applied by the DAT adapter/stream layer only after checking current supported values.

Do not hardcode assumed FPS/resolution/device capabilities into generic module code.

Do not design a generalized sensor-negotiation protocol before the actual supported DAT camera/stream configuration model is known — determine it via `search_dat_docs` first, then design the concrete mechanism against real constraints.

## Session Changes

Preferred behavior during module switches is to keep the wearable connection alive and pause processing. If current official DAT behavior requires restarting a stream/session to apply new settings, the adapter may do so.

The module should not know whether DAT paused, restarted a stream, or re-established a session.

## Device Health Telemetry (SDK 0.9.0, recorded 2026-08-21)

What the pinned SDK actually exposes about the health of the glasses. Recorded
here per the update procedure above, because the answer constrains what the app
is allowed to display: Rule 3 forbids inventing any of it.

**The only proactive signal is thermal level.**

- `WearablesInterface.deviceStateStream(for: DeviceIdentifier) -> AsyncStream<DeviceState>`
- `DeviceState` has exactly one property, `thermalLevel: ThermalLevel`.
- `ThermalLevel` is an ordinal, not a temperature: `unknown`, `none`, `light`,
  `moderate`, `severe`, `critical`, `emergency`, `shutdown`.
- There is no listener/`Announcer` variant — `AsyncStream` only.

Observed in `GlassesConnection.observeDeviceState(for:)`, which follows whichever
device `AutoDeviceSelector` reports active, and republished as
`glassesThermalLevel`.

**Absent in 0.9.0 — do not display, do not estimate:**

- Battery level and charging state. `DeviceState` carried `batteryLevel: Int` in
  0.2; it was removed, and Meta has stated battery lands in a later release. The
  only battery signal available today is the terminal `batteryCritical` error
  case below.
- Any numeric temperature, in any unit.
- `HingeState` (present in 0.2, gone in 0.9). Hinge closure is observable only
  indirectly, as `StreamError.hingesClosed`.
- Firmware version, storage, signal strength, worn/don-doff state.
- Any `DeviceStatus` / `DeviceInfo` / `Health*` / `Telemetry*` / `Advisory*` type.

**Reactive health signals** arrive as errors, at the point the stream is already
failing rather than in time to prevent it:

- `DeviceSessionError`: `thermalCritical`, `thermalEmergency`,
  `peakPowerShutdown`, `batteryCritical`.
- `StreamError` (via `stream.errorPublisher`): the same four, plus
  `hingesClosed`.

Meta's guidance is to watch `deviceStateStream` and warn the user *before* a
thermal error kills the stream. The app currently surfaces the level on the
developer screen; acting on it is not yet implemented.

**Interruptions** are modelled as state, not as a dedicated type:
`DeviceSessionState.paused` is a device-initiated interruption (e.g. cap-touch)
and resumes to `.started` on its own — do not restart a session during a pause.
`.stopped` is terminal and requires a new `createSession`.

**Testing constraint.** `MockDeviceKit` in 0.9.0 exposes only camera and
cap-touch simulation (`MockGlassesServices`), plus `fold()`/`unfold()`. Thermal
and battery escalation **cannot** be simulated, so no automated test can cover
the thermal path — it is physical-device-only.

iPhone-side health (`ProcessInfo.thermalState`, Low Power Mode, `UIDevice`
battery) is deliberately *not* here: those are Apple APIs and live in
`DeviceHealth`, outside the DAT boundary.

## Documentation Update Procedure

When DAT implementation work reveals a stable constraint:
1. Verify with `search_dat_docs`.
2. Implement behind the DAT boundary.
3. Add a concise dated note here if it affects architecture or development workflow.
4. Avoid copying large portions of Meta documentation into this repository.

## Known Setup Blockers

- **Bundle ID.** The current Xcode bundle identifier (`tv.lloyd-icloud.com.Glasses`) contains a dash. Current DAT documentation (verified via `search_dat_docs`) states the iOS Bundle ID does not support the `-` character. This must be corrected before DAT app registration (V0.2). Not yet changed as of this documentation pass — see `03-ROADMAP.md` V0.2.
- **Personal Hotspot blocks DAT provisioning.** With Personal Hotspot on, provisioning fails repeatedly with the generic `WarpDeviceConnTransport: [DCTEvent] Failed to open local channel. Scheduling retry.`, which looks like a transient network fault. The real cause shows only in the iPhone's own log (`personal hotspot is active, denying join request from bundleId='com.facebook.stellaapp'`). Fix: turn Personal Hotspot off and leave Wi-Fi available.

## Do Not

- Invent DAT API names.
- Spread Meta SDK calls throughout SwiftUI views.
- Put DAT code into tower modules.
- Treat current undocumented behavior as guaranteed.
- Make custom firmware a dependency of V1.
