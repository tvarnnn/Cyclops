# Module System

## Purpose

The tower hosts a library of interchangeable applications — conceptually, the glasses platform is a host/runtime, and each module is a cartridge: the same glasses, the same streaming/session infrastructure, and a different loaded module produces a different capability. Swapping the active module changes what the platform does without changing how sensor data reaches it.

Current and future module examples (each with its own specification under `docs/modules/`; see also `03-ROADMAP.md`): Experimental CV Lab, World Builder, Accessibility, Visual Q&A, Object Memory, Environmental Memory, and a future Translator module (`docs/modules/TRANSLATOR.md`) — plus modules not yet designed. This document defines the contract every module shares; it does not enumerate what any specific module does.

Modules must reuse shared platform infrastructure rather than reimplementing it:
- camera access;
- streaming/transport;
- session lifecycle;
- networking;
- common perception services, where a shared service is actually justified (see Model Resources, below);
- memory services, where appropriate;
- compute resources (Tower GPU/CPU allocation).

The long-term goal is for the iPhone to discover the module library dynamically and request activation; the initial implementation uses a single hardcoded module instead (see Discovery below and `03-ROADMAP.md`). One major module is active at a time in V1.

Do not design a complex plugin ABI now. The module contract below (descriptor, lifecycle, sensor profile, data behavior) is deliberately conceptual and should evolve incrementally as real modules are implemented, per `03-ROADMAP.md`'s "generalize only when justified" sequencing — not be replaced up front with a fully general plugin system.

## Module Descriptor

Every module should eventually expose metadata equivalent to:

```text
id
name
description
version
capabilities
sensorProfile
dataBehavior
settingsSchema (optional)
uiExtension (optional)
```

Exact programming-language types are intentionally deferred until the tower stack is selected.

## Sensor Profile

A module declares what it wants; it does not configure Meta DAT directly.

Potential profile fields:

```text
cameraEnabled
preferredFPS
preferredResolution
microphoneEnabled
audioOutputEnabled
latencyPriority
storagePolicy
moduleSpecificSettings
```

The transport/runtime negotiates these requests against what the current hardware/DAT version actually supports.

Unsupported requirements must produce a clear degraded/failed state rather than silently pretending they were applied.

## Data Behavior

Every module descriptor must declare its data behavior. This is a required part of the module contract, not an optional per-module afterthought. Conceptually, a module must state:
- what data it persists;
- whether raw imagery/audio is persisted, or only derived/structured data;
- retention behavior (e.g., indefinite, time-bounded, session-only);
- whether it supports clearing/purging its own stored data;
- whether any of its data leaves the local system, and under what conditions.

The full platform-level policy these declarations must satisfy is defined in `06-PRIVACY-DATA.md`. The exact interface (e.g., a `purge()` method or equivalent) is an implementation detail to be defined once the tower module contract is actually implemented — this document requires the capability, not a specific method signature.

## Lifecycle

Conceptual states:

```text
UNLOADED
LOADING
READY
ACTIVE
STOPPING
FAILED
```

Conceptual lifecycle operations:

```text
load()
start()
process(observation)
stop()
unload()
```

The final interfaces should be defined when the tower implementation language/framework is selected.

## Switching

Required transition:

```text
ACTIVE(A)
 -> pause processing
 -> A.stop
 -> A persist if needed
 -> A unload/release resources
 -> B load
 -> B initialize
 -> negotiate B sensor profile
 -> B READY
 -> B start
 -> resume processing
 -> ACTIVE(B)
```

Frames captured during an unsafe transition are dropped by default, not injected later.

## Persistence

Each module owns a data directory/storage namespace.

The shape below was illustrative and the shipped modules did not adopt it.
What they actually use, as of 2026-08-22:

```text
data/world_builder/worlds/<world_id>/...     (World Builder, DEFAULT_ROOT)
data/world_builder/captures/<capture_id>/    (shared dataset recorder)
<caller-supplied directory>                  (Object Memory ObservationStore)
```

The principle held; the path convention did not. What matters is that a
module owns its namespace and no module reads another's, not that the
directories share a prefix. `ObservationStore` takes its directory as a
constructor argument and hardcodes no default at all, which is the
better shape — the owner of the data decides where it lives.

Storage technology may differ by module. Do not impose a universal database prematurely.

## Model Resources

Modules declare their required models/resources. The runtime coordinates loading/unloading so old module resources do not unnecessarily occupy GPU memory.

A shared model cache may be introduced later if measurements show repeated model loads are costly and multiple modules genuinely share the same model.

## Discovery

The tower runtime maintains the authoritative module registry once one exists. The initial implementation hosts exactly one hardcoded module (a "registry of one" — see `03-ROADMAP.md` V0.8–V0.9); there is no dynamic registry at that stage. The iOS app requests the registry and renders available modules only once the registry is generalized (V1.0), triggered by a second production module creating real requirements — do not build dynamic discovery speculatively.

Once implemented: adding a normal tower-only module should not require a new iOS release.

## UI

Default: generic iOS UI based on module descriptor/settings metadata.

Optional: module-specific UI may be introduced for requirements that cannot reasonably fit generic controls. It must not move the module's core compute logic onto the phone.

## Failure

If a module fails:
- mark it FAILED/Unavailable;
- stop routing observations to it;
- preserve logs;
- release resources when safe;
- keep the tower/runtime and glasses connection alive where possible;
- do not return stale or fabricated results.
