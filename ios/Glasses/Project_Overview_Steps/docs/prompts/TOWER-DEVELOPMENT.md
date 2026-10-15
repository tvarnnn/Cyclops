# Claude Prompt — Tower Runtime Development

```text
Work on the Glasses persistent tower runtime.

First read docs/prompts/START-HERE.md and follow it.

Initial tower stack is pinned: Python, FastAPI, WebSockets, OpenCV; PyTorch is an optional extra required only for model-backed CV experiments, CUDA used when needed. Do not introduce gRPC, C++, distributed infrastructure, message brokers, or orchestration platforms speculatively — see `01-SYSTEM-ARCHITECTURE.md`.

The tower is the primary compute environment. It will eventually own:
- authenticated transport from iPhone;
- module registry;
- module lifecycle;
- GPU/model resource management;
- telemetry/logging;
- module execution;
- module-specific persistence.

Do not build all of those merely because they are listed. Implement only the current roadmap milestone.

Core invariants:
- one active major module at a time in V1;
- tower module registry is authoritative once it exists — the initial implementation hosts a single hardcoded module ("registry of one"); generalize only once a second production module justifies it (see `03-ROADMAP.md`);
- module switch pauses observation processing;
- stop/persist/unload old module before loading the next;
- new module must reach READY before observations resume;
- module-specific data remains in that module;
- drop transition frames by default rather than replaying stale observations;
- release module-specific GPU resources when no longer needed;
- tower failure must surface as unavailable to iOS;
- lifecycle operations use bounded timeouts with a defined FAILED transition on timeout;
- reconnection uses bounded/exponential backoff, not tight retry loops;
- module data behavior (persistence, retention, purge, transmission) must be declared per `04-MODULE-SYSTEM.md` and follow `06-PRIVACY-DATA.md`;
- GPU acceleration (CUDA, PyTorch CUDA execution, TensorRT, CV-CUDA, DeepStream, etc.) is an optional capability adopted only where profiling justifies it, not a hard dependency — see `01-SYSTEM-ARCHITECTURE.md` — GPU / Acceleration Strategy.

Keep transport, runtime orchestration, and module implementation separated.

Before editing, inspect the existing tower code and state the exact interface/change you propose. Use tests for lifecycle and state transitions. Do not claim performance targets without measurements.
```
