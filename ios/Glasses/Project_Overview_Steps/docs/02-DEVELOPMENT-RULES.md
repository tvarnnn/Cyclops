# Development Rules

These rules apply to every agent and developer working in this repository.

## 1. Preserve Boundaries

Do not collapse glasses integration, iOS UI, tower transport, tower runtime, and application logic into one component.

Meta DAT code belongs behind the glasses/transport boundary. Module code must not call Meta APIs directly.

## 2. One Active Module in V1

Only one major application module may be active at a time. Do not introduce concurrent module execution without an explicit architecture change.

## 3. Truthful State Only

Never fake:
- connectivity;
- FPS;
- latency;
- battery values;
- frames processed;
- model readiness;
- module readiness;
- tower availability.

Unknown values remain unavailable/nil.

## 4. Query Current Meta Documentation

DAT is evolving. Never invent or rely solely on remembered DAT APIs.

Before implementing or changing DAT behavior:
1. Use the Meta Wearables MCP.
2. Query `search_dat_docs`.
3. Prefer current official documentation.
4. Record meaningful constraints discovered during implementation in `05-DAT-INTEGRATION.md`.

Modules declare preferred sensor requirements conceptually. Do not design a generalized sensor-negotiation protocol before the actual supported DAT camera/stream configuration model is known — determine that model via `search_dat_docs` during DAT integration, then design the concrete negotiation mechanism against real constraints.

## 5. Keep the Phone Lightweight

Do not move heavy CV/AI workloads onto iOS merely because implementation is convenient. The intended compute target is the tower.

## 6. Modules Own Their Data

Module-specific persistence stays inside the module by default. Promote data to a shared service only when a concrete cross-module requirement exists.

## 7. Safe Module Transitions

Pause processing during module changes. A module receives sensor observations only after it reports READY.

## 8. Resource Discipline

When changing modules, release module-specific models and GPU resources that are no longer needed. Shared caching is allowed only when deliberate and measurable.

## 9. Generic UI First

The iPhone provides generic module selection/status/control by default. A module-specific UI is optional and should be introduced only when generic controls cannot satisfy a real requirement.

## 10. No Premature Scope Expansion

Do not implement future roadmap features while completing an earlier milestone unless they are required for that milestone.

## 11. Build and Test

After code changes:
- run the narrowest relevant tests;
- run an appropriate Xcode build for iOS changes;
- report warnings/errors truthfully;
- do not claim success without verification.

## 12. Security and Privacy

Do not commit credentials, tokens, private keys, signing material, or captured sensitive sensor data. Treat continuous first-person camera/audio as privacy-sensitive data. Add authentication/encryption before exposing tower endpoints beyond a trusted local development network; prefer a private overlay/tunnel (e.g., Tailscale/WireGuard) over direct public exposure when remote access is implemented (see `03-ROADMAP.md` Phase 1.5).

Raw sensor data is local-first by default: it stays within Glasses -> iPhone -> tower and must not be sent to third-party AI/cloud services by default. A future feature that genuinely requires sending raw or sensitive sensor data to a third-party service must be an explicit, documented architecture/privacy decision with clear user opt-in, disclosure of what data leaves the local system, data-minimization consideration, and retention/privacy review — not a default path for normal module development.

Real-world capture must comply with applicable law and location/institution policy. Requirements vary by jurisdiction, location, audio/video context, reasonable expectation of privacy, institutional policy, and use — this repository does not encode jurisdiction-specific legal conclusions. Avoid privacy-sensitive/private environments during development unless specifically appropriate. Minimize collection and retention of unnecessary raw sensor data. Make recording/capture state clear during controlled testing where appropriate. The technical ability to capture data does not imply unrestricted permission to capture it.

See `06-PRIVACY-DATA.md` for the platform-level data retention/deletion/transmission policy that modules must follow.

## 13. Accessibility Claims

Accessibility modules are experimental assistive systems unless validated to an appropriate safety standard. Do not represent them as guaranteed navigation or safety systems.

## 14. Daily-Driver Hardware

V1 uses supported DAT/Developer Mode paths. Custom firmware, bootloader modification, hardware modification, and destructive reverse engineering are separate future research and must not be introduced into normal development.

## 15. Bounded Operations

Lifecycle operations (module loading, stopping, connection/reconnection) must not block indefinitely. Use bounded timeouts with a defined failure transition, and bounded/exponential backoff for automatic reconnection rather than tight retry loops. The real-time perception pipeline must not accumulate an unbounded frame queue; prefer dropping stale frames over growing latency. See `01-SYSTEM-ARCHITECTURE.md` — Reliability Policies for detail. Exact timing/queue constants are implementation decisions informed by measurement, not fixed by this rule.

## 16. Epistemic Honesty / Platform Constraints

Every module and every future agent must respect `07-PLATFORM-CONSTRAINTS.md` — the canonical record of what the platform can measure, what it can only infer, and what it cannot reliably know. In particular:
- an observation is evidence, not automatically fact;
- an ML inference (depth, identity, semantic interpretation, etc.) must never be represented as equivalent to a direct sensor measurement;
- the absence of an observation must never be reported as an observation of absence;
- confidence/uncertainty attached to an inference must survive persistence, transmission, fusion, and cross-module consumption;
- capture time, network arrival time, and processing time are conceptually distinct and must not be conflated.

Before claiming a workaround "solves" a limitation, classify it per `07-PLATFORM-CONSTRAINTS.md`'s workaround classification (MITIGATES / COMPENSATES / RECOVERS / VALIDATES / REQUIRES FUTURE HARDWARE-API) and state what it does not solve.

## 17. Architecture Documents Are Current Best Design, Not Mandates

`00-PROJECT-VISION.md` through `07-PLATFORM-CONSTRAINTS.md`, and the module docs under `docs/modules/`, record the current best design — not an unquestionable mandate. When implementing future work:
- challenge unnecessary complexity;
- identify assumptions in these documents that turn out to be incorrect;
- point out an existing solution already present in the codebase before proposing a new one;
- recommend a simpler approach when the documented one does not justify its cost;
- identify performance/reliability risks in a documented approach;
- challenge a technology choice that does not justify its dependency/complexity cost — NVIDIA acceleration technologies (`01-SYSTEM-ARCHITECTURE.md` — GPU / Acceleration Strategy) are the likeliest place this applies, but the rule is general;
- explain the tradeoff before materially deviating from documented architecture, rather than silently doing something different.

Optimize for the quality, maintainability, performance, and correctness of the system — not agreement with the documentation. Silent compliance with an approach known to be wrong is not deference; it is a failure to do the job.
