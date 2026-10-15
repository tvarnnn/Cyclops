# Claude Start Here

Use this prompt at the beginning of a new substantial Glasses development session.

```text
You are working on the Glasses wearable AI platform.

Before modifying code:

1. Read:
   - docs/00-PROJECT-VISION.md
   - docs/01-SYSTEM-ARCHITECTURE.md
   - docs/02-DEVELOPMENT-RULES.md
   - docs/03-ROADMAP.md
   - docs/04-MODULE-SYSTEM.md
   - docs/05-DAT-INTEGRATION.md
   - docs/06-PRIVACY-DATA.md

2. Read the relevant prompt/module specification for the task.

3. Inspect the actual repository state, current files, git status, and recent changes. The documentation describes intended architecture; the repository is the source of truth for what currently exists.

4. Identify the current roadmap milestone and do not silently expand beyond it.

5. If the task touches Meta DAT, use the Meta Wearables MCP and `search_dat_docs` before proposing or implementing APIs. Do not guess DAT behavior.

6. Preserve these invariants:
   - iPhone = lightweight control plane/transport.
   - tower = heavy compute/runtime.
   - one major active module in V1.
   - module-specific persistence stays with the module.
   - module switching pauses processing until READY.
   - module registry starts as a single hardcoded module ("registry of one"); it becomes dynamic/tower-authoritative only once a second production module justifies generalizing it.
   - lifecycle operations use bounded timeouts; reconnection uses bounded/exponential backoff; the frame pipeline drops stale data rather than queueing unboundedly.
   - raw sensor data is local-first; no third-party transmission without an explicit documented exception.
   - truthful states only; never fake metrics or connectivity.
   - Meta-specific code stays behind the DAT boundary.
   - these documents describe the current best design, not an unquestionable mandate — challenge unnecessary complexity, flag incorrect assumptions, and explain tradeoffs before materially deviating (`02-DEVELOPMENT-RULES.md` Rule 17).

7. Before implementation, summarize:
   - what currently exists;
   - what milestone/task you believe is requested;
   - files you expect to touch;
   - tests/build verification you will run;
   - any architectural concern or documentation conflict.

If requirements are ambiguous in a way that changes architecture, ask before coding. Otherwise make the smallest change that satisfies the current milestone.

After implementation, run verification and report actual results. Update roadmap/docs only when the implementation genuinely changes their state or reveals a verified constraint.
```
