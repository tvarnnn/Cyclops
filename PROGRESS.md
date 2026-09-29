# C26 progress

- Read C26 brief, P3/P2 rules, PHONE-MASK results, research patch, and W3 decision diff.
- Confirmed branch `codex/c26-lpbig`; only this worktree is writable. No Tower or GPU use.
- Implemented the default-off solver-only rule and tests for persistence, thresholds, component cache keys, records, and surface borrowing. Focused test selection: 78 passed.
- Final required selection: OFF `319 passed, 308 warnings in 104.04s (0:01:44)`; ON `319 passed, 308 warnings in 127.68s (0:02:07)`.
- Other direct importers: `247 passed in 14.70s`. `git diff --check` exited 0.
- Ready for lead review and commit; no git writes, Tower start, GPU use, or live-store writes.
- OPEN: corpus A/B proof and Walk 6 held-out check belong to the lead.
