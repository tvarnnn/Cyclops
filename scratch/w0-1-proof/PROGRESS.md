# W0-1 progress

- Worktree: `codex/w0-1` at `44fbd13`; only this worktree was written. The P3 shared RUN progress location is read-only under this brief, so this record is kept in the allowed worktree.
- Implemented: default-off switch, private digest-checked database per draw, lazy child launch after draw 0, seed-order delivery, Job Object ownership, RAM refusal, serial fallback, child timing and fallback diagnostics in I0 only.
- Verification: CPU-only fake-child tests and the selected consensus/coherence/timing suite run with the switch off and on; exact final counts are reported in the handoff. No GPU job or Tower was started.
- Next: lead review; C18 frozen-world identity proof using `prove_identity.py` under gpulock; C22 quiet timing/live-safety proof. These are OPEN and no Tier-A or speed claim is made yet.
