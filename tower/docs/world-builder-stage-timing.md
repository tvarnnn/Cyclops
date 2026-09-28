# World Builder finish timing

`TOWER_WORLD_STAGE_TIMING` is off by default. When on, Tower writes
`worlds/<world>/sessions/<session>/stage_timing.json` beside the authoritative
`events.jsonl` journal. The diagnostic file is internal to Tower and is not a
journal event or a phone payload.

Schema 2 has a `finishes` array, newest last. Each finish has `finish_id`,
`started_at` (Unix seconds), `pid`, `context` (`solve`, `regate`, `areas`, or a
standalone stage), and `stages`. A final solve starts a finish, including when
the solve ran in a child process. Surface and appearance attach to the latest
finish only when this process, or a solve child it launched, began it;
otherwise they start their own finish, so they never overwrite an older run's
record. Re-gating in place starts its own finish and records its consensus
draws. An area build starts its own finish. Each `solve_draw` record has a
zero-based `draw_index` and its mapper `seed`. `draw_index` is the order the
draws were mapped in this finish, not the consensus draw number: in a re-gate
the published draw 0 is not mapped again, so re-gate record 0 is consensus
draw 1. Join draws on `seed`.

The file retains at most **3 whole finishes**. Within each finish, the newest
records retained per stage are: `solve` 1, `solve_draw` 7, `gate` 16, `p4` 2
(the early publish of draw 0 and the full consensus), `regate` 1, `surface` 1,
`appearance` 1, and `areas` 1. That is at most 90 stage records in all. The
`gate` limit covers the real maximum, 14, of a seven-draw consensus with P4:
the early publish gates draw 0 once, plus up to three P4 seal re-gates
(`anchor_verify.MAX_ROUNDS` = 3); the full consensus then gates the other six
draws, one withhold re-gate and up to three seal re-gates. A draw index is
assigned when its record is written; when more than seven draws occur within
one finish, the oldest draw is removed. No per-frame records are written.

Each stage record has `wall_ms`, `gpu_wait_ms`, `gpu_wait_source`, `cache`, and
`status` (`ok` or `raised`). `cache` counts observed mask, depth, and pair
lookups, each with `hits` and `misses`. Mask counts come from the initial
component lookup, excluding re-reads of masks just written; a mask then copied
from the solve's masks (the donor, `solve_masks.solve_mask_donor`) still counts
as a miss. Depth counts can include raw prediction reads and fitted-stage
lookups. Pair counts cover frozen matching and consensus link lookups. `gpu_wait_ms` is zero with
`gpu_wait_source: no_blocking_gpu_lock`, because this path has no blocking GPU
semaphore. Nested area surface and appearance work counts in `areas`.

Timing is best effort. The wrapper returns the underlying call's result object
or re-raises its original exception, including when diagnostic recording
fails. Its extra clock reads and work can move volatile duration fields that
the stage measures internally. A process killed during a stage can lose the
records buffered in that process. Writes use Tower's atomic JSON writer,
which removes its staging file even after a failed replacement.
