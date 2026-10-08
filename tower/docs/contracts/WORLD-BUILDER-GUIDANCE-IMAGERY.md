# World Builder guidance imagery v1.1 B (2026-10-08)

Authority: frozen `FOW-V11B-WIRE-AND-IOS-SPEC-20261007.md` with binding
`FOW-V11B-AMEND-1-20261007.md`. This Tower implementation supplies B1 and B2;
the iOS presentation is owned by the Mac lane.

Both `TOWER_WORLD_GUIDANCE_IMAGERY_LIVE` and
`TOWER_WORLD_GUIDANCE_IMAGERY_LANDED` default off and accept only `on`.
With both off, neither imagery route is registered. B2 also requires
`TOWER_WORLD_GUIDANCE_COVERAGE=on` and its fully merged landed solve.

## Routes

`GET /worlds/{world_id}/guidance/{session_id}/imagery/manifest` returns
`{"version":1}` plus `live` if B1 is on and `landed` if B2 is on.
Disabled keys are omitted. `live` has `segment_index` and `entries` (possibly
empty); `landed` is null before a qualifying landing. A missing or untrusted
world/session answers 404. Both routes use no-store headers.

`GET /worlds/{world_id}/guidance/{session_id}/imagery/tile/{digest}` returns
JPEG bytes or 404 for an unknown, superseded, disabled or untrusted tile.
The digest is lowercase SHA-256 hex over these exact served JPEG bytes.
Each tile is at most 16 KiB and 160 by 90 pixels. Each section carries at
most 16 entries. Entries contain `keyframe_id`, `digest`, `redaction` (exactly
`redacted`), `pose_source`, `component_reference_segment`, and
`segment_index`. B1 uses `segment_local`, null component reference, and the
live segment index. B2 uses `landed_global_solve`, the registered component's
reference segment, and null segment index. Neither section carries position.

## Production and dating

Every tile is read through `keyframe_image_bytes` with
`imagery_source=IMAGERY_REDACTED`; raw reader allowlists are not consulted.
An untrusted keyframe set or failed redaction emits no tile. There is no disk
tile cache. B1 recomputes from accepted, locally posed keyframes on manifest
fetch, newest first, at most one new 16 KiB tile per second. A segment break
clears the strip immediately. B1 may change with any local rebuild.

B2 receives only accepted in-horizon keyframes with coherent registered solve
placement from the coverage computation. After coverage publishes, the same
worker selects thumbnails within an 80 ms imagery budget. Its block replaces
the previous block as a whole; a failed job leaves the previous one visible.
An imagery manifest request racing a pending landing waits at most 80 ms for
that block; status replies and coverage publication never wait for imagery.
Coverage's solve-candidate memo prevents a local rebuild from re-dating B2.
No manifest or tile is written to disk.
