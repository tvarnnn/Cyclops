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

## Photo floor plan v1.2 (2026-10-12) -- default OFF, imagery_source unredacted

**v1.2, default OFF, `imagery_source` "unredacted" (privacy deferred; TODO
redaction).** Authority: FOW-V12-PHOTO-FLOORPLAN-DESIGN-20261012 and the
frozen wire in mail 303 (the iOS client, `codex/fow-v12-ios` a1a82bc, is built
against it). Deviations go through the lead first.

Switches: `TOWER_WORLD_PHOTO_FLOORPLAN` accepts only `on`, and also needs
`TOWER_WORLD_GUIDANCE_COVERAGE=on`. When it is off, neither route is
registered, no photo worker starts, and the coverage worker reads and publishes
exactly what it did before. `TOWER_WORLD_PHOTO_FLOORPLAN_METHOD` is
`homography` (the default: the lower-image floor homography, CPU only) or
`depth` (option 2, which needs the Exp1 live-depth cache). Any other value is
treated as `homography` and logged.

### Routes

`GET /worlds/{world_id}/guidance/{session_id}/photo-floorplan/manifest` returns
200 `application/json`, at most 8 KiB:

```
{"version": 1,
 "landed": null | {"solved_at", "geometry_revision", "frame_revision", "horizon_keyframes",
                   "components": [{"reference_segment", "origin_xyz", "up_xyz", "forward_xyz",
                                   "raster_bounds_xy", "width", "height", "digest", "bytes",
                                   "projected_keyframes", "imagery_source"}]}}
```

- `landed.solved_at`, `geometry_revision`, `frame_revision` and
  `horizon_keyframes` are copied from the v8 coverage receipt this plan was
  built from. Landing identity is (world, session, `solved_at`,
  `geometry_revision`, `frame_revision`).
- `components` has at most 16 entries, sorted by strictly ascending
  `reference_segment`. Each one is a displayed registered component of that
  receipt. `origin_xyz`, `up_xyz` and `forward_xyz` are that receipt
  component's basis, copied verbatim.
- `raster_bounds_xy` = [min_x, min_y, max_x, max_y]. It is the receipt's
  `bounds_xy` squared about its own centre and padded 15%, in the same
  ABSOLUTE coordinates as v8 `bounds_xy`: x = p·forward, y = p·(forward×up).
  Image column 0 is min_x, increasing to the right; row 0 is min_y,
  increasing downward. The four values are rounded to 1e-6 (far below a
  raster cell), which keeps 16 components inside 8 KiB.
- `width` and `height` are 1-512 and match the decoded image.
- `digest` is the lowercase SHA-256 hex of the served bytes; `bytes` is their
  exact length (at most 180 KiB).
- `projected_keyframes` is the number of keyframes that painted at least one
  cell.
- `imagery_source` is `"unredacted"`.
- A component whose plan has no usable photo (for example, a rejected floor
  plane) is still listed. Its tile is fully transparent (alpha 0 everywhere,
  meaning gray / "No usable photo here yet"). A component whose tile cannot
  fit in 180 KiB is omitted.
- `landed` is null until the first plan publishes.
- A world or session that is unknown or not contained answers 404, and so
  does a switch that is off. 404 is an answer, not something to retry.
- A manifest request that arrives while that session's landing job is in
  flight waits for the job, at most 1.6 s. Status replies never wait.

`GET /worlds/{world_id}/guidance/{session_id}/photo-floorplan/tile/{sha256}`
returns `image/png` or `image/webp` with an alpha channel. Alpha 0 means gray
(not seen). WebP colour may be lossy, but its alpha is lossless. A digest
that is unknown, superseded (not in the current landing) or malformed answers
404, as does a disabled switch. Both routes send `Cache-Control: no-store`.
Nothing is written to disk; tiles live in Tower memory only.

### Production and dating

- **When it runs.** After the coverage worker publishes a NEW receipt, it
  enqueues that landing on its own photo-floorplan daemon thread. This is
  never part of a status request, the capture path, coverage's 250 ms budget
  or imagery's 80 ms budget. A newer landing for the same session cancels the
  job in flight.
- **Time cap.** Each landing has an 8 s wall cap (was 1.5 s; manager 323), checked between
  components, frames and reads. On overrun, cancellation or any failure, that
  landing publishes nothing and the prior plan stays, with its prior date.
- **Inputs.** The job uses the exact derived tree the receipt was computed
  from, and the arrays of that solve (`solve/<session>/solution.npz`). It
  refuses the arrays unless:
  - `solution.json` still has the candidate's stat;
  - the arrays are not newer than that file;
  - the per-component point counts match.
- **Publication.** It is atomic per landing: the manifest block and its tile
  index are replaced together. A plan is replaced only by a strictly newer
  `solved_at`, and only while its receipt is still coverage's latest. A local
  rebuild therefore never re-dates or re-places it (coverage's solve-candidate
  memo does not re-offer the landing either), and a failed solve keeps the
  prior plan with its prior date.
- **Homography method.** The method:
  - estimates ONE floor plane per component from the lower envelope of the
    solve's own 3-D points, cross-checked against the camera centres, and
    rejects a plane with inadequate support;
  - warps only the bottom 40% of downward-looking, undistorted solver images
    onto that plane, with horizon, incidence, range and per-frame pose gates;
  - keeps a cell opaque only if at least two keyframes see it, or one view
    that is not grazing. Single-view cells are desaturated. No alpha is
    dilated. An opt-in bottom-band furniture mask exists in the raster core
    (default OFF, no Tower switch: on walk 8 it was not a clear improvement), so
    by default furniture in the bottom band can still smear onto the floor.
- **Imagery.** It is RAW camera pixels from the solver's own undistorted
  images. Redaction is deferred and must not be assumed of these bytes.
