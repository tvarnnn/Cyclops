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
`redacted`, or exactly `unredacted` only under the server switch below),
`pose_source`, `component_reference_segment`, and
`segment_index`. B1 uses `segment_local`, null component reference, and the
live segment index. B2 uses `landed_global_solve`, the registered component's
reference segment, and null segment index. Neither section carries position.

## Unredacted sessions (demo; privacy deferred, 2026-10-12)

`TOWER_WORLD_GUIDANCE_IMAGERY_UNREDACTED` defaults off and accepts only `on`
(any other value, including `1`/`true`, is off and changes nothing). It is the
owner-approved demo switch for the deferred-privacy baseline, where redaction
is bypassed and walks record `session.json` `"redaction": "none"`. With it on,
a session whose keyframe set's label is exactly `none` is served by both routes
and by both producers (B1 live, B2 landed, including the S3 capture dial): its
tiles are thumbnails of the stored keyframe JPEGs as persisted, with no
redaction applied, and **every entry from such a session carries
`"redaction": "unredacted"`**. Sessions with a trusted label still carry
`redacted`. Every other untrusted label (unknown strings, a re-run label, an
unreadable session) is refused exactly as with the switch off. With the switch
off, `redaction` is never `unredacted` and a `none` session is 404.

Clients must accept `unredacted` as a value of `redaction`; it changes nothing
about the entry's shape. B1 never relabels bytes: a tile read under one
`redaction` value is re-read, not reused, if the session's label changes.
This switch governs guidance imagery only; appearance, depth and world-build
trust are unchanged.

## Production and dating

Every tile is read through `keyframe_image_bytes` with
`imagery_source=IMAGERY_REDACTED` (or, for a `none` session under the
unredacted switch only, straight from the stored keyframe JPEG); raw reader
allowlists are not consulted.
An untrusted keyframe set or failed redaction emits no tile. There is no disk
tile cache. B1 recomputes from accepted, locally posed keyframes on manifest
fetch, newest first, at most one new 16 KiB tile per second. A segment break
clears the strip immediately. B1 may change with any local rebuild.

B2 receives only accepted in-horizon keyframes with coherent registered solve
placement from the coverage computation. After coverage publishes, the same
worker selects thumbnails within an 80 ms imagery budget (one per set sector
of the hero component; see "B2 capture-dial selection" below). Its block replaces
the previous block as a whole; a failed job leaves the previous one visible.

The 80 ms budget is the tile job's own CPU time (`time.thread_time()`), with
a 1 s wall-clock cap so that a stalled read cannot hold the worker. A wall-time
budget measured only how long the below-normal-priority worker waited for the
CPU: on e2e10 (2026-10-12 20:47:10), a ~2 ms job was omitted at ~110 ms wall.
A job that exceeds either limit publishes nothing and is retried once, at the
next worker tick (about 0.5 s later), for the same receipt only. The retry is
dropped if a newer receipt for that session has published by then. A job with
no tile, an untrusted set, or a cancellation is not retried.
An imagery manifest request racing a pending landing waits at most 80 ms for
that block; status replies and coverage publication never wait for imagery.
Coverage's solve-candidate memo prevents a local rebuild from re-dating B2.
No manifest or tile is written to disk.

## B2 capture-dial selection (S3, 2026-10-12)

Authority: `FOW-RING-SPEC-20261012.md` §7 S3 and manager mail 366. This is
additive and sits under the existing `TOWER_WORLD_GUIDANCE_IMAGERY_LANDED`
switch: there is no new switch, `version` stays 1, B1 is unchanged, and the
coverage receipt is byte-identical with or without it. With the switch off,
nothing below is computed.

The B2 `landed` block keeps `solved_at`, `geometry_revision`,
`frame_revision` and `entries`. It adds:

```
"landed": {"solved_at", "geometry_revision", "frame_revision",
           "entries": [{"keyframe_id", "digest", "redaction", "pose_source",
                        "component_reference_segment", "segment_index", "sector"}],
           "latest": null | {"component_reference_segment", "sector", "keyframe_id"},
           "motion_sector": null | 0-11}
```

**Inputs.** All inputs come from the coverage receipt this block is dated by.
That receipt's eligible keyframes are the accepted in-horizon keyframes with
coherent registered solve placement, restricted to its drawn (listed)
components. "Newest" means the accepted journal order (the solve's
`keyframe_ids`), never keyframe-id string order.

**Sector.** A keyframe's sector is the receipt's own binning, the same
computation that set the coverage masks. The keyframe's solve-pose forward is
projected onto the component floor plane and measured clockwise, seen from
above, from the component's `forward_xyz` (`sector_frame`
`first_qualified_forward_cw_from_up_v1`), in 30-degree sectors 0-11. A
boundary belongs to the clockwise sector. A forward with no horizontal part
has no sector: it sets no mask bit, and it is never `latest` or a tile.

**Hero component H** is the component of `latest`'s keyframe. When no eligible
keyframe has a sector, H is the component of the newest eligible keyframe.

**Selection.** The set sectors of H are the OR of `weak_mask | supported_mask`
over H's drawn stations. A station dropped by the 16-station cap does not
count. For each set sector, in ascending sector order, the tile is the newest
eligible keyframe of H whose station cell is a drawn station of H and whose
sector is that sector. If that keyframe yields no redacted tile, the next
newest is tried, with at most 3 reads per sector. A sector with no readable
candidate is omitted. There is therefore at most one entry per set sector,
never more than 12 entries, and never a tile on an unset (grey) sector. Every
entry's `component_reference_segment` is H.

**`entries[].sector`** is that sector (0-11), and it agrees with H's masks.

**`latest`** is taken from the newest eligible keyframe in any drawn
component that has a sector: `{component_reference_segment, sector,
keyframe_id}`. Its station need not be drawn, and it need not have a tile. It
is an orientation anchor for the dial, and it carries no position. It is
null when no eligible keyframe has a sector.

**`motion_sector`** describes the floor-plane displacement from the oldest to
the newest of H's newest 24 eligible keyframes (about 5 s of walking). That
displacement is projected onto H's `forward_xyz` and the right axis (forward
× up), then binned exactly like a sector. It is null when H has fewer than 2
eligible keyframes, or when the displacement is shorter than
`0.25 * cell_size` of H. The displacement itself is not published.

The threshold is scale-free on purpose (manager 374): it is a fraction of the
component's own cell, never metres, because a solve's metric scale can be
unknown (walk 9). On the 19 real solve trees in the canonical world root,
`motion_sector` is null on 5 of them. The S3 v1 rule (newest 8, `0.5 *
cell_size`) was null on 15.

Two known limits follow from being relative to the cell:
- A component whose bounds are inflated by outlier placements always reads
  null. On c81766a3, `bounds_xy` spans about 9e6 solve units, so `cell_size`
  is about 1.1e6, while the walk displacement is 2.9. Its station grid is
  just as degenerate.
- A tiny component reports a direction from a tiny displacement. On
  488ed621 the hero piece has 10 keyframes about 1 mm across; a 1.3 mm
  displacement gives sector 6.

**Budgets are unchanged.** Each tile is at most 16 KiB and 160 by 90 pixels.
The imagery job has the same 80 ms budget, now measured as CPU time (see
"Production and dating" above). If it is exceeded, nothing is published, the
previous block stays, and the job is retried once. A landing that yields no entry also
publishes nothing, and the previous block stays with its own `latest` and
`motion_sector`. The H/sector/latest/motion computation runs inside
coverage's 250 ms job; it measured at most 1.3 ms, and 0.03-0.45 ms median, on
the 19 real solve trees in the canonical world root (42-1090 horizon
keyframes). Tile encoding there took 1.6-14.4 ms per landing for 1-12 tiles.

**Clients.** A client must treat `sector`, `latest` and `motion_sector` as
optional. An older Tower, or a direct caller without dial inputs, omits all
three; that block keeps the pre-S3 selection, a deterministic stride over the
eligible keyframes, with at most 16 entries. An absent or out-of-range
`sector` places no photo.

**What the phone relies on (ios/fow-dial 7ea559c).** Each point below is
pinned by a Tower test.
- `entries` always has 1-16 items. A landing that would publish no entry
  publishes nothing: `landed` stays the previous block, or `null` before the
  first one. It is never `[]`, which the phone refuses.
- `sector`, `latest.sector` and `motion_sector` are JSON integers 0-11, never
  floats or booleans.
- `latest` has all three fields, with a non-empty string `keyframe_id`, or it
  is `null`.
- `solved_at`, `geometry_revision` and `frame_revision` are copied from the
  coverage receipt this block was selected from. A block kept after a failed
  landing still carries its own, older triple. The phone sees that it no
  longer matches the current receipt and does not use it.
- `latest` and every entry name a drawn component: one listed in that
  receipt's `components`. If the newest keyframe with a usable heading
  belongs to a component the station cap dropped, `latest` comes from the
  newest one in a drawn component instead.
- With `TOWER_WORLD_GUIDANCE_IMAGERY_LANDED=on`, the coverage worker always
  supplies the dial inputs, so every entry carries `sector`. If the dial
  computation fails, that landing publishes no B2 block at all. It never falls
  back to the stride selection, whose entries have no sector.

## Photo floor plan v1.2 (2026-10-12) -- default OFF, imagery_source unredacted

**v1.2, default OFF, `imagery_source` "unredacted" (privacy deferred; TODO
redaction).** Authority: FOW-V12-PHOTO-FLOORPLAN-DESIGN-20261012 and the
frozen wire in mail 303 (the iOS client, `codex/fow-v12-ios` a1a82bc, is built
against it). Deviations go through the lead first.

Switches: `TOWER_WORLD_PHOTO_FLOORPLAN` accepts only `on`, and also needs
`TOWER_WORLD_GUIDANCE_COVERAGE=on`. When it is off, neither route is
registered, no photo worker starts, and the coverage worker reads and publishes
exactly what it did before. The raster is always the lower-image floor
homography (CPU only); `TOWER_WORLD_PHOTO_FLOORPLAN_METHOD` is no longer read,
so any value of it, set or unset, gives the same plan.

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
    rejects a plane with inadequate support. On a partial live trajectory (no
    usable camera plane) the reference is the roll-free vertical and the tilt
    window is 20 deg; a landing still too sparse for any plane falls back to a
    level plane from the camera trajectory (height read off what the downward
    frames observed), logged as `fallback_camera_drop`, so a live plan is not
    blank. An estimated plane may be carried to the next landing by a Sim(3)
    on shared keyframes (at most two landings, then refit);
  - warps only the bottom 40% of downward-looking, undistorted solver images
    onto that plane, with horizon, incidence, range and per-frame pose gates;
  - keeps a cell opaque only if at least two keyframes see it, or one view
    that is not grazing. Single-view cells are desaturated. No alpha is
    dilated. An opt-in bottom-band furniture mask exists in the raster core
    (default OFF, no Tower switch: on walk 8 it was not a clear improvement), so
    by default furniture in the bottom band can still smear onto the floor.
- **Imagery.** It is RAW camera pixels from the solver's own undistorted
  images. Redaction is deferred and must not be assumed of these bytes.
