# World Builder — current state

**Living document.** It describes what exists now, not what changed when.
Rewrite stale parts; git history is the record of how it got here.

**Branch:** `integration/world-builder-lifecycle-v1`
**Last updated:** 2026-08-25

**One physical walk has happened** (2026-08-24, Ray-Ban Meta glasses to
iPhone to Tower). It is the reason most of this document changed. Its
artifacts are still on disk under `data/captures/` and
`data/world_builder/`, and where a figure here came from them it says so.

Everything else remains **CODE-COMPLETE and SYNTHETICALLY TESTED**, and
so does everything built on 2026-08-25 in response to that walk -- the
automatic follower attach, the live rebuild cadence, the keyframe policy
change and the pose-count correction have run against synthetic frames,
recorded frames and a real subprocess, and **never against the glasses**.
The 2026-08-24 validation does not transfer to them.

Nothing in this repository has been validated against Xcode, a Simulator,
or the iOS app -- and note that the Tower-backed `WorldBuilderClient`
used in that walk is in **no branch of this repository**. See
`docs/agent-handoffs/IOS-WORLD-BUILDER-INTEGRATION.md`.

---

## 1. The shape of the system

Three processes, deliberately. This is the architecture decision the whole
design protects, and the reason live viewing costs the frame path nothing.

```
 glasses ──DAT──> iPhone ──ws://──> TOWER WEB PROCESS
                                      ├── answers frame_result (~3 ms)
                                      ├── writes raw frames to a capture
                                      ├── SPAWNS a worker for that capture
                                      └── serves the result channel (reads only)
                                            │
                       capture dir ──> WORLD BUILD PROCESS
                                      scripts/world_build_session.py
                                      ├── keyframe selection
                                      ├── persists redacted keyframes
                                      └── build(): poses + points

                       world dir ────> INSPECTOR / RESULT CHANNEL
                                      scripts/world_inspect.py
```

The web process **never builds**. It reads what the build process has
persisted. A rebuild can take seconds and the frame path does not notice.

**It now starts that process, which it did not before**, and the
distinction matters: supervising a child is not the same as building.
`tower/capture_workers.py` runs an argv when a capture opens and reaps it
when the capture closes. It is cartridge-blind and names no cartridge;
`main.py` builds the command as plain strings, so the web process still
imports no cartridge and `test_shared_code_does_not_import_a_cartridge`
is unmodified.

**One worker per capture LINEAGE, not per capture.** `CaptureFollower`
already walks into a successor capture by itself after a reconnect, so a
second worker on the successor would put two followers, two mapping
sessions and two writer locks on one walk. The supervisor suppresses the
spawn when `continues` names a lineage a live worker already owns.

Before this, nothing in the Tower had ever started a follower. On
2026-08-24 that meant ten captures were recorded and one of them was
followed, by hand, after a human read its id off a directory listing.

---

## 2. What reaches the phone

Contract: **`docs/contracts/CARTRIDGE-RESULTS.md`**. Read that for the
wire; this is the summary.

- One new inbound message type on the **existing** `/ws` socket. No second
  transport — iOS has no HTTP client.
- `{"type":"cartridges"}` → capability declaration. `GET /cartridges`
  returns the identical object for operators.
- `result_subscribe` → a `result_subscribed` ack, then a
  `cartridge_result` snapshot, then more as the world changes.
- The payload's `model_state` and `world_snapshot` map **1:1 onto iOS's
  `WorldModelState` and `WorldSnapshot`**. Everything else in the payload
  is Tower-native evidence for those values.
- **No imagery, no poses, no points, no paths** cross the wire. Counts,
  states and summaries only. (Section 8 gave five grounds for that; one
  of them -- "iOS has no pose schema" -- turns out to be false, and the
  correction is recorded there.)
- Contract identifier is now **`world_builder.status/2026-08-25`**. It
  moved because `trajectory.pose_count` changed MEANING, not because a
  field was added. See the contract's own changelog.

Coalesced to at most ~2 Hz. One slot per subscriber, newest wins.

---

## 3. Reconstruction

**Approach:** classical incremental structure-from-motion. Two-view
initialisation (`findEssentialMat` USAC_MAGSAC + `recoverPose`), then PnP
extension against accumulated landmarks. ORB features, Lowe ratio test.
`tower/world_builder/backends/classical.py`.

**Accuracy against ground truth**, 640×360, 20 scene seeds:

| motion | direction error median | p90 | worst |
|---|---|---|---|
| lateral (`strafe`) | 0.30° | 1.32° | 4.91° |
| forward (`forward_walk`) | 5.70° | 7.36° | — |

Forward motion is an order of magnitude worse and **that is correct** —
the epipole sits inside the image. It is also what a walking person does,
and it solved 12 poses where lateral solved 24 across the same 12 seeds.
**A walking demo produces a sparser map than a sidestepping one.**

**Honesty:** zero confident-wrong poses in any motion type tested,
including **pure rotation**, which is genuinely degenerate. A refused pose
carries no translation at all. Pinned by
`tests/test_world_builder_pose_accuracy.py`.

**Two figures that were being confused.** `recoverPose` narrows its mask
in place with a cheirality test bounded by an undocumented
`distanceThresh` of 50 baselines, so the field named `inlier_ratio` was
reporting cheirality. At a 4 cm baseline the true epipolar ratio is 0.963
and the reported one was 0.004. Both are now recorded in the fields
declared for them.

**The gate still uses the cheirality ratio**, unchanged, deliberately.
Consequence, measured: a real sideways strafe at a 4–6 cm baseline
recovers direction to within 2° and is still refused, labelled
`pure_rotation`. Re-deriving that gate is the highest-value geometry work
outstanding and it needs a sweep, not a guess.

---

## 4. Depth and scale

**There is no depth estimation in World Builder.** No MiDaS, no learned
depth, no stereo. Structure comes from triangulation between keyframes.

**Scale states reachable in V1: `unknown` and `relative`. That is all.**

- `relative` means internally consistent with an arbitrary unit fixed by
  whatever baseline the first solved pair happened to have. It is **not**
  metric.
- `estimated`/`inferredMetric` and `measured`/`measuredMetric` are
  **unreachable** — `SCALE_ESTIMATED` is defined and referenced nowhere,
  `measured` is only ever defended and never written, and both backends
  declare `produces_metric_scale=False`.
- A world with **more than one segment** stays `unknown`, because segments
  do not share a coordinate frame.

**Calibrating the camera will not change any of this.** Intrinsics are
what unlock *poses*; they say nothing about how large the world is. A
calibrated monocular reconstruction is still internally consistent in an
arbitrary unit, so `docs/CALIBRATION.md` states plainly that metric scale
remains out of reach afterwards. Expecting metres to arrive with the
first calibration is the obvious misreading, and it is wrong.

**Never render any figure from this system in metres.** `format_distance`
is the single choke point and refuses.

---

## 5. Privacy

**Keyframes are face-redacted before they are written to disk.**
`tower/world_builder/redaction.py`, applied at
`engine._persist_keyframe` — the one place every persisted pixel passes.

- Detector: YuNet, already compiled into our OpenCV; weights vendored at
  `models/face_detection_yunet_2023mar.onnx` (227 KB, MIT, SHA-256 in
  `models/README.md`).
- Settings: confidence 0.30, 2× upscale, head box dilated 1.6×, solid
  fill, re-encoded at q90.
- Cost: **20.5 ms per keyframe**, and only ~35 % of frames become
  keyframes. `observe()` median is unchanged (4.99 → 4.93 ms). It runs in
  the **build process**, so the Tower frame path pays nothing.
- Geometry cost: none measurable. Keyframes and solved poses identical;
  point count moves.

**What the session records is a process claim, not an outcome claim:**
`faces-detected-and-filled/yunet-2023mar@0.30`. Never "redacted",
"anonymised" or "privacy-safe" — the detector has measured false negatives
on faces occluded past ~60 % and rotated ~90°, and profile views are a
known blind spot. `retains_raw_imagery` stays **true**; bodies, clothing,
room contents and any undetected face are still in the image.

**No imagery crosses to iOS**, before or after redaction.

A Tower with no model file says so, logs a warning at session start, and
records `none`. Historical sessions keep `none` forever.

---

## 6. Capture and reconnection

A mid-session reconnect is the **expected** case
(`docs/contracts/CARTRIDGE-RESULTS.md` §19.2): `stream_start` → frames →
socket dies → new socket → `stream_start` again → frames continuing from
the previous `seq`, with **no `stream_stop`**.

Two things make that survivable:

1. **Recordings carry an owner.** uvicorn takes 20–40 s to notice a dead
   socket while iOS reconnects in ~0.5 s, so a zombie connection's
   teardown runs *after* the new one has armed its recording. It used to
   stop it — silently, with `/health` reporting `recording: false` while
   the phone streamed on. A stop from a superseded connection is now
   refused.
2. **Captures declare lineage.** A capture that ends by disconnect leaves
   a resumable marker; the next `stream_start` within **90 s** records
   `continues_capture` in its manifest; and the follower, seeing a capture
   close *by disconnect*, waits out that window for a successor naming it
   and continues into it.

3. **The supervisor honours that lineage.** A successor capture whose
   lineage a live worker already owns does not get a second worker
   (`tower/capture_workers.py`), and the chain is tracked past one hop --
   `b1ab1d` names `b058a6`, not the capture the worker was started on.

Result: one capture lineage, one follower, **one session, one world, one
continuously climbing keyframe count**. Proven on a 24-frame walk cut in
half: 24 frames observed, 1 session, 1 segment; and again end to end
through a real socket, a real subprocess and a real world
(`tests/test_world_builder_autostart_e2e.py`).

The grace window is 90 s because iOS gives up reconnecting after about
16 s against a refused connection and at most about 75 s against a host
that never completes the upgrade (`docs/contracts/CARTRIDGE-RESULTS.md`
§19.1). Later than that is a new walk.

### This is the part the physical walk validated

**PROVEN ON PHYSICAL HARDWARE, 2026-08-24.** Ten captures in 435 seconds:

```
2e6cff  t=0.0   -> 121.9  1395 frames  disconnect   continues=None
                 ...105 s with no capture at all...
341b0f  t=226.8 -> 256.9   259 frames  disconnect   continues=None
b058a6  t=257.1 -> 263.5    18 frames  disconnect   continues=341b0f
b1ab1d  t=263.6 -> 272.0    20 frames  stop         continues=b058a6
79233e  t=272.0 -> 280.1    16 frames  disconnect   continues=None
4fb823  t=283.7 -> 296.2   104 frames  disconnect   continues=79233e
0f0c55  t=303.3 -> 344.2    75 frames  disconnect   continues=4fb823
b901bc  t=344.2 -> 374.9   309 frames  disconnect   continues=0f0c55
1a63a0  t=375.4 -> 379.9     0 frames  disconnect   continues=b901bc
854e96  t=380.0 -> 435.0   610 frames  stop         continues=1a63a0
```

Every judgement in that table is correct. Lineage chained across each
disconnect. The 105-second gap produced no lineage, because it exceeded
the 90 s grace -- that is a new walk. And `79233e` declared no
predecessor despite starting the same second `b1ab1d` ended, because
`resumable_capture()` only offers a capture that ended by *disconnect*
and that one ended by a clean `stop`.

**"Camera LIVE while World Builder said the capture had ended" was this
table, not a bug in any of it.** The follower attached to `2e6cff`
finalised correctly at the gap and exited. The wearer kept walking. Nine
further captures were recorded that nothing was reading, because nothing
started followers. That is what section 1 now fixes.

---

## 7. Resource behaviour

Everything that grows with session length has a bound, and each was
measured rather than asserted.

| | |
|---|---|
| Result snapshot | **flat ~0.75 ms** at 7 or 50,000 journal events |
| Result channel memory | 5.7 → **34.3 KiB and plateaus** over 3,600 polls |
| Result payload | **byte-constant, 3,173 B** |
| Capture follower poll | **flat 0.014 ms** (was 36.9 ms at 20k lines) |
| Measurement window | **constant 1.6 KiB** at 1k or 100k frames |
| Producer caches | capped at 64 entries |
| Subscriptions | 8 per connection |
| Capture | 900 s / 1 GiB, and it now logs when it self-stops |

---

## 8. Replay — what is honest

**The data exists. The consumer does not.**

`WorldView.trajectory(session_id)` returns, per keyframe: the pose, the
segment, the pose status, and **`image_relpath` — the actual frame the
glasses saw at that point on the path**. That is a recorded camera path
with a real first-person view at every point on it, and
`world_inspect.py --trajectory` renders it today.

**None of it goes to the phone, and it should not yet.** The grounds
given when this was written -- iOS links no 3D framework, has no pose
schema, holds summary figures rather than arrays, and has no world
storage, reload UI or world picker -- are partly out of date: `WorldPose`
exists and the app has Saved worlds and an `open(worldID:sessionID:)` path.
A pose array "cannot be displayed and would be dropped". Building a
transport for a consumer that does not exist is the fabricated contract
this project refuses.

**One of those five grounds is false and should stop being cited.** "iOS
has no pose schema" -- and `ios/docs/agent-handoffs/IOS-TO-TOWER.md` §1.4's elaboration that a
pose schema needs "position, rotation convention, handedness, coordinate
frame and units -- five Tower decisions, each of which renders plausibly
and wrongly if guessed" -- describes a decision that has already been
made. All five are settled, written down, and present in every world
artifact (`TOWER-TO-IOS.md` §3.6):

```json
{"pose_type": "T_world_camera", "quaternion_order": "wxyz",
 "handedness": "right", "camera_axes": "opencv_x_right_y_down_z_forward",
 "translation_units": "world", "world_axes_origin": "first_keyframe_camera",
 "up_axis": "unknown", "pose_dtype": "float64", "point_dtype": "float32"}
```

Sending that object verbatim converts the objection into a decode. Two
further grounds are true but circular ("iOS links no 3D framework" is a
consequence of the decision, and a 2D top-down `(x, z)` canvas needs
none while `up_axis` is `"unknown"`) or additive ("holds summary figures
rather than arrays" is one file of optional fields).

**The remaining ground still holds, and it is why nothing was sent:** the
consumer does not exist. That is a statement about scheduling, and it
stops being true the day someone writes the viewer. The exact payload for
that day -- including why points must be opt-in and budgeted against the
byte-constant 3,173 B this channel currently guarantees -- is specified
in `docs/agent-handoffs/IOS-WORLD-BUILDER-INTEGRATION.md` §4.

What must **not** travel, whatever else does: `image_relpath` and every
byte of keyframe imagery. Redaction here is a process claim, not an
outcome claim, and `retains_raw_imagery` is permanently true. A
trajectory and a point cloud are geometry; the frames are not.

**Reopening a saved world already works Tower-side**: subscribe with
`world_id` (and optionally `session_id`) and the channel reports that
world instead of the live one. That is the Tower half of iOS's
`WorldInspectionMode`; the iOS half is a UI that does not exist.

---

## 9. What is NOT known, and needs the hardware

1. **Everything about the real camera.** Distortion, rolling shutter,
   auto-exposure, real motion blur, real texture. The renderer is a
   perfect pinhole.
2. **Whether calibration works at all** — `calibrate_charuco.py` has never
   seen a printed board. **This is the single highest-value physical
   action available**, and it is what stands between this system and
   geometry: the 2026-08-24 walk produced 0 poses and 0 points purely
   because `intrinsics.source == "unknown"`.

   **The plumbing around it is now closed, and only the physical act
   remains.** `tower/world_builder/intrinsics_store.py` is a
   resolution-keyed store at `<world_root>/intrinsics/<w>x<h>.json`.
   `calibrate_charuco.py` writes there by default, and
   `world_build_session.py` looks there automatically using the
   **observed** frame size — the capture journal's `width`/`height` for
   `--follow-capture`, the first jpeg's header for `--frames`. Since the
   Tower already passes `--root <TOWER_WORLD_ROOT>` to every follower it
   spawns (`tower/main.py`), a calibration lands where the builder finds
   it with **no operator flag on either end**. A miss returns
   `CameraIntrinsics.unknown()` and logs the resolution, the path it
   checked and the calibrations that do exist; behaviour on a miss is
   otherwise byte-for-byte what it is today.

   The store **never rescales**. A 480×360 calibration does not satisfy a
   360×640 lookup, and a record whose filename and contents disagree is
   refused — `scales_linearly_across_resolutions` is `null` and DAT's
   resize-versus-crop behaviour is unestablished.

   **The physical procedure is written down**: `docs/CALIBRATION.md` —
   which board, how to verify the print, how many views at what tilts,
   how to capture them through the glasses at the real 360×640 stream,
   the exact command, and how to tell a good calibration from a bad one
   that scored well. Two things there were measured rather than repeated:
   a **uniform** print-scale error does not affect `fx` at all
   (bit-identical result at 25% larger and 50% smaller), while a **10%
   anisotropic** stretch costs 12% `fx` error at an unremarkable 0.67 px
   RMS. `--split-half` was added to `calibrate_charuco.py` as the quality
   check RMS cannot be: 0.31% and 1.94% spread on well-varied synthetic
   sets against 4.55% on a marginal one whose RMS was indistinguishable.

   Still synthetic, all of it. Nothing here has met a printed board.
3. **Whether the delivered resolution supports reconstruction.** All
   figures here are 640×360 renders.
4. ~~**Whether a real walk segments.**~~ **Settled, 2026-08-24.** It
   segments badly: the first physical walk produced **36 segments over
   1395 frames**, 155 keyframes, 10 of them alone in a segment. All 1395
   frames were replayed through the *real* `FrameTracker` and the *real*
   `KeyframeSelector`, reproducing the run bit-identically (same
   keyframes, same 35 losses, same rejection histogram, zero delta
   against every persisted `sharpness`, `survival_ratio` and
   `overlap_ratio`), so the numbers below are measurements and not
   simulations.

   **The stated suspect was wrong.** Blur was not cascading into
   `tracking_lost`; it was *masking* losses that had already happened.
   77% of `blurred` rejections occur when `survival_ratio` is already
   below 0.15. Loosening the blur gate makes segmentation monotonically
   **worse** — `min_sharpness_ratio` 0.45 gives 43 segments, turning it
   off gives 49 — and moving the survival/overlap gates ahead of blur
   gives 40, against the 36-segment baseline.

   **The real defect:** `overlap_ratio` is not an independent signal from
   `survival_ratio`. On real footage the two are equal in 1283 of 1358
   frames (max gap 0.029), because tracks *die* rather than leave frame.
   The `overlap_floor` rescue at 0.45, sitting above a survival reject at
   0.35, could therefore only fire in the band `survival` ∈ [0.35, 0.45)
   — 36 frames out of 1395, 28 of which were already being accepted for
   parallax. The gate written to guarantee "a usable weak link beats a
   broken chain" was very nearly dead.

   **Acted on:** the window is widened from the *top*, not the bottom —
   `min_overlap_ratio` 0.45 → 0.75, `min_survival_ratio` 0.35 → 0.20,
   `loss_survival_ratio` 0.15 → 0.05, giving **20 segments, 260
   keyframes, 4 single-keyframe segments**. Blur thresholds and gate
   order are unchanged, deliberately. The evidence lives in the comments
   at the constants in `tower/world_builder/keyframes.py`.

   **Still not known**, and the reason these are a hypothesis rather than
   tuned constants: this is one walk, one room, one wearer, one lighting
   condition. It costs +68% keyframes, which is unbudgeted for storage,
   for the 20.5 ms/keyframe of face redaction on the write path, and for
   build time. And it shifts the dominant promotion path from parallax to
   track-decay (`overlap_floor` accepts 28/155 → 198/260), whose effect
   on triangulation quality is **unmeasured** — this world has no poses
   to check against (intrinsics unknown → unposed backend → 0 solved
   poses). An ORB + fundamental-matrix proxy says the extra pairs are not
   garbage (median inlier ratio 0.627 → 0.649, no pair below
   `MIN_INLIERS`); that is not the same as saying the reconstruction is
   better. A second walk in a different space should reproduce the
   ordering before these are treated as settled.

   Note also why the old disagreement existed, because it generalises:
   the synthetic renderer is a perfect pinhole, so its tracks neither die
   nor leave frame and `survival_ratio` barely enters the rescue band.
   Re-measured under both policies, the two sequences the shipped test
   exercises accept **exactly the same** number of keyframes (5 for
   `pure_rotation(10)`, 4 for `strafe(10)`), a longer
   `forward_walk(30)` moves only 2 → 4, and **neither policy loses
   tracking even once** on any of them. Segmentation is the thing being
   fixed and synthetic footage does not exhibit it at all — which is why
   1 synthetic segment and 9 audited segments could disagree for so long
   and why neither could settle it.
5. **Whether redaction fires on real faces at real distances.**
6. **Whether the reconnect path works against a real WiFi drop** rather
   than a simulated one.

---

## 10. Known limitations

- **No loop closure, no bundle adjustment.** BA was implemented and
  measured by an audit: **0.00 % drift improvement** at 16, 32 and 104
  keyframes, because the observation graph is a *chain* — `_extend`
  matches only to the previous keyframe, so covisibility span is median 1
  and a chain has no constraint that resists bending. Do not add
  pycolmap; add covisibility first if this is ever revisited.
- **Drift accumulates** at roughly 0.25°/keyframe of rotation with no
  cliff.
- **`frames_observed` is not knowable during a live session** — an
  ordinary rejected frame writes no journal event.
- **Tower cannot see whether a build is running.** The writer lock is
  released before `build()` is called.
- **Four of nine declared event kinds are never emitted**, including
  `build_completed`, which is structurally unreachable.
- **`frame_revision` is a constant 1.** Never use it as a change marker.
