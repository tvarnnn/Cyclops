# Glasses: technical study

Ray-Ban Meta glasses stream camera frames through an iPhone app to a Windows PC, the Tower, which turns a walk through a room into a
photographic 3D world that the phone then shows. This document explains how, function by function.

**Provenance.** Branch `cleanup/campaign-v1` at `66414dc1`, the final campaign tree.
Every `file:line` was re-checked against that tree on 2026-10-18. The iOS tree and `docs/` are unchanged since `af805b36` (`README.md`, and the header comment of `tower/.env.example`, were brought up to date in the commit that added this file);
the Tower tree and every contract are byte-identical to `cleanup/tower-v1@e20b3b2`. `main` stays pre-cleanup (§9).
The campaign's run records cited by name (`LEDGER.md`, `LEDGER-tower.md`, `PLAN.md`, `OWNER-NOTES.md`, `IOS-INVENTORY.md`,
`KEY-FUNCTIONS.md`, `IOS-KEY-FUNCTIONS.md`, `GATE2-VERDICT.md`, `CROSSOVER-REPORT.md`, `REPLAY-PROCEDURE.md`, `NOISE-BAND.md`) and the
replay tooling of §8.3 are kept outside the repository; they are not files a reader can open in it.

## 0. How to read this

1. **Order.** Read §1 (one diagram, five hops) and §2 (a walk, stage by stage), then §5, the centre: each key function as WHAT, HOW,
   WHY, then who calls it and what it calls. The index at the top of §5 lists all 84.
2. **The five to know cold:** K9 `TowerClient.sendFrame` (how a frame leaves the phone), T3.2 `WorldBuilderEngine.observe` (each frame:
   reject, tracking lost, or keyframe), T4.6 `solve` (the GLOMAP final solve), T5.2 `surfacify` (the mesh), K14
   `TowerWorldBuilderClient.apply` (how a status snapshot becomes what the phone says and shows).
3. **The rest.** §3 and §4 say where code lives, §6 is the wire, §7 the switches, §8 the tests, §9 what the campaign changed and what is
   still broken, §10 the terms, §11 the open questions.
4. **Labels.** T1.1-T8.9 are Tower functions, K1-K29 iOS functions, S0-S11 pipeline stages (§2).
5. **`file:line`** is the line of the `def`, `func` or `init` (for an HTTP route in §6.4, its decorator). Every path resolves from the repo root
   by one rule. Tower paths are written from `tower/`: `tower/routes/ws.py` is `tower/tower/routes/ws.py`, and `scripts/…` and `tests/…`
   sit under `tower/`. iOS paths are written from `ios/Glasses/` unless they start with `ios/`. Contract docs and `README.md` are given
   from the repo root. Inside an entry, `L123` or `:123` is a line in the entry's own file, or in the module named just before it
   (`ws._offer_capture_opened` (L957) is in `ws.py`). A bare file name (`engine.py:1597`, `TowerClient.swift:1273`) is found by name
   under its side's tree; where several files share it, `engine.py` is `world_builder/engine.py`, `store.py` is
   `world_builder/store.py` and `geometry.py` is `routes/geometry.py`.

---

## 1. System map

```
 RAY-BAN META GLASSES
   │ DAT: DeviceSession + raw camera stream, 24 fps requested, .low = 360x640
   ▼
 iPHONE APP                                                   §4
   GlassesConnection: the only DAT user; seq += 1 per frame   K3, K4
   FrameRateGate: keep ~15 fps, deadline-based                K5
   ProjectManager sinks: kept frame ▸ sendFrame;              K1, K2
      camera start/stop ▸ stream_start/stream_stop
   TowerClient: one WebSocket; gates ▸ SendWindow (5) ▸       K7-K11
      JPEG ▸ base64 ▸ JSON text message
   │                                                       ▲
   │ /ws  ping, cartridges, result_subscribe,              │  /ws  pong, frame_result | frame_error,
   │      stream_start, frame x N, stream_stop             │       cartridge_result (status snapshots)
   │ HTTP POST /cartridges/world_builder/session/start     │  HTTP geometry, render page, appearance…
   ▼                                                       │
 TOWER WEB PROCESS  uvicorn tower.main:app, FastAPI :8000  │  §3
   websocket_endpoint ▸ _handle_frame_message ▸ CV reply ──┤  T1.1, T1.2, T2.2
   _fan_out_frame ▸ CaptureRecorder.write_frame            │  T1.3, T1.5
      ▸ captures/<cid>/frames/*.jpg + frames.jsonl         │  (THE HAND-OFF)
      ▸ LiveSession.offer_frame ▸ Scene, Document threads  │  T2.6
   _start_capture ▸ capture_opened ▸ spawn builder argv    │  T1.4, T2.4, T2.5
   ResultHub ◀ WorldBuilderStatusProducer ◀ world files ───┤  T7.1-T7.4
   HTTP readers: manifest, segment, render ◀ world files ──┘  T7.5, T7.6
   │ argv + files only; the builder is never imported
   ▼
 WORLD BUILDER CHILD  scripts/world_build_session.py          T4.3
                      --follow-capture <dir>
   walk: follow_capture ▸ observe ▸ redacted keyframes        T3.1-T3.4
         + keyframes.jsonl
         interim build every 4 kf · background solve          T3.6, T3.7
         every 50 kf · live depth (MoGe-2)
   Stop: final solve: SIFT ▸ match ▸ GLOMAP ▸ scale guard     T4.4-T4.8
         ▸ publish
         derived tree ▸ release lock ▸ surface ▸ appearance   T4.9, T5.1-T5.4
         [dense: off]
   writes only <TOWER_WORLD_ROOT>/worlds/<wid>/               (the "world files" above)

 PHONE VIEWER (back on the iPhone)                            §4.4
   TowerWorldBuilderClient.apply ▸ walk report                K13-K15
      ▸ WorldBuilderViewModel
   GET …/geometry/manifest + segments (diagnostic gallery)    K16
   WorldStage + ladder: what to say, what 3D to offer         K17, K18
   GET /worlds/{id}/render ▸ WKWebView (glasses-world:)       K19-K21
      + revision polling
```

**Hop 1: glasses → iPhone.** The app talks to the glasses only through Meta's Device Access Toolkit (DAT), in exactly one type,
`GlassesConnection` (§4.2). Capture starts only through `startCameraSession()` (K3). The app asks for a 24 fps raw camera stream at `.low`
(360x640), numbers every delivered frame (`seq`), and keeps about 15 a second with a deadline-based gate (K5); only kept frames are
decoded (K4). The whole capture surface is `#if DEBUG`, so the demo runs a Debug build (§7.7).

**Hop 2: iPhone → Tower.** `ProjectManager` is the only bridge between camera and socket; the two never reference each other (K1).
`TowerClient` keeps one WebSocket to `ws://<authority>/ws` and calls it online only after a ping/pong (K7). It sends each kept frame as a
JPEG inside a JSON text message, behind a 5-slot send window, and only between `stream_start` and `stream_stop` (K9). Opening the World
Builder screen also sends `POST /cartridges/world_builder/session/start` (K12); with no open session, no builder attaches (T2.3).

**Hop 3: the Tower web process.** One Python process on the Windows PC (`uvicorn tower.main:app`, FastAPI, port 8000) with three jobs.
**Ingest:** it takes the phone's JPEG frames on one websocket, answers each frame, and records it durably (T1.1-T1.5). **Build:** it
starts a separate builder process per walk, which turns the recording into a world on disk (T2.4, T2.5). **Serve:** it reads those files
back to the phone (T7). The other cartridges: Object Memory is also a follower process; Scene and Document Memory are in-process worker
threads, each fed through a single newest-wins slot (T2.6).

**Hop 4: Tower → World Builder.** The web process never builds a world and never imports the builder. It knows the builder only as a
command line (T2.5), and the two share nothing but files: the capture directory going in (`captures/<cid>/frames.jsonl` is the hand-off)
and the world tree coming out (`<TOWER_WORLD_ROOT>/worlds/<wid>/`). During the walk the builder follows the capture (T3.1), keeps and
redacts keyframes (T3.2-T3.4) and solves in the background (T3.7); after Stop it runs the final solve, the surface and the appearance
(T4.3; §2 S5-S8).

**Hop 5: World Builder → phone viewer.** Status comes back on the same socket: the phone subscribes to the result channel, and the Tower
pushes complete `world_builder.status` snapshots read from the world's files (T7.1-T7.4, K13, K14). Large data never comes down the
socket, because on the Tower the result sender shares its send lock with the frame path. The phone pulls it over HTTP instead: geometry
(T7.5, K16), saved worlds, and the Tower's composed 3D page (T7.6, K19), shown in a locked-down `WKWebView` (K21). The phone never
reconstructs anything: every word on screen is derived from something the Tower said (K17, K18).

---

## 2. The walk-to-room pipeline

**Walk 10 is the reference walk**: capture `dd87a60e…`, world `e398e1f9…`, session `dcb3d2bd…`. It was recorded live on pre-cleanup code
(`5fecd51`) with the 14 demo switches on. "Replays b1/b2" are walk 10's own frames re-streamed through `/ws` into a test Tower running that
same code (§8.3). The numbers come from the campaign's replay procedure, noise-band and gate-2 reports.

The world tree is `<TOWER_WORLD_ROOT>/worlds/<wid>/`, and `<sid>` is the session. T-numbers point to §5, which says how each step works;
this section keeps what goes in, what comes out, and the walk 10 numbers.

**S0. The phone sends** (§4.3, K9). One `frame` message per JPEG (`seq`, `tx_seq`, `width`, `height`, `format`, base64 `data`), bracketed by `stream_start` and `stream_stop`. The phone sends no `source_seq`; the Tower's parser (`parse_and_decode_frame`, `tower/frames.py:62`) accepts one as optional and falls back to `seq`, so on the Tower `source_seq` is the phone's `seq`.
Walk 10: 2,920 frames of 360x640 in 196.3 s, about 15 fps; that rate is now the iOS default (K5).

**S1. Ingest and record** (T1.1-T1.5). Out: `captures/<cid>/capture.json`, `frames.jsonl` (one row per frame: seqs, `received_at`, `relpath`, size), and `frames/<source_seq:08d>.jpg`. Walk
10: 2,920 JPEGs, 53,718,975 bytes (51.2 MiB), one capture, ended by `stop`. **Final run:** 2,920 frames and 1 capture recorded, ended by `stop` (final run f1); total bytes not reported.

**S2. Attach the builder** (T2.3-T2.5). `stream_start` mints the capture id. If the phone opened the World Builder session, the supervisor spawns `world_build_session.py --follow-capture
<dir> --register --solve --surface --appearance --rebuild-every 4`. A mid-walk reconnect is chained into the same builder, so the walk stays one world.

**S3. Keyframes** (T3.1-T3.5). Per frame: decode, quality, optical-flow tracking, keyframe decision. An accepted keyframe is face-redacted (YuNet), written to `sessions/<sid>/images/`, and
journalled in `keyframes.jsonl` and `events.jsonl`. A tracking loss opens a new segment; the relocalizer is off by default. Walk 10: 2,920 observed, 1,148 accepted. Rejects: blurred 785,
insufficient_motion 953, no_motion_evidence 2, tracking_degraded 17, tracking_lost 15 (identical in every replay). **Final run:** 2,920 observed, 1,148 accepted; rejects blurred 785, insufficient_motion 953, no_motion_evidence 2, tracking_degraded 17, tracking_lost 15, identical to the baselines (final run f1).

**S4. Live work during the walk** (T3.6-T3.8, T5.6). An interim `engine.build` every 4 keyframes rewrites `derived/<sid>/` (the phone's geometry grows from it). A background global solve
every 50 accepted keyframes writes `solve/<sid>/solution.json`; in b1 and p2 the last ran at 880 keyframes. Live MoGe-2 depth goes to `dense/<sid>/live_depth/`; it later becomes the scale
guard's evidence and gives the surface stage a head start. In the Tower process: guidance coverage for the capture dial (K22) and the photo floor-plan tiles. **Final run:** 5 background solves, the last at 864 keyframes (final run f1).

**S5. Stop, then the final solve** (T4.1-T4.8). The builder records `finalization: pending` while holding the world lock, then runs `world_solve.py --final`: GLOMAP from scratch over all
1,148 keyframes. The result goes through the publish seam (evidence gate off by default, final scale guard always on) and writes `solve/<sid>/solution.json`, `components.json` and
`final_scale_guard.json`.
**Walk 10 live:** 1,029 of 1,148 keyframes posed, in 5 components. The main one (c0) has 834 images (697 supported), 19,283 points, mean error 0.879 px. The guard published with scale
`uncertified` ("93 attached keyframes are not certified"). Room 868 / measured 644; evidence `live-depth-cache`; 2 pieces isolated. Finalize took 216.5 s. Replays took 164.3 / 173.8 s, of
which GLOMAP mapping was 79.0 / 88.3 s. **Final run:** 1,029 of 1,148 keyframes posed (871 of them with at least `MIN_IMAGE_OBSERVATIONS` = 30 observations: that is the `posed=` count in the finalize log line), in 11 components. c0 has 807 images (662 supported), 20,421 points, mean error 0.861 px. The guard published with scale `uncertified` (69 uncertified keyframes); room 860 / measured 627; evidence `live-depth-cache`; 8 pieces. Finalize took 153.8 s; GLOMAP mapping 73.8 s (final run f1).

**S6. Derived tree and release** (T4.9, T6.3). The final `engine.build` merges the global solution into `derived/<sid>/`; `finalization` becomes `complete`, the lock is released, segment
registration runs. Walk 10: 742 poses solved, 369 refused, 771 positioned; 21,637 points; 37 segments; `scale_state` unknown. **Final run:** 674 poses solved, 437 refused, 704 positioned; 20,637 points; 37 segments; `scale_state` unknown (final run f1).

**S7. Surface** (T5.1-T5.3). Output in `surface/<sid>/`. Walk 10: 2,048,433 vertices and 3,861,647 faces; 3 levels (L1 380,846 v, L2 156,540 v); 636 of 834 frames used; canonical level 0,
mobile level 2; 446.7 s. Replays: about 390-400 s, of which transients took 187.2 s and packing about 70 s. GPU peak about 2.5 GB. `privacy_safe` true, imagery `redacted`. **Final run:** 2,495,939 vertices and 4,717,718 faces; 3 levels (L1 385,808 v, L2 157,510 v); 579 of 807 frames used; canonical level 0, mobile level 2; 431.9 s (transients 194.4 s, pack 83.8 s); GPU peak in GB not reported (the report gives only utilisation, max 99.0%) (final run f1).

**S8. Appearance** (T5.4). Output in `appearance/<sid>/`. Walk 10: 715 selected (128 for the phone, 587 kept on the Tower); 433 excluded (no depth fit 314, no pose 119); 53 chunks. Phone
coverage: 0.958 seen once, 0.921 seen twice. 124.4 s. **Final run:** 668 selected (128 for the phone, 540 kept on the Tower); 480 excluded (no depth fit 357, no pose 119, no proxy in view 4); 50 chunks. Phone coverage: 0.932 seen once, 0.874 seen twice. 114.9 s (final run f1).

**S9. Dense.** `TOWER_WORLD_DENSIFY` is off, so the dense stage reads `unavailable` (T5.5).

**S10. Serve** (T7). Status goes on the result channel; HTTP carries the rest (geometry, viewer HTML with ladder appearance > surface > dense > sparse, appearance chunks, guidance imagery,
floor plan). All of it is file reads.

**S11. Recover.** When the Tower is idle, `_BackgroundChore` runs `scripts/world_finish_pending.py`: owed surface or appearance work and any owed re-gate (T4.10).

| Walk 10 timing | Live | Replays b1 / b2 | Final campaign run |
|---|---|---|---|
| Walk (stream) | 196.3 s | real pace (`--speed 1.0`) | 196.3 s (2,920 frames at 14.99 fps sent, speed 1.0) |
| Finalize (Stop → solve published, derived built) | 216.5 s | 164.3 / 173.8 s | 153.8 s |
| Surface; appearance | 446.7 s; 124.4 s | 399.7 / 387.9 s; 103.2 / 100.0 s | 431.9 s; 114.9 s |
| Stop → done | 787.6 s (13.1 min) | 667.2 / 661.7 s | 700.6 s (11.68 min) |
| Stop → photos on the phone | n/a | 11.14 min (b1) | 11.69 min (f1) |

---

## 3. Tower code tour, by package

The Tower web process and the builder child, call by call (T-numbers are in §5):

```
 PHONE (TowerClient)                    TOWER WEB PROCESS  (tower/main.py:create_app)
 POST /cartridges/world_builder/session/start ─▶ CartridgeSession = the gate ─────────┐
 {"type":"stream_start"} ─/ws─▶ websocket_endpoint ─▶ _start_capture ─▶ CaptureRecorder.start (mints id)
 {"type":"frame",…} ─────/ws─▶ _handle_frame_message                    └▶ supervisor.capture_opened
     ◀── frame_result | frame_error ─┤ ModuleContainer.process (CV Lab)              │ gate open: spawn argv
                                     └ _fan_out_frame                                ▼
                                        ├ CaptureRecorder.write_frame ─▶ captures/<cid>/frames/*.jpg
                                        │                                + frames.jsonl   (THE HAND-OFF)
                                        └ LiveSession.offer_frame ─▶ Scene / Document worker threads
 ┌─ BUILDER CHILD  scripts/world_build_session.py --follow-capture <dir> ───────────────────────────────┐
 │ follow_capture ▸ WorldBuilderEngine.observe ▸ redacted keyframes + keyframes.jsonl                    │
 │   interim engine.build every 4 kf · BackgroundSolver every 50 kf · LiveDepthWorker (MoGe-2)           │
 │ capture closed ▸ stop_session(hold_lock) ▸ run_final ▸ world_solve.py --final ▸ global_solve.solve:   │
 │   undistort ▸ SIFT ▸ sequential match ▸ GLOMAP ▸ gate_and_publish (final scale guard ▸ write)          │
 │ engine.build (derived) ▸ complete ▸ release lock ▸ register ▸ final_surface_stages:                    │
 │   surfacify (depth ▸ transients ▸ TSDF ▸ mesh ▸ LOD) ▸ build_appearance ▸ [densify: off]               │
 └─────────────────────────────── writes only under TOWER_WORLD_ROOT/worlds/<wid>/ ──────────────────────┘
                                                     │ files
 {"type":"result_subscribe"} ─/ws─▶ results_ws._subscribe ─▶ ResultHub (0.5 s poll, 2 s heartbeat)
     ◀── cartridge_result {world_builder.status …} ◀── WorldBuilderStatusProducer.snapshot ◀── files
                                                     └ CoverageWorker ▸ PhotoFloorplanWorker (daemon threads)
 GET /worlds/{id}/geometry/manifest · /segment/{n} · /render · /appearance/… ─▶ results/* readers ◀── files
 Tower idle: _BackgroundChore ▸ scripts/world_finish_pending.py (owed surface/appearance, owed re-gate)
```

What each package owns; how its main functions work is in §5 (entry points are cross-referenced).

### 3.1 Core (`tower/*.py`)

| Module | Responsibility | Entry points |
|---|---|---|
| `main.py` | App factory and wiring: CV module, recorders, supervisor and gates, cartridge sessions, live cartridges, result hub, routers; also the idle finisher chore `_BackgroundChore` | `create_app` (T2.1), `lifespan`, `_world_build_spec` (T2.5) |
| `config.py` | Every `TOWER_*` setting. `get_settings()` is for the web process; the `world_*_setting()` readers are for the solve and builder children | §7 |
| `frames.py` | Validates and decodes a `frame` message (a `FrameError` becomes `invalid_frame`) | `parse_and_decode_frame` |
| `capture.py` | The raw recorder `CaptureRecorder`, and `CaptureFollower`, which tails a capture's journal | T1.5, T3.1 |
| `capture_workers.py` | One child process per capture: gate, spawn, lineage chaining, soft stop by closing stdin, reap. Cartridge-blind | T2.4 |
| `cartridge_session.py` | Start/Pause/Stop for the out-of-process cartridges (World Builder, Object Memory); this is the supervisor's gate | T2.3, T4.1 |
| `live_session.py`, `cartridge_runtime.py` | The in-process cartridges (Scene, Document): a worker thread, a newest-wins slot, a lifecycle | T2.6 |
| `modules/` | The one CV module that answers every frame (the CV Lab wrapper), plus its failure classes | T2.2 |
| `storage.py` | Atomic JSON writes, JSONL appends that heal torn lines, the Windows replace retry | T6.5 |
| `process_ownership.py`, `stdin_stop.py`, `native_prewarm.py`, `serve_loop.py` | Child discipline (one process, job object, stop by stdin close), prewarming native libraries, a Windows event loop that survives accept errors | supervisor, chore, builder |
| `artifact_paths.py` | Refuses a relative `--root`, and artifact roots at the drive root | `artifact_root_arg` |

### 3.2 `routes/`, `results/`

**`routes/`: a thin transport layer.** `ws.py` is the single `/ws` loop (frames, stream bracket, capture start and stop, disconnect teardown) and delegates to `results_ws.py` and
`cv_lab_ws.py`. HTTP routers: `health.py`, `cartridges.py`, `sessions.py`, `geometry.py` (all of `/worlds…`); `guidance_imagery.py` and `photo_floorplan.py`, which re-export routers from
`results/`; and `observations.py`, `documents.py`, `scene.py`, `cv_lab.py`, `cv_lab_preview.py`. Invariant: a route adapts a question to a reader; it never computes a world.

**`results/`: read-only reporting.** `publisher.py` (`ResultHub`, `ConnectionChannel`, `Subscription`) does polling, coalescing and bounded sends; `registry.py` holds the declaration,
`envelope.py` the envelope and revision hash, `contracts.py` the ids. One adapter per cartridge: World Builder (`world_builder.py`, `_geometry.py`, `_render.py`, `_appearance.py`,
`_library.py`, plus the guidance imagery and floor-plan transports); `scene_understanding.py`, `document_memory.py`, `object_memory.py`, `experimental_cv.py`. Invariants, enforced by
`tests/test_architecture_boundaries.py`: `test_the_result_channel_core_is_cartridge_blind`; `test_the_result_channel_never_writes` (nothing is built on demand, so missing work is owed to the
finisher).

### 3.3 `world_builder/` (the pipeline library, about 50 modules)

| Area | Modules | Role |
|---|---|---|
| Per-frame path | `engine.py`, `frontend.py`, `keyframes.py`, `frame_quality.py`, `redaction.py`, `relocalizer.py`, `live_depth.py`, `events.py` | "Observe cheaply, build expensively" |
| Segment estimates | `backend.py`, `backends/classical.py`, `backends/unposed.py`, `geometry.py`, `bundle.py` | Poses per segment, before a global solve exists |
| Global solve, publish | `global_solve.py`, `solve_masks.py`, `coherence_publish.py`, `coherence_gate.py`, `coherence_scale.py`, `anchor_verify.py`, `final_scale_guard.py`, `components.py`, `area_build.py` | GLOMAP solve, publish seam, components record |
| Surface | `surface_pipeline.py`, `surface.py`, `transients.py`, `depth_consistency.py`, `surface_render.py` | Mesh and LOD ladder; the surface page |
| Appearance | `appearance_pipeline.py`, `appearance.py`, `appearance_render.py`, `native_chrome.py`, `photographic.py`, `raw_imagery.py` | The photographic top rung (`raw_imagery` is an off-by-default research bypass) |
| Dense | `dense_pipeline.py`, `dense.py`, `dense_render.py` | Optional point cloud (off) |
| Guidance | `guidance_coverage.py`, `guidance_imagery.py`, `photo_floorplan*.py` | The capture dial: coverage, photos, floor plan |
| Storage | `store.py`, `records.py`, `schema.py`, `intrinsics_store.py`, `reredaction.py`, `inspect.py` | The world tree on disk |
| Sparse render | `render.py` | The viewer's bottom rung |

Invariants (the how is in the entries): one writer per world (T6.1); an image is written before its journal line (T1.5, T6.2); the derived tree is rebuildable and digest-checked against
`keyframes.jsonl` (T6.4); faces are redacted before any keyframe touches disk (T3.4); `observe` never reaches `build` (T4.9); every stage is idempotent (skips work whose digest matches) and
stop-aware (records `stopped`, never a half-written artifact; T5.2, T5.5).

### 3.4 Other cartridges and `scripts/`

**Other cartridges.** `cv_lab/` + `experiments/` produce every `frame_result`; eight experiments: baseline, edge_detection, frame_quality, feature_detection, optical_flow, redaction_impact,
object_detection, depth. `scene/` (in-process): detection, tracking, facing, relations; nothing persisted. `document_memory/` (in-process): steadiness gate, text detection, dwell, OCR, a
library, BM25 search. `object_memory/`: a follower process (`scripts/object_memory_session.py`), deliberately absent from `/cartridges`.

**Scripts.** Spawned by the Tower: `world_build_session.py` (the builder, T4.3), `world_solve.py` (the solve child, `main` L45), `world_finish_pending.py` (the idle finisher),
`object_memory_session.py`. Offline tools: `world_replay.py` (deterministic rebuild from raw frames), `world_finalize.py` (finish an unfinished finalization), `world_refinish.py` (full
product rebuild); per stage `world_surface.py`, `world_appearance.py`, `world_densify.py`, `world_registration.py`, `world_reredact.py`; inspection `world_inspect.py`, `world_render.py`,
`world_coherence_report.py`, `world_builder_env_check.py`. `start_tower.ps1` loads `.env` and binds `0.0.0.0`. Every `--root` flag goes through `artifact_root_arg`, enforced by
`test_every_cli_root_flag_routes_through_the_guard`.

---

## 4. iOS code tour, by area

The phone's frame path out, and the result path back:

```
 Ray-Ban Meta glasses
   |  DAT SDK: DeviceSession + camera stream (raw codec, 24 fps requested, .low = 360x640 by default)
   v
 GlassesConnection  (GlassesConnection.swift; the ONLY type that touches DAT; capture surface is #if DEBUG)
   |  videoFramePublisher (DAT thread) -> hop to main actor
   |  frameCount += 1            seq = every delivered frame, sent or not
   |  FrameRateGate.shouldSelect(at:)   ~15 fps delivered, deadline-based
   |  makeUIImage (selected frames only) -> @Published latestCapturedFrame
   v
 ProjectManager  (Combine sinks: the only bridge; the two never reference each other)
   |  latestCapturedFrame          -> TowerClient.sendFrame
   |  cameraStreamDidStart / Stop  -> sendStreamStart / sendStreamStop   (the "stream bracket")
   |  $status == .online while streaming -> sendStreamStart (reopen the bracket after a reconnect)
   v
 TowerClient  (one URLSessionWebSocketTask)
   |  gates: online + inside bracket + not paused
   |  SendWindow: 5 slots (= 15 fps x 1/3 s), 2 s stall -> replace the socket
   |  JPEG q0.5 -> base64 -> {"type":"frame","seq","tx_seq","width","height","format":"jpeg","data"}
   v
 Tower  (/ws -> capture on disk -> World Builder process -> worlds/sessions)

 ---------------------------------------- back to the phone ----------------------------------------
 Tower --WS text--> TowerClient.receiveLoop -> handleInboundMessage (switch on "type")
    frame_result | frame_error          -> latestFrameResult / latestFrameRefusal (+ CV Lab events)
    cv_lab_status | cv_lab_error        -> CV Lab control events
    cartridges | result_subscribed | cartridge_result | result_error
                                        -> resultEvents ("cartridgeResults" publisher)
          |-> TowerWorldBuilderClient (world_builder.status/2026-09-10)
          |      -> walk report (WalkScoped) + geometry address -> WorldBuilderViewModel
          |           -> HTTP GET /worlds/{id}/geometry/manifest, /segment/{i}   (diagnostic fragment gallery)
          |           -> WorldPresentation: stage, 3D ladder -> render target
          |           -> WorldRenderViewerModel: GET /worlds/{id}/render -> WKWebView on glasses-world:
          |-> TowerExperimentalCVClient / TowerDocumentMemoryClient / TowerSceneUnderstandingClient
 Object Memory: HTTP only (/object-memory/*, /cartridges/{id}/session), never on the socket.
```

The target builds with `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor` (everything is main-actor unless marked `nonisolated`), deployment target
iOS 26.5, device family iPhone + iPad. About 59,500 lines of Swift in the app. Each area below says what it owns and where; §5
(the K entries) says how the main paths work.

### 4.1 Shell (app entry, object graph, navigation)

- **Responsibility:** start the app, configure DAT, own the one runtime graph, choose the on-screen workspace, and present the sheets
  (connections, cartridge tray, Settings, Developer Tools, onboarding).
- **Main files:**
  - `GlassesApp.swift`: `@main`; calls `Wearables.configure()` and routes `glasses://` callbacks carrying `metaWearablesAction` to
    `Wearables.shared.handleUrl`.
  - `ContentView.swift`: root view; holds `@StateObject project = ProjectManager()` and the `workspace` switch over `CartridgeWorkspace`
    (ContentView.swift:256).
  - `ProjectManager.swift`: builds and wires the graph (K1, K2).
  - `Cartridges/Cartridge.swift`: static catalog of 8 entries. Five have workspaces (World Builder, CV Lab, Object Memory, Document Memory,
    Scene Understanding); Visual Q&A, Accessibility and Environmental Memory are `.notBuilt`.
  - `Workspaces/CartridgeWorkspace.swift`, `Workspaces/HomeWorkspaceView.swift`.
  - `Views/`: `ShellStatusBar`, `ConnectionSheet`, `CartridgeDrawerView`, `SettingsView`, `DeveloperToolsView`, `OnboardingView`.
- **Invariants:**
  - Nothing puts `.id(...)` at or above `ContentView`; the workspace switch sits *below* `project`, so changing cartridge swaps a subtree
    and never tears down the camera or the socket.
  - Cartridge clients live in `CartridgeClients` (owned by `ProjectManager`), so they outlive each workspace's `@StateObject`.
  - Choosing a cartridge in the tray sends nothing to the Tower.
  - `ProjectManager` is created in exactly one place, so there is one `GlassesConnection` and one socket per scene (log line `[Glasses][Init] GlassesConnection created` once
    per launch). Known gap: iPad multi-window would create a second graph; multiple scenes are not disabled.

### 4.2 Glasses and capture

- **Responsibility:** DAT registration, device availability and camera permission (all builds). In DEBUG builds only: the capture session,
  camera stream, frame listener, Motion (IMU) probe and IMU recorder.
- **Main files:**
  - `GlassesConnection.swift` (1,915 lines); the camera/session code is inside `#if DEBUG` (lines 763-1914).
  - `FrameRateGate.swift`: the gate, plus `MonotonicClock`.
  - `SenderMetrics.swift`: sender counters and derived rates.
  - `DeviceHealth.swift`: thermal state, Low Power Mode, battery.
  - `FramePTSProbe.swift`: DEBUG-only; observes which clock DAT's frame timestamps use.
  - `MotionProbe.swift`.
  - `IMURecorder.swift`: DEBUG only; one file per session in `Documents/imu-logs/`, 200 MB default cap.
- **Invariants:**
  - Only `GlassesConnection` touches DAT; there is one camera pipeline.
  - Capture starts only through `startCameraSession()` (K3). Callers: Home, the World Builder capture button, the CV Lab camera card, the
    Object Memory coordinator.
  - Callbacks from a replaced session are ignored (`isCurrentSession`).
  - Resolution is `.low` (360x640) unless a developer changes it. 720p rejects 73.3% of frames as blurred for World Builder (`min_sharpness
    = 25.0`), and four times the pixels means four times the unredacted bystander imagery the Tower's recorder may keep.

### 4.3 Tower connection

- **Responsibility:** the one WebSocket (connect, handshake, bounded reconnect); the frame send path and stream bracket; inbound decoding
  and routing; the CV Lab control plane; the result-channel outbound messages (`cartridges`, `result_subscribe`, `result_unsubscribe`); `GET
  /health`.
- **Main files:**
  - `TowerClient.swift` (2,142 lines).
  - `SendWindow.swift`: slot accounting and capacity arithmetic.
  - `TowerSocketMessages.swift`: decoders for `frame_result`, `frame_error` and the CV Lab replies.
  - `Cartridges/Integration/CartridgeResultChannel.swift`: offers, acks, envelopes, errors, events.
  - `TowerConfiguration.swift`: chooses the Tower address, fixed once per process. A DEBUG launch environment's `GLASSES_TOWER_AUTHORITY`
    wins, then the address saved in Settings (if the private-network policy accepts it), then the built-in `100.110.156.55:8000`.
  - `TowerAddress.swift`: the private-network policy.
  - `TowerHealth.swift`, `TowerConnectionTester.swift` (Settings' read-only `GET /cartridges` probe).
- **Invariants:**
  - `status` is `.offline`, `.connecting`, `.online` or `.failed`; the Tower never speaks first.
  - Frames go out only inside a bracket and while not paused.
  - Result-channel handlers decode and publish only; they never touch `status`, the send window or the bracket.
  - Automatic code uses `connectIfIdle()`, which never refills the retry budget; a user's tap on Connect uses `connect()`, which does.

### 4.4 World Builder workspace

- **Layers:** `TowerClient` owns the socket; `TowerWorldBuilderClient` owns the subscription and the contract; `WorldBuilderViewModel`
  republishes to SwiftUI and fetches geometry over HTTP; the views draw what they are given.
- **Client and model** (`Workspaces/WorldBuilder/`):
  - `TowerWorldBuilderClient.swift`: contract `world_builder.status/2026-09-10`, payload decoders, client.
  - `WorldBuilderClient.swift`: the `WorldBuilderClient` protocol **and** `WorldBuilderViewModel`.
  - `WorldModel.swift` (`WorldModelState`), `WorldSession.swift` (session gate), `WorldWalkScope.swift` (`WalkScoped`),
    `WorldSelection.swift`.
  - `WorldBuilderSessionController.swift`: cartridge session start and stop.
- **Geometry (diagnostic fragment gallery):** `WorldGeometry.swift` (manifest and segment decoders), `WorldGeometryClient.swift` (HTTP plus
  the `WorldGeometryStore` actor cache), `WorldFragmentsView.swift`, `WorldClusters.swift`, `WorldViewport.swift`, `WorldCanvasView.swift`.
- **Presentation (pure values):** `WorldPresentation.swift` (`WorldGeometryStatus`, `WorldEvidence`, `WorldGeometryAccount`, `WorldStage`,
  `WorldReconstruction`), `WorldPhotographic.swift`, `WorldComponents.swift`, `WorldFinishProgress.swift`, `WorldPanelPhase.swift`,
  `WorldListingPresentation.swift`.
- **3D viewer:**
  - `WorldRenderViewerModel.swift` (fetch, render lifecycle, revision follower), `WorldRenderViewerState.swift` (states, rung ladder),
    `WorldRenderWebView.swift` (the one `WKWebView` and its navigation gate), `WorldAssetTransport.swift` (`glasses-world:` scheme handler).
  - `WorldChromeProtocol.swift`, `WorldChromeBridge.swift`, `WorldChromeOverlay.swift`: the "native chrome" protocol, where the page posts
    its controls and the phone draws them.
  - `WorldRenderViewer.swift`: full-screen scene.
  - `WorldInlinePanel.swift`: one panel for the whole walk: capture health and dial during the walk, finishing progress after Stop, then the
    inline 3D world (expandable to full screen).
- **Walk-time guidance:** `WorldCaptureDial.swift` + `WorldCaptureDialView.swift`, `WorldCoverage.swift`, `WorldCoverageMapView.swift` (Map
  face, off by default), `WorldImagery.swift` + `WorldImageryViews.swift` (live strip and landed thumbnails, at most 16 per stage, memory
  only), `WorldPhotoFloorplan.swift` + `WorldPhotoFloorplanView.swift`, `WorldCurrentPiece.swift`, `WorldLookBack.swift` (haptic look-back
  cue), `CaptureHealthModel.swift` + `CaptureHealthView.swift`.
- **Saved worlds:** `WorldLibrary.swift` (`GET /worlds`), `WorldPickerView.swift`.
- **Invariants:**
  - The client owns no socket and no geometry; a walk's identity and state travel as one value (`WalkScoped`).
  - A snapshot is shown as this session's only when `WorldSessionGate.binding` (WorldSession.swift:151) allows it: the phone's bracket is
    open, plus a live-capture session, a capture directory and no end time.
  - The geometry fetch is keyed on revision, not arrival; the newest fetch wins. Delta envelopes are refused; snapshots only.
  - The only 3D renderer is the Tower's page in WebKit (no SceneKit, RealityKit or Metal). The web view can load only its own page and keeps
    nothing on disk.
  - Opening the workspace sends the cartridge session `start`; leaving it sends `stop`.

### 4.5 Cartridges

- **Shared layer** (`Cartridges/Integration/`):
  - `CartridgeClient.swift`: `TowerCapabilities` (supported contract set, catalog-id to Tower-name map) and the `CartridgeClient` protocol
    (a cartridge id plus availability, deliberately nothing generic).
  - `CartridgeAvailability.swift` (`resolve`, `CartridgeFailure`; K23), `CartridgeResultSubscription.swift` (shared bookkeeping, resubscribe
    budget of 3 per connection), `CartridgeClients.swift` (the container), `CartridgePhase.swift`, `Observation.swift`,
    `VisualArtifact.swift`, `UncachedURLSession.swift`.
- **Scene Understanding:** contract `scene_understanding.live/2026-08-27`, the only cartridge whose result type is `live`. Subscribes only
  while its screen is visible (the Tower's people detector runs only while somebody streams and watches). Stop discards the reading; pause
  keeps the last-known one.
- **Experimental CV Lab:** `experimental_cv.status/2026-08-27` on the result channel, plus a socket control plane: `cv_lab_status`,
  `cv_lab_start`, `cv_lab_pause`, `cv_lab_resume`, `cv_lab_stop`, each with a bounded `request_id`. Its Pause is a frame hold on
  `TowerClient` (`pauseFrameSending`), not a camera operation.
- **Document Memory:** `document_memory.status/2026-09-07` pushed on the socket; `document_memory.library/2026-09-07` pulled over HTTP
  because it carries bulk, sensitive document text.
- **Object Memory:** HTTP only, deliberately undeclared on the socket. `object_memory.observations/2026-08-26`: `GET
  /object-memory/observations` and `/object-memory/last-seen/{class}`. Session verbs: `GET /cartridges/{id}/session` and `POST
  /cartridges/{id}/session/{start|stop|pause|resume}`.
- **Invariants:**
  - Contracts are opaque dated strings compared only for equality; a mismatch is shown as "unsupported contract" and never decoded on a
    guess. Payloads are decoded all-or-nothing.
  - No cartridge view model holds a socket or the camera (`TowerClientTests.testCartridgeViewModelsSendNothingToTheTower`).

### 4.6 Shared components and support

- `Views/Components/`: `TowerReachabilityReader` (hands a workspace one `Bool`, so it does not re-render at the Tower's ~15 Hz reply rate),
  `ViewfinderCard` (same live preview on Home and World Builder), `StatusPill`, `MetricTile`, `ReadableText`, `CartridgeStatePanel`,
  `TowerConnectionRow`, `TowerRecovery`, `ShellPieces`.
- `Presentation/StateDisplay.swift`: human words for DAT and Tower enums, so raw Swift case names never reach the screen.
- `Onboarding.swift`: first-run cards, stored in `UserDefaults`.
- `ios/scripts/`: `contract-drift-check.py` (Swift contract identifiers vs a live Tower; exit 0 agreement, 1 drift, 2 unreachable),
  `cross-stack-constants-check.py` (the Swift names every constant the Tower sends), `swift-structure-check.py` (structural check for hosts
  with no Swift toolchain).

---

## 5. Key Functions

The centre of the document: 55 Tower entries (T) and 29 iOS entries (K), curated, not exhaustive. Each gives **WHAT** (one
sentence), **HOW** (inputs → steps → outputs) and **WHY** (the non-obvious choice), then who calls it and what it calls;
parameters appear only where they change behaviour. A T entry's **Phone side** line and a K entry's **Tower side** line name the
matching code on the other end of the wire.

### Index

**Tower**

| ID | Symbol | file:line | WHAT |
|---|---|---|---|
| T1.1 | `websocket_endpoint` | `tower/routes/ws.py:1064` | The single `/ws` receive loop: routes every message type and cleans up when the socket dies. |
| T1.2 | `_handle_frame_message` | `tower/routes/ws.py:81` | Decodes one frame, sends the CV module's answer back, then fans the frame out. |
| T1.3 | `_fan_out_frame` | `tower/routes/ws.py:316` | After the reply, offers the decoded frame to the recorders, then to every live cartridge. |
| T1.4 | `_start_capture` | `tower/routes/ws.py:842` | On `stream_start`, starts a recording (this process mints the capture id) and tells the supervisor. |
| T1.5 | `CaptureRecorder.write_frame` | `tower/capture.py:300` | Durably appends one raw frame and its journal line to the open capture, the directory the builder follows. |
| T2.1 | `create_app` | `tower/main.py:1433` | Builds the FastAPI app and wires every long-lived object; `app = create_app()` runs at import (L1646). |
| T2.2 | `ModuleContainer.process` | `tower/modules/container.py:136` | Hands raw bytes to the single CV module and classifies any failure. |
| T2.3 | `apply_session_action` | `tower/routes/sessions.py:97` | `POST /cartridges/{cartridge}/session/{action}`: the phone opens, pauses, resumes or stops World Builder or Object Memory. |
| T2.4 | `CaptureWorkerSupervisor.capture_opened` | `tower/capture_workers.py:456` | When a recording starts, attaches a child process for every worker spec whose gate is open. |
| T2.5 | `_world_build_spec` | `tower/main.py:183` | Describes the World Builder child as a command line. |
| T2.6 | `LiveSession.offer_frame` | `tower/live_session.py:530` | The non-blocking hand-off of a frame to an in-process cartridge; the base class of Scene and Document Memory. |
| T3.1 | `follow_capture` | `scripts/world_build_session.py:300` | Yields frames from the capture directory as the Tower writes them, across reconnects. |
| T3.2 | `WorldBuilderEngine.observe` | `tower/world_builder/engine.py:361` | The per-frame entry point: reject, tracking lost, or accept-and-persist. |
| T3.3 | `KeyframeSelector.evaluate` | `tower/world_builder/keyframes.py:377` | The keyframe policy as a pure function: ACCEPT, REJECT, SKIP or TRACKING_LOST for one frame. |
| T3.4 | `FaceRedactor.redact` | `tower/world_builder/redaction.py:529` | Fills detected faces in a keyframe JPEG before it is stored. |
| T3.5 | `LookBackRelocalizer.note_lost` | `tower/world_builder/relocalizer.py:1270` | On a tracking loss, opens a recovery episode matching new frames against pre-loss keyframes. |
| T3.6 | `LiveDepthWorker.submit` | `tower/world_builder/live_depth.py:165` | Queues an accepted keyframe for depth prediction (`moge2-vitl`) without blocking the frame path. Always on. |
| T3.7 | `BackgroundSolver.maybe_launch` | `scripts/world_build_session.py:943` | Spawns a background global solve every `every` accepted keyframes (50 in the Tower-spawned builder). |
| T3.8 | `compute_coverage` | `tower/world_builder/guidance_coverage.py:400` | A pure computation of the status `guidance.coverage` block (which stations the solve has seen); drives the capture dial (K22). |
| T4.1 | `CartridgeSession.stop` | `tower/cartridge_session.py:371` | The phone's Stop. Never refused, from any state. |
| T4.2 | `_stop_capture` | `tower/routes/ws.py:977` | On `stream_stop`, closes this connection's recordings and tells the supervisor. |
| T4.3 | `main` (the builder process) | `scripts/world_build_session.py:2023` | The builder process's whole life. |
| T4.4 | `WorldBuilderEngine.stop_session` | `tower/world_builder/engine.py:657` | Closes the session record, optionally keeping the writer lock. |
| T4.5 | `BackgroundSolver.run_final` | `scripts/world_build_session.py:1149` | Runs the finalization solve as a child until it ends or a hard stop arrives. |
| T4.6 | `solve` | `tower/world_builder/global_solve.py:1189` | Reconstructs a session's keyframes into one global solution (pycolmap/GLOMAP) and publishes it. |
| T4.7 | `gate_and_publish` | `tower/world_builder/coherence_publish.py:1402` | The publish seam: gate the candidate (when on), scale-guard it (final solves only), write it, then write its records. |
| T4.8 | `guard_publication` | `tower/world_builder/final_scale_guard.py:927` | Checks the room's cameras against metric-scale evidence at the publish seam; returns publish or withhold. Always on. |
| T4.9 | `WorldBuilderEngine.build` | `tower/world_builder/engine.py:891` | Rewrites a session's derived tree from its persisted journal. |
| T4.10 | `regate_published` | `tower/world_builder/coherence_publish.py:1684` | Re-runs depth, metric scale and the gate in place on an already published solve, without redoing the solve. |
| T5.1 | `final_surface_stages` | `scripts/world_build_session.py:1740` | The finished world's surface stages, run after Stop in the builder process. |
| T5.2 | `surfacify` | `tower/world_builder/surface_pipeline.py:707` | Builds the surface of a solved session under `surface/<sid>/`. |
| T5.3 | `ensure_depth_stage` | `tower/world_builder/surface_pipeline.py:245` | Returns the depth stage's alignment record and work directory, running the stage first if it is missing or came from another solve. |
| T5.4 | `build_appearance` | `tower/world_builder/appearance_pipeline.py:217` | Builds the photographic layer on a solved, surfaced session: the viewer's top rung. |
| T5.5 | `densify` | `tower/world_builder/dense_pipeline.py:1809` | A dense point cloud anchored to the global solve. Off by default; walk 10 shows `unavailable`. |
| T5.6 | `build_landing` | `tower/world_builder/photo_floorplan_worker.py:497` | Builds one photo floor-plan "landing" for a newly published coverage block: one tile per room component. |
| T6.1 | `WorldStore.acquire_writer_lock` | `tower/world_builder/store.py:980` | Takes the per-world single-writer lock (`worlds/<wid>/LOCK`). |
| T6.2 | `WorldStore.append_keyframe` | `tower/world_builder/store.py:471` | Appends one keyframe record to `keyframes.jsonl`, the authoritative input to every rebuild. |
| T6.3 | `WorldStore.write_derived` | `tower/world_builder/store.py:634` | Writes a session's rebuildable outputs. |
| T6.4 | `WorldStore.read_derived` | `tower/world_builder/store.py:698` | Reads the derived reconstruction, refusing stale output. |
| T6.5 | `write_json_atomic` | `tower/storage.py:139` | The atomic-write primitive under every store. |
| T7.1 | `_subscribe` | `tower/routes/results_ws.py:153` | Validates a `result_subscribe` against the registry and registers it. |
| T7.2 | `ResultHub.poll_once` | `tower/results/publisher.py:1090` | One pass over every watched target. |
| T7.3 | `make_snapshot_for` | `tower/results/__init__.py:54` | Builds the callable mapping (cartridge, result type, world, session) to a `Snapshot`. |
| T7.4 | `WorldBuilderStatusProducer.snapshot` | `tower/results/world_builder.py:575` | One complete `world_builder.status/2026-09-10` payload from the world's files. Never partial, never a delta. |
| T7.5 | `build_manifest` | `tower/results/world_builder_geometry.py:511` | The geometry manifest the phone polls: segments with content hashes and placements. |
| T7.6 | `build_world_render` | `tower/results/world_builder_render.py:736` | The viewer HTML for one session, at the best rung available. |
| T8.1 | `ObjectMemoryEngine.observe` | `tower/object_memory/engine.py:269` | Per frame, decides which detections become recorded observations. |
| T8.2 | `build_last_seen` | `tower/results/object_memory.py:319` | Answers "when was this category last in view?" (`GET /object-memory/last-seen/{class}`). |
| T8.3 | `DocumentMemoryEngine.observe` | `tower/document_memory/engine.py:267` | The cheap per-frame path. Raw pixels never leave it. |
| T8.4 | `DocumentMemoryEngine._record_inner` | `tower/document_memory/engine.py:441` | The expensive path: crop, OCR, dedup, identify and persist one finished dwell. |
| T8.5 | `DocumentMemory.search_text` | `tower/document_memory/retrieval.py:366` | Lexical BM25 search over the stored OCR text, page by page (`GET /documents/search`). |
| T8.6 | `SceneEngine.observe` | `tower/scene/engine.py:154` | One frame in, one `SceneState` out. Nothing is persisted. |
| T8.7 | `CVLab.process` | `tower/cv_lab/lab.py:508` | Runs the armed experiment on one frame, or refuses legibly. Every `frame_result` comes from here. |
| T8.8 | `CVLab.start` | `tower/cv_lab/lab.py:793` | Arms a registered experiment on `cv_lab_start`, and returns at once. |
| T8.9 | `handle` (CV Lab socket) | `tower/routes/cv_lab_ws.py:60` | Dispatches `cv_lab_status`, `cv_lab_start`, `cv_lab_pause`, `cv_lab_resume` and `cv_lab_stop` to the Lab. Never raises, except to pass on a `WebSocketDisconnect`. |

**iOS**

| ID | Symbol | file:line | WHAT |
|---|---|---|---|
| K1 | `ProjectManager.init(...)` | `ProjectManager.swift:118` | Builds the app's single runtime graph and wires the camera to the Tower. |
| K2 | `ProjectManager.startAutomaticConnections()` | `ProjectManager.swift:283` | Once per process, does the only two things the app may do without being asked. |
| K3 | `GlassesConnection.startCameraSession()` | `GlassesConnection.swift:1015` | The single entry point that starts glasses capture: creates a DAT session, and once DAT reports `.started` adds the camera and starts its stream. |
| K4 | `GlassesConnection.setupStreamListeners(for:of:recorder:)` | `GlassesConnection.swift:1385` | Registers DAT's stream listeners; the video listener numbers every frame, gates it to the Tower rate, decodes the frames it keeps, and publishes them. |
| K5 | `FrameRateGate.shouldSelect(at:)` | `FrameRateGate.swift:126` | Decides whether a frame arriving at `now` is forwarded, holding a fixed delivered rate. |
| K6 | `GlassesConnection.startMotion(on:)` | `GlassesConnection.swift:1523` | Attaches DAT's experimental Motion (IMU) capability to a started session, for the developer probe and IMU recorder. |
| K7 | `TowerClient.openConnection(to:)` | `TowerClient.swift:573` | Opens the one WebSocket and calls it `.online` only after a ping/pong. |
| K8 | `TowerClient.scheduleReconnect()` | `TowerClient.swift:700` | After a failure, queues one delayed reconnect from a bounded schedule. |
| K9 | `TowerClient.sendFrame(_:width:height:sequence:)` | `TowerClient.swift:762` | Sends one selected frame as one JSON text message in four named steps, recording exactly one terminal outcome per frame. |
| K10 | `TowerClient.handleInboundMessage(_:)` | `TowerClient.swift:1584` | Routes every inbound text message by `type`; a frame result must pass a CV Lab staleness gate before it is published or counted. |
| K11 | `TowerClient.handleFrameError(_:)` | `TowerClient.swift:1677` | Handles the Tower's per-frame refusal: a changed refusal is published at once, a repeat at most once per second, and every decoded refusal goes on to CV Lab consumers. |
| K12 | `WorldBuilderSessionController.workspaceDidAppear(isTowerReachable:)` | `WorldBuilderSessionController.swift:164` | Tells the Tower the World Builder screen is open: `POST /cartridges/world_builder/session/start` on appear, `stop` on leave. |
| K13 | `TowerWorldBuilderClient.subscribeIfPossible()` | `TowerWorldBuilderClient.swift:1199` | Opens the World Builder status subscription when the socket, the Tower's declaration and the contract all allow it. |
| K14 | `TowerWorldBuilderClient.apply(_ envelope:)` | `TowerWorldBuilderClient.swift:1591` | Turns one status envelope into the client's walk report, and returns the geometry address to fetch when the presented state holds a world. |
| K15 | `WorldBuilderViewModel.walkReportDidChange(to:)` | `WorldBuilderClient.swift:714` | The single entry for the client's walk reports: publishes walk and state as one value, reacts to a new walk, then a new state, then forwards coverage to the imagery models. |
| K16 | `WorldBuilderViewModel.geometryDidChange(worldID:sessionID:revision:)` | `WorldBuilderClient.swift:1000` | Fetches and publishes the geometry the Tower just named, in eight named steps, refusing anything a newer push has superseded. |
| K17 | `WorldStage.stage(for:evidence:finalization:photographic:)` | `WorldPresentation.swift:465` | Names the world's stage in the screen's vocabulary (Mapping, Building, Improving, Finalizing, Saved, Partial, Interrupted, Needs retry) from the Tower's evidence. |
| K18 | `WorldReconstruction.ladder(target:stage:finalSolve:evidence:photographic:)` | `WorldPresentation.swift:1015` | Decides the screen's 3D offer through two ordered rule tables: `.final`, `.partial` with a note, or `.unavailable` with a reason. |
| K19 | `WorldRenderViewerModel.load()` | `WorldRenderViewerModel.swift:1129` | Fetches the Tower's composed 3D page and starts a bounded render; a finished fetch never counts as drawn content. |
| K20 | `WorldRenderViewerModel.followRevisions()` | `WorldRenderViewerModel.swift:752` | While a picture is on screen, polls the render revision; a better rung replaces the page by itself, a same-rung rebuild is offered as a button. |
| K21 | `WorldRenderWebView.Coordinator.webView(_:decidePolicyFor:decisionHandler:)` | `WorldRenderWebView.swift:485` | The web view's navigation gate: allows only the initial load of its own page URL and a bounded self-reload, and resets the native-chrome bridge for each allowed document. |
| K22 | `WorldCaptureDial.init(components:stationsOmitted:frameRevision:newest:motionSector:photos:)` | `WorldCaptureDial.swift:157` | Builds the walking-stage capture dial: twelve 30° directions per reconstructed piece, from the Tower's frozen coverage receipt. |
| K23 | `CartridgeAvailability.resolve(declared:supported:isTowerReachable:knownToThisBuild:)` | `CartridgeAvailability.swift:208` | Decides whether a cartridge can be used. |
| K24 | `TowerSceneUnderstandingClient.workspaceVisibilityChanged(isVisible:)` | `SceneUnderstandingClient.swift:305` | Subscribes only while the Scene screen is visible, and keeps at most one subscription open. |
| K25 | `SceneUnderstandingState.forReading(_:)` | `SceneUnderstandingModel.swift:808` | The only constructor from a decoded reading to a screen state. |
| K26 | `TowerExperimentalCVClient.project(_:)` | `ExperimentalCVClient.swift:877` | Maps a CV Lab status document to the workspace state. |
| K27 | `TowerDocumentMemoryClient.handle(_:)` | `DocumentMemoryClient.swift:427` | Applies Document Memory's status-channel events; when the library revision moves, the view model reruns its standing query over HTTP. |
| K28 | `ObjectMemoryRecordingCoordinator.startSequence()` | `ObjectMemoryRecording.swift:558` | Starts Object Memory: the Tower producer first, then the camera if nobody holds it, then a bounded check that a producer is following the capture. |
| K29 | `TowerObjectMemoryClient.apply(_:)` | `ObjectMemoryClient.swift:607` | Sends one session verb, then replaces its answer with a fresh read of liveness, keeping any refusal's explanation. |

**Tower (T1-T8).** 55 entries, curated from the Tower's key-function inventory (59 entries) and re-verified at `66414dc1`.
Four entries are folded into their callers: `SightingTracker.observe`, `PageFinder.find`, `Tracker.update` and `SceneLive.latest`.
Corrections to the inventory are in §11.A.

### T1. Frame arrives (`tower/routes/ws.py`, `tower/capture.py`)
#### T1.1 `websocket_endpoint` · `tower/routes/ws.py:1064`
**WHAT.** The single `/ws` receive loop: routes every message type and cleans up when the socket dies.
**HOW.** Per connection: a `connection_token`, a `_ConnectionSender` (send lock), a channel holder. Dispatch: `ping`→`pong`; `frame`→T1.2; `stream_start`→metrics window, GPU yield, T1.4;
`stream_stop`→T4.2; CV Lab/result types to their handlers; unknown string type→`protocol_error`; malformed JSON or non-string `type`→logged, ignored (no reply). `finally`: close channels,
stop this connection's recordings as `disconnect`; on the last client, arm the WB deferred stop and tear down off-loop.
**WHY.** The token stops a dead socket from disarming a live socket's recording: uvicorn needs 20-40 s to notice a dead socket (ping interval and timeout both 20 s), while iOS reconnects in
0.5 s and re-sends `stream_start` (`CaptureRecorder.stop` docstring).
**Called by** the router (`@router.websocket("/ws")`). **Calls** T1.2, T1.4, T4.2.
**Phone side:** K7 (`ping`), K9 (`frame`), K10 (every reply), `sendStreamStart`/`sendStreamStop` (the bracket).
#### T1.2 `_handle_frame_message` · `tower/routes/ws.py:81`
**WHAT.** Decodes one frame, sends the CV module's answer back, then fans the frame out.
**HOW.** `parse_and_decode_frame`; a `FrameError`→`frame_error invalid_frame`, not fanned out. Else `ModuleContainer.process(raw_bytes)`→`frame_result` (`seq`, `processing_ms`,
`result_value`, `result_label`, `stage_ms`, `cv_lab` provenance)→T1.3. Skip and unavailable errors→`frame_error` with the module's reason, `frame_skipped`, or `module_unavailable`; those
frames still fan out.
**WHY.** What the CV module thinks of a frame is its own business; a paused or refusing CV Lab must never starve the recording.
**Called by** T1.1. **Calls** T2.2, T1.3.
**Phone side:** K9 sends the frame; K10 and K11 handle the reply.
#### T1.3 `_fan_out_frame` · `tower/routes/ws.py:316`
**WHAT.** After the reply, offers the decoded frame to the recorders, then to every live cartridge.
**HOW.** `_record_capture` (each observer's `write_frame`, L401), then `_offer_to_cartridges` (each consumer's `offer_frame(raw_bytes, source_seq=…)`); every call has its own `try`.
**WHY.** After the reply because recording does an fsync'd write. Raw bytes mean no decode runs on the event loop for an idle cartridge. A cartridge bug can never cost the connection.
**Called by** T1.2, on all three decoded paths. **Calls** T1.5, T2.6.
#### T1.4 `_start_capture` · `tower/routes/ws.py:842`
**WHAT.** On `stream_start`, starts a recording (this process mints the capture id) and tells the supervisor.
**HOW.** For each observer, in order: (1) stop any recording still running with `END_REASON_DISCONNECT`; (2) `continues = resumable_capture()`; (3) `observer.start(owner, continues)` returns
the capture id; (4) `supervisor.capture_opened` via `asyncio.to_thread`.
**WHY.** A superseded recording ends as `disconnect`, so the reconnect is offered as a continuation (before, a mid-walk reconnect produced two worlds, each "Complete" and half the walk). The
supervisor call runs off the loop because Pause and Stop contend for the same lock (a 1.95 s stall was measured).
**Called by** T1.1. **Calls** `CaptureRecorder.start` (`capture.py:279`), T2.4.
**Phone side:** `TowerClient.sendStreamStart` (`TowerClient.swift:1065`), re-sent after a reconnect (K1).
#### T1.5 `CaptureRecorder.write_frame` · `tower/capture.py:300`
**WHAT.** Durably appends one raw frame and its journal line to the open capture, the directory the builder follows.
**HOW.** Not recording→False. A time or byte bound would be exceeded→`stop(END_REASON_BOUNDED_LIMIT)`, False. Else write `frames/<source_seq:08d>.jpg` (temp file, fsync, replace), then
`append_jsonl` its record (seqs, `received_at`, monotonic stamp, `relpath`, size, w/h).
**WHY.** Image first, line second, so a line never points at a missing image. A bound is an expected outcome, so it never raises on the frame path; the follower just sees an ordinary close.
**Called by** `ws._record_capture`, via T1.3. **Calls** `append_jsonl` (`storage.py:300`), `stop`.

### T2. Wiring, the CV module, supervision
#### T2.1 `create_app` · `tower/main.py:1433`
**WHAT.** Builds the FastAPI app and wires every long-lived object; `app = create_app()` runs at import (L1646).
**HOW.** Order: (1) `ConnectionTracker`, the CV module in a `ModuleContainer`, the recorders; (2) two gates reading `CartridgeSession.is_active()`; (3) the finisher chore (idle only when no
capture worker is alive); (4) the supervisor and two `CartridgeSession`s (World Builder uses `STOP_POLICY_REQUEST`); (5) `build_live_cartridges` become `frame_consumers`, then `build_hub`;
(6) `asyncio.run(container.load_and_start())`, then the routers.
**WHY.** The chore is built here but started in `lifespan` (L1376): tests use `TestClient(create_app())` without lifespan, and a chore started at construction would spawn a subprocess in
every test.
**Called by** module import (uvicorn), and tests. **Calls** `_build_capture_worker_supervisor` (uses T2.5), `build_hub`.
#### T2.2 `ModuleContainer.process` · `tower/modules/container.py:136`
**WHAT.** Hands raw bytes to the single CV module and classifies any failure.
**HOW.** State not `ACTIVE`→`ModuleUnavailableError`. Else `module.process(raw_bytes)`: a `FrameProcessingError`→`FrameSkippedError` carrying the module's `reason` (module stays ACTIVE); any
other exception→module FAILED, `ModuleUnavailableError`.
**WHY.** One bad frame is not a crash. The transport maps the two classes onto `frame_error` codes without knowing any module.
**Called by** T1.2. **Calls** `modules/experimental_cv.py:95`, which calls T8.7.
#### T2.3 `apply_session_action` · `tower/routes/sessions.py:97`
**WHAT.** `POST /cartridges/{cartridge}/session/{action}`: the phone opens, pauses, resumes or stops World Builder or Object Memory.
**HOW.** Look up the `CartridgeSession`, call `session.apply(action)` (`cartridge_session.py:239`, logs the press and dispatches). Reply is the envelope with `accepted: true`; a
`SessionRefused`→HTTP 409 with the reason and a snapshot.
**WHY.** This state is the supervisor's gate: with no World Builder session open no builder attaches, so a CV Lab camera session never gets an all-cores-but-two solver.
**Called by** the phone (K12; Object Memory: K28, K29). **Calls** `CartridgeSession.apply`, which calls `start` or T4.1.
#### T2.4 `CaptureWorkerSupervisor.capture_opened` · `tower/capture_workers.py:456`
**WHAT.** When a recording starts, attaches a child process for every worker spec whose gate is open.
**HOW.** Under the supervisor lock: `reap()`, then per open gate `_attach_to_registry(FROM_START)` (L899) fills the capture directory into the argv and spawns. If `continues` names a capture
whose worker is alive and not asked to stop, the new capture is chained into that worker's lineage instead.
**WHY.** Lineage chaining is what makes a reconnect one walk. `main.py`'s subclass `_SupervisorThatYieldsTheGpu` (L1051) first stops the idle finisher.
**Called by** `ws._offer_capture_opened` (L957). **Calls** `_attach_to_registry`.
#### T2.5 `_world_build_spec` · `tower/main.py:183`
**WHAT.** Describes the World Builder child as a command line.
**HOW.** None when there is no world root or autobuild is off. Else a `WorkerSpec` running `world_build_session.py --follow-capture {capture_dir} --root … --rebuild-every N`. Optional:
`--register`, `--solve`, `--densify` and `--surface` (both need solve), `--appearance` (needs surface). Always: `--max-idle-polls`, `--stop-on-stdin-close`, `stop_via_stdin=True`,
`stop_grace_seconds=30.0`.
**WHY.** The web process knows the builder only as an argv, never an import (`test_shared_code_does_not_import_a_cartridge`), so the heavy work stays out of it.
**Called by** `_build_capture_worker_supervisor` (L1089). **Calls** `WorkerSpec`.
#### T2.6 `LiveSession.offer_frame` · `tower/live_session.py:530`
**WHAT.** The non-blocking hand-off of a frame to an in-process cartridge; the base class of Scene and Document Memory.
**HOW.** Under the condition lock: (1) count the frame; (2) not RUNNING→count dropped, return; (3) a frame already waiting→count a skip; (4) replace the single slot with `(raw_bytes,
received_at, source_seq)`, notify. The worker thread (`_loop`, L926) takes the slot, runs the engine, publishes.
**WHY.** Newest wins: a busy worker loses the older waiting frame, never the new one. It runs on the event loop, so it never blocks and never raises.
**Called by** `ws._offer_to_cartridges` (L349). **Calls** T8.6 and T8.3, on the worker side.

### T3. World Builder during the walk
#### T3.1 `follow_capture` · `scripts/world_build_session.py:300`
**WHAT.** Yields frames from the capture directory as the Tower writes them, across reconnects.
**HOW.** A missing directory raises `SystemExit` at once. Else it returns a generator over `CaptureFollower.follow` (`capture.py:971`), which polls `frames.jsonl` and loads each new frame;
on close it drains, then `_await_successor` retargets onto any successor capture. It ends with no successor, at the idle bound, or when `should_stop()` (asked once per poll) says so.
**WHY.** Deliberately not itself a generator, so the directory check runs before `main` creates a world: a bad path cannot leave an empty world behind.
**Called by** T4.3. **Calls** `CaptureFollower.follow`.
#### T3.2 `WorldBuilderEngine.observe` · `tower/world_builder/engine.py:361`
**WHAT.** The per-frame entry point: reject, tracking lost, or accept-and-persist.
**HOW.** (1) Renumber a `source_seq` that went backwards. (2) Decode; bad bytes→`malformed_frame`, size change→`frame_size_changed`. (3) `analyse_frame` + `tracker.measure`, then T3.3. (4)
Lost: reset the tracker, open a new segment, close the live solve's segment, journal `tracking_lost`. (5) Accepted: `_persist_keyframe` (L1597: T3.4, the image, then T6.2), then T3.6, then
extend the live solve on the redacted bytes.
**WHY.** The live solve gets redacted bytes because `build()` decodes exactly those. A failure in the live solve, relocalizer or frame log never costs a frame. Renumbering stops a restarted
sender overwriting keyframes.
**Called by** T4.3 (`world_build_session.py:2538`). **Calls** T3.3, T3.4, T3.5, T3.6.
#### T3.3 `KeyframeSelector.evaluate` · `tower/world_builder/keyframes.py:377`
**WHAT.** The keyframe policy as a pure function: ACCEPT, REJECT, SKIP or TRACKING_LOST for one frame.
**HOW.** In order: (1) blurred→reject, or accept as `turn_bridge` mid-turn; (2) no keyframe yet→`session_seed`; (3) flow survival under the loss threshold→`tracking_held`, and after
`loss_grace_frames` in a row `tracking_lost`; (4) degraded→reject; no motion evidence or too soon→skip; overlap under the floor→accept; (5) else accept as `parallax` once displacement
(fraction of the image diagonal) reaches the target, and skip below it.
**WHY.** The grace window stops one bad flow reading opening a segment. The turn bridge keeps keyframes through fast head turns. Thresholds live in `KeyframePolicy`.
**Called by** T3.2. **Calls** nothing; it reads `FrameQuality` and `MotionSummary`.
#### T3.4 `FaceRedactor.redact` · `tower/world_builder/redaction.py:529`
**WHAT.** Fills detected faces in a keyframe JPEG before it is stored.
**HOW.** Unavailable→original bytes labelled `none`. Else `_redact`: decode, detect with OpenCV `FaceDetectorYN` (YuNet), fill each box, re-encode; returns bytes, label, region count. Any
exception→original bytes labelled `none`.
**WHY.** It never raises, because a throwing redactor would lose the keyframe. The caller makes `none` sticky for the session, so one clean frame cannot launder an unredacted one; later
stages re-redact such sessions.
**Called by** `_persist_keyframe` (`engine.py:1597`). **Calls** `_redact`, which calls `_detect`.
#### T3.5 `LookBackRelocalizer.note_lost` · `tower/world_builder/relocalizer.py:1270`
**WHAT.** On a tracking loss, opens a recovery episode matching new frames against pre-loss keyframes. **Off by default:** with `TOWER_WORLD_RELOCALIZER=off` the engine builds none
(`engine.py:1339-1345`); walk 10 ran without it.
**HOW.** `machine.lost(at, can_accept=…)`; if an episode opens, snapshot the reference keyframes (recent window, plus history if configured) and return journal events. Later `note_frame`
calls submit frames to a worker at most `scan_hz` times a second; the state machine turns results into an anchor, a prompt or a timeout.
**WHY.** All matching runs on a worker thread, so the frame path never waits.
**Called by** T3.2, via `_recovery`. **Calls** `RecoveryStateMachine.lost`.
#### T3.6 `LiveDepthWorker.submit` · `tower/world_builder/live_depth.py:165`
**WHAT.** Queues an accepted keyframe for depth prediction (`moge2-vitl`) without blocking the frame path. Always on.
**HOW.** Disabled or backed off→counted `skipped` (back-off ends after 3 in-budget calls). Else `put_nowait` into a 64-slot queue, dropping the oldest job when full. The call times itself
into a p95 window; p95 over 15.0 ms, or free VRAM under 10%→back off and drain the queue.
**WHY.** The budget is this call's cost on the keyframe path, not the prediction's. Draining frees the GPU for the solve and the viewer. Output (`dense/<sid>/live_depth/`) becomes the scale
guard's evidence and is reused by T5.3.
**Called by** T3.2. **Calls** its own thread, `_run`.
#### T3.7 `BackgroundSolver.maybe_launch` · `scripts/world_build_session.py:943`
**WHAT.** Spawns a background global solve every `every` accepted keyframes (50 in the Tower-spawned builder).
**HOW.** Reap a finished child; skip if one is running, fewer than `every` keyframes are new, or fewer than 2 exist. Else write `sources.json`, open `solve.log`, spawn `world_solve.py` (not
`--final`) with stdin closed. With `TOWER_WORLD_FINAL_IMMEDIATE=on` it also uses a job object and stops launching once the capture closes.
**WHY.** The class owns every child, so the builder never leaves an orphan solve writing a solution nothing merges; `close()`, called in the builder's `finally`, leaves nothing behind.
**Called by** T4.3's observe loop. **Calls** `global_solve.write_sources`; it spawns `world_solve.py`.
#### T3.8 `compute_coverage` · `tower/world_builder/guidance_coverage.py:400`
**WHAT.** A pure computation of the status `guidance.coverage` block (which stations the solve has seen); drives the capture dial (K22).
**HOW.** Verify the solve identity chain (solve clocks; the journal against the manifest's and solution's input digests). A keyframe is eligible only if posed with at least 30 observations
in a reliably placed component. Per component: an 8x8 station grid with 12 sectors. Cap: at most 16 stations, block at most 4,096 bytes.
**WHY.** Painted coverage must be real solve evidence. A transient mismatch raises `_TransientCoverageError` and is retried. It runs in `CoverageWorker` under a 250 ms budget, never in the
status poll.
**Called by** `compute_from_tree` (L676, call at L718), from `CoverageWorker` (L730, call at L993). **Calls** `_component`.

### T4. Stop and the final solve
#### T4.1 `CartridgeSession.stop` · `tower/cartridge_session.py:371`
**WHAT.** The phone's Stop. Never refused, from any state.
**HOW.** Under the lock: `_detach()`, state `stopped`, clear the session id. For World Builder (`stop_policy=request`), `_detach` calls `supervisor.request_stop(worker)`
(`capture_workers.py:587`), which only closes the builder's stdin; every other cartridge gets `supervisor.detach(worker, grace)`.
**WHY.** Stop must not kill the builder, because the walk's geometry is worth finishing. The worker stays registered until it exits, so `/health` stays truthful.
**Called by** `apply` / T2.3. **Calls** `request_stop`.
**Phone side:** K12 (`workspaceDidDisappear` sends `stop`), K28 (Object Memory).
#### T4.2 `_stop_capture` · `tower/routes/ws.py:977`
**WHAT.** On `stream_stop`, closes this connection's recordings and tells the supervisor.
**HOW.** `_stop_recording` (L1013) closes the recorders this connection owns and returns the ids that actually closed; then per id `asyncio.to_thread(supervisor.capture_closed, id)`
(`capture_workers.py:640`).
**WHY.** `capture_closed` takes the supervisor lock, which a concurrent `detach` holds across its grace period; on the loop this measured 33.72 s of dead event loop. The closed capture is
what ends the builder's follow loop (T3.1).
**Called by** T1.1. **Calls** `CaptureRecorder.stop` (`capture.py:392`), `capture_closed`.
#### T4.3 `main` (the builder process) · `scripts/world_build_session.py:2023`
**WHAT.** The builder process's whole life.
**HOW.** (Call sites in `main`.) Prewarm; `follow_capture`, `WorldStore`, engine, `create_world`, `start_session` (L2410). Observe loop (L2538): T3.2, interim build every `--rebuild-every`,
T3.7, rebuild when a background solve lands. Then T4.4 (`hold_lock`), T4.5, final T4.9, `mark_finalization` (`complete`/`interrupted`, in `finally`), `release_world` (L2998). Outside the
lock: `register_session` (L3056), T5.1 (L3103), T5.5 if `--densify`.
**WHY.** The lock is released before the photographic stages so the phone can open the world during the six to sixteen minutes they take (`main.py:_world_finish_spec` docstring). A hard stop
is a separate channel from the soft stdin close, so a normal Stop never kills a finalization.
**Called by** the supervisor, via T2.5. **Calls** T3.2, T4.4, T4.5, T4.9, T5.1.
#### T4.4 `WorldBuilderEngine.stop_session` · `tower/world_builder/engine.py:657`
**WHAT.** Closes the session record, optionally keeping the writer lock.
**HOW.** Close the frame-quality log and the live depth worker. `hold_lock=True`→write the session with `finalization: pending` and keep the lock; else `release_writer_lock`. Journal
`session_stopped`, keep the live solve for `build()`, return a `SessionSummary`.
**WHY.** A lock held by a live pid plus `pending` is the only on-disk fact that says "someone is still finishing this world"; a dead pid plus `pending` can be told apart from a finish never
asked for.
**Called by** T4.3. **Calls** `WorldStore.write_session`, and `release_writer_lock` only when not holding the lock.
#### T4.5 `BackgroundSolver.run_final` · `scripts/world_build_session.py:1149`
**WHAT.** Runs the finalization solve as a child until it ends or a hard stop arrives.
**HOW.** Write `sources.json`, spawn `world_solve.py --final` with stdout piped, drain stdout on a thread, poll `should_stop` every `CHILD_POLL_S`. On a stop, terminate the child and return
`interrupted`; else parse the JSON summary (`solved`, `withheld`, `components`, `timing`, or an error).
**WHY.** It never raises: a walk that reconstructed locally is worth keeping even if the global solve fails. A consensus solve that published draw 0 and then died reports `solved`, from what
it published.
**Called by** T4.3. **Calls** `world_solve.py:main` (L45), which calls T4.6.
#### T4.6 `solve` · `tower/world_builder/global_solve.py:1189`
**WHAT.** Reconstructs a session's keyframes into one global solution (pycolmap/GLOMAP) and publishes it.
**HOW.** (1) Exit early without pycolmap or with fewer than 2 keyframes. (2) `prepare_images` (L532) undistorts each keyframe once. (3) Extract SIFT into the walk database, match
sequentially (loop detection when wanted; matching frozen only for a seeded final solve). (4) `_map_candidate` runs GLOMAP, falling back to the incremental mapper if it yields nothing. (5)
Record timing, publish via T4.7 (L1683).
**WHY.** The model is rebuilt from scratch every call: GLOMAP is fast enough (43 s for 438 keyframes, per the docstring) and a from-scratch solve cannot inherit a wrong decision; the
database skips features and pairs already done. Unseeded with `threads=-1` by default, hence not reproducible (§9.5).
**Called by** `world_solve.py:main`, `world_finalize.py`. **Calls** `prepare_images`, `_map_candidate`, T4.7.
#### T4.7 `gate_and_publish` · `tower/world_builder/coherence_publish.py:1402`
**WHAT.** The publish seam: gate the candidate (when on), scale-guard it (final solves only), write it, then write its records.
**HOW.** (1) Gate on (off by default)→`gate_final_solution`, or `gate_by_consensus` for 2+ draws, then `anchor_verified`. (2) `_final_scale_guard` (T4.8), then `_records_before_publish`; a
withhold returns `(None, record)` with `publish.written: False`. (3) Else `write(workspace, solution)`, then `after_publish` (components record, depth hand-off).
**WHY.** The guard sits after the gate and before the write, so it judges exactly what is about to be published. Its records go first, and a failure writing either becomes a withhold, so a
half-recorded publication never ships.
**Called by** T4.6. **Calls** T4.8, `write_solution`, `after_publish`.
#### T4.8 `guard_publication` · `tower/world_builder/final_scale_guard.py:927`
**WHAT.** Checks the room's cameras against metric-scale evidence at the publish seam; returns publish or withhold. Always on.
**HOW.** (1) `read_evidence`: the gate's metric log, else the live depth cache (walk 10: `live-depth-cache`). (2) `room_cameras`, then `assess_v2` (L588). (3) `apply` relabels isolated
plateaus as unplaced components and builds the components record. (4) `summary` and `audit_document` return as a `GuardOutcome`.
**WHY.** It never withholds on scale: `assess_v2` always says `publish`, and with no evidence the room is published `uncertified`. Otherwise it fails closed (an exception, or a relabel
leaving no room, means withhold). It predicts no depth.
**Called by** `coherence_publish._final_scale_guard` (L1500), final solves only. **Calls** `assess_v2`, `apply`.
#### T4.9 `WorldBuilderEngine.build` · `tower/world_builder/engine.py:891`
**WHAT.** Rewrites a session's derived tree from its persisted journal.
**HOW.** (1) Refuse a purged world or resolution mismatch, pick the backend. (2) Per tracker segment: flush the live estimate if its keyframe ids match the journal exactly, else re-decode
and `estimate_window`. (3) If a global solution exists, `global_solve.merge` (L2914) replaces rows, counts and placements. (4) Set `scale_state`, write world, tree and manifest (T6.3, at
L1217).
**WHY.** `observe()` can never reach it, so the multi-second write stays off the frame path. `diagnostics.placements_source` says who owns `placements.json`.
**Called by** T4.3 (interim and final), `world_finalize.py`, the finisher. **Calls** `merge`, T6.3.
#### T4.10 `regate_published` · `tower/world_builder/coherence_publish.py:1684`
**WHAT.** Re-runs depth, metric scale and the gate in place on an already published solve, without redoing the solve.
**HOW.** `_regate_inputs`, rebuild the candidate, re-run the gate (plus any owed consensus), then T4.8; publish the relabelled solution and components, or retire the record if the gate fails
again. Raises `RegateRefused` before starting work it cannot finish.
**WHY.** A gated final solve can publish before its gate completes (draw 0 goes out with the consensus `deferred`); the finisher pays that debt when the Tower is idle, under the writer lock.
With the gate off (the default) nothing is owed, so this is dormant.
**Called by** `world_finish_pending.finish_regate` (L1628). **Calls** `gate_final_solution`, `gate_by_consensus`.

### T5. The photographic ladder and the floor plan
#### T5.1 `final_surface_stages` · `scripts/world_build_session.py:1740`
**WHAT.** The finished world's surface stages, run after Stop in the builder process.
**HOW.** In order: (1) `surfacify(force=True)` with the final `SurfaceParams()`; (2) if `ok`, `build_appearance`; (3) `prune_intermediates` on the per-frame depth work, unless dense will
prune it. Each stage is marked `running`, then terminal, even when it raises; a hard stop skips a stage rather than truncating it.
**WHY.** Appearance reads the depth work's fill masks and raw predictions, so pruning goes last. A single function, so the finisher and `world_refinish.py` share the exact order.
**Called by** T4.3, `world_finish_pending.finish` (call at L2160), `world_refinish.refinish`. **Calls** T5.2, T5.4.
#### T5.2 `surfacify` · `tower/world_builder/surface_pipeline.py:707`
**WHAT.** Builds the surface of a solved session under `surface/<sid>/`.
**HOW.** Take the build lock (held→`unavailable`) and load the solution. Unless `force`, return the previous result when the params digest matches. Else run T5.3 → transient masks (hands, arms, held phone; fp16) → optional consistency → TSDF fusion (`SurfaceVolume.integrate`, L1339) → marching cubes → LOD pack → manifest → prune superseded levels → `ok`.
**WHY.** Idempotent and stop-aware. Every outcome except losing the lock writes `status.json`, so a cold reader never sees a stale `running`. States: `ok`, `stopped`, `unavailable`,
`failed`.
**Called by** T5.1, `scripts/world_surface.py`. **Calls** T5.3, `_build`.
#### T5.3 `ensure_depth_stage` · `tower/world_builder/surface_pipeline.py:245`
**WHAT.** Returns the depth stage's alignment record and work directory, running the stage first if it is missing or came from another solve.
**HOW.** A stored `align.json` that passes `_depth_cache_usable` (digest, params, image set, redaction trust) is returned. Else reuse the predictions on disk (the old record's and T3.6's),
refit them to the sparse points, `run_depth_stage`; stamp `input_digest`, `digest` and `cache_key`. A stopped stage stays unstamped.
**WHY.** Depth is the expensive part (68 s of GPU on this corpus, per the docstring); sharing `dense/<sid>/` means surface and dense pay once. An unstamped record is never taken as complete.
**Called by** T5.2, `area_build.py`. **Calls** `run_depth_stage`.
#### T5.4 `build_appearance` · `tower/world_builder/appearance_pipeline.py:217`
**WHAT.** Builds the photographic layer on a solved, surfaced session: the viewer's top rung.
**HOW.** (1) Take the lock; `_build` refuses without a solution, intrinsics or surface. (2) Read the redacted keyframes via `appearance.keyframe_source`, cast the surface's phone level (the
proxy) into each. (3) Compute redaction consensus, exposure gains (`solve_gains`) and transient masks; select by coverage (walk 10: 128 for the phone). (4) Encode content-addressed chunks:
phone tier WebP plus ASTC when available, Tower tier WebP only. (5) Publish the manifest.
**WHY.** The imagery is the wearer's own redacted keyframes, projected onto the proxy and view-blended on the phone. The single exception is the off-by-default `raw-local-research` source,
which labels itself not privacy-safe.
**Called by** T5.1, `scripts/world_appearance.py`. **Calls** `keyframe_source`, `solve_gains`, `encode_astc`/`encode_webp`, `pack_chunk`.
#### T5.5 `densify` · `tower/world_builder/dense_pipeline.py:1809`
**WHAT.** A dense point cloud anchored to the global solve. Off by default; walk 10 shows `unavailable`.
**HOW.** Take the dense and surface locks, then three checkpointed stages under `dense/<sid>/`: (1) depth, shared with T5.3; (2) fuse: mask, cross-check against neighbour cameras, average
agreeing points, voxel-reduce; (3) pack an LOD ladder.
**WHY.** Idempotent: a finished artifact from the same solve, parameters and image set is returned unless `force`; depth and fuse each keep a cache reused while its key
matches, and pack always runs. Stop-aware: a stop is noticed per frame inside depth and fuse and recorded as `stopped`; pack, once reached, runs to the end. Scale comes from the
sparse solve. Nothing is written to `derived/`.
**Called by** T4.3 (with `--densify`), `scripts/world_densify.py`. **Calls** `run_depth_stage`, `run_fuse_stage`.
#### T5.6 `build_landing` · `tower/world_builder/photo_floorplan_worker.py:497`
**WHAT.** Builds one photo floor-plan "landing" for a newly published coverage block: one tile per room component.
**HOW.** Read that solve's `solution.npz` (refused if `solution.json` changed underneath). For each of up to `MAX_COMPONENTS` = 16 components: place the eligible keyframes, warp their photos
onto the floor plane (`homography_component`, the pinned method), encode the tile and address it by SHA-256. A tile over 180 KiB is omitted; a manifest over 8 KiB is refused.
**WHY.** It runs in the Tower, not the builder: a `PhotoFloorplanWorker` daemon thread owned by `CoverageWorker`, with an 8 s wall cap (a 1.5 s cap skipped 3 of 4 landings under load); a
newer landing cancels it. The only world imagery the Tower itself builds.
**Called by** `PhotoFloorplanWorker` (default `build`). **Calls** `homography_component`, `encode_tile`.
**Phone side:** `WorldPhotoFloorplan.swift` (§6.4).

### T6. Storage
#### T6.1 `WorldStore.acquire_writer_lock` · `tower/world_builder/store.py:980`
**WHAT.** Takes the per-world single-writer lock (`worlds/<wid>/LOCK`).
**HOW.** `os.open(O_CREAT|O_EXCL)`, write `{pid, created_at}` (the psutil process start time) with fsync. If it exists: a live holder that is another process→`WorldLockedError`; a dead
holder (pid and start time no longer match)→reclaimed; an unreadable lock→reclaimed after a grace period. Retries are bounded, then it raises.
**WHY.** Liveness is pid plus start time, not a timeout: nothing has to guess how long a writer may pause, and the start time defends against Windows reusing pids.
**Called by** the engine at session start, `world_finish_pending.py`, `world_finalize.py`. **Calls** `lock_holder`, `_lock_record` (L1616).
#### T6.2 `WorldStore.append_keyframe` · `tower/world_builder/store.py:471`
**WHAT.** Appends one keyframe record to `keyframes.jsonl`, the authoritative input to every rebuild.
**HOW.** Under the store lock: `append_jsonl(keyframes_path, keyframe.to_json_dict())`; `append_jsonl` (`storage.py:300`) first adds a newline if the file's last byte is not one.
**WHY.** Healing a torn line confines a crash to the one interrupted record. The image is always written before its line.
**Called by** `_persist_keyframe`. **Calls** `append_jsonl`.
#### T6.3 `WorldStore.write_derived` · `tower/world_builder/store.py:634`
**WHAT.** Writes a session's rebuildable outputs.
**HOW.** Under the store lock: `write_json_atomic` for `poses.json`, `points.json`, the optional `support.json`, the session manifest and the world manifest.
**WHY.** `support.json` is separate because the geometry contract pins `points.json`. JSON rather than `.npy` because at this scale JSON round-trips exactly and stays inspectable.
**Called by** T4.9 (L1217). **Calls** T6.5.
#### T6.4 `WorldStore.read_derived` · `tower/world_builder/store.py:698`
**WHAT.** Reads the derived reconstruction, refusing stale output.
**HOW.** Read the files; with `verify=True` (the default) compare the manifest's input digest with `compute_input_digest(keyframes)` (L1578: SHA-256 over the keyframe ids, in order). A
mismatch returns None.
**WHY.** Serving a tree that no longer matches its journal would be silent corruption. Stale and absent both mean "rebuild", so None is honest for both.
**Called by** the status producer (`results/world_builder.py`, T7.4), `world_registration.py`, `inspect.py`. The geometry reader (T7.5), the sparse page (`render.py`) and guidance
imagery call it with `verify=False` and flag currency instead. **Calls** `compute_input_digest`.
#### T6.5 `write_json_atomic` · `tower/storage.py:139`
**WHAT.** The atomic-write primitive under every store.
**HOW.** Create parent directories, write a staging file, flush and fsync, then `replace_with_retry(temp, path)`; the staging file is always unlinked in `finally`.
**WHY.** On Windows `replace()` fails onto a file a reader has open, and directory fsync fails too (`store.py` docstring); so no checkpoint directories, and the replace retries while a
reader holds the file.
**Called by** T6.3, `CaptureRecorder` (its manifest), and most stores. **Calls** `replace_with_retry`.

### T7. Serving results to the phone
#### T7.1 `_subscribe` · `tower/routes/results_ws.py:153`
**WHAT.** Validates a `result_subscribe` against the registry and registers it. The dispatcher `handle` (L107) routes `cartridges`, `result_subscribe` and `result_unsubscribe`.
**HOW.** Type-check `cartridge`, `result_type` and the ids, then `registry.find_offer`. Refuse with `unknown_cartridge`, `unknown_result_type`, `contract_mismatch`, `cartridge_unavailable`
or `too_many_subscriptions`; else take the first snapshot and register a `Subscription` on the connection's channel.
**WHY.** Every refusal comes from a named, closed set, so the phone can tell "can't" from "not now". Echoed ids are clipped to 120 characters: a 1,000,000-character `cartridge` once produced
a 2,000,311-character reply, under the lock the frame path shares.
**Called by** `handle`, from T1.1. **Calls** `find_offer`, the hub.
**Phone side:** K13 (World Builder), K24 (Scene); all go through `TowerClient.subscribeToResults` (`TowerClient.swift:1273`).
#### T7.2 `ResultHub.poll_once` · `tower/results/publisher.py:1090`
**WHAT.** One pass over every watched target.
**HOW.** Under `_pass_lock`, collect the distinct targets and build each snapshot on a worker thread (at most one in flight per target). `ConnectionChannel.offer` (L335) hands it to each
subscriber whose revision changed or whose 2.0 s heartbeat is due. Each subscription has one slot, so a slow reader coalesces (`coalesced` > 0) instead of queueing.
**WHY.** Snapshots read files, so they run off the event loop: a disk stall costs channel latency, never the frame path. Passes are serialised, so a forced pass can never find a background
snapshot "in flight" and deliver nothing.
**Called by** the hub loop (L656), every 0.5 s. **Calls** `_dispatch`, then `snapshot_for` (T7.3).
#### T7.3 `make_snapshot_for` · `tower/results/__init__.py:54`
**WHAT.** Builds the callable mapping (cartridge, result type, world, session) to a `Snapshot`.
**HOW.** Build the World Builder status producer once, when there is a world root. The returned function routes CV Lab status, World Builder status, Scene `live` (from `SceneLive.latest()`,
`scene/live.py:162`, called at L152) and Document Memory `status`; anything else gets an explicit "no producer is wired" unavailable snapshot. Object Memory has no producer.
**WHY.** The hub imports no cartridge, so it can be tested with a stub. Scene, Document and the CV Lab are passed in as live objects because they report work happening in this process. A
`None` from `latest()` has four meanings (stopped, loading, failed, no frame yet); `status()` says which.
**Called by** `build_hub` (L229, call at L240), from T2.1. **Calls** T7.4, `latest()`, `DocumentStatusProducer`.
#### T7.4 `WorldBuilderStatusProducer.snapshot` · `tower/results/world_builder.py:575`
**WHAT.** One complete `world_builder.status/2026-09-10` payload from the world's files. Never partial, never a delta.
**HOW.** Resolve world and session (an explicit id is pinned, else a live world beats the newest saved one); read session, events, validated manifest, keyframe digest and lock holder through a per-file cache. `geometry_current` = digests
match; lifecycle from lock-holder liveness, the `stopped` event and `finalization`. Add progress, scale, geometry, artifacts, `guidance.coverage`; the revision hashes the payload minus
volatile fields. A read error becomes an `unavailable` snapshot with a client-safe reason.
**WHY.** The lifecycle is read from on-disk evidence, not the builder's claims. Coverage is dropped if the status would exceed its size budget.
**Called by** T7.3. **Calls** `_lifecycle`, `_payload` (L728).
**Phone side:** K14 turns each snapshot into the walk report; K17 and K18 turn that into words and a 3D offer.
#### T7.5 `build_manifest` · `tower/results/world_builder_geometry.py:511`
**WHAT.** The geometry manifest the phone polls: segments with content hashes and placements.
**HOW.** Read the derived tree *without* digest verification (`read_derived(verify=False)`, in `_read`) plus a `current` flag, and the usable placements. Only an absent tree
returns None (the route's 404); a tree behind the newest keyframes is served with `current: false`. Per segment emit `content_hash`, `placement_hash`, bounds, counts and coverage;
`geometry_revision` hashes both hash lists. `build_segment` (L607) serves one segment in its own frame, optionally down-sampled with `max_points`.
**WHY.** Content hashes let the phone fetch a closed segment exactly once. A segment travels in its own frame, and its placement says where it sits. Unverified on purpose: during a
walk the journal digest moves with every keyframe, so a verified read would answer 404 until the session ended; `current` tells the phone the geometry is behind instead.
**Called by** `routes/geometry.py:86` (and `build_segment` at L101). **Calls** `usable_placements`, `segment_content_hash`.
**Phone side:** K16 (`geometryDidChange`).
#### T7.6 `build_world_render` · `tower/results/world_builder_render.py:736`
**WHAT.** The viewer HTML for one session, at the best rung available.
**HOW.** The wanted rung is the pinned `representation`; unpinned, `appearance` if the client declares it can draw it (`viewer_draws_appearance`), else `surface`; the diagnostics view starts
at sparse. It tries appearance → surface → dense → sparse, stepping down past any rung unavailable or stale (built from an inactive keyframe set). A pinned rung that cannot be served raises
`WorldRenderUnavailable`.
**WHY.** A client that cannot draw the top rung gets a lower one, never nothing, and a dense bug cannot cost the sparse page. Native chrome means the phone draws the
controls in SwiftUI.
**Called by** `routes/geometry.py:285`. **Calls** `_appearance_page`, `build_surface_page`, `build_dense_page`.
**Phone side:** K19 (`load`), K20 (`followRevisions`), K21 (the web view).

### T8. The other cartridges
#### T8.1 `ObjectMemoryEngine.observe` · `tower/object_memory/engine.py:269`
**WHAT.** Per frame, decides which detections become recorded observations.
**HOW.** (1) Decode; close sightings stale at this timestamp. (2) Detect, ask the relevance policy (`decide`) about each; non-RECORD verdicts are dropped. (3) Make a `Look`, pass it to
`SightingTracker.observe` (`sightings.py:197`), which extends the class's open sighting if within the gap, else opens a new one. (4) `_consider`: write a mature sighting, get a second
opinion on an UNVERIFIED one, wait on a TOO_BRIEF one.
**WHY.** One record per sighting, not per frame: a laptop in view for a minute is one observation. Time comes from the capture, so a replay closes sightings exactly as the walk did.
**Called by** `scripts/object_memory_session.py:775`. **Calls** `SightingTracker.observe`, the detector, the policy.
#### T8.2 `build_last_seen` · `tower/results/object_memory.py:319`
**WHAT.** Answers "when was this category last in view?" (`GET /object-memory/last-seen/{class}`).
**HOW.** `store.last_seen(class)` wrapped in the retention and recorded-classes envelope; the payload carries `observed` (a bool), `recordable`, the observation view, and where the object
was.
**WHY.** No 404, because "no record of a laptop" read as Not Found would claim the laptop does not exist; instead `observed: false` and what absence means. It says `observed`, never
`present`.
**Called by** `routes/observations.py:153`. **Calls** `ObservationStore.last_seen`.
**Phone side:** `ObjectMemoryClient.swift` (§6.4).
#### T8.3 `DocumentMemoryEngine.observe` · `tower/document_memory/engine.py:267`
**WHAT.** The cheap per-frame path. Raw pixels never leave it.
**HOW.** Timestamp the frame, convert to gray. `PageFinder.find` (`gate.py:373`) runs `FrameGate.observe`; only stable frames proceed. The text detector is rate-limited and carries its last
region forward. The dwell tracker decides: a finished dwell is `_resolve`d as `document` or `resighted`; otherwise `dwelling` or `no_page`.
**WHY.** Only a dwell (the wearer holding a page in view) triggers the expensive OCR path; steadiness gates everything, so most frames cost almost nothing.
**Called by** `document_memory/live.py:292` (the live worker) and `scripts/document_memory_session.py`. **Calls** `PageFinder.find`, `_record`.
#### T8.4 `DocumentMemoryEngine._record_inner` · `tower/document_memory/engine.py:441`
**WHAT.** The expensive path: crop, OCR, dedup, identify and persist one finished dwell.
**HOW.** Per selected frame: crop the page, `_read` it (OCR; failure gives an empty page), perceptual-hash it, merge into a duplicate or add a `PageObservation`. No page read→dwell marked
unreadable. With re-sighting on, `_find_resighting` updates an existing document; otherwise a new one is persisted.
**WHY.** Every qualifying dwell is persisted, readable or not: "we looked and found no readable text" differs from "we never looked".
**Called by** `_record`. **Calls** `_read`, `_find_resighting`, the store.
#### T8.5 `DocumentMemory.search_text` · `tower/document_memory/retrieval.py:366`
**WHAT.** Lexical BM25 search over the stored OCR text, page by page (`GET /documents/search`).
**HOW.** Tokenise; refuse with a reason if there are no documents or no searchable terms. Expand terms, score every page with `_bm25`, keep pages at or above `min_score` (best page per
document), sort by exact-term count then score.
**WHY.** It refuses rather than guesses. A no-match reason is a statement about what was captured, not about what the documents say.
**Called by** `results/document_memory.py:665`, `scripts/document_query.py`. **Calls** `_bm25`, `tokenise`.
#### T8.6 `SceneEngine.observe` · `tower/scene/engine.py:154`
**WHAT.** One frame in, one `SceneState` out. Nothing is persisted.
**HOW.** (1) `_detect` (a failure counts as an empty frame). (2) `Tracker.update` (`tracking.py:199`) associates detections within each class, confirms a track at `min_hits`, drops it after
`max_misses`. (3) `_place` (left/right, with hysteresis) plus periodic `_estimate_orientation` (facing votes). (4) Drop partial people at the bottom edge, count by label, `relate`.
**WHY.** The tracker's boxes decide what exists and the facing estimator only describes them, so two models cannot disagree on a head count. Association is within a class, so a chair cannot
become a person.
**Called by** `scene/live.py:311`, `scripts/scene_session.py`. **Calls** `Tracker.update`, `relate`.
**Phone side:** K24, K25 (the phone watches by subscribing).
#### T8.7 `CVLab.process` · `tower/cv_lab/lab.py:508`
**WHAT.** Runs the armed experiment on one frame, or refuses legibly. Every `frame_result` comes from here.
**HOW.** Count the frame. Not `RUNNING`, or nothing armed→`FrameProcessingError` with a reason code for that state. Else `experiment.run(raw_bytes)`: success records result, provenance and
preview; a crash ends the run and refuses with "select an experiment".
**WHY.** Synchronous, no awaits, so an experiment switch cannot interleave inside a frame and `frame_provenance()` always describes the frame just processed. A crash ends the run, not the
module.
**Called by** `modules/experimental_cv.py:95`, via T2.2. **Calls** `experiment.run`.
#### T8.8 `CVLab.start` · `tower/cv_lab/lab.py:793`
**WHAT.** Arms a registered experiment on `cv_lab_start`, and returns at once.
**HOW.** Refuse if the Lab is unavailable, the id is malformed or unregistered, the `[ml]` extra is missing, or a start is in flight. Else create a run, drop the old experiment, enter
`STARTING`, then `create_task(self._arm(...))`, which loads under `ARM_TIMEOUT_S` (the container's `LOAD_TIMEOUT_S`).
**WHY.** A start that downloads weights must not stop the connection sending frames. The Lab holds one experiment and never queues a second.
**Called by** `cv_lab_ws.handle` (T8.9). **Calls** `_arm`.
#### T8.9 `handle` (CV Lab socket) · `tower/routes/cv_lab_ws.py:60`
**WHAT.** Dispatches `cv_lab_status`, `cv_lab_start`, `cv_lab_pause`, `cv_lab_resume` and `cv_lab_stop` to the Lab. Never raises, except to pass on a `WebSocketDisconnect`.
**HOW.** Find `app.state.cv_lab` (absent→`cv_lab_error`). Answer a status request directly; a command calls `lab.*`. Acceptance replies `cv_lab_status` plus `accepted_command`; refusal
replies a named `cv_lab_error`. Every reply carries the request id.
**WHY.** Commands share the frame socket, so the handler is a thin translator and the Lab owns every decision. The deliberate exception to "no commands on the result channel": CV Lab
commands are plain `/ws` messages, with no HTTP start.
**Called by** T1.1. **Calls** `CVLab.start`, `pause`, `resume`, `stop`.
**Phone side:** `ExperimentalCVClient.swift` (K26).

**iOS (K1-K29).** 29 entries, grouped by area, in the same WHAT / HOW / WHY format, then callers and callees.

### K1-K2. Shell

#### K1. `ProjectManager.init(...)`, `ProjectManager.swift:118`
**WHAT:** Builds the app's single runtime graph and wires the camera to the Tower.
**HOW:** Creates or injects `SenderMetrics`, `GlassesConnection(metrics:)`, `TowerClient(metrics:autoReconnect: true)`; `CartridgeClients`
gets four socket clients (World Builder, CV Lab, Document Memory, Scene Understanding) sharing that `TowerClient`; `CartridgeClients` builds Object Memory's HTTP
client itself; `ObjectMemoryRecordingCoordinator` gets the same `GlassesConnection`. In DEBUG, four Combine sinks:
`$latestCapturedFrame` -> `sendFrame`; `cameraStreamDidStart` -> `sendStreamStart` + `deviceHealth.refresh()`; `cameraStreamDidStop` ->
`sendStreamStop`; `$status == .online` while streaming -> `sendStreamStart`.
**WHY:** The two clients never reference each other; this is the only integration point. The fourth sink exists because `stream_start` does
not survive a socket drop; without it a Tailscale blip leaves the Tower pill green while every frame is dropped for lack of a bracket.
`autoReconnect` is off by default inside `TowerClient` so tests can assert a settled `.failed`.
**Called by:** `ContentView`'s `@StateObject` (ContentView.swift:49).
**Calls:** `GlassesConnection(metrics:)`, `TowerClient(metrics:autoReconnect:)`, `TowerWorldBuilderClient(tower:subscribeAckTimeout:)`.

#### K2. `ProjectManager.startAutomaticConnections()`, `ProjectManager.swift:283`
**WHAT:** Once per process, does the only two things the app may do without being asked.
**HOW:** An idempotence flag stops a second run; `glassesConnection.checkCameraPermission(reportErrors: false)` (a pure query);
`towerClient.connectIfIdle()` opens the socket; in DEBUG, applies the UI-test hooks.
**WHY:** Auto-connect is not auto-stream: an open socket transmits nothing until a live stream sends `stream_start`. `connectIfIdle`, not
`connect`, so automation never refills the reconnect budget or replaces a live connection. Errors are not reported (a first-launch modal
would blame the user for nothing). It never calls DAT registration (hands off to the Meta AI app) or requests permission (a context-free
launch prompt is how permissions get denied).
**Called by:** `ContentView` (ContentView.swift:190).
**Calls:** `checkCameraPermission`, `connectIfIdle`, `applyUITestHooks`.

### K3-K6. Glasses and capture

#### K3. `GlassesConnection.startCameraSession()`, `GlassesConnection.swift:1015`, then `beginCameraStream(on:)` at `:1265`
**WHAT:** The single entry point that starts glasses capture: creates a DAT session, and once DAT reports `.started` adds the camera and
starts its stream.
**HOW:** Clear stale error and refusal; refuse if a session exists (`.deviceHasPausedCapture` or `.alreadyRunning`) or no device is active
(`.noActiveDevice`); reset `frameCount`, viewfinder frame, `FrameRateGate` and metrics; `wearables.createSession(deviceSelector:)` +
`session.start()`. On `handleSessionState(.started)` (:1198), `beginCameraStream` checks camera permission, builds
`StreamConfiguration(.raw, captureResolution, 24 fps)`, calls `addCamera`, sets up listeners, calls `stream.start()`, and starts Motion only
if the probe or IMU recorder is on.
**WHY:** Each refusal is a value (`lastCaptureStartRefusal`) beside the sentence, so "already running" and "the glasses paused themselves"
can be told apart. Every failure after `session.start()` goes to `abandonSessionAfterFailedStart`, which tears down synchronously; on a
physical iPhone a session left set with no camera wedged capture (one refused start, then twelve taps that did nothing). A thrown
`dwaOutOfStuRange` keeps the session but arms a 15 s watchdog (DAT's start window is about 10 s).
**Called by:** WorldBuilderWorkspaceView.swift:694, HomeWorkspaceView.swift:233, CVCameraCard.swift:199, ObjectMemoryRecording.swift:605.
**Calls:** `createSession`, `armSessionStartWatchdog`, `setupStreamListeners`, `startMotion`.

#### K4. `GlassesConnection.setupStreamListeners(for:of:recorder:)`, `GlassesConnection.swift:1385`
**WHAT:** Registers DAT's stream listeners; the video listener numbers every frame, gates it to the Tower rate, decodes the frames it keeps,
and publishes them.
**HOW:** On DAT's thread: PTS probe and IMU stamp (clocks read before the hop). Then a main-actor `Task` (dropped if the frame belongs to a
replaced session): `frameCount += 1` gives `seq`; `frameRateGate.shouldSelect(at:)` (a rejected frame goes to `metrics.recordSkip()`); a
kept frame gets its pixel size from the sample buffer, `makeUIImage()`, then `latestCapturedFrame = CapturedFrame(...)`.
**WHY:** `seq` counts every delivered frame, so the Tower can compute the source rate from gaps. Decoding only kept frames makes cost follow
the send rate (15 fps), not the capture rate (24 fps). Logs are budgeted to about one line per second per kind.
**Called by:** `beginCameraStream`.
**Calls:** `FrameRateGate.shouldSelect`, `SenderMetrics`, `IMURecorder`; `TowerClient.sendFrame` downstream via ProjectManager.

#### K5. `FrameRateGate.shouldSelect(at:)`, `FrameRateGate.swift:126`, with `configuredTowerTargetFPS` at `:103`
**WHAT:** Decides whether a frame arriving at `now` is forwarded, holding a fixed delivered rate.
**HOW:** The first frame of a session is always admitted (deadline = `now + 1/fps`); a later frame is admitted once `now >= deadline - 0.25
x interval`; the next deadline advances from the previous deadline, not `now`; if more than an interval behind, re-anchor to `now +
interval`. Rate is read once per process: env `IOS_TOWER_TARGET_FPS`, else Info.plist `GlassesTowerTargetFPS`, else 15 (valid 6-24).
**WHY:** It targets a rate, not a stride (the old 1-in-30 stride delivered 0.8 fps from a 24 fps source). The deadline accumulates because
"elapsed since the last pick" aliases: at 24 to 12 fps, sub-millisecond jitter halved the rate. Re-anchoring after a stall avoids a burst of catch-up
frames; fresh frames come first. Measured: walk 10 captured 2,920 frames in 196.3 s, 14.9 fps. The gate, send-window capacity, IMU log header and Home text all read
this one value.
**Called by:** GlassesConnection.swift:1442.
**Calls:** nothing (pure value type).

#### K6. `GlassesConnection.startMotion(on:)` at `GlassesConnection.swift:1523` and `fallBackIfRateRefused` at `:1682` (DEBUG)
**WHAT:** Attaches DAT's experimental Motion (IMU) capability to a started session, for the developer probe and IMU recorder.
**HOW:** Runs only after the camera is up. Rates tried: [60 Hz, 30 Hz] with the recorder on, [30 Hz] for the probe alone; the first
successful `addMotion` wins and every attempt goes into the IMU log header. A runtime Motion error before the first sample at 60 Hz triggers
one retry at 30 Hz.
**WHY:** The camera is the product: Motion failures are logged and shown in the probe readout, never a modal, never end the session.
Developer switches are not persisted and read once at camera start. Status from the owner's notes, not the code: real glasses report
`sensorUnavailable`, IMU work is parked; these paths are exercised on Mock Device Kit.
**Called by:** `beginCameraStream`.
**Calls:** `attachMotion`, `detachMotion`, `IMURecorder.noteEvent`.

### K7-K11. Tower connection

#### K7. `TowerClient.openConnection(to:)` at `TowerClient.swift:573`, then `validateConnection(task:)` at `:1469`
**WHAT:** Opens the one WebSocket and calls it `.online` only after a ping/pong.
**HOW:** If the endpoint changed, drop the old Tower's cartridge declaration; record `reconnectURL` (stop if already `.connecting`); cancel
any pending reconnect, tear down the old socket, set `.connecting`, `webSocketTask.resume()`. `validateConnection` sends `{"type":"ping"}`
(this send is the TCP + upgrade leg), receives with a 6 s leg timeout requiring `pong`, sets `.online`, starts the receive loop, then sends
`{"type":"cartridges"}`. A handshake watchdog (2 x 6 = 12 s) ends a hung attempt.
**WHY:** `withTimeout` cannot bound a stuck `send` (a task group must wait for its stuck child; measured, a 1 s timeout was still
`.connecting` 12 s later), so the watchdog cancels the socket from outside. `cartridges` goes only after the pong: the Tower never speaks
first, and asking earlier would read the reply into the handshake. Leg timings are logged; one 9 s outage was about 8.5 s in `.connecting`.
**Called by:** `connect(to:)` at :536 (user action; refills the budget), `connectIfIdle` at :561, `scheduleReconnect`.
**Calls:** `teardownConnection`, `armHandshakeWatchdog`, `startReceiveLoop`, `requestCartridgeDeclaration`.
**Tower side:** T1.1 answers `ping` with `pong`; `cartridges` is answered by `results_ws.handle` (§6.3).

#### K8. `TowerClient.scheduleReconnect()`, `TowerClient.swift:700`
**WHAT:** After a failure, queues one delayed reconnect from a bounded schedule.
**HOW:** Needs `autoReconnect` on, a `reconnectURL`, and nothing queued. If the dying connection stayed online at least 30 s, the budget
refills. With 5 attempts spent, set `reconnectGaveUp = true` and stay `.failed`. Otherwise sleep 0.5, 1, 2, 4 or 8 s, then reopen the same
URL unless it changed or the user disconnected.
**WHY:** Reconnect is what makes stall recovery possible. The bound means a dead Tower ends at a visible `.failed`: about 16 s when TCP is
refused, up to about 75 s when the upgrade never completes. The budget refills on the way down, not on reaching `.online`, because one pong
proves little; a Tower that wedges right after it would otherwise flap forever.
**Called by:** `fail(_:)` (TowerClient.swift:1896), after teardown.
**Calls:** `openConnection`.

#### K9. `TowerClient.sendFrame(_:width:height:sequence:)`, `TowerClient.swift:762` (DEBUG)
**WHAT:** Sends one selected frame as one JSON text message in four named steps, recording exactly one terminal outcome per frame.
**HOW:** (1) `passSessionGates` (:776): online, inside the bracket, not paused, else a session-gate drop.
(2) `passSendWindow` (:811): if the main actor was responsive (previous frame at most 1 s ago) and a full window has had no completion for 2
    s, `replaceStalledConnection` replaces the socket; a full window otherwise means a send-window drop, decided before any encoding. (3)
    `encodeFrame` (:911): JPEG quality 0.5, base64, `{type:"frame", seq, tx_seq, width, height, format:"jpeg", data}`. (4) `transmit`
    (:963): reserve the slot, claim the `tx_seq`, `task.send`; `finishFrameSend` releases the slot and records success or failure.
**WHY:** The `SendWindow` is the real limiter: capacity 15 fps x 1/3 s = 5 slots covers the ~290 ms send latency measured on the device link
(capacity 2 admitted only 6.9 fps). The 2 s stall rule cut a recorded 52 s peer stall to about 4% of its cost; the 1 s main-actor gap check
stops a UI hitch reading as a dead socket. `tx_seq` is dense over frames actually sent, so a gap at the Tower means transit loss only (one
walk lost 2,391 of 4,801 source frames and `seq` alone could not say why). No suspension point between steps 2 and 4, so no other frame can
take the slot or number.
**Called by:** the ProjectManager sink (ProjectManager.swift:199).
**Calls:** the `SendWindow` methods, `SenderMetrics`, `fail(_:task:)`.
**Tower side:** T1.2 decodes and answers the frame; T1.3 fans it out to the recorder (T1.5) and the live cartridges (T2.6).

#### K10. `TowerClient.handleInboundMessage(_:)` at `TowerClient.swift:1584`, then `handleFrameResult(_:)` at `:1615`
**WHAT:** Routes every inbound text message by `type`; a frame result must pass a CV Lab staleness gate before it is published or counted.
**HOW:** `receiveLoop` (:1561) awaits `receive()` (non-text or undecodable messages are logged and dropped), then switches on the type:
`frame_result`, `frame_error`, `cv_lab_status`, `cv_lab_error`, `cartridges`, `result_subscribed`, `result_unsubscribed`,
`cartridge_result`, `result_error`, `protocol_error`. In `handleFrameResult`, a result from a run other than the watched one is discarded;
otherwise it is published (DEBUG), any refusal cleared, and `recordFrameResult()` counts it. Result-channel types go to `resultEvents`
(exposed as `cartridgeResults`).
**WHY:** One gate at the top, because "a gate per consumer is a gate somebody forgets". The watched run is learned only from status
documents, so a stale result never nominates itself. `run_id` is `<tower_instance_id>-<n>`, which also covers a restarted Tower.
**Called by:** `receiveLoop`.
**Calls:** `handleFrameResult`, `handleFrameError`, `handleCartridgeResult` and other handlers, `resultEvents.send`.
**Tower side:** T1.2 (`frame_result`, `frame_error`), T8.9 (`cv_lab_*` replies), T7.1-T7.2 (the result channel).

#### K11. `TowerClient.handleFrameError(_:)`, `TowerClient.swift:1677`
**WHAT:** Handles the Tower's per-frame refusal: a changed refusal is published at once, a repeat at most once per second, and every decoded
refusal goes on to CV Lab consumers.
**HOW:** Decode a `TowerFrameRefusal`; if the reason changed or 1 s has passed since the last publish, publish now, else hold it pending
with a trailing flush at window end; always `cvLabControlEvents.send(.frameRefused)`.
**WHY:** One `frame_error` per frame is in the contract whenever CV Lab is not running, so an ordinary walk gets them at the send rate;
republishing each would redraw and log at about 15 Hz. Refusals leave the send window and reply counters alone. A refusal is cleared at
stream start and stop, at teardown, on a run change, and after a valid result.
**Called by:** `handleInboundMessage`, and tests.
**Calls:** `publishFrameRefusal`, `schedulePendingFrameRefusal`.
**Tower side:** T1.2 and T2.2 pick the `frame_error` reason; T8.7 refuses whenever CV Lab is not running.

### K12-K22. World Builder workspace

#### K12. `WorldBuilderSessionController.workspaceDidAppear(isTowerReachable:)`, `WorldBuilderSessionController.swift:164`
**WHAT:** Tells the Tower the World Builder screen is open: `POST /cartridges/world_builder/session/start` on appear, `stop` on leave.
**HOW:** On appear, mark on screen; if the Tower is unreachable wait (`.waitingForTower`), else `start()`. An honoured start gives
`.active`; a refusal shows the Tower's message. Losing the socket gives `.towerLost`; a reconnect starts again. `workspaceDidDisappear`
calls `stop()`.
**WHY:** The Tower attaches a builder to a capture only while this session is active (before that, Object Memory sessions grew World Builder
worlds nobody asked for). Owned by the view, not the view model, so the "view models send nothing" test still means something. "active" is
intent, not liveness.
**Called by:** `WorldBuilderWorkspaceView` appear and disappear.
**Calls:** `begin(_:pending:)` -> `CartridgeSessionHTTPClient`.
**Tower side:** T2.3 (`apply_session_action`); `stop` is T4.1.

#### K13. `TowerWorldBuilderClient.subscribeIfPossible()`, `TowerWorldBuilderClient.swift:1199`
**WHAT:** Opens the World Builder status subscription when the socket, the Tower's declaration and the contract all allow it.
**HOW:** Needs `.online` with nothing open or in flight, and a declaration offering `world_builder` with a supported contract (an
unavailable offer becomes `.unsupported` with the Tower's sentence). Then: clear the last report, set `.awaitingFirstUpdate`, send
`result_subscribe` with an optional `world_id`/`session_id` pin, arm a 10 s ack timeout.
**WHY:** No delta stream, so a reconnect cannot lose data: the first message after a subscribe is a full snapshot. Subscription ids restart
at `sub-1` on every socket, so they are cleared when the socket leaves `.online`. `handle(_:)` (:1490) checks each ack against the current
pin; a stale or duplicate ack is unsubscribed and its id retired, so queued heartbeats for it are ignored.
**Called by:** `connectionChanged(to:)` (:1172), the `$cartridgeDeclaration` sink; `inspect`, `followLive`, `askAgain` via
`restartSubscription`.
**Calls:** `TowerClient.subscribeToResults` (TowerClient.swift:1273), `armSubscribeTimeout`.
**Tower side:** T7.1 (`_subscribe`).

#### K14. `TowerWorldBuilderClient.apply(_ envelope:)`, `TowerWorldBuilderClient.swift:1591`
**WHAT:** Turns one status envelope into the client's walk report, and returns the geometry address to fetch when the presented state holds
a world.
**HOW:** Refuse a non-snapshot envelope; decode `model_state` (failure gives `.failed(.undecodableResponse)`); build a `StatusReport`
(selection, session, finalization, photographic, processing, recovery, coverage from `WorldCoverageStatus(payload:)`, keyframes, Tower
clock); `publishLastReport()` applies the session gate to give the presented state; then the capture-health sample (live only), the
look-back prompt and the finish clock. Return `geometryCoordinates` only when the presented state carries a snapshot.
**WHY:** A delta read as a full snapshot fails silently and wrongly ("waiting for first update" over a fully populated world), so it is
refused loudly. The report is published as one value before the address is sent, so the screen learns whose walk it is before it fetches the
picture. An address from a state the gate turned into "waiting" would fetch another world's fragments.
**Called by:** `handle(_:)` (:1490), for a `.result` whose subscription id is not retired.
**Calls:** `WorldBuilderResultDecoder`, `WorldCoverageStatus(payload:)`, `publishLastReport`, `lookBack.consider`.
**Tower side:** T7.4 builds each snapshot; T7.2 delivers it.

#### K15. `WorldBuilderViewModel.walkReportDidChange(to:)`, `WorldBuilderClient.swift:714`
**WHAT:** The single entry for the client's walk reports: publishes walk and state as one value, reacts to a new walk, then a new state,
then forwards coverage to the imagery models.
**HOW:** Ignore an equal report; one assignment `walkReport = next`; `walkDidMove` forgets geometry belonging to another session;
`stateDidMove` handles a 30 s bound on the first update and forgets the gallery on idle, failed, unsupported or a different world;
`imageryCoverageMoved()`.
**WHY:** Publishing identity apart from state lets a reader pair one walk's picture with another walk's words. A new session of the same
world counts as a new walk. `.awaitingFirstUpdate` keeps the cache, so a reconnect redraws without refetching.
**Called by:** `adoptClientWalkReport` (:700), from the `client.walkReportUpdates` sink.
**Calls:** `walkDidMove`, `stateDidMove`, `imageryCoverageMoved`.

#### K16. `WorldBuilderViewModel.geometryDidChange(worldID:sessionID:revision:)`, `WorldBuilderClient.swift:1000`
**WHAT:** Fetches and publishes the geometry the Tower just named, in eight named steps, refusing anything a newer push has superseded.
**HOW:** Needs a complete address. (1) `recordNamedGeometry`: a new owner clears the gallery and sets the render target. (2)
`claimRevision`: skip if this revision is on screen or in flight; spinner only before the first answer. (3) `fetchManifest`, `GET
/worlds/{id}/geometry/manifest` (failure clears the marker so the next report retries; 404 means the Tower says there is none). (4)
`acceptsPoseConvention`. (5) `fetchSegments`: drop cache entries the manifest no longer names, then each segment from cache first, else `GET
.../geometry/segment/{i}`.
(6) `publish`. (7) `rearmRetryIfIncomplete`. (8) `logSegments`. `isStillOurs` is checked after each await and before each segment request.
**WHY:** Keyed on revision, not arrival: the channel heartbeats every ~2 s, so per-message fetching would pull about 1 MB of points twice a
second for a world standing still. `isStillOurs` checks address, revision and cancellation because the revision is a content hash with no
world or session in it (two empty sessions share one). The cache key includes the placement hash, so a registration pass gets refetched.
**Called by:** `fetchGeometry(at:)` (:898), from the `client.geometryUpdates` sink.
**Calls:** `WorldGeometryClient.manifest/segment`, the `WorldGeometryStore` actor.
**Tower side:** T7.5 (`build_manifest`, `build_segment`).

#### K17. `WorldStage.stage(for:evidence:finalization:photographic:)`, `WorldPresentation.swift:465`
**WHAT:** Names the world's stage in the screen's vocabulary (Mapping, Building, Improving, Finalizing, Saved, Partial, Interrupted, Needs
retry) from the Tower's evidence.
**HOW:** A switch on `WorldModelState`. Idle, awaiting, unsupported: `nil`. Receiving: Building if the Tower reports geometry, else Mapping.
Finalizing: no geometry gives Finalizing; an area still finishing judges the room as finalized; a build in progress or unfinished
photographic room gives Improving; else Finalizing. Finalized: final solve denies a finished world gives Partial with geometry, Needs retry
without; solve pending gives Finalizing; geometry gives Saved; a reported solve with no figures at all (older record) gives Saved; else
Needs retry. Interrupted: Interrupted with geometry, Needs retry without. Failed: Needs retry only when the Tower itself reported the
failure.
**WHY:** Every word is earned from something the Tower said. A missing report is never proof that a build finished, and zeros are not
silence.
**Called by:** `WorldBuilderViewModel.stage` (WorldBuilderClient.swift:1559), TowerWorldBuilderClient.swift:1140.
**Calls:** `WorldFinalSolve(word:)`.

#### K18. `WorldReconstruction.ladder(target:stage:finalSolve:evidence:photographic:)`, `WorldPresentation.swift:1015`
**WHAT:** Decides the screen's 3D offer through two ordered rule tables: `.final`, `.partial` with a note, or `.unavailable` with a reason.
**HOW:** With no target, `unavailableLadderRules` (first match wins): the Tower claims geometry gives "...but has not named where to fetch a
picture"; the stage still changing gives "nothing built yet"; else "built no 3D world". With a target, `addressedLadderRules` (:900) (first
non-nil wins): no stage yet gives partial; Saved gives final; Mapping or Building gives partial, "still being built"; Improving and
Finalizing give partial with the most specific note (owed or unobservable photographic work, photographic building, or "final pass has not
landed"); Partial, Interrupted, Needs retry give partial, each with its own sentence.
**WHY:** The product rule: final 3D world, then the best partial, then a truthful degraded state. Converted from a 160-line nested `if` and
proven output-identical on a 3,528-case characterization matrix. `stageIsUnfinished` is an exhaustive switch, so a new stage
breaks the build until its rules are decided. Numbers behind the Improving note: the final solve took 179 s after Stop on an earlier 4-minute walk
of 795 keyframes (16 components became 6; 652 of them joined); photographic work took 459 s for 385 keyframes and 756 s for 690.
**Called by:** `WorldBuilderViewModel.reconstruction` (WorldBuilderClient.swift:1567), WorldPickerView.swift:257.
**Calls:** `unfinishedPhotographicNote`, `stageIsUnfinished`.

#### K19. `WorldRenderViewerModel.load()` at `WorldRenderViewerModel.swift:1129`, with `pageEvent(_:)` at `:1218`
**WHAT:** Fetches the Tower's composed 3D page and starts a bounded render; a finished fetch never counts as drawn content.
**HOW:** Reset the watchdog and refusal bookkeeping, `.fetching`, `GET /worlds/{id}/render`. An appearance page with no session stamp is
refetched once, then fails as retryable; an area page must declare this area and session. Then `.rendering(html)` + a 20 s render watchdog.
WebKit reports through `pageEvent`: `.rendered` gives `.ready`; a process death or self-reload goes back to `.rendering`; a failure gives
`.failed`, or reverts an automatic refresh.
**WHY:** `.ready` comes only from WebKit's `didFinish`, because a page that arrived may still be too large to draw. A failure message says
the fetch succeeded, so the reader is not sent to the wrong machine.
**Called by:** the `WorldRenderScene` `.task` (WorldRenderViewer.swift:282), the Reload and Try again buttons, `WorldInlineHost`.
**Calls:** `WorldRenderClient.page(for:)`, `startRenderWatchdog`, `areaPageMismatch`.
**Tower side:** T7.6 (`build_world_render`).

#### K20. `WorldRenderViewerModel.followRevisions()`, `WorldRenderViewerModel.swift:752`
**WHAT:** While a picture is on screen, polls the render revision; a better rung replaces the page by itself, a same-rung rebuild is offered
as a button.
**HOW:** Product view only. Sleeps 10 s, doubling to a 120 s ceiling while the Tower answers `live:false` and nothing changes; acts only
when `.ready`, asking `GET .../render/revision`. An appearance page follows its own textures; if the appearance was withdrawn and the page
did not recover within one poll, the model replaces it. On a new revision: refused rungs are skipped (one retry when a build refused while
live is reported finished); on the same walk, an upgrade (sparse < dense < surface < appearance) or a finished non-downgrade calls
`refresh`; anything else sets `newerPictureAvailable`.
**WHY:** A swap reloads the page and resets the reader's camera, so only an improvement swaps by itself. The ceiling is not a stop, because
the final surface starts a few seconds after the Tower releases the world lock.
**Called by:** WorldRenderViewer.swift:284, WorldInlinePanel.swift:84.
**Calls:** `client.revision(for:)`, `refresh`, `swapsBySelf`, `appearanceFollow`, `nextPollInterval`.
**Tower side:** `GET …/render/revision` (§6.4); the rungs are built by T5.2 and T5.4 and served by T7.6.

#### K21. `WorldRenderWebView.Coordinator.webView(_:decidePolicyFor:decisionHandler:)`, `WorldRenderWebView.swift:485`
**WHAT:** The web view's navigation gate: allows only the initial load of its own page URL and a bounded self-reload, and resets the
native-chrome bridge for each allowed document.
**HOW:** The first main-frame `.other` navigation is the initial load; a later main-frame `.reload` or `.other` counts as a self-reload
while fewer than 3 in the last 60 s; `WorldRenderNavigationPolicy.allows` requires the exact page URL; an allowed navigation sends
`bridge.receive(.pageWillLoad)` (a self-reload also `onEvent(.reloadingItself)`); answer `.allow` or `.cancel`. Around it:
`makeConfiguration` (:246) uses a non-persistent data store and one script-message handler; the page comes from the private `glasses-world:`
scheme, where `WorldAssetSchemeHandler.startTask` (WorldAssetTransport.swift:739) serves the page string from memory and proxies only a
whitelist of this world's Tower routes (everything else 404).
**WHY:** The page's CSP is `connect-src glasses-world:`, so web content cannot reach the network directly. A second navigation would reset
the camera. The self-reload exception exists because WebKit often never returns a lost WebGL context; content-process kills reload at most
twice per 60 s (:594), then fail. The imagery (~13 MB of texture blocks) lives in memory only and is dropped when the viewer closes.
**Called by:** WebKit.
**Calls:** `WorldRenderNavigationPolicy.allows`, `WorldChromeBridge.receive`, `onEvent` (reaches `WorldRenderViewerModel.pageEvent`).
**Tower side:** the page is T7.6's; the proxied routes are in §6.4.

#### K22. `WorldCaptureDial.init(components:stationsOmitted:frameRevision:newest:motionSector:photos:)`, `WorldCaptureDial.swift:157`
**WHAT:** Builds the walking-stage capture dial: twelve 30° directions per reconstructed piece, from the Tower's frozen coverage receipt.
**HOW:** `WorldCoverageStatus(payload:)` (WorldCoverage.swift:50) reads `guidance.coverage`: no key gives `.absent`, `null` gives `.notYet`,
a malformed block gives `.refused`, else `.receipt`. `WorldInlinePanel.captureDial` (:779) returns `.awaiting` (12 dashed segments) with no
receipt. Each component becomes a `Piece`; each sector is the OR over the drawn stations: supported (2 or more views), weak (1 view), else
unconfirmed. The hero piece is the one the newest view names, else the one with the most keyframes; photos are kept only on its captured
sectors.
**WHY:** The dial never claims to be complete or says where the wearer is. The top is fixed at sector 0, the way the wearer faced when the
piece began; the newest view lags the head by a minute or two, so "missing at 8 o'clock" stays at 8 o'clock. Always on.
**Called by:** `WorldInlinePanel.captureDial` / `shownDial`.
**Calls:** `Piece.init`.
**Tower side:** T3.8 (`compute_coverage`), carried in the status by T7.4.

### K23-K29. Cartridges

#### K23. `CartridgeAvailability.resolve(declared:supported:isTowerReachable:knownToThisBuild:)`, `CartridgeAvailability.swift:208`
**WHAT:** Decides whether a cartridge can be used.
**HOW:** No declared contract: a cartridge this build does not know gives `.noContract`; a known one gives `.noContract` when the socket is
up, `.towerUnreachable` when down. A declared contract outside `TowerCapabilities.supported` gives `.unsupportedContract`. Otherwise
`.available` when reachable, else `.towerUnreachable`.
**WHY:** Contracts come only from the Tower's live declaration; the local table is empty by rule. Before any connection there is no
declaration, and saying "the Tower doesn't offer this" would be false. Object Memory is deliberately absent; it is HTTP only.
**Called by:** `TowerCapabilities.availability` (CartridgeClient.swift:155), through each client's `availability(isTowerReachable:)`.
**Calls:** nothing.
**Tower side:** the `cartridges` declaration (`results/registry.py`, §6.3).

#### K24. `TowerSceneUnderstandingClient.workspaceVisibilityChanged(isVisible:)` at `SceneUnderstandingClient.swift:305`, with `adoptOrClose(_:)` at `:382`
**WHAT:** Subscribes only while the Scene screen is visible, and keeps at most one subscription open.
**HOW:** Ignore a repeated value. Visible: `subscribeIfPossible`. Hidden: unsubscribe the open id, clear the reading, keep any in-flight
subscribe marked. On ack, `adoptOrClose` adopts it only if the screen is visible and nothing is open, else unsubscribes at once.
**WHY:** The Tower's people detector runs while somebody streams and somebody watches. An orphaned subscription (screen appears, leaves and
returns before the first ack) kept it running for the life of the socket: a real bug, fixed in the campaign (§9.3).
**Called by:** the view model (:536), from the view's onAppear, onDisappear and scenePhase handlers.
**Calls:** `subscribeToResults` / `unsubscribeFromResults`.
**Tower side:** T8.6 (`SceneEngine.observe`); `live` snapshots via T7.3.

#### K25. `SceneUnderstandingState.forReading(_:)`, `SceneUnderstandingModel.swift:808`
**WHAT:** The only constructor from a decoded reading to a screen state.
**HOW:** By lifecycle: failed gives `.failed` with the Tower's reason; starting gives `.awaitingFirstScene`; running gives `.observing` with
a scene, else awaiting; paused gives `.lastKnown` if a scene was kept, else `.idle`; stopped or unknown gives `.idle`.
**WHY:** "Stop discards, pause keeps" is written once, not in a switch at every call site. `SceneUnderstandingDecoder.reading(from:)`
(SceneUnderstandingContract.swift:121) decodes all-or-nothing beforehand, because a partial scene would render as a quieter room.
**Called by:** SceneUnderstandingClient.swift:437.
**Calls:** nothing.

#### K26. `TowerExperimentalCVClient.project(_:)` (static), `ExperimentalCVClient.swift:877`
**WHAT:** Maps a CV Lab status document to the workspace state.
**HOW:** By lifecycle: unavailable gives `.unsupported`; idle gives `.idle`; starting gives the named experiment, else idle; running or
paused needs a run, else idle; stopped gives `.completed(run)` (keeps final figures); failed gives the Tower's reason; any unknown word
gives `.failed(undecodable)`.
**WHY:** An unknown state is never collapsed to idle, which would invite Start against a Lab doing anything. The client also passes every
status's run id to `TowerClient.watchCVLabRun` (TowerClient.swift:1198), which arms the frame-result staleness gate (K10) and clears the
held reading on a run change.
**Called by:** ExperimentalCVClient.swift:857.
**Calls:** nothing.
**Tower side:** T8.7-T8.9.

#### K27. `TowerDocumentMemoryClient.handle(_:)` at `DocumentMemoryClient.swift:427`, with `DocumentMemoryViewModel.libraryChanged(_:)` at `:642`
**WHAT:** Applies Document Memory's status-channel events; when the library revision moves, the view model reruns its standing query over
HTTP.
**HOW:** An ack records the subscription; a snapshot is decoded and `session` and `library` are published; an error for this cartridge:
a closing error resubscribes (3 per connection) or fails; any other error clears `session` and leaves `library` alone. In the view
model the first revision seen is the baseline; a new revision with no search running resubmits the standing query (default: the 20 most
recent).
**WHY:** Two contracts on two transports: status is small and pushed; the library is bulk sensitive text, pulled. Heartbeats must not reload
document text. An error on the status channel says nothing about what is on disk.
**Called by:** the `tower.cartridgeResults` sink.
**Calls:** `DocumentMemoryDecoder.status`, `CartridgeResultSubscription.closedByTower`, `subscribeIfPossible`.
**Tower side:** T8.3-T8.5; `status` via T7.3.

#### K28. `ObjectMemoryRecordingCoordinator.startSequence()` at `ObjectMemoryRecording.swift:558`, with `stopSequence()` at `:655`
**WHAT:** Starts Object Memory: the Tower producer first, then the camera if nobody holds it, then a bounded check that a producer is
following the capture.
**HOW:** `client.apply(.start)` (a refusal stops the sequence). `startCameraIfNeeded`: unclaimed starts the camera and marks it as started
by this screen; paused or ending is a refusal; running on another screen is used without claiming. `converge` rereads every 3 s for at most
12 s. Stop: Tower `stop` first, then only a camera this screen started, then refresh the records.
**WHY:** The Tower goes first, so a new capture does not lose its opening frames. When the camera is refused, the Tower gate deliberately
stays open for the next capture from any screen. It never stops another screen's camera.
**Called by:** the coordinator's `start()` / `stop()` actions.
**Calls:** `TowerObjectMemoryClient.apply`, `GlassesConnection.startCameraSession`, `converge`.
**Tower side:** T2.3 (the session), T8.1 (the follower's per-frame engine).

#### K29. `TowerObjectMemoryClient.apply(_:)`, `ObjectMemoryClient.swift:607`
**WHAT:** Sends one session verb, then replaces its answer with a fresh read of liveness, keeping any refusal's explanation.
**HOW:** Single-flight guard, then `.working`; POST the verb and show the honoured or refused answer at once; GET the session again. An
honoured verb gives `.known(fresh)`; a refusal keeps its message and takes the fresh snapshot.
**WHY:** A POST answer is intent, not liveness: a Pause whose producer ignores SIGTERM answers 200 `paused` while still following, and
`stop` is never refused. A refusal is a state, not an error.
**Called by:** `ObjectMemoryRecordingCoordinator`.
**Calls:** the session control client's `apply` and `read`.
**Tower side:** T2.3, T4.1 (`stop` is never refused).

### iOS, also curated (one line each; each has a WHAT/HOW/WHY doc comment in the code)

| Symbol | Location | What |
|---|---|---|
| `ContentView.workspace` | ContentView.swift:256 | The one switch that decides which cartridge workspace is on screen (exhaustive, no `.id`). |
| `TowerClient.sendStreamStart()` / `sendStreamStop()` | TowerClient.swift:1065 / :1096 | Open and close the stream bracket, once each; this also clears the run watch and the held result. |
| `WorldSessionGate.binding(...)` | WorldSession.swift:151 | Decides whether a snapshot belongs to the capture this phone has open: `.none`, `.awaiting`, `.foreign` or `.bound`. |
| `WorldBuilderViewModel.loadWorlds()` | WorldBuilderClient.swift:1358 | Lists saved worlds through one shared load that cannot be cancelled and that every concurrent caller joins. |
| `WorldGeometryAccount.account(for:evidence:)` | WorldPresentation.swift:265 | The gallery's empty-state sentence: the Tower's geometry claim outranks the phone's failed fetch. |
| `WorldRenderScene.body` | WorldRenderViewer.swift:201 | One stable web-view slot between native or legacy chrome; one `.task` runs load, components, then the follower. |
| `WorldChromeSession.fallBack()` | WorldChromeProtocol.swift:847 | Refuses native chrome for this screen and asks the page to restore its own; the page is reloaded if it does not confirm within 2 s. |
| `TowerExperimentalCVClient.subscribeIfPossible()` | ExperimentalCVClient.swift:639 | Opens one CV Lab subscription when online and the offer is supported. |
| `DocumentMemoryDecoder.library(from:)` | DocumentMemoryDecoder.swift:35 | Decodes the HTTP library response without inventing an empty memory. |
| `ObjectMemoryDecoder.observation(from:)` | ObjectMemoryModel.swift:669 | Decodes a sighting only when its claim, identity and frame reference match the contract. |
| `CartridgeResultSubscription.closedByTower()` | CartridgeResultSubscription.swift:74 | Spends one of 3 resubscribes per connection, or returns the failure to show. |

---

## 6. Contracts

Shared protocol truth lives under `docs/contracts/` and `tower/docs/contracts/`. Every message and route below names its Tower
implementation and its iOS side.

### 6.1 Where they are written

| Doc | Owns |
|---|---|
| `tower/README.md` ("Stream Lifecycle Control Messages", `frame_error`) | The base socket protocol: `frame`, `ping`, `stream_start`/`stream_stop`, the three `frame_error` reasons |
| `docs/contracts/TOWER-UNIFIED-CARTRIDGES.md` | `GET /cartridges`, the socket-vs-HTTP rule (§3), the shared lifecycle, per-cartridge rules |
| `tower/docs/contracts/CARTRIDGE-RESULTS.md` | The result channel (subscribe, envelope §4, ordering, errors, limits); `world_builder.status` (§10), Scene `live` (§14), Document status (§15), sender assumptions (§19) |
| `tower/docs/contracts/EXPERIMENTAL-CV-LAB.md` | CV Lab control, status, `frame_result` provenance, preview |
| `docs/contracts/WORLD-BUILDER-GEOMETRY.md` (`world_builder.geometry/2026-08-25`) | Manifest and segments |
| `docs/contracts/WORLD-BUILDER-SURFACE.md`, `-APPEARANCE.md`, `-DENSE.md` | The three artifacts and the render ladder |
| `docs/contracts/WORLD-BUILDER-COMPONENTS.md` | Components record, gate, scale guard, areas, relocalizer (§6) |
| `docs/contracts/WORLD-BUILDER-WORLDS.md` (`world_builder.worlds/2026-09-10`) | The saved-worlds listing |
| `docs/contracts/WORLD-BUILDER-IOS.md` | The reconciled World Builder boundary |
| `tower/docs/contracts/WORLD-BUILDER-GUIDANCE-IMAGERY.md` | Guidance imagery (B1 `live`, B2 `landed`); photo floor plan v1.2 |
| `docs/contracts/OBJECT-MEMORY.md` (`object_memory.observations/2026-08-26`) | Observations, last-seen, imagery |
| `tower/docs/contracts/IOS-TO-TOWER-RECONCILIATION.md` | The 2026-08-23 reconciliation of the iOS requests |

### 6.2 The rule, the contract ids, the envelope

**The rule** (TOWER-UNIFIED-CARTRIDGES §3): one websocket at `/ws`; everything else is HTTP.
- Commands never travel on the result channel.
- Bulk data and imagery never travel on the socket, because the result sender shares its send lock with the frame path.
- Every timestamp is Tower-receipt time.

**Contract ids** are opaque dated strings (`results/contracts.py`). The phone compares them only for equality: a mismatch is shown as
"unsupported contract" and never decoded on a guess, and payloads are decoded all-or-nothing (§4.5).

| Contract id | Transport | Tower | iOS |
|---|---|---|---|
| `world_builder.status/2026-09-10` | result channel | T7.4 (`results/world_builder.py`) | `TowerWorldBuilderClient` (K13, K14) |
| `scene_understanding.live/2026-08-27` | result channel; the only `live` result type | `SceneLive.latest()`, via T7.3 | `TowerSceneUnderstandingClient` (K24, K25) |
| `document_memory.status/2026-09-07` | result channel | `DocumentStatusProducer`, via T7.3 | `TowerDocumentMemoryClient` (K27) |
| `document_memory.library/2026-09-07` | HTTP (bulk, sensitive text) | `routes/documents.py`, T8.5 | `DocumentMemoryDecoder.library(from:)` |
| `experimental_cv.status/2026-08-27` | result channel, plus `cv_lab_*` commands on `/ws` | CV Lab status via T7.3; T8.7-T8.9 | `TowerExperimentalCVClient` (K26) |
| `object_memory.observations/2026-08-26` | HTTP only, undeclared on the socket | `routes/observations.py`, T8.2 | `TowerObjectMemoryClient` (K29), `ObjectMemoryDecoder` |
| `world_builder.geometry/2026-08-25` | HTTP | T7.5 | `WorldGeometryClient.swift`, K16 |
| `world_builder.worlds/2026-09-10` | HTTP | `results/world_builder_library.py` | `WorldLibrary.swift` (Saved Worlds) |
| `cartridge_results.envelope/2026-08-23` | result channel | `results/envelope.py` | `CartridgeResultChannel.swift` |

**The envelope** is `cartridge_results.envelope/2026-08-23`. Its fields:
- `seq`: dense per subscription.
- `revision`: opaque; it changes only when the payload changes.
- `revision_changed` and `coalesced`.
- `tower_sent_at`, with `time_basis: tower-receipt`.

### 6.3 Socket messages on `/ws`

| Client → Tower | Tower reply | Tower code | iOS side |
|---|---|---|---|
| `ping` | `pong` | T1.1 | `TowerClient.swift` (K7) |
| `frame` | exactly one `frame_result` or `frame_error` (`invalid_frame`, `frame_skipped`/module reason, `module_unavailable`) | T1.2, T2.2, T8.7 | `sendFrame` (K9), `handleFrameResult` (K10), `handleFrameError` (K11) |
| `stream_start` / `stream_stop` | none | T1.4 (mints the capture id, attaches workers) / T4.2 | `TowerClient.sendStreamStart`/`sendStreamStop` (`TowerClient.swift:1065`/`:1096`), from `GlassesConnection` (§4.2) |
| `cartridges` | `cartridges` (the declaration) | `results_ws.handle` (L107), `results/registry.py` | `TowerClient.swift:1257/1603`, `CartridgeAvailability.resolve` (K23) |
| `result_subscribe` | `result_subscribed` or `result_error`, then the `cartridge_result` stream | T7.1-T7.4 | `CartridgeResultChannel.swift`, `TowerWorldBuilderClient` (K13, K14), `SceneUnderstandingClient` (K24) |
| `result_unsubscribe` | `result_unsubscribed` | `results_ws.handle` | `CartridgeResultSubscription.swift` |
| `cv_lab_status/start/pause/resume/stop` | `cv_lab_status` (+`accepted_command`) or `cv_lab_error` | T8.9, T8.8 | `ExperimentalCVClient.swift` (K26), `TowerSocketMessages.swift` |
| any other | `protocol_error` `unknown_message_type` | T1.1 | `handleInboundMessage` (K10) |

The `result_error` reasons are a closed set: `malformed_request`, `unknown_cartridge`, `unknown_result_type`, `contract_mismatch`, `cartridge_unavailable`, `too_many_subscriptions`,
`unknown_subscription` and `snapshot_failed`.

**The iOS side of the socket.**
- **Handshake.** The Tower never speaks first. The phone sends `ping`, needs `pong` within a 6 s leg, and only then sends `cartridges`
  (K7).
- **The bracket.** Frames go out only between `stream_start` and `stream_stop`, and while not paused (K9). `stream_start` gets no reply
  and does not survive a socket drop, so `ProjectManager` re-sends it when the socket comes back online while streaming (K1).
- **Frame fields.** The phone sends `{type:"frame", seq, tx_seq, width, height, format:"jpeg", data}` (K9). `seq` counts every frame DAT
  delivered and `tx_seq` only frames actually sent, so a gap at the Tower means transit loss. The phone sends no `source_seq`: the
  Tower accepts it as optional and uses `seq` in its place (`tower/frames.py:62`, §2 S0).
- **Results.** One router, `handleInboundMessage`, switches on `type` (K10). Delta envelopes are refused, snapshots only (K14).
  Subscription ids restart at `sub-1` on every socket (K13); a closing error spends one of 3 resubscribes per connection
  (`CartridgeResultSubscription.closedByTower`).
- **Refusals.** A `frame_error` comes once per frame whenever CV Lab is not running; the phone publishes a changed refusal at once and
  a repeat at most once a second (K11).

### 6.4 HTTP routes (line = the route decorator)

| Route | Tower code | iOS side |
|---|---|---|
| `GET /health` | `routes/health.py:13` (module, capture, workers, sessions, chore) | `TowerHealth.swift`, `TowerConnectionTester.swift` |
| `GET /cartridges` | `routes/cartridges.py:27`, `results/registry.py` | `TowerConnectionTester.swift` (identity check) |
| `GET /cartridges/{c}/session`, `POST …/session/{action}` | `routes/sessions.py:91/96` (T2.3, T4.1) | `WorldBuilderSessionController` (K12), `ObjectMemorySession.swift` |
| `GET /worlds` | `routes/geometry.py:57`, `results/world_builder_library.py` | `WorldLibrary.swift` (`WorldListClient.worlds()`, `WorldLibrary.swift:216`), called through `WorldBuilderViewModel.loadWorlds()` (`WorldBuilderClient.swift:1358`) |
| `GET /worlds/{id}/geometry/manifest`, `/segment/{n}` | `geometry.py:84/96`, T7.5 | `WorldGeometryClient.swift`, `geometryDidChange` (K16) |
| `GET /worlds/{id}/render`, `/render/revision` | `geometry.py:219/197`, T7.6 | `WorldRenderViewerModel` (K19, K20), `WorldRenderWebView` (K21) |
| `GET /worlds/{id}/appearance/{sid}/manifest`, `/chunk/{d}`, `/proxy/{d}` | `geometry.py:152/183/190`, `results/world_builder_appearance.py` | `WorldAssetTransport.swift` |
| `GET /worlds/{id}/areas/{sid}/{area}/render[/revision]`, `/appearance/…` | `geometry.py:317-395` | `WorldAssetTransport.swift`, `WorldRenderViewerModel.swift` |
| `GET /worlds/{id}/guidance/{sid}/imagery/manifest`, `/tile/{d}` | `results/world_builder_guidance_imagery.py:25/51` | `WorldImagery.swift` |
| `GET /worlds/{id}/guidance/{sid}/photo-floorplan/manifest`, `/tile/{d}` | `results/world_builder_photo_floorplan.py:54/65` | `WorldPhotoFloorplan.swift`, `WorldCoverage.swift` |
| `GET /object-memory/observations`, `/last-seen/{class}`, `/observations/{id}/imagery\|frame\|crop` | `routes/observations.py:131-293`, T8.2 | `ObjectMemoryClient.swift` (K29), `ObjectMemoryImagery.swift` |
| `GET /documents`, `/search`, `/around`, `/{id}`; `GET/POST /documents-session[/start\|pause\|resume\|stop]` | `routes/documents.py:98-260`, T8.5 | `DocumentMemoryContract/Library/Model.swift` (K27) |
| `GET /scene`, `POST /scene/start\|pause\|resume\|stop` | `routes/scene.py:82-121` | Operator only; the phone "watches" by subscribing (`SceneUnderstandingClient.swift:79`) |
| `GET /cv-lab`, `/cv-lab/preview`, `/cv-lab/preview/status` | `routes/cv_lab.py:29`, `cv_lab_preview.py:149/127` | `ExperimentalCVContract.swift` |

---

## 7. Config

### 7.1 How the Tower reads its settings

Every setting is an optional environment variable, read by `tower/config.py`. `scripts/start_tower.ps1` passes `tower/.env` to uvicorn (`--env-file`); `tower/.env.example` is the documented
list. Children inherit the Tower's environment, so changing a World Builder switch means restarting the Tower. Booleans accept `1`, `true`, `yes` or `on`. A typed switch logs a typo and
falls back to off.

### 7.2 The promotions: the demo value is now the only code path

**Tower: 13 promotions** (no longer switches):

| Was | Behaviour now | Lives in |
|---|---|---|
| `TOWER_WORLD_NATIVE_CHROME=1` | Appearance page for a phone that draws its own controls | `results/world_builder_render.py`, `routes/geometry.py` |
| `…_GUIDANCE_COVERAGE=on`, `…_COVERAGE_PENDING=on` | Status carries `guidance.coverage`, including pending stations | `results/world_builder.py`, `world_builder/guidance_coverage.py` |
| `…_GUIDANCE_IMAGERY_LIVE=on`, `…_LANDED=on` | B1/B2 guidance imagery always served | `results/world_builder_guidance_imagery.py` |
| `…_PHOTO_FLOORPLAN=on`, `…_METHOD=homography`, `…_DIAGNOSTICS=on` | Floor-plan worker on, homography pinned, diagnostics on | `world_builder/photo_floorplan_worker.py` |
| `…_LIVE_DEPTH=on` | Depth predicted during the walk | `world_builder/live_depth.py`, `engine.py`, `surface_pipeline.py` |
| `…_FINAL_SCALE_GUARD=on` | Guard on every final publish; the off and shadow arms are deleted, and later v1 `assess` (p3-12), so v2 is the only assessor | `world_builder/final_scale_guard.py`, `coherence_publish.py` |
| `…_TRANSIENTS_FP16=on` | Half-precision transient masks | `world_builder/transients.py` |
| `…_KEYFRAME_TURN_BRIDGE=on` | Keyframes through fast head turns | `world_builder/keyframes.py`, `engine.py` |
| `…_UP_ESTIMATE=camera_right_all` | Room "up" from all cameras' right vectors | `world_builder/surface_render.py` |

**Also deleted, though not one of the 14:** `TOWER_WORLD_SOLVE_SELECT`. Off is now the only path, and `solve_select.py` is gone.

**Proof the promotions changed nothing:** gate 2 ran the candidate with no switch lines against the baseline with the demo env. The seeded final solve and the surface stage came out
byte-equal (§8.3).

**iOS: 4 promotions, always on now:** `IOS_FOW_DIAL` (capture dial), `IOS_FOW_ROOM_IMAGERY_LIVE`, `IOS_FOW_ROOM_IMAGERY_LANDED` (live
strip and landed thumbnails) and `IOS_FOW_PHOTO_FLOORPLAN`. Their switches, Info.plist keys and off paths were deleted; `git grep` finds no
trace in `ios/`. This matches the Tower's own promotions.

The point (§9.3): a default Tower equals the demo Tower, and a plain Xcode Run equals the demo build (§7.7).

### 7.3 The one Tower demo switch left: `TOWER_WORLD_GUIDANCE_IMAGERY_UNREDACTED`

- **Reading.** `world_guidance_imagery_unredacted_setting()` (`config.py:989`) is true only for exactly `on`. `1` and `true` mean off. **When on.** The guidance imagery routes may serve a
  session whose keyframe-set label is exactly `none`: no redactor ran, because the Tower's redactor was bypassed or unavailable when the walk was recorded, as in the privacy-deferred period around 2026-10-12 (`tower/docs/contracts/WORLD-BUILDER-GUIDANCE-IMAGERY.md`). The demo walks 9.3 and 10 themselves are redacted (`README.md`, Privacy and safety). Each entry it serves is marked `"redaction": "unredacted"`. **When off.**
  Such a session is a 404. **Either way.** Other untrusted labels are refused, and appearance, depth and world-build trust are untouched. **Why it stays a switch.** It is a privacy decision,
  not a quality one, and must be an explicit opt-in per deploy. The live `.env` sets it; `.env.example` lists it commented out, so off (§11.B, item 3).

### 7.4 Tower defaults that shape a walk

| Setting | Default | Effect |
|---|---|---|
| `TOWER_CAPTURE_ROOT` | unset | Unset means no recording, ever. Use `data` |
| `TOWER_WORLD_ROOT` | unset | Unset means iOS sees World Builder as unsupported. Use `data/world_builder` |
| `TOWER_WORLD_AUTOBUILD` / `_REBUILD_EVERY` | true / 4 | Attach a builder (when the World Builder session is open); interim builds every 4 keyframes (the script's own default is 0) |
| `…_REGISTER` / `_SOLVE` / `_SURFACE` / `_APPEARANCE` / `_FINISH_PENDING` | true | The ladder and the finisher |
| `TOWER_WORLD_DENSIFY` | false | No dense stage |
| `TOWER_WORLD_SOLVE_SEED` | unset | Unseeded and multi-threaded. An integer seeds a single-thread solve (about 3.3x the mapping time) |
| `…_SOLVE_MASKS` / `_GATE` / `_CONSENSUS` / `_ANCHOR_VERIFY` / `_AREA_BUILDS` | off / off / 1 / off / off | The coherence product: merged but not switched on |
| `TOWER_WORLD_RELOCALIZER` | off | No look-back relocalizer (`prompt` or `silent` turns it on) |
| `…_FINISH_STAGES`, `_PICTURE_BASIS`, `_FRAME_QUALITY_LOG`, `_STAGE_TIMING`, `_POSE_QUARANTINE` | off | Reporting and research |
| `TOWER_WORLD_FINAL_IMMEDIATE` | unset (off) | At capture Stop, stop the background solve and start the final solve at once instead of waiting for it. Read straight from `os.environ` by the builder (`world_build_session.py:2476`), exactly `on`; not in `config.py`, documented in `.env.example` |
| `TOWER_HOST` / `TOWER_PORT` | `0.0.0.0` / 8000 | Inert: uvicorn's `--host` and `--port` decide (`start_tower.ps1 -BindHost`) |
| `TOWER_DEV_MODE` | true | Debug logging |
| `TOWER_CV_EXPERIMENT` / `_DEVICE` / `_TORCH_THREADS` | baseline / auto / auto (2 on CUDA) | The CV Lab's boot experiment |
| `TOWER_SCENE_UNDERSTANDING` / `_AUTOSTART` / `_ORIENTATION` | auto / true / true | In-process Scene |
| `TOWER_DOCUMENT_ENABLED` / `_CAPTURE` / `_AUTOSTART` / `_RETENTION_DAYS` | true / true / false / 30 | Document Memory |
| `TOWER_OBSERVATION_ENABLED` / `_RETENTION_DAYS` / `_KEEP_IMAGERY` | true / 30 / true | Object Memory |

### 7.5 iOS switches left

| Switch | How it is set | Default | What it does |
|---|---|---|---|
| `IOS_TOWER_TARGET_FPS` | launch environment, or build setting via Info.plist `GlassesTowerTargetFPS` | **15** | Tower send rate; whole number 6-24, else 15. Read once per process (FrameRateGate.swift:89-109); also sizes the send window (15 x 1/3 s = 5 slots). Was 12 before the campaign; 15 is what the demo build passed. |
| `IOS_FOW_MAP_TAB` | launch environment, or Info.plist `GlassesFOWMapTab` | **off** | Shows the Dial \| Map control on the walking stage only when exactly `on` (WorldCaptureDial.swift:32-47). A UI experiment; off, the dial is the only face. |
| `GLASSES_BUILD_SHA` | build setting, via Info.plist `GlassesBuildSHA` | empty | The commit, written into the DEBUG IMU log header. |
| `GLASSES_TOWER_AUTHORITY` | launch environment, DEBUG only | unset | Overrides the Tower `host:port` (TowerConfiguration.swift); see §4.3 for the fallback order. |
| `GLASSES_ATS_ALLOWS_ARBITRARY_LOADS` | build setting | Debug `YES`, Release `NO` | A "Derive Info.plist" build phase removes `NSAllowsLocalNetworking` and adds `NSAllowsArbitraryLoads` in Debug, so the Simulator can reach any Tower. |

Developer-only, non-persisted controls in Developer Tools (`captureResolution`, `motionProbeEnabled`, `imuRecorderEnabled`) reset to off or
`.low` on every launch and are read once when the camera starts.

### 7.6 Info.plist keys (`ios/Glasses/Info.plist`)

- **DAT required section:** URL scheme `glasses`; `LSApplicationQueriesSchemes` = `fb-viewapp` (to open the Meta AI app);
  `UISupportedExternalAccessoryProtocols` = `com.meta.ar.wearable`; `MWDAT` dict (`AppLinkURLScheme` `glasses://`, `MetaAppID` `0` for
  Developer Mode, `TeamID`); `UIBackgroundModes` bluetooth-central, bluetooth-peripheral, external-accessory; Bluetooth, local-network and
  camera usage strings (the camera is used for Mock Device Kit's simulated feed); `NSBonjourServices`.
- **Glasses keys:** `GlassesFOWMapTab`, `GlassesTowerTargetFPS`, `GlassesBuildSHA`, all `$(BUILD_SETTING)` substitutions.
- **ATS:** this file is the Release policy; plain `ws://` and `http://` only to private networks: `NSAllowsLocalNetworking`; exception
  domains for RFC 1918, link-local, Tailscale CGNAT `100.64.0.0/10`, IPv6 `fc00::/7` and `ts.net`; the development Tower's own address.
  `TowerTransportSecurityTests` checks that this list and Settings' `TowerAddressPolicy` agree.

### 7.7 Build and install: a plain Xcode Run equals the demo build

- **A plain Xcode Run equals the demo build:** open `ios/Glasses.xcodeproj`, scheme `Glasses`, the iPhone as destination, Cmd-R. Run uses
  **Debug**; no shared scheme is committed, so Xcode's default applies. With no flags a build gets 15 fps, dial, imagery and photo floor
  plan on, Map tab off (the demo values).
- **Debug matters.** The whole capture surface is `#if DEBUG`: `GlassesConnection`'s session, camera and frame code,
  `TowerClient.sendFrame`, the stream bracket. A Release build compiles but has no camera capture; it can only read the
  Tower (status, saved worlds, viewer).
- **Command line:** build with `xcodebuild build -project Glasses.xcodeproj -scheme Glasses -configuration Debug -destination
  'generic/platform=iOS' -allowProvisioningUpdates`; install with `xcrun devicectl device install app --device <id> <path/to/Glasses.app>`.
  Walk procedures read `[Glasses][...]` stdout lines through `devicectl device process launch --console`, which is why those lines are DEBUG
  `print`s, not `os.Logger`.
- **Measured run:** pass `IOS_TOWER_TARGET_FPS=<n>`, `IOS_FOW_MAP_TAB=on` or `GLASSES_BUILD_SHA=<sha>` to `xcodebuild` as build settings, or
  set the first two in the scheme's launch environment.

---

## 8. Testing and verification

The Tower is gated by its pytest suite and by replaying walk 10 end to end; the phone by its unit and UI suites.

### 8.1 Tower: the suite

- **How to run.** From `tower/`, with the venv: `.venv\Scripts\python.exe -m pytest`. There are 297 `test_*.py` files. The `slow` marker covers tests that spawn real processes and build real
  worlds.
- **Baseline** (`5fecd51`, with coverage): 7,036 passed, 5 failed, 82 skipped, 1 xfailed; 7,124 collected; 27 minutes. All 5 failures were in
  `test_world_builder_final_scale_guard_off_identity.py`. Coverage: 79.3% combined. `tower/` packages about 89-99% of lines; `scripts/` 44.6%.
- **After the campaign** (`e20b3b2`, the Tower code in `66414dc1`): **6,797 passed, 0 failed, 83 skipped.** Tests of the deleted off-paths went with them. The 5 baseline failures went
  with the guard's off arm (the scale-guard promotion, §9.3). On the way: 6,866 / 5 at integration, 6,818 / 0 at `167c50d`, 6,821 / 0 after batch 20 (which added the Scene fix's
  tests), then 6,797 / 0 after batch 21, down 24 because the v1-only final-scale-guard tests were deleted with v1 `assess` (25 deleted, 1 added; §9.3). A tracemalloc flake turned out to be a test-only
  thread leak (fixed).
- **Architecture tests** (`tests/test_architecture_boundaries.py`): `test_shared_code_does_not_import_a_cartridge`; `test_the_result_channel_core_is_cartridge_blind`;
  `test_the_result_channel_never_writes`; `test_a_cartridge_does_not_import_another_cartridge`; `test_the_capture_worker_supervisor_is_cartridge_blind`;
  `test_importing_the_lab_does_not_import_torch`; plus `test_artifact_paths.py::test_every_cli_root_flag_routes_through_the_guard`

### 8.2 Tower: per-unit gates

- Every unit had two reviewers (R1, R2). When they split, the manager decided. Unit gates ran by script against the base and had to say SAME. Promotions were batched behind replay gates
  (§8.3). No bundle was pushed with a red suite.

### 8.3 Tower: the replay gate, walk 10 end to end

**Tool.** `world_live_replay_run.py`, pinned at `297ab0a` and frozen by `setup.ps1`. It re-streams walk 10's 2,920 JPEGs at the recorded pace, through the phone's own door: `ping`,
`cartridges`, `result_subscribe`, `POST …/world_builder/session/start`, `stream_start`, the frames, `stream_stop`. The target is a test Tower on port 8031 or higher, with a fresh data root.
That Tower spawns the real follower, so Stop, the finisher and phone-style fetches all happen. **Fixed inputs:** code: a `git archive` export; intrinsics: frozen (`intrinsics.sha256`);
switch file: `demo.env` (14 lines + `TOWER_DEV_MODE`) or `defaults.env` (`TOWER_DEV_MODE` only); environment: every inherited `TOWER_*` variable is dropped.

**Procedure** (`REPLAY-PROCEDURE.md`):
1. Run `setup.ps1` once.
2. Preflight: `:8000` is not recording, no pytest is running, and the GPU is free.
3. Run `run-replay.ps1 -Label b1` as an `/IT` scheduled task (the runner owns a kill-on-close job object). Wait for `replay\b1.DONE`.
4. Run `render-pack.ps1` (plan, 4 stations x 8 yaw views, viewer opening), then the visual-QA agent.
5. Run `compare.py --base b1 b2 --cand p1` and `pose_robust.py`.

Each run takes about 18-22 minutes. Never run it alongside the suite.

**Judging** (`NOISE-BAND.md`): **Deterministic set, band 0.** Ingest: 2,920 frames, 1,148 keyframes, and the reject histogram. Stage states: surface ok, appearance ok, dense unavailable.
Guard: published, `uncertified`, `live-depth-cache`. Surface: canonical and mobile levels, `privacy_safe`, `imagery_source`. 37 segments, 47 appearance chunks, 128 phone keyframes. **Noisy
set.** Values must fall inside the baseline [min, max], widened by half its width. **Pose.** The check is a robust Sim3 fit against b1: seeded with in-range cameras, trimmed to 90%. The
median residual/radius must be ≤ 0.14, the worst baseline pair. Plain Umeyama is unusable because far-out cameras dominate it (b1↔b2 scale 0.141). **Timings.** Information only. A run more
than 15% slower than the slower baseline is flagged. **Baselines.**

  | Run | Surface vertices | c0 | Phone coverage |
  |---|---|---|---|
  | b1 | 2.17M | 744 | 0.963 |
  | b2 | 2.16M | 750 | 0.967 |
  | b3 | 1.72M | 719 | 0.938 |

**Gate 2** (`GATE2-VERDICT.md`; candidate `2bc8abc`, all promotions): **PASS within noise.** **Exact (Tier A), on frozen inputs with switches unset.** Seeded final solve on the frozen `bs`
database, base `5fecd51` (demo env) against the candidate (defaults env): 4,920 files, 0 normalised diffs, poses digest `5fceb2e72004`. The surface stage on that solution: the manifest is
identical apart from timings, and all 6 mesh and confidence binaries are byte-equal. **Live replays.** p2 (defaults env) and p2b (demo env) matched the deterministic set exactly. Robust pose
medians: 0.033 and 0.071. Visual QA: the imagery is the same.

**Crossover** (`CROSSOVER-REPORT.md`): **code-independent.** **Exact check on a second world.** The b4-derived database gave 4,963 files, 0 diffs, poses digest `bc08fd777036`. So base
and candidate produce byte-identical worlds from identical inputs, on two worlds, through solve, gate, guard, publish and records. **Unseeded re-solves vary alike.** Re-solving the same
starting databases with each arm:

  | Database | Arm | c0 range | Runs with far-out cameras |
  |---|---|---|---|
  | b4 | base | 714-831 | 6 of 9 |
  | b4 | candidate | 741-861 | 6 of 9 |
  | p2 | base | not given | 3 of 4 |
  | p2 | candidate | not given | 2 of 4 |

  The most extreme run (max/median spread 14,162) was a base run. **Why runs vary.** Matching is nondeterministic even when seeded: Yb1 and Yb2 ran the same code with seed 0 and produced
  different databases. **The damage tally is noise.** It read 1 of 5 baselines against 4 of 4 candidates, but p1 and p1b ran solve-path code byte-identical to base and still counted as
  "damaged".

**Final campaign replay** (walk 10 on `cleanup/campaign-v1`): PASS, within noise (f1). The deterministic set (ingest, keyframes, rejects, stage states, guard decision class) is identical and every solve metric is in band; the robust pose median residual/radius against b1 is 0.036 (gate at most 0.14); the harness replay-fidelity FAIL is only client send lateness (p95 5.79 ms against 5 ms), the same flag b1-b4, p2 and p2b carry; Stop to photos 11.69 min. Visual QA (sonnet): PASS within noise, with 1 mild excursion of the known unseeded-solve class (oblique skin 13.1% against 0-5.5% in b1-b5, one ghost photo slab); no far-flung surface (0% beyond 8 m), upright and plumb, viewer up refinement accepted (7.96 deg)..

### 8.4 iOS: targets

- **`GlassesTests` (unit):** 28 files, about 165 test classes, about 1,580 tests. The last batch gate (`af805b3`, iOS complete; the iOS tree
  is unchanged at `66414dc1`) ran 1,578/1,578, with no flakes; earlier gates ran 1,572/1,577, the 5 failures being the known mock-device
  flakes (§8.6). Pre-cleanup baseline 1,558. Most tests are pure-value or run against
     loopback servers: `MockTowerServer.swift` is a Network.framework WebSocket server on localhost, and the "RealTower" classes replay
     bytes captured from a real Tower without needing one.
- **`GlassesUITests` (UI):** 10 test files plus 3 helpers, 91 tests. `MockTowerHTTPServer` and `WorldPanelMockTower` stand in for the Tower;
  some tests expect a real Tower at `TEST_RUNNER_GLASSES_UITEST_TOWER_AUTHORITY`. Pre-cleanup baseline 73 pass, 16 fail, 3 skipped. The
  gate before last (`03f6279`) gave 70 pass, 17 fail, 3 skipped, 1 expected failure; the last gate (`af805b3`) gave 67 pass, 20 fail, 3
  skipped, 1 expected failure. Its extra failures were flakes, each passing in isolation or on rerun; the expected red set is §8.7.

### 8.5 iOS: how to run (simulator; one `xcodebuild` at a time on the 8 GB Mac)

```
cd <worktree>/ios
xcodebuild test -project Glasses.xcodeproj -scheme Glasses \
  -destination 'platform=iOS Simulator,id=<simulator-udid>' \
  -derivedDataPath <derived-data-dir> \
  -only-testing:GlassesTests -resultBundlePath <out>.xcresult
# UI: same, with -only-testing:GlassesUITests and
#     TEST_RUNNER_GLASSES_UITEST_TOWER_AUTHORITY=127.0.0.1:8010 in the environment
xcrun simctl shutdown all
```

Simulator: iPhone 17 Pro. Rerun one class with `-only-testing:GlassesTests/<Class>`. Build-warning baseline is 3 lines of tool noise; the campaign
cleared the 34 real Swift 6 warnings.

### 8.6 iOS: known flaky tests (pre-existing; rerun before treating one as a regression)

- `IMURecorderMockDeviceTests`, `MotionProbeMockDeviceTests`: "mock glasses never became an active device" after a cold simulator start;
  fail about once in 4-5 full runs, pass on a class rerun.
- `IMURecorderMockDeviceTests/testA60HzRefusalAtRunTimeFallsBackTo30Once` (ProductShellTests.swift:4826): fails about 2 runs in 10 on the
  base too.
- `CameraReadinessTests`; `TowerClientTests/testDisconnectClearsTheStreamBracket`.
- `TowerClientTests/testReceiveLoopHandlesMultipleFrameResultMessages`: the loopback socket sometimes stays `connecting` for 6 s.
- `CVLabPreviewBytesTests/testAStartRecordsAPendingCommandWithTheRequestIDItSent`: loopback "never came online".
- UI: `WorldImageryUITests/testThePhotoFloorplanHoldsThenExpiresForANewerLanding`: a pixel-sample timing flake.

### 8.7 iOS: known UI failures (expected red)

- **Need a reachable Tower or a real page** (the gates run with nothing listening on 8010): `TowerSmokeUITests`
  (`testOpeningASavedWorldShowsThe3DWorld`, `testTheLabConnectsAndSwitchesExperimentsFromItsOwnScreen`); `WorldChromeUITests`
  (`testTheCanvasShareOnTheRealPage`, `testTheColdOpenSaysWhatItIsDoing`, `testTheRealPageHandsOverItsChrome`,
  `testTurningAwayShowsTheDarkLine`); `WorldInlinePanelUITests/testR1ARealPageCrossFadesInlineAndKeepsItsPose`.
- **`AccessibilityAuditUITests`:** `testCartridgeDrawer`, `testOpenedWorld`, `testSettings`,
  `testTheNoticeIsOnTheScreenWithoutScrollingAtTheDefaultSize`, `testTheWorldKeepsMostOfTheScreenAtTheLargestTextSize`,
  `testTheWorldKeepsMostOfTheScreenWithAnAreaOpenAtTheLargestTextSize`, `testWorldBuilderIdle`, `testWorldList`. These include pre-existing
  contrast failures on the World Builder idle buttons (owner backlog, §9.5).
- **`DeadEndsUITests/testTheLeavePromptShowsBothChoicesAtTheLargestTextSize`:** pre-existing in the demo (dial-on) configuration, now in
  every build; at AX5, Start may be unreachable. Owner backlog, HIGH (§9.5).
- **`WorldImageryUITests/testTheLandedThumbnailSitsInItsPieceAndATapSaysDirectionAndAge`:** an `XCTExpectFailure` on the Map-face sheet,
  reachable only with the default-off Map tab.
- **Now passes:** `WorldInlinePanelUITests/testEveryPanelStateAtAX5`, formerly a baseline failure.

### 8.8 iOS: what the main test classes protect

- **`SenderPipelineTests`:** `FrameRateGateTests` (15 fps held under jitter, re-anchoring; `testTargetIsHeldAtTheRoadmapFifteenFPS`, :86),
  `SendWindowTests` (capacity arithmetic, stalls), `SenderMetricsTests` / `SlotTimingMetricsTests` (every frame accounted for).
- **`TowerClientTests`** (loopback server): handshake, receive loop, stream bracket, send failures; `TowerReconnectGiveUpTests`;
  `WorldBuilderSessionOrderingTests`; `testCartridgeViewModelsSendNothingToTheTower` (:1513).
- **`ProductShellTests`:** catalog and workspace switch; `CartridgeIntegrationTests` (availability); `ConnectionLifetimeTests`,
  `CaptureSessionClaimTests`, `DATStartWarningWatchdogTests`; IMU log format and recorder; `DeadEndsTests`; result-subscription bookkeeping.
- **`WorldBuilderIntegrationTests`:** payloads per contract date; `TowerWorldBuilderClientTests` (subscribe, pins, acks); session binding;
  `WorldRenderViewerTests`, `WorldRenderRevisionTests`; `WorldAssetTransportTests` (scheme whitelist).
- **`WorldGeometryTests`:** manifest and segment decoders, placement cache, retry, fetch lifecycle, real-Tower fixtures.
- **`WorldPresentationTests`:** `WorldGeometryAccountTests`, `WorldStageTests`, `WorldReconstructionLadderTests`, `WorldStageMacGateTests`.
- **`WorldChromeTests`:** the protocol-1 reducer; also scans `WorldChromeProtocol.swift` to keep it free of the page's words.
- **Walk guidance:** `WorldCaptureDialTests`, `WorldCoverageTests`, `WorldImageryTests`, `WorldPhotoFloorplanTests`,
  `WorldInlinePanelTests`, `WorldFinishProgressTests`.
- **Cartridges:** `SceneUnderstandingTests` (incl. `SceneSubscribeRaceTests`), `CVLabContractTests`, `DocumentMemoryTests`,
  `ObjectMemoryTests`.
- **Settings and accessibility:** `TowerSettingsTests` (address policy, ATS agreement), `AccessibilityContrastTests` (4.5:1 text contrast).
- **Outside XCTest:** `ios/scripts/contract-drift-check.py` and `cross-stack-constants-check.py` catch drift against a live Tower that
  hand-captured fixtures cannot.

---

## 9. What the campaign changed, and known issues

Sources (run records outside the repository; see Provenance): `LEDGER.md` (iOS), `LEDGER-tower.md`, `PLAN.md`, `OWNER-NOTES.md`, `IOS-INVENTORY.md`, `GATE2-VERDICT.md`, `CROSSOVER-REPORT.md`. Baseline tag `pre-cleanup-20261015` (5fecd51); the campaign branch is `cleanup/campaign-v1`. `main` stays pre-cleanup as the interview fallback, and the rollback tag `demo-walk10-20261013` is untouched. The code itself is in §3-§5. The Tower half's own list of known limitations is merged into §9.5.

### 9.1 Goal and rules

- **Goal.** Every file checked: the repo made clean, readable and navigable, with dead code and files removed and every function checked for simplification and optimization. World Builder must keep working exactly as in the demo walk 10.
- **Two reviewers per change.** R1 is a Claude agent (opus when a miss could break World Builder output). R2 was Codex, falling back to a fresh Claude agent when Codex's quota ran out. The author never reviews its own work. Both yes: land. Both no: the file stays as is. A split goes to the manager, who reads the code and decides; the reason goes in the ledger.
- **Quality is never traded for readability.** Reviewers rejected many "cleanups" for dropping a measured number, a "why" comment or an observable call. Every measured number in comments was kept.
- **Landing tiers.** Tier A = byte-identical outputs (refactors, dead-code removal, comments). Tier B = same end result within tolerances declared beforehand. A "better" change (e.g. the offline coverage rebuild now matching the live receipt) needs both reviewers plus visual QA. Every Tower change needs the full suite with no new failures vs the baseline; every Swift change needs build plus unit tests.
- **Keep the cartridges, delete history.** Object Memory, Document Memory, Scene Understanding and CV Lab are cleaned, never removed. Handoffs, reports, research scripts and incident narration are removed; the tag keeps them. Promotion rule: the demo's switches become the defaults and their off paths are deleted.
- **Trim (10-16).** To protect the owner's walks 10.1/10.2, dropped to the post-interview phase: Tower cartridge polish, dead-test cleanup, scripts beyond World Builder; iOS dead-test cleanup and the "WB everything else" unit (it became a comment-only citation sweep).

### 9.2 Totals

Measured with `git diff 5fecd51..66414dc1` (the final campaign tree). The figures include docs and tests. A tracked-file count of "units" is approximate because ledger rows also record gates, fixes and incidents.

| Scope | Files changed | Added | Removed |
|---|---|---|---|
| Whole repo | 776 | 19,248 | 214,882 |
| `tower/` | 560 | 9,268 | 176,003 |
| `ios/` | 162 | 10,240 | 12,254 |
| Swift files only | 138 | 8,437 | 7,756 |
| Tower `.py` files only | 351 | 8,691 | 31,283 |
| Markdown files only | 224 | 1,706 | 95,568 |

- Net: roughly 196k lines gone, most of it history docs and research scripts (about 95k lines of markdown), then dead code and off paths. 371 commits on the branch.
- **iOS:** 20 units landed, all passing both reviewers. Unit tests grew from 1558 to 1578 as characterization tests were added. Build warnings fell from 37 to about 3, all tool noise.
- **Tower:** about 60 unit rows from p2 to p5 in `LEDGER-tower.md` by 10-16, then 13 late units: batch 20 (the Scene fix p5-01 and eight P3 readability units), batch 21 (p3-12, p3-13, p3-16) and p3-28 (config docs), each with R1 and R2 approving (R2 by Claude while Codex was out). The full suite moved from 7,036 pass / 5 fail (baseline) to 6,818 pass / 0 fail after the 13 promotions; the 5 baseline failures were deleted with the final scale guard's off arm (the scale-guard promotion). It ended at 6,797 pass / 0 fail / 83 skipped (`e20b3b2`): batch 21 took it down 24, because the 25 v1-only guard tests went with v1 `assess` and one test was added (§9.3).
- Fewer tests than baseline is expected: off-path tests were deleted with the paths.

### 9.3 The big changes, by theme

**Switch promotion, Tower (13 promoted, 1 remains).**
- Promoted to defaults, off paths deleted: `UP_ESTIMATE` (camera_right_all), `TRANSIENTS_FP16`, `NATIVE_CHROME`, `FLOORPLAN_METHOD` (homography only), `FLOORPLAN_DIAGNOSTICS`, `COVERAGE_PENDING` (include_pending True), `TURN_BRIDGE`, `IMAGERY_LIVE`, `IMAGERY_LANDED`, `PHOTO_FLOORPLAN`, `COVERAGE`, `LIVE_DEPTH`, and `FSG` (final scale guard v2). Names are shortened from `TOWER_WORLD_*`.
- Remaining switch: `GUIDANCE_IMAGERY_UNREDACTED` (a privacy-relevant opt-in, so it stays explicit).
- Also deleted: `SOLVE_SELECT` (+ `_BUDGET_S`), which was not in the demo and had no planned lane. `FINAL_IMMEDIATE` was kept as a live experiment and P4 latency candidate (owner decision, 10-15); p3-28 documented it in `tower/.env.example`. It stays off unless set to exactly `on` (§7.4).
- WHY: the demo ran 14 switches by hand. A default Tower must equal the demo Tower, and two code paths per switch doubled the test and review surface. Each promotion was proven Tier A on fixtures (for example, a 168-response golden matrix for native chrome; page bytes identical on 3 worlds for up estimate).

**Switch promotion, iOS, and the 15 fps default.**
- The campaign promoted the four demo iOS switches (`IOS_FOW_DIAL`, `ROOM_IMAGERY_LIVE`, `ROOM_IMAGERY_LANDED`, `PHOTO_FLOORPLAN`): switch, Info.plist key and off path deleted. The Tower send-rate default went 12 to 15 fps, the demo rate (walk 10: 2920 frames in 196.3 s).
- WHY: a plain Xcode Run build now equals the demo build, with no hidden flags to remember. On the phone, the follow-through on the Tower promotions (native chrome, coverage pending, guidance) corrected comments but deleted nothing, because each candidate path (refusals, failed handshakes, null coverage, no-session payloads) is still real.
- Side effect: because the dial is now always on, every build carries the AX5 capture-start problem (§9.5).

**World Builder splits and refactors.**
- The viewer split: `WorldRenderViewer.swift` (3,081 lines) became State (354), Model (1,278), WebView (613) and Viewer (849). A script proved a verbatim move: 1,588 code lines and 22 types in order, no access-level changes. WHY: one file held four responsibilities and was unnavigable.
- `geometryDidChange` (K16): 251 lines became 22 lines of named steps. Same five awaits in the same order, `isStillOurs` re-checks kept in place, verified against a 28-row mapping, with 3 characterization tests written first. WHY: the ordering and the re-checks are the safety property, and they are now visible.
- The ladder rule table (K18): `WorldReconstruction.ladder` in `WorldPresentation.swift` (a 160-line nested if) became a first-match rule table. A 3,528-case matrix passes identically on the old code and the table. An exhaustive `WorldStage` switch (and `CaseIterable`) restores compile-time coverage. WHY: stages are now added as rows, not branches.
- TowerClient (K9, K10): `sendFrame` 208 to 9 lines and `handleInboundMessage` 244 to 28 lines (switch plus handlers), with a pure move into `TowerSocketMessages.swift` and `TowerHealth.swift`; `TowerClient.swift` went from 2,630 to 2,040 lines in that unit (2,057 after its review fix); the later refusal throttle (K11) brought it to 2,142, its size today. 9 characterization tests first.
- Tower: dead fields and functions removed across about 50 `world_builder` modules; key-function docstrings added to the central functions (the basis of this document's Key Functions sections).
- Final scale guard v1 deleted (p3-12, batch 21, owner decision): `assess`, `plateaus`, `_scale_round` and `GUARD_ID_V1`, 163 lines of code. A reader audit found nothing that reads v1 output or branches on the v1 id. 25 v1-only test cases went; the remaining pose and excursion tests were retargeted to `assess_v2`; a new test proves that a saved world whose `solution.json` carries the v1 id still loads and is served unchanged. v2 (`final-scale-guard/2`) is now the only assessor.

**Cartridge cleanups and bugs fixed.**
- Shared cartridge code: one `CartridgeResultSubscription` for Document Memory, CV Lab and Scene; one ownership rule `CartridgeResultError.belongs`; +9 tests.
- **Scene subscription race (K24).** If the screen appeared, left and returned before the first ack, the people detector could run for the socket's life. Fix: `isSubscribing` survives a leave, and `adoptOrClose` adopts or closes every ack. Tests fail on base (4 failures), then pass. Existed before the campaign.
- **Frame-refusal redraw throttle (K11).** Every `frame_error` made every TowerClient observer redraw at the send rate during walks: about 12 Hz under the old 12 fps default, about 15 Hz at today's 15 fps default (`FrameRateGate.defaultTowerTargetFPS`). Now a refusal publishes when its reason changes, plus a leading and trailing throttle of about 1 s; every refusal is still forwarded and the counters stay exact. Existed before the campaign.
- **Document Memory error ownership (K27).** An error naming another cartridge could clear Document Memory's status. It follows the shared rule now. Defensive only: a conforming Tower never sends it.
- **Scene operator-hold bug (Tower, p5-01, fixed).** In `scene/live.py`, a stream closing called `super().stop()` and left `_operator_hold` set, so a later watched session never stopped when its watchers left, and the people detector ran with nobody watching. Fix 1: a stream-closure (demand) stop releases the hold, in `_stop_then_reconcile`. R2's first round found that fix 1 itself introduced a race: a demand stop could clear a newer operator start. Fix 2: `_hold_epoch`, bumped by every operator `start()` and recorded under `_demand` when a demand stop is decided; a start newer than the decision wins (before the stop it cancels it; during the stop the session is restarted). 3 tests (2 of them synchronized). Existed before the campaign; landed in batch 20 with R1 and R2 approving.
- Release build fixed: a DEBUG-only `stopCameraSession` call in `ContentView` meant the Release configuration did not compile at the tag.
- Other: Object Memory's unused `idempotentNoOpLine` was deleted, since a second Start is "not an error"; the `StreamManager` placeholder and its Developer Tools row were removed.

**Dead code and history removed; comment hygiene.**
- Tower: 135 research files, 55 superpowers docs, 41 handoffs, 32 plus 21 referencing docs, 16 dead misc files, 4 one-off scripts. iOS: 4 history docs (-4,203 lines), 11 unused members, 34 Swift 6 warnings cleared (no escape hatches), 17 test-only members deleted and 8 kept as documented "Test seam"s.
- Citations to deleted planning specs (for example `U-INLINE`, cited about 50 times) were removed over 89 files, keeping each explanation.
- Comments were corrected where false (3 in the TowerClient unit, 4 in the Release-build unit) and measured numbers were restored when authors dropped them (7 in one unit). Console prints stayed DEBUG-only on stdout, because the `devicectl --console` procedures read them.

**The readable-grey accessibility sweep.**
- 87 text sites changed from `.secondary` to `.readableSecondary` (4.5:1 contrast) across the cartridge and World Builder screens plus `TowerConnectionRow`. 3 icon-only sites kept `.secondary`, which passes the 3:1 non-text bar. Three conditional titles were also changed so that no title is lighter than its captions.
- WHY: the shell already used `.readableSecondary`; the cartridge screens were the inconsistent ones (about 3.5:1).

### 9.4 How behaviour was proven unchanged

1. **Exact frozen checks (Tier A).** The final solve and the final surface stage were run from frozen inputs on base and candidate. On world `bs`: 4,920 files with 0 normalised diffs, identical poses (`5fceb2e72004`); the surface manifest and all 6 mesh/confidence bins were byte-equal. On a second world (b4-derived): 4,963 files, 0 diffs. Base and candidate give byte-identical worlds from identical inputs, covering solve, gate, final scale guard, publish and records.
2. **Replay gates (walk 10).** Gate 1 (8 promotions) and gate 2 (the full candidate) replayed the recorded walk. The deterministic set equals the baseline (ingest 2920/1148 and rejects, guard states, stage states). The robust pose median residual/radius was 0.033 (p2) and 0.071 (p2b), inside the gate of 0.14. Visual QA found the imagery the same. Verdict: PASS within noise.
3. **The 2x2 crossover (database by code).** Gate 2 showed unseeded "damage" (far-placed spans, tilted rooms) in 4 of 4 candidate runs but 1 of 5 baselines. To separate code from luck, the same databases were re-solved on both code versions. Far-out cameras occurred at equal rates (p2: 3/4 base vs 2/4 candidate; b4: 6/9 vs 6/9). Two runs on unchanged solve code were "damaged" too. Verdict: CODE-INDEPENDENT. Live variation is solver randomness that existed before (see §9.5).
4. **Suites against the baseline.** Tower full suite: 6,866 pass / 5 fail at integration (the identical baseline failure set), then 6,818 / 0, and 6,797 / 0 / 83 skipped at the end (`e20b3b2`; §8.1). iOS unit suite: 1,578/1,578 at the final batch gate (`af805b3`), with no flakes. UI suite at that gate: 67 pass / 20 fail / 3 skip plus 1 expected failure; beyond the expected red set (§8.7) every failure was a flake that passed in isolation or on rerun. `testEveryPanelStateAtAX5` went red to green.
5. **Characterization tests first.** For a refactor, tests pinning the old behaviour were written and run on base before the change: the TowerClient split (9), `geometryDidChange` (3), the ladder (a 3,528-case matrix), the refusal throttle (5 failing on base), the Scene race (4 failing on base). The unit gate script marks a Tower unit "SAME" when the test counts equal base.
6. **Reviewers as a net.** Review caught real regressions that targeted tests missed: R1 found a relocalizer golden recorded with the bridge off (the turn-bridge promotion); R1 found 4 publication tests that bypassed the guard (the scale-guard promotion); R2 rejected a `_Job.submitted_at` removal that dropped observable clock calls.

### 9.5 Known issues and bug-fix backlog (after the interview)

Marked **pre** if it existed before the campaign, **new** if the campaign introduced or exposed it.

**High / demo-visible**
- **AX5 capture start with the dial (pre, in the demo configuration).** At the largest text size with the capture dial on, `DeadEndsUITests/testTheLeavePromptShowsBothChoicesAtTheLargestTextSize` fails ("the capture started", `:567`); Start may be unreachable. The pre-cleanup base fails it with `IOS_FOW_DIAL=on` and passes without. Since the promotion the dial is always on, so every build has it (new exposure). Question: test geometry, or a real bug? The test stays red.
- **Map-face thumbnail sheet (pre).** With the dial on and the default-off Map tab on, tapping a landed photo thumbnail on the Map face does not open its detail sheet (`WorldImageryUITests/testTheLandedThumbnailSitsInItsPieceAndATapSaysDirectionAndAge`). The demo and plain Run builds cannot reach it. Kept as a scoped `XCTExpectFailure`.
- **Tilted rooms from far spans (pre).** If the solve places part of the walk far away, that geometry is fused into the surface and `surf.outliers.poses_gated` stays 0. The up refinement then fails and `_world_up` silently falls back to the camera up, so the phone shows a room tilted about 8-9 degrees, and the level-2 face budget goes on misplaced geometry (21% of faces within 3 m of the desk against 60-75% in good runs). Base code does it too; the refinement that fails is `surface_render.surface_up`.
- **Non-reproducible final solves (pre).** Each final solve re-runs nondeterministic, multithreaded matching, even with `TOWER_WORLD_SOLVE_SEED=0`, until a seeded solve freezes the matching. One walk yields a main component of roughly 600 to 880 images, sometimes with a far-placed span. A candidate fix would be freezing matching or seeding threads; not decided. The live product runs unseeded and multi-threaded (T4.6, §7.4).
- **WB idle buttons' contrast (pre, demo-visible).** On the World Builder idle screen, "Connect to Tower" nearly passes and "Tower settings" fails the contrast audit (ids `tower-recovery-connect`, `wb-capture-connect`, `wb-panel-connect`, `wb-panel-settings`, `tower-recovery-settings`). The headers "WHAT THE GLASSES SEE" and "WHAT THE TOWER BUILDS" clip at larger Dynamic Type (`AccessibilityAuditUITests/testWorldBuilderIdle`, also failing at baseline). Remaining `Color.secondary` text in `WorldCaptureDialView.swift` and `ShellPieces` needs the dial background checked first.

**Races and correctness**
- **Scene start/stop race (pre, fixed).** The operator-hold bug is fixed in the final tree (p5-01, §9.3): a stream-closure stop now releases the hold (`_stop_then_reconcile`, `scene/live.py:258`), and the hold epoch stops a demand stop overriding a newer operator start. It went through two rounds because the first fix raced. Still open, the related, lesser iOS items: unknown scene lifecycle states fall back to `.stopped` and show "Not observing", against the model's own rule; the first snapshot from a duplicate subscription that the client closes is still shown (low).
- **`IMURecorder.close(completion:)` (pre, latent).** On an already-closed recorder it never calls the completion.
- **`compatible_live_records` (pre).** It does not catch `UndistortionUnavailable`. The VRAM guard is off until the 10th prediction, because the first sample is taken then.
- **One solve-placement curiosity (pre, logged only).** Image 00003637 (8 observations) was placed in 0 of 4 base re-solves and 6 of 6 later-code re-solves of the same database. It is most likely process-level randomness: p1's solve path is byte-identical to base. It changes one weak pose.

**Flaky tests (pre)**
- IMU mock-device classes (`IMURecorderMockDeviceTests`, `MotionProbeMockDeviceTests`): all 5 fail together in about 1 of 4-5 full-suite runs, rarely when run alone; suspected test-ordering or shared state. The 60 Hz IMU timing test failed 2 of 10 on base (the Mock Device Kit sends a third motion error).
- A `TowerClient` loopback flake (3 reruns of 98 clean); a `tracemalloc` memory-bound test, which turned out to be a test-only thread leak (fixed); `WorldImageryUITests/testThePhotoFloorplanHolds...`, a pixel flake; and `AccessibilityAudit/testHome`, an XCUITest snapshot error that passed on rerun.

**Protocol and performance candidates**
- `frame_error` arrives once per frame at the send rate (about 15 Hz at the 15 fps default) whenever CV Lab is not running, including during World Builder walks. Is it contractual with no experiment armed? The phone side was throttled (K11); traffic is a P4 candidate. (pre)
- `FramePTSProbe` runs in every DEBUG build (the demo is DEBUG): an unbounded sample array walked every 5 s under a lock on DAT's frame thread. Measure first. (pre)
- Off-main frame encode (JPEG, base64, JSON): needs a design (window-check invariant, slot lifetime, failure release) and an on-phone measurement. Parked for hands-on device time. (pre)
- `coverWords` computes `presentation` twice (optional, measured P4).

**Unused and stale code (cleanup follow-ups)**
- `WorldReconstruction.ladder` takes a `finalSolve:` parameter it never reads. Remove it with its call sites.
- `surface_pipeline` accepts `known_fov` and `depth_known_fov` and ignores them. `ImageryState.tile`'s `live`/`landed` are always True in production. `guidance_coverage`'s `include_pending` is the off arm of a switch that no longer exists. Some DA3 backends are registered but unrunnable through `dense_pipeline`.
- Stale comments still open: `WorldBuilderWorkspaceView.swift:54` says segments are drawn "each in its own frame because the Tower has not registered them", likely false now; `SceneUnderstandingContract.phoneSendsNothingNote` (`SceneUnderstandingContract.swift:79`, test-only) says the session "follows the camera stream", but it now needs the stream and a watcher; `coherence_publish.py:570-571`, a failed-gate log line saying the final solve is published "with no components record" (the final scale guard still builds one). The root `README.md`'s stale frame rate and removed switch names are fixed (§11.B, item 2). Already fixed by the ci-21 comment sweep: `DocumentMemoryModel.swift:189-190` (the caps are fixed Tower constants), `ExperimentalCVClient.swift:32` (now cites `walkReportUpdates`) and `TowerReachabilityReader.swift:17-20` (15 Hz, "start capture in Home").
- `TowerConnectionRow` and similar spots mirror `CVTowerConnectionLabel.actionTitle(for:)` by design. `swift-structure-check.py` misreads raw strings.
- The Tower never tells the phone the capture id directly (the premise of `WorldSessionGate`); it does indirectly, in three places (§11.B, item 1).

**Decisions waiting for the owner** (two earlier ones are settled: final scale guard v1 was deleted, §9.3; `FINAL_IMMEDIATE` was kept and documented, §7.4)
- Persisted strings that embed review ids (`REVISIT_IMPORT_*`, `FROZEN_REFUSAL_LEGACY_IMPORT`) and stale log texts: changing them changes output.
- Multi-scene: an iPad multi-window would create a second `GlassesConnection` (fix = a build setting plus a device check). Note only.

**Process incident (not a product bug).** On 2026-10-16 about 08:20, a cleanup sub-agent killed the live Tower's uvicorn on :8000 with an unchecked `taskkill` while stopping its own pytest. Nothing was reconfigured or written. The owner restarted it. Every later brief carries the rule: kill only a PID your own command launched, never by name, never :8000.

---

## 10. Glossary

Terms in the order a reader meets them, grouped by area.

**The walk and the room**
- **Walk**: one recording pass through a room with the glasses on. Walk 10 is the reference walk (2,920 frames in 196.3 s, 1,148 keyframes).
- **Capture**: the live stream of camera frames from the phone to the Tower during a walk, bracketed by `stream_start` and `stream_stop` and started and stopped from the app, and the Tower's recording of it (`captures/<cid>/`, T1.5). Only frames go to the Tower; the DEBUG IMU recorder writes on the phone (§4.2).
- **Keyframe**: a frame the Tower keeps for reconstruction; chosen by `KeyframeSelector` (T3.3) out of all received frames.
- **Landing**: a background solve landing, that is, a newly published coverage block. Each landing gets its own B2 `landed` guidance imagery and its own photo floor plan, one tile per room component (`build_landing`, T5.6); the phone shows a landed photo in its piece of the map.
- **Landed photo / thumbnail**: a kept photo shown on the Map face at its place; tapping it should open a detail sheet (known bug, §9.5).
- **Coverage receipt**: the status's `guidance.coverage` block: which stations of each room component the solve has seen, and in which of 12 directions, from `compute_coverage` (T3.8). "Null" means no receipt is available (none yet, or none that qualifies).
- **Guidance**: the on-phone hints that steer the walker toward unseen areas, derived from the coverage receipt.
- **Capture dial**: the circular on-phone control (always on; K22) drawing twelve 30° directions for one reconstructed piece of the room, from the coverage receipt; it never claims to be complete or says where the wearer is. It is the walk-time guidance surface.
- **Fog of war (FOW)**: the map idea of hiding unseen areas and revealing them as the walk covers them. In the code the only map is the Map face on the walking stage, off by default (`IOS_FOW_MAP_TAB`, §4.4, §7.5). The owner's UX plan puts the map after the walk and drops a live bird's-eye view.
- **Map face**: the other face of the walking-stage panel, behind the Dial | Map control: a fog field per room piece with its lit directions, the photo floor plan and the landed photos (`WorldCoverageMapView.swift`). The Map tab is default-off (`IOS_FOW_MAP_TAB`, §7.5).
- **Native chrome**: the phone's own SwiftUI interface around the room view (`NATIVE_CHROME`; T7.6), replacing the HTML-looking one; now the default.
- **Appearance**: the colour and texture layer of a room, built by `build_appearance` (T5.4), applied over its shape.

**Components**
- **Glasses**: the Meta smart glasses; the phone talks to them through DAT.
- **DAT**: Meta's Device Access Toolkit, the native iOS SDK used to reach the glasses' camera and sensors.
- **iPhone app**: the Swift app that connects to the glasses, sends frames, and shows the room.
- **Tower**: the Windows PC (FastAPI/uvicorn on :8000) that receives frames, runs the CV/ML pipeline and stores worlds.
- **World Builder (WB)**: the main cartridge that turns a walk into a photographic 3D room, on both sides.
- **Cartridge**: a pluggable Tower-and-app feature module with its own wire contract (World Builder, Object Memory, Document Memory, Scene Understanding, CV Lab).
- **CV Lab**: the cartridge for running experimental computer-vision modules on the live stream.
- **Object Memory**: the cartridge that remembers objects seen.
- **Document Memory**: the cartridge that remembers documents and snippets; title and snippet caps are fixed Tower constants (60 and 48 characters).
- **Scene Understanding (Scene)**: the cartridge running live scene analysis such as a people detector, with a lifecycle and an operator start and stop.
- **Operator hold**: a Scene state set by an operator `POST /scene/start`, keeping the session running with nobody watching; an operator stop or, since p5-01, the stream-closure stop releases it (§9.3).
- **World**: a stored result for one walk (database, solve output, surface, imagery).
- **WorldStore**: the Tower storage layer for worlds, with writer locks and atomic writes (T6.1-T6.5).
- **Mock Device Kit**: DAT's simulated glasses used by the mock-device tests.

**Protocol and results**
- **frame_error**: a per-frame message from the Tower saying a frame was refused, with a reason; the phone throttles how often it redraws on it (K11).
- **Result channel**: the subscription protocol on the same `/ws` socket that carries the frames (§1, §6.2), on which the Tower pushes cartridge snapshots. A client subscribes by cartridge and result type; the Tower names each subscription with an id (`sub-1`, … per socket).
- **Snapshot**: a result message holding a cartridge's current state.
- **Subscription id / `belongs`**: how a result error is attributed to a cartridge (`CartridgeResultError.belongs`).
- **Send rate**: frames per second the phone sends; the default is now 15 (demo rate).
- **Contracts**: shared protocol truth under `docs/contracts/`.

**Reconstruction pipeline**
- **GLOMAP**: the global structure-from-motion mapper used for the solve; it re-matches and re-maps, with the nondeterminism described in §9.5 (T4.6).
- **Final solve**: the global solve run after Stop on the whole walk database (`solve`, T4.6); contrasts with the background solves run during the walk.
- **Background solve**: periodic solves during the walk by `BackgroundSolver` (T3.7).
- **Matching / database**: feature matching stored in the walk database; the final solve re-matches it unless frozen.
- **Main component**: the largest connected set of images the solve places together (c0); about 600-880 images on walk 10.
- **Seeded solve**: a solve with `TOWER_WORLD_SOLVE_SEED`; only reproducible once its matching is frozen.
- **Final scale guard (FSG)**: the check at publish that vets the solved room's scale and components (T4.8); v2 is the only assessor, since v1 `assess` was deleted (§9.3).
- **Up estimate / `_world_up`**: the room's up direction; it falls back to the camera up if the refinement fails, giving tilted rooms.
- **Surface pipeline**: the stages that turn depth and poses into the room surface (`surfacify`, T5.2).
- **Photographic pipeline / ladder (Tower)**: the stage sequence that produces a photographic room (depth, surface, appearance, dense).
- **Presentation ladder (iOS)**: `WorldReconstruction.ladder`, in `WorldPresentation.swift` (K18), the rule table choosing what the phone shows for each `WorldStage`.
- **Redaction**: filling each detected face box (YuNet) in a keyframe with a constant colour before the keyframe is stored (`FaceRedactor`, T3.4). `GUIDANCE_IMAGERY_UNREDACTED`, the one remaining switch, does not skip it: it lets the guidance imagery routes serve a session on which no redactor ran, labelled `unredacted` (§7.3).
- **Relocalizer**: the module that recovers tracking when a walk loses its place (`LookBackRelocalizer`, T3.5); off by default.
- **Turn bridge**: a keyframe-policy mode that bridges fast turns; now the default (T3.3).
- **Live depth**: depth predicted during the walk by `LiveDepthWorker` (T3.6); now always on.
- **IMU**: glasses gyro; blocked since DAT reports the sensor unavailable, so no design relies on it.

**Campaign and verification**
- **Switch promotion**: making a demo `TOWER_WORLD_*` or iOS switch the default and deleting its off path (§7.2).
- **Tier A / Tier B**: Tier A is byte-identical output; Tier B is the same end result within declared tolerances.
- **R1 / R2**: the two reviewers of a unit; R2 was Codex, then Claude when Codex ran out.
- **Characterization test**: a test written first to pin existing behaviour before a refactor.
- **Baseline**: the pre-cleanup tag 5fecd51, the reference for suites and replays.
- **Frozen-bs**: the frozen inputs of the walk-10 world `bs` (database, matching), used for exact byte comparisons.
- **Replay gate**: a gate that replays the recorded walk 10 through the pipeline and compares it to the baseline (§8.3).
- **Noise band**: the run-to-run spread of the baseline replays (for example, c0 of 719-750 images across b1-b3), widened by half its width; a candidate within it passes. "Better" also passes with visual QA and both reviewers.
- **Crossover (2x2)**: re-solving two databases on both code versions to separate code effect from database or solver randomness.
- **Visual QA / render pack**: a rendered image set of a built world, reviewed by an agent for orientation, placement, missing parts and regressions.
- **Test seam**: a member kept for tests only, documented as "Test seam".
- **P4**: the planned measured performance phase after the cleanup.

---

## 11. Appendix: notes, corrections, open questions

### 11.A Corrections to the source inventories

The Key Functions were built from two inventories, the Tower's `KEY-FUNCTIONS.md` (59 entries) and the iOS
`IOS-KEY-FUNCTIONS.md`, and checked against the code. These are the places where the inventories were wrong or incomplete;
the entries in §5 follow the code.

**Tower.** Every `def` line and cited call-site line was first checked at `af805b36` and then re-checked at `66414dc1`, where the
batch 20/21 and p3-28 docstrings had moved many of them; the lines in this doc are the final ones. The call sites checked include
`ws.py:401`, `world_build_session.py:2538`, `engine.py:1217`, `global_solve.py:1683`, `geometry.py:86/101/285`,
`results/__init__.py:152/240` and `publisher.py:656`.

Content corrections (batch 21 recorded the first seven in `KEY-FUNCTIONS.md` v2.1):

1. **T1.1 `websocket_endpoint`.** "Payload-level decode errors are answered, not fatal" is wrong. Malformed JSON, a non-object, or
   a non-string `type` is logged and ignored, with no reply. Only an unknown string `type` gets `protocol_error`. The `finally`
   also closes the result channels, arms the World Builder deferred stop on the last client (`_arm_world_builder_follow_up`), and
   tears down off the loop (`_tear_down_after_disconnect`).
2. **T3.5 `LookBackRelocalizer.note_lost`.** It omits that the relocalizer is off by default. With `TOWER_WORLD_RELOCALIZER=off` the
   engine builds none (`engine.py:1339-1345`), so walk 10 ran without it.
3. **T3.7 `BackgroundSolver.maybe_launch`.** "Every N" is 50 in production: `DEFAULT_SOLVE_EVERY` (`world_build_session.py:141`).
   `_world_build_spec` passes no `--solve-every`.
4. **T5.4 `build_appearance`.** "ASTC when available, else WebP" is wrong. Phone-tier chunks are always WebP, plus ASTC when
   available. Tower-tier keyframes are WebP only (`appearance_pipeline.py:616-626`).
5. **T6.1 `acquire_writer_lock`.** It omits the main outcome. A live holder that is another process raises `WorldLockedError`,
   which refuses a second writer. An unreadable lock is reclaimed after a grace period (`store.py:1049-1091`).
6. **T7.2 `ResultHub.poll_once`.** "Else send the heartbeat" is imprecise. `ConnectionChannel.offer` (`publisher.py:335`) re-offers
   when the revision changed *or* the 2.0 s heartbeat is due. A slow reader's slot coalesces. The poll runs every 0.5 s.
7. **T7.3 `make_snapshot_for`.** It lists "Object Memory results", but there is no Object Memory producer: that request falls to
   "no producer is wired for …". Document Memory offers `status` only; its library is HTTP.
8. **Folded entries.** `SightingTracker.observe` (into T8.1), `PageFinder.find` (T8.3), `Tracker.update` (T8.6) and
   `SceneLive.latest` (T7.3).

**iOS.**

1. **`WorldCoverageStatus.init(payload:)`: the one-liner is stale.** "without claiming a map for an older
   saved world" is the old reading, which the campaign corrected in the code (WorldCoverage.swift:26-61):
   - absent `guidance` or `coverage` means no selected session, or a malformed or incomplete payload;
   - `null` means no coverage receipt is available (none yet, or none that qualifies or fits);
   - a present but malformed block is refused.
2. **`TowerExperimentalCVClient.project(_:)` is `static`** (K26). It is a pure mapping, not an instance method.
3. **`WorldRenderViewerModel.followRevisions()` needs a nuance** (K20). "Leaving appearance texture following to the
   page" holds while the appearance is served. If the appearance is withdrawn and the page has not recovered
   after one poll, the model replaces the page itself (`appearanceFollow` → `.replaceThePage`).
4. **`TowerClient.handleFrameError(_:)` is filed under "Experimental CV Lab"** (K11). It lives in `TowerClient` and
   runs on every `frame_error`, including during ordinary walks with no experiment armed.
5. **Navigation trap, not an error:** `WorldBuilderViewModel` is defined in `WorldBuilderClient.swift`, next to
   the `WorldBuilderClient` protocol. The socket client is `TowerWorldBuilderClient` in its own file.
6. **Coverage gaps.** The inventory has no entries for the shell, the capture path (`GlassesConnection`,
   `FrameRateGate`), the core `TowerClient` paths (connect, reconnect, `sendFrame`, inbound dispatch), the World
   Builder socket client, the session controller, the web-view gate or the capture dial.
   This doc adds K1-K14, K21 and K22 for them.
7. **No line numbers in the inventory.** All of its paths and symbols were verified at `af805b36` (the iOS tree is
   unchanged at `66414dc1`); the lines are in this doc.

### 11.B Open questions and stale text

1. **Does the Tower ever tell the phone the capture id?** `WorldSessionGate` (`WorldSession.swift:151`, §4.4) assumes it does not:
   `stream_started` is unimplemented, so the gate binds on liveness, not id equality. **Answer: never directly; indirectly, yes.**
   **Not directly.** `stream_start` gets no reply, and no `stream_started` message exists. The id is minted in `_start_capture`
   (`ws.py:894`) and is only logged, handed to the supervisor, and handed to the in-process cartridges. So the gate's comment holds in
   that sense. **Indirectly, in three places.** (1) The `world_builder.status` payload's `session.capture_id`
   (`results/world_builder.py:1198`, via `_session_block` at L880). It appears only after the builder writes its session record. It
   names the capture the builder was started on (`args.follow_capture.name`, `world_build_session.py:2281`); a reconnect
   continuation is followed by the same builder and does not change it. (2) The rows of `GET /worlds`
   (`results/world_builder_library.py:731`). (3) `GET /health`, field `capture.capture_id`: the latest recorder's id
   (`routes/health.py:141`). **So** an id-equality gate would have to wait for the first status snapshot. It could never have the
   id at `stream_start`.
2. **Resolved (root docs): the root `README.md` matches the promoted defaults.** Its switch table has three rows, the demo configuration's switches left:
   `…_GUIDANCE_IMAGERY_UNREDACTED`, `IOS_FOW_MAP_TAB` and `IOS_TOWER_TARGET_FPS`, with their defaults; it gives the 15 fps default, and it no longer names `IOS_FOW_DIAL` or the 13 promoted Tower switches.
3. **Resolved (p3-28): `TOWER_WORLD_GUIDANCE_IMAGERY_UNREDACTED` is now in `tower/.env.example`,** commented out (off), with its
   privacy note. It is the one switch the demo still needs (§7.3), so the live `.env` must still set it to exactly `on`.
4. **Resolved: `TOWER_WORLD_FINAL_IMMEDIATE` stays.** It is read straight from `os.environ` by the builder
   (`world_build_session.py:2476`), not by `config.py`, and only exactly `on` turns it on. The owner kept it as a live experiment and
   P4 latency candidate (§9.3), and p3-28 documented it in `tower/.env.example`.
5. **Resolved (p3-28): the `cartridge_session.py` module docstring.** It used to say Scene and Document Memory each run "in its own
   process and tails a capture journal". It now says that only World Builder and Object Memory use `CartridgeSession`, and that Scene
   Understanding and Document Memory run inside the Tower's process (in-process `LiveSession`s built by
   `cartridge_runtime.build_live_cartridges`).
6. **`WorldReconstruction.ladder` takes `finalSolve:` but never reads it** (K18; also in §9.5). The doc comment says the
   final-solve sentence is drawn separately. Remove the parameter in the bug-fix phase, or document it as intentionally unused.
7. **Possibly stale comment** (also in §9.5). The `WorldBuilderWorkspaceView` doc (WorldBuilderWorkspaceView.swift:54) says segments
   are drawn "each in its own frame because the Tower has not registered them into a shared one". The geometry path now carries
   placement hashes and registration passes (`fetchSegments` comments). This looks stale; check it before the interview, in case it is
   quoted.
8. **IMU framing.** The Motion probe and recorder code is complete and tested on Mock Device Kit (K6). Real glasses report
   `sensorUnavailable`, and the work is parked (owner notes). Be ready to say why the code stays.
9. **Capture is DEBUG-only by design.** The demo runs a Debug build: DAT Developer Mode, `MetaAppID 0` (§7.6, §7.7). Be ready to
   explain why Release is a read-only viewer.
10. **Unmeasured frame-path cost.** FrameRateGate's comment estimates the per-frame main-thread cost at 15-25 ms and bandwidth at about
    2.5 Mbit/s at 15 fps; neither is measured. Moving the encode off the main thread is parked and needs device time (also in §9.5).
11. **Units.** Walk 10's scale is `uncertified`, so sizes like `voxel 0.0221` are in solve units, not certified metres. Say so if
    asked about absolute size.
