# Glasses

Glasses is a wearable vision system built around Meta Ray-Ban glasses, an iPhone app, and a Windows GPU **Tower**. The glasses send camera frames through Meta's Wearables DAT SDK to the phone; the phone sends selected frames to the Tower over a WebSocket. Its World Builder turns a walk through a space into a saved 3D world: it selects keyframes, tracks and solves camera poses, reconstructs surfaces, and adds appearance from the walk's photos. On the phone, the World Builder workspace can show a live, evidence-based fog-of-war view during capture and an inline 3D room viewer afterward. The same transport also serves Object Memory, Document Memory, Scene Understanding, and an Experimental CV Lab. These are separate cartridges with their own sessions and results, not stages of World Builder.

This document describes the Tower at `76943f7` (`world-builder/ux-tower-v1`) and the iOS app at `29c6d1d` (the iOS UX line), which are the two demo-proven source lines intended to replace the older top-level `main`. Optional features are called out by their switches; their presence in source does not mean they are enabled on every Tower or phone.

## Architecture at a glance

```text
Meta Ray-Ban glasses -- DAT camera --> iPhone (GlassesConnection)
                                          |  selected JPEG frames, stream/session
                                          |  controls, and result subscriptions
                                          v
                                   WebSocket /ws
                                          |
                                          v
                           Windows Tower (FastAPI, port 8000)
                           | frame validation + capture journal
                           | cartridge sessions + status pushes
                           | supervised World Builder child process
                                          |
                  HTTP /worlds, geometry, render, imagery assets
                                          v
                      iPhone World Builder panel and inline viewer

After Stop: final keyframes -> global camera solve -> optional evidence/
scale checks -> depth estimation -> surface mesh -> photographic appearance
             -> saved-world status, assets, and viewer revision
```

The WebSocket carries frames, frame replies, cartridge controls, and small status updates. The phone fetches larger geometry, saved-world listings, render pages, and imagery over HTTP. The Tower's web process supervises a separate `world_build_session.py` follower; reconstruction does not run in the frame handler. Interim keyframes and builds can update during the walk. After Stop, the follower completes the final pass, while an idle-time finisher can resume owed surface or appearance work. The viewer offers the best available render rung; a sparse or incomplete result is not presented as a finished photographic room.

## Getting started

### Windows Tower

Use Python 3.12 in `tower/` and run PowerShell from that directory. The repository's [Tower README](tower/README.md) and [setup script](tower/scripts/setup_tower.ps1) cover the base transport environment:

```powershell
cd tower
powershell -NoProfile -File scripts\setup_tower.ps1
```

The script creates `.venv`, installs `.[dev]`, and creates `.env` if absent; it does **not** install the GPU or reconstruction extras. For the GPU World Builder, install the CUDA wheels **before** extras so pip does not retain a CPU torch build. The versions and index below come from [pyproject.toml](tower/pyproject.toml); check driver compatibility on a different Windows host.

```powershell
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu132 "torch==2.13.0+cu132" "torchvision==0.28.0+cu132"
.venv\Scripts\python.exe -m pip install --no-deps "git+https://github.com/microsoft/MoGe.git@<PINNED_COMMIT>"
.venv\Scripts\python.exe -m pip install -e ".[dev,ml,sfm,dense,appearance,transients]"
```

Replace `<PINNED_COMMIT>` with a tested MoGe Git commit and record it for the host. The source declares `moge>=3.0` but does **not** record a Git revision; MoGe is Git-hosted, and `--no-deps` avoids its `numpy<2` declaration downgrading the Tower's NumPy. The `sfm` extra supplies `pycolmap` for the global solve; `dense` supplies the MoGe depth stack; `appearance` supplies surface/viewer dependencies; `transients` supplies segmentation dependencies. Optional `ocr`, `semantic`, and `scene` extras support other cartridges. Model weights may download on first use. See [calibration](tower/docs/CALIBRATION.md) for camera intrinsics; without matching intrinsics, the unposed backend can produce keyframes but no solved poses.

For a normal capture-backed World Builder, set these in `tower/.env` (paths are relative to `tower/`):

```dotenv
TOWER_CAPTURE_ROOT=data
TOWER_WORLD_ROOT=data/world_builder
```

`TOWER_CAPTURE_ROOT` arms the raw capture recorder, which starts writing only when a stream starts. It retains unredacted source frames; World Builder's derived keyframes have a separate redaction path. `TOWER_WORLD_ROOT` enables world storage and the result channel. The auto-attached follower uses the same root as the offline driver's `DEFAULT_ROOT`. Run `.venv\Scripts\python.exe scripts\world_builder_env_check.py` to inspect environment readiness, then start the server with:

```powershell
powershell -NoProfile -File scripts\start_tower.ps1
```

The script defaults to `0.0.0.0:8000`. Its manual equivalent is `.venv\Scripts\python.exe -m uvicorn tower.main:app --host 0.0.0.0 --port 8000 --env-file .env`, run from `tower/`. `TOWER_HOST` and `TOWER_PORT` are parsed settings but do **not** bind Uvicorn; use the script's `-BindHost`/`-Port` arguments or Uvicorn's flags. The Tower exposes `GET /health` for runtime and capture state.

### iOS app

Open [ios/Glasses.xcodeproj](ios/Glasses.xcodeproj) in Xcode and build scheme **Glasses**, configuration **Debug**, for an iPhone or a compatible simulator. The app entry configures the Meta Wearables SDK; physical glasses require the SDK's registration and camera permissions. Debug builds also expose mock-device and capture diagnostics in Developer Tools. The project currently sets an iOS 26.5 deployment target, so use an available matching SDK and destination.

From `ios/`, a simulator build can also be requested with `xcodebuild -project Glasses.xcodeproj -scheme Glasses -configuration Debug -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build`, replacing the destination with one installed locally.

Set the Tower's `host:port` in the app's **Settings → Tower** section, then fully relaunch the app: [TowerConfiguration.swift](ios/Glasses/TowerConfiguration.swift) resolves one address per process. In Debug, `GLASSES_TOWER_AUTHORITY` is a higher-priority launch-environment override. The phone uses `ws://<host>:<port>/ws` and `http://<host>:<port>`.

Fog-of-war imagery is separately gated on each side. Pass `IOS_FOW_ROOM_IMAGERY_LIVE=on` and/or `IOS_FOW_ROOM_IMAGERY_LANDED=on` as Xcode build settings (or Debug launch environment). The app reads those values from `Info.plist` at launch; every other spelling, including unset, is off. Matching Tower `TOWER_WORLD_GUIDANCE_IMAGERY_LIVE=on` or `..._LANDED=on` enables its routes. Landed imagery additionally needs `TOWER_WORLD_GUIDANCE_COVERAGE=on`. The coverage map, current-piece view, and photo strips present distinct evidence: a current segment is never silently placed into the solved room.

## Repository layout

| Path | Contents |
| --- | --- |
| [`ios/`](ios/) | Xcode project, Swift app, DAT and Tower clients, workspaces, unit/UI tests, iOS scripts and handoffs. |
| [`tower/`](tower/) | Python package, FastAPI server, cartridge and World Builder pipelines, PowerShell setup/start scripts, tests, models, and deeper operational docs. |
| [`docs/contracts/`](docs/contracts/) | Shared wire and artifact contracts for worlds, geometry, surfaces, appearance, components, iOS integration, and other cartridges. |
| [`docs/agent-handoffs/`](docs/agent-handoffs/) | Cross-lane integration notes, evidence, review findings, and current-state handoffs. |
| [`docs/world-builder-dense/`](docs/world-builder-dense/) | Dense reconstruction evidence, architecture, decisions, and operations. |
| [`.githooks/`](.githooks/) and [`CLAUDE.md`](CLAUDE.md) | Worktree/commit guard and subsystem ownership rules. |

## In depth: Tower

The Python package is under `tower/tower/`. These are the main entry points and boundaries:

| File or group | Role |
| --- | --- |
| [`main.py`](tower/tower/main.py) | Builds the FastAPI app, wires routes and cartridge sessions, attaches capture workers, and supervises idle finishing. It spawns World Builder as a child command. |
| [`config.py`](tower/tower/config.py) | Parses `TOWER_*` settings and optional World Builder feature switches. It also defines shared default roots. |
| [`routes/`](tower/tower/routes/) | `/ws` frame/control transport and result pushes; `/health`, `/cartridges`, session and CV Lab endpoints; HTTP geometry, render, guidance imagery, observation, document, and scene routes. |
| [`capture.py`](tower/tower/capture.py), [`capture_workers.py`](tower/tower/capture_workers.py), [`frames.py`](tower/tower/frames.py) | Capture journal, child-process supervision, and incoming frame parsing/validation. |
| [`results/`](tower/tower/results/) | Cartridge declarations, revisioned status envelopes, subscription/push publication, saved-world listings, World Builder state, render/geometry adapters, and other cartridge result readers. Status reads persisted facts; it does not perform reconstruction. |
| [`world_builder/engine.py`](tower/tower/world_builder/engine.py), [`frontend.py`](tower/tower/world_builder/frontend.py), [`keyframes.py`](tower/tower/world_builder/keyframes.py), [`backends/`](tower/tower/world_builder/backends/) | Session lifecycle, frame quality and keyframe selection, live tracking/segment geometry, and calibrated classical versus unposed backend choice. |
| [`world_builder/global_solve.py`](tower/tower/world_builder/global_solve.py), [`coherence_gate.py`](tower/tower/world_builder/coherence_gate.py), [`coherence_scale.py`](tower/tower/world_builder/coherence_scale.py), [`coherence_publish.py`](tower/tower/world_builder/coherence_publish.py) | Final/global camera solve, evidence and metric-level checks, component decisions, and safe publication. The evidence gate is optional. |
| [`world_builder/final_scale_guard.py`](tower/tower/world_builder/final_scale_guard.py), [`anchor_verify.py`](tower/tower/world_builder/anchor_verify.py), [`solve_masks.py`](tower/tower/world_builder/solve_masks.py) | Optional final publication guard, image/scale/motion verification, and transient masks for solving. Their defaults preserve the unguarded path. |
| [`world_builder/dense.py`](tower/tower/world_builder/dense.py), [`dense_pipeline.py`](tower/tower/world_builder/dense_pipeline.py), [`surface.py`](tower/tower/world_builder/surface.py), [`surface_pipeline.py`](tower/tower/world_builder/surface_pipeline.py) | MoGe depth aligned to solved cameras, checkpointed fusion/packing, evidence-weighted surface reconstruction, and artifact publication. Missing evidence remains a hole rather than an invented wall. |
| [`world_builder/appearance.py`](tower/tower/world_builder/appearance.py), [`appearance_pipeline.py`](tower/tower/world_builder/appearance_pipeline.py), [`transients.py`](tower/tower/world_builder/transients.py), [`redaction.py`](tower/tower/world_builder/redaction.py) | Redacted keyframe imagery, transient masks, photographic projection/encoding, and privacy/provenance decisions. |
| [`world_builder/render.py`](tower/tower/world_builder/render.py), [`surface_render.py`](tower/tower/world_builder/surface_render.py), [`appearance_render.py`](tower/tower/world_builder/appearance_render.py), [`native_chrome.py`](tower/tower/world_builder/native_chrome.py) | Render-rung selection, HTML viewers, mobile asset responses, and optional handoff of appearance-page chrome to native iOS UI. |
| [`world_builder/guidance_coverage.py`](tower/tower/world_builder/guidance_coverage.py), [`guidance_imagery.py`](tower/tower/world_builder/guidance_imagery.py) | Optional, dated solve-based coverage blocks and redacted live/landed thumbnail manifests and tiles. Coverage work runs off the status request path. |
| [`world_builder/store.py`](tower/tower/world_builder/store.py), [`records.py`](tower/tower/world_builder/records.py), [`schema.py`](tower/tower/world_builder/schema.py) | World/session storage, manifest validation, durable records, and artifact vocabulary. |
| [`scripts/`](tower/scripts/) | `world_build_session.py` follows a live capture or replays frames; `world_solve.py`, `world_densify.py`, `world_surface.py`, `world_appearance.py`, `world_finish_pending.py`, `world_refinish.py`, and `world_inspect.py` drive or diagnose individual stages. `setup_tower.ps1` and `start_tower.ps1` are the Windows entry points. |
| [`tests/`](tower/tests/) | Pytest coverage for transport, lifecycle, contracts, storage, solver, reconstruction, routes, and failure cases; some tests spawn real processes and carry the `slow` marker. |

Important settings below are defaults **in the specified Tower source**, not a report of any live host's `.env`. Most are parsed in `config.py`; coverage and guidance imagery read their exact `on` values at their result/route boundaries.

| Setting | Default | Effect |
| --- | --- | --- |
| `TOWER_CAPTURE_ROOT` / `TOWER_WORLD_ROOT` | unset / unset | Arm source capture / enable saved worlds and World Builder reporting. |
| `TOWER_WORLD_AUTOBUILD` / `TOWER_WORLD_REBUILD_EVERY` | `true` / `4` | Attach a follower for an active World Builder cartridge; set the base live rebuild cadence, which widens for long walks. |
| `TOWER_WORLD_REGISTER` / `TOWER_WORLD_SOLVE` | `true` / `true` | Attempt segment registration and a global solve. |
| `TOWER_WORLD_DENSIFY` / `TOWER_WORLD_SURFACE` / `TOWER_WORLD_APPEARANCE` | `false` / `true` / `true` | Select the optional dense rung and downstream surface/appearance finishing where inputs and dependencies exist. |
| `TOWER_WORLD_FINISH_PENDING` | `true` | Let the Tower run owed finishing work when idle. |
| `TOWER_WORLD_SOLVE_MASKS` / `TOWER_WORLD_SOLVE_GATE` | `false` / `false` | Apply transient masks and the final evidence gate; the gate needs masks to attach non-anchor pieces. |
| `TOWER_WORLD_SOLVE_SEED` / `TOWER_WORLD_SOLVE_CONSENSUS` | unset / `1` | Seeded single-thread solve and number of consensus draws (accepted: 1, 3, 5, 7). |
| `TOWER_WORLD_ANCHOR_VERIFY` / `TOWER_WORLD_POSE_QUARANTINE` | `off` / `off` | Optional verification and publication restrictions; each accepts named parts or `on`. |
| `TOWER_WORLD_FINAL_SCALE_GUARD` | `off` | `shadow` writes an audit; `on` enforces scale/pose publication checks. |
| `TOWER_WORLD_AREA_BUILDS` / `TOWER_WORLD_RELOCALIZER` | `false` / `off` | Build separated areas / enable look-back guidance (`prompt` or `silent`). |
| `TOWER_WORLD_GUIDANCE_COVERAGE` | off | Exact `on` adds solve-placed fog-of-war coverage to status. |
| `TOWER_WORLD_GUIDANCE_IMAGERY_LIVE` / `TOWER_WORLD_GUIDANCE_IMAGERY_LANDED` | off / off | Exact `on` exposes redacted guidance thumbnails; landed also requires coverage. |
| `TOWER_WORLD_FINISH_STAGES` / `TOWER_WORLD_PICTURE_BASIS` / `TOWER_WORLD_NATIVE_CHROME` | `false` / `false` / `false` | Optional processing detail, final-versus-preview render basis, and native viewer chrome. |

See [config.py](tower/tower/config.py) for the other CV, scene, document, object-memory, device, cadence, and logging controls. A switch can enable a pipeline stage only if its inputs, model dependencies, and stored evidence are available.

## In depth: iOS

| File or group | Role |
| --- | --- |
| [`GlassesApp.swift`](ios/Glasses/GlassesApp.swift), [`ContentView.swift`](ios/Glasses/ContentView.swift), [`ProjectManager.swift`](ios/Glasses/ProjectManager.swift) | Configure DAT, install the SwiftUI shell, and own app-lifetime glasses, stream, Tower, metrics, and cartridge clients. |
| [`GlassesConnection.swift`](ios/Glasses/GlassesConnection.swift), [`StreamManager.swift`](ios/Glasses/StreamManager.swift), [`FrameRateGate.swift`](ios/Glasses/FrameRateGate.swift), [`SendWindow.swift`](ios/Glasses/SendWindow.swift) | DAT registration/camera boundary, capture lifecycle, frame selection and bounded sending. Capture and mock controls are especially visible in Debug. |
| [`TowerClient.swift`](ios/Glasses/TowerClient.swift), [`TowerConfiguration.swift`](ios/Glasses/TowerConfiguration.swift) | WebSocket frame/control transport, status subscriptions, reconnection, and the process-wide HTTP/WS Tower address. |
| [`Workspaces/`](ios/Glasses/Workspaces/) | Home and separate World Builder, Object Memory, Document Memory, Scene Understanding, and CV Lab experiences. |
| [`WorldBuilderWorkspaceView.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldBuilderWorkspaceView.swift), [`WorldBuilderSessionController.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldBuilderSessionController.swift), [`TowerWorldBuilderClient.swift`](ios/Glasses/Workspaces/WorldBuilder/TowerWorldBuilderClient.swift), [`WorldModel.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldModel.swift) | World workspace, cartridge session lifecycle, status subscription/decoding, and presentation state. |
| [`WorldInlinePanel.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldInlinePanel.swift), [`WorldPanelPhase.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldPanelPhase.swift), [`WorldFinishProgress.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldFinishProgress.swift) | Inline panel and honest receiving, finishing, partial, and viewer states. The same viewer host moves between inline and full-screen presentation. |
| [`WorldCoverage.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldCoverage.swift), [`WorldCoverageMapView.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldCoverageMapView.swift), [`WorldCurrentPiece.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldCurrentPiece.swift) | Validate a dated coverage receipt, draw solved stations and fog field, and show the provisional segment-local camera footprints separately. |
| [`WorldImagery.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldImagery.swift), [`WorldImageryViews.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldImageryViews.swift) | Decode the optional live/landed imagery manifest and render redacted thumbnail strips/overlays without a disk tile cache. |
| [`WorldRenderViewer.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldRenderViewer.swift), [`WorldAssetTransport.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldAssetTransport.swift), [`WorldChromeBridge.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldChromeBridge.swift), [`WorldChromeOverlay.swift`](ios/Glasses/Workspaces/WorldBuilder/WorldChromeOverlay.swift) | Inline `WKWebView` room viewer, scoped asset proxy, page/native message bridge, and optional native chrome. |
| [`CaptureHealthModel.swift`](ios/Glasses/Workspaces/WorldBuilder/CaptureHealthModel.swift), [`CaptureHealthView.swift`](ios/Glasses/Workspaces/WorldBuilder/CaptureHealthView.swift) | Link, keyframe pace, recent tracking breaks, relocalizer, and map-lag health during a live walk; stale or absent data is not shown as current. |
| [`SettingsView.swift`](ios/Glasses/Views/SettingsView.swift), [`DeveloperToolsView.swift`](ios/Glasses/Views/DeveloperToolsView.swift) | Saved Tower address and read-only connection probe; Debug-only mock, sender, device, and capture diagnostics. |
| [`GlassesTests/`](ios/GlassesTests/), [`GlassesUITests/`](ios/GlassesUITests/) | Unit/contract tests and UI tests, including Tower settings, World Builder panel, coverage, imagery, current piece, viewer chrome, and mock-Tower flows. |

## Contracts and deeper docs

The shared [contracts](docs/contracts/) are the authority for wire identifiers and payloads: `TOWER-UNIFIED-CARTRIDGES.md` declares cartridges; `WORLD-BUILDER-IOS.md` covers the phone/Tower handshake; `WORLD-BUILDER-GEOMETRY.md`, `WORLD-BUILDER-DENSE.md`, `WORLD-BUILDER-SURFACE.md`, and `WORLD-BUILDER-APPEARANCE.md` define artifact and viewer capabilities; `WORLD-BUILDER-WORLDS.md` defines saved-world listing and render revisions; `WORLD-BUILDER-COMPONENTS.md` defines room and unplaced pieces. `OBJECT-MEMORY.md` covers that cartridge. [Tower-local contracts](tower/docs/contracts/) add cartridge results, CV Lab, and guidance imagery details.

Cross-subsystem handoffs and review history live in [`docs/agent-handoffs/`](docs/agent-handoffs/); Tower-specific operations, experiments, and timing notes live in [`tower/docs/`](tower/docs/). Some older handoffs describe earlier defaults or proposed work. Check the named source line and current contract before treating a historical result as a present runtime guarantee.

## Testing

From `tower/`, after installing `dev`, run the full Python suite with `.venv\Scripts\python.exe -m pytest`. Use `-m slow` to target process-spawning tests, or `-m "not slow"` for a faster local pass. `scripts/world_builder_env_check.py` checks the configured World Builder environment; `GET /health` checks a running Tower. This README change does not require a reconstruction run or access to a physical capture.

From `ios/`, use one available iOS Simulator destination at a time. On the 8 GB Mac, disable parallel test simulators:

```sh
xcodebuild -project Glasses.xcodeproj -scheme Glasses -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -parallel-testing-enabled NO test
```

Choose an installed simulator if that exact name is unavailable. `GlassesTests` and `GlassesUITests` are the project test targets. A simulator test covers app logic and mock flows; a physical DAT stream and Windows GPU reconstruction require their respective hardware and Tower configuration.

## Status and known issues

The live keyframe and final reconstruction path is implemented in these lines, but a saved world can legitimately be partial. Tracking breaks leave separate pieces; an uncalibrated resolution yields no poses; global scale can remain unknown; monocular depth disagreement creates holes or removes thin structures; mobile mesh levels lose detail. Redaction is best effort and can obscure useful pixels. The [reconstruction handoff](docs/agent-handoffs/WORLD-BUILDER-RECONSTRUCTION-FINAL.md) lists measured limits and open review items, including narrow stale-lock and interrupted-depth reuse cases.

The evidence gate, anchor verification (including its retrieval candidates), scale guard, pose quarantine, area builds, coverage/imagery, and native chrome are explicitly switched. Separate live-depth and mirror-masking experiments are outside the default path described by these pinned source lines; there is no production switch for them in this Tower's `config.py`. Do not infer that an experimental branch is active from a handoff. Check the Tower environment and the produced world manifest before interpreting a demo result.
