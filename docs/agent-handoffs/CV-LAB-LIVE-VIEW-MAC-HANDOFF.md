# CV Lab live view — Mac and iPhone validation

**Branch:** `feature/cv-lab-live-visualization-v1`
**Check out:** the tip of `feature/cv-lab-live-visualization-v1` (five
commits, `36e75ff` first)
**Written:** 2026-08-29, from Windows. **No Xcode ran. No iPhone ran.**

Everything below the line marked WINDOWS was verified by running it. Everything
below MAC and PHYSICAL was not, and this document does not claim otherwise.

---

## What changed, in one paragraph

Seven of the eight CV Lab experiments now hand the Lab the derived array they
had already computed, the Lab keeps exactly one of them, and
`GET /cv-lab/preview` encodes the newest one on demand and serves it as an
image. The phone draws it at the top of the workspace and moves the telemetry
into a collapsed Diagnostics section. `baseline` deliberately has no picture.
Nothing photographic crosses: every overlay is drawn over an edge map derived
from the frame, never over the frame.

---

## VERIFIED ON WINDOWS

| Claim | Evidence |
|---|---|
| Full Tower suite | `2320 passed, 36 skipped, 1 failed` in 6m17s — the one failure is `test_object_memory_lifecycle.py::test_an_unconfigured_tower_still_serves_its_own_memory`, which asserts an empty default observation root and finds the 116 real observations in `tower/data/object_memory/` from your physical testing. **Pre-existing and environmental.** It fails the same way on a clean checkout of this machine; nothing in this change touches Object Memory. Do not delete that data to make it pass. |
| The preview suite | `tests/test_cv_lab_preview.py` — 50 tests, all passing: bounded storage, latest-only semantics, three staleness guards, failure isolation, the depth normaliser, the HTTP surface. |
| Every visual experiment renders a decodable, non-blank image | `test_every_visual_experiment_renders_something_a_person_could_look_at`, parameterised over the registry. |
| Preview cost on the frame path | Measured, 120 frames per cell at 640x360, **throttle forced off so every frame captures** — i.e. worse than production. `preview` stage: edge 0.000 ms (the array already exists), frame_quality 0.174 ms, optical_flow 0.265 ms, feature_detection 0.736 ms, redaction_impact 1.066 ms. Whole-`run()` delta was within noise in both directions. |
| Preview cost off the frame path | Encode: edge 0.44 ms / 2.1 KB, frame_quality 1.29 ms / 5.0 KB, feature_detection 2.32 ms / 8.7 KB, optical_flow 2.37 ms / 9.1 KB, redaction_impact 2.94 ms / 9.1 KB. Depth measured separately at ~2.8 ms / ~16-28 KB JPEG. All of it on a worker thread. |
| Nothing is written to disk | `test_previews_are_written_to_no_file_anywhere` chdirs into a tmp dir, runs 40 render cycles, and asserts the directory tree is unchanged. |
| Storage is bounded | `test_one_capture_and_one_encoding_exist_however_many_frames_arrive` — 200 frames, then 50 more with renders, asserting one slot and one cached encoding throughout. |
| A consumer that never fetches costs nothing | `test_a_consumer_that_never_fetches_costs_no_encodes_at_all` — 100 frames, `encoded == 0`. |
| Object detection, end to end, on the real model | `device` resolved to **cuda**; load with warm-up 2 750 ms; run 44.0 ms first / 35.7 ms median / 37.5 ms last — **no 500 ms first-inference spike**; `preview` stage 0.356 ms against 36.0 ms of inference; two boxes drawn, 4.1 KB. On a synthetic desk scene, on a machine that was also running the test suite. |
| Every preview looked at, not just measured | One of each kind rendered to PNG/JPEG and inspected. Edge: continuous white lines, a doorway reads as a doorway. Depth: bright near, dark far. Flow: arrows agreeing on direction. Redaction: the magenta rectangle with red survivors inside it. Detections: a bold accepted box and a thin faded near-miss. Frame quality: a histogram with the clipping levels marked. |
| `pip check` | `No broken requirements found.` |
| `compileall` over `tower/` | Clean. There is no linter and no type checker configured in this repository — `pytest` is the gate. |

## REQUIRES MAC / XCODE VALIDATION

Everything in Swift. Six files changed and one added; none of it has been
compiled. Specifically unverified:

- that it compiles at all;
- `@MainActor` isolation and `Sendable` across the `nonisolated`
  `CVLivePreviewHTTPClient` → `@MainActor CVLivePreviewLoader` boundary;
- SwiftUI layout — the 240 pt preview frame, the `DisclosureGroup`, and
  whether the panel reads well at that size;
- that `ExperimentalCVPreview.swift` is picked up by the target. It should be:
  the app target uses `PBXFileSystemSynchronizedRootGroup` rooted at
  `Glasses/`, so a new file under it needs no project edit. The **test**
  target is not synchronized, which is why the new tests were added to the
  existing `CVLabContractTests.swift` rather than to a new file.

## REQUIRES PHYSICAL GLASSES / IPHONE VALIDATION

Everything about whether it feels live, and both object-detection findings.

---

## The procedure

### 0. Get it

```bash
cd ~/…/Glasses
git fetch origin
git checkout feature/cv-lab-live-visualization-v1
git log --oneline -6        # oldest of the five should be 36e75ff
```

Open `ios/Glasses.xcodeproj`, build for your device. **If it does not compile,
fix the error and note what it was** — that is the single most useful thing
this handoff can learn.

### 1. Start the Tower

```powershell
cd C:\Users\tvllo\Projects\Glasses\tower
.venv\Scripts\python.exe -m uvicorn tower.main:app --host 0.0.0.0 --port 8000
```

Before touching the phone, check the new surface from the machine itself:

```powershell
curl http://127.0.0.1:8000/cv-lab/preview/status
```

With nothing running this answers `artifact: null` and a reason. That is
correct: `baseline` is the startup default and has no picture.

### 2. Edge Detection — the one that should be unmistakable

1. Open Experimental CV Lab. Start **Edge detection**.
2. **The picture is the first thing on the screen**, above the latest-result
   panel and above the experiment list.
3. Move your head. Furniture, doorways, monitor bezels and your own hands
   should appear as white lines on black and should move with you.
4. Watch the `LIVE` badge and the freshness text under the picture. It should
   read `Live` while streaming, not a growing "N.Ns behind".
5. Open **Diagnostics**. `Previews captured`, `Previews skipped by the
   throttle` and `Previews encoded` should all be moving, with `encoded` far
   lower than `captured` — that gap is the design.
6. Compare `Processing` against your recorded baseline of **~1 ms/frame**.
   Expect no meaningful change: this experiment's preview costs nothing on the
   frame path because the array already exists.

### 3. Pause, resume, stop — the staleness cases

1. **Pause.** The picture must disappear immediately and be replaced by a
   sentence. A frozen last frame under a "Paused" label would be the bug.
2. **Resume.** A new picture, not the old one.
3. **Stop.** No picture. The figures stay and say they are final.
4. **The important one:** stop Edge Detection and start **Monocular depth**
   immediately. **No edge map may appear under Depth's name at any point.**
   The Tower refuses a request naming the old run with `409` and the phone
   also checks the run on every arriving frame, so this should be impossible
   twice over — confirm it anyway.

### 4. Monocular depth

1. Start it. The arm may take up to two minutes on a cold weight cache.
2. Bright is nearer, dark is farther. Hold a hand up close: it should be the
   brightest thing in frame.
3. **Stand still and watch for flicker.** The scale is the 2nd/98th percentile
   of each frame, smoothed with a five-frame EMA, so a still scene should sit
   still. A pulsing wall means the normaliser is not working and is worth
   reporting with a video.
4. Walk through near / mid / far geometry. The picture should track the change
   within about half a second rather than snapping.
5. **Read the caption under the picture.** It must say the values are not
   metres. If any surface anywhere implies metres or feet, that is a blocker.
6. Compare against your baseline: decode 0.671, preprocess 3.937, **inference
   9.399**, postprocess 0.256, total 14.263 ms. The preview adds nothing to
   the frame path here — depth's array already exists — so `inference` and
   `total` should be unchanged. `Preview render` in Diagnostics is a separate
   figure, on a separate thread.

### 5. Feature detection

1. Start it. Green dots over a line drawing, with a faint 8x8 grid and the
   occupied cells tinted.
2. Point at a **blank wall**: few dots, few tinted cells, `coverage` low.
3. Point at a **cluttered desk**: dots across the frame, most cells tinted.
4. The caption says `N keypoints (M drawn)`. `M` is capped near 128 by design
   — a thousand markers on a phone-sized panel is a grey rectangle.
5. Baseline was ~995 keypoints, coverage ~0.57, ~6.3 ms. Expect the same
   numbers plus roughly 0.7 ms on the frames that captured a preview.

### 6. Frame quality

1. Start it. A line drawing with a luminance histogram along the bottom, and
   two vertical markers at the clipping levels.
2. **Point at a bright window**: the histogram should pile up against the
   right marker and `clipped % bright` should rise.
3. **Turn the lights down**: it should pile against the left.
4. **Move fast**: sharpness should drop.
5. **Read what the app says about sharpness.** It should place the value
   inside *this run's* observed range — "near the high end of this run's range
   so far (79 to 1309 over 767 frames)" — and must **never** say "Good",
   "Blurry" or any other verdict. These metrics have no calibrated threshold on
   this camera and both sides refuse to invent one. If you see a verdict, that
   is a blocker.

### 7. Object detection — the two findings to confirm

1. Start it. **Check `device` in Diagnostics: it should now say `cuda`.**
2. Compare the timings against your baseline of **~199 ms mean, 4,483 ms worst
   frame, ~3.67 fps**. The expectation is roughly **40-50 ms/frame** and no
   multi-second stall, because the warm-up now happens during the arm.
   - If it is still ~199 ms on CUDA, the diagnosis was wrong and the pin
     should go back. Say so and paste the numbers.
3. Put a **laptop**, a **phone**, a **chair**, a **bottle** and a **keyboard**
   in view. Boxes with class names and scores should appear over the line
   drawing.
4. **Now the person question.** Go back to the scene that produced 160 and 281
   `person` detections. Watch where the box lands.
   - The Windows-side investigation says these are **your own hands, forearms
     and torso**: 87% of the sampled boxes touch the bottom edge, are centred,
     and score around 0.66; this repository's own oracle study independently
     confirmed 81% of shipped `person` boxes against a stronger model. Nothing
     is wrong with the class map, the colour order or the normalisation — all
     three were tested directly and ruled out.
   - **What to confirm physically:** does the box land on your own arm? Put
     your hands behind your back and see whether the count stops climbing.
   - Faded, thinner boxes are detections *below* the threshold the metrics
     use. They are drawn on purpose, so a near-miss is visible rather than
     hidden.

### 8. Optical flow — not previously validated at all

1. Start it. Arrows coloured by direction, over the line drawing, with red
   dots where the forward-backward check rejected a track.
2. **Hold still**: few arrows, short.
3. **Pan left**: every arrow should point the same way and share a colour.
4. **Tilt**, then **walk forward**: forward motion should look like arrows
   radiating outward from the centre.
5. **Move fast**: expect more red dots — that is the check working, not
   failing.

### 9. Redaction impact — not previously validated at all

1. Start it. A magenta rectangle in the centre, an amber boundary band, and
   dots.
2. **Red dots inside the magenta rectangle are the finding**: texture the blur
   did not destroy. Amber dots on the boundary are artefacts of the blur edge,
   not scene texture.
3. Point at something highly textured and watch how many survive.
4. Nothing about the redaction algorithm changed. If this looks alarming, it
   is reporting something that was already true.

### 10. General robustness

1. **Rapid switching**: start Edge, immediately Depth, immediately Feature
   detection, three times. No picture may ever appear under the wrong
   experiment's name.
2. **Navigate away and back** while a run is going. The picture should stop
   and restart, and Diagnostics' `Previews captured` should keep climbing
   (the Tower is still capturing; the phone stopped fetching).
3. **Disconnect the Tower** (kill uvicorn) for ten seconds and restart it. The
   panel should say something rather than freeze, and recover.
4. **Background and foreground the app** during a run.
5. **Twenty start/stop cycles.** Watch the Tower's memory. It must not climb.

### 11. Read the counters

```powershell
curl http://127.0.0.1:8000/cv-lab | python -m json.tool
```

`run.preview` answers the question this feature has to survive:

- `captured` vs `skipped_by_throttle` — visualisation running at its own rate;
- `replaced_unread` — frames dropped because a newer one arrived, which is
  intended;
- `encoded` — far below `captured`, because nothing is encoded until asked;
- `render_ms`, `payload_bytes` — what a picture costs, on a worker thread;
- `not_modified` — the phone polling faster than the Tower produces, which
  costs a round trip and no CPU.

Compare `timings.processing_ms` against your recorded baselines. It is
deliberately NOT mixed with `preview.render_ms`.

---

## Acceptance

**Pass** if, for every experiment:

- the picture updates as you move, and feels live rather than slideshow-like;
- `timings.processing_ms` is within noise of the recorded baseline (except
  object detection, which should be **much** faster);
- no picture ever appears under the wrong experiment;
- pause and stop remove the picture immediately;
- depth is never labelled in metres and frame quality never renders a verdict;
- twenty start/stop cycles leave Tower memory flat.

**Report back** with: the object-detection device and timings; whether the
`person` boxes land on your own limbs; any Swift compile error; and any case
where a picture disagreed with the numbers beside it.

---

## Switching it off

```powershell
$env:TOWER_CV_PREVIEW = "false"
```

The frame path returns to exactly what it was — no capture, no derivation, no
`preview` stage in the timings. That is how to re-measure a baseline.

`TOWER_CV_PREVIEW_MAX_EDGE_PX` (default 320) and
`TOWER_CV_PREVIEW_MIN_INTERVAL_S` (default 0.05, a 20 Hz ceiling) are the two
other knobs.

---

## What three independent reviews found

Run after the first commit, on the code rather than on the description:
performance and architecture, CV correctness, and iOS/state/privacy. **No
blockers.** Seven findings, all seven fixed in `81d69ac`. Two were real bugs
and both came from the reviewers who ran the code:

1. **A lost write in the depth scale.** Two concurrent renders of one fresh
   capture both folded the same bounds and one update vanished. Cosmetic — one
   frame at a slightly wrong brightness — but real, and the "everything is an
   immutable handover so no lock is needed" argument genuinely did not cover
   it. The normaliser has its own lock now.
2. **A caption that contradicted the metric beside it.** The detection preview
   drew the top 24 boxes and captioned itself "24 over 0.40" next to
   `detections: 160`. Now "24 of 160 over 0.40 drawn".
3. Sub-pixel optical-flow jitter was drawn as five-pixel arrows, making a still
   room look full of independently moving things. Below half a pixel it is a
   dot.
4. A flat depth frame rendered solid black, which in INFERNO means "very far".
   Mid-grey now.
5. An empty depth array raised `IndexError` where the NaN path had a sentence.
6. The histogram's bars and clipping markers truncated independently and landed
   up to a pixel apart.
7. The iOS panel did not drop its picture on backgrounding, so iOS's
   app-switcher snapshot could capture it. It watches `scenePhase` now.

Two more came from rendering the previews and looking at them, which is the
check this whole feature exists to give a person:

8. The detection view drew twenty-three refusals scoring 0.02-0.03 beneath one
   accepted box, and their label chips covered the room. A near-miss got at
   least halfway to the threshold; below that the detector had already said no.
   The same scene now draws two boxes and 8.3 KB became 4.1 KB.
9. `X-CV-Preview-Age` does not tick, so a phone getting `304 Not Modified` from
   a Tower whose stream had stopped kept a `LIVE` badge over a frozen picture
   for two seconds. The panel adds its own wait to the Tower's figure now.

While rendering the detection case end to end against the real model on this
machine: `device` resolved to **cuda**, the load-time warm-up worked (first
frame 44 ms, median 36 ms, **no 500 ms first-inference spike**), and the
`preview` stage cost 0.36 ms against 36 ms of inference. That is the change to
compare against your 199 ms / 4 483 ms baseline.

What the reviews **confirmed**, with measurement rather than agreement:

- Storage is bounded: 5,000 frames with the consumer never fetching added
  ~4.8 MB in the first 1,000 and then flatlined for the next 9,000
  (192.85 → 192.77 MB). 200 start/stop cycles: RSS −0.07 MB.
- The throttle is real: 216–235 of every 240 offered frames were skipped when
  driven as fast as the CPU allows.
- Coordinates are correct in all four overlay kinds, verified by putting a
  known marker at a known position and finding the drawn pixel.
- `prediction["boxes"]` really is in the original frame's coordinates —
  confirmed against torchvision's `GeneralizedRCNNTransform.postprocess`.
- The depth EMA reduces frame-to-frame brightness variance **190×** against
  MiDaS's own per-frame min/max (23.0 vs 2096.6).
- INFERNO polarity is right: near reads bright (luminance 243.7), far reads
  dark (0.46).
- The edge downscale preserves **95.4%** of connected components against
  INTER_NEAREST's 74.6%; a single diagonal line stays one component instead of
  shattering into 81.
- Nothing is written to disk on this path, and no `.pbxproj` edit is needed.

---

## Known and deliberate

- **No photographic background.** Overlays sit on an edge map. A face-filtered
  real frame was considered and rejected: it needs vendored YuNet weights, so
  a Tower without them would show a blank debug viewer; a display filter is
  not a redaction; and an edge map is enough to tell a chair from a doorway.
  If a photographic background is ever wanted, the filter and its fail-closed
  behaviour already exist in `tower/object_memory/imagery.py` and this is the
  place that would call them.
- **`baseline` has no picture.** It is the control every other experiment's
  cost is read against.
- **An edge map is not anonymous.** It keeps a jawline, a hairline and a
  silhouette. That is exactly why the treatment is `raw_ephemeral` and not
  `redacted`, and why nothing persists it on either side.
