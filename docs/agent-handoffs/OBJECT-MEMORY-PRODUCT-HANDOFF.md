# Object Memory: from a backend with buttons to a cartridge

**Date:** 2026-08-29
**Branch:** `product/object-memory-self-contained-v1`
**Worktree:** `C:\Users\tvllo\Projects\Glasses-worktrees\object-memory-product`
**Lane:** cross-stack. Both `ios/` and `tower/` were changed, under an
explicit authorisation that overrides `CLAUDE.md`'s default
one-subsystem rule and `LANE-OWNERSHIP.md` §1.1.

---

## 0. What this was

Object Memory's backend worked. It detected, it persisted, it served, and
the iOS cartridge displayed real records with correct timestamps. What did
not work was the product around it. Using it physically required:

1. setting `TOWER_OBSERVATION_VERIFIER` and
   `TOWER_OBSERVATION_VERIFIER_DEVICE` in the shell,
2. launching Tower,
3. leaving the Object Memory cartridge,
4. starting the camera from Home,
5. `POST`ing the session endpoint from PowerShell,
6. returning to the cartridge,
7. stopping the session by hand afterwards.

Seven steps, five of which are implementation details, to press one
button. The cartridge had a Start button the whole time; it started the
Tower's *producer* and nothing else, so pressing it gave a session that
was honestly `active` with `attached_capture_id: null`, no frames, no
records, and no sentence on screen explaining that the camera lived
behind a differently-named button on a different screen.

**Why it ended up like that, and it was not an oversight.**
`LANE-OWNERSHIP.md` §1.1 froze `ios/` to a lane with a Swift toolchain,
and `OBJECT-MEMORY-HANDOFF.md` §7 opens "Nothing under `ios/` was
touched. These are requirements, not patches." The Tower half shipped
alone *by construction*. The iOS lane then implemented the Tower's
session verbs faithfully — Start/Pause/Resume/Stop, liveness from
`following`, the 409-is-an-answer rule, all of it — and stopped exactly
where its instructions stopped. Nobody owned the sentence "and the
camera".

---

## 1. The composed lifecycle

### 1.1 One owner, three callers

There is still exactly **one** camera pipeline and one object that talks
to DAT: `GlassesConnection`. What changed is that a third screen may now
reach it. `startCameraSession()` / `stopCameraSession()` call sites:

| Screen | Before | After |
|---|---|---|
| Home | yes | yes, unchanged |
| World Builder | yes | yes, unchanged |
| Object Memory | no | **yes, through `ObjectMemoryRecordingCoordinator`** |

`ContentView`'s long comment about "the four workspaces below receive no
`glasses`" has been rewritten to say three, why, and which invariant
actually still holds.

The coordinator is owned by **`ProjectManager`**, not by the workspace
view. That is the rule `CartridgeClients` was created for, and this is
the first object in the app for which it is not a precaution: a
workspace `@StateObject` is destroyed the moment somebody opens Home to
look at something, and a coordinator rebuilt on their return would
believe it had started nothing — so its Stop would leave the glasses
recording.

### 1.2 Start: the Tower first

```
POST /cartridges/object_memory/session/start     (idempotent; opens the gate)
      ↓  refused / unsupported / unreachable → stop here, touch no camera
camera.captureClaim
      ├─ .unclaimed    → startCameraSession(), claim ownership
      ├─ .devicePaused → say so; there is nothing to override
      └─ .running      → leave it alone, claim NOTHING
      ↓
converge: re-read the session until `following_this_session` is non-empty,
          bounded by a deadline
      ↓
remembering            (or `notObserved`, said plainly, never "remembering")
```

**The order is not interchangeable.** The Tower attaches a producer to
captures created while the session is active; a capture opened first
finds the gate closed and nothing attaches. Worse, a producer attached
afterwards runs `--attach-mode from-now` and never reads back the seconds
in between. `OBJECT-MEMORY.md` §9.3 already documented this order as the
supported one: *"Start before the camera is normal. The session goes
`active` with `attached_capture_id: null` and the next capture to open
finds the gate open."*

### 1.3 Stop: the Tower first, and only what we started

```
POST .../session/stop            (never refused; detaches the producer,
                                  which now flushes — see §3)
      ↓
stopCameraSession()              ONLY if this screen started the capture
      ↓
refresh the records
```

Stopping first closes the gate, so a reconnect during teardown — which is
common; see §2 — cannot re-attach a producer to a walk the person just
ended.

### 1.4 Pause and Resume

Tower-only, and the copy says so. **This app has no call that pauses a
DAT stream** — `GlassesConnection` calls exactly `session.start()`,
`session.stop()`, `stream.start()` and `camera.stop()`, and none of them
does. That is a statement about this app, not about the SDK: whether DAT
exposes one somewhere was not established, and nothing here claims it
does not. So Pause means "stop remembering", the camera keeps streaming
if it is running, and the screen says both things rather than letting
"paused" be read as "the camera stopped".

### 1.5 The hardware pause, which was a dead button

`DeviceSessionState.paused` is device-initiated — a temple press, a
thermal pause — and `05-DAT-INTEGRATION.md` §104-107 records that it
keeps the connection alive, stops delivery, and **resumes to `.started`
on its own**.

`isCaptureEngaged` excluded `.paused`, so during one both Home and World
Builder flipped their primary control back to "Start capture", and a tap
hit `guard deviceSession == nil else { return }` — a **silent no-op
offered as the primary action**. `isCaptureSessionClaimed` had the right
reading and was consulted only by Developer Tools.

New `CaptureClaim { unclaimed, running, devicePaused, ending }` tells the
four apart, and `CaptureStartRefusal` makes every `return` in
`startCameraSession()` machine-readable. **No override was invented,
because there is none to invent** — Object Memory reports the pause and
waits. Home and World Builder are behaviourally untouched.

---

## 2. The imagery defect, and it was not retention

### 2.1 What the evidence actually said

The premise going in was that capture-side retention had removed the
frames. It had not, and it cannot: **this repository has no capture
pruner at all.** `CaptureRecorder.purge()` has zero production callers,
there is no `TOWER_CAPTURE_RETENTION*` setting, and three separate audits
already record that `data/captures/` is never pruned.

Every frame was on disk. The route was answering **410 "the memory is
kept, the picture is gone" about pictures one directory away.**

### 2.2 The mechanism

iOS reconnects about half a second after a WiFi hiccup and re-sends
`stream_start`. `CaptureRecorder` mints a **new** capture directory whose
manifest names the old one in `continues_capture`, and `CaptureFollower`
walks the producer straight into it without missing a frame. The
producer's `session_id`, though, is `--follow-capture <dir>`'s basename,
fixed once at spawn.

So a sighting that opened before the hiccup and had its **strongest look
after it** carries the old capture's id beside the new capture's frame.
`frame_path` looked only in `captures/<session_id>/`, found nothing, and
reported expiry.

Measured on the development host's live store:

| | before | after |
|---|---|---|
| records | 116 | 116 |
| resolve in their own capture | 90 | 90 |
| resolve only along the reconnect lineage | **0 — these were the 410s** | **26** |
| answering 410 with the frame on disk | **26 (22.4%)** | **0** |

An earlier version of this table said 94 and 22. Those were real numbers
from a store that was still being written into a few hours earlier, left
stale rather than re-measured, and an independent reviewer caught them.
The figures above come from re-running the shipped resolution code over
the store as it stands: 116 records, 71 capture directories, 32 lineage
roots.

### 2.3 The fix, and the bound that was wrong

`frame_path` now searches the record's own capture first — the ordinary
case, which pays nothing extra — and then walks the reconnect lineage.

The first version carried `MAX_LINEAGE_STEPS = 8`, justified as "chains
of five are in this corpus". That was a guess, and checking it against
the corpus rather than remembering it found **one walk eighteen captures
long**: one sitting, seventeen reconnects, source frames 1 to 8,690. The
bound silently truncated it and four records in that walk still lost
their pictures.

The constant was the wrong *shape*, not the wrong number, and it was
deleted rather than raised. `seen` is what terminates the walk: each step
consumes one edge of a finite map over directories that exist, and no
node is entered twice, so a self-referencing or cyclic manifest stops on
the first step. A test now builds a 25-long chain.

### 2.4 The 410 can no longer hide a pointer bug

The wire is unchanged — a wearer is told the same true thing either way,
and there is nothing they can do differently. The **log** now
distinguishes them, via `capture_lineage_present`:

- *the recording is no longer on disk* → INFO, retention doing its job;
- *the recording is still on disk without this frame* → **WARNING**,
  naming it as a pointer defect.

This class of bug spent two days indistinguishable from expiry. It
cannot again.

---

## 2A. Object Memory now owns one picture per memory

§2 fixed the bug. This fixes the *shape* the bug was possible in.

A record is kept for 30 days. The frame it points at lives in
`data/captures/`, which this cartridge does not own, did not write, and
must not modify. Until now that survived only by accident — there is no
capture pruner, so nothing had ever actually expired. **Any** pruner, or
one human reclaiming disk, would have taken the picture off every memory
at once.

### 2A.1 What is owned

One small crop per observation, at
`<observation_root>/keyframes/<observation_id>.jpg`, written when the
sighting closes, with a required JSON sidecar beside it.

`/crop` prefers it and falls back to cropping the capture frame.
`/frame` is unchanged and still comes from the recording — a keyframe
*is* a crop and there is nothing to synthesise a context view out of, so
when the recording is gone the whole-frame view is honestly 410 and the
object's picture is not.

`prune_expired` and `purge` delete keyframes with their records, and the
`ObservationStore` **builds its own** `KeyframeStore` from its own root
rather than being handed one — there is no caller who can forget, and
therefore no configuration in which records are pruned while their
pictures survive.

`imagery_retention` is no longer a constant: it says `object-memory` when
the owned crop served the bytes and `capture-side` when the capture frame
did, with `imagery_source` naming which. `filter_means` follows the same
split — `applied-before-this-file-was-written` for a keyframe, and the
old `applied-on-read-the-stored-frame-is-unchanged` for a frame. **The
`object_memory.imagery/2026-08-27` identifier does not move**: a shipped
iOS build compares it for equality, and every added field is additive.

### 2A.2 Fail closed, and where the boundary is

`KeyframeStore.write` has no path by which the input crop reaches disk.
No filter, a filter that raises, a filter that returns nothing, an encode
failure, a write failure — every one of them writes **nothing at all**
and returns a reason. Everything that can reach the file is derived from
`face_filter.apply`'s output, and the input crop is never referenced
after that call.

The sidecar is part of the artifact, not metadata about it: an image
whose sidecar is missing is **ignored rather than served**, and a sidecar
that fails to write takes its image back off disk. So the sidecar's
presence is what makes the claim checkable.

**One promise did change, and it is written into the contract rather than
left as a divergence.** §10.3 said "a Tower whose weights are missing
serves nothing". That still holds for a capture frame, whose filter runs
on read. It does not hold for a keyframe, whose filter ran *before the
file existed* — refusing to serve one because the ONNX weights went
missing afterwards would withhold a picture on a check already passed,
more thoroughly, at write time. The corollary is the safe one: weights
missing at write time means **no keyframe exists**, and the request falls
through to the capture path and is refused there.

### 2A.3 What it costs, measured

Not estimated. The shipped `KeyframeStore` was run over all 116 real
records in this host's store, cropping the real frames they point at:

| | |
|---|---|
| crops written | 116 of 116 |
| smallest | 2.5 KB |
| median | 10.9 KB |
| mean | 11.7 KB |
| largest | 22.6 KB |
| **whole store, images + sidecars** | **1.3 MiB** |
| `data/captures/` frames on the same host | **568 MiB** |

Long side capped at 384 px, quality 80. The owned pictures for every
memory this host holds cost **0.23%** of what the recordings they came
from cost (1.31 MiB against 568.1 MiB) — and unlike the recordings, they are bounded by a retention
window that is actually enforced.

### 2A.3b The whole chain, run once, offline

Not a unit test and not a claim: one run of the real thing, on this host,
with nothing mocked. Real producer subprocess with the real ssdlite320
detector and the real YuNet weights, over **300 frames copied out of the
corpus**, then the real ASGI app and the real routes.

```
  device                 cuda          <- resolved from `auto`, in the producer
  verifier               none
  keep_imagery           True
  keyframe_filter        display-filter/yunet-2023mar@0.30
  frames_observed        300
  observations_recorded  3
  keyframes_written      3
  keyframes_refused      {}
  records over HTTP      3
  DELETED the whole capture tree
  708a8dd61de4c5a6  available=True source=object-memory-keyframe retention=object-memory frame_available=False crop=200 (13943B) frame=410
  db8a3a0daef2420b  available=True source=object-memory-keyframe retention=object-memory frame_available=False crop=200 (19245B) frame=410
  44b4e7495b539bb1  available=True source=object-memory-keyframe retention=object-memory frame_available=False crop=200 (10927B) frame=410
```

Every memory kept its picture after its recording was deleted, said
truthfully which store served it and whose retention governs it, said
truthfully that the wider context view was gone, and logged the honest
INFO — *"the recording behind a cell phone record is no longer on disk"*
— rather than the WARNING that names a pointer defect.

That is the §2A claim end to end, and the only part of it a Mac is
needed for is the button that starts it.

### 2A.4 What was deliberately not done

`data/captures/` retention was **not** extended, and no pruner was added.
Extending raw first-person capture retention to make a thumbnail survive
would have been the wrong trade in exactly the direction
`06-PRIVACY-DATA.md` warns about. The recording keeps whatever lifetime
its owner eventually gives it; the memory no longer depends on it.

`--purge-all` now prints what it removed **and what it could not**, and
exits non-zero when anything survived. A purge that prints a count while
a directory of crops is still on disk is the false claim of deletion
`CARTRIDGE-GROUNDWORK.md` calls worse than an honest failure, and this is
the command a wearer's erasure request actually runs.

---

## 3. Stop now finalises the memory

`POST .../stop` reached `CaptureWorkerSupervisor.detach` →
`Popen.terminate()`. On Windows that is `TerminateProcess`: no
unwinding, no `finally`, no `atexit`. The producer's
`finally: engine.release()` — the only code that closes the sightings
still open, writes the ones that matured, and refreshes the duration,
`frame_count` and `best_*` of the ones already written — **never ran.**

The bias is the bad part. The sighting still open when a wearer stops
walking is, by construction, the object they had been looking at longest.
Stop was discarding the best-observed memories and keeping the rest.

`DETACH_GRACE_SECONDS` was `0.0`, and **that was correct at the time**:
its comment recorded that the previous five seconds measured 5.01 every
time and bought nothing, because nothing ever *asked* the producer to
stop. The grace was waiting on a request nobody had made.

Three pieces:

1. `_StopRequest` in `scripts/object_memory_session.py` turns a stop into
   a flag the poll loop reads.
2. `capture_workers._ask_to_stop` makes the request **before**
   terminating.
3. `DETACH_GRACE_SECONDS = 3.0`, because there is finally something for
   it to wait for. Bounded: it blocks a synchronous HTTP handler, and the
   iOS client's own timeout is ten seconds.

### 3.1 Two channels, because the obvious one does not work here

The obvious channel is a signal, and on POSIX `terminate()` is `SIGTERM`
and that is the whole story. On Windows the only catchable route is a
console control event, which is why `_start` already passed
`CREATE_NEW_PROCESS_GROUP`.

**A first version of this section said the event is never delivered under
a pseudoconsole. That was wrong, and a reviewer proved it.** Under
`GetConsoleWindow() == 0` the event *is* delivered, and a child that
installed no handler dies of it with `STATUS_CONTROL_C_EXIT` and no
unwinding — which is the more dangerous half of the truth, because it
means a signal sent to the wrong worker destroys exactly the grace it was
meant to protect. That is §3.3.

The stdin channel is still first, for reasons that survive the
correction: a pipe needs no console at all, so it works where an event
genuinely cannot be delivered; it is unambiguous, where a control event's
effect depends on what the child did with its handlers; and closing it
costs nothing when the child is already gone. Two independent channels
for a request that must not be missed, neither required to succeed.

So the first channel is **closing the child's stdin**. The supervisor
holds the write end for any spec that sets `WorkerSpec.stop_via_stdin`;
closing it is an EOF a reader thread cannot miss; a pipe needs no
console, no signal disposition and no permission. Both are attempted and
each is contained separately.

It fixes something adjacent for free: a producer orphaned by a Tower that
died now gets its EOF from the operating system instead of polling for
its full fifteen-minute idle bound.

### 3.3 Only workers that opted in are asked, and that nearly went wrong

The first version asked **every** worker. A `CTRL_BREAK_EVENT` reaches a
child that installed no handler as "die now" — so on a host with a real
console this would have signalled the **world builder** at every
shutdown, and its ten-second grace, which exists precisely so a follower
can finish the build that releases its writer lock, would have become an
instant kill. The result channel would then have reported a world
`failed` for a build seconds from finishing.

**And the reason it looked fine was itself wrong.** The commit that fixed
this said the damage was invisible here because a pseudoconsole swallows
the event. It does not: a reviewer measured the event being delivered
under `GetConsoleWindow() == 0`, with a handler-less child dying of
`STATUS_CONTROL_C_EXIT` and no unwinding. So the World Builder would have
been killed on this host too, at t=0 of a ten-second grace, and the
regression was not latent — it was live and simply had not been run into
yet. `scripts/world_build_session.py` has no signal handler and no
`try/finally`: its `stop_session()` and final `build()` are on the
normal-exit path only.

`_Worker.handles_stop_request` is copied off `WorkerSpec.stop_via_stdin`
at spawn and gates both channels. Object Memory's producer opted in;
World Builder's did not and is signalled by nothing. A test pins it.

### 3.2 The check has to be inside the poll loop

The first version wrapped the frame generator and checked between yields.
That only gets control back when a frame arrives — and a wearer who puts
the glasses on a desk and presses Stop is in the quietest stretch there
is. It did not stop at all. `CaptureFollower.follow()` now takes
`should_stop`, asked once per poll, before the journal read.

**Measured after the fix, two ways.** Closing the pipe on an idle
producer: exit 0 within **0.27 s**, full report printed. And the whole
production path — a real `CaptureWorkerSupervisor.detach(grace=3.0)`
against a real producer following a real 120-frame capture, which is
`_ask_to_stop`'s *both* channels rather than the pipe alone:

```
alive after 12 s: True
detach() stopped 1 worker(s) in 0.85 s
exit code: 0
  stopped_because        SIGBREAK
  frames_observed        120
  observations_recorded  2
  keyframes_written      2
  keyframes_refused      {}
```

`stopped_because: SIGBREAK` is the correction in §3.1 showing up in
practice: the signal *is* delivered here, the producer's handler catches
it, and the flush runs. A producer without that handler would simply have
died — which is why §3.3 exists.

---

## 4. Liveness belongs to the session claiming it

`OBJECT-MEMORY.md` §9.1 carried this as a live warning:

> `following` and `captures` are supervisor-scoped, not session-scoped.
> Start a **new** session and it will report the OLD session's capture
> under the new `session_id`, having attached nothing. Under this defect
> that rule produces a **false positive**: a brand-new session that
> attached nothing renders as recording.

Reachable by the ordinary route: `_stop_worker` deliberately leaves a
worker it could not kill in the registry, so it stays visible to
`/health` and to the next shutdown.

**`following` was not narrowed.** Narrowing it would hide the un-killable
producer, and an un-killable producer is what "the Stop button failed
open" looks like — the worst thing this cartridge can do to a person and
the one thing that must never become silent.

The payload gains **`following_this_session`**: the subset started at or
after this session last went active. A client draws "remembering" from
it, the contradiction alarm from `following`, and a separate sentence
about the difference — a producer that is recording and that this
session's Stop will not reach.

Scoped by **start time**, not by ids the session kept: a capture that
opens while the gate is open is spawned without consulting the session at
all, and that is the *normal* path.

Both readings come from `CaptureWorkerSupervisor.mark()`. The first
implementation compared `_Worker.started_at` (`time.monotonic`) against
`CartridgeSession._started_at` (`time.time`) — an uptime against a Unix
epoch. Every worker looked older than every session and the
correct-looking code reported nothing at all. `mark()` exists so a caller
cannot make that mistake.

Degrading: a supervisor that cannot answer a scoped question — no
`mark()`, or a `following()` that refuses the `since` keyword — makes
`following_this_session` **null**, not empty. A false success is worse
than no answer, and `following` still carries everything.

`[]` and `null` are different claims and the difference is the whole
point: `[]` says "I started nothing that is running", which a client may
draw a warning from, and `null` says "I cannot tell you", which it may
not. An earlier draft answered `[]` for both, and a reviewer showed that
one of the two fallback paths still returned the *unscoped* list under
the scoped field — which would have raised the loud
something-else-is-recording alarm about a recording the person started
themselves.

---

## 5. Configuration

### 5.1 The ruling that was reserved, and who closed it

`OBJECT-MEMORY-HANDOFF.md` §7.4 recorded turning the verifier on by
default as an open question **for a human, not an agent**: *"the default
stays `none` because 94 crops from one home justify building it and not
switching it on for everybody."*

A human closed it. This pass was instructed that OWLv2 is the project's
intended standard configuration and that setting it by hand before every
launch is not acceptable for ordinary use.

| Variable | was | now |
|---|---|---|
| `TOWER_OBSERVATION_VERIFIER` | `none` | **`owlv2`** |
| `TOWER_OBSERVATION_VERIFIER_DEVICE` | `cuda` | **`auto`** |
| `TOWER_OBSERVATION_DEVICE` | `cpu` | **`auto`** |

`recorded_classes` therefore goes from 2 to 14 out of the box.

### 5.2 Why it is safe to *default* rather than merely offer

A host that cannot load the weights is not broken by it. `_build_verifier`
catches the failure, says so on stderr, and continues with no verifier —
narrowing back to the two classes the detector is trusted on. The report
carries **`verifier` (what ran)** beside **`verifier_requested` (what was
asked for)**, so a run cannot claim a tier it did not have. An explicitly
empty variable still means `none`: a person switching it off is obeyed
rather than defaulted over.

### 5.3 `auto`, and where it resolves

`auto` is the word `TOWER_CV_DEVICE` already uses, with the same rule
from `cartridge_runtime._resolve_device`: **auto downgrades, cuda does
not.** One deliberate difference — in the producer an explicit `cuda`
that cannot be honoured *also* downgrades, at WARNING, because raising
there costs a walk somebody started from their phone rather than an
experiment.

It resolves **in the producer**, which is the process that imports torch.
Tower's startup log says so and points at the producer's first line,
rather than logging a word that is not a device.

### 5.4 What startup says now

```
[Tower][Config] observation root <...> (one path for the producer AND the read
  routes; the web process never writes or deletes observations). Producer
  device auto, retention 30.0 days, verifier owlv2 on auto -- recording
  laptop, cell phone, remote, mouse, cup, bottle, keyboard, backpack,
  handbag, suitcase, book, umbrella, scissors, toothbrush.
[Tower][Config] a device above reads 'auto': it is resolved by the producer,
  which prints the device it actually got as its first line when a session
  starts.
[Tower][Config] TOWER_OBSERVATION_VERIFIER is unset, so the built-in default
  'owlv2' is in force. ...
```

`scripts/start_tower.ps1` gained an **Object Memory** block that resolves
and prints every `TOWER_OBSERVATION_*` value with where it came from, and
warns — in Object Memory's own terms — when the capture root is unset
(imagery answers 503) or the YuNet weights are missing (no picture is
served at all). `tower/.env.example` is new and tracked: the values are a
property of one machine, the list is a property of the project.

`start_tower.ps1 -CheckOnly` is new. It prints the whole configuration
block and exits, binding nothing and touching no data — so "what will
this Tower record, and can it show me a picture of it" is answerable
before a physical run without taking the port and having to stop
something afterwards. Running it is also what caught a mangled path in
the block itself, which no amount of reading would have.

**On this host no `.env` edit is needed.** The existing
`tower/.env` sets `TOWER_CAPTURE_ROOT`, `TOWER_WORLD_ROOT` and
`TOWER_DEV_MODE` and nothing else, so the new defaults apply on the next
launch with no change: verifier `owlv2`, both devices `auto`. That was
checked, not assumed.

### 5.5 Session actions are now in the log

`CartridgeSession.apply` logged only exceptions, so pressing Start on a
phone left no trace in the Tower console and physically testing this
cartridge meant a human reciting values off a screen. One INFO line per
action and per refusal, carrying `state`, `changed`, `session_id`,
`attached_capture_id` and `following`. Four verbs, human button presses,
and the frame path does not come through here.

---

## 5A. What review changed

Two independent reviewers, neither of which wrote the code it read.

### 5A.1 The iOS review — thirteen findings, one blocking

**The blocker.** Every control on the Object Memory screen, *including
Stop*, was disabled for up to twelve seconds after Start. `isActing`
covered the whole Start sequence, and that sequence ends with the bounded
convergence poll — so during the window in which the Tower session is
open and the camera this screen just started is streaming, the one
control a wearer has over being recorded was greyed out and the only way
out was to leave for Home. Which is the exact thing this change exists to
delete.

`isActing` now covers only the mutating step. A second flag covers the
whole sequence and is what the pushed-reading guard uses — necessary,
because the workspace's watch loop polls at the same cadence and would
otherwise turn the legal "accepted, nothing attached yet" into
`notObserved` seconds before the deadline that word belongs to.

**Reachability.** A camera-permission refusal is set by a DAT callback
*after* `startCameraSession()` returns, so the coordinator read `nil`,
claimed ownership, and reported "asked to remember, and not observed"
twelve seconds later — while the true answer was known and had a written
sentence that would never be shown.

**Claims that outran the facts.** "The camera is sending frames" was
printed from the instant of the tap, over a session that had not
connected; `CaptureClaim.running` includes `DeviceSessionState.starting`.
Three doc comments proved what this app *calls* and claimed it as what
the SDK *exposes*. A new copy line said "Restarting the Tower will" about
a producer that had already ignored `terminate()`.

**A test that passed for the wrong reason.** New copy contained a
forbidden phrase, and the forbidden-phrase test missed it only because
its fixture never populated the field that produced the line. Both were
fixed — the wording and the fixture.

**Two prominent Starts and two differently-labelled Stops** on one
screen, both routing to the same call.

**Dead code with a comment claiming otherwise.** `deinit { work?.cancel() }`
could never fire while work was outstanding, because the task held `self`
strongly — which also made the cancellation branches in `converge()`
unreachable. The blocker's fix made cancellation real, so they are live
now and the comment says which thing reaches them.

**And a rule nothing enforced.** "This view writes no prose" was stated
three times in doc comments and checked by nothing. It is now
`swift-structure-check.py --no-prose`.

### 5A.1b The Tower review — two blockers and a false claim of my own

**Two invariants this branch stated in bold and did not hold.**

`KeyframeStore.write` promised "no image, no sidecar, no partial pair".
`json.dumps` sat inside a `try` that caught `OSError` only, *after* the
image had been written — so a `filter_label` that would not serialise
raised `TypeError` out of `write()` and left an **unattributable
first-person crop on disk** for the retention window. `read()` refuses to
serve such a file, which is why nothing ever leaked; the file was there
anyway and the docstring said it could not be. The sidecar is now
serialised before the image is written, so the only step left after the
image lands is one `write_text` of bytes that already exist.

`_write_keyframe` promised "never raises". `len(filled)`,
`encoded.tobytes()` and `_obscured_fraction` all sat outside the encode
guard, so a filter returning an unexpected *shape* — a generator, an int,
a box of the wrong arity; the reviewer reproduced all three — raised
straight through. The caller is a producer inside its frame loop, so that
escape ended the walk and took `engine.release()` with it: **one bad
keyframe would have cost every sighting still open**, which is precisely
the failure §3 exists to prevent, arriving by a different door. Both the
function and its call site are now contained, and five parametrised cases
pin it.

**A claim of mine that was simply false.** §3.1 said a console control
event is never delivered under a pseudoconsole. The reviewer measured the
opposite and was right: it *is* delivered, and a child with no handler
dies of it with `STATUS_CONTROL_C_EXIT`. That correction makes §3.3 more
important rather than less — and it is why the whole production path was
then re-run end to end rather than argued about (see §3.2's second
measurement).

**The scoped-liveness fallback still widened.** `_following` retries
unscoped on a `TypeError`, which is right for the public `following`
field; `_following_this_session` returned that widened list *as* the
scoped one. A client would have been told "these are the producers you
started" about every producer on the Tower — the exact false positive the
field was added to remove. `_scoped_following` now refuses instead of
guessing, which is what `_supervisor_mark` had always done beside it.

**Two documents were left describing a Tower that no longer exists.** The
contract still said the verifier default was `none` in three places,
including a note warning clients that hard-coding two class names was
safe. And `TOWER_OBSERVATION_KEEP_IMAGERY` — the single switch governing
whether this cartridge persists first-person pixels at all — was
documented **nowhere**: not the contract, not `.env.example` whose stated
purpose is to make the list discoverable, not the launcher that prints
every other `TOWER_OBSERVATION_*`. All three now carry it, and the
launcher prints it and the retention window too.

**Numbers presented as measured that were stale.** The resolution table
said 94/22; re-running the shipped code says **90/26**. The keyframe cost
was quoted from a 26-record subset; over all 116 it is a mean of 11.7 KB
rather than 13.9. Both were real measurements left behind by a store that
kept growing, which is the least excusable kind of wrong number, and the
corrected ones are in §2.2 and §2A.3.

**And the mutation testing.** The reviewer backed up each file, broke one
behaviour, ran the targeted tests, restored and verified by hash. Seven
mutations, seven caught: dropping the lineage fallback, not unlinking an
image whose sidecar failed, serving an image with no sidecar, waiting
without asking, not pruning keyframes, moving the keyframe lookup after
the filter guard, and never building a crop. The tests prove behaviour
rather than mocking it.

### 5A.2 The break the two lanes nearly shipped past each other

Not a review finding — found while applying one, and the most valuable
thing in this section.

The Tower gained a second `filter_means` value, because a picture
filtered *before it was written* is not a picture filtered on read and
must not claim to be. The shipped iOS decoder compared that field against
a **single** constant. Every keyframe payload would have failed the parse
outright, and the whole owned-picture feature would have been invisible
on the phone — with both suites green, because neither can see the other.

`ios/scripts/cross-stack-constants-check.py` is the check that would have
caught it, and it needs no Mac and no running Tower: it imports the
Tower's own constants and greps the Swift for them. It is a tripwire for
a value that moved on one side only, which is the failure that actually
happens when two lanes ship the same day. Verified against a deliberately
injected skew.

### 5A.3 Two things I found by running rather than reading

**The lineage bound.** `MAX_LINEAGE_STEPS = 8` was written from "chains
of five are in this corpus". Checking the corpus instead of the memory of
it found an eighteen-capture chain and four records still losing their
pictures inside the fix that was supposed to save them. See §2.3.

**The world builder's grace, nearly destroyed.** The stop request went to
every worker, and a `CTRL_BREAK_EVENT` kills a child that installed no
handler. See §3.3. It looked fine here for the worst possible reason — a
pseudoconsole swallows the event — so the damage was invisible on the
only machine it was tested on.

---

## 6. What is proven, and what is not

**Proven by test, offline: the Tower half — §2, §2A, §3, §4, §5.** 2369
tests, plus the numbers in §2.2 and §2A.3, which were measured by running
the shipped code against this host's real store rather than asserted.

**§1 is the iOS half and is NOT proven.** It is reviewed, structure
checked, and cross-checked against the Tower's constants; it has not been
compiled. Every claim in §1 about what a screen does is a claim about
source that no compiler has read.

**Proven by the physical runs that already happened**, read out of
`data/object_memory/observations.jsonl` rather than from a report:

- persistence across sessions — 116 records;
- **the semantic verifier ran and agreed**, twice, on this host:
  `keyboard` (`owlv2-base-patch16-ensemble`, label *computer keyboard*,
  0.685, `ranked-first`) and `mouse` (*computer mouse*, 0.944). Both
  `agrees: true`.

**NOT proven:** the verifier's *reject* path. No record in the store
carries `agrees: false`, so "the second opinion refuses a wrong label"
remains a software-tested claim only.

**NOT proven:** anything on iOS. There is no Swift toolchain on the
development host — no `swift`, `swiftc`, `xcodebuild`, SwiftLint or
swift-format. Nothing in `ios/` has been compiled or run.
`ios/scripts/swift-structure-check.py` is new and proves only that
brackets and strings balance in all 73 `.swift` files. **A green run
there is not a build.**

With `--no-prose` it also enforces the rule that the Object Memory
workspace view writes no user-facing sentences of its own. That rule was
stated three times in doc comments and enforced by nothing until an
independent reviewer pointed it out; the sentence it exists to stop is
one that escapes `ObjectMemoryCopyTests` entirely.

```
.venv\Scripts\python.exe ios\scripts\swift-structure-check.py --no-prose ^
    ios\Glasses\Workspaces\ObjectMemory\ObjectMemoryWorkspaceView.swift
```

It flags any string literal containing a space outside a comment, which
catches a sentence and ignores an SF Symbol name. It is deliberately
pointed at named files rather than the whole tree: World Builder's and
Home's views hold prose legitimately, and a checker that shouted at them
would be turned off within a week.

---

### 6.1 Re-running every check on this host

From the worktree's own `tower/` directory — see §7 for why that matters:

```powershell
$py = "C:\Users\tvllo\Projects\Glasses\tower\.venv\Scripts\python.exe"

# The whole suite. The short basetemp is deliberate: a deep one trips
# MAX_PATH once World Builder nests two UUIDs under tmp_path, and
# manufactures about 139 phantom failures.
& $py -m pytest -q -p no:cacheprovider --basetemp=C:\Users\tvllo\AppData\Local\Temp\gf

# Just this change's own suites.
& $py -m pytest tests\test_object_memory_imagery.py `
                tests\test_object_memory_keyframes.py `
                tests\test_object_memory_graceful_stop.py `
                tests\test_cartridge_session_liveness.py `
                -q -p no:cacheprovider --basetemp=C:\Users\tvllo\AppData\Local\Temp\gf

& $py -m pip check
```

From the repository root:

```powershell
# Brackets and strings balance in every Swift file. NOT a build.
& $py ios\scripts\swift-structure-check.py `
    (Get-ChildItem -Recurse ios\Glasses,ios\GlassesTests -Filter *.swift).FullName

# The workspace view writes no sentences of its own.
& $py ios\scripts\swift-structure-check.py --no-prose `
    ios\Glasses\Workspaces\ObjectMemory\ObjectMemoryWorkspaceView.swift

# The Swift names every constant and wire key the Tower sends. This is
# the one that would have caught the filter_means split; see §5A.2.
& $py ios\scripts\cross-stack-constants-check.py
```

And the resolved configuration, without starting anything:

```powershell
cd tower; .\scripts\start_tower.ps1 -CheckOnly
```

## 6A. The next physical test

One run. It exercises the composed button, the verifier tier that has
never been deliberately targeted, pause/resume, the flush on Stop, and
the owned picture.

**Before you start**, on the Mac: build and install this branch. Nothing
under `ios/` in it has ever been compiled.

### The run

1. **Start Tower with no environment setup at all.**

   ```powershell
   cd C:\Users\tvllo\Projects\Glasses\tower
   .\scripts\start_tower.ps1
   ```

   Do **not** set `TOWER_OBSERVATION_VERIFIER` or
   `TOWER_OBSERVATION_VERIFIER_DEVICE`. That is the thing being tested.
   (`.\scripts\start_tower.ps1 -CheckOnly` prints the configuration and
   exits, if you want to read it before binding a port.)

   **Expect in the console**, before it binds:

   ```
   Object Memory
     TOWER_OBSERVATION_ROOT     <unset, defaults to data\object_memory>
     TOWER_OBSERVATION_VERIFIER <unset, defaults to owlv2>
       ...VERIFIER_DEVICE       <unset, defaults to auto>
     TOWER_OBSERVATION_DEVICE   <unset, defaults to auto>
   ```

   and then, from the app:

   ```
   [Tower][Config] ... verifier owlv2 on auto -- recording laptop, cell
     phone, remote, mouse, cup, bottle, keyboard, backpack, handbag,
     suitcase, book, umbrella, scissors, toothbrush.
   [Tower][Config] an object-memory producer will be attached to each
     capture WHILE A SESSION IS ACTIVE.
   ```

   **Fourteen classes, not two.** Two means the verifier did not load,
   and the producer will say why on its first line.

2. **On the phone: open Object Memory. Do not visit Home.** Going to Home
   first would test the old workflow.

3. **Press Start remembering.** Once.

   **Expect in the Tower console:**

   ```
   [Tower][Session] object_memory start -> state=active changed=True
     session=<hex> attached=None following=[]
   [Tower][Worker] started object-memory-session pid <n> for capture <id>
   [Tower][ObjectMemory] detector=ssdlite320 device=cuda verifier=owlv2
     verifier_device=cuda root=...
   ```

   `attached=None` on the Start line is **correct** — the gate opens
   before the camera does. The worker line follows when the phone's
   capture opens.

   **Expect on the phone:** the panel moves *starting* → *waiting* →
   **remembering**, and the record dot fills only at the last step. If it
   sits on "asked to remember, and not observed", the producer did not
   attach and the Tower console will say why.

4. **Walk for about two minutes with these in view**, a few seconds each:

   | Object | Tier | What it proves |
   |---|---|---|
   | a **laptop** | remembered | the detector-only path, and the class the last run produced |
   | a **cell phone** | remembered | the same |
   | a **bottle** | **verify** | **the semantic verifier accepting** |
   | a **cup** | **verify** | the same, second class |
   | a **book** or **backpack** | **verify** | the same, third class |

   Hold each still for **at least three seconds** — a sighting needs
   three frames to mature, and the gate at `min_score = 0.5` drops a
   glance.

   **The bottle, cup and book are the point of this run.** The last
   physical test produced only `laptop` and `cell phone`, which are the
   only two classes that bypass the verifier entirely, so the second
   opinion was configured and never exercised. Two records in the store
   now carry real OWLv2 verdicts (`keyboard`, `mouse`), so *acceptance*
   has since happened — but **no record anywhere carries `agrees:
   false`**, so the verifier *rejecting* a wrong label is still unproven
   on hardware.

   For the reject path, keep a **laptop open in view**. The corpus
   measurement found a laptop keyboard detected as `remote` at 0.87, and
   `remote` is a verify-tier class — so OWLv2 should be asked and should
   refuse. A `declined: {unverified: N}` in the final report with N > 0 is
   that path firing.

5. **Press Pause.** Expect `[Tower][Session] object_memory pause ->
   state=paused` and the producer to disappear from the process table.
   The phone should say paused **and**, if the camera is still running,
   say that too — the two are separate facts.

6. **Press Resume**, walk another twenty seconds, then **press Stop.**

   **Expect in the Tower console**, from the producer, at Stop:

   ```
   stopped_because        stdin-closed
   frames_observed        <a few thousand>
   observations_recorded  <> 0
   verifier               owlv2
   verifier_requested     owlv2
   verification           {"requested": N, "accepted": ..., "rejected": ...}
   keyframes_written      <equal to observations_recorded>
   keyframes_refused      {}
   ```

   **`stopped_because: stdin-closed` and a printed report at all** are
   the flush working. Before this branch, Stop killed the producer
   outright and it printed nothing — and every sighting still open, which
   is every object you were looking at when you pressed the button, was
   discarded.

   **`keyframes_refused` must be empty.** Anything in it means the
   display filter refused, and the reason is the key.

7. **Look at the new records.** Tap one and show its picture.

   **Expect a picture**, not "the memory is kept, the picture is gone".
   Then tap through to the whole frame and back.

### What would count as a failure

- Two recordable classes instead of fourteen at startup.
- The phone saying "remembering" while the Tower console shows no worker.
- A `[Tower][ObjectMemory] ... POINTER defect rather than retention`
  WARNING — that is this branch's own diagnostic firing, and it means a
  record points at a frame in a recording that is still on disk. It
  should not happen any more.
- `keyframes_refused` non-empty.
- Any picture that appears without the display filter having run — the
  payload's `filter` field must read `display-filter/yunet-2023mar@0.30`.

### The one thing this run cannot prove

That the app compiles. If it does not, nothing above happens, and the
compile errors are the result.

---

## 7. Temporary resources created

Recorded per `CLAUDE.md` filesystem rule 9. **Nothing here has been
deleted; rule 14 reserves that for a human.**

| Path | What | Size |
|---|---|---|
| `Projects\Glasses-worktrees\object-memory-product\` | this pass's git worktree, branch `product/object-memory-self-contained-v1` | the repo |
| `Projects\Glasses-scratch\omkf-smoke\` | an end-to-end keyframe smoke run's store and keyframes | 43 KB |
| `Projects\Glasses-scratch\det_1..3.json`, `f2e2e_91589\` | detector output from the same smoke runs | ~330 KB |
| `%LOCALAPPDATA%\Temp\gt`, `gk`, `gm`, `gr` | pytest `--basetemp` roots | transient |

The basetemp roots are short **deliberately**: the default and any deep
path trip Windows' 260-character limit once World Builder nests
`worlds/<32-hex>/sessions/<32-hex>/` under `tmp_path`, which manufactures
about 139 phantom failures and one very confusing afternoon. They are
inside the OS temp directory, which rule 12 allows.

**How the tests were run, because the trap here is real.** The worktree
has no venv; the only working interpreter is the MAIN checkout's
`tower\.venv`, and the package is installed editable pointing at the main
checkout. Run that interpreter from a neutral directory and `import
tower` resolves to the **main checkout's source**, not this branch's.
Every run here was made from the worktree's own `tower/` directory, where
the cwd wins:

```
tower package resolved from: ...\Glasses-worktrees\object-memory-product	ower	ower\__init__.py
```

which was checked rather than assumed. Anyone reproducing these numbers
must `cd` into the worktree's `tower/` first, or they will measure a
different branch and get the old answers.

**Nothing was written to `C:\` or to the home directory.** The live
`tower/data/` tree in the main checkout was read for measurement and
never written: every figure in §2.2, §2A.3 and §6 comes from running the
shipped code over it read-only.

---

## 8. What a reviewer should still not believe

- **No iOS code has been compiled or run.** See §6.
- **The verifier's reject path is unproven physically.** §6.
- The suite grew because tests were added, not because behaviour is
  asserted twice. **Before this branch: 2217 passed, 68 skipped, ~430 s.
  After: 2369 passed, 68 skipped, 390 s.** Zero failures either side, and
  `pip check` clean. The 68 skips are the same opt-in gates as before
  (`TOWER_RUN_MODEL_TESTS`, a machine-local corpus, a vendored face
  model); none of them was added or widened here.
- Every "measured" figure here names what it was measured on. Where a
  number came from a synthetic fixture rather than the corpus, it says
  so.
