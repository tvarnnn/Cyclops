"""Identifiers, states and refusal reasons for the CV Lab wire surface.

Kept in one file for the same reason `tower/results/contracts.py` is:
these are the strings a phone switches on, and a string a phone switches
on is not a detail of the module that happens to emit it.

Every identifier here is **opaque and compared for equality only**, the
rule `IOS-to-Tower.md` 0.1 sets for the whole platform. Dated rather than
numbered so that two of them are visibly different without inviting
anyone to compute which is greater. Changing one is a WIRE BREAK: iOS
will report "update the app" and stop decoding, which is the intended
behaviour. Change one only when a payload's MEANING changes in a way an
existing decoder would get wrong -- adding an optional field an older
decoder ignores is not that.
"""

# The cartridge name. Must equal `tower.results.contracts
# .CARTRIDGE_EXPERIMENTAL_CV`; a test asserts it rather than importing it,
# because the result channel core is deliberately cartridge-blind and this
# module must not become the thing that makes it otherwise.
CARTRIDGE = "experimental_cv"

# The one result type. The Lab publishes one document that answers every
# question about it -- what can run, what is running, and what it found --
# rather than a `catalog` type and a `run` type that could disagree.
RESULT_TYPE_STATUS = "status"

# The status document, on the result channel, over `GET /cv-lab`, and as
# the reply to `cv_lab_status`. All three are the same bytes.
STATUS_CONTRACT = "experimental_cv.status/2026-08-27"

# The provenance block added to every `frame_result`. Versioned SEPARATELY
# from the status document because it travels on a different transport at
# a different rate, and a change to one has no bearing on the other.
FRAME_RESULT_CONTRACT = "experimental_cv.frame_result/2026-08-27"

# The command vocabulary: start/pause/resume/stop and their replies.
# Versioned separately again -- a client may implement the read-only half
# and never send a command, which is exactly what a Release iOS build with
# no camera should do.
CONTROL_CONTRACT = "experimental_cv.control/2026-08-27"

# Every timestamp the Lab emits. There is no capture timestamp anywhere on
# the wire (`tower/frames.py` carries no time field), so a Tower timestamp
# is when the TOWER saw something, never when the glasses did.
TIME_BASIS = "tower-receipt"


# -- lifecycle ---------------------------------------------------------
#
# Seven states, and the mapping onto iOS's `ExperimentalCVState` is in
# docs/contracts/EXPERIMENTAL-CV-LAB.md. They are NOT the module's states:
# `ModuleState` describes the one Lab slot in the container, and these
# describe what is in it. A module can be ACTIVE while the Lab is idle,
# and that is the normal state of a Tower nobody has asked for an
# experiment yet.

# The Lab cannot run anything at all -- the module failed, or nothing is
# registered. A Tower limitation, not a "wait a moment".
STATE_UNAVAILABLE = "unavailable"
# Nothing armed. A catalog is available and a start would be accepted.
STATE_IDLE = "idle"
# A start was accepted and the experiment is loading. Frames arriving now
# are refused, and counted.
STATE_STARTING = "starting"
# Processing frames.
STATE_RUNNING = "running"
# Armed and deliberately not processing. The experiment stays loaded, so
# a resume costs nothing -- which is the whole difference from `stopped`.
STATE_PAUSED = "paused"
# The last run ended. Its figures are final and still readable.
#
# Deliberately not "completed": a bench run does not complete, it is
# stopped by a person. iOS renders it as `.completed` because that is the
# case its state machine has; the Tower says what actually happened.
STATE_STOPPED = "stopped"
# The last start failed. `reason` says how, and another start may be
# sent -- which is the difference between a Lab failure and a module
# failure.
#
# Reached ONLY from a failed arm. A run that dies mid-frame -- an
# experiment raising something that is not a `FrameProcessingError` --
# does not land here: `ModuleContainer` marks the MODULE failed, which is
# terminal by design, and the Lab reports `unavailable` until the Tower
# restarts. That is a limitation of the shared module lifecycle rather
# than of the Lab, it is stated in the contract's Known Limitations, and
# an earlier version of this comment claimed the opposite.
STATE_FAILED = "failed"

LIFECYCLE_STATES = (
    STATE_UNAVAILABLE,
    STATE_IDLE,
    STATE_STARTING,
    STATE_RUNNING,
    STATE_PAUSED,
    STATE_STOPPED,
    STATE_FAILED,
)


# -- where a run came from ---------------------------------------------

# The Tower started this run itself at boot, from `TOWER_CV_EXPERIMENT`
# or its default. Reported so that "the Lab is running" never reads as
# "somebody asked for this".
ORIGIN_STARTUP_DEFAULT = "startup_default"
# A client sent `cv_lab_start`.
ORIGIN_CLIENT_REQUEST = "client_request"


# -- refusal reasons ---------------------------------------------------
#
# A closed set. A client switches on these, so adding one is a contract
# change. Every one of them means the request did NOT take effect --
# there is no partial application, and the reply carries the unchanged
# status so a client never has to guess what state it is now in.

# The request was not shaped like a request.
ERR_MALFORMED = "malformed_request"
# No experiment with that id is registered on this Tower. The reply
# carries `available` so the client can correct itself.
ERR_UNKNOWN_EXPERIMENT = "unknown_experiment"
# The experiment exists but this Tower cannot run it -- most often the
# optional [ml] extra is not installed.
ERR_EXPERIMENT_UNAVAILABLE = "experiment_unavailable"
# A start or stop is already in flight. Refused rather than queued: a
# queue would let two clients each believe they chose what is running.
ERR_LAB_BUSY = "lab_busy"
# The command does not apply from the current state (resume when idle,
# pause when stopped).
ERR_INVALID_STATE = "invalid_state"
# The command named a `run_id` that is not the current one. The run it
# meant to act on is already gone, and acting on the current one instead
# would be the wrong run stopped by the wrong person.
ERR_STALE_RUN = "stale_run"
# The Lab itself cannot serve anything -- see STATE_UNAVAILABLE.
# TERMINAL from a client's point of view: iOS renders it as
# `.unsupported`, which tells a person this Tower cannot do this rather
# than inviting them to try again.
ERR_LAB_UNAVAILABLE = "lab_unavailable"
# The Tower failed while answering, and the request did not take effect.
# Deliberately NOT `lab_unavailable`: a handler bug is transient and
# retryable, and reporting it as the terminal condition would tell a
# person to give up on a Tower that is working. The Tower does not know
# what went wrong -- if it did, this would be a different reason.
ERR_INTERNAL = "internal_error"
# There is deliberately NO `start_failed` refusal. An arm is
# asynchronous -- that is the whole reason a start returns immediately --
# so by the time a load fails the command has already been answered
# `accepted`, and a second reply to a reply is not a thing the wire has.
# The outcome arrives as STATE: `lifecycle.state` goes `failed` with a
# reason, pushed on the result channel or read with `cv_lab_status`. That
# is the shape iOS's own `run(_:)` already has, and the constant that
# used to sit here was declared, imported, and never emitted.

REFUSAL_REASONS = (
    ERR_MALFORMED,
    ERR_UNKNOWN_EXPERIMENT,
    ERR_EXPERIMENT_UNAVAILABLE,
    ERR_LAB_BUSY,
    ERR_INVALID_STATE,
    ERR_STALE_RUN,
    ERR_LAB_UNAVAILABLE,
    ERR_INTERNAL,
)


# -- why a frame was not processed -------------------------------------
#
# These reach the client as `frame_error.reason`, alongside the transport's
# own `invalid_frame` / `frame_skipped` / `module_unavailable`. They exist
# because "the Lab is paused" and "that frame was undecodable" are
# different facts and a person acts on them differently.

FRAME_REFUSED_IDLE = "cv_lab_idle"
FRAME_REFUSED_STARTING = "cv_lab_starting"
FRAME_REFUSED_PAUSED = "cv_lab_paused"
FRAME_REFUSED_STOPPED = "cv_lab_stopped"
FRAME_REFUSED_FAILED = "cv_lab_failed"
# A defensive default rather than a state a client will normally see:
# when the Lab is `unavailable` the module behind it is FAILED or
# UNLOADED, so `ModuleContainer.process` refuses the frame with
# `module_unavailable` before the Lab is reached at all. It stays because
# `process()` falls back to it for any state not in the map below, and a
# fallback that names the wrong thing is worse than one that names this.
FRAME_REFUSED_UNAVAILABLE = "cv_lab_unavailable"

FRAME_REFUSAL_REASONS = {
    STATE_IDLE: FRAME_REFUSED_IDLE,
    STATE_STARTING: FRAME_REFUSED_STARTING,
    STATE_PAUSED: FRAME_REFUSED_PAUSED,
    STATE_STOPPED: FRAME_REFUSED_STOPPED,
    STATE_FAILED: FRAME_REFUSED_FAILED,
    STATE_UNAVAILABLE: FRAME_REFUSED_UNAVAILABLE,
}


# -- bounds ------------------------------------------------------------

# How long after the last frame the Lab stops claiming it is receiving
# any. The current iOS sender forwards roughly 1-in-30 of a ~24 fps
# capture, which `tower/metrics.py` records as ~0.8 frames per second
# observed. Five seconds is therefore about four missed frames -- long
# enough never to flicker during normal streaming, short enough that
# "Start was pressed and nothing is arriving" shows up while a person is
# still standing there.
STREAM_IDLE_AFTER_S = 5.0

# The most metrics one run reports. Every registered experiment emits far
# fewer (the largest, `redaction_impact`, emits twelve), so this bounds a
# future experiment rather than truncating a present one -- and when it
# does truncate it says how many it dropped rather than quietly showing a
# shorter list.
MAX_REPORTED_METRICS = 16

# There is deliberately NO cap on how many metric names a run
# accumulates, because one is unreachable and an unreachable guard reads
# as care while providing none. A name enters the accumulator only if
# `classify_metric` recognises it, and that means only if the experiment
# DECLARED it in its own `METRIC_KINDS` -- a compile-time set, twelve
# entries at its largest. A name the experiment never declared is
# excluded and counted below instead. The bound is the declaration, and
# `test_the_accumulator_is_bounded_by_the_declared_metric_set` is what
# says so.

# The most distinct STAGE names one run will accumulate. Unlike the
# metric accumulator -- bounded by the experiment's own `METRIC_KINDS`
# declaration -- a stage name is whatever an experiment passed to
# `StageTimer.stage()`, with nothing declaring it in advance. An
# experiment naming a stage per frame grew this to 926,280 entries over
# 15,438 frames in a probe, which is exactly the unbounded store a run
# open "for as long as the Tower is up" must not have. Sixteen, because
# the most stages any registered experiment uses is four.
MAX_TRACKED_STAGES = 16

# The most unclassified metric names reported. An experiment emitting a
# metric it never classified is a bug caught by
# `test_every_metric_the_experiment_emits_is_classified`; this is what
# happens if one reaches production anyway, and it must not become an
# unbounded list of names a remote producer chose.
MAX_UNCLASSIFIED_REPORTED = 8


# -- the live preview --------------------------------------------------
#
# What follows is the contract `_annotation`'s `artifact` field was
# reserved for. `IOS-to-Tower.md` 5 said artifact fetching was UNKNOWN
# because iOS "holds no URL, no id format, and no bytes, because
# inventing a fetch scheme would be exactly the fabricated contract this
# work refuses to produce" -- and the Tower agreed, because a scheme
# invented on one side is the same fabrication seen from the other. This
# lands on BOTH sides at once, in one change, with a document under
# `docs/contracts/`, which is the only way that objection is answered
# rather than ignored.
#
# What it does NOT do is relax anything. The image this serves is
# derived, never raw; it is held one at a time, never queued; it is
# written to no file; and it arrives stating a treatment iOS already has
# an enum value for, whose documented meaning is the strict one.

# The preview descriptor, inside `run.annotation.artifact`, and the
# headers on the bytes. Dated separately from the status document because
# a preview can gain a field without the document meaning anything new,
# and because a client may implement the status half and never fetch an
# image -- which is exactly what a Release iOS build with no camera does.
PREVIEW_CONTRACT = "experimental_cv.preview/2026-08-29"

# The one HTTP surface. A path, not a URL: the Tower does not know what
# address a phone reached it on, and a client that already resolved a
# base URL to ask for the status document can resolve this against the
# same one. `GET /cv-lab` beside it is the precedent.
PREVIEW_PATH = "/cv-lab/preview"

# What the artifact IS, so that a later kind (an annotated frame, a
# segmentation overlay) is a new value here rather than a client
# guessing from the media type.
ARTIFACT_KIND_LIVE_PREVIEW = "live_preview"

# -- treatment ---------------------------------------------------------
#
# The vocabulary is iOS's `RedactionState`, spelled the way this wire
# spells things. There are three values and this Tower emits exactly one
# of them, because there is exactly one honest answer:
#
#   redacted        a redaction step ran. NOT TRUE HERE. No face
#                   detector runs on the preview path, and pretending
#                   otherwise is the "switch the Tower cannot honour"
#                   that `VisualArtifact.swift` says is worse than no
#                   switch at all.
#   raw_ephemeral   untreated, live view only, never persisted and never
#                   re-served. TRUE HERE, in every clause.
#   unknown         the producer did not say. Withheld by iOS, correctly.
#
# A fourth, gentler value was considered and rejected. `IOS-to-Tower.md`
# 5 says "There is deliberately no `.probablySafe` and no lenient
# default", and "this image is only a Canny edge map" is precisely the
# argument a `.probablySafe` would encode. It is also wrong: an edge map
# of a face keeps the jawline, the glasses and the hairline, and a depth
# map keeps the silhouette. Derived is not the same as unrecognisable.
TREATMENT_RAW_EPHEMERAL = "raw_ephemeral"

# The process claim beside the treatment, in the naming discipline
# `world_builder/redaction.py` and `object_memory/imagery.py` both use:
# it says what was DONE, never what the result is safe for. "none" here
# is the whole sentence -- no detector ran, so nothing was found and
# nothing was filled.
PREVIEW_FACE_FILTER_NONE = "none"

# What a client is promised about where these bytes live. Nothing is
# written to disk on this path and nothing older than the newest frame
# survives, which is what makes `raw_ephemeral`'s "never for anything
# stored or re-served" a property of the Tower and not only a request to
# the phone.
PREVIEW_PERSISTENCE_NONE = "none"

# -- why there is no preview -------------------------------------------
#
# A closed set, on the body of every refusal and in
# `artifact_unavailable_reason`'s place in the status document. A client
# switches on these; the prose beside them is for a person.

# This Tower has previews turned off.
PREVIEW_DISABLED = "preview_disabled"
# The running experiment produces no visual output. `frame_quality` has
# nothing to draw and says so rather than drawing something.
PREVIEW_NOT_VISUAL = "experiment_has_no_visual_output"
# Previews are on, the experiment has one, and no frame has arrived yet.
PREVIEW_NONE_YET = "no_preview_yet"
# The newest preview is older than `PREVIEW_MAX_AGE_S`. A phone showing a
# four-second-old edge map while its wearer turns their head is showing
# a lie about where they are looking.
PREVIEW_STALE = "preview_stale"
# The caller named a run that is not the current one. THE staleness
# guard: experiment A stopped, B started, and a preview of A must never
# be drawn under B's name.
PREVIEW_RUN_CHANGED = "preview_run_changed"
# Rendering the newest preview raised. The experiment is unaffected --
# see `LivePreview.render`.
PREVIEW_RENDER_FAILED = "preview_render_failed"

PREVIEW_REASONS = (
    PREVIEW_DISABLED,
    PREVIEW_NOT_VISUAL,
    PREVIEW_NONE_YET,
    PREVIEW_STALE,
    PREVIEW_RUN_CHANGED,
    PREVIEW_RENDER_FAILED,
)


# -- preview bounds ----------------------------------------------------

# The longest side of a served preview, in pixels. Chosen from
# measurement rather than taste: at 320 an edge map encodes to roughly
# 1-15 KB of PNG in 0.3-1.5 ms and a colourised depth map to roughly
# 20-50 KB of JPEG in 2-4 ms, all of it on a worker thread. 384 costs
# about half as much again for detail nobody reading a phone-sized panel
# will see. Nothing is ever UPSCALED to reach it: MiDaS-small's own
# transform already caps its output near 256x192, and stretching that to
# 320 would spend bytes inventing pixels the phone can invent for free.
PREVIEW_MAX_EDGE_PX = 320

# The floor on the gap between two captures, which is the whole of the
# "visualisation is throttled independently of processing" requirement.
# 20 Hz, so a phone polling at 10 Hz always finds something no more than
# one capture old.
#
# It is a bound rather than a saving. A capture costs two attribute
# assignments -- the array the experiment already computed is handed over
# by reference and never touched on the frame path -- so raising this to
# every frame would cost nothing measurable. It exists because a bound
# that is never reached is still the thing that makes the guarantee.
PREVIEW_MIN_INTERVAL_S = 0.05

# How old the newest preview may be and still be served. Two seconds is
# about twenty missed captures at the throttle ceiling, and about two
# missed frames at the ~1-in-30 rate the current iOS sender actually
# forwards -- long enough to survive a stutter, short enough that a
# stopped stream stops the picture instead of freezing it.
PREVIEW_MAX_AGE_S = 2.0

# What the Tower suggests a client poll at. Advisory: the Tower cannot
# make a phone poll at any rate and does not try. It is here so that the
# rate is one number in one place rather than a constant hardcoded on
# the phone that nobody can change from the machine serving the frames.
PREVIEW_POLL_INTERVAL_S = 0.1

# JPEG quality for the colourised depth map. The same figure
# `object_memory/imagery.py` serves frames at. A depth map is smooth, so
# JPEG's block transform has nothing to ring against; an edge map is
# not, which is why it is PNG instead.
PREVIEW_JPEG_QUALITY = 80

# PNG effort for the edge map. Level 3, not the zlib default of 6 or
# OpenCV's 1: measured on real Canny output the three are within a
# kilobyte of each other, and 3 is the cheapest that is not simply
# storing the image.
PREVIEW_PNG_COMPRESSION = 3
