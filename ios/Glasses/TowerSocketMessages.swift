//
//  TowerSocketMessages.swift
//  Glasses
//

import Foundation

/// One `frame_result` message, as the Tower sends it.
///
/// Every field is one the Tower builds in `tower/tower/routes/ws.py` (the
/// `frame_result` payload), and every one is optional (or empty by default)
/// because the decoder must not fabricate a value the message omitted.
///
/// The Tower sends five keys unconditionally — `seq`, `processing_ms`,
/// `result_value`, `result_label` and `stage_ms` — and adds `mean_intensity`
/// and `metrics` when the experiment produced them, and the `cv_lab` block
/// when a CV Lab is running. `result_value` with `result_label` is the
/// experiment's own answer: the only thing the Tower says about what it
/// concluded rather than how long it took.
struct TowerFrameResult: Equatable, Sendable {
    /// The frame this result answers, matching the `seq` the app sent.
    let sequence: Int?
    /// Mean pixel intensity, 0...1. Present only when the experiment reported
    /// one — `nil` means the experiment said nothing, never that the frame was
    /// dark.
    let meanIntensity: Double?
    /// How long the Tower spent on the frame.
    let processingMs: Double?
    /// The experiment's headline number. Its meaning is the experiment's, not
    /// this app's: it is paired with `resultLabel` and must never be rendered
    /// without it, because a bare number implies a unit nobody promised.
    let resultValue: Double?
    /// The experiment's own name for what it measured. The one piece of
    /// provenance on this channel.
    let resultLabel: String?
    /// Per-stage timings inside the Tower's own processing, name -> ms.
    /// Empty when the Tower sent an empty object; there is no distinction on
    /// the wire between empty and absent, and none is invented here.
    let stageMs: [String: Double]
    /// Additive measurements, name -> number, omitted entirely when empty.
    /// Deliberately numbers only: `ws.py` calls this "a MEASUREMENT channel,
    /// not the structured result channel", and the structured one is blocked
    /// on module-contract work this app must not pre-empt.
    let metrics: [String: Double]
    /// Who produced this number, and for which run.
    ///
    /// `nil` from a Tower that runs no CV Lab — the block is additive and
    /// omitted entirely rather than nulled. Every field of it is read in
    /// `TowerFrameResultProvenance`, and one of them, `runID`, is read by this
    /// client itself: see `watchedCVLabRunID`.
    let cvLab: TowerFrameResultProvenance?
}

extension TowerFrameResult {
    /// Decodes a `frame_result` message. Never fails: every field is optional
    /// or empty by default.
    init(json: [String: Any]) {
        // Additive, and omitted entirely by a Tower that runs no Lab. An
        // `if let` rather than `flatMap(TowerFrameResultProvenance.init)`
        // for the isolation reason `TowerCartridgeDeclaration.init` gives.
        var cvLab: TowerFrameResultProvenance?
        if let rawCVLab = json["cv_lab"] as? [String: Any] {
            cvLab = TowerFrameResultProvenance(json: rawCVLab)
        }
        self.init(
            sequence: json["seq"] as? Int,
            meanIntensity: json["mean_intensity"] as? Double,
            processingMs: json["processing_ms"] as? Double,
            resultValue: json["result_value"] as? Double,
            resultLabel: json["result_label"] as? String,
            // `?? [:]` rather than an optional: the Tower sends `stage_ms`
            // unconditionally but may send it empty, and `metrics` is omitted
            // when empty. Both mean "no stages/measurements to report", so
            // collapsing absent and empty here loses nothing.
            stageMs: json["stage_ms"] as? [String: Double] ?? [:],
            metrics: json["metrics"] as? [String: Double] ?? [:],
            cvLab: cvLab
        )
    }
}

/// The `cv_lab` block on a `frame_result` —
/// `experimental_cv.frame_result/2026-08-27`.
///
/// Without it a reply names the number and not the experiment: no way to tell
/// which of eight experiments produced `result_value`, on which run, measured
/// or inferred, so a result from a previous experiment could be read as one
/// from the current one.
///
/// ## The one rule a client must not get wrong
///
/// > **Discard any `frame_result` whose `cv_lab.run_id` is not the run you are
/// > watching.**
///
/// The Tower makes this structural rather than checked: a new experiment is a
/// new run, and the old experiment is released **before** the new run id is
/// published, so no result computed by one experiment can carry another's name.
/// The client-side rule exists for the case the Tower cannot cover — a
/// reconnect to a **restarted** Tower, which starts counting runs again from 1.
/// `runID` is `"<tower_instance_id>-<n>"` and `towerInstanceID` is part of it
/// for exactly that reason, which is why comparing `runID` alone is sufficient
/// and no separate instance check is needed.
///
/// The gate itself is in `TowerClient.handleFrameResult`, not here: a value
/// type cannot know which run is being watched.
struct TowerFrameResultProvenance: Equatable, Sendable {
    /// Opaque, compared for equality only.
    let contract: String?
    /// Which Tower. Already inside `runID`; carried because a client that
    /// reports a run id to a person should be able to say which machine it
    /// belongs to.
    let towerInstanceID: String?
    /// `"<tower_instance_id>-<n>"`. The run this result belongs to.
    let runID: String?
    /// **Dense within the run, from 1.**
    ///
    /// This is the field that orders results. The message's own `seq` is the
    /// phone's capture index and **skips by design** — the sender forwards only
    /// the frames its rate gate selects — so it cannot order anything. That the
    /// two are different numbers with different meanings on the same message is
    /// the single easiest thing to get wrong here.
    let resultSeq: Int?
    let experimentID: String?
    /// The Tower's own name for the experiment, for display.
    let experimentName: String?
    /// `measured` or `inferred`, as a string. Interpreted by the cartridge that
    /// owns the contract — this file does not know what an experiment is.
    let provenance: String?
    /// `opencv` or `torch`.
    let backend: String?
    /// What the experiment actually ran on, or `nil` when it holds nothing.
    /// Distinct from `deviceRequested`, which is a *request*: `auto` is the
    /// question, and this is the answer.
    let device: String?
    let deviceRequested: String?
    /// The experiment's own name for the number, repeated here so the block is
    /// self-contained.
    let resultLabel: String?
    let processingMs: Double?
    /// When the **Tower** received the frame. `timeBasis` says so on every
    /// block, because there is no capture timestamp anywhere on this wire —
    /// `tower/frames.py` carries no time field.
    let towerReceivedAt: Double?
    let timeBasis: String?

    init(json: [String: Any]) {
        // Nothing is required. A block that arrived with only a `run_id` still
        // gates correctly, and a block with everything but one field still
        // displays the rest — the alternative, refusing the whole block on a
        // missing optional, would silently turn a gated result into an ungated
        // one, which is the failure this type exists to prevent.
        self.contract = json["contract"] as? String
        self.towerInstanceID = json["tower_instance_id"] as? String
        self.runID = json["run_id"] as? String
        self.resultSeq = json["result_seq"] as? Int
        self.experimentID = json["experiment_id"] as? String
        self.experimentName = json["experiment_name"] as? String
        self.provenance = json["provenance"] as? String
        self.backend = json["backend"] as? String
        self.device = json["device"] as? String
        self.deviceRequested = json["device_requested"] as? String
        self.resultLabel = json["result_label"] as? String
        self.processingMs = json["processing_ms"] as? Double
        self.towerReceivedAt = json["tower_received_at"] as? Double
        self.timeBasis = json["time_basis"] as? String
    }
}

// MARK: - The frame path's refusals

/// One `frame_error`: the Tower answering a frame it did not process.
///
/// It is what tells "the Lab is paused", "that frame was undecodable" and "the
/// socket is wedged" apart from this side, and it answers *"I pressed Start
/// and nothing happened"*.
///
/// ## Refusal, not failure
///
/// Six of the nine reasons are the CV Lab declining a frame on purpose. The
/// Tower is explicit that these are counted under `frames_rejected` and **not**
/// under `frame_processing_errors` — *"a Lab paused for five minutes has not
/// failed hundreds of times"* — and this client owes the same discipline:
/// nothing here touches `status`, the send window, the stream bracket or
/// `SenderMetrics`' error counters. It decodes, publishes, and returns.
///
/// The nine reasons are interpreted by the cartridge that owns the CV Lab
/// contract, not here. This file decodes three fields and knows what none of
/// them mean.
struct TowerFrameRefusal: Equatable, Sendable {
    /// The frame this answers — the phone's **capture index**, and `nil` when
    /// the frame failed validation before its `seq` was even readable. The
    /// Tower reports null rather than inventing one.
    let sequence: Int?
    /// The Tower's code: `cv_lab_idle`, `cv_lab_starting`, `cv_lab_paused`,
    /// `cv_lab_stopped`, `cv_lab_failed`, `cv_lab_unavailable`,
    /// `invalid_frame`, `frame_skipped` or `module_unavailable`.
    ///
    /// Kept as a **string**, like every other refusal vocabulary this client
    /// carries: a code this build does not recognise must still reach a person
    /// rather than being collapsed into "unknown".
    let reason: String
    /// Prose for a person. The Tower's message says what to send next.
    let message: String

    init?(json: [String: Any]) {
        guard let reason = json["reason"] as? String else { return nil }
        self.sequence = json["seq"] as? Int
        self.reason = reason
        self.message = json["message"] as? String ?? reason
    }
}

// MARK: - The CV Lab control plane

/// A `cv_lab_status` message: the Lab's whole state, in one snapshot.
///
/// ## `acceptedCommand` is the field that makes this two messages
///
/// The Tower uses one name for one document, whichever direction it is
/// travelling and whatever prompted it — a read, an accepted command, or a push
/// on the result channel. `acceptedCommand` is present **only** on a reply to a
/// command, and it is therefore the only way to tell an answer from a pushed
/// state. A client that treated every arriving document as an answer would
/// report "started" every two seconds for as long as the Lab was running.
///
/// A plain `cv_lab_status` read comes back with no `accepted_command` at all,
/// while `cv_lab_start`/`pause`/`resume`/`stop` each echo their own name.
///
/// The `status` document itself is passed through **undecoded**, for the same
/// reason `CartridgeResultEnvelope.payload` is: this file owns the transport
/// and the cartridge owns the contract. Nothing here knows what an experiment
/// is. The one exception is `runID`, which the frame path needs — see
/// `watchedCVLabRunID`.
struct CVLabControlReply {
    /// The status document's contract identifier. Opaque.
    let contract: String?
    /// The **control vocabulary's** identifier, which versions separately from
    /// the document's precisely so a client can implement the read-only half
    /// and never send a command.
    let controlContract: String?
    /// Echoed from the request, when one was sent and was at most 64
    /// characters. **A longer one is dropped, not refused** — the command still
    /// applies and the reply simply carries none — which is why this client
    /// bounds its own ids rather than relying on noticing the absence.
    let requestID: String?
    /// The command this answers, or `nil` when the document was pushed or read.
    let acceptedCommand: String?
    /// The whole document, undecoded.
    let status: [String: Any]

    init?(json: [String: Any]) {
        guard let status = json["status"] as? [String: Any] else { return nil }
        self.contract = json["contract"] as? String
        self.controlContract = json["control_contract"] as? String
        self.requestID = json["request_id"] as? String
        self.acceptedCommand = json["accepted_command"] as? String
        self.status = status
    }

    /// `lifecycle.run_id`, the one field this file reads out of the document.
    ///
    /// Read here rather than in the cartridge because the **frame path** needs
    /// it, and the frame path is this file's. It is a single traversal of two
    /// keys; it does not make this client cartridge-aware in any other respect.
    var runID: String? {
        Self.runID(in: status)
    }

    /// `lifecycle.run_id` of any status document, including the unchanged one
    /// a `cv_lab_error` carries.
    static func runID(in status: [String: Any]) -> String? {
        (status["lifecycle"] as? [String: Any])?["run_id"] as? String
    }

    /// `lifecycle.state`, read only so a log line says something useful.
    var lifecycleState: String? {
        (status["lifecycle"] as? [String: Any])?["state"] as? String
    }
}

/// A `cv_lab_error`: a refusal.
///
/// > **Every one of the eight reasons means the request did not take effect.
/// > There is no partial application.**
///
/// Which is why this carries the unchanged `status`: a refused client never has
/// to guess what state it is now in, and nothing in this app has to model a
/// half-applied command. Even the `lab_unavailable` refusal from a Tower with
/// no Lab at all carries a document — a hollow one, with the real contract
/// identifiers in it — so a decoder is entitled to require the field. It is
/// still optional here, because a refusal that arrived without one is worth
/// showing rather than dropping.
struct CVLabControlRefusal {
    /// `malformed_request`, `unknown_experiment`, `experiment_unavailable`,
    /// `lab_busy`, `invalid_state`, `stale_run`, `lab_unavailable` or
    /// `internal_error`. A string, so a ninth reaches a person as itself.
    let reason: String
    /// Prose for a person, from the Tower. The only text that knows *which*
    /// module was missing or *which* run is current.
    let message: String
    /// Which command was refused. Lets a refusal be attributed to a button even
    /// when no `request_id` was sent.
    let command: String?
    let requestID: String?
    let controlContract: String?
    /// `unknown_experiment` only: the ids this Tower does have.
    let available: [String]
    /// `experiment_unavailable` only.
    let experimentID: String?
    /// `stale_run` only: the run the Lab is actually on now. A stale `run_id`
    /// is refused rather than applied to whichever run is current, so a Stop
    /// drawn against a run that has already been replaced cannot end the one
    /// that replaced it.
    let currentRunID: String?
    /// The document, unchanged. Undecoded, for the reason `CVLabControlReply`
    /// gives.
    let status: [String: Any]?

    init?(json: [String: Any]) {
        guard let reason = json["reason"] as? String else { return nil }
        self.reason = reason
        self.message = json["message"] as? String ?? reason
        self.command = json["command"] as? String
        self.requestID = json["request_id"] as? String
        self.controlContract = json["control_contract"] as? String
        self.available = json["available"] as? [String] ?? []
        self.experimentID = json["experiment_id"] as? String
        self.currentRunID = json["current_run_id"] as? String
        self.status = json["status"] as? [String: Any]
    }
}

/// Everything the CV Lab's control plane delivers, as one stream.
///
/// One subject rather than three published properties, for the reason
/// `CartridgeResultEvent` gives: ordering between them is load-bearing. A
/// `cv_lab_error` and the pushed status that follows it must not be observed
/// out of order, or a client would file the refusal against the wrong state.
///
/// `frameRefused` rides this stream too. A `frame_error` is a frame-path
/// message and six of its nine reasons are the Lab's own — *"the Lab is
/// paused"* is the answer to *"why is there no result"*, which is a CV Lab
/// question wherever it arrives from.
enum CVLabEvent {
    /// A status document: an answer to a command when `acceptedCommand` is
    /// non-nil, a read or a push otherwise.
    case status(CVLabControlReply)
    /// A refused command. The request did not take effect.
    case refused(CVLabControlRefusal)
    /// A frame the Tower did not process.
    case frameRefused(TowerFrameRefusal)
}
