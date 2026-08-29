//
//  ExperimentalCVContract.swift
//  Glasses
//

import Foundation

// MARK: - The three agreements this build implements

/// The Experimental CV Lab's wire identity, as the Tower declares it.
///
/// ## Three identifiers, not one, and that is the Tower's choice
///
/// `tower/cv_lab/contracts.py` versions the status document, the control
/// vocabulary and the `frame_result` provenance block **separately**, because
/// they travel on different transports at different rates and a change to one
/// has no bearing on the others. The Tower's own words for why:
///
/// > a client may implement the read-only half and never send a command, which
/// > is exactly what a Release iOS build with no camera should do.
///
/// That is not hypothetical here — see `TowerExperimentalCVClient.canStart`.
/// So this build names all three and compares each where it is used, rather
/// than folding them into one string that would make the read-only half
/// unavailable the day the command half changed.
///
/// Every identifier is **opaque**: compared for equality, never parsed, never
/// ordered. They are dated rather than numbered precisely so that nobody
/// computes which is greater — a mismatch means "we are not talking about the
/// same agreement", which is neither newer nor older.
nonisolated enum ExperimentalCVContract {
    /// The **Tower's** name for the cartridge. Not this app's catalog id.
    ///
    /// ## The mapping that has to exist, and does not live here
    ///
    /// This app's catalog calls the cartridge `"experimental-cv"` with a
    /// hyphen (`Cartridge.catalog`); the Tower advertises `"experimental_cv"`
    /// with an underscore. Without an entry joining them in
    /// `TowerCapabilities.towerCartridgeNames`, `declaredContract(for:in:)`
    /// returns `nil`, availability resolves `.noContract`, and this cartridge
    /// reports "the Tower has declared nothing" against a Tower that has
    /// declared everything.
    ///
    /// The mapping is deliberately **not** duplicated here. `TowerCapabilities`
    /// owns it for all five cartridges so that a second cartridge cannot invent
    /// a different convention for the same join.
    static let towerCartridge = "experimental_cv"

    /// This app's catalog id, repeated here only so `cartridgeID` and the
    /// mapping's key cannot drift apart silently.
    static let catalogID = "experimental-cv"

    /// One result type. The Lab publishes **one** document that answers every
    /// question about it — what can run, what is running, and what it found —
    /// rather than a catalog type and a run type that could disagree.
    static let resultType = "status"

    /// The status document: on the result channel, over `GET /cv-lab`, and as
    /// the reply to `cv_lab_status`. Structurally the same document from one
    /// builder on all three.
    static let status = "experimental_cv.status/2026-08-27"

    /// The command vocabulary — start, pause, resume, stop, and their replies.
    static let control = "experimental_cv.control/2026-08-27"

    /// The provenance block on every `frame_result`.
    static let frameResult = "experimental_cv.frame_result/2026-08-27"

    /// The live preview: the descriptor inside `run.annotation.artifact`, and
    /// the headers on the bytes `GET /cv-lab/preview` serves.
    ///
    /// Versioned separately from the status document, and the reason is the
    /// same one that separates the other three: a client may implement the
    /// read-only half and never fetch an image, which is exactly what a
    /// Release build with no camera does. A preview may gain a field without
    /// the status document meaning anything new, and the reverse.
    ///
    /// `artifact` was `null` on every earlier version of this contract, with a
    /// stated reason — `IOS-to-Tower.md` §5 withheld any image whose treatment
    /// was unstated, and said artifact fetching itself was UNKNOWN. This is
    /// that contract, landed on both sides at once with a document behind it,
    /// and the treatment is stated on every picture.
    static let preview = "experimental_cv.preview/2026-08-29"

    // MARK: Bounds the Tower states and this client must respect

    /// The most characters a `request_id` may carry.
    ///
    /// **Longer is dropped, not refused**: the command still applies and the
    /// reply simply comes back without a `request_id`. That is the worst
    /// possible failure for a client that matches replies to buttons — the
    /// command takes effect and the answer cannot be attributed — so this build
    /// bounds its own ids rather than relying on noticing the omission.
    static let requestIDMaxLength = 64

    /// How long an arm may take before the Tower gives up on it.
    ///
    /// 120 s, and the reason is 119 MB of MiDaS weights fetched through
    /// `torch.hub` on first use, which does not fit a 10 s bound on any
    /// ordinary link. **There is no progress reporting** — `torch.hub` offers
    /// none — so an arming Lab can be described as arming and bounded, and
    /// nothing more honest than that exists to draw.
    static let armTimeoutSeconds: TimeInterval = 120
}

// MARK: - Control messages, client → Tower

/// The five control messages, as the Tower names them.
///
/// ## Socket only, and there is no HTTP equivalent to fall back on
///
/// `docs/contracts/TOWER-UNIFIED-CARTRIDGES.md` §3 puts every other cartridge's
/// Start/Pause/Stop on HTTP and makes the CV Lab the one exception: its
/// commands are plain socket messages on `/ws`. The Tower's reason is that a
/// command needs the connection it was issued on to still be there when the
/// outcome arrives — an arm may take two minutes — and a request/response route
/// would have to either block for that or lie about having finished.
///
/// So there is no HTTP client in this cartridge, and nothing here should grow
/// one.
///
/// **There is no separate `select`.** `cvLabStart` selects *and* arms in one
/// step, replacing whatever ran. Selection without arming is a state nobody
/// needs on the wire.
nonisolated enum CVLabCommand: String, CaseIterable, Sendable {
    case status = "cv_lab_status"
    case start = "cv_lab_start"
    case pause = "cv_lab_pause"
    case resume = "cv_lab_resume"
    case stop = "cv_lab_stop"
}

// MARK: - Refusals, Tower → client

/// What a `cv_lab_error` means for this app.
///
/// > **Every one of the eight reasons means the request did not take effect.
/// > There is no partial application.**
///
/// That single sentence is why the refusal carries the unchanged `status`
/// document: a refused client never has to guess what state it is now in, and —
/// more importantly — this app never has to *model* what a half-applied command
/// would have done. The client applies the document it was handed and records
/// the refusal beside it. **Nothing in this cartridge mutates lifecycle state
/// optimistically on the way out of a command**, accepted or refused.
///
/// The decode itself lives in `TowerClient` beside the other wire messages;
/// what is here is the interpretation, which is the cartridge's to make.
extension CVLabControlRefusal {

    /// The status document this refusal carried, decoded.
    ///
    /// Present on every `cv_lab_error` — including the `lab_unavailable`
    /// refusal from a Tower with no Lab at all, which carries a hollow document
    /// with the real contract identifiers in it.
    var decodedStatus: CVLabStatus? {
        guard let status else { return nil }
        return CVLabStatus(json: status)
    }

    /// What a person should be told to *do* about this refusal.
    ///
    /// The distinction this type exists for. `lab_unavailable` and
    /// `internal_error` are both "the Tower could not serve you", and telling
    /// someone to give up on a working Tower is strictly worse than telling
    /// them to try again — which is why the Tower deliberately separated them
    /// and why this app must not re-merge them.
    var disposition: Disposition {
        switch reason {
        case "lab_unavailable":
            return .terminal
        case "internal_error", "lab_busy":
            return .transient
        case "malformed_request", "unknown_experiment", "experiment_unavailable",
            "invalid_state", "stale_run":
            return .requestRefused
        default:
            return .unrecognised
        }
    }

    /// How this app is allowed to act on a refusal.
    ///
    /// Deliberately **four** cases and not two. "Terminal or retryable" would
    /// force the five request-shaped refusals into one of those buckets: as
    /// terminal they would tell a person this Tower cannot run experiments
    /// because they pressed pause twice, and as retryable they would invite a
    /// retry of a request that will be refused identically forever.
    enum Disposition: Equatable, Sendable {
        /// `lab_unavailable`. This Tower runs no CV Lab, or its module failed.
        /// Rendered as `.unsupported` — a fact about the Tower, not an error a
        /// person can act on.
        case terminal
        /// `internal_error` (the Tower failed while answering) and `lab_busy`
        /// (a start is already in flight). The same command may succeed later.
        ///
        /// **Nothing in this cartridge retries automatically.** A person
        /// pressing a button again is a retry; a client looping on one is how
        /// two clients each come to believe they chose what is running.
        case transient
        /// The request itself does not apply — to this Tower, to this state, or
        /// to that run. Sending it again unchanged is refused again.
        case requestRefused
        /// A reason this build was not written against. Neither terminal nor
        /// retryable is safe to assume, so it is neither: the Tower's own
        /// `message` is shown and nothing is claimed on top of it.
        case unrecognised
    }
}

// MARK: - Frame refusals

/// What a `frame_error` means: the Tower answering a frame it did not process.
///
/// ## Why this is not silence, and why it is not a failure either
///
/// While `lifecycle.state` is not `running`, a frame is **answered** rather
/// than dropped — otherwise "the Lab is paused" and "the socket is wedged" look
/// identical from here. The Tower states the position twice, once in prose and
/// once in its own metrics:
///
/// > **A refusal is not counted as a frame processing error.** … a Lab paused
/// > for five minutes has not failed hundreds of times.
///
/// This app owes the same discipline. `isRefusal` below is what keeps a paused
/// Lab out of the failure count, and `TowerClient` does not touch the frame
/// path, the send window or `status` for any of these — it decodes three fields
/// and knows what none of them mean. What they mean is here.
extension TowerFrameRefusal {

    /// The six the CV Lab names, plus the three the transport names.
    ///
    /// Kept as one vocabulary because they arrive on one field and a client
    /// switching on `reason` has to handle all nine. Split into two enums, the
    /// nine would become an unhandled default somewhere.
    var kind: Kind {
        switch reason {
        case "cv_lab_idle": return .labIdle
        case "cv_lab_starting": return .labArming
        case "cv_lab_paused": return .labPaused
        case "cv_lab_stopped": return .labStopped
        case "cv_lab_failed": return .labFailed
        case "cv_lab_unavailable": return .labUnavailable
        case "invalid_frame": return .invalidFrame
        case "frame_skipped": return .frameSkipped
        case "module_unavailable": return .moduleUnavailable
        default: return .unrecognised
        }
    }

    enum Kind: Equatable, Sendable {
        /// Nothing armed. A start would be accepted.
        case labIdle
        /// **Arming, and not an error.**
        ///
        /// The single most misreadable code on this wire. It is the window
        /// between an accepted `cv_lab_start` and a running experiment, bounded
        /// by the Tower's 120 s arm timeout, with no progress reporting
        /// available at all. Drawn as a failure it would put an error on screen
        /// for a model that is downloading exactly as intended; drawn as
        /// silence it would leave someone watching a frozen panel for two
        /// minutes.
        case labArming
        /// Armed and deliberately not processing. The experiment stays loaded.
        case labPaused
        /// The last run ended and its figures are final.
        case labStopped
        /// The last start failed. Another may be sent — see
        /// `ExperimentalCVState.failed`.
        case labFailed
        /// A defensive default rather than a state normally seen: when the Lab
        /// is unavailable the module behind it is FAILED or UNLOADED, so the
        /// transport answers `module_unavailable` first.
        case labUnavailable
        /// The transport could not decode the frame. **A sender problem**, and
        /// the only one of the nine that is: it never reached the Lab, and the
        /// Tower counts it in `source.frames_rejected_before_lab` so that a
        /// phone sending garbage does not read as a phone sending nothing.
        case invalidFrame
        /// The module declined the frame without naming a reason.
        case frameSkipped
        /// No module is loaded to process it.
        case moduleUnavailable
        /// A code this build was not written against.
        case unrecognised

        /// Whether the Tower declined this frame on purpose, as opposed to
        /// failing on it.
        ///
        /// Six of the nine are the Lab saying "not now", and none of them means
        /// anything is broken: the module stays ACTIVE and the next frame is
        /// accepted the moment the Lab is running again.
        var isRefusal: Bool {
            switch self {
            case .labIdle, .labArming, .labPaused, .labStopped, .labFailed, .labUnavailable:
                return true
            case .invalidFrame, .frameSkipped, .moduleUnavailable, .unrecognised:
                return false
            }
        }

        /// One sentence for the result panel, for the states where this app can
        /// say something the Tower's own prose does not already say better.
        ///
        /// `nil` means "show the Tower's `message`", which is the default and
        /// the right answer for every transport code: only the Tower knows
        /// which field was malformed or which module is missing.
        var summary: String? {
            switch self {
            case .labArming:
                return """
                    The Tower is loading the experiment. Frames are refused until it \
                    is ready, and the Tower gives up after \
                    \(Int(ExperimentalCVContract.armTimeoutSeconds)) seconds. There is \
                    no progress to report — the model download does not offer any.
                    """
            case .labPaused:
                return "The Lab is paused, so frames are being refused rather than measured."
            case .labIdle:
                return "No experiment is armed, so frames are being refused rather than measured."
            case .labStopped:
                return "The run has ended. Its figures are final and frames are being refused."
            case .labFailed:
                return "The last start failed, so frames are being refused."
            case .labUnavailable, .invalidFrame, .frameSkipped, .moduleUnavailable,
                .unrecognised:
                return nil
            }
        }
    }
}

// MARK: - The status document

/// The Lab's whole state, in one snapshot. There are no deltas to merge.
///
/// ## One document, three surfaces, and why they are not compared
///
/// `GET /cv-lab`, the reply to `cv_lab_status`, and a `cartridge_result` on the
/// `experimental_cv/status` subscription are built by one function on the
/// Tower. They agree **structurally** — same keys, same types, same meanings —
/// and they are deliberately *not* byte-identical across time: `elapsed_s`, the
/// three throughput figures, `last_frame_at`, `receiving_frames` and
/// `clients_connected` are clock- or connection-derived, so two reads a second
/// apart differ for reasons that are not the contract.
///
/// So nothing in this app compares two of these documents for equality to
/// decide whether something changed. The result channel's own `revision` is the
/// only change identity on this wire, and it is not derivable from a read.
///
/// ## Every field is read tolerantly, and nothing is defaulted into existence
///
/// Absent stays absent. The one exception is `available`, which defaults to an
/// empty catalog rather than to `nil` — a Tower that lists no experiments and a
/// Tower whose list could not be read are both "there is nothing to offer you",
/// and this app holds no list of its own to fall back on. It never invents one:
/// `docs/modules/EXPERIMENTAL-CV.md` calls its candidate list "intentionally
/// broad", so any subset hard-coded here would be the phone asserting that
/// those experiments exist.
struct CVLabStatus: Equatable, Sendable {
    /// The status contract this document declares itself to be. Compared for
    /// equality against `ExperimentalCVContract.status` by the client, never
    /// parsed.
    let contract: String?
    let controlContract: String?
    let frameResultContract: String?
    /// Which Tower this is. Part of every `run_id`, which is what makes a
    /// reconnect to a *restarted* Tower detectable by comparing run ids alone.
    let towerInstanceID: String?
    /// `"tower-receipt"` on every timestamp this Lab emits. There is no capture
    /// timestamp anywhere on this wire.
    let timeBasis: String?
    let lifecycle: Lifecycle
    /// The catalog, sorted by id by the Tower. Displayed in the order given.
    let available: [CVExperiment]
    /// The armed experiment's id, or `nil`.
    let selected: String?
    /// What this Tower arms at boot. Reported so that "the Lab is running"
    /// never reads as "somebody chose this".
    let defaultExperiment: String?
    /// `TOWER_CV_DEVICE` as a **request**, not an answer. The Tower decides
    /// what it actually used and reports that in `run.runtime`.
    let deviceRequested: String?
    /// The current or last run, or `nil` when no run exists.
    let run: CVExperimentRun?
    /// Whether anything at all is feeding this Lab. **Tower-wide.**
    let source: Source

    init?(json: [String: Any]) {
        guard let rawLifecycle = json["lifecycle"] as? [String: Any],
            let lifecycle = Lifecycle(json: rawLifecycle)
        else {
            // Without a lifecycle there is no state to render, and rendering
            // "idle" for a document we could not read would be the fabricated
            // answer this whole layer exists to prevent.
            return nil
        }
        self.contract = json["contract"] as? String
        self.controlContract = json["control_contract"] as? String
        self.frameResultContract = json["frame_result_contract"] as? String
        self.towerInstanceID = json["tower_instance_id"] as? String
        self.timeBasis = json["time_basis"] as? String
        self.lifecycle = lifecycle

        var catalog: [CVExperiment] = []
        for entry in json["available"] as? [[String: Any]] ?? [] {
            if let experiment = CVExperiment(json: entry) { catalog.append(experiment) }
        }
        self.available = catalog

        self.selected = json["selected"] as? String
        self.defaultExperiment = json["default_experiment"] as? String
        self.deviceRequested = json["device_requested"] as? String
        if let rawRun = json["run"] as? [String: Any] {
            self.run = CVExperimentRun(json: rawRun)
        } else {
            self.run = nil
        }
        self.source = Source(json: json["source"] as? [String: Any] ?? [:])
    }

    /// The experiment matching `selected`, when the catalog carries it.
    var selectedExperiment: CVExperiment? {
        guard let selected else { return nil }
        return available.first { $0.id == selected }
    }

    /// Where the Lab is in its life, and since when.
    struct Lifecycle: Equatable, Sendable {
        /// Seven values, unparsed. Kept as a string for the reason every other
        /// vocabulary here is: an eighth state must reach a person as itself.
        let state: String
        /// Prose, present only when the state needs explaining. **`nil` is not
        /// "no reason"** — it is "the state speaks for itself".
        let reason: String?
        /// When the Lab entered this state, in Tower receipt time.
        let since: Date?
        /// The current run, or `nil`.
        let runID: String?

        init?(json: [String: Any]) {
            guard let state = json["state"] as? String else { return nil }
            self.state = state
            self.reason = json["reason"] as? String
            self.since = CVLabWireTime.date(json["since"])
            self.runID = json["run_id"] as? String
        }
    }

    /// Whether anything is feeding this Lab — the question "I pressed Start and
    /// nothing happened" is really asking.
    ///
    /// > **Every figure in this block is TOWER-WIDE, not per connection.**
    ///
    /// One Tower has one Lab and one run, so `receivingFrames == true` means
    /// *somebody* is feeding it — possibly the other phone. There is no
    /// per-connection frame counter anywhere in this contract, because the Lab
    /// is handed bytes and not a connection identity. See
    /// `ExperimentalCVState.isLive`, which is where that fact stops this app
    /// from claiming a Release build's frames are arriving.
    struct Source: Equatable, Sendable {
        /// `nil` when this Tower cannot report it. `> 1` means somebody else is
        /// on this Tower too, which is worth saying out loud: the Lab has one
        /// slot and last start wins.
        let clientsConnected: Int?
        /// `lastFrameAt` within `idleAfterSeconds`.
        let receivingFrames: Bool
        let lastFrameAt: Date?
        let framesOfferedTotal: Int?
        /// Frames that arrived and the transport could not decode — a truncated
        /// JPEG, a bad base64, a missing field. They never reached the Lab, so
        /// they are **not** in `framesOfferedTotal`, and without this figure a
        /// phone sending garbage reads exactly like a phone sending nothing.
        /// Those need opposite fixes.
        let framesRejectedBeforeLab: Int?
        /// How long after the last frame the Tower stops claiming it is
        /// receiving any. 5 s, about four missed frames at the current sender's
        /// observed ~0.8 fps.
        let idleAfterSeconds: Double?

        init(json: [String: Any]) {
            self.clientsConnected = json["clients_connected"] as? Int
            self.receivingFrames = json["receiving_frames"] as? Bool ?? false
            self.lastFrameAt = CVLabWireTime.date(json["last_frame_at"])
            self.framesOfferedTotal = json["frames_offered_total"] as? Int
            self.framesRejectedBeforeLab = json["frames_rejected_before_lab"] as? Int
            self.idleAfterSeconds = json["idle_after_s"] as? Double
        }
    }
}

// MARK: - Wire time

/// Reads the Tower's epoch-second timestamps.
///
/// One place, because every one of them is `time_basis: "tower-receipt"` and
/// the conversion must not quietly become "now" at any call site. A missing or
/// null field is `nil`, never `Date()`: an iOS clock reading is arrival time
/// wearing observation time's label, which is the conflation
/// `ObservationTime` exists to prevent.
nonisolated enum CVLabWireTime {
    /// From a raw JSON field, which may be a number, a null, or absent.
    static func date(_ value: Any?) -> Date? {
        date(seconds: value as? Double)
    }

    /// From a figure already read off a message.
    ///
    /// A separate entry point rather than letting a `Double?` widen to `Any?`:
    /// an optional inside an `Any` is a cast this app should not be relying on
    /// the compiler to get right, in a conversion whose failure mode is a
    /// silently missing timestamp.
    static func date(seconds: Double?) -> Date? {
        guard let seconds else { return nil }
        return Date(timeIntervalSince1970: seconds)
    }
}
