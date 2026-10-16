//
//  ObjectMemoryModel.swift
//  Glasses
//

import Foundation

/// Object Memory's observations, as this app is allowed to understand them.
///
/// ## The one thing to read before changing anything in this file
///
/// **This cartridge does not know where anything is.** `spatial_ref` is `null`
/// at every level of every payload, always, and is nulled on read by the Tower
/// rather than merely left unset. The `where` object is a *frame reference* — a
/// capture id, a frame sequence number, a camera, and a box in normalised frame
/// coordinates. It points back into a recording. It is not a place, it does not
/// become a place by being drawn on something, and a bounding box under
/// `kind: "frame-reference"` must never be projected into any world frame.
///
/// Three further claims travel *in* the payload rather than only in
/// `docs/contracts/OBJECT-MEMORY.md`, and every one of them constrains the
/// copy in `ObjectMemoryCopy`:
///
/// - `claim: "category-was-visible-once"` — a record says a category was in
///   view **once**. Never that it is there now.
/// - `identity: "category-not-instance"` — `laptop` means *a* laptop. Not
///   "your laptop", and two `laptop` records are not evidence about the same
///   object. Nothing re-identifies anything across sightings.
/// - `absence_means: "not-observed-by-this-cartridge"` — an empty answer is a
///   statement about what this cartridge captured, never about the world. That
///   is why `last-seen` answers 200 with `observed: false` and never 404.
///
/// ## Encoding disciplines this file inherits
///
/// `null` means **absent**, never `0` and never `""`. Every nullable wire field
/// below is an Optional, and nothing is defaulted to a number: a missing
/// `best_score` is "not tracked", and substituting `0.0` would invent a claim
/// of no evidence.
///
/// The contract identifier is **opaque**. Compared for equality, never parsed,
/// never split on `/`, never ordered. A different date is a different
/// agreement, neither newer nor older.

// MARK: - The contract this build implements

/// The one Object Memory agreement this build was written against, and the
/// three claim values that travel with every payload under it.
///
/// The claim constants are here rather than inline at their comparison sites so
/// that a Tower which changed what its data *means* — the most breaking change
/// this contract can carry — fails a decode in one place instead of quietly
/// rendering under the old wording.
nonisolated enum ObjectMemoryContract {
    static let identifier = "object_memory.observations/2026-08-26"

    static let categoryClaim = "category-was-visible-once"
    static let identityScope = "category-not-instance"
    static let absenceMeaning = "not-observed-by-this-cartridge"

    /// `where.kind`. The name is the enforcement: a value that is not this is
    /// not a frame reference, and this app has no other way to read a `where`.
    static let frameReferenceKind = "frame-reference"

    /// `time_basis`. Tower-receipt time — when the Tower received the frame,
    /// never when the shutter fired. `Rule 16` forbids conflating them, so a
    /// basis this app does not recognise is shown uninterpreted rather than
    /// being rendered as a capture time anyway.
    static let towerReceiptBasis = "tower-receipt"

    /// `where.imagery_retention`. The pointer resolves into the Tower's capture
    /// store, whose lifetime this cartridge neither sets nor enforces.
    static let captureSideImagery = "capture-side"
}

// MARK: - Confidence

/// The platform's confidence vocabulary, as it arrives on this wire.
///
/// **Four cases, not three.** `docs/contracts/OBJECT-MEMORY.md` §4.4 lists
/// `low | medium | high`; the Tower's own `tower/confidence.py` also defines
/// `unknown`, returned by `Confidence.from_score(None)`. A record written
/// before scores were tracked carries it. Decoding only the documented three
/// would refuse a real record, so the fourth is here — and it is a real case
/// rather than being folded into `.low`, because "we did not score this" and
/// "we scored this poorly" are different facts.
///
/// None of these is a calibrated probability. They are an interpretation of
/// detector output.
nonisolated enum ObjectMemoryConfidence: String, Equatable, Sendable, CaseIterable {
    case unknown
    case low
    case medium
    case high

    /// Sentence-ready, and deliberately not title-cased shouting.
    var displayName: String {
        switch self {
        case .unknown: return "not recorded"
        case .low: return "low"
        case .medium: return "medium"
        case .high: return "high"
        }
    }
}

// MARK: - Frame reference

/// Where in a *recording* a sighting came from. **Never where in a room.**
///
/// `spatial_ref` has no property here, and its absence is deliberate: there is
/// nothing for it to hold. The decoder still *checks* it, because a payload
/// that started populating it would be making a claim this app must refuse
/// rather than ignore — see `ObjectMemoryDecoder.frame(from:)`.
///
/// `boundingBoxNormalized` is `(x1, y1, x2, y2)` in normalised frame
/// coordinates — the fractions of the picture's width and height the detection
/// covered (`tower/object_memory/engine.py` divides by width and height in that
/// order). It says where in the *picture*, and the only rendering allowed for
/// it is one that says so.
nonisolated struct FrameReference: Equatable, Sendable {
    /// The capture this frame belongs to. `nil` when the record carries no
    /// capture provenance at all — an older record, not a zeroth session.
    let sessionID: String?
    /// Sequence number within that capture. `nil`, never `0`, when absent.
    let frameSeq: Int?
    /// The camera the frame came from, e.g. `glasses-camera`. Renamed from the
    /// store's `source` on the wire, because "source" invites being read as a
    /// provenance system.
    let camera: String
    let boundingBoxNormalized: [Double]?
    /// `capture-side`. Carried rather than assumed so that a Tower which
    /// changed where the imagery lives cannot have this app keep saying the old
    /// thing about it.
    let imageryRetention: String

    init(
        sessionID: String?,
        frameSeq: Int?,
        camera: String,
        boundingBoxNormalized: [Double]?,
        imageryRetention: String
    ) {
        self.sessionID = sessionID
        self.frameSeq = frameSeq
        self.camera = camera
        self.boundingBoxNormalized = boundingBoxNormalized
        self.imageryRetention = imageryRetention
    }

    /// Whether there is anything to point back at. A reference with neither a
    /// capture nor a frame number resolves to nothing, and saying "frame
    /// reference" over it would promise a pointer that does not exist.
    var pointsAtACapture: Bool { sessionID != nil || frameSeq != nil }
}

// MARK: - Observation

/// Which policy tier admitted a record, and what a second opinion said.
///
/// The old class list was `("laptop", "cell phone")`, chosen from a score
/// histogram — which describes the detector's opinion of itself. Reading the
/// crops found a ceiling fan detected as `airplane` at 0.99, a white door as
/// `refrigerator` at 0.95, and the three highest-scoring `remote` sightings in
/// 18,821 frames all laptop keyboards. So the list became a policy with tiers,
/// and this is the record's own note of which one let it through.
///
/// A `remembered` record was written on the detector's word. A `verify` record
/// was written **only because something agreed**, and `verification` names
/// which model and how strongly — so the record stays auditable and can be
/// re-evaluated when the model changes.
nonisolated struct ObservationVerification: Equatable, Sendable {
    /// Whether the verifier agreed. A verifier can only refuse or confirm: a
    /// verdict naming anything else is recorded as evidence and changes
    /// nothing.
    let agrees: Bool
    /// What was put to it.
    let proposed: String?
    /// What the verifier would have called it. **Not** used to relabel the
    /// record, and not shown as the record's class.
    let label: String?
    /// A similarity, from a model whose threshold was fitted to 94 crops from
    /// one home. **Not a calibrated probability**, and worded as one nowhere.
    let score: Double?
    /// Which model. The field that makes this auditable rather than
    /// reassuring.
    let model: String?
    let reason: String?

    init(
        agrees: Bool,
        proposed: String?,
        label: String?,
        score: Double?,
        model: String?,
        reason: String?
    ) {
        self.agrees = agrees
        self.proposed = proposed
        self.label = label
        self.score = score
        self.model = model
        self.reason = reason
    }
}

/// One record: a category was visible once, and this is when.
///
/// Every field is what the Tower sent. Nothing is derived, nothing is
/// defaulted, and no two records are related to each other — there is no
/// instance identity in this cartridge and this type must never grow one.
nonisolated struct ObjectObservation: Equatable, Identifiable, Sendable {
    /// One of the envelope's `recorded_classes`. A **category**.
    let objectClass: String
    /// The interpretation of the strength fields, and the one a consumer should
    /// read. Derived by the Tower from `bestScore`.
    let confidence: ObjectMemoryConfidence
    /// Provenance: how confident the detector was in the frame this record
    /// describes. Never revised. `nil` when it was not recorded.
    let detectorScore: Double?
    /// Evidence: the strongest score while that sighting stayed in view. `nil`
    /// means **not tracked**, not zero.
    let bestScore: Double?
    /// When the category came into view, qualified by `timeBasis`.
    let observedAt: Date
    /// `tower-receipt`. See `ObjectMemoryContract.towerReceiptBasis`.
    let timeBasis: String
    /// When the Tower stored it. The privacy-relevant clock, and the one
    /// retention filters on.
    let recordedAt: Date
    let moduleID: String
    let retentionTag: String
    /// `derived-only`, plus `frame-referenced` when a frame pointer is present.
    let privacyTags: [String]
    /// The frame reference. Not a location. See `FrameReference`.
    let frame: FrameReference

    /// The handle the imagery routes take: 16 hex characters, **derived** from
    /// `session_id`, `object_class` and `observed_at` rather than minted — so
    /// records written before the field existed have one, permanently, with no
    /// migration.
    ///
    /// Optional here even though the contract says every record carries it,
    /// and the asymmetry is deliberate. A record without a handle is a record
    /// whose picture cannot be asked for; refusing the *whole listing* over
    /// one such record would take a wearer's entire memory off screen to
    /// protect them from a missing thumbnail. So the picture is simply not
    /// offered for that row, and everything else about it is still shown.
    ///
    /// **Not an object identity.** It identifies a sighting, and two sightings
    /// of `laptop` are still not evidence about the same laptop.
    let observationID: String?
    /// When the sighting stopped. `observedAt` says when it started and never
    /// moves; this accumulates. `nil` on a record written before sightings had
    /// duration — never `0`, which would claim a sighting of no length.
    let lastSeenAt: Date?
    /// How many frames the sighting spanned. Same `nil` rule.
    let frameCount: Int?
    /// `remembered` — written on the detector's word — or `verify`, written
    /// only because something agreed. `nil` on a record written before tiers
    /// existed.
    let tier: String?
    /// What a second opinion said, or `nil` when nothing was asked — which is
    /// the ordinary case for a `remembered` class on a default Tower, where
    /// the verifier ships **off**.
    let verification: ObservationVerification?

    init(
        objectClass: String,
        confidence: ObjectMemoryConfidence,
        detectorScore: Double?,
        bestScore: Double?,
        observedAt: Date,
        timeBasis: String,
        recordedAt: Date,
        moduleID: String,
        retentionTag: String,
        privacyTags: [String],
        frame: FrameReference,
        observationID: String? = nil,
        lastSeenAt: Date? = nil,
        frameCount: Int? = nil,
        tier: String? = nil,
        verification: ObservationVerification? = nil
    ) {
        self.objectClass = objectClass
        self.confidence = confidence
        self.detectorScore = detectorScore
        self.bestScore = bestScore
        self.observedAt = observedAt
        self.timeBasis = timeBasis
        self.recordedAt = recordedAt
        self.moduleID = moduleID
        self.retentionTag = retentionTag
        self.privacyTags = privacyTags
        self.frame = frame
        self.observationID = observationID
        self.lastSeenAt = lastSeenAt
        self.frameCount = frameCount
        self.tier = tier
        self.verification = verification
    }

    /// A list identity: the Tower's own handle when the record carries one,
    /// and otherwise composed from the sighting's own coordinates.
    ///
    /// The two agree by construction — `observation_id` is derived from
    /// `session_id`, `object_class` and `observed_at`, which is what the
    /// fallback below joins — so preferring the wire's value costs nothing and
    /// means a row's identity and its imagery handle are the same string on
    /// every record that has one.
    ///
    /// **Not an object identity.** Two rows with different ids are two
    /// sightings, and two sightings of `laptop` are still not evidence about
    /// the same laptop. This exists so `ForEach` can tell two *rows* apart.
    var id: String {
        if let observationID { return observationID }
        var parts: [String] = [objectClass, String(observedAt.timeIntervalSince1970)]
        parts.append(frame.sessionID ?? "no-capture")
        if let frameSeq = frame.frameSeq {
            parts.append(String(frameSeq))
        } else {
            parts.append("no-frame")
        }
        return parts.joined(separator: "|")
    }
}

// MARK: - Retention

/// The window this read was allowed to see, as the Tower reported it.
///
/// **Narrowable, never widenable.** `requestedDays` is what was asked for;
/// `effectiveDays` is what will be honoured, clamped to
/// `min(persisted, requested)`. `clamped` is `true` only when the caller asked
/// for *more* than it received — which is why the clamp is reported rather than
/// merely applied: a client that asked for 3650 days and silently got 30 would
/// have no way to learn its question was refused.
///
/// `effectiveDays == nil` is **unbounded**, reachable only when the store
/// itself was written unbounded. It is never `0`: zero days would mean "nothing
/// is visible", which is the opposite claim.
nonisolated struct ObjectMemoryRetention: Equatable, Sendable {
    let requestedDays: Double?
    let effectiveDays: Double?
    let clamped: Bool
    let policy: String

    init(requestedDays: Double?, effectiveDays: Double?, clamped: Bool, policy: String) {
        self.requestedDays = requestedDays
        self.effectiveDays = effectiveDays
        self.clamped = clamped
        self.policy = policy
    }
}

// MARK: - Envelope

/// The header both endpoints carry, including the three claims that limit what
/// any of it may be rendered as.
nonisolated struct ObjectMemoryEnvelope: Equatable, Sendable {
    let contract: String
    let claim: String
    let identity: String
    let absenceMeans: String
    /// The universe of what could ever appear. A class outside this list has
    /// **never been looked for**, which is a weaker silence than "looked for
    /// and not seen", and the two are worded differently on screen.
    let recordedClasses: [String]
    let retention: ObjectMemoryRetention
    /// The class the request narrowed to. `nil` on an unfiltered listing.
    let objectClass: String?
    /// Where the pictures are — the three URL **templates**, under
    /// `object_memory.imagery/2026-08-27`.
    ///
    /// **A descriptor, not an availability claim.** Its presence says the
    /// routes exist; it says nothing about whether any particular record has a
    /// picture behind it, which only `/imagery` can answer per record.
    ///
    /// Optional because it is additive: a Tower on the same observations
    /// contract that predates the imagery routes sends no `imagery` block, and
    /// refusing its payload over a missing descriptor would take a working
    /// memory off screen to protect a feature that was never offered. `nil`
    /// means no pictures are offered anywhere, which the workspace renders as
    /// itself rather than as an error.
    let imagery: ObjectMemoryImageryRoutes?

    init(
        contract: String,
        claim: String,
        identity: String,
        absenceMeans: String,
        recordedClasses: [String],
        retention: ObjectMemoryRetention,
        objectClass: String?,
        imagery: ObjectMemoryImageryRoutes? = nil
    ) {
        self.contract = contract
        self.claim = claim
        self.identity = identity
        self.absenceMeans = absenceMeans
        self.recordedClasses = recordedClasses
        self.retention = retention
        self.objectClass = objectClass
        self.imagery = imagery
    }

    /// Whether a class is one the cartridge ever writes. Used to tell the two
    /// silences apart when the Tower has not already answered it.
    func isRecordable(_ objectClass: String) -> Bool {
        recordedClasses.contains(objectClass)
    }
}

// MARK: - Answers

/// `GET /object-memory/observations`.
nonisolated struct ObservationListing: Equatable, Sendable {
    let envelope: ObjectMemoryEnvelope
    /// Newest first by `observed_at`, as the Tower sorted them. Not re-sorted
    /// here: a second opinion about ordering is a second thing that can be
    /// wrong.
    let observations: [ObjectObservation]

    /// Throws when the Tower's own count disagrees with what it sent.
    ///
    /// A truncated array is a broken payload, and the safe direction from a
    /// broken payload is a failure — never a smaller number of records, which
    /// on this screen would read as "you saw fewer things than you did".
    init(envelope: ObjectMemoryEnvelope, observations: [ObjectObservation], reportedCount: Int) throws {
        guard reportedCount == observations.count else {
            throw CartridgeFailure(
                kind: .undecodableResponse,
                message: """
                    The Tower reported \(reportedCount) observations and sent \
                    \(observations.count), so the answer could not be read.
                    """
            )
        }
        self.envelope = envelope
        self.observations = observations
    }

    var isEmpty: Bool { observations.isEmpty }
}

/// `GET /object-memory/last-seen/{object_class}`.
///
/// `observed`, never `present`. There is no 404 case: "no record of a laptop"
/// answered as Not Found reads as "there is no laptop", which is a claim about
/// the world this cartridge cannot make.
nonisolated struct LastSeenAnswer: Equatable, Sendable {
    let envelope: ObjectMemoryEnvelope
    /// The class that was asked about, echoed by the Tower.
    let objectClass: String
    /// Whether this class is one the cartridge ever writes. `false` means its
    /// absence carries **no information at all**.
    let recordable: Bool
    /// Whether a record exists. Not whether the object is there.
    let observed: Bool
    /// The record, or `nil` when `observed` is false.
    let observation: ObjectObservation?

    /// Throws when `observed` and `observation` disagree.
    ///
    /// The same refusal `DocumentQueryResult` makes, for the same reason: a
    /// payload claiming a sighting while carrying no record is not a coherent
    /// answer, and quietly rewriting it to "nothing observed" would manufacture
    /// a negative statement about the wearer's memory out of a decode failure.
    init(
        envelope: ObjectMemoryEnvelope,
        objectClass: String,
        recordable: Bool,
        observed: Bool,
        observation: ObjectObservation?
    ) throws {
        if observed && observation == nil {
            throw CartridgeFailure(
                kind: .undecodableResponse,
                message: """
                    The Tower reported a sighting of \(objectClass) but sent no \
                    record, so the answer could not be read.
                    """
            )
        }
        if !observed && observation != nil {
            throw CartridgeFailure(
                kind: .undecodableResponse,
                message: """
                    The Tower sent a record of \(objectClass) while reporting \
                    nothing observed, so the answer could not be read.
                    """
            )
        }
        self.envelope = envelope
        self.objectClass = objectClass
        self.recordable = recordable
        self.observed = observed
        self.observation = observation
    }

    /// The hoisted `where`, taken from the record rather than from the
    /// payload's top-level copy.
    ///
    /// The Tower sends both and they are the same object. Reading the record's
    /// removes the possibility of this app rendering a frame reference that
    /// belongs to no record it is showing.
    var frame: FrameReference? { observation?.frame }
}

/// What was asked, so an answer can be shown next to its question.
enum ObjectMemoryQuestion: Equatable, Sendable {
    /// Every record in the window, or every record of one category.
    case listing(objectClass: String?)
    /// When one category was last in view — or an honest silence.
    case lastSeen(objectClass: String)

    var objectClass: String? {
        switch self {
        case .listing(let objectClass): return objectClass
        case .lastSeen(let objectClass): return objectClass
        }
    }
}

/// One answer to one question.
enum ObjectMemoryAnswer: Equatable, Sendable {
    case listing(ObservationListing)
    case lastSeen(LastSeenAnswer)

    var envelope: ObjectMemoryEnvelope {
        switch self {
        case .listing(let listing): return listing.envelope
        case .lastSeen(let answer): return answer.envelope
        }
    }
}

// MARK: - Decoding

/// Turns the two payloads into the types above, or refuses.
///
/// Written against `tower/tower/results/object_memory.py` field by field and
/// checked against the live route over the real 55-observation corpus, not
/// against the contract document alone. Where the two disagreed — the fourth
/// `confidence` value — the route won.
///
/// ## What it refuses
///
/// - A **contract mismatch**. Equality only; a different identifier is a
///   different agreement and is surfaced as `unsupportedContract`, not decoded
///   on a guess.
/// - A **changed claim**. If `claim`, `identity` or `absence_means` ever say
///   something else, the payload means something this build does not
///   understand, and rendering it under the old wording would be the worst
///   failure available here.
/// - A **populated `spatial_ref`**. The field is reserved and always null. A
///   value in it is a new capability this app has no permission to interpret,
///   and interpreting it would be exactly the "where in the room" claim the
///   whole cartridge is built to refuse.
/// - A **row that will not decode**, which drops the whole payload rather than
///   silently shrinking the answer. `WorldGeometryDecoder` makes the same
///   choice for the same reason.
nonisolated enum ObjectMemoryDecoder {

    /// `nil` when the payload is not this contract at all.
    ///
    /// Separate from the decode so a caller can tell "the Tower speaks a
    /// different Object Memory contract" from "the Tower sent something
    /// unreadable" — the first waits for an app update, the second is a
    /// failure, and telling a user to update when the payload was simply
    /// truncated wastes their time.
    static func contractIdentifier(from json: [String: Any]) -> String? {
        json["contract"] as? String
    }

    static func envelope(from json: [String: Any]) -> ObjectMemoryEnvelope? {
        guard
            let contract = json["contract"] as? String,
            contract == ObjectMemoryContract.identifier,
            let claim = json["claim"] as? String,
            claim == ObjectMemoryContract.categoryClaim,
            let identity = json["identity"] as? String,
            identity == ObjectMemoryContract.identityScope,
            let absenceMeans = json["absence_means"] as? String,
            absenceMeans == ObjectMemoryContract.absenceMeaning,
            isExplicitlyNull(json, key: "spatial_ref"),
            let recordedClasses = json["recorded_classes"] as? [String],
            let rawRetention = json["retention"] as? [String: Any],
            let retention = self.retention(from: rawRetention)
        else { return nil }

        return ObjectMemoryEnvelope(
            contract: contract,
            claim: claim,
            identity: identity,
            absenceMeans: absenceMeans,
            recordedClasses: recordedClasses,
            retention: retention,
            // Null on an unfiltered listing, and an Optional here rather than
            // "" — the two mean different things and only one of them is what
            // the Tower said.
            objectClass: json["object_class"] as? String,
            // Additive, and therefore Optional rather than required. It was
            // being **dropped on decode** until 2026-08-27, which is why iOS
            // showed a wearer `Frame reference: capture 22e9d428…, frame 3410`
            // and never the frame itself. A block that is present and
            // malformed still decodes to `nil` here rather than failing the
            // whole envelope: a listing that cannot show pictures is a
            // degraded screen, and a listing that shows nothing is a wearer
            // being told they have no memory.
            imagery: (json["imagery"] as? [String: Any])
                .flatMap(ObjectMemoryImageryDecoder.routes(from:))
        )
    }

    static func retention(from json: [String: Any]) -> ObjectMemoryRetention? {
        // A real Bool. The Tower pins this as a genuine boolean precisely
        // because `bool` subclasses `int` in Python and a `1` here fails every
        // `as? Bool`.
        guard let clamped = json["clamped"] as? Bool,
              let policy = json["policy"] as? String
        else { return nil }

        return ObjectMemoryRetention(
            // Both nullable, and both stay nil rather than becoming 0.
            // `effective_days == nil` is *unbounded*; `== 0` would be the
            // opposite claim and the Tower never sends it.
            requestedDays: json["requested_days"] as? Double,
            effectiveDays: json["effective_days"] as? Double,
            clamped: clamped,
            policy: policy
        )
    }

    static func frame(from json: [String: Any]) -> FrameReference? {
        guard
            let kind = json["kind"] as? String,
            kind == ObjectMemoryContract.frameReferenceKind,
            // Reserved, never populated, actively nulled on read. A value here
            // is refused rather than ignored: it would be a claim about where
            // something is, and this app has no honest way to show one.
            isExplicitlyNull(json, key: "spatial_ref"),
            let camera = json["camera"] as? String,
            let imageryRetention = json["imagery_retention"] as? String
        else { return nil }

        var box: [Double]?
        if let raw = json["bounding_box_normalized"] as? [Double] {
            // Four numbers or nothing. A short box is a broken box, and a
            // partially-drawn one is worse than none.
            guard raw.count == 4 else { return nil }
            box = raw
        }

        return FrameReference(
            sessionID: json["session_id"] as? String,
            frameSeq: json["frame_seq"] as? Int,
            camera: camera,
            boundingBoxNormalized: box,
            imageryRetention: imageryRetention
        )
    }

    /// Decodes one sighting only when its claim, identity and frame reference agree.
    ///
    /// Required wire fields and claim tokens are checked before constructing the
    /// record; optional evidence stays absent rather than becoming a false zero.
    static func observation(from json: [String: Any]) -> ObjectObservation? {
        guard
            let objectClass = json["object_class"] as? String,
            // Repeated per record so a client holding one record out of context
            // still has the claim. Checked per record for the same reason.
            let claim = json["claim"] as? String,
            claim == ObjectMemoryContract.categoryClaim,
            let identity = json["identity"] as? String,
            identity == ObjectMemoryContract.identityScope,
            let confidenceWord = json["confidence"] as? String,
            let confidence = ObjectMemoryConfidence(rawValue: confidenceWord),
            let observedAt = json["observed_at"] as? Double,
            let timeBasis = json["time_basis"] as? String,
            let recordedAt = json["recorded_at"] as? Double,
            let moduleID = json["module_id"] as? String,
            let retentionTag = json["retention_tag"] as? String,
            let privacyTags = json["privacy_tags"] as? [String],
            let rawFrame = json["where"] as? [String: Any],
            let frame = self.frame(from: rawFrame)
        else { return nil }

        return ObjectObservation(
            objectClass: objectClass,
            confidence: confidence,
            // Both nullable. `null` is "not recorded" and "not tracked"; a 0.0
            // substituted here would be a claim of no evidence.
            detectorScore: json["detector_score"] as? Double,
            bestScore: json["best_score"] as? Double,
            observedAt: Date(timeIntervalSince1970: observedAt),
            timeBasis: timeBasis,
            recordedAt: Date(timeIntervalSince1970: recordedAt),
            moduleID: moduleID,
            retentionTag: retentionTag,
            privacyTags: privacyTags,
            frame: frame,
            // All additive, all Optional, and none of them defaulted to a
            // number. `frame_count: nil` is "the record predates sightings
            // having a length"; a 0 would claim a sighting that spanned no
            // frames, which is not a thing that can have happened.
            observationID: json["observation_id"] as? String,
            lastSeenAt: (json["last_seen_at"] as? Double).map(Date.init(timeIntervalSince1970:)),
            frameCount: json["frame_count"] as? Int,
            tier: json["tier"] as? String,
            verification: (json["verification"] as? [String: Any])
                .flatMap(self.verification(from:))
        )
    }

    /// What a second opinion said. `nil` when the block is absent — the
    /// ordinary case — and also `nil` when it is present and unreadable,
    /// because a verification this app cannot parse is one it must not
    /// characterise, and "nothing was asked" is the weaker and safer of the
    /// two things it could say.
    static func verification(from json: [String: Any]) -> ObservationVerification? {
        // A real Bool. This is the one field in the block that carries a
        // verdict, and a `1` decoded as absent would drop an agreement.
        guard let agrees = json["agrees"] as? Bool else { return nil }
        return ObservationVerification(
            agrees: agrees,
            proposed: json["proposed"] as? String,
            label: json["label"] as? String,
            score: json["score"] as? Double,
            model: json["model"] as? String,
            reason: json["reason"] as? String
        )
    }

    /// Throws a `CartridgeFailure` when the payload is unreadable, so the one
    /// caller does not have to invent a message for `nil`.
    static func listing(from json: [String: Any]) throws -> ObservationListing {
        guard
            let envelope = self.envelope(from: json),
            let count = json["observation_count"] as? Int,
            let rawObservations = json["observations"] as? [[String: Any]]
        else { throw Self.unreadable }

        var observations: [ObjectObservation] = []
        for raw in rawObservations {
            guard let observation = self.observation(from: raw) else { throw Self.unreadable }
            observations.append(observation)
        }

        return try ObservationListing(
            envelope: envelope, observations: observations, reportedCount: count
        )
    }

    static func lastSeen(from json: [String: Any]) throws -> LastSeenAnswer {
        guard
            let envelope = self.envelope(from: json),
            let objectClass = json["object_class"] as? String,
            let recordable = json["recordable"] as? Bool,
            let observed = json["observed"] as? Bool
        else { throw Self.unreadable }

        var observation: ObjectObservation?
        if let raw = json["observation"] as? [String: Any] {
            guard let decoded = self.observation(from: raw) else { throw Self.unreadable }
            observation = decoded
        }

        return try LastSeenAnswer(
            envelope: envelope,
            objectClass: objectClass,
            recordable: recordable,
            observed: observed,
            observation: observation
        )
    }

    /// The key is present **and** its value is JSON null.
    ///
    /// Both halves matter. The contract sends `spatial_ref` as an explicit null
    /// rather than omitting it, precisely so a consumer can see the field
    /// exists and is empty — an absent key looks like version skew and invites
    /// a client to go looking for the value somewhere else.
    static func isExplicitlyNull(_ json: [String: Any], key: String) -> Bool {
        guard let value = json[key] else { return false }
        return value is NSNull
    }

    private static var unreadable: CartridgeFailure {
        CartridgeFailure(
            kind: .undecodableResponse,
            message: """
                The Tower's object memory answered in a shape this app does not \
                understand, so nothing is shown rather than something guessed.
                """
        )
    }
}
