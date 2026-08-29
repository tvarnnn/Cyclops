//
//  ObjectMemoryImagery.swift
//  Glasses
//

import Foundation

/// The pictures behind Object Memory's records.
///
/// ## The one thing to read before changing anything in this file
///
/// **A picture is a much stronger location cue than a sentence, and no string
/// test can catch it.** Everything `ObjectMemoryCopy` does to stop this
/// cartridge claiming to know where something is can be undone by putting a
/// first-person photograph of a desk on screen with a weak caption over it.
/// The caption carries the whole burden, `ObjectMemoryCopy.pictureCaption` is
/// therefore not optional anywhere, and no code path in this app may render
/// these bytes without it.
///
/// The evidence for serving the picture at all is direct rather than assumed.
/// MemPal — 15 adults aged 62–96, in their own homes, objects retrieved after
/// a 40-minute delay — measured its own last-seen images as showing the true
/// location only **53%** of the time, and users still went from 0.81 to 0.95
/// retrieval accuracy, searching 1.1 rooms instead of 1.9. A wrong-but-
/// plausible cue plus a human closes the gap; a confident sentence does not.
/// That is the argument for a picture *and* the argument for a caption that
/// says it may be wrong.
///
/// ## The filter, and what it is not called
///
/// `display-filter/yunet-2023mar@0.30`, `filter_means:
/// "applied-on-read-the-stored-frame-is-unchanged"`. It runs **on read**. The
/// raw frame stays exactly where it is — the capture manifests record
/// `redaction: "none"` and this cartridge does not own that tree. So it is a
/// **display filter**, and the words "redacted", "anonymised" and
/// "privacy-safe" do not appear in this cartridge's vocabulary and must not be
/// added to it. `ObjectMemoryCopyTests` asserts their absence.
///
/// A Tower whose weights are missing **serves nothing** (503). There is no
/// lenient default, because the lenient default here is a raw first-person
/// frame. **Do not work around a 503.** This file has no fallback path to an
/// unfiltered image and must never grow one.
///
/// ## Bytes are not cached, and that is enforced here rather than hoped for
///
/// Both binary routes send `Cache-Control: no-store`. A proxy or a URL cache
/// holding a copy is a second store nobody chose and nobody's retention
/// governs. `ObjectMemoryImageryHTTPClient` builds its own ephemeral session
/// with `urlCache = nil` and asks for
/// `.reloadIgnoringLocalAndRemoteCacheData` on every byte request, so honouring
/// the header does not depend on whichever `URLSession` a caller happened to
/// hand it.

// MARK: - The contract this build implements

/// The imagery agreement, and the values that travel with every payload under
/// it. Opaque identifier, equality only.
nonisolated enum ObjectMemoryImageryContract {
    static let identifier = "object_memory.imagery/2026-08-27"

    /// `claim`. A frame from a recording, filtered on the way out. **Not**
    /// evidence of where anything is — the same claim `where` already makes,
    /// restated on the pixels because the pixels are more persuasive.
    static let frameFromTheRecording = "frame-from-the-recording-this-record-was-derived-from"

    /// `filter_means`, when the bytes came out of the recording. Checked
    /// rather than carried: a payload that stopped saying the stored frame is
    /// unchanged would be describing a different system, and this app's wording
    /// about the filter would become false.
    static let appliedOnRead = "applied-on-read-the-stored-frame-is-unchanged"

    /// `filter_means`, when the bytes came out of Object Memory's **own**
    /// keyframe.
    ///
    /// The filter ran once, before that file was written, so there is no
    /// unfiltered copy of it anywhere — a stronger statement than the one
    /// above, not a weaker one. Both are true statements about different
    /// stores, which is why this is a second accepted value rather than a
    /// replacement.
    static let appliedBeforePersistence = "applied-before-this-file-was-written"

    /// Every `filter_means` this build knows how to word a sentence about.
    ///
    /// **A payload carrying anything else is refused, and that is deliberate.**
    /// This value is the licence for every sentence `ObjectMemoryCopy` writes
    /// about the filter; a meaning this build has never heard of would be
    /// rendered under the old sentences, which is the one failure worse than
    /// showing nothing. The list grew by one when Object Memory started owning
    /// its own crops — the identifier is unchanged
    /// (`object_memory.imagery/2026-08-27`), because the shape did not change
    /// and neither did the meaning of any existing field.
    static let everyFilterMeaning = [appliedOnRead, appliedBeforePersistence]

    /// `imagery_retention`. Carried rather than checked — see
    /// `ObjectMemoryImageryDescription.imageryRetention`.
    static let captureSideRetention = "capture-side"

    /// The token the Tower leaves in its URL templates.
    static let observationIDPlaceholder = "{observation_id}"
}

// MARK: - Routes

/// Where the pictures are, as the envelope describes them.
///
/// **Templates, not URLs, and that is the Tower being careful rather than
/// lazy.** The Tower does not know what host a phone reached it on, and a
/// client that already holds a base URL should not be handed a second,
/// possibly different one. So this type substitutes an id into a path and the
/// caller resolves it against the base URL it is already using.
///
/// Read off the payload rather than hard-coded, for the same reason
/// `recorded_classes` is: a Tower that moved these routes would otherwise
/// leave this app fetching a 404 from a path it invented.
nonisolated struct ObjectMemoryImageryRoutes: Equatable, Sendable {
    let contract: String
    let claim: String
    let filterMeans: String
    /// `/object-memory/observations/{observation_id}/imagery` — JSON.
    let viewTemplate: String
    /// `…/frame` — `image/jpeg`, the whole frame.
    let frameTemplate: String
    /// `…/crop` — `image/jpeg`, the object with 35% padding.
    let cropTemplate: String

    init(
        contract: String,
        claim: String,
        filterMeans: String,
        viewTemplate: String,
        frameTemplate: String,
        cropTemplate: String
    ) {
        self.contract = contract
        self.claim = claim
        self.filterMeans = filterMeans
        self.viewTemplate = viewTemplate
        self.frameTemplate = frameTemplate
        self.cropTemplate = cropTemplate
    }

    func template(for kind: ObjectMemoryImageryKind) -> String {
        switch kind {
        case .view: return viewTemplate
        case .frame: return frameTemplate
        case .crop: return cropTemplate
        }
    }

    /// The path for one record's picture, or `nil` when the template does not
    /// hold the placeholder it is supposed to.
    ///
    /// A template that lost its placeholder would otherwise resolve to a
    /// single shared URL, and every row on screen would show the same picture
    /// — which on this cartridge would be a photograph attributed to the wrong
    /// record. Refusing is the only safe direction.
    func path(for kind: ObjectMemoryImageryKind, observationID: String) -> String? {
        let template = self.template(for: kind)
        guard template.contains(ObjectMemoryImageryContract.observationIDPlaceholder) else {
            return nil
        }
        // The handle is 16 hex characters today. Encoded anyway: a path
        // component built by substitution is a path component, and this app
        // does not get to assume what a future handle may contain.
        let encoded =
            observationID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)
            ?? observationID
        return template.replacingOccurrences(
            of: ObjectMemoryImageryContract.observationIDPlaceholder, with: encoded
        )
    }

    /// The absolute URL, resolved against the Tower this app is already
    /// talking to.
    func url(
        for kind: ObjectMemoryImageryKind, observationID: String, relativeTo baseURL: URL
    ) -> URL? {
        guard let path = self.path(for: kind, observationID: observationID) else { return nil }
        return URL(string: path, relativeTo: baseURL)?.absoluteURL
    }
}

/// Which store the bytes came out of.
///
/// ## Why a client has to know, when it never had to before
///
/// `/crop` and `/frame` used to be two renders of the same file, so a record's
/// object picture and its context picture lived and died together. Object
/// Memory now owns a small filtered crop per record, under its **own**
/// retention, so `/crop` keeps answering after the recording behind it has been
/// deleted while `/frame` honestly 410s. The two pictures have different
/// lifetimes, and every sentence about where a picture is kept and how long it
/// lasts is now a sentence about one of them rather than about both.
///
/// A `RawRepresentable` struct rather than an `enum`, for the reason
/// `ObjectMemoryImageryReason` is one: a source this build has never heard of
/// must survive the decode and reach the screen described as unrecognised,
/// rather than failing a parse and rendering as a broken Tower.
nonisolated struct ObjectMemoryImagerySource: RawRepresentable, Equatable, Sendable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    /// A frame out of the recording, filtered on the way out. Governed by
    /// capture-side retention, which this cartridge neither sets nor enforces.
    static let captureFrame = ObjectMemoryImagerySource(rawValue: "capture-frame")

    /// Object Memory's own keyframe, filtered before it was written. Governed
    /// by **this cartridge's** retention: it goes when the record expires or is
    /// purged, and it survives the recording it came from.
    static let objectMemoryKeyframe =
        ObjectMemoryImagerySource(rawValue: "object-memory-keyframe")
}

/// Which of the three imagery routes.
nonisolated enum ObjectMemoryImageryKind: String, Equatable, Sendable, CaseIterable {
    /// JSON: whether there is a picture and what may be said about it.
    /// Answerable **without** downloading anything, which is the point — a
    /// phone deciding between a thumbnail, a caption and "the memory is kept
    /// and the picture is gone" should not have to fetch an image to find out
    /// which.
    case view
    /// The whole frame. The **context**, not the object: where the wearer was
    /// and what else was in view is most of what makes a small crop
    /// recognisable.
    case frame
    /// The object, padded 35%. A tight crop of a 3%-of-frame object is
    /// unreadable.
    case crop
}

// MARK: - Refusal reasons

/// Why a picture could not be served.
///
/// **A value to switch on, never a sentence to display** — the Tower is
/// explicit about that, and the wording belongs to whoever is speaking to the
/// wearer, which is `ObjectMemoryCopy`.
///
/// A `RawRepresentable` struct rather than an `enum` so a reason this build has
/// never heard of survives the decode and reaches the screen as an
/// unrecognised refusal, rather than failing a parse and rendering as a
/// connection error — which is precisely the class of bug this file exists to
/// fix.
nonisolated struct ObjectMemoryImageryReason: RawRepresentable, Equatable, Sendable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    /// **404.** The handle matched nothing *within retention*. Retention is not
    /// bypassed by knowing an id: a handle resolves through the same clamped
    /// read the listing uses, so an expired record is unreachable by its own id.
    static let noSuchObservation = ObjectMemoryImageryReason(rawValue: "no-such-observation")
    /// **404.** The record never had a pointer at all.
    static let noFrameReference =
        ObjectMemoryImageryReason(rawValue: "record-has-no-frame-reference")
    /// **410. The pointer is intact and the picture is gone.** Capture-side
    /// retention removed it, `memory_retained: true` is in the body, and this
    /// is the case the whole payload shape exists for.
    static let imageryNoLongerAvailable =
        ObjectMemoryImageryReason(rawValue: "imagery-no-longer-available")
    /// **503.** No face-detection weights, or the filter failed. Nothing is
    /// served, and there is no lenient default.
    static let displayFilterUnavailable =
        ObjectMemoryImageryReason(rawValue: "display-filter-unavailable")
    /// **503.** This Tower has nowhere to look.
    static let noCaptureRootConfigured =
        ObjectMemoryImageryReason(rawValue: "no-capture-root-configured")
    /// **503.** The file is there and could not be decoded.
    static let frameUnreadable = ObjectMemoryImageryReason(rawValue: "frame-unreadable")

    /// The three that mean *this Tower cannot serve any picture right now*, as
    /// against the three that are about one record. They are a configuration
    /// answer and are worded as one — telling a person their memory has no
    /// picture, when in fact the Tower has no weights installed, would be a
    /// claim about their data made from a fact about a machine.
    static let towerCannotServeAny: [ObjectMemoryImageryReason] = [
        .displayFilterUnavailable, .noCaptureRootConfigured, .frameUnreadable,
    ]
}

/// What a description means, once the reason has been read.
///
/// The route's own docstring is the instruction here: *"The reason VALUE is on
/// the body in every case, and a client should switch on that rather than on
/// the code."* So this is derived from `reason`, and the status code is carried
/// alongside for diagnosis rather than used to decide.
nonisolated enum ObjectMemoryImagerySituation: Equatable, Sendable {
    /// Bytes can be fetched.
    case aPicture
    /// **The memory is kept and the picture is gone.** Never a broken image,
    /// never an empty row, and never a connection error.
    case thePictureIsGone
    /// The record never carried a frame pointer, so there was never a picture
    /// to keep. Different from the case above and worded differently.
    case theRecordNeverPointedAtAFrame
    /// The handle matched nothing within retention.
    case noSuchRecord
    /// This Tower cannot serve any picture at all right now.
    case theTowerServesNoPictures(ObjectMemoryImageryReason)
    /// A reason this build does not recognise. Shown uninterpreted rather than
    /// folded into the nearest one we know.
    case anUnrecognisedRefusal(ObjectMemoryImageryReason)
}

// MARK: - The description

/// What the Tower knows about the imagery behind one record.
///
/// Answerable without downloading anything. Every field here is what the Tower
/// sent; nothing is derived and nothing is defaulted.
nonisolated struct ObjectMemoryImageryDescription: Equatable, Sendable {
    let contract: String
    let observationID: String
    /// The class of the record this picture belongs to. `nil` when the handle
    /// matched nothing, because then there is no record to have a class.
    let objectClass: String?
    let claim: String

    /// Whether bytes can be served.
    ///
    /// **This is now a claim about the picture this cartridge can serve**, not
    /// about a `/frame` render. It used to be the second, and that was a
    /// blocker: a record with an owned keyframe reported `available: false`
    /// whenever the recording behind it was gone, and
    /// `ObjectMemoryPictureLoader.fetch` gates every byte request on this
    /// field — so the crop being held for that record could never be asked
    /// for. Fixed on the Tower; recorded here because this field's meaning is
    /// what the gate depends on.
    let available: Bool

    /// Whether `/frame` — the wider context view — could be served.
    ///
    /// **`Bool?`, and `nil` is not `false`.** `nil` means the Tower did not
    /// compute it: it is null on the refusal details the binary routes raise,
    /// and on every Tower older than this field. `false` is a positive claim
    /// that the recording behind this record is gone, which is a normal,
    /// expected outcome now that the crop outlives it.
    ///
    /// The same three-valued discipline `following_this_session` gets in
    /// `CartridgeSessionSnapshot`, and for the same reason: folding "it said
    /// no" together with "it did not say" would make an older Tower behave as
    /// though every context frame had expired.
    let frameAvailable: Bool?

    /// Which store served the bytes. `nil` on a Tower that does not say.
    ///
    /// Read rather than inferred from `kind`: `/crop` prefers the owned
    /// keyframe and falls back to a capture frame, so the route asked for does
    /// not determine the store answered from, and the retention sentence is
    /// written from the store.
    let imagerySource: ObjectMemoryImagerySource?

    /// `nil` when available. A value, never a sentence.
    let reason: ObjectMemoryImageryReason?
    /// **The field this shape exists for.** `true` with `available: false`
    /// means the record is still here and its picture is not.
    let memoryRetained: Bool

    /// `display-filter/yunet-2023mar@0.30`. `nil` when nothing was filtered
    /// because nothing was served. Carried rather than assumed so a Tower that
    /// changed detector or threshold cannot leave this app naming the old one.
    let filter: String?
    /// `applied-on-read-the-stored-frame-is-unchanged`.
    let filterMeans: String
    /// How many regions the filter filled. **Zero means nothing was detected**,
    /// not that there was nothing there — YuNet has measured blind spots (a
    /// face occluded past ~60%, rotated ~90° in plane, profile and rear views)
    /// and a reader must be able to tell the two apart.
    let regionsFilled: Int
    /// How much of the record's **own box** the filter covered, 0.0–1.0.
    ///
    /// This exists for a measured defect rather than for completeness. The
    /// filter fires on 40.2% of real corpus frames; of 36 firings inspected by
    /// eye, 4 were a real face and 32 were hands on a keyboard, a screen, a
    /// door or a sink. On frame 2708 of the physically validated capture — a
    /// desk with no person in it — one fill landed squarely on the mouse the
    /// record was about.
    ///
    /// The filter was **not** weakened in response, and must not be: a
    /// face-detection threshold is not a picture-quality knob, and trading
    /// detection sensitivity for a nicer thumbnail is the one trade a privacy
    /// filter may never make. The overlap is reported instead, and a client
    /// that knows the subject was covered says so or shows the context frame.
    let subjectObscured: Double
    /// Where in the **picture**, in fractions of the frame. Same caveat as
    /// `FrameReference.boundingBoxNormalized`: never a world position.
    let boundingBoxNormalized: [Double]?
    /// What the Tower calls the retention governing these bytes.
    ///
    /// Carried verbatim and **not** checked against a constant, unlike
    /// `filterMeans`: no sentence in this app is licensed by this string's
    /// value. The retention sentence a reader sees is written from
    /// `imagerySource` instead, because the source is the thing that actually
    /// determines the lifetime — an owned keyframe goes when the record does,
    /// a capture frame goes when capture-side retention says so — and a client
    /// that branched on this label would be branching on wording.
    let imageryRetention: String

    /// The HTTP status this arrived under.
    ///
    /// Carried for diagnosis, and deliberately **not** what the rendering
    /// switches on. It is here so a test can pin that a 410 really did carry
    /// `imagery-no-longer-available` and `memory_retained: true`, rather than
    /// this app inferring one of those from the other.
    let statusCode: Int

    init(
        contract: String,
        observationID: String,
        objectClass: String?,
        claim: String,
        available: Bool,
        // Defaulted so every construction written before these fields existed
        // still compiles and still means what it meant: `nil` is "the Tower did
        // not say", which is exactly what an older payload conveys.
        frameAvailable: Bool? = nil,
        imagerySource: ObjectMemoryImagerySource? = nil,
        reason: ObjectMemoryImageryReason?,
        memoryRetained: Bool,
        filter: String?,
        filterMeans: String,
        regionsFilled: Int,
        subjectObscured: Double,
        boundingBoxNormalized: [Double]?,
        imageryRetention: String,
        statusCode: Int
    ) {
        self.contract = contract
        self.observationID = observationID
        self.objectClass = objectClass
        self.claim = claim
        self.available = available
        self.frameAvailable = frameAvailable
        self.imagerySource = imagerySource
        self.reason = reason
        self.memoryRetained = memoryRetained
        self.filter = filter
        self.filterMeans = filterMeans
        self.regionsFilled = regionsFilled
        self.subjectObscured = subjectObscured
        self.boundingBoxNormalized = boundingBoxNormalized
        self.imageryRetention = imageryRetention
        self.statusCode = statusCode
    }

    /// What this means, switched on `reason` exactly as the route asks.
    var situation: ObjectMemoryImagerySituation {
        guard let reason else {
            // `available` and a null reason travel together. If they ever came
            // apart, the safe direction is to believe the reason: a picture
            // rendered from a payload that gave a reason not to would be the
            // failure this whole type is arranged to prevent.
            return available ? .aPicture : .anUnrecognisedRefusal(ObjectMemoryImageryReason(rawValue: ""))
        }
        switch reason {
        case .imageryNoLongerAvailable: return .thePictureIsGone
        case .noFrameReference: return .theRecordNeverPointedAtAFrame
        case .noSuchObservation: return .noSuchRecord
        case _ where ObjectMemoryImageryReason.towerCannotServeAny.contains(reason):
            return .theTowerServesNoPictures(reason)
        default: return .anUnrecognisedRefusal(reason)
        }
    }

    /// Whether part of the record's own subject is behind a fill.
    ///
    /// `> 0`, not a threshold: any overlap at all means the thing the record is
    /// about is partly covered, and the honest response is to say so or to
    /// show the context frame instead of the crop.
    var subjectIsBehindAFill: Bool { subjectObscured > 0 }

    /// Whether the wider context view can be asked for at all.
    ///
    /// `true` when the Tower said so **and** when it said nothing: `nil` is
    /// "not computed", and an unknown answer is a reason to offer the request
    /// and let the Tower answer it, not a reason to withhold a picture that is
    /// probably there. Only an explicit `false` closes the door, and it closes
    /// it on a claim the Tower actually made.
    var frameCanBeAskedFor: Bool { frameAvailable != false }

    /// Which route to fetch for this record, given what the filter did to it
    /// and what is left to fetch.
    ///
    /// The crop is the better picture of an object and the worse one when a
    /// fill is sitting on that object — the contract's instruction is "say the
    /// subject is behind a fill, **or fall back to `/frame`**", and doing both
    /// is strictly better than either. So the fallback is automatic and the
    /// sentence is shown as well.
    ///
    /// **The fallback is now conditional on the frame existing.** It was not,
    /// and once `/crop` started outliving the recording behind it that was a
    /// fallback *away from* a picture that is being held and *towards* one that
    /// is gone: a wearer with an obscured subject would have been shown "the
    /// memory is kept and the picture is gone" over a crop the Tower was
    /// keeping for them. `subjectObscuredLine` still says the subject is partly
    /// behind a fill, which is the half of the contract's instruction that
    /// always applies.
    ///
    /// An older Tower sends no `frame_available`, `frameCanBeAskedFor` is
    /// `true`, and this is exactly the expression it always was.
    var preferredKind: ObjectMemoryImageryKind {
        subjectIsBehindAFill && frameCanBeAskedFor ? .frame : .crop
    }
}

// MARK: - Answers

/// What asking about a record's imagery produced.
///
/// A refusal is **described**, not failed: 410, 404 and 503 all arrive with a
/// full description in the body and all three are things this app can say out
/// loud. Only the last two cases are failures, and neither of them is a status
/// code.
nonisolated enum ObjectMemoryImageryAnswer: Equatable, Sendable {
    /// The Tower described the imagery behind a record — available or not.
    case described(ObjectMemoryImageryDescription)
    /// Nothing has been asked yet, so this app holds no route templates. Not a
    /// failure: the templates arrive on an observations envelope, and a
    /// workspace that has not queried has not been told where the pictures are.
    case routesUnknown
    /// The request did not complete.
    case unreachable(String)
    /// The answer arrived and could not be read as this contract.
    case undecodable
}

/// What asking for the bytes produced.
nonisolated enum ObjectMemoryPictureAnswer: Equatable, Sendable {
    /// JPEG bytes. **Not cached anywhere**, and held only for as long as a row
    /// is on screen.
    case picture(Data)
    /// The Tower refused, and said why in a body this app can render as a
    /// sentence. Reachable even after a `.described` that said `available` —
    /// capture-side retention can close between the two requests, and the
    /// answer to that race is this case rather than a broken image.
    case refused(ObjectMemoryImageryDescription)
    case routesUnknown
    case unreachable(String)
    case undecodable
}

// MARK: - Decoding

/// Turns imagery payloads into the types above, or refuses.
///
/// Written against the running Tower's bytes: the refusal bodies arrive inside
/// FastAPI's `{"detail": …}` wrapper, which the field tables do not mention,
/// and they carry `observation_id` and `object_class`, which the tables do not
/// list. Both were read off the wire.
nonisolated enum ObjectMemoryImageryDecoder {

    /// The `imagery` block on an observations envelope.
    ///
    /// Refuses on a changed `claim` or an unknown `filter_means` for the reason
    /// the observation decoder refuses a changed `claim`: those two values are
    /// what this app's entire wording about pictures and about the filter is
    /// licensed by, and rendering new meanings under the old sentences is worse
    /// than showing nothing.
    ///
    /// `filter_means` is checked against `everyFilterMeaning` rather than
    /// against one constant, because there are now two true statements to make
    /// — the filter ran on read for a capture frame, and before persistence for
    /// an owned keyframe — and this app has a sentence for each. Widening a
    /// *membership* test is not the same as dropping it: a third meaning still
    /// refuses.
    static func routes(from json: [String: Any]) -> ObjectMemoryImageryRoutes? {
        guard
            let contract = json["contract"] as? String,
            contract == ObjectMemoryImageryContract.identifier,
            let claim = json["claim"] as? String,
            claim == ObjectMemoryImageryContract.frameFromTheRecording,
            let filterMeans = json["filter_means"] as? String,
            ObjectMemoryImageryContract.everyFilterMeaning.contains(filterMeans),
            let view = json["view"] as? String,
            let frame = json["frame"] as? String,
            let crop = json["crop"] as? String
        else { return nil }

        return ObjectMemoryImageryRoutes(
            contract: contract,
            claim: claim,
            filterMeans: filterMeans,
            viewTemplate: view,
            frameTemplate: frame,
            cropTemplate: crop
        )
    }

    /// One `/imagery` body, or one refusal body peeled out of `detail`.
    static func description(
        from json: [String: Any], statusCode: Int
    ) -> ObjectMemoryImageryDescription? {
        guard
            let contract = json["contract"] as? String,
            contract == ObjectMemoryImageryContract.identifier,
            let observationID = json["observation_id"] as? String,
            let claim = json["claim"] as? String,
            claim == ObjectMemoryImageryContract.frameFromTheRecording,
            // Real Bools, both of them, and `memory_retained` most of all: it
            // is the difference between "your memory is gone" and "the picture
            // is gone", and a `1` decoded as a `false` would say the first
            // about a record that is still there.
            let available = json["available"] as? Bool,
            let memoryRetained = json["memory_retained"] as? Bool,
            let filterMeans = json["filter_means"] as? String,
            // Two accepted meanings, one per store — see `everyFilterMeaning`.
            // A payload whose filter meaning this build has never heard of is
            // still refused, because the sentence beside the picture would be
            // describing a transformation nobody here has read about.
            ObjectMemoryImageryContract.everyFilterMeaning.contains(filterMeans),
            let regionsFilled = json["regions_filled"] as? Int,
            let subjectObscured = json["subject_obscured"] as? Double,
            let imageryRetention = json["imagery_retention"] as? String
        else { return nil }

        var box: [Double]?
        if let raw = json["bounding_box_normalized"] as? [Double] {
            // Four numbers or nothing, exactly as `FrameReference` requires. A
            // partially-drawn box over a photograph is worse than no box.
            guard raw.count == 4 else { return nil }
            box = raw
        }

        return ObjectMemoryImageryDescription(
            contract: contract,
            observationID: observationID,
            objectClass: json["object_class"] as? String,
            claim: claim,
            available: available,
            // **`as? Bool` on a missing key and on an explicit null both give
            // `nil`, and that is the wanted behaviour here** — both mean the
            // Tower did not state whether the context frame is there. What must
            // not happen is `?? false`: an older Tower would then report every
            // record's context view as expired, and the "Show the whole frame"
            // control would disappear from a screen where it works.
            frameAvailable: json["frame_available"] as? Bool,
            // Unrecognised values survive rather than failing the parse — see
            // `ObjectMemoryImagerySource`. A `nil` here is an older Tower, and
            // the retention sentence says only what such a Tower supports.
            imagerySource: (json["imagery_source"] as? String)
                .map(ObjectMemoryImagerySource.init(rawValue:)),
            // `nil` when a picture was served, and an Optional rather than ""
            // — the two mean different things.
            reason: (json["reason"] as? String).map(ObjectMemoryImageryReason.init(rawValue:)),
            memoryRetained: memoryRetained,
            // Null when nothing was served, because nothing was filtered.
            filter: json["filter"] as? String,
            filterMeans: filterMeans,
            regionsFilled: regionsFilled,
            subjectObscured: subjectObscured,
            boundingBoxNormalized: box,
            imageryRetention: imageryRetention,
            statusCode: statusCode
        )
    }

    /// A body that may or may not be wrapped in FastAPI's `detail`.
    ///
    /// 200 sends the description at the top level; 404, 410 and 503 send it
    /// inside `detail`, because the route raises an `HTTPException` for each.
    /// Unwrapping here rather than at three call sites is what stops one of
    /// them being forgotten — and a forgotten one is a 410 rendered as an
    /// unreadable answer, which is most of the way back to the bug this file
    /// was written to fix.
    static func descriptionUnwrapping(
        _ json: [String: Any], statusCode: Int
    ) -> ObjectMemoryImageryDescription? {
        if let description = description(from: json, statusCode: statusCode) {
            return description
        }
        guard let detail = json["detail"] as? [String: Any] else { return nil }
        return description(from: detail, statusCode: statusCode)
    }
}

// MARK: - HTTP

/// The three imagery routes.
///
/// ## Its own `URLSession`, and why that is not over-engineering
///
/// `ObjectMemoryHTTPClient` takes an injected session defaulting to `.shared`,
/// which is fine for JSON. These are **first-person photographs**, both binary
/// routes send `Cache-Control: no-store`, and `URLSession.shared` has a disk
/// cache. Honouring a no-store header must not depend on which session a caller
/// happened to pass, so this type builds an **ephemeral** configuration with
/// `urlCache = nil` when it is not given one, and every byte request also asks
/// for `.reloadIgnoringLocalAndRemoteCacheData`. Belt and braces, because one
/// of the two is a default someone can change from outside this file.
nonisolated struct ObjectMemoryImageryHTTPClient {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession
    /// A frame read costs a disk read, a JPEG decode, a face detection and a
    /// re-encode on the Tower. Longer than the query client's ten seconds
    /// because the work is real, and still bounded (Rule 15).
    var timeout: TimeInterval = 20

    /// The session a caller gets when it does not bring one: ephemeral, with
    /// no URL cache at all.
    ///
    /// `URLSessionConfiguration.ephemeral` already keeps nothing on disk;
    /// `urlCache = nil` removes the in-memory one as well, so there is no
    /// copy of a wearer's first-person frame anywhere in this process that
    /// outlives the view holding it.
    static func uncachedSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        return URLSession(configuration: configuration)
    }

    init(
        baseURL: URL = TowerConfiguration.httpBaseURL,
        session: URLSession? = nil,
        timeout: TimeInterval = 20
    ) {
        self.baseURL = baseURL
        self.session = session ?? Self.uncachedSession()
        self.timeout = timeout
    }

    /// Whether there is a picture, and what may be said about it.
    ///
    /// **200 even when there is no picture.** The resource — what this
    /// cartridge knows about the imagery behind a record — exists either way,
    /// and only an unknown handle is a real 404 here. So almost every call
    /// returns `.described`, and the caller reads `situation` rather than a
    /// status code.
    func description(
        for observationID: String, routes: ObjectMemoryImageryRoutes
    ) async -> ObjectMemoryImageryAnswer {
        guard
            let url = routes.url(
                for: .view, observationID: observationID, relativeTo: baseURL
            )
        else { return .routesUnknown }

        do {
            let (data, status) = try await fetch(url)
            let parsed = try? JSONSerialization.jsonObject(with: data)
            guard
                let json = parsed as? [String: Any],
                let described = ObjectMemoryImageryDecoder.descriptionUnwrapping(
                    json, statusCode: status
                )
            else { return .undecodable }
            return .described(described)
        } catch {
            return .unreachable(error.localizedDescription)
        }
    }

    /// The bytes, or the reason there are none.
    ///
    /// **A non-2xx here is decoded, not thrown.** This is the defect this file
    /// was written to remove: a 410 turned into "The Tower answered 410" and
    /// mapped onto a transport failure tells a wearer their Tower is
    /// unreachable, when in fact the Tower answered clearly and truthfully that
    /// capture-side retention has taken the picture and kept the memory. That
    /// is worse than a broken image, because it sends someone to check a
    /// network cable about a working machine.
    func picture(
        for observationID: String,
        kind: ObjectMemoryImageryKind,
        routes: ObjectMemoryImageryRoutes
    ) async -> ObjectMemoryPictureAnswer {
        guard
            kind != .view,
            let url = routes.url(
                for: kind, observationID: observationID, relativeTo: baseURL
            )
        else { return .routesUnknown }

        do {
            let (data, status) = try await fetch(url)
            if (200...299).contains(status) {
                // Empty bytes with a 200 is not a picture. Refusing beats
                // handing `UIImage` something it will silently fail to decode
                // and leave as a blank rectangle, which on this screen would
                // read as "there was nothing there".
                guard !data.isEmpty else { return .undecodable }
                return .picture(data)
            }
            let parsed = try? JSONSerialization.jsonObject(with: data)
            guard
                let json = parsed as? [String: Any],
                let described = ObjectMemoryImageryDecoder.descriptionUnwrapping(
                    json, statusCode: status
                )
            else { return .undecodable }
            return .refused(described)
        } catch {
            return .unreachable(error.localizedDescription)
        }
    }

    /// One request, returning the body and the status without judging either.
    ///
    /// Deliberately does not throw on a non-2xx: on these three routes a
    /// non-2xx body is the answer, and a transport layer that converts status
    /// codes into errors is a transport layer that has to be talked out of it
    /// by every caller.
    private func fetch(_ url: URL) async throws -> (Data, Int) {
        let request = URLRequest(
            url: url,
            cachePolicy: .reloadIgnoringLocalAndRemoteCacheData,
            timeoutInterval: timeout
        )
        let (data, response) = try await session.data(for: request)
        return (data, (response as? HTTPURLResponse)?.statusCode ?? 0)
    }
}
