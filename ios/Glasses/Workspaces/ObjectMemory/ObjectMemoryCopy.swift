//
//  ObjectMemoryCopy.swift
//  Glasses
//

import Foundation

/// Every sentence the Object Memory workspace can put in front of a person.
///
/// ## Why the copy is a type
///
/// Because the copy *is* the product constraint here, and prose scattered
/// through a `View` cannot be tested. This cartridge knows less than a reader
/// will assume it knows, and almost every natural phrasing of what it holds is
/// a lie:
///
/// | The natural sentence | Why it is false |
/// |---|---|
/// | "Your laptop was seen at 2:31" | It is a **category**. Nothing here can tell the wearer's laptop from anyone else's, and `identity: "category-not-instance"` says so on every record. |
/// | "Last seen in session 22e9d4…" | A session id is a pointer into a *recording*. Rendered as an answer to "where", it reads as a place, and this cartridge does not know where anything is. |
/// | "Your laptop is on the desk" | A record says a category was visible **once**. `claim: "category-was-visible-once"`. It says nothing about now. |
/// | "No laptop found" | An empty answer is `absence_means: "not-observed-by-this-cartridge"` — about what the camera captured, never about what exists. |
///
/// So every user-facing string is produced here, `ObjectMemoryWorkspaceView`
/// writes none of its own, and `ObjectMemoryCopyTests` runs the whole output of
/// `everyString(for:)` through the claims it is forbidden to make. A view that
/// composes its own sentence escapes that test, which is the one thing this
/// file exists to prevent.
///
/// ## The rules the strings obey
///
/// 1. **Indefinite article, always.** "A laptop was visible", never "your
///    laptop" and never "the laptop".
/// 2. **Past tense, always.** "was visible", never "is", never "is still".
/// 3. **A frame reference is labelled as one**, in the same sentence as the
///    identifiers, and is never presented as an answer to "where is it".
/// 4. **An empty answer says whose silence it is.** The cartridge's, not the
///    world's.
/// 5. **`null` is said out loud.** "not recorded" and "not tracked" are shown
///    as themselves; no number stands in for a missing one.
enum ObjectMemoryCopy {

    // MARK: - What the cartridge is

    static let cartridgeName = "Object Memory"

    /// The header. Sets the reader's expectations *down* to what is true,
    /// before any record is shown — the cheapest place to do it.
    static let summary = """
        This records that a category of thing was visible to the camera once, \
        and when. It does not know where anything is, it cannot tell one object \
        from another of the same kind, and no record says anything about now.
        """

    /// What the cartridge could ever have written down, so that a silence about
    /// anything else can be read for what it is.
    static func recordableClasses(_ classes: [String]) -> String {
        guard !classes.isEmpty else {
            return "This Tower's object memory records no categories at all."
        }
        return """
            Categories this memory can record: \(list(classes)). Anything else \
            has never been looked for.
            """
    }

    // MARK: - A record

    /// The headline for one sighting.
    ///
    /// Indefinite article and past tense, and the two together are the whole
    /// point: "A laptop was visible" is exactly as much as a record supports.
    static func sightingHeadline(_ observation: ObjectObservation) -> String {
        "\(indefiniteArticle(for: observation.objectClass).capitalized) \(observation.objectClass) was visible"
    }

    /// When, and on whose clock.
    ///
    /// Never rendered as a shutter time. `time_basis` is `tower-receipt`, and
    /// `Rule 16` forbids conflating when the glasses saw something with when
    /// the Tower received it — so the qualification travels in the same
    /// sentence as the timestamp rather than in a footnote nobody reads.
    static func timeLine(_ observation: ObjectObservation) -> String {
        let when = observation.observedAt.formatted(date: .abbreviated, time: .shortened)
        guard observation.timeBasis == ObjectMemoryContract.towerReceiptBasis else {
            // A basis this build does not recognise is shown uninterpreted. The
            // alternative — assuming it means the same as the one we know — is
            // how a receipt time quietly becomes a capture time.
            return """
                Timestamped \(when), on a basis this app does not recognise \
                (\(observation.timeBasis)), so it is shown without interpretation.
                """
        }
        return """
            The Tower received this frame on \(when). That is the Tower's \
            receipt time, not the moment the shutter fired.
            """
    }

    /// The claim, restated beside every record.
    ///
    /// Repeated per row rather than once at the top because a row is what gets
    /// screenshotted, read out of context, and remembered.
    static func claimLine(_ observation: ObjectObservation) -> String {
        """
        A category was in view once. That is the whole claim: it does not say \
        anything about now, and it cannot tell one \(observation.objectClass) \
        from another.
        """
    }

    /// The frame reference, labelled in the same breath as its identifiers.
    ///
    /// The failure this wording exists to avoid is "Your laptop: last seen in
    /// session 22e9d4…" — which reads as a place, and is the reason
    /// `ObjectMemoryCopyTests` asserts that no string can mention a capture id
    /// without also saying "Frame reference".
    static func frameLine(_ frame: FrameReference) -> String {
        guard frame.pointsAtACapture else {
            return """
                Frame reference: none. This record carries no capture \
                provenance, so there is nothing to point back at.
                """
        }
        var parts: [String] = []
        if let sessionID = frame.sessionID { parts.append("capture \(shortened(sessionID))") }
        if let frameSeq = frame.frameSeq { parts.append("frame \(frameSeq)") }
        parts.append("camera \(frame.camera)")
        return "Frame reference: " + parts.joined(separator: ", ") + "."
    }

    /// Said next to every frame reference, without exception.
    static let frameCaveat = """
        That is a pointer back into a recording, not a place. Nothing in this \
        memory knows where anything is in a room.
        """

    /// The bounding box, as what it is: a position in a picture.
    ///
    /// `nil` when there is no box. Never projected, never drawn over anything
    /// but the frame it came from, and never at the top level of a record — the
    /// Tower nests it inside `where` for exactly this reason.
    static func boxLine(_ frame: FrameReference) -> String? {
        guard let box = frame.boundingBoxNormalized, box.count == 4 else { return nil }
        return """
            Within that picture the detection covered \(percent(box[0])) to \
            \(percent(box[2])) across and \(percent(box[1])) to \(percent(box[3])) \
            down. Those are fractions of the frame, not distances in a room.
            """
    }

    /// What purging this record would and would not reach.
    ///
    /// `nil` when there is no pointer, because then there is nothing to warn
    /// about. Carried from `imagery_retention` rather than assumed, so a Tower
    /// that moved the imagery cannot leave this app repeating the old promise.
    static func imageryLine(_ frame: FrameReference) -> String? {
        guard frame.pointsAtACapture else { return nil }
        guard frame.imageryRetention == ObjectMemoryContract.captureSideImagery else {
            return """
                The frame this points at is kept under a retention this app does \
                not recognise (\(frame.imageryRetention)).
                """
        }
        return """
            The frame it points at is kept capture-side, under a retention this \
            cartridge neither sets nor enforces. Removing this record would not \
            reach that imagery.
            """
    }

    /// The three strength fields, in the order a reader should trust them, with
    /// every missing one said out loud.
    static func confidenceLine(_ observation: ObjectObservation) -> String {
        var sentences: [String] = []
        switch observation.confidence {
        case .unknown:
            sentences.append("The detector's confidence for this sighting was not recorded.")
        case .low, .medium, .high:
            sentences.append("Detector confidence: \(observation.confidence.displayName).")
        }

        if let best = observation.bestScore {
            sentences.append("Strongest score while it stayed in view: \(percent(best)).")
        } else {
            // Never "0%". `null` here means the field did not exist when the
            // record was written, and a zero would be a claim of no evidence.
            sentences.append("The strongest score was not tracked for this record.")
        }

        if let detector = observation.detectorScore {
            sentences.append("Score in the frame above: \(percent(detector)).")
        } else {
            sentences.append("The score in that frame was not recorded.")
        }

        sentences.append("None of these is a calibrated probability; they are detector output.")
        return sentences.joined(separator: " ")
    }

    /// What the record holds, from its own privacy tags. `nil` when it carries
    /// none, rather than a reassurance this app cannot support.
    static func privacyLine(_ observation: ObjectObservation) -> String? {
        guard observation.privacyTags.contains("derived-only") else { return nil }
        return "This record holds a label, a score and a box. No pixels."
    }

    // MARK: - An empty answer

    /// The headline when nothing was found. Never "not found", which reads as a
    /// verdict on the object.
    static func nothingObservedHeadline(objectClass: String?, recordable: Bool) -> String {
        guard recordable else { return "Never looked for" }
        guard let objectClass else { return "Nothing recorded in this window" }
        return "No record of \(indefiniteArticle(for: objectClass)) \(objectClass)"
    }

    /// Whose silence this is.
    ///
    /// The single most load-bearing string in the cartridge. Absence of a
    /// record is not absence of the object, and the sentence has to carry that
    /// on its own — a user reading "no results" will supply the wrong meaning
    /// for free.
    static func nothingObservedExplanation(objectClass: String?, recordable: Bool) -> String {
        guard recordable else {
            let named = objectClass ?? "that"
            return """
                “\(named)” is not a category this memory ever records, so it has \
                never been looked for. Its absence carries no information at all.
                """
        }
        let subject: String
        if let objectClass {
            subject = "\(indefiniteArticle(for: objectClass)) \(objectClass)"
        } else {
            subject = "anything"
        }
        return """
            This memory holds no record of \(subject) within the window it can \
            see. That is a statement about what the camera captured, not about \
            what exists: it may never have been pointed at one, or the detector \
            may not have scored it highly enough to write down. Absence of a \
            record is not absence of the thing.
            """
    }

    // MARK: - Retention

    /// The window the answer came from, and whether the question was narrowed.
    static func retentionLine(_ retention: ObjectMemoryRetention) -> String {
        guard let effective = retention.effectiveDays else {
            return """
                This memory was written without a time limit of its own, so \
                everything it holds is included.
                """
        }
        return """
            Showing what was recorded in the last \(days(effective)). Anything \
            older is not shown, and asking for a longer window cannot recover it.
            """
    }

    /// Shown only when the Tower actually refused a wider window, so a caller
    /// that asked for nothing is never told it was refused something.
    static func clampLine(_ retention: ObjectMemoryRetention) -> String? {
        guard retention.clamped else { return nil }
        guard let requested = retention.requestedDays, let effective = retention.effectiveDays else {
            return "A longer window was asked for than this memory keeps, and the extra was refused."
        }
        return """
            A window of \(days(requested)) was asked for; this memory keeps \
            \(days(effective)), and the difference was refused rather than served.
            """
    }

    // MARK: - Asking

    static func questionLine(_ question: ObjectMemoryQuestion) -> String {
        switch question {
        case .listing(let objectClass):
            guard let objectClass else { return "Everything recorded in the window" }
            return "Every record of \(indefiniteArticle(for: objectClass)) \(objectClass)"
        case .lastSeen(let objectClass):
            return "When \(indefiniteArticle(for: objectClass)) \(objectClass) was last in view"
        }
    }

    /// Why the ask control cannot be used, when it cannot.
    static let towerUnreachable = """
        The Tower is not connected, so its object memory cannot be asked \
        anything. Nothing already shown has changed; it is simply not being \
        refreshed.
        """

    /// The Tower answered 404: it serves no object memory at all.
    ///
    /// A statement about *configuration*, and worded so it cannot be read as a
    /// statement about what the wearer has seen.
    static let noObjectMemoryConfigured = """
        This Tower serves no object memory, so there is nothing to ask it for. \
        That is a fact about how the Tower is configured, not about what the \
        camera has seen.
        """

    static func unsupportedContract(_ identifier: String) -> String {
        """
        The Tower's object memory speaks an agreement this version of the app \
        does not implement (\(identifier)). Nothing is shown rather than \
        something guessed. Updating the app is what resolves this.
        """
    }

    // MARK: - Controls and headings

    /// The chrome, kept here with the prose rather than inline in the view.
    ///
    /// A button label is copy too, and "Find my laptop" is exactly the kind of
    /// sentence that would walk past a test that only looked at record rows.
    static let recordsHeading = "Records"
    static let showEverythingButton = "Show what was recorded"
    static let allCategories = "All categories"

    static func lastInViewButton(_ objectClass: String) -> String {
        "When was \(indefiniteArticle(for: objectClass)) \(objectClass) last in view"
    }

    /// The disclosure that holds the frame reference and the detection detail.
    ///
    /// Not "Where this was", which is the label a reader wants and the one
    /// thing this cartridge cannot supply.
    static let provenanceDisclosure = "Frame reference and detection detail"

    /// How many records the answer carried. Counts what was **served**, which
    /// is what fell inside the clamped window rather than what is on disk.
    static func recordCountLine(_ count: Int) -> String {
        count == 1
            ? "1 record within the window this read may see."
            : "\(count) records within the window this read may see."
    }

    /// Before anything has been asked. Says what asking does, because a person
    /// pressing a button on a memory cartridge deserves to know it reads rather
    /// than records.
    static let nothingAskedYet = """
        Nothing has been asked yet. Asking reads what the Tower already \
        recorded; it records nothing new and deletes nothing.
        """

    /// What this screen will be able to say, for the state panel's optional
    /// forward-looking line. Conditional, never a promise.
    static let futureDescription = """
        This workspace lists what the Tower's object memory recorded: which \
        category was in view, when, how strong the detection was, and which \
        frame of which capture the record points back at. It will never show \
        where something is, because nothing in this cartridge knows.
        """

    // MARK: - Provenance a record carries about how it was admitted

    /// How long the sighting lasted, when the record knows.
    ///
    /// `nil` when it does not — a record written before sightings had a length
    /// carries `null`, never `0`, and a "0 frames" on screen would claim a
    /// sighting that spanned nothing.
    static func durationLine(_ observation: ObjectObservation) -> String? {
        guard let lastSeenAt = observation.lastSeenAt else { return nil }
        let seconds = lastSeenAt.timeIntervalSince(observation.observedAt)
        guard seconds > 0 else { return nil }
        var sentence = "It stayed in view for about \(duration(seconds))"
        if let frameCount = observation.frameCount {
            sentence += ", across \(frameCount == 1 ? "1 frame" : "\(frameCount) frames")"
        }
        return sentence + ", on the Tower's receipt clock."
    }

    /// Which policy tier admitted the record, said as what actually happened.
    ///
    /// The distinction is worth a person's attention because it is the
    /// difference between one opinion and two. The old class list came from a
    /// score histogram, which describes the detector's opinion of itself;
    /// reading the crops found a ceiling fan detected as `airplane` at 0.99 and
    /// a white door as `refrigerator` at 0.95.
    static func tierLine(_ observation: ObjectObservation) -> String? {
        guard let tier = observation.tier else { return nil }
        switch tier {
        case "remembered":
            return """
                Written on the detector's word alone. Nothing was asked to \
                agree with it.
                """
        case "verify":
            return """
                Written only because a second model was asked and agreed. \
                Without that agreement there would be no record.
                """
        default:
            return """
                Admitted under a policy tier this app does not recognise \
                (\(tier)), so it is shown without interpretation.
                """
        }
    }

    /// What the second opinion actually said, naming the model.
    ///
    /// `nil` when nothing was asked, which is the ordinary case: the verifier
    /// ships **off**, and a reassuring sentence about a verification that never
    /// happened is exactly the kind of implied capability this cartridge must
    /// not offer.
    static func verificationLine(_ observation: ObjectObservation) -> String? {
        guard let verification = observation.verification else { return nil }
        let model = verification.model ?? "a second model"
        var sentences: [String] = []
        sentences.append(
            verification.agrees
                ? "\(model) was asked about this crop and agreed."
                : "\(model) was asked about this crop and did not agree."
        )
        if let score = verification.score {
            sentences.append("Similarity: \(percent(score)).")
        }
        if let label = verification.label, label != observation.objectClass {
            // Named, and named as what it is: the verifier can only refuse or
            // confirm, and a verdict naming something else is evidence rather
            // than a relabelling.
            sentences.append(
                """
                It would have called this “\(label)”. That was recorded as \
                evidence and did not change what this record says.
                """
            )
        }
        sentences.append(
            """
            That similarity is not a calibrated probability: its threshold was \
            fitted to 94 crops from one home.
            """
        )
        return sentences.joined(separator: " ")
    }

    /// What this app was told about how this Tower is configured, composed
    /// only from what the payload actually carried.
    ///
    /// Deliberately says what it was **not** told. This app never learns which
    /// detector produced a record — the wire carries no model name outside a
    /// `verification` block — and a screen that stayed quiet about that would
    /// let a reader assume the app knows and is choosing not to say.
    static func whatThisTowerReports(_ answer: ObjectMemoryAnswer) -> String {
        var sentences: [String] = []
        let classes = answer.envelope.recordedClasses
        if classes.isEmpty {
            sentences.append("This Tower reports that it records no categories at all.")
        } else {
            sentences.append(
                """
                This Tower reports that it records \(list(classes)). That list \
                is its configuration, not this app's assumption.
                """
            )
        }
        let verifiers = Set(
            observations(in: answer).compactMap { $0.verification?.model }
        ).sorted()
        if verifiers.isEmpty {
            sentences.append(
                """
                No record shown here was checked by a second model, so this \
                Tower is not reporting one.
                """
            )
        } else {
            sentences.append("Records shown here were checked by \(list(verifiers)).")
        }
        sentences.append(
            """
            The Tower does not report which detector wrote these records, so \
            this app cannot name one.
            """
        )
        return sentences.joined(separator: " ")
    }

    /// Every record in an answer, so the config summary can read them without
    /// the caller unwrapping the two cases.
    private static func observations(in answer: ObjectMemoryAnswer) -> [ObjectObservation] {
        switch answer {
        case .listing(let listing): return listing.observations
        case .lastSeen(let lastSeen): return lastSeen.observation.map { [$0] } ?? []
        }
    }

    // MARK: - Pictures

    /// **The caption, and it carries the whole burden.**
    ///
    /// A picture is a much stronger location cue than a sentence, and no string
    /// test can catch it: every rule the rest of this file enforces can be
    /// undone by putting a first-person photograph of a room on screen with a
    /// weak line under it. A reader shown a desk will conclude they know where
    /// the desk is, and they will be right about the desk and wrong about
    /// everything this cartridge claims.
    ///
    /// So the caption is **not optional anywhere**, it appears with every
    /// picture this app draws, and it says three things in order: what was
    /// observed, what the picture is, and what it is not.
    ///
    /// The wording is the contract's own suggestion, adapted per route, and is
    /// explicitly **to be tested on a person rather than accepted from a
    /// document**: shown cold, if the word "where" comes back, it is wrong.
    static func pictureCaption(
        objectClass: String?, kind: ObjectMemoryImageryKind
    ) -> String {
        let subject: String
        if let objectClass {
            subject = "\(indefiniteArticle(for: objectClass).capitalized) \(objectClass) was visible."
        } else {
            subject = "A category this memory records was visible."
        }
        let what: String
        switch kind {
        case .crop:
            what = """
                This is the part of the frame the detection covered, taken from \
                the recording the record was written against.
                """
        case .frame, .view:
            what = """
                This is the whole frame the record was written against, taken \
                from the recording.
                """
        }
        return """
            \(subject) \(what) It is a picture out of a recording, not a place \
            and not a map, and it does not say anything about now.
            """
    }

    /// What the display filter did to the picture that is on screen.
    ///
    /// **Never called redaction, anonymisation, or privacy-safe**, and that is
    /// not a style preference. The Tower's own privacy transformation runs
    /// before persistence, at the one choke point every persisted pixel passes;
    /// this runs on **read**, the stored frame is unchanged, and the capture
    /// manifests record `redaction: "none"`. Calling this the other thing would
    /// tell a wearer their recordings are altered when they are not.
    ///
    /// It also states the blind spots, because a filter described only by what
    /// it catches reads as a guarantee.
    ///
    /// ## The "when" clause follows `filter_means`, because there are now two
    ///
    /// A capture frame is filtered **on read** and the stored frame is
    /// unchanged. Object Memory's own keyframe is filtered **before it is
    /// written**, so there is no unfiltered copy of that file at all — a
    /// stronger statement, and one this app may only make when the payload
    /// makes it. Neither wording is chosen by this app: `filterMeans` is on the
    /// wire, `ObjectMemoryImageryContract.everyFilterMeaning` is the list of
    /// meanings this build has a sentence for, and a payload carrying anything
    /// else is refused by the decoder rather than worded from the nearest
    /// guess.
    ///
    /// The blind-spot clause is unconditional under both, because a filter
    /// described only by what it catches reads as a guarantee whichever store
    /// it ran over.
    static func filterLine(_ description: ObjectMemoryImageryDescription) -> String {
        let named = description.filter.map { " The filter that ran is \($0)." } ?? ""
        let when: String
        if description.filterMeans == ObjectMemoryImageryContract.appliedBeforePersistence {
            when = """
                 It ran before this picture was written, so no unfiltered copy \
                of this picture was kept.
                """
        } else {
            when = " It runs when a picture is read; the stored frame is unchanged."
        }
        return """
            Faces this Tower's display filter detected have been filled in on \
            the way out.\(named)\(when) Bodies, clothing, room contents, \
            screens, and any face it did not detect are all in the picture.
            """
    }

    /// How many regions were filled, worded so zero cannot be read as an
    /// all-clear.
    ///
    /// `regions_filled: 0` means **nothing was detected**, and this detector
    /// has measured blind spots: a face occluded past about 60%, a face rotated
    /// about 90° in plane, and profile and rear views.
    static func regionsFilledLine(_ description: ObjectMemoryImageryDescription) -> String {
        guard description.regionsFilled > 0 else {
            return """
                The filter filled nothing in this picture. That means it \
                detected nothing, not that there was nothing to detect.
                """
        }
        let count = description.regionsFilled
        return """
            The filter filled \(count == 1 ? "1 region" : "\(count) regions") \
            in this picture.
            """
    }

    /// Shown when part of the record's own subject is behind a fill.
    ///
    /// `nil` when nothing overlaps. Any overlap at all earns the sentence:
    /// this exists for a measured defect, not a hypothetical. The filter fires
    /// on 40.2% of real corpus frames, and of 36 firings inspected by eye 4
    /// were a real face and 32 were hands on a keyboard, a screen, a door or a
    /// sink. One fill landed squarely on the mouse a record was about.
    ///
    /// The answer to that is not a lower threshold — a face-detection
    /// threshold is not a picture-quality knob — so the overlap is said out
    /// loud and the whole frame is shown instead of the crop.
    static func subjectObscuredLine(
        _ description: ObjectMemoryImageryDescription, kind: ObjectMemoryImageryKind
    ) -> String? {
        guard description.subjectIsBehindAFill else { return nil }
        let fellBack = kind == .frame
            ? " The whole frame is shown instead of the close crop for that reason."
            : ""
        return """
            About \(percent(description.subjectObscured)) of the area this \
            record was written for is behind a fill. The filter is a face \
            detector at a fixed threshold and it does fire on things that are \
            not faces — a hand, a screen, a sink.\(fellBack)
            """
    }

    /// The two route labels, so a person can ask for the other picture.
    static let showTheWholeFrameButton = "Show the whole frame"
    static let showTheDetectionButton = "Show just the detection"
    static let showThePictureButton = "Show the picture"

    /// The headline when there are no bytes.
    ///
    /// **Never a broken image, never an empty row, and never a connection
    /// error.** Each of these is a true and useful sentence about something
    /// that actually happened, and one of them — the 410 — is the case the
    /// whole imagery payload shape exists to be able to say.
    static func noPictureHeadline(_ description: ObjectMemoryImageryDescription) -> String {
        switch description.situation {
        case .aPicture:
            // Unreachable while `available` and `reason` travel together, and
            // written anyway so a payload where they came apart cannot fall
            // through to a blank caption.
            return "No picture is shown"
        case .thePictureIsGone: return "The memory is kept, the picture is gone"
        case .theRecordNeverPointedAtAFrame: return "This record never had a picture"
        case .noSuchRecord: return "Nothing under this handle"
        case .theTowerServesNoPictures: return "This Tower is serving no pictures"
        case .anUnrecognisedRefusal: return "No picture, for a reason this app does not recognise"
        }
    }

    /// Why there are no bytes, at length.
    ///
    /// The 410 sentence is the one to be most careful with: it is not an error
    /// and not a loss of the record. Capture-side retention removed the frame,
    /// `memory_retained: true` says the record is untouched, and a person needs
    /// to hear both halves in the same breath.
    static func noPictureExplanation(_ description: ObjectMemoryImageryDescription) -> String {
        switch description.situation {
        case .aPicture:
            return "No picture is being shown for this record."

        case .thePictureIsGone:
            let kept = description.memoryRetained
                ? "This record is untouched and stays queryable."
                : """
                    The Tower did not report the record as retained, so this \
                    app cannot say the record survived either.
                    """
            return """
                The frame this record points at has passed out of the \
                recordings' own retention, which this cartridge neither sets \
                nor enforces. \(kept) Nothing failed, and nothing was deleted \
                from this memory.
                """

        case .theRecordNeverPointedAtAFrame:
            return """
                This record carries no frame pointer, so there was never a \
                picture behind it. What it holds is a label, a score and a box.
                """

        case .noSuchRecord:
            return """
                This Tower has no record under that handle within the window \
                this read may see. A handle resolves through the same clamped \
                read a listing does, so knowing one does not reach past \
                retention.
                """

        case .theTowerServesNoPictures(let reason):
            switch reason {
            case .displayFilterUnavailable:
                return """
                    This Tower cannot run its display filter, so it is serving \
                    no pictures at all. It refuses rather than serving an \
                    unfiltered frame from the camera, and this app does not \
                    work around that. The records themselves are unaffected.
                    """
            case .noCaptureRootConfigured:
                return """
                    This Tower has nowhere to look for the recordings, so it \
                    is serving no pictures at all. That is a fact about how it \
                    is configured. The records themselves are unaffected.
                    """
            case .frameUnreadable:
                return """
                    The frame behind this record is where it should be and \
                    could not be decoded, so nothing is served. The record \
                    itself is unaffected.
                    """
            default:
                return """
                    This Tower is serving no pictures, and gave the reason \
                    “\(reason.rawValue)”. The records themselves are \
                    unaffected.
                    """
            }

        case .anUnrecognisedRefusal(let reason):
            let named = reason.rawValue.isEmpty ? "none" : "“\(reason.rawValue)”"
            return """
                The Tower served no picture and gave a reason this build does \
                not implement (\(named)). Nothing is shown rather than \
                something guessed, and the record itself is unaffected.
                """
        }
    }

    /// What a record's picture is kept under.
    ///
    /// ## One sentence became three, because there are now two stores
    ///
    /// A picture used to be one thing: a render of a frame in the capture
    /// store, under a retention this cartridge neither sets nor enforces. That
    /// is no longer the only case. Object Memory now owns a small filtered crop
    /// per record, under **its own** retention, which means the crop survives
    /// the recording it came from and goes when the record does — the exact
    /// opposite lifetime, said in the same slot on the same screen.
    ///
    /// So the sentence is chosen from `imagery_source`, which is the field that
    /// knows. Not from the route asked for: `/crop` prefers the owned keyframe
    /// and falls back to a capture frame, so the route does not determine the
    /// store. Not from `imagery_retention` either, which is a label rather than
    /// a lifetime and would have this app branching on wording.
    ///
    /// The third sentence is for a Tower that names no source. It says only the
    /// part that is true of every picture this app draws — that the phone does
    /// not keep it — rather than picking one of the two retentions and being
    /// wrong half the time.
    static func pictureRetentionLine(
        _ description: ObjectMemoryImageryDescription
    ) -> String {
        // Written as comparisons rather than as a `switch` over the optional:
        // `ObjectMemoryImagerySource` is a `RawRepresentable` struct, not an
        // enum, precisely so an unknown source survives the decode — so there
        // is no exhaustive set to switch over, and `==` is what the type
        // actually offers.
        guard let source = description.imagerySource else {
            return unnamedSourceRetentionLine
        }
        if source == .objectMemoryKeyframe { return keyframeRetentionLine }
        if source == .captureFrame { return captureFrameRetentionLine }
        // A source this build has never heard of. A sentence about *which*
        // retention would be a guess, exactly as it is for a Tower that says
        // nothing.
        return unnamedSourceRetentionLine
    }

    /// This cartridge's own copy: it goes when the record goes.
    static let keyframeRetentionLine = """
        This picture is kept by this memory itself, for as long as the record \
        is. It goes when the record expires or is purged, and it outlasts the \
        recording it was taken from. It is not held on this phone after this \
        row leaves the screen.
        """

    /// A render out of the capture store, whose lifetime this cartridge does
    /// not set.
    static let captureFrameRetentionLine = """
        This picture is kept with the recordings, under a retention this \
        cartridge neither sets nor enforces, so it can go while the record \
        stays. It is not held on this phone after this row leaves the screen.
        """

    /// A Tower that does not say which store served the bytes.
    static let unnamedSourceRetentionLine = """
        This Tower did not say which store this picture came from, so nothing \
        is claimed here about how long it is kept. It is not held on this \
        phone after this row leaves the screen.
        """

    /// The context view is gone and the object picture is not.
    ///
    /// Shown in place of the "Show the whole frame" control when the Tower has
    /// said `frame_available: false`. Offering the control there would send a
    /// tap to a route that answers 410, and the wearer would read "the memory
    /// is kept and the picture is gone" over a picture that is on their screen.
    ///
    /// `nil` from `frame_available` never reaches this: an unknown answer keeps
    /// the control, because the Tower is the thing that gets to answer it.
    static let wholeFrameIsGoneLine = """
        The wider view around this is gone: it lived with the recording, which \
        has been deleted. This picture is kept by the memory itself.
        """

    /// This Tower offered no imagery routes at all.
    static let noPicturesOffered = """
        This Tower's object memory does not offer pictures. It describes what \
        was recorded and points back into a recording, and nothing more.
        """

    static let unreadableImageryAnswer = """
        The Tower answered about this picture in a shape this app does not \
        understand, so nothing is shown rather than something guessed.
        """

    // MARK: - The session

    /// The heading over the panel that reports what the Tower says.
    ///
    /// It used to be "Remembering", which is now the heading over the *control*
    /// — the one composed lifecycle that starts the producer and the camera
    /// together. Two panels both titled "Remembering" would have read as one
    /// repeated thing rather than as a control and the reading it produced, and
    /// this one has never been a control: it is what the Tower reports, drawn
    /// from `following` for liveness and from `state` only as intent.
    static let sessionHeading = "What the Tower reports"

    /// What was asked for. **Intent, never liveness** — the Tower says so
    /// itself in `state_means: "intent-not-liveness"`, and this sentence is
    /// worded to make the distinction unmissable rather than to record it.
    static func intentLine(_ snapshot: CartridgeSessionSnapshot) -> String {
        switch snapshot.state {
        case .active:
            return """
                Asked to remember. That is what the Tower has recorded as the \
                intent; it is not a report that anything is running.
                """
        case .paused:
            return """
                Asked to pause. That is the intent recorded on the Tower, not \
                a report about a producer.
                """
        case .stopped:
            return """
                Asked to stop, or never started. That is the intent recorded \
                on the Tower, not a report about a producer.
                """
        default:
            return """
                The Tower reports an intent this app does not recognise \
                (\(snapshot.state.rawValue)), so it is shown without \
                interpretation.
                """
        }
    }

    /// **The liveness sentence, drawn from `following` and from nothing else.**
    ///
    /// This is the string the whole session surface exists to be able to write
    /// truthfully. `state` is what a person asked for; `following` is what a
    /// producer is alive on. They come apart exactly when it matters most — a
    /// Pause whose producer ignores `SIGTERM` answers 200 with
    /// `state: "paused"` and `changed: true` while the process keeps recording
    /// — and a Pause button keyed on `state` tells a person they stopped being
    /// recorded when they did not.
    static func livenessLine(_ snapshot: CartridgeSessionSnapshot) -> String {
        guard snapshot.isFollowingACapture else {
            return """
                No producer is attached to a recording, so nothing is being \
                written into this memory.
                """
        }
        // The SCOPED list, to agree with the guard above it. Counting
        // `following` here while branching on `followingThisSession` would
        // let a leftover producer inflate a number describing this session.
        let count = (snapshot.followingThisSession ?? snapshot.following).count
        let captures = count == 1 ? "a recording" : "\(count) recordings"
        return """
            A producer is alive on \(captures) and is writing into this \
            memory. This is read from the Tower's list of what is actually \
            being followed, not from what was asked for.
            """
    }

    /// A producer the control on this screen did not start, and does not reach.
    ///
    /// `nil` on every ordinary reading, and `nil` on a Tower that cannot scope
    /// the question at all — see `recordingsThisControlDidNotStart`, where an
    /// absent `following_this_session` and an empty one mean two different
    /// things and must never be folded together.
    ///
    /// ## Three separate corrections live in this one sentence
    ///
    /// **It no longer says "a producer this session did not start".** The Tower
    /// scopes `following_this_session` by *started at or after this session
    /// last went active*, and that mark is re-taken on every Resume. So after a
    /// Pause whose producer survived, the survivor genuinely belongs to this
    /// session — it was started before the pause — and is correctly outside the
    /// scoped list anyway. Session ownership is not what this list knows, so
    /// the sentence may not claim it.
    ///
    /// **It no longer says "is still alive".** `"is still"` is on the
    /// forbidden-phrase lists in `ObjectMemoryTests`, and this line walked past
    /// both of them: the session fixture omits `following_this_session` by
    /// default, so the line returned `nil` in every case the copy test
    /// generated and was never once inspected. The fix was two things, and the
    /// second matters more than the wording — a case that sets
    /// `followingThisSession: []` beside a non-empty `following` now runs in
    /// `testEverySessionSentenceMakesNoClaimItCannotSupport`, so the sentence
    /// is actually exercised.
    ///
    /// **It no longer says what would stop it.** It used to end "Restarting the
    /// Tower will", about a process that has already ignored `terminate()`.
    /// Nothing establishes that a restart reaches it either.
    ///
    /// What is left is what is true in every case that reaches here: something
    /// is recording, the control on this screen did not start it, and a Stop
    /// here does not reach it. A person has to be told that even though nothing
    /// on this screen can fix it.
    static func leftoverProducerLine(
        _ snapshot: CartridgeSessionSnapshot
    ) -> String? {
        let leftovers = snapshot.recordingsThisControlDidNotStart
        guard !leftovers.isEmpty else { return nil }
        let count = leftovers.count
        let recordings = count == 1 ? "a recording" : "\(count) recordings"
        return """
            Separately: a producer that the control on this screen did not \
            start is writing into \(recordings). It is not writing into what \
            you just asked for, and stopping here does not reach it.
            """
    }

    /// Shown loudly when the Tower's two fields contradict each other in the
    /// direction that harms a person.
    ///
    /// `nil` when they agree. The other direction — asked to remember with
    /// nothing followed — gets no warning here, because it is legal: starting
    /// before the camera is running looks exactly like that, and so does a
    /// producer that died, and this app cannot tell them apart from one
    /// payload. `livenessLine` says what is true in both cases without
    /// guessing which.
    static func livenessContradictsIntentLine(
        _ snapshot: CartridgeSessionSnapshot
    ) -> String? {
        guard snapshot.intentContradictsLiveness else { return nil }
        let asked = snapshot.state == .paused ? "pause" : "stop"
        return """
            The Tower reported the \(asked) as honoured and also reports a \
            producer alive on a recording. The writing has not ended. Treat \
            this memory as being written into until that list is empty.
            """
    }

    /// Session provenance: the id, and what it is not.
    ///
    /// The `session_id` is minted at Start and is **not** a capture id. Saying
    /// so in the same sentence is the same discipline `frameLine` applies to a
    /// capture id, and for the same reason — an opaque identifier next to a
    /// memory screen invites being read as a place.
    static func sessionProvenanceLine(_ snapshot: CartridgeSessionSnapshot) -> String {
        guard let sessionID = snapshot.sessionID else {
            return "There is no session on the Tower to describe."
        }
        var sentence = "Session \(shortened(sessionID)). That is a handle for this run of "
        sentence += "the producer, not a recording and not a place."
        if let startedAt = snapshot.startedAt {
            let when = startedAt.formatted(date: .abbreviated, time: .shortened)
            sentence += """
                 It was started at \(when) on the Tower's own clock; the \
                Tower keeps no capture clock.
                """
        }
        return sentence
    }

    /// What a producer has been seen following, in the order first seen.
    ///
    /// History, and worded as history: a recording in this list and not in
    /// `following` is one the producer has finished with, and rendering the
    /// two the same way would put a stale recording under a live label.
    static func capturesLine(_ snapshot: CartridgeSessionSnapshot) -> String? {
        guard !snapshot.captures.isEmpty else { return nil }
        let count = snapshot.captures.count
        return """
            This session's producer has been seen following \
            \(count == 1 ? "1 recording" : "\(count) recordings") so far. That \
            is a history; only the live list above says what is being written \
            into now.
            """
    }

    /// The late-attachment consent point, said where the button is.
    ///
    /// A wearer who starts remembering at 15:03 has not asked for the 15:00
    /// part of the walk, and a Start that quietly swept up the earlier part of
    /// a recording would be taking a decision that is theirs.
    static let startMeaningLine = """
        Starting attaches a producer to whatever is recording from now on. It \
        does not reach back into a recording that was already under way.
        """

    static let sessionNotPersistedLine = """
        Nothing here survives a Tower restart. A Tower that comes back comes \
        back stopped, and remembering has to be asked for again.
        """

    /// Button labels, read against the Tower's own `actions` list.
    static func actionButton(_ action: CartridgeSessionAction) -> String {
        switch action {
        case .start: return "Start remembering"
        case .pause: return "Pause"
        case .resume: return "Resume"
        case .stop: return "Stop"
        }
    }

    /// An action that was honoured and moved nothing.
    ///
    /// **This is not an error and must never be drawn as one.** A second Start
    /// answers 200 with `changed: false`; so does a second Pause, and so does
    /// Stop from stopped. Every one of those is the Tower saying "you already
    /// have what you asked for".
    static func idempotentNoOpLine(_ action: CartridgeSessionAction) -> String {
        switch action {
        case .start: return "Already remembering. Nothing needed to change."
        case .pause: return "Already paused. Nothing needed to change."
        case .resume: return "Already remembering. Nothing needed to change."
        case .stop: return "Already stopped. Nothing needed to change."
        }
    }

    /// Why an action could not be honoured.
    ///
    /// **Worded from the action and the state actually reached, never from
    /// `reason`.** The contract document and the running Tower disagree about
    /// which word a `resume` from `stopped` carries — the document says
    /// `not-active`, the wire says `not-paused` — and both are truthful
    /// descriptions of the same situation. A sentence keyed on the word would
    /// be wrong against one of the two Towers. A sentence keyed on the action
    /// and the state is right against both, and stays right if a third word
    /// appears.
    static func refusalLine(_ refusal: CartridgeSessionRefusal) -> String {
        if refusal.reason == .unsupported {
            return """
                This Tower has no object memory producer to start, so \
                \(actionButton(refusal.action).lowercased()) cannot be \
                honoured. That is a fact about how the Tower is configured.
                """
        }
        if refusal.reason == .unknownAction {
            return """
                This app sent a verb this Tower does not offer. Nothing \
                changed on the Tower.
                """
        }

        let state = refusal.snapshot.state
        switch refusal.action {
        case .resume where state == .stopped:
            return """
                Resume continues a session that was paused, and this one is \
                stopped. Start it instead — starting works from stopped and \
                from paused alike.
                """
        case .pause where state == .stopped:
            return """
                There is nothing to pause: this Tower was not remembering \
                anything when it was asked.
                """
        default:
            let named = state.isRecognised ? state.rawValue : "a state this app does not recognise"
            return """
                The Tower could not \(actionButton(refusal.action).lowercased()) \
                from \(named), and nothing changed. Reading the session again \
                shows where it actually is.
                """
        }
    }

    /// The Tower's own refusal sentence, kept as provenance behind a
    /// disclosure. Written for an operator, so it is never the first thing a
    /// wearer reads.
    static func refusalProvenanceLine(_ refusal: CartridgeSessionRefusal) -> String {
        """
        The Tower's own words: “\(refusal.message)” (\(refusal.reason.rawValue)).
        """
    }

    /// This Tower has no controllable session for this cartridge.
    static let noSessionControl = """
        This Tower offers no start or stop for its object memory, so there is \
        nothing here to press. It can still be asked what it already recorded. \
        That is a fact about how the Tower is configured.
        """

    /// The Tower has a session surface but no producer behind it.
    static let sessionUnsupported = """
        This Tower has no object memory producer to start. The controls are \
        shown as unavailable rather than hidden, so it is clear the button \
        exists and this Tower cannot honour it.
        """

    static let sessionUnread = """
        The session has not been read yet, so this app cannot say whether \
        anything is being written into this memory.
        """

    static let unreadableSessionAnswer = """
        The Tower answered about its session in a shape this app does not \
        understand, so nothing is claimed about what is running.
        """

    static func unsupportedSessionContract(_ identifier: String) -> String {
        """
        The Tower's session control speaks an agreement this version of the \
        app does not implement (\(identifier)). No control is offered rather \
        than one that might do something else. Updating the app is what \
        resolves this.
        """
    }

    /// The Tower answered with a status this route does not define.
    ///
    /// Its own sentence rather than the transport one, because the two send a
    /// person to different places: this says the Tower is right there and its
    /// answer was unusable, and a connection sentence here would send someone
    /// to check a network about a machine that replied.
    static func towerAnswered(_ status: Int) -> String {
        """
        The Tower answered with a status this app does not expect on that \
        route (\(status)). The Tower was reached; its answer could not be \
        used, so nothing is shown rather than something guessed.
        """
    }

    // MARK: - Remembering, as one composed lifecycle

    /// The heading over the control that starts both halves.
    static let recordingHeading = "Remembering"

    /// The primary control's label.
    ///
    /// "Stop remembering" rather than a bare "Stop", because this button does
    /// more than the verb: it stops the Tower's producer and, when this screen
    /// is the one that started it, the glasses camera as well. Start keeps the
    /// session vocabulary's wording, which was already the composed sentence.
    static let stopRememberingButton = "Stop remembering"

    /// Why the session panel below carries no Start and no Stop.
    ///
    /// The panel used to render the Tower's whole `actions` vocabulary, which
    /// put a **second prominent Start** and a **second, differently-labelled
    /// Stop** directly under the composed control — four primary-looking
    /// buttons on one screen, two of them saying the same verb in two words
    /// and one of them not mentioning the camera it also stops.
    ///
    /// Not resolved by disabling anything. `ObjectMemorySessionPanel`'s own
    /// documentation argues, correctly, that hiding or disabling a verb on this
    /// app's guess about what the Tower would accept makes this app's model
    /// authoritative over the Tower's. **Not drawing a second copy of a control
    /// that is already on the screen is a different thing**: the verb is still
    /// offered, still sends, still reads its vocabulary off `actions`, and the
    /// panel still names any verb it has no button for. What is removed is the
    /// duplicate, not the capability.
    static let sessionPrimaryVerbsLiveAbove = """
        Start and Stop for this cartridge are the one control above, which \
        also starts and stops the glasses camera. The verbs here are the rest \
        of what this Tower offers.
        """

    /// Which label the primary control carries. `.stop` gets the composed
    /// wording above; everything else reuses the session vocabulary, so the two
    /// panels cannot come to call the same verb two different things.
    static func recordingPrimaryButton(_ action: CartridgeSessionAction) -> String {
        action == .stop ? stopRememberingButton : actionButton(action)
    }

    /// What one run of remembering has actually got to.
    ///
    /// **Four separate sentences cover what a single "recording" badge would
    /// collapse.** `starting` is a request in flight. `waitingToBeFollowed` is
    /// a request the Tower accepted with nothing attached yet. `notObserved` is
    /// the same payload after the wait ran out. Only `remembering` says a
    /// producer is alive on a recording, and only it is drawn from `following`.
    static func recordingHeadline(_ reading: ObjectMemoryRecordingReading) -> String {
        switch reading.phase {
        case .idle:
            return """
                Nothing has been asked for on this screen yet. Starting asks \
                the Tower to remember and starts the glasses camera.
                """
        case .starting:
            return """
                Asking the Tower to remember, then starting the glasses \
                camera. Neither has answered yet.
                """
        case .waitingToBeFollowed:
            return """
                The Tower accepted. Waiting for a producer to attach to a \
                recording — until one does, nothing is being written into this \
                memory.
                """
        case .notObserved:
            return """
                Asked to remember, and not observed. The Tower reports no \
                producer attached to a recording, so nothing is being written \
                into this memory. The session was accepted; a producer has not \
                been seen following anything.
                """
        case .remembering:
            return """
                A producer is alive on a recording and writing into this \
                memory. That is read from the Tower's list of what is actually \
                being followed, not from what was asked for.
                """
        case .pausing:
            return "Asking the Tower to stop remembering, for now."
        case .paused:
            return """
                Paused. The Tower reports no producer attached to a recording, \
                so nothing is being written into this memory.
                """
        case .resuming:
            return "Asking the Tower to carry on remembering."
        case .stopping:
            return "Asking the Tower to stop remembering."
        case .stopped:
            return """
                Stopped. The Tower reports no producer attached to a recording, \
                so nothing is being written into this memory.
                """
        case .stillFollowing(let action):
            return """
                The Tower reported the \(actionButton(action).lowercased()) as \
                honoured and also reports a producer alive on a recording. The \
                writing has not ended. Treat this memory as being written into \
                until that list is empty.
                """
        case .cannotTell(let failure):
            return """
                The session could not be read, so this app cannot say whether \
                anything is being written into this memory. \(failure.message)
                """
        case .refused(let refusal):
            return refusalLine(refusal)
        case .cameraRefused(let refusal):
            return cameraRefusalLine(refusal)
        case .unsupported:
            return sessionUnsupported
        case .failed(let failure):
            return failure.message
        }
    }

    /// What the glasses camera is doing, said beside the Tower's half rather
    /// than folded into it.
    ///
    /// **The two halves can disagree, and the disagreement is the point.**
    /// Pause detaches a producer on the Tower; nothing in this app pauses a DAT
    /// stream, because `GlassesConnection` has no such call. So "paused" can be
    /// true at the same moment as "the camera is open", and a screen that
    /// printed only the first would be read as saying the camera stopped.
    ///
    /// ## `.running` does not mean a frame has arrived
    ///
    /// These two sentences used to say the camera "is sending frames to the
    /// Tower", and they were printed from the instant of the tap.
    /// `GlassesConnection.captureClaim` maps `DeviceSessionState.starting` to
    /// `.running`, and `startCameraSession()` sets `.starting` synchronously —
    /// so the claim was made over a session that had not connected to anything
    /// and had certainly delivered nothing. `CaptureClaim.running`'s own
    /// documentation says delivery is *expected*, which is the weaker word and
    /// the correct one.
    ///
    /// So these say the camera is **open**, which is what the claim knows.
    /// Making a frames claim honest would need a `.streaming` signal on
    /// `ObjectMemoryCaptureOwner` — `GlassesConnection.cameraStreamState` has
    /// it — and that member is deliberately not added: the sentence a wearer
    /// needs here is "is the camera on and does Stop end it", and neither half
    /// of that gets better for knowing whether a particular frame landed.
    static func recordingCameraLine(_ reading: ObjectMemoryRecordingReading) -> String {
        guard reading.cameraIsReachable else { return recordingCameraNotInThisBuild }
        switch reading.camera {
        case .devicePaused:
            return """
                The glasses have paused the capture themselves — a press on the \
                temple, or heat. The connection is held and delivery comes back \
                on its own; this app has no way to override that and does not \
                offer one.
                """
        case .running:
            if reading.cameraStartedHere {
                return """
                    The glasses camera this screen started is open, and frames \
                    are expected to reach the Tower. This screen does not see \
                    individual frames and does not claim any have arrived. \
                    Stopping here ends the camera as well as the session.
                    """
            }
            return """
                The glasses camera was started elsewhere in this app and is \
                open, so frames are expected to reach the Tower. Stopping here \
                ends this memory's session and leaves that capture alone.
                """
        case .ending:
            return """
                The glasses camera is shutting down. A capture can be started \
                again once it has.
                """
        case .unclaimed:
            return """
                No capture is running from this phone. Starting asks the Tower \
                first, so its gate is open, and then starts the camera.
                """
        }
    }

    /// The Tower agreed to remember and the camera did not start.
    ///
    /// Every one of these says the Tower's half **stands**, because it does:
    /// the session is a gate rather than a recording, and the next capture to
    /// open — from here, from Home, or from World Builder — finds it open.
    /// Tearing it down because the other half was refused would throw away
    /// correct work and leave a person with nothing to resume.
    static func cameraRefusalLine(_ refusal: CaptureStartRefusal) -> String {
        switch refusal {
        case .alreadyRunning:
            return """
                The Tower was asked to remember. A capture was already under \
                way, so this screen left it alone rather than starting a second \
                one.
                """
        case .deviceHasPausedCapture:
            return """
                The Tower was asked to remember. The glasses have paused the \
                capture themselves — a press on the temple, or heat — and \
                delivery comes back on its own. This app cannot override that, \
                so remembering waits for the glasses.
                """
        case .captureIsShuttingDown:
            // Deliberately not worded as "a capture is already under way",
            // which is what `.alreadyRunning` says and what this screen used
            // to imply by treating a teardown as somebody else's stream. A
            // capture that is shutting down belongs to nobody, and telling a
            // wearer it is running sends them looking for a stream that is on
            // its way out.
            return """
                The Tower was asked to remember. The glasses camera is shutting \
                down and a new capture cannot open until it has finished, so \
                none was started. The session stays open on the Tower; starting \
                again in a moment opens a real capture into it.
                """
        case .noActiveDevice:
            return """
                The Tower was asked to remember, and no glasses are active yet, \
                so no capture could be started. The session stays open on the \
                Tower and the next capture to open finds it ready.
                """
        case .cameraPermissionNotGranted:
            return """
                The Tower was asked to remember, and camera access is not \
                granted, so no capture could be started. Allow it under \
                Connections, then start again.
                """
        case .datRefused(let reason):
            return """
                The Tower was asked to remember, and the glasses refused to \
                start a capture: \(reason). The session stays open on the Tower \
                and the next capture to open finds it ready.
                """
        }
    }

    /// What the one button does, said where the button is.
    static let recordingWhatStartDoes = """
        One tap does both halves: the Tower is asked to remember first, so its \
        gate is open, and then the glasses camera is started unless something \
        else in this app has already started it.
        """

    /// What Pause does, and the half it cannot touch.
    static let recordingPauseMeaning = """
        Pause detaches the producer on the Tower. It does not pause the glasses \
        camera, because nothing in this app can: a capture that is running \
        keeps running and keeps sending frames, and they are simply no longer \
        read into this memory.
        """

    /// A build with no capture surface at all — Release, and every preview.
    static let recordingCameraNotInThisBuild = """
        This build cannot start the glasses camera, so this control asks the \
        Tower only. A capture has to come from somewhere else.
        """

    // MARK: - The test seam

    /// Every string this cartridge would put on screen for one answer.
    ///
    /// `ObjectMemoryCopyTests` asserts over this, and the view renders from the
    /// same functions, so a sentence cannot reach a person without passing the
    /// test — provided the view keeps writing none of its own. That proviso is
    /// the file's only real weak point and is called out in the handoff.
    static func everyString(
        for answer: ObjectMemoryAnswer, question: ObjectMemoryQuestion? = nil
    ) -> [String] {
        var strings: [String] = [summary, recordableClasses(answer.envelope.recordedClasses)]
        strings.append(whatThisTowerReports(answer))
        strings.append(retentionLine(answer.envelope.retention))
        if let clamp = clampLine(answer.envelope.retention) { strings.append(clamp) }

        if let question { strings.append(questionLine(question)) }

        switch answer {
        case .listing(let listing):
            strings.append(recordCountLine(listing.observations.count))
            if listing.isEmpty {
                strings.append(
                    nothingObservedHeadline(
                        objectClass: listing.envelope.objectClass,
                        recordable: recordability(of: listing.envelope.objectClass, in: listing.envelope)
                    )
                )
                strings.append(
                    nothingObservedExplanation(
                        objectClass: listing.envelope.objectClass,
                        recordable: recordability(of: listing.envelope.objectClass, in: listing.envelope)
                    )
                )
            }
            for observation in listing.observations {
                strings.append(contentsOf: everyString(for: observation))
            }

        case .lastSeen(let lastSeen):
            if let observation = lastSeen.observation {
                strings.append(contentsOf: everyString(for: observation))
            } else {
                strings.append(
                    nothingObservedHeadline(
                        objectClass: lastSeen.objectClass, recordable: lastSeen.recordable
                    )
                )
                strings.append(
                    nothingObservedExplanation(
                        objectClass: lastSeen.objectClass, recordable: lastSeen.recordable
                    )
                )
            }
        }

        return strings
    }

    /// Every string that does not depend on an answer: chrome, headings, and
    /// the four sentences the workspace can show without one.
    ///
    /// Checked by the same test, because a button label reaches a reader just
    /// as directly as a record row does.
    static let everyStaticString: [String] = [
        summary,
        recordsHeading,
        showEverythingButton,
        allCategories,
        nothingAskedYet,
        futureDescription,
        provenanceDisclosure,
        frameCaveat,
        towerUnreachable,
        noObjectMemoryConfigured,
        // The session half. Chrome here is copy in exactly the way a record
        // row is, and a Pause button whose surrounding sentence claims
        // liveness from intent would walk past a test that only read rows.
        sessionHeading,
        startMeaningLine,
        sessionPrimaryVerbsLiveAbove,
        sessionNotPersistedLine,
        noSessionControl,
        sessionUnsupported,
        sessionUnread,
        unreadableSessionAnswer,
        // The picture half, which is where an overclaim does the most damage.
        showTheWholeFrameButton,
        showTheDetectionButton,
        showThePictureButton,
        // All three retention sentences, not whichever one a fixture happens to
        // produce. The two stores have opposite lifetimes and the third case is
        // a Tower that names neither, so a test that saw only one of them would
        // be checking a third of this slot.
        keyframeRetentionLine,
        captureFrameRetentionLine,
        unnamedSourceRetentionLine,
        wholeFrameIsGoneLine,
        noPicturesOffered,
        unreadableImageryAnswer,
    ]
        + CartridgeSessionAction.allCases.map(actionButton)
        + CartridgeSessionAction.allCases.map(idempotentNoOpLine)
        // The composed lifecycle. Folded in here rather than given a test of
        // its own, so that the sentences written beside a Start button are held
        // to exactly the same claims as the sentences written beside a record.
        + everyRecordingString

    /// Every string shown for one record.
    static func everyString(for observation: ObjectObservation) -> [String] {
        var strings = [
            sightingHeadline(observation),
            timeLine(observation),
            claimLine(observation),
            confidenceLine(observation),
            frameLine(observation.frame),
            frameCaveat,
        ]
        if let box = boxLine(observation.frame) { strings.append(box) }
        if let imagery = imageryLine(observation.frame) { strings.append(imagery) }
        if let privacy = privacyLine(observation) { strings.append(privacy) }
        if let duration = durationLine(observation) { strings.append(duration) }
        if let tier = tierLine(observation) { strings.append(tier) }
        if let verification = verificationLine(observation) { strings.append(verification) }
        // Every caption this record could carry, whichever route is fetched.
        // The caption is the one string that travels *with a photograph*, so
        // it is the one the overclaim test most needs to see.
        for kind in ObjectMemoryImageryKind.allCases {
            strings.append(pictureCaption(objectClass: observation.objectClass, kind: kind))
        }
        return strings
    }

    /// Every string shown about one record's picture, or the absence of one.
    ///
    /// Separate from `everyString(for:)` because a description is fetched
    /// per row on demand and is not part of a listing payload — but it goes
    /// through the same forbidden-phrase test, because a sentence next to a
    /// first-person photograph is the most dangerous sentence this app writes.
    static func everyString(
        for description: ObjectMemoryImageryDescription, kind: ObjectMemoryImageryKind
    ) -> [String] {
        var strings = [
            pictureCaption(objectClass: description.objectClass, kind: kind),
            filterLine(description),
            regionsFilledLine(description),
            pictureRetentionLine(description),
            noPictureHeadline(description),
            noPictureExplanation(description),
        ]
        if let obscured = subjectObscuredLine(description, kind: kind) {
            strings.append(obscured)
        }
        // Shown in place of the frame control when the recording behind this
        // record is gone. Generated here rather than only when a fixture
        // happens to set the field, because it is a sentence beside a
        // photograph and that is the class of sentence this test exists for.
        if !description.frameCanBeAskedFor {
            strings.append(wholeFrameIsGoneLine)
        }
        return strings
    }

    /// Every string shown about one session reading.
    static func everyString(for snapshot: CartridgeSessionSnapshot) -> [String] {
        var strings = [
            intentLine(snapshot),
            livenessLine(snapshot),
            sessionProvenanceLine(snapshot),
            startMeaningLine,
            sessionPrimaryVerbsLiveAbove,
            sessionNotPersistedLine,
        ]
        if let contradiction = livenessContradictsIntentLine(snapshot) {
            strings.append(contradiction)
        }
        if let leftover = leftoverProducerLine(snapshot) {
            strings.append(leftover)
        }
        if let captures = capturesLine(snapshot) { strings.append(captures) }
        return strings
    }

    /// Every string shown for one refusal.
    static func everyString(for refusal: CartridgeSessionRefusal) -> [String] {
        [refusalLine(refusal), refusalProvenanceLine(refusal)]
            + everyString(for: refusal.snapshot)
    }

    /// Every string shown for one reading of the composed lifecycle.
    static func everyString(for reading: ObjectMemoryRecordingReading) -> [String] {
        [
            recordingHeadline(reading),
            recordingCameraLine(reading),
            recordingWhatStartDoes,
            recordingPauseMeaning,
            recordingPrimaryButton(.start),
            recordingPrimaryButton(.stop),
        ]
    }

    /// Every sentence the composed lifecycle can produce, across every phase
    /// and every camera claim.
    ///
    /// **Enumerated rather than sampled.** The phase and the camera claim are
    /// independent — a paused session beside a running camera is the pairing
    /// this cartridge most needs to word correctly — so the product of the two
    /// is generated instead of a handful of plausible combinations, and the
    /// ownership flag is varied under both because it changes which of two
    /// sentences about Stop is true.
    ///
    /// `.refused` is deliberately absent: its sentence is `refusalLine`, which
    /// `everyString(for: CartridgeSessionRefusal)` already runs through the same
    /// test with a real refusal behind it. Generating one here would need a
    /// fabricated snapshot in production code.
    static let everyRecordingString: [String] = {
        let unreachableTower = CartridgeFailure(
            kind: .transport, message: "The Tower did not answer."
        )
        let phases: [ObjectMemoryRecordingPhase] = [
            .idle,
            .starting,
            .waitingToBeFollowed,
            .notObserved,
            .remembering,
            .pausing,
            .paused,
            .resuming,
            .stopping,
            .stopped,
            .stillFollowing(after: .pause),
            .stillFollowing(after: .stop),
            .cannotTell(unreachableTower),
            .cameraRefused(.alreadyRunning),
            .cameraRefused(.deviceHasPausedCapture),
            .cameraRefused(.captureIsShuttingDown),
            .cameraRefused(.noActiveDevice),
            // Reachable only through the convergence loop's re-read — see
            // `ObjectMemoryRecordingCoordinator.converge`. It was in this list
            // before it was reachable at all, which is how a written sentence
            // spent months unable to appear on a screen.
            .cameraRefused(.cameraPermissionNotGranted),
            .cameraRefused(.datRefused("the device session could not be created")),
            .unsupported,
            .failed(unreachableTower),
        ]
        let claims: [CaptureClaim] = [.unclaimed, .running, .devicePaused, .ending]

        var strings: [String] = [
            recordingHeading,
            stopRememberingButton,
            recordingWhatStartDoes,
            recordingPauseMeaning,
            recordingCameraNotInThisBuild,
        ]
        for phase in phases {
            for claim in claims {
                for startedHere in [true, false] {
                    strings.append(
                        contentsOf: everyString(
                            for: ObjectMemoryRecordingReading(
                                phase: phase,
                                camera: claim,
                                cameraStartedHere: startedHere,
                                cameraIsReachable: true
                            )
                        )
                    )
                }
            }
            // The Release shape, where there is no camera to describe at all.
            strings.append(
                contentsOf: everyString(
                    for: ObjectMemoryRecordingReading(
                        phase: phase,
                        camera: .unclaimed,
                        cameraStartedHere: false,
                        cameraIsReachable: false
                    )
                )
            )
        }
        return strings
    }()

    /// An unfiltered listing narrows to no class, so there is no class whose
    /// recordability could be in question — `true` keeps the wording on "this
    /// window holds nothing" rather than "that was never looked for".
    static func recordability(
        of objectClass: String?, in envelope: ObjectMemoryEnvelope
    ) -> Bool {
        guard let objectClass else { return true }
        return envelope.isRecordable(objectClass)
    }

    // MARK: - Small formatting

    /// "a" or "an", by first letter.
    ///
    /// An imperfect rule — English has "an hour" and "a user" — and it runs
    /// over a class list that is currently `laptop` and `cell phone`. It is a
    /// rule rather than a lookup table because a table would have to be edited
    /// every time the Tower widens `PERSISTED_CLASSES`, and a stale table is a
    /// worse failure than a wrong article.
    static func indefiniteArticle(for word: String) -> String {
        let vowels: Set<Character> = ["a", "e", "i", "o", "u"]
        guard let first = word.lowercased().first else { return "a" }
        return vowels.contains(first) ? "an" : "a"
    }

    /// A capture id, shortened for reading. Never shortened to the point of
    /// ambiguity, and always shown behind the "Frame reference" label.
    static func shortened(_ identifier: String) -> String {
        guard identifier.count > 10 else { return identifier }
        return String(identifier.prefix(8)) + "…"
    }

    /// A detector score as a percentage. Scores are floats in 0…1 on this wire.
    ///
    /// "on this wire" described the Tower's intent, not what arrives. `Int(_:)`
    /// **traps** on NaN, on ±∞ and on anything past `Int.max`, and this is the
    /// highest-fan-out formatter in the cartridge: `best_score`,
    /// `detector_score`, verification scores, `bounding_box_normalized`, and
    /// `subject_obscured` — which is a **required** field on the imagery sheet,
    /// so a malformed value there crashed the picture view rather than
    /// degrading it.
    ///
    /// Deferred to the shared `ObservationProvenance.percent`, which guards
    /// finiteness and clamps, so the two do not disagree about what an
    /// out-of-range score looks like.
    static func percent(_ value: Double) -> String {
        ObservationProvenance.percent(value)
    }

    /// A retention window in days, without a trailing `.0` on a whole number
    /// and without pretending a fractional window is a whole one.
    static func days(_ value: Double) -> String {
        // `rounded == rounded.rounded()` below excludes NaN (every comparison
        // with it is false) but admits `+∞` and `1e300`, and `Int(_:)` traps on
        // both. Retention windows come off the wire like everything else.
        // A quantity, because both call sites embed this in a slot expecting
        // one -- "in the last \(days(x))" and "a window of \(days(x)) was asked
        // for". A bare noun phrase reads as "in the last an unreported window".
        guard value.isFinite else { return "an unusable number of days" }
        let rounded = (value * 10).rounded() / 10
        let text: String
        if rounded == rounded.rounded() {
            text = String(Int(rounded))
        } else {
            text = String(rounded)
        }
        return rounded == 1 ? "\(text) day" : "\(text) days"
    }

    /// A span of seconds, rounded to something a person reads rather than
    /// computes.
    ///
    /// Deliberately approximate and said to be — "about" is in every sentence
    /// that uses this — because both ends of the span are Tower-receipt times
    /// and the difference between them inherits whatever the network did.
    static func duration(_ seconds: TimeInterval) -> String {
        if seconds < 60 {
            let rounded = max(1, Int(seconds.rounded()))
            return rounded == 1 ? "1 second" : "\(rounded) seconds"
        }
        let minutes = Int((seconds / 60).rounded())
        return minutes == 1 ? "1 minute" : "\(minutes) minutes"
    }

    /// "laptop and cell phone", "a, b and c".
    static func list(_ items: [String]) -> String {
        switch items.count {
        case 0: return ""
        case 1: return items[0]
        case 2: return "\(items[0]) and \(items[1])"
        default:
            let head = items.dropLast().joined(separator: ", ")
            return "\(head) and \(items[items.count - 1])"
        }
    }
}
