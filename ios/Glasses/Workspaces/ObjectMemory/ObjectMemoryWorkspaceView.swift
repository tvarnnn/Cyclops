//
//  ObjectMemoryWorkspaceView.swift
//  Glasses
//

import Foundation
import SwiftUI

/// The Object Memory workspace: what the Tower recorded as visible, and when.
///
/// ## This view writes no prose
///
/// Every sentence on screen comes from `ObjectMemoryCopy`, including button
/// labels and section headings, and `ObjectMemoryCopyTests` asserts over that
/// type's whole output. A `Text("…")` literal added here would escape the test
/// that exists to stop this screen from claiming more than the sensor can
/// support — so the rule is absolute rather than stylistic: **no user-facing
/// string literal in this file.**
///
/// ## What the layout is arguing
///
/// A row leads with what a record actually says — *a laptop was visible*, and
/// when — and keeps the claim caveat beside it rather than in a disclosure. The
/// frame reference goes **inside** the disclosure, because it is the field a
/// reader most wants to misread as a place, and it is never shown without
/// `frameCaveat` immediately under it.
///
/// The bounding box is drawn as text, not as a rectangle over anything. There
/// is no map, no floor plan, and no marker: this cartridge has `spatial_ref:
/// null` in every payload and knows where nothing is.
///
/// ## One Start button, and what it starts
///
/// **This workspace can now begin a capture, and it is the third place in the
/// app that can.** That is a deliberate change to an invariant that used to
/// read "exactly two", and the reason is that the previous arrangement did not
/// work as a product.
///
/// It does not receive the `GlassesConnection` itself. It receives
/// `ObjectMemoryRecordingCoordinator`, built once on `ProjectManager` from the
/// one connection this app has — so the set of types that can reach DAT is
/// unchanged, and the coordinator outlives every cartridge switch.
///
/// The controls this screen grew on 2026-08-27 started and stopped the Tower's
/// **producer** — the process that reads a recording and writes observations —
/// and nothing else. So a person who opened this cartridge and pressed Start
/// got a session that was genuinely `active`, with `attached_capture_id: null`,
/// no frames, no records, and no sentence on screen telling them that the
/// camera lives behind a differently-named button on a different screen. The
/// button worked and the cartridge did not.
///
/// Now one tap runs `ObjectMemoryRecordingCoordinator`, which asks the Tower
/// first — the documented order, and the one that avoids both the closed-gate
/// race and `--attach-mode from-now` losing the opening seconds — and then
/// starts the camera **only if nothing already holds it**. There is still
/// exactly one camera pipeline and one owner of it: `GlassesConnection`. This
/// screen calls `startCameraSession()` and `stopCameraSession()` and knows
/// nothing else about DAT.
///
/// Stop honours ownership. A capture Home or World Builder started is left
/// running, because ending it from here would reach across two other screens,
/// and the copy says which of the two happened.
///
/// None of this makes the query half writable. It is still two `GET`s, real
/// deletion still lives on the Tower where a human types it against a store
/// they can name, and the session router still cannot touch the store.
///
/// The composed control exists because the alternative was measured. The
/// 2026-08-26 physical run remembered 64 real observations and **every one of
/// them required a person to find a capture directory and start a producer in a
/// second terminal.** That is not a product either.
///
/// ## The one rule this screen cannot get wrong
///
/// **Liveness is drawn from `following`, never from `state`.** The session
/// payload says so itself. A Pause whose producer ignores `SIGTERM` answers 200
/// with `state: "paused"` while the process keeps recording, so a Pause button
/// keyed on `state` tells a person they stopped being recorded when they did
/// not. Every liveness affordance below reads `memory.liveness`, which reads
/// `following`, and the contradiction between the two is shown loudly rather
/// than reconciled.
struct ObjectMemoryWorkspaceView: View {
    /// A value, not the connection — see `TowerReachabilityReader`.
    let isTowerReachable: Bool

    @StateObject private var memory: ObjectMemoryViewModel

    /// The composed lifecycle: the Tower's producer and the glasses camera,
    /// started and stopped together and in the right order.
    ///
    /// **Observed, not owned**, and the asymmetry with `memory` above it is
    /// the point.
    ///
    /// `ObjectMemoryViewModel` is a `@StateObject` and stays one: it holds a
    /// query and a list of answers, and losing those on a cartridge switch
    /// costs a person one tap. This holds whether this app started the capture
    /// that is running — and losing THAT costs a person a Stop button that
    /// leaves the glasses recording, because a rebuilt coordinator believes it
    /// started nothing and therefore stops nothing.
    ///
    /// So it is built once on `ProjectManager` and handed in. That is the rule
    /// `CartridgeClients` was created for, and this is the first object in the
    /// app for which it is not a precaution.
    @ObservedObject private var recording: ObjectMemoryRecordingCoordinator

    /// The clients are injected and owned by `ProjectManager`; see
    /// `CartridgeClients`. They outlive this view, so an answer survives a
    /// workspace switch.
    ///
    /// The coordinator likewise. Its own `camera` is what is optional: a
    /// preview, a Release build, and any future host with no camera all build
    /// one with `nil` there, and the Tower half of this screen works exactly
    /// as it did. The camera half then reports that it cannot be reached
    /// rather than pretending it can.
    ///
    /// `memory` is constructed through `StateObject`'s autoclosure and
    /// therefore lazily, once per installation, which is why the coordinator
    /// takes the *client* rather than the view model: sharing one view-model
    /// instance between them would mean building it eagerly on every `init`,
    /// and `init` runs on every re-evaluation of the view above this one.
    ///
    /// No default for `recording`, deliberately. A default would let a caller
    /// build this screen with a coordinator nobody else can see, which is the
    /// mistake this parameter exists to prevent.
    init(
        isTowerReachable: Bool,
        client: any ObjectMemoryClient,
        recording: ObjectMemoryRecordingCoordinator
    ) {
        self.isTowerReachable = isTowerReachable
        self.recording = recording
        _memory = StateObject(wrappedValue: ObjectMemoryViewModel(client: client))
    }

    private var forcedPhase: CartridgePhase? {
        memory.knownAvailability(isTowerReachable: isTowerReachable)?.forcedPhase
    }

    var body: some View {
        VStack(spacing: 16) {
            header
            // Above the query controls deliberately. Whether this memory is
            // being written into right now outranks what it already holds:
            // somebody who opens this screen to check they are not being
            // recorded should not have to scroll past a list of records to
            // find out.
            //
            // The control comes first and the Tower's own reading second. That
            // ordering is the same argument one level up: a person looking at
            // this screen wants the one sentence that says whether anything is
            // being written into this memory, and then the evidence for it.
            ObjectMemoryRecordingPanel(recording: recording)
            ObjectMemorySessionPanel(memory: memory, recording: recording)
            controls

            if let forcedPhase {
                CartridgeStatePanel(
                    title: ObjectMemoryCopy.recordsHeading,
                    phase: forcedPhase,
                    explanation: memory.unavailableExplanation(isTowerReachable: isTowerReachable),
                    futureDescription: ObjectMemoryCopy.futureDescription
                )
            } else {
                recordsPanel
            }
        }
        // Liveness has a shelf life: `following` names what a producer is alive
        // on now, and a value read once when the screen opened is a claim about
        // a process that may have died since. Bounded by cancellation on
        // disappear rather than running for the life of the app.
        .onAppear {
            memory.startWatchingSession()
            // Set here rather than in `init` so the coordinator never holds a
            // reference to a view model built for a render that was discarded.
            //
            // `askAgain()` and not `askForEverything()`. The comment here used
            // to say it re-asked the reader's own question while the call
            // asked for a listing, so a reader who had narrowed with
            // `askWhenLastInView()` pressed Stop and silently got a different
            // question's answer back. The behaviour was changed to match the
            // comment rather than the other way round, because the comment was
            // stating the rule `ObjectMemoryRecordingCoordinator.refreshRecords`
            // exists to keep: the coordinator owns no question, the view does,
            // and a Stop must not widen one.
            recording.refreshRecords = { memory.askAgain() }
        }
        .onDisappear {
            memory.stopWatchingSession()
            // **The coordinator outlives this view.** It is built once on
            // `ProjectManager` (see `recording` above) precisely so that a
            // cartridge switch cannot lose the fact that this app started a
            // capture — which means a closure left on it also survives the
            // switch, holding the `@StateObject` view model of a screen that is
            // gone. Every later Stop would then re-ask a question for a view
            // nobody is looking at, and the view model would stay alive to
            // answer it.
            //
            // Cleared here rather than made weak: the closure is a *decision*
            // this view makes, and a view that is no longer on screen has no
            // decision to contribute. `refreshRecords?()` is optional-chained
            // on the coordinator, so `nil` is an ordinary, supported state and
            // a Stop with no view attached simply refreshes nothing.
            recording.refreshRecords = nil
        }
    }

    // MARK: Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(ObjectMemoryCopy.cartridgeName)
                .font(.title2.weight(.semibold))
            Text(ObjectMemoryCopy.summary)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    // MARK: Controls

    /// Two questions and a category picker.
    ///
    /// The picker's options come from the last answer's `recorded_classes`
    /// rather than from a constant in this app. Hardcoding `laptop` and
    /// `cell phone` would be the phone asserting what the Tower looks for, and
    /// it would go stale silently the day the Tower's list changes.
    @ViewBuilder
    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !memory.recordableClasses.isEmpty {
                Picker(ObjectMemoryCopy.allCategories, selection: $memory.selectedClass) {
                    Text(ObjectMemoryCopy.allCategories).tag(String?.none)
                    ForEach(memory.recordableClasses, id: \.self) { objectClass in
                        Text(objectClass).tag(String?.some(objectClass))
                    }
                }
                .pickerStyle(.segmented)
            }

            HStack(spacing: 10) {
                Button(ObjectMemoryCopy.showEverythingButton) {
                    memory.askForEverything()
                }
                .buttonStyle(.borderedProminent)

                if let objectClass = memory.selectedClass {
                    Button(ObjectMemoryCopy.lastInViewButton(objectClass)) {
                        memory.askWhenLastInView()
                    }
                    .buttonStyle(.bordered)
                }
            }
            .disabled(memory.state.phase == .waiting)

            // Shown rather than disabling the buttons. Reachability here is the
            // *socket's*, and object memory travels over HTTP — refusing to ask
            // on the strength of a different transport's state would hide a
            // request that might well succeed. So the caveat is stated and the
            // control stays live; a request that does fail lands in
            // `.failed(.transport)`, which the shell already draws as
            // disconnected rather than as an error.
            if !isTowerReachable {
                Text(ObjectMemoryCopy.towerUnreachable)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .contain)
    }

    // MARK: Records

    @ViewBuilder
    private var recordsPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(ObjectMemoryCopy.recordsHeading)

            VStack(alignment: .leading, spacing: 12) {
                switch memory.state {
                case .idle:
                    Text(ObjectMemoryCopy.nothingAskedYet)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                case .asking(let question):
                    HStack(spacing: 10) {
                        ProgressView()
                        Text(ObjectMemoryCopy.questionLine(question))
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }

                case .answered(let question, let answer):
                    ObjectMemoryAnswerView(
                        question: question, answer: answer, memory: memory
                    )

                case .noObjectMemory:
                    // Unreachable while `forcedPhase` is consulted above, and
                    // written anyway: the day this state stops forcing a phase,
                    // the panel must not fall through to a blank box.
                    Text(ObjectMemoryCopy.noObjectMemoryConfigured)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                case .failed(let failure):
                    Text(failure.message)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
        }
    }
}

/// One answer: what was asked, what window it came from, and the records — or
/// the honest silence.
struct ObjectMemoryAnswerView: View {
    let question: ObjectMemoryQuestion
    let answer: ObjectMemoryAnswer
    /// Held so a row can make its own picture loader. The row never touches a
    /// client itself — see `ObjectMemoryViewModel.pictureLoader(for:)`.
    @ObservedObject var memory: ObjectMemoryViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(ObjectMemoryCopy.questionLine(question))
                    .font(.subheadline.weight(.medium))
                Text(ObjectMemoryCopy.retentionLine(answer.envelope.retention))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let clamp = ObjectMemoryCopy.clampLine(answer.envelope.retention) {
                    Text(clamp)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            switch answer {
            case .listing(let listing):
                if listing.isEmpty {
                    // `ObjectMemoryCopy.recordability` rather than an inline
                    // `.map(envelope.isRecordable)`: a method reference handed
                    // to `map` is called from a nonisolated context under this
                    // target's default `MainActor` isolation, which is the same
                    // trap `WorldBuilderResultDecoder` documents.
                    ObjectMemoryEmptyView(
                        objectClass: listing.envelope.objectClass,
                        recordable: ObjectMemoryCopy.recordability(
                            of: listing.envelope.objectClass, in: listing.envelope
                        )
                    )
                } else {
                    Text(ObjectMemoryCopy.recordCountLine(listing.observations.count))
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                    ForEach(listing.observations) { observation in
                        ObjectObservationRow(observation: observation, memory: memory)
                    }
                }

            case .lastSeen(let lastSeen):
                if let observation = lastSeen.observation {
                    ObjectObservationRow(observation: observation, memory: memory)
                } else {
                    ObjectMemoryEmptyView(
                        objectClass: lastSeen.objectClass, recordable: lastSeen.recordable
                    )
                }
            }

            Text(ObjectMemoryCopy.recordableClasses(answer.envelope.recordedClasses))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)

            // What the Tower actually reported about itself, composed only
            // from what the payload carried — including the fact that it does
            // not name a detector, which this app must not fill in.
            Text(ObjectMemoryCopy.whatThisTowerReports(answer))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// An answer that carries no record.
///
/// **Not an error, and not "no results".** A missing record means the camera
/// was not pointed at one, or the detector scored it below threshold, or the
/// class is not one this cartridge ever writes, or the retention window closed.
/// None of those is "the object is not there", and the two wordings this view
/// chooses between are the two different silences.
struct ObjectMemoryEmptyView: View {
    let objectClass: String?
    let recordable: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(
                ObjectMemoryCopy.nothingObservedHeadline(
                    objectClass: objectClass, recordable: recordable
                ),
                systemImage: "questionmark.circle"
            )
            .font(.headline)

            Text(
                ObjectMemoryCopy.nothingObservedExplanation(
                    objectClass: objectClass, recordable: recordable
                )
            )
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

/// One record.
///
/// The headline and the claim are always visible; the frame reference and the
/// detection numbers sit behind a disclosure. That ordering is the argument:
/// what the record *says* comes first, and the identifiers a reader would
/// mistake for a location are one deliberate tap away and never appear without
/// `frameCaveat` under them.
struct ObjectObservationRow: View {
    let observation: ObjectObservation
    @ObservedObject var memory: ObjectMemoryViewModel

    /// Collapsed by default. Expanded state is per-row and not persisted:
    /// nothing about which rows a person opened is worth keeping.
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(ObjectMemoryCopy.sightingHeadline(observation))
                .font(.body.weight(.medium))

            Text(ObjectMemoryCopy.timeLine(observation))
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Text(ObjectMemoryCopy.claimLine(observation))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)

            // Only when this Tower offered imagery routes **and** the record
            // carries a handle. Neither is assumed: the routes are read off the
            // envelope, and a record with no handle simply gets no picture
            // rather than taking the row down with it.
            if memory.offersPictures, let observationID = observation.observationID {
                ObjectMemoryPictureView(
                    objectClass: observation.objectClass,
                    loader: memory.pictureLoader(for: observationID)
                )
            }

            DisclosureGroup(isExpanded: $isExpanded) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(ObjectMemoryCopy.frameLine(observation.frame))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    // Never conditional. A frame reference without its caveat
                    // is the failure mode this whole screen is built against.
                    Text(ObjectMemoryCopy.frameCaveat)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)

                    if let box = ObjectMemoryCopy.boxLine(observation.frame) {
                        Text(box)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    if let imagery = ObjectMemoryCopy.imageryLine(observation.frame) {
                        Text(imagery)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    Text(ObjectMemoryCopy.confidenceLine(observation))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)

                    // How long the sighting lasted, which policy tier admitted
                    // the record, and what — if anything — agreed with it. All
                    // three are `nil` on a record that does not carry them, and
                    // none of them is invented for one that does not.
                    ForEach(
                        [
                            ObjectMemoryCopy.durationLine(observation),
                            ObjectMemoryCopy.tierLine(observation),
                            ObjectMemoryCopy.verificationLine(observation),
                        ].compactMap { $0 },
                        id: \.self
                    ) { line in
                        Text(line)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    if let privacy = ObjectMemoryCopy.privacyLine(observation) {
                        Text(privacy)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
            } label: {
                Text(ObjectMemoryCopy.provenanceDisclosure)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 4)
        .accessibilityElement(children: .contain)
    }
}

// MARK: - The session

/// Start, Pause, Resume and Stop — and, above them, whether this memory is
/// being written into.
///
/// ## The ordering is the argument
///
/// The **liveness** line comes first and is drawn from `following`. The
/// **intent** line comes second and is drawn from `state`, labelled as what was
/// asked for. That is the opposite of the order a control panel usually takes,
/// and it is deliberate: a person looking at this panel wants to know whether
/// they are being recorded, and `state` cannot answer that question. It answers
/// a different one that looks identical.
///
/// When the two contradict each other in the harmful direction — the Tower
/// reporting a Pause as honoured while a producer is still alive on a recording
/// — the contradiction is shown in a warning role above everything else. That
/// case was reproduced during the 2026-08-27 integration and is the reason this
/// whole surface exists.
///
/// ## This view writes no prose either
///
/// Same rule as the workspace: every sentence comes from `ObjectMemoryCopy`,
/// including the button labels, and `ObjectMemoryCopyTests` asserts over that
/// type's whole output.
struct ObjectMemorySessionPanel: View {
    @ObservedObject var memory: ObjectMemoryViewModel
    /// The verbs below are sent through the composed lifecycle rather than
    /// straight at the Tower, so that a Stop pressed here stops the capture
    /// this cartridge started and a Start pressed here starts one. There are no
    /// raw Tower verbs left on this screen; what is read off the Tower is still
    /// the *vocabulary*, which is what `actions` is for.
    @ObservedObject var recording: ObjectMemoryRecordingCoordinator

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(ObjectMemoryCopy.sessionHeading)

            VStack(alignment: .leading, spacing: 10) {
                switch memory.session {
                case .unread:
                    caption(ObjectMemoryCopy.sessionUnread)

                case .working:
                    HStack(spacing: 10) {
                        ProgressView()
                        caption(ObjectMemoryCopy.sessionUnread)
                    }

                case .known(let snapshot):
                    reading(snapshot, refusal: nil)

                case .refused(let refusal):
                    reading(refusal.snapshot, refusal: refusal)

                case .noSessionControl:
                    caption(ObjectMemoryCopy.noSessionControl)

                case .failed(let failure):
                    caption(failure.message)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
        }
    }

    /// One reading, with the liveness claim above the intent claim.
    @ViewBuilder
    private func reading(
        _ snapshot: CartridgeSessionSnapshot, refusal: CartridgeSessionRefusal?
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            // The harmful contradiction, first and in the warning role. Not a
            // caption and not tertiary: this is the one sentence on the screen
            // that corrects a false belief a person may already be holding.
            if let contradiction = ObjectMemoryCopy.livenessContradictsIntentLine(snapshot) {
                Label(contradiction, systemImage: "exclamationmark.triangle.fill")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // A producer this session did not start and cannot stop. Beside
            // the contradiction and in the same warning role, because it is
            // the same class of sentence: something is recording that the
            // controls on this screen will not reach.
            if let leftover = ObjectMemoryCopy.leftoverProducerLine(snapshot) {
                Label(leftover, systemImage: "exclamationmark.triangle")
                    .font(.subheadline)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // Liveness, from what THIS session started. The glyph is filled
            // only when a producer this session started is actually alive on a
            // recording — never when `state` merely says `active`, and never
            // for a producer some earlier session left behind.
            Label(
                ObjectMemoryCopy.livenessLine(snapshot),
                systemImage: snapshot.isFollowingACapture
                    ? "record.circle.fill" : "record.circle"
            )
            .font(.subheadline)
            .foregroundStyle(snapshot.isFollowingACapture ? .primary : .secondary)
            .fixedSize(horizontal: false, vertical: true)

            // Intent, second and labelled as intent by its own sentence.
            caption(ObjectMemoryCopy.intentLine(snapshot))

            if let refusal {
                // A refusal is information, not an error banner: it says which
                // control would have worked. An idempotent no-op never reaches
                // here at all — that is a 200 and lands in `.known`.
                VStack(alignment: .leading, spacing: 4) {
                    Text(ObjectMemoryCopy.refusalLine(refusal))
                        .font(.subheadline)
                        .fixedSize(horizontal: false, vertical: true)
                    caption(ObjectMemoryCopy.refusalProvenanceLine(refusal))
                }
            }

            controls(snapshot)

            if !snapshot.supported {
                caption(ObjectMemoryCopy.sessionUnsupported)
            }
            caption(ObjectMemoryCopy.sessionProvenanceLine(snapshot))
            if let captures = ObjectMemoryCopy.capturesLine(snapshot) {
                caption(captures)
            }
            caption(ObjectMemoryCopy.startMeaningLine)
            caption(ObjectMemoryCopy.sessionNotPersistedLine)
        }
    }

    /// The verbs, read off the Tower's own `actions` list rather than
    /// hard-coded — which is what the field is for, and what lets a Tower that
    /// grows a fifth verb be noticed rather than silently half-rendered.
    ///
    /// Nothing is hidden by state. `start` from `active` is a legal idempotent
    /// no-op, `stop` is never refused, and disabling a control on this app's
    /// guess about what the Tower will accept would make this app's model of
    /// the state machine authoritative over the Tower's.
    ///
    /// ## The two verbs that are not drawn here, and why that is not the same
    ///
    /// `ObjectMemoryRecordingPanel.primaryControl`, immediately above this
    /// panel, already renders one of `start` / `stop` prominently. Drawing the
    /// whole vocabulary here put a **second prominent Start** and a **second
    /// Stop** — labelled differently, because the composed one names the camera
    /// and this one does not — under the first. Four primary-looking controls,
    /// two of them the same verb twice, on the screen whose whole job is to say
    /// plainly whether somebody is being recorded.
    ///
    /// The paragraph above argues against *disabling* a control on a guess, and
    /// it still stands: this is not that. Nothing here is hidden because this
    /// app thinks the Tower would refuse it — the Tower's answer is unchanged,
    /// the verbs are still read off `actions`, `unofferedActions` still names
    /// anything this build has no button for, and both verbs are still one tap
    /// away in the control above. **Not drawing a second copy of a control that
    /// is already on screen removes a duplicate, not a capability.**
    ///
    /// So the filter is on the *primary control's* vocabulary rather than on a
    /// hard-coded pair: whatever the Tower offers that the composed control
    /// does not already carry is drawn, which today is Pause and Resume and
    /// tomorrow is a fifth verb nobody has written yet.
    private func controls(_ snapshot: CartridgeSessionSnapshot) -> some View {
        let secondary = snapshot.offeredActions.filter { !Self.primaryVerbs.contains($0) }
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                ForEach(secondary, id: \.rawValue) { action in
                    // Every verb left after the filter is a secondary one, so
                    // there is no longer a style to choose between — the
                    // written-out `if action == .start` branch that used to
                    // live here existed only because `buttonStyle` is generic
                    // over its argument and the two concrete styles have no
                    // common type. Keeping `.bordered` uniform is what makes
                    // the control above unambiguously the primary one.
                    Button(ObjectMemoryCopy.actionButton(action)) {
                        recording.apply(action)
                    }
                    .buttonStyle(.bordered)
                }
            }
            // Unconditional, including when the filter left nothing to draw:
            // an empty control row under a "Session" heading reads as a
            // surface that failed to render, and this sentence is what says
            // where the missing verbs went.
            caption(ObjectMemoryCopy.sessionPrimaryVerbsLiveAbove)
        }
        // `isActing` is a **local** fact — a mutating step this screen started
        // is in flight — and not a guess about what the Tower would accept, so
        // gating on it does not make this app's model authoritative over the
        // Tower's. It is what stops a double tap starting two overlapping
        // sequences that would race over who owns the camera.
        //
        // It covers the `POST` and the camera call **only**. It used to cover a
        // Start's whole sequence, including a convergence poll that runs to
        // twelve seconds — so this row and the primary control above were both
        // dead for twelve seconds after every Start, over a live session and a
        // camera this screen had just turned on. See
        // `ObjectMemoryRecordingCoordinator.isActing`.
        .disabled(!snapshot.supported || recording.isActing)
    }

    /// The verbs the composed control above already carries, and which this
    /// panel therefore does not draw a second copy of.
    ///
    /// A constant rather than an inline pair so that the filter and
    /// `ObjectMemoryRecordingCoordinator.primaryAction` — which returns exactly
    /// these two — cannot drift into a state where a verb is drawn twice or
    /// not at all.
    private static let primaryVerbs: Set<CartridgeSessionAction> = [.start, .stop]

    private func caption(_ text: String) -> some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}

// MARK: - The composed lifecycle

/// One Start, one Stop, and the sentence that says what they actually did.
///
/// ## Why this is a second panel and not a fifth button
///
/// The panel below it reports what the *Tower* says: intent, liveness,
/// provenance, and the contradiction between the first two. This one reports
/// what a *person* asked for and what came of it across both halves — the
/// producer and the camera — which is a different question with a different
/// answer, and frequently a longer one. Folding them together would put a
/// sentence about a DAT stream under a heading that says "what the Tower
/// reports", which would be false.
///
/// ## Nothing here is set optimistically
///
/// Every string is drawn from `ObjectMemoryRecordingCoordinator.reading`, which
/// is only ever written from an answer — a `POST` result, a read-back, a camera
/// claim published by `GlassesConnection`, or a deadline expiring. A tap sets
/// `starting`, `pausing`, `resuming` or `stopping`, and each of those says
/// *asking*, not *done*.
///
/// The primary control is deliberately still enabled in states this app
/// believes the Tower would refuse, for the reason the session panel gives:
/// `start` from `active` is a legal idempotent no-op, `stop` is never refused,
/// and disabling on a guess would make this app's model of the state machine
/// authoritative over the Tower's. The two things it *is* disabled on are a
/// Tower that has no producer at all, and a **mutating step** this screen has
/// already started — both local facts, and the second one deliberately narrower
/// than "a sequence", which is what it used to be. See `primaryControl`.
///
/// ## This panel's control is the primary one, and the only one
///
/// `ObjectMemorySessionPanel` below it draws the Tower's vocabulary and used to
/// draw all of it, including a second prominent Start and a second Stop. It now
/// draws everything except the verbs this control already carries, and says so
/// in a sentence. See `ObjectMemorySessionPanel.controls`.
///
/// ## This view writes no prose either
///
/// Same absolute rule as the workspace: every sentence comes from
/// `ObjectMemoryCopy`, and `ObjectMemoryCopyTests` runs the whole product of
/// phase and camera claim through the claims this cartridge may not make.
struct ObjectMemoryRecordingPanel: View {
    @ObservedObject var recording: ObjectMemoryRecordingCoordinator

    var body: some View {
        // Read once into a local so every line below is drawn from the same
        // instant — the same discipline `HomeWorkspaceView.liveNumbers` applies
        // to a metrics snapshot. A phase and a camera claim sampled a render
        // apart could contradict each other on screen.
        let reading = recording.reading
        return VStack(alignment: .leading, spacing: 8) {
            SectionLabel(ObjectMemoryCopy.recordingHeading)

            VStack(alignment: .leading, spacing: 10) {
                headline(reading)
                primaryControl(reading)
                caption(ObjectMemoryCopy.recordingCameraLine(reading))
                caption(ObjectMemoryCopy.recordingWhatStartDoes)
                caption(ObjectMemoryCopy.recordingPauseMeaning)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
        }
    }

    /// The one sentence, with a glyph that is filled **only** when a producer
    /// is confirmed alive on a recording.
    ///
    /// `isFollowingACapture` is a property of the phase and is `false` for
    /// `starting`, `waitingToBeFollowed` and `notObserved` — all three of which
    /// follow a *successful* Start. A filled record dot on any of them would be
    /// this screen claiming a recording it has never observed.
    private func headline(_ reading: ObjectMemoryRecordingReading) -> some View {
        Label(
            ObjectMemoryCopy.recordingHeadline(reading),
            systemImage: reading.phase.isFollowingACapture
                ? "record.circle.fill" : "record.circle"
        )
        .font(.subheadline)
        .foregroundStyle(reading.phase.isFollowingACapture ? .primary : .secondary)
        .fixedSize(horizontal: false, vertical: true)
    }

    /// **The one control on this screen that must never be unreachable while a
    /// capture is running.**
    ///
    /// The two things it is disabled on are both local facts: a Tower that has
    /// no producer at all, and a *mutating* step this screen started that has
    /// not returned.
    ///
    /// That second one used to be the whole sequence. A Start's sequence ends
    /// with `converge()`, which runs to twelve seconds, so this button — the
    /// Stop button, for those twelve seconds — was disabled over a Tower
    /// session that was open and a glasses camera this very screen had just
    /// started. A wearer who changed their mind could only leave the screen.
    /// The gate now covers the `POST` and the camera call and nothing else, and
    /// a Stop that lands during convergence cancels the poll and runs. See
    /// `ObjectMemoryRecordingCoordinator.isActing` and `perform`.
    private func primaryControl(_ reading: ObjectMemoryRecordingReading) -> some View {
        let action = recording.primaryAction
        return Button(ObjectMemoryCopy.recordingPrimaryButton(action)) {
            recording.apply(action)
        }
        .buttonStyle(.borderedProminent)
        .disabled(reading.phase == .unsupported || recording.isActing)
    }

    private func caption(_ text: String) -> some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}

// MARK: - A record's picture

/// The picture behind one record, or the sentence explaining why there is not
/// one.
///
/// ## Every branch here is a sentence, and none of them is a broken image
///
/// The four things that can come back are a picture, "the memory is kept and
/// the picture is gone" (410), "this Tower is serving no pictures" (503) and
/// "nothing under this handle" (404). Three of those used to arrive as
/// `.transport` and render as *the Tower is unreachable* — which is worse than
/// a broken image, because it sends a person to check a network cable about a
/// machine that answered them clearly and truthfully.
///
/// ## The caption is not optional
///
/// A picture is a much stronger location cue than a sentence, and no string
/// test can catch it. `ObjectMemoryCopy.pictureCaption` appears under **every**
/// picture this view draws, unconditionally, in the same way `frameCaveat`
/// appears under every frame reference. The filter line and the fill lines go
/// with it, because a filtered picture described only by what the filter
/// catches reads as a guarantee.
///
/// ## Nothing is kept
///
/// Bytes live on the loader, the loader is this view's `@StateObject`, and
/// `onDisappear` drops them rather than waiting for deallocation. There is no
/// shared image cache keyed by observation id anywhere in this app, and adding
/// one would undo the `Cache-Control: no-store` the Tower sends.
struct ObjectMemoryPictureView: View {
    let objectClass: String
    @StateObject var loader: ObjectMemoryPictureLoader

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            switch loader.phase {
            case .unasked:
                // Not fetched until the row appears, and drawn as a control
                // rather than as an empty box: a person should be able to see
                // that a picture exists to be asked for.
                Button(ObjectMemoryCopy.showThePictureButton) { loader.load() }
                    .font(.caption)
                    .buttonStyle(.bordered)

            case .loading:
                ProgressView()

            case .picture(let data, let description):
                picture(data, description)

            case .noPicture(let description):
                noPicture(description)

            case .noPicturesOffered:
                caption(ObjectMemoryCopy.noPicturesOffered)

            case .failed(let failure):
                caption(failure.message)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onAppear { loader.load() }
        // The row leaving the screen is exactly when this app should stop
        // holding a photograph of somebody's home.
        .onDisappear { loader.forget() }
    }

    @ViewBuilder
    private func picture(
        _ data: Data, _ description: ObjectMemoryImageryDescription
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            if let image = UIImage(data: data) {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity)
                    .clipShape(.rect(cornerRadius: 12))
                    // The caption is the accessibility label too. A
                    // screen-reader user reaching an unlabelled image gets the
                    // weakest possible version of this screen.
                    .accessibilityLabel(
                        ObjectMemoryCopy.pictureCaption(
                            objectClass: description.objectClass ?? objectClass,
                            kind: loader.kind
                        )
                    )
            } else {
                // Bytes that will not decode. Said out loud rather than left as
                // the blank rectangle `Image` would otherwise draw, which on
                // this screen would read as "there was nothing there".
                caption(ObjectMemoryCopy.unreadableImageryAnswer)
            }

            // Unconditional, and directly under the picture. This is the whole
            // burden — see the type's documentation.
            Text(
                ObjectMemoryCopy.pictureCaption(
                    objectClass: description.objectClass ?? objectClass, kind: loader.kind
                )
            )
            .font(.caption)
            .fixedSize(horizontal: false, vertical: true)

            if let obscured = ObjectMemoryCopy.subjectObscuredLine(
                description, kind: loader.kind
            ) {
                caption(obscured)
            }
            caption(ObjectMemoryCopy.filterLine(description))
            caption(ObjectMemoryCopy.regionsFilledLine(description))
            // Per source, because a picture this memory owns and a render out
            // of the capture store now have opposite lifetimes. See
            // `ObjectMemoryCopy.pictureRetentionLine`.
            caption(ObjectMemoryCopy.pictureRetentionLine(description))

            // The other route, so a person who wants the context or the detail
            // can have it. Both come with the same caption.
            //
            // **Except when the Tower has said the context view is gone.**
            // `/crop` can now be served from a keyframe this memory owns, which
            // outlives the recording behind it, while `/frame` still comes from
            // that recording and honestly 410s once it has been deleted. A
            // "Show the whole frame" control in that state sends a tap to a
            // route that will refuse, and the wearer reads "the memory is kept
            // and the picture is gone" over a picture that is on their screen.
            //
            // Only an explicit `frame_available: false` closes it —
            // `frameCanBeAskedFor` treats an absent or null field as unknown
            // and keeps the control, because an older Tower has not said the
            // frame is missing and this app must not decide that for it.
            if loader.kind == .crop && !description.frameCanBeAskedFor {
                caption(ObjectMemoryCopy.wholeFrameIsGoneLine)
            } else {
                Button(
                    loader.kind == .crop
                        ? ObjectMemoryCopy.showTheWholeFrameButton
                        : ObjectMemoryCopy.showTheDetectionButton
                ) {
                    loader.show(loader.kind == .crop ? .frame : .crop)
                }
                .font(.caption)
                .buttonStyle(.bordered)
            }
        }
    }

    /// A refusal, rendered as the sentence it is.
    ///
    /// A neutral glyph, never a warning triangle and never a broken-image
    /// symbol: a 410 here is a **correct** outcome of capture-side retention
    /// doing what it is supposed to do, and dressing it as a failure would tell
    /// a person something went wrong when nothing did.
    private func noPicture(_ description: ObjectMemoryImageryDescription) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(
                ObjectMemoryCopy.noPictureHeadline(description),
                systemImage: "photo.on.rectangle.angled"
            )
            .font(.subheadline.weight(.medium))
            .fixedSize(horizontal: false, vertical: true)

            Text(ObjectMemoryCopy.noPictureExplanation(description))
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    private func caption(_ text: String) -> some View {
        Text(text)
            .font(.caption2)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}
