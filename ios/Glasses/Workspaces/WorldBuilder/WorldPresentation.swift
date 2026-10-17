//
//  WorldPresentation.swift
//  Glasses
//

import Foundation

// MARK: - Why this file exists

/// The World Builder screen's state, separated from the views that draw it.
///
/// An empty fragment array can mean five different things: (1) no geometry
/// revision (WorldBuilderResultDecoder.geometryCoordinates requires a complete
/// address), (2) a session gate withholding an address, (3) a failed manifest
/// fetch, (4) one unreadable row making WorldGeometryDecoder.manifest refuse
/// the whole manifest, or (5) a pose convention this build cannot draw. The array is derived
/// from `geometryStatus`; `geometryDidChange` assigns `.loaded` during the fetch.
/// The Tower once reported 463 keyframes, 467 camera poses and 17,674 points
/// while an empty array made it say "Nothing mapped yet." That headline now
/// requires a Tower 404 or a loaded status with no drawable or unresolved
/// fragments, without a positive Tower geometry claim. Geometry fetch failures
/// had only DEBUG output, so a Release screen must represent the distinction
/// in state.
///
/// WorldGeometryStatus carries the phone's fetch result, WorldEvidence the
/// Tower's independent claim, and WorldGeometryAccount derives their wording.
/// A snapshot reporting geometry can never produce the nothing-mapped account.
///
/// These are pure values: no view, client, URLSession or clock. Tests can
/// exercise the whole ladder without a Tower or a rendered screen. They use
/// the target's MainActor isolation to match the world model types they read;
/// WorldRenderTarget stays nonisolated because it only addresses a URL.

// MARK: - The fetch, as a state rather than as an empty array

/// Where the phone is in going and getting the geometry the Tower named.
///
/// The ladder is honest about ignorance: `.notAddressed` is *the phone does not
/// have an address*, `.towerReportsNone` is *the Tower answered that it has
/// nothing*, and those are different answers to the reader even though both
/// leave the gallery empty.
///
/// `.loaded` carries the model **and** how many segments the manifest named
/// whose points could not be fetched, because a gallery drawn from 29 of 34
/// segments is a true picture of part of a world and must say which part is
/// missing rather than reading as the whole.
enum WorldGeometryStatus: Equatable {
    /// There is no world on screen to have geometry for: `.idle`, `.failed`,
    /// `.unsupported`, or nothing reported yet.
    case noWorld
    /// A world is on screen and no geometry address has arrived for it. The
    /// Tower sends `geometry.revision: null` until a build has produced output
    /// for the session, and the client suppresses the address for a report the
    /// session gate refused — this state covers both, and the account below
    /// says only what is true of both.
    case notAddressed
    /// An address arrived and a fetch is out.
    case loading
    /// A manifest arrived, decoded, and its segments are drawn. Zero drawable
    /// segments here is a real answer — the Tower has a manifest and nothing in
    /// it resolved — and is **not** the same as any state above it.
    case loaded(WorldFragmentsModel, unfetched: Int)
    /// The Tower answered 404 on the manifest route: it has no geometry for
    /// this session. Its own words, when it sent any.
    case towerReportsNone(detail: String?)
    /// The fetch happened and did not produce a manifest this build could use.
    case failed(WorldGeometryFailure)

    /// The fragments to draw, or an empty model. The gallery reads this for
    /// drawable geometry and keeps an empty model for every other status.
    var fragments: WorldFragmentsModel {
        if case .loaded(let model, _) = self { return model }
        return WorldFragmentsModel(segments: [])
    }

    /// Whether the phone holds geometry it can draw. Distinct from
    /// `WorldEvidence.hasGeometry`, which is what the **Tower** says exists:
    /// those two can disagree, so neither can stand in for the other.
    var hasDrawableGeometry: Bool {
        if case .loaded(let model, _) = self { return !model.fragments.isEmpty }
        return false
    }

    /// Whether a fetch for this world has produced **any** answer yet — a
    /// manifest, the Tower's own 404, or a failure.
    ///
    /// Deliberately not `hasDrawableGeometry`, and the distinction is the whole
    /// point. A `.loaded` manifest whose segments are all unresolved, or all
    /// resolved without bounds, has no drawable tile — and that is the normal
    /// first minute of a walk and the permanent state of an anchors-only build.
    /// Deciding "show the spinner" from *tiles* therefore restarts the spinner
    /// on every one of the ~30 revisions a minute a live walk produces, on a
    /// world the phone has already successfully fetched. Deciding it from
    /// *having an answer* does not. `WorldBuilderViewModel.geometryDidChange`
    /// is the only caller.
    var hasAnswered: Bool {
        switch self {
        case .noWorld, .notAddressed, .loading: return false
        case .loaded, .towerReportsNone, .failed: return true
        }
    }
}

/// What went wrong going to get the geometry, in the shape the reader is owed.
///
/// A struct rather than a bare string so the diagnostics surface can branch on
/// `kind` while the normal surface shows `message` and nothing else.
struct WorldGeometryFailure: Equatable {
    enum Kind: Equatable {
        /// The request did not complete: no route, no Tower, a dropped link.
        case unreachable
        /// The manifest came back and this build could not read it as
        /// `world_builder.geometry`. Situation 4 above lives here: one bad row
        /// drops the whole manifest by design, and this is what that looks
        /// like from the outside.
        case undecodable
        /// The manifest is in a pose convention this build does not implement.
        /// Refused rather than drawn, because it would draw plausibly and
        /// wrongly.
        case poseConvention
    }

    var kind: Kind
    /// The transport's or the Tower's own words, when there were any. Never
    /// composed here.
    var detail: String?

    /// One sentence for a person. Names the thing that failed and never blames
    /// the world for a link that dropped.
    var message: String {
        switch kind {
        case .unreachable:
            if let detail, !detail.isEmpty {
                // Keep Tower or transport prose only after the text guard.
                return "The Tower's geometry could not be fetched: \(WorldTowerText.clause(detail))"
            }
            return "The Tower's geometry could not be fetched."
        case .undecodable:
            return "The Tower's geometry could not be read as the contract this build implements. "
                + "One unreadable row refuses the whole manifest, so this is not a claim that the "
                + "world is empty."
        case .poseConvention:
            return "The Tower's geometry is in a pose convention this build does not implement, "
                + "so it is not drawn. Drawing it would look like a room and mean nothing."
        }
    }
}

// MARK: - What the Tower itself claims exists

/// The Tower's own account of what it built, lifted off the snapshot.
///
/// The Tower can report 463 keyframes, 467 poses and 17,674 points even
/// when the phone cannot fetch their fragments. Keep this claim independent
/// of the fetch so the screen states both facts instead of calling the room
/// empty.
struct WorldEvidence: Equatable {
    var keyframes: Int?
    /// `world_snapshot.geometry.element_count` — points, on today's Tower, in
    /// whatever unit `representation` names.
    var elements: Int?
    var representation: String?
    /// `trajectory.pose_count`: poses carrying a position that is evidence.
    var poses: Int?
    var segments: Int?

    init(
        keyframes: Int? = nil,
        elements: Int? = nil,
        representation: String? = nil,
        poses: Int? = nil,
        segments: Int? = nil
    ) {
        self.keyframes = keyframes
        self.elements = elements
        self.representation = representation
        self.poses = poses
        self.segments = segments
    }

    /// `nil` where there is no snapshot, so a caller cannot accidentally treat
    /// "no world" as "a world that reported nothing".
    init?(snapshot: WorldSnapshot?) {
        guard let snapshot else { return nil }
        self.init(
            keyframes: snapshot.keyframeCount,
            elements: snapshot.geometry.elementCount,
            representation: snapshot.geometry.representation,
            poses: snapshot.trajectory.poseCount,
            segments: snapshot.trajectory.segments
        )
    }

    /// Whether the **Tower** says this world holds geometry.
    ///
    /// Either figure on its own is enough. A build can position cameras and
    /// recover few points, or recover points across segments whose cameras were
    /// never placed; both are geometry, and requiring both would
    /// misreport a geometry-bearing world as empty.
    var hasGeometry: Bool { (elements ?? 0) > 0 || (poses ?? 0) > 0 }

    /// The Tower's figures on one line, in its own vocabulary, or `nil` when it
    /// gave none. `representation` is the Tower's word and is quoted, never
    /// parsed — the rule `WorldSummaryView.geometryValue` already follows.
    var summary: String? {
        var parts: [String] = []
        if let elements {
            parts.append("\(elements) \(representation ?? "elements")")
        } else if let representation {
            parts.append(representation)
        }
        if let poses, poses > 0 { parts.append("\(poses) camera poses") }
        if let keyframes { parts.append("\(keyframes) keyframes") }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}

// MARK: - The sentence, decided in one place

/// What to say where the fragment gallery has nothing to draw.
///
/// The gallery receives one account instead of composing its own empty-state
/// prose. "Nothing mapped" comes from a Tower 404 or a loaded manifest with
/// no drawable or unresolved fragments, unless the Tower snapshot claims
/// geometry; tests cover a geometry-bearing snapshot under every fetch state.
struct WorldGeometryAccount: Equatable {
    /// One short line. The gallery's headline when there is nothing to draw.
    var headline: String
    /// The explanation under it, when there is one worth making.
    var detail: String?
    /// Whether this account is the phone admitting ignorance rather than
    /// reporting an answer. Drives nothing but the diagnostics label today;
    /// present so a caller never has to string-match to find out.
    var isPhoneSideUnknown: Bool

    /// The one account that says nothing was mapped.
    ///
    /// Never use this as a default: a missing account is ignorance, not
    /// evidence that the Tower built nothing. Defaults use undescribed.
    static let nothingMapped = WorldGeometryAccount(
        headline: "Nothing mapped yet",
        detail: "The Tower reports no geometry for this session.",
        isPhoneSideUnknown: false
    )

    /// The account for "nobody has said". The default everywhere one is needed.
    ///
    /// It asserts nothing about the world and nothing about the fetch, which is
    /// the correct thing for a value that exists because a caller supplied no
    /// information. Marked as the phone's own ignorance, because that is what
    /// it is.
    static let undescribed = WorldGeometryAccount(
        headline: "Geometry not described",
        detail: nil,
        isPhoneSideUnknown: true
    )

    /// Derives the gallery account from fetch status and Tower evidence.
    ///
    /// Check the Tower's independent geometry claim before describing a
    /// missing or failed phone fetch. A failed fetch cannot prove an empty
    /// world; a Tower 404 or a loaded status without drawable or unresolved
    /// fragments can support the nothing-mapped account when the snapshot
    /// does not claim geometry.
    static func account(
        for status: WorldGeometryStatus, evidence: WorldEvidence?
    ) -> WorldGeometryAccount {
        switch status {
        case .noWorld:
            return WorldGeometryAccount(
                headline: "No world",
                detail: "There is no world on screen for geometry to belong to.",
                isPhoneSideUnknown: false
            )

        case .notAddressed:
            // Situations 1 and 2. If the Tower's own snapshot says geometry
            // exists, this is the phone lacking an address for geometry that
            // is genuinely there and must not read as an empty room.
            if let evidence, evidence.hasGeometry {
                return WorldGeometryAccount(
                    headline: "Built, not yet fetched",
                    detail: "The Tower reports \(evidence.summary ?? "geometry") for this world "
                        + "but has not published an address to fetch the points from, so none are "
                        + "drawn here. The 3D reconstruction is served by a different route and "
                        + "may still be available.",
                    isPhoneSideUnknown: true
                )
            }
            return WorldGeometryAccount(
                headline: "Nothing built yet",
                detail: "The Tower has not published geometry for this session. "
                    + "Reconstruction runs in a separate Tower process, which this app "
                    + "can neither start nor see.",
                isPhoneSideUnknown: true
            )

        case .loading:
            return WorldGeometryAccount(
                headline: "Fetching the geometry…",
                detail: nil,
                isPhoneSideUnknown: true
            )

        case .failed(let failure):
            // Situations 3, 4 and 5. The reason is the Tower's or the
            // transport's; the claim about the world is the Tower's evidence,
            // if it made one.
            var detail = failure.message
            if let evidence, evidence.hasGeometry, let summary = evidence.summary {
                detail += " The Tower reports \(summary) for this world, so this is a fetch "
                    + "that failed and not a world that is empty."
            }
            return WorldGeometryAccount(
                headline: "Geometry could not be fetched",
                detail: detail,
                isPhoneSideUnknown: true
            )

        case .towerReportsNone(let towerDetail):
            // The Tower's own 404. It is entitled to say this — but if its own
            // snapshot said otherwise, the two disagree and the reader is told
            // that rather than being handed the more comfortable half.
            if let evidence, evidence.hasGeometry, let summary = evidence.summary {
                return WorldGeometryAccount(
                    headline: "The Tower disagrees with itself",
                    detail: "Its status report says \(summary) for this world; its geometry route "
                        + "answers that it has none"
                        + (towerDetail.map { " (\($0))" } ?? "")
                        + ". Nothing is drawn here, because which of the two is right is the "
                        + "Tower's question and not this app's.",
                    isPhoneSideUnknown: false
                )
            }
            // The Tower's own words when it sent any, and the standing sentence
            // otherwise. The **headline** is taken from `nothingMapped` rather
            // than retyped, so those four words genuinely exist in one place
            // and a search for them finds every way they can be drawn.
            return WorldGeometryAccount(
                headline: Self.nothingMapped.headline,
                detail: towerDetail.map { "The Tower answered: \($0)." }
                    ?? Self.nothingMapped.detail,
                isPhoneSideUnknown: false
            )

        case .loaded(let model, let unfetched):
            if !model.fragments.isEmpty || model.unresolvedCount > 0 {
                // There is something to draw, or something counted. The gallery
                // draws its own headline; this account is only consulted for
                // the empty case, and returning the gallery's own headline here
                // keeps the two from ever disagreeing.
                //
                // The `??` covers a manifest whose only rows are unresolved:
                // nothing is drawable, so `headline` is `nil` by design, and
                // "seen and not reconstructed" is what that manifest says.
                return WorldGeometryAccount(
                    headline: model.headline
                        ?? "\(model.unresolvedCount) area\(model.unresolvedCount == 1 ? "" : "s") "
                            + "seen, none reconstructed",
                    detail: unfetched > 0
                        ? "\(unfetched) segment\(unfetched == 1 ? "" : "s") the Tower named could "
                            + "not be fetched and are not drawn."
                        : nil,
                    isPhoneSideUnknown: false
                )
            }
            // An empty manifest. The Tower has a manifest for this session and
            // nothing in it resolved — but if its snapshot claimed geometry,
            // that is again a disagreement and not an empty room.
            if let evidence, evidence.hasGeometry, let summary = evidence.summary {
                return WorldGeometryAccount(
                    headline: "Nothing drawable in this session's geometry",
                    detail: "The Tower's manifest for this session names no segment that "
                        + "resolved, while its status report says \(summary). Nothing is drawn "
                        + "rather than a guess at which is right.",
                    isPhoneSideUnknown: false
                )
            }
            return .nothingMapped
        }
    }
}

// MARK: - The normal surface's vocabulary

/// What a world is doing, in words a person uses.
///
/// The brief for this vocabulary is that the ordinary screen must not read as a
/// reconstruction debugger. "2 registered, 120 refused", "coverage partial",
/// "N segments in the frame of segment 120" and the rest are all still
/// reachable — they moved behind Diagnostics, they were not deleted — and what
/// stands in their place on the normal surface is one of these words.
///
/// Every case is earned from something the Tower said. None is a guess about
/// the other machine, which is the rule the whole workspace is built on.
enum WorldStage: Equatable, CaseIterable {
    /// Frames are reaching the Tower and it has reported no geometry yet.
    case mapping
    /// The Tower is receiving and has built geometry.
    case building
    /// The session has stopped and a live process is working on a world that
    /// already has geometry. `lifecycle.build_in_progress == true` is the
    /// evidence; see `WorldModelState.finalizing`.
    case improving
    /// The session has stopped and the world is being finished. Reached with
    /// no geometry yet, or on a Tower that cannot say whether a build runs.
    case finalizing
    /// Finished, with the final solve done or with no record either way.
    case saved
    /// Finished or interrupted, holding real geometry, with the final solve
    /// absent, skipped or failed. The world is usable and is not complete.
    case partial
    /// The session ended abnormally and geometry survived.
    case interrupted
    /// Nothing usable came of it.
    case needsRetry

    /// The word on screen. Sentence case, no jargon, no acronyms.
    var label: String {
        switch self {
        case .mapping: return "Mapping"
        case .building: return "Building"
        case .improving: return "Improving"
        case .finalizing: return "Finalizing"
        case .saved: return "Saved"
        case .partial: return "Partial"
        case .interrupted: return "Interrupted"
        case .needsRetry: return "Needs retry"
        }
    }

    var systemImage: String {
        switch self {
        case .mapping: return "dot.radiowaves.left.and.right"
        case .building, .improving: return "cube"
        case .finalizing: return "cube.transparent"
        case .saved: return "cube.fill"
        case .partial: return "cube.transparent.fill"
        case .interrupted: return "exclamationmark.triangle"
        case .needsRetry: return "exclamationmark.triangle.fill"
        }
    }

    /// Whether the Tower may still change this world. `false` means nothing
    /// more is coming, which is what makes "Partial" a settled description
    /// rather than a stage on the way somewhere.
    var isStillChanging: Bool {
        switch self {
        case .mapping, .building, .improving, .finalizing: return true
        case .saved, .partial, .interrupted, .needsRetry: return false
        }
    }

    /// The stage for a state, the Tower's evidence and its finalization record.
    ///
    /// Switches on lifecycle, then uses geometry and final-solve evidence;
    /// photographic standing refines a finalizing room. A missing report is
    /// never taken as proof that a build is finished.
    ///
    /// `nil` for the states with no world in them — `.idle`,
    /// `.awaitingFirstUpdate`, `.unsupported`. Those have their own careful
    /// wording on the canvas and are deliberately not folded into this
    /// vocabulary: "Mapping" over an idle Tower would claim a build that
    /// nothing on the phone can start or see.
    static func stage(
        for state: WorldModelState,
        evidence: WorldEvidence?,
        finalization: WorldFinalizationReport?,
        photographic: WorldPhotographicReport? = nil
    ) -> WorldStage? {
        switch state {
        case .unsupported, .idle, .awaitingFirstUpdate:
            return nil
        case .receiving:
            return evidence?.hasGeometry == true ? .building : .mapping
        case .finalizing(let snapshot, let buildInProgress):
            // Improving requires geometry plus a live build or an unfinished
            // photographic room (WORLD-BUILDER-IOS.md §3a). An owed room has
            // no live builder, so the canvas explains the outstanding work
            // without drawing a spinner or denying a finished final pass.
            guard evidence?.hasGeometry == true else { return .finalizing }
            // An area may still build while the room on screen is settled
            // (WORLD-BUILDER-COMPONENTS.md §3.4).
            // Judge that room by the finalized stage; the canvas adds the
            // area's own sentence and does not show a room spinner.
            if photographic?.standing.isAreaStillFinishing == true {
                return stage(for: .finalized(snapshot), evidence: evidence, finalization: finalization)
            }
            if buildInProgress == true { return .improving }
            if photographic?.standing.isUnfinished == true { return .improving }
            return .finalizing
        case .finalized:
            let solve = WorldFinalSolve(word: finalization?.finalSolve)
            if solve.deniesAFinishedWorld {
                return evidence?.hasGeometry == true ? .partial : .needsRetry
            }
            if solve == .pending { return .finalizing }
            // `.solved`, `.notReported` and any word this build has not heard
            // of. A record that says nothing about the final solve is an older
            // record, and the world it describes was stored as finished.
            //
            // `solve == .solved` stands in for the figures only when there
            // ARE no figures — an older record that carries none. A record
            // that carries zeros is not silent: the Tower counted and found
            // nothing, and a solve reported over an empty build (unreachable
            // by every path read, and left on record rather than trusted)
            // must not turn that into "Saved".
            let figuresAbsent = evidence?.elements == nil && evidence?.poses == nil
            if evidence?.hasGeometry == true { return .saved }
            return solve == .solved && figuresAbsent ? .saved : .needsRetry
        case .interrupted:
            return evidence?.hasGeometry == true ? .interrupted : .needsRetry
        case .failed(let failure):
            // Only the Tower's own report of a failed build says anything
            // about the walk. A channel or decoding failure is about this
            // screen's subscription, and is not a stage.
            return failure.kind == .towerReportedFailure ? .needsRetry : nil
        }
    }
}

// MARK: - The final solve

/// The Tower's final-solve word, kept distinct from the world stage.
///
/// A missing word can mean an older record or a final pass not yet reported;
/// it cannot prove success or failure. Unrecognized words survive as in
/// WorldSelectionMode so the Tower's vocabulary is not silently changed.
/// The listing presentation also
/// reads this value, so it remains nonisolated.
nonisolated enum WorldFinalSolve: Equatable, Sendable {
    /// No finalization record reached this screen.
    ///
    /// **Silence, and not one kind of silence.** A `nil` here means any of: no
    /// pinned status report has arrived yet, this Tower predates the field, the
    /// world never reached finalization, or the record genuinely kept no
    /// account of it. Nothing on the phone can tell those apart, so this case
    /// says **nothing at all** — see `sentence`.
    case notReported
    case pending
    case solved
    case skipped
    case failed
    case unavailable
    case other(String)

    init(word: String?) {
        switch word {
        case nil: self = .notReported
        case "pending": self = .pending
        case "solved": self = .solved
        case "skipped": self = .skipped
        case "failed": self = .failed
        case "unavailable": self = .unavailable
        case let other?: self = .other(other)
        }
    }

    /// Whether this word says the world was **not** finished properly.
    /// `.notReported` is excluded on purpose: a record that kept no account of
    /// the final solve has not said it failed.
    var deniesAFinishedWorld: Bool {
        switch self {
        case .skipped, .failed, .unavailable: return true
        case .notReported, .pending, .solved, .other: return false
        }
    }

    /// One line for the reader, or `nil` where there is nothing worth saying.
    ///
    /// `.solved` returns `nil`: a world that finished properly does not need a
    /// sentence explaining that it finished properly, and the stage word
    /// ("Saved") already carries it.
    var sentence: String? {
        switch self {
        case .notReported:
            // Nothing. This returned "This world has no record of a final pass,
            // so it may hold less than a finished one would." — a sentence the
            // phone composed out of a `nil` it could not interpret, which is
            // the exact defect class this whole pass exists to remove: the
            // phone reporting on the Tower from silence.
            //
            // It also contradicted the screen it was drawn on. `WorldStage`
            // maps a `.finalized` state with no record to `.saved`, so the card
            // said the world was final while the line under it implied it was
            // not. A word the Tower did not say cannot resolve that; only the
            // Tower saying one can.
            return nil
        case .pending:
            return "The final pass has not run yet."
        case .solved:
            return nil
        case .skipped:
            return "The final pass was skipped, so this world is what the walk produced "
                + "without it."
        case .failed:
            return "The final pass failed, so this world is what the walk produced without it."
        case .unavailable:
            return "The final pass could not run on this Tower, so this world is what the "
                + "walk produced without it."
        case .other(let word):
            // The Tower's word, quoted. A word this build has not heard of is
            // shown rather than folded into the nearest one it knows.
            return "The Tower reports its final pass as \"\(word)\"."
        }
    }
}

/// What can and cannot be done about a world that did not finish.
///
/// **There is no rebuild route.** `tower/tower/routes/geometry.py` serves
/// `GET /worlds`, `GET /worlds/{id}/geometry/manifest`,
/// `GET /worlds/{id}/geometry/segment/{i}` and `GET /worlds/{id}/render`, and
/// nothing else; every one is a read. Nothing on the phone can ask the Tower to
/// finish, re-solve or rebuild a world. So this type says the state truthfully
/// and offers no button, which is the rule the workspace already applies to
/// "Start Mapping".
struct WorldRecoverability: Equatable {
    /// Whether real geometry survived whatever happened.
    var geometrySurvived: Bool
    /// `nil` for the stages where there is nothing to recover from — a world
    /// still being built has not lost anything yet.
    var sentence: String?

    static func of(
        stage: WorldStage, evidence: WorldEvidence?, reason: String?
    ) -> WorldRecoverability {
        let survived = evidence?.hasGeometry == true
        switch stage {
        case .interrupted, .partial:
            if survived {
                return WorldRecoverability(
                    geometrySurvived: true,
                    sentence: "What was built before this stopped is stored and can be opened. "
                        + "Finishing it is the Tower's to do; this app has no way to ask for it."
                )
            }
            return WorldRecoverability(
                geometrySurvived: false,
                sentence: "Nothing usable survived this session"
                    + (reason.map { ": \($0)" } ?? ".")
            )
        case .needsRetry:
            return WorldRecoverability(
                geometrySurvived: false,
                sentence: "Nothing usable came of this session"
                    + (reason.map { ": \($0)" } ?? ".")
                    + " Walking the space again is what produces another one; this app cannot "
                    + "ask the Tower to rebuild this one."
            )
        case .mapping, .building, .improving, .finalizing, .saved:
            // Nothing has been lost, so there is nothing to say about
            // recovering it. `reason` belongs to the interrupted cases above.
            return WorldRecoverability(geometrySurvived: survived, sentence: nil)
        }
    }
}

// MARK: - Everything the canvas needs, in one value

/// The derived half of the World Builder screen, bundled.
///
/// One parameter rather than six on `WorldCanvasView`, and — more importantly —
/// one **value** rather than six computations spread across a view body. The
/// canvas draws this and `WorldModelState` and nothing else, so what is on
/// screen is a pure function of two values a test can construct, and the whole
/// state machine is assertable without SwiftUI.
///
/// `WorldBuilderViewModel.presentation` is the only place that builds one from
/// live data.
struct WorldPresentation: Equatable {
    /// The normal surface's word for what this world is doing. `nil` where
    /// there is no world.
    var stage: WorldStage?
    /// What the Tower says it built. `nil` where there is no snapshot.
    var evidence: WorldEvidence?
    var finalSolve: WorldFinalSolve
    /// The 3D ladder, resolved.
    var reconstruction: WorldReconstruction
    /// What the sparse gallery says when it has nothing to draw.
    var account: WorldGeometryAccount
    /// What survived, and what can be done about it. `nil` where there is no
    /// world.
    var recoverability: WorldRecoverability?
    /// The Tower's `lifecycle.photographic`, or `nil` when it sent none.
    var photographic: WorldPhotographicReport?
    /// The live relocalizer's episode, for the walk this phone is streaming
    /// (`nil` otherwise: the client only publishes it while following live and
    /// bound).
    var recovery: WorldRecoveryReport?
    /// The look-back banner, under the same rule as `recovery`.
    var lookBackBanner: WorldLookBackBanner?
    /// The Tower's processing report, or `nil` when it sent none.
    var processing: WorldProcessingReport?
    /// The stage line under a finalizing world (IOS §3c), built by the view
    /// model from `state`; `nil` otherwise.
    var finishLine: WorldFinishLine?
    /// The followed walk's stop clock and away banner.
    var finishClock: WorldFinishClock

    init(
        stage: WorldStage? = nil,
        evidence: WorldEvidence? = nil,
        finalSolve: WorldFinalSolve = .notReported,
        reconstruction: WorldReconstruction = .unavailable(
            reason: "The Tower has named no world to look at."
        ),
        account: WorldGeometryAccount = .undescribed,
        recoverability: WorldRecoverability? = nil,
        photographic: WorldPhotographicReport? = nil,
        recovery: WorldRecoveryReport? = nil,
        lookBackBanner: WorldLookBackBanner? = nil,
        processing: WorldProcessingReport? = nil,
        finishLine: WorldFinishLine? = nil,
        finishClock: WorldFinishClock = .unknown
    ) {
        self.stage = stage
        self.evidence = evidence
        self.finalSolve = finalSolve
        self.reconstruction = reconstruction
        self.account = account
        self.recoverability = recoverability
        self.photographic = photographic
        self.recovery = recovery
        self.lookBackBanner = lookBackBanner
        self.processing = processing
        self.finishLine = finishLine
        self.finishClock = finishClock
    }

    /// The picture this world can open is a PREVIEW (WORLDS §4a rule 8): the
    /// world is not settled. Never presented as the final world.
    var isPreview: Bool { stage?.isStillChanging == true && reconstruction.target != nil }

    /// What the open control says: *Open the preview* for a preview.
    var openTitle: String { isPreview ? WorldPreviewCopy.openPreview : reconstruction.actionTitle }

    /// When the followed walk stopped, only while the world is unsettled.
    var stoppedAt: ContinuousClock.Instant? { stage?.isStillChanging == true ? finishClock.stoppedAt : nil }

    /// Whether this is a saved world whose photographic build failed: the one
    /// case `WORLD-BUILDER-IOS.md` §3a says is worth new copy. Only over
    /// `.saved` -- a failed photographic build needs a solved final pass, so
    /// every other stage already says something truer than "Saved".
    var isSavedWithoutItsPhotographicVersion: Bool {
        stage == .saved && photographic?.standing.isFailed == true
    }

    /// The headline word: the stage's, except over a saved world whose
    /// photographic build failed, where plain "Saved" hides the failure.
    var headline: String? {
        if isSavedWithoutItsPhotographicVersion { return WorldPhotographicCopy.failedHeadline }
        return stage?.label
    }

    /// A finished room with one of the walk's areas still being made
    /// (`scope: "area"`): the room is saved; say so, and that an area is not.
    var isSavedWithAnAreaStillFinishing: Bool {
        photographic?.standing.isAreaStillFinishing == true
            && (stage == .saved || stage == .partial)
    }

    /// Whether the canvas may draw the live-build spinner: a live process
    /// (`build_in_progress: true`) working on THIS picture. Not while only an
    /// area is building -- the room on screen will not change.
    func showsLiveBuild(buildInProgress: Bool?) -> Bool {
        buildInProgress == true && photographic?.standing.isAreaStillFinishing != true
    }

    /// What the 3D viewer says above its caption when it is opened from this
    /// screen: the ladder's note for an unfinished world, the failed
    /// photographic build's explanation for a saved one, or that an area of
    /// the walk is still being finished. `nil` otherwise.
    var viewerNote: String? {
        if case .partial(_, let note) = reconstruction { return note }
        if isSavedWithoutItsPhotographicVersion { return WorldPhotographicCopy.failedHeadline + "." }
        if isSavedWithAnAreaStillFinishing { return WorldPhotographicCopy.areaStillFinishing() }
        return nil
    }

    /// The value for a screen with no world on it. What previews and the
    /// states that carry no snapshot are drawn from.
    static let empty = WorldPresentation()

    /// The canvas's sentence under a `finalizing` world, from what the Tower
    /// actually reported. A pure function so every arm is tested.
    ///
    /// - `build_in_progress: true`: a live process is finishing it, and the
    ///   figures are not final. (The spinner is drawn for this case alone.)
    /// - otherwise the photographic word, when it names unfinished work:
    ///   `owed`, `unobservable`, or `running` reported without a live
    ///   boolean beside it.
    /// - `false` with no such word: the Tower reported that nothing is
    ///   building it.
    /// - `nil` with no such word -- a record older than the photographic
    ///   stages, which is the only case where "the Tower does not report
    ///   whether a build is running" is true.
    static func finalizingDetail(
        buildInProgress: Bool?, photographic: WorldPhotographicReport?
    ) -> String {
        // An area build can make buildInProgress true while the room is
        // finished, so do not call the room figures unfinished.
        if photographic?.standing.isAreaStillFinishing == true {
            return WorldPhotographicCopy.areaStillFinishing()
        }
        if buildInProgress == true {
            return "The Tower is still finishing this world: the final solve, the final build "
                + "and the photographic reconstruction all run after the session stops, and "
                + "these figures are not final."
        }
        switch photographic?.standing {
        case .owed: return WorldPhotographicCopy.owedSentence
        case .unobservable: return WorldPhotographicCopy.unobservableSentence
        case .building: return WorldPhotographicCopy.buildingSentence
        case .failed, .settled, .areaStillFinishing, nil: break
        }
        if buildInProgress == false {
            return "Capture has ended and these figures are not final. The Tower reports that "
                + "nothing is building this world right now."
        }
        return "Capture has ended and these figures are not final. The Tower does not report "
            + "whether a build is running, so this app cannot say whether one is."
    }
}

// MARK: - The 3D ladder

/// Which 3D reconstruction a saved world can actually show, best first.
///
/// The product rule this implements: **final 3D world > best available partial
/// 3D world > truthful degraded state.** The sparse fragment gallery is not on
/// this ladder at all — it is a diagnostic, and it lives behind Diagnostics.
///
/// The one surface in this app that can draw 3D is the `WKWebView` in
/// `WorldRenderViewer`, which shows the Tower's own composed page from
/// `GET /worlds/{id}/render`. There is no SceneKit, RealityKit or Metal in this
/// target, and there is no plan to add one for this: the Tower composes the
/// world, colours it and decimates it to a mobile budget, and duplicating that
/// on the phone would be a second renderer able to disagree with the first.
enum WorldReconstruction: Equatable {
    /// The Tower named geometry for a world nothing will add to, and its final
    /// pass ran. This is the world as it will always be.
    case final(WorldRenderTarget)
    /// The Tower named geometry for a world that is still changing, or whose
    /// final pass did not run. Real, and not the last word.
    case partial(WorldRenderTarget, note: String)
    /// No address for a 3D view. The reason is a sentence, because "the button
    /// is disabled" is not one.
    case unavailable(reason: String)

    /// The address to fetch, when there is one.
    var target: WorldRenderTarget? {
        switch self {
        case .final(let target): return target
        case .partial(let target, _): return target
        case .unavailable: return nil
        }
    }

    /// What the primary control says. A verb the app can actually deliver.
    var actionTitle: String { "Open the 3D world" }

    /// The ladder, resolved.
    ///
    /// `target` is `WorldBuilderViewModel.renderTarget`: the world and session
    /// the Tower named geometry for, or the world a person pinned. `nil` there
    /// is the honest "there is nothing to open", and it is a **different** state
    /// from the render route answering 404 — which happens after this, inside
    /// the viewer, and carries the Tower's own detail.
    /// The note for a world whose photographic room is outstanding and
    /// NOTHING is known to be building it: `owed` or `unobservable`. `nil`
    /// for every other word, and for no word at all, so the long-standing
    /// notes stand wherever the Tower did not say something more specific.
    ///
    /// `running` is deliberately not here: something IS building the world,
    /// and the Improving note's promise is true of it.
    static func unfinishedPhotographicNote(_ photographic: WorldPhotographicReport?) -> String? {
        switch photographic?.standing {
        case .owed:
            return WorldPhotographicCopy.owedSentence + " "
                + WorldPhotographicCopy.finishedLooksDifferent
        case .unobservable:
            return WorldPhotographicCopy.unobservableSentence + " "
                + WorldPhotographicCopy.finishedLooksDifferent
        case .building, .failed, .settled, .areaStillFinishing, nil:
            return nil
        }
    }

    private struct AddressedLadderInput {
        let target: WorldRenderTarget
        let stage: WorldStage?
        let photographic: WorldPhotographicReport?
        let unfinishedNote: String?
    }

    // Each row returns nil when its condition does not apply; the first
    // returned outcome wins. The order is the old ladder's branch order.
    private typealias AddressedLadderRule = (AddressedLadderInput) -> WorldReconstruction?
    private static let addressedLadderRules: [AddressedLadderRule] = [
        // A pinned address can arrive before the first status report.
        { input in
            guard input.stage == nil else { return nil }
            return .partial(input.target, note: "The Tower has not yet said what state this world is in.")
        },
        // Saved answers whether more will be built. The separate final-solve
        // sentence answers whether the record mentions a final pass.
        { input in
            guard input.stage == .saved else { return nil }
            return .final(input.target)
        },
        { input in
            guard input.stage == .mapping || input.stage == .building else { return nil }
            return .partial(input.target, note: WorldPreviewCopy.labelled(
                "This world is still being built, so it will change."))
        },
        // Owed or unobservable photographic work cannot promise that waiting
        // will finish the room. The final solve changed 16 disconnected
        // components to 6 and joined 652 of 795 keyframes. Opening the
        // earlier picture can mislead for about 170 seconds; the solve took
        // 179 s after Stop on a four-minute walk, with byte-identical
        // progress from second 3 to 179;
        // later photographic work took 459 s for 385 keyframes and 756 s
        // for 690 keyframes. A note explains the preview without claiming
        // a duration or disabling the viewer.
        { input in
            guard input.stage == .improving, let note = input.unfinishedNote else { return nil }
            return .partial(input.target, note: WorldPreviewCopy.labelled(note))
        },
        { input in
            guard input.stage == .improving else { return nil }
            return .partial(input.target, note: WorldPreviewCopy.building)
        },
        // Finalizing does not promise a live build. Photographic work may
        // remain after the writer lock is released, including a 90-second
        // post-Stop gap before a build still six minutes away. Name the
        // photographic state before making any claim about the final pass.
        { input in
            guard input.stage == .finalizing, let note = input.unfinishedNote else { return nil }
            return .partial(input.target, note: WorldPreviewCopy.labelled(note))
        },
        { input in
            guard input.stage == .finalizing, input.photographic?.standing == .building else { return nil }
            return .partial(input.target, note: WorldPreviewCopy.labelled(
                WorldPhotographicCopy.buildingSentence + " "
                    + WorldPhotographicCopy.finishedLooksDifferent))
        },
        // The canvas reports whether a process is running beside its spinner;
        // this note only names what is missing, including a live build with
        // no drawable geometry. It must not deny the spinner.
        { input in
            guard input.stage == .finalizing else { return nil }
            return .partial(input.target, note: WorldPreviewCopy.labelled(
                "The final pass has not landed for this world, and the "
                    + "finished version is very different from this one."))
        },
        // The canvas draws the final-solve sentence separately. Repeating it
        // here would print the same explanation twice.
        { input in
            guard input.stage == .partial else { return nil }
            return .partial(input.target, note: "This world did not finish, so it is what the walk "
                + "produced without a final pass.")
        },
        { input in
            guard input.stage == .interrupted else { return nil }
            return .partial(input.target, note: "This world was interrupted. What was built before it stopped is here.")
        },
        // A target is a real offer even if its route may answer 404 later.
        { input in
            guard input.stage == .needsRetry else { return nil }
            return .partial(input.target, note: "Little or nothing was reconstructed for this session.")
        },
    ]

    private struct UnavailableLadderInput {
        let stage: WorldStage?
        let evidence: WorldEvidence?
    }

    private struct UnavailableLadderRule {
        let matches: (UnavailableLadderInput) -> Bool
        let reason: (UnavailableLadderInput) -> String
    }

    private static let unavailableLadderRules: [UnavailableLadderRule] = [
        // The Tower's geometry claim outranks the phone's missing address.
        .init(matches: { $0.evidence?.hasGeometry == true }, reason: {
            "The Tower reports \($0.evidence?.summary ?? "geometry") for this world "
                + "but has not named where to fetch a picture of it."
        }),
        .init(matches: { $0.stage?.isStillChanging == true }, reason: { _ in
            "Nothing has been built to look at yet. A 3D world appears once the "
                + "Tower's first build produces one."
        }),
        .init(matches: { _ in true }, reason: { _ in
            "The Tower has built no 3D world for this session."
        }),
    ]

    /// Whether the world is still being worked on. The switch is deliberately
    /// exhaustive (no `default`): a new `WorldStage` must fail the build here
    /// until someone decides which ladder rules cover it.
    private static func stageIsUnfinished(_ stage: WorldStage?) -> Bool {
        switch stage {
        case .improving, .finalizing: true
        case .mapping, .building, .saved, .partial, .interrupted, .needsRetry, .none: false
        }
    }

    /// Resolves the screen's 3D offer from the Tower's address, stage and
    /// evidence. The ordered rules select the first truthful outcome; a
    /// missing address consults the Tower's geometry claim before its stage.
    /// The final-solve sentence is drawn separately, so it does not alter this
    /// choice or duplicate the canvas's explanation.
    static func ladder(
        target: WorldRenderTarget?,
        stage: WorldStage?,
        finalSolve: WorldFinalSolve,
        evidence: WorldEvidence?,
        photographic: WorldPhotographicReport? = nil
    ) -> WorldReconstruction {
        guard let target else {
            let input = UnavailableLadderInput(stage: stage, evidence: evidence)
            for rule in unavailableLadderRules where rule.matches(input) {
                return .unavailable(reason: rule.reason(input))
            }
            preconditionFailure("The unavailable ladder must have a fallback rule.")
        }

        let input = AddressedLadderInput(
            target: target, stage: stage, photographic: photographic,
            unfinishedNote: Self.stageIsUnfinished(stage)
                ? Self.unfinishedPhotographicNote(photographic) : nil)
        for rule in addressedLadderRules {
            if let outcome = rule(input) { return outcome }
        }
        preconditionFailure("Every world stage must have an addressed ladder rule.")
    }
}
