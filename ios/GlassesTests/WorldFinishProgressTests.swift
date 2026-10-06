//
//  WorldFinishProgressTests.swift
//  GlassesTests
//
//  U0.6 + T-UX1 (WORLD-BUILDER-IOS.md §3c; WORLDS §2b, §4a rule 8): the
//  stage line, the preview label, the elapsed time, the stop clock and the
//  banner, the picture basis, as pure functions. I1-I6 and I8-I10 of the
//  spec (I7, the client end to end, is in WorldBuilderIntegrationTests).
//

import XCTest
@testable import Glasses

@MainActor
final class WorldFinishProgressTests: XCTestCase {

    // MARK: I1: `lifecycle.processing`

    func testTheProcessingBlockDecodesKnownUnknownAndBadSteps() {
        for word in ["waiting", "preparing", "matching", "placing", "checking", "assembling"] {
            let report = WorldProcessingReport(json: ["stage": word])
            XCTAssertEqual(report?.stage.rawValue, word)
            XCTAssertEqual(report?.stage.isKnown, true, word)
            XCTAssertNil(report?.step, word)
        }
        let unknown = WorldProcessingReport(json: ["stage": "replaying"])
        XCTAssertEqual(unknown?.stage.rawValue, "replaying", "an unknown word survives as itself")
        XCTAssertEqual(unknown?.stage.isKnown, false, "and is not one of the six")

        func step(_ value: Any) -> WorldProcessingStep? {
            WorldProcessingReport(json: ["stage": "checking", "step": value] as [String: Any])?.step
        }
        XCTAssertEqual(step(["n": 2, "of": 3]), WorldProcessingStep(n: 2, of: 3))
        XCTAssertEqual(step(["n": 7, "of": 7]), WorldProcessingStep(n: 7, of: 7))
        XCTAssertNil(WorldProcessingReport(json: ["stage": "checking"])?.step, "missing")
        XCTAssertNil(step(["n": 4, "of": 3]), "n > of")
        XCTAssertNil(step(["n": 0, "of": 3]), "n < 1")
        XCTAssertNil(step(["n": 1, "of": 8]), "of > 7")
        XCTAssertNil(step(["n": true, "of": 3] as [String: Any]), "n is a Bool")
        XCTAssertNil(step(["n": 1.5, "of": 3]), "not an integer")
        XCTAssertNil(step("2 of 3"), "not an object")
        XCTAssertEqual(WorldProcessingReport(json: ["stage": "checking", "step": "x"])?.stage, .checking,
                       "a bad step is dropped, the stage kept")
        XCTAssertNil(WorldProcessingReport(json: ["stage": "placing", "step": ["n": 1, "of": 3]] as [String: Any])?.step,
                     "a step only on checking")

        XCTAssertNil(WorldProcessingReport(json: nil), "absent")
        XCTAssertNil(WorldProcessingReport(json: NSNull()), "null")
        XCTAssertNil(WorldProcessingReport(json: "placing"), "not an object")
        XCTAssertNil(WorldProcessingReport(json: ["stage": ""]), "an empty stage")
        XCTAssertNil(WorldProcessingReport(json: ["stage": 3]), "not a string")
        XCTAssertNil(WorldProcessingReport(json: ["step": ["n": 1, "of": 3]]), "no stage")
    }

    // MARK: I2: an installed phone and a Tower with both switches on

    /// A status payload WITH `lifecycle.processing` (an unknown word
    /// included) decodes everything else exactly as the same payload without
    /// it; so do a listing row and a revision body carrying the new keys.
    func testTheNewKeysChangeNothingElseThatIsDecoded() throws {
        func payload(processing: Any?) -> [String: Any] {
            var lifecycle: [String: Any] = [
                "build_in_progress": true,
                "finalization": ["state": "pending", "final_solve": "pending", "started_at": 1.0],
                "photographic": ["state": "running", "stage": "surface"],
            ]
            if let processing { lifecycle["processing"] = processing }
            return [
                "model_state": "finalizing", "model_state_reason": NSNull(),
                "session": ["session_id": "s1"],
                "selection": ["mode": "finalizing", "world_id": "w1", "session_id": "s1"],
                "world_snapshot": ["name": "Room", "world_id": "w1", "keyframe_count": 40, "revision": "r1",
                                   "geometry": ["representation": "sparse point cloud", "element_count": 900],
                                   "trajectory": ["pose_count": 40]],
                "geometry": ["revision": "g1"],
                "lifecycle": lifecycle,
            ]
        }
        let plain = payload(processing: nil)
        for extra in [["stage": "checking", "step": ["n": 2, "of": 3]], ["stage": "replaying"]] as [[String: Any]] {
            let with = payload(processing: extra)
            XCTAssertEqual(WorldBuilderResultDecoder.modelState(from: with), WorldBuilderResultDecoder.modelState(from: plain))
            XCTAssertEqual(WorldBuilderResultDecoder.finalization(from: with), WorldBuilderResultDecoder.finalization(from: plain))
            XCTAssertEqual(WorldBuilderResultDecoder.photographic(from: with), WorldBuilderResultDecoder.photographic(from: plain))
            XCTAssertEqual(WorldBuilderResultDecoder.geometryCoordinates(from: with),
                           WorldBuilderResultDecoder.geometryCoordinates(from: plain))
            XCTAssertEqual(WorldBuilderResultDecoder.selection(from: with), WorldBuilderResultDecoder.selection(from: plain))
            XCTAssertNotNil(WorldBuilderResultDecoder.processing(from: with))
        }
        XCTAssertNil(WorldBuilderResultDecoder.processing(from: plain))

        let row: [String: Any] = ["session_id": "s1", "started_at": 1.0, "frame_source": "glasses",
                                  "has_geometry": true, "state": "finalizing",
                                  "finalization": ["state": "pending", "final_solve": "pending"]]
        var rowWith = row
        rowWith["processing"] = ["stage": "placing"]
        XCTAssertEqual(WorldListingSession(json: rowWith), WorldListingSession(json: row))
        XCTAssertNotNil(WorldListingSession(json: row))

        let body = #"{"session_id":"s1","representation":"appearance","revision":"s1/appearance:1@e1","live":true,"appearance":{"revision":"s1/appearance:b1","current":true,"state":"served","epoch":"e1"},"components":null"#
        let bare = try WorldRenderClient.decodeRevision(Data((body + "}").utf8))
        for basis in [#","basis":"walk"}"#, #","basis":7}"#] {
            let decoded = try WorldRenderClient.decodeRevision(Data((body + basis).utf8))
            XCTAssertEqual(decoded.revision, bare.revision)
            XCTAssertEqual(decoded.representation, bare.representation)
            XCTAssertEqual(decoded.live, bare.live)
            XCTAssertEqual(decoded.appearance, bare.appearance)
            XCTAssertEqual(decoded.appearanceState, bare.appearanceState)
            XCTAssertEqual(decoded.components, bare.components)
        }
    }

    /// Lead override (2026-10-05): `world_snapshot.revision` is the
    /// envelope's and moves with every stage and pass. The geometry address
    /// -- what the gallery fetch is keyed on -- must not move with it, or
    /// each stage would refetch the geometry.
    func testAStageChangeMovesNoGeometryAddress() {
        func payload(snapshotRevision: String, stage: String) -> [String: Any] {
            ["model_state": "finalizing",
             "session": ["session_id": "s1"],
             "world_snapshot": ["world_id": "w1", "revision": snapshotRevision, "keyframe_count": 40],
             "geometry": ["revision": "g7"],
             "lifecycle": ["build_in_progress": true, "processing": ["stage": stage]]]
        }
        let placing = WorldBuilderResultDecoder.geometryCoordinates(from: payload(snapshotRevision: "r1", stage: "placing"))
        let checking = WorldBuilderResultDecoder.geometryCoordinates(from: payload(snapshotRevision: "r2", stage: "checking"))
        XCTAssertNotNil(placing)
        XCTAssertEqual(placing, checking, "the same geometry, whatever stage the builder is in")
        XCTAssertEqual(checking?.revision, "g7")
    }

    // MARK: I3: the stage line, against an independent oracle

    private enum Presented: CaseIterable { case receiving, finalizingTrue, finalizingFalse, finalizingNil, finalized, interrupted }

    /// IOS §3c's six rules, written from the contract text, in words.
    private func oracle(_ presented: Presented, finalSolve: String?, photographic: WorldPhotographicReport?,
                        processing: WorldProcessingReport?) -> String? {
        let buildInProgress: Bool?
        switch presented {
        case .finalizingTrue: buildInProgress = true
        case .finalizingFalse: buildInProgress = false
        case .finalizingNil: buildInProgress = nil
        case .receiving, .finalized, .interrupted: return nil   // only under a finalizing world
        }
        let state = photographic?.state.rawValue
        let isArea = photographic?.scope == .area
        // 1. the room is settled and only an area is finishing.
        if isArea, ["running", "owed", "unobservable"].contains(state ?? "") { return nil }
        // 2. a known processing word.
        if let processing {
            switch processing.stage.rawValue {
            case "waiting": return "Finishing the last live update"
            case "preparing": return "Preparing images"
            case "matching": return "Matching images"
            case "placing": return "Placing images"
            case "checking":
                if let step = processing.step { return "Checking the placement · pass \(step.n) of \(step.of)" }
                return "Checking the placement"
            case "assembling": return "Assembling the world"
            default: break   // unknown: falls through
            }
        }
        // 3. photographic running for the room.
        if state == "running" {
            switch photographic?.stage {
            case "surface": return "Building surfaces"
            case "appearance": return "Adding photos"
            default: return "Building the photographic version"
            }
        }
        // 4. owed, unobservable or failed.
        if ["owed", "unobservable", "failed"].contains(state ?? "") { return nil }
        // 5. the fallback.
        if finalSolve == "pending", buildInProgress == true { return "Placing and checking images" }
        // 6.
        return nil
    }

    func testTheStageLineFollowsTheContractsPrecedence() {
        let photographics: [WorldPhotographicReport?] = [
            nil,
            WorldPhotographicReport(state: .running, stage: "surface"),
            WorldPhotographicReport(state: .running, stage: "appearance"),
            WorldPhotographicReport(state: .running),
            WorldPhotographicReport(state: .owed, stage: "appearance"),
            WorldPhotographicReport(state: .unobservable),
            WorldPhotographicReport(state: .failed, stage: "appearance"),
            WorldPhotographicReport(state: .complete),
            WorldPhotographicReport(state: .neverRecorded),
            WorldPhotographicReport(state: .running, stage: "surface", scope: .area),
            WorldPhotographicReport(state: .owed, stage: "appearance", scope: .area),
        ]
        var processings: [WorldProcessingReport?] = [nil]
        processings += WorldProcessingStage.known.map { Optional(WorldProcessingReport(stage: $0)) }
        processings.append(WorldProcessingReport(stage: .checking, step: WorldProcessingStep(n: 2, of: 3)))
        processings.append(WorldProcessingReport(stage: WorldProcessingStage(rawValue: "replaying")))
        var cases = 0
        for presented in Presented.allCases {
            for finalSolve in [nil, "pending", "solved", "skipped"] as [String?] {
                for photographic in photographics {
                    for processing in processings {
                        let isFinalizing: Bool
                        let buildInProgress: Bool?
                        switch presented {
                        case .finalizingTrue: (isFinalizing, buildInProgress) = (true, true)
                        case .finalizingFalse: (isFinalizing, buildInProgress) = (true, false)
                        case .finalizingNil: (isFinalizing, buildInProgress) = (true, nil)
                        case .receiving, .finalized, .interrupted: (isFinalizing, buildInProgress) = (false, nil)
                        }
                        let line = WorldFinishLine.line(
                            isFinalizing: isFinalizing, buildInProgress: buildInProgress,
                            finalSolve: WorldFinalSolve(word: finalSolve), photographic: photographic,
                            processing: processing)
                        XCTAssertEqual(line?.text, oracle(presented, finalSolve: finalSolve,
                                                          photographic: photographic, processing: processing),
                                       "\(presented) \(String(describing: finalSolve)) "
                                       + "\(String(describing: photographic)) \(String(describing: processing))")
                        cases += 1
                    }
                }
            }
        }
        XCTAssertEqual(cases, 6 * 4 * 11 * 9)
        // The spoken form reads the pass as a clause.
        XCTAssertEqual(WorldFinishLine.processing(.checking, step: WorldProcessingStep(n: 2, of: 3)).spoken,
                       "Checking the placement, pass 2 of 3")
        XCTAssertEqual(
            WorldFinishCopy.spoken(showsSpinner: true, line: .processing(.placing, step: nil),
                                   elapsed: WorldElapsedText(seconds: 12 * 60)),
            "The Tower is finishing this world. Now: placing images. 12 minutes since you stopped.")
    }

    // MARK: I4: the preview label

    func testEveryStillChangingNoteIsAPreviewAndSettledOnesAreUnchanged() {
        let target = WorldRenderTarget(worldID: "w1", sessionID: "s1")
        let photographics: [WorldPhotographicReport?] = [
            nil, WorldPhotographicReport(state: .running, stage: "surface"),
            WorldPhotographicReport(state: .owed), WorldPhotographicReport(state: .unobservable)]
        for stage in [WorldStage.mapping, .building, .improving, .finalizing] {
            for photographic in photographics {
                let ladder = WorldReconstruction.ladder(target: target, stage: stage, finalSolve: .notReported,
                                                        evidence: nil, photographic: photographic)
                guard case .partial(_, let note) = ladder else { return XCTFail("\(stage): \(ladder)") }
                XCTAssertTrue(note.hasPrefix("Preview — "), "\(stage): \(note)")
                XCTAssertFalse(note.contains("twenty minutes"), "no duration is promised: \(note)")
            }
        }
        XCTAssertEqual(WorldReconstruction.ladder(target: target, stage: .improving, finalSolve: .notReported,
                                                  evidence: nil).target, target)
        if case .partial(_, let note) = WorldReconstruction.ladder(
            target: target, stage: .improving, finalSolve: .notReported, evidence: nil) {
            XCTAssertEqual(note, "Preview — your final world is still building. It will look very different, "
                           + "and it is worth waiting for Saved.")
        } else { XCTFail("improving is partial") }
        for stage in [WorldStage.partial, .interrupted, .needsRetry] {
            if case .partial(_, let note) = WorldReconstruction.ladder(
                target: target, stage: stage, finalSolve: .skipped, evidence: nil) {
                XCTAssertFalse(note.hasPrefix("Preview"), "\(stage) is settled: \(note)")
            }
        }
        XCTAssertEqual(WorldReconstruction.ladder(target: target, stage: .saved, finalSolve: .solved, evidence: nil),
                       .final(target))

        XCTAssertEqual(WorldPreviewCopy.labelled("This world is X."), "Preview — this world is X.")
        XCTAssertEqual(WorldPreviewCopy.labelled("The Tower could not tell."), "Preview — the Tower could not tell.")
        XCTAssertEqual(WorldPreviewCopy.labelled("TOWER said."), "Preview — TOWER said.")
        XCTAssertEqual(WorldPreviewCopy.labelled("A"), "Preview — A")

        for stage in [WorldStage.mapping, .building, .improving, .finalizing, .saved, .partial, .interrupted,
                      .needsRetry] {
            for hasTarget in [true, false] {
                let t = hasTarget ? target : nil
                let presentation = WorldPresentation(
                    stage: stage,
                    reconstruction: WorldReconstruction.ladder(target: t, stage: stage, finalSolve: .notReported,
                                                               evidence: nil))
                let preview = stage.isStillChanging && hasTarget
                XCTAssertEqual(presentation.isPreview, preview, "\(stage) target=\(hasTarget)")
                XCTAssertEqual(presentation.openTitle, preview ? "Open the preview" : "Open the 3D world")
            }
        }
        // The elapsed clock only while unsettled.
        let at = ContinuousClock.now
        let clock = WorldFinishClock(stoppedAt: at, showsAwayBanner: false)
        XCTAssertEqual(WorldPresentation(stage: .improving, finishClock: clock).stoppedAt, at)
        XCTAssertNil(WorldPresentation(stage: .saved, finishClock: clock).stoppedAt)
    }

    // MARK: I5: the elapsed time

    func testTheElapsedTimeIsWholeMinutesAndSaysNothingElse() {
        let cases: [(Int, String, String)] = [
            (0, "Under 1 min since you stopped", "Less than a minute since you stopped"),
            (59, "Under 1 min since you stopped", "Less than a minute since you stopped"),
            (60, "1 min since you stopped", "1 minute since you stopped"),
            (12 * 60 + 59, "12 min since you stopped", "12 minutes since you stopped"),
            (61 * 60, "1 h 1 min since you stopped", "1 hour 1 minute since you stopped"),
            (65 * 60, "1 h 5 min since you stopped", "1 hour 5 minutes since you stopped"),
            (120 * 60, "2 h since you stopped", "2 hours since you stopped"),
            (120 * 60 + 59, "2 h since you stopped", "2 hours since you stopped"),
            (60 * 60, "1 h since you stopped", "1 hour since you stopped"),
        ]
        for (seconds, visible, spoken) in cases {
            let text = WorldElapsedText(seconds: seconds)
            XCTAssertEqual(text.visible, visible, "\(seconds) s")
            XCTAssertEqual(text.spoken, spoken, "\(seconds) s")
        }
        let start = ContinuousClock.now
        XCTAssertEqual(WorldElapsedText(since: start, now: start + .seconds(125)).visible, "2 min since you stopped")
    }

    // MARK: I6: the stop clock and the banner

    func testTheStopClockAndTheBannerFollowTheFollowedWalk() {
        let a = WorldFinishWalk(worldID: "w1", sessionID: "s1")
        let b = WorldFinishWalk(worldID: "w1", sessionID: "s2")
        let t0 = ContinuousClock.now
        var watch = WorldFinishWatch()

        // Never for a walk that was never seen receiving.
        watch.report(walk: a, following: true, standing: .finishing, appActive: true, now: t0)
        XCTAssertNil(watch.clock(for: a).stoppedAt, "never received: no stop instant")

        watch.report(walk: a, following: true, standing: .receiving, appActive: true, now: t0)
        XCTAssertNil(watch.clock(for: a).stoppedAt)
        // A pinned report is not the followed walk.
        watch.report(walk: a, following: false, standing: .finishing, appActive: true, now: t0 + .seconds(1))
        XCTAssertNil(watch.clock(for: a).stoppedAt, "pinned: no stop instant")
        watch.report(walk: a, following: true, standing: .finishing, appActive: true, now: t0 + .seconds(2))
        XCTAssertEqual(watch.clock(for: a).stoppedAt, t0 + .seconds(2), "the first non-receiving report")
        watch.report(walk: a, following: true, standing: .finishing, appActive: true, now: t0 + .seconds(9))
        XCTAssertEqual(watch.clock(for: a).stoppedAt, t0 + .seconds(2), "set once")
        XCTAssertNil(watch.clock(for: b).stoppedAt, "only for its own walk")
        XCTAssertEqual(watch.clock(for: nil), .unknown)

        // Away while unsettled, then settled: the banner.
        watch.app(active: false, currentWalk: a, currentStanding: .finishing)
        XCTAssertFalse(watch.clock(for: a).showsAwayBanner)
        watch.report(walk: a, following: true, standing: .settled, appActive: false, now: t0 + .seconds(60))
        XCTAssertTrue(watch.clock(for: a).showsAwayBanner, "settled while away")
        watch.dismissBanner()
        XCTAssertFalse(watch.clock(for: a).showsAwayBanner, "dismissed")
        watch.report(walk: a, following: true, standing: .settled, appActive: true, now: t0 + .seconds(61))
        XCTAssertFalse(watch.clock(for: a).showsAwayBanner, "it does not return")

        // Seen still going after coming back: no banner later.
        var back = WorldFinishWatch()
        back.report(walk: a, following: true, standing: .receiving, appActive: true, now: t0)
        back.app(active: false, currentWalk: a, currentStanding: .receiving)
        back.report(walk: a, following: true, standing: .finishing, appActive: true, now: t0 + .seconds(5))
        back.report(walk: a, following: true, standing: .settled, appActive: true, now: t0 + .seconds(9))
        XCTAssertFalse(back.clock(for: a).showsAwayBanner, "the wearer watched it settle")

        // Never away while settled, and never for no walk.
        var settled = WorldFinishWatch()
        settled.app(active: false, currentWalk: a, currentStanding: .settled)
        settled.report(walk: a, following: true, standing: .settled, appActive: true, now: t0)
        XCTAssertFalse(settled.clock(for: a).showsAwayBanner)
        settled.app(active: false, currentWalk: nil, currentStanding: .finishing)
        XCTAssertNil(settled.awayWhileFinishing)

        // A new walk clears the old one's banner and stop.
        var next = WorldFinishWatch()
        next.report(walk: a, following: true, standing: .receiving, appActive: true, now: t0)
        next.app(active: false, currentWalk: a, currentStanding: .receiving)
        next.report(walk: a, following: true, standing: .settled, appActive: false, now: t0 + .seconds(1))
        XCTAssertTrue(next.clock(for: a).showsAwayBanner)
        next.report(walk: b, following: true, standing: .receiving, appActive: true, now: t0 + .seconds(2))
        XCTAssertFalse(next.clock(for: a).showsAwayBanner, "a new walk clears the old banner")
        // And the same walk receiving again clears its stop.
        next.report(walk: b, following: true, standing: .finishing, appActive: true, now: t0 + .seconds(3))
        XCTAssertNotNil(next.clock(for: b).stoppedAt)
        next.report(walk: b, following: true, standing: .receiving, appActive: true, now: t0 + .seconds(4))
        XCTAssertNil(next.clock(for: b).stoppedAt, "receiving again: not stopped")
        // `.other` and no walk change nothing.
        let before = next
        next.report(walk: nil, following: true, standing: .finishing, appActive: true, now: t0)
        next.report(walk: b, following: true, standing: .other, appActive: true, now: t0)
        XCTAssertEqual(next, before)
    }

    // MARK: Review MED 4: the away banner is announced once per walk

    func testTheAwayBannerIsAnnouncedOncePerWalk() {
        let a = WorldFinishWalk(worldID: "w1", sessionID: "s1")
        let b = WorldFinishWalk(worldID: "w1", sessionID: "s2")
        let t0 = ContinuousClock.now
        var watch = WorldFinishWatch()
        func settleAway(_ walk: WorldFinishWalk, at seconds: Int) {
            watch.report(walk: walk, following: true, standing: .receiving, appActive: true, now: t0 + .seconds(seconds))
            watch.app(active: false, currentWalk: walk, currentStanding: .receiving)
            watch.report(walk: walk, following: true, standing: .settled, appActive: false,
                         now: t0 + .seconds(seconds + 1))
        }
        settleAway(a, at: 0)
        XCTAssertTrue(watch.clock(for: a).showsAwayBanner)
        XCTAssertTrue(watch.clock(for: a).announcesAwayBanner, "announced when it first appears")
        watch.bannerAnnounced(for: a)
        // The banner stays (a cartridge switch, a phase change, a reappearance)
        // and is not announced again.
        XCTAssertTrue(watch.clock(for: a).showsAwayBanner, "still shown")
        XCTAssertFalse(watch.clock(for: a).announcesAwayBanner, "announced twice")
        // Another walk's announcement does not count for this one, and a
        // walk with no banner records nothing.
        watch.bannerAnnounced(for: b)
        XCTAssertFalse(watch.clock(for: a).announcesAwayBanner)
        settleAway(b, at: 10)
        XCTAssertTrue(watch.clock(for: b).announcesAwayBanner, "a new walk's banner is announced")
        XCTAssertFalse(watch.clock(for: a).showsAwayBanner)
    }

    // MARK: I8: the revision's `basis`

    func testTheRevisionBasisDecodes() throws {
        func basis(_ tail: String) throws -> WorldPictureBasis? {
            try WorldRenderClient.decodeRevision(Data((#"{"revision":"s1/surface:1""# + tail).utf8)).basis
        }
        XCTAssertEqual(try basis(#","basis":"final"}"#), .final)
        XCTAssertEqual(try basis(#","basis":"walk"}"#), .walk)
        XCTAssertNil(try basis("}"))
        XCTAssertNil(try basis(#","basis":""}"#))
        XCTAssertNil(try basis(#","basis":null}"#))
        XCTAssertNil(try basis(#","basis":7}"#))
        XCTAssertEqual(try basis(#","basis":"later"}"#)?.rawValue, "later", "open, and decides nothing")
    }

    // MARK: I9: what is on screen

    func testWhatIsOnScreenFollowsThePagesManifestOrItsFetch() {
        let since = Date(timeIntervalSince1970: 100)
        let before = Date(timeIntervalSince1970: 50), after = Date(timeIntervalSince1970: 150)
        typealias M = WorldRenderViewerModel
        // The appearance rung: the manifest the page was served decides.
        XCTAssertEqual(M.basisOnScreen(polled: nil, shownRung: .appearance, servedAppearance: .final,
                                       finalSince: nil, pageFetchedAt: nil), .final)
        XCTAssertEqual(M.basisOnScreen(polled: .final, shownRung: .appearance, servedAppearance: .walk,
                                       finalSince: since, pageFetchedAt: after), .walk,
                       "the page reloads in place: its manifest, not the fetch time")
        XCTAssertEqual(M.basisOnScreen(polled: .final, shownRung: .appearance, servedAppearance: nil,
                                       finalSince: since, pageFetchedAt: after), .finalArriving)
        XCTAssertEqual(M.basisOnScreen(polled: .walk, shownRung: .appearance, servedAppearance: nil,
                                       finalSince: nil, pageFetchedAt: after), .unknown)
        // Every other rung: the poll, and when the page was fetched.
        for rung in [WorldRenderRepresentation.surface, .sparse, .dense, nil] {
            XCTAssertEqual(M.basisOnScreen(polled: .walk, shownRung: rung, servedAppearance: .final,
                                           finalSince: nil, pageFetchedAt: nil), .walk)
            XCTAssertEqual(M.basisOnScreen(polled: .final, shownRung: rung, servedAppearance: nil,
                                           finalSince: since, pageFetchedAt: after), .final)
            XCTAssertEqual(M.basisOnScreen(polled: .final, shownRung: rung, servedAppearance: nil,
                                           finalSince: since, pageFetchedAt: before), .finalArriving)
            XCTAssertEqual(M.basisOnScreen(polled: .final, shownRung: rung, servedAppearance: nil,
                                           finalSince: since, pageFetchedAt: nil), .finalArriving)
            XCTAssertEqual(M.basisOnScreen(polled: nil, shownRung: rung, servedAppearance: .final,
                                           finalSince: nil, pageFetchedAt: after), .unknown)
        }

        func manifest(_ json: String) -> WorldPictureBasis? { WorldPictureBasis.ofAppearanceManifest(Data(json.utf8)) }
        XCTAssertEqual(manifest(#"{"quality":"live"}"#), .walk)
        XCTAssertEqual(manifest(#"{"quality":"final","proxy":{"source":{"surface_quality":"live"}}}"#), .walk)
        XCTAssertEqual(manifest(#"{"quality":"final","proxy":{"source":{"surface_quality":"final"}}}"#), .final)
        XCTAssertEqual(manifest(#"{"quality":"final"}"#), .final)
        XCTAssertNil(manifest(#"{"keyframes":[]}"#), "missing")
        XCTAssertNil(manifest(#"{"quality":"draft"}"#))
        XCTAssertNil(manifest("not json"), "malformed")
    }

    /// The handler records the basis of the manifest the PAGE was served,
    /// not of its own revalidation, and forgets it with the imagery.
    func testTheServedManifestsBasisIsRememberedUntilDropped() {
        var memory = WorldAssetMemory()
        let final = WorldAssetResponse(status: 200, mimeType: "application/json", data: Data(#"{"quality":"final"}"#.utf8))
        let live = WorldAssetResponse(status: 200, mimeType: "application/json", data: Data(#"{"quality":"live"}"#.utf8))
        memory.record(.appearanceManifest, live, now: Date())
        XCTAssertEqual(memory.servedBasis, .walk)
        memory.record(.appearanceManifest, final, now: Date(), toPage: false)
        XCTAssertEqual(memory.servedBasis, .walk, "a revalidation is not what the page loaded")
        memory.record(.appearanceManifest, final, now: Date())
        XCTAssertEqual(memory.servedBasis, .final)
        memory.drop()
        XCTAssertNil(memory.servedBasis)
    }

    // MARK: I10: the settle line

    func testTheSettleLineNeverRunsAheadOfThePicture() {
        let bases: [WorldPictureBasisOnScreen] = [.unknown, .walk, .final, .finalArriving]
        for isPreview in [false, true] {
            for walkEnded in [false, true] {
                for opened in [false, true] {
                    for basis in bases {
                        for walkNote in [false, true] {
                            let expected: String?
                            if isPreview {
                                expected = walkEnded && basis == .walk ? "This picture was made during the walk." : nil
                            } else if basis == .walk {
                                expected = walkNote ? nil : "This picture was made during the walk, not from the final pass."
                            } else if !opened {
                                expected = nil
                            } else if basis == .final {
                                expected = "This is your final world."
                            } else {
                                expected = "Your final world is ready. If this picture does not change, tap Reload."
                            }
                            XCTAssertEqual(
                                WorldPreviewCopy.settleLine(isPreview: isPreview, walkEnded: walkEnded,
                                                            openedOnPreview: opened, basisOnScreen: basis,
                                                            noteSaysWalkTime: walkNote),
                                expected, "\(isPreview) \(walkEnded) \(opened) \(basis) \(walkNote)")
                        }
                    }
                }
            }
        }
    }
}
