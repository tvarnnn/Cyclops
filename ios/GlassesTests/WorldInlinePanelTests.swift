//
//  WorldInlinePanelTests.swift
//  GlassesTests
//
//  U-INLINE (managers 209, 210): the World Builder panel's phase, its one
//  web view, the scroll gate and its words. P1-P3, P6 and P9 of the
//  amendment (P4, P5, P7 and P8 are in WorldChromeTests).
//

import SwiftUI
import UIKit
import XCTest
@testable import Glasses

@MainActor
final class WorldInlinePanelTests: XCTestCase {

    private let target = WorldRenderTarget(worldID: "w1", sessionID: "s1")
    private let other = WorldRenderTarget(worldID: "w2", sessionID: "s9")

    // MARK: P1: the phase

    private func states() -> [(String, WorldModelState)] {
        let snapshot = WorldSnapshot()
        return [
            ("idle", .idle), ("awaiting", .awaitingFirstUpdate), ("receiving", .receiving(snapshot)),
            ("finalizing", .finalizing(snapshot, buildInProgress: true)), ("finalized", .finalized(snapshot)),
            ("interrupted", .interrupted(snapshot, reason: "r")),
            ("tower-failed", .failed(CartridgeFailure(kind: .towerReportedFailure, message: "It failed."))),
            ("channel-failed", .failed(CartridgeFailure(kind: .transport, message: "The channel failed."))),
        ]
    }

    /// U-INLINE §2.1's eight rules, as a list read top to bottom.
    private func oracle(capturing: Bool, session: Bool, reachable: Bool, loaded: Bool, state: WorldModelState,
                        stage: WorldStage?, target: WorldRenderTarget?, sentence: String?) -> WorldPanelPhase {
        var isReceiving = false
        if case .receiving = state { isReceiving = true }
        var towerFailure: String?
        if case .failed(let failure) = state, failure.kind == .towerReportedFailure { towerFailure = failure.message }
        let settledWithWorld: Set<String> = ["saved", "partial", "interrupted"]
        let name = stage.map { "\($0)" } ?? ""
        switch true {
        case capturing: return .walking(hasMap: true)
        case !reachable && !loaded: return .offline
        case isReceiving: return .walking(hasMap: true)
        case stage == .mapping || stage == .building || stage == .improving || stage == .finalizing:
            return .finishing(target: target)
        case settledWithWorld.contains(name) && target != nil: return .ready(target!)
        case stage == .needsRetry: return .failed(sentence: sentence)
        case towerFailure != nil: return .failed(sentence: sentence ?? towerFailure)
        case session: return .walking(hasMap: false)
        default: return .hidden
        }
    }

    func testThePanelPhaseFollowsTheRules() {
        let stages: [WorldStage?] = [nil, .mapping, .building, .improving, .finalizing, .saved, .partial,
                                     .interrupted, .needsRetry]
        var checked = 0
        for capturing in [false, true] {
            for session in [false, true] {
                for reachable in [false, true] {
                    for loaded in [false, true] {
                        for (name, state) in states() {
                            for stage in stages {
                                for target in [nil, self.target] as [WorldRenderTarget?] {
                                    for sentence in [nil, "Nothing was built."] as [String?] {
                                        let phase = WorldPanelPhase.phase(
                                            isCapturing: capturing, sessionActive: session, towerReachable: reachable,
                                            pageLoaded: loaded, state: state, stage: stage, target: target,
                                            needsRetrySentence: sentence, hasMap: true)
                                        let expected = oracle(capturing: capturing, session: session,
                                                              reachable: reachable, loaded: loaded, state: state,
                                                              stage: stage, target: target, sentence: sentence)
                                        XCTAssertEqual(phase, expected, "\(capturing) \(session) \(reachable) "
                                                       + "\(loaded) \(name) \(String(describing: stage)) "
                                                       + "\(target != nil) \(String(describing: sentence))")
                                        checked += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        XCTAssertEqual(checked, 2 * 2 * 2 * 2 * 8 * 9 * 2 * 2)

        // A page already loaded keeps the world when the Tower drops.
        XCTAssertEqual(WorldPanelPhase.phase(isCapturing: false, sessionActive: true, towerReachable: false,
                                             pageLoaded: true, state: .finalized(WorldSnapshot()), stage: .saved,
                                             target: target, needsRetrySentence: nil, hasMap: false),
                       .ready(target))

        // Interrupted is never "Saved": the heading is the stage's own word.
        for (stage, word) in [(WorldStage.interrupted, "Interrupted"), (.partial, "Partial"), (.saved, "Saved")] {
            let headline = WorldPresentation(stage: stage).headline
            XCTAssertEqual(headline, word)
            if stage != .saved { XCTAssertFalse(headline?.contains("Saved") ?? false, "\(stage): \(headline ?? "")") }
        }
        XCTAssertEqual(WorldPresentation(stage: .saved, photographic: WorldPhotographicReport(state: .failed)).headline,
                       "Saved — the photographic version could not be built")
    }

    // MARK: P2: the web view's rule

    func testTheWebViewIsWantedOnlyWhenSeenAndReady() {
        var checked = 0
        for ready in [false, true] {
            for expanded in [false, true] {
                for visible in [false, true] {
                    for scene in [ScenePhase.active, .inactive, .background] {
                        for picker in [false, true] {
                            let wanted = WorldInlineWebPolicy.wantsWebView(
                                isReady: ready, isExpanded: expanded, panelVisible: visible,
                                scenePhase: scene, pickerShown: picker)
                            let expected = scene != .background && (expanded || (ready && visible && !picker))
                            XCTAssertEqual(wanted, expected, "\(ready) \(expanded) \(visible) \(scene) \(picker)")
                            checked += 1
                        }
                    }
                }
            }
        }
        XCTAssertEqual(checked, 48)
    }

    // MARK: P3: one model, one web view

    /// A host whose lifecycle counts loads and builds a real web view, as
    /// `load()` would, so `WorldWebViewOwner.liveCount` is the proof.
    private final class Recorder {
        var runs: [ObjectIdentifier: Int] = [:]
        var made: [WorldRenderViewerModel] = []
        var heldWhenMaking: [Bool] = []
        var totalRuns: Int { runs.values.reduce(0, +) }
    }

    private func makeHost(_ recorder: Recorder, grace: Duration = .milliseconds(300)) -> WorldInlineHost {
        var host: WorldInlineHost!
        host = WorldInlineHost(
            offscreenGrace: grace,
            makeModel: { [unowned recorder] target in
                recorder.heldWhenMaking.append(host?.model != nil)
                let model = WorldRenderViewerModel(target: target)
                recorder.made.append(model)
                return model
            },
            run: { [unowned recorder] model in
                recorder.runs[ObjectIdentifier(model), default: 0] += 1
                model.web.adopt(into: UIView(), html: "<!doctype html><title>p</title>", attempt: 1, budget: 1,
                                onEvent: nil)
            })
        return host
    }

    private func settle() async {
        for _ in 0..<5 { await Task.yield() }
        try? await Task.sleep(for: .milliseconds(20))
    }

    func testTheHostKeepsOneModelAndOneWebView() async throws {
        let base = WorldWebViewOwner.liveCount
        let recorder = Recorder()
        let host = makeHost(recorder)
        func assertOne(_ step: String, line: UInt = #line) {
            XCTAssertLessThanOrEqual(WorldWebViewOwner.liveCount - base, 1, step, line: line)
        }
        let ready = WorldInlineHost.Inputs(target: target, isReady: true)

        host.update(ready)
        await settle()
        let first = try XCTUnwrap(host.model)
        XCTAssertEqual(recorder.totalRuns, 1, "ready and visible: exactly one load")
        XCTAssertEqual(WorldWebViewOwner.liveCount - base, 1)
        assertOne("ready")

        for round in 1...3 {
            host.expand(target)
            host.coverAppeared()
            XCTAssertEqual(host.placement, .expanded)
            host.isExpanded = false
            host.coverDisappeared()
            await settle()
            XCTAssertTrue(host.model === first, "round \(round): the same model")
            XCTAssertEqual(host.placement, .inline)
            assertOne("expand/collapse \(round)")
        }
        XCTAssertEqual(recorder.totalRuns, 1, "expand and collapse never load again")

        // Off screen: kept within the grace, suspended after it.
        host.visibilityChanged(false)
        try await Task.sleep(for: .milliseconds(150))
        XCTAssertTrue(host.model === first, "inside the grace: kept")
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertNil(host.model, "past the grace: suspended")
        XCTAssertEqual(WorldWebViewOwner.liveCount, base, "and its web view torn down")
        host.visibilityChanged(true)
        await settle()
        let second = try XCTUnwrap(host.model)
        XCTAssertFalse(second === first, "a new model and a fresh load")
        XCTAssertEqual(recorder.totalRuns, 2)
        assertOne("back on screen")

        // Back on screen inside the grace: nothing is lost.
        host.visibilityChanged(false)
        try await Task.sleep(for: .milliseconds(100))
        host.visibilityChanged(true)
        try await Task.sleep(for: .milliseconds(350))
        XCTAssertTrue(host.model === second, "a brief scroll away keeps the world")

        // The background suspends at once, expanded or not; `.inactive` keeps it.
        var inactive = ready
        inactive.scenePhase = .inactive
        host.update(inactive)
        XCTAssertTrue(host.model === second, "inactive keeps it")
        var background = ready
        background.scenePhase = .background
        host.update(background)
        XCTAssertNil(host.model, "background: suspended at once")
        XCTAssertEqual(WorldWebViewOwner.liveCount, base)
        host.update(ready)
        await settle()
        XCTAssertNotNil(host.model)
        host.expand(target)
        host.update(background)
        XCTAssertNil(host.model, "expanded too")
        host.update(ready)
        await settle()
        XCTAssertNotNil(host.model, "the cover gets a fresh model when the app is back")
        host.isExpanded = false
        host.coverDisappeared()

        // Saved worlds over the screen: suspended at once.
        var picker = ready
        picker.pickerShown = true
        host.update(picker)
        XCTAssertNil(host.model, "the picker is up")
        host.update(ready)
        await settle()

        // Another world: the old one is torn down BEFORE the new one exists.
        let before = host.model
        var moved = ready
        moved.target = other
        host.update(moved)
        await settle()
        XCTAssertEqual(host.model?.target, other)
        XCTAssertFalse(host.model === before)
        XCTAssertEqual(recorder.heldWhenMaking.last, false, "no model was held while the new one was made")
        assertOne("target change")

        // Not ready (a new walk, a re-finish): suspended, unless expanded.
        var walking = moved
        walking.isReady = false
        host.update(walking)
        XCTAssertNil(host.model)
        host.expand(other)
        await settle()
        XCTAssertNotNil(host.model, "expanded on a preview: wanted")
        host.isExpanded = false
        XCTAssertNotNil(host.model, "the slide-down keeps it")
        host.coverDisappeared()
        XCTAssertNil(host.model, "a collapse into a non-ready phase suspends")

        // The workspace goes (a cartridge switch).
        host.update(moved)
        await settle()
        var gone = moved
        gone.isOnScreen = false
        host.update(gone)
        XCTAssertNil(host.model)
        XCTAssertEqual(WorldWebViewOwner.liveCount, base, "every web view is gone")
        XCTAssertFalse(recorder.heldWhenMaking.contains(true), "never two models at once")
    }

    /// An area opened in the cover REPLACES the room (C1 E5): the room's web
    /// view is held off until the area is closed.
    func testAnAreaInTheCoverHoldsTheRoomOff() async throws {
        let recorder = Recorder()
        let host = makeHost(recorder)
        host.update(WorldInlineHost.Inputs(target: target, isReady: true))
        host.expand(target)
        await settle()
        XCTAssertNotNil(host.model)
        host.setShowsArea(true)
        XCTAssertNil(host.model, "the area has the screen")
        host.setShowsArea(false)
        await settle()
        XCTAssertEqual(host.model?.target, target, "back to the room")
        host.isExpanded = false
        host.coverDisappeared()
    }

    // MARK: P6: the scroll gate

    func testTheScrollGateTakesHorizontalAndTwoFingerDrags() {
        let cases: [(CGPoint, Int, Bool, String)] = [
            (CGPoint(x: 300, y: 10), 1, true, "horizontal"),
            (CGPoint(x: -300, y: 10), 1, true, "horizontal, leftwards"),
            (CGPoint(x: 10, y: 300), 1, false, "vertical"),
            (CGPoint(x: 10, y: -300), 1, false, "vertical, upwards"),
            (CGPoint(x: 200, y: 200), 1, false, "the 45° tie is vertical"),
            (CGPoint(x: -200, y: 200), 1, false, "the 45° tie, mirrored"),
            (CGPoint(x: 10, y: 300), 2, true, "two fingers, any direction"),
            (CGPoint(x: 0, y: 0), 2, true, "a two-finger pinch"),
            (CGPoint(x: 0, y: 0), 1, false, "no movement"),
            (CGPoint(x: 201, y: 200), 1, true, "just past the tie"),
        ]
        for (velocity, touches, begins, why) in cases {
            XCTAssertEqual(WorldInlineScrollGate.begins(velocity: velocity, touches: touches), begins, why)
        }
        // The installed gate defers to exactly the screen's own pan.
        let scroll = UIScrollView()
        let gate = WorldInlineScrollGate()
        gate.enclosingScrollView = scroll
        XCTAssertTrue(gate.gestureRecognizer(gate, shouldBeRequiredToFailBy: scroll.panGestureRecognizer))
        XCTAssertFalse(gate.gestureRecognizer(gate, shouldBeRequiredToFailBy: UIPanGestureRecognizer()))
        XCTAssertTrue(gate.gestureRecognizer(gate, shouldRecognizeSimultaneouslyWith: UIPanGestureRecognizer()))
        XCTAssertFalse(gate.cancelsTouchesInView, "the page keeps its touches")
        XCTAssertFalse(gate.delaysTouchesBegan)
    }

    // MARK: P9: the words

    /// None of the panel's words, nor the U0.6 words it shows, makes a claim
    /// about what was not captured (WORLDS §4's banned phrases, as U1.1 I15).
    func testTheNewWordsMakeNoCaptureAbsenceClaim() {
        let banned = ["nothing was captured", "not captured", "never captured", "nobody looked", "no one looked",
                      "was not seen", "were not seen", "unseen", "missing", "nothing is there", "is empty",
                      "complete scan", "the whole room", "everything you saw", "%", "estimate", "remaining",
                      "almost done", "nearly"]
        var words = WorldPanelCopy.all
        words += WorldProcessingStage.known.map { WorldFinishLine.processing($0, step: nil).text }
        words.append(WorldFinishLine.processing(.checking, step: WorldProcessingStep(n: 2, of: 3)).text)
        words += ["surface", "appearance", nil].map { WorldFinishLine.photographic(stage: $0).text }
        words.append(WorldFinishLine.finalPlacement.text)
        words += [0, 59, 61 * 60, 125 * 60].flatMap { seconds -> [String] in
            let text = WorldElapsedText(seconds: seconds)
            return [text.visible, text.spoken]
        }
        for word in words {
            for phrase in banned {
                XCTAssertFalse(word.lowercased().contains(phrase), "\"\(word)\" says \"\(phrase)\"")
            }
        }
        XCTAssertEqual(WorldPanelCopy.fullScreen, "Full screen")
        XCTAssertEqual(WorldPanelCopy.fullScreenHint, "Opens the 3D world full screen.")
        XCTAssertEqual(WorldFinishCopy.finished(headline: "Saved"), "Your walk finished: Saved.")
        XCTAssertEqual(WorldFinishCopy.awayBanner(headline: "Interrupted"),
                       "While you were away, your walk finished: Interrupted.")
    }

    // MARK: Layout

    func testTheStageIsThreeQuartersOfTheCardOrHalfTheScreen() {
        XCTAssertEqual(WorldPanelLayout.stageHeight(width: 370, visibleHeight: 680), 278, "iPhone 17 Pro")
        XCTAssertEqual(WorldPanelLayout.stageHeight(width: 343, visibleHeight: 550), 257, "iPhone SE")
        XCTAssertEqual(WorldPanelLayout.stageHeight(width: 343, visibleHeight: 400), 200, "half the visible height")
        XCTAssertEqual(WorldPanelLayout.stageHeight(width: 370, visibleHeight: 0), 278, "no height yet: 3:4")
    }

    // MARK: Review HIGH 1: a stage is paired only with its own walk's picture

    private let walkA = WorldFinishWalk(worldID: "w1", sessionID: "s1")
    private let walkB = WorldFinishWalk(worldID: "w2", sessionID: "s9")

    private func scoped<T>(_ walk: WorldFinishWalk?, _ value: T) -> WalkScoped<T> {
        WalkScoped(walk: walk, value: value)
    }

    private func block(_ walk: WorldFinishWalk?, _ value: String?) -> WalkScoped<String?> {
        WalkScoped(walk: walk, value: value)
    }

    func testThePanelPairsAStageOnlyWithItsOwnWalksPicture() {
        let report = scoped(walkA, "A's stage")
        XCTAssertEqual(report.picture(target), target, "the same walk")
        XCTAssertEqual(report.value(for: target), "A's stage")
        // A Saved worlds pin names its target before its own report arrives.
        XCTAssertNil(report.picture(other), "another world: the pin before its report")
        // A new session of the same world reports before its coordinates.
        XCTAssertNil(report.picture(WorldRenderTarget(worldID: "w1", sessionID: "s2")),
                     "another session of the same world")
        XCTAssertNil(report.picture(WorldRenderTarget(worldID: "w1", sessionID: nil)),
                     "a target that names no session cannot be proved this walk's")
        XCTAssertNil(scoped(nil, "a stage").picture(target), "a report that names no walk")
        XCTAssertNil(report.picture(nil))
        XCTAssertNil(report.value(for: other))
        // And so the phase: a settled stage with another walk's target is not ready.
        let phase = WorldPanelPhase.phase(
            isCapturing: false, sessionActive: false, towerReachable: true, pageLoaded: false,
            state: .finalized(WorldSnapshot()), stage: .saved,
            target: report.picture(other), needsRetrySentence: nil, hasMap: false)
        XCTAssertFalse(phase.isReady, "\(phase)")
    }

    // MARK: Review HIGH 2: a replaced ready world never shows the first one's wait

    func testAReplacedReadyWorldNeverShowsTheFirstWorldsWait() {
        var memory = WorldPanelFinishMemory<String>()
        memory.record(block(walkA, "A's finish block"))
        XCTAssertFalse(memory.phaseChanged(from: .walking(hasMap: false), to: .finishing(target: target)))
        XCTAssertTrue(memory.phaseChanged(from: .finishing(target: target), to: .ready(target)), "arrived from the wait")
        XCTAssertEqual(memory.overlay(for: target), "A's finish block", "A's wait over A")
        // Ready A → ready B: B's wait overlay is "Opening your world…".
        XCTAssertFalse(memory.phaseChanged(from: .ready(target), to: .ready(other)))
        XCTAssertNil(memory.overlay(for: other), "A's finish block over B")
        XCTAssertNil(memory.overlay(for: target), "replaced: forgotten")
        // A block recorded for another walk never covers this one.
        var foreign = WorldPanelFinishMemory<String>()
        foreign.record(block(walkB, "B's finish block"))
        foreign.phaseChanged(from: .finishing(target: target), to: .ready(target))
        XCTAssertNil(foreign.overlay(for: target), "another walk's block")
        // Not ready: nothing to overlay.
        memory.record(block(walkA, "A again"))
        memory.phaseChanged(from: .ready(other), to: .offline)
        XCTAssertNil(memory.overlay(for: target))
    }

    func testThePanelDrawsOnlyThePhaseTargetsModel() async throws {
        let recorder = Recorder()
        let host = makeHost(recorder)
        host.update(WorldInlineHost.Inputs(target: target, isReady: true))
        await settle()
        let model = try XCTUnwrap(host.model)
        XCTAssertTrue(host.model(for: target) === model)
        XCTAssertNil(host.model(for: other), "A's page drawn under B's phase")
        XCTAssertNil(host.model(for: nil))
        host.suspend()
        withExtendedLifetime(recorder) {}
    }

    // MARK: Review MED 3: the Tower drops under an open preview

    func testTheTowerDroppingUnderAnOpenPreviewDropsTheLiveClaims() {
        typealias Block = WorldInlinePanel<EmptyView>.FinishingBlock
        let stop = ContinuousClock.now
        let live = Block(showsSpinner: true, line: .finalPlacement, stoppedAt: stop, detail: "d")
        XCTAssertEqual(live.reachable(true), live, "connected: unchanged")
        let dropped = live.reachable(false)
        XCTAssertFalse(dropped.showsSpinner, "no spinner")
        XCTAssertNil(dropped.line, "no stage")
        XCTAssertNil(dropped.detail, "no Tower sentence")
        XCTAssertTrue(dropped.offline, "the offline notice")
        XCTAssertEqual(dropped.stoppedAt, stop, "the phone's own clock stays")

        let progress = WorldViewerProgress(isPreview: true, walkEnded: true, line: .finalPlacement,
                                           stoppedAt: stop, noteSaysWalkTime: false)
        XCTAssertEqual(progress.reachable(true), progress)
        XCTAssertNil(progress.reachable(false).line, "the cover claims no live stage")
        XCTAssertTrue(progress.reachable(false).isPreview, "the loaded picture is still a preview")
        XCTAssertEqual(WorldPanelCopy.coverNotice("N.", towerReachable: true), "N.")
        XCTAssertEqual(WorldPanelCopy.coverNotice("N.", towerReachable: false), WorldPanelCopy.readyButOffline + " N.")
        XCTAssertEqual(WorldPanelCopy.coverNotice(nil, towerReachable: false), WorldPanelCopy.readyButOffline)
    }

    // MARK: Review 2, MED 1: one voice for the finishing wait

    func testOnlyThePanelNarratesTheFinishingWaitWhenItIsShowingIt() {
        XCTAssertTrue(WorldPanelPhase.finishing(target: nil).narratesFinishing, "the canvas would say it twice")
        XCTAssertTrue(WorldPanelPhase.finishing(target: target).narratesFinishing)
        for phase: WorldPanelPhase in [.hidden, .walking(hasMap: false), .walking(hasMap: true), .ready(target),
                                       .failed(sentence: nil), .offline] {
            XCTAssertFalse(phase.narratesFinishing, "\(phase): the canvas keeps its own words")
        }
    }

    // MARK: Review 2, MED 2: VoiceOver reads the world before its inline chrome

    func testVoiceOverReadsTheWorldBeforeTheInlineChromeOverIt() {
        // Siblings in the stage's container: the higher priority is read
        // first. (An XCUITest snapshot lists the hierarchy, not VoiceOver's
        // order, so this is pinned here.)
        XCTAssertGreaterThan(WorldPanelOrder.world, WorldChromeInlineOrder.dark, "the dark line before the world")
        XCTAssertGreaterThan(WorldPanelOrder.world, WorldChromeInlineOrder.hint, "the hint before the world")
        XCTAssertGreaterThan(WorldChromeInlineOrder.dark, WorldChromeInlineOrder.hint)
    }

    // MARK: Review 3, HIGH: the cover's words are its picture's walk's

    private let stop = ContinuousClock.now

    private var aWords: WorldCoverText {
        WorldCoverText(
            title: "Kitchen", note: "A's note.", notice: "A's notice.",
            progress: WorldViewerProgress(isPreview: true, walkEnded: true, line: .processing(.placing, step: nil),
                                          stoppedAt: stop, noteSaysWalkTime: false),
            inProgress: true)
    }

    private var bWords: WorldCoverText {
        WorldCoverText(
            title: "Hall", note: "B's note.", notice: "B's notice.",
            progress: WorldViewerProgress(isPreview: true, walkEnded: false, line: .processing(.matching, step: nil),
                                          stoppedAt: nil, noteSaysWalkTime: false),
            inProgress: true)
    }

    /// The cover is open on A's picture, then a report for B arrives: the
    /// picture stays A's, so the words do too -- static, no live stage --
    /// with a notice that another walk is reported. Never B's progress or
    /// text over A's picture. Closed, it says nothing.
    func testTheCoverKeepsItsOwnWalksWordsWhenAReportForAnotherWalkArrives() async throws {
        let recorder = Recorder()
        let host = makeHost(recorder)
        host.update(WorldInlineHost.Inputs(target: target, isReady: true))
        await settle()
        host.expand(target, words: scoped(walkA, aWords))
        host.coverAppeared()
        await settle()
        func shown(_ live: WalkScoped<WorldCoverText>, reachable: Bool = true) -> WorldCoverText {
            host.coverWords.words(live: live, towerReachable: reachable)
        }
        XCTAssertEqual(shown(scoped(walkA, aWords)), aWords, "A's report over A's picture")
        // A's own report moves on: the cover follows it.
        var aLater = aWords
        aLater.progress?.line = .processing(.checking, step: nil)
        host.coverReported(scoped(walkA, aLater))
        XCTAssertEqual(shown(scoped(walkA, aLater)), aLater)

        // B's report. The panel's phase moves to B under the cover.
        host.update(WorldInlineHost.Inputs(target: other, isReady: false))
        host.coverReported(scoped(walkB, bWords))
        await settle()
        XCTAssertEqual(host.model?.target, target, "the cover keeps A's picture")
        let words = shown(scoped(walkB, bWords))
        XCTAssertEqual(words.title, "Kitchen", "B's title over A's picture")
        XCTAssertEqual(words.note, "A's note.", "B's note over A's picture")
        XCTAssertEqual(words.notice, WorldPanelCopy.coverNewWalk + " A's notice.")
        XCTAssertNil(words.progress?.line, "a stage over A's picture: B's, or A's no longer current")
        XCTAssertEqual(words.progress?.stoppedAt, stop, "A's own clock, not B's")
        XCTAssertEqual(words.progress?.walkEnded, true, "A's progress, not B's")
        var bSettled = bWords
        bSettled.inProgress = false
        XCTAssertEqual(shown(scoped(walkB, bSettled)).notice, WorldPanelCopy.coverOtherWalk + " A's notice.")
        XCTAssertEqual(shown(scoped(walkB, bWords), reachable: false).notice,
                       WorldPanelCopy.readyButOffline + " " + WorldPanelCopy.coverNewWalk + " A's notice.")
        // A report naming no walk (a reconnect): A's words, static, and no
        // claim about another walk.
        let unnamed = shown(scoped(nil, bWords))
        XCTAssertEqual(unnamed.title, "Kitchen")
        XCTAssertEqual(unnamed.notice, "A's notice.")
        XCTAssertNil(unnamed.progress?.line)
        // A again: live again.
        XCTAssertEqual(shown(scoped(walkA, aLater)), aLater)

        // Collapsed: no picture, nothing held, nothing said.
        host.isExpanded = false
        host.coverDisappeared()
        XCTAssertEqual(shown(scoped(walkB, bWords)), WorldCoverText(), "closed: no picture, no words")
        host.coverReported(scoped(walkA, aWords))
        XCTAssertEqual(host.coverWords, WorldCoverBinding(), "closed: nothing recorded")
        host.suspend()
        withExtendedLifetime(recorder) {}
    }

    // MARK: Review 4, HIGH 2: A, B, then A again

    /// The cover on A, a report for B, then A's walk again. The old binding
    /// took the live words as A's the moment the walk said A -- while B's
    /// words could still be live, published separately. The words now carry
    /// their walk, so in ANY order of arrival the cover over A's picture
    /// shows A's words (live or held) and never one word of B's.
    func testTheCoverOnABThenANeverShowsBsWordsOverAsPicture() {
        var cover = WorldCoverBinding()
        cover.opened(picture: target, live: scoped(walkA, aWords))
        let arrivals = [scoped(walkB, bWords), scoped(nil, bWords), scoped(walkA, aWords),
                        scoped(walkB, bWords), scoped(walkA, aWords)]
        for live in arrivals {
            cover.reported(live)
            let words = cover.words(live: live, towerReachable: true)
            XCTAssertEqual(words.title, "Kitchen", "\(live.walk as Any)")
            XCTAssertEqual(words.note, "A's note.")
            XCTAssertNotEqual(words.progress?.line, bWords.progress?.line, "B's stage over A's picture")
            XCTAssertNotEqual(words.progress?.stoppedAt, nil, "A's clock")
            XCTAssertEqual(words.progress?.line != nil, live.walk == walkA, "a live line only while A is reported")
        }
    }

    /// Review 4, HIGH 1 in the cover: a picture opened before its walk's
    /// report (the header in the old gap, or the canvas's ladder, which may
    /// offer a pinned target before its report) says nothing of the walk
    /// that IS reported, and takes its own walk's words when they arrive.
    func testACoverOpenedBeforeItsWalksReportSaysNothingOfAnotherWalk() {
        var cover = WorldCoverBinding()
        cover.opened(picture: other, live: scoped(walkA, aWords))
        let before = cover.words(live: scoped(walkA, aWords), towerReachable: true)
        XCTAssertEqual(before, WorldCoverText(), "A's words over B's picture: \(before)")
        XCTAssertNil(cover.words(live: scoped(nil, aWords), towerReachable: true).title)
        cover.reported(scoped(walkB, bWords))
        XCTAssertEqual(cover.words(live: scoped(walkB, bWords), towerReachable: true), bWords, "B's own report")
        XCTAssertEqual(cover.words(live: scoped(walkA, aWords), towerReachable: true).title, "Hall",
                       "B's held words once A is reported again")
    }

    // MARK: Review 4: the finish block, "finished", the banner and the progress travel with their walk

    /// A pin to B while A finishes: B's picture (its phase) arrives before
    /// B's block. A's block is never shown over B, "finished" is never said;
    /// and A, B, A: B's wait replaced A's, so A's ready world after it
    /// announces nothing and shows nothing of B's -- only A's own next wait
    /// does.
    func testTheFinishBlockAndTheFinishedAnnouncementTravelWithTheirWalk() {
        var memory = WorldPanelFinishMemory<String>()
        memory.record(block(walkA, "A's block"))
        XCTAssertFalse(memory.phaseChanged(from: .finishing(target: target), to: .ready(other)), "B finished")
        XCTAssertNil(memory.overlay(for: other), "A's block over B")
        memory.record(block(walkB, "B's block"))
        XCTAssertFalse(memory.phaseChanged(from: .finishing(target: nil), to: .ready(target)), "A finished from B's wait")
        XCTAssertNil(memory.overlay(for: target), "B's block over A")
        memory.record(block(walkA, "A's block again"))
        XCTAssertTrue(memory.phaseChanged(from: .finishing(target: nil), to: .ready(target)))
        XCTAssertEqual(memory.overlay(for: target), "A's block again")
        // Not finishing (`nil`) keeps what there is, with its walk.
        memory.record(block(walkB, nil))
        XCTAssertEqual(memory.overlay(for: target), "A's block again")
    }

    /// The finish block's progress line is the report's, recorded for the
    /// report's walk -- A, B, A -- so the memory never holds one walk's line
    /// under another's identity.
    // MARK: Build loading (after a local Stop)

    /// After a local Stop with a building status: no map, no current piece,
    /// the loading view with its stage text and a progress indicator.
    func testAfterStopWhileBuildingThePanelIsTheLoadingViewAlone() throws {
        typealias Panel = WorldInlinePanel<EmptyView>
        let phase = WorldPanelPhase.phase(
            isCapturing: false, sessionActive: true, towerReachable: true, pageLoaded: true,
            state: .finalizing(WorldSnapshot(), buildInProgress: true), stage: .improving, target: nil,
            needsRetrySentence: nil, hasMap: true)
        XCTAssertEqual(phase, .finishing(target: nil))
        XCTAssertEqual(Panel.stageKind(phase), .loading)
        XCTAssertNotEqual(Panel.stageKind(phase), .map)
        let report = scoped(walkA, WorldPresentation(stage: .improving,
                                                     finishLine: .processing(.placing, step: nil)))
        let block = try XCTUnwrap(Panel.finishing(report, phase: phase, buildInProgress: true,
                                                  towerReachable: true).value)
        XCTAssertNotNil(block.line, "the U0.6 stage text")
        XCTAssertTrue(block.showsSpinner)
    }

    /// Source scan (Mac only): the loading view draws no map or piece.
    func testTheLoadingViewSourceDrawsNoMapOrPiece() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Glasses/Workspaces/WorldBuilder/WorldInlinePanel.swift")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("sources are not on a device") }
        let src = try String(contentsOf: url, encoding: .utf8)
        let body = try XCTUnwrap(src.components(separatedBy: "private func finishingContent").last?
            .components(separatedBy: "private func stageText").first)
        XCTAssertFalse(body.contains("WorldCoverageMapView") || body.contains("WorldCurrentPieceView")
                       || body.contains("map"), "the loading view draws no map")
    }

    /// The ready status shows the inline viewer with no tap or refresh: the
    /// phase follows the status alone.
    func testTheReadyStatusShowsTheInlineViewerWithoutUserAction() {
        typealias Panel = WorldInlinePanel<EmptyView>
        func phase(_ state: WorldModelState, _ stage: WorldStage) -> WorldPanelPhase {
            .phase(isCapturing: false, sessionActive: true, towerReachable: true, pageLoaded: true,
                   state: state, stage: stage, target: target, needsRetrySentence: nil, hasMap: true)
        }
        let building = phase(.finalizing(WorldSnapshot(), buildInProgress: true), .building)
        XCTAssertEqual(Panel.stageKind(building), .loading)
        let ready = phase(.finalized(WorldSnapshot()), .saved)
        XCTAssertEqual(ready, .ready(target))
        XCTAssertEqual(Panel.stageKind(ready), .viewer)
    }

    func testTheFinishingProgressIsRecordedForItsOwnWalk() {
        typealias Panel = WorldInlinePanel<EmptyView>
        let a = scoped(walkA, WorldPresentation(stage: .improving, finishLine: .processing(.placing, step: nil)))
        let b = scoped(walkB, WorldPresentation(stage: .improving, finishLine: .processing(.matching, step: nil)))
        var memory = WorldPanelFinishMemory<Panel.FinishingBlock>()
        for (report, line) in [(a, WorldFinishLine.processing(.placing, step: nil)),
                               (b, .processing(.matching, step: nil)), (a, .processing(.placing, step: nil))] {
            let finishing = Panel.finishing(report, phase: .finishing(target: nil), buildInProgress: true,
                                            towerReachable: true)
            XCTAssertEqual(finishing.walk, report.walk)
            XCTAssertEqual(finishing.value?.line, line)
            memory.record(finishing)
            XCTAssertEqual(memory.block?.walk, report.walk, "a line recorded for another walk")
        }
        XCTAssertNil(Panel.finishing(a, phase: .ready(target), buildInProgress: true, towerReachable: true).value)
        memory.phaseChanged(from: .finishing(target: nil), to: .ready(other))
        XCTAssertNil(memory.overlay(for: other), "A's line over B")
    }

    /// The away banner: drawn from the report's clock and headline, and
    /// announced for the report's walk -- one value, so A, B, A never marks
    /// one walk announced for another walk's banner.
    func testTheAwayBannerIsAnnouncedForTheWalkItWasDrawnFor() {
        typealias Panel = WorldInlinePanel<EmptyView>
        let away = WorldFinishClock(stoppedAt: nil, showsAwayBanner: true, announcesAwayBanner: true)
        let a = scoped(walkA, WorldPresentation(stage: .saved, finishClock: away))
        let b = scoped(walkB, WorldPresentation(stage: .interrupted, finishClock: away))
        for report in [a, b, a] {
            let banner = Panel.awayBanner(report, phase: .ready(target))
            XCTAssertEqual(banner?.walk, report.walk)
            XCTAssertEqual(banner?.value, WorldFinishCopy.awayBanner(headline: report.value.headline ?? "?"))
        }
        XCTAssertNotNil(Panel.awayBanner(b, phase: .failed(sentence: nil)))
        XCTAssertNil(Panel.awayBanner(a, phase: .finishing(target: target)), "not over the wait")
        XCTAssertNil(Panel.awayBanner(scoped(walkA, WorldPresentation(stage: .saved)), phase: .ready(target)),
                     "no banner on the clock")
    }

    // MARK: Review 4: the one pairing path

    /// Every surface that puts words or progress beside a picture -- the
    /// header's Picture, the panel's phase, the cover (opened with the words
    /// or given them later), the finish overlay and "finished", the Saved
    /// worlds note -- driven through the real view model, over every picture
    /// × walk combination. Each pairs exactly when `WalkScoped.describes`
    /// says the words are about the picture's walk: there is no other rule.
    func testEverySurfacePairsWordsWithAPictureOnlyThroughTheOneHelper() {
        let pictures = [target, other, WorldRenderTarget(worldID: "w1", sessionID: "s2"),
                        WorldRenderTarget(worldID: "w1", sessionID: nil)]
        let walks: [WorldFinishWalk?] = [walkA, walkB, WorldFinishWalk(worldID: "w1", sessionID: "s2"), nil]
        let surfaces: [(String, (WorldBuilderViewModel, WorldRenderTarget) -> Bool)] = [
            ("the header's Picture", { world, picture in WorldBuilderWorkspaceView.headerPicture(world) == picture }),
            ("the canvas's 3D controls", { world, picture in
                WorldCanvasPictureOffer(world.walkPresentation)?.target == picture }),
            ("the panel's phase", { world, picture in
                WorldPanelPhase.phase(
                    isCapturing: false, sessionActive: false, towerReachable: true, pageLoaded: false,
                    state: world.state, stage: world.presentation.stage, target: world.panelTarget,
                    needsRetrySentence: nil, hasMap: false) == .ready(picture)
            }),
            ("the cover, opened with the words", { world, picture in
                var cover = WorldCoverBinding()
                cover.opened(picture: picture, live: world.coverWords)
                return cover.words(live: world.coverWords, towerReachable: true).title == "Title"
            }),
            ("the cover, given the words later", { world, picture in
                var cover = WorldCoverBinding()
                cover.opened(picture: picture, live: WalkScoped(walk: nil, value: WorldCoverText()))
                cover.reported(world.coverWords)
                return cover.words(live: WalkScoped(walk: nil, value: WorldCoverText()), towerReachable: true)
                    .title == "Title"
            }),
            ("the finish overlay and \"finished\"", { world, picture in
                var memory = WorldPanelFinishMemory<String>()
                memory.record(world.walkPresentation.map { _ -> String? in "block" })
                let finished = memory.phaseChanged(from: .finishing(target: nil), to: .ready(picture))
                return finished && memory.overlay(for: picture) == "block"
            }),
            ("the Saved worlds note", { world, picture in world.walkPresentation.value(for: picture) != nil }),
        ]
        var paired = 0
        for picture in pictures {
            for walk in walks {
                let world = WorldBuilderViewModel(client: UnavailableWorldBuilderClient())
                world.open(worldID: picture.worldID, sessionID: picture.sessionID)
                world.walkReportDidChange(to: WalkScoped(walk: walk, value: WorldWalkReport(
                    state: .finalized(WorldSnapshot(name: "Title", worldID: walk?.worldID ?? picture.worldID)),
                    finalization: WorldFinalizationReport(state: .complete, finalSolve: "solved"))))
                XCTAssertEqual(world.presentation.stage, .saved)
                let expected = WalkScoped(walk: walk, value: ()).describes(picture)
                if expected { paired += 1 }
                for (surface, pairs) in surfaces {
                    XCTAssertEqual(pairs(world, picture), expected,
                                   "\(surface): picture \(picture), words of \(walk as Any)")
                }
            }
        }
        XCTAssertEqual(paired, 3, "the matrix pairs only a picture with its own walk")
    }

    /// And no surface has a rule of its own: the panel, its phase, the
    /// workspace and the canvas compare no world or session ids, read no walk
    /// or picture target apart from a `WalkScoped` (a ladder's target only
    /// through `picture(_:)`), and compare no walks -- the helper
    /// in `WorldWalkScope.swift` is the only place a picture meets words.
    func testNoSurfaceComparesAWalkWithAPictureOutsideTheOneHelper() throws {
        let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Glasses/Workspaces/WorldBuilder")
        guard FileManager.default.fileExists(atPath: folder.path) else {
            #if targetEnvironment(simulator)
            return XCTFail("sources not readable at \(folder.path)")
            #else
            throw XCTSkip("sources are not on a device")
            #endif
        }
        let banned = [#"\b(worldID|sessionID)\s*(==|!=)"#, #"(==|!=)\s*\w*\.(worldID|sessionID)\b"#,
                      #"\bpresentedWalk\b"#, #"\brenderTarget\b"#, #"\.walk\s*(==|!=)(?!\s*nil)"#,
                      #"^(?!.*\.picture\().*\breconstruction\.target\b"#]
            .map { try! NSRegularExpression(pattern: $0) }
        for file in ["WorldPanelPhase.swift", "WorldInlinePanel.swift", "WorldBuilderWorkspaceView.swift",
                     "WorldCanvasView.swift"] {
            let text = try String(contentsOf: folder.appendingPathComponent(file), encoding: .utf8)
            XCTAssertGreaterThan(text.count, 1000, file)
            for (number, line) in text.components(separatedBy: "\n").enumerated() {
                let code = line.trimmingCharacters(in: .whitespaces)
                guard !code.hasPrefix("//") else { continue }
                for pattern in banned
                where pattern.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) != nil {
                    XCTFail("\(file):\(number + 1) pairs outside the helper: \(code)")
                }
            }
        }
        let helper = try String(contentsOf: folder.appendingPathComponent("WorldWalkScope.swift"), encoding: .utf8)
        XCTAssertTrue(helper.contains("func describes(_ picture: WorldRenderTarget?) -> Bool"))
    }

    // MARK: Review 3, MED: the ready page's loading wait, offline

    func testTheTowerDroppingDuringTheReadyPagesLoadingWaitDropsTheSavedStage() {
        typealias Panel = WorldInlinePanel<EmptyView>
        let saved = Panel.FinishingBlock(showsSpinner: true, line: .processing(.assembling, step: nil),
                                         stoppedAt: stop, detail: nil)
        var memory = WorldPanelFinishMemory<Panel.FinishingBlock>()
        memory.record(scoped(walkA, Optional(saved)))
        memory.phaseChanged(from: .finishing(target: target), to: .ready(target))
        XCTAssertEqual(Panel.waitBlock(memory, over: target, towerReachable: true), saved, "connected: as it stood")
        let offline = try? XCTUnwrap(Panel.waitBlock(memory, over: target, towerReachable: false))
        XCTAssertEqual(offline?.showsSpinner, false, "a spinner with the Tower gone")
        XCTAssertNil(offline?.line, "the last live stage claimed as current")
        XCTAssertEqual(offline?.offline, true, "the offline notice")
        XCTAssertEqual(offline?.stoppedAt, stop, "the phone's own clock stays")
        XCTAssertNil(Panel.waitBlock(memory, over: other, towerReachable: false), "another world's wait")
    }

    // MARK: Review 3, LOW-MED: "finished" only for the walk that was finishing

    func testFinishedIsAnnouncedOnlyForTheWorldThatWasFinishing() {
        var memory = WorldPanelFinishMemory<String>()
        memory.record(block(walkA, "A's finish block"))
        // A pin to another world while A finishes: not "finished".
        XCTAssertFalse(memory.phaseChanged(from: .finishing(target: target), to: .ready(other)),
                       "finished for the wrong world")
        XCTAssertNil(memory.overlay(for: other))
        XCTAssertFalse(memory.phaseChanged(from: .finishing(target: nil), to: .ready(other)), "no preview either")
        // Another session of the same world is another walk.
        XCTAssertFalse(memory.phaseChanged(from: .finishing(target: target),
                                           to: .ready(WorldRenderTarget(worldID: "w1", sessionID: "s2"))))
        // A's own wait settling: announced once.
        XCTAssertTrue(memory.phaseChanged(from: .finishing(target: nil), to: .ready(target)))
        XCTAssertEqual(memory.overlay(for: target), "A's finish block")
        // No block was ever drawn: nothing came from a wait.
        var empty = WorldPanelFinishMemory<String>()
        XCTAssertFalse(empty.phaseChanged(from: .finishing(target: target), to: .ready(target)))
    }

    // MARK: The lead's ruling: no third "not connected" line

    func testOfflineSaysNotConnectedOnlyWhenTheScreenDoesNot() {
        XCTAssertNil(WorldPanelCopy.offlineLine(screenSaysOffline: true), "a third line")
        XCTAssertEqual(WorldPanelCopy.offlineLine(screenSaysOffline: false), WorldPanelCopy.notConnected)
    }
}
