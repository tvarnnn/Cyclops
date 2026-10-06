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

    func testThePanelPairsAStageOnlyWithItsOwnWalksPicture() {
        let walk = WorldFinishWalk(worldID: "w1", sessionID: "s1")
        XCTAssertEqual(WorldPanelPhase.target(target, matching: walk), target, "the same walk")
        // A Saved worlds pin names its target before its own report arrives.
        XCTAssertNil(WorldPanelPhase.target(other, matching: walk), "another world: the pin before its report")
        // A new session of the same world reports before its coordinates.
        XCTAssertNil(WorldPanelPhase.target(WorldRenderTarget(worldID: "w1", sessionID: "s2"), matching: walk),
                     "another session of the same world")
        XCTAssertNil(WorldPanelPhase.target(WorldRenderTarget(worldID: "w1", sessionID: nil), matching: walk),
                     "a target that names no session cannot be proved this walk's")
        XCTAssertNil(WorldPanelPhase.target(target, matching: nil), "a report that names no walk")
        XCTAssertNil(WorldPanelPhase.target(nil, matching: walk))
        // And so the phase: a settled stage with another walk's target is not ready.
        let phase = WorldPanelPhase.phase(
            isCapturing: false, sessionActive: false, towerReachable: true, pageLoaded: false,
            state: .finalized(WorldSnapshot()), stage: .saved,
            target: WorldPanelPhase.target(other, matching: walk), needsRetrySentence: nil, hasMap: false)
        XCTAssertFalse(phase.isReady, "\(phase)")
    }

    // MARK: Review HIGH 2: a replaced ready world never shows the first one's wait

    func testAReplacedReadyWorldNeverShowsTheFirstWorldsWait() {
        let walkA = WorldFinishWalk(worldID: "w1", sessionID: "s1")
        let walkB = WorldFinishWalk(worldID: "w2", sessionID: "s9")
        var memory = WorldPanelFinishMemory<String>()
        memory.record("A's finish block", walk: walkA)
        XCTAssertFalse(memory.phaseChanged(from: .walking(hasMap: false), to: .finishing(target: target)))
        XCTAssertTrue(memory.phaseChanged(from: .finishing(target: target), to: .ready(target)), "arrived from the wait")
        XCTAssertEqual(memory.overlay(for: target), "A's finish block", "A's wait over A")
        // Ready A → ready B: B's wait overlay is "Opening your world…".
        XCTAssertFalse(memory.phaseChanged(from: .ready(target), to: .ready(other)))
        XCTAssertNil(memory.overlay(for: other), "A's finish block over B")
        XCTAssertNil(memory.overlay(for: target), "replaced: forgotten")
        // A block recorded for another walk never covers this one.
        var foreign = WorldPanelFinishMemory<String>()
        foreign.record("B's finish block", walk: walkB)
        foreign.phaseChanged(from: .finishing(target: target), to: .ready(target))
        XCTAssertNil(foreign.overlay(for: target), "another walk's block")
        // Not ready: nothing to overlay.
        memory.record("A again", walk: walkA)
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

    /// The cover is open on A's picture, then a report for B arrives: the
    /// picture stays A's, so the words do too -- static, no live stage --
    /// with a notice that another walk is reported. Never B's progress or
    /// text over A's picture. Closed, the panel's rules apply again.
    func testTheCoverKeepsItsOwnWalksWordsWhenAReportForAnotherWalkArrives() async throws {
        let walkA = WorldFinishWalk(worldID: "w1", sessionID: "s1")
        let walkB = WorldFinishWalk(worldID: "w2", sessionID: "s9")
        let stop = ContinuousClock.now
        let aWords = WorldCoverText(
            title: "Kitchen", note: "A's note.", notice: "A's notice.",
            progress: WorldViewerProgress(isPreview: true, walkEnded: true, line: .processing(.placing, step: nil),
                                          stoppedAt: stop, noteSaysWalkTime: false))
        let bWords = WorldCoverText(
            title: "Hall", note: "B's note.", notice: "B's notice.",
            progress: WorldViewerProgress(isPreview: true, walkEnded: false, line: .processing(.matching, step: nil),
                                          stoppedAt: nil, noteSaysWalkTime: false))
        let recorder = Recorder()
        let host = makeHost(recorder)
        host.update(WorldInlineHost.Inputs(target: target, isReady: true))
        await settle()
        host.expand(target, walk: walkA, words: aWords)
        host.coverAppeared()
        await settle()
        func shown(_ live: WorldCoverText, _ walk: WorldFinishWalk?, inProgress: Bool = true,
                   reachable: Bool = true) -> WorldCoverText {
            host.coverWords.words(live: live, presented: walk, presentedInProgress: inProgress,
                                  towerReachable: reachable)
        }
        XCTAssertEqual(shown(aWords, walkA), aWords, "A's report over A's picture")
        // A's own report moves on: the cover follows it.
        var aLater = aWords
        aLater.progress?.line = .processing(.checking, step: nil)
        host.coverReported(aLater, walk: walkA)
        XCTAssertEqual(shown(aLater, walkA), aLater)

        // B's report: the walk first (published ahead of the state), then
        // B's words. The panel's phase moves to B under the cover.
        host.update(WorldInlineHost.Inputs(target: other, isReady: false))
        host.coverReported(bWords, walk: walkB)
        await settle()
        XCTAssertEqual(host.model?.target, target, "the cover keeps A's picture")
        for live in [aLater, bWords] {
            let words = shown(live, walkB)
            XCTAssertEqual(words.title, "Kitchen", "B's title over A's picture")
            XCTAssertEqual(words.note, "A's note.", "B's note over A's picture")
            XCTAssertEqual(words.notice, WorldPanelCopy.coverNewWalk + " A's notice.")
            XCTAssertNil(words.progress?.line, "a stage over A's picture: B's, or A's no longer current")
            XCTAssertEqual(words.progress?.stoppedAt, stop, "A's own clock, not B's")
            XCTAssertEqual(words.progress?.walkEnded, true, "A's progress, not B's")
        }
        XCTAssertEqual(shown(bWords, walkB, inProgress: false).notice, WorldPanelCopy.coverOtherWalk + " A's notice.")
        XCTAssertEqual(shown(bWords, walkB, reachable: false).notice,
                       WorldPanelCopy.readyButOffline + " " + WorldPanelCopy.coverNewWalk + " A's notice.")
        // A report naming no walk (a reconnect): A's words, static, and no
        // claim about another walk.
        let unnamed = shown(bWords, nil)
        XCTAssertEqual(unnamed.title, "Kitchen")
        XCTAssertEqual(unnamed.notice, "A's notice.")
        XCTAssertNil(unnamed.progress?.line)
        // A again: live again.
        XCTAssertEqual(shown(aLater, walkA), aLater)

        // Collapsed: nothing held, the panel's rules again.
        host.isExpanded = false
        host.coverDisappeared()
        XCTAssertEqual(shown(bWords, walkB), bWords, "closed: nothing held")
        host.coverReported(aWords, walk: walkA)
        XCTAssertEqual(shown(bWords, walkB), bWords, "closed: nothing recorded")
        host.suspend()
        withExtendedLifetime(recorder) {}
    }

    // MARK: Review 3, MED: the ready page's loading wait, offline

    func testTheTowerDroppingDuringTheReadyPagesLoadingWaitDropsTheSavedStage() {
        typealias Panel = WorldInlinePanel<EmptyView>
        let walk = WorldFinishWalk(worldID: "w1", sessionID: "s1")
        let stop = ContinuousClock.now
        let saved = Panel.FinishingBlock(showsSpinner: true, line: .processing(.assembling, step: nil),
                                         stoppedAt: stop, detail: nil)
        var memory = WorldPanelFinishMemory<Panel.FinishingBlock>()
        memory.record(saved, walk: walk)
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
        let walkA = WorldFinishWalk(worldID: "w1", sessionID: "s1")
        var memory = WorldPanelFinishMemory<String>()
        memory.record("A's finish block", walk: walkA)
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
