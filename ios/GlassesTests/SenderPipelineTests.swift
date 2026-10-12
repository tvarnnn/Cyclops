//
//  SenderPipelineTests.swift
//  GlassesTests
//
//  Covers the three pieces that decide, and then account for, how many
//  captured frames reach the Tower: `FrameRateGate` (the sampling cadence that
//  replaced the fixed 1-in-30 stride), `SendWindow` (the bounded set of
//  outstanding sends, which turned out to be the pipeline's real rate limiter)
//  and `SenderMetrics` (the per-stage counters that make a shortfall
//  attributable instead of mysterious).
//
//  All three are pure enough to drive with synthetic times, so these tests
//  assert exact cadence and exact arithmetic rather than sleeping and hoping.
//

import XCTest

@testable import Glasses

@MainActor
final class FrameRateGateTests: XCTestCase {

    /// Runs `frameCount` arrivals spaced at `sourceFPS` through a gate and
    /// returns the arrival indices it admitted.
    private func selectedIndices(
        sourceFPS: Double,
        targetFPS: Double,
        frameCount: Int,
        startTime: TimeInterval = 0
    ) -> [Int] {
        var gate = FrameRateGate(targetFPS: targetFPS)
        var selected: [Int] = []
        for index in 0..<frameCount {
            let now = startTime + Double(index) / sourceFPS
            if gate.shouldSelect(at: now) { selected.append(index) }
        }
        return selected
    }

    // MARK: Cadence

    /// The exact case the old code got wrong: a 24 fps source and a 12 fps
    /// target must admit every other frame, not every 30th (0.8 fps) and not
    /// every third (the aliasing failure a naive elapsed-time check produces).
    func testTwentyFourFPSSourceAtTwelveFPSTargetAdmitsEveryOtherFrame() {
        let selected = selectedIndices(sourceFPS: 24, targetFPS: 12, frameCount: 48)
        XCTAssertEqual(selected, Array(stride(from: 0, to: 48, by: 2)))
    }

    /// The regression guard for the actual bug: whatever the mechanism, one
    /// second of a 24 fps source must not yield 0.8 frames.
    func testDeliveredRateIsNoLongerOnePerThirtyFrames() {
        let selected = selectedIndices(sourceFPS: 24, targetFPS: 12, frameCount: 24)
        XCTAssertEqual(selected.count, 12, "one second of 24 fps must yield ~12 selections, not 1")
    }

    /// DAT's adaptive ladder can settle anywhere in 15...30 fps
    /// (docs/07-PLATFORM-CONSTRAINTS.md). A fixed stride of 2 would deliver
    /// 7.5 fps at the 15 fps floor — below the 10–15 fps target. A rate target
    /// must hold across the whole range.
    func testTargetRateIsHeldAcrossEverySourceRateTheLadderCanProduce() {
        for sourceFPS in [15.0, 24.0, 30.0] {
            let seconds = 60.0
            let selected = selectedIndices(
                sourceFPS: sourceFPS,
                targetFPS: 12,
                frameCount: Int(sourceFPS * seconds)
            )
            let measured = Double(selected.count) / seconds
            // Tight on purpose: the deadline accumulates rather than resetting
            // to "now", so the mean is exact rather than approximate. A loose
            // tolerance here would hide the aliasing this design exists to
            // prevent.
            XCTAssertEqual(
                measured,
                12,
                accuracy: 0.05,
                "a \(sourceFPS) fps source produced \(measured) fps"
            )
        }
    }

    /// The roadmap's V0.7 figure is ~15 fps, which does not divide 24 evenly.
    /// Proving the mechanism holds it exactly is what makes raising
    /// `towerTargetFPS` a one-constant change rather than a redesign — the
    /// shipped 12 is a headroom decision, not a limit of the gate.
    /// `IOS_TOWER_TARGET_FPS`: a whole number in 6...24, or the default 12.
    func testTheTowerTargetCanBeSetForAMeasuredRunAndOtherwiseStays12() {
        typealias Gate = FrameRateGate
        let key = Gate.towerTargetFPSInfoKey, variable = Gate.towerTargetFPSVariable
        XCTAssertEqual(Gate.configuredTowerTargetFPS(), 12, "unset")
        XCTAssertEqual(Gate.configuredTowerTargetFPS(info: [key: ""]), 12, "an empty build setting")
        XCTAssertEqual(Gate.configuredTowerTargetFPS(info: [key: "15"]), 15)
        XCTAssertEqual(Gate.configuredTowerTargetFPS(environment: [variable: "15"]), 15)
        XCTAssertEqual(Gate.configuredTowerTargetFPS(info: [key: "6"]), 6)
        XCTAssertEqual(Gate.configuredTowerTargetFPS(info: [key: "24"]), 24)
        for bad in ["0", "5", "25", "99", "-15", "abc", "15.5", " 15", "1e1"] {
            XCTAssertEqual(Gate.configuredTowerTargetFPS(info: [key: bad]), 12, bad)
            XCTAssertEqual(Gate.configuredTowerTargetFPS(environment: [variable: bad]), 12, bad)
        }
        XCTAssertEqual(Gate.configuredTowerTargetFPS(environment: [variable: "abc"], info: [key: "15"]), 15,
                       "an invalid launch value falls back to the build's")
        XCTAssertEqual(Gate.configuredTowerTargetFPS(environment: [variable: "18"], info: [key: "15"]), 18)
        XCTAssertNotNil(Bundle.main.infoDictionary?[key], "the build's key")
        XCTAssertEqual(Gate.towerTargetFPS, 12, "this build sets no override")
    }

    func testTargetIsHeldAtTheRoadmapFifteenFPS() {
        for sourceFPS in [24.0, 30.0] {
            let seconds = 60.0
            let selected = selectedIndices(
                sourceFPS: sourceFPS,
                targetFPS: 15,
                frameCount: Int(sourceFPS * seconds)
            )
            let measured = Double(selected.count) / seconds
            XCTAssertEqual(measured, 15, accuracy: 0.05, "a \(sourceFPS) fps source produced \(measured) fps")
        }
    }

    /// A gate can only reject. Asking for more than the source provides must
    /// pass everything through rather than inventing or duplicating frames.
    func testSourceSlowerThanTargetAdmitsEveryFrame() {
        let selected = selectedIndices(sourceFPS: 7, targetFPS: 12, frameCount: 21)
        XCTAssertEqual(selected, Array(0..<21))
    }

    /// Jitter around the deadline is why the gate uses a tolerance instead of
    /// a bare elapsed-time comparison. With arrivals landing a hair *early*,
    /// a bare comparison halves the rate.
    func testJitteredArrivalsStillHoldTheTargetRate() {
        var gate = FrameRateGate(targetFPS: 12)
        // 24 fps arrivals, each nudged by a deterministic sub-millisecond
        // amount that straddles the exact 12 fps boundary.
        let jitter: [Double] = [0, -0.0004, 0.0006, -0.0009, 0.0002]
        var count = 0
        for index in 0..<240 {
            let now = Double(index) / 24 + jitter[index % jitter.count]
            if gate.shouldSelect(at: now) { count += 1 }
        }
        let measured = Double(count) / 10.0
        XCTAssertEqual(measured, 12, accuracy: 1.0, "jitter dropped the rate to \(measured) fps")
    }

    // MARK: Freshness

    /// After a stall the gate must resume at the target rate, not admit a
    /// burst of catch-up frames — freshness over completeness.
    func testStallDoesNotProduceACatchUpBurst() {
        var gate = FrameRateGate(targetFPS: 12)
        XCTAssertTrue(gate.shouldSelect(at: 0))

        // Nothing arrives for two seconds (24 missed deadlines).
        XCTAssertTrue(gate.shouldSelect(at: 2.0), "the first frame after a stall is fresh; admit it")

        // The backlog then drains in one main-actor turn: many frames with
        // near-identical timestamps. Exactly one may pass.
        var admittedFromBacklog = 0
        for offset in 0..<20 {
            if gate.shouldSelect(at: 2.0 + Double(offset) * 0.0001) { admittedFromBacklog += 1 }
        }
        XCTAssertEqual(admittedFromBacklog, 0, "a drained backlog must not all pass the gate")
    }

    /// The property that makes the gate safe to consult after the main-actor
    /// hop: because eligibility is measured in wall time, a congested main
    /// actor cannot cause more expensive work than the target rate, no matter
    /// how many frames are queued behind it.
    func testCongestionCannotExceedTheTargetRate() {
        var gate = FrameRateGate(targetFPS: 12)
        var count = 0
        // 240 frames all delivered within a single 100 ms window.
        for index in 0..<240 {
            if gate.shouldSelect(at: Double(index) * 0.1 / 240) { count += 1 }
        }
        XCTAssertLessThanOrEqual(count, 3, "100 ms of arrivals admitted \(count) frames at a 12 fps target")
    }

    // MARK: Lifecycle

    func testResetMakesTheNextFrameTheFirstOfASession() {
        var gate = FrameRateGate(targetFPS: 12)
        XCTAssertTrue(gate.shouldSelect(at: 100))
        XCTAssertFalse(gate.shouldSelect(at: 100.01))

        gate.reset()
        XCTAssertTrue(
            gate.shouldSelect(at: 100.02),
            "after reset the next frame is the first of a new session and must be admitted"
        )
    }

    func testNonPositiveTargetIsClampedRatherThanStallingThePipeline() {
        var gate = FrameRateGate(targetFPS: 0)
        XCTAssertGreaterThan(gate.targetFPS, 0)
        XCTAssertTrue(gate.shouldSelect(at: 0))
        XCTAssertTrue(gate.interval.isFinite, "an infinite interval would admit one frame and stall forever")
    }

    /// Guards the shipped constant against the band this change is being
    /// measured against (10–15 fps delivered), and against the rate it
    /// replaced. Not a tautology in one direction that matters: if someone
    /// re-derives the target from a source-rate stride and lands back near
    /// 0.8, this fails.
    func testConfiguredTowerTargetSitsInsideTheMeasurementBand() {
        XCTAssertGreaterThanOrEqual(FrameRateGate.towerTargetFPS, 10)
        XCTAssertLessThanOrEqual(FrameRateGate.towerTargetFPS, 15)

        // The gate must actually deliver the configured target from the
        // configured 24 fps stream, not merely store a plausible number.
        let seconds = 60.0
        let selected = selectedIndices(
            sourceFPS: 24,
            targetFPS: FrameRateGate.towerTargetFPS,
            frameCount: Int(24 * seconds)
        )
        XCTAssertEqual(
            Double(selected.count) / seconds,
            FrameRateGate.towerTargetFPS,
            accuracy: 0.05
        )
    }
}

@MainActor
final class SenderMetricsTests: XCTestCase {

    func testBeginClearsThePreviousSession() {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordCapture(sequence: 1)
        metrics.recordSelection()
        metrics.recordSendAttempt(wireBytes: 100)
        metrics.recordSendSuccess()
        XCTAssertEqual(metrics.currentSnapshot.framesCaptured, 1)

        metrics.begin()
        let snapshot = metrics.currentSnapshot
        XCTAssertEqual(snapshot.framesCaptured, 0)
        XCTAssertEqual(snapshot.framesSelected, 0)
        XCTAssertEqual(snapshot.sendSuccesses, 0)
        XCTAssertEqual(snapshot.wireBytes, 0)
    }

    func testFinishPublishesTheFinalCountersRatherThanClearingThem() {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordCapture(sequence: 1)
        metrics.recordSelection()
        metrics.finish()

        XCTAssertEqual(metrics.snapshot.framesCaptured, 1, "Stop must leave the session's numbers readable")
        XCTAssertEqual(metrics.snapshot.framesSelected, 1)
    }

    /// Recording must not publish on every increment — at the capture rate
    /// that would invalidate the SwiftUI view tree 24 times a second purely to
    /// show a counter.
    func testRecordingIsRateLimitedRatherThanPublishingPerFrame() {
        let metrics = SenderMetrics()
        metrics.begin()
        let publishedAtBegin = metrics.snapshot.framesCaptured

        for sequence in 1...200 {
            metrics.recordCapture(sequence: sequence)
        }

        XCTAssertEqual(
            metrics.snapshot.framesCaptured,
            publishedAtBegin,
            "200 increments inside one publish interval must not republish"
        )
        XCTAssertEqual(metrics.currentSnapshot.framesCaptured, 200, "the counters themselves must still be exact")
    }

    // MARK: Session clock

    /// `startCameraSession()` calls `begin()` seconds before DAT delivers a
    /// frame — creating the session, starting it, adding the camera, starting
    /// the stream. Counting that dead time would understate every rate and
    /// send someone hunting a shortfall that does not exist.
    func testClockDoesNotRunBeforeTheFirstFrameArrives() async throws {
        let metrics = SenderMetrics()
        metrics.begin()
        // Deliberately much longer than the post-frame wait below, so the two
        // hypotheses (clock starts at begin vs. at first frame) are separated
        // by ~400 ms and the assertion cannot flake on a loaded machine.
        try? await Task.sleep(nanoseconds: 500_000_000)

        XCTAssertEqual(metrics.currentSnapshot.duration, 0, "a session with no frames has no measurable duration")
        XCTAssertNil(metrics.currentSnapshot.captureFPS)

        metrics.recordCapture(sequence: 1)
        try? await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertGreaterThan(metrics.currentSnapshot.duration, 0, "the first frame starts the clock")
        XCTAssertLessThan(
            metrics.currentSnapshot.duration,
            0.3,
            "the clock must start at the first frame, not at begin()"
        )
    }

    /// Frames still in flight at Stop, and `frame_result` replies for them,
    /// land after `finish()`. If the clock kept running, every displayed rate
    /// would visibly decay once the session ended.
    func testFinishFreezesTheElapsedTimeSoFinalRatesStopMoving() async throws {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordCapture(sequence: 1)
        metrics.recordSelection()
        try? await Task.sleep(nanoseconds: 60_000_000)
        metrics.finish()

        let frozen = metrics.snapshot.duration
        XCTAssertGreaterThan(frozen, 0)

        try? await Task.sleep(nanoseconds: 120_000_000)
        // A late reply for a frame that was in flight at Stop.
        metrics.recordFrameResult()

        XCTAssertEqual(metrics.currentSnapshot.duration, frozen, accuracy: 0.0001)
        XCTAssertEqual(metrics.currentSnapshot.frameResults, 1, "the late reply is still counted — it really arrived")
    }

    func testBeginRestartsTheClockAfterAFinishedSession() async throws {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordCapture(sequence: 1)
        try? await Task.sleep(nanoseconds: 60_000_000)
        metrics.finish()
        XCTAssertGreaterThan(metrics.snapshot.duration, 0)

        metrics.begin()
        XCTAssertEqual(metrics.currentSnapshot.duration, 0, "a new session must not inherit the old clock")
        metrics.recordCapture(sequence: 1)
        try? await Task.sleep(nanoseconds: 30_000_000)
        XCTAssertGreaterThan(metrics.currentSnapshot.duration, 0, "and must start ticking again")
    }

    // MARK: Rates

    func testRatesAreNilUntilThereIsEnoughElapsedTimeToDivideBy() {
        var snapshot = SenderMetricsSnapshot()
        snapshot.framesCaptured = 3
        snapshot.sendSuccesses = 2
        snapshot.duration = 0.05

        XCTAssertNil(snapshot.captureFPS, "3 frames in 50 ms is not a measured rate")
        XCTAssertNil(snapshot.successfulSendFPS)
    }

    func testRatesAreComputedOverTheSessionDuration() {
        var snapshot = SenderMetricsSnapshot()
        snapshot.duration = 10
        snapshot.framesCaptured = 240
        snapshot.framesSelected = 120
        snapshot.sendAttempts = 120
        snapshot.sendSuccesses = 118
        snapshot.frameResults = 117
        snapshot.wireBytes = 2_048_000

        XCTAssertEqual(snapshot.captureFPS ?? 0, 24, accuracy: 0.001)
        XCTAssertEqual(snapshot.selectedFPS ?? 0, 12, accuracy: 0.001)
        XCTAssertEqual(snapshot.sendAttemptFPS ?? 0, 12, accuracy: 0.001)
        XCTAssertEqual(snapshot.successfulSendFPS ?? 0, 11.8, accuracy: 0.001)
        XCTAssertEqual(snapshot.towerResultFPS ?? 0, 11.7, accuracy: 0.001)
        XCTAssertEqual(snapshot.wireBytesPerSecond ?? 0, 204_800, accuracy: 0.001)
    }

    /// The single number that makes the before/after comparison legible: the
    /// 0.8 fps baseline was delivering ~1 frame in 30.
    func testDeliveredFractionReproducesTheBaselineRatio() {
        var baseline = SenderMetricsSnapshot()
        baseline.duration = 79
        baseline.framesCaptured = 1860
        baseline.sendSuccesses = 63
        XCTAssertEqual(baseline.deliveredFraction ?? 0, 1.0 / 29.5, accuracy: 0.002)

        var target = SenderMetricsSnapshot()
        target.duration = 79
        target.framesCaptured = 1896
        target.sendSuccesses = 948
        XCTAssertEqual(target.deliveredFraction ?? 0, 0.5, accuracy: 0.001)
    }

    func testEncodeTimingTracksAverageAndWorstCase() {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordEncode(seconds: 0.004)
        metrics.recordEncode(seconds: 0.010)
        metrics.recordEncode(seconds: 0.006)

        let snapshot = metrics.currentSnapshot
        XCTAssertEqual(snapshot.framesEncoded, 3)
        XCTAssertEqual(snapshot.encodeMsAverage ?? 0, 20.0 / 3, accuracy: 0.001)
        XCTAssertEqual(snapshot.encodeMsMax ?? 0, 10, accuracy: 0.001)
    }

    func testEncodeTimingIsNilRatherThanZeroBeforeAnyEncode() {
        XCTAssertNil(SenderMetricsSnapshot().encodeMsAverage)
        XCTAssertNil(SenderMetricsSnapshot().encodeMsMax)
        XCTAssertNil(SenderMetricsSnapshot().deliveredFraction)
    }

    // MARK: Invariants

    /// Sequence numbers are DAT callback ordinals. If this ever stops holding,
    /// the Tower's seq-gap arithmetic — the evidence that found this bug — is
    /// no longer readable.
    func testSequenceInvariantHoldsWhenEveryCallbackIncrementsTheSequence() {
        let metrics = SenderMetrics()
        metrics.begin()
        for sequence in 1...50 {
            metrics.recordCapture(sequence: sequence)
        }
        XCTAssertTrue(metrics.currentSnapshot.sequenceInvariantHolds)
    }

    func testSequenceInvariantFailsWhenSequencesSkip() {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordCapture(sequence: 1)
        metrics.recordCapture(sequence: 3)
        XCTAssertFalse(
            metrics.currentSnapshot.sequenceInvariantHolds,
            "a gap between callbacks and sequences must be visible, not silently averaged away"
        )
    }

    /// Every selected frame must reach a terminal bucket or be in flight;
    /// anything else is backlog. Each of the six fates must clear the count.
    func testBacklogClearsForEveryTerminalOutcome() {
        let metrics = SenderMetrics()
        metrics.begin()

        for _ in 0..<10 { metrics.recordSelection() }
        XCTAssertEqual(metrics.currentSnapshot.framesUnaccounted, 10, "nothing resolved yet")

        metrics.recordDecodeFailure()
        metrics.recordEncodeFailure()
        metrics.recordSendWindowDrop()
        metrics.recordSessionGateDrop()
        for _ in 0..<6 {
            metrics.recordSendAttempt(wireBytes: 20_000)
        }
        for _ in 0..<5 { metrics.recordSendSuccess() }
        // The sixth send is still in flight — attempted, so not backlog.
        XCTAssertEqual(
            metrics.currentSnapshot.framesUnaccounted,
            0,
            "an outstanding send is accounted for, not backlog"
        )

        metrics.recordSendFailure()
        XCTAssertEqual(metrics.currentSnapshot.framesUnaccounted, 0, "and once it completes")
    }

    /// A send whose socket is torn down mid-flight completes on neither the
    /// success nor the failure path. Without its own bucket it would count as
    /// permanently in flight, quietly masking real backlog from then on.
    func testSendAbandonedByTeardownIsATerminalOutcome() {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordSelection()
        metrics.recordSendAttempt(wireBytes: 20_000)
        XCTAssertEqual(metrics.currentSnapshot.framesUnaccounted, 0)

        metrics.recordSendAbandoned()
        XCTAssertEqual(metrics.currentSnapshot.framesUnaccounted, 0)
        XCTAssertEqual(metrics.currentSnapshot.sendSuccesses, 0, "it must not be counted as delivered")
    }

    /// The signature this metric exists to expose: selections outrunning
    /// outcomes means frames are queueing rather than being dropped.
    func testBacklogCountsFramesThatWentNowhere() {
        let metrics = SenderMetrics()
        metrics.begin()
        for _ in 0..<5 { metrics.recordSelection() }
        metrics.recordSendAttempt(wireBytes: 1)
        metrics.recordSendSuccess()

        XCTAssertEqual(
            metrics.currentSnapshot.framesUnaccounted,
            4,
            "four selected frames reached no outcome and must be visible as backlog"
        )
    }
}

// MARK: - Send window

/// The send window turned out to be the pipeline's rate limiter rather than a
/// passive memory guard, so it is tested as one: occupancy, slot lifetime,
/// token safety across a teardown, and the arithmetic that ties capacity and
/// latency to a frame rate.
///
/// Every case drives synthetic times, so the assertions are exact rather than
/// dependent on how fast the test machine happens to be.
@MainActor
final class SendWindowTests: XCTestCase {

    private func makeWindow(capacity: Int = 4, stallTimeout: TimeInterval = 2) -> SendWindow {
        SendWindow(capacity: capacity, stallTimeout: stallTimeout)
    }

    // MARK: Occupancy

    func testReservesUpToCapacityAndThenRefuses() {
        var window = makeWindow(capacity: 3)
        XCTAssertNotNil(window.reserve(at: 0))
        XCTAssertNotNil(window.reserve(at: 0))
        XCTAssertNotNil(window.reserve(at: 0))
        XCTAssertEqual(window.inFlight, 3)
        XCTAssertTrue(window.isFull)
        XCTAssertNil(window.reserve(at: 0), "a full window must refuse rather than grow")
        XCTAssertEqual(window.inFlight, 3, "a refused reservation must not occupy a slot")
    }

    func testReleasingReopensExactlyOneSlot() throws {
        var window = makeWindow(capacity: 2)
        let first = try XCTUnwrap(window.reserve(at: 0))
        _ = window.reserve(at: 0)
        XCTAssertNil(window.reserve(at: 0))

        XCTAssertNotNil(window.release(first, at: 0))
        XCTAssertEqual(window.inFlight, 1)
        XCTAssertNotNil(window.reserve(at: 0), "the freed slot must be reusable")
        XCTAssertNil(window.reserve(at: 0), "and only that one slot")
    }

    /// The number the whole diagnosis rests on: a slot's lifetime is what the
    /// achievable send rate is divided by.
    func testReleaseReportsHowLongTheSlotWasHeld() throws {
        var window = makeWindow()
        let token = try XCTUnwrap(window.reserve(at: 10))
        let lifetime = window.release(token, at: 10.29)
        XCTAssertEqual(try XCTUnwrap(lifetime), 0.29, accuracy: 0.0001)
    }

    func testUnknownTokenReleasesNothing() {
        var window = makeWindow(capacity: 1)
        _ = window.reserve(at: 0)
        XCTAssertNil(window.release(9999, at: 1), "an unknown token has no slot to report on")
        XCTAssertEqual(window.inFlight, 1, "and must not free somebody else's slot")
    }

    // MARK: Teardown safety

    func testResetReopensTheWholeWindow() {
        var window = makeWindow(capacity: 2)
        _ = window.reserve(at: 0)
        _ = window.reserve(at: 0)
        window.reset()
        XCTAssertEqual(window.inFlight, 0)
        XCTAssertFalse(window.isFull)
    }

    /// The leak this design exists to prevent, from the other direction: a
    /// completion handler for a torn-down socket arriving *after* the next
    /// connection has started sending must not credit a slot it does not own.
    /// If tokens were rewound by `reset()`, the stale completion would release
    /// the new connection's frame and the window would silently widen.
    func testStaleTokenFromBeforeAResetCannotCreditASlotAfterIt() throws {
        var window = makeWindow(capacity: 1)
        let staleToken = try XCTUnwrap(window.reserve(at: 0))

        window.reset()

        let freshToken = try XCTUnwrap(window.reserve(at: 1))
        XCTAssertNotEqual(staleToken, freshToken, "tokens must not be reused across a teardown")

        XCTAssertNil(window.release(staleToken, at: 2), "the stale completion owns nothing")
        XCTAssertEqual(window.inFlight, 1, "the live send must still hold its slot")
        XCTAssertNil(window.reserve(at: 2), "the window must still be full")
    }

    // MARK: Stall detection

    func testAYoungFullWindowIsNotStalled() {
        var window = makeWindow(capacity: 2, stallTimeout: 2)
        _ = window.reserve(at: 0)
        _ = window.reserve(at: 0)
        XCTAssertFalse(window.isStalled(at: 1.9))
    }

    func testAFullWindowThatStopsDrainingIsStalled() {
        var window = makeWindow(capacity: 2, stallTimeout: 2)
        _ = window.reserve(at: 0)
        _ = window.reserve(at: 0)
        XCTAssertTrue(window.isStalled(at: 2.0))
        XCTAssertTrue(window.isStalled(at: 52.0), "the 52-second baseline outlier is unambiguously a stall")
    }

    /// A window with room is still admitting frames, so nothing is blocked and
    /// tearing the connection down would cost more than it recovered.
    func testAnAgedButNonFullWindowIsNotStalled() {
        var window = makeWindow(capacity: 3, stallTimeout: 2)
        _ = window.reserve(at: 0)
        XCTAssertFalse(window.isStalled(at: 100))
    }

    /// URLSession does not document send-completion ordering, so stall
    /// detection must key off the genuinely oldest reservation rather than
    /// assuming the first one reserved is the first one released.
    func testOldestAgeSurvivesOutOfOrderCompletion() throws {
        var window = makeWindow(capacity: 3, stallTimeout: 2)
        let oldest = try XCTUnwrap(window.reserve(at: 0))
        let middle = try XCTUnwrap(window.reserve(at: 1))
        _ = window.reserve(at: 2)

        XCTAssertNotNil(window.release(middle, at: 3), "a later send may complete first")
        XCTAssertEqual(try XCTUnwrap(window.oldestAge(at: 3)), 3, accuracy: 0.0001)

        XCTAssertNotNil(window.release(oldest, at: 3))
        XCTAssertEqual(try XCTUnwrap(window.oldestAge(at: 3)), 1, accuracy: 0.0001)
    }

    func testOldestAgeIsNilWhenNothingIsInFlight() {
        let window = makeWindow()
        XCTAssertNil(window.oldestAge(at: 5))
        XCTAssertFalse(window.isStalled(at: 5))
    }

    // MARK: Capacity arithmetic

    /// Capacity is derived from a latency budget, and this is that derivation.
    func testCapacityIsTheFrameCountThatFitsInTheLatencyBudget() {
        XCTAssertEqual(SendWindow.capacity(forTargetFPS: 12, latencyBudget: 1.0 / 3.0), 4)
        XCTAssertEqual(SendWindow.capacity(forTargetFPS: 12, latencyBudget: 0.5), 6)
        XCTAssertEqual(SendWindow.capacity(forTargetFPS: 24, latencyBudget: 0.25), 6)
    }

    func testCapacityIsNeverZeroHoweverItIsConfigured() {
        XCTAssertEqual(SendWindow.capacity(forTargetFPS: 12, latencyBudget: 0), 1)
        XCTAssertEqual(SendWindow.capacity(forTargetFPS: 0, latencyBudget: 1), 1)
        XCTAssertEqual(SendWindow.capacity(forTargetFPS: -5, latencyBudget: -5), 1)
        XCTAssertEqual(
            SendWindow.capacity(forTargetFPS: 1, latencyBudget: 0.1),
            1,
            "rounding to zero must still admit frames"
        )
    }

    /// A zero capacity would stall the pipeline permanently and a non-positive
    /// stall timeout would tear the connection down on the first full window.
    func testDegenerateConfigurationIsClampedRatherThanTrapping() {
        var window = SendWindow(capacity: 0, stallTimeout: 0)
        XCTAssertEqual(window.capacity, 1)
        XCTAssertNotNil(window.reserve(at: 0), "a clamped window must still admit one frame")
        XCTAssertFalse(window.isStalled(at: 1_000_000), "a non-positive timeout must not mean always-stalled")
    }

    // MARK: The regression this exists for

    /// The physical baseline, as arithmetic. A capacity of 2 against the
    /// measured slot lifetimes produces exactly the rates that run reported —
    /// which is what identified the window, rather than the encoder or the
    /// gate, as the constraint. The same latencies with the latency-budgeted
    /// capacity clear the 12 fps target.
    ///
    /// This is a statement about the design, not about the device: it will
    /// keep holding on any machine, because it divides two constants.
    func testTheBaselineRatesAreExplainedByCapacityOverSlotLifetime() {
        func achievableFPS(capacity: Int, slotLifetime: TimeInterval) -> Double {
            Double(capacity) / slotLifetime
        }

        // What the device measured: ~7 fps early in the run, ~3.4 fps by the end.
        XCTAssertEqual(achievableFPS(capacity: 2, slotLifetime: 0.29), 6.9, accuracy: 0.1)
        XCTAssertEqual(achievableFPS(capacity: 2, slotLifetime: 0.59), 3.4, accuracy: 0.1)

        // The same link, with capacity derived from the latency budget.
        let sized = SendWindow.capacity(
            forTargetFPS: FrameRateGate.towerTargetFPS,
            latencyBudget: TowerClient.outboundLatencyBudget
        )
        XCTAssertGreaterThanOrEqual(
            achievableFPS(capacity: sized, slotLifetime: 0.29),
            FrameRateGate.towerTargetFPS,
            "the sized window must clear the target at the latency it was budgeted for"
        )
    }
}

// MARK: - Slot-lifetime instrumentation

/// The counters added so a send-rate shortfall can be attributed to the
/// network or to the main actor instead of guessed at.
@MainActor
final class SlotTimingMetricsTests: XCTestCase {

    func testTimingsAreNilRatherThanZeroBeforeAnySend() {
        let metrics = SenderMetrics()
        metrics.begin()
        let snapshot = metrics.currentSnapshot
        XCTAssertNil(snapshot.sendLatencyMsAverage)
        XCTAssertNil(snapshot.slotLifetimeMsAverage)
        XCTAssertNil(snapshot.completionHopMsAverage)
        XCTAssertNil(snapshot.windowLimitedFPS(capacity: 4))
    }

    func testAverageAndWorstCaseAreTrackedSeparatelyForBothSpans() {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordSlotTiming(sendLatency: 0.100, slotLifetime: 0.110)
        metrics.recordSlotTiming(sendLatency: 0.300, slotLifetime: 0.390)

        let snapshot = metrics.currentSnapshot
        XCTAssertEqual(snapshot.slotSamples, 2)
        XCTAssertEqual(try XCTUnwrap(snapshot.sendLatencyMsAverage), 200, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(snapshot.sendLatencyMsMax), 300, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(snapshot.slotLifetimeMsAverage), 250, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(snapshot.slotLifetimeMsMax), 390, accuracy: 0.001)
    }

    /// The whole point of sampling the two spans separately: the difference is
    /// main-actor congestion, and it is what distinguishes "the network is
    /// slow" from "we are too busy to notice the network finished".
    func testTheHopIsTheDifferenceBetweenTheTwoSpans() throws {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordSlotTiming(sendLatency: 0.100, slotLifetime: 0.150)
        metrics.recordSlotTiming(sendLatency: 0.200, slotLifetime: 0.250)

        XCTAssertEqual(try XCTUnwrap(metrics.currentSnapshot.completionHopMsAverage), 50, accuracy: 0.001)
    }

    /// The two spans are sampled from different execution contexts, so a
    /// lifetime shorter than its own latency is a clock artefact rather than a
    /// measurement. Clamping keeps the derived hop from going negative.
    func testALifetimeShorterThanItsLatencyIsClampedRatherThanReportedNegative() throws {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordSlotTiming(sendLatency: 0.200, slotLifetime: 0.150)

        let snapshot = metrics.currentSnapshot
        XCTAssertEqual(try XCTUnwrap(snapshot.slotLifetimeMsAverage), 200, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(snapshot.completionHopMsAverage), 0, accuracy: 0.001)
    }

    func testNegativeLatencyIsClampedToZero() throws {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordSlotTiming(sendLatency: -0.05, slotLifetime: 0.05)
        XCTAssertEqual(try XCTUnwrap(metrics.currentSnapshot.sendLatencyMsAverage), 0, accuracy: 0.001)
    }

    /// The diagnosis in one number: at the measured slot lifetime, this is the
    /// most the window can deliver however many frames the gate selects.
    func testWindowLimitedRateIsCapacityOverSlotLifetime() throws {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordSlotTiming(sendLatency: 0.28, slotLifetime: 0.29)

        XCTAssertEqual(
            try XCTUnwrap(metrics.currentSnapshot.windowLimitedFPS(capacity: 2)),
            6.9,
            accuracy: 0.1,
            "the baseline window and the baseline latency must reproduce the baseline rate"
        )
        XCTAssertGreaterThanOrEqual(
            try XCTUnwrap(metrics.currentSnapshot.windowLimitedFPS(capacity: 4)),
            FrameRateGate.towerTargetFPS
        )
    }

    func testStallRecoveriesAreCounted() {
        let metrics = SenderMetrics()
        metrics.begin()
        XCTAssertEqual(metrics.currentSnapshot.stallRecoveries, 0)
        metrics.recordStallRecovery()
        metrics.recordStallRecovery()
        XCTAssertEqual(metrics.currentSnapshot.stallRecoveries, 2)
    }

    /// A stall recovery is rare and is the event the instrumentation exists to
    /// catch, so unlike the per-frame counters it publishes immediately rather
    /// than waiting for the next 2 Hz interval.
    func testStallRecoveryPublishesWithoutWaitingForTheInterval() {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordStallRecovery()
        XCTAssertEqual(metrics.snapshot.stallRecoveries, 1)
    }

    /// Slot timings are pure instrumentation: they must not disturb the
    /// accounting that says whether frames are queueing.
    func testSlotTimingDoesNotAffectTheBacklogInvariant() {
        let metrics = SenderMetrics()
        metrics.begin()
        metrics.recordSelection()
        metrics.recordSendAttempt(wireBytes: 100)
        metrics.recordSlotTiming(sendLatency: 0.1, slotLifetime: 0.1)
        metrics.recordSendSuccess()

        XCTAssertEqual(metrics.currentSnapshot.framesUnaccounted, 0)
    }
}
