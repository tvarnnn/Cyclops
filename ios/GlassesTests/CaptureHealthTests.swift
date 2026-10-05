//
//  CaptureHealthTests.swift
//  GlassesTests
//
//  U2-D0 "Capture health" (spec U2-D0-CAPTURE-HEALTH-SWIFT-SPEC-20261005 §3,
//  tests 1-5): the panel's arithmetic on scripted samples and instants, the
//  two decodes it adds, and its honesty about absent and old figures. No
//  Tower; the client's half is in `TowerWorldBuilderClientTests`.
//

import XCTest

@testable import Glasses

final class CaptureHealthTests: XCTestCase {

    private let t0 = ContinuousClock.now
    private func at(_ seconds: Double) -> ContinuousClock.Instant { t0 + .milliseconds(Int(seconds * 1000)) }

    private func sample(keyframes: Int? = nil, restarts: Int? = nil, recovery: WorldRecoveryReport? = nil,
                        lag: WorldMapLag? = nil) -> CaptureHealthSample {
        CaptureHealthSample(keyframeCount: keyframes, trackingRestarts: restarts, recovery: recovery, mapLag: lag)
    }

    private func readout(_ history: CaptureHealthHistory, _ seconds: Double, linked: Bool = true,
                         capturing: Bool = true) -> CaptureHealthReadout {
        history.readout(now: at(seconds), isLinked: linked, isCapturing: capturing)
    }

    // MARK: 1. Pace and Stalled

    /// A counter 0 → 30 over 60 s reads 30 keyframes/min; a 6 s pause shows
    /// Stalled; resuming clears it.
    func testTheRateIsTheRiseOverTheLastMinuteAndAPauseIsAStall() {
        var history = CaptureHealthHistory()
        for second in stride(from: 0, through: 60, by: 2) {
            history.record(sample(keyframes: second / 2), at: at(Double(second)))
        }
        XCTAssertEqual(readout(history, 60).pace, "30 keyframes/min")
        XCTAssertNil(readout(history, 60).stalled)

        // Heartbeats, no new keyframe, for 6 s.
        for second in [62.0, 64, 66] { history.record(sample(keyframes: 30), at: at(second)) }
        XCTAssertEqual(readout(history, 64).stalled, nil, "4 s is not a stall")
        XCTAssertEqual(readout(history, 66).stalled, "Stalled — no new keyframe for 6 s")
        XCTAssertNil(readout(history, 66, capturing: false).stalled, "only while capturing")
        // The window slides: the minute to 66 s rose from 3 to 30.
        XCTAssertEqual(readout(history, 66).pace, "27 keyframes/min")

        history.record(sample(keyframes: 31), at: at(67))
        XCTAssertNil(readout(history, 67).stalled, "resuming clears it")
        XCTAssertEqual(readout(history, 67).pace, "28 keyframes/min")
    }

    /// Watched for less than a minute: the rise since the first sample, per
    /// minute -- and no rate at all over less than 10 s. A counter that goes
    /// down is a new walk.
    func testAShortWatchIsARateSinceItsStartAndANewWalkStartsAgain() {
        var history = CaptureHealthHistory()
        history.record(sample(keyframes: 4), at: at(0))
        XCTAssertEqual(readout(history, 5).pace, "— keyframes/min", "5 s is too short for a rate")
        history.record(sample(keyframes: 9), at: at(10))
        XCTAssertEqual(readout(history, 10).pace, "30 keyframes/min", "5 in 10 s")
        history.record(sample(keyframes: 1), at: at(12))
        XCTAssertEqual(readout(history, 12).pace, "— keyframes/min", "a new walk, from its own first sample")
        XCTAssertNil(readout(history, 12).stalled, "a new walk has not stalled")
        history.record(sample(keyframes: 8), at: at(22))
        XCTAssertEqual(readout(history, 22).pace, "42 keyframes/min", "7 in 10 s")
    }

    // MARK: 2. Breaks in the last 30 s

    /// Restarts +2 at 10 s and +1 at 35 s: 3 at 36 s, 1 at 41 s. Before the
    /// panel has watched 30 s, a rise it saw is a floor and no rise is
    /// unknown, never 0.
    func testBreaksAreTheRiseOverTheLast30SecondsAndTheWindowSlides() {
        var history = CaptureHealthHistory()
        var restarts = 0
        var seen: [Int: CaptureHealthReadout] = [:]
        for second in 0...41 {
            if second == 10 { restarts += 2 }
            if second == 35 { restarts += 1 }
            history.record(sample(keyframes: second, restarts: restarts), at: at(Double(second)))
            seen[second] = readout(history, Double(second))
        }
        XCTAssertEqual(seen[5]?.breaks, "Breaks in the last 30 s: —", "5 s watched, nothing seen: unknown")
        XCTAssertEqual(seen[5]?.breaksSpoken, "Breaks in the last 30 seconds: unknown")
        XCTAssertNil(seen[5]?.breaksCount)
        XCTAssertEqual(seen[12]?.breaks, "Breaks in the last 30 s: ≥ 2", "seen, but the window is not all watched")
        XCTAssertEqual(seen[12]?.breaksSpoken, "Breaks in the last 30 seconds: at least 2")
        XCTAssertEqual(seen[30]?.breaks, "Breaks in the last 30 s: 2")
        XCTAssertEqual(seen[36]?.breaks, "Breaks in the last 30 s: 3")
        XCTAssertEqual(seen[41]?.breaks, "Breaks in the last 30 s: 1")
        XCTAssertEqual(seen[41]?.breaksSpoken, "Breaks in the last 30 seconds: 1")
        XCTAssertEqual(seen[41]?.breaksCount, 1)
        XCTAssertFalse(seen.values.contains { $0.breaks.contains("lost") }, "never \"Tracking: lost\"")
    }

    // MARK: 3. Look-back

    func testTheLookBackRowShowsTheCountsAndNeverAZeroItWasNotSent() throws {
        let block: [String: Any] = [
            "state": "recovered", "episode": 3, "prompts_enabled": true, "prompt": NSNull(),
            "counts": ["episodes": 3, "recovered": 2, "recovered_after_prompt": 1, "timed_out": 1,
                       "prompts": 1, "withheld_by_limiter": 0, "withheld_disabled": 0],
        ]
        let json = try JSONSerialization.jsonObject(with: JSONSerialization.data(withJSONObject: block))
        let report = try XCTUnwrap(WorldRecoveryReport(json: json))
        XCTAssertEqual(report.counts, WorldRecoveryCounts(recovered: 2, timedOut: 1))

        var history = CaptureHealthHistory()
        history.record(sample(keyframes: 5, restarts: 0, recovery: report), at: at(0))
        XCTAssertEqual(readout(history, 0).lookBackLine, "Linked back to what you saw before")
        XCTAssertEqual(readout(history, 0).lookBackCounts, "linked back 2 · could not link 1")

        var withoutCounts = block
        withoutCounts["counts"] = nil
        let bare = try XCTUnwrap(WorldRecoveryReport(json: withoutCounts))
        XCTAssertNil(bare.counts)
        XCTAssertNil(WorldRecoveryReport(json: block.merging(["counts": ["recovered": 2]]) { $1 })?.counts,
                     "half a block is not a count")
        history.record(sample(keyframes: 5, restarts: 0, recovery: bare), at: at(1))
        XCTAssertEqual(readout(history, 1).lookBackCounts, "linked back — · could not link —")
        history.record(sample(keyframes: 5, restarts: 0), at: at(2))
        XCTAssertEqual(readout(history, 2).lookBackCounts, "Look-back: —")
        XCTAssertNil(readout(history, 2).lookBackLine)
    }

    // MARK: 4. Map lag

    func testTheMapLagIsTheDifferenceAndHiddenWhenNoneOrUnknown() {
        XCTAssertEqual(WorldMapLag(geometry: ["built_from_keyframes": 117, "keyframes_now": 120])?.behind, 3)
        XCTAssertNil(WorldMapLag(geometry: ["built_from_keyframes": NSNull(), "keyframes_now": 4]))
        XCTAssertNil(WorldMapLag(geometry: ["keyframes_now": 4]))
        XCTAssertNil(WorldMapLag(geometry: nil))
        XCTAssertEqual(WorldMapLag(builtFromKeyframes: 12, keyframesNow: 10).behind, 0, "clamped at 0")

        var history = CaptureHealthHistory()
        history.record(sample(keyframes: 120, lag: WorldMapLag(builtFromKeyframes: 117, keyframesNow: 120)), at: at(0))
        XCTAssertEqual(readout(history, 0).mapLag, "Map: 3 keyframes behind")
        history.record(sample(keyframes: 120, lag: WorldMapLag(builtFromKeyframes: 119, keyframesNow: 120)), at: at(1))
        XCTAssertEqual(readout(history, 1).mapLag, "Map: 1 keyframe behind")
        history.record(sample(keyframes: 120, lag: WorldMapLag(builtFromKeyframes: 120, keyframesNow: 120)), at: at(2))
        XCTAssertNil(readout(history, 2).mapLag, "equal: hidden")
        history.record(sample(keyframes: 120), at: at(3))
        XCTAssertNil(readout(history, 3).mapLag, "missing: hidden")
    }

    // MARK: 5. No link, and old figures

    func testWithoutTheLinkOrWithOldReportsEveryFigureReadsUnknown() {
        var history = CaptureHealthHistory()
        let report = WorldRecoveryReport(state: .recovered, episode: 1, counts: WorldRecoveryCounts(recovered: 1, timedOut: 0))
        for second in stride(from: 0.0, through: 10, by: 2) {
            history.record(sample(keyframes: Int(second), restarts: 1, recovery: report,
                                  lag: WorldMapLag(builtFromKeyframes: 0, keyframesNow: 5)), at: at(second))
        }
        let live = readout(history, 10)
        XCTAssertEqual(live.link, "Tower: connected")
        XCTAssertEqual(live.pace, "60 keyframes/min")

        func assertUnknown(_ r: CaptureHealthReadout, _ why: String, line: UInt = #line) {
            XCTAssertEqual(r.pace, "— keyframes/min", why, line: line)
            XCTAssertNil(r.stalled, why, line: line)
            XCTAssertEqual(r.breaks, "Breaks in the last 30 s: —", why, line: line)
            XCTAssertEqual(r.lookBackCounts, "Look-back: —", why, line: line)
            XCTAssertNil(r.lookBackLine, why, line: line)
            XCTAssertEqual(r.mapLag, "Map: —", why, line: line)
            XCTAssertNil(r.breaksCount, why, line: line)
        }
        let unlinked = readout(history, 10, linked: false)
        XCTAssertEqual(unlinked.link, "Tower: not connected")
        assertUnknown(unlinked, "no link")
        XCTAssertEqual(readout(history, 15).pace, "40 keyframes/min", "5 s since the last report is still live")
        assertUnknown(readout(history, 15.5), "a report older than 5 s is not live")
        assertUnknown(CaptureHealthHistory().readout(now: at(0), isLinked: true, isCapturing: true), "no report yet")

        // A report that says nothing about a live walk: unknown figures, and
        // no stall claimed from a counter it did not send.
        history.record(.empty, at: at(16))
        let empty = readout(history, 16)
        XCTAssertEqual(empty.pace, "— keyframes/min")
        XCTAssertEqual(empty.breaks, "Breaks in the last 30 s: —")
        XCTAssertNil(empty.stalled)
        XCTAssertNil(empty.mapLag)
    }

    /// Only the Stalled and the Breaks rows are said aloud, at most once
    /// every 10 s, and a breaks count is said only when it rises from a known
    /// one.
    func testAnnouncementsAreForStallsAndBreaksAtMostEveryTenSeconds() {
        func readout(breaks: Int?, stalled: String? = nil) -> CaptureHealthReadout {
            CaptureHealthReadout(link: "Tower: connected", pace: "1 keyframes/min", stalled: stalled,
                                 breaks: "", breaksSpoken: "Breaks in the last 30 seconds: \(breaks ?? 0)",
                                 lookBackLine: nil, lookBackCounts: "", mapLag: nil, breaksCount: breaks)
        }
        var announcer = CaptureHealthAnnouncer()
        XCTAssertNil(announcer.announcement(from: readout(breaks: nil), to: readout(breaks: 2), at: at(0)),
                     "the first known count is not news")
        XCTAssertEqual(announcer.announcement(from: readout(breaks: 2), to: readout(breaks: 3), at: at(1)),
                       "Breaks in the last 30 seconds: 3")
        XCTAssertNil(announcer.announcement(from: readout(breaks: 3), to: readout(breaks: 4), at: at(5)), "10 s apart")
        XCTAssertEqual(announcer.announcement(from: readout(breaks: 4), to: readout(breaks: 4, stalled: "Stalled — no new keyframe for 5 s"),
                                              at: at(11)), "Stalled — no new keyframe for 5 s")
        XCTAssertNil(announcer.announcement(from: readout(breaks: 4), to: readout(breaks: 3), at: at(30)), "a fall is not said")
    }
}
