//
//  WorldCoverageTests.swift
//  GlassesTests
//
//  Fog of war v1 (FOW-COVERAGE-V1-SPEC-20261006, frozen): §9's three Swift
//  fixtures, §5's fail-closed decoding, and §8's honesty assertions as the
//  phone states them. The client-side walk pairing is tested end to end in
//  `WorldBuilderIntegrationTests` (`TowerWorldBuilderClientTests`).
//

import CoreGraphics
import XCTest
@testable import Glasses

/// §9's fixtures, verbatim.
enum FOWFixtures {
    static let early = #"{"guidance":{"coverage":null}}"#
    static let mid = #"{"guidance":{"coverage":{"version":1,"source":"landed_global_solve","solved_at":1791240000.0,"computed_at":1791240000.12,"horizon_keyframes":113,"keyframes_now":127,"keyframes_pending":14,"geometry_revision":"g-mid-113","frame_revision":1,"station_grid":"component_square_8x8_v1","sector_frame":"first_qualified_forward_cw_from_up_v1","components_total":2,"components_omitted":0,"stations_omitted":0,"components":[{"reference_segment":0,"posed_keyframes":82,"bounds_xy":[-4.0,-4.0,4.0,4.0],"origin_xyz":[0.0,0.0,0.0],"up_xyz":[0.0,0.0,1.0],"forward_xyz":[0.0,1.0,0.0],"cell_size":1.0,"stations":[{"x":3,"y":5,"keyframes":30,"weak_mask":1,"supported_mask":30}]},{"reference_segment":17,"posed_keyframes":19,"bounds_xy":[-2.0,-2.0,2.0,2.0],"origin_xyz":[10.0,0.0,0.0],"up_xyz":[0.0,0.0,1.0],"forward_xyz":[1.0,0.0,0.0],"cell_size":0.5,"stations":[{"x":1,"y":2,"keyframes":8,"weak_mask":2,"supported_mask":1}]}]}}}"#
    static let stop = #"{"guidance":{"coverage":{"version":1,"source":"landed_global_solve","solved_at":1791240313.0,"computed_at":1791240313.19,"horizon_keyframes":984,"keyframes_now":984,"keyframes_pending":0,"geometry_revision":"g-stop-984","frame_revision":1,"station_grid":"component_square_8x8_v1","sector_frame":"first_qualified_forward_cw_from_up_v1","components_total":1,"components_omitted":0,"stations_omitted":0,"components":[{"reference_segment":0,"posed_keyframes":616,"bounds_xy":[-8.0,-8.0,8.0,8.0],"origin_xyz":[0.0,0.0,0.0],"up_xyz":[0.0,0.0,1.0],"forward_xyz":[0.0,1.0,0.0],"cell_size":2.0,"stations":[{"x":3,"y":5,"keyframes":120,"weak_mask":0,"supported_mask":30}]}]}}}"#

    static func payload(_ json: String) -> [String: Any] {
        // swiftlint:disable:next force_cast force_try
        try! JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
    }

    static func block(_ json: String) -> [String: Any] {
        // swiftlint:disable:next force_cast
        (payload(json)["guidance"] as! [String: Any])["coverage"] as! [String: Any]
    }

    static func coverage(_ json: String) -> WorldCoverage? {
        WorldCoverageStatus(payload: payload(json)).coverage
    }

    /// The `guidance` value carrying `block`.
    static func status(_ block: [String: Any]) -> WorldCoverageStatus {
        WorldCoverageStatus(payload: ["guidance": ["coverage": block]])
    }
}

final class WorldCoverageTests: XCTestCase {
    private typealias Station = WorldCoverage.Station
    private let t0 = ContinuousClock.now

    private var mid: WorldCoverage { FOWFixtures.coverage(FOWFixtures.mid)! }
    private var stop: WorldCoverage { FOWFixtures.coverage(FOWFixtures.stop)! }

    private func receipt(_ coverage: WorldCoverage, accepted: Int?, sentAfterSolve: Double?) -> WorldCoverageReceipt {
        WorldCoverageReceipt(coverage: coverage, keyframesAccepted: accepted,
                             clock: sentAfterSolve.map { WorldTowerClock(towerSentAt: coverage.solvedAt + $0, receivedAt: t0) })
    }

    // MARK: §9: the three fixtures

    func testTheThreeContractFixturesDecode() throws {
        XCTAssertEqual(WorldCoverageStatus(payload: FOWFixtures.payload(FOWFixtures.early)), .notYet,
                       "coverage:null is no receipt yet, not absent and not refused")

        let mid = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.mid))
        XCTAssertEqual(mid.solvedAt, 1791240000.0)
        XCTAssertEqual(mid.computedAt, 1791240000.12)
        XCTAssertEqual([mid.horizonKeyframes, mid.keyframesNow, mid.keyframesPending], [113, 127, 14])
        XCTAssertEqual(mid.geometryRevision, "g-mid-113")
        XCTAssertEqual(mid.frameRevision, 1)
        XCTAssertEqual([mid.componentsTotal, mid.componentsOmitted, mid.stationsOmitted], [2, 0, 0])
        XCTAssertEqual(mid.components.map(\.referenceSegment), [0, 17])
        XCTAssertEqual(mid.components.map(\.posedKeyframes), [82, 19])
        XCTAssertEqual(mid.components.map(\.cellSize), [1.0, 0.5])
        XCTAssertEqual(mid.components[0].stations, [Station(x: 3, y: 5, keyframes: 30, weakMask: 1, supportedMask: 30)])
        XCTAssertEqual(mid.components[1].stations, [Station(x: 1, y: 2, keyframes: 8, weakMask: 2, supportedMask: 1)])

        let stop = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.stop))
        XCTAssertEqual(stop.solvedAt, 1791240313.0)
        XCTAssertEqual([stop.horizonKeyframes, stop.keyframesNow, stop.keyframesPending], [984, 984, 0])
        XCTAssertEqual(stop.geometryRevision, "g-stop-984")
        XCTAssertEqual(stop.components.count, 1)
        XCTAssertEqual(stop.components[0].stations,
                       [Station(x: 3, y: 5, keyframes: 120, weakMask: 0, supportedMask: 30)])
    }

    /// The live Tower today: its switch is off, so `guidance` is absent, and
    /// the panel is exactly today's -- no map, no card.
    func testNoGuidanceIsTodaysPanel() {
        XCTAssertEqual(WorldCoverageStatus(payload: [:]), .absent)
        XCTAssertEqual(WorldCoverageStatus(payload: ["guidance": [String: Any]()]), .absent)
        XCTAssertEqual(WorldCoverageStatus(payload: ["guidance": NSNull()]), .absent)
        XCTAssertNil(WorldPanelMap.source(live: nil, fixtureEnabled: false), "no receipt, no fixture: no map")
        XCTAssertFalse(WorldPanelPhase.walking(hasMap: false).drawsCard, "Capture health alone")
        let live = receipt(mid, accepted: 127, sentAfterSolve: 1)
        XCTAssertEqual(WorldPanelMap.source(live: live, fixtureEnabled: true), .coverage(live),
                       "the fixture never stands over a real receipt")
    }

    func testUnknownFieldsAreIgnored() throws {
        var block = FOWFixtures.block(FOWFixtures.mid)
        block["future_field"] = ["anything": [1, 2, 3]]
        var components = try XCTUnwrap(block["components"] as? [[String: Any]])
        components[0]["extra"] = "kept out"
        var stations = try XCTUnwrap(components[0]["stations"] as? [[String: Any]])
        stations[0]["radius"] = 3
        components[0]["stations"] = stations
        block["components"] = components
        XCTAssertEqual(FOWFixtures.status(block).coverage, mid)
    }

    // MARK: §5: fail closed

    private func mutated(_ mutate: (inout [String: Any]) -> Void) -> [String: Any] {
        var block = FOWFixtures.block(FOWFixtures.mid)
        mutate(&block)
        return block
    }

    private static func component(_ index: Int, _ block: inout [String: Any],
                                  _ mutate: (inout [String: Any]) -> Void) {
        var components = block["components"] as? [[String: Any]] ?? []
        mutate(&components[index])
        block["components"] = components
    }

    private static func station(_ index: Int, _ block: inout [String: Any],
                                _ mutate: (inout [String: Any]) -> Void) {
        component(0, &block) { component in
            var stations = component["stations"] as? [[String: Any]] ?? []
            mutate(&stations[index])
            component["stations"] = stations
        }
    }

    func testAMalformedKnownFieldRefusesTheWholeBlock() {
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("version 2", { $0["version"] = 2 }),
            ("another source", { $0["source"] = "local_segment" }),
            ("solved_at as a string", { $0["solved_at"] = "1791240000.0" }),
            ("solved_at negative", { $0["solved_at"] = -1.0 }),
            ("computed before solved", { $0["computed_at"] = 1791239999.0 }),
            ("a boolean count", { $0["horizon_keyframes"] = true }),
            ("a fractional count", { $0["horizon_keyframes"] = 113.5 }),
            ("a count over 65535", { $0["keyframes_now"] = 65536 }),
            ("pending not now - horizon", { $0["keyframes_pending"] = 13 }),
            ("an empty geometry revision", { $0["geometry_revision"] = "" }),
            ("a 129-character geometry revision", { $0["geometry_revision"] = String(repeating: "g", count: 129) }),
            ("a non-ASCII geometry revision", { $0["geometry_revision"] = "g-mid-é" }),
            ("a negative frame revision", { $0["frame_revision"] = -1 }),
            ("another station grid", { $0["station_grid"] = "component_square_16x16_v2" }),
            ("another sector frame", { $0["sector_frame"] = "first_forward_ccw_v1" }),
            ("components not an array", { $0["components"] = ["0": 1] }),
            ("a missing count", { $0.removeValue(forKey: "stations_omitted") }),
            ("omitted not total - drawn", { $0["components_omitted"] = 1 }),
            ("components unsorted", { $0["components"] = ($0["components"] as? [Any])?.reversed() }),
            ("a repeated component", { block in Self.component(1, &block) { $0["reference_segment"] = 0 } }),
            ("a zero cell size", { block in Self.component(0, &block) { $0["cell_size"] = 0.0 } }),
            ("inverted bounds", { block in Self.component(0, &block) { $0["bounds_xy"] = [4.0, -4.0, -4.0, 4.0] } }),
            ("short bounds", { block in Self.component(0, &block) { $0["bounds_xy"] = [0.0, 0.0, 1.0] } }),
            ("no up axis", { block in Self.component(0, &block) { $0.removeValue(forKey: "up_xyz") } }),
            ("zero posed keyframes", { block in Self.component(0, &block) { $0["posed_keyframes"] = 0 } }),
            ("a column of 8", { block in Self.station(0, &block) { $0["x"] = 8 } }),
            ("a 13-bit mask", { block in Self.station(0, &block) { $0["weak_mask"] = 4096 } }),
            ("overlapping masks", { block in Self.station(0, &block) { $0["weak_mask"] = 3 } }),
            ("a station of no keyframes", { block in Self.station(0, &block) { $0["keyframes"] = 0 } }),
            ("stations out of (y,x) order", { block in Self.component(0, &block) { component in
                let one: [String: Any] = ["x": 1, "y": 5, "keyframes": 3, "weak_mask": 0, "supported_mask": 1]
                component["stations"] = (component["stations"] as? [Any] ?? []) + [one]
            } }),
            ("a station repeated", { block in Self.component(0, &block) { component in
                let stations = component["stations"] as? [Any] ?? []
                component["stations"] = stations + stations
            } }),
        ]
        XCTAssertEqual(FOWFixtures.status(FOWFixtures.block(FOWFixtures.mid)).coverage, mid, "the unmutated block")
        for (name, mutate) in cases {
            XCTAssertEqual(FOWFixtures.status(mutated(mutate)), .refused, name)
        }
        XCTAssertEqual(WorldCoverageStatus(payload: ["guidance": ["coverage": "a string"]]), .refused)
        XCTAssertEqual(WorldCoverageStatus(payload: ["guidance": ["coverage": [1, 2]]]), .refused)
    }

    /// §5's caps bound the block: more than 16 components or 16 stations is
    /// oversized, and refused rather than drawn in part.
    func testAnOversizedBlockIsRefused() {
        func block(components: Int, stationsEach: Int) -> [String: Any] {
            var block = FOWFixtures.block(FOWFixtures.mid)
            let rows: [[String: Any]] = (0..<components).map { index in
                let stations: [[String: Any]] = (0..<stationsEach).map { s in
                    ["x": s % 8, "y": s / 8, "keyframes": 2, "weak_mask": 0, "supported_mask": 1]
                }
                return ["reference_segment": index, "posed_keyframes": 2, "bounds_xy": [0.0, 0.0, 1.0, 1.0],
                        "origin_xyz": [0.0, 0.0, 0.0], "up_xyz": [0.0, 0.0, 1.0], "forward_xyz": [1.0, 0.0, 0.0],
                        "cell_size": 0.125, "stations": stations]
            }
            block["components"] = rows
            block["components_total"] = components
            block["components_omitted"] = 0
            return block
        }
        XCTAssertEqual(FOWFixtures.status(block(components: 16, stationsEach: 1)).coverage?.components.count, 16)
        XCTAssertEqual(FOWFixtures.status(block(components: 1, stationsEach: 16)).coverage?.components.first?
            .stations.count, 16)
        XCTAssertEqual(FOWFixtures.status(block(components: 17, stationsEach: 1)), .refused, "17 components")
        XCTAssertEqual(FOWFixtures.status(block(components: 1, stationsEach: 17)), .refused, "17 stations")
        XCTAssertEqual(FOWFixtures.status(block(components: 2, stationsEach: 9)), .refused,
                       "18 stations over the whole block")
    }

    /// §5: "A decoder drops any row that arrives with stations:[]" -- and
    /// says it is not drawn, never a fake empty room.
    func testARowWithNoStationsIsDroppedAndCountedAsNotDrawn() throws {
        let block = mutated { block in Self.component(1, &block) { $0["stations"] = [Any]() } }
        let coverage = try XCTUnwrap(FOWFixtures.status(block).coverage)
        XCTAssertEqual(coverage.components.map(\.referenceSegment), [0])
        XCTAssertEqual(coverage.piecesNotDrawn, 1)
        XCTAssertEqual(WorldCoverageCopy.notDrawn(coverage), "1 more piece not drawn")
        XCTAssertNil(WorldCoverageCopy.pieces(coverage), "one piece drawn: no separate-pieces line")
    }

    // MARK: §8.1: no pointing-only coverage; weak is not covered

    func testOneViewIsWeakTwoAreSupportedAndNoBitIsGrey() {
        let station = mid.components[0].stations[0]   // weak 0b1, supported 0b11110
        XCTAssertEqual((0..<12).map(station.evidence(sector:)),
                       [.weak, .supported, .supported, .supported, .supported]
                        + Array(repeating: WorldCoverageEvidence.unconfirmed, count: 7))
        let pointed = Station(x: 0, y: 0, keyframes: 40, weakMask: 0, supportedMask: 0)
        XCTAssertEqual(Set((0..<12).map(pointed.evidence(sector:))), [.unconfirmed],
                       "keyframes in a station with no supporting bit paint nothing")
        let fan = WorldCoverageGeometry.fans(mid.components[1], side: 80)[0]
        XCTAssertEqual(fan.sectors[0], .supported)
        XCTAssertEqual(fan.sectors[1], .weak)
        XCTAssertEqual(fan.sectors.filter { $0 == .unconfirmed }.count, 10)
    }

    // MARK: §8.2: the age is always visible

    func testTheMapIsDatedAtThirtySecondsAndStaleAbove() {
        let r = receipt(mid, accepted: 127, sentAfterSolve: 12)
        XCTAssertEqual(WorldCoverageCopy.label(r, at: t0, stopped: false), "Map through keyframe 113 · updated 12 s ago")
        let thirty = t0.advanced(by: .seconds(18))
        XCTAssertEqual(r.age(at: thirty), 30)
        XCTAssertEqual(r.isStale(at: thirty), false, "exactly 30 s: dated, not stale")
        XCTAssertEqual(WorldCoverageCopy.label(r, at: thirty, stopped: false),
                       "Map through keyframe 113 · updated 30 s ago")
        let past = t0.advanced(by: .milliseconds(18_500))
        XCTAssertEqual(r.isStale(at: past), true)
        XCTAssertEqual(WorldCoverageCopy.label(r, at: past, stopped: false),
                       "Map through keyframe 113 · updated 30 s ago · stale")
        XCTAssertEqual(WorldCoverageCopy.label(r, at: t0.advanced(by: .seconds(288)), stopped: false),
                       "Map through keyframe 113 · updated 5 min ago · stale")
        let skewed = receipt(mid, accepted: 127, sentAfterSolve: -5)
        XCTAssertEqual(WorldCoverageCopy.label(skewed, at: t0, stopped: false),
                       "Map through keyframe 113 · updated 0 s ago", "a Tower clock behind its solve: never negative")
    }

    func testWithoutAMeasuredClockTheAgeIsUnavailableNeverFresh() {
        let r = receipt(mid, accepted: 127, sentAfterSolve: nil)
        XCTAssertNil(r.age(at: t0))
        XCTAssertNil(r.isStale(at: t0))
        XCTAssertEqual(WorldCoverageCopy.label(r, at: t0, stopped: false, timeZone: TimeZone(identifier: "UTC")!),
                       "Map through keyframe 113 · solved at 22:40:00 Tower time · age unavailable")
    }

    /// §8.2 after Stop and §9's post-Stop fixture: "at Stop" only when the
    /// solve held every accepted keyframe, and the live count -- not the
    /// block's snapshot zero -- says 106 are newer.
    func testAfterStopAtStopOnlyWhenTheReceiptHoldsEveryKeyframe() {
        let behind = receipt(stop, accepted: 1090, sentAfterSolve: 2)
        XCTAssertEqual(stop.keyframesPending, 0, "the snapshot says none pending")
        XCTAssertEqual(WorldCoverageCopy.label(behind, at: t0, stopped: true),
                       "Map through keyframe 984 before Stop · updated 2 s ago")
        XCTAssertEqual(WorldCoverageCopy.newer(behind, stopped: true), "106 keyframes newer than this receipt")
        let whole = receipt(stop, accepted: 984, sentAfterSolve: 2)
        XCTAssertEqual(WorldCoverageCopy.label(whole, at: t0, stopped: true),
                       "Map at Stop, through keyframe 984 · updated 2 s ago")
        XCTAssertNil(WorldCoverageCopy.newer(whole, stopped: true))
        let unknown = receipt(stop, accepted: nil, sentAfterSolve: 2)
        XCTAssertEqual(WorldCoverageCopy.label(unknown, at: t0, stopped: true),
                       "Map through keyframe 984 before Stop · updated 2 s ago", "an unknown count is not every one")
    }

    /// Pending keyframes are counted, never placed: from live progress, and
    /// the snapshot only as a snapshot.
    func testNewerKeyframesAreCountedFromLiveProgress() {
        XCTAssertEqual(WorldCoverageCopy.newer(receipt(mid, accepted: 127, sentAfterSolve: 1), stopped: false),
                       "14 newer keyframes not yet placed")
        XCTAssertEqual(WorldCoverageCopy.newer(receipt(mid, accepted: 140, sentAfterSolve: 1), stopped: false),
                       "27 newer keyframes not yet placed", "the live count, not the block's 14")
        XCTAssertEqual(WorldCoverageCopy.newer(receipt(mid, accepted: 114, sentAfterSolve: 1), stopped: false),
                       "1 newer keyframe not yet placed")
        XCTAssertNil(WorldCoverageCopy.newer(receipt(mid, accepted: 113, sentAfterSolve: 1), stopped: false))
        XCTAssertEqual(WorldCoverageCopy.newer(receipt(mid, accepted: nil, sentAfterSolve: 1), stopped: false),
                       "14 keyframes not yet placed when this map was made")
    }

    // MARK: §8.3: islands stay separate

    func testTwoComponentsAreTwoPiecesEvenWhenTheirGridsCoincide() throws {
        // Component 17 made identical to component 0 but for its identity,
        // and far smaller: still its own piece, never merged, never "the room".
        let block = mutated { block in
            var components = block["components"] as? [[String: Any]] ?? []
            var twin = components[0]
            twin["reference_segment"] = 17
            twin["posed_keyframes"] = 2
            components[1] = twin
            block["components"] = components
        }
        let coverage = try XCTUnwrap(FOWFixtures.status(block).coverage)
        XCTAssertEqual(coverage.components.count, 2)
        let frames = WorldCoverageGeometry.pieceFrames(count: 2, in: CGSize(width: 343, height: 257))
        XCTAssertEqual(frames.count, 2)
        XCTAssertFalse(frames[0].intersects(frames[1]), "two pieces, drawn apart")
        XCTAssertEqual(WorldCoverageGeometry.fans(coverage.components[0], side: 100).count, 1,
                       "a piece draws only its own stations")
        XCTAssertEqual(WorldCoverageCopy.pieces(coverage), "2 separate pieces, not placed relative to each other")
        XCTAssertEqual([0, 1].map(WorldCoverageCopy.pieceName), ["Piece 1", "Piece 2"])
        let tray = WorldCoverageGeometry.pieceFrames(count: 5, in: CGSize(width: 343, height: 257))
        for (i, a) in tray.enumerated() {
            for b in tray[(i + 1)...] { XCTAssertFalse(a.intersects(b), "the tray never overlaps two pieces") }
            XCTAssertTrue(CGRect(x: 0, y: 0, width: 343, height: 257).contains(a), "inside the stage")
        }
    }

    func testTheCapsAreSaidNotHidden() {
        var block = FOWFixtures.block(FOWFixtures.mid)
        block["components_total"] = 5
        block["components_omitted"] = 3
        block["stations_omitted"] = 7
        let coverage = FOWFixtures.status(block).coverage
        XCTAssertEqual(coverage.flatMap(WorldCoverageCopy.notDrawn), "3 more pieces and 7 more stations not drawn")
        XCTAssertNil(WorldCoverageCopy.notDrawn(mid))
    }

    // MARK: §6.4 / §8.5: geometry, and no position

    /// The synthetic north/east fixture: sector zero is the component's
    /// forward axis, drawn to the right, and sectors run clockwise as seen
    /// from above -- sector 3 (the right axis) is drawn down, along rows.
    func testSectorZeroIsForwardAndSectorsRunClockwiseFromAbove() {
        let forward = WorldCoverageGeometry.direction(sector: 0)
        XCTAssertGreaterThan(forward.dx, 0.95)
        XCTAssertGreaterThan(forward.dy, 0, "clockwise on a y-down screen")
        XCTAssertGreaterThan(WorldCoverageGeometry.direction(sector: 3).dy, 0.95, "the right axis, drawn down")
        XCTAssertLessThan(WorldCoverageGeometry.direction(sector: 6).dx, -0.95, "behind")
        XCTAssertLessThan(WorldCoverageGeometry.direction(sector: 9).dy, -0.95, "left")
        XCTAssertTrue(WorldCoverageGeometry.span(sector: 0) == (0, 30))
        XCTAssertTrue(WorldCoverageGeometry.span(sector: 11) == (330, 360))
        let far = WorldCoverage.Component(referenceSegment: 0, posedKeyframes: 4, cellSize: 1, stations: [
            Station(x: 7, y: 0, keyframes: 2, weakMask: 0, supportedMask: 1),
            Station(x: 0, y: 7, keyframes: 2, weakMask: 0, supportedMask: 1),
        ])
        let fans = WorldCoverageGeometry.fans(far, side: 80)
        XCTAssertEqual(fans.map(\.centre), [CGPoint(x: 75, y: 5), CGPoint(x: 5, y: 75)],
                       "column x along forward (right), row y along right (down)")
    }

    /// A piece draws one fan per station at its cell and nothing else: no
    /// current-position dot, no path, no north arrow (§6, §8.5).
    func testAPieceDrawsOnlyItsStationsNeverAPosition() {
        let fans = WorldCoverageGeometry.fans(mid.components[0], side: 160)
        XCTAssertEqual(fans.count, 1, "one station, one fan, and nothing else")
        XCTAssertEqual(fans.first?.centre, CGPoint(x: 70, y: 110), "at its cell (3, 5)")
        XCTAssertEqual(fans.first?.radius ?? 0, 9.6, accuracy: 1e-9, "inside its cell")
        XCTAssertEqual(fans.first?.sectors, (0..<12).map(mid.components[0].stations[0].evidence(sector:)))
    }

    /// §8.5's "never claim" list, over every word the map can say.
    func testTheMapsWordsNeverClaimWhatTheReceiptDoesNot() {
        let banned = ["hole", "room", "covered", "coverage", "complete", "surface", "texture", "wall", "floor",
                      "current", "you are", "live", "scan", "missing", "unseen", "nothing", "%", "everything",
                      "position", "estimate", "remaining", "done", "proved", "real time", "just now"]
        var words: [String] = []
        let receipts = [receipt(mid, accepted: 140, sentAfterSolve: 3), receipt(mid, accepted: 127, sentAfterSolve: 300),
                        receipt(mid, accepted: nil, sentAfterSolve: nil), receipt(stop, accepted: 1090, sentAfterSolve: 2),
                        receipt(stop, accepted: 984, sentAfterSolve: 2)]
        for r in receipts {
            for stopped in [false, true] {
                words.append(WorldCoverageCopy.label(r, at: t0, stopped: stopped))
                words += [WorldCoverageCopy.newer(r, stopped: stopped)].compactMap { $0 }
            }
            words += [WorldCoverageCopy.pieces(r.coverage), WorldCoverageCopy.notDrawn(r.coverage)].compactMap { $0 }
            words += r.coverage.components.map(WorldCoverageCopy.pieceValue)
        }
        var capped = FOWFixtures.block(FOWFixtures.mid)
        capped["components_total"] = 4
        capped["components_omitted"] = 2
        capped["stations_omitted"] = 1
        words += [FOWFixtures.status(capped).coverage.flatMap(WorldCoverageCopy.notDrawn)].compactMap { $0 }
        words += WorldCoverageEvidence.allCases.map(WorldCoverageCaption.legendWord)
        words += [WorldCoverageCopy.fixture, WorldCoverageCopy.pieceName(0), WorldCoverageCopy.mapName,
                  WorldCoverageCopy.mapValue(mid), WorldCoverageCopy.mapValue(stop)]
        XCTAssertGreaterThan(words.count, 30)
        for word in words {
            for phrase in banned {
                XCTAssertFalse(word.lowercased().contains(phrase), "\"\(word)\" says \"\(phrase)\"")
            }
        }
        XCTAssertEqual(WorldCoverageCaption.legendWord(.unconfirmed), "Not yet seen by a finished solve",
                       "grey is unconfirmed at this solve, never a proved hole")
    }
}
