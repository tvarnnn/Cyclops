//
//  WorldCaptureDialTests.swift
//  GlassesTests
//
//  FOW-RING-SPEC (2026-10-12) S1-S3, iOS: the capture dial's model on the
//  15 real coverage blocks, its gap and words, its photos from today's wire
//  (S2) and from the Tower's own bins (S3), the additive B2 fields read
//  fail-closed, the latest-up orientation and the walk cue, and what the
//  phone says when a landing's photos never come.
//

import SwiftUI
import UIKit
import XCTest
@testable import Glasses

@MainActor
final class WorldCaptureDialTests: XCTestCase {
    private typealias Copy = WorldCaptureDialCopy
    private typealias Panel = WorldInlinePanel<EmptyView>

    static func coverage(_ name: String) throws -> WorldCoverage {
        let folder = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "CoverageBlocks", withExtension: nil))
        let data = try Data(contentsOf: folder.appendingPathComponent(name).appendingPathExtension("json"))
        let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try XCTUnwrap(WorldCoverageStatus(payload: payload).coverage, name)
    }

    /// One piece whose union is exactly `supported` / `weak` (sector indices).
    static func piece(_ reference: Int = 0, posed: Int = 50, supported: [Int], weak: [Int] = [])
        -> WorldCoverage.Component {
        let mask = { (sectors: [Int]) in sectors.reduce(0) { $0 | (1 << $1) } }
        return WorldCoverage.Component(referenceSegment: reference, posedKeyframes: posed, cellSize: 1,
                                       stations: [.init(x: 3, y: 3, keyframes: posed, weakMask: mask(weak),
                                                        supportedMask: mask(supported))])
    }

    private func dial(supported: [Int], weak: [Int] = [], latest: Int? = nil, motion: Int? = nil,
                      photos: [WorldCaptureDial.Photo] = []) -> WorldCaptureDial {
        WorldCaptureDial(components: [Self.piece(supported: supported, weak: weak)], stationsOmitted: 0,
                         newest: latest.map { .init(referenceSegment: 0, sector: $0) },
                         motionSector: motion, photos: photos)
    }

    // MARK: S1: the union over one piece's drawn stations (§3.2, §7)

    func testTheRealBlocksFillLikeTheSpecSays() throws {
        for (name, count) in [("walk7-h52", 2), ("walk7-h236", 5), ("walk71-h563", 8), ("walk6-h1090", 12)] {
            let dial = WorldCaptureDial(coverage: try Self.coverage(name))
            XCTAssertEqual(dial.hero?.supportedCount, count, name)
            XCTAssertEqual(dial.top, .pieceStart, "\(name): no latest, sector zero on top")
            XCTAssertEqual(dial.anchor, 0, name)
        }
        let walk71 = WorldCaptureDial(coverage: try Self.coverage("walk71-h563"))
        XCTAssertEqual(walk71.heroIndex, 0, "the most posed keyframes")
        XCTAssertEqual(walk71.hero?.referenceSegment, 0)
        XCTAssertEqual(walk71.otherPieces.map(\.supportedCount), [5, 3], "the other drawn pieces")
        XCTAssertEqual(Copy.heroLabel(walk71), "Piece 1 · most views")
        XCTAssertEqual(walk71.gap?.positions, [6, 7, 8, 9])
        XCTAssertEqual(Copy.gapLine(walk71, age: 12),
                       "More needed at 6 to 9 o'clock, mostly behind the way you faced when this piece began.")
        let walk6 = WorldCaptureDial(coverage: try Self.coverage("walk6-h1090"))
        XCTAssertNil(walk6.gap)
        XCTAssertEqual(Copy.gapLine(walk6, age: 12), "All 12 directions captured in this piece.")
        XCTAssertEqual(Copy.heroLabel(WorldCaptureDial(coverage: try Self.coverage("walk7-h236"))), "Most views")
        // Every real block: the counts are of the hero's drawn union only.
        for name in ["walk6-h53", "walk6-h108", "walk6-h200", "walk6-h362", "walk7-h103", "walk7-h509",
                     "walk7-h991", "walk71-h52", "walk71-h104", "walk71-h156", "walk71-h254"] {
            let coverage = try Self.coverage(name)
            let dial = WorldCaptureDial(coverage: coverage)
            let hero = try XCTUnwrap(dial.hero, name)
            let union = coverage.components[hero.index].stations.reduce(0) { $0 | $1.supportedMask }
            XCTAssertEqual(hero.supportedCount, union.nonzeroBitCount, name)
        }
    }

    func testBeforeTheFirstLandingTheDialIsEmptyAndSaysWhy() {
        let dial = WorldCaptureDial.awaiting
        XCTAssertNil(dial.hero)
        XCTAssertFalse(dial.hasReceipt)
        XCTAssertNil(dial.gap)
        XCTAssertNil(dial.emphasis(age: 1))
        XCTAssertEqual((0..<12).map(dial.evidence(at:)), Array(repeating: .unconfirmed, count: 12))
        XCTAssertEqual(Copy.count(dial), "–")
        XCTAssertEqual(Copy.compactCount(dial), "–")
        XCTAssertEqual(Copy.gapLine(dial, age: 1),
                       "The dial starts filling when the Tower places your first views.")
        XCTAssertEqual(Copy.accessibilityLabel(dial), "Capture dial")
        XCTAssertEqual(Copy.accessibilityValue(dial, age: nil), Copy.awaitingLine)
        XCTAssertEqual(Panel.captureDial(nil, overlay: nil), .awaiting)
        XCTAssertTrue(Panel.drawsDial(WorldPanelDial(awaiting: true), map: nil))
        XCTAssertFalse(Panel.drawsDial(WorldPanelDial(awaiting: false), map: nil), "absent: today's panel")
    }

    // MARK: The gap (§3.5)

    func testTheGapWrapsThroughTwelveAndSaysItsSide() {
        let wrap = dial(supported: Array(2...9))
        XCTAssertEqual(wrap.gap, .init(positions: [10, 11, 0, 1], evidence: .unconfirmed))
        XCTAssertEqual(Copy.gapLine(wrap, age: 3),
                       "More needed at 10 to 1 o'clock, mostly ahead of the way you faced when this piece began.")
        let behind = dial(supported: [0, 1, 2, 3, 4, 8, 9, 10], weak: [11], latest: 0)
        XCTAssertEqual(Copy.gapLine(behind, age: 12), "More needed at 5 to 7 o'clock, behind your newest placed view.")
        XCTAssertEqual(Copy.gapLine(dial(supported: Array(0...9) + [11], latest: 0), age: 12),
                       "More needed at 10 o'clock, left of your newest placed view.")
        XCTAssertEqual(Copy.gapLine(dial(supported: [0] + Array(4...11), latest: 0), age: 12),
                       "More needed at 1 to 3 o'clock, right of your newest placed view.")
        // A midpoint on a boundary goes to ahead of or behind.
        XCTAssertEqual(Copy.gapLine(dial(supported: [0, 1, 2, 5, 6, 7, 8, 9, 10, 11], latest: 0), age: 12),
                       "More needed at 3 to 4 o'clock, right of your newest placed view.")
        XCTAssertEqual(Copy.gapLine(dial(supported: [0, 1, 2, 3, 6, 7, 8, 9, 10, 11], latest: 0), age: 12),
                       "More needed at 4 to 5 o'clock, behind your newest placed view.")
        XCTAssertEqual(Copy.gapLine(dial(supported: Array(1...8), latest: 0), age: 12),
                       "More needed at 9 to 12 o'clock, mostly ahead of your newest placed view.")
    }

    func testATieGoesToTheRunMetFirstFromTwelve() {
        XCTAssertEqual(dial(supported: [0, 1, 2, 5, 6, 7, 10, 11]).gap?.positions, [3, 4])
        // A run through 12 o'clock is met at once, before [5, 6].
        XCTAssertEqual(dial(supported: [1, 2, 3, 4, 7, 8, 9, 10]).gap?.positions, [11, 0])
        // Longer wins regardless of where it starts.
        XCTAssertEqual(dial(supported: [0, 1, 4, 5, 9, 10, 11]).gap?.positions, [6, 7, 8])
    }

    func testWithNothingUnseenTheGapIsTheLongestOneViewRun() {
        let one = dial(supported: [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], weak: [3])
        XCTAssertEqual(one.gap, .init(positions: [3], evidence: .weak))
        XCTAssertEqual(Copy.gapLine(one, age: 40), "One view only at 3 o'clock. Another view there would capture it.")
        let two = dial(supported: [0, 1, 2, 5, 6, 7, 8, 9, 10, 11], weak: [3, 4])
        XCTAssertEqual(Copy.gapLine(two, age: 4),
                       "One view only at 3 to 4 o'clock. Another view of each would capture them.")
        let none = dial(supported: [], weak: Array(0..<12))
        XCTAssertEqual(Copy.gapLine(none, age: 4),
                       "One view only in all 12 directions. Another view of each would capture them.")
        XCTAssertEqual(Copy.gapLine(dial(supported: []), age: 4), "More needed in all 12 directions.")
    }

    /// §8.2: a map older than 30 s, or one with no measured age, is never
    /// emphasised and never gives a side.
    func testAStaleOrUndatedMapIsNeverEmphasised() {
        let walking = dial(supported: [0, 1, 2, 3, 4, 8, 9, 10, 11], latest: 0)
        XCTAssertEqual(walking.emphasis(age: 30)?.positions, [5, 6, 7], "at exactly 30 s: dated, not stale")
        XCTAssertNil(walking.emphasis(age: 30.5))
        XCTAssertNil(walking.emphasis(age: nil))
        for age in [30.5, 600, nil] as [Double?] {
            XCTAssertEqual(Copy.gapLine(walking, age: age), "Not yet captured when this map was made: 5 to 7 o'clock.")
        }
        XCTAssertTrue(Copy.isFresh(0))
        XCTAssertFalse(Copy.isFresh(nil))
    }

    // MARK: Accessibility (§4)

    func testTheDialIsOneElementWithTheSpecsValue() {
        let spec = dial(supported: [0, 1, 2, 3, 4, 8, 9, 10], weak: [11], latest: 0)
        XCTAssertEqual(Copy.accessibilityValue(spec, age: 12),
                       "8 of 12 directions captured, 1 with one view. More needed at 5 to 7 o'clock, behind your "
                       + "newest placed view. Top of the dial is your newest placed view.")
        XCTAssertEqual(Copy.accessibilityLabel(spec), "Capture dial, your newest views")
        let none = dial(supported: [0, 1, 2, 3, 4, 8, 9, 10, 11])
        XCTAssertEqual(Copy.accessibilityValue(none, age: 50),
                       "9 of 12 directions captured. Not yet captured when this map was made: 5 to 7 o'clock. "
                       + "Top of the dial is the way you faced when this piece began.")
        XCTAssertEqual(Copy.compactCount(spec), "8/12")
        XCTAssertEqual(Copy.photo(at: 0), "Photo at 12 o'clock")
        XCTAssertEqual(Copy.photo(at: 3), "Photo at 3 o'clock")
        XCTAssertEqual(Copy.topLine(spec), "Top of the dial: your newest placed view.")
        XCTAssertEqual(Copy.topLine(none), "Top of the dial: the way you faced when this piece began.")
    }

    // MARK: The switch

    func testTheSwitchIsOnOnlyForOnAndAsksForB2sPhotos() {
        XCTAssertEqual(WorldCaptureDialSwitch(environment: [:], info: [:]), .off)
        XCTAssertEqual(WorldCaptureDialSwitch(environment: ["IOS_FOW_DIAL": "ON"]), .off)
        XCTAssertEqual(WorldCaptureDialSwitch(environment: ["IOS_FOW_DIAL": "1"]), .off)
        XCTAssertEqual(WorldCaptureDialSwitch(environment: [:], info: ["GlassesFOWDial": ""]), .off)
        XCTAssertTrue(WorldCaptureDialSwitch(environment: ["IOS_FOW_DIAL": "on"]).on)
        XCTAssertTrue(WorldCaptureDialSwitch(environment: [:], info: ["GlassesFOWDial": "on"]).on)
        XCTAssertNotNil(Bundle.main.infoDictionary?["GlassesFOWDial"], "the home-screen key")
        XCTAssertEqual(WorldImagerySwitches(environment: ["IOS_FOW_DIAL": "on"]),
                       WorldImagerySwitches(live: false, landed: true), "the dial's photos are B2's tiles")
        XCTAssertEqual(WorldImagerySwitches(environment: [:]), .off)
    }

    // MARK: S2: photos from today's wire (§3.4)

    func testS2PhotosGoToTheirHeadingsSectorNeverOnGreyNorInAnotherPiece() {
        let image = UIImage()
        func tile(_ digest: String, reference: Int = 0, heading: Double?) -> WorldDialThumbnail {
            WorldDialThumbnail(digest: digest, image: image, referenceSegment: reference, heading: heading)
        }
        let photos = WorldCaptureDial.photos([
            tile("a", heading: 95), tile("b", heading: 359.9), tile("c", heading: 0), tile("d", heading: nil),
            tile("e", heading: .nan), tile("f", reference: 17, heading: 100), tile("g", heading: 125),
            tile("h", heading: 61),
        ])
        XCTAssertEqual(photos.map(\.digest), ["a", "b", "c", "f", "g", "h"], "no direction, no photo")
        XCTAssertEqual(photos.map(\.sector), [3, 11, 0, 3, 4, 2], "the 30-degree bin")
        let dial = WorldCaptureDial(components: [Self.piece(supported: [0, 3, 4, 11], weak: [2]),
                                                 Self.piece(17, posed: 5, supported: [3])],
                                    stationsOmitted: 0, photos: photos)
        XCTAssertEqual(dial.photosBySector, [3: "a", 11: "b", 0: "c", 4: "g", 2: "h"],
                       "piece 2's photo stays off piece 1; a one-view segment keeps its photo")
        let grey = WorldCaptureDial(components: [Self.piece(supported: [0])], stationsOmitted: 0,
                                    photos: [.init(digest: "x", referenceSegment: 0, sector: 5)])
        XCTAssertEqual(grey.photosBySector, [:], "never on grey")
        let last = WorldCaptureDial(components: [Self.piece(supported: [1])], stationsOmitted: 0,
                                    photos: [.init(digest: "first", referenceSegment: 0, sector: 1),
                                             .init(digest: "last", referenceSegment: 0, sector: 1)])
        XCTAssertEqual(last.photosBySector, [1: "last"], "the last one wins")
    }

    /// The dial's join is `place`'s -- the same component, frame and pose --
    /// without the map's station-cell test.
    func testTheS2HeadingJoinNeedsTheLandingsFrameButNotAStationCell() throws {
        typealias F = FOWImageryFixtures
        let stop = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.stop))
        let landed = try XCTUnwrap(F.manifest(F.landed)?.landed.value)
        let headings = WorldLandedPlacement.headings(landed, on: stop, chunks: [F.chunk()])
        XCTAssertEqual(try XCTUnwrap(headings[F.landedDigest]), 90, accuracy: 1e-9)
        // Off every station cell: the map drops it, the dial keeps its direction.
        let offCell = [F.chunk(translation: [-6, 6, 0])]
        var tiles = WorldImageryTiles()
        tiles.insert(UIImage(data: F.jpeg())!, digest: F.landedDigest)
        XCTAssertEqual(WorldLandedPlacement.place(landed, on: stop, chunks: offCell, tiles: tiles), [])
        XCTAssertNotNil(WorldLandedPlacement.headings(landed, on: stop, chunks: offCell)[F.landedDigest])
        XCTAssertEqual(WorldLandedPlacement.headings(landed, on: stop, chunks: [F.chunk(frame: 2)]), [:],
                       "another frame revision")
        XCTAssertEqual(WorldLandedPlacement.headings(landed, on: stop, chunks: [F.chunk(reference: 5)]), [:],
                       "another component")
    }

    // MARK: S3: the Tower's bins, latest-up, the walk cue (§3.3, §7)

    func testLatestPutsItsSectorOnTopAndItsPieceInFront() throws {
        let latest = WorldCaptureDial(components: [Self.piece(supported: [0, 1, 2, 5])], stationsOmitted: 0,
                                      newest: .init(referenceSegment: 0, sector: 5))
        XCTAssertEqual(latest.top, .newestView)
        XCTAssertEqual(latest.anchor, 5)
        XCTAssertEqual(latest.sector(at: 0), 5)
        XCTAssertEqual(latest.position(of: 5), 0)
        XCTAssertEqual(latest.position(of: 0), 7)
        XCTAssertEqual(latest.evidence(at: 0), .supported)
        XCTAssertEqual(latest.evidence(at: 7), .supported, "sector 0, seven hours on")
        XCTAssertEqual(Copy.heroLabel(latest), "Your newest views")

        // walk 7.1: latest names piece 2, which has fewer views: it is the hero.
        let coverage = try Self.coverage("walk71-h563")
        let second = WorldCaptureDial(coverage: coverage,
                                      newest: .init(referenceSegment: 13, sector: 1))
        XCTAssertEqual(second.heroIndex, 1)
        XCTAssertEqual(second.hero?.supportedCount, 5)
        XCTAssertEqual(Copy.heroLabel(second), "Piece 2 · your newest views")
        XCTAssertEqual(second.otherPieces.map(\.name), ["Piece 1", "Piece 3"])
    }

    /// The coordinator's three S3 cases: `latest` null, `latest` naming no
    /// drawn piece, and `latest` at an undrawn station (a valid anchor, no
    /// photo, its segment as the drawn masks say).
    func testANullForeignOrUndrawnLatestIsHandled() throws {
        let coverage = try Self.coverage("walk71-h563")
        let none = WorldCaptureDial(coverage: coverage, newest: nil, motionSector: 3)
        XCTAssertEqual(none.top, .pieceStart)
        XCTAssertEqual(none.heroIndex, 0, "most views")
        XCTAssertNil(none.motionSector, "no walk cue without the latest it belongs to")
        let foreign = WorldCaptureDial(coverage: coverage, newest: .init(referenceSegment: 99, sector: 4),
                                       motionSector: 3)
        XCTAssertEqual(foreign.top, .pieceStart, "a piece not drawn here is no anchor")
        XCTAssertEqual(foreign.anchor, 0)
        XCTAssertNil(foreign.motionSector)
        // walk 7.1 piece 1 has nothing at sectors 6-9 among its drawn
        // stations; the newest view's station was not drawn.
        let undrawn = WorldCaptureDial(coverage: coverage,
                                       newest: .init(referenceSegment: 0, sector: 7),
                                       photos: [.init(digest: "p7", referenceSegment: 0, sector: 7),
                                                .init(digest: "p0", referenceSegment: 0, sector: 0)])
        XCTAssertEqual(undrawn.top, .newestView)
        XCTAssertEqual(undrawn.anchor, 7, "still the anchor")
        XCTAssertEqual(undrawn.evidence(at: 0), .unconfirmed, "drawn as the drawn masks say")
        XCTAssertEqual(undrawn.photosBySector, [0: "p0"], "no photo on it")
        XCTAssertEqual(undrawn.gap?.positions.first, 11, "the gap through the top is met at once")
    }

    /// A stale `latest` -- the Tower kept the previous landing's block --
    /// never anchors the CURRENT receipt's dial: the panel draws B2 only
    /// over the receipt it was made for, and then falls back (§3.3).
    func testAStaleLatestNeverAnchorsTheCurrentReceipt() throws {
        let stop = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.stop))
        let receipt = WorldCoverageReceipt(coverage: stop, keyframesAccepted: 984, clock: nil)
        let latest = WorldImageryLanded.Latest(componentReferenceSegment: 0, sector: 3, keyframeID: "s9:120")
        let image = UIImage(data: FOWImageryFixtures.jpeg())!
        let overlay = WorldLandedOverlay(receipt: receipt, thumbnails: [], dialEntries: [
            WorldDialThumbnail(digest: "d1", image: image, referenceSegment: 0, heading: nil, sector: 2),
        ], latest: latest, motionSector: 1)
        let current = Panel.captureDial(.coverage(receipt), overlay: Panel.landed(overlay, over: .coverage(receipt)))
        XCTAssertEqual(current.anchor, 3)
        XCTAssertEqual(current.top, .newestView)
        XCTAssertEqual(current.motionSector, 1)
        XCTAssertEqual(current.photosBySector, [2: "d1"])

        let newer = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.stop
            .replacingOccurrences(of: "1791240313.", with: "1791240320.")
            .replacingOccurrences(of: "g-stop-984", with: "g-stop-990")))
        let next = WorldCoverageReceipt(coverage: newer, keyframesAccepted: 990, clock: nil)
        let stale = Panel.captureDial(.coverage(next), overlay: Panel.landed(overlay, over: .coverage(next)))
        XCTAssertEqual(stale.anchor, 0, "the previous landing's latest")
        XCTAssertEqual(stale.top, .pieceStart)
        XCTAssertNil(stale.motionSector)
        XCTAssertEqual(stale.photosBySector, [:], "nothing carries over")
        let fixture = Panel.captureDial(.fixture(receipt), overlay: Panel.landed(overlay, over: .fixture(receipt)))
        XCTAssertEqual(fixture.top, .pieceStart, "never over the DEBUG fixture")
    }

    func testTheWalkCueIsATickAndASentence() {
        let cue = dial(supported: [0, 1, 2], latest: 5, motion: 7)
        XCTAssertEqual(cue.motionSector, 7)
        XCTAssertEqual(Copy.motion(cue), "You were moving toward 2 o'clock")
        XCTAssertTrue(Copy.accessibilityValue(cue, age: 5).contains("You were moving toward 2 o'clock."))
        XCTAssertNil(Copy.motion(dial(supported: [0], latest: 0, motion: nil)), "null: no tick")
        XCTAssertNil(dial(supported: [0], latest: 0, motion: 12).motionSector, "out of range: no tick")
    }

    // MARK: S3: the additive B2 fields, read fail-closed

    /// S3: one tile per set sector, binned by the Tower; its bin wins over
    /// the phone's join, and a tile with only a bin still has a direction.
    func testTheTowersOwnBinWinsOverTheJoin() {
        let image = UIImage()
        let photos = WorldCaptureDial.photos([
            WorldDialThumbnail(digest: "a", image: image, referenceSegment: 0, heading: 10, sector: 4),
            WorldDialThumbnail(digest: "b", image: image, referenceSegment: 0, heading: nil, sector: 7),
            WorldDialThumbnail(digest: "c", image: image, referenceSegment: 0, heading: 95, sector: nil),
        ])
        XCTAssertEqual(photos.map(\.sector), [4, 7, 3])
        let dial = WorldCaptureDial(components: [Self.piece(supported: [3, 4, 5])], stationsOmitted: 0,
                                    newest: .init(referenceSegment: 0, sector: 4), photos: photos)
        XCTAssertEqual(dial.photosBySector, [4: "a", 3: "c"], "set sectors only: a binned tile on grey is dropped")
        XCTAssertEqual(dial.position(of: 4), 0, "the newest view's own photo sits on top")
    }

    private static let digestA = "a" + String(repeating: "0", count: 63)
    private static let digestB = "b" + String(repeating: "0", count: 63)

    private static func manifest(entryA: String = #","sector":3"#, entryB: String = #","sector":7"#,
                                 extra: String = "") -> WorldImageryManifest? {
        FOWImageryFixtures.manifest(#"{"version":1,"landed":{"solved_at":1791240313.0,"geometry_revision":"g-stop-984","frame_revision":1,"entries":[{"keyframe_id":"s9:120","digest":"\#(digestA)","redaction":"redacted","pose_source":"landed_global_solve","component_reference_segment":0,"segment_index":null\#(entryA)},{"keyframe_id":"s9:121","digest":"\#(digestB)","redaction":"redacted","pose_source":"landed_global_solve","component_reference_segment":0,"segment_index":null\#(entryB)}]\#(extra)}}"#)
    }

    func testTheS3FieldsDecodeWhenPresent() throws {
        let landed = try XCTUnwrap(Self.manifest(
            extra: #","latest":{"component_reference_segment":0,"sector":7,"keyframe_id":"s9:121"},"motion_sector":2"#)?
            .landed.value)
        XCTAssertEqual(landed.entries.map(\.sector), [3, 7])
        XCTAssertEqual(landed.latest, .init(componentReferenceSegment: 0, sector: 7, keyframeID: "s9:121"))
        XCTAssertEqual(landed.motionSector, 2)
    }

    func testTheS3FieldsAbsentOrNullAreTodaysWire() throws {
        let absent = try XCTUnwrap(Self.manifest(entryA: "", entryB: "")?.landed.value)
        XCTAssertEqual(absent.entries.map(\.sector), [nil, nil])
        XCTAssertNil(absent.latest)
        XCTAssertNil(absent.motionSector)
        let null = try XCTUnwrap(Self.manifest(entryA: #","sector":null"#, entryB: "",
                                               extra: #","latest":null,"motion_sector":null"#)?.landed.value)
        XCTAssertEqual(null.entries.count, 2)
        XCTAssertNil(null.latest)
        XCTAssertNil(null.motionSector, "null: no tick")
        XCTAssertEqual(try XCTUnwrap(FOWImageryFixtures.manifest(FOWImageryFixtures.landed)?.landed.value).entries
                        .map(\.sector), [nil], "§4's fixture as it was")
    }

    /// Each additive field fails closed on its own: a malformed entry
    /// `sector` drops that entry; a malformed `latest` or `motion_sector` is
    /// read as absent, the entries kept, and the dial falls back.
    func testAMalformedSectorDropsItsEntryAndAMalformedLatestOrMotionIsDroppedAlone() throws {
        for bad in [#","sector":12"#, #","sector":-1"#, #","sector":"3""#, #","sector":2.5"#, #","sector":true"#,
                    #","sector":[3]"#] {
            let landed = try XCTUnwrap(Self.manifest(entryA: bad)?.landed.value, bad)
            XCTAssertEqual(landed.entries.map(\.digest), [Self.digestB], "\(bad): dropped, never repaired")
        }
        let motion = #","motion_sector":2"#
        for bad in [#","latest":{"component_reference_segment":0,"sector":12,"keyframe_id":"s9:1"}"#,
                    #","latest":{"component_reference_segment":0,"sector":3}"#,
                    #","latest":{"component_reference_segment":0,"sector":3,"keyframe_id":""}"#,
                    #","latest":{"component_reference_segment":"0","sector":3,"keyframe_id":"s9:1"}"#,
                    #","latest":{"sector":3,"keyframe_id":"s9:1"}"#,
                    #","latest":[0,3]"#, #","latest":"s9:1""#] {
            let landed = try XCTUnwrap(Self.manifest(extra: bad + motion)?.landed.value, "\(bad) refused the section")
            XCTAssertNil(landed.latest, bad)
            XCTAssertEqual(landed.entries.map(\.sector), [3, 7], "\(bad) cost the entries")
            XCTAssertEqual(landed.motionSector, 2, "\(bad) cost the walk cue")
        }
        let latest = #","latest":{"component_reference_segment":0,"sector":7,"keyframe_id":"s9:121"}"#
        for bad in [#","motion_sector":12"#, #","motion_sector":-1"#, #","motion_sector":"2""#,
                    #","motion_sector":true"#, #","motion_sector":1.5"#, #","motion_sector":[2]"#] {
            let landed = try XCTUnwrap(Self.manifest(extra: latest + bad)?.landed.value, "\(bad) refused the section")
            XCTAssertNil(landed.motionSector, bad)
            XCTAssertEqual(landed.latest?.sector, 7, "\(bad) cost the latest")
            XCTAssertEqual(landed.entries.count, 2, "\(bad) cost the entries")
        }
        // The dial then falls back: sector zero on top, no walk cue.
        let stop = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.stop))
        let receipt = WorldCoverageReceipt(coverage: stop, keyframesAccepted: nil, clock: nil)
        let landed = try XCTUnwrap(Self.manifest(extra: #","latest":{"sector":3},"motion_sector":"2""#)?.landed.value)
        let overlay = WorldLandedOverlay(receipt: receipt, thumbnails: [], latest: landed.latest,
                                         motionSector: landed.motionSector)
        let dial = Panel.captureDial(.coverage(receipt), overlay: overlay)
        XCTAssertEqual(dial.anchor, 0)
        XCTAssertEqual(dial.top, .pieceStart)
        XCTAssertNil(dial.motionSector)
    }

    // MARK: The words never claim what the receipt does not (§8.5)

    func testTheDialsWordsNeverClaimWhatTheReceiptDoesNot() throws {
        let banned = ["hole", "room", "covered", "coverage", "complete", "surface", "texture", "wall", "floor",
                      "current", "you are", "live", "scan", "missing", "unseen", "nothing", "%", "everything",
                      "position", "estimate", "remaining", "done", "proved", "real time", "just now",
                      "not captured", "never captured", "adequate", "turn ", "look ", "go ", "north", "compass"]
        var words = [Copy.dialName, Copy.tabLabel, Copy.awaitingLine, Copy.photosUnavailable, Copy.ofTwelve,
                     Copy.captured, Copy.legendCaptured, Copy.legendOneView, Copy.legendNotYet, Copy.allCaptured,
                     Copy.tabName(.dial), Copy.tabName(.map), Copy.more(4)]
        var dials = [WorldCaptureDial.awaiting]
        for name in ["walk7-h52", "walk7-h236", "walk71-h563", "walk6-h1090", "walk71-h104"] {
            let coverage = try Self.coverage(name)
            dials.append(WorldCaptureDial(coverage: coverage))
            dials.append(WorldCaptureDial(coverage: coverage, newest: .init(referenceSegment: coverage.components[0].referenceSegment, sector: 3),
                motionSector: 5))
        }
        dials += [dial(supported: []), dial(supported: [], weak: Array(0..<12)), dial(supported: [1], weak: [2, 3])]
        for dial in dials {
            for age in [2, 31, nil] as [Double?] {
                words += [Copy.gapLine(dial, age: age), Copy.accessibilityValue(dial, age: age)]
            }
            words += [Copy.heroLabel(dial), Copy.topLine(dial), Copy.accessibilityLabel(dial), Copy.count(dial),
                      Copy.compactCount(dial)] + [Copy.motion(dial)].compactMap { $0 }
            words += dial.pieces.flatMap { [Copy.miniLabel($0), Copy.miniAccessibilityLabel($0)] }
            words += (0..<12).map(Copy.photo(at:))
        }
        XCTAssertGreaterThan(words.count, 150)
        for word in words {
            for phrase in banned {
                XCTAssertFalse(word.lowercased().contains(phrase), "\"\(word)\" says \"\(phrase)\"")
            }
        }
    }
}

// MARK: - Photos unavailable for this update (§3.4)

@MainActor
final class WorldCaptureDialImageryTests: XCTestCase {
    private typealias F = FOWImageryFixtures
    private typealias Stub = FOWImageryStubProtocol

    private static let walk = WorldFinishWalk(worldID: "w1", sessionID: "s9")
    private static let manifest = "/worlds/w1/guidance/s9/imagery/manifest"
    private static let tile = "/worlds/w1/guidance/s9/imagery/tile/\(FOWImageryFixtures.landedDigest)"

    private var stop: WorldCoverage { FOWFixtures.coverage(FOWFixtures.stop)! }

    override func setUp() {
        super.setUp()
        Stub.reset()
        Stub.set(Self.manifest, json: F.landed)
        Stub.set(Self.tile, jpeg: F.jpeg())
    }

    private func model() -> WorldImageryModel {
        WorldImageryModel(switches: WorldImagerySwitches(live: false, landed: true),
                          client: WorldImageryClient(baseURL: URL(string: "http://tower.test")!,
                                                     session: Stub.makeSession(), timeout: 5))
    }

    private func expect(_ what: String = "", timeout: TimeInterval = 5, _ condition: () -> Bool) async {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() && Date() < deadline { try? await Task.sleep(for: .milliseconds(20)) }
        XCTAssertTrue(condition(), what)
    }

    private func settle() async { try? await Task.sleep(for: .milliseconds(300)) }

    func testPhotosInHandAreNeverUnavailable() async {
        let model = model()
        model.coverageReported(WalkScoped(walk: Self.walk, value: stop), receiving: true)
        await expect("the tile") { model.imagery.value.landedTiles[F.landedDigest] != nil }
        await settle()
        XCTAssertNil(model.imagery.value.landedUnavailable)
    }

    func testARefusedOrAbsentRouteMakesThisLandingsPhotosUnavailable() async {
        for (status, body) in [(404, #"{"detail":"Not Found"}"#), (200, #"{"version":2}"#), (200, F.live)] {
            Stub.set(Self.manifest, status: status, json: body)
            let model = model()
            model.coverageReported(WalkScoped(walk: Self.walk, value: stop), receiving: true)
            await expect("\(status) \(body)") { model.imagery.value.landedUnavailable == WorldImageryLanding(stop) }
        }
    }

    func testEveryTileFailingMakesThisLandingsPhotosUnavailable() async {
        Stub.set(Self.tile, status: 500, json: "{}")
        let model = model()
        model.coverageReported(WalkScoped(walk: Self.walk, value: stop), receiving: true)
        await expect("unavailable") { model.imagery.value.landedUnavailable == WorldImageryLanding(stop) }
    }

    /// The Tower kept the previous landing's block (no tiles for this one):
    /// unavailable once the bounded retries are spent, and the next landing
    /// starts clean.
    func testAKeptOldBlockIsUnavailableAfterTheRetriesAndTheNextLandingClearsIt() async {
        let newer = FOWFixtures.coverage(FOWFixtures.stop.replacingOccurrences(of: "1791240313.",
                                                                               with: "1791240320."))!
        let model = model()
        model.coverageReported(WalkScoped(walk: Self.walk, value: newer), receiving: true)
        await expect { Stub.count(Self.manifest) == 1 }
        await settle()
        XCTAssertNil(model.imagery.value.landedUnavailable, "one slow answer is not giving up")
        for revision in 1...WorldImageryModel.landingRetries {
            model.geometryPushed(WorldGeometryCoordinates(worldID: "w1", sessionID: "s9", revision: "g\(revision)"),
                                 receiving: true)
            await expect { Stub.count(Self.manifest) == revision + 1 }
            await settle()
        }
        await expect("given up") { model.imagery.value.landedUnavailable == WorldImageryLanding(newer) }
        model.coverageReported(WalkScoped(walk: Self.walk, value: stop), receiving: true)
        await expect("a new landing starts clean") { model.imagery.value.landedTiles[F.landedDigest] != nil }
        XCTAssertNil(model.imagery.value.landedUnavailable)
    }
}
