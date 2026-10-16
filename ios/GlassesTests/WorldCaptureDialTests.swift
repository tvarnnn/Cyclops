//
//  WorldCaptureDialTests.swift
//  GlassesTests
//
//  S1-S3, iOS: the capture dial's model on the
//  15 real coverage blocks, its gap and words, its photos from today's wire
//  (S2) and from the Tower's own bins (S3), the additive B2 fields read
//  fail-closed, the walk cue, the single look-here segment, the photos
//  kept across landings, and the model's record when a landing's photos
//  never come. Round 3 (the owner's call, 2026-10-13): start fixed on top
//  -- the ring never turns between landings -- no sentence of guidance,
//  one quiet footer line, the gap as clock positions for VoiceOver only,
//  the "Recently captured" strip, and the Map tab behind IOS_FOW_MAP_TAB.
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
            XCTAssertEqual(dial.anchor, 0, "\(name): sector zero on top")
        }
        let walk71 = WorldCaptureDial(coverage: try Self.coverage("walk71-h563"))
        XCTAssertEqual(walk71.heroIndex, 0, "the most posed keyframes")
        XCTAssertEqual(walk71.hero?.referenceSegment, 0)
        XCTAssertEqual(walk71.otherPieces.map(\.supportedCount), [5, 3], "the other drawn pieces")
        XCTAssertEqual(Copy.accessibilityLabel(walk71), "Capture dial, Piece 1")
        XCTAssertEqual(walk71.gap?.positions, [6, 7, 8, 9])
        XCTAssertEqual(walk71.lookHere?.positions, [7], "one segment, never the arc")
        XCTAssertEqual(Copy.accessibilityValue(walk71),
                       "8 of 12 directions captured; missing around 6 to 9 o'clock from where you started")
        let walk6 = WorldCaptureDial(coverage: try Self.coverage("walk6-h1090"))
        XCTAssertNil(walk6.gap)
        XCTAssertNil(walk6.lookHere)
        XCTAssertEqual(Copy.accessibilityValue(walk6), "12 of 12 directions captured")
        XCTAssertEqual(Copy.footer(walk6, age: 5), "12 of 12 directions · updated 5 s ago")
        XCTAssertEqual(Copy.accessibilityLabel(WorldCaptureDial(coverage: try Self.coverage("walk7-h236"))),
                       "Capture dial")
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
        XCTAssertNil(dial.lookHere)
        XCTAssertEqual((0..<12).map(dial.evidence(at:)), Array(repeating: .unconfirmed, count: 12))
        XCTAssertEqual(Copy.count(dial), "–")
        XCTAssertEqual(Copy.compactCount(dial), "–")
        XCTAssertEqual(Copy.footer(dial, age: nil), "Waiting for first views")
        XCTAssertEqual(Copy.footer(dial, age: 400), "Waiting for first views")
        XCTAssertEqual(Copy.accessibilityLabel(dial), "Capture dial")
        XCTAssertEqual(Copy.accessibilityValue(dial), Copy.awaitingLine)
        XCTAssertEqual(Panel.captureDial(nil, overlay: nil), .awaiting)
        XCTAssertTrue(Panel.drawsDial(WorldPanelDial(awaiting: true), map: nil))
        XCTAssertFalse(Panel.drawsDial(WorldPanelDial(awaiting: false), map: nil), "absent: today's panel")
    }

    // MARK: Start on top, fixed; the gap as clock positions from start

    /// The owner's round 3: the top is where the piece began, labelled
    /// "start", for good. A new newest view (each landing's `latest`) picks
    /// the hero piece and never turns the ring: every segment, the gap and
    /// the amber outline stay at the same clock position.
    func testTheRingNeverTurnsBetweenLandings() throws {
        let supported = [0, 1, 2, 3, 4, 8, 9, 10]
        let first = dial(supported: supported, latest: 3, motion: 2)
        let second = dial(supported: supported, latest: 9, motion: 2)
        let none = dial(supported: supported)
        XCTAssertEqual(WorldCaptureDialCopy.start, "start")
        for shown in [first, second, none] {
            XCTAssertEqual(shown.anchor, 0)
            XCTAssertEqual((0..<12).map(shown.sector(at:)), Array(0..<12), "start on top: position = sector")
            XCTAssertEqual((0..<12).map(shown.evidence(at:)), (0..<12).map(none.evidence(at:)))
            XCTAssertEqual(shown.gap, none.gap)
            XCTAssertEqual(shown.lookHere, none.lookHere)
            XCTAssertEqual(Copy.accessibilityValue(shown), Copy.accessibilityValue(none))
        }
        XCTAssertEqual(first.lookHere?.positions, [6])

        // Through the panel: two landings of one piece whose latest moves.
        let stop = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.stop))
        let receipt = WorldCoverageReceipt(coverage: stop, keyframesAccepted: 984, clock: nil)
        let anchors = [3, 7, 11].map { sector -> Int in
            let overlay = WorldLandedOverlay(receipt: receipt, thumbnails: [], latest: .init(
                componentReferenceSegment: 0, sector: sector, keyframeID: "s9:\(sector)"))
            return Panel.captureDial(.coverage(receipt), overlay: overlay).anchor
        }
        XCTAssertEqual(anchors, [0, 0, 0])
    }

    /// The value names the largest gap the ring draws, by its first and
    /// last clock positions from start; nothing on screen says it.
    func testTheValueNamesTheLargestGapFromStart() {
        func value(_ supported: [Int], weak: [Int] = []) -> String {
            Copy.accessibilityValue(dial(supported: supported, weak: weak, latest: 0))
        }
        XCTAssertEqual(value([0, 1, 2, 3, 4, 8, 9, 10], weak: [11]),
                       "8 of 12 directions captured; missing around 5 to 7 o'clock from where you started")
        XCTAssertEqual(value(Array(2...9)),
                       "8 of 12 directions captured; missing around 10 to 1 o'clock from where you started",
                       "a gap through the top")
        XCTAssertEqual(value(Array(0..<12).filter { $0 != 7 }),
                       "11 of 12 directions captured; missing around 7 o'clock from where you started")
        XCTAssertEqual(value(Array(1..<12)),
                       "11 of 12 directions captured; missing around 12 o'clock from where you started")
        XCTAssertEqual(value(Array(0..<12).filter { $0 != 3 }, weak: [3]),
                       "11 of 12 directions captured; one view only around 3 o'clock from where you started")
        XCTAssertEqual(value(Array(0..<12)), "12 of 12 directions captured")
        XCTAssertEqual(value([]), "0 of 12 directions captured", "every direction is the gap")
        XCTAssertEqual(Copy.accessibilityValue(dial(supported: [], weak: Array(0..<12))), "0 of 12 directions captured")
    }

    /// One segment -- the gap's centre -- or its middle two from six on;
    /// never the whole arc, and none when the gap is every direction.
    func testTheLookHereIsOneSegmentOrTwoForALongGap() {
        let wrap = dial(supported: Array(2...9))
        XCTAssertEqual(wrap.gap, .init(positions: [10, 11, 0, 1], evidence: .unconfirmed))
        XCTAssertEqual(wrap.lookHere, .init(positions: [11], evidence: .unconfirmed))
        XCTAssertEqual(dial(supported: [0, 1, 2, 3, 4, 8, 9, 10]).lookHere?.positions, [6])
        XCTAssertEqual(dial(supported: Array(0...6)).lookHere?.positions, [9], "5 long: its centre")
        XCTAssertEqual(dial(supported: Array(0...5)).lookHere?.positions, [8, 9], "6 long: the middle two")
        XCTAssertEqual(dial(supported: Array(0...4)).lookHere?.positions, [7, 8], "7 long: two")
        XCTAssertEqual(dial(supported: [0]).lookHere?.positions, [5, 6], "11 long: two")
        XCTAssertNil(dial(supported: []).lookHere, "12: no direction is better")
        XCTAssertNil(dial(supported: Array(0..<12)).lookHere)
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
        XCTAssertEqual(one.lookHere, .init(positions: [3], evidence: .weak))
        let none = dial(supported: [], weak: Array(0..<12), latest: 0)
        XCTAssertNil(none.lookHere)
    }

    /// The footer is one quiet line: the count and the stamp, "catching up"
    /// past the dial's own 90 s (the map's 30 s untouched), the count alone
    /// for an undated map, "Waiting for first views" before the first
    /// landing. A stale map keeps its amber outline.
    func testTheFooterIsOneQuietLine() {
        XCTAssertEqual(Copy.staleAfter, 90)
        XCTAssertEqual(WorldCoverageReceipt.staleAfter, 30, "the map's stale label threshold")
        let walking = dial(supported: [0, 1, 2, 3, 4, 8, 9, 10, 11], latest: 0)
        XCTAssertEqual(Copy.footer(walking, age: 45), "9 of 12 directions · updated 45 s ago")
        XCTAssertEqual(Copy.footer(walking, age: 90), "9 of 12 directions · updated 90 s ago", "at exactly 90 s")
        XCTAssertEqual(Copy.footer(walking, age: 90.5), "9 of 12 directions · catching up")
        XCTAssertEqual(Copy.footer(walking, age: 150), "9 of 12 directions · catching up", "no seconds counting up")
        XCTAssertEqual(Copy.footer(walking, age: nil), "9 of 12 directions", "undated: the count alone")
        XCTAssertEqual(Copy.footer(.awaiting, age: nil), "Waiting for first views")
        XCTAssertFalse(Copy.isCatchingUp(nil))
        XCTAssertTrue(Copy.isCatchingUp(91))
        XCTAssertEqual(walking.lookHere?.positions, [6], "the amber outline never depends on the age")
    }

    // MARK: Accessibility (§4)

    func testTheDialIsOneElementWithTheSpecsValue() {
        let spec = dial(supported: [0, 1, 2, 3, 4, 8, 9, 10], weak: [11], latest: 0)
        XCTAssertEqual(Copy.accessibilityValue(spec),
                       "8 of 12 directions captured; missing around 5 to 7 o'clock from where you started")
        XCTAssertEqual(Copy.accessibilityLabel(spec), "Capture dial")
        XCTAssertEqual(Copy.accessibilityValue(dial(supported: Array(0..<12), latest: 3)),
                       "12 of 12 directions captured")
        XCTAssertEqual(Copy.compactCount(spec), "8/12")
        XCTAssertEqual(Copy.photo(at: 0), "Photo at 12 o'clock")
        XCTAssertEqual(Copy.photo(at: 3), "Photo at 3 o'clock")
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

    // MARK: S3: the Tower's bins, the hero piece, the walk cue (§3.3, §7)

    /// `latest` picks the piece in front; start stays on top (round 3).
    func testLatestPicksItsPieceAndNeverTheTop() throws {
        let latest = WorldCaptureDial(components: [Self.piece(supported: [0, 1, 2, 5])], stationsOmitted: 0,
                                      newest: .init(referenceSegment: 0, sector: 5))
        XCTAssertEqual(latest.anchor, 0)
        XCTAssertEqual(latest.sector(at: 0), 0)
        XCTAssertEqual(latest.position(of: 5), 5)
        XCTAssertEqual(latest.evidence(at: 5), .supported)
        XCTAssertEqual(Copy.accessibilityLabel(latest), "Capture dial")

        // walk 7.1: latest names piece 2, which has fewer views: it is the hero.
        let coverage = try Self.coverage("walk71-h563")
        let second = WorldCaptureDial(coverage: coverage,
                                      newest: .init(referenceSegment: 13, sector: 1))
        XCTAssertEqual(second.heroIndex, 1)
        XCTAssertEqual(second.hero?.supportedCount, 5)
        XCTAssertEqual(Copy.accessibilityLabel(second), "Capture dial, Piece 2")
        XCTAssertEqual(second.otherPieces.map(\.name), ["Piece 1", "Piece 3"])
    }

    /// The coordinator's three S3 cases: `latest` null, `latest` naming no
    /// drawn piece, and `latest` at an undrawn station (a valid anchor, no
    /// photo, its segment as the drawn masks say).
    func testANullForeignOrUndrawnLatestIsHandled() throws {
        let coverage = try Self.coverage("walk71-h563")
        let none = WorldCaptureDial(coverage: coverage, newest: nil, motionSector: 3)
        XCTAssertEqual(none.anchor, 0)
        XCTAssertEqual(none.heroIndex, 0, "most views")
        XCTAssertNil(none.motionSector, "no walk cue without the latest it belongs to")
        let foreign = WorldCaptureDial(coverage: coverage, newest: .init(referenceSegment: 99, sector: 4),
                                       motionSector: 3)
        XCTAssertEqual(foreign.heroIndex, 0, "a piece not drawn here names nothing")
        XCTAssertEqual(foreign.anchor, 0)
        XCTAssertNil(foreign.motionSector)
        // walk 7.1 piece 1 has nothing at sectors 6-9 among its drawn
        // stations; the newest view's station was not drawn.
        let undrawn = WorldCaptureDial(coverage: coverage,
                                       newest: .init(referenceSegment: 0, sector: 7),
                                       photos: [.init(digest: "p7", referenceSegment: 0, sector: 7),
                                                .init(digest: "p0", referenceSegment: 0, sector: 0)])
        XCTAssertEqual(undrawn.heroIndex, 0)
        XCTAssertEqual(undrawn.anchor, 0, "start on top")
        XCTAssertEqual(undrawn.evidence(at: 7), .unconfirmed, "drawn as the drawn masks say")
        XCTAssertEqual(undrawn.photosBySector, [0: "p0"], "no photo on it")
        XCTAssertEqual(undrawn.gap?.positions, [6, 7, 8, 9])
    }

    /// A stale `latest` -- the Tower kept the previous landing's block --
    /// never counts for the CURRENT receipt's dial: the panel draws B2 only
    /// over the receipt it was made for, and then falls back (§3.3).
    func testAStaleLatestNeverCountsForTheCurrentReceipt() throws {
        let stop = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.stop))
        let receipt = WorldCoverageReceipt(coverage: stop, keyframesAccepted: 984, clock: nil)
        let latest = WorldImageryLanded.Latest(componentReferenceSegment: 0, sector: 3, keyframeID: "s9:120")
        let image = UIImage(data: FOWImageryFixtures.jpeg())!
        let overlay = WorldLandedOverlay(receipt: receipt, thumbnails: [], dialEntries: [
            WorldDialThumbnail(digest: "d1", image: image, referenceSegment: 0, heading: nil, sector: 2),
        ], latest: latest, motionSector: 1)
        let current = Panel.captureDial(.coverage(receipt), overlay: Panel.landed(overlay, over: .coverage(receipt)))
        XCTAssertEqual(current.anchor, 0, "start on top, whatever the latest")
        XCTAssertEqual(current.motionSector, 1)
        XCTAssertEqual(current.photosBySector, [2: "d1"])

        let newer = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.stop
            .replacingOccurrences(of: "1791240313.", with: "1791240320.")
            .replacingOccurrences(of: "g-stop-984", with: "g-stop-990")))
        let next = WorldCoverageReceipt(coverage: newer, keyframesAccepted: 990, clock: nil)
        let stale = Panel.captureDial(.coverage(next), overlay: Panel.landed(overlay, over: .coverage(next)))
        XCTAssertEqual(stale.anchor, 0)
        XCTAssertNil(stale.motionSector, "the previous landing's latest")
        XCTAssertEqual(stale.photosBySector, [:], "nothing carries over")
        let fixture = Panel.captureDial(.fixture(receipt), overlay: Panel.landed(overlay, over: .fixture(receipt)))
        XCTAssertNil(fixture.motionSector, "never over the DEBUG fixture")
        XCTAssertEqual(fixture.photosBySector, [:])
    }

    /// The walk cue is the tick alone: the value never names it.
    /// The dial still decodes `landed.motion_sector` fail-closed, but draws no
    /// travel arrow: the owner wants no directional guidance (round 3).
    func testTheWalkCueIsDecodedButNotADirection() {
        let cue = dial(supported: [0, 1, 2], latest: 5, motion: 7)
        XCTAssertEqual(cue.motionSector, 7)
        XCTAssertEqual(Copy.accessibilityValue(cue), Copy.accessibilityValue(dial(supported: [0, 1, 2])))
        XCTAssertNil(dial(supported: [0], latest: 0, motion: nil).motionSector, "null: no tick")
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
        XCTAssertEqual(dial.position(of: 4), 4, "the newest view's own photo stays at its own sector: start on top")
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
        // The dial then falls back: the biggest piece, no walk cue.
        let stop = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.stop))
        let receipt = WorldCoverageReceipt(coverage: stop, keyframesAccepted: nil, clock: nil)
        let landed = try XCTUnwrap(Self.manifest(extra: #","latest":{"sector":3},"motion_sector":"2""#)?.landed.value)
        let overlay = WorldLandedOverlay(receipt: receipt, thumbnails: [], latest: landed.latest,
                                         motionSector: landed.motionSector)
        let dial = Panel.captureDial(.coverage(receipt), overlay: overlay)
        XCTAssertEqual(dial.anchor, 0)
        XCTAssertNil(dial.motionSector)
    }

    // MARK: The words never claim what the receipt does not (§8.5)

    func testTheDialsWordsNeverClaimWhatTheReceiptDoesNot() throws {
        // The owner's demo decisions: "Room coverage" (2026-10-12), and
        // "missing around 7 to 9 o'clock from where you started" in the
        // VoiceOver value (2026-10-13), the top being fixed and labelled.
        let banned = ["hole", "covered", "complete", "surface", "texture", "wall", "floor",
                      "current", "you are", "live", "scan", "unseen", "nothing", "%", "everything",
                      "position", "estimate", "remaining", "done", "proved", "real time", "just now",
                      "not captured", "never captured", "adequate", " go ", "north", "compass", "stale", "o'clock to"]
        // Round 3: no sentence of guidance anywhere on screen.
        let guidance = ["look", "turn", "behind", "left", "right", "ahead", "fills in", "slowly", "fill the ring",
                        "nice", "around", "…", "keep ", "need"]
        var visible = [Copy.dialName, Copy.title, Copy.tabLabel, Copy.awaitingLine, Copy.ofTwelve,
                       Copy.captured, Copy.legendCaptured, Copy.legendOneView, Copy.legendNotYet, Copy.start,
                       Copy.recentTitle, Copy.closePhoto, Copy.tabName(.dial), Copy.tabName(.map), Copy.more(4),
                       Copy.healthRow, Copy.finishStamp(headline: "Improving", elapsed: "7 min since you stopped"),
                       Copy.finishStamp(headline: nil, elapsed: nil)]
        var spoken = [Copy.recentPhoto(1, of: 4)]
        var dials = [WorldCaptureDial.awaiting]
        for name in ["walk7-h52", "walk7-h236", "walk71-h563", "walk6-h1090", "walk71-h104"] {
            let coverage = try Self.coverage(name)
            dials.append(WorldCaptureDial(coverage: coverage))
            dials.append(WorldCaptureDial(coverage: coverage, newest: .init(referenceSegment: coverage.components[0].referenceSegment, sector: 3),
                motionSector: 5))
        }
        dials += [dial(supported: []), dial(supported: [], weak: Array(0..<12)), dial(supported: [1], weak: [2, 3])]
        for dial in dials {
            visible += [2, 31, 95, 400, nil].map { Copy.footer(dial, age: $0) }
            visible += [Copy.count(dial), Copy.compactCount(dial)]
            visible += dial.pieces.map(Copy.miniLabel)
            spoken += [Copy.accessibilityValue(dial), Copy.accessibilityLabel(dial)]
            spoken += dial.pieces.map(Copy.miniAccessibilityLabel)
            spoken += (0..<12).map(Copy.photo(at:))
        }
        XCTAssertGreaterThan(visible.count + spoken.count, 100)
        for word in visible + spoken {
            for phrase in banned {
                XCTAssertFalse(word.lowercased().contains(phrase), "\"\(word)\" says \"\(phrase)\"")
            }
        }
        for word in visible {
            for phrase in guidance {
                XCTAssertFalse(word.lowercased().contains(phrase), "guidance on screen: \"\(word)\"")
            }
        }
    }

    // MARK: The photos, kept across landings

    private static func thumb(_ digest: String, reference: Int = 0, sector: Int) -> WorldDialThumbnail {
        WorldDialThumbnail(digest: digest, image: UIImage(), referenceSegment: reference, heading: nil, sector: sector)
    }

    private func shown(_ dial: WorldCaptureDial, _ entries: [WorldDialThumbnail],
                       keeping kept: WorldCaptureDialPhotos? = nil) -> WorldCaptureDialPhotos {
        WorldCaptureDialPhotos(dial: dial, entries: entries, keeping: kept)
    }

    /// A newer landing of the same hero with no tile for a captured sector
    /// keeps the photo last shown there, instead of a flat wedge; its own
    /// photos win, and a sector no longer captured keeps nothing.
    func testANewLandingWithNoTileForASectorKeepsThePreviousPhoto() {
        let a = Self.thumb("a", sector: 0), b = Self.thumb("b", sector: 1)
        let first = shown(dial(supported: [0, 1, 2], latest: 0, photos: WorldCaptureDial.photos([a, b])), [a, b])
        XCTAssertEqual(first.referenceSegment, 0)
        XCTAssertEqual(first.bySector.mapValues(\.digest), [0: "a", 1: "b"])

        let c = Self.thumb("c", sector: 2)
        let second = shown(dial(supported: [0, 1, 2, 3], latest: 2, photos: WorldCaptureDial.photos([c])), [c],
                           keeping: first)
        XCTAssertEqual(second.bySector.mapValues(\.digest), [0: "a", 1: "b", 2: "c"], "kept, plus the new one")

        let none = shown(dial(supported: [0, 1, 2, 3], latest: 2), [], keeping: second)
        XCTAssertEqual(none.bySector.mapValues(\.digest), [0: "a", 1: "b", 2: "c"], "a landing with no photos at all")

        let d = Self.thumb("d", sector: 0)
        let newer = shown(dial(supported: [0, 1, 2, 3], latest: 0, photos: WorldCaptureDial.photos([d])), [d],
                          keeping: none)
        XCTAssertEqual(newer.bySector.mapValues(\.digest), [0: "d", 1: "b", 2: "c"], "the newest photo wins")

        let lost = shown(dial(supported: [0, 2, 3], latest: 0), [], keeping: newer)
        XCTAssertEqual(lost.bySector.mapValues(\.digest), [0: "d", 2: "c"], "never on a segment not captured now")
        XCTAssertEqual(shown(dial(supported: [0, 1], weak: [2], latest: 0), [], keeping: newer)
                        .bySector.mapValues(\.digest), [0: "d", 1: "b", 2: "c"], "one view still holds its photo")
    }

    /// Another hero piece clears the kept photos; coming back finds none.
    func testAHeroChangeClearsTheKeptPhotos() {
        let a = Self.thumb("a", sector: 0)
        let first = shown(dial(supported: [0, 1], latest: 0, photos: WorldCaptureDial.photos([a])), [a])
        let other = WorldCaptureDial(components: [Self.piece(0, posed: 50, supported: [0, 1]),
                                                  Self.piece(5, posed: 80, supported: [0, 3])],
                                     stationsOmitted: 0, newest: .init(referenceSegment: 5, sector: 3))
        let switched = shown(other, [], keeping: first)
        XCTAssertEqual(switched.referenceSegment, 5)
        XCTAssertEqual(switched.bySector, [:], "piece 1's photo never on piece 2")
        let back = shown(dial(supported: [0, 1], latest: 0), [], keeping: switched)
        XCTAssertEqual(back.bySector, [:], "dropped, not parked")
        XCTAssertEqual(shown(.awaiting, [], keeping: first), WorldCaptureDialPhotos.empty)
    }

    /// The panel reads the kept photos only beside a report of their own
    /// walk: another walk, or a report naming none (a reset), starts empty.
    func testAWalkChangeOrResetDropsTheKeptPhotos() throws {
        let stop = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.stop))
        let map = WorldPanelMapSource.coverage(WorldCoverageReceipt(coverage: stop, keyframesAccepted: nil, clock: nil))
        let walkA = WorldFinishWalk(worldID: "w1", sessionID: "s9")
        let walkB = WorldFinishWalk(worldID: "w1", sessionID: "s10")
        let kept = WalkScoped(walk: walkA, value: WorldCaptureDialPhotos(
            referenceSegment: 0, frameRevision: stop.frameRevision,
            bySector: [2: Self.thumb("k2", sector: 2), 7: Self.thumb("k7", sector: 7)]))
        let same = Panel.shownDial(map, overlay: nil, kept: kept, beside: WalkScoped(walk: walkA, value: ()))
        XCTAssertEqual(same.walk, walkA)
        XCTAssertEqual(same.value.photos.bySector.mapValues(\.digest), [2: "k2"], "sector 7 is not captured")
        XCTAssertEqual(Panel.shownDial(map, overlay: nil, kept: kept, beside: WalkScoped(walk: walkB, value: ()))
                        .value.photos.bySector, [:], "another walk")
        XCTAssertEqual(Panel.shownDial(map, overlay: nil, kept: kept, beside: WalkScoped(walk: nil, value: ()))
                        .value.photos.bySector, [:], "a reset names no walk")
        XCTAssertEqual(Panel.shownDial(nil, overlay: nil, kept: kept, beside: WalkScoped(walk: walkA, value: ()))
                        .value, WorldPanelDialShown(dial: .awaiting, photos: .empty), "before the first landing")
    }

    // MARK: Round 2: after Stop, the mini dials, Capture health

    /// After Stop the dial shows the photos already shown live for THIS walk
    /// (no rule purges them at Stop: the landed tiles live until the world is
    /// left, forgotten or pinned); never beside another walk's report, and
    /// never another hero piece's photos -- those sectors stay colour.
    func testAfterStopTheDialKeepsTheWalksPhotosAndNoOtherWalks() {
        let walkA = WorldFinishWalk(worldID: "w1", sessionID: "s9")
        let walkB = WorldFinishWalk(worldID: "w1", sessionID: "s10")
        let a = Self.thumb("a", sector: 0), b = Self.thumb("b", sector: 1)
        let last = dial(supported: [0, 1, 2], latest: 0, photos: WorldCaptureDial.photos([a, b]))
        let photos = WorldCaptureDialPhotos(dial: last, entries: [a, b])
        let final = WalkScoped(walk: walkA, value: WorldPanelDialShown(dial: last, photos: photos))

        let shown = Panel.postStopDial(final, beside: WalkScoped(walk: walkA, value: ()))
        XCTAssertEqual(shown?.dial, last)
        XCTAssertEqual(shown?.photos.bySector.mapValues(\.digest), [0: "a", 1: "b"], "the photos shown live")
        XCTAssertNil(shown?.photos.bySector[2], "a sector never shown a photo stays colour")
        XCTAssertEqual(Copy.accessibilityValue(last),
                       "3 of 12 directions captured; missing around 3 to 11 o'clock from where you started",
                       "after Stop the same value: the dimmed ring draws the same gap")
        XCTAssertNil(Panel.postStopDial(final, beside: WalkScoped(walk: walkB, value: ())), "another walk")
        XCTAssertNil(Panel.postStopDial(final, beside: WalkScoped(walk: nil, value: ())), "a reset")
        XCTAssertNil(Panel.postStopDial(nil, beside: WalkScoped(walk: walkA, value: ())), "no dial was drawn")

        let foreign = WalkScoped(walk: walkA, value: WorldPanelDialShown(
            dial: last, photos: WorldCaptureDialPhotos(referenceSegment: 5, bySector: [0: a])))
        XCTAssertEqual(Panel.postStopDial(foreign, beside: WalkScoped(walk: walkA, value: ()))?.photos, .empty,
                       "another piece's photos: colour")
    }

    /// While a newer receipt's own landing is not in (no overlay), the last
    /// landing's newest view holds: the hero, the top and the hint stay, the
    /// kept photos stay on the hero, and nothing recorded loses them. A new
    /// frame, another walk, or a report naming none holds nothing.
    func testTheNilOverlayWindowNeitherFlipsTheHeroNorWipesTheKeptPhotos() throws {
        let mid = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.mid))
        let map = WorldPanelMapSource.coverage(WorldCoverageReceipt(coverage: mid, keyframesAccepted: nil, clock: nil))
        let walkA = WorldFinishWalk(worldID: "w1", sessionID: "s9")
        let report = WalkScoped(walk: walkA, value: ())
        let held = WalkScoped(walk: walkA, value: WorldPanelHeldNewest(
            newest: .init(referenceSegment: 17, sector: 0), frameRevision: mid.frameRevision))
        let kept = WalkScoped(walk: walkA, value: WorldCaptureDialPhotos(
            referenceSegment: 17, frameRevision: mid.frameRevision,
            bySector: [0: Self.thumb("k", reference: 17, sector: 0)]))

        let window = Panel.shownDial(map, overlay: nil, kept: kept, held: held, beside: report)
        XCTAssertEqual(window.value.dial.hero?.referenceSegment, 17, "the hero holds")
        XCTAssertEqual(window.value.dial.anchor, 0, "start on top throughout")
        XCTAssertEqual(window.value.photos.bySector.mapValues(\.digest), [0: "k"], "the kept photo stays")

        let unheld = Panel.shownDial(map, overlay: nil, kept: kept, beside: report)
        XCTAssertEqual(unheld.value.dial.hero?.referenceSegment, 0, "without the hold: the biggest piece")
        let otherFrame = WalkScoped(walk: walkA, value: WorldPanelHeldNewest(
            newest: .init(referenceSegment: 17, sector: 0), frameRevision: mid.frameRevision + 1))
        XCTAssertEqual(Panel.shownDial(map, overlay: nil, kept: kept, held: otherFrame, beside: report)
                        .value.dial.hero?.referenceSegment, 0, "never across a new frame")
        XCTAssertEqual(Panel.shownDial(map, overlay: nil, kept: kept, held: held,
                                       beside: WalkScoped(walk: nil, value: ())).value.dial.hero?.referenceSegment, 0)

        // Even a flipped, photo-less dial drawn in the window never replaces
        // the walk's record that holds photos; a landing's own dial does.
        let recorded = Panel.record(window, fromLanding: true, over: nil)
        XCTAssertEqual(Panel.record(unheld, fromLanding: false, over: recorded), recorded, "nothing lost")
        XCTAssertEqual(Panel.record(unheld, fromLanding: true, over: recorded), unheld, "a landing's own dial")
        XCTAssertEqual(Panel.record(unheld, fromLanding: false, over: nil), unheld, "nothing to keep")
        let reset = WalkScoped(walk: nil, value: unheld.value)
        XCTAssertEqual(Panel.record(reset, fromLanding: true, over: recorded), recorded,
                       "a report naming no walk never overwrites the walk's record")
        XCTAssertNil(Panel.record(reset, fromLanding: true, over: recorded)?.value(beside: WalkScoped(walk: nil,
                                                                                                    value: ())),
                     "and a reset still shows none of it")
        XCTAssertNil(Panel.heldNewest(map, overlay: nil, beside: report), "no landing in hand: nothing new held")
    }

    /// A re-solve may turn a piece's frame: kept photos never cross a new
    /// `frame_revision`, even for the same piece.
    func testANewFrameRevisionDropsTheKeptPhotos() {
        let a = Self.thumb("a", sector: 0)
        let frame1 = WorldCaptureDial(components: [Self.piece(supported: [0, 1])], stationsOmitted: 0,
                                      frameRevision: 1, newest: .init(referenceSegment: 0, sector: 0),
                                      photos: WorldCaptureDial.photos([a]))
        let first = WorldCaptureDialPhotos(dial: frame1, entries: [a])
        XCTAssertEqual(first.frameRevision, 1)
        XCTAssertEqual(first.bySector.mapValues(\.digest), [0: "a"])
        let same = WorldCaptureDial(components: [Self.piece(supported: [0, 1, 2])], stationsOmitted: 0,
                                    frameRevision: 1, newest: .init(referenceSegment: 0, sector: 2))
        XCTAssertEqual(WorldCaptureDialPhotos(dial: same, entries: [], keeping: first).bySector.mapValues(\.digest),
                       [0: "a"], "the same frame keeps")
        let turned = WorldCaptureDial(components: [Self.piece(supported: [0, 1, 2])], stationsOmitted: 0,
                                      frameRevision: 2, newest: .init(referenceSegment: 0, sector: 2))
        let dropped = WorldCaptureDialPhotos(dial: turned, entries: [], keeping: first)
        XCTAssertEqual(dropped.bySector, [:], "a new frame drops them")
        XCTAssertEqual(dropped.frameRevision, 2)
    }

    /// A piece with nothing captured is never drawn as a mini dial: no "0/12".
    func testNoMiniDialSaysZeroOfTwelve() {
        let pieces = WorldCaptureDial(components: [Self.piece(0, posed: 90, supported: [0, 1, 2]),
                                                   Self.piece(3, posed: 20, supported: []),
                                                   Self.piece(5, posed: 10, supported: [], weak: [4]),
                                                   Self.piece(7, posed: 5, supported: [6])],
                                      stationsOmitted: 0)
        XCTAssertEqual(pieces.otherPieces.count, 3)
        XCTAssertEqual(pieces.miniPieces.map(\.name), ["Piece 4"], "only a piece with a captured direction")
        XCTAssertFalse(pieces.miniPieces.map(Copy.miniLabel).contains { $0.hasSuffix(" 0/12") })
        XCTAssertEqual(dial(supported: [0, 1]).miniPieces, [], "one piece: none")
    }

    /// Capture health is one collapsed row that opens to the same block.
    func testCaptureHealthIsOneCollapsedRow() {
        XCTAssertEqual(Copy.healthRow, "Capture health")
    }

    /// After Stop the footer's second line is the headline and the time
    /// since Stop; the first is the stage line.
    func testAfterStopTheFooterSaysTheStatus() {
        XCTAssertEqual(Copy.finishStamp(headline: "Improving", elapsed: "7 min since you stopped"),
                       "Improving · 7 min since you stopped")
        XCTAssertEqual(Copy.finishStamp(headline: nil, elapsed: "Under 1 min since you stopped"),
                       "Under 1 min since you stopped")
        XCTAssertEqual(Copy.finishStamp(headline: nil, elapsed: nil), " ", "never empty: two lines")
        let block = Panel.FinishingBlock(showsSpinner: true, line: .finalPlacement, stoppedAt: nil, detail: nil)
        XCTAssertEqual(Panel.postStopStatus(block), "Placing and checking images")
        XCTAssertEqual(Panel.postStopStatus(block.reachable(false)), WorldPanelCopy.finishingButOffline)
        XCTAssertEqual(Panel.postStopStatus(.init(showsSpinner: true, line: nil, stoppedAt: nil,
                                                  detail: "A long sentence of the canvas's own.")),
                       WorldFinishCopy.finishing, "the short sentence: two lines")
    }

    // MARK: Round 3: the Map tab's switch, the recent strip

    /// `IOS_FOW_MAP_TAB=on` exactly shows the Dial|Map control; unset (the
    /// demo build) or anything else hides it, live and after Stop.
    func testTheMapTabSwitchHidesTheToggleUnlessOn() throws {
        XCTAssertFalse(WorldDialMapTabSwitch(environment: [:], info: [:]).on, "unset: hidden")
        XCTAssertFalse(WorldDialMapTabSwitch(environment: ["IOS_FOW_MAP_TAB": "ON"]).on)
        XCTAssertFalse(WorldDialMapTabSwitch(environment: [:], info: ["GlassesFOWMapTab": ""]).on)
        XCTAssertTrue(WorldDialMapTabSwitch(environment: ["IOS_FOW_MAP_TAB": "on"]).on)
        XCTAssertTrue(WorldDialMapTabSwitch(environment: [:], info: ["GlassesFOWMapTab": "on"]).on)
        XCTAssertNotNil(Bundle.main.infoDictionary?["GlassesFOWMapTab"], "the map tab's home-screen key")
        XCTAssertEqual(WorldPanelDial().mapTab, false, "the default is hidden")
        let stop = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.stop))
        let map = WorldPanelMapSource.coverage(WorldCoverageReceipt(coverage: stop, keyframesAccepted: nil, clock: nil))
        XCTAssertFalse(Panel.showsMapToggle(WorldPanelDial(), map: map, stopped: false), "unset: no control")
        XCTAssertFalse(Panel.showsMapToggle(WorldPanelDial(), map: nil, stopped: true), "unset: none after Stop")
        XCTAssertTrue(Panel.showsMapToggle(WorldPanelDial(mapTab: true), map: map, stopped: false))
        XCTAssertTrue(Panel.showsMapToggle(WorldPanelDial(mapTab: true), map: nil, stopped: true))
        XCTAssertFalse(Panel.showsMapToggle(WorldPanelDial(mapTab: true), map: nil, stopped: false), "no map yet")
    }

    /// The strip: at most four, newest first; a landing's new photos lead,
    /// the newest placed view's own photo first; a photo met before in the
    /// walk is never new again.
    func testTheRecentStripKeepsTheFourNewestOnce() {
        func tiles(_ sectors: [Int], reference: Int = 0) -> [WorldDialThumbnail] {
            sectors.map { Self.thumb("d\($0)", reference: reference, sector: $0) }
        }
        let first = WorldDialRecent().adding(tiles([0, 1, 2]),
                                             newest: .init(componentReferenceSegment: 0, sector: 2, keyframeID: "k"))
        XCTAssertEqual(first.photos.map(\.digest), ["d2", "d0", "d1"], "the newest view's photo leads")
        let second = first.adding(tiles([0, 1, 2, 3, 4]), newest: nil)
        XCTAssertEqual(second.photos.map(\.digest), ["d3", "d4", "d2", "d0"], "four, the new ones first")
        let again = second.adding(tiles([0, 1, 2, 3, 4]), newest: .init(componentReferenceSegment: 0, sector: 1,
                                                                         keyframeID: "k"))
        XCTAssertEqual(again, second, "a re-sent landing adds nothing; d1 is not new again")
        let other = again.adding(tiles([5], reference: 7))
        XCTAssertEqual(other.photos.map(\.digest), ["d5", "d3", "d4", "d2"], "any piece of the walk")
        XCTAssertLessThanOrEqual(other.photos.count, WorldDialRecent.limit)
        XCTAssertEqual(WorldDialRecent.limit, 4)
        XCTAssertEqual(WorldDialRecent().adding([]).photos, [])
        XCTAssertEqual(Copy.recentPhoto(2, of: 4), "Photo 2 of 4")
        XCTAssertEqual(Copy.recentTitle, "Recently captured")
    }

    /// Walk-scoped: another walk starts its own strip, and the panel reads a
    /// strip only beside a report of its own walk.
    func testTheRecentStripIsTheWalksOwn() throws {
        let stop = try XCTUnwrap(FOWFixtures.coverage(FOWFixtures.stop))
        let receipt = WorldCoverageReceipt(coverage: stop, keyframesAccepted: nil, clock: nil)
        let walkA = WorldFinishWalk(worldID: "w1", sessionID: "s9")
        let walkB = WorldFinishWalk(worldID: "w1", sessionID: "s10")
        func landing(_ walk: WorldFinishWalk, _ digests: [String]) -> WalkScoped<WorldLandedOverlay> {
            WalkScoped(walk: walk, value: WorldLandedOverlay(
                receipt: receipt, thumbnails: [],
                dialEntries: digests.enumerated().map { Self.thumb($0.element, sector: $0.offset) }))
        }
        let a1 = Panel.recent(nil, adding: landing(walkA, ["a", "b"]))
        XCTAssertEqual(a1.walk, walkA)
        let a2 = Panel.recent(a1, adding: landing(walkA, ["a", "b", "c"]))
        XCTAssertEqual(a2.value.photos.map(\.digest), ["c", "a", "b"])
        let b1 = Panel.recent(a2, adding: landing(walkB, ["x"]))
        XCTAssertEqual(b1.walk, walkB)
        XCTAssertEqual(b1.value.photos.map(\.digest), ["x"], "another walk starts empty")
        XCTAssertNil(a2.value(beside: WalkScoped(walk: walkB, value: ())), "never beside another walk")
        XCTAssertNil(a2.value(beside: WalkScoped(walk: nil, value: ())), "nor a reset")
    }
}

// MARK: - The model's record when a landing's photos never come (§3.4)

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
        WorldImageryModel(client: WorldImageryClient(baseURL: URL(string: "http://tower.test")!,
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
        // Heartbeats of one revision: each retries B2, and the first also
        // asks B1 once (the stages share the manifest route).
        for attempt in 1...WorldImageryModel.landingRetries {
            model.geometryPushed(WorldGeometryCoordinates(worldID: "w1", sessionID: "s9", revision: "g1"),
                                 receiving: true)
            await expect { Stub.count(Self.manifest) == attempt + 2 }
            await settle()
        }
        await expect("given up") { model.imagery.value.landedUnavailable == WorldImageryLanding(newer) }
        model.coverageReported(WalkScoped(walk: Self.walk, value: stop), receiving: true)
        await expect("a new landing starts clean") { model.imagery.value.landedTiles[F.landedDigest] != nil }
        XCTAssertNil(model.imagery.value.landedUnavailable)
    }
}
