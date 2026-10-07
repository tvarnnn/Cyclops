//
//  WorldImageryTests.swift
//  GlassesTests
//
//  FOW v1.1 B (FOW-V11B-WIRE-AND-IOS-SPEC-20261007): the manifest read
//  fail-closed (acceptance 1), the §4 fixtures paired with a §9 status
//  (acceptance 5), placement on the v1 grid, OFF asks nothing, and no walk's
//  or landing's imagery reaches another.
//

import SwiftUI
import UIKit
import XCTest
@testable import Glasses

/// §4's fixtures. The spec prints each digest elided (`a3f1...e02c`), which
/// is not `content_digest()`'s 32 hex: these keep its first and last four
/// characters and fill the middle, so the fixture's one entry is a real one.
enum FOWImageryFixtures {
    static let liveDigest = "a3f1" + String(repeating: "0", count: 24) + "e02c"
    static let landedDigest = "7cde" + String(repeating: "0", count: 24) + "1190"

    static let live = #"{"version":1,"live":{"segment_index":812,"entries":[{"keyframe_id":"s9:4417","digest":"\#(liveDigest)","redaction":"redacted","pose_source":"segment_local","component_reference_segment":null,"segment_index":812}]}}"#
    static let landed = #"{"version":1,"landed":{"solved_at":1791240313.0,"geometry_revision":"g-stop-984","frame_revision":1,"entries":[{"keyframe_id":"s9:120","digest":"\#(landedDigest)","redaction":"redacted","pose_source":"landed_global_solve","component_reference_segment":0,"segment_index":null}]}}"#
    /// §4 verbatim, its digests elided.
    static let liveAsPrinted = #"{"version":1,"live":{"segment_index":812,"entries":[{"keyframe_id":"s9:4417","digest":"a3f1...e02c","redaction":"redacted","pose_source":"segment_local","component_reference_segment":null,"segment_index":812}]}}"#

    static func json(_ text: String) -> Any {
        // swiftlint:disable:next force_try
        try! JSONSerialization.jsonObject(with: Data(text.utf8))
    }

    static func manifest(_ text: String) -> WorldImageryManifest? {
        WorldImageryManifest(json: json(text))
    }

    /// A 160 x 90 JPEG, well under 16 KiB.
    static func jpeg(_ color: UIColor = .systemTeal, size: CGSize = CGSize(width: 160, height: 90)) -> Data {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        return UIGraphicsImageRenderer(size: size, format: format).jpegData(withCompressionQuality: 0.6) { context in
            color.setFill()
            context.fill(CGRect(origin: .zero, size: size))
        }
    }

    /// Segment 0 placed in reference 0, frame 1, identity, with `s9:120` at
    /// `(3, -1, 0)` looking along +x: in §9's stop grid (origin 0, forward
    /// +y, up +z, so right +x; cell 2) that is column 3.5, row 5.5 -- the
    /// supported station (3, 5) -- looking 90 degrees clockwise.
    static func chunk(keyframe: String = "s9:120", translation: [Double] = [3, -1, 0], reference: Int = 0,
                      frame: Int = 1) -> WorldSegmentChunk {
        let transform = WorldTransform(rotationWXYZ: [1, 0, 0, 0], translation: [0, 0, 0], scale: 1,
                                       referenceSegment: reference, frameRevision: frame)
        let half = 0.5.squareRoot()
        return WorldSegmentChunk(
            segmentIndex: 0, contentHash: "h0", registered: true,
            placement: WorldPlacementFields(registered: true, state: nil, refusalReason: nil, transform: transform,
                                            placementHash: "p0"),
            poses: [WorldPose(keyframeID: keyframe, status: "solved", degeneracy: "",
                              rotation: [half, 0, half, 0], translation: translation)],
            points: [], pointsSent: 0, pointsTotal: 0, pointSampling: "none")
    }
}

@MainActor
final class WorldImageryTests: XCTestCase {
    private typealias F = FOWImageryFixtures

    private var stop: WorldCoverage { FOWFixtures.coverage(FOWFixtures.stop)! }
    private var mid: WorldCoverage { FOWFixtures.coverage(FOWFixtures.mid)! }

    // MARK: Acceptance 1: decode, fail closed

    func testBothFixturesDecode() throws {
        let live = try XCTUnwrap(F.manifest(F.live))
        XCTAssertEqual(live.landed, .off, "landed absent: its switch is off")
        let section = try XCTUnwrap(live.live.value)
        XCTAssertEqual(section.segmentIndex, 812)
        XCTAssertEqual(section.entries, [WorldImageryEntry(keyframeID: "s9:4417", digest: F.liveDigest,
                                                           componentReferenceSegment: nil, segmentIndex: 812)])

        let landed = try XCTUnwrap(F.manifest(F.landed))
        XCTAssertEqual(landed.live, .off, "live absent: its switch is off")
        let block = try XCTUnwrap(landed.landed.value)
        XCTAssertEqual(block.landing, WorldImageryLanding(solvedAt: 1791240313.0, geometryRevision: "g-stop-984",
                                                          frameRevision: 1))
        XCTAssertEqual(block.entries, [WorldImageryEntry(keyframeID: "s9:120", digest: F.landedDigest,
                                                         componentReferenceSegment: 0, segmentIndex: nil)])
    }

    /// The fixture as printed: its elided digest is no digest, so its one
    /// entry is dropped and the section stands empty -- never a tile asked
    /// for by a malformed key.
    func testTheFixtureAsPrintedDropsItsElidedDigest() throws {
        let manifest = try XCTUnwrap(F.manifest(F.liveAsPrinted))
        XCTAssertEqual(manifest.live.value?.segmentIndex, 812)
        XCTAssertEqual(manifest.live.value?.entries, [])
    }

    /// Absent = switch off, distinct from "no data yet" (`entries: []`,
    /// `landed: null`).
    func testOmittedEmptyAndNullAreThreeStates() throws {
        let none = try XCTUnwrap(F.manifest(#"{"version":1}"#))
        XCTAssertEqual(none.live, .off)
        XCTAssertEqual(none.landed, .off)
        let empty = try XCTUnwrap(F.manifest(#"{"version":1,"live":{"segment_index":0,"entries":[]},"landed":null}"#))
        XCTAssertEqual(empty.live, .value(WorldImageryLive(segmentIndex: 0, entries: [])))
        XCTAssertEqual(empty.landed, .notYet)
        XCTAssertEqual(F.manifest(#"{"version":1,"live":null}"#)?.live, .notYet)
    }

    func testMalformedKnownFieldsAreRefused() {
        let entry = #"{"keyframe_id":"k","digest":"\#(F.landedDigest)","redaction":"redacted","pose_source":"landed_global_solve","component_reference_segment":0,"segment_index":null}"#
        let refused = [
            #"{"version":2}"#, #"{}"#, #"{"version":"1"}"#, #"{"version":true}"#, #"[]"#,
            #"{"version":1,"live":{"segment_index":-1,"entries":[]}}"#,
            #"{"version":1,"live":{"segment_index":"812","entries":[]}}"#,
            #"{"version":1,"live":{"segment_index":2147483648,"entries":[]}}"#,
            #"{"version":1,"live":{"segment_index":812}}"#,
            #"{"version":1,"live":{"segment_index":812,"entries":{}}}"#,
            #"{"version":1,"live":[]}"#,
            "{\"version\":1,\"live\":{\"segment_index\":1,\"entries\":["
                + Array(repeating: "{}", count: 17).joined(separator: ",") + "]}}",
            #"{"version":1,"landed":{"solved_at":1.0,"geometry_revision":"g","frame_revision":1,"entries":[]}}"#,
            #"{"version":1,"landed":{"solved_at":-1.0,"geometry_revision":"g","frame_revision":1,"entries":[\#(entry)]}}"#,
            #"{"version":1,"landed":{"solved_at":true,"geometry_revision":"g","frame_revision":1,"entries":[\#(entry)]}}"#,
            #"{"version":1,"landed":{"solved_at":1.0,"geometry_revision":"","frame_revision":1,"entries":[\#(entry)]}}"#,
            #"{"version":1,"landed":{"solved_at":1.0,"geometry_revision":"g","frame_revision":1.5,"entries":[\#(entry)]}}"#,
            #"{"version":1,"landed":{"solved_at":1.0,"frame_revision":1,"entries":[\#(entry)]}}"#,
            #"{"version":1,"landed":"on"}"#,
        ]
        for text in refused {
            XCTAssertNil(F.manifest(text), "accepted: \(text)")
        }
    }

    func testUnknownFieldsAreKept() throws {
        let text = F.landed
            .replacingOccurrences(of: #"{"version":1,"#, with: #"{"version":1,"etag":"x","future":{"a":1},"#)
            .replacingOccurrences(of: #""frame_revision":1,"#, with: #""frame_revision":1,"budget_ms":12,"#)
            .replacingOccurrences(of: #""segment_index":null}"#, with: #""segment_index":null,"quality":60}"#)
        let manifest = try XCTUnwrap(F.manifest(text), text)
        XCTAssertEqual(manifest, F.manifest(F.landed))
    }

    /// Only complete matching provenance: anything less drops the entry,
    /// and the rest of the section stands.
    func testEntriesWithoutCompleteMatchingProvenanceAreDropped() throws {
        let good = #"{"keyframe_id":"s9:1","digest":"\#(F.liveDigest)","redaction":"redacted","pose_source":"segment_local","component_reference_segment":null,"segment_index":812}"#
        func live(_ entry: String) -> [WorldImageryEntry]? {
            F.manifest(#"{"version":1,"live":{"segment_index":812,"entries":[\#(entry),\#(good)]}}"#)?.live.value?.entries
        }
        let other = "b" + String(F.liveDigest.dropFirst())
        let base = good.replacingOccurrences(of: F.liveDigest, with: other)
        let dropped = [
            base.replacingOccurrences(of: #","redaction":"redacted""#, with: ""),
            base.replacingOccurrences(of: #""redaction":"redacted""#, with: #""redaction":"raw-local-research""#),
            base.replacingOccurrences(of: #""redaction":"redacted""#, with: #""redaction":null"#),
            base.replacingOccurrences(of: #""pose_source":"segment_local""#, with: #""pose_source":"landed_global_solve""#),
            base.replacingOccurrences(of: #","pose_source":"segment_local""#, with: ""),
            base.replacingOccurrences(of: #""keyframe_id":"s9:1","#, with: ""),
            base.replacingOccurrences(of: other, with: other.uppercased()),
            base.replacingOccurrences(of: other, with: String(other.dropLast())),
            base.replacingOccurrences(of: #""segment_index":812}"#, with: #""segment_index":811}"#),
            base.replacingOccurrences(of: #""component_reference_segment":null"#, with: #""component_reference_segment":0"#),
            "17", "null",
        ]
        let kept = WorldImageryEntry(keyframeID: "s9:1", digest: F.liveDigest, componentReferenceSegment: nil,
                                     segmentIndex: 812)
        for entry in dropped {
            XCTAssertEqual(live(entry), [kept], "kept: \(entry)")
        }
        XCTAssertEqual(live(base)?.count, 2, "the well-formed pair")

        let landed = try XCTUnwrap(F.manifest(F.landed.replacingOccurrences(of: #""component_reference_segment":0,"#,
                                                                            with: "")))
        XCTAssertEqual(landed.landed.value?.entries, [], "a landed entry with no component")
    }

    // MARK: Acceptance 5: §4 beside an ordinary §9 status

    /// §4's landed JSON beside an otherwise-ordinary
    /// `world_builder.status/2026-09-10` payload carrying §9's stop block:
    /// the same landing, the same component, and the thumbnail on its
    /// supported station; beside §9's mid block, nothing.
    func testTheLandedFixturePairsWithTheStopStatus() throws {
        let status = FOWFixtures.payload(Self.ordinaryStatus(guidance: FOWFixtures.stop))
        XCTAssertEqual(status["model_state"] as? String, "receiving", "an ordinary status payload")
        let coverage = try XCTUnwrap(WorldCoverageStatus(payload: status).coverage)
        let landed = try XCTUnwrap(F.manifest(F.landed)?.landed.value)
        XCTAssertEqual(landed.landing, WorldImageryLanding(coverage))
        XCTAssertNotEqual(landed.landing, WorldImageryLanding(mid), "mid is another landing")
        XCTAssertEqual(coverage.components.map(\.referenceSegment), [landed.entries[0].componentReferenceSegment])

        var tiles = WorldImageryTiles()
        tiles.insert(try XCTUnwrap(UIImage(data: F.jpeg())), digest: F.landedDigest)
        let placed = WorldLandedPlacement.place(landed, on: coverage, chunks: [F.chunk()], tiles: tiles)
        XCTAssertEqual(placed.count, 1)
        let thumbnail = try XCTUnwrap(placed.first)
        XCTAssertEqual(thumbnail.column, 3.5, accuracy: 1e-9)
        XCTAssertEqual(thumbnail.row, 5.5, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(thumbnail.heading), 90, accuracy: 1e-6)
        XCTAssertEqual(WorldImageryCopy.landedThumbnail(heading: thumbnail.heading, age: 12.4),
                       "redacted photo, look direction 90°, 12 seconds old")
        XCTAssertEqual(WorldImageryCopy.landedThumbnail(heading: 359.6, age: 31),
                       "redacted photo, look direction 0°, 31 seconds old, stale")
        XCTAssertEqual(WorldImageryCopy.landedThumbnail(heading: nil, age: nil),
                       "redacted photo, look direction unknown, age unavailable")
    }

    /// An ordinary receiving status payload around `guidance`.
    static func ordinaryStatus(guidance: String) -> String {
        let block = (FOWFixtures.payload(guidance)["guidance"] as? [String: Any]).flatMap {
            try? JSONSerialization.data(withJSONObject: $0)
        }.flatMap { String(data: $0, encoding: .utf8) } ?? "null"
        return #"{"model_state":"receiving","model_state_reason":null,"session":{"session_id":"s9","started_at":1791240000.0},"world_snapshot":{"name":"Probe Room","world_id":"w1","keyframe_count":984,"revision":"r9","tracking":"good","scale":"relative","mapping_seconds":313.0,"calibration":"calibrated","geometry":{"representation":"sparse point cloud","element_count":1360,"is_incremental":false},"trajectory":{"pose_count":984,"path_length":2.85,"path_length_unit":"world units","scale":"relative"},"persistence":{"state":"saved","revision":"p1"}},"geometry":{"available":true,"current":true,"revision":"g-stop-984"},"progress":{"keyframes_accepted":984},"guidance":\#(block)}"#
    }

    // MARK: Placement: over the grid, never on grey, never guessed

    func testAThumbnailIsPlacedOnlyByItsOwnComponentFrameAndAnEvidenceCell() throws {
        let landed = try XCTUnwrap(F.manifest(F.landed)?.landed.value)
        var tiles0 = WorldImageryTiles()
        tiles0.insert(try XCTUnwrap(UIImage(data: F.jpeg())), digest: F.landedDigest)
        let coverage = stop
        func placed(_ chunk: WorldSegmentChunk, tiles: WorldImageryTiles? = nil) -> Int {
            WorldLandedPlacement.place(landed, on: coverage, chunks: [chunk], tiles: tiles ?? tiles0).count
        }
        XCTAssertEqual(placed(F.chunk()), 1)
        XCTAssertEqual(placed(F.chunk(), tiles: WorldImageryTiles()), 0, "no tile in hand")
        XCTAssertEqual(placed(F.chunk(keyframe: "s9:121")), 0, "no pose for its keyframe")
        XCTAssertEqual(placed(F.chunk(frame: 2)), 0, "another frame revision")
        XCTAssertEqual(placed(F.chunk(reference: 3)), 0, "placed in another component")
        XCTAssertEqual(placed(F.chunk(translation: [-3, -1, 0])), 0, "a grey cell: thumbnails only add to evidence")
        XCTAssertEqual(placed(F.chunk(translation: [9, -1, 0])), 0, "outside the piece")
        XCTAssertEqual(placed(F.chunk(translation: [Double.nan, 0, 0])), 0, "no finite centre")
    }

    func testTilesKeepAtMostSixteenLeastRecentlyUsedFirst() throws {
        let image = try XCTUnwrap(UIImage(data: F.jpeg()))
        var tiles = WorldImageryTiles()
        let digests = (0..<18).map { String(format: "%032x", $0) }
        for digest in digests.prefix(16) { tiles.insert(image, digest: digest) }
        tiles.touch([digests[0]])
        tiles.insert(image, digest: digests[16])
        tiles.insert(image, digest: digests[17])
        XCTAssertEqual(tiles.count, WorldImageryTiles.capacity)
        XCTAssertNotNil(tiles[digests[0]], "recently shown: kept")
        XCTAssertNil(tiles[digests[1]], "least recently used: evicted")
        XCTAssertNil(tiles[digests[2]])
        XCTAssertNotNil(tiles[digests[17]])
    }

    func testTheSwitchesAreOnOnlyWhenSaidOnExactly() {
        XCTAssertEqual(WorldImagerySwitches(environment: [:]), .off)
        XCTAssertEqual(WorldImagerySwitches(environment: ["IOS_FOW_ROOM_IMAGERY_LIVE": "1",
                                                          "IOS_FOW_ROOM_IMAGERY_LANDED": "ON"]), .off)
        XCTAssertEqual(WorldImagerySwitches(environment: ["IOS_FOW_ROOM_IMAGERY_LIVE": "on"]),
                       WorldImagerySwitches(live: true, landed: false))
        XCTAssertEqual(WorldImagerySwitches(environment: ["IOS_FOW_ROOM_IMAGERY_LANDED": "on"]),
                       WorldImagerySwitches(live: false, landed: true))
    }

    /// The panel puts B2's thumbnails only over the live receipt they were
    /// placed on: not over the fixture map, not over another landing's.
    func testThePanelDrawsLandedThumbnailsOnlyOverTheirOwnReceipt() {
        typealias Panel = WorldInlinePanel<EmptyView>
        let receipt = WorldCoverageReceipt(coverage: stop, keyframesAccepted: 984, clock: nil)
        let overlay = WorldLandedOverlay(receipt: receipt, thumbnails: [])
        XCTAssertEqual(Panel.landed(overlay, over: .coverage(receipt)), overlay)
        XCTAssertNil(Panel.landed(overlay, over: .fixture(receipt)), "over the DEBUG fixture")
        XCTAssertNil(Panel.landed(overlay, over: .coverage(WorldCoverageReceipt(coverage: mid, keyframesAccepted: nil,
                                                                                clock: nil))),
                     "over another landing's grid")
        XCTAssertNil(Panel.landed(nil, over: .coverage(receipt)))
    }

    /// Source scan (Mac only): the post-Stop loading view draws no imagery.
    func testTheLoadingViewSourceDrawsNoImagery() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Glasses/Workspaces/WorldBuilder/WorldInlinePanel.swift")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("sources are not on a device") }
        let src = try String(contentsOf: url, encoding: .utf8)
        let body = try XCTUnwrap(src.components(separatedBy: "private func finishingContent").last?
            .components(separatedBy: "private func stageText").first)
        for symbol in ["WorldImageryStripView", "WorldLandedPieceOverlay", "liveStrip", "landed"] {
            XCTAssertFalse(body.contains(symbol), "the loading view draws \(symbol)")
        }
    }
}

// MARK: - Fetching: OFF asks nothing; no walk or landing reaches another

@MainActor
final class WorldImageryModelTests: XCTestCase {
    private typealias F = FOWImageryFixtures
    private typealias Stub = FOWImageryStubProtocol

    private static let walkA = WorldFinishWalk(worldID: "w1", sessionID: "s9")
    private static let walkB = WorldFinishWalk(worldID: "w2", sessionID: "s2")
    private static let manifestA = "/worlds/w1/guidance/s9/imagery/manifest"
    private static let manifestB = "/worlds/w2/guidance/s2/imagery/manifest"

    private var stop: WorldCoverage { FOWFixtures.coverage(FOWFixtures.stop)! }
    private var mid: WorldCoverage { FOWFixtures.coverage(FOWFixtures.mid)! }

    override func setUp() {
        super.setUp()
        Stub.reset()
        Stub.set(Self.manifestA, json: F.live)
        Stub.set("/worlds/w1/guidance/s9/imagery/tile/\(F.liveDigest)", jpeg: F.jpeg())
        Stub.set("/worlds/w1/guidance/s9/imagery/tile/\(F.landedDigest)", jpeg: F.jpeg(.systemOrange))
    }

    private func model(live: Bool, landed: Bool) -> WorldImageryModel {
        WorldImageryModel(switches: WorldImagerySwitches(live: live, landed: landed),
                          client: WorldImageryClient(baseURL: URL(string: "http://tower.test")!,
                                                     session: Stub.makeSession(), timeout: 5))
    }

    private func push(_ model: WorldImageryModel, _ walk: WorldFinishWalk = walkA, revision: String,
                      receiving: Bool = true) {
        model.geometryPushed(WorldGeometryCoordinates(worldID: walk.worldID, sessionID: walk.sessionID,
                                                      revision: revision), receiving: receiving)
    }

    private func expect(_ what: String = "", timeout: TimeInterval = 5, _ condition: () -> Bool) async {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() && Date() < deadline { try? await Task.sleep(for: .milliseconds(20)) }
        XCTAssertTrue(condition(), what)
    }

    private func settle() async { try? await Task.sleep(for: .milliseconds(300)) }

    /// OFF: no request, whatever is pushed, and nothing held.
    func testOffAsksNothing() async {
        let off = model(live: false, landed: false)
        for revision in ["g1", "g2", "g3"] { push(off, revision: revision) }
        off.coverageReported(WalkScoped(walk: Self.walkA, value: stop), receiving: true)
        off.coverageReported(WalkScoped(walk: Self.walkA, value: mid), receiving: true)
        await settle()
        XCTAssertEqual(Stub.requests, [], "OFF made a request")
        XCTAssertNil(off.imagery.walk)
        XCTAssertEqual(off.imagery.value, WorldImagery())
    }

    /// B1 on, B2 off: coverage never asks; B2 on, B1 off: geometry never asks.
    func testEachStageAsksOnlyOnItsOwnTrigger() async {
        let liveOnly = model(live: true, landed: false)
        liveOnly.coverageReported(WalkScoped(walk: Self.walkA, value: stop), receiving: true)
        await settle()
        XCTAssertEqual(Stub.requests, [], "B1 alone asked on a coverage change")

        Stub.set(Self.manifestA, json: F.landed)
        let landedOnly = model(live: false, landed: true)
        push(landedOnly, revision: "g1")
        push(landedOnly, revision: "g2")
        await settle()
        XCTAssertEqual(Stub.requests, [], "B2 alone asked on a geometry push before any landing")
    }

    /// B1: one manifest per new geometry revision (heartbeats ask nothing),
    /// and each tile once.
    func testTheLiveManifestFollowsTheGeometryRevisionOnly() async {
        let model = model(live: true, landed: false)
        push(model, revision: "g1")
        await expect("the live tile") { model.imagery.value.liveTiles[F.liveDigest] != nil }
        XCTAssertEqual(model.imagery.walk, Self.walkA)
        XCTAssertEqual(model.imagery.value.live?.segmentIndex, 812)
        for _ in 0..<3 { push(model, revision: "g1") }
        await settle()
        XCTAssertEqual(Stub.count(Self.manifestA), 1, "a heartbeat refetched the manifest")
        push(model, revision: "g2")
        await expect { Stub.count(Self.manifestA) == 2 }
        await settle()
        XCTAssertEqual(Stub.count("/worlds/w1/guidance/s9/imagery/tile/\(F.liveDigest)"), 1, "a tile twice")

        // A tracking break: the new segment starts empty.
        Stub.set(Self.manifestA, json: #"{"version":1,"live":{"segment_index":813,"entries":[]}}"#)
        push(model, revision: "g3")
        await expect("the new segment") { model.imagery.value.live?.segmentIndex == 813 }
        XCTAssertEqual(model.imagery.value.live?.entries, [])
    }

    /// A failed fetch keeps what is shown and is asked again on the next
    /// push; a 404 (the Tower's route is off) is an answer, not a retry.
    func testAFailureRetriesOnTheNextPushAndA404DoesNot() async {
        let model = model(live: true, landed: false)
        push(model, revision: "g1")
        await expect { model.imagery.value.live?.segmentIndex == 812 }
        Stub.set(Self.manifestA, status: 500, json: "{}")
        push(model, revision: "g2")
        await expect { Stub.count(Self.manifestA) == 2 }
        await settle()
        XCTAssertEqual(model.imagery.value.live?.segmentIndex, 812, "a failed fetch cleared the strip")
        push(model, revision: "g2")
        await expect("retried on the next push") { Stub.count(Self.manifestA) == 3 }

        Stub.set(Self.manifestA, status: 404, json: #"{"detail":"Not Found"}"#)
        push(model, revision: "g4")
        await expect { model.imagery.value.live == nil }
        let asked = Stub.count(Self.manifestA)
        for _ in 0..<3 { push(model, revision: "g4") }
        await settle()
        XCTAssertEqual(Stub.count(Self.manifestA), asked, "a 404 was retried")
    }

    /// B2: asked once per landing in coverage; heartbeats and repeats of the
    /// same landing ask nothing more.
    func testTheLandedManifestFollowsTheLandingOnly() async {
        Stub.set(Self.manifestA, json: F.landed)
        let model = model(live: false, landed: true)
        model.coverageReported(WalkScoped(walk: Self.walkA, value: stop), receiving: true)
        await expect("the landed tile") { model.imagery.value.landedTiles[F.landedDigest] != nil }
        XCTAssertEqual(model.imagery.value.landed?.landing, WorldImageryLanding(stop))
        model.coverageReported(WalkScoped(walk: Self.walkA, value: stop), receiving: true)
        for revision in ["g1", "g2"] { push(model, revision: revision) }
        await settle()
        XCTAssertEqual(Stub.count(Self.manifestA), 1)
    }

    // MARK: Walk mixing

    /// A stale walk: A's manifest, out when B is adopted, never lands in B.
    func testAStaleWalksManifestNeverLandsInTheNextWalk() async {
        Stub.set(delay: 0.6, for: Self.manifestA)
        Stub.set(Self.manifestB, json: #"{"version":1,"live":{"segment_index":5,"entries":[]}}"#)
        let model = model(live: true, landed: false)
        push(model, Self.walkA, revision: "g1")
        await expect { Stub.count(Self.manifestA) == 1 }
        push(model, Self.walkB, revision: "g1")
        await expect("B's own manifest") { model.imagery.value.live?.segmentIndex == 5 }
        try? await Task.sleep(for: .milliseconds(900))
        XCTAssertEqual(model.imagery.walk, Self.walkB)
        XCTAssertEqual(model.imagery.value.live?.segmentIndex, 5, "A's manifest landed in B")
        XCTAssertNil(model.imagery.value.liveTiles[F.liveDigest], "A's tile in B")
        XCTAssertEqual(Stub.count("/worlds/w1/guidance/s9/imagery/tile/\(F.liveDigest)"), 0)
    }

    /// A superseded landing: the answer for a landing coverage has moved past
    /// is dropped, and a manifest still carrying the old landing is never
    /// shown for the new one.
    func testASupersededLandingIsNeverShown() async {
        let older = F.landed.replacingOccurrences(of: "1791240313.0", with: "1791240000.0")
            .replacingOccurrences(of: "g-stop-984", with: "g-mid-113")
        Stub.set(Self.manifestA, json: older)
        Stub.set(delay: 0.6, for: Self.manifestA)
        let model = model(live: false, landed: true)
        model.coverageReported(WalkScoped(walk: Self.walkA, value: mid), receiving: true)
        await expect { Stub.count(Self.manifestA) == 1 }
        // Coverage moves to the stop landing while mid's request is out, and
        // the Tower still serves mid's thumbnails (its job is slow).
        model.coverageReported(WalkScoped(walk: Self.walkA, value: stop), receiving: true)
        try? await Task.sleep(for: .milliseconds(1500))
        XCTAssertNil(model.imagery.value.landed, "a landing other than the one asked for")

        // Retried on the next push, a bounded number of times, and shown
        // once the Tower carries the stop landing.
        Stub.set(delay: 0, for: Self.manifestA)
        Stub.set(Self.manifestA, json: F.landed)
        push(model, revision: "g9")
        await expect("the stop landing") { model.imagery.value.landed?.landing == WorldImageryLanding(stop) }
    }

    func testASlowLandingIsAskedAgainOnlyABoundedNumberOfTimes() async {
        Stub.set(Self.manifestA, json: #"{"version":1,"landed":null}"#)
        let model = model(live: false, landed: true)
        model.coverageReported(WalkScoped(walk: Self.walkA, value: stop), receiving: true)
        for index in 0..<10 {
            await settle()
            push(model, revision: "g\(index)")
        }
        await settle()
        XCTAssertEqual(Stub.count(Self.manifestA), 1 + WorldImageryModel.landingRetries)
    }

    /// A late push after Stop: the Tower still reports the walk, and the
    /// phone asks nothing -- nothing is received any more.
    func testALatePushAfterStopAsksNothing() async {
        let model = model(live: true, landed: true)
        push(model, revision: "g1", receiving: false)
        model.coverageReported(WalkScoped(walk: Self.walkA, value: stop), receiving: false)
        await settle()
        XCTAssertEqual(Stub.requests, [])
    }

    /// And the strip, like the piece, is gone at the local Stop: a late
    /// `receiving` report for the stopped walk never brings it back.
    func testTheStripIsGatedLikeThePiece() {
        let strip = WalkScoped(walk: Self.walkA, value: WorldImageryStrip(segmentIndex: 812, thumbnails: []))
        let report = WalkScoped(walk: Self.walkA, value: 0)
        let receiving = WorldModelState.receiving(WorldSnapshot())
        var gate = WorldCurrentPieceGate()
        gate.observe(isCapturing: true, presented: Self.walkA)
        XCTAssertNotNil(gate.shown(strip, beside: report, isCapturing: true, phase: .walking(hasMap: true),
                                   state: receiving))
        XCTAssertNil(gate.shown(strip, beside: WalkScoped(walk: Self.walkB, value: 0), isCapturing: true,
                                phase: .walking(hasMap: true), state: receiving), "beside another walk")
        // Stop: not capturing, and the late receiving report for A.
        gate.observe(isCapturing: false, presented: Self.walkA)
        XCTAssertNil(gate.shown(strip, beside: report, isCapturing: false, phase: .walking(hasMap: true),
                                state: receiving))
        XCTAssertNil(gate.shown(strip, beside: report, isCapturing: false, phase: .finishing(target: nil),
                                state: receiving))
        // Start again before the Tower moves on: A stays refused.
        gate.observe(isCapturing: true, presented: Self.walkA)
        XCTAssertNil(gate.shown(strip, beside: report, isCapturing: true, phase: .walking(hasMap: true),
                                state: receiving), "the stopped walk's strip came back")
    }

    func testResetForgetsTheWalk() async {
        let model = model(live: true, landed: false)
        push(model, revision: "g1")
        await expect { model.imagery.value.live != nil }
        model.reset()
        XCTAssertNil(model.imagery.walk)
        XCTAssertEqual(model.imagery.value, WorldImagery())
    }

    /// Tiles only as the spec allows: JPEG, at most 16 KiB and 160 x 90.
    func testATileOutsideTheSpecIsRefused() async throws {
        let client = WorldImageryClient(baseURL: URL(string: "http://tower.test")!, session: Stub.makeSession(),
                                        timeout: 5)
        let path = "/worlds/w1/guidance/s9/imagery/tile/\(F.liveDigest)"
        func fetch() async -> WorldImageryFetchError? {
            do {
                _ = try await client.tile(worldID: "w1", sessionID: "s9", digest: F.liveDigest)
                return nil
            } catch {
                return error as? WorldImageryFetchError
            }
        }
        let ok = await fetch()
        XCTAssertNil(ok)
        Stub.set(path, jpeg: F.jpeg(size: CGSize(width: 320, height: 180)))
        let big = await fetch()
        XCTAssertEqual(big, .refused, "larger than 160 x 90")
        Stub.set(path, status: 200, data: F.jpeg(), type: "image/png")
        let png = await fetch()
        XCTAssertEqual(png, .refused, "not a JPEG")
        Stub.set(path, status: 200, data: Data(repeating: 0xFF, count: 17 * 1024), type: "image/jpeg")
        let heavy = await fetch()
        XCTAssertEqual(heavy, .refused, "over 16 KiB")
        Stub.set(path, status: 404, data: Data(), type: "application/json")
        let gone = await fetch()
        XCTAssertEqual(gone, .notFound)
    }
}

/// A stubbed imagery route: binary bodies, a request log, held answers.
final class FOWImageryStubProtocol: URLProtocol {
    private static var routes: [String: (status: Int, data: Data, type: String)] = [:]
    private static var delays: [String: TimeInterval] = [:]
    private static var paths: [String] = []
    private static let lock = NSLock()

    static func reset() {
        lock.lock(); defer { lock.unlock() }
        routes = [:]
        delays = [:]
        paths = []
    }

    static func set(_ path: String, status: Int = 200, data: Data, type: String) {
        lock.lock(); defer { lock.unlock() }
        routes[path] = (status, data, type)
    }

    static func set(_ path: String, status: Int = 200, json: String) {
        set(path, status: status, data: Data(json.utf8), type: "application/json")
    }

    static func set(_ path: String, jpeg: Data) {
        set(path, status: 200, data: jpeg, type: "image/jpeg")
    }

    static func set(delay: TimeInterval, for path: String) {
        lock.lock(); defer { lock.unlock() }
        delays[path] = delay
    }

    static var requests: [String] {
        lock.lock(); defer { lock.unlock() }
        return paths
    }

    static func count(_ path: String) -> Int { requests.filter { $0 == path }.count }

    static func makeSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [FOWImageryStubProtocol.self]
        configuration.urlCache = nil
        return URLSession(configuration: configuration)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let path = request.url?.path ?? ""
        Self.lock.lock()
        Self.paths.append(path)
        let route = Self.routes[path]
        let delay = Self.delays[path] ?? 0
        Self.lock.unlock()
        let answer = { [weak self] in
            guard let self, !self.isStopped else { return }
            guard let route else {
                self.client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
                return
            }
            let response = HTTPURLResponse(url: self.request.url!, statusCode: route.status, httpVersion: "HTTP/1.1",
                                           headerFields: ["Content-Type": route.type])!
            self.client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            self.client?.urlProtocol(self, didLoad: route.data)
            self.client?.urlProtocolDidFinishLoading(self)
        }
        if delay > 0 {
            DispatchQueue.global().asyncAfter(deadline: .now() + delay, execute: answer)
        } else {
            answer()
        }
    }

    private var isStopped = false

    override func stopLoading() { isStopped = true }
}
