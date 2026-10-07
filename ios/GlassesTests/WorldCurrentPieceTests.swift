//
//  WorldCurrentPieceTests.swift
//  GlassesTests
//
//  Fog of war's current piece (FOW-CURRENT-PIECE-DATA-20261006 §4): the
//  last segment only, reset at every tracking break, segment-local (no
//  world transform), walk-scoped, live only while walking, and nothing at
//  all without segment data.
//

import XCTest
@testable import Glasses

@MainActor
final class WorldCurrentPieceTests: XCTestCase {

    // MARK: Fixtures

    private static let contract = "world_builder.geometry/2026-08-25"

    /// One `poses[]` row: `t == nil` is a refused pose.
    private static func pose(_ id: Int, _ t: [Double]?, q: [Double]? = [1, 0, 0, 0]) -> String {
        let translation = t.map { "\($0)" } ?? "null"
        let rotation = t == nil ? "null" : (q.map { "\($0)" } ?? "null")
        return #"{"keyframe_id":"s1:\#(id)","status":"\#(t == nil ? "refused" : "solved")","degeneracy":"","#
            + #""rotation":\#(rotation),"translation":\#(translation)}"#
    }

    /// Unplaced, or placed far away with a scale -- which must never move a footprint.
    private static func placement(registered: Bool) -> String {
        registered
            ? #""registered":true,"registration_state":"registered","placement_hash":"p-far","#
                + #""transform_to_world":{"rotation_wxyz":[0.0,0.0,1.0,0.0],"translation":[100.0,50.0,-70.0],"#
                + #""scale":7.5,"reference_segment":0,"frame_revision":1}"#
            : #""registered":false,"transform_to_world":null"#
    }

    private static func chunkJSON(_ index: Int, hash: String, poses: [String], registered: Bool = false) -> String {
        #"{"contract":"\#(contract)","segment_index":\#(index),"content_hash":"\#(hash)","#
            + #""frame_id":"segment:\#(index)",\#(placement(registered: registered)),"#
            + #""poses":[\#(poses.joined(separator: ","))],"points":[[0.5,0.5,0.5]],"#
            + #""points_sent":1,"points_total":1,"point_sampling":"none"}"#
    }

    private static func rowJSON(_ index: Int, hash: String, registered: Bool = false) -> String {
        #"{"segment_index":\#(index),"content_hash":"\#(hash)","frame_id":"segment:\#(index)","#
            + #"\#(placement(registered: registered)),"resolution_state":"resolved","dominant_degeneracy":null,"#
            + #""keyframe_count":4,"solved_count":3,"point_count":1,"#
            + #""bounds":{"min":[0.0,0.0,0.0],"max":[1.0,1.0,1.0]}}"#
    }

    private static func manifestJSON(revision: String, rows: [String]) -> String {
        #"{"contract":"\#(contract)","world_id":"w1","session_id":"s1","geometry_revision":"\#(revision)","#
            + #""pose_convention":{"pose_type":"T_world_camera","quaternion_order":"wxyz","handedness":"right","#
            + #""camera_axes":"opencv_x_right_y_down_z_forward","translation_units":"world","#
            + #""world_axes_origin":"first_keyframe_camera","up_axis":"unknown","pose_dtype":"float64","#
            + #""point_dtype":"float32"},"segment_count":\#(rows.count),"segments":[\#(rows.joined(separator: ","))]}"#
    }

    private static func object(_ json: String) -> [String: Any] {
        // swiftlint:disable:next force_cast force_try
        try! JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
    }

    private static func chunk(_ index: Int, hash: String, poses: [String], registered: Bool = false)
        -> WorldSegmentChunk {
        WorldGeometryDecoder.chunk(from: object(chunkJSON(index, hash: hash, poses: poses, registered: registered)))!
    }

    private static func segments(_ rows: [String]) -> [WorldSegmentSummary] {
        WorldGeometryDecoder.manifest(from: object(manifestJSON(revision: "g", rows: rows)))!.segments
    }

    /// Segment 0: three posed keyframes and a refused one between them.
    private static let zeroPoses = [pose(0, [0, 0, 0]), pose(1, [1, 0, 2]), pose(2, nil), pose(3, [2, 0, 3])]
    /// Segment 1, after a tracking break: two posed keyframes.
    private static let onePoses = [pose(0, [5, 0, 5]), pose(1, [6, 0, 5])]

    private static let walkA = WorldFinishWalk(worldID: "w1", sessionID: "s1")
    private static let walkB = WorldFinishWalk(worldID: "w2", sessionID: "s9")
    private static let receiving = WorldModelState.receiving(WorldSnapshot())

    // MARK: The last segment, reset at every break

    /// A new last segment is a new piece: segment 0's footprints are gone,
    /// never joined to segment 1's, and never drawn in its place.
    func testANewSegmentResetsThePiece() throws {
        let zero = Self.chunk(0, hash: "h0", poses: Self.zeroPoses)
        let one = Self.chunk(1, hash: "h1", poses: Self.onePoses)
        let before = try XCTUnwrap(WorldCurrentPiece.current(
            segments: Self.segments([Self.rowJSON(0, hash: "h0")]), chunks: [zero.cacheKey: zero]))
        XCTAssertEqual(before.segmentIndex, 0)
        XCTAssertEqual(before.viewCount, 3)
        XCTAssertEqual(before.runs.count, 2, "a refused pose is a break, not a line through the gap")

        let after = try XCTUnwrap(WorldCurrentPiece.current(
            segments: Self.segments([Self.rowJSON(0, hash: "h0"), Self.rowJSON(1, hash: "h1")]),
            chunks: [zero.cacheKey: zero, one.cacheKey: one]))
        XCTAssertEqual(after.segmentIndex, 1)
        XCTAssertEqual(after.viewCount, 2, "segment 0's footprints survived the break")
        XCTAssertEqual(after.runs.flatMap { $0 }.map(\.x), [5, 6])
    }

    /// The same reset through the view model's own fetch chain: the
    /// revision that names segment 1 replaces segment 0's piece.
    func testTheViewModelsPieceResetsWhenTheManifestNamesANewSegment() async throws {
        let host = URL(string: "http://stub.invalid")!
        StubbedGeometryProtocol.reset(routes: [
            "/worlds/w1/geometry/manifest": (200, Self.manifestJSON(revision: "g1", rows: [Self.rowJSON(0, hash: "h0")])),
            "/worlds/w1/geometry/segment/0": (200, Self.chunkJSON(0, hash: "h0", poses: Self.zeroPoses)),
        ])
        let viewModel = WorldBuilderViewModel(
            client: UnavailableWorldBuilderClient(),
            geometry: WorldGeometryClient(baseURL: host, session: StubbedGeometryProtocol.makeSession()))
        XCTAssertNil(viewModel.currentPiece, "no geometry: no piece")

        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        let first = try XCTUnwrap(viewModel.currentPiece)
        XCTAssertEqual(first.walk, Self.walkA, "the piece names the walk whose geometry it is")
        XCTAssertEqual(first.value.segmentIndex, 0)
        XCTAssertEqual(first.value.viewCount, 3)

        StubbedGeometryProtocol.reset(routes: [
            "/worlds/w1/geometry/manifest": (200, Self.manifestJSON(
                revision: "g2", rows: [Self.rowJSON(0, hash: "h0"), Self.rowJSON(1, hash: "h1")])),
            "/worlds/w1/geometry/segment/0": (200, Self.chunkJSON(0, hash: "h0", poses: Self.zeroPoses)),
            "/worlds/w1/geometry/segment/1": (200, Self.chunkJSON(1, hash: "h1", poses: Self.onePoses)),
        ])
        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g2")
        let second = try XCTUnwrap(viewModel.currentPiece)
        XCTAssertEqual(second.value.segmentIndex, 1)
        XCTAssertEqual(second.value.viewCount, 2, "the piece did not restart at the tracking break")

        // Another walk's geometry: the old walk's piece goes with it.
        await viewModel.geometryDidChange(worldID: "w2", sessionID: "s9", revision: "g3")
        XCTAssertNil(viewModel.currentPiece, "w1's piece survived under w2's geometry")
    }

    /// Codex review HIGH: a new revision (a tracking break) is named while
    /// its fetch is still out. The old segment is not "current" any more, so
    /// nothing is shown until the new geometry lands -- while the room map
    /// keeps its silent refetch, the previous manifest still on screen.
    func testANewRevisionHidesThePieceUntilItsGeometryLands() async throws {
        let host = URL(string: "http://stub.invalid")!
        StubbedGeometryProtocol.reset(routes: [
            "/worlds/w1/geometry/manifest": (200, Self.manifestJSON(revision: "g1", rows: [Self.rowJSON(0, hash: "h0")])),
            "/worlds/w1/geometry/segment/0": (200, Self.chunkJSON(0, hash: "h0", poses: Self.zeroPoses)),
        ])
        let viewModel = WorldBuilderViewModel(
            client: UnavailableWorldBuilderClient(),
            geometry: WorldGeometryClient(baseURL: host, session: StubbedGeometryProtocol.makeSession()))
        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        XCTAssertEqual(viewModel.currentPiece?.value.segmentIndex, 0)

        StubbedGeometryProtocol.reset(routes: [
            "/worlds/w1/geometry/manifest": (200, Self.manifestJSON(
                revision: "g2", rows: [Self.rowJSON(0, hash: "h0"), Self.rowJSON(1, hash: "h1")])),
            "/worlds/w1/geometry/segment/0": (200, Self.chunkJSON(0, hash: "h0", poses: Self.zeroPoses)),
            "/worlds/w1/geometry/segment/1": (200, Self.chunkJSON(1, hash: "h1", poses: Self.onePoses)),
        ])
        StubbedGeometryProtocol.set(delay: 0.5, for: "/worlds/w1/geometry/manifest")
        let fetch = Task { await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g2") }
        let deadline = Date().addingTimeInterval(3)
        while StubbedGeometryProtocol.requestCount(for: "/worlds/w1/geometry/manifest") == 0, Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertEqual(StubbedGeometryProtocol.requestCount(for: "/worlds/w1/geometry/manifest"), 1,
                       "g2's fetch never went out")
        XCTAssertNil(viewModel.currentPiece, "segment 0 still labelled current while g2's fetch is out")
        XCTAssertEqual(viewModel.fragmentsModel.segments.count, 1, "the room map dropped g1 during the refetch")

        await fetch.value
        let after = try XCTUnwrap(viewModel.currentPiece, "g2 landed and no piece")
        XCTAssertEqual(after.value.segmentIndex, 1)
        XCTAssertEqual(after.value.viewCount, 2)
    }

    /// Codex review HIGH (second pass): the manifest route is not
    /// revision-pinned, so a request made for g2 mid-rebuild can come back as
    /// g1. What came back decides: no piece while the returned revision is
    /// not the named one, the map keeps what arrived, and the next report
    /// naming g2 asks again -- and the piece shows once g2 itself lands.
    func testAnOlderManifestReturnedForANewRevisionIsNoPieceUntilTheNamedOneLands() async throws {
        let host = URL(string: "http://stub.invalid")!
        let g1 = Self.manifestJSON(revision: "g1", rows: [Self.rowJSON(0, hash: "h0")])
        StubbedGeometryProtocol.reset(routes: [
            "/worlds/w1/geometry/manifest": (200, g1),
            "/worlds/w1/geometry/segment/0": (200, Self.chunkJSON(0, hash: "h0", poses: Self.zeroPoses)),
        ])
        let viewModel = WorldBuilderViewModel(
            client: UnavailableWorldBuilderClient(),
            geometry: WorldGeometryClient(baseURL: host, session: StubbedGeometryProtocol.makeSession()))
        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g1")
        XCTAssertEqual(viewModel.currentPiece?.value.segmentIndex, 0)

        // The Tower names g2 (a tracking break) but its manifest still serves g1.
        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g2")
        XCTAssertNil(viewModel.currentPiece, "g1's segment shown as current under g2")
        XCTAssertEqual(viewModel.fragmentsModel.segments.count, 1, "the room map lost what arrived")

        // g2 itself lands; the next report naming g2 fetches it again.
        StubbedGeometryProtocol.reset(routes: [
            "/worlds/w1/geometry/manifest": (200, Self.manifestJSON(
                revision: "g2", rows: [Self.rowJSON(0, hash: "h0"), Self.rowJSON(1, hash: "h1")])),
            "/worlds/w1/geometry/segment/0": (200, Self.chunkJSON(0, hash: "h0", poses: Self.zeroPoses)),
            "/worlds/w1/geometry/segment/1": (200, Self.chunkJSON(1, hash: "h1", poses: Self.onePoses)),
        ])
        await viewModel.geometryDidChange(worldID: "w1", sessionID: "s1", revision: "g2")
        let after = try XCTUnwrap(viewModel.currentPiece, "g2 landed and no piece")
        XCTAssertEqual(after.value.segmentIndex, 1)
        XCTAssertEqual(viewModel.fragmentsModel.segments.count, 2)
    }

    // MARK: No data, nothing drawn

    func testNoSegmentDataIsNoPiece() {
        let zero = Self.chunk(0, hash: "h0", poses: Self.zeroPoses)
        XCTAssertNil(WorldCurrentPiece.current(segments: [], chunks: [:]), "no manifest")
        XCTAssertNil(WorldCurrentPiece.current(segments: [], chunks: [zero.cacheKey: zero]), "a chunk no row names")
        // The last segment's chunk is not in hand: never the older one in its place.
        XCTAssertNil(WorldCurrentPiece.current(
            segments: Self.segments([Self.rowJSON(0, hash: "h0"), Self.rowJSON(1, hash: "h1")]),
            chunks: [zero.cacheKey: zero]), "an older segment drawn as current")
        // Every pose refused: nothing to draw, not a footprint at zero.
        let refused = Self.chunk(0, hash: "h0", poses: [Self.pose(0, nil), Self.pose(1, nil)])
        XCTAssertNil(WorldCurrentPiece.current(
            segments: Self.segments([Self.rowJSON(0, hash: "h0")]), chunks: [refused.cacheKey: refused]))
        // And the panel is handed nothing.
        XCTAssertNil(WorldCurrentPiece.shown(nil, walk: Self.walkA, phase: .walking(hasMap: false),
                                             state: Self.receiving))
    }

    // MARK: Walk-scoped and live

    func testThePieceIsShownOnlyBesideItsOwnWalkWhileWalking() throws {
        let zero = Self.chunk(0, hash: "h0", poses: Self.zeroPoses)
        let piece = try XCTUnwrap(WorldCurrentPiece(chunk: zero))
        let scoped = WalkScoped(walk: Self.walkA, value: piece)

        XCTAssertEqual(WorldCurrentPiece.shown(scoped, walk: Self.walkA, phase: .walking(hasMap: false),
                                               state: Self.receiving), piece)
        XCTAssertEqual(WorldCurrentPiece.shown(scoped, walk: Self.walkA, phase: .walking(hasMap: true),
                                               state: Self.receiving), piece, "beside the map too")
        XCTAssertNil(WorldCurrentPiece.shown(scoped, walk: Self.walkB, phase: .walking(hasMap: false),
                                             state: Self.receiving), "A's piece under B's walk")
        XCTAssertNil(WorldCurrentPiece.shown(scoped, walk: nil, phase: .walking(hasMap: false),
                                             state: Self.receiving), "a report naming no walk")
        XCTAssertNil(WorldCurrentPiece.shown(WalkScoped(walk: nil, value: piece), walk: Self.walkA,
                                             phase: .walking(hasMap: false), state: Self.receiving),
                     "a piece of no named walk")
        // After Stop the Tower is finalizing and the panel finishing: gone.
        XCTAssertNil(WorldCurrentPiece.shown(scoped, walk: Self.walkA, phase: .finishing(target: nil),
                                             state: .finalizing(WorldSnapshot())))
        XCTAssertNil(WorldCurrentPiece.shown(scoped, walk: Self.walkA, phase: .walking(hasMap: false),
                                             state: .finalizing(WorldSnapshot())),
                     "still capturing but the Tower has stopped receiving: the segment is frozen")
        XCTAssertNil(WorldCurrentPiece.shown(scoped, walk: Self.walkA, phase: .offline, state: Self.receiving))
        // Stopped on this phone, the Tower still receiving: gone at the Stop.
        XCTAssertNil(WorldCurrentPiece.shown(scoped, walk: Self.walkA, stoppedHere: Self.walkA,
                                             phase: .walking(hasMap: false), state: Self.receiving))
        XCTAssertNotNil(WorldCurrentPiece.shown(scoped, walk: Self.walkA, stoppedHere: Self.walkB,
                                                phase: .walking(hasMap: false), state: Self.receiving),
                        "another walk's Stop hid this walk's piece")
    }

    // MARK: Segment-local: no world transform

    /// A registered segment placed far away, rotated half a turn and scaled
    /// 7.5x draws exactly where its own frame says -- the Sim3 is never read.
    func testNoWorldTransformIsApplied() throws {
        let placed = Self.chunk(0, hash: "h0", poses: Self.zeroPoses, registered: true)
        XCTAssertNotNil(placed.transformToWorld, "the fixture must carry a transform to refuse")
        let local = Self.chunk(0, hash: "h0", poses: Self.zeroPoses)
        let fromPlaced = try XCTUnwrap(WorldCurrentPiece(chunk: placed))
        XCTAssertEqual(fromPlaced.runs, try XCTUnwrap(WorldCurrentPiece(chunk: local)).runs)
        let points = fromPlaced.runs.flatMap { $0 }.map { [$0.x, $0.y] }
        XCTAssertEqual(points, [[0, 0], [1, 2], [2, 3]], "segment-local x and z, untouched")
    }

    /// Seen from above, unmirrored: forward is 0, a quarter turn right
    /// (about the camera's y, which points down) is +pi/2.
    func testTheHeadingSeenFromAbove() throws {
        XCTAssertEqual(try XCTUnwrap(WorldCurrentPiece.heading([1, 0, 0, 0])), 0, accuracy: 1e-9)
        let quarter = [cos(Double.pi / 4), 0, sin(Double.pi / 4), 0]
        XCTAssertEqual(try XCTUnwrap(WorldCurrentPiece.heading(quarter)), .pi / 2, accuracy: 1e-9)
        // Looking straight down (about x by -90 degrees): no heading.
        XCTAssertNil(WorldCurrentPiece.heading([cos(-Double.pi / 4), sin(-Double.pi / 4), 0, 0]))
        XCTAssertNil(WorldCurrentPiece.heading(nil))
    }
}
