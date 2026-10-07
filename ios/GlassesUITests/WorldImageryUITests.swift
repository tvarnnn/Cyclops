//
//  WorldImageryUITests.swift
//  GlassesUITests
//
//  FOW v1.1 B (FOW-V11B-WIRE-AND-IOS-SPEC-20261007 §5), against the mock
//  Tower serving the spec's §4 fixtures on the manifest and tile routes:
//  both switches off draw nothing and ask nothing (acceptance 2); B1's strip
//  shows one thumbnail and clears at a segment bump, and goes at Stop for
//  good (acceptance 3); B2's thumbnail sits inside its piece and a tap says
//  where it looked and how old it is (acceptance 4).
//

import UIKit
import XCTest

final class WorldImageryUITests: XCTestCase {

    private var app: XCUIApplication!
    private var mock: MockTowerHTTPServer!
    private var socket: PanelSocket!
    private var mockAuthority = ""
    private var seq = 10

    private static let liveDigest = "a3f1" + String(repeating: "0", count: 24) + "e02c"
    private static let liveDigest2 = "a3f2" + String(repeating: "0", count: 24) + "e02c"
    private static let landedDigest = "7cde" + String(repeating: "0", count: 24) + "1190"
    private static let liveFixture = #"{"version":1,"live":{"segment_index":812,"entries":[{"keyframe_id":"s9:4417","digest":"\#(liveDigest)","redaction":"redacted","pose_source":"segment_local","component_reference_segment":null,"segment_index":812}]}}"#
    private static let landedFixture = #"{"version":1,"landed":{"solved_at":1791240313.0,"geometry_revision":"g-stop-984","frame_revision":1,"entries":[{"keyframe_id":"s9:120","digest":"\#(landedDigest)","redaction":"redacted","pose_source":"landed_global_solve","component_reference_segment":0,"segment_index":null}]}}"#
    /// §9's stop block, solved 12 s before this report was sent.
    private static let sentAfterStopSolve = 1791240325.0

    override func setUpWithError() throws {
        continueAfterFailure = false
        mock = try MockTowerHTTPServer(.tower)
        mockAuthority = "127.0.0.1:\(try mock.start())"
        socket = PanelSocket(mock: mock)
        socket.pinnedReply = { subscription, world, session in
            PanelReport.savedWorld(subscription: subscription, world: world, session: session)
        }
        socket.install()
        mock.setRoute(CaptureHealthUITests.sessionStart, status: 200, body: DeadEndsUITests.session(state: "active"))
        mock.setRoute(CaptureHealthUITests.sessionStop, status: 200, body: DeadEndsUITests.session(state: "stopped"))
        // Every imagery route answers, so an OFF phone that asked would be seen asking.
        mock.setRoute("GET \(Self.imagery)/manifest", status: 200, body: Self.liveFixture)
        for (digest, color) in [(Self.liveDigest, UIColor.systemTeal), (Self.liveDigest2, .systemPink),
                                (Self.landedDigest, .systemOrange)] {
            mock.setDataRoute("GET \(Self.imagery)/tile/\(digest)", data: Self.jpeg(color), contentType: "image/jpeg")
        }
    }

    override func tearDownWithError() throws {
        app?.terminate()
        mock?.stop()
    }

    private static let imagery = "/worlds/w1/guidance/s9/imagery"

    // MARK: Acceptance 2: OFF

    /// Both switches off: the map and the current piece as today, and no
    /// strip, no thumbnail, no request to an imagery route.
    func testBothSwitchesOffDrawNoImageryAndAskNothing() throws {
        launch(environment: [:])
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
        try startCapture()
        serve(revision: "gA", segments: [Self.placedZero, Self.live812])
        push(geometry: "gA", guidance: FOWGuidance.stop)
        XCTAssertTrue(element("wb-current-piece").waitForExistence(timeout: 15), "today's current piece")
        XCTAssertTrue(element("wb-panel-map").waitForExistence(timeout: 10), "today's map")
        push(geometry: "gA", guidance: FOWGuidance.stop)
        Thread.sleep(forTimeInterval: 2)
        XCTAssertFalse(element("wb-imagery-live").exists, "a strip with B1 off")
        XCTAssertFalse(element("wb-imagery-live-thumb").exists)
        XCTAssertFalse(element("wb-imagery-landed-thumb").exists, "a thumbnail with B2 off")
        let asked = mock.requestLines.filter { $0.contains("/imagery/") }
        XCTAssertEqual(asked, [], "OFF asked the imagery routes")
        XCTAssertTrue(mock.requestLines.contains { $0.contains("/geometry/manifest") }, "the geometry was fetched")
        shoot("imagery-off")
    }

    // MARK: Acceptance 3: B1

    /// B1 on with §4's live fixture: one thumbnail, never on the map; a
    /// segment bump clears it; and after Stop it is gone, a late report of
    /// the walk never bringing it back, the loading view drawing none.
    func testTheLiveStripShowsOneThumbnailClearsAtABumpAndGoesAtStop() throws {
        launch(environment: ["IOS_FOW_ROOM_IMAGERY_LIVE": "on"])
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
        try startCapture()
        serve(revision: "gA", segments: [Self.placedZero, Self.live812])
        push(geometry: "gA", guidance: FOWGuidance.stop)
        let strip = element("wb-imagery-live")
        let thumbs = app.descendants(matching: .any).matching(identifier: "wb-imagery-live-thumb")
        XCTAssertTrue(waitFor(timeout: 15) { thumbs.count == 1 }, "one thumbnail: \(thumbs.count)")
        XCTAssertEqual(text(element("wb-imagery-live-label")), "Current piece · unplaced")
        XCTAssertEqual(thumbs.firstMatch.label, "current piece photo, unplaced")
        let map = element("wb-panel-map")
        XCTAssertTrue(map.waitForExistence(timeout: 10), "the map")
        XCTAssertFalse(map.frame.intersects(strip.frame), "the strip is drawn against the map")
        XCTAssertFalse(element("wb-imagery-landed-thumb").exists, "a landed thumbnail with B2 off")
        reveal(strip)
        shoot("imagery-live-812")

        // A tracking break: segment 813 is current and its strip starts empty.
        mock.setRoute("GET \(Self.imagery)/manifest", status: 200,
                      body: #"{"version":1,"live":{"segment_index":813,"entries":[]}}"#)
        serve(revision: "gB", segments: [Self.placedZero, Self.live812, Self.live813])
        push(geometry: "gB", guidance: FOWGuidance.stop)
        XCTAssertTrue(waitFor(timeout: 15) { thumbs.count == 0 }, "cleared at the bump: \(thumbs.count)")
        XCTAssertTrue(element("wb-imagery-live-empty").waitForExistence(timeout: 5), "the new strip, empty")
        shoot("imagery-live-813-empty")

        // 813's first photo, then Stop.
        mock.setRoute("GET \(Self.imagery)/manifest", status: 200, body: #"{"version":1,"live":{"segment_index":813,"entries":[{"keyframe_id":"s9:4500","digest":"\#(Self.liveDigest2)","redaction":"redacted","pose_source":"segment_local","component_reference_segment":null,"segment_index":813}]}}"#)
        serve(revision: "gC", segments: [Self.placedZero, Self.live812, Self.live813])
        push(geometry: "gC", guidance: FOWGuidance.stop)
        XCTAssertTrue(waitFor(timeout: 15) { thumbs.count == 1 }, "813's photo")
        let stop = app.buttons["Stop capture"]
        reveal(stop)
        XCTAssertTrue(tap(stop, until: !self.app.buttons["Stop capture"].exists), "Stop")
        XCTAssertTrue(waitFor(timeout: 5) { !strip.exists }, "the strip after the local Stop")
        // The Tower's late reports of the stopped walk, a new revision among them.
        serve(revision: "gD", segments: [Self.placedZero, Self.live812, Self.live813])
        for round in 0..<3 {
            push(geometry: round == 0 ? "gD" : "gC", guidance: FOWGuidance.stop)
            XCTAssertTrue(neverExists(strip, for: 1), "the strip came back after Stop (\(round))")
        }
        push(modelState: "finalizing", geometry: "gD", guidance: FOWGuidance.stop, buildInProgress: true)
        XCTAssertTrue(neverExists(strip, for: 1.5), "the strip in the loading view")
        XCTAssertFalse(element("wb-imagery-live-thumb").exists)
        XCTAssertFalse(element("wb-imagery-landed-thumb").exists)
        shoot("imagery-live-after-stop")
    }

    // MARK: Acceptance 4: B2

    /// B2 on with §4's landed fixture beside §9's stop block: the thumbnail
    /// inside its piece, a tap saying where it looked and how old it is; a
    /// newer landing's grid drops it.
    func testTheLandedThumbnailSitsInItsPieceAndATapSaysDirectionAndAge() throws {
        mock.setRoute("GET \(Self.imagery)/manifest", status: 200, body: Self.landedFixture)
        launch(environment: ["IOS_FOW_ROOM_IMAGERY_LANDED": "on"])
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
        try startCapture()
        serve(revision: "gA", segments: [Self.placedZero, Self.live812])
        push(geometry: "gA", guidance: FOWGuidance.stop)
        let thumb = element("wb-imagery-landed-thumb")
        XCTAssertTrue(thumb.waitForExistence(timeout: 15), "the landed thumbnail")
        let piece = element("wb-panel-map-piece")
        XCTAssertTrue(piece.waitForExistence(timeout: 5), "its piece")
        print("IMAGERY-LANDED|thumb=\(thumb.frame)|piece=\(piece.frame)")
        XCTAssertTrue(piece.frame.insetBy(dx: -0.5, dy: -0.5).contains(thumb.frame),
                      "the thumbnail \(thumb.frame) outside its piece \(piece.frame)")
        XCTAssertTrue(thumb.label.hasPrefix("redacted photo, look direction 90°, "), thumb.label)
        XCTAssertTrue(thumb.label.contains("seconds old"), thumb.label)
        XCTAssertEqual(text(piece), "Piece 1", "the v1 piece's label changed")
        XCTAssertFalse(element("wb-imagery-live").exists, "a strip with B1 off")
        reveal(thumb)
        shoot("imagery-landed")

        XCTAssertTrue(tap(thumb, until: self.element("wb-imagery-landed-direction").exists), "the sheet")
        XCTAssertEqual(text(element("wb-imagery-landed-direction")), "Look direction 90°")
        let age = text(element("wb-imagery-landed-age"))
        XCTAssertTrue(age.hasSuffix("seconds old") || age.hasSuffix("seconds old, stale"), age)
        shoot("imagery-landed-sheet")
        XCTAssertTrue(tap(app.buttons["Done"], until: !self.element("wb-imagery-landed-direction").exists),
                      "the sheet closed")

        // A newer landing on the map; the Tower's manifest still carries the
        // stop landing's thumbnails: none is drawn on the new grid.
        push(geometry: "gA", guidance: FOWGuidance.mid)
        XCTAssertTrue(waitFor(timeout: 10) { !thumb.exists }, "a superseded landing's thumbnail")
        XCTAssertTrue(element("wb-panel-map").exists, "the map stays")
    }

    // MARK: The mock Tower's geometry

    private typealias Segment = (index: Int, hash: String, placed: Bool, poses: [String])

    private static func pose(_ id: String, _ t: [Double], yaw: Double = 0) -> String {
        let q = [cos(yaw / 2), 0, sin(yaw / 2), 0]
        return #"{"keyframe_id":"\#(id)","status":"solved","degeneracy":"","rotation":\#(q),"translation":\#(t)}"#
    }

    /// Placed in reference 0, frame 1, identity: `s9:120` at (3, -1, 0)
    /// looking along +x -- in §9's stop grid, station (3, 5), 90 degrees.
    private static let placedZero: Segment = (0, "h0", true, [pose("s9:120", [3, -1, 0], yaw: .pi / 2),
                                                              pose("s9:121", [3.2, -1, 0], yaw: .pi / 2)])
    private static let live812: Segment = (812, "h812", false, [pose("s9:4416", [0, 0, 0]),
                                                                pose("s9:4417", [0.4, 0, 0.8], yaw: 0.4)])
    private static let live813: Segment = (813, "h813", false, [pose("s9:4499", [0, 0, 0]),
                                                                pose("s9:4500", [-0.5, 0, 0.6], yaw: -0.6)])

    private static let contract = "world_builder.geometry/2026-08-25"

    private static func placement(_ placed: Bool) -> String {
        placed
            ? #""registered":true,"transform_to_world":{"rotation_wxyz":[1.0,0.0,0.0,0.0],"translation":[0.0,0.0,0.0],"scale":1.0,"reference_segment":0,"frame_revision":1}"#
            : #""registered":false,"transform_to_world":null"#
    }

    private func serve(revision: String, segments: [Segment]) {
        let rows = segments.map { segment in
            #"{"segment_index":\#(segment.index),"content_hash":"\#(segment.hash)","#
                + #""frame_id":"segment:\#(segment.index)",\#(Self.placement(segment.placed)),"#
                + #""resolution_state":"resolved","dominant_degeneracy":null,"#
                + #""keyframe_count":\#(segment.poses.count),"solved_count":\#(segment.poses.count),"#
                + #""point_count":1,"bounds":{"min":[0.0,0.0,0.0],"max":[1.0,1.0,1.0]}}"#
        }
        mock.setRoute("GET /worlds/w1/geometry/manifest", status: 200, body:
            #"{"contract":"\#(Self.contract)","world_id":"w1","session_id":"s9","#
            + #""geometry_revision":"\#(revision)","current":true,"pose_convention":{"pose_type":"T_world_camera","#
            + #""quaternion_order":"wxyz","handedness":"right","camera_axes":"opencv_x_right_y_down_z_forward","#
            + #""translation_units":"world","world_axes_origin":"first_keyframe_camera","up_axis":"unknown","#
            + #""pose_dtype":"float64","point_dtype":"float32"},"segment_count":\#(segments.count),"#
            + #""segments":[\#(rows.joined(separator: ","))]}"#)
        for segment in segments {
            mock.setRoute("GET /worlds/w1/geometry/segment/\(segment.index)", status: 200, body:
                #"{"contract":"\#(Self.contract)","segment_index":\#(segment.index),"#
                + #""content_hash":"\#(segment.hash)","frame_id":"segment:\#(segment.index)","#
                + #"\#(Self.placement(segment.placed)),"#
                + #""poses":[\#(segment.poses.joined(separator: ","))],"points":[[0.5,0.5,0.5]],"#
                + #""points_sent":1,"points_total":1,"point_sampling":"none"}"#)
        }
    }

    private func push(modelState: String = "receiving", geometry: String, guidance: String? = nil,
                      buildInProgress: Bool? = nil) {
        seq += 1
        let seq = self.seq
        socket.sendLive { subscription in
            PanelReport.text(subscription: subscription, seq: seq, modelState: modelState, world: "w1",
                             session: "s9", buildInProgress: buildInProgress, liveCapture: true,
                             guidance: guidance, accepted: 984, towerSentAt: Self.sentAfterStopSolve)
                .replacingOccurrences(of: #""revision":"g1"}"#, with: #""revision":"\#(geometry)"}"#)
        }
        Thread.sleep(forTimeInterval: 0.7)
    }

    private static func jpeg(_ color: UIColor) -> Data {
        let size = CGSize(width: 160, height: 90)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        return UIGraphicsImageRenderer(size: size, format: format).jpegData(withCompressionQuality: 0.6) { context in
            color.setFill()
            context.fill(CGRect(origin: .zero, size: size))
            UIColor.white.setFill()
            context.fill(CGRect(x: 60, y: 30, width: 40, height: 30))
        }
    }

    // MARK: Driving the app

    private func launch(environment: [String: String]) {
        let app = XCUIApplication()
        app.launchArguments = ["-UITestSkipOnboarding", "-UITestResetTowerAddress", "-UITestMockGlasses"]
        app.launchEnvironment["GLASSES_UITEST_AWAITING_BOUND_SECONDS"] = "600"
        switch DeadEndsUITests.cameraFeed {
        case .success(let feed): app.launchEnvironment["GLASSES_UITEST_MOCK_CAMERA_FEED"] = feed.path
        case .failure(let problem): XCTFail("the mock glasses' camera feed: \(problem.text)")
        }
        app.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = mockAuthority
        for (key, value) in environment { app.launchEnvironment[key] = value }
        addUIInterruptionMonitor(withDescription: "system alert") { alert in
            for title in ["Allow", "Don't Allow", "OK", "Not Now"] where alert.buttons[title].exists {
                alert.buttons[title].tap()
                return true
            }
            return false
        }
        app.launch()
        self.app = app
    }

    private func startCapture() throws {
        let start = app.buttons["Start capture"]
        guard waitFor(timeout: 15, { start.exists && start.isEnabled }) else {
            throw XCTSkip("Mock Device Kit gave no active device here")
        }
        start.tap()
        XCTAssertTrue(app.buttons["Stop capture"].waitForExistence(timeout: 15), "the capture started")
    }

    private func open(cartridge name: String) {
        let cartridges = app.buttons["Cartridges"]
        XCTAssertTrue(cartridges.waitForExistence(timeout: 15), "the shell's Cartridges button")
        let drawerDone = app.buttons["Done"]
        XCTAssertTrue(tap(cartridges, until: drawerDone.exists), "the cartridge drawer opened")
        app.raiseTheCartridgeDrawerOnAnIPad()
        let row = app.buttons.containing(NSPredicate(format: "label BEGINSWITH %@", name)).firstMatch
        var found = waitFor(timeout: 3) { row.exists && row.isHittable }
        for _ in 0..<6 where !found {
            app.swipeUp(velocity: .slow)
            found = waitFor(timeout: 1) { row.exists && row.isHittable }
        }
        XCTAssertTrue(found, "a drawer row for \(name)")
        XCTAssertTrue(tap(row, until: !drawerDone.exists && app.navigationBars[name].exists),
                      "the \(name) workspace opened")
    }

    private func element(_ identifier: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    private func text(_ element: XCUIElement) -> String {
        element.exists ? element.label : "(gone)"
    }

    private func reveal(_ element: XCUIElement) {
        for _ in 0..<8 where !(element.exists && element.isHittable) { app.swipeUp(velocity: .slow) }
    }

    @discardableResult
    private func tap(_ element: XCUIElement, until effect: @autoclosure () -> Bool, attempts: Int = 4) -> Bool {
        for _ in 0..<attempts {
            if element.waitForExistence(timeout: 5), element.isHittable { element.tap() }
            if waitFor(timeout: 4, effect) { return true }
        }
        return effect()
    }

    private func neverExists(_ element: XCUIElement, for seconds: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            if element.exists { return false }
            Thread.sleep(forTimeInterval: 0.1)
        }
        return !element.exists
    }

    private func waitFor(timeout: TimeInterval = 10, _ condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            Thread.sleep(forTimeInterval: 0.2)
        }
        return condition()
    }

    private func shoot(_ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
