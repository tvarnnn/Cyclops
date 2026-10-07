//
//  WorldCurrentPieceUITests.swift
//  GlassesUITests
//
//  Fog of war's current piece (FOW-CURRENT-PIECE-DATA-20261006) against the
//  mock Tower, which serves segment geometry: the card appears with its
//  label beside (never inside) the map, restarts at a new segment, is the
//  walk's own, goes after Stop, and never moves Stop.
//

import XCTest

final class WorldCurrentPieceUITests: XCTestCase {

    private var app: XCUIApplication!
    private var mock: MockTowerHTTPServer!
    private var socket: PanelSocket!
    private var mockAuthority = ""
    private var seq = 10

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
    }

    override func tearDownWithError() throws {
        app?.terminate()
        mock?.stop()
    }

    // MARK: Fills in, resets, stays out of the map, the walk's own

    func testTheCurrentPieceFillsInRestartsAtABreakAndIsItsWalksOwn() throws {
        launch()
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")

        // No segment data (the manifest 404s): nothing drawn, today's panel.
        push(modelState: "receiving", geometry: "g0")
        XCTAssertTrue(element("capture-health").waitForExistence(timeout: 15), "Capture health")
        Thread.sleep(forTimeInterval: 1.5)
        XCTAssertFalse(element("wb-current-piece").exists, "no segment data: no piece")
        XCTAssertFalse(element("wb-panel-stage").exists, "no segment data: today's panel")

        // Segment 0, three posed keyframes: the piece, beside the map.
        serve(revision: "gA", segments: [(0, "h0", Self.zeroPoses)])
        push(modelState: "receiving", geometry: "gA", guidance: FOWGuidance.mid, accepted: 127)
        let piece = element("wb-current-piece")
        XCTAssertTrue(piece.waitForExistence(timeout: 15), "the current piece")
        XCTAssertEqual(text(element("wb-current-piece-label")), "Provisional · current stretch")
        XCTAssertEqual(text(element("wb-current-piece-views")), "3 views since tracking last restarted")
        let map = element("wb-panel-map")
        XCTAssertTrue(map.waitForExistence(timeout: 10), "the room map")
        XCTAssertFalse(map.frame.intersects(piece.frame), "the piece is drawn into the room map")
        XCTAssertFalse(element("wb-panel-stage").frame.intersects(piece.frame), "the piece is in the map's stage")
        reveal(piece)
        shoot("current-piece-segment-0")

        // A tracking break: segment 1 is current, and the piece starts over.
        serve(revision: "gB", segments: [(0, "h0", Self.zeroPoses), (1, "h1", Self.onePoses)])
        push(modelState: "receiving", geometry: "gB", guidance: FOWGuidance.mid, accepted: 129)
        XCTAssertTrue(waitFor(timeout: 15) {
            self.text(self.element("wb-current-piece-views")) == "2 views since tracking last restarted"
        }, "after the break: \(text(element("wb-current-piece-views")))")
        shoot("current-piece-segment-1")

        // Another walk's report: A's piece goes.
        push(modelState: "receiving", world: "w2", session: "s9", geometry: "gW2")
        XCTAssertTrue(waitFor(timeout: 10) { !self.element("wb-current-piece").exists }, "A's piece under B's walk")
        XCTAssertTrue(element("capture-health").exists, "B's walking panel")
    }

    // MARK: Stop

    /// A capture on the mock glasses: the piece appearing leaves Stop where
    /// it was, and after Stop (the Tower then finalizing) it is gone.
    func testTheCurrentPieceNeverMovesStopAndGoesAfterStop() throws {
        launch(mockGlasses: true)
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
        let start = app.buttons["Start capture"]
        guard waitFor(timeout: 15, { start.exists && start.isEnabled }) else {
            throw XCTSkip("Mock Device Kit gave no active device here")
        }
        start.tap()
        let stop = app.buttons["Stop capture"]
        XCTAssertTrue(stop.waitForExistence(timeout: 15), "the capture started")
        let top = app.buttons["Saved worlds"]
        push(modelState: "receiving", geometry: "g0", liveCapture: true)
        Thread.sleep(forTimeInterval: 2)
        XCTAssertFalse(element("wb-current-piece").exists)
        let without = stop.frame.minY - top.frame.minY

        serve(revision: "gA", segments: [(0, "h0", Self.zeroPoses)])
        push(modelState: "receiving", geometry: "gA", liveCapture: true)
        let piece = element("wb-current-piece")
        XCTAssertTrue(piece.waitForExistence(timeout: 15), "the current piece while capturing")
        Thread.sleep(forTimeInterval: 1)
        let with = stop.frame.minY - top.frame.minY
        print("CURRENT-PIECE-STOP|without=\(without)|with=\(with)|piece=\(piece.frame.minY)|stop-max=\(stop.frame.maxY)")
        XCTAssertEqual(with, without, accuracy: 0.5, "the piece moved Stop")
        XCTAssertGreaterThanOrEqual(piece.frame.minY, stop.frame.maxY - 0.5, "the piece is above Stop")
        shoot("current-piece-capturing")

        reveal(stop)
        XCTAssertTrue(tap(stop, until: !self.app.buttons["Stop capture"].exists), "Stop")
        // Codex review MED: gone at the local Stop, with the Tower still
        // reporting `receiving` -- not only once a finalizing report lands.
        XCTAssertTrue(waitFor(timeout: 5) { !self.element("wb-current-piece").exists },
                      "the piece after the local Stop, before the finalizing report")
        push(modelState: "finalizing", geometry: "gA", liveCapture: true, buildInProgress: true)
        XCTAssertTrue(waitFor(timeout: 10) { !self.element("wb-current-piece").exists }, "the piece after Stop")
        shoot("current-piece-after-stop")
    }

    /// Codex review HIGH: Stop, then Start again before the Tower has moved
    /// on. It still reports the stopped walk `receiving`, and that walk's
    /// piece must not come back at any point; the next walk's piece shows
    /// once the Tower presents that walk.
    func testStartBeforeTheTowerAdvancesNeverBringsTheStoppedWalksPieceBack() throws {
        launch(mockGlasses: true)
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
        let start = app.buttons["Start capture"]
        guard waitFor(timeout: 15, { start.exists && start.isEnabled }) else {
            throw XCTSkip("Mock Device Kit gave no active device here")
        }
        start.tap()
        let stop = app.buttons["Stop capture"]
        XCTAssertTrue(stop.waitForExistence(timeout: 15), "the capture started")
        serve(revision: "gA", segments: [(0, "h0", Self.zeroPoses)])
        push(modelState: "receiving", geometry: "gA", liveCapture: true)
        let piece = element("wb-current-piece")
        XCTAssertTrue(piece.waitForExistence(timeout: 15), "the current piece while capturing")

        reveal(stop)
        XCTAssertTrue(tap(stop, until: !self.app.buttons["Stop capture"].exists), "Stop")
        XCTAssertTrue(waitFor(timeout: 5) { !piece.exists }, "the piece after the local Stop")

        // Start again; the Tower has not advanced and still reports s1.
        reveal(start)
        XCTAssertTrue(tap(start, until: self.app.buttons["Stop capture"].exists), "Start again")
        for round in 0..<4 {
            push(modelState: "receiving", geometry: "gA", liveCapture: true)
            XCTAssertTrue(neverExists(piece, for: 1), "the stopped walk's piece came back after Start (\(round))")
        }
        shoot("current-piece-restart-before-the-tower")

        // The Tower presents the new walk: its own piece.
        serve(world: "w2", session: "s9", revision: "gN", segments: [(0, "n0", Self.onePoses)])
        push(modelState: "receiving", world: "w2", session: "s9", geometry: "gN", liveCapture: true)
        XCTAssertTrue(piece.waitForExistence(timeout: 15), "the new walk's piece")
        XCTAssertEqual(text(element("wb-current-piece-views")), "2 views since tracking last restarted")
    }

    // MARK: The mock Tower's geometry

    private typealias Segment = (index: Int, hash: String, poses: [String])

    private static func pose(_ id: Int, _ t: [Double]?, yaw: Double = 0) -> String {
        guard let t else {
            return #"{"keyframe_id":"s1:\#(id)","status":"refused","degeneracy":"","rotation":null,"translation":null}"#
        }
        let q = [cos(yaw / 2), 0, sin(yaw / 2), 0]
        return #"{"keyframe_id":"s1:\#(id)","status":"solved","degeneracy":"","rotation":\#(q),"translation":\#(t)}"#
    }

    private static let zeroPoses = [pose(0, [0, 0, 0]), pose(1, [0.4, 0, 0.8], yaw: 0.4), pose(2, nil),
                                    pose(3, [1.0, 0, 1.2], yaw: 0.9)]
    private static let onePoses = [pose(0, [0, 0, 0]), pose(1, [-0.5, 0, 0.6], yaw: -0.6)]

    private static let contract = "world_builder.geometry/2026-08-25"
    private static let unplaced = #""registered":false,"transform_to_world":null"#

    /// The manifest naming `segments`, and each segment's chunk.
    private func serve(world: String = "w1", session: String = "s1", revision: String, segments: [Segment]) {
        let rows = segments.map { segment in
            #"{"segment_index":\#(segment.index),"content_hash":"\#(segment.hash)","#
                + #""frame_id":"segment:\#(segment.index)",\#(Self.unplaced),"resolution_state":"resolved","#
                + #""dominant_degeneracy":null,"keyframe_count":\#(segment.poses.count),"#
                + #""solved_count":\#(segment.poses.count),"point_count":1,"#
                + #""bounds":{"min":[0.0,0.0,0.0],"max":[1.0,1.0,1.0]}}"#
        }
        mock.setRoute("GET /worlds/\(world)/geometry/manifest", status: 200, body:
            #"{"contract":"\#(Self.contract)","world_id":"\#(world)","session_id":"\#(session)","#
            + #""geometry_revision":"\#(revision)","current":true,"pose_convention":{"pose_type":"T_world_camera","#
            + #""quaternion_order":"wxyz","handedness":"right","camera_axes":"opencv_x_right_y_down_z_forward","#
            + #""translation_units":"world","world_axes_origin":"first_keyframe_camera","up_axis":"unknown","#
            + #""pose_dtype":"float64","point_dtype":"float32"},"segment_count":\#(segments.count),"#
            + #""segments":[\#(rows.joined(separator: ","))]}"#)
        for segment in segments {
            mock.setRoute("GET /worlds/\(world)/geometry/segment/\(segment.index)", status: 200, body:
                #"{"contract":"\#(Self.contract)","segment_index":\#(segment.index),"#
                + #""content_hash":"\#(segment.hash)","frame_id":"segment:\#(segment.index)",\#(Self.unplaced),"#
                + #""poses":[\#(segment.poses.joined(separator: ","))],"points":[[0.5,0.5,0.5]],"#
                + #""points_sent":1,"points_total":1,"point_sampling":"none"}"#)
        }
    }

    /// A status report whose `geometry.revision` is `geometry`: a new
    /// revision is what sends the phone for the manifest and its segments.
    private func push(modelState: String, world: String = "w1", session: String = "s1", geometry: String,
                      liveCapture: Bool = false, guidance: String? = nil, accepted: Int? = nil,
                      buildInProgress: Bool? = nil) {
        seq += 1
        let seq = self.seq
        socket.sendLive { subscription in
            PanelReport.text(subscription: subscription, seq: seq, modelState: modelState, world: world,
                             session: session, buildInProgress: buildInProgress, liveCapture: liveCapture,
                             guidance: guidance, accepted: accepted)
                .replacingOccurrences(of: #""revision":"g1"}"#, with: #""revision":"\#(geometry)"}"#)
        }
        Thread.sleep(forTimeInterval: 0.7)
    }

    // MARK: Driving the app

    private func launch(mockGlasses: Bool = false) {
        let app = XCUIApplication()
        app.launchArguments = ["-UITestSkipOnboarding", "-UITestResetTowerAddress"]
        if mockGlasses {
            app.launchEnvironment["GLASSES_UITEST_AWAITING_BOUND_SECONDS"] = "600"
            app.launchArguments.append("-UITestMockGlasses")
            switch DeadEndsUITests.cameraFeed {
            case .success(let feed): app.launchEnvironment["GLASSES_UITEST_MOCK_CAMERA_FEED"] = feed.path
            case .failure(let problem): XCTFail("the mock glasses' camera feed: \(problem.text)")
            }
        }
        app.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = mockAuthority
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

    /// Best effort, for the screenshot only.
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

    /// True when `element` is absent at every sample for `seconds`.
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
