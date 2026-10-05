//
//  CaptureHealthUITests.swift
//  GlassesUITests
//
//  U2-D0 "Capture health" on the live-capture mock: `MockTowerHTTPServer`
//  answers the World Builder session start and speaks the result socket,
//  sending World Builder reports as the Tower does (spec §3, tests 6 and 7,
//  and the panel's honesty on a real socket). No Tower needed.
//

import XCTest

final class CaptureHealthUITests: XCTestCase {

    private var app: XCUIApplication!
    private var mock: MockTowerHTTPServer!
    private var mockAuthority = ""
    private let script = SocketScript()
    private let reports = ReportPump()
    private static let ax5 = "UICTContentSizeCategoryAccessibilityXXXL"

    override func setUpWithError() throws {
        continueAfterFailure = false
        mock = try MockTowerHTTPServer(.tower)
        mockAuthority = "127.0.0.1:\(try mock.start())"
    }

    override func tearDownWithError() throws {
        reports.stop()
        app?.terminate()
        mock?.stop()
    }

    // MARK: The live walk's figures, and their age

    /// With a World Builder session active and reports arriving, each row
    /// reads the walk's figures; when the reports stop, every figure turns
    /// to "—" within the panel's 5 s, while the link itself is still up.
    func testThePanelReadsTheLiveWalkAndGoesBlankWhenReportsStop() throws {
        openWorldBuilder(session: true, keyframesPerReport: 1)
        let panel = element("capture-health")
        XCTAssertTrue(panel.waitForExistence(timeout: 20), "the panel, with a session active")
        let pace = element("capture-health-pace")
        XCTAssertEqual(element("capture-health-link").label, "Tower: connected")
        let breaks = element("capture-health-breaks")
        XCTAssertEqual(breaks.label, "Breaks in the last 30 seconds: unknown",
                       "the panel has just started watching and has seen no break: not 0")
        XCTAssertTrue(waitFor(timeout: 20) { self.number(in: pace.label).map { $0 > 0 } ?? false },
                      "keyframes per minute from the reports: \(pace.label)")
        XCTAssertTrue(pace.label.hasSuffix(" keyframes/min"), pace.label)
        reports.restarts = 1
        XCTAssertTrue(waitFor { breaks.label == "Breaks in the last 30 seconds: at least 1" }, "breaks: \(breaks.label)")
        XCTAssertEqual(element("capture-health-lookback").label,
                       "Linked back to what you saw before. linked back 2 · could not link 1")
        XCTAssertEqual(element("capture-health-map").label, "Map: 3 keyframes behind")
        XCTAssertFalse(element("capture-health-stalled").exists, "no capture runs, so no stall is claimed")
        XCTAssertFalse(app.staticTexts.containing(NSPredicate(format: "label CONTAINS[c] %@", "Tracking: lost"))
            .firstMatch.exists)
        shoot("capture-health-live")

        reports.stop()
        XCTAssertTrue(waitFor(timeout: 10) { pace.label == "— keyframes/min" }, "old figures are not live: \(pace.label)")
        XCTAssertEqual(breaks.label, "Breaks in the last 30 seconds: unknown")
        XCTAssertEqual(element("capture-health-lookback").label, "Look-back: —")
        XCTAssertEqual(element("capture-health-map").label, "Map: —")
        XCTAssertEqual(element("capture-health-link").label, "Tower: connected", "the link is still up")
        shoot("capture-health-stale")
    }

    // MARK: Layout, and the screen without a session (tests 6 and 7)

    func testThePanelMovesNothingAboveItAtTheDefaultSize() throws {
        try assertThePanelMovesNothingAboveIt(size: nil, name: "default")
    }

    func testThePanelMovesNothingAboveItAtAX5() throws {
        try assertThePanelMovesNothingAboveIt(size: Self.ax5, name: "ax5")
    }

    /// The same screen with no World Builder session (the session route is
    /// not there: the panel is absent, today's screen) and with one (the
    /// panel, below the capture control): the capture control and everything
    /// above it are where they were.
    private func assertThePanelMovesNothingAboveIt(size: String?, name: String) throws {
        openWorldBuilder(session: false, size: size)
        let start = app.buttons["Start capture"]
        XCTAssertTrue(start.waitForExistence(timeout: 15), "the capture control")
        XCTAssertTrue(waitFor(timeout: 10) { self.element("wb-session-footnote").exists }, "the session line")
        Thread.sleep(forTimeInterval: 3)
        XCTAssertFalse(element("capture-health").exists, "no session: no panel")
        let without = start.frame
        let footnoteWithout = element("wb-session-footnote").frame
        shoot("capture-health-\(name)-off")
        app.terminate()
        reports.stop()

        openWorldBuilder(session: true, size: size)
        let panel = element("capture-health")
        XCTAssertTrue(panel.waitForExistence(timeout: 20), "a session: the panel")
        Thread.sleep(forTimeInterval: 2)
        let with = app.buttons["Start capture"].frame
        let window = app.windows.firstMatch.frame
        print("U2D0-LAYOUT|\(name)|window=\(Int(window.width))x\(Int(window.height))|start=\(Int(with.minY))-\(Int(with.maxY))"
              + "|panel=\(Int(panel.frame.minY))-\(Int(panel.frame.maxY))|footnote-off=\(Int(footnoteWithout.minY))"
              + "|start-on-screen=\(with.maxY <= window.maxY)")
        XCTAssertEqual(with.minY, without.minY, accuracy: 0.5, "the panel moved the capture control")
        XCTAssertEqual(with.height, without.height, accuracy: 0.5)
        XCTAssertGreaterThanOrEqual(panel.frame.minY, with.maxY - 0.5, "the panel is below the capture control")
        shoot("capture-health-\(name)-on")
    }

    /// Spec test 6's Stop: a capture running on mock glasses, the panel up,
    /// and Stop on the screen at the default size -- above the panel, which
    /// never pushes it down.
    func testStopStaysOnTheScreenWithThePanelUp() throws {
        mock.setRoute(Self.sessionStart, status: 200, body: DeadEndsUITests.session(state: "active"))
        mock.setRoute(Self.sessionStop, status: 200, body: DeadEndsUITests.session(state: "stopped"))
        scriptSocket()
        launch(size: nil, mockGlasses: true)
        open(cartridge: "World Builder")
        let start = app.buttons["Start capture"]
        guard waitFor(timeout: 15, { start.exists && start.isEnabled }) else {
            throw XCTSkip("Mock Device Kit gave no active device in this Simulator")
        }
        XCTAssertTrue(reveal(start), "Start capture is on the screen")
        start.tap()
        let stop = app.buttons["Stop capture"]
        XCTAssertTrue(stop.waitForExistence(timeout: 15), "the capture started")
        let panel = element("capture-health")
        XCTAssertTrue(panel.waitForExistence(timeout: 10), "a capture: the panel")
        Thread.sleep(forTimeInterval: 2)
        let window = app.windows.firstMatch.frame
        print("U2D0-STOP|window=\(Int(window.width))x\(Int(window.height))|stop=\(Int(stop.frame.minY))-\(Int(stop.frame.maxY))"
              + "|panel=\(Int(panel.frame.minY))")
        XCTAssertTrue(stop.isHittable, "Stop can be tapped")
        XCTAssertLessThanOrEqual(stop.frame.maxY, window.maxY, "Stop is on the screen")
        XCTAssertLessThanOrEqual(stop.frame.maxY, panel.frame.minY + 0.5, "Stop is above the panel")
        shoot("capture-health-stop")
        stop.tap()
    }

    // MARK: The mock

    static let sessionStart = "POST /cartridges/world_builder/session/start"
    static let sessionStop = "POST /cartridges/world_builder/session/stop"

    private func openWorldBuilder(session: Bool, size: String? = nil, keyframesPerReport: Int = 0) {
        if session {
            mock.setRoute(Self.sessionStart, status: 200, body: DeadEndsUITests.session(state: "active"))
            mock.setRoute(Self.sessionStop, status: 200, body: DeadEndsUITests.session(state: "stopped"))
        } else {
            mock.setRoute(Self.sessionStart, status: 404, body: #"{"detail":"Not Found"}"#)
        }
        scriptSocket()
        reports.start(mock: mock, script: script, keyframesPerReport: keyframesPerReport)
        launch(size: size, mockGlasses: false)
        open(cartridge: "World Builder")
    }

    /// The Tower's side of the socket, as `DeadEndsUITests` scripts it.
    private func scriptSocket() {
        mock.acceptsWebSocket = true
        let script = self.script
        let mock = self.mock!
        mock.onSocketText = { text in
            guard let data = text.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let type = json["type"] as? String
            else { return }
            switch type {
            case "ping":
                mock.sendSocket(text: #"{"type":"pong"}"#)
            case "cartridges":
                script.resetSubscriptions()
                mock.sendSocket(text: """
                    {"type":"cartridges",
                     "envelope_contract":"cartridge_results.envelope/2026-08-23",
                     "cartridges":[{"cartridge":"world_builder","result_type":"status",
                        "contract":"\(DeadEndsUITests.worldBuilderContract)","available":true,
                        "unavailable_reason":null,"snapshot_only":true}],
                     "not_offered":[]}
                    """)
            case "result_subscribe":
                let count = script.nextSubscription()
                let world = (json["world_id"] as? String).map { "\"\($0)\"" } ?? "null"
                let session = (json["session_id"] as? String).map { "\"\($0)\"" } ?? "null"
                mock.sendSocket(text: """
                    {"type":"result_subscribed",
                     "envelope_contract":"cartridge_results.envelope/2026-08-23",
                     "subscription_id":"sub-\(count)","cartridge":"world_builder",
                     "result_type":"status","contract":"\(DeadEndsUITests.worldBuilderContract)",
                     "snapshot_only":true,"world_id":\(world),"session_id":\(session),
                     "cursor_status":"absent"}
                    """)
            default:
                break
            }
        }
    }

    private func launch(size: String?, mockGlasses: Bool) {
        let app = XCUIApplication()
        app.launchArguments = ["-UITestSkipOnboarding", "-UITestResetTowerAddress"]
        if let size { app.launchArguments += ["-UIPreferredContentSizeCategoryName", size] }
        if mockGlasses {
            app.launchArguments.append("-UITestMockGlasses")
            switch DeadEndsUITests.cameraFeed {
            case .success(let feed): app.launchEnvironment["GLASSES_UITEST_MOCK_CAMERA_FEED"] = feed.path
            case .failure(let problem): XCTFail("the mock glasses' camera feed: \(problem.text)")
            }
        }
        app.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = mockAuthority
        addUIInterruptionMonitor(withDescription: "system alert") { alert in
            if alert.label == "Something went wrong" { return false }
            for title in ["Allow", "Don't Allow", "OK", "Not Now"] {
                let button = alert.buttons[title]
                if button.exists { button.tap(); return true }
            }
            return false
        }
        app.launch()
        self.app = app
    }

    // MARK: Helpers

    private func open(cartridge name: String) {
        let cartridges = app.buttons["Cartridges"]
        XCTAssertTrue(cartridges.waitForExistence(timeout: 15), "the shell's Cartridges button")
        let drawerDone = app.buttons["Done"]
        XCTAssertTrue(tap(cartridges, until: drawerDone.exists), "the cartridge drawer opened")
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

    private func number(in label: String) -> Int? {
        Int(label.prefix { $0.isNumber })
    }

    @discardableResult
    private func reveal(_ element: XCUIElement, attempts: Int = 12) -> Bool {
        for _ in 0..<attempts {
            if element.exists && element.isHittable { return true }
            let window = app.windows.firstMatch.frame
            let isAbove = element.exists && element.frame.midY < window.midY
            let from = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: isAbove ? 0.4 : 0.7))
            let to = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: isAbove ? 0.6 : 0.5))
            from.press(forDuration: 0.1, thenDragTo: to, withVelocity: .slow, thenHoldForDuration: 0.1)
        }
        return element.exists && element.isHittable
    }

    @discardableResult
    private func tap(_ element: XCUIElement, until effect: @autoclosure () -> Bool, attempts: Int = 4) -> Bool {
        for _ in 0..<attempts {
            if element.waitForExistence(timeout: 5), element.isHittable { element.tap() }
            if waitFor(timeout: 3, effect) { return true }
        }
        return effect()
    }

    private func waitFor(timeout: TimeInterval = 10, _ condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            Thread.sleep(forTimeInterval: 0.2)
        }
        return condition()
    }

    /// A screenshot, kept, and written to U2_SHOTS_DIR when it is set.
    private func shoot(_ name: String) {
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        guard let dir = ProcessInfo.processInfo.environment["U2_SHOTS_DIR"], !dir.isEmpty else { return }
        let suffix = ProcessInfo.processInfo.environment["U2_SHOT_SUFFIX"].map { "-\($0)" } ?? ""
        try? screenshot.pngRepresentation.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name)\(suffix).png"))
    }
}

/// Sends a World Builder report every second on the subscription the script
/// last acknowledged, as the Tower's poll and heartbeat do: a live walk whose
/// keyframe counter rises by `keyframesPerReport`, `restarts` tracking
/// restarts, the relocalizer's counts, and a map 3 keyframes behind.
final class ReportPump: @unchecked Sendable {
    private let lock = NSLock()
    private var timer: DispatchSourceTimer?
    private var restartCount = 0

    /// `trajectory.tracking_restarts` in the next reports.
    var restarts: Int {
        get { lock.withLock { restartCount } }
        set { lock.withLock { restartCount = newValue } }
    }
    private let queue = DispatchQueue(label: "capture-health.reports")

    func start(mock: MockTowerHTTPServer, script: SocketScript, keyframesPerReport: Int) {
        stop()
        restarts = 0
        var seq = 0
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 1, repeating: 1)
        timer.setEventHandler {
            guard script.subscribeCount > 0 else { return }
            seq += 1
            let keyframes = 20 + seq * keyframesPerReport
            mock.sendSocket(text: Self.report(subscription: "sub-\(script.subscribeCount)", seq: seq,
                                              keyframes: keyframes, restarts: self.restarts))
        }
        lock.withLock { self.timer = timer }
        timer.resume()
    }

    func stop() {
        lock.withLock {
            timer?.cancel()
            timer = nil
        }
    }

    static func report(subscription: String, seq: Int, keyframes: Int, restarts: Int) -> String {
        """
        {"type":"cartridge_result",
         "envelope_contract":"cartridge_results.envelope/2026-08-23",
         "subscription_id":"\(subscription)","cartridge":"world_builder","result_type":"status",
         "contract":"\(DeadEndsUITests.worldBuilderContract)","seq":\(seq),"revision":"r\(keyframes)",
         "revision_changed":true,"coalesced":0,"cursor_status":null,
         "snapshot":true,"tower_sent_at":1787463092.9,"time_basis":"tower-receipt",
         "payload":{"model_state":"receiving","model_state_reason":null,
           "world_snapshot":{"name":"Probe Room","world_id":"w1",
             "keyframe_count":\(keyframes),"revision":"r\(keyframes)",
             "tracking":"good","scale":"relative","mapping_seconds":12.5,
             "calibration":"calibrated",
             "geometry":{"representation":"sparse point cloud","element_count":1360,"is_incremental":false},
             "trajectory":{"pose_count":\(keyframes),"path_length":2.85,
                           "path_length_unit":"world units","scale":"relative"},
             "persistence":{"state":"saved","revision":"p1"}},
           "geometry":{"available":true,"current":false,"built_from_keyframes":\(keyframes - 3),
                       "keyframes_now":\(keyframes),"revision":"g1"},
           "trajectory":{"tracking_restarts":\(restarts),"chain_breaks":0,"segments":2},
           "tracking":{"recovery":{"state":"recovered","episode":2,"prompts_enabled":true,"prompt":null,
             "counts":{"episodes":3,"recovered":2,"recovered_after_prompt":1,"timed_out":1,
                       "prompts":1,"withheld_by_limiter":0,"withheld_disabled":0}}}}}
        """
    }
}
