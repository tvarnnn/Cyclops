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

    // MARK: Stop, with the panel up (spec test 6; the lead's ruling)

    /// Stop sits below the fold (ruled acceptable for the demo). What these
    /// claim, and no more: the panel never moves Stop, and from the top of
    /// the screen ONE scroll brings Stop and the panel's first row on screen
    /// together, with Stop tappable. Neither claims Stop is visible at first.
    func testThePanelNeverMovesStopAndOneScrollShowsBothAtTheDefaultSize() throws {
        try assertStopWithThePanel(size: nil, name: "default", oneScroll: true)
    }

    /// At AX5 the capture control is screens down, panel or not: only that
    /// the panel never moves it is claimed.
    func testThePanelNeverMovesStopAtAX5() throws {
        try assertStopWithThePanel(size: Self.ax5, name: "ax5", oneScroll: false)
    }

    /// The same capture twice: with the panel held off (a DEBUG-only launch
    /// flag -- Stop exists only while a capture runs, and a capture always
    /// shows the panel, so this is the only "without"), then with it. Stop
    /// is measured from the top of the screen's content (the header's Saved
    /// worlds button), so a scroll changes neither figure.
    private func assertStopWithThePanel(size: String?, name: String, oneScroll: Bool) throws {
        let off = try startCapture(size: size, hidePanel: true)
        XCTAssertFalse(element("capture-health").exists, "the panel is held off")
        let without = off.stop.frame.minY - off.top.frame.minY
        let heightWithout = off.stop.frame.height
        shoot("capture-health-stop-\(name)-off")
        app.terminate()

        let on = try startCapture(size: size, hidePanel: false)
        let stop = on.stop
        let panel = element("capture-health")
        XCTAssertTrue(panel.exists, "a capture: the panel")
        let with = stop.frame.minY - on.top.frame.minY
        let window = app.windows.firstMatch.frame
        print("U2D0-STOP|\(name)|window=\(Int(window.width))x\(Int(window.height))|without=\(Int(without))"
              + "|with=\(Int(with))|stop-h=\(Int(stop.frame.height))|panel-below-by=\(Int(panel.frame.minY - stop.frame.maxY))")
        XCTAssertEqual(with, without, accuracy: 0.5, "the panel moved Stop")
        XCTAssertEqual(stop.frame.height, heightWithout, accuracy: 0.5)
        XCTAssertGreaterThanOrEqual(panel.frame.minY, stop.frame.maxY - 0.5, "the panel is below Stop")
        shoot("capture-health-stop-\(name)-on")
        guard oneScroll else { return }

        // The screen as the operator meets it: scrolled to the top.
        let top = on.top
        for _ in 0..<6 where abs(top.frame.minY - on.topAtFirst) > 0.5 { app.swipeDown(velocity: .fast) }
        XCTAssertTrue(waitFor(timeout: 5) { abs(top.frame.minY - on.topAtFirst) <= 0.5 }, "back at the top")
        Thread.sleep(forTimeInterval: 1)
        let first = element("capture-health-link")
        print("U2D0-STOP|\(name)|at-top|stop-hittable=\(stop.isHittable)|first-row-hittable=\(first.isHittable)")

        // ONE scroll: a drag up most of the window, held so nothing coasts.
        let from = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.88))
        let to = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.12))
        from.press(forDuration: 0.1, thenDragTo: to, withVelocity: .slow, thenHoldForDuration: 0.5)
        Thread.sleep(forTimeInterval: 1)
        print("U2D0-STOP|\(name)|one-scroll|stop=\(Int(stop.frame.minY))-\(Int(stop.frame.maxY))"
              + "|first-row=\(Int(first.frame.minY))-\(Int(first.frame.maxY))|window-h=\(Int(window.height))")
        XCTAssertTrue(stop.isHittable, "one scroll: Stop can be tapped")
        XCTAssertTrue(first.isHittable, "one scroll: the panel's first row is on the screen with Stop")
        XCTAssertGreaterThanOrEqual(stop.frame.minY, window.minY)
        XCTAssertLessThanOrEqual(first.frame.maxY, window.maxY)
        shoot("capture-health-stop-\(name)-one-scroll")
        stop.tap()
        XCTAssertTrue(app.buttons["Start capture"].waitForExistence(timeout: 15), "Stop stopped the capture")
    }

    /// No World Builder session, mock glasses, a capture started; settled
    /// once the viewfinder has frames. The first update's bound is pushed out
    /// so the canvas above Stop cannot change between the two launches.
    private func startCapture(size: String?, hidePanel: Bool) throws
        -> (stop: XCUIElement, top: XCUIElement, topAtFirst: CGFloat) {
        mock.setRoute(Self.sessionStart, status: 404, body: #"{"detail":"Not Found"}"#)
        scriptSocket()
        launch(size: size, mockGlasses: true, hidePanel: hidePanel)
        open(cartridge: "World Builder")
        let start = app.buttons["Start capture"]
        guard waitFor(timeout: 15, { start.exists && start.isEnabled }) else {
            throw XCTSkip("Mock Device Kit gave no active device in this Simulator")
        }
        let top = app.buttons["Saved worlds"]
        XCTAssertTrue(top.exists, "the header, the top of the content")
        let topAtFirst = top.frame.minY
        XCTAssertTrue(reveal(start), "Start capture can be reached")
        start.tap()
        let stop = app.buttons["Stop capture"]
        XCTAssertTrue(stop.waitForExistence(timeout: 15), "the capture started")
        if !hidePanel {
            XCTAssertTrue(element("capture-health").waitForExistence(timeout: 10), "a capture: the panel")
        }
        Thread.sleep(forTimeInterval: 3)
        return (stop, top, topAtFirst)
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

    private func launch(size: String?, mockGlasses: Bool, hidePanel: Bool = false) {
        let app = XCUIApplication()
        app.launchArguments = ["-UITestSkipOnboarding", "-UITestResetTowerAddress"]
        if let size { app.launchArguments += ["-UIPreferredContentSizeCategoryName", size] }
        if hidePanel { app.launchArguments.append("-UITestHideCaptureHealth") }
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

    /// Scrolls until the whole of `element` is on the screen, clear of the
    /// bars: `isHittable` alone passes a 151 pt AX5 button whose centre is
    /// below the window, and the tap then lands on nothing.
    @discardableResult
    private func reveal(_ element: XCUIElement, attempts: Int = 12) -> Bool {
        func isClear() -> Bool {
            let window = app.windows.firstMatch.frame
            return element.exists && element.isHittable
                && element.frame.minY >= window.minY + 100 && element.frame.maxY <= window.maxY - 40
        }
        for _ in 0..<attempts {
            if isClear() { return true }
            let window = app.windows.firstMatch.frame
            let isAbove = element.exists && element.frame.midY < window.midY
            // Half a window at a time while it is off the screen (at AX5 it
            // is screens away), a fifth once it is close.
            let isFar = element.exists && (element.frame.minY > window.maxY || element.frame.maxY < window.minY)
            let step: (from: CGFloat, to: CGFloat)
            switch (isAbove, isFar) {
            case (true, true): step = (0.3, 0.8)
            case (true, false): step = (0.4, 0.6)
            case (false, true): step = (0.8, 0.3)
            case (false, false): step = (0.7, 0.5)
            }
            let from = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: step.from))
            let to = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: step.to))
            from.press(forDuration: 0.1, thenDragTo: to, withVelocity: .slow, thenHoldForDuration: 0.1)
        }
        return isClear()
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
           "session":{"session_id":"sess-1","started_at":1788895000.0},
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
