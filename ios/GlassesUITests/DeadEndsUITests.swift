//
//  DeadEndsUITests.swift
//  GlassesUITests
//
//  U0.8: the dead ends and contradictions, end to end in the Simulator.
//
//  Runs by default, and never reaches a real Tower: every launch points the
//  app at the closed loopback port or at `MockTowerHTTPServer`, served from
//  this process -- with its route table (H0) and, when a test asks for it,
//  the Tower's socket on `/ws` (H1). The glasses are Mock Device Kit's, paired
//  by the DEBUG launch hook `-UITestMockGlasses` (H2).
//
//  A state the Simulator cannot reach is an `XCTSkip` with the reason, never
//  a pass: the `TowerSmokeUITests` rule.
//

import XCTest

final class DeadEndsUITests: XCTestCase {

    private var app: XCUIApplication!
    private var mock: MockTowerHTTPServer!
    private var mockAuthority = ""
    /// Nothing listens on port 9 of this Mac, so a connection is refused at
    /// once -- the app's own socket fails fast.
    private let closedAuthority = "127.0.0.1:9"

    /// The socket script's state, shared between the test thread and the
    /// mock's queue.
    private let script = SocketScript()

    /// `world_builder`'s status contract, as the declaration offers it.
    static let worldBuilderContract = "world_builder.status/2026-09-10"
    static let sessionStart = "POST /cartridges/world_builder/session/start"
    static let sessionStop = "POST /cartridges/world_builder/session/stop"

    override func setUpWithError() throws {
        continueAfterFailure = false
        mock = try MockTowerHTTPServer(.tower)
        mockAuthority = "127.0.0.1:\(try mock.start())"
    }

    override func tearDownWithError() throws {
        // Leave nothing saved for the next run, even after a failure half-way,
        // as `TowerSettingsUITests` does.
        app?.terminate()
        let cleanup = XCUIApplication()
        cleanup.launchArguments = ["-UITestResetTowerAddress", "-UITestSkipOnboarding"]
        cleanup.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = closedAuthority
        cleanup.launch()
        cleanup.terminate()
        mock?.stop()
    }

    // MARK: Step 1 -- the World Builder session line

    static let activeSentence = "World Builder is active on the Tower."
    static let towerLostSentence = "The Tower disconnected. This phone cannot tell whether World Builder is still "
        + "active there; it asks again when the Tower reconnects."

    /// F01: once the socket drops, the line stops saying "active".
    func testTheSessionLineStopsClaimingActiveWhenTheTowerDrops() throws {
        mock.setRoute(Self.sessionStart, status: 200, body: Self.session(state: "active"))
        mock.setRoute(Self.sessionStop, status: 200, body: Self.session(state: "stopped"))
        scriptWorldBuilder(ackSubscribes: true)
        launch(tower: mockAuthority)
        open(cartridge: "World Builder")

        let footnote = element("wb-session-footnote")
        XCTAssertTrue(waitFor(timeout: 20) { footnote.exists && footnote.label == Self.activeSentence },
                      "the session line said active: \(footnote.exists ? footnote.label : "(none)")")

        mock.refusesSockets = true
        mock.dropSocket()

        XCTAssertTrue(waitFor(timeout: 10) { footnote.label == Self.towerLostSentence },
                      "the session line after the drop: \(footnote.label)")
        XCTAssertFalse(labelled(Self.activeSentence).exists, "something still claims World Builder is active")
    }

    /// F02: a failed request is asked again in place.
    func testAFailedWorldBuilderRequestCanBeAskedAgain() throws {
        mock.setRoute(Self.sessionStart, status: 500, body: #"{"detail":"boom"}"#)
        mock.setRoute(Self.sessionStop, status: 200, body: Self.session(state: "stopped"))
        scriptWorldBuilder(ackSubscribes: true)
        launch(tower: mockAuthority)
        open(cartridge: "World Builder")

        let footnote = element("wb-session-footnote")
        XCTAssertTrue(waitFor(timeout: 20) {
            footnote.exists && footnote.label.hasPrefix("World Builder could not be asked for on the Tower")
        }, "the session line: \(footnote.exists ? footnote.label : "(none)")")

        mock.setRoute(Self.sessionStart, status: 200, body: Self.session(state: "active"))
        let retry = element("wb-session-retry")
        XCTAssertTrue(reveal(retry), "the Ask again button")
        XCTAssertEqual(retry.label, "Ask again")
        retry.tap()

        XCTAssertTrue(waitFor(timeout: 10) { footnote.label == Self.activeSentence },
                      "the session line after Ask again: \(footnote.label)")
        XCTAssertFalse(retry.exists, "Ask again is gone once the Tower said yes")
        let starts = mock.requestLines.filter { $0 == "\(Self.sessionStart) HTTP/1.1" }
        XCTAssertEqual(starts.count, 2, "\(mock.requestLines)")
    }

    // MARK: Step 2 -- offline recovery everywhere

    /// F03: every "Not connected" panel offers Connect and Settings, and
    /// Connect dials.
    func testEveryDisconnectedWorkspaceOffersConnectAndSettings() throws {
        // `/ws` answers 404 (the socket off), so the app's socket fails at once.
        launch(tower: mockAuthority)

        for name in ["World Builder", "Object Memory"] {
            open(cartridge: name)
            let notConnected = app.descendants(matching: .any)
                .matching(NSPredicate(format: "label CONTAINS %@", "Not connected")).firstMatch
            XCTAssertTrue(notConnected.waitForExistence(timeout: 10), "\(name) says Not connected")
            let connect = element("tower-recovery-connect")
            let settings = element("tower-recovery-settings")
            XCTAssertTrue(connect.waitForExistence(timeout: 5), "\(name) offers Connect")
            XCTAssertTrue(reveal(settings), "\(name) offers Tower settings")
            XCTAssertTrue(tap(settings, until: app.navigationBars["Settings"].exists), "\(name): Settings opens")
            XCTAssertTrue(tap(app.buttons["Done"], until: !app.navigationBars["Settings"].exists),
                          "\(name): Settings closes")
        }

        // Only a dial after the reconnect budget is spent (0.5 + 1 + 2 + 4 +
        // 8 s) proves the tap: before it, an automatic retry would pass too.
        var dials = socketDials
        var steadySince = Date()
        let deadline = Date().addingTimeInterval(60)
        while Date() < deadline, Date().timeIntervalSince(steadySince) < 10 {
            Thread.sleep(forTimeInterval: 1)
            let now = socketDials
            if now != dials { dials = now; steadySince = Date() }
        }
        XCTAssertGreaterThanOrEqual(Date().timeIntervalSince(steadySince), 10,
                                    "the app never stopped redialling: \(dials) dials")
        let connect = element("tower-recovery-connect")
        XCTAssertTrue(reveal(connect), "Connect is on the screen")
        XCTAssertTrue(waitFor(timeout: 5) { connect.isEnabled }, "Connect is on once the phone has given up")
        connect.tap()
        XCTAssertTrue(waitFor(timeout: 5) { self.socketDials >= dials + 1 },
                      "Connect dialled the Tower: \(socketDials) dials, \(dials) before the tap")
    }

    /// D1 (manager 137): with the Tower unreachable, Start capture is off,
    /// the reason is beside it, and Connect and Settings are offered.
    func testStartIsDisabledWithAReasonWhileTheTowerIsUnreachable() throws {
        launch(tower: closedAuthority, mockGlasses: true)
        // Home's Start session is on: the glasses are active, so what turns
        // World Builder's Start off below is the Tower alone.
        try requireMockGlasses()
        open(cartridge: "World Builder")

        let start = app.buttons["Start capture"]
        XCTAssertTrue(reveal(start), "Start capture is on the screen")
        XCTAssertFalse(start.isEnabled, "Start capture is off while the Tower is unreachable")

        let reason = element("wb-capture-tower-line")
        XCTAssertTrue(reveal(reason), "the reason is beside Start")
        XCTAssertTrue(reason.label.hasPrefix("Start is off"), reason.label)
        XCTAssertTrue(reason.label.contains("Tower"), reason.label)

        let connect = element("wb-capture-connect")
        let settings = element("wb-capture-settings")
        XCTAssertTrue(connect.exists, "Connect beside Start")
        XCTAssertTrue(reveal(settings), "Tower settings beside Start")
        XCTAssertTrue(tap(settings, until: app.navigationBars["Settings"].exists), "Tower settings opens Settings")
        XCTAssertTrue(tap(app.buttons["Done"], until: !app.navigationBars["Settings"].exists))
        XCTAssertFalse(app.buttons["Stop capture"].exists, "nothing started")
    }

    // MARK: Step 3 -- the viewfinder and the paused capture

    /// Mock glasses and the mock Tower online (D1 leaves no "Start anyway"),
    /// World Builder open, Start capture tapped.
    private func startCaptureWithTheTowerOnline(env: [String: String] = [:]) throws {
        mock.setRoute(Self.sessionStart, status: 200, body: Self.session(state: "active"))
        mock.setRoute(Self.sessionStop, status: 200, body: Self.session(state: "stopped"))
        scriptWorldBuilder(ackSubscribes: true)
        launch(tower: mockAuthority, mockGlasses: true, env: env)
        try requireMockGlasses()
        open(cartridge: "World Builder")
        let start = app.buttons["Start capture"]
        XCTAssertTrue(reveal(start), "Start capture is on the screen")
        XCTAssertTrue(waitFor(timeout: 15) { start.isEnabled }, "Start capture turns on once the Tower is online")
        start.tap()
    }

    private func stopCaptureIfRunning() {
        let stop = app.buttons["Stop capture"]
        if stop.exists, stop.isHittable || reveal(stop) { stop.tap() }
    }

    /// F05: while the camera is on or coming up, the viewfinder never says
    /// "Start a capture session".
    func testTheViewfinderNeverSaysStartWhileTheCameraIsOn() throws {
        try startCaptureWithTheTowerOnline()
        let stop = app.buttons["Stop capture"]
        XCTAssertTrue(stop.waitForExistence(timeout: 15), "the capture started")

        let startWords = app.descendants(matching: .any)
            .matching(NSPredicate(format: "label BEGINSWITH %@", "Start a capture session")).firstMatch
        XCTAssertFalse(startWords.exists, "the viewfinder says Start while the camera is on")
        let starting = labelled("Starting the glasses camera…").exists
        let waiting = labelled("Camera on. Waiting for the first frame from the glasses.").exists
        print("U08|F05|viewfinder: starting=\(starting) waitingForFirstFrame=\(waiting)")
        XCTAssertTrue(starting || waiting, "the viewfinder says the camera is starting or waiting for a frame")
        stopCaptureIfRunning()
    }

    /// F06: a capture the glasses paused offers Stop, not a silent Start.
    func testAPausedCaptureOffersStop() throws {
        try startCaptureWithTheTowerOnline(env: ["GLASSES_UITEST_MOCK_DOFF_AFTER_SECONDS": "2"])
        let pausedLine = element("capture-paused-line")
        guard pausedLine.waitForExistence(timeout: 15) else {
            let shell = app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "Camera")).firstMatch
            let seen = "shell: \(shell.exists ? shell.label : "(none)"); Stop capture "
                + (app.buttons["Stop capture"].exists ? "shown" : "absent")
            stopCaptureIfRunning()
            throw XCTSkip("Mock doff() did not pause the DAT session in this Simulator (\(seen))")
        }
        let stop = app.buttons["Stop capture"]
        XCTAssertTrue(stop.exists, "Stop capture while paused")
        XCTAssertTrue(stop.isEnabled, "Stop capture is on while paused")
        XCTAssertFalse(app.buttons["Start capture"].exists, "no Start while the glasses hold the capture")
        let badge = app.descendants(matching: .any)
            .matching(NSPredicate(format: "label BEGINSWITH %@", "Paused by the glasses")).firstMatch
        XCTAssertTrue(badge.exists, "the viewfinder says the glasses paused it")
        stopCaptureIfRunning()
    }

    // MARK: Step 4 -- glasses readiness words

    /// The shell status bar's words: its label and, when SwiftUI puts the
    /// pills' values there, its value.
    private func shellText() -> String {
        let bar = element("shell-status-bar")
        guard bar.exists else { return "" }
        let value = (bar.value as? String) ?? ""
        return value.isEmpty ? bar.label : "\(bar.label), \(value)"
    }

    /// F07: the Glasses pill says "Active" only while glasses are active on
    /// this phone, and never "Registered" as ready without one.
    func testTheGlassesPillSaysActiveOnlyForAnActiveDevice() throws {
        launch(tower: closedAuthority, mockGlasses: true)
        let bar = element("shell-status-bar")
        XCTAssertTrue(bar.waitForExistence(timeout: 15), "the shell status bar")
        let first = shellText()
        if first.contains("Not registered") || first.contains("Meta AI unavailable") {
            throw XCTSkip("the mock is not a registered device in this Simulator (\(first))")
        }
        try requireMockGlasses()
        XCTAssertTrue(waitFor(timeout: 5) { self.shellText().contains("Active") },
                      "the pill says Active for the active mock: \(shellText())")

        let developer = app.buttons["Developer tools"]
        XCTAssertTrue(tap(developer, until: app.navigationBars["Developer"].exists), "Developer tools opens")
        let disable = app.buttons["Disable Mock Device Kit"]
        XCTAssertTrue(revealInSheet(disable), "Disable Mock Device Kit")
        disable.tap()
        XCTAssertTrue(tap(app.buttons["Done"], until: !app.navigationBars["Developer"].exists), "Developer tools closes")

        let settled = waitFor(timeout: 10) { !self.shellText().contains("Active") }
        let text = shellText()
        XCTAssertTrue(settled, "the pill still says Active with the mock disabled: \(text)")
        XCTAssertFalse(text.contains("Registered"), "a registered word stood for ready: \(text)")
        let honest = ["Not active", "Not registered", "Registering…", "Meta AI unavailable"].contains { text.contains($0) }
        XCTAssertTrue(honest, "the pill's word: \(text)")
    }

    // MARK: Step 5 -- the glasses alert

    /// F08: the glasses alert opens Connections, where its problem is fixed.
    func testTheGlassesAlertOpensConnections() throws {
        launch(tower: closedAuthority, args: ["-UITestGlassesAlert"])
        let alert = app.alerts["Something went wrong"]
        XCTAssertTrue(alert.waitForExistence(timeout: 15), "the glasses alert")
        let open = alert.buttons["Open Connections"]
        XCTAssertTrue(open.exists, "the alert offers Open Connections")
        XCTAssertTrue(alert.buttons["OK"].exists, "and OK")
        open.tap()

        XCTAssertTrue(app.navigationBars["Connections"].waitForExistence(timeout: 10), "Connections opened")
        let camera = app.buttons.matching(NSPredicate(format: "label ENDSWITH %@", "Camera access")).firstMatch
        XCTAssertTrue(revealInSheet(camera), "Connections has the camera-access row")
        XCTAssertFalse(alert.exists, "the alert is gone")
    }

    // MARK: Step 6 -- saved worlds

    static let worldsList = "GET /worlds"

    /// World Builder open on the mock Tower (its socket off: the list is
    /// HTTP only), then its Saved worlds sheet.
    private func openSavedWorlds() {
        let button = app.buttons["Saved worlds"]
        XCTAssertTrue(reveal(button), "the Saved worlds button")
        XCTAssertTrue(tap(button, until: app.navigationBars["Saved worlds"].exists), "the Saved worlds sheet opened")
    }

    private func containing(_ text: String) -> XCUIElement {
        app.descendants(matching: .any).matching(NSPredicate(format: "label CONTAINS %@", text)).firstMatch
    }

    /// F09: a refresh that fails keeps the last list, says it may be out of
    /// date, and offers Try again and Connections.
    func testSavedWorldsStayListedWhenARefreshFails() throws {
        mock.setRoute(Self.worldsList, status: 200, body: Self.worldsListing)
        launch(tower: mockAuthority)
        open(cartridge: "World Builder")
        openSavedWorlds()
        let failed = containing("Failed fixture (Mac B0)")
        XCTAssertTrue(failed.waitForExistence(timeout: 15), "the list arrived")
        XCTAssertTrue(tap(app.buttons["Close"], until: !app.navigationBars["Saved worlds"].exists), "the sheet closed")

        mock.setRoute(Self.worldsList, status: 503, body: "down")
        openSavedWorlds()
        let problem = element("worlds-problem")
        XCTAssertTrue(problem.waitForExistence(timeout: 15), "the refresh's failure is worded")
        XCTAssertTrue(problem.label.contains("It may be out of date"), problem.label)
        XCTAssertTrue(failed.exists, "the last list is still there")
        let retry = element("worlds-retry")
        XCTAssertTrue(retry.exists, "Try again")
        XCTAssertEqual(retry.label, "Try again")
        let connections = element("worlds-connections")
        XCTAssertTrue(connections.exists, "Connections")
        XCTAssertTrue(tap(connections, until: app.navigationBars["Connections"].exists), "Connections opened")
    }

    /// F09: the Tower's "no world root" answer offers Check again, and no
    /// Connections: the Tower was reached.
    func testANoWorldRootAnswerOffersCheckAgainOnly() throws {
        mock.setRoute(Self.worldsList, status: 404, body: #"{"detail":"no world root is configured"}"#)
        launch(tower: mockAuthority)
        open(cartridge: "World Builder")
        openSavedWorlds()
        let problem = element("worlds-problem")
        XCTAssertTrue(problem.waitForExistence(timeout: 15), "the answer is worded")
        XCTAssertTrue(problem.label.hasPrefix("This Tower has no folder for saved worlds"), problem.label)
        let retry = element("worlds-retry")
        XCTAssertTrue(retry.exists, "Check again")
        XCTAssertEqual(retry.label, "Check again")
        XCTAssertFalse(element("worlds-connections").exists, "no Connections for a Tower that answered")
    }

    /// The fixture listing with one more world first: a walk whose geometry
    /// is gone (`interrupted`, no geometry).
    static var worldsListingWithAnInterruptedWalk: String {
        let interrupted = #"""
{"created_at":1788719719.9,"display_name":"Interrupted walk (U0.8 fixture)","live":false,"session_count":1,"sessions":[{"abandoned":false,"appearance":null,"capture_id":null,"dense":null,"end_reason":"stop","ended_at":1788720319.9,"finalization":null,"frame_source":"synthetic","has_geometry":false,"keyframes_accepted":3,"keyframes_journaled":0,"photographic":null,"session_id":"9999aaaa0000bbbb1111cccc2222dddd","started_at":1788719719.9,"state":"interrupted"}],"updated_at":1788720319.9,"world_id":"0808aaaa0000bbbb1111cccc2222dddd"},
"""#
        return worldsListing
            .replacingOccurrences(of: #""world_count":4"#, with: #""world_count":5"#)
            .replacingOccurrences(of: #""worlds":["#, with: #""worlds":["# + interrupted)
    }

    /// F10: a row with nothing to open says what makes another one, and
    /// "Go to capture" goes back to the live screen.
    func testANoGeometryRowSaysWhatToDoAndGoesToCapture() throws {
        mock.setRoute(Self.worldsList, status: 200, body: Self.worldsListingWithAnInterruptedWalk)
        launch(tower: mockAuthority)
        open(cartridge: "World Builder")
        openSavedWorlds()
        XCTAssertTrue(containing("Needs retry").waitForExistence(timeout: 15), "the Needs retry badge")
        XCTAssertTrue(containing("Walking the space again makes a new one").exists, "the caption names the next step")
        let goToCapture = element("worlds-go-to-capture")
        XCTAssertTrue(revealInSheet(goToCapture), "Go to capture")
        XCTAssertTrue(tap(goToCapture, until: !app.navigationBars["Saved worlds"].exists), "the sheet went away")
        let start = app.buttons["Start capture"]
        let noCapture = containing("Capture is not available in this build.")
        XCTAssertTrue(waitFor(timeout: 5) { start.exists || noCapture.exists }, "back on the live screen")
    }

    // MARK: Launch (H3)

    /// `tower` is the socket's `host:port`; `nil` uses the saved address.
    @discardableResult
    private func launch(tower: String?, reset: Bool = true, mockGlasses: Bool = false,
                        args: [String] = [], env: [String: String] = [:]) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = ["-UITestSkipOnboarding"]
        if reset { app.launchArguments.append("-UITestResetTowerAddress") }
        if mockGlasses { app.launchArguments.append("-UITestMockGlasses") }
        app.launchArguments += args
        if let tower { app.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = tower }
        for (name, value) in env { app.launchEnvironment[name] = value }
        // A system alert (location, notifications) would otherwise sit over
        // the app and every wait below would time out on it.
        addUIInterruptionMonitor(withDescription: "system alert") { alert in
            for title in ["Allow", "Don't Allow", "OK", "Not Now"] {
                let button = alert.buttons[title]
                if button.exists { button.tap(); return true }
            }
            return false
        }
        app.launch()
        self.app = app
        return app
    }

    /// Mock Device Kit gave an active device: a Start control is enabled --
    /// Home's "Start session", or World Builder's "Start capture" where that
    /// is the screen. Skipped, not failed, when it never does.
    private func requireMockGlasses(timeout: TimeInterval = 15) throws {
        let starts = app.buttons.matching(NSPredicate(format: "label IN %@", ["Start session", "Start capture"]))
        let active = waitFor(timeout: timeout) {
            starts.allElementsBoundByIndex.contains { $0.exists && $0.isEnabled }
        }
        if !active { throw XCTSkip("Mock Device Kit gave no active device in this Simulator") }
    }

    // MARK: The mock Tower's socket (H1)

    /// The Tower's side of the socket, as `WorldBuilderIntegrationTests`
    /// scripts it: `ping` → `pong`; `cartridges` → a declaration offering
    /// World Builder's status; `result_subscribe` → `result_subscribed`,
    /// echoing the pin, only while `ackSubscribes`. Subscriptions are
    /// numbered per connection, as the Tower's are.
    private func scriptWorldBuilder(ackSubscribes: Bool) {
        script.ackSubscribes = ackSubscribes
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
                        "contract":"\(Self.worldBuilderContract)","available":true,
                        "unavailable_reason":null,"snapshot_only":true}],
                     "not_offered":[]}
                    """)
            case "result_subscribe":
                let count = script.nextSubscription()
                guard script.ackSubscribes else { return }
                let world = (json["world_id"] as? String).map { "\"\($0)\"" } ?? "null"
                let session = (json["session_id"] as? String).map { "\"\($0)\"" } ?? "null"
                mock.sendSocket(text: """
                    {"type":"result_subscribed",
                     "envelope_contract":"cartridge_results.envelope/2026-08-23",
                     "subscription_id":"sub-\(count)","cartridge":"world_builder",
                     "result_type":"status","contract":"\(Self.worldBuilderContract)",
                     "snapshot_only":true,"world_id":\(world),"session_id":\(session),
                     "cursor_status":"absent"}
                    """)
            default:
                break
            }
        }
    }

    /// One World Builder snapshot for the subscription the script last
    /// acknowledged; one for any other id is dropped by the app as retired.
    /// The world has geometry when `elementCount > 0 || poseCount > 0`.
    private func sendSnapshot(modelState: String, elementCount: Int, poseCount: Int, seq: Int = 1) {
        mock.sendSocket(text: """
            {"type":"cartridge_result",
             "envelope_contract":"cartridge_results.envelope/2026-08-23",
             "subscription_id":"sub-\(script.subscribeCount)","cartridge":"world_builder","result_type":"status",
             "contract":"\(Self.worldBuilderContract)","seq":\(seq),"revision":"r\(seq)",
             "revision_changed":true,"coalesced":0,"cursor_status":null,
             "snapshot":true,"tower_sent_at":1787463092.9,"time_basis":"tower-receipt",
             "payload":{"model_state":"\(modelState)","model_state_reason":null,
               "world_snapshot":{"name":"Probe Room","world_id":"w1",
                 "keyframe_count":\(poseCount),"revision":"r\(seq)",
                 "tracking":"good","scale":"relative","mapping_seconds":12.5,
                 "calibration":"calibrated",
                 "geometry":{"representation":"sparse point cloud","element_count":\(elementCount),
                             "is_incremental":false},
                 "trajectory":{"pose_count":\(poseCount),"path_length":2.85,
                               "path_length_unit":"world units","scale":"relative"},
                 "persistence":{"state":"saved","revision":"p1"}}}}
            """)
    }

    // MARK: Fixtures (H0), verbatim

    /// A `cartridge_session.control/2026-08-27` snapshot for `world_builder`
    /// (`WorldBuilderIntegrationTests.session(state:)`).
    static func session(state: String, accepted: Bool = true, changed: Bool = true) -> String {
        let sessionID = state == "stopped" ? "null" : #""sess-1""#
        return """
            {"contract":"cartridge_session.control/2026-08-27","cartridge":"world_builder",\
            "worker":"world-build-session","supported":true,"state":"\(state)",\
            "state_means":"intent-not-liveness","states":["stopped","active","paused"],\
            "actions":["start","pause","resume","stop"],"session_id":\(sessionID),\
            "started_at":1788895000.0,"changed_at":1788895000.0,"following":[],\
            "following_this_session":[],"captures":[],"accepted":\(accepted),"changed":\(changed),\
            "attached_capture_id":null,"stop_policy":"request"}
            """
    }

    /// `GET /worlds` as the Tower served its four fixture worlds
    /// (`WorldPhotographicTests.swift`, `WorldPhotographicTowerFixtureTests.listingJSON`).
    static let worldsListing = #"""
{"contract":"world_builder.worlds/2026-09-10","world_count":4,"worlds":[{"created_at":1790048283.193669,"display_name":"Owed fixture (Mac B0)","live":false,"session_count":1,"sessions":[{"abandoned":false,"appearance":null,"capture_id":"appearance-fixture","dense":null,"end_reason":"stop","ended_at":1790048583.193669,"finalization":{"detail":null,"final_solve":"solved","started_at":1790048583.193669,"state":"complete","updated_at":1790048683.193669},"frame_source":"synthetic-fixture","has_geometry":true,"keyframes_accepted":24,"keyframes_journaled":0,"photographic":{"detail":"the appearance stage is unfinished and no process is working on it","stage":"appearance","state":"owed"},"session_id":"s1","started_at":1790048283.193669,"state":"finalizing"}],"updated_at":1790160840.5748532,"world_id":"w2"},{"created_at":1790048283.193669,"display_name":"Failed fixture (Mac B0)","live":false,"session_count":1,"sessions":[{"abandoned":false,"appearance":null,"capture_id":"appearance-fixture","dense":null,"end_reason":"stop","ended_at":1790048583.193669,"finalization":{"detail":null,"final_solve":"solved","started_at":1790048583.193669,"state":"complete","updated_at":1790048683.193669},"frame_source":"synthetic-fixture","has_geometry":true,"keyframes_accepted":24,"keyframes_journaled":0,"photographic":{"detail":"fixture: the appearance encode raised","stage":"appearance","state":"failed"},"session_id":"s1","started_at":1790048283.193669,"state":"complete"}],"updated_at":1790048883.193669,"world_id":"w3"},{"created_at":1790048283.193669,"display_name":"Appearance fixture (Mac)","live":false,"session_count":1,"sessions":[{"abandoned":false,"appearance":{"bytes":239242,"format":"wb-appearance-keyframes/1","imagery":"first-person keyframe imagery of a private space; best-effort face redaction with measured false negatives; not anonymised; screens, documents and bodies are not redacted","imagery_source":"redacted","keyframe_image_set":null,"keyframes":24,"keyframes_phone":24,"label_trusted":true,"privacy_safe":true,"privacy_tags":["raw-imagery","first-person"],"quality":"final","redaction":"faces-detected-and-filled/yunet-2023mar@0.30+plausibility3","redaction_effective":"faces-detected-and-filled/yunet-2023mar@0.30+plausibility3","retains_raw_imagery":true,"retention":"kept with the world under appearance/<session>/ until the session is rebuilt or the world is purged; derived from the session keyframes, so it is deleted and rebuilt, never edited","state":"served"},"capture_id":"appearance-fixture","dense":null,"end_reason":"stop","ended_at":1790048583.193669,"finalization":{"detail":null,"final_solve":"solved","started_at":1790048583.193669,"state":"complete","updated_at":1790048683.193669},"frame_source":"synthetic-fixture","has_geometry":true,"keyframes_accepted":24,"keyframes_journaled":0,"photographic":{"detail":"no Tower ever ran a photographic stage for this session, which is not the same as one that tried and failed","stage":null,"state":"never_recorded"},"session_id":"s1","started_at":1790048283.193669,"state":"complete"}],"updated_at":1790048883.193669,"world_id":"w1"},{"created_at":1788719719.9311092,"display_name":"Synthetic room (Mac fixture)","live":false,"session_count":2,"sessions":[{"abandoned":false,"appearance":null,"capture_id":null,"dense":null,"end_reason":"stop","ended_at":1788720319.9311092,"finalization":{"at":1788720319.9311092,"detail":null,"final_solve":"skipped","state":"complete"},"frame_source":"synthetic","has_geometry":true,"keyframes_accepted":30,"keyframes_journaled":30,"photographic":{"detail":"this session has no finished global solve to build a photographic room from","stage":null,"state":"unattempted"},"session_id":"1111aaaa2222bbbb3333cccc4444dddd","started_at":1788719719.9311092,"state":"complete"},{"abandoned":false,"appearance":null,"capture_id":null,"dense":null,"end_reason":"stop","ended_at":1788721419.9311092,"finalization":null,"frame_source":"synthetic","has_geometry":false,"keyframes_accepted":3,"keyframes_journaled":0,"photographic":{"detail":"this session has no finished global solve to build a photographic room from","stage":null,"state":"unattempted"},"session_id":"5555eeee6666ffff7777000088881111","started_at":1788721319.9311092,"state":"unbuilt"}],"updated_at":1788723259.9311092,"world_id":"a1b2c3d4e5f60718293a4b5c6d7e8f90"}]}
"""#

    // MARK: Helpers (copied from TowerSettingsUITests and TowerSmokeUITests)

    private func element(_ identifier: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    /// Any element whose label is exactly `label`.
    private func labelled(_ label: String) -> XCUIElement {
        app.descendants(matching: .any).matching(NSPredicate(format: "label == %@", label)).firstMatch
    }

    /// Scroll until `element` can be tapped and is clear of the bars, moving
    /// towards it. A lazy stack may not have built an element that is off
    /// screen, so `builtAbove` says which way to look for one that does not
    /// exist yet.
    @discardableResult
    private func reveal(_ element: XCUIElement, builtAbove: Bool = false, attempts: Int = 12) -> Bool {
        for _ in 0..<attempts {
            let clearTop = topInset()
            if element.exists && element.isHittable && element.frame.minY >= clearTop - 1 { return true }
            let window = app.windows.firstMatch.frame
            let isAbove = element.exists
                ? element.frame.minY < clearTop || element.frame.midY < window.midY
                : builtAbove
            let from = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: isAbove ? 0.4 : 0.7))
            let to = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: isAbove ? 0.6 : 0.5))
            from.press(forDuration: 0.1, thenDragTo: to, withVelocity: .slow, thenHoldForDuration: 0.1)
        }
        return element.exists && element.isHittable && element.frame.minY >= topInset() - 1
    }

    /// The bottom of the navigation bar in front, or of the pinned status
    /// bar under it when there is one.
    private func topInset() -> CGFloat {
        let bar = app.navigationBars.firstMatch
        var top = bar.exists ? bar.frame.maxY : 0
        let status = element("shell-status-bar")
        if status.exists, status.frame.minY < top + 4 { top = max(top, status.frame.maxY) }
        return top
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

    private func open(cartridge name: String) {
        let cartridges = app.buttons["Cartridges"]
        XCTAssertTrue(cartridges.waitForExistence(timeout: 10), "the shell's Cartridges button")
        let drawerDone = app.buttons["Done"]
        XCTAssertTrue(tap(cartridges, until: drawerDone.exists), "the cartridge drawer opened")
        let row = app.buttons.containing(NSPredicate(format: "label BEGINSWITH %@", name)).firstMatch
        XCTAssertTrue(revealInSheet(row), "a drawer row for \(name)")
        // The workspace has arrived when the drawer is gone and the shell's
        // title names it -- not when a static text with that name exists,
        // which the drawer row itself satisfies.
        XCTAssertTrue(tap(row, until: !drawerDone.exists && app.navigationBars[name].exists),
                      "the \(name) workspace opened")
    }

    /// A row of a lazy list in a sheet is not built until it is scrolled to.
    /// Settle first -- a sheet still sliding in reads as not hittable -- and
    /// then only swipe up: a drag down at the top of a sheet pulls it away.
    @discardableResult
    private func revealInSheet(_ element: XCUIElement) -> Bool {
        var found = waitFor(timeout: 3) { element.exists && element.isHittable }
        for _ in 0..<6 where !found {
            app.swipeUp(velocity: .slow)
            found = waitFor(timeout: 1) { element.exists && element.isHittable }
        }
        return found
    }

    /// How many times the app has dialled the socket.
    private var socketDials: Int {
        mock.requestLines.filter { $0 == "GET /ws HTTP/1.1" }.count
    }
}

/// The socket script's state: whether to acknowledge a subscribe, and how
/// many this connection has made.
final class SocketScript: @unchecked Sendable {
    private let lock = NSLock()
    private var ack = true
    private var count = 0

    var ackSubscribes: Bool {
        get { lock.withLock { ack } }
        set { lock.withLock { ack = newValue } }
    }

    /// The id the script last handed out is `sub-\(subscribeCount)`.
    var subscribeCount: Int { lock.withLock { count } }

    func resetSubscriptions() { lock.withLock { count = 0 } }

    func nextSubscription() -> Int {
        lock.withLock {
            count += 1
            return count
        }
    }
}
