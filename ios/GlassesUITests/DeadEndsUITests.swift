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
        // The drawer is a lazy list in a half-height sheet: a row below the
        // fold is not built until it is scrolled to. Settle first -- a sheet
        // still sliding in reads as not hittable -- and then only swipe up:
        // a drag down at the top of a sheet pulls the sheet away.
        var found = waitFor(timeout: 3) { row.exists && row.isHittable }
        for _ in 0..<6 where !found {
            app.swipeUp(velocity: .slow)
            found = waitFor(timeout: 1) { row.exists && row.isHittable }
        }
        XCTAssertTrue(found, "a drawer row for \(name)")
        // The workspace has arrived when the drawer is gone and the shell's
        // title names it -- not when a static text with that name exists,
        // which the drawer row itself satisfies.
        XCTAssertTrue(tap(row, until: !drawerDone.exists && app.navigationBars[name].exists),
                      "the \(name) workspace opened")
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
