//
//  WorldInlinePanelUITests.swift
//  GlassesUITests
//
//  U-INLINE (managers 209, 210) and U0.6 on the Simulator: the World Builder
//  panel's phases, its one web view, expand and collapse, the gesture gate,
//  the raw marker, the honest endings, Saved worlds, the layout and the
//  VoiceOver order (UI1-UI10), U0.6's stage line, preview label and banner
//  (U1-U4), and R1 on a Mac scratch Tower.
//
//  The mock tests never reach a real Tower: `MockTowerHTTPServer` serves the
//  render routes (the U1.1 bridge test page, counting its own loads) and
//  speaks the World Builder socket (`PanelSocket`). R1 is a REQUIRED gate:
//  it fails, never skips, without GLASSES_UITEST_TOWER_AUTHORITY naming a
//  real local Tower with TOWER_WORLD_NATIVE_CHROME=1 and the fixture world.
//  UI8 is run on each phone U-INLINE §1.2 names (iPhone 17 Pro, 17e, SE 3)
//  and fails on any other iPhone; on an iPad it skips.
//
//  Screenshots go to UINLINE_SHOTS_DIR when it is set, suffixed with
//  UINLINE_SHOT_SUFFIX.
//

import UIKit
import XCTest

final class WorldInlinePanelUITests: XCTestCase {

    private var app: XCUIApplication!
    private var mock: MockTowerHTTPServer!
    private var socket: PanelSocket!
    private var mockAuthority = ""
    private static let ax5 = "UICTContentSizeCategoryAccessibilityXXXL"
    private typealias Page = WorldChromeTestPage
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
    }

    override func tearDownWithError() throws {
        app?.terminate()
        mock?.stop()
    }

    // MARK: UI1: the panel never moves Stop

    func testThePanelNeverMovesStop() throws {
        for size in [nil, Self.ax5] {
            let name = size == nil ? "default" : "ax5"
            let held = try startCapture(size: size, extra: ["-UITestHideCaptureHealth"])
            let without = held.stop.frame.minY - held.top.frame.minY
            app.terminate()
            for map in [false, true] {
                let run = try startCapture(size: size, extra: map ? ["-WBPanelMapFixture"] : [])
                // Without a map the walking panel is Capture health alone.
                let panel = element(map ? "wb-panel" : "capture-health")
                XCTAssertTrue(panel.waitForExistence(timeout: 10), "\(name) map=\(map): the walking panel")
                if map { XCTAssertTrue(element("wb-panel-map").waitForExistence(timeout: 10), "the map slot") }
                Thread.sleep(forTimeInterval: 1)
                let with = run.stop.frame.minY - run.top.frame.minY
                print("UINLINE-STOP|\(name)|map=\(map)|without=\(Int(without))|with=\(Int(with))"
                      + "|panel=\(Int(panel.frame.minY))|stop-max=\(Int(run.stop.frame.maxY))")
                XCTAssertEqual(with, without, accuracy: 0.5, "\(name) map=\(map): the panel moved Stop")
                XCTAssertGreaterThanOrEqual(panel.frame.minY, run.stop.frame.maxY - 0.5,
                                            "\(name) map=\(map): the panel is below Stop")
                shoot("ui1-walking-\(name)\(map ? "-map" : "")")
                app.terminate()
            }
        }
    }

    /// FOW review MED-1: the walking caption runs to about eight lines, so
    /// the panel's height follows the receipt -- below Stop, never moving
    /// it. Stop's frame is identical with no coverage block and with the
    /// longest one: two pieces, the not-drawn note, newer keyframes, stale,
    /// and the legend.
    func testTheCoverageCaptionNeverMovesStop() throws {
        for size in [nil, Self.ax5] {
            let name = size == nil ? "default" : "ax5"
            let run = try startCapture(size: size, extra: [])
            XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "\(name): the live subscription")
            // A live-capture session, so the reports are this capture's walk
            // and not a foreign one.
            push(modelState: "receiving", liveCapture: true, accepted: 127)
            XCTAssertTrue(element("capture-health").waitForExistence(timeout: 10), "\(name): the walking panel")
            Thread.sleep(forTimeInterval: 1)
            XCTAssertFalse(element("wb-panel-map").exists, "\(name): no block, no map")
            let without = run.stop.frame
            push(modelState: "receiving", liveCapture: true, guidance: FOWGuidance.maximal, accepted: 127,
                 towerSentAt: FOWGuidance.midSolvedAt + 45)
            let notDrawn = element("wb-panel-map-not-drawn")
            if !notDrawn.waitForExistence(timeout: 15) { print("FOW-AX|\(app.debugDescription)") }
            XCTAssertTrue(notDrawn.exists, "\(name): the not-drawn note")
            XCTAssertEqual(text(notDrawn), "1 more piece and 5 more stations not drawn")
            XCTAssertTrue(element("wb-panel-map-legend").exists, "\(name): the legend")
            XCTAssertTrue(element("wb-panel-map-pieces").exists, "\(name): the separate-pieces line")
            XCTAssertTrue(element("wb-panel-map-newer").exists, "\(name): the newer line")
            XCTAssertTrue(text(element("wb-panel-map-label")).hasSuffix(" · stale"), "\(name): stale")
            XCTAssertEqual(app.descendants(matching: .any).matching(identifier: "wb-panel-map-piece").count, 2)
            Thread.sleep(forTimeInterval: 1)
            let with = run.stop.frame
            print("FOW-STOP|\(name)|without=\(without)|with=\(with)|panel=\(element("wb-panel").frame)")
            XCTAssertEqual(with, without, "\(name): the coverage caption moved Stop")
            shoot("fow-stop-\(name)")
            app.terminate()
        }
    }

    // MARK: UI2 + U1: finishing, then ready, in place

    func testFinishingThenReadyCrossFadesInPlace() throws {
        mock.setRoute("GET /worlds/w1/render", status: 200, body: Page.html(steps: [.init(state: Page.state())],
                                                                             countsLoads: true))
        mock.setRoute("GET /worlds/w1/render/revision", status: 200, body: Page.revision())
        openLive()
        push(modelState: "receiving")
        push(modelState: "finalizing", buildInProgress: true, finalizationState: "pending", finalSolve: "pending",
             processing: #"{"stage":"placing"}"#)
        let line = element("wb-finish-line")
        XCTAssertTrue(line.waitForExistence(timeout: 15), "the finish block")
        XCTAssertTrue(waitFor { line.label.contains("Now: placing images.") }, line.label)
        XCTAssertTrue(line.label.contains("since you stopped"), "the elapsed line: \(text(line))")
        // U1: the canvas card reads Open the preview, and its note is a preview.
        let open = element("wb-open-3d")
        XCTAssertTrue(open.exists)
        XCTAssertEqual(open.label, "Open the preview")
        XCTAssertTrue(beginning("Preview — your final world is still building.").exists, "the preview note")
        XCTAssertEqual(element("wb-panel-expand").label, "Open the preview")

        push(modelState: "finalizing", buildInProgress: true, finalizationState: "pending", finalSolve: "pending",
             processing: #"{"stage":"checking","step":{"n":2,"of":3}}"#)
        XCTAssertTrue(waitFor { line.label.contains("checking the placement, pass 2 of 3") }, line.label)
        XCTAssertTrue(app.staticTexts["Checking the placement · pass 2 of 3"].exists || line.exists)
        let stage = element("wb-panel-stage")
        XCTAssertTrue(reveal(stage))
        XCTAssertTrue(stage.frame.insetBy(dx: -1, dy: -1).contains(line.frame), "inside the stage")
        XCTAssertEqual(webViewCount(), 0, "no web view while finishing")
        shoot("ui2-finishing")

        push(modelState: "finalized", finalizationState: "complete", finalSolve: "solved",
             photographic: #"{"state":"complete","stage":"appearance"}"#)
        let web = app.webViews.firstMatch
        XCTAssertTrue(web.waitForExistence(timeout: 30), "the world arrives")
        XCTAssertTrue(waitForTheWorldShown(), "the cross-fade completed")
        XCTAssertTrue(waitFor(timeout: 20) { self.element("world-chrome-status").exists && self.element("world-chrome-status").label == "loads 1" },
                      "the page drew: \(text(element("world-chrome-status")))")
        XCTAssertEqual(webViewCount(), 1)
        XCTAssertTrue(stage.frame.insetBy(dx: -1, dy: -1).contains(web.frame), "the world is in the stage")
        XCTAssertFalse(app.buttons["Close"].exists, "no sheet")
        XCTAssertEqual(element("wb-panel-headline").label, "Saved")
        XCTAssertFalse(element("wb-finish-line").exists, "U1: no finish block once settled")
        XCTAssertEqual(element("wb-open-3d").label, "Open the 3D world")
        shoot("ui2-ready")
    }

    // MARK: Review 3, HIGH: the cover keeps its own walk's words

    /// The cover is open on A's preview when a report for B arrives: the
    /// picture stays A's, so its title stays and no stage is claimed over
    /// it -- B's least of all -- and a notice says a new walk is in
    /// progress. Closed, the panel follows B.
    func testTheCoverKeepsItsWalksWordsWhenAReportForAnotherWalkArrives() throws {
        mock.setRoute("GET /worlds/w1/render", status: 200, body: Page.html(steps: [.init(state: Page.state())]))
        mock.setRoute("GET /worlds/w1/render/revision", status: 200, body: Page.revision())
        openLive()
        push(modelState: "receiving")
        push(modelState: "finalizing", buildInProgress: true, finalizationState: "pending", finalSolve: "pending",
             processing: #"{"stage":"placing"}"#)
        let expand = element("wb-panel-expand")
        XCTAssertTrue(expand.waitForExistence(timeout: 15), "Open the preview")
        XCTAssertTrue(reveal(expand))
        expand.tap()
        let line = element("world-render-finish-line")
        XCTAssertTrue(waitFor(timeout: 20) { line.exists && line.label.contains("placing images") },
                      "A's stage in the cover: \(text(line))")
        XCTAssertTrue(app.navigationBars["Probe Room"].exists, "A's title")

        push(modelState: "receiving", world: "w2", session: "s9", name: "Other Room")
        let notice = element("world-render-notice")
        XCTAssertTrue(waitFor(timeout: 10) { notice.exists && notice.label.contains("A new walk is in progress.") },
                      "the notice: \(text(notice))")
        XCTAssertTrue(app.navigationBars["Probe Room"].exists, "A's title stays")
        XCTAssertFalse(app.navigationBars["Other Room"].exists, "B's title over A's picture")
        XCTAssertFalse(line.exists && line.label.contains("Now:"), "a stage over A's picture: \(text(line))")
        XCTAssertTrue(app.webViews.firstMatch.exists, "A's picture stays")
        shoot("r3-cover-other-walk")

        closeTheCover()
        XCTAssertTrue(waitFor { !self.element("wb-panel-expand").exists && !self.element("wb-finish-line").exists },
                      "the panel's own rules again: B is walking, with no picture")
    }

    // MARK: UI3: expand and collapse never reload

    func testExpandAndCollapseDoNotReload() throws {
        openReadyLiveWorld()
        let status = element("world-chrome-status")
        XCTAssertTrue(waitFor(timeout: 20) { status.exists && status.label == "loads 1" }, status.label)
        let renders = mock.requestLines.filter { $0.hasPrefix("GET /worlds/w1/render?") }.count
        let expand = element("wb-panel-expand")
        XCTAssertTrue(reveal(expand))
        expand.tap()
        let window = app.windows.firstMatch.frame
        let web = app.webViews.firstMatch
        XCTAssertTrue(waitFor { web.frame.height > window.height * 0.6 }, "full screen: \(web.frame)")
        XCTAssertEqual(webViewCount(), 1)
        XCTAssertTrue(waitFor { status.exists && status.label == "loads 1" }, "no reload: \(text(status))")
        shoot("ui3-expanded")
        closeTheCover()
        let stage = element("wb-panel-stage")
        XCTAssertTrue(waitFor { stage.exists && stage.frame.insetBy(dx: -1, dy: -1).contains(web.frame) },
                      "back in the stage")
        XCTAssertEqual(webViewCount(), 1)
        XCTAssertTrue(waitFor { status.exists && status.label == "loads 1" }, "still no reload: \(text(status))")
        XCTAssertEqual(mock.requestLines.filter { $0.hasPrefix("GET /worlds/w1/render?") }.count, renders,
                       "the page was not fetched again")
        for round in 1...2 {
            XCTAssertTrue(reveal(expand))
            expand.tap()
            XCTAssertTrue(waitFor { web.frame.height > window.height * 0.6 }, "round \(round)")
            closeTheCover()
            XCTAssertTrue(waitFor { stage.frame.insetBy(dx: -1, dy: -1).contains(web.frame) }, "round \(round)")
        }
        XCTAssertTrue(waitFor { status.exists && status.label == "loads 1" }, "three rounds, one load: \(text(status))")
        // The cover is not a cartridge switch: the World Builder session on
        // the Tower was never stopped (nor started again) by expanding.
        let sessionCalls = mock.requestLines.filter { $0.contains("/cartridges/world_builder/session/") }
        XCTAssertFalse(sessionCalls.contains { $0.contains("/session/stop") },
                       "expanding the world stopped the World Builder session: \(sessionCalls)")
        XCTAssertEqual(sessionCalls.filter { $0.contains("/session/start") }.count, 1,
                       "one start, at the screen's appearance: \(sessionCalls)")
    }

    // MARK: UI4: off screen and in the background, torn down

    func testOffScreenAndBackgroundTearDown() throws {
        openReadyLiveWorld()
        let status = element("world-chrome-status")
        XCTAssertTrue(waitFor(timeout: 20) { status.exists && status.label == "loads 1" })
        // Scrolled away for longer than 5 s.
        let top = app.buttons["Saved worlds"]
        for _ in 0..<8 where !(top.exists && top.isHittable) { app.swipeDown(velocity: .fast) }
        let panel = element("wb-panel")
        let window = app.windows.firstMatch.frame
        if UIDevice.current.userInterfaceIdiom == .pad,
           panel.frame.minY <= window.maxY - 0.2 * panel.frame.height, panel.isHittable {
            throw XCTSkip("geometry: an iPad window is tall enough that the panel never scrolls off it (\(panel.frame))")
        }
        XCTAssertTrue(panel.frame.minY > window.maxY - 0.2 * panel.frame.height || !panel.isHittable,
                      "the panel is off the screen: \(panel.frame)")
        Thread.sleep(forTimeInterval: 6.5)
        XCTAssertEqual(webViewCount(), 0, "torn down after 5 s off screen")
        XCTAssertTrue(reveal(element("wb-panel-stage")))
        XCTAssertTrue(app.webViews.firstMatch.waitForExistence(timeout: 20), "back: a new web view")
        XCTAssertTrue(waitFor(timeout: 20) { status.exists && status.label == "loads 1" }, "a fresh page: \(text(status))")
        XCTAssertEqual(webViewCount(), 1)

        XCUIDevice.shared.press(.home)
        Thread.sleep(forTimeInterval: 3)
        app.activate()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 10))
        XCTAssertTrue(app.webViews.firstMatch.waitForExistence(timeout: 20), "reloaded after the background")
        XCTAssertTrue(waitFor(timeout: 20) { status.exists && status.label == "loads 1" }, "a new web view: \(text(status))")
        let renders = mock.requestLines.filter { $0.hasPrefix("GET /worlds/w1/render?") }.count
        XCTAssertGreaterThanOrEqual(renders, 3, "fetched again each time: \(renders)")
    }

    // MARK: UI5: the raw marker is always on the inline world

    func testTheRawMarkerIsAlwaysOnTheInlineWorld() throws {
        let marker = "RESEARCH T: UNREDACTED"
        var state = Page.state(raw: true, marker: marker)
        state["status"] = NSNull()
        // (a) native, the page's own marker.
        mock.setRoute("GET /worlds/w1/render", status: 200,
                      body: Page.html(steps: [.init(state: state)], raw: true, marker: marker))
        mock.setRoute("GET /worlds/w1/render/revision", status: 200, body: Page.revision())
        openReadyLiveWorld(configure: false)
        let band = element("world-chrome-research")
        XCTAssertTrue(band.waitForExistence(timeout: 30), "(a) the marker over the inline world")
        let stage = element("wb-panel-stage")
        XCTAssertLessThanOrEqual(band.frame.maxY, stage.frame.minY + 1, "(a) above the stage")
        shoot("ui5-native-marker")
        // (c) expand and collapse: still there.
        let expand = element("wb-panel-expand")
        XCTAssertTrue(reveal(expand))
        expand.tap()
        XCTAssertTrue(element("wb-cover-close").waitForExistence(timeout: 10))
        closeTheCover()
        XCTAssertTrue(waitFor { stage.exists && band.exists }, "(c) after expand and collapse")
        XCTAssertLessThanOrEqual(band.frame.maxY, stage.frame.minY + 1)
        app.terminate()

        // (b) no echo; the header alone raises it.
        let warning = "Research T: unredacted imagery, from the header"
        mock.setRoute("GET /worlds/w1/appearance/s1/manifest", status: 200, body: "{}",
                      headers: ["X-World-Imagery": "raw-local-research", "X-World-Imagery-Warning": warning])
        mock.setRoute("GET /worlds/w1/render", status: 200,
                      body: Page.html(steps: [.init(state: Page.state())], echo: false, fetchManifest: true))
        openReadyLiveWorld(configure: false)
        XCTAssertTrue(band.waitForExistence(timeout: 30), "(b) the header's marker, no echo")
        XCTAssertTrue(band.label.contains(warning.uppercased()), band.label)
        XCTAssertLessThanOrEqual(band.frame.maxY, stage.frame.minY + 1, "(b) above the stage")
        shoot("ui5-header-marker")
    }

    // MARK: UI6: honest endings

    func testHonestEndings() throws {
        mock.setRoute("GET /worlds/w1/render", status: 200, body: Page.html(steps: [.init(state: Page.state())]))
        mock.setRoute("GET /worlds/w1/render/revision", status: 200, body: Page.revision())
        openLive()
        push(modelState: "interrupted", reason: "The builder stopped.")
        let heading = element("wb-panel-headline")
        XCTAssertTrue(waitFor(timeout: 15) { heading.exists && heading.label == "Interrupted" }, heading.label)
        XCTAssertTrue(app.webViews.firstMatch.waitForExistence(timeout: 30), "the world is shown")
        let panel = element("wb-panel")
        let saved = panel.descendants(matching: .any).matching(NSPredicate(format: "label CONTAINS %@", "Saved"))
        XCTAssertEqual(saved.count, 0, "no Saved anywhere in the panel")
        shoot("ui6-interrupted")

        push(modelState: "finalized", elements: 0, poses: 0, finalizationState: "complete", finalSolve: "solved")
        XCTAssertTrue(waitFor(timeout: 15) { heading.label == "Needs retry" }, heading.label)
        XCTAssertTrue(element("wb-panel-failed").waitForExistence(timeout: 10), "the failed text")
        XCTAssertTrue(waitFor { self.webViewCount() == 0 }, "no web view")
        shoot("ui6-needs-retry")

        // The Tower drops before any page is loaded.
        mock.refusesSockets = true
        mock.dropSocket()
        XCTAssertTrue(element("wb-panel-connect").waitForExistence(timeout: 20), "offline, with the way to reconnect")
        // The capture control says it already: no third line (the lead).
        XCTAssertTrue(element("wb-capture-tower-line").exists, "the screen's own offline line")
        XCTAssertFalse(element("wb-panel-offline").exists, "a third \"not connected\" line")
        XCTAssertEqual(webViewCount(), 0)
        shoot("ui6-offline")
    }

    // MARK: UI7: a saved world opens as a preview card

    func testASavedWorldOpensAsAPreviewCard() throws {
        mock.setRoute("GET /worlds", status: 200, body: Page.listing(withArea: true))
        mock.setRoute("GET /worlds/w1/render", status: 200, body: Page.html(steps: [.init(state: Page.state())]))
        mock.setRoute("GET /worlds/w1/render/revision", status: 200, body: Page.revision(withArea: true))
        mock.setRoute("GET /worlds/w1/areas/s1/\(Page.areaID)/render", status: 200,
                      body: Page.html(steps: [.init(state: Page.state())], kind: "area"))
        mock.setRoute("GET /worlds/w1/areas/s1/\(Page.areaID)/render/revision", status: 200, body: Page.areaRevision)
        launch()
        open(cartridge: "World Builder")
        openSavedWorlds()
        let row = app.staticTexts["Appearance fixture (Mac)"].firstMatch
        XCTAssertTrue(row.waitForExistence(timeout: 20))
        XCTAssertTrue(tap(row, until: !app.navigationBars["Saved worlds"].exists), "the row pinned it")
        let panel = element("wb-panel")
        let expand = element("wb-panel-expand")
        XCTAssertTrue(expand.waitForExistence(timeout: 30), "the pinned world in the panel")
        XCTAssertTrue(waitFor { panel.isHittable }, "scrolled into view")
        XCTAssertTrue(app.webViews.firstMatch.waitForExistence(timeout: 30))
        XCTAssertEqual(webViewCount(), 1)
        shoot("ui7-pinned")
        XCTAssertTrue(reveal(expand))
        expand.tap()
        XCTAssertTrue(element("wb-cover-close").waitForExistence(timeout: 10), "Full screen opens the cover")
        XCTAssertEqual(webViewCount(), 1)
        closeTheCover()

        // An area row still pushes; the panel's web view is gone meanwhile.
        openSavedWorlds()
        let area = app.buttons.containing(NSPredicate(format: "label BEGINSWITH %@", "Area 1")).firstMatch
        XCTAssertTrue(area.waitForExistence(timeout: 20), "the area row")
        XCTAssertTrue(tap(area, until: self.app.webViews.firstMatch.exists), "the area pushed")
        Thread.sleep(forTimeInterval: 2)
        XCTAssertEqual(webViewCount(), 1, "one web view")
        XCTAssertTrue(app.navigationBars["Saved worlds"].exists || app.buttons["Saved worlds"].exists,
                      "pushed within the sheet")
        shoot("ui7-area-pushed")
    }

    // MARK: UI8: the layout on each phone

    /// U-INLINE §1.2's phones (window points → the card width W).
    private static let specPhones: [String: (name: String, width: CGFloat)] = [
        "402x874": ("iPhone 17 Pro", 370),
        "390x844": ("iPhone 17e", 358),
        "375x667": ("iPhone SE 3", 343),
    ]

    func testTheLayoutOnEachPhone() throws {
        // UI8 is the spec's phones' geometry; an iPad window is none of them
        // (the iPad device run skips it, the iPhone 17 Pro Simulator runs it).
        if UIDevice.current.userInterfaceIdiom == .pad {
            throw XCTSkip("UI8 proves the spec's phones (17 Pro, 17e, SE 3), not an iPad")
        }
        for size in [nil, Self.ax5] {
            let name = size == nil ? "default" : "ax5"
            mock.setRoute("GET /worlds/w1/render", status: 200,
                          body: Page.html(steps: [.init(state: Page.state(status: "loads 1"))],
                                          labels: Page.realLabels(kind: "room")))
            mock.setRoute("GET /worlds/w1/render/revision", status: 200, body: Page.revision())
            openReadyLiveWorld(size: size, configure: false)
            let panel = element("wb-panel"), expand = element("wb-panel-expand"), stage = element("wb-panel-stage")
            XCTAssertTrue(reveal(panel, whole: true), "\(name): the whole panel")
            Thread.sleep(forTimeInterval: 1)
            let window = app.windows.firstMatch.frame
            let key = "\(Int(window.width))x\(Int(window.height))"
            guard let phone = Self.specPhones[key] else {
                XCTFail("UI8 proves the spec's phones (17 Pro, 17e, SE 3); a \(key) window is none of them")
                return
            }
            print("UINLINE-LAYOUT|\(phone.name)|\(name)|window=\(Int(window.width))x\(Int(window.height))"
                  + "|panel=\(Int(panel.frame.width))x\(Int(panel.frame.height))"
                  + "|stage=\(Int(stage.frame.width))x\(Int(stage.frame.height))"
                  + "|expand=\(Int(expand.frame.width))x\(Int(expand.frame.height))")
            XCTAssertTrue(expand.isHittable, name)
            XCTAssertGreaterThanOrEqual(expand.frame.width, 44, name)
            XCTAssertGreaterThanOrEqual(expand.frame.height, 44, name)
            XCTAssertLessThanOrEqual(panel.frame.height, window.height, "\(name): the panel fits the window")
            // §1.1: W is the card's width; ready, the stage is
            // H = min(round(W × 0.75), round(0.5 × B)) -- the formula is
            // P3's; here, its bounds: never over 3:4 of W, nor over half of
            // what is under the navigation bar (B is at most that).
            XCTAssertEqual(stage.frame.width, phone.width, accuracy: 1, "\(phone.name) \(name): W")
            XCTAssertLessThanOrEqual(stage.frame.height, (phone.width * 0.75).rounded() + 1,
                                     "\(phone.name) \(name): H ≤ 3:4 of W")
            let underBar = window.maxY - app.navigationBars.firstMatch.frame.maxY
            XCTAssertLessThanOrEqual(stage.frame.height, (underBar * 0.5).rounded() + 1,
                                     "\(phone.name) \(name): H ≤ half the screen under the bar")
            shoot("ui8-layout-\(name)")
            try audit(name)
            app.terminate()
        }
    }

    // MARK: UI9: drags are split between the world and the screen

    func testDragsAreSplitBetweenTheWorldAndTheScreen() throws {
        openReadyLiveWorld()
        let status = element("world-chrome-status")
        XCTAssertTrue(waitFor(timeout: 20) { status.exists && status.label == "loads 1" })
        let web = app.webViews.firstMatch
        let panel = element("wb-panel")
        let stage = element("wb-panel-stage")
        XCTAssertTrue(reveal(stage, whole: true))
        Thread.sleep(forTimeInterval: 1)
        let window = app.windows.firstMatch.frame
        // Clear of the home indicator: a sideways swipe there switches apps.
        XCTAssertLessThan(web.frame.midY, window.maxY - 80, "the drags are clear of the bottom edge")
        let before = panel.frame.minY
        let from = web.coordinate(withNormalizedOffset: CGVector(dx: 0.2, dy: 0.5))
        let to = web.coordinate(withNormalizedOffset: CGVector(dx: 0.85, dy: 0.5))
        from.press(forDuration: 0.05, thenDragTo: to, withVelocity: .default, thenHoldForDuration: 0.2)
        XCTAssertTrue(waitFor { status.exists && status.label == "got drag" }, "a horizontal drag reached the page: \(text(status))")
        XCTAssertEqual(panel.frame.minY, before, accuracy: 0.5, "and the screen did not scroll")
        // Downwards: the panel is near the end of the screen's content, so
        // the screen can always scroll back towards its top.
        let high = web.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.15))
        let low = web.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.85))
        high.press(forDuration: 0.05, thenDragTo: low, withVelocity: .default, thenHoldForDuration: 0.2)
        Thread.sleep(forTimeInterval: 1)
        print("UINLINE-DRAG|before=\(Int(before))|after=\(Int(panel.frame.minY))")
        XCTAssertGreaterThan(abs(panel.frame.minY - before), 20, "a vertical drag scrolls the screen")

        // Two fingers: a pinch on the world is the page's, start to end, and
        // the screen does not move under it.
        XCTAssertTrue(reveal(stage, whole: true))
        Thread.sleep(forTimeInterval: 1)
        let still = panel.frame.minY
        web.pinch(withScale: 2.5, velocity: 1.5)
        XCTAssertTrue(waitFor { status.exists && status.label == "got pinch" },
                      "a two-finger gesture reached the page, uncancelled: \(text(status))")
        Thread.sleep(forTimeInterval: 0.5)
        XCTAssertEqual(status.label, "got pinch", "the screen did not take the two fingers away")
        XCTAssertEqual(panel.frame.minY, still, accuracy: 0.5, "and the screen did not scroll")
    }

    // MARK: UI10: the VoiceOver order

    func testTheVoiceOverOrderOfThePanel() throws {
        let marker = "RESEARCH ORDER T"
        // With the inline dark line and hint over the world (review 2, MED 2).
        var state = Page.state(status: "Status T", hint: ["text": "Hint T", "opacity": 0.9], dark: true, edge: "left")
        state["research"] = ["raw": true, "marker": marker]
        mock.setRoute("GET /worlds/w1/render", status: 200,
                      body: Page.html(steps: [.init(state: state)], raw: true, marker: marker))
        mock.setRoute("GET /worlds/w1/render/revision", status: 200, body: Page.revision())
        openReadyLiveWorld(configure: false)
        XCTAssertTrue(element("world-chrome-research").waitForExistence(timeout: 30))
        XCTAssertTrue(waitFor(timeout: 20) { self.element("world-chrome-status").exists && self.element("world-chrome-status").label == "Status T" })
        XCTAssertTrue(element("world-chrome-dark").waitForExistence(timeout: 20), "the inline dark line")
        XCTAssertTrue(element("world-chrome-hint").exists, "the inline hint")
        // The world, then the chrome drawn over it. (A snapshot lists the
        // hierarchy, not VoiceOver's sort order: the priorities themselves
        // are pinned by WorldInlinePanelTests.)
        let expected = ["world-chrome-research", "wb-panel-headline", "wb-panel-stage", "web-view",
                        "world-chrome-dark", "world-chrome-hint", "world-chrome-status", "wb-panel-expand"]
        let snapshot = try element("wb-panel").snapshot()
        var order: [String] = []
        func visit(_ node: XCUIElementSnapshot) {
            let name = node.elementType == .webView ? "web-view" : node.identifier
            if expected.contains(name), !order.contains(name) { order.append(name) }
            node.children.forEach(visit)
        }
        visit(snapshot)
        print("UINLINE-ORDER|\(order.joined(separator: ","))")
        XCTAssertEqual(order, expected)
        XCTAssertEqual(element("wb-panel-expand").label, "Full screen")
    }

    // MARK: U3: the foreground banner

    func testTheAwayBannerShowsOnceForAWalkThatSettledInTheBackground() throws {
        mock.setRoute("GET /worlds/w1/render", status: 200, body: Page.html(steps: [.init(state: Page.state())]))
        mock.setRoute("GET /worlds/w1/render/revision", status: 200, body: Page.revision())
        openLive()
        push(modelState: "receiving")
        push(modelState: "finalizing", buildInProgress: true, finalizationState: "pending", finalSolve: "pending",
             processing: #"{"stage":"placing"}"#)
        XCTAssertTrue(element("wb-finish-line").waitForExistence(timeout: 15))
        XCUIDevice.shared.press(.home)
        Thread.sleep(forTimeInterval: 2)
        push(modelState: "finalized", finalizationState: "complete", finalSolve: "solved",
             photographic: #"{"state":"complete","stage":"appearance"}"#)
        Thread.sleep(forTimeInterval: 1)
        app.activate()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 10))
        // A reconnect after the background re-answers with the settled world.
        socket.liveReply = { subscription in
            PanelReport.text(subscription: subscription, seq: 99, modelState: "finalized",
                             finalizationState: "complete", finalSolve: "solved",
                             photographic: #"{"state":"complete","stage":"appearance"}"#)
        }
        let banner = element("wb-finish-banner")
        XCTAssertTrue(banner.waitForExistence(timeout: 20), "the banner")
        let text = app.staticTexts.containing(NSPredicate(format: "label == %@",
                                                          "While you were away, your walk finished: Saved.")).firstMatch
        XCTAssertTrue(text.exists, "its words")
        shoot("u3-banner")
        let ok = element("wb-finish-banner-ok")
        XCTAssertTrue(reveal(ok))
        ok.tap()
        XCTAssertTrue(waitFor { !banner.exists }, "OK dismisses it")
        Thread.sleep(forTimeInterval: 3)
        XCTAssertFalse(banner.exists, "it does not return")
    }

    // MARK: U4: an older Tower, and owed work

    func testAnOlderTowerSaysTheFallbackAndOwedWorkSaysWhy() throws {
        openLive()
        push(modelState: "receiving")
        push(modelState: "finalizing", buildInProgress: true, finalizationState: "pending", finalSolve: "pending")
        let line = element("wb-finish-line")
        XCTAssertTrue(line.waitForExistence(timeout: 15))
        XCTAssertTrue(waitFor { line.label.contains("Now: placing and checking images.") }, line.label)
        // One voice for the wait: the canvas's own is not drawn (review 2).
        XCTAssertFalse(app.staticTexts["The Tower is finishing this world."].exists, "the canvas narrates it too")
        shoot("u4-fallback")
        push(modelState: "finalizing", buildInProgress: false, finalizationState: "complete", finalSolve: "solved",
             photographic: #"{"state":"owed","stage":"appearance","detail":"d"}"#)
        let detail = element("wb-panel-finishing-detail")
        XCTAssertTrue(detail.waitForExistence(timeout: 15), "today's owed sentence")
        XCTAssertTrue(detail.label.hasPrefix("This world's photographic version is not finished"), detail.label)
        let owed = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH %@",
                                                        "This world's photographic version is not finished"))
        XCTAssertEqual(owed.count, 1, "the owed sentence once, the panel's")
        XCTAssertFalse(line.exists && line.label.contains("Now:"), "no stage line: \(text(line))")
    }

    // MARK: U2 and the screenshots: every state, default and AX5

    func testEveryPanelStateAtTheDefaultSize() throws {
        try everyState(size: nil, name: "default")
    }

    func testEveryPanelStateAtAX5() throws {
        try everyState(size: Self.ax5, name: "ax5")
    }

    private func everyState(size: String?, name: String) throws {
        mock.setRoute("GET /worlds/w1/render", status: 200,
                      body: Page.html(steps: [.init(state: Page.state(status: "1 of 24 images"))],
                                      labels: Page.realLabels(kind: "room")))
        mock.setRoute("GET /worlds/w1/render/revision", status: 200, body: Page.revision())
        openLive(size: size, extra: ["-WBPanelMapFixture"])
        push(modelState: "receiving")
        let map = element("wb-panel-map")
        XCTAssertTrue(map.waitForExistence(timeout: 15), "walking: the map slot")
        XCTAssertTrue(reveal(map, whole: true), "\(name): the walking map slot")
        shoot("state-walking-\(name)")

        push(modelState: "finalizing", buildInProgress: true, finalizationState: "pending", finalSolve: "pending",
             processing: #"{"stage":"checking","step":{"n":2,"of":3}}"#)
        let line = element("wb-finish-line")
        XCTAssertTrue(line.waitForExistence(timeout: 15))
        XCTAssertTrue(reveal(line, whole: true), "\(name): the finish block can be read")
        let window = app.windows.firstMatch.frame
        XCTAssertTrue(window.insetBy(dx: -1, dy: -1).contains(line.frame), "\(name): wholly inside the window")
        shoot("state-finishing-\(name)")
        let open = element("wb-open-3d")
        XCTAssertTrue(reveal(open, whole: true), "\(name): Open the preview")
        XCTAssertTrue(window.insetBy(dx: -1, dy: -1).contains(open.frame))
        if size != nil { try audit("finishing-\(name)") }

        push(modelState: "finalized", finalizationState: "complete", finalSolve: "solved",
             photographic: #"{"state":"complete","stage":"appearance"}"#)
        // The card above was brought into view, so the panel may be off the
        // screen: there it holds no web view (U-INLINE §4). Back to it first.
        XCTAssertTrue(element("wb-panel-stage").waitForExistence(timeout: 15))
        // The footer first, then the stage: a stage at the screen's bottom
        // edge leaves the panel under its visibility threshold, and then it
        // holds no web view (the 17e at AX5).
        XCTAssertTrue(element("wb-panel-expand").waitForExistence(timeout: 15), "\(name): the ready footer")
        XCTAssertTrue(reveal(element("wb-panel-expand")), "\(name): the ready panel's footer")
        XCTAssertTrue(reveal(element("wb-panel-stage")), "\(name): the ready panel")
        XCTAssertTrue(app.webViews.firstMatch.waitForExistence(timeout: 30))
        XCTAssertTrue(waitFor(timeout: 20) { self.element("world-chrome-status").exists && self.element("world-chrome-status").label == "1 of 24 images" })
        XCTAssertTrue(reveal(element("wb-panel"), whole: size == nil))
        Thread.sleep(forTimeInterval: 1)
        shoot("state-ready-\(name)")
        let expand = element("wb-panel-expand")
        XCTAssertTrue(reveal(expand))
        expand.tap()
        XCTAssertTrue(element("wb-cover-close").waitForExistence(timeout: 10))
        Thread.sleep(forTimeInterval: 2)
        shoot("state-expanded-\(name)")
        closeTheCover()

        push(modelState: "finalized", elements: 0, poses: 0, finalizationState: "complete", finalSolve: "solved")
        XCTAssertTrue(element("wb-panel-failed").waitForExistence(timeout: 15))
        XCTAssertTrue(reveal(element("wb-panel-failed")))
        shoot("state-failed-\(name)")

        mock.refusesSockets = true
        mock.dropSocket()
        // The capture control already says the Tower is not connected: the
        // panel shows only its own actions, never a third line (the lead).
        let connect = element("wb-panel-connect")
        XCTAssertTrue(connect.waitForExistence(timeout: 20), "the panel's own actions")
        XCTAssertTrue(element("wb-capture-tower-line").exists, "the screen's own offline line")
        XCTAssertFalse(element("wb-panel-offline").exists, "a third \"not connected\" line")
        XCTAssertTrue(reveal(connect))
        shoot("state-offline-\(name)")
    }

    // MARK: FOW v1: the coverage map

    /// The mock Tower serves FOW §9's fixtures. No `guidance` (the live
    /// Tower today) and `coverage:null` are today's panel; the mid-walk
    /// block draws two separate pieces with grey, muted and colored headings
    /// under its dated label, beside Capture health; another walk's report
    /// takes the map away and that walk's own block brings its own; a
    /// receipt older than 30 s says `stale`.
    func testTheCoverageMapIsDatedSeparateAndItsWalksOwn() throws {
        openLive()
        push(modelState: "receiving")
        XCTAssertTrue(element("capture-health").waitForExistence(timeout: 15), "Capture health")
        XCTAssertFalse(element("wb-panel-map").exists, "no guidance: no map")
        XCTAssertFalse(element("wb-panel-stage").exists, "no guidance: today's panel, Capture health alone")
        push(modelState: "receiving", guidance: FOWGuidance.none, accepted: 40)
        XCTAssertFalse(element("wb-panel-map").waitForExistence(timeout: 2), "coverage:null: no map yet")
        XCTAssertFalse(element("wb-panel-stage").exists, "coverage:null: today's panel")

        push(modelState: "receiving", guidance: FOWGuidance.mid, accepted: 127,
             towerSentAt: FOWGuidance.midSolvedAt + 12)
        let map = element("wb-panel-map")
        if !map.waitForExistence(timeout: 15) { print("FOW-AX|\(app.debugDescription)") }
        XCTAssertTrue(map.exists, "the map")
        XCTAssertTrue(reveal(map, whole: true), "the map on screen")
        let label = element("wb-panel-map-label")
        XCTAssertTrue(label.waitForExistence(timeout: 5), "the label")
        XCTAssertTrue(text(label).hasPrefix("Map through keyframe 113 · updated "), text(label))
        XCTAssertTrue(text(label).hasSuffix(" s ago"), "dated, not stale: \(text(label))")
        XCTAssertEqual(text(element("wb-panel-map-newer")), "14 newer keyframes not yet placed")
        XCTAssertEqual(text(element("wb-panel-map-pieces")), "2 separate pieces, not placed relative to each other")
        let pieces = app.descendants(matching: .any).matching(identifier: "wb-panel-map-piece")
        XCTAssertEqual(pieces.count, 2, "two components, two pieces")
        let first = pieces.element(boundBy: 0), second = pieces.element(boundBy: 1)
        XCTAssertEqual(first.label, "Piece 1")
        XCTAssertEqual(first.value as? String, "1 station. Two or more views: 4 headings. One view: 1. "
                       + "Not yet seen by a finished solve: 7.")
        XCTAssertEqual(second.label, "Piece 2")
        XCTAssertEqual(second.value as? String, "1 station. Two or more views: 1 headings. One view: 1. "
                       + "Not yet seen by a finished solve: 10.")
        XCTAssertFalse(first.frame.intersects(second.frame), "two pieces, never merged")
        XCTAssertTrue(map.frame.contains(first.frame) && map.frame.contains(second.frame), "in the map slot")
        XCTAssertTrue(element("capture-health").exists, "Capture health stays")
        shoot("fow-mid")
        // The pixels: grey, muted and colored headings are all drawn -- on an
        // iPad too. Sampled in piece 1's own screenshot, so no frame-to-raster
        // arithmetic is needed (see `pixels`).
        let shot = first.screenshot().image
        print("FOW-PIXELS|piece=\(first.frame)|shot=\(shot.size)@\(shot.scale)|orientation=\(shot.imageOrientation.rawValue)")
        for (name, rgb) in [("grey", (87, 87, 87)), ("muted", (0x2E, 0x6B, 0x66)), ("colored", (0x4F, 0xD8, 0xC8))] {
            XCTAssertGreaterThan(Self.pixels(in: shot, near: rgb), 3, "\(name) headings in piece 1")
        }

        // Another walk's report, with no block: A's map goes.
        push(modelState: "receiving", world: "w2", session: "s9", name: "Other Room")
        XCTAssertTrue(waitFor(timeout: 10) { !self.element("wb-panel-map").exists }, "A's map under B's walk")
        XCTAssertTrue(element("capture-health").exists, "B's walking panel is today's")

        // B's own block, 45 s old: B's map, stale.
        push(modelState: "receiving", world: "w2", session: "s9", name: "Other Room", guidance: FOWGuidance.stop,
             accepted: 1090, towerSentAt: FOWGuidance.stopSolvedAt + 45)
        XCTAssertTrue(map.waitForExistence(timeout: 15), "B's map")
        XCTAssertTrue(waitFor(timeout: 5) { self.text(label).hasPrefix("Map through keyframe 984 · updated ") },
                      text(label))
        XCTAssertTrue(text(label).hasSuffix(" s ago · stale"), text(label))
        XCTAssertEqual(text(element("wb-panel-map-newer")), "106 newer keyframes not yet placed")
        XCTAssertEqual(pieces.count, 1, "one component, one piece")
        XCTAssertFalse(element("wb-panel-map-pieces").exists, "one piece: no separate-pieces line")
        XCTAssertTrue(reveal(map, whole: true))
        shoot("fow-stale")
    }

    /// Pixels anywhere in `image` whose colour is within 20 of `rgb`.
    ///
    /// The whole raster of an element's own screenshot, never a box cut
    /// from the app's: on the iPad the app's screenshot is a portrait raster
    /// (EXIF-rotated) holding only part of the landscape window, offset from
    /// the element frames (measured on the iPad Air: a 1180x820 app came
    /// back as 2360x1640 tagged orientation 8, its content 360 pt down and
    /// cut off past x = 820), so no scale or offset maps a frame into it.
    /// A count over the whole raster does not depend on its orientation.
    private static func pixels(in image: UIImage, near rgb: (Int, Int, Int)) -> Int {
        guard let cg = image.cgImage else { return 0 }
        let width = cg.width, height = cg.height
        var data = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(data: &data, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return 0 }
        context.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        var count = 0
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                if abs(Int(data[i]) - rgb.0) <= 20, abs(Int(data[i + 1]) - rgb.1) <= 20,
                   abs(Int(data[i + 2]) - rgb.2) <= 20 { count += 1 }
            }
        }
        return count
    }

    // MARK: R1: a real page on a Mac scratch Tower

    /// Saved worlds → the fixture world → the panel shows the real page
    /// inline; Full screen moves it; a step taken full screen survives a
    /// collapse and a second expand (same `world-chrome-position`): no
    /// reload, the pose kept.
    func testR1ARealPageCrossFadesInlineAndKeepsItsPose() throws {
        // A REQUIRED gate: never a silent skip.
        guard let authority = ProcessInfo.processInfo.environment["GLASSES_UITEST_TOWER_AUTHORITY"],
              !authority.isEmpty
        else {
            XCTFail("R1 is required: set GLASSES_UITEST_TOWER_AUTHORITY=host:port of a real local Tower "
                    + "(TOWER_WORLD_NATIVE_CHROME=1) serving the fixture world.")
            return
        }
        launch(authority: authority)
        open(cartridge: "World Builder")
        openSavedWorlds()
        let row = app.staticTexts["Appearance fixture (Mac)"].firstMatch
        XCTAssertTrue(row.waitForExistence(timeout: 30), "the fixture world")
        XCTAssertTrue(tap(row, until: !app.navigationBars["Saved worlds"].exists))
        let stage = element("wb-panel-stage")
        let web = app.webViews.firstMatch
        XCTAssertTrue(web.waitForExistence(timeout: 60), "the real page inline")
        XCTAssertTrue(reveal(stage))
        XCTAssertTrue(waitFor(timeout: 60) { stage.frame.insetBy(dx: -1, dy: -1).contains(web.frame) })
        XCTAssertEqual(element("wb-panel-headline").label, "Saved")
        XCTAssertTrue(waitForTheWorldShown(timeout: 60), "the real page cross-faded in")
        Thread.sleep(forTimeInterval: 3)
        shoot("r1-inline")

        let expand = element("wb-panel-expand")
        XCTAssertTrue(reveal(expand))
        expand.tap()
        let window = app.windows.firstMatch.frame
        XCTAssertTrue(waitFor(timeout: 20) { web.frame.height > window.height * 0.5 }, "full screen")
        // Native chrome (TOWER_WORLD_NATIVE_CHROME on): the bar is the phone's.
        let native = element("world-chrome-best").waitForExistence(timeout: 60)
        print("UINLINE-R1|native=\(native)")
        XCTAssertTrue(native, "the Tower's page offered native chrome: run it with TOWER_WORLD_NATIVE_CHROME=1")
        let position = element("world-chrome-position")
        XCTAssertTrue(position.waitForExistence(timeout: 30), "the walk position")
        let next = element("world-chrome-next")
        XCTAssertTrue(next.waitForExistence(timeout: 10), "the next pose")
        if next.isEnabled { next.tap() }
        Thread.sleep(forTimeInterval: 3)
        XCTAssertTrue(position.exists, "the walk position after a step")
        let before = position.label
        shoot("r1-expanded")
        closeTheCover()
        XCTAssertTrue(waitFor(timeout: 15) { stage.frame.insetBy(dx: -1, dy: -1).contains(web.frame) }, "collapsed")
        XCTAssertEqual(webViewCount(), 1)
        Thread.sleep(forTimeInterval: 2)
        XCTAssertTrue(reveal(expand))
        expand.tap()
        XCTAssertTrue(waitFor(timeout: 20) { web.frame.height > window.height * 0.5 })
        XCTAssertTrue(position.waitForExistence(timeout: 30), "the walk position, expanded again")
        print("UINLINE-R1|position-before=\(before)|after=\(position.label)")
        XCTAssertEqual(position.label, before, "the pose survived the move")
        shoot("r1-expanded-again")
    }

    // MARK: Driving the mock

    private func openLive(size: String? = nil, extra: [String] = []) {
        mock.setRoute(CaptureHealthUITests.sessionStart, status: 200, body: DeadEndsUITests.session(state: "active"))
        mock.setRoute(CaptureHealthUITests.sessionStop, status: 200, body: DeadEndsUITests.session(state: "stopped"))
        launch(size: size, extra: extra)
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
    }

    /// A finished live world, its page loaded in the panel.
    private func openReadyLiveWorld(size: String? = nil, configure: Bool = true) {
        if configure {
            mock.setRoute("GET /worlds/w1/render", status: 200,
                          body: Page.html(steps: [.init(state: Page.state())], countsLoads: true))
            mock.setRoute("GET /worlds/w1/render/revision", status: 200, body: Page.revision())
        }
        openLive(size: size)
        push(modelState: "finalized", finalizationState: "complete", finalSolve: "solved",
             photographic: #"{"state":"complete","stage":"appearance"}"#)
        // The panel is below the fold at first; off screen for 5 s it holds
        // no web view (U-INLINE §4), so it is brought into view.
        XCTAssertTrue(element("wb-panel-stage").waitForExistence(timeout: 15), "the ready panel")
        XCTAssertTrue(reveal(element("wb-panel-stage")), "the panel on screen")
        XCTAssertTrue(app.webViews.firstMatch.waitForExistence(timeout: 30), "the world in the panel")
        XCTAssertTrue(waitForTheWorldShown(), "the page drew and the wait overlay went")
    }

    /// The wait overlay has gone: the page is shown.
    private func waitForTheWorldShown(timeout: TimeInterval = 30) -> Bool {
        waitFor(timeout: timeout) {
            !self.element("wb-panel-opening").exists && !self.element("wb-panel-finishing-detail").exists
                && !self.element("wb-finish-line").exists && self.app.webViews.firstMatch.exists
        }
    }

    /// WKWebViews on screen. XCUITest lists one web view as three nested
    /// `WebView` elements (the view, its scroll view, its content); only the
    /// outermost of each is counted.
    private func webViewCount() -> Int {
        guard let snapshot = try? app.snapshot() else { return app.webViews.count }
        var count = 0
        func visit(_ node: XCUIElementSnapshot, insideWebView: Bool) {
            let isWeb = node.elementType == .webView
            if isWeb, !insideWebView { count += 1 }
            node.children.forEach { visit($0, insideWebView: insideWebView || isWeb) }
        }
        visit(snapshot, insideWebView: false)
        return count
    }

    private func push(modelState: String, world: String = "w1", session: String = "s1", name: String = "Probe Room",
                      elements: Int = 1360, poses: Int = 40, reason: String? = nil,
                      buildInProgress: Bool? = nil, finalizationState: String? = nil, finalSolve: String? = nil,
                      processing: String? = nil, photographic: String? = nil, liveCapture: Bool = false,
                      guidance: String? = nil, accepted: Int? = nil, towerSentAt: Double = 1787463092.9) {
        seq += 1
        let seq = self.seq
        socket.sendLive { subscription in
            PanelReport.text(subscription: subscription, seq: seq, modelState: modelState, world: world,
                             session: session, name: name, elements: elements,
                             poses: poses, reason: reason, buildInProgress: buildInProgress,
                             finalizationState: finalizationState, finalSolve: finalSolve,
                             processing: processing, photographic: photographic, liveCapture: liveCapture,
                             guidance: guidance, accepted: accepted, towerSentAt: towerSentAt)
        }
        Thread.sleep(forTimeInterval: 0.7)
    }

    /// A capture on the mock glasses, the panel walking; settled once Stop is
    /// up. Skipped when Mock Device Kit gives no device.
    private func startCapture(size: String?, extra: [String]) throws -> (stop: XCUIElement, top: XCUIElement) {
        mock.setRoute(CaptureHealthUITests.sessionStart, status: 404, body: #"{"detail":"Not Found"}"#)
        launch(size: size, extra: extra, mockGlasses: true)
        open(cartridge: "World Builder")
        let start = app.buttons["Start capture"]
        guard waitFor(timeout: 15, { start.exists && start.isEnabled }) else {
            throw XCTSkip("Mock Device Kit gave no active device in this Simulator")
        }
        let top = app.buttons["Saved worlds"]
        XCTAssertTrue(reveal(start), "Start capture")
        start.tap()
        let stop = app.buttons["Stop capture"]
        XCTAssertTrue(stop.waitForExistence(timeout: 15), "the capture started")
        Thread.sleep(forTimeInterval: 3)
        return (stop, top)
    }

    private func launch(size: String? = nil, extra: [String] = [], mockGlasses: Bool = false,
                        authority: String? = nil) {
        let app = XCUIApplication()
        app.launchArguments = ["-UITestSkipOnboarding", "-UITestResetTowerAddress"] + extra
        if let size { app.launchArguments += ["-UIPreferredContentSizeCategoryName", size] }
        if mockGlasses {
            app.launchEnvironment["GLASSES_UITEST_AWAITING_BOUND_SECONDS"] = "600"
            app.launchArguments.append("-UITestMockGlasses")
            switch DeadEndsUITests.cameraFeed {
            case .success(let feed): app.launchEnvironment["GLASSES_UITEST_MOCK_CAMERA_FEED"] = feed.path
            case .failure(let problem): XCTFail("the mock glasses' camera feed: \(problem.text)")
            }
        }
        app.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = authority ?? mockAuthority
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

    private func openSavedWorlds() {
        let saved = app.buttons["Saved worlds"]
        XCTAssertTrue(reveal(saved), "Saved worlds")
        XCTAssertTrue(tap(saved, until: app.navigationBars["Saved worlds"].exists), "the Saved worlds sheet")
    }

    // MARK: The audit (UI8, U2)

    private func audit(_ name: String) throws {
        let shot = app.screenshot().image
        let window = app.windows.firstMatch.frame
        var issues: [String] = []
        try app.performAccessibilityAudit(for: [.contrast, .dynamicType, .textClipped, .hitRegion,
                                                .sufficientElementDescription]) { issue in
            let identifier = issue.element?.identifier ?? ""
            let label = issue.element?.label ?? ""
            let line = "\(name)|\(issue.auditType.rawValue)|\(identifier)|\(label.prefix(50))|\(issue.compactDescription)"
            if let waiver = self.waiver(issue, identifier: identifier, label: label, shot: shot, window: window) {
                print("UINLINE-AUDIT|WAIVED|\(line)|\(waiver)")
                return true
            }
            let measured = issue.element.flatMap { element in element.exists
                ? AccessibilityAuditUITests.measuredContrast(in: shot, window: window, frame: element.frame, ink: .dominant)
                : nil }
            print("UINLINE-AUDIT|ISSUE|\(line)|measured=\(measured.map { String(format: "%.2f", $0) } ?? "none")")
            issues.append(line)
            return true
        }
        XCTAssertEqual(issues, [], "\(name): every audit issue is fixed or waived with a reason")
    }

    /// Why an audit issue is not the panel's to fix, or `nil`.
    private func waiver(_ issue: XCUIAccessibilityAuditIssue, identifier: String, label: String,
                        shot: UIImage, window: CGRect) -> String? {
        guard let element = issue.element, element.exists else { return "no element: the audit could not name one" }
        let frame = element.frame
        let panel = self.element("wb-panel").frame
        if !panel.insetBy(dx: -2, dy: -2).intersects(frame) {
            return "outside the panel: the screen around it is unchanged by U-INLINE"
        }
        if issue.auditType == .contrast {
            // Measured on the pixels as they are NOW: the audit may have
            // moved the screen since the shot taken before it.
            let now = app.screenshot().image
            let ratios = [shot, now].compactMap {
                AccessibilityAuditUITests.measuredContrast(in: $0, window: window, frame: frame, ink: .dominant)
            }
            guard let ratio = ratios.max() else { return nil }
            return ratio >= 4.5 ? "measures \(String(format: "%.2f", ratio)):1 on the pixels" : nil
        }
        // Capped on purpose (U-INLINE §5, 168 #3): the footer button and the
        // inline dark line at AX1 with the large content viewer, the status
        // and hint at AX2, the research band at AX1.
        let capped = ["wb-panel-expand", "wb-panel-retry", "world-chrome-dark", "world-chrome-status",
                      "world-chrome-hint", "world-chrome-research", "wb-panel-connect", "wb-panel-settings"]
        if issue.auditType == .dynamicType || issue.auditType == .textClipped, capped.contains(identifier) {
            return "capped at AX1/AX2 with the large content viewer (U-INLINE §5)"
        }
        // The page's own text inside the web view is the page's (WORLDS §4).
        if app.webViews.firstMatch.exists, app.webViews.firstMatch.frame.contains(frame) {
            return "inside the page: the page's own text"
        }
        return nil
    }

    // MARK: Helpers

    /// An element's label, or "(gone)": reading a missing element's label
    /// throws, and a failure message must not.
    private func text(_ element: XCUIElement) -> String {
        element.exists ? element.label : "(gone)"
    }

    /// Close the cover, retrying a tap that landed while it was still
    /// presenting.
    private func closeTheCover(file: StaticString = #filePath, line: UInt = #line) {
        let close = element("wb-cover-close")
        XCTAssertTrue(tap(close, until: !close.exists), "Close collapses the cover", file: file, line: line)
    }

    private func element(_ identifier: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    private func beginning(_ text: String) -> XCUIElement {
        app.descendants(matching: .any).matching(NSPredicate(format: "label BEGINSWITH %@", text)).firstMatch
    }

    /// Scrolls until `element` is on screen and clear of the bars (`whole`:
    /// all of it, when it fits), by drags measured from where it is -- in
    /// the right-hand margin, so a drag that starts on the inline world is
    /// never the world's.
    /// 28 drags: at AX5 on the SE the panel is some 4,200 pt down, the first
    /// drags only collapse the shell's status bar, and 14 ran out (UI8).
    @discardableResult
    private func reveal(_ element: XCUIElement, whole: Bool = false, attempts: Int = 28) -> Bool {
        func needed() -> CGFloat? {
            guard element.exists else { return nil }
            let window = app.windows.firstMatch.frame
            let frame = element.frame
            let top = topInset(), bottom = window.maxY - 20
            if whole, frame.height <= bottom - top - 4 {
                // Accepted anywhere clear of the bars and the home
                // indicator; when it is not, aimed well inside, so a drag
                // that overshoots by its touch slop still lands.
                if frame.minY >= top - 1, frame.maxY <= bottom { return 0 }
                if frame.minY < top { return top + 20 - frame.minY }
                return (window.maxY - 60) - frame.maxY
            }
            if element.isHittable, frame.minY >= top - 1, frame.minY < bottom - 20 { return 0 }
            return top + 20 - frame.minY
        }
        var last: CGRect?
        for _ in 0..<attempts {
            guard let delta = needed() else {
                app.swipeUp(velocity: .slow)
                continue
            }
            if delta == 0 { return true }
            // A frame that did not move after a drag is not where the
            // element is (an element screens away can report a stale frame
            // at the accessibility sizes): the screen is at its end, so look
            // the other way.
            if let previous = last, abs(previous.minY - element.frame.minY) < 1 {
                if delta > 0 { app.swipeUp(velocity: .slow) } else { app.swipeDown(velocity: .slow) }
                last = nil
                continue
            }
            last = element.frame
            print("UINLINE-REVEAL|\(element.identifier)|\(element.frame)|hittable=\(element.isHittable)|delta=\(Int(delta))|top=\(Int(topInset()))")
            // Inside the scrolling area only: a drag that starts on the
            // pinned status bar or the navigation bar scrolls nothing. At
            // least 30 pt, past the touch slop.
            let window = app.windows.firstMatch.frame
            let top = topInset() + 20, bottom = window.maxY - 60
            let room = max(60, (bottom - top) * 0.8)
            var step = max(-room, min(room, delta))
            if abs(step) < 30 { step = step < 0 ? -30 : 30 }
            let mid = (top + bottom) / 2
            let origin = app.coordinate(withNormalizedOffset: CGVector(dx: 0.98, dy: 0))
            let from = origin.withOffset(CGVector(dx: 0, dy: mid - step / 2))
            let to = origin.withOffset(CGVector(dx: 0, dy: mid + step / 2))
            from.press(forDuration: 0.1, thenDragTo: to, withVelocity: .slow, thenHoldForDuration: 0.4)
        }
        return needed() == 0
    }

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
            if waitFor(timeout: 4, effect) { return true }
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

    private func shoot(_ name: String) {
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        guard let dir = ProcessInfo.processInfo.environment["UINLINE_SHOTS_DIR"], !dir.isEmpty else { return }
        let suffix = ProcessInfo.processInfo.environment["UINLINE_SHOT_SUFFIX"].map { "-\($0)" } ?? ""
        try? screenshot.pngRepresentation.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name)\(suffix).png"))
    }
}
