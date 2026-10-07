//
//  WorldChromeUITests.swift
//  GlassesUITests
//
//  U1.1 native chrome (WORLD-BUILDER-WORLDS.md §4c; IOS §10), in a real
//  `WKWebView` in the Simulator: U1–U10 against `MockTowerHTTPServer` and
//  the text-only bridge test page (`WorldChromeTestPage`), and R1–R5 against
//  a Mac scratch Tower on the U1.1 branch with TOWER_WORLD_NATIVE_CHROME on.
//
//  The mock tests never reach a real Tower: every launch points the app at
//  the mock's loopback port. They prove the phone's half in a real web view
//  under the appearance page's real CSP `<meta>`: the handler in the page
//  world, the handshake, activation, every state's drawing and hiding,
//  actions, the fallbacks, layout, the canvas share at the default size and
//  at AX5, the accessibility audit, and the VoiceOver order. The mock keys on
//  method and path and ignores the query, so it cannot see `wb-chrome=native`
//  on the request; `WorldChromeTests` I1 does.
//
//  R1–R5 need GLASSES_UITEST_TOWER_AUTHORITY (forwarded as
//  TEST_RUNNER_GLASSES_UITEST_TOWER_AUTHORITY) and skip without it, the
//  `TowerSmokeUITests` rule.
//

import UIKit
import XCTest

final class WorldChromeUITests: XCTestCase {

    private var app: XCUIApplication!
    private var mock: MockTowerHTTPServer!
    private var mockAuthority = ""
    /// The mock's World Builder socket: a Saved worlds pin is answered with
    /// the saved fixture world's report, so the World Builder panel shows it
    /// (U-INLINE §2.4) and Full screen opens the viewer these tests read.
    private var socket: PanelSocket!
    private let closedAuthority = "127.0.0.1:9"
    private static let ax5 = "UICTContentSizeCategoryAccessibilityXXXL"
    private typealias Page = WorldChromeTestPage

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

    // MARK: U1–U5: the protocol on screen

    /// U1: `hello` → native; each control shows the page's words and is
    /// named by the page's names. The page hides its own chrome only after.
    func testANativeChromePageIsDrawnFromItsLabels() throws {
        openViewer(page: Page.html(steps: [.init(state: Page.state())]))
        let best = element("world-chrome-best")
        XCTAssertTrue(best.waitForExistence(timeout: 30), "the native bar")
        XCTAssertEqual(best.label, "Best view T: fly")
        XCTAssertEqual(element("world-chrome-face").label, "Face the room T")
        XCTAssertEqual(element("world-chrome-previous").label, "Previous T")
        XCTAssertEqual(element("world-chrome-next").label, "Next T")
        XCTAssertEqual(element("world-chrome-reset").label, "Reset T")
        XCTAssertEqual(element("world-chrome-next").value as? String, "1 / 3", "the walk is the arrows' value")
        XCTAssertEqual(element("world-chrome-head").label, "Head T: captured images")
        XCTAssertEqual(element("world-chrome-about").label, "AboutT")
        XCTAssertEqual(element("world-chrome-ring").label, "Ring name T")
        XCTAssertEqual(element("world-chrome-position").label, "1 / 3")
        // The page put its own chrome away, after the phone drew its own.
        XCTAssertTrue(waitFor(timeout: 10) { !self.pageChrome.exists }, "the page hid its own chrome")
        // Today's caption line is replaced, not duplicated (IOS §10).
        XCTAssertFalse(legacyCaption.exists, "no native caption line under native chrome")
        XCTAssertTrue(element("world-render-reload").exists, "Reload stays in the toolbar")
        for control in ["world-chrome-best", "world-chrome-face", "world-chrome-previous", "world-chrome-next",
                        "world-chrome-reset"] {
            let frame = element(control).frame
            XCTAssertGreaterThanOrEqual(frame.width, 44, control)
            XCTAssertGreaterThanOrEqual(frame.height, 44, control)
        }
        shoot("u1-native")
    }

    /// U2: every state is shown and hidden as the state says; hidden means
    /// absent from the tree.
    func testEveryStateIsShownAndHiddenAsTheStateSays() throws {
        let marker = "RESEARCH MARKER T"
        func research(_ state: [String: Any]) -> [String: Any] {
            var state = state
            state["research"] = ["raw": true, "marker": marker]
            return state
        }
        let steps: [Page.Step] = [
            // Long enough to reach the full screen while it lasts: the page
            // starts in the World Builder panel and is expanded (U-INLINE).
            .init(state: research(Page.state(drawn: false, status: "Loading T",
                                             hint: ["text": "One moment T", "opacity": 0.9])), after: 12000),
            .init(state: research(Page.state(dark: true, edge: "left"))),
            .init(state: research(Page.state(hint: ["text": "Movement T", "opacity": 0.6], edge: "right"))),
            .init(state: research(Page.state(message: "Withdrawn T", phase: "withdrawn")), after: 4000),
            .init(state: research(Page.state(holding: true))),
        ]
        openViewer(page: Page.html(steps: steps, raw: true, marker: marker))

        // Before the first frame: the status, the hint and the marker, nothing else.
        let band = element("world-chrome-research")
        XCTAssertTrue(band.waitForExistence(timeout: 30), "the research band")
        XCTAssertTrue(band.label.contains(marker), band.label)
        let status = element("world-chrome-status")
        XCTAssertTrue(status.waitForExistence(timeout: 10), "the status before the first frame")
        XCTAssertEqual(status.label, "Loading T")
        XCTAssertEqual(element("world-chrome-hint").label, "One moment T")
        for absent in ["world-chrome-best", "world-chrome-head", "world-chrome-ring", "world-chrome-dark",
                       "world-chrome-about", "world-chrome-edge", "world-chrome-position"] {
            XCTAssertFalse(element(absent).exists, "\(absent) before the first frame")
        }
        shoot("u2-0-before-drawn")

        // Drawn, dark, an edge on the left.
        let dark = element("world-chrome-dark")
        XCTAssertTrue(dark.waitForExistence(timeout: 15), "the dark line")
        XCTAssertEqual(dark.label, "Dark name T")
        XCTAssertEqual(element("world-chrome-edge").value as? String, "left")
        XCTAssertTrue(element("world-chrome-ring").exists)
        XCTAssertTrue(element("world-chrome-head").exists)
        XCTAssertTrue(element("world-chrome-best").exists)
        XCTAssertFalse(status.exists, "no empty status element")
        XCTAssertFalse(element("world-chrome-hint").exists)
        XCTAssertLessThanOrEqual(band.frame.maxY, app.webViews.firstMatch.frame.minY + 1, "the band is above the canvas")
        shoot("u2-1-dark")

        // A step: the hint, no dark line, the edge on the right.
        element("world-chrome-next").tap()
        XCTAssertTrue(waitFor { status.exists && status.label == "got next" },
                      "the page took next; status \(status.exists ? status.label : "absent"), legacy "
                        + "\(legacyCaption.exists), page chrome \(pageChrome.exists), "
                        + "next \(element("world-chrome-next").exists)")
        XCTAssertTrue(waitFor { !dark.exists }, "the dark line left the tree")
        XCTAssertEqual(element("world-chrome-hint").label, "Movement T")
        XCTAssertEqual(element("world-chrome-edge").value as? String, "right")
        // The hint, the dark line and the status never overlap.
        XCTAssertLessThanOrEqual(element("world-chrome-hint").frame.maxY, status.frame.minY + 1)

        // The caption panel: sections, and the toggle's words.
        let toggle = element("world-chrome-about")
        toggle.tap()
        let panel = element("world-chrome-panel")
        XCTAssertTrue(panel.waitForExistence(timeout: 5), "the caption panel")
        XCTAssertEqual(toggle.label, "LessT")
        XCTAssertEqual(app.descendants(matching: .any).matching(identifier: "world-chrome-section").count, 2)
        XCTAssertTrue(app.staticTexts["Body 1 T."].exists)
        XCTAssertTrue(app.staticTexts["Tail T"].exists)
        XCTAssertFalse(element("world-chrome-ring").exists, "the ring steps aside for the panel")
        XCTAssertLessThan(band.frame.minY, panel.frame.minY, "the marker is above the panel")
        shoot("u2-2-panel")
        toggle.tap()
        XCTAssertTrue(waitFor { !panel.exists }, "the panel closed")
        XCTAssertEqual(toggle.label, "AboutT")

        // The message: in place of the picture; the bar, head and ring go.
        element("world-chrome-reset").tap()
        let message = element("world-chrome-message")
        XCTAssertTrue(message.waitForExistence(timeout: 10), "the message")
        XCTAssertEqual(message.label, "Withdrawn T")
        for absent in ["world-chrome-best", "world-chrome-head", "world-chrome-ring", "world-chrome-status",
                       "world-chrome-hint", "world-chrome-about", "world-chrome-edge"] {
            XCTAssertFalse(element(absent).exists, "\(absent) under a message")
        }
        XCTAssertTrue(band.exists, "the marker stays")
        XCTAssertLessThan(band.frame.maxY, message.frame.minY + 1, "the marker is above the message")
        shoot("u2-3-message")

        // Holding: the bar is back.
        XCTAssertTrue(element("world-chrome-best").waitForExistence(timeout: 10), "the bar after the message")
        XCTAssertFalse(message.exists)
    }

    /// U3: every control reaches the page; a disabled one sends nothing.
    func testEveryControlReachesThePage() throws {
        let buttons = ["best": true, "face": true, "previous": false, "next": true, "reset": true]
        openViewer(page: Page.html(steps: [.init(state: Page.state(dark: true, edge: "left", buttons: buttons))]))
        XCTAssertTrue(element("world-chrome-best").waitForExistence(timeout: 30))
        let status = element("world-chrome-status")
        func press(_ identifier: String, expect name: String) {
            let control = element(identifier)
            XCTAssertTrue(control.waitForExistence(timeout: 5), identifier)
            control.tap()
            XCTAssertTrue(waitFor { status.exists && status.label == "got \(name)" },
                          "\(identifier) → got \(name); status reads \(status.exists ? status.label : "nothing")")
        }
        press("world-chrome-best", expect: "best")
        press("world-chrome-ring", expect: "face")
        press("world-chrome-next", expect: "next")
        press("world-chrome-dark", expect: "face")
        press("world-chrome-reset", expect: "reset")
        press("world-chrome-edge", expect: "face")
        press("world-chrome-best", expect: "best")
        press("world-chrome-face", expect: "face")
        let previous = element("world-chrome-previous")
        XCTAssertFalse(previous.isEnabled, "the page said previous is disabled")
        previous.tap()
        Thread.sleep(forTimeInterval: 2)
        XCTAssertEqual(status.label, "got face", "a disabled control sends nothing")
        // Empty chrome space passes the touch to the page's canvas.
        app.webViews.firstMatch.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.45)).tap()
        XCTAssertTrue(waitFor { status.exists && status.label == "got touch" },
                      "a tap on the empty canvas reached the page; status reads \(status.label)")
    }

    /// U4: no echo, today's screen.
    func testAPageWithoutTheEchoKeepsTodaysScreen() throws {
        openViewer(page: Page.html(steps: [.init(state: Page.state())], echo: false))
        XCTAssertTrue(legacyCaption.waitForExistence(timeout: 30), "today's caption line")
        // This page fetches no manifest, so no `X-World-Imagery` header and no
        // page word reach the app: the caption claims no redaction either way
        // (final-gate review of cb865eb, LOW).
        XCTAssertTrue(legacyCaption.label.contains("redaction unknown"), "today's words: \(legacyCaption.label)")
        XCTAssertFalse(legacyCaption.label.contains("faces redacted"), legacyCaption.label)
        XCTAssertTrue(element("world-render-reload").exists)
        XCTAssertTrue(pageChrome.waitForExistence(timeout: 10), "the page's own chrome")
        Thread.sleep(forTimeInterval: 3)
        XCTAssertTrue(pageChrome.exists, "and it stays")
        assertNoNativeChrome()
        shoot("u4-legacy")
    }

    /// On today's screen -- a page without the echo, as a Tower that serves
    /// no native chrome sends -- the research marker the served header
    /// raised is drawn above the caption: the header is the source that
    /// still speaks when the page's own marker does not (IOS §10, spec C15r
    /// §3.7). The page's own chrome stays.
    func testTheHeadersResearchMarkerIsOnTodaysScreen() throws {
        let warning = "Research T: unredacted imagery, from the header"
        mock.setRoute("GET /worlds/w1/appearance/s1/manifest", status: 200, body: "{}",
                      headers: ["X-World-Imagery": "raw", "X-World-Imagery-Warning": warning])
        openViewer(page: Page.html(steps: [.init(state: Page.state())], echo: false, fetchManifest: true))
        XCTAssertTrue(legacyCaption.waitForExistence(timeout: 30), "today's caption line")
        let band = element("world-chrome-research")
        XCTAssertTrue(band.waitForExistence(timeout: 10), "the header's research marker on today's screen")
        XCTAssertTrue(band.label.contains(warning.uppercased()), band.label)
        XCTAssertLessThanOrEqual(band.frame.maxY, legacyCaption.frame.minY + 1, "above the caption")
        // And the caption under it makes no redaction claim it cannot keep.
        XCTAssertTrue(waitFor(timeout: 5) { self.legacyCaption.label.contains("not redacted") },
                      "the caption says the imagery is not redacted: \(legacyCaption.label)")
        XCTAssertFalse(legacyCaption.label.contains("faces redacted"), legacyCaption.label)
        XCTAssertTrue(pageChrome.exists, "the page keeps its own chrome")
        for id in ["world-chrome-best", "world-chrome-head", "world-chrome-status"] {
            XCTAssertFalse(element(id).exists, "\(id) on today's screen")
        }
        shoot("u4b-legacy-research")
    }

    /// U5: a bad hello, or a page that says nothing, falls back to the web
    /// chrome for the screen.
    func testABadHelloOrASilentPageFallsBackToTheWebChrome() throws {
        openViewer(page: Page.html(steps: [.init(state: Page.state())], hello: .missingHooks))
        XCTAssertTrue(legacyCaption.waitForExistence(timeout: 30), "declined: today's caption line")
        XCTAssertTrue(pageChrome.waitForExistence(timeout: 10), "the page keeps its own chrome")
        assertNoNativeChrome()
        shoot("u5-declined")
        app.terminate()

        openViewer(page: Page.html(steps: [.init(state: Page.state())], hello: .silent))
        XCTAssertTrue(app.webViews.firstMatch.waitForExistence(timeout: 30))
        let started = Date()
        XCTAssertTrue(legacyCaption.waitForExistence(timeout: 20), "silent: today's screen after the timeout")
        print("U11-FALLBACK|silent|legacy after \(String(format: "%.1f", Date().timeIntervalSince(started))) s")
        XCTAssertTrue(pageChrome.exists)
        assertNoNativeChrome()
    }

    // MARK: U6–U8: layout

    /// U6: the canvas share at the default size.
    func testTheCanvasShareAtTheDefaultSize() throws {
        openViewer(page: Page.html(steps: [.init(state: Page.state())], labels: Page.realLabels(kind: "room")))
        XCTAssertTrue(element("world-chrome-best").waitForExistence(timeout: 30))
        Thread.sleep(forTimeInterval: 1)
        let share = canvasShare("default-room")
        let window = app.windows.firstMatch.frame
        XCTAssertGreaterThanOrEqual(share, window.height >= 800 ? 0.80 : 0.60)
        XCTAssertLessThanOrEqual(element("world-chrome-head").frame.maxY, app.webViews.firstMatch.frame.minY + 1,
                                 "the head is whole above the canvas")
        shoot("u6-share-default")
    }

    /// U7: at AX5 the canvas keeps 60 %, and all five controls can be hit, at
    /// least 44 × 44, on two rows.
    func testTheBarAndTheCanvasAtAX5() throws {
        try skipOnAnIPad("the bar's two rows at AX5 are a phone's width; an iPad's bar is one row")
        openViewer(page: Page.html(steps: [.init(state: Page.state())], labels: Page.realLabels(kind: "room")),
                   size: Self.ax5)
        XCTAssertTrue(element("world-chrome-best").waitForExistence(timeout: 30))
        Thread.sleep(forTimeInterval: 1)
        let share = canvasShare("ax5-room")
        XCTAssertGreaterThanOrEqual(share, 0.60)
        assertTheBarIsReachable(twoRows: true)
        // The band's words scroll in their own 18 % at AX5, whole, above the
        // canvas -- never one line with the canvas laid over the rest.
        let scroller = app.scrollViews.containing(.any, identifier: "world-chrome-head").firstMatch
        XCTAssertTrue(scroller.exists, "the words scroll at AX5")
        let web = app.webViews.firstMatch.frame
        print("U11-WORDS|scroller=\(Int(scroller.frame.minY))-\(Int(scroller.frame.maxY))|web=\(Int(web.minY))")
        XCTAssertGreaterThan(scroller.frame.height, 60, "more than one line of words at AX5")
        XCTAssertLessThanOrEqual(scroller.frame.maxY, web.minY + 1, "the words end above the canvas")
        shoot("u7-share-ax5")
    }

    /// U8: the area viewer at AX5: *Back to the room* and the caption toggle
    /// in the top row, the area's own face control, and 60 % of the screen.
    func testTheAreaViewerAtAX5() throws {
        try skipOnAnIPad("the bar's two rows at AX5 are a phone's width; an iPad's bar is one row")
        openViewer(page: Page.html(steps: [.init(state: Page.state())], labels: Page.realLabels(kind: "room")),
                   area: Page.html(steps: [.init(state: Page.state())], kind: "area",
                                   labels: Page.realLabels(kind: "area")),
                   size: Self.ax5)
        let areas = element("world-chrome-areas")
        XCTAssertTrue(areas.waitForExistence(timeout: 30), "Areas, for a room with an area")
        areas.tap()
        let row = element("world-render-areas")
        XCTAssertTrue(row.waitForExistence(timeout: 5), "the areas panel")
        let entry = row.buttons.firstMatch.exists ? row.buttons.firstMatch
            : app.buttons.containing(NSPredicate(format: "label BEGINSWITH %@", "Area 1")).firstMatch
        XCTAssertTrue(entry.waitForExistence(timeout: 5), "the area's entry")
        entry.tap()
        let back = element("world-render-back-to-room")
        XCTAssertTrue(back.waitForExistence(timeout: 30), "Back to the room")
        let face = element("world-chrome-face")
        XCTAssertTrue(face.waitForExistence(timeout: 30), "the area's bar")
        XCTAssertTrue(face.label.hasPrefix("Face the area"), face.label)
        let toggle = element("world-chrome-about")
        XCTAssertTrue(toggle.waitForExistence(timeout: 5))
        let web = app.webViews.firstMatch
        XCTAssertTrue(back.isHittable)
        XCTAssertTrue(toggle.isHittable)
        XCTAssertLessThanOrEqual(back.frame.maxY, web.frame.minY + 1, "Back to the room is in the top band")
        XCTAssertLessThanOrEqual(toggle.frame.maxY, web.frame.minY + 1, "the toggle is in the top band")
        XCTAssertGreaterThanOrEqual(canvasShare("ax5-area"), 0.60)
        assertTheBarIsReachable(twoRows: true)
        shoot("u8-area-ax5")
    }

    /// U11: the canvas stays where it first appeared when the page's head
    /// arrives. Native geometry begins with the echo, before the page has
    /// said its head; the band above the canvas must not grow when a head of
    /// several lines arrives at the default size, nor when the caption
    /// toggle arrives with it or changes its word, so the first drawn frame
    /// does not resize the canvas (spec C15r §3.5). The walk carries a notice
    /// long enough to wrap, so a narrower column of words would move the
    /// canvas too.
    func testTheCanvasStaysStillWhenAMultilineHeadArrives() throws {
        let head = "Head T: captured images on reconstructed geometry, with the walk's other areas "
            + "shown separately -- a head of several lines at the default size"
        let notice = "Notice T: the Tower could not finish every pass this walk asked for, so some of "
            + "the walls are drawn from fewer images than usual. An owner can run the walk again from "
            + "the Tower, and the picture here changes when it does. Nothing else about the room is affected."
        let steps: [Page.Step] = [
            .init(state: Page.state(drawn: false, status: "Loading T"), after: 12000),
            .init(state: Page.state(head: head)),
        ]
        openViewer(page: Page.html(steps: steps, labels: Page.realLabels(kind: "room")), notice: notice)
        let web = app.webViews.firstMatch
        XCTAssertTrue(web.waitForExistence(timeout: 30), "the canvas")
        XCTAssertTrue(element("world-chrome-status").waitForExistence(timeout: 10), "native, before the first frame")
        XCTAssertTrue(element("world-render-notice").exists, "the walk's notice is in the band")
        let first = web.frame
        let headElement = element("world-chrome-head")
        XCTAssertFalse(headElement.exists, "measured before the head arrived")
        shoot("u11-before-head")

        XCTAssertTrue(headElement.waitForExistence(timeout: 20), "the head arrived")
        XCTAssertEqual(headElement.label, head, "the whole head, verbatim")
        let toggle = element("world-chrome-about")
        XCTAssertTrue(toggle.waitForExistence(timeout: 5), "the caption toggle arrived with it")
        Thread.sleep(forTimeInterval: 1)
        print("U11-STILL|first=\(Int(first.minY))-\(Int(first.maxY))|drawn=\(Int(web.frame.minY))-\(Int(web.frame.maxY))"
              + "|head=\(Int(headElement.frame.minY))-\(Int(headElement.frame.maxY))")
        assertTheCanvas(web, isAt: first, "after the head and the toggle arrived")
        shoot("u11-after-head")

        // The toggle's two words differ in width: opening and closing the
        // panel moves nothing either.
        toggle.tap()
        XCTAssertTrue(element("world-chrome-panel").waitForExistence(timeout: 5), "the caption panel")
        assertTheCanvas(web, isAt: first, "with the panel open")
        toggle.tap()
        XCTAssertTrue(waitFor { !self.element("world-chrome-panel").exists }, "the panel closed")
        assertTheCanvas(web, isAt: first, "with the panel closed again")
    }

    // MARK: U9, U10: accessibility

    /// U9: XCUITest's audit, at the default size and at AX5. Every issue is
    /// fixed or waived with its reason, printed on a `U11-AUDIT|` line; a
    /// "Contrast failed" stands only if the text also measures under 4.5:1 on
    /// the rendered pixels (the `AccessibilityAuditUITests` rule: the audit
    /// samples text over a picture badly).
    func testTheNativeChromePassesTheAccessibilityAudit() throws {
        let page = Page.html(steps: [.init(state: Page.state(hint: ["text": "Movement stops here", "opacity": 0.9],
                                                             dark: true, edge: "left"))],
                             labels: Page.realLabels(kind: "room"))
        for size in [nil, Self.ax5] {
            openViewer(page: page, size: size)
            XCTAssertTrue(element("world-chrome-best").waitForExistence(timeout: 30))
            Thread.sleep(forTimeInterval: 2)
            let mode = size == nil ? "default" : "ax5"
            let shot = app.screenshot().image
            let window = app.windows.firstMatch.frame
            var issues: [String] = []
            try app.performAccessibilityAudit(for: [.contrast, .dynamicType, .hitRegion, .textClipped,
                                                    .sufficientElementDescription]) { issue in
                let identifier = issue.element?.identifier ?? ""
                let label = issue.element?.label ?? ""
                let line = "\(mode)|\(Self.typeName(issue.auditType))|\(identifier)|\(label.prefix(50))"
                    + "|\(issue.compactDescription)"
                if let waiver = self.waiver(issue, identifier: identifier, label: label, shot: shot, window: window) {
                    print("U11-AUDIT|WAIVED|\(line)|\(waiver)")
                    return true
                }
                print("U11-AUDIT|ISSUE|\(line)")
                issues.append(line)
                return true
            }
            XCTAssertEqual(issues, [], "\(mode): every audit issue is either fixed or waived with a reason")
            app.terminate()
        }
    }

    private static func typeName(_ type: XCUIAccessibilityAuditType) -> String {
        switch type {
        case .contrast: return "contrast"
        case .dynamicType: return "dynamicType"
        case .hitRegion: return "hitRegion"
        case .textClipped: return "textClipped"
        case .sufficientElementDescription: return "description"
        default: return "\(type.rawValue)"
        }
    }

    /// Why an audit issue is not the native chrome's to fix, or `nil`.
    private func waiver(_ issue: XCUIAccessibilityAuditIssue, identifier: String, label: String,
                        shot: UIImage, window: CGRect) -> String? {
        guard let element = issue.element, element.exists else {
            return "no element: the audit could not name one"
        }
        let frame = element.frame
        if app.navigationBars.allElementsBoundByIndex.contains(where: {
            $0.exists && $0.frame.insetBy(dx: -2, dy: -2).contains(frame)
        }) {
            return "the system navigation bar (title, Reload), unchanged by U1.1"
        }
        if issue.auditType == .contrast {
            guard let ratio = AccessibilityAuditUITests.measuredContrast(in: shot, window: window, frame: frame,
                                                                          ink: .dominant)
            else { return nil }
            return ratio >= 4.5 ? "measures \(String(format: "%.2f", ratio)):1 on the pixels" : nil
        }
        // Capped on purpose (§3.5, Q3): the bar, the band's controls and the
        // dark line at AX1, with the large content viewer; the pills at AX2;
        // the ring's label, the position and the research band at AX1. The
        // dark line's and the ring's texts are reported without the control's
        // identifier, so they are known by the page's words.
        let capped = ["world-chrome-best", "world-chrome-face", "world-chrome-previous", "world-chrome-next",
                      "world-chrome-reset", "world-chrome-about", "world-chrome-areas", "world-render-back-to-room",
                      "world-chrome-dark", "world-chrome-status", "world-chrome-hint", "world-chrome-position",
                      "world-chrome-research", "world-chrome-ring", "world-render-newer-picture",
                      "world-render-retry-refused"]
        let cappedWords = ["Not reconstructed from here", "Tap to turn back", "reconstructed\nfrom here",
                           "reconstructed", "from here", "Movement stops here", "1 / 3"]
        if issue.auditType == .dynamicType || issue.auditType == .textClipped,
           capped.contains(identifier) || cappedWords.contains(label) {
            return "capped at AX1/AX2 with the large content viewer (§3.5, Q3)"
        }
        return nil
    }

    /// U10: the VoiceOver order of §3.6, as the accessibility tree lists it.
    func testTheVoiceOverOrderOfTheNativeChrome() throws {
        let marker = "RESEARCH MARKER T"
        var state = Page.state(status: "Status T", hint: ["text": "Hint T", "opacity": 0.9], dark: true)
        state["research"] = ["raw": true, "marker": marker]
        openViewer(page: Page.html(steps: [.init(state: state)], raw: true, marker: marker))
        XCTAssertTrue(element("world-chrome-best").waitForExistence(timeout: 30))
        Thread.sleep(forTimeInterval: 1)
        let expected = ["world-chrome-research", "world-chrome-head", "world-chrome-about", "world-chrome-ring",
                        "world-chrome-dark", "world-chrome-hint", "world-chrome-status", "world-chrome-best",
                        "world-chrome-face", "world-chrome-previous", "world-chrome-next", "world-chrome-reset"]
        let snapshot = try app.snapshot()
        var order: [String] = []
        func visit(_ node: XCUIElementSnapshot) {
            if expected.contains(node.identifier), !order.contains(node.identifier) { order.append(node.identifier) }
            node.children.forEach(visit)
        }
        visit(snapshot)
        print("U11-ORDER|\(order.joined(separator: ","))")
        XCTAssertEqual(order, expected)
    }

    // MARK: R1–R5: the real page on a Mac scratch Tower

    /// R1: the real page hands over its chrome; → moves the walk; Reset
    /// works; Best view becomes enabled.
    func testTheRealPageHandsOverItsChrome() throws {
        try openRealWorld()
        let head = element("world-chrome-head")
        XCTAssertTrue(head.waitForExistence(timeout: 120), "the page's head line, natively")
        XCTAssertFalse(head.label.isEmpty)
        let web = app.webViews.firstMatch
        XCTAssertTrue(waitFor(timeout: 15) { web.buttons.count == 0 }, "the page's own buttons left the tree")
        let next = element("world-chrome-next")
        XCTAssertTrue(next.waitForExistence(timeout: 30))
        let position = element("world-chrome-position")
        let before = position.exists ? position.label : ""
        print("U11-R1|head=\(head.label.prefix(60))|position=\(before)|best=\(element("world-chrome-best").isEnabled)")
        shoot("r1-before")
        if next.isEnabled {
            next.tap()
            XCTAssertTrue(waitFor(timeout: 20) { position.exists && position.label != before },
                          "→ moved the walk position (was \(before))")
        }
        let reset = element("world-chrome-reset")
        if reset.isEnabled { reset.tap() }
        let best = element("world-chrome-best")
        XCTAssertTrue(waitFor(timeout: 30) { best.exists && best.isEnabled }, "Best view becomes enabled")
        print("U11-R1|after|position=\(position.exists ? position.label : "-")|best=\(best.isEnabled)")
        shoot("r1-after")
    }

    /// R2: the cold open says what it is doing, before any bar, head or ring.
    /// One snapshot of the tree per pass, so a short pre-draw window is not
    /// missed between separate queries.
    ///
    /// U-INLINE: a world opened from Saved worlds now opens in the World
    /// Builder panel, and its cold open happens there, under the panel's
    /// wait overlay (`wb-panel-opening`), with the page's status in the
    /// panel's footer. "Before the picture" is: while that overlay is up.
    func testTheColdOpenSaysWhatItIsDoing() throws {
        try openRealWorld(expand: false)
        let watched = ["world-chrome-status", "world-chrome-best", "world-chrome-head", "world-chrome-ring",
                       "wb-panel-opening"]
        var coldStatus: String?
        var headFirstSeenWithoutAColdStatus = false
        let deadline = Date().addingTimeInterval(90)
        while Date() < deadline {
            guard let snapshot = try? app.snapshot() else { continue }
            var seen: [String: String] = [:]
            func visit(_ node: XCUIElementSnapshot) {
                if watched.contains(node.identifier) { seen[node.identifier] = node.label }
                node.children.forEach(visit)
            }
            visit(snapshot)
            if coldStatus == nil, let status = seen["world-chrome-status"], seen["world-chrome-best"] == nil,
               seen["world-chrome-head"] == nil, seen["world-chrome-ring"] == nil,
               seen["wb-panel-opening"] != nil {
                coldStatus = status
                shoot("r2-cold-open")
            }
            if seen["world-chrome-head"] != nil || (coldStatus != nil && seen["wb-panel-opening"] == nil) {
                headFirstSeenWithoutAColdStatus = coldStatus == nil
                break
            }
        }
        print("U11-R2|coldStatus=\(coldStatus ?? "none")|headFirst=\(headFirstSeenWithoutAColdStatus)")
        XCTAssertNotNil(coldStatus, "a status sentence showed natively, alone, before the first frame")
        XCTAssertFalse(headFirstSeenWithoutAColdStatus, "no bar, head or ring before the picture")
    }

    /// R3: the canvas share on the real page, default and AX5.
    func testTheCanvasShareOnTheRealPage() throws {
        try skipOnAnIPad("the bar's two rows at AX5 are a phone's width; an iPad's bar is one row")
        for size in [nil, Self.ax5] {
            try openRealWorld(size: size)
            XCTAssertTrue(element("world-chrome-best").waitForExistence(timeout: 120), "the native bar")
            Thread.sleep(forTimeInterval: 2)
            let mode = size == nil ? "default" : "ax5"
            let share = canvasShare("real-\(mode)")
            let window = app.windows.firstMatch.frame
            XCTAssertGreaterThanOrEqual(share, size == nil ? (window.height >= 800 ? 0.70 : 0.60) : 0.60)
            if size != nil { assertTheBarIsReachable(twoRows: true) }
            shoot("r3-\(mode)")
            app.terminate()
        }
    }

    /// R4 (best effort): turning away until the page says it is dark; the
    /// dark line clears it.
    func testTurningAwayShowsTheDarkLine() throws {
        try openRealWorld()
        XCTAssertTrue(element("world-chrome-best").waitForExistence(timeout: 120))
        let web = app.webViews.firstMatch
        let dark = element("world-chrome-dark")
        for _ in 0..<16 where !dark.exists {
            let from = web.coordinate(withNormalizedOffset: CGVector(dx: 0.25, dy: 0.5))
            let to = web.coordinate(withNormalizedOffset: CGVector(dx: 0.85, dy: 0.5))
            from.press(forDuration: 0.05, thenDragTo: to, withVelocity: .slow, thenHoldForDuration: 0.6)
            _ = dark.waitForExistence(timeout: 1.5)
        }
        guard dark.exists else { throw XCTSkip("the fixture never went dark from the drags tried") }
        shoot("r4-dark")
        dark.tap()
        XCTAssertTrue(waitFor(timeout: 15) { !dark.exists }, "the dark line turned the view back")
    }

    // MARK: Helpers

    private var legacyCaption: XCUIElement {
        app.staticTexts.containing(NSPredicate(format: "label CONTAINS[c] %@", "not to scale")).firstMatch
    }

    private var pageChrome: XCUIElement {
        app.webViews.firstMatch.staticTexts["PAGE CHROME"]
    }

    private func assertNoNativeChrome(file: StaticString = #filePath, line: UInt = #line) {
        for id in ["world-chrome-best", "world-chrome-head", "world-chrome-ring", "world-chrome-status",
                   "world-chrome-research"] {
            XCTAssertFalse(element(id).exists, "\(id) on today's screen", file: file, line: line)
        }
    }

    private func assertTheBarIsReachable(twoRows: Bool, file: StaticString = #filePath, line: UInt = #line) {
        var frames: [String: CGRect] = [:]
        for id in ["world-chrome-best", "world-chrome-face", "world-chrome-previous", "world-chrome-next",
                   "world-chrome-reset"] {
            let control = element(id)
            XCTAssertTrue(control.isHittable, "\(id) can be hit", file: file, line: line)
            XCTAssertGreaterThanOrEqual(control.frame.width, 44, id, file: file, line: line)
            XCTAssertGreaterThanOrEqual(control.frame.height, 44, id, file: file, line: line)
            frames[id] = control.frame
        }
        let rows = Set(frames.values.map { Int($0.midY.rounded()) }).count
        print("U11-BAR|rows=\(rows)|" + frames.sorted { $0.key < $1.key }
            .map { "\($0.key.replacingOccurrences(of: "world-chrome-", with: ""))=\(Int($0.value.minX)),\(Int($0.value.minY)) \(Int($0.value.width))x\(Int($0.value.height))" }
            .joined(separator: "|"))
        if twoRows, let best = frames["world-chrome-best"], let next = frames["world-chrome-next"] {
            XCTAssertLessThan(best.maxY, next.minY + 1, "two rows: Best view above the arrows", file: file, line: line)
        }
    }

    /// The canvas share: the web view's height ÷ (window − the viewer's
    /// navigation bar maxY − the bottom inset), printed as `U11-SHARE|`.
    @discardableResult
    private func canvasShare(_ name: String) -> CGFloat {
        let window = app.windows.firstMatch.frame
        let web = app.webViews.firstMatch.frame
        // The viewer's own bar: the lowest navigation bar above the picture.
        let bars = app.navigationBars.allElementsBoundByIndex.map(\.frame).filter { $0.maxY <= web.minY + 1 }
        let bar = bars.max { $0.maxY < $1.maxY } ?? app.navigationBars.firstMatch.frame
        let bottomInset: CGFloat = window.height >= 800 ? 34 : 0
        let below = window.height - bar.maxY - bottomInset
        let share = web.height / below
        print("U11-SHARE|\(name)|window=\(Int(window.width))x\(Int(window.height))|bar=\(Int(bar.maxY))"
              + "|web=\(Int(web.minY))-\(Int(web.maxY))|B=\(Int(below))|share=\(String(format: "%.2f", share))")
        return share
    }

    /// The canvas's frame is `frame`, to the half point.
    private func assertTheCanvas(_ web: XCUIElement, isAt frame: CGRect, _ message: String,
                                 file: StaticString = #filePath, line: UInt = #line) {
        let now = web.frame
        for (name, a, b) in [("minY", now.minY, frame.minY), ("maxY", now.maxY, frame.maxY),
                             ("minX", now.minX, frame.minX), ("width", now.width, frame.width)] {
            XCTAssertEqual(a, b, accuracy: 0.5, "the canvas \(name) moved \(message): \(frame) -> \(now)",
                           file: file, line: line)
        }
    }

    private func openViewer(page: String, area: String? = nil, size: String? = nil, notice: String? = nil) {
        mock.setRoute("GET /worlds", status: 200, body: Page.listing(withArea: area != nil, notice: notice))
        // The pinned world's report carries the walk's notice too, as the
        // Tower's `lifecycle.finalization` does (the row's own record).
        socket.pinnedReply = { subscription, world, session in
            PanelReport.savedWorld(subscription: subscription, world: world, session: session, notice: notice)
        }
        mock.setRoute("GET /worlds/w1/render", status: 200, body: page)
        mock.setRoute("GET /worlds/w1/render/revision", status: 200, body: Page.revision(withArea: area != nil))
        if let area {
            mock.setRoute("GET /worlds/w1/areas/s1/\(Page.areaID)/render", status: 200, body: area)
            mock.setRoute("GET /worlds/w1/areas/s1/\(Page.areaID)/render/revision", status: 200,
                          body: Page.areaRevision)
        }
        launch(authority: mockAuthority, size: size)
        openTheWorld(named: "Appearance fixture (Mac)")
    }

    private func openRealWorld(size: String? = nil, expand: Bool = true) throws {
        guard let authority = ProcessInfo.processInfo.environment["GLASSES_UITEST_TOWER_AUTHORITY"],
              !authority.isEmpty
        else { throw XCTSkip("Set GLASSES_UITEST_TOWER_AUTHORITY=host:port of a U1.1 Tower with the fixture world.") }
        launch(authority: authority, size: size)
        openTheWorld(named: "Appearance fixture (Mac)", expand: expand)
    }

    private func launch(authority: String, size: String?) {
        app = XCUIApplication()
        app.launchArguments = ["-UITestSkipOnboarding", "-UITestResetTowerAddress"]
        if let size { app.launchArguments += ["-UIPreferredContentSizeCategoryName", size] }
        app.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = authority
        app.launch()
    }

    private func openTheWorld(named name: String, expand shouldExpand: Bool = true) {
        let cartridges = app.buttons["Cartridges"]
        XCTAssertTrue(cartridges.waitForExistence(timeout: 15), "Home")
        let drawerDone = app.buttons["Done"]
        XCTAssertTrue(tap(cartridges, until: drawerDone.exists), "the cartridge drawer opened")
        app.raiseTheCartridgeDrawerOnAnIPad()
        let row = app.buttons.containing(NSPredicate(format: "label BEGINSWITH %@", "World Builder")).firstMatch
        // An iPad's drawer is a centred sheet: `reveal` drags at the
        // screen's edge, outside it.
        XCTAssertTrue(UIDevice.current.userInterfaceIdiom == .pad ? app.revealInTheCartridgeDrawer(row) : reveal(row),
                      "the World Builder row")
        XCTAssertTrue(tap(row, until: !drawerDone.exists && app.navigationBars["World Builder"].exists),
                      "World Builder opened")
        let saved = app.buttons["Saved worlds"]
        XCTAssertTrue(reveal(saved), "Saved worlds")
        XCTAssertTrue(tap(saved, until: app.navigationBars["Saved worlds"].exists), "the Saved worlds sheet")
        let world = app.staticTexts[name].firstMatch
        XCTAssertTrue(world.waitForExistence(timeout: 20), "the listing")
        XCTAssertTrue(reveal(world), "the world's row")
        // U-INLINE §2.4 / §6.3: the row pins the world and the sheet goes;
        // the World Builder panel shows it, and Full screen opens the viewer.
        XCTAssertTrue(tap(world, until: !app.navigationBars["Saved worlds"].exists, attempts: 3), "the row pinned it")
        guard shouldExpand else { return }
        let expand = element("wb-panel-expand")
        XCTAssertTrue(expand.waitForExistence(timeout: 30), "the panel shows the pinned world")
        XCTAssertTrue(reveal(expand), "Full screen")
        let window = app.windows.firstMatch.frame
        XCTAssertTrue(tap(expand, until: self.app.webViews.firstMatch.exists
                            && self.app.webViews.firstMatch.frame.height > window.height * 0.4, attempts: 3),
                      "the viewer opened full screen")
        let drawing = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Drawing the world")).firstMatch
        _ = waitFor(timeout: 30) { !drawing.exists }
    }

    private func element(_ identifier: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    private func waitFor(timeout: TimeInterval = 10, _ condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            Thread.sleep(forTimeInterval: 0.2)
        }
        return condition()
    }

    @discardableResult
    private func tap(_ element: XCUIElement, until effect: @autoclosure () -> Bool, attempts: Int = 4) -> Bool {
        for _ in 0..<attempts {
            if element.waitForExistence(timeout: 5), element.isHittable { element.tap() }
            if waitFor(timeout: 5, effect) { return true }
        }
        return effect()
    }

    /// Slow drags in the right-hand margin, up then down, until `element`
    /// can be hit.
    @discardableResult
    /// iPhone-geometry tests skip on an iPad (the iPhone 17 Pro Simulator runs them).
    private func skipOnAnIPad(_ why: String) throws {
        if UIDevice.current.userInterfaceIdiom == .pad { throw XCTSkip("iPhone geometry: \(why)") }
    }

    private func reveal(_ element: XCUIElement, attempts: Int = 12) -> Bool {
        if element.waitForExistence(timeout: 5), element.isHittable { return true }
        for direction in [(0.85, 0.45), (0.45, 0.85)] {
            for _ in 0..<attempts {
                let start = app.coordinate(withNormalizedOffset: CGVector(dx: 0.97, dy: direction.0))
                let end = app.coordinate(withNormalizedOffset: CGVector(dx: 0.97, dy: direction.1))
                start.press(forDuration: 0.1, thenDragTo: end, withVelocity: .slow, thenHoldForDuration: 0.1)
                if element.exists && element.isHittable { return true }
            }
        }
        return element.exists && element.isHittable
    }

    /// A screenshot, kept, and written to U11_SHOTS_DIR when it is set.
    private func shoot(_ name: String) {
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        guard let dir = ProcessInfo.processInfo.environment["U11_SHOTS_DIR"], !dir.isEmpty else { return }
        let suffix = ProcessInfo.processInfo.environment["U11_SHOT_SUFFIX"].map { "-\($0)" } ?? ""
        try? screenshot.pngRepresentation.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name)\(suffix).png"))
    }
}
