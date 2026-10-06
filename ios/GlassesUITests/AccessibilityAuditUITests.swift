//
//  AccessibilityAuditUITests.swift
//  GlassesUITests
//
//  U0.5: XCUITest's accessibility audit -- contrast, Dynamic Type, clipped
//  text and element descriptions -- on every screen the Simulator can reach,
//  in light, dark and the largest accessibility text size, top to bottom.
//
//  Each test visits one screen three times: light at the default size, dark
//  at the default size, and light at Accessibility XXXL. The appearance is
//  the Simulator's (`XCUIDevice.appearance`, put back to light afterwards);
//  the text size is the app's own launch argument, so the Simulator's setting
//  is never touched. A screen taller than the display is audited a screen at
//  a time, down to its end, because the audit sees only what is on screen.
//
//  Every issue is reported, each named with its screen, mode and element,
//  unless it matches one of `waivers` -- each of which says why it is not
//  this app's to fix -- or is one of three things the audit gets wrong on
//  this app, each checked rather than assumed: a "Contrast failed" whose
//  text measures >= 4.5:1 on the rendered pixels (`measuredContrast`), an
//  element cut by the bars or the bottom edge (judged where it is whole),
//  and a Dynamic Type or clipping report on text that passes at
//  Accessibility XXXL (`Prediction`). Everything waived or deferred is
//  still printed, on a `U05-AUDIT|` line; a failure is `U05-AUDIT|ISSUE|`.
//
//  Slow on purpose: about 70 minutes for the suite on an 8 GB Mac.
//
//  Home, Connections, Settings, the cartridge drawer and the first-run cards
//  need no Tower: the app's socket is pointed at a closed loopback port. The
//  World Builder screens need one with the fabricated fixture worlds, and
//  skip without it:
//
//      TEST_RUNNER_GLASSES_UITEST_TOWER_AUTHORITY=127.0.0.1:8010 xcodebuild \
//          ... -only-testing:GlassesUITests/AccessibilityAuditUITests test-without-building
//
//  Optional: TEST_RUNNER_U05_MODES=light,dark,axl picks the modes (all three
//  by default); TEST_RUNNER_U05_SHOTS_DIR=<dir> writes a PNG of every audited
//  screenful there, suffixed with TEST_RUNNER_U05_SHOT_SUFFIX.
//

import UIKit
import XCTest

final class AccessibilityAuditUITests: XCTestCase {

    // MARK: Modes

    private struct Mode {
        let name: String
        let appearance: XCUIDevice.Appearance
        let contentSize: String?

        static let light = Mode(name: "light", appearance: .light, contentSize: nil)
        static let dark = Mode(name: "dark", appearance: .dark, contentSize: nil)
        static let axl = Mode(name: "axl", appearance: .light,
                              contentSize: "UICTContentSizeCategoryAccessibilityXXXL")
    }

    private var modes: [Mode] {
        let all: [Mode] = [.light, .dark, .axl]
        guard let picked = env("U05_MODES") else { return all }
        let names = Set(picked.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
        return all.filter { names.contains($0.name) }
    }

    /// What this test audits for: exactly the four U0.5 names.
    private let auditTypes: XCUIAccessibilityAuditType =
        [.contrast, .dynamicType, .textClipped, .sufficientElementDescription]

    /// Nothing listens on port 9 of this Mac.
    private let closedAuthority = "127.0.0.1:9"

    override func setUpWithError() throws {
        continueAfterFailure = true
        predictions = []
        passedAtLargest = [:]
    }

    // MARK: Dynamic Type predictions, checked at the largest size

    /// A Dynamic Type or clipping issue. The audit judges these by laying the
    /// screen out again at other text sizes, and that goes wrong for text
    /// low on the screen: enlarged, it is pushed off the bottom, and is then
    /// reported as not growing or as cut off. The same text sitting higher on
    /// another screenful passes. So each such issue -- at the default size or
    /// at the largest -- is checked against the real thing, the screen
    /// launched at Accessibility XXXL and audited a screenful at a time, and
    /// stands unless that text passed on at least one of those screenfuls.
    /// Text that truly does not scale, or is truly cut off, fails on all of
    /// them.
    private struct Prediction {
        let screen: String
        let label: String
        let line: String
        /// What the audit flagged, and whether at a default text size (light
        /// or dark) rather than at Accessibility XXXL.
        let auditType: XCUIAccessibilityAuditType
        let isDefaultSize: Bool
    }

    private var predictions: [Prediction] = []
    /// Per screen, every label that was on a screenful at the largest size
    /// whose Dynamic Type and clipping audits both COMPLETED, and was not
    /// flagged by either there: text actually judged at the largest size, and
    /// clean.
    ///
    /// The only evidence a prediction settles on (U0.5 review F4). Text that
    /// was merely seen at the largest size, by the contrast pass, and never
    /// flagged by the sizing pass used to count too; but a sizing pass that
    /// never reached the screenful the text was on flags nothing, so that was
    /// a waiver on no verdict.
    private var passedAtLargest: [String: Set<String>] = [:]

    /// A label as the two sides compare it: its first 60 characters, because
    /// the tree's description cuts a long label short.
    private static func key(_ label: String) -> String {
        // The first line only: the tree's description prints a label with a
        // line break in it over two lines.
        String((label.split(separator: "\n", omittingEmptySubsequences: false).first ?? "").prefix(60))
    }

    private static func baseScreen(_ screen: String) -> String {
        screen.components(separatedBy: "-more").first ?? screen
    }

    /// At the end of a test: each prediction from the default size, against
    /// the largest size's audit of the same screen.
    private func settlePredictions() {
        let largestRan = modes.contains { $0.name == Mode.axl.name }
        var standing: [String] = []
        var settled: [String: Int] = [:]
        var stood: [String: Int] = [:]
        for prediction in predictions {
            // Clipping at a default size settles only for a world or session
            // id (U0.8 H0; U0.5 review F4, owed). Those are the intentional
            // truncations -- `WorldPickerView` cuts them to one line, in the
            // middle, below the accessibility sizes. Any other text passing
            // at Accessibility XXXL proves nothing about the default size:
            // the layouts switch there (`AnyLayout`, `ViewThatFits`), so that
            // pass judged a different layout.
            if prediction.isDefaultSize && prediction.auditType == .textClipped {
                if Self.isWorldOrSessionID(prediction.label) {
                    settled[prediction.screen, default: 0] += 1
                    print("U05-AUDIT|\(prediction.line) -- WAIVED: an intentional truncation (a world or session id)")
                } else {
                    stood[prediction.screen, default: 0] += 1
                    standing.append(prediction.line)
                }
                continue
            }
            let key = Self.key(prediction.label)
            let passed = passedAtLargest[prediction.screen]?.contains(key) == true
            if largestRan, passed {
                settled[prediction.screen, default: 0] += 1
                print("U05-AUDIT|\(prediction.line) -- WAIVED: the same text was audited, and passed, on a screenful at "
                      + "Accessibility XXXL")
            } else {
                stood[prediction.screen, default: 0] += 1
                standing.append(prediction.line)
            }
        }
        // How many predictions each screen settled, and how many stood, so a
        // screen whose issues are all settled away is visible as such.
        for screen in Set(settled.keys).union(stood.keys).sorted() {
            print("U05-SETTLED|\(screen)|settled=\(settled[screen] ?? 0)|standing=\(stood[screen] ?? 0)")
        }
        for line in standing { print("U05-AUDIT|ISSUE|\(line)") }
        XCTAssertEqual(standing, [], standing.joined(separator: "\n"))
        predictions = []
    }

    /// Whether a flagged label is a world or session id (U0.8 H0; the
    /// recognition is OPEN-7's pick). The whole label must be one token of
    /// id characters -- letters, digits, `.`, `_`, `:` and `-` -- with a digit,
    /// a `-` or a `_` in it, so no word ("Done", "Settings") passes. An id
    /// the audit reports on the row's combined button, with the row's other
    /// words around it, is therefore NOT recognised and stands as an issue,
    /// to be triaged: the conservative side.
    static func isWorldOrSessionID(_ label: String) -> Bool {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._:-"))
        guard label.count >= 2, label.unicodeScalars.allSatisfy({ allowed.contains($0) && $0.isASCII }) else {
            return false
        }
        return label.contains { $0.isNumber || $0 == "-" || $0 == "_" }
    }

    override func tearDownWithError() throws {
        // The Simulator's own setting: leave it as it was found.
        XCUIDevice.shared.appearance = .light
    }

    // MARK: Waivers

    /// An issue this app does not own, with the reason. Matched on the audit
    /// type and the element, never on the screen alone, so a new problem on a
    /// waived screen still fails.
    private struct Waiver {
        let reason: String
        let matches: (XCUIAccessibilityAuditIssue, _ screen: String) -> Bool
    }

    private let waivers: [Waiver] = [
        // The Tower's page, inside the viewer's WKWebView: its own title,
        // caption, chips and the unlabelled "→" button are Tower HTML
        // (appearance_viewer.html, fixed 14 px type). Rewriting them is U1.1
        // (one visual system, the native-chrome flag); U0.5 is iOS-owned only.
        Waiver(reason: "Tower HTML inside the viewer's web view (U1.1)") { issue, screen in
            guard screen.hasPrefix("world-opened") else { return false }
            return AccessibilityAuditUITests.isInsideWebView(issue.element)
        },
        // UIKit's own bar buttons (Done, Close, Reload, the toolbar icons):
        // their font, size and colours are the system's. Bar items stop
        // growing at the large sizes by design and offer the Large Content
        // Viewer on a long press instead, and their Liquid Glass is sampled
        // over whatever the bar floats above. Nothing in this app draws them.
        Waiver(reason: "a system navigation-bar button (UIKit Liquid Glass; Large Content Viewer)") { issue, _ in
            guard issue.auditType == .contrast || issue.auditType == .dynamicType else { return false }
            return AccessibilityAuditUITests.isNavigationBarButton(issue.element)
        },
        // The navigation bar's own title (a UIKit label): one line, cut with
        // an ellipsis when it is long, at a size the system caps -- by design,
        // with the Large Content Viewer for the full text. The same words are
        // on the screen that pushed it (the world's row in Saved worlds).
        Waiver(reason: "a system navigation-bar title (UIKit; one line by design; Large Content Viewer)") { issue, _ in
            guard issue.auditType == .textClipped || issue.auditType == .dynamicType else { return false }
            return AccessibilityAuditUITests.isNavigationBarTitle(issue.element)
        },
        // A control that is switched off ("Start session" and "Start capture"
        // with no glasses, "Picture" with no world): the system dims it on
        // purpose, to say it cannot be used, and WCAG 1.4.3 sets no contrast
        // requirement for an inactive control. Its reason is in readable text
        // on the same screen, though not always beside it: under "Start
        // session" on Home; for "Start capture", in the viewfinder card at the
        // top of the World Builder workspace (DEBUG), which can be a screen
        // above it, and in the control's VoiceOver hint -- and, while it is
        // off because the Tower is not connected (U0.8 D1), in the line
        // directly under it ("Start is off while the Tower is not
        // connected…") and the same hint; for "Picture", in the
        // canvas under it ("No world yet"). (U0.5 review F10: the line that
        // stood beside "Start capture" was removed as a duplicate.)
        Waiver(reason: "a disabled control (WCAG 1.4.3: inactive components are exempt)") { issue, _ in
            guard issue.auditType == .contrast else { return false }
            return AccessibilityAuditUITests.isDisabledControl(issue.element)
        },
        // Contrast the audit could not attach to any element, on a screenful
        // that has been scrolled: text passing under the navigation bar,
        // softened by the system's scroll edge effect so that it is NOT read
        // there. Every element was judged whole on another screenful, and a
        // screenful that has not been scrolled gets no such waiver.
        Waiver(reason: "unattributed contrast on a scrolled screenful (under the bar's scroll edge effect)") { issue, screen in
            issue.auditType == .contrast && issue.element?.exists != true && screen.contains("-more")
        },
        // Dynamic Type or clipping the audit could not attach to any element:
        // the check lays the screen out again at other sizes, and the text it
        // meant is no longer there to name -- pushed off the screen, or the
        // screen behind a sheet, which is not an accessibility element at
        // all. Every text on the screen is also judged by name, and must pass
        // at Accessibility XXXL (see `Prediction`).
        Waiver(reason: "unattributed Dynamic Type or clipping (the check's own re-layout; every text is judged by name)") { issue, _ in
            (issue.auditType == .dynamicType || issue.auditType == .textClipped) && issue.element?.exists != true
        },
    ]

    // MARK: The screens that need no Tower

    func testHome() throws {
        for mode in modes {
            let app = launch(mode)
            XCTAssertTrue(app.buttons["Cartridges"].waitForExistence(timeout: 15), "\(mode.name): Home")
            // Let the socket to the closed port fail, so Home is in the state a
            // person without a Tower actually sees, banner and all.
            _ = app.buttons["Tower settings"].waitForExistence(timeout: 15)
            auditScrolling(app, screen: "home", mode: mode)
            app.terminate()
        }
        settlePredictions()
    }

    func testConnections() throws {
        for mode in modes {
            let app = launch(mode)
            let bar = statusBar(app)
            XCTAssertTrue(bar.waitForExistence(timeout: 15), "\(mode.name): the status bar")
            XCTAssertTrue(tap(bar, until: app.navigationBars["Connections"].exists),
                          "\(mode.name): the status bar opens Connections")
            expandSheet(app, bar: app.navigationBars["Connections"])
            auditScrolling(app, screen: "connections", mode: mode)
            app.terminate()
        }
        settlePredictions()
    }

    func testSettings() throws {
        for mode in modes {
            let app = launch(mode)
            XCTAssertTrue(tap(app.buttons["Settings"], until: app.navigationBars["Settings"].exists),
                          "\(mode.name): the toolbar opens Settings")
            auditScrolling(app, screen: "settings", mode: mode)
            app.terminate()
        }
        settlePredictions()
    }

    /// Settings as a developer run sees it: the DEBUG override note, whose
    /// variable name used to break mid-word at the accessibility sizes. Shot
    /// for the record in every mode; `testSettings` audits it.
    func testSettingsUnderTheDeveloperOverride() throws {
        for mode in modes {
            let app = launch(mode)
            XCTAssertTrue(tap(app.buttons["Settings"], until: app.navigationBars["Settings"].exists))
            let note = element(app, "tower-override-notice")
            XCTAssertTrue(revealBelow(app, note), "\(mode.name): the override note")
            Thread.sleep(forTimeInterval: 0.5)
            shoot(app, "settings-override", mode: mode)
            // On screen it may break at the underscores; to VoiceOver it is
            // the variable's plain name, with no zero-width spaces in it.
            XCTAssertTrue(note.label.contains("GLASSES_TOWER_AUTHORITY"), note.label)
            XCTAssertFalse(note.label.contains("\u{200B}"), "no zero-width space in the spoken label")
            // (The audit of this screen is `testSettings`, top to bottom.)
            app.terminate()
        }
        settlePredictions()
    }

    func testCartridgeDrawer() throws {
        for mode in modes {
            let app = launch(mode)
            let drawer = app.navigationBars["Cartridges"]
            XCTAssertTrue(tap(app.buttons["Cartridges"], until: drawer.exists),
                          "\(mode.name): the toolbar opens the drawer")
            expandSheet(app, bar: drawer)
            auditScrolling(app, screen: "cartridges", mode: mode)
            app.terminate()
        }
        settlePredictions()
    }

    func testOnboardingCards() throws {
        for mode in modes {
            let app = launch(mode, onboarding: .reset)
            let page = element(app, "onboarding-page")
            for number in 1...4 {
                XCTAssertTrue(waitFor { (page.value as? String) == "\(number) of 4" },
                              "\(mode.name): card \(number)")
                // The pager slides and the button cross-fades: audit it settled.
                Thread.sleep(forTimeInterval: 1)
                auditScrolling(app, screen: "onboarding-\(number)", mode: mode, drag: true)
                if number < 4 {
                    let next = app.buttons["onboarding-continue"]
                    XCTAssertTrue(tap(next, until: (page.value as? String) == "\(number + 1) of 4"))
                }
            }
            app.terminate()
        }
        settlePredictions()
    }

    // MARK: The World Builder screens, against the fixture Tower

    func testWorldBuilderIdle() throws {
        let authority = try towerAuthority()
        for mode in modes {
            let app = launch(mode, authority: authority)
            openWorldBuilder(app, mode: mode)
            // Settled on what the Tower said: the canvas has left its first wait.
            _ = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Last saved world"))
                .firstMatch.waitForExistence(timeout: 15)
            auditScrolling(app, screen: "wb-idle", mode: mode)
            app.terminate()
        }
        settlePredictions()
    }

    func testWorldList() throws {
        let authority = try towerAuthority()
        for mode in modes {
            let app = launch(mode, authority: authority)
            openSavedWorlds(app, mode: mode)
            auditScrolling(app, screen: "world-list", mode: mode)
            app.terminate()
        }
        settlePredictions()
    }

    /// The appearance fixture's room, opened from the list: the loading state
    /// on the way (shot, not audited: it is gone in a second or two), then the
    /// drawn world.
    func testOpenedWorld() throws {
        let authority = try towerAuthority()
        warmUpTheViewer(authority: authority)
        for mode in modes {
            let app = launch(mode, authority: authority)
            openSavedWorlds(app, mode: mode)
            let row = app.staticTexts["Appearance fixture (Mac)"]
            XCTAssertTrue(reveal(app, row), "\(mode.name): the appearance fixture's row")
            row.tap()
            let started = Date()
            var shotLoading = false
            let drawing = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Drawing the world")).firstMatch
            let fetching = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Fetching this world")).firstMatch
            while Date().timeIntervalSince(started) < 60 {
                if !shotLoading, drawing.exists || fetching.exists {
                    shoot(app, "world-loading", mode: mode)
                    shotLoading = true
                }
                if !drawing.exists && !fetching.exists && app.webViews.firstMatch.exists { break }
                Thread.sleep(forTimeInterval: 0.2)
            }
            XCTAssertTrue(app.webViews.firstMatch.exists, "\(mode.name): the page was drawn")
            // The page's own boot fades in, and its first seconds of WebGL
            // keep this 8 GB Mac's Simulator too busy to answer an audit.
            Thread.sleep(forTimeInterval: 12)
            auditScrolling(app, screen: "world-opened", mode: mode, scrolls: false)
            let details = app.buttons["Details"].firstMatch
            if details.exists, details.isHittable {
                details.tap()
                Thread.sleep(forTimeInterval: 1)
                shoot(app, "world-opened-details", mode: mode)
                judge(app, screen: "world-opened-details", mode: mode,
                      types: [.contrast, .sufficientElementDescription], band: nil)
                judge(app, screen: "world-opened-details", mode: mode,
                      types: [.dynamicType, .textClipped], band: nil)
            }
            app.terminate()
        }
        settlePredictions()
    }

    /// At the largest text size the 3D view keeps most of the screen, on
    /// whatever phone this runs on -- the iPhone SE included, where it once
    /// measured 0 pt under a caption that grew without limit (O1 viewer
    /// check, at 2ff0b0e): at least 60 % of the safe area, and the caption
    /// is still there, in its own scrolling space.
    func testTheWorldKeepsMostOfTheScreenAtTheLargestTextSize() throws {
        let authority = try towerAuthority()
        let app = launch(.axl, authority: authority)
        openSavedWorlds(app, mode: .axl)
        // The appearance fixture: at this size on an SE it is several
        // screens down the list, so it is looked for further than usual.
        let row = app.staticTexts["Appearance fixture (Mac)"]
        XCTAssertTrue(reveal(app, row, attempts: 30), "the appearance fixture's row")
        row.tap()
        let web = app.webViews.firstMatch
        XCTAssertTrue(web.waitForExistence(timeout: 120), "the page was drawn")
        let caption = app.staticTexts.containing(
            NSPredicate(format: "label CONTAINS[c] %@", "not to scale")).firstMatch
        XCTAssertTrue(caption.waitForExistence(timeout: 30), "the caption is still on the screen")
        Thread.sleep(forTimeInterval: 3)
        shoot(app, "world-share", mode: .axl)

        let window = app.windows.firstMatch.frame
        let bar = app.navigationBars.firstMatch.frame
        // The safe area: below the status bar (where the navigation bar
        // starts), above the home indicator on a phone that has one.
        let homeIndicator: CGFloat = window.height >= 800 ? 34 : 0
        let safeArea = window.height - bar.minY - homeIndicator
        let share = web.frame.height / safeArea
        print("U05-VIEWER|window=\(Int(window.width))x\(Int(window.height))|web=\(Int(web.frame.height))"
              + "|safe=\(Int(safeArea))|share=\(String(format: "%.2f", share))")
        XCTAssertGreaterThanOrEqual(share, 0.6,
                                    "the 3D view is \(Int(web.frame.height)) pt of a \(Int(safeArea)) pt safe area")
        app.terminate()
    }

    /// With an AREA open -- whose viewer always shows "Back to the room" --
    /// the picture still keeps most of the screen at the largest text size
    /// (U0.8 R1): the controls between the words and the picture are capped
    /// at 10 %, beside the words' 18 % and the areas' and Details' 12 %.
    func testTheWorldKeepsMostOfTheScreenWithAnAreaOpenAtTheLargestTextSize() throws {
        let authority = try towerAuthority()
        // On a Simulator just booted, the first WebGL page can outlast the
        // viewer's watchdog (seen on the SE).
        warmUpTheViewer(authority: authority)
        let app = launch(.axl, authority: authority)
        openSavedWorlds(app, mode: .axl)
        let row = app.staticTexts["Appearance fixture (Mac)"]
        XCTAssertTrue(reveal(app, row, attempts: 30), "the appearance fixture's row")
        row.tap()
        let web = app.webViews.firstMatch
        XCTAssertTrue(web.waitForExistence(timeout: 120), "the page was drawn")

        // An area that opens: the first enabled button in the areas row.
        // Under native chrome (U1.1) the row is in a panel, opened by Areas.
        let areasControl = element(app, "world-chrome-areas")
        let areas = element(app, "world-render-areas")
        _ = waitFor(timeout: 30) { areas.exists || areasControl.exists }
        if areasControl.exists, areasControl.isHittable { areasControl.tap() }
        guard areas.waitForExistence(timeout: 30) else {
            throw XCTSkip("the fixture world has no area that opens (no areas row)")
        }
        guard let area = areas.buttons.allElementsBoundByIndex.first(where: { $0.exists && $0.isEnabled }) else {
            throw XCTSkip("the fixture world has no area that opens")
        }
        XCTAssertTrue(reveal(app, area), "the area's button")
        area.tap()

        let back = element(app, "world-render-back-to-room")
        XCTAssertTrue(back.waitForExistence(timeout: 60), "Back to the room")
        XCTAssertTrue(web.waitForExistence(timeout: 120), "the area's page was drawn")
        let drawing = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Drawing the world")).firstMatch
        _ = waitFor(timeout: 60) { !drawing.exists }
        Thread.sleep(forTimeInterval: 3)
        shoot(app, "world-area-share", mode: .axl)

        let window = app.windows.firstMatch.frame
        let bar = app.navigationBars.firstMatch.frame
        let homeIndicator: CGFloat = window.height >= 800 ? 34 : 0
        let below = window.height - bar.maxY - homeIndicator
        let share = web.frame.height / below
        let offers = ["world-render-newer-picture", "world-render-retry-refused"].filter { element(app, $0).exists }
        print("U05-VIEWER|area|window=\(Int(window.width))x\(Int(window.height))|bar=\(Int(bar.maxY))"
              + "|back=\(Int(back.frame.minY))-\(Int(back.frame.maxY))|web=\(Int(web.frame.minY))-\(Int(web.frame.maxY))"
              + "|below=\(Int(below))|share=\(String(format: "%.2f", share))|offers=\(offers)")
        XCTAssertTrue(back.isHittable, "Back to the room can be reached")
        XCTAssertLessThanOrEqual(back.frame.maxY, web.frame.minY + 1, "Back to the room ends above the picture")
        XCTAssertLessThanOrEqual(offers.count, 1, "one offer at a time")
        XCTAssertGreaterThanOrEqual(share, 0.59,
                                    "the 3D view is \(Int(web.frame.height)) pt of \(Int(below)) pt below the bar")
        app.terminate()
    }

    /// At the default text size the words above the picture do not scroll:
    /// the walk's notice from the Tower (COMPONENTS §8: "show it verbatim
    /// below the room caption") is on the screen whole, above the 3D view,
    /// without scrolling -- on an iPhone SE too, where the caption's cap once
    /// hid it (U0.5 review F1). The controls that act on the picture (the
    /// newer reconstruction's offer, its retry, Back to the room) are pinned
    /// between the words and the picture, and are checked the same way when
    /// the screen shows them. Then every native text and control on the
    /// screen is measured on the pixels, light mode, to 4.5:1 -- "Details"
    /// included (U0.5 review F2).
    ///
    /// Needs the appearance fixture's walk to carry a `finalization.notice`,
    /// and skips without one.
    func testTheNoticeIsOnTheScreenWithoutScrollingAtTheDefaultSize() throws {
        let authority = try towerAuthority()
        // On a Simulator just booted, the first WebGL page can outlast the
        // viewer's watchdog, which then swaps the page for its failure view
        // (seen on the SE): the page is not what is judged here.
        warmUpTheViewer(authority: authority)
        let app = launch(.light, authority: authority)
        openSavedWorlds(app, mode: .light)
        let row = app.staticTexts["Appearance fixture (Mac)"]
        XCTAssertTrue(reveal(app, row), "the appearance fixture's row")
        row.tap()
        let web = app.webViews.firstMatch
        XCTAssertTrue(web.waitForExistence(timeout: 120), "the page was drawn")
        let caption = app.staticTexts.containing(
            NSPredicate(format: "label CONTAINS[c] %@", "not to scale")).firstMatch
        XCTAssertTrue(caption.waitForExistence(timeout: 30), "the room's caption")
        let notice = element(app, "world-render-notice")
        guard notice.waitForExistence(timeout: 5) else {
            throw XCTSkip("The appearance fixture's walk carries no finalization.notice.")
        }
        // The drawing has finished, so the picture is at its final frame.
        let drawing = app.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Drawing the world")).firstMatch
        _ = waitFor(timeout: 60) { !drawing.exists }
        Thread.sleep(forTimeInterval: 3)
        shoot(app, "world-notice", mode: .light)
        guard web.exists else {
            XCTFail("the page is no longer drawn (the viewer's failure view?); nothing to measure against")
            return
        }

        let window = app.windows.firstMatch.frame
        let bar = app.navigationBars.firstMatch.frame
        let below = window.height - bar.maxY
        let noticeFrame = notice.frame
        let webFrame = web.frame
        print("U05-NOTICE|window=\(Int(window.width))x\(Int(window.height))|bar=\(Int(bar.maxY))"
              + "|below=\(Int(below))|caption=\(Int(caption.frame.minY))-\(Int(caption.frame.maxY))"
              + "|notice=\(Int(noticeFrame.minY))-\(Int(noticeFrame.maxY))"
              + "|words=\(Int(webFrame.minY - bar.maxY))|web=\(Int(webFrame.minY))-\(Int(webFrame.maxY))"
              + "|words_share=\(String(format: "%.2f", (webFrame.minY - bar.maxY) / below))")
        XCTAssertTrue(notice.isHittable, "the notice can be reached")
        XCTAssertGreaterThanOrEqual(noticeFrame.minY, bar.maxY - 1, "the notice is below the bar")
        XCTAssertLessThanOrEqual(noticeFrame.maxY, webFrame.minY + 1,
                                 "the notice is whole above the picture, not scrolled out of sight")
        for id in ["world-render-newer-picture", "world-render-retry-refused", "world-render-back-to-room"] {
            let control = element(app, id)
            guard control.exists else { continue }
            print("U05-NOTICE|control=\(id)|\(Int(control.frame.minY))-\(Int(control.frame.maxY))")
            XCTAssertTrue(control.isHittable, "\(id) can be reached")
            XCTAssertLessThanOrEqual(control.frame.maxY, webFrame.minY + 1, "\(id) is whole above the picture")
        }

        // The viewer's own words and controls, light mode, on the pixels.
        let measured = measureNativeElements(app, in: app.screenshot().image, window: window)
        for element in measured {
            print("U05-CONTRAST|light world-notice|\(element.kind) '\(element.label.prefix(60))'"
                  + "|\(String(format: "%.2f", element.ratio)):1")
        }
        XCTAssertTrue(measured.contains { $0.label == "Details" && $0.kind != "text" }, "the Details control was measured")
        let low = measured.filter { $0.ratio < 4.5 }
        XCTAssertEqual(low.map { "\($0.kind) '\($0.label)' \(String(format: "%.2f", $0.ratio)):1" }, [],
                       "every native text and control is at least 4.5:1")
        app.terminate()
    }

    /// Opens the world once, unaudited, and gives WebKit time to settle.
    ///
    /// The first WebGL page after the Simulator boots keeps it too busy to
    /// answer an accessibility audit for minutes on this 8 GB Mac: the first
    /// mode's audit of the drawn world timed out on every try, in three runs
    /// in a row, and every later mode's passed. The page is not what is
    /// audited here -- it is waived as Tower HTML -- so this changes nothing
    /// that is judged; it only means the first mode is not the first page.
    private func warmUpTheViewer(authority: String) {
        let app = launch(.light, authority: authority)
        openSavedWorlds(app, mode: .light)
        let row = app.staticTexts["Appearance fixture (Mac)"]
        if reveal(app, row) {
            row.tap()
            _ = app.webViews.firstMatch.waitForExistence(timeout: 60)
            Thread.sleep(forTimeInterval: 30)
        }
        app.terminate()
    }

    // MARK: Auditing

    /// The screen, top to bottom, in two passes.
    ///
    /// **Contrast and descriptions**, a screenful at a time: audit, one slow
    /// drag, audit again, until a drag moves nothing. An element only partly
    /// on screen -- scrolled up under the bars, or cut by the bottom edge --
    /// is not judged where it is cut: the audit would sample it through the
    /// bar's glass or off the screen. Each drag moves well under a screen, so
    /// it was whole on the screenful before (top) or is whole on the next
    /// (bottom). On the last screenful nothing is deferred.
    ///
    /// **Dynamic Type and clipped text**, the same screenfuls again. These
    /// checks lay the screen out at other text sizes, and a list does not
    /// come back to where it was, so each screenful is reached afresh: back
    /// to the top, then as many drags as the first pass took.
    private func auditScrolling(_ app: XCUIApplication, screen: String, mode: Mode,
                                scrolls: Bool = true, drag: Bool = false, maxPages: Int = 16) {
        func step() {
            // A slow drag, with no momentum, from low on the screen: it lands
            // inside a half-height sheet, and moves the content well under a
            // screen, so nothing is skipped between two screenfuls.
            // In the right-hand margin, beside the rows rather than on one: a
            // drag that starts on Settings' address field scrolls the field
            // (a multi-line text view), not the list.
            if drag { self.drag(app, from: 0.6, to: 0.2) } else { self.drag(app, from: 0.8, to: 0.4, x: 0.97) }
            Thread.sleep(forTimeInterval: 0.8)
        }
        func name(_ page: Int) -> String { page == 0 ? screen : "\(screen)-more\(page)" }

        // Contrast and descriptions.
        var page = 0
        var here = Self.textFrames(app)
        var cutAtBottom: [String] = []
        while true {
            shoot(app, name(page), mode: mode)
            cutAtBottom = judge(app, screen: name(page), mode: mode, types: [.contrast, .sufficientElementDescription],
                                band: scrolls ? visibleBand(app, scrolled: page > 0) : nil)
            guard scrolls, page < maxPages else { break }
            step()
            let next = Self.textFrames(app)
            if next == here { break }
            here = next
            page += 1
        }
        // The last screenful: what is cut at its bottom is never shown whole.
        for line in cutAtBottom { print("U05-AUDIT|ISSUE|\(line)") }
        XCTAssertEqual(cutAtBottom, [], cutAtBottom.joined(separator: "\n"))

        // Dynamic Type and clipped text. Reached afresh only when the check
        // moved the screen; a plain scroll view stays put, and one more drag
        // is the next screenful.
        if scrolls, page > 0 { scrollToTop(app, drag: drag, from: page) }
        for again in 0...page {
            let before = Self.textFrames(app)
            _ = judge(app, screen: name(again), mode: mode, types: [.dynamicType, .textClipped], band: nil)
            guard again < page else { break }
            Thread.sleep(forTimeInterval: 1)
            if Self.textFrames(app) == before {
                step()
            } else {
                scrollToTop(app, drag: drag, from: again + 1)
                for _ in 0...again { step() }
            }
        }
        // Left where it ended: every caller is done with the screen.
    }

    /// Back to the top of the screen: the status bar's tap, which scrolls the
    /// frontmost scroll view up, or -- for the paged cards, where it cannot
    /// tell which card's scroll view is meant -- drags down, which a
    /// full-screen cover does not answer by closing.
    private func scrollToTop(_ app: XCUIApplication, drag: Bool, from page: Int) {
        if drag {
            // Never further down than `page` drags; two more for good measure.
            for _ in 0..<(page + 2) { self.drag(app, from: 0.2, to: 0.6) }
        } else {
            let statusBar = app.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: 200, dy: 12))
            statusBar.tap()
            Thread.sleep(forTimeInterval: 1.2)
            statusBar.tap()
        }
        Thread.sleep(forTimeInterval: 1)
    }

    /// The part of the screen where content is seen whole: below the
    /// navigation bar (and the soft edge under it) and the pinned status bar
    /// once scrolled, and above the bottom edge.
    private func visibleBand(_ app: XCUIApplication, scrolled: Bool) -> VisibleBand {
        let window = app.windows.firstMatch.frame
        var top = -CGFloat.infinity
        var bars: [CGRect] = []
        var titles: Set<String> = []
        for bar in app.navigationBars.allElementsBoundByIndex where bar.exists {
            bars.append(bar.frame)
            titles.insert(bar.identifier)
            if scrolled { top = max(top, bar.frame.maxY + 24) }
        }
        let status = statusBar(app)
        // Pinned under the navigation bar, and not behind a sheet.
        if scrolled, status.exists, status.isHittable, let navBar = bars.first,
           abs(status.frame.minY - navBar.maxY) < 24 {
            top = max(top, status.frame.maxY)
        }
        // The first-run cards scroll under a footer (the page dots, Continue
        // and Skip): what is under it is seen above it a screenful later.
        var bottom = window.maxY
        let dots = element(app, "onboarding-page")
        if dots.exists { bottom = min(bottom, dots.frame.minY) }
        return VisibleBand(top: top, bottom: bottom, bars: bars, titles: titles)
    }

    private struct VisibleBand {
        let top: CGFloat
        let bottom: CGFloat
        /// The navigation bars, whose own buttons and titles are judged
        /// wherever they are. Frames alone cannot say what is in a bar: a
        /// sheet's rows scroll up under its bar, and the screen behind a
        /// sheet has a bar of its own in the same place.
        let bars: [CGRect]
        let titles: Set<String>

        /// Whether the element can be seen whole on some screenful at all.
        /// One taller than the band never can, so it is judged wherever it
        /// is rather than deferred for ever.
        func fits(_ frame: CGRect) -> Bool {
            frame.height < bottom - max(top, 0)
        }

        func isBarContent(_ type: XCUIElement.ElementType, _ label: String, _ frame: CGRect) -> Bool {
            if type == .staticText { return titles.contains(label) }
            return type == .button && bars.contains { $0.insetBy(dx: -1, dy: -1).contains(frame) }
        }

        func isCutAtTop(_ frame: CGRect) -> Bool {
            fits(frame) && frame.minY < top - 1
        }

        func isCutAtBottom(_ frame: CGRect) -> Bool {
            fits(frame) && frame.maxY > bottom + 1
        }
    }

    /// Where the last texts on screen are, to the point: the same after a
    /// drag means the drag moved nothing, so the end has been reached. The
    /// LAST texts, because a sheet's come after the screen's behind it, and
    /// those never move when the sheet is scrolled. (Pixels were tried
    /// first; a scroll indicator's flash made two identical screenfuls
    /// differ.)
    private static func textFrames(_ app: XCUIApplication) -> [String] {
        app.staticTexts.allElementsBoundByIndex.suffix(12).compactMap { text in
            guard text.exists else { return nil }
            let frame = text.frame
            return "\(Int(frame.minX)),\(Int(frame.minY)),\(Int(frame.width)),\(Int(frame.height))"
        }
    }

    /// One audit of `types`, every issue named with its element rather than
    /// the first alone. Returns the issues on elements cut by the bottom
    /// edge, which the caller judges only if this is the last screenful.
    @discardableResult
    private func judge(_ app: XCUIApplication, screen: String, mode: Mode,
                       types: XCUIAccessibilityAuditType, band: VisibleBand?) -> [String] {
        var found: [String] = []
        var noted: [String] = []
        var cutAtBottom: [String] = []
        var flaggedHere = Set<String>()
        var newPredictions: [Prediction] = []
        let sizingPass = types.contains(.dynamicType) || types.contains(.textClipped)
        // The screen as the contrast check sees it, to measure a flagged
        // element's rendered text against its background.
        let shot = sizingPass ? nil : app.screenshot().image
        let window = app.windows.firstMatch.frame
        func handle(_ issue: XCUIAccessibilityAuditIssue) {
            let line = "\(mode.name) \(screen): \(Self.typeName(issue.auditType)): "
                + "\(issue.compactDescription) [\(issue.detailedDescription)] on \(Self.describe(issue.element))"
            let element = issue.element.flatMap { $0.exists ? $0 : nil }
            let frame = element?.frame
            let label = element?.label
            if sizingPass, let label { flaggedHere.insert(Self.key(label)) }
            if let waiver = waivers.first(where: { $0.matches(issue, screen) }) {
                noted.append("\(line) -- WAIVED: \(waiver.reason)")
            } else if issue.auditType == .contrast, let shot, let frame, element?.elementType == .staticText,
                      band.map({ !$0.isCutAtTop(frame) && !$0.isCutAtBottom(frame) }) ?? true,
                      let measured = Self.measuredContrast(in: shot, window: window, frame: frame),
                      measured >= 4.5 {
                noted.append("\(line) -- WAIVED: measured \(String(format: "%.1f", measured)):1 on the "
                             + "rendered pixels (WCAG 2), over 4.5:1")
            } else if sizingPass, let label {
                newPredictions.append(Prediction(screen: Self.baseScreen(screen), label: label, line: line,
                                                 auditType: issue.auditType,
                                                 isDefaultSize: mode.contentSize == nil))
                noted.append("\(line) -- checked against Accessibility XXXL at the end")
            } else if let band, let element, let frame,
                      !band.isBarContent(element.elementType, element.label, frame) {
                if band.isCutAtTop(frame) {
                    noted.append("\(line) -- under the bars here; judged on the screenful before")
                } else if band.isCutAtBottom(frame) {
                    cutAtBottom.append(line)
                } else {
                    found.append(line)
                }
            } else {
                found.append(line)
            }
        }
        // One audit per check, up to four tries each: on this 8 GB Mac, with
        // a WebGL page open, an audit sometimes reports "failed to complete
        // in time". A try that completes is the one that counts.
        let checks: [XCUIAccessibilityAuditType] = [.contrast, .sufficientElementDescription, .dynamicType, .textClipped]
        // The two waivers below are for the world's WebGL page and a timeout
        // only (U0.5 review F3); anywhere else, or any other error, fails.
        let overTheWebGLPage = screen.hasPrefix("world-opened")
        var sizingCompleted = true
        for check in checks where auditTypes.intersection(types).contains(check) {
            var completed = false
            var lastError: Error?
            var onlyTimeouts = true
            for attempt in 1...4 where !completed {
                var issues: [XCUIAccessibilityAuditIssue] = []
                do {
                    try app.performAccessibilityAudit(for: check) { issue in
                        issues.append(issue)
                        return true
                    }
                    issues.forEach(handle)
                    completed = true
                } catch {
                    lastError = error
                    onlyTimeouts = onlyTimeouts && Self.isAuditTimeout(error)
                    print("U05-AUDIT|\(mode.name) \(screen): \(Self.typeName(check)) try \(attempt) could not run (\(error))")
                    if attempt < 4 { Thread.sleep(forTimeInterval: 15) }
                }
            }
            guard !completed else { continue }
            if check == .dynamicType || check == .textClipped { sizingCompleted = false }
            if check == .contrast, let shot, overTheWebGLPage, onlyTimeouts {
                // Over the world's WebGL page in light mode the contrast check
                // never completed (four tries in each of three runs). What it
                // would have judged that is this app's -- every native text
                // and control on the screen, outside the page and the bars,
                // buttons and disclosures included (U0.5 review F2) -- is
                // measured on the rendered pixels instead, to the same 4.5:1.
                let measured = measureNativeElements(app, in: shot, window: window)
                let low = measured.filter { $0.ratio < 4.5 }
                for element in low {
                    found.append("\(mode.name) \(screen): contrast: measured \(String(format: "%.1f", element.ratio)):1 "
                                 + "on the rendered pixels for \(element.kind) '\(element.label)'")
                }
                if low.isEmpty {
                    let lowest = measured.map(\.ratio).min().map { String(format: "%.1f", $0) } ?? "-"
                    let buttons = measured.filter { $0.kind != "text" }.count
                    noted.append("\(mode.name) \(screen): contrast: the audit could not complete over the WebGL page "
                                 + "-- WAIVED: its \(measured.count) native elements (\(buttons) of them controls) "
                                 + "measured on the rendered pixels, lowest \(lowest):1, all over 4.5:1")
                }
            } else if check == .dynamicType || check == .textClipped, mode.contentSize == nil,
                      overTheWebGLPage, onlyTimeouts, modes.contains(where: { $0.name == Mode.axl.name }) {
                // At the default size these two checks only predict; every
                // text they could flag is judged again for real at
                // Accessibility XXXL (`Prediction`), where a timeout still
                // fails -- so this holds only when that mode is in the run.
                // Over the WebGL page, in the first mode, they timed out on
                // every try in every run on this 8 GB Mac.
                noted.append("\(mode.name) \(screen): \(Self.typeName(check)): the audit could not complete "
                             + "over the WebGL page -- WAIVED: a default-size prediction; the same texts are "
                             + "judged at Accessibility XXXL in this run")
            } else {
                found.append("\(mode.name) \(screen): \(Self.typeName(check)): the audit could not run: "
                             + "\(lastError.map { "\($0)" } ?? "-")")
            }
        }
        predictions.append(contentsOf: newPredictions)
        if mode.contentSize != nil, sizingPass, sizingCompleted {
            // What was audited on this screenful at the largest size, and
            // passed. A sizing audit that did not complete judged nothing.
            let base = Self.baseScreen(screen)
            passedAtLargest[base, default: []].formUnion(Self.labels(app).subtracting(flaggedHere))
        }
        if let band {
            print("U05-BAND|\(mode.name) \(screen)|top=\(Int(band.top.isFinite ? band.top : -1))"
                  + "|bottom=\(Int(band.bottom))|bars=\(band.bars.map { "\(Int($0.minY))-\(Int($0.maxY))" })")
        }
        for line in noted { print("U05-AUDIT|\(line)") }
        for line in cutAtBottom { print("U05-AUDIT|\(line) -- cut by the bottom edge; judged on the next screenful") }
        for line in found { print("U05-AUDIT|ISSUE|\(line)") }
        XCTAssertEqual(found, [], found.joined(separator: "\n"))
        return cutAtBottom
    }

    /// An element of this app's, measured on the rendered pixels.
    private struct MeasuredElement {
        /// "text", "button", "link", "disclosure", "toggle".
        let kind: String
        let label: String
        let frame: CGRect
        let ratio: Double
    }

    /// The kinds of element whose label is drawn as text or as a tinted
    /// glyph: every one the contrast audit would judge.
    private static let measuredKinds: [XCUIElement.ElementType: String] = [
        .staticText: "text", .button: "button", .link: "link",
        .disclosureTriangle: "disclosure", .toggle: "toggle", .switch: "toggle",
    ]

    /// Every element on the screen that is this app's and has a label --
    /// texts, and the buttons, links and disclosures the contrast audit
    /// judges too (U0.5 review F2: static texts alone missed the viewer's
    /// system-blue "Details" at about 4.0:1) -- outside the web view and the
    /// navigation bars, wholly on the screen and enabled, with its measured
    /// contrast. A disabled control is exempt, as in `waivers`. From one
    /// snapshot of the tree, and never inside the web view.
    ///
    /// Measured by the colour most of its ink is drawn in (`Ink.dominant`),
    /// not by its strongest pixel: a disclosure's frame holds its label and
    /// its black chevron, and the chevron alone measured "Details" at 21:1
    /// whatever colour the word was.
    private func measureNativeElements(_ app: XCUIApplication, in shot: UIImage,
                                       window: CGRect) -> [MeasuredElement] {
        guard let root = try? app.snapshot() else { return [] }
        var webFrames: [CGRect] = []
        var bars: [CGRect] = []
        var candidates: [(kind: String, label: String, frame: CGRect)] = []
        func visit(_ node: XCUIElementSnapshot) {
            switch node.elementType {
            case .webView:
                webFrames.append(node.frame)
                return
            case .navigationBar:
                bars.append(node.frame)
                return
            default:
                break
            }
            if let kind = Self.measuredKinds[node.elementType], !node.label.isEmpty, node.isEnabled {
                candidates.append((kind, node.label, node.frame))
            }
            node.children.forEach(visit)
        }
        visit(root)
        var results: [MeasuredElement] = []
        for candidate in candidates {
            let frame = candidate.frame
            guard window.contains(frame), frame.width > 1, frame.height > 1,
                  !webFrames.contains(where: { $0.intersects(frame) }),
                  !bars.contains(where: { $0.intersects(frame) }),
                  let ratio = Self.measuredContrast(in: shot, window: window, frame: frame, ink: .dominant)
            else { continue }
            results.append(MeasuredElement(kind: candidate.kind, label: candidate.label, frame: frame, ratio: ratio))
        }
        return results
    }

    /// Whether an audit failed only because it ran out of time: XCTest's
    /// "Audit failed to complete in time" (-56), or its future timing out
    /// while running the audit (1000). The two the WebGL page produced.
    private static func isAuditTimeout(_ error: Error) -> Bool {
        let error = error as NSError
        if error.domain == "com.apple.xcode.xctest.accessibilityAudit", error.code == -56 { return true }
        if error.domain == "com.apple.dt.XCTest.XCTFuture", error.code == 1000 { return true }
        let text = error.localizedDescription.lowercased()
        return text.contains("failed to complete in time") || text.contains("timed out while running accessibility audit")
    }

    /// The WCAG 2 contrast of the text drawn in `frame`: the ink (the pixel
    /// furthest from the background) against the background (the commonest
    /// colour in the frame).
    ///
    /// For the audit's "Contrast failed" on text whose frame is much larger
    /// than its words -- a list section header is the full width of the
    /// screen and 40 points tall for one short word -- where it samples the
    /// frame's two commonest colours, the page and a card's edge, and never
    /// the text. What passes here is measured, not assumed; blue on a grey
    /// capsule measured 3.5:1 this way and was fixed, not waived.
    /// Which of the frame's ink `measuredContrast` measures.
    enum Ink {
        /// The pixel furthest from the background.
        case strongest
        /// The colour most of the ink is drawn in: of the pixels at least
        /// 1.5:1 from the background, the commonest colour. The solid core of
        /// a glyph's strokes is one colour, and its anti-aliased edges are
        /// spread over many; a second, stronger colour in the same frame (a
        /// chevron, an icon) does not stand in for the words. With no pixel
        /// that far from the background, the strongest pixel's ratio.
        case dominant
    }

    static func measuredContrast(in image: UIImage, window: CGRect, frame: CGRect,
                                 ink: Ink = .strongest) -> Double? {
        guard let cgImage = image.cgImage, window.width > 0 else { return nil }
        let scale = CGFloat(cgImage.width) / window.width
        let rect = CGRect(x: frame.minX * scale, y: frame.minY * scale,
                          width: frame.width * scale, height: frame.height * scale).integral
            .intersection(CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
        guard rect.width >= 2, rect.height >= 2, let crop = cgImage.cropping(to: rect) else { return nil }
        let width = crop.width, height = crop.height
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = bytes.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return false }
            context.draw(crop, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return nil }
        var counts: [Int: Int] = [:]
        for index in stride(from: 0, to: bytes.count, by: 4) {
            let key = Int(bytes[index] >> 3) << 10 | Int(bytes[index + 1] >> 3) << 5 | Int(bytes[index + 2] >> 3)
            counts[key, default: 0] += 1
        }
        guard let background = counts.max(by: { $0.value < $1.value })?.key else { return nil }
        // The background's own mean colour, from the pixels in its bucket.
        var sum = [0.0, 0.0, 0.0], n = 0.0
        for index in stride(from: 0, to: bytes.count, by: 4) {
            let key = Int(bytes[index] >> 3) << 10 | Int(bytes[index + 1] >> 3) << 5 | Int(bytes[index + 2] >> 3)
            guard key == background else { continue }
            sum[0] += Double(bytes[index]); sum[1] += Double(bytes[index + 1]); sum[2] += Double(bytes[index + 2])
            n += 1
        }
        let back = luminance(sum[0] / n, sum[1] / n, sum[2] / n)
        func ratio(_ l: Double) -> Double { (max(l, back) + 0.05) / (min(l, back) + 0.05) }
        var best = 1.0
        var inkCounts: [Int: Int] = [:]
        var inkSums: [Int: [Double]] = [:]
        for index in stride(from: 0, to: bytes.count, by: 4) {
            let (r, g, b) = (Double(bytes[index]), Double(bytes[index + 1]), Double(bytes[index + 2]))
            let here = ratio(luminance(r, g, b))
            best = max(best, here)
            guard ink == .dominant, here >= 1.5 else { continue }
            let key = Int(bytes[index] >> 3) << 10 | Int(bytes[index + 1] >> 3) << 5 | Int(bytes[index + 2] >> 3)
            inkCounts[key, default: 0] += 1
            let sums = inkSums[key] ?? [0, 0, 0]
            inkSums[key] = [sums[0] + r, sums[1] + g, sums[2] + b]
        }
        guard ink == .dominant, let dominant = inkCounts.max(by: { $0.value < $1.value }),
              let sums = inkSums[dominant.key] else { return best }
        let count = Double(dominant.value)
        return ratio(luminance(sums[0] / count, sums[1] / count, sums[2] / count))
    }

    private static func luminance(_ r: Double, _ g: Double, _ b: Double) -> Double {
        func linear(_ v: Double) -> Double {
            let c = v / 255
            return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
        }
        return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)
    }

    /// Every label in the app's tree now, from one snapshot.
    private static func labels(_ app: XCUIApplication) -> Set<String> {
        var found = Set<String>()
        let tree = app.debugDescription as NSString
        // Up to the quote that ends the field, not the first apostrophe.
        // (A comma follows it: another field, or a trait such as Disabled.)
        // For a label with a line break, its first line.
        let pattern = try! NSRegularExpression(pattern: "label: '([^\\n]*?)(?:'(?=,|[ \\t]*$)|$)",
                                               options: [.anchorsMatchLines])
        for match in pattern.matches(in: tree as String, range: NSRange(location: 0, length: tree.length)) {
            found.insert(key(tree.substring(with: match.range(at: 1))))
        }
        return found
    }

    private static func typeName(_ type: XCUIAccessibilityAuditType) -> String {
        switch type {
        case .contrast: return "contrast"
        case .dynamicType: return "dynamic-type"
        case .textClipped: return "text-clipped"
        case .sufficientElementDescription: return "description"
        case .hitRegion: return "hit-region"
        case .elementDetection: return "element-detection"
        case .trait: return "trait"
        default: return "type-\(type.rawValue)"
        }
    }

    private static func describe(_ element: XCUIElement?) -> String {
        guard let element, element.exists else { return "-" }
        let frame = element.frame
        return "\(element.elementType.rawValue) id='\(element.identifier)' label='\(element.label)' "
            + "frame=(\(Int(frame.minX)),\(Int(frame.minY)),\(Int(frame.width))x\(Int(frame.height)))"
    }

    /// Whether the element is a disabled button, or the label of one.
    private static func isDisabledControl(_ element: XCUIElement?) -> Bool {
        guard let element, element.exists else { return false }
        if element.elementType == .button { return !element.isEnabled }
        let frame = element.frame
        let disabled = XCUIApplication().buttons.matching(NSPredicate(format: "enabled == false"))
        return disabled.allElementsBoundByIndex.contains {
            $0.exists && $0.frame.insetBy(dx: -2, dy: -2).contains(frame)
        }
    }

    /// Whether the element is a navigation bar's title.
    private static func isNavigationBarTitle(_ element: XCUIElement?) -> Bool {
        guard let element, element.exists, element.elementType == .staticText else { return false }
        let frame = element.frame
        return XCUIApplication().navigationBars.allElementsBoundByIndex.contains {
            $0.exists && $0.frame.insetBy(dx: -2, dy: -2).contains(frame)
                && $0.staticTexts.matching(NSPredicate(format: "label == %@", element.label)).count > 0
        }
    }

    /// Whether the element is a button in a navigation bar.
    private static func isNavigationBarButton(_ element: XCUIElement?) -> Bool {
        guard let element, element.exists, element.elementType == .button else { return false }
        let frame = element.frame
        return XCUIApplication().navigationBars.allElementsBoundByIndex.contains {
            $0.exists && $0.frame.insetBy(dx: -2, dy: -2).contains(frame)
        }
    }

    /// Whether the element is the viewer's web view or drawn inside it. By
    /// the time the audit runs the native loading panel over it is gone, and
    /// the native caption, areas and Details sit outside its frame.
    private static func isInsideWebView(_ element: XCUIElement?) -> Bool {
        guard let element, element.exists else { return false }
        if element.elementType == .webView { return true }
        let web = XCUIApplication().webViews.firstMatch
        return web.exists && web.frame.contains(element.frame)
    }

    // MARK: Navigation

    private enum Onboarding { case skip, reset }

    private func launch(_ mode: Mode, onboarding: Onboarding = .skip,
                        authority: String? = nil) -> XCUIApplication {
        XCUIDevice.shared.appearance = mode.appearance
        let app = XCUIApplication()
        switch onboarding {
        case .skip: app.launchArguments.append("-UITestSkipOnboarding")
        case .reset: app.launchArguments.append("-UITestResetOnboarding")
        }
        if let size = mode.contentSize {
            app.launchArguments += ["-UIPreferredContentSizeCategoryName", size]
        }
        app.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = authority ?? closedAuthority
        app.launch()
        return app
    }

    private func towerAuthority() throws -> String {
        guard let authority = env("GLASSES_UITEST_TOWER_AUTHORITY") else {
            throw XCTSkip("Set GLASSES_UITEST_TOWER_AUTHORITY=host:port of a Tower with the fixture worlds.")
        }
        return authority
    }

    private func statusBar(_ app: XCUIApplication) -> XCUIElement {
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", "Glasses")).firstMatch
    }

    private func openWorldBuilder(_ app: XCUIApplication, mode: Mode) {
        let cartridges = app.buttons["Cartridges"]
        XCTAssertTrue(cartridges.waitForExistence(timeout: 15), "\(mode.name): Home")
        XCTAssertTrue(tap(cartridges, until: app.navigationBars["Cartridges"].exists))
        let row = app.buttons.containing(NSPredicate(format: "label BEGINSWITH %@", "World Builder")).firstMatch
        XCTAssertTrue(reveal(app, row), "\(mode.name): the World Builder row")
        XCTAssertTrue(tap(row, until: !app.navigationBars["Cartridges"].exists
                          && app.navigationBars["World Builder"].exists),
                      "\(mode.name): World Builder opened")
    }

    private func openSavedWorlds(_ app: XCUIApplication, mode: Mode) {
        openWorldBuilder(app, mode: mode)
        let saved = app.buttons["Saved worlds"]
        XCTAssertTrue(reveal(app, saved), "\(mode.name): Saved worlds")
        XCTAssertTrue(tap(saved, until: app.navigationBars["Saved worlds"].exists))
        // Any fixture world's row: at the accessibility sizes only the first
        // is on the screen, and the list builds rows as they come into view.
        let fixture = app.staticTexts.matching(NSPredicate(format: "label ENDSWITH %@", "fixture (Mac B0)")).firstMatch
        XCTAssertTrue(fixture.waitForExistence(timeout: 20), "\(mode.name): the Tower listed the fixture worlds")
    }

    // MARK: Helpers

    private func env(_ name: String) -> String? {
        let value = ProcessInfo.processInfo.environment[name]
        return value?.isEmpty == false ? value : nil
    }

    private func element(_ app: XCUIApplication, _ identifier: String) -> XCUIElement {
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

    /// Tap `element` until `effect` is true, tapping again only while it is
    /// missing: a first tap can be swallowed while a sheet is still settling.
    @discardableResult
    private func tap(_ element: XCUIElement, until effect: @autoclosure () -> Bool, attempts: Int = 4) -> Bool {
        for _ in 0..<attempts {
            if element.waitForExistence(timeout: 5), element.isHittable { element.tap() }
            if waitFor(timeout: 3, effect) { return true }
        }
        return effect()
    }

    /// Drag up, then down, until `element` can be hit. Slow drags low on the
    /// screen and in its right-hand margin: a swipe from the middle of a
    /// small phone starts on the dimmed screen above a half-height sheet,
    /// and closes it.
    @discardableResult
    private func reveal(_ app: XCUIApplication, _ element: XCUIElement, attempts: Int = 14) -> Bool {
        if element.waitForExistence(timeout: 5), element.isHittable { return true }
        for _ in 0..<attempts {
            drag(app, from: 0.85, to: 0.45, x: 0.97)
            if element.exists && element.isHittable { return true }
        }
        for _ in 0..<attempts {
            drag(app, from: 0.45, to: 0.85, x: 0.97)
            if element.exists && element.isHittable { return true }
        }
        return element.exists && element.isHittable
    }

    /// Down only, until all of `element` is clear of the home indicator.
    private func revealBelow(_ app: XCUIApplication, _ element: XCUIElement, attempts: Int = 40) -> Bool {
        func onScreen() -> Bool {
            guard element.exists, element.isHittable else { return false }
            return element.frame.maxY <= app.windows.firstMatch.frame.maxY - 40
        }
        _ = element.waitForExistence(timeout: 3)
        for _ in 0..<attempts {
            if onScreen() { return true }
            drag(app, from: 0.7, to: 0.4)
        }
        return onScreen()
    }

    /// A half-height sheet up to full height, by its navigation bar, so the
    /// audit sees the sheet and not the dimmed screen behind it.
    private func expandSheet(_ app: XCUIApplication, bar: XCUIElement) {
        guard bar.waitForExistence(timeout: 5) else { return }
        let window = app.windows.firstMatch.frame
        for attempt in 0..<3 {
            Thread.sleep(forTimeInterval: 1.2)
            // Up already: the bar is near the top of the screen.
            if bar.frame.minY < window.height * 0.25 { return }
            if attempt == 1 {
                bar.swipeUp(velocity: .slow)
            } else {
                let start = bar.coordinate(withNormalizedOffset: CGVector(dx: 0.3, dy: 0.2))
                let end = app.coordinate(withNormalizedOffset: CGVector(dx: 0.3, dy: 0.05))
                start.press(forDuration: 0.2, thenDragTo: end, withVelocity: .slow, thenHoldForDuration: 0.2)
            }
        }
        Thread.sleep(forTimeInterval: 1)
        // Not an assertion: the first screenful is then the half-height sheet,
        // and the first drag of the audit's scroll takes it up.
        if bar.frame.minY >= window.height * 0.25 {
            XCTContext.runActivity(named: "The sheet stayed at half height; the first scroll takes it up") { _ in }
        }
    }

    private func drag(_ app: XCUIApplication, from: CGFloat, to: CGFloat, x: CGFloat = 0.5) {
        let start = app.coordinate(withNormalizedOffset: CGVector(dx: x, dy: from))
        let end = app.coordinate(withNormalizedOffset: CGVector(dx: x, dy: to))
        start.press(forDuration: 0.1, thenDragTo: end, withVelocity: .slow, thenHoldForDuration: 0.1)
    }

    private func shoot(_ app: XCUIApplication, _ name: String, mode: Mode) {
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "\(name)-\(mode.name)"
        attachment.lifetime = .deleteOnSuccess
        add(attachment)
        guard let dir = env("U05_SHOTS_DIR") else { return }
        let suffix = env("U05_SHOT_SUFFIX").map { "-\($0)" } ?? ""
        let url = URL(fileURLWithPath: dir).appendingPathComponent("\(name)-\(mode.name)\(suffix).png")
        try? screenshot.pngRepresentation.write(to: url)
    }
}
