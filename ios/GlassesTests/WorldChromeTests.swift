//
//  WorldChromeTests.swift
//  GlassesTests
//
//  U1.1 native chrome (WORLD-BUILDER-WORLDS.md §4c, protocol 1; IOS §10),
//  the phone's half without a web view: the request, the echo, the decoder
//  and its bounds, the frame check, the reducer and its exactly-once replies,
//  the research marker, the ring's drawing rule, the header capture, and the
//  rule that the chrome's sources carry none of the page's words.
//
//  The message fixtures are shaped exactly as the page's bridge posts them
//  (`appearance_chrome_bridge.js` on the U1.1 Tower branch @ cdc84db): every
//  field at the top level, `seq` from 0 at `hello`, and passed through a JSON
//  round trip so the values arrive as WebKit hands them over (NSNumber,
//  NSString, NSNull, CFBoolean).
//

import WebKit
import XCTest

@testable import Glasses

@MainActor
final class WorldChromeTests: XCTestCase {

    nonisolated private static let host = URL(string: "http://stub.invalid")!
    nonisolated private static let pageURL = WorldAssetScheme.pageURL(worldID: "w1")!
    nonisolated private static let frame = WorldChromeFrame(
        isMainFrame: true, url: "glasses-world://tower/worlds/w1/render",
        originProtocol: "glasses-world", originHost: "tower", isPageWorld: true)
    nonisolated private static let pageID = "abcdefgh12345678"

    // MARK: Fixtures

    /// What WebKit hands the handler: the JSON value, as Foundation objects.
    nonisolated private static func webKit(_ object: [String: Any]) -> Any {
        let data = try! JSONSerialization.data(withJSONObject: object)
        return try! JSONSerialization.jsonObject(with: data)
    }

    nonisolated private static func labels(face: String = "Face the room") -> [String: Any] {
        [
            "best": ["text": "Best view", "name": "Best view: fly to the clearest vantage"],
            "face": ["text": face, "name": "\(face): turn to the nearest reconstructed direction"],
            "previous": ["text": "\u{2190}", "name": "Previous recorded view"],
            "next": ["text": "\u{2192}", "name": "Next recorded view"],
            "reset": ["text": "Reset", "name": "Reset: return to the opening view"],
            "ring": ["name": "Which directions are reconstructed from here",
                     "label": "reconstructed\nfrom here", "center": "YOU"],
            "about": ["open": "About", "close": "Less"],
            "dark": ["title": "Not reconstructed from here", "tap": "Tap to turn back",
                     "name": "Not reconstructed from here. Tap to turn back"],
            "edge": "Movement stops here",
        ]
    }

    nonisolated private static func helloBody(pageID: String = pageID, seq: Int = 0, kind: String = "room",
                                  raw: Bool = false, marker: Any = NSNull(),
                                  edit: (inout [String: Any]) -> Void = { _ in }) -> [String: Any] {
        var body: [String: Any] = [
            "v": 1, "type": "hello", "pageId": pageID, "seq": seq, "protocol": [1], "kind": kind,
            "labels": labels(face: kind == "area" ? "Face the area" : "Face the room"),
            "research": ["raw": raw, "marker": marker],
        ]
        edit(&body)
        return body
    }

    nonisolated private static func stateFields(drawn: Bool = true, active: Bool = false, status: Any = NSNull(),
                                    message: Any = NSNull(), dark: Bool = false,
                                    buttons: [String: Bool] = ["best": true, "face": true, "previous": true,
                                                               "next": true, "reset": true],
                                    raw: Bool = false, marker: Any = NSNull()) -> [String: Any] {
        [
            "active": active, "phase": drawn ? "ready" : "loading", "drawn": drawn, "holding": false,
            "restoring": false, "status": status, "message": message, "hint": NSNull(), "dark": dark,
            "edge": NSNull(), "buttons": buttons, "walk": drawn ? "3 / 24" : NSNull(),
            "ring": ["shown": drawn, "lit": Array(repeating: 0.5, count: 36), "sense": 1],
            "caption": drawn ? [
                "head": "Captured images on reconstructed geometry",
                "line": "24 of 24 images loaded",
                "sections": [["title": "What you are looking at", "body": "The camera's own images."]],
                "tail": NSNull(),
            ] as [String: Any] : NSNull(),
            "research": ["raw": raw, "marker": marker],
        ]
    }

    nonisolated private static func stateBody(pageID: String = pageID, seq: Int, nonce: String,
                                  fields: [String: Any] = stateFields(),
                                  edit: (inout [String: Any]) -> Void = { _ in }) -> [String: Any] {
        var body: [String: Any] = ["v": 1, "type": "state", "pageId": pageID, "seq": seq, "nonce": nonce]
        body.merge(fields) { _, new in new }
        edit(&body)
        return body
    }

    nonisolated private static func viewBody(pageID: String = pageID, seq: Int, nonce: String,
                                 heading: Double = 0.3, half: Double = 0.5) -> [String: Any] {
        ["v": 1, "type": "view", "pageId": pageID, "seq": seq, "nonce": nonce,
         "headingRad": heading, "halfFovRad": half]
    }

    nonisolated private static func awaitBody(pageID: String = pageID, seq: Int, nonce: String) -> [String: Any] {
        ["v": 1, "type": "await", "pageId": pageID, "seq": seq, "nonce": nonce]
    }

    nonisolated private static func decode(_ body: [String: Any]) -> Result<WorldChromeMessage, WorldChromeRefusal> {
        WorldChromeDecoder.decode(webKit(body))
    }

    nonisolated private static func refusal(_ body: [String: Any], raw: Bool = false) -> WorldChromeRefusal? {
        let result = raw ? WorldChromeDecoder.decode(body) : decode(body)
        if case .failure(let refusal) = result { return refusal }
        return nil
    }

    /// Drives a session as the bridge does, numbering messages.
    private final class Driver {
        var session: WorldChromeSession
        var nextID = 0
        var log: [WorldChromeSession.Effect] = []

        init(kind: WorldChromeKind = .room, nonce: String = "N1") {
            session = WorldChromeSession(kind: kind, pageURL: WorldChromeTests.pageURL, makeNonce: { nonce })
        }

        @discardableResult
        func input(_ input: WorldChromeSession.Input) -> [WorldChromeSession.Effect] {
            let effects = session.receive(input)
            log += effects
            return effects
        }

        @discardableResult
        func send(_ body: [String: Any], frame: WorldChromeFrame = WorldChromeTests.frame)
            -> (id: Int, effects: [WorldChromeSession.Effect]) {
            let id = nextID
            nextID += 1
            return (id, input(.message(id: id, frame: frame, body: WorldChromeTests.decode(body))))
        }

        /// Echo, hello, a held await, the first state, `didFinish`, drawn,
        /// activated.
        func activate() {
            input(.pageWillLoad(echo: true))
            send(WorldChromeTests.helloBody())
            send(WorldChromeTests.stateBody(seq: 1, nonce: "N1"))
            send(WorldChromeTests.awaitBody(seq: 2, nonce: "N1"))
            input(.pageFinished)
            input(.firstStateDrawn)
        }
    }

    // MARK: I1, I2: the request and the echo

    func testThePageRequestAsksForNativeChromeAndThePollsDoNot() {
        let pinned = WorldRenderTarget(worldID: "w1", sessionID: "s1")
        XCTAssertEqual(WorldRenderClient.url(for: pinned, baseURL: Self.host)?.absoluteString,
                       "http://stub.invalid/worlds/w1/render?session_id=s1&viewer=appearance-1&wb-chrome=native")
        let unpinned = WorldRenderTarget(worldID: "w1", sessionID: nil)
        XCTAssertEqual(WorldRenderClient.url(for: unpinned, baseURL: Self.host)?.absoluteString,
                       "http://stub.invalid/worlds/w1/render?viewer=appearance-1&wb-chrome=native")
        let areaID = "a1b2c3d4e5f60718"
        let area = WorldRenderTarget(worldID: "w1", sessionID: "s1", areaID: areaID)
        XCTAssertEqual(WorldRenderClient.url(for: area, baseURL: Self.host)?.absoluteString,
                       "http://stub.invalid/worlds/w1/areas/s1/\(areaID)/render?wb-chrome=native")
        let diagnostics = WorldRenderTarget(worldID: "w1", sessionID: "s1", view: .diagnostics)
        XCTAssertEqual(WorldRenderClient.url(for: diagnostics, baseURL: Self.host)?.absoluteString,
                       "http://stub.invalid/worlds/w1/render?session_id=s1&view=diagnostics&viewer=appearance-1",
                       "the diagnostics page is sparse and keeps its own chrome")
        // Neither poll carries it: the revision is one for both variants.
        XCTAssertEqual(WorldRenderClient.revisionURL(for: pinned, baseURL: Self.host)?.query,
                       "session_id=s1&viewer=appearance-1")
        XCTAssertEqual(WorldRenderClient.revisionURL(for: unpinned, baseURL: Self.host)?.query,
                       "viewer=appearance-1")
        XCTAssertNil(WorldRenderClient.revisionURL(for: area, baseURL: Self.host)?.query)
        XCTAssertEqual(WorldChromeEcho.queryName, "wb-chrome")
        XCTAssertEqual(WorldChromeEcho.queryValue, "native")
        XCTAssertEqual(WorldChromeEcho.metaName, "wb-chrome")
    }

    func testTheEchoIsReadFromTheHeadOnly() {
        let head = #"<!doctype html><html><head><meta name="wb-representation" content="appearance">"#
            + #"<meta name="wb-revision" content="s1/appearance:1@e1"><meta name="wb-chrome" content="native">"#
        XCTAssertTrue(WorldChromeEcho.isOffered(in: head + "</head><body></body></html>"))
        XCTAssertFalse(WorldChromeEcho.isOffered(in: head.replacingOccurrences(of: "content=\"native\"",
                                                                               with: "content=\"web\"")))
        XCTAssertFalse(WorldChromeEcho.isOffered(in: head.replacingOccurrences(of: "content=\"native\"",
                                                                               with: "content=\"NATIVE\"")))
        XCTAssertFalse(WorldChromeEcho.isOffered(in: "<html><head></head><body></body></html>"))
        // Past the first 4096 characters, or only in the body: not an echo.
        let late = "<html><head>" + String(repeating: " ", count: 4100)
            + #"<meta name="wb-chrome" content="native"></head></html>"#
        XCTAssertFalse(WorldChromeEcho.isOffered(in: late))
        let body = "<html><head></head><body>" + String(repeating: "x", count: 5000)
            + #"<p>name="wb-chrome" content="native"</p></body></html>"#
        XCTAssertFalse(WorldChromeEcho.isOffered(in: body))
    }

    // MARK: I3, I4, I5: the decoder

    func testHelloDecodesAndEveryMalformedHelloIsRefused() throws {
        guard case .success(.hello(let hello, let seq)) = Self.decode(Self.helloBody()) else {
            return XCTFail("a well-formed hello")
        }
        XCTAssertEqual(seq, 0)
        XCTAssertEqual(hello.pageID, Self.pageID)
        XCTAssertEqual(hello.kind, .room)
        XCTAssertEqual(hello.labels.best, WorldChromeLabel(text: "Best view", name: "Best view: fly to the clearest vantage"))
        XCTAssertEqual(hello.labels.ringLabel, "reconstructed\nfrom here")
        XCTAssertEqual(hello.labels.ringCenter, "YOU")
        XCTAssertEqual(hello.labels.aboutOpen, "About")
        XCTAssertEqual(hello.labels.aboutClose, "Less")
        XCTAssertEqual(hello.labels.edge, "Movement stops here")
        XCTAssertEqual(hello.research, WorldChromeResearch(raw: false, marker: nil))
        guard case .success(.hello(let area, _)) = Self.decode(Self.helloBody(kind: "area")) else {
            return XCTFail("an area hello")
        }
        XCTAssertEqual(area.kind, .area)
        XCTAssertEqual(area.labels.face.text, "Face the area")

        let long41 = String(repeating: "x", count: 41)
        let cases: [(String, Any, WorldChromeRefusal)] = [
            ("not an object", ["a"] as Any, .notADictionary),
            ("v 2", Self.helloBody { $0["v"] = 2 }, .wrongVersion),
            ("v missing", Self.helloBody { $0["v"] = nil }, .wrongVersion),
            ("no type", Self.helloBody { $0["type"] = nil }, .missing("type")),
            ("type", Self.helloBody { $0["type"] = "goodbye" }, .unknownType("goodbye")),
            ("protocol 2 only", Self.helloBody { $0["protocol"] = [2] }, .protocolUnsupported),
            ("no protocol", Self.helloBody { $0["protocol"] = nil }, .missing("protocol")),
            ("kind", Self.helloBody { $0["kind"] = "street" }, .missing("kind")),
            ("no pageId", Self.helloBody { $0["pageId"] = nil }, .missing("pageId")),
            ("short pageId", Self.helloBody(pageID: "abc"), .badCount("pageId")),
            ("pageId chars", Self.helloBody(pageID: "abcdefgh<12345>"), .badCount("pageId")),
            ("pageId long", Self.helloBody(pageID: String(repeating: "a", count: 65)), .badCount("pageId")),
            ("seq", Self.helloBody { $0["seq"] = -1 }, .missing("seq")),
            ("seq fraction", Self.helloBody { $0["seq"] = 1.5 }, .missing("seq")),
            ("no labels", Self.helloBody { $0["labels"] = nil }, .missing("labels")),
            ("no dark (hooks missing)", Self.helloBody { body in
                var labels = body["labels"] as! [String: Any]
                labels["dark"] = nil
                body["labels"] = labels
            }, .missing("labels.dark")),
            ("no edge", Self.helloBody { body in
                var labels = body["labels"] as! [String: Any]
                labels["edge"] = nil
                body["labels"] = labels
            }, .missing("labels.edge")),
            ("best text 41", Self.helloBody { body in
                var labels = body["labels"] as! [String: Any]
                labels["best"] = ["text": long41, "name": "n"]
                body["labels"] = labels
            }, .tooLong("labels.best.text")),
            ("ring center 9", Self.helloBody { body in
                var labels = body["labels"] as! [String: Any]
                labels["ring"] = ["name": "n", "label": "l", "center": "123456789"]
                body["labels"] = labels
            }, .tooLong("labels.ring.center")),
            ("marker 201", Self.helloBody(raw: true, marker: String(repeating: "m", count: 201)),
             .tooLong("research.marker")),
            ("raw not a bool", Self.helloBody { $0["research"] = ["raw": 1, "marker": NSNull()] },
             .missing("research.raw")),
            ("too large", Self.helloBody { $0["padding"] = String(repeating: "p", count: 33 * 1024) },
             .tooLarge(0)),
        ]
        for (name, body, expected) in cases {
            let result: Result<WorldChromeMessage, WorldChromeRefusal>
            if let object = body as? [String: Any] {
                result = Self.decode(object)
            } else {
                result = WorldChromeDecoder.decode(body)
            }
            guard case .failure(let refusal) = result else {
                XCTFail("\(name): decoded"); continue
            }
            if case .tooLarge = expected, case .tooLarge(let size) = refusal {
                XCTAssertGreaterThan(size, WorldChromeLimits.helloBytes, name)
            } else {
                XCTAssertEqual(refusal, expected, name)
            }
        }
        // Exactly at the bound decodes: the page cuts to the bound, so a
        // string within it is never refused (UTF-16 code units).
        let fortyWithPairs = String(repeating: "\u{1F600}", count: 20)
        XCTAssertEqual(fortyWithPairs.utf16.count, 40)
        guard case .success = Self.decode(Self.helloBody { body in
            var labels = body["labels"] as! [String: Any]
            labels["best"] = ["text": fortyWithPairs, "name": "n"]
            body["labels"] = labels
        }) else { return XCTFail("40 UTF-16 code units is within the bound") }
    }

    func testStateDecodesAndEveryBoundIsEnforced() {
        guard case .success(.state(let state, let pageID, let seq, let nonce)) =
                Self.decode(Self.stateBody(seq: 1, nonce: "N1")) else {
            return XCTFail("a well-formed state")
        }
        XCTAssertEqual(pageID, Self.pageID)
        XCTAssertEqual(seq, 1)
        XCTAssertEqual(nonce, "N1")
        XCTAssertEqual(state.phase, .ready)
        XCTAssertTrue(state.drawn)
        XCTAssertEqual(state.walk, "3 / 24")
        XCTAssertEqual(state.ring.lit?.count, 36)
        XCTAssertEqual(state.caption?.sections.first?.title, "What you are looking at")
        XCTAssertNil(state.caption?.tail)
        XCTAssertEqual(state.buttons, WorldChromeButtons(best: true, face: true, previous: true, next: true, reset: true))

        // `hint`, `edge` and an empty caption as the page sends them.
        guard case .success(.state(let other, _, _, _)) = Self.decode(Self.stateBody(seq: 1, nonce: "N1") {
            $0["hint"] = ["text": "One moment — you can look around as soon as it draws", "opacity": 0.9]
            $0["edge"] = "left"
            $0["caption"] = NSNull()
            $0["ring"] = ["shown": false, "lit": NSNull(), "sense": -1]
        }) else { return XCTFail("a state with a hint and an edge") }
        XCTAssertEqual(other.hint?.text, "One moment — you can look around as soon as it draws")
        XCTAssertEqual(other.edge, .left)
        XCTAssertNil(other.caption)
        XCTAssertNil(other.ring.lit)
        XCTAssertEqual(other.ring.sense, -1)
        // Every string at its bound, eight sections: well over the first
        // draft's 24 KiB, and within the 64 KiB the Tower asks for.
        guard case .success = Self.decode(Self.stateBody(seq: 1, nonce: "N1") {
            $0["status"] = String(repeating: "s", count: 200)
            $0["message"] = NSNull()
            $0["caption"] = ["head": String(repeating: "h", count: 300), "line": String(repeating: "l", count: 600),
                             "sections": Array(repeating: ["title": String(repeating: "t", count: 80),
                                                           "body": String(repeating: "b", count: 1600)], count: 8),
                             "tail": String(repeating: "t", count: 200)]
        }) else { return XCTFail("a state at every bound decodes") }

        func over(_ n: Int) -> String { String(repeating: "s", count: n + 1) }
        func caption(_ edit: (inout [String: Any]) -> Void) -> [String: Any] {
            var caption: [String: Any] = ["head": "h", "line": NSNull(),
                                          "sections": [["title": "t", "body": "b"]], "tail": NSNull()]
            edit(&caption)
            return caption
        }
        let cases: [(String, [String: Any], WorldChromeRefusal)] = [
            ("status", Self.stateBody(seq: 1, nonce: "N1") { $0["status"] = over(200) }, .tooLong("status")),
            ("message", Self.stateBody(seq: 1, nonce: "N1") { $0["message"] = over(800) }, .tooLong("message")),
            ("hint", Self.stateBody(seq: 1, nonce: "N1") { $0["hint"] = ["text": over(200), "opacity": 1] },
             .tooLong("hint.text")),
            ("walk", Self.stateBody(seq: 1, nonce: "N1") { $0["walk"] = over(24) }, .tooLong("walk")),
            ("head", Self.stateBody(seq: 1, nonce: "N1") { $0["caption"] = caption { $0["head"] = over(300) } },
             .tooLong("caption.head")),
            ("line", Self.stateBody(seq: 1, nonce: "N1") { $0["caption"] = caption { $0["line"] = over(600) } },
             .tooLong("caption.line")),
            ("tail", Self.stateBody(seq: 1, nonce: "N1") { $0["caption"] = caption { $0["tail"] = over(200) } },
             .tooLong("caption.tail")),
            ("section title", Self.stateBody(seq: 1, nonce: "N1") {
                $0["caption"] = caption { $0["sections"] = [["title": over(80), "body": "b"]] }
            }, .tooLong("caption.sections.title")),
            ("section body", Self.stateBody(seq: 1, nonce: "N1") {
                $0["caption"] = caption { $0["sections"] = [["title": "t", "body": over(1600)]] }
            }, .tooLong("caption.sections.body")),
            ("no sections", Self.stateBody(seq: 1, nonce: "N1") { $0["caption"] = caption { $0["sections"] = [] } },
             .badCount("caption.sections")),
            ("nine sections", Self.stateBody(seq: 1, nonce: "N1") {
                $0["caption"] = caption { $0["sections"] = Array(repeating: ["title": "t", "body": "b"], count: 9) }
            }, .badCount("caption.sections")),
            ("marker", Self.stateBody(seq: 1, nonce: "N1") {
                $0["research"] = ["raw": true, "marker": over(200)]
            }, .tooLong("research.marker")),
            ("lit 35", Self.stateBody(seq: 1, nonce: "N1") {
                $0["ring"] = ["shown": true, "lit": Array(repeating: 0.1, count: 35), "sense": 1]
            }, .badCount("ring.lit")),
            ("sense 0", Self.stateBody(seq: 1, nonce: "N1") {
                $0["ring"] = ["shown": true, "lit": NSNull(), "sense": 0]
            }, .missing("ring.sense")),
            ("phase", Self.stateBody(seq: 1, nonce: "N1") { $0["phase"] = "dreaming" }, .missing("phase")),
            ("edge", Self.stateBody(seq: 1, nonce: "N1") { $0["edge"] = "up" }, .missing("edge")),
            ("buttons", Self.stateBody(seq: 1, nonce: "N1") { $0["buttons"] = ["best": true] },
             .missing("buttons.face")),
            ("drawn as a number", Self.stateBody(seq: 1, nonce: "N1") { $0["drawn"] = 1 }, .missing("drawn")),
            ("no nonce", Self.stateBody(seq: 1, nonce: "N1") { $0["nonce"] = nil }, .missing("nonce")),
            ("nonce 129", Self.stateBody(seq: 1, nonce: String(repeating: "n", count: 129)), .tooLong("nonce")),
            ("too large", Self.stateBody(seq: 1, nonce: "N1") { $0["pad"] = String(repeating: "p", count: 66_000) },
             .tooLarge(0)),
        ]
        for (name, body, expected) in cases {
            guard let refusal = Self.refusal(body) else { XCTFail("\(name): decoded"); continue }
            if case .tooLarge = expected, case .tooLarge(let size) = refusal {
                XCTAssertGreaterThan(size, WorldChromeLimits.stateBytes, name)
            } else {
                XCTAssertEqual(refusal, expected, name)
            }
        }
        // Non-finite numbers cannot be JSON; WebKit can still hand them over.
        var nan = Self.stateBody(seq: 1, nonce: "N1")
        nan["ring"] = ["shown": true, "lit": [Double.nan] + Array(repeating: 0.1, count: 35), "sense": 1]
        XCTAssertEqual(Self.refusal(nan, raw: true), .notFinite("ring.lit"))
        var infinite = Self.stateBody(seq: 1, nonce: "N1")
        infinite["hint"] = ["text": "t", "opacity": Double.infinity]
        XCTAssertEqual(Self.refusal(infinite, raw: true), .notFinite("hint.opacity"))
    }

    func testViewDecodesAndNonFiniteIsRefused() {
        guard case .success(.view(let view, _, let seq, _)) = Self.decode(Self.viewBody(seq: 3, nonce: "N1")) else {
            return XCTFail("a well-formed view")
        }
        XCTAssertEqual(view, WorldChromeView(headingRad: 0.3, halfFovRad: 0.5))
        XCTAssertEqual(seq, 3)
        XCTAssertEqual(Self.refusal(Self.viewBody(seq: 3, nonce: "N1", heading: .nan), raw: true),
                       .notFinite("headingRad"))
        XCTAssertEqual(Self.refusal(Self.viewBody(seq: 3, nonce: "N1", half: 0)), .notFinite("halfFovRad"))
        XCTAssertEqual(Self.refusal(Self.viewBody(seq: 3, nonce: "N1", half: Double.pi / 2)),
                       .notFinite("halfFovRad"))
        var big = Self.viewBody(seq: 3, nonce: "N1")
        big["pad"] = String(repeating: "p", count: 600)
        if case .tooLarge(let size)? = Self.refusal(big) {
            XCTAssertGreaterThan(size, WorldChromeLimits.viewBytes)
        } else {
            XCTFail("a view over 512 bytes")
        }
        var bigAwait = Self.awaitBody(seq: 4, nonce: "N1")
        bigAwait["pad"] = String(repeating: "p", count: 300)
        if case .tooLarge? = Self.refusal(bigAwait) {} else { XCTFail("an await over 256 bytes") }
    }

    // MARK: I6: where a message comes from

    func testAMessageFromAnotherFrameURLOriginOrWorldIsRefused() {
        let page = Self.pageURL
        XCTAssertTrue(Self.frame.admits(pageURL: page))
        XCTAssertTrue(WorldChromeFrame(isMainFrame: true, url: page.absoluteString, originProtocol: "glasses-world:",
                                       originHost: "tower", isPageWorld: true).admits(pageURL: page))
        let refused: [(String, WorldChromeFrame)] = [
            ("a subframe", WorldChromeFrame(isMainFrame: false, url: page.absoluteString,
                                            originProtocol: "glasses-world", originHost: "tower", isPageWorld: true)),
            ("another URL", WorldChromeFrame(isMainFrame: true, url: page.absoluteString + "?x=1",
                                             originProtocol: "glasses-world", originHost: "tower", isPageWorld: true)),
            ("another world's page", WorldChromeFrame(isMainFrame: true, url: "glasses-world://tower/worlds/w2/render",
                                                      originProtocol: "glasses-world", originHost: "tower",
                                                      isPageWorld: true)),
            ("another origin", WorldChromeFrame(isMainFrame: true, url: page.absoluteString,
                                                originProtocol: "https", originHost: "tower", isPageWorld: true)),
            ("another host", WorldChromeFrame(isMainFrame: true, url: page.absoluteString,
                                              originProtocol: "glasses-world", originHost: "evil", isPageWorld: true)),
            ("another content world", WorldChromeFrame(isMainFrame: true, url: page.absoluteString,
                                                       originProtocol: "glasses-world", originHost: "tower",
                                                       isPageWorld: false)),
            ("no URL", WorldChromeFrame(isMainFrame: true, url: nil, originProtocol: "glasses-world",
                                        originHost: "tower", isPageWorld: true)),
        ]
        for (name, frame) in refused {
            XCTAssertFalse(frame.admits(pageURL: page), name)
            // And the session declines its hello and falls back for the screen.
            let driver = Driver()
            driver.input(.pageWillLoad(echo: true))
            let (id, effects) = driver.send(Self.helloBody(), frame: frame)
            XCTAssertEqual(effects.first, .reply(id: id, .decline), name)
            XCTAssertTrue(effects.contains(.setMode(.legacy)), name)
            XCTAssertTrue(driver.session.refusedForScreen, name)
        }
        // The area viewer's page URL is its own.
        let areaPage = WorldAssetScheme.pageURL(worldID: "w1", scope: .area(sessionID: "s1", areaID: "a1b2c3d4e5f60718"))
        XCTAssertFalse(Self.frame.admits(pageURL: areaPage))
    }

    // MARK: I7: activation

    func testActivationFollowsTheFirstDrawnStateAndNeverPrecedesIt() {
        let driver = Driver()
        XCTAssertEqual(driver.input(.pageWillLoad(echo: true)), [.setMode(.pending), .arm(.noHello)])
        XCTAssertEqual(driver.input(.firstStateDrawn), [], "nothing is drawn before a state")
        let hello = driver.send(Self.helloBody())
        guard case .success(.hello(let decoded, _)) = Self.decode(Self.helloBody()) else { return XCTFail() }
        XCTAssertEqual(hello.effects, [.reply(id: hello.id, .welcome(nonce: "N1")), .publishHello(decoded),
                                       .cancel(.noHello), .arm(.noState)])
        XCTAssertEqual(driver.input(.pageFinished), [], "didFinish alone: no state has been drawn")
        // The page awaits before any state: held, nothing to say yet.
        let firstAwait = driver.send(Self.awaitBody(seq: 1, nonce: "N1"))
        XCTAssertEqual(firstAwait.effects, [.hold(id: firstAwait.id)])
        let view = driver.send(Self.viewBody(seq: 2, nonce: "N1"))
        XCTAssertEqual(view.effects, [.reply(id: view.id, nil), .publishView(WorldChromeView(headingRad: 0.3, halfFovRad: 0.5))])
        XCTAssertEqual(driver.session.mode, .pending, "a view is not a state")
        let state = driver.send(Self.stateBody(seq: 3, nonce: "N1"))
        XCTAssertTrue(state.effects.contains(.setMode(.native)))
        XCTAssertTrue(state.effects.contains(.cancel(.noState)))
        XCTAssertFalse(state.effects.contains { if case .reply(let id, _) = $0 { return id == firstAwait.id }; return false },
                       "a state alone never activates: the phone has not drawn it yet")
        XCTAssertEqual(driver.input(.firstStateDrawn), [.reply(id: firstAwait.id, .activate)])
        XCTAssertEqual(driver.input(.firstStateDrawn), [], "activate is sent once per page")
        // The next await is held: nothing more to say.
        let next = driver.send(Self.awaitBody(seq: 4, nonce: "N1"))
        XCTAssertEqual(next.effects, [.hold(id: next.id)])
    }

    /// The page posts `hello` while it is still being parsed, and its first
    /// state can follow before `didFinish`. The overlay then draws that
    /// state under the opaque rendering panel ("Drawing the world…"), so the
    /// phone's chrome is not yet on screen and `activate` waits for
    /// `didFinish` (WORLDS §4c: the page draws its own chrome until the phone
    /// has drawn its own). Driven through the web view's coordinator, as
    /// WebKit drives it.
    func testActivationWaitsForDidFinishWhenTheStateArrivesWhileParsing() {
        let model = WorldChromeModel()
        let bridge = WorldChromeBridge(model: model, kind: .room, pageURL: Self.pageURL)
        let target = WorldRenderTarget(worldID: "w1", sessionID: "s1")
        let assets = WorldAssetSchemeHandler(worldID: "w1", scope: WorldAssetScope.of(target))
        let coordinator = WorldRenderWebView.Coordinator(target: target, assets: assets, bridge: bridge)
        var events: [WorldRenderPageEvent] = []
        coordinator.onEvent = { events.append($0) }
        var replies: [String: [String: Any]] = [:]
        func post(_ name: String, _ body: [String: Any]) {
            bridge.receive(body: Self.webKit(body), frame: Self.frame) { reply, _ in
                replies[name] = reply as? [String: Any] ?? [:]
            }
        }
        // The navigation was allowed; the page is being parsed.
        bridge.receive(.pageWillLoad(echo: true))
        post("hello", Self.helloBody())
        let nonce = replies["hello"]?["nonce"] as? String ?? ""
        XCTAssertFalse(nonce.isEmpty, "welcomed")
        post("state", Self.stateBody(seq: 1, nonce: nonce))
        post("await", Self.awaitBody(seq: 2, nonce: nonce))
        XCTAssertEqual(model.mode, .native)
        // The overlay has drawn the state, under the rendering panel.
        bridge.receive(.firstStateDrawn)
        XCTAssertNil(replies["await"], "no activate while the rendering panel covers the native chrome")
        XCTAssertEqual(bridge.outstandingReplies, 1, "the await is held")

        // didFinish, late: the panel goes, and the page may put its chrome away.
        coordinator.webView(WKWebView(frame: .zero), didFinish: nil)
        XCTAssertEqual(events, [.rendered])
        XCTAssertEqual(replies["await"]?["type"] as? String, "activate")
        XCTAssertEqual(bridge.outstandingReplies, 0)
        bridge.receive(.teardown)
    }

    /// The reducer's half of the same rule: `activate` needs both the drawn
    /// state and `didFinish`, in either order, once per page, and a new
    /// document waits for its own `didFinish`.
    func testActivationNeedsTheDrawnStateAndTheFinishedPageInEitherOrder() {
        let driver = Driver()
        driver.input(.pageWillLoad(echo: true))
        driver.send(Self.helloBody())
        driver.send(Self.stateBody(seq: 1, nonce: "N1"))
        let held = driver.send(Self.awaitBody(seq: 2, nonce: "N1"))
        XCTAssertEqual(driver.input(.firstStateDrawn), [], "drawn under the rendering panel")
        XCTAssertEqual(driver.input(.pageFinished), [.reply(id: held.id, .activate)])
        XCTAssertEqual(driver.input(.pageFinished), [], "once per page")
        XCTAssertEqual(driver.input(.firstStateDrawn), [], "once per page")

        // The page reloads: the last document's didFinish is not this one's.
        let page = "secondpage123456"
        driver.input(.pageWillLoad(echo: true))
        driver.send(Self.helloBody(pageID: page, seq: 0))
        driver.send(Self.stateBody(pageID: page, seq: 1, nonce: "N1"))
        let second = driver.send(Self.awaitBody(pageID: page, seq: 2, nonce: "N1"))
        XCTAssertEqual(driver.input(.firstStateDrawn), [], "this document has not finished")
        XCTAssertEqual(driver.input(.pageFinished), [.reply(id: second.id, .activate)])

        // didFinish before the state (the usual order on the Tower's page).
        let usual = Driver()
        usual.input(.pageWillLoad(echo: true))
        usual.send(Self.helloBody())
        XCTAssertEqual(usual.input(.pageFinished), [])
        usual.send(Self.stateBody(seq: 1, nonce: "N1"))
        let waiting = usual.send(Self.awaitBody(seq: 2, nonce: "N1"))
        XCTAssertEqual(usual.input(.firstStateDrawn), [.reply(id: waiting.id, .activate)])
        // Nothing for a page that fell back.
        let fallen = Driver()
        fallen.input(.pageWillLoad(echo: true))
        fallen.input(.timer(.noHello))
        XCTAssertEqual(fallen.input(.pageFinished), [])
        XCTAssertEqual(fallen.input(.firstStateDrawn), [])
    }

    // MARK: I8: every message answered exactly once

    func testEveryMessageIsAnsweredExactlyOnce() {
        var rng = SeededGenerator(seed: 0x5EED_C0DE)
        for round in 0..<300 {
            let driver = Driver(nonce: "N1")
            var ids: [Int] = []
            var seq = 0
            var page = Self.pageID
            for _ in 0..<40 {
                switch Int.random(in: 0..<15, using: &rng) {
                case 0:
                    driver.input(.pageWillLoad(echo: Bool.random(using: &rng)))
                    seq = 0
                    page = Bool.random(using: &rng) ? Self.pageID : "page\(Int.random(in: 1000...9999, using: &rng))"
                case 1:
                    ids.append(driver.send(Self.helloBody(pageID: page, seq: seq)).id); seq += 1
                case 2, 3:
                    ids.append(driver.send(Self.stateBody(pageID: page, seq: seq, nonce: "N1",
                                                          fields: Self.stateFields(active: Bool.random(using: &rng)))).id)
                    seq += 1
                case 4:
                    ids.append(driver.send(Self.viewBody(pageID: page, seq: seq, nonce: "N1")).id); seq += 1
                case 5, 6:
                    ids.append(driver.send(Self.awaitBody(pageID: page, seq: seq, nonce: "N1")).id); seq += 1
                case 7:
                    // A refused one: a wrong nonce, a regressed seq, or junk.
                    switch Int.random(in: 0..<3, using: &rng) {
                    case 0: ids.append(driver.send(Self.awaitBody(pageID: page, seq: seq, nonce: "X")).id)
                    case 1: ids.append(driver.send(Self.stateBody(pageID: page, seq: 0, nonce: "N1")).id)
                    default: ids.append(driver.send(["junk": true]).id)
                    }
                case 8: driver.input(.firstStateDrawn)
                case 9: driver.input(.tapped(WorldChromeAction.allCases.randomElement(using: &rng)!))
                case 10: driver.input(.timer(.noHello))
                case 11: driver.input(.timer(.noState))
                case 12: driver.input(.timer(.noDeactivateConfirm))
                case 13: driver.input(.pageFinished)
                default: driver.input(.teardown)
                }
            }
            driver.input(.teardown)
            var answered: [Int: Int] = [:]
            var holds: [Int] = []
            for effect in driver.log {
                if case .reply(let id, _) = effect { answered[id, default: 0] += 1 }
                if case .hold(let id) = effect { holds.append(id) }
            }
            for id in ids {
                XCTAssertEqual(answered[id], 1, "round \(round): message \(id) answered \(answered[id] ?? 0) times")
            }
            XCTAssertEqual(Set(answered.keys), Set(ids), "round \(round): a reply to no message")
            XCTAssertFalse(driver.session.isHoldingAnAwait, "round \(round): teardown answers the held await")
        }
    }

    /// The bridge stores each WebKit reply block and calls it once, the held
    /// one included (`teardown`).
    func testTheBridgeCallsEveryReplyBlockOnce() {
        let model = WorldChromeModel()
        let bridge = WorldChromeBridge(model: model, kind: .room, pageURL: Self.pageURL)
        var calls: [String: Int] = [:]
        var replies: [String: [String: Any]] = [:]
        func post(_ name: String, _ body: [String: Any]) {
            bridge.receive(body: Self.webKit(body), frame: Self.frame) { reply, error in
                XCTAssertNil(error)
                calls[name, default: 0] += 1
                replies[name] = reply as? [String: Any]
            }
        }
        bridge.receive(.pageWillLoad(echo: true))
        XCTAssertEqual(model.mode, .pending)
        post("hello", Self.helloBody())
        XCTAssertEqual(replies["hello"]?["type"] as? String, "welcome")
        XCTAssertEqual(replies["hello"]?["protocol"] as? Int, 1)
        let nonce = try? XCTUnwrap(replies["hello"]?["nonce"] as? String)
        XCTAssertNotNil(model.hello)
        post("state", Self.stateBody(seq: 1, nonce: nonce ?? ""))
        XCTAssertEqual(model.mode, .native)
        XCTAssertEqual(model.firstStateToken, 1)
        XCTAssertTrue(model.everNative)
        post("await", Self.awaitBody(seq: 2, nonce: nonce ?? ""))
        XCTAssertNil(calls["await"], "held")
        XCTAssertEqual(bridge.outstandingReplies, 1)
        bridge.receive(.teardown)
        XCTAssertEqual(calls, ["hello": 1, "state": 1, "await": 1])
        XCTAssertEqual(replies["await"]?["type"] as? String, "close")
        XCTAssertEqual(bridge.outstandingReplies, 0)
        // Table C, exactly.
        XCTAssertEqual(WorldChromeCommand.action(.next).reply["name"] as? String, "next")
        XCTAssertEqual(WorldChromeCommand.decline.reply.count, 2)
        XCTAssertEqual(WorldChromeCommand.activate.reply["v"] as? Int, 1)
    }

    // MARK: I9: a navigation

    func testANavigationAnswersTheHeldAwaitAndResetsTheSession() {
        let driver = Driver()
        driver.input(.pageWillLoad(echo: true))
        driver.send(Self.helloBody())
        driver.send(Self.stateBody(seq: 1, nonce: "N1"))
        let held = driver.send(Self.awaitBody(seq: 2, nonce: "N1"))
        XCTAssertEqual(held.effects, [.hold(id: held.id)])
        // The page reloads itself (or the app swaps it).
        let effects = driver.input(.pageWillLoad(echo: true))
        XCTAssertEqual(effects.first, .reply(id: held.id, .close))
        XCTAssertTrue(effects.contains(.setMode(.pending)))
        XCTAssertTrue(effects.contains(.arm(.noHello)))
        XCTAssertEqual(driver.session.mode, .pending)
        XCTAssertFalse(driver.session.refusedForScreen)
        // The old document's last posts before it unloads: answered, ignored.
        let late = driver.send(Self.stateBody(seq: 3, nonce: "N1"))
        XCTAssertEqual(late.effects, [.reply(id: late.id, nil)])
        let lateAwait = driver.send(Self.awaitBody(seq: 4, nonce: "N1"))
        XCTAssertEqual(lateAwait.effects, [.reply(id: lateAwait.id, .close)])
        XCTAssertEqual(driver.session.mode, .pending)
        // The new page starts again from seq 0, with its own id.
        let hello = driver.send(Self.helloBody(pageID: "newpage12345678x", seq: 0))
        XCTAssertEqual(hello.effects.first, .reply(id: hello.id, .welcome(nonce: "N1")))
        // A navigation to a page with no echo is today's screen.
        XCTAssertTrue(driver.input(.pageWillLoad(echo: false)).contains(.setMode(.legacy)))
        let stray = driver.send(Self.helloBody(pageID: "straypage1234567"))
        XCTAssertEqual(stray.effects, [.reply(id: stray.id, .decline)])
    }

    /// `seq` is 0 at `hello` (WORLDS §4c: "rising by one per page message,
    /// from 0 at `hello`"). A first `hello` with any other number is an
    /// invalid `hello`: declined, and the screen keeps the page's own chrome
    /// (spec C15r §3.1: "pending | an invalid hello | Reply .decline.
    /// refuseForScreen, setMode(.legacy)").
    func testAHelloThatDoesNotStartAtSeqZeroIsDeclined() {
        let driver = Driver()
        driver.input(.pageWillLoad(echo: true))
        let hello = driver.send(Self.helloBody(seq: 1))
        XCTAssertEqual(hello.effects, [.reply(id: hello.id, .decline), .refuseForScreen, .setMode(.legacy),
                                       .cancel(.noHello)])
        XCTAssertEqual(driver.session.mode, .legacy)
        XCTAssertTrue(driver.session.refusedForScreen)
        // For the life of the screen: a later hello, from 0, is declined too.
        let again = driver.send(Self.helloBody(pageID: "secondpage123456", seq: 0))
        XCTAssertEqual(again.effects, [.reply(id: again.id, .decline)])
        XCTAssertTrue(driver.input(.pageWillLoad(echo: true)).contains(.setMode(.legacy)))

        for seq in [2, 41, 1_000_000] {
            let other = Driver()
            other.input(.pageWillLoad(echo: true))
            let refused = other.send(Self.helloBody(seq: seq))
            XCTAssertEqual(refused.effects.first, .reply(id: refused.id, .decline), "seq \(seq)")
            XCTAssertTrue(other.session.refusedForScreen, "seq \(seq)")
        }
        // From 0 it is welcomed, and the next message is 1.
        let good = Driver()
        good.input(.pageWillLoad(echo: true))
        let welcomed = good.send(Self.helloBody(seq: 0))
        XCTAssertEqual(welcomed.effects.first, .reply(id: welcomed.id, .welcome(nonce: "N1")))
        let state = good.send(Self.stateBody(seq: 1, nonce: "N1"))
        XCTAssertTrue(state.effects.contains(.setMode(.native)))
    }

    // MARK: I10: a refusal after activation

    func testARefusalAfterActivationDeactivatesAndFallsBackForTheScreen() {
        let driver = Driver()
        driver.activate()
        // The activate went to the held await. A wrong nonce now.
        let bad = driver.send(Self.stateBody(seq: 5, nonce: "WRONG"))
        XCTAssertEqual(bad.effects, [.reply(id: bad.id, nil), .refuseForScreen, .setMode(.legacy),
                                     .arm(.noDeactivateConfirm)])
        XCTAssertTrue(driver.session.refusedForScreen)
        // The next await gets the deactivate.
        let next = driver.send(Self.awaitBody(seq: 6, nonce: "N1"))
        XCTAssertEqual(next.effects, [.reply(id: next.id, .deactivate)])
        // The page confirms with active:false.
        let confirm = driver.send(Self.stateBody(seq: 7, nonce: "N1", fields: Self.stateFields(active: false)))
        XCTAssertEqual(confirm.effects, [.reply(id: confirm.id, nil), .cancel(.noDeactivateConfirm)])
        XCTAssertEqual(driver.input(.timer(.noDeactivateConfirm)), [], "cancelled")
        // For the life of the screen: a later page with the echo is today's.
        XCTAssertTrue(driver.input(.pageWillLoad(echo: true)).contains(.setMode(.legacy)))

        // Each other refusal after activation does the same.
        let refusals: [(String, (Driver) -> [String: Any])] = [
            ("seq regressed", { _ in Self.stateBody(seq: 1, nonce: "N1") }),
            ("pageId changed", { _ in Self.stateBody(pageID: "otherpage1234567", seq: 9, nonce: "N1") }),
            ("bounds", { _ in Self.stateBody(seq: 9, nonce: "N1") { $0["status"] = String(repeating: "s", count: 201) } }),
            ("a second hello", { _ in Self.helloBody(seq: 9) }),
        ]
        for (name, make) in refusals {
            let driver = Driver()
            driver.activate()
            let refused = driver.send(make(driver))
            XCTAssertTrue(refused.effects.contains(.setMode(.legacy)), name)
            XCTAssertTrue(refused.effects.contains(.arm(.noDeactivateConfirm)), name)
            XCTAssertTrue(driver.session.refusedForScreen, name)
        }
        let wrongFrame = Driver()
        wrongFrame.activate()
        let fromAFrame = wrongFrame.send(Self.stateBody(seq: 9, nonce: "N1"), frame: WorldChromeFrame(
            isMainFrame: false, url: Self.pageURL.absoluteString, originProtocol: "glasses-world",
            originHost: "tower", isPageWorld: true))
        XCTAssertTrue(fromAFrame.effects.contains(.setMode(.legacy)))

        // No confirmation: one reload, and the reloaded page is declined.
        let silent = Driver()
        silent.activate()
        let held = silent.send(Self.awaitBody(seq: 3, nonce: "N1"))
        let refusal = silent.send(Self.viewBody(seq: 4, nonce: "N1", heading: 0.3, half: 0.5) .merging(["nonce": "X"]) { $1 })
        XCTAssertTrue(refusal.effects.contains(.reply(id: held.id, .deactivate)), "delivered to the held await")
        XCTAssertEqual(silent.input(.timer(.noDeactivateConfirm)), [.reloadPage])
        XCTAssertEqual(silent.input(.timer(.noDeactivateConfirm)), [], "once")
        XCTAssertTrue(silent.input(.pageWillLoad(echo: true)).contains(.setMode(.legacy)))
        let again = silent.send(Self.helloBody(pageID: "reloaded12345678"))
        XCTAssertEqual(again.effects, [.reply(id: again.id, .decline)])
    }

    // MARK: I11: silence

    func testNoHelloOrNoStateInFiveSecondsFallsBackToLegacy() {
        let bridge = WorldChromeBridge(model: WorldChromeModel(), kind: .room, pageURL: Self.pageURL)
        XCTAssertEqual(bridge.noHelloTimeout, .seconds(5))
        XCTAssertEqual(bridge.noStateTimeout, .seconds(5))
        XCTAssertEqual(bridge.noDeactivateConfirmTimeout, .seconds(2))

        let noHello = Driver()
        noHello.input(.pageWillLoad(echo: true))
        XCTAssertEqual(noHello.input(.timer(.noHello)), [.refuseForScreen, .setMode(.legacy)])
        let late = noHello.send(Self.helloBody())
        XCTAssertEqual(late.effects, [.reply(id: late.id, .decline)], "a hello after the fallback is declined")

        let noState = Driver()
        noState.input(.pageWillLoad(echo: true))
        noState.send(Self.helloBody())
        let held = noState.send(Self.awaitBody(seq: 1, nonce: "N1"))
        let effects = noState.input(.timer(.noState))
        // The contract has the page confirm every deactivate with
        // `active:false`, activated or not (WORLDS §4c): the phone waits for it.
        XCTAssertEqual(effects, [.refuseForScreen, .setMode(.legacy), .arm(.noDeactivateConfirm),
                                 .reply(id: held.id, .deactivate)])
        let confirm = noState.send(Self.stateBody(seq: 2, nonce: "N1", fields: Self.stateFields(drawn: false)))
        XCTAssertEqual(confirm.effects, [.reply(id: confirm.id, nil), .cancel(.noDeactivateConfirm)])
        // A stale timer from a page already gone does nothing.
        let stale = Driver()
        stale.input(.pageWillLoad(echo: true))
        stale.input(.pageWillLoad(echo: false))
        XCTAssertEqual(stale.input(.timer(.noHello)), [])
    }

    func testTheBridgeFallsBackOnItsOwnTimer() async throws {
        let model = WorldChromeModel()
        let bridge = WorldChromeBridge(model: model, kind: .room, pageURL: Self.pageURL)
        bridge.noHelloTimeout = .milliseconds(30)
        bridge.receive(.pageWillLoad(echo: true))
        XCTAssertEqual(model.mode, .pending)
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertEqual(model.mode, .legacy)
        XCTAssertTrue(model.refusedForScreen)
    }

    // MARK: The half-point guards

    /// The viewer's two height write-backs (its own height, and a capped
    /// scroller's content) take a change of half a point or more, and
    /// nothing smaller: the boundary, both ways.
    func testAMeasuredHeightIsWrittenBackFromHalfAPoint() {
        XCTAssertEqual(MeasuredHeight.threshold, 0.5)
        XCTAssertFalse(MeasuredHeight.moved(from: 708, to: 708))
        XCTAssertFalse(MeasuredHeight.moved(from: 708, to: 708.49))
        XCTAssertFalse(MeasuredHeight.moved(from: 708, to: 707.51))
        XCTAssertFalse(MeasuredHeight.moved(from: 708, to: 708 + 1.0 / 3), "a third of a point, one pixel at 3x")
        XCTAssertTrue(MeasuredHeight.moved(from: 708, to: 708.5))
        XCTAssertTrue(MeasuredHeight.moved(from: 708, to: 707.5))
        XCTAssertTrue(MeasuredHeight.moved(from: 708, to: 708 + 2.0 / 3), "two pixels at 3x")
        XCTAssertTrue(MeasuredHeight.moved(from: 0, to: 708), "the first measurement")
        XCTAssertTrue(MeasuredHeight.moved(from: 52, to: 62), "a band that grew a line")
        // Both write-backs go through it: no other half-point test remains.
        let source = try? String(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Glasses/Workspaces/WorldBuilder/WorldRenderViewer.swift"), encoding: .utf8)
        XCTAssertNotNil(source)
        XCTAssertEqual(source?.components(separatedBy: "MeasuredHeight.moved(").count, 3,
                       "the scene's height and CappedScroll's content height")
        XCTAssertFalse(source?.contains(">= 0.5") ?? true)
    }

    // MARK: I12: the research marker

    func testTheResearchMarkerIsStickyAndAlsoRaisedByTheHeader() {
        let model = WorldChromeModel()
        XCTAssertNil(model.researchMarker)
        model.raiseResearch(headerWarning: nil)
        XCTAssertNil(model.researchMarker, "iOS writes no words of its own")
        model.raiseResearch(headerWarning: "Research build: the header's sentence")
        XCTAssertEqual(model.researchMarker, "Research build: the header's sentence")
        // The page's own words win once they arrive.
        let rawState = Self.decodedState(Self.stateFields(raw: true, marker: "The page's marker"))
        model.publishState(rawState)
        XCTAssertEqual(model.researchMarker, "The page's marker")
        model.raiseResearch(headerWarning: "Another sentence")
        XCTAssertEqual(model.researchMarker, "The page's marker")
        // Never lowered by raw:false, a page swap, a fallback.
        model.publishState(Self.decodedState(Self.stateFields(raw: false)))
        XCTAssertEqual(model.researchMarker, "The page's marker")
        model.setMode(.pending)
        model.setMode(.legacy)
        model.refuseForScreen()
        XCTAssertEqual(model.researchMarker, "The page's marker")
        // Only the viewer closing drops it.
        model.viewerClosed()
        XCTAssertNil(model.researchMarker)
        XCTAssertFalse(model.everNative)

        // The header's source, end to end: the scheme handler reports a
        // non-redacted imagery, and never a redacted one.
        let handler = WorldAssetSchemeHandler(worldID: "w1")
        var reported: [String?] = []
        handler.onImagery = { _, warning in reported.append(warning) }
        let viewer = WorldRenderViewerModel(target: WorldRenderTarget(worldID: "w1", sessionID: "s1"),
                                            assets: handler)
        XCTAssertNotNil(handler.onImagery, "the viewer wires the header to its chrome")
        handler.onImagery?("raw-local", "Research build: unredacted")
        XCTAssertEqual(viewer.chrome.researchMarker, "Research build: unredacted")
        _ = reported
    }

    nonisolated private static func decodedState(_ fields: [String: Any]) -> WorldChromeState {
        guard case .success(.state(let state, _, _, _)) = decode(stateBody(seq: 1, nonce: "N1", fields: fields)) else {
            fatalError("fixture state did not decode")
        }
        return state
    }

    // MARK: I13: the ring

    func testRingBinAnglesAndColoursAreThePagesRule() {
        let g = WorldChromeRingGeometry.self
        let step = 2 * Double.pi / 36
        XCTAssertEqual(g.size, 58)
        XCTAssertEqual(g.discRadius, 26)
        XCTAssertEqual(g.arcRadius, 21)
        XCTAssertEqual(g.lineWidth, 5.5)
        XCTAssertEqual(g.sectorRadius, 16.5)
        // −π/2 + sense·(i·2π/36 − heading): bin 0 straight up at heading 0.
        XCTAssertEqual(g.binAngle(0, sense: 1, headingRad: 0), -Double.pi / 2, accuracy: 1e-12)
        XCTAssertEqual(g.binAngle(9, sense: 1, headingRad: 0), 0, accuracy: 1e-12)
        XCTAssertEqual(g.binAngle(9, sense: -1, headingRad: 0), -Double.pi, accuracy: 1e-12)
        XCTAssertEqual(g.binAngle(9, sense: 1, headingRad: Double.pi / 2), -Double.pi / 2, accuracy: 1e-12,
                       "turning to bin 9 brings it to the top")
        let arc = g.binArc(3, sense: -1, headingRad: 0.2)
        let centre = -Double.pi / 2 - (3 * step - 0.2)
        XCTAssertEqual(arc.start, centre - step * 0.52, accuracy: 1e-12)
        XCTAssertEqual(arc.end, centre + step * 0.52, accuracy: 1e-12)
        // Colours, rounded as the page rounds them.
        typealias C = WorldChromeRingGeometry.Colour
        XCTAssertEqual(g.colour(support: 0), C(red: 154, green: 160, blue: 168, alpha: 0.14))
        XCTAssertEqual(g.colour(support: 0.01), C(red: 154, green: 160, blue: 168, alpha: 0.14))
        XCTAssertEqual(g.colour(support: 1), C(red: 100, green: 210, blue: 255, alpha: 0.9))
        XCTAssertEqual(g.colour(support: 0.5), C(red: 110, green: 195, blue: 228, alpha: 0.55))
        XCTAssertEqual(g.support(nil, 4), 0, "no support field yet: every bin dim")
        XCTAssertEqual(g.support(Array(repeating: 0.25, count: 36), 4), 0.25)
        XCTAssertEqual(g.disc, C(red: 8, green: 10, blue: 13, alpha: 0.58))
        XCTAssertEqual(g.sector, C(red: 255, green: 255, blue: 255, alpha: 0.10))
        XCTAssertEqual(g.pointer, C(red: 232, green: 233, blue: 236, alpha: 0.9))
        XCTAssertEqual(g.centre, C(red: 232, green: 233, blue: 236, alpha: 0.62))
        let spread = g.sectorArc(halfFovRad: 0.4)
        XCTAssertEqual(spread.start, -Double.pi / 2 - 0.4, accuracy: 1e-12)
        XCTAssertEqual(spread.end, -Double.pi / 2 + 0.4, accuracy: 1e-12)
        let points = g.pointerPoints(centre: 29)
        XCTAssertEqual(points.map(\.x), [29, 25.6, 32.4])
        XCTAssertEqual(points.map(\.y), [2.5, -2.5, -2.5])
    }

    // MARK: I14, I15: words

    private static func chromeSources() throws -> [(String, String)] {
        let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Glasses/Workspaces/WorldBuilder")
        return try ["WorldChromeProtocol.swift", "WorldChromeBridge.swift", "WorldChromeOverlay.swift"].map {
            ($0, try String(contentsOf: folder.appendingPathComponent($0), encoding: .utf8))
        }
    }

    func testTheChromeSourcesContainNoneOfThePagesWords() throws {
        let pageWords = ["Best view", "Face the room", "Face the area", "Previous recorded view",
                         "Next recorded view", "Reset: return", "Which directions", "reconstructed", "YOU",
                         "Not reconstructed", "Tap to turn back", "Movement stops here", "One moment",
                         "About", "Less"]
        let sources = try Self.chromeSources()
        XCTAssertEqual(sources.count, 3)
        for (file, text) in sources {
            XCTAssertGreaterThan(text.count, 1000, file)
            for word in pageWords {
                XCTAssertFalse(text.contains(word), "\(file) holds the page's word \"\(word)\"")
            }
        }
    }

    func testTheIOSOwnedChromeWordsMakeNoCaptureAbsenceClaim() {
        let banned = ["not photographed", "not captured", "never looked", "nobody photographed",
                      "never photographed", "was not seen"]
        let words = WorldChromeStyle.iOSWords + [
            WorldRenderScene.areaNoLongerServedSentence,
            "A newer reconstruction is ready. Show it",
            "A newer reconstruction could not be drawn on this phone. Try again",
        ]
        XCTAssertTrue(words.contains("Areas"))
        XCTAssertTrue(words.contains("Back to the room"))
        for word in words {
            for phrase in banned {
                XCTAssertFalse(word.lowercased().contains(phrase), "\"\(word)\" says \"\(phrase)\"")
            }
        }
    }

    // MARK: I16: the header

    func testTheAssetClientCapturesTheImageryHeaders() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ImageryStubProtocol.self]
        let client = WorldAssetClient(baseURL: URL(string: "http://tower.test")!,
                                      session: URLSession(configuration: configuration))
        ImageryStubProtocol.headers = ["Content-Type": "application/json", "X-World-Imagery": "raw-local",
                                       "X-World-Imagery-Warning": "Research build: unredacted local capture"]
        let raw = try await client.fetch(.appearanceManifest, worldID: "w1", sessionID: "s1")
        XCTAssertEqual(raw.status, 200)
        XCTAssertEqual(raw.imagery, "raw-local")
        XCTAssertEqual(raw.imageryWarning, "Research build: unredacted local capture")
        ImageryStubProtocol.headers = ["Content-Type": "application/json", "X-World-Imagery": "redacted"]
        let product = try await client.fetch(.appearanceManifest, worldID: "w1", sessionID: "s1")
        XCTAssertEqual(product.imagery, "redacted")
        XCTAssertNil(product.imageryWarning)
        ImageryStubProtocol.headers = ["Content-Type": "application/json"]
        let older = try await client.fetch(.appearanceManifest, worldID: "w1", sessionID: "s1")
        XCTAssertNil(older.imagery)
        // The headers never reach WebKit.
        XCTAssertEqual(Set(WorldAssetSchemeHandler.responseHeaders(mimeType: "a/b", byteCount: 1).keys),
                       ["Content-Type", "Content-Length", "Cache-Control", "X-Content-Type-Options"])
    }

    // MARK: I17: actions

    func testActionsAreOnlySentFromEnabledControls() {
        let driver = Driver()
        driver.input(.pageWillLoad(echo: true))
        driver.send(Self.helloBody())
        driver.input(.pageFinished)
        XCTAssertEqual(driver.input(.tapped(.next)), [], "nothing before a state")
        driver.send(Self.stateBody(seq: 1, nonce: "N1", fields: Self.stateFields(
            buttons: ["best": false, "face": true, "previous": false, "next": true, "reset": true])))
        driver.input(.firstStateDrawn)
        let held = driver.send(Self.awaitBody(seq: 2, nonce: "N1"))
        XCTAssertEqual(held.effects, [.reply(id: held.id, .activate)])
        let waiting = driver.send(Self.awaitBody(seq: 3, nonce: "N1"))
        XCTAssertEqual(driver.input(.tapped(.best)), [], "Best is disabled")
        XCTAssertEqual(driver.input(.tapped(.previous)), [], "the first view has no previous")
        XCTAssertEqual(driver.input(.tapped(.next)), [.reply(id: waiting.id, .action(.next))])
        // Queued when no await is held, delivered on the next.
        XCTAssertEqual(driver.input(.tapped(.face)), [])
        let later = driver.send(Self.awaitBody(seq: 4, nonce: "N1"))
        XCTAssertEqual(later.effects, [.reply(id: later.id, .action(.face))])
        // Not before the picture, and not over a message.
        driver.send(Self.stateBody(seq: 5, nonce: "N1", fields: Self.stateFields(message: "Restoring the view…")))
        XCTAssertEqual(driver.input(.tapped(.reset)), [])
        driver.send(Self.stateBody(seq: 6, nonce: "N1", fields: Self.stateFields(drawn: false)))
        XCTAssertEqual(driver.input(.tapped(.reset)), [])
    }
}

/// Answers every request 200 with `headers`.
final class ImageryStubProtocol: URLProtocol {
    nonisolated(unsafe) static var headers: [String: String] = [:]

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
                                       headerFields: Self.headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data("{}".utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// A small deterministic generator (SplitMix64), so I8's sequences repeat.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
