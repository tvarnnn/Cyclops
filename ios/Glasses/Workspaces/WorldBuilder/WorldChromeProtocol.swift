//
//  WorldChromeProtocol.swift
//  Glasses
//
//  U1.1 native chrome (WORLD-BUILDER-WORLDS.md §4c, protocol 1; IOS §10):
//  the pure half. The message types, the decoder and its bounds, the frame
//  check, the ring's drawing rule, and the reducer that decides every reply.
//  No WebKit and no SwiftUI here, so all of it is driven by unit tests.
//
//  The appearance page posts what its own chrome shows; the phone draws it.
//  Every word the phone draws for the page comes from the page, verbatim
//  (`hello.labels`, `state`). This file holds none of the page's words, and
//  `WorldChromeTests` scans it to keep it that way.
//

import Foundation

// MARK: - What the page says (Tables H, S, V)

/// One bar control: its visible words and its accessible name.
nonisolated struct WorldChromeLabel: Equatable, Sendable {
    let text: String
    let name: String
}

/// Table H's `labels`: every word of the page's chrome, from the page.
nonisolated struct WorldChromeLabels: Equatable, Sendable {
    let best, face, previous, next, reset: WorldChromeLabel
    let ringName: String, ringLabel: String, ringCenter: String
    let darkTitle: String, darkTap: String, darkName: String
    let edge: String
    let aboutOpen: String, aboutClose: String
}

nonisolated enum WorldChromeKind: String, Equatable, Sendable { case room, area }

/// APPEARANCE §6.6: whether the imagery is unredacted, and the marker's words.
nonisolated struct WorldChromeResearch: Equatable, Sendable {
    let raw: Bool
    let marker: String?
}

nonisolated struct WorldChromeHello: Equatable, Sendable {
    let pageID: String
    let kind: WorldChromeKind
    let labels: WorldChromeLabels
    let research: WorldChromeResearch
}

/// The page's own `S.phase`.
nonisolated enum WorldChromePhase: String, Equatable, Sendable {
    case starting, loading, ready, withdrawn, failed
}

/// Which of the five bar controls are enabled.
nonisolated struct WorldChromeButtons: Equatable, Sendable {
    let best, face, previous, next, reset: Bool

    func isEnabled(_ action: WorldChromeAction) -> Bool {
        switch action {
        case .best: return best
        case .face: return face
        case .previous: return previous
        case .next: return next
        case .reset: return reset
        }
    }
}

nonisolated enum WorldChromeEdge: String, Equatable, Sendable { case left, right }

nonisolated struct WorldChromeHint: Equatable, Sendable {
    let text: String
    let opacity: Double
}

/// The ring: whether it is shown, the 36 bins' support (`nil` before the
/// page has a support field: every bin dim), and the screen direction of a
/// rising yaw.
nonisolated struct WorldChromeRing: Equatable, Sendable {
    let shown: Bool
    let lit: [Double]?
    let sense: Int
}

nonisolated struct WorldChromeSection: Equatable, Sendable {
    let title: String
    let body: String
}

/// The page's caption, verbatim: the head line, then (in the panel) the
/// count line, the titled sections and the tail.
nonisolated struct WorldChromeCaption: Equatable, Sendable {
    let head: String
    let line: String?
    let sections: [WorldChromeSection]
    let tail: String?
}

/// Table S: a complete snapshot of what the page's chrome shows.
nonisolated struct WorldChromeState: Equatable, Sendable {
    let active: Bool
    let phase: WorldChromePhase
    let drawn: Bool
    let holding: Bool
    let restoring: Bool
    let status: String?
    let message: String?
    let hint: WorldChromeHint?
    let dark: Bool
    let edge: WorldChromeEdge?
    let buttons: WorldChromeButtons
    let walk: String?
    let ring: WorldChromeRing
    let caption: WorldChromeCaption?
    let research: WorldChromeResearch
}

/// Table V: the heading and the view's horizontal half-spread, per frame.
nonisolated struct WorldChromeView: Equatable, Sendable {
    let headingRad: Double
    let halfFovRad: Double
}

/// Table C's `action` names: the five bar controls. The ring, the edge
/// chevron and the dark line are all `face`.
nonisolated enum WorldChromeAction: String, Equatable, Sendable, CaseIterable {
    case best, face, previous, next, reset
}

/// Table C: the phone's one command per reply.
nonisolated enum WorldChromeCommand: Equatable, Sendable {
    case welcome(nonce: String), decline, activate, action(WorldChromeAction), deactivate, close

    /// The reply value handed to WebKit (bridged to an `NSDictionary`).
    var reply: [String: Any] {
        switch self {
        case .welcome(let nonce):
            return ["v": 1, "type": "welcome", "protocol": WorldChromeLimits.protocolVersion, "nonce": nonce]
        case .decline:
            return ["v": 1, "type": "decline"]
        case .activate:
            return ["v": 1, "type": "activate"]
        case .action(let action):
            return ["v": 1, "type": "action", "name": action.rawValue]
        case .deactivate:
            return ["v": 1, "type": "deactivate"]
        case .close:
            return ["v": 1, "type": "close"]
        }
    }
}

/// Which screen the viewer shows.
nonisolated enum WorldChromeMode: Equatable, Sendable {
    /// No echo, a refusal, or a fallback: today's 0f1249e screen, unchanged.
    case legacy
    /// The echo is in the page and nothing valid has arrived yet: native
    /// geometry, nothing page-derived drawn.
    case pending
    /// A valid hello and state have arrived: the phone draws the chrome
    /// (activate follows the first draw).
    case native
}

/// One decoded page message. Every message carries the page's `pageId`; the
/// ones after `welcome` carry the phone's nonce too.
nonisolated enum WorldChromeMessage: Equatable, Sendable {
    case hello(WorldChromeHello, seq: Int)
    case state(WorldChromeState, pageID: String, seq: Int, nonce: String)
    case view(WorldChromeView, pageID: String, seq: Int, nonce: String)
    case await(pageID: String, seq: Int, nonce: String)

    var pageID: String {
        switch self {
        case .hello(let hello, _): return hello.pageID
        case .state(_, let pageID, _, _), .view(_, let pageID, _, _), .await(let pageID, _, _): return pageID
        }
    }

    var isAwait: Bool {
        if case .await = self { return true }
        return false
    }

    var isHello: Bool {
        if case .hello = self { return true }
        return false
    }
}

/// Why a message was refused. A refused message is answered (empty, or a
/// decline for a `hello`) and costs the screen its native chrome.
nonisolated enum WorldChromeRefusal: Error, Equatable, Sendable {
    case notADictionary, tooLarge(Int), wrongVersion, unknownType(String), protocolUnsupported,
         missing(String), tooLong(String), notFinite(String), badCount(String), wrongFrame,
         wrongKind, pageMismatch, nonceMismatch, seqRegressed
}

// MARK: - Bounds (Tables H, S, V; counted as the page counts them)

/// The contract's bounds. Strings are counted in UTF-16 code units, which is
/// how the page cuts them (`clamp`), so a string the page sent within bounds
/// is never refused here.
nonisolated enum WorldChromeLimits {
    static let protocolVersion = 1

    static let helloBytes = 32 * 1024
    /// Not the draft's first 24 KiB: the WORLDS v5 draft sizes `state` by its
    /// UTF-16 string bounds, and the Tower (886eed0) asks for no cap below
    /// 64 KiB.
    static let stateBytes = 64 * 1024
    static let viewBytes = 512
    static let awaitBytes = 256

    static let text = 40, name = 120
    static let ringName = 120, ringLabel = 60, ringCenter = 8
    static let darkTitle = 80, darkTap = 80, darkName = 160
    static let edge = 80, toggle = 20, marker = 200
    static let status = 200, message = 800, hint = 200, walk = 24
    static let head = 300, line = 600, sectionTitle = 80, sectionBody = 1600, tail = 200
    static let sections = 1...8
    static let ringBins = 36
    static let pageID = 8...64
    static let nonce = 1...128
}

// MARK: - The decoder

nonisolated enum WorldChromeDecoder {
    /// A WebKit message body, as protocol 1 says it must look, or why not.
    /// Never interprets text: every string is shown with `Text(verbatim:)`.
    static func decode(_ body: Any) -> Result<WorldChromeMessage, WorldChromeRefusal> {
        do {
            return .success(try message(body))
        } catch let refusal as WorldChromeRefusal {
            return .failure(refusal)
        } catch {
            return .failure(.notADictionary)
        }
    }

    private static func message(_ body: Any) throws -> WorldChromeMessage {
        guard let object = body as? [String: Any] else { throw WorldChromeRefusal.notADictionary }
        // A non-finite number cannot be JSON, but WebKit hands one over as an
        // NSNumber; name it rather than calling the whole message malformed.
        if let path = nonFinitePath(object, path: "") { throw WorldChromeRefusal.notFinite(path) }
        guard JSONSerialization.isValidJSONObject(object) else { throw WorldChromeRefusal.notADictionary }
        let r = Fields(object, path: "")
        guard r.integer("v") == 1 else { throw WorldChromeRefusal.wrongVersion }
        guard let type = object["type"] as? String else { throw WorldChromeRefusal.missing("type") }
        let limit: Int
        switch type {
        case "hello": limit = WorldChromeLimits.helloBytes
        case "state": limit = WorldChromeLimits.stateBytes
        case "view": limit = WorldChromeLimits.viewBytes
        case "await": limit = WorldChromeLimits.awaitBytes
        default: throw WorldChromeRefusal.unknownType(String(type.prefix(40)))
        }
        let size = (try? JSONSerialization.data(withJSONObject: object).count) ?? Int.max
        guard size <= limit else { throw WorldChromeRefusal.tooLarge(size) }
        let pageID = try r.pageID()
        guard let seq = r.integer("seq"), seq >= 0 else { throw WorldChromeRefusal.missing("seq") }

        switch type {
        case "hello":
            guard let versions = object["protocol"] as? [Any] else { throw WorldChromeRefusal.missing("protocol") }
            guard versions.contains(where: { Fields.integerValue($0) == WorldChromeLimits.protocolVersion }) else {
                throw WorldChromeRefusal.protocolUnsupported
            }
            guard let kindWord = object["kind"] as? String, let kind = WorldChromeKind(rawValue: kindWord) else {
                throw WorldChromeRefusal.missing("kind")
            }
            let labels = try Self.labels(r.object("labels"))
            let research = try Self.research(r.object("research"))
            return .hello(WorldChromeHello(pageID: pageID, kind: kind, labels: labels, research: research), seq: seq)
        case "state":
            let nonce = try r.nonce()
            return .state(try state(r), pageID: pageID, seq: seq, nonce: nonce)
        case "view":
            let nonce = try r.nonce()
            guard let heading = r.number("headingRad") else { throw WorldChromeRefusal.missing("headingRad") }
            guard heading.isFinite else { throw WorldChromeRefusal.notFinite("headingRad") }
            guard let half = r.number("halfFovRad") else { throw WorldChromeRefusal.missing("halfFovRad") }
            guard half.isFinite, half > 0, half < Double.pi / 2 else {
                throw WorldChromeRefusal.notFinite("halfFovRad")
            }
            return .view(WorldChromeView(headingRad: heading, halfFovRad: half),
                         pageID: pageID, seq: seq, nonce: nonce)
        default:
            let nonce = try r.nonce()
            return .await(pageID: pageID, seq: seq, nonce: nonce)
        }
    }

    /// The dotted path of the first NaN or infinity in `value`, if any.
    private static func nonFinitePath(_ value: Any, path: String) -> String? {
        switch value {
        case let object as [String: Any]:
            for key in object.keys.sorted() {
                if let found = nonFinitePath(object[key] as Any, path: path.isEmpty ? key : path + "." + key) {
                    return found
                }
            }
        case let array as [Any]:
            for element in array {
                if let found = nonFinitePath(element, path: path) { return found }
            }
        case let number as NSNumber:
            if CFGetTypeID(number) != CFBooleanGetTypeID(), !number.doubleValue.isFinite { return path }
        default:
            break
        }
        return nil
    }

    private static func label(_ r: Fields) throws -> WorldChromeLabel {
        WorldChromeLabel(text: try r.string("text", max: WorldChromeLimits.text),
                         name: try r.string("name", max: WorldChromeLimits.name))
    }

    static func labels(_ r: Fields) throws -> WorldChromeLabels {
        let ring = try r.object("ring")
        let dark = try r.object("dark")
        let toggle = try r.object("about")
        return WorldChromeLabels(
            best: try label(r.object("best")), face: try label(r.object("face")),
            previous: try label(r.object("previous")), next: try label(r.object("next")),
            reset: try label(r.object("reset")),
            ringName: try ring.string("name", max: WorldChromeLimits.ringName),
            ringLabel: try ring.string("label", max: WorldChromeLimits.ringLabel),
            ringCenter: try ring.string("center", max: WorldChromeLimits.ringCenter),
            darkTitle: try dark.string("title", max: WorldChromeLimits.darkTitle),
            darkTap: try dark.string("tap", max: WorldChromeLimits.darkTap),
            darkName: try dark.string("name", max: WorldChromeLimits.darkName),
            edge: try r.string("edge", max: WorldChromeLimits.edge),
            aboutOpen: try toggle.string("open", max: WorldChromeLimits.toggle),
            aboutClose: try toggle.string("close", max: WorldChromeLimits.toggle)
        )
    }

    static func research(_ r: Fields) throws -> WorldChromeResearch {
        WorldChromeResearch(raw: try r.bool("raw"),
                            marker: try r.optionalString("marker", max: WorldChromeLimits.marker))
    }

    private static func state(_ r: Fields) throws -> WorldChromeState {
        guard let phaseWord = r.raw("phase") as? String, let phase = WorldChromePhase(rawValue: phaseWord) else {
            throw WorldChromeRefusal.missing("phase")
        }
        let hint: WorldChromeHint?
        if let h = try r.optionalObject("hint") {
            guard let opacity = h.number("opacity") else { throw WorldChromeRefusal.missing("hint.opacity") }
            guard opacity.isFinite else { throw WorldChromeRefusal.notFinite("hint.opacity") }
            hint = WorldChromeHint(text: try h.string("text", max: WorldChromeLimits.hint),
                                   opacity: min(1, max(0, opacity)))
        } else {
            hint = nil
        }
        let edge: WorldChromeEdge?
        switch r.raw("edge") {
        case nil, is NSNull: edge = nil
        case let word as String:
            guard let side = WorldChromeEdge(rawValue: word) else { throw WorldChromeRefusal.missing("edge") }
            edge = side
        default: throw WorldChromeRefusal.missing("edge")
        }
        let b = try r.object("buttons")
        let buttons = WorldChromeButtons(best: try b.bool("best"), face: try b.bool("face"),
                                         previous: try b.bool("previous"), next: try b.bool("next"),
                                         reset: try b.bool("reset"))
        let ringFields = try r.object("ring")
        let lit: [Double]?
        switch ringFields.raw("lit") {
        case nil, is NSNull: lit = nil
        case let values as [Any]:
            guard values.count == WorldChromeLimits.ringBins else { throw WorldChromeRefusal.badCount("ring.lit") }
            lit = try values.map { value in
                guard let number = Fields.numberValue(value) else { throw WorldChromeRefusal.missing("ring.lit") }
                guard number.isFinite else { throw WorldChromeRefusal.notFinite("ring.lit") }
                return number
            }
        default: throw WorldChromeRefusal.missing("ring.lit")
        }
        guard let sense = ringFields.integer("sense"), sense == 1 || sense == -1 else {
            throw WorldChromeRefusal.missing("ring.sense")
        }
        let ring = WorldChromeRing(shown: try ringFields.bool("shown"), lit: lit, sense: sense)
        let caption: WorldChromeCaption?
        if let c = try r.optionalObject("caption") {
            guard let raw = c.raw("sections") as? [Any] else { throw WorldChromeRefusal.missing("caption.sections") }
            guard WorldChromeLimits.sections.contains(raw.count) else {
                throw WorldChromeRefusal.badCount("caption.sections")
            }
            let sections = try raw.map { entry -> WorldChromeSection in
                guard let section = entry as? [String: Any] else { throw WorldChromeRefusal.missing("caption.sections") }
                let s = Fields(section, path: "caption.sections.")
                return WorldChromeSection(title: try s.string("title", max: WorldChromeLimits.sectionTitle),
                                          body: try s.string("body", max: WorldChromeLimits.sectionBody))
            }
            caption = WorldChromeCaption(head: try c.string("head", max: WorldChromeLimits.head),
                                         line: try c.optionalString("line", max: WorldChromeLimits.line),
                                         sections: sections,
                                         tail: try c.optionalString("tail", max: WorldChromeLimits.tail))
        } else {
            caption = nil
        }
        return WorldChromeState(
            active: try r.bool("active"), phase: phase, drawn: try r.bool("drawn"),
            holding: try r.bool("holding"), restoring: try r.bool("restoring"),
            status: try r.optionalString("status", max: WorldChromeLimits.status),
            message: try r.optionalString("message", max: WorldChromeLimits.message),
            hint: hint, dark: try r.bool("dark"), edge: edge, buttons: buttons,
            walk: try r.optionalString("walk", max: WorldChromeLimits.walk),
            ring: ring, caption: caption, research: try research(r.object("research"))
        )
    }

    /// One JSON object's fields, with the refusals named by their path.
    nonisolated struct Fields {
        let object: [String: Any]
        let path: String

        init(_ object: [String: Any], path: String) {
            self.object = object
            self.path = path
        }

        func raw(_ key: String) -> Any? { object[key] }

        func object(_ key: String) throws -> Fields {
            guard let nested = object[key] as? [String: Any] else { throw WorldChromeRefusal.missing(path + key) }
            return Fields(nested, path: path + key + ".")
        }

        /// `nil` for JSON null or an absent field.
        func optionalObject(_ key: String) throws -> Fields? {
            switch object[key] {
            case nil, is NSNull: return nil
            case let nested as [String: Any]: return Fields(nested, path: path + key + ".")
            default: throw WorldChromeRefusal.missing(path + key)
            }
        }

        func string(_ key: String, max: Int) throws -> String {
            guard let value = object[key] as? String else { throw WorldChromeRefusal.missing(path + key) }
            guard value.utf16.count <= max else { throw WorldChromeRefusal.tooLong(path + key) }
            return value
        }

        func optionalString(_ key: String, max: Int) throws -> String? {
            switch object[key] {
            case nil, is NSNull: return nil
            case let value as String:
                guard value.utf16.count <= max else { throw WorldChromeRefusal.tooLong(path + key) }
                return value
            default: throw WorldChromeRefusal.missing(path + key)
            }
        }

        /// A JSON boolean, and only a boolean: `1` is not `true`.
        func bool(_ key: String) throws -> Bool {
            guard let number = object[key] as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw WorldChromeRefusal.missing(path + key)
            }
            return number.boolValue
        }

        func number(_ key: String) -> Double? { Self.numberValue(object[key]) }

        func integer(_ key: String) -> Int? { Self.integerValue(object[key]) }

        func pageID() throws -> String {
            guard let id = object["pageId"] as? String else { throw WorldChromeRefusal.missing("pageId") }
            guard WorldChromeLimits.pageID.contains(id.utf8.count),
                  id.utf8.allSatisfy(Self.isPageIDByte)
            else { throw WorldChromeRefusal.badCount("pageId") }
            return id
        }

        func nonce() throws -> String {
            guard let nonce = object["nonce"] as? String else { throw WorldChromeRefusal.missing("nonce") }
            guard WorldChromeLimits.nonce.contains(nonce.utf16.count) else { throw WorldChromeRefusal.tooLong("nonce") }
            return nonce
        }

        /// A JSON number that is not a boolean.
        static func numberValue(_ value: Any?) -> Double? {
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
            return number.doubleValue
        }

        static func integerValue(_ value: Any?) -> Int? {
            guard let number = numberValue(value), number.isFinite, number.rounded() == number,
                  abs(number) < 9_007_199_254_740_992 else { return nil }
            return Int(number)
        }

        /// `[A-Za-z0-9_-]`
        static func isPageIDByte(_ byte: UInt8) -> Bool {
            (48...57).contains(byte) || (65...90).contains(byte) || (97...122).contains(byte)
                || byte == 95 || byte == 45
        }
    }
}

// MARK: - Where a message came from

/// The parts of a `WKScriptMessage` the bridge admits a message on.
nonisolated struct WorldChromeFrame: Equatable, Sendable {
    let isMainFrame: Bool
    let url: String?
    let originProtocol: String
    let originHost: String
    let isPageWorld: Bool

    /// Main frame, exactly the page URL, origin `glasses-world://tower`, the
    /// page's content world.
    func admits(pageURL: URL?) -> Bool {
        guard let pageURL, isMainFrame, isPageWorld, url == pageURL.absoluteString else { return false }
        let scheme = originProtocol.hasSuffix(":") ? String(originProtocol.dropLast()) : originProtocol
        return scheme.lowercased() == WorldAssetScheme.name && originHost == WorldAssetScheme.host
    }
}

// MARK: - The ring's drawing rule (WORLDS §4c)

/// The page's `drawCompass()`, as numbers: a 58 pt ring of 36 arcs centred
/// at `−π/2 + sense·(i·2π/36 − heading)`, each spanning ±0.52 of a bin.
nonisolated enum WorldChromeRingGeometry {
    static let size: Double = 58
    static let discRadius: Double = 26
    static let arcRadius: Double = 21
    static let lineWidth: Double = 5.5
    static let sectorRadius: Double = 16.5
    static var step: Double { 2 * Double.pi / Double(WorldChromeLimits.ringBins) }

    /// An RGBA colour with 0–255 channels and a 0–1 alpha, as the page writes it.
    nonisolated struct Colour: Equatable, Sendable {
        let red: Double, green: Double, blue: Double, alpha: Double
    }

    static let disc = Colour(red: 8, green: 10, blue: 13, alpha: 0.58)
    static let sector = Colour(red: 255, green: 255, blue: 255, alpha: 0.10)
    static let pointer = Colour(red: 232, green: 233, blue: 236, alpha: 0.9)
    static let centre = Colour(red: 232, green: 233, blue: 236, alpha: 0.62)

    /// Bin `i`'s centre, in radians clockwise from +x with y down.
    static func binAngle(_ i: Int, sense: Int, headingRad: Double) -> Double {
        -Double.pi / 2 + Double(sense) * (Double(i) * step - headingRad)
    }

    /// Bin `i`'s arc.
    static func binArc(_ i: Int, sense: Int, headingRad: Double) -> (start: Double, end: Double) {
        let centre = binAngle(i, sense: sense, headingRad: headingRad)
        return (centre - step * 0.52, centre + step * 0.52)
    }

    /// Bin `i`'s support: `lit[i]`, or 0 before the page has a support field.
    static func support(_ lit: [Double]?, _ i: Int) -> Double {
        guard let lit, lit.indices.contains(i) else { return 0 }
        return min(1, max(0, lit[i]))
    }

    /// The page's arc colour for support `k`, rounded as the page rounds it.
    static func colour(support k: Double) -> Colour {
        if k <= 0.01 { return Colour(red: 154, green: 160, blue: 168, alpha: 0.14) }
        return Colour(red: (120 - 20 * k).rounded(), green: (180 + 30 * k).rounded(),
                      blue: (200 + 55 * k).rounded(), alpha: ((0.2 + 0.7 * k) * 1000).rounded() / 1000)
    }

    /// The view's horizontal spread, around straight up.
    static func sectorArc(halfFovRad: Double) -> (start: Double, end: Double) {
        (-Double.pi / 2 - halfFovRad, -Double.pi / 2 + halfFovRad)
    }

    /// The pointer at the top, for a ring centred at `c`.
    static func pointerPoints(centre c: Double) -> [(x: Double, y: Double)] {
        [(c, c - 26.5), (c - 3.4, c - 31.5), (c + 3.4, c - 31.5)]
    }
}

// MARK: - The reducer

/// The phone's half of protocol 1 on one screen: what to answer, what to
/// show, and when to give up. Pure; `WorldChromeBridge` executes its effects.
///
/// **Every message is answered exactly once** (I8). A message is answered
/// as it arrives, except the one outstanding `await`, which is held for the
/// next command and answered at the latest by `pageWillLoad` or `teardown`,
/// with `close`.
///
/// **Messages from a page this screen has left** (a page id retired by
/// `pageWillLoad`) are answered and otherwise ignored: the old document keeps
/// running until the new navigation commits, and its last posts are not a
/// reason to refuse the new page.
nonisolated struct WorldChromeSession: Sendable {
    nonisolated enum Input: Sendable {
        /// Every allowed main-frame navigation, and a content-process kill.
        case pageWillLoad(echo: Bool)
        case message(id: Int, frame: WorldChromeFrame, body: Result<WorldChromeMessage, WorldChromeRefusal>)
        /// The overlay committed its first frame with this page's state.
        case firstStateDrawn
        /// Only from enabled controls.
        case tapped(WorldChromeAction)
        /// `.noHello` (5 s), `.noState` (5 s), `.noDeactivateConfirm` (2 s).
        case timer(Timer)
        case teardown
        nonisolated enum Timer: Sendable, Hashable { case noHello, noState, noDeactivateConfirm }
    }

    nonisolated enum Effect: Equatable, Sendable {
        /// `nil` is an empty reply (`state`, `view`, a refused message).
        case reply(id: Int, WorldChromeCommand?)
        /// The one outstanding `await`.
        case hold(id: Int)
        case publishHello(WorldChromeHello), publishState(WorldChromeState), publishView(WorldChromeView)
        case setMode(WorldChromeMode), refuseForScreen, reloadPage, arm(Input.Timer), cancel(Input.Timer)
    }

    /// Which page this screen opened: a `hello` of the other kind is refused.
    let kind: WorldChromeKind
    /// The one URL a message may come from.
    let pageURL: URL?
    private let makeNonce: @Sendable () -> String

    /// Set by the first fallback, for the life of the screen.
    private(set) var refusedForScreen = false
    private(set) var mode: WorldChromeMode = .legacy

    private var pageID: String?
    private var nonce: String?
    private var lastSeq: Int?
    private var retiredPageIDs: Set<String> = []
    private var held: Int?
    private var queue: [WorldChromeCommand] = []
    private var lastState: WorldChromeState?
    private var activateQueued = false
    private var awaitingConfirm = false
    private var reloadedForConfirm = false
    private var armed: Set<Input.Timer> = []

    init(kind: WorldChromeKind, pageURL: URL?, makeNonce: @escaping @Sendable () -> String = { UUID().uuidString }) {
        self.kind = kind
        self.pageURL = pageURL
        self.makeNonce = makeNonce
    }

    /// Whether an `await` is being held now. For tests.
    var isHoldingAnAwait: Bool { held != nil }

    mutating func receive(_ input: Input) -> [Effect] {
        switch input {
        case .pageWillLoad(let echo):
            return pageWillLoad(echo: echo)
        case .message(let id, let frame, let body):
            return message(id: id, frame: frame, body: body)
        case .firstStateDrawn:
            guard mode == .native, !activateQueued else { return [] }
            activateQueued = true
            return enqueue(.activate)
        case .tapped(let action):
            guard mode == .native, let state = lastState, state.drawn, state.message == nil,
                  state.phase != .failed, state.buttons.isEnabled(action)
            else { return [] }
            return enqueue(.action(action))
        case .timer(let timer):
            guard armed.remove(timer) != nil else { return [] }
            switch timer {
            case .noHello, .noState:
                guard mode == .pending else { return [] }
                return fallBack()
            case .noDeactivateConfirm:
                guard awaitingConfirm, !reloadedForConfirm else { return [] }
                awaitingConfirm = false
                reloadedForConfirm = true
                return [.reloadPage]
            }
        case .teardown:
            var effects: [Effect] = []
            if let held {
                self.held = nil
                effects.append(.reply(id: held, .close))
            }
            effects += cancelTimers()
            forgetThePage()
            mode = .legacy
            return effects
        }
    }

    // MARK: Inputs

    private mutating func pageWillLoad(echo: Bool) -> [Effect] {
        var effects: [Effect] = []
        if let held {
            self.held = nil
            effects.append(.reply(id: held, .close))
        }
        effects += cancelTimers()
        forgetThePage()
        awaitingConfirm = false
        let next: WorldChromeMode = echo && !refusedForScreen ? .pending : .legacy
        mode = next
        effects.append(.setMode(next))
        if next == .pending {
            armed.insert(.noHello)
            effects.append(.arm(.noHello))
        }
        return effects
    }

    private mutating func message(
        id: Int, frame: WorldChromeFrame, body: Result<WorldChromeMessage, WorldChromeRefusal>
    ) -> [Effect] {
        if case .success(let message) = body, message.pageID != pageID,
           retiredPageIDs.contains(message.pageID) {
            return [.reply(id: id, message.isAwait ? .close : nil)]
        }
        switch mode {
        case .legacy:
            return legacyMessage(id: id, frame: frame, body: body)
        case .pending, .native:
            return liveMessage(id: id, frame: frame, body: body)
        }
    }

    /// Nothing is drawn natively: answer, confirm a deactivation, and decline
    /// any new `hello`.
    private mutating func legacyMessage(
        id: Int, frame: WorldChromeFrame, body: Result<WorldChromeMessage, WorldChromeRefusal>
    ) -> [Effect] {
        guard case .success(let message) = body else { return [.reply(id: id, nil)] }
        switch message {
        case .hello:
            return [.reply(id: id, .decline)]
        case .state(let state, let pageID, _, let nonce):
            var effects: [Effect] = [.reply(id: id, nil)]
            if awaitingConfirm, !state.active, pageID == self.pageID, nonce == self.nonce,
               frame.admits(pageURL: pageURL) {
                awaitingConfirm = false
                armed.remove(.noDeactivateConfirm)
                effects.append(.cancel(.noDeactivateConfirm))
            }
            return effects
        case .view:
            return [.reply(id: id, nil)]
        case .await:
            if !queue.isEmpty { return [.reply(id: id, dequeue())] }
            return [.reply(id: id, .close)]
        }
    }

    private mutating func liveMessage(
        id: Int, frame: WorldChromeFrame, body: Result<WorldChromeMessage, WorldChromeRefusal>
    ) -> [Effect] {
        let message: WorldChromeMessage
        switch body {
        case .failure:
            return refuse(id: id, isHello: false)
        case .success(let decoded):
            message = decoded
        }
        guard frame.admits(pageURL: pageURL) else { return refuse(id: id, isHello: message.isHello) }
        switch message {
        case .hello(let hello, let seq):
            // One hello per page load; a second one is a page this screen did
            // not see arrive.
            guard pageID == nil, mode == .pending, hello.kind == kind else { return refuse(id: id, isHello: true) }
            let nonce = makeNonce()
            pageID = hello.pageID
            self.nonce = nonce
            lastSeq = seq
            armed.remove(.noHello)
            armed.insert(.noState)
            return [.reply(id: id, .welcome(nonce: nonce)), .publishHello(hello),
                    .cancel(.noHello), .arm(.noState)]
        case .state(_, let pageID, let seq, let nonce),
             .view(_, let pageID, let seq, let nonce),
             .await(let pageID, let seq, let nonce):
            guard let current = self.pageID, pageID == current, nonce == self.nonce,
                  seq > (lastSeq ?? -1)
            else { return refuse(id: id, isHello: false) }
            lastSeq = seq
        }
        switch message {
        case .hello:
            return []
        case .state(let state, _, _, _):
            lastState = state
            var effects: [Effect] = [.reply(id: id, nil), .publishState(state)]
            if mode == .pending {
                mode = .native
                armed.remove(.noState)
                effects += [.setMode(.native), .cancel(.noState)]
            }
            return effects
        case .view(let view, _, _, _):
            return [.reply(id: id, nil), .publishView(view)]
        case .await:
            var effects: [Effect] = []
            // The page keeps one outstanding; a newer one supersedes it.
            if let old = held {
                held = nil
                effects.append(.reply(id: old, nil))
            }
            if !queue.isEmpty {
                effects.append(.reply(id: id, dequeue()))
            } else {
                held = id
                effects.append(.hold(id: id))
            }
            return effects
        }
    }

    // MARK: Helpers

    /// Answer a message this screen will not accept, and stop drawing the
    /// chrome for the life of the screen.
    private mutating func refuse(id: Int, isHello: Bool) -> [Effect] {
        [.reply(id: id, isHello ? .decline : nil)] + fallBack()
    }

    /// Back to today's screen for good. A welcomed page is told to put its
    /// own chrome back and must confirm with `active:false` -- which the
    /// contract has it post on every `deactivate`, whether or not it was
    /// activated (WORLDS §4c) -- or it is reloaded once.
    private mutating func fallBack() -> [Effect] {
        var effects: [Effect] = []
        if !refusedForScreen {
            refusedForScreen = true
            effects.append(.refuseForScreen)
        }
        mode = .legacy
        effects.append(.setMode(.legacy))
        for timer in [Input.Timer.noHello, .noState] where armed.remove(timer) != nil {
            effects.append(.cancel(timer))
        }
        queue = []
        if nonce != nil {
            if !awaitingConfirm {
                awaitingConfirm = true
                armed.insert(.noDeactivateConfirm)
                effects.append(.arm(.noDeactivateConfirm))
            }
            effects += enqueue(.deactivate)
        }
        return effects
    }

    /// Queue a command, and deliver it at once to a held `await`.
    private mutating func enqueue(_ command: WorldChromeCommand) -> [Effect] {
        queue.append(command)
        guard let held else { return [] }
        self.held = nil
        return [.reply(id: held, dequeue())]
    }

    private mutating func dequeue() -> WorldChromeCommand {
        queue.removeFirst()
    }

    private mutating func cancelTimers() -> [Effect] {
        let effects = [Input.Timer.noHello, .noState, .noDeactivateConfirm]
            .filter { armed.contains($0) }.map { Effect.cancel($0) }
        armed = []
        return effects
    }

    private mutating func forgetThePage() {
        if let pageID { retiredPageIDs.insert(pageID) }
        pageID = nil
        nonce = nil
        lastSeq = nil
        queue = []
        lastState = nil
        activateQueued = false
    }
}
