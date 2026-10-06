//
//  WorldChromeBridge.swift
//  Glasses
//
//  U1.1 native chrome (WORLD-BUILDER-WORLDS.md §4c; IOS §10): the WebKit half.
//  The page posts to `wbChromeV1`, a script message handler WITH REPLIES in
//  the page's content world; the app evaluates no script in the page, and
//  every command it has for the page is the reply to an `await` the page
//  keeps outstanding. `WorldChromeSession` decides every reply; this file
//  only carries messages in and effects out, and publishes what the overlay
//  draws.
//

import Combine
import Foundation
import SwiftUI
import WebKit

// MARK: - The echo and the request

/// `wb-chrome=native`: asked for on the page request, answered by a
/// `<meta name="wb-chrome" content="native">` in the page's head.
nonisolated enum WorldChromeEcho {
    static let queryName = "wb-chrome"
    static let queryValue = "native"
    static let metaName = "wb-chrome"
    /// The script message handler's name (protocol 1).
    static let handlerName = "wbChromeV1"

    /// Whether `html` echoes the request: exactly `native`, in the first
    /// 4096 characters (`WorldRenderRepresentation.meta`).
    static func isOffered(in html: String) -> Bool {
        WorldRenderRepresentation.meta(named: metaName, in: html) == queryValue
    }
}

// MARK: - What the overlay draws

/// The heading, per drawn frame. Its own object, so only the ring redraws
/// at frame rate.
@MainActor
final class WorldChromeHeadingModel: ObservableObject {
    @Published private(set) var view: WorldChromeView?

    func publish(_ view: WorldChromeView?) {
        if self.view != view { self.view = view }
    }
}

/// What the native chrome shows, for one viewer. Owned by
/// `WorldRenderViewerModel`; fed by `WorldChromeBridge`.
@MainActor
final class WorldChromeModel: ObservableObject {
    @Published private(set) var mode: WorldChromeMode = .legacy
    @Published private(set) var hello: WorldChromeHello?
    @Published private(set) var state: WorldChromeState?
    /// Sticky for the life of the viewer (IOS §10). Never cleared by
    /// `raw:false`.
    @Published private(set) var researchMarker: String?
    /// Whether this screen ever drew native chrome (keeps the marker band
    /// after a fallback).
    @Published private(set) var everNative = false
    /// Whether the served header said this viewer's imagery is not redacted
    /// (`X-World-Imagery`, APPEARANCE §9). Sticky, like the marker.
    @Published private(set) var rawByHeader = false
    /// The screen fell back to today's chrome, for good.
    @Published private(set) var refusedForScreen = false
    /// Per frame; only `WorldChromeRing` observes it.
    let heading = WorldChromeHeadingModel()
    /// Bumped when a page's first state arrives; the overlay answers it with
    /// `firstStateDrawn` once that state is on screen.
    @Published private(set) var firstStateToken = 0

    /// Whether the marker's words came from the page (they win over the
    /// header's warning, which is only a stand-in until the page speaks).
    private var markerFromPage = false
    private var lastAnnounced: [Announcement: Date] = [:]
    private enum Announcement { case dark, hint, message, research }

    /// The overlay draws page-derived chrome only in this mode.
    var isDrawingNative: Bool { mode == .native && state != nil && hello != nil }

    /// Whether this viewer knows its imagery is not redacted: the served
    /// header said so, or the page raised its research marker. Sticky for the
    /// viewer. Today's caption then makes no redaction claim.
    var imageryIsRaw: Bool { rawByHeader || researchMarker != nil }

    /// The research marker on today's screen (legacy geometry): after a
    /// fallback from native chrome, and whenever the served header raised it.
    /// The header is the source that still speaks when the page's own marker
    /// does not -- a Tower serving no native chrome, a page that fell back
    /// before its first state -- so it is shown there too (IOS §10, spec
    /// C15r §3.7). The page's own marker may show beside it.
    var legacyResearchMarker: String? {
        guard everNative || rawByHeader else { return nil }
        return researchMarker
    }

    // MARK: Effects from the bridge

    func setMode(_ mode: WorldChromeMode) {
        if mode != .native {
            // A new page (pending) or today's screen (legacy): nothing of the
            // last page's chrome is drawn again.
            hello = nil
            state = nil
            heading.publish(nil)
        }
        if self.mode != mode { self.mode = mode }
        guard mode == .native else { return }
        everNative = true
        firstStateToken += 1
        if let head = state?.caption?.head {
            // Focus must not stay on a web element that has just been
            // hidden: hand it to the page's own head line.
            UIAccessibility.post(notification: .layoutChanged, argument: head)
        }
    }

    func refuseForScreen() {
        refusedForScreen = true
    }

    func publishHello(_ hello: WorldChromeHello) {
        self.hello = hello
        raiseResearch(fromPage: hello.research)
    }

    func publishState(_ state: WorldChromeState) {
        let previous = self.state
        raiseResearch(fromPage: state.research)
        if previous != state { self.state = state }
        announceChanges(from: previous, to: state)
    }

    func publishView(_ view: WorldChromeView) {
        heading.publish(view)
    }

    /// The `X-World-Imagery` header said the imagery is not redacted.
    /// `warning` is `X-World-Imagery-Warning`, verbatim; iOS writes no words
    /// of its own here.
    func raiseResearch(headerWarning warning: String?) {
        if !rawByHeader { rawByHeader = true }
        guard !markerFromPage, researchMarker == nil, let warning, !warning.isEmpty else { return }
        researchMarker = warning
        announce(.research, warning)
    }

    /// The viewer closed: the marker and everything else kept for the life of
    /// the viewer goes. The page's mode and words are the bridge's to change.
    func viewerClosed() {
        researchMarker = nil
        markerFromPage = false
        everNative = false
        rawByHeader = false
        lastAnnounced = [:]
    }

    // MARK: Private

    private func raiseResearch(fromPage research: WorldChromeResearch) {
        guard let marker = research.marker, !marker.isEmpty else { return }
        guard !markerFromPage else { return }
        markerFromPage = true
        let raisedBefore = researchMarker != nil
        researchMarker = marker
        if !raisedBefore { announce(.research, marker) }
    }

    private func announceChanges(from previous: WorldChromeState?, to state: WorldChromeState) {
        if state.dark, previous?.dark != true, let name = hello?.labels.darkName {
            announce(.dark, name)
        }
        if let hint = state.hint, previous?.hint?.text != hint.text {
            announce(.hint, hint.text)
        }
        if let message = state.message, previous?.message != message {
            announce(.message, message)
        }
    }

    /// Best effort, and at most one per four seconds of each kind.
    private func announce(_ kind: Announcement, _ text: String) {
        let now = Date()
        if let last = lastAnnounced[kind], now.timeIntervalSince(last) < 4 { return }
        lastAnnounced[kind] = now
        AccessibilityNotification.Announcement(text).post()
    }
}

// MARK: - The bridge

/// Carries WebKit's messages into `WorldChromeSession` and its effects out:
/// replies to WebKit, timers, a reload, and what `WorldChromeModel` shows.
///
/// **Every reply exactly once, on the main thread.** A reply block WebKit
/// handed over is stored by message id until the session answers it; the held
/// `await` is answered by the next command, a navigation, a kill or
/// `teardown` (`dismantleUIView`).
@MainActor
final class WorldChromeBridge {
    let model: WorldChromeModel
    private var session: WorldChromeSession
    private var replies: [Int: @MainActor (Any?, String?) -> Void] = [:]
    private var nextID = 0
    private var timers: [WorldChromeSession.Input.Timer: Task<Void, Never>] = [:]

    /// Injectable so a test can run the fallbacks in milliseconds.
    var noHelloTimeout: Duration = .seconds(5)
    var noStateTimeout: Duration = .seconds(5)
    var noDeactivateConfirmTimeout: Duration = .seconds(2)

    /// Reloads the page once when a deactivation is never confirmed. Set by
    /// the web view's coordinator.
    var reloadPage: (() -> Void)?

    init(model: WorldChromeModel, kind: WorldChromeKind, pageURL: URL?) {
        self.model = model
        self.session = WorldChromeSession(kind: kind, pageURL: pageURL)
    }

    /// Whether an `await` is held, and how many replies are owed. For tests.
    var outstandingReplies: Int { replies.count }

    func receive(_ input: WorldChromeSession.Input) {
        perform(session.receive(input))
    }

    /// One message from the page.
    func receive(_ message: WKScriptMessage, replyHandler: @escaping @MainActor (Any?, String?) -> Void) {
        let frame = WorldChromeFrame(
            isMainFrame: message.frameInfo.isMainFrame,
            url: message.frameInfo.request.url?.absoluteString,
            originProtocol: message.frameInfo.securityOrigin.protocol,
            originHost: message.frameInfo.securityOrigin.host,
            isPageWorld: message.world == WKContentWorld.page
        )
        receive(body: message.body, frame: frame, replyHandler: replyHandler)
    }

    /// The same, from its parts. Split out so the reply bookkeeping is tested
    /// without a `WKScriptMessage`.
    func receive(body: Any, frame: WorldChromeFrame, replyHandler: @escaping @MainActor (Any?, String?) -> Void) {
        let id = nextID
        nextID += 1
        replies[id] = replyHandler
        perform(session.receive(.message(id: id, frame: frame, body: WorldChromeDecoder.decode(body))))
        // The session answers or holds every message; a reply block left
        // behind would be released uncalled.
        assert(replies[id] == nil || session.isHoldingAnAwait, "a chrome message was neither answered nor held")
    }

    private func perform(_ effects: [WorldChromeSession.Effect]) {
        for effect in effects {
            switch effect {
            case .reply(let id, let command):
                guard let reply = replies.removeValue(forKey: id) else { continue }
                reply(command?.reply, nil)
            case .hold:
                break
            case .publishHello(let hello):
                model.publishHello(hello)
            case .publishState(let state):
                model.publishState(state)
            case .publishView(let view):
                model.publishView(view)
            case .setMode(let mode):
                model.setMode(mode)
            case .refuseForScreen:
                model.refuseForScreen()
            case .reloadPage:
                reloadPage?()
            case .arm(let timer):
                arm(timer)
            case .cancel(let timer):
                timers.removeValue(forKey: timer)?.cancel()
            }
        }
    }

    private func arm(_ timer: WorldChromeSession.Input.Timer) {
        timers.removeValue(forKey: timer)?.cancel()
        let duration: Duration
        switch timer {
        case .noHello: duration = noHelloTimeout
        case .noState: duration = noStateTimeout
        case .noDeactivateConfirm: duration = noDeactivateConfirmTimeout
        }
        timers[timer] = Task { [weak self] in
            try? await Task.sleep(for: duration)
            guard !Task.isCancelled, let self else { return }
            self.timers[timer] = nil
            self.receive(.timer(timer))
        }
    }
}

// MARK: - The handler WebKit holds

/// What `WKUserContentController` retains (strongly), holding the bridge
/// weakly so a closed viewer is not kept alive by its web view's
/// configuration.
final class WorldChromeMessageProxy: NSObject, WKScriptMessageHandlerWithReply {
    private weak var bridge: WorldChromeBridge?

    init(bridge: WorldChromeBridge) {
        self.bridge = bridge
        super.init()
    }

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage,
        replyHandler: @escaping @MainActor (Any?, String?) -> Void
    ) {
        guard let bridge else {
            // No viewer to answer for: a decline-shaped empty answer, so the
            // page keeps its own chrome.
            replyHandler(nil, nil)
            return
        }
        bridge.receive(message, replyHandler: replyHandler)
    }
}
