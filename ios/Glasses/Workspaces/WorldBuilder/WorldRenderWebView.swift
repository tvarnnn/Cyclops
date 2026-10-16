import Foundation
import SwiftUI
import UIKit
import WebKit

// MARK: - The web view

/// Decides what the web view may navigate to: the page it was handed, and
/// nothing else. Pure, so it is tested without a `WKWebView`.
///
/// The page is loaded from the app's private scheme
/// (`WorldAssetScheme.pageURL`), whose handler serves the string
/// `WorldRenderClient` fetched, so its only navigation is the initial one, to
/// exactly that URL. Anything else — a link the page does not contain today, a
/// `window.location` a future page might set, `about:blank`, another world's
/// page, the same page with a query — is refused, and so is a *second*
/// navigation to the page URL, which would reload it and reset the camera.
///
/// This governs navigations. Subresource loads — a `fetch`, an `<img src>`,
/// a `<script src>`, a WebSocket — are not navigations and are not seen
/// here. They are governed twice: the page's CSP allows `connect-src
/// glasses-world:` and nothing else, and the scheme handler serves only the
/// whitelist in `WorldAssetRequest`.
///
/// **One exception: the page reloading itself** (`isReload`, a main-frame
/// `.reload` of exactly the page URL). The appearance page asks for that when
/// WebKit never gives back a lost WebGL context -- it often does not -- rather
/// than sit on "Restoring…" for good. The reload is served the
/// same string from memory by the scheme handler, costs the wearer's camera,
/// and goes nowhere else.
nonisolated enum WorldRenderNavigationPolicy {
    static func allows(_ url: URL?, isInitialLoad: Bool, pageURL: URL?, isReload: Bool = false) -> Bool {
        guard isInitialLoad || isReload, let url, let pageURL else { return false }
        return url.absoluteString == pageURL.absoluteString
    }
}

/// The one `WKWebView` of one viewer, and the coordinator that drives it.
/// Owned by `WorldRenderViewerModel` as `web`, beside
/// `assets`, `chrome` and `bridge`, so the web view can MOVE between
/// containers -- the World Builder panel and the full-screen cover -- without
/// a reload: the WebContent process, the WebGL context, the camera pose and
/// the chrome bridge's session all survive a reparent, and a reparent sends
/// the page nothing.
///
/// The web view is a NON-PERSISTENT-store `WKWebView` serving the string the
/// model fetched at `glasses-world://tower/worlds/<world>/render`; nothing
/// the page does outlives the viewer.
@MainActor
final class WorldWebViewOwner {
    let coordinator: WorldRenderWebView.Coordinator
    private(set) var webView: WKWebView?

    #if DEBUG
    /// Every web view any owner holds now: the panel's memory rule is "at
    /// most one", and the tests count it.
    static var liveCount = 0
    #endif

    init(target: WorldRenderTarget, assets: WorldAssetSchemeHandler, bridge: WorldChromeBridge) {
        coordinator = WorldRenderWebView.Coordinator(target: target, assets: assets, bridge: bridge)
    }

    /// Put the web view in `container` and hand it the page. The first call
    /// builds the web view (what `makeUIView` used to do); a later call MOVES
    /// it (`removeFromSuperview` + `addSubview`). Either way the coordinator
    /// is asked to load, and it skips an identical page, so a move never
    /// reloads.
    func adopt(into container: UIView, html: String, attempt: Int, budget: Int,
               onEvent: (@MainActor @Sendable (WorldRenderPageEvent) -> Void)?) {
        let webView = self.webView ?? makeWebView()
        // Re-assigned on every adoption: the closure captures the current
        // model, while the coordinator persists.
        coordinator.onEvent = onEvent
        if webView.superview !== container {
            webView.removeFromSuperview()
            webView.frame = container.bounds
            webView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
            container.addSubview(webView)
        }
        coordinator.load(html, attempt: attempt, budget: budget, into: webView)
    }

    /// `container` is going away. The web view leaves it only if it is still
    /// there: a container that already gave it to another one releases
    /// nothing.
    func release(from container: UIView) {
        guard let webView, webView.superview === container else { return }
        webView.removeFromSuperview()
    }

    /// The web view is going away for good: the viewer closed, the page
    /// failed, or the panel's memory rule took it.
    ///
    /// Without a teardown hook there was no
    /// `stopLoading()`, no delegate cleared, no scheme task cancelled,
    /// nothing dropped. The last of those is the dangerous one -- a
    /// `WKURLSchemeTask` answered after its web view is gone raises
    /// `NSInternalInconsistencyException`, which Swift cannot catch. The
    /// imagery is NOT dropped here; `viewerClosed()` drops it.
    ///
    /// The bridge's held `await` is answered with `close` (not `deactivate`),
    /// its timers are cancelled, and its handler leaves with the web view
    /// (IOS §10). The page dies with the view, so no `state` with
    /// `active: false` is expected or waited for.
    func tearDown() {
        guard let webView else { return }
        webView.stopLoading()
        webView.navigationDelegate = nil
        coordinator.onEvent = nil
        coordinator.assets.detach()
        coordinator.bridge.receive(.teardown)
        coordinator.bridge.reloadPage = nil
        webView.configuration.userContentController.removeScriptMessageHandler(
            forName: WorldChromeEcho.handlerName, contentWorld: .page)
        webView.removeFromSuperview()
        coordinator.forgetThePage()
        self.webView = nil
        #if DEBUG
        Self.liveCount -= 1
        #endif
    }

    private func makeWebView() -> WKWebView {
        let configuration = WorldRenderWebView.makeConfiguration(assets: coordinator.assets,
                                                                 bridge: coordinator.bridge)
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = coordinator
        webView.allowsBackForwardNavigationGestures = false
        webView.allowsLinkPreview = false
        webView.isOpaque = false
        // The page's own background, so nothing changes colour between the
        // loading panel, the web view before the page paints, and the page.
        webView.backgroundColor = WorldRenderLoadingPanel.pageBackground
        // The page draws on a canvas that owns every touch (`touch-action:
        // none`) and sets `overflow: hidden`; the scroll view underneath it
        // would otherwise bounce on the first drag and steal the orbit.
        webView.scrollView.isScrollEnabled = false
        webView.scrollView.bounces = false
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        // A previous web view for this viewer may have been torn down; the
        // handler is the model's and outlives both.
        coordinator.assets.attach()
        // A deactivation the page never confirmed reloads it once (§4c).
        coordinator.bridge.reloadPage = { [weak coordinator, weak webView] in
            guard let coordinator, let webView else { return }
            coordinator.reloadForTheChrome(webView)
        }
        self.webView = webView
        #if DEBUG
        Self.liveCount += 1
        #endif
        return webView
    }
}

/// The container a `WorldWebViewOwner` puts its web view in. Inline in the
/// World Builder screen's scroll view it carries the scroll gate and,
/// under native chrome, the tap that expands the world.
final class WorldWebContainerView: UIView {
    weak var owner: WorldWebViewOwner?
    /// Whether the web view outlives this container (the panel and its
    /// cover). A screen that owns its own viewer tears the web view down
    /// with it, as before.
    var hosted = false
    private var gate: WorldInlineScrollGate?
    private var tap: UITapGestureRecognizer?
    private var onTap: (() -> Void)?

    func configure(inline: Bool, onTap: (() -> Void)?) {
        self.onTap = onTap
        if inline, gate == nil {
            let gate = WorldInlineScrollGate()
            addGestureRecognizer(gate)
            self.gate = gate
            gate.enclosingScrollView = enclosingScrollView
        } else if !inline, let gate {
            removeGestureRecognizer(gate)
            self.gate = nil
        }
        if inline, onTap != nil, tap == nil {
            let tap = UITapGestureRecognizer(target: self, action: #selector(tapped))
            tap.cancelsTouchesInView = false
            tap.delaysTouchesBegan = false
            tap.delaysTouchesEnded = false
            addGestureRecognizer(tap)
            self.tap = tap
        } else if !(inline && onTap != nil), let tap {
            removeGestureRecognizer(tap)
            self.tap = nil
        }
    }

    @objc private func tapped() { onTap?() }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        gate?.enclosingScrollView = enclosingScrollView
    }

    /// The nearest scroll view above this container: the screen's own.
    private var enclosingScrollView: UIScrollView? {
        var view = superview
        while let current = view {
            if let scroll = current as? UIScrollView { return scroll }
            view = current.superview
        }
        return nil
    }
}

/// The SwiftUI side of a viewer's web view: an empty container that holds
/// the owner's `WKWebView` while `isActive`. Two containers
/// can exist for one owner -- the panel's and the cover's -- and the active
/// one adopts the web view; a container going away only releases it, unless
/// the screen is not hosted, where it tears it down as before.
// Explicit, like its owner's: with only the module default, the compiler
// reads `onEvent` as non-Sendable when it is handed to `adopt`.
@MainActor
struct WorldRenderWebView: UIViewRepresentable {
    let owner: WorldWebViewOwner
    let html: String
    /// Which `load()` this page belongs to. A retry of the **same** page after
    /// a render failure carries a new number, which is what makes the reload
    /// happen at all; see `WorldRenderViewerModel.renderAttempt`.
    var attempt: Int = 0
    /// Which content-process-kill budget this page is under. A new value is a
    /// fresh budget; see `WorldRenderViewerModel.renderBudgetToken`.
    var budgetToken: Int = 0
    /// Whether this container holds the web view now.
    var isActive: Bool = true
    /// The web view outlives this container (`WorldInlineHost`).
    var hosted: Bool = false
    /// Inside the World Builder screen's scroll view: the scroll gate.
    var inline: Bool = false
    /// A tap on the inline world (native chrome only): expand.
    var onTap: (() -> Void)? = nil
    /// What the page did. No cycle: the coordinator holds this closure, the
    /// closure holds the model, and the model holds neither.
    var onEvent: (@MainActor @Sendable (WorldRenderPageEvent) -> Void)? = nil

    func makeCoordinator() -> Coordinator { owner.coordinator }

    /// The configuration every viewer web view is built with. Split out so
    /// the data store and the scheme registration are tested without a view.
    static func makeConfiguration(assets: WorldAssetSchemeHandler, bridge: WorldChromeBridge) -> WKWebViewConfiguration {
        let configuration = WKWebViewConfiguration()
        // The default store is persistent; this viewer uses memory only.
        configuration.websiteDataStore = .nonPersistent()
        // Must be registered before the web view exists.
        configuration.setURLSchemeHandler(assets, forURLScheme: WorldAssetScheme.name)
        // The page needs its script and nothing else: no media, no data
        // detectors turning a coordinate into a phone number.
        configuration.dataDetectorTypes = []
        configuration.allowsInlineMediaPlayback = false
        // Native chrome (WORLDS §4c): one script message handler WITH
        // REPLIES, in the page's own content world. The app evaluates no
        // script in the page. Registered here, before any load: the page
        // posts `hello` while it is still being parsed. Installed for every
        // viewer web view; a page without the echo never posts, and a stray
        // post is declined. The proxy holds the bridge weakly, because the
        // content controller retains its handlers.
        configuration.userContentController.addScriptMessageHandler(
            WorldChromeMessageProxy(bridge: bridge), contentWorld: .page, name: WorldChromeEcho.handlerName)
        return configuration
    }

    func makeUIView(context: Context) -> WorldWebContainerView {
        let container = WorldWebContainerView()
        container.backgroundColor = WorldRenderLoadingPanel.pageBackground
        container.owner = owner
        update(container)
        return container
    }

    func updateUIView(_ container: WorldWebContainerView, context: Context) {
        update(container)
    }

    private func update(_ container: WorldWebContainerView) {
        container.owner = owner
        container.hosted = hosted
        container.configure(inline: inline, onTap: onTap)
        guard isActive else { return }
        owner.adopt(into: container, html: html, attempt: attempt, budget: budgetToken, onEvent: onEvent)
    }

    /// The container is going away. Hosted: the web view only leaves it (the
    /// other container, or the host, decides its fate). Not hosted: the
    /// screen that owned it is gone, and so is the web view.
    static func dismantleUIView(_ container: WorldWebContainerView, coordinator: Coordinator) {
        guard let owner = container.owner else { return }
        if container.hosted {
            owner.release(from: container)
        } else {
            owner.tearDown()
        }
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        let target: WorldRenderTarget
        /// Serves the page and proxies this world's whitelisted routes. The
        /// MODEL owns it (`WorldRenderViewerModel.assets`) and it outlives any
        /// one web view; this is a reference, and the configuration retains it
        /// too for as long as the web view exists.
        let assets: WorldAssetSchemeHandler
        /// The one URL this web view may navigate to.
        let pageURL: URL?
        /// The native chrome's bridge, reset on every allowed main-frame
        /// navigation and every content-process kill.
        let bridge: WorldChromeBridge

        init(target: WorldRenderTarget, assets: WorldAssetSchemeHandler, bridge: WorldChromeBridge) {
            self.target = target
            self.assets = assets
            self.bridge = bridge
            // The room's page, or an area's (`WORLD-BUILDER-COMPONENTS.md`
            // §5.5): the navigation policy admits exactly this one URL.
            self.pageURL = WorldAssetScheme.pageURL(worldID: target.worldID, scope: assets.scope)
            super.init()
        }

        /// The string on screen, so `updateUIView` — called on every parent
        /// re-render — reloads only when the page actually changed.
        private var loaded: String?
        /// Which attempt that string belongs to. Without it, "Try again" after
        /// a render failure is a no-op: the html is identical, `loaded != html`
        /// is false, and nothing reloads.
        private var loadedAttempt: Int?
        /// Which budget token the string on screen carries; a change is the
        /// only thing that resets the kill budget.
        private var loadedBudgetToken: Int?
        /// Whether the one allowed navigation has been decided. Each
        /// `loadHTMLString` is one; the flag is reset when one is issued.
        private var hasDecidedInitialLoad = false
        /// What the page did, back to the model.
        var onEvent: (@MainActor (WorldRenderPageEvent) -> Void)?

        /// How many content-process kills this coordinator will put the page
        /// back after, before it stops and says so.
        ///
        /// `webViewWebContentProcessDidTerminate` used to reload
        /// unconditionally and forever. A page that runs the content process
        /// out of memory on load runs it out of memory on the reload too, so
        /// the honest reading of that loop is: an app that reloads a
        /// multi-megabyte point cloud into a dying process indefinitely, with
        /// nothing on screen and no way for the reader to learn why. Two
        /// retries is enough for a transient kill — another app spiking, a
        /// backgrounded return — and short of a loop.
        ///
        /// Counted over a **sliding window** (`terminationWindow`), **not**
        /// reset by a successful `didFinish`: a page that renders and then
        /// OOMs on the first gesture is the same failure arriving later, and
        /// resetting on `didFinish` would make the budget unbounded again for
        /// exactly that case. A page that kills the process on load kills it
        /// three times well inside a minute and is still stopped.
        ///
        /// It was counted for the life of the coordinator, and that was wrong
        /// once this screen became one a wearer keeps open for a whole walk:
        /// iOS routinely kills a backgrounded app's WebContent process, so
        /// three pocket-locks spread over twenty minutes read "too large to
        /// draw on this phone" over a world that drew fine each time.
        ///
        /// A deliberate retry resets it, because that is the reader asking for
        /// a fresh budget with full knowledge; see `load(_:attempt:into:)`.
        private static let reloadBudget = 2
        nonisolated static let terminationWindow: TimeInterval = 60
        private var terminationTimes: [Date] = []

        /// The terminations that still count at `now`: those less than
        /// `window` seconds old. Pure, so the window is tested without a
        /// `WKWebView`.
        nonisolated static func recentTerminations(
            _ times: [Date], now: Date, window: TimeInterval
        ) -> [Date] {
            times.filter { now.timeIntervalSince($0) < window }
        }

        /// How many times this coordinator has loaded a page into a web view.
        /// A reparent must not change it.
        private(set) var pageLoads = 0

        /// The web view went away (`WorldWebViewOwner.tearDown`): the next one
        /// loads its page whatever was on the last.
        func forgetThePage() {
            loaded = nil
            loadedAttempt = nil
            loadedBudgetToken = nil
            hasDecidedInitialLoad = false
        }

        func load(_ html: String, attempt: Int, budget: Int, into webView: WKWebView) {
            guard loaded != html || loadedAttempt != attempt else { return }
            loaded = html
            if loadedBudgetToken != budget {
                // The reader asking again, knowingly, or a page that already
                // drew once on this phone being put back. Fresh budget. A
                // forced reload of an identical string is NOT that, and carries
                // the same token.
                terminationTimes = []
                selfReloads = []
            }
            loadedAttempt = attempt
            loadedBudgetToken = budget
            reload(into: webView)
        }

        private func reload(into webView: WKWebView) {
            guard let html = loaded else { return }
            guard let pageURL else {
                // Asynchronously: `reload` is reached synchronously from
                // `makeUIView`, so reporting from here would set `@Published`
                // state inside SwiftUI's own view-construction pass ("Modifying
                // state during view update…"). Unreachable in practice -- an id
                // that will not percent-encode already fails in
                // `WorldRenderClient.url` -- but the shape was wrong.
                let report = onEvent
                Task { @MainActor in
                    report?(.failed("this world's identifier could not be made into a page address"))
                }
                return
            }
            hasDecidedInitialLoad = false
            // The handler serves THIS string for the page URL, and the session
            // the page draws bounds which appearance routes it may reach. The
            // page's own fetches follow new appearance builds without a reload.
            let session = WorldAssetSchemeHandler.session(for: target, page: html)
            if session != assets.sessionID {
                // A different walk's page. `WorldAssetMemory.entries` is keyed
                // by digest alone and `authorizedAt` was earned by the OTHER
                // session's manifest, so carrying either over would let one
                // session's authorisation answer for another's bytes. Nothing
                // wrong has ever been shown -- a hit needs two sessions to have
                // byte-identical content-addressed chunks -- but the invariant
                // was enforced by luck.
                assets.dropCache()
            }
            assets.pageHTML = html
            assets.sessionID = session
            pageLoads += 1
            webView.load(URLRequest(url: pageURL))
        }

        /// The page hid its chrome and never confirmed putting it back
        /// (WORLDS §4c): load it again, once. The screen goes back to a
        /// bounded wait, as for the page's own reload, and the reloaded
        /// page's `hello` is declined.
        func reloadForTheChrome(_ webView: WKWebView) {
            onEvent?(.reloadingItself)
            reload(into: webView)
        }

        /// Whether the page string on screen echoes `wb-chrome=native`.
        private var offersNativeChrome: Bool {
            loaded.map(WorldChromeEcho.isOffered(in:)) ?? false
        }

        /// How many times the PAGE may put itself back, within the same window
        /// as the kill budget.
        ///
        /// The appearance page reloads itself when WebKit never gives back a
        /// lost WebGL context. Two things about that navigation
        /// are not in WebKit's gift to promise: that a script-initiated reload
        /// arrives as `.reload` rather than `.other` (it maps from
        /// `FrameLoadType::Reload`, which *should*, and the SDK does not say
        /// so), and that it arrives at all. Cancelling it because the type was
        /// the other one leaves the wearer on "Restoring…" for good, which is
        /// the exact failure the reload exists to prevent.
        ///
        /// So both types are accepted for a main-frame navigation to **exactly**
        /// the page URL after the initial load -- and counted, because "accept
        /// `.other` to the page URL" without a bound is an invitation to a
        /// reload loop that nothing else would stop. Three inside a minute is
        /// well past any real recovery.
        private static let selfReloadBudget = 3
        private var selfReloads: [Date] = []

        /// Allows only the initial page navigation or a bounded reload to the
        /// same URL. It checks frame, navigation type, URL and reload budget,
        /// resets the chrome bridge for an allowed document, then answers the
        /// delegate. The bound lets WebGL recovery proceed without a reload loop.
        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void
        ) {
            let isMainFrame = navigationAction.targetFrame?.isMainFrame == true
            let isInitialLoad = !hasDecidedInitialLoad
                && navigationAction.navigationType == .other
                && isMainFrame
            let now = Date()
            let couldBeSelfReload = hasDecidedInitialLoad && isMainFrame
                && (navigationAction.navigationType == .reload
                    || navigationAction.navigationType == .other)
            let recent = Self.recentTerminations(selfReloads, now: now, window: Self.terminationWindow)
            let isReload = couldBeSelfReload && recent.count < Self.selfReloadBudget
            let allowed = WorldRenderNavigationPolicy.allows(
                navigationAction.request.url, isInitialLoad: isInitialLoad, pageURL: pageURL,
                isReload: isReload
            )
            if allowed {
                hasDecidedInitialLoad = true
                // A new document: the held `await` is answered, the overlay
                // goes, and the next `hello` starts again (IOS §10).
                bridge.receive(.pageWillLoad(echo: offersNativeChrome))
                if isReload {
                    selfReloads = recent + [now]
                    // The page on screen is gone until this finishes, so the
                    // screen goes back to a bounded wait with the overlay over
                    // it rather than claiming to be ready over a black
                    // rectangle.
                    onEvent?(.reloadingItself)
                }
            }
            decisionHandler(allowed ? .allow : .cancel)
        }

        /// The page loaded and its script ran. The one signal that says the
        /// black rectangle became a world.
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            // `.rendered` asks for the opaque rendering panel to go. It is
            // still on screen in this turn: SwiftUI removes it with its next
            // commit, so the chrome is told the page finished only after that
            // (`WorldRenderCover`), never from here. Told now, `activate` could
            // reach the page -- and the page hide its own chrome -- while the
            // native chrome under the panel is still covered (WORLDS §4c).
            onEvent?(.rendered)
        }

        /// The load failed after it had started.
        func webView(
            _ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error
        ) {
            report(error)
        }

        /// The load failed before it had started. Both are reported the same
        /// way: from the reader's side there is no difference between a page
        /// that never began and one that stopped.
        func webView(
            _ webView: WKWebView,
            didFailProvisionalNavigation navigation: WKNavigation!,
            withError error: Error
        ) {
            report(error)
        }

        /// A failure worth telling the reader about, or nothing.
        ///
        /// **Cancellations are not failures here.** `decidePolicyFor` refuses
        /// every navigation but the first, by design, and a refusal surfaces as
        /// a provisional-navigation failure with `NSURLErrorCancelled` or
        /// WebKit's "frame load interrupted by policy change". Reporting those
        /// would turn the navigation policy doing its job into "the page could
        /// not be drawn" — on a page that is on screen and working.
        ///
        /// The WebKit one is spelled out by hand because the public `WKError`
        /// enum has no case for it: WebKit reports a policy refusal under its
        /// legacy `WebKitErrorDomain` with code 102
        /// (`WebKitErrorFrameLoadInterruptedByPolicyChange`), and neither the
        /// domain string nor the code is exported to Swift on iOS. The first
        /// version of this line named a `WKError.Code.frameLoadInterrupted`
        /// that does not exist; use the legacy domain and code instead.
        nonisolated static let webKitLegacyErrorDomain = "WebKitErrorDomain"
        nonisolated static let webKitFrameLoadInterruptedByPolicyChange = 102

        /// Whether a navigation error is the policy refusing a navigation
        /// (not a failure) rather than the page failing to load (a failure).
        nonisolated static func isNavigationCancellation(_ error: Error) -> Bool {
            let ns = error as NSError
            return (ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled)
                || (ns.domain == webKitLegacyErrorDomain
                    && ns.code == webKitFrameLoadInterruptedByPolicyChange)
        }

        private func report(_ error: Error) {
            guard !Self.isNavigationCancellation(error) else { return }
            onEvent?(.failed((error as NSError).localizedDescription))
        }

        /// WebKit's content process was killed — under memory pressure, a
        /// multi-megabyte page redrawing 80k points on every gesture is a
        /// candidate — and the view is now blank. The string is still here,
        /// so put it back rather than leaving the reader a black rectangle
        /// with nothing to do but Close and reopen.
        ///
        /// **Up to `reloadBudget` times within `terminationWindow`.** Past that
        /// the page is not put back and the reader is told, because a page
        /// that kills the process on load kills it again on the reload, and an
        /// unbounded loop of that is a blank screen plus a warm phone.
        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            let now = Date()
            terminationTimes = Self.recentTerminations(
                terminationTimes, now: now, window: Self.terminationWindow
            )
            terminationTimes.append(now)
            let terminations = terminationTimes.count
            guard terminations <= Self.reloadBudget else {
                // The page is gone and is not coming back: its held `await`
                // is answered, and nothing native is drawn over the failure.
                bridge.receive(.pageWillLoad(echo: false))
                onEvent?(.gaveUpAfterTerminations(terminations))
                return
            }
            // The page that held the chrome is gone (IOS §10).
            bridge.receive(.pageWillLoad(echo: offersNativeChrome))
            // Announced before the reload, so the screen is back in a bounded
            // `.rendering` before the page has a chance to die again silently.
            onEvent?(.reloadingAfterTermination)
            reload(into: webView)
        }
    }
}
