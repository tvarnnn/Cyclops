import Foundation

// MARK: - The viewer's state

/// What the viewer sheet knows. The failure carries its sentence so the view
/// composes nothing.
///
/// ## Why fetching and rendering are two states
///
/// They were one, called `.loading`, and it ended at the moment the HTTP
/// response arrived. Everything after that was unobserved: the page was handed
/// to a `WKWebView` and the app never learned whether it drew. There was no
/// `didFinish`, no `didFail`, and no timeout covering the render — the client's
/// 30 s bound is on the **fetch** — so a page that failed to execute left the
/// wearer looking at a black rectangle that is indistinguishable from a crash,
/// with no spinner, no message and no end. On a multi-megabyte point cloud on a
/// phone under memory pressure that is not a hypothetical.
///
/// So the page's own life is a state now. `.rendering` means the string is in
/// the web view and `didFinish` has not arrived; it is bounded by
/// `WorldRenderViewerModel.renderTimeout` and resolves to `.ready` or to a
/// `.failed` that says which of the two halves gave up.
enum WorldRenderViewerState: Equatable {
    /// Asking the Tower for the page. Bounded by `WorldRenderClient.timeout`.
    case fetching
    /// The page is in the web view and has not finished loading. Bounded by
    /// `WorldRenderViewerModel.renderTimeout`.
    case rendering(html: String)
    /// `didFinish` arrived: the page is on screen.
    case ready(html: String)
    case failed(message: String, retryable: Bool)

    /// The page to hand the web view, in the two states that have one.
    ///
    /// Read by the view so that **one** `WorldRenderWebView` spans both:
    /// branching on the state inside a `ViewBuilder` would put the two in
    /// different `_ConditionalContent` arms, and SwiftUI would tear down the
    /// `WKWebView` and build a new one at the exact moment the first finished
    /// rendering — throwing away the render this state exists to observe.
    var html: String? {
        switch self {
        case .rendering(let html), .ready(let html): return html
        case .fetching, .failed: return nil
        }
    }

    /// Whether the page is in the web view and has not reported finishing.
    var isRendering: Bool {
        if case .rendering = self { return true }
        return false
    }

    /// Whether the page is still being asked for. The one state in which
    /// "Reload" would do nothing but restart a fetch already in flight.
    var isFetching: Bool {
        if case .fetching = self { return true }
        return false
    }

    /// The failure's sentence, or `nil`. A property rather than an `if case`
    /// binding in the view for the reason this codebase gives everywhere else:
    /// a result-builder block with a binding in it caused trouble.
    var failureMessage: String? {
        if case .failed(let message, _) = self { return message }
        return nil
    }

    var failureIsRetryable: Bool {
        if case .failed(_, let retryable) = self { return retryable }
        return false
    }

    /// Which reconstruction the Tower's page is, once there is a page.
    var representation: WorldRenderRepresentation? {
        html.flatMap(WorldRenderRepresentation.declared(in:))
    }

    /// The revision the Tower stamped into the page on screen, if any.
    var revision: String? {
        html.flatMap { WorldRenderRepresentation.meta(named: "wb-revision", in: $0) }
    }
}

/// Which rung of the Tower's reconstruction ladder a page is.
///
/// The Tower serves the best reconstruction a session has -- a surface, then
/// dense points, then the sparse points -- and each page names its rung in a
/// `<meta name="wb-representation">` tag. It is read from the page itself,
/// rather than from a response header, so `WorldRenderClient.page(for:)` keeps
/// its signature and every stub of it keeps working.
///
/// It exists for ONE sentence: the caption under the viewer. That caption used
/// to say "Not a surface" unconditionally, which was true of every page the
/// Tower could serve until it could serve a surface. A caption that denies what
/// is on screen is the same failure as one that overclaims.
nonisolated enum WorldRenderRepresentation: String, Equatable, Sendable {
    /// The wearer's own redacted keyframes blended over the surface
    /// (`WORLD-BUILDER-APPEARANCE.md`). The top rung.
    case appearance
    case surface
    case dense
    case sparse

    /// Whether a page of rung `latest` is a better picture than one of rung
    /// `shown`: sparse (or unknown) < dense < surface < appearance. Only an
    /// improvement
    /// replaces the picture on screen by itself; a rebuild of the same rung
    /// is offered instead, because a swap reloads the page and resets the
    /// reader's camera mid-look.
    static func isUpgrade(
        from shown: WorldRenderRepresentation?, to latest: WorldRenderRepresentation?
    ) -> Bool {
        rank(latest) > rank(shown)
    }

    private static func rank(_ representation: WorldRenderRepresentation?) -> Int {
        switch representation {
        case .appearance: return 3
        case .surface: return 2
        case .dense: return 1
        case .sparse, nil: return 0
        }
    }

    /// The rung a page declares, or `nil` for a page that declares none -- every
    /// page from a Tower that predates the declaration.
    static func declared(in html: String) -> WorldRenderRepresentation? {
        meta(named: "wb-representation", in: html).flatMap(WorldRenderRepresentation.init(rawValue:))
    }

    /// The `content` of a `<meta name=...>` the Tower wrote into the head.
    ///
    /// A plain scan, not a parser: the tags are written by the Tower, in the
    /// head, in one fixed spelling, and the page can be megabytes long, so
    /// only its first few kilobytes are looked at.
    static func meta(named name: String, in html: String) -> String? {
        let head = html.prefix(4096)
        guard let marker = head.range(of: "name=\"\(name)\" content=\"") else {
            return nil
        }
        let rest = head[marker.upperBound...]
        guard let end = rest.firstIndex(of: "\"") else { return nil }
        let value = String(rest[..<end])
        return value.isEmpty ? nil : value
    }

    /// `html` with the Tower's `wb-revision` stamp taken out of its head.
    ///
    /// Two pages that differ only in that stamp are the same picture. A page
    /// composed while the Tower could not read its manifest carries no stamp
    /// (`WORLD-BUILDER-WORLDS.md` §4a rule 7), and the same page fetched a
    /// moment later carries one -- which compared unequal, so the follower
    /// swapped an identical mesh in, with the "Drawing" overlay and a camera
    /// reset.
    static func withoutRevisionStamp(_ html: String) -> String {
        guard let value = meta(named: "wb-revision", in: html),
              let range = html.prefix(4096).range(of: "<meta name=\"wb-revision\" content=\"\(value)\">")
        else { return html }
        var unstamped = html
        unstamped.removeSubrange(range)
        return unstamped
    }

    /// The session a revision names. The Tower writes every revision as
    /// `<session>/<rung revision>` (§4a); `nil` for one with no session in it.
    static func session(of revision: String?) -> String? {
        guard let revision, let slash = revision.firstIndex(of: "/") else { return nil }
        return String(revision[..<slash])
    }

    /// The caption for a page of this rung, or for a page not yet known.
    ///
    /// Every variant keeps "not to scale" in it, because the one claim no rung
    /// can make yet is a real-world size: scale is unknown on every world.
    ///
    /// `rawImagery`: the viewer knows the CURRENT build's imagery is NOT
    /// redacted (the latest served `X-World-Imagery` header, or the page's
    /// latest `research` word:
    /// APPEARANCE §6.6, `raw-local-research/no-redaction`). The appearance
    /// sentence then says so instead of claiming faces were redacted. `nil`:
    /// neither the header nor the page has said, and the sentence says
    /// exactly that -- "redaction unknown" -- rather than claiming a
    /// redaction nobody reported.
    static func caption(for representation: WorldRenderRepresentation?, rawImagery: Bool? = false) -> String {
        switch representation {
        case .appearance where rawImagery == true:
            return "The camera's own images, not redacted, placed on the reconstructed room. Grey haze is where no kept image looked; only cracks a few pixels wide are filled, from the images beside them. Not to scale."
        case .appearance where rawImagery == nil:
            return "The camera's own images, redaction unknown, placed on the reconstructed room. Grey haze is where no kept image looked; only cracks a few pixels wide are filled, from the images beside them. Not to scale."
        case .appearance:
            // Agrees with the page's own caption and WORLD-BUILDER-WORLDS.md §4
            // ("what it draws"), which is where the page's drawing rules live.
            //
            // Why this caption names both the filled cracks and the grey haze
            //
            // It used to end "Dark gaps were not seen or were masked as
            // unreliable; nothing is filled in." Both halves stopped being true
            // in opposite directions:
            //
            // - **Nothing is filled in** is no longer true. A pixel with no
            //   surface whose two sides, within a few device pixels, lie on one
            //   plane is placed on that plane and shaded (`FS_FILL`). The
            //   PIXELS are still real camera pixels -- APPEARANCE §2 claim 1 and
            //   §3 hold -- but the GEOMETRY under them is invented there, and a
            //   wearer who is told "nothing is filled in" would read a closed
            //   crack as measured. It is bounded and the caption says so: only
            //   cracks a few pixels wide, only across one plane.
            // - **Dark gaps** is no longer the shape of the error either. A
            //   void is painted as an unlit grey fog lifted from the mean of
            //   what is drawn around it, not as black. "Grey haze" is what the
            //   wearer actually sees, and (since the fog's mean is
            //   coverage-weighted) it is always darker than the room beside it,
            //   so "haze" and "darker" agree.
            //
            // Short on purpose: this sits under a picture on a phone, and the
            // page's own About carries the long form.
            return "The camera's own images, faces redacted, placed on the reconstructed room. Grey haze is where no kept image looked; only cracks a few pixels wide are filled, from the images beside them. Not to scale."
        case .surface:
            // Agrees with the page's own caption and WORLD-BUILDER-SURFACE.md
            // section 2 claim 2: a gap is NOT proof that nobody looked. On the
            // canonical capture most measured depth with no surface near it
            // was outvoted or lacked a second agreeing view.
            return "Surfaces the Tower reconstructed from the walk, only where the cameras measured them. A gap is not proof that nothing is there. Not to scale."
        case .dense:
            return "Points the Tower measured densely from the walk. Not a surface, and not to scale."
        case .sparse:
            return "Points the Tower measured from the walk, with the camera path through them. Not a surface, and not to scale."
        case nil:
            return "What the Tower reconstructed from the walk. Not to scale."
        }
    }
}

/// Something the page itself did, reported by the web view's delegate.
///
/// A value rather than three closures, so the model has one entry point and a
/// test can drive the whole render lifecycle without a `WKWebView`.
nonisolated enum WorldRenderPageEvent: Equatable {
    /// `didFinish`: the page loaded and its script ran.
    case rendered
    /// `didFail` / `didFailProvisionalNavigation` with an error that is not a
    /// navigation this app deliberately cancelled.
    case failed(String)
    /// The content process died and the page is being put back. The page on
    /// screen is gone until it finishes, so this returns the screen to
    /// `.rendering` and re-arms its bound — without it, a reload that never
    /// completes is the same unobserved black rectangle one layer down.
    case reloadingAfterTermination
    /// The page asked to reload ITSELF (the appearance page's last-ditch
    /// recovery from a WebGL context WebKit never gave back) and the navigation
    /// policy allowed it. Handled exactly like a termination reload: the page
    /// on screen is gone until the new one finishes, so the screen goes back to
    /// a bounded `.rendering` with the "Drawing the world…" overlay over it.
    ///
    /// Without this the screen stayed `.ready` through the one recovery that
    /// exists to avoid an unexplained black rectangle, and `followRevisions`
    /// kept polling into a document that was mid-reload.
    case reloadingItself
    /// WebKit's content process was killed this many times and the web view
    /// has stopped putting the page back. See `Coordinator.reloadBudget`.
    case gaveUpAfterTerminations(Int)
}

/// What one revision poll means for an appearance page whose imagery was
/// withdrawn while it was on screen.
///
/// Pure, because the interesting cases are a relabel that comes back while the
/// Tower is briefly unreachable, and a page that has already fixed itself — and
/// neither is reproducible with a real `WKWebView`.
///
/// ## Two recoveries, and why the page's wins
///
/// When withdrawn imagery is served again there are two mechanisms that can put
/// the picture back, and both used to run:
///
/// - **The page's.** `appearance_viewer.html`'s own follower sees a non-null
///   `appearance.revision` against its nulled `S.revision`, refetches the
///   manifest and redraws **keeping the wearer's camera**. Nothing is
///   re-downloaded that the handler still holds.
/// - **The app's.** `refresh(to:evenIfUnchanged:)` replaces the whole document:
///   the camera resets to the opening pose, the "Drawing the world…" overlay
///   covers the screen, and because the withdrawn poll dropped the memory copy
///   the page fetches every chunk again — about 13 MB over Tailscale.
///
/// Both pollers run at 10 s, so they collided routinely and the slower, more
/// expensive one won. The app now waits one poll: if the page
/// fetched a served manifest through the scheme handler in the meantime
/// (`WorldAssetSchemeHandler.servedAppearanceToPageAt`), the page is alive and
/// has already recovered, and the app does nothing at all. Only a page that did
/// not take the imagery back — its script died, or it never started one — is
/// replaced. Exactly one mechanism acts.
nonisolated enum WorldAppearanceFollow: Equatable, Sendable {
    /// Nothing to do this poll; `withdrawn` is what the flag becomes.
    case keepWatching(withdrawn: Bool)
    /// Served again, and the page has already redrawn itself. Forget the
    /// withdrawal; do not touch the document.
    case leaveItToThePage
    /// Served again, but the page has not fetched a manifest yet. Give it one
    /// poll interval before replacing the document under it.
    case waitOnePollForThePage
    /// Served again and the page did not take it. Replace the document.
    case replaceThePage
}

nonisolated extension WorldRenderFetchError {
    /// What the sheet says. Each case names the thing that is missing or
    /// broken, and never blames the world for a link that dropped.
    var message: String {
        switch self {
        case .absent(let detail?) where detail == Self.unmatchedRouteDetail:
            // FastAPI's default for a path it does not serve, which is never
            // one of the contract's five sentences. That is a Tower older
            // than the route, not a world with nothing built.
            return "This Tower is too old to serve a picture of a saved world. Its worlds are still listed and their diagnostics still open."
        case .absent(let detail?) where WorldAreaRouteAnswer.isAreaAnswer(detail):
            // One of the area routes' four stable sentences (§5.1).
            return "The Tower has no picture of this area: \(detail)."
        case .absent(let detail?):
            // No "yet": the detail says which — a world still being built
            // answers "no geometry yet", and a world that does not exist
            // answers "no world", and only the Tower knows which. The route's
            // `detail` is Tower prose, and only shown.
            return "The Tower has no picture for this world: \(WorldTowerText.clause(detail))."
        case .absent(nil):
            return "The Tower has no picture for this world — no geometry has been built for it, or the world is gone."
        case .badAddress:
            return "This world's identifier could not be made into a Tower address."
        case .towerError(let status):
            return "The Tower could not produce the picture (HTTP \(status))."
        case .undecodable:
            return "The Tower's page could not be read as text."
        case .transport(let detail):
            return "The picture could not be fetched: \(detail)"
        }
    }

    /// Whether trying again could change the answer. A world still being
    /// built answers 404 until its first solve lands, so `absent` is
    /// retryable; a bad id is not, and neither is a Tower without the route.
    var isRetryable: Bool {
        switch self {
        case .badAddress, .undecodable: return false
        case .absent(let detail?) where detail == Self.unmatchedRouteDetail: return false
        // An area that is gone, could not be built, or never existed will not
        // appear on a retry; one not built YET might (§5.1).
        case .absent(let detail?) where WorldAreaRouteAnswer.isTerminal(detail): return false
        case .absent, .towerError, .transport: return true
        }
    }

    /// What FastAPI answers for a path no route serves.
    static let unmatchedRouteDetail = "Not Found"
}

/// Owns one fetch of one page. Created by the sheet and destroyed with it;
/// it holds a string, not a connection, so losing it loses nothing.
