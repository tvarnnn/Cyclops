import Combine
import Foundation

// MARK: - What is being looked at

/// Which of the Tower's two renderings of the same world is wanted.
///
/// The Tower's render page is the **product** view of a world: the largest
/// connected component composed as one world and coloured by real photometric
/// RGB. A diagnostics variant of the same page — segment-index colouring, the
/// solver's own view — is requested with `?view=diagnostics`.
///
/// **Two ways this can silently do nothing, both known and both accepted.**
///
/// 1. This app serves the fetched page through its private `glasses-world:`
///    scheme and refuses other navigation. The page URL carries no `view`
///    query, so `location.search` cannot select diagnostics. This isolation
///    preserves the reason `WorldRenderClient` fetches the page first.
/// 2. So the Tower has to honour `view` **server-side**, when it composes
///    the page. It does: `tower/routes/geometry.py::world_render` declares
///    `view`, and `render.py` substitutes the opening mode into the script
///    rather than reading it from a URL the phone cannot supply.
///
/// A Tower too old to declare `view` answers with the ordinary world page
/// and no error, because FastAPI ignores query parameters a route does not
/// declare. That is the right failure: the wearer asked to see something and
/// sees their world.
///
/// The sparse-geometry diagnostics that *this* app owns — the fragment gallery,
/// the placement counts, the refusal prose, the segment origins — do not depend
/// on any of that. They are behind the world screen's own Diagnostics
/// disclosure and are reachable whatever the Tower does here.
nonisolated enum WorldRenderView: String, Equatable, Hashable, Sendable {
    /// The default. Sends no `view` parameter at all, so the URL is byte-for-byte
    /// the one this app has always asked for.
    case product
    case diagnostics
}

/// A world, and optionally one of its sessions, whose viewer page can be
/// asked for: `GET /worlds/{world_id}/render`, `WORLD-BUILDER-WORLDS.md` §4.
///
/// `sessionID` is `nil` when the Tower is to choose — the newest session with
/// geometry — which is what a picker tap on the world's own row means. It is
/// never the empty string.
nonisolated struct WorldRenderTarget: Equatable, Hashable, Sendable, Identifiable {
    let worldID: String
    let sessionID: String?
    /// A `var` with a default rather than a `let` with one: a `let` with an
    /// initial value is left out of the memberwise initialiser entirely, which
    /// would make `WorldRenderTarget(worldID:sessionID:view:)` unavailable to
    /// the one call site that needs it.
    var view: WorldRenderView = .product
    /// One of the session's areas (`WORLD-BUILDER-COMPONENTS.md` §5), or `nil`
    /// for the room. An area target always names its session: the area routes
    /// carry it in the path.
    var areaID: String? = nil

    /// Includes the view and the area, so a navigation destination keyed on
    /// the target treats the room and each area as different screens -- which
    /// is what makes opening an area REPLACE the room viewer rather than stack
    /// on it.
    var id: String {
        "\(worldID)/\(sessionID ?? "")/\(view.rawValue)" + (areaID.map { "/area:\($0)" } ?? "")
    }

    var isArea: Bool { areaID != nil }

    /// The same world and session, in the other rendering.
    func showing(_ view: WorldRenderView) -> WorldRenderTarget {
        WorldRenderTarget(worldID: worldID, sessionID: sessionID, view: view, areaID: areaID)
    }

    /// One of this session's areas, by the id the listing or revision gave.
    func area(_ areaID: String) -> WorldRenderTarget {
        WorldRenderTarget(worldID: worldID, sessionID: sessionID, areaID: areaID)
    }

    /// The room of the same world and session: *Back to the room*.
    var room: WorldRenderTarget {
        WorldRenderTarget(worldID: worldID, sessionID: sessionID)
    }
}

/// What an area viewer needs to say about the area it shows, taken from the
/// list it was opened from and kept for the life of the screen: its
/// number, how many areas the walk has, and when it was captured.
nonisolated struct WorldAreaOpening: Equatable, Hashable, Sendable {
    let number: Int
    let total: Int
    let spans: [WorldCaptureSpan]

    /// The area viewer's header (§5.4), whole: the navigation title, which is
    /// also what accessibility reads.
    var header: String { WorldComponentsPresentation.areaHeader(number: number, of: total) }
    /// The header's two halves, drawn on two lines: on a phone the one-line
    /// title truncated to "Area 1 of 1 — not placed i…", losing the half that
    /// matters.
    var numberLine: String { "Area \(number) of \(total)" }
    static let notPlacedLine = "not placed in the room"
}

// MARK: - The fetch

nonisolated enum WorldRenderFetchError: Error, Equatable {
    /// The Tower answered 404: no root, no such world or session, or no
    /// geometry built yet. Carries the Tower's own `detail`, which names
    /// which, when it could be read.
    case absent(detail: String?)
    /// A `world_id` or `session_id` the URL machinery refused. Both are
    /// wire-supplied, so this is a claim about the Tower's output, not this
    /// code, and is reported rather than force-unwrapped.
    case badAddress
    /// Any other non-2xx.
    case towerError(status: Int)
    /// A body that is not text.
    case undecodable
    case transport(String)
}

/// What `GET /worlds/{id}/render/revision` said, `WORLD-BUILDER-WORLDS.md` §4a.
nonisolated struct WorldRenderRevision: Equatable, Sendable {
    /// Opaque. Compared for equality only (§4a rule 1).
    let revision: String
    /// The rung that revision would serve, or `nil` for a value this app does
    /// not know. Read from the payload rather than from the revision, which
    /// is opaque: this is what decides whether a new picture replaces the one
    /// on screen by itself or is only offered.
    let representation: WorldRenderRepresentation?
    /// Whether the Tower is building this session right now. `nil` when the
    /// Tower did not say. `false` means "slow down", not "stop" (§4a rule 6).
    let live: Bool?
    /// The session's appearance build (`appearance.revision`), or `nil` when
    /// none is served. Opaque, and deliberately NOT part of `revision`: the
    /// appearance page follows these builds itself, overwriting its texture
    /// layers in place, so the follower never reloads the page for one -- that
    /// would reset the wearer's camera on every solve.
    var appearance: String? = nil
    /// `appearance.state` (`WORLD-BUILDER-APPEARANCE.md` §9): `served`,
    /// `rebuilding` (the label changed at Stop and the final build is coming;
    /// the page keeps what it drew), `withdrawn` or `absent`. `nil` from a
    /// Tower that does not say.
    var appearanceState: String? = nil
    /// The resolved session's `components` (`WORLD-BUILDER-COMPONENTS.md`
    /// §3.2), or `nil` when not computed. Deliberately NOT part of
    /// `revision`: an area finishing never swaps the room page. The room
    /// screen uses it to keep its areas row current.
    var components: WorldComponents? = nil
    /// What the picture §4 would serve now was built from (WORLDS §4a
    /// `basis`; absent with `TOWER_WORLD_PICTURE_BASIS` off). Decides
    /// only the label, never a swap.
    var basis: WorldPictureBasis? = nil
}

/// Fetches the viewer page over HTTP, as a string.
///
/// ## Why `URLSession` fetches the page and the web view only renders it
///
/// A `WKWebView` pointed straight at the route would work until the day it
/// did not, and on that day it would say nothing useful: its navigation
/// delegate sees a 404's status but never its body, so the Tower's careful
/// "no geometry yet" versus "no such world" would collapse into a blank
/// page. Fetching here means the failure arrives as a typed error with the
/// Tower's own words in it, the request carries the app's timeout and cache
/// policy like every other HTTP client in this app, and the web view is handed
/// that string through the app's own `glasses-world:` scheme
/// (`WorldAssetSchemeHandler`), which serves it and a short whitelist of this
/// world's routes and nothing else — so refusing every other navigation is
/// still a one-line policy rather than an allow-list.
///
/// Modelled on `WorldListClient`: a struct holding a `URL` and a session,
/// defaulted so a test can substitute a stubbed one.
nonisolated struct WorldRenderClient {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    /// Ephemeral and cache-less (`WorldAssetClient.sharedUncachedSession`), not
    /// `.shared`: the appearance page's configuration names the imagery it
    /// fetches, and `URLSession.shared` writes responses to a disk cache unless
    /// told otherwise. For privacy, first-person room imagery stays in memory,
    /// not a disk cache (`ios/Glasses/Project_Overview_Steps/docs/06-PRIVACY-DATA.md`,
    /// "Raw Frame/Audio Retention"; `WORLD-BUILDER-IOS.md` §10).
    var session: URLSession = WorldAssetClient.sharedUncachedSession
    /// Longer than the list's ten seconds: a page for a full walk is a few
    /// megabytes of point coordinates, and a slow link is not a stalled one
    /// at that size. Still bounded (Rule 15).
    var timeout: TimeInterval = 30

    func page(for target: WorldRenderTarget) async throws -> String {
        guard let url = Self.url(for: target, baseURL: baseURL) else {
            throw WorldRenderFetchError.badAddress
        }
        // Local AND remote caches ignored, for the reason `WorldGeometryClient`
        // gives and for privacy: first-person room imagery stays out of disk
        // caches (`ios/Glasses/Project_Overview_Steps/docs/06-PRIVACY-DATA.md`,
        // "Raw Frame/Audio Retention"; `WORLD-BUILDER-IOS.md` §10);
        // the Tower also answers `Cache-Control:
        // no-store`, and a world under construction changes with every build.
        let request = Self.request(url, timeout: timeout)
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw WorldRenderFetchError.transport(error.localizedDescription)
        }
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            if http.statusCode == 404 {
                throw WorldRenderFetchError.absent(detail: Self.detail(in: data))
            }
            throw WorldRenderFetchError.towerError(status: http.statusCode)
        }
        guard let html = String(data: data, encoding: .utf8) else {
            throw WorldRenderFetchError.undecodable
        }
        return html
    }

    /// Which picture the Tower would serve now, or `nil` when there is no
    /// revision to follow.
    ///
    /// A few hundred bytes, so an open picture can learn that a better one was
    /// built -- during a walk the surface is rebuilt each time a global solve
    /// lands -- without re-downloading megabytes to find out.
    ///
    /// `nil` means ONLY a Tower without the route: a 404 whose detail is
    /// FastAPI's own `Not Found`. Every other 404 is one of §4's sentences --
    /// "no geometry yet" among them, which a world mid-build can answer while
    /// its derived tree is replaced -- and is thrown, so the follower asks
    /// again rather than giving up for the life of the screen
    /// (`WORLD-BUILDER-WORLDS.md` §4a rule 5).
    func revision(for target: WorldRenderTarget) async throws -> WorldRenderRevision? {
        guard let url = Self.revisionURL(for: target, baseURL: baseURL) else {
            throw WorldRenderFetchError.badAddress
        }
        let request = Self.request(url, timeout: 10)
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw WorldRenderFetchError.transport(error.localizedDescription)
        }
        if let http = response as? HTTPURLResponse {
            if http.statusCode == 404 {
                let detail = Self.detail(in: data)
                if detail == WorldRenderFetchError.unmatchedRouteDetail { return nil }
                throw WorldRenderFetchError.absent(detail: detail)
            }
            if !(200..<300).contains(http.statusCode) {
                throw WorldRenderFetchError.towerError(status: http.statusCode)
            }
        }
        return try Self.decodeRevision(data)
    }

    /// Every request this client makes. Split out so the cache policy is
    /// tested without a stubbed session.
    static func request(_ url: URL, timeout: TimeInterval) -> URLRequest {
        URLRequest(url: url, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData,
                   timeoutInterval: timeout)
    }

    /// The revision route's body. Split out so the decoding is tested without
    /// a stubbed session.
    static func decodeRevision(_ data: Data) throws -> WorldRenderRevision {
        guard
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let revision = json["revision"] as? String, !revision.isEmpty
        else { throw WorldRenderFetchError.undecodable }
        let representation = (json["representation"] as? String)
            .flatMap(WorldRenderRepresentation.init(rawValue:))
        let appearanceObject = json["appearance"] as? [String: Any]
        let appearance = appearanceObject?["revision"] as? String
        let appearanceState = appearanceObject?["state"] as? String
        return WorldRenderRevision(
            revision: revision,
            representation: representation,
            live: json["live"] as? Bool,
            appearance: (appearance?.isEmpty ?? true) ? nil : appearance,
            appearanceState: (appearanceState?.isEmpty ?? true) ? nil : appearanceState,
            components: WorldComponents(json: json["components"]),
            basis: (json["basis"] as? String).flatMap { $0.isEmpty ? nil : WorldPictureBasis(rawValue: $0) }
        )
    }

    /// The revision route's address: the page's own path plus `/revision`,
    /// with the session and the `viewer` declaration, and without `view`,
    /// which does not change what is built. The declaration must match the
    /// page's: a revision answered without it names the surface rung while an
    /// appearance page is on screen.
    static func revisionURL(for target: WorldRenderTarget, baseURL: URL) -> URL? {
        if target.isArea {
            // `/worlds/<w>/areas/<s>/<a>/render/revision`, with no query at all
            // (§5.2): the same address the page's own poll is proxied to, so
            // the two cannot disagree.
            guard let page = url(for: target, baseURL: baseURL),
                  var components = URLComponents(url: page, resolvingAgainstBaseURL: false)
            else { return nil }
            components.percentEncodedPath += "/revision"
            components.queryItems = nil
            return components.url
        }
        guard
            let page = url(for: target, baseURL: baseURL),
            var components = URLComponents(url: page, resolvingAgainstBaseURL: false)
        else { return nil }
        components.percentEncodedPath += "/revision"
        let kept = (components.queryItems ?? []).filter {
            $0.name == "session_id" || $0.name == WorldAssetScheme.viewerQueryName
        }
        components.queryItems = kept.isEmpty ? nil : kept
        return components.url
    }

    /// The route's address, or `nil` when an id will not go into a URL.
    ///
    /// The id is percent-encoded as **one** path component by hand.
    /// `appendingPathComponent` does not encode `/` — a test showed
    /// `"a/b"` becoming two segments — so a wire-supplied id could otherwise
    /// address a route this client never meant to. Everything outside the
    /// unreserved set is encoded, `/` included.
    static func url(for target: WorldRenderTarget, baseURL: URL) -> URL? {
        guard !target.worldID.isEmpty, target.sessionID != "" else { return nil }
        if let areaID = target.areaID {
            // `/worlds/<w>/areas/<s>/<a>/render` (`WORLD-BUILDER-COMPONENTS.md`
            // §5.1). The session is in the path, never a query. No `viewer`:
            // the area route ignores it and offers the appearance rung to
            // every client that can name it. No `view`: an area has
            // no diagnostics rendering.
            guard
                let sessionID = target.sessionID,
                let session = sessionID.addingPercentEncoding(withAllowedCharacters: Self.unreserved),
                WorldComponent.isAreaID(areaID),
                var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false),
                let world = target.worldID.addingPercentEncoding(withAllowedCharacters: Self.unreserved)
            else { return nil }
            let base = components.percentEncodedPath.hasSuffix("/")
                ? String(components.percentEncodedPath.dropLast())
                : components.percentEncodedPath
            components.percentEncodedPath = "\(base)/worlds/\(world)/areas/\(session)/\(areaID)/render"
            // (COMPONENTS §5.1, WORLDS §4c): ask the appearance page to
            // hand its chrome to the phone. An older Tower ignores it, and
            // the revision poll drops every query (`revisionURL`).
            components.queryItems = [URLQueryItem(name: WorldChromeEcho.queryName,
                                                  value: WorldChromeEcho.queryValue)]
            return components.url
        }
        guard
            var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false),
            let world = target.worldID.addingPercentEncoding(withAllowedCharacters: Self.unreserved)
        else { return nil }
        let base = components.percentEncodedPath.hasSuffix("/")
            ? String(components.percentEncodedPath.dropLast())
            : components.percentEncodedPath
        components.percentEncodedPath = "\(base)/worlds/\(world)/render"
        var query: [URLQueryItem] = []
        if let sessionID = target.sessionID {
            query.append(URLQueryItem(name: "session_id", value: sessionID))
        }
        // Only for the diagnostics rendering. The product view sends no `view`
        // parameter, so a Tower that has never heard of the parameter is never
        // sent it.
        if target.view != .product {
            query.append(URLQueryItem(name: "view", value: target.view.rawValue))
        }
        // What this app can draw (`WORLD-BUILDER-WORLDS.md` §4): the Tower's
        // `auto` offers the appearance page only to a client that says so,
        // because an older build cannot fetch its imagery. A Tower older than
        // the parameter ignores it (FastAPI drops unknown query parameters).
        query.append(URLQueryItem(name: WorldAssetScheme.viewerQueryName,
                                  value: WorldAssetScheme.viewerCapability))
        // (WORLDS §4c): on the page request only, and only for the
        // product view -- the diagnostics page is the sparse one, which keeps
        // its own chrome. Not a capability declaration and decides no rung;
        // `revisionURL` keeps only `session_id` and `viewer`, so the polls
        // never carry it (IOS §10).
        if target.view == .product {
            query.append(URLQueryItem(name: WorldChromeEcho.queryName, value: WorldChromeEcho.queryValue))
        }
        components.queryItems = query.isEmpty ? nil : query
        return components.url
    }

    /// RFC 3986 unreserved characters: the only ones an id may keep.
    private static let unreserved = CharacterSet(
        charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    )

    /// FastAPI's `{"detail": "..."}`, when that is what came back.
    static func detail(in data: Data) -> String? {
        guard
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let detail = json["detail"] as? String, !detail.isEmpty
        else { return nil }
        return detail
    }
}

/// Owns one fetch of one page. Its owner creates it and it goes with that
/// owner: the viewer sheet's scene, or `WorldInlineHost`, which keeps one model
/// across its inline and expanded views. It holds a string, not a connection,
/// so losing it loses nothing.
@MainActor
final class WorldRenderViewerModel: ObservableObject {
    @Published private(set) var state: WorldRenderViewerState = .fetching {
        didSet {
            // A failure view never keeps a dead web view alive.
            if case .failed = state { web.tearDown() }
        }
    }

    /// What the picture on screen is known to be built from (IOS §3c):
    /// the label never runs ahead of the picture.
    @Published private(set) var basisOnScreen: WorldPictureBasisOnScreen = .unknown
    /// When the revision route first said `basis: "final"`, until it stops.
    private var finalBasisSince: Date?
    /// The last `basis` the revision route said.
    private var polledBasis: WorldPictureBasis?
    /// When the page on screen was FETCHED (not re-rendered).
    private var pageFetchedAt: Date?

    /// The world, the session and which rendering was asked for. A `let`: this
    /// screen shows one page.
    ///
    /// It was briefly a `@Published var` with a `show(_:)` beside it, so a
    /// segmented control in Details could switch renderings. That control is
    /// gone — see `details` for why — and with it the only reason this could
    /// change. A reader who wants the other rendering either uses the page's
    /// own button, which costs nothing, or opens it from the world screen's
    /// Diagnostics section, which is a new screen with its own model.
    let target: WorldRenderTarget
    private let client: WorldRenderClient

    /// How long the page gets to draw itself after the fetch succeeds.
    ///
    /// Generous: the Tower decimates to a mobile budget, but the script still
    /// builds typed arrays over tens of thousands of points on a phone. Bounded
    /// all the same (Rule 15) — an unbounded wait here is exactly the black
    /// rectangle this state was added to end. Injectable so a test can prove
    /// the bound exists in milliseconds rather than spending twenty seconds.
    var renderTimeout: Duration = .seconds(20)

    /// Bumped on every `load()`.
    ///
    /// Handed to `WorldRenderWebView` so a **retry of the same page** actually
    /// reloads. The coordinator skips a page load when the string has
    /// not changed — which is right for the re-renders SwiftUI does constantly,
    /// and wrong for a deliberate "Try again" after a render failure, where the
    /// string is identical and reloading it is the entire point.
    @Published private(set) var renderAttempt = 0

    /// Bumped by each `didFinish` that leaves the page `.ready`. The view
    /// answers it with `renderingPanelGone()` once the commit that took the
    /// rendering panel away is on screen (`WorldRenderCover`).
    @Published private(set) var pageFinishedToken = 0

    /// Which content-process-kill budget the page on screen is under.
    ///
    /// Separate from `renderAttempt` because the two questions came apart.
    /// "Load this string again even though it is identical" is
    /// `renderAttempt`; "and give it a fresh kill budget" is this. They agreed
    /// while the only forced reloads were the reader's own "Try again" and a
    /// revert of a page that had already drawn — both of which do deserve a
    /// fresh budget. The withdrawal-recovery reload does not: it fires once per
    /// relabel cycle, and handing the page a new budget each time made
    /// `Coordinator.reloadBudget` unbounded for exactly the page most likely to
    /// need it.
    @Published private(set) var renderBudgetToken = 0

    /// Bounds `.rendering`. Cancelled when the page reports either way.
    ///
    /// Unstructured (`Task { }`), so it does not die with the SwiftUI `.task`
    /// that started the fetch; `[weak self]` so a dismissed sheet is not kept
    /// alive for the length of the timeout.
    private var renderWatchdog: Task<Void, Never>?

    /// How often an open picture asks whether a better one has been built,
    /// while the Tower says it is building.
    ///
    /// Ten seconds against a live surface that is rebuilt when a global solve
    /// lands -- tens of seconds apart on a real walk -- so a new picture is on
    /// screen within one interval of existing. Injectable so a test proves the
    /// behaviour in milliseconds.
    var revisionPollInterval: Duration = .seconds(10)

    /// The longest the follower waits between asks when nothing is building.
    ///
    /// A saved, finished world never changes, and asking every ten seconds for
    /// as long as it is open keeps the radio awake for nothing. So while the
    /// Tower answers `live: false` and the revision stays put, the interval
    /// doubles up to this. It is a ceiling and not a stop, because `false` is
    /// not a promise: the Tower starts the final surface a few seconds AFTER it
    /// releases the world lock (`WORLD-BUILDER-WORLDS.md` §4a rule 6), and a
    /// follower that stopped there would miss the picture the walk was for.
    var revisionPollCeiling: Duration = .seconds(120)

    /// The revision stamped into the page on screen (or, for a page that
    /// carries none, the one that led to it). Starts as the page's own stamp, so
    /// a build that lands between fetching the page and the first poll is still
    /// noticed.
    private var shownRevision: String?

    /// The last revision the follower ACTED on: fetched, offered, or found to
    /// change nothing. Separate from `shownRevision` because the two can differ
    /// for good -- the Tower's revision route can say "surface" about a page it
    /// then serves as a lower rung (§4a rule 4) -- and comparing only with the
    /// page's stamp would download that page again on every poll.
    private var handledRevision: String?

    /// Revisions whose page could not be drawn on this phone. Never offered or
    /// swapped in again by the follower -- except once, for a revision refused
    /// while its build was live that the Tower then reports finished
    /// (`refusedWhileLive`) -- and forgotten by the reader's own "Try again"
    /// (`newerPictureRefused`, then `load()`).
    private var refusedRevisions: Set<String> = []

    /// The subset of `refusedRevisions` refused while the Tower said the build
    /// was live, each owed one try when the same revision is reported finished.
    /// See `followRevisions`.
    private var refusedWhileLive: Set<String> = []

    /// How many refreshes to each rung failed to draw, and the rungs that have
    /// failed twice. A rung too large for this phone used to be refused per
    /// REVISION, so every live build of it was downloaded, killed and reverted
    /// again. Twice, not once: one failure can be memory
    /// another app was holding.
    ///
    /// Only a failure to go UP a rung counts -- a page of the rung already on
    /// screen drew on this phone, so its rebuild failing is pressure, not size
    /// -- and a refresh of the rung that does draw forgets its failures. Both
    /// were missing, so two transient failures in a walk refused the rung for
    /// the life of the screen. Cleared by `load()`.
    private var rungDrawFailures: [WorldRenderRepresentation: Int] = [:]
    private var refusedRungs: Set<WorldRenderRepresentation> = []

    /// Refused rungs a FINISHED build (`live == false`) has already been let
    /// through once. The finished world after Stop is the picture the walk was
    /// for, and the capture that was competing for memory has stopped; it gets
    /// one more try per rung, and only one, so a rung that really is too large
    /// is not downloaded on every poll of a saved world.
    private var finishedBuildRetried: Set<WorldRenderRepresentation> = []

    /// A refresh could not be drawn and the previous page was put back. Published
    /// so the scene can offer "Try again" (`load()`), which forgets every
    /// refusal. A reverted screen is `.ready`, where the failure view and its
    /// button are not shown, so without this the refusals could only be lifted
    /// by closing the screen.
    @Published private(set) var newerPictureRefused = false

    /// The page that was on screen when an automatic refresh replaced it, and
    /// the revision it carried, kept until the replacement reports drawn. A
    /// refresh that cannot be drawn -- watchdog, `didFail`, or the content
    /// process running out of memory, which is exactly what a larger rung does
    /// -- returns to it instead of to a failure. `wasLive` is whether the
    /// Tower said the build being swapped IN was live, so a refusal can be
    /// told apart from one of a finished build (`refusedWhileLive`).
    private var fallback: (html: String, revision: String?, wasLive: Bool, fetchedAt: Date?)?

    /// A newer build of the SAME rung, waiting for the reader to ask for it.
    ///
    /// Published so the scene can show a button. A same-rung rebuild is not
    /// swapped in by itself: a swap reloads the page, which covers it with the
    /// "Drawing" overlay and resets the camera and the walk/orbit mode, and a
    /// surface is rebuilt on every global solve during a walk.
    @Published private(set) var newerPictureAvailable = false
    private var pendingRevision: String?

    /// The appearance page on screen was told its imagery is withdrawn (a
    /// relabel, a purge: `appearance.state` neither `served` nor `rebuilding`)
    /// and dropped its textures. When a served appearance comes back the page is
    /// replaced by itself, whatever its page revision says: the
    /// page recovers on its own, but one from a Tower that predates epochs (or
    /// one whose script died) would otherwise show "no longer served" for good,
    /// because an unchanged page revision is never fetched again. Not set by
    /// `rebuilding`, the ordinary Stop, where the page keeps its textures and
    /// picks the final build up in place without a reload.
    private var appearanceWithdrawnWhileShown = false
    /// When that flag was last set, so a manifest the PAGE fetched afterwards
    /// can be told from one it fetched before (`WorldAppearanceFollow`).
    private var appearanceWithdrawnAt: Date?
    /// Whether the page has already been given its one poll's grace to recover
    /// on its own since the appearance came back.
    private var appearanceWaitedForPage = false

    /// Serves the page and proxies this world's whitelisted routes.
    ///
    /// **The viewer owns it, not the web view.** The in-memory copy of the
    /// imagery is the VIEWER's (`WORLD-BUILDER-IOS.md` §10): in-memory reuse
    /// ends when the viewer closes. A "Try again"
    /// destroys the `WKWebView` and builds another one. Owning it here also
    /// gives the follower the one fact only the handler knows: whether the page
    /// itself has fetched a served manifest (`WorldAppearanceFollow`).
    let assets: WorldAssetSchemeHandler

    /// The room's `components` (`WORLD-BUILDER-COMPONENTS.md` §3.2): seeded
    /// from the listing row the viewer was opened from, then kept current from
    /// every revision the follower reads. `nil` is not computed -- every older
    /// world -- and draws exactly today's screen. Never set on an area viewer.
    @Published private(set) var components: WorldComponents?

    /// The Tower said this area will not be served (§5.2): kept picture,
    /// follower stopped, and the scene offers the way back to the room.
    @Published private(set) var areaNoLongerServed = false

    /// Native chrome (WORLDS §4c, IOS §10): what the phone draws in place
    /// of the appearance page's own chrome, and the bridge that feeds it.
    /// Owned here, like `assets`, so a "Try again" web view re-attaches the
    /// same bridge (whose `pageWillLoad` resets it) and the research marker
    /// lasts for the life of the viewer.
    let chrome: WorldChromeModel
    let bridge: WorldChromeBridge
    private var chromeChanges: AnyCancellable?

    /// The one web view, which can move between the panel and the cover
    /// without a reload.
    let web: WorldWebViewOwner

    /// `assets` is injectable for the same reason `client` is: the follower's
    /// one piece of evidence about the page's own recovery comes through the
    /// handler, and a test needs to be able to put it there from a stubbed
    /// Tower rather than a real one.
    init(
        target: WorldRenderTarget,
        client: WorldRenderClient = WorldRenderClient(),
        assets: WorldAssetSchemeHandler? = nil,
        components: WorldComponents? = nil
    ) {
        self.target = target
        self.components = target.isArea ? nil : components
        self.client = client
        let assets = assets
            ?? WorldAssetSchemeHandler(worldID: target.worldID, scope: WorldAssetScope.of(target))
        self.assets = assets
        let chrome = WorldChromeModel()
        self.chrome = chrome
        self.bridge = WorldChromeBridge(
            model: chrome, kind: target.isArea ? .area : .room,
            pageURL: WorldAssetScheme.pageURL(worldID: target.worldID, scope: assets.scope))
        self.web = WorldWebViewOwner(target: target, assets: assets, bridge: bridge)
        // The second source of the research marker: the served `X-World-Imagery`
        // headers (docs/contracts/WORLD-BUILDER-APPEARANCE.md §9; WORLD-BUILDER-IOS §10).
        assets.onImagery = { [weak chrome] imagery, warning in chrome?.noteImagery(imagery, warning: warning) }
        // The scene lays itself out by the chrome's mode, so it re-renders
        // when the chrome changes. The per-frame heading is a separate
        // object and does not come through here.
        chromeChanges = chrome.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        // The label follows the manifest the page was last served.
        assets.onAppearanceManifest = { [weak self] in self?.recomputeBasis() }
    }

    /// The viewer closed. Drops the in-memory imagery and its authorisation,
    /// which `WORLD-BUILDER-IOS.md` §10 promises and which
    /// nothing did: `dropCache()` had no caller at all, and up to 64 MB of
    /// first-person room imagery was left to ARC releasing an object graph that
    /// `WKWebView` teardown is exactly the place to linger in.
    func viewerClosed() {
        web.tearDown()
        assets.tearDown()
        chrome.viewerClosed()
    }

    /// What the picture on screen is known to be built from (IOS §3c,
    /// WORLDS §4a rule 8). The appearance page loads the final build IN PLACE,
    /// so its label follows the manifest the handler last served it; any
    /// other page is final only when it was fetched after the route first
    /// said `basis: "final"`. Pure; table-tested.
    nonisolated static func basisOnScreen(polled: WorldPictureBasis?, shownRung: WorldRenderRepresentation?,
                                          servedAppearance: WorldPictureBasis?, finalSince: Date?,
                                          pageFetchedAt: Date?) -> WorldPictureBasisOnScreen {
        if shownRung == .appearance {
            switch servedAppearance {
            case .final?: return .final
            case .walk?: return .walk
            default: break
            }
            return polled == .final ? .finalArriving : .unknown
        }
        switch polled {
        case .walk?: return .walk
        case .final?:
            if let since = finalSince, let at = pageFetchedAt, at > since { return .final }
            return .finalArriving
        default: return .unknown
        }
    }

    /// Recompute the label from what is known now. Cheap; called on every
    /// poll, every served manifest and every fetched page.
    func recomputeBasis() {
        let next = Self.basisOnScreen(polled: polledBasis, shownRung: state.representation,
                                      servedAppearance: assets.servedAppearanceBasis,
                                      finalSince: finalBasisSince, pageFetchedAt: pageFetchedAt)
        if next != basisOnScreen { basisOnScreen = next }
    }

    /// The route said `basis`; `at` is when.
    func noteBasis(_ basis: WorldPictureBasis?, at now: Date = Date()) {
        polledBasis = basis
        finalBasisSince = basis == .final ? (finalBasisSince ?? now) : nil
        recomputeBasis()
    }

    /// A fetched page became the page on screen.
    private func pageFetched(at now: Date = Date()) {
        pageFetchedAt = now
        recomputeBasis()
    }

    /// Keep the areas row current from a revision the Tower sent (§3.2),
    /// republishing only when something a reader can see moved.
    ///
    /// A `nil` from the Tower replaces the list the viewer had: `null` is the
    /// Tower's current answer ("not computed"), and a re-solve that dropped
    /// its components must not keep offering areas whose routes now 404.
    func adoptComponents(_ latest: WorldComponents?) {
        guard !target.isArea else { return }
        if latest?.displayKey != components?.displayKey {
            components = latest
        }
    }

    /// One revision read, for the areas row alone, as the screen opens: the
    /// follower's first poll is an interval away, and a room opened from the
    /// live screen has no listing row to seed it. Errors are ignored -- the
    /// follower asks again -- and the page is never swapped from here.
    func refreshComponents() async {
        guard !target.isArea, target.view == .product else { return }
        guard let latest = try? await client.revision(for: target) else { return }
        guard !Task.isCancelled else { return }
        adoptComponents(latest.components)
    }

    /// Keep the picture current for as long as the screen is open. It polls
    /// revisions while ready, then swaps a better rung or offers a same-rung
    /// rebuild; the page follows appearance textures itself to keep the camera.
    ///
    /// Runs in the screen's `.task`, so it ends when the screen goes. Asks only
    /// while a page is actually on screen (`.ready`): a page still drawing, or a
    /// failure the reader is reading, is left alone. Returns for good when the
    /// Tower has no revision route, and never starts for the diagnostics
    /// rendering: that page is always the sparse one, and §4a would compare it
    /// with the product ladder's revision and refetch it on every rebuild.
    ///
    /// A better RUNG replaces the page by itself; a rebuild of the same rung
    /// sets `newerPictureAvailable` and waits for `showNewerPicture()`.
    ///
    /// **An appearance build is never acted on here.** It arrives as
    /// `WorldRenderRevision.appearance`, which is deliberately not part of
    /// `revision`: the appearance page polls the same route through the scheme
    /// handler and overwrites its texture layers in place, keeping the camera.
    /// Only the PAGE revision -- a rung change, a session change, a page
    /// program change -- reloads, exactly as before.
    func followRevisions() async {
        guard target.view == .product else { return }
        var interval = revisionPollInterval
        while !Task.isCancelled {
            try? await Task.sleep(for: interval)
            guard !Task.isCancelled else { return }
            guard case .ready = state else { continue }
            if shownRevision == nil { shownRevision = state.revision }
            let latest: WorldRenderRevision?
            do {
                latest = try await client.revision(for: target)
            } catch {
                // An area the Tower says will not be served -- re-finished
                // away, failed, or never computed (§5.2) -- is terminal
                // for this id: keep the picture, stop following, and say so,
                // so the reader can go back to the room, whose areas row is
                // current. Everything else is transient.
                if target.isArea, case .absent(let detail) = error as? WorldRenderFetchError,
                   WorldAreaRouteAnswer.isTerminal(detail) {
                    areaNoLongerServed = true
                    return
                }
                // A dropped request, or one of §4's own 404 sentences, is not a
                // reason to change anything on screen; the next interval asks
                // again.
                continue
            }
            guard !Task.isCancelled else { return }
            guard let latest else { return }
            adoptComponents(latest.components)
            noteBasis(latest.basis)
            // A revision refused while its build was LIVE gets one more try
            // when the Tower reports that same revision FINISHED -- the
            // per-revision form of `finishedBuildRetried`, and for the same
            // reason: the finished world is the picture the walk was for, and
            // the capture that was competing for memory has stopped.
            //
            // It matters for the appearance, whose page revision
            // (`<session>/appearance:1@<epoch>`) survives the ordinary Stop.
            // Without it, one walk-time appearance page that could not be drawn
            // refused the FINISHED world's page too: the Stop gap's surface
            // then drew, took the "Try again" button away with it, and the
            // photographic world that followed under the same revision was
            // never tried, never offered and never mentioned -- the bare
            // surface stayed on screen over it.
            //
            // A revision refused when it was already finished is not retried,
            // and this one retry is spent once taken, so a page too large for
            // this phone is still fetched a bounded number of times.
            if latest.live == false, refusedWhileLive.remove(latest.revision) != nil {
                refusedRevisions.remove(latest.revision)
                if handledRevision == latest.revision { handledRevision = nil }
            }
            if state.representation == .appearance {
                let pageRecovered = appearanceWithdrawnAt.map { withdrawnAt in
                    (assets.servedAppearanceToPageAt ?? .distantPast) > withdrawnAt
                } ?? false
                switch Self.appearanceFollow(
                    withdrawn: appearanceWithdrawnWhileShown,
                    waitedForThePage: appearanceWaitedForPage,
                    pageRecovered: pageRecovered,
                    latest: latest
                ) {
                case .keepWatching(let withdrawn):
                    if withdrawn, !appearanceWithdrawnWhileShown { appearanceWithdrawnAt = Date() }
                    if !withdrawn { forgetTheWithdrawal() }
                    appearanceWithdrawnWhileShown = withdrawn
                case .leaveItToThePage:
                    forgetTheWithdrawal()
                case .waitOnePollForThePage:
                    appearanceWaitedForPage = true
                    interval = revisionPollInterval
                    continue
                case .replaceThePage:
                    let sameWalk = target.sessionID != nil
                        || WorldRenderRepresentation.session(of: latest.revision)
                            == WorldRenderRepresentation.session(of: shownRevision)
                    if sameWalk {
                        interval = revisionPollInterval
                        // **The flag survives a failed fetch**.
                        // It used to be cleared before this call, and `refresh`
                        // returns silently on any fetch failure — so one
                        // unreachable moment, from the same Tower that had just
                        // finished the rebuild, disarmed the only recovery this
                        // screen had and left the wearer on "no longer served"
                        // until they closed it. On a pre-epoch world the page
                        // revision never changes, so that flag is
                        // the ONLY path back.
                        if await refresh(to: latest.revision, evenIfUnchanged: true,
                                         live: latest.live == true) {
                            forgetTheWithdrawal()
                        }
                        continue
                    }
                    // Another walk's world. The flag is deliberately NOT
                    // cleared: this walk's imagery is still withdrawn on the
                    // page, and the rules below offer the other walk's build
                    // rather than swapping the reader's world out.
                }
            }
            if latest.revision == shownRevision, newerPictureAvailable {
                // The build that was offered is no longer what the Tower would
                // serve; the button must not outlive its reason.
                pendingRevision = nil
                newerPictureAvailable = false
            }
            let isNew = latest.revision != shownRevision
                && latest.revision != handledRevision
                && !refusedRevisions.contains(latest.revision)
            interval = Self.nextPollInterval(
                after: interval, base: revisionPollInterval, ceiling: revisionPollCeiling,
                live: latest.live, changed: isNew
            )
            guard isNew else { continue }
            if let rung = latest.representation, refusedRungs.contains(rung) {
                if latest.live == false, !finishedBuildRetried.contains(rung) {
                    // The finished build gets one more try (see
                    // `finishedBuildRetried`), and goes through every rule below.
                    finishedBuildRetried.insert(rung)
                } else {
                    // This phone could not draw that rung twice; neither
                    // swapped nor offered. "Try again" still can.
                    handledRevision = latest.revision
                    continue
                }
            }
            if WorldRenderRepresentation.isUpgrade(from: latest.representation, to: state.representation) {
                // A worse rung than the one on screen -- a surface briefly
                // undrawable on the Tower, a pruned artifact -- is not "a newer
                // reconstruction" and is not offered as one.
                handledRevision = latest.revision
                continue
            }
            // With no session named the Tower answers its newest session with
            // geometry, which can be another walk; the session in the revision
            // is what tells the two apart.
            let sameWalk = target.sessionID != nil
                || WorldRenderRepresentation.session(of: latest.revision)
                    == WorldRenderRepresentation.session(of: shownRevision)
            if Self.swapsBySelf(shown: state.representation, latest: latest, sameWalk: sameWalk) {
                await refresh(to: latest.revision, live: latest.live == true)
            } else {
                handledRevision = latest.revision
                pendingRevision = latest.revision
                newerPictureAvailable = true
            }
        }
    }

    /// Whether a new revision replaces the page without asking.
    ///
    /// A better rung always does. So does a build of the same rung that the
    /// Tower says is FINISHED (`live == false`): that is the final surface
    /// built after Stop, a world does not rebuild again after it, and a wearer
    /// who stopped walking should see the finished world rather than a button.
    /// A same-rung build while the walk is live is offered, because live builds
    /// land on every solve and a swap resets the camera mid-look. Nothing
    /// replaces a page with a worse rung by itself.
    ///
    /// Nothing swaps by itself unless `sameWalk`: the screen names its session,
    /// or the revision names the session already on screen. With no session in
    /// the target the Tower picks the newest session with geometry, so a new
    /// revision -- a better rung as much as a finished build -- can be ANOTHER
    /// walk's world, and swapping it in without asking would replace the world
    /// the reader opened. It is
    /// offered instead. No default: a caller that forgets to say was the
    /// regression easy to miss at the call site.
    nonisolated static func swapsBySelf(
        shown: WorldRenderRepresentation?, latest: WorldRenderRevision, sameWalk: Bool
    ) -> Bool {
        guard sameWalk else { return false }
        if WorldRenderRepresentation.isUpgrade(from: shown, to: latest.representation) {
            return true
        }
        let downgrade = WorldRenderRepresentation.isUpgrade(
            from: latest.representation, to: shown
        )
        return !downgrade && latest.live == false
    }

    /// What one poll means for an appearance page on screen. Pure, so the whole
    /// rule is tested without a page, a web view or a network.
    ///
    /// - a served appearance, nothing withdrawn: `keepWatching(false)`;
    /// - a served appearance of this rung after a withdrawal: the page's own
    ///   recovery first (`leaveItToThePage` if it already fetched a manifest,
    ///   else `waitOnePollForThePage` once, then `replaceThePage`);
    /// - a served appearance the app is NOT being given (the revision names a
    ///   lower rung): `keepWatching(false)` — there is nothing to reload into;
    /// - `rebuilding` (the ordinary Stop): the flag is unchanged, because that
    ///   is not a withdrawal and the page kept its textures;
    /// - anything else with no served appearance: `keepWatching(true)`.
    nonisolated static func appearanceFollow(
        withdrawn: Bool, waitedForThePage: Bool, pageRecovered: Bool, latest: WorldRenderRevision
    ) -> WorldAppearanceFollow {
        guard latest.appearance != nil else {
            if latest.appearanceState == "rebuilding" { return .keepWatching(withdrawn: withdrawn) }
            return .keepWatching(withdrawn: true)
        }
        guard withdrawn, latest.representation == .appearance else {
            return .keepWatching(withdrawn: false)
        }
        if pageRecovered { return .leaveItToThePage }
        return waitedForThePage ? .replaceThePage : .waitOnePollForThePage
    }

    /// Why an area viewer will not draw `html`, or `nil` when it may: the
    /// page's `wb-area` must be this area, and its `wb-revision` must name this
    /// session's area (`<s>/area:<a>/…`, `WORLD-BUILDER-COMPONENTS.md` §5.1).
    /// Always `nil` for the room.
    nonisolated static func areaPageMismatch(html: String, target: WorldRenderTarget) -> String? {
        guard let areaID = target.areaID, let sessionID = target.sessionID else { return nil }
        let declaredArea = WorldRenderRepresentation.meta(named: "wb-area", in: html)
        let revision = WorldRenderRepresentation.meta(named: "wb-revision", in: html)
        guard declaredArea == areaID, revision?.hasPrefix("\(sessionID)/area:\(areaID)/") == true else {
            return "The Tower's page is not a page of this area, so it was not drawn."
        }
        return nil
    }

    /// Whether an appearance page omitted the session needed for imagery.
    /// Without it, `WorldAssetRequest.parse` refuses every imagery route. Pure.
    nonisolated static func pageCannotReachItsImagery(
        html: String, target: WorldRenderTarget
    ) -> Bool {
        WorldRenderRepresentation.declared(in: html) == .appearance
            && WorldAssetSchemeHandler.session(for: target, page: html) == nil
    }

    /// The withdrawal is over, however it ended.
    private func forgetTheWithdrawal() {
        appearanceWithdrawnWhileShown = false
        appearanceWaitedForPage = false
        appearanceWithdrawnAt = nil
    }

    /// How long to wait before the next ask.
    ///
    /// Back to `base` whenever the revision changed or the Tower is building;
    /// otherwise double, up to `ceiling`. A Tower that does not say (`nil`) is
    /// treated as not building: slower, never silent.
    nonisolated static func nextPollInterval(
        after current: Duration, base: Duration, ceiling: Duration, live: Bool?, changed: Bool
    ) -> Duration {
        if changed || live == true { return base }
        return min(max(current * 2, base), ceiling)
    }

    /// Swap in the newer same-rung picture the reader asked for.
    func showNewerPicture() async {
        guard let revision = pendingRevision, case .ready = state else { return }
        pendingRevision = nil
        newerPictureAvailable = false
        // A tap is not the follower's live swap; its failure is not owed a retry.
        await refresh(to: revision, live: false)
    }

    /// Fetch the newer page and put it on screen, or leave the old one.
    ///
    /// Never `.fetching` and never `.failed`: the reader is looking at a world,
    /// and a refresh that fails must not take it away -- not the fetch, and not
    /// the draw either (see `fallback`). `renderAttempt` is NOT bumped, so the
    /// web view's content-process kill budget is not reset by an automatic
    /// refresh. The page's own stamped revision is recorded, not the polled one,
    /// so a build that lands between the poll and the fetch costs no second
    /// download.
    /// Returns whether a page was actually put on screen to be drawn — which is
    /// what tells a withdrawal-recovery that it worked (`WorldAppearanceFollow`).
    @discardableResult
    private func refresh(
        to revision: String, evenIfUnchanged: Bool = false, live: Bool = false
    ) async -> Bool {
        let html: String
        do {
            html = try await client.page(for: target)
        } catch {
            return false
        }
        guard !Task.isCancelled, case .ready(let current) = state else { return false }
        handledRevision = revision
        let stamped = WorldRenderViewerState.ready(html: html).revision ?? revision
        // `evenIfUnchanged`: the page on screen dropped its textures, so the
        // same page must be loaded again (a new attempt reloads an identical
        // string; see `renderAttempt`).
        if evenIfUnchanged, html == current {
            renderWatchdog?.cancel()
            // Keep a fallback, like every other path into `.rendering`.
            // Clearing it here left an identical-page reload that hit the
            // watchdog or a `didFail` with nothing to revert to; a world that
            // had been drawing a moment earlier went away. Reverting to the
            // same string is not a no-op: it
            // reloads it with a fresh kill budget and offers "Try again".
            fallback = (html: current, revision: shownRevision, wasLive: live, fetchedAt: pageFetchedAt)
            shownRevision = stamped
            pageFetched()
            // The identity changes so an identical string is loaded again; the
            // BUDGET token does not, because nobody asked for a fresh
            // content-process-kill budget.
            renderAttempt += 1
            state = .rendering(html: html)
            startRenderWatchdog()
            return true
        }
        guard evenIfUnchanged || (html != current
              && WorldRenderRepresentation.withoutRevisionStamp(html)
                != WorldRenderRepresentation.withoutRevisionStamp(current))
        else {
            // The same picture, possibly now carrying the stamp it lacked.
            shownRevision = stamped
            return false
        }
        // The page is what decides, not the poll that led to it. A tapped offer
        // whose build has since become undrawable on the Tower, or a fetch that
        // raced a manifest replace, is served a lower rung, and swapping that in
        // replaced a surface with points.
        guard !WorldRenderRepresentation.isUpgrade(
            from: WorldRenderRepresentation.declared(in: html), to: state.representation
        ) else { return false }
        guard !refusedRevisions.contains(stamped) else { return false }
        // Whatever was offered is superseded by the page now on its way.
        pendingRevision = nil
        newerPictureAvailable = false
        renderWatchdog?.cancel()
        fallback = (html: current, revision: shownRevision, wasLive: live, fetchedAt: pageFetchedAt)
        shownRevision = stamped
        pageFetched()
        state = .rendering(html: html)
        startRenderWatchdog()
        return true
    }

    /// Put the last drawn page back after an automatic refresh failed to draw.
    ///
    /// Returns `false` when there is nothing to fall back to -- no refresh was
    /// in flight, or the fallback page itself is what failed -- and the caller
    /// reports the failure as before. Bounded: the refused revision is never
    /// swapped in again, and `fallback` is consumed here.
    private func revertRefresh() -> Bool {
        guard let previous = fallback else { return false }
        fallback = nil
        if let refused = shownRevision { refusedRevisions.insert(refused) }
        if let handled = handledRevision { refusedRevisions.insert(handled) }
        if previous.wasLive {
            if let refused = shownRevision { refusedWhileLive.insert(refused) }
            if let handled = handledRevision { refusedWhileLive.insert(handled) }
        }
        if let rung = state.representation,
           rung != WorldRenderRepresentation.declared(in: previous.html) {
            let failures = rungDrawFailures[rung, default: 0] + 1
            rungDrawFailures[rung] = failures
            if failures >= 2 { refusedRungs.insert(rung) }
        }
        newerPictureRefused = true
        shownRevision = previous.revision
        pageFetchedAt = previous.fetchedAt
        recomputeBasis()
        // A fresh kill budget for a page that already drew once on this phone.
        renderAttempt += 1
        renderBudgetToken += 1
        state = .rendering(html: previous.html)
        startRenderWatchdog()
        return true
    }

    // No `deinit` cancelling the watchdog, deliberately. It captures `[weak
    // self]`, so a dismissed sheet is released immediately and the task that
    // outlives it holds nothing but a sleeping timer — and a `deinit` on a
    // global-actor-isolated class reaching for an isolated stored property is
    // the kind of thing that is legal today and an error the next time the
    // language mode moves. The `[weak self]` is the guarantee; the `deinit`
    // would only have been an optimisation of a timer.

    /// Fetch this target's page and start a bounded render. It clears the
    /// previous attempt, refetches once if an appearance page lacks its session,
    /// and checks the returned page's session and area. It then enters rendering
    /// or reports a failure with retryability set by the failure path. Fetch
    /// success alone never claims the web view has drawn the world.
    func load() async {
        renderWatchdog?.cancel()
        renderWatchdog = nil
        renderAttempt += 1
        renderBudgetToken += 1
        forgetTheWithdrawal()
        fallback = nil
        pendingRevision = nil
        newerPictureAvailable = false
        handledRevision = nil
        rungDrawFailures = [:]
        refusedRungs = []
        finishedBuildRetried = []
        // The refused REVISIONS too, as `WORLD-BUILDER-IOS.md` §10 and the
        // property's own comment always said. They were the one refusal this
        // kept, and for the appearance rung that is not a detail: its page
        // revision (`<session>/appearance:1@<epoch>`) does not change between
        // the builds of one epoch, and the ordinary Stop keeps the epoch. So
        // one walk-time appearance page that failed to draw refused the
        // FINISHED world's page as well, for the life of the screen -- and a
        // "Try again" tapped while the Tower served the surface in the Stop
        // gap cleared the button without clearing the refusal, leaving the
        // bare surface on screen over a photographic world with nothing
        // saying so.
        refusedRevisions = []
        refusedWhileLive = []
        newerPictureRefused = false
        state = .fetching
        do {
            var html = try await client.page(for: target)
            guard !Task.isCancelled else { return }
            if Self.pageCannotReachItsImagery(html: html, target: target) {
                // One refetch, then say so. The Tower composes
                // an appearance page with no `wb-revision` stamp when
                // `_appearance_revision` raced away between choosing the rung
                // and stamping it (WORLDS §4a rule 7). With no pinned session
                // that stamp is the ONLY thing that tells the scheme handler
                // which session's routes to proxy, so every manifest, chunk and
                // proxy request is refused locally, nothing reaches the Tower,
                // the Tower log shows nothing, and the field report reads "the
                // world opened empty". The race is narrow and the refetch wins
                // it; the sentence is for when it does not.
                html = try await client.page(for: target)
                guard !Task.isCancelled else { return }
                if Self.pageCannotReachItsImagery(html: html, target: target) {
                    state = .failed(
                        message: "The Tower's page for this world arrived without the session its "
                            + "images belong to, so none of them could be fetched.",
                        retryable: true
                    )
                    return
                }
            }
            // An area viewer draws only the area it was opened for:
            // the page must say so in its own head. A page naming another
            // area, another session, or no area at all is a Tower answering a
            // different question, and drawing it would put the wrong walk's
            // imagery under this area's caption.
            if let refusal = Self.areaPageMismatch(html: html, target: target) {
                state = .failed(message: refusal, retryable: true)
                return
            }
            // Not `.ready`. The page has arrived; nothing has drawn it yet.
            state = .rendering(html: html)
            shownRevision = state.revision
            pageFetched()
            startRenderWatchdog()
        } catch let error as WorldRenderFetchError {
            // A dismissed sheet cancels the task mid-fetch; that is not a
            // failure to report, and nobody is looking.
            guard !Task.isCancelled else { return }
            state = .failed(message: error.message, retryable: error.isRetryable)
        } catch {
            guard !Task.isCancelled else { return }
            state = .failed(message: error.localizedDescription, retryable: true)
        }
    }

    /// The rendering panel that `.rendered` took away has left the screen:
    /// the native chrome under it is visible, and the page may now be told
    /// to put its own away (WORLDS §4c). Not if the page went back to a
    /// bounded wait in the meantime -- that is a new page, with its own
    /// `didFinish` to come.
    func renderingPanelGone() {
        guard case .ready = state else { return }
        bridge.receive(.pageFinished)
    }

    /// What the page did. The only route from the web view into this model.
    func pageEvent(_ event: WorldRenderPageEvent) {
        renderWatchdog?.cancel()
        renderWatchdog = nil
        switch event {
        case .rendered:
            // The panel goes with this turn's state; the chrome hears of it
            // after the commit (`renderingPanelGone`).
            defer { if case .ready = state { pageFinishedToken += 1 } }
            // Only from `.rendering`. A `didFinish` that arrives after a
            // timeout already reported failure must not quietly un-fail the
            // screen underneath a reader who has started reading the message.
            guard case .rendering(let html) = state else { return }
            state = .ready(html: html)
            if fallback != nil, let rung = state.representation {
                // A refresh to this rung DREW. Whatever failed it before has
                // passed; a later failure starts counting again.
                rungDrawFailures[rung] = nil
                refusedRungs.remove(rung)
                newerPictureRefused = false
            }
            fallback = nil
        case .reloadingAfterTermination, .reloadingItself:
            // Back to a bounded wait, with the overlay over it. Truthful: the
            // page really is being drawn again. The two cases differ only in
            // who asked — WebKit's content process died, or the page gave up on
            // a WebGL context it was never given back — and from the reader's
            // side they are the same wait.
            guard let html = state.html else { return }
            state = .rendering(html: html)
            startRenderWatchdog()
        case .failed(let detail):
            if revertRefresh() { return }
            state = .failed(
                message: "The Tower's page arrived but could not be drawn: \(detail)",
                retryable: true
            )
        case .gaveUpAfterTerminations(let count):
            if revertRefresh() { return }
            // Retryable, and the sentence says why it stopped on its own.
            // Memory pressure is transient — closing another app can genuinely
            // change the answer — so this is not a control that cannot work.
            // What it must not do is keep reloading a page that keeps killing
            // the process, which is what it did before there was a budget.
            state = .failed(
                message: "This world was too large to draw on this phone: the page ran out of "
                    + "memory \(count) times and was not reloaded again.",
                retryable: true
            )
        }
    }

    /// Fail truthfully if the page never reports finishing -- or, when the page
    /// that did not finish was an automatic refresh, put the previous one back.
    ///
    /// The message says the fetch succeeded, because it did — blaming the Tower
    /// for a page it delivered would send the reader to the wrong machine.
    private func startRenderWatchdog() {
        let timeout = renderTimeout
        renderWatchdog = Task { [weak self] in
            try? await Task.sleep(for: timeout)
            guard !Task.isCancelled, let self, self.state.isRendering else { return }
            if self.revertRefresh() { return }
            self.state = .failed(
                message: "The Tower's page arrived but did not finish drawing. This world may be "
                    + "too large to render on this phone.",
                retryable: true
            )
        }
    }
}
