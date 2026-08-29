//
//  ObjectMemoryClient.swift
//  Glasses
//

import Combine
import Foundation

// MARK: - Transport failures

/// Why a request to the Tower's object memory did not produce an answer.
///
/// Four cases, and the first two are **not** errors about the wearer's memory:
/// they are facts about the Tower. Collapsing them into "something went wrong"
/// is how a Tower with no object memory configured comes to look like a person
/// who has never seen a laptop.
enum ObjectMemoryFetchError: Error, Equatable {
    /// **404.** `TOWER_OBSERVATION_ROOT` is unset: this Tower serves no object
    /// memory at all.
    ///
    /// A statement about configuration, and never the answer to a question
    /// about a class — the Tower is explicit that an unobserved class is
    /// answered 200 with `observed: false` precisely so that a 404 can keep
    /// meaning only this.
    case noObjectMemoryConfigured
    /// The Tower's object memory speaks a different agreement. Opaque and
    /// compared for equality: not newer, not older, just different.
    case unsupportedContract(identifier: String)
    /// The answer arrived and could not be read as this contract.
    case undecodable
    /// **The Tower answered, and the answer was a status this route does not
    /// define.**
    ///
    /// Deliberately **not** `.transport`, and this case exists because the
    /// absence of it was a real defect. Every non-404 non-2xx used to become
    /// `.transport("The Tower answered N.")`, which `ObjectMemoryState.phase`
    /// maps to `.disconnected` — so a Tower that replied clearly was presented
    /// to a wearer as a connection failure, telling them to check a network
    /// about a machine that had just answered them. The Tower was reached; the
    /// answer was one this build cannot use. Those are different sentences and
    /// different glyphs, and only one of them is true.
    case towerAnswered(status: Int)
    /// The request did not complete. The Tower is unreachable, or the network
    /// is.
    case transport(String)
}

// MARK: - HTTP

/// The two `GET`s, and nothing else.
///
/// ## Read-only, deliberately and permanently
///
/// `ObservationStore` has `purge()` and `prune_expired()`. **Neither is on the
/// wire**, and the Tower has two tests holding it there. This client has no
/// delete, no write, and no method that could grow into one — real deletion
/// lives with `scripts/object_query.py --purge-all`, where a human types it
/// against a store they can name. An unauthenticated LAN endpoint that erases a
/// wearer's memory is not a feature, and neither is a phone button that calls
/// one.
///
/// ## Why this mirrors `WorldGeometryClient` rather than inventing anything
///
/// Same shape, same `JSONSerialization` decoding, same "a 404 is its own case"
/// handling, same injectable `URLSession`. A second, differently-opinionated
/// networking layer in one app is two places for a timeout policy to be wrong.
///
/// ## Bounded, and uncached
///
/// Every request carries an explicit timeout (Rule 15: bounded operations) and
/// `reloadIgnoringLocalCacheData`. A memory query answered out of a URL cache
/// would show a record as current that the retention window may since have
/// closed over, which is the one kind of staleness this cartridge cannot
/// tolerate.
nonisolated struct ObjectMemoryHTTPClient {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession = .shared
    /// Long enough for a Tailscale round trip to a Tower reading a JSONL file,
    /// short enough that a dead Tower becomes a visible state rather than a
    /// spinner nobody can end.
    var timeout: TimeInterval = 10

    /// Every observation still within the retention window, newest first.
    ///
    /// - Parameters:
    ///   - objectClass: narrow to one category, or `nil` for all of them.
    ///   - retentionDays: **narrow** the window this read may see. It cannot
    ///     widen it: the store clamps every read to `min(persisted, requested)`
    ///     and reports the clamp in `retention.clamped`. Passing a large number
    ///     here does not recover an expired record and never will.
    func listing(
        objectClass: String? = nil, retentionDays: Double? = nil
    ) async throws -> ObservationListing {
        var query: [URLQueryItem] = []
        if let objectClass {
            query.append(URLQueryItem(name: "object_class", value: objectClass))
        }
        appendRetention(retentionDays, to: &query)

        let url = baseURL.appendingPathComponent("object-memory/observations")
        let json = try await get(url, query: query)
        try requireThisContract(json)
        return try ObjectMemoryDecoder.listing(from: json)
    }

    /// When a category was last in view, or an honest silence.
    ///
    /// Always 200 when a root is configured — there is no 404 for an unobserved
    /// class, because "not found" reads as "there is no laptop", which is a
    /// claim about the world.
    ///
    /// `objectClass` is appended as its own path component so that a class with
    /// a space in it (`cell phone`, which is one of the two real ones) is
    /// percent-encoded rather than splitting the path.
    func lastSeen(
        objectClass: String, retentionDays: Double? = nil
    ) async throws -> LastSeenAnswer {
        var query: [URLQueryItem] = []
        appendRetention(retentionDays, to: &query)

        let url = baseURL
            .appendingPathComponent("object-memory/last-seen")
            .appendingPathComponent(objectClass)
        let json = try await get(url, query: query)
        try requireThisContract(json)
        return try ObjectMemoryDecoder.lastSeen(from: json)
    }

    /// `0` is a meaningful value on this parameter — "no limit of my own",
    /// still clamped to the persisted window — so it is only omitted when the
    /// caller asked for nothing at all. A negative value is refused here rather
    /// than being sent for the route to answer 422.
    private func appendRetention(_ days: Double?, to query: inout [URLQueryItem]) {
        guard let days, days >= 0 else { return }
        query.append(URLQueryItem(name: "retention_days", value: String(days)))
    }

    /// Equality, and nothing else. A different identifier is surfaced as its
    /// own case so the app can say "update the app" rather than decoding a
    /// payload whose fields may mean something new.
    private func requireThisContract(_ json: [String: Any]) throws {
        guard let identifier = ObjectMemoryDecoder.contractIdentifier(from: json) else {
            throw ObjectMemoryFetchError.undecodable
        }
        guard identifier == ObjectMemoryContract.identifier else {
            throw ObjectMemoryFetchError.unsupportedContract(identifier: identifier)
        }
    }

    private func get(_ url: URL, query: [URLQueryItem]) async throws -> [String: Any] {
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            throw ObjectMemoryFetchError.transport("The Tower address could not be read as a URL.")
        }
        // Empty rather than `[]`: an empty `queryItems` array puts a bare `?`
        // on the URL, and `nil` is what "no query" means.
        components.queryItems = query.isEmpty ? nil : query
        guard let requestURL = components.url else {
            throw ObjectMemoryFetchError.transport("The request URL could not be built.")
        }

        let request = URLRequest(
            url: requestURL,
            cachePolicy: .reloadIgnoringLocalCacheData,
            timeoutInterval: timeout
        )

        do {
            let (data, response) = try await session.data(for: request)
            if let http = response as? HTTPURLResponse, http.statusCode == 404 {
                throw ObjectMemoryFetchError.noObjectMemoryConfigured
            }
            if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                // Its own case, never `.transport`. See
                // `ObjectMemoryFetchError.towerAnswered` for the defect this
                // replaced: a Tower that answered was being reported as a
                // Tower that could not be reached, and the imagery routes —
                // whose 410 is a true and useful sentence about capture-side
                // retention — were the worst-affected caller of the two.
                throw ObjectMemoryFetchError.towerAnswered(status: http.statusCode)
            }
            // Parsed in its own `do`, because `jsonObject(with:)` *throws* on a
            // malformed body rather than returning something the cast rejects.
            // Left to the `catch` below, that throw would be relabelled
            // `.transport` — which the workspace maps to `.disconnected`, a
            // claim about the network that is false when the Tower answered.
            // A body that arrived and could not be read is a disagreement about
            // the answer, not a failure to get one.
            let parsed: Any
            do {
                parsed = try JSONSerialization.jsonObject(with: data)
            } catch {
                throw ObjectMemoryFetchError.undecodable
            }
            guard let json = parsed as? [String: Any] else {
                throw ObjectMemoryFetchError.undecodable
            }
            return json
        } catch let error as ObjectMemoryFetchError {
            throw error
        } catch {
            throw ObjectMemoryFetchError.transport(error.localizedDescription)
        }
    }
}

// MARK: - What this app has learned about the Tower's object memory

/// What a request has taught this app about the other machine.
///
/// ## Why this exists at all, and is not `TowerCapabilities`
///
/// World Builder's contract is **declared** over the socket, before anything is
/// asked for, so `TowerCapabilities` can resolve its availability from a
/// declaration it already holds. Object Memory's is not declared anywhere: it
/// travels in the `contract` field of an answer, so the only way to learn
/// whether this Tower serves object memory — and under which agreement — is to
/// ask it.
///
/// That makes `.unprobed` a real and unavoidable state, and it is the one thing
/// `CartridgeAvailability`'s four cases cannot express: "nothing has been
/// declared" and "nothing has been asked" are different, and only the first is
/// `.noContract`. So the knowledge lives here, `availability(isTowerReachable:)`
/// projects it onto the shared vocabulary for anything that needs the shared
/// vocabulary, and `knownAvailability(isTowerReachable:)` returns `nil` in the
/// one case where the projection would be a claim rather than a fact.
enum ObjectMemoryService: Equatable, Sendable {
    /// Nothing has been asked, so nothing is known. **Not** "unavailable".
    case unprobed
    /// A payload arrived under the contract this build implements.
    case serving(CartridgeContract)
    /// A payload arrived under a different agreement.
    case speaksAnotherContract(CartridgeContract)
    /// The Tower answered 404: it serves no object memory.
    case notConfigured
    /// The request did not complete.
    case unreachable(String)
}

// MARK: - State

/// What the Object Memory workspace should be showing.
///
/// ## Why a failure carries no answer
///
/// `CartridgePhase.mayCarryData` is false for every phase but `.live` and
/// `.settled`, and that invariant is checked across every cartridge at once. So
/// a failed refresh clears the answer rather than leaving the previous one on
/// screen beneath a warning. That is the stricter choice and it is the right
/// one here: a record whose age the reader can no longer trust is exactly what
/// a cartridge about *when something was visible* must not display.
enum ObjectMemoryState: Equatable, Sendable {
    /// Nothing has been asked yet. Not "empty" — nobody has asked a question.
    case idle
    case asking(ObjectMemoryQuestion)
    /// An answer, which may carry no records. **An empty answer is an answer**,
    /// not a failure, and `ObjectMemoryCopy` words it as one.
    case answered(question: ObjectMemoryQuestion, answer: ObjectMemoryAnswer)
    /// The Tower serves no object memory at all.
    case noObjectMemory
    case failed(CartridgeFailure)

    var answer: ObjectMemoryAnswer? {
        if case .answered(_, let answer) = self { return answer }
        return nil
    }

    var question: ObjectMemoryQuestion? {
        switch self {
        case .asking(let question): return question
        case .answered(let question, _): return question
        case .idle, .noObjectMemory, .failed: return nil
        }
    }

    var phase: CartridgePhase {
        switch self {
        case .idle: return .idle
        case .asking: return .waiting
        // Settled even when it carries nothing: the Tower is not still working
        // on it, and a spinner over an honest silence would suggest otherwise.
        case .answered: return .settled
        case .noObjectMemory: return .unsupported
        case .failed(let failure):
            // A transport failure is the shell's `.disconnected`, not its
            // `.failed`: the capability exists and the Tower could not be
            // reached, which is a connection state and calls for a different
            // headline and glyph than a broken answer.
            return failure.kind == .transport ? .disconnected : .failed
        }
    }
}

// MARK: - The client

/// Supplies `ObjectMemoryState` to the Object Memory workspace.
///
/// A fourth interaction shape, after World Builder's continuous push,
/// Experimental CV's bounded job and Document Memory's point query: a
/// **read-only historical query over HTTP**, against a store that exists
/// whether or not a session is running. Nothing here streams, nothing
/// subscribes, and nothing is current.
@MainActor
protocol ObjectMemoryClient: CartridgeClient {
    var state: ObjectMemoryState { get }

    /// Every state after the one `state` held when the view model was built.
    var stateUpdates: AnyPublisher<ObjectMemoryState, Never> { get }

    /// What asking has taught this app about the Tower's object memory.
    var service: ObjectMemoryService { get }

    /// Availability, or `nil` while nothing has been asked and the projection
    /// onto the shared four cases would be a claim rather than a fact. See
    /// `ObjectMemoryService`.
    func knownAvailability(isTowerReachable: Bool) -> CartridgeAvailability?

    /// Asks one question. Never throws: an unreachable Tower, a Tower with no
    /// object memory, and an unreadable answer are all **states**, because all
    /// three are things the screen has to be able to say.
    func ask(_ question: ObjectMemoryQuestion) async

    // MARK: The session

    /// What was last read from `/cartridges/object_memory/session`.
    ///
    /// **This is intent, not liveness.** Read liveness from
    /// `session.isFollowingACapture`, which comes from `following`. See
    /// `ObjectMemorySession.swift`.
    var session: ObjectMemorySessionState { get }

    /// Every session state after the one `session` held when the view model
    /// was built.
    var sessionUpdates: AnyPublisher<ObjectMemorySessionState, Never> { get }

    /// Reads the session. The only way to learn liveness, and therefore the
    /// thing a workspace repeats while it is on screen — a `following` read
    /// once at launch is a claim about a producer that may have died since.
    func readSession() async

    /// Sends one verb. Never throws: a refusal is an outcome and a state.
    func apply(_ action: CartridgeSessionAction) async

    // MARK: The pictures

    /// Whether there is a picture behind a record, and what may be said about
    /// it. Answered **without** fetching any bytes.
    func imageryDescription(for observationID: String) async -> ObjectMemoryImageryAnswer

    /// The bytes, or the reason there are none.
    ///
    /// **The returned `Data` is not cached and must not be persisted by the
    /// caller.** Both binary routes send `Cache-Control: no-store`; a copy this
    /// app keeps is a second store nobody chose and nobody's retention
    /// governs.
    func picture(
        for observationID: String, kind: ObjectMemoryImageryKind
    ) async -> ObjectMemoryPictureAnswer
}

/// The Object Memory client, backed by the Tower's two read-only routes.
///
/// ## Runtime ownership
///
/// Owns no socket and touches no frame path. It holds a `URLSession`-backed
/// struct and makes a request when a person asks for one — there is no polling,
/// no subscription, and no timer. It is held in `CartridgeClients` so that an
/// answer survives a workspace switch, which is the whole reason that container
/// exists.
///
/// ## It holds no sample observations
///
/// For the reason `UnavailableDocumentMemoryClient` gives: a handful of
/// plausible rows would be indistinguishable on screen from real records of
/// what the wearer's own camera saw, which is the most misleading fixture this
/// app could ship. Fixtures live in the test target, where they cannot reach a
/// device.
@MainActor
final class TowerObjectMemoryClient: ObjectMemoryClient {

    let cartridgeID = "object-memory"

    private(set) var state: ObjectMemoryState = .idle {
        didSet {
            guard state != oldValue else { return }
            stateSubject.send(state)
        }
    }

    var stateUpdates: AnyPublisher<ObjectMemoryState, Never> {
        stateSubject.eraseToAnyPublisher()
    }

    private(set) var service: ObjectMemoryService = .unprobed

    private(set) var session: ObjectMemorySessionState = .unread {
        didSet {
            guard session != oldValue else { return }
            sessionSubject.send(session)
        }
    }

    var sessionUpdates: AnyPublisher<ObjectMemorySessionState, Never> {
        sessionSubject.eraseToAnyPublisher()
    }

    private let stateSubject = PassthroughSubject<ObjectMemoryState, Never>()
    private let sessionSubject = PassthroughSubject<ObjectMemorySessionState, Never>()
    private let http: ObjectMemoryHTTPClient
    private let control: CartridgeSessionHTTPClient
    private let pictures: ObjectMemoryImageryHTTPClient
    /// One question at a time. A second ask while one is in flight is dropped
    /// rather than queued: they are the same button, and two answers racing
    /// into one screen is a worse outcome than one ignored tap.
    private var isAsking = false
    /// The same discipline for the control surface, and a separate flag from
    /// `isAsking` on purpose: a person who presses Stop while a listing is
    /// loading must be obeyed, and making one button wait on the other's
    /// request would be the worst possible place to introduce a delay.
    private var isControlling = false

    /// The route templates from the most recent answer.
    ///
    /// Read off the payload rather than constructed here — see
    /// `ObjectMemoryImageryRoutes`. `nil` until something has been asked, which
    /// is why `ObjectMemoryImageryAnswer` has a `.routesUnknown` case: a
    /// workspace that has not queried has not been told where the pictures
    /// are, and that is a fact rather than a failure.
    private var imageryRoutes: ObjectMemoryImageryRoutes? {
        state.answer?.envelope.imagery
    }

    init(
        http: ObjectMemoryHTTPClient = ObjectMemoryHTTPClient(),
        control: CartridgeSessionHTTPClient = CartridgeSessionHTTPClient(),
        pictures: ObjectMemoryImageryHTTPClient = ObjectMemoryImageryHTTPClient()
    ) {
        self.http = http
        self.control = control
        self.pictures = pictures
    }

    // MARK: Availability

    /// The learned state of the Tower's object memory, projected onto the
    /// shell's shared four-case vocabulary.
    ///
    /// `.unprobed` with a reachable Tower projects to `.noContract`, which is
    /// the conservative reading — as far as this app has been *told*, nothing
    /// has been declared — and it is why the workspace uses
    /// `knownAvailability(isTowerReachable:)` instead. Hiding the one control
    /// that would find out, on the grounds that nobody has used it yet, is a
    /// loop that cannot terminate.
    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        switch service {
        // What the Tower can do outranks whether we can reach it, exactly as
        // `CartridgeAvailability.resolve` orders it: an unsupported agreement
        // is not fixed by reconnecting, and a Tower with no object memory is
        // not fixed by reconnecting either.
        case .speaksAnotherContract(let contract):
            return .unsupportedContract(declared: contract)
        case .notConfigured:
            return .noContract
        case .serving(let contract):
            return isTowerReachable ? .available(contract) : .towerUnreachable
        case .unreachable:
            return .towerUnreachable
        case .unprobed:
            return isTowerReachable ? .noContract : .towerUnreachable
        }
    }

    func knownAvailability(isTowerReachable: Bool) -> CartridgeAvailability? {
        if case .unprobed = service, isTowerReachable { return nil }
        return availability(isTowerReachable: isTowerReachable)
    }

    // MARK: Asking

    func ask(_ question: ObjectMemoryQuestion) async {
        guard !isAsking else { return }
        isAsking = true
        state = .asking(question)
        defer { isAsking = false }

        do {
            let answer: ObjectMemoryAnswer
            switch question {
            case .listing(let objectClass):
                answer = .listing(try await http.listing(objectClass: objectClass))
            case .lastSeen(let objectClass):
                answer = .lastSeen(try await http.lastSeen(objectClass: objectClass))
            }
            // Learned from the answer itself, which is the only place this
            // contract is ever declared.
            service = .serving(
                CartridgeContract(
                    cartridgeID: cartridgeID, identifier: answer.envelope.contract
                )
            )
            state = .answered(question: question, answer: answer)
        } catch let error as ObjectMemoryFetchError {
            apply(error)
        } catch {
            // A `CartridgeFailure` thrown by the decoder passes through
            // unchanged; anything else is `.transport`, which is the only
            // honest attribution available without knowing where it came from.
            //
            // `service` is deliberately left where it was. An answer this build
            // could not read says nothing about whether the Tower serves object
            // memory, and downgrading what we know on the strength of one bad
            // payload would report a Tower as absent because we misread it.
            state = .failed(CartridgeFailure.wrapping(error))
        }
    }

    private func apply(_ error: ObjectMemoryFetchError) {
        switch error {
        case .noObjectMemoryConfigured:
            service = .notConfigured
            // Its own state rather than a failure. This Tower serves no object
            // memory; nothing failed, and nothing about the wearer's memory is
            // being reported.
            state = .noObjectMemory

        case .unsupportedContract(let identifier):
            service = .speaksAnotherContract(
                CartridgeContract(cartridgeID: cartridgeID, identifier: identifier)
            )
            state = .failed(
                CartridgeFailure(
                    kind: .notSupported,
                    message: ObjectMemoryCopy.unsupportedContract(identifier)
                )
            )

        case .undecodable:
            // The service is left as it was: an unreadable answer says nothing
            // about whether this Tower serves object memory.
            state = .failed(
                CartridgeFailure(
                    kind: .undecodableResponse,
                    message: """
                        The Tower's object memory answered in a shape this app \
                        does not understand, so nothing is shown rather than \
                        something guessed.
                        """
                )
            )

        case .towerAnswered(let status):
            // The service is left as it was, for the reason `.undecodable`
            // leaves it: one unusable status says nothing about whether this
            // Tower serves object memory.
            //
            // `.towerReportedFailure`, not `.transport` — which is the whole
            // point of the case. `ObjectMemoryState.phase` sends `.transport`
            // to `.disconnected`, and this must reach `.failed`: the Tower is
            // right there and answering.
            state = .failed(
                CartridgeFailure(
                    kind: .towerReportedFailure,
                    message: ObjectMemoryCopy.towerAnswered(status)
                )
            )

        case .transport(let description):
            service = .unreachable(description)
            // The message carries only what this app learned, because
            // `CartridgeAvailability.towerUnreachable` already supplies the
            // sentence explaining what an unreachable Tower means — and the
            // two are joined before they reach the screen. Repeating the
            // framing here would print it twice.
            state = .failed(
                CartridgeFailure(
                    kind: .transport, message: "The Tower did not answer: \(description)"
                )
            )
        }
    }

    // MARK: The session

    /// Reads `/cartridges/object_memory/session`.
    ///
    /// The read does **not** go through `isControlling`: a refresh must never
    /// be dropped because an action is in flight, since the whole reason to
    /// refresh is to find out what the action actually did to `following`.
    func readSession() async {
        do {
            session = .known(try await control.read())
        } catch {
            applySessionFailure(error)
        }
    }

    /// Sends one verb, then reads the session back.
    ///
    /// **The read-back is the point, not politeness.** A `POST` answers with
    /// the state the Tower moved to, and that is intent. Whether the producer
    /// actually let go of the capture is `following`, sampled after the fact —
    /// and a Pause whose producer ignores `SIGTERM` answers 200 with
    /// `state: "paused"` while still following. Trusting the `POST` body alone
    /// would draw exactly the screen this cartridge must never draw.
    func apply(_ action: CartridgeSessionAction) async {
        guard !isControlling else { return }
        isControlling = true
        session = .working
        defer { isControlling = false }

        let outcome: CartridgeSessionOutcome
        do {
            outcome = try await control.apply(action)
        } catch {
            applySessionFailure(error)
            return
        }

        // Shown immediately, so a refusal is not held back behind a second
        // request. A refusal is a state, not a failure: "resume continues a
        // paused session; this one is stopped" is a true and actionable
        // sentence, and routing it through `.failed` would put it under an
        // error glyph beside "the Tower is unreachable".
        switch outcome {
        case .honoured(let snapshot): session = .known(snapshot)
        case .refused(let refusal): session = .refused(refusal)
        }

        // Then re-read, rather than trusting what the action claimed. `stop`
        // in particular is never refused, which means a 200 from it is not
        // evidence that anything stopped.
        let fresh: CartridgeSessionSnapshot
        do {
            fresh = try await control.read()
        } catch {
            applySessionFailure(error)
            return
        }

        switch outcome {
        case .honoured:
            session = .known(fresh)
        case .refused(let refusal):
            // The refusal's *explanation* survives the refresh and its
            // *reading* does not. Replacing the whole state with `.known`
            // here would make the sentence that says which control would have
            // worked flash and vanish, and keeping the refusal's own stale
            // snapshot would leave a liveness claim from before the round
            // trip on screen — which is the one thing this surface may not do.
            session = .refused(
                CartridgeSessionRefusal(
                    action: refusal.action,
                    reason: refusal.reason,
                    message: refusal.message,
                    snapshot: fresh
                )
            )
        }
    }

    private func applySessionFailure(_ error: Error) {
        guard let error = error as? CartridgeSessionFetchError else {
            session = .failed(CartridgeFailure.wrapping(error))
            return
        }
        switch error {
        case .noSuchCartridgeSession:
            // Configuration, not failure. This Tower has no producer to start.
            session = .noSessionControl

        case .unsupportedContract(let identifier):
            session = .failed(
                CartridgeFailure(
                    kind: .notSupported,
                    message: ObjectMemoryCopy.unsupportedSessionContract(identifier)
                )
            )

        case .undecodable:
            session = .failed(
                CartridgeFailure(
                    kind: .undecodableResponse,
                    message: ObjectMemoryCopy.unreadableSessionAnswer
                )
            )

        case .transport(let description):
            session = .failed(
                CartridgeFailure(
                    kind: .transport, message: "The Tower did not answer: \(description)"
                )
            )
        }
    }

    // MARK: The pictures

    func imageryDescription(for observationID: String) async -> ObjectMemoryImageryAnswer {
        guard let routes = imageryRoutes else { return .routesUnknown }
        return await pictures.description(for: observationID, routes: routes)
    }

    func picture(
        for observationID: String, kind: ObjectMemoryImageryKind
    ) async -> ObjectMemoryPictureAnswer {
        guard let routes = imageryRoutes else { return .routesUnknown }
        return await pictures.picture(for: observationID, kind: kind, routes: routes)
    }
}

// MARK: - View model

/// Publishes Object Memory state into SwiftUI.
///
/// Holds no runtime references, for the reason `WorldBuilderViewModel` gives:
/// it is destroyed and rebuilt on every workspace switch, and anything durable
/// belongs on the client in `CartridgeClients`.
@MainActor
final class ObjectMemoryViewModel: ObservableObject {
    @Published private(set) var state: ObjectMemoryState

    /// The session, as last read. **Intent, not liveness** — see `liveness`.
    @Published private(set) var session: ObjectMemorySessionState

    /// The category the next question will narrow to, or `nil` for all of them.
    ///
    /// Owned here rather than in the view so a question can also arrive from
    /// somewhere with no picker — the same seam `DocumentMemoryViewModel`
    /// keeps for an input layer that does not exist yet.
    @Published var selectedClass: String?

    private let client: any ObjectMemoryClient
    private var cancellables: Set<AnyCancellable> = []
    /// The repeating session read. Owned here so it dies with the workspace;
    /// see `startWatchingSession`.
    private var livenessWatch: Task<Void, Never>?

    /// No default argument, for the reason `WorldBuilderViewModel.init(client:)`
    /// gives: a default would make constructing a second client at the point of
    /// use the path of least resistance, and this one must be the app's single
    /// long-lived instance.
    init(client: any ObjectMemoryClient) {
        self.client = client
        self.state = client.state
        self.session = client.session

        client.stateUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in self?.state = state }
            .store(in: &cancellables)

        client.sessionUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] session in self?.session = session }
            .store(in: &cancellables)
    }

    // MARK: What the screen may say

    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        client.availability(isTowerReachable: isTowerReachable)
    }

    func knownAvailability(isTowerReachable: Bool) -> CartridgeAvailability? {
        client.knownAvailability(isTowerReachable: isTowerReachable)
    }

    /// The phase to draw.
    ///
    /// Availability outranks the client's own state **once it is known** — a
    /// Tower that has told us it serves no object memory must not render as
    /// `.idle`, which would invite a person to press a button that cannot work.
    /// Before anything has been asked there is nothing to outrank it with, and
    /// the state's own `.idle` is the truthful answer.
    func phase(isTowerReachable: Bool) -> CartridgePhase {
        knownAvailability(isTowerReachable: isTowerReachable)?.forcedPhase ?? state.phase
    }

    func unavailableExplanation(isTowerReachable: Bool) -> String {
        guard let availability = knownAvailability(isTowerReachable: isTowerReachable) else {
            return ""
        }
        // `.noContract` reaches this cartridge by a different route than it
        // reaches the others, and the shared sentence — "this Tower has not
        // declared a contract" — would be inaccurate for it. The Tower *did*
        // answer; it answered 404, which is a statement about how it is
        // configured. Its own wording is the honest one, and it stands alone.
        if case .noContract = availability {
            return ObjectMemoryCopy.noObjectMemoryConfigured
        }
        return availability.explanation(
            cartridgeName: ObjectMemoryCopy.cartridgeName, clientReason: clientReason
        )
    }

    private var clientReason: String? {
        switch state {
        case .noObjectMemory: return ObjectMemoryCopy.noObjectMemoryConfigured
        case .failed(let failure): return failure.message
        case .idle, .asking, .answered: return nil
        }
    }

    /// The categories this Tower's object memory can ever record, learned from
    /// the last answer.
    ///
    /// Empty until something has been asked, because until then this app has
    /// not been told — and a hardcoded `["laptop", "cell phone"]` would be this
    /// app asserting what the Tower looks for.
    var recordableClasses: [String] {
        state.answer?.envelope.recordedClasses ?? []
    }

    // MARK: Asking

    func askForEverything() {
        ask(.listing(objectClass: selectedClass))
    }

    func askWhenLastInView() {
        guard let objectClass = selectedClass else { return }
        ask(.lastSeen(objectClass: objectClass))
    }

    /// Re-asks whatever question is standing, without changing it.
    ///
    /// **The refresh a Stop triggers, and the reason it is not
    /// `askForEverything()`.** That call builds `.listing(objectClass:)`, so it
    /// preserved the reader's *category* and silently replaced the *shape* of
    /// their question: somebody who had asked "when was a laptop last in view"
    /// pressed Stop and got a listing back. Two of the three things a reader
    /// chose — the shape and the class — have to survive a refresh, or the
    /// refresh is a different question wearing the old one's answer.
    ///
    /// `state.question` carries both, for `.asking` and `.answered` alike.
    /// `askForEverything()` is the fallback for `.idle`, `.noObjectMemory` and
    /// `.failed`, where there is no standing question to preserve and a listing
    /// is the widest honest thing to show.
    func askAgain() {
        guard let standing = state.question else {
            askForEverything()
            return
        }
        ask(standing)
    }

    func ask(_ question: ObjectMemoryQuestion) {
        Task { await client.ask(question) }
    }

    // MARK: The session

    /// How often `following` is re-read while this workspace is on screen.
    ///
    /// **Liveness has a shelf life.** `following` names the captures a producer
    /// is alive on *right now*; a value read once when the screen opened is a
    /// claim about a process that may have died since, and this cartridge's
    /// whole reason for having a control surface is that a producer dying
    /// silently is the failure it was built to make visible.
    ///
    /// Three seconds because a `GET` here reads a dictionary and nothing else
    /// — no disk, no model — and because a person who presses Pause and
    /// watches for the recording indicator to go out should not be left
    /// wondering for longer than that.
    static let livenessRefreshInterval: Duration = .seconds(3)

    /// Whether a capture is being recorded into this memory **right now**.
    ///
    /// **Derived from `following`, and from nothing else.** Not from `state`,
    /// which is intent; not from `sessionID`, which survives a Pause; not from
    /// `captures`, which is a history. `nil` means nothing has been read yet,
    /// and it is `nil` rather than `false` because "we have not asked" is not
    /// "nothing is being recorded" — drawing the second from the first is the
    /// same error as drawing "recording" from `state`, one step earlier.
    var liveness: Bool? { session.isFollowingACapture }

    /// Whether the Tower's own two fields contradict each other in the
    /// direction that harms a person: an action reported as honoured while the
    /// producer kept recording. Shown loudly when true.
    var intentContradictsLiveness: Bool {
        session.snapshot?.intentContradictsLiveness ?? false
    }

    /// The verbs to draw, read off the Tower's `actions` rather than
    /// hard-coded. Empty until a session has been read, which is honest: this
    /// app has not been told what the surface offers.
    var offeredActions: [CartridgeSessionAction] {
        session.snapshot?.offeredActions ?? []
    }

    func readSession() {
        Task { await client.readSession() }
    }

    /// Sends one verb straight at the Tower.
    ///
    /// **The workspace does not call this any more, and new callers should
    /// think before they do.** It moves the Tower's producer and nothing else,
    /// so a `stop` sent through here ends the session and leaves the glasses
    /// camera streaming, and a `start` opens the gate with no capture behind
    /// it — which is precisely the half-finished product
    /// `ObjectMemoryRecordingCoordinator` was written to replace. It is kept
    /// because the Tower half is genuinely separable and a future caller with
    /// no camera at all is a real possibility; it is documented because the
    /// obvious call site is the wrong one.
    func apply(_ action: CartridgeSessionAction) {
        Task { await client.apply(action) }
    }

    /// Starts re-reading the session while the workspace is visible.
    ///
    /// Bounded by cancellation rather than by a count (Rule 15): the task is
    /// owned here and torn down in `stopWatchingSession`, which the view calls
    /// on disappear. A second call replaces the first rather than running two.
    func startWatchingSession() {
        stopWatchingSession()
        livenessWatch = Task { [weak self] in
            guard let self else { return }
            await self.client.readSession()
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: Self.livenessRefreshInterval)
                } catch {
                    // Cancellation. The only way out of this loop, and not a
                    // failure worth reporting.
                    return
                }
                guard !Task.isCancelled else { return }
                await self.client.readSession()
            }
        }
    }

    func stopWatchingSession() {
        livenessWatch?.cancel()
        livenessWatch = nil
    }

    deinit { livenessWatch?.cancel() }

    // MARK: The pictures

    /// A loader for one row's picture. See `ObjectMemoryPictureLoader`.
    ///
    /// Made here rather than by the row so that the row never holds a client,
    /// and so the bytes' lifetime is the loader's and the loader's is the
    /// row's.
    func pictureLoader(for observationID: String) -> ObjectMemoryPictureLoader {
        ObjectMemoryPictureLoader(client: client, observationID: observationID)
    }

    /// Whether this Tower offered imagery routes at all on the last answer.
    /// `false` is not a failure — an older Tower on the same observations
    /// contract simply serves no pictures.
    var offersPictures: Bool { state.answer?.envelope.imagery != nil }
}

// MARK: - One row's picture

/// Fetches and holds the imagery for a single record, for as long as its row
/// is on screen and not one moment longer.
///
/// ## Why the bytes live here and nowhere else
///
/// Both binary routes send `Cache-Control: no-store`. A copy of a wearer's
/// first-person frame held in a URL cache, a disk cache, or an app-level image
/// cache is **a second store nobody chose and nobody's retention governs** —
/// Object Memory's retention would not reach it, and neither would
/// capture-side's. So:
///
/// - the transport uses an ephemeral `URLSession` with `urlCache = nil`
///   (`ObjectMemoryImageryHTTPClient.uncachedSession`);
/// - the decoded bytes are held on this object, which is a row's
///   `@StateObject` and dies with the row;
/// - `forget()` drops them on disappear rather than waiting for deallocation;
/// - **there is no shared cache keyed by observation id**, and adding one would
///   undo all of the above in one line.
///
/// ## Why the description is fetched before the bytes
///
/// `/imagery` answers 200 in almost every case and says whether there is a
/// picture, how much of the record's own box the filter covered, and — when
/// there is not — *why*. Fetching it first is what lets a 410 arrive as "the
/// memory is kept and the picture is gone" rather than as an image request
/// that failed, and it is what decides between `/crop` and `/frame` without
/// downloading one to find out the other was wanted.
@MainActor
final class ObjectMemoryPictureLoader: ObservableObject {

    /// What is on screen for this row.
    enum Phase: Equatable {
        /// Nothing has been asked. Rows do not fetch until they appear.
        case unasked
        case loading
        /// JPEG bytes, and the description that licenses the caption over them.
        case picture(Data, ObjectMemoryImageryDescription)
        /// The Tower described the imagery and there is none. **This is a
        /// rendered sentence, never a broken image and never an empty row.**
        case noPicture(ObjectMemoryImageryDescription)
        /// This Tower serves no pictures at all — no imagery block on the
        /// envelope.
        case noPicturesOffered
        case failed(CartridgeFailure)
    }

    @Published private(set) var phase: Phase = .unasked

    /// Which route the bytes came from, so the caption can say whether this is
    /// the whole frame or the object.
    @Published private(set) var kind: ObjectMemoryImageryKind = .crop

    private let client: any ObjectMemoryClient
    private let observationID: String
    private var work: Task<Void, Never>?

    init(client: any ObjectMemoryClient, observationID: String) {
        self.client = client
        self.observationID = observationID
    }

    /// Asks for the picture. Idempotent while one is in flight, and a no-op
    /// once something has been shown — a row scrolling back into view must not
    /// re-fetch a frame it is already holding.
    func load() {
        switch phase {
        case .unasked, .failed: break
        case .loading, .picture, .noPicture, .noPicturesOffered: return
        }
        fetch(kind: nil)
    }

    /// Fetches the other route on request — the whole frame when a crop is
    /// mostly fill, or the crop when the frame is too wide to read.
    func show(_ kind: ObjectMemoryImageryKind) {
        guard kind != .view else { return }
        fetch(kind: kind)
    }

    /// Drops the bytes.
    ///
    /// Called on disappear rather than left to deallocation, because a
    /// `@StateObject` outlives a scroll and "the row is off screen" is exactly
    /// when this app should stop holding a photograph of somebody's home.
    func forget() {
        work?.cancel()
        work = nil
        phase = .unasked
    }

    deinit { work?.cancel() }

    private func fetch(kind requested: ObjectMemoryImageryKind?) {
        work?.cancel()
        phase = .loading
        work = Task { [weak self] in
            guard let self else { return }

            // The description first, always. It is what turns a 410 into a
            // sentence instead of a failed image load.
            let described = await self.client.imageryDescription(for: self.observationID)
            guard !Task.isCancelled else { return }

            let description: ObjectMemoryImageryDescription
            switch described {
            case .described(let value):
                description = value
            case .routesUnknown:
                self.phase = .noPicturesOffered
                return
            case .unreachable(let reason):
                self.phase = .failed(
                    CartridgeFailure(
                        kind: .transport, message: "The Tower did not answer: \(reason)"
                    )
                )
                return
            case .undecodable:
                self.phase = .failed(
                    CartridgeFailure(
                        kind: .undecodableResponse,
                        message: ObjectMemoryCopy.unreadableImageryAnswer
                    )
                )
                return
            }

            guard description.available else {
                // 410, 404 and the 503 family all land here, and all three are
                // sentences `ObjectMemoryCopy` can write. None of them is an
                // error state and none of them is an empty row.
                self.phase = .noPicture(description)
                return
            }

            // `/frame` rather than `/crop` when a fill is sitting on the thing
            // the record is about — the contract's own fallback, taken
            // automatically, with the sentence shown as well rather than
            // instead.
            let kind = requested ?? description.preferredKind
            self.kind = kind

            let answer = await self.client.picture(for: self.observationID, kind: kind)
            guard !Task.isCancelled else { return }

            switch answer {
            case .picture(let data):
                self.phase = .picture(data, description)
            case .refused(let refusal):
                // Reachable even though the description said available:
                // capture-side retention can close between the two requests.
                // The refusal body is a full description, so the race ends in
                // the same rendered sentence rather than a broken image.
                self.phase = .noPicture(refusal)
            case .routesUnknown:
                self.phase = .noPicturesOffered
            case .unreachable(let reason):
                self.phase = .failed(
                    CartridgeFailure(
                        kind: .transport, message: "The Tower did not answer: \(reason)"
                    )
                )
            case .undecodable:
                self.phase = .failed(
                    CartridgeFailure(
                        kind: .undecodableResponse,
                        message: ObjectMemoryCopy.unreadableImageryAnswer
                    )
                )
            }
        }
    }
}
