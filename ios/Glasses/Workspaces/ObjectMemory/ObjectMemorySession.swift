//
//  ObjectMemorySession.swift
//  Glasses
//

import Foundation

/// Start, Pause, Resume and Stop for a cartridge, over HTTP.
///
/// ## The one rule to read before changing anything in this file
///
/// **`state` is intent. `following` is fact.** The payload says so itself, in
/// `state_means: "intent-not-liveness"`, and this is not a stylistic note —
/// the two come apart exactly when it matters most.
///
/// A Pause whose producer ignores `SIGTERM` answers **200** with
/// `state: "paused"` and `changed: true` — a positive claim that the action
/// took effect — while `following` still names the capture and the process is
/// still alive and still recording. That was reproduced during the
/// 2026-08-27 integration; it is not a hypothetical. **A Pause button keyed on
/// `state` tells a person they stopped being recorded when they did not**, and
/// there is no worse sentence this app can put in front of a wearer.
///
/// So every liveness claim in this file and in `ObjectMemoryCopy` reads
/// `following`, and `state` is only ever rendered as *what was asked for*. The
/// two are named apart in the API — `isFollowingACapture` against `state` —
/// so that a future caller reaching for the wrong one has to type the word
/// "state" to get it.
///
/// ## Why this type is generic and not `ObjectMemorySession`
///
/// The Tower's surface is keyed by cartridge id and knows no cartridge: the
/// next producer that needs a button gets one for free. Today only
/// `object_memory` answers it and any other name is a 404, which is a
/// *configuration* answer rather than a refusal. Mirroring that shape here
/// costs nothing and means the second cartridge to grow a Start does not
/// arrive by copying this file.
///
/// ## Nothing is persisted, deliberately
///
/// A Tower that restarts comes back `stopped`. This app must therefore never
/// cache a session state across a launch, or re-assert one it remembers:
/// resuming a memory of what a camera sees without anybody asking again is the
/// wrong direction to fail in. Every state here comes from a live read.

// MARK: - The contract this build implements

/// The generic session agreement, and the two values that travel with every
/// payload under it.
///
/// The identifier is **opaque**. Compared for equality, never parsed, never
/// ordered — a different date is a different agreement, neither newer nor
/// older, exactly as `ObjectMemoryContract.identifier` is treated.
nonisolated enum CartridgeSessionContract {
    static let identifier = "cartridge_session.control/2026-08-27"

    /// `state_means`. Checked rather than assumed: a Tower that changed what
    /// `state` claims would be changing the single fact this whole file is
    /// built around, and rendering the new meaning under the old wording is
    /// the most breaking failure available here.
    static let intentNotLiveness = "intent-not-liveness"

    /// The cartridge id this app asks about. Object Memory is **absent from**
    /// `GET /cartridges` deliberately, so the id is a constant here rather
    /// than something learned from a declaration that will never mention it.
    static let objectMemoryCartridge = "object_memory"
}

// MARK: - Vocabulary

/// A session state, as the Tower names it.
///
/// A `RawRepresentable` struct rather than an `enum` so that a state this
/// build has never heard of arrives **as itself** instead of failing a decode
/// or, far worse, defaulting to one of the three we know. A Tower that grows a
/// fourth state is describing something real; showing it uninterpreted is
/// honest, and showing it as `stopped` would be a claim.
nonisolated struct CartridgeSessionState: RawRepresentable, Equatable, Sendable, Hashable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    /// No producer, and no session id. Where a restarted Tower comes back.
    static let stopped = CartridgeSessionState(rawValue: "stopped")
    /// A person asked this cartridge to run. **Not** a claim that anything is.
    static let active = CartridgeSessionState(rawValue: "active")
    /// The producer was detached; the session id survives.
    static let paused = CartridgeSessionState(rawValue: "paused")

    /// The three this build can word a sentence about.
    static let known: [CartridgeSessionState] = [.stopped, .active, .paused]

    var isRecognised: Bool { Self.known.contains(self) }
}

/// A verb this app can send.
///
/// An `enum` rather than a struct, unlike the state above, because the
/// asymmetry is real: a state is something the Tower *tells* this app, and an
/// unknown one must still be displayable. An action is something this app
/// *sends*, and there is nothing useful it can do with a verb it does not
/// implement — the vocabulary is read from `actions` and anything outside
/// these four is reported rather than offered as a button.
nonisolated enum CartridgeSessionAction: String, Equatable, Sendable, CaseIterable {
    /// **Idempotent**, and works from `stopped` *and* `paused`. It means "be
    /// running, whatever the app thought". A second start is 200 with
    /// `changed: false` — honoured, nothing moved. **That is not an error and
    /// must never be shown as one.**
    case start
    /// Detaches the producer. The process stops, which is observable in the
    /// process table and cannot go stale.
    case pause
    /// **Stricter than start**, because it claims to be continuing something.
    /// From `stopped` it is refused 409.
    case resume
    /// **Never refused, from any state.** Refusing it would leave a Tower
    /// restart as the only way out of a bad state.
    case stop
}

/// Why an action could not be honoured from the state it was sent in.
///
/// A `RawRepresentable` struct for the same reason the state is one, and for a
/// second reason specific to this value: **the contract document and the wire
/// disagree about it today.**
///
/// `TOWER-UNIFIED-CARTRIDGES.md` §4.1 records `resume` from `stopped` as
/// `not-active`. The running Tower answers `not-paused`. Both words are in
/// §10's vocabulary and both are truthful descriptions of the same situation,
/// so neither is a bug — but a client that switched on one of them and wrote
/// the sentence for the other would be wrong half the time, and on a different
/// Tower build it would be wrong the other half.
///
/// **So this app does not word a refusal from the reason.** `ObjectMemoryCopy`
/// words it from the action that was sent and the state actually reached, both
/// of which the refusal body carries and neither of which is in dispute. The
/// reason is carried, shown as provenance and compared in tests; it is not
/// what decides what a person reads.
nonisolated struct CartridgeSessionRefusalReason: RawRepresentable, Equatable, Sendable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    /// This Tower has no producer to start at all.
    static let unsupported = CartridgeSessionRefusalReason(rawValue: "unsupported")
    /// There is nothing running to act on. What `pause` from `stopped` gets.
    static let notActive = CartridgeSessionRefusalReason(rawValue: "not-active")
    /// There is nothing paused to continue. What `resume` from `stopped` gets
    /// on the wire, where the document says `not-active`.
    static let notPaused = CartridgeSessionRefusalReason(rawValue: "not-paused")
    /// The verb is not one of the four.
    static let unknownAction = CartridgeSessionRefusalReason(rawValue: "unknown-action")

    /// The two the wire and the document disagree between. Grouped so a test
    /// can assert that both produce the same sentence rather than asserting
    /// each separately and letting the second rot.
    static let interchangeableStateRefusals: [CartridgeSessionRefusalReason] = [
        .notActive, .notPaused,
    ]
}

// MARK: - The snapshot

/// One reading of a cartridge's session: what was asked for, and what is
/// actually being followed.
///
/// `accepted`, `changed` and `attachedCaptureID` are `nil` on a `GET` and
/// populated on a `POST`. They are Optionals rather than defaulted booleans
/// because "this was a read" and "this was an action that changed nothing" are
/// different facts, and a `false` standing in for the first would make a
/// refresh look like a no-op double tap.
nonisolated struct CartridgeSessionSnapshot: Equatable, Sendable {
    let contract: String
    /// The cartridge this session belongs to, echoed by the Tower.
    let cartridge: String
    /// The producer name the Tower would spawn, e.g. `object-memory-session`.
    /// Provenance, never a capability claim.
    let worker: String?
    /// Whether this Tower has a producer to start **at all**. `false` on a
    /// Tower with the cartridge switched off — and a Start button that
    /// silently does nothing is worse than one that says why it cannot.
    let supported: Bool

    /// **Intent.** What a person asked for. See this file's header.
    let state: CartridgeSessionState
    /// `intent-not-liveness`, carried rather than assumed.
    let stateMeans: String
    /// The state vocabulary, so controls can be drawn without hard-coding it.
    let states: [CartridgeSessionState]
    /// The action vocabulary, likewise. Raw strings: a verb this build cannot
    /// send is still worth knowing the Tower offers.
    let actions: [String]

    /// Minted at Start, kept across Pause, cleared at Stop. **Not** a capture
    /// id, and never shown as one.
    let sessionID: String?
    let startedAt: Date?
    let changedAt: Date?

    /// **Fact.** Capture ids a producer is *alive* on, right now — every one
    /// of them, including a producer left over from a session that could not
    /// kill it.
    ///
    /// That breadth is deliberate and is why this is no longer the field a
    /// liveness claim is drawn from. An un-killable producer is what "the Stop
    /// button failed open" looks like, and it must stay visible somewhere;
    /// `intentContradictsLiveness` is drawn from here and stays drawn from
    /// here.
    let following: [String]

    /// **Fact, scoped to this session.** The subset of `following` that *this*
    /// session started. `nil` from a Tower that does not send the field.
    ///
    /// The Tower's own contract carried this defect as a warning before it was
    /// fixed: `following` is supervisor-scoped, so a session that started
    /// nothing reports a previous session's capture under a new `session_id`
    /// and renders as recording. Pressing Stop against a producer that will not
    /// die and then pressing Start is the whole reproduction.
    ///
    /// `nil` is not `[]`. A Tower that never sends the field has not said this
    /// session is following nothing; it has said nothing at all, and
    /// `isFollowingACapture` falls back to `following` there rather than
    /// silently reporting a Tower with a producer as one without.
    let followingThisSession: [String]?
    /// Every capture this session's producer has been seen following, in the
    /// order first seen. History, not liveness — a capture in here and not in
    /// `following` is one the producer has finished with.
    let captures: [String]

    /// `POST` only. The Tower honoured the action.
    let accepted: Bool?
    /// `POST` only. `false` means honoured and **nothing moved** — a double
    /// tap, not an error.
    let changed: Bool?
    /// `POST` only. The capture a producer was just started against, if any.
    /// `nil` is normal and legal: starting before the camera is running gives
    /// an `active` session with nothing attached, and the next capture to open
    /// finds the gate open.
    let attachedCaptureID: String?

    init(
        contract: String,
        cartridge: String,
        worker: String?,
        supported: Bool,
        state: CartridgeSessionState,
        stateMeans: String,
        states: [CartridgeSessionState],
        actions: [String],
        sessionID: String?,
        startedAt: Date?,
        changedAt: Date?,
        following: [String],
        followingThisSession: [String]? = nil,
        captures: [String],
        accepted: Bool? = nil,
        changed: Bool? = nil,
        attachedCaptureID: String? = nil
    ) {
        self.contract = contract
        self.cartridge = cartridge
        self.worker = worker
        self.supported = supported
        self.state = state
        self.stateMeans = stateMeans
        self.states = states
        self.actions = actions
        self.sessionID = sessionID
        self.startedAt = startedAt
        self.changedAt = changedAt
        self.following = following
        self.followingThisSession = followingThisSession
        self.captures = captures
        self.accepted = accepted
        self.changed = changed
        self.attachedCaptureID = attachedCaptureID
    }

    // MARK: Liveness

    /// **The liveness fact, and the only one.**
    ///
    /// Whether a producer *this session started* is alive on a capture right
    /// now. Deliberately not from `state`, deliberately not from `sessionID`,
    /// and deliberately not from `captures`, which is a history and stays
    /// populated after a producer has finished.
    ///
    /// Read from `followingThisSession` when the Tower sends it, and from
    /// `following` when it does not. The fallback is not caution about a field
    /// that might be missing — it is the only honest answer available from a
    /// Tower that cannot scope it, and it errs towards reporting a producer
    /// that exists rather than towards silence about one that does.
    var isFollowingACapture: Bool { !(followingThisSession ?? following).isEmpty }

    /// Recordings being written into that **the control on this screen did not
    /// start**.
    ///
    /// Empty on every ordinary reading. Non-empty means a producer is writing
    /// that a Stop from here does not reach — which is a sentence a wearer
    /// needs, and is not the same sentence as "you are remembering".
    ///
    /// ## Named for what it knows, which is narrower than session ownership
    ///
    /// It was `producersThisSessionDidNotStart`, and that name was a claim the
    /// field cannot support. The Tower scopes `following_this_session` by
    /// *started at or after this session last went active*, and **re-takes that
    /// mark on every Resume** — so a producer that survived a Pause is this
    /// session's, from before the pause, and is still correctly outside the
    /// scoped list. Calling it "this session did not start it" would be false
    /// in exactly the case a wearer is most likely to hit.
    ///
    /// Renamed to match `ObjectMemoryCopy.leftoverProducerLine`, which is
    /// worded from the same narrower fact, so the property name and the
    /// sentence cannot drift apart again.
    ///
    /// **`nil` is not `[]`.** The three-valued field is the whole point:
    /// a list scopes, `[]` is a positive claim that nothing this control
    /// started is running, and `nil` (or an omitted key) means the Tower cannot
    /// scope the question. The `guard` returns `[]` for `nil` because *nothing
    /// can be called unreached when nothing is scoped* — an unscoped Tower must
    /// draw no warning at all, rather than a warning about every capture it is
    /// following.
    var recordingsThisControlDidNotStart: [String] {
        guard let mine = followingThisSession else { return [] }
        return following.filter { !mine.contains($0) }
    }

    /// Whether the Tower's own two fields contradict each other in the
    /// direction that harms a person.
    ///
    /// `paused` or `stopped` while still following a capture means the action
    /// was reported as honoured and the recording did not stop. That is the
    /// reproduced `SIGTERM` failure, and it is the one state in this whole
    /// cartridge that must be shown loudly rather than reconciled.
    ///
    /// The other direction — `active` with nothing followed — is **not** here,
    /// because it is legal: starting before the camera is running looks
    /// exactly like that, and so does a producer that died. This app cannot
    /// tell those apart from one payload and must not guess; it says what it
    /// knows, which is that nothing is being followed.
    var intentContradictsLiveness: Bool {
        // From `following`, the WIDE field, and not from
        // `isFollowingACapture`, which is now scoped to this session. A
        // producer left over from an earlier session is still a producer
        // recording a person who asked to stop being recorded, and scoping
        // this alarm would be the one narrowing that makes it useless.
        guard !following.isEmpty else { return false }
        return state == .paused || state == .stopped
    }

    /// Whether the action that produced this snapshot was honoured and moved
    /// nothing. `nil` on a read, where the question does not arise.
    ///
    /// Exists so the one caller does not write `snapshot.changed == false`,
    /// which is `true` for a `GET` under Optional comparison and would report
    /// every refresh as a double tap.
    var wasAnIdempotentNoOp: Bool? {
        guard let changed else { return nil }
        return !changed
    }

    /// The four verbs the Tower offered that this build can actually send.
    /// Read off `actions` rather than assumed, which is what the field is for.
    var offeredActions: [CartridgeSessionAction] {
        actions.compactMap(CartridgeSessionAction.init(rawValue:))
    }

    /// Verbs the Tower offers that this build has no button for. Surfaced
    /// rather than dropped: a control surface that silently hides half its
    /// vocabulary looks complete and is not.
    var unofferedActions: [String] {
        actions.filter { CartridgeSessionAction(rawValue: $0) == nil }
    }
}

/// A 409: the action could not be honoured **from this state**.
///
/// Carries the state that was actually reached, so a client that refreshes
/// after a refusal does not need a second request to find out where it is.
nonisolated struct CartridgeSessionRefusal: Equatable, Sendable {
    /// The verb that was sent. What the copy is worded from — see
    /// `CartridgeSessionRefusalReason`.
    let action: CartridgeSessionAction
    let reason: CartridgeSessionRefusalReason
    /// The Tower's own sentence. Kept as provenance and shown behind a
    /// disclosure; never the sentence a wearer reads first, because it is
    /// written for an operator.
    let message: String
    /// Where the session actually is, not where the action asked it to go.
    let snapshot: CartridgeSessionSnapshot
}

/// What a `POST` produced.
nonisolated enum CartridgeSessionOutcome: Equatable, Sendable {
    /// **200.** Including the idempotent no-op — a second `start`, a second
    /// `pause`, a `stop` from `stopped`. Check `snapshot.changed`; do not
    /// check the status code twice.
    case honoured(CartridgeSessionSnapshot)
    /// **409.** Not honoured from the state it was sent in.
    case refused(CartridgeSessionRefusal)

    /// The current session either way. A refusal is still a reading.
    var snapshot: CartridgeSessionSnapshot {
        switch self {
        case .honoured(let snapshot): return snapshot
        case .refused(let refusal): return refusal.snapshot
        }
    }
}

// MARK: - Transport failures

/// Why a session request did not produce a reading.
///
/// The first case is **not** an error about the session: it is a fact about
/// the Tower's configuration, exactly as the observations router's 404 is.
nonisolated enum CartridgeSessionFetchError: Error, Equatable {
    /// **404.** This Tower has no controllable session for that cartridge.
    /// A configuration answer, and never the answer to "may I start" — which
    /// is a 409 and arrives as `.refused`.
    case noSuchCartridgeSession
    /// The Tower's session surface speaks a different agreement.
    case unsupportedContract(identifier: String)
    /// The answer arrived and could not be read as this contract.
    case undecodable
    /// The request did not complete.
    case transport(String)
}

// MARK: - Decoding

/// Turns a session payload into a snapshot, or refuses.
///
/// Written against the running Tower's bytes rather than against §9.1's field
/// table alone — which is how `cartridge` and `worker`, present on the wire and
/// absent from the table, are decoded here, and how the `not-paused` /
/// `not-active` disagreement was found rather than inherited.
nonisolated enum CartridgeSessionDecoder {

    /// `nil` when the payload does not name a contract at all.
    static func contractIdentifier(from json: [String: Any]) -> String? {
        json["contract"] as? String
    }

    static func snapshot(from json: [String: Any]) -> CartridgeSessionSnapshot? {
        guard
            let contract = json["contract"] as? String,
            contract == CartridgeSessionContract.identifier,
            let cartridge = json["cartridge"] as? String,
            // A real Bool. `bool` subclasses `int` in Python and a `1` here
            // fails every `as? Bool`; the Tower pins these as genuine booleans
            // and this decode is what would notice if that stopped being true.
            let supported = json["supported"] as? Bool,
            let stateWord = json["state"] as? String,
            let stateMeans = json["state_means"] as? String,
            // Checked, not merely carried. This value is the Tower saying that
            // `state` is not liveness, and a payload that stopped saying it
            // means something this build's whole rendering assumes.
            stateMeans == CartridgeSessionContract.intentNotLiveness,
            let states = json["states"] as? [String],
            let actions = json["actions"] as? [String],
            let following = json["following"] as? [String],
            let captures = json["captures"] as? [String]
        else { return nil }

        return CartridgeSessionSnapshot(
            contract: contract,
            cartridge: cartridge,
            worker: json["worker"] as? String,
            supported: supported,
            state: CartridgeSessionState(rawValue: stateWord),
            stateMeans: stateMeans,
            states: states.map(CartridgeSessionState.init(rawValue:)),
            actions: actions,
            // Null at rest and after a Stop. Optional rather than "" — the two
            // mean different things and only one of them is what was sent.
            sessionID: json["session_id"] as? String,
            startedAt: date(json["started_at"]),
            changedAt: date(json["changed_at"]),
            following: following,
            // Absent from a Tower older than 2026-08-29. Left `nil` rather
            // than defaulted to `[]`, so `isFollowingACapture` can tell
            // "this session is following nothing" from "this Tower does not
            // answer that question".
            followingThisSession: json["following_this_session"] as? [String],
            captures: captures,
            accepted: json["accepted"] as? Bool,
            changed: json["changed"] as? Bool,
            attachedCaptureID: json["attached_capture_id"] as? String
        )
    }

    /// A 409 body. The refusal fields sit **beside** a full snapshot rather
    /// than instead of one, so the same decode serves both halves.
    static func refusal(
        from json: [String: Any], action: CartridgeSessionAction
    ) -> CartridgeSessionRefusal? {
        guard
            let snapshot = self.snapshot(from: json),
            let reason = json["reason"] as? String,
            let message = json["message"] as? String
        else { return nil }
        return CartridgeSessionRefusal(
            action: action,
            reason: CartridgeSessionRefusalReason(rawValue: reason),
            message: message,
            snapshot: snapshot
        )
    }

    /// Tower-receipt epoch seconds, or `nil`.
    ///
    /// Never defaulted to `Date()` or to the epoch: a session that has not
    /// started has no start time, and inventing one would put a duration on
    /// screen for a session that has no duration.
    private static func date(_ value: Any?) -> Date? {
        guard let seconds = value as? Double else { return nil }
        return Date(timeIntervalSince1970: seconds)
    }
}

// MARK: - HTTP

/// `GET` the session, and `POST` the four verbs.
///
/// ## Why this is separate from `ObjectMemoryHTTPClient`
///
/// Because it mutates, and the Tower keeps the two apart for exactly that
/// reason: the observations router is asserted by a test to expose only `GET`,
/// which is what lets "this cartridge cannot delete your memory" stay an
/// absolute claim rather than "absolute except for the bits that are not". A
/// single Swift type holding both a read-only query and a `POST` would make
/// that boundary a comment. Two types make it a type.
///
/// This surface still cannot touch a store. It starts and stops a *producer*.
nonisolated struct CartridgeSessionHTTPClient {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession = .shared
    /// A `POST /pause` blocks on the Tower for up to its detach grace while a
    /// producer finishes the record it is writing. Ten seconds is the same
    /// bound the query client uses and comfortably clears that grace; a longer
    /// one would turn a dead Tower into a spinner nobody can end.
    var timeout: TimeInterval = 10
    /// The cartridge whose session this client controls.
    var cartridge: String = CartridgeSessionContract.objectMemoryCartridge

    func read() async throws -> CartridgeSessionSnapshot {
        let json: [String: Any]
        do {
            json = try await send("GET", to: sessionURL())
        } catch is Refused {
            // A 409 has no meaning on a read: nothing was asked for, so
            // nothing can have been refused. Caught explicitly rather than
            // left to propagate, because an escaping `Refused` is not a
            // `CartridgeSessionFetchError` and would be wrapped as
            // `.transport` — reporting a Tower that answered as one that
            // could not be reached, which is the exact defect this cartridge
            // spent this change removing from the imagery path.
            throw CartridgeSessionFetchError.undecodable
        }
        try requireThisContract(json)
        guard let snapshot = CartridgeSessionDecoder.snapshot(from: json) else {
            throw CartridgeSessionFetchError.undecodable
        }
        return snapshot
    }

    /// Sends one verb. A **409 is not thrown** — it is an outcome, because it
    /// is a true and useful thing to tell a person ("resume continues a paused
    /// session; this one is stopped") and throwing it would push it through
    /// the same channel as an unreachable Tower.
    func apply(_ action: CartridgeSessionAction) async throws -> CartridgeSessionOutcome {
        let url = sessionURL().appendingPathComponent(action.rawValue)
        do {
            let json = try await send("POST", to: url)
            try requireThisContract(json)
            guard let snapshot = CartridgeSessionDecoder.snapshot(from: json) else {
                throw CartridgeSessionFetchError.undecodable
            }
            return .honoured(snapshot)
        } catch let refusal as Refused {
            // Unwrapped from FastAPI's `detail` envelope before it gets here.
            try requireThisContract(refusal.body)
            guard
                let decoded = CartridgeSessionDecoder.refusal(
                    from: refusal.body, action: action
                )
            else { throw CartridgeSessionFetchError.undecodable }
            return .refused(decoded)
        }
    }

    private func sessionURL() -> URL {
        baseURL
            .appendingPathComponent("cartridges")
            // Its own component so a cartridge id with anything unusual in it
            // is percent-encoded rather than splitting the path — the same
            // discipline `lastSeen` applies to `cell phone`.
            .appendingPathComponent(cartridge)
            .appendingPathComponent("session")
    }

    private func requireThisContract(_ json: [String: Any]) throws {
        guard let identifier = CartridgeSessionDecoder.contractIdentifier(from: json) else {
            throw CartridgeSessionFetchError.undecodable
        }
        guard identifier == CartridgeSessionContract.identifier else {
            throw CartridgeSessionFetchError.unsupportedContract(identifier: identifier)
        }
    }

    /// A 409 on its way out of `send`, carrying the body a refusal decodes
    /// from. Private, and never surfaced: `apply` converts it into an outcome
    /// before anybody else sees it.
    private struct Refused: Error { let body: [String: Any] }

    private func send(_ method: String, to url: URL) async throws -> [String: Any] {
        var request = URLRequest(
            url: url,
            // A session state answered out of a URL cache is a claim about a
            // running producer made from a stale copy, which is the one kind
            // of staleness a Start/Stop control cannot tolerate.
            cachePolicy: .reloadIgnoringLocalAndRemoteCacheData,
            timeoutInterval: timeout
        )
        request.httpMethod = method

        do {
            let (data, response) = try await session.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0

            if status == 404 {
                // "No such cartridge session on this Tower." A configuration
                // answer — and note it is *not* what a refused action gets,
                // which is 409 and reaches the caller as a readable outcome.
                throw CartridgeSessionFetchError.noSuchCartridgeSession
            }

            // Parsed in its own `do`, because `jsonObject(with:)` throws on a
            // malformed body. Left to the catch below it would be relabelled
            // `.transport` — a claim about the network that is false when the
            // Tower answered.
            let parsed: Any
            do {
                parsed = try JSONSerialization.jsonObject(with: data)
            } catch {
                throw CartridgeSessionFetchError.undecodable
            }
            guard let json = parsed as? [String: Any] else {
                throw CartridgeSessionFetchError.undecodable
            }

            if status == 409 {
                // FastAPI wraps an `HTTPException` detail in `{"detail": …}`,
                // and the session router raises one for every refusal. The
                // body inside is a complete snapshot plus `reason` and
                // `message`; the wrapper is transport packaging and is peeled
                // here rather than being carried into the decoder.
                guard let detail = json["detail"] as? [String: Any] else {
                    throw CartridgeSessionFetchError.undecodable
                }
                throw Refused(body: detail)
            }
            guard (200...299).contains(status) else {
                // Deliberately **not** `.transport`. The Tower was reached and
                // it answered; reporting that as a connection failure would
                // tell a person to check their network about a machine that
                // replied. `.undecodable` is the honest reading of a status
                // this surface does not define.
                throw CartridgeSessionFetchError.undecodable
            }
            return json
        } catch let error as CartridgeSessionFetchError {
            throw error
        } catch let refusal as Refused {
            throw refusal
        } catch {
            throw CartridgeSessionFetchError.transport(error.localizedDescription)
        }
    }
}

// MARK: - State

/// What the session half of the Object Memory workspace should be showing.
nonisolated enum ObjectMemorySessionState: Equatable, Sendable {
    /// Nothing has been read. **Not** "stopped" — a Tower that has not been
    /// asked has not said it is stopped, and drawing a Stopped badge from
    /// silence is the same error as drawing "recording" from `state`.
    case unread
    /// A read or an action is in flight.
    case working
    /// A live reading.
    case known(CartridgeSessionSnapshot)
    /// The last action was refused, and this is where the session actually is.
    /// Still a reading: controls stay drawn from `refusal.snapshot`.
    case refused(CartridgeSessionRefusal)
    /// **404.** This Tower has no controllable session for this cartridge.
    case noSessionControl
    case failed(CartridgeFailure)

    /// The current reading, wherever it came from. `nil` only when there has
    /// never been one.
    var snapshot: CartridgeSessionSnapshot? {
        switch self {
        case .known(let snapshot): return snapshot
        case .refused(let refusal): return refusal.snapshot
        case .unread, .working, .noSessionControl, .failed: return nil
        }
    }

    /// **The liveness question, answered from `following` and nothing else.**
    ///
    /// `nil` — not `false` — when there is no reading. "We have not asked" and
    /// "nothing is being followed" are different, and only one of them is
    /// something to tell a person about what is being recorded.
    var isFollowingACapture: Bool? { snapshot?.isFollowingACapture }

    /// Whether a Start button can do anything at all on this Tower.
    var isSupported: Bool { snapshot?.supported ?? false }
}
