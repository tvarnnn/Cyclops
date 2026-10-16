//
//  SceneUnderstandingClient.swift
//  Glasses
//

import Combine
import Foundation

/// Supplies `SceneUnderstandingState` to the Scene Understanding workspace.
///
/// The fourth interaction shape: a state that is **replaced wholesale** and
/// never accumulates. World Builder pushes one accumulating artefact,
/// Experimental CV Lab runs a bounded job, Document Memory answers point
/// queries and keeps what it recorded, and this one describes a moment and
/// keeps nothing. All four sit behind `CartridgeClient` for the one question
/// they share and behind their own protocol for everything else — which is what
/// keeps the shared layer at a handful of small files instead of a framework.
@MainActor
protocol SceneUnderstandingClient: CartridgeClient {
    var state: SceneUnderstandingState { get }

    /// Every reading after the one `state` held when the view model was built.
    ///
    /// **This is the cartridge whose real client emits fastest**, so whatever
    /// conforms to it should coalesce before it publishes. The Tower already
    /// does most of that work: the payload is published at the standard 0.5 s
    /// poll with a 2 s heartbeat, **not at frame rate**, and `observed_at`,
    /// `staleness_seconds` and every `frames_*` counter are excluded from
    /// `revision` so an unchanged scene coalesces on the wire. What arrives
    /// here is therefore already at a rate a `@Published` property can carry.
    var stateUpdates: AnyPublisher<SceneUnderstandingState, Never> { get }

    /// The Scene screen appeared or disappeared.
    ///
    /// A REQUIREMENT, not only an extension default, and the distinction is
    /// the whole feature. The view model holds its client as
    /// `any SceneUnderstandingClient`. A method that lives only in a
    /// protocol extension is dispatched STATICALLY through an existential,
    /// so `client.workspaceVisibilityChanged(isVisible:)` would always run
    /// the no-op default below and never reach
    /// `TowerSceneUnderstandingClient`'s override -- which compiles, and
    /// silently means the live subscription is never opened or closed by
    /// the screen appearing or disappearing. Since the Tower runs a scene
    /// session only while somebody streams AND somebody watches, that
    /// leaves the detector running for as long as the socket lives, which
    /// is exactly what "the phone watches only while the Scene screen is
    /// open" exists to prevent.
    func workspaceVisibilityChanged(isVisible: Bool)
}

extension SceneUnderstandingClient {
    var stateUpdates: AnyPublisher<SceneUnderstandingState, Never> {
        Empty(completeImmediately: false).eraseToAnyPublisher()
    }

    /// The Scene screen appeared or disappeared. A default no-op, so a client
    /// with nothing to subscribe to (the stub) needs no code for it.
    func workspaceVisibilityChanged(isVisible: Bool) {}
}

// MARK: - The Tower-backed client

/// The real Scene Understanding client.
///
/// ## Where it sits
///
/// ```
/// TowerClient                   owns the socket, decodes the envelope, knows no cartridge
///     ↓ cartridgeResults
/// TowerSceneUnderstandingClient owns the subscription and this contract
///     ↓ stateUpdates
/// SceneUnderstandingViewModel   republishes into SwiftUI
///     ↓
/// SceneUnderstandingWorkspaceView renders counts, and the disclosure they are owed
/// ```
///
/// ## What it sends: nothing but a subscription
///
/// No `stream_start`, no `POST /scene/start`, no control message of any kind.
/// `docs/agent-handoffs/IOS-TO-TOWER.md` §6 (item 2) is explicit that opening a cartridge on the phone
/// sends nothing and a Tower-side test asserts the wire stays silent. The
/// session runs while somebody streams and somebody watches: the camera
/// stream the app already opens and closes for its own reasons, and the
/// subscription this client holds while the Scene screen is open. See
/// `SceneUnderstandingContract.phoneSendsNothingNote` for the two measured
/// reasons a Start button here would be worse than useless.
///
/// ## What it refuses to keep: everything, the moment a session ends
///
/// This is the one place this app differs from `TowerWorldBuilderClient` on
/// purpose. That client deliberately keeps its last world across a socket drop,
/// because a partly-built world is still true when the connection returns and
/// availability already renders the disconnection.
///
/// **A scene is not.** A scene held past the end of a session is a claim about
/// a room the wearer has left, and no staleness number makes it safe, because a
/// client that draws counts above a staleness line shows the room first. So
/// this client discards on all three of:
///
/// - `lifecycle.state == "stopped"` — the Tower has already discarded its own
///   copy (`scene_available: false`, `counts`/`where`/`people` null), and
///   `SceneUnderstandingState.forReading` maps it to `.idle`;
/// - the socket leaving `.online` — a disconnect *ends the session* here, since
///   the session follows the stream. World Builder's "keep it, availability
///   explains" reasoning does not transfer;
/// - `lifecycle.session_id` changing — two payloads from two tracking sessions
///   must not be compared, and holding one across the boundary is the mildest
///   form of comparing them.
///
/// Pause is the deliberately different case and is the one thing that *is*
/// kept: `.lastKnown`, with its age, visually apart from observing.
@MainActor
final class TowerSceneUnderstandingClient: SceneUnderstandingClient {

    let cartridgeID = "scene-understanding"

    private(set) var state: SceneUnderstandingState = .idle(nil) {
        didSet {
            guard state != oldValue else { return }
            stateSubject.send(state)
        }
    }

    var stateUpdates: AnyPublisher<SceneUnderstandingState, Never> {
        stateSubject.eraseToAnyPublisher()
    }

    private let stateSubject = PassthroughSubject<SceneUnderstandingState, Never>()
    private let tower: TowerClient
    private var cancellables: Set<AnyCancellable> = []

    /// This cartridge's subscription on the result channel.
    private var subscription = CartridgeResultSubscription()
    /// The tracking session the held reading belongs to. See the type note.
    private var heldSessionID: Int?
    /// Whether the Scene Understanding screen is on screen right now.
    ///
    /// **This is the phone's demand for the detector on the Tower.** Without
    /// an operator hold, a Tower runs a scene session only while somebody is
    /// STREAMING and somebody is WATCHING, and a `result_subscribe` for the
    /// live scene is what "watching" means. (An operator's `POST /scene/start`
    /// is a separate hold: it runs the session with or without a stream or a
    /// watcher, and keeps it when this screen leaves. That session ends when
    /// the operator stops it or the last open stream closes.) Subscribing at connection time would keep a people
    /// detector running behind World Builder's and the CV Lab's camera for as
    /// long as the app was connected. So the subscription is opened when this
    /// screen appears and closed when it disappears; the socket, the
    /// declaration and the resubscribe budget all still gate it, but none of
    /// them can open it while this is `false`.
    private var workspaceVisible = false

    init(tower: TowerClient) {
        self.tower = tower

        // `.receive(on:)` on both, and it is load-bearing rather than
        // stylistic: a `@Published` publisher fires from `willSet`, so a sink
        // that reads the property it was notified about sees the value *before*
        // the change. `TowerWorldBuilderClient` documents the same trap.
        tower.$status
            .removeDuplicates()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] status in self?.connectionChanged(to: status) }
            .store(in: &cancellables)

        tower.$cartridgeDeclaration
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.subscribeIfPossible() }
            .store(in: &cancellables)

        tower.cartridgeResults
            .sink { [weak self] event in self?.handle(event) }
            .store(in: &cancellables)
    }

    // MARK: Availability

    /// Resolved against the Tower's **live** declaration.
    ///
    /// It goes through `CartridgeAvailability.resolve` — the shared precedence —
    /// with this cartridge's own socket offer and identifier rather than
    /// through `TowerCapabilities`, because those tables are owned by the
    /// integration layer and this client must not reach across a boundary into
    /// them. They map `scene-understanding` to the same
    /// `SceneUnderstandingContract.towerCartridge` and list the same
    /// identifier, so both read the same declaration and the same constants.
    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        CartridgeAvailability.resolve(
            declared: declaredContract,
            supported: [SceneUnderstandingContract.identifier],
            isTowerReachable: isTowerReachable,
            // This build implements the contract, so a silent socket means
            // "nobody has asked yet", not "no such contract": the exact case
            // `knownToThisBuild` exists for, which a cold launch reaches.
            knownToThisBuild: true
        )
    }

    private var declaredContract: CartridgeContract? {
        guard
            let offer = tower.cartridgeDeclaration?
                .offer(forTowerCartridge: SceneUnderstandingContract.towerCartridge)
        else { return nil }
        return CartridgeContract(cartridgeID: cartridgeID, identifier: offer.contract)
    }

    // MARK: Connection lifecycle

    private func connectionChanged(to status: TowerStatus) {
        guard status == .online else {
            subscription.connectionLost()
            // The discard that World Builder does not do. A dropped socket ends
            // the session that produced this scene — `stream_stop` or a
            // disconnect is how a wearable's session normally ends — so the
            // counts on screen stop describing anything. `.idle(nil)` rather
            // than `.idle(reading)`: keeping the reading would keep its
            // `scene_unavailable_reason`, which would say "stopped" about a
            // session the Tower may still consider running.
            heldSessionID = nil
            state = .idle(nil)
            return
        }
        subscribeIfPossible()
    }

    /// Idempotent by construction: every path into it is guarded by the same
    /// two flags, so a status change and a republished declaration racing each
    /// other cannot open two subscriptions. Leaving the screen does not clear
    /// the in-flight flag either; see `workspaceVisibilityChanged`.
    private func subscribeIfPossible() {
        guard workspaceVisible else { return }
        guard tower.status == .online, subscription.id == nil, !subscription.isSubscribing else { return }
        guard let declaration = tower.cartridgeDeclaration else { return }
        guard
            let offer = declaration.offer(
                forTowerCartridge: SceneUnderstandingContract.towerCartridge
            )
        else {
            // The Tower said nothing about Scene Understanding. Availability
            // renders that as `.noContract`; there is nothing to subscribe to
            // and nothing for the domain state to add.
            return
        }
        guard offer.contract == SceneUnderstandingContract.identifier else {
            // A contract this build does not implement. `.unsupportedContract`
            // availability already outranks any state, so the state is left
            // where it is rather than given a second, weaker wording of the
            // same fact.
            return
        }
        guard offer.resultType == SceneUnderstandingContract.resultType else {
            // The one cartridge whose result type is not `status`. If the
            // declaration ever disagrees with this build about that, sending
            // the subscribe anyway earns an `unknown_result_type` refusal and
            // an empty screen with no explanation; refusing here at least
            // leaves the availability wording intact.
            return
        }
        guard offer.available else {
            // Offered and unserveable: switched off
            // (`TOWER_SCENE_UNDERSTANDING=off`), or the session could not be
            // constructed. The Tower's own prose names the variable or what
            // failed, so it is shown verbatim rather than paraphrased.
            state = .unsupported(
                reason: offer.unavailableReason ?? Self.unexplainedUnavailable
            )
            return
        }

        subscription.isSubscribing = true
        // A new subscription is answered with a complete snapshot, so whatever
        // was held describes a socket that is gone.
        heldSessionID = nil
        state = .idle(nil)
        tower.subscribeToResults(
            cartridge: offer.cartridge,
            resultType: offer.resultType,
            contract: offer.contract
        )
    }

    static let unexplainedUnavailable = """
        This Tower cannot serve Scene Understanding, and did not say why.
        """

    // MARK: Visibility

    /// Opens the live subscription when the Scene screen appears and closes it
    /// when the screen goes. That subscription is the phone's demand for the
    /// Tower's people detector; without an operator hold, the detector runs
    /// only while some watcher is subscribed and a stream is open.
    ///
    /// How: a change of visibility only (a repeat is ignored). Appearing
    /// subscribes, through every gate in `subscribeIfPossible`. Leaving sends
    /// `result_unsubscribe` for the open subscription, if there is one, and
    /// discards the reading.
    ///
    /// Why leaving keeps an unanswered subscribe marked in flight
    /// (`subscription.isSubscribing` is left as it is): that subscribe will
    /// still be answered. If the screen is still gone when its ack lands,
    /// `adoptOrClose` closes it; if the screen came back, it is adopted. A
    /// return inside that window therefore sends no second subscribe, and so
    /// can never leave two subscriptions open for one screen, one of them
    /// orphaned for the life of the socket.
    func workspaceVisibilityChanged(isVisible: Bool) {
        guard isVisible != workspaceVisible else { return }
        workspaceVisible = isVisible
        if isVisible {
            subscribeIfPossible()
            return
        }
        // With no other watcher, the unsubscribe ends the Tower's session and
        // releases its detector. It is best-effort, and its
        // `result_unsubscribed` matches nothing, because `subscription.id` is
        // cleared here. The Tower treats a closed socket as sufficient cleanup
        // anyway; this spares it a session nobody is reading. The reading is
        // discarded for the same reason a stop discards it: nothing is
        // refreshing it any more.
        if let id = subscription.id {
            tower.unsubscribeFromResults(subscriptionID: id)
        }
        subscription.id = nil
        heldSessionID = nil
        state = .idle(nil)
    }

    // MARK: Result channel

    private func handle(_ event: CartridgeResultEvent) {
        switch event {
        case .declaration:
            // Handled through `$cartridgeDeclaration` instead, so the cached
            // value and the trigger cannot disagree.
            break

        case .subscribed(let ack):
            guard ack.cartridge == SceneUnderstandingContract.towerCartridge else { return }
            adoptOrClose(ack)

        case .unsubscribed(let id):
            guard id == subscription.id else { return }
            subscription.id = nil
            // The subscription is gone, so nothing is refreshing this scene.
            // Holding it would leave a live-looking count that stopped being
            // fed, which is the same failure as caching across a stop.
            heldSessionID = nil
            state = .idle(nil)

        case .result(let envelope):
            guard envelope.cartridge == SceneUnderstandingContract.towerCartridge else { return }
            apply(envelope)

        case .failed(let error):
            guard error.belongs(
                toTowerCartridge: SceneUnderstandingContract.towerCartridge,
                openSubscriptionID: subscription.id
            ) else { return }
            apply(error)
        }
    }

    /// Takes the Tower's answer to a `result_subscribe`: adopts the
    /// subscription, or closes it, so that at most one is ever open.
    ///
    /// How: the answer ends the subscribe in flight. The ack is adopted only
    /// while the screen is showing and no subscription is open; otherwise it
    /// is closed at once with `result_unsubscribe`.
    ///
    /// Why each close:
    /// - **The screen has gone.** Leaving can only unsubscribe an id it holds,
    ///   and before the ack there is none. Adopted, this ack would be a watcher
    ///   that belongs to no screen, and nothing would retract it before the
    ///   socket closed.
    /// - **A subscription is already open.** Two subscribes can be out at once
    ///   when the Tower closes a subscription (`channel_failed`,
    ///   `consumer_too_slow`) while another subscribe still awaits its ack: the
    ///   resubscribe goes out beside it. Writing this id over the open one
    ///   would orphan the open one.
    ///
    /// Either orphan would keep a people detector running, because the Tower
    /// runs one while somebody streams and somebody watches.
    private func adoptOrClose(_ ack: CartridgeSubscriptionAck) {
        subscription.isSubscribing = false
        guard workspaceVisible, subscription.id == nil else {
            tower.unsubscribeFromResults(subscriptionID: ack.subscriptionID)
            return
        }
        subscription.id = ack.subscriptionID
    }

    private func apply(_ envelope: CartridgeResultEnvelope) {
        guard envelope.isSnapshot else {
            // This build knows how to merge exactly nothing. A delta would not
            // fail loudly — a partial payload decodes cleanly into a scene with
            // fewer things in it, which is a *quieter room* rendered with no
            // error and nothing on screen to suggest anything was missed.
            heldSessionID = nil
            state = .failed(
                CartridgeFailure(
                    kind: .notSupported,
                    message: """
                        The Tower sent a Scene Understanding result marked as a partial \
                        update rather than a complete one. This build can only read \
                        complete readings, so it is showing nothing rather than a scene \
                        assembled from a piece it does not know how to merge.
                        """
                )
            )
            return
        }

        guard let reading = SceneUnderstandingDecoder.reading(from: envelope.payload) else {
            heldSessionID = nil
            state = .failed(
                CartridgeFailure(
                    kind: .undecodableResponse,
                    message: """
                        The Tower sent a Scene Understanding result this build could not \
                        read. It declared contract \(envelope.contract ?? "none"), which \
                        this app implements, so the two disagree about what that contract \
                        means. Nothing is shown rather than a room assembled from a \
                        payload that was not understood.
                        """
                )
            )
            return
        }

        // Two payloads with different `session_id` came from different tracking
        // sessions and must not be compared. Nothing here carries anything
        // across that boundary — the payload is a complete snapshot and
        // `forReading` builds the state from it alone — so what this tracks is
        // the boundary itself, for the disconnect and unsubscribe paths above
        // to clear against and for anything that later wants to know a session
        // changed rather than merely that the counts did.
        heldSessionID = reading.lifecycle.sessionID
        state = SceneUnderstandingState.forReading(reading)
    }

    private func apply(_ error: CartridgeResultError) {
        if error.closesSubscription {
            // The Tower's message says to subscribe again to resume, and nothing
            // else reacts to this error.
            if let exhausted = subscription.closedByTower() {
                state = .failed(exhausted)
            } else {
                subscribeIfPossible()
            }
            return
        }
        // Any error of ours ends an in-flight subscribe attempt; see
        // `CartridgeResultSubscription.isSubscribing`.
        subscription.isSubscribing = false
        heldSessionID = nil
        state = .failed(
            CartridgeFailure(
                kind: error.reason == "cartridge_unavailable" ? .notSupported : .towerReportedFailure,
                message: error.message
            )
        )
    }
}

// MARK: - The stub, for a Tower that declares nothing

/// The client for a Tower that has declared no Scene Understanding contract.
///
/// It produces **no sample scene**. A demo reading with two people in it would
/// be indistinguishable on screen from a real observation of whoever is
/// actually in the room, and it would be this app asserting that the glasses
/// had detected people. That is the single most consequential fake datum in
/// this codebase, and it is not shipped, not behind a flag, and not in
/// `#if DEBUG`.
///
/// ## What its reason may and may not say
///
/// Only what is observable from here: this Tower has declared nothing. It
/// makes no claim about what any Tower does or stores, because this app has no
/// channel through which it could know that. A sentence such as "the Tower
/// does not analyse scenes" or "nothing about anyone the glasses pass reaches
/// this app" would be a privacy assurance about bystanders, on the one screen
/// whose subject is bystanders, and it is false: the Tower has a detector, a
/// tracker and a typed live contract, and a count of the people in front of
/// the wearer does reach this app when the cartridge is enabled.
@MainActor
final class UnavailableSceneUnderstandingClient: SceneUnderstandingClient {
    static let reason = """
        This Tower has not declared a Scene Understanding contract, so there is \
        nothing to subscribe to and no scene to show. That is a statement about \
        what this Tower offers, not about what is in front of you.
        """

    let cartridgeID = "scene-understanding"

    let state: SceneUnderstandingState =
        .unsupported(reason: UnavailableSceneUnderstandingClient.reason)

    init() {}
}

// MARK: - View model

/// Publishes Scene Understanding state into SwiftUI.
///
/// Holds no runtime references, for the reason given on `WorldBuilderViewModel`.
/// That matters more here than anywhere else: a view model that held the camera
/// and was destroyed on a cartridge switch would end a live session, and this
/// is the cartridge whose subject matter most invites someone to wire it
/// straight to the frame stream.
@MainActor
final class SceneUnderstandingViewModel: ObservableObject {
    @Published private(set) var state: SceneUnderstandingState

    private let client: any SceneUnderstandingClient
    private var cancellables: Set<AnyCancellable> = []

    /// No default argument — see `WorldBuilderViewModel.init(client:)`.
    init(client: any SceneUnderstandingClient) {
        self.client = client
        self.state = client.state

        client.stateUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in self?.state = state }
            .store(in: &cancellables)
    }

    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        client.availability(isTowerReachable: isTowerReachable)
    }

    /// The screen appeared or disappeared. Forwarded to the client, which is
    /// what opens and closes the live subscription — and with it the Tower's
    /// session.
    func workspaceVisibilityChanged(isVisible: Bool) {
        client.workspaceVisibilityChanged(isVisible: isVisible)
    }

    func phase(isTowerReachable: Bool) -> CartridgePhase {
        availability(isTowerReachable: isTowerReachable).forcedPhase ?? state.phase
    }

    func unavailableExplanation(isTowerReachable: Bool) -> String {
        availability(isTowerReachable: isTowerReachable)
            .explanation(cartridgeName: "Scene Understanding", clientReason: clientReason)
    }

    private var clientReason: String? {
        switch state {
        case .unsupported(let reason): return reason
        case .failed(let failure): return failure.message
        case .idle, .awaitingFirstScene, .observing, .lastKnown: return nil
        }
    }
}
