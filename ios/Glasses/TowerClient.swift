//
//  TowerClient.swift
//  Glasses
//
//  Created by Tristan Varner on 8/18/26.
//

import Combine
import Foundation

#if DEBUG
import UIKit
#endif

/// Connection status to the Tower (the project's base-station/hub service).
enum TowerStatus: Equatable {
    case offline
    case connecting
    case online
    case failed(String)
}

@MainActor
protocol FrameRefusalScheduling {
    func schedule(
        after delay: TimeInterval,
        _ action: @escaping @MainActor @Sendable () -> Void
    ) -> @MainActor @Sendable () -> Void
}

@MainActor
private struct MainQueueFrameRefusalScheduler: FrameRefusalScheduling {
    func schedule(
        after delay: TimeInterval,
        _ action: @escaping @MainActor @Sendable () -> Void
    ) -> @MainActor @Sendable () -> Void {
        let item = DispatchWorkItem { Task { @MainActor in action() } }
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: item)
        return { item.cancel() }
    }
}

/// WebSocket client for the Tower connection. Validates connectivity with an
/// initial ping/pong handshake, then keeps a continuous receive loop running
/// for as long as the connection is online so the client can observe
/// `frame_result` messages and — just as importantly — actually notice when
/// the Tower or the OS closes the socket out from under it. A
/// `URLSessionWebSocketDelegate` close callback is a second, independent
/// signal for the same event — though note that callback only fires for a real
/// close *frame*, so the receive loop is the one that catches a dropped link.
///
/// A dropped or wedged connection is re-established automatically when
/// `autoReconnect` is set, on a bounded backoff.
///
/// The one socket carries three message families, demultiplexed by `type` in
/// `handleInboundMessage`: the frame path (`frame` out, which only a DEBUG
/// build sends; `frame_result` and `frame_error` in), the CV Lab control
/// plane, and the read-only result channel.
///
/// Neither of the last two touches the send path: nothing they do changes the
/// connection's `status`, the send window, the stream bracket or `tx_seq`,
/// and a control or result message that fails to send is logged, never
/// escalated. (The frame hold, `pauseFrameSending()`, is a separate call on
/// this client, not a control-plane message.) The CV Lab does decide which
/// `frame_result`s are accepted: a status document sets the watched run
/// (`watchCVLabRun`), `handleFrameResult` discards a result from any other run
/// before it is published or counted as a reply, and a change of run clears
/// the held reading and refusal. Status documents reach that gate as
/// `cv_lab_status`, inside a `cv_lab_error`, or through the CV Lab cartridge
/// client when one arrives on the result channel.
///
/// The decoders live in `TowerSocketMessages.swift` and
/// `CartridgeResultChannel.swift`; `GET /health` in `TowerHealth.swift`.
@MainActor
final class TowerClient: NSObject, ObservableObject {
    @Published private(set) var status: TowerStatus = .offline

    /// Whether the automatic reconnect schedule has been spent and this
    /// client is no longer trying.
    ///
    /// `status` cannot say this. It is `.failed(message)` from the first
    /// dropped socket to the last refused retry and after, so a screen
    /// reading it could only say "not connected" — the same words while the
    /// phone was still retrying and after it had stopped. Without this the
    /// only signal at the give-up point would be a log line, which the wearer
    /// does not read. The World Builder capture control reads this to say,
    /// instead, that a tap on Connect is now the only thing that will bring
    /// the Tower back.
    ///
    /// Set in exactly one place, the give-up branch of `scheduleReconnect`.
    /// Cleared wherever a socket is actually opened — `openConnection`, which
    /// every connect path ends in — and by `cancelReconnect`, because a
    /// deliberate disconnect is not the phone giving up. It cannot survive a
    /// successful reconnect: no reconnect happens without a socket opening.
    @Published private(set) var reconnectGaveUp = false

    #if DEBUG
    /// How many `frame_result` messages the receive loop has processed — the
    /// only end-to-end proof that the Tower received a frame and replied.
    /// `@Published` so the dashboard can show it live; nothing reads it to
    /// make a decision. It invalidates the view tree at the Tower's reply
    /// rate (target ~12 Hz), the same order as `GlassesConnection.frameCount`
    /// does at 24 Hz.
    @Published private(set) var frameResultCount = 0

    /// The most recent `frame_result` the Tower returned.
    ///
    /// The Tower's own reading of a frame's content, which makes it the
    /// evidence the app can show that the round trip is doing something
    /// rather than merely completing. Surfacing it lets a workspace describe
    /// what the Tower actually does without inventing a capability it does
    /// not have.
    ///
    /// Republished at the reply rate, like `frameResultCount` beside it.
    ///
    /// **Gated on `cv_lab.run_id`.** A reply belonging to a run this client is
    /// not watching never reaches here — see `watchedCVLabRunID`.
    @Published private(set) var latestFrameResult: TowerFrameResult?

    #endif

    /// The most recent `frame_error` the Tower returned.
    ///
    /// **Deliberately not `#if DEBUG`, unlike `latestFrameResult` beside it.**
    /// That one is gated because it is a *figure a screen renders*, and a
    /// Release build has no camera, sends no frame, and would render a stale
    /// number from a bracket it cannot open. This is a refusal, and the honest
    /// Release rendering of a refusal is the same as the honest Release
    /// rendering of everything else on the frame path: nothing arrives, because
    /// nothing was sent. Publishing it unconditionally costs one optional and
    /// keeps the CV Lab client free of a second conditional-compilation branch
    /// in the middle of its state machine.
    ///
    /// Cleared at stream start and stop, connection teardown, run change and
    /// after a valid `frame_result`: an earlier refusal is not the current
    /// answer in a new bracket or run, or beside a fresh reading. Each clear
    /// also cancels a pending refresh so the old answer cannot return.
    @Published private(set) var latestFrameRefusal: TowerFrameRefusal?
    /// A one-second window bounds redraws and logs while preserving a useful
    /// capture number when the Tower refuses frames at the send rate.
    private static let frameRefusalPublishInterval: TimeInterval = 1.0
    private var lastFrameRefusalPublishedAt: TimeInterval?
    private var pendingFrameRefusal: TowerFrameRefusal?
    private var cancelFrameRefusalFlush: (@MainActor @Sendable () -> Void)?
    private var frameRefusalFlushToken = 0
    private let frameRefusalNow: () -> TimeInterval
    private let frameRefusalScheduler: any FrameRefusalScheduling

    /// The CV Lab run this client is watching, or `nil`.
    ///
    /// ## The whole staleness rule, in one property
    ///
    /// > **Discard any `frame_result` whose `cv_lab.run_id` is not the run you
    /// > are watching.**
    ///
    /// Learned from the **status document**, never from a `frame_result`, and
    /// that direction is the point: the status document is what says which run
    /// is current, and adopting a run id from a result would make the result
    /// its own authority — the stale reply would nominate itself as the run to
    /// watch and then match.
    ///
    /// While this is `nil` nothing is discarded. That is not a hole: it is the
    /// state of a client that has never read a status, so there is no run it
    /// could be said to be watching and no claim to contradict. The moment a
    /// document arrives, gating begins.
    ///
    /// Because `run_id` is `"<tower_instance_id>-<n>"`, comparing it alone also
    /// covers a reconnect to a **restarted** Tower, whose run numbering starts
    /// again at 1. No separate instance-id check is needed and none is done.
    ///
    /// Cleared at exactly the three points `latestFrameResult` is cleared —
    /// `sendStreamStart()`, `sendStreamStop()` and `teardownConnection` — and
    /// for the same reason each time: a run id held across a boundary the
    /// results did not survive would gate the next bracket's replies against
    /// the previous one's run.
    private(set) var watchedCVLabRunID: String?

    /// How many `frame_result` replies were discarded as belonging to another
    /// run. Non-zero is worth knowing about; it is not an error.
    private(set) var staleFrameResultCount = 0

    /// True between a sent `stream_start` and the matching `stream_stop`.
    /// `sendFrame` will not forward anything while this is false, so a frame
    /// captured in the brief window after `stopCameraSession()` fires (but
    /// before DAT actually tears the stream down) can never reach the Tower.
    ///
    /// **Not `#if DEBUG`, while everything that writes it is.** The two
    /// functions that set it — `sendStreamStart()` and `sendStreamStop()` —
    /// live in the DEBUG-only frame path, so in a Release build this is
    /// permanently `false`, which is the truth about a build with no capture
    /// control on any screen.
    ///
    /// It is readable in both configurations because `TowerWorldBuilderClient`
    /// needs it: "this phone has a capture open" is a fact about the phone's
    /// own situation, and it is what `WorldSessionGate` compares the Tower's
    /// session against. A Release build asks the same question and correctly
    /// gets "no", rather than the question being unaskable there.
    @Published private(set) var isStreamingToTower = false

    /// Whether frames are being **held on the phone** rather than sent.
    ///
    /// ## Why a gate here, and not a camera pause or a `stream_stop`
    ///
    /// The CV Lab's camera card offers Pause and Resume, and neither of the
    /// two obvious implementations is available. DAT offers no app-initiated
    /// camera pause: `StreamState.paused` is something the **glasses** do, on
    /// a temple press or on heat, and `GlassesConnection` documents that it
    /// cannot be overridden in either direction. And sending `stream_stop`
    /// would end the capture lineage on the Tower — the bracket is what a
    /// capture is keyed to, and in the operator's configuration a new bracket
    /// spawns a new follower on the next start.
    ///
    /// So the gate sits on the one hop this app fully owns. While it is
    /// closed the glasses camera keeps running, the socket and the stream
    /// bracket stay up, the Tower keeps the experiment armed, and the only
    /// thing the Tower sees is silence: its `source.receiving_frames` turns
    /// false after `idle_after_s` (5 s), which is exactly what the Lab's LIVE
    /// indicator already reads. Nothing is sent to announce the hold, because
    /// there is no message for it and the Tower's own idle detection is the
    /// truthful report.
    ///
    /// Readable in both configurations, like `isStreamingToTower` beside it,
    /// so `disconnect()` can clear it without a build-conditional. It is
    /// consulted only on the frame path, which is `#if DEBUG`.
    ///
    /// Cleared by `disconnect()` and by `sendStreamStop()`, and deliberately
    /// **not** by `teardownConnection`: a socket that drops and reconnects on
    /// its own schedule is not a person changing their mind, and a hold they
    /// set should still be there when the link comes back.
    @Published private(set) var isFrameSendingPaused = false

    /// Closes the frame gate. Idempotent.
    ///
    /// Frames selected while it is closed are counted as session-gate drops:
    /// they reached a terminal outcome on this side, and
    /// `SenderMetrics.framesUnaccounted` — the one number that exists to prove
    /// frames are not quietly queueing — must not read them as backlog.
    func pauseFrameSending() {
        guard !isFrameSendingPaused else { return }
        isFrameSendingPaused = true
        log("frame sending paused — frames are held on the phone; the camera and the stream bracket stay up")
    }

    /// Reopens the frame gate. Idempotent.
    func resumeFrameSending() {
        guard isFrameSendingPaused else { return }
        isFrameSendingPaused = false
        log("frame sending resumed")
    }

    /// How much outbound latency a frame may carry before the window that
    /// admitted it is considered oversized.
    ///
    /// The send window's capacity is `targetFPS * this` (see
    /// `SendWindow.capacity(forTargetFPS:latencyBudget:)`), so this constant —
    /// not a frame count — is the reviewable decision. At the default 15 fps
    /// target it yields a capacity of 5.
    ///
    /// 1/3 s covers the ~290 ms send completion time measured early in a run
    /// on a physical link: a capacity of 2 against a 290 ms slot lifetime
    /// admits 6.9 fps, which is what that run delivered, and covering that
    /// latency at 15 fps needs 5 slots. It is deliberately not larger: at
    /// 15 fps every extra slot is another 67 ms a frame can be stale before it
    /// is even written.
    static let outboundLatencyBudget: TimeInterval = 1.0 / 3.0

    /// How long a *full* send window may go without a single completion before
    /// the socket is treated as wedged and replaced.
    ///
    /// `URLSessionWebSocketTask` cannot cancel or time out one outstanding
    /// `send`, so the only lever is the connection itself — which makes this a
    /// deliberately reluctant threshold rather than a latency target. 2 s is
    /// long enough that ordinary congestion, a cellular handover or a Tailscale
    /// path change is ridden out rather than answered with a reconnect, and
    /// short enough to cut a recorded 52-second peer-side stall to about 4% of
    /// its cost.
    ///
    /// Anything above this is unrecoverable staleness regardless: a frame
    /// written 2 s after it was captured is not a real-time frame, so nothing
    /// is lost by abandoning it.
    static let sendStallTimeout: TimeInterval = 2.0

    /// The longest gap between consecutive `sendFrame` calls that still counts
    /// as "the main actor was running normally".
    ///
    /// This exists because a send window slot is held from `reserve` until the
    /// completion handler's hop *back onto the main actor* has run — so a slot
    /// that looks old may be a slow network **or** a busy main actor, and those
    /// are exactly the two diagnoses the rest of this file works to keep apart.
    /// Tearing down a perfectly healthy socket because the main thread hitched
    /// would be the worst possible reading of the evidence.
    ///
    /// A main-actor stall is directly observable here: `sendFrame` is called at
    /// the selection rate (~67 ms apart at 15 fps), so a gap far larger than
    /// that means this actor was not running. When that happens the stall
    /// verdict is skipped for one frame, by which time the completion hops
    /// queued during the hitch have run and released their slots.
    ///
    /// 1 s is ~15 missed frames — far beyond any normal scheduling jitter, and
    /// well under `sendStallTimeout`, so a genuine transport stall is still
    /// caught on the very next frame.
    static let mainActorGapAllowance: TimeInterval = 1.0

    /// The bounded set of sends outstanding on the current socket, and the
    /// pipeline's actual rate limiter. See `SendWindow` for why its capacity is
    /// derived from a latency budget rather than picked.
    private var sendWindow: SendWindow

    /// A dense counter over frames this client actually handed to the socket.
    ///
    /// Sent as `tx_seq` beside `seq`. `seq` is the DAT capture index, and this
    /// sender forwards only a fraction of those, so a gap in `seq` at the
    /// Tower has three indistinguishable causes: deliberate sampling, a
    /// sender-side drop, or genuine transit loss. Those are opposite
    /// diagnoses with opposite fixes, and the Tower (`tower/metrics.py`,
    /// `tx_seq_gap_total`) tells them apart by this field.
    ///
    /// The scale of the question: on one physical walk roughly half the
    /// captured frames never reached the Tower — 474 of 953 source indices on
    /// one capture, 2,391 of 4,801 on another — and one gap was 410 source
    /// frames, 17 seconds, which split the reconstruction in two. Without
    /// `tx_seq` nothing could say which of the three causes it was.
    ///
    /// Dense over frames SENT, which is what makes it diagnostic: a gap here
    /// is transit loss and nothing else. So it is claimed at the send itself
    /// (`transmit`), not when a frame is picked up — every earlier step that
    /// ends a frame leaves the number unclaimed for the next frame to use.
    private var txSequence: Int = 0

    /// When `sendFrame` last ran, used only to tell a wedged socket from a
    /// wedged main actor. See `mainActorGapAllowance`. Cleared on teardown, so
    /// the first frame of a new connection never inherits the old one's pulse.
    private var lastSendFrameAt: TimeInterval?

    /// Capacity of the send window, exposed so the developer surface can show
    /// the `capacity / slotLifetime` arithmetic that explains the send rate,
    /// and so tests can assert the derived sizing.
    var maxFramesInFlight: Int { sendWindow.capacity }

    /// Per-frame logging cadence, in send calls. At the target rate this path
    /// runs ~12 times a second and `print` with string interpolation is not
    /// free, so routine success and routine drops are decimated. The
    /// authoritative per-stage counts live in `metrics`.
    private static let frameLogStride = 12
    private var frameLogCounter = 0
    /// Separate budget from `frameLogCounter` so the outbound and inbound
    /// lines cannot crowd each other out — each stays at ~1 Hz.
    private var resultLogCounter = 0

    /// Sender-side instrumentation. Shared with `GlassesConnection` via
    /// `ProjectManager`, which owns both.
    private let metrics: SenderMetrics

    // MARK: Result channel

    /// The Tower's most recent capability declaration, cached.
    ///
    /// Requested once per connection, immediately after the pong — the contract
    /// requires discovery to follow handshake validation, and asking earlier
    /// would read our own reply into the handshake.
    ///
    /// **Deliberately not cleared on teardown.** What the Tower can do is a
    /// property of the Tower's build, not of this socket. Clearing it would
    /// turn every dropped connection into `.noContract` — "this will never
    /// work" — when the truthful reading is `.towerUnreachable`, and those two
    /// call for opposite responses from a person.
    @Published private(set) var cartridgeDeclaration: TowerCartridgeDeclaration?

    /// Every result-channel message, in arrival order.
    ///
    /// A subject rather than four `@Published` properties because ordering
    /// between them is load-bearing: `result_subscribed` is followed
    /// immediately by the first `cartridge_result`, and a consumer that saw
    /// them out of order would file the snapshot against no subscription.
    private let resultEvents = PassthroughSubject<CartridgeResultEvent, Never>()

    /// The result channel, for whoever owns a cartridge's contract.
    ///
    /// `TowerClient` decodes the envelope and nothing else. It does not know
    /// what a world is, does not subscribe on anyone's behalf, and holds no
    /// cartridge state — the cartridge client owned by `ProjectManager` does
    /// all three. That split is what keeps this file cartridge-blind.
    var cartridgeResults: AnyPublisher<CartridgeResultEvent, Never> {
        resultEvents.eraseToAnyPublisher()
    }

    // MARK: CV Lab control plane

    private let cvLabControlEvents = PassthroughSubject<CVLabEvent, Never>()

    /// The CV Lab's control plane, for whoever owns its contract.
    ///
    /// The same split as `cartridgeResults`: this file decodes the message
    /// envelope and passes the status document through undecoded. It does not
    /// know what an experiment is, holds no lifecycle state, and sends no
    /// command on anyone's behalf.
    ///
    /// **The CV Lab's commands travel here rather than on the result channel,
    /// and that is the contract's own exception.** `tower/results/` is a
    /// read-only reporting surface — a Tower test forbids a call named
    /// `observe` or `build` anywhere inside it — so a mutation must not travel
    /// on it. The CV Lab's start, pause and stop are plain socket messages on
    /// `/ws` instead, and there is **no HTTP surface for any of them**: a
    /// command needs the connection it was issued on to still be there when the
    /// outcome arrives, and an arm may take two minutes.
    var cvLabEvents: AnyPublisher<CVLabEvent, Never> {
        cvLabControlEvents.eraseToAnyPublisher()
    }

    private var session: URLSession?
    private var webSocketTask: URLSessionWebSocketTask?
    private var validationTask: Task<Void, Never>?
    private var receiveTask: Task<Void, Never>?

    // MARK: Reconnect

    /// Whether a connection that drops or stalls is re-established
    /// automatically.
    ///
    /// Defaults to `false` — and the production graph in `ProjectManager`
    /// passes `true`. Off by default because reconnect makes `status` a
    /// *sequence* rather than a settled value, and every existing test that
    /// asserts "a dropped connection ends at `.failed`" is asserting about the
    /// settled value. Opting in explicitly keeps those assertions meaningful
    /// instead of racing a reconnect.
    private let autoReconnect: Bool

    /// Endpoint to return to. Set on every `openConnection(to:)` and cleared
    /// by `disconnect()`, so a user-initiated disconnect is never undone by a
    /// reconnect scheduled moments earlier.
    private var reconnectURL: URL?
    private var reconnectTask: Task<Void, Never>?
    /// Failed attempts since the last connection that *held*. Drives the
    /// backoff and the give-up point.
    private var reconnectAttempt = 0

    /// When the current connection reached `.online`, or `nil` if it has not.
    ///
    /// The budget is refilled from this on the way *down* rather than the way
    /// up, because reaching `.online` proves only that the socket opened and
    /// the Tower answered one ping. A Tower that accepts a connection and then
    /// immediately wedges would otherwise reset the counter on every lap and
    /// reconnect forever — a flap loop the bounded schedule exists to prevent.
    private var becameOnlineAt: TimeInterval?

    /// How long a connection must survive before it counts as healthy enough
    /// to earn a fresh reconnect budget. Comfortably longer than one full
    /// backoff schedule, so a flapping Tower always exhausts the schedule and
    /// stops, while a session that runs for minutes and then drops is treated
    /// as the isolated blip it is.
    private static let reconnectBudgetRefillAfter: TimeInterval = 30

    /// Backoff schedule, in seconds. Bounded on purpose: a Tower that is
    /// simply not running must end at a visible `.failed` rather than retrying
    /// forever behind a pill that never settles, and the app has a manual
    /// Connect control for the deliberate retry.
    ///
    /// Five attempts, delayed 0.5, 1, 2, 4 and 8 s — 15.5 s of waiting —
    /// and each attempt is bounded by the handshake watchdog
    /// (`handshakeLegTimeout × 2`, 12 s as shipped), not by the per-leg
    /// timeouts it backs up. So the give-up point depends on how the host
    /// fails: ~16 s against one that refuses the TCP connection outright,
    /// and up to ~75 s against one that accepts TCP and never completes the
    /// WebSocket upgrade.
    private static let reconnectBackoff: [TimeInterval] = [0.5, 1, 2, 4, 8]

    /// The schedule this instance runs. The shipped one unless a test
    /// substitutes a shorter one — the give-up point is observable behaviour,
    /// and a test that reached it by sleeping through the real schedule
    /// would cost the suite the better part of a minute per case.
    private let reconnectBackoff: [TimeInterval]

    /// The shipped send-window capacity, as the arithmetic that justifies it
    /// rather than as a literal.
    static var defaultMaxFramesInFlight: Int {
        SendWindow.capacity(
            forTargetFPS: FrameRateGate.towerTargetFPS,
            latencyBudget: outboundLatencyBudget
        )
    }

    /// Every setting at its shipped value, with metrics of its own.
    override convenience init() {
        self.init(metrics: SenderMetrics())
    }

    /// - Parameters:
    ///   - metrics: Shared sender instrumentation.
    ///   - maxFramesInFlight: Overridable so tests can drive the bounded send
    ///     window deterministically. `nil` uses the latency-budgeted capacity
    ///     described on `outboundLatencyBudget`.
    ///   - stallTimeout: Overridable so tests can trip stall detection without
    ///     waiting `sendStallTimeout` seconds. `nil` uses the shipped value.
    ///   - autoReconnect: See the property of the same name.
    ///   - reconnectBackoff: Overridable so a test can spend the reconnect
    ///     budget in well under a second. `nil` uses the shipped schedule.
    ///   - frameRefusalNow: Monotonic clock for refusal display cadence;
    ///     injectable so the cadence can be exercised without waiting.
    ///   - frameRefusalScheduler: Schedules one cancellable trailing refresh;
    ///     injectable so a test can fire or cancel it without waiting.
    ///
    /// Actor-isolated defaults are resolved in the body rather than in default
    /// arguments, which are evaluated outside this type's actor.
    init(
        metrics: SenderMetrics,
        maxFramesInFlight: Int? = nil,
        stallTimeout: TimeInterval? = nil,
        autoReconnect: Bool = false,
        handshakeLegTimeout: Int? = nil,
        reconnectBackoff: [TimeInterval]? = nil,
        frameRefusalNow: @escaping () -> TimeInterval = { MonotonicClock.now },
        frameRefusalScheduler: (any FrameRefusalScheduling)? = nil
    ) {
        self.metrics = metrics
        self.handshakeLegTimeout = handshakeLegTimeout ?? Self.defaultHandshakeLegTimeout
        self.reconnectBackoff = reconnectBackoff ?? Self.reconnectBackoff
        self.sendWindow = SendWindow(
            capacity: maxFramesInFlight ?? Self.defaultMaxFramesInFlight,
            stallTimeout: stallTimeout ?? Self.sendStallTimeout
        )
        self.autoReconnect = autoReconnect
        self.frameRefusalNow = frameRefusalNow
        self.frameRefusalScheduler = frameRefusalScheduler ?? MainQueueFrameRefusalScheduler()
        super.init()
    }

    /// - Parameter url: The Tower endpoint. `nil` uses the real Tower
    ///   (`TowerConfiguration.webSocketURL`); overridable so tests can point
    ///   this at a local mock server instead. `nil`-defaulted and resolved in
    ///   the body for the reason `init` gives: a default argument is evaluated
    ///   outside this type's actor, and `webSocketURL` is main-actor isolated.
    ///
    /// A caller-initiated connect also refills the reconnect budget: an
    /// exhausted schedule is how the client says "I have stopped trying", and
    /// a deliberate tap on Connect is the user saying to try again.
    func connect(to url: URL? = nil) {
        let url = url ?? TowerConfiguration.webSocketURL
        // Refilled only when this call is actually going to open a socket. A
        // redundant tap during an in-flight connect opens nothing (see
        // `openConnection`'s `.connecting` guard), so it must not resurrect an
        // exhausted schedule either. The budget is meant to say "I have
        // stopped trying"; a no-op must not undo that.
        if status != .connecting { reconnectAttempt = 0 }
        openConnection(to: url)
    }

    /// Connects only if nothing is connected or in flight, and **without**
    /// refilling the reconnect budget.
    ///
    /// The entry point for automation — app launch, specifically. It is
    /// deliberately not `connect()`: that call means "the user asked to try
    /// again", which is why it refills the budget and why it is allowed to
    /// replace a live connection. Neither is true of code running on its own
    /// initiative, and routing automation through the same door would dissolve
    /// the bound that stops a dead Tower from being retried forever.
    ///
    /// Guarding on `.offline` also makes this safe to call more than once: it
    /// will not disturb a healthy connection, cancel a pending reconnect, or
    /// restart a schedule that has already given up. A Tower that has failed
    /// stays failed and visible until the user acts.
    func connectIfIdle(to url: URL? = nil) {
        let url = url ?? TowerConfiguration.webSocketURL
        guard status == .offline else {
            log("automatic connect skipped — status is \(status)")
            return
        }
        openConnection(to: url)
    }

    /// The connect path itself, without refilling the reconnect budget — so a
    /// scheduled reconnect advances through the backoff rather than resetting
    /// it and retrying forever.
    private func openConnection(to url: URL) {
        // A different Tower is a different set of capabilities.
        //
        // `cartridgeDeclaration` deliberately survives a teardown, and that is
        // right: what a Tower can do is a property of its build, not of one
        // socket, so clearing it on every drop would turn a reconnect into
        // "this will never work". None of that reasoning applies when the
        // endpoint itself changes — the previous Tower's declaration says
        // nothing about this one. Left standing, it would race the new
        // connection's own `cartridges` reply and could drive the first
        // `result_subscribe` with the old Tower's contract, which the new one
        // answers with `contract_mismatch` and prose about the wrong thing.
        if let previous = reconnectURL, previous != url {
            log("endpoint changed \(previous) -> \(url); dropping the previous Tower's declaration")
            cartridgeDeclaration = nil
        }

        // Recorded before the in-flight guard below, so that a connect made
        // while one is already under way still retargets a later reconnect. Do
        // not move this after the guard: a pending reconnect would then quietly
        // return to the *previous* endpoint.
        reconnectURL = url

        guard status != .connecting else { return }

        // A caller-initiated connect supersedes any pending reconnect, so the
        // two cannot both open a socket. The reconnect path clears this itself
        // before calling in, so this is a no-op there rather than
        // self-cancellation.
        reconnectTask?.cancel()
        reconnectTask = nil

        if webSocketTask != nil {
            log("connect() called with a previous connection still active — tearing it down first")
        }
        teardownConnection(cancelWith: .normalClosure)

        log("connection attempt: \(url)")
        status = .connecting
        // A socket is being opened, so "stopped trying" is no longer true —
        // whoever asked for it. Here and not in `connect(to:)`, so a reconnect
        // and `connectIfIdle` clear it too; after the `.connecting` guard, so
        // a redundant tap that opens nothing changes nothing.
        reconnectGaveUp = false

        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        self.session = session
        let task = session.webSocketTask(with: url)
        webSocketTask = task
        task.resume()
        log("WebSocket opened (resume() called)")

        validationTask = Task { [weak self] in
            await self?.validateConnection(task: task)
        }
        armHandshakeWatchdog(for: task)
    }

    /// Guarantee that a connection attempt ends.
    ///
    /// `validateConnection` awaits a `send` and a `receive` on the socket, and
    /// neither reliably returns when a peer accepts the TCP connection and then
    /// never completes the WebSocket upgrade. Structured-concurrency timeouts
    /// cannot rescue that — a task group has to await its own child before it
    /// can throw, and that child is the stuck call — so the bound is enforced
    /// from outside, on the thing that really is cancellable: the socket.
    ///
    /// `fail(_:task:)` tears the connection down, which makes the pending calls
    /// error out, sets `.failed`, and lets `scheduleReconnect` advance. Without
    /// it a hung connect would park the client in `.connecting` permanently:
    /// the attempt budget never spent, nothing trying again, and nothing
    /// saying so. A 9-second reconnect has been measured with ~8.5 s of it
    /// spent in `.connecting`; this does not make that faster — it makes a
    /// hung connect terminate.
    private func armHandshakeWatchdog(for task: URLSessionWebSocketTask) {
        handshakeWatchdog?.cancel()
        // Both legs plus room for the ordinary case to finish on its own, so
        // this fires only when the per-leg bounds have failed to.
        let budget = handshakeLegTimeout * 2
        handshakeWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(budget))
            guard !Task.isCancelled else { return }
            self?.handshakeDidTimeOut(task: task, after: budget)
        }
    }

    private func handshakeDidTimeOut(task: URLSessionWebSocketTask, after seconds: Int) {
        // Only if this attempt is still the current one and still unresolved. A
        // connection that came online, failed on its own, or was superseded has
        // already had its outcome.
        guard isCurrent(task), status == .connecting else { return }
        fail("Handshake did not complete within \(seconds)s", task: task)
    }

    func disconnect() {
        log("disconnect() called")
        // Cleared before teardown, so a failure observed on the way down
        // cannot schedule a reconnect the user just asked to stop. `fail()`
        // also refuses to act once `status` is `.offline`, but that is set
        // after teardown — this is what closes the window between the two.
        cancelReconnect()
        teardownConnection(cancelWith: .normalClosure)
        status = .offline
        // The user asked for this connection to end, so the hold they set
        // during it ends too — a fresh connect streams. See the property for
        // why an *automatic* teardown does not do this.
        isFrameSendingPaused = false
        log("disconnect cleanup complete")
    }

    private func cancelReconnect() {
        reconnectTask?.cancel()
        reconnectTask = nil
        reconnectURL = nil
        reconnectAttempt = 0
        becameOnlineAt = nil
        // The user asked for this connection to end. That is not the phone
        // giving up, and a screen must not say it was.
        reconnectGaveUp = false
    }

    /// Queues one delayed reconnect attempt, if automatic reconnect is enabled
    /// and the schedule has not been exhausted.
    ///
    /// Reconnect is not an optimisation here — it is what makes stall recovery
    /// possible at all. Tearing down a wedged socket without restoring one
    /// would trade a stalled pipeline for a dead one.
    private func scheduleReconnect() {
        guard autoReconnect, let url = reconnectURL else { return }
        guard reconnectTask == nil else { return }

        // Refill the budget only for a connection that actually held. Reading
        // it here, at the point of failure, is what makes "it worked for a
        // while" mean something — see `becameOnlineAt`.
        if let onlineAt = becameOnlineAt,
           MonotonicClock.now - onlineAt >= Self.reconnectBudgetRefillAfter {
            reconnectAttempt = 0
        }
        becameOnlineAt = nil

        guard reconnectAttempt < reconnectBackoff.count else {
            log("reconnect given up after \(reconnectAttempt) attempts — use Connect to retry")
            // `status` stays `.failed` and cannot carry this; see the property.
            reconnectGaveUp = true
            return
        }

        let delay = reconnectBackoff[reconnectAttempt]
        reconnectAttempt += 1
        let attempt = reconnectAttempt
        log("reconnect attempt \(attempt) scheduled in \(delay)s")

        reconnectTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled else { return }
            guard let self else { return }
            // Cleared before `connect(to:)` so that call's own
            // supersede-any-pending-reconnect step does not cancel the task it
            // is currently running on.
            self.reconnectTask = nil
            guard self.reconnectURL == url else {
                self.log("reconnect attempt \(attempt) abandoned — endpoint changed or disconnected")
                return
            }
            self.log("reconnect attempt \(attempt) starting")
            self.openConnection(to: url)
        }
    }

    #if DEBUG
    /// Sends one camera frame to the Tower as a JSON text message on the open
    /// socket: `type` `frame`, `seq`, `tx_seq`, `width`, `height`, `format`
    /// `jpeg`, and `data`, the base64 of a quality-0.5 JPEG. One frame at a
    /// time: no batching, no binary framing, no rate adaptation (see
    /// docs/03-ROADMAP.md V0.7 for where those belong).
    ///
    /// Four steps. Each one that ends the frame records exactly one terminal
    /// outcome in `metrics`, so `framesUnaccounted` stays truthful:
    ///
    /// 1. `passSessionGates`: online, inside a stream bracket, not paused.
    ///    Otherwise a session-gate drop.
    /// 2. `passSendWindow`: backpressure. A stalled window replaces the
    ///    connection; a full one drops the frame. Either way a send-window drop.
    /// 3. `encodeFrame`: JPEG, base64, JSON. Otherwise an encode failure.
    /// 4. `transmit`: reserves the slot, claims the `tx_seq`, sends.
    ///    `finishFrameSend` records the completion.
    ///
    /// Main-actor isolated with no suspension point, so no other frame can
    /// take the slot between steps 2 and 4, or the same `tx_seq`.
    func sendFrame(_ image: UIImage, width: Int, height: Int, sequence: Int) {
        frameLogCounter += 1
        let shouldLog = frameLogCounter % Self.frameLogStride == 1

        guard let task = passSessionGates(sequence: sequence, shouldLog: shouldLog) else { return }
        guard passSendWindow(task: task, sequence: sequence, shouldLog: shouldLog) else { return }
        guard let frame = encodeFrame(image, width: width, height: height, sequence: sequence) else { return }
        transmit(frame, on: task, shouldLog: shouldLog)
    }

    /// Step 1 of `sendFrame`: the three session gates.
    ///
    /// - Returns: the socket to send on, or `nil` once a session-gate drop has
    ///   been counted.
    private func passSessionGates(sequence: Int, shouldLog: Bool) -> URLSessionWebSocketTask? {
        guard status == .online, let task = webSocketTask else {
            metrics.recordSessionGateDrop()
            if shouldLog {
                log("frame #\(sequence) not sent — Tower not online (status=\(status))")
            }
            return nil
        }
        guard isStreamingToTower else {
            metrics.recordSessionGateDrop()
            if shouldLog {
                log("frame #\(sequence) not sent — no stream_start sent yet (or stream_stop already sent)")
            }
            return nil
        }
        guard !isFrameSendingPaused else {
            // A person closed the gate. Not an error, and not a stall: the
            // socket is fine and nothing is queued. Counted as a session-gate
            // drop for the reason `pauseFrameSending()` gives, and checked
            // before the stall test in step 2 so a long hold cannot be misread
            // as a socket that stopped draining.
            metrics.recordSessionGateDrop()
            if shouldLog {
                log("frame #\(sequence) held — frame sending is paused")
            }
            return nil
        }
        return task
    }

    /// Step 2 of `sendFrame`: backpressure. A stalled window replaces the
    /// connection; a full one drops the frame.
    ///
    /// - Returns: whether the window has room. `false` once a send-window drop
    ///   has been counted.
    private func passSendWindow(task: URLSessionWebSocketTask, sequence: Int, shouldLog: Bool) -> Bool {
        let now = MonotonicClock.now

        // How long since the previous frame was offered. This is the main
        // actor's own pulse, and it is what stops a main-actor hitch from
        // being misread as a wedged socket — see `mainActorGapAllowance`.
        let sinceLastFrame = lastSendFrameAt.map { now - $0 }
        lastSendFrameAt = now

        // A full window that has not returned a single slot within
        // `stallTimeout` is a socket that is not draining. There is no way to
        // cancel the outstanding sends individually, so the connection is the
        // unit of recovery — and without this the pipeline simply reports
        // send-window drops for as long as the peer takes to resume, which
        // has been recorded at 52 seconds.
        //
        // Checked here, on the send path, rather than from a timer: this runs
        // at the selection rate whenever there are frames to send, which is
        // exactly when a stall costs something. With no frames arriving there
        // is no throughput to lose, and the next frame detects it immediately.
        //
        // The gap test is the false-positive guard. A slot's age includes the
        // completion handler's hop back to this actor, so if this actor has
        // itself been stalled, every slot looks old through no fault of the
        // socket. Requiring that the previous frame was offered recently means
        // the verdict is only ever reached while the main actor is demonstrably
        // running — and one frame later, the hops queued during the hitch have
        // drained. A first frame (`nil`) is treated as a gap, since there is no
        // pulse to judge yet.
        let mainActorWasResponsive = (sinceLastFrame ?? .infinity) <= Self.mainActorGapAllowance
        if mainActorWasResponsive, sendWindow.isStalled(at: now) {
            replaceStalledConnection(task: task, at: now)
            return false
        }

        // Checked before encoding, so a frame we are going to drop never costs
        // a JPEG encode.
        guard !sendWindow.isFull else {
            metrics.recordSendWindowDrop()
            if shouldLog {
                log("frame #\(sequence) dropped — \(sendWindow.inFlight) sends already in flight (window \(sendWindow.capacity))")
            }
            return false
        }
        return true
    }

    /// The stall branch of step 2: logs the verdict beside the healthy
    /// distribution, accounts for the frame that found the stall, and replaces
    /// the connection.
    private func replaceStalledConnection(task: URLSessionWebSocketTask, at now: TimeInterval) {
        let age = sendWindow.oldestAge(at: now) ?? 0
        // The healthy distribution alongside the verdict, from the same rows
        // the developer surface shows. A stall is only interpretable against
        // what this link normally does: an oldest-slot age of 2 s means one
        // thing when the running max is 90 ms and quite another when it is
        // 1.8 s.
        let snapshot = metrics.snapshot
        let slotMax: String = snapshot.slotLifetimeMsMax.map { String(format: "%.0f", $0) } ?? "-"
        let sendMax: String = snapshot.sendLatencyMsMax.map { String(format: "%.0f", $0) } ?? "-"
        let ageText: String = String(format: "%.1f", age)
        let outstanding: Int = sendWindow.inFlight
        let priorRecoveries: Int = snapshot.stallRecoveries
        let diagnosis: String = "slot ms max \(slotMax), send ms max \(sendMax), prior stall recoveries \(priorRecoveries)"
        log("send window stalled — \(outstanding) sends outstanding, oldest \(ageText)s (\(diagnosis)); replacing the connection")
        // This frame still has to reach a terminal outcome, or every stall
        // would leave one selected frame permanently unaccounted for and
        // `framesUnaccounted` would drift upwards — the one number that
        // exists to prove frames are not quietly queueing. A window drop is
        // the honest label: `isStalled` implies `isFull`, so the frame was
        // dropped for a full window, exactly like the ones before it.
        metrics.recordSendWindowDrop()
        // Recorded after the teardown it describes, so the counter can only
        // ever report recoveries that actually happened.
        fail("Send stalled for \(String(format: "%.1f", age))s", task: task)
        metrics.recordStallRecovery()
    }

    /// One frame, encoded and not yet sent: what step 3 hands to step 4.
    private struct EncodedFrame {
        let sequence: Int
        let width: Int
        let height: Int
        /// The `tx_seq` inside `text`. Read, not yet claimed: see `transmit`.
        let txSeq: Int
        /// The whole message, as the socket sends it.
        let text: String
        /// Size of the serialised JSON, which is what `wireBytes` counts.
        let wireBytes: Int
        let jpegBytes: Int
        /// When encoding finished. Also the instant the send-window slot is
        /// reserved: one clock read ends the encode figure and starts the slot
        /// timings.
        let encodedAt: TimeInterval
    }

    /// Step 3 of `sendFrame`: JPEG, base64 and JSON, timed into `metrics`.
    ///
    /// - Returns: the encoded frame, or `nil` once an encode failure has been
    ///   counted.
    private func encodeFrame(_ image: UIImage, width: Int, height: Int, sequence: Int) -> EncodedFrame? {
        // A clock read of its own, so the encode figure covers the encode and
        // nothing else: step 2's `now` was taken before the stall and window
        // checks.
        let encodeStart = MonotonicClock.now
        guard let jpegData = image.jpegData(compressionQuality: 0.5) else {
            metrics.recordEncodeFailure()
            log("frame #\(sequence) failed to encode as JPEG")
            return nil
        }

        // Read, not yet claimed: `transmit` claims it. Every `return` between
        // here and the send leaves this number for the next frame, so `tx_seq`
        // stays dense over frames the socket actually received. Claiming it
        // here instead would make an encode failure or a closed send window
        // look identical to transit loss at the Tower — the precise confusion
        // `txSequence` exists to remove.
        let txSeq = txSequence
        let payload: [String: Any] = [
            "type": "frame",
            "seq": sequence,
            "tx_seq": txSeq,
            "width": width,
            "height": height,
            "format": "jpeg",
            "data": jpegData.base64EncodedString(),
        ]

        guard
            let jsonData = try? JSONSerialization.data(withJSONObject: payload),
            let jsonText = String(data: jsonData, encoding: .utf8)
        else {
            metrics.recordEncodeFailure()
            log("frame #\(sequence) failed to serialize JSON payload")
            return nil
        }
        let encodedAt = MonotonicClock.now
        metrics.recordEncode(seconds: encodedAt - encodeStart)
        return EncodedFrame(
            sequence: sequence,
            width: width,
            height: height,
            txSeq: txSeq,
            text: jsonText,
            wireBytes: jsonData.count,
            jpegBytes: jpegData.count,
            encodedAt: encodedAt
        )
    }

    /// Step 4 of `sendFrame`: reserves the slot, claims the `tx_seq`, and hands
    /// the frame to the socket. The completion is recorded by `finishFrameSend`.
    private func transmit(_ frame: EncodedFrame, on task: URLSessionWebSocketTask, shouldLog: Bool) {
        let sequence = frame.sequence
        let reservedAt = frame.encodedAt

        // Cannot fail: `passSendWindow` checked the window not-full and
        // nothing else can reserve in between — `sendFrame` is main-actor
        // isolated and contains no suspension point. Handled rather than
        // force-unwrapped so a future edit that breaks that property loses a
        // frame instead of trapping in the user's hands.
        guard let token = sendWindow.reserve(at: reservedAt) else {
            metrics.recordSendWindowDrop()
            log("frame #\(sequence) dropped — send window closed between check and reserve")
            return
        }
        metrics.recordSendAttempt(wireBytes: frame.wireBytes)
        // Claimed here, at the last statement before the frame goes out. The
        // window is reserved, the JSON exists, and nothing between this line
        // and `task.send` can decline to send. `sendFrame` is main-actor
        // isolated with no suspension point, so no other frame can interleave
        // and take the same number.
        txSequence += 1
        if shouldLog {
            log("frame #\(sequence) (tx \(frame.txSeq)) sending \(frame.wireBytes) bytes (\(frame.width)x\(frame.height), jpeg \(frame.jpegBytes) bytes)")
        }

        task.send(.string(frame.text)) { [weak self] error in
            // Sampled here, in the transport's own completion handler, before
            // the main-actor hop. That is the whole point: the difference
            // between this instant and the one measured after the hop
            // separates "the network is slow" from "the main actor is busy",
            // which are opposite diagnoses.
            let completedAt = MonotonicClock.now
            // `[weak self]` on the Task itself, not inherited from the send
            // completion's capture. A weak capture is a mutable binding, and
            // reading the enclosing closure's copy from concurrently-executing
            // code is an error under the Swift 6 language mode. Re-capturing
            // here is evaluated when the Task is created, which is allowed,
            // and keeps the lifetime semantics identical.
            Task { @MainActor [weak self] in
                self?.finishFrameSend(
                    token: token,
                    task: task,
                    reservedAt: reservedAt,
                    completedAt: completedAt,
                    error: error,
                    sequence: sequence,
                    shouldLog: shouldLog
                )
            }
        }
    }

    /// The main-actor half of a frame send's completion: returns the slot and
    /// records the outcome.
    private func finishFrameSend(
        token: Int,
        task: URLSessionWebSocketTask,
        reservedAt: TimeInterval,
        completedAt: TimeInterval,
        error: (any Error)?,
        sequence: Int,
        shouldLog: Bool
    ) {
        // A completion for a socket this client no longer owns: its
        // reservation was already cleared by teardown, so `release`
        // returns nil and the slot count is left alone — otherwise
        // this would credit a slot on the *next* connection and
        // permanently widen its window. The outcome is also not this
        // connection's to report — but it still has to be *recorded*,
        // or the frame would look permanently in flight and the
        // accounting invariant would false-alarm after every
        // disconnect.
        let releasedAt = MonotonicClock.now
        guard
            isCurrent(task),
            let slotLifetime = sendWindow.release(token, at: releasedAt)
        else {
            metrics.recordSendAbandoned()
            return
        }
        metrics.recordSlotTiming(
            sendLatency: completedAt - reservedAt,
            slotLifetime: slotLifetime
        )

        if let error {
            metrics.recordSendFailure()
            log("frame #\(sequence) send failed: \(error.localizedDescription)")
            fail("Send failed: \(error.localizedDescription)", task: task)
        } else {
            metrics.recordSendSuccess()
            if shouldLog {
                log("frame #\(sequence) sent")
            }
        }
    }

    /// Marks the stream as active and sends `{"type":"stream_start"}` once,
    /// over the existing Tower WebSocket, so the Tower knows to expect
    /// frames. Fire-and-forget: no response is awaited or expected. A no-op
    /// if already streaming, so a redundant call (e.g. DAT re-delivering the
    /// `.streaming` state) can't send it twice for the same session.
    func sendStreamStart() {
        guard !isStreamingToTower else {
            log("stream_start suppressed — already streaming")
            return
        }
        // The flag is set only if the marker actually reached a socket. Set
        // first, a start attempted while the Tower was offline would leave
        // `isStreamingToTower == true` with the Tower never told, so every
        // frame of that session would go out outside any stream bracket and
        // the eventual `stream_stop` would be unmatched.
        guard sendLifecycleMarker(type: "stream_start") else { return }
        isStreamingToTower = true
        // Scoped to one stream bracket, which is a narrower thing than a
        // camera session: a dropped connection reopens the bracket on its
        // own. The product screen therefore reads
        // `SenderMetrics.frameResults` instead; this counter stays per-bracket
        // and is shown only on the developer surface, where "replies on the
        // current bracket" is the useful reading. A lifetime-cumulative
        // counter would diverge by tens of thousands over a long run and
        // invite reading the pair as a delivery ratio.
        frameResultCount = 0
        // Scoped to the bracket for the same reason as the count above: a reply
        // from the previous bracket displayed against a fresh one is a stale
        // claim about the current session.
        latestFrameResult = nil
        clearCVLabRunWatch()
    }

    /// Marks the stream as inactive and sends `{"type":"stream_stop"}` once.
    /// From this point, `sendFrame` will not forward anything until the next
    /// `sendStreamStart()`. A no-op if not currently streaming.
    func sendStreamStop() {
        // A hold belongs to the camera session it was set in. The camera has
        // stopped; a later start must stream, not inherit a pause nobody
        // remembers setting and that no control on that later screen shows.
        // Cleared BEFORE the bracket guard, deliberately: a socket drop
        // already closed the bracket (`teardownConnection`), and a camera
        // stopped during that gap would otherwise keep the hold into the
        // next session, where Home has no control that shows it.
        isFrameSendingPaused = false
        guard isStreamingToTower else {
            log("stream_stop suppressed — not currently streaming")
            return
        }
        isStreamingToTower = false
        // Cleared with the bracket it belongs to. The tile that shows it is
        // captioned "latest Tower reply", and after a stop there is no current
        // reply - leaving the last one on screen would date it silently.
        latestFrameResult = nil
        clearCVLabRunWatch()
        _ = sendLifecycleMarker(type: "stream_stop")
    }

    #endif

    /// Forget which CV Lab run this client is watching, and the last refusal.
    ///
    /// ## Why this is called from exactly three places
    ///
    /// The same three that clear `latestFrameResult`: `sendStreamStart()`,
    /// `sendStreamStop()` and `teardownConnection`. The held run id and the
    /// held reply have identical lifetimes because they answer the same
    /// question — *what is the Tower currently telling us about frames* — and a
    /// run id that outlived a bracket would silently change what the gate
    /// means: replies from the next bracket would be measured against a run
    /// that ended, and every one of them discarded.
    ///
    /// It is **not** called when the run merely changes. That case is handled
    /// where the new run id arrives, by adopting it and dropping the reading
    /// that belonged to the old one — see `watchCVLabRun(_:)`.
    ///
    /// Outside `#if DEBUG` although two of its three callers are inside it, so
    /// that `teardownConnection`'s Release path can call it: the run watch is
    /// fed by the status document, which is a read-only surface a Release build
    /// reaches perfectly well.
    private func clearCVLabRunWatch() {
        watchedCVLabRunID = nil
        clearFrameRefusal()
    }

    /// Adopt the run the Tower's status document says is current.
    ///
    /// The **only** writer of `watchedCVLabRunID` other than the three clears,
    /// and it is fed exclusively from status documents — a `cv_lab_status`
    /// reply, a pushed one, or the unchanged document a `cv_lab_error` carries.
    /// Never from a `frame_result`: a result that nominated its own run as the
    /// one to watch would always match, which is the gate deleting itself.
    ///
    /// When the run **changes**, the held reading is dropped with it. A run is
    /// the unit of provenance, and the Tower makes the same move on its side —
    /// starting a different experiment mints a new run and takes the previous
    /// one's figures out of the document entirely, *"because keeping an old
    /// summary beside a new one is how a number from the wrong experiment ends
    /// up on a screen"*. Leaving the last reading on screen across a switch
    /// would put the old experiment's number under the new experiment's name.
    ///
    /// A `nil` run id — the Lab is idle, or unavailable — clears the watch
    /// rather than being ignored: there is no current run, so there is nothing
    /// to gate against, and holding the previous one would discard the results
    /// of whatever starts next.
    ///
    /// ## Why this is not `private`
    ///
    /// The two callers in this file are both `cv_lab_status` messages, and
    /// **the Tower sends `cv_lab_status` only in reply to a message on that
    /// same connection** (`tower/routes/cv_lab_ws.py`) — there is no
    /// unsolicited push. So they cover exactly one case: a run change *this
    /// phone caused*.
    ///
    /// The Lab has **one slot shared by every connection, and last start wins**.
    /// A run change caused by anyone else — a second operator, the bench's own
    /// `cv_lab_smoke.py` — reaches this app only as a `cartridge_result` on the
    /// subscription. If nothing on that path could update the gate:
    ///
    ///  - every subsequent `frame_result` would carry the new run and be
    ///    discarded, so `SenderMetrics.frameResults` — the counter the product
    ///    screen reads — **would freeze while frames were being answered
    ///    normally**;
    ///  - `latestFrameResult` would never be cleared, so the card would keep
    ///    showing the **previous experiment's figures** under the new
    ///    experiment's name.
    ///
    /// That is precisely the outcome this gate exists to prevent — a number
    /// from the wrong experiment on a screen — produced *by* the gate.
    ///
    /// So the cartridge client calls this when it applies a status document it
    /// received over the subscription. That does not weaken the rule the doc
    /// above states: the feed is still a **status document** and never a
    /// `frame_result`, which is the distinction that matters. It widens the set
    /// of status documents from "the ones we asked for" to "all of them".
    ///
    /// Idempotent by the first guard, so a client that also asked directly can
    /// call it twice with no effect.
    func watchCVLabRun(_ runID: String?) {
        guard runID != watchedCVLabRunID else { return }
        log("cv_lab run watch: \(watchedCVLabRunID ?? "none") -> \(runID ?? "none")")
        watchedCVLabRunID = runID
        #if DEBUG
        latestFrameResult = nil
        #endif
        clearFrameRefusal()
    }

    #if DEBUG

    /// Shared send path for the two stream lifecycle markers — same
    /// WebSocket, same fire-and-forget `send` used by `sendFrame`, no new
    /// connection, no reply awaited. Deliberately bypasses the frame send
    /// window: markers are two-byte payloads that define session boundaries,
    /// and delaying or dropping one corrupts every frame count on either side
    /// of it.
    ///
    /// - Returns: whether the marker was handed to a socket. Not whether the
    ///   Tower received it — that is still fire-and-forget.
    private func sendLifecycleMarker(type: String) -> Bool {
        guard status == .online, let task = webSocketTask else {
            log("\(type) not sent — Tower not online (status=\(status))")
            return false
        }
        guard
            let jsonData = try? JSONSerialization.data(withJSONObject: ["type": type]),
            let jsonText = String(data: jsonData, encoding: .utf8)
        else {
            log("\(type) failed to serialize JSON payload")
            return false
        }
        task.send(.string(jsonText)) { [weak self] error in
            // Re-captured weakly here for the same reason as in `transmit`.
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let error {
                    self.log("\(type) send failed: \(error.localizedDescription)")
                    self.fail("Send failed: \(error.localizedDescription)", task: task)
                } else {
                    self.log("\(type) sent")
                }
            }
        }
        return true
    }
    #endif

    // MARK: - Result channel: outbound

    /// Asks the Tower what it can report on.
    ///
    /// Sent once per connection, from `validateConnection` after the pong has
    /// been read — never before. Not `#if DEBUG`: the result channel is
    /// read-only and says nothing about frames, so a Release build that cannot
    /// stream is still entitled to a truthful answer about what the Tower can
    /// do.
    func requestCartridgeDeclaration() {
        sendResultMessage(["type": "cartridges"], label: "cartridges")
    }

    /// Opens a subscription. The reply is a `result_subscribed` followed
    /// immediately by a complete snapshot, whatever cursor was sent.
    ///
    /// `contract` is included so the Tower refuses outright rather than
    /// serving a payload this build was not written against — a
    /// `contract_mismatch` error is a better outcome than a silent
    /// misinterpretation.
    ///
    /// `worldID` and `sessionID` pin the subscription to a stored world; the
    /// Tower resolves the newest live world when neither is sent. The keys are
    /// inserted only when non-nil, so an unpinned subscribe is byte-for-byte
    /// the message it always was — a `"world_id": null` would be a claim about
    /// a world rather than the absence of one.
    func subscribeToResults(
        cartridge: String, resultType: String, contract: String,
        worldID: String? = nil, sessionID: String? = nil
    ) {
        var message: [String: Any] = [
            "type": "result_subscribe",
            "cartridge": cartridge,
            "result_type": resultType,
            "contract": contract,
        ]
        if let worldID { message["world_id"] = worldID }
        if let sessionID { message["session_id"] = sessionID }
        sendResultMessage(message, label: "result_subscribe(\(cartridge))")
    }

    /// Closes a subscription. Not required before disconnecting — the Tower
    /// treats a closed socket as sufficient cleanup — so this exists for the
    /// case where the connection outlives the reason to be subscribed.
    func unsubscribeFromResults(subscriptionID: String) {
        sendResultMessage(
            ["type": "result_unsubscribe", "subscription_id": subscriptionID],
            label: "result_unsubscribe(\(subscriptionID))"
        )
    }

    // MARK: - CV Lab control: outbound

    /// Asks the Lab for its whole state.
    ///
    /// **Not `#if DEBUG`, and this is the message that proves the rule.** The
    /// CV Lab's read-only half is reachable from a build with no camera: a
    /// Release build can enumerate the experiments and read the Lab's state
    /// truthfully, and the contract versions the control vocabulary separately
    /// from the status document precisely so that *"a client may implement the
    /// read-only half and never send a command"*.
    ///
    /// - Parameter requestID: echoed back on the reply, so an answer can be
    ///   matched to the button that was pressed. Bounded by
    ///   `boundedRequestID(_:)` before it goes out.
    func sendCVLabStatusRequest(requestID: String? = nil) {
        sendCVLabCommand(.status, requestID: requestID)
    }

    /// Selects **and arms** an experiment, replacing whatever was running.
    ///
    /// There is deliberately no separate select message: selection without
    /// arming is a state nobody needs on the wire.
    ///
    /// The reply is immediate and says `starting`, not `running` — an arm is
    /// asynchronous, and a start that later fails to load has **already been
    /// answered `accepted`** by then. The outcome arrives as state on a later
    /// status document, which is why a client that sends this must also read
    /// status.
    func sendCVLabStart(experimentID: String, requestID: String? = nil) {
        sendCVLabCommand(
            .start,
            extra: ["experiment_id": experimentID],
            requestID: requestID
        )
    }

    /// Stops processing and keeps the experiment loaded.
    ///
    /// - Parameter runID: the run the button was drawn against. **Send it
    ///   whenever there is one.** A command naming a run that is no longer
    ///   current is refused `stale_run` rather than applied to whichever run
    ///   is — which is the difference between a refusal and the wrong run being
    ///   paused by the wrong person.
    func sendCVLabPause(runID: String? = nil, requestID: String? = nil) {
        sendCVLabCommand(.pause, runID: runID, requestID: requestID)
    }

    /// Resumes a paused run. Costs nothing: the experiment never left memory.
    func sendCVLabResume(runID: String? = nil, requestID: String? = nil) {
        sendCVLabCommand(.resume, runID: runID, requestID: requestID)
    }

    /// Ends the run, releases the experiment, and **keeps the figures**.
    ///
    /// The only way to keep a run readable: starting a different experiment
    /// mints a new run and takes the previous one's figures out of the document
    /// entirely.
    func sendCVLabStop(runID: String? = nil, requestID: String? = nil) {
        sendCVLabCommand(.stop, runID: runID, requestID: requestID)
    }

    /// Shared shape for the five control messages.
    ///
    /// Sent down `sendUnescalatedMessage` rather than `sendLifecycleMarker`,
    /// and that choice is the important one: **a refused or undeliverable
    /// command must not tear down the frame socket.** A start is a request
    /// about what the Tower should compute; the frames are the session. Failing
    /// the connection because a control message did not land would let the
    /// control plane do the one thing the contract promises it cannot — affect
    /// the frame path.
    private func sendCVLabCommand(
        _ command: CVLabCommand,
        extra: [String: Any] = [:],
        runID: String? = nil,
        requestID: String? = nil
    ) {
        var object: [String: Any] = ["type": command.rawValue]
        for (key, value) in extra { object[key] = value }
        if let runID { object["run_id"] = runID }
        if let requestID = boundedRequestID(requestID) { object["request_id"] = requestID }
        sendUnescalatedMessage(object, label: command.rawValue)
    }

    /// Bounds a `request_id` to the 64 characters the Tower echoes.
    ///
    /// **A longer one is dropped, not refused**: the command applies and the
    /// reply simply comes back carrying no `request_id`. That is the worst
    /// possible failure for a client that matches replies to buttons — the
    /// command takes effect and the answer cannot be attributed to it — so this
    /// client refuses to send one it knows will be dropped rather than
    /// discovering the loss on the reply.
    ///
    /// Dropped locally rather than truncated: a truncated id is a *different*
    /// id, and one that could collide with another button's.
    private func boundedRequestID(_ requestID: String?) -> String? {
        guard let requestID, !requestID.isEmpty else { return nil }
        guard requestID.count <= ExperimentalCVContract.requestIDMaxLength else {
            log(
                "cv_lab request_id dropped locally: \(requestID.count) characters, "
                    + "the Tower echoes at most \(ExperimentalCVContract.requestIDMaxLength) "
                    + "and drops longer ones without refusing the command"
            )
            return nil
        }
        return requestID
    }

    /// Shared send path for the three result-channel messages.
    ///
    /// Kept as a named wrapper rather than folded into
    /// `sendUnescalatedMessage`: "these three are the result channel" is a fact
    /// worth keeping visible, because a fourth kind of message — the CV Lab's
    /// commands, which are **not** on the result channel — shares the same send
    /// discipline.
    private func sendResultMessage(_ object: [String: Any], label: String) {
        sendUnescalatedMessage(object, label: label)
    }

    /// Sends a message whose failure must not cost the frame path.
    ///
    /// **A send failure here is logged and not escalated**, which is the one
    /// way this differs from `sendLifecycleMarker`. A lifecycle marker defines
    /// a frame bracket and losing one corrupts the counts on both sides, so
    /// that path fails the connection; a subscribe is a request for a report
    /// and a CV Lab command is a request about what to compute, and tearing
    /// down the socket the camera is streaming over because either did not land
    /// would let a control surface do the one thing the contract promises it
    /// cannot — affect the frame path. If the socket really is gone, the
    /// receive loop notices it on its own terms.
    ///
    /// Note that this sits **outside** the `#if DEBUG` block that covers the
    /// frame path, so commands can be sent from a Release build. That is not an
    /// oversight: a Release build has no camera and will get no `frame_result`,
    /// but nothing about the control plane depends on frames, and a Tower
    /// driven from a phone that is not the one streaming is a real bench
    /// configuration.
    private func sendUnescalatedMessage(_ object: [String: Any], label: String) {
        guard status == .online, let task = webSocketTask else {
            log("\(label) not sent — Tower not online (status=\(status))")
            return
        }
        guard
            let jsonData = try? JSONSerialization.data(withJSONObject: object),
            let jsonText = String(data: jsonData, encoding: .utf8)
        else {
            log("\(label) failed to serialize JSON payload")
            return
        }
        task.send(.string(jsonText)) { [weak self] error in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let error {
                    self.log("\(label) send failed (not escalated): \(error.localizedDescription)")
                } else {
                    self.log("\(label) sent")
                }
            }
        }
    }

    /// The bound on each half of the opening handshake — the connect-and-ping
    /// leg, and the pong. See `validateConnection`.
    private static let defaultHandshakeLegTimeout = 6
    /// Injectable so a test can prove the bound exists without spending six
    /// seconds per leg doing it.
    private let handshakeLegTimeout: Int
    /// Bounds one whole connection attempt. See `armHandshakeWatchdog`.
    private var handshakeWatchdog: Task<Void, Never>?

    /// Sends one ping and validates the pong within a bounded timeout. On
    /// success, hands off to the continuous receive loop.
    private func validateConnection(task: URLSessionWebSocketTask) async {
        do {
            let pingPayload = try JSONSerialization.data(withJSONObject: ["type": "ping"])
            guard let pingText = String(data: pingPayload, encoding: .utf8) else {
                fail("Could not encode ping payload", task: task)
                return
            }

            // Leg timings, so a slow reconnect can be decomposed rather than
            // guessed at. One 9-second outage decomposed to ~8.5 s in
            // `.connecting` — a brand-new socket could not complete its upgrade
            // either — so the teardown decision cost well under a second and
            // the path or the peer cost the rest. Three hypotheses fit that (a
            // new flow paying SYN retransmission, a Tower event loop blocked in
            // reconstruction, a genuinely dead link), and they imply different
            // fixes.
            let handshakeBegan = MonotonicClock.now

            // NOT wrapped in `withTimeout`, deliberately — `armHandshakeWatchdog`
            // is what bounds this.
            //
            // This `send` cannot return until TCP connect and the HTTP Upgrade
            // have completed, so it *is* the connect leg. `withTimeout` **does
            // not bound it**: a throwing task group must await its remaining
            // child before it can propagate the sleeper's error, and that child
            // is the call that is stuck. Measured against a listener that
            // accepts TCP and never upgrades: a one-second `withTimeout` here
            // still left the client `.connecting` twelve seconds later.
            //
            // The same caveat applies to the pong's `withTimeout` below. It is
            // kept because it does bound the ordinary case; it is simply not
            // the guarantee. Cancelling the socket is.
            try await task.send(.string(pingText))
            let connectMs = (MonotonicClock.now - handshakeBegan) * 1000
            log("ping sent (connect+upgrade \(Int(connectMs)) ms): \(pingText)")

            let message = try await withTimeout(seconds: handshakeLegTimeout) {
                try await task.receive()
            }
            log("message received: \(message)")

            guard
                case .string(let text) = message,
                let data = text.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
                json["type"] == "pong"
            else {
                fail("Unexpected/malformed response from Tower", task: task)
                return
            }

            let handshakeMs = (MonotonicClock.now - handshakeBegan) * 1000
            log(
                "pong validated (connect+upgrade \(Int(connectMs)) ms,"
                    + " pong \(Int(handshakeMs - connectMs)) ms,"
                    + " handshake total \(Int(handshakeMs)) ms)"
            )
            guard !Task.isCancelled, isCurrent(task) else { return }
            // Deliberately does *not* refill the reconnect budget yet — see
            // `becameOnlineAt`. Reaching `.online` is not evidence of a working
            // connection; staying there is.
            becameOnlineAt = MonotonicClock.now
            // The handshake is done; its bound has nothing left to bound.
            handshakeWatchdog?.cancel()
            handshakeWatchdog = nil
            status = .online
            startReceiveLoop(task: task)
            // After the pong, never before. The Tower never speaks first, so
            // nothing could have arrived early — but asking before validating
            // would mean reading our own reply into the handshake, which is
            // the failure the contract warns about. The receive loop is
            // already running, so the answer has somewhere to land.
            requestCartridgeDeclaration()
        } catch is CancellationError {
            // disconnect() was called mid-validation; state already handled there.
        } catch {
            fail("Connection failed: \(error.localizedDescription)", task: task)
        }
    }

    private func startReceiveLoop(task: URLSessionWebSocketTask) {
        receiveTask?.cancel()
        receiveTask = Task { [weak self] in
            await self?.receiveLoop(task: task)
        }
    }

    /// Runs for the lifetime of one connection, continuously draining
    /// inbound messages (chiefly `frame_result`). A `receive()` failure is
    /// the definitive signal that the connection is gone, so it's the one
    /// place (alongside the delegate close callback) responsible for moving
    /// `status` off `.online` truthfully instead of leaving it stale.
    private func receiveLoop(task: URLSessionWebSocketTask) async {
        log("receive loop started")
        while !Task.isCancelled {
            do {
                let message = try await task.receive()
                guard isCurrent(task) else {
                    log("receive loop stopped (superseded connection)")
                    return
                }
                handleInboundMessage(message)
            } catch {
                guard isCurrent(task) else {
                    log("receive loop stopped (superseded connection)")
                    return
                }
                log("receive failed: \(error.localizedDescription)")
                fail("Connection lost: \(error.localizedDescription)", task: task)
                return
            }
        }
        log("receive loop stopped (cancelled)")
    }

    private func handleInboundMessage(_ message: URLSessionWebSocketTask.Message) {
        guard case .string(let text) = message else {
            log("unknown message type: non-text frame received")
            return
        }
        guard
            let data = text.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let type = json["type"] as? String
        else {
            log("unknown message type: undecodable payload")
            return
        }

        switch type {
        case "frame_result": handleFrameResult(json)
        case "frame_error": handleFrameError(json)
        case "cv_lab_status": handleCVLabStatus(json)
        case "cv_lab_error": handleCVLabError(json)
        case "cartridges": handleCartridgeDeclaration(json)
        case "result_subscribed": handleResultSubscribed(json)
        case "result_unsubscribed": handleResultUnsubscribed(json)
        case "cartridge_result": handleCartridgeResult(json)
        case "result_error": handleResultError(json)
        case "protocol_error": handleProtocolError(json)
        default: log("unknown message type: \(type)")
        }
    }

    // MARK: Inbound: the frame path

    private func handleFrameResult(_ json: [String: Any]) {
        let result = TowerFrameResult(json: json)

        // >>> The staleness gate. <<<
        //
        // A result belonging to a run this client is not watching is
        // discarded here and reaches nothing below — not the published
        // reading, not the reply counter, not the metrics. It is the whole
        // of the rule the contract states in one line, and the reason it
        // sits at the top of this handler rather than at the point of display
        // is that there is more than one consumer downstream and a gate per
        // consumer is a gate somebody forgets.
        //
        // `watchedCVLabRunID == nil` means no status has ever been read, so
        // there is no run being watched and nothing to contradict — every
        // result is passed through. A reply with no `cv_lab` block at all
        // is passed through for the same reason: a Tower that attaches no
        // provenance has made no claim about which run this belongs to, and
        // discarding it would drop every result from a Tower running no Lab.
        if let watched = watchedCVLabRunID, let runID = result.cvLab?.runID, runID != watched {
            staleFrameResultCount += 1
            log(
                "frame_result discarded: run \(runID) is not the watched run "
                    + "\(watched) (stale results so far: \(staleFrameResultCount))"
            )
            return
        }

        // Decimated on the same 1-in-`frameLogStride` cadence as the send
        // path. This arrives once per delivered frame, so at the target
        // rate an unguarded line here is ~12 prints a second — and the
        // string builds two `Optional.map` allocations before `print` even
        // takes its lock. `metrics.frameResults` is the real count.
        resultLogCounter += 1
        if resultLogCounter % Self.frameLogStride == 1 {
            log(
                "frame_result received: seq=\(result.sequence.map(String.init) ?? "?")"
                    + " mean_intensity=\(result.meanIntensity.map { String($0) } ?? "?")"
                    + " processing_ms=\(result.processingMs.map { String($0) } ?? "?")"
            )
        }
        #if DEBUG
        frameResultCount += 1
        // Kept for every reply, not only the one-in-twelve that is logged:
        // the counters are exact and the surfaced value has to be too. The
        // publish is the only real cost, the same order as `frameResultCount`
        // next to it.
        latestFrameResult = result
        #endif
        // A result cleared the gate, so whatever refusal was last on screen
        // has been answered. Cleared here rather than left to age: "the Lab
        // is paused" beside a fresh reading is two claims that cannot both
        // be current.
        clearFrameRefusal()
        metrics.recordFrameResult()
    }

    /// Publishes changed refusals immediately and the latest repeat by the end of its window.
    /// Decodes each reply -> checks reason and cadence -> publishes or replaces
    /// the pending value; every decoded reply still reaches CV Lab consumers.
    // Internal so cadence tests can deliver decoded wire fields synchronously;
    // socket delivery is covered by the frame_error transport test.
    func handleFrameError(_ json: [String: Any]) {
        // A paused Lab for five minutes has not failed hundreds of times.
        // Refusals leave the send window and reply counters intact; the Tower
        // counts each under frames_rejected, not frame_processing_errors.
        guard let refusal = TowerFrameRefusal(json: json) else {
            log("frame_error could not be decoded")
            return
        }
        let now = frameRefusalNow()
        let changed = latestFrameRefusal?.reason != refusal.reason
        let refreshDue = lastFrameRefusalPublishedAt.map {
            now - $0 >= Self.frameRefusalPublishInterval
        } ?? true
        if changed || refreshDue {
            cancelPendingFrameRefusal()
            publishFrameRefusal(refusal, at: now)
        } else {
            pendingFrameRefusal = refusal
            if cancelFrameRefusalFlush == nil, let lastFrameRefusalPublishedAt {
                schedulePendingFrameRefusal(
                    after: max(0, Self.frameRefusalPublishInterval - (now - lastFrameRefusalPublishedAt))
                )
            }
        }
        cvLabControlEvents.send(.frameRefused(refusal))
    }

    private func publishFrameRefusal(_ refusal: TowerFrameRefusal, at now: TimeInterval) {
        latestFrameRefusal = refusal
        lastFrameRefusalPublishedAt = now
        log(
            "frame_error: seq=\(refusal.sequence.map(String.init) ?? "?")"
                + " reason=\(refusal.reason) — \(refusal.message)"
        )
    }

    private func schedulePendingFrameRefusal(after delay: TimeInterval) {
        frameRefusalFlushToken &+= 1
        let token = frameRefusalFlushToken
        cancelFrameRefusalFlush = frameRefusalScheduler.schedule(after: delay) { [weak self] in
            self?.flushPendingFrameRefusal(token: token)
        }
    }

    private func flushPendingFrameRefusal(token: Int) {
        guard token == frameRefusalFlushToken, let pendingFrameRefusal else { return }
        cancelFrameRefusalFlush = nil
        let now = frameRefusalNow()
        if let lastFrameRefusalPublishedAt {
            let remaining = Self.frameRefusalPublishInterval - (now - lastFrameRefusalPublishedAt)
            if remaining > 0 {
                schedulePendingFrameRefusal(after: remaining)
                return
            }
        }
        self.pendingFrameRefusal = nil
        publishFrameRefusal(pendingFrameRefusal, at: now)
    }

    private func cancelPendingFrameRefusal() {
        pendingFrameRefusal = nil
        cancelFrameRefusalFlush?()
        cancelFrameRefusalFlush = nil
        frameRefusalFlushToken &+= 1
    }

    private func clearFrameRefusal() {
        cancelPendingFrameRefusal()
        latestFrameRefusal = nil
        lastFrameRefusalPublishedAt = nil
    }

    // MARK: Inbound: CV Lab control plane
    //
    // Both handlers are additive and neither touches the send path. Both can
    // move the run gate (`watchCVLabRun`) that decides which `frame_result`s
    // are accepted, and both pass the status document through undecoded —
    // this file owns the transport, the cartridge owns the contract.

    private func handleCVLabStatus(_ json: [String: Any]) {
        guard let reply = CVLabControlReply(json: json) else {
            log("cv_lab_status could not be decoded")
            return
        }
        log(
            "cv_lab_status: state=\(reply.lifecycleState ?? "?")"
                + " run=\(reply.runID ?? "none")"
                // The field that separates an answer from a push. Logged
                // because reading a pushed heartbeat as the answer to a
                // command is the mistake this field exists to prevent, and
                // a log that does not distinguish them cannot show it.
                + " accepted_command=\(reply.acceptedCommand ?? "—")"
                + " request_id=\(reply.requestID ?? "—")"
        )
        watchCVLabRun(reply.runID)
        cvLabControlEvents.send(.status(reply))
    }

    private func handleCVLabError(_ json: [String: Any]) {
        guard let refusal = CVLabControlRefusal(json: json) else {
            log("cv_lab_error could not be decoded")
            return
        }
        // Every one of the eight reasons means the request did not take
        // effect, so nothing is applied here — not even optimistically. The
        // refusal carries the unchanged document and the run watch is
        // updated from it, which is the same path an accepted command
        // takes: state comes from the document, never from the command.
        log(
            "cv_lab_error: \(refusal.reason) for \(refusal.command ?? "?")"
                + " — \(refusal.message)"
        )
        if let status = refusal.status {
            watchCVLabRun(CVLabControlReply.runID(in: status))
        }
        cvLabControlEvents.send(.refused(refusal))
    }

    // MARK: Inbound: result channel
    //
    // Every handler below is additive: they decode, publish, and return.
    // Nothing here touches `status`, the send window, the stream bracket or
    // the run gate — which is the iOS half of the guarantee the contract
    // makes on the Tower side. (A subscriber can still move the run gate: the
    // CV Lab cartridge client calls `watchCVLabRun` with a status it receives
    // here.)

    private func handleCartridgeDeclaration(_ json: [String: Any]) {
        let declaration = TowerCartridgeDeclaration(json: json)
        log(
            "cartridges declared: "
                + declaration.offers
                .map { "\($0.cartridge)/\($0.resultType) available=\($0.available)" }
                .joined(separator: ", ")
        )
        cartridgeDeclaration = declaration
        resultEvents.send(.declaration(declaration))
    }

    private func handleResultSubscribed(_ json: [String: Any]) {
        guard let ack = CartridgeSubscriptionAck(json: json) else {
            log("result_subscribed could not be decoded")
            return
        }
        log("result_subscribed: \(ack.subscriptionID) \(ack.cartridge)/\(ack.resultType)")
        resultEvents.send(.subscribed(ack))
    }

    private func handleResultUnsubscribed(_ json: [String: Any]) {
        guard let id = json["subscription_id"] as? String else { return }
        log("result_unsubscribed: \(id)")
        resultEvents.send(.unsubscribed(subscriptionID: id))
    }

    private func handleCartridgeResult(_ json: [String: Any]) {
        guard let envelope = CartridgeResultEnvelope(json: json) else {
            log("cartridge_result could not be decoded")
            return
        }
        // Decimated like `frame_result`, and for a weaker reason: this
        // arrives at most twice a second. One line per change rather than
        // one per heartbeat is still the useful reading.
        if envelope.revisionChanged {
            log(
                "cartridge_result: \(envelope.cartridge)/\(envelope.resultType)"
                    + " seq=\(envelope.sequence.map(String.init) ?? "?")"
                    + " revision=\(envelope.revision ?? "?")"
                    + " coalesced=\(envelope.coalesced)"
            )
        }
        resultEvents.send(.result(envelope))
    }

    private func handleResultError(_ json: [String: Any]) {
        guard let error = CartridgeResultError(json: json) else {
            log("result_error could not be decoded")
            return
        }
        log("result_error: \(error.reason) — \(error.message)")
        resultEvents.send(.failed(error))
    }

    private func handleProtocolError(_ json: [String: Any]) {
        // The Tower telling us it does not implement something we sent.
        // Additive on its side and non-fatal on ours: it is what tells "not
        // implemented" from "lost in flight" on this side.
        let messageType = json["message_type"].map { String(describing: $0) } ?? "nil"
        log("protocol_error from Tower: \(json["reason"] as? String ?? "?") for \(messageType)")
    }

    /// True only if `task` is still the socket this client currently owns —
    /// used to ignore work (receive-loop errors, validation results) that
    /// belongs to a connection already superseded by a later connect()/
    /// disconnect(), so a stale callback can never clobber current state.
    private func isCurrent(_ task: URLSessionWebSocketTask) -> Bool {
        guard let current = webSocketTask else { return false }
        return current === task
    }

    /// Fails only if `task` is still current; otherwise the failure belongs
    /// to an already-superseded connection and is logged, not acted on.
    private func fail(_ message: String, task: URLSessionWebSocketTask) {
        guard isCurrent(task) else {
            log("ignoring stale failure (superseded connection): \(message)")
            return
        }
        fail(message)
    }

    private func fail(_ message: String) {
        guard status != .offline else { return }
        log("error: \(message)")
        teardownConnection(cancelWith: .abnormalClosure)
        status = .failed(message)
        // After the status is settled, so an observer that reacts to `.failed`
        // sees a consistent client. Documented URLSession behaviour is that
        // one send error fails *all* outstanding work on the task, so the
        // sibling completions arriving next are already stale by `isCurrent`
        // and cannot schedule a second reconnect.
        scheduleReconnect()
    }

    private func teardownConnection(cancelWith closeCode: URLSessionWebSocketTask.CloseCode) {
        handshakeWatchdog?.cancel()
        handshakeWatchdog = nil
        validationTask?.cancel()
        validationTask = nil
        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: closeCode, reason: nil)
        webSocketTask = nil
        // `URLSession` retains its delegate — this object — until it is
        // invalidated. Dropping the reference is not enough: without this, the
        // session, its delegate queue and this client all outlive every
        // teardown, once per connect — and `autoReconnect` re-opens on a
        // schedule whose budget refills after every 30s of healthy connection,
        // which is once per drop on exactly the flaky link this client was
        // built for.
        //
        // `invalidateAndCancel` rather than `finishTasksAndInvalidate`: the
        // only task was cancelled on the line above, so there is nothing left
        // to finish, and waiting would keep the session alive past the point
        // this method promises it is gone. Late delegate callbacks from the
        // invalidated session are already ignored — `webSocketTask` is nil by
        // then, so `handleDelegateClose`'s identity guard drops them.
        session?.invalidateAndCancel()
        session = nil

        // The send window belongs to one socket. Any completion handlers still
        // pending for the old task are ignored by their `isCurrent` guard, so
        // clearing here is the only thing that reopens the window for the next
        // connection — otherwise a dropped connection would permanently leak
        // window slots and eventually stop sending altogether. `SendWindow`
        // does not rewind its token counter, so those late completions cannot
        // release a slot belonging to the *next* connection either.
        sendWindow.reset()
        // Belongs to the socket that is going away: the next connection's
        // first frame must not be judged against the old one's pulse.
        lastSendFrameAt = nil
        // Same reason, and it matters more than it looks. The Tower counts
        // `tx_seq` gaps per CONNECTION (`SessionMetrics` is built per socket),
        // so carrying the counter across a reconnect would present the frames
        // this client never sent on the old socket as a gap on the new one —
        // manufacturing exactly the transit loss `tx_seq` exists to measure.
        // A physical walk has reconnected mid-session.
        txSequence = 0

        #if DEBUG
        // `isStreamingToTower` means "a stream_start has been sent and not yet
        // matched by a stream_stop". No stream_start survives a socket, so
        // leaving this true across a teardown would be a lie, and would let
        // frames flow to a Tower that never received a start for the
        // connection they arrive on. Set directly rather than via
        // `sendStreamStop()`: there is no socket left to send on.
        isStreamingToTower = false
        // Scoped to the socket that carried it, for the same reason as the
        // line above. `HomeWorkspaceView` renders this under the caption
        // "latest Tower reply", and a reading from a connection that is gone
        // is not the latest anything. The stream bracket alone would not
        // clear it: a socket that drops mid-capture would leave the dead
        // connection's number on screen for the whole outage — and forever
        // once the reconnect budget is spent, because no `stream_stop` is
        // ever sent for a socket that is not there.
        latestFrameResult = nil
        #endif
        // The third of the three clears, and the only one that runs in a
        // Release build. A run id is a fact about the Tower on the other end of
        // *this* socket, and the socket is what just went away: the next one
        // may reach a different Tower, or the same one restarted, and either
        // way the run it names has to be learned again from a document rather
        // than assumed to have survived. Holding a run id across a teardown
        // would gate the next connection's replies against the last one's run
        // and discard every one of them until a status arrived.
        clearCVLabRunWatch()
    }

    // MARK: - What the Tower says about itself

    /// The reader for `GET /health`. A `var` so a test can point it at a
    /// stubbed `URLSession`; nothing in the app ever reassigns it.
    var healthClient = TowerHealthHTTPClient()

    /// What this app has been told about the Tower's own state, including
    /// whether its dataset recorder is armed and writing.
    ///
    /// Starts at `.notFetched`, which is a real state and not a stand-in for
    /// "nothing is happening" — see `TowerHealthState`.
    @Published private(set) var healthState: TowerHealthState = .notFetched

    /// The one read in flight, or `nil`. Held so a second tap on Refresh
    /// cannot start a second request that would race the first into the same
    /// property.
    private var healthTask: Task<Void, Never>?

    /// Asks the Tower how it is, once.
    ///
    /// ## Why this is a button and not a timer
    ///
    /// Nothing in the app makes a decision from this; it exists so a person
    /// can *ask*, and the answer they get is the answer from the moment they
    /// asked, stamped with that moment. A poll would spend battery and a
    /// Tailscale round trip every few seconds to keep a developer screen warm
    /// that is usually not open, and it would add a timer to manage across
    /// backgrounding — a resource with no reader.
    ///
    /// ## Why it never throws
    ///
    /// An unreachable Tower and an unreadable answer are both things the
    /// screen has to be able to say, so they are states rather than errors,
    /// exactly as `TowerObjectMemoryClient.ask` treats the same two.
    func refreshHealth() {
        // A second tap while one is in flight is dropped rather than queued:
        // they are the same button, and two answers racing into one property
        // is a worse outcome than one ignored tap. Same rule as
        // `TowerObjectMemoryClient.isAsking`.
        guard healthTask == nil else { return }
        healthState = .fetching

        // Copied out before the hop so the request itself never has to touch
        // this actor, and captured directly rather than through `self`.
        let client = healthClient
        healthTask = Task { [weak self] in
            let outcome: Result<TowerHealth, TowerHealthFetchError>
            do {
                outcome = .success(try await client.health())
            } catch let error as TowerHealthFetchError {
                outcome = .failure(error)
            } catch {
                // The only honest attribution available without knowing where
                // it came from.
                outcome = .failure(.transport(error.localizedDescription))
            }

            // Resolved *after* the await, so this client is only held for the
            // hop back and not for the lifetime of the request. The body is a
            // single bounded await rather than a loop, which is what makes
            // that safe here — a `guard let self` outside an unbounded `for
            // await` is the shape that promotes a weak capture to a strong one
            // for the task's whole life.
            guard let self else { return }
            self.healthTask = nil
            // A cancelled read has no answer to report — but it cannot simply
            // return, because `.fetching` was set *before* the await and the
            // previous answer is already gone. Leaving `.fetching` standing
            // would leave `DeveloperToolsView.isCheckingHealth` true forever,
            // with the button reading "Checking…" and disabled while nothing is
            // in flight: a control that has quietly stopped working, which is
            // exactly the failure this screen exists to make visible.
            //
            // `.notFetched` is what is true afterwards — the question was
            // abandoned, not answered, and nobody has asked since. Guarded on
            // `.fetching` so a state somebody else has since written is not
            // stamped over.
            if Task.isCancelled {
                if case .fetching = self.healthState { self.healthState = .notFetched }
                return
            }

            let at = Date()
            switch outcome {
            case .success(let health):
                self.healthState = .fetched(health, at: at)
            case .failure(let error):
                self.healthState = .failed(error, at: at)
            }
        }
    }

    private func log(_ message: String) {
        #if DEBUG
        print("[Glasses][Tower] \(message)")
        #endif
    }
}

extension TowerClient: URLSessionWebSocketDelegate {
    /// Independent, socket-level signal that the Tower (or the OS) closed
    /// the connection — a second detection path alongside the receive
    /// loop's error case, for whichever one notices first. Hops to the main
    /// actor before touching any state, and only acts if the closed task is
    /// still the one this client currently owns, so it can never race a
    /// receive-loop failure (or a newer connection) into a conflicting
    /// status update.
    nonisolated func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
        reason: Data?
    ) {
        let closedTaskID = ObjectIdentifier(webSocketTask)
        let codeValue = closeCode.rawValue
        let reasonText = reason.flatMap { String(data: $0, encoding: .utf8) }
        Task { @MainActor [weak self] in
            self?.handleDelegateClose(closedTaskID: closedTaskID, codeValue: codeValue, reason: reasonText)
        }
    }

    private func handleDelegateClose(closedTaskID: ObjectIdentifier, codeValue: Int, reason: String?) {
        guard let current = webSocketTask, ObjectIdentifier(current) == closedTaskID else {
            log("delegate close ignored (stale/superseded connection), code=\(codeValue)")
            return
        }
        log("delegate close: code=\(codeValue) reason=\(reason ?? "none")")
        fail("Tower closed the connection (code \(codeValue))")
    }

    #if DEBUG
    /// Test-only hook: invokes the real `didCloseWith` delegate callback for
    /// the current connection, exercising the exact production code path
    /// without requiring a real socket-level close frame from the network.
    func simulateDelegateCloseForTesting(code: URLSessionWebSocketTask.CloseCode) {
        guard let task = webSocketTask else { return }
        urlSession(session ?? URLSession(configuration: .default), webSocketTask: task, didCloseWith: code, reason: nil)
    }
    #endif
}

/// Races an async operation against a timeout, since
/// `URLSessionWebSocketTask.receive()` has no built-in timeout.
private func withTimeout<T: Sendable>(
    seconds: Int,
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await operation() }
        group.addTask {
            try await Task.sleep(for: .seconds(seconds))
            throw TowerClientError.timedOut
        }
        guard let result = try await group.next() else {
            throw TowerClientError.timedOut
        }
        group.cancelAll()
        return result
    }
}

private enum TowerClientError: LocalizedError {
    case timedOut

    var errorDescription: String? {
        switch self {
        case .timedOut: return "Timed out waiting for pong"
        }
    }
}
