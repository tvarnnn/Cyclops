//
//  SendWindow.swift
//  Glasses
//

import Foundation

/// The bounded set of frame sends currently outstanding on one WebSocket.
///
/// Its own type because the send window is the pipeline's *rate limiter*, not
/// merely a memory guard, and a rate limiter deserves to be reasoned about —
/// and tested — on its own.
///
/// ## Why this is a rate limiter
///
/// `TowerClient` admits a frame only while a slot is free, and a slot is held
/// from the moment a frame is handed to `URLSessionWebSocketTask.send` until
/// that send's completion handler runs. In steady state that makes the
/// achievable send rate exactly
///
///     admittedFPS = capacity / averageSlotLifetime
///
/// which is bounded *above* by the frame-rate gate and bounded *below* by
/// nothing at all. A capacity of 2 against a 290 ms slot lifetime is 6.9 fps
/// no matter how much bandwidth is spare, and against a 590 ms slot lifetime
/// is 3.4 fps — the two rates a physical run reported.
///
/// ## Capacity is a latency budget, not a magic number
///
/// Capacity is therefore derived rather than picked: it is how many frames the
/// target rate puts on the wire within the outbound latency we are willing to
/// carry (`SendWindow.capacity(forTargetFPS:latencyBudget:)`). Raising it
/// trades staleness for throughput on a latency-dominated link, and *only*
/// there — on a bandwidth-dominated link the slot lifetime grows in proportion
/// and the delivered rate is unchanged, which is the correct behaviour: the
/// window still sheds load rather than queueing it.
///
/// ## Stall detection is what makes that trade safe
///
/// A larger window is only safe if a wedged socket cannot sit on every slot
/// indefinitely. `URLSessionWebSocketTask` offers no way to cancel or time out
/// an individual outstanding `send`, so the only available remedy is at
/// connection granularity — which is why this type reports `isStalled(at:)`
/// and leaves the remedy to its owner. A 52-second peer-side stall has been
/// recorded; with no detection that is 52 seconds during which every slot is
/// held, every frame is dropped, and the frames eventually written are 52
/// seconds stale. Bounding that is the freshness invariant in
/// docs/03-ROADMAP.md V0.7, stated in time rather than in frames.
///
/// A pure value type taking the current time as a parameter, so slot lifetime,
/// exhaustion and stall detection are all unit testable without a socket, a
/// timer, or real elapsed time — the same shape as `FrameRateGate`.
///
/// Every member is explicitly `nonisolated`: the project builds with
/// `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, which would otherwise make even
/// a capacity read main-actor-only. The stored properties are not annotated —
/// they are `Sendable` members of a global-actor-isolated value type, which
/// `SWIFT_APPROACHABLE_CONCURRENCY = YES` makes implicitly nonisolated.
/// `FrameRateGate` relies on exactly the same rule; if that build setting is
/// ever turned off, both types need revisiting together.
struct SendWindow {

    /// How many sends may be outstanding at once. Always at least 1.
    let capacity: Int

    /// How long the oldest outstanding send may remain outstanding, on a full
    /// window, before the socket is considered wedged. Always positive.
    let stallTimeout: TimeInterval

    /// The occupied slots, in reservation order — which is also send order.
    ///
    /// `token` is never reused, including across `reset()`, so a completion
    /// handler belonging to a torn-down socket can never release a slot
    /// belonging to the current one.
    ///
    /// Bounded by `capacity`, so the linear scans below are over at most a
    /// handful of elements and a dictionary would cost more than it saved.
    ///
    /// A tuple rather than a nested struct on purpose: the project builds with
    /// `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, so an unannotated nested
    /// type would be main-actor isolated and could not be constructed from the
    /// `nonisolated` methods below.
    private var outstanding: [(token: Int, startedAt: TimeInterval)] = []

    /// Monotonically increasing for the lifetime of the value — deliberately
    /// *not* rewound by `reset()`, which is what keeps a late completion from a
    /// torn-down socket from matching a live reservation.
    private var nextToken = 0

    nonisolated init(capacity: Int, stallTimeout: TimeInterval) {
        // Clamped rather than trapped: a zero capacity would stall the
        // pipeline permanently, and a non-positive timeout would report every
        // full window as stalled and tear the connection down under load —
        // the exact opposite of what stall detection is for.
        self.capacity = max(1, capacity)
        self.stallTimeout = stallTimeout > 0 ? stallTimeout : .infinity
    }

    /// Capacity sized from an explicit outbound latency budget: how many
    /// frames a `targetFPS` source puts on the wire in `latencyBudget`
    /// seconds.
    ///
    /// This is the whole justification for whatever number `TowerClient` ends
    /// up using. Stating it as an arithmetic makes the trade reviewable —
    /// "how stale may a frame be by the time it is written?" is a question
    /// about a real-time system; "is 2 enough?" is not.
    nonisolated static func capacity(
        forTargetFPS targetFPS: Double,
        latencyBudget: TimeInterval
    ) -> Int {
        guard targetFPS > 0, latencyBudget > 0 else { return 1 }
        return max(1, Int((targetFPS * latencyBudget).rounded()))
    }

    // MARK: Occupancy

    nonisolated var inFlight: Int { outstanding.count }

    nonisolated var isFull: Bool { outstanding.count >= capacity }

    /// Takes a slot if one is free. The returned token must be handed back to
    /// `release(_:at:)` when the send completes.
    ///
    /// - Returns: the slot's token, or `nil` if the window is full — in which
    ///   case the caller must drop the frame. Declining to admit is the only
    ///   available backpressure: a WebSocket stream cannot be reordered, so an
    ///   already-queued frame cannot be replaced by a fresher one.
    nonisolated mutating func reserve(at now: TimeInterval) -> Int? {
        guard !isFull else { return nil }
        let token = nextToken
        nextToken += 1
        outstanding.append((token: token, startedAt: now))
        return token
    }

    /// Returns the slot identified by `token`.
    ///
    /// - Returns: how long the slot was held — the send's true slot lifetime,
    ///   which is the denominator of the rate arithmetic above — or `nil` if
    ///   the token is unknown. An unknown token means the reservation was
    ///   cleared by `reset()` while the send was still outstanding, i.e. the
    ///   socket was torn down underneath it. That is not an error, but the
    ///   slot is not this window's to give back either, so the count is left
    ///   alone.
    nonisolated mutating func release(_ token: Int, at now: TimeInterval) -> TimeInterval? {
        guard let index = outstanding.firstIndex(where: { $0.token == token }) else { return nil }
        let reservation = outstanding.remove(at: index)
        return max(0, now - reservation.startedAt)
    }

    // MARK: Stall detection

    /// How long the oldest outstanding send has been outstanding, or `nil`
    /// when nothing is in flight.
    nonisolated func oldestAge(at now: TimeInterval) -> TimeInterval? {
        guard let oldest = outstanding.min(by: { $0.startedAt < $1.startedAt }) else { return nil }
        return max(0, now - oldest.startedAt)
    }

    /// Whether the socket should be considered wedged.
    ///
    /// Requires the window to be **full** as well as old, deliberately. An
    /// aged reservation on a window that still has room is a slow send on a
    /// connection that is nonetheless admitting frames, and tearing that
    /// connection down would cost more throughput than it recovered. Only a
    /// window that is both full and not draining is actually blocking the
    /// pipeline.
    ///
    /// The two conditions almost always coincide in practice: frames go onto a
    /// single TCP stream in order, so an oldest send that has not flushed means
    /// none behind it have either, and the window fills within a few frames.
    /// Requiring both is therefore close to free — and it is what keeps this
    /// correct given that Apple documents completion ordering only for
    /// `sendPing`, never for `send`. Nothing here assumes the oldest
    /// reservation is the first to be released; `oldestAge(at:)` searches for
    /// it.
    nonisolated func isStalled(at now: TimeInterval) -> Bool {
        guard isFull, let age = oldestAge(at: now) else { return false }
        return age >= stallTimeout
    }

    // MARK: Lifecycle

    /// Abandons every outstanding reservation, reopening the whole window.
    ///
    /// Called when the socket those sends belonged to is gone. Their
    /// completion handlers may still run afterwards; because `nextToken` is
    /// not rewound, those late `release(_:at:)` calls return `nil` and are
    /// accounted as abandoned rather than silently crediting a slot on the
    /// *next* connection.
    nonisolated mutating func reset() {
        outstanding.removeAll(keepingCapacity: true)
    }
}
