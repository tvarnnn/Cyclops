//
//  CartridgeResultSubscription.swift
//  Glasses
//

import Foundation

/// One cartridge's subscription on the Tower's result channel, as its client
/// tracks it on the current socket.
///
/// Document Memory, the CV Lab and Scene Understanding keep exactly this
/// bookkeeping, so it lives here once. Each client still decides when to
/// subscribe and what its own state does around these steps. World Builder
/// keeps its own: it pins subscriptions to a world, retires the ids it leaves
/// and bounds the wait for an ack, none of which fits here.
struct CartridgeResultSubscription {
    /// The open subscription, or `nil`. The Tower's ids are per connection and
    /// restart at `sub-1` on every socket, so nothing keeps one across a drop.
    var id: String?

    /// A `result_subscribe` has been sent and not yet answered. Without this a
    /// declaration republished while the ack is in flight would open a second
    /// subscription for the same cartridge.
    ///
    /// Cleared by the ack and by **any** error of ours, not only by the two
    /// that close a subscription. The Tower answers a `result_subscribe` with
    /// either an ack or an error, so once an error of ours arrives nothing is
    /// in flight. Every subscribe is guarded on this flag, so leaving it set
    /// after `snapshot_failed` (sent instead of the ack when the first snapshot
    /// raises), `contract_mismatch`, `unknown_cartridge`,
    /// `cartridge_unavailable` or `too_many_subscriptions` would stop the
    /// client subscribing for the life of the connection.
    var isSubscribing = false

    /// Resubscribes spent on this connection. See `resubscribeBudget`.
    private var resubscribesUsed = 0

    /// How many times one connection may reopen a subscription the Tower
    /// closed.
    ///
    /// `channel_failed` and `consumer_too_slow` are the two errors the Tower
    /// sends unsolicited, after which the subscription is gone and a new
    /// `result_subscribe` is the only way to resume; the Tower's message says
    /// so. Nothing else would send one: a client subscribes from its
    /// declaration sink and on going `.online`, and neither fires on a
    /// mid-connection error. `consumer_too_slow` is routine: it fires when the
    /// phone does not accept a result inside the send timeout, which a
    /// backgrounded or thermally throttled phone on a walk does.
    ///
    /// Bounded, and equal to `TowerWorldBuilderClient`'s budget: a Tower that
    /// closes a subscription three times on one connection will not be fixed
    /// by a fourth attempt, and unbounded retries against the Tower's
    /// 8-subscription cap starve the other cartridges.
    static let resubscribeBudget = 3

    /// The socket left `.online`, and the subscription went with it. Nothing
    /// is sent to close it: the Tower treats a closed socket as sufficient
    /// cleanup.
    ///
    /// The next connection gets a fresh budget. The cap stops a hopeless retry
    /// loop within one socket; it does not remember a bad socket forever.
    mutating func connectionLost() {
        id = nil
        isSubscribing = false
        resubscribesUsed = 0
    }

    /// The Tower closed the subscription (`CartridgeResultError.closesSubscription`).
    ///
    /// Clears it, then either spends one resubscribe and returns `nil`, in
    /// which case the caller subscribes again, or, with the budget spent,
    /// returns the failure the caller shows instead.
    mutating func closedByTower() -> CartridgeFailure? {
        id = nil
        isSubscribing = false
        guard resubscribesUsed < Self.resubscribeBudget else {
            return CartridgeFailure(
                kind: .transport,
                message: """
                    The Tower closed this cartridge's result subscription \
                    \(Self.resubscribeBudget) times on one connection. \
                    Reconnecting is what resolves it.
                    """
            )
        }
        resubscribesUsed += 1
        return nil
    }
}
