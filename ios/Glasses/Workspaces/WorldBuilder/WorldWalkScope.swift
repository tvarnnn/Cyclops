//
//  WorldWalkScope.swift
//  Glasses
//
//  A walk's state and the walk it describes travel as ONE value (U-INLINE,
//  review 4). Three review rounds each found another surface that paired
//  one walk's picture with another walk's words or progress, and all of
//  them had one cause: the walk's identity and its state were published
//  separately -- the identity ahead of the state -- so any reader could see
//  a new identity with old words, or old words under a new identity.
//
//  So nothing publishes a walk apart from what it says about that walk.
//  The client publishes `WalkScoped<WorldWalkReport>` as one value; the view
//  model republishes it as one value; every surface that puts words or
//  progress beside a picture asks `WalkScoped.value(for:)` -- the one place
//  a picture is paired with words -- and shows nothing of a report whose
//  walk is not the picture's.
//

import Foundation

/// A value together with the walk it describes: the world AND the session,
/// or `nil` when it describes no walk that can be named (no snapshot, or a
/// report that named either id not). Built and published as one value, so
/// the two can never be read apart.
nonisolated struct WalkScoped<Value> {
    let walk: WorldFinishWalk?
    let value: Value

    init(walk: WorldFinishWalk?, value: Value) {
        self.walk = walk
        self.value = value
    }

    /// The same walk, saying something derived from what it said.
    func map<T>(_ transform: (Value) -> T) -> WalkScoped<T> {
        WalkScoped<T>(walk: walk, value: transform(value))
    }

    /// THE pairing rule. `true` only when `picture` shows the walk this value
    /// describes -- the same world AND the same session. A picture that
    /// names no session cannot be proved any walk's, and a value that names
    /// no walk describes no picture.
    func describes(_ picture: WorldRenderTarget?) -> Bool {
        guard let walk, let picture, let shown = WorldFinishWalk(picture: picture) else { return false }
        return shown == walk
    }

    /// The value, for a picture of its own walk; `nil` for any other picture.
    /// Every surface that pairs words or progress with a picture asks this.
    func value(for picture: WorldRenderTarget?) -> Value? {
        describes(picture) ? value : nil
    }

    /// `picture`, when this value describes the walk it shows; `nil` otherwise.
    func picture(_ picture: WorldRenderTarget?) -> WorldRenderTarget? {
        describes(picture) ? picture : nil
    }
}

extension WalkScoped: Equatable where Value: Equatable {}
extension WalkScoped: Sendable where Value: Sendable {}

extension WorldFinishWalk {
    /// The walk a picture shows, or `nil` for a picture naming no session.
    init?(picture: WorldRenderTarget) {
        guard let sessionID = picture.sessionID else { return nil }
        self.init(worldID: picture.worldID, sessionID: sessionID)
    }
}

/// What the Tower last said about the presented walk: everything the
/// screen's words and progress are drawn from. Published only inside a
/// `WalkScoped`, beside the walk it describes.
nonisolated struct WorldWalkReport: Equatable, Sendable {
    /// Already gated (`WorldSessionGate`).
    var state: WorldModelState
    var finalization: WorldFinalizationReport?
    var photographic: WorldPhotographicReport?
    /// `lifecycle.processing` (U0.6).
    var processing: WorldProcessingReport?
    /// The followed walk's stop clock and away banner (U0.6 §5.3).
    var finishClock: WorldFinishClock

    init(state: WorldModelState, finalization: WorldFinalizationReport? = nil,
         photographic: WorldPhotographicReport? = nil, processing: WorldProcessingReport? = nil,
         finishClock: WorldFinishClock = .unknown) {
        self.state = state
        self.finalization = finalization
        self.photographic = photographic
        self.processing = processing
        self.finishClock = finishClock
    }
}

extension WalkScoped where Value == WorldWalkReport {
    /// No report, no walk.
    static func unreported(_ state: WorldModelState) -> Self {
        WalkScoped(walk: nil, value: WorldWalkReport(state: state))
    }
}
