//
//  FrameRateGate.swift
//  Glasses
//

import Foundation

/// Monotonic time source for measuring intervals and durations.
///
/// Backed by `DispatchTime` (`mach_absolute_time`), so a read costs a handful
/// of nanoseconds and is unaffected by wall-clock adjustments — safe to call
/// on every camera frame. It does not advance while the device is asleep,
/// which is the behaviour we want here: a suspended app is not capturing
/// frames, so that time must not count against a session's measured rate.
///
/// Explicitly `nonisolated`: the project builds with
/// `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, which would otherwise make a
/// pure clock read main-actor-only — unusable from a default argument, and
/// unusable from DAT's callback thread if the rate gate is ever moved ahead of
/// the main-actor hop.
enum MonotonicClock {
    nonisolated static var now: TimeInterval {
        TimeInterval(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000
    }
}

/// Decides which frames of an incoming stream to forward, targeting a fixed
/// delivered frame rate.
///
/// A rate rather than a stride ("every Nth frame"). A fixed 1-in-30 stride
/// throttles to roughly 1 Hz at 24 fps, which capped delivery at
/// 24/30 ≈ 0.8 fps; and a stride silently multiplies any change in the source
/// rate, whereas the pipeline's requirement is expressed in delivered frames
/// per second and holds whether DAT produces 15, 24 or 30 fps.
///
/// Selection is deadline-based rather than "has `interval` elapsed since the
/// last selection". The latter aliases badly: at a 24 fps source and a 12 fps
/// target the elapsed-time boundary lands exactly on a frame arrival, so a
/// fraction of a millisecond of jitter rejects every other candidate and
/// halves the delivered rate. Here the deadline advances by whole intervals
/// and a frame is admitted once it is within `tolerance` of that deadline,
/// which keeps the mean on target under jitter.
///
/// A pure value type taking the current time as a parameter, so the cadence
/// is unit testable without a camera, a timer, or real elapsed time.
struct FrameRateGate {
    /// Frames per second this gate aims to admit. A gate can only ever
    /// *reject* frames, so when the source is slower than the target every
    /// frame is admitted and delivery equals the source rate.
    let targetFPS: Double

    /// How early a frame may be admitted relative to its deadline, as a
    /// fraction of one interval. This absorbs source-clock jitter and the
    /// exact-boundary case above. At 0.25 the worst-case *instantaneous*
    /// spacing is 0.75 intervals, while the mean rate stays on target because
    /// the deadline accumulates rather than resetting to "now".
    nonisolated static let tolerance = 0.25

    /// Rate at which captured DAT frames are forwarded to the Tower.
    ///
    /// `configuredTowerTargetFPS` resolves it once per process: the launch
    /// environment's `IOS_TOWER_TARGET_FPS`, else the build setting of the same
    /// name (through the Info.plist key), else `defaultTowerTargetFPS`.
    ///
    /// The invariant: every reader sees this one value. The gate's selection
    /// rate, the sender's send-window capacity
    /// (`TowerClient.defaultMaxFramesInFlight`), the IMU log's header and the
    /// Home text must agree, which is why nothing re-reads the setting.
    ///
    /// The default, 12, is a step up from the 0.8 fps a 1-in-30 stride
    /// delivered, not a ceiling. docs/03-ROADMAP.md V0.7 targets ~15 fps, and
    /// the gate reaches it exactly — the deadline accumulates, so a target need
    /// not divide the source rate (see
    /// `FrameRateGateTests.testTargetIsHeldAtTheRoadmapFifteenFPS`).
    ///
    /// The cost of a higher rate is on the main actor and the link. Per
    /// selected frame the main thread runs `makeUIImage`, a JPEG encode,
    /// base64, JSON serialisation and a viewfinder render: an *estimated*
    /// 15–25 ms, roughly 20–30% of the main thread at 12 fps and uncomfortable
    /// at 15. Bandwidth is an *estimated* ~2 Mbit/s sustained at 12, ~2.5 at 15,
    /// over a remote Tailscale path. Neither figure is measured; the developer
    /// surface's `Encode ms` and `Backlog` rows measure them, and 15 needs a
    /// physical run showing the headroom first.
    static let towerTargetFPS: Double = configuredTowerTargetFPS(
        environment: ProcessInfo.processInfo.environment, info: Bundle.main.infoDictionary ?? [:])

    /// The rate unless a build or launch says otherwise.
    nonisolated static let defaultTowerTargetFPS: Double = 12
    /// What `IOS_TOWER_TARGET_FPS` may set: a whole number of frames per
    /// second in this range. Anything else -- empty, unset, out of range or
    /// not a number -- keeps the default, so a default build is unchanged.
    nonisolated static let towerTargetFPSRange = 6...24
    nonisolated static let towerTargetFPSVariable = "IOS_TOWER_TARGET_FPS"
    nonisolated static let towerTargetFPSInfoKey = "GlassesTowerTargetFPS"

    /// The launch environment's value when it is valid, else the Info.plist
    /// key's (`$(IOS_TOWER_TARGET_FPS)` at build time), else 12.
    nonisolated static func configuredTowerTargetFPS(environment: [String: String] = [:],
                                                     info: [String: Any] = [:]) -> Double {
        for raw in [environment[towerTargetFPSVariable], info[towerTargetFPSInfoKey] as? String] {
            if let raw, let fps = Int(raw), towerTargetFPSRange.contains(fps) { return Double(fps) }
        }
        return defaultTowerTargetFPS
    }

    /// The time the next frame becomes eligible. `nil` before the first
    /// frame of a session, which is always admitted.
    private var nextDeadline: TimeInterval?

    nonisolated init(targetFPS: Double) {
        // Clamped so a zero or negative configuration cannot produce an
        // infinite interval, which would admit exactly one frame and then
        // stall the pipeline permanently.
        self.targetFPS = max(0.1, targetFPS)
    }

    nonisolated var interval: TimeInterval { 1.0 / targetFPS }

    /// Whether the frame arriving at `now` should be forwarded. Call exactly
    /// once per candidate frame, in arrival order.
    nonisolated mutating func shouldSelect(at now: TimeInterval) -> Bool {
        guard let deadline = nextDeadline else {
            // The first frame of a session is always admitted, so the
            // viewfinder and the Tower both get something immediately rather
            // than after one interval of nothing.
            nextDeadline = now + interval
            return true
        }

        guard now >= deadline - interval * Self.tolerance else { return false }

        // Advance from the previous deadline, not from `now`, so an early or
        // late frame does not permanently shift the schedule and the mean
        // rate stays on target.
        var advanced = deadline + interval
        if advanced <= now {
            // More than a full interval behind: a stall, a backgrounded app,
            // or a source slower than the target. Re-anchor to now instead of
            // carrying the debt forward, otherwise the gate would admit a
            // burst of catch-up frames — the opposite of what a
            // freshness-first pipeline wants.
            advanced = now + interval
        }
        nextDeadline = advanced
        return true
    }

    /// Forgets the schedule, so the next frame is treated as the first of a
    /// new session.
    nonisolated mutating func reset() {
        nextDeadline = nil
    }
}
