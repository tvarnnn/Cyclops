//
//  CaptureHealthModel.swift
//  Glasses
//
//  U2-D0 "Capture health": the operator's five rows during a walk -- the
//  Tower link, keyframes per minute (and "stalled"), tracking breaks in the
//  last 30 s, the look-back relocalizer with its counts, and how far the map
//  is behind. Zero Tower change: every input is already on the wire
//  (`world_snapshot.keyframe_count`, the payload's `trajectory`,
//  `tracking.recovery`, `geometry`), and the rates are differenced here, on
//  the phone's own clock.
//
//  Honest about absence and age: a missing input reads "—", and so does
//  every figure once the Tower link is down, the last report is older than
//  `staleAfter`, or the subscription that sent it is gone -- a number is
//  never shown as live when it is not, and another walk's never as this one's.
//

import Combine
import Foundation

// MARK: - One report's inputs

/// `geometry.built_from_keyframes` / `geometry.keyframes_now`
/// (`CARTRIDGE-RESULTS.md`, `geometry`): how many keyframes the map was built
/// from, and how many exist now.
nonisolated struct WorldMapLag: Equatable, Sendable {
    let builtFromKeyframes: Int
    let keyframesNow: Int

    /// Keyframes not yet in the map, never negative.
    var behind: Int { max(0, keyframesNow - builtFromKeyframes) }

    /// `nil` unless both are integers: a build that has not happened sends
    /// `null` for the first, which is "unknown", not 0.
    init?(geometry: Any?) {
        guard
            let geometry = geometry as? [String: Any],
            let built = geometry["built_from_keyframes"] as? Int,
            let now = geometry["keyframes_now"] as? Int
        else { return nil }
        self.init(builtFromKeyframes: built, keyframesNow: now)
    }

    init(builtFromKeyframes: Int, keyframesNow: Int) {
        self.builtFromKeyframes = builtFromKeyframes
        self.keyframesNow = keyframesNow
    }
}

/// What one World Builder report says about the live walk, for the panel.
/// Every field `nil` when the report says nothing about a live walk this
/// phone follows (`empty`): a pinned saved world, a state that is not
/// `receiving`, a report the session gate held back, an unreadable one.
nonisolated struct CaptureHealthSample: Equatable, Sendable {
    var keyframeCount: Int?
    var trackingRestarts: Int?
    var recovery: WorldRecoveryReport?
    var mapLag: WorldMapLag?
    /// The walk these figures are about: `world_snapshot.world_id` and
    /// `session.session_id`, as reported. The counters are differenced only
    /// within one walk; another walk's counters are another series, whether
    /// or not they happen to be lower.
    var worldID: String?
    var sessionID: String?

    static let empty = CaptureHealthSample()

    /// The live walk's figures from a report whose presented `state` is
    /// `receiving`; `empty` for every other state.
    static func live(state: WorldModelState, recovery: WorldRecoveryReport?, worldID: String?, sessionID: String?,
                     payload: [String: Any]) -> CaptureHealthSample {
        guard case .receiving(let snapshot) = state else { return .empty }
        return CaptureHealthSample(
            keyframeCount: snapshot.keyframeCount,
            trackingRestarts: snapshot.trajectory.trackingRestarts,
            recovery: recovery,
            mapLag: WorldMapLag(geometry: payload["geometry"]),
            worldID: worldID,
            sessionID: sessionID
        )
    }

    /// Whether the report named its walk: a world id AND a session id. A
    /// sample without both is no walk's, so none of its figures are shown.
    var hasIdentity: Bool {
        !(worldID ?? "").isEmpty && !(sessionID ?? "").isEmpty
    }

    /// Whether `other` is about the same walk: the same world and session,
    /// both named. Two missing ids are NOT the same walk -- reports from
    /// different walks would otherwise share pace, breaks and stall history
    /// until identity arrived (Codex HIGH, 2026-10-05).
    func isSameWalk(as other: CaptureHealthSample) -> Bool {
        hasIdentity && other.hasIdentity && worldID == other.worldID && sessionID == other.sessionID
    }
}

// MARK: - The rows, pure

/// What the panel shows, as text. The words are the phone's own.
nonisolated struct CaptureHealthReadout: Equatable, Sendable {
    static let unknown = "—"

    var link: String
    var pace: String
    /// "Stalled — no new keyframe for N s", while capturing and stalled.
    var stalled: String?
    var breaks: String
    /// The breaks row for VoiceOver: "seconds", spelled out.
    var breaksSpoken: String
    /// The relocalizer's line (the canvas's), when there is an episode.
    var lookBackLine: String?
    var lookBackCounts: String
    /// `nil` hides the row: no lag, or (with fresh figures) an unknown one.
    var mapLag: String?

    /// For the announcements: the breaks count and the stall, when known.
    var breaksCount: Int?
    var isStalled: Bool { stalled != nil }
}

/// The samples the panel has seen, and the arithmetic over them. A value, so
/// the tests drive it with scripted instants.
nonisolated struct CaptureHealthHistory: Equatable, Sendable {
    typealias Instant = ContinuousClock.Instant

    /// Keyframes per minute is the rise over this window, or over the time
    /// since the panel's first sample when that is shorter, per minute.
    static let paceWindow: Duration = .seconds(60)
    /// A rate needs this much time behind it: over less, one keyframe more
    /// or less swings it by tens a minute.
    static let minimumPaceSpan: Duration = .seconds(10)
    static let breaksWindow: Duration = .seconds(30)
    /// No new keyframe for this long, while capturing, is "stalled".
    static let stallAfter: Duration = .seconds(5)
    /// A report older than this is not live. The Tower re-sends an unchanged
    /// snapshot about every 2 s (`CARTRIDGE-RESULTS.md`, the heartbeat), so
    /// two missed heartbeats and a margin.
    static let staleAfter: Duration = .seconds(5)

    struct Point: Equatable, Sendable {
        let at: Instant
        let value: Int
    }

    private(set) var keyframes: [Point] = []
    private(set) var restarts: [Point] = []
    private(set) var lastKeyframeIncrease: Instant?
    private(set) var lastSampleAt: Instant?
    private(set) var last: CaptureHealthSample?

    init() {}

    /// What was held is no longer live -- the socket dropped, the
    /// subscription restarted, the screen pinned a saved world -- so it is
    /// forgotten at once, not 5 s later: every figure reads "—" until the next
    /// report, as for an old one.
    mutating func invalidate() {
        self = CaptureHealthHistory()
    }

    mutating func record(_ sample: CaptureHealthSample, at now: Instant) {
        // Until a report names both its world and its session, it says
        // nothing about a walk this panel can follow: every figure stays
        // "—", as for `empty`, and the walk that is named later starts its
        // histories from its own first sample (a nil → id change is a new
        // identity, by `isSameWalk`).
        let sample = sample.hasIdentity ? sample : .empty
        if let last, !sample.isSameWalk(as: last) {
            // Another walk, or none: nothing seen so far is about it -- not
            // its pace, not its breaks, not how long since its last keyframe.
            keyframes = []
            restarts = []
            lastKeyframeIncrease = nil
        }
        lastSampleAt = now
        last = sample
        if let count = sample.keyframeCount {
            if let previous = keyframes.last?.value, count < previous {
                // A counter that went down is another walk: start again.
                keyframes = []
            }
            if keyframes.isEmpty || count > keyframes.last!.value {
                lastKeyframeIncrease = now
            }
            Self.append(Point(at: now, value: count), to: &keyframes, window: Self.paceWindow)
        } else {
            keyframes = []
            lastKeyframeIncrease = nil
        }
        if let count = sample.trackingRestarts {
            if let previous = restarts.last?.value, count < previous { restarts = [] }
            Self.append(Point(at: now, value: count), to: &restarts, window: Self.breaksWindow)
        } else {
            restarts = []
        }
    }

    /// The five rows at `now`.
    func readout(now: Instant, isLinked: Bool, isCapturing: Bool) -> CaptureHealthReadout {
        let unknown = CaptureHealthReadout.unknown
        let link = isLinked ? "Tower: connected" : "Tower: not connected"
        guard isLinked, let lastSampleAt, let last, now - lastSampleAt <= Self.staleAfter else {
            return CaptureHealthReadout(
                link: link, pace: "\(unknown) keyframes/min", stalled: nil,
                breaks: "Breaks in the last 30 s: \(unknown)", breaksSpoken: "Breaks in the last 30 seconds: unknown",
                lookBackLine: nil, lookBackCounts: "Look-back: \(unknown)", mapLag: "Map: \(unknown)",
                breaksCount: nil
            )
        }

        let pace = Self.perMinute(keyframes, over: Self.paceWindow, now: now)
        var stalled: String?
        if isCapturing, !keyframes.isEmpty, let since = lastKeyframeIncrease, now - since >= Self.stallAfter {
            stalled = "Stalled — no new keyframe for \(Self.wholeSeconds(now - since)) s"
        }
        let breaks = Self.breaks(in: restarts, now: now)

        var lookBackLine: String?
        var lookBackCounts = "Look-back: \(unknown)"
        if let recovery = last.recovery {
            lookBackLine = recovery.displayLine
            if let counts = recovery.counts {
                lookBackCounts = "linked back \(counts.recovered) · could not link \(counts.timedOut)"
            } else {
                lookBackCounts = "linked back \(unknown) · could not link \(unknown)"
            }
        }

        var mapLag: String?
        if let behind = last.mapLag?.behind, behind > 0 {
            mapLag = behind == 1 ? "Map: 1 keyframe behind" : "Map: \(behind) keyframes behind"
        }

        return CaptureHealthReadout(
            link: link,
            pace: "\(pace.map(String.init) ?? unknown) keyframes/min",
            stalled: stalled,
            breaks: "Breaks in the last 30 s: \(breaks?.text ?? unknown)",
            breaksSpoken: "Breaks in the last 30 seconds: \(breaks?.spoken ?? "unknown")",
            lookBackLine: lookBackLine,
            lookBackCounts: lookBackCounts,
            mapLag: mapLag,
            breaksCount: breaks?.count
        )
    }

    // MARK: Arithmetic

    /// Keyframes per minute: the rise over the last `window`, or since the
    /// first sample when that is later, scaled to a minute. `nil` with no
    /// points, or with less than `minimumPaceSpan` behind it.
    static func perMinute(_ series: [Point], over window: Duration, now: Instant) -> Int? {
        guard let first = series.first, let newest = series.last else { return nil }
        let start = max(now - window, first.at)
        let span = now - start
        guard span >= minimumPaceSpan else { return nil }
        let base = series.last(where: { $0.at <= start }) ?? first
        let rise = Double(max(0, newest.value - base.value))
        return Int((rise * 60 / seconds(span)).rounded(.down))
    }

    /// The breaks in the last 30 s. Exact once the panel has watched the
    /// whole window. Before that a rise it saw is a floor ("≥ N": there may
    /// have been more before it started watching), and no rise is unknown,
    /// never 0.
    static func breaks(in series: [Point], now: Instant) -> (count: Int, text: String, spoken: String)? {
        guard let first = series.first, let rise = change(in: series, over: breaksWindow, now: now) else {
            return nil
        }
        if first.at <= now - breaksWindow { return (rise, "\(rise)", "\(rise)") }
        guard rise > 0 else { return nil }
        return (rise, "≥ \(rise)", "at least \(rise)")
    }

    static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }

    /// The counter's rise over the last `window`: the newest value minus the
    /// value it held at the window's start (the last point at or before it,
    /// or the first point when the series is younger than the window).
    /// `nil` with no points.
    static func change(in series: [Point], over window: Duration, now: Instant) -> Int? {
        guard let newest = series.last else { return nil }
        let start = now - window
        let base = series.last(where: { $0.at <= start }) ?? series[0]
        return max(0, newest.value - base.value)
    }

    /// Keeps every point inside the window and the last one before it, which
    /// is the window's base.
    private static func append(_ point: Point, to series: inout [Point], window: Duration) {
        series.append(point)
        let start = point.at - window
        if let firstInside = series.firstIndex(where: { $0.at > start }), firstInside > 1 {
            series.removeFirst(firstInside - 1)
        }
    }

    static func wholeSeconds(_ duration: Duration) -> Int {
        Int(duration.components.seconds)
    }
}

/// Whether a change of the readout is said aloud: only the Stalled and the
/// Breaks rows, and at most once every 10 s.
nonisolated struct CaptureHealthAnnouncer: Equatable, Sendable {
    static let minimumGap: Duration = .seconds(10)
    private(set) var lastAt: ContinuousClock.Instant?

    init() {}

    /// The sentence to announce for `old` → `new` at `now`, or `nil`.
    mutating func announcement(from old: CaptureHealthReadout, to new: CaptureHealthReadout,
                               at now: ContinuousClock.Instant) -> String? {
        var sentence: String?
        if new.isStalled, !old.isStalled, let stalled = new.stalled {
            sentence = stalled
        } else if let count = new.breaksCount, let before = old.breaksCount, count > before {
            sentence = new.breaksSpoken
        }
        guard let sentence else { return nil }
        if let lastAt, now - lastAt < Self.minimumGap { return nil }
        lastAt = now
        return sentence
    }
}

// MARK: - The model

/// Feeds the panel from the status stream the canvas already uses
/// (`WorldBuilderClient.healthSamples`): no new network call and no new poll.
@MainActor
final class CaptureHealthModel: ObservableObject {
    @Published private(set) var history = CaptureHealthHistory()
    private var cancellable: AnyCancellable?

    init(client: any WorldBuilderClient) {
        cancellable = client.healthSamples
            .receive(on: DispatchQueue.main)
            .sink { [weak self] sample in
                guard let self else { return }
                if let sample { history.record(sample, at: .now) } else { history.invalidate() }
            }
    }
}
