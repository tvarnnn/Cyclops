//
//  WorldFinishProgress.swift
//  Glasses
//
//  U0.6 + T-UX1 (WORLD-BUILDER-IOS.md §3c; WORLDS §2b, §4a rule 8): what the
//  phone says during the wait after Stop. One stage line under the stage
//  word, one elapsed line on the phone's own clock, the PREVIEW label on
//  every picture of an unsettled world, and the foreground banner. No
//  percentage, no estimate, no duration other than the time since Stop.
//
//  Everything here is a value or a pure function, `nonisolated`, with no
//  UIKit, so each rule is table-tested without a screen.
//

import Foundation

// MARK: - `lifecycle.processing` (WORLDS §2b)

/// `lifecycle.processing.stage`. Open: a word this build has not heard of
/// survives as itself and is not one of the six -- it is "not said", never
/// "finished".
nonisolated struct WorldProcessingStage: RawRepresentable, Equatable, Sendable, Hashable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    static let waiting = Self(rawValue: "waiting")
    static let preparing = Self(rawValue: "preparing")
    static let matching = Self(rawValue: "matching")
    static let placing = Self(rawValue: "placing")
    static let checking = Self(rawValue: "checking")
    static let assembling = Self(rawValue: "assembling")
    static let known: [Self] = [.waiting, .preparing, .matching, .placing, .checking, .assembling]

    var isKnown: Bool { Self.known.contains(self) }
}

/// Which consensus pass `checking` is on: `1 <= n <= of <= 7`.
nonisolated struct WorldProcessingStep: Equatable, Sendable {
    let n: Int
    let of: Int
}

nonisolated struct WorldProcessingReport: Equatable, Sendable {
    let stage: WorldProcessingStage
    /// Only on `checking`, only when `1 <= n <= of <= 7`; otherwise `nil`.
    let step: WorldProcessingStep?

    /// The most passes a consensus may ask for (`config.WORLD_SOLVE_CONSENSUS_VALUES`).
    static let maxPasses = 7

    init(stage: WorldProcessingStage, step: WorldProcessingStep? = nil) {
        self.stage = stage
        self.step = stage == .checking ? step.flatMap(Self.valid) : nil
    }

    /// `nil` for `null`, an absent key, a non-object, or an object with no
    /// non-empty string `stage`. A `step` that is not an object of two
    /// integers (not booleans) in range is dropped and the stage kept.
    init?(json: Any?) {
        guard let object = json as? [String: Any],
              let word = object["stage"] as? String, !word.isEmpty
        else { return nil }
        var step: WorldProcessingStep?
        if let raw = object["step"] as? [String: Any],
           let n = Self.integer(raw["n"]), let of = Self.integer(raw["of"]) {
            step = WorldProcessingStep(n: n, of: of)
        }
        self.init(stage: WorldProcessingStage(rawValue: word), step: step)
    }

    private static func valid(_ step: WorldProcessingStep) -> WorldProcessingStep? {
        (1...maxPasses).contains(step.of) && (1...step.of).contains(step.n) ? step : nil
    }

    /// A JSON integer: an `NSNumber` that is not a boolean and has no
    /// fraction (`JSONSerialization` bridges `true` to an `NSNumber` too).
    static func integer(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID()
        else { return nil }
        let double = number.doubleValue
        guard double.rounded() == double, abs(double) < 1e9 else { return nil }
        return number.intValue
    }
}

// MARK: - The stage line (IOS §3c)

nonisolated enum WorldFinishLine: Equatable, Sendable {
    case processing(WorldProcessingStage, step: WorldProcessingStep?)
    case photographic(stage: String?)
    /// The fallback: the final pass is pending and a live process holds the
    /// world (a Tower with the switch off, an older Tower, or a stage the
    /// Tower could not prove).
    case finalPlacement

    var text: String {
        switch self {
        case .processing(let stage, let step):
            switch stage {
            case .waiting: return "Finishing the last live update"
            case .preparing: return "Preparing images"
            case .matching: return "Matching images"
            case .placing: return "Placing images"
            case .checking:
                guard let step else { return "Checking the placement" }
                return "Checking the placement · pass \(step.n) of \(step.of)"
            case .assembling: return "Assembling the world"
            default: return WorldFinishLine.finalPlacement.text
            }
        case .photographic(let stage):
            switch stage {
            case "surface": return "Building surfaces"
            case "appearance": return "Adding photos"
            default: return "Building the photographic version"
            }
        case .finalPlacement:
            return "Placing and checking images"
        }
    }

    /// What VoiceOver says: the visible words, with the pass read as a
    /// clause rather than a middle dot.
    var spoken: String {
        if case .processing(.checking, let step?) = self {
            return "Checking the placement, pass \(step.n) of \(step.of)"
        }
        return text
    }

    /// IOS §3c's precedence, first match wins. Pure; table-tested (I3).
    static func line(isFinalizing: Bool, buildInProgress: Bool?, finalSolve: WorldFinalSolve,
                     photographic: WorldPhotographicReport?, processing: WorldProcessingReport?) -> WorldFinishLine? {
        guard isFinalizing else { return nil }
        let standing = photographic?.standing
        if standing == .areaStillFinishing { return nil }
        if let processing, processing.stage.isKnown {
            return .processing(processing.stage, step: processing.stage == .checking ? processing.step : nil)
        }
        if standing == .building { return .photographic(stage: photographic?.stage) }
        if let standing, standing.isUnfinished || standing.isFailed { return nil }
        if finalSolve == .pending, buildInProgress == true { return .finalPlacement }
        return nil
    }
}

/// The finish copy that is not a stage line (U0.6 §5.8; U-INLINE §2.3).
nonisolated enum WorldFinishCopy {
    /// Today's sentence beside the spinner, moved here so the panel and the
    /// canvas cannot word it differently.
    static let finishing = "The Tower is finishing this world."

    static func awayBanner(headline: String) -> String {
        "While you were away, your walk finished: \(headline)."
    }

    /// The one announcement when the panel's world becomes ready.
    static func finished(headline: String) -> String {
        "Your walk finished: \(headline)."
    }

    /// The finish block as one spoken element.
    static func spoken(showsSpinner: Bool, line: WorldFinishLine?, elapsed: WorldElapsedText?) -> String {
        var parts: [String] = []
        if showsSpinner { parts.append(finishing) }
        if let line { parts.append("Now: " + lowercasedFirst(line.spoken) + ".") }
        if let elapsed { parts.append(elapsed.spoken + ".") }
        return parts.joined(separator: " ")
    }

    /// `Placing images` → `placing images`, only for a capital followed by
    /// a lower-case letter.
    static func lowercasedFirst(_ text: String) -> String {
        guard let first = text.first, first.isUppercase,
              let second = text.dropFirst().first, second.isLowercase
        else { return text }
        return first.lowercased() + text.dropFirst()
    }
}

// MARK: - The PREVIEW (WORLDS §4a rule 8)

nonisolated enum WorldPreviewCopy {
    static let prefix = "Preview — "
    static let openPreview = "Open the preview"
    static let building = "Preview — your final world is still building. It will look very different, "
        + "and it is worth waiting for Saved."
    static let finalComing = "Your final world is ready. If this picture does not change, tap Reload."
    static let finalShown = "This is your final world."
    /// A settled world whose picture was built during the walk.
    static let walkTimePicture = "This picture was made during the walk, not from the final pass."
    /// The photographic preview after Stop, until the final manifest
    /// arrives (lead override 2026-10-05: the walk's photos stay served
    /// across Stop, so the preview looks like the room).
    static let walkTimePreview = "This picture was made during the walk."

    /// `"Preview — "` + `sentence`, its first letter lower-cased only when it
    /// is a capital followed by a lower-case letter (*This…* → *this…*, *The
    /// Tower…* → *the Tower…*; *TOWER…* stays).
    static func labelled(_ sentence: String) -> String {
        prefix + WorldFinishCopy.lowercasedFirst(sentence)
    }

    /// The line under the viewer's note about what is on screen (IOS §3c).
    /// Pure; table-tested (I10).
    static func settleLine(isPreview: Bool, walkEnded: Bool, openedOnPreview: Bool,
                           basisOnScreen: WorldPictureBasisOnScreen, noteSaysWalkTime: Bool) -> String? {
        if isPreview {
            // The preview note says what the world is; this says what the
            // picture is, once the Tower's own manifest has said it.
            return walkEnded && basisOnScreen == .walk ? walkTimePreview : nil
        }
        if basisOnScreen == .walk { return noteSaysWalkTime ? nil : walkTimePicture }
        guard openedOnPreview else { return nil }
        return basisOnScreen == .final ? finalShown : finalComing
    }
}

/// What a picture was built from (WORLDS §4a `basis`). Open, like the
/// stage word: anything else decides nothing.
nonisolated struct WorldPictureBasis: RawRepresentable, Equatable, Sendable, Hashable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }
    static let final = Self(rawValue: "final")
    static let walk = Self(rawValue: "walk")

    /// From an appearance manifest's bytes (APPEARANCE §7): `quality: "live"`,
    /// or `proxy.source.surface_quality: "live"` → `.walk`; `quality: "final"`
    /// over a final (or unnamed) surface → `.final`; anything else → `nil`.
    static func ofAppearanceManifest(_ data: Data) -> Self? {
        guard let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        let quality = json["quality"] as? String
        let source = (json["proxy"] as? [String: Any])?["source"] as? [String: Any]
        let surfaceQuality = source?["surface_quality"] as? String
        if quality == "live" || surfaceQuality == "live" { return .walk }
        if quality == "final", surfaceQuality == nil || surfaceQuality == "final" { return .final }
        return nil
    }
}

/// What the picture on screen is known to be built from.
nonisolated enum WorldPictureBasisOnScreen: Equatable, Sendable {
    case unknown, walk, final
    /// The Tower says the final pass is served, and the page on screen was
    /// fetched before it said so: a Reload (or the page itself) brings it.
    case finalArriving
}

/// What the 3D viewer is told about the wait, from the workspace's
/// presentation.
nonisolated struct WorldViewerProgress: Equatable, Sendable {
    var isPreview: Bool
    /// The walk has stopped (Improving, Finalizing): the preview is a
    /// picture made during a walk that is over.
    var walkEnded: Bool = false
    var line: WorldFinishLine?
    var stoppedAt: ContinuousClock.Instant?
    /// The ladder's note already says the picture is walk-time (Partial).
    var noteSaysWalkTime: Bool
}

// MARK: - The elapsed time (U0.6 §2.5)

/// "12 min since you stopped": whole minutes, floored, on the phone's
/// monotonic clock. Never an estimate, never "of".
nonisolated struct WorldElapsedText: Equatable, Sendable {
    let visible: String
    let spoken: String

    init(since start: ContinuousClock.Instant, now: ContinuousClock.Instant) {
        self.init(seconds: Self.seconds(now - start))
    }

    init(seconds: Int) {
        let total = max(0, seconds)
        let minutes = total / 60
        if minutes < 1 {
            visible = "Under 1 min since you stopped"
            spoken = "Less than a minute since you stopped"
        } else if minutes < 60 {
            visible = "\(minutes) min since you stopped"
            spoken = "\(Self.count(minutes, "minute")) since you stopped"
        } else {
            let hours = minutes / 60, rest = minutes % 60
            visible = rest == 0 ? "\(hours) h since you stopped" : "\(hours) h \(rest) min since you stopped"
            spoken = rest == 0
                ? "\(Self.count(hours, "hour")) since you stopped"
                : "\(Self.count(hours, "hour")) \(Self.count(rest, "minute")) since you stopped"
        }
    }

    private static func count(_ n: Int, _ unit: String) -> String { n == 1 ? "1 \(unit)" : "\(n) \(unit)s" }

    static func seconds(_ duration: Duration) -> Int {
        Int(duration.components.seconds)
    }
}

// MARK: - The stop clock and the banner (U0.6 §5.3)

nonisolated struct WorldFinishWalk: Equatable, Sendable, Hashable {
    let worldID: String
    let sessionID: String
}

/// What the screen may say about the wait. `.unknown` for every client with
/// no Tower, and after a relaunch (memory only).
nonisolated struct WorldFinishClock: Equatable, Sendable {
    var stoppedAt: ContinuousClock.Instant?
    var showsAwayBanner: Bool
    /// The banner shows and has not been announced for this walk yet.
    var announcesAwayBanner = false
    static let unknown = WorldFinishClock(stoppedAt: nil, showsAwayBanner: false)
}

/// The walk this phone followed live: when it stopped, and whether it
/// settled while the app was away. Owned by the client, which outlives the
/// workspace, so the clock survives a cartridge switch; never persisted.
nonisolated struct WorldFinishWatch: Equatable, Sendable {
    enum Standing: Equatable, Sendable { case receiving, finishing, settled, other }

    struct Stop: Equatable, Sendable {
        let walk: WorldFinishWalk
        let at: ContinuousClock.Instant
    }

    private(set) var receivingWalk: WorldFinishWalk?
    private(set) var stopped: Stop?
    private(set) var awayWhileFinishing: WorldFinishWalk?
    private(set) var bannerFor: WorldFinishWalk?
    /// The walks whose banner VoiceOver has announced: once per walk,
    /// whether or not the banner is still on screen -- and after another
    /// walk's, so A, B, A never announces A twice (review 3).
    private(set) var announced: Set<WorldFinishWalk> = []

    init() {}

    /// A report the Tower just sent, as presented. `following`: unpinned.
    mutating func report(walk: WorldFinishWalk?, following: Bool, standing: Standing,
                         appActive: Bool, now: ContinuousClock.Instant) {
        guard let walk, standing != .other else { return }
        if following, standing == .receiving {
            receivingWalk = walk
            if stopped?.walk == walk { stopped = nil }
            if let banner = bannerFor, banner != walk { bannerFor = nil }
        }
        if following, standing == .finishing || standing == .settled,
           receivingWalk == walk, stopped?.walk != walk {
            stopped = Stop(walk: walk, at: now)
        }
        if standing == .settled, awayWhileFinishing == walk {
            bannerFor = walk
            awayWhileFinishing = nil
        }
        if standing == .receiving || standing == .finishing, appActive, awayWhileFinishing == walk {
            awayWhileFinishing = nil
        }
    }

    /// The app went to the background (`false`) or became active (`true`).
    /// `currentWalk` is the walk the screen follows live (`nil` while a
    /// saved world is pinned: the banner is for the walk this phone followed).
    mutating func app(active: Bool, currentWalk: WorldFinishWalk?, currentStanding: Standing) {
        guard !active, let currentWalk,
              currentStanding == .receiving || currentStanding == .finishing
        else { return }
        awayWhileFinishing = currentWalk
    }

    mutating func dismissBanner() { bannerFor = nil }

    /// The banner for `walk` was announced: never again for that walk.
    mutating func bannerAnnounced(for walk: WorldFinishWalk?) {
        guard let walk, bannerFor == walk else { return }
        announced.insert(walk)
    }

    func clock(for walk: WorldFinishWalk?) -> WorldFinishClock {
        guard let walk else { return .unknown }
        return WorldFinishClock(stoppedAt: stopped?.walk == walk ? stopped?.at : nil,
                                showsAwayBanner: bannerFor == walk,
                                announcesAwayBanner: bannerFor == walk && !announced.contains(walk))
    }
}
