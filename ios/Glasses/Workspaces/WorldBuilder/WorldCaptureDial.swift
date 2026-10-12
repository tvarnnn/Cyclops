//
//  WorldCaptureDial.swift
//  Glasses
//
//  The FOW capture dial (FOW-RING-SPEC 2026-10-12): a ring of twelve
//  directions that fills as the Tower's landed solves give each direction
//  two or more views, one dial per piece, with the piece's own photos in
//  its filled segments. This file is the pure model and its words; the
//  drawing is `WorldCaptureDialView.swift`.
//
//  What the dial is, and is not. Its segment states are the OR of the
//  frozen coverage receipt's masks over one piece's DRAWN stations (§3.2):
//  `supported` (two or more views), `weak` (one view), or not yet captured.
//  It is never "the room", never complete, and never says where the wearer
//  is or faces. Its top is the newest placed view when the Tower says which
//  that is (`landed.latest`, S3), and otherwise the way the wearer faced
//  when the piece began (sector zero) -- never a compass. No clock, image
//  or view state lives here.
//

import Foundation

// MARK: - The switch

/// The phone's switch for the dial, `IOS_FOW_DIAL` (environment) or the
/// `GlassesFOWDial` Info.plist key a home-screen build fills from the same
/// build setting. `on` exactly; anything else, an empty setting included,
/// is off, and off is today's panel exactly: no toggle, no new element.
nonisolated struct WorldCaptureDialSwitch: Equatable, Sendable {
    static let variable = "IOS_FOW_DIAL"
    static let infoKey = "GlassesFOWDial"
    static let off = Self(on: false)

    let on: Bool

    init(on: Bool) { self.on = on }

    init(environment: [String: String], info: [String: Any] = [:]) {
        on = environment[Self.variable] == "on" || info[Self.infoKey] as? String == "on"
    }

    static var current: Self {
        Self(environment: ProcessInfo.processInfo.environment, info: Bundle.main.infoDictionary ?? [:])
    }
}

/// The two faces of the walking stage when the dial is on (§5).
nonisolated enum WorldPanelTab: String, CaseIterable, Sendable {
    case dial, map
}

/// What the panel needs for the dial beyond the map: `nil` is the switch
/// off. Both flags are already the walk's own (the view model's gates).
nonisolated struct WorldPanelDial: Equatable, Sendable {
    /// `guidance.coverage: null` for the walk being received: the Tower's
    /// switch is on and no landing has merged yet (§6 "Before the first
    /// landing").
    var awaiting = false
    /// The phone gave up on the current landing's photos (§3.4).
    var photosUnavailable = false
}

// MARK: - The model

nonisolated struct WorldCaptureDial: Equatable, Sendable {
    static let sectors = WorldCoverage.sectors

    /// One drawn component, read as twelve directions.
    struct Piece: Equatable, Sendable {
        /// Position in the receipt's `reference_segment` order: "Piece n".
        let index: Int
        let referenceSegment: Int
        let posedKeyframes: Int
        /// By sector, the OR over the piece's drawn stations (§3.2).
        let sectors: [WorldCoverageEvidence]

        var supportedCount: Int { sectors.filter { $0 == .supported }.count }
        var weakCount: Int { sectors.filter { $0 == .weak }.count }
        var name: String { WorldCoverageCopy.pieceName(index) }

        init(index: Int, component: WorldCoverage.Component) {
            self.index = index
            referenceSegment = component.referenceSegment
            posedKeyframes = component.posedKeyframes
            sectors = (0..<WorldCaptureDial.sectors).map { sector in
                let bit = 1 << sector
                if component.stations.contains(where: { $0.supportedMask & bit != 0 }) { return .supported }
                if component.stations.contains(where: { $0.weakMask & bit != 0 }) { return .weak }
                return .unconfirmed
            }
        }
    }

    /// One in-hand landed tile and the sector it belongs to: the Tower's
    /// own bin (S3) or the phone's join (S2).
    struct Photo: Equatable, Sendable {
        let digest: String
        let referenceSegment: Int
        let sector: Int
    }

    /// The longest clockwise run of one evidence kind, in DISPLAY positions
    /// (0 is 12 o'clock).
    struct Gap: Equatable, Sendable {
        let positions: [Int]
        let evidence: WorldCoverageEvidence
    }

    /// The newest placed view (S3's `landed.latest`): its piece and sector.
    struct Newest: Equatable, Sendable {
        let referenceSegment: Int
        let sector: Int
    }

    /// What the top of the dial means (§3.3).
    enum Top: Equatable, Sendable {
        /// `landed.latest.sector`: the newest placed view.
        case newestView
        /// Sector zero: the way the wearer faced when the piece began.
        case pieceStart
    }

    let pieces: [Piece]
    /// `nil` before the first landing (or for a receipt with no drawn piece).
    let heroIndex: Int?
    /// The sector drawn at 12 o'clock.
    let anchor: Int
    let top: Top
    /// `landed.motion_sector`, only beside the `newest` that names the hero.
    let motionSector: Int?
    let stationsOmitted: Int
    /// Hero sector -> digest; the last eligible photo in manifest order wins,
    /// and a photo is never drawn on a segment that is not yet captured.
    let photosBySector: [Int: String]

    /// Before the first landing: twelve dashed segments and "–".
    static let awaiting = Self(components: [], stationsOmitted: 0)

    init(coverage: WorldCoverage, newest: Newest? = nil, motionSector: Int? = nil, photos: [Photo] = []) {
        self.init(components: coverage.components, stationsOmitted: coverage.stationsOmitted,
                  newest: newest, motionSector: motionSector, photos: photos)
    }

    init(components: [WorldCoverage.Component], stationsOmitted: Int,
         newest: Newest? = nil, motionSector: Int? = nil, photos: [Photo] = []) {
        let pieces = components.enumerated().map { Piece(index: $0.offset, component: $0.element) }
        self.pieces = pieces
        self.stationsOmitted = stationsOmitted
        // §3.1: the piece the newest view names, else the one with the most
        // posed keyframes (ties: the first in reference order). A newest view
        // in a piece not drawn here anchors nothing.
        if let newest, (0..<Self.sectors).contains(newest.sector),
           let named = pieces.firstIndex(where: { $0.referenceSegment == newest.referenceSegment }) {
            heroIndex = named
            anchor = newest.sector
            top = .newestView
            self.motionSector = motionSector.flatMap { (0..<Self.sectors).contains($0) ? $0 : nil }
        } else {
            heroIndex = pieces.indices.max { pieces[$0].posedKeyframes < pieces[$1].posedKeyframes }
            anchor = 0
            top = .pieceStart
            self.motionSector = nil
        }
        var accepted: [Int: String] = [:]
        if let heroIndex {
            let hero = pieces[heroIndex]
            for photo in photos where photo.referenceSegment == hero.referenceSegment
                && (0..<Self.sectors).contains(photo.sector) && hero.sectors[photo.sector] != .unconfirmed {
                accepted[photo.sector] = photo.digest
            }
        }
        photosBySector = accepted
    }

    /// The overlay's in-hand tiles as dial photos: the Tower's `sector` when
    /// it sent one (S3), otherwise the joined heading's 30-degree bin (S2).
    /// A tile with neither has no direction and is left out.
    static func photos(_ entries: [WorldDialThumbnail]) -> [Photo] {
        entries.compactMap { entry in
            let sector = entry.sector ?? entry.heading.flatMap { heading -> Int? in
                guard heading.isFinite, heading >= 0 else { return nil }
                return Int(heading / 30) % sectors
            }
            return sector.map { Photo(digest: entry.digest, referenceSegment: entry.referenceSegment, sector: $0) }
        }
    }

    var hasReceipt: Bool { heroIndex != nil }

    var hero: Piece? { heroIndex.map { pieces[$0] } }

    /// The pieces drawn as mini dials, in reference order.
    var otherPieces: [Piece] { pieces.filter { $0.index != heroIndex } }

    func sector(at position: Int) -> Int { ((position + anchor) % Self.sectors + Self.sectors) % Self.sectors }

    func position(of sector: Int) -> Int { ((sector - anchor) % Self.sectors + Self.sectors) % Self.sectors }

    func evidence(sector: Int) -> WorldCoverageEvidence { hero?.sectors[sector] ?? .unconfirmed }

    func evidence(at position: Int) -> WorldCoverageEvidence { evidence(sector: sector(at: position)) }

    /// §3.5: the longest clockwise run of not-yet-captured segments, else of
    /// one-view segments; a tie goes to the run met first sweeping clockwise
    /// from 12 o'clock (a run through 12 o'clock is met at once). `nil` with
    /// all twelve captured, or before the first landing.
    var gap: Gap? {
        guard hasReceipt else { return nil }
        for kind in [WorldCoverageEvidence.unconfirmed, .weak] {
            let marked = (0..<Self.sectors).map { evidence(at: $0) == kind }
            guard marked.contains(true) else { continue }
            guard marked.contains(false) else { return Gap(positions: Array(0..<Self.sectors), evidence: kind) }
            var best: (run: [Int], met: Int)?
            for start in 0..<Self.sectors where marked[start] && !marked[(start + Self.sectors - 1) % Self.sectors] {
                var run: [Int] = []
                var position = start
                while marked[position] {
                    run.append(position)
                    position = (position + 1) % Self.sectors
                }
                let met = run.contains(0) ? -1 : start
                if let current = best, current.run.count > run.count
                    || (current.run.count == run.count && current.met <= met) { continue }
                best = (run, met)
            }
            return best.map { Gap(positions: $0.run, evidence: kind) }
        }
        return nil
    }

    /// The look-here emphasis: the gap, only while the map is fresh (§3.5,
    /// §4). Never for a stale or undated map.
    func emphasis(age: Double?) -> Gap? {
        WorldCaptureDialCopy.isFresh(age) ? gap : nil
    }
}

// MARK: - The words

/// Everything the dial says (§3, §4). Never "the room", never complete,
/// never a percentage or an instruction; nothing is voiced on its own.
nonisolated enum WorldCaptureDialCopy {
    static let dialName = "Capture dial"
    static let tabLabel = "Panel view"
    static let awaitingLine = "The dial starts filling when the Tower places your first views."
    static let photosUnavailable = "Photos unavailable for this update"
    static let ofTwelve = "of 12"
    static let captured = "captured"
    static let notYetNumber = "–"
    static let legendCaptured = "Captured"
    static let legendOneView = "One view"
    static let legendNotYet = "Not yet captured"
    static let allCaptured = "All 12 directions captured in this piece."
    static let topNewest = "your newest placed view"
    static let topPieceStart = "the way you faced when this piece began"

    static func tabName(_ tab: WorldPanelTab) -> String {
        switch tab {
        case .dial: return "Dial"
        case .map: return "Map"
        }
    }

    /// ≤ 30 s on the measured Tower clock (§8.2); an undated map is not fresh.
    static func isFresh(_ age: Double?) -> Bool {
        age.map { $0 <= WorldCoverageReceipt.staleAfter } ?? false
    }

    /// "12", "1" ... "11": the clock hour of a display position.
    static func hour(_ position: Int) -> Int { position == 0 ? 12 : position }

    static func clock(_ positions: [Int]) -> String {
        guard positions.count < WorldCaptureDial.sectors else { return "all 12 directions" }
        guard let first = positions.first, let last = positions.last else { return "" }
        return positions.count == 1 ? "\(hour(first)) o'clock" : "\(hour(first)) to \(hour(last)) o'clock"
    }

    static func reference(_ top: WorldCaptureDial.Top) -> String {
        top == .newestView ? topNewest : topPieceStart
    }

    /// §3.1: "Your newest views" / "Most views", "Piece n · …" with several.
    static func heroLabel(_ dial: WorldCaptureDial) -> String {
        guard let hero = dial.hero else { return dialName }
        let what = dial.top == .newestView ? "your newest views" : "most views"
        return dial.pieces.count > 1 ? "\(hero.name) · \(what)" : what.prefix(1).uppercased() + what.dropFirst()
    }

    /// The footer's orientation line (§3.3).
    static func topLine(_ dial: WorldCaptureDial) -> String {
        "Top of the dial: \(reference(dial.top))."
    }

    /// §3.5's gap line. Never imperative, never voiced on its own, never haptic.
    static func gapLine(_ dial: WorldCaptureDial, age: Double?) -> String {
        guard dial.hasReceipt else { return awaitingLine }
        guard let gap = dial.gap else { return allCaptured }
        let at = clock(gap.positions)
        let whole = gap.positions.count == WorldCaptureDial.sectors
        if gap.evidence == .weak {
            let place = whole ? "in \(at)" : "at \(at)"
            return gap.positions.count == 1
                ? "One view only \(place). Another view there would capture it."
                : "One view only \(place). Another view of each would capture them."
        }
        guard isFresh(age) else { return "Not yet captured when this map was made: \(at)." }
        if whole { return "More needed in \(at)." }
        let mostly = gap.positions.count >= 4 ? "mostly " : ""
        return "More needed at \(at), \(mostly)\(side(gap)) \(reference(dial.top))."
    }

    /// From the arc's midpoint hour: 10.5–1.5 ahead of, 1.5–4.5 right of,
    /// 4.5–7.5 behind, 7.5–10.5 left of (§3.5). A midpoint exactly on a
    /// boundary (a run of even length) goes to ahead of or behind.
    static func side(_ gap: WorldCaptureDial.Gap) -> String {
        let first = Double(gap.positions.first ?? 0)
        let mid = (first + Double(gap.positions.count - 1) / 2).truncatingRemainder(dividingBy: 12)
        if mid <= 1.5 || mid >= 10.5 { return "ahead of" }
        if (4.5...7.5).contains(mid) { return "behind" }
        return mid < 4.5 ? "right of" : "left of"
    }

    /// "You were moving toward 2 o'clock" (§3.3's walk cue), spoken only.
    static func motion(_ dial: WorldCaptureDial) -> String? {
        dial.motionSector.map { "You were moving toward \(hour(dial.position(of: $0))) o'clock" }
    }

    /// The centre: the count, or "–" before the first landing.
    static func count(_ dial: WorldCaptureDial) -> String {
        dial.hero.map { "\($0.supportedCount)" } ?? notYetNumber
    }

    /// The centre at accessibility sizes: "8/12".
    static func compactCount(_ dial: WorldCaptureDial) -> String {
        dial.hero.map { "\($0.supportedCount)/12" } ?? notYetNumber
    }

    static func accessibilityLabel(_ dial: WorldCaptureDial) -> String {
        guard let hero = dial.hero else { return dialName }
        let what = dial.top == .newestView ? "your newest views" : "most views"
        return dial.pieces.count > 1 ? "\(dialName), \(hero.name), \(what)" : "\(dialName), \(what)"
    }

    /// §4: "8 of 12 directions captured, 1 with one view. More needed at 5
    /// to 7 o'clock, behind your newest placed view. Top of the dial is your
    /// newest placed view." The age stays in the stamp line.
    static func accessibilityValue(_ dial: WorldCaptureDial, age: Double?) -> String {
        guard let hero = dial.hero else { return awaitingLine }
        var parts = ["\(hero.supportedCount) of 12 directions captured"
            + (hero.weakCount > 0 ? ", \(hero.weakCount) with one view." : "."), gapLine(dial, age: age)]
        if let motion = motion(dial) { parts.append(motion + ".") }
        parts.append("Top of the dial is \(reference(dial.top)).")
        return parts.joined(separator: " ")
    }

    /// A custom action and a photo's own label: "Photo at 3 o'clock".
    static func photo(at position: Int) -> String { "Photo at \(hour(position)) o'clock" }

    /// A mini dial: "Piece 2 · 3/12", spoken "Piece 2, 3 of 12 directions captured".
    static func miniCount(_ piece: WorldCaptureDial.Piece) -> String { "\(piece.supportedCount)/12" }

    static func miniLabel(_ piece: WorldCaptureDial.Piece) -> String {
        "\(piece.name) · \(miniCount(piece))"
    }

    static func miniAccessibilityLabel(_ piece: WorldCaptureDial.Piece) -> String {
        "\(piece.name), \(piece.supportedCount) of 12 directions captured"
    }

    static func more(_ count: Int) -> String { "+\(count) more" }
}
