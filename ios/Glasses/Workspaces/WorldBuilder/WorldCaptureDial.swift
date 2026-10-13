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
//  It is never complete, and never says where the wearer is. Its top is the
//  newest placed view when the Tower says which that is (`landed.latest`,
//  S3) -- close enough to the way the wearer faces that the one-line hint
//  turns from it -- and otherwise the way the wearer faced when the piece
//  began (sector zero), when the hint names no direction. Never a compass.
//  No clock or view state lives here; the photos kept across landings are
//  `WorldCaptureDialPhotos`, held by the panel WITH their walk.
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
/// off. The flag is already the walk's own (the view model's gate).
nonisolated struct WorldPanelDial: Equatable, Sendable {
    /// `guidance.coverage: null` for the walk being received: the Tower's
    /// switch is on and no landing has merged yet (§6 "Before the first
    /// landing").
    var awaiting = false
}

/// The newest placed view of the last landing whose photos were in hand,
/// WITH its frame: held while a newer receipt's photos are not in yet, so
/// the hero, the top and the hint do not flip back for that window.
nonisolated struct WorldPanelHeldNewest: Equatable, Sendable {
    let newest: WorldCaptureDial.Newest
    let frameRevision: Int
}

/// The walking dial as the panel draws it: the current receipt's dial and
/// the photos it shows (this landing's, and those kept for its hero).
nonisolated struct WorldPanelDialShown: Equatable {
    let dial: WorldCaptureDial
    let photos: WorldCaptureDialPhotos
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
    /// The receipt's `frame_revision`: a re-solve may turn a piece's frame,
    /// and kept photos never cross one. `nil` before the first landing.
    let frameRevision: Int?
    /// Hero sector -> digest; the last eligible photo in manifest order wins,
    /// and a photo is never drawn on a segment that is not yet captured.
    let photosBySector: [Int: String]

    /// Before the first landing: twelve dashed segments and "–".
    static let awaiting = Self(components: [], stationsOmitted: 0)

    init(coverage: WorldCoverage, newest: Newest? = nil, motionSector: Int? = nil, photos: [Photo] = []) {
        self.init(components: coverage.components, stationsOmitted: coverage.stationsOmitted,
                  frameRevision: coverage.frameRevision, newest: newest, motionSector: motionSector, photos: photos)
    }

    init(components: [WorldCoverage.Component], stationsOmitted: Int, frameRevision: Int? = nil,
         newest: Newest? = nil, motionSector: Int? = nil, photos: [Photo] = []) {
        let pieces = components.enumerated().map { Piece(index: $0.offset, component: $0.element) }
        self.pieces = pieces
        self.stationsOmitted = stationsOmitted
        self.frameRevision = frameRevision
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

    /// The pieces other than the hero, in reference order.
    var otherPieces: [Piece] { pieces.filter { $0.index != heroIndex } }

    /// The pieces drawn as mini dials, live only: another piece with at
    /// least one captured direction. A "0/12" piece is never drawn.
    var miniPieces: [Piece] { otherPieces.filter { $0.supportedCount >= 1 } }

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

    /// The look-here emphasis: the single next-best segment -- the centre of
    /// the gap, or its middle two when the gap is six or more -- never the
    /// whole arc. Stale or undated maps keep it. `nil` with nothing to look
    /// for, or when every direction is the gap (no direction is better).
    var lookHere: Gap? {
        guard let gap, gap.positions.count < Self.sectors else { return nil }
        let count = gap.positions.count >= 6 ? 2 : 1
        let start = (gap.positions.count - count) / 2
        return Gap(positions: Array(gap.positions[start..<start + count]), evidence: gap.evidence)
    }
}

// MARK: - The photos, kept across landings

/// What the dial's segments show as photos (§3.4, kept): this landing's
/// in-hand tiles, and on each captured hero sector it left without one, the
/// photo last shown there by an earlier landing of the same hero piece in
/// the same frame. The panel holds the last value WITH its walk
/// (`WalkScoped`), so another walk or a reset starts with none, and a new
/// hero piece or a new `frame_revision` (a re-solve may turn the piece's
/// frame) drops them here.
nonisolated struct WorldCaptureDialPhotos: Equatable {
    /// The hero piece the photos are of; `nil` without one.
    let referenceSegment: Int?
    /// The frame the sectors are in.
    let frameRevision: Int?
    /// Hero sector -> the photo drawn there.
    let bySector: [Int: WorldDialThumbnail]

    static var empty: Self { Self(referenceSegment: nil, bySector: [:]) }

    init(referenceSegment: Int?, frameRevision: Int? = nil, bySector: [Int: WorldDialThumbnail]) {
        self.referenceSegment = referenceSegment
        self.frameRevision = frameRevision
        self.bySector = bySector
    }

    /// `dial`'s accepted photos from `entries`, over `kept` when `kept` is of
    /// the same hero piece in the same frame: a kept photo only fills a
    /// sector this landing left bare, and never one not captured now.
    init(dial: WorldCaptureDial, entries: [WorldDialThumbnail], keeping kept: Self? = nil) {
        guard let hero = dial.hero else {
            self = .empty
            return
        }
        let tiles = Dictionary(entries.map { ($0.digest, $0) }, uniquingKeysWith: { _, last in last })
        var shown = dial.photosBySector.compactMapValues { tiles[$0] }
        if let kept, kept.referenceSegment == hero.referenceSegment, kept.frameRevision == dial.frameRevision {
            for (sector, photo) in kept.bySector
            where shown[sector] == nil && (0..<WorldCaptureDial.sectors).contains(sector)
                && hero.sectors[sector] != .unconfirmed {
                shown[sector] = photo
            }
        }
        self.init(referenceSegment: hero.referenceSegment, frameRevision: dial.frameRevision, bySector: shown)
    }
}

// MARK: - The words

/// Everything the dial says (§3, §4, the demo-polish pass): one fixed title,
/// a one-line hint turned from the dial's top, and the update stamp. Never a
/// percentage or a clock face; nothing is voiced on its own.
nonisolated enum WorldCaptureDialCopy {
    static let dialName = "Capture dial"
    static let title = "Room coverage"
    static let tabLabel = "Panel view"
    static let awaitingLine = "Fills in as the Tower places your first views."
    static let ofTwelve = "of 12"
    static let captured = "captured"
    static let notYetNumber = "–"
    static let legendCaptured = "Captured"
    static let legendOneView = "One view"
    static let legendNotYet = "Not yet captured"
    static let allCaptured = "All directions captured — nice."
    static let lookAround = "Look around the room"
    static let ahead = "Keep looking ahead to capture more"
    static let right = "Turn right to capture more"
    static let rightBehind = "Look right and behind you"
    static let behind = "Look behind you"
    static let leftBehind = "Look left and behind you"
    static let left = "Turn left to capture more"
    static let updating = "Updating…"
    static let undated = "Update time unavailable"
    /// With the dial on, Capture health's one collapsed row.
    static let healthRow = "Capture health"

    /// Above this many seconds the stamp is just "Updating…". The
    /// dial's own threshold: the map's `WorldCoverageReceipt.staleAfter` is
    /// unchanged, so the switch OFF is today's panel exactly.
    static let staleAfter: Double = 90

    static func tabName(_ tab: WorldPanelTab) -> String {
        switch tab {
        case .dial: return "Dial"
        case .map: return "Map"
        }
    }

    /// Older than `staleAfter` on the measured Tower clock; an undated map
    /// is not said to be updating (its stamp says it has no time).
    static func isUpdating(_ age: Double?) -> Bool {
        age.map { $0 > staleAfter } ?? false
    }

    /// "12", "1" ... "11": the clock hour of a display position.
    static func hour(_ position: Int) -> Int { position == 0 ? 12 : position }

    /// The one-line hint. The top is the newest placed view, so the
    /// look-here segment's side of it is the way to turn; without `latest`
    /// the top is only where the piece began, and no direction is given.
    static func hint(_ dial: WorldCaptureDial) -> String {
        guard dial.hasReceipt else { return awaitingLine }
        guard dial.gap != nil else { return allCaptured }
        guard dial.top == .newestView, let target = dial.lookHere else { return lookAround }
        return turn(target)
    }

    /// From the look-here centre's angle clockwise from the top: ahead
    /// within 15°, behind within 15° of 180°, the diagonals behind out to
    /// 45° either side of it, and right or left between.
    static func turn(_ target: WorldCaptureDial.Gap) -> String {
        let first = Double(target.positions.first ?? 0)
        let centre = (first + Double(target.positions.count - 1) / 2).truncatingRemainder(dividingBy: 12)
        switch centre * 30 {
        case ...15, 345...: return ahead
        case ..<135: return right
        case ..<165: return rightBehind
        case ...195: return behind
        case ...225: return leftBehind
        default: return left
        }
    }

    /// The footer's second line, muted: "Updated 45 s ago", or just
    /// "Updating…" past `staleAfter` (no elapsed seconds to count up).
    static func stamp(age: Double?) -> String {
        guard let age else { return undated }
        return isUpdating(age) ? updating : "Updated \(WorldCoverageCopy.ago(age)) ago"
    }

    /// After Stop, the footer's second line: the panel's headline and the
    /// time since Stop, "Improving · 7 min since you stopped"; never empty,
    /// so the footer keeps its two lines.
    static func finishStamp(headline: String?, elapsed: String?) -> String {
        let parts = [headline, elapsed].compactMap { $0 }.filter { !$0.isEmpty }
        return parts.isEmpty ? " " : parts.joined(separator: " · ")
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
        guard let hero = dial.hero, dial.pieces.count > 1 else { return dialName }
        return "\(dialName), \(hero.name)"
    }

    /// "8 of 12 directions captured, 1 with one view. Look behind you." The
    /// hint, as the footer says it; the age stays in the stamp line. After
    /// Stop no hint is shown, so none is read: the count alone.
    static func accessibilityValue(_ dial: WorldCaptureDial, hint showsHint: Bool = true) -> String {
        guard let hero = dial.hero else { return awaitingLine }
        let count = "\(hero.supportedCount) of 12 directions captured"
            + (hero.weakCount > 0 ? ", \(hero.weakCount) with one view." : ".")
        guard showsHint else { return count }
        let hint = hint(dial)
        return count + " " + hint + (hint.hasSuffix(".") ? "" : ".")
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
