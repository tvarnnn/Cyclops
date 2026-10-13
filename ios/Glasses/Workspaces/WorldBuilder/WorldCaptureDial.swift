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
//  It is never complete, and never says where the wearer is. Its top is
//  fixed (round 3, the owner's call): the way the wearer faced when the
//  piece began (sector zero), labelled "start" -- never the newest placed
//  view, which lags the head by a minute or two, so the ring never turns
//  between landings and "missing at 8 o'clock" stays at 8 o'clock. It says
//  no sentence of guidance: the gaps are drawn, the amber outline marks the
//  largest, and the footer is one quiet count line. Never a compass. No
//  clock or view state lives here; the photos kept across landings are
//  `WorldCaptureDialPhotos`, and the recent strip `WorldDialRecent`, both
//  held by the panel WITH their walk.
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

/// The Dial|Map control's switch, `IOS_FOW_MAP_TAB` (environment) or the
/// `GlassesFOWMapTab` Info.plist key filled from the same build setting.
/// `on` exactly shows the control; anything else, unset included, hides it
/// and the dial is the walking stage's only face. Read only with the dial on.
nonisolated struct WorldDialMapTabSwitch: Equatable, Sendable {
    static let variable = "IOS_FOW_MAP_TAB"
    static let infoKey = "GlassesFOWMapTab"

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
    /// `IOS_FOW_MAP_TAB=on`: the Dial|Map control is shown. Off (the
    /// default), the dial is the only face.
    var mapTab = false
}

/// The newest placed view of the last landing whose photos were in hand,
/// WITH its frame: held while a newer receipt's photos are not in yet, so
/// the hero piece does not flip back for that window.
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
    /// It picks the hero piece; it never turns the ring.
    struct Newest: Equatable, Sendable {
        let referenceSegment: Int
        let sector: Int
    }

    let pieces: [Piece]
    /// `nil` before the first landing (or for a receipt with no drawn piece).
    let heroIndex: Int?
    /// The sector drawn at 12 o'clock: always sector zero, the way the
    /// wearer faced when the piece began ("start"). Fixed, so the ring
    /// never turns between landings.
    var anchor: Int { 0 }
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
        // in a piece not drawn here names nothing. Either way start is on top.
        if let newest, (0..<Self.sectors).contains(newest.sector),
           let named = pieces.firstIndex(where: { $0.referenceSegment == newest.referenceSegment }) {
            heroIndex = named
            self.motionSector = motionSector.flatMap { (0..<Self.sectors).contains($0) ? $0 : nil }
        } else {
            heroIndex = pieces.indices.max { pieces[$0].posedKeyframes < pieces[$1].posedKeyframes }
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

    /// The amber "scan here" outline, purely visual: the single segment at
    /// the centre of the largest gap, or its middle two when the gap is six
    /// or more -- never the whole arc. Stale or undated maps keep it. `nil`
    /// with nothing to look for, or when every direction is the gap.
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

// MARK: - The recent strip

/// "Recently captured" (round 3): the walk's landed photos as they came in,
/// newest first, at most `limit` -- the tiles already in hand, nothing
/// fetched for it. A landing's photos not met before in this walk lead, the
/// newest placed view's own photo first among them; a photo met before is
/// never new again, however often a later landing re-sends it. The panel
/// holds it WITH its walk, so another walk or a reset starts empty.
nonisolated struct WorldDialRecent: Equatable {
    static let limit = 4

    /// Every digest met this walk.
    private(set) var seen: Set<String> = []
    /// Newest first, at most `limit`.
    private(set) var photos: [WorldDialThumbnail] = []

    func adding(_ entries: [WorldDialThumbnail], newest: WorldImageryLanded.Latest? = nil) -> Self {
        var fresh: [WorldDialThumbnail] = []
        for entry in entries where !seen.contains(entry.digest) && !fresh.contains(where: { $0.digest == entry.digest }) {
            fresh.append(entry)
        }
        guard !fresh.isEmpty else { return self }
        if let newest, let lead = fresh.firstIndex(where: { entry in
            entry.referenceSegment == newest.componentReferenceSegment
                && WorldCaptureDial.photos([entry]).first?.sector == newest.sector
        }) {
            fresh.insert(fresh.remove(at: lead), at: 0)
        }
        var next = self
        next.seen.formUnion(fresh.map(\.digest))
        next.photos = Array((fresh + photos).prefix(Self.limit))
        return next
    }
}

// MARK: - The words

/// Everything the dial says (round 3, the owner's call): no sentence of
/// guidance at all. One fixed title, the centre's count, one quiet footer
/// line -- the count and the update stamp -- and, for VoiceOver, the largest
/// gap as clock positions from where the piece started (the top is fixed
/// and labelled "start", so a clock position means the same every landing).
/// Never a percentage; nothing is voiced on its own.
nonisolated enum WorldCaptureDialCopy {
    static let dialName = "Capture dial"
    static let title = "Room coverage"
    static let tabLabel = "Panel view"
    /// The footer, and the value, before the first landing.
    static let awaitingLine = "Waiting for first views"
    static let ofTwelve = "of 12"
    static let captured = "captured"
    static let notYetNumber = "–"
    static let legendCaptured = "Captured"
    static let legendOneView = "One view"
    static let legendNotYet = "Not yet captured"
    /// The tiny caption by the 12 o'clock marker.
    static let start = "start"
    static let catchingUp = "catching up"
    static let recentTitle = "Recently captured"
    static let closePhoto = "Close"
    /// With the dial on, Capture health's one collapsed row.
    static let healthRow = "Capture health"

    /// Above this many seconds the footer says "catching up" instead of the
    /// age. The dial's own threshold: the map's `WorldCoverageReceipt
    /// .staleAfter` is unchanged, so the switch OFF is today's panel exactly.
    static let staleAfter: Double = 90

    static func tabName(_ tab: WorldPanelTab) -> String {
        switch tab {
        case .dial: return "Dial"
        case .map: return "Map"
        }
    }

    /// Older than `staleAfter` on the measured Tower clock; an undated map
    /// is not said to be catching up (it has no time).
    static func isCatchingUp(_ age: Double?) -> Bool {
        age.map { $0 > staleAfter } ?? false
    }

    /// "12", "1" ... "11": the clock hour of a display position.
    static func hour(_ position: Int) -> Int { position == 0 ? 12 : position }

    /// The footer's one quiet line: "8 of 12 directions · updated 45 s ago",
    /// "8 of 12 directions · catching up" past `staleAfter`, the count alone
    /// for an undated map, and "Waiting for first views" before the first
    /// landing.
    static func footer(_ dial: WorldCaptureDial, age: Double?) -> String {
        guard let hero = dial.hero else { return awaitingLine }
        let count = "\(hero.supportedCount) of 12 directions"
        guard let age else { return count }
        return count + " · " + (isCatchingUp(age) ? catchingUp : "updated \(WorldCoverageCopy.ago(age)) ago")
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

    /// "8 of 12 directions captured; missing around 7 to 9 o'clock from where
    /// you started": the count, and the largest gap the ring draws (its
    /// first and last clock positions). A gap of one-view segments only is
    /// "one view only around ..."; with all twelve captured, or every
    /// direction the gap, the count alone.
    static func accessibilityValue(_ dial: WorldCaptureDial) -> String {
        guard let hero = dial.hero else { return awaitingLine }
        let count = "\(hero.supportedCount) of 12 directions captured"
        guard let gap = dial.gap, gap.positions.count < WorldCaptureDial.sectors,
              let first = gap.positions.first, let last = gap.positions.last else { return count }
        let around = first == last ? "\(hour(first)) o'clock" : "\(hour(first)) to \(hour(last)) o'clock"
        let what = gap.evidence == .weak ? "one view only" : "missing"
        return "\(count); \(what) around \(around) from where you started"
    }

    /// A custom action and a photo's own label: "Photo at 3 o'clock".
    static func photo(at position: Int) -> String { "Photo at \(hour(position)) o'clock" }

    /// A recent-strip photo's label: "Photo 1 of 4", newest first.
    static func recentPhoto(_ index: Int, of count: Int) -> String { "Photo \(index) of \(count)" }

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
