//
//  WorldCoverage.swift
//  Glasses
//
//  Fog of war v1 (FOW-COVERAGE-V1-SPEC-20261006, frozen): the Tower's
//  `guidance.coverage` status receipt -- dated, solve-placed station x
//  12-sector evidence -- read fail-closed (§5), and what the phone may say
//  about it (§7, §8).
//
//  What the receipt is, and is not. A supported sector means at least two
//  accepted keyframes, placed in this component by a landed global solve,
//  pointed that way from this station. It is never a reconstructed wall,
//  surface or texture. A grey sector is "not yet seen by a finished solve",
//  never a proved hole. Keyframes newer than the solve are counted, never
//  placed. Two components are two pieces, never one room. The receipt is
//  dated by `solved_at`, never presented as live, and carries no position
//  of the wearer.
//

import CoreGraphics
import Foundation

// MARK: - The receipt, decoded (§5)

/// What `guidance.coverage` said, read fail-closed.
nonisolated enum WorldCoverageStatus: Equatable, Sendable {
    /// No `guidance` object, or one without a `coverage` key: the Tower's
    /// switch is off (the live Tower today) or no session is selected. The
    /// panel is exactly today's.
    case absent
    /// `coverage: null`: no qualifying landed solve has been merged yet.
    case notYet
    /// A block was sent and a known field broke §5. Nothing of it is drawn.
    case refused
    case receipt(WorldCoverage)

    var coverage: WorldCoverage? {
        if case .receipt(let coverage) = self { return coverage }
        return nil
    }

    /// `payload.guidance.coverage` of a `world_builder.status/2026-09-10` payload.
    init(payload: [String: Any]) {
        guard let guidance = payload["guidance"] as? [String: Any], guidance.keys.contains("coverage") else {
            self = .absent
            return
        }
        let raw = guidance["coverage"]
        if raw is NSNull {
            self = .notYet
            return
        }
        self = WorldCoverage(json: raw).map(Self.receipt) ?? .refused
    }
}

/// One `guidance.coverage` block: a replacement, never a delta (§4).
nonisolated struct WorldCoverage: Equatable, Sendable {
    static let source = "landed_global_solve"
    static let stationGrid = "component_square_8x8_v1"
    static let sectorFrame = "first_qualified_forward_cw_from_up_v1"
    /// §5's display caps; over either, the block is refused as oversized.
    static let maxComponents = 16
    static let maxStations = 16
    static let sectors = 12
    static let gridSide = 8

    /// Unix seconds on the Tower's clock: the as-of stamp.
    let solvedAt: Double
    /// Unix seconds on the Tower's clock; diagnostic latency only.
    let computedAt: Double
    /// Accepted keyframes available to the named solve: "map through keyframe N".
    let horizonKeyframes: Int
    /// The accepted count when the block was computed: a snapshot, not live.
    let keyframesNow: Int
    /// `keyframesNow - horizonKeyframes`, floored at zero: a snapshot, not live.
    let keyframesPending: Int
    /// Provenance only; geometry is still fetched by `geometry.revision`.
    let geometryRevision: String
    let frameRevision: Int
    let componentsTotal: Int
    let componentsOmitted: Int
    let stationsOmitted: Int
    /// Sorted by `reference_segment`: each one a separate piece.
    let components: [Component]

    /// Components the Tower counted that are not drawn here: those its caps
    /// omitted and any row a decoder dropped. Never merged into a drawn one.
    var piecesNotDrawn: Int { max(0, componentsTotal - components.count) }

    struct Component: Equatable, Sendable {
        /// Identity within this solve and session only.
        let referenceSegment: Int
        let posedKeyframes: Int
        let cellSize: Double
        /// Sorted `(y, x)`; never empty.
        let stations: [Station]
    }

    struct Station: Equatable, Sendable {
        /// Column, along the component's sector-zero (forward) axis.
        let x: Int
        /// Row, along the component's right axis.
        let y: Int
        let keyframes: Int
        /// Bit i: exactly one supporting keyframe in sector i.
        let weakMask: Int
        /// Bit i: two or more distinct supporting keyframes in sector i.
        let supportedMask: Int

        func evidence(sector: Int) -> WorldCoverageEvidence {
            let bit = 1 << sector
            if supportedMask & bit != 0 { return .supported }
            if weakMask & bit != 0 { return .weak }
            return .unconfirmed
        }
    }

    /// `nil` for anything that breaks §5: a missing or mistyped known field,
    /// a value out of its bound, overlapping or over-wide masks, unsorted or
    /// repeated rows, inconsistent counts, or more than §5's 16 components
    /// or 16 stations (oversized). A row with `stations: []` is dropped, as
    /// §5 directs, and the rest kept. Unknown fields are ignored.
    init?(json: Any?) {
        typealias R = WorldCoverageReader
        guard let object = json as? [String: Any],
              R.integer(object["version"], 1...1) != nil,
              object["source"] as? String == Self.source,
              let solvedAt = R.number(object["solved_at"]), solvedAt >= 0,
              let computedAt = R.number(object["computed_at"]), computedAt >= solvedAt,
              let horizon = R.integer(object["horizon_keyframes"], 0...65535),
              let now = R.integer(object["keyframes_now"], 0...65535),
              let pending = R.integer(object["keyframes_pending"], 0...65535), pending == max(0, now - horizon),
              let revision = object["geometry_revision"] as? String, (1...128).contains(revision.utf8.count),
              revision.unicodeScalars.allSatisfy(\.isASCII),
              let frameRevision = R.integer(object["frame_revision"], 0...Int(Int32.max)),
              object["station_grid"] as? String == Self.stationGrid,
              object["sector_frame"] as? String == Self.sectorFrame,
              let total = R.integer(object["components_total"], 0...65535),
              let omitted = R.integer(object["components_omitted"], 0...65535),
              let stationsOmitted = R.integer(object["stations_omitted"], 0...65535),
              let rows = object["components"] as? [Any], rows.count <= Self.maxComponents,
              total >= rows.count, omitted == total - rows.count
        else { return nil }
        var components: [Component] = []
        var stations = 0
        var lastSegment = -1
        for row in rows {
            guard let parsed = R.component(row), parsed.referenceSegment > lastSegment else { return nil }
            lastSegment = parsed.referenceSegment
            stations += parsed.stations.count
            guard stations <= Self.maxStations else { return nil }
            // §5: "A decoder drops any row that arrives with stations:[]."
            if !parsed.stations.isEmpty { components.append(parsed) }
        }
        self.solvedAt = solvedAt
        self.computedAt = computedAt
        self.horizonKeyframes = horizon
        self.keyframesNow = now
        self.keyframesPending = pending
        self.geometryRevision = revision
        self.frameRevision = frameRevision
        self.componentsTotal = total
        self.componentsOmitted = omitted
        self.stationsOmitted = stationsOmitted
        self.components = components
    }
}

/// One heading sector's evidence at a station. Grey (`unconfirmed`) is "not
/// yet seen by a finished solve", never a proved hole; `weak` is one view and
/// is not covered; `supported` is two or more views, not a surface.
nonisolated enum WorldCoverageEvidence: Equatable, Sendable, CaseIterable {
    case unconfirmed, weak, supported
}

/// §5's field readers. JSON numbers only: a boolean (which
/// `JSONSerialization` bridges to `NSNumber`) is never a number here.
nonisolated enum WorldCoverageReader {
    static func number(_ value: Any?) -> Double? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
        let double = number.doubleValue
        return double.isFinite ? double : nil
    }

    static func integer(_ value: Any?, _ range: ClosedRange<Int>) -> Int? {
        guard let double = number(value), double.rounded() == double, abs(double) <= 9_007_199_254_740_992
        else { return nil }
        let integer = Int(double)
        return range.contains(integer) ? integer : nil
    }

    static func vector(_ value: Any?, count: Int) -> [Double]? {
        guard let array = value as? [Any], array.count == count else { return nil }
        let numbers = array.compactMap(number)
        return numbers.count == count ? numbers : nil
    }

    /// A component row; its stations may be empty (the caller drops it).
    static func component(_ value: Any) -> WorldCoverage.Component? {
        guard let object = value as? [String: Any],
              let segment = integer(object["reference_segment"], 0...Int(Int32.max)),
              let posed = integer(object["posed_keyframes"], 1...65535),
              let bounds = vector(object["bounds_xy"], count: 4), bounds[2] >= bounds[0], bounds[3] >= bounds[1],
              vector(object["origin_xyz"], count: 3) != nil,
              vector(object["up_xyz"], count: 3) != nil,
              vector(object["forward_xyz"], count: 3) != nil,
              let cellSize = number(object["cell_size"]), cellSize > 0,
              let rows = object["stations"] as? [Any], rows.count <= WorldCoverage.maxStations
        else { return nil }
        var stations: [WorldCoverage.Station] = []
        for row in rows {
            guard let station = station(row) else { return nil }
            if let last = stations.last, (last.y, last.x) >= (station.y, station.x) { return nil }
            stations.append(station)
        }
        return WorldCoverage.Component(referenceSegment: segment, posedKeyframes: posed, cellSize: cellSize,
                                       stations: stations)
    }

    static func station(_ value: Any) -> WorldCoverage.Station? {
        let side = 0...(WorldCoverage.gridSide - 1)
        let masks = 0...((1 << WorldCoverage.sectors) - 1)
        guard let object = value as? [String: Any],
              let x = integer(object["x"], side), let y = integer(object["y"], side),
              let keyframes = integer(object["keyframes"], 1...65535),
              let weak = integer(object["weak_mask"], masks), let supported = integer(object["supported_mask"], masks),
              weak & supported == 0
        else { return nil }
        return WorldCoverage.Station(x: x, y: y, keyframes: keyframes, weakMask: weak, supportedMask: supported)
    }
}

// MARK: - The receipt as the phone holds it (§7)

/// The Tower's clock, read at a status receipt: `tower_sent_at` and the
/// phone's instant then. The measured offset §7 asks for; without
/// `tower_sent_at` there is none, and the age is unavailable.
nonisolated struct WorldTowerClock: Equatable, Sendable {
    let towerSentAt: Double
    let receivedAt: ContinuousClock.Instant

    /// The Tower's clock at `instant`, estimated from the receipt. The
    /// transit time is not known and not added, so an age read from it is
    /// short by that transit, never long.
    func towerNow(at instant: ContinuousClock.Instant) -> Double {
        let elapsed = receivedAt.duration(to: instant).components
        return towerSentAt + Double(elapsed.seconds) + Double(elapsed.attoseconds) * 1e-18
    }
}

/// A coverage block as the walk report carries it: the block, the live
/// accepted count from the report that carried it, and the clock reading.
nonisolated struct WorldCoverageReceipt: Equatable, Sendable {
    let coverage: WorldCoverage
    /// `progress.keyframes_accepted` of the report carrying the block: live,
    /// unlike the block's own snapshot counts. `nil` when the Tower sent none.
    let keyframesAccepted: Int?
    /// `nil`: no measured Tower clock, so "age unavailable".
    let clock: WorldTowerClock?

    /// Above this age the map is `stale`; at exactly this age it is dated
    /// but not stale (§8.2).
    static let staleAfter: Double = 30

    /// The receipt after a report: none without a block; an unchanged block
    /// keeps its first clock reading (re-reading it every report would
    /// republish the walk for nothing, and the age is `solved_at`'s, so it
    /// is never re-dated either way).
    static func next(after previous: Self?, coverage: WorldCoverage?, keyframesAccepted: Int?,
                     clock: WorldTowerClock?) -> Self? {
        guard let coverage else { return nil }
        let kept = previous?.coverage == coverage ? previous?.clock : nil
        return Self(coverage: coverage, keyframesAccepted: keyframesAccepted, clock: kept ?? clock)
    }

    /// Accepted keyframes newer than this receipt, from live progress and
    /// floored at zero; `nil` when the Tower sent no live count.
    var newerKeyframes: Int? {
        keyframesAccepted.map { max(0, $0 - coverage.horizonKeyframes) }
    }

    /// Seconds since `solved_at` on the Tower's clock, at least zero, or
    /// `nil` when no clock was measured.
    ///
    /// Known LOW (FOW review): this age reads SHORT by the report's network
    /// transit -- `tower_sent_at` is taken as the Tower's clock at the
    /// phone's receipt, so the one-way delay (typically well under a second
    /// on the LAN, more over a slow link) is never counted. The map can
    /// therefore read a little fresher than it is, and `stale` arrives that
    /// much late; it never reads older than it is.
    func age(at instant: ContinuousClock.Instant) -> Double? {
        clock.map { max(0, $0.towerNow(at: instant) - coverage.solvedAt) }
    }

    /// `nil` when the age is unavailable.
    func isStale(at instant: ContinuousClock.Instant) -> Bool? {
        age(at: instant).map { $0 > Self.staleAfter }
    }

    /// After Stop, "at Stop" only when the solve included every accepted
    /// keyframe; an unknown live count is not "every".
    var includesEveryAcceptedKeyframe: Bool {
        keyframesAccepted.map { $0 <= coverage.horizonKeyframes } ?? false
    }
}

// MARK: - The words (§7, §8.2, §8.5)

/// Everything the map says. Never: grey = proved hole; color = surface or
/// texture; pending = placed; two pieces = one room; a minutes-old solve =
/// the current position; Stop = hole-free (§8.5).
nonisolated enum WorldCoverageCopy {
    static let fixture = "Fixture map"
    static let mapName = "Map"

    /// The map element's value: how many pieces, never "the room".
    static func mapValue(_ coverage: WorldCoverage) -> String {
        let count = coverage.components.count
        return count == 1 ? "1 piece" : "\(count) separate pieces"
    }
    static let unconfirmed = "Not yet seen by a finished solve"
    static let weak = "One view"
    static let supported = "Two or more views"

    /// The mandatory label: "Map through keyframe N · updated X s ago",
    /// `· stale` above 30 s, and with no measured clock the Tower's own
    /// `solved_at` time and "age unavailable" -- never a fresh badge. After
    /// Stop: "at Stop" only when the solve included every accepted keyframe.
    static func label(_ receipt: WorldCoverageReceipt, at instant: ContinuousClock.Instant, stopped: Bool,
                      timeZone: TimeZone = .current) -> String {
        let n = receipt.coverage.horizonKeyframes
        let through: String
        if stopped {
            through = receipt.includesEveryAcceptedKeyframe
                ? "Map at Stop, through keyframe \(n)" : "Map through keyframe \(n) before Stop"
        } else {
            through = "Map through keyframe \(n)"
        }
        guard let age = receipt.age(at: instant) else {
            return "\(through) · solved at \(towerTime(receipt.coverage.solvedAt, timeZone: timeZone)) Tower time"
                + " · age unavailable"
        }
        let stale = age > WorldCoverageReceipt.staleAfter ? " · stale" : ""
        return "\(through) · updated \(ago(age)) ago\(stale)"
    }

    /// "12 s", "119 s", "2 min": whole units, rounded down.
    static func ago(_ seconds: Double) -> String {
        let whole = Int(seconds.rounded(.down))
        return whole < 120 ? "\(whole) s" : "\(whole / 60) min"
    }

    static func towerTime(_ unixSeconds: Double, timeZone: TimeZone) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: Date(timeIntervalSince1970: unixSeconds))
    }

    /// Keyframes newer than the receipt: counted, never placed. From the
    /// live count when there is one; otherwise the block's own snapshot,
    /// said as a snapshot.
    static func newer(_ receipt: WorldCoverageReceipt, stopped: Bool) -> String? {
        guard let newer = receipt.newerKeyframes else {
            let pending = receipt.coverage.keyframesPending
            guard pending > 0 else { return nil }
            return "\(keyframes(pending)) not yet placed when this map was made"
        }
        guard newer > 0 else { return nil }
        if stopped { return "\(keyframes(newer)) newer than this receipt" }
        return newer == 1 ? "1 newer keyframe not yet placed" : "\(newer) newer keyframes not yet placed"
    }

    /// Two or more pieces: said to be separate, never one room.
    static func pieces(_ coverage: WorldCoverage) -> String? {
        let count = coverage.components.count
        return count > 1 ? "\(count) separate pieces, not placed relative to each other" : nil
    }

    /// What the caps left out, so a capped map never passes for all of it.
    static func notDrawn(_ coverage: WorldCoverage) -> String? {
        var parts: [String] = []
        let pieces = coverage.piecesNotDrawn
        if pieces > 0 { parts.append(pieces == 1 ? "1 more piece" : "\(pieces) more pieces") }
        let stations = coverage.stationsOmitted
        if stations > 0 { parts.append(stations == 1 ? "1 more station" : "\(stations) more stations") }
        return parts.isEmpty ? nil : parts.joined(separator: " and ") + " not drawn"
    }

    /// "Piece 1", "Piece 2": in `reference_segment` order, never by size,
    /// and never "the room".
    static func pieceName(_ index: Int) -> String { "Piece \(index + 1)" }

    /// A piece's spoken value: its stations and how many headings carry
    /// each kind of evidence.
    static func pieceValue(_ component: WorldCoverage.Component) -> String {
        var counts: [WorldCoverageEvidence: Int] = [:]
        for station in component.stations {
            for sector in 0..<WorldCoverage.sectors { counts[station.evidence(sector: sector), default: 0] += 1 }
        }
        let stations = component.stations.count == 1 ? "1 station" : "\(component.stations.count) stations"
        return "\(stations). \(supported): \(counts[.supported, default: 0]) headings. "
            + "\(weak): \(counts[.weak, default: 0]). \(unconfirmed): \(counts[.unconfirmed, default: 0])."
    }

    private static func keyframes(_ count: Int) -> String {
        count == 1 ? "1 keyframe" : "\(count) keyframes"
    }
}

// MARK: - The drawing's geometry (§6)

/// Where a piece draws its fans. Column x (the component's forward axis)
/// runs to the right and row y (its right axis) runs down, so sector zero
/// points right and sectors run clockwise on screen -- clockwise as seen
/// from above, as §6.4 defines them. Only station fans are drawn: no
/// position, path or wearer, and no north arrow or ruler (§6, §8.5).
nonisolated enum WorldCoverageGeometry {
    struct Fan: Equatable {
        let centre: CGPoint
        let radius: CGFloat
        let sectors: [WorldCoverageEvidence]
    }

    /// Sector `i`'s span in screen degrees, clockwise from screen-right.
    static func span(sector: Int) -> (start: Double, end: Double) {
        (Double(sector) * 30, Double(sector + 1) * 30)
    }

    /// The unit direction of sector `i`'s middle, on a y-down screen.
    static func direction(sector: Int) -> CGVector {
        let radians = (Double(sector) * 30 + 15) * .pi / 180
        return CGVector(dx: cos(radians), dy: sin(radians))
    }

    /// One fan per station in a square piece of side `side`, and nothing else.
    static func fans(_ component: WorldCoverage.Component, side: CGFloat) -> [Fan] {
        let cell = side / CGFloat(WorldCoverage.gridSide)
        return component.stations.map { station in
            Fan(centre: CGPoint(x: (CGFloat(station.x) + 0.5) * cell, y: (CGFloat(station.y) + 0.5) * cell),
                radius: cell * 0.48,
                sectors: (0..<WorldCoverage.sectors).map { station.evidence(sector: $0) })
        }
    }

    /// Square pieces for `count` components inside `size`, in a grid that
    /// gives each the largest side, left to right then top to bottom.
    static func pieceFrames(count: Int, in size: CGSize, inset: CGFloat = 12, gap: CGFloat = 12) -> [CGRect] {
        guard count > 0 else { return [] }
        let width = max(0, size.width - 2 * inset), height = max(0, size.height - 2 * inset)
        var best: (columns: Int, side: CGFloat) = (1, 0)
        for columns in 1...count {
            let rows = (count + columns - 1) / columns
            let side = min((width - gap * CGFloat(columns - 1)) / CGFloat(columns),
                           (height - gap * CGFloat(rows - 1)) / CGFloat(rows))
            if side > best.side { best = (columns, side) }
        }
        let side = max(0, best.side)
        let columns = best.columns
        let rows = (count + columns - 1) / columns
        let usedWidth = side * CGFloat(columns) + gap * CGFloat(columns - 1)
        let usedHeight = side * CGFloat(rows) + gap * CGFloat(rows - 1)
        let originX = inset + (width - usedWidth) / 2, originY = inset + (height - usedHeight) / 2
        return (0..<count).map { index in
            CGRect(x: originX + CGFloat(index % columns) * (side + gap),
                   y: originY + CGFloat(index / columns) * (side + gap), width: side, height: side)
        }
    }
}

// MARK: - The DEBUG fixture

extension WorldCoverageReceipt {
    /// §9's mid-walk fixture -- two separate pieces, 14 keyframes newer --
    /// with no measured clock: what `-WBPanelMapFixture` draws.
    static let fixture: WorldCoverageReceipt? = {
        let json = #"""
        {"version":1,"source":"landed_global_solve","solved_at":1791240000.0,"computed_at":1791240000.12,"horizon_keyframes":113,"keyframes_now":127,"keyframes_pending":14,"geometry_revision":"g-mid-113","frame_revision":1,"station_grid":"component_square_8x8_v1","sector_frame":"first_qualified_forward_cw_from_up_v1","components_total":2,"components_omitted":0,"stations_omitted":0,"components":[{"reference_segment":0,"posed_keyframes":82,"bounds_xy":[-4.0,-4.0,4.0,4.0],"origin_xyz":[0.0,0.0,0.0],"up_xyz":[0.0,0.0,1.0],"forward_xyz":[0.0,1.0,0.0],"cell_size":1.0,"stations":[{"x":3,"y":5,"keyframes":30,"weak_mask":1,"supported_mask":30}]},{"reference_segment":17,"posed_keyframes":19,"bounds_xy":[-2.0,-2.0,2.0,2.0],"origin_xyz":[10.0,0.0,0.0],"up_xyz":[0.0,0.0,1.0],"forward_xyz":[1.0,0.0,0.0],"cell_size":0.5,"stations":[{"x":1,"y":2,"keyframes":8,"weak_mask":2,"supported_mask":1}]}]}
        """#
        guard let object = try? JSONSerialization.jsonObject(with: Data(json.utf8)),
              let coverage = WorldCoverage(json: object)
        else { return nil }
        return WorldCoverageReceipt(coverage: coverage, keyframesAccepted: 127, clock: nil)
    }()
}
