//
//  WorldCurrentPiece.swift
//  Glasses
//
//  Fog of war's "current piece" (FOW-CURRENT-PIECE-DATA-20261006): the
//  provisional splat that fills in, within seconds, as the wearer looks
//  around. It is the LAST segment's posed keyframes -- the one segment still
//  taking keyframes -- drawn from above in that segment's OWN frame, one
//  camera footprint (a position and a heading wedge) per keyframe.
//
//  What it may honestly show (§4), and so all it does:
//  - segment-local only: `transformToWorld` is never read, even for a
//    registered segment, and the piece is never drawn on, aligned with or
//    scaled against the room map -- it is its own card beside it;
//  - no metric scale: shape only (turn direction, rough path), fitted to
//    its box, with no distance anywhere;
//  - it restarts empty at every tracking break: a new last segment is a new
//    frame, and an older segment is never current again;
//  - it is the walk's own (`WalkScoped`) and live only while the Tower is
//    receiving that walk, so it goes at Stop;
//  - no segment data draws nothing, and the panel is as it was.
//
//  Already fetched (spec §3): the view model's geometry pull on every
//  `geometry.revision`. Nothing here fetches.
//

import SwiftUI

/// The current segment's camera footprints, segment-local and unitless.
nonisolated struct WorldCurrentPiece: Equatable, Sendable {
    /// One posed keyframe seen from above, in the segment's own frame: `x`
    /// is the camera x, `y` the camera z (OpenCV: x right, y down, z
    /// forward, right-handed -- so seen from above, x right and z up is not
    /// mirrored: a turn to the right draws to the right).
    struct Footprint: Equatable, Sendable {
        let x: Double
        let y: Double
        /// Where the camera looked, from above: radians clockwise from +y
        /// (the segment's first forward). `nil` without a rotation, or when
        /// it looked (nearly) straight up or down and so had no heading.
        let heading: Double?
    }

    let segmentIndex: Int
    /// Posed keyframes in order, split at every refused pose: a refused pose
    /// is a break, never a line drawn through the gap, never a zero.
    let runs: [[Footprint]]

    /// Posed keyframes in this piece.
    var viewCount: Int { runs.reduce(0) { $0 + $1.count } }

    /// The newest posed keyframe.
    var latest: Footprint? { runs.last?.last }

    /// The current piece of the geometry `segments` names, from the chunks
    /// in hand. Current is `segments.last` (append-only, ascending: spec §1).
    ///
    /// `nil` -- nothing drawn -- with no segment, no chunk in hand for the
    /// last one, or no posed keyframe in it. Never an older segment's chunk
    /// in its place: an older segment is frozen and never current again.
    static func current(segments: [WorldSegmentSummary],
                        chunks: [String: WorldSegmentChunk]) -> WorldCurrentPiece? {
        guard let last = segments.last, let chunk = chunks[last.cacheKey],
              chunk.segmentIndex == last.segmentIndex else { return nil }
        return WorldCurrentPiece(chunk: chunk)
    }

    /// The chunk's poses as the Tower sent them: `translation` is in the
    /// segment's own frame and is used untouched. `transformToWorld` is
    /// deliberately never read here (spec §4).
    init?(chunk: WorldSegmentChunk) {
        var runs: [[Footprint]] = []
        var run: [Footprint] = []
        for pose in chunk.poses {
            guard let t = pose.translation, t.count == 3, t.allSatisfy(\.isFinite) else {
                if !run.isEmpty { runs.append(run) }
                run = []
                continue
            }
            run.append(Footprint(x: t[0], y: t[2], heading: Self.heading(pose.rotation)))
        }
        if !run.isEmpty { runs.append(run) }
        guard !runs.isEmpty else { return nil }
        self.segmentIndex = chunk.segmentIndex
        self.runs = runs
    }

    /// The camera's forward (+z) seen from above, from `T_world_camera`'s
    /// rotation (wxyz): R's third column, read in x and z.
    static func heading(_ rotation: [Double]?) -> Double? {
        guard let q = rotation, q.count == 4, q.allSatisfy(\.isFinite) else { return nil }
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).squareRoot()
        guard norm > 1e-9 else { return nil }
        let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)
        let forwardX = 2 * (x * z + w * y)
        let forwardZ = 1 - 2 * (x * x + y * y)
        // Looking (nearly) straight up or down: no heading from above.
        guard forwardX * forwardX + forwardZ * forwardZ > 0.04 else { return nil }
        return atan2(forwardX, forwardZ)
    }

    /// What the panel draws: the piece of the walk the panel describes,
    /// while that walk is live -- the panel walking and the Tower receiving.
    /// Another walk's piece, a piece of no named walk, the finishing wait
    /// after Stop and every other phase draw nothing.
    ///
    /// `stoppedHere` is the walk whose capture this phone stopped: its piece
    /// goes at the local Stop, not when the Tower's (possibly delayed)
    /// finalizing report arrives (Codex review MED).
    ///
    /// Generic so that every value of the current segment -- the piece, and
    /// FOW v1.1 B1's live strip -- goes by this one rule.
    static func shown<Value>(_ piece: WalkScoped<Value>?, walk: WorldFinishWalk?,
                             stoppedHere: WorldFinishWalk? = nil,
                             phase: WorldPanelPhase, state: WorldModelState) -> Value? {
        guard case .walking = phase, case .receiving = state else { return nil }
        guard let piece, let owner = piece.walk, owner == walk, owner != stoppedHere else { return nil }
        return piece.value
    }
}

/// This phone's side of the piece's lifecycle (Codex review HOLD on ed3112b):
/// the piece is shown ONLY while this phone's own capture runs, and never for
/// a walk this phone is not capturing.
///
/// - After a local Stop nothing is shown, whatever the Tower reports, until
///   the next local Start: the Tower's reports lag, so a `.receiving` report
///   arriving after the Stop says nothing about this phone's capture.
/// - Every walk the Tower presents while this phone is NOT capturing is
///   refused -- the walk stopped here, and equally a walk a rapid Start ->
///   Stop ended before the Tower ever named it (its delayed `.receiving`
///   report lands after the Stop). It is not the next capture's walk, so the
///   refusal holds past the next Start until, during that capture, the Tower
///   presents a different walk.
///
/// `observe` is fed the screen's two inputs whenever either changes. `shown`
/// reads the live capture flag rather than a stored one, so the piece goes
/// in the very render that sees the Stop.
nonisolated struct WorldCurrentPieceGate: Equatable, Sendable {
    /// The walk whose piece is refused: one the Tower presented while this
    /// phone was not capturing.
    private(set) var refused: WorldFinishWalk?

    /// Whether this phone captures now, and the walk the Tower presents now.
    mutating func observe(isCapturing: Bool, presented: WorldFinishWalk?) {
        guard let presented else { return }
        if !isCapturing {
            refused = presented
        } else if presented != refused {
            refused = nil
        }
    }

    /// What the panel draws: nothing unless this phone is capturing; then
    /// the piece of the presented walk, if that walk is not refused and the
    /// received data is that walk's (`WorldCurrentPiece.shown`).
    func shown<Value>(_ piece: WalkScoped<Value>?, walk: WorldFinishWalk?, isCapturing: Bool,
                      phase: WorldPanelPhase, state: WorldModelState) -> Value? {
        guard isCapturing else { return nil }
        return WorldCurrentPiece.shown(piece, walk: walk, stoppedHere: refused, phase: phase, state: state)
    }

    /// The same, beside the walk `report` describes (FOW v1.1 B1's strip).
    func shown<Value, Report>(_ value: WalkScoped<Value>?, beside report: WalkScoped<Report>, isCapturing: Bool,
                              phase: WorldPanelPhase, state: WorldModelState) -> Value? {
        shown(value, walk: report.walk, isCapturing: isCapturing, phase: phase, state: state)
    }
}

extension WorldBuilderViewModel {
    /// The current piece of the geometry on screen, WITH the walk whose
    /// geometry it is: the panel shows it only beside that walk's report.
    ///
    /// Only while the published geometry is the revision the Tower last
    /// named. A newer revision still being fetched may have started a new
    /// segment (a tracking break), so the published last segment is not
    /// known to be current: nothing is shown until that revision lands
    /// (Codex review HIGH). The room map keeps the published revision.
    var currentPiece: WalkScoped<WorldCurrentPiece>? {
        guard let owner = geometryOwner, let walk = WorldFinishWalk(picture: owner),
              let published = publishedGeometryRevision, published == namedGeometryRevision,
              let piece = WorldCurrentPiece.current(segments: fragmentsModel.segments, chunks: geometryChunks)
        else { return nil }
        return WalkScoped(walk: walk, value: piece)
    }
}

enum WorldCurrentPieceCopy {
    static let label = "Provisional · current stretch"
    static let honesty = "Shape only, not to scale, not placed on the map. Starts over when tracking restarts."

    static func views(_ count: Int) -> String {
        count == 1 ? "1 view since tracking last restarted" : "\(count) views since tracking last restarted"
    }
}

// MARK: - The card

/// The current piece: its own card under the map, never inside it.
struct WorldCurrentPieceView: View {
    let piece: WorldCurrentPiece

    @Environment(\.dynamicTypeSize) private var typeSize

    static let tint = Color(red: 1.0, green: 0.70, blue: 0.28)

    var body: some View {
        let layout = typeSize.isAccessibilitySize
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: 10))
            : AnyLayout(HStackLayout(alignment: .top, spacing: 12))
        layout {
            WorldCurrentPieceCanvas(piece: piece)
                .frame(width: 112, height: 112)
                .background(Color(white: 0.10), in: RoundedRectangle(cornerRadius: 10))
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(WorldCurrentPieceCopy.label)
                .accessibilityValue(WorldCurrentPieceCopy.views(piece.viewCount))
                .accessibilityIdentifier("wb-current-piece-canvas")
            VStack(alignment: .leading, spacing: 4) {
                Text(WorldCurrentPieceCopy.label)
                    .font(.subheadline.weight(.semibold))
                    .accessibilityIdentifier("wb-current-piece-label")
                Text(WorldCurrentPieceCopy.views(piece.viewCount))
                    .font(.footnote)
                    .accessibilityIdentifier("wb-current-piece-views")
                Text(WorldCurrentPieceCopy.honesty)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 18))
        .overlay {
            // Dashed: provisional, and visibly not the map's card.
            RoundedRectangle(cornerRadius: 18)
                .strokeBorder(Self.tint.opacity(0.6), style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("wb-current-piece")
    }
}

/// The footprints, fitted to the box with one uniform factor: shape only.
/// +y (the segment's first forward) is up, x is right.
struct WorldCurrentPieceCanvas: View {
    let piece: WorldCurrentPiece

    var body: some View {
        Canvas { context, size in
            let wedge: CGFloat = 13
            let box = CGRect(origin: .zero, size: size).insetBy(dx: wedge + 3, dy: wedge + 3)
            let all = piece.runs.flatMap { $0 }
            guard !all.isEmpty, box.width > 0, box.height > 0 else { return }
            let xs = all.map(\.x), ys = all.map(\.y)
            let minX = xs.min()!, maxX = xs.max()!, minY = ys.min()!, maxY = ys.max()!
            let span = max(maxX - minX, maxY - minY)
            let factor = span > 1e-9 ? Double(min(box.width, box.height)) / span : 0
            let midX = (minX + maxX) / 2, midY = (minY + maxY) / 2
            func place(_ f: WorldCurrentPiece.Footprint) -> CGPoint {
                CGPoint(x: Double(box.midX) + (f.x - midX) * factor,
                        y: Double(box.midY) - (f.y - midY) * factor)
            }
            let tint = WorldCurrentPieceView.tint
            for run in piece.runs where run.count > 1 {
                var path = Path()
                path.addLines(run.map(place))
                context.stroke(path, with: .color(tint.opacity(0.45)), lineWidth: 1.5)
            }
            let latest = piece.latest
            for footprint in all {
                let p = place(footprint)
                let isLatest = footprint == latest
                if let heading = footprint.heading {
                    // Screen angle: clockwise from up is atan2's clockwise
                    // from +x minus a quarter turn.
                    let centre = Angle.radians(heading - .pi / 2)
                    var fan = Path()
                    fan.move(to: p)
                    fan.addArc(center: p, radius: isLatest ? wedge + 3 : wedge,
                               startAngle: centre - .degrees(24), endAngle: centre + .degrees(24), clockwise: false)
                    fan.closeSubpath()
                    context.fill(fan, with: .color(tint.opacity(isLatest ? 0.9 : 0.28)))
                }
                let r: CGFloat = isLatest ? 3.5 : 2
                context.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)),
                             with: .color(tint.opacity(isLatest ? 1 : 0.7)))
            }
        }
    }
}
