//
//  WorldCaptureDialView.swift
//  Glasses
//
//  The FOW capture dial, drawn (FOW-RING-SPEC §4): twelve rounded annular
//  segments with constant 3 pt gaps on an always-dark stage, the piece's
//  photos upright in its filled segments, the count in the centre, the
//  look-here emphasis while the map is fresh, mini dials for the other
//  pieces, and the Dial|Map control. The footer under the stage follows the
//  system theme. Nothing here announces, plays a sound or buzzes.
//

import SwiftUI
import UIKit

// MARK: - Style

enum WorldDialStyle {
    /// The stage and the centre disc, `#0B0D10`.
    static let stage = WorldChromeStyle.rgb(0x0B0D10)
    static let unseen = WorldChromeStyle.rgb(0x1B2230)
    /// 3.3:1 on the stage; `unseenEdgeContrast` with Increase Contrast.
    static let unseenEdge = WorldChromeStyle.rgb(0x5A6678)
    static let unseenEdgeContrast = WorldChromeStyle.rgb(0x8A96A8)
    static let lit = WorldChromeStyle.rgb(0x6DF5DD)
    /// The current piece's amber: the look-here emphasis.
    static let amber = WorldChromeStyle.rgb(0xFFB347)
    static let discEdge = Color.white.opacity(20.0 / 255)

    static let supportedFill = lit.opacity(0.75)
    static let weakFill = lit.opacity(0.28)
    static let weakRim = lit.opacity(0.6)
}

/// The dial's measurements inside a stage of `size` (§4 Geometry).
nonisolated struct WorldDialGeometry: Equatable {
    static let innerRatio: CGFloat = 0.42
    static let gap: CGFloat = 3
    static let corner: CGFloat = 4
    /// Room around the ring for the marks drawn outside it.
    static let markRoom: CGFloat = 24

    let diameter: CGFloat

    /// `D = min(stageH − 40, stageW − 120, 300)`: about 231 pt on the
    /// walking card's 361 × 271 stage.
    init(stage size: CGSize) {
        diameter = max(96, min(size.height - 40, size.width - 120, 300))
    }

    init(diameter: CGFloat) { self.diameter = diameter }

    var radius: CGFloat { diameter / 2 }
    var inner: CGFloat { radius * Self.innerRatio }
    var mid: CGFloat { (radius + inner) / 2 }

    /// A sector's centre, in the dial's own square (y down, 0 at the top).
    func centre(of sector: Int) -> CGPoint {
        let angle = AnnularSector.angle(of: sector)
        return CGPoint(x: radius + mid * cos(angle), y: radius + mid * sin(angle))
    }

    /// The upright photo's square: §4's `1.15(R−r)`, grown just enough that
    /// an upright square covers its segment at any turn of the ring.
    var photoSide: CGFloat {
        let half = Double.pi / 12
        let outer = hypot(Double(radius) * cos(half) - Double(mid), Double(radius) * sin(half))
        let innerCorner = hypot(Double(mid) - Double(inner) * cos(half), Double(inner) * sin(half))
        return max(1.15 * (radius - inner), CGFloat(2 * max(outer, innerCorner)) + 1)
    }
}

// MARK: - The segment

/// `count` adjacent 30° directions of an annulus starting at `first` (sector
/// zero centred on 12 o'clock, clockwise), with a constant LINEAR gap at
/// each end -- the edges are parallel to the boundary rays, not radial --
/// and rounded corners. Insettable, so `strokeBorder` draws inner rims.
nonisolated struct AnnularSector: InsettableShape {
    var first: Int
    var count = 1
    var innerRatio: CGFloat = WorldDialGeometry.innerRatio
    var gap: CGFloat = WorldDialGeometry.gap
    var cornerRadius: CGFloat = WorldDialGeometry.corner
    var insetAmount: CGFloat = 0

    /// Screen radians of a sector's centre: 12 o'clock is −π/2.
    static func angle(of sector: Int) -> Double {
        -Double.pi / 2 + Double(sector) * Double.pi / 6
    }

    func inset(by amount: CGFloat) -> AnnularSector {
        var shape = self
        shape.insetAmount += amount
        return shape
    }

    func path(in rect: CGRect) -> Path {
        let full = Double(min(rect.width, rect.height)) / 2
        let outer = full - Double(insetAmount)
        let inner = full * Double(innerRatio) + Double(insetAmount)
        let origin = CGPoint(x: rect.midX, y: rect.midY)
        guard outer - inner > 0.5 else { return Path() }
        if count >= WorldCaptureDial.sectors {
            var ring = Path()
            ring.addEllipse(in: CGRect(x: origin.x - outer, y: origin.y - outer, width: 2 * outer, height: 2 * outer))
            ring.addEllipse(in: CGRect(x: origin.x - inner, y: origin.y - inner, width: 2 * inner, height: 2 * inner))
            return ring
        }
        let halfGap = Double(gap) / 2 + Double(insetAmount)
        let start = Self.angle(of: first) - Double.pi / 12
        let end = start + Double(count) * Double.pi / 6
        let innerRoom = inner * (end - start) - 2 * halfGap
        guard inner > halfGap, innerRoom > 0.5 else { return Path() }
        let corner = max(0, min(Double(cornerRadius - insetAmount), (outer - inner) / 2.2, innerRoom / 2.4))
        return Self.outline(origin: origin, outer: outer, inner: inner, start: start, end: end,
                            halfGap: halfGap, corner: corner)
    }

    private typealias V = (x: Double, y: Double)

    /// The rounded outline: outer arc clockwise, inner arc back, a fillet of
    /// `corner` at each of the four corners, each edge `halfGap` off its ray.
    private static func outline(origin: CGPoint, outer: Double, inner: Double, start: Double, end: Double,
                                halfGap: Double, corner: Double) -> Path {
        // The start edge runs along `u0`, offset into the shape along `n0`;
        // the end edge along `u1`, offset along `n1`.
        let u0: V = (cos(start), sin(start)), n0: V = (-sin(start), cos(start))
        let u1: V = (cos(end), sin(end)), n1: V = (sin(end), -cos(end))
        func point(_ v: V) -> CGPoint { CGPoint(x: origin.x + v.x, y: origin.y + v.y) }
        func angle(_ v: V) -> Double { atan2(v.y, v.x) }
        func minus(_ a: V, _ b: V) -> V { (a.x - b.x, a.y - b.y) }
        /// A fillet between an edge and a circle of radius `rho` (inside the
        /// circle for the outer arc, outside it for the inner): its centre and
        /// its tangent points on the edge and on the circle.
        func fillet(_ n: V, _ u: V, rho: Double, outside: Bool) -> (centre: V, edge: V, circle: V) {
            let h = halfGap + corner
            let d = outside ? rho + corner : rho - corner
            let t = max(0, d * d - h * h).squareRoot()
            let f: V = (n.x * h + u.x * t, n.y * h + u.y * t)
            let length = max(1e-9, (f.x * f.x + f.y * f.y).squareRoot())
            return (f, (f.x - n.x * corner, f.y - n.y * corner), (f.x / length * rho, f.y / length * rho))
        }
        func arc(_ path: inout Path, around c: V, radius: Double, from a: Double, by delta: Double) {
            let steps = max(2, Int((abs(delta) / (Double.pi / 90)).rounded(.up)))
            for step in 1...steps {
                let angle = a + delta * Double(step) / Double(steps)
                path.addLine(to: point((c.x + radius * cos(angle), c.y + radius * sin(angle))))
            }
        }
        /// Round the corner around `f` from `a` to `b`, the short way.
        func round(_ path: inout Path, _ f: V, from a: V, to b: V) {
            guard corner > 0.01 else {
                path.addLine(to: point(b))
                return
            }
            let from = angle(minus(a, f))
            arc(&path, around: f, radius: corner, from: from,
                by: remainder(angle(minus(b, f)) - from, 2 * Double.pi))
        }
        func clockwise(_ delta: Double) -> Double {
            let wrapped = delta.truncatingRemainder(dividingBy: 2 * Double.pi)
            return wrapped < 0 ? wrapped + 2 * Double.pi : wrapped
        }

        let a = fillet(n0, u0, rho: outer, outside: false)
        let b = fillet(n1, u1, rho: outer, outside: false)
        let c = fillet(n1, u1, rho: inner, outside: true)
        let d = fillet(n0, u0, rho: inner, outside: true)
        var path = Path()
        path.move(to: point(a.edge))
        round(&path, a.centre, from: a.edge, to: a.circle)
        arc(&path, around: (0, 0), radius: outer, from: angle(a.circle),
            by: clockwise(angle(b.circle) - angle(a.circle)))
        round(&path, b.centre, from: b.circle, to: b.edge)
        path.addLine(to: point(c.edge))
        round(&path, c.centre, from: c.edge, to: c.circle)
        arc(&path, around: (0, 0), radius: inner, from: angle(c.circle),
            by: -clockwise(angle(c.circle) - angle(d.circle)))
        round(&path, d.centre, from: d.circle, to: d.edge)
        path.closeSubpath()
        return path
    }
}

/// One sector of a dial of `diameter`, drawn in a smaller frame centred on
/// the sector's own centre: the photo's mask and its hit area.
nonisolated struct WorldDialCellShape: Shape {
    let sector: Int
    let diameter: CGFloat

    func path(in rect: CGRect) -> Path {
        let centre = WorldDialGeometry(diameter: diameter).centre(of: sector)
        return AnnularSector(first: sector).path(in: CGRect(x: rect.midX - centre.x, y: rect.midY - centre.y,
                                                            width: diameter, height: diameter))
    }
}

// MARK: - The stage: the dial or the map, and the control between them

/// The walking stage with the dial on (§4 Layout, §5): the dial (or, on the
/// Map tab, today's floor plan, unchanged), and the Dial|Map control
/// top-trailing once there is a map to switch to.
struct WorldCaptureDialStage<MapContent: View>: View {
    let dial: WorldCaptureDial
    /// The CURRENT receipt (never a held one), or `nil` before the first landing.
    let receipt: WorldCoverageReceipt?
    /// B2's overlay, already of `receipt`'s landing; `nil` otherwise.
    let overlay: WorldLandedOverlay?
    @Binding var tab: WorldPanelTab
    @ViewBuilder let map: () -> MapContent

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var showsToggle: Bool { receipt != nil }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            if showsToggle, tab == .map {
                map()
                    .transition(.opacity)
            } else {
                WorldCaptureDialView(dial: dial, receipt: receipt, overlay: overlay)
                    .transition(.opacity)
            }
            if showsToggle {
                Picker(WorldCaptureDialCopy.tabLabel, selection: $tab) {
                    ForEach(WorldPanelTab.allCases, id: \.self) { tab in
                        Text(WorldCaptureDialCopy.tabName(tab)).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .controlSize(.small)
                .fixedSize()
                .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
                .padding(.top, 10)
                .padding(.trailing, 12)
                .accessibilityIdentifier("wb-panel-tab")
            }
        }
        .animation(.easeInOut(duration: reduceMotion ? 0.15 : 0.25), value: tab)
    }
}

// MARK: - The dial

struct WorldCaptureDialView: View {
    let dial: WorldCaptureDial
    let receipt: WorldCoverageReceipt?
    let overlay: WorldLandedOverlay?

    var body: some View {
        // The phone's one-second tick, as the map's own label: the age moves
        // the gap line and the emphasis from fresh to stale. Never the Tower.
        TimelineView(.periodic(from: .now, by: 1)) { _ in
            WorldCaptureDialFace(dial: dial, receipt: receipt, overlay: overlay, age: receipt?.age(at: .now))
        }
    }
}

/// One frame of the dial at a known age.
struct WorldCaptureDialFace: View {
    let dial: WorldCaptureDial
    let receipt: WorldCoverageReceipt?
    let overlay: WorldLandedOverlay?
    let age: Double?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.colorSchemeContrast) private var contrast
    @Environment(\.accessibilityVoiceOverEnabled) private var voiceOver
    @Environment(\.dynamicTypeSize) private var typeSize

    /// The ring's accumulated turn in degrees, so each new anchor is reached
    /// by the shortest way round; `nil` until the first change.
    @State private var turn: Double?
    @State private var selected: WorldLandedThumbnail?

    private var fresh: Bool { WorldCaptureDialCopy.isFresh(age) }
    private var increased: Bool { contrast == .increased }

    /// What the ring is turned by now.
    private var ringTurn: Double {
        let resting = -Double(dial.anchor) * 30
        return reduceMotion ? resting : (turn ?? resting)
    }

    private var tiles: [String: WorldDialThumbnail] {
        Dictionary((overlay?.dialEntries ?? []).map { ($0.digest, $0) }, uniquingKeysWith: { _, last in last })
    }

    var body: some View {
        GeometryReader { proxy in
            let geometry = WorldDialGeometry(stage: proxy.size)
            ZStack {
                WorldDialStyle.stage
                dialFace(geometry)
                    .frame(width: geometry.diameter, height: geometry.diameter)
                    .position(x: proxy.size.width / 2, y: proxy.size.height / 2)
                minis
                    .padding(.leading, 12)
                    .padding(.bottom, 12)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
            }
        }
        .environment(\.colorScheme, .dark)
        .onChange(of: dial.anchor) { old, new in
            let current = turn ?? -Double(old) * 30
            var delta = (-Double(new) * 30 - current).truncatingRemainder(dividingBy: 360)
            if delta > 180 { delta -= 360 } else if delta <= -180 { delta += 360 }
            guard !reduceMotion else {
                turn = current + delta
                return
            }
            withAnimation(.spring(response: 0.6, dampingFraction: 0.85)) { turn = current + delta }
        }
        // An open sheet never outlives its photo: a new landing closes it.
        .onChange(of: dial.photosBySector) { _, photos in
            if let shown = selected, !photos.values.contains(shown.digest) { selected = nil }
        }
        .sheet(item: $selected) { thumbnail in
            WorldLandedThumbnailSheet(thumbnail: thumbnail, receipt: receipt)
        }
    }

    // MARK: The face

    private func dialFace(_ geometry: WorldDialGeometry) -> some View {
        ZStack {
            ring(geometry)
                // Reduce Motion: a new top is a 0.2 s cross-fade, not a turn.
                .id(reduceMotion ? dial.anchor : -1)
                .transition(.opacity)
            Circle()
                .fill(WorldDialStyle.stage)
                .overlay(Circle().strokeBorder(WorldDialStyle.discEdge, lineWidth: 1))
                .frame(width: geometry.inner * 2 - WorldDialGeometry.gap * 2,
                       height: geometry.inner * 2 - WorldDialGeometry.gap * 2)
                .allowsHitTesting(false)
                .accessibilityHidden(true)
            centre(geometry)
        }
        .frame(width: geometry.diameter, height: geometry.diameter)
        // Marks outside the ring are overlays: a ZStack would hand their
        // larger frame to every flexible segment.
        .overlay {
            WorldDialTopMarker(radius: geometry.radius)
                .frame(width: geometry.diameter + WorldDialGeometry.markRoom * 2,
                       height: geometry.diameter + WorldDialGeometry.markRoom * 2)
                .accessibilityHidden(true)
        }
        .animation(reduceMotion ? .easeInOut(duration: 0.2) : nil, value: dial.anchor)
        .background {
            // The dial's one element (§4): a leaf under the drawing, so a
            // photo's own tap is never taken by it.
            Circle()
                .fill(Color.white.opacity(0.001))
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(WorldCaptureDialCopy.accessibilityLabel(dial))
                .accessibilityValue(WorldCaptureDialCopy.accessibilityValue(dial, age: age))
                .accessibilityIdentifier("wb-panel-dial")
                .accessibilityActions {
                    ForEach(photoPositions, id: \.self) { position in
                        Button(WorldCaptureDialCopy.photo(at: position)) { open(sector: dial.sector(at: position)) }
                    }
                }
                .accessibilitySortPriority(2)
        }
    }

    /// Display positions holding a photo, clockwise from 12 o'clock.
    private var photoPositions: [Int] {
        (0..<WorldCaptureDial.sectors).filter { position in
            dial.photosBySector[dial.sector(at: position)].flatMap { tiles[$0] } != nil
        }
    }

    private func ring(_ geometry: WorldDialGeometry) -> some View {
        let emphasis = dial.emphasis(age: age)
        let marks = geometry.diameter + WorldDialGeometry.markRoom * 2
        return ZStack {
            ForEach(0..<WorldCaptureDial.sectors, id: \.self) { sector in
                segment(sector, geometry: geometry)
                    .frame(width: geometry.diameter, height: geometry.diameter)
            }
        }
        .frame(width: geometry.diameter, height: geometry.diameter)
        .overlay {
            if let emphasis {
                WorldDialLookHere(first: dial.sector(at: emphasis.positions[0]), count: emphasis.positions.count,
                                  radius: geometry.radius)
                    .frame(width: marks, height: marks)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
        }
        .overlay {
            if let motion = dial.motionSector {
                WorldDialMotionTick(sector: motion, radius: geometry.radius)
                    .frame(width: marks, height: marks)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
        }
        .rotationEffect(.degrees(ringTurn))
    }

    private func segment(_ sector: Int, geometry: WorldDialGeometry) -> some View {
        let state = dial.evidence(sector: sector)
        let position = dial.position(of: sector)
        let shape = AnnularSector(first: sector)
        let photo = dial.photosBySector[sector].flatMap { tiles[$0] }
        let fill = reduceMotion ? Animation.easeInOut(duration: 0.2)
            : .spring(response: 0.45, dampingFraction: 0.75).delay(0.03 * Double(position))
        return ZStack {
            shape.fill(base(state, photo: photo != nil, geometry: geometry))
                .animation(fill, value: state)
                .accessibilityHidden(true)
            if let photo {
                photoView(photo, state: state, sector: sector, position: position, geometry: geometry)
                    .transition(.opacity)
            }
            rim(shape, state: state, photo: photo != nil)
                .animation(fill, value: state)
                .allowsHitTesting(false)
                .accessibilityHidden(true)
        }
        .animation(.easeIn(duration: 0.25), value: photo?.digest)
    }

    private func photoView(_ photo: WorldDialThumbnail, state: WorldCoverageEvidence, sector: Int, position: Int,
                           geometry: WorldDialGeometry) -> some View {
        let side = geometry.photoSide
        let weak = state == .weak
        let cell = WorldDialCellShape(sector: sector, diameter: geometry.diameter)
        return Image(uiImage: photo.image)
            .resizable()
            .scaledToFill()
            .frame(width: side, height: side)
            .clipped()
            .saturation(weak ? 0.2 : (fresh ? 1 : 0.6))
            .overlay(Color.black.opacity(weak ? 0 : 0.15))
            .opacity(weak ? 0.5 : (fresh ? 1 : 0.8))
            // Upright whatever the ring's turn.
            .rotationEffect(.degrees(-ringTurn))
            .frame(width: side, height: side)
            .mask(cell)
            .contentShape(cell)
            .onTapGesture { open(sector: sector) }
            .position(geometry.centre(of: sector))
            .frame(width: geometry.diameter, height: geometry.diameter)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(WorldCaptureDialCopy.photo(at: position))
            .accessibilityAddTraits(.isButton)
            .accessibilityAction { open(sector: sector) }
            .accessibilityIdentifier("wb-panel-dial-photo")
            // VoiceOver reaches the photos through the dial's own actions:
            // one element (§4). Other tools (Voice Control, tests) see them.
            .accessibilityHidden(voiceOver)
    }

    /// A captured segment with no photo yet is lit teal, a little brighter
    /// towards the rim; under a photo the stage shows through a stale one.
    private func base(_ state: WorldCoverageEvidence, photo: Bool, geometry: WorldDialGeometry) -> AnyShapeStyle {
        switch state {
        case .supported:
            guard !photo else { return AnyShapeStyle(WorldDialStyle.stage) }
            return AnyShapeStyle(RadialGradient(colors: [WorldDialStyle.lit.opacity(0.6), WorldDialStyle.lit.opacity(0.86)],
                                                center: .center, startRadius: geometry.inner,
                                                endRadius: geometry.radius))
        case .weak: return AnyShapeStyle(WorldDialStyle.weakFill)
        case .unconfirmed: return AnyShapeStyle(WorldDialStyle.unseen)
        }
    }

    @ViewBuilder
    private func rim(_ shape: AnnularSector, state: WorldCoverageEvidence, photo: Bool) -> some View {
        switch state {
        case .supported:
            shape.strokeBorder(WorldDialStyle.lit, lineWidth: photo || increased ? 1.5 : 1)
        case .weak:
            shape.strokeBorder(WorldDialStyle.weakRim, style: increased
                               ? StrokeStyle(lineWidth: 1.5)
                               : StrokeStyle(lineWidth: 1, dash: [3.5, 2.5]))
        case .unconfirmed:
            shape.strokeBorder(increased ? WorldDialStyle.unseenEdgeContrast : WorldDialStyle.unseenEdge,
                               style: increased ? StrokeStyle(lineWidth: 1.5)
                                   : StrokeStyle(lineWidth: 1, dash: [3.5, 2.5]))
        }
    }

    @ViewBuilder
    private func centre(_ geometry: WorldDialGeometry) -> some View {
        let count = dial.hero?.supportedCount ?? 0
        Group {
            if typeSize >= .accessibility1 {
                Text(WorldCaptureDialCopy.compactCount(dial))
                    .font(.system(.title2, design: .rounded, weight: .semibold).monospacedDigit())
                    .contentTransition(.numericText(value: Double(count)))
            } else {
                VStack(spacing: 0) {
                    Text(WorldCaptureDialCopy.count(dial))
                        .font(.system(.largeTitle, design: .rounded, weight: .semibold).monospacedDigit())
                        .contentTransition(.numericText(value: Double(count)))
                    Text(WorldCaptureDialCopy.ofTwelve)
                        .font(.system(.caption, design: .rounded, weight: .semibold))
                        .foregroundStyle(Color.white.opacity(0.78))
                    Text(WorldCaptureDialCopy.captured)
                        .font(.system(.caption2, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.6))
                }
            }
        }
        .foregroundStyle(Color.white)
        .lineLimit(1)
        .minimumScaleFactor(0.6)
        .dynamicTypeSize(...DynamicTypeSize.accessibility1)
        .frame(width: geometry.inner * 1.6)
        .animation(reduceMotion ? nil : .spring(response: 0.45, dampingFraction: 0.8), value: count)
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    // MARK: The other pieces

    @ViewBuilder
    private var minis: some View {
        let others = dial.otherPieces
        if !others.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(others.prefix(3), id: \.referenceSegment) { piece in
                    WorldDialMini(piece: piece)
                }
                if others.count > 3 {
                    Text(WorldCaptureDialCopy.more(others.count - 3))
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(Color.white.opacity(0.8))
                        .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
                }
            }
            .allowsHitTesting(false)
        }
    }

    private func open(sector: Int) {
        guard let digest = dial.photosBySector[sector], let photo = tiles[digest] else { return }
        selected = WorldLandedThumbnail(digest: photo.digest, image: photo.image,
                                        referenceSegment: photo.referenceSegment, column: 0, row: 0,
                                        heading: photo.heading)
    }
}

// MARK: - The marks

/// 12 o'clock: a 2 pt white rim over the top segment with a notch pointing
/// in (§4). It never turns: the top is the dial's own reference.
struct WorldDialTopMarker: View {
    let radius: CGFloat

    var body: some View {
        Canvas { context, size in
            let centre = CGPoint(x: size.width / 2, y: size.height / 2)
            var arc = Path()
            arc.addArc(center: centre, radius: radius + 4, startAngle: .degrees(-90 - 11),
                       endAngle: .degrees(-90 + 11), clockwise: false)
            context.stroke(arc, with: .color(.white), style: StrokeStyle(lineWidth: 2, lineCap: .round))
            var notch = Path()
            notch.move(to: CGPoint(x: centre.x - 4.5, y: centre.y - radius - 11))
            notch.addLine(to: CGPoint(x: centre.x + 4.5, y: centre.y - radius - 11))
            notch.addLine(to: CGPoint(x: centre.x, y: centre.y - radius - 5.5))
            notch.closeSubpath()
            context.fill(notch, with: .color(.white))
        }
        .allowsHitTesting(false)
    }
}

/// `landed.motion_sector`: a short white arrow just outside the rim,
/// pointing outward, turning with the ring (§3.3).
struct WorldDialMotionTick: View {
    let sector: Int
    let radius: CGFloat

    var body: some View {
        Canvas { context, size in
            let centre = CGPoint(x: size.width / 2, y: size.height / 2)
            let angle = AnnularSector.angle(of: sector)
            let along = CGVector(dx: cos(angle), dy: sin(angle))
            let across = CGVector(dx: -along.dy, dy: along.dx)
            func at(_ r: CGFloat, _ side: CGFloat) -> CGPoint {
                CGPoint(x: centre.x + along.dx * r + across.dx * side, y: centre.y + along.dy * r + across.dy * side)
            }
            var arrow = Path()
            arrow.move(to: at(radius + 3, 0))
            arrow.addLine(to: at(radius + 13, 0))
            arrow.move(to: at(radius + 8.5, -4))
            arrow.addLine(to: at(radius + 13, 0))
            arrow.addLine(to: at(radius + 8.5, 4))
            context.stroke(arrow, with: .color(.white.opacity(0.7)),
                           style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
        }
    }
}

/// The look-here emphasis over the gap, fresh maps only (§4): a 2.5 pt
/// amber outline with an 8 pt glow and an outward chevron, pulsing
/// 0.35 ↔ 1.0 every 1.2 s; with Reduce Motion a static 3 pt outline.
struct WorldDialLookHere: View {
    let first: Int
    let count: Int
    let radius: CGFloat

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let outline = AnnularSector(first: first, count: count)
            .strokeBorder(WorldDialStyle.amber, lineWidth: reduceMotion ? 3 : 2.5)
            .frame(width: radius * 2, height: radius * 2)
            .shadow(color: WorldDialStyle.amber.opacity(0.75), radius: reduceMotion ? 3 : 8)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay {
                if count < WorldCaptureDial.sectors { chevron }
            }
        if reduceMotion {
            outline
        } else {
            // Only the opacity pulses: a repeating animation must never
            // catch a layout change.
            outline.phaseAnimator([1.0, 0.35]) { content, opacity in
                content.opacity(opacity)
            } animation: { _ in .easeInOut(duration: 0.6) }
        }
    }

    private var chevron: some View {
        Canvas { context, size in
            let centre = CGPoint(x: size.width / 2, y: size.height / 2)
            let angle = AnnularSector.angle(of: first) + Double(count - 1) * Double.pi / 12
            let along = CGVector(dx: cos(angle), dy: sin(angle))
            let across = CGVector(dx: -along.dy, dy: along.dx)
            func at(_ r: CGFloat, _ side: CGFloat) -> CGPoint {
                CGPoint(x: centre.x + along.dx * r + across.dx * side, y: centre.y + along.dy * r + across.dy * side)
            }
            var path = Path()
            path.move(to: at(radius + 12, -5.5))
            path.addLine(to: at(radius + 17, 0))
            path.addLine(to: at(radius + 12, 5.5))
            context.stroke(path, with: .color(WorldDialStyle.amber),
                           style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round))
        }
        .shadow(color: WorldDialStyle.amber.opacity(0.6), radius: 4)
    }
}

/// Another piece, colour only and not interactive (§3.7): 44 pt across,
/// 7 pt thick, its count inside and its name under it.
struct WorldDialMini: View {
    let piece: WorldCaptureDial.Piece

    var body: some View {
        VStack(alignment: .center, spacing: 2) {
            ZStack {
                ForEach(0..<WorldCaptureDial.sectors, id: \.self) { sector in
                    AnnularSector(first: sector, innerRatio: 15.0 / 22, gap: 1.5, cornerRadius: 1.5)
                        .fill(fill(piece.sectors[sector]))
                }
                Text(WorldCaptureDialCopy.miniCount(piece))
                    .font(.system(size: 9, weight: .semibold, design: .rounded).monospacedDigit())
                    .foregroundStyle(Color.white)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .frame(width: 26)
            }
            .frame(width: 44, height: 44)
            Text(piece.name)
                .font(.caption2.weight(.medium))
                .foregroundStyle(Color.white.opacity(0.8))
                .lineLimit(1)
                .fixedSize()
                .dynamicTypeSize(...DynamicTypeSize.xLarge)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(WorldCaptureDialCopy.miniAccessibilityLabel(piece))
        .accessibilityIdentifier("wb-panel-dial-mini")
    }

    private func fill(_ state: WorldCoverageEvidence) -> Color {
        switch state {
        case .supported: return WorldDialStyle.supportedFill
        case .weak: return WorldDialStyle.weakFill
        case .unconfirmed: return WorldDialStyle.unseen
        }
    }
}

// MARK: - The footer

/// Under the stage, in the system theme (§3.5, §3.6, §4): the piece, the
/// gap line, the orientation, the dated stamp, the receipt's own notes, and
/// the legend. The gap line is emphasised only while the map is fresh.
struct WorldCaptureDialCaption: View {
    let dial: WorldCaptureDial
    /// The map the dial draws, or `nil` before the first landing.
    let source: WorldPanelMapSource?
    var photosUnavailable = false

    @Environment(\.colorSchemeContrast) private var contrast
    @ScaledMetric(relativeTo: .footnote) private var dot: CGFloat = 7
    @ScaledMetric(relativeTo: .caption) private var swatch: CGFloat = 10

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { _ in
            content(age: source?.receipt.age(at: .now))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func content(age: Double?) -> some View {
        let fresh = WorldCaptureDialCopy.isFresh(age) && dial.gap != nil
        return VStack(alignment: .leading, spacing: 5) {
            if dial.hasReceipt {
                line(WorldCaptureDialCopy.heroLabel(dial), "wb-panel-dial-hero",
                     font: .subheadline.weight(.semibold), color: .primary)
            }
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                if fresh {
                    Circle()
                        .fill(WorldDialStyle.amber)
                        .frame(width: dot, height: dot)
                        .accessibilityHidden(true)
                }
                line(WorldCaptureDialCopy.gapLine(dial, age: age), "wb-panel-dial-gap",
                     font: fresh ? .footnote.weight(.semibold) : .footnote, color: fresh ? .primary : .secondary)
            }
            if dial.hasReceipt {
                line(WorldCaptureDialCopy.topLine(dial), "wb-panel-dial-top", font: .footnote, color: .secondary)
            }
            if let source {
                notes(source)
            }
            legend
                .padding(.top, 2)
        }
    }

    @ViewBuilder
    private func notes(_ source: WorldPanelMapSource) -> some View {
        let receipt = source.receipt
        line(WorldCoverageCopy.label(receipt, at: .now, stopped: false), "wb-panel-dial-stamp",
             font: .footnote.weight(.semibold).monospacedDigit(), color: .primary)
        if let newer = WorldCoverageCopy.newer(receipt, stopped: false) {
            line(newer, "wb-panel-dial-newer", font: .footnote, color: .secondary)
        }
        if let pieces = WorldCoverageCopy.pieces(receipt.coverage) {
            line(pieces, "wb-panel-dial-pieces", font: .footnote, color: .secondary)
        }
        if let notDrawn = WorldCoverageCopy.notDrawn(receipt.coverage) {
            line(notDrawn, "wb-panel-dial-not-drawn", font: .footnote, color: .secondary)
        }
        if photosUnavailable {
            line(WorldCaptureDialCopy.photosUnavailable, "wb-panel-dial-photos-unavailable",
                 font: .footnote, color: .secondary)
        }
        if source.isFixture {
            line(WorldCoverageCopy.fixture, "wb-panel-dial-fixture", font: .footnote, color: .secondary)
        }
    }

    private func line(_ text: String, _ identifier: String, font: Font, color: Color) -> some View {
        Text(text)
            .font(font)
            .foregroundStyle(color)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityIdentifier(identifier)
    }

    private var legend: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 14) { legendItems }
            VStack(alignment: .leading, spacing: 4) { legendItems }
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("wb-panel-dial-legend")
    }

    @ViewBuilder
    private var legendItems: some View {
        let increased = contrast == .increased
        legendItem(WorldCaptureDialCopy.legendCaptured, fill: WorldDialStyle.supportedFill,
                   edge: WorldDialStyle.lit, dashed: false)
        legendItem(WorldCaptureDialCopy.legendOneView, fill: WorldDialStyle.weakFill,
                   edge: WorldDialStyle.weakRim, dashed: !increased)
        legendItem(WorldCaptureDialCopy.legendNotYet, fill: WorldDialStyle.unseen,
                   edge: increased ? WorldDialStyle.unseenEdgeContrast : WorldDialStyle.unseenEdge,
                   dashed: !increased)
    }

    private func legendItem(_ word: String, fill: Color, edge: Color, dashed: Bool) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 5) {
            RoundedRectangle(cornerRadius: 3)
                .fill(WorldDialStyle.stage)
                .overlay(RoundedRectangle(cornerRadius: 3).fill(fill))
                .overlay(RoundedRectangle(cornerRadius: 3)
                    .strokeBorder(edge, style: StrokeStyle(lineWidth: 1, dash: dashed ? [2.5, 1.5] : [])))
                .frame(width: swatch * 1.4, height: swatch)
                .accessibilityHidden(true)
            Text(word)
                .font(.caption)
                .foregroundStyle(Color.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
